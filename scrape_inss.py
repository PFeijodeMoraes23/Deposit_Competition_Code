## scrape_inss.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-03-06
#
# Purpose: Download INSS "Benefícios Mantidos por Município" data and build
#          an MCA × quarter panel of retirement/pension beneficiary counts.
#
#   Data source:
#     Portal Dados Abertos — "Benefícios Mantidos por Município"
#     Dataset URL: https://dados.gov.br/dados/conjuntos-dados/
#                    beneficios-mantidos-por-municipio
#
#     The dataset is published as monthly ZIP archives, each containing a CSV
#     with active (maintained) benefit records per municipality × benefit type.
#     Files follow the naming pattern:
#       beneficios-mantidos-<YYYY><MM>.zip
#     hosted at:
#       https://dadosabertos.mte.gov.br/...   (MDIC) — OR —
#       https://www.previdencia.gov.br/dados-abertos/...
#     The exact URL base changes with government restructuring.
#
#     NOTE: The most stable access path (as of 2025) is via the INSS
#     open-data endpoint catalogued on dados.gov.br. The INSS_BASE_URL and
#     filename pattern below reflect the latest known structure.  If downloads
#     fail, check:
#       https://dados.gov.br/dados/conjuntos-dados/beneficios-mantidos-por-municipio
#     for current resource URLs.
#
#   Key variables used:
#     municipio    : 6-digit IBGE code (string like "330455")
#     competencia  : YYYY-MM period (e.g. "2023-06")
#     especie      : benefit species code (see SPECIES_RETIREMENT below)
#     quantidade   : number of active benefits
#
#   Target benefit species (retirement + pensions):
#     41  Aposentadoria por Invalidez Previdenciária
#     42  Aposentadoria por Idade Previdenciária
#     43  Aposentadoria por Tempo de Contribuição Previdenciária
#     46  Aposentadoria Especial Previdenciária
#     21  Pensão por Morte Previdenciária
#     (32 = Auxílio-Doença excluded — not a long-term dependency benefit)
#
#   Output:
#     INSS/inss_mca_panel.csv
#       Columns: mca_code, year, quarter,
#                retirees_total,           (sum of retirement + pension benefits)
#                retirees_per1000,         (retirees_total / pop * 1000)
#                n_municipalities          (municipalities with INSS data)
#
# Notes:
#   • Some files use 6-digit municipality codes, some use 7-digit including the
#     check digit.  Both forms are handled: 7-digit codes are truncated to 6
#     and the crosswalk municipality_code is also truncated for the merge.
#   • Very small municipalities may have suppressed data ("<5" or "0").
###─────────────────────────────────────────────────────────────────────────────

import concurrent.futures
import io
import os
import re as _re
import threading
import time
import zipfile
import logging
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import requests
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

## ─────────────────────────────────────────────────────────────────────────────
## 1) PATHS & CONSTANTS
## ─────────────────────────────────────────────────────────────────────────────
BASE      = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
INSS_DIR  = os.path.join(BASE, "INSS")
# Raw ZIPs/CSVs are large transient files.  If this project lives inside
# OneDrive, set INSS_RAW_DIR to a path outside OneDrive to avoid the
# file-locking errors ([WinError 32]) that OneDrive causes mid-sync.
# Example (PowerShell):  $env:INSS_RAW_DIR = "C:\Temp\inss_raw"
RAW_DIR   = os.environ.get("INSS_RAW_DIR", os.path.join(INSS_DIR, "raw"))
IBGE_DIR  = os.path.join(BASE, "IBGE")
MCA_CSV   = os.path.join(IBGE_DIR, "muni_mca_regions_2010_2024_panel.csv")
DEMO_CSV  = os.path.join(IBGE_DIR, "mca_demographics_panel.csv")
OUTPUT_CSV = os.path.join(INSS_DIR, "inss_mca_panel.csv")

os.makedirs(RAW_DIR,  exist_ok=True)
os.makedirs(INSS_DIR, exist_ok=True)

PANEL_START_YEAR = 2013
PANEL_END_YEAR   = 2024

# Benefit species considered "retirement/pension" for the retirees_total count
SPECIES_RETIREMENT = {41, 42, 43, 46, 21}

# ── INSS CKAN open-data API ─────────────────────────────────────────────────
# Catalog endpoint: https://dadosabertos.inss.gov.br/api/3/action/
# Two packages cover our period (2021-09 onward; earlier months unavailable):
INSS_CKAN_BASE = "https://dadosabertos.inss.gov.br/api/3/action"
INSS_PKG_OLD   = "inss-beneficios-mantidos"             # ~2021-09 to 2023-05
INSS_PKG_NEW   = (                                       # 2023-06 onward
    "beneficios-mantidos-plano-de-dados-abertos-jun-2023-a-jun-2025"
)
HEADERS = {"User-Agent": "research/pedro.feijodemoraes@yale.edu"}

# Portuguese month names used in CKAN resource titles
_MONTHS_PT: dict[str, int] = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "março": 3,
    "abril": 4,   "maio": 5,      "junho": 6, "julho": 7,
    "agosto": 8,  "setembro": 9,  "outubro": 10,
    "novembro": 11, "dezembro": 12,
}

# Catalog cache (built once, shared across threads)
_CATALOG_LOCK:  threading.Lock          = threading.Lock()
_CATALOG_CACHE: dict[tuple, str] | None = None


## ─────────────────────────────────────────────────────────────────────────────
## 2) DOWNLOAD ONE MONTHLY FILE
## ─────────────────────────────────────────────────────────────────────────────

def _get_inss_catalog() -> dict[tuple[int, int], str]:
    """Query both INSS CKAN packages and build (year, month) → download-URL map."""
    global _CATALOG_CACHE
    with _CATALOG_LOCK:
        if _CATALOG_CACHE is not None:
            return _CATALOG_CACHE
        catalog: dict[tuple[int, int], str] = {}
        for pkg_id in (INSS_PKG_OLD, INSS_PKG_NEW):
            resources: list = []
            for attempt in range(3):
                try:
                    resp = requests.get(
                        f"{INSS_CKAN_BASE}/package_show",
                        params={"id": pkg_id},
                        headers=HEADERS,
                        timeout=60,
                    )
                    resp.raise_for_status()
                    resources = resp.json().get("result", {}).get("resources", [])
                    break
                except Exception as exc:
                    if attempt == 2:
                        logging.warning(f"CKAN {pkg_id}: {exc}")
                    time.sleep(2 ** attempt * 5.0)
            for res in resources:
                title = (res.get("name") or res.get("description") or "").lower()
                url   = res.get("url", "")
                if not url or "ativos" not in title:
                    continue
                month_found = next(
                    (m for pt, m in _MONTHS_PT.items() if pt in title), None
                )
                m_yr = _re.search(r"\b(20\d{2})\b", title)
                if month_found and m_yr:
                    catalog[(int(m_yr.group(1)), month_found)] = url
        logging.info(f"INSS catalog: {len(catalog)} periods indexed")
        _CATALOG_CACHE = catalog
        return catalog


def _safe_remove(path: str) -> None:
    """Remove a file, silently ignoring OS-level errors (e.g. file locked)."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def download_inss_month(year: int, month: int,
                        force: bool = False) -> str | None:
    """
    Download and extract the INSS monthly benefits ZIP for year/month.
    Discovers the download URL via the INSS CKAN catalog.
    Returns the local CSV path, or None if unavailable (pre-2021-09 months
    are not published in the open-data portal).
    """
    dest_csv = os.path.join(RAW_DIR, f"beneficios-mantidos-{year}{month:02d}.csv")
    dest_zip = os.path.join(RAW_DIR, f"beneficios-mantidos-{year}{month:02d}.zip")

    if os.path.exists(dest_csv) and not force:
        return dest_csv

    catalog = _get_inss_catalog()
    url = catalog.get((year, month))
    if url is None:
        logging.debug(f"  {year}-{month:02d}: not in INSS catalog — skipping.")
        return None

    # Phase 1: Download ZIP (skip if already present)
    if not os.path.exists(dest_zip):
        try:
            logging.info(f"  {year}-{month:02d}: downloading from {url} …")
            resp = requests.get(url, headers=HEADERS, timeout=300, stream=True)
            resp.raise_for_status()
            with open(dest_zip, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        except Exception as exc:
            logging.warning(f"  {year}-{month:02d}: download failed: {exc}")
            _safe_remove(dest_zip)
            return None

    # Phase 2: Extract CSV from ZIP
    try:
        with zipfile.ZipFile(dest_zip) as zf:
            # INSS ZIPs may use .CSV or .CVS extensions
            csv_names = [
                n for n in zf.namelist()
                if n.lower().endswith((".csv", ".cvs"))
            ]
            if not csv_names:
                logging.warning(
                    f"  {year}-{month:02d}: no CSV inside ZIP. "
                    f"Contents: {zf.namelist()}"
                )
                _safe_remove(dest_zip)
                return None
            with zf.open(csv_names[0]) as src, open(dest_csv, "wb") as dst:
                dst.write(src.read())
        _safe_remove(dest_zip)
        logging.info(f"  {year}-{month:02d}: extracted to {dest_csv}")
        return dest_csv
    except Exception as exc:
        logging.warning(f"  {year}-{month:02d}: ZIP extraction failed: {exc}")
        _safe_remove(dest_zip)
        _safe_remove(dest_csv)
        return None


## ─────────────────────────────────────────────────────────────────────────────
## 3) PARSE ONE MONTHLY CSV
## ─────────────────────────────────────────────────────────────────────────────

# Column name aliases across different INSS file versions
_COL_ALIASES: dict[str, list[str]] = {
    "municipio":   ["municipio", "MUNICIPIO", "cod_municipio", "COD_MUNICIPIO",
                    "codigo_municipio", "CD_MUNICIPIO", "ibge",
                    "cd_municipio_ibge", "CD_MUNICIPIO_IBGE",
                    "nr_municipio_ibge", "NR_MUNICIPIO_IBGE",
                    "CO_MUNICIPIO_IBGE", "co_municipio_ibge",
                    "CO_MUNICIPIO",     "co_municipio"],
    "competencia": ["competencia", "COMPETENCIA", "competência", "COMPETÊNCA",
                    "data_competencia", "DATA_COMPETENCIA", "periodo"],
    "especie":     ["especie", "ESPECIE", "espécie", "ESPÉCIE",
                    "cd_especie", "CD_ESPECIE", "cd_beneficio", "CD_BENEFICIO",
                    "CD_ESPECIE_BENEFICIO", "cd_especie_beneficio",
                    "NR_ESPECIE", "nr_especie"],
    "quantidade":  ["quantidade", "QUANTIDADE", "qt_beneficio", "QT_BENEFICIO",
                    "qtd", "QTD", "qtde", "QTDE",
                    "QT_BEN_ATIVO_MES", "qt_ben_ativo_mes",
                    "QT_BENEFICIO_ATIVO", "qt_beneficio_ativo"],
}


def _find_col(df: pd.DataFrame, key: str) -> str | None:
    for alias in _COL_ALIASES.get(key, []):
        if alias in df.columns:
            return alias
    lower_map = {c.lower(): c for c in df.columns}
    for alias in _COL_ALIASES.get(key, []):
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def parse_inss_csv(filepath: str, year: int, month: int) -> pd.DataFrame | None:
    """
    Parse one INSS monthly CSV.  Returns DataFrame:
      mun_code6 (int), year (int), month (int),
      especie (int), quantidade (float)
    or None on failure.
    """
    for enc in ("latin-1", "utf-8", "cp1252"):
        for sep in (";", ",", "\t"):
            try:
                df = pd.read_csv(filepath, sep=sep, encoding=enc, dtype=str,
                                 low_memory=False, on_bad_lines="skip")
                if len(df.columns) >= 3:
                    break
            except Exception:
                continue
        else:
            continue
        break
    else:
        logging.error(f"Cannot parse {filepath}")
        return None

    df.columns = [c.strip() for c in df.columns]

    mun_col  = _find_col(df, "municipio")
    esp_col  = _find_col(df, "especie")
    qtd_col  = _find_col(df, "quantidade")

    if mun_col is None:
        logging.error(f"Missing municipality column in {filepath}. "
                      f"Columns: {list(df.columns)[:10]}")
        return None

    result = pd.DataFrame()
    # Municipality code: strip trailing check digit if 7-digit
    mun_raw = df[mun_col].astype(str).str.strip().str.replace(r"\D", "", regex=True)
    result["mun_code7"] = pd.to_numeric(mun_raw, errors="coerce")
    # Truncate to 6 digits for crosswalk merge
    result["mun_code6"] = (result["mun_code7"] // 10).where(
        result["mun_code7"] >= 1000000,
        result["mun_code7"]
    ).astype("Int64")

    if esp_col is not None:
        result["especie"] = pd.to_numeric(df[esp_col], errors="coerce")
    else:
        result["especie"] = np.nan

    if qtd_col is not None:
        result["quantidade"] = (
            df[qtd_col].astype(str)
                       .str.replace(".", "", regex=False)
                       .str.replace(",", ".", regex=False)
                       .str.strip()
                       .replace({"<5": "2.5", "-": "0", "": "0"})
                       .pipe(pd.to_numeric, errors="coerce")
        )
    else:
        # Micro-data format: one row = one benefit record
        result["quantidade"] = 1.0

    result["year"]  = year
    result["month"] = month
    result.dropna(subset=["mun_code6", "quantidade"], inplace=True)
    result["mun_code6"] = result["mun_code6"].astype(int)

    logging.info(f"  Parsed {os.path.basename(filepath)}: {len(result):,} rows")
    return result[["mun_code6", "year", "month", "especie", "quantidade"]]


## ─────────────────────────────────────────────────────────────────────────────
## 4) AGGREGATE TO MCA × QUARTER
## ─────────────────────────────────────────────────────────────────────────────

def load_mca_crosswalk() -> pd.DataFrame:
    """Load crosswalk; 6-digit municipality codes for matching INSS data."""
    mca = pd.read_csv(MCA_CSV,
                      usecols=["municipality_code", "mca_code", "year"],
                      dtype={"municipality_code": int, "mca_code": str, "year": int})
    mca["mun_code6"] = mca["municipality_code"] // 10   # drop check digit → 6-digit
    return mca[["mun_code6", "mca_code", "year"]]


def aggregate_to_mca_quarter(inss: pd.DataFrame,
                              crosswalk: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to retirement/pension species, add quarter, merge MCA, aggregate.
    """
    # Keep only retirement + pension species
    ret = inss[inss["especie"].isin(SPECIES_RETIREMENT)].copy()
    ret["quarter"] = ((ret["month"] - 1) // 3 + 1).astype(int)

    # Merge MCA crosswalk
    ret = ret.merge(crosswalk, on=["mun_code6", "year"], how="left")
    n_miss = ret["mca_code"].isna().sum()
    if n_miss > 0:
        logging.warning(f"{n_miss:,} INSS rows unmatched to MCA — dropped.")
    ret = ret.dropna(subset=["mca_code"])

    # Aggregate within MCA × year × quarter using the last-month-of-quarter stock
    # (take max month in quarter as stock)
    last_month_in_qtr = ret.groupby(["mca_code", "year", "quarter"])["month"].transform("max")
    stock = ret[ret["month"] == last_month_in_qtr].copy()

    agg = (
        stock.groupby(["mca_code", "year", "quarter"])
             .agg(
                 retirees_total   = ("quantidade",  "sum"),
                 n_municipalities = ("mun_code6",   "nunique"),
             )
             .reset_index()
    )
    return agg


def merge_population(panel: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(DEMO_CSV):
        logging.warning("Demographics panel not found; retirees_per1000 will be NaN.")
        panel["retirees_per1000"] = np.nan
        return panel

    demo = pd.read_csv(DEMO_CSV,
                       usecols=["mca_code", "year", "pop_total"],
                       dtype={"mca_code": str, "year": int})
    panel = panel.merge(demo, on=["mca_code", "year"], how="left")
    panel["retirees_per1000"] = np.where(
        panel["pop_total"] > 0,
        panel["retirees_total"] / (panel["pop_total"] / 1000),
        np.nan,
    )
    panel.drop(columns=["pop_total"], inplace=True)
    return panel


## ─────────────────────────────────────────────────────────────────────────────
## 5) MAIN
## ─────────────────────────────────────────────────────────────────────────────

def _download_and_parse_month(year: int, month: int) -> pd.DataFrame | None:
    """Worker: download and parse one INSS monthly file (called concurrently)."""
    fp = download_inss_month(year, month)
    if fp is None:
        return None
    return parse_inss_csv(fp, year, month)


def main():
    crosswalk = load_mca_crosswalk()

    # Build flat list of (year, month) pairs to avoid sequential nested loops
    periods = [
        (y, m)
        for y in range(PANEL_START_YEAR, PANEL_END_YEAR + 1)
        for m in range(1, 13)
        if not (y == PANEL_END_YEAR and m > 8)
    ]
    logging.info(f"Downloading {len(periods)} INSS monthly files "
                 f"(up to 4 concurrent) …")

    all_frames: list[pd.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {
            pool.submit(_download_and_parse_month, y, m): (y, m)
            for y, m in periods
        }
        for future in concurrent.futures.as_completed(future_map):
            year, month = future_map[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    all_frames.append(df)
            except Exception as exc:
                logging.error(f"  {year}-{month:02d} error: {exc}")

    if not all_frames:
        raise RuntimeError("No INSS data loaded.")

    inss_all = pd.concat(all_frames, ignore_index=True)
    logging.info(f"Total INSS rows: {len(inss_all):,}")

    panel = aggregate_to_mca_quarter(inss_all, crosswalk)
    panel = merge_population(panel)

    col_order = ["mca_code", "year", "quarter",
                 "retirees_total", "retirees_per1000", "n_municipalities"]
    panel = panel[[c for c in col_order if c in panel.columns]]
    panel.sort_values(["mca_code", "year", "quarter"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    panel.to_csv(OUTPUT_CSV, index=False, encoding="latin-1")
    logging.info(f"Saved INSS panel to {OUTPUT_CSV}")

    print(
        f"\nINSS retirement panel summary"
        f"\n  Rows:                {len(panel):,}"
        f"\n  MCAs:                {panel['mca_code'].nunique()}"
        f"\n  Years:               {panel['year'].min()} – {panel['year'].max()}"
        f"\n  retirees_per1000 non-null: {panel['retirees_per1000'].notna().sum():,}"
        f"\n  Output:              {OUTPUT_CSV}"
    )
    print(panel.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
