## scrape_anatel.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-03-06
#
# Purpose: Download ANATEL "Acessos em Telefonia Móvel" flat files and
#          build an MCA × quarter panel of mobile-connectivity indicators.
#
#   Data source:
#     ANATEL Dados Abertos — "Acessos em Telefonia Móvel por Município"
#     URL (annual flat files, ~100 MB each, CSV ISO-8859-1 encoded):
#       https://dados.anatel.gov.br/dataset/df4fd8f0-0ad3-4b8d-8d6e-88bcf23db88f/
#         resource/<resource_id>/download/acessos_telefonia_movel_<YYYY>.csv
#
#     NOTE: ANATEL updates resource IDs when they republish files. The dict
#     ANATEL_URLS below was compiled from the Portal Dados Abertos metadata as
#     of early 2025.  If a URL returns 404, visit
#       https://dados.gov.br/dados/conjuntos-dados/acessos-em-telefonia-movel
#     to find the current resource IDs and update the dict.
#
#   Input files (downloaded):
#     ANATEL/raw/acessos_telefonia_movel_<YYYY>.csv  (one per year, 2013–2024)
#
#   Output:
#     ANATEL/anatel_mca_panel.csv
#       Columns: mca_code, year, quarter,
#                connections_total,        (sum of all active accesses in MCA)
#                connections_per100,       (connections_total / pop * 100)
#                connections_4g5g,         (4G + 5G accesses)
#                frac_4g5g,                (connections_4g5g / connections_total)
#                n_municipalities          (municipalities contributing to MCA-quarter)
#
# Notes:
#   • ANATEL files contain monthly snapshots (Mês column) in long form.
#     Only the last month of each quarter is retained to represent a stock.
#   • Technology classification: column "Tecnologia" with values
#     "2G", "3G", "4G", "4G+", "5G", etc.  4G5G_TECHS set below.
#   • Municipality codes are typically 7-digit IBGE codes in the "Código IBGE"
#     column (or "CodMunicipio"), though naming varies across years.
#     A column-name normalizer handles the main variants.
#   • ANATEL publishes a separate "Accesses" column for each operator/technology
#     combination per row.  After normalizing, sum Acessos by municipality ×
#     month × technology.
###─────────────────────────────────────────────────────────────────────────────

import gc
import os
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

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

## ─────────────────────────────────────────────────────────────────────────────
## 1) PATHS & CONSTANTS
## ─────────────────────────────────────────────────────────────────────────────
from utils import paths
BASE       = str(paths.OPEN_FINANCE)
ANATEL_DIR = str(paths.ANATEL_DIR)   # processed anatel_*_panel outputs (kept in place)
RAW_DIR    = str(paths.ANATEL_RAW)   # raw acessos_* downloads (consolidated under raw/)
IBGE_DIR   = str(paths.IBGE_DIR)
MCA_CSV    = os.path.join(IBGE_DIR, "muni_mca_regions_2010_2024_panel.csv")
DEMO_CSV   = os.path.join(IBGE_DIR, "mca_demographics_panel.csv")
OUTPUT_CSV = os.path.join(ANATEL_DIR, "anatel_mca_panel.csv")

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "scrape_anatel",
        {
            "anatel_dir": ANATEL_DIR,
            "raw_dir": RAW_DIR,
            "mca_csv": MCA_CSV,
            "demo_csv": DEMO_CSV,
            "output_csv": OUTPUT_CSV,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    ANATEL_DIR = _paths["anatel_dir"]
    RAW_DIR = _paths["raw_dir"]
    MCA_CSV = _paths["mca_csv"]
    DEMO_CSV = _paths["demo_csv"]
    OUTPUT_CSV = _paths["output_csv"]

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ANATEL_DIR, exist_ok=True)

# Technologies counted as 4G/5G (fast mobile).
# Includes both the generation labels ("4G","5G",...) used in the
# "Tecnologia Geração" column AND the raw-technology labels in the "Tecnologia"
# column: "LTE" (=4G) and "NR" (=5G New Radio). NR variants are listed so 5G is
# never dropped when only the raw "Tecnologia" column is present.
FAST_TECHS = {"4G", "4G+", "4.5G", "5G", "4G-LTE", "LTE", "LTE-A",
              "NR", "NR NSA", "NR SA", "5G NR", "5G-NR", "5GNR"}

# Panel coverage
PANEL_START_YEAR = 2013
PANEL_END_YEAR   = 2025

# ── ANATEL direct-download URLs ───────────────────────────────────────────────
# Source: https://dados.gov.br/dados/conjuntos-dados/acessos-em-telefonia-movel
# These are large CSVs (~100–300 MB each). Update resource IDs if they change.
# ANATEL omnibus ZIP (all years in a single ~2.9 GB archive)
# Source: https://www.anatel.gov.br/dadosabertos/paineis_de_dados/acessos/
ANATEL_ZIP_URL  = (
    "https://www.anatel.gov.br/dadosabertos/paineis_de_dados/acessos/"
    "acessos_telefonia_movel.zip"
)
ANATEL_ZIP_PATH = os.path.join(RAW_DIR, "acessos_telefonia_movel.zip")


# Sidecar storing the Last-Modified / ETag of the cached ANATEL zip, so we can
# detect when ANATEL republishes a revised omnibus file (HTTP conditional check).
ANATEL_ZIP_META = ANATEL_ZIP_PATH + ".meta"


def _anatel_remote_validators() -> tuple[str | None, str | None]:
    """HEAD the ANATEL zip and return (Last-Modified, ETag), or (None, None) on failure."""
    headers = {"User-Agent": "research/pedro.feijodemoraes@yale.edu"}
    try:
        r = requests.head(ANATEL_ZIP_URL, headers=headers, timeout=60, allow_redirects=True)
        r.raise_for_status()
        return r.headers.get("Last-Modified"), r.headers.get("ETag")
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"ANATEL HEAD request failed ({exc}); cannot check for a revised file.")
        return None, None


def _read_anatel_meta() -> dict:
    import json
    try:
        with open(ANATEL_ZIP_META, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _write_anatel_meta(last_modified: str | None, etag: str | None) -> None:
    import json
    try:
        with open(ANATEL_ZIP_META, "w", encoding="utf-8") as fh:
            json.dump({"last_modified": last_modified, "etag": etag}, fh)
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"Could not write ANATEL meta sidecar: {exc}")


def _ensure_anatel_zip() -> bool:
    """Ensure the ANATEL omnibus ZIP is present and current. Returns True on success.

    The zip is a single ~2.9 GB omnibus file (all years), so the rolling
    period-window used by the other scrapers does not apply.  Instead we use an
    HTTP conditional check: a cheap HEAD returns the server's Last-Modified /
    ETag, and the zip is re-downloaded only when ANATEL has republished a revised
    file.  If the HEAD fails we keep the existing copy (never re-pull 2.9 GB on a
    transient network error)."""
    remote_lm, remote_etag = _anatel_remote_validators()

    zip_valid = False
    if os.path.exists(ANATEL_ZIP_PATH):
        try:
            with zipfile.ZipFile(ANATEL_ZIP_PATH) as zf:
                zip_valid = len(zf.namelist()) > 0
        except Exception:
            logging.warning("Existing ANATEL ZIP is corrupt or incomplete; deleting and redownloading...")
            try:
                os.remove(ANATEL_ZIP_PATH)
            except Exception as e:
                logging.error(f"Could not remove corrupt ZIP: {e}")
                return False

    if zip_valid:
        if remote_lm is None and remote_etag is None:
            logging.info("ANATEL: server revision check unavailable; using existing cached ZIP.")
            return True
        meta = _read_anatel_meta()
        unchanged = (
            (remote_etag and remote_etag == meta.get("etag"))
            or (remote_lm and remote_lm == meta.get("last_modified"))
        )
        if unchanged:
            logging.info(f"ANATEL ZIP unchanged on server (Last-Modified {remote_lm}); skipping re-download.")
            return True
        logging.info(f"ANATEL ZIP revised on server (Last-Modified {remote_lm}); re-downloading...")
        try:
            os.remove(ANATEL_ZIP_PATH)
        except OSError:
            pass

    headers = {"User-Agent": "research/pedro.feijodemoraes@yale.edu"}
    logging.info(f"Downloading ANATEL omnibus ZIP (~2.9 GB) from {ANATEL_ZIP_URL} ...")
    try:
        with requests.get(ANATEL_ZIP_URL, headers=headers, timeout=3600,
                          stream=True) as resp:
            resp.raise_for_status()
            with open(ANATEL_ZIP_PATH, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=4 << 20):
                    fh.write(chunk)
            dl_lm = resp.headers.get("Last-Modified", remote_lm)
            dl_etag = resp.headers.get("ETag", remote_etag)
        logging.info(
            f"Saved ANATEL omnibus ZIP "
            f"({os.path.getsize(ANATEL_ZIP_PATH) / 1e9:.2f} GB) -> {ANATEL_ZIP_PATH}"
        )
        _write_anatel_meta(dl_lm, dl_etag)
        return True
    except Exception as exc:
        logging.error(f"ANATEL ZIP download failed: {exc}")
        try:
            if os.path.exists(ANATEL_ZIP_PATH):
                os.remove(ANATEL_ZIP_PATH)
        except OSError:
            pass
        return False


def download_anatel_file(year: int, semester: str | None = None,
                         force: bool = False) -> str | None:
    """
    Extract the ANATEL CSV for a given year (and semester for 2019+) from the
    omnibus ZIP.  Returns the local CSV path, or None on failure.
      year <= 2018: semester=None -> Acessos_Telefonia_Movel_2005-2018_Tecnologia.csv
      year >= 2019: semester in ("1S", "2S")
    """
    if year <= 2018:
        member = "Acessos_Telefonia_Movel_2005-2018_Tecnologia.csv"
        dest   = os.path.join(RAW_DIR, "acessos_telefonia_movel_2005-2018.csv")
    else:
        sem    = semester or "1S"
        member = f"Acessos_Telefonia_Movel_{year}_{sem}.csv"
        dest   = os.path.join(RAW_DIR, f"acessos_telefonia_movel_{year}_{sem}.csv")

    if os.path.exists(dest) and not force:
        return dest

    if not _ensure_anatel_zip():
        return None

    try:
        with zipfile.ZipFile(ANATEL_ZIP_PATH) as zf:
            names = zf.namelist()
            match = next(
                (n for n in names if n.split("/")[-1].lower() == member.lower()),
                None,
            )
            if match is None:
                match = next(
                    (n for n in names if n.lower().endswith(member.lower())), None
                )
            if match is None:
                logging.warning(
                    f"Member {member!r} not found in ZIP. "
                    f"Available: {[n.split('/')[-1] for n in names if 'Colunas' not in n][:10]}"
                )
                return None
            with zf.open(match) as src, open(dest, "wb") as dst:
                dst.write(src.read())
        logging.info(f"Extracted {member} -> {dest}")
        return dest
    except Exception as exc:
        logging.error(f"Failed to extract {member}: {exc}")
        if os.path.exists(dest):
            os.remove(dest)
        return None


## ─────────────────────────────────────────────────────────────────────────────
## 3) PARSE AN ANATEL CSV
## ─────────────────────────────────────────────────────────────────────────────

# Column aliases across different ANATEL file years
_COL_ALIASES: dict[str, list[str]] = {
    "mun_code":   ["Código IBGE Município", "Código IBGE", "CodMunicipio",
                   "Cod_Municipio", "codigo_ibge", "COD_MUNICIPIO",
                   "cod_municipio", "ibge"],
    "tecnologia": ["Tecnologia", "Tecnolog", "technology", "TECNOLOGIA"],
    "acessos":    ["Acessos", "acessos", "ACESSOS", "Qtd_Acessos", "QTD_ACESSOS"],
    "mes":        ["Mês", "mes", "MES", "Mes", "month", "MONTH", "Competência"],
    "ano":        ["Ano", "ano", "ANO", "year", "YEAR"],
}


def _find_col(df: pd.DataFrame, key: str) -> str | None:
    """Return the first column in df whose name matches any alias for `key`."""
    for alias in _COL_ALIASES.get(key, []):
        if alias in df.columns:
            return alias
    # Case-insensitive fallback
    lower_map = {c.lower(): c for c in df.columns}
    for alias in _COL_ALIASES.get(key, []):
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def parse_anatel_csv(filepath: str, year: int,
                     year_filter: int | None = None) -> pd.DataFrame | None:
    """
    Read and normalize one ANATEL CSV file.
    Returns DataFrame: mun_code (int), year (int), month (int),
                       tecnologia (str), acessos (float)
    or None if the file cannot be parsed.
    If year_filter is set, keep only rows where year == year_filter.
    """
    # Try utf-8-sig first (strips UTF-8 BOM, common in Brazilian govt files),
    # then fallbacks for older files
    for enc in ("utf-8-sig", "latin-1", "utf-8", "cp1252"):
        try:
            df = pd.read_csv(filepath, sep=";", encoding=enc, dtype=str,
                             low_memory=False, on_bad_lines="warn")
            if len(df.columns) < 3:
                # Maybe comma-delimited
                df = pd.read_csv(filepath, sep=",", encoding=enc, dtype=str,
                                 low_memory=False, on_bad_lines="warn")
            if len(df.columns) >= 3:
                break
        except Exception:
            continue
    else:
        logging.error(f"Cannot parse {filepath}")
        return None

    df.columns = [c.strip() for c in df.columns]

    # Identify required columns
    mun_col  = _find_col(df, "mun_code")
    # Prefer the unambiguous generation column ("Tecnologia Geração": 2G/3G/4G/5G)
    # over the raw "Tecnologia" column (LTE/NR/GSM/WCDMA). In the raw column 5G is
    # labelled "NR", which FAST_TECHS historically missed -> 5G undercounted. The
    # generation column uses "5G" directly, which FAST_TECHS matches. Falls back to
    # the raw column (now NR-aware via FAST_TECHS) if no generation column exists.
    tec_col  = (next((c for c in df.columns
                      if "tecnolog" in c.lower() and "gera" in c.lower()), None)
                or _find_col(df, "tecnologia"))
    acc_col  = _find_col(df, "acessos")
    mes_col  = _find_col(df, "mes")
    ano_col  = _find_col(df, "ano")

    if mun_col is None or acc_col is None:
        logging.error(f"Cannot identify municipality/acessos columns in {filepath}. "
                      f"Columns: {list(df.columns)[:10]}")
        return None

    result = pd.DataFrame()
    result["mun_code"]   = pd.to_numeric(df[mun_col].str.strip(), errors="coerce")
    result["acessos"]    = (
        df[acc_col].str.replace(".", "", regex=False)
                   .str.replace(",", ".", regex=False)
                   .str.strip()
                   .pipe(pd.to_numeric, errors="coerce")
    )
    result["tecnologia"] = df[tec_col].str.strip().str.upper() if tec_col else "UNKNOWN"

    if mes_col is not None:
        result["month"] = pd.to_numeric(df[mes_col].str.strip(), errors="coerce")
    else:
        result["month"] = 12   # assume December if no month column

    if ano_col is not None:
        result["year"] = pd.to_numeric(df[ano_col].str.strip(), errors="coerce")
    else:
        result["year"] = year

    result.dropna(subset=["mun_code", "acessos", "year", "month"], inplace=True)
    result["mun_code"] = result["mun_code"].astype(int)
    result["year"]     = result["year"].astype(int)
    result["month"]    = result["month"].astype(int)

    if year_filter is not None:
        result = result[result["year"] == year_filter].copy()

    logging.info(f"  Parsed {filepath}: {len(result):,} rows")
    return result


## ─────────────────────────────────────────────────────────────────────────────
## 4) AGGREGATE TO MCA × QUARTER
## ─────────────────────────────────────────────────────────────────────────────

def add_quarter_and_fast(df: pd.DataFrame) -> pd.DataFrame:
    # Avoid copying the full frame; compute in-place on the existing object
    df = df.assign(
        quarter = ((df["month"] - 1) // 3 + 1).astype(int),
        is_fast = df["tecnologia"].isin(FAST_TECHS).astype(int),
    )
    return df


def load_mca_crosswalk() -> pd.DataFrame:
    return pd.read_csv(MCA_CSV,
                       usecols=["municipality_code", "mca_code", "year"],
                       dtype={"municipality_code": int, "mca_code": str, "year": int})


def aggregate_to_mca_quarter(anatel: pd.DataFrame,
                              crosswalk: pd.DataFrame,
                              _muni_frames: list | None = None) -> pd.DataFrame:
    """
    1. Keep only the last month of each quarter (stock snapshot).
    2. Merge MCA code.
    3. Aggregate to MCA × year × quarter.
    If `_muni_frames` list is passed, appends municipality-level data for σ computation.
    """
    # Last month of quarter: month 3, 6, 9, 12
    last_month = {1: 3, 2: 6, 3: 9, 4: 12}
    mask = anatel["month"] == anatel["quarter"].map(last_month)
    # Filter first, then copy to avoid consolidating the full ~28M-row frame
    anatel = anatel.loc[mask, ["mun_code", "year", "month", "quarter", "acessos", "is_fast"]].copy()

    # Merge MCA
    anatel = anatel.merge(crosswalk, left_on=["mun_code", "year"],
                          right_on=["municipality_code", "year"], how="left")
    n_miss = anatel["mca_code"].isna().sum()
    if n_miss > 0:
        logging.warning(f"{n_miss:,} ANATEL rows unmatched to MCA — dropped.")
    anatel = anatel.dropna(subset=["mca_code"])
    anatel["acessos_fast"] = anatel["acessos"].where(anatel["is_fast"] == 1, 0)

    # Save municipality-level intermediate for within-MCA σ computation
    if _muni_frames is not None:
        _muni_frames.append(
            anatel[["mun_code", "mca_code", "year", "quarter",
                    "acessos", "acessos_fast"]].copy()
        )

    agg = (
        anatel.groupby(["mca_code", "year", "quarter"])
              .agg(
                  connections_total = ("acessos",      "sum"),
                  connections_4g5g  = ("acessos_fast", "sum"),
                  n_municipalities  = ("mun_code",     "nunique"),
              )
              .reset_index()
    )
    return agg


def merge_population_for_per100(panel: pd.DataFrame) -> pd.DataFrame:
    """Add connections_per100 using MCA population from the demographics panel."""
    if not os.path.exists(DEMO_CSV):
        logging.warning(f"Demographics panel not found at {DEMO_CSV}. "
                        "connections_per100 will be NaN.")
        panel["connections_per100"] = np.nan
        panel["frac_4g5g"]          = np.nan
        return panel

    demo = pd.read_csv(DEMO_CSV,
                       usecols=["mca_code", "year", "pop_total"],
                       dtype={"mca_code": str, "year": int})

    panel = panel.merge(demo, on=["mca_code", "year"], how="left")
    panel["connections_per100"] = np.where(
        panel["pop_total"] > 0,
        panel["connections_total"] / panel["pop_total"] * 100,
        np.nan,
    )
    panel["frac_4g5g"] = np.where(
        panel["connections_total"] > 0,
        panel["connections_4g5g"] / panel["connections_total"],
        np.nan,
    )
    panel.drop(columns=["pop_total"], inplace=True)
    return panel


## ─────────────────────────────────────────────────────────────────────────────
## 5) PRE-2019 DDD APPORTIONMENT (Option B — population-weighted)
## ─────────────────────────────────────────────────────────────────────────────
#
# ANATEL files prior to 2019 report connections at the DDD area-code level.
# The 67 Brazilian DDD codes do not align 1-to-1 with the 468 MCAs: 81 MCAs
# (17%) span multiple DDDs, so we cannot simply assign a DDD total to an MCA.
#
# Solution: build a municipality→DDD crosswalk from the first 2019+ file (which
# has both Código IBGE Município and Código Nacional), then estimate each
# municipality's share of its DDD's population by splitting each MCA's
# population equally among its member municipalities, and finally aggregate the
# allocated connections up to the MCA level.
#
# The assumption "uniform pop within MCA" is conservative; it would only
# matter for the 81 multi-DDD MCAs, which are typically large geographic MCAs
# where the dominant DDD holds >90% of the population.
# ─────────────────────────────────────────────────────────────────────────────

def _build_ddd_mun_crosswalk() -> pd.DataFrame:
    """
    Build a municipality → DDD crosswalk from the earliest available 2019+ file.
    Municipality → DDD is injective (every municipality belongs to exactly one
    DDD), verified from the 2024 file.

    Returns DataFrame with columns:
        mun_code  (int, 7-digit IBGE code)
        ddd       (str, e.g. '11')
    """
    fp = os.path.join(RAW_DIR, "acessos_telefonia_movel_2019_1S.csv")
    if not os.path.exists(fp):
        fp = download_anatel_file(2019, semester="1S")
    if fp is None or not os.path.exists(fp):
        raise RuntimeError(
            "Cannot build DDD crosswalk: 2019_1S ANATEL file is unavailable. "
            "Run download_anatel_file(2019, '1S') first."
        )

    logging.info("Building municipality→DDD crosswalk from 2019_1S file …")
    df = pd.read_csv(
        fp, sep=";", encoding="utf-8-sig", dtype=str, low_memory=False,
        usecols=["Código IBGE Município", "Código Nacional"],
    )
    df.columns = ["mun_code", "ddd"]
    df["mun_code"] = pd.to_numeric(df["mun_code"].str.strip(), errors="coerce")
    df["ddd"]      = df["ddd"].str.strip()
    df = df.dropna(subset=["mun_code", "ddd"])
    df["mun_code"] = df["mun_code"].astype(int)

    # Municipality → DDD is injective; one row per municipality is enough.
    xwalk = (
        df.drop_duplicates("mun_code")[["mun_code", "ddd"]]
          .reset_index(drop=True)
    )
    logging.info(f"DDD crosswalk: {len(xwalk):,} municipalities mapped to "
                 f"{xwalk['ddd'].nunique()} DDDs")
    return xwalk


def _build_mun_pop_weights(years: list[int]) -> pd.DataFrame:
    """
    Estimate each municipality's population share within its DDD for each year
    in `years`.

    Method
    ------
    1. From the MCA crosswalk compute n_munis = number of municipalities in each
       MCA×year.
    2. From the demographics panel get MCA pop_total per year.
    3. Approximate municipality population as pop_MCA / n_munis (uniform split).
    4. Sum municipality pop estimates within each DDD×year to get ddd_pop_total.
    5. pop_weight = mun_pop_est / ddd_pop_total.

    Returns DataFrame with columns:
        mun_code, mca_code, ddd, year, pop_weight
    """
    xwalk = pd.read_csv(
        MCA_CSV,
        usecols=["municipality_code", "mca_code", "year"],
        dtype={"municipality_code": int, "mca_code": str, "year": int},
    )
    xwalk = xwalk[xwalk["year"].isin(years)].copy()

    # Number of municipalities per MCA × year
    n_munis = (
        xwalk.groupby(["mca_code", "year"])["municipality_code"]
             .count()
             .reset_index()
             .rename(columns={"municipality_code": "n_munis"})
    )
    xwalk = xwalk.merge(n_munis, on=["mca_code", "year"], how="left")

    # MCA-level population from demographics panel
    if not os.path.exists(DEMO_CSV):
        raise RuntimeError(
            f"Demographics panel not found at {DEMO_CSV}. "
            "Run scrape_ibge_demographics.py first."
        )
    demo = pd.read_csv(
        DEMO_CSV,
        usecols=["mca_code", "year", "pop_total"],
        dtype={"mca_code": str, "year": int},
    )
    demo = demo[demo["year"].isin(years)]
    xwalk = xwalk.merge(demo, on=["mca_code", "year"], how="left")

    # Uniform split: municipality pop ≈ MCA pop / n_municipalities
    xwalk["mun_pop_est"] = xwalk["pop_total"] / xwalk["n_munis"]

    # Attach DDD code to each municipality
    ddd_xwalk = _build_ddd_mun_crosswalk()
    xwalk = xwalk.merge(
        ddd_xwalk, left_on="municipality_code", right_on="mun_code", how="inner"
    )

    # DDD-level total population in each year
    ddd_pop = (
        xwalk.groupby(["ddd", "year"])["mun_pop_est"]
             .sum()
             .reset_index()
             .rename(columns={"mun_pop_est": "ddd_pop_total"})
    )
    xwalk = xwalk.merge(ddd_pop, on=["ddd", "year"], how="left")

    # Population weight of each municipality within its DDD×year
    xwalk["pop_weight"] = (
        xwalk["mun_pop_est"] / xwalk["ddd_pop_total"]
    ).fillna(0.0)

    result = (
        xwalk[["municipality_code", "mca_code", "ddd", "year", "pop_weight"]]
        .rename(columns={"municipality_code": "mun_code"})
        .copy()
    )
    logging.info(
        f"Municipality pop-weights: {len(result):,} rows, "
        f"years {years[0]}–{years[-1]}, "
        f"{(result['pop_weight'] == 0).sum()} zero-weight entries"
    )
    return result


def build_pre2019_panel(crosswalk: pd.DataFrame) -> pd.DataFrame | None:
    """
    Build an MCA × quarter ANATEL panel for years PANEL_START_YEAR–2018
    using population-weighted DDD apportionment.

    Steps
    -----
    1. Read the 2005–2018 ANATEL flat file (DDD-level granularity).
    2. Filter to PANEL_START_YEAR ≤ year ≤ 2018.
    3. Keep only last month of each quarter (stock snapshot).
    4. Aggregate connections to DDD × year × quarter (total & 4G/5G).
    5. Load municipality pop-weights (share within DDD×year).
    6. Allocate DDD connections to municipalities proportionally.
    7. Aggregate to MCA × year × quarter, merging with the MCA crosswalk.
    8. Add connections_per100 / frac_4g5g via merge_population_for_per100.

    Returns DataFrame with the same columns as the post-2018 panel, or None.
    """
    fp_pre = os.path.join(RAW_DIR, "acessos_telefonia_movel_2005-2018.csv")
    if not os.path.exists(fp_pre):
        logging.info("Pre-2019 ANATEL file not cached; extracting from ZIP …")
        fp_pre = download_anatel_file(2016)   # any year ≤ 2018 triggers the extract
    if fp_pre is None or not os.path.exists(fp_pre):
        logging.warning("Pre-2019 ANATEL file unavailable — skipping 2016-2018.")
        return None

    logging.info("Processing pre-2019 ANATEL file (DDD apportionment) …")

    # ── Step 1: read raw file ──────────────────────────────────────────────
    raw = None
    for enc in ("utf-8-sig", "latin-1", "utf-8", "cp1252"):
        try:
            raw = pd.read_csv(
                fp_pre, sep=";", encoding=enc, dtype=str, low_memory=False
            )
            if len(raw.columns) >= 5:
                break
        except Exception:
            raw = None
    if raw is None or raw.empty:
        logging.error(f"Could not parse pre-2019 ANATEL file at {fp_pre}")
        return None

    raw.columns = [c.strip() for c in raw.columns]

    # Rename columns to canonical names (handle BOM and encoding variants)
    col_map: dict[str, str] = {}
    for col in raw.columns:
        lc = col.lower()
        if "ano" in lc:
            col_map[col] = "year"
        elif "m" in lc and "s" in lc and len(col) <= 4:   # Mês / Mes / MÃªs
            col_map[col] = "month"
        elif "digo nacional" in lc or "codigo nacional" in lc or col_map.get(col) == "ddd":
            col_map[col] = "ddd"
        elif "gera" in lc:
            col_map[col] = "tec_gen"
        elif "acessos" in lc.replace("ç", "c"):
            col_map[col] = "acessos"
    # Also handle clean column names (utf-8-sig decoded cleanly)
    for col in raw.columns:
        if col == "Ano":           col_map[col] = "year"
        elif col == "Mês":         col_map[col] = "month"
        elif col == "Código Nacional": col_map[col] = "ddd"
        elif col == "Tecnologia Geração": col_map[col] = "tec_gen"
        elif col == "Acessos":     col_map[col] = "acessos"
    raw.rename(columns=col_map, inplace=True)

    required = {"year", "month", "ddd", "acessos"}
    missing  = required - set(raw.columns)
    if missing:
        logging.error(
            f"Pre-2019 file missing columns {missing} after rename. "
            f"Available: {list(raw.columns)}"
        )
        return None

    raw["year"]   = pd.to_numeric(raw["year"],  errors="coerce")
    raw["month"]  = pd.to_numeric(raw["month"], errors="coerce")
    raw["acessos"] = (
        raw["acessos"].str.replace(".", "", regex=False)
                      .str.replace(",", ".", regex=False)
                      .str.strip()
                      .pipe(pd.to_numeric, errors="coerce")
    )
    raw["ddd"] = raw["ddd"].str.strip()
    raw.dropna(subset=["year", "month", "ddd", "acessos"], inplace=True)
    raw["year"]  = raw["year"].astype(int)
    raw["month"] = raw["month"].astype(int)

    # ── Step 2: filter to panel years ─────────────────────────────────────
    target_years = list(range(PANEL_START_YEAR, 2019))
    raw = raw[raw["year"].isin(target_years)].copy()
    if raw.empty:
        logging.warning("No data in target years in pre-2019 ANATEL file.")
        return None

    # ── Step 3: quarter + fast-tech flag + keep last month of quarter ─────
    raw["quarter"] = ((raw["month"] - 1) // 3 + 1).astype(int)
    last_month_map = {1: 3, 2: 6, 3: 9, 4: 12}
    raw = raw[raw["month"] == raw["quarter"].map(last_month_map)].copy()

    if "tec_gen" in raw.columns:
        raw["is_fast"] = (
            raw["tec_gen"].str.strip().str.upper()
                .isin({"4G", "4G+", "4.5G", "5G"})
        ).astype(int)
    else:
        raw["is_fast"] = 0

    # ── Step 4: aggregate to DDD × year × quarter ─────────────────────────
    total = (
        raw.groupby(["ddd", "year", "quarter"])["acessos"]
           .sum().reset_index().rename(columns={"acessos": "conn_total"})
    )
    fast = (
        raw[raw["is_fast"] == 1]
           .groupby(["ddd", "year", "quarter"])["acessos"]
           .sum().reset_index().rename(columns={"acessos": "conn_fast"})
    )
    ddd_panel = total.merge(fast, on=["ddd", "year", "quarter"], how="left")
    ddd_panel["conn_fast"] = ddd_panel["conn_fast"].fillna(0.0)

    ddds_in_file = set(ddd_panel["ddd"].unique())
    logging.info(
        f"Pre-2019 DDD panel: {len(ddd_panel):,} rows, "
        f"{len(ddds_in_file)} unique DDDs, "
        f"years {ddd_panel['year'].min()}–{ddd_panel['year'].max()}"
    )

    # ── Step 5: load population weights ───────────────────────────────────
    actual_years = sorted(ddd_panel["year"].unique().tolist())
    try:
        pop_weights = _build_mun_pop_weights(actual_years)
    except Exception as exc:
        logging.error(f"Could not build pop weights: {exc}")
        return None

    # Warn about DDDs in the file with no municipality mapping
    unmapped_ddds = ddds_in_file - set(pop_weights["ddd"].unique())
    if unmapped_ddds:
        pct_conn = (
            ddd_panel[ddd_panel["ddd"].isin(unmapped_ddds)]["conn_total"].sum()
            / ddd_panel["conn_total"].sum() * 100
        )
        logging.warning(
            f"{len(unmapped_ddds)} DDDs in pre-2019 file have no municipality "
            f"mapping and will be dropped ({pct_conn:.1f}% of total connections): "
            f"{sorted(unmapped_ddds)}"
        )

    # ── Step 6: allocate DDD connections → municipalities ─────────────────
    # pop_weights has: mun_code, mca_code, ddd, year, pop_weight
    # ddd_panel has:   ddd, year, quarter, conn_total, conn_fast
    # Merge on ddd × year to broadcast quarterly entries across municipalities
    allocated = pop_weights.merge(ddd_panel, on=["ddd", "year"], how="inner")
    allocated["mun_conn_total"] = allocated["conn_total"] * allocated["pop_weight"]
    allocated["mun_conn_fast"]  = allocated["conn_fast"]  * allocated["pop_weight"]

    # ── Step 7: aggregate to MCA × year × quarter ─────────────────────────
    mca_agg = (
        allocated.groupby(["mca_code", "year", "quarter"])
                 .agg(
                     connections_total = ("mun_conn_total", "sum"),
                     connections_4g5g  = ("mun_conn_fast",  "sum"),
                     n_municipalities  = ("mun_code",       "nunique"),
                 )
                 .reset_index()
    )
    # Note: connections_per100 and frac_4g5g are computed in main() after
    # all panels are concatenated via merge_population_for_per100.

    logging.info(
        f"Pre-2019 MCA panel: {len(mca_agg):,} rows, "
        f"{mca_agg['mca_code'].nunique()} MCAs, "
        f"years {mca_agg['year'].min()}–{mca_agg['year'].max()}"
    )
    return mca_agg


## ─────────────────────────────────────────────────────────────────────────────
## 6) MAIN
## ─────────────────────────────────────────────────────────────────────────────

def _download_and_parse(year: int) -> pd.DataFrame | None:
    """Worker: download and parse one ANATEL annual file (2019+ only)."""
    if year <= 2018:
        return None   # handled by build_pre2019_panel
    else:
        frames: list[pd.DataFrame] = []
        for sem in ("1S", "2S"):
            fp = download_anatel_file(year, semester=sem)
            if fp is None:
                logging.warning(f"  {year} {sem}: download failed — skipping.")
                continue
            raw = parse_anatel_csv(fp, year)
            if raw is None or raw.empty:
                logging.warning(f"  {year} {sem}: parse returned no data — skipping.")
                continue
            frames.append(add_quarter_and_fast(raw))
        if not frames:
            logging.warning(f"  {year}: no semester data — skipping.")
            return None
        return pd.concat(frames, ignore_index=True)


def main():
    # Early exit: if output already exists AND muni intermediate exists, skip
    muni_csv = os.path.join(ANATEL_DIR, "anatel_muni_panel.csv")
    if (os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
            and os.path.exists(muni_csv) and os.path.getsize(muni_csv) > 0):
        print(f"Output already exists, skipping: {OUTPUT_CSV}")
        logging.info(f"Output already exists -- skipping rebuild: {OUTPUT_CSV}")
        return

    years     = list(range(PANEL_START_YEAR, PANEL_END_YEAR + 1))
    crosswalk = load_mca_crosswalk()

    all_panels: list[pd.DataFrame] = []
    muni_frames: list[pd.DataFrame] = []

    # ── Pre-2019: population-weighted DDD apportionment ───────────────────
    if PANEL_START_YEAR <= 2018:
        logging.info("Building pre-2019 ANATEL panel via DDD apportionment …")
        try:
            pre = build_pre2019_panel(crosswalk)
            if pre is not None and not pre.empty:
                all_panels.append(pre)
                logging.info(
                    f"Pre-2019 panel appended: {len(pre):,} rows "
                    f"({pre['year'].min()}–{pre['year'].max()})"
                )
        except Exception as exc:
            logging.error(f"Pre-2019 panel error: {exc}")

    # ── 2019+: municipality-level data ────────────────────────────────────
    post_years = [y for y in years if y >= 2019]
    logging.info(f"Downloading/parsing {len(post_years)} ANATEL annual files (serial) …")
    for year in post_years:
        try:
            df = _download_and_parse(year)
            if df is not None:
                agg = aggregate_to_mca_quarter(df, crosswalk, _muni_frames=muni_frames)
                del df      # free raw rows
                gc.collect()
                if agg is not None and not agg.empty:
                    all_panels.append(agg)
        except Exception as exc:
            logging.error(f"  Year {year} error: {exc}")

    if not all_panels:
        raise RuntimeError("No ANATEL data loaded.")

    # Save municipality-level intermediate for within-MCA σ computation
    if muni_frames:
        muni_all = pd.concat(muni_frames, ignore_index=True)
        muni_out = os.path.join(ANATEL_DIR, "anatel_muni_panel.csv")
        muni_all.to_csv(muni_out, index=False)
        logging.info(f"Saved municipality-level ANATEL to {muni_out}")
        del muni_all, muni_frames
        gc.collect()

    panel = pd.concat(all_panels, ignore_index=True)
    panel = merge_population_for_per100(panel)

    col_order = ["mca_code", "year", "quarter",
                 "connections_total", "connections_per100",
                 "connections_4g5g", "frac_4g5g",
                 "n_municipalities"]
    col_order = [c for c in col_order if c in panel.columns]
    panel = panel[col_order]
    panel.sort_values(["mca_code", "year", "quarter"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    import pyarrow as pa
    import pyarrow.csv as pa_csv
    pa_csv.write_csv(pa.Table.from_pandas(panel, preserve_index=False), OUTPUT_CSV)
    logging.info(f"Saved ANATEL panel to {OUTPUT_CSV}")

    print(
        f"\nANATEL mobile panel summary"
        f"\n  Rows:              {len(panel):,}"
        f"\n  MCAs:              {panel['mca_code'].nunique()}"
        f"\n  Years:             {panel['year'].min()} – {panel['year'].max()}"
        f"\n  connections_per100 non-null: {panel['connections_per100'].notna().sum():,}"
        f"\n  Output:            {OUTPUT_CSV}"
    )
    print(panel.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
