## scrape_cadunico.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-03-06
#
# Purpose: Download CadUnico (Cadastro Único para Programas Sociais) monthly
#          summary data by municipality and build an MCA × quarter panel of
#          low-income family registration counts.
#
#   Data source:
#     SAGI/Ministério do Desenvolvimento e Assistência Social (MDS)
#     "Quantidade de famílias cadastradas no Cadastro Único por município"
#     URL: https://aplicacoes.mds.gov.br/sagi/servicos/misocial/
#            tabela_csv_geral.php
#     (SAGI = Secretaria de Avaliação e Gestão da Informação)
#
#     This endpoint returns a CSV for a given competência (month) with
#     per-municipality family counts, stratified by per-capita income bracket.
#
#     Alternatively, flat files are published at Portal Dados Abertos:
#       https://dados.gov.br/dados/conjuntos-dados/
#         cadastro-unico-por-municipio-e-por-programa
#     as ZIP archives of semi-annual or annual panel CSVs.
#
#   API parameters (SAGI endpoint):
#     cod_ibge  : leave empty for all municipalities
#     periodo   : YYYY-MM (e.g. "2019-06")
#     ind       : indicator code  (see IND_FAMILIES_TOTAL below)
#
#   Key indicator codes (CadUnico SAGI):
#     1   Total families registered (all income bands)
#     13  Families with per-capita income ≤ R$89 (extreme poverty — Bolsa Família threshold 2023)
#     14  Families with per-capita income R$89–R$178 (poverty)
#
#   Output:
#     CadUnico/cadunico_mca_panel.csv
#       Columns: mca_code, year, quarter,
#                cadunico_families,          (total registered families in MCA)
#                cadunico_families_per1000,  (cadunico_families / pop * 1000)
#                cadunico_extreme_poverty,   (families with per-capita ≤ R$89)
#                cadunico_poverty,           (families per-capita R$89–R$178)
#                n_municipalities
#
# Notes:
#   • The SAGI endpoint responds slowly (~5 s per request for a full national
#     CSV at ~5500 municipalities).  Requests are serialized with a pause.
#   • For years before 2013, CadUnico data may be missing or of lower quality.
#   • Municipality codes returned by SAGI are 6-digit IBGE codes.
#   • Income thresholds changed several times (2009, 2014, 2023); extreme poverty
#     definition here uses the current threshold throughout for comparability.
###─────────────────────────────────────────────────────────────────────────────

import concurrent.futures
import io
import os
import time
import logging
import contextlib
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
BASE       = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CAD_DIR    = os.path.join(BASE, "CadUnico")
RAW_DIR    = os.path.join(CAD_DIR, "raw")
IBGE_DIR   = os.path.join(BASE, "IBGE")
MCA_CSV    = os.path.join(IBGE_DIR, "muni_mca_regions_2010_2024_panel.csv")
DEMO_CSV   = os.path.join(IBGE_DIR, "mca_demographics_panel.csv")
OUTPUT_CSV = os.path.join(CAD_DIR, "cadunico_mca_panel.csv")

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "scrape_cadunico",
        {
            "cadunico_dir": CAD_DIR,
            "raw_dir": RAW_DIR,
            "mca_csv": MCA_CSV,
            "demo_csv": DEMO_CSV,
            "output_csv": OUTPUT_CSV,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    CAD_DIR = _paths["cadunico_dir"]
    RAW_DIR = _paths["raw_dir"]
    MCA_CSV = _paths["mca_csv"]
    DEMO_CSV = _paths["demo_csv"]
    OUTPUT_CSV = _paths["output_csv"]

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CAD_DIR, exist_ok=True)

PANEL_START_YEAR = 2013
PANEL_END_YEAR   = 2024

# SAGI Solr API — CadUnico municipality-level monthly aggregates
SAGI_SOLR_URL = "https://aplicacoes.mds.gov.br/sagi/servicos/misocial/"

# Fields to request from the Solr index
SOLR_FL = (
    "codigo_ibge,"
    "cadun_qtd_familias_cadastradas_i,"
    "cadun_qtde_fam_sit_extrema_pobreza_s,"
    "cadun_qtde_fam_sit_pobreza_s"
)

HEADERS = {
    "User-Agent": "research/pedro.feijodemoraes@yale.edu",
    "Referer":    "https://aplicacoes.mds.gov.br/sagi/",
    "Accept":     "application/json, */*",
}

# Months to fetch: one per quarter (last month = March, June, Sep, Dec)
FETCH_MONTHS = [3, 6, 9, 12]


## ─────────────────────────────────────────────────────────────────────────────
## 2) FETCH ONE PERIOD FROM SAGI
## ─────────────────────────────────────────────────────────────────────────────

def fetch_sagi_period(year: int, month: int,
                      retries: int = 3, pause: float = 5.0) -> pd.DataFrame | None:
    """
    Fetch CadUnico municipality data for one (year, month) from SAGI Solr.
    Returns DataFrame: cod_ibge6, year, month, quarter,
                       families_total, families_extreme, families_poverty.
    """
    period_str = f"{year}{month:02d}"
    dest_csv   = os.path.join(RAW_DIR, f"cadunico_{period_str}.csv")

    if os.path.exists(dest_csv):
        with contextlib.suppress(Exception):
            df = pd.read_csv(dest_csv, dtype=str, encoding="utf-8")
            logging.info(f"  {year}-{month:02d}: loaded from cache ({len(df):,} rows)")
            return _coerce_solr_types(df, year, month)

    params = {
        "q":    f"tipo_s:mes_mu AND anomes_s:{period_str}",
        "rows": 6000,
        "wt":   "json",
        "fl":   SOLR_FL,
    }

    for attempt in range(retries):
        try:
            resp = requests.get(SAGI_SOLR_URL, params=params, headers=HEADERS,
                                timeout=90)
            if resp.status_code in {429, 503}:
                wait = (2 ** attempt) * pause
                logging.warning(
                    f"HTTP {resp.status_code} for {year}-{month:02d}; "
                    f"waiting {wait:.0f}s …"
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == retries - 1:
                logging.error(f"Failed {year}-{month:02d}: {exc}")
                return None
            time.sleep((2 ** attempt) * pause)
    else:
        return None

    try:
        body  = resp.json()
        docs  = body.get("response", {}).get("docs", [])
        found = body.get("response", {}).get("numFound", 0)
    except Exception as exc:
        logging.error(f"JSON parse failed for {year}-{month:02d}: {exc}")
        return None

    if not docs:
        logging.warning(f"  {year}-{month:02d}: numFound={found}, no docs returned.")
        return None

    logging.info(f"  {year}-{month:02d}: {found} municipalities from Solr")
    df = pd.DataFrame(docs)

    # Cache for future runs
    with contextlib.suppress(Exception):
        df.to_csv(dest_csv, index=False, encoding="utf-8")

    return _coerce_solr_types(df, year, month)


def _coerce_solr_types(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame | None:
    """Convert Solr response fields to typed DataFrame for one period."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename = {
        "codigo_ibge":                          "cod_ibge6",
        "cadun_qtd_familias_cadastradas_i":     "families_total",
        "cadun_qtde_fam_sit_extrema_pobreza_s": "families_extreme",
        "cadun_qtde_fam_sit_pobreza_s":         "families_poverty",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if not {"cod_ibge6", "families_total"}.issubset(df.columns):
        logging.error(
            f"  {year}-{month:02d}: missing Solr fields. "
            f"Columns: {list(df.columns)[:10]}"
        )
        return None

    result = pd.DataFrame()
    result["cod_ibge6"]       = pd.to_numeric(df["cod_ibge6"],       errors="coerce")
    result["families_total"]   = pd.to_numeric(df["families_total"],   errors="coerce")
    result["families_extreme"] = pd.to_numeric(
        df.get("families_extreme", pd.Series(dtype=str)), errors="coerce"
    )
    result["families_poverty"] = pd.to_numeric(
        df.get("families_poverty", pd.Series(dtype=str)), errors="coerce"
    )
    result["year"]    = year
    result["month"]   = month
    result["quarter"] = (month - 1) // 3 + 1

    result = result.dropna(subset=["cod_ibge6", "families_total"])
    result["cod_ibge6"] = result["cod_ibge6"].astype(int)
    return result


## ─────────────────────────────────────────────────────────────────────────────
## 3) DOWNLOAD ALL PERIODS
## ─────────────────────────────────────────────────────────────────────────────

def download_all_periods() -> pd.DataFrame:
    """
    Fetch CadUnico data for all (year, month) periods via SAGI Solr.
    Each request returns all three family counts in one JSON call.
    Uses max_workers=2 to stay polite with the SAGI server.
    Returns a single long DataFrame:
      cod_ibge6 (int), year (int), month (int), quarter (int),
      families_total (float), families_extreme (float), families_poverty (float)
    """
    fetch_periods = [
        (y, m)
        for y in range(PANEL_START_YEAR, PANEL_END_YEAR + 1)
        for m in FETCH_MONTHS
        if y != PANEL_END_YEAR or m <= 9
    ]

    logging.info(
        f"Fetching {len(fetch_periods)} SAGI Solr CadUnico requests "
        f"(max 2 concurrent) …"
    )

    all_rows: list[pd.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future_map = {
            pool.submit(fetch_sagi_period, y, m): (y, m)
            for y, m in fetch_periods
        }
        for future in concurrent.futures.as_completed(future_map):
            y, m = future_map[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    all_rows.append(df)
                    logging.info(f"  {y}-{m:02d}: {len(df):,} municipalities")
                else:
                    logging.warning(f"  {y}-{m:02d}: no data returned")
            except Exception as exc:
                logging.error(f"  SAGI {y}-{m:02d} error: {exc}")

    if not all_rows:
        raise RuntimeError("No CadUnico data downloaded.")

    return pd.concat(all_rows, ignore_index=True)


## ─────────────────────────────────────────────────────────────────────────────
## 4) AGGREGATE TO MCA × QUARTER
## ─────────────────────────────────────────────────────────────────────────────

def load_mca_crosswalk() -> pd.DataFrame:
    """Load MCA crosswalk; normalise municipality code to 6 digits."""
    mca = pd.read_csv(MCA_CSV,
                      usecols=["municipality_code", "mca_code", "year"],
                      dtype={"municipality_code": int, "mca_code": str, "year": int})
    # IBGE 7-digit → 6-digit (drop check digit) for match with SAGI data
    mca["cod_ibge6"] = mca["municipality_code"] // 10
    return mca[["cod_ibge6", "mca_code", "year"]]


def aggregate_to_mca_quarter(cad: pd.DataFrame,
                              crosswalk: pd.DataFrame) -> pd.DataFrame:
    cad = cad.merge(crosswalk, on=["cod_ibge6", "year"], how="left")
    n_miss = cad["mca_code"].isna().sum()
    if n_miss > 0:
        logging.warning(f"{n_miss:,} CadUnico rows unmatched to MCA — dropped.")
    cad = cad.dropna(subset=["mca_code"])

    agg = (
        cad.groupby(["mca_code", "year", "quarter"])
           .agg(
               cadunico_families       = ("families_total",   "sum"),
               cadunico_extreme_poverty = ("families_extreme", "sum"),
               cadunico_poverty        = ("families_poverty",  "sum"),
               n_municipalities        = ("cod_ibge6",         "nunique"),
           )
           .reset_index()
    )
    return agg


def merge_population(panel: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(DEMO_CSV):
        logging.warning("Demographics panel not found; cadunico_families_per1000 will be NaN.")
        panel["cadunico_families_per1000"] = np.nan
        return panel

    demo = pd.read_csv(DEMO_CSV,
                       usecols=["mca_code", "year", "pop_total"],
                       dtype={"mca_code": str, "year": int})
    panel = panel.merge(demo, on=["mca_code", "year"], how="left")
    panel["cadunico_families_per1000"] = np.where(
        panel["pop_total"] > 0,
        panel["cadunico_families"] / (panel["pop_total"] / 1000),
        np.nan,
    )
    panel.drop(columns=["pop_total"], inplace=True)
    return panel


## ─────────────────────────────────────────────────────────────────────────────
## 5) MAIN
## ─────────────────────────────────────────────────────────────────────────────

def main():
    # Early exit: if output already exists, skip the full rebuild
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        print(f"Output already exists, skipping: {OUTPUT_CSV}")
        logging.info(f"Output already exists -- skipping rebuild: {OUTPUT_CSV}")
        return

    crosswalk = load_mca_crosswalk()
    cad_raw   = download_all_periods()
    logging.info(f"Total CadUnico raw rows: {len(cad_raw):,}")

    panel = aggregate_to_mca_quarter(cad_raw, crosswalk)
    panel = merge_population(panel)

    col_order = [
        "mca_code", "year", "quarter",
        "cadunico_families", "cadunico_families_per1000",
        "cadunico_extreme_poverty", "cadunico_poverty",
        "n_municipalities",
    ]
    panel = panel[[c for c in col_order if c in panel.columns]]
    panel.sort_values(["mca_code", "year", "quarter"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    panel.to_csv(OUTPUT_CSV, index=False, encoding="latin-1")
    logging.info(f"Saved CadUnico panel to {OUTPUT_CSV}")

    print(
        f"\nCadUnico panel summary"
        f"\n  Rows:                           {len(panel):,}"
        f"\n  MCAs:                           {panel['mca_code'].nunique()}"
        f"\n  Years:                          {panel['year'].min()} – {panel['year'].max()}"
        f"\n  cadunico_families_per1000 non-null: {panel['cadunico_families_per1000'].notna().sum():,}"
        f"\n  Output:                         {OUTPUT_CSV}"
    )
    print(panel.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
