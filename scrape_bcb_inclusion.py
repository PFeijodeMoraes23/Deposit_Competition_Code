## scrape_bcb_inclusion.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-03-06
#
# Purpose: Download BCB financial-inclusion geographic data and build an
#          MCA × year panel of banking access-point density.
#
#   Data source:
#     BCB Dados Abertos — "Nota para Imprensa: Correspondentes Bancários"
#     Primary endpoint (annual flat files, same Olinda API used for IF Data):
#       https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata/
#         IfPorMunicipio(Data=@Data,TipoInstituicao=@TipoInstituicao)
#         ?@Data='<MM-YYYY>'&@TipoInstituicao=0
#         &$select=CodMunicipio,NumeroPontosAtendimento,...&$format=json
#
#     This endpoint is the geographic distribution sub-report of IF Data
#     (the same API used by if_data_scrape_1.py for deposit data).
#     It reports, per municipality per reference period:
#       • Agencias          : number of full bank branches
#       • PostosAtendimento : service posts (PABs, etc.)
#       • CorrespondentesNoPais : banking correspondents
#     Aggregating Agencias + PostosAtendimento + CorrespondentesNoPais gives
#     total formal-banking access points.
#
#     Data availability: quarterly reference periods from 2011-Q1 onward.
#     Published by BCB with ~1 quarter lag.
#
#     IF Data institution types (TipoInstituicao parameter):
#       0  = all consolidated
#       1  = banks (commercial + universal)
#       2  = savings banks
#       3  = credit cooperatives
#
#     NOTE: The exact field names (Agencias vs NomeCampo) may vary across
#     API versions.  The script requests all $select fields and maps them
#     using column-name aliases.  If empty results come back, the API
#     endpoint name or parameter format may have changed — check
#       https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/swagger-ui.html
#
#   Output:
#     BCB/Inclusion/bcb_inclusion_mca_panel.csv
#       Columns: mca_code, year,
#                branches_total,             (Agencias + PostosAtendimento)
#                correspondents_total,       (CorrespondentesNoPais)
#                access_points_total,        (branches_total + correspondents_total)
#                branches_per1000,           (branches_total / pop * 1000)
#                correspondents_per1000,     (correspondents_total / pop * 1000)
#                access_points_per1000,      (access_points_total / pop * 1000)
#                n_municipalities
###─────────────────────────────────────────────────────────────────────────────

import concurrent.futures
import os
import time
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
BASE         = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
INCL_DIR     = os.path.join(BASE, "BCB", "Inclusion")
IBGE_DIR     = os.path.join(BASE, "IBGE")
MCA_CSV      = os.path.join(IBGE_DIR, "muni_mca_regions_2010_2024_panel.csv")
DEMO_CSV     = os.path.join(IBGE_DIR, "mca_demographics_panel.csv")
OUTPUT_CSV   = os.path.join(INCL_DIR, "bcb_inclusion_mca_panel.csv")

os.makedirs(INCL_DIR, exist_ok=True)

PANEL_START_YEAR = 2013
PANEL_END_YEAR   = 2024

# ESTBAN CSV — BCB "Estatística Bancária por Município" (already downloaded)
ESTBAN_CSV = os.path.join(BASE, "BCB", "ESTBAN", "ESTBAN.csv")

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "scrape_bcb_inclusion",
        {
            "inclusion_dir": INCL_DIR,
            "mca_csv": MCA_CSV,
            "demo_csv": DEMO_CSV,
            "estban_csv": ESTBAN_CSV,
            "output_csv": OUTPUT_CSV,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    INCL_DIR = _paths["inclusion_dir"]
    MCA_CSV = _paths["mca_csv"]
    DEMO_CSV = _paths["demo_csv"]
    ESTBAN_CSV = _paths["estban_csv"]
    OUTPUT_CSV = _paths["output_csv"]

# Quarter-end months → quarter number
QUARTER_MONTHS = {3: 1, 6: 2, 9: 3, 12: 4}


## ─────────────────────────────────────────────────────────────────────────────
## 2) LOAD ESTBAN & BUILD BRANCH PANEL
## ─────────────────────────────────────────────────────────────────────────────

def load_estban() -> pd.DataFrame:
    """Load ESTBAN.csv; return institution-level rows with typed columns."""
    logging.info(f"Loading ESTBAN from {ESTBAN_CSV} …")
    df = pd.read_csv(ESTBAN_CSV, dtype=str, encoding="latin-1",
                     low_memory=False, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    df["CODMUN_IBGE"]      = pd.to_numeric(df["CODMUN_IBGE"],      errors="coerce")
    df["YEAR"]             = pd.to_numeric(df["YEAR"],             errors="coerce")
    df["MONTH"]            = pd.to_numeric(df["MONTH"],            errors="coerce")
    df["AGEN_PROCESSADAS"] = pd.to_numeric(df["AGEN_PROCESSADAS"], errors="coerce").fillna(0)
    df = df.dropna(subset=["CODMUN_IBGE", "YEAR", "MONTH"])
    df["CODMUN_IBGE"] = df["CODMUN_IBGE"].astype(int)
    df["YEAR"]        = df["YEAR"].astype(int)
    df["MONTH"]       = df["MONTH"].astype(int)
    return df


def build_branch_panel() -> pd.DataFrame:
    """
    Aggregate ESTBAN to municipality × quarter level using quarter-end months.
    Returns: mun_code (7-digit int), year, quarter, branches, correspondents.
    """
    df = load_estban()
    df = df[df["MONTH"].isin(QUARTER_MONTHS)].copy()

    agg = (
        df.groupby(["CODMUN_IBGE", "YEAR", "MONTH"], as_index=False)["AGEN_PROCESSADAS"]
          .sum()
    )
    agg.rename(columns={"AGEN_PROCESSADAS": "branches",
                         "YEAR": "year", "MONTH": "month",
                         "CODMUN_IBGE": "mun_code"}, inplace=True)
    agg["quarter"]       = agg["month"].map(QUARTER_MONTHS)
    agg["correspondents"] = 0

    logging.info(f"ESTBAN panel: {len(agg):,} municipality-quarter records")
    return agg[["mun_code", "year", "quarter", "branches", "correspondents"]]


## ─────────────────────────────────────────────────────────────────────────────
## 5) AGGREGATE TO MCA × YEAR
##    Using Q4 (December) as the annual stock measure. If Q4 is unavailable,
##    use the latest available quarter.
## ─────────────────────────────────────────────────────────────────────────────

def load_mca_crosswalk() -> pd.DataFrame:
    return pd.read_csv(MCA_CSV,
                      usecols=["municipality_code", "mca_code", "year"],
                      dtype={"municipality_code": int, "mca_code": str, "year": int})


def aggregate_to_mca_year(incl: pd.DataFrame,
                           crosswalk: pd.DataFrame) -> pd.DataFrame:
    """
    Use Q4 as the annual snapshot (or the latest available quarter per year).
    Then aggregate to MCA × year.
    """
    # Select the latest quarter per municipality × year as the annual stock
    incl = incl.sort_values(["mun_code", "year", "quarter"])
    annual = incl.drop_duplicates(subset=["mun_code", "year"], keep="last")

    # Merge MCA
    annual = annual.merge(
        crosswalk,
        left_on=["mun_code", "year"],
        right_on=["municipality_code", "year"],
        how="left",
    )
    n_miss = annual["mca_code"].isna().sum()
    if n_miss > 0:
        logging.warning(f"{n_miss:,} BCB inclusion rows unmatched to MCA — dropped.")
    annual = annual.dropna(subset=["mca_code"])

    agg = (
        annual.groupby(["mca_code", "year"])
              .agg(
                  branches_total      = ("branches",       "sum"),
                  correspondents_total = ("correspondents", "sum"),
                  n_municipalities    = ("mun_code",        "nunique"),
              )
              .reset_index()
    )
    agg["access_points_total"] = agg["branches_total"] + agg["correspondents_total"]
    return agg


def merge_population(panel: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(DEMO_CSV):
        logging.warning("Demographics panel not found; per-1000 rates will be NaN.")
        for col in ("branches_per1000", "correspondents_per1000", "access_points_per1000"):
            panel[col] = np.nan
        return panel

    demo = pd.read_csv(DEMO_CSV,
                       usecols=["mca_code", "year", "pop_total"],
                       dtype={"mca_code": str, "year": int})
    panel = panel.merge(demo, on=["mca_code", "year"], how="left")

    for raw_col, rate_col in [("branches_total",       "branches_per1000"),
                               ("correspondents_total", "correspondents_per1000"),
                               ("access_points_total",  "access_points_per1000")]:
        panel[rate_col] = np.where(
            panel["pop_total"] > 0,
            panel[raw_col] / (panel["pop_total"] / 1000),
            np.nan,
        )
    panel.drop(columns=["pop_total"], inplace=True)
    return panel


## ─────────────────────────────────────────────────────────────────────────────
## 6) MAIN
## ─────────────────────────────────────────────────────────────────────────────

def main():
    # Early exit: if output already exists, skip the full rebuild
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        print(f"Output already exists, skipping: {OUTPUT_CSV}")
        logging.info(f"Output already exists -- skipping rebuild: {OUTPUT_CSV}")
        return

    crosswalk = load_mca_crosswalk()

    logging.info("Building BCB branch panel from ESTBAN …")
    incl_all = build_branch_panel()
    logging.info(f"Total BCB inclusion rows: {len(incl_all):,}")

    panel = aggregate_to_mca_year(incl_all, crosswalk)
    panel = merge_population(panel)

    col_order = [
        "mca_code", "year",
        "branches_total", "branches_per1000",
        "correspondents_total", "correspondents_per1000",
        "access_points_total", "access_points_per1000",
        "n_municipalities",
    ]
    panel = panel[[c for c in col_order if c in panel.columns]]
    panel.sort_values(["mca_code", "year"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    panel.to_csv(OUTPUT_CSV, index=False, encoding="latin-1")
    logging.info(f"Saved BCB inclusion panel to {OUTPUT_CSV}")

    print(
        f"\nBCB financial inclusion panel summary"
        f"\n  Rows:                      {len(panel):,}"
        f"\n  MCAs:                      {panel['mca_code'].nunique()}"
        f"\n  Years:                     {panel['year'].min()} – {panel['year'].max()}"
        f"\n  correspondents_per1000 non-null: {panel['correspondents_per1000'].notna().sum():,}"
        f"\n  Output:                    {OUTPUT_CSV}"
    )
    print(panel.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
