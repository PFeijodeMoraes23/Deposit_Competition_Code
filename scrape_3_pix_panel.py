## scrape_pix_panel.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-03-06
#
# Purpose: Process the BCB monthly PIX transaction CSV files already
#          downloaded to BCB/PIX/ into an MCA × quarter panel that can
#          be merged with the deposit competition model data.
#
#   Inputs:
#     1. BCB/PIX/TransacoesPixMunicipio_YYYYMM.csv  (Nov 2020 – Aug 2024)
#          Key columns:
#            AnoMes            : YYYYMM integer (e.g. 202011)
#            Municipio_Ibge    : 7-digit IBGE municipality code
#            QT_PES_PagadorPF  : unique PF individuals making PIX payments
#            QT_PES_PagadorPJ  : unique PJ entities making PIX payments
#            QT_PES_RecebedorPF: unique PF individuals receiving PIX
#            QT_PagadorPF      : total PF PIX payment transactions
#            VL_PagadorPF      : total PF PIX payment value (R$)
#     2. IBGE/mca_demographics_panel.csv
#          For population denominators (pop_total per mca_code × year).
#     3. IBGE/muni_mca_regions_2010_2024_panel.csv
#          Municipality → MCA crosswalk (municipality_code, mca_code, year).
#
#   Output:
#     BCB/PIX/pix_mca_panel.csv
#       Columns: mca_code, year, quarter,
#                pix_active,             (1 once PIX launched, i.e. 2020-Q4 onward)
#                pix_users_pf,           (sum QT_PES_PagadorPF across MCA municipalities)
#                pix_users_pf_per1000,   (pix_users_pf / (pop_total/1000))
#                pix_txns_pf,            (sum QT_PagadorPF)
#                pix_value_pf_r1000,     (sum VL_PagadorPF / 1000, R$ thousands)
#                pix_users_pj,           (sum QT_PES_PagadorPJ)
#                n_municipalities        (municipalities with PIX data in this MCA × quarter)
#
# Notes:
#   • BCB raw files use ',' as BOTH field separator AND decimal mark inside
#     quoted numeric strings, e.g.: "1.234,56" in a comma-delimited file.
#     Strategy: read with sep=';' first; if that yields only 1 column, re-try
#     with sep=',' and parse numeric columns manually (strip dots, replace comma
#     with dot).
#   • PIX launched nationally in November 2020 (AnoMes = 202011).
#     Quarters before 2020-Q4 receive pix_active=0 and all PIX quantities=0.
#   • Municipality codes in BCB files have 7 digits. The MCA crosswalk also
#     uses 7-digit codes — no truncation needed.
#   • Because the BCB files only cover Nov 2020 onward, pre-PIX quarters (every
#     MCA × quarter before 2020-Q4) are filled with zeros and flagged
#     pix_active=0 so that the sleepiness regression can use a single balanced
#     panel from 2013-Q1 to 2024-Q3.
###─────────────────────────────────────────────────────────────────────────────

import concurrent.futures
import os
import glob
import logging
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

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
PIX_DIR    = os.path.join(BASE, "BCB", "PIX")
IBGE_DIR   = os.path.join(BASE, "IBGE")
MCA_CSV    = os.path.join(IBGE_DIR, "muni_mca_regions_2010_2024_panel.csv")
DEMO_CSV   = os.path.join(IBGE_DIR, "mca_demographics_panel.csv")
OUTPUT_CSV = os.path.join(PIX_DIR, "pix_mca_panel.csv")

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "scrape_pix_panel",
        {
            "pix_dir": PIX_DIR,
            "mca_csv": MCA_CSV,
            "demo_csv": DEMO_CSV,
            "output_csv": OUTPUT_CSV,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    PIX_DIR = _paths["pix_dir"]
    MCA_CSV = _paths["mca_csv"]
    DEMO_CSV = _paths["demo_csv"]
    OUTPUT_CSV = _paths["output_csv"]

# PIX launch: November 2020
PIX_LAUNCH_YYYYMM = 202011

# Panel coverage (year × quarter)
PANEL_START_YEAR = 2013
PANEL_END_YEAR   = 2024

## Numeric columns that need decimal-comma cleaning
VALUE_COLS = [
    "VL_PagadorPF",    "VL_PagadorPJ",
    "VL_RecebedorPF",  "VL_RecebedorPJ",
]
COUNT_COLS = [
    "QT_PES_PagadorPF", "QT_PES_PagadorPJ",
    "QT_PES_RecebedorPF", "QT_PES_RecebedorPJ",
    "QT_PagadorPF",     "QT_PagadorPJ",
    "QT_RecebedorPF",   "QT_RecebedorPJ",
]


## ─────────────────────────────────────────────────────────────────────────────
## 2) READ & PARSE BCB PIX FILES
## ─────────────────────────────────────────────────────────────────────────────

def _clean_numeric(series: pd.Series) -> pd.Series:
    """
    Convert Brazilian-formatted numeric strings like '1.234.567,89' to floats.
    Handles both already-numeric values and string values.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return (
        series.astype(str)
              .str.replace(".", "", regex=False)   # remove thousands separator
              .str.replace(",", ".", regex=False)  # decimal comma → dot
              .str.strip()
              .replace("", np.nan)
              .pipe(pd.to_numeric, errors="coerce")
    )


def read_pix_file(filepath: str) -> pd.DataFrame | None:
    """
    Read a single TransacoesPixMunicipio_YYYYMM.csv file.
    Tries semicolon separator first, falls back to comma separator.
    Returns a cleaned DataFrame or None if the file cannot be parsed.
    """
    for sep in (";", ","):
        try:
            df = pd.read_csv(filepath, sep=sep, encoding="latin-1", dtype=str,
                             low_memory=False)
            # A successful parse has multiple columns
            if len(df.columns) < 5:
                continue

            # Normalize column names (strip whitespace)
            df.columns = [c.strip() for c in df.columns]

            # Require the key columns to be present
            required = {"AnoMes", "Municipio_Ibge", "QT_PES_PagadorPF"}
            if not required.issubset(df.columns):
                logging.warning(f"Missing required columns in {filepath} (sep={sep!r}); "
                                 f"columns found: {list(df.columns)[:8]}")
                continue

            # Parse numeric columns
            for col in COUNT_COLS + VALUE_COLS:
                if col in df.columns:
                    df[col] = _clean_numeric(df[col])

            df["AnoMes"]       = pd.to_numeric(df["AnoMes"],       errors="coerce").astype("Int64")
            df["Municipio_Ibge"] = pd.to_numeric(df["Municipio_Ibge"], errors="coerce").astype("Int64")

            df = df.dropna(subset=["AnoMes", "Municipio_Ibge"])
            return df

        except Exception as exc:
            logging.debug(f"read_pix_file sep={sep!r} failed for {filepath}: {exc}")

    logging.error(f"Could not parse {filepath} with any separator.")
    return None


def download_missing_pix_files():
    import urllib.request
    import time
    import ssl

    start = pd.Period("2020-11", "M")
    end = pd.Period(pd.Timestamp("today").replace(day=1) + pd.DateOffset(months=1), "M")
    
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    
    for p in pd.period_range(start, end):
        yyyymm = p.strftime("%Y%m")
        filepath = os.path.join(PIX_DIR, f"TransacoesPixMunicipio_{yyyymm}.csv")
        if os.path.exists(filepath):
            continue
            
        url = f"https://olinda.bcb.gov.br/olinda/servico/Pix_DadosAbertos/versao/v1/odata/TransacoesPixPorMunicipio(DataBase=@DataBase)?@DataBase=%27{yyyymm}%27&%24filter=AnoMes%20eq%20{yyyymm}&%24top=100000&%24format=text/csv"
        
        logging.info(f"Downloading PIX municipality data for {yyyymm}...")
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=45, context=context) as response:
                    content = response.read().decode('utf-8', errors='replace')
                if len(content.splitlines()) < 2:
                    logging.info(f"No PIX data found for {yyyymm} (reached end?).")
                    break
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                logging.info(f"Successfully downloaded {yyyymm}")
                break
            except Exception as e:
                if attempt == 3:
                    logging.error(f"Failed to download PIX data for {yyyymm} after 4 attempts: {e}")
                else:
                    time.sleep(2)

def load_all_pix_files() -> pd.DataFrame:
    """
    Load and concatenate all TransacoesPixMunicipio_*.csv files in PIX_DIR.
    Returns a single long DataFrame.
    """
    pattern = os.path.join(PIX_DIR, "TransacoesPixMunicipio_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No TransacoesPixMunicipio_*.csv files found in {PIX_DIR}."
        )

    logging.info(f"Found {len(files)} PIX municipality files — loading …")
    frames = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {pool.submit(read_pix_file, fp): fp for fp in files}
        for future in concurrent.futures.as_completed(future_map):
            fp = future_map[future]
            try:
                df = future.result()
                if df is not None:
                    frames.append(df)
                else:
                    logging.warning(f"Skipping unreadable file: {fp}")
            except Exception as exc:
                logging.warning(f"Error reading {fp}: {exc}")

    if not frames:
        raise RuntimeError("All PIX files failed to parse.")

    combined = pd.concat(frames, ignore_index=True)
    logging.info(f"Combined PIX data: {len(combined):,} rows  "
                 f"AnoMes range: {combined['AnoMes'].min()} – {combined['AnoMes'].max()}")
    return combined


## ─────────────────────────────────────────────────────────────────────────────
## 3) ADD YEAR / QUARTER COLUMNS
## ─────────────────────────────────────────────────────────────────────────────

def add_year_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'year' and 'quarter' columns derived from the AnoMes integer."""
    df = df.copy()
    df["year"]    = (df["AnoMes"] // 100).astype(int)
    df["month"]   = (df["AnoMes"] % 100).astype(int)
    df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)
    return df


## ─────────────────────────────────────────────────────────────────────────────
## 4) MAP MUNICIPALITY → MCA
## ─────────────────────────────────────────────────────────────────────────────

def load_mca_crosswalk() -> pd.DataFrame:
    """
    Load the MCA crosswalk. Returns DataFrame:
      municipality_code (int), mca_code (str), year (int)
    """
    return pd.read_csv(MCA_CSV,
                      usecols=["municipality_code", "mca_code", "year"],
                      dtype={"municipality_code": int, "mca_code": str, "year": int})


def merge_mca(pix: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """
    Merge MCA code onto PIX data using municipality_code × year.
    Municipalities not in the crosswalk for a given year are dropped with a warning.
    """
    pix = pix.copy()
    pix["municipality_code"] = pix["Municipio_Ibge"].astype(int)

    merged = pix.merge(crosswalk, on=["municipality_code", "year"], how="left")

    n_missing = merged["mca_code"].isna().sum()
    if n_missing > 0:
        logging.warning(
            f"{n_missing:,} PIX rows could not be matched to an MCA "
            f"(municipality not in crosswalk for that year) — dropped."
        )
    merged = merged.dropna(subset=["mca_code"])
    return merged


## ─────────────────────────────────────────────────────────────────────────────
## 5) AGGREGATE TO MCA × QUARTER
## ─────────────────────────────────────────────────────────────────────────────

def aggregate_to_mca_quarter(pix: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PIX data to MCA × year × quarter.
    All QT_* flow measures are summed; within-quarter deduplication of unique
    users is not possible from these files, so QT_PES_* sums are kept as upper
    bounds (acknowledged in model documentation).
    """
    agg = (
        pix.groupby(["mca_code", "year", "quarter"])
           .agg(
               pix_users_pf    = ("QT_PES_PagadorPF",  "sum"),
               pix_txns_pf     = ("QT_PagadorPF",       "sum"),
               pix_users_pj    = ("QT_PES_PagadorPJ",   "sum"),
               pix_value_pf_r1 = ("VL_PagadorPF",       "sum"),   # in R$
               n_municipalities = ("municipality_code",  "nunique"),
           )
           .reset_index()
    )

    # Convert R$ to R$ thousands for value column
    agg["pix_value_pf_r1000"] = agg["pix_value_pf_r1"] / 1000
    agg.drop(columns=["pix_value_pf_r1"], inplace=True)

    # PIX active flag: 1 from 2020-Q4 onward
    agg["pix_active"] = (
        (agg["year"] > 2020) | ((agg["year"] == 2020) & (agg["quarter"] == 4))
    ).astype(int)

    return agg


## ─────────────────────────────────────────────────────────────────────────────
## 6) BUILD COMPLETE BALANCED PANEL (incl. pre-PIX zeroes)
## ─────────────────────────────────────────────────────────────────────────────

def build_full_panel(pix_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Expand pix_agg to the full panel 2013-Q1 through 2024-Q3 for every MCA.
    Pre-PIX quarters get pix_active=0 and all quantity/value columns = 0.
    """
    # All MCA × year × quarter combinations in the target panel
    all_mcas = pix_agg["mca_code"].unique()
    quarters  = [(y, q)
                 for y in range(PANEL_START_YEAR, PANEL_END_YEAR + 1)
                 for q in range(1, 5)
                 if y != PANEL_END_YEAR or q != 4]

    rows = [{"mca_code": m, "year": y, "quarter": q}
            for m in all_mcas for y, q in quarters]
    skeleton = pd.DataFrame(rows)

    # Merge in observed PIX quantities
    merged = skeleton.merge(pix_agg, on=["mca_code", "year", "quarter"], how="left")

    # Fill quantity and value columns with 0 for quarters with no PIX data
    fill_cols = ["pix_users_pf", "pix_txns_pf", "pix_users_pj",
                 "pix_value_pf_r1000", "n_municipalities"]
    merged[fill_cols] = merged[fill_cols].fillna(0)

    # pix_active flag: 0 before 2020-Q4, 1 from 2020-Q4 onward
    merged["pix_active"] = (
        (merged["year"] > 2020) | ((merged["year"] == 2020) & (merged["quarter"] == 4))
    ).astype(int)

    return merged


## ─────────────────────────────────────────────────────────────────────────────
## 7) MERGE IN POPULATION & COMPUTE PER-1000 RATE
## ─────────────────────────────────────────────────────────────────────────────

def merge_population(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Merge MCA-level annual population from the demographics panel to compute
    pix_users_pf_per1000 = pix_users_pf / (pop_total / 1000).
    """
    if not os.path.exists(DEMO_CSV):
        logging.warning(
            f"Demographics panel not found at {DEMO_CSV}. "
            "Run ibge_demographics_panel.py first. "
            "pix_users_pf_per1000 will be NaN."
        )
        panel["pix_users_pf_per1000"] = np.nan
        return panel

    demo = pd.read_csv(DEMO_CSV,
                       usecols=["mca_code", "year", "pop_total"],
                       dtype={"mca_code": str, "year": int})

    panel = panel.merge(demo, on=["mca_code", "year"], how="left")

    panel["pix_users_pf_per1000"] = np.where(
        panel["pop_total"] > 0,
        panel["pix_users_pf"] / (panel["pop_total"] / 1000),
        np.nan,
    )
    panel.drop(columns=["pop_total"], inplace=True)
    return panel


## ─────────────────────────────────────────────────────────────────────────────
## 8) MAIN
## ─────────────────────────────────────────────────────────────────────────────

def main():
    # 0. Download any missing files from BCB Olinda directly
    download_missing_pix_files()

    # 1. Load all raw PIX municipality files
    pix_raw = load_all_pix_files()

    # 2. Add year / quarter columns
    pix_raw = add_year_quarter(pix_raw)

    # 3. Map municipalities to MCAs
    crosswalk = load_mca_crosswalk()
    pix_raw   = merge_mca(pix_raw, crosswalk)

    # 4. Aggregate to MCA × quarter
    pix_agg = aggregate_to_mca_quarter(pix_raw)
    logging.info(f"PIX MCA × quarter aggregation: {len(pix_agg):,} rows")

    # 5. Build full panel (fill 0s for pre-PIX quarters)
    panel = build_full_panel(pix_agg)

    # 6. Add per-1000 population rate
    panel = merge_population(panel)

    # 7. Final column ordering
    col_order = [
        "mca_code", "year", "quarter",
        "pix_active",
        "pix_users_pf", "pix_users_pf_per1000",
        "pix_txns_pf", "pix_value_pf_r1000",
        "pix_users_pj", "n_municipalities",
    ]
    col_order = [c for c in col_order if c in panel.columns]
    panel = panel[col_order]
    panel.sort_values(["mca_code", "year", "quarter"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    # 8. Save
    panel.to_csv(OUTPUT_CSV, index=False, encoding="latin-1")
    logging.info(f"Saved PIX panel to {OUTPUT_CSV}")

    print(
        f"\nPIX MCA panel summary"
        f"\n  Rows:                  {len(panel):,}"
        f"\n  MCAs:                  {panel['mca_code'].nunique()}"
        f"\n  Years:                 {panel['year'].min()} – {panel['year'].max()}"
        f"\n  pix_active == 1:       {panel['pix_active'].sum():,} rows"
        f"\n  pix_users_pf_per1000:  {panel['pix_users_pf_per1000'].notna().sum():,} non-null"
        f"\n  Output:                {OUTPUT_CSV}"
    )
    print(panel[panel["pix_active"] == 1].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
