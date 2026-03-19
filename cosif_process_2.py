## cosif_process_2.py
# Author: Pedro Feijo de Moraes
# Last edited: 2026-02-26
#
# Objective: Post-process the monthly CNPJ-level COSIF output from
#            cosif_process_1.py into a quarterly CNPJ-level panel of
#            implicit funding rates -- ready for the INSTITUTION-LEVEL
#            panel (egan_panel_build_2.py).
#
# Why a separate script?
#   cosif_process_1.py outputs monthly CNPJ-level data (one CSV per
#   taxonomy group).  For the conglomerate pipeline (egan_panel_build.py),
#   the aggregation to conglomerates AND the monthly-to-quarterly
#   collapsing happen together inside egan_panel_build.py.
#
#   For the institution-level pipeline, we skip conglomerate aggregation
#   but still need quarterly collapsing.  This script handles that step,
#   producing a single `cosif_quarterly_institution.csv` file.
#
# Input:  BCB/Egan_et_al_2025_Rep/processed/COSIF_PROCESSED/custos_implicitos_*.csv
# Output: BCB/Egan_et_al_2025_Rep/processed/COSIF_PROCESSED/cosif_quarterly_institution.csv
#
# Prerequisites:
#   Run cosif_process_1.py first.
## ---------------------------------------------------------------------------

## 1) Load necessary pacjages and define paths:

# Packages:
import pandas as pd
import numpy as np
import logging
import os
import glob
from logging.handlers import RotatingFileHandler

# Paths:
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BCB_PATH   = os.path.join(PARENT_DIR, "BCB")

EGAN_PATH      = os.path.join(BCB_PATH, "Egan_et_al_2025_Rep")
PROCESSED_PATH = os.path.join(EGAN_PATH, "processed")
COSIF_PROCESSED_PATH = os.path.join(PROCESSED_PATH, "COSIF_PROCESSED")

# Logging
_log_file = os.path.join(SCRIPT_DIR, "cosif_process_2.log")
_handler  = RotatingFileHandler(
    _log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
logging.basicConfig(
    handlers=[_handler],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Taxonomy types that are pre-aggregated at the conglomerate level.
# Their "CNPJ" field is NOT a real institution CNPJ -- exclude them.
AGGREGATE_TAXONOMIES = {
    "CONGLOMERADO_PRUDENCIAL",
    "CONGLOMERADO_FINANCEIRO",
}

## 2) User-defined functions:

def load_cosif_monthly(cosif_dir: str = COSIF_PROCESSED_PATH) -> pd.DataFrame:
    """
    Load all monthly CNPJ-level custos_implicitos_*.csv files produced by
    cosif_process_1.py.  Exclude pre-aggregated taxonomy types.

    Returns a DataFrame with columns:
        CNPJ, DATA_BASE, Estoque_Total, Estoque_Total_Lag,
        Despesa_Captacao_Marginal, Custo_Efetivo_Blended, cosif_taxonomy
    """
    pattern = os.path.join(cosif_dir, "custos_implicitos_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No COSIF processed files found in {cosif_dir}.\n"
            "Please run cosif_process_1.py first."
        )

    logging.info(f"Loading {len(files)} COSIF monthly files from {cosif_dir}")
    frames = []
    n_skipped = 0
    for fpath in files:
        taxonomy = (
            os.path.basename(fpath)
            .replace("custos_implicitos_", "")
            .replace(".csv", "")
        )
        if taxonomy.upper() in AGGREGATE_TAXONOMIES:
            logging.info(f"  Skipping aggregate taxonomy: {taxonomy}")
            n_skipped += 1
            continue
        try:
            # Only read the columns we actually need
            _needed = [
                "CNPJ", "DATA_BASE",
                "Estoque_Total", "Despesa_Captacao_Marginal",
                "Estoque_Prepago", "Desp_Prepago_Marginal",
            ]
            # usecols silently ignores missing columns (e.g. pre-2025 files
            # lack Estoque_Prepago / Desp_Prepago_Marginal).
            import contextlib
            with contextlib.suppress(ValueError):
                pass
            available = pd.read_csv(fpath, dtype={"CNPJ": str}, nrows=0).columns.tolist()
            usecols = [c for c in _needed if c in available]
            df = pd.read_csv(
                fpath, dtype={"CNPJ": str}, usecols=usecols,
            )
            # Ensure missing prepaid columns exist as NaN / 0
            for _col, _fill in [("Estoque_Prepago", np.nan),
                                 ("Desp_Prepago_Marginal", 0.0)]:
                if _col not in df.columns:
                    df[_col] = _fill
            frames.append(df)
            logging.info(f"  {taxonomy}: {len(df):,} rows")
        except Exception as e:
            logging.warning(f"  Error loading {fpath}: {e}")

    if not frames:
        raise ValueError("All COSIF files were empty or excluded.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["DATA_BASE"] = pd.to_datetime(df_all["DATA_BASE"], errors="coerce")
    df_all["CNPJ"] = df_all["CNPJ"].astype(str).str.strip().str.zfill(8)

    # Use categorical CNPJ for faster groupby
    df_all["CNPJ"] = df_all["CNPJ"].astype("category")

    logging.info(
        f"COSIF monthly loaded: {len(df_all):,} rows | "
        f"{df_all['CNPJ'].nunique()} institutions | "
        f"{n_skipped} aggregate taxonomy files excluded"
    )
    return df_all
def collapse_to_quarterly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse CNPJ x month data to CNPJ x quarter.

    - Estoque_Total (stock): quarter-end value (month in {3, 6, 9, 12})
    - Despesa_Captacao_Marginal (flow): sum of all months in the quarter
    - cosif_implicit_rate: |quarterly expense| / lagged total deposits (decimal)

    Parameters
    ----------
    df_monthly : DataFrame with CNPJ, DATA_BASE, Estoque_Total,
                 Despesa_Captacao_Marginal columns.

    Returns
    -------
    df_quarterly : DataFrame with CNPJ, AnoMes (YYYYMM quarter-end),
                   cosif_implicit_rate, Estoque_Total_Q, Desp_Captacao_Q.
    """
    df = df_monthly.copy()
    df["year"]    = df["DATA_BASE"].dt.year
    df["month"]   = df["DATA_BASE"].dt.month
    df["q_month"] = ((df["month"] - 1) // 3 + 1) * 3   # 3, 6, 9, 12
    df["AnoMes"]  = (df["year"] * 100 + df["q_month"]).astype("Int64")

    # ---- Quarter-end STOCK: keep only months 3, 6, 9, 12 ----
    df_stocks = (
        df.loc[
            df["month"].isin([3, 6, 9, 12]),
            ["CNPJ", "AnoMes", "Estoque_Total"],
        ]
        .copy()
        .rename(columns={"Estoque_Total": "Estoque_Total_Q"})
    )

    # ---- Quarterly FLOW: sum monthly expenses within each quarter ----
    df_flows = (
        df
        .groupby(["CNPJ", "AnoMes"])["Despesa_Captacao_Marginal"]
        .sum()
        .reset_index()
        .rename(columns={"Despesa_Captacao_Marginal": "Desp_Captacao_Q"})
    )

    # Merge stock with flow
    df_q = df_stocks.merge(df_flows, on=["CNPJ", "AnoMes"], how="left")

    # Lagged total deposits (one quarter earlier)
    df_q = df_q.sort_values(["CNPJ", "AnoMes"])
    df_q["Estoque_Total_Lag_Q"] = (
        df_q.groupby("CNPJ")["Estoque_Total_Q"].shift(1)
    )

    # Blended implicit rate = |quarterly expense| / lagged total deposits
    df_q["cosif_implicit_rate"] = (
        df_q["Desp_Captacao_Q"].abs() / df_q["Estoque_Total_Lag_Q"]
    )
    df_q.loc[
        ~np.isfinite(df_q["cosif_implicit_rate"]),
        "cosif_implicit_rate",
    ] = np.nan
    # Cap extreme outliers (> 50 % per quarter is implausible)
    df_q.loc[df_q["cosif_implicit_rate"] > 0.5, "cosif_implicit_rate"] = np.nan

    # ---- Prepaid rate (2025+ only, zero / NaN for earlier periods) ----
    # Quarter-end prepaid stock
    if "Estoque_Prepago" in df.columns:
        df_prepaid_stocks = (
            df.loc[
                df["month"].isin([3, 6, 9, 12]),
                ["CNPJ", "AnoMes", "Estoque_Prepago"],
            ]
            .copy()
            .rename(columns={"Estoque_Prepago": "Estoque_Prepago_Q"})
        )
        df_q = df_q.merge(df_prepaid_stocks, on=["CNPJ", "AnoMes"], how="left")
    else:
        df_q["Estoque_Prepago_Q"] = np.nan

    # Quarterly prepaid expense flow
    if "Desp_Prepago_Marginal" in df.columns:
        df_prepaid_flows = (
            df
            .groupby(["CNPJ", "AnoMes"])["Desp_Prepago_Marginal"]
            .sum()
            .reset_index()
            .rename(columns={"Desp_Prepago_Marginal": "Desp_Prepago_Q"})
        )
        df_q = df_q.merge(df_prepaid_flows, on=["CNPJ", "AnoMes"], how="left")
        df_q["Desp_Prepago_Q"] = df_q["Desp_Prepago_Q"].fillna(0.0)
    else:
        df_q["Desp_Prepago_Q"] = 0.0

    # Lagged prepaid stock
    df_q["Estoque_Prepago_Lag_Q"] = (
        df_q.groupby("CNPJ")["Estoque_Prepago_Q"].shift(1)
    )

    # Prepaid implicit rate = |prepaid expense| / lagged prepaid stock
    df_q["cosif_prepaid_rate"] = (
        df_q["Desp_Prepago_Q"].abs() / df_q["Estoque_Prepago_Lag_Q"]
    )
    df_q.loc[
        ~np.isfinite(df_q["cosif_prepaid_rate"]),
        "cosif_prepaid_rate",
    ] = np.nan
    df_q.loc[df_q["cosif_prepaid_rate"] > 0.5, "cosif_prepaid_rate"] = np.nan

    logging.info(
        f"Quarterly collapse: {len(df_q):,} obs | "
        f"{df_q['CNPJ'].nunique()} institutions | "
        f"{df_q['AnoMes'].nunique()} quarters | "
        f"cosif_implicit_rate coverage: "
        f"{df_q['cosif_implicit_rate'].notna().sum():,} | "
        f"cosif_prepaid_rate coverage: "
        f"{df_q['cosif_prepaid_rate'].notna().sum():,}"
    )
    return df_q

## 3) Main execution:

if __name__ == "__main__":
    import time
    t0 = time.perf_counter()

    logging.info("=" * 60)
    logging.info("cosif_process_2.py -- Quarterly institution-level COSIF")
    logging.info("=" * 60)
    print("cosif_process_2.py -- Collapsing monthly COSIF to quarterly ...")

    # 3.1) Load monthly CNPJ-level data from cosif_process_1
    df_monthly = load_cosif_monthly()

    # 3.2) Collapse to quarterly
    df_quarterly = collapse_to_quarterly(df_monthly)

    # 3.3) Save output (CSV + Parquet for fast downstream reads)
    out_csv = os.path.join(COSIF_PROCESSED_PATH, "cosif_quarterly_institution.csv")
    out_pq  = os.path.join(COSIF_PROCESSED_PATH, "cosif_quarterly_institution.parquet")
    df_quarterly.to_csv(out_csv, index=False)
    df_quarterly.to_parquet(out_pq, index=False, engine="pyarrow")
    logging.info(f"Saved to {out_csv} + {out_pq}")

    t1 = time.perf_counter()
    print(f"Done in {t1 - t0:.1f}s. Saved {len(df_quarterly):,} obs to:")
    print(f"  {out_csv}")
    print(f"  {out_pq}")
    print(f"  Institutions: {df_quarterly['CNPJ'].nunique()}")
    print(f"  Quarters:     {df_quarterly['AnoMes'].nunique()}")
    print(f"  Rate coverage: {df_quarterly['cosif_implicit_rate'].notna().sum():,}")
