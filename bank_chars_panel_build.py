"""
bank_chars_panel_build.py
=========================
Builds the bank characteristics panel containing size and solvency variables
(total assets, equity, equity ratio, log total assets) from IF Data reports,
and market segment + IP subsidiary indicators from IF Data List files.

This script processes:
  - IF_DATA_type_1_report_1.csv (for financial variables like Total Assets and Equity)

The resulting panel is saved to Panel/bank_chars_panel.csv.

Key design note:
  The CodConglomeradoPrudencial column is read DIRECTLY from Report 1 (it is
  already present on every row).  This ensures codes are the same as those
  used by deposits_panel.csv, which is critical for the downstream merge in
  build_market_panel.py.  Only rows with a blank conglomerate code (rare)
  fall back to a 'CNPJ_<value>' sentinel.
"""

import os
import logging
import numpy as np
import pandas as pd

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------------------------------------------------------
# 1) PATHS & CONSTANTS
# -----------------------------------------------------------------------------
BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

IF_AGG_DIR  = os.path.join(BASE, "BCB", "IF Data", "Aggregated Data")
OUTPUT_DIR  = os.path.join(BASE, "BCB", "Egan_et_al_2025_Rep", "processed")

REPORT_1_CSV = os.path.join(IF_AGG_DIR, "IF_DATA_type_1_report_1.csv")
OUT_CSV      = os.path.join(OUTPUT_DIR, "bank_chars_panel.csv")

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "bank_chars_panel_build",
        {
            "if_agg_dir": IF_AGG_DIR,
            "output_dir": OUTPUT_DIR,
            "report_1_csv": REPORT_1_CSV,
            "out_csv": OUT_CSV,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    IF_AGG_DIR = _paths["if_agg_dir"]
    OUTPUT_DIR = _paths["output_dir"]
    REPORT_1_CSV = _paths["report_1_csv"]
    OUT_CSV = _paths["out_csv"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 2) PANEL BUILD
# -----------------------------------------------------------------------------
def build_panel() -> pd.DataFrame:
    """
    Process IF Data Report 1 to build the bank characteristics panel.

    Conglomerate codes are taken directly from the CodConglomeradoPrudencial
    column that already exists in Report 1.  This is the same coding scheme
    used by the deposits panel, ensuring a clean merge in build_market_panel.py.
    """
    # --- 2.1) Load Report 1 --------------------------------------------------
    if not os.path.exists(REPORT_1_CSV):
        logging.error(f"IF Data report 1 not found: {REPORT_1_CSV}")
        return pd.DataFrame()

    logging.info("Loading IF Data Report 1 ...")
    df = pd.read_csv(REPORT_1_CSV, encoding="latin1", low_memory=False)

    if "CodConglomeradoPrudencial" not in df.columns:
        logging.error("CodConglomeradoPrudencial column missing from Report 1.")
        return pd.DataFrame()

    # Fill any blank conglomerate codes with a CNPJ-based fallback sentinel
    mask_no_cong = (
        df["CodConglomeradoPrudencial"].isna() |
        (df["CodConglomeradoPrudencial"].astype(str).str.strip() == "")
    )
    if mask_no_cong.any():
        df.loc[mask_no_cong, "CodConglomeradoPrudencial"] = (
            "CNPJ_" + df.loc[mask_no_cong, "CNPJ"].astype(str)
        )

    # --- 2.2) Time variables -------------------------------------------------
    df["Month"] = pd.to_numeric(df.get("Month"), errors="coerce")
    df["Year"]  = pd.to_numeric(df.get("Year"),  errors="coerce")
    df = df.dropna(subset=["Month", "Year"]).copy()

    df["Quarter"] = df["Month"].map({3: 1, 6: 2, 9: 3, 12: 4})
    df = df.dropna(subset=["Quarter"]).copy()
    df["Quarter"] = df["Quarter"].astype(int)
    df["Year"]    = df["Year"].astype(int)

    # --- 2.3) Segment and IP status ------------------------------------------
    if "SegmentoTb" in df.columns:
        df["is_ip"] = (
            df["SegmentoTb"].astype(str).str.contains("Institu", case=False, na=False) &
            df["SegmentoTb"].astype(str).str.contains("Pagamento", case=False, na=False)
        )
    else:
        df["is_ip"] = False

    if "Sr" in df.columns:
        df["segment_raw"] = df["Sr"].astype(str).str.strip().str.upper()
        df["segment_raw"] = df["segment_raw"].where(
            df["segment_raw"].isin(["S1", "S2", "S3", "S4", "S5"]), np.nan
        )
    else:
        df["segment_raw"] = np.nan

    cat_agg = (
        df.groupby(["CodConglomeradoPrudencial", "Year", "Quarter"])
        .agg(has_ip=("is_ip", "max"), segment=("segment_raw", "first"))
        .reset_index()
    )

    # --- 2.4) Financial variables (accounts 78182 = total assets, 78186 = equity) ---
    df["Conta"] = pd.to_numeric(df.get("NumeroConta"), errors="coerce").astype("Int64")
    df["Value"] = pd.to_numeric(df.get("Value"),       errors="coerce")

    df_fin = df[df["Conta"].isin([78182, 78186])].copy()
    df_fin["prop_name"] = df_fin["Conta"].map({78182: "total_assets", 78186: "equity"})

    pivot = (
        df_fin.groupby(["CodConglomeradoPrudencial", "Year", "Quarter", "prop_name"])["Value"]
        .sum().unstack("prop_name").reset_index()
    )
    pivot.columns.name = None

    for col in ["total_assets", "equity"]:
        if col not in pivot.columns:
            pivot[col] = np.nan

    # Merge category info
    pivot = pivot.merge(cat_agg, on=["CodConglomeradoPrudencial", "Year", "Quarter"], how="right")

    # Derived metrics
    pivot["equity_ratio"]     = pivot["equity"] / pivot["total_assets"]
    pivot["log_total_assets"] = np.log(pivot["total_assets"].clip(lower=1))

    # --- 2.5) Lag the financial variables (use *previous* quarter's values) ---
    pivot = pivot.sort_values(["CodConglomeradoPrudencial", "Year", "Quarter"])
    for col in ["total_assets", "equity", "equity_ratio", "log_total_assets"]:
        pivot[col] = pivot.groupby("CodConglomeradoPrudencial")[col].shift(1)

    # Segment dummies
    for s in ["S2", "S3", "S4", "S5"]:
        pivot[f"seg_{s}"] = (pivot["segment"] == s).astype(int)

    pivot.rename(columns={"Year": "year", "Quarter": "quarter"}, inplace=True)
    pivot.sort_values(["CodConglomeradoPrudencial", "year", "quarter"], inplace=True)

    logging.info(
        f"Bank chars panel: {len(pivot):,} rows | "
        f"{pivot['CodConglomeradoPrudencial'].nunique()} conglomerates | "
        f"total_assets non-null (lagged): {pivot['total_assets'].notna().sum():,}"
    )
    return pivot


# -----------------------------------------------------------------------------
# 3) MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    panel = build_panel()
    if not panel.empty:
        panel.to_csv(OUT_CSV, index=False)
        logging.info(f"Saved bank characteristics panel to {OUT_CSV}")
    else:
        logging.warning("Build returned an empty panel.")


if __name__ == "__main__":
    main()
