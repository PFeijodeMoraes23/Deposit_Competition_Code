## build_loo_instruments.py
# Author: Pedro Feijo de Moraes
# Last edited: 2026-06-01
#
# Objective: Build leave-one-out (LOO) rival-characteristic instruments for
#            the Egan et al. (2025) deposit demand BLP model at the
#            INDIVIDUAL INSTITUTION (CNPJ) level.
#
# Motivation (BLP / Hausman IV):
#   In markets with differentiated products, a valid instrument for a firm's
#   price (deposit rate) is the sum of characteristics of rival firms in the
#   same market.  These "BLP instruments" exploit cost-side variation at rivals
#   while being excluded from the demand equation (Berry, Levinsohn & Pakes,
#   1995; Gandhi & Houde, 2023).
#
#   National-level LOO (this script):
#   ─────────────────────────────────
#   The institution panel (egan_panel_institution.csv) has no MCA granularity
#   (deposits are national totals per CNPJ).  LOO groups are therefore
#   computed nationally:
#       group = deposit_type × AnoMes
#   For each institution i in group g:
#       loo_x_i = Σ_{j≠i, j∈g} x_j  = total_sum_g(x) − x_i
#
#   This matches the existing Hausman IV design in estimation_2.py
#   (leave-one-out mean *spread* grouped by deposit_type × AnoMes).
#
#   Variables computed for each rival characteristic x:
#       loo_{x}           – LOO sum    of {x} (sum excluding own value)
#       loo_mean_{x}      – LOO mean   of {x} (sum / n_rivals)  [more stable]
#
#   Characteristics (all institution-level, lagged in source panel):
#     log_total_assets      – rival scale
#     equity_ratio          – rival capital structure
#     indice_basileia_lag   – rival regulatory capital (Basel ratio)
#     credit_assets_ratio_lag – rival asset-side ALM pressure (credit book)
#     lci_lca_ratio_lag     – rival wholesale / retail funding substitutability
#     wholesale_ratio_lag   – rival overall wholesale reliance
#     npl_provision_ratio_lag – rival credit quality / provisioning pressure
#
# Output columns (added to key CNPJ × deposit_type × AnoMes):
#   n_rivals_{dt}          – count of other institutions in same type × quarter
#   loo_log_assets         – Σ_rivals log_total_assets
#   loo_mean_log_assets    – mean_rivals log_total_assets
#   loo_equity_ratio       – Σ_rivals equity_ratio
#   loo_mean_equity_ratio
#   loo_indice_basileia    – Σ_rivals indice_basileia_lag
#   loo_mean_indice_basileia
#   loo_credit_assets      – Σ_rivals credit_assets_ratio_lag
#   loo_mean_credit_assets
#   loo_lci_lca_ratio      – Σ_rivals lci_lca_ratio_lag
#   loo_mean_lci_lca_ratio
#   loo_wholesale_ratio    – Σ_rivals wholesale_ratio_lag
#   loo_mean_wholesale_ratio
#   loo_npl_provision      – Σ_rivals npl_provision_ratio_lag
#   loo_mean_npl_provision
#
# Input:  BCB/Egan_et_al_2025_Rep/processed/PANEL_INTERMED/
#             egan_panel_institution.parquet  (preferred)
#             egan_panel_institution.csv      (fallback)
#
# Output: BCB/Egan_et_al_2025_Rep/processed/PANEL_INTERMED/
#             egan_panel_institution_loo.csv
#             egan_panel_institution_loo.parquet
#
# Usage: python build_loo_instruments.py
# ──────────────────────────────────────────────────────────────────────────────

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PARENT_DIR   = SCRIPT_DIR.parents[1]
BCB_PATH     = PARENT_DIR / "BCB"
PANEL_DIR    = BCB_PATH / "Egan_et_al_2025_Rep" / "processed" / "PANEL_INTERMED"

IN_PARQUET   = PANEL_DIR / "egan_panel_institution.parquet"
IN_CSV       = PANEL_DIR / "egan_panel_institution.csv"
OUT_CSV      = PANEL_DIR / "egan_panel_institution_loo.csv"
OUT_PARQUET  = PANEL_DIR / "egan_panel_institution_loo.parquet"

PANEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
_log_file = SCRIPT_DIR / "build_loo_instruments.log"
_handler  = RotatingFileHandler(
    str(_log_file), maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
)
_console  = logging.StreamHandler()
_console.setLevel(logging.INFO)
logging.basicConfig(
    handlers=[_handler, _console],
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── LOO specification ──────────────────────────────────────────────────────
# Each tuple: (source_column_in_panel, loo_output_prefix)
# The source columns are the instrument candidates from egan_panel_build_2.
LOO_VARS: list[tuple[str, str]] = [
    ("log_total_assets",         "loo_log_assets"),
    ("equity_ratio",             "loo_equity_ratio"),
    ("indice_basileia_lag",      "loo_indice_basileia"),
    ("credit_assets_ratio_lag",  "loo_credit_assets"),
    ("lci_lca_ratio_lag",        "loo_lci_lca_ratio"),
    ("wholesale_ratio_lag",      "loo_wholesale_ratio"),
    ("npl_provision_ratio_lag",  "loo_npl_provision"),
]

# LOO group dimension (mirrors Hausman IV grouping in estimation_2.py)
GROUP_COLS = ["deposit_type", "AnoMes"]

# Key columns for the output panel
KEY_COLS   = ["CNPJ", "deposit_type", "AnoMes"]


# ── Functions ──────────────────────────────────────────────────────────────

def load_panel() -> pd.DataFrame:
    """Load the institution panel (parquet preferred for speed)."""
    if IN_PARQUET.exists():
        logging.info(f"Loading institution panel from parquet: {IN_PARQUET.name}")
        df = pd.read_parquet(str(IN_PARQUET))
    elif IN_CSV.exists():
        logging.info(f"Loading institution panel from CSV: {IN_CSV.name}")
        df = pd.read_csv(str(IN_CSV), low_memory=False, dtype={"CNPJ": str})
    else:
        raise FileNotFoundError(
            f"Institution panel not found at:\n  {IN_PARQUET}\n  {IN_CSV}\n"
            "Run egan_panel_build_2.py first."
        )
    if "CNPJ" in df.columns:
        df["CNPJ"] = df["CNPJ"].astype(str).str.strip().str.zfill(8)
    logging.info(
        f"Panel loaded: {len(df):,} rows | "
        f"{df['CNPJ'].nunique():,} institutions | "
        f"AnoMes {df['AnoMes'].min()}–{df['AnoMes'].max()}"
    )
    return df


def compute_loo_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each characteristic in LOO_VARS, compute leave-one-out sum and mean
    within each (deposit_type × AnoMes) group.

    NaN-handling:
      - Use nansum / nanmean so that NaNs in one institution don't zero out
        others, BUT set loo = NaN when no rivals have a valid observation
        (i.e. all other values are NaN).
      - n_rivals counts non-NaN rivals (institutions with valid x_j).
    """
    out = df[KEY_COLS].copy()

    for src_col, loo_prefix in LOO_VARS:
        if src_col not in df.columns:
            logging.warning(
                f"  Source column '{src_col}' not in panel — "
                f"'{loo_prefix}' will be NaN."
            )
            out[loo_prefix]             = np.nan
            out[f"mean_{loo_prefix}"]   = np.nan
            continue

        x = df[src_col].copy()
        grp = df.groupby(GROUP_COLS, observed=True)

        # NaN-safe group totals and non-NaN counts
        g_sum   = grp[src_col].transform(lambda s: s.sum(min_count=1))
        g_count = grp[src_col].transform("count")   # non-NaN count

        # LOO sum: subtract own value (if own is NaN, subtract 0)
        own_fill = x.fillna(0.0)
        loo_sum  = g_sum - own_fill

        # When OWN is NaN: loo_sum already correct (own contributes 0)
        # Rival count: n non-NaN rivals = total_nonNaN − indicator(own is nonNaN)
        is_valid     = x.notna().astype(int)
        n_rivals     = g_count - is_valid           # non-NaN OTHER observations

        # If no rival has a valid value, set loo_sum to NaN
        loo_sum = loo_sum.where(n_rivals > 0, other=np.nan)

        # LOO mean
        loo_mean = loo_sum / n_rivals.where(n_rivals > 0, other=np.nan)

        out[loo_prefix]           = loo_sum.values
        out[f"mean_{loo_prefix}"] = loo_mean.values

        n_valid = out[loo_prefix].notna().sum()
        logging.info(
            f"  {loo_prefix}: {n_valid:,} non-null observations "
            f"(own column: {x.notna().sum():,} non-null)"
        )

    # Number of non-NaN rivals for log_total_assets (most populated characteristic)
    ref_col    = "log_total_assets"
    if ref_col in df.columns:
        g_count_ref  = df.groupby(GROUP_COLS, observed=True)[ref_col].transform("count")
        is_valid_ref = df[ref_col].notna().astype(int)
        out["n_rivals"] = (g_count_ref - is_valid_ref).astype("Int64")
    else:
        out["n_rivals"] = pd.NA

    logging.info(f"LOO instruments computed. Output shape: {out.shape}")
    return out


def save_output(df_loo: pd.DataFrame) -> None:
    """Save LOO panel as CSV and parquet."""
    df_loo.to_csv(str(OUT_CSV), index=False)
    logging.info(f"Saved CSV  -> {OUT_CSV}")
    try:
        df_loo.to_parquet(str(OUT_PARQUET), index=False, compression="snappy")
        logging.info(f"Saved parquet -> {OUT_PARQUET}")
    except Exception as e:
        logging.warning(f"Could not save parquet: {e}")


def print_summary(df_loo: pd.DataFrame) -> None:
    """Log a quick coverage table for all LOO columns."""
    logging.info("=" * 60)
    logging.info("LOO instrument coverage summary:")
    n_total = len(df_loo)
    for col in [c for c in df_loo.columns if c not in KEY_COLS]:
        n_valid = df_loo[col].notna().sum()
        pct     = 100.0 * n_valid / n_total if n_total > 0 else 0.0
        logging.info(f"  {col:<35s}: {n_valid:>8,} / {n_total:,} ({pct:5.1f}%)")
    logging.info("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    logging.info("=" * 60)
    logging.info("build_loo_instruments.py  START")
    logging.info("=" * 60)

    df       = load_panel()
    df_loo   = compute_loo_instruments(df)
    print_summary(df_loo)
    save_output(df_loo)

    logging.info("build_loo_instruments.py  DONE")


if __name__ == "__main__":
    main()
