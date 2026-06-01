"""
panel_7b_cosif_fees.py
=======================
Map COSIF service-fee ratios (from scrape_7b_cosif_service_fees.py) onto the
market panel produced by panel_6_market.py.

Steps
-----
1. Load the institution-level COSIF panel (133K rows, BANCOS+SOCIEDADES+COOPERATIVAS).
2. Map each institution's CNPJ to its CodConglomeradoPrudencial using IF Data.
3. Aggregate fee ratios to conglomerate × quarter (median across member institutions).
4. Produce cosif_fee_quarterly_conglomerate.csv in the same format as
   tarifas_fee_summary.csv so panel_6_market.py can load it via its existing
   merge_tarifas() pattern (or the new merge_cosif_fees() added here).
5. Optionally patch market_panel.csv in-place: left-join the fee columns.

Output
------
  BCB/Tarifas/processed/cosif_fee_quarterly_conglomerate.csv
  BCB/Egan_et_al_2025_Rep/processed/market_panel_with_fees.csv  (optional)

Usage
-----
  python panel_7b_cosif_fees.py
  python panel_7b_cosif_fees.py --patch-market   # also patches market_panel.csv
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO      = Path(__file__).resolve().parents[2]
PANEL_DIR  = _REPO / "BCB" / "Egan_et_al_2025_Rep" / "processed"
IF_LIST_DIR = _REPO / "BCB" / "IF Data" / "List"
TARIF_DIR  = _REPO / "BCB" / "Tarifas" / "processed"

COSIF_INST_PARQUET = TARIF_DIR / "cosif_service_fees_institution.parquet"
COSIF_INST_CSV     = TARIF_DIR / "cosif_service_fees_institution.csv"
MARKET_PANEL_CSV   = PANEL_DIR / "market_panel.csv"
OUTPUT_CSV         = TARIF_DIR / "cosif_fee_quarterly_conglomerate.csv"

FEE_COLS = [
    "fee_ratio_demand",
    "fee_ratio_savings",
    "fee_ratio_time",
    "fee_ratio_all",
    "fee_ratio_total_deposits",
]


# ---------------------------------------------------------------------------
# 1. Load COSIF institution panel
# ---------------------------------------------------------------------------

def load_cosif_institution() -> pd.DataFrame:
    if COSIF_INST_PARQUET.exists():
        try:
            import polars as pl
            df = pl.read_parquet(str(COSIF_INST_PARQUET)).to_pandas()
            log.info("Loaded COSIF parquet: %d rows", len(df))
            return df
        except Exception:
            pass
    if COSIF_INST_CSV.exists():
        df = pd.read_csv(COSIF_INST_CSV, dtype={"cnpj": str}, low_memory=False)
        log.info("Loaded COSIF CSV: %d rows", len(df))
        return df
    raise FileNotFoundError(f"COSIF institution panel not found at {COSIF_INST_PARQUET}")


# ---------------------------------------------------------------------------
# 2. Build CNPJ → CodConglomeradoPrudencial map from IF Data
# ---------------------------------------------------------------------------

def build_cnpj_cong_map() -> pd.DataFrame:
    """
    Returns DataFrame with columns: cnpj (8-digit str), cod_cong_prudencial (str).
    Follows the same logic as load_conglomerate_map() in scrape_7_fees.py.
    """
    list_files = sorted(IF_LIST_DIR.glob("IF_DATA_List_*.csv"))
    list_files = [f for f in list_files if f.stat().st_size > 500]
    if not list_files:
        log.warning("No IF Data List files found — conglomerate mapping will be empty.")
        return pd.DataFrame(columns=["cnpj", "cod_cong_prudencial"])

    latest = list_files[-1]
    log.info("Conglomerate map from: %s", latest.name)
    df = pd.read_csv(latest, sep=",", encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["CodInst_str"] = df["CodInst"].astype(str).str.strip()

    # (A) Numeric CodInst rows → direct CNPJ key
    numeric_mask = df["CodInst_str"].str.match(r"^\d+$")
    part_a = (
        df.loc[numeric_mask, ["CodInst_str", "CodConglomeradoPrudencial"]]
        .rename(columns={"CodInst_str": "cnpj",
                         "CodConglomeradoPrudencial": "cod_cong_prudencial"})
        .copy()
    )
    part_a["cnpj"] = part_a["cnpj"].str.zfill(8)

    # (B) CnpjInstituicaoLider column
    leader_valid = df["CnpjInstituicaoLider"].notna()
    part_b = (
        df.loc[leader_valid, ["CnpjInstituicaoLider", "CodConglomeradoPrudencial"]]
        .rename(columns={"CnpjInstituicaoLider": "cnpj",
                         "CodConglomeradoPrudencial": "cod_cong_prudencial"})
        .copy()
    )
    part_b["cnpj"] = (
        part_b["cnpj"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(8)
    )

    cmap = (
        pd.concat([part_a, part_b], ignore_index=True)
        .dropna(subset=["cod_cong_prudencial"])
        .drop_duplicates(subset="cnpj", keep="first")
        .reset_index(drop=True)
    )
    cmap["cod_cong_prudencial"] = cmap["cod_cong_prudencial"].astype(str).str.strip()
    log.info("Conglomerate map: %d CNPJs → %d conglomerates",
             len(cmap), cmap["cod_cong_prudencial"].nunique())
    return cmap


# ---------------------------------------------------------------------------
# 3. Aggregate to conglomerate × quarter
# ---------------------------------------------------------------------------

def to_quarter(data_base_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert YYYYMM int/str to (year, quarter) integer series."""
    dbs = data_base_series.astype(str).str.strip().str[:6]
    year    = dbs.str[:4].astype(int)
    month   = dbs.str[4:6].astype(int)
    quarter = ((month - 1) // 3 + 1).astype(int)
    return year, quarter


def aggregate_to_conglomerate_quarter(
    cosif: pd.DataFrame,
    cmap: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join CNPJ → conglomerate, convert monthly → quarterly,
    aggregate fee ratios (median across institutions, then use the last month
    of each quarter as the representative observation).
    """
    cosif = cosif.copy()
    cosif["cnpj"] = cosif["cnpj"].astype(str).str.zfill(8)

    # Map CNPJ → conglomerate
    cosif = cosif.merge(cmap, on="cnpj", how="left")

    # For institutions without a conglomerate mapping, use the CNPJ itself
    # as a pseudo-conglomerate code (fine for independent institutions)
    cosif["cod_cong_prudencial"] = cosif["cod_cong_prudencial"].fillna(cosif["cnpj"])

    # Add year and quarter
    cosif["year"], cosif["quarter"] = to_quarter(cosif["data_base"])

    # Use the end-of-quarter month as the representative observation
    # (Mar, Jun, Sep, Dec) — drop partial-quarter months
    cosif["month"] = cosif["data_base"].astype(str).str[4:6].astype(int)
    eom_months = {3, 6, 9, 12}
    cosif_eom = cosif[cosif["month"].isin(eom_months)].copy()

    if cosif_eom.empty:
        log.warning("No end-of-quarter observations found — using all months.")
        cosif_eom = cosif.copy()

    # Aggregate: median fee ratio across member institutions per conglomerate × quarter
    grp = (
        cosif_eom.groupby(["cod_cong_prudencial", "year", "quarter"], as_index=False)
        [FEE_COLS]
        .median()
    )

    # Replace inf with NaN
    for col in FEE_COLS:
        if col in grp.columns:
            grp[col] = grp[col].replace([np.inf, -np.inf], np.nan)

    # Rename for market panel compatibility
    rename = {c: f"cosif_{c}" for c in FEE_COLS}
    grp = grp.rename(columns=rename)

    log.info(
        "Conglomerate-quarter panel: %d rows | %d conglomerates | %d quarters",
        len(grp),
        grp["cod_cong_prudencial"].nunique(),
        grp[["year", "quarter"]].drop_duplicates().__len__(),
    )
    return grp


# ---------------------------------------------------------------------------
# 4. Merge onto market_panel.csv
# ---------------------------------------------------------------------------

def patch_market_panel(fee_panel: pd.DataFrame) -> None:
    """Left-join COSIF fee columns onto market_panel.csv, saving as market_panel_with_fees.csv."""
    if not MARKET_PANEL_CSV.exists():
        log.error("market_panel.csv not found at %s", MARKET_PANEL_CSV)
        return

    market = pd.read_csv(MARKET_PANEL_CSV, dtype={"CodConglomeradoPrudencial": str}, low_memory=False)
    log.info("market_panel: %d rows × %d cols", len(market), len(market.columns))

    # Normalise join key
    market["CodConglomeradoPrudencial"] = market["CodConglomeradoPrudencial"].astype(str).str.strip()
    fee_panel = fee_panel.copy()
    fee_panel["cod_cong_prudencial"] = fee_panel["cod_cong_prudencial"].astype(str).str.strip()

    merged = market.merge(
        fee_panel.rename(columns={"cod_cong_prudencial": "CodConglomeradoPrudencial"}),
        on=["CodConglomeradoPrudencial", "year", "quarter"],
        how="left",
    )

    if len(merged) != len(market):
        log.error("Row count changed after merge! %d → %d. Aborting patch.", len(market), len(merged))
        return

    cosif_cols = [c for c in merged.columns if c.startswith("cosif_")]
    matched = merged[cosif_cols[0]].notna().sum() if cosif_cols else 0
    log.info(
        "Patched: %d cosif_fee columns | %d/%d rows matched",
        len(cosif_cols), matched, len(merged),
    )

    out_path = PANEL_DIR / "market_panel_with_fees.csv"
    merged.to_csv(out_path, index=False)
    log.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(patch_market: bool = False) -> None:
    # 1. Load COSIF
    cosif = load_cosif_institution()

    # 2. Build CNPJ → conglomerate map
    cmap = build_cnpj_cong_map()

    # 3. Aggregate
    fee_panel = aggregate_to_conglomerate_quarter(cosif, cmap)

    # 4. Save quarterly conglomerate panel
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fee_panel.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved conglomerate fee panel: %s (%d rows)", OUTPUT_CSV.name, len(fee_panel))

    # 5. Optionally patch market panel
    if patch_market:
        patch_market_panel(fee_panel)

    log.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COSIF fee → market panel joiner")
    parser.add_argument("--patch-market", action="store_true",
                        help="Also patch market_panel.csv → market_panel_with_fees.csv")
    args = parser.parse_args()
    main(patch_market=args.patch_market)
