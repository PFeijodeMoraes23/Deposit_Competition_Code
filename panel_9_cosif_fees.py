"""
panel_9_cosif_fees.py
=======================
Merge all fee-price columns onto the market panel.

Three fee sources
-----------------
1. COSIF realized revenue ratios   (scrape_18_cosif_service_fees.py)
   - Quarterly, conglomerate-level  |  2013-Q1 → 2026-Q1
   - 5 fee/deposit ratio columns    |  cosif_fee_ratio_*
   - NOTE: 2025+ data has a COSIF account reclassification artifact
     (mandatory new plan from Jan 2025). Set cosif_fee_valid=0 for year>=2025.

2. BCB Tarifas listed maximum prices  (scrape_20_bcb_tariff_vigencia.py)
   - Static snapshot, no time variation  |  34 well-covered conglomerates
   - Key priority services (PF)          |  listed_fee_<name>_pf columns
   - Useful as cross-sectional instruments for BLP identification

3. DataVigencia price-stickiness instrument  (scrape_20_bcb_tariff_vigencia.py)
   - Time-varying, conglomerate × quarter  |  2013-Q1 → latest
   - Mean years since last tariff change across BCB priority services
   - tarifa_stickiness_yrs: grows until a price change, resets to 0 at change

Output
------
  BCB/Tarifas/processed/cosif_fee_quarterly_conglomerate.csv  (COSIF only)
  BCB/Egan_et_al_2025_Rep/processed/market_panel_with_fees.csv  (all sources)

Usage
-----
  python panel_9_cosif_fees.py
  python panel_9_cosif_fees.py --patch-market

Note
----
  Estimation scripts read market_panel.csv, NOT this file (changed 2026-08-04). The fee columns
  it adds are used by no estimator: they appear in no .jl, in neither X_COLS nor D_COLS, and in
  no sleepiness design — they reached the demand parquets only via IV_FEE, which is skipped when
  absent. The old "prefer with_fees if it exists" rule meant a stale copy of this file silently
  shadowed a freshly rebuilt market_panel, which is exactly what happened between 2026-07-28 and
  2026-08-03. Set USE_FEE_PANEL=1 to opt back in (e.g. to estimate a fee specification); see
  utils/paths.market_panel_csv. This script remains the way to BUILD the fee panel.
"""

from __future__ import annotations

import argparse
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
from utils import paths
_REPO           = Path(__file__).resolve().parents[2]
PANEL_DIR       = paths.PROCESSED
IF_LIST_DIR     = paths.IF_DATA_LIST
TARIF_DIR       = paths.TARIFAS_PROC

COSIF_INST_PARQUET  = TARIF_DIR / "cosif_service_fees_institution.parquet"
COSIF_INST_CSV      = TARIF_DIR / "cosif_service_fees_institution.csv"
VIGENCIA_HIST_CSV   = TARIF_DIR / "tariff_vigencia_history.csv"
VIGENCIA_SUMM_CSV   = TARIF_DIR / "tariff_vigencia_summary.csv"
MARKET_PANEL_CSV    = PANEL_DIR / "market_panel.csv"
OUTPUT_CSV          = TARIF_DIR / "cosif_fee_quarterly_conglomerate.csv"

# COSIF fee ratio columns (from scrape_12)
FEE_COLS = [
    "fee_ratio_demand",
    "fee_ratio_savings",
    "fee_ratio_time",
    "fee_ratio_all",
    "fee_ratio_total_deposits",
]

# BCB priority services (Resolucao CMN 3.919/2010 basket + ATM)
PRIORITY_SERVICES = {
    1101: "acct_reg",        # Cadastro para inicio de relacionamento
    1201: "card_2nd_debit",  # 2a via cartao debito
    1203: "cheque_stop",     # Contra-ordem de cheque
    1204: "cheque_countermand",  # Oposicao cheque
    1213: "statement",       # Extrato periodo anterior
    1501: "atm_withdrawal",  # Saque em terminal (ATM)
}

# Year of the COSIF structural break; rows with year >= this value get cosif_fee_valid=0
COSIF_BREAK_YEAR = 2025

# ---------------------------------------------------------------------------
# COSIF fee/deposit ratio plausibility band.
#
# A quarterly service-fee / deposit ratio is economically O(0.001-0.1): fees are
# at most a few percent of the deposit base per quarter.  The clean 2016-2018
# distribution confirms this (per-year MEDIAN ~0.009 for fee_ratio_all, ~0.05
# for the smallest-denominator fee_ratio_demand; p95 <= 0.13 and <= 0.57
# respectively).  The MEDIANS are stable and sane in EVERY year -- the cross-year
# "break" (means of 100-3000, maxima up to 1e10) is a pure SMALL-DENOMINATOR
# explosion, not a units mismatch or an account-definition change:
#
#   * a bank whose demand/savings/time deposits are a near-zero sliver of its
#     real balance sheet (e.g. R$142 of classified demand deposits against
#     R$21bn total) -- the dep_sub sub-sum divides fee revenue by that sliver;
#   * a payment institution with a token deposit stock (R$0.02, R$380) against
#     large fee revenue.
#
# Such values are not fee ratios and are treated as MISSING, exactly as
# panel_3_master_panel_build NaNs rate_a4 above RATE_A4_MAX_MULT x Selic and
# cosif_implicit_rate above 0.5.  Negative ratios (from svc_revenue_inc within-
# semester reversals) are likewise implausible for a fee ratio and NaN'd.
# The ceiling is env-overridable (mirrors panel_3's RATE_A4_MAX_MULT); the
# default 1.0 = "quarterly fees cannot exceed 100% of the deposit base" leaves
# the entire clean central distribution (p95 <= 0.57) untouched while removing
# every explosion.  A cosif_fee_guarded flag marks affected cong-quarters,
# mirroring panel_3's rate_a4_guarded.
COSIF_FEE_RATIO_MAX = float(os.environ.get("COSIF_FEE_RATIO_MAX", 1.0))


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
# 2. Build CNPJ -> CodConglomeradoPrudencial map from IF Data
# ---------------------------------------------------------------------------

def build_cnpj_cong_map() -> pd.DataFrame:
    """Returns DataFrame: cnpj (8-digit str), cod_cong_prudencial (str)."""
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

    numeric_mask = df["CodInst_str"].str.match(r"^\d+$")
    part_a = (
        df.loc[numeric_mask, ["CodInst_str", "CodConglomeradoPrudencial"]]
        .rename(columns={"CodInst_str": "cnpj",
                         "CodConglomeradoPrudencial": "cod_cong_prudencial"})
        .copy()
    )
    part_a["cnpj"] = part_a["cnpj"].str.zfill(8)

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
    log.info("Conglomerate map: %d CNPJs, %d conglomerates",
             len(cmap), cmap["cod_cong_prudencial"].nunique())
    return cmap


# ---------------------------------------------------------------------------
# 3. COSIF: aggregate to conglomerate x quarter
# ---------------------------------------------------------------------------

def _to_quarter(data_base_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    dbs   = data_base_series.astype(str).str.strip().str[:6]
    year  = dbs.str[:4].astype(int)
    month = dbs.str[4:6].astype(int)
    return year, ((month - 1) // 3 + 1).astype(int)


def aggregate_cosif_to_conglomerate_quarter(
    cosif: pd.DataFrame,
    cmap:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate COSIF institution panel to conglomerate × quarter.

    Uses monthly revenue increments (not cumulative/6) to compute correct
    quarterly fee ratios: Q_ratio = sum(3 monthly increments) / EOM deposit stock.
    If the parquet pre-dates the increment fix, recomputes increments on the fly.
    """
    cosif = cosif.copy()
    cosif["cnpj"] = cosif["cnpj"].astype(str).str.zfill(8)
    cosif = cosif.merge(cmap, on="cnpj", how="left")
    cosif["cod_cong_prudencial"] = cosif["cod_cong_prudencial"].fillna(cosif["cnpj"])
    cosif["year"], cosif["quarter"] = _to_quarter(cosif["data_base"])
    cosif["month"] = cosif["data_base"].astype(str).str[4:6].astype(int)

    # ── Ensure svc_revenue_inc is available ────────────────────────────────
    # scrape_12 >= fix: column already present.  scrape_12 < fix: recompute.
    if "svc_revenue_inc" not in cosif.columns:
        log.info("svc_revenue_inc not in parquet — computing from cumulative svc_revenue")
        cosif["half_yr"] = np.where(cosif["month"] <= 6, 1, 2)
        cosif = cosif.sort_values(["cnpj", "year", "half_yr", "month"])
        for col, out in [("svc_revenue", "svc_revenue_inc"),
                         ("svc_revenue_pf", "svc_revenue_pf_inc"),
                         ("svc_revenue_pj", "svc_revenue_pj_inc")]:
            if col in cosif.columns:
                d = cosif.groupby(["cnpj", "year", "half_yr"])[col].diff()
                cosif[out] = d.fillna(cosif[col])
            else:
                cosif[out] = np.nan
    else:
        log.info("Using pre-computed svc_revenue_inc from parquet")
        for col, out in [("svc_revenue_pf", "svc_revenue_pf_inc"),
                         ("svc_revenue_pj", "svc_revenue_pj_inc")]:
            if out not in cosif.columns and col in cosif.columns:
                cosif["half_yr"] = cosif.get("half_yr",
                                             np.where(cosif["month"] <= 6, 1, 2))
                d = cosif.groupby(["cnpj", "year", "half_yr"])[col].diff()
                cosif[out] = d.fillna(cosif[col])

    # ── Quarterly aggregation ───────────────────────────────────────────────
    # Step 1: sum monthly revenue increments within (conglomerate, year, quarter)
    cosif = cosif.sort_values(["cod_cong_prudencial", "year", "quarter", "month"])
    q_rev = (
        cosif.groupby(["cod_cong_prudencial", "year", "quarter"], as_index=False)
        .agg(
            svc_revenue_q    =("svc_revenue_inc",    "sum"),
            svc_revenue_pf_q =("svc_revenue_pf_inc", "sum"),
            svc_revenue_pj_q =("svc_revenue_pj_inc", "sum"),
            dep_demand       =("dep_demand",  "last"),
            dep_savings      =("dep_savings", "last"),
            dep_time         =("dep_time",    "last"),
            dep_total        =("dep_total",   "last"),
        )
    )

    # Step 2: compute quarterly fee ratios
    dep_sub = q_rev[["dep_demand", "dep_savings", "dep_time"]].sum(axis=1, min_count=1)
    dep_sub = dep_sub.where(dep_sub > 0, other=q_rev["dep_total"])

    q_rev["fee_ratio_demand"]         = q_rev["svc_revenue_q"] / q_rev["dep_demand"]
    q_rev["fee_ratio_savings"]        = q_rev["svc_revenue_q"] / q_rev["dep_savings"]
    q_rev["fee_ratio_time"]           = q_rev["svc_revenue_q"] / q_rev["dep_time"]
    q_rev["fee_ratio_all"]            = q_rev["svc_revenue_q"] / dep_sub
    q_rev["fee_ratio_total_deposits"] = q_rev["svc_revenue_q"] / q_rev["dep_total"]

    grp = q_rev[["cod_cong_prudencial", "year", "quarter"] + FEE_COLS].copy()
    for col in FEE_COLS:
        if col in grp.columns:
            grp[col] = grp[col].replace([np.inf, -np.inf], np.nan)

    # ── Plausibility guard: NaN small-denominator explosions ────────────────
    # (see COSIF_FEE_RATIO_MAX above.)  Each ratio column is guarded
    # independently so that a broken sub-classification (which blows up
    # fee_ratio_all / _demand) does not discard a bank's still-valid
    # fee_ratio_total_deposits.  cosif_fee_guarded = 1 marks any cong-quarter
    # where at least one fee ratio fell outside the sane [0, MAX] band.
    guarded = pd.Series(False, index=grp.index)
    for col in FEE_COLS:
        if col not in grp.columns:
            continue
        col_bad = ((grp[col] < 0) | (grp[col] > COSIF_FEE_RATIO_MAX)).fillna(False)
        guarded = guarded | col_bad
        grp.loc[col_bad, col] = np.nan
    grp["cosif_fee_guarded"] = guarded.astype(int)
    log.info(
        "COSIF fee plausibility guard (ratio < 0 or > %.3g): "
        "%d of %d cong-quarters flagged (%.2f%%) -> NaN",
        COSIF_FEE_RATIO_MAX, int(guarded.sum()), len(grp),
        100 * guarded.mean() if len(grp) else 0.0,
    )

    grp = grp.rename(columns={c: f"cosif_{c}" for c in FEE_COLS})
    log.info("COSIF panel: %d rows | %d conglomerates | %d year-quarters",
             len(grp), grp["cod_cong_prudencial"].nunique(),
             grp[["year", "quarter"]].drop_duplicates().__len__())
    return grp


# ---------------------------------------------------------------------------
# 4. BCB Tarifas: static listed max prices (cross-sectional instrument)
# ---------------------------------------------------------------------------

def load_tarifa_listed_prices() -> pd.DataFrame:
    """
    Load BCB Tarifas listed maximum prices for key priority services (PF only).
    Returns a conglomerate-level DataFrame (static — no time variation).
    Covers ~61% of market-panel conglomerates; NaN for the rest.

    Columns added:
      listed_fee_<name>_pf  (valor_maximo_median per conglomerate × service)
    """
    if not VIGENCIA_SUMM_CSV.exists():
        log.warning("tariff_vigencia_summary.csv not found — skipping listed prices.")
        return pd.DataFrame(columns=["cod_cong_prudencial"])

    vs = pd.read_csv(VIGENCIA_SUMM_CSV, low_memory=False)
    vs["cod_cong_prudencial"] = vs["cod_cong_prudencial"].astype(str).str.strip()

    pf_rows = vs[
        (vs["customer_type"] == "F") &
        (vs["codigo_servico"].isin(PRIORITY_SERVICES))
    ].copy()

    pf_rows["col_name"] = pf_rows["codigo_servico"].map(
        {k: f"listed_fee_{v}_pf" for k, v in PRIORITY_SERVICES.items()}
    )

    wide = pf_rows.pivot_table(
        index="cod_cong_prudencial",
        columns="col_name",
        values="valor_maximo_median",
        aggfunc="median",
    ).reset_index()
    wide.columns.name = None

    log.info(
        "Listed price columns: %s | %d conglomerates",
        [c for c in wide.columns if c.startswith("listed_")],
        len(wide),
    )
    return wide


# ---------------------------------------------------------------------------
# 5. DataVigencia stickiness: time-varying instrument
# ---------------------------------------------------------------------------

def build_stickiness_panel(cmap: pd.DataFrame) -> pd.DataFrame:
    """
    For each conglomerate × (year, quarter), compute:
      tarifa_stickiness_yrs = mean years since last tariff change,
                              averaged across BCB priority services and
                              all member institutions.

    Time-varying: stickiness grows quarter-by-quarter until a price change
    resets it. Computed from data_vigencia snapshots in tariff_vigencia_history.csv.

    Note: stickiness is NaN for quarters before a tariff's data_vigencia
    (i.e., where we have no evidence the tariff existed).
    """
    if not VIGENCIA_HIST_CSV.exists():
        log.warning("tariff_vigencia_history.csv not found — skipping stickiness.")
        return pd.DataFrame(columns=["cod_cong_prudencial", "year", "quarter"])

    vh = pd.read_csv(VIGENCIA_HIST_CSV, low_memory=False)
    vh["cnpj"] = vh["cnpj"].astype(str).str.zfill(8)

    # Filter to priority services PF only
    vh_pf = vh[
        (vh["customer_type"] == "F") &
        (vh["codigo_servico"].isin(PRIORITY_SERVICES))
    ].copy()

    vh_pf["data_vigencia"] = pd.to_datetime(vh_pf["data_vigencia"], errors="coerce")
    vh_pf = vh_pf.dropna(subset=["data_vigencia"])

    # Map CNPJ -> conglomerate
    vh_pf = vh_pf.merge(cmap, on="cnpj", how="left")
    vh_pf["cod_cong_prudencial"] = vh_pf["cod_cong_prudencial"].fillna(vh_pf["cnpj"])
    vh_pf["cod_cong_prudencial"] = vh_pf["cod_cong_prudencial"].astype(str).str.strip()

    # One data_vigencia per (conglomerate, service) — take the earliest date
    # across member institutions (most conservative: when the conglomerate first set the price)
    cong_svc = (
        vh_pf.groupby(["cod_cong_prudencial", "codigo_servico"])["data_vigencia"]
        .min()
        .reset_index()
    )
    cong_svc.columns = ["cod_cong_prudencial", "codigo_servico", "data_vigencia"]

    # Quarter end dates: 2013-Q1 through 2026-Q1
    quarters = []
    for yr in range(2013, 2027):
        for q, (m, d) in enumerate([(3, 31), (6, 30), (9, 30), (12, 31)], start=1):
            if yr == 2026 and q > 1:
                break
            quarters.append({"year": yr, "quarter": q,
                              "quarter_end": pd.Timestamp(yr, m, d)})
    qdf = pd.DataFrame(quarters)

    # Vectorised cross-join: cong_svc x quarters -> filter -> aggregate
    # cong_svc has ~6K rows, qdf has 53 rows -> cross is ~318K rows (manageable)
    cong_svc["_key"] = 1
    qdf["_key"] = 1
    cross = cong_svc.merge(qdf, on="_key").drop(columns="_key")

    cross = cross[cross["quarter_end"] >= cross["data_vigencia"]].copy()
    cross["stickiness_yrs"] = (
        (cross["quarter_end"] - cross["data_vigencia"]).dt.days / 365.25
    )

    stick = (
        cross.groupby(["cod_cong_prudencial", "year", "quarter"], as_index=False)
        .agg(
            tarifa_stickiness_yrs=("stickiness_yrs", "mean"),
            tarifa_stickiness_n=("stickiness_yrs", "count"),
        )
    )
    log.info(
        "Stickiness panel: %d rows | %d conglomerates | %d year-quarters",
        len(stick),
        stick["cod_cong_prudencial"].nunique(),
        stick[["year", "quarter"]].drop_duplicates().__len__(),
    )
    return stick


# ---------------------------------------------------------------------------
# 6. Merge all fee panels onto market_panel.csv
# ---------------------------------------------------------------------------

def patch_market_panel(
    cosif_panel:   pd.DataFrame,
    listed_prices: pd.DataFrame,
    stickiness:    pd.DataFrame,
) -> None:
    """
    Left-join all fee columns onto market_panel.csv → market_panel_with_fees.csv.

    Column groups added:
      cosif_fee_*           COSIF realized revenue ratios (time-varying)
      cosif_fee_valid       1 if year < COSIF_BREAK_YEAR (2025), else 0
      listed_fee_*_pf       BCB Tarifas listed max prices (static)
      tarifa_stickiness_yrs DataVigencia stickiness instrument (time-varying)
      tarifa_stickiness_n   Number of priority services observed (data quality)
    """
    if not MARKET_PANEL_CSV.exists():
        log.error("market_panel.csv not found at %s", MARKET_PANEL_CSV)
        return

    market = pd.read_csv(MARKET_PANEL_CSV, dtype={"CodConglomeradoPrudencial": str},
                         low_memory=False)
    log.info("market_panel: %d rows x %d cols", len(market), len(market.columns))
    n_orig = len(market)

    def _norm(df, col):
        df[col] = df[col].astype(str).str.strip()
        return df

    market = _norm(market, "CodConglomeradoPrudencial")

    # ── 6a. COSIF time-varying ──────────────────────────────────────────────
    cpanel = cosif_panel.copy()
    cpanel = _norm(cpanel, "cod_cong_prudencial")
    cpanel = cpanel.rename(columns={"cod_cong_prudencial": "CodConglomeradoPrudencial"})

    merged = market.merge(cpanel, on=["CodConglomeradoPrudencial", "year", "quarter"], how="left")
    cosif_cols = [c for c in merged.columns if c.startswith("cosif_")]
    matched_cosif = merged[cosif_cols[0]].notna().sum() if cosif_cols else 0
    log.info("COSIF: %d cols | %d/%d rows matched", len(cosif_cols), matched_cosif, n_orig)

    # Validity flag: 0 in reclassification years (2025+)
    merged["cosif_fee_valid"] = (merged["year"] < COSIF_BREAK_YEAR).astype(int)

    # ── 6b. Listed prices (static — join on conglomerate only) ─────────────
    if not listed_prices.empty and "cod_cong_prudencial" in listed_prices.columns:
        lp = listed_prices.copy()
        lp = _norm(lp, "cod_cong_prudencial")
        lp = lp.rename(columns={"cod_cong_prudencial": "CodConglomeradoPrudencial"})
        merged = merged.merge(lp, on="CodConglomeradoPrudencial", how="left")
        listed_cols = [c for c in merged.columns if c.startswith("listed_fee_")]
        matched_listed = merged[listed_cols[0]].notna().sum() if listed_cols else 0
        log.info("Listed prices: %d cols | %d/%d rows matched", len(listed_cols),
                 matched_listed, n_orig)

    # ── 6c. Stickiness time-varying ─────────────────────────────────────────
    if not stickiness.empty and "cod_cong_prudencial" in stickiness.columns:
        st = stickiness.copy()
        st = _norm(st, "cod_cong_prudencial")
        st = st.rename(columns={"cod_cong_prudencial": "CodConglomeradoPrudencial"})
        merged = merged.merge(st, on=["CodConglomeradoPrudencial", "year", "quarter"], how="left")
        matched_stick = merged["tarifa_stickiness_yrs"].notna().sum()
        log.info("Stickiness: %d/%d rows matched", matched_stick, n_orig)

    if len(merged) != n_orig:
        log.error("Row count changed after merges! %d -> %d. Aborting.", n_orig, len(merged))
        return

    # ── 6d. Defragment then save ─────────────────────────────────────────────
    merged = merged.copy()  # collapses fragmented frame
    out_path = PANEL_DIR / "market_panel_with_fees.csv"
    merged.to_csv(out_path, index=False)

    new_cols = [c for c in merged.columns if c not in market.columns or c == "cosif_fee_valid"]
    log.info("Saved: %s | %d rows x %d cols (+%d new fee cols)",
             out_path.name, len(merged), len(merged.columns), len(new_cols))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(patch_market: bool = False) -> None:
    # 1. Load COSIF institution panel
    cosif = load_cosif_institution()

    # 2. Build CNPJ -> conglomerate map (shared across all sources)
    cmap = build_cnpj_cong_map()

    # 3. COSIF: aggregate to conglomerate x quarter
    cosif_panel = aggregate_cosif_to_conglomerate_quarter(cosif, cmap)

    # 4. Save COSIF quarterly conglomerate panel (standalone output)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cosif_panel.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved COSIF conglomerate panel: %s (%d rows)", OUTPUT_CSV.name, len(cosif_panel))

    # 5. BCB Tarifas listed prices (static)
    listed_prices = load_tarifa_listed_prices()

    # 6. DataVigencia stickiness (time-varying)
    stickiness = build_stickiness_panel(cmap)

    # 7. Patch market panel with all fee sources
    if patch_market:
        patch_market_panel(cosif_panel, listed_prices, stickiness)

    log.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fee panel builder: COSIF + BCB Tarifas listed prices + DataVigencia stickiness"
    )
    parser.add_argument("--patch-market", action="store_true",
                        help="Merge all fee columns onto market_panel.csv -> market_panel_with_fees.csv")
    args = parser.parse_args()
    main(patch_market=args.patch_market)
