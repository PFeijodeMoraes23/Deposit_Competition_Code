"""
panel_12_local_cdb_spread.py
============================
Pipeline Step 4d-bis.  Toggleable, idempotent, REVERSIBLE patch that gives the
type-4 (CDB / time-deposit) spread genuine LOCAL (MCA) variation, layered on top
of the calibrated NATIONAL rate built by panel_3 / panel_6.

WHY.  panel_3 builds rate_a4/spread_a4 at CodConglomeradoPrudencial x quarter and
panel_6 broadcasts them identically to every MCA a conglomerate serves (verified:
0% within-(congl x quarter) variance).  ESTBAN carries real municipal time-deposit
balances (V432), so the QUANTITY side already varies by MCA while the PRICE does
not.  This step reconstructs a genuinely local CDB rate from REAL data — each
conglomerate's member-CNPJ COSIF CDB rates weighted by each CNPJ's ESTBAN V432
balance per municipality — and adds ONLY the within-conglomerate deviation

    dev_qoq(congl, mca, t) = r_mca(congl, mca, t) - r_congl(congl, t)

to the national rate.  The national level / identification is untouched; single-
CNPJ conglomerates get dev=0 (local == national).  See diag_k4_spread_local_variation.py
and proto_1_member_cnpj_local_panel.py for the diagnostics that motivated this.

TOGGLE.  Gated by env LOCAL_CDB_SPREAD (default "1" = ON).  "0" restores the
national columns from the *_national snapshots and drops the patch columns.

GUARD (two-sided).  A patched cell is kept only if its local rate lands in
    [RATE_A4_MIN_MULT * Selic,  min(RATE_Q_CAP, RATE_A4_MAX_MULT * Selic)]
(defaults: Selic/3 .. 3*Selic, i.e. log-symmetric); otherwise that cell reverts to
the national rate and its dev is zeroed.  The FLOOR was added 2026-07-24: the guard
had been upper-only (`rate_loc > 0`), so implausibly low member-CNPJ rates — a small
or noisy stk_time_lag denominator rather than real local pricing — passed through and
produced large positive spread deviations that flipped a cell's sign (worst observed:
rate = 0.01-0.04 x Selic, spread_ann -0.02 -> +0.13).  Impact is small and local: the
affected cells were ~0.04% of panel rows, alpha moved only in the 3rd-4th decimal, and
the LOO Hausman instrument was essentially unchanged (corr(local, national) = 0.99996),
so this does NOT invalidate estimates produced before the fix.

IDEMPOTENT.  On apply, national values are snapshotted once to
{spread_a4,rate_a4,spread_ann_a4,leave_one_out_mean_spread_a4}_national.  Re-running
restores from the snapshot first, so the deviation is never double-counted.

WHAT CHANGES.  Only the type-4 (*_a4) columns:
  rate_a4, spread_a4, spread_ann_a4               (fractions; matches panel_6 units)
  leave_one_out_mean_spread_a4                    (recomputed from the patched spread;
                                                   panel_6 formula, year x quarter)
rate_a4 is not read by demand-prep/sleep (only spread_a4 -> spread_qoq and
spread_ann_a4 -> spread_ann feed the parquet), but it is patched for consistency
with desc_*/cosif_*.  leave_one_out_mean_spread_a4 IS a sleep HausmanFull instrument
(specs 4/8/12), so it is recomputed to reflect the new spread.

PIPELINE PLACEMENT.  Run AFTER panel_10_estban_instruments.py (so estban_rival_branches_lag
is present) and BEFORE panel_9_cosif_fees.py --patch-market (so the local spread propagates
into market_panel_with_fees.csv, the file the estimators actually read).  Keyed
(CodConglomeradoPrudencial, mca_code, year, quarter); the merge is 1:1 with the wide panel.

Usage:  python panel_12_local_cdb_spread.py            # apply (or LOCAL_CDB_SPREAD=1)
        LOCAL_CDB_SPREAD=0 python panel_12_local_cdb_spread.py   # revert to national
"""
from __future__ import annotations
import os
import sys
import logging

import numpy as np
import pandas as pd

try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except Exception:
    pass

from utils import paths
import cosif_process_2_calibrate as cc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("panel_12")

PANEL_CSV = paths.PROCESSED / "market_panel.csv"
FOUNDATION = paths.PROCESSED / "COSIF_PROCESSED" / "custos_implicitos_v2_foundation.csv"
ESTBAN_CSV = paths.ESTBAN_CSV
MCA_XWALK = paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"

KEYS = ["CodConglomeradoPrudencial", "mca_code", "year", "quarter"]
RATE_Q_CAP = 0.5                                        # per-CNPJ rate plausibility cap (matches cosif_process_2)
RATE_A4_MAX_MULT = float(os.environ.get("RATE_A4_MAX_MULT", 3.0))   # panel_3's rate_a4 band multiple of Selic
# LOWER band multiple. The guard used to be one-sided (rate>0 only), so implausibly LOW member-CNPJ
# rates — driven by a small/noisy stk_time_lag denominator, not by local pricing — passed straight
# through and produced large POSITIVE spread deviations that could flip a cell's sign (observed:
# rate = 0.01-0.04 x Selic, spread_ann -0.02 -> +0.13). Default is the RECIPROCAL of the max multiple,
# making the admissible band log-symmetric: rate in [Selic/3, 3*Selic]. A time deposit paying under a
# third of the risk-free rate is not a local pricing decision. Empirically this reverts ~1% of the
# applied cells (p1 of rate/Selic = 0.33) while leaving the legitimate dispersion (p50 = 0.93) intact.
RATE_A4_MIN_MULT = float(os.environ.get("RATE_A4_MIN_MULT", 1.0 / RATE_A4_MAX_MULT))
# The patched columns and their national snapshots.
_PATCH_COLS = ["rate_a4", "spread_a4", "spread_ann_a4", "leave_one_out_mean_spread_a4"]
_NAT_COLS = {c: f"{c}_national" for c in _PATCH_COLS}
_AUX_COLS = ["dev_a4_qoq"]


# ---------------------------------------------------------------------------
# Real-data member-CNPJ V432-weighted local deviation (from proto_1, promoted)
# ---------------------------------------------------------------------------
def build_local_deviation() -> pd.DataFrame:
    """Return (CodConglomeradoPrudencial, mca_code, year, quarter, dev_qoq, n_cnpj),
    dev_qoq = r_mca - r_congl (qoq fraction). Per-CNPJ CDB rate = |exp_cdb|/stk_time_lag
    (real 2025+ COSIF leaf) with |exp_lump|/stk_total_lag proxy pre-2025; weighted by
    ESTBAN V432 (time-deposit) municipal balances aggregated to MCA."""
    f = pd.read_csv(FOUNDATION, dtype={"CNPJ": str})
    f["CNPJ"] = f["CNPJ"].str.strip().str.zfill(8)
    f["AnoMes"] = pd.to_numeric(f["AnoMes"], errors="coerce")
    for c in ["exp_cdb", "exp_lump", "stk_time_lag", "stk_total_lag"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")

    def _rate(num, den):                               # expenses stored NEGATIVE -> abs()
        r = num.abs() / den.replace(0.0, np.nan)
        return r.where(np.isfinite(r) & (r > 0) & (r <= RATE_Q_CAP))

    f["r_cnpj"] = _rate(f["exp_cdb"], f["stk_time_lag"]).fillna(_rate(f["exp_lump"], f["stk_total_lag"]))
    f = f.dropna(subset=["AnoMes", "r_cnpj"]).assign(AnoMes=lambda d: d["AnoMes"].astype(int))

    cmap = cc.build_cnpj_to_congl()
    fc = cc.map_foundation_to_congl(f[["CNPJ", "AnoMes", "r_cnpj"]], cmap)[
        ["CNPJ", "AnoMes", "CodConglomeradoPrudencial", "r_cnpj"]]
    log.info("member-CNPJ CDB rates: %d CNPJ-quarters, %d conglomerates, years %d-%d",
             len(fc), fc["CodConglomeradoPrudencial"].nunique(),
             fc["AnoMes"].min() // 100, fc["AnoMes"].max() // 100)

    est = pd.read_csv(ESTBAN_CSV, usecols=["CNPJ", "CODMUN_IBGE", "YEAR", "MONTH", "V432"],
                      dtype={"CNPJ": str}, low_memory=False, encoding="latin-1")
    est = est[est["MONTH"].isin([3, 6, 9, 12])].copy()
    est["CNPJ"] = est["CNPJ"].str.strip().str.zfill(8)
    est["V432"] = pd.to_numeric(est["V432"], errors="coerce").fillna(0.0).clip(lower=0)
    est["CODMUN_IBGE"] = pd.to_numeric(est["CODMUN_IBGE"], errors="coerce")
    est = est.dropna(subset=["CODMUN_IBGE"])
    est["CODMUN_IBGE"] = est["CODMUN_IBGE"].astype(int)
    est["AnoMes"] = (est["YEAR"].astype(int) * 100 + est["MONTH"].astype(int)).astype(int)
    est["year"] = est["YEAR"].astype(int)
    est["quarter"] = (est["MONTH"] // 3).astype(int)

    xw = pd.read_csv(MCA_XWALK, usecols=["municipality_code", "year", "mca_code"])
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    xw["mca_code"] = xw["mca_code"].astype(str)
    est = est.merge(xw[["municipality_code", "mca_code"]],
                    left_on="CODMUN_IBGE", right_on="municipality_code", how="inner")
    est = est.merge(fc, on=["CNPJ", "AnoMes"], how="inner")
    est = est[est["V432"] > 0]

    est["wr"] = est["V432"] * est["r_cnpj"]
    C, M = "CodConglomeradoPrudencial", "mca_code"
    mca = est.groupby([C, M, "year", "quarter"], as_index=False).agg(
        wr=("wr", "sum"), w=("V432", "sum"), n_cnpj=("CNPJ", "nunique"))
    mca["r_mca"] = mca["wr"] / mca["w"]
    congl = est.groupby([C, "year", "quarter"], as_index=False).agg(wr=("wr", "sum"), w=("V432", "sum"))
    congl["r_congl"] = congl["wr"] / congl["w"]
    out = mca.merge(congl[[C, "year", "quarter", "r_congl"]], on=[C, "year", "quarter"], how="left")
    out["dev_a4_qoq"] = out["r_mca"] - out["r_congl"]
    out[C] = out[C].astype(str)
    out["mca_code"] = out["mca_code"].astype(str)
    log.info("local deviation: %d (congl x mca x quarter) cells; non-zero |dev|>1bp: %.2f%%",
             len(out), 100 * (out["dev_a4_qoq"].abs() > 1e-4).mean())
    return out[KEYS + ["dev_a4_qoq", "n_cnpj"]]


# ---------------------------------------------------------------------------
def _adjust_loo_spread_a4(panel: pd.DataFrame) -> pd.Series:
    """Update leave_one_out_mean_spread_a4 for the local spread by adding the exact
    DELTA to panel_6's national value, rather than recomputing from scratch.

    panel_6_market.calculate_hausman_iv_wide sets LOO_i = (S - s_i)/(N-1) with S,N the
    (year,quarter) sum/count of spread_a4.  For s_i -> s_i + d_i:
        LOO_i^loc = LOO_i^nat + (ΣΔ - d_i)/(N-1).
    Using the *_national snapshot as the baseline preserves panel_6's value EXACTLY
    (a from-scratch recompute drifts ~2.5 bps from the stored value under any
    population/precision mismatch); in the full Stage-4 rebuild the delta is exact
    because panel_6 and panel_12 share the same rows."""
    base = pd.to_numeric(panel["leave_one_out_mean_spread_a4_national"], errors="coerce")
    d = (pd.to_numeric(panel["spread_a4"], errors="coerce")
         - pd.to_numeric(panel["spread_a4_national"], errors="coerce")).fillna(0.0)
    panel = panel.assign(_d=d)
    g = panel.groupby(["year", "quarter"])
    dS = g["_d"].transform("sum")
    N = g["spread_a4_national"].transform(lambda x: x.notnull().sum())
    denom = (N - panel["spread_a4_national"].notnull().astype(int)).clip(lower=1)
    return base + (dS - d) / denom


def _restore_national(panel: pd.DataFrame) -> pd.DataFrame:
    """If snapshots exist, copy them back into the working columns (base for re-apply / revert)."""
    if all(nat in panel.columns for nat in _NAT_COLS.values()):
        for col, nat in _NAT_COLS.items():
            panel[col] = panel[nat]
        log.info("restored national values from *_national snapshots")
    return panel


def apply_local_spread(panel: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    panel["CodConglomeradoPrudencial"] = panel["CodConglomeradoPrudencial"].astype(str)
    panel["mca_code"] = panel["mca_code"].astype(str)

    # 1) work from the national base (snapshot on first apply; else restore from snapshot)
    if all(nat in panel.columns for nat in _NAT_COLS.values()):
        panel = _restore_national(panel)
    else:
        for col, nat in _NAT_COLS.items():
            panel[nat] = panel[col]
        log.info("snapshotted national values -> *_national")

    # 2) merge the deviation (cells without member-CNPJ data -> dev 0 = national)
    for c in _AUX_COLS:
        if c in panel.columns:
            panel = panel.drop(columns=[c])
    panel = panel.merge(dev.drop(columns=["n_cnpj"]), on=KEYS, how="left")
    panel["dev_a4_qoq"] = pd.to_numeric(panel["dev_a4_qoq"], errors="coerce").fillna(0.0)

    rate_nat = pd.to_numeric(panel["rate_a4_national"], errors="coerce")
    spr_nat = pd.to_numeric(panel["spread_a4_national"], errors="coerce")
    rf_qoq = spr_nat + rate_nat                             # recover Selic (risk_free_qoq)
    rate_loc = rate_nat + panel["dev_a4_qoq"]

    # 3) TWO-SIDED guard to panel_3's band:
    #        rate_loc in [RATE_A4_MIN_MULT * Selic, min(0.5, RATE_A4_MAX_MULT * Selic)]
    #    else revert that cell to the national rate. The lower bound matters as much as the upper:
    #    without it, near-zero member-CNPJ rates (noisy stk_time_lag denominators) sail through the
    #    old `rate_loc > 0` test and flip the cell's spread sign. Counts are logged per bound so the
    #    band can be tuned from the run log rather than by guesswork.
    upper = np.minimum(RATE_Q_CAP, RATE_A4_MAX_MULT * rf_qoq)
    lower = RATE_A4_MIN_MULT * rf_qoq
    nonfinite = ~np.isfinite(rate_loc) | ~np.isfinite(rf_qoq)
    too_low = (~nonfinite) & (rate_loc < lower)
    too_high = (~nonfinite) & (rate_loc > upper)
    bad = nonfinite | too_low | too_high | (rate_loc <= 0)
    rate_loc = rate_loc.where(~bad, rate_nat)
    panel.loc[bad.fillna(True), "dev_a4_qoq"] = 0.0        # record the effective (guarded) deviation
    n_applied = int((panel["dev_a4_qoq"].abs() > 0).sum())
    log.info("rate guard [%.3g x Selic, min(%.2f, %.3g x Selic)]: reverted %d cells below the floor, "
             "%d above the ceiling, %d non-finite",
             RATE_A4_MIN_MULT, RATE_Q_CAP, RATE_A4_MAX_MULT,
             int(too_low.fillna(False).sum()), int(too_high.fillna(False).sum()),
             int(nonfinite.fillna(True).sum()))

    # 4) rebuild the type-4 columns (fractions)
    panel["rate_a4"] = rate_loc
    panel["spread_a4"] = rf_qoq - rate_loc
    rf_ann = (1.0 + rf_qoq) ** 4 - 1.0
    rate_loc_ann = (1.0 + rate_loc) ** 4 - 1.0
    panel["spread_ann_a4"] = rf_ann - rate_loc_ann
    # where the national base was NaN, keep national (avoid manufacturing values)
    keep_nat = ~np.isfinite(rf_qoq)
    for col, nat in _NAT_COLS.items():
        if col == "leave_one_out_mean_spread_a4":
            continue
        panel.loc[keep_nat, col] = panel.loc[keep_nat, nat]

    # 5) update the leave-one-out mean spread instrument by the exact local delta
    #    (preserves panel_6's national baseline; see _adjust_loo_spread_a4)
    panel["leave_one_out_mean_spread_a4"] = _adjust_loo_spread_a4(panel)

    log.info("applied local CDB spread to %d rows (%.2f%% of panel); "
             "within-(congl x quarter) spread_a4 SD now: %.6g (was 0)",
             n_applied, 100 * n_applied / max(len(panel), 1),
             panel.groupby(["CodConglomeradoPrudencial", "year", "quarter"])["spread_a4"]
             .transform("std").fillna(0).mean())
    return panel


def revert_national(panel: pd.DataFrame) -> pd.DataFrame:
    if not all(nat in panel.columns for nat in _NAT_COLS.values()):
        log.info("no *_national snapshots found — panel already national; nothing to revert.")
        return panel
    panel = _restore_national(panel)
    drop = [c for c in list(_NAT_COLS.values()) + _AUX_COLS if c in panel.columns]
    panel = panel.drop(columns=drop)
    log.info("reverted to national spread and dropped %d patch columns", len(drop))
    return panel


def main() -> None:
    if not PANEL_CSV.exists():
        raise FileNotFoundError(f"market_panel.csv not found at {PANEL_CSV} — run panel_6/panel_7 first.")
    on = os.environ.get("LOCAL_CDB_SPREAD", "1").strip().lower() not in ("0", "false", "off", "no")
    log.info("LOCAL_CDB_SPREAD = %s", "ON (apply local spread)" if on else "OFF (revert to national)")

    log.info("Loading %s", PANEL_CSV)
    panel = pd.read_csv(PANEL_CSV, low_memory=False, dtype={"mca_code": str, "CODMUN_IBGE": str})

    if on:
        dev = build_local_deviation()
        panel = apply_local_spread(panel, dev)
    else:
        panel = revert_national(panel)

    log.info("Writing %s (%d rows x %d cols)", PANEL_CSV, len(panel), len(panel.columns))
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    pa_csv.write_csv(pa.Table.from_pandas(panel, preserve_index=False), str(PANEL_CSV))
    log.info("Done.")


if __name__ == "__main__":
    main()
