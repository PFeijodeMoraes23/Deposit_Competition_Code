"""
diag_advertising_in_estimation.py
=================================
Can the assembled advertising series actually serve as the in-window demand regressor?

Three questions, in the order that settles the answer:

 1. COVERAGE. What share of the estimation sample's rows and of its deposits have advertising
    observed, year by year, on the 2016-2024 window?
 2. WITHIN-BANK VARIATION. The identifying variation is within bank, so how many conglomerates
    have enough in-window quarters, and how wide is the within-bank spread?
 3. SURVIVAL UNDER TWO-WAY FE. The demand equation carries conglomerate and quarter fixed
    effects. If those absorb the series it cannot identify anything - the fate of frac_4g5g,
    98.9% absorbed. Reported as the share of variance left after entity and quarter demeaning,
    computed by alternating projections.

Sources are NEVER spliced (an assembly decision: a source change inside a bank's history is a
level jump that bank fixed effects cannot absorb). So ONE source is chosen per conglomerate and
kept for the whole window, rather than preferring a different source in each quarter: the central
bank's accounts where they cover the window (member institutions summed, else the conglomerate
document), and another source only when it covers more than a year of quarters beyond them.

Reads only finished outputs: the advertising panels, the demand-prep parquet and the market panel.
Nothing in the estimation pipeline is touched.

Usage
-----
  python diagnostics/diag_advertising_in_estimation.py
  python diagnostics/diag_advertising_in_estimation.py --routine 4 --from-year 2016 --to-year 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/ and pipeline modules
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from utils import paths

PROC = paths.AWARENESS_PROC
# See diag_advertising_coverage.py: an imputed cell (BRB `imputed_from_row_total`) counts as
# observed here too (user decision, 2026-09-28).
OBSERVED = ("observed_positive", "observed_zero", "observed_negative", "imputed_from_row_total")
# One canonical measure per source, as in diag_advertising_coverage.py.
MEASURE = {"cvm": "as_filed", "cosif_conglomerate": "adv", "cosif_institution": "adv",
           "statebank_own": "adv_production", "caixa_own": "adv_production"}
# The central bank's accounts come first: the member institutions summed are the main measure and
# the conglomerate document the fallback (user decision 2026-10-05). A source lower in the list
# is chosen only when it covers more than SLACK quarters beyond every source above it.
PRIORITY = ["cosif_institution", "cosif_conglomerate", "cvm", "caixa_own", "statebank_own"]
SLACK = 4


def estimation_frame(routine: int) -> pd.DataFrame:
    f = paths.demand_prep_root() / f"demand_{routine}_index_spec_12.parquet"
    if not f.exists():
        raise SystemExit(f"missing {f}")
    d = pd.read_parquet(f, columns=["CodConglomeradoPrudencial", "year", "quarter", "is_B",
                                    "deposit_type", "deposit_balance"])
    return (d.groupby(["CodConglomeradoPrudencial", "year", "quarter"])
             .agg(rows=("deposit_balance", "size"), deposits=("deposit_balance", "sum"),
                  is_B=("is_B", "max")).reset_index()
             .rename(columns={"CodConglomeradoPrudencial": "panel_code"}))


def advertising_one_source(lo: int, hi: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One advertising series per conglomerate-quarter, from a single source per conglomerate."""
    q = pd.read_parquet(PROC / "advertising_panel_quarterly.parquet")
    keep = pd.concat([q[(q["source"] == s) & (q["measure"] == m)] for s, m in MEASURE.items()])
    keep = keep[keep["value_status"].isin(OBSERVED) & keep["year"].between(lo, hi)]
    # A quarter short of a month is not an observed quarter (Banrisul 2024Q4 has no November).
    keep = keep[~keep["flag_notes"].fillna("").str.contains("incomplete_quarter")]
    n = keep.groupby(["panel_code", "source"])["quarter"].size().rename("n_quarters").reset_index()
    n["rank"] = n["source"].map({s: i for i, s in enumerate(PRIORITY)})
    best = n.groupby("panel_code")["n_quarters"].transform("max")
    n = n[n["n_quarters"] >= best - SLACK].sort_values(["panel_code", "rank"])
    chosen = n.groupby("panel_code").first().reset_index()[["panel_code", "source", "n_quarters"]]
    out = keep.merge(chosen[["panel_code", "source"]], on=["panel_code", "source"])
    cols = ["panel_code", "year", "quarter", "source", "amount_brl", "amount_brl_real",
            "total_assets_brl", "adv_over_assets", "validated"]
    return out[cols], chosen


def twoway_residual_share(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Share of variance surviving entity, then entity and quarter, demeaning."""
    d = df.dropna(subset=[col])
    if d[col].nunique() < 3:
        return float("nan"), float("nan")
    y = d[col].to_numpy(dtype=float)
    ent = d["panel_code"].to_numpy()
    tim = (d["year"].astype(int) * 10 + d["quarter"].astype(int)).to_numpy()
    total = float(np.var(y))
    if not total:
        return float("nan"), float("nan")
    r = y - y.mean()
    for _ in range(200):                       # alternating projections
        prev = r.copy()
        r = r - pd.Series(r).groupby(ent).transform("mean").to_numpy()
        r = r - pd.Series(r).groupby(tim).transform("mean").to_numpy()
        if np.max(np.abs(r - prev)) < 1e-12:
            break
    one_way = y - pd.Series(y).groupby(ent).transform("mean").to_numpy()
    return float(np.var(one_way) / total), float(np.var(r) / total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routine", type=int, default=3)
    ap.add_argument("--from-year", type=int, default=2016)
    ap.add_argument("--to-year", type=int, default=2024)
    ap.add_argument("--out-dir", type=Path, default=PROC)
    a = ap.parse_args()

    est = estimation_frame(a.routine)
    est = est[est["year"].between(a.from_year, a.to_year)]
    adv, chosen = advertising_one_source(a.from_year, a.to_year)
    log.info("source chosen per conglomerate (most in-window quarters):\n%s",
             chosen["source"].value_counts().to_string())

    m = est.merge(adv, on=["panel_code", "year", "quarter"], how="left", indicator=True)
    m["covered"] = m["_merge"] == "both"

    rows = []
    for y, g in m.groupby("year"):
        rows.append({"year": int(y), "congl_in_sample": g["panel_code"].nunique(),
                     "congl_covered": g.loc[g["covered"], "panel_code"].nunique(),
                     "share_rows": 100 * g.loc[g["covered"], "rows"].sum() / g["rows"].sum(),
                     "share_deposits": 100 * g.loc[g["covered"], "deposits"].sum()
                                       / g["deposits"].sum()})
    by_year = pd.DataFrame(rows)

    cov = m[m["covered"]]
    per_bank = (cov.groupby("panel_code")
                  .agg(n_quarters=("quarter", "size"), source=("source", "first"),
                       mean_intensity=("adv_over_assets", "mean"),
                       sd_intensity=("adv_over_assets", "std"),
                       sd_log_adv=("amount_brl_real", lambda s: float(
                           np.std(np.log(s[s > 0])) if (s > 0).sum() > 1 else np.nan)))
                  .reset_index())
    per_bank["cv_intensity"] = per_bank["sd_intensity"] / per_bank["mean_intensity"]

    panel = cov.drop_duplicates(["panel_code", "year", "quarter"]).copy()
    panel["log_adv"] = np.log(panel["amount_brl_real"].where(panel["amount_brl_real"] > 0))
    fe_rows = []
    for col, lab in (("log_adv", "log real advertising"),
                     ("adv_over_assets", "advertising / total assets")):
        one, two = twoway_residual_share(panel, col)
        fe_rows.append({"variable": lab, "share_after_entity_FE": 100 * one,
                        "share_after_entity_and_quarter_FE": 100 * two,
                        "absorbed_by_two_way_FE": 100 * (1 - two)})
    fe = pd.DataFrame(fe_rows)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    by_year.to_csv(a.out_dir / "advertising_estimation_coverage.csv", index=False)
    per_bank.to_csv(a.out_dir / "advertising_estimation_within_bank.csv", index=False)
    fe.to_csv(a.out_dir / "advertising_estimation_fe_absorption.csv", index=False)

    log.info("\n1. Coverage of the E%d spec-12 estimation sample, %d-%d:\n%s", a.routine,
             a.from_year, a.to_year, by_year.round(1).to_string(index=False))
    log.info("\n2. Within-bank variation on the covered set (%d conglomerates):", len(per_bank))
    log.info("   conglomerates with >= 8 in-window quarters: %d; >= 20: %d",
             int((per_bank["n_quarters"] >= 8).sum()), int((per_bank["n_quarters"] >= 20).sum()))
    log.info("   median within-bank CV of advertising/assets: %.2f; median sd of log real "
             "advertising: %.2f", float(per_bank["cv_intensity"].median()),
             float(per_bank["sd_log_adv"].median()))
    log.info("%s", per_bank.sort_values("n_quarters", ascending=False).round(4)
                           .to_string(index=False))
    log.info("\n3. Variance surviving fixed effects, %% of total:\n%s",
             fe.round(1).to_string(index=False))
    log.info("\nwrote three CSVs to %s", a.out_dir)


if __name__ == "__main__":
    main()
