"""switch_cost_r4_locked.py -- the locked share r_4 of time deposits, by conglomerate and year,
for the F3 arm of the switch-cost demand prep (common attention plus locked CDB balances).

r_4 is the share of a bank's time-deposit balances that cannot move within the quarter. Two
bounds, both reported (user decision of 2026-10-13):

  r4_hi  CONTRACTUAL: time deposits with remaining maturity beyond 90 days, from the
         statement notes (raw/DEPOSIT_MATURITY/deposit_maturity_summary.csv, 11 banks,
         2016-2024): 1 - share_no_maturity - share_le90. An upper bound, because a long CDB
         with daily liquidity counts here although it can be redeemed at any time.
  r4_lo  EARLIEST WITHDRAWAL: retail funding that can be withdrawn only after six months or
         more, as a share of retail funding with a stated term, from the Pillar 3 NSFR tables
         (raw/PILAR3/liq2_retail_buckets.csv, row "retail funding", 5 deposit banks,
         2020 onward). A lower bound: the NSFR dates a deposit by the first day the client
         can withdraw it, and three to six months of lock-in is not separated.

Fill rule, per bound: the bank's own year; else the bank's nearest observed year; else the
median over observed banks in that year (the median of the nearest year with observations
when a year has none, i.e. 2016-2019 for r4_lo). Banks outside the sources get the year
median (user decision of 2026-10-13). The prep reads the `__median__` rows for them.

Reads only the two raw tables above; writes
ESTIMATION_OUTPUT/SWITCH_COST/switch_cost_inputs/r4_locked_bank_year.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from utils import switch_cost_sandbox  # noqa: F401  (output tree: import before any path is resolved)
from utils import paths

YEARS = list(range(2016, 2025))
MATURITY_CSV = paths.RAW / "DEPOSIT_MATURITY" / "deposit_maturity_summary.csv"
NSFR_CSV = paths.RAW / "PILAR3" / "liq2_retail_buckets.csv"
OUT_DIR = switch_cost_sandbox.check_dir(switch_cost_sandbox.LOCAL_BASE / "switch_cost_inputs", "OUT_DIR")
OUT_CSV = OUT_DIR / "r4_locked_bank_year.csv"
MEDIAN_KEY = "__median__"

# Statement-note firm keys and NSFR CNPJs -> prudential conglomerate of the market panel.
FIRM_TO_CONGL = {
    "abc": "C0080312", "banrisul": "C0080154", "bb": "C0080329", "bmg": "C0080178",
    "bradesco": "C0080075", "btg": "C0080336", "daycoval": "C0081744", "itau": "C0080099",
    "nubank": "C0084693", "pan": "C0080257", "santander_br": "C0080185",
}
# BNDES (33657248000189) takes no deposits and is left out of the median.
CNPJ_TO_CONGL = {
    191: "C0080329", 360305000104: "C0080738", 60701190000104: "C0080099",
    60746948000112: "C0080075", 90400888000142: "C0080185",
}


def contractual_bound() -> pd.DataFrame:
    d = pd.read_csv(MATURITY_CSV)
    # one source per bank-year: the CVM filing when present (every bank but Nubank has it)
    d["_pref"] = (d["source"] != "cvm").astype(int)
    d = d.sort_values(["firm_key", "fiscal_year", "_pref"]).drop_duplicates(["firm_key", "fiscal_year"])
    d["value"] = 1.0 - d["share_no_maturity"].fillna(0.0) - d["share_le90"]
    d["CodConglomeradoPrudencial"] = d["firm_key"].map(FIRM_TO_CONGL)
    if d["CodConglomeradoPrudencial"].isna().any():
        raise SystemExit(f"unmapped firm keys: {sorted(d.loc[d['CodConglomeradoPrudencial'].isna(), 'firm_key'].unique())}")
    return d.rename(columns={"fiscal_year": "year"})[["CodConglomeradoPrudencial", "year", "value", "source"]]


def withdrawal_bound() -> pd.DataFrame:
    n = pd.read_csv(NSFR_CSV)
    n = n[(n["row_label"] == "retail funding") & n["cnpj"].isin(CNPJ_TO_CONGL)].copy()
    term = n["under_6m"] + n["6m_to_1y"] + n["1y_or_more"]
    n["value"] = (n["6m_to_1y"] + n["1y_or_more"]) / term.replace(0, np.nan)
    n["year"] = n["quarter"].astype(str).str[:4].astype(int)
    n["CodConglomeradoPrudencial"] = n["cnpj"].map(CNPJ_TO_CONGL)
    g = n.groupby(["CodConglomeradoPrudencial", "year"], as_index=False)["value"].mean()
    g["source"] = "pillar3_liq2"
    return g


def fill(obs: pd.DataFrame, name: str) -> pd.DataFrame:
    obs = obs.dropna(subset=["value"])
    med = obs[obs["year"].isin(YEARS)].groupby("year")["value"].median()
    yrs_obs = sorted(med.index)
    if not yrs_obs:
        raise SystemExit(f"{name}: no observation inside {YEARS[0]}-{YEARS[-1]}")
    rows = []
    for y in YEARS:
        if y in med.index:
            rows.append((MEDIAN_KEY, y, float(med[y]), "year_median"))
        else:
            near = min(yrs_obs, key=lambda z: (abs(z - y), z))
            rows.append((MEDIAN_KEY, y, float(med[near]), f"year_median_of_{near}"))
    for c, g in obs.groupby("CodConglomeradoPrudencial"):
        own = g.set_index("year")["value"]
        for y in YEARS:
            if y in own.index:
                rows.append((c, y, float(own[y]), "own"))
            else:
                near = min(own.index, key=lambda z: (abs(z - y), z))
                rows.append((c, y, float(own[near]), f"own_{near}"))
    return pd.DataFrame(rows, columns=["CodConglomeradoPrudencial", "year", name, f"{name}_fill"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    hi = fill(contractual_bound(), "r4_hi")
    lo = fill(withdrawal_bound(), "r4_lo")
    keys = pd.concat([hi[["CodConglomeradoPrudencial", "year"]], lo[["CodConglomeradoPrudencial", "year"]]]).drop_duplicates()
    out = keys.merge(hi, how="left").merge(lo, how="left")
    med = out[out["CodConglomeradoPrudencial"] == MEDIAN_KEY].set_index("year")
    for b in ("r4_hi", "r4_lo"):
        miss = out[b].isna()
        out.loc[miss, b] = out.loc[miss, "year"].map(med[b])
        out.loc[miss, f"{b}_fill"] = "year_median"
    out = out.sort_values(["CodConglomeradoPrudencial", "year"])
    if not ((out["r4_lo"] <= out["r4_hi"] + 1e-12) | out["r4_lo"].isna()).all():
        print("[note] r4_lo exceeds r4_hi in some bank-years (different sources and populations)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(med[["r4_hi", "r4_hi_fill", "r4_lo", "r4_lo_fill"]].round(3).to_string())
    print(f"\n{len(out)} rows, {out['CodConglomeradoPrudencial'].nunique() - 1} banks -> {OUT_CSV}")


if __name__ == "__main__":
    main()
