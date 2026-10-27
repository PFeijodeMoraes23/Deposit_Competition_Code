"""
utils/type5_rates.py
====================
Posted balance-rate rules for type-5 deposits (balances in prepaid payment accounts).

panel_deposit_rates.py (step 16) prices a type-5 balance with a chain: the conglomerate's
payment-institution rate, else the quarter's median of those rates, else the CDI. Where an issuer's
own documents state what the balance earns, config/type5_balance_rates.csv replaces that chain for
the days the rule covers. Each rule is a percent of the CDI (0 = the balance earns nothing, 100 = the
CDI) in force from `start` to `end`, both inclusive, for one prudential conglomerate.

Within a quarter the rules compound day by day. The CDI accrues daily, so a rule paying p% of the CDI
earns p/100 of each day's CDI, and the days no rule covers keep the chain's rate. With 63 business
days a quarter, the quarter's rate is

    prod_k (1 + p_k/100 * d)^(63 f_k) * (1 + o)^(63 (1 - sum_k f_k)) - 1,

where d is the daily CDI implied by cdi_qoq, o is the daily rate implied by the chain's rate, and f_k
is the share of the quarter's calendar days that rule k covers. A full quarter at 100% returns cdi_qoq
exactly; a full quarter at 0% returns 0.

Rules of one conglomerate may not overlap. Quarters are keyed AnoMes = year * 100 + 3 * quarter, the
quarter-end month, as in step 16.
"""
from __future__ import annotations

import calendar
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "type5_balance_rates.csv"
BUSINESS_DAYS_PER_QUARTER = 63
COLUMNS = ["conglomerate", "issuer", "start", "end", "pct_cdi", "decision", "evidence"]


def load_rules(path: Path = RULES_PATH) -> pd.DataFrame:
    """The rule table, checked: known columns, dates in order, pct_cdi >= 0, no overlap within a
    conglomerate. An absent file means no rules."""
    if not Path(path).is_file():
        return pd.DataFrame(columns=COLUMNS)
    r = pd.read_csv(path, dtype=str, comment="#").fillna("")
    missing = [c for c in COLUMNS if c not in r.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    r["conglomerate"] = r["conglomerate"].str.strip()
    r["start"] = pd.to_datetime(r["start"], format="%Y-%m-%d")
    r["end"] = pd.to_datetime(r["end"], format="%Y-%m-%d")
    r["pct_cdi"] = pd.to_numeric(r["pct_cdi"])
    bad = r[(r["end"] < r["start"]) | (r["pct_cdi"] < 0) | ~r["conglomerate"].str.match(r"^C\d{7}$")]
    if len(bad):
        raise ValueError(f"{path}: invalid rows\n{bad.to_string()}")
    r = r.sort_values(["conglomerate", "start"]).reset_index(drop=True)
    prev_end = r.groupby("conglomerate")["end"].shift(1)
    overlap = r[prev_end.notna() & (r["start"] <= prev_end)]
    if len(overlap):
        raise ValueError(f"{path}: overlapping rules\n{overlap.to_string()}")
    r["rule_id"] = r.index.astype(int)
    return r


def _quarter_bounds(anomes: int) -> tuple[dt.date, dt.date]:
    y, m = divmod(int(anomes), 100)
    first = dt.date(y, m - 2, 1)
    return first, dt.date(y, m, calendar.monthrange(y, m)[1])


def quarter_segments(rules: pd.DataFrame, anomes_values) -> pd.DataFrame:
    """One row per (conglomerate, AnoMes, rule) with the share of the quarter's calendar days the
    rule covers."""
    out = []
    for q in sorted({int(a) for a in anomes_values}):
        q0, q1 = _quarter_bounds(q)
        ndays = (q1 - q0).days + 1
        for _, x in rules.iterrows():
            lo = max(q0, x["start"].date())
            hi = min(q1, x["end"].date())
            if hi >= lo:
                out.append((x["conglomerate"], q, x["rule_id"], float(x["pct_cdi"]),
                            ((hi - lo).days + 1) / ndays))
    return pd.DataFrame(out, columns=["conglomerate", "AnoMes", "rule_id", "pct_cdi", "frac"])


def apply_rules(aux: pd.DataFrame, rules: pd.DataFrame | None = None, rate_col: str = "rate_a5",
                cdi_col: str = "cdi_qoq", key: str = "CodConglomeradoPrudencial"
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace `rate_col` where a rule covers part of a conglomerate-quarter and add the 0/1 flag
    `<rate_col>_posted`. Returns the frame and one row per changed conglomerate-quarter with the
    chain's rate, the new rate, the covered share and the rules used."""
    rules = load_rules() if rules is None else rules
    out = aux.copy()
    flag = f"{rate_col}_posted"
    out[flag] = 0
    empty = pd.DataFrame(columns=[key, "AnoMes", "rate_chain", "rate_posted", "covered", "rules"])
    if rules.empty:
        return out, empty
    keys = out.loc[out[key].astype(str).isin(set(rules["conglomerate"])), [key, "AnoMes"]]
    seg = quarter_segments(rules, keys["AnoMes"].unique())
    if seg.empty:
        return out, empty
    seg = seg.rename(columns={"conglomerate": key})
    out[key] = out[key].astype(str)
    base = out[[key, "AnoMes", rate_col, cdi_col]].drop_duplicates([key, "AnoMes"])
    seg = seg.merge(base, on=[key, "AnoMes"], how="inner")
    n = BUSINESS_DAYS_PER_QUARTER
    seg["d"] = (1.0 + seg[cdi_col]) ** (1.0 / n) - 1.0
    seg["log_g"] = n * seg["frac"] * np.log1p(seg["pct_cdi"] / 100.0 * seg["d"])
    agg = seg.groupby([key, "AnoMes"]).agg(
        log_g=("log_g", "sum"), covered=("frac", "sum"),
        rate_chain=(rate_col, "first"), cdi=(cdi_col, "first"),
        rules=("rule_id", lambda s: ",".join(str(int(v)) for v in s))).reset_index()
    o = (1.0 + agg["rate_chain"]) ** (1.0 / n) - 1.0
    agg["rate_posted"] = np.expm1(agg["log_g"] + n * (1.0 - agg["covered"]) * np.log1p(o))
    ok = agg["cdi"].notna() & agg["rate_chain"].notna()
    agg = agg[ok]
    out = out.merge(agg[[key, "AnoMes", "rate_posted"]], on=[key, "AnoMes"], how="left")
    hit = out["rate_posted"].notna()
    out.loc[hit, rate_col] = out.loc[hit, "rate_posted"]
    out.loc[hit, flag] = 1
    out = out.drop(columns=["rate_posted"])
    diff = agg[[key, "AnoMes", "rate_chain", "rate_posted", "covered", "rules"]].reset_index(drop=True)
    return out, diff
