"""
diag_openfinance_freshness.py
=============================
Empirically verify the Open Finance Brasil open-data update schedule.

The problem
-----------
OFB fee values are supposed to refresh monthly, published on the 10th BUSINESS day
(BCB IN 32/2020), referencing the previous month. But the API response carries NO
reference-period/timestamp field, so freshness cannot be read from the payload — it
must be INFERRED by watching values change across scrapes. This is why the collector
is run on a weekly sweep (see run_openfinance_monthly.py): several snapshots per month
let us see, per bank, on which day its monthly update actually landed.

What this does
--------------
Reads the appended long panel (openfinance_fees_panel_long.csv, all data_coleta
snapshots). For each institution (cnpj8) and service, detects when the customer-weighted
price changed between consecutive snapshots. Then, per (cnpj8, calendar month), finds
the FIRST snapshot in the month at which any service changed — the implied "update
landing" — and reports which BUSINESS day of the month that was (via utils.br_calendar).
Aggregates the distribution of implied publication business-days across banks so the
10th-business-day rule can be confirmed/refuted and the safe scrape buffer chosen.

Output
------
  BCB/Tarifas/processed/openfinance_freshness_report.csv
    cnpj8, org_name, month, n_snapshots, first_change_data_coleta,
    implied_business_day, implied_calendar_day, n_services_changed, status

Usage
-----
  python diag_openfinance_freshness.py
"""

from __future__ import annotations

import datetime as dt
import logging

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from utils import paths
from utils.br_calendar import business_day_index

LONG_CSV = paths.TARIFAS_PROC / "openfinance_fees_panel_long.csv"
OUT_CSV  = paths.TARIFAS_PROC / "openfinance_freshness_report.csv"

# A price is "changed" between two snapshots if it moves by more than this
# (guards against float-formatting jitter in the source strings).
EPS = 1e-6


def load_long() -> pd.DataFrame:
    if not LONG_CSV.exists():
        raise FileNotFoundError(f"{LONG_CSV} not found — run scrape_19 first.")
    df = pd.read_csv(LONG_CSV, low_memory=False, dtype={"cnpj8": str})
    df["cnpj8"] = df["cnpj8"].astype(str).str.zfill(8)
    # Restrict to per-event fees (back-compat: pre-multifamily rows had no price_kind).
    if "price_kind" in df.columns:
        df = df[df["price_kind"].fillna("per_event_brl") == "per_event_brl"].copy()
    df["price_weighted_avg"] = pd.to_numeric(df["price_weighted_avg"], errors="coerce")
    df["data_coleta"] = df["data_coleta"].astype(str)
    return df


def detect_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Per-snapshot change counts vs the immediately preceding snapshot, by cnpj8."""
    # Identify each service uniquely within a bank.
    svc_keys = ["cnpj8", "customer_type", "account_type", "service_code"]
    svc_keys = [c for c in svc_keys if c in df.columns]

    d = df.sort_values(svc_keys + ["data_coleta"]).copy()
    d["prev_price"] = d.groupby(svc_keys)["price_weighted_avg"].shift()
    d["prev_seen"] = d.groupby(svc_keys).cumcount() > 0
    # Changed = both observed and moved beyond EPS. Appearance/disappearance of a
    # service also counts as a change (schedule signal), except the very first obs.
    both = d["price_weighted_avg"].notna() & d["prev_price"].notna()
    moved = both & ((d["price_weighted_avg"] - d["prev_price"]).abs() > EPS)
    appeared = d["prev_seen"] & (both == False)
    d["changed"] = (moved | appeared).astype(int)

    per_snap = (
        d[d["prev_seen"]]                      # drop each service's first-ever obs
        .groupby(["cnpj8", "data_coleta"], as_index=False)
        .agg(n_services=("service_code", "nunique"),
             n_changed=("changed", "sum"))
    )
    return per_snap


def build_report(df: pd.DataFrame, per_snap: pd.DataFrame) -> pd.DataFrame:
    names = (df.groupby("cnpj8")["org_name"].first()
             if "org_name" in df.columns else None)

    per_snap = per_snap.copy()
    per_snap["month"] = per_snap["data_coleta"].str.slice(0, 7)
    per_snap = per_snap.sort_values(["cnpj8", "data_coleta"])

    # snapshots per (cnpj8, month)
    n_snap = (per_snap.groupby(["cnpj8", "month"])["data_coleta"]
              .nunique().rename("n_snapshots").reset_index())

    # first snapshot in the month with a detected change
    changed = per_snap[per_snap["n_changed"] > 0]
    first_change = (changed.sort_values("data_coleta")
                    .groupby(["cnpj8", "month"], as_index=False)
                    .first()
                    .rename(columns={"data_coleta": "first_change_data_coleta",
                                     "n_changed": "n_services_changed"}))

    rep = n_snap.merge(first_change[["cnpj8", "month", "first_change_data_coleta",
                                     "n_services_changed"]],
                       on=["cnpj8", "month"], how="left")

    def _bd(ds):
        if isinstance(ds, str) and len(ds) >= 10:
            try:
                return business_day_index(dt.date.fromisoformat(ds[:10]))
            except ValueError:
                return np.nan
        return np.nan

    rep["implied_business_day"] = rep["first_change_data_coleta"].map(_bd)
    rep["implied_calendar_day"] = rep["first_change_data_coleta"].map(
        lambda s: int(s[8:10]) if isinstance(s, str) and len(s) >= 10 else np.nan)

    def _status(row):
        if pd.isna(row["first_change_data_coleta"]):
            return "stable" if row["n_snapshots"] >= 2 else "insufficient"
        return "changed"
    rep["status"] = rep.apply(_status, axis=1)

    if names is not None:
        rep["org_name"] = rep["cnpj8"].map(names)

    cols = ["cnpj8", "org_name", "month", "n_snapshots", "first_change_data_coleta",
            "implied_business_day", "implied_calendar_day", "n_services_changed", "status"]
    rep = rep[[c for c in cols if c in rep.columns]].sort_values(["month", "cnpj8"])
    return rep


def summarize(rep: pd.DataFrame) -> None:
    changed = rep[rep["status"] == "changed"]
    n_months = rep["month"].nunique()
    log.info("Freshness report: %d (bank,month) cells | changed=%d, stable=%d, insufficient=%d",
             len(rep), (rep["status"] == "changed").sum(),
             (rep["status"] == "stable").sum(),
             (rep["status"] == "insufficient").sum())
    if n_months < 2:
        log.warning("Only %d calendar month(s) of snapshots present — any 'change' below is "
                    "INTRA-month jitter, NOT a monthly update. The implied business-day is NOT a "
                    "valid schedule measurement yet; accumulate weekly snapshots across >=2 months.",
                    n_months)
    if changed.empty:
        log.info("No cross-snapshot changes detected yet. Accumulate weekly snapshots "
                 "across at least two months for the schedule to become observable.")
        return
    bd = changed["implied_business_day"].dropna()
    if not bd.empty:
        log.info("Implied publication business-day distribution (rule says ~10):")
        log.info("  min=%.0f  p25=%.0f  median=%.0f  p75=%.0f  max=%.0f  mean=%.1f",
                 bd.min(), bd.quantile(.25), bd.median(), bd.quantile(.75), bd.max(), bd.mean())
        by_bd10 = (bd <= 10).mean() * 100
        log.info("  %.0f%% of updates landed by the 10th business day; "
                 "worst-case = business day %.0f (=> pick a scrape buffer beyond it).",
                 by_bd10, bd.max())


def main() -> None:
    df = load_long()
    log.info("Loaded %d per-event rows | %d cnpj8 | %d snapshots (%s)",
             len(df), df["cnpj8"].nunique(), df["data_coleta"].nunique(),
             sorted(df["data_coleta"].unique()))
    per_snap = detect_changes(df)
    rep = build_report(df, per_snap)
    rep.to_csv(OUT_CSV, index=False)
    log.info("Saved: %s (%d rows)", OUT_CSV.name, len(rep))
    summarize(rep)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
