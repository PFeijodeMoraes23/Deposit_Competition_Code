"""deposit_maturity_summary.py -- one bank-year summary of TIME deposits by remaining maturity,
from the two statement extractions (scrape_deposit_maturity_sec.py, scrape_deposit_maturity_cvm.py).

For each bank and fiscal year: the total, the share with no stated maturity, the shares maturing
within 90 and within 360 days, and the share beyond 360 days.
- Buckets that straddle a cut-off are split linearly in days.
- Only series whose buckets add up to the printed total within 0.5 percent are used.
- When a year has several tables with a time-deposit line, the largest is kept (the all-client
  table rather than a sub-portfolio), and `n_tables` records the ambiguity.

Reads  raw/DEPOSIT_MATURITY/deposit_maturity_sec.csv and deposit_maturity_cvm.csv.
Writes raw/DEPOSIT_MATURITY/deposit_maturity_summary.csv, the input of the deposit-type notes'
Section 4 and of switch_cost_r4_locked.py (the upper bound of the locked CDB share).

usage: python deposit_maturity_summary.py [FOLDER] [--out FILE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths

DEFAULT_DIR = paths.RAW / "DEPOSIT_MATURITY"


def load(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sec = pd.read_csv(folder / "deposit_maturity_sec.csv")
    cvm = pd.read_csv(folder / "deposit_maturity_cvm.csv")
    sec = sec[(sec["instrument"] == "time_deposits") & (sec["table_kind"] == "all_time_deposits")].copy()
    cvm = cvm[cvm["instrument"] == "time_deposits"].copy()
    for d in (sec, cvm):
        sc = d["sum_check"].astype(str).str.strip().str.lower()
        num = pd.to_numeric(d["sum_check"], errors="coerce")
        for c in ("lo_days", "hi_days", "value"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        # SEC stores the relative gap; CVM stores "ok"/"fail" against the same 0.5% rule.
        d["ok"] = (num <= 0.005) | (sc == "ok")
        d["table_id"] = d[[c for c in ("doc_url", "page", "table_title") if c in d.columns]].astype(str).agg("|".join, axis=1)
    return sec, cvm


def shares(g: pd.DataFrame) -> pd.Series:
    tot = g["value"].sum()
    out = {"total": tot, "n_buckets": len(g)}
    nomat_rows = (g["lo_days"].fillna(-1) == 0) & (g["hi_days"].fillna(-1) == 0)
    nomat = g.loc[nomat_rows, "value"].sum()
    out["share_no_maturity"] = nomat / tot if tot else np.nan
    rest = g[~nomat_rows]
    for cut in (90, 360):
        v = 0.0
        for _, r in rest.iterrows():
            lo = 0 if pd.isna(r["lo_days"]) else r["lo_days"]
            hi = np.inf if pd.isna(r["hi_days"]) else r["hi_days"]
            if hi <= cut:
                v += r["value"]
            elif lo < cut < hi and np.isfinite(hi):
                v += r["value"] * (cut - lo) / (hi - lo)
        out[f"share_le{cut}"] = v / tot if tot else np.nan
    out["share_gt360"] = 1 - out["share_no_maturity"] - out["share_le360"]
    return pd.Series(out)


def summarise(sec: pd.DataFrame, cvm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, d in (("sec_20f", sec), ("cvm", cvm)):
        d = d[d["ok"]]
        for (f, y), gg in d.groupby(["firm_key", "fiscal_year"]):
            tables = gg.groupby("table_id")["value"].sum()
            g = gg[gg["table_id"] == tables.idxmax()]
            s = shares(g)
            s["n_tables"] = len(tables)
            s["firm_key"], s["fiscal_year"], s["source"], s["unit"] = f, y, src, g["unit"].iloc[0]
            rows.append(s)
    out = pd.DataFrame(rows)[["firm_key", "fiscal_year", "source", "unit", "total", "n_buckets", "n_tables",
                              "share_no_maturity", "share_le90", "share_le360", "share_gt360"]]
    return out.sort_values(["firm_key", "fiscal_year", "source"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", nargs="?", default=str(DEFAULT_DIR))
    p.add_argument("--out", default=None, help="write here instead of FOLDER/deposit_maturity_summary.csv")
    a = p.parse_args()
    folder = Path(a.folder)
    out = summarise(*load(folder))
    dest = Path(a.out) if a.out else folder / "deposit_maturity_summary.csv"
    out.to_csv(dest, index=False, encoding="utf-8")
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
