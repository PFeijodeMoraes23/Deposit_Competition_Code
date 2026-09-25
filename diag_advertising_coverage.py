"""
diag_advertising_coverage.py
============================
How much of the market panel the advertising data actually covers, in DEPOSIT terms.

The share of deposits matters more than the count of banks: a series covering twelve
conglomerates can be most of the market or a corner of it. Coverage is reported three ways,
because they answer different questions:

  quarterly, full year : all four quarters of the year observed for the conglomerate, from a
                         source that files quarterly. This is the set a within-year regressor
                         can use.
  quarterly, any        : at least one quarter observed. The upper bound on a quarterly series.
  plus annual           : the union with the SEC 20-F annual series, which cannot enter a
                         quarterly regression but does carry the four biggest private banks.

A source's canonical measure is fixed here (see MEASURE): the COSIF accounts give `adv`, the
filings `as_filed`, the banks' own files `adv_production`. The BB and BNB sponsorship series is
a robustness object, not advertising, so it is excluded.

Usage
-----
  python diag_advertising_coverage.py
  python diag_advertising_coverage.py --years 2016 2020 2024
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from utils import paths

PROC = paths.AWARENESS_PROC
MARKET_PANEL = Path(str(paths.market_panel_csv()).replace(".csv", ".parquet"))
OBSERVED = ("observed_positive", "observed_zero", "observed_negative")
MEASURE = {"cosif_conglomerate": "adv", "cosif_institution": "adv", "cvm": "as_filed",
           "caixa_own": "adv_production", "statebank_own": "adv_production"}


def deposits() -> pd.DataFrame:
    cols = ["CodConglomeradoPrudencial", "year", "quarter", "is_B", "dep_a1", "dep_a2", "dep_a4",
            "dep_a5"]
    mp = pd.read_parquet(MARKET_PANEL, columns=cols)
    mp["dep"] = mp[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].sum(axis=1, min_count=1)
    g = (mp.groupby(["CodConglomeradoPrudencial", "year", "quarter"])
           .agg(dep=("dep", "sum"), is_B=("is_B", "max")).reset_index())
    return g.rename(columns={"CodConglomeradoPrudencial": "panel_code"})


def observed_quarters(q: pd.DataFrame) -> pd.DataFrame:
    """One row per conglomerate-year: quarters observed by any quarterly source, and by source."""
    keep = pd.concat([q[(q["source"] == src) & (q["measure"] == m)] for src, m in MEASURE.items()])
    keep = keep[keep["value_status"].isin(OBSERVED)]
    # A quarter built from two of its three months is not an observed quarter: Banrisul's 2024Q4
    # has no November, the month whose PDF carries a broken font. The panel flags these; counting
    # them as observed would overstate coverage and understate the flow.
    bad = keep["flag_notes"].fillna("").str.contains("incomplete_quarter")
    if bad.any():
        log.info("excluded %d incomplete quarters (fewer than three months filed)", int(bad.sum()))
    keep = keep[~bad]
    per_source = (keep.groupby(["panel_code", "year", "source"])["quarter"].nunique()
                      .unstack(fill_value=0))
    any_source = keep.groupby(["panel_code", "year"])["quarter"].nunique().rename("n_quarters")
    return per_source.join(any_source, how="outer").fillna(0).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--out-dir", type=Path, default=PROC)
    a = ap.parse_args()

    q = pd.read_parquet(PROC / "advertising_panel_quarterly.parquet")
    ann = pd.read_parquet(PROC / "advertising_panel_annual.parquet")
    ann = ann[ann["source"] == "sec"] if "source" in ann else ann
    dep = deposits()
    obs = observed_quarters(q)

    # Q4 deposits are the denominator, matching the convention of the written record; the
    # full-year mean is reported beside it because a bank can enter or leave mid-year.
    q4 = dep[dep["quarter"] == 4].set_index(["panel_code", "year"])["dep"]
    yr = dep.groupby(["panel_code", "year"])["dep"].mean()

    years = a.years or sorted(int(y) for y in dep["year"].unique())
    rows = []
    for y in years:
        d4 = q4.xs(y, level="year", drop_level=True) if y in q4.index.get_level_values(1) else \
            pd.Series(dtype=float)
        dy = yr.xs(y, level="year", drop_level=True) if y in yr.index.get_level_values(1) else \
            pd.Series(dtype=float)
        o = obs[obs["year"] == y]
        full = set(o[o["n_quarters"] >= 4]["panel_code"])
        anyq = set(o[o["n_quarters"] >= 1]["panel_code"])
        sec = set(ann[ann["fiscal_year"] == y]["panel_code"].dropna()) if len(ann) else set()
        tot4, toty = float(d4.sum()), float(dy.sum())

        def share(codes: set, s: pd.Series, tot: float) -> float:
            return 100 * float(s.reindex([c for c in codes if c in s.index]).sum()) / tot \
                if tot else float("nan")

        rows.append({"year": y, "congl_in_panel": int(d4.notna().sum()),
                     "n_full_year": len(full & set(d4.index)),
                     "share_q4_full_year": share(full, d4, tot4),
                     "share_yearmean_full_year": share(full, dy, toty),
                     "n_any_quarter": len(anyq & set(d4.index)),
                     "share_q4_any_quarter": share(anyq, d4, tot4),
                     "n_plus_sec": len((full | sec) & set(d4.index)),
                     "share_q4_plus_sec": share(full | sec, d4, tot4),
                     "share_q4_sec_only": share(sec, d4, tot4)})
    cov = pd.DataFrame(rows)

    # Per-source shares of Q4 deposits, full-year coverage only.
    src_rows = []
    for y in years:
        d4 = q4.xs(y, level="year", drop_level=True) if y in q4.index.get_level_values(1) else \
            pd.Series(dtype=float)
        tot = float(d4.sum())
        o = obs[obs["year"] == y]
        for src in MEASURE:
            if src not in o:
                continue
            codes = set(o[o[src] >= 4]["panel_code"])
            src_rows.append({"year": y, "source": src, "n_conglomerates": len(codes),
                             "share_q4": 100 * float(d4.reindex(
                                 [c for c in codes if c in d4.index]).sum()) / tot if tot else 0.0})
    by_source = pd.DataFrame(src_rows)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    cov.to_csv(a.out_dir / "advertising_coverage_by_year.csv", index=False)
    by_source.to_csv(a.out_dir / "advertising_coverage_by_source.csv", index=False)

    log.info("\nCoverage of panel deposits (Q4 denominator), %%:\n%s",
             cov.round(1).to_string(index=False))
    log.info("\nFull-year coverage by source, share of Q4 deposits %%:\n%s",
             by_source.pivot(index="year", columns="source", values="share_q4").round(1)
                      .to_string())
    log.info("\nwrote %s and %s", a.out_dir / "advertising_coverage_by_year.csv",
             a.out_dir / "advertising_coverage_by_source.csv")


if __name__ == "__main__":
    main()
