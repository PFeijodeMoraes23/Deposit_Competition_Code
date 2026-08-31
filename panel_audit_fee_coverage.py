"""panel_audit_fee_coverage.py -- READ-ONLY audit of the fee columns in market_panel_with_fees.csv.

Author: Pedro Feijó de Moraes

Written after the 2026-08-31 fee inventory, which found that fee coverage failed
asymmetrically between the two firm types and that nothing in the pipeline could see it:

  * panel_fee_merge built its CNPJ -> conglomerate map from the LATEST IF Data List
    snapshot alone, so a firm whose prudential code changed mid-sample carried its
    current code on the fee side and its canonical code in the market panel, and every
    quarter before the rename failed to join;
  * unmapped CNPJs fell back to a bare "18236120" on the fee side against the panel's
    own "CNPJ_18236120", so institutions outside the prudential framework -- payment
    institutions, i.e. the D side -- could never match at all;
  * for the payment institutions that DID match, COSIF 717xxx books card interchange
    rather than deposit-account tariffs, so the ratio measured the wrong object -- and
    the obvious flag for them, has_ip, is 1 for any conglomerate that merely OWNS a
    payment institution, Bradesco and Banco do Brasil included;
  * panel_fee_merge's only guard was `len(merged) != n_orig`, which a total merge
    failure passes: a left join keeps every row and fills the fee columns with NaN.

This audit is the missing check. It reports, per firm type and year, how much of the
panel the fee merge actually reaches, and asserts floors under --strict.

Which panel this reads
----------------------
market_panel_with_fees.csv DIRECTLY, not utils.paths.market_panel_csv(). That inverts
panel_audit_mca_coverage's rule on purpose: the columns audited here EXIST only in the
fees panel, so resolving through market_panel_csv() would audit a file with no fee
columns whenever USE_FEE_PANEL is unset -- which is the default.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from utils import load_panel_cached, paths as _paths_mod

FEE_PANEL_CSV = _paths_mod.PROCESSED / "market_panel_with_fees.csv"
BASE_PANEL_CSV = _paths_mod.PROCESSED / "market_panel.csv"
COSIF_SOURCE = _paths_mod.TARIFAS_PROC / "cosif_service_fees_institution.parquet"
COSIF_QUARTERLY = _paths_mod.TARIFAS_PROC / "cosif_fee_quarterly_conglomerate.csv"

RATIO_COLS = ["cosif_fee_ratio_demand", "cosif_fee_ratio_savings", "cosif_fee_ratio_time",
              "cosif_fee_ratio_all", "cosif_fee_ratio_total_deposits"]
FLAG_COLS = ["cosif_fee_valid", "cosif_fee_guarded", "cosif_fee_ip_masked"]
STICK_COLS = ["tarifa_stickiness_yrs", "tarifa_stickiness_n"]

# service_fees_notes.md section 2.3: fee_ratio_total_deposits divides by the COSIF header
# total (account 41000007), which is never a thin sliver for any institution type, so it
# is the column to use.  A merge that leaves it far below the others is the failure this
# audit is looking for, whatever the headline cosif_fee_ratio_all coverage says.
HEADLINE_RATIO = "cosif_fee_ratio_total_deposits"

# Floors for --strict, set just under the levels measured on 2026-08-31 once the merge
# was fixed: 100.0% coverage of estimable B rows and 99.7% of 2016-2024 rows matched.
# Held a few points below so an ordinary data refresh does not trip them, but close
# enough that losing a large bank's fee history does.
FEE_MIN_B_COVERAGE = float(os.environ.get("FEE_MIN_B_COVERAGE", 0.95))
FEE_MIN_MATCH = float(os.environ.get("FEE_MIN_MATCH", 0.95))
# Window the coverage floor applies to: the estimation window, where a hole actually
# costs something.  2025+ is excluded by cosif_fee_valid anyway.
MATCH_WINDOW = (2016, 2024)

# Named firms and whether the licence test should mask them: an IP licence with no bank
# or credit-union charter.  Inter, C6, PicPay and PagSeguro each hold a Banco Multiplo.
SPOT_FIRMS = {
    "nu pagamentos": True,
    "mercado pago": True,
    "neon pagamentos": True,
    "picpay": False,
    "inter": False,
    "c6": False,
    "bradesco": False,
    "banco do brasil": False,
}


def _mtimes() -> dict:
    out = {}
    for p in (FEE_PANEL_CSV, BASE_PANEL_CSV, COSIF_SOURCE, COSIF_QUARTERLY):
        out[p.name] = datetime.fromtimestamp(p.stat().st_mtime) if p.exists() else None
    return out


def _fmt_share(x) -> str:
    return "     -" if not np.isfinite(x) else f"{x:>6.1%}"


def block_integrity(df: pd.DataFrame, fails: list) -> None:
    print("=" * 78)
    print("1. INTEGRITY AND STALENESS")
    print("=" * 78)
    mt = _mtimes()
    for name, ts in mt.items():
        print(f"  {name:<45s} {ts if ts else 'MISSING'}")

    if BASE_PANEL_CSV.exists():
        n_base = sum(1 for _ in open(BASE_PANEL_CSV, "rb")) - 1
        print(f"\n  rows: {FEE_PANEL_CSV.name} {len(df):,} | {BASE_PANEL_CSV.name} {n_base:,}")
        if n_base != len(df):
            fails.append(f"row count differs from the base panel ({len(df):,} vs {n_base:,}): "
                         f"the fee merge duplicated or dropped rows, or the two panels are "
                         f"different vintages")

    fee_t, base_t, cosif_t = (mt[FEE_PANEL_CSV.name], mt[BASE_PANEL_CSV.name],
                              mt[COSIF_SOURCE.name])
    if fee_t and base_t and fee_t < base_t:
        fails.append(f"{FEE_PANEL_CSV.name} is OLDER than {BASE_PANEL_CSV.name} -- it was "
                     f"built from a previous vintage of the base panel; re-run "
                     f"panel_fee_merge.py --patch-market")
    if fee_t and cosif_t and fee_t < cosif_t:
        fails.append(f"{FEE_PANEL_CSV.name} is OLDER than {COSIF_SOURCE.name} -- the COSIF "
                     f"source has been refreshed since; re-run panel_fee_merge.py")
    print()


def block_coverage(df: pd.DataFrame, cells: pd.DataFrame) -> None:
    print("=" * 78)
    print("2. MERGE MATCH AND NON-NULL COVERAGE, BY FIRM TYPE x YEAR")
    print("=" * 78)
    print("  matched = the COSIF cong-quarter joined at all (cosif_fee_guarded present);")
    print("  a row can be matched and still carry NaN ratios (guarded, or IP-masked).\n")

    hdr = (f"{'type':<5s}{'year':>6s}{'rows':>9s}{'matched':>9s}{'ratio_all':>11s}"
           f"{'ratio_totdep':>14s}{'listed':>9s}{'sticky':>9s}{'guarded':>9s}{'masked':>9s}")
    print(hdr)
    print("-" * len(hdr))
    listed_col = next((c for c in df.columns if c.startswith("listed_fee_")), None)
    for is_b, g_type in df.groupby("_type", sort=True):
        for year, g in g_type.groupby("year", sort=True):
            n = len(g)
            print(f"{is_b:<5s}{year:>6d}{n:>9,}"
                  f"{_fmt_share(g['_matched'].mean()):>9s}"
                  f"{_fmt_share(g['cosif_fee_ratio_all'].notna().mean()):>11s}"
                  f"{_fmt_share(g[HEADLINE_RATIO].notna().mean()):>14s}"
                  f"{_fmt_share(g[listed_col].notna().mean()) if listed_col else '     -':>9s}"
                  f"{_fmt_share(g['tarifa_stickiness_yrs'].notna().mean()):>9s}"
                  f"{_fmt_share((g['cosif_fee_guarded'] == 1).mean()):>9s}"
                  f"{_fmt_share((g['cosif_fee_ip_masked'] == 1).mean()):>9s}")
        print("-" * len(hdr))

    print("\n  Same, one observation per conglomerate x quarter (unweighted by municipality):")
    hdr2 = f"{'type':<5s}{'cells':>9s}{'matched':>9s}{'ratio_all':>11s}{'ratio_totdep':>14s}"
    print("  " + hdr2)
    print("  " + "-" * len(hdr2))
    for is_b, g in cells.groupby("_type", sort=True):
        print(f"  {is_b:<5s}{len(g):>9,}{_fmt_share(g['_matched'].mean()):>9s}"
              f"{_fmt_share(g['cosif_fee_ratio_all'].notna().mean()):>11s}"
              f"{_fmt_share(g[HEADLINE_RATIO].notna().mean()):>14s}")
    print()


def block_inversion(df: pd.DataFrame, cells: pd.DataFrame) -> None:
    print("=" * 78)
    print("3. WHY ratio_total_deposits AND ratio_all DIVERGE")
    print("=" * 78)
    print("  The two ratios share a numerator and differ only in the denominator, so a")
    print("  large 'ratio_all only' block means the 410 header total is going missing")
    print("  where the demand/savings/time sub-total survives.  That was the state until")
    print("  2026-08-31: aggregating deposits with 'last' took ONE arbitrary member's")
    print("  balance -- a DTVM with no deposits, for Itau and Bradesco -- against a")
    print("  numerator summed over every member.  Summing member balances instead closed")
    print("  it.  A non-zero 'ratio_all only' block here means that has regressed.\n")

    for label, frame in (("rows", df), ("cong-quarter cells", cells)):
        matched = frame[frame["_matched"]]
        both = matched[HEADLINE_RATIO].notna() & matched["cosif_fee_ratio_all"].notna()
        only_all = matched["cosif_fee_ratio_all"].notna() & matched[HEADLINE_RATIO].isna()
        only_td = matched[HEADLINE_RATIO].notna() & matched["cosif_fee_ratio_all"].isna()
        neither = matched[HEADLINE_RATIO].isna() & matched["cosif_fee_ratio_all"].isna()
        print(f"  {label}: {len(matched):,} matched")
        print(f"    both present            {both.sum():>9,} ({both.mean():>6.1%})")
        print(f"    ratio_all only          {only_all.sum():>9,} ({only_all.mean():>6.1%})")
        print(f"    ratio_total_deposits only {only_td.sum():>7,} ({only_td.mean():>6.1%})")
        print(f"    neither                 {neither.sum():>9,} ({neither.mean():>6.1%})")
        print(f"    unmatched (not shown)   {(~frame['_matched']).sum():>9,}\n")

    matched = df[df["_matched"]]
    gap = matched[matched["cosif_fee_ratio_all"].notna() & matched[HEADLINE_RATIO].isna()]
    if not gap.empty:
        top = (gap.rename(columns={"_name": "name"})
                  .groupby(["CodConglomeradoPrudencial", "name"])
                  .size().rename("panel_rows").reset_index()
                  .sort_values("panel_rows", ascending=False).head(10))
        print("  Top conglomerates contributing rows with ratio_all but no "
              "ratio_total_deposits:")
        print(f"    {'conglomerate':<14s}{'name':<38s}{'panel rows':>11s}")
        for r in top.itertuples(index=False):
            print(f"    {r.CodConglomeradoPrudencial:<14s}{str(r.name)[:36]:<38s}"
                  f"{r.panel_rows:>11,}")
        print("\n    These are matched rows, so the merge key is not the problem: the")
        print("    conglomerate reports a deposit sub-total but no 410 header. Check the")
        print("    deposit aggregation in panel_fee_merge.aggregate_cosif_to_")
        print("    conglomerate_quarter and the 410 extraction in "
              "scrape_cosif_service_fees.py.")
    else:
        print("  No matched row has ratio_all without ratio_total_deposits.")
    print()


def block_keyspace(df: pd.DataFrame, fails: list) -> None:
    print("=" * 78)
    print("4. KEY-FORMAT REGRESSION TEST")
    print("=" * 78)
    print("  Institutions outside the prudential framework are keyed 'CNPJ_<int>' by the")
    print("  market panel.  The fee sources must use the same spelling or they cannot join.\n")

    keys = df["CodConglomeradoPrudencial"].astype(str)
    fallback = keys.str.startswith("CNPJ_")
    n_keys = keys.nunique()
    fb_keys = sorted(keys[fallback].unique())
    print(f"  panel conglomerate keys: {n_keys:,} distinct, "
          f"{len(fb_keys)} of the CNPJ_ form")
    if fb_keys:
        matched_fb = df.loc[fallback].groupby("CodConglomeradoPrudencial")["_matched"].mean()
        for k in fb_keys:
            print(f"    {k:<18s} rows {int(fallback.sum() and (keys == k).sum()):>7,}"
                  f"   matched {matched_fb.get(k, float('nan')):>6.1%}")
        if float(df.loc[fallback, "_matched"].mean()) == 0.0:
            fails.append("no CNPJ_-keyed row matched any fee source -- the fee side is "
                         "still writing bare zero-padded CNPJs (see "
                         "panel_fee_merge.cnpj_fallback_key)")
    else:
        print("    (no CNPJ_ keys in the panel; nothing to test)")
    print()


def block_ip_mask(df: pd.DataFrame, fails: list, pre_mask: bool) -> None:
    print("=" * 78)
    print("5. PAYMENT-INSTITUTION MASK")
    print("=" * 78)
    masked = df["cosif_fee_ip_masked"] == 1
    leaked = int((masked & df[RATIO_COLS].notna().any(axis=1)).sum())
    print(f"  cosif_fee_ip_masked == 1 {int(masked.sum()):>9,} rows "
          f"({df.loc[masked, 'CodConglomeradoPrudencial'].nunique()} conglomerates)")
    print(f"  masked rows still carrying a COSIF ratio: {leaked:,}")
    if "has_ip" in df.columns:
        ip = pd.to_numeric(df["has_ip"], errors="coerce") == 1
        print(f"\n  For contrast, has_ip == 1 covers {int(ip.sum()):,} rows "
              f"({df.loc[ip, 'CodConglomeradoPrudencial'].nunique()} conglomerates,"
              f" {ip.mean():.0%} of the panel).")
        print("  has_ip is 1 whenever ANY member of the conglomerate is a payment")
        print("  institution, so it is 1 for Bradesco, Itau and Banco do Brasil. The mask")
        print("  uses the licence test in panel_fee_merge.load_nonbank_ip_conglomerates")
        print("  instead: an IP licence and no bank or credit-union charter.")
    if pre_mask:
        print("\n  (pre-mask panel: nothing masked yet)")
    elif leaked:
        fails.append(f"{leaked:,} masked rows still carry a COSIF fee ratio -- the mask "
                     f"in panel_fee_merge.patch_market_panel did not apply")

    print("\n  Named firms (these must appear as MATCHED, not silently absent):")
    print(f"    {'firm':<20s}{'rows':>8s}{'matched':>9s}{'masked':>8s}{'expect':>8s}"
          f"{'ratio_all':>10s}{'listed':>8s}")
    listed_col = next((c for c in df.columns if c.startswith("listed_fee_")), None)
    for firm, expect_masked in SPOT_FIRMS.items():
        sel = df["_name"].str.contains(firm, case=False, na=False)
        if not sel.any():
            print(f"    {firm:<20s}{'ABSENT FROM PANEL':>35s}")
            continue
        g = df[sel]
        got = float((g["cosif_fee_ip_masked"] == 1).mean())
        print(f"    {firm:<20s}{len(g):>8,}{g['_matched'].mean():>8.1%}{got:>8.1%}"
              f"{('mask' if expect_masked else 'keep'):>8s}"
              f"{g['cosif_fee_ratio_all'].notna().mean():>10.1%}"
              f"{(g[listed_col].notna().mean() if listed_col else float('nan')):>8.1%}")
        if not pre_mask and expect_masked and got < 1.0:
            fails.append(f"'{firm}' should be masked by the licence test but only "
                         f"{got:.0%} of its rows are")
        if not pre_mask and not expect_masked and got > 0.0:
            fails.append(f"'{firm}' holds a deposit-taking charter but {got:.0%} of its "
                         f"rows are masked")
    print()


def block_assertions(df: pd.DataFrame, fails: list) -> None:
    print("=" * 78)
    print("6. FLOORS")
    print("=" * 78)
    lo, hi = MATCH_WINDOW
    window = df[(df["year"] >= lo) & (df["year"] <= hi)]

    match_share = float(window["_matched"].mean()) if len(window) else float("nan")
    print(f"  cosif match, {lo}-{hi}, all rows        {match_share:>7.1%}  "
          f"floor {FEE_MIN_MATCH:.0%}")
    if np.isfinite(match_share) and match_share < FEE_MIN_MATCH:
        fails.append(f"COSIF merge reaches only {match_share:.1%} of {lo}-{hi} rows "
                     f"(floor {FEE_MIN_MATCH:.0%})")

    b_est = window[(window["_type"] == "B") & (window["cosif_fee_valid"] == 1) &
                   (window["cosif_fee_ip_masked"] != 1)]
    b_cov = float(b_est[HEADLINE_RATIO].notna().mean()) if len(b_est) else float("nan")
    print(f"  {HEADLINE_RATIO} on B rows  {b_cov:>7.1%}  floor {FEE_MIN_B_COVERAGE:.0%}")
    print(f"    (B, {lo}-{hi}, cosif_fee_valid == 1, not IP-masked: {len(b_est):,} rows)")
    if np.isfinite(b_cov) and b_cov < FEE_MIN_B_COVERAGE:
        fails.append(f"{HEADLINE_RATIO} covers only {b_cov:.1%} of estimable B rows "
                     f"(floor {FEE_MIN_B_COVERAGE:.0%})")
    print()


def main(strict: bool) -> int:
    if not FEE_PANEL_CSV.exists() and not FEE_PANEL_CSV.with_suffix(".parquet").exists():
        print(f"FAIL: {FEE_PANEL_CSV} not found.\n"
              f"Build it with: python panel_fee_merge.py --patch-market")
        return 1

    df = load_panel_cached(FEE_PANEL_CSV, dtype={"CodConglomeradoPrudencial": str})
    print(f"panel: {FEE_PANEL_CSV.name}  {len(df):,} rows x {len(df.columns)} cols\n")

    missing = [c for c in RATIO_COLS + STICK_COLS + ["cosif_fee_valid", "cosif_fee_guarded"]
               if c not in df.columns]
    if missing:
        print(f"FAIL: fee columns absent from the panel: {missing}\n"
              f"This file was built before those columns existed -- re-run "
              f"panel_fee_merge.py --patch-market.")
        return 1

    # A panel built before the payment-institution mask has no such column. Auditing it
    # is the point of a baseline run, so stand in a column of zeros and say so rather
    # than refusing to read the file.
    pre_mask = "cosif_fee_ip_masked" not in df.columns
    if pre_mask:
        print("NOTE: cosif_fee_ip_masked absent -- this panel predates the "
              "payment-institution mask.\n      Block 5 will report what the mask "
              "WOULD cover.\n")
        df["cosif_fee_ip_masked"] = 0

    df["CodConglomeradoPrudencial"] = df["CodConglomeradoPrudencial"].astype(str).str.strip()
    # cosif_fee_guarded comes from the COSIF panel itself, so it is present exactly when
    # the cong-quarter joined -- unlike the ratios, which are also NaN'd by the guard and
    # by the IP mask.  It is therefore the only honest test of whether the MERGE worked.
    df["_matched"] = df["cosif_fee_guarded"].notna()
    df["_type"] = np.where(pd.to_numeric(df.get("is_B"), errors="coerce") == 1, "B", "D")
    df["_name"] = (df["NomeInstituicao"].astype(str)
                   if "NomeInstituicao" in df.columns else "")

    cells = (df.sort_values(["CodConglomeradoPrudencial", "year", "quarter"])
               .drop_duplicates(subset=["CodConglomeradoPrudencial", "year", "quarter"]))

    fails: list[str] = []
    block_integrity(df, fails)
    block_coverage(df, cells)
    block_inversion(df, cells)
    block_keyspace(df, fails)
    block_ip_mask(df, fails, pre_mask)
    block_assertions(df, fails)

    print("=" * 78)
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED")
        for f in fails:
            print(f"  - {f}")
    else:
        print("All checks pass.")
    print("=" * 78)
    return 1 if (fails and strict) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any failure")
    sys.exit(main(ap.parse_args().strict))
