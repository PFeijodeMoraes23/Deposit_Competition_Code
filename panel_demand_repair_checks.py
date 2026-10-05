"""
panel_demand_repair_checks.py
=============================
Tests of `processed/demand_repair.parquet` (written by panel_demand_repair.py) against the files
it was built from. One identity or count per repaired quantity; each prints PASS or FAIL and the
script exits non-zero if any fails.

  T1  balances of 2013-2015: the stored balance over the raw ESTBAN file is exactly 2 in every
      municipality of 2015-12 and exactly 1 in 2016-03; every repaired balance is the stored one
      divided by that factor; the repaired 2015Q4 total sits on the raw file's scale
  T2  2015Q4 characteristics: each spliced value times the members' one-quarter change returns the
      firm's 2016Q1 filing; the back-test (2016Q1 predicted from 2016Q2) stays within the error
      bounds recorded when the method was chosen
  T3  true-lag fills equal the firm's filing at t-1
  T4  nearest-filing fills equal the firm's nearest filing, carry its distance (1 to the maximum),
      and no closer filing exists; a tie went to the earlier quarter
  T5  segments: a back-filled segment is the firm's first recorded one; a recorded segment is
      untouched; a never-classified firm is S1 and flagged; the dummies match the segment
  T6  the borrowings ratio: at one date it never exceeds 1; no infinite value; its leave-one-out
      pair is finite on every row
  T7  flags: a repaired value never comes without its flag, nor a flag without its value
  T8  rival branches of 2015Q4: equal to a direct count from the raw 2015-12 file
  T9  applied to the panel (all four stages): one row per key; the only rows of 2016-2024 left
      without a lagged log of assets are those flagged for dropping

Reads   processed/demand_repair.parquet, market_panel.parquet, bank_chars_panel.csv,
        deposits_panel.csv, the raw IF.data and ESTBAN files panel_demand_repair.py reads
Writes  nothing

Usage:  python panel_demand_repair_checks.py
"""
from __future__ import annotations

import sys

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd

from utils import demand_repair as dr
import panel_demand_repair as b

C = b.C
RESULTS = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> int:
    R = pd.read_parquet(b.OUT_PARQUET)
    R[C] = R[C].astype(str); R["mca_code"] = R["mca_code"].astype(str)
    panel = b.load_panel()
    bc = b.load_bank_chars()
    memb = b.member_map()
    win = R[R["row_kind"] == "window"].merge(panel, on=dr.KEYS, how="left", suffixes=("", "_panel"))
    win["tq"] = b.tq(win["year"], win["quarter"])
    fq = win.drop_duplicates([C, "year", "quarter"]).copy()
    cuts = b.stored_cutoffs(panel)

    # ---------------------------------------------------------------- T1
    dp = pd.read_csv(b.DEPOSITS_CSV, low_memory=False,
                     usecols=["CODMUN_IBGE", "Year", "Quarter", "Source"] + dr.DEP_COLS)
    dp = dp[dp["Source"] == "ESTBAN"]
    for ym, (y, q), want in (("201512", (2015, 4), 2.0), ("201603", (2016, 1), 1.0)):
        raw = b.read_raw_estban(ym, b.DEP_SUBSTR).groupby("CODMUN_IBGE")[dr.DEP_COLS].sum()
        pan = dp[(dp["Year"] == y) & (dp["Quarter"] == q)].groupby("CODMUN_IBGE")[dr.DEP_COLS].sum()
        j = raw.join(pan, lsuffix="_raw", rsuffix="_pan", how="outer")
        r = np.concatenate([(j.loc[j[f"{c}_raw"] > 0, f"{c}_pan"] / j.loc[j[f"{c}_raw"] > 0, f"{c}_raw"]).to_numpy()
                            for c in dr.DEP_COLS])
        check(f"T1 stored over raw, {ym}", np.all(np.abs(r - want) < 1e-9),
              f"{len(j)} municipalities, {len(r)} cells, all equal to {want:g} (worst deviation {np.abs(r - want).max():.1e})")
    pre = R[R["row_kind"] == "pre2016"].merge(panel[dr.KEYS + dr.DEP_COLS], on=dr.KEYS, how="left")
    worst = max(float(np.nanmax(np.abs(pre["r__" + c] * 2.0 - pre[c]))) for c in dr.DEP_COLS)
    check("T1 repaired balance = stored / 2", len(pre) == int((panel["year"] < 2016).sum()) and worst == 0.0,
          f"{len(pre):,} rows of 2013-2015, every one; largest |2 x repaired - stored| = {worst:g}")
    raw15 = b.read_raw_estban("201512", b.DEP_SUBSTR)[dr.DEP_COLS].sum()
    raw16 = b.read_raw_estban("201603", b.DEP_SUBSTR)[dr.DEP_COLS].sum()
    p15 = pre[(pre["year"] == 2015) & (pre["quarter"] == 4)]
    p16 = panel[(panel["year"] == 2016) & (panel["quarter"] == 1) & (panel["Source"] == "ESTBAN")]
    r15 = {c: float(p15["r__" + c].sum() / raw15[c]) for c in dr.DEP_COLS}
    r16 = {c: float(p16[c].sum() / raw16[c]) for c in dr.DEP_COLS}
    check("T1 repaired 2015Q4 total on the raw scale", all(abs(r15[c] - r16[c]) < 0.03 and r15[c] < 1.0 + 1e-9 for c in dr.DEP_COLS),
          "panel total over raw total, repaired 2015Q4 " + ", ".join(f"{c} {r15[c]:.4f}" for c in dr.DEP_COLS)
          + " | stored 2016Q1 " + ", ".join(f"{c} {r16[c]:.4f}" for c in dr.DEP_COLS)
          + " (below 1 because the panel leaves out some reporting firms)")

    # ---------------------------------------------------------------- T2
    m15_9, m15_12, m16_3 = b.member_sums(2015, 9, memb), b.member_sums(2015, 12, memb), b.member_sums(2016, 3, memb)
    mem_prev, mem_T = b.member_chars(b.quarter_flows(m15_9, m15_12)), b.member_chars(m16_3)
    own = bc[(bc["year"] == 2016) & (bc["quarter"] == 1)].set_index(C)
    s = fq[fq["chars_src"] == 2].set_index(C)
    d_log = (s["r__log_total_assets_lag"] + (mem_T["log_total_assets"].reindex(s.index) - mem_prev["log_total_assets"].reindex(s.index))
             - own["log_total_assets"].reindex(s.index)).abs()
    check("T2 spliced log assets return the 2016Q1 filing", len(s) > 0 and float(d_log.max()) < 1e-9,
          f"{len(s)} firm-quarters of 2016Q1; largest |spliced + members' change - filing| = {float(d_log.max()):.1e}")
    ip, it = mem_prev["equity_ratio"].reindex(s.index), mem_T["equity_ratio"].reindex(s.index)
    ok = ip.notna() & it.notna() & (ip.abs() > 0) & (it.abs() > 0)
    d_eq = (s.loc[ok, "r__equity_ratio_lag"] * it[ok] / ip[ok] - own["equity_ratio"].reindex(s.index)[ok]).abs()
    check("T2 spliced equity ratio returns the 2016Q1 filing", float(d_eq.max()) < 1e-9,
          f"{int(ok.sum())} firm-quarters with a member ratio at both dates; largest deviation {float(d_eq.max()):.1e}")
    check("T2 Basel index of 2015Q4 is the firm's own 2016Q1 value, flagged",
          bool((s["basel_imputed"] == 1).all()) and bool((fq.loc[fq["chars_src"] != 2, "basel_imputed"] == 0).all()),
          f"{int((s['basel_imputed'] == 1).sum())} firm-quarters flagged; none outside the 2015Q4 values")
    # back-test on the firms of the 2016Q1 panel, by firm type
    m3, m6 = m16_3, b.member_sums(2016, 6, memb)
    own_T = bc[(bc["year"] == 2016) & (bc["quarter"] == 2)].set_index(C)[b.SPLICED]
    truth = bc[(bc["year"] == 2016) & (bc["quarter"] == 1)].set_index(C)[b.SPLICED]
    pred, _ = b.splice_back(own_T, b.member_chars(m3), b.member_chars(b.quarter_flows(m3, m6)))
    q1 = panel[(panel["year"] == 2016) & (panel["quarter"] == 1)].groupby(C)["is_B"].max()
    bounds = {("log_total_assets", 1): (0.005, 0.03), ("log_total_assets", 0): (0.005, 0.03),
              ("equity_ratio", 1): (0.001, 0.006), ("personnel_cost_ratio", 1): (0.0001, 0.0005),
              ("admin_cost_ratio", 1): (0.0001, 0.0006)}
    for (v, t), (bm, bp) in bounds.items():
        idx = [c for c in q1.index[q1 == t] if c in pred.index and c in truth.index]
        e = (pred.loc[idx, v] - truth.loc[idx, v]).abs().dropna()
        ec = (own_T.loc[idx, v] - truth.loc[idx, v]).abs().dropna()
        check(f"T2 back-test {v}, {'B' if t else 'D'} firms", len(e) > 20 and e.median() <= bm and e.quantile(.9) <= bp,
              f"n {len(e)}; splice median / p90 abs. error {e.median():.5f} / {e.quantile(.9):.5f} (bounds {bm} / {bp}); "
              f"value of 2016Q2 carried back {ec.median():.5f} / {ec.quantile(.9):.5f}")

    # ---------------------------------------------------------------- T3, T4
    lv = bc[(bc["total_assets"] > 0) & (bc["year"] >= 2016)]
    by_firm = {c: g.set_index("tq") for c, g in lv.groupby(C)}
    plain = ["log_total_assets_lag", "equity_ratio_lag", "npl_provision_ratio_lag", "total_assets_lag"]

    def source_row(code, t, dist):
        g = by_firm[code]
        cands = [q for q in (t - 1 - dist, t - 1 + dist) if q in g.index]      # earlier first
        closer = [q for q in g.index if abs(q - (t - 1)) < dist]
        return (g.loc[cands[0]] if cands else None), len(closer)

    for src, name in ((1, "T3 true-lag fills"), (3, "T4 nearest-filing fills")):
        sub = fq[fq["chars_src"] == src]
        bad, bad_closer, n_tie = 0, 0, 0
        for r in sub.itertuples(index=False):
            row, n_closer = source_row(getattr(r, C), int(r.tq), int(r.chars_dist))
            bad_closer += n_closer
            if row is None:
                bad += 1
                continue
            g = by_firm[getattr(r, C)]
            n_tie += int((int(r.tq) - 1 - int(r.chars_dist)) in g.index and (int(r.tq) - 1 + int(r.chars_dist)) in g.index and r.chars_dist > 0)
            for col in plain:
                a, bb = getattr(r, "r__" + col), row[b.LEVEL_OF[col]]
                if not (np.isclose(a, bb, rtol=1e-12, atol=0, equal_nan=True)):
                    bad += 1
            for col in b.WINSORIZED:
                lo, hi = cuts[(col, int(r.is_B))]
                want = row[b.LEVEL_OF[col]]
                if col == "indice_basileia_lag":
                    want = min(max(want, b.BASEL_BOUNDS[0]), b.BASEL_BOUNDS[1])
                want = min(max(want, lo), hi) if np.isfinite(want) else want
                if not np.isclose(getattr(r, "r__" + col), want, rtol=1e-12, atol=0, equal_nan=True):
                    bad += 1
        dist_ok = bool((sub["chars_dist"] == 0).all()) if src == 1 else bool(sub["chars_dist"].between(1, b.MAX_DISTANCE).all())
        check(name, bad == 0 and bad_closer == 0 and dist_ok,
              f"{len(sub)} firm-quarters; values differing from the source filing {bad}; closer filings overlooked {bad_closer}; "
              + ("all at t-1" if src == 1 else f"distances {sub['chars_dist'].value_counts().sort_index().to_dict()}; ties resolved to the earlier quarter {n_tie}"))
    drop = fq[fq["chars_src"] == dr.CHARS_DROPPED]
    far = 0
    for r in drop.itertuples(index=False):
        g = by_firm.get(getattr(r, C))
        if g is not None and np.abs(np.asarray(g.index) - (int(r.tq) - 1)).min() <= b.MAX_DISTANCE:
            far += 1
    check("T4 firm-quarters flagged for dropping have no filing within the maximum distance", far == 0,
          f"{len(drop)} firm-quarters in {drop[C].nunique()} firms; with a filing within {b.MAX_DISTANCE} quarters: {far}")

    # ---------------------------------------------------------------- T5
    seg = bc.dropna(subset=["segment"]).sort_values([C, "tq"])
    first = seg.drop_duplicates(C, keep="first").set_index(C)["segment"]
    s1 = fq[fq["segment_src"] == 1]
    check("T5 back-filled segment = the firm's first recorded segment",
          bool((s1["r__segment"].to_numpy() == first.reindex(s1[C]).to_numpy()).all()),
          f"{len(s1)} firm-quarters in {s1[C].nunique()} firms; become {s1['r__segment'].value_counts().sort_index().to_dict()}")
    s0 = fq[fq["segment_src"] == 0]
    same = (s0["r__segment"] == s0["segment"]).all() and all((s0["r__" + c] == s0[c]).all() for c in dr.SEG_COLS)
    check("T5 recorded segments untouched", bool(same), f"{len(s0)} firm-quarters")
    s3 = fq[fq["segment_src"] == 3]
    check("T5 never-classified firms stay S1, flagged",
          bool((s3["r__segment"] == "S1").all()) and not s3[C].isin(first.index).any(),
          f"{len(s3)} firm-quarters in {s3[C].nunique()} firms; none of them has a recorded segment in any quarter")
    dsum = fq[["r__" + c for c in dr.SEG_COLS]].sum(axis=1)
    cons = all(((fq["r__segment"] == c[-2:]).astype(float) == fq["r__" + c]).all() for c in dr.SEG_COLS)
    check("T5 dummies match the segment", bool(cons) and bool(dsum.le(1).all()) and not fq[["r__" + c for c in dr.SEG_COLS]].isna().any().any(),
          f"no missing dummy on {len(fq):,} firm-quarters; at most one dummy per row")

    # ---------------------------------------------------------------- T6
    bl = fq["borrow_assets_lag"].to_numpy(float)
    keep = fq["chars_src"] != dr.CHARS_DROPPED
    check("T6 borrowings ratio: no infinity, never above 1", not np.isinf(bl).any() and np.nanmax(bl) <= 1.0,
          f"max {np.nanmax(bl):.3f}; missing on {int(np.isnan(bl).sum())} firm-quarters, "
          f"{int((np.isnan(bl) & keep.to_numpy()).sum())} of them outside the firm-quarters flagged for dropping")
    lb = win[["loo_borrow_assets", "mean_loo_borrow_assets"]].to_numpy(float)
    check("T6 leave-one-out pair of the ratio finite on every row", bool(np.isfinite(lb).all()),
          f"{len(win):,} rows; stored pair infinite or missing on {int((~np.isfinite(win['loo_credit_assets'].to_numpy(float))).sum()):,}")
    check("T6 changed finite stored values are flagged",
          int(fq["borrow_changed"].sum()) == int((np.isfinite(fq["credit_assets_lag"].to_numpy(float))
                                                  & ~(np.isfinite(bl) & (np.abs(bl - fq["credit_assets_lag"].to_numpy(float)) <= 1e-12))).sum()),
          f"{int(fq['borrow_changed'].sum())} firm-quarters flagged borrow_changed; {int(fq['borrow_den_guarded'].sum())} flagged borrow_den_guarded")

    # ---------------------------------------------------------------- T7
    has_val = win[["r__" + c for c in dr.CHAR_COLS]].notna().any(axis=1)
    flagged = win["chars_src"].isin([1, 2, 3])
    check("T7 characteristics: value <=> flag", bool((has_val == flagged).all()),
          f"rows with a repaired characteristic {int(has_val.sum()):,}; rows flagged 1, 2 or 3 {int(flagged.sum()):,}")
    check("T7 rival branches: value <=> flag", bool((win["r__estban_rival_branches_lag"].notna() == (win["estban_lag_src"] == 1)).all()),
          f"{int((win['estban_lag_src'] == 1).sum()):,} rows, all of 2016Q1: {bool((win.loc[win['estban_lag_src'] == 1, 'lag_from_2015q4'] == 1).all())}")
    fl = win[dr.FLAG_COLS]
    check("T7 every row of 2016-2024 carries every flag", not fl.isna().any().any(), f"{len(win):,} rows x {len(dr.FLAG_COLS)} flags")

    # ---------------------------------------------------------------- T8
    rb = b.rival_branches("201512")
    q = win[(win["estban_lag_src"] == 1)][[C, "mca_code", "r__estban_rival_branches_lag"]].merge(rb, on=[C, "mca_code"], how="left")
    check("T8 rival branches of 2015Q4 = direct count from the raw file",
          bool(np.allclose(q["r__estban_rival_branches_lag"], np.log1p(q["rival_br"]), rtol=0, atol=1e-12)),
          f"{len(q):,} rows; zero on {int((q['rival_br'] == 0).sum()):,} (no rival branch in the market)")

    # ---------------------------------------------------------------- T9
    full = pd.read_parquet(b.PANEL_PARQUET)
    full["mca_code"] = full["mca_code"].astype(str)
    out = dr.apply(full, explicit_stages=(1, 2, 3, 4))
    w = out[out["year"].between(b.MIN_YEAR, b.MAX_YEAR)]
    miss = w["log_total_assets_lag"].isna()
    n_add = int((R["row_kind"] == "added_2015q4").sum())
    check("T9 applied panel: one row per key, rows = panel + appended", not out.duplicated(dr.KEYS).any() and len(out) == len(full) + n_add,
          f"{len(out):,} rows = {len(full):,} + {n_add}")
    check("T9 applied panel: a missing lagged log of assets only where flagged for dropping",
          bool((miss == (w["chars_src"] == dr.CHARS_DROPPED)).all()) and not (w["log_total_assets_lag"] == 0).any(),
          f"rows of 2016-2024 without it: {int(miss.sum()):,}, all flagged 9; stored zeros left: {int((w['log_total_assets_lag'] == 0).sum())}")
    untouched = [c for c in full.columns if c not in dr.CHAR_COLS + dr.SEG_COLS + dr.LOO_COLS + dr.DEP_COLS
                 + ["segment", "estban_rival_branches_lag"] + dr.FLAG_COLS + dr.BORROW_COLS]
    a, o = full[untouched], out.iloc[:len(full)][untouched]
    same_cols = all((a[c].to_numpy() == o[c].to_numpy()).all() or a[c].equals(o[c].reset_index(drop=True)) for c in untouched)
    check("T9 applied panel: every other column of the panel is unchanged", bool(same_cols), f"{len(untouched)} columns compared")

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail} of {len(RESULTS)} checks passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
