"""
bbl_probe_compare.py
============================
Does the deposit-rate stability cap remove the explosive psi rows (a few firms whose psi sit orders
of magnitude above the rest) without moving the typical firms?

Reads, for ONE shard of ONE routine, the forward-simulation psi of three tags, in the layout
bbl_fwd_sim.jl writes and bbl_solve.py reads:

    psi_dev_<key><tag>_shard<i>of<N>.parquet   shock, firm, is_B, start_q (multi-start only),
                                               psi1, psi2_omega, psi3_gamma_<z>..., psi4_zeta
    psi_eq_<key><tag>.parquet                  firm, is_B, start_q, rf_source, rf_bar_beta, the
                                               same psi blocks. Written by shard 0 only, so a
                                               probe of shard 1 has none; it is read where it exists.

The default tags (--tags):
    _ms1          beta=0.9,   T=50    the production multi-start run: the BASELINE
    _probe250     beta=0.979, T=250   no cap (the 2026-09-24 memory probe ran shards 1 and 8)
    _probe250cap  beta=0.979, T=250   with the deposit-rate stability cap (the code probe)

For every psi component (psi1, psi2_omega, psi4_zeta, then the psi3_gamma_<z>) and every tag it
prints, across the shard's firm x start x shock rows: max |psi|, the 99th percentile and the
median of |psi|, each also as a ratio to the baseline's value. Then, on psi2_omega:
  - the rows and firms whose |psi2| exceeds --explosive-mult (100) x that tag's own median;
  - the --top (10) firms by max |psi2| under the uncapped tag, and the same firms under the
    capped tag and the baseline;
  - row for row (matched on shock, firm, start_q), how far the cap moved each component, over all
    rows and over the TYPICAL rows only (those not explosive under the uncapped tag);
and a four-line SUMMARY that answers the question above in numbers.

The beta/T labels above are the defaults' designs; each tag's psi_starts sidecar, where one
exists, is printed as the file's own record. Reads only; writes nothing.

Runs on the cluster in the BBL solve's Python environment (numpy, pandas, pyarrow):
    python bbl_probe_compare.py --key E3_spec_12_extended --shard 1 --n-shards 300
The folder is --dir, else $CF_COST_FWD, else $CL_STEP_BBL, else ../data/output/bbl beside this file.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROW_KEY = ("shock", "firm", "start_q")
DESIGN = {"_ms1": "beta=0.9   T=50  baseline",
          "_probe250": "beta=0.979 T=250 no cap",
          "_probe250cap": "beta=0.979 T=250 capped"}
REL_SAME = 1e-12      # |cap - uncapped| <= this x |uncapped| counts as unchanged
REL_MOVED = 1e-6      # ... and above this as moved, in the SUMMARY


def default_dir() -> Path:
    for v in ("CF_COST_FWD", "CL_STEP_BBL"):
        if os.environ.get(v):
            return Path(os.environ[v])
    return Path(__file__).resolve().parent.parent / "data" / "output" / "bbl"


def psi_columns(df: pd.DataFrame) -> list:
    """psi1, psi2_omega, psi4_zeta first, then every other psi* block (the psi3_gamma_<z>) in the
    writer's order. rf_bar_beta and the key columns are not psi."""
    first = [c for c in ("psi1", "psi2_omega", "psi4_zeta") if c in df.columns]
    return first + [c for c in df.columns if c.startswith("psi") and c not in first]


def abs_stats(x) -> dict:
    """max, p99, median of |x| over its FINITE values, and how many were not finite."""
    x = np.asarray(x, float)
    fin = np.isfinite(x)
    a = np.abs(x[fin])
    if a.size == 0:
        return dict(max=np.nan, p99=np.nan, med=np.nan, n=0, nonfin=int((~fin).sum()))
    return dict(max=float(a.max()), p99=float(np.percentile(a, 99)), med=float(np.median(a)),
                n=int(a.size), nonfin=int((~fin).sum()))


def ratio(a, b) -> float:
    return float(a / b) if (np.isfinite(a) and np.isfinite(b) and b != 0) else np.nan


def f(v, spec=".3e") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, float) and np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return format(v, spec)


def load(path: Path):
    if not path.is_file():
        return None
    df = pd.read_parquet(path)
    if "start_q" in df.columns:
        df["start_q"] = df["start_q"].astype(str)
    return df


def sidecar(bbl_dir: Path, key: str, tag: str) -> str:
    p = bbl_dir / f"psi_starts_{key}{tag}.json"
    if not p.is_file():
        return "no psi_starts sidecar"
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        return f"psi_starts: beta={m.get('beta')} T={m.get('T')} S={m.get('S')} P={m.get('P')}"
    except (OSError, ValueError) as exc:
        return f"psi_starts unreadable ({exc})"


def stats_table(title, frames, base, cols):
    """One block per component: max / p99 / median of |psi| per tag, and each over the base's."""
    print(f"\n== {title} ==")
    for c in cols:
        have = [t for t, df in frames.items() if df is not None and c in df.columns]
        if not have:
            continue
        st = {t: abs_stats(frames[t][c]) for t in have}
        b = st.get(base)
        print(f"\n  {c}: |psi| across rows")
        print(f"  {'tag':<14} {'rows':>7} {'max':>11} {'p99':>11} {'median':>11} {'non-fin':>7}"
              f" {'max/base':>10} {'p99/base':>10} {'med/base':>10}")
        for t in have:
            s = st[t]
            r = [ratio(s[k], b[k]) if b else np.nan for k in ("max", "p99", "med")]
            print(f"  {t:<14} {s['n'] + s['nonfin']:>7} {f(s['max']):>11} {f(s['p99']):>11} "
                  f"{f(s['med']):>11} {s['nonfin']:>7} {f(r[0], '.3g'):>10} {f(r[1], '.3g'):>10} "
                  f"{f(r[2], '.3g'):>10}")


def firm_max(df: pd.DataFrame, col: str) -> pd.Series:
    """Per firm, max |col| over its rows; a non-finite value ranks as +inf."""
    a = np.abs(df[col].to_numpy(float))
    a = np.where(np.isfinite(a), a, np.inf)
    return pd.Series(a, index=df.index).groupby(df["firm"]).max()


def main():
    ap = argparse.ArgumentParser(description="Compare the psi of a capped probe shard with the "
                                             "uncapped probe and the baseline.")
    ap.add_argument("--dir", type=Path, default=None, help="the BBL step folder (default: see docstring)")
    ap.add_argument("--key", default="E3_spec_12_extended", help="E{k}_spec_12_{stage}, without the tag")
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--n-shards", type=int, default=300)
    ap.add_argument("--tags", default="_ms1,_probe250,_probe250cap",
                    help="BASELINE,UNCAPPED,CAPPED (comma-separated, in that order)")
    ap.add_argument("--col", default="psi2_omega", help="the component the explosive / top-firm "
                                                         "analysis ranks on")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--explosive-mult", type=float, default=100.0,
                    help="a row is explosive when |col| > this x the tag's own median |col|")
    a = ap.parse_args()

    bbl_dir = a.dir or default_dir()
    tags = [t.strip() for t in a.tags.split(",")]
    if len(tags) != 3:
        sys.exit(f"--tags needs exactly three tags (BASELINE,UNCAPPED,CAPPED), got {tags}")
    base, unc, cap = tags
    shard = f"_shard{a.shard}of{a.n_shards}"
    print(f"bbl_probe_compare: {bbl_dir}")
    print(f"  key {a.key}, shard {a.shard} of {a.n_shards}; baseline {base}, uncapped {unc}, capped {cap}")

    dev, eq = {}, {}
    for t in tags:
        p_dev = bbl_dir / f"psi_dev_{a.key}{t}{shard}.parquet"
        p_eq = bbl_dir / f"psi_eq_{a.key}{t}.parquet"
        dev[t], eq[t] = load(p_dev), load(p_eq)
        if dev[t] is not None:
            ks = [k for k in ROW_KEY if k in dev[t].columns]
            n0 = len(dev[t])
            dev[t] = dev[t].drop_duplicates(subset=ks).reset_index(drop=True)
            what = f"{len(dev[t])} rows, {dev[t]['firm'].nunique()} firms"
            if n0 != len(dev[t]):
                what += f", {n0 - len(dev[t])} duplicate row(s) dropped"
        else:
            what = "MISSING " + p_dev.name
        what_eq = f"{len(eq[t])} rows" if eq[t] is not None else "none"
        print(f"  {t:<14} {DESIGN.get(t, ''):<26} psi_dev: {what} | psi_eq: {what_eq} | "
              f"{sidecar(bbl_dir, a.key, t)}")
    if dev[unc] is None or dev[cap] is None:
        sys.exit(f"REFUSING: the comparison needs both probe shards ({unc}, {cap}); see MISSING above.")

    cols = psi_columns(dev[cap])
    stats_table(f"psi_dev shard {a.shard} of {a.n_shards}: |psi| over firm x start x shock rows",
                dev, base, cols)
    if any(e is not None for e in eq.values()):
        stats_table("psi_eq (tags that have one): |psi| over firm x start rows", eq, base, cols)

    # ---- explosive rows: |col| beyond mult x the tag's own median ----------------------------
    c = a.col
    print(f"\n== explosive rows: |{c}| > {a.explosive_mult:g} x the tag's own median |{c}| ==")
    print(f"  {'tag':<14} {'threshold':>11} {'rows':>13} {'firms':>6}  firms (worst first)")
    expl = {}
    for t in tags:
        df = dev[t]
        if df is None or c not in df.columns:
            continue
        s = abs_stats(df[c])
        thr = a.explosive_mult * s["med"]
        v = np.abs(df[c].to_numpy(float))
        m = ~np.isfinite(v) | (v > thr)
        expl[t] = m
        fm = firm_max(df.loc[m], c).sort_values(ascending=False) if m.any() else pd.Series(dtype=float)
        names = ", ".join(fm.index[:6].astype(str)) + (" ..." if len(fm) > 6 else "")
        print(f"  {t:<14} {f(thr):>11} {f'{int(m.sum())} of {len(df)}':>13} {len(fm):>6}  {names}")

    # ---- the top firms under the uncapped probe, and the same firms elsewhere ---------------
    top = firm_max(dev[unc], c).sort_values(ascending=False).head(a.top)
    fcap = firm_max(dev[cap], c)
    fbase = firm_max(dev[base], c) if dev[base] is not None and c in dev[base].columns else pd.Series(dtype=float)
    medu = dev[unc].groupby("firm")[c].apply(lambda s: float(np.median(np.abs(s.to_numpy(float)))))
    medc = dev[cap].groupby("firm")[c].apply(lambda s: float(np.median(np.abs(s.to_numpy(float)))))
    isb = dev[unc].groupby("firm")["is_B"].first() if "is_B" in dev[unc].columns else pd.Series(dtype=bool)
    nrow = dev[unc].groupby("firm").size()
    print(f"\n== the {len(top)} firms with the largest |{c}| under {unc}, and the same firms under "
          f"{cap} and {base} ==")
    print(f"  {'firm':<12} {'type':<4} {'rows':>5} {'max ' + unc:>15} {'max ' + cap:>18} {'cap/unc':>9}"
          f" {'med ' + unc:>15} {'med ' + cap:>18} {'max ' + base:>11} {'unc/base':>9}")
    for firm, vu in top.items():
        vc = fcap.get(firm, np.nan)
        vb = fbase.get(firm, np.nan)
        typ = ("B" if bool(isb.get(firm)) else "D") if firm in isb.index else "?"
        print(f"  {str(firm):<12} {typ:<4} {int(nrow.get(firm, 0)):>5} {f(vu):>15} {f(vc):>18} "
              f"{f(ratio(vc, vu), '.3g'):>9} {f(medu.get(firm, np.nan)):>15} {f(medc.get(firm, np.nan)):>18} "
              f"{f(vb):>11} {f(ratio(vu, vb), '.3g'):>9}")

    # ---- row for row: how far the cap moved each component ----------------------------------
    ks = [k for k in ROW_KEY if k in dev[unc].columns and k in dev[cap].columns]
    mu = dev[unc].assign(_expl=expl.get(unc, np.zeros(len(dev[unc]), bool)))
    mm = mu.merge(dev[cap], on=ks, how="inner", suffixes=("_u", "_c"))
    print(f"\n== row for row, {cap} vs {unc}: {len(mm)} rows matched on ({', '.join(ks)}) of "
          f"{len(dev[unc])} / {len(dev[cap])} ==")
    print(f"  relative change |cap - uncapped| / |uncapped|; 'unchanged' = <= {REL_SAME:g}. TYPICAL = "
          f"the rows NOT explosive under {unc} ({int((~mm['_expl']).sum())} rows)")
    print(f"  {'component':<40} {'unchanged':>10} {'median':>10} {'p99':>10} {'max':>10} "
          f"{'| typ: moved>' + format(REL_MOVED, 'g'):>18} {'median':>10} {'p99':>10} {'max':>10}")
    summary_typ = None
    for col in cols:
        if f"{col}_u" not in mm.columns or f"{col}_c" not in mm.columns:
            continue
        u = mm[f"{col}_u"].to_numpy(float)
        v = mm[f"{col}_c"].to_numpy(float)
        ok = np.isfinite(u) & np.isfinite(v)
        rel = np.full(len(mm), np.nan)
        den = np.abs(u)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel[ok] = np.where(den[ok] > 0, np.abs(v[ok] - u[ok]) / den[ok],
                               np.where(v[ok] == u[ok], 0.0, np.inf))
        typ = ok & ~mm["_expl"].to_numpy(bool)

        def q(x, p):
            x = x[np.isfinite(x)]
            return float(np.percentile(x, p)) if x.size else np.nan

        r_all, r_typ = rel[ok], rel[typ]
        same = f"{int(np.sum(r_all <= REL_SAME))}/{r_all.size}"
        moved = f"{int(np.sum(r_typ > REL_MOVED))}/{r_typ.size}"
        print(f"  {col:<40} {same:>10} {f(q(r_all, 50), '.2e'):>10} {f(q(r_all, 99), '.2e'):>10} "
              f"{f(float(np.max(r_all)) if r_all.size else np.nan, '.2e'):>10} {moved:>18} "
              f"{f(q(r_typ, 50), '.2e'):>10} {f(q(r_typ, 99), '.2e'):>10} "
              f"{f(float(np.max(r_typ)) if r_typ.size else np.nan, '.2e'):>10}")
        if col == c:
            summary_typ = (moved, q(r_typ, 50), q(r_typ, 99))
        nbad = int((~ok).sum())
        if nbad:
            print(f"  {'':<40} ({nbad} matched row(s) non-finite in either tag, left out above)")

    # ---- SUMMARY -----------------------------------------------------------------------------
    su, sc = abs_stats(dev[unc][c]), abs_stats(dev[cap][c])
    sb = abs_stats(dev[base][c]) if dev[base] is not None and c in dev[base].columns else None
    eu, ec = expl.get(unc), expl.get(cap)
    print(f"\nSUMMARY ({c}, shard {a.shard} of {a.n_shards}, {a.key})")
    print(f"  explosive rows (> {a.explosive_mult:g} x own median): {unc} {int(eu.sum())} in "
          f"{dev[unc].loc[eu, 'firm'].nunique()} firm(s) -> {cap} {int(ec.sum())} in "
          f"{dev[cap].loc[ec, 'firm'].nunique()} firm(s)")
    print(f"  max |{c}|: {unc} {f(su['max'])} (x{f(ratio(su['max'], sb['max']) if sb else np.nan, '.3g')} "
          f"the {base} max) -> {cap} {f(sc['max'])} (x{f(ratio(sc['max'], sb['max']) if sb else np.nan, '.3g')})")
    print(f"  median |{c}|: {unc} {f(su['med'])} -> {cap} {f(sc['med'])} (x{f(ratio(sc['med'], su['med']), '.6g')});"
          f" {base} {f(sb['med']) if sb else '-'}")
    if summary_typ:
        print(f"  typical rows (not explosive under {unc}): the cap moved {summary_typ[0]} by more than "
              f"{REL_MOVED:g} relative; median change {f(summary_typ[1], '.2e')}, p99 {f(summary_typ[2], '.2e')}")


if __name__ == "__main__":
    main()
