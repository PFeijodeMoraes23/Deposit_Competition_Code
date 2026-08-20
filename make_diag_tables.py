#!/usr/bin/env python
"""
make_diag_tables.py — export every per-routine diagnostic exhibit as a V_Main-callable
artifact.

The identification battery's tables live inline in identification_notes.md; the paper needs
them as standalone \\input-able booktabs fragments next to V_Main.tex. This script is the
single exporter: it reads ONLY the authoritative CSVs in DIAG_PHI_SEPARATION (never
recomputes), writes tab_*.tex fragments to the Drafts folder, and copies the figures the
notes reference so V_Main can \\includegraphics them from its own directory. Idempotent;
run after any diagnostic re-run. Already-exported tables with their own makers
(tab_d2_3_cross_routine, tab_d6c_pix_gradient, tab_selic_wakeup, tab_bbl_*) are left to
those makers.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_phi_augmented_tests import OUT_DIR  # noqa: E402
from utils import paths as _paths             # noqa: E402
from utils import routines as _routines       # noqa: E402

DRAFTS = _paths.drafts_dir()
NOTES = DRAFTS / "identification_notes.md"
ROMAN = {e: _routines.est_roman(e) for e in _routines.ACTIVE}
LAB = {e: _routines.est_diag_label(e) for e in _routines.ACTIVE}
ESTS = tuple(_routines.ACTIVE)          # every reported routine
NONLINEAR = tuple(_routines.LINK_ESTS)  # the single-index pair


def w(name: str, lines: list[str]) -> None:
    txt = "\n".join([r"\begin{tabular}" + lines[0], r"\toprule"] + lines[1:]
                    + [r"\bottomrule", r"\end{tabular}"]) + "\n"
    (DRAFTS / name).write_text(txt, encoding="utf-8")
    print(f"  -> {name}")


def tab_d0():
    rows = []
    for e in ESTS:
        f = OUT_DIR / ("d_augmented_identity.csv" if e == 2
                       else f"d_augmented_identity_E{e}.csv")
        d = pd.read_csv(f)
        cens = float(d.loc[d["param"] == "share_val_neg", "coef"].iloc[0])
        match = float(d.loc[d["param"] == "matched_share", "coef"].iloc[0])
        rows.append(f"{LAB[e]} & {100*cens:.1f}\\% & {100*match:.1f}\\% \\\\")
    w("tab_d0_censoring.tex",
      ["{lcc}", r"Routine & Cells censored ($A<0$) & Reaching the demand parquet \\",
       r"\midrule"] + rows)


def tab_d4b():
    d = pd.read_csv(OUT_DIR / "d_interactions_types.csv")
    o = d[d["spec"] == "OLSxTech(all k)"]
    lvl1 = float(o.loc[o["param"] == "phi_k1", "coef"].iloc[0])
    rows = [f"1 (savings, regulated) & {lvl1:.3f} & --- & --- \\\\"]
    for k in (2, 4, 5):
        r = o[o["param"] == f"Zdiff_k{k}"]
        rows.append(f"{k} & {float(r['phi_level'].iloc[0]):.3f} & "
                    f"{float(r['coef'].iloc[0]):+.3f} & {float(r['p_wcb'].iloc[0]):.3f} \\\\")
    w("tab_d4b_types.tex",
      ["{lccc}", r"Deposit type $k$ & $\hat\phi_k$ & $\hat\phi_k-\hat\phi_1$ & WCB $p$ \\",
       r"\midrule"] + rows)


def tab_d4c():
    d = pd.read_csv(OUT_DIR / "d4c_acf_routine_bands.csv")
    rows = []
    for e in ESTS:
        g = d[d["estim"] == e]
        r0 = g.iloc[0]
        cov = {k: 100 * float(gg["inside"].mean())
               for k, gg in g.groupby("kind")}
        rows.append(f"{LAB[e]} & {r0['phi_mean']:.3f} [{r0['phi_p10']:.3f}, "
                    f"{r0['phi_p90']:.3f}] & {cov['level']:.0f}\\% & "
                    f"{cov['fitted phi_j']:.0f}\\% \\\\")
    w("tab_d4c_routine_bands.tex",
      ["{lccc}", r"Routine & mean $\bar\phi$ (p10--p90 of $\phi_j$) & "
       r"Level band coverage & Fitted $\phi_j$ band \\", r"\midrule"] + rows)


def tab_d5():
    obs = None
    rows = []
    for e in NONLINEAR:
        d = pd.read_csv(OUT_DIR / f"d_augmented_blpelast_E{e}.csv")
        if obs is None:
            r = d[d["param"] == "dDep_drho"].iloc[0]
            obs = (float(r["coef"]), float(r["p_wcb"]))
        imp = d[d["param"] == "implied"]
        rows.append(f"{LAB[e]} & {float(imp['coef'].iloc[0]):+.2f} & sign mismatch --- "
                    f"$k$ not interpretable \\\\")
    hdr = (r"Routine (own $\hat\alpha\times$ own parquet) & implied "
           r"$\partial\mathrm{Dep}/\partial\rho$ & vs observed "
           f"{obs[0]:+.2f} (p={obs[1]:.2f}) \\\\")
    w("tab_d5_diagonal.tex", ["{lcc}", hdr, r"\midrule"] + rows)


def tab_d6_entry_curves():
    d = pd.read_csv(OUT_DIR / "d6_routine_curves.csv")
    s = d[d["h"].isna()] if d["h"].isna().any() else d.drop_duplicates("estim")
    rows = []
    for e in ESTS:
        m = s[s["estim"] == e]
        if not len(m):
            continue
        r = m.iloc[0]
        # share_explosive counts EVENTS whose average accrual tips them over; with a
        # per-quarter g an event can also be explosive only part of the time, which
        # share_explosive_qtrs reports. The two coincide under a constant g.
        q = r.get("share_explosive_qtrs", np.nan)
        qcell = f"{100*q:.0f}\\%" if pd.notna(q) else "--"
        rows.append(f"{LAB[e]} & {r['phi_mean']:.3f} [{r['phi_p10']:.3f}, "
                    f"{r['phi_p90']:.3f}] & {100*r['share_explosive']:.0f}\\% & "
                    f"{qcell} & {r['sse_vs_data']:.4f} \\\\")
    w("tab_d6_routine_curves.tex",
      ["{lcccc}", r"Routine & mean $\phi_m$ (p10--p90) & Entry markets explosive "
       r"($\phi_m \bar g_e\ge1$) & Event-quarters explosive & SSE vs data median \\",
       r"\midrule"] + rows)


def tab_d6_explosive():
    """Stationarity diagnostics for the nonlinear routines, at full phi_{m,t} x g_{e,h}
    resolution. E1/E2 are excluded: the paper's stationarity discussion is about the
    routines whose fitted carry approaches one.

    Reports depth and persistence alongside frequency, because they read differently -- a
    breach of 0.01 that appears in some quarters is a different object from a market that
    sits above the bound throughout.
    """
    f = OUT_DIR / "d6_explosive_diag_phipath.csv"
    if not f.exists():
        print(f"  [skip] {f.name} absent -- run diag_entry_dynamics.py --phi-mode path")
        return
    d = pd.read_csv(f).set_index("estim")
    rows = []
    for e in NONLINEAR:
        if e not in d.index:
            continue
        r = d.loc[e]
        pop = (f"{100*r['pop_share_expl']:.1f}\\%" if pd.notna(r.get("pop_share_expl"))
               else "--")
        rows.append(
            f"{ROMAN[e]} {LAB[e].split(' ', 1)[1]} & {r['phi_mean']:.3f} & "
            f"{r['phi_g_median']:.3f} & {100*r['share_cells_expl']:.0f}\\% & "
            f"{100*r['share_events_ever']:.0f}\\% & {100*r['share_events_always']:.0f}\\% & "
            f"{r['mean_excess']:.4f} & {pop} \\\\")
    w("tab_d6_explosive.tex",
      ["{lccccccc}",
       r"Strategy & mean $\phi_{m,t}$ & median $\phi g$ & \multicolumn{3}{c}{Entry markets} "
       r"& mean excess & All market--\\",
       r"\cmidrule(lr){4-6}",
       r" &  &  & quarters & entrants & entrants & $(\phi g - 1)$ & quarters \\",
       r" &  &  & $\phi g\ge1$ & ever & always &  & $\phi g\ge1$ \\",
       r"\midrule"] + rows)


def tab_wcb():
    d = pd.read_csv(OUT_DIR / "d_wcb_reference_battery.csv")
    esc = "\\_"
    rows = []
    for _, r in d.iterrows():
        pname = str(r["param"]).replace("_", esc)
        rows.append(f"{r['arm']} & {pname} & {r['coef']:+.4f} & "
                    f"{r['p_normal']:.3f} & {r['p_wcu_t']:.3f} & {r['p_wcr_t']:.3f} \\\\")
    w("tab_wcb_reference.tex",
      ["{llcccc}", r"Arm & Coefficient & $\hat\beta$ & $p$ normal & $p$ WCU-$t$ & "
       r"$p$ WCR-$t$ \\", r"\midrule"] + rows)


def copy_figures():
    refs = re.findall(r"identification_figs/([A-Za-z0-9_.]+\.png)",
                      NOTES.read_text(encoding="utf-8"))
    n = 0
    for f in sorted(set(refs)):
        src = OUT_DIR / f
        if src.exists():
            shutil.copy2(src, DRAFTS / f)
            n += 1
        else:
            print(f"  !! figure missing at source: {f}")
    print(f"  -> {n} notes-referenced figures copied beside V_Main.tex")


if __name__ == "__main__":
    for fn in (tab_d0, tab_d4b, tab_d4c, tab_d5, tab_d6_entry_curves,
               tab_d6_explosive, tab_wcb):
        fn()
    copy_figures()
    print("done")
