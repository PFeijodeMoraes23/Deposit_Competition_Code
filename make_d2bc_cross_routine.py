#!/usr/bin/env python
"""
make_d2bc_cross_routine.py
==========================
Collate the D2b (lagged awake inflow) and D2c (censoring margin) augmented tests across every
sleepiness routine into one table and one figure.

Why this exists. `diag_phi_augmented_tests.py --arm lagdepact --estim N` writes one CSV per
routine, so the cross-routine comparison -- which is the whole point, since the censoring rate
varies by routine (23.7% at E1, higher at the link routines) and the reconstructed awake flow
moves with it -- only existed by reading the files side by side. This builds the comparison as
an artefact.

SCOPE, stated because it is easy to over-read the output. `--estim N` repoints the PHI SOURCE:
`Dep_Act`, the uncensored inflow rebuilt from it, and the censoring indicator all come from
routine N. The SECOND STAGE is E2's linear kernel throughout (`run_augmented`), and the first
stage is shared by construction -- `estimation_sleep_common.py` imports `run_pooled_first_stage`
from `estimation_2_sleep.py`, so E3/E4 and E2 project the spread on the instruments identically.
So each row is "routine N's awake flow tested in a common kernel", not "routine N's own test".
That is the link-independent part of the identifying assumption, and it is the part the
assumption actually concerns; the routine's own link is Appendix A's unapplied patch.

Outputs (OUT_DIR = .../DIAG_PHI_SEPARATION):
    tab_d2_3_cross_routine.tex   booktabs table, also copied next to V_Main.tex
    tab_d2_3_cross_routine.md    the same table as markdown, for the notes
    fig_d2bc_cross_routine.png   coefficient plot with WCB confidence marks

Usage:
    python make_d2bc_cross_routine.py
    python make_d2bc_cross_routine.py --ests 1 2 5 6 7 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_phi_augmented_tests import OUT_DIR  # noqa: E402
from utils import paths as P  # noqa: E402
from utils import routines as R  # noqa: E402

DRAFTS = P.drafts_dir()

LABEL = {e: R.est_diag_label(e) for e in R.ACTIVE}

SPECS = [("spec12(k=4,5)", "spec 12 (k=4,5)"), ("OLSxTech(all k)", "OLS$\\times$Tech (all k)")]


def load(est: int) -> pd.DataFrame | None:
    """Read one routine's lagdepact CSV. E2 is the unsuffixed file (the historical default)."""
    fp = OUT_DIR / (f"d_augmented_lagdepact{'' if est == 2 else f'_E{est}'}.csv")
    if not fp.exists():
        print(f"  [E{est}] MISSING {fp.name} -- run: python diag_phi_augmented_tests.py "
              f"--arm lagdepact --estim {est}")
        return None
    d = pd.read_csv(fp)
    d["estim"] = est
    return d


def pick(d: pd.DataFrame, spec: str, param: str, col: str = "coef"):
    """One scalar from the long CSV, or NaN. The uncensored D2b rows carry a '|uncensored'
    suffix on `spec` while D2c and the AR(1) row do not, so match on the prefix. Every mask is
    built from `m` itself -- masking `m` with a mask built from `d` makes pandas reindex, which
    happens to line up here but silently would not if the CSV ever gained duplicate rows."""
    m = d[(d["spec"].astype(str).str.startswith(spec)) & (d["param"] == param)]
    if param in ("A_full_lag", "phi_avg_market_base", "phi_avg_market_aug"):
        m = m[m["spec"].astype(str).str.contains("uncensored", na=False)]
    return float(m[col].iloc[0]) if len(m) else float("nan")


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ests", nargs="+", type=int,
                    default=[int(x) for x in
                             os.environ.get("SLEEP_ACTIVE_ESTS", R.ACTIVE_ENV_DEFAULT).split()])
    a = ap.parse_args()

    frames = {e: d for e in a.ests if (d := load(e)) is not None}
    if not frames:
        print("no routine CSVs found; nothing to collate")
        return 1
    print(f"  collating {len(frames)} routine(s): {', '.join(f'E{e}' for e in frames)}")

    rows = []
    for est, d in frames.items():
        r = {"estim": est, "label": LABEL.get(est, f"E{est}")}
        for skey, _ in SPECS:
            r[f"{skey}|phi"] = pick(d, skey, "phi_avg_market_base")
            r[f"{skey}|d2b"] = pick(d, skey, "A_full_lag")
            r[f"{skey}|d2b_p"] = pick(d, skey, "A_full_lag", "p_wcb")
            r[f"{skey}|d2c"] = pick(d, skey, "C_lag_x_Z")
            r[f"{skey}|d2c_p"] = pick(d, skey, "C_lag_x_Z", "p_wcb")
            r[f"{skey}|ar1"] = pick(d, skey, "resid_ar1")
        rows.append(r)
    T = pd.DataFrame(rows).sort_values("estim")

    # ---- markdown -----------------------------------------------------------------------
    # The baseline regression is essentially independent of --estim -- it never touches
    # Dep_Act, so only the augmenting column carries the routine. It is not EXACTLY invariant:
    # run_augmented drops rows where the augmenting column is missing, and that set shifts a
    # little with the routine (measured row loss 4.0-4.6%), which moves the baseline in the
    # 6th decimal. So these are reported once per spec rather than per row, and the tolerance
    # below is set to catch a real divergence (anything visible at the 3dp actually printed)
    # while ignoring the sample-composition jitter that is expected.
    TOL = 5e-4
    md = []
    for skey, sname in SPECS:
        phis = T[f"{skey}|phi"].to_numpy(float)
        ar1s = T[f"{skey}|ar1"].to_numpy(float)
        for nm, arr in (("baseline phi", phis), ("resid AR(1)", ar1s)):
            spread = float(np.nanmax(arr) - np.nanmin(arr))
            if spread > TOL:
                print(f"  !! {nm} varies MATERIALLY across routines in {sname} "
                      f"({np.nanmin(arr):.6f}..{np.nanmax(arr):.6f}, spread {spread:.2e}) -- "
                      f"larger than the augmentation's dropna can explain. Investigate "
                      f"before quoting.")
        md.append(f"**{sname}** — common baseline $\\hat\\phi$ = {np.nanmean(phis):.3f}, "
                  f"baseline residual AR(1) = {np.nanmean(ar1s):+.4f} "
                  f"(the baseline fit does not use the $\\hat\\phi$ source, so these are "
                  f"common to every row up to the augmentation's own sample loss)\n")
        md.append("| $\\hat\\phi$ source | D2b $\\hat\\beta_{A_{t-1}}$ | WCB p | "
                  "D2c margin | WCB p |")
        md.append("|---|---|---|---|---|")
        for _, r in T.iterrows():
            md.append(f"| {r['label']} | {r[f'{skey}|d2b']:+.4f}"
                      f"{stars(r[f'{skey}|d2b_p'])} | {r[f'{skey}|d2b_p']:.2f} | "
                      f"{r[f'{skey}|d2c']:+.4f}{stars(r[f'{skey}|d2c_p'])} | "
                      f"{r[f'{skey}|d2c_p']:.2f} |")
        md.append("")
    (OUT_DIR / "tab_d2_3_cross_routine.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- latex --------------------------------------------------------------------------
    tex = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"$\hat\phi$ source & D2b $\hat\beta_{A_{t-1}}$ & WCB $p$ & D2c margin & "
           r"WCB $p$ \\", r"\midrule"]
    for skey, sname in SPECS:
        tex.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{sname}}}: common baseline "
                   rf"$\hat\phi={np.nanmean(T[f'{skey}|phi']):.3f}$, residual "
                   rf"AR(1)$={np.nanmean(T[f'{skey}|ar1']):+.4f}$}} \\")
        for _, r in T.iterrows():
            tex.append(f"\\quad {r['label']} & {r[f'{skey}|d2b']:+.4f} & "
                       f"{r[f'{skey}|d2b_p']:.2f} & {r[f'{skey}|d2c']:+.4f} & "
                       f"{r[f'{skey}|d2c_p']:.2f} \\\\")
        tex.append(r"\addlinespace")
    tex += [r"\bottomrule", r"\end{tabular}"]
    txt = "\n".join(tex) + "\n"
    (OUT_DIR / "tab_d2_3_cross_routine.tex").write_text(txt, encoding="utf-8")
    (DRAFTS / "tab_d2_3_cross_routine.tex").write_text(txt, encoding="utf-8")

    # ---- figure -------------------------------------------------------------------------
    # Coefficients on a common axis with the null at zero. The point of the panel is that
    # nothing separates from zero at ANY phi level, so the carry level is drawn alongside:
    # it runs from ~0.89 to ~1.00 across routines while the test statistics do not move.
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9),
                             gridspec_kw={"width_ratios": [1, 1, 0.75]})
    y = np.arange(len(T))
    colors = {"spec12(k=4,5)": "#1f4e79", "OLSxTech(all k)": "#a63603"}

    for ax, (key, title) in zip(axes[:2], [("d2b", "D2b: lagged awake inflow"),
                                           ("d2c", "D2c: censoring margin")]):
        for off, (skey, sname) in zip((-0.16, 0.16), SPECS):
            c = T[f"{skey}|{key}"].to_numpy(float)
            p = T[f"{skey}|{key}_p"].to_numpy(float)
            ax.scatter(c, y + off, s=52, color=colors[skey], zorder=3,
                       label=sname.replace("$\\times$", "x"),
                       marker="o", edgecolor="white", linewidth=0.8)
            # A WCB p-value implies a |coef| scale under the null; draw it as a bar so the
            # reader sees the precision, not just the point.
            with np.errstate(divide="ignore", invalid="ignore"):
                half = np.abs(c) / np.maximum(
                    np.abs(np.sqrt(2) * np.vectorize(_probit)(1 - p / 2)), 1e-9)
            for yi, ci, hi in zip(y + off, c, 1.96 * half):
                if np.isfinite(hi):
                    ax.plot([ci - hi, ci + hi], [yi, yi], color=colors[skey],
                            lw=1.4, alpha=0.55, zorder=2)
        ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.set_yticks(y); ax.set_yticklabels(T["label"], fontsize=9)
        ax.invert_yaxis(); ax.set_title(title, fontsize=10.5, weight="bold")
        ax.set_xlabel("coefficient (null $=0$)", fontsize=9)
        ax.grid(axis="x", alpha=0.25)
    # Figure-level legend below the panels: inside axes[0] it sat on top of the bottom row.
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=9, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06), frameon=False)

    # Third panel: D0's censoring rate. This is the quantity that actually VARIES with the
    # routine -- 23.7% at E1, higher at the link routines -- and it is why a cross-routine
    # D2b/D2c matters at all: it is how much of the awake flow each routine's phi-hat sends
    # below zero before the test ever sees it. Plotting the baseline phi here instead would
    # plot a constant.
    ax = axes[2]
    cens = []
    for est in T["estim"]:
        fp = OUT_DIR / (f"d_augmented_identity{'' if est == 2 else f'_E{est}'}.csv")
        v = float("nan")
        if fp.exists():
            di = pd.read_csv(fp)
            hit = di[di["param"] == "share_val_neg"]
            if len(hit):
                v = 100.0 * float(hit["coef"].iloc[0])
        cens.append(v)
    ax.barh(y, cens, height=0.55, color="#6b6b6b", alpha=0.85, zorder=3)
    for yi, v in zip(y, cens):
        if np.isfinite(v):
            ax.text(v + 0.6, yi, f"{v:.1f}%", va="center", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_title("D0 censoring rate", fontsize=10.5, weight="bold")
    ax.set_xlabel("% of rows with $A<0$", fontsize=9)
    ax.set_xlim(0, max([v for v in cens if np.isfinite(v)] + [1]) * 1.28)
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle("D2b / D2c across sleepiness routines — common (E2) second stage, "
                 "routine-specific $\\hat\\phi$ source", fontsize=11, y=1.03)
    fig.tight_layout()
    fp = OUT_DIR / "fig_d2bc_cross_routine.png"
    fig.savefig(fp, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"  -> {OUT_DIR / 'tab_d2_3_cross_routine.md'}")
    print(f"  -> {OUT_DIR / 'tab_d2_3_cross_routine.tex'}  (+ copy next to V_Main.tex)")
    print(f"  -> {fp}")
    print("\n" + "\n".join(md))
    return 0


def _probit(q):
    """Inverse standard normal CDF without scipy, via the Beasley-Springer-Moro tail."""
    from math import erf, sqrt
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1 + erf(mid / sqrt(2))) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    sys.exit(main())
