"""
compare_si_links_spec12.py
==========================
Spec-12 comparison of the three single-index sleepiness estimators:
  Est 6  Single-Index   (logit-direction index + cubic sieve link; link uncertainty only)
  Est 7  Joint SI sieve (theta & monotone I-spline link estimated jointly)
  Est 8  Joint SI kernel(theta & kernel local-linear link estimated jointly; rearranged)

Reports a UNIFIED, comparable average marginal effect for every estimator:
  - dummies (Pix): discrete 0->1 effect on phi  (bounded in [-1,1])
  - continuous   : effect of a +1 SD change in the raw regressor on phi (phi-points)
both computed on the actual phi = G(S'theta) via phi_from_native, so magnitudes are
comparable across estimators regardless of theta's normalisation.

Outputs (Drafts/Deposit Competition):
  compare_si_links_spec12.png/.md  -- national phi_t with 95% bands + unified-AME table.

Est 8's robust kernel link degenerates to the [0,1] ceiling here, so the kernel is shown
BOTH ways: robust (with band, the official-loss result) and LS (point path, well-behaved,
matching the sieve level). Est 6 and Est 7 are re-fit with phi_band=True.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["MPLBACKEND"] = "Agg"
import pickle
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

from estimation_2_sleep import build_pooled_data, define_specifications, run_pooled_first_stage
from utils.sleep_links import (fit_nlls_link, fit_single_index, fit_joint_single_index,
                               phi_from_native, NonLinearResults)
from utils import paths as P

DRAFTS = pathlib.Path(r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)"
                      r"\Open Finance\Open-Finance\Drafts\Deposit Competition")
EST8_PKL = P.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "est8" / "estimation_results.pkl"

CLEAN = {"pix_exists": "Pix Available", "gdp_per_capita": "GDP p.c.",
         "cadunico_families_per1000": "CadUnico families", "fraction_65plus": "Fraction 65+",
         "fraction_young": "Fraction young", "risk_free_qoq_lag": "Lagged Selic",
         "connections_per100": "Broadband"}


def unified_ame(res, df, link, idx_cols):
    """Comparable AME on phi: discrete 0->1 for dummies, +1 SD effect for continuous."""
    base = phi_from_native(df, res, link)
    out = {}
    for k in idx_cols:
        col = df[k].values.astype(float)
        uniq = np.unique(col[~np.isnan(col)])
        is_dummy = (len(uniq) == 2) and (0.0 in uniq) and (1.0 in uniq)
        if is_dummy:
            d1 = df.copy(); d1[k] = 1.0
            d0 = df.copy(); d0[k] = 0.0
            out[k] = ("d(0->1)", float(np.mean(phi_from_native(d1, res, link)
                                               - phi_from_native(d0, res, link))))
        else:
            sd = float(np.nanstd(col))
            dd = df.copy(); dd[k] = col + sd
            out[k] = ("+1 SD", float(np.mean(phi_from_native(dd, res, link) - base)))
    return out


def natl_phi_point(df, res, link):
    """Pop-weighted national phi_t point path (no band) from a fitted result."""
    phi = phi_from_native(df, res, link)
    t = df["time_id"].astype(str).values
    m = df["CODMUN_IBGE"].astype(str).values if "CODMUN_IBGE" in df.columns else np.array(["0"] * len(df))
    pop = df["pop_total"].fillna(0).values.astype(float) if "pop_total" in df.columns else np.ones(len(df))
    g = pd.DataFrame({"t": t, "m": m, "phi": phi, "w": pop})
    gm = g.groupby(["t", "m"]).agg(phi=("phi", "mean"), w=("w", "first")).reset_index()
    num = (gm["phi"] * gm["w"]).groupby(gm["t"]).sum()
    den = gm["w"].groupby(gm["t"]).sum().replace(0, np.nan)
    out = (num / den).rename("phi_t").reset_index()
    out["_d"] = pd.PeriodIndex(out["t"].str.replace("Q", "Q"), freq="Q").to_timestamp()
    return out.sort_values("_d").reset_index(drop=True)


def main():
    print("Building spec-12 frame + first stage ...")
    df = build_pooled_data()
    _, iv_specs, sb = define_specifications()
    s_cols = sb["Tech"]
    iv = iv_specs["IV_HausmanFull"]
    df, _ = run_pooled_first_stage(df, iv, [c for c in s_cols if c != "constant"])
    idx_cols = [c for c in s_cols if c != "constant"]

    cachef = P.PROCESSED / "ESTIMATION_OUTPUT" / "Rout" / "cmp_fits_cache.pkl"
    if cachef.exists() and os.environ.get("CMP_REFIT") != "1":
        print(f"Loading Est 6/7 fits from cache ({cachef.name}); set CMP_REFIT=1 to re-fit.")
        e6, e7 = pickle.load(open(cachef, "rb"))
    else:
        logit = fit_nlls_link(df, s_cols, has_cf=True, link="logit", loss="cauchy")
        init = np.array([float(logit.params_native.get(f"interaction_{c}", 0.0)) for c in idx_cols])
        print("Fitting Est 6 (single-index, conditional) ...")
        e6 = fit_single_index(df, s_cols, has_cf=True, logit_res=logit, degree=3, phi_band=True)
        print("Fitting Est 7 (joint sieve) ...")
        e7 = fit_joint_single_index(df, s_cols, has_cf=True, link="sieve", loss="robust",
                                    init_theta=init, n_starts=2, boot_B=999, boot_scheme="webb",
                                    seed=0, phi_band=True, label="cmp/E7")
        cachef.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((e6, e7), open(cachef, "wb"))
        print(f"Cached Est 6/7 fits -> {cachef.name}")

    # Est 8 from the pipeline pickle: robust (with band) + LS (point path).
    e8r = e8ls = None
    if EST8_PKL.exists():
        r = pickle.load(open(EST8_PKL, "rb")).get("IV_HausmanFull x Tech", {})
        e8r = r.get("second_stage")
        e8ls = r.get("second_stage_ls")
    if e8r is None or getattr(e8r, "phi_t_boot", None) is None:
        raise SystemExit("Est 8 robust band not found in pickle — run estimation_8_sleep.py --spec12 first.")

    # ---- unified comparable AMEs ----
    ame6 = unified_ame(e6, df, "index", idx_cols)
    ame7 = unified_ame(e7, df, "sieve", idx_cols)
    ame8 = unified_ame(e8r, df, "kernel", idx_cols)

    cols = [("Est 6 (Single-Index)", e6, ame6), ("Est 7 (Joint sieve)", e7, ame7),
            ("Est 8 (Joint kernel, robust)", e8r, ame8)]
    L = ["# Spec-12 comparison — single-index sleepiness links (Est 6 / 7 / 8)", "",
         "All three estimators model depositor sleepiness as a single index "
         "$\\phi_{mt}=G(\\mathbf S_{mt}'\\theta)$ (the fraction of non-reoptimising depositors), "
         "and differ only in **how the index direction $\\theta$ and the link $G$ are obtained**.",
         "",
         "## The three estimators", "",
         "- **Est 6 — Single-Index (conditional).** Takes the index direction $\\theta$ from the "
         "logit fit (treated as *known*) and estimates only the link $G$ with a monotone cubic "
         "sieve. Fast and stable, but inference is *conditional* on $\\theta$.",
         "- **Est 7 — Joint sieve.** Estimates $\\theta$ ($\\lVert\\theta\\rVert=1$) **and** a "
         "monotone I-spline link $G$ *together* by Ichimura (1993) semiparametric least squares. "
         "The I-spline is monotone and in $[0,1]$ by construction.",
         "- **Est 8 — Joint kernel.** Same joint SLS, but $G$ is a kernel local-linear smoother "
         "(monotonised *ex post* by rearrangement). The most flexible link, and the least "
         "regularised.",
         "",
         "## How they differ in practice", "",
         "- **Index direction.** Est 6 *borrows* $\\theta$ from the logit (identified by the "
         "parametric / cross-sectional structure); Est 7 and Est 8 identify $\\theta$ jointly, only "
         "from within-entity variation, which is weak here — so their jointly-estimated link comes "
         "out **nearly flat** and their marginal effects shrink toward zero. Est 6's larger, "
         "curved link ($\\hat\\phi$ dips to ~0.80 mid-decade) leans on the borrowed direction.",
         "- **Inference.** Est 6's standard errors reflect **only link uncertainty** ($\\theta$ "
         "fixed), so every regressor looks highly significant; Est 7/8's score/multiplier wild "
         "cluster bootstrap also absorbs **direction uncertainty**, and then only **Pix** survives. "
         "Est 6's precision is therefore partly *borrowed*, not a sign of stronger identification.",
         "- **Link regularisation.** Est 7's monotone-bounded I-spline keeps $G$ a well-behaved CDF "
         "even under the robust (Cauchy) loss; Est 8's unconstrained kernel does **not**, so its "
         "robust link **degenerates to the $[0,1]$ ceiling** ($\\hat\\phi_t\\approx1$). The kernel "
         "**LS** link is well-behaved (~0.93, matching the sieve), so Est 8 confirms the *level* "
         "under LS but its robust variant must not be used for $\\phi$.",
         "- **Multistart.** The sieve (Est 6/7) is solved by a fast bounds-constrained least "
         "squares, so Est 7 can afford a couple of starts to settle its optimum; the kernel (Est 8) "
         "profiles $G$ by iterative backfitting, which is too costly to multistart at full $N$, so "
         "Est 8 uses a single logit-warm-started search.",
         "",
         "## Unified, comparable marginal effects", "",
         "**Unified AMEs** on $\\phi$: dummies = discrete 0$\\to$1 effect; "
         "continuous = effect of a $+1$ SD change in the raw regressor (both in $\\phi$-points, "
         "computed on the actual $\\phi=G(\\mathbf S'\\theta)$ — so magnitudes are comparable across "
         "estimators regardless of $\\theta$'s normalisation). $\\hat\\phi_t$ = national pop-weighted "
         "sleepiness.", "",
         "| Variable (effect) | " + " | ".join(c[0] for c in cols) + " |",
         "|---|" + "---|" * len(cols)]
    for k in idx_cols:
        kind = ame6[k][0]
        cells = [f"{a[k][1]:+.4f}" for _, _, a in cols]
        L.append(f"| {CLEAN.get(k, k)} ({kind}) | " + " | ".join(cells) + " |")
    L.append("| **$\\hat\\phi_t$ mean** | " +
             " | ".join(f"{r.phi_t_boot['phi_t'].mean():.3f}" for _, r, _ in cols) + " |")
    L.append("| **$\\hat\\phi_t$ range** | " +
             " | ".join(f"[{r.phi_t_boot['phi_t'].min():.3f}, {r.phi_t_boot['phi_t'].max():.3f}]"
                       for _, r, _ in cols) + " |")
    L.append("| Obs / $G^*$ | " +
             " | ".join(f"{int(r.nobs):,} / {r.G_star:.1f}" for _, r, _ in cols) + " |")
    L.append("")
    L.append("**Caveat — Est 8 robust degeneracy.** The kernel local-linear link under the "
             "robust (Cauchy) loss drifts to the $[0,1]$ ceiling ($\\hat\\phi_t\\approx0.996$): "
             "downweighting the mega-banks leaves small banks with $D\\approx\\widetilde D_{t-1}$, "
             "i.e. $\\phi\\to1$, and the unconstrained local-linear link has nothing to regularise it "
             "(the sieve's monotone-bounded I-spline does). The kernel **LS** link is well-behaved "
             "($\\hat\\phi_t\\approx0.93$, matching the sieve), so the kernel confirms the *level* "
             "under LS; its robust variant should not be used for $\\phi$. Est 7 (sieve) is the "
             "reliable joint estimator.")
    L.append("")
    L.append("![National sleepiness $\\hat\\phi_t$, spec 12, 95% bands]"
             "(compare_si_links_spec12.png){width=95%}")
    L.append("")
    L.append("## Bottom line")
    L.append("")
    L.append("The implied sleepiness **level is broadly consistent (~0.93–0.97)** across the "
             "well-behaved variants (Est 6, Est 7, and Est 8-LS), with heavily overlapping bands "
             "from 2018 on. What changes across estimators is the **marginal-effect structure**: "
             "letting the data choose the index direction jointly (Est 7/8) flattens the link and "
             "shrinks every effect toward zero, leaving **Pix availability as the one robustly "
             "identified driver** of attention; the demographic/macro regressors are not "
             "individually pinned down once $\\theta$ is honestly estimated. **Est 7 (joint sieve) "
             "is the reliable joint estimator**; Est 6 is a useful, more-precise-looking benchmark "
             "whose precision is partly borrowed; Est 8 corroborates the level under LS but its "
             "robust kernel is degenerate.")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(11, 6))
    band = {"Est 6 (Single-Index)": ("#1b9e77", e6), "Est 7 (Joint sieve)": ("#7570b3", e7),
            "Est 8 (Joint kernel, robust)": ("#d95f02", e8r)}
    for name, (c, r) in band.items():
        pb = r.phi_t_boot.copy()
        x = pd.PeriodIndex(pb["time_id"].str.replace("Q", "Q"), freq="Q").to_timestamp()
        ax.plot(x, pb["phi_t"], color=c, lw=2, label=name)
        ax.fill_between(x, pb["lo"], pb["hi"], color=c, alpha=0.18, linewidth=0)
    if e8ls is not None:
        p = natl_phi_point(df, e8ls, "kernel")
        ax.plot(p["_d"], p["phi_t"], color="#d95f02", lw=1.8, ls="--",
                label="Est 8 (Joint kernel, LS)")
    ax.set_ylabel(r"National implied sleepiness  $\hat{\phi}_t$")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Spec 12: implied national sleepiness — three single-index links "
                 "(95% wild-cluster-bootstrap bands)")
    fig.tight_layout()
    fig.savefig(DRAFTS / "compare_si_links_spec12.png", dpi=200); plt.close(fig)
    (DRAFTS / "compare_si_links_spec12.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] -> {DRAFTS / 'compare_si_links_spec12.png'}")
    print("\n".join(L))


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
