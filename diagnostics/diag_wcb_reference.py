"""diag_wcb_reference.py -- what the wild cluster bootstrap REFERENCE distribution costs.

Author: Pedro Feijo de Moraes

The repo's "wild cluster bootstrap p-values" (utils/sleep_links.cluster_wild_bootstrap)
are not bootstrap p-values. The code takes the standard DEVIATION of the bootstrap draws
as a standard error and then reads the p-value off a NORMAL reference:

    sd = np.std(draws[k], ddof=1);  z = |ame| / sd;  p = 2 * Phi(-z)

so the bootstrap supplies the SE but not the reference distribution, and the small-G
refinement that motivates the wild bootstrap is never realised. It is also the
UNRESTRICTED variant (WCU): `theta_b = theta_hat + wv @ IF_cl` perturbs unrestricted
residuals and centres the DGP at theta_hat, so H0 is never imposed. MacKinnon & Webb
(2017, 2018) show the RESTRICTED variant (WCR) dominates at small/imbalanced G.

utils/sleep_links.py now carries an OPT-IN correction behind SLEEP_WCB_MODE; the default
is untouched so every archived p-value stays reproducible. This script QUANTIFIES what
opting in would do, in three sections:

  tier0    oracle: the current module's DEFAULT path must reproduce the pre-edit module
           (git HEAD) bit-for-bit -- same SEs, same p-values, same RNG stream. This is
           the guard that the opt-in did not disturb production.
  size     synthetic size experiment at the repo's own cluster geometry (G=456 drawn to
           CV ~ 8.5 => Carter-Schnepel-Steigerwald G* ~ 6). Rejection rates under a TRUE
           null for the three schemes, so "the normal reference over-rejects" is a
           measured number, not an assertion.
  battery  the real phi-separation regressions (D4b type contrast, D4a attractiveness
           placebo, D2 lagged awake inflow, D6b pooled Pix break) refit ONCE each, with
           all three schemes read off the SAME fit. Legacy p-values are printed next to
           the archived DIAG_PHI_SEPARATION csv values as a replication check.

Usage:
  python diagnostics/diag_wcb_reference.py --only tier0            # ~20 s, no data needed
  python diagnostics/diag_wcb_reference.py --only size --reps 400  # ~2 min, synthetic
  python diagnostics/diag_wcb_reference.py --only battery          # ~6 min, needs the sleep frame
  python diagnostics/diag_wcb_reference.py                         # all three

Outputs: <PROCESSED>/ESTIMATION_OUTPUT/DIAG_PHI_SEPARATION/d_wcb_reference_*.csv

RESULT AS OF 2026-08-03 (re-run to refresh; do not trust this header over the csvs):
  tier0    worst |diff| between the current default path and git HEAD = 0.0.
  size     at G=456/CV=8.54/G*=6.17, a NOMINAL 5% test rejects 27.0% of the time under
           the normal reference, 13.8% under WCU-t, 8.3% under WCR-t (CRVE+t(G*): 14.5%).
           Balanced G=20 control: 14.0% / 6.0% / 6.8%.
  battery  11/11 legacy p-values reproduce the archived csvs. All nulls stay null
           (D2 0.7015 -> 0.736/0.721; D4a 0.5413 -> 0.582/0.582; D6b 0.1679 ->
           0.246/0.291). The single rejection, D4b's CDB carry excess (0.0131), goes to
           0.152 under WCU-t but 0.010 under WCR-t -- the two corrected variants
           DISAGREE on it, and the size section says WCR is the one to believe here.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/ and pipeline modules
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from utils import paths as _paths
from utils.cluster_stats import effective_cluster_stats
from utils.sleep_links import (cluster_wild_bootstrap, linear_wild_cluster_bootstrap,
                               wcb_mode, _boot_pval, _linear_wcb_t)

HERE = Path(__file__).resolve().parent
OUT_DIR = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_PHI_SEPARATION"
OUT_DIR.mkdir(parents=True, exist_ok=True)

B_DEFAULT = 999          # matches apply_imbalanced_cluster_correction's call
SCHEME = "webb"          # repo default; Rademacher would make the studentisation vacuous
SEED = 0                 # matches the production call


def _ts():
    return time.strftime("%H:%M:%S")


def _hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# ==============================================================================
# tier0 -- the DEFAULT path must be bit-identical to the pre-edit module
# ==============================================================================
def _load_head_module(scratch):
    """Import utils/sleep_links.py as it stands at git HEAD, under a private name.

    GOTCHA: the file must be loadable STANDALONE (it only imports os/numpy/pandas/
    statsmodels/scipy, no intra-package relatives), otherwise this oracle silently
    can't run and the guard is worthless -- so a failure here is reported, not
    swallowed."""
    import importlib.util
    src = subprocess.run(["git", "show", "HEAD:utils/sleep_links.py"],
                         cwd=str(HERE), capture_output=True, text=True, encoding="utf-8")
    if src.returncode != 0:
        raise RuntimeError(f"git show failed: {src.stderr.strip()}")
    p = Path(scratch) / "_sleep_links_head.py"
    p.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_sleep_links_head", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, p


def _toy_cluster_ols(n_cl=120, seed=11, k=3):
    """Small clustered OLS fit with imbalanced clusters -- the oracle's test object."""
    rng = np.random.default_rng(seed)
    sizes = np.maximum(1, np.round(rng.lognormal(0.0, 1.6, n_cl) * 6)).astype(int)
    cl = np.repeat(np.arange(n_cl), sizes)
    n = len(cl)
    a = rng.normal(0, 1.0, n_cl)[cl]
    X = np.column_stack([np.ones(n)] + [rng.normal(0, 1, n) + 0.5 * a for _ in range(k - 1)])
    y = X @ np.array([0.3] + [0.1] * (k - 1)) + a + rng.normal(0, 1, n)
    df = pd.DataFrame(X, columns=["const"] + [f"x{j}" for j in range(1, k)])
    res = sm.OLS(y, df).fit(cov_type="cluster",
                            cov_kwds={"groups": pd.Series(cl).astype(str)}, use_t=True)
    return res


def section_tier0(scratch):
    _hr("TIER-0 ORACLE: default path vs the pre-edit module (git HEAD)")
    res = _toy_cluster_ols()
    head, path = _load_head_module(scratch)
    print(f"  HEAD copy -> {path}")
    ok_env = os.environ.get("SLEEP_WCB_MODE")
    if ok_env not in (None, "", "normal"):
        print(f"  [!] SLEEP_WCB_MODE={ok_env!r} is set; the oracle forces 'normal' for "
              f"this section only")
    prev = os.environ.pop("SLEEP_WCB_MODE", None)
    try:
        b_new, t_new, p_new = linear_wild_cluster_bootstrap(res, B=199, scheme=SCHEME, seed=7)
        b_old, t_old, p_old = head.linear_wild_cluster_bootstrap(res, B=199, scheme=SCHEME, seed=7)
    finally:
        if prev is not None:
            os.environ["SLEEP_WCB_MODE"] = prev
    rows = []
    worst = 0.0
    for nm in res.params.index:
        d_b = abs(float(b_new[nm]) - float(b_old[nm]))
        d_p = abs(float(p_new[nm]) - float(p_old[nm]))
        worst = max(worst, d_b, d_p)
        rows.append({"param": nm, "se_head": float(b_old[nm]), "se_now": float(b_new[nm]),
                     "p_head": float(p_old[nm]), "p_now": float(p_new[nm]),
                     "abs_diff_se": d_b, "abs_diff_p": d_p})
        print(f"    {nm:<8s} se {float(b_old[nm]):.10f} -> {float(b_new[nm]):.10f} | "
              f"p {float(p_old[nm]):.10f} -> {float(p_new[nm]):.10f}")

    # the generic (nonlinear-AME) entry point too, since E3/E4 and se_national use it
    rng_a = np.random.default_rng(3)
    IF = rng_a.normal(0, 0.01, (80, 4))
    th = np.array([0.9, -0.2, 0.05, 0.4])
    fn = lambda t: {f"a{j}": float(np.tanh(t[j])) for j in range(4)}
    prev = os.environ.pop("SLEEP_WCB_MODE", None)
    try:
        gb_n, gp_n = cluster_wild_bootstrap(th, IF, fn, fn(th), B=199, scheme=SCHEME,
                                            rng=np.random.default_rng(5))
        gb_o, gp_o = head.cluster_wild_bootstrap(th, IF, fn, fn(th), B=199, scheme=SCHEME,
                                                 rng=np.random.default_rng(5))
    finally:
        if prev is not None:
            os.environ["SLEEP_WCB_MODE"] = prev
    for k in gb_n:
        d_b, d_p = abs(gb_n[k] - gb_o[k]), abs(gp_n[k] - gp_o[k])
        worst = max(worst, d_b, d_p)
        rows.append({"param": f"generic:{k}", "se_head": gb_o[k], "se_now": gb_n[k],
                     "p_head": gp_o[k], "p_now": gp_n[k],
                     "abs_diff_se": d_b, "abs_diff_p": d_p})
    print(f"    generic AME path: max |diff| over {len(gb_n)} AMEs = "
          f"{max(max(abs(gb_n[k]-gb_o[k]), abs(gp_n[k]-gp_o[k])) for k in gb_n):.3e}")

    # VERDICT is computed, never asserted
    print(f"\n  VERDICT: worst absolute difference (SE or p) = {worst:.3e}")
    if worst == 0.0:
        print("  => the default path is BIT-IDENTICAL to the pre-edit module. Every "
              "archived p-value remains reproducible; the correction is opt-in only.")
    else:
        print("  => *** THE DEFAULT MOVED *** -- the opt-in is not clean. Do not commit; "
              "archived p-values would no longer reproduce.")
    pd.DataFrame(rows).to_csv(OUT_DIR / "d_wcb_reference_tier0.csv", index=False)
    return worst == 0.0


# ==============================================================================
# size -- synthetic rejection rates at the repo's cluster geometry
# ==============================================================================
def _calibrated_sizes(G, cv_target, n_total, b_hi=12.0):
    """Deterministic Zipf/Pareto cluster sizes hitting a TARGET realised CV.

    sizes_i proportional to i^-b, i = 1..G, rescaled to n_total, floored at 1; b is
    found by bisection on the REALISED coefficient of variation.

    GOTCHA (this bit the first version of this section): drawing sizes from a lognormal
    with the matching POPULATION CV does not work. The CV of a heavy tail is carried by
    draws a 456-sample almost never contains, so the realised CV came out at 3.6 against
    a target of 8.5 and the whole experiment silently ran at G* = 33 instead of G* = 6 --
    i.e. at a cluster geometry where the normal reference is fine, which would have
    "shown" no size distortion. Calibrate on the realised statistic, always."""
    i = np.arange(1, G + 1, dtype=float)

    def build(b):
        w = i ** (-b)
        s = np.maximum(1.0, np.round(w / w.sum() * n_total))
        return s

    def cv_of(b):
        s = build(b)
        return float(s.std(ddof=0) / s.mean())

    lo, hi = 0.0, b_hi
    if cv_of(hi) < cv_target:
        print(f"  [!] target CV {cv_target} unreachable at G={G}; using b={hi} "
              f"(CV={cv_of(hi):.2f})")
        return build(hi).astype(int)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cv_of(mid) < cv_target:
            lo = mid
        else:
            hi = mid
    return build(0.5 * (lo + hi)).astype(int)


def _one_size_rep(sizes, rng, B, schemes):
    """One replication under a TRUE null for x1. Returns {scheme: p-value}."""
    G = len(sizes)
    cl = np.repeat(np.arange(G), sizes)
    n = len(cl)
    a = rng.normal(0, 1.0, G)[cl]                      # cluster random effect in y
    xa = rng.normal(0, 1.0, G)[cl]                     # cluster component of x1
    x1 = xa + rng.normal(0, 1.0, n)
    x2 = rng.normal(0, 1.0, G)[cl] + rng.normal(0, 1, n)
    y = 0.0 * x1 + 0.2 * x2 + a + rng.normal(0, 1.0, n)     # H0: beta_1 = 0 TRUE
    df = pd.DataFrame({"const": 1.0, "x1": x1, "x2": x2})
    res = sm.OLS(y, df).fit(cov_type="cluster",
                            cov_kwds={"groups": pd.Series(cl).astype(str)}, use_t=True)
    out = {}
    if "normal" in schemes:
        _, _, p = linear_wild_cluster_bootstrap(res, B=B, scheme=SCHEME, seed=0,
                                                mode="normal")
        out["normal"] = float(p["x1"])
    for m in ("wcu", "wcr"):
        if m in schemes:
            _, _, p = linear_wild_cluster_bootstrap(res, B=B, scheme=SCHEME, seed=0, mode=m)
            out[m] = float(p["x1"])
    if "crve_tGstar" in schemes:
        st = effective_cluster_stats(np.bincount(cl))
        nu = max(st["G_star"] - 1.0, 1.0)
        out["crve_tGstar"] = float(2 * stats.t(df=nu).sf(abs(res.tvalues["x1"])))
    return out


def section_size(reps, G, cv_target, n_total, B, seed=20260803):
    _hr(f"SIZE EXPERIMENT (synthetic): G={G}, target CV={cv_target}, N~{n_total:,}, "
        f"B={B}, reps={reps}")
    print("  SYNTHETIC DATA -- not the Brazilian panel. Only the cluster geometry "
          "(G, CV, G*) is\n  matched to the estimation sample; the DGP is a clustered "
          "random-effects linear model\n  with beta_1 = 0 imposed, so every rejection is "
          "a false one.")
    rng = np.random.default_rng(seed)
    sizes = _calibrated_sizes(G, cv_target, n_total)
    st = effective_cluster_stats(sizes)
    top5 = np.sort(sizes)[::-1][:5].sum() / sizes.sum()
    print(f"  realised: G={st['G_nominal']}, CV={st['cv']:.3f}, G*={st['G_star']:.2f}, "
          f"N={sizes.sum():,}, largest cluster={sizes.max():,} "
          f"({sizes.max()/sizes.sum():.1%} of rows), top-5 share={top5:.1%}")
    schemes = ("normal", "wcu", "wcr", "crve_tGstar")
    P = {m: [] for m in schemes}
    t0 = time.time()
    for r in range(reps):
        for m, v in _one_size_rep(sizes, rng, B, schemes).items():
            P[m].append(v)
        if (r + 1) % max(1, reps // 10) == 0:
            print(f"    [{_ts()}] rep {r+1}/{reps}  ({time.time()-t0:.0f}s)")
    rows = []
    print(f"\n  {'scheme':<14s}{'rej@10%':>10s}{'rej@5%':>10s}{'rej@1%':>10s}"
          f"{'  (nominal 10/5/1)'}")
    for m in schemes:
        p = np.asarray(P[m], float)
        r10, r05, r01 = (float(np.mean(p < a)) for a in (0.10, 0.05, 0.01))
        se5 = np.sqrt(0.05 * 0.95 / max(len(p), 1))
        rows.append({"scheme": m, "reps": len(p), "rej10": r10, "rej05": r05,
                     "rej01": r01, "mc_se_at_5pct": se5})
        print(f"  {m:<14s}{r10:>10.3f}{r05:>10.3f}{r01:>10.3f}")
    d = pd.DataFrame(rows).set_index("scheme")
    se5 = float(d["mc_se_at_5pct"].iloc[0])
    print(f"\n  Monte Carlo SE at the 5% level ~ {se5:.4f} "
          f"(2 SE = {2*se5:.4f}); differences smaller than that are noise.")
    print("\n  VERDICT (computed):")
    base = float(d.loc["normal", "rej05"])
    for m in ("wcu", "wcr"):
        v = float(d.loc[m, "rej05"])
        better = abs(v - 0.05) < abs(base - 0.05) - 1e-12
        print(f"    {m:<4s} rejects {v:.3f} vs normal {base:.3f} at nominal 5% -> "
              f"{'CLOSER to nominal' if better else 'NOT closer to nominal'}")
    over = base > 0.05 + 2 * se5
    print(f"    the normal reference is {'OVERSIZED' if over else 'not detectably oversized'} "
          f"at these settings ({base:.3f} vs 0.05, 2 MC SE = {2*se5:.3f}).")
    d.reset_index().to_csv(OUT_DIR / "d_wcb_reference_size.csv", index=False)
    return d


# ==============================================================================
# battery -- the real phi-separation regressions under all three schemes
# ==============================================================================
# Archived legacy p-values (DIAG_PHI_SEPARATION csvs, read 2026-08-03) so the
# replication is checked rather than trusted.
ARCHIVED = {
    ("D4b", "OLSxTech(all k)", "Zdiff_k4"): 0.013076,
    ("D4b", "OLSxTech(all k)", "Zdiff_k5"): 0.060668,
    ("D4b", "OLSxTech(all k)", "Zdiff_k2"): 0.213056,
    ("D4b", "spec12(k=4,5)", "Zdiff_k5"): 0.621844,
    ("D4a", "spec12(k=4,5)", "rank_x_Z"): 0.541269,
    ("D4a", "OLSxTech(all k)", "rank_x_Z"): 0.312231,
    ("D2", "spec12(k=4,5)", "A_full_lag"): 0.701456,
    ("D2", "OLSxTech(all k)", "A_full_lag"): 0.737835,
    ("D6b", "spec12(k=4,5)", "postZ_x"): 0.167945,
    ("D6b", "spec12(k=4,5)", "postZ"): 0.506229,
    ("D6b", "spec12(k=4,5)", "Z_x"): 0.302509,
}


def _fit_aug(df, s_cols, extra_cols, has_cf):
    """The augmented second stage EXACTLY as step_phi_augmented_tests.run_augmented
    builds it (same columns, same all-columns dropna, same two-way demeaning, same
    cluster), but fitting only the augmented model -- the baseline fit is not needed
    here and costs ~20 s. Pinned locally on purpose: sleep_ident_augmented.py is
    under concurrent edit, and this comparison must not move underneath itself."""
    from sleep_ident_augmented import demean_variables_2way
    df = df.copy()
    X_cols = []
    for sv in s_cols:
        col = "nr_lagged_dep" if sv == "constant" else f"interaction_{sv}"
        df[col] = df["nr_lagged_dep"] if sv == "constant" else df[sv] * df["nr_lagged_dep"]
        X_cols.append(col)
    if has_cf:
        X_cols.append("v_hat_x_lagged_dep")
    cols = X_cols + list(extra_cols)
    d = df.dropna(subset=cols + ["deposit_balance"]).copy()
    if len(d) == 0:
        raise RuntimeError("empty sample after dropna")
    cl = d["CodConglomeradoPrudencial"].astype(str)
    y_dm = demean_variables_2way(d, ["deposit_balance"], "entity_id", "time_id")["deposit_balance"]
    X_dm = demean_variables_2way(d, cols, "entity_id", "time_id")
    res = sm.OLS(y_dm, X_dm).fit(cov_type="cluster", cov_kwds={"groups": cl}, use_t=True)
    return res, d, cl


def _three_schemes(res, targets, B):
    """All three references off ONE fit. Returns a list of row dicts."""
    se_n, _, p_n = linear_wild_cluster_bootstrap(res, B=B, scheme=SCHEME, seed=SEED,
                                                 mode="normal")
    tu, tr = {}, {}
    se_u, _, p_u = linear_wild_cluster_bootstrap(res, B=B, scheme=SCHEME, seed=SEED,
                                                 mode="wcu", tstats_out=tu)
    _, _, p_r = linear_wild_cluster_bootstrap(res, B=B, scheme=SCHEME, seed=SEED,
                                              mode="wcr", tstats_out=tr)
    out = []
    for nm in targets:
        t_obs = tu[nm][0]
        c_u = float(np.nanpercentile(np.abs(tu[nm][1]), 95))
        c_r = float(np.nanpercentile(np.abs(tr[nm][1]), 95))
        out.append({"param": nm, "coef": float(res.params[nm]),
                    "se_boot_sd": float(se_n[nm]), "se_crve": float(se_u[nm]),
                    "t_obs": t_obs,
                    "p_normal": float(p_n[nm]), "p_wcu_t": float(p_u[nm]),
                    "p_wcr_t": float(p_r[nm]),
                    "crit95_normal": 1.959964, "crit95_wcu": c_u, "crit95_wcr": c_r})
    return out


def _print_block(label, spec, rows, n):
    print(f"\n  [{label} | {spec}] n={n:,}")
    print(f"    {'param':<14s}{'coef':>11s}{'t_obs':>8s} | {'p normal':>10s}"
          f"{'p WCU-t':>10s}{'p WCR-t':>10s} | {'|t*|.95 WCU':>12s}{'WCR':>8s}"
          f"{'   archived':>12s}")
    for r in rows:
        arch = ARCHIVED.get((label, spec, r["param"]))
        a = f"{arch:.4f}" if arch is not None else "-"
        flag = ""
        if arch is not None:
            flag = " OK" if abs(arch - r["p_normal"]) < 5e-4 else " MISMATCH"
        print(f"    {r['param']:<14s}{r['coef']:>+11.5f}{r['t_obs']:>8.2f} | "
              f"{r['p_normal']:>10.4f}{r['p_wcu_t']:>10.4f}{r['p_wcr_t']:>10.4f} | "
              f"{r['crit95_wcu']:>12.2f}{r['crit95_wcr']:>8.2f}{a:>12s}{flag}")


def section_battery(arms, B):
    _hr("BATTERY: real phi-separation regressions, three references off one fit")
    from sleep_ident_augmented import (load_sleep_frame, build_uncensored_inflow,
                                          lag_within_entity)
    print(f"  [{_ts()}] building the sleep frame ...")
    df0, s_cols = load_sleep_frame()
    print(f"  [{_ts()}] frame ready: {df0.shape}")
    rows = []

    if "d5" in arms:
        # D4b (step_phi_interaction_tests.arm_types): difference-coded type carries.
        for spec, has_cf, ks, base_k in (("OLSxTech(all k)", False, [1, 2, 4, 5], 1),
                                         ("spec12(k=4,5)", True, [4, 5], 4)):
            d0 = df0.copy()
            extra = []
            for k in ks:
                if k == base_k:
                    continue
                nm = f"Zdiff_k{k}"
                d0[nm] = (d0["deposit_type"] == k).astype(float) * d0["nr_lagged_dep"]
                extra.append(nm)
            res, d, _ = _fit_aug(d0, s_cols, extra, has_cf)
            r = _three_schemes(res, extra, B)
            _print_block("D4b", spec, r, len(d))
            rows += [dict(arm="D4b", spec=spec, n=len(d), **x) for x in r]
            print(f"  [{_ts()}] D4b {spec} done")

    if "d3" in arms:
        # D4a (arm_attractiveness): predetermined 2016 within-market share rank x carry.
        df = df0.copy()
        base = df[df["year"] == 2016].copy()
        base["_rank"] = base.groupby(["mca_code", "deposit_type", "time_id"],
                                     observed=True)["deposit_balance"].rank(pct=True)
        rank = base.groupby("entity_id")["_rank"].mean().rename("attr_rank")
        df = df.merge(rank, on="entity_id", how="left")
        df["attr_rank_c"] = df["attr_rank"] - df["attr_rank"].mean()
        df["rank_x_Z"] = df["attr_rank_c"] * df["nr_lagged_dep"]
        for spec, has_cf in (("spec12(k=4,5)", True), ("OLSxTech(all k)", False)):
            res, d, _ = _fit_aug(df, s_cols, ["rank_x_Z"], has_cf)
            r = _three_schemes(res, ["rank_x_Z"], B)
            _print_block("D4a", spec, r, len(d))
            rows += [dict(arm="D4a", spec=spec, n=len(d), **x) for x in r]
            print(f"  [{_ts()}] D4a {spec} done")

    if "d2b" in arms:
        # D2 (arm_lagdepact, uncensored headline): lagged awake inflow A_full_lag.
        df = build_uncensored_inflow(df0.copy())
        df = lag_within_entity(df, "A_full")
        for spec, has_cf in (("spec12(k=4,5)", True), ("OLSxTech(all k)", False)):
            res, d, _ = _fit_aug(df, s_cols, ["A_full_lag"], has_cf)
            r = _three_schemes(res, ["A_full_lag"], B)
            _print_block("D2", spec, r, len(d))
            rows += [dict(arm="D2", spec=spec, n=len(d), **x) for x in r]
            print(f"  [{_ts()}] D2 {spec} done")

    if "d6b" in arms:
        # D6b (arm_pixpooled, 'pooled' variant): post x carry x exposure.
        df = df0.copy()
        s6 = [c for c in s_cols if c != "pix_exists"]
        LAUNCH = 2020 * 4 + 3
        df["_post"] = (df["qidx"] >= LAUNCH).astype(float)
        pre = df[df["year"] < 2020].groupby("entity_id")["connections_per100"].mean()
        e = df["entity_id"].map(pre)
        df["_expo"] = ((e - e.mean()) / e.std(ddof=0)).fillna(0.0)
        df["postZ"] = df["_post"] * df["nr_lagged_dep"]
        df["postZ_x"] = df["postZ"] * df["_expo"]
        df["Z_x"] = df["nr_lagged_dep"] * df["_expo"]
        for spec, has_cf in (("spec12(k=4,5)", True),):
            extra = ["postZ", "postZ_x", "Z_x"]
            res, d, _ = _fit_aug(df, s6, extra, has_cf)
            r = _three_schemes(res, extra, B)
            _print_block("D6b", spec, r, len(d))
            rows += [dict(arm="D6b", spec=spec, n=len(d), **x) for x in r]
            print(f"  [{_ts()}] D6b {spec} done")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "d_wcb_reference_battery.csv", index=False)

    # ---- VERDICT: computed from the table, never asserted ----
    print("\n  VERDICT (computed from the rows above):")
    known = out[out.apply(lambda r: (r["arm"], r["spec"], r["param"]) in ARCHIVED, axis=1)]
    if len(known):
        bad = known[known.apply(
            lambda r: abs(ARCHIVED[(r["arm"], r["spec"], r["param"])] - r["p_normal"]) >= 5e-4,
            axis=1)]
        print(f"    replication: {len(known)-len(bad)}/{len(known)} legacy p-values match "
              f"the archived csvs to 5e-4"
              + ("" if len(bad) == 0 else f"; MISMATCHES: {list(bad['param'])}"))
    for a in (0.05, 0.10):
        n_n = int((out["p_normal"] < a).sum())
        n_u = int((out["p_wcu_t"] < a).sum())
        n_r = int((out["p_wcr_t"] < a).sum())
        print(f"    significant at {a:.0%}: normal {n_n}, WCU-t {n_u}, WCR-t {n_r} "
              f"(of {len(out)} coefficients)")
    flips = out[(out["p_normal"] < 0.05) & ((out["p_wcu_t"] >= 0.05) | (out["p_wcr_t"] >= 0.05))]
    if len(flips):
        for _, r in flips.iterrows():
            print(f"    *** REJECTION LOST: {r['arm']} {r['spec']} {r['param']}: "
                  f"p {r['p_normal']:.4f} -> WCU-t {r['p_wcu_t']:.4f}, "
                  f"WCR-t {r['p_wcr_t']:.4f}")
    else:
        print("    no legacy 5% rejection is overturned by either corrected reference.")
    gains = out[(out["p_normal"] >= 0.05) & ((out["p_wcu_t"] < 0.05) | (out["p_wcr_t"] < 0.05))]
    if len(gains):
        for _, r in gains.iterrows():
            print(f"    *** NEW REJECTION: {r['arm']} {r['spec']} {r['param']}: "
                  f"p {r['p_normal']:.4f} -> WCU-t {r['p_wcu_t']:.4f}, "
                  f"WCR-t {r['p_wcr_t']:.4f}")
    else:
        print("    no null becomes a rejection under either corrected reference.")
    med_u = float(np.median(out["crit95_wcu"]))
    med_r = float(np.median(out["crit95_wcr"]))
    print(f"    median bootstrap 95% critical value: WCU-t {med_u:.2f}, WCR-t {med_r:.2f}, "
          f"against 1.96 for the normal reference "
          f"({'WIDER' if min(med_u, med_r) > 1.96 else 'NOT wider'}).")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=["tier0", "size", "battery"],
                    choices=["tier0", "size", "battery"])
    ap.add_argument("--arms", default="d5,d3,d2b",
                    help="battery arms to run (comma-separated subset of d5,d3,d2b,d6b)")
    ap.add_argument("--reps", type=int, default=400, help="size-experiment replications")
    ap.add_argument("--size-B", type=int, default=299, help="B inside the size experiment")
    ap.add_argument("--B", type=int, default=B_DEFAULT, help="B for the real regressions")
    ap.add_argument("--G", type=int, default=456)
    ap.add_argument("--cv", type=float, default=8.54)
    ap.add_argument("--n-total", type=int, default=40000)
    ap.add_argument("--scratch", default=os.environ.get("TEMP", "."))
    a = ap.parse_args()
    print(f"[{_ts()}] SLEEP_WCB_MODE resolved to {wcb_mode()!r} for THIS process "
          f"(all comparisons below pass mode= explicitly, so the env does not affect them)")
    if "tier0" in a.only:
        section_tier0(a.scratch)
    if "size" in a.only:
        section_size(a.reps, a.G, a.cv, a.n_total, a.size_B)
    if "battery" in a.only:
        section_battery([x.strip().lower() for x in a.arms.split(",") if x.strip()], a.B)
    print(f"\n[{_ts()}] outputs -> {OUT_DIR}")
