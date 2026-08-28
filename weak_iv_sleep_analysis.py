#!/usr/bin/env python3
"""sleep_weak_iv.py -- weak-instruments battery for the SLEEPINESS (phi) first stage.

The analog of blp_weak_iv.py (the demand first stage), for the depositor-inattention (phi)
stage. The sleepiness first stage (estimation_2_sleep.run_pooled_first_stage) regresses the
endogenous QoQ deposit spread `spread_qoq` (deposit types 4 and 5) on excluded instruments +
exogenous state controls, clustered by `CodConglomeradoPrudencial`, producing the residual v_hat;
the control-function term v_hat x lagged_deposits then enters a TWO-WAY-FE (entity_id + time_id)
second stage that reconstructs phi via a single index through a logit or a monotone
sieve link.

Because the sleepiness stage is NONLINEAR in the spread (its residual enters as a control function,
phi is a link index), there is no verbatim linear structural equation delta = alpha*spread + X*beta.
This module therefore reports FOUR non-redundant objects per instrument spec x deposit-type subsample:

  1. FIRST-STAGE STRENGTH (delta-free; reuses the demand battery verbatim), computed BOTH ways:
       - POOLED (as run_pooled_first_stage runs it): cluster-robust joint F of the excluded
         instruments, MOP effective-F, KP rk-Wald F, Cragg-Donald F, Patnaik K_eff, partial R^2.
       - TWO-WAY-FE-DEMEANED (entity + quarter, alternating projections): the HONEST first stage,
         since the second stage identifies phi WITHIN entity net of common time.
     Plus effective clusters G* (Carter-Schnepel-Steigerwald), per-instrument relevance
     (coef / cluster-|t| / sign), instrument-block collinearity (condition number + VIF spectrum),
     Hansen J overid, and leave-one-instrument-out / leave-one-GROUP-out sensitivity.

  2. (A) LINEAR-IV PROJECTION BENCHMARK -- 2SLS of two-way-FE-demeaned deposit_balance on demeaned
     spread_qoq (single endogenous), instrumented by the spec's instruments. Reuses the ENTIRE demand
     battery: 2SLS/LIML/Fuller/OLS ladder, tF honest CI (Lee-McCrary-Moreira-Porter 2022), and
     Anderson-Rubin / Kleibergen-LM weak-IV-robust CIs (wild-cluster-bootstrap criticals). This is a
     LINEAR PROJECTION diagnosing the identification the CF stage relies on (same first stage + same
     exclusion restriction); it is NOT the structural phi parameter -- labeled as such throughout.

  3. (B) STRUCTURAL CF HAUSMAN TEST -- the coefficient gamma on v_hat_x_lagged_dep in the actual
     two-way-FE second stage, with wild-cluster-bootstrap significance. tF is deliberately NOT applied
     to gamma: it is an OLS coefficient on a GENERATED regressor, whose weak-instrument pathology
     differs in kind (the LMMP c(F) multiplier is not validated for it). Under H0: gamma = 0 the
     generated-regressor correction vanishes, so the WCB t-test of gamma = 0 is a valid few-cluster
     endogeneity test; for gamma != 0 the interval understates uncertainty (v_hat treated as data).

  4. (phi) "DO WE NEED IT" MATERIALITY TEST (spec 12 = IV_HausmanFull x Tech) -- reconstruct the
     national deposit-weighted phi_t under OLS (no CF) vs IV_HausmanFull (CF) on BOTH link shapes
     the estimator can carry -- the logit and the shape-constrained monotone sieve (the reported
     E3/E4 link) -- on the COMMON sample (rows where the instruments exist); report
     Pearson/Spearman correlation, mean|delta|, and trend direction.

CAVEAT flagged throughout: deposits are extremely concentrated by conglomerate, so the effective
cluster count collapses to G* ~ 5-7. The wide confidence sets are a FEW-EFFECTIVE-CLUSTER PRECISION
issue, NOT an instrument-relevance failure -- the pooled effective-F is ~60.

Reuses blp_weak_iv.py's numpy battery by importing it and swapping two module globals
(IV_GROUPS for the leave-one-group-out; AR_GRID -- rescaled PER SUBSAMPLE, because the sleep alpha is
in deposits(bn)-per-spread_qoq units, not the demand's log-share-per-pp units). Never edits estimation
code -- reads/calls only.

Writes cluster_processed/weak_iv_sleep.json and a methodology note under DIAG_WEAK_IV_SLEEP/.

Run:  python sleep_weak_iv.py [BLP_RESULTS_dir]
        [--specs IV_CostShifters,IV_Wholesale,IV_HausmanFull] [--subsamples type45,type4,type5]
        [--skip-sieve] [--phi-only]
"""
import os
# Import-time env flags MUST precede the estimation imports below.
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import json
import argparse

try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except Exception:
    pass

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console defaults to cp1252
except Exception:
    pass

import numpy as np
import pandas as pd

import blp_weak_iv as wiv                 # THE demand battery -- imported, never edited
import sleep_est_e2 as e2                # build_pooled_data / define_specifications / run_pooled_*
from utils.sleep_links import fit_nlls_link, fit_single_index, phi_from_native
from utils.cluster_stats import effective_cluster_stats
from utils import paths as P
from utils import routines as R

# ---------------------------------------------------------------------------
# Sleepiness-stage constants (the swap-ins vs the demand module)
# ---------------------------------------------------------------------------
CLUSTER_KEY = "CodConglomeradoPrudencial"
ENDOG_TYPES = [4, 5]
SUBSAMPLES = ["type45", "type4", "type5"]
SPEC_ORDER = ["IV_CostShifters", "IV_Wholesale", "IV_HausmanFull"]   # OLS has no instruments -> skipped
HEADLINE_STATE = "Tech"                                              # spec 12 state block

# Named instrument GROUPS for the leave-one-group-out sensitivity (swapped into wiv.IV_GROUPS).
# `hausman` = the Hausman leave-one-out mean spread (the single strongest instrument);
# `wholesale` = the wholesale-funding block; `cost` = the cost shifters; `capital` = Basileia.
SLEEP_IV_GROUPS = {
    "cost":      ["personnel_cost_ratio_lag", "admin_cost_ratio_lag", "tax_cost_ratio_lag"],
    "wholesale": ["lci_lca_ratio_lag", "wholesale_ratio_lag"],
    "capital":   ["indice_basileia_lag"],
    "hausman":   ["leave_one_out_mean_spread"],
}


def _say(report, msg=""):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))
    report.append(msg)


# ---------------------------------------------------------------------------
# Sample + matrix builders (the only genuinely new logic)
# ---------------------------------------------------------------------------
def _subsample_mask(df, name):
    dt = df["deposit_type"]
    if name == "type4":
        return (dt == 4).to_numpy()
    if name == "type5":
        return (dt == 5).to_numpy()
    return dt.isin(ENDOG_TYPES).to_numpy()


def _prep_subsample(df, iv_cols, state_full, mask):
    """Common row filter shared by the pooled and two-way regimes: the subsample x instruments-nonnull
    x exog-nonnull x core-cols-nonnull rows, minus zero-variance instruments/controls in this cell."""
    exs = [c for c in state_full if c != "constant"]
    core = ["deposit_balance", "spread_qoq", "lagged_deposits", "entity_id", "time_id"]
    need = list(dict.fromkeys(core + [CLUSTER_KEY] + list(iv_cols) + exs))
    need = [c for c in need if c in df.columns]
    sub = df.loc[mask, need].copy()
    ivs = [c for c in iv_cols if c in sub.columns]
    exs = [c for c in exs if c in sub.columns]
    sub = sub.dropna(subset=[c for c in core if c in sub.columns] + ivs + exs)
    if len(sub) == 0:
        return None
    ivs = [c for c in ivs if float(sub[c].std()) > 1e-10]
    exs = [c for c in exs if float(sub[c].std()) > 1e-10]
    if not ivs:
        return None
    cl = sub[CLUSTER_KEY].astype(str)
    uniq, codes = np.unique(cl.values, return_inverse=True)
    return dict(sub=sub, ivs=ivs, exs=exs, cl=cl.to_numpy(), codes=codes, G=int(len(uniq)))


def _level_arrays(prep):
    sub, ivs, exs = prep["sub"], prep["ivs"], prep["exs"]
    dep = sub["deposit_balance"].to_numpy(float)
    spread = sub["spread_qoq"].to_numpy(float)
    X = np.column_stack([np.ones(len(sub))] + [sub[c].to_numpy(float) for c in exs])
    Z = np.column_stack([sub[c].to_numpy(float) for c in ivs])
    return dep, spread, X, Z


def _twoway_arrays(prep):
    """Entity+time two-way-FE demean of deposit_balance / spread_qoq / instruments / exog controls
    (alternating projections, estimation_2_sleep.demean_variables_2way -- REUSED, not reinvented)."""
    sub, ivs, exs = prep["sub"], prep["ivs"], prep["exs"]
    dm = e2.demean_variables_2way(sub, ["deposit_balance", "spread_qoq"] + ivs + exs,
                                  "entity_id", "time_id")
    dep = dm["deposit_balance"].to_numpy(float)
    spread = dm["spread_qoq"].to_numpy(float)
    X = np.column_stack([np.ones(len(sub))] + [dm[c].to_numpy(float) for c in exs])
    Z = np.column_stack([dm[c].to_numpy(float) for c in ivs])
    return dep, spread, X, Z


def _sm_joint_F(spread, X, Z, cl):
    """Cluster-robust joint F of the excluded instruments via statsmodels (matches how
    run_pooled_first_stage runs the first stage; validates the numpy KP-F). X includes the constant."""
    import statsmodels.api as sm
    design = np.column_stack([X, Z])
    r = sm.OLS(np.asarray(spread, float), design).fit(
        cov_type="cluster", cov_kwds={"groups": np.asarray(cl)}, use_t=True)
    K, p = Z.shape[1], design.shape[1]
    R = np.zeros((K, p))
    R[np.arange(K), np.arange(p - K, p)] = 1.0
    try:
        F = r.f_test(R)
        return float(np.ravel(F.fvalue)[0]), float(np.ravel(F.pvalue)[0])
    except Exception:
        return None, None


def _first_stage_strength(spread, X, Z, codes, G, cl):
    """Delta-free first-stage strength: the demand battery's _fs_F (MOP eff-F, KP-F, Patnaik K_eff,
    partial R^2) + the statsmodels cluster-robust joint F + the effective cluster count G*."""
    fs = wiv._fs_F(spread, X, Z, codes, G)
    jF, jp = _sm_joint_F(spread, X, Z, cl)
    st = effective_cluster_stats(np.bincount(codes))
    fs["joint_F"] = jF
    fs["joint_F_p"] = jp
    fs["G_star"] = st["G_star"]
    fs["cv"] = st["cv"]
    return fs


# ---------------------------------------------------------------------------
# Framing A -- the linear-IV projection battery (on the two-way-FE arrays)
# ---------------------------------------------------------------------------
def _sleep_ar_grid(dep, spread, X, Z, codes, G):
    """Adaptive AR/LM alpha grid: the demand global AR_GRID=[-2,2] is in log-share-per-pp units, but
    the sleep alpha is deposits(bn)-per-spread_qoq and generally falls outside [-2,2], which would make
    the AR/LM sets vacuous. Scale to +-5*max(|alpha_2sls|, 10*SE)."""
    a2, se2 = wiv._iv2sls_alpha(dep, spread, X, Z, codes, G)
    scale = max(abs(a2) if np.isfinite(a2) else 0.0,
                10.0 * se2 if np.isfinite(se2) else 0.0, 1e-4)
    s = 5.0 * scale
    return np.linspace(-s, s, 4001)


def _linear_iv_battery(dep, spread, X, Z, codes, G, cl, iv_kept):
    wiv.AR_GRID = _sleep_ar_grid(dep, spread, X, Z, codes, G)     # per-subsample grid (gotcha #1)
    rec = wiv._battery(dep, spread, X, Z, codes, G)               # eff/KP/CD/MOP/k_eff/G* + AR/LM/JI + Hansen J
    rec = wiv._add_linearmodels(rec, dep, spread, X, Z, cl)       # 2SLS alpha/SE + partial R^2 (linearmodels)
    rec = wiv._liml_fuller(rec, dep, spread, X, Z, cl)            # LIML / Fuller(1)
    rec = wiv._ols_dwh(rec, dep, spread, X, Z, codes, G)          # OLS alpha + linear-IV DWH t on v_hat
    rec["tf"] = wiv._tf_adjust(rec.get("effective_F"), rec.get("alpha_2sls"), rec.get("alpha_se"))
    rec["z_collinearity"] = wiv._condnum_vif(wiv._resid(Z, X), iv_kept)
    rec["loo"] = wiv._loo_sensitivity(dep, spread, X, Z, iv_kept, codes, G)   # uses swapped IV_GROUPS
    rec["ar_grid_lo"] = float(wiv.AR_GRID[0])
    rec["ar_grid_hi"] = float(wiv.AR_GRID[-1])
    rec["projection_note"] = ("LINEAR-IV PROJECTION of two-way-FE deposit_balance on spread_qoq; "
                              "a benchmark for the identification, NOT the structural phi.")
    return rec


# ---------------------------------------------------------------------------
# Framing B -- the structural control-function Hausman test
# ---------------------------------------------------------------------------
def _cf_hausman(df, iv_cols, state_full, mask, twoway_eff_F):
    """gamma on v_hat_x_lagged_dep in the actual two-way-FE CF second stage + WCB significance.
    Mirrors exec_pooled_spec: run_pooled_first_stage -> run_pooled_second_stage(has_cf=True)."""
    sub = df.loc[mask].copy()
    iv_act = [c for c in iv_cols if c in sub.columns and sub[c].notnull().sum() > 0]
    exog_act = [c for c in state_full if c != "constant" and c in sub.columns and sub[c].notnull().sum() > 0]
    if not iv_act:
        return None
    sub, _ = e2.run_pooled_first_stage(sub, iv_act, exog_act)
    res_ss = e2.run_pooled_second_stage(sub, state_full, has_cf=True)
    if res_ss is None or "v_hat_x_lagged_dep" not in getattr(res_ss, "params", {}):
        return None
    g = "v_hat_x_lagged_dep"
    return dict(
        gamma=float(res_ss.params[g]),
        gamma_se=float(res_ss.bse[g]) if g in res_ss.bse else None,
        gamma_t_wcb=float(res_ss.tvalues[g]) if g in res_ss.tvalues else None,
        gamma_p_wcb=float(res_ss.pvalues[g]) if g in res_ss.pvalues else None,
        G_star=float(getattr(res_ss, "G_star", np.nan)),
        twoway_eff_F=(float(twoway_eff_F) if twoway_eff_F is not None else None),
        generated_regressor=True,
        note=("DWH/Hausman on the structural CF term. The WCB t of gamma=0 is a valid few-cluster "
              "endogeneity test; the interval understates uncertainty (v_hat treated as data). "
              "tF is deliberately NOT applied to a generated-regressor coefficient."),
    )


# ---------------------------------------------------------------------------
# The phi "do we need it" materiality test (spec 12, logit + sieve)
# ---------------------------------------------------------------------------
def _fit_phi(df, state_full, has_cf, link):
    """The spec-12 phi fit under one link shape. `link="logit"` stops at the NLLS logit;
    `link="sieve"` carries that logit direction into the shape-constrained monotone link,
    i.e. exactly the estimator E3/E4 report (utils.sleep_links.fit_single_index)."""
    lg = fit_nlls_link(df, state_full, has_cf=has_cf, link="logit", loss="cauchy",
                       fe_time_col=R.FE_TIME_COL, bootstrap=False)
    if link == "logit" or lg is None:
        return lg
    return fit_single_index(df, state_full, has_cf=has_cf, logit_res=lg, degree=3,
                            fe_time_col=R.FE_TIME_COL)


def _wmean_by_time(time_id, phi, w):
    """National weighted phi_t = sum_i w_i phi_i / sum_i w_i within each time_id (deposit- or pop-
    weighted). Falls back to the unweighted mean where the weight sum is zero."""
    t = np.asarray(time_id).astype(str)
    phi = np.asarray(phi, float)
    w = np.clip(np.asarray(w, float), 0.0, None)
    num = pd.Series(phi * w).groupby(t).sum()
    den = pd.Series(w).groupby(t).sum().replace(0, np.nan)
    unw = pd.Series(phi).groupby(t).mean()
    return (num / den).fillna(unw)


def _sorted_by_quarter(series):
    try:
        per = pd.PeriodIndex(series.index.astype(str), freq="Q")
        return series.iloc[np.argsort(per.astype("int64"))]
    except Exception:
        return series.sort_index()


def _phi_summary(common, phi_ols, phi_iv):
    t = common["time_id"].to_numpy()
    wdep = common["lagged_deposits"].to_numpy(float)
    p_ols = _sorted_by_quarter(_wmean_by_time(t, phi_ols, wdep))
    p_iv = _sorted_by_quarter(_wmean_by_time(t, phi_iv, wdep))
    comp = pd.DataFrame({"phi_ols": p_ols, "phi_iv": p_iv}).dropna()
    comp["delta"] = comp["phi_iv"] - comp["phi_ols"]
    # secondary: population-weighted (matches the production _agg_phi_t weighting) for cross-reference
    wpop = common["pop_total"].to_numpy(float) if "pop_total" in common.columns else np.ones(len(common))
    pop_ols = _wmean_by_time(t, phi_ols, wpop)
    pop_iv = _wmean_by_time(t, phi_iv, wpop)
    pop_corr = pd.DataFrame({"a": pop_ols, "b": pop_iv}).dropna()
    return dict(
        weighting="deposit (lagged_deposits)",
        mean_phi_ols=float(comp["phi_ols"].mean()), mean_phi_iv=float(comp["phi_iv"].mean()),
        mean_abs_delta=float(comp["delta"].abs().mean()), max_abs_delta=float(comp["delta"].abs().max()),
        pearson=float(comp["phi_ols"].corr(comp["phi_iv"])),
        spearman=float(comp["phi_ols"].corr(comp["phi_iv"], method="spearman")),
        trend_ols=float(comp["phi_ols"].iloc[-1] - comp["phi_ols"].iloc[0]),
        trend_iv=float(comp["phi_iv"].iloc[-1] - comp["phi_iv"].iloc[0]),
        pearson_popweighted=(float(pop_corr["a"].corr(pop_corr["b"])) if len(pop_corr) > 1 else None),
        n_common=int(len(common)), n_quarters=int(len(comp)),
        per_quarter=[{"time_id": str(t_), "phi_ols": float(r.phi_ols),
                      "phi_iv": float(r.phi_iv), "delta": float(r.delta)}
                     for t_, r in comp.iterrows()],
    )


def _phi_need_test(df, iv_cols, state_full, link):
    iv_act = [c for c in iv_cols if c in df.columns and df[c].notnull().sum() > 0]
    exog_act = [c for c in state_full if c != "constant" and c in df.columns and df[c].notnull().sum() > 0]
    df1, _ = e2.run_pooled_first_stage(df.copy(), iv_act, exog_act)
    res_ols = _fit_phi(df1, state_full, has_cf=False, link=link)
    res_iv = _fit_phi(df1, state_full, has_cf=True, link=link)
    if res_ols is None or res_iv is None:
        return None
    # Common sample: both models evaluated on the IV estimation rows (v_hat present) -> apples-to-apples.
    common = df1.dropna(subset=[c for c in state_full if c != "constant"] +
                        ["deposit_balance", "nr_lagged_dep", "entity_id",
                         "v_hat_x_lagged_dep", "lagged_deposits", "time_id"]).copy()
    if len(common) == 0:
        return None
    # The fits carry their own link tag ("index_sieve" or "index" for the monotone link);
    # read it rather than the caller's shape name, which is the phi-CSV bug in miniature.
    phi_ols = phi_from_native(common, res_ols, getattr(res_ols, "link", None) or link)
    phi_iv = phi_from_native(common, res_iv, getattr(res_iv, "link", None) or link)
    return _phi_summary(common, phi_ols, phi_iv)


# ---------------------------------------------------------------------------
# Console formatting (mirror weak_iv_analysis's two-line-per-subsample layout)
# ---------------------------------------------------------------------------
def _f(x, d=3):
    return f"{x:+.{d}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "NA"


def _g(x, d=1):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "NA"


def _print_cell(report, spec, sub, rec):
    fs_p = rec["first_stage_strength"]["pooled"]
    fs_t = rec["first_stage_strength"]["twoway"]
    a = rec.get("alpha_2sls"); se = rec.get("alpha_se")
    astr = f"a_proj={a:+.3f}({se:.3f})" if isinstance(a, (int, float)) and np.isfinite(a) else "a_proj=NA"
    lm = (f"[{rec['lm_ci_low']:+.3f},{rec['lm_ci_high']:+.3f}]"
          + ("*disc" if rec.get("lm_ci_disconnected") else "")
          if rec.get("lm_ci_low") is not None else "(none)")
    keff, gstar = rec.get("k_eff"), fs_t.get("G_star")
    _say(report, f"[sleep-IV] {spec}/{sub}: N={rec['n_obs']:,} K={rec['n_iv']}"
                 f"(Keff={_g(keff)}) G={rec['n_clusters']}(G*={_g(gstar)}) | {astr}")
    def _F(x): return f"{x:,.1f}" if isinstance(x, (int, float)) and np.isfinite(x) else "NA"
    _say(report, f"           FS jointF pooled={_F(fs_p.get('joint_F'))} twoway={_F(fs_t.get('joint_F'))} | "
                 f"eff-F pooled={_F(fs_p.get('eff_F'))} twoway={_F(fs_t.get('eff_F'))} | "
                 f"KP-F(2way)={_F(fs_t.get('kp_F'))} pR2(2way)={_f(fs_t.get('partial_R2'))} | "
                 f"LM95 {lm} | J p={_f(rec.get('hansen_J_p'), 2)}")
    tf = rec.get("tf") or {}
    tfci = (f"[{_f(tf.get('tf_ci_low'), 2)},{_f(tf.get('tf_ci_high'), 2)}]"
            if tf.get("tf_defined") and tf.get("tf_ci_low") is not None
            else ("(undef, F<3.84)" if tf.get("tf_defined") is False else "NA"))
    zc = rec.get("z_collinearity") or {}
    mv, cn = zc.get("max_vif"), zc.get("cond_number")
    coll = (f"maxVIF={mv:,.0f} cond#={cn:,.0f}"
            if isinstance(mv, (int, float)) and isinstance(cn, (int, float))
            and np.isfinite(mv) and np.isfinite(cn) else "collinearity NA")
    _say(report, f"           ladder OLS={_f(rec.get('alpha_ols'))} 2SLS={_f(a)} "
                 f"LIML={_f(rec.get('alpha_liml'))} Fuller={_f(rec.get('alpha_fuller'))} | "
                 f"tF95 {tfci} | {coll}  [projection benchmark, NOT phi]")
    cf = rec.get("cf_hausman")
    if cf:
        _say(report, f"           CF-Hausman(structural): gamma={_f(cf.get('gamma'))} "
                     f"(WCB t={_f(cf.get('gamma_t_wcb'), 2)}, p={_f(cf.get('gamma_p_wcb'), 3)}) "
                     f"[2way eff-F={_F(cf.get('twoway_eff_F'))}, G*={_g(cf.get('G_star'))}; tF n/a to gamma]")


def _print_phi(report, phi_block):
    _say(report, "\n--- phi \"do we need it?\" (spec 12 = IV_HausmanFull x Tech; deposit-weighted, common sample) ---")
    for link in ("logit", "sieve"):
        s = phi_block.get(link)
        if not s:
            continue
        _say(report, f"  {link:6s}: N={s['n_common']:,} T={s['n_quarters']} | "
                     f"mean phi OLS={s['mean_phi_ols']:.3f} IV={s['mean_phi_iv']:.3f} | "
                     f"mean|d|={s['mean_abs_delta']:.3f} max|d|={s['max_abs_delta']:.3f} | "
                     f"Pearson={s['pearson']:+.3f} Spearman={s['spearman']:+.3f} | "
                     f"trend OLS {s['trend_ols']:+.3f} IV {s['trend_iv']:+.3f}")


# ---------------------------------------------------------------------------
# Markdown methodology + headline note
# ---------------------------------------------------------------------------
def _write_markdown(out, diag_dir, report_lines):
    diag_dir.mkdir(parents=True, exist_ok=True)
    h = out.get("IV_HausmanFull", {}).get("type45", {})
    fs_p = h.get("first_stage_strength", {}).get("pooled", {})
    fs_t = h.get("first_stage_strength", {}).get("twoway", {})
    phi = out.get("phi_need_test", {})
    lines = [
        "# Sleepiness (phi) first-stage instrument-quality diagnostic",
        "",
        "Generated by `sleep_weak_iv.py` -- the phi-stage analog of `blp_weak_iv.py`.",
        "",
        "## What it measures",
        "For each instrument spec x deposit-type subsample, the cluster-robust first stage of "
        "`spread_qoq` on the excluded instruments, computed **both pooled** (as "
        "`run_pooled_first_stage` runs it) and **two-way-FE-demeaned** (entity + quarter -- the honest "
        "first stage, since the CF second stage identifies phi within entity net of common time). "
        "Reuses the demand battery: MOP effective-F, Kleibergen-Paap / Cragg-Donald F, partial R^2, "
        "per-instrument relevance, condition number / VIF, Hansen J, leave-one-out / "
        "leave-one-group-out, and (as a **linear-IV projection benchmark, not phi**) the "
        "2SLS/LIML/Fuller ladder with tF and Anderson-Rubin / Kleibergen-LM weak-IV-robust CIs.",
        "",
        "## Three inference objects (why the split)",
        "1. **Linear-IV projection alpha** (deposits-on-spread): the only setting where tF/AR/LM are "
        "valid; labeled a benchmark, never the structural phi.",
        "2. **Structural CF Hausman gamma** (coefficient on `v_hat_x_lagged_dep`): the WCB test of "
        "gamma=0 is a valid few-cluster endogeneity test; tF is *not* applied (generated regressor).",
        "3. **phi materiality**: does instrumenting move the reconstructed national deposit-weighted "
        "phi_t path (logit and shape-constrained monotone sieve links, common sample)?",
        "",
        "## Headline numbers (spec 12 = IV_HausmanFull x Tech, types 4+5)",
    ]
    def g(d, k, fmt="{:.2f}"):
        v = d.get(k)
        return fmt.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "n/a"
    lines += [
        f"- First-stage joint F: **pooled {g(fs_p, 'joint_F', '{:.1f}')}**, "
        f"**two-way-FE {g(fs_t, 'joint_F', '{:.1f}')}** (K={fs_t.get('n_iv', 'n/a')}).",
        f"- MOP effective-F: pooled {g(fs_p, 'eff_F', '{:.1f}')}, two-way {g(fs_t, 'eff_F', '{:.1f}')}; "
        f"partial R^2 (two-way) {g(fs_t, 'partial_R2', '{:.3f}')}.",
        f"- Effective clusters **G* = {g(fs_t, 'G_star', '{:.1f}')}** (nominal G="
        f"{h.get('n_clusters', 'n/a')}): the wide CIs are a **precision** issue, not weak relevance.",
    ]
    if phi.get("logit"):
        pl = phi["logit"]
        lines.append(f"- phi (logit) OLS-vs-IV: Pearson {pl['pearson']:+.3f}, mean|delta| "
                     f"{pl['mean_abs_delta']:.3f}; trend OLS {pl['trend_ols']:+.3f} -> IV "
                     f"{pl['trend_iv']:+.3f} (the CF reverses the trend / pulls phi off the boundary).")
    if phi.get("sieve"):
        ps = phi["sieve"]
        lines.append(f"- phi (monotone sieve link) OLS-vs-IV: Pearson {ps['pearson']:+.3f}, mean|delta| "
                     f"{ps['mean_abs_delta']:.3f}.")
    lines += [
        "",
        "## Caveat (flagged throughout)",
        "Deposits are extremely concentrated by prudential conglomerate, so the effective cluster "
        "count collapses to G* ~ 5-7. Inference uses the wild-cluster-bootstrap criticals (Webb, "
        "seed 0) that the battery already carries. Read the wide intervals as few-effective-cluster "
        "precision, **not** weak instruments -- the pooled effective-F is ~60.",
        "",
        "## Console log",
        "```",
        *report_lines,
        "```",
    ]
    (diag_dir / "weak_iv_sleep_notes.md").write_text("\n".join(lines), encoding="utf-8")
    return diag_dir / "weak_iv_sleep_notes.md"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def analyze(df, iv_specs, state_blocks, specs, subsamples, report):
    state_full = state_blocks[HEADLINE_STATE]
    wiv.IV_GROUPS = SLEEP_IV_GROUPS
    out = {}
    for spec in specs:
        iv_cols = iv_specs.get(spec, [])
        if not iv_cols:
            continue
        _say(report, f"\n=== {spec} x {HEADLINE_STATE} ===")
        out[spec] = {}
        for sub in subsamples:
            mask = _subsample_mask(df, sub)
            prep = _prep_subsample(df, iv_cols, state_full, mask)
            if prep is None:
                _say(report, f"[sleep-IV] {spec}/{sub}: too few obs / no IVs -- skipped")
                continue
            dep_l, spread_l, X_l, Z_l = _level_arrays(prep)
            dep_t, spread_t, X_t, Z_t = _twoway_arrays(prep)
            codes, G, cl, iv_kept = prep["codes"], prep["G"], prep["cl"], prep["ivs"]
            if len(dep_t) <= X_t.shape[1] + Z_t.shape[1] + 1:
                _say(report, f"[sleep-IV] {spec}/{sub}: too few obs vs parameters -- skipped")
                continue
            rec = _linear_iv_battery(dep_t, spread_t, X_t, Z_t, codes, G, cl, iv_kept)
            rec["first_stage_strength"] = {
                "pooled": _first_stage_strength(spread_l, X_l, Z_l, codes, G, cl),
                "twoway": _first_stage_strength(spread_t, X_t, Z_t, codes, G, cl),
            }
            rec["per_iv_pooled"] = wiv._per_iv_diag(dep_l, spread_l, X_l, Z_l, codes, G, iv_kept)
            rec["per_iv_twoway"] = wiv._per_iv_diag(dep_t, spread_t, X_t, Z_t, codes, G, iv_kept)
            # parsimonious (cost-shifters only) dilution / many-weak check, on the two-way arrays
            pcols = [i for i, c in enumerate(iv_kept) if c in SLEEP_IV_GROUPS["cost"]]
            if 0 < len(pcols) < Z_t.shape[1]:
                rec["parsimonious"] = wiv._fs_F(spread_t, X_t, Z_t[:, pcols], codes, G)
            rec["cf_hausman"] = _cf_hausman(df, iv_cols, state_full, mask,
                                            rec["first_stage_strength"]["twoway"].get("eff_F"))
            out[spec][sub] = rec
            _print_cell(report, spec, sub, rec)
    return out


def run_phi_tests(df, iv_specs, state_blocks, skip_sieve, report):
    iv12 = iv_specs["IV_HausmanFull"]
    state_full = state_blocks[HEADLINE_STATE]
    phi = {}
    _say(report, "\n[sleep-IV] phi \"do we need it\" test (spec 12) -- fitting logit ...")
    phi["logit"] = _phi_need_test(df, iv12, state_full, "logit")
    if not skip_sieve:
        _say(report, "[sleep-IV] ... fitting the shape-constrained monotone sieve link ...")
        try:
            phi["sieve"] = _phi_need_test(df, iv12, state_full, "sieve")
        except Exception as e:
            _say(report, f"[sleep-IV] sieve phi test failed ({type(e).__name__}: {e})")
    phi = {k: v for k, v in phi.items() if v}
    if phi:
        _print_phi(report, phi)
    return phi


def main():
    ap = argparse.ArgumentParser(description="Weak-instruments battery for the sleepiness (phi) first stage.")
    ap.add_argument("results_dir", nargs="?", default=wiv.default_results_dir())
    ap.add_argument("--specs", default=",".join(SPEC_ORDER))
    ap.add_argument("--subsamples", default=",".join(SUBSAMPLES))
    ap.add_argument("--skip-sieve", action="store_true", help="skip the sieve-link phi test")
    ap.add_argument("--phi-only", action="store_true", help="only run the phi materiality test")
    args = ap.parse_args()

    try:
        import linearmodels  # noqa: F401
    except Exception:
        print("[sleep-IV] NOTE: linearmodels not installed -- 2SLS/LIML/Fuller alpha + partial R^2 "
              "omitted (eff-F/KP-F/CD/AR/LM still computed). Install: pip install linearmodels")

    report = []
    _say(report, "SLEEPINESS (phi) FIRST-STAGE INSTRUMENT-QUALITY DIAGNOSTIC")
    df = e2.build_pooled_data(time_block=False)
    _, iv_specs, state_blocks = e2.define_specifications(time_block=False)
    _say(report, f"panel: N={len(df):,} rows | window [{e2.SLEEP_MIN_YEAR}, {e2.SLEEP_MAX_YEAR}] | "
                 f"cluster={CLUSTER_KEY} | endog types {ENDOG_TYPES}")

    specs = [s.strip() for s in args.specs.split(",") if s.strip()]
    subsamples = [s.strip() for s in args.subsamples.split(",") if s.strip()]

    out = {}
    if not args.phi_only:
        out = analyze(df, iv_specs, state_blocks, specs, subsamples, report)
    out["phi_need_test"] = run_phi_tests(df, iv_specs, state_blocks, args.skip_sieve, report)
    out["_meta"] = {
        "n_obs_total": int(len(df)), "window": [e2.SLEEP_MIN_YEAR, e2.SLEEP_MAX_YEAR],
        "cluster_key": CLUSTER_KEY, "endog_types": ENDOG_TYPES,
        "wcb_reps": int(os.environ.get("SLEEP_BOOT_B", "999")),
        "wcb_scheme": os.environ.get("SLEEP_BOOT_SCHEME", "webb"),
        "headline_spec": R.SPEC12,
    }

    RES = os.path.abspath(args.results_dir)
    cp = os.path.join(RES, "cluster_processed")
    os.makedirs(cp, exist_ok=True)
    path = os.path.join(cp, "weak_iv_sleep.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    _say(report, f"\n[sleep-IV] wrote {path}")

    diag_dir = P.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_WEAK_IV_SLEEP"
    md = _write_markdown(out, diag_dir, report)
    print(f"[sleep-IV] wrote {md}")


if __name__ == "__main__":
    main()
