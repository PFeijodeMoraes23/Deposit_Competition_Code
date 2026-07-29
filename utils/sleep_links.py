"""
utils/sleep_links.py
================================================================================
Shared estimation kernels for the link-based sleepiness routines Est4/Est5/Est6
(Constrained Linear / Probit / Single-Index). These generalise the logistic NLLS
of estimation_3_sleep.py to an arbitrary CDF link G for the sleepiness function

    phi_mt = F_eta(index) = G( S_mt' theta ),

so each estimator is one choice of the depositor-attention shock distribution:

    'logit'   -> eta ~ Logistic   (Est3; kept inline in estimation_3_sleep.py)
    'probit'  -> eta ~ Normal     (Est5)
    'uniform' -> eta ~ Uniform    (Est4 = constrained linear, bound imposed)
    'index'   -> eta ~ nonparametric, monotone (Est6 = cubic sieve single index)

IMPORTANT (user guardrail): Average Marginal Effects are for REPORTING ONLY.
Every phi construction — each routine's market_panel_phis.csv and the demand-prep
shares — uses the NATIVE index coefficients + the link via ``phi_from_native``,
never the AMEs. ``NonLinearResults.params`` holds AMEs (tables);
``NonLinearResults.params_native`` holds the native index coefficients (phi).

Inference (Est3/4/5/6): a score/multiplier WILD CLUSTER BOOTSTRAP at the
conglomerate level (Cameron-Gelbach-Miller 2008; MacKinnon-Webb 2017), perturbing
the cluster-summed influence functions by wild weights and re-reading the AMEs.
This replaces the earlier homoskedastic NLLS delta-method SEs (Est3/4/5 were NOT
cluster-robust) and the conditional clustered-OLS SE (Est6). The Imbens-Kolesar
(2016) BRL does not apply to these nonlinear / profiled-link estimators (no linear
hat matrix); the Carter-Schnepel-Steigerwald (2017) G* effective-cluster count is
retained only as a reported diagnostic. References for the single index: Ichimura
(1993); Klein & Spady (1993); Robertson, Wright & Dykstra (1988, PAVA).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
from scipy.optimize import least_squares


# ==============================================================================
# Results container (matches estimation_3_sleep.NonLinearResults so the table
# exporters and downstream code see an identical interface).
# ==============================================================================
class NonLinearResults:
    def __init__(self, params, bse, tvalues, pvalues, df_resid, params_native=None,
                 nobs=None, rsquared=None, fvalue=None, f_pvalue=None, G_nominal=None,
                 cov_ame=None, nlls_status=None, nlls_message=None,
                 si_b=None, si_vmu=None, si_vsd=None, link=None):
        self.params = params
        self.bse = bse
        self.tvalues = tvalues
        self.pvalues = pvalues
        self.df_resid = df_resid
        self.params_native = params_native if params_native is not None else params
        self.G_star = df_resid
        self.nobs = nobs
        self.rsquared = rsquared
        self.fvalue = fvalue
        self.f_pvalue = f_pvalue
        self.G_nominal = G_nominal
        self.cov_ame = cov_ame
        self.nlls_status = nlls_status
        self.nlls_message = nlls_message
        # Single-index sieve parameters (Est6 only): link G(v)=sum_d si_b[d]*((v-mu)/sd)^d
        self.si_b = si_b
        self.si_vmu = si_vmu
        self.si_vsd = si_vsd
        self.link = link

    def cov_params(self):
        if self.cov_ame is not None:
            return pd.DataFrame(self.cov_ame, index=self.params.index, columns=self.params.index)
        return pd.DataFrame(np.diag(self.bse ** 2), index=self.params.index, columns=self.params.index)


# ==============================================================================
# Link helpers
# ==============================================================================
def link_cdf(z, link):
    if link == "probit":
        return norm.cdf(z)
    if link == "uniform":
        return np.clip(z, 0.0, 1.0)
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))
    raise ValueError(f"unknown link {link!r}")


def _demean_col(df_ss, col):
    return (df_ss[col] - df_ss.groupby("entity_id")[col].transform("mean")).values.astype(float)


def _G_star(cluster_series):
    from utils.cluster import effective_cluster_stats   # single source of G*/CV
    st = effective_cluster_stats(cluster_series.value_counts().values)
    return st["G_star"], st["G_nominal"]


def _build_phi_X(df: pd.DataFrame, phi_params) -> np.ndarray:
    """Regressor matrix for phi: 'nr_lagged_dep' -> 1, 'interaction_{sv}' -> sv."""
    n = len(df)
    X = np.zeros((n, len(phi_params)))
    for i, p in enumerate(phi_params):
        if p == "nr_lagged_dep":
            X[:, i] = 1.0
        elif str(p).startswith("interaction_"):
            sv = p[len("interaction_"):]
            if sv in df.columns:
                X[:, i] = df[sv].fillna(0.0).values.astype(float)
    return X


# ==============================================================================
# Generic link NLLS  (Est4 uniform, Est5 probit; also used for Est6's direction)
# ==============================================================================
def _nlls_resid(params, y_dm, X, Z, CF, entity_idx, link, ecounts=None, tinv=None, tcounts=None):
    K = X.shape[1]
    phi = link_cdf(X @ params[:K], link)
    yhat = phi * Z
    if CF.shape[1] > 0:
        yhat = yhat + CF @ params[K:]
    if tinv is None:
        sums = np.bincount(entity_idx, weights=yhat)
        cnt = np.bincount(entity_idx)
        yhat_dm = yhat - (sums / cnt)[entity_idx]
    else:                                   # two-way (entity + time) FE
        yhat_dm = _twoway_demean(yhat, entity_idx, ecounts, tinv, tcounts)
    return y_dm - yhat_dm


def _link_density(z, link):
    """Link density g'(z) = dF_eta/dz for the analytic average marginal effect."""
    if link == "probit":
        return norm.pdf(z)
    if link == "uniform":
        return ((z > 0.0) & (z < 1.0)).astype(float)
    if link == "logit":
        p = link_cdf(z, link)
        return p * (1.0 - p)
    raise ValueError(f"unknown link {link!r}")


def _dummy_spec(X, K, names=None):
    """Per-column (is_dummy, lo, hi) driving the discrete-difference AME.

    A binary regressor's estimand is the discrete difference between its two
    levels, not a derivative.  The levels used to be assumed to be literally
    {0, 1}.  Once the state block is GRAND-MEAN CENTRED a dummy takes values
    {-p_bar, 1-p_bar}, that test fails silently, and the dummy reverts to a
    continuous average derivative -- the exact defect fixed on 2026-06-26.

    So the levels come from utils.state_transform (explicit metadata, which
    cannot drift with the data).  The fallback for columns that registry does
    not know about is deliberately "two distinct values exactly 1 apart": it
    covers both {0,1} and any centred/shifted image of it, and it is a strict
    superset of the old rule, so behaviour is unchanged when nothing is centred.
    """
    from utils.state_transform import dummy_levels, NEVER_BINARY

    flags = np.zeros(K, dtype=bool)
    lo = np.zeros(K, dtype=float)
    hi = np.ones(K, dtype=float)
    for k in range(K):
        nm = str(names[k]).replace("interaction_", "") if names is not None and k < len(names) else None
        if nm in NEVER_BINARY:
            continue
        lv = dummy_levels(nm) if nm else None
        if lv is not None:
            # STALE-TRANSFORM GUARD. The registry levels are only right if this frame
            # was centred by exactly the persisted mean. If the column is two-valued
            # and those values disagree, the discrete difference would be computed
            # between the wrong pair -- silently, and still inside [0,1]. Fail loudly.
            u = np.unique(X[:, k][~np.isnan(X[:, k])])
            if len(u) == 2 and max(abs(u[0] - lv[0]), abs(u[1] - lv[1])) > 1e-8:
                raise ValueError(
                    f"state_transform: '{nm}' carries levels {tuple(np.round(u, 8))} but the "
                    f"registry says {tuple(np.round(lv, 8))}. The frame was built under a "
                    "different centering than state_centering_means.json. Rebuild the frame, "
                    "or rebuild the JSON with `python estimation_2_sleep.py --write-centers`."
                )
        else:
            u = np.unique(X[:, k][~np.isnan(X[:, k])])
            if len(u) == 2 and abs((u[1] - u[0]) - 1.0) < 1e-9:
                lv = (float(u[0]), float(u[1]))
        if lv is not None:
            flags[k], lo[k], hi[k] = True, lv[0], lv[1]
    return flags, lo, hi


def _generic_ame(theta_full, X, link, K, G, h=1e-5, dummy_spec=None, names=None):
    """Analytic average marginal effects: continuous regressors use the exact
    derivative E[g'(S'theta)]*theta_k; dummies use the discrete difference
    G(idx+theta_k(hi-x)) - G(idx+theta_k(lo-x)) between their two levels, all
    without copying X (fast in the bootstrap loop). At (lo,hi)=(0,1) this is
    identical to the pre-centering formula. dummy_spec may be precomputed."""
    theta_X = theta_full[:K]
    idx = X @ theta_X
    mean_dens = float(np.mean(_link_density(idx, link)))
    flags, lo, hi = _dummy_spec(X, K, names) if dummy_spec is None else dummy_spec
    AME = np.zeros(K + G)
    for k in range(K):
        if flags[k]:
            col = X[:, k]
            idx1 = idx + theta_X[k] * (hi[k] - col)
            idx0 = idx + theta_X[k] * (lo[k] - col)
            AME[k] = float(np.mean(link_cdf(idx1, link) - link_cdf(idx0, link)))
        else:
            AME[k] = mean_dens * theta_X[k]
    if G > 0:
        AME[K:] = theta_full[K:]
    return AME


def _generic_ame_cov(theta_full, cov_full, X, link, K, G, h=1e-5, names=None):
    spec = _dummy_spec(X, K, names)
    AME = _generic_ame(theta_full, X, link, K, G, dummy_spec=spec)
    J = np.zeros((K + G, K + G))
    for i in range(K + G):
        tp = theta_full.copy(); tp[i] += h
        J[:, i] = (_generic_ame(tp, X, link, K, G, dummy_spec=spec) - AME) / h
    cov_AME = J @ cov_full @ J.T
    return AME, np.sqrt(np.abs(np.diag(cov_AME))), cov_AME


def _linear_warm_start(y_dm, X, Z, CF):
    """OLS warm start for the uniform link: regress y_dm on (s*Z) and CF."""
    R = X * Z[:, None]
    design = np.column_stack([R, CF]) if CF.shape[1] > 0 else R
    try:
        beta, *_ = np.linalg.lstsq(design, y_dm, rcond=None)
        return beta
    except Exception:
        return np.zeros(X.shape[1] + CF.shape[1])


def fit_nlls_link(df, state_cols, has_cf, link, loss="cauchy", fe_time_col=None,
                  bootstrap=True, init=None, n_starts=1):
    """NLLS sleepiness fit with link in {'logit','probit','uniform'}. Returns a
    NonLinearResults (params = AMEs for tables; params_native = index coefs for phi).
    fe_time_col (e.g. 'time_id') adds a second additive FE => two-way (entity+time)
    concentration; None reproduces the entity-only within estimator exactly.

    bootstrap=False skips the wild cluster bootstrap and returns NaN AMEs/SEs, keeping
    only params_native (and the index). Used by the single-index/joint-sieve estimators,
    which fit this logit purely to warm-start theta and never read its AMEs or SEs."""
    CF_cols = ["v_hat_x_lagged_dep"] if has_cf else []
    cols = state_cols + ["deposit_balance", "nr_lagged_dep", "entity_id"]
    df_ss = df.dropna(subset=cols + CF_cols).copy()
    if len(df_ss) == 0:
        return None

    entities = df_ss["entity_id"].unique()
    emap = {e: i for i, e in enumerate(entities)}
    entity_idx = df_ss["entity_id"].map(emap).values
    ecounts = np.bincount(entity_idx).astype(float)
    if fe_time_col is not None:
        _, tinv = np.unique(df_ss[fe_time_col].values, return_inverse=True)
        tcounts = np.bincount(tinv).astype(float)
    else:
        tinv = tcounts = None
    y_raw = df_ss["deposit_balance"].values.astype(float)
    y_dm = _twoway_demean(y_raw, entity_idx, ecounts, tinv, tcounts)
    X = df_ss[state_cols].values.astype(float)
    Z = df_ss["nr_lagged_dep"].values.astype(float)
    CF = df_ss[CF_cols].values.astype(float) if has_cf else np.empty((len(df_ss), 0), dtype=float)
    K, G = X.shape[1], CF.shape[1]

    # `init` lets a caller probe whether this single start is landing in a local
    # optimum. E3/E4 otherwise run ONE start from zeros, and E5/E6 inherit whatever
    # direction that produces (fit_single_index never re-optimises theta), so a bad
    # basin here propagates silently to four reported estimators.
    _args = (y_dm, X, Z, CF, entity_idx, link, ecounts, tinv, tcounts)

    if init is not None:
        starts = [("caller", np.asarray(init, float))]
        if starts[0][1].shape != (K + G,):
            raise ValueError(f"init has shape {starts[0][1].shape}, expected {(K + G,)}")
    else:
        # MULTISTART. This used to be a single start from zeros, which is thin for a
        # non-convex M-estimator -- and it propagates: fit_single_index (E5/E6) never
        # re-optimises theta, it inherits whatever direction this fit produces. On the
        # joint-sieve full-sample candidate scan the logit direction scored WORST of four
        # (164,913 vs 160,940 for the best), so the inherited direction was measurably poor.
        starts = [("zeros", np.zeros(K + G))]                       # production baseline
        lw = _linear_warm_start(y_dm, X, Z, CF)                     # OLS-implied direction
        if np.all(np.isfinite(lw)):
            starts.append(("linear", lw))
        # Random directions, SCALE-AWARE: the state columns are centred but not
        # standardised (gdp_per_capita ~O(1) vs fraction_65plus ~O(0.03)), so an isotropic
        # draw in raw coefficient space would be dominated by the small-SD columns. Draw in
        # standardised space and divide back, then normalise so the index has unit variance.
        sd = X.std(axis=0); sd[sd <= 0] = 1.0
        rng_ms = np.random.default_rng(12345)
        while len(starts) < max(1, int(n_starts)):
            th = rng_ms.standard_normal(K) / sd
            v = X @ th
            s = float(v.std())
            if not np.isfinite(s) or s <= 0:
                continue
            starts.append((f"rand{len(starts)}", np.concatenate([th / s, np.zeros(G)])))

    best = None
    costs = []
    for nm, s0 in starts:
        try:
            r = least_squares(_nlls_resid, s0, args=_args, method="trf", loss=loss)
        except Exception as exc:                      # a bad start must not kill the fit
            print(f"  [NLLS-{link}] start {nm} failed: {exc}")
            continue
        costs.append((float(r.cost), nm))
        if best is None or r.cost < best[0].cost:
            best = (r, nm)
    if best is None:
        return None
    res_lsq, best_nm = best

    if len(costs) > 1:
        costs.sort()
        spread = costs[-1][0] - costs[0][0]
        print(f"  [NLLS-{link}] multistart: " + ", ".join(f"{nm}={c:.6g}" for c, nm in costs) +
              f" | winner={best_nm} | spread={spread:.3g}")
        if spread <= 1e-6 * max(1.0, abs(costs[0][0])):
            print(f"  [NLLS-{link}] NOTE: starts agree to <1e-6 -- the index direction is "
                  "weakly identified here; the optimizer is not what is choosing it.")
        elif best_nm != "zeros":
            print(f"  [NLLS-{link}] NOTE: '{best_nm}' beat the production 'zeros' start "
                  f"by {costs[-1][0] - costs[0][0]:.6g} -- the single-start fit was in a "
                  "worse basin.")
    idx = [f"interaction_{sv}" if sv != "constant" else "nr_lagged_dep" for sv in state_cols] + CF_cols

    cl = df_ss["CodConglomeradoPrudencial"].astype(str)
    G_star, G_nominal = _G_star(cl)
    cl_u, cl_inv = np.unique(cl.values, return_inverse=True)
    n_cl = len(cl_u)

    # Inference: score/multiplier wild cluster bootstrap (replaces the old
    # homoskedastic delta-method SEs, which were NOT cluster-robust).
    if bootstrap:
        ame_d, bse_d, pval_d = nlls_link_wild_bootstrap(res_lsq, X, link, K, G, idx,
                                                        cl_inv, n_cl,
                                                        state_names=state_cols)
        ps = pd.Series(ame_d).reindex(idx)
        bs = pd.Series(bse_d).reindex(idx)
        pvals = pd.Series(pval_d).reindex(idx)
        tvals = ps / bs.replace(0, np.nan)
    else:
        ps = pd.Series(np.nan, index=idx)
        bs = pd.Series(np.nan, index=idx)
        pvals = pd.Series(np.nan, index=idx)
        tvals = pd.Series(np.nan, index=idx)

    tss = float(np.sum((y_dm - y_dm.mean()) ** 2))
    rss = float(np.sum(res_lsq.fun ** 2))
    rsq = 1 - rss / tss if tss > 0 else np.nan
    ps_native = pd.Series(res_lsq.x, index=idx)
    B_used, scheme_used = boot_cfg()
    boot_msg = f"wild boot B={B_used} ({scheme_used})" if bootstrap else "wild boot SKIPPED (warm start)"
    print(f"  [NLLS-{link}] status={res_lsq.status} | nfev={res_lsq.nfev} | "
          f"cost={res_lsq.cost:.4g} | {boot_msg}")
    return NonLinearResults(ps, bs, tvals, pvals, G_star, params_native=ps_native,
                            nobs=len(y_dm), rsquared=rsq, G_nominal=G_nominal,
                            cov_ame=None, link=link)


# ==============================================================================
# Monotone single index  (Est6): logit-direction index + smooth cubic sieve link
# ==============================================================================
def pava_increasing(y, w=None):
    y = np.asarray(y, float)
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, float)
    bval, bw, bidx = [], [], []
    for j in range(n):
        cv, cw, ci = y[j], w[j], [j]
        while bval and bval[-1] > cv:
            pv, pw, pi = bval.pop(), bw.pop(), bidx.pop()
            cv = (pv * pw + cv * cw) / (pw + cw)
            cw = pw + cw
            ci = pi + ci
        bval.append(cv); bw.append(cw); bidx.append(ci)
    out = np.empty(n)
    for v_, idx_ in zip(bval, bidx):
        for k in idx_:
            out[k] = v_
    return out


def fit_single_index(df, state_cols, has_cf, logit_res, degree=3, phi_band=False,
                     fe_time_col=None):
    """Smooth cubic-sieve monotone single index in the logit-direction index.
    Returns NonLinearResults carrying average-derivative AMEs (params; tables),
    the index direction (params_native = logit native coefs), and the sieve
    params si_b/si_vmu/si_vsd (used by phi_from_native for Est6's phi).
    fe_time_col adds a second additive FE (two-way entity+time concentration)."""
    phi_params = [p for p in logit_res.params.index if not str(p).startswith("v_hat")]
    theta = logit_res.params_native[phi_params].values.astype(float)

    CF_cols = ["v_hat_x_lagged_dep"] if has_cf else []
    cols = state_cols + ["deposit_balance", "nr_lagged_dep", "entity_id"]
    df_ss = df.dropna(subset=cols + CF_cols).copy()
    if len(df_ss) == 0:
        return None

    X = _build_phi_X(df_ss, phi_params)
    v = X @ theta
    Z = df_ss["nr_lagged_dep"].values.astype(float)
    vmu, vsd = float(v.mean()), float(v.std())
    if vsd <= 0:
        vsd = 1.0
    vs = (v - vmu) / vsd
    P = np.column_stack([vs ** d for d in range(degree + 1)])
    R = P * Z[:, None]
    cols_p = [f"_p{d}" for d in range(degree + 1)]
    work = pd.DataFrame(R, columns=cols_p, index=df_ss.index)
    work["entity_id"] = df_ss["entity_id"].values
    work["_y"] = df_ss["deposit_balance"].values
    design = cols_p[:]
    if has_cf:
        work["_cf"] = df_ss["v_hat_x_lagged_dep"].values
        design = cols_p + ["_cf"]
    _, einv = np.unique(df_ss["entity_id"].values, return_inverse=True)
    ecounts = np.bincount(einv).astype(float)
    if fe_time_col is not None:
        _, tinv = np.unique(df_ss[fe_time_col].values, return_inverse=True)
        tcounts = np.bincount(tinv).astype(float)
    else:
        tinv = tcounts = None
    y_dm = _twoway_demean(work["_y"].values.astype(float), einv, ecounts, tinv, tcounts)
    Xdm_arr = _twoway_demean(work[design].values.astype(float), einv, ecounts, tinv, tcounts)

    cl = df_ss["CodConglomeradoPrudencial"].astype(str)
    res = sm.OLS(y_dm, Xdm_arr).fit()
    b_full = np.asarray(res.params, float)
    b = b_full[:degree + 1]
    G_star, G_nominal = _G_star(cl)
    cl_u, cl_inv = np.unique(cl.values, return_inverse=True)
    n_cl = len(cl_u)

    # average-derivative AME_k = theta_k * mean_i G'(v_i), with G'(v) read off the
    # sieve coefficients b. mean_slope depends only on b[:degree+1].
    sv_pows = np.column_stack([vs ** (d - 1) for d in range(1, degree + 1)]) \
        if degree >= 1 else np.empty((len(vs), 0))
    dcoef = np.arange(1, degree + 1, dtype=float)

    def _mean_slope(bfull):
        bb = np.asarray(bfull, float)[1:degree + 1]
        return float(np.mean(sv_pows @ (dcoef * bb))) / vsd if degree >= 1 else 0.0

    names = [nm for nm in phi_params if nm != "nr_lagged_dep"]
    ths = {nm: th for nm, th in zip(phi_params, theta) if nm != "nr_lagged_dep"}

    # 0/1 DUMMIES get the DISCRETE-DIFFERENCE AME, not the continuous average
    # derivative: the marginal effect of a binary regressor is
    # E[G(idx | x=1) - G(idx | x=0)] on the structural (clipped) sieve link, which
    # is bounded to [-1,1]. Treating a dummy as continuous (theta_k * mean_slope)
    # is the wrong estimand and explodes when the inherited (unnormalised) logit
    # direction gives a weakly-identified dummy a huge coefficient (e.g. pix_exists,
    # near-collinear with the quarter FE -> theta_pix ~ 300 -> AME ~ 6.9). The flip
    # terms below depend only on (vs, theta_k, vsd, the dummy column) -- all fixed
    # across the link bootstrap -- so we precompute their standardised-index powers.
    # Levels come from _dummy_spec (registry-driven), so a CENTRED dummy with values
    # {-p_bar, 1-p_bar} is still recognised; a literal {0,1} test would not see it.
    _dsub = X[:, [phi_params.index(nm) for nm in names]]
    _dflag, _dlo, _dhi = _dummy_spec(_dsub, len(names), names)
    dcols = {}
    for j, nm in enumerate(names):
        if not _dflag[j]:
            continue
        coln = _dsub[:, j]
        z1 = vs + ths[nm] * (_dhi[j] - coln) / vsd     # standardised index at the HIGH level
        z0 = vs + ths[nm] * (_dlo[j] - coln) / vsd     # standardised index at the LOW level
        dcols[nm] = (np.column_stack([z1 ** dd for dd in range(degree + 1)]),
                     np.column_stack([z0 ** dd for dd in range(degree + 1)]))

    def _ame_fn(bfull):
        ms = _mean_slope(bfull)
        bb = np.asarray(bfull, float)[:degree + 1]
        out = {}
        for nm in names:
            if nm in dcols:                              # discrete difference (bounded)
                Pz1, Pz0 = dcols[nm]
                out[nm] = float(np.mean(np.clip(Pz1 @ bb, 0.0, 1.0)
                                        - np.clip(Pz0 @ bb, 0.0, 1.0)))
            else:                                        # continuous average derivative
                out[nm] = ths[nm] * ms
        return out

    mean_slope = _mean_slope(b_full)
    ame = _ame_fn(b_full)
    # Inference: score/multiplier wild cluster bootstrap on the sieve OLS
    # (replaces the earlier conditional clustered-OLS delta-method SE).
    bse, pv = ols_sieve_wild_bootstrap(Xdm_arr, np.asarray(res.resid, float),
                                       cl_inv, n_cl, b_full, _ame_fn, ame)
    ps = pd.Series(ame); bs = pd.Series(bse); pvs = pd.Series(pv)
    tss = float(np.sum((y_dm - y_dm.mean()) ** 2))
    rss = float(np.sum(res.resid ** 2))
    rsq = 1 - rss / tss if tss > 0 else getattr(res, "rsquared", np.nan)
    B_used, scheme_used = boot_cfg()
    print(f"  [SingleIndex] cubic sieve deg={degree} | mean_slope={mean_slope:.4g} | "
          f"wild boot B={B_used} ({scheme_used})")
    res_obj = NonLinearResults(ps, bs, ps / bs.replace(0, np.nan), pvs, G_star,
                               params_native=pd.Series(theta, index=phi_params),
                               nobs=len(df_ss), rsquared=rsq, G_nominal=G_nominal,
                               cov_ame=None, si_b=b, si_vmu=vmu, si_vsd=vsd, link="index")
    # National phi_t band from the sieve-link (b) wild cluster bootstrap; the
    # logit index direction is held fixed, so the band reflects link uncertainty.
    if phi_band:
        bread = np.linalg.pinv(Xdm_arr.T @ Xdm_arr)
        score = Xdm_arr * np.asarray(res.resid, float)[:, None]
        IF_cl = _cluster_if(score, bread, cl_inv, n_cl)
        gs = _phi_t_group_struct(df_ss)
        vpow = np.column_stack([vs ** dd for dd in range(degree + 1)])

        def _phi_of_b(bfull):
            return np.clip(vpow @ np.asarray(bfull, float)[:degree + 1], 0.0, 1.0)

        _B, _scheme = boot_cfg()

        def _draw(rng_):
            w = _wild_weights(n_cl, _scheme, rng_)
            return _phi_of_b(b_full + w @ IF_cl)

        res_obj.phi_t_boot = _phi_t_band(gs, _phi_of_b(b_full), _draw,
                                         min(_B, 400), _scheme, np.random.default_rng(20240624))
    return res_obj


# ==============================================================================
# Canonical phi construction (used by BOTH market_panel_phis and demand prep).
# Always native index + link; AMEs never enter here.
# ==============================================================================
def phi_from_native(df: pd.DataFrame, res, link: str) -> np.ndarray:
    """phi_mt at each row of df from the native index coefficients + the link.
    link in {'logit','probit','uniform','index'}. Returns phi in [0,1]."""
    native = res.params_native
    phi_params = [p for p in native.index if not str(p).startswith("v_hat")]
    X = _build_phi_X(df, phi_params)
    index = X @ native[phi_params].values.astype(float)
    if link == "index":
        b = np.asarray(res.si_b)
        vs = (index - res.si_vmu) / (res.si_vsd if res.si_vsd else 1.0)
        g = np.zeros(len(vs))
        for d in range(len(b)):
            g += b[d] * vs ** d
        return np.clip(g, 0.0, 1.0)
    if link in ("sieve", "kernel"):
        # Joint single index (Est7/Est8): index over the NON-constant state vars,
        # link stored as a monotone grid (kernel) or B-spline (sieve). Both are
        # evaluated through si_vgrid/si_ggrid (a fine monotone lookup) for a single
        # canonical path. Index uses theta over interaction_* only (no constant).
        idx_params = [p for p in phi_params if p != "nr_lagged_dep"]
        Xi = _build_phi_X(df, idx_params)
        vv = Xi @ native[idx_params].values.astype(float)
        g = np.interp(vv, res.si_vgrid, res.si_ggrid)
        return np.clip(g, 0.0, 1.0)
    return np.clip(link_cdf(index, link), 0.0, 1.0)


# ==============================================================================
# JOINT SINGLE INDEX (Est7 sieve, Est8 kernel) — Ichimura (1993) SLS
# ==============================================================================
# Estimate the index direction theta AND the link G jointly by semiparametric
# least squares: min over theta (||theta||=1) of the entity-demeaned SSR of
#   D - G(S'theta)*Z - gamma*(vhat*Z),
# with G profiled out at each theta. Est7: monotone cubic B-spline link with
# ordered coefficients (Ramsay 1988; Newey 1997). Est8: kernel local-linear link
# (Ichimura 1993; Fan 1992), monotonised by rearrangement (CFG 2009). Inference:
# score/multiplier wild cluster bootstrap (Kline-Santos 2012). phi is built from
# theta + the stored monotone link via phi_from_native; AMEs are reporting-only.
def _fast_demean(M, entity_idx, counts):
    """Within-entity demean each column of M (n x k) given integer entity_idx."""
    M = np.asarray(M, float)
    if M.ndim == 1:
        s = np.bincount(entity_idx, weights=M, minlength=len(counts))
        return M - (s / counts)[entity_idx]
    out = np.empty_like(M)
    for j in range(M.shape[1]):
        s = np.bincount(entity_idx, weights=M[:, j], minlength=len(counts))
        out[:, j] = M[:, j] - (s / counts)[entity_idx]
    return out


def _twoway_demean(M, einv, ecounts, tinv, tcounts, n_iter=15, tol=1e-9):
    """Two-way (entity + time) additive FE removal by alternating projections
    (Gaure 2013). Concentrates out D = ... + alpha_i + delta_t exactly in the
    least-squares sense. Reduces to one-way entity demeaning when tinv is None."""
    M = np.asarray(M, float)
    if tinv is None:
        return _fast_demean(M, einv, ecounts)
    one_d = M.ndim == 1
    X = (M.reshape(-1, 1) if one_d else M).astype(float, copy=True)
    for _ in range(n_iter):
        prev = X.copy()
        for j in range(X.shape[1]):          # entity sweep
            s = np.bincount(einv, weights=X[:, j], minlength=len(ecounts))
            X[:, j] -= (s / ecounts)[einv]
        for j in range(X.shape[1]):          # time sweep
            s = np.bincount(tinv, weights=X[:, j], minlength=len(tcounts))
            X[:, j] -= (s / tcounts)[tinv]
        if np.max(np.abs(X - prev)) < tol:
            break
    return X.ravel() if one_d else X


def _bspline_design(v, n_interior, degree=3):
    """Cubic B-spline design at interior knots placed on quantiles of v.
    Returns (B [n x K], knots). Monotone G = sum_k c_k B_k(v) iff c is nondecreasing."""
    from scipy.interpolate import BSpline
    v = np.asarray(v, float)
    lo, hi = np.min(v), np.max(v)
    if hi <= lo:
        hi = lo + 1.0
    qs = np.linspace(0, 1, n_interior + 2)[1:-1]
    interior = np.quantile(v, qs) if n_interior > 0 else np.array([])
    interior = np.clip(interior, lo + 1e-9, hi - 1e-9)
    # clamped knot vector
    t = np.concatenate(([lo] * (degree + 1), np.sort(interior), [hi] * (degree + 1)))
    K = len(t) - degree - 1
    B = BSpline.design_matrix(np.clip(v, lo, hi), t, degree).toarray()
    return B, t


def _ramp_design(v, t, degree=3):
    """Monotone I-spline-style ramps R_j(v)=sum_{k>=j} B_k(v) from the cubic
    B-spline basis on knot vector t. R_1≡1; R_2..R_K rise 0→1. A nonneg combination
    sum_j beta_j R_j(v) is monotone nondecreasing (beta_1 is the floor)."""
    from scipy.interpolate import BSpline
    lo, hi = t[0], t[-1]
    B = BSpline.design_matrix(np.clip(v, lo, hi), t, degree).toarray()
    R = np.cumsum(B[:, ::-1], axis=1)[:, ::-1]   # R[:,j] = sum_{k>=j} B[:,k]
    return R


def _fit_link_sieve(R, Z, cf_dm, y_dm, einv, counts, w=None, dm=None):
    """Profiled monotone sieve link given the index. Design = [R_j*Z]_j (+ CF),
    entity-demeaned; coefficients beta_j >= 0 (monotone, floor at beta_1), gamma free.
    Solved by bounds-constrained LS (fast). Returns (beta, gamma, resid, ssr, c)
    where c = cumsum(beta) are the B-spline coefficients of G. `dm`, if given, is a
    demeaner callable (e.g. two-way entity+time FE) used in place of entity demeaning."""
    from scipy.optimize import lsq_linear
    K = R.shape[1]
    Rz = R * Z[:, None]
    cols = [Rz] + ([cf_dm[:, None]] if cf_dm is not None else [])
    X = np.column_stack(cols)
    Xdm = dm(X) if dm is not None else _fast_demean(X, einv, counts)
    sw = np.ones(len(y_dm)) if w is None else np.sqrt(np.maximum(w, 0.0))
    p = X.shape[1]
    lb = np.r_[np.zeros(K), np.full(p - K, -np.inf)]
    ub = np.full(p, np.inf)
    sol = lsq_linear(Xdm * sw[:, None], y_dm * sw, bounds=(lb, ub),
                     method="bvls", max_iter=200)
    b = sol.x
    beta = b[:K]
    gamma = b[K:] if p > K else np.array([])
    resid = y_dm - Xdm @ b
    ww = np.ones(len(y_dm)) if w is None else w
    ssr = float(np.sum(ww * resid * resid))
    return beta, gamma, resid, ssr, np.cumsum(beta)


def _cauchy_weights(resid, scale=None):
    """IRLS weights for the Cauchy (Lorentzian) robust loss; scale = MAD-based."""
    r = np.asarray(resid, float)
    s = scale if scale else (1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12)
    return 1.0 / (1.0 + (r / (2.385 * s)) ** 2)


# ---- Est8 kernel local-linear link (joint SLS via backfitting) ---------------
def _kernel_bw(v):
    """Undersmoothed Gaussian bandwidth (Silverman x undersmoothing factor).
    Undersmoothing keeps the link bias negligible for sqrt-n theta inference
    (Hardle-Hall-Ichimura 1993)."""
    v = np.asarray(v, float)
    n = len(v)
    sd = float(np.std(v))
    q75, q25 = np.percentile(v, [75, 25])
    iqr = q75 - q25
    a = min(sd, iqr / 1.349) if iqr > 0 else sd
    if a <= 0:
        a = 1.0
    return 0.7 * (0.9 * a * n ** (-0.2))   # 0.7 = undersmoothing factor


def _local_linear_Zweighted(v, Z, a, grid, bw, wts=None, nbins=400):
    """Binned local-linear fit of the MULTIPLICATIVE model a_i ~ G(v_i)*Z_i,
    evaluated on grid. At each grid point g it solves
        min_{al,be} sum_i K_h(v_i-g) [ a_i - (al + be (v_i-g)) Z_i ]^2  (x wts_i)
    and returns G(g)=al-hat. Binned over v (Fan-Marron 1994) for speed: per-bin
    moment sums are formed once, then convolved with the Gaussian kernel at each
    grid point. Local LINEAR (not NW) for the standard O(h^2) boundary bias."""
    v = np.asarray(v, float); Z = np.asarray(Z, float); a = np.asarray(a, float)
    Z2 = Z * Z
    Za = Z * a
    if wts is not None:
        wts = np.asarray(wts, float)
        Z2 = Z2 * wts
        Za = Za * wts
    lo, hi = grid[0], grid[-1]
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, nbins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    bidx = np.clip(np.searchsorted(edges, v) - 1, 0, nbins - 1)
    sZ2 = np.bincount(bidx, weights=Z2, minlength=nbins)
    sZ2v = np.bincount(bidx, weights=Z2 * v, minlength=nbins)
    sZ2v2 = np.bincount(bidx, weights=Z2 * v * v, minlength=nbins)
    sZa = np.bincount(bidx, weights=Za, minlength=nbins)
    sZav = np.bincount(bidx, weights=Za * v, minlength=nbins)
    # Vectorised over grid: K is (n_grid x n_bins), all moments via one matmul each.
    g = np.asarray(grid, float)
    Kmat = np.exp(-0.5 * ((g[:, None] - ctr[None, :]) / bw) ** 2)
    S00 = Kmat @ sZ2
    S0v = Kmat @ sZ2v
    S01 = S0v - g * S00
    S11 = (Kmat @ sZ2v2) - 2.0 * g * S0v + g * g * S00
    b0 = Kmat @ sZa
    b1 = (Kmat @ sZav) - g * b0
    det = S00 * S11 - S01 * S01
    with np.errstate(divide="ignore", invalid="ignore"):
        G_ll = (S11 * b0 - S01 * b1) / det           # local-linear alpha-hat
        G_nw = np.where(S00 > 1e-12, b0 / S00, 0.0)   # NW fallback at thin grid points
    bad = (S00 <= 1e-12) | (np.abs(det) < 1e-12 * (np.abs(S00 * S11) + 1e-12)) | ~np.isfinite(G_ll)
    return np.where(bad, G_nw, G_ll)


def _fit_link_kernel(v, Z, cf_dm, y_dm, einv, counts, bw, grid, wts=None,
                     G_init=None, tol=1e-4, max_iter=40):
    """Profiled kernel local-linear link for the FE + multiplicative-Z model, by
    BACKFITTING (Gauss-Seidel): alternate a weighted local-linear G-step (the
    entity-mean coupling held at the current G, then added back into the pseudo-
    response) with a 1-D OLS gamma-step for the control function. G is left
    UNCONSTRAINED here; monotonicity is imposed ex post by rearrangement (CFG
    2009) in the caller. Returns (Ggrid, gamma, resid, ssr, G_obs).

    The FE and G(v)Z are strongly coupled, so cold backfitting converges slowly
    (a fixed few sweeps badly under-shoots the link level). We therefore iterate
    to a convergence tolerance (max|dG| on the grid) up to max_iter, and accept a
    warm-start grid G_init (the caller passes a quick exact sieve fit) so the loop
    starts near the converged level and needs only a handful of sweeps."""
    n = len(v)
    if G_init is not None:
        Ggrid = np.asarray(G_init, float).copy()
        G_obs = np.interp(v, grid, Ggrid)
    else:
        G_obs = np.zeros(n)
        Ggrid = np.zeros(len(grid))
    gamma = 0.0
    wcf = (wts if wts is not None else 1.0)
    for _ in range(max_iter):
        Gprev = Ggrid
        t = G_obs * Z
        m = (np.bincount(einv, weights=t, minlength=len(counts)) / counts)[einv]
        if cf_dm is not None:
            tZ_dm = t - m
            den = float(np.dot(cf_dm * wcf, cf_dm))
            num = float(np.dot(cf_dm * wcf, y_dm - tZ_dm))
            gamma = num / den if den > 0 else 0.0
            target = y_dm - gamma * cf_dm
        else:
            gamma = 0.0
            target = y_dm
        a = target + m                      # add back current entity-mean coupling
        Ggrid = _local_linear_Zweighted(v, Z, a, grid, bw, wts=wts)
        # phi is structurally a CDF: impose the [0,1] bound IN-LOOP. Without it the
        # robust (Cauchy) IRLS diverges -- it keeps pushing G past 1 because the
        # unconstrained local-linear link has nothing keeping it a valid CDF (the
        # sieve gets this for free from its monotone-bounded basis). Monotonicity
        # is still imposed ex post by rearrangement in the caller.
        Ggrid = np.clip(Ggrid, 0.0, 1.0)
        G_obs = np.interp(v, grid, Ggrid)
        if np.max(np.abs(Ggrid - Gprev)) < tol:
            break
    t = G_obs * Z
    m = (np.bincount(einv, weights=t, minlength=len(counts)) / counts)[einv]
    resid = y_dm - (t - m) - (gamma * cf_dm if cf_dm is not None else 0.0)
    ww = wts if wts is not None else 1.0
    ssr = float(np.sum(ww * resid * resid))
    return Ggrid, gamma, resid, ssr, G_obs


def fit_joint_single_index(df, state_cols, has_cf, link="sieve", loss="ls",
                           n_interior=5, degree=3, n_starts=4, init_theta=None,
                           theta_candidates=None, refine_maxiter=0,
                           boot_B=199, boot_scheme="rademacher", seed=0, label="",
                           phi_band=False, fe_time_col=None, theta_fixed=None):
    """Joint single-index sleepiness by Ichimura (1993) SLS: estimate the index
    direction theta (||theta||=1) and the link G TOGETHER.
      link='sieve'  (Est7): monotone cubic I-spline ramps, nonneg coefs (shape
                    restriction built in).
      link='kernel' (Est8): local-linear link profiled by BACKFITTING for the
                    entity FE + multiplicative-Z structure (binned, Fan-Marron
                    1994), monotonised EX POST by rearrangement (CFG 2009).
    loss in {'ls','robust'} (robust=Cauchy IRLS). Inference: score/multiplier wild
    cluster bootstrap on theta -> AMEs. Returns a NonLinearResults with theta in
    params_native (index over non-constant state vars), average-derivative AMEs in
    params, a stored monotone link grid (si_vgrid/si_ggrid), bootstrap SEs/pvalues,
    and link in {'sieve','kernel'}."""
    from scipy.interpolate import BSpline
    if fe_time_col is not None and link == "kernel":
        raise NotImplementedError("two-way FE (fe_time_col) is wired for the sieve link only")
    rng = np.random.default_rng(seed)
    CF_cols = ["v_hat_x_lagged_dep"] if has_cf else []
    cols = state_cols + ["deposit_balance", "nr_lagged_dep", "entity_id"]
    df_ss = df.dropna(subset=cols + CF_cols).copy()
    if len(df_ss) == 0:
        return None

    idx_cols = [c for c in state_cols if c != "constant"]   # index excludes the constant (it is in G)
    S = df_ss[idx_cols].values.astype(float)
    # standardise index regressors for numerical conditioning of ||theta||=1
    S_mu = S.mean(0); S_sd = S.std(0); S_sd[S_sd <= 0] = 1.0
    Sn = (S - S_mu) / S_sd
    Z = df_ss["nr_lagged_dep"].values.astype(float)
    y = df_ss["deposit_balance"].values.astype(float)
    cf = df_ss["v_hat_x_lagged_dep"].values.astype(float) if has_cf else None
    cl = df_ss["CodConglomeradoPrudencial"].astype(str).values
    _, einv = np.unique(df_ss["entity_id"].values, return_inverse=True)
    counts = np.bincount(einv).astype(float)
    cl_u, cl_inv = np.unique(cl, return_inverse=True)
    n_cl = len(cl_u)
    # Optional second additive FE (e.g. time): two-way concentration via alternating
    # projections. fe_time_col=None reproduces the entity-only within estimator exactly.
    if fe_time_col is not None:
        _, tinv = np.unique(df_ss[fe_time_col].values, return_inverse=True)
        tcounts = np.bincount(tinv).astype(float)
        demean = lambda M: _twoway_demean(M, einv, counts, tinv, tcounts)
    else:
        demean = lambda M: _fast_demean(M, einv, counts)
    y_dm = demean(y)
    cf_dm = demean(cf) if has_cf else None
    d = Sn.shape[1]

    def _cauchy_obj(resid):
        s = 1.4826 * np.median(np.abs(resid)) + 1e-12
        return float(np.sum(np.log1p((resid / (2.385 * s)) ** 2)))

    def _fit_link(th, want_grid=False):
        """Profile the link at direction th. Returns the objective; if want_grid,
        also returns (v, vgrid_std, ggrid_mono01, gpgrid_nonneg, resid). Both link
        families share this signature so the multistart and the tail are common."""
        v = Sn @ th
        if link == "kernel":
            lo, hi = np.quantile(v, [0.005, 0.995])
            vgrid = np.linspace(lo, hi, 200)
            bw = _kernel_bw(v)
            # Warm-start the kernel backfit from a quick EXACT sieve fit at this theta:
            # the sieve solves FE+link in one BVLS, giving a near-converged G so the
            # backfit needs only a few sweeps (cold backfitting converges slowly here).
            qs = np.linspace(0, 1, n_interior + 2)[1:-1]
            interior = np.clip(np.quantile(v, qs), lo + 1e-9, hi - 1e-9)
            t_ws = np.concatenate(([lo] * (degree + 1), np.sort(interior), [hi] * (degree + 1)))
            R_ws = _ramp_design(v, t_ws, degree)
            _b, _g, _r, _s, c_ws = _fit_link_sieve(R_ws, Z, cf_dm, y_dm, einv, counts, None)
            G_init = np.clip(BSpline(t_ws, c_ws, degree, extrapolate=True)(vgrid), 0.0, 1.0)
            Gg, gamma, resid, ssr, _ = _fit_link_kernel(v, Z, cf_dm, y_dm, einv, counts, bw,
                                                        vgrid, G_init=G_init)
            if loss == "robust":
                for _ in range(2):
                    wts = _cauchy_weights(resid)
                    Gg, gamma, resid, ssr, _ = _fit_link_kernel(v, Z, cf_dm, y_dm, einv,
                                                                counts, bw, vgrid, wts=wts, G_init=Gg)
                obj = _cauchy_obj(resid)
            else:
                obj = ssr
            if not want_grid:
                return obj
            ggrid_mono = np.clip(np.maximum.accumulate(Gg), 0.0, 1.0)   # rearrange ex post (CFG 2009)
            gp = np.clip(np.gradient(ggrid_mono, vgrid), 0.0, None)
            return obj, v, vgrid, ggrid_mono, gp, resid
        # ---- sieve (Est7): monotone I-spline ramps, nonneg coefs ----
        lo, hi = np.quantile(v, [0.001, 0.999])
        qs = np.linspace(0, 1, n_interior + 2)[1:-1]
        interior = np.clip(np.quantile(v, qs), lo + 1e-9, hi - 1e-9)
        t = np.concatenate(([lo] * (degree + 1), np.sort(interior), [hi] * (degree + 1)))
        R = _ramp_design(v, t, degree)
        if loss == "robust":
            # ONE unweighted fit, then TWO weighted IRLS passes -- matching the Julia
            # engine exactly (sleep_joint_sieve.jl::profile_obj) and the kernel branch
            # above, both of which do 1+2.
            #
            # This branch used to run `for _ in range(2)` with w=None on the first pass,
            # i.e. only TWO fits, so the objective was read off residuals that were one
            # IRLS pass less converged. That made Python report a systematically HIGHER
            # objective than Julia FOR THE SAME theta (E8: 166,865 vs 158,680), and the
            # full-sample candidate scan then rejected Julia's own -- better -- answer in
            # favour of a worse one. The two objectives must be the same computation for
            # the scan to mean anything.
            beta, gamma, resid, ssr, c = _fit_link_sieve(R, Z, cf_dm, y_dm, einv, counts,
                                                         None, dm=demean)
            for _ in range(2):
                w = _cauchy_weights(resid)
                beta, gamma, resid, ssr, c = _fit_link_sieve(R, Z, cf_dm, y_dm, einv, counts,
                                                             w, dm=demean)
            obj = _cauchy_obj(resid)
        else:
            beta, gamma, resid, ssr, c = _fit_link_sieve(R, Z, cf_dm, y_dm, einv, counts, None, dm=demean)
            obj = ssr
        if not want_grid:
            return obj
        spl = BSpline(t, c, degree, extrapolate=True)
        vgrid = np.linspace(t[0], t[-1], 200)
        ggrid_mono = np.clip(np.maximum.accumulate(spl(vgrid)), 0.0, 1.0)
        gp = np.clip(spl.derivative()(vgrid), 0.0, None)
        return obj, v, vgrid, ggrid_mono, gp, resid

    # ---- multistart over the unit sphere ----
    from scipy.optimize import minimize
    # Start order: (1) logit warm start, (2) the deterministic equal-weight
    # ("ones") direction, (3+) random unit vectors. The ones-vector is a cheap,
    # reliable second start that reliably settles the sieve optimum; random starts
    # only kick in for n_starts>=3. n_starts is the true total of starts, so the
    # kernel can use n_starts=1 (warm only) to avoid a costly random-start wander.
    starts = []
    if init_theta is not None:
        iv = np.asarray(init_theta, float)
        if len(iv) == d:
            starts.append(iv / (np.linalg.norm(iv) + 1e-12))
    ones = np.ones(d) / np.sqrt(d)
    if not starts:
        starts.append(ones)                 # no warm start -> ones is start 1
    elif n_starts >= 2:
        starts.append(ones)                 # warm + deterministic ones as start 2
    while len(starts) < n_starts:
        r = rng.standard_normal(d)
        starts.append(r / np.linalg.norm(r))

    def _obj(theta):
        return _fit_link(theta / (np.linalg.norm(theta) + 1e-12), want_grid=False)

    # Kernel evals are dearer and the in-loop CDF clip roughens the objective, so
    # cap the kernel search HARD: the logit warm-start is already a strong index
    # direction, so a few dozen refinement steps suffice (vs the sieve's full search).
    _maxit = (20 * d if link == "kernel" else 300 * d)
    if theta_fixed is not None:
        # The index direction was found externally (e.g. the Julia engine on a
        # SUBSAMPLE). It is a starting point, not an answer: on 2026-07-29 the
        # 20%-subsample search drove E7 to a corner (0.988 of a unit-norm theta on
        # risk_free_qoq_lag, a national series with ~36 distinct values), which
        # saturated the monotone link into a 1.5pp band of phi and collapsed EVERY
        # AME by ~700x at essentially unchanged R2 (0.95230 vs 0.95250). Freezing it
        # meant the full sample never got to disagree.
        #
        # So: score every candidate direction on the FULL-SAMPLE objective, keep the
        # best, then optionally polish. All candidates are normalised into the same
        # standardised Sn frame (same S_mu/S_sd as here).
        cands = [("external", theta_fixed)]
        if theta_candidates:
            cands += [(str(nm), th) for nm, th in theta_candidates]
        if init_theta is not None:
            cands.append(("warm", init_theta))
        cands.append(("ones", ones))

        # POLISH EVERY CANDIDATE, THEN COMPARE. Scoring candidates unpolished and then
        # refining only the winner is not a like-for-like comparison and actively picks the
        # wrong direction: `external` is Julia's OPTIMISED theta (8 starts) but expressed in
        # Julia's own metric -- which differs from this one because the Julia engine bins the
        # ramp design (nbins=1000) and solves the bounded LS by a different algorithm. Scored
        # cold against a raw `ones` vector it loses, gets no refinement, and the estimate
        # degrades: on E7 that produced a 5x flatter link (G span 0.051 vs 0.266), a lower
        # R2, and HALF the AMEs versus simply trusting Julia's direction. Refining each
        # candidate first lets a direction that starts worse but converges better win.
        scored = []
        for nm, th in cands:
            th = np.asarray(th, float)
            if th.shape != (d,) or not np.all(np.isfinite(th)):
                continue
            th = th / (np.linalg.norm(th) + 1e-12)
            o0 = float(_fit_link(th, want_grid=False))
            o1, th1 = o0, th
            if refine_maxiter and refine_maxiter > 0:
                rr = minimize(_obj, th, method="Nelder-Mead",
                              options={"maxiter": int(refine_maxiter),
                                       "xatol": 1e-3, "fatol": 1e-5})
                if np.all(np.isfinite(rr.x)) and float(rr.fun) < o1:
                    o1 = float(rr.fun)
                    th1 = rr.x / (np.linalg.norm(rr.x) + 1e-12)
            scored.append((o1, o0, nm, th1))
        scored.sort(key=lambda t: t[0])
        obj_best, _, nm_best, theta_hat = scored[0]
        if len(scored) > 1:
            spread = scored[-1][0] - scored[0][0]
            print(f"  [joint-{link}] full-sample candidate scan (each polished "
                  f"{int(refine_maxiter)} iters): " +
                  ", ".join(f"{nm}={o0:.6g}->{o1:.6g}" for o1, o0, nm, _ in scored) +
                  f" | winner={nm_best} | spread={spread:.3g}")
            if abs(spread) <= 1e-6 * max(1.0, abs(obj_best)):
                print(f"  [joint-{link}] WARNING: candidates are within 1e-6 of each other -- "
                      "the index DIRECTION is weakly identified; the optimizer, not the data, "
                      "is choosing it. Treat theta as set-identified.")
    else:
        best = None
        for s0 in starts:
            res = minimize(_obj, s0, method="Nelder-Mead",
                           options={"maxiter": _maxit, "xatol": 1e-3, "fatol": 1e-5})
            if best is None or res.fun < best[0]:
                best = (res.fun, res.x)
        theta_hat = best[1] / (np.linalg.norm(best[1]) + 1e-12)
    obj, v, vgrid, ggrid, gpgrid, resid = _fit_link(theta_hat, want_grid=True)

    # 0/1 DUMMIES use the DISCRETE-DIFFERENCE AME E[G(idx|x=1) - G(idx|x=0)] on the
    # fixed monotone link (bounded to the link range), NOT the continuous average
    # derivative theta_k * mean_slope. The unit-norm direction keeps theta_k moderate
    # here (so the bug is invisible -- pix AME ~ 2e-3), but a binary regressor's
    # estimand is still the discrete difference, so we treat it correctly and
    # consistently with the single-index / logit paths. Flip terms are fixed in i.
    # Levels from _dummy_spec (registry-driven) rather than a literal {0,1} test, which
    # a CENTRED dummy ({-p_bar, 1-p_bar}) fails silently. Note the flip terms are
    # INVARIANT to centering: with S_new = a(S - m), S_sd scales by a too, so
    # (hi - S_new)/S_sd_new = (1 - S_raw)/S_sd_old exactly.
    _dummy, _dlo, _dhi = _dummy_spec(S, d, idx_cols)
    _flip1 = [((_dhi[k] - S[:, k]) / S_sd[k]) if _dummy[k] else None for k in range(d)]
    _flip0 = [((_dlo[k] - S[:, k]) / S_sd[k]) if _dummy[k] else None for k in range(d)]

    def _ames(theta):
        th = theta / (np.linalg.norm(theta) + 1e-12)
        vv = Sn @ th
        gp = np.interp(vv, vgrid, gpgrid)          # link held fixed; only the index moves
        mean_slope = float(np.mean(gp))
        out = {}
        for k in range(d):
            if _dummy[k]:                          # discrete difference (bounded by the link)
                # both flips now carry their own sign: _flip0 = (lo - S)/S_sd, so it is
                # ADDED here. (It used to be S/S_sd and subtracted; same thing at lo=0.)
                vv1 = vv + th[k] * _flip1[k]
                vv0 = vv + th[k] * _flip0[k]
                out[idx_cols[k]] = float(np.mean(np.interp(vv1, vgrid, ggrid)
                                                 - np.interp(vv0, vgrid, ggrid)))
            else:                                  # continuous average derivative
                out[idx_cols[k]] = th[k] / S_sd[k] * mean_slope
        return out

    ame_hat = _ames(theta_hat)

    # ---- score/multiplier wild cluster bootstrap on theta -> AMEs ----
    gp_obs = np.interp(v, vgrid, gpgrid)
    gS = (gp_obs * Z)[:, None] * Sn               # N x d  (d/dtheta of G(v)Z)
    gS_dm = demean(gS)
    Minv = np.linalg.pinv(gS_dm.T @ gS_dm)
    IF = (resid[:, None] * gS_dm) @ Minv.T         # N x d influence functions for theta
    IF = IF - np.outer(IF @ theta_hat, theta_hat)  # project to the sphere tangent
    IF_cl = np.zeros((n_cl, d))
    for k in range(d):
        IF_cl[:, k] = np.bincount(cl_inv, weights=IF[:, k], minlength=n_cl)

    bse, pvals = cluster_wild_bootstrap(theta_hat, IF_cl, _ames, ame_hat,
                                        B=boot_B, scheme=boot_scheme, rng=rng)

    ps = pd.Series({f"interaction_{k}": ame_hat[k] for k in idx_cols})
    bs = pd.Series({f"interaction_{k}": bse[k] for k in idx_cols})
    pv = pd.Series({f"interaction_{k}": pvals[k] for k in idx_cols})
    G_star, G_nominal = _G_star(df_ss["CodConglomeradoPrudencial"].astype(str))
    tss = float(np.sum((y_dm - y_dm.mean()) ** 2))
    rsq = 1 - float(np.sum(resid ** 2)) / tss if tss > 0 else np.nan
    theta_native = pd.Series({f"interaction_{idx_cols[k]}": float(theta_hat[k] / S_sd[k]) for k in range(d)})
    print(f"  [JointSI-{link}-{loss}] starts={len(starts)} | obj={obj:.5g} | "
          f"||theta||=1 | boot B={boot_B} ({boot_scheme})")
    res = NonLinearResults(ps, bs, ps / bs.replace(0, np.nan), pv, G_star,
                           params_native=theta_native, nobs=len(df_ss), rsquared=rsq,
                           G_nominal=G_nominal, cov_ame=None, link=link)
    # phi_from_native evaluates the NATIVE (raw) index = sum native_k*S_raw_k, which
    # equals the standardised index v plus a constant offset; store the grid in that
    # native-index frame so np.interp aligns.
    offset = float(np.sum(theta_hat * S_mu / S_sd))
    res.si_vgrid = vgrid + offset
    res.si_ggrid = ggrid
    res.si_degree = degree
    res.boot_B = boot_B; res.boot_scheme = boot_scheme

    # National phi_t confidence band: perturb theta by the cluster-summed IFs,
    # hold the link fixed (matches the AME bootstrap), aggregate to national phi_t.
    if phi_band:
        gs = _phi_t_group_struct(df_ss)
        phi_pt = np.clip(np.interp(v, vgrid, ggrid), 0.0, 1.0)

        def _draw(rng_):
            w = _wild_weights(n_cl, boot_scheme, rng_)
            th_b = theta_hat + w @ IF_cl
            th_b = th_b / (np.linalg.norm(th_b) + 1e-12)
            return np.clip(np.interp(Sn @ th_b, vgrid, ggrid), 0.0, 1.0)

        res.phi_t_boot = _phi_t_band(gs, phi_pt, _draw, min(boot_B, 400),
                                     boot_scheme, np.random.default_rng(seed + 12345))
    return res


def _wild_weights(n, scheme, rng):
    if scheme == "webb":   # 6-point Webb weights (good for severe cluster imbalance)
        vals = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
        return rng.choice(vals, size=n)
    return rng.choice(np.array([-1.0, 1.0]), size=n)   # Rademacher


def cluster_wild_bootstrap(theta_hat, IF_cl, ame_fn, ame_hat, B=199,
                           scheme="rademacher", rng=None):
    """Score/multiplier wild cluster bootstrap (Kline & Santos 2012): perturb the
    cluster-summed influence functions by wild weights, recompute the (linearised)
    AME via ame_fn, and read SEs/p-values off the bootstrap distribution.
    Returns (bse dict, pvals dict)."""
    rng = rng or np.random.default_rng(0)
    n_cl = IF_cl.shape[0]
    keys = list(ame_hat.keys())
    draws = {k: np.empty(B) for k in keys}
    for b in range(B):
        wv = _wild_weights(n_cl, scheme, rng)
        theta_b = theta_hat + wv @ IF_cl
        a_b = ame_fn(theta_b)
        for k in keys:
            draws[k][b] = a_b[k]
    bse, pvals = {}, {}
    for k in keys:
        sd = float(np.std(draws[k], ddof=1))
        bse[k] = sd
        # symmetric bootstrap p for H0: AME=0, via the studentised pivot
        z = abs(ame_hat[k]) / sd if sd > 0 else np.inf
        pvals[k] = float(2 * stats.norm.sf(z)) if np.isfinite(z) else 0.0
    return bse, pvals


def linear_wild_cluster_bootstrap(res, B=999, scheme="webb", seed=0):
    """Score/multiplier wild cluster bootstrap SEs/p-values for a fitted statsmodels
    cluster-OLS result, so the LINEAR sleepiness estimators (Est1/Est2) share ONE
    inference method with the single-index/joint columns (Cameron-Gelbach-Miller 2008;
    MacKinnon-Webb 2017; Kline-Santos 2012). The cluster influence functions are read
    straight off the fit -- IF_cl[g] = (X'X)^{-1} sum_{i in g} X_i u_hat_i -- and
    perturbed by wild weights via the same cluster_wild_bootstrap used for the
    nonlinear AMEs (identity map: the parameters ARE the coefficients). No refit.
    Returns (bse, tvalues, pvalues) as pandas Series indexed like res.params."""
    idx = res.params.index
    names = list(idx)
    beta = np.asarray(res.params, float)
    X = np.asarray(res.model.exog, float)
    u = np.asarray(res.resid, float)
    cl = pd.Series(np.asarray(res.cov_kwds["groups"])).astype(str).values
    cl_u, cl_inv = np.unique(cl, return_inverse=True)
    n_cl = len(cl_u)
    K = X.shape[1]
    bread = np.linalg.pinv(X.T @ X)
    score = X * u[:, None]                                   # N x K per-obs scores
    s_cl = np.zeros((n_cl, K))
    for k in range(K):
        s_cl[:, k] = np.bincount(cl_inv, weights=score[:, k], minlength=n_cl)
    IF_cl = s_cl @ bread.T                                   # n_cl x K cluster IFs on beta
    ame_hat = {nm: float(b) for nm, b in zip(names, beta)}
    ame_fn = lambda th: {nm: float(th[i]) for i, nm in enumerate(names)}
    bse, pvals = cluster_wild_bootstrap(beta, IF_cl, ame_fn, ame_hat,
                                        B=B, scheme=scheme, rng=np.random.default_rng(seed))
    bse_s = pd.Series({nm: bse[nm] for nm in names}).reindex(idx)
    pv_s = pd.Series({nm: pvals[nm] for nm in names}).reindex(idx)
    tv_s = pd.Series({nm: (ame_hat[nm] / bse[nm] if bse[nm] else np.nan)
                      for nm in names}).reindex(idx)
    return bse_s, tv_s, pv_s


# ==============================================================================
# Shared wild-cluster-bootstrap inference for the M-estimators Est3/4/5 (link
# NLLS) and Est6 (profiled sieve OLS). The earlier homoskedastic delta-method
# SEs (Est3/4/5) and conditional clustered-OLS SE (Est6) are replaced by a
# score/multiplier wild cluster bootstrap (Cameron-Gelbach-Miller 2008;
# MacKinnon-Webb 2017): perturb the cluster-summed influence functions by wild
# weights, recompute the (linearised) AMEs, and read SEs/p-values off the draws.
# ==============================================================================
def boot_cfg():
    """Bootstrap settings, overridable by env for quick smoke runs.
    SLEEP_BOOT_B (default 999); SLEEP_BOOT_SCHEME in {webb, rademacher} (default webb)."""
    B = int(os.environ.get("SLEEP_BOOT_B", "999"))
    scheme = os.environ.get("SLEEP_BOOT_SCHEME", "webb")
    return B, scheme


def _cluster_if(score, bread, cl_inv, n_cl):
    """Per-cluster influence functions IF_g = sum_{i in g} (bread @ score_i)."""
    IF = score @ bread.T                       # N x p
    p = IF.shape[1]
    IF_cl = np.zeros((n_cl, p))
    for k in range(p):
        IF_cl[:, k] = np.bincount(cl_inv, weights=IF[:, k], minlength=n_cl)
    return IF_cl


def nlls_link_wild_bootstrap(res_lsq, X, link, K, G, idx_names, cl_inv, n_cl,
                             B=None, scheme=None, seed=0, ame_fn=None,
                             state_names=None):
    """Score/multiplier wild cluster bootstrap of the AMEs for the link-NLLS
    M-estimators (Est3 logit, Est4 uniform, Est5 probit). The estimator solves
    min_psi sum_i r_i(psi)^2, r = y_dm - f(psi); the influence function is
    IF_i = (J'J)^{-1} (df/dpsi)_i r_i, with res_lsq.jac = dr/dpsi = -(df/dpsi).
    Returns (ame dict, bse dict, pval dict). A custom ame_fn(psi)->array (length
    K+G, same order as idx_names) may be passed (Est3 uses its analytic AME);
    otherwise the generic finite-difference _generic_ame is used."""
    if B is None or scheme is None:
        _B, _s = boot_cfg(); B = B if B is not None else _B; scheme = scheme or _s
    psi = res_lsq.x
    jac = res_lsq.jac                          # dr/dpsi  (N x p)
    resid = res_lsq.fun                        # N
    bread = np.linalg.pinv(jac.T @ jac)
    score = (-jac) * resid[:, None]            # (df/dpsi)*r  (sign irrelevant for variance)
    IF_cl = _cluster_if(score, bread, cl_inv, n_cl)
    if ame_fn is not None:
        _ame_arr = ame_fn
    else:
        # precompute the (is_dummy, lo, hi) spec once for the B-loop; `names` lets
        # the registry answer for centred dummies instead of sniffing {0,1} values
        _spec = _dummy_spec(X, K, state_names)
        _ame_arr = lambda p_: _generic_ame(p_, X, link, K, G, dummy_spec=_spec)

    def _ame_dict(p_):
        a = _ame_arr(p_)
        return {idx_names[j]: float(a[j]) for j in range(len(idx_names))}

    ame_hat = _ame_dict(psi)
    rng = np.random.default_rng(seed)
    bse, pvals = cluster_wild_bootstrap(psi, IF_cl, _ame_dict, ame_hat,
                                        B=B, scheme=scheme, rng=rng)
    return ame_hat, bse, pvals


def ols_sieve_wild_bootstrap(Xdm, resid, cl_inv, n_cl, b_full, ame_fn, ame_hat,
                             B=None, scheme=None, seed=0):
    """Score/multiplier wild cluster bootstrap for the Est6 profiled sieve OLS
    (link coefficients conditional on the logit index direction). IF_i for OLS
    is (X'X)^{-1} x_i u_i; perturb the cluster sums, recompute the AME via
    ame_fn(b)->dict, return (bse dict, pvals dict)."""
    if B is None or scheme is None:
        _B, _s = boot_cfg(); B = B if B is not None else _B; scheme = scheme or _s
    Xdm = np.asarray(Xdm, float)
    bread = np.linalg.pinv(Xdm.T @ Xdm)
    score = Xdm * np.asarray(resid, float)[:, None]
    IF_cl = _cluster_if(score, bread, cl_inv, n_cl)
    rng = np.random.default_rng(seed)
    return cluster_wild_bootstrap(b_full, IF_cl, ame_fn, ame_hat,
                                  B=B, scheme=scheme, rng=rng)


# ==============================================================================
# National phi_t confidence bands (score/multiplier wild cluster bootstrap).
# phi_t = pop-weighted mean over markets, per quarter, of phi_mt; the band
# propagates parameter uncertainty (theta for the joint single index; the link
# coefficients for Est6) through to the national sleepiness path.
# ==============================================================================
def _phi_t_group_struct(df_ss):
    """Precompute the (time, market) group structure for fast pop-weighted
    national phi_t aggregation: national phi_t = sum_market pop_market *
    mean_market(phi_mt) / sum_market pop_market, per time."""
    t = df_ss["time_id"].astype(str).values
    if "CODMUN_IBGE" in df_ss.columns:
        m = df_ss["CODMUN_IBGE"].astype(str).values
    else:
        m = np.array(["0"] * len(df_ss))
    pop = (df_ss["pop_total"].fillna(0).values.astype(float)
           if "pop_total" in df_ss.columns else np.ones(len(df_ss)))
    key = np.char.add(np.char.add(t, "|"), m)
    gcode, _ = pd.factorize(key)
    ng = int(gcode.max()) + 1
    gcount = np.bincount(gcode, minlength=ng).astype(float)
    gpop = np.bincount(gcode, weights=pop, minlength=ng) / np.maximum(gcount, 1.0)
    tcode, tuniq = pd.factorize(df_ss["time_id"].astype(str).values)
    tcode_g = np.zeros(ng, int)
    tcode_g[gcode] = tcode          # time is constant within a (time,market) group
    return dict(gcode=gcode, gcount=gcount, gpop=gpop, tcode_g=tcode_g,
                nt=len(tuniq), tuniq=np.asarray(tuniq))


def _agg_phi_t(phi, gs):
    gmean = np.bincount(gs["gcode"], weights=phi, minlength=len(gs["gcount"])) / gs["gcount"]
    num = np.bincount(gs["tcode_g"], weights=gmean * gs["gpop"], minlength=gs["nt"])
    den = np.bincount(gs["tcode_g"], weights=gs["gpop"], minlength=gs["nt"])
    return num / np.maximum(den, 1e-12)


def _phi_t_band(gs, phi_point, draw_phi_fn, B, scheme, rng, alpha=0.05):
    """National phi_t point path + (1-alpha) percentile band from B wild draws.
    draw_phi_fn(rng) returns a per-observation phi vector for one bootstrap draw.
    Returns a DataFrame [time_id, phi_t, lo, hi] sorted by quarter."""
    pt = _agg_phi_t(phi_point, gs)
    draws = np.empty((B, gs["nt"]))
    for b in range(B):
        draws[b] = _agg_phi_t(draw_phi_fn(rng), gs)
    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    out = pd.DataFrame({"time_id": gs["tuniq"], "phi_t": pt, "lo": lo, "hi": hi})
    try:
        out["_d"] = pd.PeriodIndex(out["time_id"].str.replace("Q", "Q"), freq="Q").to_timestamp()
        out = out.sort_values("_d").reset_index(drop=True)
    except Exception:
        out = out.sort_values("time_id").reset_index(drop=True)
    return out
