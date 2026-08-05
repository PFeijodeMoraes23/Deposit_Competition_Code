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

REFERENCE DISTRIBUTION (env SLEEP_WCB_MODE, default unchanged): the historical
p-values use the bootstrap only for the SE and a NORMAL reference, which at
nu = G*-1 = 5.2 is materially undersized. SLEEP_WCB_MODE=studentised (bootstrap-t,
unrestricted) and =wcr (bootstrap-t with H0 imposed on the DGP residuals) are OPT-IN
corrections; see the comment block above cluster_wild_bootstrap for why the default
is NOT flipped and what must be re-run to promote it. Quantified side-by-side in
diag_wcb_reference.py.
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

    # Quarter-clustered WCB for the national rows (pix_exists, risk_free_qoq_lag). Same
    # bootstrap, same _ame_fn, resampling the QUARTER instead of the conglomerate -- which is
    # the only dimension along which a national regressor actually varies. Mirrors the block in
    # fit_joint_single_index; stored separately so `bs` is untouched and the exporters choose
    # per row. Driscoll-Kraay is not produced here: it is analytic and would need a
    # delta-method push through _ame_fn, which the bootstrap already does exactly.
    _nat_time = _nat_pv = None
    if fe_time_col is not None and fe_time_col in df_ss.columns:
        try:
            _per = df_ss[fe_time_col].astype(str).values
            _uniq, _t_inv = np.unique(_per, return_inverse=True)
            _bt, _pt = ols_sieve_wild_bootstrap(Xdm_arr, np.asarray(res.resid, float),
                                                _t_inv, len(_uniq), b_full, _ame_fn, ame)
            _nat_time = pd.Series(_bt); _nat_pv = pd.Series(_pt)
            from utils.se_national import is_national
            _shown = [k for k in ps.index if is_national(k)]
            if _shown:
                print(f"  [national-SE single-index] T={len(_uniq)} | " + ", ".join(
                    f"{str(k).replace('interaction_','')}: congl={float(bs[k]):.4g} / "
                    f"quarter={float(_nat_time[k]):.4g}" for k in _shown))
        except Exception as _e:
            print(f"  [national-SE single-index] skipped ({type(_e).__name__}: {_e})")

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
    # Quarter-clustered SEs for the national rows, computed above. Kept alongside `bse` so the
    # exporters can choose PER ROW without a second estimation pass.
    if _nat_time is not None:
        res_obj.bse_time = _nat_time
        res_obj.pvalues_time = _nat_pv
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
                           boot_B=199, boot_scheme="rademacher", seed=0, label="",
                           phi_band=False, fe_time_col=None, theta_fixed=None,
                           polish_evals=0):
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
    import time as _time
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
        # Trust the direction the Julia engine found; refit the link and run the tail
        # (AMEs, bootstrap, phi grid) on the FULL sample here. theta arrives in the
        # standardised Sn frame (same S_mu/S_sd as here).
        #
        # 2026-07-29: a Python-side "candidate scan" was added here and then REVERTED --
        # score several candidate directions on THIS objective, keep the best, polish it.
        # Two measured reasons it does not work:
        #
        #  1. IT PICKS THE WRONG DIRECTION. `external` is Julia's OPTIMISED theta, but
        #     optimised in JULIA's metric, which is not this one: the engine bins the ramp
        #     design (nbins=1000) and solves the bounded LS by a different algorithm
        #     (Gram cross-products + NNLS-normal vs scipy lsq_linear here). Scored on this
        #     objective against a raw equal-weight vector it LOSES. On E7 that produced a 5x
        #     flatter link (G span 0.051 vs 0.266), lower R2, and HALF the AMEs compared with
        #     simply trusting Julia's answer. Cross-implementation objective values are NOT
        #     comparable; only same-implementation ones are.
        #
        #  2. FIXING (1) BY POLISHING EVERY CANDIDATE IS UNAFFORDABLE. Nelder-Mead
        #     `maxiter=60` is ~180 FUNCTION EVALUATIONS (reflection/expansion/contraction,
        #     and a shrink evaluates d+1 points), and each evaluation is one _fit_link = 3
        #     full-sample sieve solves on 487k rows. That is ~540 solves per candidate,
        #     measured at 3-5 HOURS per spec -- i.e. 24-40h of scan alone per estimator on
        #     the 8-spec grid.
        #
        # What actually fixed the original degeneracy was SLEEP_SUBSAMPLE_FRAC 0.2 -> 1.0
        # (link span x18, mean slope x570, R2 slightly up) plus passing n_starts through to
        # julia_theta so the engine really does multistart (objective -5.5%). Neither needs a
        # Python-side search. Do not re-add one without first counting function evaluations.
        tf = np.asarray(theta_fixed, float)
        theta_hat = tf / (np.linalg.norm(tf) + 1e-12)
        # OPTIONAL POLISH (2026-08-02). The Julia engine minimises a DIFFERENT objective from
        # this one -- it bins the ramp design and solves the bounded LS by another algorithm --
        # so its theta is a good START but not the optimum of `_obj`. Measured on E7 spec 12
        # under LS: Julia 26,054 vs 25,528 for a 33h cold Python search; ~100 Nelder-Mead evals
        # from Julia's theta recover 80% of that gap. Polishing HERE reuses `_obj` directly, so
        # each evaluation is one link solve -- not a whole fit+bootstrap+grid pass.
        # polish_evals=0 (default) keeps the previous behaviour exactly.
        if polish_evals and int(polish_evals) > 0:
            _o0 = _obj(theta_hat)
            _t0 = _time.time()
            _r = minimize(_obj, theta_hat, method="Nelder-Mead",
                          options={"maxfev": int(polish_evals), "xatol": 1e-4,
                                   "fatol": 1e-3, "adaptive": True})
            if np.isfinite(_r.fun) and _r.fun < _o0:
                theta_hat = np.asarray(_r.x, float)
                theta_hat = theta_hat / (np.linalg.norm(theta_hat) + 1e-12)
            print(f"  [polish/{loss}] {int(_r.nfev)} evals in {(_time.time()-_t0)/60:.1f} min | "
                  f"obj {_o0:.0f} -> {min(_r.fun, _o0):.0f} "
                  f"({100*(_o0-min(_r.fun,_o0))/max(_o0,1e-12):.2f}% better)")
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

    # Quarter-clustered WCB for the national rows (pix_exists, risk_free_qoq_lag): same
    # bootstrap, same _ames map, influence functions aggregated by QUARTER rather than by
    # conglomerate. Conglomerate clustering cannot see their sampling variation because they
    # are constant across firms within a period. Stored separately; `bs` is untouched.
    _nat_time = _nat_pv = None
    if fe_time_col is not None and fe_time_col in df_ss.columns:
        try:
            from utils.se_national import _aggregate_if_by_period
            IF_t, _n_t = _aggregate_if_by_period(IF, df_ss[fe_time_col])
            _bt, _pt = cluster_wild_bootstrap(theta_hat, IF_t, _ames, ame_hat,
                                              B=boot_B, scheme=boot_scheme,
                                              rng=np.random.default_rng(seed))
            _nat_time = pd.Series({f"interaction_{k}": _bt[k] for k in idx_cols})
            _nat_pv = pd.Series({f"interaction_{k}": _pt[k] for k in idx_cols})
            from utils.se_national import is_national
            _shown = [k for k in idx_cols if is_national(k)]
            if _shown:
                print(f"  [national-SE joint-{link}] T={_n_t} | " + ", ".join(
                    f"{k}: congl={float(bs['interaction_'+k]):.4g} / "
                    f"quarter={float(_nat_time['interaction_'+k]):.4g}" for k in _shown))
        except Exception as _e:
            print(f"  [national-SE joint-{link}] skipped ({type(_e).__name__}: {_e})")
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
    # Quarter-clustered SEs for the national rows, computed above. Kept alongside `bse` so the
    # exporters can choose PER ROW (conglomerate for firm-level regressors, quarter for the
    # national ones) without a second estimation pass.
    if _nat_time is not None:
        res.bse_time = _nat_time
        res.pvalues_time = _nat_pv

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


# ==============================================================================
# WCB REFERENCE DISTRIBUTION  --  OPT-IN, DEFAULT DELIBERATELY UNCHANGED
# ==============================================================================
# WHAT WAS WRONG (and still is, by default). The historical `cluster_wild_bootstrap`
# below takes the standard DEVIATION of the bootstrap draws as a standard error and
# then reads the p-value off a NORMAL reference:
#
#       sd = np.std(draws[k], ddof=1);  z = |ame| / sd;  p = 2 * Phi(-z)
#
# So the bootstrap supplies the SE but NOT the reference distribution. That is a
# Wald test with a bootstrap SE, not a wild cluster bootstrap test: the asymptotic
# refinement that motivates the wild bootstrap at small G is never realised. With
# G* = 6.2 effective clusters (nu = G*-1 = 5.2) the normal reference is far too
# narrow -- the exact t(5.2) 97.5% quantile is 2.53 against 1.96, so a nominal-5%
# normal test has size ~10-11%. Second, the scheme is the UNRESTRICTED variant
# (WCU): `theta_b = theta_hat + wv @ IF_cl` perturbs the UNRESTRICTED residuals and
# centres the bootstrap DGP at theta_hat, so the null is never imposed.
# MacKinnon & Webb (2017, 2018) show WCR (impose H0 when generating the bootstrap
# DGP) dominates WCU at small/imbalanced G precisely because WCU over-rejects.
#
# NOTE on the algebra: for OLS the score/multiplier draw IS the WCU refit draw --
#   beta*_b = (X'X)^{-1} X'(X beta_hat + w_g u_hat) = beta_hat + (X'X)^{-1} sum_g w_g s_g
#             = theta_hat + wv @ IF_cl.
# So the numerator here was already the textbook WCU numerator; what is missing is
# (a) studentisation by a PER-DRAW SE and (b) the null restriction.
#
# WHY THE DEFAULT IS **NOT** FLIPPED. utils/sleep_links.py is production code for
# E1-E8, the BBL policy function, utils/se_national.py and the whole phi-separation
# battery. Every p-value archived in Drafts/Deposit Competition (V_Main.tex tables,
# identification_notes.md, the DIAG_PHI_SEPARATION csvs, sleep_first_stage_pooled*.tex)
# was produced under the normal reference. Silently changing the default would make
# the code disagree with every archived number with no audit trail, and the tables
# and the drafts would drift apart mid-revision. So the corrected schemes are OPT-IN
# through SLEEP_WCB_MODE and the default is bit-identical to the historical path
# (same RNG stream, same draws, same bse, same p-values -- verified by the Tier-0
# check in diag_wcb_reference.py --check-default).
#
# WHAT IT COSTS, MEASURED (diag_wcb_reference.py, run 2026-08-03; numbers in
# <PROCESSED>/ESTIMATION_OUTPUT/DIAG_PHI_SEPARATION/d_wcb_reference_*.csv):
#   * Size, synthetic panel at THIS repo's cluster geometry (G=456 Zipf sizes calibrated
#     to CV=8.54 => G*=6.17, top-5 clusters = 66% of rows; 400 reps, B=299, beta_1 = 0
#     imposed). Rejection rate of a NOMINAL 5% test:
#         normal reference 0.270 | WCU-t 0.138 | WCR-t 0.083 | CRVE+t(G*) 0.145
#     The pure reference-distribution arithmetic (a t(5.2) pivot judged against 1.96)
#     accounts for 0.105 of that; the rest is the CRVE's own small-G noise. A balanced
#     G=20 control run gives normal 0.140, WCU-t 0.060, WCR-t 0.068 -- i.e. the machinery
#     is calibrated where it should be, and WCR is the best-behaved variant at OUR
#     geometry. If the default is ever flipped, flip it to 'wcr', NOT to 'studentised'.
#   * The phi-separation battery (11 coefficients, D5/D3/D2b/D6b, refit once each):
#     every null stays null; the ONE legacy 5% rejection, D5's CDB carry excess
#     Zdiff_k4 = +0.1849 (p_normal = 0.0131), goes to p = 0.152 under WCU-t but
#     p = 0.010 under WCR-t. WCU and WCR DISAGREE on the only rejection in the battery,
#     which is precisely why the promotion decision cannot be made silently.
#   * Reported SEs barely move: bootstrap sd vs CRVE-from-IF differ by <= 3.5% across
#     all 11 coefficients, so the mode changes the p-value, not the standard error.
#
# TO PROMOTE the corrected reference to default, re-run and re-export, in order:
#   1. estimation_2_sleep.py (E1/E2, all specs)  -> est1/est2 pickles
#   2. estimation_sleep_common.py --est 3..8     -> E3-E8 pickles (~11h)
#   3. estimation_bbl_1_polfunc.py               -> BBL policy-function SEs
#   4. export_1_sleep_results.py + the sleep_first_stage_pooled*.tex exporters
#   5. run_phi_diagnostics.py (full D0-D12 battery) -> DIAG_PHI_SEPARATION csvs
#   6. hand-update the p-values quoted in identification_notes.md and V_Main.tex
# Point estimates are untouched by any of this: the mode changes only the reference
# distribution used to convert a statistic into a p-value.
#
# MODES (env SLEEP_WCB_MODE):
#   unset / "normal"      historical behaviour (bootstrap-sd SE + normal reference)
#   "studentised"/"wcu"   bootstrap-t: store t*_b = (theta*_b - theta_hat)/se*_b and
#                         read p = P(|t*| >= |t_obs|); null NOT imposed
#   "wcr"                 restricted bootstrap-t: residuals used to build the
#                         bootstrap DGP come from the fit with H0: theta_k = 0
#                         imposed (exact in the linear path; see the fallback note)
# ==============================================================================
_WCB_ALIASES = {
    "": "normal", "normal": "normal", "default": "normal", "legacy": "normal",
    "off": "normal", "wald": "normal",
    "studentised": "wcu", "studentized": "wcu", "wcu": "wcu", "wcu-t": "wcu",
    "boott": "wcu", "boot-t": "wcu",
    "wcr": "wcr", "wcr-t": "wcr", "restricted": "wcr",
}
_WCB_WARNED = set()


def wcb_mode(default="normal"):
    """Resolve SLEEP_WCB_MODE -> one of {'normal', 'wcu', 'wcr'}.

    GOTCHA: an unrecognised value RAISES rather than silently falling back, because
    a typo'd 'SLEEP_WCB_MODE=wrc' that quietly produced legacy p-values labelled as
    corrected is exactly the failure this whole exercise is about."""
    raw = os.environ.get("SLEEP_WCB_MODE", default)
    m = _WCB_ALIASES.get(str(raw).strip().lower())
    if m is None:
        raise ValueError(f"SLEEP_WCB_MODE={raw!r} not understood; use one of "
                         f"{sorted(set(_WCB_ALIASES.values()))} "
                         f"(aliases: {sorted(_WCB_ALIASES)})")
    return m


def _boot_pval(t_obs, t_star):
    """Symmetric equal-tail bootstrap p: p = (1 + #{|t*_b| >= |t_obs|}) / (1 + B).

    The (1+.)/(1+B) form (Davidson & MacKinnon 2004, 4.62) is the finite-B unbiased
    version and can never return exactly 0, which the normal-reference path could."""
    t_star = np.asarray(t_star, float)
    ok = np.isfinite(t_star)
    B_eff = int(ok.sum())
    if B_eff == 0 or not np.isfinite(t_obs):
        return float("nan")
    # 1e-12 slack so a draw that ties |t_obs| to machine precision counts as >=
    hits = int(np.sum(np.abs(t_star[ok]) >= abs(t_obs) - 1e-12))
    return float((1.0 + hits) / (B_eff + 1.0))


def _ame_cluster_if(theta_hat, IF_cl, ame_fn, keys, h_rel=1e-5):
    """Per-cluster influence of each AME: IF^a_{g,k} = grad_theta AME_k . IF_g.

    Needed to studentise the multiplier bootstrap: the bootstrap analogue of the
    CRVE at draw b is V*_b = sum_g w_g^2 IF^a_g IF^a_g' (E[w^2]=1, so E[V*_b] is the
    CRVE itself). One-sided finite differences, p extra ame_fn calls, all done
    OUTSIDE the draw loop so the RNG stream is untouched.
    GOTCHA: with Rademacher weights w^2 == 1, so V*_b is constant across draws and
    the studentisation is vacuous (no asymptotic refinement) -- Webb weights, the
    repo default, put w^2 in {0.5, 1, 1.5} and do vary."""
    theta_hat = np.asarray(theta_hat, float)
    p = theta_hat.size
    a0 = ame_fn(theta_hat)
    J = np.empty((p, len(keys)))
    for j in range(p):
        h = h_rel * max(1.0, abs(float(theta_hat[j])))
        tp = theta_hat.copy()
        tp[j] += h
        a1 = ame_fn(tp)
        for ki, k in enumerate(keys):
            J[j, ki] = (float(a1[k]) - float(a0[k])) / h
    return np.asarray(IF_cl, float) @ J          # (n_cl x p) @ (p x K) -> n_cl x K


def cluster_wild_bootstrap(theta_hat, IF_cl, ame_fn, ame_hat, B=199,
                           scheme="rademacher", rng=None, mode=None):
    """Score/multiplier wild cluster bootstrap (Kline & Santos 2012): perturb the
    cluster-summed influence functions by wild weights, recompute the (linearised)
    AME via ame_fn, and read SEs/p-values off the bootstrap distribution.
    Returns (bse dict, pvals dict).

    mode=None reads SLEEP_WCB_MODE (default 'normal' = historical behaviour).
      'normal' -> bse = sd(draws), p = 2*Phi(-|ame|/sd)          [legacy]
      'wcu'    -> bse = CRVE-from-IF, p = bootstrap tail mass of |t*|, t* studentised
                  by the per-draw V*_b = sum_g w_g^2 IF^a_g IF^a_g'
      'wcr'    -> not available on this generic M-estimator path (imposing H0 needs a
                  RESTRICTED refit, which this signature has no handle on); falls back
                  to 'wcu' with a one-time warning. The linear path
                  (linear_wild_cluster_bootstrap) implements WCR exactly.
    The draws themselves are identical across modes -- same weights, same RNG
    consumption -- so switching mode cannot move a point estimate or a draw."""
    rng = rng or np.random.default_rng(0)
    mode = wcb_mode() if mode is None else mode
    if mode == "wcr":
        if "generic-wcr" not in _WCB_WARNED:
            _WCB_WARNED.add("generic-wcr")
            print("  [WCB] SLEEP_WCB_MODE=wcr requested on the generic score path "
                  "(nonlinear AMEs): H0 cannot be imposed without a restricted refit; "
                  "using studentised WCU here. The linear estimators use true WCR.")
        mode = "wcu"
    n_cl = IF_cl.shape[0]
    keys = list(ame_hat.keys())
    draws = {k: np.empty(B) for k in keys}
    stud = (mode != "normal")
    if stud:
        IF_a = _ame_cluster_if(theta_hat, IF_cl, ame_fn, keys)     # n_cl x K
        se_hat = np.sqrt(np.sum(IF_a ** 2, axis=0))                # CRVE-from-IF
        IF_a2 = IF_a ** 2
        Vb = np.empty((B, len(keys)))
    for b in range(B):
        wv = _wild_weights(n_cl, scheme, rng)
        theta_b = theta_hat + wv @ IF_cl
        a_b = ame_fn(theta_b)
        for k in keys:
            draws[k][b] = a_b[k]
        if stud:
            Vb[b] = (wv ** 2) @ IF_a2       # diag of the bootstrap-draw CRVE
    bse, pvals = {}, {}
    for ki, k in enumerate(keys):
        sd = float(np.std(draws[k], ddof=1))
        if not stud:
            bse[k] = sd
            # symmetric p for H0: AME=0, NORMAL reference (legacy; see block above)
            z = abs(ame_hat[k]) / sd if sd > 0 else np.inf
            pvals[k] = float(2 * stats.norm.sf(z)) if np.isfinite(z) else 0.0
            continue
        s0 = float(se_hat[ki])
        bse[k] = s0 if s0 > 0 else sd
        se_b = np.sqrt(np.clip(Vb[:, ki], 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_star = np.where(se_b > 0, (draws[k] - ame_hat[k]) / se_b, np.nan)
        t_obs = (ame_hat[k] / s0) if s0 > 0 else np.nan
        pvals[k] = _boot_pval(t_obs, t_star)
    return bse, pvals


def _linear_wcb_t(res, B, scheme, seed, restricted, tstats_out=None):
    """True (restricted or unrestricted) wild cluster bootstrap-t for a fitted
    statsmodels cluster-OLS. Opt-in path for SLEEP_WCB_MODE in {wcu, wcr}; the
    default 'normal' path never reaches here.

    Per coefficient j, H0: beta_j = 0.
      WCR: beta~ = OLS with column j DROPPED (the null imposed), u~ = y - X beta~.
      WCU: beta~ = beta_hat (unrestricted), u~ = u_hat.
      draw b: u*_ig = w_g u~_ig,  y* = X beta~ + u*,
              beta*_b = beta~ + A^{-1} sum_g w_g s~_g       (A = X'X)
              u_hat*_ig = w_g u~_ig - X_i'(beta*_b - beta~)
              s*_g = w_g s~_g - Q_g (beta*_b - beta~),       Q_g = X_g' X_g
              V*_b = A^{-1} (sum_g s*_g s*_g') A^{-1}
              t*_b = (beta*_bj - beta~_j) / se*_bj
    NOTE the numerator is `delta_j = beta*_bj - beta~_j` under BOTH variants: under
    WCR the DGP's true beta_j is 0 and beta~_j = 0, so beta*_bj - 0 = delta_j; under
    WCU the DGP's true beta_j is beta_hat_j = beta~_j. The two variants differ ONLY
    in which residuals generate the bootstrap DGP -- that is the whole of the WCR/WCU
    distinction, and it is why WCR is essentially free here.
    t_obs = beta_hat_j / se_hat_j with se_hat from the SAME (uncorrected) CRVE
    formula, so the G/(G-1)*(N-1)/(N-K) finite-sample factor cancels in the test and
    is omitted. p = (1 + #{|t*| >= |t_obs|}) / (1 + B).

    Everything is done in cluster-sum space -- Sy_g = sum_{i in g} X_i y_i and
    Q_g -- so no per-coefficient pass over the N ~ 1e6 rows is ever needed:
    S(b) = Sy - Q @ b for any coefficient vector b."""
    idx = res.params.index
    names = list(idx)
    beta = np.asarray(res.params, float)
    X = np.asarray(res.model.exog, float)
    y = np.asarray(res.model.endog, float)
    cl = pd.Series(np.asarray(res.cov_kwds["groups"])).astype(str).values
    cl_u, cl_inv = np.unique(cl, return_inverse=True)
    n_cl, K = len(cl_u), X.shape[1]

    Sy = np.empty((n_cl, K))                     # sum_{i in g} X_i y_i
    for a in range(K):
        Sy[:, a] = np.bincount(cl_inv, weights=X[:, a] * y, minlength=n_cl)
    Q = np.empty((n_cl, K, K))                   # X_g' X_g, per cluster
    for a in range(K):
        for c in range(a, K):
            v = np.bincount(cl_inv, weights=X[:, a] * X[:, c], minlength=n_cl)
            Q[:, a, c] = v
            Q[:, c, a] = v
    A = Q.sum(axis=0)                            # X'X
    Ainv = np.linalg.pinv(A)
    Xty = Sy.sum(axis=0)                         # X'y

    def _crve_diag(S):
        M = Ainv @ (S.T @ S) @ Ainv
        return np.diag(M)

    se_hat = np.sqrt(np.clip(_crve_diag(Sy - Q @ beta), 0.0, None))

    bse, pvals = {}, {}
    for j, nm in enumerate(names):
        s0 = float(se_hat[j])
        bse[nm] = s0
        t_obs = (beta[j] / s0) if s0 > 0 else np.nan
        if restricted:
            keep = [c for c in range(K) if c != j]
            b_null = np.zeros(K)
            if keep:
                kk = np.ix_(keep, keep)
                b_null[keep] = np.linalg.lstsq(A[kk], Xty[list(keep)], rcond=None)[0]
        else:
            b_null = beta
        S0 = Sy - Q @ b_null                     # cluster scores of the DGP residuals
        # own RNG per coefficient: the restricted DGP differs by coefficient, so a
        # shared stream would make the draws mutually dependent for no benefit.
        rng = np.random.default_rng(seed + 7919 * (j + 1))
        t_star = np.empty(B)
        for b in range(B):
            w = _wild_weights(n_cl, scheme, rng)
            delta = Ainv @ (w @ S0)              # beta*_b - b_null
            Sb = S0 * w[:, None] - Q @ delta     # bootstrap-sample cluster scores
            v = float((Ainv @ (Sb.T @ Sb) @ Ainv)[j, j])
            t_star[b] = delta[j] / np.sqrt(v) if v > 0 else np.nan
        pvals[nm] = _boot_pval(t_obs, t_star)
        if tstats_out is not None:      # diagnostics only (diag_wcb_reference.py)
            tstats_out[nm] = (t_obs, t_star.copy())
    return bse, pvals


def linear_wild_cluster_bootstrap(res, B=999, scheme="webb", seed=0, mode=None,
                                  tstats_out=None):
    """Score/multiplier wild cluster bootstrap SEs/p-values for a fitted statsmodels
    cluster-OLS result, so the LINEAR sleepiness estimators (Est1/Est2) share ONE
    inference method with the single-index/joint columns (Cameron-Gelbach-Miller 2008;
    MacKinnon-Webb 2017; Kline-Santos 2012). The cluster influence functions are read
    straight off the fit -- IF_cl[g] = (X'X)^{-1} sum_{i in g} X_i u_hat_i -- and
    perturbed by wild weights via the same cluster_wild_bootstrap used for the
    nonlinear AMEs (identity map: the parameters ARE the coefficients). No refit.
    Returns (bse, tvalues, pvalues) as pandas Series indexed like res.params.

    mode=None reads SLEEP_WCB_MODE; 'normal' (the default) is the historical code
    below, bit-for-bit. 'wcu'/'wcr' route to the bootstrap-t in _linear_wcb_t."""
    idx = res.params.index
    names = list(idx)
    beta = np.asarray(res.params, float)
    mode = wcb_mode() if mode is None else mode
    if mode != "normal":
        bse, pvals = _linear_wcb_t(res, B=B, scheme=scheme, seed=seed,
                                   restricted=(mode == "wcr"), tstats_out=tstats_out)
        bse_s = pd.Series({nm: bse[nm] for nm in names}).reindex(idx)
        pv_s = pd.Series({nm: pvals[nm] for nm in names}).reindex(idx)
        tv_s = pd.Series({nm: (float(b) / bse[nm] if bse[nm] else np.nan)
                          for nm, b in zip(names, beta)}).reindex(idx)
        return bse_s, tv_s, pv_s
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
    bse, pvals = cluster_wild_bootstrap(beta, IF_cl, ame_fn, ame_hat, B=B, scheme=scheme,
                                        rng=np.random.default_rng(seed), mode="normal")
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
def phi_t_group_struct(df_ss, market_key=None, weight="mean", time_key=None):
    """(time, market) group structure for national phi_t aggregation, parameterised by the
    two conventions that actually differ across this codebase.

    weight="mean"  each market enters weighted by its POPULATION. This is the estimand as
                   written down in estimation_1_sleep.py:394 ("phi_t = sum_m phi_mt*M_mt /
                   sum_m M_mt over MARKETS m") and in this module's header.
    weight="sum"   each market enters weighted by population x THE NUMBER OF BANK ROWS in
                   the cell. pop_total is constant within a (quarter, market) cell, so
                   M_mt=("market_size","sum") -- what calculate_phis and
                   calculate_pooled_phis both use -- multiplies by the bank count.

    Measured 2026-08-04 on est6 spec 12: the two weights move the LEVEL of national
    sleepiness by 1.90 pp (0.9720 vs 0.9531) from identical phi_mt, against a 3.76 pp
    movement being interpreted; the RANGE is nearly unchanged (3.76 vs 3.50 pp). The
    market key matters far less (CODMUN_IBGE vs mca_code: 0.21 pp). The parameter exists
    so a band can always be built on EXACTLY the convention its own point series uses --
    the mismatch between the two is what made the stored E5/E6 bands sit 1.5-2.2 pp away
    from the phi_t they shipped next to.
    """
    if weight not in ("mean", "sum"):
        raise ValueError(f"weight must be 'mean' or 'sum', got {weight!r}")
    tcol = time_key or ("time_id" if "time_id" in df_ss.columns else "year_quarter")
    t = df_ss[tcol].astype(str).values
    if market_key is None:
        # MARKET = MCA (V_Main.tex:182); municipality only as a fallback. See
        # _phi_t_group_struct for the 2026-08-05 note.
        market_key = next((k for k in ("mca_code", "CODMUN_IBGE") if k in df_ss.columns), None)
    if market_key is not None and market_key in df_ss.columns:
        m = df_ss[market_key].astype(str).values
    else:
        m = np.array(["0"] * len(df_ss))
    pop = (df_ss["pop_total"].fillna(0).values.astype(float)
           if "pop_total" in df_ss.columns else np.ones(len(df_ss)))
    key = np.char.add(np.char.add(t, "|"), m)
    gcode, _ = pd.factorize(key)
    ng = int(gcode.max()) + 1
    gcount = np.bincount(gcode, minlength=ng).astype(float)
    gpop = np.bincount(gcode, weights=pop, minlength=ng)
    if weight == "mean":
        gpop = gpop / np.maximum(gcount, 1.0)
    tcode, tuniq = pd.factorize(df_ss[tcol].astype(str).values)
    tcode_g = np.zeros(ng, int)
    tcode_g[gcode] = tcode
    return dict(gcode=gcode, gcount=gcount, gpop=gpop, tcode_g=tcode_g,
                nt=len(tuniq), tuniq=np.asarray(tuniq))


def linear_phi_t_band(res, Z, coef_names, df_agg, market_key=None, weight="mean",
                      time_key=None, B=None, scheme=None, seed=20240624):
    """National phi_t point path + wild-cluster percentile band for the LINEAR sleepiness
    estimators (Est1/Est2), where phi_mt = Z @ beta_state is linear in the fitted
    coefficients, so no refit is needed at any draw.

    Same inference scheme as every other column: the cluster influence functions are read
    off the fit, IF_g = (X'X)^{-1} sum_{i in g} X_i u_i, perturbed by wild weights
    (Cameron-Gelbach-Miller 2008; MacKinnon-Webb 2017), and the band is the percentile
    interval of the resulting national paths.

    Z          (n_agg x k) design for phi on the AGGREGATION sample, built by the caller so
               that Z @ beta reproduces its own phi_mt exactly -- including whatever NaN fill
               that caller uses (Est1 median-fills, Est2 zero-fills).
    coef_names names of the k coefficients, aligned to Z's columns, as they appear in
               res.params.
    df_agg     the frame Z was built from; supplies the time/market/pop columns.
    weight     aggregation convention -- pass the one the caller's point series uses, so the
               band and the point path can never drift apart (see phi_t_group_struct).
    """
    names = list(res.params.index)
    beta = np.asarray(res.params, float)
    missing = [c for c in coef_names if c not in names]
    if missing:
        raise KeyError(f"coefficients absent from the fit: {missing}")
    pos = np.array([names.index(c) for c in coef_names], int)
    Z = np.asarray(Z, float)
    if Z.shape[1] != len(coef_names):
        raise ValueError(f"Z has {Z.shape[1]} columns but {len(coef_names)} coefficient names")
    if len(Z) != len(df_agg):
        raise ValueError(f"Z has {len(Z)} rows but df_agg has {len(df_agg)}")

    X = np.asarray(res.model.exog, float)
    u = np.asarray(res.resid, float)
    cl = pd.Series(np.asarray(res.cov_kwds["groups"])).astype(str).values
    _, cl_inv = np.unique(cl, return_inverse=True)
    n_cl = int(cl_inv.max()) + 1
    IF_cl = _cluster_if(X * u[:, None], np.linalg.pinv(X.T @ X), cl_inv, n_cl)

    _B, _scheme = boot_cfg()
    B = _B if B is None else int(B)
    scheme = _scheme if scheme is None else scheme
    gs = phi_t_group_struct(df_agg, market_key=market_key, weight=weight, time_key=time_key)

    def _draw(rng_):
        return Z @ (beta + _wild_weights(n_cl, scheme, rng_) @ IF_cl)[pos]

    return _phi_t_band(gs, Z @ beta[pos], _draw, B, scheme, np.random.default_rng(seed))


def attach_phi_band(national_agg, res, Z, coef_names, df_agg, phi_mt, safe_key,
                    market_key=None, weight="mean", time_key="year_quarter"):
    # weight default flipped "sum" -> "mean" 2026-08-05 together with the three point-series
    # writers: the reported national phi_t now weights markets by POPULATION (the stated
    # estimand), not population x bank count. The self-check below enforces the match.
    """Add phi_t_lo_<key> / phi_t_hi_<key> to a linear estimator's national phi_t frame.

    Two self-checks run every time, because both failures have actually happened here:
      1. Z @ beta must reproduce the caller's own phi_mt (catches a design matrix built with a
         different NaN fill or column order than the phi it is supposed to explain);
      2. the band's point path must reproduce the point series it is attached to (catches the
         band and the point path being aggregated on different conventions -- the Est5-Est8
         bug found 2026-08-04, where the two sat 1.5-2.2 pp apart).
    Either mismatch prints a loud warning rather than failing the run, and a failure to build
    the band at all leaves national_agg untouched: an estimation that produced good point
    estimates should not be lost to a reporting extra.

    Set SLEEP_PHI_BAND_LINEAR=0 to skip (smoke runs).
    """
    if os.environ.get("SLEEP_PHI_BAND_LINEAR", "1") != "1":
        return national_agg
    try:
        beta = np.asarray(res.params[coef_names], float)
        d_phi = float(np.nanmax(np.abs(np.asarray(Z, float) @ beta - np.asarray(phi_mt, float))))
        if d_phi > 1e-8:
            print(f"  [phi-band {safe_key}] WARNING design/phi mismatch {d_phi:.3g} "
                  f"-- band would describe a different phi than the one reported")
        band = linear_phi_t_band(res, Z, coef_names, df_agg, market_key=market_key,
                                 weight=weight, time_key=time_key)
        pt_col = f"phi_t_{safe_key}"
        # Carry EVERY band column the frame offers, not just lo/hi. The bias-corrected columns
        # (lo_bc/hi_bc/p_below) are computed by _phi_t_band for free, and the linear estimators
        # store their band ONLY here -- there is no phi_t_boot on a statsmodels result to fall
        # back on, unlike E5-E8. Selecting a fixed lo/hi pair silently dropped them.
        ren = {"time_id": time_key, "phi_t": "_band_pt",
               "lo": f"phi_t_lo_{safe_key}", "hi": f"phi_t_hi_{safe_key}",
               "lo_bc": f"phi_t_lobc_{safe_key}", "hi_bc": f"phi_t_hibc_{safe_key}",
               "p_below": f"phi_t_pbelow_{safe_key}"}
        cols = [c for c in ren if c in band.columns]
        merged = national_agg.merge(band[cols].rename(columns=ren), on=time_key, how="left")
        d_pt = float(np.nanmax(np.abs(merged["_band_pt"] - merged[pt_col])))
        if d_pt > 1e-6:
            print(f"  [phi-band {safe_key}] WARNING band point path differs from the reported "
                  f"phi_t by up to {100*d_pt:.3f} pp -- aggregation conventions disagree")
        merged = merged.drop(columns=["_band_pt"])
        w = 100 * float((merged[f"phi_t_hi_{safe_key}"] - merged[f"phi_t_lo_{safe_key}"]).mean())
        rng = 100 * float(merged[pt_col].max() - merged[pt_col].min())
        extra = ""
        if f"phi_t_lobc_{safe_key}" in merged.columns:
            w_bc = 100 * float((merged[f"phi_t_hibc_{safe_key}"]
                                - merged[f"phi_t_lobc_{safe_key}"]).mean())
            pb = merged[f"phi_t_pbelow_{safe_key}"]
            n_undef = int(((pb <= 0.025) | (pb >= 0.975)).sum())
            extra = (f" | BC width {w_bc:.2f}pp"
                     + (f" | {n_undef} quarter(s) beyond BC repair" if n_undef else ""))
        print(f"  [phi-band {safe_key}] mean width {w:.2f}pp | phi_t range {rng:.2f}pp | "
              f"width/range {w/max(rng, 1e-9):.2f}{extra}")
        return merged
    except Exception as e:
        print(f"  [phi-band {safe_key}] skipped ({type(e).__name__}: {e})")
        return national_agg


def _phi_t_group_struct(df_ss, market_key=None):
    """Precompute the (time, market) group structure for fast pop-weighted
    national phi_t aggregation: national phi_t = sum_market pop_market *
    mean_market(phi_mt) / sum_market pop_market, per time.

    MARKET = MCA (2026-08-05). V_Main.tex:182 defines the local market as the Minimal
    Comparable Area -- "B firms compete in local markets - defined here as Minimal Comparable
    Areas (MCAs)" -- and :303/:406 sum over m in that market set. This function previously
    hardcoded CODMUN_IBGE (municipality), which splits MCAs that were created precisely to keep
    territorial units comparable across boundary changes; measured effect on national phi_t is
    ~0.21 pp, small but wrong by the model's own definition. Pass market_key explicitly to
    reproduce a legacy CODMUN_IBGE band (the unconditional-band correctness gate does this)."""
    t = df_ss["time_id"].astype(str).values
    if market_key is None:
        market_key = next((k for k in ("mca_code", "CODMUN_IBGE") if k in df_ss.columns), None)
    if market_key is not None and market_key in df_ss.columns:
        m = df_ss[market_key].astype(str).values
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
    """National phi_t point path + bootstrap band from B wild draws.

    draw_phi_fn(rng) returns a per-observation phi vector for one bootstrap draw.
    Returns a DataFrame [time_id, phi_t, lo, hi, lo_bc, hi_bc, p_below] sorted by quarter.

    TWO bands are returned, deliberately:

    `lo`/`hi` -- the RAW percentile interval (unchanged; every stored band predating
    2026-08-04 is this). It is kept because its failure mode is diagnostic: a raw percentile
    interval can EXCLUDE its own point estimate, and on E7 spec 12 it does so in 26 of 35
    quarters. That happens when the map from parameters to phi is nonlinear enough to displace
    the draw cloud off the estimate. The joint sieve stacks three nonlinearities in one draw --
    renormalise theta onto the unit sphere, interpolate through a kinked link grid, clip to
    [0,1] -- and when the fitted link is nearly flat (E7's spans 1.33 pp) with the point sitting
    ON its floor, the draws end up one-sided. The single-index path cannot do this: its draw is
    `vpow @ (b + w.IF)`, linear in the perturbed coefficients up to the clip, so the draws are
    mechanically centred (0 violations in all 32 E5/E6 cells).

    `lo_bc`/`hi_bc` -- Efron's BIAS-CORRECTED percentile interval (Efron 1987), which is the
    repair: measure the median bias of the draw cloud as z0 = Phi^-1(P[draw < estimate]) and read
    the interval off the shifted quantile levels Phi(2*z0 +/- z_{alpha/2}). z0 = 0 (a centred
    cloud) reproduces the raw interval exactly, so this is a strict generalisation and the
    linear/single-index paths are unaffected.

    NOT done here: recentring the draws on the point estimate. That would force containment
    everywhere and destroy the very signal that exposed E7's degenerate link.

    `p_below` -- P[draw < estimate], the displacement diagnostic itself, computed with the
    (1+.)/(1+B) finite-B correction used elsewhere in this module so z0 stays finite. Quarters
    where it hits the clamp are DISPLACED BEYOND CORRECTION: no bootstrap draw reaches the
    estimate, so no monotone reparametrisation of the quantile levels can bracket it and the BC
    interval is undefined-by-construction there. Callers must flag those, never report silently.
    """
    pt = _agg_phi_t(phi_point, gs)
    draws = np.empty((B, gs["nt"]))
    for b in range(B):
        draws[b] = _agg_phi_t(draw_phi_fn(rng), gs)
    return _band_from_draws(pt, draws, gs["tuniq"], alpha=alpha)


def _band_from_draws(pt, draws, tuniq, alpha=0.05, label="phi_t band"):
    """Assemble the band frame from a precomputed (B x nt) national draw matrix.

    Split out of _phi_t_band (2026-08-05) so the unconditional band -- whose draw loop must run
    once and feed THREE variants (total / direction-only / link-only) from the same expensive
    re-profiled fits -- can reuse the identical raw-percentile + Efron-BC + p_below assembly
    instead of re-implementing it. Same math, same warnings, same column layout."""
    B = draws.shape[0]
    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)

    # --- Efron (1987) bias correction, per quarter ------------------------------------------
    from scipy.stats import norm as _norm
    p_below = (1.0 + np.sum(draws < pt[None, :], axis=0)) / (1.0 + B)
    z0 = _norm.ppf(np.clip(p_below, 1.0 / (1.0 + B), B / (1.0 + B)))
    za_lo, za_hi = _norm.ppf(alpha / 2), _norm.ppf(1 - alpha / 2)
    a_lo = 100.0 * _norm.cdf(2 * z0 + za_lo)
    a_hi = 100.0 * _norm.cdf(2 * z0 + za_hi)
    nt = draws.shape[1]
    lo_bc = np.empty(nt)
    hi_bc = np.empty(nt)
    for t in range(nt):
        lo_bc[t] = np.percentile(draws[:, t], a_lo[t])
        hi_bc[t] = np.percentile(draws[:, t], a_hi[t])

    out = pd.DataFrame({"time_id": np.asarray(tuniq), "phi_t": pt, "lo": lo, "hi": hi,
                        "lo_bc": lo_bc, "hi_bc": hi_bc, "p_below": p_below})
    n_out = int(np.sum((pt < lo) | (pt > hi)))
    n_undef = int(np.sum((p_below <= alpha / 2) | (p_below >= 1 - alpha / 2)))
    if n_out or n_undef:
        print(f"  [{label}] RAW percentile interval excludes the point estimate in "
              f"{n_out}/{nt} quarters; {n_undef} quarter(s) displaced beyond BC repair "
              f"(p_below at the clamp). Report the BC interval, and flag the undefined quarters.")
    try:
        out["_d"] = pd.PeriodIndex(out["time_id"].str.replace("Q", "Q"), freq="Q").to_timestamp()
        out = out.sort_values("_d").reset_index(drop=True)
    except Exception:
        out = out.sort_values("time_id").reset_index(drop=True)
    return out


# ==============================================================================
# UNCONDITIONAL national phi_t band for the single-index estimators (Est5/Est6)
# ==============================================================================
# The stored bands condition on the estimated DIRECTION theta-hat: they perturb only the
# sieve-link coefficients b through vpow built once at theta-hat. But theta-hat is the object
# the 2026-08 multiplicity diagnostics show is weakly identified, and conditioning on a weakly
# identified quantity can only understate uncertainty. The two functions below produce the
# joint (direction + link) band without refitting the production estimator:
#
#   nlls_direction_if      cluster influence functions of the NLLS direction, reconstructed at
#                          the STORED solution (they were never persisted: the E5/E6 pipeline
#                          calls fit_nlls_link with bootstrap=False).
#   unconditional_phi_t_band
#                          per wild draw w: theta_b = theta_hat + w@IF_theta; re-standardize
#                          the index; RE-PROFILE the sieve link exactly (one OLS -- the link
#                          solver has no constraints); add the conditional-link channel with
#                          the SAME w. One refit per draw feeds three variants (total /
#                          direction-only / link-only); link-only must reproduce the stored
#                          conditional band, which is the blocking correctness gate.
#
# Composition note (no double counting): the refit b-hat(theta_b) on the ORIGINAL y carries the
# same level noise e as the baseline b-hat(theta-hat); e cancels in the deviation, leaving the
# exact d b-hat / d theta channel. The link's own sampling noise enters once, through +w@IF_b.
# Sharing w across the two channels keeps the theta-b covariance (Kline-Santos multiplier
# bootstrap on the stacked estimating equations, with the theta->b Jacobian handled exactly by
# re-profiling). Not propagated (documented): the first-stage CF v_hat (generated regressor)
# and the aggregation weights.
def nlls_direction_if(df, state_cols, has_cf, theta_native, loss, link="logit",
                      fe_time_col=None, fd_check=True):
    """Cluster IFs of the NLLS direction at a stored solution. Returns a dict:
    IF_cl (n_cl x K, theta rows only), gamma_hat, foc_norm, cl_inv, n_cl, diag.

    The production fit discarded the scipy result, so everything is rebuilt: the arg tuple
    exactly as fit_nlls_link (dropna subset, demean index arrays), the CF coefficient gamma
    PROFILED at theta-hat under the matching loss (gamma enters the residual linearly), an
    ANALYTIC Jacobian, and the psi-weighted M-estimator sandwich:

        score_i = rho'(f_i^2) * f_i * J_i          (LS: rho' == 1)
        bread   = pinv( sum_i max(rho' + 2 rho'' f^2, 0) * J_i J_i' )   (clamped GN)

    Reusing scipy's res.jac naively would be WRONG under robust loss: scipy stores the
    sqrt(weight)-scaled Jacobian with RAW residuals, and its EPS clamp zeroes the scores of
    exactly the influential rows (verified scipy 1.17.1 _lsq/trf.py:409-412). For
    loss='linear' the sandwich below reduces to the standard OLS-type one, exactly."""
    CF_cols = ["v_hat_x_lagged_dep"] if has_cf else []
    cols = state_cols + ["deposit_balance", "nr_lagged_dep", "entity_id"]
    df_ss = df.dropna(subset=cols + CF_cols).copy()
    entities = df_ss["entity_id"].unique()
    emap = {e: i for i, e in enumerate(entities)}
    entity_idx = df_ss["entity_id"].map(emap).values
    ecounts = np.bincount(entity_idx).astype(float)
    if fe_time_col is not None:
        _, tinv = np.unique(df_ss[fe_time_col].values, return_inverse=True)
        tcounts = np.bincount(tinv).astype(float)
    else:
        tinv = tcounts = None
    y_dm = _twoway_demean(df_ss["deposit_balance"].values.astype(float),
                          entity_idx, ecounts, tinv, tcounts)
    X = df_ss[state_cols].values.astype(float)
    Z = df_ss["nr_lagged_dep"].values.astype(float)
    CF = df_ss[CF_cols].values.astype(float) if has_cf else np.empty((len(df_ss), 0))
    K, G = X.shape[1], CF.shape[1]

    idx = [f"interaction_{sv}" if sv != "constant" else "nr_lagged_dep"
           for sv in state_cols]
    theta = np.asarray(pd.Series(theta_native).reindex(idx).values, float)
    if not np.all(np.isfinite(theta)):
        raise ValueError(f"theta_native missing entries for {idx}")

    def _dm(M):
        return _twoway_demean(M, entity_idx, ecounts, tinv, tcounts)

    # --- profile gamma at theta-hat under the matching loss ---------------------------------
    # r(theta, gamma) = y_dm - dm(phi(X theta) Z) - dm(CF) gamma  is linear in gamma.
    phi_v = link_cdf(X @ theta, link)
    u = y_dm - _dm(phi_v * Z)
    if G > 0:
        CF_dm = np.column_stack([_dm(CF[:, j]) for j in range(G)])
        gamma, *_ = np.linalg.lstsq(CF_dm, u, rcond=None)
        if loss == "cauchy":                    # IRLS with rho'(f^2) weights
            for _it in range(25):
                f = u - CF_dm @ gamma
                wls = 1.0 / (1.0 + f * f)       # rho'(z) = 1/(1+z), z = f^2
                sw = np.sqrt(wls)
                g_new, *_ = np.linalg.lstsq(CF_dm * sw[:, None], u * sw, rcond=None)
                if np.max(np.abs(g_new - gamma)) < 1e-10 * max(1.0, float(np.max(np.abs(gamma)))):
                    gamma = g_new
                    break
                gamma = g_new
    else:
        CF_dm = np.empty((len(df_ss), 0))
        gamma = np.empty(0)

    psi = np.concatenate([theta, gamma])
    f = _nlls_resid(psi, y_dm, X, Z, CF, entity_idx, link, ecounts, tinv, tcounts)

    # --- analytic Jacobian of the residual ---------------------------------------------------
    # dr/dtheta_k = -dm( g'(X theta) * X_k * Z );  dr/dgamma_j = -dm( CF_j )
    gp = _link_density(X @ theta, link)
    base = gp * Z
    J = np.empty((len(df_ss), K + G))
    for k in range(K):
        J[:, k] = -_dm(base * X[:, k])
    if G > 0:
        J[:, K:] = -CF_dm

    if fd_check:
        rng_fd = np.random.default_rng(0)
        sub = rng_fd.choice(len(df_ss), size=min(20000, len(df_ss)), replace=False)
        worst = 0.0
        for k in rng_fd.choice(K + G, size=min(3, K + G), replace=False):
            h = 1e-6 * max(1.0, abs(psi[k]))
            pp = psi.copy()
            pp[k] += h
            fd = (_nlls_resid(pp, y_dm, X, Z, CF, entity_idx, link,
                              ecounts, tinv, tcounts) - f) / h
            num = float(np.max(np.abs(fd[sub] - J[sub, k])))
            den = max(1e-12, float(np.max(np.abs(J[sub, k]))))
            worst = max(worst, num / den)
        if worst > 1e-4:
            raise RuntimeError(f"analytic Jacobian fails FD check (rel err {worst:.2e})")
    else:
        worst = np.nan

    # --- psi-weighted M-estimator sandwich ---------------------------------------------------
    if loss == "linear":
        rho1 = np.ones(len(f))
        w2 = np.ones(len(f))
    elif loss == "cauchy":
        z = f * f
        rho1 = 1.0 / (1.0 + z)                              # rho'(z)
        w2 = np.maximum((1.0 - z) / (1.0 + z) ** 2, 0.0)    # rho' + 2 rho'' z, clamped
    else:
        raise ValueError(f"unsupported loss {loss!r}")
    score = J * (rho1 * f)[:, None]
    foc = J.T @ (rho1 * f)
    foc_norm = float(np.linalg.norm(foc) / max(1.0, float(np.linalg.norm(rho1 * f))))
    bread = np.linalg.pinv((J * w2[:, None]).T @ J)

    cl = df_ss["CodConglomeradoPrudencial"].astype(str)
    cl_u, cl_inv = np.unique(cl.values, return_inverse=True)
    n_cl = len(cl_u)
    IF_full = _cluster_if(score, bread, cl_inv, n_cl)       # n_cl x (K+G)

    return dict(IF_cl=IF_full[:, :K], gamma_hat=gamma, foc_norm=foc_norm,
                cl_inv=cl_inv, n_cl=n_cl, theta=theta, idx=idx,
                diag=dict(jac_fd_relerr=worst, n=len(df_ss), K=K, G=G,
                          clamped_rows=int(np.sum(w2 <= 0.0))))


def unconditional_phi_t_band(df, state_cols, has_cf, si_res, loss, degree=3,
                             fe_time_col=None, B=400, scheme=None, seed=20240624,
                             keep_draws=True):
    """Joint (direction + link) national phi_t band for a stored fit_single_index result.

    Per wild draw with ONE weight vector w per cluster:
        theta_b = theta_hat + w @ IF_theta          (nlls_direction_if, at the stored solution)
        re-standardize the index at theta_b         (vmu_b, vsd_b recomputed -- this makes the
                                                     band exactly invariant to scale/shift
                                                     noise in theta, as phi itself is)
        b_hat(theta_b) by exact re-profiling        (one OLS on the original y)
        total draw  = clip(vpow_b @ (b_hat(theta_b) + w @ IF_b)[:degree+1])
        theta-only  = clip(vpow_b @  b_hat(theta_b)[:degree+1])
        link-only   = clip(vpow   @ (b_full        + w @ IF_b)[:degree+1])   (no refit)

    One refit per draw feeds all three variants. The link-only variant must reproduce the
    stored conditional phi_t_boot (same seed, same B) -- the blocking correctness gate.
    Returns a dict with the three band frames, per-draw diagnostics, meta, and (optionally)
    the raw (B x nt) draw matrices."""
    # Build the index name list from state_cols, NOT from si_res.params.index: `params` holds
    # the AMEs, which OMIT the constant term (nr_lagged_dep) because a constant has no marginal
    # effect, while params_native carries it. Subsetting params_native by the AME index
    # therefore silently drops the constant from theta -- caught by the synthetic smoke test
    # 2026-08-05, and it would have produced a 7-of-8-coefficient index on the real cells.
    idx_expect = [f"interaction_{sv}" if sv != "constant" else "nr_lagged_dep"
                  for sv in state_cols]
    theta_ser = si_res.params_native.reindex(idx_expect)
    if theta_ser.isna().any():
        missing = [k for k in idx_expect if k not in si_res.params_native.index]
        raise KeyError(f"params_native lacks {missing}; state_cols/fit mismatch")

    dirif = nlls_direction_if(df, state_cols, has_cf, theta_ser, loss,
                              link="logit", fe_time_col=fe_time_col)
    IF_th, cl_inv, n_cl = dirif["IF_cl"], dirif["cl_inv"], dirif["n_cl"]
    theta = dirif["theta"]

    # --- rebuild the link stage at theta-hat, mirroring fit_single_index -------------------
    CF_cols = ["v_hat_x_lagged_dep"] if has_cf else []
    cols = state_cols + ["deposit_balance", "nr_lagged_dep", "entity_id"]
    df_ss = df.dropna(subset=cols + CF_cols).copy()
    X = _build_phi_X(df_ss, idx_expect)
    Z = df_ss["nr_lagged_dep"].values.astype(float)
    v = X @ theta
    vmu, vsd = float(v.mean()), float(v.std())
    if vsd <= 0:
        vsd = 1.0
    # fingerprint gates: the frame must reproduce the stored standardization
    if abs(vmu - float(si_res.si_vmu)) > 1e-6 * max(1.0, abs(vmu)) or \
       abs(vsd - float(si_res.si_vsd)) > 1e-6 * max(1.0, abs(vsd)):
        raise RuntimeError(f"vmu/vsd fingerprint mismatch: rebuilt ({vmu:.10g},{vsd:.10g}) "
                           f"vs stored ({si_res.si_vmu:.10g},{si_res.si_vsd:.10g}) -- data drift")
    vs = (v - vmu) / vsd

    _, einv = np.unique(df_ss["entity_id"].values, return_inverse=True)
    ecounts = np.bincount(einv).astype(float)
    if fe_time_col is not None:
        _, tinv = np.unique(df_ss[fe_time_col].values, return_inverse=True)
        tcounts = np.bincount(tinv).astype(float)
    else:
        tinv = tcounts = None

    def _dm(M):
        return _twoway_demean(M, einv, ecounts, tinv, tcounts)

    y_dm = _dm(df_ss["deposit_balance"].values.astype(float))
    Z_dm = _dm(Z)                                            # the d=0 column, never changes
    if has_cf:
        CF_raw = df_ss["v_hat_x_lagged_dep"].values.astype(float)
        CF_dm = _dm(CF_raw)
    ncols = degree + 1 + (1 if has_cf else 0)

    def _profile(vs_loc):
        """Exact link re-profile at a standardized index: demean the 3 changing columns,
        assemble, one lstsq. Returns b_full (ncols,)."""
        D = np.empty((len(df_ss), ncols))
        D[:, 0] = Z_dm
        for d in range(1, degree + 1):
            D[:, d] = _dm((vs_loc ** d) * Z)
        if has_cf:
            D[:, degree + 1] = CF_dm
        b, *_ = np.linalg.lstsq(D, y_dm, rcond=None)
        return b

    b_full = _profile(vs)
    if np.max(np.abs(b_full[:degree + 1] - np.asarray(si_res.si_b, float))) > 1e-6:
        raise RuntimeError(f"si_b fingerprint mismatch: refit {b_full[:degree+1]} "
                           f"vs stored {np.asarray(si_res.si_b)} -- data drift")

    # conditional-link IFs at theta-hat (mirrors fit_single_index :548-551)
    D0 = np.empty((len(df_ss), ncols))
    D0[:, 0] = Z_dm
    for d in range(1, degree + 1):
        D0[:, d] = _dm((vs ** d) * Z)
    if has_cf:
        D0[:, degree + 1] = CF_dm
    resid0 = y_dm - D0 @ b_full
    bread_b = np.linalg.pinv(D0.T @ D0)
    IF_b = _cluster_if(D0 * resid0[:, None], bread_b, cl_inv, n_cl)

    # TWO group structures from one draw loop. The per-draw REFIT is the expensive part and is
    # independent of how phi is aggregated, so emitting both market keys costs one extra
    # bincount per draw:
    #   gs      MCA -- the market as defined in V_Main.tex:182; the reported deliverable.
    #   gs_leg  CODMUN_IBGE -- the key every STORED band was built on. Needed so the link-only
    #           variant can be checked against those bands bit-for-bit; that gate is what
    #           proves this rebuild path is faithful, and it would be unavailable if the
    #           aggregation changed at the same time as the statistics.
    gs = _phi_t_group_struct(df_ss)
    gs_leg = _phi_t_group_struct(df_ss, market_key="CODMUN_IBGE")
    vpow0 = np.column_stack([vs ** d for d in range(degree + 1)])
    phi_pt = np.clip(vpow0 @ b_full[:degree + 1], 0.0, 1.0)
    pt = _agg_phi_t(phi_pt, gs)
    pt_leg = _agg_phi_t(phi_pt, gs_leg)

    # zero-weight reproduction gate
    if float(np.max(np.abs(np.clip(vpow0 @ (b_full + 0.0 * IF_b[0])[:degree + 1], 0, 1)
                           - phi_pt))) > 1e-12:
        raise RuntimeError("zero-weight draw does not reproduce the point path")

    _B, _scheme = boot_cfg()
    B = int(B if B is not None else min(_B, 400))
    scheme = scheme or _scheme
    rng = np.random.default_rng(seed)
    nt = gs["nt"]
    draws_tot = np.empty((B, nt))
    draws_th = np.empty((B, nt))
    draws_ln = np.empty((B, nt))
    draws_ln_leg = np.empty((B, gs_leg["nt"]))     # legacy key, for the reproduction gate only
    diag_rows = []
    n_fail = 0
    for ib in range(B):
        w = _wild_weights(n_cl, scheme, rng)
        th_b = theta + w @ IF_th
        v_b = X @ th_b
        vmu_b, vsd_b = float(v_b.mean()), float(v_b.std())
        flag_vsd = vsd_b <= 0
        if flag_vsd:
            vsd_b = 1.0
        vs_b = (v_b - vmu_b) / vsd_b
        try:
            b_th = _profile(vs_b)
            if not np.all(np.isfinite(b_th)):
                raise FloatingPointError("non-finite b")
        except Exception:
            b_th = b_full.copy()
            n_fail += 1
        vpow_b = np.column_stack([vs_b ** d for d in range(degree + 1)])
        db = w @ IF_b
        phi_tot = np.clip(vpow_b @ (b_th + db)[:degree + 1], 0.0, 1.0)
        phi_th = np.clip(vpow_b @ b_th[:degree + 1], 0.0, 1.0)
        phi_ln = np.clip(vpow0 @ (b_full + db)[:degree + 1], 0.0, 1.0)
        draws_tot[ib] = _agg_phi_t(phi_tot, gs)
        draws_th[ib] = _agg_phi_t(phi_th, gs)
        draws_ln[ib] = _agg_phi_t(phi_ln, gs)
        draws_ln_leg[ib] = _agg_phi_t(phi_ln, gs_leg)
        nth = float(np.linalg.norm(theta))
        diag_rows.append(dict(
            cos=float(theta @ th_b / max(1e-300, nth * float(np.linalg.norm(th_b)))),
            vsd_ratio=vsd_b / vsd, b_shift=float(np.linalg.norm(b_th - b_full)),
            clip_lo=float(np.mean((vpow_b @ (b_th + db)[:degree + 1]) < 0.0)),
            clip_hi=float(np.mean((vpow_b @ (b_th + db)[:degree + 1]) > 1.0)),
            vsd_flag=bool(flag_vsd)))
        if (ib + 1) % 50 == 0:
            print(f"    [uncond] draw {ib+1}/{B}", flush=True)
    if n_fail:
        print(f"  [uncond] WARNING {n_fail}/{B} refits failed (theta channel suppressed there)"
              + ("  <-- DEGRADED" if n_fail > 0.01 * B else ""))

    out = dict(
        band_total=_band_from_draws(pt, draws_tot, gs["tuniq"], label="uncond total"),
        band_theta_only=_band_from_draws(pt, draws_th, gs["tuniq"], label="uncond theta-only"),
        band_link_only=_band_from_draws(pt, draws_ln, gs["tuniq"], label="uncond link-only"),
        # legacy-key link-only: NOT for reporting -- it exists so the caller can check this
        # rebuild against the stored CODMUN_IBGE conditional band.
        band_link_only_legacy=_band_from_draws(pt_leg, draws_ln_leg, gs_leg["tuniq"],
                                               label="uncond link-only [legacy key]"),
        per_draw_diag=pd.DataFrame(diag_rows),
        meta=dict(B=B, scheme=scheme, seed=seed, loss=loss, degree=degree,
                  n=len(df_ss), n_cl=n_cl, n_fail=n_fail,
                  foc_norm=dirif["foc_norm"], gamma_hat=dirif["gamma_hat"],
                  dir_diag=dirif["diag"], vmu=vmu, vsd=vsd,
                  theta=theta, idx=idx_expect, b_full=b_full,
                  market_key="mca_code (V_Main:182); legacy variant on CODMUN_IBGE"))
    if keep_draws:
        out["draws"] = dict(total=draws_tot, theta_only=draws_th, link_only=draws_ln)
    return out
