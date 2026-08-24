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
    only params_native (and the index). Used by the single-index estimators, which fit this
    logit purely to warm-start theta and never read its AMEs or SEs."""
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
    # optimum. This fit otherwise runs ONE start from zeros, and E3/E4 inherit whatever
    # direction it produces (fit_single_index never re-optimises theta), so a bad
    # basin here propagates silently to both reported single-index estimators.
    _args = (y_dm, X, Z, CF, entity_idx, link, ecounts, tinv, tcounts)

    if init is not None:
        starts = [("caller", np.asarray(init, float))]
        if starts[0][1].shape != (K + G,):
            raise ValueError(f"init has shape {starts[0][1].shape}, expected {(K + G,)}")
    else:
        # MULTISTART. A single start from zeros is thin for a non-convex M-estimator, and it
        # propagates: fit_single_index (E3/E4) never re-optimises theta, it inherits whatever
        # direction this fit produces. Scored on the full sample against alternative index
        # directions, the zeros-start logit direction comes last of four (164,913 against
        # 160,940 for the best), so one start leaves a measurably poor direction in place.
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


# ------------------------------------------------------------------------------
# SHAPE-CONSTRAINED LINK ESTIMATION
#
# phi is a share of inert depositors -- a probability -- and in a single-index model G must be
# a CDF: monotone and valued in [0,1] (V_Main.tex eq.(6)/:296 derives phi as Prob(.), :408 gives
# boundedness as the reason for leaving the linear form, :414 promises a "monotone cubic B-spline
# (sieve) link", and :507 divides by 1-phi so phi<1 strictly). Until 2026-08-06 neither property
# was imposed anywhere: `fit_single_index` ran an unconstrained OLS on raw monomials and `np.clip`
# was applied only when phi was reported. Measured on the trimmed spec-12 cells: 18-26% of rows
# pinned at phi=1, and 37-41% sitting on a DECREASING segment of the fitted cubic.
#
# WHY NOT JUST CONSTRAIN THE CUBIC. Measured, same date: it degenerates. A non-constant cubic is
# unbounded, so `0<=phi<=1` on an interval caps the attainable slope at O(1/width) (Markov
# brothers). The estimation index spans [-2.89,+12.00] SD in E5/robust, and over that range the
# constrained cubic collapses to an exact constant -- b=[0.9753,0,0,0], phi_t range 0.003pp
# against 1.79pp unconstrained. That is the FAMILY failing, not a tuning choice, and it happens
# in both robust cells.
#
# WHAT WE DO INSTEAD. Fit the link on a monotone I-spline ramp basis
# (`_ramp_design`), with beta >= 0 (monotone, floor at beta_1) AND sum(beta) <= 1 (ceiling: every
# ramp tends to 1 at the top of the knot hull). That is a valid CDF on the WHOLE real line --
# `_ramp_design` clips v into the knot hull, so G is flat outside it, which is what makes
# V_Main eq.(18)'s Pix-removed counterfactual well defined off the fitted support. It keeps
# 1.97-2.60pp of phi_t range and fits strictly better than the constrained cubic in all four
# cells. It is also, literally, the "monotone cubic B-spline sieve" the paper describes: a
# nonneg combination of ramps is a cubic B-spline with nondecreasing coefficients.
#
# `sum(beta) <= 1` is exactly the constraint BVLS cannot express (`lsq_linear` takes box bounds
# only), so the ceiling is imposed by the shape QP (`solve_ispline_qp`) rather than by a bounded
# least-squares solve.
# ------------------------------------------------------------------------------
SI_N_INTERIOR = int(os.environ.get("SLEEP_SI_KNOTS", "5"))   # interior knots of the ramp basis
SI_GRID_N = 200                                              # length of the stored link grid


def link_constrained():
    """True (default) when the link is estimated under its shape constraints.
    SLEEP_LINK_CONSTRAINED=0 selects the unconstrained cubic instead, which exists only to
    reproduce stored results bit-for-bit."""
    return os.environ.get("SLEEP_LINK_CONSTRAINED", "1") != "0"


def _pix_shifted_range(X, vs, theta, phi_params, vsd):
    """Range of the STANDARDISED index once the Pix indicator is forced to its low level.

    V_Main eq.(18) builds the no-Pix counterfactual as G(index - theta_pix * Pix), i.e. it
    evaluates the link OFF the fitted support. Knots are laid over the union of the observed and
    the shifted support so that counterfactual lands inside the fitted hull instead of on the
    flat extrapolation. Falls back to the observed range if there is no recognised Pix dummy."""
    names = [nm for nm in phi_params if nm != "nr_lagged_dep"]
    lo_v, hi_v = float(vs.min()), float(vs.max())
    if not names:
        return lo_v, hi_v
    sub = X[:, [phi_params.index(nm) for nm in names]]
    try:
        flag, lo, _hi = _dummy_spec(sub, len(names), names)
    except Exception:
        return lo_v, hi_v
    for j, nm in enumerate(names):
        if not flag[j] or "pix" not in str(nm).lower():
            continue
        th = float(theta[phi_params.index(nm)])
        z = vs + th * (lo[j] - sub[:, j]) / vsd
        lo_v = min(lo_v, float(z.min()))
        hi_v = max(hi_v, float(z.max()))
    return lo_v, hi_v


def _qp_violation(p, A, lb, ub):
    """Max constraint violation of p, 0 if feasible. Infinite bounds are ignored."""
    r = A @ np.asarray(p, float)
    hi = np.max(r - ub, initial=0.0, where=np.isfinite(ub))
    lo = np.max(lb - r, initial=0.0, where=np.isfinite(lb))
    return float(max(hi, lo, 0.0))


def _qp_gram(D, y):
    """Gram matrix in unit-norm columns, plus the objective normaliser.

    Both rescalings are load-bearing. The design mixes Z (lagged deposits, O(1e9)) with powers
    of a standardised index, O(1); and y is on the deposit scale, so ||Dp-y||^2 ~ 1e24 and the
    solver's gtol is unreachable -- it then terminates on failure and the caller falls back to
    its starting value, returning an unconstrained fit while reporting success. Solve in
    q = s*p on the normalised objective ||Ds q - y||^2 / y'y instead."""
    s = np.linalg.norm(D, axis=0)
    s[s <= 0] = 1.0
    Ds = D / s
    yty = float(y @ y)
    return Ds.T @ Ds, Ds.T @ y, s, (yty if yty > 0 else 1.0)


def _qp_solve_sub(G, c, s, A, lb, ub, q0, nrm=1.0):
    """Constrained QP on a working set, in rescaled coordinates. Returns the best FEASIBLE
    candidate -- ranking two solvers by objective alone is wrong, because an infeasible point
    always scores lower and would always win."""
    from scipy.optimize import LinearConstraint, minimize
    H = 2.0 * G
    f = lambda q: float(q @ (G @ q) - 2.0 * (c @ q)) / nrm
    jac = lambda q: (H @ q - 2.0 * c) / nrm
    lc = LinearConstraint(A / s, lb, ub)
    Hn = H / nrm
    cands = [minimize(f, q0, jac=jac, hess=lambda q: Hn, method="trust-constr",
                      constraints=[lc],
                      options=dict(maxiter=2000, gtol=1e-12, xtol=1e-14, verbose=0))]
    if cands[0].status not in (1, 2) or _qp_violation(cands[0].x / s, A, lb, ub) > 1e-7:
        cands.append(minimize(f, q0, jac=jac, method="SLSQP", constraints=[lc],
                              options=dict(maxiter=1000, ftol=1e-16)))
    scored = []
    for cd in cands:
        q = getattr(cd, "x", None)
        if q is None or not np.all(np.isfinite(q)):
            continue
        scored.append((_qp_violation(q / s, A, lb, ub) > 1e-7, f(q), q))
    if not scored:
        return q0
    scored.sort(key=lambda t: (t[0], t[1]))       # feasible first, then lowest SSR
    return scored[0][2]


def solve_shape_qp(D, y, A, lb, ub, p0, tol=1e-8, max_rounds=25, batch=6, strict=True):
    """min ||D p - y||^2 s.t. lb <= A p <= ub, by CONSTRAINT GENERATION on the Gram matrix.

    Handing the solver all 2*ngrid constraint rows costs ~220 s per solve, and that cost is
    independent of n, so it does not amortise inside a 400-999-draw bootstrap. With k = 4-10
    unknowns at most k constraints can be active at the optimum, so we solve on a small working
    set, check every row, add the worst violators and repeat -- typically 2-6 rounds. The relaxed
    optimum lower-bounds the constrained one, so stopping when it is feasible for the FULL set
    certifies optimality (verified against a full solve to 6e-10 relative, ~40-96x faster).

    Returns (p, info) with the active set, rounds, working-set size and final violation. `strict`
    raises rather than return an infeasible link -- returning one silently is the precise defect
    this machinery exists to remove."""
    G, c, s, nrm = _qp_gram(D, y)
    A = np.asarray(A, float); lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    q = np.asarray(p0, float) * s
    work = np.zeros(len(lb), dtype=bool)
    rnd = 0
    for rnd in range(max_rounds):
        q = (_qp_solve_sub(G, c, s, A[work], lb[work], ub[work], q, nrm=nrm) if work.any()
             else np.linalg.lstsq(G, c, rcond=None)[0])
        r = A @ (q / s)
        v = np.maximum(np.where(np.isfinite(ub), r - ub, -np.inf),
                       np.where(np.isfinite(lb), lb - r, -np.inf))
        if float(v.max()) <= tol:
            break
        cand = np.argsort(v)[::-1][:batch]
        cand = cand[v[cand] > tol]
        if work[cand].all():                      # nothing new to add
            break
        work[cand] = True
    p = q / s
    rr = A @ p
    act = ((np.isfinite(ub) & (rr >= ub - 1e-7)) | (np.isfinite(lb) & (rr <= lb + 1e-7)))
    viol = _qp_violation(p, A, lb, ub)
    info = dict(rounds=rnd + 1, n_work=int(work.sum()), violation=viol, active=act,
                n_active=int(act.sum()), ok=bool(viol <= max(tol, 1e-7)))
    if strict and not info["ok"]:
        raise RuntimeError(
            f"solve_shape_qp did not reach a feasible point: max violation {viol:.3e} after "
            f"{info['rounds']} rounds. Refusing to return an infeasible link.")
    return p, info


def proj_simplex_box(x, K, cap=1.0):
    """Euclidean projection of x[:K] onto {b >= 0, sum(b) <= cap}; x[K:] untouched.
    Clip at zero, and if the sum still exceeds cap project onto the simplex
    (Duchi, Shalev-Shwartz, Singer & Chandra 2008). Closed form, O(K log K)."""
    out = np.array(x, float)
    v = np.maximum(out[:K], 0.0)
    if v.sum() <= cap:
        out[:K] = v
        return out
    u = np.sort(out[:K])[::-1]
    css = np.cumsum(u) - cap
    idx = np.arange(1, K + 1)
    rho = idx[u - css / idx > 0][-1]
    out[:K] = np.maximum(out[:K] - css[rho - 1] / rho, 0.0)
    return out


def solve_ispline_qp(D, y, K, cap=1.0, iters=5000, tol=1e-12, warm=None):
    """min ||D p - y||^2 over {p[:K] >= 0, sum(p[:K]) <= cap}, p[K:] free. EXACT and
    start-independent.

    WHY NOT solve_shape_qp HERE. Constraint generation certifies optimality by "the relaxed
    solution is feasible for every constraint" -- which is only a proof if the relaxed problem
    was solved EXACTLY. scipy's trust-constr is not reliably exact on this design, and measured
    2026-08-06 it stopped at a feasible point with the wrong active set: same data, objective
    1.1e-05 higher than the true optimum and the fitted link 1.6e-03 away in phi. Four different
    starting points reached the true optimum; the estimator's own solve did not. A near-flat
    valley in a collinear ramp basis makes that failure quiet, not loud.

    This set is a SIMPLEX-BOX, whose Euclidean projection is closed form, so accelerated
    projected gradient (FISTA with adaptive restart) applies directly and converges to the
    unique minimiser of a convex quadratic from any start. The Gram matrix is (K+extra)^2, so
    20k iterations cost microseconds. The active set is then polished by an exact KKT solve on
    the free coordinates, which removes the last few ulps of first-order error.

    `warm` supplies a starting point, which is what makes this affordable INSIDE a bootstrap.
    Each draw perturbs the criterion slightly, so consecutive solutions share an active set
    almost always, and the active-set loop below can start from it directly. FISTA is then only
    needed cold. That matters a great deal at this conditioning: FISTA does not hit its early
    exit here and runs its full iteration budget, measured at ~490 ms per solve, i.e. ~33 min
    per spec of pure projection cost at B=999 over two bootstraps and two losses. Warm-started,
    the same solve is a handful of 10x10 linear systems. The loop keeps its drop AND release
    steps, so it converges to the same optimum from any starting face -- the warm start changes
    the path, never the answer, and a warm solve that fails its optimality check falls back to
    the cold one.
    """
    G = D.T @ D
    c = D.T @ y
    p = D.shape[1]
    f = lambda q: float(q @ (G @ q) - 2.0 * (c @ q))

    def _fista():
        L = float(np.linalg.eigvalsh(G)[-1]) or 1.0
        b_ = proj_simplex_box(np.zeros(p), K, cap)
        z, t = b_.copy(), 1.0
        for _ in range(iters):
            b_new = proj_simplex_box(z - (G @ z - c) / L, K, cap)
            if f(b_new) > f(b_):                  # adaptive restart on objective increase
                z, t = b_.copy(), 1.0
                b_new = proj_simplex_box(b_ - (G @ b_ - c) / L, K, cap)
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = b_new + ((t - 1.0) / t_new) * (b_new - b_)
            if np.max(np.abs(b_new - b_)) <= tol * max(1.0, np.max(np.abs(b_new))):
                return b_new
            b_, t = b_new, t_new
        return b_

    if warm is not None:
        b = proj_simplex_box(np.asarray(warm, float).copy(), K, cap)
    else:
        b = _fista()
    # --- exact polish: primal ACTIVE-SET loop on the face FISTA identified -------------------
    # FISTA gets the face approximately right but leaves inactive coordinates at ~1e-8 rather
    # than 0, so the face has to be both DROPPED into (zero out what is at the bound) and
    # RELEASED from (put back anything whose reduced cost says the objective still falls if it
    # moves off the bound). Only dropping is not enough: a genuinely tiny but nonzero
    # coefficient gets zeroed, the KKT system is solved on too small a face, and the answer sits
    # ~2e-07 above the optimum -- small, but it desynchronises the band's re-profile from the
    # estimator, which is what the fingerprint gate exists to detect.
    #
    # Reduced cost of a coordinate held at zero is  r_j = (G b - c)_j + mu, where mu is the
    # multiplier on sum(beta) = cap, pinned by stationarity on the free coordinates. r_j < 0
    # means releasing j lowers the objective, so j belongs in the face.
    def _polish(b0):
        """Primal active-set loop from the face implied by b0. Returns (b, converged).

        THREE optimality conditions, all required. Dropping a coordinate that goes negative and
        releasing one whose reduced cost is negative are the familiar pair; the third is the
        SIGN OF THE CEILING MULTIPLIER. With sum(beta) <= cap held as an equality, stationarity
        gives G b - c = -mu e, and a genuine minimum needs mu >= 0. If mu < 0 the ceiling is not
        really active and must be released. Without that test a warm start that happens to sit
        at the cap keeps it there and certifies a WRONG optimum -- measured 8.75e-02 away from
        the cold solve, which is why the warm and cold paths are checked against each other
        rather than assumed to agree."""
        bb = b0
        scale = max(1.0, float(np.max(np.abs(bb[:K]))))
        free = np.ones(p, bool)
        free[:K] = bb[:K] > 1e-9 * scale
        cap_active = float(np.sum(np.maximum(bb[:K], 0.0))) >= cap - 1e-9
        for _ in range(6 * K + 16):
            at_cap = cap_active
            if not free.any():
                return bb, True
            idx = np.flatnonzero(free)
            Gf, cf_ = G[np.ix_(idx, idx)], c[idx]
            e = (idx < K).astype(float)
            try:
                if at_cap and e.any():
                    KKT = np.block([[Gf, e[:, None]], [e[None, :], np.zeros((1, 1))]])
                    sol = np.linalg.solve(KKT, np.concatenate([cf_, [cap]]))
                    mu, sol = float(sol[-1]), sol[:len(idx)]
                else:
                    sol = np.linalg.solve(Gf, cf_)
                    mu = 0.0
            except np.linalg.LinAlgError:
                return bb, False
            cand = np.zeros(p)
            cand[idx] = sol
            neg = (idx < K) & (sol < -1e-12 * scale)
            if neg.any():                               # infeasible: drop the worst offender
                free[idx[np.argmin(np.where(neg, sol, np.inf))]] = False
                continue
            if cand[:K].sum() > cap + 1e-10:            # ceiling violated: activate it
                if at_cap:
                    return bb, False
                cap_active = True
                bb = proj_simplex_box(cand, K, cap)
                free[:K] = bb[:K] > 1e-9 * scale
                continue
            g = G @ cand - c
            gs_ = max(1.0, float(np.max(np.abs(g))))
            if at_cap and mu < -1e-11 * gs_:            # ceiling not really binding: release it
                cap_active = False
                bb = cand
                continue
            # feasible on this face -- can any bound-held coordinate be released?
            held = np.flatnonzero(~free[:K])
            if len(held):
                r_ = g[held] + mu
                if float(np.min(r_)) < -1e-11 * gs_:
                    free[int(held[np.argmin(r_)])] = True
                    continue
            # no drop, no release, feasible => KKT satisfied on this face: optimal
            return (cand if f(cand) <= f(bb) + 1e-12 * abs(f(bb)) else bb), True
        return bb, False

    b, ok = _polish(b)
    if not ok and warm is not None:
        # the warm face did not certify; pay for the cold solve rather than return a guess
        b, ok = _polish(_fista())
    r = b[:K]
    eps = 1e-9 * max(1.0, float(np.max(np.abs(r))))
    info = dict(n_active=int(np.sum(r <= eps)) + int(float(r.sum()) >= cap - 1e-10),
                n_zero=int(np.sum(r <= eps)), at_cap=bool(float(r.sum()) >= cap - 1e-10),
                sum_beta=float(r.sum()), violation=float(max(0.0, -r.min(),
                                                             r.sum() - cap)), ok=True)
    return b, info


def ispline_constraints(K, n_extra=0):
    """Constraint block for the monotone I-spline link: beta_j >= 0 (monotone, floor beta_1)
    and sum(beta) <= 1 (ceiling). n_extra = trailing design columns (the control function)
    that carry no constraint."""
    A = np.vstack([np.hstack([np.eye(K), np.zeros((K, n_extra))]),
                   np.hstack([np.ones((1, K)), np.zeros((1, n_extra))])])
    lb = np.concatenate([np.zeros(K), [-np.inf]])
    ub = np.concatenate([np.full(K, np.inf), [1.0]])
    return A, lb, ub


def _shape_cone_at(A, lb, ub, b_at, tol=1e-7):
    """Homogeneous tangent cone of {lb <= A p <= ub} at b_at: A_j d <= 0 on upper-active rows,
    A_j d >= 0 on lower-active ones. Returns (A_c, lb_c, ub_c, active_mask), with A_c None when
    nothing is active. The cone belongs to the POINT it is taken at, so a bootstrap draw whose
    own solution sits on a different face must take it at that solution.

    `unconditional_phi_t_band` keeps its own closure over the same algebra: its outputs are gated
    against stored bands, so it is not rewired here."""
    _r = np.asarray(A, float) @ np.asarray(b_at, float)
    _hi = np.isfinite(ub) & (_r >= np.asarray(ub, float) - tol)
    _lo = np.isfinite(lb) & (_r <= np.asarray(lb, float) + tol)
    _rows, _clb, _cub = [], [], []
    for j in np.flatnonzero(_hi):
        _rows.append(A[j]); _clb.append(-np.inf); _cub.append(0.0)
    for j in np.flatnonzero(_lo):
        _rows.append(A[j]); _clb.append(0.0); _cub.append(np.inf)
    if not _rows:
        return None, None, None, (_hi | _lo)
    return (np.vstack(_rows), np.asarray(_clb, float), np.asarray(_cub, float), (_hi | _lo))


def project_to_shape(p_lin, G, A, lb, ub, tol=1e-10, K=None, cap=1.0, warm=None):
    """G-norm projection of an unconstrained (linearised) draw onto the constraint set:
        argmin_{p: lb <= Ap <= ub} (p - p_lin)' G (p - p_lin).

    This is the bootstrap analogue of the constrained estimator. For a quadratic criterion the
    constrained fit on a perturbed sample IS the projection of the perturbed unconstrained fit,
    so a draw must be projected too -- perturbing the constrained coefficients directly would
    wander outside the parameter space and put mass on links that are not CDFs.

    NOTE (Andrews 2000): when constraints are ACTIVE at the estimate -- they are, in every
    measured cell -- this projection bootstrap is not consistent for the boundary case. It is
    reported alongside the tangent-cone variant, which is the valid one there; the two coincide
    when nothing binds."""
    Gs = np.asarray(G, float)
    L = np.linalg.cholesky(Gs + tol * np.eye(len(Gs)) * float(np.trace(Gs)) / len(Gs))
    Dp, yp = L.T, L.T @ np.asarray(p_lin, float)
    if K is not None:                      # simplex-box: use the exact solver
        return solve_ispline_qp(Dp, yp, K, cap=cap, warm=warm)
    return solve_shape_qp(Dp, yp, A, lb, ub, p_lin, strict=False)


def _si_ame_map(X, phi_params, theta, vs, vsd, P, knots, degree, n_basis,
                constrained, dummy_spec=None):
    """Average-marginal-effect map of the single-index link at ONE (direction, geometry) pair.

    Everything the AME depends on besides the link coefficients -- the direction `theta`, the
    standardised index `vs` and its scale `vsd`, the basis `P` and its `knots` -- is an argument,
    so a bootstrap draw that moves the direction rebuilds the WHOLE map rather than refreshing a
    subset of it. That matters because the AMEs are exactly invariant to theta -> a*theta (a>0)
    and to a shift of the constant coefficient: a map that rescaled `ths` while reusing the
    estimate's slope weights would produce distinct, plausible and wrong numbers.

    `dummy_spec` (the `_dummy_spec` triple) depends only on X's columns and their names, both
    fixed across draws, so it is passed in to hoist it out of a draw loop; None recomputes it.

    Returns dict(ame_fn, mean_slope, phi_at, names, ths, slope_w, dcols, dummy_spec).
    """
    # ---- link evaluation and its derivative, in the basis actually being used ----------------
    # Constrained: G = sum_j beta_j R_j = sum_k c_k B_k with c = cumsum(beta), so the derivative
    # is the B-spline's -- analytic, and no clip is needed anywhere because G is a CDF by
    # construction. Unconstrained: the raw cubic and its polynomial derivative, clipped on use.
    if constrained:
        from scipy.interpolate import BSpline

        _vs_in = np.clip(vs, knots[0], knots[-1])

        def _phi_at(bfull, zz=None):
            """G evaluated at standardised index zz (default: the sample)."""
            if zz is None:
                return P @ np.asarray(bfull, float)[:n_basis]
            return _ramp_design(zz, knots, degree) @ np.asarray(bfull, float)[:n_basis]

        # PRECOMPUTED AVERAGE DERIVATIVE. G' = sum_k c_k B'_k with c = cumsum(beta), so
        #   mean_i G'(v_i) = m . c = (L' m) . beta,   m_k = mean_i B'_k(v_i),  L = cumsum matrix,
        # i.e. the average derivative is a FIXED linear functional of beta. Evaluate the K basis
        # derivatives once and the AME bootstrap costs a dot product per draw instead of building
        # a spline and evaluating it on ~487k rows B times. This is exact, and it is available
        # only because the constrained link needs no clip -- a clip would make the row-average
        # nonlinear in beta and force the full per-draw evaluation.
        _m = np.array([float(np.mean(BSpline(knots, np.eye(n_basis)[k], degree,
                                             extrapolate=True).derivative()(_vs_in)))
                       for k in range(n_basis)])
        _slope_w = np.cumsum(_m[::-1])[::-1] / vsd        # (L' m) / vsd

        def _mean_slope(bfull):
            return float(_slope_w @ np.asarray(bfull, float)[:n_basis])
    else:
        _slope_w = None
        sv_pows = np.column_stack([vs ** (d - 1) for d in range(1, degree + 1)]) \
            if degree >= 1 else np.empty((len(vs), 0))
        dcoef = np.arange(1, degree + 1, dtype=float)

        def _phi_at(bfull, zz=None):
            bb = np.asarray(bfull, float)[:degree + 1]
            zz = vs if zz is None else zz
            return np.column_stack([zz ** d for d in range(degree + 1)]) @ bb

        def _mean_slope(bfull):
            bb = np.asarray(bfull, float)[1:degree + 1]
            return float(np.mean(sv_pows @ (dcoef * bb))) / vsd if degree >= 1 else 0.0

    names = [nm for nm in phi_params if nm != "nr_lagged_dep"]
    ths = {nm: th for nm, th in zip(phi_params, theta) if nm != "nr_lagged_dep"}

    # 0/1 DUMMIES get the DISCRETE-DIFFERENCE AME, not the continuous average
    # derivative: the marginal effect of a binary regressor is
    # E[G(idx | x=1) - G(idx | x=0)] on the structural link, which is bounded to
    # [-1,1]. Treating a dummy as continuous (theta_k * mean_slope) is the wrong
    # estimand and explodes when the inherited (unnormalised) logit direction gives a
    # weakly-identified dummy a huge coefficient (e.g. pix_exists, near-collinear with
    # the quarter FE -> theta_pix ~ 300 -> AME ~ 6.9). The flip terms below depend only
    # on (vs, theta_k, vsd, the dummy column) -- all fixed across the link bootstrap --
    # so we precompute their basis expansions once.
    # Levels come from _dummy_spec (registry-driven), so a CENTRED dummy with values
    # {-p_bar, 1-p_bar} is still recognised; a literal {0,1} test would not see it.
    _dsub = X[:, [phi_params.index(nm) for nm in names]]
    if dummy_spec is None:
        dummy_spec = _dummy_spec(_dsub, len(names), names)
    _dflag, _dlo, _dhi = dummy_spec

    def _basis_at(zz):
        if constrained:
            return _ramp_design(zz, knots, degree)
        return np.column_stack([zz ** dd for dd in range(degree + 1)])

    dcols = {}
    for j, nm in enumerate(names):
        if not _dflag[j]:
            continue
        coln = _dsub[:, j]
        z1 = vs + ths[nm] * (_dhi[j] - coln) / vsd     # standardised index at the HIGH level
        z0 = vs + ths[nm] * (_dlo[j] - coln) / vsd     # standardised index at the LOW level
        B1, B0 = _basis_at(z1), _basis_at(z0)
        # Constrained: E[G(z1) - G(z0)] = (mean B1 - mean B0) . beta, a fixed linear functional,
        # so collapse the two n x K blocks to one K-vector now and the AME costs a dot product
        # per bootstrap draw. Unconstrained: the clip sits between the basis and the average, so
        # the full blocks must be kept and re-evaluated per draw.
        dcols[nm] = ((B1.mean(axis=0) - B0.mean(axis=0)) if constrained else (B1, B0))

    # CLIP-AWARENESS IS NOW VACUOUS, deliberately. Under the constrained link G is already in
    # [0,1] everywhere, so the discrete difference needs no clip and the continuous average
    # derivative needs no pinned-row indicator -- the two AME branches finally measure the same
    # object. Without the constraints they do not: the dummy branch clips and the continuous
    # branch does not, which overstates the continuous AMEs by 2.6-4.7x in the robust cells.
    # That gap is a property of the unconstrained fit, not of the reporting.
    def _ame_fn(bfull):
        ms = _mean_slope(bfull)
        bb = np.asarray(bfull, float)[:n_basis]
        out = {}
        for nm in names:
            if nm in dcols:                              # discrete difference (bounded)
                if constrained:
                    out[nm] = float(dcols[nm] @ bb)
                else:
                    Pz1, Pz0 = dcols[nm]
                    out[nm] = float(np.mean(np.clip(Pz1 @ bb, 0.0, 1.0)
                                            - np.clip(Pz0 @ bb, 0.0, 1.0)))
            else:                                        # continuous average derivative
                out[nm] = ths[nm] * ms
        return out

    return dict(ame_fn=_ame_fn, mean_slope=_mean_slope, phi_at=_phi_at, names=names,
                ths=ths, slope_w=_slope_w, dcols=dcols, dummy_spec=dummy_spec)


def fit_single_index(df, state_cols, has_cf, logit_res, degree=3, phi_band=False,
                     fe_time_col=None):
    """Monotone single index in the logit-direction index (approaches III/IV, E3/E4).

    The direction theta is INHERITED from the logit and never re-optimised here; only the link
    G is estimated, by least squares on the FE-demeaned multiplicative design.

    Two link families, selected by SLEEP_LINK_CONSTRAINED (see link_constrained):

      constrained (default) -- monotone I-spline ramps with beta >= 0 and sum(beta) <= 1, i.e. a
        genuine CDF: monotone, in [0,1], on the whole real line. Stored as si_vgrid/si_ggrid in
        the NATIVE index frame plus si_beta/si_knots, with link="index_sieve".
      unconstrained cubic -- raw monomials stored as si_b with link="index", bounded to [0,1]
        only by a clip applied when phi is reported. Available for reproducing stored results;
        it does not deliver a valid share function (measured on the trimmed spec-12 cells:
        18-26% of rows at phi=1, 37-41% on a decreasing segment).

    Returns NonLinearResults carrying average-derivative AMEs (params; tables), the index
    direction (params_native), and the link. fe_time_col adds a second additive FE (two-way
    entity+time concentration) -- note callers pass it for BOTH E3 and E4."""
    constrained = link_constrained()
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
    if constrained:
        # Knots span the observed index AND the Pix-removed index, so V_Main eq.(18)'s
        # counterfactual is evaluated inside the fitted hull rather than on the flat
        # extrapolation. (Outside the hull G is still a valid CDF -- just constant.)
        _cf_lo, _cf_hi = _pix_shifted_range(X, vs, theta, phi_params, vsd)
        _knot_src = np.concatenate([vs, np.array([_cf_lo, _cf_hi])])
        _, si_knots = _bspline_design(_knot_src, n_interior=SI_N_INTERIOR, degree=degree)
        P = _ramp_design(vs, si_knots, degree)          # n x K monotone ramps
    else:
        si_knots = None
        P = np.column_stack([vs ** d for d in range(degree + 1)])
    n_basis = P.shape[1]
    R = P * Z[:, None]
    cols_p = [f"_p{d}" for d in range(n_basis)]
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
    b_unc = np.asarray(res.params, float)
    qp_A = qp_lb = qp_ub = None
    qp_info = None
    if constrained:
        # beta >= 0 and sum(beta) <= 1 on the ramp coefficients; the CF column is unconstrained.
        qp_A, qp_lb, qp_ub = ispline_constraints(n_basis, n_extra=len(design) - n_basis)
        b_full, qp_info = solve_ispline_qp(Xdm_arr, y_dm, n_basis)
        resid = y_dm - Xdm_arr @ b_full
    else:
        b_full = b_unc
        resid = np.asarray(res.resid, float)
    b = b_full[:n_basis]
    G_star, G_nominal = _G_star(cl)
    cl_u, cl_inv = np.unique(cl.values, return_inverse=True)
    n_cl = len(cl_u)

    # The AME map is built by _si_ame_map, which takes every theta-dependent object as an
    # argument; the two-stage AME bootstrap calls the SAME factory at each drawn direction.
    _amemap = _si_ame_map(X, phi_params, theta, vs, vsd, P, si_knots, degree, n_basis,
                          constrained)
    _ame_fn = _amemap["ame_fn"]
    _mean_slope = _amemap["mean_slope"]
    _phi_at = _amemap["phi_at"]

    mean_slope = _mean_slope(b_full)
    ame = _ame_fn(b_full)
    # Inference: score/multiplier wild cluster bootstrap on the sieve OLS
    # (replaces the earlier conditional clustered-OLS delta-method SE).
    # Under the constrained link every draw is PROJECTED back onto the constraint set, so the
    # bootstrap distribution lives on valid CDFs -- the draw analogue of the estimator itself.
    _bread = np.linalg.pinv(Xdm_arr.T @ Xdm_arr)
    _proj = None
    if constrained:
        _XtX = Xdm_arr.T @ Xdm_arr

        # Bootstrap draws perturb the criterion only slightly, so each projection starts from
        # the previous one's solution: the active set is almost always the same and the solve
        # collapses to a few tiny linear systems. Cold, it is ~490 ms per draw at this
        # conditioning, which is ~33 min per spec across two bootstraps and two losses.
        _warm = [b_full.copy()]

        def _proj(b_lin):
            out = project_to_shape(b_lin, _XtX, qp_A, qp_lb, qp_ub, K=n_basis,
                                   warm=_warm[0])[0]
            _warm[0] = out
            return out

    bse, pv = ols_sieve_wild_bootstrap(Xdm_arr, resid, cl_inv, n_cl, b_full, _ame_fn, ame,
                                       project=_proj)
    ps = pd.Series(ame); bs = pd.Series(bse); pvs = pd.Series(pv)

    # Quarter-clustered WCB for the national rows (pix_exists, risk_free_qoq_lag). Same
    # bootstrap, same _ame_fn, resampling the QUARTER instead of the conglomerate -- which is
    # the only dimension along which a national regressor actually varies. Stored separately so
    # `bs` is untouched and the exporters choose per row. Driscoll-Kraay is not produced here:
    # it is analytic and would need a
    # delta-method push through _ame_fn, which the bootstrap already does exactly.
    _nat_time = _nat_pv = None
    if fe_time_col is not None and fe_time_col in df_ss.columns:
        try:
            _per = df_ss[fe_time_col].astype(str).values
            _uniq, _t_inv = np.unique(_per, return_inverse=True)
            _bt, _pt = ols_sieve_wild_bootstrap(Xdm_arr, resid, _t_inv, len(_uniq),
                                                b_full, _ame_fn, ame, project=_proj)
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
    rss = float(np.sum(resid ** 2))
    rsq = 1 - rss / tss if tss > 0 else getattr(res, "rsquared", np.nan)
    B_used, scheme_used = boot_cfg()
    if constrained:
        _ssr_unc = float(np.sum((y_dm - Xdm_arr @ b_unc) ** 2))
        _phi_pt = _phi_at(b_full)
        # SATURATED SHARE. A monotone spline whose ceiling sum(beta)=1 is ACTIVE reaches exactly
        # 1 once the highest basis function with beta_j>0 finishes rising, and is flat at 1 from
        # there on -- so a whole region of the index can sit at phi=1, unlike a cubic which can
        # only touch a bound at isolated points. That is a real property of the fit, and it
        # matters downstream because demand prep forms (1-phi) through PHI_CAP=0.99
        # (estimation_demand_link_common:92,320): rows at phi=1 have their phi replaced by the
        # cap, so their heterogeneity is discarded. Reported per cell so the share is visible
        # rather than rediscovered.
        _sat = 100.0 * float((_phi_pt >= 1.0 - 1e-9).mean())
        print(f"  [SingleIndex] MONOTONE I-spline link K={n_basis} deg={degree} | "
              f"phi in [{_phi_pt.min():.4f},{_phi_pt.max():.4f}] sum(beta)={b.sum():.4f} "
              f"(unconstrained {b_unc[:n_basis].sum():+.4g}) | {qp_info['n_active']} active "
              f"constraint(s) | phi=1 on {_sat:.2f}% of rows | SSR x{rss/_ssr_unc:.6f} | "
              f"mean_slope={mean_slope:.4g} | wild boot B={B_used} ({scheme_used})")
        if _sat > 5.0:
            print(f"  [SingleIndex] NOTE {_sat:.1f}% of rows are at phi=1 (the link saturates "
                  f"inside the data). Downstream those rows are capped at PHI_CAP before "
                  f"forming 1-phi, so their phi heterogeneity does not reach the shares.")
    else:
        print(f"  [SingleIndex] unconstrained cubic link deg={degree} | "
              f"mean_slope={mean_slope:.4g} | wild boot B={B_used} ({scheme_used})")
    res_obj = NonLinearResults(ps, bs, ps / bs.replace(0, np.nan), pvs, G_star,
                               params_native=pd.Series(theta, index=phi_params),
                               nobs=len(df_ss), rsquared=rsq, G_nominal=G_nominal,
                               cov_ame=None, si_b=b, si_vmu=vmu, si_vsd=vsd,
                               link=("index_sieve" if constrained else "index"))
    res_obj.si_constrained = bool(constrained)
    if constrained:
        # The link is stored as a grid in the FULL native index frame, constant included: this
        # index HAS a constant (theta over `nr_lagged_dep` -> a column of ones), so the grid is
        # read against the whole index. That is what the "index_sieve" tag tells every consumer.
        _gv = np.linspace(si_knots[0], si_knots[-1], SI_GRID_N)
        res_obj.si_vgrid = _gv * vsd + vmu                     # native index units
        res_obj.si_ggrid = np.clip(_ramp_design(_gv, si_knots, degree) @ b, 0.0, 1.0)
        res_obj.si_beta = b
        res_obj.si_knots = np.asarray(si_knots, float)
        res_obj.si_degree = degree
        res_obj.si_b_unc = b_unc[:n_basis]
        res_obj.si_qp = dict(n_active=qp_info["n_active"], n_zero=qp_info["n_zero"],
                             at_cap=qp_info["at_cap"], violation=qp_info["violation"],
                             sum_beta=float(b.sum()), sum_beta_unc=float(b_unc[:n_basis].sum()),
                             ssr_ratio=rss / _ssr_unc, n_basis=int(n_basis),
                             saturated_pct=_sat, phi_lo=float(_phi_pt.min()),
                             phi_hi=float(_phi_pt.max()))
    # Quarter-clustered SEs for the national rows, computed above. Kept alongside `bse` so the
    # exporters can choose PER ROW without a second estimation pass.
    if _nat_time is not None:
        res_obj.bse_time = _nat_time
        res_obj.pvalues_time = _nat_pv
    # National phi_t band from the link wild cluster bootstrap; the logit index direction is
    # held fixed, so this band reflects LINK uncertainty only (the unconditional band in
    # unconditional_phi_t_band adds the direction channel, which is the larger of the two).
    if phi_band:
        score = Xdm_arr * resid[:, None]
        IF_cl = _cluster_if(score, _bread, cl_inv, n_cl)
        gs = _phi_t_group_struct(df_ss)

        def _phi_of_b(bfull):
            return np.clip(_phi_at(bfull), 0.0, 1.0)

        _B, _scheme = boot_cfg()

        def _draw(rng_):
            w = _wild_weights(n_cl, _scheme, rng_)
            b_lin = b_full + w @ IF_cl
            return _phi_of_b(_proj(b_lin) if _proj is not None else b_lin)

        res_obj.phi_t_boot = _phi_t_band(gs, _phi_of_b(b_full), _draw,
                                         min(_B, 400), _scheme, np.random.default_rng(20240624))
    return res_obj


# ==============================================================================
# Canonical phi construction (used by BOTH market_panel_phis and demand prep).
# Always native index + link; AMEs never enter here.
# ==============================================================================
def phi_from_native(df: pd.DataFrame, res, link: str) -> np.ndarray:
    """phi_mt at each row of df from the native index coefficients + the link.
    link in {'logit','probit','uniform','index','index_sieve'}.
    Returns phi in [0,1]."""
    native = res.params_native
    phi_params = [p for p in native.index if not str(p).startswith("v_hat")]
    X = _build_phi_X(df, phi_params)
    index = X @ native[phi_params].values.astype(float)
    if link == "index_sieve":
        # E3/E4 shape-constrained link: monotone I-spline stored as a grid over the FULL native
        # index, constant INCLUDED. Outside the grid the link is flat, which is the correct CDF
        # extension and is why eq.(18)'s Pix-removed index is well defined; the clip is a no-op
        # kept only as a numerical guard.
        return np.clip(np.interp(index, res.si_vgrid, res.si_ggrid), 0.0, 1.0)
    if link == "index":
        b = np.asarray(res.si_b)
        vs = (index - res.si_vmu) / (res.si_vsd if res.si_vsd else 1.0)
        g = np.zeros(len(vs))
        for d in range(len(b)):
            g += b[d] * vs ** d
        return np.clip(g, 0.0, 1.0)
    return np.clip(link_cdf(index, link), 0.0, 1.0)


# ==============================================================================
# INDEX AND LINK BUILDING BLOCKS
# ==============================================================================
# FE concentration plus the spline bases the single-index link is fitted on. The
# monotone link is a nonneg combination of I-spline ramps built from a cubic
# B-spline basis with ordered coefficients (Ramsay 1988; Newey 1997), i.e. a cubic
# B-spline with nondecreasing coefficients, which is what makes it a valid CDF shape.
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
# E1-E4, the BBL policy function, utils/se_national.py and the whole phi-separation
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
#   2. estimation_sleep_common.py --est 3, --est 4  -> E3/E4 pickles
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
                           scheme="rademacher", rng=None, mode=None, project=None):
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
    consumption -- so switching mode cannot move a point estimate or a draw.

    `project`, if given, maps a linearised draw back into the parameter space before the AME is
    evaluated. It is how the SHAPE-CONSTRAINED link (see solve_shape_qp) is bootstrapped: for a
    quadratic criterion the constrained fit on a perturbed sample is the projection of the
    perturbed unconstrained fit, so an unprojected draw would put bootstrap mass on links that
    are not CDFs. It does not consume RNG, so weights and draw order are unchanged.
    CAVEAT: the studentisation below linearises the AME at the ESTIMATE and is left unprojected.
    With constraints active that linearisation is only directionally valid (Andrews 2000), which
    is exactly why unconditional_phi_t_band also reports a tangent-cone band."""
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
        if project is not None:
            theta_b = project(theta_b)
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
                             B=None, scheme=None, seed=0, project=None):
    """Score/multiplier wild cluster bootstrap for the E3/E4 profiled sieve OLS
    (link coefficients conditional on the logit index direction). IF_i for OLS
    is (X'X)^{-1} x_i u_i; perturb the cluster sums, recompute the AME via
    ame_fn(b)->dict, return (bse dict, pvals dict). `project` maps each draw back onto the
    shape constraints when the link is constrained -- see cluster_wild_bootstrap."""
    if B is None or scheme is None:
        _B, _s = boot_cfg(); B = B if B is not None else _B; scheme = scheme or _s
    Xdm = np.asarray(Xdm, float)
    bread = np.linalg.pinv(Xdm.T @ Xdm)
    score = Xdm * np.asarray(resid, float)[:, None]
    IF_cl = _cluster_if(score, bread, cl_inv, n_cl)
    rng = np.random.default_rng(seed)
    return cluster_wild_bootstrap(b_full, IF_cl, ame_fn, ame_hat,
                                  B=B, scheme=scheme, rng=rng, project=project)


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
                   written down in V_Main.tex:311 ("phi_t = sum_m phi_mt*M_mt / sum_m M_mt
                   over MARKETS m") and in this module's header, and it is the convention
                   every reported phi_t series uses (calculate_phis, calculate_pooled_phis,
                   _calculate_phis and the bands all aggregate this way).
    weight="sum"   each market enters weighted by population x THE NUMBER OF BANK ROWS in
                   the cell, since pop_total is constant within a (quarter, market) cell and
                   summing it multiplies by the bank count. Nothing in the model corresponds
                   to that product; the option is kept only to reproduce a series built
                   under it.

    The choice moves the LEVEL of national sleepiness by 1.90 pp (0.9720 vs 0.9531 on est4
    spec 12) from identical phi_mt, against a 3.76 pp movement being interpreted; the RANGE
    is nearly unchanged (3.76 vs 3.50 pp). The market key matters far less (CODMUN_IBGE vs
    mca_code: 0.21 pp). The parameter is explicit so a band is always built on EXACTLY the
    convention its own point series uses: a mismatch here puts a band 1.5-2.2 pp away from
    the phi_t it is drawn around.
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

    # phi_t is LINEAR in beta, so its per-cluster influence values are exact:
    #   A[t, j] = national aggregation of Z[:, j] in quarter t   (aggregation is linear)
    #   IFphi[g, t] = sum_j IF_cl[g, pos_j] * A[t, j]
    # and every draw is pt + w @ IFphi -- identical (to fp roundoff) to aggregating the
    # perturbed per-row phi, but exposing exactly what the 2026-08-06 additions need:
    #   se_t   = sqrt(sum_g IFphi^2)                        cluster-robust SE of phi_t
    #   a_t    = sum_g IFphi^3 / (6 se_t^3)                 Efron acceleration, closed form
    #   t*_bt  = (draw - pt)/sqrt(sum_g w_bg^2 IFphi^2)     score-studentized bootstrap-t
    k = len(pos)
    A = np.column_stack([_agg_phi_t(Z[:, j], gs) for j in range(k)])       # nt x k
    IFphi = IF_cl[:, pos] @ A.T                                            # n_cl x nt
    pt = A @ beta[pos]
    se = np.sqrt(np.maximum(np.sum(IFphi ** 2, axis=0), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        accel = np.where(se > 0, np.sum(IFphi ** 3, axis=0) / (6.0 * se ** 3), 0.0)

    rng = np.random.default_rng(seed)
    nt = gs["nt"]
    draws = np.empty((B, nt))
    tstats = np.empty((B, nt))
    IFphi2 = IFphi ** 2
    for b in range(B):
        w = _wild_weights(n_cl, scheme, rng)
        draws[b] = pt + w @ IFphi
        vb = (w ** 2) @ IFphi2
        tstats[b] = (draws[b] - pt) / np.sqrt(np.maximum(vb, 1e-300))

    return _band_from_draws(pt, draws, gs["tuniq"], label="linear phi_t band",
                            accel=accel, tstats=tstats, se=se)


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
        # back on, unlike E3/E4. Selecting a fixed lo/hi pair silently dropped them.
        ren = {"time_id": time_key, "phi_t": "_band_pt",
               "lo": f"phi_t_lo_{safe_key}", "hi": f"phi_t_hi_{safe_key}",
               "lo_bc": f"phi_t_lobc_{safe_key}", "hi_bc": f"phi_t_hibc_{safe_key}",
               "p_below": f"phi_t_pbelow_{safe_key}",
               # 2026-08-06: BCa (closed-form acceleration) and score-studentized bootstrap-t
               "lo_bca": f"phi_t_lobca_{safe_key}", "hi_bca": f"phi_t_hibca_{safe_key}",
               "lo_bt": f"phi_t_lobt_{safe_key}", "hi_bt": f"phi_t_hibt_{safe_key}",
               "se_score": f"phi_t_se_{safe_key}"}
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

    MARKET = MCA. V_Main.tex:182 defines the local market as the Minimal Comparable Area --
    "B firms compete in local markets - defined here as Minimal Comparable Areas (MCAs)" --
    and :303/:406 sum over m in that market set. Keying on CODMUN_IBGE (municipality) instead
    splits MCAs that exist precisely to keep territorial units comparable across boundary
    changes, and moves national phi_t by ~0.21 pp. Pass market_key explicitly to build a band
    on the CODMUN_IBGE key (the unconditional-band correctness gate does this)."""
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
    interval can EXCLUDE its own point estimate whenever the map from parameters to phi is
    nonlinear enough to displace the draw cloud off the estimate. An estimator that renormalises
    theta onto the unit sphere, interpolates through a kinked link grid and clips to [0,1] stacks
    three such nonlinearities, and with a nearly flat fitted link whose point sits ON its floor
    the draws come out one-sided; measured at 26 of 35 quarters. The E3/E4 path cannot do this:
    its draw is `vpow @ (b + w.IF)`, linear in the perturbed coefficients up to the clip, so the
    draws are mechanically centred (0 violations in all 32 cells).

    `lo_bc`/`hi_bc` -- Efron's BIAS-CORRECTED percentile interval (Efron 1987), which is the
    repair: measure the median bias of the draw cloud as z0 = Phi^-1(P[draw < estimate]) and read
    the interval off the shifted quantile levels Phi(2*z0 +/- z_{alpha/2}). z0 = 0 (a centred
    cloud) reproduces the raw interval exactly, so this is a strict generalisation and the
    linear/single-index paths are unaffected.

    NOT done here: recentring the draws on the point estimate. That would force containment
    everywhere and destroy the very signal a displaced draw cloud carries about a degenerate link.

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
    band = _band_from_draws(pt, draws, gs["tuniq"], alpha=alpha)
    # The raw draw matrix rides along in .attrs (pickled with the frame, ~B*nt*8 bytes) so the
    # unconditional band's reproduction gate can compare DRAWS rather than interpolated band
    # edges: an empirical quantile moves by up to one local order-statistic gap under
    # sub-tolerance draw perturbations, so an edge gate can fire at a knife edge while every
    # draw reproduces. Bands stored before this attribute existed carry no draws; the gate
    # falls back to edges with a one-gap tolerance there.
    band.attrs["draws"] = draws
    return band


def _band_from_draws(pt, draws, tuniq, alpha=0.05, label="phi_t band",
                     accel=None, tstats=None, se=None):
    """Assemble the band frame from a precomputed (B x nt) national draw matrix.

    Split out of _phi_t_band (2026-08-05) so the unconditional band -- whose draw loop must run
    once and feed THREE variants (total / direction-only / link-only) from the same expensive
    re-profiled fits -- can reuse the identical raw-percentile + Efron-BC + p_below assembly
    instead of re-implementing it. Same math, same warnings, same column layout.

    2026-08-06 additions (both optional; existing columns are byte-unchanged when omitted):
      accel   (nt,) Efron acceleration constants -> lo_bca/hi_bca columns. Computed by the
              caller in closed form from the per-cluster influence values of phi_t
              (a_t = sum IF^3 / (6 (sum IF^2)^1.5), Efron 1987 eq. 7.3) -- no jackknife loop.
      tstats  (B x nt) studentized draws t*_bt = (phi*_bt - phi_t)/se*_bt, with se
              (nt,) the point cluster-robust SE -> lo_bt/hi_bt bootstrap-t columns,
              read as phi_t - q_{1-a/2}(t*) * se  /  phi_t - q_{a/2}(t*) * se
              (equal-tail percentile-t; the refinement MacKinnon-Nielsen-Webb recommend)."""
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

    if accel is not None:
        # BCa: percentile levels Phi(z0 + (z0+z_a)/(1 - a (z0+z_a))). a=0 reduces to BC.
        a = np.asarray(accel, float)
        lo_bca = np.empty(nt)
        hi_bca = np.empty(nt)
        for t in range(nt):
            zl, zh = z0[t] + za_lo, z0[t] + za_hi
            al = 100.0 * _norm.cdf(z0[t] + zl / max(1e-12, 1.0 - a[t] * zl))
            ah = 100.0 * _norm.cdf(z0[t] + zh / max(1e-12, 1.0 - a[t] * zh))
            lo_bca[t] = np.percentile(draws[:, t], np.clip(al, 0.0, 100.0))
            hi_bca[t] = np.percentile(draws[:, t], np.clip(ah, 0.0, 100.0))
        out["lo_bca"] = lo_bca
        out["hi_bca"] = hi_bca

    if tstats is not None and se is not None:
        se = np.asarray(se, float)
        q_lo = np.percentile(tstats, 100 * alpha / 2, axis=0)
        q_hi = np.percentile(tstats, 100 * (1 - alpha / 2), axis=0)
        out["lo_bt"] = pt - q_hi * se        # equal-tail bootstrap-t
        out["hi_bt"] = pt - q_lo * se
        out["se_score"] = se
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
# UNCONDITIONAL national phi_t band for the single-index estimators (E3/E4)
# ==============================================================================
# The stored bands condition on the estimated DIRECTION theta-hat: they perturb only the
# sieve-link coefficients b through vpow built once at theta-hat. But theta-hat is the object
# the 2026-08 multiplicity diagnostics show is weakly identified, and conditioning on a weakly
# identified quantity can only understate uncertainty. The two functions below produce the
# joint (direction + link) band without refitting the production estimator:
#
#   nlls_direction_if      cluster influence functions of the NLLS direction, reconstructed at
#                          the STORED solution (they were never persisted: the E3/E4 pipeline
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
                      fe_time_col=None, fd_check=True,
                      cluster_col="CodConglomeradoPrudencial", return_parts=False):
    """Cluster IFs of the NLLS direction at a stored solution. Returns a dict:
    IF_cl (n_cl x K, theta rows only), gamma_hat, foc_norm, cl_inv, n_cl, diag.

    `return_parts` adds the raw ingredients a two-stage bootstrap needs (see
    twostage_ame_boot): the per-row `score_rows` it aggregates its perturbation from, the
    full-width `IF_rows`, the `bread`, `psi_hat`, the `foc` vector whose norm is `foc_norm`,
    the cluster label vector and row index for alignment gates, and `refit` -- plain arrays
    only, so a spawned worker can rebuild the demeaner without re-reading the frame.

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

    cl = df_ss[cluster_col].astype(str)
    cl_u, cl_inv = np.unique(cl.values, return_inverse=True)
    n_cl = len(cl_u)
    IF_full = _cluster_if(score, bread, cl_inv, n_cl)       # n_cl x (K+G)

    out = dict(IF_cl=IF_full[:, :K], gamma_hat=gamma, foc_norm=foc_norm,
               cl_inv=cl_inv, n_cl=n_cl, theta=theta, idx=idx,
               diag=dict(jac_fd_relerr=worst, n=len(df_ss), K=K, G=G,
                         clamped_rows=int(np.sum(w2 <= 0.0))))
    if return_parts:
        out.update(score_rows=score, IF_rows=score @ bread.T, bread=bread, psi_hat=psi,
                   foc=foc, cl_labels=cl.values, row_index=df_ss.index,
                   refit=dict(y_dm=y_dm, X=X, Z=Z, CF=CF, entity_idx=entity_idx,
                              ecounts=ecounts, tinv=tinv, tcounts=tcounts, CF_dm=CF_dm,
                              link=link, loss=loss, K=K, G=G))
    return out


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

    # Which link does the STORED fit carry? Read it off the result, never off the environment:
    # the band has to rebuild the estimator that produced this pickle, and a run with a
    # different SLEEP_LINK_CONSTRAINED must fail the fingerprint gate rather than quietly
    # rebuild a different estimator.
    constrained = (getattr(si_res, "si_constrained", False)
                   or getattr(si_res, "link", None) == "index_sieve")

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
    CF_raw = (df_ss["v_hat_x_lagged_dep"].values.astype(float) if has_cf else None)

    # --- basis, per link family ------------------------------------------------------------
    # The KNOTS are part of the estimator (quantiles of the index, extended to cover the
    # Pix-removed support), so an exact re-profile at a drawn theta_b re-places them: the draw
    # must trace the estimator as a function of the sample, not a frozen basis. phi is only ever
    # evaluated through the matching (basis, coefficients) pair, so this is well defined.
    def _knots_at(vs_loc):
        lo, hi = _pix_shifted_range(X, vs_loc, theta, idx_expect, vsd)
        _, t = _bspline_design(np.concatenate([vs_loc, np.array([lo, hi])]),
                               n_interior=SI_N_INTERIOR, degree=degree)
        return t

    if constrained:
        knots0 = np.asarray(si_res.si_knots, float)
        n_basis = len(knots0) - degree - 1
        _basis = lambda vs_loc, kn: _ramp_design(vs_loc, kn, degree)
        qp_A, qp_lb, qp_ub = ispline_constraints(n_basis, n_extra=1 if has_cf else 0)
        b_stored = np.asarray(si_res.si_beta, float)
    else:
        knots0 = None
        n_basis = degree + 1
        _basis = lambda vs_loc, kn: np.column_stack([vs_loc ** d for d in range(degree + 1)])
        qp_A = qp_lb = qp_ub = None
        b_stored = np.asarray(si_res.si_b, float)
    ncols = n_basis + (1 if has_cf else 0)

    def _design(vs_loc, kn):
        """Assemble and demean the link design EXACTLY as fit_single_index does: one
        `_twoway_demean` call over the whole block, control function included.

        Demeaning the CF column separately (and reusing it across draws) looks like a free
        saving and is not: `_twoway_demean` stops on max|X-prev| taken JOINTLY across columns,
        so a column demeaned on its own converges to a different point than the same column
        inside the full block. Under the constrained link an active constraint amplifies that
        into ~2e-3 on beta -- caught by the si_beta fingerprint gate below."""
        P_loc = _basis(vs_loc, kn)
        raw = P_loc * Z[:, None]
        if has_cf:
            raw = np.column_stack([raw, CF_raw])
        return _dm(raw), P_loc

    def _profile(vs_loc, kn=None, p0=None):
        """Exact link re-profile at a standardized index. Returns (b_full, P_loc).
        Constrained: the same shape-constrained QP the estimator solves, so a draw is the
        estimator applied to the perturbed criterion -- not an unconstrained fit pretending."""
        D, P_loc = _design(vs_loc, kn)
        if not constrained:
            b_u, *_ = np.linalg.lstsq(D, y_dm, rcond=None)
            return b_u, P_loc
        b, _i = solve_ispline_qp(D, y_dm, n_basis)   # exact, start-independent
        return b, P_loc

    b_full, P0 = _profile(vs, knots0)
    # FINGERPRINT ON THE LINK, NOT ON THE COEFFICIENTS.
    # For the cubic the monomial coefficients are identified and a coefficient gate is the
    # sharp one. For the constrained I-spline they are NOT: the ramp basis is strongly
    # collinear and, once sum(beta)=1 is active, the criterion is nearly flat along several
    # directions -- two solvers reach beta vectors ~2e-3 apart that describe the SAME function
    # to ~1e-9. Gating on beta there would reject a faithful rebuild. G = P beta is what is
    # identified, what is stored (si_ggrid), and what every downstream consumer evaluates, so
    # that is what must reproduce. The coefficient drift is still reported, as a diagnostic.
    # TOLERANCE. Both paths reproduce their stored link to ~1e-13 or better, so 1e-6 is a real
    # gate rather than a formality: the failures encountered while building this were orders
    # larger (a solve that stopped on the wrong active set, 1.6e-03; the wrong FE structure,
    # 2.2e-04). It is a phi tolerance, not a coefficient one -- see the note above.
    _tol_fp = 1e-6
    _d_b = float(np.max(np.abs(b_full[:n_basis] - b_stored)))
    _d_phi = float(np.max(np.abs(P0 @ b_full[:n_basis] - P0 @ b_stored)))
    if _d_phi > _tol_fp:
        raise RuntimeError(
            f"{'link' if constrained else 'si_b'} fingerprint mismatch: rebuilt link differs "
            f"from the stored one by {_d_phi:.3e} in phi (coefficient drift {_d_b:.3e}). "
            f"Refit {np.round(b_full[:n_basis], 8)} vs stored {np.round(b_stored, 8)} -- data "
            f"drift, or the stored fit was produced under a different SLEEP_LINK_CONSTRAINED "
            f"setting than this rebuild (stored link={getattr(si_res,'link',None)!r}).")
    if constrained and _d_b > 1e-6:
        print(f"  [uncond] link reproduces to {_d_phi:.2e} in phi while the coefficients differ "
              f"by {_d_b:.2e} -- expected: with sum(beta)=1 active the collinear ramp basis "
              f"leaves beta only weakly determined, the FUNCTION is what is identified.")

    # conditional-link IFs at theta-hat (mirrors the phi_band block in fit_single_index)
    D0, _ = _design(vs, knots0)
    resid0 = y_dm - D0 @ b_full
    XtX0 = D0.T @ D0
    bread_b = np.linalg.pinv(XtX0)
    IF_b = _cluster_if(D0 * resid0[:, None], bread_b, cl_inv, n_cl)

    # --- projections used by the two constrained-draw variants -----------------------------
    # PROJECTED: b* = Proj_C(b_hat + w IF_b). The constrained fit on a perturbed criterion IS
    #   the projection of the perturbed unconstrained fit (quadratic objective), so this is the
    #   estimator's own draw. Valid when no constraint binds; Andrews (2000) shows it is NOT
    #   consistent when the truth sits on the boundary.
    # TANGENT-CONE: b* = b_hat + Proj_{T(b_hat)}(w IF_b), projecting the FLUCTUATION onto the
    #   tangent cone of the active set (Hong-Li / Fang-Santos numerical delta method). This is
    #   the valid construction under active constraints. The two coincide exactly when nothing
    #   is active, which is the diagnostic reported alongside them.
    _act0 = None
    if constrained:
        def _cone_at(b_at):
            """Homogeneous tangent cone at b_at: A_j d <= 0 on upper-active rows, >= 0 on
            lower-active ones. The cone belongs to the POINT it is taken at -- the link-only
            variant takes it at b_hat, the total variant at that draw's own re-profiled
            solution, whose active set generally differs."""
            _r = qp_A @ b_at
            _hi = np.isfinite(qp_ub) & (_r >= qp_ub - 1e-7)
            _lo = np.isfinite(qp_lb) & (_r <= qp_lb + 1e-7)
            _rows, _clb, _cub = [], [], []
            for j in np.flatnonzero(_hi):
                _rows.append(qp_A[j]); _clb.append(-np.inf); _cub.append(0.0)
            for j in np.flatnonzero(_lo):
                _rows.append(qp_A[j]); _clb.append(0.0); _cub.append(np.inf)
            if not _rows:
                return None, None, None, (_hi | _lo)
            return (np.vstack(_rows), np.asarray(_clb, float), np.asarray(_cub, float),
                    (_hi | _lo))

        _cone_A, _cone_lb, _cone_ub, _act0 = _cone_at(b_full)

        _warm_lvl = [b_full.copy()]

        def _proj_level(b_lin):
            out = project_to_shape(b_lin, XtX0, qp_A, qp_lb, qp_ub, K=n_basis,
                                   warm=_warm_lvl[0])[0]
            _warm_lvl[0] = out
            return out

        def _proj_cone(d_lin):
            if _cone_A is None:
                return d_lin
            return project_to_shape(d_lin, XtX0, _cone_A, _cone_lb, _cone_ub)[0]

        def _proj_cone_at(b_at, d_lin):
            A_c, lb_c, ub_c, _ = _cone_at(b_at)
            if A_c is None:
                return d_lin
            return project_to_shape(d_lin, XtX0, A_c, lb_c, ub_c)[0]
    else:
        _proj_level = _proj_cone = _proj_cone_at = None

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
    vpow0 = P0                                   # basis at theta-hat (ramps, or monomials)
    phi_pt = np.clip(vpow0 @ b_full[:n_basis], 0.0, 1.0)
    pt = _agg_phi_t(phi_pt, gs)
    pt_leg = _agg_phi_t(phi_pt, gs_leg)

    # zero-weight reproduction gate
    if float(np.max(np.abs(np.clip(vpow0 @ (b_full + 0.0 * IF_b[0])[:n_basis], 0, 1)
                           - phi_pt))) > 1e-12:
        raise RuntimeError("zero-weight draw does not reproduce the point path")

    # --- per-cluster influence values of NATIONAL phi_t, for BCa + bootstrap-t (2026-08-06) --
    # Linearize phi_t in (theta, b) at the point estimate:
    #   link channel  (exact): d phi_t / d b_j = aggregation of vpow0[:, j] on the un-clipped
    #                          rows (the clip's derivative is an indicator);
    #   theta channel (FD)   : K re-profiled evaluations phi_t(theta + h e_k) -- the profiling
    #                          absorbs d b-hat/d theta exactly, so this is the derivative of
    #                          the PROFILED path, which is the object the draws follow.
    # Then IFphi[g, t] = (J_theta IF_theta')_g,t + (A_b IF_b[:, :d+1]')_g,t with shared wild
    # weights -- the same composition the draws use, so t*_bt = (draw - pt)/sqrt(sum w^2 IF^2)
    # is internally consistent. se/accel are the usual closed forms.
    # Under the CONSTRAINED link phi is already in [0,1], so there is no clip and no indicator:
    # A_b is the exact derivative. With a clip in place the derivative is the interior indicator
    # `_inb`, which is where the Fang-Santos non-differentiability enters.
    _g0 = vpow0 @ b_full[:n_basis]
    _inb = np.ones(len(_g0), bool) if constrained else ((_g0 > 0.0) & (_g0 < 1.0))
    A_b = np.column_stack([_agg_phi_t(np.where(_inb, vpow0[:, j], 0.0), gs)
                           for j in range(n_basis)])                         # nt x n_basis
    J_th = np.empty((gs["nt"], len(theta)))                                  # nt x K
    for k in range(len(theta)):
        h = 1e-5 * max(1.0, abs(theta[k]))
        th_e = theta.copy()
        th_e[k] += h
        v_e = X @ th_e
        vsd_e = float(v_e.std()) or 1.0
        vs_e = (v_e - float(v_e.mean())) / vsd_e
        kn_e = _knots_at(vs_e) if constrained else None
        b_e, P_e = _profile(vs_e, kn_e, p0=b_full)
        phi_e = np.clip(P_e @ b_e[:n_basis], 0.0, 1.0)
        J_th[:, k] = (_agg_phi_t(phi_e, gs) - pt) / h
    IFphi = IF_th @ J_th.T + IF_b[:, :n_basis] @ A_b.T                       # n_cl x nt
    se_tot = np.sqrt(np.maximum(np.sum(IFphi ** 2, axis=0), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        accel_tot = np.where(se_tot > 0,
                             np.sum(IFphi ** 3, axis=0) / (6.0 * se_tot ** 3), 0.0)
    IFphi2 = IFphi ** 2

    _B, _scheme = boot_cfg()
    B = int(B if B is not None else min(_B, 400))
    scheme = scheme or _scheme
    rng = np.random.default_rng(seed)
    nt = gs["nt"]
    draws_tot = np.empty((B, nt))
    draws_th = np.empty((B, nt))
    draws_ln = np.empty((B, nt))
    draws_tc = np.empty((B, nt))                   # tangent-cone, LINK channel (constrained)
    draws_tct = np.empty((B, nt))                  # tangent-cone, BOTH channels (constrained)
    draws_ln_leg = np.empty((B, gs_leg["nt"]))     # legacy key, for the reproduction gate only
    tstats_tot = np.empty((B, nt))
    diag_rows = []
    n_fail = 0
    n_proj = 0
    for ib in range(B):
        w = _wild_weights(n_cl, scheme, rng)
        th_b = theta + w @ IF_th
        v_b = X @ th_b
        vmu_b, vsd_b = float(v_b.mean()), float(v_b.std())
        flag_vsd = vsd_b <= 0
        if flag_vsd:
            vsd_b = 1.0
        vs_b = (v_b - vmu_b) / vsd_b
        kn_b = _knots_at(vs_b) if constrained else None
        try:
            b_th, P_b = _profile(vs_b, kn_b, p0=b_full)
            if not np.all(np.isfinite(b_th)):
                raise FloatingPointError("non-finite b")
        except Exception:
            b_th, P_b = b_full.copy(), _basis(vs_b, kn_b)
            n_fail += 1
        db = w @ IF_b
        if constrained:
            b_tot = _proj_level(b_th + db)                 # projected level
            b_lnk = _proj_level(b_full + db)
            b_tc = b_full + _proj_cone(db)                 # tangent-cone fluctuation
            # TOTAL under the valid construction: the draw's own re-profiled solution b_th
            # carries the direction channel, and the link fluctuation is projected onto the
            # cone AT b_th -- that draw's active set, not b_hat's. Evaluated at P_b (the basis
            # at the drawn theta) for the same reason band_total is.
            b_tct = b_th + _proj_cone_at(b_th, db)
            n_proj += int(np.max(np.abs(b_tot - (b_th + db))) > 1e-9)
        else:
            b_tot, b_lnk, b_tc = b_th + db, b_full + db, b_full + db
            b_tct = b_th + db
        phi_tot = np.clip(P_b @ b_tot[:n_basis], 0.0, 1.0)
        phi_th = np.clip(P_b @ b_th[:n_basis], 0.0, 1.0)
        phi_ln = np.clip(vpow0 @ b_lnk[:n_basis], 0.0, 1.0)
        phi_tc = np.clip(vpow0 @ b_tc[:n_basis], 0.0, 1.0)
        phi_tct = np.clip(P_b @ b_tct[:n_basis], 0.0, 1.0)
        draws_tot[ib] = _agg_phi_t(phi_tot, gs)
        draws_th[ib] = _agg_phi_t(phi_th, gs)
        draws_ln[ib] = _agg_phi_t(phi_ln, gs)
        draws_tc[ib] = _agg_phi_t(phi_tc, gs)
        draws_tct[ib] = _agg_phi_t(phi_tct, gs)
        draws_ln_leg[ib] = _agg_phi_t(phi_ln, gs_leg)
        vb = (w ** 2) @ IFphi2
        tstats_tot[ib] = (draws_tot[ib] - pt) / np.sqrt(np.maximum(vb, 1e-300))
        nth = float(np.linalg.norm(theta))
        _raw = P_b @ (b_th + db)[:n_basis]
        diag_rows.append(dict(
            cos=float(theta @ th_b / max(1e-300, nth * float(np.linalg.norm(th_b)))),
            vsd_ratio=vsd_b / vsd, b_shift=float(np.linalg.norm(b_th - b_full)),
            clip_lo=float(np.mean(_raw < 0.0)), clip_hi=float(np.mean(_raw > 1.0)),
            proj_move=float(np.linalg.norm(b_tot - (b_th + db))) if constrained else 0.0,
            vsd_flag=bool(flag_vsd)))
        if (ib + 1) % 50 == 0:
            print(f"    [uncond] draw {ib+1}/{B}", flush=True)
    if n_fail:
        print(f"  [uncond] WARNING {n_fail}/{B} refits failed (theta channel suppressed there)"
              + ("  <-- DEGRADED" if n_fail > 0.01 * B else ""))
    if constrained:
        print(f"  [uncond] shape constraints: {int(_act0.sum())} active at the estimate; "
              f"{n_proj}/{B} draws needed a non-trivial projection. The tangent-cone band is "
              f"the valid one under active constraints (Andrews 2000); it coincides with the "
              f"projected band when nothing binds.")

    out = dict(
        band_total=_band_from_draws(pt, draws_tot, gs["tuniq"], label="uncond total",
                                    accel=accel_tot, tstats=tstats_tot, se=se_tot),
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
                  constrained=bool(constrained), n_basis=int(n_basis),
                  n_active=int(_act0.sum()) if constrained else 0, n_proj=int(n_proj),
                  market_key="mca_code (V_Main:182); legacy variant on CODMUN_IBGE"))
    if constrained:
        out["band_tangent_cone"] = _band_from_draws(pt, draws_tc, gs["tuniq"],
                                                    label="uncond link-only [tangent cone]")
        # The reporting object: BOTH channels under the valid construction. `band_total`
        # projects the level and is inconsistent on the boundary (Andrews 2000);
        # `band_tangent_cone` is valid but carries the link channel only, so it understates
        # by the direction channel.
        out["band_tangent_cone_total"] = _band_from_draws(
            pt, draws_tct, gs["tuniq"], label="uncond total [tangent cone]")
    if keep_draws:
        out["draws"] = dict(total=draws_tot, theta_only=draws_th, link_only=draws_ln,
                            tangent_cone=draws_tc, tangent_cone_total=draws_tct)
    return out


# ==============================================================================
# TWO-STAGE (direction + link) wild cluster bootstrap of the E3/E4 AMEs
# ==============================================================================
# WHAT THIS FIXES. `fit_single_index` bootstraps the LINK ONLY: theta-hat is inherited from the
# Cauchy NLLS-logit and frozen across every draw, so for a continuous regressor j
#     AME_j(beta) = (theta_j / sd_j) * mean_slope(beta) = c_j * m,
# with c_j a fixed scalar and m the only random object. The scalar cancels out of
# t_j = c_j*m_hat / (|c_j| sd(m*)), and every continuous row of a column reports the SAME
# |t| -- 1.313121091 across four rows of est3 spec 12, 1.209007720 across five of est4. Each
# single-index column therefore carries ONE test ("is the link flat") plus the Pix dummy's,
# whose discrete-difference AME is a different functional of beta and so escapes.
#
# THE CONSTRUCTION. One wild weight vector w per draw drives BOTH stages, so the direction-link
# covariance is estimated rather than assumed away (Kline-Santos 2012 multiplier bootstrap on
# the stacked estimating equations):
#
#   stage A (direction)  perturb the RECENTRED score equation
#                            R(psi) = Psi(psi) - Psi(psi_hat) + w @ S_g = 0,
#                        Psi = J'(rho' f) the criterion gradient and S_g its cluster sums, and
#                        take its first Newton step psi* = psi_hat - bread_A @ (w @ S_g).
#                        `theta_mode="newton"` iterates that equation by Levenberg-Marquardt
#                        instead; see _stage_a_perturbed_solve for why the linearisation is what
#                        is reported and what the iteration measures;
#   stage B (link)       re-place the knots at the drawn index, re-assemble and re-demean the
#                        design, and re-solve the estimator's own shape-constrained QP on the
#                        ORIGINAL y; then add the link's own score channel w @ IF_b;
#   AMEs                 rebuild the WHOLE AME map at (theta*, geometry*) and evaluate it.
#
# Recentring makes psi_hat an exact root of R at w = 0 -- Psi recomputed at psi_hat reproduces
# the stored foc bit-for-bit, even though psi_hat is only an approximate root of Psi itself
# (foc_norm 1.10e-04 on est3, 1.91e-04 on est4) -- so a zero-weight draw returns the estimate
# rather than drifting off it, and the iterated mode starts from a consistent equation.
#
# SIGNS. The two stages' scores are built in OPPOSITE orientations in this module:
# `nlls_direction_if` stores J*(rho1*f) with J = dr/dpsi, i.e. +the gradient of the criterion,
# while `ols_sieve_wild_bootstrap` stores Xdm*resid, i.e. the moment X'u = -1/2 the gradient.
# Writing both stages in the moment orientation gives the coherent pairing
#     psi* = psi_hat - bread_A @ (w @ S_g)      paired with      b* = b_hat + w @ IF_b,
# i.e. a MINUS on the direction channel. `twostage_ame_boot` verifies that by finite-differencing
# the criterion (sign_gate) rather than trusting the derivation.
#
# NOT PROPAGATED (state it, do not let a referee find it): the first-stage control function
# v_hat_x_lagged_dep is a generated regressor whose uncertainty enters neither channel; and the
# link channel is the estimator's own score draw with a directionally-valid treatment of the
# active constraints (tangent cone), not the exact constrained solution of the perturbed problem.
def _stage_a_perturbed_solve(psi_hat, foc_hat, g_w, refit, bread_A, mode="if",
                             max_iter=12, tol=1e-6, max_lam=4, lam0=1e-3):
    """Perturbed stage-A direction for one bootstrap draw.

    Two modes, both solving the RECENTRED score equation R(psi) = Psi(psi) - foc_hat + g_w = 0:

      mode="if"      (default, reported) the first Newton step psi_hat - bread_A @ g_w, i.e. the
                     cluster-influence-function perturbation. This is the construction
                     `unconditional_phi_t_band` uses and the one the AME bootstrap reports.
      mode="newton"  Levenberg-Marquardt on ||R||, damping lam*diag(H) with H the clamped
                     Gauss-Newton Hessian, adapted by the usual accept/reject rule. Available
                     as a diagnostic and nested exactly inside the "if" path, since its first
                     iterate IS the "if" answer.

    WHY "if" IS THE REPORTED PATH, MEASURED. Recentring makes psi_hat an exact root at w = 0:
    Psi recomputed at psi_hat reproduces foc_hat bit-for-bit, so R(psi_hat) is exactly zero and
    a zero-weight draw returns the estimate. But at a real draw the perturbation is enormous
    relative to the curvature: ||foc_hat|| is 2.53e-03 on est3 spec 12 while ||g_w|| is 2.5-4.6,
    about a thousand times larger, because the index direction is weakly identified (its own
    cluster-robust |t| runs 0.32-1.52). Measured on the first three est3 draws, the LM iteration
    moves the direction cosine to 0.85-0.99 and leaves ||R||/||g_w|| at 3e-2 to 1.9 -- the root
    of the perturbed score equation, where one exists, is far outside the region in which the
    quadratic model holds, and no amount of damping brings it inside. Reporting a
    non-convergent solve would put a different estimator behind every draw; reporting the
    linearisation is a stated approximation with a known form.

    `res_ratio = ||R|| / ||g_w||` measures the residual against the perturbation being solved
    for. It is the honest scale: `rel`, which divides by max(1, ||rho1*f||) to be comparable
    with `foc_norm`, describes the criterion rather than the draw.

    `refit` is the plain-array bundle `nlls_direction_if(..., return_parts=True)["refit"]`.
    Consumes no RNG. Returns (psi, info)."""
    psi_hat = np.asarray(psi_hat, float)
    g_w = np.asarray(g_w, float)
    psi = psi_hat - bread_A @ g_w
    if mode == "if":
        return psi, dict(iters=0, converged=True, rel=np.nan, res_ratio=np.nan,
                         n_backtrack=0, step_norm=float(np.linalg.norm(psi - psi_hat)))
    y_dm = refit["y_dm"]; X = refit["X"]; Z = refit["Z"]; CF = refit["CF"]
    einv = refit["entity_idx"]; ecounts = refit["ecounts"]
    tinv = refit["tinv"]; tcounts = refit["tcounts"]; CF_dm = refit["CF_dm"]
    link = refit["link"]; loss = refit["loss"]; K = refit["K"]; G = refit["G"]
    p = K + G

    def _eval(pv):
        """(R, ||R||, rel, J, w2) at psi = pv, with Psi and its Jacobian rebuilt exactly as
        nlls_direction_if builds them. The Jacobian's PER-COLUMN `_twoway_demean` calls are
        load-bearing: the demeaner's stopping rule is joint across the columns it is handed, so
        one call over the whole block converges somewhere else."""
        f = _nlls_resid(pv, y_dm, X, Z, CF, einv, link, ecounts, tinv, tcounts)
        if loss == "linear":
            rho1 = np.ones(len(f)); w2 = np.ones(len(f))
        elif loss == "cauchy":
            z = f * f
            rho1 = 1.0 / (1.0 + z)
            w2 = np.maximum((1.0 - z) / (1.0 + z) ** 2, 0.0)
        else:
            raise ValueError(f"unsupported loss {loss!r}")
        base = _link_density(X @ pv[:K], link) * Z
        J = np.empty((len(f), p))
        for k in range(K):
            J[:, k] = -_twoway_demean(base * X[:, k], einv, ecounts, tinv, tcounts)
        if G > 0:
            J[:, K:] = -CF_dm
        R = J.T @ (rho1 * f) - foc_hat + g_w
        nrm = float(np.linalg.norm(R))
        rel = nrm / max(1.0, float(np.linalg.norm(rho1 * f)))
        return R, nrm, rel, J, w2

    ng = max(float(np.linalg.norm(g_w)), 1e-300)
    R, nrm, rel, J, w2 = _eval(psi)
    lam = lam0
    n_rej = 0
    it = 0
    while nrm > tol * ng and it < max_iter:
        H = (J * w2[:, None]).T @ J
        Dg = np.maximum(np.diag(H), 1e-300)
        accepted = False
        for _ in range(max_lam):
            try:
                step = -np.linalg.solve(H + lam * np.diag(Dg), R)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(H + lam * np.diag(Dg), R, rcond=None)[0]
            Rc, nrmc, relc, Jc, w2c = _eval(psi + step)
            if nrmc < nrm:
                psi, R, nrm, rel, J, w2 = psi + step, Rc, nrmc, relc, Jc, w2c
                lam = max(lam / 3.0, 1e-12)
                accepted = True
                break
            lam *= 8.0
            n_rej += 1
        it += 1
        if not accepted:
            break
    res_ratio = nrm / ng
    return psi, dict(iters=it, converged=bool(res_ratio <= tol), rel=rel,
                     res_ratio=float(res_ratio), n_backtrack=n_rej, lam=float(lam),
                     step_norm=float(np.linalg.norm(psi - psi_hat)))


def _ame_band_from_draws(names, ame_hat, draws, alpha=0.05, label=""):
    """Per-ROW raw-percentile + Efron (1987) bias-corrected interval for an AME draw matrix.

    Same arithmetic as `_band_from_draws`, one row per regressor instead of one per quarter.
    It does NOT delegate there: that function's tail parses `time_id` as a PeriodIndex and
    otherwise sorts by it, which on AME names would silently alphabetise the rows.

    `p_bc` is the p-value implied by the printed BC interval, so stars and interval can never
    disagree: the BC lower endpoint clears zero iff alpha/2 > Phi(Phi^-1(p_zero) - 2 z0), with
    p_zero the draw mass below zero. Columns: name, ame, se, lo, hi, lo_bc, hi_bc, p_below,
    p_bc, stars."""
    from scipy.stats import norm as _norm
    draws = np.asarray(draws, float)
    B = draws.shape[0]
    pt = np.array([float(ame_hat[k]) for k in names])
    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    p_below = (1.0 + np.sum(draws < pt[None, :], axis=0)) / (1.0 + B)
    z0 = _norm.ppf(np.clip(p_below, 1.0 / (1.0 + B), B / (1.0 + B)))
    za_lo, za_hi = _norm.ppf(alpha / 2), _norm.ppf(1 - alpha / 2)
    a_lo = 100.0 * _norm.cdf(2 * z0 + za_lo)
    a_hi = 100.0 * _norm.cdf(2 * z0 + za_hi)
    nk = draws.shape[1]
    lo_bc = np.array([np.percentile(draws[:, j], a_lo[j]) for j in range(nk)])
    hi_bc = np.array([np.percentile(draws[:, j], a_hi[j]) for j in range(nk)])
    p_zero = (1.0 + np.sum(draws < 0.0, axis=0)) / (1.0 + B)
    q = _norm.cdf(_norm.ppf(np.clip(p_zero, 1.0 / (1.0 + B), B / (1.0 + B))) - 2 * z0)
    p_bc = 2.0 * np.minimum(q, 1.0 - q)
    st = np.where(p_bc < 0.01, "***", np.where(p_bc < 0.05, "**",
                                               np.where(p_bc < 0.1, "*", "")))
    out = pd.DataFrame({"name": list(names), "ame": pt,
                        "se": np.std(draws, axis=0, ddof=1), "lo": lo, "hi": hi,
                        "lo_bc": lo_bc, "hi_bc": hi_bc, "p_below": p_below,
                        "p_bc": p_bc, "stars": st})
    n_out = int(np.sum((pt < lo) | (pt > hi)))
    if n_out and label:
        print(f"  [{label}] RAW percentile interval excludes the point estimate on "
              f"{n_out}/{nk} row(s); report the BC interval.")
    return out


# --- per-draw kernel, shared by the serial and the worker-process paths ----------------------
# Every ON draw is a PURE function of its own weight vector: the QP warm start is the fixed
# b_full (never the previous draw's solution) and no RNG is touched, so draws can be dispatched
# across processes in any order without moving a number. That is what makes `workers>1`
# reproduce the serial run bit-for-bit and makes the answer independent of the worker count.
_TS_CTX = {}


def _ts_worker_init(ctx):
    _TS_CTX.clear()
    _TS_CTX.update(ctx)


def _ts_worker_task(arg):
    scheme, ib, w, cold = arg
    return ib, _ts_one_draw(_TS_CTX, scheme, np.asarray(w, float), cold_check=cold)


def _ts_worker_ping(_i):
    """No-op that forces a worker process to finish starting, so the pool's startup cost is
    paid and measured before the draw loop rather than inside its timing."""
    return len(_TS_CTX)


def _ts_one_draw(ctx, scheme, w, cold_check=False):
    """One two-stage draw. Returns a dict with the cone/level AME vectors (ordered by
    ctx['names']) and the per-draw diagnostics."""
    K = ctx["K"]
    X = ctx["X"]; Z = ctx["Z"]; y_dm = ctx["y_dm"]; CF_raw = ctx["CF_raw"]
    degree = ctx["degree"]; n_basis = ctx["n_basis"]; constrained = ctx["constrained"]
    b_full = ctx["b_full"]; names = ctx["names"]
    theta_hat = ctx["theta_hat"]
    d = dict(ok=True, fail=0, newton_fail=0, cos_neg=0, vsd_fail=0, proj=0, knot_tie=0,
             warm_vs_cold=np.nan)

    g_w = w @ ctx["S"][scheme]
    psi_star, ninfo = _stage_a_perturbed_solve(ctx["psi_hat"], ctx["foc_hat"], g_w,
                                               ctx["refit"], ctx["bread_A"],
                                               mode=ctx["theta_mode"])
    th_b = psi_star[:K]
    d["newton_fail"] = int(not ninfo["converged"])

    v_b = X @ th_b
    vmu_b, vsd_b = float(v_b.mean()), float(v_b.std())
    if not np.isfinite(vsd_b) or vsd_b <= 0 or not np.all(np.isfinite(v_b)):
        # The phi_t band substitutes vsd_b = 1.0 here, which is harmless for phi and NOT for an
        # AME: it would rescale the whole draw by an arbitrary factor. Discard instead.
        d.update(ok=False, vsd_fail=1)
        return d
    vs_b = (v_b - vmu_b) / vsd_b
    cos_b = float(theta_hat @ th_b / max(1e-300, float(np.linalg.norm(theta_hat))
                                         * float(np.linalg.norm(th_b))))
    d["cos_neg"] = int(cos_b < 0)

    if constrained:
        # The knot placement is INSIDE the estimator (quantiles of the fitted index, extended
        # over the Pix-removed support), so the draw re-places it at its own index -- and the
        # counterfactual hull is measured at the DRAWN theta and the DRAWN scale, which is what
        # fit_single_index does with the theta and scale that built its index. A frozen hull
        # would leave part of vs_b pinned at `_ramp_design`'s clamp, where the fitted derivative
        # is zero, dragging every continuous AME draw toward zero.
        lo_b, hi_b = _pix_shifted_range(X, vs_b, th_b, ctx["phi_params"], vsd_b)
        _, kn_b = _bspline_design(np.concatenate([vs_b, np.array([lo_b, hi_b])]),
                                  n_interior=SI_N_INTERIOR, degree=degree)
        if len(kn_b) != len(ctx["knots0"]):
            d.update(ok=False, fail=1)
            return d
        d["knot_tie"] = int(not np.all(np.diff(kn_b[degree + 1:-(degree + 1)]) > 0))
        P_b = _ramp_design(vs_b, kn_b, degree)
    else:
        kn_b = None
        P_b = np.column_stack([vs_b ** dd for dd in range(degree + 1)])

    raw = P_b * Z[:, None]
    if ctx["has_cf"]:
        raw = np.column_stack([raw, CF_raw])
    # ONE joint _twoway_demean over the whole block. Demeaning the control-function column on
    # its own and caching it across draws is the obvious saving and is wrong: the demeaner stops
    # on max|X-prev| taken JOINTLY across the columns it is given, so a column demeaned alone
    # converges elsewhere and moves beta by ~2e-3 under an active constraint.
    D_b = _twoway_demean(raw, ctx["einv"], ctx["ecounts"], ctx["tinv"], ctx["tcounts"])

    qp_A, qp_lb, qp_ub = ctx["qp_A"], ctx["qp_lb"], ctx["qp_ub"]
    if constrained:
        # THE EXACT RE-PROFILE: the estimator's own shape-constrained QP, at the drawn index, on
        # the ORIGINAL y_dm. Solving on the original y is what makes the level noise cancel
        # between b(th_b) and b(theta_hat), leaving the d beta / d theta channel alone; the
        # link's own noise enters once, through w @ IF_b below. The warm start is the FIXED
        # b_full, never the previous draw's solution, which is what keeps each draw a pure
        # function of its own weights.
        try:
            b_th, qp_i = solve_ispline_qp(D_b, y_dm, n_basis, warm=b_full)
            if not np.all(np.isfinite(b_th)):
                raise FloatingPointError("non-finite b")
        except Exception:
            b_th, qp_i = b_full.copy(), None
            d["fail"] = 1
        d["qp_viol"] = _qp_violation(b_th, qp_A, qp_lb, qp_ub)
        if cold_check and not d["fail"]:
            # A warm face that keeps a constraint it should release certifies a wrong optimum
            # silently (measured 8.75e-02 in solve_ispline_qp's own docstring), and this scheme
            # pays ~2000 warm solves per cell. Compare in PHI, not in coefficients.
            b_cold, _ = solve_ispline_qp(D_b, y_dm, n_basis)
            d["warm_vs_cold"] = float(np.max(np.abs(P_b @ (b_th - b_cold)[:n_basis])))
    else:
        b_th, *_ = np.linalg.lstsq(D_b, y_dm, rcond=None)
        qp_i = None
        d["qp_viol"] = 0.0

    db = w @ ctx["IF_b"][scheme]
    if constrained:
        # TANGENT CONE AT THE DRAW'S OWN SOLUTION, not at b_hat: the direction channel is
        # precisely what moves the active set (Hong-Li / Fang-Santos numerical delta method;
        # Andrews 2000 for why the projected level is inconsistent on the boundary).
        A_c, lb_c, ub_c, _act = _shape_cone_at(qp_A, qp_lb, qp_ub, b_th)
        d_cone = db if A_c is None else project_to_shape(db, ctx["XtX0"], A_c, lb_c, ub_c)[0]
        b_cone = b_th + d_cone
        b_lvl = project_to_shape(b_th + db, ctx["XtX0"], qp_A, qp_lb, qp_ub,
                                 K=n_basis, warm=b_th)[0]
        d["proj"] = int(float(np.max(np.abs(b_lvl - (b_th + db)))) > 1e-9)
        d["proj_move"] = float(np.linalg.norm(b_lvl - (b_th + db)))
        d["n_active"] = int(_act.sum())
    else:
        b_cone = b_lvl = b_th + db
        d["proj_move"] = 0.0
        d["n_active"] = 0

    m_b = _si_ame_map(X, ctx["phi_params"], th_b, vs_b, vsd_b, P_b, kn_b, degree, n_basis,
                      constrained, dummy_spec=ctx["dummy_spec0"])
    a_cone = m_b["ame_fn"](b_cone)
    a_lvl = m_b["ame_fn"](b_lvl)
    d["ame_cone"] = np.array([a_cone[k] for k in names])
    d["ame_level"] = np.array([a_lvl[k] for k in names])
    d.update(cos=cos_b, rel_pert=float(np.linalg.norm(psi_star - ctx["psi_hat"])
                                       / max(1e-300, float(np.linalg.norm(ctx["psi_hat"])))),
             vsd_ratio=vsd_b / ctx["vsd"], newton_iters=ninfo["iters"],
             newton_rel=ninfo["rel"], newton_res_ratio=ninfo["res_ratio"],
             newton_converged=bool(ninfo["converged"]), newton_backtrack=ninfo["n_backtrack"],
             b_shift=float(np.linalg.norm(b_th - b_full)),
             n_zero=(int(qp_i["n_zero"]) if qp_i else -1),
             at_cap=(bool(qp_i["at_cap"]) if qp_i else False),
             sum_beta=(float(qp_i["sum_beta"]) if qp_i else np.nan),
             ms_ratio=(m_b["mean_slope"](b_cone) / ctx["mean_slope_hat"]
                       if ctx["mean_slope_hat"] else np.nan))
    return d


def twostage_ame_boot(df, state_cols, has_cf, si_res, loss, degree=3, fe_time_col=None,
                      B=None, scheme=None, seed=0, theta_channel=True, theta_mode="if",
                      alpha=0.05, keep_draws=True, cold_check_every=50, progress_every=25,
                      workers=1):
    """Two-stage (direction + link) wild cluster bootstrap of the E3/E4 AMEs, from a STORED
    `fit_single_index` result -- no re-estimation.

    THE OFF SWITCH. `theta_channel=False` is a STRUCTURAL SHORT-CIRCUIT, not `g_w = 0`: the
    stage-A solve, the knot re-placement, the re-profile and the AME rebuild are not executed at
    all, and the loop body reduces to `w = W[ib]; b = proj(b_full + w @ IF_b); ame_fn0(b)` --
    operation-for-operation `cluster_wild_bootstrap`, on the same operands in the same order. It
    reproduces the stored conditional SEs, which is the regression guard for everything above.
    Setting g_w = 0 would NOT do that: the QP would still be re-solved and would return b_full
    only to solver tolerance. The OFF path therefore also runs SERIAL, conglomerate first,
    sharing ONE projection warm cell -- because the estimator shares `_warm` across its two
    `ols_sieve_wild_bootstrap` calls, so `bse_time` depends on the conglomerate loop having
    consumed its draws first.

    Returns dict(congl=..., quarter=..., meta=...); each per-scheme value carries ame, bse,
    pvalues, band (the reporting object), the level-projected diagnostic variants, cov,
    the raw draws and the per-draw diagnostics."""
    import time as _time
    t0 = _time.time()
    if wcb_mode() != "normal":
        raise RuntimeError(f"twostage_ame_boot requires SLEEP_WCB_MODE=normal, got "
                           f"{wcb_mode()!r}: the studentised path linearises the AME at a "
                           f"frozen theta, which is the wrong studentiser here.")
    _B, _scheme = boot_cfg()
    B = int(_B if B is None else B)
    scheme = scheme or _scheme

    # ---- index names, asserted elementwise against the stored fit --------------------------
    idx_expect = [f"interaction_{sv}" if sv != "constant" else "nr_lagged_dep"
                  for sv in state_cols]
    phi_params = list(si_res.params_native.index)
    if phi_params != idx_expect:
        raise RuntimeError(f"index name mismatch: fit carries {phi_params} but state_cols "
                           f"imply {idx_expect}")
    constrained = (getattr(si_res, "si_constrained", False)
                   or getattr(si_res, "link", None) == "index_sieve")

    # ---- rebuild the estimation frame exactly as fit_single_index --------------------------
    CF_cols = ["v_hat_x_lagged_dep"] if has_cf else []
    cols = state_cols + ["deposit_balance", "nr_lagged_dep", "entity_id"]
    df_ss = df.dropna(subset=cols + CF_cols).copy()
    X = _build_phi_X(df_ss, phi_params)
    Z = df_ss["nr_lagged_dep"].values.astype(float)
    _, einv = np.unique(df_ss["entity_id"].values, return_inverse=True)
    ecounts = np.bincount(einv).astype(float)
    if fe_time_col is not None:
        _, tinv = np.unique(df_ss[fe_time_col].values, return_inverse=True)
        tcounts = np.bincount(tinv).astype(float)
    else:
        tinv = tcounts = None
    y_dm = _twoway_demean(df_ss["deposit_balance"].values.astype(float),
                          einv, ecounts, tinv, tcounts)
    CF_raw = df_ss["v_hat_x_lagged_dep"].values.astype(float) if has_cf else None
    cl = df_ss["CodConglomeradoPrudencial"].astype(str)
    cl_u, cl_inv = np.unique(cl.values, return_inverse=True)
    n_cl = len(cl_u)
    if fe_time_col is None or fe_time_col not in df_ss.columns:
        raise RuntimeError("fe_time_col is required: the quarter-clustered scheme is the one "
                           "the national rows are reported from, and a silent skip there would "
                           "leave the tables describing a calculation that did not run")
    _per = df_ss[fe_time_col].astype(str).values
    _uniq, _t_inv = np.unique(_per, return_inverse=True)
    n_t = len(_uniq)

    # ---- direction and its standardisation, gated on the stored fingerprint ----------------
    theta_hat = si_res.params_native[phi_params].values.astype(float)
    v = X @ theta_hat
    vmu, vsd = float(v.mean()), float(v.std())
    if vsd <= 0:
        vsd = 1.0
    if abs(vmu - float(si_res.si_vmu)) > 1e-6 * max(1.0, abs(vmu)) or \
       abs(vsd - float(si_res.si_vsd)) > 1e-6 * max(1.0, abs(vsd)):
        raise RuntimeError(f"vmu/vsd fingerprint mismatch: rebuilt ({vmu:.10g},{vsd:.10g}) vs "
                           f"stored ({si_res.si_vmu:.10g},{si_res.si_vsd:.10g}) -- data drift")
    vs = (v - vmu) / vsd

    # ---- the link at theta-hat, COLD (which is how the estimator solves it) ----------------
    if constrained:
        knots0 = np.asarray(si_res.si_knots, float)
        n_basis = len(knots0) - degree - 1
        P0 = _ramp_design(vs, knots0, degree)
        b_stored = np.asarray(si_res.si_beta, float)
    else:
        knots0 = None
        n_basis = degree + 1
        P0 = np.column_stack([vs ** d for d in range(degree + 1)])
        b_stored = np.asarray(si_res.si_b, float)
    raw0 = P0 * Z[:, None]
    if has_cf:
        raw0 = np.column_stack([raw0, CF_raw])
    D0 = _twoway_demean(raw0, einv, ecounts, tinv, tcounts)
    if constrained:
        b_full, qp_info0 = solve_ispline_qp(D0, y_dm, n_basis)
        qp_A, qp_lb, qp_ub = ispline_constraints(n_basis, n_extra=raw0.shape[1] - n_basis)
    else:
        b_full, *_ = np.linalg.lstsq(D0, y_dm, rcond=None)
        qp_info0 = None
        qp_A = qp_lb = qp_ub = None
    resid0 = y_dm - D0 @ b_full
    XtX0 = D0.T @ D0
    bread_link = np.linalg.pinv(XtX0)
    # Fingerprint on the LINK, not on the coefficients: the ramp basis is collinear, so two
    # faithful solves can sit ~1e-3 apart in beta and describe the same function to ~1e-9.
    d_b = float(np.max(np.abs(b_full[:n_basis] - b_stored)))
    d_phi = float(np.max(np.abs(P0 @ b_full[:n_basis] - P0 @ b_stored)))
    if d_phi > 1e-6:
        raise RuntimeError(f"link fingerprint mismatch: rebuilt link differs from the stored "
                           f"one by {d_phi:.3e} in phi (coefficient drift {d_b:.3e})")

    # ---- the AME map at theta-hat ----------------------------------------------------------
    amemap0 = _si_ame_map(X, phi_params, theta_hat, vs, vsd, P0, knots0, degree, n_basis,
                          constrained)
    names = amemap0["names"]
    ame_fn0 = amemap0["ame_fn"]
    ame_hat = ame_fn0(b_full)
    mean_slope_hat = amemap0["mean_slope"](b_full)
    d_ame = max(abs(ame_hat[k] - float(si_res.params[k]))
                / max(1e-300, abs(float(si_res.params[k]))) for k in names)
    # d_b / d_phi / d_ame are the reproduction budget of the whole run and are printed every
    # time, because they set the wording of the theta-frozen reproduction claim: the OFF path
    # re-solves the link from the rebuilt frame, so whatever separates b_full from the stored
    # si_beta separates its SEs from the stored ones too. The gate is 1e-6, the same phi
    # tolerance unconditional_phi_t_band uses -- large enough not to reject a faithful rebuild
    # of a collinear ramp basis, small enough that every failure encountered while building this
    # (a solve stopped on the wrong active set, 1.6e-03; the wrong FE structure, 2.2e-04) trips.
    print(f"  [2s] rebuild: d_beta={d_b:.3e}  d_phi={d_phi:.3e}  d_ame(rel)={d_ame:.3e}  "
          f"n={len(df_ss)}  G={n_cl}  T={n_t}  K={n_basis}")
    if d_ame > 1e-6:
        raise RuntimeError(f"AME fingerprint mismatch: rebuilt AMEs differ from the stored "
                           f"params by {d_ame:.3e} relative")

    # ---- direction influence functions and the stage-A refit bundle ------------------------
    dirif = nlls_direction_if(df, state_cols, has_cf, pd.Series(theta_hat, index=idx_expect),
                              loss, link="logit", fe_time_col=fe_time_col, return_parts=True)
    if list(dirif["idx"]) != idx_expect:
        raise RuntimeError("nlls_direction_if index mismatch")
    # ALIGNMENT. A silent misalignment would pair cluster g's direction influence with cluster
    # h's link influence and destroy exactly the covariance the shared w exists to capture, so
    # the two frames are compared row by row rather than by length.
    d_X = float(np.max(np.abs(np.asarray(dirif["refit"]["X"], float) - X)))
    if d_X != 0.0 or len(dirif["cl_labels"]) != len(cl.values) \
            or not np.array_equal(np.asarray(dirif["cl_labels"]), cl.values) \
            or not dirif["row_index"].equals(df_ss.index):
        raise RuntimeError(f"stage-A / stage-B frame misalignment (max|dX|={d_X:.3e})")
    dirif["refit"]["X"] = X            # one array object, so a worker payload carries it once
    psi_hat = dirif["psi_hat"]; foc_hat = dirif["foc"]; bread_A = dirif["bread"]
    score_rows = dirif["score_rows"]
    K = dirif["diag"]["K"]

    # ---- cluster sums of the stage-A score, and the link IFs, per scheme -------------------
    def _csum(inv, ng):
        return np.column_stack([np.bincount(inv, weights=score_rows[:, j], minlength=ng)
                                for j in range(score_rows.shape[1])])
    S_of = {"congl": _csum(cl_inv, n_cl), "quarter": _csum(_t_inv, n_t)}
    IF_b_of = {"congl": _cluster_if(D0 * resid0[:, None], bread_link, cl_inv, n_cl),
               "quarter": _cluster_if(D0 * resid0[:, None], bread_link, _t_inv, n_t)}
    n_of = {"congl": n_cl, "quarter": n_t}

    # ---- SIGN CALIBRATION by finite difference, not by derivation --------------------------
    sign_gate = _stage_a_sign_gate(psi_hat, foc_hat, dirif["refit"])
    print(f"  [2s] sign gate: {sign_gate['verdict']} (rel dev +{sign_gate['rel_plus']:.2e} / "
          f"-{sign_gate['rel_minus']:.2e} on coords {sign_gate['coords']})")
    if sign_gate["verdict"] == "FLIPPED":
        raise RuntimeError("the stage-A score is MINUS the criterion gradient; flip g_w's sign "
                           "in _ts_one_draw before running -- do not proceed on the derivation")
    if sign_gate["verdict"] != "OK":
        print("  [2s] WARNING sign gate inconclusive; the direction channel's sign rests on "
              "the derivation alone.")

    # ---- PRE-DRAW the full weight matrix in the parent, from the seeded RNG -----------------
    # One `_wild_weights` call per draw, so the stream is identical to drawing inside the loop;
    # a fresh generator per scheme, because the estimator's two `ols_sieve_wild_bootstrap` calls
    # both run on the hardcoded default seed. Workers never seed or advance an RNG -- they are
    # handed their row of W -- which is what makes the result independent of the worker count.
    W_of = {}
    for s in ("congl", "quarter"):
        rng = np.random.default_rng(seed)
        W_of[s] = np.stack([_wild_weights(n_of[s], scheme, rng) for _ in range(B)])

    meta = dict(B=B, scheme=scheme, seed=seed, loss=loss, theta_mode=theta_mode,
                theta_channel=bool(theta_channel), degree=degree, n=len(df_ss),
                n_cl=n_cl, n_t=n_t, foc_norm=dirif["foc_norm"], gamma_hat=dirif["gamma_hat"],
                dir_diag=dirif["diag"], vmu=vmu, vsd=vsd, theta=theta_hat, idx=idx_expect,
                b_full=b_full, d_b=d_b, d_phi=d_phi, d_ame=d_ame, n_basis=int(n_basis),
                constrained=bool(constrained), mean_slope_hat=mean_slope_hat,
                sign_gate=sign_gate, names=list(names),
                n_active_at_estimate=(int(qp_info0["n_active"]) if qp_info0 else 0),
                workers=int(workers), alpha=alpha)

    out = {"meta": meta}
    if not theta_channel:
        out.update(_ts_off_path(names, ame_fn0, ame_hat, b_full, IF_b_of, W_of, B,
                                constrained, XtX0, qp_A, qp_lb, qp_ub, n_basis, alpha))
        meta["runtime_s"] = _time.time() - t0
        return out

    ctx = dict(X=X, Z=Z, y_dm=y_dm, CF_raw=CF_raw, has_cf=bool(has_cf), einv=einv,
               ecounts=ecounts, tinv=tinv, tcounts=tcounts, phi_params=phi_params,
               theta_hat=theta_hat, vsd=vsd, knots0=knots0, degree=degree,
               n_basis=int(n_basis), constrained=bool(constrained), qp_A=qp_A, qp_lb=qp_lb,
               qp_ub=qp_ub, XtX0=XtX0, b_full=b_full, dummy_spec0=amemap0["dummy_spec"],
               names=names, refit=dirif["refit"], psi_hat=psi_hat, foc_hat=foc_hat,
               bread_A=bread_A, theta_mode=theta_mode, K=K, S=S_of, IF_b=IF_b_of,
               mean_slope_hat=mean_slope_hat)

    # ---- INVARIANCE. Every AME is exactly invariant to theta -> a*theta (a>0) and to a shift
    # of the constant coefficient, and roughly half the perturbation energy lies in that
    # subspace -- so a map that refreshed only part of (ths, slope_w, dcols, knots, vs, vsd)
    # would give distinct, plausible, WRONG t's and fail nothing else. Drive the geometry
    # rebuild with those two perturbations and require the AMEs back unchanged.
    meta["invariance_gate"] = _ts_invariance_gate(ctx, ame_hat, names)
    print(f"  [2s] invariance gate: scale {meta['invariance_gate']['scale']:.2e}  "
          f"const-shift {meta['invariance_gate']['shift']:.2e}  "
          f"{meta['invariance_gate']['verdict']}")
    if meta["invariance_gate"]["verdict"] != "PASS":
        raise RuntimeError("AME scale/shift invariance violated -- the draw geometry is only "
                           "partially rebuilt")

    # ONE worker pool for BOTH schemes. On Windows spawn each worker re-imports this module's
    # dependency stack and unpickles the ~55 MB context, which is minutes on a loaded box --
    # far more than a scheme's draws cost at small B. A driver calling this with workers > 1
    # must have an `if __name__ == "__main__":` guard, or spawn re-runs its top-level work in
    # every worker; estimation_ame_twostage.py has one.
    pool = None
    try:
        if workers and workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            print(f"  [2s] starting {int(workers)} worker processes", flush=True)
            _t_pool = _time.time()
            pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_ts_worker_init,
                                       initargs=(ctx,))
            # Force every worker up and time it. The startup is NOT small: each child re-imports
            # the caller's whole module graph and unpickles the ~55 MB context, measured at tens
            # of seconds per worker on this repo (the modules live on a OneDrive-backed path) and
            # minutes each when the box is busy. It is paid once per call, so it is amortised at
            # production B and dominates at smoke B -- which is why the number is printed rather
            # than hidden, and why `workers=1` is the right choice for short runs.
            list(pool.map(_ts_worker_ping, range(int(workers) * 4)))
            meta["pool_startup_s"] = _time.time() - _t_pool
            print(f"  [2s] worker pool ready in {meta['pool_startup_s']:.1f}s", flush=True)
        for s in ("congl", "quarter"):
            out[s] = _ts_run_scheme(ctx, s, W_of[s], B, names, ame_hat, alpha, keep_draws,
                                    cold_check_every, progress_every, workers, pool=pool)
    finally:
        if pool is not None:
            pool.shutdown()
    meta["runtime_s"] = _time.time() - t0
    return out


def _stage_a_sign_gate(psi_hat, foc_hat, refit, h_rel=1e-5, n_coord=3, tol=1e-2, sep=100.0):
    """Is the stored stage-A score PLUS or MINUS the gradient of the criterion scipy minimises?

    Central-differences cost(psi) = 0.5*sum(ln(1+f^2)) (cauchy) or 0.5*sum(f^2) (linear) on the
    coordinates with the largest |foc|, and compares against foc. Verdict 'OK' means
    Psi = +gradient, hence that the coherent perturbation is psi_hat - bread @ (w @ S) paired
    with b_hat + w @ IF_b. 'FLIPPED' means the caller must flip g_w's sign.

    The test is about a SIGN, so the tolerance is loose and the discrimination comes from the
    separation between the two candidate orientations: cost sums ~487k terms of order 40, so a
    central difference of it carries ~1e-4 relative cancellation error, while the wrong
    orientation is off by a factor of two. Requiring the winner to be `sep` times better than
    the loser is what makes 1e-4 versus 2.0 a verdict rather than a coincidence."""
    y_dm = refit["y_dm"]; X = refit["X"]; Z = refit["Z"]; CF = refit["CF"]
    einv = refit["entity_idx"]; ecounts = refit["ecounts"]
    tinv = refit["tinv"]; tcounts = refit["tcounts"]
    link = refit["link"]; loss = refit["loss"]

    def _cost(p):
        f = _nlls_resid(p, y_dm, X, Z, CF, einv, link, ecounts, tinv, tcounts)
        return float(0.5 * np.sum(f * f) if loss == "linear"
                     else 0.5 * np.sum(np.log1p(f * f)))

    coords = np.argsort(-np.abs(np.asarray(foc_hat, float)))[:n_coord]
    fd, an = [], []
    for k in coords:
        h = h_rel * max(1.0, abs(float(psi_hat[k])))
        pp = np.asarray(psi_hat, float).copy(); pp[k] += h
        pm = np.asarray(psi_hat, float).copy(); pm[k] -= h
        fd.append((_cost(pp) - _cost(pm)) / (2 * h))
        an.append(float(foc_hat[k]))
    fd = np.asarray(fd); an = np.asarray(an)
    rel_plus = float(np.max(np.abs(fd - an) / np.maximum(np.abs(an), 1e-300)))
    rel_minus = float(np.max(np.abs(fd + an) / np.maximum(np.abs(an), 1e-300)))
    verdict = ("OK" if rel_plus <= tol and rel_minus >= sep * max(rel_plus, 1e-300) else
               "FLIPPED" if rel_minus <= tol and rel_plus >= sep * max(rel_minus, 1e-300)
               else "INCONCLUSIVE")
    return dict(verdict=verdict, max_rel=min(rel_plus, rel_minus), rel_plus=rel_plus,
                rel_minus=rel_minus, coords=[int(c) for c in coords],
                fd=fd.tolist(), foc=an.tolist())


def _ts_invariance_gate(ctx, ame_hat, names, tol=1e-9):
    """Rerun the draw kernel's geometry rebuild at th_b = 1.7*theta_hat and at
    theta_hat + 0.3*e_const with a ZERO link fluctuation, and require the AMEs back."""
    a0 = np.array([ame_hat[k] for k in names])
    res = {}
    K = ctx["K"]
    for tag, th_b in (("scale", 1.7 * ctx["theta_hat"]),
                      ("shift", ctx["theta_hat"] + 0.3 * np.eye(len(ctx["theta_hat"]))[0])):
        psi_b = np.concatenate([th_b, ctx["psi_hat"][K:]])
        a = _ts_geometry_ame(ctx, psi_b)
        res[tag] = float(np.max(np.abs(a - a0) / np.maximum(np.abs(a0), 1e-300)))
    res["verdict"] = "PASS" if max(res["scale"], res["shift"]) <= tol else "FAIL"
    return res


def _ts_geometry_ame(ctx, psi_b):
    """AMEs at a GIVEN psi with no link fluctuation: the draw kernel's geometry rebuild and
    re-profile with db = 0. Used by the invariance gate, which is about the rebuild rather
    than about the stage-A solve."""
    K = ctx["K"]
    th_b = psi_b[:K]
    X = ctx["X"]; Z = ctx["Z"]
    v_b = X @ th_b
    vmu_b, vsd_b = float(v_b.mean()), float(v_b.std())
    vs_b = (v_b - vmu_b) / vsd_b
    degree, n_basis = ctx["degree"], ctx["n_basis"]
    if ctx["constrained"]:
        lo_b, hi_b = _pix_shifted_range(X, vs_b, th_b, ctx["phi_params"], vsd_b)
        _, kn_b = _bspline_design(np.concatenate([vs_b, np.array([lo_b, hi_b])]),
                                  n_interior=SI_N_INTERIOR, degree=degree)
        P_b = _ramp_design(vs_b, kn_b, degree)
    else:
        kn_b = None
        P_b = np.column_stack([vs_b ** dd for dd in range(degree + 1)])
    raw = P_b * Z[:, None]
    if ctx["has_cf"]:
        raw = np.column_stack([raw, ctx["CF_raw"]])
    D_b = _twoway_demean(raw, ctx["einv"], ctx["ecounts"], ctx["tinv"], ctx["tcounts"])
    if ctx["constrained"]:
        b_th, _ = solve_ispline_qp(D_b, ctx["y_dm"], n_basis, warm=ctx["b_full"])
    else:
        b_th, *_ = np.linalg.lstsq(D_b, ctx["y_dm"], rcond=None)
    m_b = _si_ame_map(X, ctx["phi_params"], th_b, vs_b, vsd_b, P_b, kn_b, degree, n_basis,
                      ctx["constrained"], dummy_spec=ctx["dummy_spec0"])
    a = m_b["ame_fn"](b_th)
    return np.array([a[k] for k in ctx["names"]])


def _ts_run_scheme(ctx, s, W, B, names, ame_hat, alpha, keep_draws, cold_check_every,
                   progress_every, workers, pool=None):
    """Run one clustering scheme's B draws, serially or on a worker pool.

    `pool` is an already-started executor holding ctx; the caller opens ONE for both schemes,
    because on Windows spawn each worker re-imports this module's whole dependency stack and
    unpickles the ~55 MB context, which costs far more than a scheme's worth of draws at smoke
    sizes."""
    import time as _time
    t0 = _time.time()
    nk = len(names)
    draws_cone = {k: np.full(B, np.nan) for k in names}
    draws_level = {k: np.full(B, np.nan) for k in names}
    diag_rows = [None] * B
    counters = dict(n_fail=0, n_newton_fail=0, n_cos_neg=0, n_vsd_fail=0, n_proj=0,
                    n_knot_tie=0, n_drop=0)
    tasks = [(s, ib, W[ib], bool(cold_check_every and (ib + 1) % cold_check_every == 0))
             for ib in range(B)]

    def _absorb(ib, d):
        for key, cnt in (("fail", "n_fail"), ("newton_fail", "n_newton_fail"),
                         ("cos_neg", "n_cos_neg"), ("vsd_fail", "n_vsd_fail"),
                         ("proj", "n_proj"), ("knot_tie", "n_knot_tie")):
            counters[cnt] += int(d.get(key, 0))
        if not d.get("ok", False):
            counters["n_drop"] += 1
            diag_rows[ib] = dict(draw=ib, ok=False)
            return
        for j, k in enumerate(names):
            draws_cone[k][ib] = d["ame_cone"][j]
            draws_level[k][ib] = d["ame_level"][j]
        diag_rows[ib] = {kk: vv for kk, vv in d.items()
                         if kk not in ("ame_cone", "ame_level")}
        diag_rows[ib]["draw"] = ib

    if pool is not None:
        for n_done, (ib, d) in enumerate(pool.map(_ts_worker_task, tasks, chunksize=1), 1):
            _absorb(ib, d)
            if progress_every and n_done % progress_every == 0:
                print(f"    [2s {s}] draw {n_done}/{B}  "
                      f"({(_time.time()-t0)/n_done:.2f}s/draw)", flush=True)
    else:
        for n_done, task in enumerate(tasks, 1):
            _, ib, w, cold = task
            _absorb(ib, _ts_one_draw(ctx, s, w, cold_check=cold))
            if progress_every and n_done % progress_every == 0:
                print(f"    [2s {s}] draw {n_done}/{B}  "
                      f"({(_time.time()-t0)/n_done:.2f}s/draw)", flush=True)

    keep = np.array([bool(diag_rows[ib].get("ok", False)) for ib in range(B)])
    Dc = np.column_stack([draws_cone[k][keep] for k in names])
    Dl = np.column_stack([draws_level[k][keep] for k in names])
    bse = {k: float(np.std(draws_cone[k][keep], ddof=1)) for k in names}
    pv = {k: (float(2 * stats.norm.sf(abs(ame_hat[k]) / bse[k])) if bse[k] > 0 else 0.0)
          for k in names}
    bse_l = {k: float(np.std(draws_level[k][keep], ddof=1)) for k in names}
    pv_l = {k: (float(2 * stats.norm.sf(abs(ame_hat[k]) / bse_l[k])) if bse_l[k] > 0 else 0.0)
            for k in names}
    cov = np.cov(Dc, rowvar=False, ddof=1).reshape(nk, nk)
    d_cov = float(np.max(np.abs(np.sqrt(np.diag(cov)) - np.array([bse[k] for k in names]))))
    if d_cov > 1e-12 * max(1e-300, max(bse.values())):
        raise RuntimeError(f"cov/bse inconsistency {d_cov:.3e}")
    diag = pd.DataFrame([r for r in diag_rows if r is not None])
    if "cos" in diag:
        q = diag["cos"].quantile([0.05, 0.5, 0.95])
        wc = diag["warm_vs_cold"].dropna()
        print(f"  [2s {s}] cos 5/50/95%: {q.iloc[0]:.4f}/{q.iloc[1]:.4f}/{q.iloc[2]:.4f} | "
              f"newton iters med {diag['newton_iters'].median():.0f} | "
              f"fail={counters['n_fail']} newton_fail={counters['n_newton_fail']} "
              f"cos_neg={counters['n_cos_neg']} vsd_fail={counters['n_vsd_fail']} "
              f"proj={counters['n_proj']}/{int(keep.sum())}"
              + (f" | warm-vs-cold max {wc.max():.2e} (n={len(wc)})" if len(wc) else "")
              + f" | {(_time.time()-t0)/60:.1f} min")
    res = dict(ame=ame_hat, bse=bse, pvalues=pv, bse_level=bse_l, pvalues_level=pv_l,
               band=_ame_band_from_draws(names, ame_hat, Dc, alpha=alpha,
                                         label=f"AME 2s [{s}]"),
               band_level=_ame_band_from_draws(names, ame_hat, Dl, alpha=alpha),
               cov=cov, per_draw_diag=diag, n_cl=int(W.shape[1]), B_used=int(keep.sum()),
               runtime_s=_time.time() - t0, **counters)
    if keep_draws:
        res["draws_cone"] = {k: draws_cone[k] for k in names}
        res["draws_level"] = {k: draws_level[k] for k in names}
    return res


def _ts_off_path(names, ame_fn0, ame_hat, b_full, IF_b_of, W_of, B, constrained, XtX0,
                 qp_A, qp_lb, qp_ub, n_basis, alpha):
    """The theta-frozen regression guard: `cluster_wild_bootstrap`'s loop body, on the same
    operands in the same order, with ONE projection warm cell shared across the two schemes and
    the conglomerate loop run first -- which is how the estimator produces bse then bse_time."""
    _warm = [b_full.copy()]

    def _proj(b_lin):
        out = project_to_shape(b_lin, XtX0, qp_A, qp_lb, qp_ub, K=n_basis, warm=_warm[0])[0]
        _warm[0] = out
        return out

    out = {}
    for s in ("congl", "quarter"):
        IF_b = IF_b_of[s]
        W = W_of[s]
        draws = {k: np.empty(B) for k in names}
        for ib in range(B):
            b = b_full + W[ib] @ IF_b
            if constrained:
                b = _proj(b)
            a = ame_fn0(b)
            for k in names:
                draws[k][ib] = a[k]
        bse = {k: float(np.std(draws[k], ddof=1)) for k in names}
        pv = {k: (float(2 * stats.norm.sf(abs(ame_hat[k]) / bse[k])) if bse[k] > 0 else 0.0)
              for k in names}
        Dc = np.column_stack([draws[k] for k in names])
        out[s] = dict(ame=ame_hat, bse=bse, pvalues=pv, bse_level=bse, pvalues_level=pv,
                      band=_ame_band_from_draws(names, ame_hat, Dc, alpha=alpha),
                      band_level=None, cov=np.cov(Dc, rowvar=False, ddof=1),
                      per_draw_diag=pd.DataFrame(), draws_cone=draws, draws_level=None,
                      n_cl=int(W.shape[1]), B_used=B, n_fail=0, n_newton_fail=0, n_cos_neg=0,
                      n_vsd_fail=0, n_proj=0, n_knot_tie=0, n_drop=0, runtime_s=np.nan)
    return out
