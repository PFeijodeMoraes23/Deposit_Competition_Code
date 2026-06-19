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

Inference mirrors estimation_3_sleep.py: AMEs with a numerical-Jacobian delta
method and the Imbens-Kolesar (2016) effective-cluster t correction.
References for the single index: Ichimura (1993); Klein & Spady (1993);
Robertson, Wright & Dykstra (1988, PAVA).
"""
from __future__ import annotations

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
    sizes = cluster_series.value_counts()
    mean_ng = np.mean(sizes)
    cv2 = (np.std(sizes) / mean_ng) ** 2 if mean_ng > 0 else 1.0
    return max(1.0, len(sizes) / (1 + cv2)), int(len(sizes))


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
def _nlls_resid(params, y_dm, X, Z, CF, entity_idx, link):
    K = X.shape[1]
    phi = link_cdf(X @ params[:K], link)
    yhat = phi * Z
    if CF.shape[1] > 0:
        yhat = yhat + CF @ params[K:]
    sums = np.bincount(entity_idx, weights=yhat)
    counts = np.bincount(entity_idx)
    yhat_dm = yhat - (sums / counts)[entity_idx]
    return y_dm - yhat_dm


def _generic_ame(theta_full, X, link, K, G, h=1e-5):
    theta_X = theta_full[:K]
    AME = np.zeros(K + G)
    for k in range(K):
        col = X[:, k]
        uniq = np.unique(col[~np.isnan(col)])
        is_dummy = (len(uniq) == 2) and (0.0 in uniq) and (1.0 in uniq)
        if is_dummy:
            X1 = X.copy(); X1[:, k] = 1.0
            X0 = X.copy(); X0[:, k] = 0.0
            AME[k] = np.mean(link_cdf(X1 @ theta_X, link) - link_cdf(X0 @ theta_X, link))
        else:
            Xp = X.copy(); Xp[:, k] += h
            Xm = X.copy(); Xm[:, k] -= h
            AME[k] = np.mean((link_cdf(Xp @ theta_X, link) - link_cdf(Xm @ theta_X, link)) / (2 * h))
    if G > 0:
        AME[K:] = theta_full[K:]
    return AME


def _generic_ame_cov(theta_full, cov_full, X, link, K, G, h=1e-5):
    AME = _generic_ame(theta_full, X, link, K, G)
    J = np.zeros((K + G, K + G))
    for i in range(K + G):
        tp = theta_full.copy(); tp[i] += h
        J[:, i] = (_generic_ame(tp, X, link, K, G) - AME) / h
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


def fit_nlls_link(df, state_cols, has_cf, link, loss="cauchy"):
    """NLLS sleepiness fit with link in {'logit','probit','uniform'}. Returns a
    NonLinearResults (params = AMEs for tables; params_native = index coefs for phi)."""
    CF_cols = ["v_hat_x_lagged_dep"] if has_cf else []
    cols = state_cols + ["deposit_balance", "nr_lagged_dep", "entity_id"]
    df_ss = df.dropna(subset=cols + CF_cols).copy()
    if len(df_ss) == 0:
        return None

    entities = df_ss["entity_id"].unique()
    emap = {e: i for i, e in enumerate(entities)}
    entity_idx = df_ss["entity_id"].map(emap).values
    y_dm = _demean_col(df_ss, "deposit_balance")
    X = df_ss[state_cols].values.astype(float)
    Z = df_ss["nr_lagged_dep"].values.astype(float)
    CF = df_ss[CF_cols].values.astype(float) if has_cf else np.empty((len(df_ss), 0), dtype=float)
    K, G = X.shape[1], CF.shape[1]

    if link == "uniform":
        init = _linear_warm_start(y_dm, X, Z, CF)          # bound imposed around the OLS fit
    else:
        init = np.zeros(K + G)

    res_lsq = least_squares(_nlls_resid, init, args=(y_dm, X, Z, CF, entity_idx, link),
                            method="trf", loss=loss)
    Jr = res_lsq.jac
    try:
        cov = np.linalg.pinv(Jr.T @ Jr) * (np.sum(res_lsq.fun ** 2) / (len(y_dm) - len(init)))
    except Exception:
        cov = np.eye(len(init))

    ame, bse, cov_ame = _generic_ame_cov(res_lsq.x, cov, X, link, K, G)
    idx = [f"interaction_{sv}" if sv != "constant" else "nr_lagged_dep" for sv in state_cols] + CF_cols
    ps = pd.Series(ame, index=idx)
    bs = pd.Series(bse, index=idx)
    tvals = ps / bs

    cl = df_ss["CodConglomeradoPrudencial"].astype(str)
    G_star, G_nominal = _G_star(cl)
    pvals = pd.Series(stats.t.sf(np.abs(tvals), df=G_star) * 2, index=idx)

    tss = float(np.sum((y_dm - y_dm.mean()) ** 2))
    rss = float(np.sum(res_lsq.fun ** 2))
    rsq = 1 - rss / tss if tss > 0 else np.nan
    ps_native = pd.Series(res_lsq.x, index=idx)
    print(f"  [NLLS-{link}] status={res_lsq.status} | nfev={res_lsq.nfev} | cost={res_lsq.cost:.4g}")
    return NonLinearResults(ps, bs, tvals, pvals, G_star, params_native=ps_native,
                            nobs=len(y_dm), rsquared=rsq, G_nominal=G_nominal,
                            cov_ame=cov_ame, link=link)


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


def fit_single_index(df, state_cols, has_cf, logit_res, degree=3):
    """Smooth cubic-sieve monotone single index in the logit-direction index.
    Returns NonLinearResults carrying average-derivative AMEs (params; tables),
    the index direction (params_native = logit native coefs), and the sieve
    params si_b/si_vmu/si_vsd (used by phi_from_native for Est6's phi)."""
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
    y_dm = work["_y"] - work.groupby("entity_id")["_y"].transform("mean")
    Xdm = work[design] - work.groupby("entity_id")[design].transform("mean")

    cl = df_ss["CodConglomeradoPrudencial"].astype(str)
    res = sm.OLS(y_dm.values, Xdm.values).fit(cov_type="cluster", cov_kwds={"groups": cl})
    b = np.asarray(res.params)[:degree + 1]
    cov_b = np.asarray(res.cov_params())[:degree + 1, :degree + 1]
    G_star, G_nominal = _G_star(cl)

    # average-derivative AMEs (reporting): AME_k = theta_k * mean_i G'(v_i)
    dG = np.zeros(len(vs))
    for d in range(1, degree + 1):
        dG += d * b[d] * vs ** (d - 1)
    dG /= vsd
    mean_slope = float(np.mean(dG))
    c = np.zeros(degree + 1)
    for d in range(1, degree + 1):
        c[d] = (d / vsd) * float(np.mean(vs ** (d - 1)))
    try:
        var_slope = float(c @ cov_b @ c)
    except Exception:
        var_slope = np.nan

    ame, bse, pv = {}, {}, {}
    for nm, th in zip(phi_params, theta):
        if nm == "nr_lagged_dep":
            continue
        ame[nm] = th * mean_slope
        se = abs(th) * np.sqrt(var_slope) if np.isfinite(var_slope) else np.nan
        bse[nm] = se
        pv[nm] = (float(stats.t.sf(abs(ame[nm] / se), df=G_star if G_star > 1 else 30) * 2)
                  if (se and np.isfinite(se) and se > 0) else np.nan)
    ps = pd.Series(ame); bs = pd.Series(bse); pvs = pd.Series(pv)
    tss = float(np.sum((y_dm.values - y_dm.values.mean()) ** 2))
    rss = float(np.sum(res.resid ** 2))
    rsq = 1 - rss / tss if tss > 0 else getattr(res, "rsquared", np.nan)
    print(f"  [SingleIndex] cubic sieve deg={degree} | mean_slope={mean_slope:.4g}")
    return NonLinearResults(ps, bs, ps / bs.replace(0, np.nan), pvs, G_star,
                            params_native=pd.Series(theta, index=phi_params),
                            nobs=len(df_ss), rsquared=rsq, G_nominal=G_nominal,
                            cov_ame=None, si_b=b, si_vmu=vmu, si_vsd=vsd, link="index")


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
    return np.clip(link_cdf(index, link), 0.0, 1.0)
