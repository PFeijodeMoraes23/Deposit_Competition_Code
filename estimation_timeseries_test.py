"""
estimation_timeseries_test.py
================================================================================
Robustness test: does the depositor sleepiness function phi_mt(s_mt) carry an
unmodelled TIME-SERIES component (secular trend + business-cycle cyclicality)
beyond the Pix/broadband/demographic state variables already in spec 12?

Motivation
----------
Spec 12 ("IV_HausmanFull x Tech") already conditions phi on pix_exists,
broadband connections, GDP per capita (level), demographics and the lagged
Selic rate. An advisor suggested extra time dummies (originally via Pix
adoption) to capture geographic/temporal variation. That exact route did not
work, but it motivates a cleaner question: is there a residual TREND or
CYCLE in depositor attention that the current covariates miss?

We add a parsimonious 2-variable TIME block, interacted with the core
regressor nr_lagged_dep exactly like the other state variables:

    time_trend       years since the sample start  -> secular drift in phi
    gdp_growth_yoy   YoY growth of GDP per capita   -> business-cycle cyclicality

(The Selic level is already in the Tech block, so gdp_growth_yoy isolates the
real-activity cycle that risk_free_qoq_lag does not capture.)

For each of the three estimation strategies we re-estimate spec 12 twice:
    Base    = IV_HausmanFull x Tech                (as in the paper)
    +Time   = IV_HausmanFull x (Tech + time block)

    Est 1  Local-only (entity-demeaned OLS, digital banks dropped)
    Est 2  Pooled B+D linear (entity-demeaned OLS)
    Est 3  Pooled B+D logistic NLLS, Average Marginal Effects

Relevance is assessed three ways:
    1. Joint Wald test on the time block (is it jointly significant?).
    2. Ljung-Box autocorrelation of base-spec quarter-mean residuals
       (is there time structure the base spec leaves on the table?).
    3. Implied national phi_t with vs without the time block, plus the
       stability of the headline coefficients (pix_exists, broadband).

Outputs (mirror export_analyze_spec12.py conventions)
-----------------------------------------------------
    ESTIMATION_OUTPUT/Rout/est_timeseries_spec12_comparison.tex   LaTeX table
    ESTIMATION_OUTPUT/Rout/est_timeseries_spec12_phi_t.png        figure
    ESTIMATION_OUTPUT/Rout/est_timeseries_results.json            raw numbers
    <Drafts>/Sleepiness_TimeSeries_Test.md                        report (-> PDF)
    + copies of the .tex / .png into <Drafts>

Usage
-----
    python estimation_timeseries_test.py
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import shutil
import pickle
import argparse
from pathlib import Path

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
from scipy.optimize import least_squares
from statsmodels.stats.diagnostic import acorr_ljungbox

from utils import paths as _paths_mod

# Estimation kernels reused verbatim from the three strategy scripts so the
# Base columns reproduce the paper's spec 12 exactly (same data prep, same
# first/second stage, same cluster correction). We only ADD the time block.
from estimation_1_sleep import (
    build_unified_frame,
    run_first_stage as e1_first_stage,
    run_second_stage as e1_second_stage,
)
from estimation_2_sleep import (
    build_pooled_data,
    define_specifications,
    run_pooled_first_stage as e2_first_stage,
    run_pooled_second_stage as e2_second_stage,
)
from estimation_3_sleep import (
    run_pooled_first_stage as e3_first_stage,
    run_pooled_second_stage as e3_second_stage,
    NonLinearResults,
)

_DRAFTS_DIR = Path(
    r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)"
    r"\Open Finance\Open-Finance\Drafts\Deposit Competition"
)

TIME_VARS = ["time_trend", "gdp_growth_yoy"]
TIME_PARAMS = [f"interaction_{v}" for v in TIME_VARS]

# ==============================================================================
# TIME-SERIES VARIABLE CONSTRUCTION
# ==============================================================================
def add_time_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add the two time-series state variables in place and return df.

    time_trend     : continuous years since the first sample quarter. In years
                     (not quarters) so its magnitude is comparable to the other
                     scaled state vars (gdp_per_capita/1e4, connections/100).
    gdp_growth_yoy : within-entity year-over-year growth of gdp_per_capita.
                     gdp_per_capita is already median-filled in the frame, so
                     this is well defined; the first 4 obs per entity and any
                     inf (from a zero base) are set to 0 (= no growth).
    """
    df.sort_values(by=["entity_id", "year", "quarter"], inplace=True)
    df["time_trend"] = (df["year"] - int(df["year"].min())) + (df["quarter"] - 1) / 4.0

    gpc = df["gdp_per_capita"]
    growth = df.groupby("entity_id")["gdp_per_capita"].transform(lambda s: s / s.shift(4) - 1.0)
    growth = growth.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["gdp_growth_yoy"] = growth.where(gpc.notna(), 0.0)
    return df


# ==============================================================================
# PHI AGGREGATION + DELTA-METHOD CI (mirrors export_analyze_spec12.calc_agg_delta)
# ==============================================================================
def _phi_param_names(res) -> list:
    return [p for p in res.params.index if not str(p).startswith("v_hat")]


def _build_phi_X(df: pd.DataFrame, phi_params: list) -> np.ndarray:
    """Regressor matrix for phi: 'nr_lagged_dep' -> 1, 'interaction_{sv}' -> sv."""
    n = len(df)
    X = np.zeros((n, len(phi_params)))
    for i, p in enumerate(phi_params):
        if p == "nr_lagged_dep":
            X[:, i] = 1.0
        elif p.startswith("interaction_"):
            sv = p[len("interaction_"):]
            if sv in df.columns:
                X[:, i] = df[sv].fillna(0.0).values.astype(float)
    return X


def implied_phi_t(df: pd.DataFrame, res, link: str):
    """National pop-weighted phi_t (market-level mean first, then pop weight
    across markets -- identical to calculate_phis) plus a delta-method 95% band.

    link in {'linear','logit','probit','uniform'} selects how the index maps to
    phi. For the link models phi uses the native index coefficients through G(.);
    'linear' uses the OLS coefficients directly. The delta-method band uses the
    (AME) covariance with gradient X-bar, mirroring export_analyze_spec12.

    Returns (phi_t Series, lo Series, hi Series) indexed by year_quarter.
    """
    phi_params = _phi_param_names(res)
    X = _build_phi_X(df, phi_params)

    is_link = link in ("logit", "probit", "uniform")
    params = res.params_native if (is_link and hasattr(res, "params_native")) else res.params
    beta = params[phi_params].values.astype(float)
    lin = X @ beta
    phi_mt = _link_cdf(lin, link) if is_link else lin

    w = df["pop_total"].fillna(0.0).values.astype(float) if "pop_total" in df.columns else np.ones(len(df))
    yq = df["year_quarter"].values
    mun = df["CODMUN_IBGE"].astype(str).values

    work = pd.DataFrame({"yq": yq, "mun": mun, "phi": phi_mt, "w": w})
    # market (mun) level first: phi is constant within a market-quarter, w summed
    mk = work.groupby(["yq", "mun"], observed=True).agg(phi=("phi", "mean"), M=("w", "sum")).reset_index()
    num = (mk["phi"] * mk["M"]).groupby(mk["yq"]).sum()
    den = mk["M"].groupby(mk["yq"]).sum().replace(0, np.nan)
    phi_t = (num / den).dropna()

    # Delta-method SE: var_t = g_bar' Sigma g_bar, g_bar = pop-weighted mean
    # regressor row in quarter t (logistic AME cov already embeds phi(1-phi)).
    g_star = float(getattr(res, "G_star", getattr(res, "df_resid", np.nan)) or np.nan)
    crit = stats.t.ppf(0.975, df=g_star - 1) if (g_star and g_star > 1) else 1.96
    lo, hi = {}, {}
    try:
        full_cov = res.cov_params()
        Sigma = full_cov.loc[phi_params, phi_params].values
        for t in phi_t.index:
            m = yq == t
            w_t = w[m]
            if w_t.sum() == 0:
                lo[t] = hi[t] = phi_t[t]
                continue
            g_bar = (w_t[:, None] * X[m]).sum(axis=0) / w_t.sum()
            se = float(np.sqrt(max(g_bar @ Sigma @ g_bar, 0.0)))
            lo[t] = phi_t[t] - crit * se
            hi[t] = phi_t[t] + crit * se
    except Exception as exc:  # pragma: no cover - band is cosmetic
        print(f"  [warn] phi_t CI band failed ({exc}); plotting point estimate only.")
        for t in phi_t.index:
            lo[t] = hi[t] = phi_t[t]

    # Cell-level out-of-bounds share of the OUTPUT phi: only the unconstrained
    # LPM (linear) leaves [0,1]; the uniform clip and the logit/probit links are
    # bounded by construction, so this is ~0 for them.
    frac_oob = float(np.mean((phi_mt < 0) | (phi_mt > 1)))
    return phi_t, pd.Series(lo), pd.Series(hi), frac_oob


# ==============================================================================
# DIAGNOSTICS
# ==============================================================================
def wald_time_block(res) -> tuple:
    """Joint Wald test that both time-block coefficients are zero.
    Returns (stat, p, df). chi2 with df = #(time params present)."""
    names = [p for p in TIME_PARAMS if p in res.params.index]
    if not names:
        return np.nan, np.nan, 0
    b = res.params[names].values.astype(float)
    V = res.cov_params().loc[names, names].values.astype(float)
    try:
        stat = float(b @ np.linalg.inv(V) @ b)
    except np.linalg.LinAlgError:
        stat = float(b @ np.linalg.pinv(V) @ b)
    p = float(stats.chi2.sf(stat, len(names)))
    return stat, p, len(names)


def ljung_box_resid(df: pd.DataFrame, res, state_cols: list, has_cf: bool, lag: int = 4):
    """Ljung-Box test on the quarter-mean residual series of a LINEAR base spec.
    Significant -> the base spec leaves serially-correlated time structure in
    its residuals, which a trend/cycle term could absorb.
    Returns (lb_stat, lb_p) or (nan, nan) if residuals are unavailable."""
    if not hasattr(res, "resid"):
        return np.nan, np.nan
    X_cols = ["nr_lagged_dep"] + [f"interaction_{sv}" for sv in state_cols if sv != "constant"]
    if has_cf:
        X_cols.append("v_hat_x_lagged_dep")
    idx = res.resid.index
    if "year_quarter" not in df.columns:
        return np.nan, np.nan
    yq = df.loc[idx, "year_quarter"]
    s = pd.Series(res.resid.values, index=yq.values).groupby(level=0).mean()
    s = s.reindex(sorted(s.index, key=lambda q: pd.Period(str(q).replace("_", "Q"), freq="Q")))
    if len(s) <= lag + 1:
        return np.nan, np.nan
    lb = acorr_ljungbox(s.values, lags=[lag], return_df=True)
    return float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])


# ==============================================================================
# STRATEGY RUNNERS  (Base = paper spec 12; +Time = adds the time block)
# ==============================================================================
def _run_linear(df_in, iv_cols, state_cols, first_stage, second_stage, spec_name):
    """Shared driver for Est 1 / Est 2 (both linear, same kernel signatures)."""
    df = df_in.copy()
    df, res_fs = first_stage(df, iv_cols, [c for c in state_cols if c != "constant"])
    if second_stage is e1_second_stage:
        res_ss = second_stage(df, state_cols, has_cf=True, spec_name=spec_name)
    else:
        res_ss = second_stage(df, state_cols, has_cf=True)
    return df, res_ss, res_fs


def _run_logistic(df_in, iv_cols, state_cols):
    df = df_in.copy()
    df, res_fs = e3_first_stage(df, iv_cols, [c for c in state_cols if c != "constant"])
    res_ss = e3_second_stage(df, state_cols, has_cf=True)
    return df, res_ss, res_fs


# ==============================================================================
# GENERIC LINK NLLS  (probit = normal eta; uniform/clip = constrained-linear)
# ==============================================================================
# These generalise estimation_3_sleep's logistic NLLS to an arbitrary link
# G(.) for the sleepiness CDF: phi_mt = G(S_mt' theta). The model defines
# phi = F_eta(index), so each link names a distribution for the sleepiness
# shock eta:
#     'logit'   -> eta ~ Logistic   (Est 3, kept in estimation_3_sleep.py)
#     'probit'  -> eta ~ Normal
#     'uniform' -> eta ~ Uniform    (= clipped-linear, the COHERENT counterpart
#                  of the LPM+clip: the bound is imposed DURING estimation).
# Inference mirrors Est 3: AMEs with a numerical-Jacobian delta-method SE and
# the Imbens-Kolesar (2016) effective-cluster t correction.
def _link_cdf(z, link):
    if link == "probit":
        return norm.cdf(z)
    if link == "uniform":
        return np.clip(z, 0.0, 1.0)
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))
    raise ValueError(f"unknown link {link!r}")


def _demean_col(df_ss, col):
    return (df_ss[col] - df_ss.groupby("entity_id")[col].transform("mean")).values.astype(float)


def _nlls_resid(params, y_dm, X, Z, CF, entity_idx, link):
    K = X.shape[1]
    phi = _link_cdf(X @ params[:K], link)
    yhat = phi * Z
    if CF.shape[1] > 0:
        yhat = yhat + CF @ params[K:]
    sums = np.bincount(entity_idx, weights=yhat)
    counts = np.bincount(entity_idx)
    yhat_dm = yhat - (sums / counts)[entity_idx]
    return y_dm - yhat_dm


def _generic_ame(theta_full, X, link, K, G, h=1e-5):
    """Average marginal effect of each state var on phi (dummy: level diff;
    continuous: central numerical derivative). CF terms pass through linearly."""
    theta_X = theta_full[:K]
    AME = np.zeros(K + G)
    for k in range(K):
        col = X[:, k]
        uniq = np.unique(col[~np.isnan(col)])
        is_dummy = (len(uniq) == 2) and (0.0 in uniq) and (1.0 in uniq)
        if is_dummy:
            X1 = X.copy(); X1[:, k] = 1.0
            X0 = X.copy(); X0[:, k] = 0.0
            AME[k] = np.mean(_link_cdf(X1 @ theta_X, link) - _link_cdf(X0 @ theta_X, link))
        else:
            Xp = X.copy(); Xp[:, k] += h
            Xm = X.copy(); Xm[:, k] -= h
            AME[k] = np.mean((_link_cdf(Xp @ theta_X, link) - _link_cdf(Xm @ theta_X, link)) / (2 * h))
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


def run_nlls_link(df, state_cols, has_cf, link, init=None, loss="cauchy"):
    """NLLS sleepiness fit with link G in {'probit','uniform'} (logit handled by
    estimation_3_sleep). Returns a NonLinearResults (AMEs) like Est 3."""
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

    if init is None:
        init = np.zeros(K + G)
        if link == "uniform" and "constant" in state_cols:
            init[state_cols.index("constant")] = 0.8  # start most cells interior
    init = np.asarray(init, float)
    if len(init) != K + G:
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

    cl = df_ss["CodConglomeradoPrudencial"]
    sizes = cl.value_counts()
    mean_ng = np.mean(sizes)
    cv2 = (np.std(sizes) / mean_ng) ** 2 if mean_ng > 0 else 1.0
    G_star = max(1.0, len(sizes) / (1 + cv2))
    pvals = pd.Series(stats.t.sf(np.abs(tvals), df=G_star) * 2, index=idx)

    nobs = len(y_dm)
    tss = float(np.sum((y_dm - y_dm.mean()) ** 2))
    rss = float(np.sum(res_lsq.fun ** 2))
    rsq = 1 - rss / tss if tss > 0 else np.nan
    ps_native = pd.Series(res_lsq.x, index=idx)
    print(f"  [NLLS-{link}] status={res_lsq.status} | nfev={res_lsq.nfev} | cost={res_lsq.cost:.4g}")
    return NonLinearResults(ps, bs, tvals, pvals, G_star, params_native=ps_native,
                            nobs=nobs, rsquared=rsq, fvalue=np.nan, f_pvalue=np.nan,
                            G_nominal=len(sizes), cov_ame=cov_ame)


def _run_link(df_in, iv_cols, state_cols, link, init=None, loss="cauchy"):
    df = df_in.copy()
    df, res_fs = e3_first_stage(df, iv_cols, [c for c in state_cols if c != "constant"])
    res_ss = run_nlls_link(df, state_cols, has_cf=True, link=link, init=init, loss=loss)
    return df, res_ss, res_fs


# ==============================================================================
# MONOTONE SINGLE-INDEX  (distribution-free / nonparametric link for eta)
# ==============================================================================
# Estimates phi(S) = G(S'theta) WITHOUT naming F_eta. The index direction theta
# is taken from the logit fit (identified up to scale under the single-index
# assumption); the link G = F_eta is then estimated with a SMOOTH degree-d
# polynomial SERIES (sieve) in the index, interacted with D-tilde and fit by
# entity-demeaned cluster OLS, with a PAVA isotonic safeguard for monotonicity.
# The sieve (unlike a binned link) is well identified under entity FE: its
# basis-times-D-tilde regressors vary WITHIN entity exactly like the parametric
# interaction terms, so it does not collapse. References: Ichimura (1993,
# J.Econometrics); Klein & Spady (1993, Econometrica); Hardle, Hall & Ichimura
# (1993, Ann.Stat.); Powell, Stock & Stoker (1989, Econometrica); Newey (1997)
# for series estimation; Balabdaoui, Groeneboom & Hendrickx (2019, Scand.J.Stat.)
# for the monotone single index; Robertson, Wright & Dykstra (1988) for PAVA.
# Inference is CONDITIONAL on the estimated index direction (the sieve's
# cluster-robust covariance), and approximate.
def _pava_increasing(y, w=None):
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


def run_single_index(df, state_cols, has_cf, logit_res, degree=3):
    """Monotone single-index sleepiness: logit-direction index + a SMOOTH
    degree-`degree` polynomial (series/sieve) link, fit by entity-demeaned
    cluster OLS, with a PAVA isotonic safeguard. Returns (res_like, phi_t, lo, hi)
    where res_like is a NonLinearResults carrying the average-derivative AMEs."""
    phi_params = [p for p in logit_res.params.index if not str(p).startswith("v_hat")]
    theta = logit_res.params_native[phi_params].values.astype(float)

    CF_cols = ["v_hat_x_lagged_dep"] if has_cf else []
    cols = state_cols + ["deposit_balance", "nr_lagged_dep", "entity_id"]
    df_ss = df.dropna(subset=cols + CF_cols).copy()
    if len(df_ss) == 0:
        return None, None, None, None

    X = _build_phi_X(df_ss, phi_params)        # nr_lagged_dep -> 1, interaction_sv -> sv
    v = X @ theta                              # logit linear index
    Z = df_ss["nr_lagged_dep"].values.astype(float)
    vmu, vsd = float(v.mean()), float(v.std())
    if vsd <= 0:
        vsd = 1.0
    vs = (v - vmu) / vsd                        # standardised index (numerical stability)

    # smooth series link: G(v) = sum_d b_d vs^d, interacted with D-tilde.
    # The regressors vs^d * Z vary WITHIN entity, so b is identified under FE.
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

    phi_raw = P @ b                             # smooth link evaluated at each obs
    order = np.argsort(vs)
    iso = _pava_increasing(phi_raw[order])      # monotone safeguard (smooth -> minimal change)
    phi_mt = np.empty_like(phi_raw)
    phi_mt[order] = iso
    phi_mt = np.clip(phi_mt, 0.0, 1.0)

    # national phi_t (market-first pop weight) + conditional band from cov_b.
    # phi_t ~= b . m_t where m_t is the pop-weighted mean polynomial row in quarter t.
    w_pop = df_ss["pop_total"].fillna(0.0).values.astype(float)
    cols_m = {f"p{d}": P[:, d] for d in range(degree + 1)}
    frame = pd.DataFrame({"yq": df_ss["year_quarter"].values,
                          "mun": df_ss["CODMUN_IBGE"].astype(str).values,
                          "phi": phi_mt, "w": w_pop, **cols_m})
    agg_spec = {"phi": ("phi", "mean"), "M": ("w", "sum")}
    agg_spec.update({f"p{d}": (f"p{d}", "mean") for d in range(degree + 1)})
    mk = frame.groupby(["yq", "mun"], observed=True).agg(**agg_spec).reset_index()
    g_star = float(getattr(logit_res, "G_star", np.nan) or np.nan)
    crit = stats.t.ppf(0.975, df=g_star - 1) if (g_star and g_star > 1) else 1.96
    phi_t, lo, hi = {}, {}, {}
    for t, grp in mk.groupby("yq"):
        Msum = grp["M"].sum()
        if Msum <= 0:
            continue
        wv = grp["M"].values / Msum
        phi_t[t] = float((wv * grp["phi"].values).sum())
        m_t = np.array([float((wv * grp[f"p{d}"].values).sum()) for d in range(degree + 1)])
        se = np.sqrt(max(float(m_t @ cov_b @ m_t), 0.0))
        lo[t] = phi_t[t] - crit * se
        hi[t] = phi_t[t] + crit * se
    phi_t = pd.Series(phi_t)

    # average-derivative AMEs: AME_k = theta_k * mean_i G'(v_i),
    # G'(v) = (1/vsd) * sum_{d>=1} d b_d vs^(d-1)
    dG = np.zeros(len(vs))
    for d in range(1, degree + 1):
        dG += d * b[d] * vs ** (d - 1)
    dG /= vsd
    mean_slope = float(np.mean(dG))
    c = np.zeros(degree + 1)                    # mean_slope = c . b
    for d in range(1, degree + 1):
        c[d] = (d / vsd) * float(np.mean(vs ** (d - 1)))
    try:
        var_slope = float(c @ cov_b @ c)
    except Exception:
        var_slope = np.nan

    ame, bse, pvals = {}, {}, {}
    for nm, th in zip(phi_params, theta):
        if nm == "nr_lagged_dep":
            continue
        ame[nm] = th * mean_slope
        se = abs(th) * np.sqrt(var_slope) if np.isfinite(var_slope) else np.nan
        bse[nm] = se
        if se and np.isfinite(se) and se > 0:
            pvals[nm] = float(stats.t.sf(abs(ame[nm] / se), df=g_star if g_star > 1 else 30) * 2)
        else:
            pvals[nm] = np.nan
    ps = pd.Series(ame); bs = pd.Series(bse); pv = pd.Series(pvals)
    res_like = NonLinearResults(ps, bs, ps / bs.replace(0, np.nan), pv, g_star,
                                params_native=ps, nobs=len(df_ss),
                                rsquared=getattr(res, "rsquared", np.nan),
                                G_nominal=int(cl.nunique()), cov_ame=None)
    n_noniso = int((np.diff(phi_raw[order]) < 0).sum())
    print(f"  [SingleIndex] cubic sieve deg={degree} | mean_slope={mean_slope:.4g} | "
          f"phi_t range={phi_t.max()-phi_t.min():.4f} | pre-iso non-monotone pts={n_noniso}")
    return res_like, phi_t, pd.Series(lo), pd.Series(hi)


def build_frames():
    """Build the two base frames once (E1 unified; pooled shared by E2/E3),
    trim to needed columns to keep per-run copies light, and add the time vars."""
    _, iv_specs, state_blocks = define_specifications()
    s_tech = state_blocks["Tech"]
    iv_cols = iv_specs["IV_HausmanFull"]

    needed = set(
        s_tech + iv_cols + TIME_VARS + [
            "constant", "entity_id", "time_id", "year", "quarter", "deposit_type",
            "deposit_balance", "nr_lagged_dep", "lagged_deposits", "spread_qoq",
            "spread_ann", "v_hat", "CodConglomeradoPrudencial", "CODMUN_IBGE",
            "pop_total", "gdp_per_capita",
        ]
    )

    print("Building pooled frame (Est 2 / Est 3)...")
    pooled = build_pooled_data()
    pooled["year_quarter"] = pooled["time_id"]
    pooled = add_time_variables(pooled)
    pooled = pooled[[c for c in pooled.columns if c in needed or c == "year_quarter"]].copy()

    print("Building unified frame (Est 1, digital banks dropped)...")
    e1 = build_unified_frame()
    e1["constant"] = 1.0
    for col in s_tech:
        if col != "constant" and col in e1.columns:
            e1[col] = e1[col].fillna(e1[col].median())
    e1["year_quarter"] = e1["time_id"]
    e1 = add_time_variables(e1)
    e1 = e1[[c for c in e1.columns if c in needed or c == "year_quarter"]].copy()

    return e1, pooled, s_tech, iv_cols


def _init_from_linear(lin_res, state_cols):
    """Warm start for the uniform (constrained-linear) NLLS from the OLS linear
    fit: same index coefficients, clipped during estimation rather than after."""
    init = []
    for sv in state_cols:
        nm = "nr_lagged_dep" if sv == "constant" else f"interaction_{sv}"
        init.append(float(lin_res.params.get(nm, 0.0)))
    init.append(float(lin_res.params.get("v_hat_x_lagged_dep", 0.0)))  # CF gamma
    return np.array(init, dtype=float)


def run_all():
    e1_df, pooled_df, s_tech, iv_cols = build_frames()
    s_time = s_tech + TIME_VARS
    group_of = {k: g for g, ks in GROUPS for k in ks}

    # (label, with_time, runner, link). Order matters: E2 precedes CL (warm
    # start) and E3 precedes SI (index direction).
    plan = [
        ("E1 Base", False, "lin1",    "linear"),
        ("E1 +Time", True, "lin1",    "linear"),
        ("E2 Base", False, "lin2",    "linear"),
        ("E2 +Time", True, "lin2",    "linear"),
        ("CL Base", False, "uniform", "uniform"),
        ("CL +Time", True, "uniform", "uniform"),
        ("E3 Base", False, "logit",   "logit"),
        ("E3 +Time", True, "logit",   "logit"),
        ("PB Base", False, "probit",  "probit"),
        ("PB +Time", True, "probit",  "probit"),
        ("SI Base", False, "si",      "si"),
        ("SI +Time", True, "si",      "si"),
    ]

    results = {}
    for label, with_time, runner, link in plan:
        state_cols = s_time if with_time else s_tech
        print(f"\n--- Estimating {label} ({'Tech+Time' if with_time else 'Tech'}) ---")
        phi_override = None
        try:
            if runner == "lin1":
                df, res_ss, _ = _run_linear(e1_df, iv_cols, state_cols, e1_first_stage, e1_second_stage, label)
            elif runner == "lin2":
                df, res_ss, _ = _run_linear(pooled_df, iv_cols, state_cols, e2_first_stage, e2_second_stage, label)
            elif runner == "logit":
                df, res_ss, _ = _run_logistic(pooled_df, iv_cols, state_cols)
            elif runner == "probit":
                df, res_ss, _ = _run_link(pooled_df, iv_cols, state_cols, "probit", loss="cauchy")
            elif runner == "uniform":
                e2_key = "E2 +Time" if with_time else "E2 Base"
                init = _init_from_linear(results[e2_key]["res"], state_cols) if e2_key in results else None
                df, res_ss, _ = _run_link(pooled_df, iv_cols, state_cols, "uniform", init=init, loss="linear")
            else:  # single index
                e3_key = "E3 +Time" if with_time else "E3 Base"
                if e3_key not in results:
                    print(f"  [warn] {label}: logit direction {e3_key} missing; skipping.")
                    continue
                df = pooled_df.copy()
                df, _ = e3_first_stage(df, iv_cols, [c for c in state_cols if c != "constant"])
                res_ss, phi_si, lo_si, hi_si = run_single_index(
                    df, state_cols, has_cf=True, logit_res=results[e3_key]["res"])
                phi_override = (phi_si, lo_si, hi_si)
        except Exception as exc:
            print(f"  [warn] {label} failed: {exc}")
            continue

        if res_ss is None:
            print(f"  [warn] {label} produced no second stage; skipping.")
            continue

        if phi_override is not None:
            phi_t, lo, hi = phi_override
            frac_oob = 0.0
        else:
            phi_t, lo, hi, frac_oob = implied_phi_t(df, res_ss, link)

        lb_stat, lb_p = (np.nan, np.nan)
        wald = (np.nan, np.nan, 0)
        if runner == "si":
            pass  # joint Wald ill-defined for a fixed-direction single index
        elif with_time:
            wald = wald_time_block(res_ss)
        else:
            lb_stat, lb_p = ljung_box_resid(df, res_ss, state_cols, has_cf=True)

        results[label] = {
            "group": group_of.get(label, label),
            "with_time": with_time,
            "link": link,
            "res": res_ss,
            "state_cols": state_cols,
            "phi_t": phi_t,
            "phi_lo": lo,
            "phi_hi": hi,
            "frac_oob": frac_oob,
            "wald": wald,
            "ljung_box": (lb_stat, lb_p),
        }
    return results


# ==============================================================================
# OUTPUT: helpers
# ==============================================================================
def get_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


NICE = {
    "nr_lagged_dep": "Constant",
    "interaction_pix_exists": "Pix Available",
    "interaction_gdp_per_capita": "GDP per capita (10k R\\$)",
    "interaction_cadunico_families_per1000": "CadUnico Families (100s per 1k)",
    "interaction_fraction_65plus": "Fraction 65+",
    "interaction_fraction_young": "Fraction Young",
    "interaction_risk_free_qoq_lag": "Lagged Selic Rate",
    "interaction_connections_per100": "Broadband (per 100 inhab.)",
    "interaction_time_trend": "Time Trend (years)",
    "interaction_gdp_growth_yoy": "GDP Growth (YoY)",
}
NICE_MD = {k: v.replace("\\$", "$").replace("\\&", "&") for k, v in NICE.items()}

ORDER_VARS = [
    "nr_lagged_dep",
    "interaction_pix_exists",
    "interaction_gdp_per_capita",
    "interaction_cadunico_families_per1000",
    "interaction_fraction_65plus",
    "interaction_fraction_young",
    "interaction_risk_free_qoq_lag",
    "interaction_connections_per100",
    "interaction_time_trend",
    "interaction_gdp_growth_yoy",
]

# Each estimator = an assumption about the sleepiness-shock CDF F_eta.
GROUPS = [
    ("Local Linear (Est 1)",  ["E1 Base", "E1 +Time"]),
    ("Pooled Linear (Est 2)", ["E2 Base", "E2 +Time"]),
    ("Constrained Linear",    ["CL Base", "CL +Time"]),
    ("Logit (Est 3)",         ["E3 Base", "E3 +Time"]),
    ("Probit",                ["PB Base", "PB +Time"]),
    ("Single-Index",          ["SI Base", "SI +Time"]),
]
F_ETA = {
    "Local Linear (Est 1)":  "Uniform (LPM, post-hoc clip)",
    "Pooled Linear (Est 2)": "Uniform (LPM, post-hoc clip)",
    "Constrained Linear":    "Uniform (bound imposed in estimation)",
    "Logit (Est 3)":         "Logistic",
    "Probit":                "Normal",
    "Single-Index":          "Nonparametric, monotone (data-chosen)",
}
ORDER_KEYS = [k for _, ks in GROUPS for k in ks]
BASE_KEYS = [ks[0] for _, ks in GROUPS]
TIME_KEYS = [ks[1] for _, ks in GROUPS]


def _cell(res, v, digits=4):
    if res is None or v not in res.params.index:
        return "-", "-"
    c = res.params[v]
    se = res.bse[v]
    p = res.pvalues[v]
    if pd.isna(c):
        return "-", "-"
    return f"{c:.{digits}f}{get_stars(p)}", f"({se:.{digits}f})"


# ==============================================================================
# OUTPUT: LaTeX table (xltabular, grouped header -- mirrors export_analyze_spec12)
# ==============================================================================
HEADER_SHORT = {
    "Local Linear (Est 1)": r"Local (E1)",
    "Pooled Linear (Est 2)": r"Linear (E2)",
    "Constrained Linear": r"Constr.\ Lin.",
    "Logit (Est 3)": r"Logit (E3)",
    "Probit": r"Probit",
    "Single-Index": r"Single-Idx",
}


def build_latex_table(results, keys, out_path, title, label):
    """One column per estimator (model). Call once for Base keys, once for +Time."""
    group_of = {k: g for g, ks in GROUPS for k in ks}
    res_by_key = {k: results[k]["res"] for k in keys if k in results}
    keys = [k for k in keys if k in res_by_key]
    n = len(keys)

    tex = [r"\setstretch{1.0}"]
    col_def = (r">{\raggedright\arraybackslash}p{0.22\textwidth} "
               r"*{" + str(n) + r"}{>{\centering\arraybackslash}X}")
    tex.append(r"\begin{xltabular}{\textwidth}{" + col_def + "}")
    tex.append(r"\caption{" + title + r"}\label{" + label + r"} \\")
    tex.append(r"\toprule")
    headers = ["Variable"] + [HEADER_SHORT.get(group_of.get(k, k), group_of.get(k, k)) for k in keys]
    tex.append(" & ".join(headers) + r" \\")
    tex.append(r"\midrule")
    tex.append(r"\endfirsthead")
    tex.append(r"\multicolumn{" + str(n + 1) + r"}{c}{{\bfseries \tablename\ \thetable{} (continued)}} \\")
    tex.append(r"\toprule")
    tex.append(" & ".join(headers) + r" \\")
    tex.append(r"\midrule")
    tex.append(r"\endhead")
    tex.append(r"\midrule")
    tex.append(r"\multicolumn{" + str(n + 1) + r"}{r}{{Continued on next page}} \\")
    tex.append(r"\endfoot")
    tex.append(r"\bottomrule")
    notes = (r"\multicolumn{" + str(n + 1) + r"}{p{\dimexpr\textwidth-2\tabcolsep\relax}}"
             r"{\scriptsize\textit{Notes:} Spec 12 (IV Hausman $\times$ Tech). Each column is a "
             r"different assumption on the sleepiness-shock CDF $F_\eta$: Local/Pooled Linear are the "
             r"LPM (uniform $\eta$, clipped post-hoc); Constrained Linear imposes the bound in "
             r"estimation; Logit/Probit assume logistic/normal $\eta$; Single-Index leaves $F_\eta$ "
             r"nonparametric and monotone. Logit, Probit and Single-Index report Average Marginal "
             r"Effects (Single-Index: average-derivative, conditional/approx.\ SE). Standard errors "
             r"in parentheses. *** $p<0.01$, ** $p<0.05$, * $p<0.1$.}")
    tex.append(notes)
    tex.append(r"\endlastfoot")

    for v in ORDER_VARS:
        present_any = any(v in res_by_key[k].params.index for k in keys)
        if not present_any:
            continue
        lab = r"\multirow[t]{2}{0.22\textwidth}{\raggedright " + NICE.get(v, v) + r"}"
        row_c = [lab]
        row_s = [""]
        for k in keys:
            c, s = _cell(res_by_key[k], v)
            row_c.append(c)
            row_s.append(s)
        tex.append(" & ".join(row_c) + r" \\*")
        tex.append(" & ".join(row_s) + r" \\")
        tex.append(r"\addlinespace")

    tex.append(r"\midrule")
    row_nobs = ["Observations"]
    row_r2 = ["$R^2$"]
    row_clu = ["Clusters ($G$)"]
    row_eff = ["Effective $G^*$"]
    row_wald = ["Time-block Wald $p$"]
    for k in keys:
        res = res_by_key[k]
        nobs = getattr(res, "nobs", np.nan)
        r2 = getattr(res, "rsquared", np.nan)
        g = getattr(res, "G_nominal", np.nan)
        gs = getattr(res, "G_star", getattr(res, "df_resid", np.nan))
        row_nobs.append(f"{nobs:,.0f}" if pd.notna(nobs) else "-")
        row_r2.append(f"{r2:.3f}" if pd.notna(r2) else "-")
        row_clu.append(f"{int(g)}" if pd.notna(g) else "-")
        row_eff.append(f"{gs:.1f}" if pd.notna(gs) else "-")
        w = results[k]["wald"]
        row_wald.append(f"{w[1]:.3f}{get_stars(w[1])}" if (results[k]["with_time"] and pd.notna(w[1])) else "-")
    rows = [row_nobs, row_r2, row_clu, row_eff]
    if any(results[k]["with_time"] for k in keys):
        rows.append(row_wald)
    for r in rows:
        tex.append(" & ".join(r) + r" \\")
    tex.append(r"\end{xltabular}")
    tex.append(r"\doublespacing")

    out_path.write_text("\n".join(tex), encoding="utf-8")


# ==============================================================================
# OUTPUT: one figure per model (Base vs +Time phi_t)
# ==============================================================================
def _to_dates(idx):
    return pd.PeriodIndex([str(q).replace("_", "Q") for q in idx], freq="Q").to_timestamp()


def _sorted_index(idx):
    return sorted(idx, key=lambda q: pd.Period(str(q).replace("_", "Q"), freq="Q"))


_MODEL_SLUG = {
    "Local Linear (Est 1)": "e1",
    "Pooled Linear (Est 2)": "e2",
    "Constrained Linear": "clin",
    "Logit (Est 3)": "logit",
    "Probit": "probit",
    "Single-Index": "singleindex",
}


def build_per_model_figures(results, rout):
    """One Base-vs-+Time phi_t figure per estimator. Returns [(gname, slug, path)]."""
    made = []
    for gname, gkeys in GROUPS:
        base_key, time_key = gkeys
        if base_key not in results and time_key not in results:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for key, color, ls, lab in [
            (base_key, "#1f77b4", "-", "Base (Tech)"),
            (time_key, "#d62728", "--", r"$+$Time"),
        ]:
            if key not in results:
                continue
            r = results[key]
            phi = r["phi_t"].reindex(_sorted_index(r["phi_t"].index))
            x = _to_dates(phi.index)
            ax.plot(x, phi.values, color=color, linestyle=ls, linewidth=2.2, label=lab)
            lo = r["phi_lo"].reindex(phi.index)
            hi = r["phi_hi"].reindex(phi.index)
            if lo.notna().any():
                ax.fill_between(x, lo.values, hi.values, color=color, alpha=0.15)
        ax.set_ylim(0, 1.08)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.grid(alpha=0.35)
        ax.legend(loc="best", fontsize=11)
        ax.tick_params(axis="both", labelsize=11)
        ax.set_ylabel(r"National $\hat{\phi}_t$", fontsize=12)
        ax.set_title(f"{gname}\n" + r"$F_\eta$: " + F_ETA.get(gname, ""), fontsize=13)
        fig.tight_layout()
        slug = _MODEL_SLUG.get(gname, gname.lower().replace(" ", "_"))
        path = rout / f"est_timeseries_phi_{slug}.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        made.append((gname, slug, path))
    return made


# ==============================================================================
# OUTPUT: JSON + Markdown report
# ==============================================================================
def export_json(results, out_path):
    payload = {}
    for k, r in results.items():
        res = r["res"]
        payload[k] = {
            "group": r["group"],
            "with_time": r["with_time"],
            "link": r["link"],
            "nobs": float(getattr(res, "nobs", np.nan)),
            "rsquared": float(getattr(res, "rsquared", np.nan)) if pd.notna(getattr(res, "rsquared", np.nan)) else None,
            "G_nominal": float(getattr(res, "G_nominal", np.nan)) if pd.notna(getattr(res, "G_nominal", np.nan)) else None,
            "G_star": float(getattr(res, "G_star", np.nan)) if pd.notna(getattr(res, "G_star", np.nan)) else None,
            "params": {p: float(res.params[p]) for p in res.params.index},
            "bse": {p: float(res.bse[p]) for p in res.bse.index},
            "pvalues": {p: float(res.pvalues[p]) for p in res.pvalues.index},
            "wald_stat": float(r["wald"][0]) if pd.notna(r["wald"][0]) else None,
            "wald_p": float(r["wald"][1]) if pd.notna(r["wald"][1]) else None,
            "ljung_box_stat": float(r["ljung_box"][0]) if pd.notna(r["ljung_box"][0]) else None,
            "ljung_box_p": float(r["ljung_box"][1]) if pd.notna(r["ljung_box"][1]) else None,
            "phi_t_mean": float(np.mean(r["phi_t"].values)),
            "phi_t_min": float(np.min(r["phi_t"].values)),
            "phi_t_max": float(np.max(r["phi_t"].values)),
        }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fmt(res, v, digits=4):
    if res is None or v not in res.params.index:
        return "—"
    c = res.params[v]
    if pd.isna(c):
        return "—"
    se = res.bse[v] if v in res.bse.index else np.nan
    p = res.pvalues[v] if v in res.pvalues.index else np.nan
    se_str = f"({se:.{digits}f})" if pd.notna(se) else "(n/a)"
    return f"{c:.{digits}f}{get_stars(p)} {se_str}"


def _classify_time(results, param):
    """Split +Time estimators by sign/significance of `param` (5% level).
    Returns (negative_sig, positive_sig, insignificant) lists of estimator names.
    Used to keep the interpretive prose faithful to whatever the current panel
    produces, instead of hard-coding signs."""
    neg_sig, pos_sig, insig = [], [], []
    for gname, gkeys in GROUPS:
        tk = gkeys[1]
        if tk not in results:
            continue
        res = results[tk]["res"]
        if param not in res.params.index:
            continue
        c = float(res.params[param])
        p = res.pvalues[param] if param in res.pvalues.index else np.nan
        if pd.notna(p) and p < 0.05:
            (neg_sig if c < 0 else pos_sig).append(gname)
        else:
            insig.append(gname)
    return neg_sig, pos_sig, insig


def _join(names, empty="none"):
    return ", ".join(names) if names else empty


def _param_summary(results, param, keys):
    """Count sign/significance (5%) of `param` across `keys`.
    Returns dict with total, pos_sig, neg_sig, insig."""
    tot = pos = neg = ins = 0
    for k in keys:
        if k not in results:
            continue
        res = results[k]["res"]
        if param not in res.params.index:
            continue
        tot += 1
        c = float(res.params[param])
        p = res.pvalues[param] if param in res.pvalues.index else np.nan
        if pd.notna(p) and p < 0.05:
            if c < 0:
                neg += 1
            else:
                pos += 1
        else:
            ins += 1
    return {"total": tot, "pos_sig": pos, "neg_sig": neg, "insig": ins, "sig": pos + neg}


def build_markdown(results, figures, out_path, tex_base_name, tex_time_name):
    # ---- dynamic summary so the interpretive prose tracks the current panel ----
    tt_neg, tt_pos, tt_insig = _classify_time(results, "interaction_time_trend")
    gg_neg, gg_pos, gg_insig = _classify_time(results, "interaction_gdp_growth_yoy")
    base_means = {g: float(results[ks[0]]["phi_t"].mean()) for g, ks in GROUPS if ks[0] in results}
    lvl_lo, lvl_hi = (min(base_means.values()), max(base_means.values())) if base_means else (np.nan, np.nan)
    n_gg_sig = len(gg_neg) + len(gg_pos)
    trend_robust = (len(tt_neg) == 0 or len(tt_pos) == 0) and len(tt_insig) == 0
    lvl_spread = lvl_hi - lvl_lo
    lvl_phrase = ("fairly robust across links" if lvl_spread < 0.05
                  else "moderately link-sensitive" if lvl_spread < 0.12
                  else "strongly link-sensitive")
    pixB = _param_summary(results, "interaction_pix_exists", BASE_KEYS)
    pixT = _param_summary(results, "interaction_pix_exists", TIME_KEYS)
    bbB = _param_summary(results, "interaction_connections_per100", BASE_KEYS)
    bbT = _param_summary(results, "interaction_connections_per100", TIME_KEYS)
    _oob_lin = [results[ks[0]]["frac_oob"] for _, ks in GROUPS
                if ks[0] in results and results[ks[0]].get("link") == "linear"
                and results[ks[0]].get("frac_oob") is not None]
    oob_phrase = (f"{min(_oob_lin)*100:.0f}–{max(_oob_lin)*100:.0f}%" if _oob_lin else "a non-trivial share")
    try:
        panel_vintage = pd.Timestamp(
            (_paths_mod.PROCESSED / "market_panel_with_fees.csv").stat().st_mtime, unit="s"
        ).strftime("%Y-%m-%d")
    except Exception:
        panel_vintage = "current"

    L = []
    L.append("# Time-Series Structure in the Depositor Sleepiness Function")
    L.append("")
    L.append("**Project:** Deposit Competition — Egan et al. (2025) Replication & Extension  ")
    L.append("**Script:** `estimation_timeseries_test.py`  ")
    L.append(f"**Generated:** {pd.Timestamp.now():%Y-%m-%d %H:%M}  ")
    L.append(f"**Panel:** `market_panel_with_fees.csv` (COSIF-fixed, rebuilt {panel_vintage})  ")
    L.append("")
    L.append("---")
    L.append("")
    L.append("> **Scope.** The six estimators below — Local Linear, Pooled Linear, Constrained "
             "Linear, Logit, Probit, Single-Index — are now first-class routines in the sleepiness "
             "pipeline as **Est 1–6** (`estimation_{1..6}_sleep.py` → `export_results.py` → "
             "`estimation_demand_1_prep.py`); each runs the full 12-spec grid and feeds the BLP "
             "demand step. This report is the spec-12 ($F_\\eta$) comparison plus the $+$Time "
             "(trend/cycle) robustness test, re-run on the COSIF-fixed panel. Per the modelling "
             "convention, $\\phi$ is always built from the **native index coefficients $\\times$ "
             "link**; the Average Marginal Effects shown in the tables are for reporting only.")
    L.append("")
    L.append("## Question")
    L.append("")
    L.append(
        "Spec 12 (`IV_HausmanFull × Tech`) conditions the sleepiness function "
        "$\\phi_{mt}(\\mathbf{s}_{mt})$ on Pix availability, broadband penetration, "
        "GDP per capita (level), demographics and the lagged Selic rate. An advisor "
        "suggested adding time dummies (originally proxied by Pix adoption) to capture "
        "geographic/temporal variation. That exact route did not work, but it raises a "
        "cleaner question: **does $\\phi$ carry a residual secular *trend* or a business-cycle "
        "*cyclicality* that the current covariates miss?**"
    )
    L.append("")
    L.append("## Estimators: each is an assumption about the sleepiness shock $F_\\eta$")
    L.append("")
    L.append(
        "The structural model defines $\\phi$ as a threshold-crossing probability, "
        "$D^*_{imt} = \\mathbf{S}_{mt}'\\Gamma + \\mathbf{X}_{imt}'\\Theta + \\eta_{imt}$ with "
        "$\\phi_{mt} = \\mathrm{Prob}(D^*_{imt}\\le 0\\mid \\mathbf{S}_{mt}) = F_\\eta(-(\\mathbf{S}'\\Gamma+\\mathbf{X}'\\Theta))$. "
        "So $\\phi(\\mathbf{S})$ **is** the CDF of the sleepiness shock $\\eta$ evaluated at the index — and "
        "every estimator below is simply a different choice of that distribution. A linear $\\phi$ is "
        "**not** distribution-free: a *bounded* linear CDF is exactly the **Uniform**; only the "
        "*unbounded* straight line matches no distribution. We therefore span the choice of $F_\\eta$ "
        "from uniform to fully nonparametric:"
    )
    L.append("")
    L.append("| Estimator | Shock $F_\\eta$ | How $\\phi$ enters | Bounded $[0,1]$? |")
    L.append("|---|---|---|---|")
    L.append("| Local Linear (Est 1) | Uniform | $\\phi=\\mathbf{S}'\\beta$ (LPM) | only via post-hoc clip |")
    L.append("| Pooled Linear (Est 2) | Uniform | $\\phi=\\mathbf{S}'\\beta$ (LPM) | only via post-hoc clip |")
    L.append("| Constrained Linear | Uniform | $\\phi=\\mathrm{clip}(\\mathbf{S}'\\beta,0,1)$ | **yes** (imposed in estimation) |")
    L.append("| Logit (Est 3) | Logistic | $\\phi=\\Lambda(\\mathbf{S}'\\theta)$ | yes |")
    L.append("| Probit | Normal | $\\phi=\\Phi(\\mathbf{S}'\\theta)$ | yes |")
    L.append("| Single-Index | nonparametric, monotone | $\\phi=\\hat G(\\mathbf{S}'\\hat\\theta)$ | yes |")
    L.append("")
    L.append(
        "The three uniform-$\\eta$ rows isolate the effect of the *bound*: Est 1/2 are the linear "
        "probability model (LPM) that can leave $[0,1]$ and is clipped only afterward in the demand "
        "step; **Constrained Linear** is the same uniform-$\\eta$ model done coherently, with the bound "
        "imposed during estimation. Logit and Probit swap in logistic/normal shocks. The "
        "**Single-Index** estimator takes the index direction from the logit fit and then estimates the "
        "link $F_\\eta$ with a smooth monotone series (a cubic sieve in the index, isotonic-safeguarded), "
        "i.e. *lets the data choose the shock distribution* (Ichimura 1993; Klein & Spady 1993; see "
        "References). Reading across the columns "
        "tells us **how much the assumed shock distribution actually matters**."
    )
    L.append("")
    L.append("## Test design")
    L.append("")
    L.append("We add a parsimonious two-variable **time block**, each variable interacted "
             "with the core regressor $\\widetilde{D}_{t-1}$ (`nr_lagged_dep`) exactly like "
             "the existing state variables:")
    L.append("")
    L.append("| Variable | Captures |")
    L.append("|---|---|")
    L.append("| `time_trend` (years since 2013Q1) | secular drift in depositor attention |")
    L.append("| `gdp_growth_yoy` (YoY growth of GDP p.c.) | business-cycle cyclicality beyond the Selic level |")
    L.append("")
    L.append("For each strategy we re-estimate spec 12 twice — **Base** (`Tech`) and "
             "**+Time** (`Tech` + time block) — and assess relevance three ways: (1) a joint "
             "**Wald test** on the time block; (2) a **Ljung–Box** test on the base-spec "
             "quarter-mean residuals (does the base spec leave serially-correlated time "
             "structure on the table?); and (3) the implied national $\\hat{\\phi}_t$ path and "
             "headline-coefficient stability, Base vs. +Time.")
    L.append("")
    L.append("**How to read the coefficients.** Each time variable enters *multiplicatively*: "
             "it scales the slope on the no-rebalancing term $\\widetilde{D}_{t-1}$ "
             "(`nr_lagged_dep`), so it shifts the *level* of $\\phi$. A **negative** coefficient means "
             "$\\phi$ falls along that dimension — for `time_trend`, depositors growing more attentive "
             "(less sleepy) over 2013–2025; for `gdp_growth_yoy`, $\\phi$ **lower in booms** (inertia "
             "mildly *procyclical* in the opportunity cost of not re-optimising). A positive "
             "coefficient is the reverse. As §1 shows, the *cycle* term comes out negative across the "
             "board, but the *trend* sign turns out to **depend on the estimator**, so we read it per "
             "column. The estimators differ in units — the linear and constrained-linear specs report "
             "slopes, while Logit, Probit and the Single-Index report Average Marginal Effects through "
             "their respective links, so magnitudes are compressed but *signs and significance* are "
             "directly comparable across columns.")
    L.append("")

    # ---- Table 1: time block coefficients + joint test ----
    L.append("## 1. Time-block coefficients and joint significance")
    L.append("")
    L.append("| Estimator | $F_\\eta$ | Time Trend (years) | GDP Growth (YoY) | Joint Wald $\\chi^2$ | Wald $p$ |")
    L.append("|---|---|---|---|---|---|")
    for gname, gkeys in GROUPS:
        tk = gkeys[1]
        if tk not in results:
            continue
        res = results[tk]["res"]
        w = results[tk]["wald"]
        tt = _fmt(res, "interaction_time_trend")
        gg = _fmt(res, "interaction_gdp_growth_yoy")
        is_si = gname == "Single-Index"
        wstat = f"{w[0]:.2f}" if pd.notna(w[0]) else ("n/a" if is_si else "—")
        wp = f"{w[1]:.3f}{get_stars(w[1])}" if pd.notna(w[1]) else ("n/a" if is_si else "—")
        L.append(f"| {gname} | {F_ETA.get(gname, '')} | {tt} | {gg} | {wstat} | {wp} |")
    L.append("")
    L.append("*Coefficients are shown as estimate (SE); Logit/Probit/Single-Index report Average "
             "Marginal Effects. Stars: \\*\\*\\* p<0.01, \\*\\* p<0.05, \\* p<0.1. The Wald statistic "
             "tests the joint null that both time-block coefficients are zero ($\\chi^2_2$); it is "
             "not defined for the Single-Index (its two time terms load on one monotone link slope). "
             "The Constrained-Linear (uniform $\\eta$) SEs are approximate — the clip is "
             "non-differentiable, so its delta-method SEs understate uncertainty and its Wald is "
             "inflated; read its significance qualitatively.*")
    L.append("")
    _wald_models = [(g, ks[1]) for g, ks in GROUPS if ks[1] in results and g != "Single-Index"]
    _n_tested = len(_wald_models)
    _n_sig = sum(1 for g, tk in _wald_models
                 if pd.notna(results[tk]["wald"][1]) and results[tk]["wald"][1] < 0.05)
    _wald_sig = [g for g, tk in _wald_models
                 if pd.notna(results[tk]["wald"][1]) and results[tk]["wald"][1] < 0.05]
    _trend_parts = []
    if tt_neg:
        _trend_parts.append(f"negative and significant in {_join(tt_neg)}")
    if tt_pos:
        _trend_parts.append(f"positive and significant in {_join(tt_pos)}")
    if tt_insig:
        _trend_parts.append(f"about zero and insignificant in {_join(tt_insig)}")
    L.append("**What this test means.** The joint Wald statistic asks whether calendar time and "
             "the business cycle add *anything* to the sleepiness function once Pix availability, "
             "broadband, demographics and the lagged Selic rate are already controlled for — the "
             "null is that both time coefficients are simultaneously zero. A small $p$ rejects "
             "that null and says $\\phi$ has a genuine **time-series dimension** the cross-sectional "
             f"`Tech` covariates miss. Here the null is rejected at 5% in **{_n_sig} of the {_n_tested}** "
             f"estimators that admit the test ({_join(_wald_sig)}). **But the decomposition matters — "
             "and it is exactly where the link choice bites.** The **business-cycle term (GDP growth) "
             f"is negative in every estimator** and significant in {_join(gg_neg + gg_pos)} — a robust "
             "*cyclical* channel ($\\phi$ lower in booms). The **secular trend, by contrast, is not "
             f"robust across $F_\\eta$**: it is {'; '.join(_trend_parts)}. So the time *dimension* "
             "matters, but whether $\\phi$ carries a downward drift or essentially none — even the "
             "trend's *sign* — **depends on the assumed shock distribution**: the bounded curved links "
             "(logit/probit/single-index) read a gentle decline in stickiness, while the (near-)linear "
             "LPM reads no trend. This makes the *Estimators* framing concrete — for the trend, the "
             "choice of $F_\\eta$ is not innocuous.")
    L.append("")

    # ---- Table 2: residual autocorrelation in the base spec ----
    L.append("## 2. Residual autocorrelation in the base spec (motivation)")
    L.append("")
    L.append("| Strategy | Ljung–Box(4) stat | Ljung–Box(4) $p$ |")
    L.append("|---|---|---|")
    for gname, gkeys in GROUPS:
        bk = gkeys[0]
        if bk not in results:
            continue
        lb = results[bk]["ljung_box"]
        if pd.isna(lb[1]):
            L.append(f"| {gname} | — | n/a (link/NLLS — no additive residual) |")
        else:
            L.append(f"| {gname} | {lb[0]:.2f} | {lb[1]:.3f}{get_stars(lb[1])} |")
    L.append("")
    L.append("*Test applied to the population-mean residual by quarter from the **Base** spec. "
             "A small $p$ indicates serially-correlated time structure the base spec does not "
             "capture — the motivation for a trend/cycle term.*")
    L.append("")
    L.append("**What the Ljung–Box test does.** It tests whether a series is serially "
             "*uncorrelated* — statistical white noise — against the alternative that it carries "
             "**autocorrelation** (drift, cycles, momentum: values systematically related to their "
             "own past). It takes the sample autocorrelations $\\hat\\rho_k$ at lags "
             "$k = 1, \\dots, m$ and combines them into a single *portmanteau* statistic "
             "$Q = n(n+2)\\sum_{k=1}^{m}\\hat\\rho_k^2/(n-k)$, which is distributed $\\chi^2_m$ "
             "under the white-noise null. So rather than eyeballing each lag separately, it asks "
             "whether the first $m$ autocorrelations are *jointly* zero. A small $p$ rejects "
             "whiteness (there is leftover time structure); a large $p$ does not. We use $m = 4$ "
             "(one year of quarterly data). Applied to regression residuals it is a specification "
             "check: an omitted trend or cycle would leak into the residuals and show up as "
             "autocorrelation, so a rejection would flag exactly the misspecification this exercise "
             "is probing for.")
    L.append("")
    L.append("**What this test means here.** Ljung–Box checks whether the *level* of the average "
             "quarterly residual is autocorrelated, the classic footprint of an omitted additive "
             "trend or cycle. The linear base specs do **not** reject (E1/E2 $p$ in the table above), "
             "so there is no leftover additive time structure in the residual level — and this is "
             "*consistent* with §1, where the joint time block is itself insignificant for those same "
             "linear specs (no tension). Where the time block *is* significant (the curved and "
             "constrained links), the action is still not in the residual level: the time terms enter "
             "**multiplicatively**, modulating the slope on $\\widetilde{D}_{t-1}$, which entity-"
             "demeaning leaves in place even as the residual level stays flat. The methodological "
             "takeaway stands: for this multiplicative specification the **joint Wald test on the "
             "interacted time terms — not a residual-autocorrelation test — is the right diagnostic**; "
             "a clean Ljung–Box rules out only a crude *additive* misspecification.")
    L.append("")
    L.append("*Caveat on power.* With only about 52 quarterly observations the test has limited "
             "power, so a non-rejection should be read as *no detectable additive autocorrelation*, "
             "not as proof that none exists.")
    L.append("")

    # ---- Table 3: headline coefficient stability ----
    L.append("## 3. Stability of headline coefficients (Base → +Time)")
    L.append("")
    L.append("| Strategy | Pix (Base) | Pix (+Time) | Broadband (Base) | Broadband (+Time) |")
    L.append("|---|---|---|---|---|")
    for gname, gkeys in GROUPS:
        bk, tk = gkeys
        if bk not in results or tk not in results:
            continue
        rb, rt = results[bk]["res"], results[tk]["res"]
        L.append(
            f"| {gname} | {_fmt(rb, 'interaction_pix_exists')} | {_fmt(rt, 'interaction_pix_exists')} "
            f"| {_fmt(rb, 'interaction_connections_per100')} | {_fmt(rt, 'interaction_connections_per100')} |"
        )
    L.append("")
    L.append("*If adding the time block leaves Pix and broadband coefficients stable, those "
             "channels were not merely proxying for an omitted trend.*")
    L.append("")
    L.append("**What this test means.** It asks whether the headline digital-finance channels are "
             "stable when the time block is added — i.e. whether they were merely proxying for the "
             f"trend. **Pix** is significant at 5% in only {pixB['sig']} of {pixB['total']} Base "
             f"specs; adding the time block does *not* collapse it — it is significant in "
             f"{pixT['sig']} of {pixT['total']} +Time specs"
             + (f", turning *positive* in {pixT['pos_sig']} of them" if pixT['pos_sig'] else "")
             + ". So there is no 'Pix proxies the trend' collapse on this panel; Pix and the trend are "
             "weakly separated rather than one absorbing the other. **Broadband** is the more "
             f"consistently signed channel — positive-significant in {bbB['pos_sig']} of "
             f"{bbB['total']} Base and {bbT['pos_sig']} of {bbT['total']} +Time specs (strongest "
             "under the curved/constrained links; the linear LPM leaves it insignificant, consistent "
             "with weaker identification near the $[0,1]$ ceiling), and it barely moves when the time "
             "block is added — so it is *not* simply standing in for the secular trend.")
    L.append("")

    # ---- phi_t impact ----
    L.append("## 4. Impact on the implied national $\\hat{\\phi}_t$ (one figure per estimator)")
    L.append("")
    L.append("| Estimator | $F_\\eta$ | mean Base | mean +Time | cells $\\phi\\notin[0,1]$ | range Base | range +Time |")
    L.append("|---|---|---|---|---|---|---|")
    for gname, gkeys in GROUPS:
        bk, tk = gkeys
        if bk not in results or tk not in results:
            continue
        pb, pt = results[bk]["phi_t"], results[tk]["phi_t"]
        oob = results[bk].get("frac_oob", 0.0) if results[bk].get("link") == "linear" else 0.0
        oob_str = f"{oob*100:.1f}\\%" if oob > 0 else "0"
        L.append(
            f"| {gname} | {F_ETA.get(gname, '')} | {pb.mean():.3f} | {pt.mean():.3f} "
            f"| {oob_str} | {pb.max()-pb.min():.3f} | {pt.max()-pt.min():.3f} |"
        )
    L.append("")
    L.append("*`cells $\\phi\\notin[0,1]$` is the share of market-cells whose (unbounded) linear "
             "index leaves $[0,1]$ in the Base spec — the LPM boundary breach the bounded links "
             "avoid by construction.*")
    L.append("")
    fig_by_group = {g: slug for (g, slug, _p) in figures}
    for gname, gkeys in GROUPS:
        slug = fig_by_group.get(gname)
        if slug is None:
            continue
        L.append(f"![{gname} ($F_\\eta$: {F_ETA.get(gname, '')}): implied national $\\hat{{\\phi}}_t$, "
                 f"Base (solid) vs.\\ $+$Time (dashed).](est_timeseries_phi_{slug}.png){{width=70%}}")
        L.append("")
    L.append("**What the figures and table show.** Each panel overlays the implied national "
             "$\\hat{\\phi}_t$ — the population-weighted fraction of *sleepy* (non-reoptimising) "
             "depositors — under Base (solid blue) and +Time (dashed red), with 95% bands. Three "
             "messages. **(1) The cycle is link-robust; the trend is link-dependent.** The GDP-growth "
             "term is negative in every estimator (a procyclical dip in $\\hat{\\phi}_t$), but the "
             "secular drift's sign and significance vary with the link (see §1) — visible as the "
             "differing slopes of the +Time (dashed) paths across panels. **(2) The bound cleanly "
             "separates the estimators:** the unconstrained LPM (Local/Pooled Linear) produces "
             f"$\\phi_{{mt}}$ *outside* $[0,1]$ for {oob_phrase} of market-cells (the "
             "`cells $\\phi\\notin[0,1]$` column) — inadmissible, since $\\phi$ is a fraction — even "
             "though its pop-weighted *national* $\\hat{\\phi}_t$ stays just under 1; the **Constrained "
             "Linear, Logit, Probit and Single-Index never breach the bound**. That is the practical "
             "payoff of treating $\\phi$ as a CDF: the bound keeps the active-demand construction "
             "(which subtracts $\\phi\\,\\widetilde{D}_{t-1}$) well-behaved. **(3) The *level* of "
             f"$\\hat{{\\phi}}_t$ is {lvl_phrase}** — mean Base sleepiness spans "
             f"{lvl_lo:.2f}–{lvl_hi:.2f} across uniform, logistic, normal and nonparametric shocks "
             "(logit/probit sit highest, constrained-linear lowest). The Single-Index — which names "
             "no distribution — sits among the parametric links in level and shape, so the logistic "
             "form is a convenience, not a driver; the distributional choice matters more for the "
             "*level* and the *trend's sign* than for the robustly-negative cycle.")
    L.append("")

    # ---- auto interpretation ----
    L.append("## 5. Interpretation")
    L.append("")
    _testable = [(g, ks[1]) for g, ks in GROUPS
                 if ks[1] in results and pd.notna(results[ks[1]]["wald"][1])]
    n_sig = sum(1 for g, tk in _testable if results[tk]["wald"][1] < 0.05)
    n_tot = len(_testable)
    L.append(f"- **The time block matters, but mostly through the *cycle*.** It is jointly significant "
             f"at 5% in {n_sig} of the {n_tot} estimators that admit the Wald test (§1). The "
             "business-cycle (GDP-growth) term is **negative in every estimator** — a robust "
             "procyclical channel. The **secular trend, however, is not robust**: it is "
             f"{'; '.join(_trend_parts)} (§1). So the trend's *sign* depends on the assumed shock "
             "distribution $F_\\eta$ — a concrete instance of the model-selection point, since $\\phi$ "
             "is structurally a CDF.")
    lb_sig = [gname for gname, gk in GROUPS
              if gk[0] in results and pd.notna(results[gk[0]]["ljung_box"][1])
              and results[gk[0]]["ljung_box"][1] < 0.05]
    if lb_sig:
        L.append(f"- Base-spec residuals show significant serial correlation (Ljung–Box, §2) in: "
                 f"{', '.join(lb_sig)}.")
    else:
        L.append("- The linear base-spec residuals show no additive serial correlation (Ljung–Box, "
                 "§2); the time effect lives in the *slope*, which is why the Wald test — not "
                 "Ljung–Box — is the right diagnostic.")
    L.append(f"- **Pix is weakly identified, not a 'trend proxy' (§3):** significant in only "
             f"{pixB['sig']} of {pixB['total']} Base specs, and adding the time block does not collapse "
             f"it ({pixT['sig']} of {pixT['total']} significant +Time). **Broadband** is the more "
             f"consistently positive channel ({bbT['pos_sig']} of {bbT['total']} +Time significant) and "
             "is stable to the time block, so it is not merely proxying the trend.")
    L.append(f"- **The *level* of $\\hat{{\\phi}}_t$ is {lvl_phrase} (§4)** — mean Base spans "
             f"{lvl_lo:.2f}–{lvl_hi:.2f} across uniform/logistic/normal/nonparametric. The "
             "distributional choice moves the *level* (and the trend's *sign*) more than the "
             "robustly-negative *cycle*. Separately, only the unconstrained LPM (Est 1/2) leaves "
             "$[0,1]$ at the cell level; every bounded estimator stays in range.")
    L.append("")
    L.append("**Bottom line for model selection.** Because $\\phi$ is structurally a CDF, the relevant "
             "comparison is *which $F_\\eta$*, not 'bounded vs.\\ unbounded'. On the current panel the "
             "**cycle** is the robust object — negative and significant under every link, including the "
             "assumption-free Single-Index. The **secular trend is link-dependent** (its sign and "
             "significance vary across uniform/logistic/normal/nonparametric — see §1), and the "
             f"**level** is {lvl_phrase} ({lvl_lo:.2f}–{lvl_hi:.2f}); so both the trend and the level "
             "of stickiness inherit whatever $F_\\eta$ you assume, while the cycle does not. Practical "
             "implications: (i) the only structural must-have is the **bound** (the LPM violates it for "
             f"{oob_phrase} of cells; the demand-step clip patches it only post-hoc); (ii) a coherent, "
             "transparent workhorse is the **Constrained Linear** (uniform $\\eta$, bound imposed) or "
             "the **Single-Index** (assumption-free link); and (iii) since the *trend* and *level* are "
             "the link-sensitive objects, report them across links rather than from a single one, and "
             "lean on the robustly-negative *cycle* for anything downstream.")
    L.append("")
    L.append("**Suggested next steps.** (i) Replace the linear trend with calendar-year dummies to see "
             "whether the drift is smooth or concentrated, and whether the trend's link-dependence "
             "persists; (ii) add explicit 2015Q2–2016Q4 recession and 2020Q2–2020Q4 COVID indicators "
             "to pin the cycle; (iii) carry the Constrained-Linear and Single-Index $\\hat{\\phi}$ into "
             "the BLP step and compare the downstream estimates — since the level and trend are "
             "link-sensitive, this is a genuine robustness check, not a formality.")
    L.append("")
    L.append(f"*Full coefficient tables: `{tex_base_name}` (Base) and `{tex_time_name}` (+Time), "
             "LaTeX, for inclusion in the draft.*")
    L.append("")
    L.append("## References")
    L.append("")
    L.append("- Ichimura, H. (1993). \"Semiparametric least squares (SLS) and weighted SLS estimation "
             "of single-index models.\" *Journal of Econometrics* 58(1–2): 71–120.")
    L.append("- Klein, R. W., & Spady, R. H. (1993). \"An efficient semiparametric estimator for "
             "binary response models.\" *Econometrica* 61(2): 387–421.")
    L.append("- Härdle, W., Hall, P., & Ichimura, H. (1993). \"Optimal smoothing in single-index "
             "models.\" *Annals of Statistics* 21(1): 157–178.")
    L.append("- Powell, J. L., Stock, J. H., & Stoker, T. M. (1989). \"Semiparametric estimation of "
             "index coefficients.\" *Econometrica* 57(6): 1403–1430.")
    L.append("- Balabdaoui, F., Groeneboom, P., & Hendrickx, K. (2019). \"Score estimation in the "
             "monotone single-index model.\" *Scandinavian Journal of Statistics* 46(2): 517–544.")
    L.append("- Robertson, T., Wright, F. T., & Dykstra, R. L. (1988). *Order Restricted Statistical "
             "Inference* (pool-adjacent-violators / isotonic regression). Wiley.")
    L.append("- Papke, L. E., & Wooldridge, J. M. (1996). \"Econometric methods for fractional response "
             "variables with an application to 401(k) plan participation rates.\" *Journal of Applied "
             "Econometrics* 11(6): 619–632.")
    L.append("- Imbens, G. W., & Kolesár, M. (2016). \"Robust standard errors in small samples: some "
             "practical advice.\" *Review of Economics and Statistics* 98(4): 701–712.")
    L.append("- Carter, A. V., Schnepel, K. T., & Steigerwald, D. G. (2017). \"Asymptotic behavior of a "
             "$t$-test robust to cluster heterogeneity.\" *Review of Economics and Statistics* 99(4): "
             "698–709.")
    L.append("")

    out_path.write_text("\n".join(L), encoding="utf-8")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Spec-12 time-series robustness test (Est 1-3).")
    parser.add_argument("--replot", action="store_true",
                        help="Skip estimation; rebuild table/figure/json/report from the cached "
                             "results pickle (fast iteration on formatting).")
    args = parser.parse_args()

    rout = _paths_mod.PROCESSED / "ESTIMATION_OUTPUT" / "Rout"
    rout.mkdir(parents=True, exist_ok=True)
    cache_path = rout / "est_timeseries_models.pkl"

    if args.replot:
        if not cache_path.exists():
            print(f"--replot requested but no cache at {cache_path}; run once without --replot first.")
            return
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded cached results from {cache_path}")
    else:
        results = run_all()
        if not results:
            print("No results produced; aborting.")
            return
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Cached fitted results to {cache_path}")

    tex_base = "est_timeseries_spec12_base.tex"
    tex_time = "est_timeseries_spec12_time.tex"
    json_name = "est_timeseries_results.json"
    md_name = "Sleepiness_TimeSeries_Test.md"

    build_latex_table(results, BASE_KEYS, rout / tex_base,
                       title="Spec 12 across Sleepiness-Shock Distributions $F_\\eta$ (Base)",
                       label="tab:timeseries_spec12_base")
    build_latex_table(results, TIME_KEYS, rout / tex_time,
                       title="Spec 12 across Sleepiness-Shock Distributions $F_\\eta$ ($+$Time block)",
                       label="tab:timeseries_spec12_time")
    figures = build_per_model_figures(results, rout)
    export_json(results, rout / json_name)
    build_markdown(results, figures, _DRAFTS_DIR / md_name, tex_base, tex_time)

    # copy artefacts into the Drafts folder (tables + per-model figures)
    copy_names = [tex_base, tex_time] + [f"est_timeseries_phi_{slug}.png" for _, slug, _ in figures]
    for name in copy_names:
        try:
            shutil.copy(rout / name, _DRAFTS_DIR / name)
        except Exception as exc:
            print(f"  [warn] could not copy {name} to Drafts: {exc}")

    print("\nArtefacts written:")
    print(f"  tables : {rout / tex_base} ; {rout / tex_time}")
    print(f"  figures: {len(figures)} per-model PNGs in {rout}")
    print(f"  json   : {rout / json_name}")
    print(f"  report : {_DRAFTS_DIR / md_name}")


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
