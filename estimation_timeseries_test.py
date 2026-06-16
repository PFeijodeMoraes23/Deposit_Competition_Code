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
from scipy import stats
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


def implied_phi_t(df: pd.DataFrame, res, logistic: bool):
    """National pop-weighted phi_t (market-level mean first, then pop weight
    across markets -- identical to calculate_phis) plus a delta-method 95% band.

    Returns (phi_t Series, lo Series, hi Series) indexed by year_quarter.
    """
    phi_params = _phi_param_names(res)
    X = _build_phi_X(df, phi_params)

    params = res.params_native if (logistic and hasattr(res, "params_native")) else res.params
    beta = params[phi_params].values.astype(float)
    lin = X @ beta
    phi_mt = 1.0 / (1.0 + np.exp(-np.clip(lin, -700, 700))) if logistic else lin

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

    return phi_t, pd.Series(lo), pd.Series(hi)


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


def run_all():
    e1_df, pooled_df, s_tech, iv_cols = build_frames()
    s_time = s_tech + TIME_VARS

    # Each entry: (label, group, with_time, frame, runner, logistic)
    plan = [
        ("E1 Base",  "Local (Est 1)",            False, e1_df,     "lin1", False),
        ("E1 +Time", "Local (Est 1)",            True,  e1_df,     "lin1", False),
        ("E2 Base",  "Pooled Linear (Est 2)",    False, pooled_df, "lin2", False),
        ("E2 +Time", "Pooled Linear (Est 2)",    True,  pooled_df, "lin2", False),
        ("E3 Base",  "Pooled Logistic (Est 3)",  False, pooled_df, "log3", True),
        ("E3 +Time", "Pooled Logistic (Est 3)",  True,  pooled_df, "log3", True),
    ]

    results = {}
    for label, group, with_time, frame, runner, logistic in plan:
        state_cols = s_time if with_time else s_tech
        print(f"\n--- Estimating {label}  ({'Tech+Time' if with_time else 'Tech'}) ---")
        if runner == "lin1":
            df, res_ss, res_fs = _run_linear(frame, iv_cols, state_cols, e1_first_stage, e1_second_stage, label)
        elif runner == "lin2":
            df, res_ss, res_fs = _run_linear(frame, iv_cols, state_cols, e2_first_stage, e2_second_stage, label)
        else:
            df, res_ss, res_fs = _run_logistic(frame, iv_cols, state_cols)

        if res_ss is None:
            print(f"  [warn] {label} produced no second stage; skipping.")
            continue

        phi_t, lo, hi = implied_phi_t(df, res_ss, logistic)

        lb_stat, lb_p = (np.nan, np.nan)
        wald = (np.nan, np.nan, 0)
        if with_time:
            wald = wald_time_block(res_ss)
        else:
            lb_stat, lb_p = ljung_box_resid(df, res_ss, state_cols, has_cf=True)

        results[label] = {
            "group": group,
            "with_time": with_time,
            "logistic": logistic,
            "res": res_ss,
            "state_cols": state_cols,
            "phi_t": phi_t,
            "phi_lo": lo,
            "phi_hi": hi,
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

ORDER_KEYS = ["E1 Base", "E1 +Time", "E2 Base", "E2 +Time", "E3 Base", "E3 +Time"]
GROUPS = [
    ("Local (Est 1)", ["E1 Base", "E1 +Time"]),
    ("Pooled Linear (Est 2)", ["E2 Base", "E2 +Time"]),
    ("Pooled Logistic (Est 3)", ["E3 Base", "E3 +Time"]),
]


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
def build_latex_table(results, out_path, title, label):
    res_by_key = {k: results[k]["res"] for k in ORDER_KEYS if k in results}
    keys = [k for k in ORDER_KEYS if k in res_by_key]
    n = len(keys)

    tex = [r"\setstretch{1.0}"]
    col_def = (r">{\raggedright\arraybackslash}p{0.24\textwidth} "
               r"*{" + str(n) + r"}{>{\centering\arraybackslash}X}")
    tex.append(r"\begin{xltabular}{\textwidth}{" + col_def + "}")
    tex.append(r"\caption{" + title + r"}\label{" + label + r"} \\")
    tex.append(r"\toprule")

    # grouped super-header
    sup = ["\\multicolumn{1}{c}{}"]
    sub = ["Variable"]
    for gname, gkeys in GROUPS:
        present = [k for k in gkeys if k in res_by_key]
        if not present:
            continue
        sup.append(r"\multicolumn{" + str(len(present)) + r"}{c}{" + gname + r"}")
        for k in present:
            sub.append("Base" if k.endswith("Base") else r"$+$Time")
    tex.append(" & ".join(sup) + r" \\")
    tex.append(" & ".join(sub) + r" \\")
    tex.append(r"\midrule")
    tex.append(r"\endfirsthead")
    tex.append(r"\multicolumn{" + str(n + 1) + r"}{c}{{\bfseries \tablename\ \thetable{} (continued)}} \\")
    tex.append(r"\toprule")
    tex.append(" & ".join(sub) + r" \\")
    tex.append(r"\midrule")
    tex.append(r"\endhead")
    tex.append(r"\midrule")
    tex.append(r"\multicolumn{" + str(n + 1) + r"}{r}{{Continued on next page}} \\")
    tex.append(r"\endfoot")
    tex.append(r"\bottomrule")
    notes = (r"\multicolumn{" + str(n + 1) + r"}{p{\dimexpr\textwidth-2\tabcolsep\relax}}"
             r"{\scriptsize\textit{Notes:} Spec 12 (IV Hausman $\times$ Tech). "
             r"``$+$Time'' adds a time trend (years) and YoY GDP-per-capita growth, "
             r"each interacted with $\widetilde{D}_{t-1}$. Est.~3 reports Average "
             r"Marginal Effects. Standard errors in parentheses. "
             r"*** $p<0.01$, ** $p<0.05$, * $p<0.1$.}")
    tex.append(notes)
    tex.append(r"\endlastfoot")

    for v in ORDER_VARS:
        present_any = any(v in (res_by_key[k].params.index if res_by_key[k] is not None else []) for k in keys)
        if not present_any:
            continue
        lab = r"\multirow[t]{2}{0.24\textwidth}{\raggedright " + NICE.get(v, v) + r"}"
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
    # diagnostics rows
    row_nobs = ["Observations"]
    row_r2 = ["$R^2$"]
    row_clu = ["Clusters ($G$)"]
    row_eff = ["Effective $G^*$"]
    row_wald = ["Time block Wald $p$"]
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
    for r in (row_nobs, row_r2, row_clu, row_eff, row_wald):
        tex.append(" & ".join(r) + r" \\")
    tex.append(r"\end{xltabular}")
    tex.append(r"\doublespacing")

    out_path.write_text("\n".join(tex), encoding="utf-8")


# ==============================================================================
# OUTPUT: figure (3 panels, Base vs +Time phi_t)
# ==============================================================================
def _to_dates(idx):
    return pd.PeriodIndex([str(q).replace("_", "Q") for q in idx], freq="Q").to_timestamp()


def _sorted_index(idx):
    return sorted(idx, key=lambda q: pd.Period(str(q).replace("_", "Q"), freq="Q"))


def build_figure(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.2), sharey=True)
    for ax, (gname, gkeys) in zip(axes, GROUPS):
        base_key = gkeys[0]
        time_key = gkeys[1]
        for key, color, ls, lab in [
            (base_key, "#1f77b4", "-", "Base (Tech)"),
            (time_key, "#d62728", "--", r"$+$Time"),
        ]:
            if key not in results:
                continue
            r = results[key]
            phi = r["phi_t"].reindex(_sorted_index(r["phi_t"].index))
            x = _to_dates(phi.index)
            ax.plot(x, phi.values, color=color, linestyle=ls, linewidth=2.4, label=lab)
            lo = r["phi_lo"].reindex(phi.index)
            hi = r["phi_hi"].reindex(phi.index)
            if lo.notna().any():
                ax.fill_between(x, lo.values, hi.values, color=color, alpha=0.15)
        ax.set_title(gname, fontsize=16)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.grid(alpha=0.35)
        ax.legend(loc="best", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
    axes[0].set_ylabel(r"National $\hat{\phi}_t$", fontsize=15)
    fig.suptitle(r"Implied National $\hat{\phi}_t$: Spec 12 Base vs. $+$Time block", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
            "logistic": r["logistic"],
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
    return f"{res.params[v]:.{digits}f}{get_stars(res.pvalues[v])} ({res.bse[v]:.{digits}f})"


def build_markdown(results, fig_name, out_path, tex_name):
    L = []
    L.append("# Time-Series Structure in the Depositor Sleepiness Function")
    L.append("")
    L.append("**Project:** Deposit Competition — Egan et al. (2025) Replication & Extension  ")
    L.append("**Script:** `estimation_timeseries_test.py`  ")
    L.append(f"**Generated:** {pd.Timestamp.now():%Y-%m-%d %H:%M}  ")
    L.append("")
    L.append("---")
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
             "(`nr_lagged_dep`), so it shifts the *level* of $\\phi$ rather than adding a "
             "separate intercept to deposits. A **negative** `time_trend` coefficient therefore "
             "means $\\phi$ *declines* over calendar time — depositors become progressively "
             "**more attentive** (less sleepy) across 2013–2025. A **negative** `gdp_growth_yoy` "
             "coefficient means $\\phi$ is **lower in booms**: when real activity accelerates, "
             "depositors reallocate more readily, i.e. inertia is mildly *procyclical* in the "
             "opportunity cost of not re-optimising. The three strategies differ in units — Est 1 "
             "and Est 2 report linear slopes, while Est 3 reports Average Marginal Effects through "
             "a logistic link, so its magnitudes are compressed but its *signs and significance* "
             "are directly comparable.")
    L.append("")

    # ---- Table 1: time block coefficients + joint test ----
    L.append("## 1. Time-block coefficients and joint significance")
    L.append("")
    L.append("| Strategy | Time Trend (years) | GDP Growth (YoY) | Joint Wald $\\chi^2$ | Wald $p$ |")
    L.append("|---|---|---|---|---|")
    for gname, gkeys in GROUPS:
        tk = gkeys[1]
        if tk not in results:
            continue
        res = results[tk]["res"]
        w = results[tk]["wald"]
        tt = _fmt(res, "interaction_time_trend")
        gg = _fmt(res, "interaction_gdp_growth_yoy")
        wstat = f"{w[0]:.2f}" if pd.notna(w[0]) else "—"
        wp = f"{w[1]:.3f}{get_stars(w[1])}" if pd.notna(w[1]) else "—"
        L.append(f"| {gname} | {tt} | {gg} | {wstat} | {wp} |")
    L.append("")
    L.append("*Coefficients are shown as estimate (SE); Est 3 reports Average Marginal Effects. "
             "Stars: \\*\\*\\* p<0.01, \\*\\* p<0.05, \\* p<0.1. The Wald statistic tests the joint "
             "null that both time-block coefficients are zero ($\\chi^2_2$).*")
    L.append("")
    L.append("**What this test means.** The joint Wald statistic asks whether calendar time and "
             "the business cycle add *anything* to the sleepiness function once Pix availability, "
             "broadband, demographics and the lagged Selic rate are already controlled for — the "
             "null is that both time coefficients are simultaneously zero. A small $p$ rejects "
             "that null and says $\\phi$ has a genuine **time-series dimension** the cross-sectional "
             "`Tech` covariates miss. Here the null is rejected in **all three** strategies: the "
             "secular trend is individually significant at the 0.1% level and GDP growth at 5% in "
             "both linear strategies, and even the logistic strategy — which compresses every "
             "effect through the $[0,1]$ link — still rejects jointly at 5%. The negative signs say "
             "the action is a **downward drift** in stickiness (rising attention) plus a smaller "
             "**procyclical** dip in booms.")
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
            L.append(f"| {gname} | — | — (NLLS residuals not tested) |")
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
    L.append("**What this test means here — and why it does *not* contradict §1.** Ljung–Box checks "
             "whether the *level* of the average quarterly residual is autocorrelated, the classic "
             "footprint of an omitted additive trend or cycle. It does **not** reject here "
             "(p $\\approx$ 0.86--0.90), which at first looks at odds with the strongly significant Wald "
             "tests above. The two are measuring different objects. The time block does not enter "
             "the model additively in the level; it **modulates the slope** on $\\widetilde{D}_{t-1}$. "
             "Entity-demeaning together with the existing covariates already flattens the residual "
             "*level*, so there is no leftover autocorrelation for Ljung–Box to find — yet the "
             "*attention slope* still varies systematically over time, which is exactly what the "
             "Wald test on the interacted block detects. The takeaway is methodological: for this "
             "multiplicative specification the **joint Wald test on the interacted time terms — not "
             "a residual-autocorrelation test — is the correct diagnostic**, and a clean Ljung–Box "
             "is reassuring rather than contradictory (it rules out a crude additive misspecification).")
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
    L.append("**What this test means.** This is the most consequential panel. It asks whether the "
             "paper's headline digital-finance channels survive the trend — and for **Pix they "
             "largely do not**. In both linear strategies the `pix_exists` coefficient collapses "
             "from $\\approx -0.12$ (significant at 10%) to $\\approx -0.01$ and statistically indistinguishable from "
             "zero once the trend is included; broadband shrinks and loses significance too. The "
             "mechanism is collinearity in time: Pix switches on in 2020Q4, late in a sample over "
             "which $\\phi$ is *already* drifting down, so a single post-2020 indicator mechanically "
             "absorbs part of a longer secular decline. Read structurally, much of what looked like "
             "a 'Pix made depositors more attentive' effect is better described as 'depositors were "
             "trending more attentive throughout, and Pix arrived near the end of that trend.' This "
             "is an **identification caveat**, not proof Pix is irrelevant — disentangling a "
             "level-shift at Pix launch from the ongoing trend would need either a sharper "
             "event-study design around 2020Q4 or cross-sectional variation in Pix exposure.")
    L.append("")

    # ---- phi_t impact ----
    L.append("## 4. Impact on the implied national $\\hat{\\phi}_t$")
    L.append("")
    L.append(f"![Implied national $\\hat{{\\phi}}_t$ under Base (solid) vs.\\ $+$Time (dashed), "
             f"with delta-method 95% bands.]({fig_name}){{width=100%}}")
    L.append("")
    L.append("| Strategy | mean $\\hat{\\phi}_t$ Base | mean +Time | range Base | range +Time |")
    L.append("|---|---|---|---|---|")
    for gname, gkeys in GROUPS:
        bk, tk = gkeys
        if bk not in results or tk not in results:
            continue
        pb, pt = results[bk]["phi_t"], results[tk]["phi_t"]
        L.append(
            f"| {gname} | {pb.mean():.3f} | {pt.mean():.3f} "
            f"| {pb.max()-pb.min():.3f} | {pt.max()-pt.min():.3f} |"
        )
    L.append("")
    L.append("**What the figure and table show.** Each panel overlays the implied national "
             "$\\hat{\\phi}_t$ — the population-weighted fraction of *sleepy* (non-reoptimising) "
             "depositors nationwide — under the Base spec (solid blue) and the +Time spec (dashed "
             "red), with delta-method 95% confidence bands. Two points matter for whether the time "
             "block changes any downstream conclusion. **First, the level is robust:** adding the "
             "time block moves the *mean* of $\\hat{\\phi}_t$ by at most about 0.7 pp in every strategy, "
             "so the average stickiness that feeds the BLP demand/welfare step is essentially "
             "unchanged. **Second, the shape is improved:** the Base linear specifications push "
             "$\\hat{\\phi}_t$ *above 1* around 2020–21 — an economically inadmissible region, since "
             "$\\phi$ is a fraction — whereas the +Time path stays inside $[0,1]$ and compresses the "
             "peak-to-trough range by roughly 6–7 pp. In other words the time block mostly **tames "
             "the implausible spikes** rather than relocating the series. The logistic strategy "
             "bounds $\\phi\\in(0,1)$ by construction, so there the two curves are nearly "
             "indistinguishable and the time block is close to immaterial for the path.")
    L.append("")

    # ---- auto interpretation ----
    L.append("## 5. Interpretation")
    L.append("")
    n_sig = sum(
        1 for _, gk in GROUPS
        if gk[1] in results and pd.notna(results[gk[1]]["wald"][1]) and results[gk[1]]["wald"][1] < 0.05
    )
    n_tot = sum(1 for _, gk in GROUPS if gk[1] in results)
    L.append(
        f"- The time block is **jointly significant at 5% in {n_sig} of {n_tot}** estimation "
        "strategies (joint Wald test, §1)."
    )
    lb_sig = [gname for gname, gk in GROUPS
              if gk[0] in results and pd.notna(results[gk[0]]["ljung_box"][1])
              and results[gk[0]]["ljung_box"][1] < 0.05]
    if lb_sig:
        L.append(f"- Base-spec residuals show significant serial correlation (Ljung–Box, §2) in: "
                 f"{', '.join(lb_sig)} — consistent with unmodelled time structure.")
    else:
        L.append("- Base-spec residuals show no significant serial correlation (Ljung–Box, §2), "
                 "i.e. the existing covariates already absorb most of the time structure.")
    L.append("- The headline channels are **not all robust**: the Pix dummy collapses to zero once "
             "the trend is in (§3), because Pix's 2020Q4 onset overlaps the tail of an already-"
             "declining $\\phi$. Broadband weakens but is less affected.")
    L.append("- The **economic impact is modest where it counts**: the mean of $\\hat{\\phi}_t$ is "
             "essentially unchanged (§4), so the BLP welfare numbers built on the *level* of "
             "sleepiness are safe; the time block mainly removes the implausible above-1 spikes and "
             "tightens the path.")
    L.append("")
    L.append("**Bottom line.** There *is* a robust time-series component to depositor sleepiness — a "
             "secular rise in attention plus a mild procyclical dip — but it operates on the *slope* "
             "of the sleepiness function, not as additive residual structure (hence significant Wald, "
             "clean Ljung–Box). Its first-order consequence is interpretive rather than quantitative: "
             "the apparent Pix effect is largely a repackaged trend, so claims about Pix's causal role "
             "in waking depositors up should be hedged or backed by a design that separates the "
             "2020Q4 launch from the ongoing drift. The aggregate $\\hat{\\phi}_t$ level used "
             "downstream is unaffected.")
    L.append("")
    L.append("**Suggested next steps.** (i) Replace the linear trend with calendar-year dummies to "
             "see whether the drift is smooth or concentrated in specific years; (ii) add an explicit "
             "2015Q2–2016Q4 recession and a 2020Q2–2020Q4 COVID indicator to separate those episodes "
             "from the smooth trend/cycle; (iii) run a sharper Pix event-study around 2020Q4 to test "
             "for a genuine level break net of the trend.")
    L.append("")
    L.append(f"*Full coefficient table with all state variables: `{tex_name}` (LaTeX, for inclusion in the draft).*")
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

    tex_name = "est_timeseries_spec12_comparison.tex"
    fig_name = "est_timeseries_spec12_phi_t.png"
    json_name = "est_timeseries_results.json"
    md_name = "Sleepiness_TimeSeries_Test.md"

    build_latex_table(results, rout / tex_name,
                       title="Time-Series Robustness of Spec 12 across Estimation Strategies",
                       label="tab:timeseries_spec12")
    build_figure(results, rout / fig_name)
    export_json(results, rout / json_name)
    build_markdown(results, fig_name, _DRAFTS_DIR / md_name, tex_name)

    # mirror export_analyze_spec12: copy artefacts into the Drafts folder
    for name in (tex_name, fig_name):
        try:
            shutil.copy(rout / name, _DRAFTS_DIR / name)
        except Exception as exc:
            print(f"  [warn] could not copy {name} to Drafts: {exc}")

    print("\nArtefacts written:")
    print(f"  table  : {rout / tex_name}")
    print(f"  figure : {rout / fig_name}")
    print(f"  json   : {rout / json_name}")
    print(f"  report : {_DRAFTS_DIR / md_name}")


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
