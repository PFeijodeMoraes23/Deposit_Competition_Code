"""
estimation_bbl_1_polfunc.py
================================
Estimates parametric policy functions for endogenous deposit types k=4,5
(BBL Step 1, sec:empirical:cost in V_Main.tex).

Following Egan, Hortacsu & Matvos (2025, NBER), Ryan (2012, Econometrica),
and Matvos & Seru (2014, AER), we estimate flexible parametric mappings from
state variables to equilibrium deposit spreads for deposit types whose rates
are set endogenously by banking institutions (k=4: time/CDB deposits;
k=5: prepaid accounts).

Coefficients are allowed to differ by firm type (B = brick-and-mortar,
D = digital/national) and by deposit type. For D-type firms, which operate
nationally rather than at the MCA level, we test three demographic
specifications:
    Option A — Exclude MCA-level demographics entirely.
    Option B — Use population-weighted national averages of MCA demographics.
    Option C — Pool B and D firms under a single coefficient vector
               (D-firms receive national demographics as their regressors).
Option C is still estimated and kept in the summary/pickle as a robustness record, but it is NOT
shown in the tables: Step 2 consumes only B and D-Option-B, and pooling the few-but-huge B clusters
with the many-small D ones collapses the effective cluster count (G* ~ 9) that the WCB relies on.

Pipeline position
-----------------
  estimation_1_sleep  →  estimation_1_demand  →  **estimation_bbl_1_polfunc**
                          →  estimation_bbl_2_fwd_sim  →  estimation_bbl_3_solve

NO SPEC LABEL.  The policy function never touches a sleepiness specification — it regresses the
observed spread (or deposit rate) on the pricing state — so it is SPEC-INVARIANT and its outputs
carry no `_spec_{id}` suffix. An earlier version replicated identical files under every spec label
for pipeline cosmetics; that was removed (2026-07-20) as misleading.

INFERENCE.  Standard errors are a score/multiplier wild cluster bootstrap at the CONGLOMERATE
level (utils.sleep_links.linear_wild_cluster_bootstrap) — the same scheme and clustering unit as
the sleepiness and BLP stages, so every SE in the paper is produced one way. G and G* are reported
as descriptive cluster-paucity statistics, not as the inference. (Previously: CRVE + t(G*).)

Usage
-----
  python estimation_bbl_1_polfunc.py                  # spread (feeds BBL Step 2)
  python estimation_bbl_1_polfunc.py --depvar rate    # annualized deposit rate (robustness)

References
----------
  Bajari, Benkard & Levin (2007, Econometrica)
  Egan, Hortacsu & Matvos (2025, NBER WP)
  Matvos & Seru (2014, AER)
  Ryan (2012, Econometrica)
  Cameron, Gelbach & Miller (2008); MacKinnon & Webb (2017)  [wild cluster bootstrap]
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import sys
import os
import json
import pickle
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import warnings

# Suppress statsmodels rank-deficiency warnings due to cluster corrections
warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")


# ==============================================================================
# 0. Paths & Constants
# ==============================================================================
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
PANEL_CSV = DATA_DIR / "market_panel.csv"

# Estimation window [2016, 2024] — defined once in utils/window.py (full rationale + the
# DEMAND_MIN_YEAR / DEMAND_MAX_YEAR env overrides, which it reads).
from utils.window import MIN_YEAR as POLFUNC_MIN_YEAR, MAX_YEAR as POLFUNC_MAX_YEAR  # noqa: E402
OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "COST_POLFUNC"

# Endogenous deposit types (spreads set by institutions)
K_ENDOG = [4, 5]
K_LABELS = {4: "Time_CDB", 5: "Prepaid"}

# --------------------------------------------------------------------------
# State variable groups
# --------------------------------------------------------------------------
# Bank-level characteristics (available for all firms)
BANK_CHARS = [
    'log_total_assets_lag', 'equity_ratio_lag',
    'has_ip',
    'seg_S2', 'seg_S3', 'seg_S4', 'seg_S5',
]

# Balance-sheet / risk characteristics (added 2026-07-17; each ~96.5% populated in 2016-2024).
# `age_interpolated` was requested too but is 0% populated in-window, so it is intentionally omitted
# (including an all-NaN regressor would drop the entire complete-case sample).
BALANCE_SHEET = [
    'asset_return_qoq_lag',      # realized portfolio yield (profitability / asset-side pricing)
    'npl_provision_ratio_lag',   # credit-risk / provisioning intensity
    'credit_assets_lag',         # loan intensity (credit / assets)
]

# Operating cost shifters (from COSIF DRE report)
COST_SHIFTERS = [
    'personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag',
]

# Capital adequacy & wholesale funding
CAPITAL_WHOLESALE = [
    'indice_basileia_lag', 'wholesale_ratio_lag', 'lci_lca_ratio_lag',
]

# Market-level demographics (MCA-level for B-firms, aggregated for D-firms)
DEMOGRAPHICS = [
    'gdp_per_capita', 'fraction_65plus', 'fraction_young',
    'pix_users_pf_per1000', 'connections_per100',
    'branches_per1000', 'cadunico_families_per1000',
]

# Macro variable
MACRO = ['risk_free_qoq']

# ---- Outlier control ------------------------------------------------------------------------
# The accounting ratios carry a small number of corrupt observations, concentrated among D firms:
# `indice_basileia_lag` reaches 16,069.7 for D (sd 205.6) against a B-firm max of 7.10 (sd 0.05),
# and `personnel_cost_ratio_lag` reaches 2.32 (232% of assets). These are near-zero-denominator
# artifacts, and because leverage concentrates in a handful of rows they both drive the point
# estimates and collapse the wild-cluster-bootstrap SEs (the Basel row printed 0.0001 with a
# 0.0000 SE at *** before this was applied).
#
# They are winsorized at the 1st/99th percentile SEPARATELY WITHIN B and D. Within-type is the
# point: B and D have genuinely different balance sheets (D-firm equity ratios are ~4x B-firm
# ones), so pooled percentiles would clip real cross-type variation instead of the corrupt tail.
# Demographics are NOT winsorized (market-level, clean), nor is risk_free_qoq (macro), nor
# log_total_assets_lag (already a log). Quadratic terms are rebuilt FROM the winsorized bases.
WINSOR_PCT = 0.01
WINSOR_VARS = BALANCE_SHEET + COST_SHIFTERS + CAPITAL_WHOLESALE + ['equity_ratio_lag']

# ---- Centering -------------------------------------------------------------------------------
# Continuous regressors are demeaned WITHIN each estimation sample, so the intercept is the
# prediction at the AVERAGE state instead of at x=0. Uncentered, x=0 means a bank with R$1 of
# assets (log assets = 0), far outside the support: the k=4 B-type constant was +170pp purely to
# offset the log-asset terms (-375pp linear, +209pp quadratic, netting ~+3pp at the median).
#
# This is a REPARAMETRIZATION, not a different model. span{1, x, x^2} == span{1, x-xbar,
# (x-xbar)^2}, so R^2, residuals and every fitted value are unchanged -- which is exactly what
# the fitted-values identity check in main() verifies against the pre-centering policy CSV.
# Slopes are unchanged too, EXCEPT the linear terms of the three variables that also enter
# quadratically (log assets, equity ratio, risk-free): those become the marginal effect AT THE
# MEAN rather than at zero, which is the interpretable margin anyway.
#
# Dummies are NOT centered, so the intercept keeps a reference-category reading: segment S1,
# no IP subsidiary, every continuous regressor at its mean. Centering them would turn the
# intercept into a grand mean and throw that reading away.
NO_CENTER_VARS = {'has_ip', 'seg_S2', 'seg_S3', 'seg_S4', 'seg_S5'}

# Quadratic terms (appended as _sq)
QUADRATIC_BASE = ['log_total_assets_lag', 'equity_ratio_lag', 'risk_free_qoq']

# Full non-demographic regressors (common to all firm types)
COMMON_REGRESSORS = BANK_CHARS + BALANCE_SHEET + COST_SHIFTERS + CAPITAL_WHOLESALE + MACRO

# D-firm demographic options
D_DEMO_OPTIONS = ['A', 'B', 'C']
D_DEMO_LABELS = {
    'A': 'No demographics',
    'B': 'National (pop-weighted) demographics',
    'C': 'Pooled with B-type coefficients',
}

# ---- Regressand configuration (the default; overridden by --depvar in main) ----------------
# Default LHS is the QoQ deposit spread `spread_qoq` (the policy fed to BBL Step 2). `--depvar rate`
# switches the LHS to the ANNUALIZED deposit rate rate_ann = (1+rate_qoq)^4 - 1 and writes to a
# SEPARATE `polfunc_rate_*` namespace, so it never overwrites the spread policy Step 2 consumes.
CFG = {
    'regressand':  'spread_qoq',
    'out_prefix':  'polfunc',
    'tab_label':   'tab:polfunc',
    'caption':     'Policy Function Estimates for Endogenous Deposit Spreads (BBL Step~1)',
    'lhs_title':   'Deposit Spread',
    # Appended to the table caption. Empty for the spread tables (the headline): their caption is
    # just "Policy Function Estimates: <deposit type>".
    'caption_suffix': '',
    # 400 = 4 x 100: SIMPLE annualization of the QoQ spread, the same convention Step 2's
    # forward-sim uses (_POLFUNC_QOQ_TO_ANN_PP = 400.0). This is NOT the panel's exact compounded
    # spread_ann_a{k} = (1+rf)^4-(1+rate)^4, which differs by ~0.4pp on average (p99 ~0.9pp), so the
    # unit is labelled "simple-annualized" rather than "annualized". The REGRESSAND stays spread_qoq:
    # Step 2 consumes the QoQ policy, and OLS is scale-equivariant so this is display-only.
    'lhs_display': 400.0,
    'lhs_unit':    r'pp, simple-annualized ($4\times$QoQ)',
    'lhs_short':   'quarterly deposit spread',
    'depvar_note': (r'Dependent variable: quarterly deposit spread '
                    r'$\rho_{jkmt}=r^{f}_{t}-r^{\mathrm{dep}}_{jkmt}$, 2016--2024.'),
}

_CFG_RATE = {
    'regressand':  'rate_ann',
    'out_prefix':  'polfunc_rate',
    'tab_label':   'tab:polfunc_rate',
    'caption':     'Policy Function Estimates for the Annualized Deposit Rate (BBL Step~1)',
    'lhs_title':   'Annualized Deposit Rate',
    # NON-empty here on purpose: the rate tables are a robustness variant of the same regressions,
    # so without an LHS descriptor their captions would be identical to the spread tables'.
    'caption_suffix': r' --- Annualized Deposit Rate',
    # 100 = fraction -> pp. The regressand is ALREADY exactly compounded (rate_ann), so unlike the
    # spread variant this is the exact annualized rate, not a simple-annualization approximation.
    'lhs_display': 100.0,
    'lhs_unit':    'pp of the annualized deposit rate',
    'lhs_short':   'annualized deposit rate',
    'depvar_note': (r'Dependent variable: annualized deposit rate '
                    r'$r^{\mathrm{dep,ann}}_{jkmt}=(1+r^{\mathrm{dep}}_{jkmt})^{4}-1$, 2016--2024.'),
}


# ==============================================================================
# 1. Data Loading & Preparation
# ==============================================================================
def winsorize_within_type(df: pd.DataFrame, pct: float = None, verbose: bool = True) -> pd.DataFrame:
    """Clip WINSOR_VARS to their [pct, 1-pct] quantiles separately within B and within D.

    Returns the same frame (modified in place). Prints an audit line per variable that was
    actually clipped, so the effect on each regressor is visible in the run log."""
    pct = WINSOR_PCT if pct is None else pct
    lo_q, hi_q = pct, 1.0 - pct
    if verbose:
        print(f"  Winsorizing accounting ratios at {pct:.0%}/{1-pct:.0%} within firm type:")
    for v in WINSOR_VARS:
        if v not in df.columns:
            continue
        n_clip, before_max = 0, pd.to_numeric(df[v], errors='coerce').max()
        for is_b in (True, False):
            mask = df['is_B'] == is_b
            s = pd.to_numeric(df.loc[mask, v], errors='coerce')
            if s.notna().sum() < 100:          # too few to form stable percentiles
                continue
            lo, hi = s.quantile(lo_q), s.quantile(hi_q)
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
                continue
            clipped = s.clip(lo, hi)
            # NaN != NaN is True in pandas, so guard with notna() or the count degenerates into
            # the missing-value count (which is ~37k here and would badly overstate the clipping).
            n_clip += int(((clipped != s) & s.notna()).sum())
            df.loc[mask, v] = clipped
        if verbose and n_clip:
            after_max = pd.to_numeric(df[v], errors='coerce').max()
            print(f"    {v:28s} clipped {n_clip:>6,d} obs   max {before_max:>12.4f} -> {after_max:.4f}")
    return df


def load_and_prepare_panel() -> pd.DataFrame:
    """Load market_panel.csv, reshape to long for k=4,5, and construct lags.

    Returns a long-format DataFrame with one row per (conglomerate, deposit_type,
    mca_code, year, quarter) tuple, restricted to deposit types 4 and 5.
    """
    print(f"Loading panel from {PANEL_CSV}...")
    df_raw = pd.read_csv(PANEL_CSV, dtype={'mca_code': str}, low_memory=False)
    print(f"  Raw panel: {len(df_raw):,} rows, {len(df_raw.columns)} columns")

    # Restrict to the estimation window (2016-2024), matching the sleep/demand/cost stages.
    # This is not cosmetic: the policy regressors are bank characteristics, which do not exist on the
    # prudential-conglomerate basis before 2016.  Fitting/predicting outside the window drove
    # compute_fitted_values' fillna(0) to return the bare intercept (~190pp — an impossible deposit
    # spread) for ~19% of k∈{4,5} rows, all of them 2013-2015.  Windowing removes that fabrication at
    # source rather than filtering it downstream.  See counterfactuals_plan.md §0A.
    if 'year' in df_raw.columns:
        _n0 = len(df_raw)
        df_raw = df_raw[(df_raw['year'] >= POLFUNC_MIN_YEAR) & (df_raw['year'] <= POLFUNC_MAX_YEAR)].copy()
        print(f"  Year window [{POLFUNC_MIN_YEAR}, {POLFUNC_MAX_YEAR}]: kept {len(df_raw):,} of {_n0:,} rows")

    # ----- Reshape wide → long for k=4,5 ---------------------------------
    id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
    df_raw = df_raw.drop_duplicates(subset=id_vars)

    df = pd.wide_to_long(
        df_raw,
        stubnames=['dep_a', 'spread_a', 'rate_a'],
        i=id_vars,
        j='deposit_type'
    ).reset_index()

    df = df.rename(columns={'dep_a': 'deposit_balance', 'spread_a': 'spread_qoq',
                            'rate_a': 'rate_qoq'})

    # Restrict to endogenous deposit types
    df = df[df['deposit_type'].isin(K_ENDOG)].copy()
    print(f"  After restricting to k in {K_ENDOG}: {len(df):,} rows")

    # Annualized deposit rate (compound) — the alternative regressand for --depvar rate.
    # Verified to machine precision that the panel builds spread_ann = (1+rf_qoq)^4 - (1+rate_qoq)^4,
    # so this annualizes the deposit rate the SAME way the panel annualizes the spread.
    df['rate_qoq'] = pd.to_numeric(df['rate_qoq'], errors='coerce')
    df['rate_ann'] = (1.0 + df['rate_qoq']) ** 4 - 1.0

    # ----- Firm type classification ---------------------------------------
    df['is_B'] = (df['CODMUN_IBGE'].astype(str) != '0')

    # ----- Entity key & time identifiers ----------------------------------
    df['entity_id'] = (
        df['CodConglomeradoPrudencial'].astype(str) + '_'
        + df['deposit_type'].astype(str) + '_'
        + df['mca_code'].astype(str)
    )
    df['time_id'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)
    df.sort_values(by=['entity_id', 'year', 'quarter'], inplace=True)

    # ----- Coerce all regressor columns to numeric ------------------------
    # Some columns (e.g. has_ip) may arrive as object dtype from the CSV.
    all_regressor_cols = (
        COMMON_REGRESSORS + DEMOGRAPHICS
        + [f'{c}_sq' for c in QUADRATIC_BASE]
    )
    for col in all_regressor_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ----- Winsorize the accounting ratios within firm type ---------------
    df = winsorize_within_type(df)

    # ----- Quadratic terms (built AFTER winsorizing, so squares of a clipped
    #       base stay consistent with the base itself) ---------------------
    for col in QUADRATIC_BASE:
        if col in df.columns:
            df[f'{col}_sq'] = df[col].astype(float) ** 2

    # ----- Drop rows with missing dependent variable ----------------------
    regressand = CFG['regressand']
    n_before = len(df)
    df = df.dropna(subset=[regressand])
    print(f"  Dropped {n_before - len(df):,} rows with missing {regressand}")

    print(f"  Final panel: {len(df):,} rows "
          f"(B={df['is_B'].sum():,}, D={(~df['is_B']).sum():,})")
    return df


# ==============================================================================
# 2. National Demographics Computation
# ==============================================================================
def compute_national_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute population-weighted national averages of MCA demographics.

    For each (year, quarter), average across MCAs weighted by pop_total.
    These national values are then assigned to D-type firm rows.

    Returns the input DataFrame with additional columns `{demo}_natl` for
    each demographic variable.
    """
    print("  Computing population-weighted national demographics...")

    # Use only B-firm rows with valid population for weighting
    df_b = df[df['is_B']].copy()
    demo_plus_pop = DEMOGRAPHICS + ['pop_total']
    cols_needed = ['year', 'quarter', 'mca_code'] + demo_plus_pop
    cols_avail = [c for c in cols_needed if c in df_b.columns]
    df_mca = df_b[cols_avail].drop_duplicates(subset=['mca_code', 'year', 'quarter'])
    df_mca = df_mca.dropna(subset=['pop_total'])

    def _weighted_mean(group: pd.DataFrame, col: str) -> float:
        vals = group[col]
        wgts = group['pop_total']
        valid = vals.notna() & wgts.notna() & (wgts > 0)
        if valid.sum() == 0:
            return np.nan
        return np.average(vals[valid], weights=wgts[valid])

    natl_records = []
    for (yr, qtr), grp in df_mca.groupby(['year', 'quarter']):
        rec = {'year': yr, 'quarter': qtr}
        for col in DEMOGRAPHICS:
            if col in grp.columns:
                rec[f'{col}_natl'] = _weighted_mean(grp, col)
            else:
                rec[f'{col}_natl'] = np.nan
        natl_records.append(rec)

    df_natl = pd.DataFrame(natl_records)

    # Merge national demographics onto the main panel
    df = df.merge(df_natl, on=['year', 'quarter'], how='left')

    n_filled = sum(1 for c in DEMOGRAPHICS if f'{c}_natl' in df.columns)
    print(f"    Created {n_filled} national demographic columns")

    return df


# ==============================================================================
# 3. Cluster-Robust Standard Errors (IK2016 / Carter et al. 2017)
# ==============================================================================
def cluster_structure(cluster_series: pd.Series):
    """Descriptive cluster structure: (G, G*) with G* = G/(1+cv²) the effective cluster
    count of Imbens-Kolesar (2016) / Carter-Schnepel-Steigerwald (2017).

    NOTE (2026-07-20): G* is now reported as a DESCRIPTIVE statistic only — it is the
    cluster-paucity measure that MOTIVATES the wild cluster bootstrap (see desc_3.py), not
    the inference itself. Inference used to be CRVE + t(G*) here; it is now the same
    score/multiplier wild cluster bootstrap the rest of the paper uses, so every standard
    error in the paper is produced by one scheme. See run_single_regression.
    """
    sizes = cluster_series.value_counts()
    G_nominal = int(len(sizes))
    cv_Ng = float(np.std(sizes, ddof=0) / np.mean(sizes)) if np.mean(sizes) > 0 else 0.0
    G_star = float(max(1.0, G_nominal / (1 + cv_Ng ** 2)))
    return G_nominal, G_star


# ==============================================================================
# 4. Policy Function OLS Estimation
# ==============================================================================
def _build_regressor_list(firm_type: str, d_option: str) -> list:
    """Determine which regressors enter the policy function regression.

    Parameters
    ----------
    firm_type : {'B', 'D', 'pooled'}
    d_option  : {'A', 'B', 'C'}  (only relevant for D and pooled)

    Returns a list of column names to use as regressors (will be intersected
    with available columns at estimation time).
    """
    quad_cols = [f'{c}_sq' for c in QUADRATIC_BASE]
    base = COMMON_REGRESSORS + quad_cols

    if firm_type == 'B':
        # B-firms always get local MCA demographics
        return base + DEMOGRAPHICS

    elif firm_type == 'D':
        if d_option == 'A':
            # No demographics
            return base
        elif d_option == 'B':
            # National (pop-weighted) demographics
            return base + [f'{c}_natl' for c in DEMOGRAPHICS]
        else:
            # Option C handled via pooled
            raise ValueError("D-firm option C should use pooled estimation")

    elif firm_type == 'pooled':
        # Same regressors as B-firms; D-firms get national demos mapped
        # to the same column names (done at data prep)
        return base + DEMOGRAPHICS

    else:
        raise ValueError(f"Unknown firm_type: {firm_type}")


def apply_centering(frame: pd.DataFrame, cols: list, means: dict) -> pd.DataFrame:
    """Subtract `means` from the continuous columns of `frame`, then REBUILD every `_sq` column
    from its freshly centered base.

    Shared by estimation and prediction so both use the identical transform -- the prediction
    path must reuse the ESTIMATION means (passed in), never recompute its own, or the fitted
    values silently shift. Modifies and returns `frame`.
    """
    # Bases first...
    for c in cols:
        if c.endswith('_sq') or c not in means or c not in frame.columns:
            continue
        frame[c] = frame[c] - means[c]
    # ...then the squares, from the now-centered bases.
    for c in cols:
        if not c.endswith('_sq'):
            continue
        base = c[:-3]
        if base in frame.columns and base in means:
            frame[c] = frame[base] ** 2
    return frame


def centering_means(df_w: pd.DataFrame, avail: list) -> dict:
    """Mean of every centerable regressor in this estimation sample (dummies and `_sq` excluded;
    the squares are not centered, they are rebuilt from centered bases)."""
    return {c: float(df_w[c].mean()) for c in avail
            if c not in NO_CENTER_VARS and not c.endswith('_sq') and c in df_w.columns}


def run_single_regression(
    df_sub: pd.DataFrame,
    dep_var: str,
    regressors: list,
    label: str,
) -> dict:
    """Run a single OLS policy function regression with cluster-robust SEs.

    Parameters
    ----------
    df_sub     : subset of panel for this regression
    dep_var    : dependent variable column name
    regressors : list of regressor column names
    label      : human-readable label for logging

    Returns
    -------
    dict with keys: 'label', 'res' (statsmodels result), 'n_obs', 'n_clusters',
    'G_star', 'r_squared', 'regressors', 'coefficients', 'std_errors', 'pvalues'
    """
    # Intersect with available columns
    avail = [c for c in regressors if c in df_sub.columns]
    missing = set(regressors) - set(avail)
    if missing:
        print(f"    [{label}] Regressors not in data (dropped): {missing}")

    # Working copy with complete cases
    cols_needed = [dep_var] + avail + ['CodConglomeradoPrudencial', 'year']
    df_w = df_sub[cols_needed].dropna().copy()

    if len(df_w) < len(avail) + 1:
        print(f"    [{label}] Insufficient observations ({len(df_w)}). Skipping.")
        return None

    # CENTER the continuous regressors (see NO_CENTER_VARS above). Pure reparametrization:
    # the intercept becomes the prediction at the average state; fit and fitted values are
    # unchanged. The means are stored and REUSED verbatim at prediction time.
    center_means = centering_means(df_w, avail)
    df_w = apply_centering(df_w, avail, center_means)

    y = df_w[dep_var]
    X = sm.add_constant(df_w[avail], has_constant='add')

    # CLUSTER AT THE CONGLOMERATE LEVEL — the same unit as every other stage of the paper
    # (estimation_{1,2}_sleep.py first/second stages, the BLP stage). This previously
    # clustered on bank×year, a finer and therefore LESS CONSERVATIVE unit, which left the
    # policy function inconsistent with everything it feeds.
    cluster = df_w['CodConglomeradoPrudencial'].astype(str)
    model = sm.OLS(y, X)
    res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster}, use_t=True)

    # G / G*: descriptive cluster structure (the paucity that motivates the WCB), not the inference.
    G_nominal, G_star = cluster_structure(cluster)

    # INFERENCE: score/multiplier wild cluster bootstrap — the SAME function the sleepiness
    # stages call (utils.sleep_links; Cameron-Gelbach-Miller 2008, MacKinnon-Webb 2017), and
    # the same B/scheme config (SLEEP_BOOT_B / SLEEP_BOOT_SCHEME), so every SE in the paper is
    # produced one way. Replaces the old CRVE + t(G*) ("IK2016") path.
    from utils.sleep_links import boot_cfg, linear_wild_cluster_bootstrap
    B, scheme = boot_cfg()
    bse, tvals, pvals = linear_wild_cluster_bootstrap(res, B=B, scheme=scheme, seed=0)

    coef_names = list(res.params.index)
    out = {
        'label': label,
        'res': res,
        'n_obs': int(res.nobs),
        'n_clusters': G_nominal,
        'G_star': G_star,
        'r_squared': float(res.rsquared),
        'r_squared_adj': float(res.rsquared_adj),
        'regressors': coef_names,
        'coefficients': res.params.to_dict(),
        'std_errors': {k: float(v) for k, v in bse.items()},
        'pvalues': {k: float(v) for k, v in pvals.items()},
        'se_method': f'wild cluster bootstrap (B={B}, {scheme}); clusters = conglomerate',
        # Anchor for reading the intercept. The reported constant is the prediction at x=0, which
        # is far outside the support (log assets = 0 means assets of R$1), so it is large and not
        # interpretable on its own -- for B/k=4 it is +170pp purely to offset the log-assets terms
        # (-375pp linear, +209pp quadratic, netting ~+3pp at the median). OLS with an intercept
        # forces mean(fitted) == mean(y), so the fitted value AT THE SAMPLE MEAN state is exactly
        # this number; reporting it gives the reader the scale anchor the constant does not.
        'mean_depvar': float(y.mean()),
        # The estimation-sample means used to center; compute_fitted_values MUST reuse these.
        'center_means': center_means,
    }

    print(f"    [{label}] N={out['n_obs']:,} | R²={out['r_squared']:.4f} "
          f"| G={out['n_clusters']} | G*={out['G_star']:.2f}")

    return out


def _prepare_pooled_data(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Prepare pooled dataset for Option C: map national demographics into
    the same column names as local demographics for D-firm rows."""
    df_k = df[df['deposit_type'] == k].copy()

    # For D-firm rows, replace local demographic columns with national values
    d_mask = ~df_k['is_B']
    for col in DEMOGRAPHICS:
        natl_col = f'{col}_natl'
        if natl_col in df_k.columns and col in df_k.columns:
            df_k.loc[d_mask, col] = df_k.loc[d_mask, natl_col]

    return df_k


def estimate_all_policy_functions(df: pd.DataFrame) -> dict:
    """Run all policy function regressions across deposit types and options.

    Returns a dict keyed by regression label with regression result dicts.
    """
    results = {}
    tasks = []

    for k in K_ENDOG:
        k_label = K_LABELS[k]
        df_k = df[df['deposit_type'] == k]
        df_B = df_k[df_k['is_B']].copy()
        df_D = df_k[~df_k['is_B']].copy()

        # ---- B-type regression (same across all D-options) ---------------
        regs_B = _build_regressor_list('B', 'A')
        label_B = f"k{k}_{k_label}_B"
        tasks.append((df_B, CFG['regressand'], regs_B, label_B))

        # ---- D-type regressions (options A and B) ------------------------
        for opt in ['A', 'B']:
            regs_D = _build_regressor_list('D', opt)
            label_D = f"k{k}_{k_label}_D_opt{opt}"
            tasks.append((df_D, CFG['regressand'], regs_D, label_D))

        # ---- Pooled regression (option C) --------------------------------
        df_pooled = _prepare_pooled_data(df, k)
        regs_pooled = _build_regressor_list('pooled', 'C')
        label_pooled = f"k{k}_{k_label}_pooled_optC"
        tasks.append((df_pooled, CFG['regressand'], regs_pooled, label_pooled))

    # Run regressions in parallel via ThreadPoolExecutor
    # (statsmodels OLS releases GIL during BLAS calls)
    print(f"\n  Submitting {len(tasks)} regressions to ThreadPoolExecutor...")
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 4)) as pool:
        futures = {
            pool.submit(run_single_regression, *task): task[3]
            for task in tasks
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                res = future.result()
                if res is not None:
                    results[label] = res
            except Exception as exc:
                print(f"    [{label}] FAILED: {exc}")

    elapsed = time.perf_counter() - t0
    print(f"  All regressions completed in {elapsed:.1f}s")

    return results


# ==============================================================================
# 5. Fitted Values & Output
# ==============================================================================
def compute_fitted_values(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Compute fitted spreads from estimated policy functions.

    For each regression result, produce predicted values for the corresponding
    subset of the data. Returns a DataFrame with fitted value columns appended.
    """
    df_out = df[['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter',
                 'deposit_type', 'is_B', CFG['regressand'], 'entity_id', 'time_id']].copy()

    for label, res_dict in results.items():
        res = res_dict['res']
        coef_names = res_dict['regressors']

        # Determine which rows this regression applies to
        parts = label.split('_')
        k = int(parts[0][1])  # e.g., "k4" → 4
        is_pooled = 'pooled' in label

        if is_pooled:
            # Option C: pooled, use national demos for D-firms
            mask = df_out['deposit_type'] == k
            df_pred = _prepare_pooled_data(df, k)
        elif '_B' in label and 'opt' not in label:
            mask = (df_out['deposit_type'] == k) & df_out['is_B']
            df_pred = df[mask].copy()
        elif '_D_' in label:
            mask = (df_out['deposit_type'] == k) & (~df_out['is_B'])
            df_pred = df[mask].copy()
        else:
            continue

        # Build X matrix
        reg_cols = [c for c in coef_names if c != 'const']
        avail_cols = [c for c in reg_cols if c in df_pred.columns]

        if not avail_cols:
            continue

        # PREDICT ON COMPLETE CASES ONLY. This used to be `df_pred[avail_cols].fillna(0)`, which does
        # NOT impute a neutral value — it forces every missing regressor to zero, so a row missing all
        # of them collapses to the bare intercept. For k=4 that intercept is ~0.475 qoq-frac = ~190pp
        # annualized: an impossible deposit spread, silently written out as if it were a fitted policy.
        # It hit ~26% of rows before the 2016-2024 window and still ~3.6% after (the 2016Q1 rows, whose
        # _lag bank characteristics lag into the excluded 2015). Leave incomplete cases as NaN instead;
        # the consumer (estimation_bbl_2_fwd_sim::equilibrium_spreads) already falls back to the observed spread
        # for non-finite fits. Fabricating is never better than admitting the gap.
        # Apply the ESTIMATION centering (never recomputed here -- reusing the stored means is
        # what keeps the fitted values identical to the uncentered parametrization).
        df_pred = apply_centering(df_pred, avail_cols, res_dict.get('center_means', {}))

        complete = df_pred[avail_cols].notna().all(axis=1)
        X_pred = sm.add_constant(df_pred.loc[complete, avail_cols], has_constant='add')

        # Align columns with the estimated model
        for c in coef_names:
            if c not in X_pred.columns:
                X_pred[c] = 0.0
        X_pred = X_pred[coef_names]

        # Build a df_pred-length vector, NaN on the incomplete rows, and assign POSITIONALLY into
        # df_out[mask] — exactly the alignment the original used (the pooled branch's df_pred does not
        # necessarily share df_out's index, so an index-based write could silently misalign).
        fitted_full = pd.Series(np.nan, index=df_pred.index, dtype=float)
        if len(X_pred):
            fitted_full.loc[X_pred.index] = (X_pred @ res.params).astype(float)

        col_name = f'fitted_{label}'
        df_out.loc[mask, col_name] = fitted_full.values
        n_inc = int((~complete).sum())
        if n_inc:
            print(f"    [{label}] {n_inc:,} of {len(df_pred):,} rows left NaN "
                  f"(incomplete regressors — not fabricated to the intercept)")

    return df_out


def save_outputs(results: dict, df_fitted: pd.DataFrame) -> None:
    """Persist all estimation outputs to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Pickle with full statsmodels results --------------------------
    pkl_path = OUTPUT_DIR / f"{CFG['out_prefix']}_results.pkl"
    # Strip non-picklable items for safety; keep full result objects
    pkl_data = {}
    for label, res_dict in results.items():
        pkl_data[label] = {
            k: v for k, v in res_dict.items()
        }
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_data, f)
    print(f"  Saved pickle: {pkl_path.name}")

    # --- 2. Fitted values CSV ---------------------------------------------
    csv_path = OUTPUT_DIR / f"{CFG['out_prefix']}_fitted.csv"
    df_fitted.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved fitted values: {csv_path.name}")

    # --- 3. Summary JSON --------------------------------------------------
    summary = {}
    for label, res_dict in results.items():
        summary[label] = {
            'n_obs': res_dict['n_obs'],
            'n_clusters': res_dict['n_clusters'],
            'G_star': res_dict['G_star'],
            'r_squared': res_dict['r_squared'],
            'r_squared_adj': res_dict['r_squared_adj'],
            'coefficients': res_dict['coefficients'],
            'std_errors': res_dict['std_errors'],
            'pvalues': res_dict['pvalues'],
        }

    json_path = OUTPUT_DIR / f"{CFG['out_prefix']}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved summary JSON: {json_path.name}")

    # --- 4. LaTeX tables per regression -----------------------------------
    for label, res_dict in results.items():
        tex_path = OUTPUT_DIR / f"{CFG['out_prefix']}_{label}.tex"
        try:
            with open(tex_path, 'w') as f:
                f.write(res_dict['res'].summary().as_latex())
        except Exception as exc:
            print(f"  [WARN] LaTeX export failed for {label}: {exc}")
    print(f"  Saved {len(results)} LaTeX tables")


def print_summary_table(results: dict) -> None:
    """Print a compact summary table of all regressions."""
    print("\n" + "=" * 80)
    print("  POLICY FUNCTION ESTIMATION SUMMARY")
    print("=" * 80)
    header = f"  {'Label':<40s} {'N':>8s} {'R2':>8s} {'G':>5s} {'G*':>7s}"
    print(header)
    print("  " + "-" * 70)
    for label, res_dict in sorted(results.items()):
        print(f"  {label:<40s} {res_dict['n_obs']:>8,d} "
              f"{res_dict['r_squared']:>8.4f} {res_dict['n_clusters']:>5d} "
              f"{res_dict['G_star']:>7.2f}")
    print("=" * 80 + "\n")


# ==============================================================================
# 7. LaTeX Export & PDF Compilation
# ==============================================================================
DRAFTS_DIR = _ROOT / "Drafts" / "Deposit Competition"

# Human-readable labels for variable names in LaTeX
_VAR_LABELS = {
    'const': 'Constant',
    'log_total_assets_lag': 'Log Total Assets ($t-1$)',
    'equity_ratio_lag': 'Equity Ratio ($t-1$)',
    'has_ip': 'IP Subsidiary',
    'asset_return_qoq_lag': 'Asset Return (QoQ, $t-1$)',
    'npl_provision_ratio_lag': 'NPL Provisions Ratio ($t-1$)',
    'credit_assets_lag': 'Credit / Assets ($t-1$)',
    'seg_S2': 'Segment S2',
    'seg_S3': 'Segment S3',
    'seg_S4': 'Segment S4',
    'seg_S5': 'Segment S5',
    'personnel_cost_ratio_lag': 'Personnel Cost Ratio ($t-1$)',
    'admin_cost_ratio_lag': 'Admin Cost Ratio ($t-1$)',
    'tax_cost_ratio_lag': 'Tax Cost Ratio ($t-1$)',
    'indice_basileia_lag': 'Basel Index ($t-1$)',
    'wholesale_ratio_lag': 'Wholesale Ratio ($t-1$)',
    'lci_lca_ratio_lag': 'LCI/LCA Ratio ($t-1$)',
    'risk_free_qoq': 'Risk-Free Rate (QoQ)',
    'log_total_assets_lag_sq': 'Log Total Assets$^2$',
    'equity_ratio_lag_sq': 'Equity Ratio$^2$',
    'risk_free_qoq_sq': 'Risk-Free Rate$^2$',
    'gdp_per_capita': r'GDP \textit{per capita}',
    'fraction_65plus': 'Fraction 65+',
    'fraction_young': 'Fraction Young',
    'pix_users_pf_per1000': 'Pix Users',
    'connections_per100': 'Broadband Connections',
    'branches_per1000': 'Branches',
    'cadunico_families_per1000': 'CadUnico Families',
    'gdp_per_capita_natl': r'GDP \textit{per capita} (Natl.)',
    'fraction_65plus_natl': 'Fraction 65+ (Natl.)',
    'fraction_young_natl': 'Fraction Young (Natl.)',
    'pix_users_pf_per1000_natl': 'Pix Users (Natl.)',
    'connections_per100_natl': 'Broadband Connections (Natl.)',
    'branches_per1000_natl': 'Branches (Natl.)',
    'cadunico_families_per1000_natl': 'CadUnico Families (Natl.)',
}


def _stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.10:
        return '*'
    return ''


def _clean_var(v: str) -> str:
    """Map a variable name to its LaTeX-safe label."""
    return _VAR_LABELS.get(v, v.replace('_', '\\_'))


def _polfunc_col_keys(k: int) -> list:
    """The four regression labels (columns) for deposit type k."""
    kl = K_LABELS[k]
    # NOTE: the pooled (Option C) regression is still estimated and stored in the summary/pickle as a
    # robustness record, but it is NOT displayed: Step 2 consumes only B and D_optB, and pooling
    # few-but-huge B clusters with many-small D ones collapses the effective cluster count.
    return [f'k{k}_{kl}_B', f'k{k}_{kl}_D_optA', f'k{k}_{kl}_D_optB']


# Column headers, in the order of _polfunc_col_keys.
# The two D columns sit under ONE spanning 'D-type' header, formatted exactly like 'B-type'.
# They stay distinguishable without per-column labels because the Demographics indicator row
# reads Yes/No/Yes: the D column marked No is the no-demographics specification, the one marked
# Yes carries the population-weighted national demographics.
_COL_HEADERS = ['B-type', 'D-type']

# Demographic bases whose local and `_natl` variants are collapsed to ONE display row:
# each column shows its own geography (B/Pooled = local MCA, D natl. = national averages).
_DEMO_BASES = ['gdp_per_capita', 'fraction_65plus', 'fraction_young',
               'pix_users_pf_per1000', 'connections_per100', 'branches_per1000',
               'cadunico_families_per1000']


def _polfunc_cell(res_dict, v):
    """(coef, se, p) for variable v in a regression result dict, or None if absent."""
    if v in res_dict['coefficients']:
        return res_dict['coefficients'][v], res_dict['std_errors'][v], res_dict['pvalues'][v]
    return None


def _polfunc_variant_cell(res_dict, base):
    """For a demographic base, return the cell for whichever variant (base or base_natl)
    this column actually uses."""
    for v in (base, base + '_natl'):
        c = _polfunc_cell(res_dict, v)
        if c is not None:
            return c
    return None


def _polfunc_emit_rows(label, cells, rows, mult=1.0):
    """Append a coefficient line + a standard-error line for one regressor across columns.

    `mult` = LHS_display · unit_scale rescales coefficient AND standard error together (a pure
    change of units, so t-stats and stars are unaffected)."""
    coef_cells, se_cells = [], []
    for c in cells:
        if c is None:
            coef_cells.append('')
            se_cells.append('')
        else:
            cf, se, p = c
            coef_cells.append(f'${cf * mult:.4f}^{{{_stars(p)}}}$')
            se_cells.append(f'$({se * mult:.4f})$')
    rows.append(f'{label} & ' + ' & '.join(coef_cells) + r' \\')
    rows.append(' & ' + ' & '.join(se_cells) + r' \\')


def _row_label(v):
    """Row label, WITHOUT a bracketed unit. The display SCALING still applies (see
    _display_unit / _DISPLAY_UNITS); only the '[unit]' suffix is omitted, because the units are
    stated in the table notes and in the surrounding prose instead of in column 1."""
    return _clean_var(v)


def _polfunc_panel_rows(results: dict, k: int, include_segments: bool = False) -> list:
    """LaTeX coefficient rows for one deposit type k across the four columns.

    Non-demographic regressors are the ordered union; demographics are merged (local + national
    into one row, each column showing its own variant). Segment dummies are EXCLUDED unless
    `include_segments` — in the main table they collapse to a 'Segment FE: Yes' indicator row.
    Every coefficient is shown in its display unit (see _DISPLAY_UNITS)."""
    cols = _polfunc_col_keys(k)
    demo_all = set(_DEMO_BASES) | {d + '_natl' for d in _DEMO_BASES}
    lhs = float(CFG.get('lhs_display', 1.0))

    # ordered union of NON-demographic regressors (first appearance across columns)
    order, seen = [], set()
    for ck in cols:
        if ck not in results:
            continue
        for v in results[ck]['regressors']:
            if v in demo_all or v in seen:
                continue
            if v in _SEGMENT_VARS and not include_segments:
                continue
            order.append(v)
            seen.add(v)

    rows = []
    for v in order:
        cells = [_polfunc_cell(results[ck], v) if ck in results else None for ck in cols]
        _polfunc_emit_rows(_row_label(v), cells, rows, mult=lhs * _display_unit(v)[0])

    # Merged demographic rows -- DETAIL ONLY. In the main table they collapse to a
    # 'Demographics' indicator row (Yes/No/Yes: B carries local MCA demographics, D~(no demo.)
    # carries none, D~(natl. demo.) carries national ones), exactly like the segment dummies.
    demo_rows = []
    if not include_segments:
        return rows
    for base in _DEMO_BASES:
        cells = [_polfunc_variant_cell(results[ck], base) if ck in results else None for ck in cols]
        if all(c is None for c in cells):
            continue
        _polfunc_emit_rows(_row_label(base), cells, demo_rows, mult=lhs * _display_unit(base)[0])
    if demo_rows:
        rows.append(r'\addlinespace[0.3ex]')
        rows.extend(demo_rows)
    return rows


def _polfunc_stat_row(label, values, fmt) -> str:
    cells = [fmt(v) if v is not None else '' for v in values]
    return f'{label} & ' + ' & '.join(cells) + r' \\'


def _polfunc_notes() -> str:
    """Table Notes: the inference paragraph only.

    Deliberately short. Everything the notes used to carry -- units, winsorization, centering,
    the segment/demographic indicators, the column definitions -- is documented in the prose of
    the paper and in this module\'s comments, and repeating it under every table crowded out the
    one thing a reader needs at the table: how the standard errors were produced.

    Citations are real \\parencite keys, not typeset-by-hand author strings, so they resolve
    against References.bib and stay correct if an entry changes. The paper uses biblatex/biber
    (authoryear-comp), where \\parencite is the parenthetical form.
    """
    return (
        r'\textit{Notes:} Standard errors in parentheses are from a score/multiplier wild cluster '
        r'bootstrap at the conglomerate level (Webb weights) '
        r'\parencite{cameron2008bootstrap,mackinnon2017wild,webb2023reworking}. '
        r'$G$ and the effective cluster count '
        r'$G^{*}=G/(1+\mathrm{cv}^{2})$ \parencite{imbens2016robust,carter2017asymptotic} are '
        r'reported as the cluster-paucity statistics that motivate the bootstrap, not as the '
        r'inference. *** $p<0.01$, ** $p<0.05$, * $p<0.1$. '
        + CFG['depvar_note']
    )


_K_TITLE = {4: r'Time Deposits / CDB ($k=4$)', 5: r'Prepaid Accounts ($k=5$)'}

# Segment dummies: absorbed out of the main table into a "Segment FE: Yes" indicator row, and
# shown explicitly only in the `_segment` companion table.
_SEGMENT_VARS = ['seg_S2', 'seg_S3', 'seg_S4', 'seg_S5']

# ---- Display units -------------------------------------------------------------------------
# Coefficients are reported per the unit in brackets in the row label. Units are pinned to the
# REST OF THE PAPER, not chosen freely: any regressor that also appears in the sleepiness/demand
# tables is displayed in the unit it carries there, so a coefficient here is comparable to one
# there. That pinning is now ENFORCED rather than documented: the unit table lives once, in
# utils/state_transform.DISPLAY, and the sleepiness exporters, the descriptive tables and this
# module all read the same object. Shares and rates are in percentage points throughout
# (fraction_65plus / fraction_young moved from "fraction" to "pp" when the sleepiness tables
# did, which is the whole reason the two are still comparable).
# Re-exported for `make_polfunc_md_table.py`, which imports this name and indexes it as a
# dict (`_DISPLAY_UNITS.get(base, (1.0, ""))`). It IS the shared registry table, not a copy,
# so there is still exactly one source of truth for units across the paper.
from utils.state_transform import DISPLAY as _DISPLAY_UNITS  # noqa: E402,F401


def _display_unit(v):
    """(scale, unit label) for a regressor; `_natl` variants inherit the base unit.

    The table now lives in utils/state_transform.DISPLAY, shared with the sleepiness
    exporters and the descriptive tables. The comment above used to ask a reader to keep
    this dict in sync with export_sleep_link_common.py by hand; the units are now a single
    object, so cross-table comparability is structural rather than aspirational.

    NOTE the polfunc builds its regressors in RAW panel units (it does not go through
    state_transform.apply_scale), which is why `scaled=False` is the right convention
    here and the returned scale is the display unit measured in raw units."""
    from utils.state_transform import display_unit
    return display_unit(v)


def build_polfunc_table(results: dict, k: int, include_segments: bool = False) -> str:
    """Paper-style, page-breaking `xltabular` fragment for ONE deposit type k
    (four columns: B / D no-demo / D natl.-demo / Pooled). Bare fragment — `\\input`-able
    into V_Main.tex (the paper preamble already provides xltabular/booktabs/setspace).

    `include_segments=False` (the main table) hides the four segment dummies and reports them as a
    'Segment FE: Yes' indicator row; `True` builds the `_segment` companion that shows them."""
    colspec = r'>{\raggedright\arraybackslash}p{5.4cm} *{3}{>{\centering\arraybackslash}X}'
    # 'D-type' spans the two D columns via \multicolumn. No \cmidrule under it: the \midrule that
    # follows already closes the header, and adding one stacks a second rule on top of it.
    head = ' & ' + _COL_HEADERS[0] + r' & \multicolumn{2}{c}{' + _COL_HEADERS[1] + r'} \\'
    caption = 'Policy Function Estimates: ' + _K_TITLE[k] + CFG.get('caption_suffix', '')
    label = CFG['tab_label'] + f'_k{k}' + ('_segment' if include_segments else '')
    cols = _polfunc_col_keys(k)

    L = [
        r'\begin{spacing}{1.0}',
        r'\footnotesize',
        r'\begin{xltabular}{\textwidth}{' + colspec + '}',
        r'\caption{' + caption + '}',
        r'\label{' + label + r'} \\',
        r'\toprule',
        head,
        r'\midrule',
        r'\endfirsthead',
        r'\multicolumn{4}{c}{\bfseries Table \thetable\ (continued)} \\',
        r'\toprule',
        head,
        r'\midrule',
        r'\endhead',
        r'\midrule',
        r'\multicolumn{4}{r}{\textit{Continued on next page}} \\',
        r'\endfoot',
        r'\bottomrule',
        r'\multicolumn{4}{@{}p{\dimexpr\textwidth-2\tabcolsep\relax}@{}}{\scriptsize '
        + _polfunc_notes() + r'} \\',
        r'\endlastfoot',
    ]

    L.extend(_polfunc_panel_rows(results, k, include_segments=include_segments))
    L.append(r'\midrule')
    r2 = [results[ck]['r_squared'] if ck in results else None for ck in cols]
    obs = [results[ck]['n_obs'] if ck in results else None for ck in cols]
    G = [results[ck]['n_clusters'] if ck in results else None for ck in cols]
    Gs = [results[ck]['G_star'] if ck in results else None for ck in cols]
    if not include_segments:
        # Segment dummies are in every regression; report them as an FE indicator.
        present = ['Yes' if (ck in results and any(v in results[ck]['coefficients']
                                                   for v in _SEGMENT_VARS)) else 'No'
                   for ck in cols]
        L.append('Segment FE & ' + ' & '.join(present) + r' \\')
        # Demographics: Yes where the column actually carries demographic regressors. This reads
        # Yes/No/Yes -- B has local MCA demographics, D (no demo.) has none by construction, and
        # D (natl. demo.) has the population-weighted national ones.
        _demo_all = set(_DEMO_BASES) | {d + '_natl' for d in _DEMO_BASES}
        demo_present = ['Yes' if (ck in results and any(v in results[ck]['coefficients']
                                                        for v in _demo_all)) else 'No'
                        for ck in cols]
        L.append('Demographics & ' + ' & '.join(demo_present) + r' \\')
    # Scale anchor for the intercept: the constant is the prediction at x=0 (assets of R$1), which
    # is out of support and therefore large; mean(fitted) == mean(y) under OLS, so this row IS the
    # fitted value at the average state and is what the reader should read the levels against.
    _lhs = float(CFG.get('lhs_display', 1.0))
    mdv = [results[ck]['mean_depvar'] * _lhs if ck in results and 'mean_depvar' in results[ck]
           else None for ck in cols]
    L.append(_polfunc_stat_row('Mean dep.\\ var.', mdv, lambda x: f'{x:.4f}'))
    L.append(_polfunc_stat_row(r'$R^{2}$', r2, lambda x: f'{x:.4f}'))
    L.append(_polfunc_stat_row('Observations', obs, lambda x: f'{x:,}'))
    L.append(_polfunc_stat_row(r'Clusters ($G$)', G, lambda x: f'{x}'))
    L.append(_polfunc_stat_row(r'Eff.\ clusters ($G^{*}$)', Gs, lambda x: f'{x:.1f}'))

    L.append(r'\end{xltabular}')
    L.append(r'\end{spacing}')
    return '\n'.join(L)


def write_polfunc_fragments(results: dict) -> dict:
    """Write the `\\input`-able fragments: one MAIN table per deposit type (segment dummies
    collapsed to an FE indicator) plus a `_segment` companion that shows them explicitly."""
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    frags = {}
    for k in K_ENDOG:
        for seg in (False, True):
            frag = build_polfunc_table(results, k, include_segments=seg)
            suffix = '_segment' if seg else ''
            path = DRAFTS_DIR / f"{CFG['out_prefix']}_k{k}{suffix}.tex"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(frag + '\n')
            print(f"  Wrote fragment: {path.name}  (label {CFG['tab_label']}_k{k}{suffix})")
            if not seg:
                frags[k] = frag          # preview shows the MAIN tables
    return frags


def compile_polfunc_preview(frags: dict) -> None:
    """Compile ONE standalone preview PDF holding all deposit-type fragments (one per page),
    rendered with the paper's own preamble (utils.tex_preamble.wrap_table)."""
    import subprocess
    try:
        from utils.tex_preamble import wrap_table
    except Exception as exc:
        print(f"  [WARN] utils.tex_preamble not importable ({exc}); fragments saved, preview skipped.")
        return
    body = '\n\n\\clearpage\n\n'.join(frags[k] for k in sorted(frags))
    name = f"{CFG['out_prefix']}_preview"
    tex_path = DRAFTS_DIR / f"{name}.tex"
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(wrap_table(body))
    print(f"  Compiling preview PDF ({name}.pdf)...")
    try:
        # pdflatex -> biber -> pdflatex x2. The biber pass is what resolves the \parencite keys in
        # the Notes against References.bib (the preamble already \addbibresource's it). Without it
        # the preview renders the citations as unresolved markers, which reads like a broken table
        # even though the fragment is fine in V_Main.tex, where biber does run.
        subprocess.run(['pdflatex', '-interaction=nonstopmode', f'{name}.tex'],
                       cwd=str(DRAFTS_DIR), capture_output=True, text=True)
        try:
            subprocess.run(['biber', name], cwd=str(DRAFTS_DIR), capture_output=True, text=True)
        except FileNotFoundError:
            print("  [WARN] biber not found; preview citations will render unresolved "
                  "(fragments are unaffected).")
        for _ in range(2):
            subprocess.run(['pdflatex', '-interaction=nonstopmode', f'{name}.tex'],
                           cwd=str(DRAFTS_DIR), capture_output=True, text=True)
        pdf_path = DRAFTS_DIR / f"{name}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            print(f"  Preview PDF generated: {pdf_path.name}")
        else:
            print(f"  [WARN] preview PDF may not have compiled — check {name}.log")
    except FileNotFoundError:
        print("  [WARN] pdflatex not found; fragment saved, preview skipped.")
    except Exception as exc:
        print(f"  [WARN] preview compilation error: {exc}")


# ==============================================================================
# 8. Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Policy Function Estimation for Deposit Types k=4,5 (BBL Step 1)")
    parser.add_argument(
        '--depvar', choices=['spread', 'rate'], default='spread',
        help='Regressand: "spread" (default; QoQ deposit spread, fed to BBL Step 2) or '
             '"rate" (annualized deposit rate = (1+rate_qoq)^4-1). "rate" writes a SEPARATE '
             'polfunc_rate_* namespace and does NOT overwrite the spread policy.')
    args = parser.parse_args()

    # Regressand selection: switch CFG to the rate namespace if requested.
    if args.depvar == 'rate':
        CFG.update(_CFG_RATE)

    print("=" * 70)
    print("  Policy Function Estimation (BBL Step 1)")
    print(f"  Regressand: {CFG['regressand']}  (--depvar {args.depvar}) -> {CFG['out_prefix']}_*")
    print(f"  Deposit types: {K_ENDOG}")
    print(f"  D-firm demographic options: {D_DEMO_OPTIONS}")
    print("=" * 70)

    # ---- Load data once --------------------------------------------------
    t_start = time.perf_counter()
    df = load_and_prepare_panel()
    df = compute_national_demographics(df)

    # ---- Run estimation --------------------------------------------------
    results = estimate_all_policy_functions(df)

    if not results:
        print("[FATAL] No regressions succeeded. Exiting.")
        sys.exit(1)

    # ---- Print summary ---------------------------------------------------
    print_summary_table(results)

    # ---- Compute fitted values -------------------------------------------
    print("Computing fitted values...")
    df_fitted = compute_fitted_values(df, results)

    # ---- Save outputs -----------------------------------------------------
    # The policy function is SPEC-INVARIANT (it never touches a sleepiness spec), so outputs
    # carry no spec label. The old `_spec_{id}` replication was pipeline cosmetics only.
    print("\n  --- Saving outputs ---")
    save_outputs(results, df_fitted)

    # ---- Paper-style table fragments (one per deposit type) + preview PDF -----
    print("\n  --- Paper table fragments + preview ---")
    frags = write_polfunc_fragments(results)
    compile_polfunc_preview(frags)

    elapsed = time.perf_counter() - t_start
    print(f"\n[DONE] Policy function estimation complete in {elapsed:.1f}s")


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    main()

