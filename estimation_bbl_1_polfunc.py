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

Pipeline position
-----------------
  estimation_1_sleep  →  estimation_1_demand  →  **estimation_bbl_1_polfunc**
                          →  estimation_bbl_2_fwd_sim  →  estimation_bbl_3_solve

Usage
-----
  python estimation_bbl_1_polfunc.py
  python estimation_bbl_1_polfunc.py --spec all

CLI Flags
---------
  --spec {1..12|all}   Sleepiness specification label for output naming
                       (policy function itself is spec-invariant). Default=1.

References
----------
  Bajari, Benkard & Levin (2007, Econometrica)
  Egan, Hortacsu & Matvos (2025, NBER WP)
  Matvos & Seru (2014, AER)
  Ryan (2012, Econometrica)

CLI Options:
------------
usage: estimation_bbl_1_polfunc.py [-h] [--spec SPEC]

Policy Function Estimation for Deposit Types k=4,5 (BBL Step 1)

options:
  -h, --help   show this help message and exit
  --spec SPEC  Sleepiness specification label (1-12 or "all") for output
               naming. The policy function estimation itself is spec-
               invariant.
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
    'lhs_short':   'annualized deposit rate',
    'depvar_note': (r'Dependent variable: annualized deposit rate '
                    r'$r^{\mathrm{dep,ann}}_{jkmt}=(1+r^{\mathrm{dep}}_{jkmt})^{4}-1$, 2016--2024.'),
}


# ==============================================================================
# 1. Data Loading & Preparation
# ==============================================================================
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

    # ----- Quadratic terms ------------------------------------------------
    for col in QUADRATIC_BASE:
        sq_col = f'{col}_sq'
        if col in df.columns:
            df[sq_col] = df[col].astype(float) ** 2

    # ----- Coerce all regressor columns to numeric ------------------------
    # Some columns (e.g. has_ip) may arrive as object dtype from the CSV.
    all_regressor_cols = (
        COMMON_REGRESSORS + DEMOGRAPHICS
        + [f'{c}_sq' for c in QUADRATIC_BASE]
    )
    for col in all_regressor_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

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
def apply_cluster_correction(res, cluster_series: pd.Series):
    """Apply Imbens-Kolesar (2016) / Carter-Schnepel-Steigerwald (2017)
    effective cluster correction to a statsmodels RegressionResults object.

    The effective number of clusters G* = G / (1 + cv²) is used as the
    degrees of freedom for the t-distribution governing inference.
    """
    sizes = cluster_series.value_counts()
    G_nominal = len(sizes)
    cv_Ng = np.std(sizes, ddof=0) / np.mean(sizes) if np.mean(sizes) > 0 else 0
    G_star = max(1.0, G_nominal / (1 + cv_Ng ** 2))

    res.G_nominal = G_nominal
    res.G_star = G_star
    res.df_resid = G_star

    t_dist = stats.t(df=G_star)
    new_pvals = t_dist.sf(np.abs(res.tvalues)) * 2
    res._results.__dict__['pvalues'] = new_pvals

    return res


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

    y = df_w[dep_var]
    X = sm.add_constant(df_w[avail], has_constant='add')

    # Create bank-year cluster variable
    df_w['bank_year'] = df_w['CodConglomeradoPrudencial'].astype(str) + "_" + df_w['year'].astype(str)
    
    # Cluster-robust estimation
    cluster = df_w['bank_year']
    model = sm.OLS(y, X)
    res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster}, use_t=True)
    res = apply_cluster_correction(res, cluster)

    # Extract results
    coef_names = list(res.params.index)
    out = {
        'label': label,
        'res': res,
        'n_obs': int(res.nobs),
        'n_clusters': res.G_nominal,
        'G_star': float(res.G_star),
        'r_squared': float(res.rsquared),
        'r_squared_adj': float(res.rsquared_adj),
        'regressors': coef_names,
        'coefficients': res.params.to_dict(),
        'std_errors': res.bse.to_dict(),
        'pvalues': {k: float(v) for k, v in zip(coef_names, res.pvalues)},
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


def save_outputs(results: dict, df_fitted: pd.DataFrame, spec_id: str) -> None:
    """Persist all estimation outputs to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Pickle with full statsmodels results --------------------------
    pkl_path = OUTPUT_DIR / f"{CFG['out_prefix']}_results_spec_{spec_id}.pkl"
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
    csv_path = OUTPUT_DIR / f"{CFG['out_prefix']}_fitted_spec_{spec_id}.csv"
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

    json_path = OUTPUT_DIR / f"{CFG['out_prefix']}_summary_spec_{spec_id}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved summary JSON: {json_path.name}")

    # --- 4. LaTeX tables per regression -----------------------------------
    for label, res_dict in results.items():
        tex_path = OUTPUT_DIR / f"{CFG['out_prefix']}_{label}_spec_{spec_id}.tex"
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
    'has_ip': 'Has IP Subsidiary',
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
    'indice_basileia_lag': 'Basel Index (bp, $t-1$)',
    'wholesale_ratio_lag': 'Wholesale Ratio ($t-1$)',
    'lci_lca_ratio_lag': 'LCI/LCA Ratio ($t-1$)',
    'risk_free_qoq': 'Risk-Free Rate (QoQ)',
    'log_total_assets_lag_sq': 'Log Total Assets$^2$',
    'equity_ratio_lag_sq': 'Equity Ratio$^2$',
    'risk_free_qoq_sq': 'Risk-Free Rate$^2$',
    'gdp_per_capita': r'GDP \textit{per capita}',
    'fraction_65plus': 'Fraction 65+',
    'fraction_young': 'Fraction Young',
    'pix_users_pf_per1000': 'PIX Users (per 1k)',
    'connections_per100': 'Broadband (per 100)',
    'branches_per1000': 'Branches (per 1k)',
    'cadunico_families_per1000': 'CadUnico Families (per 1k)',
    'gdp_per_capita_natl': r'GDP \textit{per capita} (Natl.)',
    'fraction_65plus_natl': 'Fraction 65+ (Natl.)',
    'fraction_young_natl': 'Fraction Young (Natl.)',
    'pix_users_pf_per1000_natl': 'PIX Users (per 1k, Natl.)',
    'connections_per100_natl': 'Broadband (per 100, Natl.)',
    'branches_per1000_natl': 'Branches (per 1k, Natl.)',
    'cadunico_families_per1000_natl': 'CadUnico Families (per 1k, Natl.)',
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
    return [f'k{k}_{kl}_B', f'k{k}_{kl}_D_optA', f'k{k}_{kl}_D_optB', f'k{k}_{kl}_pooled_optC']


# Column headers, in the order of _polfunc_col_keys.
_COL_HEADERS = ['B-type', 'D (no demo.)', r'D (natl.\ demo.)', r'Pooled ($B{=}D$)']

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


def _polfunc_emit_rows(label, cells, rows):
    """Append a coefficient line + a standard-error line for one regressor across columns."""
    coef_cells, se_cells = [], []
    for c in cells:
        if c is None:
            coef_cells.append('')
            se_cells.append('')
        else:
            cf, se, p = c
            coef_cells.append(f'${cf:.4f}^{{{_stars(p)}}}$')
            se_cells.append(f'$({se:.4f})$')
    rows.append(f'{label} & ' + ' & '.join(coef_cells) + r' \\')
    rows.append(' & ' + ' & '.join(se_cells) + r' \\')


def _polfunc_panel_rows(results: dict, k: int) -> list:
    """LaTeX coefficient rows for one deposit type k across the four columns.
    Non-demographic regressors are the ordered union; demographics are merged
    (local + national into one row, each column showing its own variant)."""
    cols = _polfunc_col_keys(k)
    demo_all = set(_DEMO_BASES) | {d + '_natl' for d in _DEMO_BASES}

    # ordered union of NON-demographic regressors (first appearance across columns)
    order, seen = [], set()
    for ck in cols:
        if ck not in results:
            continue
        for v in results[ck]['regressors']:
            if v in demo_all or v in seen:
                continue
            order.append(v)
            seen.add(v)

    rows = []
    for v in order:
        cells = [_polfunc_cell(results[ck], v) if ck in results else None for ck in cols]
        _polfunc_emit_rows(_clean_var(v), cells, rows)

    # merged demographic rows (skip a base with no coefficient in any column)
    demo_rows = []
    for base in _DEMO_BASES:
        cells = [_polfunc_variant_cell(results[ck], base) if ck in results else None for ck in cols]
        if all(c is None for c in cells):
            continue
        _polfunc_emit_rows(_clean_var(base), cells, demo_rows)
    if demo_rows:
        rows.append(r'\addlinespace[0.3ex]')
        rows.extend(demo_rows)
    return rows


def _polfunc_stat_row(label, values, fmt) -> str:
    cells = [fmt(v) if v is not None else '' for v in values]
    return f'{label} & ' + ' & '.join(cells) + r' \\'


def _polfunc_notes() -> str:
    """Table Notes, with the LHS description and dependent-variable sentence from CFG."""
    return (
        r'\textit{Notes:} OLS of the observed ' + CFG['lhs_short'] + r' on the pricing state, '
        r'estimated separately by firm type and deposit type (BBL Step~1). '
        r'\textbf{B} = brick-and-mortar firms (local MCA demographics); '
        r'\textbf{D (no demo.)} / \textbf{D (natl.\ demo.)} = digital/national firms without '
        r'demographics and with population-weighted national demographics; '
        r'\textbf{Pooled} = B and D under one coefficient vector (D firms given national demographics). '
        r'Demographic rows show each column\textquotesingle s own geography. '
        r'Standard errors clustered at the bank$\times$year level in parentheses, with the '
        r'effective-cluster correction $G^{*}=G/(1+\mathrm{cv}^{2})$ '
        r'(Imbens \& Kolesar 2016; Carter et al.\ 2017). '
        r'*** $p<0.01$, ** $p<0.05$, * $p<0.1$. '
        + CFG['depvar_note']
    )


_K_TITLE = {4: r'Time Deposits / CDB ($k=4$)', 5: r'Prepaid Accounts ($k=5$)'}


def build_polfunc_table(results: dict, k: int) -> str:
    """Paper-style, page-breaking `xltabular` fragment for ONE deposit type k
    (four columns: B / D no-demo / D natl.-demo / Pooled). Bare fragment — `\\input`-able
    into V_Main.tex (the paper preamble already provides xltabular/booktabs/setspace)."""
    colspec = r'>{\raggedright\arraybackslash}p{5.0cm} *{4}{>{\centering\arraybackslash}X}'
    head = ' & ' + ' & '.join(_COL_HEADERS) + r' \\'
    caption = 'Policy Function Estimates: ' + _K_TITLE[k] + r' --- ' + CFG['lhs_title'] + ' (BBL Step~1)'
    label = CFG['tab_label'] + f'_k{k}'
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
        r'\multicolumn{5}{c}{\bfseries Table \thetable\ (continued)} \\',
        r'\toprule',
        head,
        r'\midrule',
        r'\endhead',
        r'\midrule',
        r'\multicolumn{5}{r}{\textit{Continued on next page}} \\',
        r'\endfoot',
        r'\bottomrule',
        r'\multicolumn{5}{@{}p{\dimexpr\textwidth-2\tabcolsep\relax}@{}}{\scriptsize '
        + _polfunc_notes() + r'} \\',
        r'\endlastfoot',
    ]

    L.extend(_polfunc_panel_rows(results, k))
    L.append(r'\midrule')
    r2 = [results[ck]['r_squared'] if ck in results else None for ck in cols]
    obs = [results[ck]['n_obs'] if ck in results else None for ck in cols]
    G = [results[ck]['n_clusters'] if ck in results else None for ck in cols]
    Gs = [results[ck]['G_star'] if ck in results else None for ck in cols]
    L.append(_polfunc_stat_row(r'$R^{2}$', r2, lambda x: f'{x:.4f}'))
    L.append(_polfunc_stat_row('Observations', obs, lambda x: f'{x:,}'))
    L.append(_polfunc_stat_row(r'Clusters ($G$)', G, lambda x: f'{x}'))
    L.append(_polfunc_stat_row(r'Eff.\ clusters ($G^{*}$)', Gs, lambda x: f'{x:.1f}'))

    L.append(r'\end{xltabular}')
    L.append(r'\end{spacing}')
    return '\n'.join(L)


def write_polfunc_fragments(results: dict, spec_id: str) -> dict:
    """Write ONE bare `\\input`-able fragment per deposit type to the paper's drafts folder."""
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    frags = {}
    for k in K_ENDOG:
        frag = build_polfunc_table(results, k)
        path = DRAFTS_DIR / f"{CFG['out_prefix']}_k{k}_spec{spec_id}.tex"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(frag + '\n')
        print(f"  Wrote fragment: {path.name}  (\\input into V_Main.tex; label {CFG['tab_label']}_k{k})")
        frags[k] = frag
    return frags


def compile_polfunc_preview(frags: dict, spec_id: str) -> None:
    """Compile ONE standalone preview PDF holding all deposit-type fragments (one per page),
    rendered with the paper's own preamble (utils.tex_preamble.wrap_table)."""
    import subprocess
    try:
        from utils.tex_preamble import wrap_table
    except Exception as exc:
        print(f"  [WARN] utils.tex_preamble not importable ({exc}); fragments saved, preview skipped.")
        return
    body = '\n\n\\clearpage\n\n'.join(frags[k] for k in sorted(frags))
    name = f"{CFG['out_prefix']}_preview_spec{spec_id}"
    tex_path = DRAFTS_DIR / f"{name}.tex"
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(wrap_table(body))
    print(f"  Compiling preview PDF ({name}.pdf)...")
    try:
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
        '--spec', type=str, default='1',
        help='Sleepiness specification label (1-12 or "all") for output naming. '
             'The policy function estimation itself is spec-invariant.')
    parser.add_argument(
        '--depvar', choices=['spread', 'rate'], default='spread',
        help='Regressand: "spread" (default; QoQ deposit spread, fed to BBL Step 2) or '
             '"rate" (annualized deposit rate = (1+rate_qoq)^4-1). "rate" writes a SEPARATE '
             'polfunc_rate_* namespace and does NOT overwrite the spread policy.')
    args = parser.parse_args()

    # Regressand selection: switch CFG to the rate namespace if requested.
    if args.depvar == 'rate':
        CFG.update(_CFG_RATE)

    # Determine spec IDs
    if args.spec.lower() == 'all':
        spec_ids = [str(i) for i in range(1, 13)]
    else:
        spec_ids = [args.spec]

    print("=" * 70)
    print("  Policy Function Estimation (BBL Step 1)")
    print(f"  Regressand: {CFG['regressand']}  (--depvar {args.depvar}) -> {CFG['out_prefix']}_*")
    print(f"  Deposit types: {K_ENDOG}")
    print(f"  D-firm demographic options: {D_DEMO_OPTIONS}")
    print(f"  Spec labels: {spec_ids}")
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

    # ---- Save outputs (replicated per spec label for pipeline compat) ----
    for spec_id in spec_ids:
        print(f"\n  --- Saving outputs for spec {spec_id} ---")
        save_outputs(results, df_fitted, spec_id)

    # ---- Paper-style table fragments (one per deposit type) + preview PDF -----
    print("\n  --- Paper table fragments + preview ---")
    frags = write_polfunc_fragments(results, spec_ids[0])
    compile_polfunc_preview(frags, spec_ids[0])

    elapsed = time.perf_counter() - t_start
    print(f"\n[DONE] Policy function estimation complete in {elapsed:.1f}s")


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    main()

