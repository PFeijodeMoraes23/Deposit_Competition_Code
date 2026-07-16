"""
estimation_1_cost_1_polfunc.py
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
  estimation_1_sleep  →  estimation_1_demand  →  **estimation_1_cost_1_polfunc**
                                                   →  estimation_1_cost_2_fwd (future)

Usage
-----
  python estimation_1_cost_1_polfunc.py
  python estimation_1_cost_1_polfunc.py --spec all

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
usage: estimation_1_cost_1_polfunc.py [-h] [--spec SPEC]

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

# Estimation window — keep in sync with estimation_demand_link_common.MIN_YEAR/MAX_YEAR and
# estimation_2_sleep.SLEEP_MIN_YEAR/MAX_YEAR.  The BBL Step-1 policy function is fit on bank
# characteristics, which only exist on the prudential-conglomerate basis from 2016; 2025 carries the
# fin_income/COSIF/renumbering breaks.  See counterfactuals_plan.md §0A.
POLFUNC_MIN_YEAR = int(os.environ.get('DEMAND_MIN_YEAR', 2016))
POLFUNC_MAX_YEAR = int(os.environ.get('DEMAND_MAX_YEAR', 2024))
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
COMMON_REGRESSORS = BANK_CHARS + COST_SHIFTERS + CAPITAL_WHOLESALE + MACRO

# D-firm demographic options
D_DEMO_OPTIONS = ['A', 'B', 'C']
D_DEMO_LABELS = {
    'A': 'No demographics',
    'B': 'National (pop-weighted) demographics',
    'C': 'Pooled with B-type coefficients',
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
        stubnames=['dep_a', 'spread_a'],
        i=id_vars,
        j='deposit_type'
    ).reset_index()

    df = df.rename(columns={'dep_a': 'deposit_balance', 'spread_a': 'spread_qoq'})

    # Restrict to endogenous deposit types
    df = df[df['deposit_type'].isin(K_ENDOG)].copy()
    print(f"  After restricting to k in {K_ENDOG}: {len(df):,} rows")

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
    n_before = len(df)
    df = df.dropna(subset=['spread_qoq'])
    print(f"  Dropped {n_before - len(df):,} rows with missing spread_qoq")

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
        tasks.append((df_B, 'spread_qoq', regs_B, label_B))

        # ---- D-type regressions (options A and B) ------------------------
        for opt in ['A', 'B']:
            regs_D = _build_regressor_list('D', opt)
            label_D = f"k{k}_{k_label}_D_opt{opt}"
            tasks.append((df_D, 'spread_qoq', regs_D, label_D))

        # ---- Pooled regression (option C) --------------------------------
        df_pooled = _prepare_pooled_data(df, k)
        regs_pooled = _build_regressor_list('pooled', 'C')
        label_pooled = f"k{k}_{k_label}_pooled_optC"
        tasks.append((df_pooled, 'spread_qoq', regs_pooled, label_pooled))

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
                 'deposit_type', 'is_B', 'spread_qoq', 'entity_id', 'time_id']].copy()

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
        # the consumer (cost_2_fwd_sim::equilibrium_spreads) already falls back to the observed spread
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
    pkl_path = OUTPUT_DIR / f"polfunc_results_spec_{spec_id}.pkl"
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
    csv_path = OUTPUT_DIR / f"polfunc_fitted_spec_{spec_id}.csv"
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

    json_path = OUTPUT_DIR / f"polfunc_summary_spec_{spec_id}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved summary JSON: {json_path.name}")

    # --- 4. LaTeX tables per regression -----------------------------------
    for label, res_dict in results.items():
        tex_path = OUTPUT_DIR / f"polfunc_{label}_spec_{spec_id}.tex"
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


def build_polfunc_table(results: dict, k: int) -> str:
    """Build a single booktabs LaTeX table for deposit type k.

    Columns: B-type | D-type Opt A | D-type Opt B | Pooled Opt C
    """
    k_label = K_LABELS[k]
    col_keys = [
        f'k{k}_{k_label}_B',
        f'k{k}_{k_label}_D_optA',
        f'k{k}_{k_label}_D_optB',
        f'k{k}_{k_label}_pooled_optC',
    ]
    col_headers = [
        'B-type',
        'D-type (no demo)',
        'D-type (natl. demo)',
        'Pooled (B=D)',
    ]

    # Collect the union of variable names across all columns in this table
    all_vars = []
    seen = set()
    for ck in col_keys:
        if ck not in results:
            continue
        for v in results[ck]['regressors']:
            if v not in seen:
                all_vars.append(v)
                seen.add(v)

    n_cols = len(col_headers)

    lines = [
        '\\begin{table}[htbp]\\centering',
        f'\\caption{{Policy Function: Deposit Type $k={k}$ ({k_label.replace("_", " ")})}}',
        f'\\label{{tab:polfunc_k{k}}}',
        '\\small',
        '\\begin{tabular}{l' + 'c' * n_cols + '}\\toprule',
        ' & ' + ' & '.join(col_headers) + ' \\\\ \\midrule',
    ]

    for v in all_vars:
        coef_cells = []
        se_cells = []
        for ck in col_keys:
            if ck not in results or v not in results[ck]['coefficients']:
                coef_cells.append('')
                se_cells.append('')
            else:
                c = results[ck]['coefficients'][v]
                se = results[ck]['std_errors'][v]
                p = results[ck]['pvalues'][v]
                coef_cells.append(f'${c:.4f}^{{{_stars(p)}}}$')
                se_cells.append(f'$({se:.4f})$')
        lines.append(f'{_clean_var(v)} & ' + ' & '.join(coef_cells) + ' \\\\')
        lines.append(' & ' + ' & '.join(se_cells) + ' \\\\')

    # Footer statistics
    obs_cells, rsq_cells, g_cells, gstar_cells = [], [], [], []
    for ck in col_keys:
        if ck not in results:
            obs_cells.append('')
            rsq_cells.append('')
            g_cells.append('')
            gstar_cells.append('')
        else:
            r = results[ck]
            obs_cells.append(f'{r["n_obs"]:,}')
            rsq_cells.append(f'{r["r_squared"]:.4f}')
            g_cells.append(str(r['n_clusters']))
            gstar_cells.append(f'{r["G_star"]:.2f}')

    lines.extend([
        '\\midrule',
        'Observations & ' + ' & '.join(obs_cells) + ' \\\\',
        '$R^2$ & ' + ' & '.join(rsq_cells) + ' \\\\',
        'Clusters ($G$) & ' + ' & '.join(g_cells) + ' \\\\',
        'Effective Clusters ($G^*$) & ' + ' & '.join(gstar_cells) + ' \\\\',
        '\\bottomrule',
        '\\multicolumn{' + str(n_cols + 1) + '}{p{0.95\\textwidth}}'
        '{\\scriptsize \\textit{Notes:} Standard errors clustered at the '
        'conglomerate level in parentheses, with effective-cluster correction '
        'following Imbens \\& Kolesar (2016) and Carter et al.\\ (2017). '
        'Significance levels: *** $p<0.01$, ** $p<0.05$, * $p<0.1$. '
        'Dependent variable: deposit spread $\\rho_{jkmt}$ '
        '(risk-free rate minus deposit rate, QoQ).} \\\\',
        '\\end{tabular}',
        '\\end{table}',
    ])

    return '\n'.join(lines)


def export_latex_pdf(results: dict, spec_id: str) -> None:
    """Build a standalone LaTeX document with policy function tables and
    compile it to PDF via pdflatex (two passes for cross-references).
    """
    import subprocess

    print("\n  --- LaTeX Export & PDF Compilation ---")

    # Build tables for each deposit type
    table_k4 = build_polfunc_table(results, 4)
    table_k5 = build_polfunc_table(results, 5)

    preamble = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{letterpaper, margin=0.8in}
\usepackage{caption}
\usepackage{float}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue}

\begin{document}

\title{Policy Function Estimation (BBL Step 1)\\[0.5em]
\large Deposit Types $k=4$ (Time/CDB) and $k=5$ (Prepaid)}
\author{Autogenerated Report}
\date{\today}
\maketitle

\section*{Overview}

This document reports the policy function estimates for endogenous deposit
types $k=4$ and $k=5$, following the BBL two-step procedure
(Bajari, Benkard \& Levin, 2007) as applied in Egan, Hortacsu \& Matvos
(2025, NBER), Ryan (2012), and Matvos \& Seru (2014).

The policy function maps state variables to equilibrium deposit spreads
$\rho_{jkmt} = r_t^f - r_{jkmt}^{\mathrm{dep}}$. For each deposit type,
we estimate four specifications:

\begin{enumerate}
    \item \textbf{B-type}: Brick-and-mortar firms with local MCA demographics.
    \item \textbf{D-type (no demo)}: Digital firms, excluding demographics.
    \item \textbf{D-type (natl.\ demo)}: Digital firms, using population-weighted
          national averages of MCA demographics.
    \item \textbf{Pooled}: B and D firms pooled under shared coefficients
          (D-firms receive national demographics).
\end{enumerate}

Standard errors are clustered at the conglomerate level with the
Imbens--Kolesar (2016) / Carter et al.\ (2017) effective-cluster correction.

"""

    tex_doc = (
        preamble
        + '\\section*{Deposit Type $k=4$ (Time/CDB)}\n\n'
        + table_k4
        + '\n\n\\newpage\n'
        + '\\section*{Deposit Type $k=5$ (Prepaid)}\n\n'
        + table_k5
        + '\n\n\\end{document}\n'
    )

    # Write .tex file
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    tex_name = f"PolicyFunction_Export_spec{spec_id}"
    tex_path = DRAFTS_DIR / f"{tex_name}.tex"
    print(f"  Writing LaTeX to {tex_path.name}...")
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_doc)

    # Compile via pdflatex (two passes)
    print("  Compiling PDF...")
    try:
        for _ in range(2):
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', f'{tex_name}.tex'],
                cwd=str(DRAFTS_DIR), capture_output=True, text=True,
            )
        pdf_path = DRAFTS_DIR / f"{tex_name}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            print(f"  PDF successfully generated: {pdf_path.name}")
        else:
            print("  [WARN] PDF may not have compiled correctly. Check the .log file.")
    except FileNotFoundError:
        print("  [WARN] pdflatex not found. LaTeX .tex file saved but PDF not compiled.")
    except Exception as exc:
        print(f"  [WARN] PDF compilation error: {exc}")


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
    args = parser.parse_args()

    # Determine spec IDs
    if args.spec.lower() == 'all':
        spec_ids = [str(i) for i in range(1, 13)]
    else:
        spec_ids = [args.spec]

    print("=" * 70)
    print("  Policy Function Estimation (BBL Step 1)")
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

    # ---- LaTeX export & PDF compilation ----------------------------------
    export_latex_pdf(results, spec_ids[0])

    elapsed = time.perf_counter() - t_start
    print(f"\n[DONE] Policy function estimation complete in {elapsed:.1f}s")


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    main()

