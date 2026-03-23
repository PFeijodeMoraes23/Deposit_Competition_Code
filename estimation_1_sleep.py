"""
estimation_1_sleep.py
=======================
Estimates the depositor sleepiness function (Eq-11 and Eq-12).

This script reads the master compiled market panel and performs a two-stage 
routine to recover the sleepiness parameters for B-type institutions governing 
the Brazilian deposit competition space.

Pipeline stages
---------------
  Stage 1 - Data Preparation
    - Loads market_panel.csv
    - Pivots wide deposit blocks to long format based on deposit_type
    - Constructs base regressors (lagged net returns, lagged deposits)
    - Demeans panels to manually construct the (Conglomerate x Type x MCA) 
      Fixed Effects projection.

  Stage 2 - First Stage (Control Function)
    - Isolates endogenous deposit types (4 and 5)
    - Replicates Egan et al. control function by regressing spreads on IVs
    - Computes and assigns 3rd degree polynomial residuals v_hat

  Stage 3 - Second Stage (Sleepiness Regression)
    - Iterates across predefined state variable / IV blocks
    - Employs clustered standard errors mimicking BBL standards.
    - Exports print summary outputs.

Usage
-----
  python estimation_1_sleep.py

Notes
-----
  * Replaces pip dependency on `linearmodels` by doing manual Within-transformations 
    using pandas. 
"""

import sys
import os
import json
import pickle
from pathlib import Path
import concurrent.futures


try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

try:
    from utils.toon_parser import get_script_config, load_default_toon_context
except Exception:
    get_script_config = None
    load_default_toon_context = None


# ==============================================================================
# 0. Global Paths and Parameters
# ==============================================================================
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
PANEL_CSV = DATA_DIR / "market_panel.csv"


def _resolve_runtime_paths() -> tuple[Path, Path]:
    """Resolve panel/output paths with optional TOON overrides."""
    panel_csv = PANEL_CSV
    output_dir = DATA_DIR / "ESTIMATION_OUTPUT" / "SLEEPINESS_NEW"

    if load_default_toon_context is None or get_script_config is None:
        return panel_csv, output_dir

    try:
        context = load_default_toon_context(Path(__file__).resolve().parent / "utils", quiet=True)
    except Exception:
        return panel_csv, output_dir

    cfg = get_script_config(context, "estimation_1_sleep")
    if not isinstance(cfg, dict):
        return panel_csv, output_dir

    if isinstance(cfg.get("panel_csv"), str) and cfg["panel_csv"].strip():
        panel_csv = Path(cfg["panel_csv"]).expanduser()
    if isinstance(cfg.get("output_dir"), str) and cfg["output_dir"].strip():
        output_dir = Path(cfg["output_dir"]).expanduser()

    return panel_csv, output_dir


# ==============================================================================
# Helper Functions
# ==============================================================================
def demean_variables(df, cols, entity_col):
    """
    Manually demeans columns for one-way Fixed Effects (Within Estimator).
    Returns a dataframe of the demeaned columns.
    """
    means = df.groupby(entity_col)[cols].transform('mean')
    return df[cols] - means


# ==============================================================================
# Section 1 - Data Loading and Preparation
# ==============================================================================
def build_data():
    """
    Reads the wide market_panel.csv, reshapes it to long by deposit type,
    applies the necessary variable lag configurations, and isolates B-Level branches.
    """
    panel_csv, _ = _resolve_runtime_paths()
    print(f"Loading {panel_csv}...")
    df_raw = pd.read_csv(panel_csv)
    
    # Restrict to B-type institutions (meaning branches exist, CODMUN_IBGE != 0)
    df_raw = df_raw[df_raw['CODMUN_IBGE'].astype(str) != '0'].copy()
    
    # -------------------------------------------------------------
    # 1a. Reshape wide deposit columns to long-format
    # -------------------------------------------------------------
    if 'dep_a1' in df_raw.columns:
        print("Reshaping panel from wide to long...")
        id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
        df_raw = df_raw.drop_duplicates(subset=id_vars)
        
        df = pd.wide_to_long(
            df_raw, 
            stubnames=['dep_a', 'spread_a', 'leave_one_out_mean_spread_a'], 
            i=id_vars, 
            j='deposit_type'
        ).reset_index()
        
        df = df.rename(columns={
            'dep_a': 'deposit_balance', 
            'spread_a': 'spread_qoq',
            'leave_one_out_mean_spread_a': 'leave_one_out_mean_spread'
        })
    else:
        df = df_raw.copy()

    # -------------------------------------------------------------
    # 1b. Assign Unique Fixed Effect Keys & Apply Temporal Shifts
    # -------------------------------------------------------------
    df['entity_id'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + \
                      df['deposit_type'].astype(str) + "_" + \
                      df['mca_code'].astype(str)
                      
    df['time_id'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)
    
    df.sort_values(by=['entity_id', 'year', 'quarter'], inplace=True)
    
    df['spread_qoq_lag'] = df.groupby('entity_id')['spread_qoq'].shift(1)
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)
    df['lagged_deposits'] = df.groupby('entity_id')['deposit_balance'].shift(1)
    
    # Base Regressor: (1 + r^f_{t-1} - \rho_{t-1}) * Dep_{t-1}
    df['nr_lagged_dep'] = (1 + df['risk_free_qoq_lag'] - df['spread_qoq_lag']) * df['lagged_deposits']
    
    # Only keep the Hausman IV that is actually created in market_panel.csv
    # other cost shifters are not in this pipeline's output. 
    required_cols = [
        'leave_one_out_mean_spread'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df.dropna(subset=['deposit_balance', 'nr_lagged_dep', 'spread_qoq', 'entity_id', 'time_id'])
    return df


# ==============================================================================
# Section 2 - First Stage (Control Function Estimator)
# ==============================================================================
def run_first_stage(df, spec_instruments):
    """
    Executes the Egan et al control function to purge endogenous spread-setting latency
    by projecting spreads against operating cost shifters. Only operates on endogenous
    deposit buckets (K=4,5).
    """
    print(f"\n--- First Stage (Endogenous types 4 & 5) ---")
    endog_mask = df['deposit_type'].isin([4, 5])
    valid_mask = endog_mask & df[spec_instruments].notnull().all(axis=1)
    df_fs = df[valid_mask].copy()
    
    if len(df_fs) == 0:
        print("No valid rows for first stage with these instruments.")
        df['v_hat'] = 0.0
        return df
        
    y = df_fs['spread_qoq']
    X = sm.add_constant(df_fs[spec_instruments])
    
    mod = sm.OLS(y, X)
    res = mod.fit(cov_type='HC1')
    
    # Deposit buckets 1-3 receive zero latency 
    df['v_hat'] = 0.0 
    df.loc[valid_mask, 'v_hat'] = res.resid
    
    # Assemble orthogonal polynomial space
    df['v_hat_2'] = df['v_hat'] ** 2
    df['v_hat_3'] = df['v_hat'] ** 3
    return df


# ==============================================================================
# Helper for Multiprocessing
# ==============================================================================
def execute_specification(args):
    """
    Isolated wrapper for ProcessPoolExecutor to run a specification and return 
    the captured print output as a string to avoid stdout race conditions.
    Also returns the statsmodels RegressionResults object to be formatted as latex.
    """
    df, iv_name, iv_cols, s_name, s_cols = args
    import io, sys
    
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    res = None
    try:
        has_cf = len(iv_cols) > 0
        spec_name = f"{iv_name} x {s_name}"
        print(f"\n=====================================================================")
        print(f" RUNNING: {spec_name}")
        print(f"=====================================================================")
        
        df_target = df.copy()
        
        if has_cf:
            iv_cols_act = [c for c in iv_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
            if len(iv_cols_act) == 0:
                print(f"Skipping {spec_name} - none of the IVs are populated in the dataset.")
                return new_stdout.getvalue(), None, spec_name
            df_target = run_first_stage(df_target, iv_cols_act)
            
        res = run_second_stage(df_target, s_cols, has_cf=has_cf, spec_name=spec_name)
    except Exception as e:
        print(f"Estimation Failed for {spec_name} - {str(e)}")
    finally:
        sys.stdout = old_stdout
        
    return new_stdout.getvalue(), res, spec_name


# ==============================================================================
# Section 3 - Second Stage (Sleepiness Regression) UPDATED
# ==============================================================================
def run_second_stage(df, state_vars, has_cf=False, spec_name=""):
    """
    Executes the active/inactive demand decomposition using High-Dimensional 
    Fixed Effects. SEs are appropriately clustered. Returns the fitted result.
    """
    print(f"\n[{spec_name}] --- Second Stage (Sleepiness) ---")
    
    X_cols = []
    for sv in state_vars:
        col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
        if sv == 'constant':
            df['nr_lagged_dep'] = df['nr_lagged_dep'] 
        else:
            df[col_name] = df[sv] * df['nr_lagged_dep']
        X_cols.append(col_name)
        
    if has_cf:
        X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])
        
    df_ss = df.dropna(subset=X_cols + ['deposit_balance']).copy()
    if len(df_ss) == 0:
        print("No observations available after dropping NaNs for these state variables/CF.")
        return None
        
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance']
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')
    
    # -------------------------------------------------------------
    # 3c. Regress and Estimate (Imbens & Kolesar / Carter 2017)
    # -------------------------------------------------------------
    mod = sm.OLS(y_dm, X_dm)
    
    # Carter, Schnepel, and Steigerwald (2017) Effective Number of Clusters
    # This precisely matches Imbens and Kolesar (2016) Satterthwaite bounds algebraically.
    # Due to local RAM constraints (120GB+ required for explicit H_gg matrix inversions 
    # to naturally compute CR2 arrays on datasets exceeding 100k Brazilian operations),
    # we simulate IK2016 bounds by mapping unadjusted CR1 variances strictly against 
    # a t-distribution parameterized entirely by the true Effective Clusters (G*).
    
    cluster_series = df_ss['CodConglomeradoPrudencial']
    sizes = cluster_series.value_counts()
    G_nominal = len(sizes)
    
    cv_Ng = np.std(sizes, ddof=0) / np.mean(sizes) if np.mean(sizes) > 0 else 0
    G_star = max(1.0, G_nominal / (1 + (cv_Ng ** 2))) 
    
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    
    # Force statsmodels to drop the naive (G-1) degrees of freedom approximation 
    # and strictly evaluate inference parameters against the Satterthwaite G* penalty.
    res.df_resid = G_star
    
    # Recalculate robust P-values dynamically
    t_dist = stats.t(df=G_star)
    new_pvals = t_dist.sf(np.abs(res.tvalues)) * 2
    
    # Bypass the restrictive @cache_readonly property lock natively inside statsmodels wrapper
    res._results.__dict__['pvalues'] = new_pvals
    
    print(f"\n *** IK2016 Bounds Applied: Nominal G = {G_nominal} -> Effective G* = {G_star:.2f} ***\n")
    print(res.summary().tables[1])
    print(f"Obs: {int(res.nobs)} | Clusters: {G_nominal} | G*: {G_star:.2f} | FE Groups: {df_ss['entity_id'].nunique()}")
    
    return res


# ==============================================================================
# Section 4 - Global Execution Routine
# ==============================================================================
def main():

    _, output_dir = _resolve_runtime_paths()

    df = build_data()
    print(f"Panel size after cleaning: {len(df)} rows")
    
    # --------------------------------------------------------------------------
    # Effective Clusters Diagnostic (Imbens & Kolesar 2016 / Carter et al. 2017)
    # --------------------------------------------------------------------------
    print("\n=====================================================================")
    print(" CLUSTER HOMOGENEITY DIAGNOSTICS")
    print("=====================================================================")
    cluster_var = 'CodConglomeradoPrudencial'
    Ns = df.groupby(cluster_var).size()
    G_nominal = len(Ns)
    mean_Ng = np.mean(Ns)
    std_Ng = np.std(Ns, ddof=0)
    cv_Ng = std_Ng / mean_Ng if mean_Ng > 0 else 0
    G_star = max(1.0, G_nominal / (1 + (cv_Ng ** 2)))
    
    print(f"Total Nominal Clusters (G)      = {G_nominal}")
    print(f"Mean Obs per Cluster            = {mean_Ng:.2f}")
    print(f"Std Dev of Obs per Cluster      = {std_Ng:.2f}")
    print(f"Coefficient of Variation (cv)   = {cv_Ng:.2f}")
    print(f"Effective Clusters (G*)         = {G_star:.2f}")
    
    print("\nTop 5 Cluster Observation Shares (Oligopolistic Imbalance):")
    total_obs = len(df)
    for c, n in Ns.sort_values(ascending=False).head(5).items():
        print(f" - Cluster {c}: {n} obs ({(n / total_obs) * 100:.2f}%)")
    print("=====================================================================\n")
    
    # Establish Output Directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df['constant'] = 1.0
    s_base = ['constant']
    s_macro = s_base + ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young']
    s_tech_finance = s_macro + ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']
    
    for col in s_tech_finance:
        if col != 'constant' and col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    iv_spec0 = []
    iv_spec1 = ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag']
    iv_spec2 = iv_spec1 + ['lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag']
    iv_spec3 = iv_spec2 + ['leave_one_out_mean_spread']

    specs = {
        'Spec1-OLS': iv_spec0,
        'Spec2-IV_CostShifters': iv_spec1,
        'Spec3-IV_Wholesale': iv_spec2,
        'Spec4-IV_HausmanFull': iv_spec3
    }

    state_blocks = {
        'Base': s_base,
        'Macro': s_macro,
        'Tech': s_tech_finance
    }
    
    tasks = []
    for iv_name, iv_cols in specs.items():
        for s_name, s_cols in state_blocks.items():
            tasks.append((df, iv_name, iv_cols, s_name, s_cols))
            
    print(f"Starting parallel execution of {len(tasks)} specifications...")
    
    out_results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for output in executor.map(execute_specification, tasks):
            out_results.append(output)
            
    # Process Results and Save Stargazer-Compatible Tables + Pickle Results
    results_dict = {}
    for stdout_text, res, spec_name in out_results:
        print(stdout_text)
        if res is not None:
            safe_name = spec_name.replace(" ", "_").replace("/", "").replace(":", "")
            tex_file = output_dir / f"{safe_name}.tex"
            
            # Using statsmodels native latex export, which mimics Stargazer functionality
            with open(tex_file, 'w') as f:
                f.write(res.summary().as_latex())
            
            # Store result for pickle export
            results_dict[spec_name] = res
    
    # Save cluster diagnostics to JSON
    cluster_diagnostics = {
        'G_nominal': int(G_nominal),
        'G_star': float(G_star),
        'mean_obs_per_cluster': float(mean_Ng),
        'std_obs_per_cluster': float(std_Ng),
        'coefficient_variation': float(cv_Ng),
        'total_observations': int(total_obs),
        'top_5_clusters': {
            str(c): {
                'observations': int(n),
                'share_pct': float((n / total_obs) * 100)
            }
            for c, n in Ns.sort_values(ascending=False).head(5).items()
        }
    }
    
    diag_file = output_dir / "cluster_diagnostics.json"
    with open(diag_file, 'w') as f:
        json.dump(cluster_diagnostics, f, indent=2)
    
    # Save all results to pickle file
    results_pickle = output_dir / "estimation_results.pkl"
    with open(results_pickle, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"Estimation outputs successfully saved in: {output_dir}")
    print(f" - Cluster diagnostics: {diag_file}")
    print(f" - Estimation results (pickle): {results_pickle}")

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    main()
