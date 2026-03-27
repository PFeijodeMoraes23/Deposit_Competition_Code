"""
estimation_2_sleep_1_cfa.py
==============================
Direct time-series estimation of the sleepiness parameter phi_t for National (NB) digital/wholesale banks
(Appendix/V_Main.tex section on extending local estimates to national aggregates).

Takes the main processed market panel, filters for 'IFDATA' (NB-type firms), and provides 
three econometric strategies to avoid rank-deficiency caused by purely deterministic time-series trends:
  1. Option 1: Direct Macro Time-Series Estimation (Baseline)
  2. Option 2: Interactions with Firm Heterogeneity (lagged Total Assets)
  3. Option 3: Latent Index via Principal Component Analysis (PCA)

Usage
-----
  python estimation_2_sleep_1_cfa.py --option 1
  python estimation_2_sleep_1_cfa.py --run-all

CLI Flags
---------
  --option {1|2|3}    Run specifically one of the options above.
  --run-all           Loop over all three options sequentially for comparative output.
"""
import argparse
from pathlib import Path
import sys
import io
import pickle

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA

# ==============================================================================
# Helper to read results
# ==============================================================================
def _resolve_runtime_paths() -> tuple[Path, Path]:
    _ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
    PANEL_CSV = DATA_DIR / "market_panel.csv"
    OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "SLEEPINESS_NATIONAL"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return PANEL_CSV, OUTPUT_DIR

def demean_variables(df, cols, entity_col):
    # Optimized vectorized demeaning
    df_out = df.copy()
    for col in cols:
        df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
    df_out[cols] = df_out[cols] - df_out.groupby(entity_col)[cols].transform('mean')
    return df_out

def build_data():
    panel_csv, _ = _resolve_runtime_paths()
    df_raw = pd.read_csv(panel_csv, dtype={'year_quarter': str, 'CodIbge': str})
    
    # We only want NB firms (IFDATA source)
    df = df_raw[df_raw['Source'] == 'IFDATA'].copy()
    
    # NB-type firms do not vary by region, so safely drop duplicates
    df = df.drop_duplicates(subset=['CodConglomeradoPrudencial', 'year', 'quarter'])

    if 'deposit_balance' not in df.columns:
        id_vars = [c for c in df.columns if not c.startswith('dep_a') and not c.startswith('spread_a') and not c.startswith('leave_one_out')]
        df = pd.wide_to_long(
            df,
            stubnames=['dep_a', 'spread_a', 'leave_one_out_mean_spread_a'],
            i=id_vars,
            j='deposit_type'
        ).reset_index()
        df = df.rename(columns={
            'dep_a': 'deposit_balance',
            'spread_a': 'spread_qoq',
            'leave_one_out_mean_spread_a': 'leave_one_out_mean_spread'
        })
        # Filter again since melt brings everything back
        df = df[df['Source'] == 'IFDATA'].copy()
    
    if 'deposit_type' in df.columns:
        df = df[~df['deposit_type'].astype(str).str.contains('3')].copy()
    df['entity_id'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['deposit_type'].astype(str)
    
    # Sort to ensure lags are correct
    df = df.sort_values(by=['entity_id', 'year', 'quarter'])

    if 'year_quarter' not in df.columns:
        df['year_quarter'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)

    df['lagged_deposits'] = df.groupby('entity_id')['deposit_balance'].shift(1)
    df['spread_qoq_lag'] = df.groupby('entity_id')['spread_qoq'].shift(1)
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)

    df['nr_lagged_dep'] = (1 + df['risk_free_qoq_lag'] - df['spread_qoq_lag']) * df['lagged_deposits']

    # Scale massive variables
    if 'gdp_per_capita' in df.columns:
        df['gdp_per_capita'] = df['gdp_per_capita'] / 10000.0
    if 'cadunico_families_per1000' in df.columns:
        df['cadunico_families_per1000'] = df['cadunico_families_per1000'] / 100.0
    if 'pix_users_pf_per1000' in df.columns:
        df['pix_users_pf_per1000'] = df['pix_users_pf_per1000'] / 100.0
    if 'connections_per100' in df.columns:
        df['connections_per100'] = df['connections_per100'] / 100.0
        
    df['constant'] = 1.0
    df['post_2020'] = (df['year'] >= 2020).astype(int)
    return df

# ==============================================================================
# Pipeline Models
# ==============================================================================
def run_model_option1(df, state_vars, has_cf=False):
    """
    Option 1: Brute Force Time-Series.
    y_dm ~ nr_lagged_dep * S_t
    """
    X_cols = []
    for sv in state_vars:
        col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
        if sv != 'constant':
            df[col_name] = df[sv] * df['nr_lagged_dep']
        X_cols.append(col_name)

    if has_cf:
        X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])

    df_ss = df.dropna(subset=X_cols + ['deposit_balance']).copy()
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance'].astype(float)
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')[X_cols].astype(float)

    mod = sm.OLS(y_dm, X_dm)
    return mod.fit(cov_type='cluster', cov_kwds={'groups': df_ss['CodConglomeradoPrudencial']}, use_t=True)

def run_model_option2(df, state_vars, has_cf=False):
    """
    Option 2: Firm-Targeting State Interactions.
    y_dm ~ nr_lagged_dep * S_t * X_j
    X_j = log_total_assets_lag
    """
    X_cols = []
    
    # Fill in missing lag assets carefully just in case
    df['log_total_assets_lag'] = df['log_total_assets_lag'].fillna(df['log_total_assets_lag'].median())
    
    for sv in state_vars:
        if sv == 'constant':
            col_name = "nr_lagged_dep"
            X_cols.append(col_name)
        else:
            # We interact S_t with the firm's assets log
            col_name_base = f"interaction_{sv}"
            col_name_cross = f"interaction_{sv}_x_assets"
            
            df[col_name_base] = df[sv] * df['nr_lagged_dep']
            df[col_name_cross] = df[sv] * df['log_total_assets_lag'] * df['nr_lagged_dep']

            X_cols.extend([col_name_base, col_name_cross])

    if has_cf:
        X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])

    df_ss = df.dropna(subset=X_cols + ['deposit_balance']).copy()
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance'].astype(float)
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')[X_cols].astype(float)

    mod = sm.OLS(y_dm, X_dm)
    return mod.fit(cov_type='cluster', cov_kwds={'groups': df_ss['CodConglomeradoPrudencial']}, use_t=True)

def run_model_option3(df, state_vars, has_cf=False):
    """
    Option 3: Dimensionality Reduction (PCA Indexing).
    y_dm ~ nr_lagged_dep * PCA(S_t)
    """
# Remove print statements to avoid terminal clutter
    sv_dynamic = [sv for sv in state_vars if sv != 'constant']
    
    if not sv_dynamic:
        # Falls back to option 1 if only constant
        return run_model_option1(df, state_vars, has_cf)
        
    # Drop rows missing the dynamic vars
    df_pca = df.dropna(subset=sv_dynamic).copy()
    
    pca = PCA(n_components=1)
    # We fit the PCA purely on the time series of the state variables
    # Since these are repeated per banking firm, let's just do it over unique quarters
    ts_df = df_pca[['year_quarter'] + sv_dynamic].drop_duplicates().sort_values('year_quarter')
    ts_vals = ts_df[sv_dynamic].values
    
    # Standardize for PCA
    ts_mean = np.mean(ts_vals, axis=0)
    ts_std = np.std(ts_vals, axis=0)
    ts_std[ts_std == 0] = 1 # prevent div/0
    
    ts_norm = (ts_vals - ts_mean) / ts_std
    ts_pca = pca.fit_transform(ts_norm)
    ts_df['pca_macro_tech_index'] = ts_pca[:, 0]
    
    # Merge back
    df_ss = df.merge(ts_df[['year_quarter', 'pca_macro_tech_index']], on='year_quarter', how='inner')
    
    df_ss['nr_lagged_dep'] = df_ss['nr_lagged_dep']
    df_ss['interaction_pca_index'] = df_ss['pca_macro_tech_index'] * df_ss['nr_lagged_dep']
    
    X_cols = ['nr_lagged_dep', 'interaction_pca_index']
    
    if has_cf:
        X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])
        
    df_ss = df_ss.dropna(subset=X_cols + ['deposit_balance']).copy()
    
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance'].astype(float)
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')[X_cols].astype(float)

    mod = sm.OLS(y_dm, X_dm)
    return mod.fit(cov_type='cluster', cov_kwds={'groups': df_ss['CodConglomeradoPrudencial']}, use_t=True)

def first_stage_cf(df, spec_instruments, exogenous_controls):
    """
    Standard Control Function First Stage from 1_sleep_1_cfa.py
    """
    endog_mask = df['deposit_type'].isin([4, 5])
    first_stage_vars = list(set(spec_instruments + exogenous_controls))
    if 'constant' in first_stage_vars:
        first_stage_vars.remove('constant')

    valid_mask = endog_mask & df[first_stage_vars].notnull().all(axis=1)
    df_fs = df[valid_mask].copy()
    
    if len(df_fs) == 0:
        df['v_hat'] = 0.0
        return df, None

    y = df_fs['spread_qoq']
    X = sm.add_constant(df_fs[first_stage_vars])

    mod = sm.OLS(y, X)
    res = mod.fit(cov_type='HC1')
    
    df['v_hat'] = 0.0
    df.loc[valid_mask, 'v_hat'] = res.resid
    valid_mask_all = df['v_hat'].notnull()
    df.loc[valid_mask_all, 'v_hat_2'] = df.loc[valid_mask_all, 'v_hat'] ** 2
    df.loc[valid_mask_all, 'v_hat_3'] = df.loc[valid_mask_all, 'v_hat'] ** 3
    df.loc[~valid_mask_all, ['v_hat_2', 'v_hat_3']] = 0.0
    return df, res

def main():
    _, output_dir = _resolve_runtime_paths()
    parser = argparse.ArgumentParser(description="Estimate National Phi for NB firms")
    parser.add_argument("--option", type=int, choices=[1, 2, 3], help="1: Brute Force, 2: X_j Interactions, 3: PCA Index")
    parser.add_argument("--run-all", action="store_true", help="Run all options sequentially and print results")
    args = parser.parse_args()
    
    df = build_data()
    print(f"Panel size for IFDATA NB banks: {len(df)} rows")
    
    s_base = ['constant', 'post_2020']
    s_macro = s_base + ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young']
    s_tech = s_macro + ['pix_users_pf_per1000', 'connections_per100', 'branches_per_1000'] # Using a proxy if available
    s_tech = [c for c in s_tech if c in df.columns] # safeguard
    
    state_blocks = {
        'Base': s_base,
        'Macro': s_macro,
        'Tech': s_tech
    }

    iv_specs = {
        'OLS': [],
        'IV_CostShifters': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag'],
        'IV_Wholesale': ['wholesale_ratio_lag', 'indice_basileia_lag', 'leave_one_out_mean_spread'],
        'IV_HausmanFull': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag', 'leave_one_out_mean_spread']
    }
    options_to_run = [args.option] if args.option else ([1, 2, 3] if args.run_all else [])
    
    if not options_to_run:
        print("Please specify --option {1,2,3} or --run-all.")
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = []
    
    def process_task(opt, b_name, s_cols, iv_name, iv_cols):
        exog_cols_act = [c for c in s_cols if c in df.columns and df[c].notnull().sum() > 0]
        iv_cols_act = [c for c in iv_cols if c in df.columns and df[c].notnull().sum() > 0]

        if len(iv_cols_act) > 0:
            df_target, res_fs = first_stage_cf(df.copy(), iv_cols_act, exog_cols_act)
            has_cf = True
        else:
            df_target = df.copy()
            res_fs = None
            has_cf = False

        if opt == 1:
            res_ss = run_model_option1(df_target, exog_cols_act, has_cf)
        elif opt == 2:
            res_ss = run_model_option2(df_target, exog_cols_act, has_cf)
        elif opt == 3:
            res_ss = run_model_option3(df_target, exog_cols_act, has_cf)
        else:
            res_ss = None

        return opt, b_name, iv_name, res_ss, res_fs

    import concurrent.futures
    results_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for opt in options_to_run:
            for b_name, s_cols in state_blocks.items():
                for iv_name, iv_cols in iv_specs.items():
                    tasks.append(executor.submit(process_task, opt, b_name, s_cols, iv_name, iv_cols))

        for future in concurrent.futures.as_completed(tasks):
            opt, b_name, iv_name, res_ss, res_fs = future.result()
            print(f"Processed OPTION {opt} - Specification: {iv_name} x {b_name} ")

            # Store result for pickle export
            spec_name = f"Option_{opt}_{iv_name}_{b_name}"
            results_dict[spec_name] = {
                'option': opt,
                'block': b_name,
                'second_stage': res_ss,
                'first_stage': res_fs
            }

            if res_ss is not None:
                # Same logic to save tables
                safe_name = f"Option_{opt}_{iv_name}_" + b_name.replace(" ", "_").replace("/", "").replace(":", "")
                tex_file = output_dir / f"{safe_name}.tex"
                with open(tex_file, 'w') as f:
                    f.write(res_ss.summary().as_latex())

                if res_fs is not None:
                    tex_file_fs = output_dir / f"{safe_name}_FirstStage.tex"    
                    with open(tex_file_fs, 'w') as f:
                        f.write(res_fs.summary().as_latex())
    
    # Save all results to pickle file
    results_pickle = output_dir / "estimation_results.pkl"
    with open(results_pickle, 'wb') as f:
        pickle.dump(results_dict, f)
        
    print(f"\nEstimation outputs successfully saved in: {output_dir}")        
    print(f" - Estimation results (pickle): {results_pickle}")

if __name__ == "__main__":
    main()
