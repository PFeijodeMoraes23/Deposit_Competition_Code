"""
estimation_1_sleep.py
================================================================================
Estimates the depositor sleepiness function (Eq-11 and Eq-12) for B-type institutions,
then immediately computes the local phi_mt and national phi_t aggregates.

Pipeline stages
---------------
  Phase 1 - Estimation (CFA)
    - Loads market_panel.csv, pivots to long format.
    - Runs Control Function (First Stage) and Sleepiness Regression (Second Stage).
    - Applies Imbens & Kolesar (2016) Satterthwaite bounds for Effective Clusters.
    - Exports print summary outputs (TeX tables, cluster_diagnostics.json, estimation_results.pkl).

  Phase 2 - Phi Construction & Export
    - Uses regression estimates to compute local (phi_mt) and national (phi_t).
    - Exports extended market_panel_phis.csv and national_phi_t.csv.

Usage
-----
  python estimation_1_sleep.py

CLI Flags
---------
  --skip-estimation   Skip the regression phase, only compute Phi outputs using existing pickle.
  --skip-phi          Skip the Phi construction phase, only generate regressions.
"""

import sys
import os
import json
import pickle
import argparse
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
import warnings

warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")

from scipy import stats

try:
    from utils.toon_parser import get_script_config, load_default_toon_context
except Exception:
    get_script_config = None
    load_default_toon_context = None

def _resolve_runtime_paths() -> tuple[Path, Path]:
    _ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
    PANEL_CSV = DATA_DIR / "market_panel.csv"
    OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "SLEEPINESS" / "SLEEPINESS_NEW"

    if load_default_toon_context is None or get_script_config is None:
        return PANEL_CSV, OUTPUT_DIR

    try:
        context = load_default_toon_context(Path(__file__).resolve().parent / "utils", quiet=True)
        cfg = get_script_config(context, "estimation_1_sleep")
        if isinstance(cfg, dict):
            if isinstance(cfg.get("panel_csv"), str) and cfg["panel_csv"].strip():
                PANEL_CSV = Path(cfg["panel_csv"]).expanduser()
            if isinstance(cfg.get("output_dir"), str) and cfg["output_dir"].strip():
                OUTPUT_DIR = Path(cfg["output_dir"]).expanduser()
    except Exception:
        pass
    return PANEL_CSV, OUTPUT_DIR

def apply_imbalanced_cluster_correction(res, cluster_series):
    sizes = cluster_series.value_counts()
    G_nominal = len(sizes)
    cv_Ng = np.std(sizes, ddof=0) / np.mean(sizes) if np.mean(sizes) > 0 else 0
    G_star = max(1.0, G_nominal / (1 + (cv_Ng ** 2)))
    
    res.G_nominal = G_nominal
    res.G_star = G_star

    res.df_resid = G_star
    t_dist = stats.t(df=G_star)
    new_pvals = t_dist.sf(np.abs(res.tvalues)) * 2
    res._results.__dict__['pvalues'] = new_pvals
    
    return res

def demean_variables(df, cols, entity_col):
    means = df.groupby(entity_col)[cols].transform('mean')
    return df[cols] - means


# ==============================================================================
# PHASE 1: ESTIMATION (First Stage & Second Stage)
# ==============================================================================

def build_estimation_data():
    panel_csv, _ = _resolve_runtime_paths()
    print(f"Loading {panel_csv}...")
    df_raw = pd.read_csv(panel_csv)
    df_raw = df_raw[df_raw['CODMUN_IBGE'].astype(str) != '0'].copy()
    
    if 'dep_a1' in df_raw.columns:
        print("Reshaping panel from wide to long...")
        id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
        df_raw = df_raw.drop_duplicates(subset=id_vars)
        df = pd.wide_to_long(df_raw, stubnames=['dep_a', 'spread_a', 'leave_one_out_mean_spread_a'], i=id_vars, j='deposit_type').reset_index()
        df = df.rename(columns={'dep_a': 'deposit_balance', 'spread_a': 'spread_qoq', 'leave_one_out_mean_spread_a': 'leave_one_out_mean_spread'})
    else:
        df = df_raw.copy()

    if 'deposit_type' in df.columns: df = df[df['deposit_type'] != 3].copy()

    df['entity_id'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['deposit_type'].astype(str) + "_" + df['mca_code'].astype(str)
    df['time_id'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)
    
    df.sort_values(by=['entity_id', 'year', 'quarter'], inplace=True)
    df['spread_qoq_lag'] = df.groupby('entity_id')['spread_qoq'].shift(1)
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)
    df['lagged_deposits'] = df.groupby('entity_id')['deposit_balance'].shift(1)
    df['nr_lagged_dep'] = (1 + df['risk_free_qoq_lag'] - df['spread_qoq_lag']) * df['lagged_deposits']
    
    for col in ['leave_one_out_mean_spread']:
        if col not in df.columns: df[col] = np.nan

    df['post_2020'] = (df['year'] >= 2020).astype(int)
    df['bank_year'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['year'].astype(str)
    df = df.dropna(subset=['deposit_balance', 'nr_lagged_dep', 'spread_qoq', 'entity_id', 'time_id'])
    return df

def run_first_stage(df, spec_instruments, exogenous_controls):
    endog_mask = df['deposit_type'].isin([4, 5])
    first_stage_vars = list(set(spec_instruments + exogenous_controls))
    if 'constant' in first_stage_vars: first_stage_vars.remove('constant')

    valid_mask = endog_mask & df[first_stage_vars].notnull().all(axis=1)
    df_fs = df[valid_mask].copy()
    
    if len(df_fs) == 0:
        df['v_hat'] = 0.0
        return df, None

    y = df_fs['spread_qoq']
    X = sm.add_constant(df_fs[first_stage_vars])
    mod = sm.OLS(y, X)
    cluster_series = df_fs['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    res = apply_imbalanced_cluster_correction(res, cluster_series)

    df['v_hat'] = 0.0
    df.loc[valid_mask, 'v_hat'] = res.resid
    df['v_hat_2'] = df['v_hat'] ** 2
    df['v_hat_3'] = df['v_hat'] ** 3
    return df, res

def run_second_stage(df, state_vars, has_cf=False, spec_name=""):
    X_cols = []
    for sv in state_vars:
        col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
        if sv == 'constant': df['nr_lagged_dep'] = df['nr_lagged_dep'] 
        else: df[col_name] = df[sv] * df['nr_lagged_dep']
        X_cols.append(col_name)
        
    if has_cf: X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])
        
    df_ss = df.dropna(subset=X_cols + ['deposit_balance']).copy()
    if len(df_ss) == 0: return None
        
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance']
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')
    
    mod = sm.OLS(y_dm, X_dm)
    cluster_series = df_ss['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    res = apply_imbalanced_cluster_correction(res, cluster_series)
    return res

def execute_specification(args):
    df, iv_name, iv_cols, s_name, s_cols, spec_number = args
    import io, sys
    
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    res = None
    res_fs = None
    try:
        has_cf = len(iv_cols) > 0
        spec_name = f"{iv_name} x {s_name}"
        spec_label = f"({spec_number}) {spec_name}"
        print("\n=====================================================================")
        print(f" RUNNING: {spec_label}")
        print("=====================================================================")

        df_target = df.copy()

        if has_cf:
            iv_cols_act = [c for c in iv_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
            exog_cols_act = [c for c in s_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
            if not iv_cols_act:
                print(f"Skipping {spec_label} - none of the IVs are populated.")
                return new_stdout.getvalue(), None, spec_name, None, spec_number
            df_target, res_fs = run_first_stage(df_target, iv_cols_act, exog_cols_act)

        res = run_second_stage(df_target, s_cols, has_cf=has_cf, spec_name=spec_name)
        if res is not None:
             print(f"\n *** IK2016 Bounds Applied: Nominal G = {res.G_nominal} -> Effective G* = {res.G_star:.2f} ***\n")
             print(res.summary().tables[1])
             print(f"Obs: {int(res.nobs)} | Clusters: {res.G_nominal} | G*: {res.G_star:.2f}")
    except Exception as e:
        print(f"Estimation Failed for {spec_label} - {str(e)}")
    finally:
        sys.stdout = old_stdout

    return new_stdout.getvalue(), res, spec_name, res_fs, spec_number

def print_cluster_diagnostics(df):
    cluster_var = 'CodConglomeradoPrudencial'
    Ns = df.groupby(cluster_var).size()
    G_nominal = len(Ns)
    mean_Ng = np.mean(Ns)
    std_Ng = np.std(Ns, ddof=0)
    cv_Ng = std_Ng / mean_Ng if mean_Ng > 0 else 0
    G_star = max(1.0, G_nominal / (1 + (cv_Ng ** 2)))
    total_obs = len(df)
    return G_nominal, G_star, mean_Ng, std_Ng, cv_Ng, total_obs, Ns

def scale_magnitudes(df):
    scale_cols = {
        'gdp_per_capita': 10000.0, 'cadunico_families_per1000': 100.0,
        'pix_users_pf_per1000': 100.0, 'connections_per100': 100.0,
        'deposit_balance': 1000000000.0, 'nr_lagged_dep': 1000000000.0,
        'lagged_deposits': 1000000000.0
    }
    for col, factor in scale_cols.items():
        if col in df.columns: df[col] /= factor
    return df

def define_specifications():
    s_base = ['constant', 'post_2020']
    s_macro = s_base + ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young', 'risk_free_qoq_lag']
    s_tech_finance = s_macro + ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']

    iv_specs = {'OLS': [], 'IV_CostShifters': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag'],
               'IV_Wholesale': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag'],
               'IV_HausmanFull': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag', 'leave_one_out_mean_spread']}
    state_blocks = {'Base': s_base, 'Macro': s_macro, 'Tech': s_tech_finance}
    return s_tech_finance, iv_specs, state_blocks

def do_estimation():
    print("=== PHASE 1: ESTIMATION (CFA) ===")
    _, output_dir = _resolve_runtime_paths()
    df = build_estimation_data()
    print(f"Panel size after cleaning: {len(df)} rows")

    G_nominal, G_star, mean_Ng, std_Ng, cv_Ng, total_obs, Ns = print_cluster_diagnostics(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    df['constant'] = 1.0
    df = scale_magnitudes(df)

    s_tech_finance, specs, state_blocks = define_specifications()
    for col in s_tech_finance:
        if col != 'constant' and col in df.columns: df[col] = df[col].fillna(df[col].median())
    
    iv_order = ['OLS', 'IV_CostShifters', 'IV_Wholesale', 'IV_HausmanFull']
    state_order = ['Base', 'Macro', 'Tech']

    import itertools
    tasks = [(df, iv_name, specs[iv_name], s_name, state_blocks[s_name], i)
             for i, (s_name, iv_name) in enumerate(itertools.product(state_order, iv_order), start=1)]

    print(f"Starting parallel execution of {len(tasks)} specifications...")     
    results_dict = {}
    out_results = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        out_results.extend(executor.map(execute_specification, tasks))
        
    for res in out_results:
        stdout_text, res_ss, spec_name, res_fs, spec_number = res
        print(stdout_text)
        if res_ss is not None:
            safe_name = f"Spec{spec_number:02d}_" + spec_name.replace(" ", "_").replace("/", "").replace(":", "")
            tex_file = output_dir / f"{safe_name}.tex"
            with open(tex_file, 'w') as f: f.write(res_ss.summary().as_latex())
            if res_fs is not None:
                tex_file_fs = output_dir / f"{safe_name}_FirstStage.tex"
                with open(tex_file_fs, 'w') as f: f.write(res_fs.summary().as_latex())
            results_dict[spec_name] = {'spec_number': spec_number, 'spec_label': f"({spec_number}) {spec_name}", 'second_stage': res_ss, 'first_stage': res_fs}
            
    cluster_diagnostics = {
        'G_nominal': G_nominal, 'G_star': float(G_star), 'mean_obs_per_cluster': float(mean_Ng), 
        'std_obs_per_cluster': float(std_Ng), 'coefficient_variation': float(cv_Ng), 'total_observations': total_obs,
        'top_5_clusters': {str(c): {'observations': n, 'share_pct': float((n / total_obs) * 100)} for c, n in Ns.sort_values(ascending=False).head(5).items()}
    }

    diag_file = output_dir / "cluster_diagnostics.json"
    with open(diag_file, 'w') as f: json.dump(cluster_diagnostics, f, indent=2)
    results_pickle = output_dir / "estimation_results.pkl"
    with open(results_pickle, 'wb') as f: pickle.dump(results_dict, f)
    print(f"Estimation outputs successfully saved in: {output_dir}")

# ==============================================================================
# PHASE 2: PHI GENERATION
# ==============================================================================
def calculate_phis(df, res_dict, state_blocks):
    df['constant'] = 1.0
    scale_cols = {'gdp_per_capita': 10000.0, 'cadunico_families_per1000': 100.0, 'pix_users_pf_per1000': 100.0, 'connections_per100': 100.0}
    for col, factor in scale_cols.items():
        if col in df.columns: df[col] /= factor

    phi_results = {}
    all_s_cols = {col for cols in state_blocks.values() for col in cols if col != 'constant'}
    filled_cols = {sv: df[sv].fillna(df[sv].median()) if sv in df.columns else np.zeros(len(df)) for sv in all_s_cols}

    if 'lagged_deposits' in df.columns: df['market_size'] = df['lagged_deposits']
    elif 'deposit_balance' in df.columns: df['market_size'] = df['deposit_balance']
    else: df['market_size'] = 1.0

    for spec_name, s_cols in state_blocks.items():
        model_key = f"IV_HausmanFull x {spec_name}"
        # We need to search effectively because of how the dict string is named (might have spec prefix)
        actual_key = next((k for k in res_dict.keys() if model_key in k), None)
        if not actual_key: continue
            
        ss_res = res_dict[actual_key]['second_stage']
        if ss_res is None: continue
            
        phi_mt = np.zeros(len(df))
        for sv in s_cols:
            col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
            if col_name in ss_res.params:
                c = ss_res.params[col_name]
                phi_mt += c if sv == 'constant' else c * filled_cols[sv]
        
        df[f'phi_mt_{spec_name}'] = phi_mt
        market_agg = df.groupby(['year_quarter', 'CODMUN_IBGE'], observed=True).agg(phi_mt=(f'phi_mt_{spec_name}', 'mean'), M_mt=('market_size', 'sum')).reset_index()
        weighted_phi = market_agg['phi_mt'] * market_agg['M_mt']
        sum_weighted = weighted_phi.groupby(market_agg['year_quarter']).sum()
        sum_m_mt = market_agg['M_mt'].groupby(market_agg['year_quarter']).sum()
        national_agg = (sum_weighted / sum_m_mt.replace(0, np.nan)).fillna(0).reset_index(name=f'phi_t_{spec_name}')
        phi_results[spec_name] = national_agg
        
    return df, phi_results

def do_phi_generation():
    print("\n=== PHASE 2: PHI CONSTRUCTION & EXPORT ===")
    panel_csv, output_dir = _resolve_runtime_paths()
    results_pickle_path = output_dir / "estimation_results.pkl"
    
    df = pd.read_csv(panel_csv, low_memory=False)
    if 'year' in df.columns: df['post_2020'] = (df['year'] >= 2020).astype(int)

    df['entity_id'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['mca_code'].astype(str)
    df.sort_values(by=['entity_id', 'year', 'quarter'], inplace=True)
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)

    print(f"Loading results from {results_pickle_path}")
    try:
        with open(results_pickle_path, 'rb') as f: res_dict = pickle.load(f)
    except Exception as e:
        print(f"Could not load pickle! Did the estimation run successfully? Error: {e}")
        return

    s_tech_finance, specs, state_blocks = define_specifications()
    df, national_phis = calculate_phis(df, res_dict, state_blocks)
    
    out_path1 = output_dir / "market_panel_phis.csv"
    df.to_csv(out_path1, index=False)
    print(f"Exported panel data with computed phi_mt to {out_path1}")

    from functools import reduce
    if national_phis:
        agg_df = reduce(lambda left, right: pd.merge(left, right, on='year_quarter', how='outer'), national_phis.values())
        out_path2 = output_dir / "national_phi_t.csv"
        agg_df.to_csv(out_path2, index=False)
        print(f"Exported National Phi_t to {out_path2}")
        
    print("Done!")

def main():
    pd.options.mode.chained_assignment = None
    parser = argparse.ArgumentParser(description="Unified Estimation 1 Sleep script.")
    parser.add_argument("--skip-estimation", action="store_true", help="Skip the first/second stage estimation phase.")
    parser.add_argument("--skip-phi", action="store_true", help="Skip the phi generation phase.")
    args = parser.parse_args()

    if not args.skip_estimation: do_estimation()
    if not args.skip_phi: do_phi_generation()

if __name__ == "__main__":
    main()
