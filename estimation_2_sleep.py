"""
estimation_2_sleep.py
==============================
Robustness check: Estimates the depositor sleepiness function omitting the 
post_2020 structural break dummy.

This script sequentially runs:
  1. Local (B-type) CFA estimation (Base, Base_Selic)
  2. Local implied Phi_mt generation
  3. National (D-type) direct estimation (Base, Base_Selic, Options 1/2/3)
  4. Plots containing these robustness bounds

Outputs are directed to ESTIMATION_OUTPUT/SLEEPINESS/SLEEPINESS_NO_BREAK

CLI Options:
------------
usage: estimation_2_sleep.py [-h] [--spec12]

Estimation 2: Sleepiness Robustness

options:
  -h, --help  show this help message and exit
  --spec12    Only run spec 12 (Tech x IV_HausmanFull)
"""
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import sys
import json
import pickle
import warnings
from contextlib import suppress
import concurrent.futures

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

try:
    from utils import load_panel_cached
except Exception:
    load_panel_cached = None

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")

# ==============================================================================
# GLOBAL SETUP
# ==============================================================================
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
PANEL_CSV = DATA_DIR / "market_panel.csv"
OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "rout_2"

LOCAL_DIR = OUTPUT_DIR / "LOCAL"
NATIONAL_DIR = OUTPUT_DIR / "NATIONAL"
PLOTS_DIR = OUTPUT_DIR / "PLOTS"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
NATIONAL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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

def define_specifications():
    s_base = ['constant']
    s_base_selic = ['constant', 'risk_free_qoq_lag']
    s_macro = s_base_selic + ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young']
    s_tech = s_macro + ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']

    iv_specs = {'OLS': [], 'IV_CostShifters': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag'],
               'IV_Wholesale': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag'],
               'IV_HausmanFull': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag', 'leave_one_out_mean_spread']}
    state_blocks = {'Base': s_base, 'Base_Selic': s_base_selic, 'Macro': s_macro, 'Tech': s_tech}
    return ['risk_free_qoq_lag'], iv_specs, state_blocks

# ==============================================================================
# PHASE 1 & 2: LOCAL (B-TYPE) ESTIMATION AND PHI
# ==============================================================================
def build_local_data():
    df_raw = load_panel_cached(PANEL_CSV) if load_panel_cached else pd.read_csv(PANEL_CSV, dtype={'mca_code': str}, low_memory=False)
    # Restrict to B-type
    df_raw = df_raw[df_raw['CODMUN_IBGE'].astype(str) != '0'].copy()
    
    if 'dep_a1' in df_raw.columns:
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
    
    if 'leave_one_out_mean_spread' not in df.columns: df['leave_one_out_mean_spread'] = np.nan

    df['bank_year'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['year'].astype(str)
    df = df.dropna(subset=['deposit_balance', 'nr_lagged_dep', 'spread_qoq', 'entity_id', 'time_id'])
    
    df['constant'] = 1.0
    scale_cols = {'deposit_balance': 1e9, 'nr_lagged_dep': 1e9, 'lagged_deposits': 1e9}
    for col, factor in scale_cols.items():
        if col in df.columns: df[col] /= factor
        
    if 'gdp_per_capita' in df.columns: df['gdp_per_capita'] /= 10000.0
    if 'cadunico_families_per1000' in df.columns: df['cadunico_families_per1000'] /= 100.0
    if 'pix_users_pf_per1000' in df.columns: df['pix_users_pf_per1000'] /= 100.0
    if 'connections_per100' in df.columns: df['connections_per100'] /= 100.0

    s_tech_finance = ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']
    for col in s_tech_finance:
        if col in df.columns: df[col] = df[col].fillna(df[col].median())
    return df

def run_local_first_stage(df, spec_instruments, exogenous_controls):
    endog_mask = df['deposit_type'].isin([4, 5])
    first_stage_vars = list(set(spec_instruments + exogenous_controls))
    if 'constant' in first_stage_vars: first_stage_vars.remove('constant')

    valid_mask = endog_mask & df[first_stage_vars].notnull().all(axis=1)
    df_fs = df[valid_mask].copy()
    
    if len(df_fs) == 0:
        df['v_hat'] = 0.0
        return df, None

    mod = sm.OLS(df_fs['spread_qoq'], sm.add_constant(df_fs[first_stage_vars]))
    cluster_series = df_fs['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    res = apply_imbalanced_cluster_correction(res, cluster_series)

    df['v_hat'] = 0.0
    df.loc[valid_mask, 'v_hat'] = res.resid
    df['v_hat_2'] = df['v_hat'] ** 2
    df['v_hat_3'] = df['v_hat'] ** 3
    return df, res

def run_local_second_stage(df, state_vars, has_cf=False):
    X_cols = []
    for sv in state_vars:
        col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
        df[col_name] = df[sv] * df['nr_lagged_dep'] if sv != 'constant' else df['nr_lagged_dep']
        X_cols.append(col_name)
    if has_cf: X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])
        
    df_ss = df.dropna(subset=X_cols + ['deposit_balance']).copy()
    if len(df_ss) == 0: return None
        
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance']
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')
    
    mod = sm.OLS(y_dm, X_dm)
    cluster_series = df_ss['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    return apply_imbalanced_cluster_correction(res, cluster_series)

def exec_local_spec(args):
    df, iv_name, iv_cols, s_name, s_cols = args
    has_cf = len(iv_cols) > 0
    spec_name = f"{iv_name} x {s_name}"

    df_target = df
    res_fs = None
    if has_cf:
        iv_cols_act = [c for c in iv_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        exog_cols_act = [c for c in s_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        if not iv_cols_act: return None, spec_name, None
        df_target, res_fs = run_local_first_stage(df_target, iv_cols_act, exog_cols_act)

    res_ss = run_local_second_stage(df_target, s_cols, has_cf=has_cf)
    return res_ss, spec_name, res_fs

def calculate_local_phis(df, res_dict, state_blocks):
    phi_results = {}
    # Use population as market-size weight for phi aggregation.
    # Under the constant-fraction assumption (M_mt = c * pop_mt), c cancels
    # in the ratio Σ(phi * M) / Σ(M), making pop_total the correct weight.
    if 'pop_total' in df.columns:
        df['market_size'] = df['pop_total'].fillna(0)
    else:
        df['market_size'] = 1.0

    for spec_name, s_cols in state_blocks.items():
        model_key = f"IV_HausmanFull x {spec_name}"
        if model_key not in res_dict or res_dict[model_key]['second_stage'] is None: continue
            
        ss_res = res_dict[model_key]['second_stage']
        phi_mt = np.zeros(len(df))
        for sv in s_cols:
            col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
            if col_name in ss_res.params:
                c = ss_res.params[col_name]
                phi_mt += c if sv == 'constant' else c * df[sv].fillna(0)
        
        df[f'phi_mt_{spec_name}'] = phi_mt
        market_agg = df.groupby(['year_quarter', 'CODMUN_IBGE'], observed=True).agg(phi_mt=(f'phi_mt_{spec_name}', 'mean'), M_mt=('market_size', 'sum')).reset_index()
        weighted_phi = market_agg['phi_mt'] * market_agg['M_mt']
        national_agg = (weighted_phi.groupby(market_agg['year_quarter']).sum() / market_agg['M_mt'].groupby(market_agg['year_quarter']).sum().replace(0, np.nan)).fillna(0).reset_index(name=f'phi_t_{spec_name}')
        phi_results[spec_name] = national_agg
    return df, phi_results

def run_local_phase(spec12_only=False):
    print("=== PHASE 1: LOCAL ROBUSTNESS ===")
    df = build_local_data()
    _, iv_specs, state_blocks = define_specifications()
    
    if spec12_only:
        tasks = [(df, 'IV_HausmanFull', iv_specs['IV_HausmanFull'], 'Tech', state_blocks['Tech'])]
    else:
        tasks = [(df, iv_name, iv_specs[iv_name], s_name, state_blocks[s_name])
                 for s_name in state_blocks.keys() for iv_name in ['OLS', 'IV_CostShifters', 'IV_Wholesale', 'IV_HausmanFull']]
    
    results_dict = {}
    _nw = max(1, (os.cpu_count() or 4) // max(1, int(os.environ.get('SLEEP_PIPELINE_NSLOTS', '1'))))
    with concurrent.futures.ProcessPoolExecutor(max_workers=_nw) as executor:
        for res_ss, spec_name, res_fs in executor.map(exec_local_spec, tasks):
            if res_ss is not None:
                results_dict[spec_name] = {'second_stage': res_ss, 'first_stage': res_fs}
                print(f"Local Computed: {spec_name}")

    with open(LOCAL_DIR / "estimation_results.pkl", 'wb') as f: pickle.dump(results_dict, f)
    
    df['year_quarter'] = df['time_id']
    df, national_phis = calculate_local_phis(df, results_dict, state_blocks)
    df.to_csv(LOCAL_DIR / "market_panel_phis.csv", index=False)
    print("Saved Local Robustness Pickles and Phis.\n")


# ==============================================================================
# PHASE 3: NATIONAL (D-TYPE) ESTIMATION
# ==============================================================================
def build_national_data():
    df_raw = load_panel_cached(PANEL_CSV) if load_panel_cached else pd.read_csv(PANEL_CSV, dtype={'mca_code': str}, low_memory=False)
    df = df_raw[df_raw['Source'] == 'IFDATA'].copy()
    df = df.drop_duplicates(subset=['CodConglomeradoPrudencial', 'year', 'quarter'])

    if 'deposit_balance' not in df.columns:
        _key_cols = ['CodConglomeradoPrudencial', 'year', 'quarter']
        _stub_prefixes = ('dep_a', 'spread_a', 'leave_one_out_mean_spread_a')
        _stub_cols = {c for c in df.columns if any(c.startswith(p) for p in _stub_prefixes)}
        _other_cols = [c for c in df.columns if c not in _stub_cols and c not in _key_cols]
        df_pivoted = pd.wide_to_long(
            df[_key_cols + list(_stub_cols)],
            stubnames=['dep_a', 'spread_a', 'leave_one_out_mean_spread_a'],
            i=_key_cols, j='deposit_type',
        ).reset_index()
        df_pivoted = df_pivoted.rename(columns={'dep_a': 'deposit_balance', 'spread_a': 'spread_qoq'})
        df = df_pivoted.merge(df[_key_cols + _other_cols].drop_duplicates(_key_cols), on=_key_cols, how='left')
        df = df[df['Source'] == 'IFDATA'].copy()
    
    if 'deposit_type' in df.columns: df = df[~df['deposit_type'].astype(str).str.contains('3')].copy()
    df['entity_id'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['deposit_type'].astype(str)
    df['year_quarter'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)
    
    df = df.sort_values(by=['entity_id', 'year', 'quarter'])
    df['lagged_deposits'] = df.groupby('entity_id')['deposit_balance'].shift(1)
    df['spread_qoq_lag'] = df.groupby('entity_id')['spread_qoq'].shift(1)
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)
    df['nr_lagged_dep'] = (1 + df['risk_free_qoq_lag'] - df['spread_qoq_lag']) * df['lagged_deposits']

    df['constant'] = 1.0
    scale_cols = {'deposit_balance': 1e9, 'nr_lagged_dep': 1e9, 'lagged_deposits': 1e9}
    for col, factor in scale_cols.items():
        if col in df.columns: df[col] /= factor
        
    if 'gdp_per_capita' in df.columns: df['gdp_per_capita'] /= 10000.0
    if 'cadunico_families_per1000' in df.columns: df['cadunico_families_per1000'] /= 100.0
    if 'pix_users_pf_per1000' in df.columns: df['pix_users_pf_per1000'] /= 100.0
    if 'connections_per100' in df.columns: df['connections_per100'] /= 100.0

    s_tech_finance = ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']
    for col in s_tech_finance:
        if col in df.columns: df[col] = df[col].fillna(df[col].median())
    return df

def run_nat_first_stage(df, spec_instruments, exogenous_controls):
    endog_mask = df['deposit_type'].isin([4, 5])
    first_stage_vars = list(set(spec_instruments + exogenous_controls))
    if 'constant' in first_stage_vars: first_stage_vars.remove('constant')

    valid_mask = endog_mask & df[first_stage_vars].notnull().all(axis=1)
    df_fs = df[valid_mask].copy()
    if len(df_fs) == 0:
        df['v_hat'] = 0.0
        return df, None

    mod = sm.OLS(df_fs['spread_qoq'], sm.add_constant(df_fs[first_stage_vars]))
    cluster_series = df_fs['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    res = apply_imbalanced_cluster_correction(res, cluster_series)

    df['v_hat'] = 0.0
    df.loc[valid_mask, 'v_hat'] = res.resid
    df['v_hat_2'] = df['v_hat'] ** 2
    df['v_hat_3'] = df['v_hat'] ** 3
    return df, res

def exec_nat_model_opt1(df, svs, has_cf):
    X_cols = [f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep" for sv in svs]
    for sv, c in zip(svs, X_cols): df[c] = df[sv] * df['nr_lagged_dep'] if sv != 'constant' else df['nr_lagged_dep']
    if has_cf: X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])
    
    df_ss = df.dropna(subset=X_cols + ['deposit_balance']).copy()
    if len(df_ss) == 0: return None
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance']
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')[X_cols]
    mod = sm.OLS(y_dm, X_dm)
    cluster_series = df_ss['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    return apply_imbalanced_cluster_correction(res, cluster_series)
    
def exec_nat_model_opt2(df, svs, has_cf):
    X_cols = []
    df['log_total_assets_lag'] = df['log_total_assets_lag'].fillna(df['log_total_assets_lag'].median())
    for sv in svs:
        if sv == 'constant': X_cols.append("nr_lagged_dep")
        else:
            c_base = f"interaction_{sv}"
            c_cross = f"interaction_{sv}_x_assets"
            df[c_base] = df[sv] * df['nr_lagged_dep']
            df[c_cross] = df[sv] * df['log_total_assets_lag'] * df['nr_lagged_dep']
            X_cols.extend([c_base, c_cross])
    if has_cf: X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])
    df_ss = df.dropna(subset=X_cols + ['deposit_balance']).copy()
    if len(df_ss) == 0: return None
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance']
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')[X_cols]
    mod = sm.OLS(y_dm, X_dm)
    cluster_series = df_ss['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    return apply_imbalanced_cluster_correction(res, cluster_series)

def get_pca_index(df_q, s_cols):
    sv_dynamic = [sv for sv in s_cols if sv != 'constant']
    if len(sv_dynamic) < 1: return None
    df_pca = df_q.dropna(subset=sv_dynamic).copy()
    ts_df = df_pca[['year_quarter'] + sv_dynamic].drop_duplicates().sort_values('year_quarter')
    ts_vals = ts_df[sv_dynamic].values
    pca = PCA(n_components=1)
    ts_mean = np.mean(ts_vals, axis=0)
    ts_std = np.std(ts_vals, axis=0)
    ts_std[ts_std == 0] = 1 
    ts_norm = (ts_vals - ts_mean) / ts_std
    ts_pca = pca.fit_transform(ts_norm)
    ts_df['pca_index'] = ts_pca[:, 0]
    return ts_df

def exec_nat_model_opt3(df, svs, has_cf):
    sv_dynamic = [sv for sv in svs if sv != 'constant']
    if len(sv_dynamic) < 1: return None
    
    ts_df = get_pca_index(df, svs)
    if ts_df is None: return None
        
    df_ss = df.merge(ts_df[['year_quarter', 'pca_index']], on='year_quarter', how='inner')
    df_ss['interaction_pca_index'] = df_ss['pca_index'] * df_ss['nr_lagged_dep']
    X_cols = ['nr_lagged_dep', 'interaction_pca_index']
    
    if has_cf: X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])
        
    df_ss = df_ss.dropna(subset=X_cols + ['deposit_balance']).copy()
    if len(df_ss) == 0: return None
    
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance'].astype(float)
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')[X_cols].astype(float)

    mod = sm.OLS(y_dm, X_dm)
    cluster_series = df_ss['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    return apply_imbalanced_cluster_correction(res, cluster_series)


def run_national_phase(spec12_only=False):
    print("=== PHASE 2: NATIONAL ROBUSTNESS ===")
    df = build_national_data()
    _, iv_specs, state_blocks = define_specifications()

    results_dict = {}
    
    states_to_run = ['Tech'] if spec12_only else state_blocks.keys()
    ivs_to_run = ['IV_HausmanFull'] if spec12_only else iv_specs.keys()
    
    for opt in [1, 2, 3]:
        for s_name in states_to_run:
            s_cols = state_blocks[s_name]
            for iv_name in ivs_to_run:
                iv_cols = iv_specs.get(iv_name, [])
                has_cf = len(iv_cols) > 0
                res_fs = None

                if has_cf:
                    iv_cols_act = [c for c in iv_cols if c in df.columns and df[c].notnull().sum() > 0]
                    exog_cols_act = [c for c in s_cols if c in df.columns and df[c].notnull().sum() > 0]
                    if not iv_cols_act: continue
                    df_t, res_fs = run_nat_first_stage(df.copy(), iv_cols_act, exog_cols_act)
                else:
                    df_t = df
                    
                res_ss = None
                if opt == 1: res_ss = exec_nat_model_opt1(df_t, s_cols, has_cf)
                elif opt == 2: res_ss = exec_nat_model_opt2(df_t, s_cols, has_cf)
                elif opt == 3: res_ss = exec_nat_model_opt3(df_t, s_cols, has_cf)
                
                if res_ss is not None:
                    spec_name = f"Option_{opt}_{iv_name}_{s_name}"
                    results_dict[spec_name] = {'option': opt, 'block': s_name, 'second_stage': res_ss, 'first_stage': res_fs}
                    print(f"Nat Computed: {spec_name}")

    with open(NATIONAL_DIR / "estimation_results.pkl", 'wb') as f: pickle.dump(results_dict, f)
    print("Saved National Robustness Pickles.\n")

# ==============================================================================
# PHASE 4: PLOTTING
# ==============================================================================
def predict_phi(params, cov_params, param_vector):
    if params is None or cov_params is None: return np.nan, np.nan
    V = np.array([param_vector.get(name, 0) for name in params.index])
    return V.dot(params), np.sqrt(V.dot(cov_params).dot(V))

def run_plotting_phase(spec12_only=False):
    print("=== PHASE 3: PLOTTING ROBUSTNESS ===")
    try:
        with open(LOCAL_DIR / "estimation_results.pkl", 'rb') as f: loc_res = pickle.load(f)
        with open(NATIONAL_DIR / "estimation_results.pkl", 'rb') as f: nat_res = pickle.load(f)
    except FileNotFoundError:
        print("Missing Pickles.")
        return

    df = load_panel_cached(PANEL_CSV) if load_panel_cached else pd.read_csv(PANEL_CSV, low_memory=False)
    df['year_quarter'] = df['year'].astype(int).astype(str) + "Q" + df['quarter'].astype(int).astype(str)
    # Population weights for plotting aggregation (consistent with phi construction)
    df['market_size'] = df['pop_total'].fillna(0) if 'pop_total' in df.columns else 1.0
    df['entity_id'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['mca_code'].astype(str)
    df = df.sort_values(by=['entity_id', 'year', 'quarter'])
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)
    df['log_total_assets_lag'] = df['log_total_assets_lag'].fillna(df['log_total_assets_lag'].median())
    
    scale_cols = {'gdp_per_capita': 10000.0, 'cadunico_families_per1000': 100.0, 'pix_users_pf_per1000': 100.0, 'connections_per100': 100.0}
    for col, factor in scale_cols.items():
        if col in df.columns: df[col] /= factor
        
    s_tech_finance = ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']
    for col in s_tech_finance:
        if col in df.columns: df[col] = df[col].fillna(df[col].median())
    
    yq_list = sorted(df['year_quarter'].dropna().unique())
    assets_agg = {}
    for yq in yq_list:
        sub = df[df['year_quarter'] == yq]
        m_tot = sub['market_size'].sum()
        assets_agg[yq] = (sub['log_total_assets_lag'] * sub['market_size']).sum() / m_tot if m_tot > 0 else 0

    _, iv_specs, state_blocks = define_specifications()
    
    states_to_run = ['Tech'] if spec12_only else state_blocks.keys()
    ivs_to_run = ['IV_HausmanFull'] if spec12_only else ['OLS', 'IV_CostShifters', 'IV_Wholesale', 'IV_HausmanFull']

    for s_name in states_to_run:
        s_cols = state_blocks[s_name]
        for iv_name in ivs_to_run:
            loc_model = loc_res.get(f"{iv_name} x {s_name}", {}).get('second_stage')
            n1_model = nat_res.get(f"Option_1_{iv_name}_{s_name}", {}).get('second_stage')
            n2_model = nat_res.get(f"Option_2_{iv_name}_{s_name}", {}).get('second_stage')
            n3_model = nat_res.get(f"Option_3_{iv_name}_{s_name}", {}).get('second_stage')
            
            pca_df = get_pca_index(df, s_cols)
            pca_dict = pca_df.set_index('year_quarter')['pca_index'].to_dict() if pca_df is not None else {}
            
            def ext(m): return (m.params, m.cov_params()) if m is not None else (None, None)
            
            res_df = []
            for yq in yq_list:
                sub = df[df['year_quarter'] == yq]
                if len(sub) == 0: continue
                
                s_vals = {sv: (sub[sv].median() if sv != 'constant' else 1.0) for sv in s_cols}

                vec_loc = {}
                for sv in s_cols:
                    nm = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
                    vec_loc[nm] = s_vals[sv]
                
                e_loc, se_loc = predict_phi(*ext(loc_model), vec_loc)

                vec_n1 = vec_loc.copy()
                e_n1, se_n1 = predict_phi(*ext(n1_model), {**vec_n1, 'constant': 1.0})
                
                vec_n2 = vec_loc.copy()
                a_lag = assets_agg.get(yq, 0)
                for sv in s_cols:
                    if sv != 'constant': vec_n2[f"interaction_{sv}_x_assets"] = s_vals[sv] * a_lag
                e_n2, se_n2 = predict_phi(*ext(n2_model), {**vec_n2, 'constant': 1.0})
                
                e_n3, se_n3 = np.nan, np.nan
                if n3_model is not None and yq in pca_dict:
                    vec_n3 = {'nr_lagged_dep': 1.0, 'interaction_pca_index': pca_dict[yq]}
                    e_n3, se_n3 = predict_phi(*ext(n3_model), {**vec_n3, 'constant': 1.0})
                
                res_df.append({
                    'year_quarter': yq, 'loc_est': e_loc, 'loc_se': se_loc,
                    'n1_est': e_n1, 'n1_se': se_n1, 'n2_est': e_n2, 'n2_se': se_n2,
                    'n3_est': e_n3, 'n3_se': se_n3
                })
            
            res_df = pd.DataFrame(res_df)
            res_df['date'] = pd.PeriodIndex(res_df['year_quarter'].str.replace('_', 'Q'), freq='Q').to_timestamp()
            res_df = res_df.sort_values('date')

            configs_full = [
                ('loc', r'Local $\hat{\phi}^{Loc}$', 'blue'),
                ('n1', r'National $\hat{\phi}^{Nat,1}$', 'green'),
                ('n2', r'National $\hat{\phi}^{Nat,2}$', 'orange'),
                ('n3', r'National $\hat{\phi}^{Nat,3}$', 'purple')
            ]
            configs_default = [
                ('loc', r'Local $\hat{\phi}^{Loc}$', 'blue'),
                ('n3', r'National $\hat{\phi}^{Nat,3}$', 'purple')
            ]

            def plot_for_config(configs, suffix):
                fig, ax = plt.subplots(figsize=(10, 6))
                plotted = False
                for pfx, label, color in configs:
                    e, se = res_df[f'{pfx}_est'], res_df[f'{pfx}_se']
                    if e.notnull().sum() > 0:
                        plotted = True
                        ax.plot(res_df['date'], e, label=label, color=color, linewidth=2)
                        ax.fill_between(res_df['date'], e - 1.96*se, e + 1.96*se, color=color, alpha=0.15)
                        
                if plotted:
                    ax.set_title(f"Robustness: {iv_name} x {s_name}", fontsize=14)
                    ax.set_ylabel(r"National $\hat{\phi}_t$")
                    ax.set_ylim(bottom=0)
                    ax.grid()
                    ax.legend(loc='best')
                    fig.tight_layout()
                    plt.savefig(PLOTS_DIR / f"Robustness_{iv_name}_{s_name}{suffix}.png", dpi=300)
                plt.close(fig)

            plot_for_config(configs_full, "_full")
            plot_for_config(configs_default, "")
    print("Plots generated.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimation 2: Sleepiness Robustness")
    parser.add_argument('--spec12', action='store_true', help='Only run spec 12 (Tech x IV_HausmanFull)')
    args = parser.parse_args()

    pd.options.mode.chained_assignment = None
    run_local_phase(spec12_only=args.spec12)
    run_national_phase(spec12_only=args.spec12)
    run_plotting_phase(spec12_only=args.spec12)
