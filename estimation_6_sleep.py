"""
estimation_6_sleep.py
==============================
Robustness check: Estimates the depositor sleepiness function using pooled B and D-type firms.
Extends the analysis by introducing controls and interactions for cooperative ('is_coop')
and state-owned ('is_state_owned') institutions across both Linear and Logistic specifications.

CLI Options:
------------
  --model-type   : linear, logistic, both (default: both)
  --alt          : 1, 2, both (default: 2)
  --spec12-only  : Flag to only run Spec 12 (Tech x IV_HausmanFull)
"""
import argparse
from pathlib import Path
import pickle
import warnings
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
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")

# ==============================================================================
# GLOBAL SETUP
# ==============================================================================
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
PANEL_CSV = DATA_DIR / "market_panel.csv"
OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "rout_6"

POOLED_DIR = OUTPUT_DIR / "POOLED"
PLOTS_DIR = OUTPUT_DIR / "PLOTS"

POOLED_DIR.mkdir(parents=True, exist_ok=True)
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

def define_specifications(alt):
    s_base = ['constant', 'dummy_D_type', 'pix_exists']
    
    s_base_selic = ['constant', 'dummy_D_type', 'pix_exists', 'risk_free_qoq_lag', 'dummy_D_type_x_risk_free_qoq_lag']
    s_macro = s_base_selic + ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young']
    s_macro += ['dummy_D_type_x_fraction_65plus', 'dummy_D_type_x_fraction_young']
    s_tech = s_macro + ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']

    base_state_blocks = {'Base': s_base, 'Base_Selic': s_base_selic, 'Macro': s_macro, 'Tech': s_tech}
    state_blocks = {}
    
    for key, block in base_state_blocks.items():
        if alt == 1:
            # Alternative 1: Interact is_coop and is_state_owned with ALL state space variables
            new_block = list(block)
            new_block.extend(['is_coop', 'is_state_owned'])
            for var in block:
                if var != 'constant':
                    new_block.append(f"is_coop_x_{var}")
                    new_block.append(f"is_state_owned_x_{var}")
            state_blocks[key] = list(set(new_block))
            
        elif alt == 2:
            # Alternative 2: Dummy intercept + targeted interactions
            new_block = list(block)
            new_block.extend(['is_coop', 'is_state_owned'])
            
            for interact_var in ['risk_free_qoq_lag', 'fraction_65plus', 'fraction_young']:
                if interact_var in new_block:
                    new_block.extend([f"is_coop_x_{interact_var}", f"is_state_owned_x_{interact_var}"])
            state_blocks[key] = list(set(new_block))
            
    # Guarantee identical ordering
    for key in state_blocks.keys():
        state_blocks[key] = sorted(state_blocks[key])
        if 'constant' in state_blocks[key]:
            state_blocks[key].insert(0, state_blocks[key].pop(state_blocks[key].index('constant')))

    iv_specs = {'OLS': [], 'IV_CostShifters': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag'],
               'IV_Wholesale': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag'],
               'IV_HausmanFull': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag', 'leave_one_out_mean_spread']}
               
    return ['risk_free_qoq_lag'], iv_specs, state_blocks

# ==============================================================================
# DATA BUILDING
# ==============================================================================
def build_pooled_data(alt):
    df_raw = pd.read_csv(PANEL_CSV, dtype={'mca_code': str}, low_memory=False)
    
    df_raw['dummy_D_type'] = (df_raw['CODMUN_IBGE'].astype(str) == '0').astype(float)
    df_raw['pix_exists'] = ((df_raw['year'] > 2020) | ((df_raw['year'] == 2020) & (df_raw['quarter'] == 4))).astype(float)
    
    # Fill defaults for ownership types if missing
    if 'is_coop' not in df_raw.columns: df_raw['is_coop'] = 0.0
    if 'is_state_owned' not in df_raw.columns: df_raw['is_state_owned'] = 0.0
    
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
    
    df['dummy_D_type_x_risk_free_qoq_lag'] = df['dummy_D_type'] * df['risk_free_qoq_lag']
    df['dummy_D_type_x_fraction_65plus'] = df['dummy_D_type'] * df['fraction_65plus']
    df['dummy_D_type_x_fraction_young'] = df['dummy_D_type'] * df['fraction_young']
    
    _, _, state_blocks = define_specifications(alt)
    all_vars = set(val for subset in state_blocks.values() for val in subset)
    
    # Generate all requested dynamic interactions
    base_vars_for_interact = ['risk_free_qoq_lag', 'fraction_65plus', 'fraction_young', 'dummy_D_type', 'gdp_per_capita', 'cadunico_families_per1000', 'pix_exists', 'pix_users_pf_per1000', 'connections_per100', 'branches_per1000', 'dummy_D_type_x_risk_free_qoq_lag', 'dummy_D_type_x_fraction_65plus', 'dummy_D_type_x_fraction_young']
    for v in base_vars_for_interact:
        if v in df.columns:
            if f"is_coop_x_{v}" in all_vars:
                df[f"is_coop_x_{v}"] = df['is_coop'] * df[v]
            if f"is_state_owned_x_{v}" in all_vars:
                df[f"is_state_owned_x_{v}"] = df['is_state_owned'] * df[v]
    
    if 'leave_one_out_mean_spread' not in df.columns: df['leave_one_out_mean_spread'] = np.nan

    df['bank_year'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['year'].astype(str)
    df = df.dropna(subset=['deposit_balance', 'nr_lagged_dep', 'spread_qoq', 'entity_id', 'time_id'])
    
    df['constant'] = 1.0
    scale_cols = {'deposit_balance': 1e9, 'nr_lagged_dep': 1e9, 'lagged_deposits': 1e9}
    for col, factor in scale_cols.items():
        if col in df.columns: df[col] /= factor
        
    for fix_c, fact in [('gdp_per_capita', 10000.0), ('cadunico_families_per1000', 100.0), ('pix_users_pf_per1000', 100.0), ('connections_per100', 100.0)]:
        if fix_c in df.columns: df[fix_c] /= fact

    s_tech_finance = ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young', 'pix_users_pf_per1000', 'connections_per100', 'branches_per1000']
    
    if 'pop_total' in df.columns:
        df_b = df[df['dummy_D_type'] == 0].copy()
        
        for yq in df['time_id'].unique():
            b_yq = df_b[df_b['time_id'] == yq]
            w = b_yq['pop_total'].fillna(0).values
            w_sum = w.sum()
            
            d_idx = (df['time_id'] == yq) & (df['dummy_D_type'] == 1)
            
            for col in s_tech_finance:
                if col in df.columns:
                    if w_sum > 0:
                        vals = b_yq[col].fillna(b_yq[col].median()).values
                        nat_avg = np.average(vals, weights=w)
                    else:
                        nat_avg = b_yq[col].median()
                    df.loc[d_idx, col] = nat_avg

    for col in s_tech_finance:
        if col in df.columns: df[col] = df[col].fillna(df[col].median())
    return df

# ==============================================================================
# ESTIMATION KERNELS
# ==============================================================================
def run_pooled_first_stage(df, spec_instruments, exogenous_controls):
    endog_mask = df['deposit_type'].isin([4, 5])
    first_stage_vars = list(set(spec_instruments + exogenous_controls))
    if 'constant' in first_stage_vars: first_stage_vars.remove('constant')

    valid_mask = endog_mask & df[first_stage_vars].notnull().all(axis=1)
    df_fs = df[valid_mask].copy()
    
    if len(df_fs) == 0:
        df['v_hat'] = 0.0
        return df, None

    mod = sm.OLS(df_fs['spread_qoq'].astype(float), sm.add_constant(df_fs[first_stage_vars].astype(float)))
    cluster_series = df_fs['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    res = apply_imbalanced_cluster_correction(res, cluster_series)

    df['v_hat'] = 0.0
    df.loc[valid_mask, 'v_hat'] = res.resid
    df['v_hat_2'] = df['v_hat'] ** 2
    df['v_hat_3'] = df['v_hat'] ** 3
    return df, res

# --- Linear (From Est 4) ---
def run_pooled_second_stage_linear(df, state_vars, has_cf=False):
    X_cols = []
    for sv in state_vars:
        col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
        df[col_name] = df[sv] * df['nr_lagged_dep'] if sv != 'constant' else df['nr_lagged_dep']
        X_cols.append(col_name)
    if has_cf: X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])
        
    df_ss = df.dropna(subset=X_cols + ['deposit_balance']).copy()
    if len(df_ss) == 0: return None
        
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance'].astype(float)
    X_dm = demean_variables(df_ss, X_cols, 'entity_id').astype(float)
    
    mod = sm.OLS(y_dm, X_dm)
    cluster_series = df_ss['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    return apply_imbalanced_cluster_correction(res, cluster_series)

# --- Logistic (From Est 5) ---
class NonLinearResults:
    def __init__(self, params, bse, tvalues, pvalues, df_resid):
        self.params = params
        self.bse = bse
        self.tvalues = tvalues
        self.pvalues = pvalues
        self.df_resid = df_resid
        self.G_star = df_resid

def nlls_objective(params, y_dm, X, Z, CF, entity_idx):
    theta = params[:X.shape[1]]
    gamma = params[X.shape[1]:] if CF.shape[1] > 0 else []
    X_disp = np.clip(np.dot(X, theta), -700, 700)
    phi = 1.0 / (1.0 + np.exp(-X_disp))
    Y_hat = phi * Z
    if CF.shape[1] > 0: Y_hat += np.dot(CF, gamma)
    sums = np.bincount(entity_idx, weights=Y_hat)
    counts = np.bincount(entity_idx)
    Y_hat_dm = Y_hat - (sums / counts)[entity_idx]
    return y_dm - Y_hat_dm

def run_pooled_second_stage_logistic(df, state_vars, has_cf=False):
    cols = state_vars + ['deposit_balance', 'nr_lagged_dep', 'entity_id']
    CF_cols = ['v_hat', 'v_hat_2', 'v_hat_3'] if has_cf else []
    df_ss = df.dropna(subset=cols + CF_cols).copy()
    if len(df_ss) == 0: return None
    
    entities = df_ss['entity_id'].unique()
    entity_map = {e: i for i, e in enumerate(entities)}
    entity_idx = df_ss['entity_id'].map(entity_map).values
    
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance'].values.astype(float)
    X = df_ss[state_vars].values.astype(float)
    Z = df_ss['nr_lagged_dep'].values.astype(float)
    CF = df_ss[CF_cols].values.astype(float) if has_cf else np.empty((len(df_ss), 0), dtype=float)
    
    init_params = np.zeros(X.shape[1] + CF.shape[1])
    res_lsq = least_squares(nlls_objective, init_params, args=(y_dm, X, Z, CF, entity_idx), method='lm', max_nfev=500)
    
    J = res_lsq.jac
    try: cov = np.linalg.pinv(J.T.dot(J)) * (np.sum(res_lsq.fun**2) / (len(y_dm) - len(init_params)))
    except: cov = np.eye(len(init_params))
    bse = np.sqrt(np.abs(np.diag(cov)))
    
    idx = [f'interaction_{sv}' if sv != 'constant' else 'nr_lagged_dep' for sv in state_vars] + CF_cols
    ps = pd.Series(index=idx, data=res_lsq.x)
    bs = pd.Series(index=idx, data=bse)
    tvals = ps / bs
    
    cluster_series = df_ss['CodConglomeradoPrudencial']
    sizes = cluster_series.value_counts()
    G_star = max(1.0, len(sizes) / (1 + (np.std(sizes)/np.mean(sizes))**2 if np.mean(sizes)>0 else 1))
    pvals = pd.Series(stats.t.sf(np.abs(tvals), df=G_star) * 2, index=idx)
    
    return NonLinearResults(ps, bs, tvals, pvals, G_star)

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
def exec_pooled_spec(args):
    df, iv_name, iv_cols, s_name, s_cols, model_type = args
    has_cf = len(iv_cols) > 0
    spec_name = f"{iv_name} x {s_name} x {model_type}"
    
    df_target = df.copy()
    res_fs = None
    if has_cf:
        iv_cols_act = [c for c in iv_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        exog_cols_act = [c for c in s_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        if not iv_cols_act: return None, spec_name, None
        df_target, res_fs = run_pooled_first_stage(df_target, iv_cols_act, exog_cols_act)

    if model_type == 'linear':
        res_ss = run_pooled_second_stage_linear(df_target, s_cols, has_cf=has_cf)
    else:
        res_ss = run_pooled_second_stage_logistic(df_target, s_cols, has_cf=has_cf)
        
    return res_ss, spec_name, res_fs

def calculate_pooled_phis(df, res_dict, state_blocks):
    phi_results = {}
    df['market_size'] = df['lagged_deposits'] if 'lagged_deposits' in df.columns else (df['deposit_balance'] if 'deposit_balance' in df.columns else 1.0)
        
    for model_key, res_item in res_dict.items():
        if res_item['second_stage'] is None: continue
        ss_res = res_item['second_stage']
        
        try:
            m_parts = model_key.split(' x ')
            spec_name = m_parts[1]
            model_type = m_parts[2]
        except ValueError: continue
            
        s_cols = state_blocks.get(spec_name, [])
        if not s_cols: continue
        
        X_theta = np.zeros(len(df))
        for sv in s_cols:
            col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
            if col_name in ss_res.params:
                c = ss_res.params[col_name]
                if sv == 'constant': X_theta += c
                else: X_theta += c * df[sv].astype(float).fillna(0)
                    
        if model_type == 'logistic':
            phi_mt = 1.0 / (1.0 + np.exp(-np.clip(X_theta.astype(float), -700, 700)))
        else:
            phi_mt = X_theta # linear representation
            
        safe_key = model_key.replace(' ', '_').replace('.', '')
        df[f'phi_mt_{safe_key}'] = phi_mt
        
        market_agg = df.groupby(['year_quarter', 'CODMUN_IBGE'], observed=True).agg(phi_mt=(f'phi_mt_{safe_key}', 'mean'), M_mt=('market_size', 'sum')).reset_index()
        weighted_phi = market_agg['phi_mt'] * market_agg['M_mt']
        national_agg = (weighted_phi.groupby(market_agg['year_quarter']).sum() / market_agg['M_mt'].groupby(market_agg['year_quarter']).sum().replace(0, np.nan)).fillna(0).reset_index(name=f'phi_t_{safe_key}')
        phi_results[safe_key] = national_agg
        
    return df, phi_results

def run_pipeline_for_alt(alt, model_type_arg, spec12_only):
    print(f"\n=== ESTIMATION 6: ALT {alt} ===")
    df = build_pooled_data(alt)
    _, iv_specs, state_blocks = define_specifications(alt)
    
    models_to_run = ['linear', 'logistic'] if model_type_arg == 'both' else [model_type_arg]
    
    tasks = []
    for mt in models_to_run:
        if spec12_only:
            tasks.append((df, 'IV_HausmanFull', iv_specs['IV_HausmanFull'], 'Tech', state_blocks['Tech'], mt))
        else:
            tasks.extend([(df, iv_name, iv_specs[iv_name], s_name, state_blocks[s_name], mt)
                          for s_name in state_blocks.keys() for iv_name in ['OLS', 'IV_CostShifters', 'IV_Wholesale', 'IV_HausmanFull']])
    
    results_dict = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res_ss, spec_name, res_fs in executor.map(exec_pooled_spec, tasks):
            if res_ss is not None:
                results_dict[spec_name] = {'second_stage': res_ss, 'first_stage': res_fs}
                print(f"Local Computed [{spec_name}] (Alt {alt})")

    out_dir = POOLED_DIR / f"ALT_{alt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "estimation_results.pkl", 'wb') as f: pickle.dump(results_dict, f)
    
    df['year_quarter'] = df['time_id']
    df, national_phis = calculate_pooled_phis(df, results_dict, state_blocks)
    df.to_csv(out_dir / "market_panel_phis.csv", index=False)
    print(f"Saved results and Phis for Alt {alt} -> {out_dir}")

def run_plotting_phase(alt_list, spec12_only):
    print("\n=== PLOTTING PHASE ===")
    
    for alt in alt_list:
        plot_out_dir = PLOTS_DIR / f"ALT_{alt}"
        plot_out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = POOLED_DIR / f"ALT_{alt}" / "market_panel_phis.csv"
        
        try: df = pd.read_csv(csv_path, low_memory=False)
        except FileNotFoundError: continue
            
        cols_to_plot = [c for c in df.columns if c.startswith('phi_mt_OLS') or c.startswith('phi_mt_IV')]
        if spec12_only: cols_to_plot = [c for c in cols_to_plot if 'Tech' in c]
            
        for col_name in cols_to_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            df_b = df[df['dummy_D_type'] == 0]
            df_d = df[df['dummy_D_type'] == 1]
            
            b_weighted = df_b[col_name] * df_b['market_size']
            b_agg = (b_weighted.groupby(df_b['year_quarter']).sum() / df_b['market_size'].groupby(df_b['year_quarter']).sum()).reset_index(name='phi_b')
            b_agg['date'] = pd.PeriodIndex(b_agg['year_quarter'].str.replace('_', 'Q'), freq='Q').to_timestamp()
            ax.plot(b_agg['date'], b_agg['phi_b'], label=f'B-Type (National Avg) $\hat{{\phi}}$', color='blue', linewidth=2)

            if len(df_d) > 0:
                df_d_active = df_d[df_d['market_size'] > 0]
                if len(df_d_active) > 0:
                    d_weighted = df_d_active[col_name] * df_d_active['market_size']
                    d_agg = (d_weighted.groupby(df_d_active['year_quarter']).sum() / df_d_active['market_size'].groupby(df_d_active['year_quarter']).sum()).reset_index(name='phi_d')
                    d_agg['date'] = pd.PeriodIndex(d_agg['year_quarter'].str.replace('_', 'Q'), freq='Q').to_timestamp()
                    ax.plot(d_agg['date'], d_agg['phi_d'], label=f'D-Type (Digital/National) $\hat{{\phi}}$', color='red', linewidth=2)
            
            ax.set_title(f"Pooled Sleepiness (Alt {alt}): {col_name.replace('phi_mt_', '')}", fontsize=14)
            ax.set_ylabel(r"National $\hat{\phi}_t$")
            ax.grid()
            ax.legend(loc='best')
            fig.tight_layout()
            plt.savefig(plot_out_dir / f"Robustness_{col_name.replace('phi_mt_', '')}.png", dpi=300)
            plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimation 6: Sleepiness Robustness with Coop/State Controls")
    parser.add_argument('--spec12-only', action='store_true', help='Only run spec 12 (Tech x IV_HausmanFull)')
    parser.add_argument('--model-type', choices=['linear', 'logistic', 'both'], default='both', help='Type of NLLS optimization')
    parser.add_argument('--alt', choices=['1', '2', 'both'], default='2', help='Alternative specifications for interactions (1, 2, or both)')
    args = parser.parse_args()

    pd.options.mode.chained_assignment = None
    
    alts_to_run = [1, 2] if args.alt == 'both' else [int(args.alt)]
    
    for current_alt in alts_to_run:
        run_pipeline_for_alt(current_alt, args.model_type, args.spec12_only)
        
    run_plotting_phase(alts_to_run, args.spec12_only)
    print("\n--- Pipeline 6 Completed ---")
