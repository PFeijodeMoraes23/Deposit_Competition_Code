"""
estimation_2_sleep.py
==============================
Robustness check: Estimates the depositor sleepiness function using pooled B and D-type firms.
Extends the analysis by introducing controls and interactions for cooperative ('is_coop')
and state-owned ('is_state_owned') institutions across both Linear and Logistic specifications.

CLI Options:
------------
  --model-type   : linear, logistic, both (default: both)
  --spec12-only  : Flag to only run Spec 12 (Tech x IV_HausmanFull)
"""
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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

try:
    from utils import load_panel_cached
except Exception:
    load_panel_cached = None

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
OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "est5"

POOLED_DIR = OUTPUT_DIR
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

def define_specifications():
    s_base = ['constant', 'dummy_D_type', 'pix_exists']
    
    s_base_selic = ['constant', 'dummy_D_type', 'pix_exists', 'risk_free_qoq_lag', 'dummy_D_type_x_risk_free_qoq_lag']
    s_macro = s_base_selic + ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young']
    # % dummy_D_type_x_fraction_65plus removed: within D-type firms fraction_young + fraction_65plus ≈ const
    # % (corr = -1.000), making both interactions perfectly collinear. The remaining
    # % dummy_D_type_x_fraction_young absorbs the full D-type age effect.
    s_macro += ['dummy_D_type_x_fraction_young']
    s_tech = s_macro + ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']

    base_state_blocks = {'Base': s_base, 'Macro': s_macro, 'Tech': s_tech}
    state_blocks = {}
    
    for key, block in base_state_blocks.items():
        # Dummy intercept + targeted interactions for is_coop and is_state_owned
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
def build_pooled_data():
    df_raw = load_panel_cached(PANEL_CSV) if load_panel_cached else pd.read_csv(PANEL_CSV, dtype={'mca_code': str}, low_memory=False)
    
    df_raw['dummy_D_type'] = (df_raw['CODMUN_IBGE'].astype(str) == '0').astype(float)
    df_raw['pix_exists'] = ((df_raw['year'] > 2020) | ((df_raw['year'] == 2020) & (df_raw['quarter'] == 4))).astype(float)
    
    # Fill defaults for ownership types if missing
    if 'is_coop' in df_raw.columns:
        df_raw['is_coop'] = df_raw['is_coop'].fillna(0.0)
    else:
        df_raw['is_coop'] = 0.0
        
    if 'is_state_owned' in df_raw.columns:
        df_raw['is_state_owned'] = df_raw['is_state_owned'].fillna(0.0)
    else:
        df_raw['is_state_owned'] = 0.0
    
    if 'dep_a1' in df_raw.columns:
        id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
        df_raw = df_raw.drop_duplicates(subset=id_vars)
        df = pd.wide_to_long(df_raw, stubnames=['dep_a', 'spread_a', 'spread_ann_a', 'leave_one_out_mean_spread_a'], i=id_vars, j='deposit_type').reset_index()
        df = df.rename(columns={'dep_a': 'deposit_balance', 'spread_a': 'spread_qoq', 'spread_ann_a': 'spread_ann', 'leave_one_out_mean_spread_a': 'leave_one_out_mean_spread'})
    else:
        df = df_raw.copy()

    if 'deposit_type' in df.columns: df = df[df['deposit_type'] != 3].copy()

    # B-type firms with only national-level k=5 (prepaid) observations: CODMUN_IBGE=0
    # because ESTBAN has no branch-level prepaid data, but these firms are B-type
    # (they have local rows for other deposit types). Do NOT label them as D-type.
    _b_natl_firms = set(df.loc[df['CODMUN_IBGE'].astype(str) != '0', 'CodConglomeradoPrudencial'].unique())
    _k5_natl_mask = (df['deposit_type'] == 5) & (df['CODMUN_IBGE'].astype(str) == '0') & df['CodConglomeradoPrudencial'].isin(_b_natl_firms)
    df.loc[_k5_natl_mask, 'dummy_D_type'] = 0.0

    df['entity_id'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['deposit_type'].astype(str) + "_" + df['mca_code'].astype(str)
    df['time_id'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)
    
    df.sort_values(by=['entity_id', 'year', 'quarter'], inplace=True)
    df['spread_qoq_lag'] = df.groupby('entity_id')['spread_qoq'].shift(1)
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)
    df['lagged_deposits'] = df.groupby('entity_id')['deposit_balance'].shift(1)
    df['nr_lagged_dep'] = (1 + df['risk_free_qoq_lag'] - df['spread_qoq_lag']) * df['lagged_deposits']
    
    df['dummy_D_type_x_risk_free_qoq_lag'] = df['dummy_D_type'] * df['risk_free_qoq_lag']
    # Rescale to pp and center to break near-perfect collinearity with dummy_D_type
    for _c in ('fraction_65plus', 'fraction_young'):
        if _c in df.columns:
            df[_c] = df[_c] * 100.0
            df[_c] = df[_c] - df[_c].mean()
    df['dummy_D_type_x_fraction_65plus'] = df['dummy_D_type'] * df['fraction_65plus']
    df['dummy_D_type_x_fraction_young'] = df['dummy_D_type'] * df['fraction_young']
    
    _, _, state_blocks = define_specifications()
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
    if 'indice_basileia_lag' in df.columns: df['indice_basileia_lag'] *= 100.0

    s_tech_finance = ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young', 'pix_users_pf_per1000', 'connections_per100', 'branches_per1000']
    
    if 'pop_total' in df.columns:
        cols_to_fill = [col for col in s_tech_finance if col in df.columns]
        b_mask = (df['dummy_D_type'] == 0)
        # national_mask: rows with no local market (CODMUN_IBGE=0) — includes D-type digital
        # firms AND B-type firms' k=5 national-level prepaid aggregates. Both receive
        # population-weighted national-average demographics.
        national_mask = (df['CODMUN_IBGE'].astype(str) == '0')
        df_b = df.loc[b_mask & ~national_mask, ['time_id', 'pop_total'] + cols_to_fill].copy()
        col_medians = df_b[cols_to_fill].median()
        for col in cols_to_fill:
            df_b[col] = df_b[col].fillna(col_medians[col])
        df_b['_w'] = df_b['pop_total'].fillna(0)
        w_sum_by_t = df_b.groupby('time_id')['_w'].sum()
        nat_avg_df = {}
        for col in cols_to_fill:
            wv = (df_b[col] * df_b['_w']).groupby(df_b['time_id']).sum()
            nat_avg_df[col] = (wv / w_sum_by_t.replace(0, np.nan)).fillna(col_medians[col])
        natl_time_ids = df.loc[national_mask, 'time_id']
        for col in cols_to_fill:
            df.loc[national_mask, col] = natl_time_ids.map(nat_avg_df[col]).values

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
# (Removed: est5 uses logistic only)

# --- Logistic ---
class NonLinearResults:
    def __init__(self, params, bse, tvalues, pvalues, df_resid, params_native=None, nobs=None, rsquared=None, fvalue=None, f_pvalue=None, G_nominal=None, cov_ame=None, nlls_status=None, nlls_message=None):
        self.params = params
        self.bse = bse
        self.tvalues = tvalues
        self.pvalues = pvalues
        self.df_resid = df_resid
        self.params_native = params_native if params_native is not None else params
        self.G_star = df_resid
        self.nobs = nobs
        self.rsquared = rsquared
        self.fvalue = fvalue
        self.f_pvalue = f_pvalue
        self.G_nominal = G_nominal
        self.cov_ame = cov_ame
        self.nlls_status = nlls_status
        self.nlls_message = nlls_message

    def cov_params(self):
        """Return full AME covariance matrix as a DataFrame (mirrors statsmodels interface)."""
        import pandas as pd
        import numpy as np
        if self.cov_ame is not None:
            return pd.DataFrame(self.cov_ame, index=self.params.index, columns=self.params.index)
        return pd.DataFrame(np.diag(self.bse ** 2), index=self.params.index, columns=self.params.index)

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

def get_nlls_ame_and_se(theta_full_hat, cov_full_hat, X, CF_shape):
    import numpy as np
    
    K = X.shape[1]
    G = CF_shape
    total_len = K + G
    
    AME = np.zeros(total_len)
    
    theta_X = theta_full_hat[:K]
    X_disp = np.clip(np.dot(X, theta_X), -700, 700)
    P_base = 1.0 / (1.0 + np.exp(-X_disp))
    
    for k in range(K):
        col_vals = X[:, k]
        valid_vals = col_vals[~np.isnan(col_vals)]
        if len(valid_vals) == 0: continue
            
        unique_vals = np.unique(valid_vals)
        is_dummy = (len(unique_vals) == 2) and (0.0 in unique_vals) and (1.0 in unique_vals)
        
        if is_dummy:
            X1 = X.copy(); X1[:, k] = 1.0
            P1 = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X1, theta_X), -700, 700)))
            
            X0 = X.copy(); X0[:, k] = 0.0
            P0 = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X0, theta_X), -700, 700)))
            
            AME[k] = np.mean(P1 - P0)
        else:
            dP_dk = P_base * (1.0 - P_base) * theta_X[k]
            AME[k] = np.mean(dP_dk)
            
    if G > 0: AME[K:] = theta_full_hat[K:]
        
    J = np.zeros((total_len, total_len))
    h = 1e-5
    
    for i in range(total_len):
        theta_step = theta_full_hat.copy()
        theta_step[i] += h
        
        ame_step = np.zeros(total_len)
        t_X_step = theta_step[:K]
        X_disp_s = np.clip(np.dot(X, t_X_step), -700, 700)
        P_base_s = 1.0 / (1.0 + np.exp(-X_disp_s))
        
        for k in range(K):
            col_vals = X[:, k]
            valid_vals = col_vals[~np.isnan(col_vals)]
            if len(valid_vals) == 0: continue
            
            unique_vals = np.unique(valid_vals)
            is_dummy = (len(unique_vals) == 2) and (0.0 in unique_vals) and (1.0 in unique_vals)
            
            if is_dummy:
                X1 = X.copy(); X1[:, k] = 1.0
                P1 = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X1, t_X_step), -700, 700)))
                X0 = X.copy(); X0[:, k] = 0.0
                P0 = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X0, t_X_step), -700, 700)))
                ame_step[k] = np.mean(P1 - P0)
            else:
                dP_dk_s = P_base_s * (1.0 - P_base_s) * t_X_step[k]
                ame_step[k] = np.mean(dP_dk_s)
                
        if G > 0: ame_step[K:] = theta_step[K:]
        J[:, i] = (ame_step - AME) / h
        
    cov_AME = J @ cov_full_hat @ J.T
    bse_AME = np.sqrt(np.abs(np.diag(cov_AME)))
    
    return AME, bse_AME, cov_AME


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
    # trf method with cauchy loss down-weights large residuals (robust to outliers).
    # No max_nfev cap — scipy default of 100*(n_params+1) evaluations ensures convergence.
    res_lsq = least_squares(nlls_objective, init_params, args=(y_dm, X, Z, CF, entity_idx), method='trf', loss='cauchy')
    _STATUS_LABELS = {-1: 'budget exhausted', 1: 'gtol', 2: 'ftol', 3: 'xtol', 4: 'ftol+xtol'}
    print(f"  [NLLS] status={res_lsq.status} ({_STATUS_LABELS.get(res_lsq.status, '?')}) | nfev={res_lsq.nfev} | cost={res_lsq.cost:.4g}")
    
    J = res_lsq.jac
    try: cov = np.linalg.pinv(J.T.dot(J)) * (np.sum(res_lsq.fun**2) / (len(y_dm) - len(init_params)))
    except: cov = np.eye(len(init_params))
    
    # Calculate Average Marginal Effects and adjust SEs
    ps_ame, bse, cov_ame = get_nlls_ame_and_se(res_lsq.x, cov, X, CF.shape[1] if has_cf else 0)
    
    idx = [f'interaction_{sv}' if sv != 'constant' else 'nr_lagged_dep' for sv in state_vars] + CF_cols
    ps = pd.Series(index=idx, data=ps_ame)
    bs = pd.Series(index=idx, data=bse)
    tvals = ps / bs
    
    cluster_series = df_ss['CodConglomeradoPrudencial']
    sizes = cluster_series.value_counts()
    G_star = max(1.0, len(sizes) / (1 + (np.std(sizes)/np.mean(sizes))**2 if np.mean(sizes)>0 else 1))
    pvals = pd.Series(stats.t.sf(np.abs(tvals), df=G_star) * 2, index=idx)
    
    nobs = len(y_dm)
    G_nominal = len(sizes)
    tss = np.sum((y_dm - np.mean(y_dm))**2)
    rss = np.sum(res_lsq.fun**2)
    rsquared = 1 - (rss / tss) if tss > 0 else np.nan
    
    k = len(res_lsq.x)
    if tss > 0 and nobs > k and k > 1:
        fvalue = ((tss - rss) / (k - 1)) / (rss / (nobs - k))
        f_pvalue = stats.f.sf(fvalue, k - 1, nobs - k)
    else:
        fvalue, f_pvalue = np.nan, np.nan
    
    ps_native = pd.Series(index=idx, data=res_lsq.x)
    return NonLinearResults(ps, bs, tvals, pvals, G_star, params_native=ps_native,
                            nobs=nobs, rsquared=rsquared, fvalue=fvalue, f_pvalue=f_pvalue, G_nominal=G_nominal,
                            cov_ame=cov_ame, nlls_status=res_lsq.status, nlls_message=res_lsq.message)

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
def exec_pooled_spec(args):
    df, iv_name, iv_cols, s_name, s_cols, model_type = args
    has_cf = len(iv_cols) > 0
    spec_name = f"{iv_name} x {s_name} x {model_type}"

    df_target = df
    res_fs = None
    if has_cf:
        iv_cols_act = [c for c in iv_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        exog_cols_act = [c for c in s_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        if not iv_cols_act: return None, spec_name, None
        df_target, res_fs = run_pooled_first_stage(df_target, iv_cols_act, exog_cols_act)

    if model_type == 'linear':
        res_ss = run_pooled_second_stage_logistic(df_target, s_cols, has_cf=has_cf)  # always logistic
    else:
        res_ss = run_pooled_second_stage_logistic(df_target, s_cols, has_cf=has_cf)
        
    return res_ss, spec_name, res_fs

def calculate_pooled_phis(df, res_dict, state_blocks):
    phi_results = {}
    # Use population as market-size weight for phi aggregation.
    # Under the constant-fraction assumption (M_mt = c * pop_mt), c cancels
    # in the ratio Σ(phi * M) / Σ(M), making pop_total the correct weight.
    if 'pop_total' in df.columns:
        df['market_size'] = df['pop_total'].fillna(0)
    else:
        df['market_size'] = 1.0
        
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
            if hasattr(ss_res, 'params_native'):
                param_dict = ss_res.params_native
            else:
                param_dict = ss_res.params
            if col_name in param_dict:
                c = param_dict[col_name]
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

def run_pipeline(spec12_only):
    print(f"\n=== ESTIMATION 5 ===")
    df = build_pooled_data()
    _, iv_specs, state_blocks = define_specifications()
    
    tasks = []
    if spec12_only:
        tasks.append((df, 'IV_HausmanFull', iv_specs['IV_HausmanFull'], 'Tech', state_blocks['Tech'], 'logistic'))
    else:
        tasks.extend([(df, iv_name, iv_specs[iv_name], s_name, state_blocks[s_name], 'logistic')
                      for s_name in state_blocks.keys() for iv_name in ['OLS', 'IV_CostShifters', 'IV_Wholesale', 'IV_HausmanFull']])
    
    from joblib import Parallel, delayed
    results_dict = {}
    
    # Process 4 models at a time to stay deep within 32GB bounds while crushing latency
    _nw = max(1, (os.cpu_count() or 4) // max(1, int(os.environ.get('SLEEP_PIPELINE_NSLOTS', '1'))))
    results = Parallel(n_jobs=min(4, _nw))(delayed(exec_pooled_spec)(t) for t in tasks)
    
    for res_ss, spec_name, res_fs in results:
        if res_ss is not None:
            results_dict[spec_name] = {'second_stage': res_ss, 'first_stage': res_fs}
            print(f"Local Computed [{spec_name}]")

    POOLED_DIR.mkdir(parents=True, exist_ok=True)
    with open(POOLED_DIR / "estimation_results.pkl", 'wb') as f: pickle.dump(results_dict, f)
    
    df['year_quarter'] = df['time_id']
    df, national_phis = calculate_pooled_phis(df, results_dict, state_blocks)
    df.to_csv(POOLED_DIR / "market_panel_phis.csv", index=False)
    print(f"Saved results and Phis -> {POOLED_DIR}")

def run_plotting_phase(spec12_only):
    print("\n=== PLOTTING PHASE ===")
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = POOLED_DIR / "market_panel_phis.csv"
    
    try: df = pd.read_csv(csv_path, low_memory=False)
    except FileNotFoundError: return
        
    cols_to_plot = [c for c in df.columns if c.startswith('phi_mt_OLS') or c.startswith('phi_mt_IV')]
    if spec12_only: cols_to_plot = [c for c in cols_to_plot if 'Tech' in c]
        
    for col_name in cols_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def get_agg_with_se(d_sub, col):
            import scipy.stats as stats
            if len(d_sub) == 0: return pd.DataFrame()
            w = d_sub['market_size']
            num = (d_sub[col] * w).groupby(d_sub['year_quarter']).sum()
            den = w.groupby(d_sub['year_quarter']).sum()
            mean = num / den
            
            merged = d_sub[['year_quarter', col]].copy()
            merged['w'] = w
            merged['mean'] = merged['year_quarter'].map(mean)
            
            var_num = (merged['w'] * (merged[col] - merged['mean'])**2).groupby(merged['year_quarter']).sum()
            v1 = merged['w'].groupby(merged['year_quarter']).sum()
            v2 = (merged['w']**2).groupby(merged['year_quarter']).sum()
            
            var = var_num / (v1 - (v2 / v1))
            
            # Use effective sample size for n
            n_eff = (v1**2) / v2
            
            se = (var / n_eff).apply(lambda x: x**0.5 if pd.notnull(x) and x > 0 else 0.0)
            
            # Compute critical value from t-distribution based on effective df
            df_res = pd.DataFrame({'phi': mean, 'se': se, 'n_eff': n_eff}).reset_index()
            df_res['cv'] = df_res['n_eff'].apply(lambda n: stats.t.ppf(0.975, max(1, n - 1)) if pd.notnull(n) and n > 1 else 1.96)
            
            df_res['date'] = pd.PeriodIndex(df_res['year_quarter'].str.replace('_', 'Q'), freq='Q').to_timestamp()
            return df_res

        df_b = df[df['dummy_D_type'] == 0]
        df_d = df[df['dummy_D_type'] == 1]
        
        b_agg = get_agg_with_se(df_b, col_name)
        if not b_agg.empty:
            ax.plot(b_agg['date'], b_agg['phi'], label=f'B-Type (National Avg) $\hat{{\phi}}$', color='blue', linewidth=2)
            ax.fill_between(b_agg['date'], b_agg['phi'] - b_agg['cv']*b_agg['se'], b_agg['phi'] + b_agg['cv']*b_agg['se'], color='blue', alpha=0.15)
            
        if len(df_d) > 0:
            df_d_active = df_d[df_d['market_size'] > 0]
            d_agg = get_agg_with_se(df_d_active, col_name)
            if not d_agg.empty:
                ax.plot(d_agg['date'], d_agg['phi'], label=f'D-Type (Digital/National) $\hat{{\phi}}$', color='red', linewidth=2)
                ax.fill_between(d_agg['date'], d_agg['phi'] - d_agg['cv']*d_agg['se'], d_agg['phi'] + d_agg['cv']*d_agg['se'], color='red', alpha=0.15)
        
        ax.set_ylabel(r"National $\hat{\phi}_t$")
        ax.set_ylim(bottom=0)
        ax.grid()
        ax.legend(loc='best')
        fig.tight_layout()
        plt.savefig(PLOTS_DIR / f"Robustness_{col_name.replace('phi_mt_', '')}.png", dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimation 5: Sleepiness Robustness with Coop/State Controls (Logistic)")
    parser.add_argument('--spec12-only', action='store_true', help='Only run spec 12 (Tech x IV_HausmanFull)')
    args = parser.parse_args()

    pd.options.mode.chained_assignment = None
    
    run_pipeline(args.spec12_only)
    run_plotting_phase(args.spec12_only)
    print("\n--- Pipeline 5 Completed ---")




