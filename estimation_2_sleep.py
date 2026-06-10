"""
estimation_2_sleep.py
==============================
Strategy 2: Estimates the depositor sleepiness function using pooled B and D-type firms.
D-type firms receive national population-weighted average state variables.
No bank-type dummies, no D-type interactions, no endogenous variables (branches, pix volume).
CF correction: v_hat x lagged_deposits (linear in residual, interacted with lagged deposits).

Outputs are directed to ESTIMATION_OUTPUT/DEMAND_PREP/est2

CLI Options:
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
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")

# ==============================================================================
# GLOBAL SETUP
# ==============================================================================
from utils import paths as _paths_mod
DATA_DIR = _paths_mod.PROCESSED
_PANEL_WITH_FEES = DATA_DIR / "market_panel_with_fees.csv"
PANEL_CSV = _PANEL_WITH_FEES if _PANEL_WITH_FEES.exists() else DATA_DIR / "market_panel.csv"
OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "est2"

PLOTS_DIR = OUTPUT_DIR / "PLOTS"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    s_base = ['constant', 'pix_exists']
    s_macro = s_base + ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young', 'risk_free_qoq_lag']
    s_tech = s_macro + ['connections_per100']

    iv_specs = {
        'OLS': [],
        'IV_CostShifters': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag'],
        'IV_Wholesale': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag'],
        'IV_HausmanFull': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag', 'leave_one_out_mean_spread'],
    }
    state_blocks = {'Base': s_base, 'Macro': s_macro, 'Tech': s_tech}
    return s_tech, iv_specs, state_blocks

# ==============================================================================
# DATA BUILDING
# ==============================================================================
def build_pooled_data():
    df_raw = load_panel_cached(PANEL_CSV) if load_panel_cached else pd.read_csv(PANEL_CSV, dtype={'mca_code': str}, low_memory=False)
    df_raw = df_raw.copy()  # defragment: market_panel_with_fees has many columns from merges

    df_raw['pix_exists'] = ((df_raw['year'] > 2020) | ((df_raw['year'] == 2020) & (df_raw['quarter'] == 4))).astype(float)

    if 'dep_a1' in df_raw.columns:
        id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
        df_raw = df_raw.drop_duplicates(subset=id_vars)
        df = pd.wide_to_long(df_raw, stubnames=['dep_a', 'spread_a', 'spread_ann_a', 'leave_one_out_mean_spread_a'], i=id_vars, j='deposit_type').reset_index()
        df = df.rename(columns={'dep_a': 'deposit_balance', 'spread_a': 'spread_qoq', 'spread_ann_a': 'spread_ann', 'leave_one_out_mean_spread_a': 'leave_one_out_mean_spread'})
    else:
        df = df_raw.copy()

    if 'deposit_type' in df.columns:
        df = df[df['deposit_type'] != 3].copy()

    df['entity_id'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['deposit_type'].astype(str) + "_" + df['mca_code'].astype(str)
    df['time_id'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)

    df.sort_values(by=['entity_id', 'year', 'quarter'], inplace=True)
    df['spread_qoq_lag'] = df.groupby('entity_id')['spread_qoq'].shift(1)
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)
    df['lagged_deposits'] = df.groupby('entity_id')['deposit_balance'].shift(1)
    df['nr_lagged_dep'] = (1 + df['risk_free_qoq_lag'] - df['spread_qoq_lag']) * df['lagged_deposits']

    if 'leave_one_out_mean_spread' not in df.columns:
        df['leave_one_out_mean_spread'] = np.nan

    df['bank_year'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['year'].astype(str)
    df = df.dropna(subset=['deposit_balance', 'nr_lagged_dep', 'spread_qoq', 'entity_id', 'time_id'])

    df['constant'] = 1.0
    scale_cols = {'deposit_balance': 1e9, 'nr_lagged_dep': 1e9, 'lagged_deposits': 1e9}
    for col, factor in scale_cols.items():
        if col in df.columns: df[col] /= factor

    if 'gdp_per_capita' in df.columns: df['gdp_per_capita'] /= 10000.0
    if 'cadunico_families_per1000' in df.columns: df['cadunico_families_per1000'] /= 100.0
    if 'connections_per100' in df.columns: df['connections_per100'] /= 100.0
    if 'indice_basileia_lag' in df.columns: df['indice_basileia_lag'] *= 100.0

    state_vars_to_fill = ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young', 'connections_per100']

    # Assign national population-weighted averages to D-type (CODMUN_IBGE == '0') rows.
    # This includes both true digital banks and B-type prepaid (k=5) national aggregates.
    if 'pop_total' in df.columns:
        cols_to_fill = [col for col in state_vars_to_fill if col in df.columns]
        b_mask = (df['CODMUN_IBGE'].astype(str) != '0')
        national_mask = (df['CODMUN_IBGE'].astype(str) == '0')
        df_b = df.loc[b_mask, ['time_id', 'pop_total'] + cols_to_fill].copy()
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

    for col in state_vars_to_fill:
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
        df['v_hat_x_lagged_dep'] = 0.0
        return df, None

    mod = sm.OLS(df_fs['spread_qoq'], sm.add_constant(df_fs[first_stage_vars]))
    cluster_series = df_fs['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    res = apply_imbalanced_cluster_correction(res, cluster_series)

    df['v_hat'] = 0.0
    df.loc[valid_mask, 'v_hat'] = res.resid
    df['v_hat_x_lagged_dep'] = df['v_hat'] * df['lagged_deposits']
    return df, res

def run_pooled_second_stage(df, state_vars, has_cf=False):
    X_cols = []
    for sv in state_vars:
        col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
        df[col_name] = df[sv] * df['nr_lagged_dep'] if sv != 'constant' else df['nr_lagged_dep']
        X_cols.append(col_name)
    if has_cf: X_cols.append('v_hat_x_lagged_dep')

    df_ss = df.dropna(subset=X_cols + ['deposit_balance']).copy()
    if len(df_ss) == 0: return None

    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance']
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')

    mod = sm.OLS(y_dm, X_dm)
    cluster_series = df_ss['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    return apply_imbalanced_cluster_correction(res, cluster_series)

def exec_pooled_spec(args):
    df, iv_name, iv_cols, s_name, s_cols = args
    has_cf = len(iv_cols) > 0
    spec_name = f"{iv_name} x {s_name}"

    df_target = df
    res_fs = None
    if has_cf:
        iv_cols_act = [c for c in iv_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        exog_cols_act = [c for c in s_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        if not iv_cols_act: return None, spec_name, None
        df_target, res_fs = run_pooled_first_stage(df_target, iv_cols_act, exog_cols_act)

    res_ss = run_pooled_second_stage(df_target, s_cols, has_cf=has_cf)
    return res_ss, spec_name, res_fs

def calculate_pooled_phis(df, res_dict, state_blocks):
    phi_results = {}
    if 'pop_total' in df.columns:
        df['market_size'] = df['pop_total'].fillna(0)
    else:
        df['market_size'] = 1.0

    for model_key, res_item in res_dict.items():
        if res_item['second_stage'] is None: continue

        try:
            m_type, spec_name = model_key.split(' x ')
        except ValueError: continue

        s_cols = state_blocks.get(spec_name, [])
        if not s_cols: continue

        ss_res = res_item['second_stage']
        phi_mt = np.zeros(len(df))
        for sv in s_cols:
            col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
            if col_name in ss_res.params:
                c = ss_res.params[col_name]
                phi_mt += c if sv == 'constant' else c * df[sv].fillna(0)

        safe_key = model_key.replace(' ', '_').replace('.', '')
        df[f'phi_mt_{safe_key}'] = phi_mt
        market_agg = df.groupby(['year_quarter', 'CODMUN_IBGE'], observed=True).agg(
            phi_mt=(f'phi_mt_{safe_key}', 'mean'), M_mt=('market_size', 'sum')).reset_index()
        weighted_phi = market_agg['phi_mt'] * market_agg['M_mt']
        national_agg = (weighted_phi.groupby(market_agg['year_quarter']).sum() /
                        market_agg['M_mt'].groupby(market_agg['year_quarter']).sum().replace(0, np.nan)
                        ).fillna(0).reset_index(name=f'phi_t_{safe_key}')
        phi_results[safe_key] = national_agg
    return df, phi_results

# ==============================================================================
# PIPELINE
# ==============================================================================
def run_pooled_phase(spec12_only=False):
    print("=== PHASE 1: POOLED B+D ESTIMATION ===")
    df = build_pooled_data()
    _, iv_specs, state_blocks = define_specifications()

    if spec12_only:
        tasks = [(df, 'IV_HausmanFull', iv_specs['IV_HausmanFull'], 'Tech', state_blocks['Tech'])]
    else:
        tasks = [(df, iv_name, iv_specs[iv_name], s_name, state_blocks[s_name])
                 for s_name in state_blocks.keys()
                 for iv_name in ['OLS', 'IV_CostShifters', 'IV_Wholesale', 'IV_HausmanFull']]

    results_dict = {}
    _nw = min(4, max(1, (os.cpu_count() or 4) // max(1, int(os.environ.get('SLEEP_PIPELINE_NSLOTS', '1')))))
    with concurrent.futures.ProcessPoolExecutor(max_workers=_nw) as executor:
        for res_ss, spec_name, res_fs in executor.map(exec_pooled_spec, tasks):
            if res_ss is not None:
                results_dict[spec_name] = {'second_stage': res_ss, 'first_stage': res_fs}
                print(f"Computed: {spec_name}")

    with open(OUTPUT_DIR / "estimation_results.pkl", 'wb') as f: pickle.dump(results_dict, f)

    df['year_quarter'] = df['time_id']
    df, national_phis = calculate_pooled_phis(df, results_dict, state_blocks)
    df.to_csv(OUTPUT_DIR / "market_panel_phis.csv", index=False)

    from functools import reduce
    if national_phis:
        agg_df = reduce(lambda left, right: pd.merge(left, right, on='year_quarter', how='outer'), national_phis.values())
        agg_df.to_csv(OUTPUT_DIR / "national_phi_t.csv", index=False)
    print("Saved Pooled B+D Pickles and Phis.\n")

def run_plotting_phase(spec12_only=False):
    print("=== PHASE 2: PLOTTING ===")
    try:
        df = pd.read_csv(OUTPUT_DIR / "market_panel_phis.csv", low_memory=False)
    except FileNotFoundError:
        print("Run phase 1 first.")
        return

    cols_to_plot = [c for c in df.columns if c.startswith('phi_mt_OLS') or c.startswith('phi_mt_IV')]
    if spec12_only:
        cols_to_plot = [c for c in cols_to_plot if 'Tech' in c]

    if 'market_size' not in df.columns:
        df['market_size'] = 1.0

    for col_name in cols_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        w = df['market_size']
        num = (df[col_name] * w).groupby(df['year_quarter']).sum()
        den = w.groupby(df['year_quarter']).sum()
        phi_agg = (num / den.replace(0, np.nan)).fillna(0).reset_index()
        phi_agg.columns = ['year_quarter', 'phi']
        phi_agg['date'] = pd.PeriodIndex(phi_agg['year_quarter'].str.replace('_', 'Q'), freq='Q').to_timestamp()
        phi_agg = phi_agg.sort_values('date')
        ax.plot(phi_agg['date'], phi_agg['phi'], color='blue', linewidth=2)
        ax.set_title(f"Pooled B+D Sleepiness: {col_name.replace('phi_mt_', '')}", fontsize=14)
        ax.set_ylabel(r"National $\hat{\phi}_t$")
        ax.set_ylim(bottom=0)
        ax.grid()
        fig.tight_layout()
        plt.savefig(PLOTS_DIR / f"Pooled_{col_name.replace('phi_mt_', '')}.png", dpi=300)
        plt.close(fig)
    print("Plots generated.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimation 2: Pooled B+D Sleepiness (Linear)")
    parser.add_argument('--spec12', action='store_true', help='Only run spec 12 (Tech x IV_HausmanFull)')
    args = parser.parse_args()

    pd.options.mode.chained_assignment = None
    run_pooled_phase(spec12_only=args.spec12)
    run_plotting_phase(spec12_only=args.spec12)
