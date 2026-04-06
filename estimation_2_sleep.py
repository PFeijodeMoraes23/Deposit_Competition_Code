"""
estimation_2_sleep.py
==============================
Unified script for direct time-series estimation of the sleepiness parameter phi_t for National (D) digital/wholesale banks,
followed immediately by the generation of the corresponding specification plots.

Usage
-----
  python estimation_2_sleep.py --run-all --all-options

CLI Flags
---------
  Estimation Flags:
  --option {1|2|3}    Run specifically one of the options (1: Brute Force, 2: X_j Interactions, 3: PCA Index).
  --run-all           Loop over all three regression options sequentially.
  
  Plotting Flags:
  --all-options       Generate single plot with all 4 model variants superimposed.
  --default-only      Generate single plot with Local + Opt 3 PCA.
  
  Pipeline Control:
  --skip-estimation   Skip the estimation phase, only generate plots using existing pickle outputs.
  --skip-plots        Skip the plotting phase, only generate regression tables and pickles.
"""
import argparse
from pathlib import Path
import sys
import io
import pickle
import warnings
from contextlib import suppress

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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")


# ==============================================================================
# ESTIMATION PHASE
# ==============================================================================

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

def _resolve_estimation_paths():
    _ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
    PANEL_CSV = DATA_DIR / "market_panel.csv"
    OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "SLEEPINESS_NATIONAL"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return PANEL_CSV, OUTPUT_DIR

def demean_variables(df, cols, entity_col):
    df_out = df.copy()
    for col in cols:
        df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
    df_out[cols] = df_out[cols] - df_out.groupby(entity_col)[cols].transform('mean')
    return df_out

def build_estimation_data():
    panel_csv, _ = _resolve_estimation_paths()
    df_raw = pd.read_csv(panel_csv, dtype={'year_quarter': str, 'CodIbge': str})
    df = df_raw[df_raw['Source'] == 'IFDATA'].copy()
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
        df = df[df['Source'] == 'IFDATA'].copy()
    
    if 'deposit_type' in df.columns:
        df = df[~df['deposit_type'].astype(str).str.contains('3')].copy()
    df['entity_id'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['deposit_type'].astype(str)
    
    df = df.sort_values(by=['entity_id', 'year', 'quarter'])

    if 'year_quarter' not in df.columns:
        df['year_quarter'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)

    df['bank_year'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['year'].astype(str)

    df['lagged_deposits'] = df.groupby('entity_id')['deposit_balance'].shift(1)
    df['spread_qoq_lag'] = df.groupby('entity_id')['spread_qoq'].shift(1)
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)

    df['nr_lagged_dep'] = (1 + df['risk_free_qoq_lag'] - df['spread_qoq_lag']) * df['lagged_deposits']

    if 'gdp_per_capita' in df.columns: df['gdp_per_capita'] /= 10000.0
    if 'cadunico_families_per1000' in df.columns: df['cadunico_families_per1000'] /= 100.0
    if 'pix_users_pf_per1000' in df.columns: df['pix_users_pf_per1000'] /= 100.0
    if 'connections_per100' in df.columns: df['connections_per100'] /= 100.0
    if 'deposit_balance' in df.columns:
        df['deposit_balance'] /= 1e9
        df['nr_lagged_dep'] /= 1e9
        df['lagged_deposits'] /= 1e9
        
    df['constant'] = 1.0
    df['post_2020'] = (df['year'] >= 2020).astype(int)
    return df

def run_model_option1(df, state_vars, has_cf=False):
    X_cols = []
    for sv in state_vars:
        col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
        if sv != 'constant': df[col_name] = df[sv] * df['nr_lagged_dep']
        X_cols.append(col_name)

    if has_cf: X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])

    df_ss = df.dropna(subset=X_cols + ['deposit_balance']).copy()
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance'].astype(float)
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')[X_cols].astype(float)

    mod = sm.OLS(y_dm, X_dm)
    cluster_series = df_ss['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    return apply_imbalanced_cluster_correction(res, cluster_series)

def run_model_option2(df, state_vars, has_cf=False):
    X_cols = []
    df['log_total_assets_lag'] = df['log_total_assets_lag'].fillna(df['log_total_assets_lag'].median())
    
    for sv in state_vars:
        if sv == 'constant':
            col_name = "nr_lagged_dep"
            X_cols.append(col_name)
        else:
            col_name_base = f"interaction_{sv}"
            col_name_cross = f"interaction_{sv}_x_assets"
            df[col_name_base] = df[sv] * df['nr_lagged_dep']
            df[col_name_cross] = df[sv] * df['log_total_assets_lag'] * df['nr_lagged_dep']
            X_cols.extend([col_name_base, col_name_cross])

    if has_cf: X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])

    df_ss = df.dropna(subset=X_cols + ['deposit_balance']).copy()
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance'].astype(float)
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')[X_cols].astype(float)

    mod = sm.OLS(y_dm, X_dm)
    cluster_series = df_ss['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    return apply_imbalanced_cluster_correction(res, cluster_series)

def run_model_option3(df, state_vars, has_cf=False):
    sv_dynamic = [sv for sv in state_vars if sv != 'constant']
    if not sv_dynamic: return run_model_option1(df, state_vars, has_cf)
        
    df_pca = df.dropna(subset=sv_dynamic).copy()
    pca = PCA(n_components=1)
    ts_df = df_pca[['year_quarter'] + sv_dynamic].drop_duplicates().sort_values('year_quarter')
    ts_vals = ts_df[sv_dynamic].values
    
    ts_mean = np.mean(ts_vals, axis=0)
    ts_std = np.std(ts_vals, axis=0)
    ts_std[ts_std == 0] = 1 
    
    ts_norm = (ts_vals - ts_mean) / ts_std
    ts_pca = pca.fit_transform(ts_norm)
    ts_df['pca_macro_tech_index'] = ts_pca[:, 0]
    
    df_ss = df.merge(ts_df[['year_quarter', 'pca_macro_tech_index']], on='year_quarter', how='inner')
    df_ss['interaction_pca_index'] = df_ss['pca_macro_tech_index'] * df_ss['nr_lagged_dep']
    X_cols = ['nr_lagged_dep', 'interaction_pca_index']
    
    if has_cf: X_cols.extend(['v_hat', 'v_hat_2', 'v_hat_3'])
        
    df_ss = df_ss.dropna(subset=X_cols + ['deposit_balance']).copy()
    y_dm = demean_variables(df_ss, ['deposit_balance'], 'entity_id')['deposit_balance'].astype(float)
    X_dm = demean_variables(df_ss, X_cols, 'entity_id')[X_cols].astype(float)

    mod = sm.OLS(y_dm, X_dm)
    cluster_series = df_ss['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    return apply_imbalanced_cluster_correction(res, cluster_series)

def first_stage_cf(df, spec_instruments, exogenous_controls):
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
    valid_mask_all = df['v_hat'].notnull()
    df.loc[valid_mask_all, 'v_hat_2'] = df.loc[valid_mask_all, 'v_hat'] ** 2
    df.loc[valid_mask_all, 'v_hat_3'] = df.loc[valid_mask_all, 'v_hat'] ** 3
    df.loc[~valid_mask_all, ['v_hat_2', 'v_hat_3']] = 0.0
    return df, res

def do_estimation(args):
    print("=== ESTIMATION PHASE ===")
    _, output_dir = _resolve_estimation_paths()
    df = build_estimation_data()
    print(f"Panel size for IFDATA D banks: {len(df)} rows")
    
    s_base = ['constant', 'post_2020']
    s_macro = s_base + ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young', 'risk_free_qoq_lag']
    s_tech = s_macro + ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']
    s_tech = [c for c in s_tech if c in df.columns] 

    state_blocks = {'Base': s_base, 'Macro': s_macro, 'Tech': s_tech}
    iv_specs = {
        'OLS': [],
        'IV_CostShifters': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag'],
        'IV_Wholesale': ['wholesale_ratio_lag', 'indice_basileia_lag', 'leave_one_out_mean_spread'],
        'IV_HausmanFull': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag', 'leave_one_out_mean_spread']
    }
    options_to_run = [args.option] if args.option else ([1, 2, 3] if args.run_all else [])
    
    if not options_to_run:
        print("No estimation options specified. Pass --run-all or --option {1,2,3}.")
        return

    import concurrent.futures
    tasks = []
    
    def process_task(opt, b_name, s_cols, iv_name, iv_cols):
        exog_cols_act = [c for c in s_cols if c in df.columns and df[c].notnull().sum() > 0]
        iv_cols_act = [c for c in iv_cols if c in df.columns and df[c].notnull().sum() > 0]

        if iv_cols_act:
            df_target, res_fs = first_stage_cf(df.copy(), iv_cols_act, exog_cols_act)
            has_cf = True
        else:
            df_target = df.copy()
            res_fs = None
            has_cf = False

        if opt == 1: res_ss = run_model_option1(df_target, exog_cols_act, has_cf)
        elif opt == 2: res_ss = run_model_option2(df_target, exog_cols_act, has_cf)
        elif opt == 3: res_ss = run_model_option3(df_target, exog_cols_act, has_cf)
        else: res_ss = None

        return opt, b_name, iv_name, res_ss, res_fs

    results_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for opt in options_to_run:
            for b_name, s_cols in state_blocks.items():
                for iv_name, iv_cols in iv_specs.items():
                    tasks.append(executor.submit(process_task, opt, b_name, s_cols, iv_name, iv_cols))

        for future in concurrent.futures.as_completed(tasks):
            opt, b_name, iv_name, res_ss, res_fs = future.result()
            print(f"Processed OPTION {opt} - Specification: {iv_name} x {b_name} ")

            spec_name = f"Option_{opt}_{iv_name}_{b_name}"
            results_dict[spec_name] = {'option': opt, 'block': b_name, 'second_stage': res_ss, 'first_stage': res_fs}

            if res_ss is not None:
                safe_name = f"Option_{opt}_{iv_name}_" + b_name.replace(" ", "_").replace("/", "").replace(":", "")
                tex_file = output_dir / f"{safe_name}.tex"
                with open(tex_file, 'w') as f: f.write(res_ss.summary().as_latex())

                if res_fs is not None:
                    tex_file_fs = output_dir / f"{safe_name}_FirstStage.tex"    
                    with open(tex_file_fs, 'w') as f: f.write(res_fs.summary().as_latex())
    
    results_pickle = output_dir / "estimation_results.pkl"
    with open(results_pickle, 'wb') as f:
        pickle.dump(results_dict, f)
        
    print(f"\nEstimation outputs successfully saved in: {output_dir}")        


# ==============================================================================
# PLOTTING PHASE
# ==============================================================================

def resolve_plot_paths():
    _ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
    PANEL_CSV = DATA_DIR / "market_panel.csv"
    
    LOCAL_RESULTS_PICKLE = DATA_DIR / "ESTIMATION_OUTPUT" / "SLEEPINESS_NEW" / "estimation_results.pkl"
    NATIONAL_RESULTS_PICKLE = DATA_DIR / "ESTIMATION_OUTPUT" / "SLEEPINESS_NATIONAL" / "estimation_results.pkl"
    
    OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "PLOTS"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return PANEL_CSV, LOCAL_RESULTS_PICKLE, NATIONAL_RESULTS_PICKLE, OUTPUT_DIR

def build_plot_data(panel_csv):
    s_macro = ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young']
    s_tech = ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']
    use_cols = ['year', 'quarter', 'year_quarter', 'lagged_deposits', 'deposit_balance', 'log_total_assets_lag', 
                'risk_free_qoq', 'CodConglomeradoPrudencial', 'mca_code', 'deposit_type'] + s_macro + s_tech
    
    with suppress(Exception):
        sample = pd.read_csv(panel_csv, nrows=5)
        use_cols = [c for c in use_cols if c in sample.columns]
        
    df_raw = pd.read_csv(panel_csv, usecols=use_cols, low_memory=False)

    if 'year_quarter' not in df_raw.columns and 'year' in df_raw.columns:
        df_raw['year_quarter'] = df_raw['year'].astype(int).astype(str) + "Q" + df_raw['quarter'].astype(int).astype(str)

    scale_cols = {'gdp_per_capita': 10000.0, 'cadunico_families_per1000': 100.0, 'pix_users_pf_per1000': 100.0, 'connections_per100': 100.0}
    for col, factor in scale_cols.items():
        if col in df_raw.columns: df_raw[col] /= factor
            
    df_raw['constant'] = 1.0
    if 'year' in df_raw.columns:
        df_raw['post_2020'] = (df_raw['year'] >= 2020).astype(int)
    else:
        df_raw['post_2020'] = (df_raw['year_quarter'].str.extract('(\d{4})')[0].astype(float) >= 2020).astype(int)

    if 'lagged_deposits' in df_raw.columns: df_raw['market_size'] = df_raw['lagged_deposits']
    elif 'deposit_balance' in df_raw.columns: df_raw['market_size'] = df_raw['deposit_balance']
    else: df_raw['market_size'] = 1.0
        
    df_raw['log_total_assets_lag'] = df_raw['log_total_assets_lag'].fillna(df_raw['log_total_assets_lag'].median())
    
    if 'mca_code' in df_raw.columns: df_raw['entity_id'] = df_raw['CodConglomeradoPrudencial'].astype(str) + "_" + df_raw['mca_code'].astype(str)
    else: df_raw['entity_id'] = df_raw['CodConglomeradoPrudencial'].astype(str)
    
    df_raw = df_raw.sort_values(by=['entity_id', 'year', 'quarter'])
    df_raw['risk_free_qoq_lag'] = df_raw.groupby('entity_id')['risk_free_qoq'].shift(1)

    return df_raw

def get_pca_index(df_q, s_cols):
    sv_dynamic = [sv for sv in s_cols if sv != 'constant']
    if not sv_dynamic: return None
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
    return ts_df.set_index('year_quarter')['pca_index'].to_dict()

def predict_phi(params, cov_params, param_vector):
    if params is None or cov_params is None: return np.nan, np.nan
    p_names = params.index
    V = np.zeros(len(p_names))
    for i, name in enumerate(p_names):
        if name in param_vector: V[i] = param_vector[name]
    
    est = V.dot(params)
    var = V.dot(cov_params).dot(V)
    return est, np.sqrt(var)

def get_models_for_spec(iv_name, s_name, loc_res, nat_res):
    loc_key = f"{iv_name} x {s_name}"
    def ext(m): return (m.params, m.cov_params()) if m is not None else (None, None)
    return {
        'loc': ext(loc_res.get(loc_key, {}).get('second_stage', None)),
        'n1': ext(nat_res.get(f"Option_1_{iv_name}_{s_name}", {}).get('second_stage', None)),
        'n2': ext(nat_res.get(f"Option_2_{iv_name}_{s_name}", {}).get('second_stage', None)),
        'n3': ext(nat_res.get(f"Option_3_{iv_name}_{s_name}", {}).get('second_stage', None))
    }

def get_base_vector(s_cols, s_vals):
    vec = {}
    for sv in s_cols:
        nm = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
        vec[nm] = s_vals[sv]
    return vec

def generate_phi_plot(res_df, iv_name, s_name, spec_number, out_dir, all_options=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    if all_options:
        configs = [
            ('loc', r'Local Implied $\hat{\phi}_t^{\mathrm{Loc}}$', 'blue'),
            ('n1', r'Direct National $\hat{\phi}_t^{\mathrm{Nat},1}$', 'green'),
            ('n2', r'Direct National $\hat{\phi}_t^{\mathrm{Nat},2}$', 'orange'),
            ('n3', r'Direct National $\hat{\phi}_t^{\mathrm{Nat},3}$', 'purple')
        ]
        suffix = "_All"
    else:
        configs = [
            ('loc', r'Local Implied $\hat{\phi}_t^{\mathrm{Loc}}$', 'blue'),
            ('n3', r'Direct National $\hat{\phi}_t^{\mathrm{Nat},3}$', 'purple')
        ]
        suffix = ""

    plotted = False
    for pfx, label, color in configs:
        est = res_df[f'{pfx}_est']
        se = res_df[f'{pfx}_se']
        if est.notnull().sum() > 0:
            plotted = True
            ax.plot(res_df['date'], est, label=label, color=color, linewidth=2)
            ax.fill_between(res_df['date'], est - 1.96*se, est + 1.96*se, color=color, alpha=0.15)

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(f"Specification {spec_number:02d}: {iv_name} x {s_name}", fontsize=14)
    ax.set_ylabel(r"National $\hat{\phi}_t^{\mathrm{Nat}}$")
    ax.set_xlabel("Year-Quarter")
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()
    safe_name = f"Spec{spec_number:02d}_{iv_name}_{s_name}{suffix}".replace(" ", "_").replace("/", "")
    plt.savefig(out_dir / f"{safe_name}.png", dpi=300)
    plt.close(fig)

def do_plotting(args):
    print("\n=== PLOTTING PHASE ===")
    gen_all = args.all_options or not args.default_only
    gen_default = args.default_only or not args.all_options
    
    panel_csv, local_pkl, nat_pkl, out_dir = resolve_plot_paths()
    df = build_plot_data(panel_csv)
    
    try:
        with open(local_pkl, 'rb') as f: loc_res = pickle.load(f)
    except FileNotFoundError:
        print(f"[Warning] Could not find {local_pkl}. Plots might lack Local estimates.")
        loc_res = {}
        
    try:
        with open(nat_pkl, 'rb') as f: nat_res = pickle.load(f)
    except FileNotFoundError:
        print(f"[Warning] Could not find {nat_pkl}. Plots might lack National estimates.")
        nat_res = {}
        
    yq_list = sorted(df['year_quarter'].dropna().unique())
    
    s_base = ['constant', 'post_2020']
    s_macro = s_base + ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young', 'risk_free_qoq_lag']
    s_tech = s_macro + ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']
    s_tech = [c for c in s_tech if c in df.columns]
    state_blocks = {'Base': s_base, 'Macro': s_macro, 'Tech': s_tech}
    
    assets_agg = {}
    for yq in yq_list:
        sub = df[df['year_quarter'] == yq]
        m_tot = sub['market_size'].sum()
        if m_tot > 0: assets_agg[yq] = (sub['log_total_assets_lag'] * sub['market_size']).sum() / m_tot
        else: assets_agg[yq] = 0

    pca_dict_map = {
        'Base': get_pca_index(df, state_blocks['Base']),
        'Macro': get_pca_index(df, state_blocks['Macro']),
        'Tech': get_pca_index(df, state_blocks['Tech'])
    }

    iv_order = ['OLS', 'IV_CostShifters', 'IV_Wholesale', 'IV_HausmanFull']
    state_order = ['Base', 'Macro', 'Tech']
    
    spec_number = 1
    for s_name in state_order:
        s_cols = state_blocks[s_name]
        pca_dict = pca_dict_map[s_name] if s_name in ['Macro', 'Tech'] else None
        
        for iv_name in iv_order:
            print(f"Generating plot for Specification {spec_number:02d}: {iv_name} x {s_name}")
            models = get_models_for_spec(iv_name, s_name, loc_res, nat_res)
            res_df = []

            for yq in yq_list:
                sub = df[df['year_quarter'] == yq]
                if len(sub) == 0: continue
                s_vals = {sv: sub[sv].median() for sv in s_cols}

                vec_loc = get_base_vector(s_cols, s_vals)
                e_loc, se_loc = predict_phi(models['loc'][0], models['loc'][1], vec_loc)

                vec_n1 = vec_loc.copy()
                e_n1, se_n1 = predict_phi(models['n1'][0], models['n1'][1], {**vec_n1, 'constant': 1})  

                vec_n2 = vec_loc.copy()
                a_lag = assets_agg.get(yq, 0)
                for sv in s_cols:
                    if sv != 'constant': vec_n2[f"interaction_{sv}_x_assets"] = s_vals[sv] * a_lag
                e_n2, se_n2 = predict_phi(models['n2'][0], models['n2'][1], {**vec_n2, 'constant': 1})  

                e_n3, se_n3 = np.nan, np.nan
                if models['n3'][0] is not None and pca_dict is not None and yq in pca_dict:
                    vec_n3 = {'nr_lagged_dep': 1.0, 'interaction_pca_index': pca_dict[yq]}
                    e_n3, se_n3 = predict_phi(models['n3'][0], models['n3'][1], {**vec_n3, 'constant': 1})

                res_df.append({
                    'year_quarter': yq, 'loc_est': e_loc, 'loc_se': se_loc,
                    'n1_est': e_n1, 'n1_se': se_n1, 'n2_est': e_n2, 'n2_se': se_n2,
                    'n3_est': e_n3, 'n3_se': se_n3
                })

            res_df = pd.DataFrame(res_df)
            res_df['date'] = pd.PeriodIndex(res_df['year_quarter'].str.replace('_', 'Q'), freq='Q').to_timestamp()
            res_df = res_df.sort_values('date')

            if gen_default: generate_phi_plot(res_df, iv_name, s_name, spec_number, out_dir, all_options=False)
            if gen_all: generate_phi_plot(res_df, iv_name, s_name, spec_number, out_dir, all_options=True)
            
            spec_number += 1
            
    print(f"Plots successfully generated in: {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Unified script: Estimate National Phi and generate plots.")
    parser.add_argument("--option", type=int, choices=[1, 2, 3], help="Estimation: Run specifically one option.")
    parser.add_argument("--run-all", action="store_true", help="Estimation: Run all 3 regression options.")
    parser.add_argument("--all-options", action="store_true", help="Plotting: Generate plots with all 4 variants.")
    parser.add_argument("--default-only", action="store_true", help="Plotting: Generate default plots only.")
    parser.add_argument("--skip-estimation", action="store_true", help="Skip the regression phase.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip the plotting phase.")
    
    args = parser.parse_args()

    if not args.skip_estimation:
        do_estimation(args)
    else:
        print("Skipping estimation phase as per --skip-estimation flag.")
        
    if not args.skip_plots:
        do_plotting(args)
    else:
        print("Skipping plotting phase as per --skip-plots flag.")

if __name__ == "__main__":
    main()
