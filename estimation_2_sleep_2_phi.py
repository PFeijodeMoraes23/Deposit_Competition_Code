"""
estimation_2_sleep_2_phi.py
===========================
Generates 12 plots representing each of the 12 Specifications.
For each specification, plots the aggregated national \phi_t derived from:
- Estimation 1 (Locally Implied National \phi_t)
- Estimation 2, Option 1 (Direct Macro Time-Series Baseline)
- Estimation 2, Option 2 (Firm-Targeting State Interactions)
- Estimation 2, Option 3 (PCA Indexing)
"""
from pathlib import Path
import sys
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
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def resolve_paths():
    _ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
    PANEL_CSV = DATA_DIR / "market_panel.csv"
    
    LOCAL_RESULTS_PICKLE = DATA_DIR / "ESTIMATION_OUTPUT" / "SLEEPINESS_NEW" / "estimation_results.pkl"
    NATIONAL_RESULTS_PICKLE = DATA_DIR / "ESTIMATION_OUTPUT" / "SLEEPINESS_NATIONAL" / "estimation_results.pkl"
    
    OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "PLOTS"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return PANEL_CSV, LOCAL_RESULTS_PICKLE, NATIONAL_RESULTS_PICKLE, OUTPUT_DIR

def build_data(panel_csv):
    s_macro = ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young']
    s_tech = ['pix_users_pf_per1000', 'connections_per100', 'branches_per_1000']
    use_cols = ['year', 'quarter', 'year_quarter', 'lagged_deposits', 'deposit_balance', 'log_total_assets_lag'] + s_macro + s_tech
    
    # Check which columns actually exist first without loading whole file
    with suppress(Exception):
        sample = pd.read_csv(panel_csv, nrows=5)
        # Drop columns that aren't in the CSV to avoid ValueError
        use_cols = [c for c in use_cols if c in sample.columns]
        
    df_raw = pd.read_csv(panel_csv, usecols=use_cols, low_memory=False)

    if 'year_quarter' not in df_raw.columns and 'year' in df_raw.columns:
        df_raw['year_quarter'] = df_raw['year'].astype(int).astype(str) + "Q" + df_raw['quarter'].astype(int).astype(str)

    # Scales
    scale_cols = {
        'gdp_per_capita': 10000.0,
        'cadunico_families_per1000': 100.0,
        'pix_users_pf_per1000': 100.0,
        'connections_per100': 100.0
    }
    for col, factor in scale_cols.items():
        if col in df_raw.columns:
            df_raw[col] /= factor
            
    df_raw['constant'] = 1.0
    if 'year' in df_raw.columns:
        df_raw['post_2020'] = (df_raw['year'] >= 2020).astype(int)
    else:
        df_raw['post_2020'] = (df_raw['year_quarter'].str.extract('(\d{4})')[0].astype(float) >= 2020).astype(int)

    if 'lagged_deposits' in df_raw.columns:
        df_raw['market_size'] = df_raw['lagged_deposits']
    elif 'deposit_balance' in df_raw.columns:
        df_raw['market_size'] = df_raw['deposit_balance']
    else:
        df_raw['market_size'] = 1.0
        
    df_raw['log_total_assets_lag'] = df_raw['log_total_assets_lag'].fillna(df_raw['log_total_assets_lag'].median())
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

def predict_phi(model, param_vector):
    """
    Given a fitted statsmodels result and a dictionary of mapping {param_name: value},
    compute the point estimate and the standard error.
    """
    if model is None: return np.nan, np.nan
    p_names = model.params.index
    V = np.zeros(len(p_names))
    for i, name in enumerate(p_names):
        if name in param_vector:
            V[i] = param_vector[name]
    
    est = V.dot(model.params)
    var = V.dot(model.cov_params()).dot(V)
    return est, np.sqrt(var)

def get_models_for_spec(iv_name, s_name, loc_res, nat_res):
    loc_key = f"{iv_name} x {s_name}"
    return {
        'loc': loc_res.get(loc_key, {}).get('second_stage', None),
        'n1': nat_res.get(f"Option_1_{iv_name}_{s_name}", {}).get('second_stage', None),
        'n2': nat_res.get(f"Option_2_{iv_name}_{s_name}", {}).get('second_stage', None),
        'n3': nat_res.get(f"Option_3_{iv_name}_{s_name}", {}).get('second_stage', None)
    }

def get_base_vector(s_cols, s_vals):
    vec = {}
    for sv in s_cols:
        nm = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
        vec[nm] = s_vals[sv]
    return vec

def generate_phi_plot(res_df, iv_name, s_name, spec_number, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = [
        ('loc', 'Local Implied \phi_t', 'blue'),
        ('n3', 'Direct National (Opt 3)', 'purple')
    ]

    for pfx, label, color in configs:
        est = res_df[f'{pfx}_est']
        se = res_df[f'{pfx}_se']
        if est.notnull().sum() > 0:
            ax.plot(res_df['date'], est, label=label, color=color, linewidth=2)
            ax.fill_between(res_df['date'], est - 1.96*se, est + 1.96*se, color=color, alpha=0.15)

    ax.set_title(f"Specification {spec_number:02d}: {iv_name} x {s_name}", fontsize=14)
    ax.set_ylabel("National \phi_t")
    ax.set_xlabel("Year-Quarter")
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()
    safe_name = f"Spec{spec_number:02d}_{iv_name}_{s_name}".replace(" ", "_").replace("/", "")
    plt.savefig(out_dir / f"{safe_name}.png", dpi=300)
    plt.close(fig)

def process_specification_phi(
    df, yq_list, s_cols, pca_dict, assets_agg, iv_name, s_name, spec_number, 
    loc_res, nat_res, out_dir
):
    print(f"Generating plot for Specification {spec_number:02d}: {iv_name} x {s_name}")

    models = get_models_for_spec(iv_name, s_name, loc_res, nat_res)

    res_df = []

    for yq in yq_list:
        sub = df[df['year_quarter'] == yq]
        if len(sub) == 0: continue
        s_vals = {sv: sub[sv].median() for sv in s_cols}

        vec_loc = get_base_vector(s_cols, s_vals)
        e_loc, se_loc = predict_phi(models['loc'], vec_loc)

        vec_n1 = vec_loc.copy()
        e_n1, se_n1 = predict_phi(models['n1'], {**vec_n1, 'constant': 1})  

        vec_n2 = vec_loc.copy()
        a_lag = assets_agg.get(yq, 0)
        for sv in s_cols:
            if sv != 'constant':
                vec_n2[f"interaction_{sv}_x_assets"] = s_vals[sv] * a_lag
        e_n2, se_n2 = predict_phi(models['n2'], {**vec_n2, 'constant': 1})  

        e_n3, se_n3 = np.nan, np.nan
        if models['n3'] is not None and pca_dict is not None and yq in pca_dict:
            vec_n3 = {'nr_lagged_dep': 1.0, 'interaction_pca_index': pca_dict[yq]}
            e_n3, se_n3 = predict_phi(models['n3'], {**vec_n3, 'constant': 1})

        res_df.append({
            'year_quarter': yq,
            'loc_est': e_loc, 'loc_se': se_loc,
            'n1_est': e_n1, 'n1_se': se_n1,
            'n2_est': e_n2, 'n2_se': se_n2,
            'n3_est': e_n3, 'n3_se': se_n3
        })

    res_df = pd.DataFrame(res_df)
    res_df['date'] = pd.PeriodIndex(res_df['year_quarter'].str.replace('_', 'Q'), freq='Q').to_timestamp()
    res_df = res_df.sort_values('date')

    generate_phi_plot(res_df, iv_name, s_name, spec_number, out_dir)

def main():
    panel_csv, local_pkl, nat_pkl, out_dir = resolve_paths()
    
    df = build_data(panel_csv)
    
    with open(local_pkl, 'rb') as f:
        loc_res = pickle.load(f)
        
    with open(nat_pkl, 'rb') as f:
        nat_res = pickle.load(f)
        
    yq_list = sorted(df['year_quarter'].dropna().unique())

    s_base = ['constant', 'post_2020']
    s_macro = s_base + ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young']
    s_tech = s_macro + ['pix_users_pf_per1000', 'branches_per_1000']
    s_tech = [c for c in s_tech if c in df.columns]

    # Pre-calculate assets averages for Option 2 per quarter
    # We need sum(M_jt * Assets) / sum(M_jt)
    assets_agg = {}
    for yq in yq_list:
        sub = df[df['year_quarter'] == yq]
        m_tot = sub['market_size'].sum()
        if m_tot > 0:
            avg_a = (sub['log_total_assets_lag'] * sub['market_size']).sum() / m_tot
        else:
            avg_a = 0
        assets_agg[yq] = avg_a

    pca_base = get_pca_index(df, s_base)
    pca_macro = get_pca_index(df, s_macro)
    pca_tech = get_pca_index(df, s_tech)

    iv_order = ['OLS', 'IV_CostShifters', 'IV_Wholesale', 'IV_HausmanFull']
    state_order = ['Base', 'Macro', 'Tech']
    
    state_blocks = {
        'Base': s_base,
        'Macro': s_macro,
        'Tech': s_tech
    }
    
    spec_number = 1
    for s_name in state_order:
        s_cols = state_blocks[s_name]
        pca_dict = pca_tech if s_name == 'Tech' else (pca_macro if s_name == 'Macro' else None)
        
        for iv_name in iv_order:
            process_specification_phi(
                df, yq_list, s_cols, pca_dict, assets_agg, iv_name, s_name, 
                spec_number, loc_res, nat_res, out_dir
            )
            spec_number += 1

if __name__ == '__main__':
    main()