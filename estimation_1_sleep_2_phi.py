import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ==============================================================================
# Helper to read results
# ==============================================================================
def resolve_paths():
    _ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
    PANEL_CSV = DATA_DIR / "market_panel.csv"
    OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "SLEEPINESS_NEW"
    RESULTS_PICKLE = OUTPUT_DIR / "estimation_results.pkl"
    return PANEL_CSV, RESULTS_PICKLE, OUTPUT_DIR

# ==============================================================================
# Computes locally expected Share of Asleep Depositors (\phi_{mt}) 
# and nationally weighted aggregate (\phi_t)
# ==============================================================================
def calculate_phis(df, res_dict, state_blocks):
    df['constant'] = 1.0
    
    # Same scaling configuration as in estimation 1
    scale_cols = {
        'gdp_per_capita': 10000.0,
        'cadunico_families_per1000': 100.0,
        'pix_users_pf_per1000': 100.0,
        'connections_per100': 100.0
    }
    for col, factor in scale_cols.items():
        if col in df.columns:
            df[col] /= factor

    phi_results = {}
    
    # Precompute medians and filled series to avoid redundant computation in loop
    all_s_cols = {col for cols in state_blocks.values() for col in cols if col != 'constant'}
    filled_cols = {
        sv: df[sv].fillna(df[sv].median()) if sv in df.columns else np.zeros(len(df))
        for sv in all_s_cols
    }

    # Calculate market size M_mt (total deposits)
    if 'lagged_deposits' in df.columns:
        df['market_size'] = df['lagged_deposits']
    elif 'deposit_balance' in df.columns:
        df['market_size'] = df['deposit_balance']
    else:
        df['market_size'] = 1.0 # fallback

    for spec_name, s_cols in state_blocks.items():
        # Using Hausman (Full) specification as the canonical parameter
        model_key = f"Spec4-IV_HausmanFull x {spec_name}"
        if model_key not in res_dict:
            continue
            
        ss_res = res_dict[model_key]['second_stage']
        if ss_res is None:
            continue
            
        phi_mt = np.zeros(len(df))
        
        for sv in s_cols:
            col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
            if col_name in ss_res.params:
                c = ss_res.params[col_name]
                if sv == 'constant':
                    phi_mt += c
                else: 
                    phi_mt += c * filled_cols[sv]
        
        # Save phi_mt local parameters
        df[f'phi_mt_{spec_name}'] = phi_mt
        
        # We need sum by market 
        market_agg = df.groupby(['year_quarter', 'CodIbge'], observed=True).agg(
            phi_mt=(f'phi_mt_{spec_name}', 'mean'),
            M_mt=('market_size', 'sum')
        ).reset_index()

        # National aggregate (\phi_t) = sum(phi_mt * M_mt) / sum(M_mt)
        weighted_phi = market_agg['phi_mt'] * market_agg['M_mt']
        sum_weighted = weighted_phi.groupby(market_agg['year_quarter']).sum()
        sum_m_mt = market_agg['M_mt'].groupby(market_agg['year_quarter']).sum()
        
        # Vectorized aggregation and division
        national_agg = (sum_weighted / sum_m_mt.replace(0, np.nan)).fillna(0).reset_index(name=f'phi_t_{spec_name}')
        
        phi_results[spec_name] = national_agg
        
    return df, phi_results

def main():
    panel_csv_path, results_pickle_path, output_dir = resolve_paths()
    
    print(f"Loading data from {panel_csv_path}")
    df = pd.read_csv(panel_csv_path, low_memory=False)
    
    print(f"Loading results from {results_pickle_path}")
    with open(results_pickle_path, 'rb') as f:
        res_dict = pickle.load(f)
        
    s_base = ['constant']
    s_macro = s_base + ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young']
    s_tech_finance = s_macro + ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']
    
    state_blocks = {
        'Base': s_base,
        'Macro': s_macro,
        'Tech': s_tech_finance
    }

    df, national_phis = calculate_phis(df, res_dict, state_blocks)
    
    # Save the expanded panel data locally 
    phi_df_path = output_dir / "market_panel_phis.csv"
    df.to_csv(phi_df_path, index=False)
    print(f"Exported panel data with computed phi_mt to {phi_df_path}")
    
    # Output to pickle or CSV for subsequent usage
    from functools import reduce
    
    if national_phis:
        agg_df = reduce(lambda left, right: pd.merge(left, right, on='year_quarter', how='outer'), national_phis.values())
        agg_path = output_dir / "national_phi_t.csv"
        agg_df.to_csv(agg_path, index=False)
        print(f"Exported National Phi_t to {agg_path}")
        
    print("Done!")

if __name__ == "__main__":
    main()
