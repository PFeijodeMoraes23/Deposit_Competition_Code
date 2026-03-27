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
    if 'gdp_per_capita' in df.columns:
        df['gdp_per_capita'] = df['gdp_per_capita'] / 10000.0
    if 'cadunico_families_per1000' in df.columns:
        df['cadunico_families_per1000'] = df['cadunico_families_per1000'] / 100.0
    if 'pix_users_pf_per1000' in df.columns:
        df['pix_users_pf_per1000'] = df['pix_users_pf_per1000'] / 100.0
    if 'connections_per100' in df.columns:
        df['connections_per100'] = df['connections_per100'] / 100.0

    phi_results = {}
    
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
                    # use the appropriate variable (imputed where necessary as in first stage)
                    med_val = df[sv].median() if sv in df.columns else 0
                    local_s = df[sv].fillna(med_val) if sv in df.columns else 0
                    phi_mt += c * local_s
        
        # Save phi_mt local parameters
        df[f'phi_mt_{spec_name}'] = phi_mt
        
        # Calculate market size M_mt (total deposits)
        temp_df = df.copy()
        # Ensure correct column
        if 'lagged_deposits' in temp_df.columns:
            temp_df['market_size'] = temp_df['lagged_deposits']
        elif 'deposit_balance' in temp_df.columns:
            temp_df['market_size'] = temp_df['deposit_balance']
        else:
            temp_df['market_size'] = 1.0 # fallback
            
        # We need sum by market 
        market_agg = temp_df.groupby(['year_quarter', 'CodIbge']).agg(
            phi_mt=(f'phi_mt_{spec_name}', 'mean'),
            M_mt=('market_size', 'sum')
        ).reset_index()

        # National aggregate (\phi_t) = sum(phi_mt * M_mt) / sum(M_mt)
        national_agg = market_agg.groupby('year_quarter').apply(
            lambda x: np.sum(x['phi_mt'] * x['M_mt']) / np.sum(x['M_mt']) if np.sum(x['M_mt']) > 0 else 0
        ).reset_index(name=f'phi_t_{spec_name}')
        
        phi_results[spec_name] = national_agg
        
    return df, phi_results

def main():
    panel_csv_path, results_pickle_path, output_dir = resolve_paths()
    
    print(f"Loading data from {panel_csv_path}")
    df = pd.read_csv(panel_csv_path)
    
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
    agg_df = None
    for name, nat_df in national_phis.items():
        if agg_df is None:
            agg_df = nat_df
        else:
            agg_df = agg_df.merge(nat_df, on='year_quarter', how='outer')
            
    if agg_df is not None:
        agg_path = output_dir / "national_phi_t.csv"
        agg_df.to_csv(agg_path, index=False)
        print(f"Exported National Phi_t to {agg_path}")
        
    print("Done!")

if __name__ == "__main__":
    main()
