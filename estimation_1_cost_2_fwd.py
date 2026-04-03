import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def get_paths():
    base = Path(__file__).resolve().parent.parent.parent / "BCB" / "Egan_et_al_2025_Rep"
    processed = base / "processed"
    estim_out = processed / "ESTIMATION_OUTPUT"
    blp_out = estim_out / "BLP_RESULTS"
    cost_out = estim_out / "COST_FWD"
    cost_out.mkdir(parents=True, exist_ok=True)
    return processed, blp_out, cost_out

def execute_forward_simulation(args):
    print(f"======================================================================")
    print(f"  Cost Forward Simulation (BBL Step 2)")
    print(f"  Spec: {args.spec} | N_shocks: {args.shocks} | Beta: {args.beta}")
    print(f"======================================================================")

    processed, blp_out, cost_out = get_paths()
    
    # 1. Load Data
    sp_id = args.spec
    panel_path = processed / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / f"demand_final_spec_{sp_id}.pkl"
    if not panel_path.exists():
        print(f"  [FATAL] Missing BLP prepared panel: {panel_path}")
        return
        
    df = pd.read_pickle(panel_path)
    t0 = df['time_id'].max()
    df_t0 = df[df['time_id'] == t0].copy().reset_index(drop=True)
    
    print(f"  Baseline T=0 selected: time_id={t0} with {len(df_t0)} observations.")

    # 2. Load BLP Parameters
    blp_pkl = blp_out / f"blp_results_spec_{sp_id}_logit.pkl"
    
    if not blp_pkl.exists():
        print(f"  [WARNING] Could not find {blp_pkl.name}. Simulating without structural parameters (Testing mode).")
        # Fallback parameters
        alpha_spread = -1.0 # default downward sloping demand
        delta_base = df_t0['share_B_cond'].values if 'share_B_cond' in df_t0 else np.zeros(len(df_t0))
    else:
        print(f"  Loaded BLP structural parameters from {blp_pkl.name}")
        with open(blp_pkl, 'rb') as f:
            blp_data = pickle.load(f)
        try:
            alpha_spread = blp_data['theta1'][0] # naive extraction
            print(f"  Detected alpha_spread = {alpha_spread:.4f}")
        except:
            alpha_spread = -1.0
        delta_base = blp_data.get('delta', np.zeros(len(df_t0)))

    # 3. Vectorized Simulation Engine
    N_shocks = args.shocks
    N_obs = len(df_t0)
    beta = args.beta
    T_HORIZON = 50
    
    # Identify B vs D firms
    is_B = df_t0['is_B'].values
    is_D = ~is_B
    
    # Identify k=4, 5
    is_k_endog = df_t0['deposit_type'].isin([4, 5]).values
    
    # Initialize policy perturbations (shocks) -> Normal(0, 0.05^2) for spreads
    # Shape: (N_obs, shocks). Only apply to endogenous types.
    shocks = np.zeros((N_obs, N_shocks))
    shocks[is_k_endog, :] = np.random.normal(loc=0.0, scale=0.05, size=(is_k_endog.sum(), N_shocks))
    
    # Baseline vectors
    r_f = np.full(N_obs, 0.02) # Proxy risk free rate for the forward simulation
    spread_obs = df_t0['spread_qoq'].values
    dep_0 = df_t0['Dep_Act'].values if 'Dep_Act' in df_t0.columns else np.zeros(N_obs)
    phi_0 = df_t0['phi_mt'].values if 'phi_mt' in df_t0.columns else np.full(N_obs, 0.5)
    z_cost = df_t0['log_total_assets_lag'].values if 'log_total_assets_lag' in df_t0.columns else np.zeros(N_obs)
    
    # Allocate cost parameter bases: Base 1, Base Omega, Base Gamma, Base Zeta
    # Shape: (N_obs, shocks, 4 cost bases)
    V_bases = np.zeros((N_obs, N_shocks, 4), dtype=np.float64)
    
    # Active simulation arrays
    dep_current = np.tile(dep_0[:, None], (1, N_shocks)) # (N_obs, shocks)
    spread_perturbed = spread_obs[:, None] + shocks     # (N_obs, shocks)
    
    print(f"  Executing 50-quarter forward simulation loop over tensor {spread_perturbed.shape} ...")
    
    for t in range(T_HORIZON):
        beta_t = beta ** t
        
        # 1. Update Market Shares (BLP Dependency)
        delta_shift = -alpha_spread * shocks 
        # In full BLP, this recalculates shares. As a proxy, logit shares evolve by exp(shift)
        awake_shares_pct = np.clip(np.exp(delta_shift), 0.01, 10.0) 
        
        # 2. Advance Deposits (Sleepiness retention + Awake influx)
        retention_rate = phi_0[:, None] * (1.0 + r_f[:, None] - spread_perturbed)
        
        # We assume total market size for Awake influx grows at population or stays flat
        # For memory, we use a simple additive influx proportional to base scale and share shift
        awake_influx = (1.0 - phi_0[:, None]) * dep_0[:, None] * awake_shares_pct
        
        dep_current = dep_current * retention_rate + awake_influx
        
        # 3. Accumulate cost bases for V
        # Base 0: Revenue = Dep * (spread) [actually (r^f - r_bank) == spread in eq?]
        V_bases[:, :, 0] += beta_t * (dep_current * spread_perturbed)
        # Base 1: -Omega base = Dep
        V_bases[:, :, 1] += beta_t * dep_current
        # Base 2: -Gamma base = Dep * Z
        V_bases[:, :, 2] += beta_t * (dep_current * z_cost[:, None])
        # Base 3: -Zeta base = r^f * Dep
        V_bases[:, :, 3] += beta_t * (r_f[:, None] * dep_current)

    print(f"  Simulation complete. Aggregating Corporate constraints...")
    
    # 4. Aggregate via np.add.reduceat
    sort_idx = np.argsort(df_t0['entity_id'].values)
    firm_ids = df_t0['entity_id'].values[sort_idx]
    V_bases_sorted = V_bases[sort_idx, :, :]
    
    unique_firms, unq_idx = np.unique(firm_ids, return_index=True)
    V_firm = np.add.reduceat(V_bases_sorted, unq_idx, axis=0)
    
    out_file = cost_out / f"violations_firm_spec_{args.spec}.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump({"firms": unique_firms, "V_firm_bases": V_firm}, f)
        
    print(f"  Saved corporate violation bounds to {out_file.name}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward Simulation for BBL Cost estimation.")
    parser.add_argument("--spec", type=str, default="1", help="Specification ID to run (e.g. 1)")
    parser.add_argument("--shocks", type=int, default=50, choices=[20, 50, 100, 200], help="Number of policy violations (tilde sigma) to draw.")
    parser.add_argument("--beta", type=float, default=0.90, help="Discount factor (beta) - typically 0.90 or 0.95.")
    args = parser.parse_args()
    
    execute_forward_simulation(args)
