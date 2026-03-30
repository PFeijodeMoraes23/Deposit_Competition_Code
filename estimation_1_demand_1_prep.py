"""
estimation_1_demand.py
========================
Prepares demand-side variables for the BLP step (Eq-15 and Eq-16 in V_Main.tex).

This script reads the output of `estimation_1_sleep.py` (which contains 12 specifications
for the depositor sleepiness function phi_mt) and the raw market panel. For each 
specification, it computes the implied phi_mt, aggregates it to the national level phi_t,
computes "Active Deposits" (Dep^Act) deducting the slept-on balances, and finally 
computes the data-implied conditional market shares for B-type and D-type institutions.

Outputs a consolidated long panel to be ingested by the BLP fixed-point contraction.
"""

import os
import sys
import json
import pickle
from pathlib import Path

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import numpy as np

try:
    from utils.toon_parser import get_script_config, load_default_toon_context
except Exception:
    get_script_config = None
    load_default_toon_context = None

# ==============================================================================
# 0. Global Paths and Parameters
# ==============================================================================
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
PANEL_CSV = DATA_DIR / "market_panel.csv"

def _resolve_runtime_paths() -> tuple[Path, Path, Path]:
    """Resolve panel/output paths with optional TOON overrides."""
    panel_csv = PANEL_CSV
    sleep_output_dir = DATA_DIR / "ESTIMATION_OUTPUT" / "SLEEPINESS_NEW"
    demand_output_dir = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP"

    if load_default_toon_context is None or get_script_config is None:
        return panel_csv, sleep_output_dir, demand_output_dir

    try:
        context = load_default_toon_context(Path(__file__).resolve().parent / "utils", quiet=True)
    except Exception:
        return panel_csv, sleep_output_dir, demand_output_dir

    cfg = get_script_config(context, "estimation_1_demand")
    if not isinstance(cfg, dict):
        return panel_csv, sleep_output_dir, demand_output_dir

    if isinstance(cfg.get("panel_csv"), str) and cfg["panel_csv"].strip():
        panel_csv = Path(cfg["panel_csv"]).expanduser()
    if isinstance(cfg.get("sleep_output_dir"), str) and cfg["sleep_output_dir"].strip():
        sleep_output_dir = Path(cfg["sleep_output_dir"]).expanduser()
    if isinstance(cfg.get("demand_output_dir"), str) and cfg["demand_output_dir"].strip():
        demand_output_dir = Path(cfg["demand_output_dir"]).expanduser()

    return panel_csv, sleep_output_dir, demand_output_dir

# ==============================================================================
# Helper Extractors
# ==============================================================================
def extract_upsilon_terms(res_ss):
    """
    Extract perfectly mapped state variable names and their Upsilon 
    coefficients from the statsmodels RegressionResults object.
    
    The raw coefficients correspond to:
      - 'nr_lagged_dep' -> 'constant' coefficient
      - 'interaction_{X}' -> 'X' coefficient
    """
    params = res_ss.params
    upsilon = {}
    
    for var_name, coef in params.items():
        if var_name == "nr_lagged_dep":
            upsilon["constant"] = coef
        elif var_name.startswith("interaction_"):
            clean_name = var_name.replace("interaction_", "")
            upsilon[clean_name] = coef
            
    return upsilon

def _reshape_panel_from_wide(df_raw: pd.DataFrame) -> pd.DataFrame:
    print("Reshaping panel from wide to long...")
    id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
    df_raw = df_raw.drop_duplicates(subset=id_vars)
    df_raw['is_B'] = (df_raw['CODMUN_IBGE'].astype(str) != '0')
    
    df = pd.wide_to_long(
        df_raw, 
        stubnames=['dep_a', 'spread_a'], 
        i=id_vars, 
        j='deposit_type'
    ).reset_index()
    
    return df.rename(columns={
        'dep_a': 'deposit_balance', 
        'spread_a': 'spread_qoq'
    })

def _reshape_panel(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw['mca_code'] = df_raw['mca_code'].astype(str)
    
    if 'dep_a1' in df_raw.columns:
        df = _reshape_panel_from_wide(df_raw)
    else:
        df = df_raw.copy()
        if 'is_B' not in df.columns:
            df['is_B'] = (df['CODMUN_IBGE'].astype(str) != '0')
            
    if 'deposit_type' in df.columns:
        df = df[df['deposit_type'] != 3].copy()
        
    return df

def build_base_panel(panel_csv: Path) -> pd.DataFrame:
    """
    Reads and pivots wide format to long format, applying the necessary standardizations
    and lags consistent with the first stage.
    """
    print(f"Loading {panel_csv}...")
    df_raw = pd.read_csv(panel_csv)
    
    df = _reshape_panel(df_raw)
            
    # -------------------------------------------------------------
    # 2. Assign Keys & Shifts
    # -------------------------------------------------------------
    df['entity_id'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + \
                      df['deposit_type'].astype(str) + "_" + \
                      df['mca_code'].astype(str)
                      
    df['time_id'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)
    df.sort_values(by=['entity_id', 'year', 'quarter'], inplace=True)
    
    df['spread_qoq_lag'] = df.groupby('entity_id')['spread_qoq'].shift(1)
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)
    df['lagged_deposits'] = df.groupby('entity_id')['deposit_balance'].shift(1)
    
    # Deposit Rate: risk-free - spread
    df['deposit_rate_lag'] = df['risk_free_qoq_lag'] - df['spread_qoq_lag']
    # Gross Return term: 1 + r_{t-1}
    df['gross_return_lag'] = 1 + df['deposit_rate_lag']
    
    df = df.dropna(subset=['deposit_balance', 'lagged_deposits', 'spread_qoq', 'entity_id', 'time_id'])
    
    # Scale State Variables (Matching exactly estimation_1_sleep)
    df['constant'] = 1.0
    if 'year' in df.columns:
        df['post_2020'] = (df['year'] >= 2020).astype(int)
    if 'gdp_per_capita' in df.columns:
        df['gdp_per_capita'] = df['gdp_per_capita'] / 10000.0
    if 'cadunico_families_per1000' in df.columns:
        df['cadunico_families_per1000'] = df['cadunico_families_per1000'] / 100.0
    if 'pix_users_pf_per1000' in df.columns:
        df['pix_users_pf_per1000'] = df['pix_users_pf_per1000'] / 100.0
    if 'connections_per100' in df.columns:
        df['connections_per100'] = df['connections_per100'] / 100.0
        
    s_tech_finance = ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']
    for col in s_tech_finance:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    # Population weights for phi_t aggregation.
    # MCAs with missing pop_total are excluded from phi_t weighting (not filled).
    if 'pop_total' not in df.columns:
        df['pop_total'] = np.nan
        
    return df

# ==============================================================================
# Main Routine
# ==============================================================================
def process_specification(args):
    """
    Worker function to process a single specification isolated in its own process.
    Returns the processed df_spec DataFrame, a summary dictionary, and the spec_name.
    """
    spec_name, spec_res, df_base = args
    # Imports must be local for ProcessPool execution on Windows
    import numpy as np
    import pandas as pd
    
    res_ss = spec_res.get('second_stage')
    if res_ss is None:
        return None, None, spec_name
        
    upsilon = extract_upsilon_terms(res_ss)
    
    # Working copy for this specification
    df_spec = df_base[['entity_id', 'time_id', 'CodConglomeradoPrudencial', 'mca_code', 
                       'year', 'quarter', 'deposit_type', 'is_B', 
                       'deposit_balance', 'lagged_deposits', 'gross_return_lag', 'pop_total']].copy()
    
    # 2. Compute phi_mt = Upsilon' * S_mt
    df_spec['phi_mt'] = 0.0
    missing_sv = False
    
    for sv_name, beta in upsilon.items():
        if sv_name not in df_base.columns:
            missing_sv = True
            break
        df_spec['phi_mt'] += beta * df_base[sv_name]
        
    if missing_sv:
        return None, None, spec_name
        
    # Drop markets that had missing demographics (NaN S_mt)
    df_spec = df_spec.dropna(subset=['phi_mt'])
    
    df_spec['phi_mt'] = df_spec['phi_mt'].clip(lower=0.0, upper=1.0)
    
    # 3. Compute phi_t (National Population-Weighted Mean of phi_mt)
    # Only MCAs with valid pop_total participate in the weighting.
    df_mca_level = df_spec[['mca_code', 'time_id', 'phi_mt', 'pop_total']].drop_duplicates()
    df_mca_level = df_mca_level.dropna(subset=['pop_total'])
    
    def weighted_mean(g):
        v, w = g['phi_mt'], g['pop_total']
        return v.mean() if w.sum() == 0 else np.average(v, weights=w)
        
    phi_t_map = df_mca_level.groupby('time_id').apply(weighted_mean, include_groups=False).rename("phi_t")
    df_spec = df_spec.join(phi_t_map, on='time_id')
    df_spec['phi_t'] = df_spec['phi_t'].fillna(0.0).clip(lower=0.0, upper=1.0)
    
    # 4. Compute Active Deposits
    df_spec['Dep_Act'] = 0.0
    
    # B-type
    val_B = df_spec['deposit_balance'] - df_spec['phi_mt'] * df_spec['gross_return_lag'] * df_spec['lagged_deposits']
    df_spec.loc[df_spec['is_B'], 'Dep_Act'] = np.maximum(0.0, val_B[df_spec['is_B']])
    
    # D-type
    val_D = df_spec['deposit_balance'] - df_spec['phi_t'] * df_spec['gross_return_lag'] * df_spec['lagged_deposits']
    df_spec.loc[~df_spec['is_B'], 'Dep_Act'] = np.maximum(0.0, val_D[~df_spec['is_B']])
    
    # 5. Compute Data-Implied Shares
    nat_active = df_spec.groupby(['time_id', 'deposit_type'])['Dep_Act'].transform('sum')
    df_spec['nat_active'] = nat_active
    df_spec['share_D'] = np.where(~df_spec['is_B'], df_spec['Dep_Act'] / nat_active, np.nan)
    
    local_B_active = df_spec[df_spec['is_B']].groupby(['mca_code', 'time_id', 'deposit_type'])['Dep_Act'].transform('sum')
    
    df_loc_map = df_spec[df_spec['is_B']][['entity_id', 'time_id']].copy()
    df_loc_map['local_B_active'] = local_B_active
    df_spec = pd.merge(df_spec, df_loc_map, on=['entity_id', 'time_id'], how='left')
    
    mask_valid_denom = (df_spec['is_B']) & (df_spec['local_B_active'] > 0)
    df_spec['share_B_cond'] = np.where(mask_valid_denom, df_spec['Dep_Act'] / df_spec['local_B_active'], np.nan)
    df_spec.loc[(df_spec['is_B']) & ~mask_valid_denom, 'share_B_cond'] = 0.0

    # Mapping from V_Main.tex Table 1
    SPEC_MAP = {
        'OLS x Base': 1,
        'IV_CostShifters x Base': 2,
        'IV_Wholesale x Base': 3,
        'IV_HausmanFull x Base': 4,
        'OLS x Macro': 5,
        'IV_CostShifters x Macro': 6,
        'IV_Wholesale x Macro': 7,
        'IV_HausmanFull x Macro': 8,
        'OLS x Tech': 9,
        'IV_CostShifters x Tech': 10,
        'IV_Wholesale x Tech': 11,
        'IV_HausmanFull x Tech': 12
    }
    
    spec_id = SPEC_MAP.get(spec_name, spec_name)
    df_spec['Spec_ID'] = spec_id
    
    b_count = df_spec['is_B'].sum()
    d_count = (~df_spec['is_B']).sum()
    summary = {
        "Name": spec_name,
        "Spec_ID": spec_id,
        "Rows": len(df_spec),
        "B_firms": int(b_count),
        "D_firms": int(d_count),
        "Mean_phi_mt": float(df_spec['phi_mt'].mean()),
        "Mean_phi_t": float(df_spec['phi_t'].mean()),
        "B_Cond_Share_NaNs": int(df_spec[df_spec['is_B']]['share_B_cond'].isna().sum()),
        "D_Share_NaNs": int(df_spec[~df_spec['is_B']]['share_D'].isna().sum())
    }
    
    return df_spec, summary, spec_id

def main():
    import concurrent.futures
    
    panel_csv, sleep_output_dir, demand_output_dir = _resolve_runtime_paths()
    
    results_pickle = sleep_output_dir / "estimation_results.pkl"
    if not results_pickle.exists():
        print(f"ERROR: Pickle file missing at {results_pickle}. Run estimation_1_sleep.py first.")
        sys.exit(1)
        
    print(f"Loading estimation results from {results_pickle}...")
    with open(results_pickle, 'rb') as f:
        results_dict = pickle.load(f)
        
    df_base = build_base_panel(panel_csv)
    print(f"Base Panel rows (with valid lagged structure): {len(df_base)}")
    
    demand_output_dir.mkdir(parents=True, exist_ok=True)
    
    spec_summaries = {}
    
    print("\n--- Processing Specifications in Parallel ---")
    tasks = [(name, res, df_base) for name, res in results_dict.items()]
    
    n_saved = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for df_spec, summary, spec_id in executor.map(process_specification, tasks):
            if df_spec is not None:
                # Save per-spec CSV
                out_csv = demand_output_dir / f"demand_prep_spec_{spec_id}.csv"
                df_spec.to_csv(out_csv, index=False, float_format='%.6f')
                print(f" > Saved Spec {spec_id} -> {out_csv.name} ({len(df_spec)} rows)")
                spec_summaries[str(spec_id)] = summary
                n_saved += 1
            else:
                print(f"   [!] Failed or skipped Spec: {spec_id}")

    if n_saved == 0:
        print("No valid specifications were processed.")
        sys.exit(1)
        
    out_json = demand_output_dir / "demand_prep_summary.json"
    with open(out_json, "w") as f:
        json.dump(spec_summaries, f, indent=4)
        
    print(f"\nSummary saved to: {out_json}")
    print(f"[SUCCESS] {n_saved} per-spec CSVs written to {demand_output_dir}.")

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
