"""
estimation_2_demand_1_prep.py
===============================
Hybrid demand preparation script that combines the logic of
estimation_1_demand_1_prep.py (phi → active deposits → shares) and
estimation_1_demand_2_secondprep.py (panel merge + pkl serialization)
in a single sweep.

Key difference from the estimation_1 pipeline
----------------------------------------------
For specifications (5)–(12) — i.e. Macro and Tech state-variable blocks —
D-type (national/digital) firms use the PCA-implied national sleepiness
function phi_t^{Nat,3} from the estimation_2_sleep pipeline (Option 3,
estimated in estimation_2_sleep_1_cfa.py), instead of the population-
weighted local aggregate phi_t^{Loc}.

For specifications (1)–(4) — the Base block — D-type firms continue to
use the locally-implied national aggregate phi_t^{Loc}, identical to
estimation_1_demand_1_prep.py.

B-type (brick-and-mortar) firms always use their local phi_mt from the
estimation_1_sleep pipeline, unchanged across all 12 specifications.

Outputs
-------
  demand_2_final_spec_{spec_id}.pkl    (one per specification, in DEMAND_PREP)
  demand_prep_2_summary.json           (summary diagnostics)

Usage
-----
  python estimation_2_demand_1_prep.py               # all specs
  python estimation_2_demand_1_prep.py --spec 5       # single spec
  python estimation_2_demand_1_prep.py --spec all     # explicit all

References
----------
  Egan, Hortacsu & Matvos (2025, NBER WP)
  Conlon & Gortmaker (2020, RAND J. Econ.)

CLI Options:
------------
usage: estimation_2_demand_1_prep.py [-h] [--spec SPEC]

Hybrid demand prep: B-type local phi + D-type PCA phi

options:
  -h, --help   show this help message and exit
  --spec SPEC  Specification ID (1-12) or "all"
"""

import sys
import os
import json
import pickle
import argparse
from pathlib import Path

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# ==============================================================================
# 0. Paths & Constants
# ==============================================================================
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
PANEL_CSV = DATA_DIR / "market_panel.csv"

SLEEP_LOCAL_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "rout_1"
SLEEP_NATIONAL_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "rout_2"
DEMAND_PREP_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP"

# Product characteristics and instrument columns (shared with BLP loop)
X_COLS = ['fgc_covered', 'has_ip', 'seg_S2', 'seg_S3', 'seg_S4', 'seg_S5',
          'log_total_assets_lag', 'equity_ratio_lag']
K_LIST = [1, 2, 4, 5]

D_COLS = ['gdp_per_capita', 'fraction_65plus', 'fraction_young',
          'pix_users_pf_per1000', 'connections_per100', 'frac_4g5g',
          'branches_per1000', 'cadunico_families_per1000']

IV_BLP_LOO = ['loo_log_assets', 'mean_loo_log_assets',
              'loo_equity_assets', 'mean_loo_equity_ratio',
              'loo_indice_basileia', 'mean_loo_basileia',
              'loo_credit_assets', 'mean_loo_credit_assets',
              'loo_npl_provision', 'mean_loo_npl_provision',
              'n_rivals']
IV_COST = ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag',
           'tax_cost_ratio_lag']
IV_CAPITAL = ['indice_basileia_lag']

MP_KEEP_COLS = (
    ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter', 'CODMUN_IBGE']
    + [f'spread_a{k}' for k in K_LIST]
    + [f'dep_a{k}' for k in K_LIST]
    + X_COLS + D_COLS + ['pop_total']
    + IV_BLP_LOO + IV_COST + IV_CAPITAL
    + ['segment']
)

# Canonical specification ordering (V_Main.tex Table 1)
SPEC_MAP = {
    'OLS x Base': 1,        'IV_CostShifters x Base': 2,
    'IV_Wholesale x Base': 3,'IV_HausmanFull x Base': 4,
    'OLS x Macro': 5,       'IV_CostShifters x Macro': 6,
    'IV_Wholesale x Macro': 7,'IV_HausmanFull x Macro': 8,
    'OLS x Tech': 9,        'IV_CostShifters x Tech': 10,
    'IV_Wholesale x Tech': 11,'IV_HausmanFull x Tech': 12,
}

# Inverse map: spec_id -> (iv_name, state_block)
SPEC_ID_TO_KEYS = {v: k for k, v in SPEC_MAP.items()}

IV_ORDER = ['OLS', 'IV_CostShifters', 'IV_Wholesale', 'IV_HausmanFull']
STATE_ORDER = ['Base', 'Macro', 'Tech']

# State variable definitions (must match estimation_1_sleep_1_cfa.py exactly)
S_BASE = ['constant', 'post_2020']
S_MACRO = S_BASE + ['gdp_per_capita', 'cadunico_families_per1000',
                     'fraction_65plus', 'fraction_young', 'risk_free_qoq_lag']
S_TECH = S_MACRO + ['pix_users_pf_per1000', 'connections_per100',
                     'branches_per1000']

STATE_BLOCKS = {'Base': S_BASE, 'Macro': S_MACRO, 'Tech': S_TECH}


# ==============================================================================
# 1. Data Loading
# ==============================================================================
def build_base_panel(panel_csv: Path) -> pd.DataFrame:
    """Load market_panel.csv, reshape wide → long, construct lags and scale
    state variables to match the sleepiness estimation exactly."""
    print(f"Loading {panel_csv}...")
    df_raw = pd.read_csv(panel_csv, dtype={'mca_code': str}, low_memory=False)
    df_raw['mca_code'] = df_raw['mca_code'].astype(str)

    # Reshape wide → long
    if 'dep_a1' in df_raw.columns:
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
        df = df.rename(columns={'dep_a': 'deposit_balance',
                                'spread_a': 'spread_qoq'})
    else:
        df = df_raw.copy()
        if 'is_B' not in df.columns:
            df['is_B'] = (df['CODMUN_IBGE'].astype(str) != '0')

    # Drop interbank deposits (k=3)
    if 'deposit_type' in df.columns:
        df = df[df['deposit_type'] != 3].copy()

    # Entity key & temporal lags
    df['entity_id'] = (df['CodConglomeradoPrudencial'].astype(str) + "_"
                       + df['deposit_type'].astype(str) + "_"
                       + df['mca_code'].astype(str))
    df['time_id'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)
    df.sort_values(by=['entity_id', 'year', 'quarter'], inplace=True)

    df['spread_qoq_lag'] = df.groupby('entity_id')['spread_qoq'].shift(1)
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)
    df['lagged_deposits'] = df.groupby('entity_id')['deposit_balance'].shift(1)

    # Deposit rate and gross return
    df['deposit_rate_lag'] = df['risk_free_qoq_lag'] - df['spread_qoq_lag']
    df['gross_return_lag'] = 1 + df['deposit_rate_lag']

    df = df.dropna(subset=['deposit_balance', 'lagged_deposits',
                           'spread_qoq', 'entity_id', 'time_id'])

    # Scale state variables (matching estimation_1_sleep_1_cfa.py)
    df['constant'] = 1.0
    if 'year' in df.columns:
        df['post_2020'] = (df['year'] >= 2020).astype(int)
    for col, factor in [('gdp_per_capita', 10000.0),
                        ('cadunico_families_per1000', 100.0),
                        ('pix_users_pf_per1000', 100.0),
                        ('connections_per100', 100.0)]:
        if col in df.columns:
            df[col] = df[col] / factor

    # Fill tech demographics with median (matching estimation_1)
    for col in ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if 'pop_total' not in df.columns:
        df['pop_total'] = np.nan

    # year_quarter for PCA merge
    df['year_quarter'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)

    return df


def load_panel_selective(panel_csv: Path) -> pd.DataFrame:
    """Load the selective panel with BLP instruments and product
    characteristics for the final merge (same as estimation_1_demand_2_secondprep)."""
    print(f"Loading selective instrument panel from {panel_csv}...")
    available = pd.read_csv(panel_csv, nrows=0).columns.tolist()
    cols_to_load = [c for c in MP_KEEP_COLS if c in available]
    df = pd.read_csv(panel_csv, usecols=cols_to_load,
                     dtype={'mca_code': str, 'CODMUN_IBGE': str})
    df['is_B'] = (df['CODMUN_IBGE'] != '0')
    id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
    df = df.drop_duplicates(subset=id_vars)

    records = []
    for k in K_LIST:
        dep_col = f'dep_a{k}'
        spr_col = f'spread_a{k}'
        if dep_col not in df.columns:
            continue
        chunk = df.copy()
        chunk['deposit_type'] = k
        chunk['deposit_balance'] = chunk[dep_col]
        chunk['spread_qoq'] = (chunk[spr_col]
                               if spr_col in chunk.columns else np.nan)
        records.append(chunk)

    df_long = pd.concat(records, ignore_index=True)
    drop_cols = [c for c in df_long.columns
                 if c.startswith('dep_a') or c.startswith('spread_a')]
    df_long = df_long.drop(columns=drop_cols, errors='ignore')

    df_long['entity_id'] = (df_long['CodConglomeradoPrudencial'].astype(str)
                            + "_" + df_long['deposit_type'].astype(str)
                            + "_" + df_long['mca_code'].astype(str))
    df_long['time_id'] = (df_long['year'].astype(str) + "Q"
                          + df_long['quarter'].astype(str))

    print(f"  Panel loaded: {len(df_long)} rows, "
          f"{df_long['entity_id'].nunique()} entities")
    return df_long


# ==============================================================================
# 2. PCA Index Construction (for D-type firms, specs 5–12)
# ==============================================================================
def build_pca_index(df: pd.DataFrame, s_cols: list) -> dict:
    """Reconstruct the PCA macro-tech index from state variables, replicating
    the logic in estimation_2_sleep_1_cfa.py run_model_option3().

    Returns a dict mapping year_quarter -> PCA index value.
    """
    sv_dynamic = [sv for sv in s_cols if sv != 'constant' and sv in df.columns]
    if not sv_dynamic:
        return {}

    df_pca = df.dropna(subset=sv_dynamic).copy()
    ts_df = (df_pca[['year_quarter'] + sv_dynamic]
             .drop_duplicates()
             .sort_values('year_quarter'))
    ts_vals = ts_df[sv_dynamic].values

    # Standardize
    ts_mean = np.mean(ts_vals, axis=0)
    ts_std = np.std(ts_vals, axis=0)
    ts_std[ts_std == 0] = 1  # prevent div/0

    ts_norm = (ts_vals - ts_mean) / ts_std
    pca = PCA(n_components=1)
    ts_pca = pca.fit_transform(ts_norm)
    ts_df = ts_df.copy()
    ts_df['pca_index'] = ts_pca[:, 0]

    return ts_df.set_index('year_quarter')['pca_index'].to_dict()


# ==============================================================================
# 3. Helpers for Phi Extraction
# ==============================================================================
def extract_upsilon_terms(res_ss) -> dict:
    """Extract mapped state variable names and their Upsilon coefficients
    from the statsmodels RegressionResults object.

    'nr_lagged_dep' -> 'constant' coefficient
    'interaction_{X}' -> 'X' coefficient
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


def compute_pca_phi_t(nat_res_ss, pca_dict: dict,
                      year_quarters: np.ndarray) -> pd.Series:
    """Compute phi_t^{Nat,3} for each year_quarter using the PCA Option 3
    model parameters and the reconstructed PCA index.

    The Option 3 model has two coefficients:
      nr_lagged_dep          -> baseline phi (constant component)
      interaction_pca_index  -> PCA interaction component

    phi_t = beta_0 * 1 + beta_1 * PCA_index_t
    """
    params = nat_res_ss.params
    beta_0 = params.get('nr_lagged_dep', 0.0)
    beta_1 = params.get('interaction_pca_index', 0.0)

    phi_values = pd.Series(np.nan, index=range(len(year_quarters)))
    for i, yq in enumerate(year_quarters):
        if yq in pca_dict:
            phi_values.iloc[i] = beta_0 + beta_1 * pca_dict[yq]
        else:
            # Fallback: use baseline only
            phi_values.iloc[i] = beta_0

    return phi_values


# ==============================================================================
# 4. Per-Specification Processing
# ==============================================================================
def process_specification(spec_id: int, df_base: pd.DataFrame,
                          local_results: dict, national_results: dict,
                          pca_dicts: dict) -> tuple:
    """Process a single specification: compute hybrid phi, active deposits,
    and data-implied shares.

    For specs (1)–(4) [Base block]:
      - B-type: local phi_mt from estimation_1
      - D-type: population-weighted local aggregate phi_t from estimation_1

    For specs (5)–(12) [Macro / Tech blocks]:
      - B-type: local phi_mt from estimation_1
      - D-type: PCA Option 3 phi_t^{Nat,3} from estimation_2

    Returns (df_spec, summary_dict, spec_id) or (None, None, spec_id).
    """
    spec_key = SPEC_ID_TO_KEYS.get(spec_id)
    if spec_key is None:
        print(f"  [!] Unknown spec_id {spec_id}. Skipping.")
        return None, None, spec_id

    iv_name, state_block = spec_key.split(' x ')
    s_cols = STATE_BLOCKS.get(state_block, S_BASE)
    use_pca_for_d = spec_id >= 5  # Macro and Tech blocks

    # ----- Extract B-type coefficients from local (estimation_1) results -----
    local_entry = local_results.get(spec_key)
    if local_entry is None:
        print(f"  [!] No local result for '{spec_key}'. Skipping.")
        return None, None, spec_id

    res_ss_local = local_entry.get('second_stage')
    if res_ss_local is None:
        print(f"  [!] No second_stage result for '{spec_key}'. Skipping.")
        return None, None, spec_id

    upsilon = extract_upsilon_terms(res_ss_local)

    # Working copy
    keep_cols = ['entity_id', 'time_id', 'CodConglomeradoPrudencial',
                 'mca_code', 'year', 'quarter', 'deposit_type', 'is_B',
                 'deposit_balance', 'lagged_deposits', 'gross_return_lag',
                 'pop_total', 'year_quarter']
    df_spec = df_base[keep_cols].copy()

    # ----- Compute phi_mt (local, for B-type, and also for D-type in 1–4) ---
    df_spec['phi_mt'] = 0.0
    missing_sv = False
    for sv_name, beta in upsilon.items():
        if sv_name not in df_base.columns:
            missing_sv = True
            break
        df_spec['phi_mt'] += beta * df_base[sv_name].values

    if missing_sv:
        print(f"  [!] Missing state variable for spec {spec_id}. Skipping.")
        return None, None, spec_id

    df_spec = df_spec.dropna(subset=['phi_mt'])
    df_spec['phi_mt'] = df_spec['phi_mt'].clip(lower=0.0, upper=1.0)

    # ----- Compute phi_t for D-type firms -----------------------------------
    if use_pca_for_d:
        # PCA Option 3 from estimation_2 national results
        nat_key = f"Option_3_{iv_name}_{state_block}"
        nat_entry = national_results.get(nat_key)

        if nat_entry is None or nat_entry.get('second_stage') is None:
            print(f"  [WARN] No national Option 3 result for '{nat_key}'. "
                  f"Falling back to local aggregate for D-type.")
            use_pca_for_d = False

    if use_pca_for_d:
        # Compute phi_t from PCA model
        res_ss_nat = nat_entry['second_stage']
        pca_dict = pca_dicts.get(state_block, {})
        yq_arr = df_spec['year_quarter'].values

        phi_t_pca = compute_pca_phi_t(res_ss_nat, pca_dict, yq_arr)
        df_spec['phi_t'] = phi_t_pca.values
        df_spec['phi_t'] = df_spec['phi_t'].clip(lower=0.0, upper=1.0)
        print(f"  Spec {spec_id}: D-type phi_t from PCA Option 3 "
              f"(mean={df_spec['phi_t'].mean():.4f})")
    else:
        # Population-weighted local aggregate (identical to estimation_1)
        df_mca = (df_spec[['mca_code', 'time_id', 'phi_mt', 'pop_total']]
                  .drop_duplicates())
        df_mca = df_mca.dropna(subset=['pop_total'])

        def weighted_mean(g):
            v, w = g['phi_mt'], g['pop_total']
            return v.mean() if w.sum() == 0 else np.average(v, weights=w)

        phi_t_map = (df_mca.groupby('time_id')
                     .apply(weighted_mean, include_groups=False)
                     .rename("phi_t"))
        df_spec = df_spec.join(phi_t_map, on='time_id')
        df_spec['phi_t'] = df_spec['phi_t'].fillna(0.0).clip(lower=0.0,
                                                              upper=1.0)
        print(f"  Spec {spec_id}: D-type phi_t from local aggregate "
              f"(mean={df_spec['phi_t'].mean():.4f})")

    # ----- Active Deposits (Eq-15) ------------------------------------------
    df_spec['Dep_Act'] = 0.0

    # B-type: uses local phi_mt
    val_B = (df_spec['deposit_balance']
             - df_spec['phi_mt'] * df_spec['gross_return_lag']
             * df_spec['lagged_deposits'])
    df_spec.loc[df_spec['is_B'], 'Dep_Act'] = np.maximum(
        0.0, val_B[df_spec['is_B']])

    # D-type: uses phi_t (PCA or local aggregate depending on spec)
    val_D = (df_spec['deposit_balance']
             - df_spec['phi_t'] * df_spec['gross_return_lag']
             * df_spec['lagged_deposits'])
    df_spec.loc[~df_spec['is_B'], 'Dep_Act'] = np.maximum(
        0.0, val_D[~df_spec['is_B']])

    # ----- Data-Implied Shares (Eq-16) --------------------------------------
    nat_active = df_spec.groupby(
        ['time_id', 'deposit_type'])['Dep_Act'].transform('sum')
    df_spec['nat_active'] = nat_active
    df_spec['share_D'] = np.where(~df_spec['is_B'],
                                  df_spec['Dep_Act'] / nat_active, np.nan)

    local_B_active = (df_spec[df_spec['is_B']]
                      .groupby(['mca_code', 'time_id', 'deposit_type'])
                      ['Dep_Act'].transform('sum'))

    df_loc_map = df_spec[df_spec['is_B']][['entity_id', 'time_id']].copy()
    df_loc_map['local_B_active'] = local_B_active
    df_spec = pd.merge(df_spec, df_loc_map,
                       on=['entity_id', 'time_id'], how='left')

    mask_valid_denom = (df_spec['is_B']) & (df_spec['local_B_active'] > 0)
    df_spec['share_B_cond'] = np.where(
        mask_valid_denom,
        df_spec['Dep_Act'] / df_spec['local_B_active'], np.nan)
    df_spec.loc[(df_spec['is_B']) & ~mask_valid_denom, 'share_B_cond'] = 0.0

    df_spec['Spec_ID'] = spec_id

    # Summary
    b_count = df_spec['is_B'].sum()
    d_count = (~df_spec['is_B']).sum()
    summary = {
        "Name": spec_key,
        "Spec_ID": spec_id,
        "Rows": len(df_spec),
        "B_firms": int(b_count),
        "D_firms": int(d_count),
        "Mean_phi_mt": float(df_spec['phi_mt'].mean()),
        "Mean_phi_t": float(df_spec['phi_t'].mean()),
        "D_phi_source": "PCA_Option3" if (spec_id >= 5) else "Local_Aggregate",
        "B_Cond_Share_NaNs": int(
            df_spec[df_spec['is_B']]['share_B_cond'].isna().sum()),
        "D_Share_NaNs": int(
            df_spec[~df_spec['is_B']]['share_D'].isna().sum())
    }

    return df_spec, summary, spec_id


# ==============================================================================
# 5. Panel Merge & Serialization
# ==============================================================================
def merge_panel_with_prep(df_panel: pd.DataFrame,
                          df_prep: pd.DataFrame) -> pd.DataFrame:
    """Merge demand-prep output with the selective instrument panel,
    replicating estimation_1_demand_2_secondprep.py merge logic."""
    prep_cols = ['entity_id', 'time_id', 'Dep_Act', 'share_D',
                 'share_B_cond', 'phi_mt', 'phi_t', 'is_B',
                 'local_B_active', 'nat_active']
    df_prep_slim = df_prep[prep_cols].copy()

    panel_merge_cols = (
        ['entity_id', 'time_id', 'CodConglomeradoPrudencial',
         'mca_code', 'deposit_type', 'spread_qoq']
        + [c for c in X_COLS if c in df_panel.columns]
        + [c for c in D_COLS if c in df_panel.columns]
        + ['pop_total']
        + [c for c in IV_BLP_LOO if c in df_panel.columns]
        + [c for c in IV_COST if c in df_panel.columns]
        + [c for c in IV_CAPITAL if c in df_panel.columns]
    )
    panel_merge_cols = list(set(panel_merge_cols))
    df_panel_slim = df_panel[
        [c for c in panel_merge_cols if c in df_panel.columns]].copy()

    df = pd.merge(df_prep_slim, df_panel_slim,
                  on=['entity_id', 'time_id'], how='inner')
    df = df.dropna(subset=['spread_qoq', 'Dep_Act'])
    return df


# ==============================================================================
# 6. CLI & Main
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid demand prep: B-type local phi + D-type PCA phi")
    parser.add_argument('--spec', type=str, default='all',
                        help='Specification ID (1-12) or "all"')
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which specs to process
    if args.spec.lower() == 'all':
        spec_ids = list(range(1, 13))
    elif '-' in args.spec:
        start, end = map(int, args.spec.split('-'))
        spec_ids = list(range(start, end + 1))
    else:
        spec_ids = [int(args.spec)]

    # ----- Load estimation results pickles ----------------------------------
    local_pkl = SLEEP_LOCAL_DIR / "estimation_results.pkl"
    national_pkl = SLEEP_NATIONAL_DIR / "estimation_results.pkl"

    if not local_pkl.exists():
        print(f"ERROR: Missing local results at {local_pkl}. "
              f"Run estimation_1_sleep_1_cfa.py first.")
        sys.exit(1)

    if not national_pkl.exists():
        print(f"ERROR: Missing national results at {national_pkl}. "
              f"Run estimation_2_sleep_1_cfa.py first.")
        sys.exit(1)

    print(f"Loading local sleepiness results from {local_pkl.name}...")
    with open(local_pkl, 'rb') as f:
        local_results = pickle.load(f)

    print(f"Loading national sleepiness results from {national_pkl.name}...")
    with open(national_pkl, 'rb') as f:
        national_results = pickle.load(f)

    # ----- Build panels -----------------------------------------------------
    df_base = build_base_panel(PANEL_CSV)
    print(f"Base panel: {len(df_base)} rows")

    df_panel = load_panel_selective(PANEL_CSV)

    # ----- Pre-compute PCA indices for Macro and Tech blocks ----------------
    print("Pre-computing PCA indices for Macro and Tech state blocks...")
    pca_dicts = {}
    for block_name in ['Macro', 'Tech']:
        s_cols = STATE_BLOCKS[block_name]
        pca_dicts[block_name] = build_pca_index(df_base, s_cols)
        n_quarters = len(pca_dicts[block_name])
        print(f"  {block_name}: {n_quarters} year-quarters indexed")

    # ----- Process each specification ---------------------------------------
    DEMAND_PREP_DIR.mkdir(parents=True, exist_ok=True)
    spec_summaries = {}
    n_saved = 0

    for spec_id in spec_ids:
        print(f"\n--- Specification {spec_id} ---")
        df_spec, summary, sid = process_specification(
            spec_id, df_base, local_results, national_results, pca_dicts)

        if df_spec is None:
            print(f"  [!] Failed or skipped spec {spec_id}.")
            continue

        # Merge with instrument panel
        df_final = merge_panel_with_prep(df_panel, df_spec)

        # Serialize
        out_path = DEMAND_PREP_DIR / f"demand_2_final_spec_{spec_id}.pkl"
        df_final.to_pickle(out_path)
        print(f"  -> Saved {out_path.name} with {len(df_final)} rows")

        spec_summaries[str(spec_id)] = summary
        n_saved += 1

    # ----- Summary JSON -----------------------------------------------------
    if n_saved == 0:
        print("\nNo valid specifications were processed.")
        sys.exit(1)

    out_json = DEMAND_PREP_DIR / "demand_prep_2_summary.json"
    with open(out_json, "w") as f:
        json.dump(spec_summaries, f, indent=4)

    print(f"\nSummary saved to: {out_json}")
    print(f"[SUCCESS] {n_saved} per-spec PKLs written to {DEMAND_PREP_DIR}.")


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    main()
