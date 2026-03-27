"""
estimation_1_demand_2_secondprep.py
====================================
Data aggregating process to output ready-to-run dataframes for BLP loops.
Reads market_panel.csv and demand_prep_spec_{id}.csv, then merges them
and saves the final merged panel as demand_final_spec_{id}.pkl.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Use the exact same constants and column definitions
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
PANEL_CSV = DATA_DIR / "market_panel.csv"
DEMAND_PREP_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP"

X_COLS = ['fgc_covered', 'has_ip', 'seg_S2', 'seg_S3', 'seg_S4', 'seg_S5',
          'log_total_assets_lag', 'equity_ratio_lag']
L_PROD = len(X_COLS)
K_TYPES = 4
K_LIST = [1, 2, 4, 5]

D_COLS = ['gdp_per_capita', 'fraction_65plus', 'fraction_young',
          'pix_users_pf_per1000', 'connections_per100', 'frac_4g5g',
          'branches_per1000', 'cadunico_families_per1000']
D_DIM = len(D_COLS)

IV_BLP_LOO = ['loo_log_assets', 'mean_loo_log_assets',
              'loo_equity_assets', 'mean_loo_equity_ratio',
              'loo_indice_basileia', 'mean_loo_basileia',
              'loo_credit_assets', 'mean_loo_credit_assets',
              'loo_npl_provision', 'mean_loo_npl_provision',
              'n_rivals']
IV_COST = ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag']
IV_CAPITAL = ['indice_basileia_lag']

MP_KEEP_COLS = (
    ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter', 'CODMUN_IBGE']
    + [f'spread_a{k}' for k in K_LIST]
    + [f'dep_a{k}' for k in K_LIST]
    + X_COLS + D_COLS + ['pop_total']
    + IV_BLP_LOO + IV_COST + IV_CAPITAL
    + ['segment']
)

def load_panel_selective(panel_csv: Path) -> pd.DataFrame:
    print(f"Loading selective columns from {panel_csv}...")
    available = pd.read_csv(panel_csv, nrows=0).columns.tolist()
    cols_to_load = [c for c in MP_KEEP_COLS if c in available]
    df = pd.read_csv(panel_csv, usecols=cols_to_load, dtype={'mca_code': str, 'CODMUN_IBGE': str})
    df['is_B'] = (df['CODMUN_IBGE'] != '0')
    id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
    df = df.drop_duplicates(subset=id_vars)

    records = []
    for k in K_LIST:
        dep_col = f'dep_a{k}'
        spr_col = f'spread_a{k}'
        if dep_col not in df.columns: continue
        chunk = df.copy()
        chunk['deposit_type'] = k
        chunk['deposit_balance'] = chunk[dep_col]
        chunk['spread_qoq'] = chunk[spr_col] if spr_col in chunk.columns else np.nan
        records.append(chunk)

    df_long = pd.concat(records, ignore_index=True)
    drop_cols = [c for c in df_long.columns if c.startswith('dep_a') or c.startswith('spread_a')]
    df_long = df_long.drop(columns=drop_cols, errors='ignore')

    df_long['entity_id'] = (df_long['CodConglomeradoPrudencial'].astype(str) + "_"
                            + df_long['deposit_type'].astype(str) + "_"
                            + df_long['mca_code'].astype(str))
    df_long['time_id'] = df_long['year'].astype(str) + "Q" + df_long['quarter'].astype(str)

    print(f"  Panel loaded: {len(df_long)} rows, {df_long['entity_id'].nunique()} entities")
    return df_long

def load_demand_prep(spec_id: int) -> pd.DataFrame:
    csv_path = DEMAND_PREP_DIR / f"demand_prep_spec_{spec_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    return pd.read_csv(csv_path, dtype={'mca_code': str})

def merge_panel_with_prep(df_panel: pd.DataFrame, df_prep: pd.DataFrame) -> pd.DataFrame:
    prep_cols = ['entity_id', 'time_id', 'Dep_Act', 'share_NB', 'share_B_cond',
                 'phi_mt', 'phi_t', 'is_B', 'local_B_active', 'nat_active']
    df_prep_slim = df_prep[prep_cols].copy()

    panel_merge_cols = (['entity_id', 'time_id', 'CodConglomeradoPrudencial',
                         'mca_code', 'deposit_type', 'spread_qoq']
                        + [c for c in X_COLS if c in df_panel.columns]
                        + [c for c in D_COLS if c in df_panel.columns]
                        + ['pop_total']
                        + [c for c in IV_BLP_LOO if c in df_panel.columns]
                        + [c for c in IV_COST if c in df_panel.columns]
                        + [c for c in IV_CAPITAL if c in df_panel.columns])
    panel_merge_cols = list(set(panel_merge_cols))
    df_panel_slim = df_panel[[c for c in panel_merge_cols if c in df_panel.columns]].copy()

    df = pd.merge(df_prep_slim, df_panel_slim, on=['entity_id', 'time_id'], how='inner')
    df = df.dropna(subset=['spread_qoq', 'Dep_Act'])
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=str, default='12', help='Specification ID (1-12) or "all"')
    args = parser.parse_args()

    if args.spec.lower() == 'all':
        import re
        spec_ids = []
        if DEMAND_PREP_DIR.exists():
            for f in DEMAND_PREP_DIR.glob("demand_prep_spec_*.csv"):
                if m := re.search(r'demand_prep_spec_(\d+)\.csv', f.name):
                    spec_ids.append(int(m.group(1)))
        spec_ids.sort()
        if not spec_ids:
            spec_ids = list(range(1, 13))
    elif '-' in args.spec:
        start, end = map(int, args.spec.split('-'))
        spec_ids = list(range(start, end + 1))
    else:
        spec_ids = [int(args.spec)]

    DEMAND_PREP_DIR.mkdir(parents=True, exist_ok=True)
    df_panel = load_panel_selective(PANEL_CSV)

    for spec_id in spec_ids:
        print(f"\nProcessing Specification {spec_id}...")
        try:
            df_prep = load_demand_prep(spec_id)
            df_final = merge_panel_with_prep(df_panel, df_prep)
            out_path = DEMAND_PREP_DIR / f"demand_final_spec_{spec_id}.pkl"
            df_final.to_pickle(out_path)
            print(f"  -> Saved {out_path.name} with {len(df_final)} rows")
        except FileNotFoundError:
            print(f"  -> [!] Missing prep CSV for spec {spec_id}. Skipping.")

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    main()
