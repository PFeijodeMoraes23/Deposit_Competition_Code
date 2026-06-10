"""
estimation_3_demand_1_prep.py
================================================================================
Prepares demand-side variables for the BLP step (Eq-15 and Eq-16 in V_Main.tex).

Estimation strategy 3: Pooled B+D Logistic AME — sleep model estimated jointly
on B-type (branch-based) and D-type (digital/national) institutions using a
logistic specification; coefficients are Average Marginal Effects.  The inverse-
logit (sigmoid) transformation is applied when building phi_mt so the fraction
remains in [0, 1].  D-type rows receive the market-level aggregate phi_t.

Reads the output of `estimation_3_sleep.py` (12 specifications for phi_mt) and
the market panel.  For each specification it computes phi_mt (with sigmoid),
aggregates to the national phi_t, computes Active Deposits (Dep^Act), and builds
data-implied conditional market shares for B-type and D-type institutions.

``has_ip`` and ``fgc_covered`` are expected to be pre-computed in the panel
pipeline (panel_5_flag_digital.py / panel_6_market.py) and read from the CSV.

Usage
-----
  python estimation_3_demand_1_prep.py --spec all
  python estimation_3_demand_1_prep.py --spec 12

CLI Options:
------------
  --spec SPEC  Specification ID (1-12) or "all"
"""

import os
import sys
import logging
import json
import pickle
import argparse
import re
import unicodedata
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
from utils import paths
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = paths.PROCESSED
_PANEL_WITH_FEES = DATA_DIR / "market_panel_with_fees.csv"
PANEL_CSV = _PANEL_WITH_FEES if _PANEL_WITH_FEES.exists() else DATA_DIR / "market_panel.csv"
BANKED_CSV = paths.INCLUSION_DIR / "bcb_banked_mca_panel.csv"

X_COLS = ['fgc_covered', 'has_ip', 'seg_S2', 'seg_S3', 'seg_S4', 'seg_S5',
          'log_total_assets_lag', 'equity_ratio_lag']
D_COLS = ['gdp_per_capita', 'fraction_65plus', 'fraction_young',
          'pix_users_pf_per1000', 'connections_per100', 'frac_4g5g',
          'branches_per1000', 'cadunico_families_per1000']
IV_BLP_LOO = ['loo_log_assets', 'mean_loo_log_assets',
              'loo_equity_ratio', 'mean_loo_equity_ratio',
              'loo_basileia', 'mean_loo_basileia',
              'loo_credit_assets', 'mean_loo_credit_assets',
              'loo_npl_provision', 'mean_loo_npl_provision',
              'n_rivals']
IV_COST = ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag']
IV_CAPITAL = ['indice_basileia_lag']
IV_FEE = [
    'cosif_fee_ratio_all',
    'cosif_fee_ratio_total_deposits',
    'cosif_fee_valid',
    'listed_fee_atm_withdrawal_pf',
    'listed_fee_statement_pf',
    'tarifa_stickiness_yrs',
    'tarifa_stickiness_n',
]
EXTRA_KEEP_COLS = (X_COLS + D_COLS + IV_BLP_LOO + IV_COST + IV_CAPITAL + IV_FEE
                   + ['segment', 'spread_qoq', 'spread_ann'])

def _resolve_runtime_paths() -> tuple[Path, Path, Path]:
    panel_csv = PANEL_CSV
    sleep_output_dir = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "est3"
    demand_output_dir = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP"
    return panel_csv, sleep_output_dir, demand_output_dir

class NonLinearResults:
    def __init__(self, params, bse, tvalues, pvalues, G_star, params_native=None):
        self.params = params
        self.bse = bse
        self.tvalues = tvalues
        self.pvalues = pvalues
        self.G_star = G_star
        self.df_resid = G_star

sys.modules['estimation_3_sleep'] = type('FakeModule', (), {'NonLinearResults': NonLinearResults})

def extract_upsilon_terms(res_ss):
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
    df = pd.wide_to_long(df_raw, stubnames=['dep_a', 'spread_a', 'spread_ann_a'], i=id_vars, j='deposit_type').reset_index()
    return df.rename(columns={'dep_a': 'deposit_balance', 'spread_a': 'spread_qoq', 'spread_ann_a': 'spread_ann'})

def _reshape_panel(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw['mca_code'] = df_raw['mca_code'].astype(str)
    if 'dep_a1' in df_raw.columns: return _reshape_panel_from_wide(df_raw)
    df = df_raw.copy()
    if 'is_B' not in df.columns: df['is_B'] = (df['CODMUN_IBGE'].astype(str) != '0')
    if 'deposit_type' in df.columns: df = df[df['deposit_type'] != 3].copy()
    pre_k5_mask = (df['deposit_type'] == 5) & ((df['year'] < 2020) | ((df['year'] == 2020) & (df['quarter'] < 2)))
    if pre_k5_mask.any():
        logging.info(f"  Dropped {pre_k5_mask.sum()} k=5 rows before 2020Q2 (product did not exist).")
        df = df[~pre_k5_mask].copy()
    return df

def build_base_panel(panel_csv: Path) -> pd.DataFrame:
    print(f"Loading {panel_csv}...")
    df_raw = pd.read_csv(panel_csv, dtype={'mca_code': str}, low_memory=False)
    df = _reshape_panel(df_raw)

    # fgc_covered: FGC insures types 1 (savings), 2 (demand), 4 (time deposits).
    # Type 5 (prepaid) is NOT FGC-covered (BCB Res. 80/2021; CMN 4.678/2018).
    # The market panel is wide (no deposit_type column), so compute here post-reshape.
    df['fgc_covered'] = df['deposit_type'].astype('Int64').isin([1, 2, 4]).astype(int)

    # has_ip comes from bank_chars_panel.csv (panel_4_bank_chars.py reads IF-Data List files).
    if 'has_ip' not in df.columns:
        logging.warning(
            "Column 'has_ip' not found in panel CSV. "
            "Re-run panel_4_bank_chars.py to derive it from IF-Data List files. Defaulting to 0."
        )
        df['has_ip'] = 0

    df['entity_id'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['deposit_type'].astype(str) + "_" + df['mca_code'].astype(str)
    df['time_id'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)
    df.sort_values(by=['entity_id', 'year', 'quarter'], inplace=True)

    df['spread_qoq_lag'] = df.groupby('entity_id')['spread_qoq'].shift(1)
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)
    df['lagged_deposits'] = df.groupby('entity_id')['deposit_balance'].shift(1)

    df['deposit_rate_lag'] = df['risk_free_qoq_lag'] - df['spread_qoq_lag']
    df['gross_return_lag'] = 1 + df['deposit_rate_lag']

    df['spread_qoq'] = df['spread_qoq'] * 10_000
    df['spread_ann'] = df['spread_ann'] * 10_000

    df = df.dropna(subset=['deposit_balance', 'lagged_deposits', 'spread_qoq', 'spread_ann', 'entity_id', 'time_id'])

    df['constant'] = 1.0
    if 'year' in df.columns:
        df['post_2020'] = (df['year'] >= 2020).astype(int)
        df['pix_exists'] = ((df['year'] > 2020) | ((df['year'] == 2020) & (df['quarter'] == 4))).astype(float)

    if 'CODMUN_IBGE' in df.columns:
        df['dummy_D_type'] = (df['CODMUN_IBGE'].astype(str) == '0').astype(float)

    if 'is_coop' in df.columns:
        df['is_coop'] = df['is_coop'].fillna(0.0)
    else:
        df['is_coop'] = 0.0

    if 'is_state_owned' in df.columns:
        df['is_state_owned'] = df['is_state_owned'].fillna(0.0)
    else:
        df['is_state_owned'] = 0.0

    if 'dummy_D_type' in df.columns:
        _new_cols = {}
        if 'risk_free_qoq_lag' in df.columns:
            _new_cols['dummy_D_type_x_risk_free_qoq_lag'] = df['dummy_D_type'] * df['risk_free_qoq_lag']
        if 'fraction_65plus' in df.columns:
            _new_cols['dummy_D_type_x_fraction_65plus'] = df['dummy_D_type'] * df['fraction_65plus']
        if 'fraction_young' in df.columns:
            _new_cols['dummy_D_type_x_fraction_young'] = df['dummy_D_type'] * df['fraction_young']
        if _new_cols:
            df = pd.concat([df, pd.DataFrame(_new_cols, index=df.index)], axis=1)

    if 'gdp_per_capita' in df.columns: df['gdp_per_capita'] /= 10000.0
    if 'cadunico_families_per1000' in df.columns: df['cadunico_families_per1000'] /= 100.0
    if 'pix_users_pf_per1000' in df.columns: df['pix_users_pf_per1000'] /= 100.0
    if 'connections_per100' in df.columns: df['connections_per100'] /= 100.0

    s_tech_finance = ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']
    for col in s_tech_finance:
        if col in df.columns: df[col] = df[col].fillna(df[col].median())

    if 'pop_total' not in df.columns: df['pop_total'] = np.nan

    cosif_ratio_cols = [c for c in df.columns if c.startswith('cosif_fee_ratio')]
    if cosif_ratio_cols and 'cosif_fee_valid' in df.columns:
        invalid = df['cosif_fee_valid'] == 0
        if invalid.any():
            df.loc[invalid, cosif_ratio_cols] = np.nan
            logging.info("  NaN-ed %d rows of cosif_fee_ratio_* (cosif_fee_valid=0, year>=2025).",
                         invalid.sum())

    if BANKED_CSV.exists():
        banked = pd.read_csv(
            BANKED_CSV,
            usecols=['mca_code', 'year', 'banked_correction'],
            dtype={'mca_code': str, 'year': int},
        )
        banked['banked_correction'] = pd.to_numeric(banked['banked_correction'], errors='coerce')
        df = df.merge(banked, on=['mca_code', 'year'], how='left')
    else:
        logging.warning(f"Banked correction panel not found at {BANKED_CSV}; fallback 1.1 used.")
        df['banked_correction'] = np.nan

    return df

def process_specification(args):
    spec_name, spec_res, df_base = args
    import numpy as np
    import pandas as pd

    res_ss = spec_res.get('second_stage')
    if res_ss is None: return None, None, spec_name

    upsilon = extract_upsilon_terms(res_ss)

    base_cols = ['entity_id', 'time_id', 'CodConglomeradoPrudencial', 'mca_code',
                 'year', 'quarter', 'deposit_type', 'is_B',
                 'deposit_balance', 'lagged_deposits', 'gross_return_lag', 'pop_total']

    keep_cols = list(set(base_cols + [c for c in EXTRA_KEEP_COLS + ['banked_correction'] if c in df_base.columns]))
    df_spec = df_base[keep_cols].copy()
    if 'is_B' in df_spec.columns: df_spec['is_B'] = df_spec['is_B'].astype(bool)

    df_spec['phi_mt'] = 0.0
    missing_sv = False

    for sv_name, beta in upsilon.items():
        if sv_name not in df_base.columns:
            if sv_name == 'is_coop': df_spec[sv_name] = df_base.get('is_coop', 0.0)
            elif sv_name == 'is_state_owned': df_spec[sv_name] = df_base.get('is_state_owned', 0.0)
            elif '_x_' in sv_name:
                parts = sv_name.split('_x_')
                p1, p2 = parts[0], parts[1]
                v1 = df_spec[p1] if p1 in df_spec.columns else df_base.get(p1, None)
                v2 = df_spec[p2] if p2 in df_spec.columns else df_base.get(p2, None)
                if v1 is not None and v2 is not None:
                    df_spec[sv_name] = v1 * v2
                else:
                    missing_sv = True
                    break
            else:
                missing_sv = True
                break
        else:
            if sv_name not in df_spec.columns: df_spec[sv_name] = df_base[sv_name]

        df_spec['phi_mt'] += beta * df_spec[sv_name]
    if missing_sv: return None, None, spec_name

    df_spec = df_spec.dropna(subset=['phi_mt', 'spread_qoq'])

    # Est3 is logistic: always apply the inverse-logit transformation.
    df_spec['phi_mt'] = 1.0 / (1.0 + np.exp(-df_spec['phi_mt'].astype(float)))

    df_spec['phi_mt'] = df_spec['phi_mt'].clip(lower=0.0, upper=1.0)

    df_mca_level = df_spec[['mca_code', 'time_id', 'phi_mt', 'pop_total']].drop_duplicates()
    df_mca_level = df_mca_level.dropna(subset=['pop_total'])

    def weighted_mean(g):
        v, w = g['phi_mt'], g['pop_total']
        return v.mean() if w.sum() == 0 else np.average(v, weights=w)

    phi_t_map = df_mca_level.groupby('time_id').apply(weighted_mean, include_groups=False).rename("phi_t")
    df_spec = df_spec.join(phi_t_map, on='time_id')
    df_spec['phi_t'] = df_spec['phi_t'].fillna(0.0).clip(lower=0.0, upper=1.0)

    df_spec['Dep_Act'] = 0.0
    val_B = df_spec['deposit_balance'] - df_spec['phi_mt'] * df_spec['gross_return_lag'] * df_spec['lagged_deposits']
    df_spec.loc[df_spec['is_B'], 'Dep_Act'] = np.maximum(0.0, val_B[df_spec['is_B']]).astype(float).values

    val_D = df_spec['deposit_balance'] - df_spec['phi_t'] * df_spec['gross_return_lag'] * df_spec['lagged_deposits']
    df_spec.loc[~df_spec['is_B'], 'Dep_Act'] = np.maximum(0.0, val_D[~df_spec['is_B']]).astype(float).values

    df_spec = df_spec.dropna(subset=['Dep_Act'])
    df_spec = df_spec[df_spec['Dep_Act'] > 1e-6]

    FALLBACK_BC = 1.1
    df_spec['_bc'] = df_spec['banked_correction'].fillna(FALLBACK_BC)

    mkt = df_spec.groupby(['mca_code', 'time_id']).agg(
        _pop=('pop_total', 'first'),
        _bc_mt=('_bc', 'first'),
    )

    b_dep_mt = (df_spec[df_spec['is_B']]
                .groupby(['mca_code', 'time_id'])['Dep_Act'].sum()
                .rename('_dep_B'))
    mkt = mkt.join(b_dep_mt, how='left').fillna({'_dep_B': 0.0})
    local_ratio = (mkt['_dep_B'] / mkt['_pop']).replace([np.inf, -np.inf], np.nan)
    max_local = local_ratio.max() if local_ratio.notna().any() else 0.0

    all_dep_t = df_spec.groupby('time_id')['Dep_Act'].sum()
    nat_pop_t = mkt.groupby('time_id')['_pop'].sum()
    nat_ratio = (all_dep_t / nat_pop_t).replace([np.inf, -np.inf], np.nan)
    max_nat = nat_ratio.max() if nat_ratio.notna().any() else 0.0

    max_ratio = max(max_local, max_nat)
    if max_ratio <= 0:
        max_ratio = 1.0

    mkt['_b_mkt'] = mkt['_bc_mt'] * max_ratio * mkt['_pop']
    df_spec = df_spec.merge(
        mkt[['_b_mkt']].reset_index(),
        on=['mca_code', 'time_id'], how='left',
    )

    mkt_r = mkt.reset_index()
    mkt_r['_w'] = mkt_r['_bc_mt'] * mkt_r['_pop']
    nat_dbar = (
        mkt_r.groupby('time_id')
             .apply(lambda g: (g['_w'].sum() / g['_pop'].sum()) * max_ratio
                    if g['_pop'].sum() > 0 else FALLBACK_BC * max_ratio,
                    include_groups=False)
             .rename('_dbar_nat')
    )
    nat_pop = mkt_r.groupby('time_id')['_pop'].sum().rename('_pop_nat')
    d_mkt = (nat_dbar * nat_pop).rename('_d_mkt').reset_index()
    df_spec = df_spec.merge(d_mkt, on='time_id', how='left')

    df_spec['share_B_cond'] = np.where(
        df_spec['is_B'],
        df_spec['Dep_Act'] / df_spec['_b_mkt'],
        np.nan,
    )
    df_spec['share_D'] = np.where(
        ~df_spec['is_B'],
        df_spec['Dep_Act'] / df_spec['_d_mkt'],
        np.nan,
    )
    df_spec.drop(columns=['_b_mkt', '_d_mkt', '_bc'], inplace=True)

    SPEC_MAP = {
        'OLS x Base': 1, 'IV_CostShifters x Base': 2, 'IV_Wholesale x Base': 3, 'IV_HausmanFull x Base': 4,
        'OLS x Macro': 5, 'IV_CostShifters x Macro': 6, 'IV_Wholesale x Macro': 7, 'IV_HausmanFull x Macro': 8,
        'OLS x Tech': 9, 'IV_CostShifters x Tech': 10, 'IV_Wholesale x Tech': 11, 'IV_HausmanFull x Tech': 12
    }
    spec_id = SPEC_MAP.get(spec_name, spec_name)
    df_spec['Spec_ID'] = spec_id

    b_count = df_spec['is_B'].sum()
    d_count = (~df_spec['is_B']).sum()
    summary = {
        "Name": spec_name, "Spec_ID": spec_id, "Rows": len(df_spec),
        "B_firms": int(b_count), "D_firms": int(d_count),
        "Mean_phi_mt": float(df_spec['phi_mt'].mean()), "Mean_phi_t": float(df_spec['phi_t'].mean()),
        "B_Cond_Share_NaNs": int(df_spec[df_spec['is_B']]['share_B_cond'].isna().sum()),
        "D_Share_NaNs": int(df_spec[~df_spec['is_B']]['share_D'].isna().sum())
    }
    return df_spec, summary, spec_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=str, default='all', help='Specification ID (1-12) or "all"')
    args = parser.parse_args()

    if args.spec.lower() == 'all':
        spec_ids_to_run = list(range(1, 13))
    elif '-' in args.spec:
        start, end = map(int, args.spec.split('-'))
        spec_ids_to_run = list(range(start, end + 1))
    else:
        spec_ids_to_run = [int(args.spec)]

    from pathlib import Path
    import json
    import pickle

    panel_csv, sleep_output_dir, demand_output_dir = _resolve_runtime_paths()

    print(f"Loading Base Panel {panel_csv}...")
    df_base = build_base_panel(panel_csv)
    print(f"Base Panel rows (with valid lagged structure): {len(df_base)}")

    demand_output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    spec_summaries = {}

    SPEC_MAP = {
        1: 'OLS x Base', 2: 'IV_CostShifters x Base', 3: 'IV_Wholesale x Base', 4: 'IV_HausmanFull x Base',
        5: 'OLS x Macro', 6: 'IV_CostShifters x Macro', 7: 'IV_Wholesale x Macro', 8: 'IV_HausmanFull x Macro',
        9: 'OLS x Tech', 10: 'IV_CostShifters x Tech', 11: 'IV_Wholesale x Tech', 12: 'IV_HausmanFull x Tech'
    }

    results_pickle = sleep_output_dir / "estimation_results.pkl"
    if not results_pickle.exists():
        print(f"[!] No pickle found at {results_pickle}. Run estimation_3_sleep.py first.")
    else:
        print(f"\n=== Processing est3 (Pooled B+D Logistic AME) from {results_pickle} ===")
        with open(results_pickle, 'rb') as f: results_dict = pickle.load(f)

        for target_id in spec_ids_to_run:
            target_name = SPEC_MAP.get(target_id)
            if not target_name: continue

            actual_key = next((k for k in results_dict.keys() if target_name in k), None)
            if not actual_key:
                continue

            task = (actual_key, results_dict[actual_key], df_base)
            df_spec, summary, spec_id = process_specification(task)

            if df_spec is not None:
                out_pkl = demand_output_dir / f"demand_3_logistic_spec_{target_id}.parquet"
                df_spec.to_parquet(out_pkl, engine='pyarrow')
                print(f" > Saved Spec {target_id} (logistic) -> {out_pkl.name} ({len(df_spec)} rows)")
                spec_summaries[f"3_logistic_{target_id}"] = summary
                total_saved += 1
            else:
                print(f"   [!] Failed or skipped Spec: {target_id} (logistic)")

    if total_saved > 0:
        summary_file = demand_output_dir / "demand_prep_summary_3.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_data.update(spec_summaries)
                    spec_summaries = existing_data
            except Exception: pass
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(spec_summaries, f, indent=4)
        print(f"\nSummary saved to: {summary_file}")
        print(f"[SUCCESS] {total_saved} per-spec parquets written to {demand_output_dir}")
    else:
        print(f"\n[WARNING] No parquets written for Estimation 3")

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
