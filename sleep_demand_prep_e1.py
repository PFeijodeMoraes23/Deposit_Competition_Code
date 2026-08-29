"""
sleep_demand_prep_e1.py
================================================================================
Prepares demand-side variables for the BLP step (Eq-15 and Eq-16 in V_Main.tex).

Estimation strategy 1: Local B-type — sleep model estimated only on B-type
(branch-based) institutions.  The depositor sleepiness function phi_mt is
re-applied to the full panel; D-type firms use the market-level aggregate phi_t.

Reads the output of `sleep_est_e1.py` (12 specifications for phi_mt) and
the market panel.  For each specification it computes phi_mt, aggregates to the
national phi_t, computes Active Deposits (Dep^Act), and builds data-implied
conditional market shares for B-type and D-type institutions.

``has_ip`` and ``fgc_covered`` are expected to be pre-computed in the panel
pipeline (panel_digital_flags.py / panel_market.py) and read from the CSV.

Usage
-----
  python sleep_demand_prep_e1.py --spec all
  python sleep_demand_prep_e1.py --spec 12

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
from utils import load_panel_cached
from utils import state_transform as _st
from sleep_demand_prep_link import build_market_size_and_shares, MAX_YEAR, MIN_YEAR
# market_panel.csv, NOT the fees variant — see utils/paths.market_panel_csv (USE_FEE_PANEL=1
# opts back in). The old "fees panel if it exists" fallback silently pinned the pipeline to a
# stale vintage: on 2026-08-04 the base panel was 6 days newer than the fees one.
# The frame is read through load_panel_cached, which serves the .parquet twin of this path
# whenever that is the authoritative copy. On the cluster it always is: the data bundle ships
# market_panel.parquet (39 MB) and no market_panel.csv, so a bare read_csv finds nothing.
PANEL_CSV = paths.market_panel_csv()
BANKED_CSV = paths.INCLUSION_DIR / "bcb_banked_mca_panel.csv"

X_COLS = ['fgc_covered', 'has_ip', 'seg_S2', 'seg_S3', 'seg_S4', 'seg_S5',
          'log_total_assets_lag', 'equity_ratio_lag', 'is_state_owned']
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
IV_ESTBAN = ['estban_rival_branches_lag']   # ESTBAN branch-competition IV (panel_10; within-congl var)
IV_FEE = [
    'cosif_fee_ratio_all',
    'cosif_fee_ratio_total_deposits',
    'cosif_fee_valid',
    'listed_fee_atm_withdrawal_pf',
    'listed_fee_statement_pf',
    'tarifa_stickiness_yrs',
    'tarifa_stickiness_n',
]
# CF2 needs the bank's ASSET return r^j (V_Main eq 16, ψ1 row): the return on what deposits fund.
# `asset_gross_return_lag` = 1 + lagged quarterly asset yield (built in panel_bank_chars.py).
# Consumed by bbl_fwd_sim.jl --asset-return-col. NOT to be confused with `gross_return_lag`,
# which is 1 + the DEPOSIT rate (liability side) — see counterfactuals_plan.md §9.4.
CF_COST_COLS = ['asset_gross_return_lag', 'asset_return_imputed',
                # LEVEL r^f for the BBL/CF stack. `risk_free_qoq_lag` is grand-mean
                # CENTRED below (utils/state_transform.CENTER), so it reaches the parquet
                # as a deviation (mean ~0, negative values) and must never be used as a
                # level. `risk_free_qoq` is contemporaneous; `risk_free_qoq_lag_level` is
                # the uncentred lag, snapshotted just before center().
                'risk_free_qoq', 'risk_free_qoq_lag_level']
EXTRA_KEEP_COLS = (X_COLS + D_COLS + IV_BLP_LOO + IV_ESTBAN + IV_COST + IV_CAPITAL + IV_FEE
                   + CF_COST_COLS
                   + ['segment', 'spread_qoq', 'spread_ann'])

def _resolve_runtime_paths() -> tuple[Path, Path, Path]:
    panel_csv = PANEL_CSV
    # est_dir follows SLEEP_OUT_ROOT, so a sandboxed run reads the fit it just produced.
    # demand_parquet_dir is the parquets' own seam: it equals demand_prep_root() unless
    # DEMAND_PREP_DIR names a separate step folder (the cluster's data/output/demand_prep),
    # where the fits and the parquets they generate live in different step directories.
    sleep_output_dir = paths.est_dir(1)
    demand_output_dir = paths.demand_parquet_dir()
    return panel_csv, sleep_output_dir, demand_output_dir

class NonLinearResults:
    def __init__(self, params, bse, tvalues, pvalues, G_star, params_native=None):
        self.params = params
        self.bse = bse
        self.tvalues = tvalues
        self.pvalues = pvalues
        self.G_star = G_star
        self.df_resid = G_star

sys.modules['estimation_1_sleep'] = type('FakeModule', (), {'NonLinearResults': NonLinearResults})

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

def _resolve_is_B(df: pd.DataFrame) -> pd.Series:
    """Firm type as a boolean Series: True = B (brick-and-mortar), False = D (digital).

    `is_B` is decided once, in panel_6_market.attach_mca_code, from the physical-network
    verdict in digital_banks_diagnostic.csv, and stored in market_panel.csv as 0/1. It is
    used verbatim whenever present -- never recomputed or "corrected" here, and in
    particular not reconciled with mca_code: firm type and market tier are separate facts.
    A panel written before the column existed falls back to the CODMUN_IBGE sentinel,
    which approximates the verdict by where a bank books its deposits.
    """
    if 'is_B' in df.columns:
        s = df['is_B']
        if pd.api.types.is_bool_dtype(s):
            return s.astype(bool)
        if pd.api.types.is_numeric_dtype(s):
            # 0/1 as stored; a missing value reads as D, matching the Julia side's
            # Bool.(coalesce.(df.is_B, false)).
            return pd.to_numeric(s, errors='coerce').fillna(0) != 0
        # Text, from a CSV writer that spelled the column out ("true"/"True"/"1").
        return s.astype(str).str.strip().str.lower().isin(('1', 'true', 't', 'yes'))
    logging.warning(
        "Column 'is_B' not found in the market panel: this panel predates the stored "
        "firm-type column. Falling back to the CODMUN_IBGE sentinel; re-run "
        "panel_market.py to store the authoritative column."
    )
    return df['CODMUN_IBGE'].astype(str) != '0'


def _reshape_panel_from_wide(df_raw: pd.DataFrame) -> pd.DataFrame:
    print("Reshaping panel from wide to long...")
    id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
    df_raw = df_raw.drop_duplicates(subset=id_vars).copy()  # de-fragment
    df_raw['is_B'] = _resolve_is_B(df_raw)
    df = pd.wide_to_long(df_raw, stubnames=['dep_a', 'spread_a', 'spread_ann_a'], i=id_vars, j='deposit_type').reset_index()
    return df.rename(columns={'dep_a': 'deposit_balance', 'spread_a': 'spread_qoq', 'spread_ann_a': 'spread_ann'})

def _reshape_panel(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw['mca_code'] = df_raw['mca_code'].astype(str)
    if 'dep_a1' in df_raw.columns: return _reshape_panel_from_wide(df_raw)
    df = df_raw.copy()
    df['is_B'] = _resolve_is_B(df)
    if 'deposit_type' in df.columns: df = df[df['deposit_type'] != 3].copy()
    pre_k5_mask = (df['deposit_type'] == 5) & ((df['year'] < 2020) | ((df['year'] == 2020) & (df['quarter'] < 2)))
    if pre_k5_mask.any():
        logging.info(f"  Dropped {pre_k5_mask.sum()} k=5 rows before 2020Q2 (product did not exist).")
        df = df[~pre_k5_mask].copy()
    return df

def build_base_panel(panel_csv: Path) -> pd.DataFrame:
    print(f"Loading {panel_csv}...")
    df_raw = load_panel_cached(panel_csv, dtype={'mca_code': str}, low_memory=False)
    df = _reshape_panel(df_raw)

    # fgc_covered: FGC insures types 1 (savings), 2 (demand), 4 (time deposits).
    # Type 5 (prepaid) is NOT FGC-covered (BCB Res. 80/2021; CMN 4.678/2018).
    # The market panel is wide (no deposit_type column), so compute here post-reshape.
    df['fgc_covered'] = df['deposit_type'].astype('Int64').isin([1, 2, 4]).astype(int)

    # has_ip comes from bank_chars_panel.csv (panel_bank_chars.py reads IF-Data List files).
    if 'has_ip' not in df.columns:
        logging.warning(
            "Column 'has_ip' not found in panel CSV. "
            "Re-run panel_bank_chars.py to derive it from IF-Data List files. Defaulting to 0."
        )
        df['has_ip'] = 0

    df = df.copy()  # de-fragment before the column inserts below

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

    # Demand sample ends at MAX_YEAR (2025 = last full year in the panel; keeps partial 2026 out).
    # The 2025 IF-Data recode does NOT break the bank characteristics — the earlier "+50% jump in
    # total_assets" was an analysis error (summing a conta across reports; report-1-only gives a
    # matched-bank ratio of 1.015). See estimation_demand_link_common.MAX_YEAR / plan §0A.2.
    # Applied after the lags, so the final quarter keeps its lag.
    if 'year' in df.columns:
        _n0 = len(df)
        if MIN_YEAR is not None:
            df = df[df['year'] >= MIN_YEAR].copy()
        if MAX_YEAR is not None:
            df = df[df['year'] <= MAX_YEAR].copy()
        logging.info(f"  year window [{MIN_YEAR}, {MAX_YEAR}]: kept {len(df):,} of {_n0:,} rows "
                     f"(dropped {_n0 - len(df):,})")

    df['constant'] = 1.0
    if 'year' in df.columns:
        df['post_2020'] = (df['year'] >= 2020).astype(int)
        df['pix_exists'] = ((df['year'] > 2020) | ((df['year'] == 2020) & (df['quarter'] == 4))).astype(float)

    # D-type dummy: the complement of the authoritative firm-type column.
    df['dummy_D_type'] = (~_resolve_is_B(df)).astype(float)

    if 'is_coop' in df.columns:
        df['is_coop'] = df['is_coop'].fillna(0.0)
    else:
        df['is_coop'] = 0.0

    if 'is_state_owned' in df.columns:
        df['is_state_owned'] = pd.to_numeric(df['is_state_owned'], errors='coerce').fillna(0.0)
    else:
        df['is_state_owned'] = 0.0

    # Raw panel units -> estimation units. Factors live in utils/state_transform.SCALE,
    # shared with estimation_2_sleep.build_pooled_data: phi is rebuilt here as
    # params_native x these columns, so a scale that differs from the one theta was
    # estimated under gives a silently wrong phi that still lies in [0,1].
    _st.apply_scale(df)

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
            usecols=['mca_code', 'year', 'banked_correction', 'findex_banked_frac'],
            dtype={'mca_code': str, 'year': int},
        )
        for _c in ('banked_correction', 'findex_banked_frac'):
            banked[_c] = pd.to_numeric(banked[_c], errors='coerce')
        df = df.merge(banked, on=['mca_code', 'year'], how='left')
    else:
        logging.warning(f"Banked correction panel not found at {BANKED_CSV}; fallback 1.1 used.")
        df['banked_correction'] = np.nan

    # GRAND-MEAN CENTERING -- last, and by the SAME persisted means the sleepiness
    # estimation used (loaded, never recomputed: this frame is a different sample, so a
    # locally-computed mean would silently shift the index phi is rebuilt from).
    # Snapshot the UNCENTRED lagged r^f before centring. bbl_fwd_sim.jl and
    # cf1_franchise.jl need a LEVEL (a centred r^f biases omega-hat by the grand
    # mean, 0.0216/q = 8.6pp/yr) and cf_deposit_sim.jl's accrual clamp pins 82%
    # of rows at zero when handed a deviation. See identification_notes.md section 9.
    if 'risk_free_qoq_lag' in df.columns:
        df['risk_free_qoq_lag_level'] = df['risk_free_qoq_lag']
    _st.load_transform().center(df)
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

    if 'logistic' in spec_name.lower():
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

    # Market size M_mt / M_nat and the shares of ACTIVE depositors.
    # SHARED implementation — see estimation_demand_link_common.build_market_size_and_shares()
    # and counterfactuals_plan.md §0. This block used to be copy-pasted here, which is exactly
    # how the (1-phi), anchor and bc defects survived: a fix in one copy never reached the others.
    df_spec = build_market_size_and_shares(df_spec)

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
        print(f"[!] No pickle found at {results_pickle}. Run sleep_est_e1.py first.")
    else:
        print(f"\n=== Processing est1 (Local B-type) from {results_pickle} ===")
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
                out_pkl = demand_output_dir / f"demand_1_spec_{target_id}.parquet"
                df_spec.to_parquet(out_pkl, engine='pyarrow')
                print(f" > Saved Spec {target_id} -> {out_pkl.name} ({len(df_spec)} rows)")
                spec_summaries[f"1_{target_id}"] = summary
                total_saved += 1
            else:
                print(f"   [!] Failed or skipped Spec: {target_id}")

    if total_saved > 0:
        summary_file = demand_output_dir / "demand_prep_summary_1.json"
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
        print(f"\n[WARNING] No parquets written for Estimation 1")

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
