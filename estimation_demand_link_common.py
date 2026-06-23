"""
estimation_demand_link_common.py
================================================================================
Shared demand-prep for the link-based sleepiness routines Est4/Est5/Est6.
Mirrors estimation_3_demand_1_prep.py (reshape panel, reconstruct phi_mt, build
Active Deposits Dep^Act and data-implied conditional B/D shares) but:

  * phi is reconstructed from the NATIVE index coefficients (params_native),
    not the AMEs (user guardrail: AMEs are reporting-only); and
  * the link is parametrised — 'uniform' (clip), 'probit' (Phi), 'index'
    (the stored cubic sieve G) — so the three wrappers stay tiny.

This reproduces, in demand prep, exactly the phi each estimation_N_sleep.py
writes to its market_panel_phis.csv.
"""
import os
import sys
import logging
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
from scipy.stats import norm

from utils import paths
from utils.sleep_links import NonLinearResults  # noqa: F401 (needed for unpickling)

DATA_DIR = paths.PROCESSED
_PANEL_WITH_FEES = DATA_DIR / "market_panel_with_fees.csv"
PANEL_CSV = _PANEL_WITH_FEES if _PANEL_WITH_FEES.exists() else DATA_DIR / "market_panel.csv"
BANKED_CSV = paths.INCLUSION_DIR / "bcb_banked_mca_panel.csv"

X_COLS = ['fgc_covered', 'has_ip', 'seg_S2', 'seg_S3', 'seg_S4', 'seg_S5',
          'log_total_assets_lag', 'equity_ratio_lag']
D_COLS = ['gdp_per_capita', 'fraction_65plus', 'fraction_young',
          'pix_users_pf_per1000', 'connections_per100', 'frac_4g5g',
          'branches_per1000', 'cadunico_families_per1000']
IV_BLP_LOO = ['loo_log_assets', 'mean_loo_log_assets', 'loo_equity_ratio', 'mean_loo_equity_ratio',
              'loo_basileia', 'mean_loo_basileia', 'loo_credit_assets', 'mean_loo_credit_assets',
              'loo_npl_provision', 'mean_loo_npl_provision', 'n_rivals']
IV_COST = ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag']
IV_CAPITAL = ['indice_basileia_lag']
IV_FEE = ['cosif_fee_ratio_all', 'cosif_fee_ratio_total_deposits', 'cosif_fee_valid',
          'listed_fee_atm_withdrawal_pf', 'listed_fee_statement_pf',
          'tarifa_stickiness_yrs', 'tarifa_stickiness_n']
EXTRA_KEEP_COLS = (X_COLS + D_COLS + IV_BLP_LOO + IV_COST + IV_CAPITAL + IV_FEE
                   + ['segment', 'spread_qoq', 'spread_ann'])

SPEC_MAP = {
    1: 'OLS x Base', 2: 'IV_CostShifters x Base', 3: 'IV_Wholesale x Base', 4: 'IV_HausmanFull x Base',
    5: 'OLS x Macro', 6: 'IV_CostShifters x Macro', 7: 'IV_Wholesale x Macro', 8: 'IV_HausmanFull x Macro',
    9: 'OLS x Tech', 10: 'IV_CostShifters x Tech', 11: 'IV_Wholesale x Tech', 12: 'IV_HausmanFull x Tech',
}
SPEC_MAP_INV = {v: k for k, v in SPEC_MAP.items()}


def extract_upsilon_terms(res_ss):
    """NATIVE index coefficients (phi is built from these, never from AMEs)."""
    params = getattr(res_ss, "params_native", None)
    if params is None:
        params = res_ss.params
    upsilon = {}
    for var_name, coef in params.items():
        if var_name == "nr_lagged_dep":
            upsilon["constant"] = coef
        elif var_name.startswith("interaction_"):
            upsilon[var_name.replace("interaction_", "")] = coef
    return upsilon


def _reshape_panel_from_wide(df_raw):
    id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
    df_raw = df_raw.drop_duplicates(subset=id_vars).copy()
    df_raw['is_B'] = (df_raw['CODMUN_IBGE'].astype(str) != '0')
    df = pd.wide_to_long(df_raw, stubnames=['dep_a', 'spread_a', 'spread_ann_a'], i=id_vars, j='deposit_type').reset_index()
    return df.rename(columns={'dep_a': 'deposit_balance', 'spread_a': 'spread_qoq', 'spread_ann_a': 'spread_ann'})


def _reshape_panel(df_raw):
    df_raw['mca_code'] = df_raw['mca_code'].astype(str)
    if 'dep_a1' in df_raw.columns:
        return _reshape_panel_from_wide(df_raw)
    df = df_raw.copy()
    if 'is_B' not in df.columns:
        df['is_B'] = (df['CODMUN_IBGE'].astype(str) != '0')
    if 'deposit_type' in df.columns:
        df = df[df['deposit_type'] != 3].copy()
    pre_k5 = (df['deposit_type'] == 5) & ((df['year'] < 2020) | ((df['year'] == 2020) & (df['quarter'] < 2)))
    if pre_k5.any():
        df = df[~pre_k5].copy()
    return df


def build_base_panel(panel_csv):
    print(f"Loading {panel_csv}...")
    df_raw = pd.read_csv(panel_csv, dtype={'mca_code': str}, low_memory=False)
    df = _reshape_panel(df_raw)
    df['fgc_covered'] = df['deposit_type'].astype('Int64').isin([1, 2, 4]).astype(int)
    if 'has_ip' not in df.columns:
        df['has_ip'] = 0
    df = df.copy()
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
    for c in ('is_coop', 'is_state_owned'):
        df[c] = df[c].fillna(0.0) if c in df.columns else 0.0
    # Same scaling as estimation_*_sleep.build_pooled_data so the native index matches.
    if 'gdp_per_capita' in df.columns: df['gdp_per_capita'] /= 10000.0
    if 'cadunico_families_per1000' in df.columns: df['cadunico_families_per1000'] /= 100.0
    if 'pix_users_pf_per1000' in df.columns: df['pix_users_pf_per1000'] /= 100.0
    if 'connections_per100' in df.columns: df['connections_per100'] /= 100.0
    for col in ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']:
        if col in df.columns: df[col] = df[col].fillna(df[col].median())
    if 'pop_total' not in df.columns: df['pop_total'] = np.nan
    cosif_ratio_cols = [c for c in df.columns if c.startswith('cosif_fee_ratio')]
    if cosif_ratio_cols and 'cosif_fee_valid' in df.columns:
        df.loc[df['cosif_fee_valid'] == 0, cosif_ratio_cols] = np.nan
    if BANKED_CSV.exists():
        banked = pd.read_csv(BANKED_CSV, usecols=['mca_code', 'year', 'banked_correction'],
                             dtype={'mca_code': str, 'year': int})
        banked['banked_correction'] = pd.to_numeric(banked['banked_correction'], errors='coerce')
        df = df.merge(banked, on=['mca_code', 'year'], how='left')
    else:
        df['banked_correction'] = np.nan
    return df


def _apply_link(index_series, link, res_ss):
    """Map the native linear index to phi in [0,1] under the estimator's link."""
    idx = index_series.astype(float)
    if link == 'uniform':
        phi = np.clip(idx, 0.0, 1.0)
    elif link == 'probit':
        phi = norm.cdf(idx)
    elif link == 'index':
        b = np.asarray(res_ss.si_b)
        vsd = res_ss.si_vsd if getattr(res_ss, 'si_vsd', None) else 1.0
        vs = (idx - res_ss.si_vmu) / vsd
        phi = np.zeros(len(vs))
        for d in range(len(b)):
            phi = phi + b[d] * vs ** d
    elif link in ('sieve', 'kernel'):
        # Joint single index (Est7/Est8): the native index has NO constant (it is
        # absorbed in G); the monotone link is stored as a grid in the native-index
        # frame. This mirrors phi_from_native exactly.
        phi = np.interp(idx.values, res_ss.si_vgrid, res_ss.si_ggrid)
    else:
        raise ValueError(f"unknown link {link!r}")
    return pd.Series(phi, index=index_series.index).clip(lower=0.0, upper=1.0)


def process_specification(spec_name, spec_res, df_base, link):
    res_ss = spec_res.get('second_stage')
    if res_ss is None:
        return None, None, spec_name
    upsilon = extract_upsilon_terms(res_ss)

    base_cols = ['entity_id', 'time_id', 'CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter',
                 'deposit_type', 'is_B', 'deposit_balance', 'lagged_deposits', 'gross_return_lag', 'pop_total']
    keep_cols = list(set(base_cols + [c for c in EXTRA_KEEP_COLS + ['banked_correction'] if c in df_base.columns]))
    df_spec = df_base[keep_cols].copy()
    if 'is_B' in df_spec.columns:
        df_spec['is_B'] = df_spec['is_B'].astype(bool)

    df_spec['phi_mt'] = 0.0
    for sv_name, beta in upsilon.items():
        if sv_name not in df_base.columns:
            if sv_name in ('is_coop', 'is_state_owned'):
                df_spec[sv_name] = df_base.get(sv_name, 0.0)
            elif '_x_' in sv_name:
                p1, p2 = sv_name.split('_x_')
                v1 = df_spec[p1] if p1 in df_spec.columns else df_base.get(p1)
                v2 = df_spec[p2] if p2 in df_spec.columns else df_base.get(p2)
                if v1 is None or v2 is None:
                    return None, None, spec_name
                df_spec[sv_name] = v1 * v2
            else:
                return None, None, spec_name
        elif sv_name not in df_spec.columns:
            df_spec[sv_name] = df_base[sv_name]
        df_spec['phi_mt'] += beta * df_spec[sv_name]

    df_spec = df_spec.dropna(subset=['phi_mt', 'spread_qoq'])
    # NATIVE index -> phi via the estimator's link (AMEs are NOT used here).
    df_spec['phi_mt'] = _apply_link(df_spec['phi_mt'], link, res_ss)

    df_mca = df_spec[['mca_code', 'time_id', 'phi_mt', 'pop_total']].drop_duplicates().dropna(subset=['pop_total'])

    def weighted_mean(g):
        v, w = g['phi_mt'], g['pop_total']
        return v.mean() if w.sum() == 0 else np.average(v, weights=w)

    phi_t_map = df_mca.groupby('time_id').apply(weighted_mean, include_groups=False).rename("phi_t")
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
    mkt = df_spec.groupby(['mca_code', 'time_id']).agg(_pop=('pop_total', 'first'), _bc_mt=('_bc', 'first'))
    b_dep_mt = df_spec[df_spec['is_B']].groupby(['mca_code', 'time_id'])['Dep_Act'].sum().rename('_dep_B')
    mkt = mkt.join(b_dep_mt, how='left').fillna({'_dep_B': 0.0})
    local_ratio = (mkt['_dep_B'] / mkt['_pop']).replace([np.inf, -np.inf], np.nan)
    max_local = local_ratio.max() if local_ratio.notna().any() else 0.0
    all_dep_t = df_spec.groupby('time_id')['Dep_Act'].sum()
    nat_pop_t = mkt.groupby('time_id')['_pop'].sum()
    nat_ratio = (all_dep_t / nat_pop_t).replace([np.inf, -np.inf], np.nan)
    max_nat = nat_ratio.max() if nat_ratio.notna().any() else 0.0
    max_ratio = max(max_local, max_nat) or 1.0

    mkt['_b_mkt'] = mkt['_bc_mt'] * max_ratio * mkt['_pop']
    df_spec = df_spec.merge(mkt[['_b_mkt']].reset_index(), on=['mca_code', 'time_id'], how='left')
    mkt_r = mkt.reset_index()
    mkt_r['_w'] = mkt_r['_bc_mt'] * mkt_r['_pop']
    nat_dbar = (mkt_r.groupby('time_id')
                .apply(lambda g: (g['_w'].sum() / g['_pop'].sum()) * max_ratio if g['_pop'].sum() > 0 else FALLBACK_BC * max_ratio,
                       include_groups=False).rename('_dbar_nat'))
    nat_pop = mkt_r.groupby('time_id')['_pop'].sum().rename('_pop_nat')
    d_mkt = (nat_dbar * nat_pop).rename('_d_mkt').reset_index()
    df_spec = df_spec.merge(d_mkt, on='time_id', how='left')

    df_spec['share_B_cond'] = np.where(df_spec['is_B'], df_spec['Dep_Act'] / df_spec['_b_mkt'], np.nan)
    df_spec['share_D'] = np.where(~df_spec['is_B'], df_spec['Dep_Act'] / df_spec['_d_mkt'], np.nan)
    df_spec.drop(columns=['_b_mkt', '_d_mkt', '_bc'], inplace=True)

    spec_id = SPEC_MAP_INV.get(spec_name, spec_name)
    df_spec['Spec_ID'] = spec_id
    summary = {
        "Name": spec_name, "Spec_ID": spec_id, "Rows": len(df_spec),
        "B_firms": int(df_spec['is_B'].sum()), "D_firms": int((~df_spec['is_B']).sum()),
        "Mean_phi_mt": float(df_spec['phi_mt'].mean()), "Mean_phi_t": float(df_spec['phi_t'].mean()),
        "B_Cond_Share_NaNs": int(df_spec[df_spec['is_B']]['share_B_cond'].isna().sum()),
        "D_Share_NaNs": int(df_spec[~df_spec['is_B']]['share_D'].isna().sum()),
    }
    return df_spec, summary, spec_id


def run(est_num, link, tag):
    """Demand prep for Est{est_num} with the given link. CLI: --spec ID|all."""
    parser = argparse.ArgumentParser(description=f"Demand 1 Prep — Est{est_num} ({tag})")
    parser.add_argument('--spec', type=str, default='all', help='Specification ID (1-12) or "all"')
    args = parser.parse_args()
    if args.spec.lower() == 'all':
        spec_ids = list(range(1, 13))
    elif '-' in args.spec:
        a, b = map(int, args.spec.split('-')); spec_ids = list(range(a, b + 1))
    else:
        spec_ids = [int(args.spec)]

    sleep_output_dir = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / f"est{est_num}"
    demand_output_dir = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP"
    demand_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Base Panel {PANEL_CSV}...")
    df_base = build_base_panel(PANEL_CSV)
    print(f"Base Panel rows (with valid lagged structure): {len(df_base)}")

    results_pickle = sleep_output_dir / "estimation_results.pkl"
    if not results_pickle.exists():
        print(f"[!] No pickle at {results_pickle}. Run estimation_{est_num}_sleep.py first.")
        return
    with open(results_pickle, 'rb') as f:
        results_dict = pickle.load(f)

    total, spec_summaries = 0, {}
    for sid in spec_ids:
        target_name = SPEC_MAP.get(sid)
        if not target_name:
            continue
        key = next((k for k in results_dict if target_name in k), None)
        if not key:
            continue
        df_spec, summary, _ = process_specification(key, results_dict[key], df_base, link)
        if df_spec is not None:
            out = demand_output_dir / f"demand_{est_num}_{tag}_spec_{sid}.parquet"
            df_spec.to_parquet(out, engine='pyarrow')
            print(f" > Saved Spec {sid} ({tag}) -> {out.name} ({len(df_spec)} rows)")
            spec_summaries[f"{est_num}_{tag}_{sid}"] = summary
            total += 1
        else:
            print(f"   [!] Failed/skipped Spec {sid} ({tag})")

    if total > 0:
        summary_file = demand_output_dir / f"demand_prep_summary_{est_num}.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f); existing.update(spec_summaries); spec_summaries = existing
            except Exception:
                pass
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(spec_summaries, f, indent=4)
        print(f"\n[SUCCESS] {total} per-spec parquets written to {demand_output_dir}")
    else:
        print(f"\n[WARNING] No parquets written for Estimation {est_num}")
