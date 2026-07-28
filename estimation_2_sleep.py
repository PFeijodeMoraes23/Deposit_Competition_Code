"""
estimation_2_sleep.py
==============================
Strategy 2: Estimates the depositor sleepiness function using pooled B and D-type firms.
D-type firms receive national population-weighted average state variables.
No bank-type dummies, no D-type interactions, no endogenous variables (branches, pix volume).
CF correction: v_hat x lagged_deposits (linear in residual, interacted with lagged deposits).

Outputs are directed to ESTIMATION_OUTPUT/DEMAND_PREP/est2

CLI Options:
  --spec12          Only run spec 12 (Tech x IV_HausmanFull)
  --write-centers   Rebuild ESTIMATION_OUTPUT/DEMAND_PREP/state_centering_means.json
                    (the grand means of the state block) and exit. THE ONLY WRITER --
                    every other caller, including E1 and all three demand preps, only
                    ever LOADS it. See utils/state_transform.py.
"""
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import pickle
import warnings
import concurrent.futures

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

try:
    from utils import load_panel_cached
except Exception:
    load_panel_cached = None

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")

# ==============================================================================
# GLOBAL SETUP
# ==============================================================================
from utils import paths as _paths_mod
from utils import state_transform as _st
DATA_DIR = _paths_mod.PROCESSED
_PANEL_WITH_FEES = DATA_DIR / "market_panel_with_fees.csv"
PANEL_CSV = _PANEL_WITH_FEES if _PANEL_WITH_FEES.exists() else DATA_DIR / "market_panel.csv"
OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "est2"

PLOTS_DIR = OUTPUT_DIR / "PLOTS"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def apply_imbalanced_cluster_correction(res, cluster_series):
    from utils.cluster import effective_cluster_stats   # single source of G*/CV
    _st = effective_cluster_stats(cluster_series.value_counts().values)
    G_nominal, G_star = _st["G_nominal"], _st["G_star"]
    res.G_nominal = G_nominal
    res.G_star = G_star
    res.df_resid = G_star
    # Inference: score/multiplier wild cluster bootstrap (Cameron-Gelbach-Miller 2008;
    # MacKinnon-Webb 2017) -- the SAME scheme as the single-index/joint estimators, so
    # every comparison-table column shares one inference method. G* is kept as the
    # reported effective-cluster diagnostic. Falls back to CRVE + t(G*) if unavailable.
    try:
        from utils.sleep_links import linear_wild_cluster_bootstrap
        bse, tvals, pvals = linear_wild_cluster_bootstrap(res)
        for _nm, _v in (("bse", bse.values), ("tvalues", tvals.values), ("pvalues", pvals.values)):
            res._results._cache[_nm] = _v          # statsmodels reads cached props from _cache
            res._results.__dict__[_nm] = _v
    except Exception as e:
        print(f"  [linear WCB] failed ({e}); CRVE + t(G*) fallback")
        res._results.__dict__['pvalues'] = stats.t(df=G_star).sf(np.abs(res.tvalues)) * 2
    return res

def demean_variables(df, cols, entity_col):
    means = df.groupby(entity_col)[cols].transform('mean')
    return df[cols] - means


def demean_variables_2way(df, cols, entity_col, time_col, n_iter=15, tol=1e-9):
    """Two-way (entity + time) additive FE removal by alternating projections
    (Gaure 2013). Exact within transform for D = ... + alpha_i + delta_t."""
    _, einv = np.unique(df[entity_col].values, return_inverse=True)
    ec = np.bincount(einv).astype(float)
    _, tinv = np.unique(df[time_col].values, return_inverse=True)
    tc = np.bincount(tinv).astype(float)
    M = df[cols].to_numpy(dtype=float, copy=True)
    for _ in range(n_iter):
        prev = M.copy()
        for j in range(M.shape[1]):
            M[:, j] -= (np.bincount(einv, M[:, j]) / ec)[einv]
            M[:, j] -= (np.bincount(tinv, M[:, j]) / tc)[tinv]
        if np.max(np.abs(M - prev)) < tol:
            break
    return pd.DataFrame(M, columns=cols, index=df.index)


# Time block (E4/E6/E8 "+Time" variants). time_trend was DROPPED: with quarter
# fixed effects (delta_t) now absorbing aggregate time additively, a pure-time
# linear trend in the index is redundant/collinear with the time FE. The block
# keeps only gdp_growth_yoy (a genuine entity-time business-cycle covariate).
TIME_VARS = ['gdp_growth_yoy']


def add_time_variables(df):
    """Add the two time-series state variables in place and return df.
    time_trend     : continuous years since the first sample quarter.
    gdp_growth_yoy : within-entity year-over-year growth of gdp_per_capita
                     (scale-invariant ratio; first 4 obs/entity and inf -> 0)."""
    df.sort_values(by=['entity_id', 'year', 'quarter'], inplace=True)
    df['time_trend'] = (df['year'] - int(df['year'].min())) + (df['quarter'] - 1) / 4.0
    gpc = df['gdp_per_capita']
    growth = df.groupby('entity_id')['gdp_per_capita'].transform(lambda s: s / s.shift(4) - 1.0)
    growth = growth.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df['gdp_growth_yoy'] = growth.where(gpc.notna(), 0.0)
    return df


def define_specifications(time_block=False):
    s_base = ['constant']
    s_macro = ['constant', 'pix_exists', 'gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young', 'risk_free_qoq_lag']
    s_tech = s_macro + ['connections_per100']

    if time_block:
        # +Time variants: append the time block to every state block.
        s_base = s_base + TIME_VARS
        s_macro = s_macro + TIME_VARS
        s_tech = s_tech + TIME_VARS

    iv_specs = {
        'OLS': [],
        'IV_CostShifters': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag'],
        'IV_Wholesale': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag'],
        'IV_HausmanFull': ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'indice_basileia_lag', 'leave_one_out_mean_spread'],
    }
    state_blocks = {'Base': s_base, 'Macro': s_macro, 'Tech': s_tech}
    return s_tech, iv_specs, state_blocks

# ==============================================================================
# DATA BUILDING
# ==============================================================================
# The estimation window [2016, 2024] is defined once in utils/window.py — see that module for the full
# rationale on both bounds.  utils.window already reads DEMAND_MIN_YEAR / DEMAND_MAX_YEAR; the
# SLEEP_*_YEAR env vars below stay as a sleep-specific override on top of it.
from utils.window import MIN_YEAR as _WINDOW_MIN_YEAR, MAX_YEAR as _WINDOW_MAX_YEAR  # noqa: E402

SLEEP_MIN_YEAR = int(os.environ.get('SLEEP_MIN_YEAR', _WINDOW_MIN_YEAR))
SLEEP_MAX_YEAR = int(os.environ.get('SLEEP_MAX_YEAR', _WINDOW_MAX_YEAR))


def build_pooled_data(time_block=False, center=True):
    """The canonical pooled B+D prep. E1 (via build_unified_frame) and E3-E9 (via
    estimation_sleep_common) both go through it, so the state block is scaled and
    centred in exactly one place.

    center=False is used ONLY by --write-centers, which must see the frame in
    scaled-but-uncentred units to compute the grand means in the first place."""
    df_raw = load_panel_cached(PANEL_CSV) if load_panel_cached else pd.read_csv(PANEL_CSV, dtype={'mca_code': str}, low_memory=False)
    df_raw = df_raw.copy()  # defragment: market_panel_with_fees has many columns from merges

    # Restrict the whole sleep estimation to the [SLEEP_MIN_YEAR, SLEEP_MAX_YEAR] window (see note above).
    if 'year' in df_raw.columns:
        _n0 = len(df_raw)
        df_raw = df_raw[(df_raw['year'] >= SLEEP_MIN_YEAR) & (df_raw['year'] <= SLEEP_MAX_YEAR)].copy()
        print(f"  [sleep] year window [{SLEEP_MIN_YEAR}, {SLEEP_MAX_YEAR}]: kept {len(df_raw):,} of {_n0:,} rows")

    df_raw['pix_exists'] = ((df_raw['year'] > 2020) | ((df_raw['year'] == 2020) & (df_raw['quarter'] == 4))).astype(float)

    if 'dep_a1' in df_raw.columns:
        id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
        # panel_6_market.aggregate_to_mca() guarantees one row per market. Fail loudly if
        # that ever regresses: the previous silent drop_duplicates() here kept ONE arbitrary
        # municipality per MCA, discarding ~74% of rows and understating every market's
        # deposits -- invisibly, for as long as the panel had been municipality-level.
        _dups = int(df_raw.duplicated(subset=id_vars).sum())
        if _dups:
            raise ValueError(
                f"market panel is not unique at {id_vars}: {_dups:,} duplicate rows "
                f"({len(df_raw):,} rows / {df_raw.drop_duplicates(subset=id_vars).shape[0]:,} markets). "
                "The municipality->MCA aggregation (panel_6_market.aggregate_to_mca) is missing or "
                "stale -- rebuild the panel; do NOT de-duplicate here."
            )
        df = pd.wide_to_long(df_raw, stubnames=['dep_a', 'spread_a', 'spread_ann_a', 'leave_one_out_mean_spread_a'], i=id_vars, j='deposit_type').reset_index()
        df = df.rename(columns={'dep_a': 'deposit_balance', 'spread_a': 'spread_qoq', 'spread_ann_a': 'spread_ann', 'leave_one_out_mean_spread_a': 'leave_one_out_mean_spread'})
    else:
        df = df_raw.copy()

    if 'deposit_type' in df.columns:
        df = df[df['deposit_type'] != 3].copy()

    df['entity_id'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['deposit_type'].astype(str) + "_" + df['mca_code'].astype(str)
    df['time_id'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)

    df.sort_values(by=['entity_id', 'year', 'quarter'], inplace=True)
    df['spread_qoq_lag'] = df.groupby('entity_id')['spread_qoq'].shift(1)
    df['risk_free_qoq_lag'] = df.groupby('entity_id')['risk_free_qoq'].shift(1)
    df['lagged_deposits'] = df.groupby('entity_id')['deposit_balance'].shift(1)
    df['nr_lagged_dep'] = (1 + df['risk_free_qoq_lag'] - df['spread_qoq_lag']) * df['lagged_deposits']

    if 'leave_one_out_mean_spread' not in df.columns:
        df['leave_one_out_mean_spread'] = np.nan

    df['bank_year'] = df['CodConglomeradoPrudencial'].astype(str) + "_" + df['year'].astype(str)
    df = df.dropna(subset=['deposit_balance', 'nr_lagged_dep', 'spread_qoq', 'entity_id', 'time_id'])

    df['constant'] = 1.0
    scale_cols = {'deposit_balance': 1e9, 'nr_lagged_dep': 1e9, 'lagged_deposits': 1e9}
    for col, factor in scale_cols.items():
        if col in df.columns: df[col] /= factor

    # Raw panel units -> estimation units. The factors live in utils/state_transform.SCALE,
    # which the three demand preps import too: phi is rebuilt downstream as params_native x
    # parquet columns, so a site that scales differently from the site that estimated theta
    # yields a silently wrong phi that still lies in [0,1].
    _st.apply_scale(df)

    state_vars_to_fill = ['gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young', 'connections_per100']

    # Assign national population-weighted averages to D-type (CODMUN_IBGE == '0') rows.
    # This includes both true digital banks and B-type prepaid (k=5) national aggregates.
    if 'pop_total' in df.columns:
        cols_to_fill = [col for col in state_vars_to_fill if col in df.columns]
        b_mask = (df['CODMUN_IBGE'].astype(str) != '0')
        national_mask = (df['CODMUN_IBGE'].astype(str) == '0')
        df_b = df.loc[b_mask, ['time_id', 'pop_total'] + cols_to_fill].copy()
        col_medians = df_b[cols_to_fill].median()
        for col in cols_to_fill:
            df_b[col] = df_b[col].fillna(col_medians[col])
        df_b['_w'] = df_b['pop_total'].fillna(0)
        w_sum_by_t = df_b.groupby('time_id')['_w'].sum()
        nat_avg_df = {}
        for col in cols_to_fill:
            wv = (df_b[col] * df_b['_w']).groupby(df_b['time_id']).sum()
            nat_avg_df[col] = (wv / w_sum_by_t.replace(0, np.nan)).fillna(col_medians[col])
        natl_time_ids = df.loc[national_mask, 'time_id']
        for col in cols_to_fill:
            df.loc[national_mask, col] = natl_time_ids.map(nat_avg_df[col]).values

    for col in state_vars_to_fill:
        if col in df.columns: df[col] = df[col].fillna(df[col].median())

    if time_block:
        df = add_time_variables(df)

    # GRAND-MEAN CENTERING -- must be the LAST thing that touches the state block.
    #   * after the :207 dropna, or S_bar is not the estimation-sample mean;
    #   * after both imputations above, or the persisted mean is an unfilled-sample mean;
    #   * after add_time_variables, because gdp_growth_yoy = gpc/gpc.shift(4) - 1 is
    #     scale-invariant but NOT shift-invariant -- centering gdp_per_capita first puts
    #     near-zero values in that denominator and corrupts the whole time block.
    # Pure reparametrisation: slopes/AMEs/phi unchanged, theta_0 -> theta_0 + sum_k theta_k S_bar_k,
    # i.e. the constant becomes phi-hat at the average market instead of phi at S = 0.
    if center:
        _st.load_transform().center(df)
    return df

# ==============================================================================
# ESTIMATION KERNELS
# ==============================================================================
def run_pooled_first_stage(df, spec_instruments, exogenous_controls):
    endog_mask = df['deposit_type'].isin([4, 5])
    first_stage_vars = list(set(spec_instruments + exogenous_controls))
    if 'constant' in first_stage_vars: first_stage_vars.remove('constant')

    valid_mask = endog_mask & df[first_stage_vars].notnull().all(axis=1)
    df_fs = df[valid_mask].copy()

    if len(df_fs) == 0:
        df['v_hat'] = 0.0
        df['v_hat_x_lagged_dep'] = 0.0
        return df, None

    mod = sm.OLS(df_fs['spread_qoq'], sm.add_constant(df_fs[first_stage_vars]))
    cluster_series = df_fs['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    res = apply_imbalanced_cluster_correction(res, cluster_series)

    # Rows excluded from the first stage (endogenous k=4,5 rows whose instruments are missing — chiefly
    # 2013-2015, where prudential-conglomerate bank characteristics do not exist) get v_hat = NaN, NOT 0.
    # A zero here would smuggle those rows into the control-function (IV) second stage treating their
    # endogenous spread as exogenous, biasing the very coefficients that reconstruct φ̂ downstream. With
    # NaN, v_hat_x_lagged_dep is NaN there and the second stage's dropna(X_cols) drops them — so the
    # INSTRUMENTED specs are cleanly floored to where the instruments exist (~2016+), while the OLS/state
    # specs (has_cf=False, no v_hat term) keep the full 2013+ sample. See counterfactuals_plan.md §0A.
    df['v_hat'] = np.nan
    df.loc[valid_mask, 'v_hat'] = res.resid
    df['v_hat_x_lagged_dep'] = df['v_hat'] * df['lagged_deposits']
    return df, res

def run_pooled_second_stage(df, state_vars, has_cf=False):
    X_cols = []
    for sv in state_vars:
        col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
        df[col_name] = df[sv] * df['nr_lagged_dep'] if sv != 'constant' else df['nr_lagged_dep']
        X_cols.append(col_name)
    if has_cf: X_cols.append('v_hat_x_lagged_dep')

    df_ss = df.dropna(subset=X_cols + ['deposit_balance']).copy()
    if len(df_ss) == 0: return None

    y_dm = demean_variables_2way(df_ss, ['deposit_balance'], 'entity_id', 'time_id')['deposit_balance']
    X_dm = demean_variables_2way(df_ss, X_cols, 'entity_id', 'time_id')

    mod = sm.OLS(y_dm, X_dm)
    cluster_series = df_ss['CodConglomeradoPrudencial'].astype(str)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': cluster_series}, use_t=True)
    return apply_imbalanced_cluster_correction(res, cluster_series)

def exec_pooled_spec(args):
    df, iv_name, iv_cols, s_name, s_cols = args
    has_cf = len(iv_cols) > 0
    spec_name = f"{iv_name} x {s_name}"

    df_target = df
    res_fs = None
    if has_cf:
        iv_cols_act = [c for c in iv_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        exog_cols_act = [c for c in s_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        if not iv_cols_act: return None, spec_name, None
        df_target, res_fs = run_pooled_first_stage(df_target, iv_cols_act, exog_cols_act)

    res_ss = run_pooled_second_stage(df_target, s_cols, has_cf=has_cf)
    return res_ss, spec_name, res_fs

def calculate_pooled_phis(df, res_dict, state_blocks):
    phi_results = {}
    if 'pop_total' in df.columns:
        df['market_size'] = df['pop_total'].fillna(0)
    else:
        df['market_size'] = 1.0

    for model_key, res_item in res_dict.items():
        if res_item['second_stage'] is None: continue

        try:
            m_type, spec_name = model_key.split(' x ')
        except ValueError: continue

        s_cols = state_blocks.get(spec_name, [])
        if not s_cols: continue

        ss_res = res_item['second_stage']
        phi_mt = np.zeros(len(df))
        for sv in s_cols:
            col_name = f"interaction_{sv}" if sv != 'constant' else "nr_lagged_dep"
            if col_name in ss_res.params:
                c = ss_res.params[col_name]
                phi_mt += c if sv == 'constant' else c * df[sv].fillna(0)

        safe_key = model_key.replace(' ', '_').replace('.', '')
        df[f'phi_mt_{safe_key}'] = phi_mt
        # phi_t = sum_m phi_mt*M_mt / sum_m M_mt over MARKETS m (a market is an MCA).
        market_agg = df.groupby(['year_quarter', 'mca_code'], observed=True).agg(
            phi_mt=(f'phi_mt_{safe_key}', 'mean'), M_mt=('market_size', 'sum')).reset_index()
        weighted_phi = market_agg['phi_mt'] * market_agg['M_mt']
        national_agg = (weighted_phi.groupby(market_agg['year_quarter']).sum() /
                        market_agg['M_mt'].groupby(market_agg['year_quarter']).sum().replace(0, np.nan)
                        ).fillna(0).reset_index(name=f'phi_t_{safe_key}')
        phi_results[safe_key] = national_agg
    return df, phi_results

# ==============================================================================
# PIPELINE
# ==============================================================================
def run_pooled_phase(spec12_only=False):
    print("=== PHASE 1: POOLED B+D ESTIMATION ===")
    df = build_pooled_data()
    _, iv_specs, state_blocks = define_specifications()

    if spec12_only:
        tasks = [(df, 'IV_HausmanFull', iv_specs['IV_HausmanFull'], 'Tech', state_blocks['Tech'])]
    else:
        tasks = [(df, iv_name, iv_specs[iv_name], s_name, state_blocks[s_name])
                 for s_name in state_blocks.keys()
                 for iv_name in ['OLS', 'IV_CostShifters', 'IV_Wholesale', 'IV_HausmanFull']]

    results_dict = {}
    _nw = min(4, max(1, (os.cpu_count() or 4) // max(1, int(os.environ.get('SLEEP_PIPELINE_NSLOTS', '1')))))
    with concurrent.futures.ProcessPoolExecutor(max_workers=_nw) as executor:
        for res_ss, spec_name, res_fs in executor.map(exec_pooled_spec, tasks):
            if res_ss is not None:
                results_dict[spec_name] = {'second_stage': res_ss, 'first_stage': res_fs}
                print(f"Computed: {spec_name}")

    with open(OUTPUT_DIR / "estimation_results.pkl", 'wb') as f: pickle.dump(results_dict, f)

    df['year_quarter'] = df['time_id']
    df, national_phis = calculate_pooled_phis(df, results_dict, state_blocks)
    df.to_csv(OUTPUT_DIR / "market_panel_phis.csv", index=False)

    from functools import reduce
    if national_phis:
        agg_df = reduce(lambda left, right: pd.merge(left, right, on='year_quarter', how='outer'), national_phis.values())
        agg_df.to_csv(OUTPUT_DIR / "national_phi_t.csv", index=False)
    print("Saved Pooled B+D Pickles and Phis.\n")

def run_plotting_phase(spec12_only=False):
    print("=== PHASE 2: PLOTTING ===")
    try:
        df = pd.read_csv(OUTPUT_DIR / "market_panel_phis.csv", low_memory=False)
    except FileNotFoundError:
        print("Run phase 1 first.")
        return

    cols_to_plot = [c for c in df.columns if c.startswith('phi_mt_OLS') or c.startswith('phi_mt_IV')]
    if spec12_only:
        cols_to_plot = [c for c in cols_to_plot if 'Tech' in c]

    if 'market_size' not in df.columns:
        df['market_size'] = 1.0

    for col_name in cols_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        w = df['market_size']
        num = (df[col_name] * w).groupby(df['year_quarter']).sum()
        den = w.groupby(df['year_quarter']).sum()
        phi_agg = (num / den.replace(0, np.nan)).fillna(0).reset_index()
        phi_agg.columns = ['year_quarter', 'phi']
        phi_agg['date'] = pd.PeriodIndex(phi_agg['year_quarter'].str.replace('_', 'Q'), freq='Q').to_timestamp()
        phi_agg = phi_agg.sort_values('date')
        ax.plot(phi_agg['date'], phi_agg['phi'], color='blue', linewidth=2)
        ax.set_title(f"Pooled B+D Sleepiness: {col_name.replace('phi_mt_', '')}", fontsize=14)
        ax.set_ylabel(r"National $\hat{\phi}_t$")
        ax.set_ylim(0, 1.0)
        ax.grid()
        fig.tight_layout()
        plt.savefig(PLOTS_DIR / f"Pooled_{col_name.replace('phi_mt_', '')}.png", dpi=300)
        plt.close(fig)
    print("Plots generated.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimation 2: Pooled B+D Sleepiness (Linear)")
    parser.add_argument('--spec12', action='store_true', help='Only run spec 12 (Tech x IV_HausmanFull)')
    parser.add_argument('--write-centers', action='store_true',
                        help='Rebuild state_centering_means.json from the pooled frame and exit')
    args = parser.parse_args()

    pd.options.mode.chained_assignment = None

    if args.write_centers:
        # ALWAYS the time_block=True frame, so the persisted vector carries every CENTER
        # entry (a time_block=False frame would silently omit gdp_growth_yoy and leave two
        # JSONs racing for one path). add_time_variables drops no rows, so the seven shared
        # means are identical either way -- but "write once, load always" makes that a
        # guarantee rather than a coincidence.
        _df = build_pooled_data(time_block=True, center=False)
        _st.write_transform(_df)
        raise SystemExit(0)

    run_pooled_phase(spec12_only=args.spec12)
    run_plotting_phase(spec12_only=args.spec12)
