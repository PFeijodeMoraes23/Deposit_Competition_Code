# Switch-cost investigation copy of sleep_demand_prep_link.py (copied 2026-10-13 17:09, source md5 ab6975a96d497813ad72161198e181d6).
# It writes only to the switch-cost tree (utils/switch_cost_sandbox.py); the original is untouched.
"""
switch_cost_sleep_demand_prep_link.py
================================================================================
Shared demand-prep for the link-based sleepiness routines E3/E4 (see DEMAND_CFG at
the bottom). Mirrors production's E2 demand prep, sleep_demand_prep_e2.py (reshape panel, reconstruct
phi_mt, build Active Deposits Dep^Act and data-implied conditional B/D shares) but:

  * phi is reconstructed from the NATIVE index coefficients (params_native),
    not the AMEs (user guardrail: AMEs are reporting-only); and
  * the link is parametrised — 'uniform' (clip), 'probit' (Phi), 'index'
    (the stored cubic sieve G) — so one `--est N` invocation covers each routine.

This reproduces, in demand prep, exactly the phi each estimation_N_sleep.py
writes to its market_panel_phis.csv.

SWITCH-COST ARMS (--arm). The investigation lets persistence differ by deposit type. Every arm
writes the same parquet layout into its own folder, switch_cost_demand_<arm>:

  pooled  the production construction, unchanged, on the pooled fit (--fits-dir).
  F1      waking hazard by the type held. phi_k = G(S'theta + lambda_k) from the type-shift fit.
          Active deposits Dep - phi_k (1+r) Dep_{t-1}. Awake pool
              A_mt = (1 - phi_0,mt) M_mt - sum_rows (phi_row - phi_0,row) (1+r) Dep_{t-1},
          i.e. every lagged balance wakes at its own type's rate, and the outside holders at
          the pooled rate phi_0 (user decision of 2026-10-13).
  F3hi,   common attention plus locked CDBs. Active deposits as in F1 (the measured carry
  F3lo    beta_k is the type-shift phi_k). Choosing pool
              P_mt = (1 - phi_0,mt) (M_mt - sum_{k=4 rows} r_4 (1+r) Dep_{t-1}),
          with r_4 the locked share of the bank's time deposits: contractual maturity beyond
          90 days (F3hi) or NSFR withdrawal after 6+ months (F3lo); `switch_cost_tools.py locked`.

In every arm the market sizes M_mt and M_nat are the pooled construction's (user decision of
2026-10-13), phi_0 is the pooled fit's phi, and the national market (D firms) uses the
population-weighted national phi_k,t and all rows' lagged balances. With phi_k = phi_0 (F1) or
r_4 = 0 (F3) every arm returns the pooled parquet exactly; `switch_cost_tools.py prep` tests it.
The share columns keep their meaning: share = Dep_Act / pool, and `_1mphi` is pool / M.
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

from utils import switch_cost_sandbox  # noqa: F401,E402  (output tree: import before any path is resolved)
from utils import paths
from utils import load_panel_cached
from utils import routines as R
from utils import state_transform as _st
from utils import demand_chars   # corrected characteristics, segments and balances (always applied)
from utils.switch_cost_sleep_links import NonLinearResults  # noqa: F401 (needed for unpickling)

# market_panel.csv, NOT the fees variant — see utils/paths.market_panel_csv (USE_FEE_PANEL=1).
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
IV_BLP_LOO = ['loo_log_assets', 'mean_loo_log_assets', 'loo_equity_ratio', 'mean_loo_equity_ratio',
              'loo_basileia', 'mean_loo_basileia', 'loo_credit_assets', 'mean_loo_credit_assets',
              'loo_npl_provision', 'mean_loo_npl_provision', 'n_rivals']
IV_COST = ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag']
IV_CAPITAL = ['indice_basileia_lag']
# ESTBAN branch-competition instrument (panel_estban_instrument.py): log1p lagged count of RIVAL
# branches in the local market (MCA). Unlike the accounting IVs it has within-conglomerate variation
# (across the conglomerate's municipalities) and is independent + relevant for demand-deposit spreads.
IV_ESTBAN = ['estban_rival_branches_lag']
IV_FEE = ['cosif_fee_ratio_all', 'cosif_fee_ratio_total_deposits', 'cosif_fee_valid',
          'listed_fee_atm_withdrawal_pf', 'listed_fee_statement_pf',
          'tarifa_stickiness_yrs', 'tarifa_stickiness_n']
# CF2 needs the bank's ASSET return r^j (V_Main eq 16, ψ1 row): the return on what deposits fund.
# `asset_gross_return_lag` = 1 + lagged quarterly asset yield (built in panel_bank_chars.py).
# Consumed by bbl_fwd_sim.jl --asset-return-col. NOT `gross_return_lag`, which is 1 + the
# DEPOSIT rate (liability side) — see counterfactuals_plan.md §9.4.
CF_COST_COLS = ['asset_gross_return_lag', 'asset_return_imputed',
                # LEVEL r^f for the BBL/CF stack. `risk_free_qoq_lag` is grand-mean
                # CENTRED below (utils/state_transform.CENTER), so it reaches the parquet
                # as a deviation (mean ~0, negative values) and must never be used as a
                # level. `risk_free_qoq` is contemporaneous; `risk_free_qoq_lag_level` is
                # the uncentred lag, snapshotted just before center().
                'risk_free_qoq', 'risk_free_qoq_lag_level']

# ── Market-size / active-share knobs (V_Main pp.26-27; counterfactuals_plan.md §9.8) ────────────
# PHI_CAP  φ is capped before forming (1-φ). The BLP share is a share of ACTIVE depositors, so its
#          denominator is the ACTIVE market (1-φ)·M; without a cap, (1-φ)→0 in near-comatose markets
#          and a single market-quarter would set r̂ for the whole country. The SAME capped φ is used
#          in the anchor and in the share, which is what guarantees Σ_j s_j ≤ 1/bc < 1.
#          0.99 barely binds: p90(φ)=0.977.
# RMAX_RULE  anchor for d̄ = bc·r̂. The original 'max' is set by an outlier — 16.4× the p99, a
#          74,000-population municipality — which crushes the typical market's inside share to
#          ~0.001 (a 99.9% outside option) and leaves the model with no competitive interaction.
#          'p99' winsorises it. Use 'max' to reproduce the original construction.
# TOTAL_SHARE_CAP  a percentile anchor does not bound every market, so a thin tail would get
#          Σ_j s_j > 1 (impossible — the BLP contraction needs a positive outside option). A floor on
#          M (the market must be big enough to hold what is in it) caps the TOTAL inside share at
#          this value in every market, which a per-share clip cannot do.
# BC_RULE  the banked correction. 'findex_gdp' (default) = 1/(findex · gdppc/median(gdppc)); 'findex'
#          = 1/findex (no local access); 'legacy' = the original deposit-intensity bc (robustness).
# ACCESS_LO/HI  the access index is CENTRED at 1 (not capped at 1), so it does not systematically
#          inflate M; clipped to keep GDP-per-capita outliers (mining/refinery towns) in check.
# FINDEX_FALLBACK  used where the World Bank series is missing.
PHI_CAP         = float(os.environ.get('DEMAND_PHI_CAP', 0.99))
RMAX_RULE       = os.environ.get('DEMAND_RMAX_RULE', 'q95')       # q95 | p99 | p999 | max
BC_RULE         = os.environ.get('DEMAND_BC_RULE', 'findex_gdp')  # findex_gdp | findex | legacy
ACCESS_LO       = float(os.environ.get('DEMAND_ACCESS_LO', 0.35))
ACCESS_HI       = float(os.environ.get('DEMAND_ACCESS_HI', 2.5))
FINDEX_FALLBACK = float(os.environ.get('DEMAND_FINDEX_FALLBACK', 0.77))
TOTAL_SHARE_CAP = float(os.environ.get('DEMAND_TOTAL_SHARE_CAP', 0.95))

# MIN_YEAR / MAX_YEAR  the estimation window [2016, 2024], defined once in utils/window.py — see that
#          module for the full rationale on both bounds (and the DEMAND_MIN_YEAR / DEMAND_MAX_YEAR env
#          overrides, which it reads).  Re-exported here so the existing
#          `from estimation_demand_link_common import MIN_YEAR, MAX_YEAR` importers keep working.
from utils.window import MIN_YEAR, MAX_YEAR  # noqa: E402

# NB: M_mt / M_nat are BUILT inside process_specification (after the keep_cols filter) and so reach
# the parquet without needing to be listed here.
EXTRA_KEEP_COLS = (X_COLS + D_COLS + IV_BLP_LOO + IV_ESTBAN + IV_COST + IV_CAPITAL + IV_FEE
                   + CF_COST_COLS
                   + ['segment', 'spread_qoq', 'spread_ann'])
# The flags and the re-timed borrowings columns of utils/demand_chars.py ride into the parquet.
EXTRA_KEEP_COLS = EXTRA_KEEP_COLS + demand_chars.KEEP_COLS

SPEC_MAP = {
    1: 'OLS x Base', 2: 'IV_CostShifters x Base', 3: 'IV_Wholesale x Base', 4: 'IV_RivalSpread x Base',
    5: 'OLS x Macro', 6: 'IV_CostShifters x Macro', 7: 'IV_Wholesale x Macro', 8: 'IV_RivalSpread x Macro',
    9: 'OLS x Tech', 10: 'IV_CostShifters x Tech', 11: 'IV_Wholesale x Tech', 12: 'IV_RivalSpread x Tech',
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


def _ensure_is_B(df, where):
    """Attach the firm-type verdict, preferring the column stored on the panel.

    `is_B` is decided once in panel_6_market and written to market_panel.csv: True for a
    brick-and-mortar (local) firm, False for a digital/national one. Deriving it again from
    CODMUN_IBGE here would silently overrule that verdict, so the stored column wins whenever
    it is present and CODMUN_IBGE is only a fallback for panels built before it existed.
    """
    if 'is_B' in df.columns and df['is_B'].notna().any():
        df['is_B'] = df['is_B'].astype(bool)
        return df
    if 'CODMUN_IBGE' not in df.columns:
        raise KeyError(f"{where}: neither is_B nor CODMUN_IBGE is available to set firm type")
    logging.warning(
        f"{where}: market panel has no is_B column -- falling back to CODMUN_IBGE != '0'. "
        f"Rebuild the panel with panel_market.py to get the stored verdict.")
    df['is_B'] = (df['CODMUN_IBGE'].astype(str) != '0')
    return df


def _reshape_panel_from_wide(df_raw):
    id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
    df_raw = df_raw.drop_duplicates(subset=id_vars).copy()
    df_raw = _ensure_is_B(df_raw, "_reshape_panel_from_wide")
    df = pd.wide_to_long(df_raw, stubnames=['dep_a', 'spread_a', 'spread_ann_a'], i=id_vars, j='deposit_type').reset_index()
    return df.rename(columns={'dep_a': 'deposit_balance', 'spread_a': 'spread_qoq', 'spread_ann_a': 'spread_ann'})


def _reshape_panel(df_raw):
    df_raw['mca_code'] = df_raw['mca_code'].astype(str)
    if 'dep_a1' in df_raw.columns:
        return _reshape_panel_from_wide(df_raw)
    df = _ensure_is_B(df_raw.copy(), "_reshape_panel")
    if 'deposit_type' in df.columns:
        df = df[df['deposit_type'] != 3].copy()
    pre_k5 = (df['deposit_type'] == 5) & ((df['year'] < 2020) | ((df['year'] == 2020) & (df['quarter'] < 2)))
    if pre_k5.any():
        df = df[~pre_k5].copy()
    return df


def build_base_panel(panel_csv, time_block=False):
    """The E3-E9 demand frame.

    `time_block` is handled HERE rather than by the caller so that the ordering
    constraint lives in one place: gdp_growth_yoy is a RATIO of gdp_per_capita to
    its own 4-quarter lag, so it must be built BEFORE the state block is centred
    (a ratio is scale-invariant but not shift-invariant -- centring first puts
    near-zero values in that denominator)."""
    print(f"Loading {panel_csv}...")
    df_raw = load_panel_cached(panel_csv, dtype={'mca_code': str}, low_memory=False)
    df_raw = demand_chars.apply(df_raw)
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

    # Bound the demand sample to [MIN_YEAR, MAX_YEAR].  MAX (2025): keep partial 2026 out.  MIN (2016):
    # prudential-conglomerate bank characteristics — and the instruments built from them — do not exist
    # on the model's basis before 2016 (see the MIN_YEAR note above).  Applied AFTER the lags are formed,
    # so the boundary quarter keeps its lag.
    if 'year' in df.columns:
        _n0 = len(df)
        if MIN_YEAR is not None:
            df = df[df['year'] >= MIN_YEAR].copy()
        if MAX_YEAR is not None:
            df = df[df['year'] <= MAX_YEAR].copy()
        logging.info(f"  year window [{MIN_YEAR}, {MAX_YEAR}]: kept {len(df):,} of {_n0:,} rows "
                     f"(dropped {_n0 - len(df):,})")

    df['constant'] = 1.0
    # The deposit-type shifts of the type-shift fit (switch_cost_sleep_est_e2.TYPE_DUMMIES):
    # uncentred 0/1, demand deposits the base type.
    for _k in (2, 4, 5):
        df[f'dtype_{_k}'] = (df['deposit_type'].astype(int) == _k).astype(float)
    if 'year' in df.columns:
        df['post_2020'] = (df['year'] >= 2020).astype(int)
        df['pix_exists'] = ((df['year'] > 2020) | ((df['year'] == 2020) & (df['quarter'] == 4))).astype(float)
    # The D-type dummy is the complement of the stored verdict, never an independent test.
    if 'is_B' in df.columns:
        df['dummy_D_type'] = (~df['is_B'].astype(bool)).astype(float)
    elif 'CODMUN_IBGE' in df.columns:
        logging.warning("dummy_D_type: no is_B column -- falling back to CODMUN_IBGE == '0'")
        df['dummy_D_type'] = (df['CODMUN_IBGE'].astype(str) == '0').astype(float)
    for c in ('is_coop', 'is_state_owned'):
        # is_state_owned arrives as bool (Tc==1); cast to float 0/1 since it is now a demand regressor.
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0) if c in df.columns else 0.0
    # Raw panel units -> estimation units, from utils/state_transform.SCALE -- the same
    # object estimation_2_sleep.build_pooled_data uses, so the native index matches by
    # construction rather than by a comment asking two copies to agree.
    _st.apply_scale(df)
    for col in ['pix_users_pf_per1000', 'connections_per100', 'branches_per1000']:
        if col in df.columns: df[col] = df[col].fillna(df[col].median())
    if 'pop_total' not in df.columns: df['pop_total'] = np.nan
    cosif_ratio_cols = [c for c in df.columns if c.startswith('cosif_fee_ratio')]
    if cosif_ratio_cols and 'cosif_fee_valid' in df.columns:
        df.loc[df['cosif_fee_valid'] == 0, cosif_ratio_cols] = np.nan
    if BANKED_CSV.exists():
        # findex_banked_frac (World Bank Global Findex, national, interpolated by year) is the BANKED
        # FRACTION of the population. It is the economically meaningful part of the banked correction:
        # a fully-served market's inside share should equal the banked fraction, because the UNBANKED
        # ARE the outside option. `banked_correction` is kept for the legacy/robustness path.
        banked = pd.read_csv(BANKED_CSV,
                             usecols=['mca_code', 'year', 'banked_correction', 'findex_banked_frac'],
                             dtype={'mca_code': str, 'year': int})
        for _c in ('banked_correction', 'findex_banked_frac'):
            banked[_c] = pd.to_numeric(banked[_c], errors='coerce')
        df = df.merge(banked, on=['mca_code', 'year'], how='left')
    else:
        df['banked_correction'] = np.nan
        df['findex_banked_frac'] = np.nan

    if time_block:
        from switch_cost_sleep_est_e2 import add_time_variables
        df = add_time_variables(df)
        print("  [+Time] added time_trend + gdp_growth_yoy to the demand frame")

    # GRAND-MEAN CENTERING -- last, after the time block is built, and by the SAME
    # persisted means the sleepiness estimation used (loaded, never recomputed: this
    # frame is a different sample, so a locally-computed mean would silently shift the
    # index phi is rebuilt from).
    # Snapshot the UNCENTRED lagged r^f before centring. bbl_fwd_sim.jl and
    # cf1_franchise.jl need a LEVEL (a centred r^f biases omega-hat by the grand
    # mean, 0.0216/q = 8.6pp/yr) and cf_deposit_sim.jl's accrual clamp pins 82%
    # of rows at zero when handed a deviation. See identification_notes.md section 9.
    if 'risk_free_qoq_lag' in df.columns:
        df['risk_free_qoq_lag_level'] = df['risk_free_qoq_lag']
    _st.load_transform().center(df)
    return df


def _apply_link(index_series, link, res_ss):
    """Map the native linear index to phi in [0,1] under the estimator's link.

    THE RESULT'S OWN TAG DECIDES THE BRANCH; `link` is only the fallback for a result that
    carries none. The caller passes a kind-level default (DEMAND_CFG), and the two disagree
    exactly where it is most damaging: a shape-constrained single-index fit is configured as
    "index" but stores link="index_sieve" with si_b holding ~10 I-spline ramp betas. The
    'index' branch below would read those as monomial coefficients of a degree-9 polynomial,
    which explodes across the panel and clips to 0 or 1 almost everywhere -- the same
    mechanism that produced an 8.45pp national phi_t in the derived CSVs, here in the
    consumer that feeds the demand parquets and the BLP.
    """
    link = getattr(res_ss, "link", None) or link
    idx = index_series.astype(float)
    if link == 'logit':
        phi = 1.0 / (1.0 + np.exp(-np.clip(idx, -700, 700)))
    elif link == 'uniform':
        phi = np.clip(idx, 0.0, 1.0)
    elif link == 'probit':
        phi = norm.cdf(idx)
    elif link == 'index_sieve':
        # E3/E4 shape-constrained link (monotone I-spline, beta>=0 and sum(beta)<=1). Stored as
        # a grid over the FULL native index, constant INCLUDED. Mirrors phi_from_native exactly.
        phi = np.interp(idx.values, res_ss.si_vgrid, res_ss.si_ggrid)
    elif link == 'index':
        b = np.asarray(res_ss.si_b)
        vsd = res_ss.si_vsd if getattr(res_ss, 'si_vsd', None) else 1.0
        vs = (idx - res_ss.si_vmu) / vsd
        phi = np.zeros(len(vs))
        for d in range(len(b)):
            phi = phi + b[d] * vs ** d
    else:
        raise ValueError(f"unknown link {link!r}")
    return pd.Series(phi, index=index_series.index).clip(lower=0.0, upper=1.0)



# ==============================================================================
# SHARED market size + ACTIVE-depositor shares  (V_Main pp.26-27; see
# counterfactuals_plan.md §0). This lives in ONE place; in production it is also imported by
# the E1 and E2 demand preps, which have no switch-cost copy.
#
# It used to be copy-pasted into all three scripts, which is exactly how the
# (1-phi) / anchor / bc defects survived: a fix in one copy never reached the
# others. Do NOT re-inline it.
#
# Requires on df_spec: is_B, phi_mt, phi_t, Dep_Act, pop_total, mca_code,
#                      time_id, year, gdp_per_capita, banked_correction,
#                      findex_banked_frac
# Adds:  M_mt, M_nat, share_B_cond, share_D
# ==============================================================================
def build_market_size_and_shares(df_spec: pd.DataFrame) -> pd.DataFrame:
    """Market size M and the shares of ACTIVE depositors. See counterfactuals_plan.md §0."""
    # RMAX_RULE: the raw `max` anchor is set by an outlier (16.4× the p99; a 74k-population
    # municipality), which crushes the typical market's inside share to ~0.001. 'p99'/'p999' winsorise
    # it. Set RMAX_RULE='max' to reproduce the original construction.
    # ── The banked correction bc_mt ────────────────────────────────────────────────────────────
    #   BC_RULE='findex_gdp' (default):  bc = 1 / ( findex_t · access_mt ),  access = gdppc/median(gdppc)
    #   BC_RULE='findex'              :  bc = 1 / findex_t                  (no market-level access)
    #   BC_RULE='legacy'              :  the original banked_correction     (robustness column only)
    #
    # findex is the BANKED FRACTION: a fully-served market's inside share should equal it, because the
    # UNBANKED ARE the outside option. `access` adds local heterogeneity (V_Main line 472).
    #
    # Why NOT the original bc = 1/(findex·min{d^pc/Q95,1}): the intensity term is the market's own
    # DEPOSIT DENSITY, so it enlarges M using the very outcome the share then divides by — the same
    # information twice. It drives bc to a median of 4.0 and a MAX OF 991, inflating M to 77× GDP per
    # capita and crushing the typical market's inside share to 0.001 (a 99.9% outside option).
    # `access` is instead built from an EXOGENOUS local characteristic (GDP per capita: the strongest
    # available proxy for deposit density, corr 0.34; branch density 0.16 and CadÚnico 0.18 are weaker,
    # and a 3-variable index adds nothing (R 0.349 vs 0.344) while flipping CadÚnico's sign).
    # It is CENTRED at 1 (not capped at 1) so it does not systematically inflate M.
    FALLBACK_BC = 1.1
    df_spec['_1mphi'] = 1.0 - np.where(df_spec['is_B'],
                                       df_spec['phi_mt'], df_spec['phi_t']).clip(0.0, PHI_CAP)
    _fx = pd.to_numeric(df_spec.get('findex_banked_frac'), errors='coerce').fillna(FINDEX_FALLBACK) \
        if 'findex_banked_frac' in df_spec.columns else pd.Series(FINDEX_FALLBACK, index=df_spec.index)
    _fx = _fx.clip(0.3, 1.0)

    if BC_RULE == 'legacy':
        df_spec['_bc'] = df_spec['banked_correction'].fillna(FALLBACK_BC)
        df_spec['_access'] = 1.0
    else:
        if BC_RULE == 'findex_gdp':
            _x = pd.to_numeric(df_spec['gdp_per_capita'], errors='coerce')
            _med = _x.groupby(df_spec['year']).transform('median')
            _acc = (_x / _med).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(ACCESS_LO, ACCESS_HI)
        elif BC_RULE == 'findex':
            _acc = pd.Series(1.0, index=df_spec.index)
        else:
            raise ValueError(f"BC_RULE must be findex_gdp|findex|legacy (got {BC_RULE})")
        df_spec['_access'] = _acc
        df_spec['_bc'] = 1.0 / (_fx * _acc)

    mkt = df_spec.groupby(['mca_code', 'time_id']).agg(_pop=('pop_total', 'first'),
                                                       _bc_mt=('_bc', 'first'),
                                                       _acc_mt=('_access', 'first'))
    # (1-φ) for the LOCAL market must come from a B row (φ_mt). Taking the group's `first` row can
    # pick a D row, whose (1-φ) is built from the NATIONAL φ_t — which would break the floor bound.
    _b = df_spec[df_spec['is_B']]
    mkt = mkt.join(_b.groupby(['mca_code', 'time_id'])['_1mphi'].first().rename('_1mphi'), how='left')
    mkt['_1mphi'] = mkt['_1mphi'].fillna(1.0 - PHI_CAP)
    b_dep_mt = _b.groupby(['mca_code', 'time_id'])['Dep_Act'].sum().rename('_dep_B')
    mkt = mkt.join(b_dep_mt, how='left').fillna({'_dep_B': 0.0})

    # ── The anchor r̂ ───────────────────────────────────────────────────────────────────────────
    # The share is  s = Dep_Act / ((1-φ)·bc·r̂·Pop) = dens · findex · access / r̂ ,
    # with dens = Dep_Act/((1-φ)·Pop) the ACTIVE-market density. So the market sitting at the chosen
    # quantile of (dens · access) gets an inside share of exactly `findex` — the banked fraction.
    # THAT is why the anchor is taken over dens·access, and why Q95 is the right quantile: `bc`'s own
    # definition names Q95 as its saturated reference. The original `max` anchor sat 16× ABOVE that
    # reference (it was set by a 74,000-population municipality — an ESTBAN HQ-booking artifact),
    # which is internally inconsistent with bc and is what produced the 99.9% outside option.
    local_ratio = ((mkt['_dep_B'] / (mkt['_1mphi'] * mkt['_pop'])) * mkt['_acc_mt']) \
        .replace([np.inf, -np.inf], np.nan)
    dep_over_1mphi = (df_spec['Dep_Act'] / df_spec['_1mphi']).replace([np.inf, -np.inf], np.nan)
    all_dep_t = dep_over_1mphi.groupby(df_spec['time_id']).sum()
    nat_pop_t = mkt.groupby('time_id')['_pop'].sum()
    nat_ratio = (all_dep_t / nat_pop_t).replace([np.inf, -np.inf], np.nan)

    def _anchor(s):
        s = s.dropna()
        if s.empty:
            return 0.0
        if RMAX_RULE == 'max':   return float(s.max())          # original construction (robustness)
        if RMAX_RULE == 'p999':  return float(s.quantile(0.999))
        if RMAX_RULE == 'p99':   return float(s.quantile(0.99))
        if RMAX_RULE == 'q95':   return float(s.quantile(0.95))  # bc's own declared reference
        raise ValueError(f"RMAX_RULE must be max|p999|p99|q95 (got {RMAX_RULE})")

    max_ratio = max(_anchor(local_ratio), _anchor(nat_ratio)) or 1.0

    mkt['_b_mkt'] = mkt['_bc_mt'] * max_ratio * mkt['_pop']          # TOTAL market M_mt

    # MARKET-SIZE FLOOR. A percentile anchor (unlike `max`) does not bound every market, so ~1% of
    # market-quarters would get Σ_j s_j > 1 — an impossible share vector (the BLP contraction needs a
    # positive outside option). Rather than clip the shares (which does not fix the SUM), impose the
    # economically-obvious constraint that the market is at least large enough to hold what is in it:
    #
    #     M_mt ≥ Σ_j Dep_Act_jmt / ((1-φ_mt) · TOTAL_SHARE_CAP)
    #
    # ⇒ Σ_j s_j = Σ_j Dep_Act / ((1-φ)·M) ≤ TOTAL_SHARE_CAP < 1, by construction, in EVERY market.
    # It binds only in the thin tail above the anchor; elsewhere the anchor governs.
    # Build the floor from EXACTLY the quantity that appears in the share sum, row-wise:
    #     Σ_j s_j = Σ_j Dep_Act_j/((1-φ_j)·M) = [Σ_j Dep_Act_j/(1-φ_j)] / M
    # so the floor is that numerator ÷ CAP. Using a market-level (1-φ) would break the bound wherever
    # φ varies within a market-quarter.
    _floor_B = ((_b.assign(_x=lambda x: x['Dep_Act'] / x['_1mphi'])
                   .groupby(['mca_code', 'time_id'])['_x'].sum()) / TOTAL_SHARE_CAP)
    _floor_B = _floor_B.reindex(mkt.index).replace([np.inf, -np.inf], np.nan)
    _n_bind = int((_floor_B > mkt['_b_mkt']).sum())
    mkt['_b_mkt'] = np.maximum(mkt['_b_mkt'], _floor_B.fillna(0.0))
    if _n_bind:
        print(f"  [market size] floor bound in {_n_bind}/{len(mkt)} market-quarters "
              f"({100*_n_bind/len(mkt):.2f}%)  [RMAX_RULE={RMAX_RULE}, PHI_CAP={PHI_CAP}, "
              f"TOTAL_SHARE_CAP={TOTAL_SHARE_CAP}]")

    df_spec = df_spec.merge(mkt[['_b_mkt']].reset_index(), on=['mca_code', 'time_id'], how='left')
    mkt_r = mkt.reset_index()
    mkt_r['_w'] = mkt_r['_bc_mt'] * mkt_r['_pop']
    nat_dbar = (mkt_r.groupby('time_id')
                .apply(lambda g: (g['_w'].sum() / g['_pop'].sum()) * max_ratio if g['_pop'].sum() > 0 else FALLBACK_BC * max_ratio,
                       include_groups=False).rename('_dbar_nat'))
    nat_pop = mkt_r.groupby('time_id')['_pop'].sum().rename('_pop_nat')
    d_mkt = (nat_dbar * nat_pop).rename('_d_mkt').reset_index()
    # Same floor, nationally, for the D-firm market.
    _d_num = (df_spec.loc[~df_spec['is_B']]
              .assign(_x=lambda x: x['Dep_Act'] / x['_1mphi'])
              .groupby('time_id')['_x'].sum().rename('_dfloor'))
    d_mkt = d_mkt.merge((_d_num / TOTAL_SHARE_CAP).reset_index(), on='time_id', how='left')
    d_mkt['_d_mkt'] = np.maximum(d_mkt['_d_mkt'], d_mkt['_dfloor'].fillna(0.0))
    d_mkt = d_mkt.drop(columns=['_dfloor'])
    df_spec = df_spec.merge(d_mkt, on='time_id', how='left')

    # Shares of the ACTIVE market (1-φ)·M. Bounded by TOTAL_SHARE_CAP by construction (see floor).
    df_spec['share_B_cond'] = np.where(
        df_spec['is_B'], df_spec['Dep_Act'] / (df_spec['_1mphi'] * df_spec['_b_mkt']), np.nan)
    df_spec['share_D'] = np.where(
        ~df_spec['is_B'], df_spec['Dep_Act'] / (df_spec['_1mphi'] * df_spec['_d_mkt']), np.nan)

    # PERSIST the market size (V_Main §"Demand Parameters", M_mt = d̄_mt·Pop_mt with
    # d̄_mt = bc_mt·r̂_max). It was previously computed here, used for the shares, and then DROPPED —
    # so the counterfactuals could not see it and each re-invented their own market size as a
    # per-type scalar dbar·pop_total that ignores banked_correction entirely. That meant the demand
    # model was ESTIMATED under one market size and the CFs SIMULATED under another. Keeping these
    # two columns lets cf_deposit_sim.jl consume the estimation's own M. See §9.8.
    df_spec['M_mt'] = df_spec['_b_mkt']      # local market size (B firms)
    df_spec['M_nat'] = df_spec['_d_mkt']     # national market size (D firms)
    df_spec.drop(columns=['_b_mkt', '_d_mkt', '_bc'], inplace=True)
    return df_spec


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

    # ── Market size M and the ACTIVE-depositor shares ──────────────────────────────────────────
    # The BLP share is the share of ACTIVE (awake) depositors, so the denominator is the ACTIVE
    # market (1-φ)·M — NOT the total market M. Dividing by M alone made the fitted shares (1-φ)×
    # too small (≈13.6× here), which in turn made the simulator's active deposit channel — the ONLY
    # channel through which spreads move deposits — 13.6× too weak. See counterfactuals_plan.md §9.8.
    #
    #   Dep_Act_j = (1-φ)·M·s_j     (law of motion, V_Main eq 324)
    #   ⇒  s_j = Dep_Act_j / ((1-φ)·M)
    #
    # PHI_CAP: (1-φ) → 0 as φ → 1, so the ratio explodes in near-comatose markets and a single such
    # market-quarter would otherwise set r̂ for the whole country. Cap φ, and use the SAME capped φ in
    # BOTH the anchor and the share — that is what guarantees Σ_j s_j ≤ 1/bc  < 1.
    # Market size M_mt / M_nat and the ACTIVE-depositor shares (share_B_cond, share_D).
    # Single shared implementation — see build_market_size_and_shares() above and
    # counterfactuals_plan.md §0. Do NOT re-inline: three divergent copies is how the
    # (1-phi), anchor and bc defects survived.
    df_spec = build_market_size_and_shares(df_spec)

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


# ==============================================================================
# SWITCH-COST ARMS (see the module docstring)
# ==============================================================================
ARMS = ("pooled", "F1", "F3hi", "F3lo")
DEP_TYPES = (1, 2, 4, 5)
LOCKED_MEDIAN_KEY = "__median__"


def _index_from(df_spec, df_base, upsilon):
    """The native index sum_s theta_s S_s on df_spec, by process_specification's own rule for
    states the frame lacks (None when one is unavailable). Same terms, same order."""
    idx = pd.Series(0.0, index=df_spec.index)
    for sv_name, beta in upsilon.items():
        if sv_name not in df_base.columns:
            if sv_name in ('is_coop', 'is_state_owned'):
                df_spec[sv_name] = df_base.get(sv_name, 0.0)
            elif '_x_' in sv_name:
                p1, p2 = sv_name.split('_x_')
                v1 = df_spec[p1] if p1 in df_spec.columns else df_base.get(p1)
                v2 = df_spec[p2] if p2 in df_spec.columns else df_base.get(p2)
                if v1 is None or v2 is None:
                    return None
                df_spec[sv_name] = v1 * v2
            else:
                return None
        elif sv_name not in df_spec.columns:
            df_spec[sv_name] = df_base[sv_name]
        idx = idx + beta * df_spec[sv_name]
    return idx


def _national_phi(df_spec, col):
    """Population-weighted national phi per quarter: process_specification's phi_t rule."""
    df_mca = df_spec[['mca_code', 'time_id', col, 'pop_total']].drop_duplicates().dropna(subset=['pop_total'])

    def weighted_mean(g):
        v, w = g[col], g['pop_total']
        return v.mean() if w.sum() == 0 else np.average(v, weights=w)

    return df_mca.groupby('time_id').apply(weighted_mean, include_groups=False)


def load_locked(csv_path, bound):
    """r_4 by (conglomerate, year) for one bound ('hi' | 'lo'), plus the year medians for the
    banks the sources do not cover."""
    t = pd.read_csv(csv_path, dtype={'CodConglomeradoPrudencial': str})
    col = f"r4_{bound}"
    med = t[t['CodConglomeradoPrudencial'] == LOCKED_MEDIAN_KEY].set_index('year')[col]
    own = t[t['CodConglomeradoPrudencial'] != LOCKED_MEDIAN_KEY].set_index(
        ['CodConglomeradoPrudencial', 'year'])[col]
    return own, med


def process_specification_arm(spec_name, spec_res, pooled_res, df_base, link, arm, locked=None):
    """F1 / F3 demand frame for one spec. spec_res is the type-shift fit, pooled_res the pooled
    fit (phi_0 and the market sizes). Returns (df, summary, spec_id) like process_specification."""
    res_ts = spec_res.get('second_stage')
    res_0 = pooled_res.get('second_stage')
    if res_ts is None or res_0 is None:
        return None, None, spec_name

    # 1. The pooled construction, unchanged: the market sizes and the pooled awake fraction.
    df0, _summary0, _ = process_specification(spec_name, pooled_res, df_base, link)
    if df0 is None:
        return None, None, spec_name
    mkt_keys = ['mca_code', 'time_id']
    M_loc = df0.groupby(mkt_keys)['M_mt'].first()
    acc_loc = df0.groupby(mkt_keys)['_access'].first() if '_access' in df0.columns else None
    M_nat = df0.groupby('time_id')['M_nat'].first()
    omp0_loc = df0[df0['is_B']].groupby(mkt_keys)['_1mphi'].first()     # the anchor's (1 - phi_0)
    omp0_nat = df0[~df0['is_B']].groupby('time_id')['_1mphi'].first()

    # 2. One frame carrying phi_k (type-shift fit) and phi_0 (pooled fit) row by row.
    base_cols = ['entity_id', 'time_id', 'CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter',
                 'deposit_type', 'is_B', 'deposit_balance', 'lagged_deposits', 'gross_return_lag', 'pop_total']
    keep_cols = list(set(base_cols + [c for c in EXTRA_KEEP_COLS + ['banked_correction'] if c in df_base.columns]))
    df = df_base[keep_cols].copy()
    df['is_B'] = df['is_B'].astype(bool)
    up_ts, up_0 = extract_upsilon_terms(res_ts), extract_upsilon_terms(res_0)
    idx_ts = _index_from(df, df_base, up_ts)
    idx_0 = _index_from(df, df_base, up_0)
    if idx_ts is None or idx_0 is None:
        return None, None, spec_name
    df['_idx_ts'], df['_idx_0'] = idx_ts, idx_0
    df = df.dropna(subset=['_idx_ts', '_idx_0', 'spread_qoq'])
    ktype = df['deposit_type'].astype(int)
    lam = {1: 0.0, **{k: float(up_ts.get(f'dtype_{k}', 0.0)) for k in (2, 4, 5)}}
    free = df['_idx_ts'] - sum(lam[k] * (ktype == k).astype(float) for k in (2, 4, 5))

    df['phi_mt'] = _apply_link(df['_idx_ts'], link, res_ts)              # own type
    df['phi0_mt'] = _apply_link(df['_idx_0'], link, res_0)
    for k in DEP_TYPES:
        df[f'phi_k{k}_mt'] = _apply_link(free + lam[k], link, res_ts)
    nat = {k: _national_phi(df, f'phi_k{k}_mt') for k in DEP_TYPES}
    df['phi_t'] = 0.0
    for k in DEP_TYPES:
        m = ktype == k
        df.loc[m, 'phi_t'] = df.loc[m, 'time_id'].map(nat[k]).fillna(0.0).clip(lower=0.0, upper=1.0).values
    df['phi0_t'] = df['time_id'].map(_national_phi(df, 'phi0_mt')).fillna(0.0).clip(lower=0.0, upper=1.0)

    # 3. Active deposits at the row's own type: phi_k,mt for B rows, the national phi_k,t for D.
    df['Dep_Act'] = 0.0
    val_B = df['deposit_balance'] - df['phi_mt'] * df['gross_return_lag'] * df['lagged_deposits']
    df.loc[df['is_B'], 'Dep_Act'] = np.maximum(0.0, val_B[df['is_B']]).astype(float).values
    val_D = df['deposit_balance'] - df['phi_t'] * df['gross_return_lag'] * df['lagged_deposits']
    df.loc[~df['is_B'], 'Dep_Act'] = np.maximum(0.0, val_D[~df['is_B']]).astype(float).values

    # 4. The arm's correction to the awake pool, from every row's carried lagged balance.
    #    a = pool / pooled pool, one factor per local market (B) and per quarter (national).
    carried = (df['gross_return_lag'] * df['lagged_deposits']).fillna(0.0)
    capd = lambda s: s.clip(lower=0.0, upper=PHI_CAP)
    bmask = df['is_B']
    if arm == 'F1':
        dev = (capd(df['phi_mt']) - capd(df['phi0_mt'])) * carried
        dev_nat = (capd(df['phi_t']) - capd(df['phi0_t'])) * carried
        red_loc = dev[bmask].groupby([df.loc[bmask, 'mca_code'], df.loc[bmask, 'time_id']]).sum()
        red_nat = dev_nat.groupby(df['time_id']).sum()
        a_loc = 1.0 - red_loc.reindex(M_loc.index).fillna(0.0) / (omp0_loc.reindex(M_loc.index) * M_loc)
        a_nat = 1.0 - red_nat.reindex(M_nat.index).fillna(0.0) / (omp0_nat.reindex(M_nat.index) * M_nat)
        df['r4_locked'] = np.nan
    elif arm in ('F3hi', 'F3lo'):
        own, med = locked
        key = pd.MultiIndex.from_arrays([df['CodConglomeradoPrudencial'].astype(str), df['year'].astype(int)])
        r4 = pd.Series(own.reindex(key).values, index=df.index)
        r4 = r4.fillna(df['year'].astype(int).map(med)).astype(float)
        r4 = r4.where(ktype == 4, 0.0)
        df['r4_locked'] = r4
        lk = r4 * carried
        lk_loc = lk[bmask].groupby([df.loc[bmask, 'mca_code'], df.loc[bmask, 'time_id']]).sum()
        lk_nat = lk.groupby(df['time_id']).sum()
        a_loc = 1.0 - lk_loc.reindex(M_loc.index).fillna(0.0) / M_loc
        a_nat = 1.0 - lk_nat.reindex(M_nat.index).fillna(0.0) / M_nat
    else:
        raise ValueError(f"unknown arm {arm!r}")

    df = df.dropna(subset=['Dep_Act'])
    df = df[df['Dep_Act'] > 1e-6].copy()
    df['M_mt'] = pd.Series(M_loc.reindex(pd.MultiIndex.from_frame(df[mkt_keys])).values, index=df.index)
    df['M_nat'] = df['time_id'].map(M_nat)
    if acc_loc is not None:
        df['_access'] = pd.Series(acc_loc.reindex(pd.MultiIndex.from_frame(df[mkt_keys])).values, index=df.index)
    n_no_market = int((df['is_B'] & df['M_mt'].isna()).sum() + (~df['is_B'] & df['M_nat'].isna()).sum())
    df = df[~((df['is_B'] & df['M_mt'].isna()) | (~df['is_B'] & df['M_nat'].isna()))].copy()

    # 5. Pooled denominators (row-level, as the pooled shares use them), times the arm's factor,
    #    floored so the inside shares of every market add up to at most TOTAL_SHARE_CAP.
    omp0_row = 1.0 - np.where(df['is_B'], df['phi0_mt'], df['phi0_t']).clip(0.0, PHI_CAP)
    den0 = omp0_row * np.where(df['is_B'], df['M_mt'], df['M_nat'])
    df['_s0'] = df['Dep_Act'] / den0
    bm = df['is_B']
    s0_loc = df.loc[bm].groupby(mkt_keys)['_s0'].sum()
    s0_nat = df.loc[~bm].groupby('time_id')['_s0'].sum()
    a_loc_f = np.maximum(a_loc.reindex(s0_loc.index), s0_loc / TOTAL_SHARE_CAP)
    a_nat_f = np.maximum(a_nat.reindex(s0_nat.index), s0_nat / TOTAL_SHARE_CAP)
    # counted with a relative tolerance: where the pooled floor already binds, s0/CAP equals the
    # pooled factor 1 up to rounding, and that is not the arm's floor binding
    n_floor_loc = int((a_loc_f > a_loc.reindex(s0_loc.index) * (1.0 + 1e-9)).sum())
    n_floor_nat = int((a_nat_f > a_nat.reindex(s0_nat.index) * (1.0 + 1e-9)).sum())
    a_row = np.where(bm,
                     a_loc_f.reindex(pd.MultiIndex.from_frame(df[mkt_keys])).values,
                     df['time_id'].map(a_nat_f).values)
    df['pool_factor'] = a_row
    df['_1mphi'] = omp0_row * a_row
    df['share_B_cond'] = np.where(bm, df['Dep_Act'] / (df['_1mphi'] * df['M_mt']), np.nan)
    df['share_D'] = np.where(~bm, df['Dep_Act'] / (df['_1mphi'] * df['M_nat']), np.nan)
    df = df.drop(columns=['_idx_ts', '_idx_0', '_s0'])

    spec_id = SPEC_MAP_INV.get(spec_name, spec_name)
    df['Spec_ID'] = spec_id
    a_b = pd.Series(a_row[bm.values])
    summary = {
        "Name": spec_name, "Spec_ID": spec_id, "Arm": arm, "Rows": len(df),
        "B_firms": int(bm.sum()), "D_firms": int((~bm).sum()),
        "Mean_phi_mt": float(df['phi_mt'].mean()), "Mean_phi_t": float(df['phi_t'].mean()),
        "Mean_phi0_mt": float(df['phi0_mt'].mean()),
        "lambda": {f"dtype_{k}": lam[k] for k in (2, 4, 5)},
        "pool_factor_B": {"mean": float(a_b.mean()), "p05": float(a_b.quantile(0.05)),
                          "p50": float(a_b.median()), "p95": float(a_b.quantile(0.95))},
        "pool_factor_national": {str(t): float(v) for t, v in a_nat_f.items()},
        "floor_bound_markets": n_floor_loc, "floor_bound_quarters_national": n_floor_nat,
        "rows_without_pooled_market": n_no_market,
        "B_Cond_Share_NaNs": int(df[bm]['share_B_cond'].isna().sum()),
        "D_Share_NaNs": int(df[~bm]['share_D'].isna().sum()),
    }
    print(f"  [{arm}] lambda {summary['lambda']}; pool factor B mean {a_b.mean():.4f} "
          f"[p05 {a_b.quantile(0.05):.4f}, p95 {a_b.quantile(0.95):.4f}]; floor bound in "
          f"{n_floor_loc} markets, {n_floor_nat} national quarters; {n_no_market} rows without a pooled market")
    return df, summary, spec_id


def _load_results(fits_dir, est_num):
    p = Path(fits_dir) / f"est{est_num}" / "estimation_results.pkl"
    if not p.exists():
        print(f"[!] No pickle at {p}. Run switch_cost_sleep_est_single.py --est {est_num} first.")
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


def run(est_num, link, tag, time_block=False, spec="all", arm="pooled", fits_dir=None,
        pooled_fits_dir=None, locked_csv=None, zero_lock=False):
    """Demand prep for Est{est_num} with the given link. spec = 'all' | 'N' | 'a-b'.
    time_block=True (E4) adds the time block (time_trend + gdp_growth_yoy)
    to the demand frame so phi reconstructs the time interactions.

    arm: 'pooled' | 'F1' | 'F3hi' | 'F3lo' (module docstring). fits_dir holds est{N}/ with the
    fit the arm reads (the pooled fit for 'pooled', the type-shift fit otherwise; read only);
    pooled_fits_dir the pooled fit that supplies phi_0 and the market sizes to F1/F3.
    zero_lock (F3, tests only) sets r_4 = 0, under which F3 must return the pooled parquet."""
    spec = str(spec)
    if spec.lower() == 'all':
        spec_ids = list(range(1, 13))
    elif '-' in spec:
        a, b = map(int, spec.split('-')); spec_ids = list(range(a, b + 1))
    else:
        spec_ids = [int(spec)]
    if arm not in ARMS:
        raise SystemExit(f"--arm must be one of {ARMS} (got {arm!r})")

    # demand_parquet_dir is DEMAND_PREP_DIR, set and checked by the CLI below.
    demand_output_dir = switch_cost_sandbox.check_dir(paths.demand_parquet_dir(), "DEMAND_PREP_DIR")
    demand_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Base Panel {PANEL_CSV}...")
    df_base = build_base_panel(PANEL_CSV, time_block=time_block)
    print(f"Base Panel rows (with valid lagged structure): {len(df_base)}")

    print(f"[{arm}] fits: {fits_dir}" + (f"; pooled fits: {pooled_fits_dir}" if arm != 'pooled' else ""))
    # A missing fit is an error, not an empty success: a job that exits 0 having written
    # nothing reads as done.
    results_dict = _load_results(fits_dir, est_num)
    if results_dict is None:
        raise SystemExit(2)
    pooled_dict, locked = None, None
    if arm != 'pooled':
        pooled_dict = _load_results(pooled_fits_dir, est_num)
        if pooled_dict is None:
            raise SystemExit(2)
    if arm in ('F3hi', 'F3lo'):
        own, med = load_locked(locked_csv, 'hi' if arm == 'F3hi' else 'lo')
        if zero_lock:
            own, med = own * 0.0, med * 0.0
        locked = (own, med)

    total, spec_summaries = 0, {}
    for sid in spec_ids:
        target_name = SPEC_MAP.get(sid)
        if not target_name:
            continue
        key = next((k for k in results_dict if target_name in k), None)
        if not key:
            continue
        if arm == 'pooled':
            df_spec, summary, _ = process_specification(key, results_dict[key], df_base, link)
        else:
            key0 = next((k for k in pooled_dict if target_name in k), None)
            if not key0:
                print(f"   [!] Spec {sid}: no pooled fit to take phi_0 and M from")
                continue
            df_spec, summary, _ = process_specification_arm(key, results_dict[key], pooled_dict[key0],
                                                            df_base, link, arm, locked)
        if df_spec is not None:
            out = demand_output_dir / f"demand_{est_num}_{tag}_spec_{sid}.parquet"
            df_spec = demand_chars.finalize(df_spec, out, summary)
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
        raise SystemExit(3)


# ── Config-driven CLI for E3/E4 demand prep (link, tag, time_block) ──────────────
#  (E1/E2 have their own demand-prep scripts.) Run:  python switch_cost_sleep_demand_prep_link.py --est N --spec X
#
# The link string here is a DEFAULT only: _apply_link prefers the stored result's own tag, so
# a constrained single-index fit ("index_sieve") is evaluated on its I-spline grid even though
# this table says "index". The tag is what lands in the parquet filename
# (demand_{est}_{tag}_spec_{spec}.parquet) and is what the Julia routine discovery keys on.
DEMAND_CFG = {e: ("index", R.demand_tag(e), R.ROUTINES[e].time_block) for e in R.LINK_ESTS}

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    p = argparse.ArgumentParser(description="Switch-cost demand prep E3/E4, one arm per run.")
    p.add_argument("--est", type=int, required=True, choices=sorted(DEMAND_CFG),
                   help="Estimator id 3 or 4 (E1/E2 have their own demand-prep scripts)")
    p.add_argument("--spec", type=str, default="12", help="Specification ID (1-12) or 'all'")
    p.add_argument("--arm", default="pooled", choices=ARMS)
    p.add_argument("--fits-dir", default=None,
                   help="Folder holding est{N}/estimation_results.pkl (read only). Default: the local "
                        "switch_cost_pooled tree for --arm pooled, switch_cost_typeshift otherwise.")
    p.add_argument("--pooled-fits-dir", default=None,
                   help="F1/F3: folder of the pooled fit (phi_0, market sizes). Default: switch_cost_pooled.")
    p.add_argument("--locked-csv", default=None,
                   help="F3: r4_locked_bank_year.csv (switch_cost_tools.py locked). Default: switch_cost_inputs.")
    p.add_argument("--out-dir", default=None,
                   help="Parquet folder; its name must contain 'switch_cost'. "
                        "Default: SWITCH_COST/switch_cost_demand_<arm>.")
    p.add_argument("--zero-lock", action="store_true", help="F3 test only: r_4 = 0 everywhere.")
    a = p.parse_args()
    _base = switch_cost_sandbox.LOCAL_BASE
    fits = a.fits_dir or str(switch_cost_sandbox.local_fit_root("pooled" if a.arm == "pooled" else "typeshift"))
    pooled_fits = a.pooled_fits_dir or str(switch_cost_sandbox.local_fit_root("pooled"))
    locked_csv = a.locked_csv or str(_base / "switch_cost_inputs" / "r4_locked_bank_year.csv")
    # resolved: an unresolved "../.." prefix pushes the audit file past Windows' 260-character path limit
    out_dir = switch_cost_sandbox.check_dir(
        Path(a.out_dir or (_base / f"switch_cost_demand_{a.arm}")).resolve(), "--out-dir")
    os.environ["DEMAND_PREP_DIR"] = str(out_dir)
    _link, _tag, _tb = DEMAND_CFG[a.est]
    run(a.est, _link, _tag, time_block=_tb, spec=a.spec, arm=a.arm, fits_dir=fits,
        pooled_fits_dir=pooled_fits, locked_csv=locked_csv, zero_lock=a.zero_lock)
