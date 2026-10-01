"""
bbl_polfunc.py
================================
Estimates parametric policy functions for endogenous deposit types k=4,5
(BBL Step 1, sec:empirical:cost in V_Main.tex).

Following Egan, Hortacsu & Matvos (2025, NBER), Ryan (2012, Econometrica),
and Matvos & Seru (2014, AER), we estimate flexible parametric mappings from
state variables to equilibrium deposit spreads for deposit types whose rates
are set endogenously by banking institutions (k=4: time/CDB deposits;
k=5: prepaid accounts).

Coefficients are allowed to differ by firm type (B = brick-and-mortar,
D = digital/national) and by deposit type. For D-type firms, which operate
nationally rather than at the MCA level, we test three demographic
specifications:
    Option A — Exclude MCA-level demographics entirely.
    Option B — Use population-weighted national averages of MCA demographics.
    Option C — Pool B and D firms under a single coefficient vector
               (D-firms receive national demographics as their regressors).
Option C is still estimated and kept in the summary/pickle as a robustness record, but it is NOT
shown in the tables: Step 2 consumes only B and D-Option-B, and pooling the few-but-huge B clusters
with the many-small D ones collapses the effective cluster count (G* ~ 9) that the WCB relies on.

Pipeline position
-----------------
  estimation_1_sleep  →  estimation_1_demand  →  **estimation_bbl_1_polfunc**
                          →  estimation_bbl_2_fwd_sim  →  estimation_bbl_3_solve

NO SPEC LABEL.  The policy function never touches a sleepiness specification — it regresses the
observed spread (or deposit rate) on the pricing state — so it is SPEC-INVARIANT and its outputs
carry no `_spec_{id}` suffix. An earlier version replicated identical files under every spec label
for pipeline cosmetics; that was removed (2026-07-20) as misleading.

INFERENCE.  Standard errors are a score/multiplier wild cluster bootstrap at the CONGLOMERATE
level (utils.sleep_links.linear_wild_cluster_bootstrap) — the same scheme and clustering unit as
the sleepiness and BLP stages, so every SE in the paper is produced one way. G and G* are reported
as descriptive cluster-paucity statistics, not as the inference. (Previously: CRVE + t(G*).)
The NATIONAL regressors (the risk-free rate and its square; the `*_natl` demographics of D-Option-B)
take one value per quarter across all firms, so their rows report the same bootstrap clustered on the
QUARTER instead (utils.se_national, the convention of the sleepiness tables), marked with a dagger;
see NATIONAL_BY_NAME.

Usage
-----
  python bbl_polfunc.py                  # compounded annual spread (feeds BBL Step 2)
  python bbl_polfunc.py --depvar rate    # annualized deposit rate (robustness)

References
----------
  Bajari, Benkard & Levin (2007, Econometrica)
  Egan, Hortacsu & Matvos (2025, NBER WP)
  Matvos & Seru (2014, AER)
  Ryan (2012, Econometrica)
  Cameron, Gelbach & Miller (2008); MacKinnon & Webb (2017)  [wild cluster bootstrap]
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import sys
import os
import json
import pickle
import argparse
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import warnings

# Suppress statsmodels rank-deficiency warnings due to cluster corrections
warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")


# ==============================================================================
# 0. Paths & Constants
# ==============================================================================
# Anchors come from utils.paths, the same accessors the sleep stages use. A
# `Path(__file__).parents[N]` walk hard-codes the repo's position inside the data tree, which
# holds on this machine and nowhere else: bbl_job.sh runs this file from HEAD/scripts, where
# parents[2] is the parent of HEAD and the panel read below finds nothing. market_panel_csv() is
# also the single source of truth for WHICH panel every estimation stage reads.
from utils import paths as _paths  # noqa: E402
DATA_DIR = _paths.PROCESSED
PANEL_CSV = _paths.market_panel_csv()

# The panel goes through the cached loader, exactly as estimation_1_sleep /
# estimation_2_sleep read it. It picks the .parquet twin of PANEL_CSV whenever that is
# the authoritative copy, which on the cluster is ALWAYS: the data bundle ships
# market_panel.parquet (39 MB) and no market_panel.csv, so a bare pd.read_csv here finds
# nothing. Locally it is the same frame, read 3-5x faster.
# The guard keeps this file runnable from a checkout where utils/ has not been imported
# yet (the same reason the sleep estimators carry it), falling back to the raw CSV read.
try:
    from utils import load_panel_cached  # noqa: E402
except Exception:
    load_panel_cached = None

# Estimation window [2016, 2024] — defined once in utils/window.py (full rationale + the
# DEMAND_MIN_YEAR / DEMAND_MAX_YEAR env overrides, which it reads).
from utils.window import MIN_YEAR as POLFUNC_MIN_YEAR, MAX_YEAR as POLFUNC_MAX_YEAR  # noqa: E402
from utils.winsorize import winsorize_within_type as _winsorize_within_type  # noqa: E402

# polfunc_dir() is the ONE place this file's outputs are decided, shared with every consumer:
# it defaults to estimation_output()/COST_POLFUNC and honours COST_POLFUNC_DIR.
#
# COST_POLFUNC_DIR is a STEP-LOCATION seam, not a vintage seam — it names which folder of a
# per-step output tree holds this step's artifacts (the cluster's data/output/bbl, where
# bbl_job.sh exports it and bbl_fwd_sim.jl picks polfunc_fitted.csv up via
# --policy-csv). It moves writer and reader together, which is exactly what a vintage override
# would not do here: the policy function is SPEC-INVARIANT (see the module docstring) — it
# regresses the observed spread on the pricing state and reads market_panel.csv, never touching
# an est{e} fit — so it belongs to no sleepiness vintage, and SLEEP_OUT_ROOT is deliberately not
# consulted. Its consumers set no SLEEP_OUT_ROOT, so following the sandbox would aim them at a
# directory this step never fills.
OUTPUT_DIR = _paths.polfunc_dir()

# Endogenous deposit types (spreads set by institutions)
K_ENDOG = [4, 5]
K_LABELS = {4: "Time_CDB", 5: "Prepaid"}

# --------------------------------------------------------------------------
# State variable groups
# --------------------------------------------------------------------------
# Bank-level characteristics (available for all firms)
BANK_CHARS = [
    'log_total_assets_lag', 'equity_ratio_lag',
    'has_ip',
    'seg_S2', 'seg_S3', 'seg_S4', 'seg_S5',
]

# Balance-sheet / risk characteristics (added 2026-07-17; each ~96.5% populated in 2016-2024).
# `age_interpolated` was requested too but is 0% populated in-window, so it is intentionally omitted
# (including an all-NaN regressor would drop the entire complete-case sample).
BALANCE_SHEET = [
    'asset_return_qoq_lag',      # realized portfolio yield (profitability / asset-side pricing)
    'npl_provision_ratio_lag',   # credit-risk / provisioning intensity
    'credit_assets_lag',         # loan intensity (credit / assets)
]

# Operating cost shifters (from COSIF DRE report)
COST_SHIFTERS = [
    'personnel_cost_ratio_lag', 'admin_cost_ratio_lag', 'tax_cost_ratio_lag',
]

# Capital adequacy & wholesale funding
CAPITAL_WHOLESALE = [
    'indice_basileia_lag', 'wholesale_ratio_lag', 'lci_lca_ratio_lag',
]

# Market-level demographics (MCA-level for B-firms, aggregated for D-firms)
DEMOGRAPHICS = [
    'gdp_per_capita', 'fraction_65plus', 'fraction_young',
    'pix_users_pf_per1000', 'connections_per100',
    'branches_per1000', 'cadunico_families_per1000',
]

# Macro variable
MACRO = ['risk_free_qoq']

# ---- Outlier control ------------------------------------------------------------------------
# The accounting ratios carry a small number of corrupt observations, concentrated among D firms:
# `indice_basileia_lag` reaches 16,069.7 for D (sd 205.6) against a B-firm max of 7.10 (sd 0.05),
# and `personnel_cost_ratio_lag` reaches 2.32 (232% of assets). These are near-zero-denominator
# artifacts, and because leverage concentrates in a handful of rows they both drive the point
# estimates and collapse the wild-cluster-bootstrap SEs (the Basel row printed 0.0001 with a
# 0.0000 SE at *** before this was applied).
#
# They are winsorized at the 1st/99th percentile SEPARATELY WITHIN B and D. Within-type is the
# point: B and D have genuinely different balance sheets (D-firm equity ratios are ~4x B-firm
# ones), so pooled percentiles would clip real cross-type variation instead of the corrupt tail.
# Demographics are NOT winsorized (market-level, clean), nor is risk_free_qoq (macro), nor
# log_total_assets_lag (already a log). Quadratic terms are rebuilt FROM the winsorized bases.
WINSOR_PCT = 0.01
WINSOR_VARS = BALANCE_SHEET + COST_SHIFTERS + CAPITAL_WHOLESALE + ['equity_ratio_lag']

# ---- Centering -------------------------------------------------------------------------------
# Continuous regressors are demeaned WITHIN each estimation sample, so the intercept is the
# prediction at the AVERAGE state instead of at x=0. Uncentered, x=0 means a bank with R$1 of
# assets (log assets = 0), far outside the support: the k=4 B-type constant was +170pp purely to
# offset the log-asset terms (-375pp linear, +209pp quadratic, netting ~+3pp at the median).
#
# This is a REPARAMETRIZATION, not a different model. span{1, x, x^2} == span{1, x-xbar,
# (x-xbar)^2}, so R^2, residuals and every fitted value are unchanged -- which is exactly what
# the fitted-values identity check in main() verifies against the pre-centering policy CSV.
# Slopes are unchanged too, EXCEPT the linear terms of the three variables that also enter
# quadratically (log assets, equity ratio, risk-free): those become the marginal effect AT THE
# MEAN rather than at zero, which is the interpretable margin anyway.
#
# Dummies are NOT centered, so the intercept keeps a reference-category reading: segment S1,
# no IP subsidiary, every continuous regressor at its mean. Centering them would turn the
# intercept into a grand mean and throw that reading away.
NO_CENTER_VARS = {'has_ip', 'seg_S2', 'seg_S3', 'seg_S4', 'seg_S5'}

# Quadratic terms (appended as _sq)
QUADRATIC_BASE = ['log_total_assets_lag', 'equity_ratio_lag', 'risk_free_qoq']

# Full non-demographic regressors (common to all firm types)
COMMON_REGRESSORS = BANK_CHARS + BALANCE_SHEET + COST_SHIFTERS + CAPITAL_WHOLESALE + MACRO

# D-firm demographic options
D_DEMO_OPTIONS = ['A', 'B', 'C']
D_DEMO_LABELS = {
    'A': 'No demographics',
    'B': 'National (pop-weighted) demographics',
    'C': 'Pooled with B-type coefficients',
}

# ---- Regressand configuration (the default; overridden by --depvar in main) ----------------
# Default LHS is the COMPOUNDED annual deposit spread `spread_ann` = the panel's spread_ann_a{k} =
# (1 + r^f_q)^4 - (1 + r^dep_q)^4, a fraction: the definition of the demand price (blp_logit.jl reads
# the same column) and of the simulator's rho (cf_deposit_sim.jl _rdep_from_annual inverts exactly
# this). It is the policy fed to BBL Step 2. `--depvar rate` switches the LHS to the ANNUALIZED
# deposit rate rate_ann = (1+rate_qoq)^4 - 1 and writes to a SEPARATE `polfunc_rate_*` namespace, so
# it never overwrites the spread policy Step 2 consumes.
CFG = {
    'regressand':  'spread_ann',
    'out_prefix':  'polfunc',
    'tab_label':   'tab:polfunc',
    'caption':     'Policy Function Estimates for Endogenous Deposit Spreads (BBL Step~1)',
    'lhs_title':   'Deposit Spread',
    # The regressand's noun in the notes' sentences ("B firms' prepaid spread ...").
    'lhs_noun':    'spread',
    # Appended to the table caption. Empty for the spread tables (the headline): their caption is
    # just "Policy Function Estimates: <deposit type>".
    'caption_suffix': '',
    # 100 = fraction -> pp: the regressand is already the compounded annual spread.
    'lhs_display': 100.0,
    'lhs_unit':    'pp, compounded annual',
    'lhs_short':   'compounded annual deposit spread',
    # The prepaid (k=5) spread is itself about zero, so at three decimals in pp most of its table
    # would print 0.000: that table -- coefficients, SEs, the mean -- is shown in BASIS POINTS
    # (user decision 2026-09-29), x100 on the pp display. C.10 (k=4) stays in pp.
    'lhs_display_by_k': {5: 10000.0},
    'lhs_unit_name':    'percentage points',
    'lhs_unit_name_by_k': {5: 'basis points'},
    # {unit} is filled per deposit type from lhs_unit_name(_by_k).
    'depvar_note': (r'The dependent variable is the deposit spread '
                    r'$\rho_{jkmt} = r^f_t - r^{\mathrm{dep}}_{jkmt}$, annualized by compounding, '
                    r'$(1 + r)^4 - 1$, in {unit}, the same definition as the demand price.'),
    # The fitted-policy CSV (the contract with bbl_fwd_sim.jl): the observed column it has always
    # carried, and the unit of its fitted_* columns, written on every row as `lhs_unit`.
    'csv_obs_col': 'spread_qoq',
    'fitted_unit': 'spread_ann_frac',
}

_CFG_RATE = {
    'regressand':  'rate_ann',
    'out_prefix':  'polfunc_rate',
    'tab_label':   'tab:polfunc_rate',
    'caption':     'Policy Function Estimates for the Annualized Deposit Rate (BBL Step~1)',
    'lhs_title':   'Annualized Deposit Rate',
    'lhs_noun':    'rate',
    # NON-empty here on purpose: the rate tables are a robustness variant of the same regressions,
    # so without an LHS descriptor their captions would be identical to the spread tables'.
    'caption_suffix': r' --- Annualized Deposit Rate',
    # 100 = fraction -> pp. The regressand is ALREADY exactly compounded (rate_ann), so unlike the
    # spread variant this is the exact annualized rate, not a simple-annualization approximation.
    'lhs_display': 100.0,
    # No per-type override: the prepaid deposit RATE is not near zero. Explicit, because --depvar
    # rate applies this dict with CFG.update, which would otherwise keep the spread's k=5 override.
    'lhs_display_by_k': {},
    'lhs_unit_name':    'percentage points',
    'lhs_unit_name_by_k': {},
    'lhs_unit':    'pp of the annualized deposit rate',
    'lhs_short':   'annualized deposit rate',
    'depvar_note': (r'Dependent variable: annualized deposit rate '
                    r'$r^{\mathrm{dep,ann}}_{jkmt}=(1+r^{\mathrm{dep}}_{jkmt})^{4}-1$, 2016--2024.'),
    'csv_obs_col': 'rate_ann',
    'fitted_unit': 'rate_ann_frac',
}


def _lhs_display(k: int, cfg: dict | None = None) -> float:
    """The dependent variable's display multiplier for deposit type k: the variant's lhs_display,
    or its per-type override (k=5 spread: basis points)."""
    c = CFG if cfg is None else cfg
    return float((c.get('lhs_display_by_k') or {}).get(k, c.get('lhs_display', 1.0)))


def _lhs_unit_name(k: int, cfg: dict | None = None) -> str:
    c = CFG if cfg is None else cfg
    return (c.get('lhs_unit_name_by_k') or {}).get(k, c.get('lhs_unit_name', 'percentage points'))


def _unit_clause(k: int) -> str:
    """The unit the note states for deposit type k. A type shown in a unit other than the
    variant's default (the k=5 spread, in basis points) also names its sibling table's unit, so
    the two tables of the same spread are not read as contradictory (reviewer, 2026-09-30):
    'basis points (Table~\\ref{tab:polfunc_k4}: percentage points)'."""
    name = _lhs_unit_name(k)
    default = CFG.get('lhs_unit_name', 'percentage points')
    # Rendered abbreviated (the text defines pp and bp once); the full names stay the stored values.
    unit = _UNIT_ABBR.get(name, name)
    if name == default:
        return unit
    sib = next((j for j in K_ENDOG if j != k and _lhs_unit_name(j) == default), None)
    return unit if sib is None else unit + rf" (Table~\ref{{{CFG['tab_label']}_k{sib}}}: {_UNIT_ABBR.get(default, default)})"


# ==============================================================================
# 1. Data Loading & Preparation
# ==============================================================================
def winsorize_within_type(df: pd.DataFrame, pct: float = None, verbose: bool = True) -> pd.DataFrame:
    """Clip WINSOR_VARS to their [pct, 1-pct] quantiles separately within B and within D.

    Delegates to utils.winsorize so the rule has ONE implementation. market_panel.csv already
    arrives bounded and winsorized from panel_7_instruments, so on a current panel this is a
    near no-op that clips only what the reshape to k=4,5 rows re-exposes; it is kept because
    this frame is the one the policy function is actually fitted on, and a stale panel would
    otherwise reach the regression untreated.

    Returns the same frame (modified in place), with an audit line per clipped variable."""
    return _winsorize_within_type(df, WINSOR_VARS,
                                  pct=WINSOR_PCT if pct is None else pct,
                                  type_key='is_B', verbose=verbose)


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
    print("[WARNING] Column 'is_B' not found in the market panel: this panel predates the "
          "stored firm-type column. Falling back to the CODMUN_IBGE sentinel; re-run "
          "panel_market.py to store the authoritative column.")
    return df['CODMUN_IBGE'].astype(str) != '0'


def _check_compounded_spread(df: pd.DataFrame, tol: float = 1e-10) -> None:
    """The regressand must be the compounded annual spread of the rows' own quarterly rates:
    spread_ann = (1 + r^f_q)^4 - (1 + r^f_q - spread_qoq)^4 on every estimation row. Measured on
    the 2016-2024 panel the largest deviation is at machine precision (~1e-16); a larger one means
    the panel's spread_ann_a{k} was built from different rates than spread_a{k}, and the fit stops."""
    rf = pd.to_numeric(df['risk_free_qoq'], errors='coerce')
    sq = pd.to_numeric(df['spread_qoq'], errors='coerce')
    implied = (1.0 + rf) ** 4 - (1.0 + rf - sq) ** 4
    diff = (pd.to_numeric(df['spread_ann'], errors='coerce') - implied).abs()
    ok = diff.notna()
    worst = float(diff[ok].max()) if ok.any() else float('nan')
    print(f"  spread_ann vs (1+r^f)^4-(1+r^f-spread_qoq)^4 on {int(ok.sum()):,} rows: "
          f"max |diff| = {worst:.2e}  ({int((~ok).sum()):,} rows without r^f or spread_qoq)")
    if not worst <= tol:
        raise SystemExit(f"[FATAL] spread_ann is not the compounded annual spread of spread_qoq "
                         f"(max |diff| {worst:.3e} > {tol:g}): the panel's spread columns disagree.")


def load_and_prepare_panel() -> pd.DataFrame:
    """Load the market panel, reshape to long for k=4,5, and construct lags.

    Returns a long-format DataFrame with one row per (conglomerate, deposit_type,
    mca_code, year, quarter) tuple, restricted to deposit types 4 and 5.
    """
    print(f"Loading panel from {PANEL_CSV}...")
    df_raw = (load_panel_cached(PANEL_CSV, dtype={'mca_code': str}, low_memory=False)
              if load_panel_cached
              else pd.read_csv(PANEL_CSV, dtype={'mca_code': str}, low_memory=False))
    print(f"  Raw panel: {len(df_raw):,} rows, {len(df_raw.columns)} columns")

    # Restrict to the estimation window (2016-2024), matching the sleep/demand/cost stages.
    # This is not cosmetic: the policy regressors are bank characteristics, which do not exist on the
    # prudential-conglomerate basis before 2016.  Fitting/predicting outside the window drove
    # compute_fitted_values' fillna(0) to return the bare intercept (~190pp — an impossible deposit
    # spread) for ~19% of k∈{4,5} rows, all of them 2013-2015.  Windowing removes that fabrication at
    # source rather than filtering it downstream.  See counterfactuals_plan.md §0A.
    if 'year' in df_raw.columns:
        _n0 = len(df_raw)
        df_raw = df_raw[(df_raw['year'] >= POLFUNC_MIN_YEAR) & (df_raw['year'] <= POLFUNC_MAX_YEAR)].copy()
        print(f"  Year window [{POLFUNC_MIN_YEAR}, {POLFUNC_MAX_YEAR}]: kept {len(df_raw):,} of {_n0:,} rows")

    # ----- Reshape wide → long for k=4,5 ---------------------------------
    id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
    df_raw = df_raw.drop_duplicates(subset=id_vars)

    # spread_ann_a{k} is the panel's compounded annual spread (the default regressand). The stub
    # regexes are anchored (^stub\d+$), so 'spread_a' never picks up spread_ann_a{k} and neither
    # stub picks up the *_national columns.
    df = pd.wide_to_long(
        df_raw,
        stubnames=['dep_a', 'spread_a', 'rate_a', 'spread_ann_a'],
        i=id_vars,
        j='deposit_type'
    ).reset_index()

    df = df.rename(columns={'dep_a': 'deposit_balance', 'spread_a': 'spread_qoq',
                            'rate_a': 'rate_qoq', 'spread_ann_a': 'spread_ann'})

    # Restrict to endogenous deposit types
    df = df[df['deposit_type'].isin(K_ENDOG)].copy()
    print(f"  After restricting to k in {K_ENDOG}: {len(df):,} rows")

    # Annualized deposit rate (compound) — the alternative regressand for --depvar rate.
    # Verified to machine precision that the panel builds spread_ann = (1+rf_qoq)^4 - (1+rate_qoq)^4,
    # so this annualizes the deposit rate the SAME way the panel annualizes the spread.
    df['rate_qoq'] = pd.to_numeric(df['rate_qoq'], errors='coerce')
    df['rate_ann'] = (1.0 + df['rate_qoq']) ** 4 - 1.0

    # ----- Firm type classification ---------------------------------------
    # Read the panel's stored verdict; see _resolve_is_B.
    df['is_B'] = _resolve_is_B(df)

    # ----- Entity key & time identifiers ----------------------------------
    df['entity_id'] = (
        df['CodConglomeradoPrudencial'].astype(str) + '_'
        + df['deposit_type'].astype(str) + '_'
        + df['mca_code'].astype(str)
    )
    df['time_id'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)
    df.sort_values(by=['entity_id', 'year', 'quarter'], inplace=True)

    # ----- Coerce all regressor columns to numeric ------------------------
    # Some columns (e.g. has_ip) may arrive as object dtype from the CSV.
    all_regressor_cols = (
        COMMON_REGRESSORS + DEMOGRAPHICS
        + [f'{c}_sq' for c in QUADRATIC_BASE]
    )
    for col in all_regressor_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ----- Winsorize the accounting ratios within firm type ---------------
    df = winsorize_within_type(df)

    # ----- Quadratic terms (built AFTER winsorizing, so squares of a clipped
    #       base stay consistent with the base itself) ---------------------
    for col in QUADRATIC_BASE:
        if col in df.columns:
            df[f'{col}_sq'] = df[col].astype(float) ** 2

    # ----- Drop rows with missing dependent variable ----------------------
    regressand = CFG['regressand']
    n_before = len(df)
    df = df.dropna(subset=[regressand])
    print(f"  Dropped {n_before - len(df):,} rows with missing {regressand}")
    if regressand == 'spread_ann':
        _check_compounded_spread(df)

    print(f"  Final panel: {len(df):,} rows "
          f"(B={df['is_B'].sum():,}, D={(~df['is_B']).sum():,})")
    return df


# ==============================================================================
# 2. National Demographics Computation
# ==============================================================================
def compute_national_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute population-weighted national averages of MCA demographics.

    For each (year, quarter), average across MCAs weighted by pop_total.
    These national values are then assigned to D-type firm rows.

    Returns the input DataFrame with additional columns `{demo}_natl` for
    each demographic variable.
    """
    print("  Computing population-weighted national demographics...")

    # Use only B-firm rows with valid population for weighting
    df_b = df[df['is_B']].copy()
    demo_plus_pop = DEMOGRAPHICS + ['pop_total']
    cols_needed = ['year', 'quarter', 'mca_code'] + demo_plus_pop
    cols_avail = [c for c in cols_needed if c in df_b.columns]
    df_mca = df_b[cols_avail].drop_duplicates(subset=['mca_code', 'year', 'quarter'])
    df_mca = df_mca.dropna(subset=['pop_total'])

    def _weighted_mean(group: pd.DataFrame, col: str) -> float:
        vals = group[col]
        wgts = group['pop_total']
        valid = vals.notna() & wgts.notna() & (wgts > 0)
        if valid.sum() == 0:
            return np.nan
        return np.average(vals[valid], weights=wgts[valid])

    natl_records = []
    for (yr, qtr), grp in df_mca.groupby(['year', 'quarter']):
        rec = {'year': yr, 'quarter': qtr}
        for col in DEMOGRAPHICS:
            if col in grp.columns:
                rec[f'{col}_natl'] = _weighted_mean(grp, col)
            else:
                rec[f'{col}_natl'] = np.nan
        natl_records.append(rec)

    df_natl = pd.DataFrame(natl_records)

    # Merge national demographics onto the main panel
    df = df.merge(df_natl, on=['year', 'quarter'], how='left')

    n_filled = sum(1 for c in DEMOGRAPHICS if f'{c}_natl' in df.columns)
    print(f"    Created {n_filled} national demographic columns")

    return df


# ==============================================================================
# 3. Cluster-Robust Standard Errors (IK2016 / Carter et al. 2017)
# ==============================================================================
def cluster_structure(cluster_series: pd.Series):
    """Descriptive cluster structure: (G, G*) with G* = G/(1+cv²) the effective cluster
    count of Imbens-Kolesar (2016) / Carter-Schnepel-Steigerwald (2017).

    NOTE (2026-07-20): G* is now reported as a DESCRIPTIVE statistic only — it is the
    cluster-paucity measure that MOTIVATES the wild cluster bootstrap (see sleep_desc_clusters.py), not
    the inference itself. Inference used to be CRVE + t(G*) here; it is now the same
    score/multiplier wild cluster bootstrap the rest of the paper uses, so every standard
    error in the paper is produced by one scheme. See run_single_regression.
    """
    sizes = cluster_series.value_counts()
    G_nominal = int(len(sizes))
    cv_Ng = float(np.std(sizes, ddof=0) / np.mean(sizes)) if np.mean(sizes) > 0 else 0.0
    G_star = float(max(1.0, G_nominal / (1 + cv_Ng ** 2)))
    return G_nominal, G_star


# ==============================================================================
# 4. Policy Function OLS Estimation
# ==============================================================================
def _build_regressor_list(firm_type: str, d_option: str) -> list:
    """Determine which regressors enter the policy function regression.

    Parameters
    ----------
    firm_type : {'B', 'D', 'pooled'}
    d_option  : {'A', 'B', 'C'}  (only relevant for D and pooled)

    Returns a list of column names to use as regressors (will be intersected
    with available columns at estimation time).
    """
    quad_cols = [f'{c}_sq' for c in QUADRATIC_BASE]
    base = COMMON_REGRESSORS + quad_cols

    if firm_type == 'B':
        # B-firms always get local MCA demographics
        return base + DEMOGRAPHICS

    elif firm_type == 'D':
        if d_option == 'A':
            # No demographics
            return base
        elif d_option == 'B':
            # National (pop-weighted) demographics
            return base + [f'{c}_natl' for c in DEMOGRAPHICS]
        else:
            # Option C handled via pooled
            raise ValueError("D-firm option C should use pooled estimation")

    elif firm_type == 'pooled':
        # Same regressors as B-firms; D-firms get national demos mapped
        # to the same column names (done at data prep)
        return base + DEMOGRAPHICS

    else:
        raise ValueError(f"Unknown firm_type: {firm_type}")


def apply_centering(frame: pd.DataFrame, cols: list, means: dict) -> pd.DataFrame:
    """Subtract `means` from the continuous columns of `frame`, then REBUILD every `_sq` column
    from its freshly centered base.

    Shared by estimation and prediction so both use the identical transform -- the prediction
    path must reuse the ESTIMATION means (passed in), never recompute its own, or the fitted
    values silently shift. Modifies and returns `frame`.
    """
    # Bases first...
    for c in cols:
        if c.endswith('_sq') or c not in means or c not in frame.columns:
            continue
        frame[c] = frame[c] - means[c]
    # ...then the squares, from the now-centered bases.
    for c in cols:
        if not c.endswith('_sq'):
            continue
        base = c[:-3]
        if base in frame.columns and base in means:
            frame[c] = frame[base] ** 2
    return frame


def centering_means(df_w: pd.DataFrame, avail: list) -> dict:
    """Mean of every centerable regressor in this estimation sample (dummies and `_sq` excluded;
    the squares are not centered, they are rebuilt from centered bases)."""
    return {c: float(df_w[c].mean()) for c in avail
            if c not in NO_CENTER_VARS and not c.endswith('_sq') and c in df_w.columns}


# ---- National regressors ---------------------------------------------------------------------
# Regressors that take ONE value per quarter across all firms: the risk-free rate and its square
# (rebuilt from the centred base, still a function of the quarter alone) in every column, and the
# population-weighted national demographics `*_natl` of D-Option-B, merged on (year, quarter) in
# compute_national_demographics. The pooled Option-C demographics are NOT national: its B rows keep
# their local MCA values, so they vary within a quarter. Clustering on the conglomerate treats each
# firm's copy of a national value as independent evidence, so these rows report the wild bootstrap
# clustered on the QUARTER (utils.se_national.attach_national_ses, the convention of the sleepiness
# tables), marked with a dagger. utils.se_national.NATIONAL_VARS holds the sleepiness names
# (pix_exists, risk_free_qoq_lag), so the policy function keeps its own rule here.
NATIONAL_BY_NAME = {'risk_free_qoq', 'risk_free_qoq_sq'}


def _is_national_name(v: str) -> bool:
    return v in NATIONAL_BY_NAME or v.endswith('_natl')


def national_regressors(X: pd.DataFrame, periods, label: str = '') -> list:
    """The regressors of one estimation sample that are NATIONAL: named so (NATIONAL_BY_NAME or a
    `_natl` suffix) AND, in this sample, constant within every quarter while varying across
    quarters. A column constant in both directions (Segment S5 among B firms is all zero) is
    degenerate, not national. A regressor counts only when the name and the data agree; every
    disagreement is printed."""
    per = pd.Series(np.asarray(periods)).astype(str).values
    out = []
    for v in X.columns:
        if v == 'const':
            continue
        x = X[v].to_numpy(dtype=float)
        g = pd.Series(x).groupby(per)
        within = float((g.max() - g.min()).max())
        means = g.mean()
        between = float(means.max() - means.min())
        tol = 1e-12 * max(1.0, float(np.abs(x).max()))
        time_only = within <= tol and between > tol
        named = _is_national_name(v)
        if named and time_only:
            out.append(v)
        elif named != time_only:
            print(f"    [{label}] national check: {v} is {'' if named else 'not '}named national but "
                  f"its within-quarter range is {within:.3g} and between-quarter range {between:.3g}"
                  f"; it keeps conglomerate clustering")
    return out


def run_single_regression(
    df_sub: pd.DataFrame,
    dep_var: str,
    regressors: list,
    label: str,
) -> dict:
    """Run a single OLS policy function regression with cluster-robust SEs.

    Parameters
    ----------
    df_sub     : subset of panel for this regression
    dep_var    : dependent variable column name
    regressors : list of regressor column names
    label      : human-readable label for logging

    Returns
    -------
    dict with keys: 'label', 'res' (statsmodels result), 'n_obs', 'n_clusters',
    'G_star', 'r_squared', 'regressors', 'coefficients', 'std_errors', 'pvalues'
    """
    # Intersect with available columns
    avail = [c for c in regressors if c in df_sub.columns]
    missing = set(regressors) - set(avail)
    if missing:
        print(f"    [{label}] Regressors not in data (dropped): {missing}")

    # ESTIMATION SAMPLE: rows with a POSITIVE balance of the deposit type. A firm-market-quarter
    # that holds none of the type offers no product, so its recorded spread is not a price: among
    # the D firms' k=4 rows two in three had a zero balance, and on those with a zero rate the
    # "spread" was the risk-free rate itself (2026-09-30). The fit sees priced rows only; the
    # fitted values are still computed for EVERY row, from these coefficients
    # (compute_fitted_values), because the forward simulation masks the dead firm-quarters
    # downstream rather than dropping them. A missing or non-numeric balance counts as not positive.
    if 'deposit_balance' not in df_sub.columns:
        raise KeyError(f"[{label}] no deposit_balance column: the estimation sample is the rows "
                       f"with a positive balance of the type")
    positive = pd.to_numeric(df_sub['deposit_balance'], errors='coerce') > 0
    n_nonpositive = int((~positive & df_sub[dep_var].notna()).sum())
    df_sub = df_sub[positive]

    # Working copy with complete cases
    cols_needed = [dep_var] + avail + ['CodConglomeradoPrudencial', 'year']
    df_w = df_sub[cols_needed].dropna().copy()

    if len(df_w) < len(avail) + 1:
        print(f"    [{label}] Insufficient observations ({len(df_w)}). Skipping.")
        return None

    # CENTER the continuous regressors (see NO_CENTER_VARS above). Pure reparametrization:
    # the intercept becomes the prediction at the average state; fit and fitted values are
    # unchanged. The means are stored and REUSED verbatim at prediction time.
    center_means = centering_means(df_w, avail)
    df_w = apply_centering(df_w, avail, center_means)

    y = df_w[dep_var]
    X = sm.add_constant(df_w[avail], has_constant='add')

    # CLUSTER AT THE CONGLOMERATE LEVEL — the same unit as every other stage of the paper
    # (estimation_{1,2}_sleep.py first/second stages, the BLP stage). This previously
    # clustered on bank×year, a finer and therefore LESS CONSERVATIVE unit, which left the
    # policy function inconsistent with everything it feeds.
    cluster = df_w['CodConglomeradoPrudencial'].astype(str)
    model = sm.OLS(y, X)
    res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster}, use_t=True)

    # G / G*: descriptive cluster structure (the paucity that motivates the WCB), not the inference.
    G_nominal, G_star = cluster_structure(cluster)

    # INFERENCE: score/multiplier wild cluster bootstrap — the SAME function the sleepiness
    # stages call (utils.sleep_links; Cameron-Gelbach-Miller 2008, MacKinnon-Webb 2017), and
    # the same B/scheme config (SLEEP_BOOT_B / SLEEP_BOOT_SCHEME), so every SE in the paper is
    # produced one way. Replaces the old CRVE + t(G*) ("IK2016") path.
    from utils.sleep_links import boot_cfg, linear_wild_cluster_bootstrap
    B, scheme = boot_cfg()
    bse, tvals, pvals = linear_wild_cluster_bootstrap(res, B=B, scheme=scheme, seed=0)

    coef_names = list(res.params.index)
    std_errors = {k: float(v) for k, v in bse.items()}
    pvalues = {k: float(v) for k, v in pvals.items()}
    se_scheme = {k: 'congl' for k in coef_names}
    std_errors_dk, n_periods = {}, None

    # NATIONAL ROWS (see NATIONAL_BY_NAME): the same wild bootstrap clustered on the QUARTER, by
    # utils.se_national.attach_national_ses -- the call sleep_est_e2 makes, with the same B, scheme
    # and seed -- replaces the conglomerate SE and p-value of these rows only; the conglomerate
    # numbers of every row stay in std_errors_congl / pvalues_congl. The quarter is read off the
    # rows that survived dropna, so the estimation sample and every estimate are those of the fit.
    # BREAD: statsmodels' normalized_cov_params, (X'X)^{-1} from the SVD of X, and not
    # pinv(X'X). The regressors are in raw panel units (GDP per capita in R$, the squared quarterly
    # rate ~1e-4), so the eigenvalues of X'X span up to 26 orders of magnitude and pinv's cut-off
    # (1e-15 of the largest) drops the squared rate's direction: its bootstrap draws collapse to
    # ~0 whatever the clustering. From X itself the inverse matches a column-equilibrated one to
    # ~1e-12 on every identified row (measured 2026-09-29).
    periods = df_sub.loc[df_w.index, 'time_id'].astype(str).values
    national = national_regressors(X, periods, label)
    if national:
        from utils.se_national import attach_national_ses
        Xa = np.asarray(res.model.exog, float)
        ua = np.asarray(res.resid, float)
        attach_national_ses(res, Xa * ua[:, None], np.asarray(res.normalized_cov_params, float),
                            periods, np.asarray(res.params, float), coef_names,
                            B=B, scheme=scheme, seed=0, label=label)
        bse_t, pv_t = getattr(res, 'bse_time', None), getattr(res, 'pvalues_time', None)
        if bse_t is None or pv_t is None:
            print(f"    [{label}] quarter-clustered WCB unavailable: the national rows keep "
                  f"conglomerate clustering (no dagger)")
        else:
            n_periods = int(res.n_periods)
            bse_dk = getattr(res, 'bse_dk', None)
            for v in national:
                std_errors[v], pvalues[v], se_scheme[v] = float(bse_t[v]), float(pv_t[v]), 'quarter'
                if bse_dk is not None:
                    std_errors_dk[v] = float(bse_dk[v])
                print(f"    [{label}] national {v}: SE congl {bse[v]:.4g} -> quarter "
                      f"{std_errors[v]:.4g} (DK {std_errors_dk.get(v, float('nan')):.4g}), "
                      f"p {pvals[v]:.3g} -> {pvalues[v]:.3g}, T={n_periods}")

    quarter_rows = [v for v in coef_names if se_scheme[v] == 'quarter']
    out = {
        'label': label,
        'res': res,
        'n_obs': int(res.nobs),
        'n_clusters': G_nominal,
        'G_star': G_star,
        'r_squared': float(res.rsquared),
        'r_squared_adj': float(res.rsquared_adj),
        'regressors': coef_names,
        'coefficients': res.params.to_dict(),
        # The REPORTED standard errors and p-values: conglomerate WCB, quarter WCB on the rows
        # se_scheme marks 'quarter'.
        'std_errors': std_errors,
        'pvalues': pvalues,
        'se_scheme': se_scheme,
        # The conglomerate WCB of every row, national ones included; Driscoll-Kraay of the national
        # rows (a comparison, not reported); the number of quarters the quarter WCB clusters on.
        'std_errors_congl': {k: float(v) for k, v in bse.items()},
        'pvalues_congl': {k: float(v) for k, v in pvals.items()},
        'std_errors_dk': std_errors_dk,
        'n_periods': n_periods,
        # Rows with the regressand but a zero, missing or non-numeric balance of the type, left
        # out of the fit (the estimation sample is the rows with a positive balance).
        'n_nonpositive_balance': n_nonpositive,
        'se_method': (f'wild cluster bootstrap (B={B}, {scheme}); clusters = conglomerate'
                      + (f'; national rows ({", ".join(quarter_rows)}): clusters = quarter '
                         f'(T={n_periods})' if quarter_rows else '')),
        # Anchor for reading the intercept. The reported constant is the prediction at x=0, which
        # is far outside the support (log assets = 0 means assets of R$1), so it is large and not
        # interpretable on its own -- for B/k=4 it is +170pp purely to offset the log-assets terms
        # (-375pp linear, +209pp quadratic, netting ~+3pp at the median). OLS with an intercept
        # forces mean(fitted) == mean(y), so the fitted value AT THE SAMPLE MEAN state is exactly
        # this number; reporting it gives the reader the scale anchor the constant does not.
        'mean_depvar': float(y.mean()),
        # The estimation-sample means used to center; compute_fitted_values MUST reuse these.
        'center_means': center_means,
    }

    print(f"    [{label}] N={out['n_obs']:,} | R²={out['r_squared']:.4f} "
          f"| G={out['n_clusters']} | G*={out['G_star']:.2f} "
          f"| left out with no positive balance: {n_nonpositive:,}")

    return out


def _prepare_pooled_data(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Prepare pooled dataset for Option C: map national demographics into
    the same column names as local demographics for D-firm rows."""
    df_k = df[df['deposit_type'] == k].copy()

    # For D-firm rows, replace local demographic columns with national values
    d_mask = ~df_k['is_B']
    for col in DEMOGRAPHICS:
        natl_col = f'{col}_natl'
        if natl_col in df_k.columns and col in df_k.columns:
            df_k.loc[d_mask, col] = df_k.loc[d_mask, natl_col]

    return df_k


def estimate_all_policy_functions(df: pd.DataFrame) -> dict:
    """Run all policy function regressions across deposit types and options.

    Returns a dict keyed by regression label with regression result dicts.
    """
    results = {}
    tasks = []

    for k in K_ENDOG:
        k_label = K_LABELS[k]
        df_k = df[df['deposit_type'] == k]
        df_B = df_k[df_k['is_B']].copy()
        df_D = df_k[~df_k['is_B']].copy()

        # ---- B-type regression (same across all D-options) ---------------
        regs_B = _build_regressor_list('B', 'A')
        label_B = f"k{k}_{k_label}_B"
        tasks.append((df_B, CFG['regressand'], regs_B, label_B))

        # ---- D-type regressions (options A and B) ------------------------
        for opt in ['A', 'B']:
            regs_D = _build_regressor_list('D', opt)
            label_D = f"k{k}_{k_label}_D_opt{opt}"
            tasks.append((df_D, CFG['regressand'], regs_D, label_D))

        # ---- Pooled regression (option C) --------------------------------
        df_pooled = _prepare_pooled_data(df, k)
        regs_pooled = _build_regressor_list('pooled', 'C')
        label_pooled = f"k{k}_{k_label}_pooled_optC"
        tasks.append((df_pooled, CFG['regressand'], regs_pooled, label_pooled))

    # Run regressions in parallel via ThreadPoolExecutor
    # (statsmodels OLS releases GIL during BLAS calls)
    print(f"\n  Submitting {len(tasks)} regressions to ThreadPoolExecutor...")
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 4)) as pool:
        futures = {
            pool.submit(run_single_regression, *task): task[3]
            for task in tasks
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                res = future.result()
                if res is not None:
                    results[label] = res
            except Exception as exc:
                print(f"    [{label}] FAILED: {exc}")

    elapsed = time.perf_counter() - t0
    print(f"  All regressions completed in {elapsed:.1f}s")

    return results


# ==============================================================================
# 5. Fitted Values & Output
# ==============================================================================
def compute_fitted_values(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Compute fitted spreads from estimated policy functions.

    For each regression result, produce predicted values for the corresponding
    subset of the data. Returns a DataFrame with fitted value columns appended.
    """
    # The CSV's columns are a contract with bbl_fwd_sim.jl (_load_policy_map): the identifiers, the
    # observed column it has always carried (csv_obs_col), the fitted_* columns in the REGRESSAND's
    # unit, and `lhs_unit`, which names that unit on every row (spread_ann_frac: the compounded
    # annual spread as a fraction).
    df_out = df[['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter',
                 'deposit_type', 'is_B', CFG.get('csv_obs_col', CFG['regressand']),
                 'entity_id', 'time_id']].copy()

    for label, res_dict in results.items():
        res = res_dict['res']
        coef_names = res_dict['regressors']

        # Determine which rows this regression applies to
        parts = label.split('_')
        k = int(parts[0][1])  # e.g., "k4" → 4
        is_pooled = 'pooled' in label

        if is_pooled:
            # Option C: pooled, use national demos for D-firms
            mask = df_out['deposit_type'] == k
            df_pred = _prepare_pooled_data(df, k)
        elif '_B' in label and 'opt' not in label:
            mask = (df_out['deposit_type'] == k) & df_out['is_B']
            df_pred = df[mask].copy()
        elif '_D_' in label:
            mask = (df_out['deposit_type'] == k) & (~df_out['is_B'])
            df_pred = df[mask].copy()
        else:
            continue

        # Build X matrix
        reg_cols = [c for c in coef_names if c != 'const']
        avail_cols = [c for c in reg_cols if c in df_pred.columns]

        if not avail_cols:
            continue

        # PREDICT ON COMPLETE CASES ONLY. This used to be `df_pred[avail_cols].fillna(0)`, which does
        # NOT impute a neutral value — it forces every missing regressor to zero, so a row missing all
        # of them collapses to the bare intercept. For k=4 that intercept is ~0.475 qoq-frac = ~190pp
        # annualized: an impossible deposit spread, silently written out as if it were a fitted policy.
        # It hit ~26% of rows before the 2016-2024 window and still ~3.6% after (the 2016Q1 rows, whose
        # _lag bank characteristics lag into the excluded 2015). Leave incomplete cases as NaN instead;
        # the consumer (estimation_bbl_2_fwd_sim::equilibrium_spreads) already falls back to the observed spread
        # for non-finite fits. Fabricating is never better than admitting the gap.
        # Apply the ESTIMATION centering (never recomputed here -- reusing the stored means is
        # what keeps the fitted values identical to the uncentered parametrization).
        df_pred = apply_centering(df_pred, avail_cols, res_dict.get('center_means', {}))

        complete = df_pred[avail_cols].notna().all(axis=1)
        X_pred = sm.add_constant(df_pred.loc[complete, avail_cols], has_constant='add')

        # Align columns with the estimated model
        for c in coef_names:
            if c not in X_pred.columns:
                X_pred[c] = 0.0
        X_pred = X_pred[coef_names]

        # Build a df_pred-length vector, NaN on the incomplete rows, and assign POSITIONALLY into
        # df_out[mask] — exactly the alignment the original used (the pooled branch's df_pred does not
        # necessarily share df_out's index, so an index-based write could silently misalign).
        fitted_full = pd.Series(np.nan, index=df_pred.index, dtype=float)
        if len(X_pred):
            fitted_full.loc[X_pred.index] = (X_pred @ res.params).astype(float)

        col_name = f'fitted_{label}'
        df_out.loc[mask, col_name] = fitted_full.values
        n_inc = int((~complete).sum())
        if n_inc:
            print(f"    [{label}] {n_inc:,} of {len(df_pred):,} rows left NaN "
                  f"(incomplete regressors — not fabricated to the intercept)")

    df_out['lhs_unit'] = CFG.get('fitted_unit', CFG['regressand'])
    return df_out


def fitted_csv_frame(df_fitted: pd.DataFrame) -> pd.DataFrame:
    """The fitted-policy CSV's columns in a FIXED order: the identifier and observed columns in the
    order compute_fitted_values builds them, then the fitted_* columns sorted by name, then
    `lhs_unit` last. The regressions finish in thread order (as_completed), so the order their
    columns were added in differs from run to run; fixing it here makes the file byte-reproducible,
    which the BBL sweep relies on (it compares the CSV's sha256 with the one the psi recorded).
    bbl_fwd_sim.jl matches the columns by name, so it reads every order alike."""
    fitted = sorted(c for c in df_fitted.columns if str(c).startswith('fitted_'))
    ids = [c for c in df_fitted.columns if not str(c).startswith('fitted_') and c != 'lhs_unit']
    tail = ['lhs_unit'] if 'lhs_unit' in df_fitted.columns else []
    return df_fitted[ids + fitted + tail]


def write_fitted_csv(df_fitted: pd.DataFrame, csv_path) -> None:
    """Write the fitted-policy CSV atomically and byte-reproducibly (%.6f, fitted_csv_frame's order).

    Through a temporary name in the same folder, renamed into place: the forward simulation's
    shards read this file at startup and the sweep hashes it, so a reader must see either the
    previous complete file or the new complete one, never a partial write. os.replace is atomic
    within one filesystem on POSIX and Windows alike. The temporary does not end in .csv, so no
    glob picks it up."""
    csv_path = pathlib.Path(csv_path)
    tmp_path = csv_path.with_name(f".tmp_{csv_path.name}.{os.getpid()}")
    try:
        fitted_csv_frame(df_fitted).to_csv(tmp_path, index=False, float_format='%.6f')
        os.replace(tmp_path, csv_path)
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def save_outputs(results: dict, df_fitted: pd.DataFrame) -> None:
    """Persist all estimation outputs to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Pickle with full statsmodels results --------------------------
    pkl_path = OUTPUT_DIR / f"{CFG['out_prefix']}_results.pkl"
    # Strip non-picklable items for safety; keep full result objects
    pkl_data = {}
    for label, res_dict in results.items():
        pkl_data[label] = {
            k: v for k, v in res_dict.items()
        }
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_data, f)
    print(f"  Saved pickle: {pkl_path.name}")

    # --- 2. Fitted values CSV ---------------------------------------------
    csv_path = OUTPUT_DIR / f"{CFG['out_prefix']}_fitted.csv"
    write_fitted_csv(df_fitted, csv_path)
    print(f"  Saved fitted values: {csv_path.name}")

    # --- 3. Summary JSON --------------------------------------------------
    summary = {}
    for label, res_dict in results.items():
        summary[label] = {
            'n_obs': res_dict['n_obs'],
            'n_clusters': res_dict['n_clusters'],
            'G_star': res_dict['G_star'],
            'r_squared': res_dict['r_squared'],
            'r_squared_adj': res_dict['r_squared_adj'],
            'coefficients': res_dict['coefficients'],
            'std_errors': res_dict['std_errors'],
            'pvalues': res_dict['pvalues'],
            # build_polfunc_table anchors the intercept's scale on this row, and it
            # blanks the row rather than raising when the key is absent -- so a summary
            # without it renders a table that looks complete and is missing a statistic.
            'mean_depvar': res_dict.get('mean_depvar'),
            'regressors': res_dict.get('regressors'),
            # Which rows report the quarter-clustered WCB (the dagger in the tables), and the
            # conglomerate / Driscoll-Kraay numbers kept beside it (see run_single_regression).
            'se_scheme': res_dict.get('se_scheme'),
            'std_errors_congl': res_dict.get('std_errors_congl'),
            'pvalues_congl': res_dict.get('pvalues_congl'),
            'std_errors_dk': res_dict.get('std_errors_dk'),
            'n_periods': res_dict.get('n_periods'),
            'n_nonpositive_balance': res_dict.get('n_nonpositive_balance'),
            # Largest |regressand| on the estimation rows (fraction), for the notes' "zero to within"
            # sentence where only the summary is read (the md twin).
            'depvar_absmax': _depvar_absmax(res_dict),
            'se_method': res_dict.get('se_method'),
        }

    json_path = OUTPUT_DIR / f"{CFG['out_prefix']}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved summary JSON: {json_path.name}")

    # --- 4. LaTeX tables per regression -----------------------------------
    for label, res_dict in results.items():
        tex_path = OUTPUT_DIR / f"{CFG['out_prefix']}_{label}.tex"
        try:
            with open(tex_path, 'w') as f:
                f.write(res_dict['res'].summary().as_latex())
        except Exception as exc:
            print(f"  [WARN] LaTeX export failed for {label}: {exc}")
    print(f"  Saved {len(results)} LaTeX tables")


def print_summary_table(results: dict) -> None:
    """Print a compact summary table of all regressions."""
    print("\n" + "=" * 80)
    print("  POLICY FUNCTION ESTIMATION SUMMARY")
    print("=" * 80)
    header = f"  {'Label':<40s} {'N':>8s} {'R2':>8s} {'G':>5s} {'G*':>7s}"
    print(header)
    print("  " + "-" * 70)
    for label, res_dict in sorted(results.items()):
        print(f"  {label:<40s} {res_dict['n_obs']:>8,d} "
              f"{res_dict['r_squared']:>8.4f} {res_dict['n_clusters']:>5d} "
              f"{res_dict['G_star']:>7.2f}")
    print("=" * 80 + "\n")


# ==============================================================================
# 7. LaTeX Export & PDF Compilation
# ==============================================================================
DRAFTS_DIR = _paths.drafts_dir()


def _polfunc_redirected() -> bool:
    """True when COST_POLFUNC_DIR points this step's outputs anywhere other than the default
    polfunc_dir(), ESTIMATION_OUTPUT/COST_POLFUNC: a sandbox, or the cluster's data/output/bbl
    step folder."""
    default = _paths.estimation_output() / "COST_POLFUNC"
    try:
        return pathlib.Path(OUTPUT_DIR).resolve() != default.resolve()
    except OSError:
        return str(OUTPUT_DIR) != str(default)


def _fragment_dir() -> pathlib.Path:
    """Where the paper fragments and the preview are written: drafts_dir() in production, and the
    step's own output folder (OUTPUT_DIR) whenever COST_POLFUNC_DIR redirects it. drafts_dir()
    does not follow COST_POLFUNC_DIR, so without this a sandboxed run rewrote the paper's C.10 and
    C.11 with its own numbers (2026-09-29: a B=19 verification run). The sleepiness exporters
    follow the same rule through rout_dir() under SLEEP_OUT_ROOT: a redirected run's exhibits go
    beside its own fits. On the cluster (COST_POLFUNC_DIR = data/output/bbl) the fragments now
    land in the archived step folder rather than in the unarchived Drafts skeleton."""
    return OUTPUT_DIR if _polfunc_redirected() else DRAFTS_DIR


# Human-readable labels for variable names in LaTeX
_VAR_LABELS = {
    'const': 'Constant',
    'log_total_assets_lag': 'Log Total Assets ($t-1$)',
    'equity_ratio_lag': 'Equity Ratio ($t-1$)',
    'has_ip': 'Group Contains IP',
    'asset_return_qoq_lag': 'Asset Return (QoQ, $t-1$)',
    'npl_provision_ratio_lag': 'NPL Provisions Ratio ($t-1$)',
    'credit_assets_lag': 'Credit / Assets ($t-1$)',
    'seg_S2': 'Segment S2',
    'seg_S3': 'Segment S3',
    'seg_S4': 'Segment S4',
    'seg_S5': 'Segment S5',
    'personnel_cost_ratio_lag': 'Personnel Cost Ratio ($t-1$)',
    'admin_cost_ratio_lag': 'Admin Cost Ratio ($t-1$)',
    'tax_cost_ratio_lag': 'Tax Cost Ratio ($t-1$)',
    'indice_basileia_lag': 'Basel Index ($t-1$)',
    'wholesale_ratio_lag': 'Wholesale Ratio ($t-1$)',
    'lci_lca_ratio_lag': 'LCI/LCA Ratio ($t-1$)',
    'risk_free_qoq': 'Risk-Free Rate (QoQ)',
    'log_total_assets_lag_sq': 'Log Total Assets$^2$',
    'equity_ratio_lag_sq': 'Equity Ratio$^2$',
    'risk_free_qoq_sq': 'Risk-Free Rate (QoQ)$^2$',
    'gdp_per_capita': r'GDP \textit{per capita}',
    'fraction_65plus': 'Fraction 65+',
    'fraction_young': 'Fraction Young',
    'pix_users_pf_per1000': 'Pix Users',
    # ANATEL "Acessos em Telefonia Movel": active mobile-telephony accesses (2G-5G) per 100
    # inhabitants (scrape_anatel_mobile.py), not broadband -- the label every other table uses.
    'connections_per100': 'Mobile Lines',
    'branches_per1000': 'Branches',
    'cadunico_families_per1000': r"Cad\'{U}nico Families",
    'gdp_per_capita_natl': r'GDP \textit{per capita} (Natl.)',
    'fraction_65plus_natl': 'Fraction 65+ (Natl.)',
    'fraction_young_natl': 'Fraction Young (Natl.)',
    'pix_users_pf_per1000_natl': 'Pix Users (Natl.)',
    'connections_per100_natl': 'Mobile Lines (Natl.)',
    'branches_per1000_natl': 'Branches (Natl.)',
    'cadunico_families_per1000_natl': r"Cad\'{U}nico Families (Natl.)",
}


def _stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.10:
        return '*'
    return ''


def _clean_var(v: str) -> str:
    """Map a variable name to its LaTeX-safe label."""
    return _VAR_LABELS.get(v, v.replace('_', '\\_'))


def _polfunc_col_keys(k: int) -> list:
    """The four regression labels (columns) for deposit type k."""
    kl = K_LABELS[k]
    # NOTE: the pooled (Option C) regression is still estimated and stored in the summary/pickle as a
    # robustness record, but it is NOT displayed: Step 2 consumes only B and D_optB, and pooling
    # few-but-huge B clusters with many-small D ones collapses the effective cluster count.
    return [f'k{k}_{kl}_B', f'k{k}_{kl}_D_optA', f'k{k}_{kl}_D_optB']


# Column headers, one per column, in the order of _polfunc_col_keys. Each of the three says what
# it is on its own: a shared 'D-type' spanner over the two D columns left the reader to work out
# from the Demographics indicator row which D specification was which.
_COL_HEADERS = ['B', 'D, no demographics', 'D, demographics at national means']

# The two lines that make every page break of the xltabular carry the continued head and foot:
# a zero kern closing \endlastfoot (so the notes' depth counts as height in longtable's own test)
# and, after the last body row, a reservation of the notes' height that the page builder must fit
# together with that row. make_bbl_cost_tables.py _wrap() documents the failure they close; the
# two tables share the house pattern and so share the fix.
_LASTFOOT_KERN = r'\noalign{\kern0pt}'
_LASTROW_RESERVE = (r'\noalign{\nobreak\dimen0=\dimexpr\ht\csname LT@lastfoot\endcsname'
                    r'+\dp\csname LT@lastfoot\endcsname-\ht\csname LT@foot\endcsname+2pt\relax'
                    r'\ifdim\dimen0<0pt \dimen0=0pt\fi\kern\dimen0\penalty9999\kern-\dimen0}')

# Demographic bases whose local and `_natl` variants are collapsed to ONE display row:
# each column shows its own geography (B/Pooled = local MCA, D natl. = national averages).
_DEMO_BASES = ['gdp_per_capita', 'fraction_65plus', 'fraction_young',
               'pix_users_pf_per1000', 'connections_per100', 'branches_per1000',
               'cadunico_families_per1000']

# The note sentence stating the estimation sample (run_single_regression).
_SAMPLE_NOTE = r'Estimated on firm--market--quarters with a positive balance of the type.'

# A column whose regressand is zero at the displayed precision: when more than _SMALL_SHARE of its
# displayed coefficient cells print below _SMALL in absolute value, the note says why (user,
# 2026-09-30: C.11's B column stays in basis points, explained rather than rescaled). The sentence
# says the regressand is near zero, so it is emitted only while |regressand| stays below
# _NEAR_ZERO display units (1 bp for the k=5 spread, 1 pp otherwise).
_SMALL = 0.010
_SMALL_SHARE = 0.5
_NEAR_ZERO = 1.0
_TYPE_NOUN = {4: 'time-deposit', 5: 'prepaid'}
_UNIT_ABBR = {'basis points': 'bp', 'percentage points': 'pp'}


def _depvar_absmax(res_dict):
    """Largest |regressand| on the estimation rows, in the regressand's own unit (a fraction): from
    the stored fit when present (an estimation run, --from-pkl), else the summary's
    `depvar_absmax`; None when neither is available."""
    res = res_dict.get('res')
    if res is not None:
        try:
            return float(np.max(np.abs(np.asarray(res.model.endog, float))))
        except Exception:
            pass
    v = res_dict.get('depvar_absmax')
    return None if v is None else float(v)


def _mostly_small(printed) -> bool:
    """True when more than _SMALL_SHARE of a column's printed coefficient cells are below _SMALL
    in absolute value (0.000 to 0.009 at three decimals)."""
    vals = []
    for s in printed:
        try:
            vals.append(abs(float(str(s).replace('{,}', '').replace(',', ''))))
        except ValueError:
            continue
    return bool(vals) and sum(v < _SMALL for v in vals) > _SMALL_SHARE * len(vals)


def _small_spread_sentence(res_dict, k, ftype, md=False, cfg=None) -> str:
    """'B firms' prepaid spread is zero to within 0.013 bp (mean -0.001 bp), so most of their
    coefficients print below 0.010.' from the fit: the bound is the largest |regressand| on the
    estimation rows rounded UP to three decimals, the mean is Mean dep. var.; without the bound
    the sentence states the mean alone. Units and the regressand's noun follow the variant (`cfg`,
    default CFG). '' when the regressand (the bound, else the mean) reaches _NEAR_ZERO display
    units, where 'near zero' would not be the reason."""
    c = CFG if cfg is None else cfg
    m = _lhs_display(k, c)
    unit = _UNIT_ABBR.get(_lhs_unit_name(k, c), _lhs_unit_name(k, c))
    mean, bound = res_dict.get('mean_depvar'), _depvar_absmax(res_dict)
    level = bound if bound is not None else mean
    if level is None or abs(level * m) >= _NEAR_ZERO:
        return ''

    def num(s):
        return s if md else f'${s}$'
    head = f"{ftype} firms' {_TYPE_NOUN.get(k, 'deposit')} {c.get('lhs_noun', 'spread')} "
    if bound is not None:
        up = float(np.ceil(bound * m * 1000.0 - 1e-9)) / 1000.0
        s = head + f"is zero to within {num(f'{up:.3f}')} {unit}"
        if mean is not None:
            s += f" (mean {num(_fmt3(mean * m))} {unit})"
    else:
        s = head + f"averages {num(_fmt3(mean * m))} {unit}"
    return s + f", so most of their coefficients print below {num(f'{_SMALL:.3f}')}."


# The dagger on a quarter-clustered SE cell, and the note sentence explaining it, worded as in the
# sleepiness comparison table (Table 4). The sentence is emitted only when a displayed cell carries
# the dagger.
_NATIONAL_MARK = r'^{\dagger}'
_NATIONAL_NOTE = (r'$\dagger$: quarter-clustered, national regressors '
                  r'(Section~\ref{sec:empirical:sleep}). ')

# The seed run_single_regression passes to both of its bootstraps (conglomerate, and quarter for
# the national rows). The results do not record it, so the notes restate it from here; the draw
# count and the weights are read from each result's se_method, which records them.
_POLFUNC_BOOT_SEED = 0
_WEIGHTS_NAME = {'webb': 'Webb weights', 'rademacher': 'Rademacher weights'}


def _boot_settings(results, k):
    """{(B, scheme)} of the wild bootstrap behind deposit type k's displayed columns, parsed from
    se_method ('wild cluster bootstrap (B=999, webb); ...'); None for a result that lacks it."""
    import re
    got = set()
    for ck in _polfunc_col_keys(k):
        if ck in (results or {}):
            m = re.search(r'B=(\d+),\s*(\w+)', str(results[ck].get('se_method') or ''))
            got.add((int(m.group(1)), m.group(2).lower()) if m else None)
    return got


def _boot_clause(results, k, md=False):
    """'$B=999$ draws, Webb weights, seed 0' when every displayed column records the same
    bootstrap, else '' (the note then keeps its generic wording)."""
    got = _boot_settings(results, k) if k is not None else set()
    if len(got) != 1 or None in got:
        return ''
    B, scheme = next(iter(got))
    w = _WEIGHTS_NAME.get(scheme, f'{scheme} weights')
    n = f'{B:,}'
    if md:
        return f'B = {n} draws, {w}, seed {_POLFUNC_BOOT_SEED}'
    return f'$B={n.replace(",", "{,}")}$ draws, {w}, seed {_POLFUNC_BOOT_SEED}'


def _zero_column(res_dict, v) -> bool:
    """True when regressor v's design column is zero on every row of the estimation sample, read
    off the stored fit (`res`, present after an estimation run and in the results pickle). Segment
    S5 among B firms is the case: no B firm is in it, so its coefficient is zero by construction
    (the minimum-norm solution leaves it at zero up to rounding, and no bootstrap draw can move
    it), and a t-statistic, hence stars, is meaningless. A result without `res` (the summary JSON)
    reports False, and its cells print as before."""
    res = res_dict.get('res')
    if res is None or v == 'const':
        return False
    try:
        names = list(res.params.index)
        if v not in names:
            return False
        return not np.any(np.asarray(res.model.exog)[:, names.index(v)])
    except Exception:
        return False


def _polfunc_cell(res_dict, v):
    """(coef, se, p, se_scheme) for variable v in a regression result dict, or None if absent.
    se_scheme is 'quarter' where the SE and p-value are the quarter-clustered WCB of a national
    regressor (NATIONAL_BY_NAME), 'zero' where the regressor is zero on every row (_zero_column),
    else 'congl'; a result stored without se_scheme reads 'congl'."""
    if v in res_dict['coefficients']:
        scheme = ('zero' if _zero_column(res_dict, v)
                  else (res_dict.get('se_scheme') or {}).get(v, 'congl'))
        return (res_dict['coefficients'][v], res_dict['std_errors'][v], res_dict['pvalues'][v],
                scheme)
    return None


def _polfunc_variant_cell(res_dict, base):
    """For a demographic base, return the cell for whichever variant (base or base_natl)
    this column actually uses."""
    for v in (base, base + '_natl'):
        c = _polfunc_cell(res_dict, v)
        if c is not None:
            return c
    return None


# ---- Number display (V_Main rule, 2026-09-28) -------------------------------------------------
# Every displayed number has exactly three decimals and counts are integers with thousands
# separators. A value that rounds to zero prints 0.000, never -0.000; a NONZERO value that prints
# 0.000 is kept as it is (the display unit is not changed for it) and listed in _ZERO_PRINTS,
# which write_polfunc_fragments reports on stdout.
_ND = 3
_ZERO_PRINTS: list = []
# Cells of all-zero regressors (_zero_column), printed 0.000 by construction rather than rounded;
# reported on stdout apart from _ZERO_PRINTS.
_ZERO_COLUMN_CELLS: list = []


def _fmt3(x, where: str = '') -> str:
    v = float(x)
    s = f'{v:,.{_ND}f}'
    if float(s.replace(',', '')) == 0.0:
        if v != 0.0 and where:
            _ZERO_PRINTS.append((where, v))
        s = f'{0.0:.{_ND}f}'
    return s


def _polfunc_emit_rows(label, cells, rows, mult=1.0, where='', var=None, zero_cells=None,
                       col_cells=None):
    """Append a coefficient line + a standard-error line for one regressor across columns.

    `mult` = LHS_display · unit_scale rescales coefficient AND standard error together (a pure
    change of units, so t-stats and stars are unaffected). A quarter-clustered SE (a national
    regressor) carries a dagger, as in the sleepiness tables. A regressor that is zero on every
    row of a column (scheme 'zero') prints 0.000 (0.000) without stars, and (column header, var)
    is appended to `zero_cells` for the note. `col_cells`, when a dict, collects each column's
    printed coefficients (index -> list of text) for _mostly_small."""
    coef_cells, se_cells = [], []
    for j, c in enumerate(cells):
        if c is None:
            coef_cells.append('')
            se_cells.append('')
        else:
            cf, se, p, scheme = c
            w = f'{where} {label} [{_COL_HEADERS[j]}]'
            if scheme == 'zero':
                coef_cells.append(f"${0.0:.{_ND}f}^{{}}$")
                se_cells.append(f"$({0.0:.{_ND}f})$")
                if zero_cells is not None:
                    zero_cells.append((_COL_HEADERS[j], var, w))
                if col_cells is not None:
                    col_cells.setdefault(j, []).append(f'{0.0:.{_ND}f}')
                continue
            mark = _NATIONAL_MARK if scheme == 'quarter' else ''
            txt = _fmt3(cf * mult, w)
            if col_cells is not None:
                col_cells.setdefault(j, []).append(txt)
            coef_cells.append(f"${txt.replace(',', '{,}')}^{{{_stars(p)}}}$")
            se_cells.append(f"$({_fmt3(se * mult, w + ' SE').replace(',', '{,}')}){mark}$")
    rows.append(f'{label} & ' + ' & '.join(coef_cells) + r' \\')
    rows.append(' & ' + ' & '.join(se_cells) + r' \\')


def _row_label(v):
    """Row label, WITHOUT a bracketed unit. The display SCALING still applies (see
    _display_unit / _DISPLAY_UNITS); only the '[unit]' suffix is omitted, because the units are
    stated in the table notes and in the surrounding prose instead of in column 1."""
    return _clean_var(v)


def _polfunc_panel_rows(results: dict, k: int, include_segments: bool = False,
                        zero_cells=None, col_cells=None) -> list:
    """LaTeX coefficient rows for one deposit type k across the four columns.

    Non-demographic regressors are the ordered union; demographics are merged (local + national
    into one row, each column showing its own variant). Segment dummies are EXCLUDED unless
    `include_segments` — in the main table they collapse to a 'Segment FE: Yes' indicator row.
    Every coefficient is shown in its display unit (see _DISPLAY_UNITS). `zero_cells`, when a
    list, receives the displayed cells of all-zero regressors (see _polfunc_emit_rows)."""
    cols = _polfunc_col_keys(k)
    demo_all = set(_DEMO_BASES) | {d + '_natl' for d in _DEMO_BASES}
    lhs = _lhs_display(k)

    # ordered union of NON-demographic regressors (first appearance across columns)
    order, seen = [], set()
    for ck in cols:
        if ck not in results:
            continue
        for v in results[ck]['regressors']:
            if v in demo_all or v in seen:
                continue
            if v in _SEGMENT_VARS and not include_segments:
                continue
            order.append(v)
            seen.add(v)

    rows = []
    where = f"C k={k}" + (" segment" if include_segments else "")
    for v in order:
        cells = [_polfunc_cell(results[ck], v) if ck in results else None for ck in cols]
        _polfunc_emit_rows(_row_label(v), cells, rows, mult=lhs * _display_unit(v)[0],
                           where=where, var=v, zero_cells=zero_cells, col_cells=col_cells)

    # Merged demographic rows -- DETAIL ONLY. In the main table they collapse to a
    # 'Demographics' indicator row (Yes/No/Yes: B carries local MCA demographics, D~(no demo.)
    # carries none, D~(natl. demo.) carries national ones), exactly like the segment dummies.
    demo_rows = []
    if not include_segments:
        return rows
    for base in _DEMO_BASES:
        cells = [_polfunc_variant_cell(results[ck], base) if ck in results else None for ck in cols]
        if all(c is None for c in cells):
            continue
        _polfunc_emit_rows(_row_label(base), cells, demo_rows, mult=lhs * _display_unit(base)[0],
                           where=where, var=base, zero_cells=zero_cells, col_cells=col_cells)
    if demo_rows:
        rows.append(r'\addlinespace[0.3ex]')
        rows.extend(demo_rows)
    return rows


def _polfunc_stat_row(label, values, fmt) -> str:
    cells = [fmt(v) if v is not None else '' for v in values]
    return f'{label} & ' + ' & '.join(cells) + r' \\'


def _polfunc_notes(k: int | None = None, results: dict | None = None,
                   dagger: bool = False, zero_cells=None, small=None) -> str:
    """Table Notes: the inference paragraph, the regressor units, then what the dependent variable is.

    Deliberately short. Everything the notes used to carry -- winsorization, centering, the
    segment/demographic indicators -- is documented in the prose of the paper and in this
    module\'s comments. What stays is what a reader needs at the table: how the standard errors
    were produced, the units the row labels do not carry, and the exact regressand. For k=5 a D
    column whose regressors explain almost nothing is said to be what it is, a constant near zero.
    `dagger` (some displayed SE cell is quarter-clustered) adds the sentence that explains the mark.

    Citations are real \parencite keys, not typeset-by-hand author strings, so they resolve
    against References.bib and stay correct if an entry changes. The paper uses biblatex/biber
    (authoryear-comp), where \parencite is the parenthetical form.
    """
    # Held to the appendix length of the V_Main style guide (2026-09-28): ~150 words at
    # \footnotesize. The regressor units are stated once here: the ratios, the Basel index included,
    # are fractions in the panel but their coefficients are per percentage point, and the two rates
    # (risk-free, asset return) are quarterly, as their labels say.
    boot = _boot_clause(results, k)
    note = (
        r'\textit{Notes:} Standard errors in parentheses: score/multiplier wild cluster bootstrap '
        r'by conglomerate, ' + (boot or 'Webb weights') + ' '
        r'\parencite{cameron2008bootstrap,mackinnon2017wild,webb2023reworking}; $G$ and '
        r'$G^{*}=G/(1+\mathrm{cv}^{2})$ \parencite{imbens2016robust,carter2017asymptotic} '
        r'measure cluster paucity and are not the inference. '
        + (_NATIONAL_NOTE if dagger else '')
        + r'*** $p<0.01$, ** $p<0.05$, '
        r'* $p<0.1$. Ratios and rates, the Basel index included, are fractions in the panel; '
        r'their coefficients are per pp (per pp$^{2}$ for squares), and the '
        r'risk-free rate and asset return are quarterly. '
        # str.replace, not str.format: the note is LaTeX and full of braces.
        + CFG['depvar_note'].replace('{unit}', _unit_clause(k if k is not None else 4))
        # The estimation sample (run_single_regression).
        + ' ' + _SAMPLE_NOTE
    )
    if k == 5 and results:
        d_cols = [ck for ck in _polfunc_col_keys(k)[1:] if ck in results]
        r2 = [float(results[ck]['r_squared']) for ck in d_cols]
        if r2 and max(r2) < 0.05:
            note += (r' In both D columns the fitted policy is essentially a constant near zero '
                     r'($R^{2}$ of ' + ' and '.join(f'${v:.3f}$' for v in r2) + ').')
    # A column whose coefficients print mostly below 0.010 (build_polfunc_table, _mostly_small).
    for ftype, res_dict in (small or []):
        s = _small_spread_sentence(res_dict, k if k is not None else 4, ftype)
        if s:
            note += ' ' + s
    # One clause per displayed all-zero regressor (_zero_column), printed 0.000 without stars.
    for head, var in dict.fromkeys((h, v) for h, v, _w in (zero_cells or [])):
        if var in _SEGMENT_VARS:
            note += (f' No {head} firm is in segment {var.split("_", 1)[1]}; its coefficient is '
                     f'zero by construction.')
        else:
            note += (f' {_clean_var(var)} is zero for every {head} firm; its coefficient is zero '
                     f'by construction.')
    return note


_K_TITLE = {4: r'Time Deposits / CDB ($k=4$)', 5: r'Prepaid Accounts ($k=5$)'}

# Segment dummies: absorbed out of the main table into a "Segment FE: Yes" indicator row, and
# shown explicitly only in the `_segment` companion table.
_SEGMENT_VARS = ['seg_S2', 'seg_S3', 'seg_S4', 'seg_S5']

# ---- Display units -------------------------------------------------------------------------
# Coefficients are reported per the unit in brackets in the row label. Units are pinned to the
# REST OF THE PAPER, not chosen freely: any regressor that also appears in the sleepiness/demand
# tables is displayed in the unit it carries there, so a coefficient here is comparable to one
# there. That pinning is now ENFORCED rather than documented: the unit table lives once, in
# utils/state_transform.DISPLAY, and the sleepiness exporters, the descriptive tables and this
# module all read the same object. Shares and rates are in percentage points throughout
# (fraction_65plus / fraction_young moved from "fraction" to "pp" when the sleepiness tables
# did, which is the whole reason the two are still comparable).
# Re-exported for `make_polfunc_md_table.py`, which imports this name and indexes it as a
# dict (`_DISPLAY_UNITS.get(base, (1.0, ""))`). It IS the shared registry table, not a copy,
# so there is still exactly one source of truth for units across the paper.
from utils.state_transform import DISPLAY as _DISPLAY_UNITS  # noqa: E402,F401


def _display_unit(v):
    """(scale, unit label) for a regressor; `_natl` variants inherit the base unit.

    The table now lives in utils/state_transform.DISPLAY, shared with the sleepiness
    exporters and the descriptive tables. The comment above used to ask a reader to keep
    this dict in sync with sleep_export_link.py by hand; the units are now a single
    object, so cross-table comparability is structural rather than aspirational.

    NOTE the polfunc builds its regressors in RAW panel units (it does not go through
    state_transform.apply_scale), which is why `scaled=False` is the right convention
    here and the returned scale is the display unit measured in raw units."""
    from utils.state_transform import display_unit
    return display_unit(v)


def build_polfunc_table(results: dict, k: int, include_segments: bool = False) -> str:
    """Paper-style, page-breaking `xltabular` fragment for ONE deposit type k
    (four columns: B / D no-demo / D natl.-demo / Pooled). Bare fragment — `\\input`-able
    into V_Main.tex (the paper preamble already provides xltabular/booktabs/setspace).

    `include_segments=False` (the main table) hides the four segment dummies and reports them as a
    'Segment FE: Yes' indicator row; `True` builds the `_segment` companion that shows them."""
    colspec = r'>{\raggedright\arraybackslash}p{5.4cm} *{3}{>{\centering\arraybackslash}X}'
    # One label per column; the centered X columns wrap the longer two onto a second line.
    head = ' & ' + ' & '.join(_COL_HEADERS) + r' \\'
    caption = 'Policy Function Estimates: ' + _K_TITLE[k] + CFG.get('caption_suffix', '')
    label = CFG['tab_label'] + f'_k{k}' + ('_segment' if include_segments else '')
    cols = _polfunc_col_keys(k)
    # The body first: the notes explain the dagger, and an all-zero regressor, only if a
    # displayed cell carries one.
    zero_cells, col_cells = [], {}
    body = _polfunc_panel_rows(results, k, include_segments=include_segments,
                               zero_cells=zero_cells, col_cells=col_cells)
    # Firm types whose column prints mostly below 0.010 (_mostly_small): B from the first column,
    # D from the first D column that does.
    small = []
    for ftype, js in (('B', (0,)), ('D', (1, 2))):
        hit = next((j for j in js if j < len(cols) and cols[j] in results
                    and _mostly_small(col_cells.get(j, []))), None)
        if hit is not None:
            small.append((ftype, results[cols[hit]]))
    dagger = any(_NATIONAL_MARK in r for r in body)
    _ZERO_COLUMN_CELLS.extend(w for _h, _v, w in zero_cells)

    L = [
        r'\begin{spacing}{1.0}',
        r'\footnotesize',
        r'\begin{xltabular}{\textwidth}{' + colspec + '}',
        r'\caption{' + caption + '}',
        r'\label{' + label + r'} \\',
        r'\toprule',
        head,
        r'\midrule',
        r'\endfirsthead',
        r'\multicolumn{4}{c}{\bfseries Table \thetable\ (continued)} \\',
        r'\toprule',
        head,
        r'\midrule',
        r'\endhead',
        r'\midrule',
        r'\multicolumn{4}{r}{\textit{Continued on next page}} \\',
        r'\endfoot',
        r'\bottomrule',
        r'\multicolumn{4}{@{}p{\dimexpr\textwidth-2\tabcolsep\relax}@{}}{\footnotesize '
        + _polfunc_notes(k, results, dagger=dagger, zero_cells=zero_cells, small=small)
        + r'} \\',
        _LASTFOOT_KERN,
        r'\endlastfoot',
    ]

    L.extend(body)
    L.append(r'\midrule')
    r2 = [results[ck]['r_squared'] if ck in results else None for ck in cols]
    obs = [results[ck]['n_obs'] if ck in results else None for ck in cols]
    G = [results[ck]['n_clusters'] if ck in results else None for ck in cols]
    Gs = [results[ck]['G_star'] if ck in results else None for ck in cols]
    if not include_segments:
        # Segment dummies are in every regression; report them as an FE indicator.
        present = ['Yes' if (ck in results and any(v in results[ck]['coefficients']
                                                   for v in _SEGMENT_VARS)) else 'No'
                   for ck in cols]
        L.append('Segment FE & ' + ' & '.join(present) + r' \\')
        # Demographics: Yes where the column actually carries demographic regressors. This reads
        # Yes/No/Yes -- B has local MCA demographics, D (no demo.) has none by construction, and
        # D (natl. demo.) has the population-weighted national ones.
        _demo_all = set(_DEMO_BASES) | {d + '_natl' for d in _DEMO_BASES}
        demo_present = ['Yes' if (ck in results and any(v in results[ck]['coefficients']
                                                        for v in _demo_all)) else 'No'
                        for ck in cols]
        L.append('Demographics & ' + ' & '.join(demo_present) + r' \\')
    # Scale anchor for the intercept: the constant is the prediction at x=0 (assets of R$1), which
    # is out of support and therefore large; mean(fitted) == mean(y) under OLS, so this row IS the
    # fitted value at the average state and is what the reader should read the levels against.
    _lhs = _lhs_display(k)
    mdv = [results[ck]['mean_depvar'] * _lhs if ck in results and 'mean_depvar' in results[ck]
           else None for ck in cols]
    where = f"C k={k}" + (" segment" if include_segments else "")
    L.append(_polfunc_stat_row('Mean dep.\\ var.', mdv,
                               lambda x: _fmt3(x, f'{where} Mean dep. var.')))
    L.append(_polfunc_stat_row(r'$R^{2}$', r2, lambda x: _fmt3(x, f'{where} R2')))
    L.append(_polfunc_stat_row('Observations', obs, lambda x: f'{int(x):,}'))
    L.append(_polfunc_stat_row(r'Clusters ($G$)', G, lambda x: f'{int(x):,}'))
    L.append(_polfunc_stat_row(r'Eff.\ clusters ($G^{*}$)', Gs, lambda x: _fmt3(x)))

    L.append(_LASTROW_RESERVE)
    L.append(r'\end{xltabular}')
    L.append(r'\end{spacing}')
    return '\n'.join(L)


def write_polfunc_fragments(results: dict) -> dict:
    """Write the `\\input`-able fragments: one MAIN table per deposit type (segment dummies
    collapsed to an FE indicator) plus a `_segment` companion that shows them explicitly. They go
    to _fragment_dir(): Drafts in production, the redirected step folder otherwise."""
    out_dir = _fragment_dir()
    if out_dir != DRAFTS_DIR:
        print(f"  [sandbox] COST_POLFUNC_DIR redirects this run: fragments and preview go to "
              f"{out_dir}, not to Drafts.")
    out_dir.mkdir(parents=True, exist_ok=True)
    _ZERO_PRINTS.clear()
    _ZERO_COLUMN_CELLS.clear()
    frags = {}
    for k in K_ENDOG:
        for seg in (False, True):
            frag = build_polfunc_table(results, k, include_segments=seg)
            suffix = '_segment' if seg else ''
            path = out_dir / f"{CFG['out_prefix']}_k{k}{suffix}.tex"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(frag + '\n')
            print(f"  Wrote fragment: {path.name}  (label {CFG['tab_label']}_k{k}{suffix})")
            if not seg:
                frags[k] = frag          # preview shows the MAIN tables
    if _ZERO_PRINTS:
        print(f"  [display] {len(_ZERO_PRINTS)} nonzero value(s) print as 0.000 at {_ND} decimals "
              f"(not rescaled):")
        for where, v in _ZERO_PRINTS:
            print(f"      {where}: {v:.3e}")
    if _ZERO_COLUMN_CELLS:
        print(f"  [display] {len(_ZERO_COLUMN_CELLS)} cell(s) of an all-zero regressor print "
              f"0.000 (0.000) without stars (zero by construction; the note says so):")
        for where in _ZERO_COLUMN_CELLS:
            print(f"      {where}")
    return frags


def compile_polfunc_preview(frags: dict) -> None:
    """Compile ONE standalone preview PDF holding all deposit-type fragments (one per page),
    rendered with the paper's own preamble (utils.tex_preamble.wrap_table), in _fragment_dir()
    beside the fragments."""
    import subprocess
    try:
        from utils.tex_preamble import wrap_table
    except Exception as exc:
        print(f"  [WARN] utils.tex_preamble not importable ({exc}); fragments saved, preview skipped.")
        return
    out_dir = _fragment_dir()
    body = '\n\n\\clearpage\n\n'.join(frags[k] for k in sorted(frags))
    name = f"{CFG['out_prefix']}_preview"
    tex_path = out_dir / f"{name}.tex"
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(wrap_table(body))
    print(f"  Compiling preview PDF ({name}.pdf)...")
    try:
        # pdflatex -> biber -> pdflatex x2. The biber pass is what resolves the \parencite keys in
        # the Notes against References.bib (the preamble already \addbibresource's it). Without it
        # the preview renders the citations as unresolved markers, which reads like a broken table
        # even though the fragment is fine in V_Main.tex, where biber does run.
        subprocess.run(['pdflatex', '-interaction=nonstopmode', f'{name}.tex'],
                       cwd=str(out_dir), capture_output=True, text=True)
        try:
            subprocess.run(['biber', name], cwd=str(out_dir), capture_output=True, text=True)
        except FileNotFoundError:
            print("  [WARN] biber not found; preview citations will render unresolved "
                  "(fragments are unaffected).")
        for _ in range(2):
            subprocess.run(['pdflatex', '-interaction=nonstopmode', f'{name}.tex'],
                           cwd=str(out_dir), capture_output=True, text=True)
        pdf_path = out_dir / f"{name}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            print(f"  Preview PDF generated: {pdf_path.name}")
        else:
            print(f"  [WARN] preview PDF may not have compiled — check {name}.log")
    except FileNotFoundError:
        print("  [WARN] pdflatex not found; fragment saved, preview skipped.")
    except Exception as exc:
        print(f"  [WARN] preview compilation error: {exc}")


def _resolve_results_pkl(spec: str):
    """-> the pickle to render from. A bare --from-pkl prefers polfunc_dir(), which
    utils.paths declares authoritative, then the BBL cluster_processed tree that older
    ingests wrote into."""
    name = f"{CFG['out_prefix']}_results.pkl"
    if spec != 'auto':
        p = pathlib.Path(spec)
        if not p.exists():
            sys.exit(f"[FATAL] --from-pkl path does not exist: {p}")
        return p
    cands = [OUTPUT_DIR / name,
             _paths.bbl_output_dir() / 'cluster_processed' / name]
    for c in cands:
        if c.exists():
            return c
    sys.exit("[FATAL] no {} found in: {}".format(name, "  |  ".join(str(c) for c in cands)))


def render_fragments_from_pkl(spec: str) -> None:
    """Rebuild the paper fragments from a stored fit, estimating nothing.

    build_polfunc_table reads plain values only -- regressors, coefficients, std_errors,
    pvalues, se_scheme, r_squared, n_obs, n_clusters, G_star, mean_depvar -- so refreshing the tables
    after a new fit lands does not require re-running the regressions. The identifying
    statistics are printed so the vintage being rendered is visible in the log rather than
    inferred from a file date.
    """
    p = _resolve_results_pkl(spec)
    st = p.stat()
    print(f"  source : {p}")
    print(f"  written: {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}"
          f"  ({st.st_size:,} B)")
    with open(p, 'rb') as f:
        results = pickle.load(f)
    if not results:
        sys.exit("[FATAL] the pickle holds no results.")

    print("")
    print(f"  {'variant':26s} {'N':>10s} {'G':>6s} {'R2':>9s}  mean_depvar")
    missing = []
    for label in sorted(results):
        r = results[label]
        if 'mean_depvar' not in r:
            missing.append(label)
        mdv = r.get('mean_depvar')
        shown = '--' if mdv is None else format(mdv, '.6g')
        print(f"  {label:26s} {r['n_obs']:>10,} {r['n_clusters']:>6} "
              f"{r['r_squared']:>9.5f}  {shown}")
    if missing:
        print(f"  [WARN] {len(missing)} variant(s) carry no mean_depvar; that row will be "
              f"blank: {', '.join(missing)}")

    print("")
    print("  --- Paper table fragments + preview ---")
    frags = write_polfunc_fragments(results)
    compile_polfunc_preview(frags)


# ==============================================================================
# 8. Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Policy Function Estimation for Deposit Types k=4,5 (BBL Step 1)")
    parser.add_argument(
        '--depvar', choices=['spread', 'rate'], default='spread',
        help='Regressand: "spread" (default; compounded annual deposit spread, fed to BBL Step 2) or '
             '"rate" (annualized deposit rate = (1+rate_qoq)^4-1). "rate" writes a SEPARATE '
             'polfunc_rate_* namespace and does NOT overwrite the spread policy.')
    parser.add_argument(
        '--from-pkl', nargs='?', const='auto', default=None, metavar='PATH',
        help='Render the paper table fragments from an EXISTING results pickle '
             'instead of re-fitting. A bare --from-pkl resolves it from '
             'polfunc_dir(), then the BBL cluster_processed tree. Only the .tex '
             'fragments and the preview are written; the fitted CSV and summary '
             'JSON are left alone.')
    args = parser.parse_args()

    # Regressand selection: switch CFG to the rate namespace if requested.
    if args.depvar == 'rate':
        CFG.update(_CFG_RATE)

    if args.from_pkl is not None:
        print("=" * 70)
        print("  Policy Function table fragments -- rendered from a stored fit")
        print(f"  Namespace: {CFG['out_prefix']}_*   (no estimation is run)")
        print("=" * 70)
        render_fragments_from_pkl(args.from_pkl)
        return

    print("=" * 70)
    print("  Policy Function Estimation (BBL Step 1)")
    print(f"  Regressand: {CFG['regressand']}  (--depvar {args.depvar}) -> {CFG['out_prefix']}_*")
    print(f"  Deposit types: {K_ENDOG}")
    print(f"  D-firm demographic options: {D_DEMO_OPTIONS}")
    print("=" * 70)

    # ---- Load data once --------------------------------------------------
    t_start = time.perf_counter()
    df = load_and_prepare_panel()
    df = compute_national_demographics(df)

    # ---- Run estimation --------------------------------------------------
    results = estimate_all_policy_functions(df)

    if not results:
        print("[FATAL] No regressions succeeded. Exiting.")
        sys.exit(1)

    # ---- Print summary ---------------------------------------------------
    print_summary_table(results)

    # ---- Compute fitted values -------------------------------------------
    print("Computing fitted values...")
    df_fitted = compute_fitted_values(df, results)

    # ---- Save outputs -----------------------------------------------------
    # The policy function is SPEC-INVARIANT (it never touches a sleepiness spec), so outputs
    # carry no spec label. The old `_spec_{id}` replication was pipeline cosmetics only.
    print("\n  --- Saving outputs ---")
    save_outputs(results, df_fitted)

    # ---- Paper-style table fragments (one per deposit type) + preview PDF -----
    print("\n  --- Paper table fragments + preview ---")
    frags = write_polfunc_fragments(results)
    compile_polfunc_preview(frags)

    elapsed = time.perf_counter() - t_start
    print(f"\n[DONE] Policy function estimation complete in {elapsed:.1f}s")


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    main()

