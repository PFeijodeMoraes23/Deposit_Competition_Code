"""
estimation_1_demand_2_loop.py
==============================
BLP outer-inner demand estimation loop (Appendix-BLP, V_Main.tex).

Takes per-spec demand prep CSVs from estimation_1_demand_1_prep.py and
the market panel, then recovers theta_2* = (Pi*, Sigma*) via GMM (Eq-A1)
and theta_1* via linear IV (Eq-A5) for each sleepiness specification.

Usage
-----
  python estimation_1_demand_2_loop.py --spec 1 --stage logit
  python estimation_1_demand_2_loop.py --spec all --stage full --R 500

CLI Flags
---------
  --spec {1..12|all}        Single spec for testing; default=12
  --stage {logit|sigma|full|extended}  Sequential CG2020 build-up
  --R {int}                 Simulation draws (default=500)
  --seed {int}              RNG seed (default=42)
  --tol-inner {float}       Inner contraction tolerance (default=1e-14)
  --max-inner {int}         Max inner iterations (default=2000)
  --tol-outer {float}       Outer GMM tolerance (default=1e-6)
  --method {l-bfgs-b|nelder-mead}  Outer minimiser (default=l-bfgs-b)

References
----------
  Berry, Levinsohn & Pakes (1995, Econometrica)
  Conlon & Gortmaker (2020, RAND J. Econ.)
  Egan, Hortacsu & Matvos (2017, AER)
  Nevo (2001, Econometrica)
"""

import os
import sys
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

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, approx_fprime
from scipy.stats.qmc import Halton
import statsmodels.api as sm

# ==============================================================================
# 0. Paths & Constants
# ==============================================================================
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
PANEL_CSV = DATA_DIR / "market_panel.csv"
DEMAND_PREP_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP"
BLP_OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "BLP_RESULTS"

# Product characteristics (non-price) entering the utility function
X_COLS = ['fgc_covered', 'has_ip', 'seg_S2', 'seg_S3', 'seg_S4', 'seg_S5',
          'log_total_assets_lag', 'equity_ratio_lag']
L_PROD = len(X_COLS)  # 8
K_TYPES = 4           # deposit types 1, 2, 4, 5 (k=3 dropped)
K_LIST = [1, 2, 4, 5]

# Demographics for Pi interactions
D_COLS = ['gdp_per_capita', 'fraction_65plus', 'fraction_young',
          'pix_users_pf_per1000', 'connections_per100', 'frac_4g5g',
          'branches_per1000', 'cadunico_families_per1000']
D_DIM = len(D_COLS)   # 8

# BLP LOO instruments (demand-side)
IV_BLP_LOO = ['loo_log_assets', 'mean_loo_log_assets',
              'loo_equity_assets', 'mean_loo_equity_ratio',
              'loo_indice_basileia', 'mean_loo_basileia',
              'loo_credit_assets', 'mean_loo_credit_assets',
              'loo_npl_provision', 'mean_loo_npl_provision',
              'n_rivals']

# Cost-shifter IVs (for endogenous types k=4,5)
IV_COST = ['personnel_cost_ratio_lag', 'admin_cost_ratio_lag',
           'tax_cost_ratio_lag']

# Capital adequacy IVs (robustness)
IV_CAPITAL = ['indice_basileia_lag']

# Columns to load from market_panel.csv
MP_KEEP_COLS = (
    ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter', 'CODMUN_IBGE']
    + [f'spread_a{k}' for k in K_LIST]
    + [f'dep_a{k}' for k in K_LIST]
    + X_COLS + D_COLS + ['pop_total']
    + IV_BLP_LOO + IV_COST + IV_CAPITAL
    + ['segment']
)


# ==============================================================================
# 1. Data Loading
# ==============================================================================
def load_panel_selective(panel_csv: Path) -> pd.DataFrame:
    """Load market_panel.csv with only needed columns."""
    print(f"Loading selective columns from {panel_csv}...")
    available = pd.read_csv(panel_csv, nrows=0).columns.tolist()
    cols_to_load = [c for c in MP_KEEP_COLS if c in available]
    missing = [c for c in MP_KEEP_COLS if c not in available]
    if missing:
        print(f"  Warning: {len(missing)} requested columns not in panel: {missing}")

    df = pd.read_csv(panel_csv, usecols=cols_to_load,
                     dtype={'mca_code': str, 'CODMUN_IBGE': str})
    df['is_B'] = (df['CODMUN_IBGE'] != '0')

    # Pivot wide deposits/spreads to long
    id_vars = ['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter']
    df = df.drop_duplicates(subset=id_vars)

    # Build entity_id and time_id (same logic as prep script)
    records = []
    for k in K_LIST:
        dep_col = f'dep_a{k}'
        spr_col = f'spread_a{k}'
        if dep_col not in df.columns:
            continue
        chunk = df.copy()
        chunk['deposit_type'] = k
        chunk['deposit_balance'] = chunk[dep_col]
        chunk['spread_qoq'] = chunk[spr_col] if spr_col in chunk.columns else np.nan
        records.append(chunk)

    df_long = pd.concat(records, ignore_index=True)
    # Drop the wide deposit/spread columns
    drop_cols = [c for c in df_long.columns if c.startswith('dep_a') or c.startswith('spread_a')]
    df_long = df_long.drop(columns=drop_cols, errors='ignore')

    df_long['entity_id'] = (df_long['CodConglomeradoPrudencial'].astype(str) + "_"
                            + df_long['deposit_type'].astype(str) + "_"
                            + df_long['mca_code'].astype(str))
    df_long['time_id'] = df_long['year'].astype(str) + "Q" + df_long['quarter'].astype(str)

    print(f"  Panel loaded: {len(df_long)} rows, {df_long['entity_id'].nunique()} entities")
    return df_long


def load_demand_prep(spec_id: int) -> pd.DataFrame:
    """Load per-spec demand prep CSV."""
    csv_path = DEMAND_PREP_DIR / f"demand_prep_spec_{spec_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    return pd.read_csv(csv_path, dtype={'mca_code': str})


def merge_panel_with_prep(df_panel: pd.DataFrame,
                          df_prep: pd.DataFrame) -> pd.DataFrame:
    """Merge panel variables with demand prep data on (entity_id, time_id)."""
    prep_cols = ['entity_id', 'time_id', 'Dep_Act', 'share_NB', 'share_B_cond',
                 'phi_mt', 'phi_t', 'is_B', 'local_B_active', 'nat_active']
    df_prep_slim = df_prep[prep_cols].copy()

    # Panel has product chars, demographics, IVs
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


# ==============================================================================
# 2. Simulation Draws
# ==============================================================================
def generate_halton_draws(R: int, dim: int, seed: int) -> np.ndarray:
    """Generate R Halton quasi-random draws of dimension `dim`, transformed
    to standard normal via inverse CDF (Nevo 2001)."""
    sampler = Halton(d=dim, scramble=True, seed=seed)
    u = sampler.random(n=R)
    # Clip to avoid infinite tails
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return stats.norm.ppf(u)  # (R, dim)

def generate_demographic_draws(df: pd.DataFrame, R: int,
                               seed: int) -> dict:
    """For each (mca_code, time_id), draw R demographic vectors from
    Normal(mu_m, sigma_m^2) parametrised by MCA aggregates (Nevo 2001).

    Returns dict: (mca_code, time_id) -> (R, D) array.
    """
    rng = np.random.default_rng(seed)
    d_cols_avail = [c for c in D_COLS if c in df.columns]
    D = len(d_cols_avail)

    # Get MCA-level means (unique per mca x time)
    mca_level = df[['mca_code', 'time_id'] + d_cols_avail].drop_duplicates(
        subset=['mca_code', 'time_id'])

    # National std for each demographic (used as sigma)
    nat_std = mca_level[d_cols_avail].std().values  # (D,)
    nat_std = np.where(nat_std == 0, 1.0, nat_std)

    draws = {}
    for _, row in mca_level.iterrows():
        key = (row['mca_code'], row['time_id'])
        mu = row[d_cols_avail].values.astype(float)
        mu = np.nan_to_num(mu, nan=0.0)
        d_draw = rng.normal(loc=mu, scale=nat_std * 0.1, size=(R, D))
        draws[key] = d_draw

    return draws

# ==============================================================================
# 3. Theta2 Structure
# ==============================================================================
def build_theta2_structure(stage: str):
    """Define which parameters are active for each stage.

    Returns (sigma_indices, pi_interactions, total_params).
    - sigma_indices: list of indices into the (L+5) coefficient vector that
      get a sigma parameter.
    - pi_interactions: list of (coef_idx, demo_idx) pairs for Pi entries.
    """
    # Coefficient vector: [spread_1, ..., spread_5, x_1, ..., x_L]
    # spread indices: 0..K_TYPES-1, x indices: K_TYPES..K_TYPES+L-1

    if stage == 'logit':
        return [], [], 0

    elif stage == 'sigma':
        # sigma on spread only (one param shared across all 5 spread types)
        return [0], [], 1

    elif stage == 'full':
        # Sigma: spread (idx 0), log_total_assets (idx K_TYPES + idx in coef vector)
        sigma_idx = [0, K_TYPES + X_COLS.index('log_total_assets_lag')]
        # Pi: spread(0) x gdp(0), spread(0) x age(1), spread(0) x connections(4)
        pi_inter = [(0, D_COLS.index('gdp_per_capita')),
                     (0, D_COLS.index('fraction_65plus')),
                     (0, D_COLS.index('connections_per100'))]
        return sigma_idx, pi_inter, len(sigma_idx) + len(pi_inter)

    elif stage == 'extended':
        # Stage 3 params + product x demographic interactions
        sigma_idx = [0, K_TYPES + X_COLS.index('log_total_assets_lag')]
        pi_inter = [
            # Spread interactions (from stage 3)
            (0, D_COLS.index('gdp_per_capita')),
            (0, D_COLS.index('fraction_65plus')),
            (0, D_COLS.index('connections_per100')),
            # Product x demographic
            (K_TYPES + X_COLS.index('log_total_assets_lag'), D_COLS.index('gdp_per_capita')),
            (K_TYPES + X_COLS.index('fgc_covered'), D_COLS.index('fraction_65plus')),
            (K_TYPES + X_COLS.index('equity_ratio_lag'), D_COLS.index('cadunico_families_per1000')),
        ]
        return sigma_idx, pi_inter, len(sigma_idx) + len(pi_inter)

    else:
        raise ValueError(f"Unknown stage: {stage}")


def unpack_theta2(theta2_vec: np.ndarray, sigma_indices: list,
                  pi_interactions: list):
    """Unpack flat theta2 vector into Sigma diagonal and Pi entries."""
    n_sigma = len(sigma_indices)
    n_pi = len(pi_interactions)
    sigma_vals = theta2_vec[:n_sigma]
    pi_vals = theta2_vec[n_sigma:n_sigma + n_pi]
    return sigma_vals, pi_vals


# ==============================================================================
# 4. Compute mu_rjkmt (Eq-A2)
# ==============================================================================
def compute_mu(prod_vec: np.ndarray, nu_draws: np.ndarray,
               aligned_demo_draws: np.ndarray, sigma_vals: np.ndarray,
               sigma_indices: list, pi_vals: np.ndarray,
               pi_interactions: list, R: int, coef_dim: int) -> np.ndarray:
    """Compute mu_rjkmt efficiently using vectorized array ops."""
    
    # 1. Sigma term:
    sigma_diag = np.zeros(coef_dim)
    for idx_pos, coef_idx in enumerate(sigma_indices):
        if coef_idx < coef_dim:
            sigma_diag[coef_idx] = sigma_vals[idx_pos]
            
    sigma_nu = sigma_diag * nu_draws[:, :coef_dim]  # (R, coef_dim)
    mu = np.dot(prod_vec, sigma_nu.T)  # (N, R)
    
    # 2. Pi term:
    if len(pi_interactions) > 0:
        D = aligned_demo_draws.shape[2]
        for pi_idx, (coef_idx, demo_idx) in enumerate(pi_interactions):
            if coef_idx < coef_dim and demo_idx < D:
                p_v = prod_vec[:, coef_idx]  # (N,)
                d_v = aligned_demo_draws[:, :, demo_idx]  # (N, R)
                mu += pi_vals[pi_idx] * p_v[:, np.newaxis] * d_v
                
    return mu

# ==============================================================================
# 5. Inner Loop: BLP Contraction (Eq-A4)
# ==============================================================================
def compute_model_shares(delta: np.ndarray, mu: np.ndarray,
                         df: pd.DataFrame, R: int) -> tuple:
    """Compute model-implied shares s^B and s^NB from (delta, mu).

    Returns (s_model, omega_mt).
    - s_model: (N,) array of model-implied shares for each obs.
    - omega_mt: (N,) array of Omega_mt values for B-type obs.
    """
    N = len(df)
    is_B = df['is_B'].values
    mca_codes = df['mca_code'].values
    time_ids = df['time_id'].values
    deposit_types = df['deposit_type'].values
    pop_total = df['pop_total'].fillna(0).values

    # For each r, compute q_rjkmt = softmax(delta + mu_r) within each market
    # Build market group indices
    if is_B.any():
        # B-type: market = (mca_code, time_id)
        b_market_key = np.array([f"{m}_{t}" for m, t in zip(mca_codes, time_ids)])
    else:
        b_market_key = np.array([''] * N)

    # NB-type: market = time_id only
    nb_market_key = time_ids.copy()

    # Combined market key for softmax computation
    # All products in the same market compete
    market_key = np.where(is_B, b_market_key, nb_market_key)

    # Identify unique markets and build group indices
    unique_markets = np.unique(market_key)
    market_to_idx = {m: i for i, m in enumerate(unique_markets)}
    market_indices = np.array([market_to_idx[m] for m in market_key])

    # Compute average choice probability across draws
    q_avg = np.zeros(N)
    for r in range(R):
        v = delta + mu[:, r]  # (N,)

        # Log-sum-exp within each market
        max_v = np.full(len(unique_markets), -np.inf)
        np.maximum.at(max_v, market_indices, v)
        v_shifted = v - max_v[market_indices]

        exp_v = np.exp(v_shifted)
        sum_exp = np.zeros(len(unique_markets))
        np.add.at(sum_exp, market_indices, exp_v)

        q_r = exp_v / sum_exp[market_indices]
        q_avg += q_r

    q_avg /= R  # (N,) model-implied conditional shares

    # s^B_model = q_avg for B-type (Eq-A3-B is just 1/R sum)
    # s^NB_model = sum_m (M_mt/M_t) * q_avg for NB-type (Eq-A3-NB)
    s_model = q_avg.copy()

    # For NB: aggregate across markets weighted by pop
    nb_mask = ~is_B
    if nb_mask.any():
        # Group by (time_id, deposit_type, entity)
        # NB shares are national: s^NB_jkt = sum_m (M_mt/M_t) * q_rjkmt
        # But NB firms only appear once (national), so s_model = q_avg is correct
        pass

    # Compute Omega_mt (Eq-14): residual B share in market mt
    # Omega_mt = 1 - sum_{NB} s^NB_jkt for B-type markets
    # This is the share of active deposits going to B-type firms in market m
    omega_mt = np.ones(N)
    if is_B.any():
        for t in np.unique(time_ids):
            for k in np.unique(deposit_types):
                t_k_mask = (time_ids == t) & (deposit_types == k)
                nb_share_sum = s_model[t_k_mask & nb_mask].sum()
                b_in_tk = t_k_mask & is_B
                omega_mt[b_in_tk] = max(1e-10, 1.0 - nb_share_sum)

    return s_model, omega_mt

def blp_contraction(df: pd.DataFrame, mu: np.ndarray, R: int,
                    tol: float = 1e-14, max_iter: int = 2000) -> tuple:
    """BLP inner fixed-point contraction (Eq-A4-B and Eq-A4-NB).

    Returns (delta, converged, n_iter, norm_history).
    """
    N = len(df)
    is_B = df['is_B'].values

    # Data-implied shares (precomputed in demand prep)
    s_data_NB = df['share_NB'].values.copy()
    s_data_B_cond = df['share_B_cond'].values.copy()

    # Clamp to avoid log(0)
    s_data_NB = np.clip(s_data_NB, 1e-15, None)
    s_data_B_cond = np.clip(s_data_B_cond, 1e-15, None)
    ln_s_data_NB = np.log(s_data_NB)
    ln_s_data_B_cond = np.log(s_data_B_cond)

    # Initialize delta
    delta = np.zeros(N)
    delta[~is_B] = ln_s_data_NB[~is_B]
    delta[is_B] = ln_s_data_B_cond[is_B]

    norm_history = []
    for h in range(max_iter):
        s_model, omega_mt = compute_model_shares(delta, mu, df, R)
        s_model = np.clip(s_model, 1e-15, None)
        ln_s_model = np.log(s_model)
        ln_omega = np.log(np.clip(omega_mt, 1e-15, None))

        delta_new = delta.copy()
        # NB update (Eq-A4-B label in tex, but conceptually NB)
        nb_mask = ~is_B
        delta_new[nb_mask] = delta[nb_mask] + ln_s_data_NB[nb_mask] - ln_s_model[nb_mask]

        # B update (Eq-A4-NB label in tex, but conceptually B)
        b_mask = is_B
        delta_new[b_mask] = (delta[b_mask]
                             + ln_s_data_B_cond[b_mask]
                             + ln_omega[b_mask]
                             - ln_s_model[b_mask])

        norm = np.max(np.abs(delta_new - delta))
        norm_history.append(norm)

        if norm < tol:
            return delta_new, True, h + 1, norm_history

        delta = delta_new

    return delta, False, max_iter, norm_history


# ==============================================================================
# 6. Linear IV for theta_1 (Eq-A5)
# ==============================================================================
def estimate_theta1(df: pd.DataFrame, delta: np.ndarray) -> tuple:
    """Regress delta on (rho, x) via OLS (k=1,2,3) and 2SLS (k=4,5).

    Returns (theta1, xi_residuals, theta1_se).
    """
    N = len(df)
    deposit_types = df['deposit_type'].values

    # Build regressor matrix: deposit-type-specific spread + x_jt
    # Spreads: K_TYPES columns (one per active type, zero elsewhere)
    spread_cols = np.zeros((N, K_TYPES))
    for idx, k in enumerate(K_LIST):
        mask = deposit_types == k
        spread_cols[mask, idx] = df.loc[mask, 'spread_qoq'].values

    # x columns
    x_mat = np.zeros((N, L_PROD))
    for i, col in enumerate(X_COLS):
        if col in df.columns:
            x_mat[:, i] = df[col].fillna(0).values

    X_full = np.hstack([spread_cols, x_mat])  # (N, 5+L)

    # IVs for k=4,5: replace endogenous spread with instruments
    iv_cols_avail = [c for c in IV_BLP_LOO + IV_COST + IV_CAPITAL if c in df.columns]

    # Separate exogenous (k=1,2,3) and endogenous (k=4,5) observations
    exog_mask = np.isin(deposit_types, [1, 2, 3])
    endog_mask = np.isin(deposit_types, [4, 5])

    # IV matrix: for endog obs, use instruments; for exog, use spreads themselves
    Z_mat = np.zeros((N, len(iv_cols_avail)))
    for i, col in enumerate(iv_cols_avail):
        if col in df.columns:
            Z_mat[:, i] = df[col].fillna(0).values

    # Full instrument matrix: [x, iv_for_endogenous_spread]
    # For 2SLS: Z = [x_mat, spread_exog_dummies, Z_instruments]
    # Construct H = [included exogenous, excluded instruments]
    H = np.hstack([x_mat, Z_mat])  # (N, L + n_iv)

    # Cluster variable for SEs
    clusters = df['CodConglomeradoPrudencial'].values

    # Simple 2SLS implementation
    # First stage: project endogenous spreads on H
    spread_hat = spread_cols.copy()
    for k_idx, k_val in enumerate(K_LIST):
        if k_val in [4, 5]:  # Endogenous types
            k_mask = deposit_types == k_val
            if k_mask.any():
                H_k = H[k_mask]
                spread_k = spread_cols[k_mask, k_idx]
                valid = np.isfinite(H_k).all(axis=1) & np.isfinite(spread_k)
                if valid.sum() > H_k.shape[1]:
                    H_valid = H_k[valid]
                    try:
                        beta_fs = np.linalg.lstsq(H_valid, spread_k[valid], rcond=None)[0]
                        spread_hat[k_mask, k_idx] = H_k @ beta_fs
                    except np.linalg.LinAlgError:
                        pass  # Keep original if projection fails

    X_hat = np.hstack([spread_hat, x_mat])

    # Second stage: regress delta on X_hat
    valid = np.isfinite(X_hat).all(axis=1) & np.isfinite(delta)
    X_v = X_hat[valid]
    delta_v = delta[valid]

    try:
        theta1 = np.linalg.lstsq(X_v, delta_v, rcond=None)[0]
    except np.linalg.LinAlgError:
        theta1 = np.zeros(X_hat.shape[1])

    xi = delta - X_full @ theta1  # Residuals using original X

    # SEs with IK2016 cluster correction
    cluster_v = clusters[valid]
    unique_cl = np.unique(cluster_v)
    G = len(unique_cl)
    sizes = np.array([np.sum(cluster_v == c) for c in unique_cl])
    cv_Ng = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
    G_star = max(1.0, G / (1 + cv_Ng ** 2))

    # Cluster-robust covariance
    try:
        model = sm.OLS(delta_v, X_v)
        res = model.fit(cov_type='cluster',
                        cov_kwds={'groups': cluster_v}, use_t=True)
        res.df_resid = G_star
        theta1_se = res.bse
    except Exception:
        theta1_se = np.full(X_hat.shape[1], np.nan)

    return theta1, xi, theta1_se


# ==============================================================================
# 7. GMM Objective (Eq-A1)
# ==============================================================================
def compute_gmm_moments(xi: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Compute G(theta2) = (1/N) * sum xi * h(Z).

    Returns moment vector G of dimension n_instruments.
    """
    iv_cols_avail = [c for c in IV_BLP_LOO + IV_COST + IV_CAPITAL if c in df.columns]
    Z = np.zeros((len(df), len(iv_cols_avail)))
    for i, col in enumerate(iv_cols_avail):
        Z[:, i] = df[col].fillna(0).values

    N = len(df)
    G = (xi[:, np.newaxis] * Z).mean(axis=0)  # (n_iv,)
    return G


def gmm_objective(theta2_vec: np.ndarray, df: pd.DataFrame,
                   prod_vec: np.ndarray, nu_draws: np.ndarray,
                   aligned_demo_draws: np.ndarray, sigma_indices: list,
                   pi_interactions: list, R: int, coef_dim: int,
                   W: np.ndarray, tol_inner: float,
                   max_inner: int) -> float:
    """Evaluate Q(theta2) = G(theta2)' W G(theta2)."""
    sigma_vals, pi_vals = unpack_theta2(theta2_vec, sigma_indices,
                                         pi_interactions)

    # Compute mu
    mu = compute_mu(prod_vec, nu_draws, aligned_demo_draws, sigma_vals,
                    sigma_indices, pi_vals, pi_interactions, R, coef_dim)

    # Inner loop: contraction
    delta, converged, n_iter, _ = blp_contraction(df, mu, R,
                                                   tol=tol_inner,
                                                   max_iter=max_inner)
    if not converged:
        print(f"  [!] Inner loop did not converge in {n_iter} iterations")

    # Linear IV
    theta1, xi, _ = estimate_theta1(df, delta)

    # GMM moments
    G = compute_gmm_moments(xi, df)

    # Objective
    Q = G @ W @ G
    return Q


# ==============================================================================
# 8. Outer Loop & SE Computation
# ==============================================================================
def run_blp_for_spec(spec_id: int, df_panel: pd.DataFrame, args) -> dict:
    """Full BLP estimation for a single specification."""
    print(f"\n{'='*60}")
    print(f"  Specification {spec_id}")
    print(f"{'='*60}")

    # Load and merge
    df_prep = load_demand_prep(spec_id)
    df = merge_panel_with_prep(df_panel, df_prep)
    del df_prep
    print(f"  Merged panel: {len(df)} observations")

    if len(df) == 0:
        print(f"  [!] No observations after merge. Skipping.")
        return None

    # Build theta2 structure
    sigma_indices, pi_interactions, n_params = build_theta2_structure(args.stage)

    # Simulation draws (shared across specs but must index by this df's markets)
    nu_draws = generate_halton_draws(args.R, K_TYPES + L_PROD, args.seed)

    results = {'spec_id': spec_id, 'stage': args.stage}

    if args.stage == 'logit':
        # ---- STAGE 1: LOGIT (theta2 = 0) ----
        print("  Stage: LOGIT (theta2 = 0)")
        # delta = ln(s_data) directly
        is_B = df['is_B'].values
        delta = np.zeros(len(df))
        s_NB = np.clip(df['share_NB'].values, 1e-15, None)
        s_B = np.clip(df['share_B_cond'].values, 1e-15, None)
        delta[~is_B] = np.log(s_NB[~is_B])
        delta[is_B] = np.log(s_B[is_B])

        theta1, xi, theta1_se = estimate_theta1(df, delta)
        G = compute_gmm_moments(xi, df)
        iv_avail = [c for c in IV_BLP_LOO + IV_COST + IV_CAPITAL if c in df.columns]
        W = np.eye(len(iv_avail))
        Q = G @ W @ G

        results.update({
            'theta1': theta1,
            'theta1_se': theta1_se,
            'theta2': np.array([]),
            'theta2_se': np.array([]),
            'delta': delta,
            'xi': xi,
            'Q_value': Q,
            'converged': True,
            'param_names_theta1': [f'alpha_{k}' for k in K_LIST] + X_COLS
        })
        print(f"  theta1 (alpha_k): {theta1[:K_TYPES]}")
        print(f"  Q(0) = {Q:.6e}")

    else:
        # ---- STAGES 2-4: BLP with theta2 != 0 ----
        print(f"  Stage: {args.stage.upper()} ({n_params} parameters)")

        print("  Precomputing arrays for vectorized logic...")
        demo_draws = generate_demographic_draws(df, args.R, args.seed)
        d_cols_avail = [c for c in D_COLS if c in df.columns]
        D_dim = len(d_cols_avail)
        N_obs = len(df)
        
        aligned_demo_draws = np.zeros((N_obs, args.R, D_dim))
        mca_codes = df['mca_code'].values
        time_ids = df['time_id'].values
        for obs_idx in range(N_obs):
            key = (mca_codes[obs_idx], time_ids[obs_idx])
            if key in demo_draws:
                aligned_demo_draws[obs_idx, :, :] = demo_draws[key]
                
        # Build prod_vec
        coef_dim = K_TYPES + L_PROD
        prod_vec = np.zeros((N_obs, coef_dim))
        deposit_types = df['deposit_type'].values
        spreads = df['spread_qoq'].values
        for idx, k in enumerate(K_LIST):
            mask = deposit_types == k
            prod_vec[mask, idx] = spreads[mask]
        for i, col in enumerate(X_COLS):
            if col in df.columns:
                prod_vec[:, K_TYPES + i] = df[col].fillna(0).values

        # Initial weighting matrix W = (Z'Z)^-1
        iv_avail = [c for c in IV_BLP_LOO + IV_COST + IV_CAPITAL if c in df.columns]
        Z = np.zeros((len(df), len(iv_avail)))
        for i, col in enumerate(iv_avail):
            Z[:, i] = df[col].fillna(0).values
        try:
            W = np.linalg.inv(Z.T @ Z / len(df))
        except np.linalg.LinAlgError:
            W = np.eye(len(iv_avail))

        # Starting values: theta2_0 = small random values near 0
        rng = np.random.default_rng(args.seed)
        theta2_0 = rng.normal(0, 0.01, size=n_params)

        print(f"  Outer minimisation: {args.method}")
        print(f"  Inner tolerance: {args.tol_inner}")

        # Minimise
        obj_fn = lambda t2: gmm_objective(
            t2, df, prod_vec, nu_draws, aligned_demo_draws,
            sigma_indices, pi_interactions, args.R, coef_dim,
            W, args.tol_inner, args.max_inner)

        if args.method == 'l-bfgs-b':
            result = minimize(obj_fn, theta2_0, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': args.tol_outer,
                                       'disp': True})
        else:
            result = minimize(obj_fn, theta2_0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': args.tol_outer,
                                       'fatol': args.tol_outer, 'disp': True})

        theta2_star = result.x
        print(f"  Optimiser converged: {result.success}")
        print(f"  Q(theta2*) = {result.fun:.6e}")
        print(f"  theta2* = {theta2_star}")

        # Recover theta1 at theta2*
        sigma_vals, pi_vals = unpack_theta2(theta2_star, sigma_indices,
                                             pi_interactions)
        mu_star = compute_mu(prod_vec, nu_draws, aligned_demo_draws, sigma_vals,
                             sigma_indices, pi_vals, pi_interactions, args.R, coef_dim)
        delta_star, conv, n_it, _ = blp_contraction(df, mu_star, args.R,
                                                     tol=args.tol_inner,
                                                     max_iter=args.max_inner)
        theta1_star, xi_star, theta1_se = estimate_theta1(df, delta_star)

        # Theta2 SEs: GMM sandwich with numerical Jacobian
        def moment_fn(t2):
            sv, pv = unpack_theta2(t2, sigma_indices, pi_interactions)
            mu_t = compute_mu(prod_vec, nu_draws, aligned_demo_draws, sv, sigma_indices,
                              pv, pi_interactions, args.R, coef_dim)
            d_t, _, _, _ = blp_contraction(df, mu_t, args.R,
                                           tol=args.tol_inner,
                                           max_iter=args.max_inner)
            _, xi_t, _ = estimate_theta1(df, d_t)
            return compute_gmm_moments(xi_t, df)

        try:
            D_jac = np.zeros((len(iv_avail), n_params))
            eps = 1e-5
            G_base = moment_fn(theta2_star)
            for p in range(n_params):
                t2_up = theta2_star.copy()
                t2_up[p] += eps
                G_up = moment_fn(t2_up)
                D_jac[:, p] = (G_up - G_base) / eps

            bread = np.linalg.inv(D_jac.T @ W @ D_jac)
            meat_inner = D_jac.T @ W
            theta2_var = bread @ meat_inner @ np.linalg.inv(W) @ meat_inner.T @ bread
            theta2_se = np.sqrt(np.diag(theta2_var) / len(df))
        except Exception as e:
            print(f"  [!] SE computation failed: {e}")
            theta2_se = np.full(n_params, np.nan)

        results.update({
            'theta1': theta1_star,
            'theta1_se': theta1_se,
            'theta2': theta2_star,
            'theta2_se': theta2_se,
            'delta': delta_star,
            'xi': xi_star,
            'Q_value': result.fun,
            'converged': result.success,
            'n_outer_iter': result.nit,
            'param_names_theta1': [f'alpha_{k}' for k in K_LIST] + X_COLS,
            'sigma_indices': sigma_indices,
            'pi_interactions': pi_interactions
        })
        print(f"  theta1 (alpha_k): {theta1_star[:K_TYPES]}")

    return results


def worker_blp(task):
    """Isolated worker for multiprocessing."""
    sp, df_panel, args = task
    try:
        res = run_blp_for_spec(sp, df_panel, args)
        return sp, res, None
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return sp, None, f"{e}\n{tb}"


# ==============================================================================
# 9. Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="BLP Demand Estimation Loop (Appendix-BLP)")
    parser.add_argument('--spec', type=str, default='12',
                        help='Specification ID (1-12) or "all"')
    parser.add_argument('--stage', type=str, default='logit',
                        choices=['logit', 'sigma', 'full', 'extended', 'sequence'])
    parser.add_argument('--R', type=int, default=500,
                        help='Number of simulation draws')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tol-inner', type=float, default=1e-14,
                        dest='tol_inner')
    parser.add_argument('--max-inner', type=int, default=2000,
                        dest='max_inner')
    parser.add_argument('--tol-outer', type=float, default=1e-6,
                        dest='tol_outer')
    parser.add_argument('--method', type=str, default='l-bfgs-b',
                        choices=['l-bfgs-b', 'nelder-mead'])
    args = parser.parse_args()

    # Determine specs to run
    if args.spec == 'all':
        spec_ids = list(range(1, 13))
    else:
        spec_ids = [int(args.spec)]

    print(f"BLP Demand Estimation Loop")
    print(f"  Stage: {args.stage} | Specs: {spec_ids}")
    print(f"  R={args.R} | seed={args.seed} | method={args.method}")
    print(f"  tol_inner={args.tol_inner} | tol_outer={args.tol_outer}")

    # Load panel once
    df_panel = load_panel_selective(PANEL_CSV)

    # Output directory
    BLP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import multiprocessing
    import concurrent.futures
    max_w = min(len(spec_ids), multiprocessing.cpu_count() - 1, 6)
    if max_w < 1: max_w = 1
    
    stages_to_run = ['logit', 'sigma', 'full', 'extended'] if args.stage == 'sequence' else [args.stage]
    
    for current_stage in stages_to_run:
        args.stage = current_stage
        all_results = {}
        tasks = [(sp, df_panel, args) for sp in spec_ids]
        
        print(f"  Starting parallel execution of {len(spec_ids)} specs for stage '{current_stage}' with {max_w} workers...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_w) as executor:
            for sp, res, err in executor.map(worker_blp, tasks):
                if err is not None:
                    print(f"  [!] Spec {sp} failed:\n{err}")
                elif res is not None:
                    out_pkl = BLP_OUTPUT_DIR / f"blp_results_spec_{sp}_{args.stage}.pkl"
                    with open(out_pkl, 'wb') as f:
                        pickle.dump(res, f)
                    print(f"  Saved: {out_pkl.name}")
                    all_results[sp] = {
                        'Q_value': float(res['Q_value']),
                        'converged': bool(res['converged']),
                        'theta1_alpha': res['theta1'][:K_TYPES].tolist(),
                        'theta2': res['theta2'].tolist() if len(res['theta2']) > 0 else [],
                        'stage': args.stage
                    }

        # Summary JSON
        summary_path = BLP_OUTPUT_DIR / f"blp_summary_{args.stage}.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
        print(f"[DONE] BLP Estimation ({args.stage}) complete for specs {spec_ids}.")


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    main()
