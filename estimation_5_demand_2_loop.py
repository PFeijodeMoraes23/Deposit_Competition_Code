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
  --R {100|500|1000}         Simulation draws; default=100 (CG2020: 50-200 Halton draws sufficient for testing)
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

CLI Options:
------------
usage: estimation_1_demand_2_loop.py [-h] [--spec SPEC]
                                     [--stage {logit,sigma,full,extended,sequence}]
                                     [--R {100,500,1000}] [--seed SEED]
                                     [--tol-inner TOL_INNER]
                                     [--max-inner MAX_INNER]
                                     [--tol-outer TOL_OUTER]
                                     [--method {l-bfgs-b,nelder-mead}]
                                     [--workers WORKERS] [--hpc]

BLP Demand Estimation Loop (Appendix-BLP)

options:
  -h, --help            show this help message and exit
  --spec SPEC           Specification ID (1-12) or "all"
  --stage {logit,sigma,full,extended,sequence}
  --R {100,500,1000}    Number of simulation draws (CG2020: 100 for testing,
                        1000 for final)
  --seed SEED
  --tol-inner TOL_INNER
  --max-inner MAX_INNER
  --tol-outer TOL_OUTER
  --method {l-bfgs-b,nelder-mead}
  --workers WORKERS     Number of simultaneous multiprocessing workers
  --hpc                 Use HPC cluster path structure
"""
import os
import sys
try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except ImportError:
    pass

import json
import pickle
import argparse
import smtplib
import threading
import datetime
import time
import copy
from email.mime.text import MIMEText
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, approx_fprime
from scipy.stats.qmc import Halton
import statsmodels.api as sm

# ==============================================================================
# 0. Paths & Constants
# ==============================================================================

def get_paths(is_hpc: bool) -> tuple:
    """Return (input_dir, output_dir) based on environment."""
    if is_hpc:
        input_dir = Path("/home/pf382/dep_comp/data/input")
        output_dir = Path("/home/pf382/dep_comp/data/output")
    else:
        local_dir_arg = None
        for i, a in enumerate(sys.argv):
            if a == '--local-dir' and i + 1 < len(sys.argv):
                local_dir_arg = sys.argv[i + 1]
                break
                
        if local_dir_arg:
            DATA_DIR = Path(local_dir_arg).resolve()
        else:
            _ROOT = Path(__file__).resolve().parents[2]
            DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
            
        input_dir = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP"
        output_dir = DATA_DIR / "ESTIMATION_OUTPUT" / "BLP_RESULTS"
    return input_dir, output_dir

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
              'loo_equity_ratio', 'mean_loo_equity_ratio',
              'loo_basileia', 'mean_loo_basileia',
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
# 0b. Email Status Notifications
# ==============================================================================
EMAIL_TO   = "pedro.feijodemoraes@yale.edu"
EMAIL_FROM = "pedro.feijodemoraes@yale.edu"
_SMTP_HOST = "smtp.yale.edu"   # Yale unauthenticated relay available on HPC nodes
_SMTP_PORT = 25

_email_log: list = []          # accumulates status lines throughout the run
_email_lock = threading.Lock()


def _log_status(msg: str) -> None:
    """Append a timestamped status line to the in-memory log and print it."""
    stamped = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _email_lock:
        _email_log.append(stamped)
    print(stamped, flush=True)
    # Also write to file
    try:
        is_hpc = '--hpc' in sys.argv
        _, out_dir = get_paths(is_hpc)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "blp_progress.log", "a") as f:
            f.write(stamped + "\\n")
    except Exception:
        pass


def _send_status_email(subject: str, body: str) -> None:
    """Send a plain-text email via Yale SMTP relay (no authentication required)."""
    if '--hpc' in sys.argv:
        return  # Suppress SMTP on HPC nodes where it fails. Rely on SLURM logs.
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = EMAIL_FROM
    msg["To"]      = EMAIL_TO
    try:
        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=15) as server:
            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
    except Exception as exc:
        print(f"  [EMAIL-WARN] Could not send email: {exc}", flush=True)


def _hourly_email_worker(job_id: str, stop_event: threading.Event) -> None:
    """Background thread: every hour during 07:00-21:00 send a status digest."""
    while not stop_event.wait(timeout=3600):
        hour = datetime.datetime.now().hour
        if 7 <= hour < 21:
            with _email_lock:
                recent = list(_email_log[-200:])
            body = "\n".join(recent) if recent else "No status lines yet."
            subject = (f"[BLP-HPC] Hourly Status | Job {job_id} | "
                       f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
            _send_status_email(subject, body)


# ==============================================================================
# 1. Data Loading
# ==============================================================================
def load_merged_spec_data(spec_id: int, is_hpc: bool = False, alt: str = "alt2logistic") -> pd.DataFrame:
    '''Load the per-spec pre-merged dataframe created by estimation_2_demand_1_prep.py'''
    input_dir, _ = get_paths(is_hpc)
    pkl_path = input_dir / f"demand_5_{alt}_final_spec_{spec_id}.parquet"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {pkl_path}")
    return pd.read_parquet(pkl_path, engine='pyarrow')

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
               stacked_draws: np.ndarray, obs_key_idx: np.ndarray,
               sigma_vals: np.ndarray, sigma_indices: list,
               pi_vals: np.ndarray, pi_interactions: list,
               R: int, coef_dim: int) -> np.ndarray:
    """Compute mu_rjkmt efficiently using vectorized array ops."""
    
    # 1. Sigma term:
    sigma_diag = np.zeros(coef_dim)
    for idx_pos, coef_idx in enumerate(sigma_indices):
        if coef_idx < coef_dim:
            sigma_diag[coef_idx] = sigma_vals[idx_pos]
            
    sigma_nu = sigma_diag * nu_draws[:, :coef_dim]  # (R, coef_dim)
    mu = np.dot(prod_vec, sigma_nu.T)  # (N, R)
    
    # 2. Pi term:
    if pi_interactions:
        D = stacked_draws.shape[2]
        for pi_idx, (coef_idx, demo_idx) in enumerate(pi_interactions):
            if coef_idx < coef_dim and demo_idx < D:
                p_v = prod_vec[:, coef_idx]  # (N,)
                d_v = stacked_draws[obs_key_idx, :, demo_idx]  # (N, R)
                mu += pi_vals[pi_idx] * p_v[:, np.newaxis] * d_v
                
    return mu

# ==============================================================================
# 5. Inner Loop: BLP Contraction (Eq-A4)
# ==============================================================================
def _build_market_indices(mca_codes: np.ndarray, time_ids: np.ndarray,
                          is_B: np.ndarray) -> tuple:
    """Build unified (mca, time) market group indices for ALL products.

    Per Eq-4 / Eq-13-B, the softmax denominator sums over J_mt = J^B_mt âˆª J^D_t.
    D-firms (is_B=False) appear in every local market because they operate
    nationally; their rows in the panel carry mca_code='0'. We assign them
    the per-time-id market index for each (mca, time) market by broadcasting
    their contributions separately (see compute_model_shares).

    Returns
    -------
    b_mkt_idx : (N_B,) int  â€” market index for each B-firm row
    d_time_enc : (N_D,) int â€” time-period encoding for each D-firm row
    mca_time_pairs : list of (mca, time) tuples, one per unique B-market
    unique_times : array of unique time_ids
    """
    b_mask = is_B
    d_mask = ~is_B

    # Unique (mca, time) B-markets
    b_mca  = mca_codes[b_mask]
    b_time = time_ids[b_mask]
    raw_pairs = list(zip(b_mca.tolist(), b_time.tolist()))
    unique_pairs = sorted(set(raw_pairs))
    pair_to_idx  = {p: i for i, p in enumerate(unique_pairs)}
    b_mkt_idx    = np.array([pair_to_idx[(m, t)] for m, t in raw_pairs], dtype=np.int64)

    # Time encoding for D-firms (they contribute to every B-market in their period)
    unique_times = np.unique(time_ids)
    time_to_idx  = {t: i for i, t in enumerate(unique_times.tolist())}
    d_time_enc   = np.array([time_to_idx[t] for t in time_ids[d_mask].tolist()], dtype=np.int64)

    return b_mkt_idx, d_time_enc, unique_pairs, unique_times


def compute_model_shares(delta: np.ndarray, mu: np.ndarray,
                         df: pd.DataFrame, R: int,
                         precomp: dict | None = None) -> tuple:
    """Compute model-implied shares s^{Act,B} and s^{Act,D} (Eq-13-B/D).

    All products â€” B and D â€” compete in the same softmax denominator for
    each (mca, time) market, consistent with Eq-4 of V_Main.tex.

    D-firms are national (single row per time period) but participate in
    every local (mca, time) market. Their exp(Î´+Î¼) sum is computed once per
    time period and added to every corresponding market's denominator.

    Returns
    -------
    s_B     : (N_B,)  model-implied local share for B-firm rows (Eq-13-B avg)
    s_D_nat : (N_D,)  model-implied national share for D-firm rows (Eq-13-D)
    omega   : (N_B,)  Î©_mt per B-firm row (Eq-14)
    is_B    : (N,)    boolean mask (B-type rows)
    """
    N     = len(df)
    is_B  = df['is_B'].values
    mca_codes    = df['mca_code'].values
    time_ids     = df['time_id'].values
    deposit_types = df['deposit_type'].values
    pop_total    = df['pop_total'].fillna(0).values

    b_mask = is_B
    d_mask = ~is_B
    N_B    = b_mask.sum()
    N_D    = d_mask.sum()

    # --- Pull precomputed indices or build them ---
    if precomp is not None:
        b_mkt_idx   = precomp['b_mkt_idx']
        d_time_enc  = precomp['d_time_enc']
        unique_pairs = precomp['unique_pairs']
        unique_times = precomp['unique_times']
        pair_time_enc = precomp['pair_time_enc']   # (n_pairs,) maps each B-market to its time index
        pop_weights   = precomp['pop_weights']     # (n_pairs,) population weight per B-market
    else:
        b_mkt_idx, d_time_enc, unique_pairs, unique_times = _build_market_indices(
            mca_codes, time_ids, is_B)
        pair_times   = np.array([p[1] for p in unique_pairs])
        ut_map       = {t: i for i, t in enumerate(unique_times.tolist())}
        pair_time_enc = np.array([ut_map[t] for t in pair_times.tolist()], dtype=np.int64)
        # Population weight for each B-market (for D-type national share aggregation, Eq-13-D)
        pop_b        = pop_total[b_mask]
        pair_pop     = np.bincount(b_mkt_idx, weights=pop_b, minlength=len(unique_pairs))
        time_pop     = np.bincount(pair_time_enc, weights=pair_pop, minlength=len(unique_times))
        pair_pop_norm = pair_pop / np.maximum(time_pop[pair_time_enc], 1e-30)
        pop_weights   = pair_pop_norm   # (n_pairs,) sums to 1 within each time period

    n_pairs  = len(unique_pairs)
    n_times  = len(unique_times)

    # --- Slice delta and mu for B and D rows ---
    delta_B = delta[b_mask]   # (N_B,)
    delta_D = delta[d_mask]   # (N_D,)
    mu_B    = mu[b_mask, :]   # (N_B, R)
    mu_D    = mu[d_mask, :]   # (N_D, R)

    # --- Vectorised over R: clamp V before log-sum-exp to prevent overflow (Fix 4) ---
    V_B = np.clip(delta_B[:, np.newaxis] + mu_B, -500.0, 500.0)   # (N_B, R)
    V_D = np.clip(delta_D[:, np.newaxis] + mu_D, -500.0, 500.0)   # (N_D, R)

    # For each (time, R), sum exp(V_D) over all D-products in that time period
    # We keep this in log-space: max_VD_time and log_sum_D_shifted
    log_sum_D_shifted = np.zeros((n_times, R))
    max_VD_time       = np.full((n_times, R), -np.inf)
    
    # d_time_enc is (N_D,)
    for i in range(n_times):
        mask_t = d_time_enc == i
        if mask_t.any():
            V_D_t  = V_D[mask_t, :]         # (n_D_t, R)
            m_vd   = V_D_t.max(axis=0)      # (R,)
            
            log_s  = np.log(np.maximum(1e-300, np.exp(V_D_t - m_vd).sum(axis=0)))
            log_sum_D_shifted[i, :] = log_s
            max_VD_time[i, :]       = m_vd

    # For each B-market, fetch the D elements
    max_VD_per_pair = max_VD_time[pair_time_enc, :]               # (n_pairs, R)
    log_sum_D_per_pair = log_sum_D_shifted[pair_time_enc, :]      # (n_pairs, R)
    log_D_sum = max_VD_per_pair + log_sum_D_per_pair

    # Stable softmax denominator for B-products within each market
    # Step 1: per-market max over B-products (for numerical stability)
    max_VB_per_mkt = np.full((n_pairs, R), -np.inf)
    np.maximum.at(max_VB_per_mkt, b_mkt_idx, V_B)            # scatter-max

    # Step 2: exp(V_B - max) and sum per market
    V_B_shifted  = V_B - max_VB_per_mkt[b_mkt_idx, :]        # (N_B, R)  stable
    exp_VB       = np.exp(V_B_shifted)                       # (N_B, R)
    sum_exp_B_per_mkt = np.zeros((n_pairs, R))
    np.add.at(sum_exp_B_per_mkt, b_mkt_idx, exp_VB)          # (n_pairs, R)
    
    log_B_sum = max_VB_per_mkt + np.log(np.maximum(1e-300, sum_exp_B_per_mkt))
    
    # Joint log-denominator with true outside option
    # OUTSIDE_EPS = 1.0 corresponds to V_0 = 0 -> exp(0) = 1.0
    OUTSIDE_EPS = 1.0
    log_outside = np.log(OUTSIDE_EPS)  # 0.0
    joint_max = np.maximum(np.maximum(log_B_sum, log_D_sum), log_outside)
    log_denom = joint_max + np.log(np.maximum(1e-300,
        np.exp(log_outside - joint_max)
        + np.exp(log_B_sum - joint_max)
        + np.exp(log_D_sum - joint_max)))

    # B-firm individual softmax probabilities: Eq-13-B averaged over R
    # q_B = exp(V_B - log_denom)
    log_denom_B = log_denom[b_mkt_idx, :]
    q_B         = np.exp(V_B - log_denom_B)                     # (N_B, R)
    s_B         = q_B.mean(axis=1)                              # (N_B,)

    # --- D-firm national shares: Eq-13-D  s^{Act,D} = Î£_m (M_mt/M_t) * q_D_m ---
    # Vectorised approach: s_D_nat[d] = mean_r[ exp(V_D_d,r) * sum_m (w_m * exp(-log_denom_m,r)) ]
    
    neg_log_denom = -log_denom
    max_neg_ld = np.full((n_times, R), -np.inf)
    np.maximum.at(max_neg_ld, pair_time_enc, neg_log_denom)
    
    shifted_inv_denom = np.exp(neg_log_denom - max_neg_ld[pair_time_enc])    # (n_pairs, R)
    wtd_shifted_inv_denom = shifted_inv_denom * pop_weights[:, np.newaxis]   # (n_pairs, R)
    
    sum_wtd_inv_denom = np.zeros((n_times, R))
    np.add.at(sum_wtd_inv_denom, pair_time_enc, wtd_shifted_inv_denom)       # (n_times, R)
    
    log_inv_denom_wtd = max_neg_ld + np.log(np.maximum(1e-300, sum_wtd_inv_denom)) # (n_times, R)
    
    log_s_D_r = V_D + log_inv_denom_wtd[d_time_enc, :]                         # (N_D, R)
    row_max_s_D = log_s_D_r.max(axis=1, keepdims=True)
    s_D_nat = np.exp(log_s_D_r - row_max_s_D).mean(axis=1) * np.exp(row_max_s_D.squeeze()) # (N_D,)


    # --- Î©_mt: Eq-14 = 1 - Î£_{jâˆˆJ^D_t, k} s^{Act,D}_{jk,local,mt} ---
    # D_share_p_r = exp(log_D_sum_p,r - log_denom_p,r)
    # Omega correction removed: share_B_cond = Dep_Act / (d_bar * pop) is already
    # an unconditional share; ln(Omega) double-corrected and biased delta_B downward.

    return s_B, s_D_nat, is_B

def _contraction_step(delta, mu, df, R, b_mask, d_mask,
                      ln_s_data_D, ln_s_data_B_cond, precomp):
    """Standard BLP contraction T(delta) = delta + ln(s_data) - ln(s_model).
    share_B_cond is unconditional (Dep_Act / (d_bar * pop)); no Omega term needed.
    """
    s_B, s_D_nat, _ = compute_model_shares(delta, mu, df, R, precomp=precomp)
    delta_new = delta.copy()
    delta_new[d_mask] = (delta[d_mask]
                         + ln_s_data_D[d_mask]
                         - np.log(np.clip(s_D_nat, 1e-15, None)))
    delta_new[b_mask] = (delta[b_mask]
                         + ln_s_data_B_cond[b_mask]
                         - np.log(np.clip(s_B, 1e-15, None)))
    return np.clip(delta_new, -500.0, 500.0)


def blp_contraction(df, mu, R, tol=1e-12, max_iter=1500,
                    delta_init=None, precomp=None):
    """Anderson(m=5)-accelerated BLP inner contraction (Walker & Ni 2011).

    Maintains last m residuals and solves a (m x m) constrained LS problem each
    iteration to extrapolate towards the fixed point.  Empirically 10-50x faster
    than SQUAREM when the Jacobian spectral radius is near 1
    (Conlon & Gortmaker 2020, Sec 3.3).

    Parameters
    ----------
    delta_init : optional warm-start vector.
    precomp    : optional dict of pre-built market-index arrays.

    Returns
    -------
    (delta, converged, n_iter, norm_history)
    """
    N      = len(df)
    is_B   = df['is_B'].values
    b_mask = is_B
    d_mask = ~is_B

    s_data_D      = np.clip(df['share_D'].values.copy(),      1e-15, None)
    s_data_B_cond = np.clip(df['share_B_cond'].values.copy(), 1e-15, None)
    ln_s_data_D      = np.log(s_data_D)
    ln_s_data_B_cond = np.log(s_data_B_cond)

    if delta_init is not None and delta_init.shape == (N,):
        delta = delta_init.copy()
    else:
        delta = np.zeros(N)
        delta[d_mask] = ln_s_data_D[d_mask]
        delta[b_mask] = ln_s_data_B_cond[b_mask]

    # Anderson(m=5) history buffers
    m_aa     = 5
    F_hist   = np.zeros((N, m_aa))   # residual history (N, m)
    X_hist   = np.zeros((N, m_aa))   # iterate  history (N, m)
    ptr      = 0
    hist_len = 0

    norm_history = []
    for h in range(max_iter):
        delta_new = _contraction_step(delta, mu, df, R, b_mask, d_mask,
                                      ln_s_data_D, ln_s_data_B_cond, precomp)
        if not np.all(np.isfinite(delta_new)):
            n_bad = (~np.isfinite(delta_new)).sum()
            print(f"    [contraction ABORT] {n_bad}/{N} non-finite entries.", flush=True)
            return delta, False, h + 1, norm_history

        f    = delta_new - delta
        norm = np.max(np.abs(f))
        norm_history.append(norm)

        if h % 50 == 0 or norm < tol:
            print(f"    [Anderson iter={h+1}/{max_iter}] norm={norm:.3e}", flush=True)

        if norm < tol:
            return delta_new, True, h + 1, norm_history

        # Anderson mixing update
        F_hist[:, ptr] = f
        X_hist[:, ptr] = delta
        ptr      = (ptr + 1) % m_aa
        hist_len = min(hist_len + 1, m_aa)

        if hist_len > 1:
            F_k = F_hist[:, :hist_len]
            dF  = F_k[:, 1:] - F_k[:, [0]]        # (N, k-1)
            try:
                rhs   = -dF.T @ F_k[:, 0]          # (k-1,)
                A_aa  = dF.T @ dF + 1e-10 * np.eye(hist_len - 1)
                c_bar = np.linalg.solve(A_aa, rhs)  # (k-1,)
                c     = np.empty(hist_len)
                c[0]  = 1.0 - c_bar.sum()
                c[1:] = c_bar
                d_aa  = (X_hist[:, :hist_len] + F_hist[:, :hist_len]) @ c
                delta = (np.clip(d_aa, -500.0, 500.0)
                         if np.all(np.isfinite(d_aa)) else delta_new)
            except np.linalg.LinAlgError:
                delta = delta_new
        else:
            delta = delta_new

    return delta, False, max_iter, norm_history



# ==============================================================================
# 6. Linear IV for theta_1 (Eq-A5)
# ==============================================================================
def _build_regressor_matrices(df: pd.DataFrame, deposit_types: np.ndarray, N: int) -> tuple:
    spread_cols = np.zeros((N, K_TYPES))
    for idx, k in enumerate(K_LIST):
        mask = deposit_types == k
        spread_cols[mask, idx] = df.loc[mask, 'spread_qoq'].values

    x_mat = np.zeros((N, L_PROD))
    for i, col in enumerate(X_COLS):
        if col in df.columns:
            x_mat[:, i] = df[col].fillna(0).values

    iv_cols_avail = [c for c in IV_BLP_LOO + IV_COST + IV_CAPITAL if c in df.columns]
    Z_mat = np.zeros((N, len(iv_cols_avail)))
    for i, col in enumerate(iv_cols_avail):
        if col in df.columns:
            Z_mat[:, i] = df[col].fillna(0).values
            
    return spread_cols, x_mat, Z_mat

def _project_endogenous_spreads(spread_cols: np.ndarray, H: np.ndarray, deposit_types: np.ndarray) -> np.ndarray:
    spread_hat = spread_cols.copy()
    for k_idx, k_val in enumerate(K_LIST):
        if k_val in [4, 5]:
            k_mask = deposit_types == k_val
            if k_mask.any():
                H_k = H[k_mask]
                spread_k = spread_cols[k_mask, k_idx]
                valid = np.isfinite(H_k).all(axis=1) & np.isfinite(spread_k)
                if valid.sum() > H_k.shape[1]:
                    import contextlib
                    with contextlib.suppress(np.linalg.LinAlgError):
                        beta_fs = np.linalg.lstsq(H_k[valid], spread_k[valid], rcond=None)[0]
                        k_spread_hat = spread_k.copy()
                        k_spread_hat[valid] = H_k[valid] @ beta_fs
                        spread_hat[k_mask, k_idx] = k_spread_hat
    return spread_hat

def _compute_cluster_robust_se(delta_v: np.ndarray, X_v: np.ndarray, n_cols: int, clusters: np.ndarray) -> np.ndarray:
    unique_cl = np.unique(clusters)
    G = len(unique_cl)
    sizes = np.array([np.sum(clusters == c) for c in unique_cl])
    cv_Ng = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
    
    try:
        model = sm.OLS(delta_v, X_v)
        res = model.fit(cov_type='cluster', cov_kwds={'groups': clusters}, use_t=True)
        res.df_resid = max(1.0, G / (1 + cv_Ng ** 2))
        return res.bse
    except Exception:
        return np.full(n_cols, np.nan)

def estimate_theta1(df: pd.DataFrame, delta: np.ndarray) -> tuple:
    """Regress delta on (rho, x) via OLS (k=1,2,3) and 2SLS (k=4,5).

    Returns (theta1, xi_residuals, theta1_se).
    """
    N = len(df)
    deposit_types = df['deposit_type'].values

    spread_cols, x_mat, Z_mat = _build_regressor_matrices(df, deposit_types, N)

    X_full = np.hstack([spread_cols, x_mat])  # (N, 5+L)
    H = np.hstack([x_mat, Z_mat])  # (N, L + n_iv)

    spread_hat = _project_endogenous_spreads(spread_cols, H, deposit_types)
    X_hat = np.hstack([spread_hat, x_mat])

    valid = np.isfinite(X_hat).all(axis=1) & np.isfinite(delta)
    X_v = X_hat[valid]
    delta_v = delta[valid]

    try:
        theta1 = np.linalg.lstsq(X_v, delta_v, rcond=None)[0]
    except np.linalg.LinAlgError:
        theta1 = np.zeros(X_hat.shape[1])

    xi = delta - X_full @ theta1  # Residuals using original X

    theta1_se = _compute_cluster_robust_se(delta_v, X_v, X_hat.shape[1], (df['CodConglomeradoPrudencial'].astype(str) + "_" + df['time_id'].str.split('Q').str[0]).values[valid])

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
        Z[:, i] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0).values

    N = len(df)
    return (xi[:, np.newaxis] * Z).mean(axis=0)  # (n_iv,)


def gmm_objective(theta2_vec: np.ndarray, df: pd.DataFrame,
                   prod_vec: np.ndarray, nu_draws: np.ndarray,
                   stacked_draws: np.ndarray, obs_key_idx: np.ndarray,
                   sigma_indices: list,
                   pi_interactions: list, R: int, coef_dim: int,
                   W: np.ndarray, tol_inner: float,
                   max_inner: int,
                   delta_cache: dict | None = None,
                   precomp: dict | None = None) -> float:
    """Evaluate Q(theta2) = G(theta2)' W G(theta2)."""
    sigma_vals, pi_vals = unpack_theta2(theta2_vec, sigma_indices,
                                         pi_interactions)

    # Compute mu
    mu = compute_mu(prod_vec, nu_draws, stacked_draws, obs_key_idx, sigma_vals,
                    sigma_indices, pi_vals, pi_interactions, R, coef_dim)

    # Warm-start delta from previous outer iteration if available
    delta_init = delta_cache.get('last_delta') if delta_cache is not None else None

    # Inner loop: contraction
    delta, converged, n_iter, _ = blp_contraction(
        df, mu, R, tol=tol_inner, max_iter=max_inner,
        delta_init=delta_init, precomp=precomp)
    if not converged:
        print(f"  [!] Inner loop did not converge in {n_iter} iterations", flush=True)
    # Always cache latest delta for warm-starting (CG2020 ÂSection 3.2)
    if delta_cache is not None:
        delta_cache['last_delta'] = delta.copy()

    # Linear IV
    theta1, xi, _ = estimate_theta1(df, delta)

    # GMM moments
    G = compute_gmm_moments(xi, df)

    # Objective
    return G @ W @ G


# ==============================================================================
# 8. Outer Loop & SE Computation
# ==============================================================================
def run_blp_for_spec(spec_id: int, args) -> dict:
    '''Full BLP estimation for a single specification.'''
    print(f"\n{'='*60}")
    print(f"  Specification {spec_id}")
    print(f"{'='*60}")

    # Load pre-merged spec dataframe
    try:
        df = load_merged_spec_data(spec_id, getattr(args, 'hpc', False), getattr(args, 'alt', 'alt2logistic'))
        print(f"  Merged panel loaded: {len(df)} observations")
    except FileNotFoundError:
        print(f"  [!] No merged data for spec {spec_id}. Skipping.")
        return None

    if len(df) == 0:
        print("  [!] No observations after merge. Skipping.")
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
        s_D = np.clip(df['share_D'].values, 1e-15, None)
        s_B = np.clip(df['share_B_cond'].values, 1e-15, None)
        delta[~is_B] = np.log(s_D[~is_B])
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
        
        mca_codes = df['mca_code'].values
        time_ids = df['time_id'].values
        # Build lookup: unique (mca, time) keys mapped to integer index
        unique_keys = list(demo_draws.keys())
        key_to_idx = {k: i for i, k in enumerate(unique_keys)}
        # Stack all demo draw matrices -> (n_unique, R, D)
        stacked_draws = np.stack([demo_draws[k] for k in unique_keys], axis=0)
        # Map each observation to its key index, tracking missing inputs to a null zero lookup index
        zero_draws = np.zeros((args.R, D_dim))
        stacked_draws_padded = np.concatenate([stacked_draws, zero_draws[np.newaxis, :, :]], axis=0)
        padding_idx = stacked_draws.shape[0]
        
        obs_key_idx = np.array([key_to_idx.get((mca_codes[i], time_ids[i]), padding_idx)
                                 for i in range(N_obs)])
        # Eliminated `aligned_demo_draws` allocation: Memory-efficient broadcasts will utilize stacked_draws_padded.
                
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
            Z[:, i] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0).values
        try:
            W = np.linalg.inv(Z.T @ Z / len(df))
        except np.linalg.LinAlgError:
            W = np.eye(len(iv_avail))

        # Starting values: warm-start from previous-stage checkpoint if available (Fix 3).
        # Saves 10+ hours when full/extended start from sigma*/full* instead of N(0,0.01).
        chk_dir = get_paths(getattr(args, 'hpc', False))[1]
        prev_stages = {'full': 'sigma', 'extended': 'full'}
        theta2_0 = None
        if args.stage in prev_stages:
            prev_chk = chk_dir / f"blp_checkpoint_spec_{spec_id}_{prev_stages[args.stage]}.pkl"
            if prev_chk.exists():
                try:
                    with open(prev_chk, 'rb') as _f:
                        _chk = pickle.load(_f)
                    _t2_prev = _chk.get('theta2_star', None)
                    if _t2_prev is not None and len(_t2_prev) <= n_params:
                        theta2_0 = np.zeros(n_params)
                        theta2_0[:len(_t2_prev)] = _t2_prev
                        print(f"  [WARM-START] theta2_0 loaded from {prev_chk.name}", flush=True)
                        _d_prev = _chk.get('delta_star', None)
                        if _d_prev is not None and _d_prev.shape == (len(df),):
                            delta_cache['last_delta'] = _d_prev.copy()
                except Exception as _e:
                    print(f"  [WARM-START] Could not load checkpoint: {_e}", flush=True)
        if theta2_0 is None:
            rng = np.random.default_rng(args.seed)
            theta2_0 = rng.normal(0, 0.01, size=n_params)

        print(f"  Outer minimisation: {args.method}", flush=True)
        print(f"  Inner tolerance: {args.tol_inner}", flush=True)

        # Pre-build market index arrays once (reused across all contraction calls)
        mca_codes_arr = df['mca_code'].values
        time_ids_arr  = df['time_id'].values
        is_B_arr      = df['is_B'].values
        b_mkt_idx, d_time_enc, unique_pairs, unique_times = _build_market_indices(
            mca_codes_arr, time_ids_arr, is_B_arr)
        pair_times    = np.array([p[1] for p in unique_pairs])
        ut_map        = {t: i for i, t in enumerate(unique_times.tolist())}
        pair_time_enc = np.array([ut_map[t] for t in pair_times.tolist()], dtype=np.int64)
        pop_b         = df['pop_total'].fillna(0).values[is_B_arr]
        pair_pop      = np.bincount(b_mkt_idx, weights=pop_b, minlength=len(unique_pairs))
        time_pop      = np.bincount(pair_time_enc, weights=pair_pop, minlength=len(unique_times))
        pair_pop_norm = pair_pop / np.maximum(time_pop[pair_time_enc], 1e-30)
        precomp_idx   = {
            'b_mkt_idx':    b_mkt_idx,
            'd_time_enc':   d_time_enc,
            'unique_pairs': unique_pairs,
            'unique_times': unique_times,
            'pair_time_enc': pair_time_enc,
            'pop_weights':   pair_pop_norm,
        }

        # Delta warm-start cache (shared across outer iterations)
        delta_cache: dict = {}

        # DRY RUN LOGIC
        if getattr(args, 'dry_run', False):
            print(f"  [DRY RUN] Running 10 contraction iterations for timing...")
            sigma_vals, pi_vals = unpack_theta2(theta2_0, sigma_indices, pi_interactions)
            mu_t = compute_mu(prod_vec, nu_draws, stacked_draws_padded, obs_key_idx, sigma_vals,
                              sigma_indices, pi_vals, pi_interactions, args.R, coef_dim)
            t0 = time.time()
            d_t, conv, n_it, n_hist = blp_contraction(df, mu_t, args.R,
                                           tol=args.tol_inner, max_iter=10, precomp=precomp_idx)
            t1 = time.time()
            elapsed = t1 - t0
            print(f"  [DRY RUN] 10 iterations in {elapsed:.2f}s ({elapsed/10:.3f} s/iter).")
            print(f"  [DRY RUN] Exiting early.")
            return {'dry_run': True}

        # Outer callback for progress logging (replaces deprecated disp=True)
        _outer_iter = [0]
        def _outer_callback(xk):
            _outer_iter[0] += 1
            _log_status(f"  [OUTER iter={_outer_iter[0]}] theta2={np.round(xk, 4)}")

        def obj_fn(t2):
            return gmm_objective(
                t2, df, prod_vec, nu_draws, stacked_draws_padded, obs_key_idx,
                sigma_indices, pi_interactions, args.R, coef_dim,
                W, args.tol_inner, args.max_inner,
                delta_cache=delta_cache, precomp=precomp_idx)

        # Bounds to prevent float64 overflow during line search
        bnds = [(-15.0, 15.0)] * n_params
        if args.method == 'l-bfgs-b':
            result = minimize(obj_fn, theta2_0, method='L-BFGS-B',
                              bounds=bnds,
                              options={'maxiter': 500, 'ftol': args.tol_outer},
                              callback=_outer_callback)
        else:
            result = minimize(obj_fn, theta2_0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': args.tol_outer,
                                       'fatol': args.tol_outer},
                              callback=_outer_callback)

        theta2_star = result.x
        print(f"  Optimiser converged: {result.success}")
        print(f"  Q(theta2*) = {result.fun:.6e}")
        print(f"  theta2* = {theta2_star}")

        # Recover theta1 at theta2*
        sigma_vals, pi_vals = unpack_theta2(theta2_star, sigma_indices,
                                             pi_interactions)
        mu_star = compute_mu(prod_vec, nu_draws, stacked_draws_padded, obs_key_idx, sigma_vals,
                             sigma_indices, pi_vals, pi_interactions, args.R, coef_dim)
        delta_star, conv, n_it, _ = blp_contraction(df, mu_star, args.R,
                                                     tol=args.tol_inner,
                                                     max_iter=args.max_inner,
                                                     delta_init=delta_cache.get('last_delta'),
                                                     precomp=precomp_idx)
        theta1_star, xi_star, theta1_se = estimate_theta1(df, delta_star)

        # Theta2 SEs: GMM sandwich with numerical Jacobian
        def moment_fn(t2):
            sv, pv = unpack_theta2(t2, sigma_indices, pi_interactions)
            mu_t = compute_mu(prod_vec, nu_draws, stacked_draws_padded, obs_key_idx, sv, sigma_indices,
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
            for p_idx in range(n_params):
                t2_up = theta2_star.copy()
                t2_up[p_idx] += eps
                G_up = moment_fn(t2_up)
                D_jac[:, p_idx] = (G_up - G_base) / eps

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
        # Save stage checkpoint so next stage can warm-start (Fix 3)
        try:
            _chk_dir = get_paths(getattr(args, 'hpc', False))[1]
            _chk_dir.mkdir(parents=True, exist_ok=True)
            _chk_path = _chk_dir / f"blp_checkpoint_spec_{spec_id}_{args.stage}.pkl"
            with open(_chk_path, 'wb') as _cf:
                pickle.dump({'theta2_star': theta2_star, 'delta_star': delta_star}, _cf)
            print(f"  [CHECKPOINT] Saved {_chk_path.name}", flush=True)
        except Exception as _ce:
            print(f"  [CHECKPOINT-WARN] {_ce}", flush=True)


    return results


def worker_blp(task):
    '''Isolated worker for multiprocessing.'''
    sp, args = task
    try:
        res = run_blp_for_spec(sp, args)
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
    parser.add_argument('--alt', type=str, default='alt2logistic', choices=['alt1', 'alt2', 'alt2linear', 'alt2logistic', 'all'],
                        help='Alternative variant to run (alt1, alt2, alt2linear, alt2logistic, or all)')
    parser.add_argument('--stage', type=str, default='logit',
                        choices=['logit', 'sigma', 'full', 'extended', 'sequence'])
    parser.add_argument('--R', type=int, default=100,
                        choices=[10, 100, 500, 1000],
                        help='Number of simulation draws (CG2020: 100 for testing, 1000 for final)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tol-inner', type=float, default=1e-12,
                        dest='tol_inner')
    parser.add_argument('--max-inner', type=int, default=2000,
                        dest='max_inner')
    parser.add_argument('--tol-outer', type=float, default=1e-6,
                        dest='tol_outer')
    parser.add_argument('--method', type=str, default='l-bfgs-b',
                        choices=['l-bfgs-b', 'nelder-mead'])
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of simultaneous multiprocessing workers')
    parser.add_argument('--hpc', action='store_true',
                        help='Use HPC cluster path structure')
    parser.add_argument('--local-dir', type=str, default=None,
                        help='Local path to the "processed" directory where ESTIMATION_OUTPUT is located')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run 10 contraction iters and exist for timing')
    args = parser.parse_args()

    # Determine specs to run
    spec_ids = list(range(1, 13)) if args.spec == 'all' else [int(args.spec)]

    _log_status("BLP Demand Estimation Loop \u2014 START")
    _log_status(f"  Stage: {args.stage} | Specs: {spec_ids}")
    _log_status(f"  R={args.R} | seed={args.seed} | method={args.method} | HPC={args.hpc}")
    _log_status(f"  tol_inner={args.tol_inner} | tol_outer={args.tol_outer} | workers={args.workers}")

    # --- Launch hourly email status thread ---
    job_id   = os.environ.get('SLURM_JOB_ID', 'LOCAL')
    _stop_email = threading.Event()
    _email_thread = threading.Thread(
        target=_hourly_email_worker, args=(job_id, _stop_email), daemon=True)
    _email_thread.start()
    # Send an immediate start notification
    _send_status_email(
        f"[BLP-HPC] Job {job_id} STARTED â€” {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Estimation started.\nStage={args.stage} | Specs={spec_ids} | R={args.R} | workers={args.workers}"
    )

    # Directories
    input_dir, BLP_OUTPUT_DIR = get_paths(args.hpc)
    print(f"\n  [DIAGNOSTIC] Input Directory: {input_dir}", flush=True)
    print(f"  [DIAGNOSTIC] Output Directory: {BLP_OUTPUT_DIR}", flush=True)
    
    if not input_dir.exists():
        print(f"  [FATAL] Input directory DOES NOT EXIST: {input_dir}", flush=True)
    else:
        pkl_files = list(input_dir.glob('demand_5_*_final_spec_*.parquet'))
        print(f"  [DIAGNOSTIC] Found {len(pkl_files)} matched .pkl files in input directory.", flush=True)

    BLP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import multiprocessing
    import concurrent.futures
    import gc
    
    max_w = args.workers
    
    stages_to_run = ['logit', 'sigma', 'full', 'extended'] if args.stage == 'sequence' else [args.stage]
    
    alts_to_process = ['alt1', 'alt2linear', 'alt2logistic'] if args.alt == 'all' else [args.alt]
    for current_alt in alts_to_process:
        args.alt = current_alt
        
        for current_stage in stages_to_run:
            args.stage = current_stage
            
            # In sequence mode, use R=100 for sigma to save time (sufficient per CG2020)
            if hasattr(args, '_orig_R') is False:
                args._orig_R = args.R
            
            if args.stage == 'sequence' and current_stage == 'sigma':
                pass # handled above, sequence doesn't equal sigma
                
            if current_stage == 'sigma' and 'sequence' in sys.argv: # user passed --stage sequence
                args.R = 100
                print(f"  [SEQUENCE] Overriding R=100 for sigma stage per CG2020.")
            else:
                args.R = args._orig_R
                
            all_results = {}
            tasks = [(sp, copy.copy(args)) for sp in spec_ids]
            
            print(f"  Starting parallel execution of {len(spec_ids)} specs for stage '{current_stage}' with alt '{current_alt}' using {max_w} workers...")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_w) as executor:
                for sp, res, err in executor.map(worker_blp, tasks):
                    if err is not None:
                        print(f"  [!] Spec {sp} failed:\\n{err}")
                    elif res is not None:
                        out_pkl = BLP_OUTPUT_DIR / f"blp_results_spec_6_{current_alt}_{sp}_{args.stage}.pkl"
                        with open(out_pkl, 'wb') as f:
                            pickle.dump(res, f)
                        print(f"  Saved: {out_pkl.name}")
                        all_results[sp] = {
                            'Q_value': float(res.get('Q_value', 0.0)),
                            'converged': bool(res.get('converged', True)),
                            'theta1_alpha': res.get('theta1', np.zeros(K_TYPES))[:K_TYPES].tolist(),
                            'theta2': res['theta2'].tolist() if 'theta2' in res and len(res['theta2']) > 0 else [],
                            'stage': args.stage,
                            'alt': current_alt
                        }

            # Summary JSON
            summary_path = BLP_OUTPUT_DIR / f"blp_summary_{current_alt}_6_{args.stage}.json"
            with open(summary_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            _log_status(f"Summary saved to: {summary_path}")
            _log_status(f"[DONE] BLP Estimation ({current_stage}) complete for alt {current_alt} specs {spec_ids}.")

    # --- Finalise ---
    _stop_email.set()
    _send_status_email(
        f"[BLP-HPC] Job {job_id} FINISHED â€” {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "\n".join(_email_log[-300:])
    )


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    main()


