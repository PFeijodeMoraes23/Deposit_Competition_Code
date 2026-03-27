## estimation_demand_1.py
# Author: Pedro Feijo de Moraes
# Last edited: 2026-03-06
#
# Objective: Berry (1994) demand estimation using ACTIVE market shares,
#            at the CONGLOMERATE level. Parallels estimation_1.py.
#
# For each of the 10 supply-side specifications from estimation_1.py:
#   1. Extract the structural (sleeping-deposit) prediction:
#         hat_Y_jkt = Upsilon' * phi(S,X) * nr * D_{jkt-1}
#      CF polynomial terms (v_hat, v_hat^2, v_hat^3) are EXCLUDED from the
#      structural prediction because they absorb endogeneity, not sleeping deposits.
#
#   2. Construct active deposits:
#         D_active_jkt = max(0, D_jkt - hat_Y_jkt)
#
#   3. Construct active market shares (national level, across ALL deposit types at t):
#         s_active_jkt = D_active_jkt / sum_{l in J_t, k'} D_active_lk't
#
#   4. Estimate demand (Berry 1994, eq. 11/13):
#         log(s_active_jkt) = alpha_k * sigma_jkt + delta_j + mu_kt + e_jkt
#      where:
#         alpha_k   = type-specific deposit-spread coefficient (spread_T1...spread_T5)
#         delta_j   = conglomerate x type entity FE  (entity = CodConglPrud_type)
#         mu_t      = plain quarter time FE  (time_idx = dense rank of AnoMes)
#         sigma_jkt = spread_qoq  (= risk_free_qoq - deposit_rate_qoq, opportunity cost >= 0)
#
# NOTE: the time FE is a PLAIN quarter FE (one dummy per AnoMes quarter),
#       NOT a deposit_type x AnoMes interaction.  Identification of the
#       type-specific alpha_k coefficients comes entirely from the
#       bank x type FE (entity) and cross-sectional spread variation,
#       NOT from a type-time interaction.
#
# Endogeneity:
#   Types 1-3: rates are regulated or market-determined -> exogenous. OLS only.
#   Types 4 (CDB) and 5 (prepaid): rates set by each conglomerate -> endogenous.
#   CF correction is applied to types 4-5 only; the CF residual is set to 0 for
#   types 1-3 so those observations are included but receive no CF adjustment.
#
# Identification mirrors supply side:
#   OLS specs 1-4  -> direct OLS (pooled / with two-way FE)
#   CF specs 5-8   -> CF: cost-only first stage on spread_qoq (types 4-5)
#   CF specs 9-10  -> CF: Hausman IV + costs first stage on spread_qoq (types 4-5)
#
# NOTE on FE identification (Frisch-Waugh):
#   Supply-side models 3, 4, 7, 8 are fit with entity+time FE absorbed via
#   linearmodels.PanelOLS (which internally demeans; NOT explicit pre-demeaning).
#   By the Frisch-Waugh theorem the within-estimator coefficients equal those
#   from a specification with full dummy variables; applying them to the
#   ORIGINAL (undemeaned) regressors recovers hat_Y_jkt correctly.
#
# Input:  BCB/Egan_et_al_2025_Rep/processed/PANEL_INTERMED/egan_panel_deposits.csv
#         (loaded via estimation_1.load_and_prepare())
# Output: BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/
#             demand_results_conglomerate.csv
#             demand_table_conglomerate.tex
## ---------------------------------------------------------------------------

## 1) Imports and paths:

import os
import sys
import warnings
import logging
from logging.handlers import RotatingFileHandler

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

# -- Paths ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BCB_PATH   = os.path.join(PARENT_DIR, "BCB")
EGAN_PATH  = os.path.join(BCB_PATH, "Egan_et_al_2025_Rep")
OUTPUT_DIR = os.path.join(EGAN_PATH, "processed", "ESTIMATION_OUTPUT")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Logging ----------------------------------------------------------------
_log_file = os.path.join(SCRIPT_DIR, "estimation_demand_1.log")
_fh = RotatingFileHandler(_log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
logging.basicConfig(
    handlers=[_fh, _ch],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -- Import supply estimation functions from estimation_1.py ----------------
sys.path.insert(0, SCRIPT_DIR)
import estimation_1_OLD as e1  # noqa: E402

# -- Constants --------------------------------------------------------------
# CF polynomial column names -- excluded from structural (sleeping) predictions
_CF_POLY = frozenset([
    "cf_resid", "cf_resid_sq", "cf_resid_cu",
    "cf_resid_h", "cf_resid_h_sq", "cf_resid_h_cu",
])

LEVEL_LABEL = "Conglomerate"
ENTITY_ID   = "CodConglPrud"  # primary entity column in estimation_1 panel


## 2) User-defined functions:

# 2.1) Helper functions:

def _stars(pval: float) -> str:
    if pval < 0.01:   return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.10: return "*"
    return ""

def _two_way_demean(df_in, dep, regs,
                    entity_col="entity", time_col="time_idx",
                    max_iter=200, tol=1e-8):
    """Iterative two-way within transformation (entity + time FE)."""
    cols = [dep] + regs
    Z = df_in[cols].copy().astype(float)
    for _ in range(max_iter):
        Z_prev = Z.copy()
        Z = Z.sub(df_in.groupby(entity_col)[cols].transform("mean"), axis=0)
        Z = Z.sub(df_in.groupby(time_col)[cols].transform("mean"), axis=0)
        if (Z - Z_prev).abs().max().max() < tol:
            break
    return Z[dep], Z[regs]

# 2.2) Active deposit construction:

def compute_sleeping_deposits(df_est: pd.DataFrame, results: dict) -> pd.DataFrame:
    """
    For each supply specification, compute the structural (sleeping-deposit)
    prediction:

        hat_Y_jkt = sum_k { params[k] * df_est[k] }

    where the sum is over STRUCTURAL regressors only (CF polynomial and
    intercept terms are excluded).

    For FE specs (3, 4, 7, 8):  model was fit on demeaned data, but by the
    Frisch-Waugh theorem the within-estimator coefficients are identical to
    those from a specification with full entity/time dummy variables.
    Applying them to the ORIGINAL (undemeaned) regressors recovers hat_Y_jkt.

    Returns df_est with new columns  'sleeping_{spec_name}'.
    """
    df_out = df_est.copy()
    for spec_name, d in results.items():
        if spec_name.startswith("_"):
            continue
        m      = d["model"]
        params = m.params

        # Structural regressors: exclude CF poly + constant
        struct_cols = [
            c for c in params.index
            if c not in _CF_POLY and c != "const"
        ]
        available = [c for c in struct_cols if c in df_out.columns]
        if not available:
            logging.warning(
                f"No structural columns found for supply spec '{spec_name}' -- skipping"
            )
            continue

        hat_Y = pd.Series(0.0, index=df_out.index)
        for c in available:
            hat_Y = hat_Y + params[c] * df_out[c]

        col = f"sleeping_{spec_name}"
        df_out[col] = hat_Y
        logging.info(
            f"  Supply spec '{spec_name}': "
            f"hat_Y computed for {hat_Y.notna().sum():,} obs "
            f"(struct cols = {available})"
        )
    return df_out

def compute_active_deposits(df: pd.DataFrame, spec_name: str) -> pd.Series:
    """
    D_active_jkt = max(0, deposit_balance - hat_Y_jkt)
    Returns a Series aligned with df.index.
    """
    col = f"sleeping_{spec_name}"
    if col not in df.columns:
        raise KeyError(
            f"Column '{col}' not found -- run compute_sleeping_deposits first."
        )
    return (df["deposit_balance"] - df[col]).clip(lower=0)

def compute_active_shares(df: pd.DataFrame, spec_name: str) -> pd.Series:
    """
    s_active_jkt = D_active_jkt / sum_{l in J_t, k'} D_active_lk't

    Market J_t = ALL institutions across ALL deposit types at quarter t
    (national-level market, shares defined over the aggregate deposit pool).
    Zeros stay in the denominator; observations with zero active deposits
    are excluded from the log in downstream estimation.
    """
    active = compute_active_deposits(df, spec_name)
    total  = active.groupby(df["AnoMes"]).transform("sum")
    return active / total.replace(0, np.nan)

# 2.3) Demand estimation:

def run_demand_estimation(df: pd.DataFrame, supply_results: dict) -> dict:
    """
    Estimate the Berry (1994) demand equation for each supply-side spec:

        log(s_active_jkt) = alpha_k * rho_jkt + delta_j + mu_kt + e_jkt

    Price sensitivity alpha_k is type-specific, implemented via type-interacted
    deposit rates (rate_T1 ... rate_T5).  delta_j is absorbed as entity FE and
    mu_t as a plain quarter time FE (time_idx = AnoMes rank) where applicable.

    Endogeneity:
      - Types 1-3: rates are regulated or market-determined -> exogenous. No CF.
      - Types 4 (CDB) and 5 (prepaid): rates set by each conglomerate -> endogenous.
        A control function is applied to types 4-5 only. The CF residual is set to
        zero for types 1-3 so those observations require no correction.

    Estimation strategy mirrors the supply side:
      - OLS specs  (no CF) -> plain OLS, HC-robust / clustered SE
      - CF-Cost specs      -> demand CF with cost-only FS on deposit_rate_qoq (types 4-5)
      - CF-Hausman specs   -> demand CF with Hausman IV + costs FS (types 4-5)

    Returns
    -------
    demand_results : dict  keyed by  'D' + supply_spec_name
    """
    demand_results = {}

    # -- Required columns -------------------------------------------------
    required = ["spread_qoq", "deposit_rate_qoq", "deposit_type", "AnoMes",
                "entity", "deposit_balance", "time_idx"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.error(f"Required columns missing: {missing} -- aborting demand estimation")
        return demand_results

    # -- Type-interacted deposit spreads (alpha_k identification) ----------
    # Use spread_qoq = deposit_rate_qoq - risk_free_qoq as the price variable.
    type_vals = sorted(df["deposit_type"].dropna().unique().tolist())
    for t in type_vals:
        df[f"spread_T{int(t)}"] = np.where(
            df["deposit_type"] == t, df["spread_qoq"], 0.0
        )
    rate_cols = [f"spread_T{int(t)}" for t in type_vals]
    logging.info(f"Demand -- type-interacted spread columns: {rate_cols}")

    # -- Endogenous deposit types requiring CF correction ----------------
    # Types 1-3: regulated or market-determined rates -> exogenous, no CF needed.
    # Types 4 (CDB) and 5 (prepaid): conglomerate-set rates -> endogenous.
    # CF residuals are set to 0 for types 1-3 so no correction is applied there.
    ENDO_TYPES = [4, 5]
    df_endo = df[df["deposit_type"].isin(ENDO_TYPES)].copy()
    logging.info(
        f"Endogenous types {ENDO_TYPES}: {len(df_endo):,} obs "
        f"({len(df_endo) / len(df) * 100:.1f}% of full sample)"
    )

    # -- Demand CF first-stage cost shifters (types 4-5 only) -------------
    candidate_cost = [
        "cosif_lag", "log_assets_std", "equity_ratio",
        "lagged_cdi", "lagged_selic",
        "personnel_cost_ratio_lag", "admin_cost_ratio_lag",
    ]
    fs_cost = [
        v for v in candidate_cost
        if v in df_endo.columns and df_endo[v].notna().sum() > 100
    ]
    logging.info(f"Demand CF cost shifters (types 4-5): {fs_cost}")

    # Single type dummy in FS: type 5 vs type 4 (baseline)
    # Create on full df so prediction step can access it via df.loc[valid_a, all_fs_d]
    df["_type5_d"]    = (df["deposit_type"] == 5).astype(float)
    df_endo["_type5_d"] = (df_endo["deposit_type"] == 5).astype(float)
    all_fs_d = fs_cost + ["_type5_d"]

    # -- Demand CF First Stage A: cost-only, types 4-5 --------------------
    # Residuals default to 0.0 for types 1-3 (no endogeneity correction).
    # First stage instruments the spread (spread_qoq) with cost shifters.
    d_cf_resid_45 = pd.Series(0.0, index=df.index)
    if fs_cost:
        fs_cols_a = all_fs_d + ["spread_qoq"]
        df_fsa    = df_endo.dropna(subset=fs_cols_a).copy()
        X_fsa     = sm.add_constant(df_fsa[all_fs_d])
        m_fsa     = sm.OLS(df_fsa["spread_qoq"], X_fsa).fit(cov_type="HC1")
        logging.info(
            f"Demand CF-A (types 4-5): N={int(m_fsa.nobs):,}, R^2={m_fsa.rsquared:.4f}"
        )
        try:
            f_cost = float(m_fsa.f_test([f"{v} = 0" for v in fs_cost]).fvalue)
            logging.info(f"  Joint F (cost vars): {f_cost:.2f} "
                         f"({'strong' if f_cost > 10 else 'weak'})")
        except Exception:
            pass
        valid_a = df_endo[fs_cols_a].dropna().index
        pred_a  = pd.Series(
            m_fsa.predict(sm.add_constant(df.loc[valid_a, all_fs_d])),
            index=valid_a,
        )
        d_cf_resid_45.loc[valid_a] = df.loc[valid_a, "spread_qoq"] - pred_a
    df["d_cf_resid_45"]    = d_cf_resid_45
    df["d_cf_resid_45_sq"] = df["d_cf_resid_45"] ** 2
    df["d_cf_resid_45_cu"] = df["d_cf_resid_45"] ** 3

    # -- Demand CF First Stage B: + Hausman IV (leave-one-out mean spread, types 4-5) --
    # IV is the leave-one-out mean spread (spread_qoq) for the same type x quarter.
    # IV computed within endogenous types only (not contaminated by exogenous types).
    grp_e = df_endo.groupby(["deposit_type", "AnoMes"])
    sum_r = grp_e["spread_qoq"].transform("sum")
    cnt_r = grp_e["spread_qoq"].transform("count")
    df_endo["_haus_iv_spread"] = np.where(
        cnt_r > 1, (sum_r - df_endo["spread_qoq"]) / (cnt_r - 1), np.nan
    )
    # Expose on full df (NaN for exogenous types 1-3) so prediction works
    df["_haus_iv_spread"] = np.nan
    df.loc[df_endo.index, "_haus_iv_spread"] = df_endo["_haus_iv_spread"]
    all_fs_h = ["_haus_iv_spread"] + all_fs_d
    d_cf_resid_45_h = pd.Series(0.0, index=df.index)
    if fs_cost:
        fs_cols_h = all_fs_h + ["spread_qoq"]
        df_fsh    = df_endo.dropna(subset=fs_cols_h).copy()
        X_fsh     = sm.add_constant(df_fsh[all_fs_h])
        m_fsh     = sm.OLS(df_fsh["spread_qoq"], X_fsh).fit(cov_type="HC1")
        logging.info(
            f"Demand CF-B (types 4-5, +Hausman IV): N={int(m_fsh.nobs):,}, R^2={m_fsh.rsquared:.4f}"
        )
        try:
            f_h = float(m_fsh.f_test("_haus_iv_spread = 0").fvalue)
            logging.info(f"  F (Hausman IV): {f_h:.2f} "
                         f"({'strong' if f_h > 10 else 'weak'})")
        except Exception:
            pass
        valid_h = df_endo[fs_cols_h].dropna().index
        pred_h  = pd.Series(
            m_fsh.predict(sm.add_constant(df.loc[valid_h, all_fs_h])),
            index=valid_h,
        )
        d_cf_resid_45_h.loc[valid_h] = df.loc[valid_h, "spread_qoq"] - pred_h
    df["d_cf_resid_45_h"]    = d_cf_resid_45_h
    df["d_cf_resid_45_h_sq"] = df["d_cf_resid_45_h"] ** 2
    df["d_cf_resid_45_h_cu"] = df["d_cf_resid_45_h"] ** 3

    d_poly_a = ["d_cf_resid_45", "d_cf_resid_45_sq", "d_cf_resid_45_cu"]
    d_poly_h = ["d_cf_resid_45_h", "d_cf_resid_45_h_sq", "d_cf_resid_45_h_cu"]

    # -- Iterate over supply specs -----------------------------------------
    supply_specs = [k for k in supply_results if not k.startswith("_")]
    for spec_name in supply_specs:
        sleep_col = f"sleeping_{spec_name}"
        if sleep_col not in df.columns:
            logging.warning(
                f"Sleeping column '{sleep_col}' not found -- skipping '{spec_name}'"
            )
            continue

        # Compute active deposits and national active shares
        # Denominator: total active deposits across ALL types at quarter t
        active   = compute_active_deposits(df, spec_name)
        total_m  = active.groupby(df["AnoMes"]).transform("sum")
        s_active = (active / total_m.replace(0, np.nan))

        # Keep only valid demand observations (log well-defined + spread available)
        valid_mask = (
            (active > 0)
            & s_active.notna()
            & df["spread_qoq"].notna()
        )
        df_d = df[valid_mask].copy()
        df_d["log_s_active"] = np.log(s_active[valid_mask])

        n_total = len(df)
        n_zero  = int((active <= 0).sum())
        n_valid = len(df_d)
        logging.info(
            f"\n=== Demand: supply spec '{spec_name}' "
            f"| N_valid={n_valid:,}, N_zero_active={n_zero:,} / {n_total:,} ==="
        )
        if n_valid < 100:
            logging.warning(f"  Insufficient observations ({n_valid}) -- skipping")
            continue
        logging.info(
            f"  log(s_active): mean={df_d['log_s_active'].mean():.4f}, "
            f"sd={df_d['log_s_active'].std():.4f}"
        )

        # Classify the supply spec type
        is_fe    = supply_results[spec_name].get("entity_fe", "No") == "Yes"
        is_clust = "Clust" in spec_name or "clust" in spec_name
        is_cf    = "CF" in spec_name or "cf" in spec_name.lower()
        is_haus  = "Haus" in spec_name or "haus" in spec_name.lower()
        d_poly   = d_poly_h if is_haus else d_poly_a

        rc = [c for c in rate_cols if c in df_d.columns]
        demand_key = f"D{spec_name}"

        def _log_coeffs(m, label):
            for c in rc:
                if c in m.params.index:
                    logging.info(
                        f"  {label} {c}: coef={m.params[c]:.6f} "
                        f"(SE={m.bse[c]:.6f}, p={m.pvalues[c]:.4f})"
                        f"{_stars(m.pvalues[c])}"
                    )

        try:
            # -- OLS, no FE -----------------------------------------------
            if not is_fe and not is_cf:
                df_s = df_d.dropna(subset=rc + ["log_s_active"]).copy()
                X    = df_s[rc]
                m    = sm.OLS(df_s["log_s_active"], X).fit(cov_type="HC1")
                demand_results[demand_key] = {
                    "model": m, "type": "sm",
                    "entity_fe": "No", "time_fe": "No",
                    "se_type": "HC-robust",
                    "supply_spec": spec_name,
                    "n_active_zeros": n_zero,
                }
                _log_coeffs(m, demand_key)

            # -- OLS + two-way FE -----------------------------------------
            elif is_fe and not is_cf:
                needed = rc + ["log_s_active", "entity", "time_idx"]
                df_s   = df_d.dropna(subset=needed).copy()
                if len(df_s) < 200:
                    logging.warning(
                        f"  {demand_key}: FE demand sample too small ({len(df_s)}) -- skipping"
                    )
                    continue
                y_w, X_w = _two_way_demean(df_s, "log_s_active", rc)
                cov_t  = "cluster" if is_clust else "HC1"
                cov_kw = (
                    {"groups": df_s.loc[X_w.index, "entity"]} if is_clust else {}
                )
                m = sm.OLS(y_w, X_w).fit(
                    cov_type=cov_t,
                    cov_kwds=cov_kw,
                )
                demand_results[demand_key] = {
                    "model": m, "type": "sm",
                    "entity_fe": "Yes", "time_fe": "Yes",
                    "se_type": "Cong-clust" if is_clust else "HC-robust",
                    "supply_spec": spec_name,
                    "n_active_zeros": n_zero,
                }
                _log_coeffs(m, demand_key)

            # -- CF, no FE ------------------------------------------------
            elif not is_fe and is_cf:
                dp   = [c for c in d_poly if c in df_d.columns]
                regs = rc + dp
                df_s = df_d.dropna(subset=regs + ["log_s_active"]).copy()
                X    = df_s[regs]
                m    = sm.OLS(df_s["log_s_active"], X).fit(cov_type="HC1")
                demand_results[demand_key] = {
                    "model": m, "type": "sm",
                    "entity_fe": "No", "time_fe": "No",
                    "se_type": "HC-robust",
                    "supply_spec": spec_name,
                    "n_active_zeros": n_zero,
                }
                _log_coeffs(m, demand_key)

            # -- CF + two-way FE ------------------------------------------
            else:
                dp     = [c for c in d_poly if c in df_d.columns]
                regs   = rc + dp
                needed = regs + ["log_s_active", "entity", "time_idx"]
                df_s   = df_d.dropna(subset=needed).copy()
                if len(df_s) < 200:
                    logging.warning(
                        f"  {demand_key}: CF+FE demand sample too small ({len(df_s)}) -- skipping"
                    )
                    continue
                y_w, X_w = _two_way_demean(df_s, "log_s_active", regs)
                cov_t  = "cluster" if is_clust else "HC1"
                cov_kw = (
                    {"groups": df_s.loc[X_w.index, "entity"]} if is_clust else {}
                )
                m = sm.OLS(y_w, X_w).fit(
                    cov_type=cov_t,
                    cov_kwds=cov_kw,
                )
                demand_results[demand_key] = {
                    "model": m, "type": "sm",
                    "entity_fe": "Yes", "time_fe": "Yes",
                    "se_type": "Cong-clust" if is_clust else "HC-robust",
                    "supply_spec": spec_name,
                    "n_active_zeros": n_zero,
                }
                _log_coeffs(m, demand_key)

        except Exception as exc:
            logging.warning(f"  {demand_key}: estimation failed -- {exc}")
            continue

    return demand_results

# 2.4) Export results:

def export_demand_results(
    demand_results: dict,
    output_dir: str,
    suffix: str = "",
) -> pd.DataFrame:
    """Export demand coefficient estimates to CSV."""
    rows = []
    for spec_name, d in demand_results.items():
        m      = d["model"]
        params = m.params
        ses    = m.bse
        pvals  = m.pvalues
        nobs   = int(m.nobs)
        r2     = float(m.rsquared)
        for var in params.index:
            se = ses[var]
            rows.append({
                "spec":           spec_name,
                "supply_spec":    d.get("supply_spec", ""),
                "variable":       var,
                "coef":           params[var],
                "se":             se,
                "pvalue":         pvals[var],
                "ci_lower":       params[var] - 1.96 * se,
                "ci_upper":       params[var] + 1.96 * se,
                "n":              nobs,
                "r2":             r2,
                "entity_fe":      d.get("entity_fe", ""),
                "time_fe":        d.get("time_fe", ""),
                "se_type":        d.get("se_type", ""),
                "n_active_zeros": d.get("n_active_zeros", ""),
            })

    df_out = pd.DataFrame(rows)
    fname  = f"demand_results_conglomerate{suffix}.csv"
    path   = os.path.join(output_dir, fname)
    df_out.to_csv(path, index=False)
    logging.info(f"Demand results exported to {path}")
    return df_out

def export_demand_latex(
    demand_results: dict,
    output_dir: str,
    caption: str = "Demand Estimation: Berry (1994) Active Market Shares (Conglomerate Level)",
    label: str   = "tab:demand_conglom",
    filename: str = "demand_table_conglomerate.tex",
) -> str:
    r"""
    Export a LaTeX table of demand (alpha_k) estimates.
    Rows = rate_Tk coefficients; columns = demand specs.
    """
    spec_names = [k for k in demand_results if not k.startswith("_")]
    if not spec_names:
        return ""

    all_vars = set()
    for sn in spec_names:
        all_vars.update(demand_results[sn]["model"].params.index)

    # Spread columns first, then CF poly, then rest
    rate_v = sorted([v for v in all_vars if v.startswith("spread_T")])
    cf_v   = [v for v in [
        "d_cf_resid_45", "d_cf_resid_45_sq", "d_cf_resid_45_cu",
        "d_cf_resid_45_h", "d_cf_resid_45_h_sq", "d_cf_resid_45_h_cu",
    ] if v in all_vars]
    other_v = sorted([v for v in all_vars if v not in rate_v + cf_v and v != "const"])
    var_list = rate_v + cf_v + other_v

    friendly = {
        "spread_T1": r"$\hat{\alpha}_1$ (Type 1 spread, exog.)",
        "spread_T2": r"$\hat{\alpha}_2$ (Type 2 spread, exog.)",
        "spread_T3": r"$\hat{\alpha}_3$ (Type 3 spread, exog.)",
        "spread_T4": r"$\hat{\alpha}_4$ (CDB spread, endog.)",
        "spread_T5": r"$\hat{\alpha}_5$ (Prepaid spread, endog.)",
        "d_cf_resid_45":    r"$\hat{u}_{45}$ (CF types 4-5, 1st)",
        "d_cf_resid_45_sq": r"$\hat{u}_{45}^2$ (2nd)",
        "d_cf_resid_45_cu": r"$\hat{u}_{45}^3$ (3rd)",
        "d_cf_resid_45_h":    r"$\hat{u}_{45}^H$ (Hausman CF types 4-5, 1st)",
        "d_cf_resid_45_h_sq": r"$(\hat{u}_{45}^H)^2$ (2nd)",
        "d_cf_resid_45_h_cu": r"$(\hat{u}_{45}^H)^3$ (3rd)",
    }

    def _tex_stars(p):
        if p < 0.01: return "^{***}"
        if p < 0.05: return "^{**}"
        if p < 0.10: return "^{*}"
        return ""

    def _fmt_val(x):
        """Format a number: 2 dp when |x| >= 100, else 3 dp."""
        return f"{x:.2f}" if abs(x) >= 100 else f"{x:.3f}"

    n_s   = len(spec_names)
    col_s = "l" + "c" * n_s
    L     = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(r"\begin{threeparttable}")
    L.append(r"\caption{" + caption + "}")
    L.append(r"\label{" + label + "}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{" + col_s + "}")
    L.append(r"\toprule")
    L.append(
        r" & \multicolumn{" + str(n_s) + r"}{c}{"
        r"$\log s^{\mathrm{Active}}_{jkt}$} \\"
    )
    L.append(r"\cmidrule(lr){2-" + str(n_s + 1) + "}")
    hdr = "".join(f" & {sn}" for sn in spec_names) + r" \\"
    L.append(hdr)
    L.append(r"\midrule")

    for var in var_list:
        lbl       = friendly.get(var, var.replace("_", r"\_"))
        coef_row  = ""
        se_row    = ""
        for sn in spec_names:
            m      = demand_results[sn]["model"]
            params = m.params
            if var in params.index:
                se  = m.bse[var]
                p   = m.pvalues[var]
                coef_row += f" & ${_fmt_val(params[var])}{_tex_stars(p)}$"
                se_row   += f" & $({_fmt_val(se)})$"
            else:
                coef_row += " & "
                se_row   += " & "
        L.append(f"{lbl}{coef_row}" + r" \\")
        L.append(f"{se_row}" + r" \\[3pt]")

    L.append(r"\midrule")
    obs_row = "Observations"
    for sn in spec_names:
        obs_row += f" & {int(demand_results[sn]['model'].nobs):,}"
    L.append(obs_row + r" \\")
    r2_row = "$R^2$"
    for sn in spec_names:
        r2_row += f" & {demand_results[sn]['model'].rsquared:.3f}"
    L.append(r2_row + r" \\")
    for lbl_t, key in [
        (r"Entity$\times$Type FE", "entity_fe"),
        ("Time (Type$\\times t$) FE", "time_fe"),
    ]:
        row = lbl_t
        for sn in spec_names:
            row += " & " + demand_results[sn].get(key, "")
        L.append(row + r" \\")
    se_r = "SE type"
    for sn in spec_names:
        se_r += " & " + demand_results[sn].get("se_type", "")
    L.append(se_r + r" \\")
    sup_r = "Supply spec"
    for sn in spec_names:
        sup_r += " & " + demand_results[sn].get("supply_spec", "")
    L.append(sup_r + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\begin{tablenotes}")
    L.append(r"\small")
    L.append(
        r"\item \textit{Notes:} $^{***}p<0.01$; $^{**}p<0.05$; $^{*}p<0.1$. "
        r"Standard errors in parentheses. "
        r"The dependent variable is $\log s^{\mathrm{Active}}_{jkt}$ "
        r"(log active market share over the total deposit pool at quarter $t$, all types). "
        r"Regressors are type-interacted deposit spreads (\texttt{spread\_qoq} "
        r"$= R^F_t - r_{jkt}$, the opportunity cost). "
        r"For each supply-side spec the sleeping-deposit prediction $\hat{Y}_{jkt}$ "
        r"is subtracted from observed deposits and shares are recomputed. "
        r"CF corrections (cost shifters / Hausman IV on leave-one-out mean spread) "
        r"applied to endogenous Types 4--5 only; "
        r"Types 1--3 spreads are exogenous and require no correction. "
        r"Panel unit: conglomerate (CodConglPrud) $\times$ deposit type."
    )
    L.append(r"\end{tablenotes}")
    L.append(r"\end{threeparttable}")
    L.append(r"\end{table}")

    tex_path = os.path.join(output_dir, filename)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    logging.info(f"Demand LaTeX table exported to {tex_path}")
    return tex_path

## 3) Main execution:

if __name__ == "__main__":
    logging.info("=" * 80)
    logging.info("DEMAND ESTIMATION -- CONGLOMERATE LEVEL")
    logging.info("Berry (1994) active market shares | estimation_demand_1.py")
    logging.info("=" * 80)

    # 3.1) Load panel and run supply estimation -----------------------
    logging.info("\n>>> Step 1: Loading panel and running supply estimation ...")
    df_est = e1.load_and_prepare()

    results_ols = e1.run_structural_estimation(df_est)
    results_cf  = e1.run_control_function_estimation(df_est)
    # NOTE: run_control_function_estimation adds cf_resid* and hausman_iv_spread
    # columns directly to df_est (pandas pass-by-reference).

    results_all = {
        **results_ols,
        **{k: v for k, v in results_cf.items() if not k.startswith("_")},
    }
    logging.info(
        f"Supply specs available: "
        f"{[k for k in results_all if not k.startswith('_')]}"
    )

    # 3.2) Compute structural sleeping-deposit predictions ------------
    logging.info("\n>>> Step 2: Computing structural sleeping-deposit predictions ...")
    df_est = compute_sleeping_deposits(df_est, results_all)

    # 3.3) Active deposit diagnostics --------------------------------
    logging.info("\n>>> Step 3: Active deposit diagnostics by supply spec ...")
    for sn in [k for k in results_all if not k.startswith("_")]:
        if f"sleeping_{sn}" not in df_est.columns:
            continue
        active   = compute_active_deposits(df_est, sn)
        pct_pos  = (active > 0).mean() * 100
        pct_zero = (active <= 0).mean() * 100
        mean_pos = active[active > 0].mean()
        logging.info(
            f"  '{sn}': active>0={pct_pos:.1f}%, "
            f"active=0={pct_zero:.1f}%, "
            f"mean(active|active>0)={mean_pos:.1f}"
        )

    # 3.4) Demand estimation ------------------------------------------
    logging.info("\n>>> Step 4: Estimating Berry (1994) demand equation ...")
    demand_results = run_demand_estimation(df_est, results_all)

    # 3.5) Export -----------------------------------------------------
    logging.info("\n>>> Step 5: Exporting demand results ...")
    if demand_results:
        export_demand_results(demand_results, OUTPUT_DIR)
        export_demand_latex(demand_results, OUTPUT_DIR)
        logging.info(
            f"Demand specs estimated: {list(demand_results.keys())}"
        )
    else:
        logging.warning("No demand results produced.")

    logging.info("Done -- estimation_demand_1.py complete.")
