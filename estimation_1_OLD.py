## estimation_1.py
# Author: Pedro Feijo de Moraes
# Last edited: 2026-02-24
#
# Objective: Estimate Upsilon (sleepiness) parameters of the Egan et al.
#            (2025) deposit demand model at the CONGLOMERATE level.
#
# This is the conglomerate-level counterpart of estimation_2.py.
# The structural model is identical; the entity is:
#       entity = CodConglPrud x deposit_type  (~200 conglomerates)
#
# Structural equation (Egan et al. 2025, Sections 3.3-3.4):
#
#   Dep_jkt = phi(S_t, X_jt) * nr_t * Dep_jkt-1 + epsilon_jkt   [estimated in LEVELS]
#
#   where phi(S_t, X_jt) = Upsilon'_1 S_t + Upsilon'_2 X_jt
#         nr_t = 1 + (R^F_{t-1} - rho_{jkt-1}) / 100
#
# Identification:
#   Types 1-3 have exogenous rates (regulated or market-determined).
#   Types 4 (CDB) and 5 (prepaid) have conglomerate-specific endogenous
#   rates that are a function of banking competition.  We use a
#   CONTROL FUNCTION approach (Petrin & Train, 2010):
#
#     Stage 1: Regress endogenous deposit_rate on instruments + exogenous X
#              rho_jkt = Z'_jkt gamma + X'_jkt delta + v_jkt
#     Stage 2: Include first-stage residuals v^ in the structural equation
#              deposit_balance = phi(S,X)*nr*D_{t-1} + lambda_*v^ + eps
#
#   Instruments / cost shifters in first stage:
#     - Cost shifters: lagged COSIF implicit rate, log assets, equity ratio,
#                      lagged CDI, lagged Selic, deposit-type dummies
#     - Hausman IV:   leave-one-out mean deposit SPREAD (same type x quarter)
#       ("lagged own deposit rate" is NOT used as an instrument in the code)
#
# Specifications:
#   OLS Panel (without control function):
#     (1) Pooled national-level (Y_1 only), HC-robust SE
#     (2) + Bank characteristics (Y_1 + Y_2), HC-robust SE
#     (3) + BankxType FE + Time FE (robust SE)
#     (4) Same as (3), bank-clustered SE
#
#   Control Function -- Cost-shifters first stage (specs 5-8):
#     (5) CF-Natl:  national-level + poly(v^), HC-robust SE
#     (6) CF-Bank:  + bank chars + segment + poly(v^), HC-robust SE
#     (7) CF+FE:    + entityxtype FE + time FE + poly(v^), HC-robust SE
#     (8) CF+Clust: same as (7), bank-clustered SE
#
#   Control Function -- Hausman IV first stage (specs 9-10):
#     (9)  CF-Haus-Natl: national-level + poly(v^_H), HC-robust SE
#     (10) CF-Haus-Bank: + bank chars + segment + poly(v^_H), HC-robust SE
#
#   A significant lambda_ on v^ (joint F on cubic polynomial) indicates OLS bias.
#
# Input:  BCB/Egan_et_al_2025_Rep/processed/PANEL_INTERMED/egan_panel_deposits.csv
# Output: BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/
#             estimation_results.csv
#             estimation_table.tex
## ---------------------------------------------------------------------------

## 1) Load necessary packages and set paths:

# Packages:
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from logging.handlers import RotatingFileHandler

# Paths:
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BCB_PATH   = os.path.join(PARENT_DIR, "BCB")
EGAN_PATH  = os.path.join(BCB_PATH, "Egan_et_al_2025_Rep")
PANEL_PATH = os.path.join(EGAN_PATH, "processed", "PANEL_INTERMED", "egan_panel_deposits.csv")
OUTPUT_DIR = os.path.join(EGAN_PATH, "processed", "ESTIMATION_OUTPUT")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging
_log_file = os.path.join(SCRIPT_DIR, "estimation_1.log")
_file_handler = RotatingFileHandler(
    _log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
logging.basicConfig(
    handlers=[_file_handler, _console_handler],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

## 2) User-defined functions:

# 2.1) Loading and preparing panel:

def load_and_prepare():
    """
    Load the panel produced by egan_panel_build.py and construct the
    structural regressors for estimation.

    The panel already contains:
      - deposit_balance, lagged_deposits, risk_free_qoq, deposit_rate_qoq
      - total_assets, equity, equity_ratio, log_total_assets
      - segment, seg_S2 ... seg_S5
      - meta_selic

    Returns estimation DataFrame with columns:
      deposit_balance, nr_lag, selic_nr_lag, logA_nr_lag, eq_nr_lag,
      seg_S2_nr_lag ... seg_S5_nr_lag, entity, time_idx, CodConglPrud, AnoMes
    """
    # --- Load panel ---
    logging.info("Loading panel ...")
    df = pd.read_csv(PANEL_PATH, low_memory=False, dtype={"CodConglPrud": str})
    logging.info(f"Raw panel: {len(df):,} rows, {df.shape[1]} cols")

    # Log coverage of bank characteristics (already in panel from egan_panel_build)
    for col in ["total_assets", "equity_ratio", "segment"]:
        if col in df.columns:
            n_ok = df[col].notna().sum()
            logging.info(f"  {col} coverage: {n_ok:,} / {len(df):,}")

    # --- Basic cleaning ---
    df = df.dropna(subset=["deposit_balance", "lagged_deposits",
                        "risk_free_qoq", "deposit_rate_qoq"])
    df = df[df["lagged_deposits"] > 0].copy()

    # --- Net return ---
    # nr = 1 + (R^F - rho) / 100
    df = df.sort_values(["CodConglPrud", "deposit_type", "AnoMes"])
    df["lagged_rf"] = (
        df.groupby(["CodConglPrud", "deposit_type"])["risk_free_qoq"].shift(1)
    )
    df["lagged_dep_rate"] = (
        df.groupby(["CodConglPrud", "deposit_type"])["deposit_rate_qoq"].shift(1)
    )
    df["nr"] = 1 + (df["lagged_rf"] - df["lagged_dep_rate"]) / 100

    # --- Lagged Selic level (annualized target, as in Egan's lagged FFR) ---
    df["lagged_selic"] = (
        df.groupby(["CodConglPrud", "deposit_type"])["meta_selic"].shift(1)
    )

    # --- Standardize continuous bank characteristics ---
    if "log_total_assets" in df.columns:
        mu_a = df["log_total_assets"].mean()
        sd_a = df["log_total_assets"].std()
        df["log_assets_std"] = (df["log_total_assets"] - mu_a) / sd_a
        logging.info(f"log_total_assets: mean={mu_a:.2f}, sd={sd_a:.2f} "
                    f"(standardized for estimation)")
    else:
        # Fallback: use total_deposits from panel
        df["log_assets_std"] = (
            np.log(df["total_deposits"].clip(lower=1))
        )
        mu = df["log_assets_std"].mean()
        sd = df["log_assets_std"].std()
        df["log_assets_std"] = (df["log_assets_std"] - mu) / sd
        logging.info(f"Fallback: using log(total_deposits) as size proxy")

    # --- Structural regressors: characteristic x nr x D_{jkt-1} ---
    # Implements D_jkt = phi(S,X) * nr * D_jkt-1 + eps directly in levels.
    df["nr_lag"]       = df["nr"] * df["lagged_deposits"]
    df["selic_nr_lag"] = df["lagged_selic"] * df["nr"] * df["lagged_deposits"]
    df["logA_nr_lag"]  = df["log_assets_std"] * df["nr"] * df["lagged_deposits"]

    if "equity_ratio" in df.columns:
        df["eq_nr_lag"] = df["equity_ratio"] * df["nr"] * df["lagged_deposits"]

    for s in ["S2", "S3", "S4", "S5"]:
        col = f"seg_{s}"
        if col in df.columns:
            df[f"{col}_nr_lag"] = df[col] * df["nr"] * df["lagged_deposits"]

    # --- Panel indices ---
    df["entity"] = (
        df["CodConglPrud"].astype(str) + "_" + df["deposit_type"].astype(str)
    )
    df["time_idx"] = df["AnoMes"].rank(method="dense").astype(int)

    # --- Build estimation sample ---
    core_vars = ["deposit_balance", "lagged_deposits",
                 "nr_lag", "selic_nr_lag", "logA_nr_lag",
                 "entity", "CodConglPrud", "deposit_type", "AnoMes", "time_idx"]
    extra_vars = []
    if "eq_nr_lag" in df.columns:
        extra_vars += ["eq_nr_lag", "equity_ratio"]
    for s in ["S2", "S3", "S4", "S5"]:
        col = f"seg_{s}_nr_lag"
        if col in df.columns:
            extra_vars += [col, f"seg_{s}"]
    if "segment" in df.columns:
        extra_vars.append("segment")
    extra_vars.append("log_assets_std")
    extra_vars.append("lagged_selic")

    # --- Control function variables ---
    # For the new spread-based CF first stage we need:
    #   spread_qoq  : outcome in first stage
    #   cosif_lag   : lagged COSIF implicit funding cost (cost shifter)
    #   lagged_cdi  : lagged CDI benchmark (macro shifter)
    cf_vars = ["deposit_rate_qoq", "spread_qoq"]
    if "cosif_implicit_rate" in df.columns:
        df["cosif_lag"] = (
            df.groupby(["CodConglPrud", "deposit_type"])["cosif_implicit_rate"].shift(1)
        )
        cf_vars.append("cosif_lag")
    if "cdi_qoq" in df.columns:
        df["lagged_cdi"] = (
            df.groupby(["CodConglPrud", "deposit_type"])["cdi_qoq"].shift(1)
        )
        cf_vars.append("lagged_cdi")

    # DRE labour-cost ratios are already lagged in the panel
    for dre_col in ["personnel_cost_ratio_lag", "admin_cost_ratio_lag"]:
        if dre_col in df.columns:
            cf_vars.append(dre_col)

    all_vars = core_vars + [v for v in extra_vars if v in df.columns] \
                         + [v for v in cf_vars if v in df.columns]
    df_est = df[all_vars].dropna(
        subset=["deposit_balance", "nr_lag", "selic_nr_lag", "logA_nr_lag"]
    ).copy()

    # --- Winsorize deposit_balance (trim extreme outliers) ---
    p01, p99 = df_est["deposit_balance"].quantile([0.01, 0.99])
    n_before = len(df_est)
    df_est = df_est[(df_est["deposit_balance"] >= p01) & (df_est["deposit_balance"] <= p99)]
    logging.info(f"Winsorized deposit_balance at [p1,p99]=[{p01:.1f},{p99:.1f}]: "
                f"dropped {n_before - len(df_est):,} outliers")

    # --- Diagnostics ---
    logging.info(f"Estimation sample: {len(df_est):,} rows")
    logging.info(f"  Banks:    {df_est['CodConglPrud'].nunique()}")
    logging.info(f"  Quarters: {df_est['AnoMes'].nunique()}")
    logging.info(f"  Types:    {sorted(df_est['deposit_type'].unique())}")
    logging.info(f"Dep var (deposit_balance, levels) summary:")
    logging.info(f"  mean={df_est['deposit_balance'].mean():.1f}, "
                f"sd={df_est['deposit_balance'].std():.1f}, "
                f"p5={df_est['deposit_balance'].quantile(0.05):.1f}, "
                f"p95={df_est['deposit_balance'].quantile(0.95):.1f}")
    logging.info(f"nr_lag (nr x D_{{t-1}}) summary:")
    logging.info(f"  mean={df_est['nr_lag'].mean():.6f}, "
                f"sd={df_est['nr_lag'].std():.6f}")
    logging.info(f"Lagged Selic (p.a.) summary:")
    s = df_est['lagged_selic'].dropna()
    logging.info(f"  mean={s.mean():.2f}, sd={s.std():.2f}, "
                f"min={s.min():.2f}, max={s.max():.2f}")
    if "segment" in df_est.columns:
        logging.info(f"Segment distribution: "
                    f"{df_est['segment'].value_counts().to_dict()}")

    return df_est

# 2.3) Estimation functions:

def _stars_txt(pval):
    """Return plain-text significance stars."""
    if pval < 0.01:   return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.10: return "*"
    return ""


def _stars_tex(pval):
    """Return LaTeX significance stars."""
    if pval < 0.01:   return "^{***}"
    elif pval < 0.05: return "^{**}"
    elif pval < 0.10: return "^{*}"
    return ""


def _fmt_val(x):
    """Format a number: 2 dp when |x| >= 100, else 3 dp."""
    return f"{x:.2f}" if abs(x) >= 100 else f"{x:.3f}"


def format_coef(coef, se, pval):
    """Format coefficient with significance stars."""
    return coef, se, _stars_txt(pval)


def print_table(results, dep_label="Dep_jkt / Dep_jkt-1"):
    """Print an Egan-style regression table."""
    spec_names = [k for k in results.keys() if not k.startswith("_")]
    all_vars = set()
    for sn in spec_names:
        all_vars.update(results[sn]["model"].params.index)
    # Order: nr_lag first, then selic_nr_lag, then bank chars, then segment, then CF
    order = ["nr_lag", "selic_nr_lag", "logA_nr_lag", "eq_nr_lag",
            "seg_S2_nr_lag", "seg_S3_nr_lag", "seg_S4_nr_lag", "seg_S5_nr_lag",
            "cf_resid", "cf_resid_sq", "cf_resid_cu",
            "cf_resid_h", "cf_resid_h_sq", "cf_resid_h_cu"]
    var_list = [v for v in order if v in all_vars]
    # Add any remaining (e.g., const)
    for v in sorted(all_vars):
        if v not in var_list:
            var_list.append(v)

    # Friendly names for display
    friendly = {
        "nr_lag":          "Y_1,cons  (nr x D_{t-1})",
        "selic_nr_lag":    "Y_1,RF   (selic x nr x D_{t-1})",
        "logA_nr_lag":     "Y_2,size (logA x nr x D_{t-1})",
        "eq_nr_lag":       "Y_2,eq   (E/A x nr x D_{t-1})",
        "seg_S2_nr_lag":   "Y_2,S2  (seg_S2 x nr x D_{t-1})",
        "seg_S3_nr_lag":   "Y_2,S3  (seg_S3 x nr x D_{t-1})",
        "seg_S4_nr_lag":   "Y_2,S4  (seg_S4 x nr x D_{t-1})",
        "seg_S5_nr_lag":   "Y_2,S5  (seg_S5 x nr x D_{t-1})",
        "cf_resid":        "beta_1 v^  (CF-cost, 1st)",
        "cf_resid_sq":     "beta_2 v^^2 (CF-cost, 2nd)",
        "cf_resid_cu":     "beta_3 v^^3 (CF-cost, 3rd)",
        "cf_resid_h":      "beta_1 v^H (CF-Hausman, 1st)",
        "cf_resid_h_sq":   "beta_2 v^H^2 (CF-Hausman, 2nd)",
        "cf_resid_h_cu":   "beta_3 v^H^3 (CF-Hausman, 3rd)",
    }

    col_w = 16
    name_w = 30
    logging.info(f"{'Dep var: ' + dep_label:<{name_w}}")
    logging.info("-" * (name_w + col_w * len(spec_names)))
    header = f"{'':>{name_w}}"
    for sn in spec_names:
        header += f"{sn:>{col_w}}"
    logging.info(header)
    logging.info("-" * (name_w + col_w * len(spec_names)))

    for var in var_list:
        label = friendly.get(var, var)
        row_c = f"  {label:<{name_w - 2}}"
        row_se = f"  {'':>{name_w - 2}}"
        for sn in spec_names:
            m = results[sn]["model"]
            if var in m.params.index:
                se_val = (
                    m.std_errors[var]
                    if hasattr(m, "std_errors")
                    else m.bse[var]
                )
                c, se, stars = format_coef(m.params[var], se_val, m.pvalues[var])
                row_c  += f"  {c:>10.4f}{stars:<4}"
                row_se += f"  ({se:>9.4f}) "
            else:
                row_c  += f"{'':>{col_w}}"
                row_se += f"{'':>{col_w}}"
        logging.info(row_c)
        logging.info(row_se)

    # Footer
    logging.info("-" * (name_w + col_w * len(spec_names)))
    for label, getter in [
        ("N", lambda m: f"{int(m.nobs):>,}"),
        ("R^2", lambda m: f"{getattr(m, 'rsquared_within', getattr(m, 'rsquared', np.nan)):>.4f}"),
    ]:
        row = f"  {label:<{name_w - 2}}"
        for sn in spec_names:
            row += f"{getter(results[sn]['model']):>{col_w}}"
        logging.info(row)

    # FE flags
    for label, key in [("BankxType FE", "entity_fe"), ("Time FE", "time_fe"),
                    ("SE type", "se_type")]:
        row = f"  {label:<{name_w - 2}}"
        for sn in spec_names:
            row += f"{results[sn].get(key, ''):>{col_w}}"
        logging.info(row)
    logging.info("-" * (name_w + col_w * len(spec_names)))

def run_structural_estimation(df):
    """
    Estimate the structural sleepiness equation:

    dep_balance_jkt = Y_1,cons * nr_lag
                  + Y_1,RF * selic_nr_lag
                  + Y_2,size * logA_nr_lag
                  + Y_2,eq * eq_nr_lag
                  + Y_2,Ss * seg_Ss_nr_lag
                  + [alpha_j + delta_t]
                  + epsilon

    Structural equation in levels: D_jkt = phi(S,X) * nr * D_{t-1} + eps
    No standalone intercept -- everything goes through nr * D_{t-1}.
    """
    logging.info("=" * 80)
    logging.info("STRUCTURAL ESTIMATION: deposit_balance = phi(S,X) x nr x D_{t-1}")
    logging.info("  phi(S,X) = Y'_1 S_t + Y'_2 X_jt  (Egan et al. 2025, Table 3)")
    logging.info("=" * 80)

    results = {}

    # --- Available regressors (all interacted with nr x D_{t-1}) ---
    agg_vars  = ["nr_lag", "selic_nr_lag"]
    bank_vars = ["logA_nr_lag"]
    if "eq_nr_lag" in df.columns and df["eq_nr_lag"].notna().sum() > 1000:
        bank_vars.append("eq_nr_lag")
    seg_vars = [c for c in ["seg_S2_nr_lag", "seg_S3_nr_lag", "seg_S4_nr_lag",
                            "seg_S5_nr_lag"] if c in df.columns]

    all_struct = agg_vars + bank_vars + seg_vars

    # =================================================================
    # SPEC 1: National-level only (Y_1 part), no FE, no intercept
    # =================================================================
    X1 = df[agg_vars].copy()
    m1 = sm.OLS(df["deposit_balance"], X1).fit(cov_type="HC1")
    results["(1) Natl"] = {
        "model": m1, "type": "sm",
        "entity_fe": "No", "time_fe": "No", "se_type": "HC-robust",
    }

    # =================================================================
    # SPEC 2: + Bank characteristics (Y_1 + Y_2), no FE
    # =================================================================
    X2 = df[all_struct].dropna()
    y2 = df.loc[X2.index, "deposit_balance"]
    m2 = sm.OLS(y2, X2).fit(cov_type="HC1")
    results["(2) +Bank"] = {
        "model": m2, "type": "sm",
        "entity_fe": "No", "time_fe": "No", "se_type": "HC-robust",
    }

    # =================================================================
    # SPEC 3: Entity FE + Time FE
    # =================================================================
    fe_regressors = agg_vars + bank_vars
    df_fe = df.dropna(subset=fe_regressors + ["entity", "time_idx"]).copy()
    df_fe_p = df_fe.set_index(["entity", "time_idx"])

    m3 = PanelOLS(
        df_fe_p["deposit_balance"],
        df_fe_p[fe_regressors],
        entity_effects=True, time_effects=True, drop_absorbed=True,
    ).fit(cov_type="robust")
    results["(3) +FE"] = {
        "model": m3, "type": "lm",
        "entity_fe": "Yes", "time_fe": "Yes", "se_type": "HC-robust",
    }

    # =================================================================
    # SPEC 4: Same as 3, bank-clustered SEs
    # =================================================================
    m4 = PanelOLS(
        df_fe_p["deposit_balance"],
        df_fe_p[fe_regressors],
        entity_effects=True, time_effects=True, drop_absorbed=True,
    ).fit(cov_type="clustered", cluster_entity=True)
    results["(4) Clust"] = {
        "model": m4, "type": "lm",
        "entity_fe": "Yes", "time_fe": "Yes", "se_type": "Bank-clust",
    }

    # --- Print results ---
    print_table(results, dep_label="Dep_jkt (levels)")

    # --- Interpretation ---
    logging.info("Interpretation guide:")
    logging.info("  Y_1,cons = coef on nr_lag = phi_cons (retention scaling factor)")
    logging.info("  Y_1,RF < 0 means depositors sleepier when rates are low")
    logging.info("  Y_2,size > 0 means larger banks retain more deposits")
    if "(1) Natl" in results:
        u1c = results["(1) Natl"]["model"].params.get("nr_lag", np.nan)
        logging.info(f"Baseline retention coef on nr_lag (Spec 1): {u1c:.4f}")

    return results


# 2.3b) Control Function estimation:

def run_control_function_estimation(df):
    """
    Structural estimation via semi-parametric Control Function (CF) approach
    (Blundell & Powell 2003; Rivers & Vuong 1988) at the CONGLOMERATE level.

    Motivation:
        All deposit types may be endogenous: each conglomerate sets its spread
        (risk_free - deposit_rate) partly as a function of unobserved depositor
        demand.  Banks with more loyal depositors (high phi) can afford lower
        spreads (i.e. pay less relative to the risk-free rate), so OLS conflates
        retained-deposit inertia with demand.

    CF Design:
        Stage 1 (pooled OLS, all types and quarters):
            spread_jkt = f(cost_shifters_jt, type_dummies) + v_jkt
            cost_shifters:  cosif_lag, log_assets_std, equity_ratio,
                            lagged_cdi, lagged_selic, type-k dummies
            Residual v_hat captures demand/competition shocks.

        Stage 2 (structural equation with cubic CF polynomial):
            deposit_balance = phi(S,X)*nr*D_{t-1}  +  b1*v + b2*v^2 + b3*v^3  + eps
            Polynomial NOT interacted with nr.
            Joint significance of (b1,b2,b3) indicates OLS endogeneity bias.

    Separate Hausman IV specification:
        Stage 1 adds the leave-one-out mean spread (same type x quarter) as
        an excluded instrument (BLP rationale: shifts own spread through
        competition/costs, uncorrelated with own idiosyncratic demand).

    Specifications:
        CF-Cost (cost shifters only in first stage):
          (5)  CF national-level          + poly(v_hat)
          (6)  CF + bank characteristics  + poly(v_hat)
          (7)  CF + entity x time FE      + poly(v_hat)  [pooled resid in FE]
          (8)  CF + FE + bank-clustered SE

        CF-Hausman (cost shifters + Hausman IV in first stage):
          (9)  CF-Hausman national        + poly(v_hat_H)
          (10) CF-Hausman + bank chars    + poly(v_hat_H)
    """
    logging.info("=" * 80)
    logging.info("CONTROL FUNCTION ESTIMATION (semi-parametric CF) [CONGLOMERATE]")
    logging.info("  Stage 1: spread_jkt ~ cost_shifters + type_dummies  -->  v_hat")
    logging.info("  Stage 2: deposit_balance = phi(S,X)*nr*D_{t-1} + b1*v + b2*v^2 + b3*v^3 + eps")
    logging.info("  All deposit types treated as potentially endogenous.")
    logging.info("=" * 80)

    results = {}

    # -- Verify spread_qoq is present --------------------------------------
    if "spread_qoq" not in df.columns:
        logging.warning("spread_qoq missing; computing as lagged_rf - deposit_rate")
        if "deposit_rate_qoq" in df.columns and "lagged_rf" in df.columns:
            df["spread_qoq"] = df["lagged_rf"] - df["deposit_rate_qoq"]
        else:
            logging.error("Cannot compute spread_qoq -- skipping CF estimation")
            return results

    # -- Cost shifters available in estimation sample ----------------------
    candidate_cost = ["cosif_lag", "log_assets_std", "equity_ratio",
                      "lagged_cdi", "lagged_selic",
                      "personnel_cost_ratio_lag", "admin_cost_ratio_lag"]
    fs_cost_vars = [v for v in candidate_cost
                    if v in df.columns and df[v].notna().sum() > 500]
    if not fs_cost_vars:
        logging.warning("No cost shifters available -- skipping CF")
        return results
    logging.info(f"First-stage cost shifters: {fs_cost_vars}")

    # -- Type dummies in first stage (type 1 = demand deposits = baseline) -
    for t in [2, 3, 4, 5]:
        df[f"type{t}_d"] = (df["deposit_type"] == t).astype(float)
    type_cols   = ["type2_d", "type3_d", "type4_d", "type5_d"]
    all_fs_vars = fs_cost_vars + type_cols

    # -- Helper: full-sample residuals from pooled first stage -------------
    def _resid_full(m_fs, fs_vars):
        valid  = df[fs_vars + ["spread_qoq"]].dropna().index
        X_pred = sm.add_constant(df.loc[valid, fs_vars])
        pred   = pd.Series(m_fs.predict(X_pred), index=valid)
        return (df.loc[valid, "spread_qoq"] - pred).reindex(df.index)

    # -- FIRST STAGE A: Cost shifters only ---------------------------------
    df_fs = df.dropna(subset=all_fs_vars + ["spread_qoq"]).copy()
    X_fs  = sm.add_constant(df_fs[all_fs_vars])
    m_fs  = sm.OLS(df_fs["spread_qoq"], X_fs).fit(cov_type="HC1")
    logging.info(f"First-stage A -- N={int(m_fs.nobs):,}, R^2={m_fs.rsquared:.4f}")
    for v in all_fs_vars:
        logging.info(f"  {v}: coef={m_fs.params.get(v, np.nan):.6f}, "
                     f"p={m_fs.pvalues.get(v, np.nan):.4f}")
    try:
        f_cost = float(m_fs.f_test([f"{v} = 0" for v in fs_cost_vars]).fvalue)
        logging.info(f"  Joint F (cost vars): {f_cost:.2f} "
                     f"({'strong' if f_cost > 10 else 'weak'})")
    except Exception as e:
        logging.warning(f"  Joint F-test failed: {e}")

    v_hat_a = _resid_full(m_fs, all_fs_vars)
    df["cf_resid"]    = v_hat_a
    df["cf_resid_sq"] = df["cf_resid"] ** 2
    df["cf_resid_cu"] = df["cf_resid"] ** 3

    # -- FIRST STAGE B: + Hausman IV (leave-one-out mean spread) -----------
    logging.info("Computing Hausman IV (leave-one-out spread mean by type x quarter) ...")
    grp   = df.groupby(["deposit_type", "AnoMes"])
    sum_s = grp["spread_qoq"].transform("sum")
    cnt_s = grp["spread_qoq"].transform("count")
    df["hausman_iv_spread"] = np.where(
        cnt_s > 1, (sum_s - df["spread_qoq"]) / (cnt_s - 1), np.nan
    )
    all_fs_h_vars = ["hausman_iv_spread"] + all_fs_vars
    df_fs_h = df.dropna(subset=all_fs_h_vars + ["spread_qoq"]).copy()
    X_fs_h  = sm.add_constant(df_fs_h[all_fs_h_vars])
    m_fs_h  = sm.OLS(df_fs_h["spread_qoq"], X_fs_h).fit(cov_type="HC1")
    logging.info(f"First-stage B -- N={int(m_fs_h.nobs):,}, R^2={m_fs_h.rsquared:.4f}")
    for v in ["hausman_iv_spread"] + fs_cost_vars:
        logging.info(f"  {v}: coef={m_fs_h.params.get(v, np.nan):.6f}, "
                     f"p={m_fs_h.pvalues.get(v, np.nan):.4f}")
    try:
        f_h = float(m_fs_h.f_test("hausman_iv_spread = 0").fvalue)
        logging.info(f"  F (Hausman IV): {f_h:.2f} "
                     f"({'strong' if f_h > 10 else 'weak'})")
    except Exception as e:
        logging.warning(f"  Hausman F-test failed: {e}")

    v_hat_h = _resid_full(m_fs_h, all_fs_h_vars)
    df["cf_resid_h"]    = v_hat_h
    df["cf_resid_h_sq"] = df["cf_resid_h"] ** 2
    df["cf_resid_h_cu"] = df["cf_resid_h"] ** 3

    for tag, col in [("Cost-only", "cf_resid"), ("Hausman", "cf_resid_h")]:
        rsd = df[col].dropna()
        logging.info(f"CF residuals ({tag}): N={len(rsd):,}, "
                     f"mean={rsd.mean():.6f}, sd={rsd.std():.6f}")

    # -- Second stage setup ------------------------------------------------
    agg_vars      = ["nr_lag", "selic_nr_lag"]
    bank_vars     = ["logA_nr_lag"]
    if "eq_nr_lag" in df.columns and df["eq_nr_lag"].notna().sum() > 1000:
        bank_vars.append("eq_nr_lag")
    seg_vars      = [c for c in ["seg_S2_nr_lag", "seg_S3_nr_lag", "seg_S4_nr_lag",
                                  "seg_S5_nr_lag"] if c in df.columns]
    fe_regressors = agg_vars + bank_vars
    poly_a = ["cf_resid", "cf_resid_sq", "cf_resid_cu"]
    poly_h = ["cf_resid_h", "cf_resid_h_sq", "cf_resid_h_cu"]

    def _log_poly(m, poly_cols, label):
        """Log individual and joint significance of polynomial CF terms."""
        present = [c for c in poly_cols if c in m.params.index]
        for c in present:
            coef = m.params[c]; pval = m.pvalues[c]
            stars = "***" if pval < 0.01 else ("**" if pval < 0.05
                    else ("*" if pval < 0.10 else ""))
            logging.info(f"  {label} {c}: {coef:.6f} (p={pval:.4f}) {stars}")
        if len(present) > 1:
            try:
                fj = float(m.f_test([f"{c} = 0" for c in present]).fvalue)
                logging.info(f"  {label} joint F on poly: {fj:.2f} "
                             + ("-> endogeneity" if fj > 3.84 else "-> no bias"))
            except Exception:
                pass

    # -- Spec 5: CF-Cost national ------------------------------------------
    s5  = agg_vars + poly_a
    df5 = df.dropna(subset=s5 + ["deposit_balance"]).copy()
    m5  = sm.OLS(df5["deposit_balance"], df5[s5]).fit(cov_type="HC1")
    results["(5) CF Natl"] = {
        "model": m5, "type": "sm",
        "entity_fe": "No", "time_fe": "No", "se_type": "HC-robust",
    }
    _log_poly(m5, poly_a, "CF-Cost(5)")

    # -- Spec 6: CF-Cost + bank characteristics ----------------------------
    s6  = agg_vars + bank_vars + seg_vars + poly_a
    df6 = df.dropna(subset=s6 + ["deposit_balance"]).copy()
    m6  = sm.OLS(df6["deposit_balance"], df6[s6]).fit(cov_type="HC1")
    results["(6) CF+Bank"] = {
        "model": m6, "type": "sm",
        "entity_fe": "No", "time_fe": "No", "se_type": "HC-robust",
    }
    _log_poly(m6, poly_a, "CF-Cost(6)")

    # -- Specs 7-8: CF-Cost + entity x time FE ----------------------------
    # Use POOLED first-stage residuals in the FE spec: they retain within-entity
    # time variation from cost shifters, so are NOT absorbed by entity FE.
    s78  = fe_regressors + poly_a
    df78 = df.dropna(subset=s78 + ["entity", "time_idx", "deposit_balance"]).copy()
    if len(df78) > 500:
        df78_p = df78.set_index(["entity", "time_idx"])
        try:
            m7 = PanelOLS(
                df78_p["deposit_balance"], df78_p[s78],
                entity_effects=True, time_effects=True, drop_absorbed=True,
            ).fit(cov_type="robust")
            results["(7) CF+FE"] = {
                "model": m7, "type": "lm",
                "entity_fe": "Yes", "time_fe": "Yes", "se_type": "HC-robust",
            }
            for c in poly_a:
                logging.info(f"  CF+FE(7) {c}: "
                             f"{m7.params.get(c, np.nan):.6f} "
                             f"(p={m7.pvalues.get(c, np.nan):.4f})")
            m8 = PanelOLS(
                df78_p["deposit_balance"], df78_p[s78],
                entity_effects=True, time_effects=True, drop_absorbed=True,
            ).fit(cov_type="clustered", cluster_entity=True)
            results["(8) CF+Clust"] = {
                "model": m8, "type": "lm",
                "entity_fe": "Yes", "time_fe": "Yes", "se_type": "Bank-clust",
            }
        except Exception as e:
            logging.warning(f"CF-Cost FE specs (7-8) failed: {e}")
    else:
        logging.warning(f"CF FE sample too small ({len(df78)}) -- skipping specs 7-8")

    # -- Spec 9: CF-Hausman national ---------------------------------------
    s9  = agg_vars + poly_h
    df9 = df.dropna(subset=s9 + ["deposit_balance"]).copy()
    m9  = sm.OLS(df9["deposit_balance"], df9[s9]).fit(cov_type="HC1")
    results["(9) CF-Haus"] = {
        "model": m9, "type": "sm",
        "entity_fe": "No", "time_fe": "No", "se_type": "HC-robust",
    }
    _log_poly(m9, poly_h, "CF-Hausman(9)")

    # -- Spec 10: CF-Hausman + bank characteristics ------------------------
    s10  = agg_vars + bank_vars + seg_vars + poly_h
    df10 = df.dropna(subset=s10 + ["deposit_balance"]).copy()
    m10  = sm.OLS(df10["deposit_balance"], df10[s10]).fit(cov_type="HC1")
    results["(10) CF-H+Bank"] = {
        "model": m10, "type": "sm",
        "entity_fe": "No", "time_fe": "No", "se_type": "HC-robust",
    }
    _log_poly(m10, poly_h, "CF-Hausman(10)")

    if results:
        print_table(results, dep_label="Dep_jkt (levels, CF)")

    results["_fs_cost"]    = {"model": m_fs,   "n_obs": int(m_fs.nobs)}
    results["_fs_hausman"] = {"model": m_fs_h,  "n_obs": int(m_fs_h.nobs)}
    return results



# 2.4) Exporting results:

def export_results(results, output_dir, suffix=""):
    """Export all coefficient estimates to CSV."""
    rows = []
    for spec_name, d in results.items():
        if spec_name.startswith("_"):
            continue  # skip internal metadata (e.g. _first_stage)
        m = d["model"]
        for var in m.params.index:
            se = m.std_errors[var] if hasattr(m, "std_errors") else m.bse[var]
            rows.append({
                "spec": spec_name,
                "variable": var,
                "coef": m.params[var],
                "se": se,
                "pvalue": m.pvalues[var],
                "ci_lower": m.params[var] - 1.96 * se,
                "ci_upper": m.params[var] + 1.96 * se,
                "n": int(m.nobs),
                "r2": getattr(m, "rsquared_within", getattr(m, "rsquared", np.nan)),
                "entity_fe": d.get("entity_fe", ""),
                "time_fe": d.get("time_fe", ""),
                "se_type": d.get("se_type", ""),
            })

    df_out = pd.DataFrame(rows)
    fname = f"estimation_results{suffix}.csv"
    path = os.path.join(output_dir, fname)
    df_out.to_csv(path, index=False)
    logging.info(f"Results exported to {path}")
    return df_out


def export_latex_table(results, output_dir,
                       dep_label=r"$\mathrm{Dep}_{jkt}$ (levels)",
                       caption="Structural Sleepiness Estimation",
                       label="tab:structural",
                       filename="estimation_table.tex"):
    r"""
    Export a stargazer-style LaTeX regression table.

    Produces a self-contained .tex file (booktabs + threeparttable)
    that can be \input{} directly into a paper.
    """
    # Filter out internal metadata keys
    spec_names = [k for k in results.keys() if not k.startswith("_")]
    n_specs = len(spec_names)

    # --- Collect and order regressors ---
    all_vars = set()
    for sn in spec_names:
        all_vars.update(results[sn]["model"].params.index)

    order = ["nr_lag", "selic_nr_lag", "logA_nr_lag", "eq_nr_lag",
             "seg_S2_nr_lag", "seg_S3_nr_lag", "seg_S4_nr_lag", "seg_S5_nr_lag",
             "cf_resid", "cf_resid_sq", "cf_resid_cu",
             "cf_resid_h", "cf_resid_h_sq", "cf_resid_h_cu"]
    var_list = [v for v in order if v in all_vars]
    for v in sorted(all_vars):
        if v not in var_list:
            var_list.append(v)

    # LaTeX-friendly variable names
    friendly = {
        "nr_lag":          r"$\Upsilon_{1,\mathrm{cons}}$ ($nr \times D_{t-1}$)",
        "selic_nr_lag":    r"$\Upsilon_{1,RF}$ (Selic $\times\, nr \times D_{t-1}$)",
        "logA_nr_lag":     r"$\Upsilon_{2,\mathrm{size}}$ ($\ln A \times nr \times D_{t-1}$)",
        "eq_nr_lag":       r"$\Upsilon_{2,\mathrm{eq}}$ ($E\!/\!A \times nr \times D_{t-1}$)",
        "seg_S2_nr_lag":   r"$\Upsilon_{2,S2}$ ($\mathbb{1}_{S2} \times nr \times D_{t-1}$)",
        "seg_S3_nr_lag":   r"$\Upsilon_{2,S3}$ ($\mathbb{1}_{S3} \times nr \times D_{t-1}$)",
        "seg_S4_nr_lag":   r"$\Upsilon_{2,S4}$ ($\mathbb{1}_{S4} \times nr \times D_{t-1}$)",
        "seg_S5_nr_lag":   r"$\Upsilon_{2,S5}$ ($\mathbb{1}_{S5} \times nr \times D_{t-1}$)",
        "cf_resid":        r"$\hat{v}$ (cost CF, 1st)",
        "cf_resid_sq":     r"$\hat{v}^2$ (2nd)",
        "cf_resid_cu":     r"$\hat{v}^3$ (3rd)",
        "cf_resid_h":      r"$\hat{v}^H$ (Hausman CF, 1st)",
        "cf_resid_h_sq":   r"$(\hat{v}^H)^2$ (2nd)",
        "cf_resid_h_cu":   r"$(\hat{v}^H)^3$ (3rd)",
    }

    # --- Build LaTeX lines ---
    L = []  # accumulator
    col_spec = "l" + "c" * n_specs

    # Table environment
    L.append(r"\begin{table}[htbp]")
    L.append(r"\centering")
    L.append(r"\begin{threeparttable}")
    L.append(r"\caption{" + caption + "}")
    L.append(r"\label{" + label + "}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{" + col_spec + "}")
    L.append(r"\toprule")

    # Dependent-variable header spanning all columns
    L.append(
        r" & \multicolumn{" + str(n_specs) + r"}{c}{"
        + dep_label + r"} \\"
    )
    L.append(r"\cmidrule(lr){2-" + str(n_specs + 1) + "}")

    # Column labels
    header = ""
    for sn in spec_names:
        header += f" & {sn}"
    header += r" \\"
    L.append(header)
    L.append(r"\midrule")

    # ---- Coefficient rows ----
    for var in var_list:
        lbl = friendly.get(var, var.replace("_", r"\_"))
        coef_cells = ""
        se_cells   = ""
        for sn in spec_names:
            m = results[sn]["model"]
            if var in m.params.index:
                se_val = (
                    m.std_errors[var]
                    if hasattr(m, "std_errors")
                    else m.bse[var]
                )
                stars = _stars_tex(m.pvalues[var])
                coef_cells += f" & ${_fmt_val(m.params[var])}{stars}$"
                se_cells   += f" & $({_fmt_val(se_val)})$"
            else:
                coef_cells += " & "
                se_cells   += " & "
        L.append(f"{lbl}{coef_cells}" + r" \\")
        L.append(f"{se_cells}" + r" \\[3pt]")

    L.append(r"\midrule")

    # ---- Footer statistics ----
    # N
    n_row = "Observations"
    for sn in spec_names:
        n_row += f" & {int(results[sn]['model'].nobs):,}"
    L.append(n_row + r" \\")

    # R^2
    r2_row = "$R^2$"
    for sn in spec_names:
        m = results[sn]["model"]
        r2 = getattr(m, "rsquared_within",
                     getattr(m, "rsquared", np.nan))
        r2_row += f" & {r2:.3f}"
    L.append(r2_row + r" \\")

    # FE flags
    for lbl_tex, key in [
        (r"Bank$\times$Type FE", "entity_fe"),
        ("Time FE",              "time_fe"),
    ]:
        row = lbl_tex
        for sn in spec_names:
            row += " & " + results[sn].get(key, "")
        L.append(row + r" \\")

    # SE type
    se_row = "SE type"
    for sn in spec_names:
        se_row += " & " + results[sn].get("se_type", "")
    L.append(se_row + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")

    # Table notes (stargazer-style)
    L.append(r"\begin{tablenotes}")
    L.append(r"\small")
    L.append(
        r"\item \textit{Notes:} $^{***}p<0.01$; $^{**}p<0.05$; $^{*}p<0.1$. "
        r"Standard errors in parentheses. "
        r"The dependent variable is deposit balance in \textit{levels} "
        r"($\mathrm{Dep}_{jkt}$, not the ratio). "
        r"All regressors are interacted with the net return "
        r"$nr_{jkt}=1+(R^F_{t-1}-\rho_{jkt-1})/100$ and lagged deposits $D_{t-1}$. "
        r"Specifications follow Egan et al.\ (2025), Table~3. "
        r"Panel unit: conglomerate (CodConglPrud) $\times$ deposit type."
    )
    L.append(r"\end{tablenotes}")
    L.append(r"\end{threeparttable}")
    L.append(r"\end{table}")

    # --- Write file ---
    tex_path = os.path.join(output_dir, filename)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    logging.info(f"LaTeX table exported to {tex_path}")
    return tex_path


## 3) Main execution:

if __name__ == "__main__":
    df = load_and_prepare()

    # -- OLS structural estimation (Specs 1-4) --
    results_ols = run_structural_estimation(df)
    export_results(results_ols, OUTPUT_DIR)
    export_latex_table(results_ols, OUTPUT_DIR)

    # -- Control function estimation (Specs 5-8, mirroring 1-4) --
    results_cf = run_control_function_estimation(df)
    exportable_cf = {k: v for k, v in results_cf.items() if not k.startswith("_")}

    if exportable_cf:
        export_results(exportable_cf, OUTPUT_DIR, suffix="_cf")
        export_latex_table(
            exportable_cf, OUTPUT_DIR,
            caption="Control Function Estimation (Conglomerate Level)",
            label="tab:cf_conglom",
            filename="estimation_table_cf.tex",
        )

    # -- Combined OLS + CF table for direct endogeneity comparison --
    combined = {**results_ols, **exportable_cf}
    if combined:
        export_results(combined, OUTPUT_DIR, suffix="_combined")
        export_latex_table(
            combined, OUTPUT_DIR,
            caption="Structural Estimation: OLS vs.~Control Function (Conglomerate Level)",
            label="tab:ols_vs_cf_conglom",
            filename="estimation_table_combined.tex",
        )

    logging.info("Done.")