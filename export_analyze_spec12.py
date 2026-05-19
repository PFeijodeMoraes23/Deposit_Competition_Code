import os
import sys
import shutil
from pathlib import Path
import pickle

_DRAFTS_DIR = Path(r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Drafts\Deposit Competition")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

# Mock NonLinearResults class for unpickling estimation_2_sleep pickles.
# Must match the real class's __init__ signature so pickle restores __dict__ correctly.
class NonLinearResults:
    def __init__(self, params, bse, tvalues, pvalues, df_resid,
                 params_native=None, nobs=None, rsquared=None,
                 fvalue=None, f_pvalue=None, G_nominal=None, cov_ame=None,
                 nlls_status=None, nlls_message=None):
        self.params = params
        self.bse = bse
        self.tvalues = tvalues
        self.pvalues = pvalues
        self.df_resid = df_resid
        self.params_native = params_native if params_native is not None else params
        self.G_star = df_resid
        self.nobs = nobs
        self.rsquared = rsquared
        self.fvalue = fvalue
        self.f_pvalue = f_pvalue
        self.G_nominal = G_nominal
        self.cov_ame = cov_ame
        self.nlls_status = nlls_status
        self.nlls_message = nlls_message

    def cov_params(self):
        """Return full AME covariance matrix as a DataFrame (mirrors statsmodels interface)."""
        if self.cov_ame is not None:
            return pd.DataFrame(self.cov_ame, index=self.params.index, columns=self.params.index)
        return pd.DataFrame(np.diag(self.bse ** 2), index=self.params.index, columns=self.params.index)

# Register fake module for unpickling
sys.modules['estimation_2_sleep'] = type('FakeModule', (), {'NonLinearResults': NonLinearResults})
sys.modules['estimation_2_sleep'] = type('FakeModule', (), {'NonLinearResults': NonLinearResults})

# Try to respect the project's venv guard
try:
    from utils.venv_guard import ensure_project_venv  # type: ignore[import-untyped]
    ensure_project_venv(__file__)
except ImportError:
    pass

def get_stars(pval):
    if pd.isna(pval): return ""
    if pval < 0.01: return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.1: return "*"
    return ""

def format_value(coef, se, pval, digits=4):
    if pd.isna(coef):
        return "-", "-"
    stars = get_stars(pval)
    return f"{coef:.{digits}f}{stars}", f"({se:.{digits}f})"


def pastelize_color(color, blend=0.7):
    """Blend color toward white for CI bands."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - blend) * rgb + blend * np.array([1.0, 1.0, 1.0]))

def _t_crit(res, alpha=0.025):
    """Exact two-tailed 95% CI critical value: t_{G*-1, 1-alpha} from effective cluster df."""
    if res is None:
        return float(stats.norm.ppf(1 - alpha))
    g_star = getattr(res, 'G_star', getattr(res, 'df_resid', None))
    try:
        g_star = float(g_star)
    except (TypeError, ValueError):
        return float(stats.norm.ppf(1 - alpha))
    if np.isnan(g_star) or g_star <= 1:
        return float(stats.norm.ppf(1 - alpha))
    return float(stats.t.ppf(1 - alpha, df=g_star - 1))

# All market_panel_phis.csv files store demographic/infrastructure columns already
# in the rescaled units used for estimation (gdp/10k, cadunico/100, etc.), so no
# additional scaling is needed when reconstructing the regressor matrix.
_SCALE_COLS: dict = {}


def _build_phi_regressors(df_sub: pd.DataFrame, phi_params) -> np.ndarray:
    """
    Build regressor matrix X (n_obs × len(phi_params)) from market panel columns.
    Applies the same scaling as calculate_phis() in estimation_1_sleep.py.
    Maps:  'nr_lagged_dep'      → constant 1.0
           'interaction_{sv}'   → column sv (scaled)
           other param names    → matching column if present, else 0.0
    """
    n = len(df_sub)
    X = np.zeros((n, len(phi_params)))
    for i, pname in enumerate(phi_params):
        if pname == 'nr_lagged_dep':
            X[:, i] = 1.0
        elif pname.startswith('interaction_'):
            sv = pname[len('interaction_'):]
            if sv == 'pix_exists':
                if sv in df_sub.columns:
                    v = df_sub[sv].values.astype(float)
                elif 'year' in df_sub.columns and 'quarter' in df_sub.columns:
                    v = ((df_sub['year'] > 2020) |
                         ((df_sub['year'] == 2020) & (df_sub['quarter'] == 4))
                         ).astype(float).values
                else:
                    v = np.zeros(n)
            elif sv in df_sub.columns:
                v = df_sub[sv].fillna(0.0).values.astype(float) * _SCALE_COLS.get(sv, 1.0)
            else:
                v = np.zeros(n)
            X[:, i] = v
        elif pname in df_sub.columns:
            X[:, i] = df_sub[pname].fillna(0.0).values.astype(float)
        # else: leave column as 0.0
    return X


def calc_agg_delta(d_sub: pd.DataFrame, col: str, res, is_logistic: bool) -> tuple:
    """
    Compute deposit-weighted national phi_t and its analytical (delta-method) SE.

    SE²_t = X̄_t' Σ X̄_t  where X̄_t = pop-weighted mean regressor vector for period t.

    Linear models  : Σ = cov_OLS (params = OLS beta)
    Logistic models: Σ = cov_AME (params = AME).  The phi*(1-phi) scaling is already
                     embedded in cov_AME via the delta-method Jacobian in
                     get_nlls_ame_and_se; the gradient of phi_t wrt AME is plain X̄.
    Phi-specific params = all params whose name does NOT start with 'v_hat'.
    """
    if len(d_sub) == 0 or col not in d_sub.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    w = d_sub['market_size'] if 'market_size' in d_sub.columns else pd.Series(1.0, index=d_sub.index)
    w_arr = w.values.astype(float)
    mean = (d_sub[col] * w).groupby(d_sub['year_quarter']).sum() / w.groupby(d_sub['year_quarter']).sum()

    if res is None:
        return mean, pd.Series(0.0, index=mean.index)

    all_idx = res.params.index
    phi_mask = ~pd.Series(list(all_idx)).str.startswith('v_hat').values
    phi_params = all_idx[phi_mask]

    # Covariance sub-matrix (phi params only).
    # Both statsmodels results and NonLinearResults expose cov_params().
    full_cov = res.cov_params()
    p_idx = [j for j, p in enumerate(all_idx) if p in set(phi_params)]
    Sigma = full_cov.values[np.ix_(p_idx, p_idx)]

    X = _build_phi_regressors(d_sub, phi_params)
    phi_vals = d_sub[col].fillna(0.0).values.astype(float)
    yq_arr = d_sub['year_quarter'].values

    # For both linear and logistic the gradient of phi_t w.r.t. the *reported*
    # params is plain X.
    # - Linear  : params = OLS beta, cov_params = cov_beta  → d(phi)/d(beta) = X  ✓
    # - Logistic: params = AME,      cov_params = cov_AME   → d(phi)/d(AME) ≈ X
    #   (the phi*(1-phi) factor is already embedded in cov_AME via the delta-method
    #   Jacobian in get_nlls_ame_and_se; applying it again here would double-scale
    #   and make logistic CIs ~11× too narrow for phi≈0.9)
    G = X

    se_vals: dict = {}
    for t in mean.index:
        mask = yq_arr == t
        w_t = w_arr[mask]
        g_t = G[mask]
        w_sum = w_t.sum()
        if w_sum == 0:
            se_vals[t] = 0.0
            continue
        g_bar = (w_t[:, None] * g_t).sum(axis=0) / w_sum   # (k,)
        var_t = float(g_bar @ Sigma @ g_bar)
        se_vals[t] = np.sqrt(max(var_t, 0.0))

    return mean, pd.Series(se_vals)

def nice_var_name(var):
    v = str(var).replace('interaction_', '')
    labels = {
        'nr_lagged_dep': 'Constant',
        'gdp_per_capita': 'GDP \\textit{per capita} (10k R\\$)',
        'cadunico_families_per1000': 'CadUnico Families (100s per 1k)',
        'fraction_65plus': 'Fraction 65+',
        'fraction_young': 'Fraction Young',
        'risk_free_qoq_lag': 'Lagged Selic Rate',
        'pix_users_pf_per1000': 'Pix Users (100s per 1k)',
        'connections_per100': 'Broadband Connections (per 100 inhabitants)',
        'branches_per1000': 'Branches per 1k',
        'post_2020': 'Post 2020 Dummy',
        'dummy_D_type': 'D-Type Dummy',
        'state_owned': 'State-Owned Indicator',
        'cooperative': 'Cooperative Indicator',
        'pix_exists': 'Pix Available',
        'const': 'Constant',
        'constant': 'Constant',
        'tax_cost_ratio_lag': 'Tax Cost Ratio ($t-1$)',
        'personnel_cost_ratio_lag': 'Personnel Cost Ratio ($t-1$)',
        'admin_cost_ratio_lag': 'Admin Cost Ratio ($t-1$)',
        'indice_basileia_lag': 'Basel Index (pp, $t-1$)',
        'lci_lca_ratio_lag': 'LCI/LCA Ratio ($t-1$)',
        'wholesale_ratio_lag': 'Wholesale Ratio ($t-1$)',
        'leave_one_out_mean_spread': 'Leave-out Mean Spread',
        'dummy_D_type_x_fraction_65plus': 'D-Type $\\times$ Fraction 65+',
        'dummy_D_type_x_fraction_young': 'D-Type $\\times$ Fraction Young',
        'dummy_D_type_x_risk_free_qoq_lag': 'D-Type $\\times$ Lagged Selic'
    }
    return labels.get(v, v.replace('_', '\\_'))

def build_latex_table(results_dict, order_keys, target_vars, out_path, title="", label=""):
    tex = []
    
    # Reduce font size and line spacing for this specific table by grouping it
    tex.append(r"{")
    tex.append(r"\footnotesize")
    tex.append(r"\renewcommand{\arraystretch}{0.75}")
    
    col_def = "l" + "c" * len(order_keys)
    tex.append(r"\begin{longtable}[c]{" + col_def + "}")
    tex.append(r"\caption{" + title + r"}\label{" + label + r"} \\")
    
    # First Header
    tex.append(r"\toprule")
    rename_map = {
        '1 B firms': 'Local Only',
        '3 Pooled': 'Pooled B + D',
        '4 Pooled Logistic': '(+) Logistic',
        '5 Dummies Logistic': '(+) Dummies'
    }
    headers = ["Variable"] + [rename_map.get(k, k) for k in order_keys]
    tex.append(" & ".join(headers) + r" \\")
    tex.append(r"\midrule")
    tex.append(r"\endfirsthead")
    
    # Next Headers
    tex.append(r"\multicolumn{" + str(len(order_keys) + 1) + r"}{c}{{\bfseries \tablename\ \thetable{} -- " + title + r" (continued from previous page)}} \\")
    tex.append(r"\toprule")
    tex.append(" & ".join(headers) + r" \\")
    tex.append(r"\midrule")
    tex.append(r"\endhead")
    
    # Footers
    tex.append(r"\midrule")
    tex.append(r"\multicolumn{" + str(len(order_keys) + 1) + r"}{r}{{Continued on next page}} \\")
    tex.append(r"\endfoot")
    
    # Last Footer
    tex.append(r"\bottomrule")
    notes_str = r"\multicolumn{" + str(len(order_keys) + 1) + r"}{@{}l}{\parbox[t]{\linewidth}{\footnotesize\textit{Notes:} Standard errors are in parentheses. Significance levels: * $p < 0.1$, ** $p < 0.05$, *** $p < 0.01$.}}"
    tex.append(notes_str)
    tex.append(r"\endlastfoot")
    
    # Collect data for vars
    vars_to_print = []
    # Identify all unique vars
    for col in order_keys:
        res = results_dict.get(col)
        if res is not None:
            params = getattr(res, 'params', pd.Series(dtype=float))
            for v in params.index:
                if v not in vars_to_print:
                    vars_to_print.append(v)
    
    # Sort vars to put nice target vars first
    ordered_vars = [v for v in target_vars if v in vars_to_print]

    is_first_stage = ("first_stage" in str(out_path).lower() or "stage1" in str(out_path).lower())

    if is_first_stage:
        # First stage table: ONLY exhibit the target instruments, not all variables
        pass
    else:
        # Second stage table: Include everything else, but exclude first-stage residuals
        other_vars = [v for v in vars_to_print if v not in ordered_vars]
        exclude_patterns = ["reside", "v_hat", "v_hat_2", "v_hat_3"]
        other_vars = [v for v in other_vars if not any(pattern in v.lower() for pattern in exclude_patterns)]
        ordered_vars += other_vars

    for v in ordered_vars:
        if v == "is_coop" or v == "is_state_owned":
            # Exclude is_coop and is_state_owned from print loops (handled in bottom group)
            continue
        if "is_coop" in v or "is_state_owned" in v:
            # Exclude interaction terms too
            continue

        row_cf = [nice_var_name(v)]
        row_se = [""]
        for col in order_keys:
            res = results_dict.get(col)
            if res is None:
                row_cf.append("-")
                row_se.append("-")
                continue
            
            params = getattr(res, 'params', pd.Series(dtype=float))
            bse = getattr(res, 'bse', pd.Series(dtype=float))
            pvalues = getattr(res, 'pvalues', pd.Series(dtype=float))
            
            if v in params.index:
                c_str, se_str = format_value(params[v], bse[v], pvalues[v], digits=4)
                row_cf.append(c_str)
                row_se.append(se_str)
            else:
                row_cf.append("-")
                row_se.append("-")
                
        tex.append(" & ".join(row_cf) + r" \\")
        tex.append(" & ".join(row_se) + r" \\")
        tex.append(r"\addlinespace")
        
    tex.append(r"\midrule")
    
    # Stats
    row_nobs = ["Observations"]
    row_r2 = ["$R^2$"]
    row_fstat = ["F-Statistic"]
    row_cluster = ["Clusters ($G$)"]
    row_eff_cluster = ["Effective Clusters ($G^*$)"]
    row_state = ["State Ownership Control"]
    row_coop = ["Cooperative Control"]

    for col in order_keys:
        if '5 Dummies' in col:
            row_state.append("Yes")
            row_coop.append("Yes")
        else:
            row_state.append("No")
            row_coop.append("No")

        res = results_dict.get(col)
        if res is None:
            row_nobs.append("-")
            row_r2.append("-")
            row_fstat.append("-")
            row_cluster.append("-")
            row_eff_cluster.append("-")
            continue
            
        nobs = getattr(res, 'nobs', getattr(res, 'n_obs', np.nan))
        r2 = getattr(res, 'rsquared', np.nan)
        fstat = getattr(res, 'fvalue', np.nan)
        fpval = getattr(res, 'f_pvalue', np.nan)

        # Map NLLS/Logistic columns to their linear counterpart for fallback stats
        _nlls_fallback = {
            '4 Pooled Logistic': '3 Pooled',
            '5 Dummies Logistic': '5 Dummies Linear',
        }

        # If NLLS/Logistic, copy the missing metrics from the counterpart Linear model
        if pd.isna(nobs) or pd.isna(r2):
            fallback_col = _nlls_fallback.get(col)
            if fallback_col and fallback_col in results_dict:
                f_res = results_dict[fallback_col]
                if f_res is not None:
                    if pd.isna(nobs): nobs = getattr(f_res, 'nobs', getattr(f_res, 'n_obs', np.nan))
                    if pd.isna(r2): r2 = getattr(f_res, 'rsquared', np.nan)
                    if pd.isna(fstat): fstat = getattr(f_res, 'fvalue', np.nan)
                    if pd.isna(fpval): fpval = getattr(f_res, 'f_pvalue', np.nan)

        # clusters
        clusters = "-"
        # Fetch from itself first (linear statsmodels results)
        if hasattr(res, 'cov_kwds') and res.cov_kwds.get('groups', None) is not None:
            groups = res.cov_kwds.get('groups', None)
            clusters = str(groups.nunique() if hasattr(groups, 'nunique') else len(set(groups)))
        elif hasattr(res, 'G_nominal') and not pd.isna(getattr(res, 'G_nominal', np.nan)):
            # NLLS results store G_nominal directly
            clusters = str(int(res.G_nominal))
        else:
            # Fallback to linear counterpart
            fallback_col = _nlls_fallback.get(col)
            if fallback_col and fallback_col in results_dict:
                f_res = results_dict[fallback_col]
                if f_res is not None and hasattr(f_res, 'cov_kwds') and f_res.cov_kwds.get('groups', None) is not None:
                    groups = f_res.cov_kwds.get('groups', None)
                    clusters = str(groups.nunique() if hasattr(groups, 'nunique') else len(set(groups)))
        
        # effective clusters
        g_star = getattr(res, 'G_star', getattr(res, 'df_resid', np.nan))
        
        fstat_str = f"{fstat:.3f}{get_stars(fpval)}" if pd.notna(fstat) else "-"
        row_nobs.append(f"{nobs:,.0f}" if pd.notna(nobs) else "-")
        row_r2.append(f"{r2:.3f}" if pd.notna(r2) else "-")
        row_fstat.append(fstat_str)
        row_cluster.append(clusters)
        row_eff_cluster.append(f"{g_star:.1f}" if pd.notna(g_star) else "-")
        
    tex.append(" & ".join(row_nobs) + r" \\")
    tex.append(" & ".join(row_r2) + r" \\")
    if is_first_stage:
        tex.append(" & ".join(row_fstat) + r" \\")
    tex.append(" & ".join(row_cluster) + r" \\")
    tex.append(" & ".join(row_eff_cluster) + r" \\")
    tex.append(" & ".join(row_state) + r" \\")
    tex.append(" & ".join(row_coop) + r" \\")
    
    tex.append(r"\end{longtable}")
    tex.append(r"}")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))

import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze Specification 12 Results")
    parser.add_argument('--skip-est2', action='store_true', help='Skip estimation 2 (B firms Robust)')
    args = parser.parse_args()

    print("Collecting Estimation results for Spec 12 (IV_HausmanFull x Tech)...")
    _ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
    SLEEP_DIR = DATA_DIR / "ESTIMATION_OUTPUT"
    
    if not SLEEP_DIR.exists():
        print(f"ERROR: Cannot find {SLEEP_DIR}")
        return

    mapping = {
        '1 B firms': SLEEP_DIR / "DEMAND_PREP" / "est1",
        '3 Pooled': SLEEP_DIR / "DEMAND_PREP" / "est3",
        '4 Pooled Logistic': SLEEP_DIR / "DEMAND_PREP" / "est4",
        '5 Dummies Logistic': SLEEP_DIR / "DEMAND_PREP" / "est5",
    }

    if getattr(args, 'skip_est2', False):
        mapping.pop('2 B firms Robust', None)

    stage1_res = {}
    stage2_res = {}
    phi_data = {}
    models_dict = {}

    for label, d in mapping.items():
        pkl_path = d / "estimation_results.pkl"
        if not pkl_path.exists():
            print(f"  [Warning] Missing {pkl_path}, skipping {label}.")
            continue
            
        with open(pkl_path, 'rb') as f:
            try:
                res_dict = pickle.load(f)
            except Exception as e:
                print(f"  [Error] Failed to load {pkl_path}: {e}")
                continue
                
        target_keys = {
            '1 B firms': 'IV_HausmanFull x Tech',
            '3 Pooled': 'IV_HausmanFull x Tech',
            '4 Pooled Logistic': 'IV_HausmanFull x Tech', 
            '5 Dummies Logistic': 'IV_HausmanFull x Tech x logistic',
        }

        # Fallback if the script saved differently (linear vs logistic)
        tk = target_keys.get(label, 'IV_HausmanFull x Tech')
        spec_data = res_dict.get(tk)
        
        if not spec_data:
            print(f"  [Warning] {tk} not found in {label}, skipping.")
            continue
            
        models_dict[label] = spec_data
        
        # Attach models for table builder
        if spec_data.get('first_stage') is not None:
            stage1_res[label] = spec_data['first_stage']
        if spec_data.get('second_stage') is not None:
            stage2_res[label] = spec_data['second_stage']

        csv_path = d / "market_panel_phis.csv"
        if csv_path.exists():
            try:
                df_phi = pd.read_csv(csv_path, low_memory=False)
                phi_data[label] = df_phi
            except Exception as e:
                print(f"  [Warning] Could not load phi CSV for {label}: {e}")

    # ---- 1) Generate DataFrames & Export LaTeX Tables ----
    out_dir = SLEEP_DIR / "Rout"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    order = list(mapping.keys())
    if '5 Dummies Linear' in order:
        order.remove('5 Dummies Linear')
        
    target_vars = [
        'nr_lagged_dep', 'interaction_gdp_per_capita', 'interaction_cadunico_families_per1000',
        'interaction_fraction_65plus', 'interaction_fraction_young', 'interaction_risk_free_qoq_lag',
        'interaction_pix_users_pf_per1000', 'interaction_connections_per100',
        'interaction_branches_per1000', 'interaction_dummy_D_type', 'interaction_state_owned',
        'interaction_cooperative', 'dummy_D_type', 'interaction_dummy_D_type_x_fraction_65plus',
        'interaction_dummy_D_type_x_fraction_young', 'interaction_dummy_D_type_x_risk_free_qoq_lag'
    ]
    
    first_stage_target_vars = [
        'tax_cost_ratio_lag', 'personnel_cost_ratio_lag', 'admin_cost_ratio_lag',
        'indice_basileia_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag',
        'leave_one_out_mean_spread'
    ]

    build_latex_table(stage1_res, order, first_stage_target_vars, out_dir / "est1-5_spec12_stage1_comparison.tex", title="First Stage IV Results across Specifications", label="tab:spec12_stage1_comparison")
    build_latex_table(stage2_res, order, target_vars, out_dir / "est1-5_spec12_stage2_comparison.tex", title="Second Stage Results across Specifications", label="tab:spec12_stage2_comparison")
    shutil.copy(out_dir / "est1-5_spec12_stage1_comparison.tex", _DRAFTS_DIR / "est1-5_spec12_stage1_comparison.tex")
    shutil.copy(out_dir / "est1-5_spec12_stage2_comparison.tex", _DRAFTS_DIR / "est1-5_spec12_stage2_comparison.tex")
    print(f"Exported LaTeX Tables to {out_dir} and copied to {_DRAFTS_DIR}")

    # ---- 2) Pickle Model Information ----
    with open(out_dir / "est1-5_spec12_all_models.pkl", "wb") as f:
        pickle.dump(models_dict, f)
    print(f"Exported combined model instances to {out_dir / 'est1-5_spec12_all_models.pkl'}")

    # ---- 3) Plot Implied National Phi_t ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Pre-define a color map for consistent colors across both subplots
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {lbl: base_colors[i % len(base_colors)] for i, lbl in enumerate(phi_data.keys())}
    # Dummies specs use dashed lines so they remain distinguishable when values overlap
    linestyle_map = {
        '1 B firms': '-',
        '3 Pooled': '-',
        '4 Pooled Logistic': '-',
        '5 Dummies Linear': '--',
        '5 Dummies Logistic': '--',
    }

    label_rename_map = {
        '1 B firms': 'Local Only',
        '3 Pooled': 'Pooled B + D',
        '4 Pooled Logistic': '(+) Logistic',
        '5 Dummies Linear': '(+) Dummies Linear',
        '5 Dummies Logistic': '(+) Dummies',
    }

    ci_b: list = []
    ci_d: list = []
    for label, df_phi in phi_data.items():
        if 'year_quarter' not in df_phi.columns:
            continue
            
        phi_col_b = f"phi_b" if "phi_b" in df_phi.columns else None
        phi_col_d = f"phi_d" if "phi_d" in df_phi.columns else None
        
        if "1 B firms" in label:
            df_b = df_phi
            df_d = df_phi.copy()
        elif 'dummy_D_type' in df_phi.columns:
            # When pooled and has the dummy, we can separate
            df_b = df_phi[df_phi['dummy_D_type'] == 0]
            df_d = df_phi[df_phi['dummy_D_type'] == 1]
            if len(df_d) == 0 and 'phi_d' in df_phi.columns:
                df_b = df_phi
                df_d = df_phi
        elif "2 D firms" in label:
            df_b = pd.DataFrame(columns=df_phi.columns)
            df_d = df_phi
        else:
            # Assume everything else is B firms entirely, EXCEPT pooled where dummy_D_type might be missing but we know it's pooled?
            # Actually, Alt2 Logistic might apply to both but missing dummy?
            # Let's check if the dataframe has 'is_B' instead.
            if 'is_B' in df_phi.columns:
                df_b = df_phi[df_phi['is_B'] == 1]
                df_d = df_phi[df_phi['is_B'] == 0]
            else:
                # Default fallback
                df_b = df_phi
                df_d = pd.DataFrame(columns=df_phi.columns)

        tar_col = "phi_mt_IV_HausmanFull_x_Tech"
        
        # Explicitly map target columns based on spec label to prevent collisions
        if "Linear" in label:
            possible_cols = ["phi_mt_IV_HausmanFull_x_Tech_x_linear", "phi_mt_IV_HausmanFull_x_Tech_linear"]
        elif "Logistic" in label:
            if "5 Dummies Logistic" in label:
                possible_cols = ["phi_mt_IV_HausmanFull_x_Tech_x_logistic", "phi_mt_IV_HausmanFull_x_Tech_logistic"]
            else:
                possible_cols = ["phi_mt_IV_HausmanFull_x_Tech"]
        elif "2 D firms" in label:
            possible_cols = ["phi_mt_Option_2_IV_HausmanFull_Tech"]
        else:
            possible_cols = ["phi_mt_IV_HausmanFull_x_Tech"]
            
        # Fallback search if the precise requested column is missing
        possible_cols += ["phi_mt_IV_HausmanFull_x_Tech", "phi_mt_Option_2_IV_HausmanFull_Tech", "phi_mt_Tech"]
        
        for p_col in possible_cols:
            if p_col in df_phi.columns:
                tar_col = p_col
                break
                
        plot_label = label_rename_map.get(label, label)
        c = color_map[label]
        ls = linestyle_map.get(label, '-')
        crit_val = _t_crit(stage2_res.get(label))
        res = stage2_res.get(label)
        is_logistic = 'Logistic' in label

        if tar_col in df_phi.columns:
            agg_b, se_b = calc_agg_delta(df_b, tar_col, res, is_logistic)
            agg_d, se_d = calc_agg_delta(df_d, tar_col, res, is_logistic)

            if not agg_b.empty:
                idx_dates = pd.PeriodIndex(agg_b.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
                axes[0].plot(idx_dates, agg_b.values, label=plot_label, color=c, linewidth=2, linestyle=ls)
                ci_b.append((idx_dates, agg_b.values - crit_val * se_b.values, agg_b.values + crit_val * se_b.values, pastelize_color(c)))

            if not agg_d.empty:
                idx_dates = pd.PeriodIndex(agg_d.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
                axes[1].plot(idx_dates, agg_d.values, label=plot_label, color=c, linewidth=2, linestyle=ls)
                ci_d.append((idx_dates, agg_d.values - crit_val * se_d.values, agg_d.values + crit_val * se_d.values, pastelize_color(c)))
                
    axes[0].set_title("B-Type Firms (Spec 12)", fontsize=14)
    axes[0].set_ylabel(r"National $\hat{\phi}_t$")
    axes[0].set_ylim(bottom=0)
    axes[0].grid()
    axes[0].legend(loc='best')

    axes[1].set_title("D-Type Firms (Spec 12)", fontsize=14)
    axes[1].set_ylim(bottom=0)
    axes[1].grid()
    axes[1].legend(loc='best')
    
    fig.tight_layout()
    plot_path = out_dir / "est1-5_spec12_phi_t_comparison.png"
    plt.savefig(plot_path, dpi=300)
    for x, lo, hi, pc in ci_b:
        axes[0].fill_between(x, lo, hi, color=pc, alpha=0.45)
    for x, lo, hi, pc in ci_d:
        axes[1].fill_between(x, lo, hi, color=pc, alpha=0.45)
    plot_path_ci = out_dir / "est1-5_spec12_phi_t_comparison_ci_pastel.png"
    plt.savefig(plot_path_ci, dpi=300)
    plt.close(fig)
    shutil.copy(plot_path, _DRAFTS_DIR / "est1-5_spec12_phi_t_comparison.png")
    shutil.copy(plot_path_ci, _DRAFTS_DIR / "est1-5_spec12_phi_t_comparison_ci_pastel.png")
    print(f"Exported combined plots to {plot_path} and {plot_path_ci}; copied to {_DRAFTS_DIR}")

    # ---- 4) Subset plot: 1 B firms, 3 Pooled, 4 Pooled Logistic, 5 Dummies Logistic ----
    _SUBSET_LABELS = {'1 B firms', '3 Pooled', '4 Pooled Logistic', '5 Dummies Logistic'}
    _SUBSET_RENAME = {
        '1 B firms': 'Local Only',
        '3 Pooled': 'Pooled B + D',
        '4 Pooled Logistic': '(+) Logistic',
        '5 Dummies Logistic': '(+) Dummies',
    }
    _SUBSET_LINESTYLE = {
        '1 B firms': '-',
        '3 Pooled': '-',
        '4 Pooled Logistic': '-',
        '5 Dummies Logistic': '--',
    }
    phi_data_sub = {k: v for k, v in phi_data.items() if k in _SUBSET_LABELS}

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    ci_b2: list = []
    ci_d2: list = []
    for label, df_phi in phi_data_sub.items():
        if 'year_quarter' not in df_phi.columns:
            continue

        if "1 B firms" in label:
            df_b = df_phi
            df_d = df_phi.copy()
        elif 'dummy_D_type' in df_phi.columns:
            df_b = df_phi[df_phi['dummy_D_type'] == 0]
            df_d = df_phi[df_phi['dummy_D_type'] == 1]
            if len(df_d) == 0 and 'phi_d' in df_phi.columns:
                df_b = df_phi
                df_d = df_phi
        elif 'is_B' in df_phi.columns:
            df_b = df_phi[df_phi['is_B'] == 1]
            df_d = df_phi[df_phi['is_B'] == 0]
        else:
            df_b = df_phi
            df_d = pd.DataFrame(columns=df_phi.columns)

        def calc_agg2(d_sub, col):
            # Kept as alias for calc_agg_delta; is_logistic resolved from outer `label`
            return calc_agg_delta(d_sub, col, stage2_res.get(label), 'Logistic' in label)

        if "Logistic" in label:
            if "5 Dummies Logistic" in label:
                possible_cols = ["phi_mt_IV_HausmanFull_x_Tech_x_logistic", "phi_mt_IV_HausmanFull_x_Tech_logistic"]
            else:
                possible_cols = ["phi_mt_IV_HausmanFull_x_Tech"]
        elif "Linear" in label:
            possible_cols = ["phi_mt_IV_HausmanFull_x_Tech_x_linear", "phi_mt_IV_HausmanFull_x_Tech_linear"]
        elif "2 D firms" in label:
            possible_cols = ["phi_mt_Option_2_IV_HausmanFull_Tech"]
        else:
            possible_cols = ["phi_mt_IV_HausmanFull_x_Tech"]
        possible_cols += ["phi_mt_IV_HausmanFull_x_Tech", "phi_mt_Tech"]

        tar_col = possible_cols[0]
        for p_col in possible_cols:
            if p_col in df_phi.columns:
                tar_col = p_col
                break

        c = color_map[label]
        ls2 = _SUBSET_LINESTYLE.get(label, '-')
        plot_label = _SUBSET_RENAME.get(label, label)
        crit_val = _t_crit(stage2_res.get(label))

        if tar_col in df_phi.columns:
            agg_b, se_b = calc_agg2(df_b, tar_col)
            agg_d, se_d = calc_agg2(df_d, tar_col)

            if not agg_b.empty:
                idx_dates = pd.PeriodIndex(agg_b.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
                axes2[0].plot(idx_dates, agg_b.values, label=plot_label, color=c, linewidth=2, linestyle=ls2)
                ci_b2.append((idx_dates, agg_b.values - crit_val * se_b.values, agg_b.values + crit_val * se_b.values, pastelize_color(c)))

            if not agg_d.empty:
                idx_dates = pd.PeriodIndex(agg_d.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
                axes2[1].plot(idx_dates, agg_d.values, label=plot_label, color=c, linewidth=2, linestyle=ls2)
                ci_d2.append((idx_dates, agg_d.values - crit_val * se_d.values, agg_d.values + crit_val * se_d.values, pastelize_color(c)))

    axes2[0].set_title("B-Type Firms (Spec 12, subset)", fontsize=14)
    axes2[0].set_ylabel(r"National $\hat{\phi}_t$")
    axes2[0].set_ylim(bottom=0)
    axes2[0].grid()
    axes2[0].legend(loc='lower left')

    axes2[1].set_title("D-Type Firms (Spec 12, subset)", fontsize=14)
    axes2[1].set_ylim(bottom=0)
    axes2[1].grid()
    axes2[1].legend(loc='lower left')

    fig2.tight_layout()
    plot_path2 = out_dir / "est1345_spec12_phi_t_comparison.png"
    plt.savefig(plot_path2, dpi=300)
    for x, lo, hi, pc in ci_b2:
        axes2[0].fill_between(x, lo, hi, color=pc, alpha=0.45)
    for x, lo, hi, pc in ci_d2:
        axes2[1].fill_between(x, lo, hi, color=pc, alpha=0.45)
    plot_path2_ci = out_dir / "est1345_spec12_phi_t_comparison_ci_pastel.png"
    plt.savefig(plot_path2_ci, dpi=300)
    plt.close(fig2)
    shutil.copy(plot_path2, _DRAFTS_DIR / "est1345_spec12_phi_t_comparison.png")
    shutil.copy(plot_path2_ci, _DRAFTS_DIR / "est1345_spec12_phi_t_comparison_ci_pastel.png")
    print(f"Exported subset plots to {plot_path2} and {plot_path2_ci}; copied to {_DRAFTS_DIR}")

    # ---- 5) Single-panel: Dummies Logistic, B and D in same axes ----
    _DUMMIES_LABEL = '5 Dummies Logistic'
    if _DUMMIES_LABEL in phi_data:
        df_dum = phi_data[_DUMMIES_LABEL]
        if 'year_quarter' in df_dum.columns:
            if 'dummy_D_type' in df_dum.columns:
                df_dum_b = df_dum[df_dum['dummy_D_type'] == 0]
                df_dum_d = df_dum[df_dum['dummy_D_type'] == 1]
            elif 'is_B' in df_dum.columns:
                df_dum_b = df_dum[df_dum['is_B'] == 1]
                df_dum_d = df_dum[df_dum['is_B'] == 0]
            else:
                df_dum_b = df_dum
                df_dum_d = pd.DataFrame(columns=df_dum.columns)

            possible_cols_dum = [
                "phi_mt_IV_HausmanFull_x_Tech_x_logistic",
                "phi_mt_IV_HausmanFull_x_Tech_logistic",
                "phi_mt_IV_HausmanFull_x_Tech",
                "phi_mt_Tech",
            ]
            tar_col_dum = next((c for c in possible_cols_dum if c in df_dum.columns), None)

            if tar_col_dum is not None:
                fig3, ax3 = plt.subplots(figsize=(10, 6))

                _res5 = stage2_res.get(_DUMMIES_LABEL)
                agg_b3, se_b3 = calc_agg_delta(df_dum_b, tar_col_dum, _res5, is_logistic=True)
                agg_d3, se_d3 = calc_agg_delta(df_dum_d, tar_col_dum, _res5, is_logistic=True)

                crit_val3 = _t_crit(stage2_res.get(_DUMMIES_LABEL))
                ci3: list = []
                if not agg_b3.empty:
                    idx_b = pd.PeriodIndex(agg_b3.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
                    ax3.plot(idx_b, agg_b3.values, label='B-Type Firms', color='steelblue', linewidth=2)
                    ci3.append((idx_b, agg_b3.values - crit_val3 * se_b3.values, agg_b3.values + crit_val3 * se_b3.values, pastelize_color('steelblue')))

                if not agg_d3.empty:
                    idx_d = pd.PeriodIndex(agg_d3.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
                    ax3.plot(idx_d, agg_d3.values, label='D-Type Firms', color='tomato', linewidth=2, linestyle='--')
                    ci3.append((idx_d, agg_d3.values - crit_val3 * se_d3.values, agg_d3.values + crit_val3 * se_d3.values, pastelize_color('tomato')))

                ax3.set_title(r"(+) Dummies — B vs D Firms (Spec 12)", fontsize=14)
                ax3.set_ylabel(r"National $\hat{\phi}_t$")
                ax3.set_ylim(bottom=0)
                ax3.axhline(1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.9)
                y_top = max(1.02, ax3.get_ylim()[1])
                ax3.set_ylim(0, y_top)
                current_ticks = list(ax3.get_yticks())
                if 1.0 not in current_ticks:
                    current_ticks.append(1.0)
                    ax3.set_yticks(sorted(current_ticks))
                ax3.grid()
                ax3.legend(loc='lower left')
                fig3.tight_layout()

                plot_path3 = out_dir / "est5_spec12_phi_t_BvD.png"
                plt.savefig(plot_path3, dpi=300)
                for x, lo, hi, pc in ci3:
                    ax3.fill_between(x, lo, hi, color=pc, alpha=0.45)
                plot_path3_ci = out_dir / "est5_spec12_phi_t_BvD_ci_pastel.png"
                plt.savefig(plot_path3_ci, dpi=300)
                plt.close(fig3)
                shutil.copy(plot_path3, _DRAFTS_DIR / "est5_spec12_phi_t_BvD.png")
                shutil.copy(plot_path3_ci, _DRAFTS_DIR / "est5_spec12_phi_t_BvD_ci_pastel.png")
                print(f"Exported B-vs-D dummies plots to {plot_path3} and {plot_path3_ci}; copied to {_DRAFTS_DIR}")

if __name__ == "__main__":
    main()









