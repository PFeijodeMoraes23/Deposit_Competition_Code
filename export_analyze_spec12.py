import os
import sys
import shutil
from pathlib import Path
import pickle

_DRAFTS_DIR = Path(r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Drafts\Deposit Competition")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Mock NonLinearResults class for unpickling
class NonLinearResults:
    def __init__(self, params, bse, tvalues, pvalues, df_resid):
        self.params = params
        self.bse = bse
        self.tvalues = tvalues
        self.pvalues = pvalues
        self.df_resid = df_resid
        self.G_star = df_resid

# Register fake module for unpickling
sys.modules['estimation_2_sleep'] = type('FakeModule', (), {'NonLinearResults': NonLinearResults})
sys.modules['estimation_2_sleep'] = type('FakeModule', (), {'NonLinearResults': NonLinearResults})

# Try to respect the project's venv guard
try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except ImportError:
    pass

def get_stars(pval):
    if pd.isna(pval): return ""
    if pval < 0.01: return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.1: return "*"
    return ""

def format_value(coef, se, pval):
    if pd.isna(coef):
        return "-", "-"
    stars = get_stars(pval)
    return f"{coef:.4f}{stars}", f"({se:.4f})"

def nice_var_name(var):
    rename_dict = {
        'nr_lagged_dep': r"Constant",
        'interaction_gdp_per_capita': r"GDP per Capita",
        'interaction_cadunico_families_per1000': r"Families in CadÚnico per 1k",
        'interaction_fraction_65plus': r"Fraction $>65$ years",
        'interaction_fraction_young': r"Fraction $<25$ years",
        'interaction_risk_free_qoq_lag': r"Risk-free Rate (Lag)",
        'interaction_pix_users_pf_per1000': r"PIX Users per 1k",
        'interaction_connections_per100': r"Internet Connections per 100",
        'interaction_branches_per1000': r"Branches per 1k",
        'interaction_dummy_D_type': r"Digital Bank Indicator",
        'dummy_D_type': r"Digital Bank Indicator",
        'interaction_state_owned': r"State-owner Indicator",
        'interaction_cooperative': r"Cooperative Indicator",
        'interaction_pix_exists': r"PIX Exists Indicator",
        'interaction_post_2020': r"Post-2020 Indicator",
        'post_2020': r"Post-2020 Indicator",
        'pix_exists': r"PIX Exists Indicator",
        'v_hat': r"1st Stage Control Function Residual",
        'constant': r"Constant"
    }
    return rename_dict.get(var, var.replace("_", r"\_").replace("interaction\_", ""))

def build_latex_table(results_dict, order_keys, target_vars, out_path, title=""):
    tex = []
    tex.append(r"\documentclass{article}")
    tex.append(r"\usepackage{graphicx} % Required for inserting images")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\usepackage{longtable}")
    tex.append(r"\usepackage{natbib}")
    tex.append(r"\usepackage{rotating}")
    tex.append(r"\usepackage{geometry}")
    tex.append(r"\geometry{landscape, margin=1in}")
    tex.append(r"\begin{document}")
    
    # Reduce font size and line spacing
    tex.append(r"\footnotesize")
    tex.append(r"\renewcommand{\arraystretch}{0.75}")
    
    col_def = "l" + "c" * len(order_keys)
    tex.append(r"\begin{longtable}{" + col_def + "}")
    tex.append(r"\caption{" + title + r"} \\")
    
    # First Header
    tex.append(r"\toprule")
    headers = ["Variable"] + [k for k in order_keys]
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
    notes_str = r"\multicolumn{" + str(len(order_keys) + 1) + r"}{p{\textwidth}}{\footnotesize\textit{Notes:} Standard errors are in parentheses. Significance levels: * $p < 0.1$, ** $p < 0.05$, *** $p < 0.01$.}"
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
    ordered_vars += [v for v in vars_to_print if v not in ordered_vars]
    
    for v in ordered_vars:
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
                c_str, se_str = format_value(params[v], bse[v], pvalues[v])
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
    row_cluster = ["Clusters"]
    row_eff_cluster = ["Effective Clusters ($G^*$)"]
    
    for col in order_keys:
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
    tex.append(" & ".join(row_fstat) + r" \\")
    tex.append(" & ".join(row_cluster) + r" \\")
    tex.append(" & ".join(row_eff_cluster) + r" \\")
    
    tex.append(r"\end{longtable}")
    tex.append(r"\end{document}")
    
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
        '2 B firms Robust': SLEEP_DIR / "DEMAND_PREP" / "est2" / "LOCAL", 
        '3 Pooled': SLEEP_DIR / "DEMAND_PREP" / "est3",
        '4 Pooled Logistic': SLEEP_DIR / "DEMAND_PREP" / "est4",
        '5 Dummies Linear': SLEEP_DIR / "DEMAND_PREP" / "est5",
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
            '2 B firms Robust': 'IV_HausmanFull x Tech',
            '3 Pooled': 'IV_HausmanFull x Tech',
            '4 Pooled Logistic': 'IV_HausmanFull x Tech',
            '5 Dummies Linear': 'IV_HausmanFull x Tech x linear', 
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
    target_vars = [
        'nr_lagged_dep', 'interaction_gdp_per_capita', 'interaction_cadunico_families_per1000',
        'interaction_fraction_65plus', 'interaction_fraction_young', 'interaction_risk_free_qoq_lag',
        'interaction_pix_users_pf_per1000', 'interaction_connections_per100',
        'interaction_branches_per1000', 'interaction_dummy_D_type', 'interaction_state_owned',
        'interaction_cooperative', 'dummy_D_type'
    ]
    
    build_latex_table(stage1_res, order, target_vars, out_dir / "est1-5_spec12_stage1_comparison.tex", title="First Stage IV Results across Specifications")
    build_latex_table(stage2_res, order, target_vars, out_dir / "est1-5_spec12_stage2_comparison.tex", title="Second Stage Results across Specifications")
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
    
    label_rename_map = {}
    
    for label, df_phi in phi_data.items():
        if 'year_quarter' not in df_phi.columns:
            continue
            
        phi_col_b = f"phi_b" if "phi_b" in df_phi.columns else None
        phi_col_d = f"phi_d" if "phi_d" in df_phi.columns else None
        
        if 'dummy_D_type' in df_phi.columns:
            # When pooled and has the dummy, we can separate
            df_b = df_phi[df_phi['dummy_D_type'] == 0]
            df_d = df_phi[df_phi['dummy_D_type'] == 1]
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

        def calc_agg(d_sub, col):
            if len(d_sub) == 0: return pd.Series(dtype=float), pd.Series(dtype=float)
            w = d_sub.get('market_size', pd.Series(1.0, index=d_sub.index))
            num = (d_sub[col] * w).groupby(d_sub['year_quarter']).sum()
            den = w.groupby(d_sub['year_quarter']).sum()
            mean = num / den
            
            # Compute weighted variance and standard error of the mean
            if len(d_sub) > 1:
                # Merge mean back to compute variance
                merged = d_sub[['year_quarter', col]].copy()
                merged['w'] = w
                merged['mean'] = merged['year_quarter'].map(mean)
                # Weighted variance
                var_num = (merged['w'] * (merged[col] - merged['mean'])**2).groupby(merged['year_quarter']).sum()
                # Unbiased weighted variance (reliability weights)
                v1 = merged['w'].groupby(merged['year_quarter']).sum()
                v2 = (merged['w']**2).groupby(merged['year_quarter']).sum()
                var = var_num / (v1 - (v2 / v1))
                
                # Standard error of the mean (weighted)
                n = d_sub.groupby('year_quarter').size()
                se = np.sqrt(var / n)
            else:
                se = pd.Series(0.0, index=mean.index)
                
            return mean, se
        
        tar_col = "phi_mt_IV_HausmanFull_x_Tech"
        
        # Explicitly map target columns based on spec label to prevent collisions
        if "Linear" in label:
            possible_cols = ["phi_mt_IV_HausmanFull_x_Tech_x_linear", "phi_mt_IV_HausmanFull_x_Tech_linear"]
        elif "Logistic" in label:
            possible_cols = ["phi_mt_IV_HausmanFull_x_Tech_x_logistic", "phi_mt_IV_HausmanFull_x_Tech_logistic"]
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
        
        if tar_col in df_phi.columns:
            agg_b, se_b = calc_agg(df_b, tar_col)
            agg_d, se_d = calc_agg(df_d, tar_col)
            
            # Use 1.96 standard errors for approx 95% CI
            if not agg_b.empty:
                idx_dates = pd.PeriodIndex(agg_b.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
                axes[0].plot(idx_dates, agg_b.values, label=plot_label, color=c, linewidth=2)
                axes[0].fill_between(idx_dates, agg_b.values - (1.96*se_b.values), agg_b.values + (1.96*se_b.values), color=c, alpha=0.2)
                
            if not agg_d.empty:
                idx_dates = pd.PeriodIndex(agg_d.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
                axes[1].plot(idx_dates, agg_d.values, label=plot_label, color=c, linewidth=2)
                axes[1].fill_between(idx_dates, agg_d.values - (1.96*se_d.values), agg_d.values + (1.96*se_d.values), color=c, alpha=0.2)
                
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
    plt.close(fig)
    shutil.copy(plot_path, _DRAFTS_DIR / "est1-5_spec12_phi_t_comparison.png")
    print(f"Exported combined plot to {plot_path} and copied to {_DRAFTS_DIR}")

    # ---- 4) Subset plot: 1 B firms, 3 Pooled, 4 Pooled Logistic, 5 Dummies Logistic ----
    _SUBSET_LABELS = {'1 B firms', '3 Pooled', '4 Pooled Logistic', '5 Dummies Logistic'}
    _SUBSET_RENAME = {
        '1 B firms': 'B Data',
        '3 Pooled': 'B + D Pooled',
        '4 Pooled Logistic': 'Pooled Logistic',
        '5 Dummies Logistic': 'Pooled + Dummy Logistic',
    }
    phi_data_sub = {k: v for k, v in phi_data.items() if k in _SUBSET_LABELS}

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for label, df_phi in phi_data_sub.items():
        if 'year_quarter' not in df_phi.columns:
            continue

        if 'dummy_D_type' in df_phi.columns:
            df_b = df_phi[df_phi['dummy_D_type'] == 0]
            df_d = df_phi[df_phi['dummy_D_type'] == 1]
        elif 'is_B' in df_phi.columns:
            df_b = df_phi[df_phi['is_B'] == 1]
            df_d = df_phi[df_phi['is_B'] == 0]
        else:
            df_b = df_phi
            df_d = pd.DataFrame(columns=df_phi.columns)

        def calc_agg2(d_sub, col):
            if len(d_sub) == 0: return pd.Series(dtype=float), pd.Series(dtype=float)
            w = d_sub.get('market_size', pd.Series(1.0, index=d_sub.index))
            num = (d_sub[col] * w).groupby(d_sub['year_quarter']).sum()
            den = w.groupby(d_sub['year_quarter']).sum()
            mean = num / den
            if len(d_sub) > 1:
                merged = d_sub[['year_quarter', col]].copy()
                merged['w'] = w
                merged['mean'] = merged['year_quarter'].map(mean)
                var_num = (merged['w'] * (merged[col] - merged['mean'])**2).groupby(merged['year_quarter']).sum()
                v1 = merged['w'].groupby(merged['year_quarter']).sum()
                v2 = (merged['w']**2).groupby(merged['year_quarter']).sum()
                var = var_num / (v1 - (v2 / v1))
                n = d_sub.groupby('year_quarter').size()
                se = np.sqrt(var / n)
            else:
                se = pd.Series(0.0, index=mean.index)
            return mean, se

        if "Logistic" in label:
            possible_cols = ["phi_mt_IV_HausmanFull_x_Tech_x_logistic", "phi_mt_IV_HausmanFull_x_Tech_logistic"]
        elif "Linear" in label:
            possible_cols = ["phi_mt_IV_HausmanFull_x_Tech_x_linear", "phi_mt_IV_HausmanFull_x_Tech_linear"]
        else:
            possible_cols = ["phi_mt_IV_HausmanFull_x_Tech"]
        possible_cols += ["phi_mt_IV_HausmanFull_x_Tech", "phi_mt_Tech"]

        tar_col = possible_cols[0]
        for p_col in possible_cols:
            if p_col in df_phi.columns:
                tar_col = p_col
                break

        c = color_map[label]
        plot_label = _SUBSET_RENAME.get(label, label)

        if tar_col in df_phi.columns:
            agg_b, se_b = calc_agg2(df_b, tar_col)
            agg_d, se_d = calc_agg2(df_d, tar_col)

            if not agg_b.empty:
                idx_dates = pd.PeriodIndex(agg_b.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
                axes2[0].plot(idx_dates, agg_b.values, label=plot_label, color=c, linewidth=2)
                axes2[0].fill_between(idx_dates, agg_b.values - (1.96*se_b.values), agg_b.values + (1.96*se_b.values), color=c, alpha=0.2)

            if not agg_d.empty:
                idx_dates = pd.PeriodIndex(agg_d.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
                axes2[1].plot(idx_dates, agg_d.values, label=plot_label, color=c, linewidth=2)
                axes2[1].fill_between(idx_dates, agg_d.values - (1.96*se_d.values), agg_d.values + (1.96*se_d.values), color=c, alpha=0.2)

    axes2[0].set_title("B-Type Firms (Spec 12, subset)", fontsize=14)
    axes2[0].set_ylabel(r"National $\hat{\phi}_t$")
    axes2[0].set_ylim(bottom=0)
    axes2[0].grid()
    axes2[0].legend(loc='best')

    axes2[1].set_title("D-Type Firms (Spec 12, subset)", fontsize=14)
    axes2[1].set_ylim(bottom=0)
    axes2[1].grid()
    axes2[1].legend(loc='best')

    fig2.tight_layout()
    plot_path2 = out_dir / "est1345_spec12_phi_t_comparison.png"
    plt.savefig(plot_path2, dpi=300)
    plt.close(fig2)
    shutil.copy(plot_path2, _DRAFTS_DIR / "est1345_spec12_phi_t_comparison.png")
    print(f"Exported subset plot to {plot_path2} and copied to {_DRAFTS_DIR}")

if __name__ == "__main__":
    main()









