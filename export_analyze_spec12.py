import os
import sys
from pathlib import Path
import pickle
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
sys.modules['estimation_5_sleep'] = type('FakeModule', (), {'NonLinearResults': NonLinearResults})
sys.modules['estimation_6_sleep'] = type('FakeModule', (), {'NonLinearResults': NonLinearResults})

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
        'nr_lagged_dep': r"Lagged Deposit Ratio",
        'interaction_gdp_per_capita': r"Lag Deposit Ratio $\times$ GDP per Capita",
        'interaction_cadunico_families_per1000': r"Lag Deposit Ratio $\times$ Families in CadÚnico per 1k",
        'interaction_fraction_65plus': r"Lag Deposit Ratio $\times$ Fraction $>65$ years",
        'interaction_fraction_young': r"Lag Deposit Ratio $\times$ Fraction $<25$ years",
        'interaction_risk_free_qoq_lag': r"Lag Deposit Ratio $\times$ Risk-free Rate (Lag)",
        'interaction_pix_users_pf_per1000': r"Lag Deposit Ratio $\times$ PIX Users per 1k",
        'interaction_connections_per100': r"Lag Deposit Ratio $\times$ Internet Connections per 100",
        'interaction_branches_per1000': r"Lag Deposit Ratio $\times$ Branches per 1k",
        'interaction_dummy_D_type': r"Lag Deposit Ratio $\times$ Digital Bank Indicator",
        'dummy_D_type': r"Digital Bank Indicator",
        'interaction_state_owned': r"Lag Deposit Ratio $\times$ State-owner Indicator",
        'interaction_cooperative': r"Lag Deposit Ratio $\times$ Cooperative Indicator",
        'post_2020': r"Post-2020 Indicator",
        'v_hat': r"1st Stage Control Function Residual",
        'constant': r"Constant"
    }
    return rename_dict.get(var, var.replace("_", r"\_"))

def build_latex_table(results_dict, order_keys, target_vars, out_path, title=""):
    tex = []
    tex.append(r"\documentclass{article}")
    tex.append(r"\usepackage{graphicx} % Required for inserting images")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\usepackage[para,online,flushleft]{threeparttable}")
    tex.append(r"\usepackage{natbib}")
    tex.append(r"\usepackage{rotating}")
    tex.append(r"\usepackage{geometry}")
    tex.append(r"\geometry{landscape, margin=1in}")
    tex.append(r"\begin{document}")
    tex.append(r"\begin{table}[ht]")
    tex.append(r"\centering")
    tex.append(r"\caption{" + title + r"}")
    tex.append(r"\begin{threeparttable}")
    
    col_def = "l" + "c" * len(order_keys)
    tex.append(r"\begin{tabular}{" + col_def + "}")
    tex.append(r"\toprule")
    
    headers = ["Variable"] + [k.replace("_", " ") for k in order_keys]
    tex.append(" & ".join(headers) + r" \\")
    tex.append(r"\midrule")
    
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
        
        # clusters
        clusters = "-"
        if hasattr(res, 'cov_kwds'):
            groups = res.cov_kwds.get('groups', None)
            if groups is not None:
                clusters = str(groups.nunique() if hasattr(groups, 'nunique') else len(set(groups)))
        
        # effective clusters
        g_star = getattr(res, 'G_star', getattr(res, 'df_resid', np.nan))
        
        row_nobs.append(f"{nobs:,.0f}" if pd.notna(nobs) else "-")
        row_r2.append(f"{r2:.3f}" if pd.notna(r2) else "-")
        row_fstat.append(f"{fstat:.3f}" if pd.notna(fstat) else "-")
        row_cluster.append(clusters)
        row_eff_cluster.append(f"{g_star:.1f}" if pd.notna(g_star) else "-")
        
    tex.append(" & ".join(row_nobs) + r" \\")
    tex.append(" & ".join(row_r2) + r" \\")
    tex.append(" & ".join(row_fstat) + r" \\")
    tex.append(" & ".join(row_cluster) + r" \\")
    tex.append(" & ".join(row_eff_cluster) + r" \\")
    
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\begin{tablenotes}")
    tex.append(r"\small")
    tex.append(r"\item \textit{Notes:} Standard errors are in parentheses. Significance levels: * $p < 0.1$, ** $p < 0.05$, *** $p < 0.01$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")
    tex.append(r"\end{document}")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))

def main():
    print("Collecting Estimation results for Spec 12 (IV_HausmanFull x Tech)...")
    _ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
    SLEEP_DIR = DATA_DIR / "ESTIMATION_OUTPUT"
    
    if not SLEEP_DIR.exists():
        print(f"ERROR: Cannot find {SLEEP_DIR}")
        return

    mapping = {
        '1_B_firms': SLEEP_DIR / "rout_1",
        '2_D_firms_Break': SLEEP_DIR / "rout_2",
        '3_B_firms_Robust': SLEEP_DIR / "rout_3" / "LOCAL", 
        '4_Pooled': SLEEP_DIR / "rout_4" / "POOLED",
        '5_Pooled_NLLS': SLEEP_DIR / "rout_5" / "POOLED",
        '6_Alt1': SLEEP_DIR / "rout_6" / "POOLED" / "ALT_1",
        '6_Alt2': SLEEP_DIR / "rout_6" / "POOLED" / "ALT_2",
    }

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
            '1_B_firms': 'IV_HausmanFull x Tech',
            '2_D_firms_Break': 'Option_2_IV_HausmanFull_Tech',
            '3_B_firms_Robust': 'IV_HausmanFull x Tech',
            '4_Pooled': 'IV_HausmanFull x Tech',
            '5_Pooled_NLLS': 'IV_HausmanFull x Tech',
            '6_Alt1': 'IV_HausmanFull x Tech x logistic', 
            '6_Alt2': 'IV_HausmanFull x Tech x logistic',
        }

        # Fallback if the script saved differently (linear vs logistic)
        tk = target_keys.get(label, 'IV_HausmanFull x Tech')
        spec_data = res_dict.get(tk)
        
        # If not found, try the linear key in 6
        if not spec_data and "6_Alt" in label:
            tk = 'IV_HausmanFull x Tech x linear'
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
    out_dir = DATA_DIR / "ANALYSIS" / "Spec12"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    order = list(mapping.keys())
    target_vars = [
        'nr_lagged_dep', 'interaction_gdp_per_capita', 'interaction_cadunico_families_per1000',
        'interaction_fraction_65plus', 'interaction_fraction_young', 'interaction_risk_free_qoq_lag',
        'interaction_pix_users_pf_per1000', 'interaction_connections_per100',
        'interaction_branches_per1000', 'interaction_dummy_D_type', 'interaction_state_owned',
        'interaction_cooperative', 'dummy_D_type'
    ]
    
    build_latex_table(stage1_res, order, target_vars, out_dir / "stage1_comparison.tex", title="First Stage IV Results across Specifications")
    build_latex_table(stage2_res, order, target_vars, out_dir / "stage2_comparison.tex", title="Second Stage Results across Specifications")
    print(f"Exported LaTeX Tables to {out_dir}")

    # ---- 2) Pickle Model Information ----
    with open(out_dir / "spec12_all_models.pkl", "wb") as f:
        pickle.dump(models_dict, f)
    print(f"Exported combined model instances to {out_dir / 'spec12_all_models.pkl'}")

    # ---- 3) Plot Implied National Phi_t ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    for label, df_phi in phi_data.items():
        if 'year_quarter' not in df_phi.columns:
            continue
            
        phi_col_b = f"phi_b" if "phi_b" in df_phi.columns else None
        phi_col_d = f"phi_d" if "phi_d" in df_phi.columns else None
        
        if 'dummy_D_type' in df_phi.columns:
            df_b = df_phi[df_phi['dummy_D_type'] == 0]
            df_d = df_phi[df_phi['dummy_D_type'] == 1]
        elif "2_D_firms" in label:
            df_b = pd.DataFrame(columns=df_phi.columns)
            df_d = df_phi
        else:
            df_b = df_phi
            df_d = pd.DataFrame(columns=df_phi.columns)

        def calc_agg(d_sub, col):
            if len(d_sub) == 0: return pd.Series(dtype=float)
            w = d_sub['market_size'] if 'market_size' in d_sub.columns else pd.Series(1.0, index=d_sub.index)
            num = (d_sub[col] * w).groupby(d_sub['year_quarter']).sum()
            den = w.groupby(d_sub['year_quarter']).sum()
            return num / den
        
        tar_col = "phi_mt_IV_HausmanFull_x_Tech"
        possible_cols = [
            "phi_mt_IV_HausmanFull_x_Tech", 
            "phi_mt_Option_2_IV_HausmanFull_Tech",
            "phi_mt_IV_HausmanFull_x_Tech_x_logistic",
            "phi_mt_Tech"
        ]
        
        for p_col in possible_cols:
            if p_col in df_phi.columns:
                tar_col = p_col
                break
                
        if tar_col in df_phi.columns:
            agg_b = calc_agg(df_b, tar_col)
            agg_d = calc_agg(df_d, tar_col)
            
            if not agg_b.empty:
                idx_dates = pd.PeriodIndex(agg_b.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
                axes[0].plot(idx_dates, agg_b.values, label=label, linewidth=2)
            if not agg_d.empty:
                idx_dates = pd.PeriodIndex(agg_d.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
                axes[1].plot(idx_dates, agg_d.values, label=label, linewidth=2)
                
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
    plot_path = out_dir / "phi_t_comparison_spec12.png"
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Exported combined plot to {plot_path}")

if __name__ == "__main__":
    main()