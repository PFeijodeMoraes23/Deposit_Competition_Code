import os
import sys
import shutil
from pathlib import Path
import pickle

_DRAFTS_DIR = Path(r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Drafts\Deposit Competition")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

# Mock NonLinearResults for unpickling estimation_3_sleep pickles.
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

# Register fake module for unpickling est3 (Pooled Logistic) pickles.
sys.modules['estimation_3_sleep'] = type('FakeModule', (), {'NonLinearResults': NonLinearResults})

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

# market_panel_phis.csv stores demographic/infrastructure columns already
# in the rescaled units used for estimation (gdp/10k, cadunico/100, etc.)
_SCALE_COLS: dict = {}


def _build_phi_regressors(df_sub: pd.DataFrame, phi_params) -> np.ndarray:
    """
    Build regressor matrix X (n_obs × len(phi_params)) from market panel columns.
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
    Compute market_size-weighted national phi_t and its analytical (delta-method) SE.

    SE²_t = X̄_t' Σ X̄_t  where X̄_t = pop-weighted mean regressor vector for period t.

    Linear models  : Σ = cov_OLS (params = OLS beta)
    Logistic models: Σ = cov_AME (params = AME). The phi*(1-phi) scaling is already
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

    full_cov = res.cov_params()
    p_idx = [j for j, p in enumerate(all_idx) if p in set(phi_params)]
    Sigma = full_cov.values[np.ix_(p_idx, p_idx)]

    X = _build_phi_regressors(d_sub, phi_params)
    yq_arr = d_sub['year_quarter'].values

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
        'connections_per100': 'Broadband Connections (per 100 inhabitants)',
        'pix_exists': 'Pix Available',
        'v_hat_x_lagged_dep': 'CF: $\\hat{v} \\times$ Lagged Deposits',
        'const': 'Constant',
        'constant': 'Constant',
        'tax_cost_ratio_lag': 'Tax Cost Ratio ($t-1$)',
        'personnel_cost_ratio_lag': 'Personnel Cost Ratio ($t-1$)',
        'admin_cost_ratio_lag': 'Admin Cost Ratio ($t-1$)',
        'indice_basileia_lag': 'Basel Index (pp, $t-1$)',
        'lci_lca_ratio_lag': 'LCI/LCA Ratio ($t-1$)',
        'wholesale_ratio_lag': 'Wholesale Ratio ($t-1$)',
        'leave_one_out_mean_spread': 'Leave-out Mean Spread',
    }
    return labels.get(v, v.replace('_', '\\_'))

def build_latex_table(results_dict, order_keys, target_vars, out_path, title="", label=""):
    tex = []

    # \setstretch{1.0} matches the paper's other tables (est1_first_stage_table.tex
    # etc.), all of which open with \setstretch{1.0} and no wrapping group.
    # Font size (\footnotesize) and \arraystretch (1.08) are applied automatically
    # by the document preamble via \AtBeginEnvironment{xltabular}{\footnotesize}
    # and \renewcommand{\arraystretch}{1.08}, so we do not override them locally.
    tex.append(r"\setstretch{1.0}")

    # xltabular pins the table to \textwidth and distributes the remaining width
    # equally among the X data columns (same as tabularx but supports longtable
    # headers/footers). The first column is a fixed-width raggedright p column so
    # long labels (e.g. "Broadband Connections (per 100 inhabitants)") wrap rather
    # than forcing the table past the text block. 0.26\textwidth leaves enough room
    # for "Pooled (Logistic AME)" to fit on one line in each X column.
    n_data = len(order_keys)
    col_def = (r">{\raggedright\arraybackslash}p{0.26\textwidth} "
               r"*{" + str(n_data) + r"}{>{\centering\arraybackslash}X}")
    tex.append(r"\begin{xltabular}{\textwidth}{" + col_def + "}")
    tex.append(r"\caption{" + title + r"}\label{" + label + r"} \\")

    # First Header
    tex.append(r"\toprule")
    rename_map = {
        '1 Local': 'Local Only',
        '2 Pooled Linear': 'Pooled (Linear)',
        '3 Pooled Logistic': 'Pooled (Logistic AME)',
    }
    headers = ["Variable"] + [rename_map.get(k, k) for k in order_keys]
    tex.append(" & ".join(headers) + r" \\")
    tex.append(r"\midrule")
    tex.append(r"\endfirsthead")

    # Next Headers
    # NOTE: keep this continuation line short. longtable computes column widths
    # from *every* head/foot row, so embedding the full (long) title here forces
    # an unbreakable \multicolumn wider than the page, which balloons the columns
    # and pushes the rules off the right margin even on the first page.
    tex.append(r"\multicolumn{" + str(len(order_keys) + 1) + r"}{c}{{\bfseries \tablename\ \thetable{} (continued from previous page)}} \\")
    tex.append(r"\toprule")
    tex.append(" & ".join(headers) + r" \\")
    tex.append(r"\midrule")
    tex.append(r"\endhead")

    # Footers
    tex.append(r"\midrule")
    tex.append(r"\multicolumn{" + str(len(order_keys) + 1) + r"}{r}{{Continued on next page}} \\")
    tex.append(r"\endfoot")

    # Last Footer — notes style matches the paper's other sleep tables:
    # \scriptsize font, stars in descending order (***/**/*), p{} column type.
    tex.append(r"\bottomrule")
    # \dimexpr\textwidth-2\tabcolsep\relax is exactly the usable width of a
    # full-span multicolumn in a \textwidth-wide xltabular: the table occupies
    # \textwidth, but the outer \tabcolsep margins on left and right eat 2*3.5pt=7pt,
    # leaving \textwidth-7pt for the cell content.
    notes_str = (r"\multicolumn{" + str(len(order_keys) + 1) + r"}{p{\dimexpr\textwidth-2\tabcolsep\relax}}"
                 r"{\scriptsize\textit{Notes:} Standard errors are in parentheses. "
                 r"Est.~3 reports Average Marginal Effects (AME) from NLLS logistic, "
                 r"following \textcite{imbens2016robust} and \textcite{carter2017asymptotic}. "
                 r"Significance levels: *** $p<0.01$, ** $p<0.05$, * $p<0.1$.}")
    tex.append(notes_str)
    tex.append(r"\endlastfoot")

    # Collect all unique vars from loaded results
    vars_to_print = []
    for col in order_keys:
        res = results_dict.get(col)
        if res is not None:
            params = getattr(res, 'params', pd.Series(dtype=float))
            for v in params.index:
                if v not in vars_to_print:
                    vars_to_print.append(v)

    # Target vars first, then any remaining (excluding CF nuisance term)
    ordered_vars = [v for v in target_vars if v in vars_to_print]

    is_first_stage = ("first_stage" in str(out_path).lower() or "stage1" in str(out_path).lower())

    if not is_first_stage:
        other_vars = [v for v in vars_to_print if v not in ordered_vars]
        exclude_patterns = ["v_hat"]
        other_vars = [v for v in other_vars if not any(pattern in v.lower() for pattern in exclude_patterns)]
        ordered_vars += other_vars
        # Never show the control-function term in the second-stage table
        ordered_vars = [v for v in ordered_vars if 'v_hat' not in v.lower()]

    for v in ordered_vars:
        # Two table rows per variable: label spans both via \multirow[t]{2} so
        # it stays anchored even when it wraps. We give the explicit column width
        # (0.26\textwidth, matching the p-column spec) rather than = (infer) because
        # in V_Main.tex's \doublespacing context = computes 7pt wider than the column.
        # \\* on the coeff row forbids a page break between coefficient and SE.
        label_cell = r"\multirow[t]{2}{0.26\textwidth}{\raggedright " + nice_var_name(v) + r"}"
        row_cf = [label_cell]
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

        tex.append(" & ".join(row_cf) + r" \\*")
        tex.append(" & ".join(row_se) + r" \\")
        tex.append(r"\addlinespace")

    tex.append(r"\midrule")

    row_nobs = ["Observations"]
    row_r2 = ["$R^2$"]
    row_fstat = ["F-Statistic"]
    row_cluster = ["Clusters ($G$)"]
    row_eff_cluster = ["Effective Clusters ($G^*$)"]

    # Logistic est3 borrows N and R² from the linear est2 counterpart
    _nlls_fallback = {'3 Pooled Logistic': '2 Pooled Linear'}

    for col in order_keys:
        res = results_dict.get(col)
        if res is None:
            row_nobs.append("-"); row_r2.append("-"); row_fstat.append("-")
            row_cluster.append("-"); row_eff_cluster.append("-")
            continue

        nobs  = getattr(res, 'nobs', np.nan)
        r2    = getattr(res, 'rsquared', np.nan)
        fstat = getattr(res, 'fvalue', np.nan)
        fpval = getattr(res, 'f_pvalue', np.nan)

        if pd.isna(nobs) or pd.isna(r2):
            fallback_col = _nlls_fallback.get(col)
            if fallback_col and fallback_col in results_dict:
                f_res = results_dict[fallback_col]
                if f_res is not None:
                    if pd.isna(nobs):  nobs  = getattr(f_res, 'nobs', np.nan)
                    if pd.isna(r2):    r2    = getattr(f_res, 'rsquared', np.nan)
                    if pd.isna(fstat): fstat = getattr(f_res, 'fvalue', np.nan)
                    if pd.isna(fpval): fpval = getattr(f_res, 'f_pvalue', np.nan)

        clusters = "-"
        if hasattr(res, 'cov_kwds') and res.cov_kwds.get('groups', None) is not None:
            groups = res.cov_kwds.get('groups', None)
            clusters = str(groups.nunique() if hasattr(groups, 'nunique') else len(set(groups)))
        elif hasattr(res, 'G_nominal') and not pd.isna(getattr(res, 'G_nominal', np.nan)):
            clusters = str(int(res.G_nominal))

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

    tex.append(r"\end{xltabular}")
    # Restore the document's double spacing; \setstretch{1.0} at the top of the
    # table suppressed it, so without this the following body text stays single-spaced.
    tex.append(r"\doublespacing")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))

import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze Specification 12 Results (Est 1-3)")
    parser.add_argument('--skip-est2', action='store_true', help='Skip estimation 2 (Pooled Linear)')
    args = parser.parse_args()

    print("Collecting Estimation results for Spec 12 (IV_HausmanFull x Tech)...")
    _ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
    SLEEP_DIR = DATA_DIR / "ESTIMATION_OUTPUT"

    if not SLEEP_DIR.exists():
        print(f"ERROR: Cannot find {SLEEP_DIR}")
        return

    mapping = {
        '1 Local':           SLEEP_DIR / "DEMAND_PREP" / "est1",
        '2 Pooled Linear':   SLEEP_DIR / "DEMAND_PREP" / "est2",
        '3 Pooled Logistic': SLEEP_DIR / "DEMAND_PREP" / "est3",
    }

    if getattr(args, 'skip_est2', False):
        mapping.pop('2 Pooled Linear', None)

    # All three estimations store spec 12 under the same key
    target_keys = {
        '1 Local':           'IV_HausmanFull x Tech',
        '2 Pooled Linear':   'IV_HausmanFull x Tech',
        '3 Pooled Logistic': 'IV_HausmanFull x Tech',
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

        tk = target_keys.get(label, 'IV_HausmanFull x Tech')
        spec_data = res_dict.get(tk)

        if not spec_data:
            print(f"  [Warning] '{tk}' not found in {label}, skipping.")
            continue

        models_dict[label] = spec_data

        if spec_data.get('first_stage') is not None:
            stage1_res[label] = spec_data['first_stage']
        if spec_data.get('second_stage') is not None:
            stage2_res[label] = spec_data['second_stage']

        csv_path = d / "market_panel_phis.csv"
        if csv_path.exists():
            try:
                phi_data[label] = pd.read_csv(csv_path, low_memory=False)
            except Exception as e:
                print(f"  [Warning] Could not load phi CSV for {label}: {e}")

    # ---- 1) Export LaTeX Tables ----
    out_dir = SLEEP_DIR / "Rout"
    out_dir.mkdir(parents=True, exist_ok=True)

    order = list(mapping.keys())

    target_vars = [
        'nr_lagged_dep',
        'interaction_gdp_per_capita',
        'interaction_cadunico_families_per1000',
        'interaction_fraction_65plus',
        'interaction_fraction_young',
        'interaction_risk_free_qoq_lag',
        'interaction_connections_per100',
        'interaction_pix_exists',
        'v_hat_x_lagged_dep',
    ]

    first_stage_target_vars = [
        'tax_cost_ratio_lag', 'personnel_cost_ratio_lag', 'admin_cost_ratio_lag',
        'indice_basileia_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag',
        'leave_one_out_mean_spread',
    ]

    build_latex_table(
        stage1_res, order, first_stage_target_vars,
        out_dir / "est1-3_spec12_stage1_comparison.tex",
        title="First Stage IV Results across Specifications (Spec 12)",
        label="tab:spec12_stage1_comparison",
    )
    build_latex_table(
        stage2_res, order, target_vars,
        out_dir / "est1-3_spec12_stage2_comparison.tex",
        title="Second Stage Results across Specifications (Spec 12)",
        label="tab:spec12_stage2_comparison",
    )
    shutil.copy(out_dir / "est1-3_spec12_stage1_comparison.tex", _DRAFTS_DIR / "est1-3_spec12_stage1_comparison.tex")
    shutil.copy(out_dir / "est1-3_spec12_stage2_comparison.tex", _DRAFTS_DIR / "est1-3_spec12_stage2_comparison.tex")
    print(f"Exported LaTeX tables to {out_dir} and copied to {_DRAFTS_DIR}")

    # ---- 2) Pickle Model Information ----
    with open(out_dir / "est1-3_spec12_all_models.pkl", "wb") as f:
        pickle.dump(models_dict, f)
    print(f"Exported combined model instances to {out_dir / 'est1-3_spec12_all_models.pkl'}")

    # ---- 3) Plot Implied National Phi_t (single panel, all 3 strategies) ----
    #
    # Est1 saves phi columns as phi_mt_{block} (e.g. phi_mt_Tech).
    # Est2/3 save phi columns as phi_mt_{safe_key} where safe_key replaces spaces
    # with '_', so "IV_HausmanFull x Tech" → phi_mt_IV_HausmanFull_x_Tech.
    phi_col_map = {
        '1 Local':           'phi_mt_Tech',
        '2 Pooled Linear':   'phi_mt_IV_HausmanFull_x_Tech',
        '3 Pooled Logistic': 'phi_mt_IV_HausmanFull_x_Tech',
    }

    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {lbl: base_colors[i % len(base_colors)] for i, lbl in enumerate(mapping.keys())}
    linestyle_map = {
        '1 Local':           '-',
        '2 Pooled Linear':   '-',
        '3 Pooled Logistic': '--',
    }
    label_rename = {
        '1 Local':           'Local Only (Est 1)',
        '2 Pooled Linear':   'Pooled Linear (Est 2)',
        '3 Pooled Logistic': 'Pooled Logistic (Est 3)',
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    ci_list: list = []

    for label, df_phi in phi_data.items():
        if 'year_quarter' not in df_phi.columns:
            continue

        # Prefer the pre-specified phi column; fall back to phi_mt_Tech
        preferred = phi_col_map.get(label, 'phi_mt_Tech')
        fallbacks = [preferred, 'phi_mt_IV_HausmanFull_x_Tech', 'phi_mt_Tech']
        tar_col = next((c for c in fallbacks if c in df_phi.columns), None)
        if tar_col is None:
            print(f"  [Warning] No phi column found for {label}, skipping plot.")
            continue

        res = stage2_res.get(label)
        is_logistic = 'Logistic' in label
        crit_val = _t_crit(res)

        agg, se = calc_agg_delta(df_phi, tar_col, res, is_logistic)
        if agg.empty:
            continue

        idx_dates = pd.PeriodIndex(agg.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
        c = color_map[label]
        ls = linestyle_map.get(label, '-')
        plot_label = label_rename.get(label, label)

        ax.plot(idx_dates, agg.values, label=plot_label, color=c, linewidth=2, linestyle=ls)
        ci_list.append((
            idx_dates,
            agg.values - crit_val * se.values,
            agg.values + crit_val * se.values,
            pastelize_color(c),
        ))

    ax.set_title(r"Implied National $\hat{\phi}_t$ — Spec 12 (IV Hausman $\times$ Tech)", fontsize=14, pad=12)
    ax.set_ylabel(r"National $\hat{\phi}_t$")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.grid(alpha=0.4)
    ax.legend(loc='best')
    for x, lo, hi, pc in ci_list:
        ax.fill_between(x, lo, hi, color=pc, alpha=0.45)
    fig.tight_layout()

    # The figure already carries the pastel-toned confidence-interval bands
    # (fill_between with pastelize_color above). V_Main.tex includes the
    # "_ci_pastel" filename, so that is the canonical output name. We also write
    # the plain name for backward compatibility with any other references.
    plot_name = "est1-3_spec12_phi_t_comparison_ci_pastel.png"
    plot_path = out_dir / plot_name
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    shutil.copy(plot_path, _DRAFTS_DIR / plot_name)
    shutil.copy(plot_path, out_dir / "est1-3_spec12_phi_t_comparison.png")
    shutil.copy(plot_path, _DRAFTS_DIR / "est1-3_spec12_phi_t_comparison.png")
    print(f"Exported phi_t plot (pastel CIs) to {plot_path}; copied to {_DRAFTS_DIR}")

if __name__ == "__main__":
    main()
