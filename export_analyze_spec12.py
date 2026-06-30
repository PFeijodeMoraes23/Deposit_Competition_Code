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
        'gdp_growth_yoy': 'GDP Growth (YoY)',
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

def build_latex_table(results_dict, order_keys, target_vars, out_path, title="", label="",
                      mean_phi=None):
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
        '1 Local': 'Local (Lin.)',
        '2 Pooled Linear': 'Pooled (Lin.)',
        '5 Single-Index': 'Single-Idx (AME)',
        '6 Single-Index Time': 'Single-Idx +T (AME)',
        '7 Joint Sieve': 'Joint Sieve (AME)',
        '8 Joint Sieve Time': 'Joint Sieve +T (AME)',
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
                 r"{\scriptsize\textit{Notes:} Standard errors (score/multiplier wild cluster "
                 r"bootstrap at the conglomerate level, Webb 6-point $B=999$, applied uniformly to "
                 r"every column; \textcite{cameron2008bootstrap}, "
                 r"\textcite{mackinnon2017wild}) in parentheses. The Local and Pooled columns report "
                 r"linear sleepiness coefficients; the Single-Index and Joint Sieve columns report "
                 r"average marginal effects (AME). The single-index/joint estimators carry no "
                 r"constant AME --- the baseline level is absorbed into the monotone link --- so the "
                 r"`Mean $\hat{\phi}$' row gives the comparable implied level across all columns. "
                 r"\textcite{carter2017asymptotic} effective clusters $G^*$ are a diagnostic. "
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

    # Implied national mean phi: the comparable "level" across linear / single-index / joint.
    # The single-index/joint estimators carry no constant AME (the level is in the monotone
    # link), so this row gives the interpretable baseline sleepiness for every column.
    if mean_phi is not None and not is_first_stage:
        row_meanphi = [r"Mean $\hat{\phi}$ (level)"]
        for col in order_keys:
            mp = mean_phi.get(col)
            row_meanphi.append(f"{mp:.3f}" if mp is not None and pd.notna(mp) else "-")
        tex.append(" & ".join(row_meanphi) + r" \\")

    row_nobs = ["Observations"]
    row_r2 = ["$R^2$"]
    row_fstat = ["F-Statistic"]
    row_cluster = ["Clusters ($G$)"]
    row_eff_cluster = ["Effective Clusters ($G^*$)"]

    # E5-E8 (single-index/joint) carry their own nobs/rsquared; no fallback needed.
    _nlls_fallback: dict = {}

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


def build_latex_table_landscape(results_dict, order_keys, target_vars, out_path,
                                title="", label="", mean_phi=None, placement="ht",
                                first_stage=False):
    """Landscape variant of the stage-2 comparison for V_Main inclusion.

    Follows the house landscape style (blp_compare_E*_spec12.tex): a real
    `table` float wrapped in pdflscape's `landscape`, with booktabs rules and
    threeparttable notes -- all already loaded by V_Main.tex. The float takes a
    SOFT placement specifier (default [ht], NOT a hard [H]) so it settles around
    the \\input location. Carries the SAME \\label as the portrait table, so
    swapping the \\input in V_Main keeps every \\ref resolving. Natural centred
    columns (not xltabular's X) + two-line \\shortstack headers keep the seven
    columns readable across the rotated page.
    """
    n = len(order_keys)
    rename_map = {
        '1 Local':             r'\shortstack{Local\\(Lin.)}',
        '2 Pooled Linear':     r'\shortstack{Pooled\\(Lin.)}',
        '5 Single-Index':      r'\shortstack{Single-Idx\\(AME)}',
        '6 Single-Index Time': r'\shortstack{Single-Idx $+$T\\(AME)}',
        '7 Joint Sieve':       r'\shortstack{Joint Sieve\\(AME)}',
        '8 Joint Sieve Time':  r'\shortstack{Joint Sieve $+$T\\(AME)}',
    }
    headers = [""] + [rename_map.get(k, k) for k in order_keys]

    # Variable order: target_vars first, then any extras, never the CF nuisance term.
    vars_to_print = []
    for col in order_keys:
        res = results_dict.get(col)
        if res is not None:
            for v in getattr(res, 'params', pd.Series(dtype=float)).index:
                if v not in vars_to_print:
                    vars_to_print.append(v)
    ordered_vars = [v for v in target_vars if v in vars_to_print]
    if not first_stage:   # second stage also lists any extra coefs; first stage = instruments only
        ordered_vars += [v for v in vars_to_print if v not in ordered_vars and 'v_hat' not in v.lower()]
        ordered_vars = [v for v in ordered_vars if 'v_hat' not in v.lower()]

    tex = [r"\begin{landscape}",
           r"\begin{table}[" + placement + r"]",
           r"\centering",
           r"\begin{threeparttable}",
           r"\caption{" + title + r"}",
           r"\label{" + label + r"}",
           r"\footnotesize",
           r"\setlength{\tabcolsep}{6pt}",
           r"\renewcommand{\arraystretch}{1.15}",
           r"\begin{tabular}{>{\raggedright\arraybackslash}p{4.8cm} *{" + str(n) + r"}{c}}",
           r"\toprule",
           " & ".join(headers) + r" \\",
           r"\midrule"]

    for v in ordered_vars:
        row_cf = [nice_var_name(v)]
        row_se = [""]
        for col in order_keys:
            res = results_dict.get(col)
            if res is None:
                row_cf.append("-"); row_se.append("-"); continue
            params = getattr(res, 'params', pd.Series(dtype=float))
            bse = getattr(res, 'bse', pd.Series(dtype=float))
            pvalues = getattr(res, 'pvalues', pd.Series(dtype=float))
            if v in params.index:
                c_str, se_str = format_value(params[v], bse[v], pvalues[v], digits=4)
                row_cf.append(c_str); row_se.append(se_str)
            else:
                row_cf.append("-"); row_se.append("-")
        tex.append(" & ".join(row_cf) + r" \\")
        tex.append(" & ".join(row_se) + r" \\")
        tex.append(r"\addlinespace[0.2ex]")

    tex.append(r"\midrule")
    if mean_phi is not None:
        row_mp = [r"Mean $\hat{\phi}$ (level)"]
        for col in order_keys:
            mp = mean_phi.get(col)
            row_mp.append(f"{mp:.3f}" if mp is not None and pd.notna(mp) else "-")
        tex.append(" & ".join(row_mp) + r" \\")

    row_nobs = ["Observations"]; row_r2 = ["$R^2$"]; row_fstat = ["F-Statistic"]
    row_cl = [r"Clusters ($G$)"]; row_gs = [r"Effective Clusters ($G^*$)"]
    for col in order_keys:
        res = results_dict.get(col)
        if res is None:
            for r_ in (row_nobs, row_r2, row_fstat, row_cl, row_gs): r_.append("-")
            continue
        nobs = getattr(res, 'nobs', np.nan); r2 = getattr(res, 'rsquared', np.nan)
        fstat = getattr(res, 'fvalue', np.nan); fpval = getattr(res, 'f_pvalue', np.nan)
        clusters = "-"
        if hasattr(res, 'cov_kwds') and res.cov_kwds.get('groups', None) is not None:
            g = res.cov_kwds.get('groups', None)
            clusters = str(g.nunique() if hasattr(g, 'nunique') else len(set(g)))
        elif hasattr(res, 'G_nominal') and not pd.isna(getattr(res, 'G_nominal', np.nan)):
            clusters = str(int(res.G_nominal))
        g_star = getattr(res, 'G_star', getattr(res, 'df_resid', np.nan))
        row_nobs.append(f"{nobs:,.0f}" if pd.notna(nobs) else "-")
        row_r2.append(f"{r2:.3f}" if pd.notna(r2) else "-")
        row_fstat.append(f"{fstat:.3f}{get_stars(fpval)}" if pd.notna(fstat) else "-")
        row_cl.append(clusters)
        row_gs.append(f"{g_star:.1f}" if pd.notna(g_star) else "-")
    diag_rows = [row_nobs, row_r2] + ([row_fstat] if first_stage else []) + [row_cl, row_gs]
    for r_ in diag_rows:
        tex.append(" & ".join(r_) + r" \\")

    tex += [r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item \textit{Notes:} Standard errors (score/multiplier wild cluster bootstrap at the "
            r"conglomerate level, Webb 6-point $B=999$, applied uniformly to every column; "
            r"\textcite{cameron2008bootstrap}, \textcite{mackinnon2017wild}) in "
            r"parentheses. The Local and Pooled columns report linear sleepiness coefficients; the "
            r"Single-Index and Joint Sieve columns report average marginal effects (AME). The "
            r"single-index/joint estimators carry no constant AME --- the baseline level is absorbed "
            r"into the monotone link --- so the `Mean $\hat{\phi}$' row gives the comparable implied "
            r"level across all columns. \textcite{carter2017asymptotic} effective clusters $G^*$ are a "
            r"diagnostic. Significance levels: *** $p<0.01$, ** $p<0.05$, * $p<0.1$.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
            r"\end{landscape}"]

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

    # Modern lineup (post two-way-FE relineup): drop the artifactual logit (E3), add the
    # bounded single-index (E5/E6) and the canonical joint sieve (E7 = BLP input, E8 = +Time).
    mapping = {
        '1 Local':             SLEEP_DIR / "DEMAND_PREP" / "est1",
        '2 Pooled Linear':     SLEEP_DIR / "DEMAND_PREP" / "est2",
        '5 Single-Index':      SLEEP_DIR / "DEMAND_PREP" / "est5",
        '6 Single-Index Time': SLEEP_DIR / "DEMAND_PREP" / "est6",
        '7 Joint Sieve':       SLEEP_DIR / "DEMAND_PREP" / "est7",
        '8 Joint Sieve Time':  SLEEP_DIR / "DEMAND_PREP" / "est8",
    }

    if getattr(args, 'skip_est2', False):
        mapping.pop('2 Pooled Linear', None)

    # All estimators store spec 12 under the same key.
    target_keys = {k: 'IV_HausmanFull x Tech' for k in mapping}

    stage1_res = {}
    stage2_res = {}
    phi_data = {}
    mean_phi = {}      # implied national mean phi_t level per estimator (the comparable "level" row)
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

        natl_path = d / "national_phi_t.csv"
        if natl_path.exists():
            try:
                nd = pd.read_csv(natl_path)
                pc = 'phi_t_IV_HausmanFull_x_Tech'
                if pc in nd.columns:
                    mean_phi[label] = float(nd[pc].mean())
            except Exception as e:
                print(f"  [Warning] Could not load national phi for {label}: {e}")

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
        'interaction_gdp_growth_yoy',
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
        mean_phi=mean_phi,
    )
    # Landscape variants of BOTH stages for V_Main inclusion (7 columns need the rotated
    # page). Same \labels as the portrait versions, so V_Main only switches which file it
    # \inputs (\input{est1-3_spec12_stage{1,2}_comparison_landscape.tex}).
    build_latex_table_landscape(
        stage1_res, order, first_stage_target_vars,
        out_dir / "est1-3_spec12_stage1_comparison_landscape.tex",
        title="First Stage IV Results across Specifications (Spec 12)",
        label="tab:spec12_stage1_comparison",
        placement="ht", first_stage=True,
    )
    build_latex_table_landscape(
        stage2_res, order, target_vars,
        out_dir / "est1-3_spec12_stage2_comparison_landscape.tex",
        title="Second Stage Results across Specifications (Spec 12)",
        label="tab:spec12_stage2_comparison",
        mean_phi=mean_phi, placement="ht",
    )

    # Nonlinear-only comparison (E5-E8): four columns fit PORTRAIT. Distinct \labels so
    # these can sit alongside the full tables in V_Main.
    order_nl = [k for k in order if k.split()[0] in {"5", "6", "7", "8"}]
    build_latex_table(
        stage1_res, order_nl, first_stage_target_vars,
        out_dir / "est5-8_spec12_stage1_comparison.tex",
        title="First Stage IV Results, Single-Index \\& Joint-Sieve Estimators (Spec 12)",
        label="tab:spec12_stage1_comparison_nl",
    )
    build_latex_table(
        stage2_res, order_nl, target_vars,
        out_dir / "est5-8_spec12_stage2_comparison.tex",
        title="Second Stage Results, Single-Index \\& Joint-Sieve Estimators (Spec 12)",
        label="tab:spec12_stage2_comparison_nl",
        mean_phi=mean_phi,
    )

    for fn in ("est1-3_spec12_stage1_comparison.tex",
               "est1-3_spec12_stage2_comparison.tex",
               "est1-3_spec12_stage1_comparison_landscape.tex",
               "est1-3_spec12_stage2_comparison_landscape.tex",
               "est5-8_spec12_stage1_comparison.tex",
               "est5-8_spec12_stage2_comparison.tex"):
        shutil.copy(out_dir / fn, _DRAFTS_DIR / fn)
    print(f"Exported LaTeX tables to {out_dir} and copied to {_DRAFTS_DIR}")

    # ---- 2) Pickle Model Information ----
    with open(out_dir / "est1-3_spec12_all_models.pkl", "wb") as f:
        pickle.dump(models_dict, f)
    print(f"Exported combined model instances to {out_dir / 'est1-3_spec12_all_models.pkl'}")

    # ---- 3) Plot Implied National Phi_t (single panel, all six strategies) ----
    #
    # Two band sources, by estimator family:
    #   * Linear E1/E2: aggregate per-market phi_mt to national with a delta-method
    #     SE band on the market panel (calc_agg_delta). Columns are saved as
    #     phi_mt_{safe_key}: "IV_HausmanFull x Tech" -> phi_mt_IV_HausmanFull_x_Tech.
    #   * Single-index / joint sieve E5/E6/E7/E8: plot the bootstrap point path and
    #     the score/multiplier wild-cluster-bootstrap CI band cached in
    #     Rout/ts_link_band_est{N}.pkl (cols time_id, phi_t, lo, hi, _d). Point and
    #     band come from the SAME fit, so the line sits inside its band by
    #     construction; national_phi_t.csv tracks the same *level* (means agree to
    #     ~1e-3) but its per-quarter wiggle is not what the tight band bounds.
    phi_col_map = {k: 'phi_mt_IV_HausmanFull_x_Tech' for k in mapping}

    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {lbl: base_colors[i % len(base_colors)] for i, lbl in enumerate(mapping.keys())}
    linestyle_map = {
        '1 Local':             '-',
        '2 Pooled Linear':     '-',
        '5 Single-Index':      '-',
        '6 Single-Index Time': '--',
        '7 Joint Sieve':       '-',
        '8 Joint Sieve Time':  '--',
    }
    label_rename = {
        '1 Local':             'Local Linear (E1)',
        '2 Pooled Linear':     'Pooled Linear (E2)',
        '5 Single-Index':      'Single-Index (E5)',
        '6 Single-Index Time': 'Single-Index +Time (E6)',
        '7 Joint Sieve':       'Joint Sieve (E7)',
        '8 Joint Sieve Time':  'Joint Sieve +Time (E8)',
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    ci_list: list = []

    for label in mapping.keys():
        est_num = int(str(label).split()[0])
        c = color_map[label]
        ls = linestyle_map.get(label, '-')
        plot_label = label_rename.get(label, label)

        band_pkl = out_dir / f"ts_link_band_est{est_num}.pkl"
        if est_num >= 5 and band_pkl.exists():
            # Single-index / joint sieve: bootstrap point path + wild-cluster band.
            try:
                with open(band_pkl, "rb") as fb:
                    bd = pickle.load(fb)
                bd = bd.sort_values("_d")
                ax.plot(bd["_d"], bd["phi_t"], label=plot_label, color=c, linewidth=2, linestyle=ls)
                ci_list.append((bd["_d"].values, bd["lo"].values, bd["hi"].values, pastelize_color(c)))
                continue
            except Exception as e:
                print(f"  [Warning] band load failed for {label} ({e}); falling back to delta band.")

        # Linear E1/E2: aggregate per-market phi to national with a delta-method band.
        df_phi = phi_data.get(label)
        if df_phi is None or 'year_quarter' not in df_phi.columns:
            continue
        tar_col = next((cc for cc in ('phi_mt_IV_HausmanFull_x_Tech', 'phi_mt_Tech')
                        if cc in df_phi.columns), None)
        if tar_col is None:
            print(f"  [Warning] No phi column found for {label}, skipping plot.")
            continue

        res = stage2_res.get(label)
        crit_val = _t_crit(res)
        agg, se = calc_agg_delta(df_phi, tar_col, res, False)
        if agg.empty:
            continue

        idx_dates = pd.PeriodIndex(agg.index.str.replace('_', 'Q'), freq='Q').to_timestamp()
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
