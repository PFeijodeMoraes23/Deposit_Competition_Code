"""
Automated LaTeX Table generation for BLP Demand Estimation outputs.
Compiles outputs from all estimation modules and prints significance-starred
results inside longtable fragments compatible with \\input{} in V_Main.tex.

Outputs
-------
  blp_logit_spec12_comparison.tex  -- logit baseline (E1-E5), 3 panels
  blp_est{i}_spec12_results.tex    -- full BLP GMM for estimation i (if available)

All fragments are wrapped in \\begin{spacing}{1.0}...\\end{spacing} to match
the formatting convention of the sleep estimation export scripts.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import pathlib
import json
import argparse
import shutil
import numpy as np
import scipy.stats as stats

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT.parents[1] / "BCB" / "Egan_et_al_2025_Rep" / "processed"
RESULTS_DIR  = DATA_DIR / "ESTIMATION_OUTPUT" / "BLP_RESULTS"
TABLES_DIR   = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"
_DRAFTS_DIR  = pathlib.Path(
    r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)"
    r"\Open Finance\Open-Finance\Drafts\Deposit Competition"
)

TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ── Variables Mapping ────────────────────────────────────────────────────────
VAR_MAP = {
    # Price coefficient
    'alpha':   r'$\hat{\alpha}$ (Spread)',
    'alpha_1': r'$\hat{\alpha}_{1}$ (Spread)',
    'alpha_2': r'$\hat{\alpha}_{2}$ (Spread)',
    'alpha_4': r'$\hat{\alpha}_{4}$ (Spread)',
    'alpha_5': r'$\hat{\alpha}_{5}$ (Spread)',

    # Linear characteristics (X_COLS)
    'fgc_covered':            'FGC Covered',
    'has_ip':                 'Has Investment Products (IP)',
    'seg_S2':                 'Segment S2',
    'seg_S3':                 'Segment S3',
    'seg_S4':                 'Segment S4',
    'seg_S5':                 'Segment S5',
    'log_total_assets_lag':   r'$\ln(\text{Total Assets}_{t-1})$',
    'equity_ratio_lag':       r'Equity Ratio ($t-1$)',
    'dummy_D_type':           'Dummy D-Type',

    # Demographics (D_COLS — used for Pi interactions)
    'gdp_per_capita':              r'GDP \textit{per capita}',
    'fraction_65plus':             r'Fraction 65+',
    'fraction_young':              r'Fraction Young',
    'pix_users_pf_per1000':        r'Pix Users (per 1,000)',
    'connections_per100':          r'Internet (per 100)',
    'frac_4g5g':                   r'Fraction 4G/5G',
    'branches_per1000':            r'Branches (per 1,000)',
    'cadunico_families_per1000':   r'CadUnico Families (per 1,000)',
}


def _stars(p_val: float) -> str:
    if p_val < 0.01:  return r'^{***}'
    if p_val < 0.05:  return r'^{**}'
    if p_val < 0.10:  return r'^{*}'
    return ''

def format_cell(coef, se, G_star=None):
    """Return (coef_str, se_str) with significance stars.

    Uses t(G*) per Imbens & Kolesár (2016) when G_star is available;
    otherwise falls back to normal approximation.
    """
    if coef is None or se is None or np.isnan(coef) or np.isnan(se):
        return '-', ''
    t = coef / max(se, 1e-15)
    if G_star is not None and G_star > 1:
        p = 2 * stats.t.sf(abs(t), df=G_star)
    else:
        p = 2 * (1 - stats.norm.cdf(abs(t)))
    coef_str = f'${coef:.4f}{_stars(p)}$'
    se_str   = f'$({se:.4f})$'
    return coef_str, se_str

def map_var(name: str) -> str:
    """Human-readable label for a BLP parameter name."""
    if name in VAR_MAP:
        return VAR_MAP[name]
    if name.startswith('sigma_'):
        inner = map_var(name[6:]).strip('$')
        return r'$\Sigma_{\text{' + inner + r'}}$'
    if name.startswith('pi_'):
        parts = name[3:].split('_x_')
        if len(parts) == 2:
            l = map_var(parts[0]).strip('$')
            r = map_var(parts[1]).strip('$')
            return r'$\Pi_{\text{' + l + r'}, \text{' + r + r'}}$'
    return name.replace('_', r'\_')

# ── Notes string (shared) ────────────────────────────────────────────────────
_NOTES = (
    r'\scriptsize \textit{Notes:} Standard errors in parentheses, cluster-robust '
    r'at the conglomerate level following \textcite{imbens2016robust} and '
    r'\textcite{carter2017asymptotic}. '
    r'Significance levels: *** $p<0.01$, ** $p<0.05$, * $p<0.1$.'
)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGIT TABLE (local, from logit_summary_spec_12.json)
# ═══════════════════════════════════════════════════════════════════════════════

# Sub-model display names (columns) and estimation strategies (panels)
_SUB_MODELS   = [('priceonly', 'Price only'), ('full', 'Full'), ('full_dtype', 'Full + D-Type')]
_ESTIM_LABELS = ['E1', 'E2', 'E3', 'E4', 'E5']

def build_logit_table() -> str:
    """Build an \\input{}-compatible longtable fragment for the logit baseline.

    Reads: BLP_RESULTS/logit_summary_spec_12.json
    Three panels (one per sub-model), five columns (E1–E5).
    """
    json_path = RESULTS_DIR / 'logit_summary_spec_12.json'
    if not json_path.exists():
        print(f'  [!] Logit summary not found: {json_path}')
        return ''

    with open(json_path, 'r') as f:
        data = json.load(f)

    ncols   = len(_ESTIM_LABELS)
    col_fmt = 'l' + 'c' * ncols
    caption = 'BLP Logit Baseline Estimation (Spec 12, Non-RC, $\\hat{\\theta}_2 = 0$)'
    label   = 'tab:blp_logit_spec12'

    lines = [
        r'\begin{spacing}{1.0}',
        rf'\begin{{longtable}}[c]{{{col_fmt}}}',
        rf'    \caption{{{caption}}}\label{{{label}}} \\',
        r'    \toprule',
        '    & ' + ' & '.join(_ESTIM_LABELS) + r' \\',
        r'    \midrule',
        r'    \endfirsthead',
        '',
        rf'    \multicolumn{{{ncols + 1}}}{{c}}{{{{\bfseries Table \thetable\ continued from previous page}}}} \\',
        r'    \toprule',
        '    & ' + ' & '.join(_ESTIM_LABELS) + r' \\',
        r'    \midrule',
        r'    \endhead',
        '',
        r'    \midrule',
        rf'    \multicolumn{{{ncols + 1}}}{{r}}{{\textit{{Continued on next page}}}} \\',
        r'    \endfoot',
        '',
        r'    \bottomrule',
        rf'    \multicolumn{{{ncols + 1}}}{{p{{0.85\textwidth}}}}{{{_NOTES}}} \\',
        r'    \endlastfoot',
        '',
    ]

    for pi, (sm_key, sm_label) in enumerate(_SUB_MODELS):
        # Panel header
        if pi > 0:
            lines += [r'    \addlinespace[1.5em]', r'    \toprule']
        lines += [
            rf'    \multicolumn{{{ncols + 1}}}{{l}}{{\textbf{{Panel {"ABC"[pi]}: {sm_label}}}}} \\',
            r'    \midrule',
            '    & ' + ' & '.join(_ESTIM_LABELS) + r' \\',
            r'    \midrule',
        ]

        # Collect all param names appearing in this sub-model across estimations
        param_names_ordered = []
        for el in _ESTIM_LABELS:
            key  = f'{el}_{sm_key}'
            entry = data.get(key, {})
            for p in entry.get('param_names', []):
                if p not in param_names_ordered:
                    param_names_ordered.append(p)

        # Parameter rows
        for p in param_names_ordered:
            row_c, row_s = [map_var(p)], ['']
            for el in _ESTIM_LABELS:
                key   = f'{el}_{sm_key}'
                entry = data.get(key, {})
                pnames = entry.get('param_names', [])
                if p in pnames:
                    idx   = pnames.index(p)
                    coef  = entry['theta1'][idx]
                    se    = entry['se'][idx]
                    G_star = entry.get('G_star')
                    c_str, s_str = format_cell(coef, se, G_star)
                    row_c.append(c_str)
                    row_s.append(s_str)
                else:
                    row_c.append('-')
                    row_s.append('')
            lines.append('    ' + ' & '.join(row_c) + r' \\')
            lines.append('    ' + ' & '.join(row_s) + r' \\')
            lines.append(r'    \addlinespace')

        # Footer stats
        obs_l, q_l, g_l, gstar_l = [], [], [], []
        for el in _ESTIM_LABELS:
            entry = data.get(f'{el}_{sm_key}', {})
            obs_l.append(f"{entry.get('n_obs', '---'):,}" if entry.get('n_obs') else '---')
            qv = entry.get('Q_value')
            q_l.append(f'{qv:.4f}' if qv is not None else '---')
            g_l.append(str(entry.get('n_clusters', '---')))
            gsv = entry.get('G_star')
            gstar_l.append(f'{gsv:.2f}' if gsv is not None else '---')

        lines += [
            r'    \midrule',
            '    Observations & '         + ' & '.join(obs_l)   + r' \\',
            '    GMM Objective $Q$ & '    + ' & '.join(q_l)     + r' \\',
            '    Clusters ($G$) & '       + ' & '.join(g_l)     + r' \\',
            '    Effective Clusters ($G^*$) & ' + ' & '.join(gstar_l) + r' \\',
            r'    \bottomrule',
        ]

    lines += [r'\end{longtable}', r'\end{spacing}']
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# BLP GMM TABLE (HPC results, from blp_results_E{i}_spec_{sp}_{stage}.json)
# ═══════════════════════════════════════════════════════════════════════════════

_STAGES_PRIORITY = ['extended', 'full', 'sigma', 'logit']

# D_COLS order from blp_draws.jl / blp_estimation.jl (used to decode pi indices)
_D_COLS = [
    'gdp_per_capita', 'fraction_65plus', 'fraction_young',
    'pix_users_pf_per1000', 'connections_per100', 'frac_4g5g',
    'branches_per1000', 'cadunico_families_per1000',
]


def _load_specs(est_id: int) -> dict:
    """Load the highest-priority result for each spec 1–12 for est_id."""
    specs = {}
    for sp in range(1, 13):
        for stage in _STAGES_PRIORITY:
            path = RESULTS_DIR / f'blp_results_E{est_id}_spec_{sp}_{stage}.json'
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        res = json.load(f)
                    res['__stage__'] = stage
                    specs[sp] = res
                    break
                except Exception:
                    pass
    return specs


def _build_theta2_names(r: dict) -> list:
    """Reconstruct theta2 parameter names from sigma_indices and pi_interactions."""
    names = []
    pnames = r.get('param_names_theta1', [])
    for idx in r.get('sigma_indices', []):
        try:
            names.append(f'sigma_{pnames[idx]}')
        except IndexError:
            names.append(f'sigma_{idx}')
    for k_i, d_i in r.get('pi_interactions', []):
        try:
            k_name = pnames[k_i]
            d_name = _D_COLS[d_i] if d_i < len(_D_COLS) else str(d_i)
            names.append(f'pi_{k_name}_x_{d_name}')
        except (IndexError, KeyError):
            names.append(f'pi_{k_i}_{d_i}')
    return names


def build_gmm_table(est_id: int) -> str:
    """Build an \\input{}-compatible longtable fragment for BLP GMM results.

    Columns = specifications 1–12 (only those with results are shown).
    Rows    = linear params (θ₁), then non-linear (θ₂: σ, Π).
    """
    specs = _load_specs(est_id)
    if not specs:
        print(f'  [!] No BLP GMM results found for Est {est_id}')
        return ''

    found_specs = sorted(specs.keys())
    ncols = len(found_specs)
    col_fmt = 'l' + 'c' * ncols
    caption = f'BLP Demand Estimation --- Estimation {est_id} (Spec 12, RC)'
    label   = f'tab:blp_est{est_id}_results'

    # Union of parameter names across all found specs
    theta1_master = []
    theta2_master = []
    for sp in found_specs:
        r = specs[sp]
        for p in r.get('param_names_theta1', []):
            if p not in theta1_master:
                theta1_master.append(p)
        for p in _build_theta2_names(r):
            if p not in theta2_master:
                theta2_master.append(p)

    header_nums = ' & '.join(f'({sp})' for sp in found_specs)

    lines = [
        r'\begin{spacing}{1.0}',
        rf'\begin{{longtable}}[c]{{{col_fmt}}}',
        rf'    \caption{{{caption}}}\label{{{label}}} \\',
        r'    \toprule',
        '    Parameter & ' + header_nums + r' \\',
        r'    \midrule',
        r'    \endfirsthead',
        '',
        rf'    \multicolumn{{{ncols + 1}}}{{c}}{{{{\bfseries Table \thetable\ continued from previous page}}}} \\',
        r'    \toprule',
        '    Parameter & ' + header_nums + r' \\',
        r'    \midrule',
        r'    \endhead',
        '',
        r'    \midrule',
        rf'    \multicolumn{{{ncols + 1}}}{{r}}{{\textit{{Continued on next page}}}} \\',
        r'    \endfoot',
        '',
        r'    \bottomrule',
        rf'    \multicolumn{{{ncols + 1}}}{{p{{0.85\textwidth}}}}{{{_NOTES}}} \\',
        r'    \endlastfoot',
        '',
    ]

    # ── Linear parameters ──
    lines.append(rf'    \multicolumn{{{ncols + 1}}}{{c}}{{\textit{{Linear Parameters ($\hat{{\theta}}_1$)}}}} \\')
    lines.append(r'    \midrule')
    for p in theta1_master:
        row_c = [map_var(p)]
        row_s = ['']
        for sp in found_specs:
            r = specs[sp]
            pnames = r.get('param_names_theta1', [])
            if p in pnames:
                idx   = pnames.index(p)
                coef  = r['theta1'][idx]
                se    = r['theta1_se'][idx] if 'theta1_se' in r else np.nan
                c, s  = format_cell(coef, se, r.get('G_star'))
                row_c.append(c); row_s.append(s)
            else:
                row_c.append('-'); row_s.append('')
        lines.append('    ' + ' & '.join(row_c) + r' \\')
        lines.append('    ' + ' & '.join(row_s) + r' \\')
        lines.append(r'    \addlinespace')

    # ── Non-linear parameters ──
    if theta2_master:
        lines.append(r'    \midrule')
        lines.append(rf'    \multicolumn{{{ncols + 1}}}{{c}}{{\textit{{Non-Linear Parameters ($\hat{{\theta}}_2$)}}}} \\')
        lines.append(r'    \midrule')
        for p in theta2_master:
            row_c = [map_var(p)]
            row_s = ['']
            for sp in found_specs:
                r = specs[sp]
                local_t2 = _build_theta2_names(r)
                if p in local_t2:
                    idx  = local_t2.index(p)
                    t2   = r.get('theta2', [])
                    t2se = r.get('theta2_se', [])
                    coef = t2[idx]   if idx < len(t2)   else np.nan
                    se   = t2se[idx] if idx < len(t2se) else np.nan
                    c, s = format_cell(coef, se, r.get('G_star'))
                    row_c.append(c); row_s.append(s)
                else:
                    row_c.append('-'); row_s.append('')
            lines.append('    ' + ' & '.join(row_c) + r' \\')
            lines.append('    ' + ' & '.join(row_s) + r' \\')
            lines.append(r'    \addlinespace')

    # ── Footer ──
    stage_l, q_l, conv_l, gstar_l = [], [], [], []
    for sp in found_specs:
        r = specs[sp]
        stage_l.append(r.get('__stage__', '---'))
        qv = r.get('Q_value')
        q_l.append(f'{qv:.4f}' if qv is not None else '---')
        conv_l.append('Yes' if r.get('converged', False) else 'No')
        gsv = r.get('G_star')
        gstar_l.append(f'{gsv:.2f}' if gsv is not None else '---')

    lines += [
        r'    \midrule',
        '    Optimization Stage & '        + ' & '.join(stage_l) + r' \\',
        '    GMM Objective $Q$ & '         + ' & '.join(q_l)     + r' \\',
        '    Converged & '                 + ' & '.join(conv_l)  + r' \\',
        '    Effective Clusters ($G^*$) & '+ ' & '.join(gstar_l) + r' \\',
        r'    \bottomrule',
    ]

    lines += [r'\end{longtable}', r'\end{spacing}']
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry points
# ═══════════════════════════════════════════════════════════════════════════════

def _write(path: pathlib.Path, content: str):
    if not content:
        return
    path.write_text(content + '\n', encoding='utf-8')
    print(f'  ==> {path}')
    if _DRAFTS_DIR.exists():
        shutil.copy(path, _DRAFTS_DIR / path.name)
        print(f'  ==> (copied to Drafts)')


def run_logit_table():
    print('\n--- Building Logit Baseline Table ---')
    frag = build_logit_table()
    _write(TABLES_DIR / 'blp_logit_spec12_comparison.tex', frag)


def process_estimation(est_id: int):
    print(f'\n--- Building BLP GMM Table for Estimation {est_id} ---')
    frag = build_gmm_table(est_id)
    _write(TABLES_DIR / f'blp_est{est_id}_spec12_results.tex', frag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create BLP LaTeX table fragments')
    parser.add_argument('--est', type=int, choices=[1, 2, 3, 4, 5],
                        help='BLP GMM estimation ID (1-5). Omit to build logit table only.')
    parser.add_argument('--logit', action='store_true',
                        help='(Re-)build the logit baseline table.')
    args = parser.parse_args()

    if args.logit or args.est is None:
        run_logit_table()

    if args.est is not None:
        process_estimation(args.est)
    elif not args.logit:
        # No flags: build everything available
        run_logit_table()
        for i in range(1, 6):
            process_estimation(i)
