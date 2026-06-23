"""
Generate publication-ready LaTeX tables for BLP Logit results (Specification 12).

This script reads logit_summary_spec_12.json and creates tables with one column
per logit sub-model (priceonly, full, full+dtype) for any estimation strategy.

Tables include cluster-robust standard errors and significance stars following
IK2016 (effective clusters, t-distribution inference).

Usage
-----
  # Generate tables for all coherence routines E1-E6 (default)
  python make_blp_logit_table.py

  # Generate table for a single coherence routine
  python make_blp_logit_table.py --est 6

  # Generate tables for legacy 5-routine build (E1-E5)
  python make_blp_logit_table.py --legacy

Output
------
  est{i}_spec12_logit.tex  (in ESTIMATION_OUTPUT/Rout/ and Drafts/)
"""

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import pathlib
import json
import argparse
import numpy as np
import scipy.stats as stats

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT.parents[1] / "BCB" / "Egan_et_al_2025_Rep" / "processed"
RESULTS_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "BLP_RESULTS"
TABLES_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"
DRAFTS_DIR = pathlib.Path(
    r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)"
    r"\Open Finance\Open-Finance\Drafts\Deposit Competition"
)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Variable Mapping (human-readable labels) ──────────────────────────────────
VAR_MAP = {
    'alpha':                r'Price coefficient ($\alpha$)',
    'fgc_covered':          'FGC Covered',
    'has_ip':               'Has Payment Institution',
    'log_total_assets_lag': r'$\ln(\text{Total Assets}_{t-1})$',
    'seg_S2':               'Segment S2',
    'seg_S3':               'Segment S3',
    'seg_S4':               'Segment S4',
    'seg_S5':               'Segment S5',
    'dummy_D_type':         'D Type',
}

# Fixed display order for parameter rows in the table
ROW_ORDER = [
    'alpha', 'fgc_covered', 'has_ip', 'log_total_assets_lag',
    'seg_S2', 'seg_S3', 'seg_S4', 'seg_S5', 'dummy_D_type',
]

def map_var(name: str) -> str:
    """Human-readable label for a parameter name."""
    return VAR_MAP.get(name, name.replace('_', r'\_'))

def _stars(p_val: float) -> str:
    """Return LaTeX superscript stars based on p-value."""
    if p_val < 0.01:
        return r'^{***}'
    if p_val < 0.05:
        return r'^{**}'
    if p_val < 0.10:
        return r'^{*}'
    return ''

def format_cell(coef, se, G_star=None):
    """Return (coef_str, se_str) with significance stars.

    Uses t(G*) per Imbens & Kolesár (2016) when G_star is available;
    otherwise falls back to normal approximation.
    """
    if coef is None or se is None or np.isnan(coef) or np.isnan(se):
        return '-', ''

    t = coef / max(se, 1e-15)

    # Calculate p-value using effective clusters (IK2016)
    if G_star is not None and G_star > 1:
        p = 2 * stats.t.sf(abs(t), df=G_star)
    else:
        p = 2 * (1 - stats.norm.cdf(abs(t)))

    coef_str = f'${coef:.4f}{_stars(p)}$'
    se_str = f'$({se:.4f})$'
    return coef_str, se_str

def format_q_value(qv, L):
    """Format Q-value with chi2(L) p-value significance.

    Under the null that E[ξ·Z]=0, Q ~ chi2(L) where L is number of IVs.
    Lower Q is better (fewer moment violations).
    """
    if qv is None or np.isnan(qv) or L <= 0:
        return '---'

    # Chi-squared CDF: p-value is 1 - CDF (upper tail test)
    p = 1 - stats.chi2.cdf(qv, df=L)
    return f'${qv:.4f}{_stars(p)}$'

def build_logit_table(est_id: int, simple_title: bool = False,
                      coherence: bool = False) -> str:
    """Build a longtable for estimation est_id Spec 12 logit results.

    Four columns: priceonly, core, full, full+dtype
    Parameter rows with SE underneath, in fixed ROW_ORDER.
    Footer with diagnostics (observations, Q-value, effective clusters).

    When ``coherence`` is True, reads the post-coherence-fix summary
    (``logit_summary_spec_12_coherence.json``, written by the 6-routine
    blp_1_logit_coherence.jl) instead of the legacy 5-routine summary.
    """
    summary_name = ('logit_summary_spec_12_coherence.json' if coherence
                    else 'logit_summary_spec_12.json')
    json_path = RESULTS_DIR / summary_name
    if not json_path.exists():
        print(f'ERROR: {json_path} not found.')
        if coherence:
            print('Run: julia --project=. --threads=auto blp_1_logit_coherence.jl')
        else:
            print('Run: julia --project=. --threads=4 blp_logit_local.jl')
        return ''

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Sub-models for estimation
    submodels = [
        ('priceonly',  r'Price Only'),
        ('core',       r'Price + Core'),
        ('full',       r'Price + Chars'),
        ('full_dtype', r'+ D-Type'),
    ]

    ncols = len(submodels)
    # Full-\textwidth xltabular (longtable-capable tabularx): a fixed-width
    # raggedright label column + equal centered X columns. This pins the table
    # to \textwidth so the notes row (also \textwidth) lines up with the table
    # edges, matching the spec-12 comparison tables.
    col_fmt = (r'>{\raggedright\arraybackslash}p{0.24\textwidth} '
               r'*{' + str(ncols) + r'}{>{\centering\arraybackslash}X}')

    # Build header section
    title = 'Demand Logit Estimation' if simple_title else f'Demand Logit Estimation -- Estimation {est_id}, Specification 12'
    label = 'tab:demand_logit' if simple_title else f'tab:demand_logit_est{est_id}_spec12'

    lines = [
        r'\begin{spacing}{1.0}',
        rf'\begin{{xltabular}}{{\textwidth}}{{{col_fmt}}}',
        rf'    \caption{{{title}}}',
        rf'    \label{{{label}}} \\',
        r'    \toprule',
        '    Parameter & ' + ' & '.join(sm[1] for sm in submodels) + r' \\',
        r'    \midrule',
        r'    \endfirsthead',
        '',
        rf'    \multicolumn{{{ncols + 1}}}{{c}}{{\bfseries Table \thetable\ continued from previous page}} \\',
        r'    \toprule',
        '    Parameter & ' + ' & '.join(sm[1] for sm in submodels) + r' \\',
        r'    \midrule',
        r'    \endhead',
        '',
        r'    \midrule',
        rf'    \multicolumn{{{ncols + 1}}}{{r}}{{\textit{{Continued on next page}}}} \\',
        r'    \endfoot',
        '',
        r'    \bottomrule',
        # Notes span the full table width (\textwidth minus the two outer
        # \tabcolsep margins of the multicolumn), so the box matches the table
        # edges instead of sitting in a narrow centered minipage.
        r'    \multicolumn{' + str(ncols + 1) + r'}{p{\dimexpr\textwidth-2\tabcolsep\relax}}{\scriptsize \textit{Notes:} Cluster-robust standard errors in parentheses, clustered at conglomerate level following \textcite{imbens2016robust} and \textcite{carter2017asymptotic}. Significance: *** $p<0.01$, ** $p<0.05$, * $p<0.1$. $Q$ denotes the GMM overidentification test statistic ($\chi^2_L$, $L$ = \# instruments); $G^*$ is effective clusters.} \\',
        r'    \endlastfoot',
        '',
    ]

    # Parameter rows in fixed display order
    for i, p in enumerate(ROW_ORDER):
        row_c = [map_var(p)]
        row_s = ['']

        for sm_key, _ in submodels:
            key = f'E{est_id}_{sm_key}'
            entry = data.get(key, {})
            pnames = entry.get('param_names', [])

            if p in pnames:
                idx = pnames.index(p)
                coef = entry['theta1'][idx]
                se = entry['se'][idx]
                G_star = entry.get('G_star')
                c_str, s_str = format_cell(coef, se, G_star)
                row_c.append(c_str)
                row_s.append(s_str)
            else:
                row_c.append('-')
                row_s.append('')

        lines.append('    ' + ' & '.join(row_c) + r' \\')
        lines.append('    ' + ' & '.join(row_s) + r' \\')
        if i < len(ROW_ORDER) - 1:
            lines.append(r'    \addlinespace')

    # Footer statistics
    lines.append(r'    \midrule')

    obs_l, q_l, gstar_l, n_iv_l = [], [], [], []
    for sm_key, _ in submodels:
        key = f'E{est_id}_{sm_key}'
        entry = data.get(key, {})

        obs = entry.get('n_obs')
        obs_l.append(f'{obs:,}' if obs else '---')

        qv = entry.get('Q_value')
        n_iv = entry.get('n_iv', 0)
        n_iv_l.append(n_iv)
        q_l.append(format_q_value(qv, n_iv) if qv is not None else '---')

        gsv = entry.get('G_star')
        gstar_l.append(f'{gsv:.2f}' if gsv is not None else '---')

    lines += [
        '    Observations & ' + ' & '.join(obs_l) + r' \\',
        r'    $Q$ (GMM overidentification) & ' + ' & '.join(q_l) + r' \\',
        r'    Degrees of freedom (overidentification) & ' + ' & '.join(str(n) for n in n_iv_l) + r' \\',
        r'    Effective Clusters ($G^*$) & ' + ' & '.join(gstar_l) + r' \\',
    ]

    lines += [
        r'\end{xltabular}',
        r'\end{spacing}',
    ]

    return '\n'.join(lines)

def build_instruments_table() -> str:
    """Build table documenting BLP IV instruments used in logit estimation."""

    instruments = [
        # BLP Leave-One-Out instruments
        ('loo_log_assets', 'BLP LOO', r'Leave-one-out mean $\ln(\text{Total Assets}_{t-1})$ of rivals'),
        ('mean_loo_log_assets', 'BLP LOO', r'Mean $\ln(\text{Total Assets}_{t-1})$ across all leave-one-out observations'),
        ('loo_equity_ratio', 'BLP LOO', r'Leave-one-out mean Equity Ratio of rivals'),
        ('mean_loo_equity_ratio', 'BLP LOO', r'Mean Equity Ratio across all leave-one-out observations'),
        ('loo_basileia', 'BLP LOO', r'Leave-one-out mean Basel Index of rivals'),
        ('mean_loo_basileia', 'BLP LOO', r'Mean Basel Index across all leave-one-out observations'),
        ('loo_credit_assets', 'BLP LOO', r'Leave-one-out mean credit assets ratio of rivals'),
        ('mean_loo_credit_assets', 'BLP LOO', r'Mean credit assets ratio across all leave-one-out observations'),
        ('loo_npl_provision', 'BLP LOO', r'Leave-one-out mean non-performing loan provision of rivals'),
        ('mean_loo_npl_provision', 'BLP LOO', r'Mean NPL provision across all leave-one-out observations'),
        ('n_rivals', 'BLP LOO', r'Number of rival institutions in the MCA-time cell'),

        # Cost shifters
        ('personnel_cost_ratio_lag', 'Cost Shifter', r'Personnel cost ratio (lagged), $t-1$'),
        ('admin_cost_ratio_lag', 'Cost Shifter', r'Administrative cost ratio (lagged), $t-1$'),
        ('tax_cost_ratio_lag', 'Cost Shifter', r'Tax cost ratio (lagged), $t-1$'),

        # Capital adequacy
        ('indice_basileia_lag', 'Capital', r'Basel Index (regulatory capital adequacy ratio), lagged $t-1$'),
    ]

    ncols = 3

    lines = [
        r'\begin{spacing}{1.0}',
        r'\begin{xltabular}{\textwidth}{>{\raggedright\arraybackslash}p{0.30\textwidth} >{\raggedright\arraybackslash}p{0.16\textwidth} >{\raggedright\arraybackslash}X}',
        r'    \caption{Instrumental Variables for BLP Logit Estimation (Specification 12)}',
        r'    \label{tab:blp_instruments} \\',
        r'    \toprule',
        r'    Instrument Name & Category & Description \\',
        r'    \midrule',
        r'    \endfirsthead',
        '',
        r'    \caption[]{Instrumental Variables for BLP Logit Estimation (Continued)} \\',
        r'    \toprule',
        r'    Instrument Name & Category & Description \\',
        r'    \midrule',
        r'    \endhead',
        '',
        r'    \midrule',
        rf'    \multicolumn{{{ncols}}}{{r}}{{\textit{{Continued on next page}}}} \\',
        r'    \endfoot',
        '',
        r'    \bottomrule',
        rf'    \multicolumn{{{ncols}}}{{p{{\dimexpr\textwidth-2\tabcolsep\relax}}}}{{\scriptsize \textit{{Notes:}} BLP LOO instruments are constructed as leave-one-out competitors'' characteristics within the same MCA-time cell, following \textcite{{berry1995automobile}}. Cost shifters are lagged firm-level variables plausibly exogenous to demand. Capital adequacy is a regulatory measure. Total: 15 instruments for 2SLS estimation of equation (Eq-15/Eq-16).}} \\',
        r'    \endlastfoot',
        '',
    ]

    # Add rows
    for inst_name, category, description in instruments:
        lines.append(rf'    {inst_name} & {category} & {description} \\')

    lines += [
        r'\end{xltabular}',
        r'\end{spacing}',
    ]

    return '\n'.join(lines)

def build_prod_chars_table() -> str:
    """Build product characteristics table (updated, without equity_ratio)."""

    lines = [
        r'\begin{spacing}{1.0}',
        r'    \begin{longtable}{>{\ttfamily\small\raggedright\arraybackslash}p{4.5cm} >{\raggedright\arraybackslash}p{4cm} >{\raggedright\arraybackslash}p{7cm}}',
        r'        \caption{Product Characteristics and Price Variables} \label{tab:prod_chars} \\',
        r'        \toprule',
        r'        \normalfont\textbf{Column} & \textbf{Category} & \textbf{Description} \\',
        r'        \midrule',
        r'        \endfirsthead',
        '',
        r'        \caption[]{Product Characteristics and Price Variables (Continued)} \\',
        r'        \toprule',
        r'        \normalfont\textbf{Column} & \textbf{Category} & \textbf{Description} \\',
        r'        \midrule',
        r'        \endhead',
        '',
        r'        \midrule',
        r'        \multicolumn{3}{r}{\textit{Continued on next page}} \\',
        r'        \endfoot',
        '',
        r'        \bottomrule',
        r'        \endlastfoot',
        '',
        r'        \multicolumn{3}{l}{\textit{Identifiers}} \\* \addlinespace[0.5ex]',
        r'        deposit\_type       & Product type & 1 = demand, 2 = savings, 3 = interbank, 4 = time/CDB, 5 = prepaid \\',
        r'        deposit\_type\_name  & Label        & Verbal label \\',
        r'        \midrule',
        '',
        r'        \multicolumn{3}{l}{\textit{Price}} \\* \addlinespace[0.5ex]',
        r'        spread\_qoq        & Price variable $\rho_{jkmt}$ & Selic QoQ minus deposit rate QoQ \\',
        r'        \midrule',
        '',
        r'        \multicolumn{3}{l}{\textit{Deposit insurance}} \\* \addlinespace[0.5ex]',
        r'        fgc\_covered       & Insurance dummy & 1 if FGC-insured (types 1, 2, 4) \\',
        r'        \midrule',
        '',
        r'        \multicolumn{3}{l}{\textit{Conglomerate classification}} \\* \addlinespace[0.5ex]',
        r'        has\_ip            & IP subsidiary flag & 1 if the conglomerate contains a Payment Institution \\',
        r'        segment           & BCB prudential segment & S1--S5 \\',
        r'        seg\_S2--seg\_S5   & Segment dummies & Binary indicators \\',
        r'        \midrule',
        '',
        r'        \multicolumn{3}{l}{\textit{Bank size}} \\* \addlinespace[0.5ex]',
        r'        total\_assets      & Size (R\$)   & Total balance-sheet assets, lagged by one quarter \\',
        r'        log\_total\_assets  & Log size     & $\ln(\text{total\_assets})$, lagged by one quarter \\',
        r'    \end{longtable}',
        r'\end{spacing}',
    ]

    return '\n'.join(lines)

def build_demographic_chars_table() -> str:
    """Build demographic characteristics table."""

    lines = [
        r'\begin{spacing}{1.0}',
        r'    \begin{longtable}{>{\ttfamily\small\raggedright\arraybackslash}p{5cm} >{\raggedright\arraybackslash}p{3.5cm} >{\raggedright\arraybackslash}p{7cm}}',
        r'        \caption{Market-Level Demographic Variables} \label{tab:demographic_chars} \\',
        r'        \toprule',
        r'        \normalfont\textbf{Column} & \textbf{Source} & \textbf{Description} \\',
        r'        \midrule',
        r'        \endfirsthead',
        '',
        r'        \caption[]{Market-Level Demographic Variables (Continued)} \\',
        r'        \toprule',
        r'        \normalfont\textbf{Column} & \textbf{Source} & \textbf{Description} \\',
        r'        \midrule',
        r'        \endhead',
        '',
        r'        \midrule',
        r'        \multicolumn{3}{r}{\textit{Continued on next page}} \\',
        r'        \endfoot',
        '',
        r'        \bottomrule',
        r'        \endlastfoot',
        '',
        r'        pop\_total                & IBGE          & Total population of the MCA, used as the market size $M_{mt}$ \\',
        r'        gdp\_per\_capita         & IBGE          & Municipal GDP aggregated to MCA and divided by population \\',
        r'        fraction\_65plus         & IBGE (Census) & Share of population aged 65 and over \\',
        r'        fraction\_young          & IBGE (Census) & Share of population in the young demographic (ages 15--20) \\',
        r'        pix\_users\_pf\_per1000  & BCB           & Individual PIX users per 1,000 inhabitants \\',
        r'        connections\_per100      & ANATEL        & Broadband/mobile subscriptions per 100 inhabitants \\',
        r'        frac\_4g5g               & ANATEL        & Share of mobile connections using 4G or 5G technology \\',
        r'        branches\_per1000        & BCB           & Number of physical bank branches per 1,000 inhabitants \\',
        r'        cadunico\_families\_per1000 & SAGI/MDS   & Low-income families registered in CadÚnico per 1,000 inhabitants \\',
        r'    \end{longtable}',
        r'\end{spacing}',
    ]

    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description='Generate BLP Logit tables and documentation')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true',
                       help='Generate tables for all estimations in the active build')
    group.add_argument('--est', type=int, default=None,
                       help='Generate table for a single estimation (e.g. --est 2); '
                            'omit to run all estimations for the active build')
    parser.add_argument('--docs', action='store_true',
                       help='Also generate documentation tables (instruments, prod_chars, demographic_chars)')
    parser.add_argument('--simple-title', action='store_true',
                       help='Use simple title without estimation/specification numbers')
    parser.add_argument('--coherence', action='store_true', default=True,
                       help='Use post-coherence-fix results (default); reads '
                            'logit_summary_spec_12_coherence.json, runs E1-E6')
    parser.add_argument('--legacy', dest='coherence', action='store_false',
                       help='Use legacy 5-routine results (logit_summary_spec_12.json), runs E1-E5')
    args = parser.parse_args()

    # Coherence build: 8 routines (E1-E8). Legacy build: 5 routines (E1-E5).
    # With no --est, run all routines for the active build.
    if args.coherence:
        est_ids = [args.est] if (args.est is not None and not args.all) else list(range(1, 9))
    else:
        est_ids = [args.est] if (args.est is not None and not args.all) else list(range(1, 6))

    suffix = '_coherence' if args.coherence else ''
    build_note = ' [coherence]' if args.coherence else ''
    print(f'Generating BLP Logit tables for Specification 12{build_note}...')

    success = True
    for est_id in est_ids:
        print(f'\n[E{est_id}] Generating table{build_note}...')

        tex_content = build_logit_table(est_id, simple_title=args.simple_title,
                                        coherence=args.coherence)

        if not tex_content:
            print(f'ERROR: Failed to generate table for E{est_id}.')
            success = False
            continue

        # Save to primary location (Egan repository)
        output_path = TABLES_DIR / f'est{est_id}_spec12_logit{suffix}.tex'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(tex_content)
        print(f'[OK] Table saved: {output_path}')

        # Save copy to Drafts directory
        drafts_path = DRAFTS_DIR / f'est{est_id}_spec12_logit{suffix}.tex'
        with open(drafts_path, 'w', encoding='utf-8') as f:
            f.write(tex_content)
        print(f'[OK] Copy saved: {drafts_path}')

    if success:
        print(f'\n[SUCCESS] Generated tables for E{est_ids}')
        print(f'\nInsert logit tables into V_Main.tex with:')
        for est_id in est_ids:
            print(f'  \\input{{est{est_id}_spec12_logit{suffix}.tex}}')
    else:
        print('[FAILED] Some tables could not be generated')

    # Generate documentation tables if requested
    if args.docs:
        print(f'\n[DOCS] Generating documentation tables...')

        # Instruments table
        print(f'[DOCS] Building instruments table...')
        inst_content = build_instruments_table()
        inst_path = TABLES_DIR / 'tab_blp_instruments.tex'
        with open(inst_path, 'w', encoding='utf-8') as f:
            f.write(inst_content)
        inst_drafts = DRAFTS_DIR / 'tab_blp_instruments.tex'
        with open(inst_drafts, 'w', encoding='utf-8') as f:
            f.write(inst_content)
        print(f'[OK] Instruments table: {inst_path}')
        print(f'[OK] Copy: {inst_drafts}')

        # Product characteristics table
        print(f'[DOCS] Building product characteristics table...')
        prod_content = build_prod_chars_table()
        prod_path = TABLES_DIR / 'tab_prod_chars_updated.tex'
        with open(prod_path, 'w', encoding='utf-8') as f:
            f.write(prod_content)
        prod_drafts = DRAFTS_DIR / 'tab_prod_chars_updated.tex'
        with open(prod_drafts, 'w', encoding='utf-8') as f:
            f.write(prod_content)
        print(f'[OK] Product chars table: {prod_path}')
        print(f'[OK] Copy: {prod_drafts}')

        # Demographic characteristics table
        print(f'[DOCS] Building demographic characteristics table...')
        demo_content = build_demographic_chars_table()
        demo_path = TABLES_DIR / 'tab_demographic_chars_updated.tex'
        with open(demo_path, 'w', encoding='utf-8') as f:
            f.write(demo_content)
        demo_drafts = DRAFTS_DIR / 'tab_demographic_chars_updated.tex'
        with open(demo_drafts, 'w', encoding='utf-8') as f:
            f.write(demo_content)
        print(f'[OK] Demographic chars table: {demo_path}')
        print(f'[OK] Copy: {demo_drafts}')

        print(f'\n[DOCS] Replace in V_Main.tex:')
        print(f'  \\input{{tab_blp_instruments.tex}}')
        print(f'  \\input{{tab_prod_chars_updated.tex}}')
        print(f'  \\input{{tab_demographic_chars_updated.tex}}')

    return success

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
