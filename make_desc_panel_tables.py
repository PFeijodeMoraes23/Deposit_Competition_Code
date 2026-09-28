"""
descriptive_1.py
==============================
Generates summary statistics for the Egan et al. market panel setup.
Outputs are saved as CSV and LaTeX tables partitioned by overall, bank type, and region,
as well as broken down by year.

CLI Options:
------------
usage: descriptive_1.py [-h] [--weight-col WEIGHT_COL]

Generate descriptive statistics for the market panel.

options:
  -h, --help            show this help message and exit
  --weight-col WEIGHT_COL
                        Column to use for market-weighted statistics (e.g.,
                        pop_total)
"""
import argparse
from pathlib import Path
import sys
import warnings

try:
    from utils.venv_guard import ensure_project_venv
except ImportError:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import numpy as np

from utils.window import apply_window, MIN_YEAR, MAX_YEAR
from utils import paths as _paths
# One definition of the spread unit for every descriptive table: annualized, compounded,
# percentage points (see make_desc_compressed_tables.annualized_spread_pp).
from make_desc_compressed_tables import annualized_spread_pp, SPREAD_UNIT, SPREAD_CONVENTION_TEX

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
PANEL_CSV = DATA_DIR / "market_panel.csv"
OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"

# Map exactly to the 5 macro-regions of Brazil using the 1st digit of CODMUN_IBGE
REGION_MAPPING = {
    '1': 'North',
    '2': 'Northeast',
    '3': 'Southeast',
    '4': 'South',
    '5': 'Center-West'
}

def _resolve_is_B(df):
    """Firm type as a boolean Series: True = B (brick-and-mortar), False = D (digital).

    `is_B` is decided once, in panel_6_market.attach_mca_code, from the physical-network
    verdict in digital_banks_diagnostic.csv, and stored in market_panel.csv as 0/1. It is
    used verbatim whenever present -- never recomputed or "corrected" here, and in
    particular not reconciled with mca_code: firm type and market tier are separate facts.
    A panel written before the column existed falls back to the CODMUN_IBGE sentinel,
    which approximates the verdict by where a bank books its deposits.
    """
    if 'is_B' in df.columns:
        s = df['is_B']
        if pd.api.types.is_bool_dtype(s):
            return s.astype(bool)
        if pd.api.types.is_numeric_dtype(s):
            # 0/1 as stored; a missing value reads as D, matching the Julia side's
            # Bool.(coalesce.(df.is_B, false)).
            return pd.to_numeric(s, errors='coerce').fillna(0) != 0
        # Text, from a CSV writer that spelled the column out ("true"/"True"/"1").
        return s.astype(str).str.strip().str.lower().isin(('1', 'true', 't', 'yes'))
    print("[WARNING] Column 'is_B' not found in the market panel: this panel predates the "
          "stored firm-type column. Falling back to the CODMUN_IBGE sentinel; re-run "
          "panel_market.py to store the authoritative column.")
    return df['CODMUN_IBGE'].astype(str).str.split('.').str[0] != '0'


def weighted_mean_std(df, col, weight_col):
    sub = df[[col, weight_col]].dropna()
    if len(sub) == 0:
        return np.nan, np.nan
    w = np.array(sub[weight_col], dtype=float)
    x = np.array(sub[col], dtype=float)
    w_sum = np.sum(w)
    if w_sum == 0.0:
        return np.nan, np.nan
    mean = np.average(x, weights=w)
    var = np.average((x - mean)**2, weights=w)
    return mean, np.sqrt(var)

def generate_summary(df, group_cols, vars_to_summarize, weight_col=None):
    results = []
    
    if group_cols:
        groups = df.groupby(group_cols)
    else:
        groups = [('Overall', df)]

    for name, group in groups:
        row = {'Group': name if not isinstance(name, tuple) else ' - '.join(map(str, name))}
        
        for v in vars_to_summarize:
            if v not in group.columns:
                continue
            
            if weight_col and weight_col in group.columns:
                mean, std = weighted_mean_std(group, v, weight_col)
            else:
                mean = group[v].mean()
                std = group[v].std()
                
            row[f'{v}_Mean'] = mean
            row[f'{v}_SD'] = std
        
        row['N_obs'] = len(group)
        results.append(row)
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Generate descriptive statistics for the market panel.")
    parser.add_argument('--weight-col', type=str, default=None, help='Column to use for market-weighted statistics (e.g., pop_total)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(PANEL_CSV, low_memory=False)

    # The descriptives must describe the same sample the model is fit on. See utils/window.py.
    df = apply_window(df, label="desc_1")

    # Forward-fill COSIF balance-sheet variables within each conglomerate to cover periods
    # where COSIF reporting has not yet been ingested (e.g. most-recent year after panel
    # extension). This carries the last available quarterly value forward in time.
    _cosif_cols = [c for c in ['total_assets', 'equity_ratio', 'dep_a5'] if c in df.columns]
    if _cosif_cols:
        df = df.sort_values(['CodConglomeradoPrudencial', 'year', 'quarter'])
        df[_cosif_cols] = (
            df.groupby('CodConglomeradoPrudencial')[_cosif_cols]
              .transform(lambda x: x.ffill())
        )

    # 1. Identify Bank Type (D vs B) from the panel's authoritative firm-type column:
    # is_B = 1 for brick-and-mortar, 0 for D (digital / national).
    df['CODMUN_IBGE_str'] = df['CODMUN_IBGE'].astype(str).str.split('.').str[0]
    df['is_B'] = _resolve_is_B(df)
    df['bank_type'] = np.where(df['is_B'], 'B', 'D')
    
    # 2. Extract Region for B-type
    df['region_code'] = df['CODMUN_IBGE_str'].str[0]
    df['region'] = df['region_code'].map(REGION_MAPPING)
    df.loc[df['bank_type'] == 'D', 'region'] = 'National'

    df["total_deposits"] = df[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].sum(axis=1, min_count=1)

    # market_panel stores spread_a{k} and risk_free_qoq as quarterly fractions; the tables
    # report spreads annualized, compounded and in percentage points.
    for k in (1, 2, 3, 4, 5):
        if f'spread_a{k}' in df.columns:
            df[f'spread_pp_a{k}'] = annualized_spread_pp(df[f'spread_a{k}'], df['risk_free_qoq'])

    # Undefined periods are missing, not zero. Type-5 (prepaid) spreads exist only where a
    # type-5 balance is recorded (from 2020Q2), and Pix usage only from the launch (2020Q4);
    # the panel's pre-period placeholders would otherwise average structural zeros into the
    # pooled means.
    post_pix = (df['year'] > 2020) | ((df['year'] == 2020) & (df['quarter'] >= 4))
    if 'spread_pp_a5' in df.columns and 'dep_a5' in df.columns:
        df.loc[df['dep_a5'].isna(), 'spread_pp_a5'] = np.nan
    for c in ('pix_users_pf_per1000', 'pix_txns_pf'):
        if c in df.columns:
            df.loc[~post_pix, c] = np.nan

    # Defining target analytical variables
    vars_to_summarize = [
        'dep_a1', 'dep_a2', 'dep_a3', 'dep_a4', 'dep_a5', 'total_deposits',
        'spread_pp_a1', 'spread_pp_a2', 'spread_pp_a3', 'spread_pp_a4', 'spread_pp_a5',
        'gdp_per_capita', 'pop_total', 'fraction_65plus', 'cadunico_families_per1000',
        'pix_users_pf_per1000', 'pix_txns_pf', 'has_ip', 'connections_per100', 'branches_per1000',
        'total_assets', 'equity_ratio'
    ]

    UNIT_MAP = {
        'dep_a1': 'R$', 'dep_a2': 'R$', 'dep_a3': 'R$', 'dep_a4': 'R$', 'dep_a5': 'R$',
        'total_deposits': 'R$',
        'spread_pp_a1': 'pp p.a.', 'spread_pp_a2': 'pp p.a.', 'spread_pp_a3': 'pp p.a.',
        'spread_pp_a4': 'pp p.a.', 'spread_pp_a5': 'pp p.a.',
        'gdp_per_capita': 'R$', 'pop_total': 'Count', 'fraction_65plus': 'Fraction',
        'cadunico_families_per1000': 'Per 1000', 'pix_users_pf_per1000': 'Per 1000',
        'pix_txns_pf': 'Count', 'has_ip': 'Binary', 'connections_per100': 'Per 100',
        'branches_per1000': 'Per 1000', 'total_assets': 'R$', 'equity_ratio': 'Fraction'
    }

    # Purge any missing columns from the target list safely
    vars_to_summarize = [v for v in vars_to_summarize if v in df.columns]
    
    weight_str = f"_weighted_by_{args.weight_col}" if args.weight_col else "_unweighted"

    print(f"Generating summaries ({'weighted by ' + args.weight_col if args.weight_col else 'unweighted'})...")

    # Overall Summary
    # NOTE: National-by-Year mixes B-type (municipal) and D-type (national-aggregate) rows.
    # D-type rows carry pop_total ~= sum of all MCAs (~191M), so an unweighted mean over
    # rows jumps artificially when D-type rows enter the panel (no D-type in 2013, ~400/yr
    # from 2014 onward). To produce interpretable by-year statistics, we stratify the
    # by-year tables by bank type. The "National Overall" (all years pooled) table is
    # retained for reference.
    print("  -> National Overall (pooled all years)")
    sum_nat = generate_summary(df, None, vars_to_summarize, args.weight_col)

    df_b = df[df['bank_type'] == 'B']
    df_d = df[df['bank_type'] == 'D']

    print("  -> National by Year (all types combined)")
    sum_nat_yr = generate_summary(df, ['year'], vars_to_summarize, args.weight_col)
    print("  -> B-type by Year (municipal markets)")
    sum_nat_yr_b = generate_summary(df_b, ['year'], vars_to_summarize, args.weight_col)
    print("  -> D-type by Year (national fintechs / digital banks)")
    sum_nat_yr_d = generate_summary(df_d, ['year'], vars_to_summarize, args.weight_col)

    # By Bank Type
    print("  -> By Bank Type (All vs B vs D)")
    sum_bt = generate_summary(df, ['bank_type'], vars_to_summarize, args.weight_col)
    # Prepend pooled "All" panel: reuse the National Overall summary, relabel its
    # Group from 'Overall' to 'All', and concatenate so column order is All | B | D.
    sum_bt_all = sum_nat.copy()
    sum_bt_all['Group'] = 'All'
    sum_bt = pd.concat([sum_bt_all, sum_bt], ignore_index=True)

    # By Region (B-Type only)
    print("  -> By Region (B-type)")
    sum_reg = generate_summary(df_b, ['region'], vars_to_summarize, args.weight_col)
    sum_reg_yr = generate_summary(df_b, ['region', 'year'], vars_to_summarize, args.weight_col)

    DRAFTS_DIR = _paths.drafts_dir()
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

    # Where the type-5 and Pix rows are defined (see the blanking above), and what the
    # spread rows average over.
    PERIOD_NOTE = (
        "Spread rows average over all market-quarters, including those where the conglomerate "
        "holds no deposits of that type. Type-5 rows cover the quarters with a recorded type-5 "
        "balance (from 2020Q2) and Pix rows the quarters from 2020Q4."
    )

    # Human-readable labels and display scaling for each variable.
    # Tuple: (display label, display unit string, scale divisor or None)
    # Monetary values are all nominal R$ millions (deposits and assets on one scale).
    LABEL_MAP = {
        'dep_a1':   ('Deposits (1)',                   'R\\$M',        1e6),
        'dep_a2':   ('Deposits (2)',                   'R\\$M',        1e6),
        'dep_a3':   ('Deposits (3)',                   'R\\$M',        1e6),
        'dep_a4':   ('Deposits (4)',                   'R\\$M',        1e6),
        'dep_a5':   ('Deposits (5)',                   'R\\$M',        1e6),
        'total_deposits': ('Total Deposits (1+2+4+5)', 'R\\$M',        1e6),
        # Already annualized percentage points (annualized_spread_pp): no rescaling.
        'spread_pp_a1': ('Spread (1)',                 SPREAD_UNIT,    None),
        'spread_pp_a2': ('Spread (2)',                 SPREAD_UNIT,    None),
        'spread_pp_a3': ('Spread (3)',                 SPREAD_UNIT,    None),
        'spread_pp_a4': ('Spread (4)',                 SPREAD_UNIT,    None),
        'spread_pp_a5': ('Spread (5)',                 SPREAD_UNIT,    None),
        'gdp_per_capita':             ('GDP \\textit{per capita}',   'R\\$',         None),
        'pop_total':                  ('Population',                  'Thousands',    1e3),
        # Shares in PERCENTAGE POINTS, matching the unit their coefficients carry in the
        # sleepiness / BBL tables (utils/state_transform.DISPLAY: fraction_65plus = (0.01,'pp')).
        'fraction_65plus':            ('Share Aged 65+',              'pp',           0.01),
        'cadunico_families_per1000':  ('CadÚnico Families',           'per 1,000',    None),
        'pix_users_pf_per1000':       ('PIX Users (PF)',              'per 1,000',    None),
        'pix_txns_pf':                ('PIX Transactions (PF)',       'Millions',     1e6),
        'has_ip':                     ('Has IP Rate',                 'Indicator',    None),
        # ANATEL active mobile-telephony accesses, all technologies (scrape_anatel_mobile.py).
        'connections_per100':         ('Mobile Lines',                'per 100 inhabitants', None),
        'branches_per1000':           ('Bank Branches',               'per 1,000',    None),
        'total_assets':               ('Total Assets',                'R\\$M',        1e6),
        'equity_ratio':               ('Equity Ratio',                '',             None),
    }

    def _fmt_num(val, var_base, compact=False):
        """Format a cell value according to the variable's scale and unit."""
        if pd.isna(val):
            return '--'
        info = LABEL_MAP.get(var_base)
        if info is None:
            return f"{val:.3f}"
        _lbl, unit, scale = info
        v = val / scale if scale is not None else val
        if unit in ('R\\$M', 'R\\$B', 'Thousands', 'Millions'):
            return f"{v:,.0f}" if compact else f"{v:,.2f}"
        elif unit in ('pp', 'bp', SPREAD_UNIT):
            return f"{v:.2f}"
        elif unit == 'R\\$':
            return f"{v:,.0f}" if compact else f"{v:,.2f}"
        else:
            return f"{v:,.2f}" if compact else f"{v:,.3f}"

    def _fmt_val(val, var_base, compact=False):
        # A value that rounds to zero prints unsigned ("-0.00" carries no sign information).
        s = _fmt_num(val, var_base, compact)
        if s.startswith('-') and s[1:].replace(',', '').replace('.', '').strip('0') == '':
            s = s[1:]
        return s

    def _esc(s):
        return str(s).replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')

    # Batch save output forms
    def save_output(res_df, name, caption_title, tex_name=None, append=False):
        out_path = OUTPUT_DIR / f"{name}{weight_str}.csv"
        res_df.to_csv(out_path, index=False)

        _tex_name       = tex_name if tex_name is not None else name
        tex_path        = DRAFTS_DIR  / f"{_tex_name}{weight_str}.tex"
        try:
            transposed = res_df.set_index('Group').T
            groups      = list(transposed.columns)
            n_groups    = len(groups)
            n_cols      = n_groups + 1          # label col + one col per group

            # Wide tables (many columns) use xltabular with auto-fit X columns and
            # compact number formatting so cells never overflow the page width.
            wide    = n_groups > 8
            compact = wide
            if wide:
                col_spec  = 'l@{\\hspace{0.2em}}' + '>{\\centering\\arraybackslash}X' * n_groups
            else:
                col_spec  = 'l@{\\hspace{0.35em}}' + '>{\\centering\\arraybackslash}X' * n_groups
            # Fixed width scaled to the column count so the footnote can match the
            # table width exactly; X columns share it evenly (no lopsided last gap).
            narrow_w = f'{min(1.0, 0.40 + 0.12 * n_groups):.2f}\\textwidth'

            weight_label = (' (Population Weighted)'
                            if args.weight_col else ' (Unweighted)')
            full_caption = caption_title + weight_label
            tab_label    = f"tab:{name}{weight_str}"
            group_header = ' & '.join([_esc(g) for g in groups])

            # Parse rows into (var_base, stat_type, {group: value}) triples
            stat_rows = []
            for idx in transposed.index:
                vals = {g: transposed.loc[idx, g] for g in groups}
                if idx == 'N_obs':
                    stat_rows.append(('N_obs', 'N_obs', vals))
                elif idx.endswith('_Mean'):
                    stat_rows.append((idx[:-5], 'Mean', vals))
                elif idx.endswith('_SD'):
                    stat_rows.append((idx[:-3], 'SD',   vals))

            # Ordered unique variable bases (preserving first-occurrence order)
            seen_vars, var_order = set(), []
            for vb, stat, _ in stat_rows:
                if vb != 'N_obs' and vb not in seen_vars:
                    var_order.append(vb)
                    seen_vars.add(vb)

            # Drop variables that are entirely missing across every group (fix C):
            # keeps tables compact and removes rows that read as "--" everywhere.
            def _all_missing(vb):
                mean_row = next((r for r in stat_rows if r[0] == vb and r[1] == 'Mean'), None)
                if mean_row is None:
                    return True
                return all(pd.isna(mean_row[2][g]) for g in groups)
            var_order = [vb for vb in var_order if not _all_missing(vb)]

            # Build data row strings
            row_lines = []
            for i, vb in enumerate(var_order):
                info = LABEL_MAP.get(vb)
                if info:
                    lbl, unit, _ = info
                    cell_label = lbl + (f' ({unit})' if unit else '')
                else:
                    cell_label = _esc(vb)

                mean_row = next((r for r in stat_rows if r[0] == vb and r[1] == 'Mean'), None)
                sd_row   = next((r for r in stat_rows if r[0] == vb and r[1] == 'SD'),   None)

                if mean_row:
                    vals_str = ' & '.join([_fmt_val(mean_row[2][g], vb, compact) for g in groups])
                    row_lines.append(f"    {cell_label} & {vals_str} \\\\*")
                if sd_row:
                    vals_str = ' & '.join([f"({_fmt_val(sd_row[2][g], vb, compact)})" for g in groups])
                    row_lines.append(f"    & {vals_str} \\\\")
                if i < len(var_order) - 1:
                    row_lines.append('    \\addlinespace[0.3em]')

            # Observations row
            n_obs_row = next((r for r in stat_rows if r[0] == 'N_obs'), None)
            if n_obs_row:
                row_lines.append('    \\midrule')
                vals_str = ' & '.join(
                    [f"{int(n_obs_row[2][g]):,}" if not pd.isna(n_obs_row[2][g]) else '--'
                     for g in groups]
                )
                row_lines.append(f"    Observations & {vals_str} \\\\")

            rows_text = '\n'.join(row_lines)

            weight_note = (f" Population-weighted using \\texttt{{{_esc(args.weight_col)}}}."
                           if args.weight_col else "")
            notes_text = (
                f"\\scriptsize \\textit{{Notes:}} Means are reported with standard deviations "
                f"in parentheses below, computed over all market-quarter observations. "
                f"Monetary values are nominal R\\$ millions. {SPREAD_CONVENTION_TEX}"
                f"{PERIOD_NOTE}{weight_note}"
            )

            font_cmd  = '\\tiny'       if wide else '\\footnotesize'
            tabcolsep = '1pt'          if wide else '3pt'
            begin_env = (f'\\begin{{xltabular}}{{\\textwidth}}{{{col_spec}}}'
                         if wide else f'\\begin{{xltabular}}{{{narrow_w}}}{{{col_spec}}}')
            end_env   = '\\end{xltabular}'
            note_w    = (r'\dimexpr\textwidth-2\tabcolsep\relax' if wide
                         else f'\\dimexpr{narrow_w}-2\\tabcolsep\\relax')

            lines = [
                '\\setstretch{1.0}',
                '\\setlength{\\LTleft}{\\fill}',
                '\\setlength{\\LTright}{\\fill}',
                '\\begingroup',
                font_cmd,
                f'\\setlength{{\\tabcolsep}}{{{tabcolsep}}}',
                begin_env,
                f'    \\caption{{{full_caption}}}\\label{{{tab_label}}} \\\\',
                '    \\toprule',
                f'    & {group_header} \\\\',
                '    \\midrule',
                '    \\endfirsthead',
                '',
                f'    \\multicolumn{{{n_cols}}}{{c}}{{{{\\bfseries Table \\thetable\\ continued from previous page}}}} \\\\',
                '    \\toprule',
                f'    & {group_header} \\\\',
                '    \\midrule',
                '    \\endhead',
                '',
                '    \\midrule',
                f'    \\multicolumn{{{n_cols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\',
                '    \\endfoot',
                '',
                '    \\bottomrule',
                f'    \\multicolumn{{{n_cols}}}{{p{{{note_w}}}}}{{{notes_text}}} \\\\',
                '    \\endlastfoot',
                '',
                rows_text,
                '',
                end_env,
                '\\endgroup',
            ]
            latex_str = '\n'.join(lines)

            for path in [tex_path]:
                mode = 'a' if append else 'w'
                with open(path, mode, encoding='utf-8') as f:
                    if append:
                        f.write('\n\n\\bigskip\n\n')
                    f.write(latex_str)

            print(f"  Saved: {tex_path.name}")
        except Exception as e:
            print(f"Failed to generate latex for {name}: {e}")
            import traceback; traceback.print_exc()

    def save_landscape_by_year(summary_df, name, caption_title, panel_descr=""):
        """Landscape, multi-page, single-panel xltabular indexed by calendar year.

        Same stylistic conventions as ``save_banktype_by_year_combined``:
          * ``\\begin{landscape}`` + ``xltabular`` (handles page breaks).
          * Labels: ``Deposits (A*)`` -> ``Deposits (*)``; ``Spread (A*)`` -> ``Spread (*)``.
          * Drops ``has_ip`` row.
          * Pre-2020 cells for PIX/A5 variables render as blank ``\\multirow{2}{*}{}``.
          * Generic rule: cells where both Mean and SD are NaN or 0 render blank.
        """
        tab_label = f"tab:{name}{weight_str}"
        summary_df.to_csv(OUTPUT_DIR / f"{name}{weight_str}.csv", index=False)

        PRE2020_EMPTY = {"pix_users_pf_per1000", "pix_txns_pf", "spread_a5", "dep_a5"}
        EXCLUDE_VARS  = {"has_ip"}
        compact = True

        def _override_label(vb):
            info = LABEL_MAP.get(vb)
            if info is None:
                return _esc(vb)
            lbl, unit, _ = info
            if vb.startswith("dep_a"):
                lbl = f"Deposits ({vb.split('_a')[-1]})"
            elif vb.startswith("spread_a"):
                lbl = f"Spread ({vb.split('_a')[-1]})"
            return lbl + (f" ({unit})" if unit else "")

        def _is_blank_cell(vb, group, mean_val, sd_val):
            if vb in PRE2020_EMPTY:
                try:
                    if int(group) < 2020:
                        return True
                except (TypeError, ValueError):
                    pass
            m_empty = pd.isna(mean_val) or float(mean_val) == 0.0
            s_empty = pd.isna(sd_val)   or float(sd_val)   == 0.0
            return m_empty and s_empty

        years = sorted(list(summary_df["Group"]), key=lambda x: int(x))
        n_groups = len(years)
        n_cols   = n_groups + 1

        vars_to_include = []
        for col in summary_df.columns:
            if col.endswith("_Mean"):
                vb = col[:-5]
                if vb in EXCLUDE_VARS or vb in vars_to_include:
                    continue
                vars_to_include.append(vb)

        gset = summary_df.set_index("Group")

        def _all_missing(vb):
            col = f"{vb}_Mean"
            return col not in gset.columns or gset[col].isna().all()
        var_order = [v for v in vars_to_include if not _all_missing(v)]

        body = []
        for i, vb in enumerate(var_order):
            cell_label = _override_label(vb)
            mean_cells, sd_cells = [], []
            for g in years:
                mean_val = gset.loc[g, f"{vb}_Mean"] if f"{vb}_Mean" in gset.columns else float("nan")
                sd_val   = gset.loc[g, f"{vb}_SD"]   if f"{vb}_SD"   in gset.columns else float("nan")
                if _is_blank_cell(vb, g, mean_val, sd_val):
                    mean_cells.append(r"\multirow{2}{*}{}")
                    sd_cells.append("")
                else:
                    mean_cells.append(_fmt_val(mean_val, vb, compact))
                    sd_cells.append(f"({_fmt_val(sd_val, vb, compact)})")
            body.append(f"    {cell_label} & " + " & ".join(mean_cells) + r" \\*")
            body.append("    & " + " & ".join(sd_cells) + r" \\")
            if i < len(var_order) - 1:
                body.append(r"    \addlinespace[0.3em]")

        if "N_obs" in gset.columns:
            body.append(r"    \midrule")
            nvals = []
            for g in years:
                v = gset.loc[g, "N_obs"]
                nvals.append(f"{int(v):,}" if not pd.isna(v) else "--")
            body.append("    Observations & " + " & ".join(nvals) + r" \\")

        col_spec = "l@{\\hspace{0.2em}}" + ">{\\centering\\arraybackslash}X" * n_groups
        weight_label = (" (Population Weighted)"
                        if args.weight_col else " (Unweighted)")
        full_caption = caption_title + weight_label
        group_header = " & ".join([_esc(str(g)) for g in years])

        weight_note = (f" Population-weighted using \\texttt{{{_esc(args.weight_col)}}}."
                       if args.weight_col else "")
        scope_note = f" {panel_descr}" if panel_descr else ""
        notes_text = (
            "\\scriptsize \\textit{Notes:} Means are reported with standard deviations in "
            "parentheses immediately below, computed over market-quarter observations within "
            f"each calendar year.{scope_note} Cells left blank denote variables that are "
            "undefined or unobserved for that year (Pix usage and type-5 products are not "
            f"defined before 2020). Monetary values are nominal R\\$ millions. {SPREAD_CONVENTION_TEX}"
            f"{PERIOD_NOTE}{weight_note}"
        )

        lines = [
            r"\begin{landscape}",
            r"\setstretch{1.0}",
            r"\setlength{\LTleft}{\fill}",
            r"\setlength{\LTright}{\fill}",
            r"\begingroup",
            r"\tiny",
            r"\setlength{\tabcolsep}{1pt}",
            f"\\begin{{xltabular}}{{\\linewidth}}{{{col_spec}}}",
            f"    \\caption{{{full_caption}}}\\label{{{tab_label}}} \\\\",
            r"    \toprule",
            f"    & {group_header} \\\\",
            r"    \midrule",
            r"    \endfirsthead",
            "",
            f"    \\multicolumn{{{n_cols}}}{{c}}{{{{\\bfseries Table \\thetable\\ continued from previous page}}}} \\\\",
            r"    \toprule",
            f"    & {group_header} \\\\",
            r"    \midrule",
            r"    \endhead",
            "",
            r"    \midrule",
            f"    \\multicolumn{{{n_cols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\",
            r"    \endfoot",
            "",
            r"    \bottomrule",
            f"    \\multicolumn{{{n_cols}}}{{p{{\\dimexpr\\textwidth-2\\tabcolsep\\relax}}}}{{{notes_text}}} \\\\",
            r"    \endlastfoot",
            "",
            *body,
            "",
            r"\end{xltabular}",
            r"\endgroup",
            r"\end{landscape}",
            r"\doublespacing",
        ]
        latex_str = "\n".join(lines)

        tex_path        = DRAFTS_DIR / f"{name}{weight_str}.tex"
        for path in [tex_path]:
            with open(path, "w", encoding="utf-8") as f:
                f.write(latex_str)
        print(f"  Saved landscape by-year table: {tex_path.name}")

    def save_banktype_by_year_combined(df_b_summary, df_d_summary):
        """Build a single landscape, two-panel (B / D) xltabular that breaks across pages.

        Customisations vs. the generic ``save_output``:
          * Landscape via ``pdflscape``'s ``landscape`` env (preserves page breaks).
          * Strips ``A`` from deposit/spread labels: ``Deposits (A1)`` -> ``Deposits (1)``.
          * Drops the ``has_ip`` row entirely.
          * Cells where both mean and SD are NaN/0, and PIX / A5 cells for years < 2020,
            are rendered as a single empty ``\\multirow{2}{*}{}`` spanning Mean+SD.
        """
        name          = "Summary_BankType_by_Year"
        caption_title = "Summary Statistics by Bank Type and Year"
        tab_label     = f"tab:{name}{weight_str}"

        # Persist raw CSVs for both panels (preserve prior behaviour)
        df_b_summary.to_csv(OUTPUT_DIR / f"{name}_B{weight_str}.csv", index=False)
        df_d_summary.to_csv(OUTPUT_DIR / f"{name}_D{weight_str}.csv", index=False)

        PRE2020_EMPTY = {"pix_users_pf_per1000", "pix_txns_pf", "spread_a5", "dep_a5"}
        EXCLUDE_VARS  = {"has_ip"}

        def _override_label(vb):
            info = LABEL_MAP.get(vb)
            if info is None:
                return _esc(vb)
            lbl, unit, _ = info
            if vb.startswith("dep_a"):
                lbl = f"Deposits ({vb.split('_a')[-1]})"
            elif vb.startswith("spread_a"):
                lbl = f"Spread ({vb.split('_a')[-1]})"
            return lbl + (f" ({unit})" if unit else "")

        def _is_blank_cell(vb, group, mean_val, sd_val):
            # Hard rule: PIX & A5 variables undefined before 2020
            if vb in PRE2020_EMPTY:
                try:
                    if int(group) < 2020:
                        return True
                except (TypeError, ValueError):
                    pass
            # Generic rule: both stats missing or exactly zero
            m_empty = pd.isna(mean_val) or float(mean_val) == 0.0
            s_empty = pd.isna(sd_val)   or float(sd_val)   == 0.0
            return m_empty and s_empty

        compact = True

        # Use union of years across both panels so headers align across panels
        years_b = list(df_b_summary["Group"])
        years_d = list(df_d_summary["Group"])
        years   = sorted(set(years_b) | set(years_d), key=lambda x: int(x))
        n_groups = len(years)
        n_cols   = n_groups + 1

        # Variable order: take from B summary's column list, then drop excluded
        # and variables that are entirely missing across BOTH panels.
        vars_to_include = []
        for col in df_b_summary.columns:
            if col.endswith("_Mean"):
                vb = col[:-5]
                if vb in EXCLUDE_VARS or vb in vars_to_include:
                    continue
                vars_to_include.append(vb)

        def _all_missing(vb):
            for df_s in (df_b_summary, df_d_summary):
                col = f"{vb}_Mean"
                if col in df_s.columns and not df_s[col].isna().all():
                    return False
            return True
        var_order = [v for v in vars_to_include if not _all_missing(v)]

        def _build_panel_rows(summary_df):
            gset = summary_df.set_index("Group")
            out = []
            for i, vb in enumerate(var_order):
                cell_label = _override_label(vb)
                mean_cells, sd_cells = [], []
                for g in years:
                    mean_val = sd_val = float("nan")
                    if g in gset.index:
                        if f"{vb}_Mean" in gset.columns:
                            mean_val = gset.loc[g, f"{vb}_Mean"]
                        if f"{vb}_SD" in gset.columns:
                            sd_val = gset.loc[g, f"{vb}_SD"]
                    if _is_blank_cell(vb, g, mean_val, sd_val):
                        mean_cells.append(r"\multirow{2}{*}{}")
                        sd_cells.append("")
                    else:
                        mean_cells.append(_fmt_val(mean_val, vb, compact))
                        sd_cells.append(f"({_fmt_val(sd_val, vb, compact)})")
                out.append(f"    {cell_label} & " + " & ".join(mean_cells) + r" \\*")
                out.append("    & " + " & ".join(sd_cells) + r" \\")
                if i < len(var_order) - 1:
                    out.append(r"    \addlinespace[0.3em]")
            # Observations row (per panel)
            if "N_obs" in gset.columns:
                out.append(r"    \midrule")
                nvals = []
                for g in years:
                    if g in gset.index and not pd.isna(gset.loc[g, "N_obs"]):
                        nvals.append(f"{int(gset.loc[g, 'N_obs']):,}")
                    else:
                        nvals.append("--")
                out.append("    Observations & " + " & ".join(nvals) + r" \\")
            return out

        col_spec = "l@{\\hspace{0.2em}}" + ">{\\centering\\arraybackslash}X" * n_groups
        weight_label = (" (Population Weighted)"
                        if args.weight_col else " (Unweighted)")
        full_caption = caption_title + weight_label
        group_header = " & ".join([_esc(str(g)) for g in years])

        weight_note = (f" Population-weighted using \\texttt{{{_esc(args.weight_col)}}}."
                       if args.weight_col else "")
        notes_text = (
            "\\scriptsize \\textit{Notes:} Means are reported with standard deviations in "
            "parentheses immediately below, computed over market-quarter observations within "
            "each calendar year. Panel A reports B-type firms (municipal deposit markets); "
            "Panel B reports D-type firms (national digital banks). Cells left blank denote "
            "variables that are undefined or unobserved for that year (Pix usage and "
            "type-5 products are not defined before 2020). Monetary values are nominal "
            f"R\\$ millions. {SPREAD_CONVENTION_TEX}{PERIOD_NOTE}{weight_note}"
        )

        panel_a_rows = _build_panel_rows(df_b_summary)
        panel_b_rows = _build_panel_rows(df_d_summary)

        lines = [
            r"\begin{landscape}",
            r"\setstretch{1.0}",
            r"\setlength{\LTleft}{\fill}",
            r"\setlength{\LTright}{\fill}",
            r"\begingroup",
            r"\tiny",
            r"\setlength{\tabcolsep}{1pt}",
            f"\\begin{{xltabular}}{{\\linewidth}}{{{col_spec}}}",
            f"    \\caption{{{full_caption}}}\\label{{{tab_label}}} \\\\",
            r"    \toprule",
            f"    & {group_header} \\\\",
            r"    \midrule",
            r"    \endfirsthead",
            "",
            f"    \\multicolumn{{{n_cols}}}{{c}}{{{{\\bfseries Table \\thetable\\ continued from previous page}}}} \\\\",
            r"    \toprule",
            f"    & {group_header} \\\\",
            r"    \midrule",
            r"    \endhead",
            "",
            r"    \midrule",
            f"    \\multicolumn{{{n_cols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\",
            r"    \endfoot",
            "",
            r"    \bottomrule",
            f"    \\multicolumn{{{n_cols}}}{{p{{\\dimexpr\\textwidth-2\\tabcolsep\\relax}}}}{{{notes_text}}} \\\\",
            r"    \endlastfoot",
            "",
            f"    \\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{Panel A: B-Type Firms (Municipal Markets)}}}} \\\\",
            r"    \midrule",
            *panel_a_rows,
            r"    \midrule",
            r"    \addlinespace[0.6em]",
            f"    \\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{Panel B: D-Type Firms (National Digital Banks)}}}} \\\\",
            r"    \midrule",
            *panel_b_rows,
            "",
            r"\end{xltabular}",
            r"\endgroup",
            r"\end{landscape}",
            r"\doublespacing",
        ]
        latex_str = "\n".join(lines)

        tex_path        = DRAFTS_DIR / f"{name}{weight_str}.tex"
        for path in [tex_path]:
            with open(path, "w", encoding="utf-8") as f:
                f.write(latex_str)
        print(f"  Saved combined two-panel landscape table: {tex_path.name}")

    def save_banktype_summary(summary_df):
        """Landscape, multi-page xltabular for the B-vs-D summary.

        Applies the same stylistic conventions as ``save_banktype_by_year_combined``:
          * Landscape via ``pdflscape``.
          * Strips the ``A`` from ``Deposits (A*)`` / ``Spread (A*)`` labels.
          * Drops the ``has_ip`` row.
          * Blanks (\\multirow{2}{*}{}) cells where both mean and SD are NaN or 0.

        The pre-2020 PIX / A5 rule is inapplicable here (no year axis); those rows
        will simply blank out via the generic zero/NaN rule when undefined.
        """
        name          = "Summary_BankType"
        caption_title = "Summary Statistics by Bank Type"
        tab_label     = f"tab:{name}{weight_str}"

        # Persist raw CSV (preserve prior behaviour)
        summary_df.to_csv(OUTPUT_DIR / f"{name}{weight_str}.csv", index=False)

        EXCLUDE_VARS = {"has_ip"}
        compact      = False  # only 2 group columns; full-precision formatting fits

        def _override_label(vb):
            info = LABEL_MAP.get(vb)
            if info is None:
                return _esc(vb)
            lbl, unit, _ = info
            if vb.startswith("dep_a"):
                lbl = f"Deposits ({vb.split('_a')[-1]})"
            elif vb.startswith("spread_a"):
                lbl = f"Spread ({vb.split('_a')[-1]})"
            return lbl + (f" ({unit})" if unit else "")

        # Population is the MCA population on B rows and the national population on D rows
        # (the D firms' single national market), so a pooled mean mixes two different
        # geographic units: the "All" cell is left blank.
        NOT_POOLED = {"pop_total"}

        def _is_blank_cell(vb, group, mean_val, sd_val):
            if group == "All" and vb in NOT_POOLED:
                return True
            m_empty = pd.isna(mean_val) or float(mean_val) == 0.0
            s_empty = pd.isna(sd_val)   or float(sd_val)   == 0.0
            return m_empty and s_empty

        groups   = list(summary_df["Group"])
        n_groups = len(groups)
        n_cols   = n_groups + 1

        vars_to_include = []
        for col in summary_df.columns:
            if col.endswith("_Mean"):
                vb = col[:-5]
                if vb in EXCLUDE_VARS or vb in vars_to_include:
                    continue
                vars_to_include.append(vb)

        gset = summary_df.set_index("Group")

        def _all_missing(vb):
            col = f"{vb}_Mean"
            return col not in gset.columns or gset[col].isna().all()
        var_order = [v for v in vars_to_include if not _all_missing(v)]

        body = []
        for i, vb in enumerate(var_order):
            cell_label = _override_label(vb)
            mean_cells, sd_cells = [], []
            for g in groups:
                mean_val = gset.loc[g, f"{vb}_Mean"] if f"{vb}_Mean" in gset.columns else float("nan")
                sd_val   = gset.loc[g, f"{vb}_SD"]   if f"{vb}_SD"   in gset.columns else float("nan")
                if _is_blank_cell(vb, g, mean_val, sd_val):
                    mean_cells.append(r"\multirow{2}{*}{}")
                    sd_cells.append("")
                else:
                    mean_cells.append(_fmt_val(mean_val, vb, compact))
                    sd_cells.append(f"({_fmt_val(sd_val, vb, compact)})")
            body.append(f"    {cell_label} & " + " & ".join(mean_cells) + r" \\*")
            body.append("    & " + " & ".join(sd_cells) + r" \\")
            if i < len(var_order) - 1:
                body.append(r"    \addlinespace[0.3em]")

        if "N_obs" in gset.columns:
            body.append(r"    \midrule")
            nvals = []
            for g in groups:
                v = gset.loc[g, "N_obs"]
                nvals.append(f"{int(v):,}" if not pd.isna(v) else "--")
            body.append("    Observations & " + " & ".join(nvals) + r" \\")

        # Portrait, fixed-width so the footnote (below) can match the table width
        # exactly; X data columns share the width evenly (no lopsided last gap).
        table_w  = r"0.80\textwidth"
        col_spec = "l@{\\hspace{0.35em}}" + ">{\\centering\\arraybackslash}X" * n_groups
        weight_label = (" (Population Weighted)"
                        if args.weight_col else " (Unweighted)")
        full_caption = caption_title + weight_label
        group_header = " & ".join([_esc(str(g)) for g in groups])

        weight_note = (f" Population-weighted using \\texttt{{{_esc(args.weight_col)}}}."
                       " The unweighted version of this table is available upon request."
                       if args.weight_col else "")
        notes_text = (
            "\\scriptsize \\textit{Notes:} Means are reported with standard deviations in "
            "parentheses immediately below, computed over all market-quarter observations, "
            f"{MIN_YEAR}--{MAX_YEAR}. Column ``B'' covers B firms in their MCA deposit markets; "
            "column ``D'' covers D firms, whose single national market carries the national "
            "population and population-weighted national averages of the other market-level "
            "variables; column ``All'' pools both. Population is therefore the MCA population "
            "for B and the national population for D, and is not pooled (blank under ``All''). "
            "Monetary values are nominal R\\$ millions. "
            f"{SPREAD_CONVENTION_TEX}{PERIOD_NOTE}{weight_note}"
        )

        lines = [
            r"\setstretch{1.0}",
            r"\setlength{\LTleft}{\fill}",
            r"\setlength{\LTright}{\fill}",
            r"\begingroup",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{6pt}",
            f"\\begin{{xltabular}}{{{table_w}}}{{{col_spec}}}",
            f"    \\caption{{{full_caption}}}\\label{{{tab_label}}} \\\\",
            r"    \toprule",
            f"    & {group_header} \\\\",
            r"    \midrule",
            r"    \endfirsthead",
            "",
            f"    \\multicolumn{{{n_cols}}}{{c}}{{{{\\bfseries Table \\thetable\\ continued from previous page}}}} \\\\",
            r"    \toprule",
            f"    & {group_header} \\\\",
            r"    \midrule",
            r"    \endhead",
            "",
            r"    \midrule",
            f"    \\multicolumn{{{n_cols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\",
            r"    \endfoot",
            "",
            r"    \bottomrule",
            f"    \\multicolumn{{{n_cols}}}{{p{{\\dimexpr{table_w}-2\\tabcolsep\\relax}}}}{{{notes_text}}} \\\\",
            r"    \endlastfoot",
            "",
            *body,
            "",
            r"\end{xltabular}",
            r"\endgroup",
            r"\doublespacing",
        ]
        latex_str = "\n".join(lines)

        tex_path        = DRAFTS_DIR / f"{name}{weight_str}.tex"
        for path in [tex_path]:
            with open(path, "w", encoding="utf-8") as f:
                f.write(latex_str)
        print(f"  Saved BankType table: {tex_path.name}")

    def save_region_by_year_panels(summary_df):
        """Landscape, multi-page xltabular with one Panel per macro-region.

        The raw ``Summary_Region_by_Year`` dataframe carries one row per
        (region, year) tuple, flattened to a ``"Region - Year"`` string in the
        ``Group`` column. The generic renderer produces a >60-column mess; here
        we split it into five vertically-stacked panels (one per region) with
        years as columns, matching the conventions used elsewhere:
          * Strip ``A`` from deposit / spread labels.
          * Drop the ``has_ip`` row.
          * Pre-2020 PIX / A5 cells render as blank ``\\multirow{2}{*}{}``.
          * Generic rule: cells where both Mean and SD are NaN or 0 render blank.
        """
        name          = "Summary_Region_by_Year"
        caption_title = "Summary Statistics by Region and Year (B-Type Firms)"
        tab_label     = f"tab:{name}{weight_str}"
        summary_df.to_csv(OUTPUT_DIR / f"{name}{weight_str}.csv", index=False)

        PRE2020_EMPTY = {"pix_users_pf_per1000", "pix_txns_pf", "spread_a5", "dep_a5"}
        EXCLUDE_VARS  = {"has_ip"}
        compact = True

        def _override_label(vb):
            info = LABEL_MAP.get(vb)
            if info is None:
                return _esc(vb)
            lbl, unit, _ = info
            if vb.startswith("dep_a"):
                lbl = f"Deposits ({vb.split('_a')[-1]})"
            elif vb.startswith("spread_a"):
                lbl = f"Spread ({vb.split('_a')[-1]})"
            return lbl + (f" ({unit})" if unit else "")

        def _is_blank_cell(vb, year, mean_val, sd_val):
            if vb in PRE2020_EMPTY:
                try:
                    if int(year) < 2020:
                        return True
                except (TypeError, ValueError):
                    pass
            m_empty = pd.isna(mean_val) or float(mean_val) == 0.0
            s_empty = pd.isna(sd_val)   or float(sd_val)   == 0.0
            return m_empty and s_empty

        # Split the flat "Region - Year" rows into region -> {year: row}
        parsed = summary_df.copy()
        split  = parsed["Group"].astype(str).str.split(" - ", n=1, expand=True)
        parsed["_region"] = split[0]
        parsed["_year"]   = pd.to_numeric(split[1], errors="coerce").astype("Int64")
        parsed = parsed.dropna(subset=["_year"]).copy()
        parsed["_year"] = parsed["_year"].astype(int)

        # Ordered region list (north -> south, then center-west); only keep those present
        REGION_ORDER = ["North", "Northeast", "Southeast", "South", "Center-West"]
        regions      = [r for r in REGION_ORDER if r in set(parsed["_region"])]
        years        = sorted(parsed["_year"].unique().tolist())
        n_groups     = len(years)
        n_cols       = n_groups + 1

        vars_to_include = []
        for col in summary_df.columns:
            if col.endswith("_Mean"):
                vb = col[:-5]
                if vb in EXCLUDE_VARS or vb in vars_to_include:
                    continue
                vars_to_include.append(vb)

        def _all_missing(vb):
            col = f"{vb}_Mean"
            return col not in summary_df.columns or summary_df[col].isna().all()
        var_order = [v for v in vars_to_include if not _all_missing(v)]

        def _panel_rows(region_df):
            gset = region_df.set_index("_year")
            out = []
            for i, vb in enumerate(var_order):
                cell_label = _override_label(vb)
                mean_cells, sd_cells = [], []
                for y in years:
                    mean_val = sd_val = float("nan")
                    if y in gset.index:
                        if f"{vb}_Mean" in gset.columns:
                            mean_val = gset.loc[y, f"{vb}_Mean"]
                        if f"{vb}_SD" in gset.columns:
                            sd_val = gset.loc[y, f"{vb}_SD"]
                    if _is_blank_cell(vb, y, mean_val, sd_val):
                        mean_cells.append(r"\multirow{2}{*}{}")
                        sd_cells.append("")
                    else:
                        mean_cells.append(_fmt_val(mean_val, vb, compact))
                        sd_cells.append(f"({_fmt_val(sd_val, vb, compact)})")
                out.append(f"    {cell_label} & " + " & ".join(mean_cells) + r" \\*")
                out.append("    & " + " & ".join(sd_cells) + r" \\")
                if i < len(var_order) - 1:
                    out.append(r"    \addlinespace[0.3em]")
            if "N_obs" in gset.columns:
                out.append(r"    \midrule")
                nvals = []
                for y in years:
                    if y in gset.index and not pd.isna(gset.loc[y, "N_obs"]):
                        nvals.append(f"{int(gset.loc[y, 'N_obs']):,}")
                    else:
                        nvals.append("--")
                out.append("    Observations & " + " & ".join(nvals) + r" \\")
            return out

        col_spec = "l@{\\hspace{0.2em}}" + ">{\\centering\\arraybackslash}X" * n_groups
        weight_label = (" (Population Weighted)"
                        if args.weight_col else " (Unweighted)")
        full_caption = caption_title + weight_label
        group_header = " & ".join([_esc(str(y)) for y in years])

        weight_note = (f" Population-weighted using \\texttt{{{_esc(args.weight_col)}}}."
                       if args.weight_col else "")
        notes_text = (
            "\\scriptsize \\textit{Notes:} Means are reported with standard deviations in "
            "parentheses immediately below, computed over market-quarter observations within "
            "each region-year cell, restricted to B-type firms (municipal deposit markets). "
            "Each panel covers one of the five Brazilian macro-regions. Cells left blank denote "
            "variables that are undefined or unobserved for that year (Pix usage and "
            "type-5 products are not defined before 2020). Monetary values are nominal "
            f"R\\$ millions. {SPREAD_CONVENTION_TEX}{PERIOD_NOTE}{weight_note}"
        )

        panel_blocks = []
        for k, region in enumerate(regions):
            region_df = parsed[parsed["_region"] == region]
            panel_letter = chr(ord('A') + k)
            panel_blocks.append(
                f"    \\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{Panel {panel_letter}: {_esc(region)}}}}} \\\\"
            )
            panel_blocks.append(r"    \midrule")
            panel_blocks.extend(_panel_rows(region_df))
            if k < len(regions) - 1:
                panel_blocks.append(r"    \midrule")
                panel_blocks.append(r"    \addlinespace[0.6em]")

        lines = [
            r"\begin{landscape}",
            r"\setstretch{1.0}",
            r"\setlength{\LTleft}{\fill}",
            r"\setlength{\LTright}{\fill}",
            r"\begingroup",
            r"\tiny",
            r"\setlength{\tabcolsep}{1pt}",
            f"\\begin{{xltabular}}{{\\linewidth}}{{{col_spec}}}",
            f"    \\caption{{{full_caption}}}\\label{{{tab_label}}} \\\\",
            r"    \toprule",
            f"    & {group_header} \\\\",
            r"    \midrule",
            r"    \endfirsthead",
            "",
            f"    \\multicolumn{{{n_cols}}}{{c}}{{{{\\bfseries Table \\thetable\\ continued from previous page}}}} \\\\",
            r"    \toprule",
            f"    & {group_header} \\\\",
            r"    \midrule",
            r"    \endhead",
            "",
            r"    \midrule",
            f"    \\multicolumn{{{n_cols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\",
            r"    \endfoot",
            "",
            r"    \bottomrule",
            f"    \\multicolumn{{{n_cols}}}{{p{{\\dimexpr\\textwidth-2\\tabcolsep\\relax}}}}{{{notes_text}}} \\\\",
            r"    \endlastfoot",
            "",
            *panel_blocks,
            "",
            r"\end{xltabular}",
            r"\endgroup",
            r"\end{landscape}",
            r"\doublespacing",
        ]
        latex_str = "\n".join(lines)

        tex_path        = DRAFTS_DIR / f"{name}{weight_str}.tex"
        for path in [tex_path]:
            with open(path, "w", encoding="utf-8") as f:
                f.write(latex_str)
        print(f"  Saved per-region landscape table: {tex_path.name}")

    # Note: Summary_National (pooled), Summary_National_byYear_B, and
    # Summary_National_byYear_D are intentionally NOT saved as standalone tables.
    # They are fully redundant with (a) the "All" column of Summary_BankType and
    # (b) the B/D panels of Summary_BankType_by_Year. The underlying summaries
    # (sum_nat, sum_nat_yr_b, sum_nat_yr_d) are still computed because they feed
    # those composite tables.
    save_landscape_by_year(sum_nat_yr,   "Summary_National_by_Year",
                           "National Summary Statistics by Year",
                           panel_descr="All bank types pooled (B-type municipal markets and D-type national digital banks).")
    save_banktype_summary(sum_bt)
    save_banktype_by_year_combined(sum_nat_yr_b, sum_nat_yr_d)
    save_output(sum_reg,        "Summary_Region",             "Summary Statistics by Region (B-Type Firms)")
    save_region_by_year_panels(sum_reg_yr)

    print("Generating D-Type firm summary...")
    d_firms = df[~df["is_B"]].copy()
    d_firms["cnpj_8"] = pd.to_numeric(d_firms["CNPJ_Lider"], errors="coerce").astype("Int64").astype(str).str.zfill(8)

    path_flags = list(_ROOT.glob("**/digital_banks_diagnostic.csv"))[0]
    df_flags = pd.read_csv(path_flags)
    dig_cands = set(df_flags[df_flags["is_digital_candidate"] == True]["CNPJ_root"].astype(str).str.zfill(8))
    d_firms = d_firms[(d_firms["Source"] == "IFDATA") | (d_firms["cnpj_8"].isin(dig_cands))]

    path_ip = list(_ROOT.glob("**/ip_rates_quarterly.csv"))[0]
    df_ip = pd.read_csv(path_ip, low_memory=False)
    cnpjs_with_ip_rates = set(df_ip["CNPJ"].astype(str).str.zfill(8).unique())

    d_summ = d_firms.groupby(["CodConglomeradoPrudencial", "NomeInstituicao"]).agg(
        total_assets=("total_assets", "max"),
        first_year=("year", "min"),
        first_quarter=("quarter", "min"),
        obs=("year", "count")
    ).reset_index()

    d_summ = d_summ.sort_values(["total_assets", "first_year", "first_quarter"], ascending=[False, True, True])

    for idx, row in d_summ.iterrows():
        cp = row["CodConglomeradoPrudencial"]
        f_rows = d_firms[d_firms["CodConglomeradoPrudencial"] == cp]
        matches = f_rows["cnpj_8"].isin(cnpjs_with_ip_rates).any()
        d_summ.at[idx, "native_k5_rate"] = "Yes (IP Rates)" if matches else "No (COSIF Fallback)"

    d_summ["Asset_Size"] = d_summ["total_assets"].apply(lambda x: f"{x/1e9:.2f} B" if pd.notna(x) else "Unknown")
    
    out_csv = OUTPUT_DIR / "d_type_firms_summary_final.csv"
    d_summ.to_csv(out_csv, index=False)
    print(f"Saved D-Type summary to: {out_csv}")

    print(f"Descriptive statistics successfully saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()