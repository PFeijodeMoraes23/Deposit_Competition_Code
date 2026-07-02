
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import os
import sys
import json
import pickle
from pathlib import Path
import subprocess
import shutil
import warnings
warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")

_DRAFTS_DIR = Path(r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Drafts\Deposit Competition")

# ---------------------------------------------------------------------------
# Allow import of sibling module estimation_1_sleep / utils
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Paths
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "est1"
RESULTS_PICKLE = OUTPUT_DIR / "estimation_results.pkl"

TEX_OUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"
OUT_DIR = str(TEX_OUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEX_OUT_DIR.mkdir(parents=True, exist_ok=True)

def stars(p):
    if p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    return ''

def clean_name(v):
    v = str(v).replace('interaction_', '')
    labels = {
        'nr_lagged_dep': 'Lagged Deposits',
        'gdp_per_capita': 'GDP \\textit{per capita} (10k R\\$)',
        'cadunico_families_per1000': 'CadUnico Families (100s per 1k)',
        'fraction_65plus': 'Fraction 65+',
        'fraction_young': 'Fraction Young',
        'risk_free_qoq_lag': 'Lagged Selic Rate',
        'pix_users_pf_per1000': 'Pix Users (100s per 1k)',
        'connections_per100': 'Broadband Connections (per 100 inhabitants)',
        'branches_per1000': 'Branches per 1k',
        'v_hat_x_lagged_dep': 'CF: $\\hat{v} \\times$ Lagged Deposits',
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
    }
    return labels.get(v, v.replace('_', '\\_'))

def build_first_stage_table(results_dict):
    panels = ['Base', 'Macro', 'Tech']
    panel_labels = {
        'Base': 'Base Specifications',
        'Macro': 'Macro Specifications',
        'Tech': 'Tech Specifications',
    }
    panel_letters = ['A', 'B', 'C']
    fs_spec_numbers = {
        ('Base', 'IV_CostShifters'): 2,
        ('Base', 'IV_Wholesale'): 3,
        ('Base', 'IV_HausmanFull'): 4,
        ('Macro', 'IV_CostShifters'): 6,
        ('Macro', 'IV_Wholesale'): 7,
        ('Macro', 'IV_HausmanFull'): 8,
        ('Tech', 'IV_CostShifters'): 10,
        ('Tech', 'IV_Wholesale'): 11,
        ('Tech', 'IV_HausmanFull'): 12,
    }
    ivs = [
        ('IV_CostShifters', 'IV Cost'),
        ('IV_Wholesale', 'IV Wholesale'),
        ('IV_HausmanFull', 'Hausman'),
    ]
    multispan = 4
    caption = "Local Data Only - First Stage Estimation"
    label = "tab:est1_first_stage"
    notes = (
        r"\footnotesize \textit{Notes:} Standard errors (wild cluster bootstrap at the "
        r"conglomerate level; \textcite{cameron2008bootstrap}, \textcite{mackinnon2017wild}) "
        r"in parentheses. Significance levels: *** $p<0.01$, ** $p<0.05$, * $p<0.1$."
    )

    def _get_res(iv_key, p):
        entry = results_dict.get(f"{iv_key} x {p}")
        return entry.get('first_stage') if isinstance(entry, dict) else None

    p0, l0 = panels[0], panel_letters[0]
    iv_nums_0 = [(il, fs_spec_numbers[(p0, ik)]) for ik, il in ivs]

    lines = [
        r"\setstretch{1.0}",
        r"\begin{xltabular}{\textwidth}{>{\raggedright\arraybackslash}p{0.34\textwidth} *{3}{>{\centering\arraybackslash}X}}",
        rf"    \caption{{{caption}}}\label{{{label}}} \\",
        r"    \toprule",
        rf"    \multicolumn{{{multispan}}}{{l}}{{\textbf{{Panel {l0}: {panel_labels[p0]}}}}} \\",
        r"    \midrule",
        "    & " + " & ".join(il for il, _ in iv_nums_0) + r" \\",
        "    & " + " & ".join(f"({n})" for _, n in iv_nums_0) + r" \\",
        r"    \midrule",
        r"    \endfirsthead",
        "",
        rf"    \multicolumn{{{multispan}}}{{c}}{{{{\bfseries Table \thetable\ continued from previous page}}}} \\",
        r"    \toprule",
        "    & " + " & ".join(il for il, _ in iv_nums_0) + r" \\",
        r"    \midrule",
        r"    \endhead",
        "",
        r"    \midrule",
        rf"    \multicolumn{{{multispan}}}{{r}}{{\textit{{Continued on next page}}}} \\",
        r"    \endfoot",
        "",
        r"    \bottomrule",
        rf"    \multicolumn{{{multispan}}}{{p{{\dimexpr\textwidth-2\tabcolsep\relax}}}}{{{notes}}} \\",
        r"    \endlastfoot",
        "",
    ]

    for pi, panel in enumerate(panels):
        letter = panel_letters[pi]
        plabel = panel_labels[panel]
        iv_nums = [(il, fs_spec_numbers[(panel, ik)]) for ik, il in ivs]

        if pi > 0:
            lines += [
                r"    \addlinespace[1.5em]",
                "",
                r"    \toprule",
                rf"    \multicolumn{{{multispan}}}{{l}}{{\textbf{{Panel {letter}: {plabel}}}}} \\",
                r"    \midrule",
                "    & " + " & ".join(il for il, _ in iv_nums) + r" \\",
                "    & " + " & ".join(f"({n})" for _, n in iv_nums) + r" \\",
                r"    \midrule",
            ]

        all_vars_fs = [
            'tax_cost_ratio_lag',
            'personnel_cost_ratio_lag',
            'admin_cost_ratio_lag',
            'indice_basileia_lag',
            'lci_lca_ratio_lag',
            'wholesale_ratio_lag',
            'leave_one_out_mean_spread',
        ]
        # include any unexpected variables not in the fixed list
        vs_panel = all_vars_fs

        for var in vs_panel:
            coef_strs, se_strs, has_val = [], [], False
            for ik, _ in ivs:
                res = _get_res(ik, panel)
                if res is not None and var in res.params:
                    has_val = True
                    c, se, pval = res.params[var], res.bse[var], res.pvalues[var]
                    digits = 4
                    coef_strs.append(f"${c:.{digits}f}^{{{stars(pval)}}}$")
                    se_strs.append(f"$({se:.{digits}f})$")
                else:
                    coef_strs.append("")
                    se_strs.append("")
            if has_val:
                lines.append(f"    {clean_name(var)} & " + " & ".join(coef_strs) + r" \\*")
                lines.append("    & " + " & ".join(se_strs) + r" \\")

        obs_l, rsq_l, fstat_l, g_l = [], [], [], []
        for ik, _ in ivs:
            res = _get_res(ik, panel)
            if res is None:
                obs_l.append("---"); rsq_l.append("---"); fstat_l.append("---")
                g_l.append("---")
                continue
            obs_l.append(f"{int(res.nobs):,}")
            rsq_l.append(f"{res.rsquared:.4f}")
            fv = getattr(res, 'fvalue', None)
            fp = getattr(res, 'f_pvalue', 1.0)
            fstat_l.append(f"${fv:.2f}^{{{stars(fp)}}}$" if fv is not None else "---")
            g_l.append(str(getattr(res, 'G_nominal', '---')))

        lines += [
            r"    \midrule",
            "    Observations & " + " & ".join(obs_l) + r" \\",
            "    $R^2$ & " + " & ".join(rsq_l) + r" \\",
            "    F-Statistic & " + " & ".join(fstat_l) + r" \\",
            "    Fixed Effects & No & No & No \\\\",
            "    Clusters ($G$) & " + " & ".join(g_l) + r" \\",
            r"    \bottomrule",
        ]

    lines += [r"\end{xltabular}", r"\doublespacing"]
    return "\n".join(lines)


def build_second_stage_table(results_dict):
    panels = ['Base', 'Macro', 'Tech']
    panel_labels = {
        'Base': 'Base Specifications',
        'Macro': 'Macro Specifications',
        'Tech': 'Tech Specifications',
    }
    panel_letters = ['A', 'B', 'C']
    ss_spec_numbers = {
        ('Base', 'OLS'): 1,
        ('Base', 'IV_CostShifters'): 2,
        ('Base', 'IV_Wholesale'): 3,
        ('Base', 'IV_HausmanFull'): 4,
        ('Macro', 'OLS'): 5,
        ('Macro', 'IV_CostShifters'): 6,
        ('Macro', 'IV_Wholesale'): 7,
        ('Macro', 'IV_HausmanFull'): 8,
        ('Tech', 'OLS'): 9,
        ('Tech', 'IV_CostShifters'): 10,
        ('Tech', 'IV_Wholesale'): 11,
        ('Tech', 'IV_HausmanFull'): 12,
    }
    estimators = [
        ('OLS', 'OLS'),
        ('IV_CostShifters', 'IV Cost'),
        ('IV_Wholesale', 'IV Wholesale'),
        ('IV_HausmanFull', 'Hausman'),
    ]
    all_vars = [
        'const', 'pix_exists', 'gdp_per_capita',
        'cadunico_families_per1000', 'fraction_65plus', 'fraction_young',
        'risk_free_qoq_lag', 'connections_per100',
    ]
    multispan = 5
    caption = "Local Data Only - Second Stage Estimation"
    label = "tab:est1_second_stage"
    notes = (
        r"\footnotesize \textit{Notes:} Standard errors (wild cluster bootstrap at the "
        r"conglomerate level; \textcite{cameron2008bootstrap}, \textcite{mackinnon2017wild}) "
        r"in parentheses. Significance levels: *** $p<0.01$, ** $p<0.05$, * $p<0.1$."
    )

    def _get_res(ek, p):
        entry = results_dict.get(f"{ek} x {p}")
        return entry.get('second_stage') if isinstance(entry, dict) else None

    p0, l0 = panels[0], panel_letters[0]
    est_nums_0 = [(el, ss_spec_numbers[(p0, ek)]) for ek, el in estimators]

    lines = [
        r"\setstretch{1.0}",
        r"\begin{xltabular}{\textwidth}{>{\raggedright\arraybackslash}p{0.28\textwidth} *{4}{>{\centering\arraybackslash}X}}",
        rf"    \caption{{{caption}}}\label{{{label}}} \\",
        r"    \toprule",
        rf"    \multicolumn{{{multispan}}}{{l}}{{\textbf{{Panel {l0}: {panel_labels[p0]}}}}} \\",
        r"    \midrule",
        "     & " + " & ".join(el for el, _ in est_nums_0) + r" \\",
        "    & " + " & ".join(f"({n})" for _, n in est_nums_0) + r" \\",
        r"    \midrule",
        r"    \endfirsthead",
        "",
        rf"    \multicolumn{{{multispan}}}{{c}}{{{{\bfseries Table \thetable\ continued from previous page}}}} \\",
        r"    \toprule",
        "     & " + " & ".join(el for el, _ in est_nums_0) + r" \\",
        r"    \midrule",
        r"    \endhead",
        "",
        r"    \midrule",
        rf"    \multicolumn{{{multispan}}}{{r}}{{\textit{{Continued on next page}}}} \\",
        r"    \endfoot",
        "",
        r"    \bottomrule",
        rf"    \multicolumn{{{multispan}}}{{p{{\dimexpr\textwidth-2\tabcolsep\relax}}}}{{{notes}}} \\",
        r"    \endlastfoot",
        "",
    ]

    for pi, panel in enumerate(panels):
        letter = panel_letters[pi]
        plabel = panel_labels[panel]
        est_nums = [(el, ss_spec_numbers[(panel, ek)]) for ek, el in estimators]

        if pi > 0:
            lines += [
                r"    \addlinespace[1.5em]",
                "",
                r"    \toprule",
                rf"    \multicolumn{{{multispan}}}{{l}}{{\textbf{{Panel {letter}: {plabel}}}}} \\",
                r"    \midrule",
                "     & " + " & ".join(el for el, _ in est_nums) + r" \\",
                "    & " + " & ".join(f"({n})" for _, n in est_nums) + r" \\",
                r"    \midrule",
            ]

        for vshort in all_vars:
            coef_strs, se_strs, has_val = [], [], False
            for ek, _ in estimators:
                res = _get_res(ek, panel)
                var = vshort
                if vshort in ('const', 'constant') and res is not None and 'nr_lagged_dep' in res.params:
                    var = 'nr_lagged_dep'
                elif res is not None and var not in res.params and f"interaction_{var}" in res.params:
                    var = f"interaction_{var}"
                if res is not None and var in res.params:
                    has_val = True
                    c, se, pval = res.params[var], res.bse[var], res.pvalues[var]
                    coef_strs.append(f"${c:.4f}^{{{stars(pval)}}}$")
                    se_strs.append(f"$({se:.4f})$")
                else:
                    coef_strs.append("")
                    se_strs.append("")
            if has_val:
                lines.append(f"    {clean_name(vshort)} & " + " & ".join(coef_strs) + r" \\")
                lines.append("    & " + " & ".join(se_strs) + r" \\")

        obs_l, rsq_l, g_l = [], [], []
        for ek, _ in estimators:
            res = _get_res(ek, panel)
            if res is None:
                obs_l.append("---"); rsq_l.append("---")
                g_l.append("---")
                continue
            nv = getattr(res, 'nobs', None)
            obs_l.append(f"{int(nv):,}" if nv is not None else "---")
            rv = getattr(res, 'rsquared', None)
            rsq_l.append(f"{rv:.4f}" if rv is not None else "---")
            g_l.append(str(getattr(res, 'G_nominal', '---')))

        lines += [
            r"    \midrule",
            "    Observations & " + " & ".join(obs_l) + r" \\",
            "    $R^2$ & " + " & ".join(rsq_l) + r" \\",
            "    Fixed Effects & Yes & Yes & Yes & Yes \\\\",
            "    Clusters ($G$) & " + " & ".join(g_l) + r" \\",
            r"    \bottomrule",
        ]

    lines += [r"\end{xltabular}", r"\doublespacing"]
    return "\n".join(lines)

_STANDALONE_PREAMBLE = r"""\documentclass[12pt]{article}
\usepackage[letterpaper, margin=1in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[english]{babel}
\usepackage{amssymb, mathrsfs, amsthm, mathtools}
\usepackage{graphicx, float}
\usepackage{setspace}
\usepackage{multirow}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage[font=small,labelfont=bf]{caption}
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.08}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue}
"""


def main():
    print("=" * 70)
    print(" EXPORT EST1 SLEEP RESULTS")
    print("=" * 70)

    if not RESULTS_PICKLE.exists():
        print("Results not found. Run estimation_1_sleep.py first.")
        return

    print(" - Reading pickled model estimates...")
    with open(RESULTS_PICKLE, 'rb') as fh:
        results_dict = pickle.load(fh)

    print(" - Building table fragments...")
    fs_frag = build_first_stage_table(results_dict)
    ss_frag = build_second_stage_table(results_dict)

    os.makedirs(OUT_DIR, exist_ok=True)

    fs_path = os.path.join(OUT_DIR, "est1_first_stage_table.tex")
    ss_path = os.path.join(OUT_DIR, "est1_second_stage_table.tex")
    with open(fs_path, 'w', encoding='utf-8') as fh:
        fh.write(fs_frag + "\n")
    with open(ss_path, 'w', encoding='utf-8') as fh:
        fh.write(ss_frag + "\n")
    shutil.copy(fs_path, _DRAFTS_DIR / "est1_first_stage_table.tex")
    shutil.copy(ss_path, _DRAFTS_DIR / "est1_second_stage_table.tex")
    print(f" - Fragments written and copied to {_DRAFTS_DIR}")

    tex_doc = (
        _STANDALONE_PREAMBLE
        + r"\begin{document}" + "\n"
        + r"\title{Sleep Estimation Results --- Est 1 (B-Type Firms)}" + "\n"
        + r"\author{Autogenerated}" + "\n"
        + r"\date{\today}" + "\n"
        + r"\maketitle" + "\n\n"
        + r"\section*{First Stage}" + "\n"
        + r"\input{est1_first_stage_table.tex}" + "\n\n"
        + r"\section*{Second Stage}" + "\n"
        + r"\input{est1_second_stage_table.tex}" + "\n"
        + r"\end{document}" + "\n"
    )

    wrapper_path = os.path.join(OUT_DIR, "est1_sleep_results.tex")
    with open(wrapper_path, 'w', encoding='utf-8') as fh:
        fh.write(tex_doc)
    shutil.copy(wrapper_path, _DRAFTS_DIR / "est1_sleep_results.tex")
    print(f" - Standalone wrapper written and copied to {_DRAFTS_DIR}")

    print(" - Compiling PDF...")
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "est1_sleep_results.tex"],
                       cwd=OUT_DIR, capture_output=True, text=True)
        res_final = subprocess.run(["pdflatex", "-interaction=nonstopmode", "est1_sleep_results.tex"],
                                   cwd=OUT_DIR, capture_output=True, text=True)
        pdf_path = os.path.join(OUT_DIR, "est1_sleep_results.pdf")
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            print("\n *** PDF SUCCESSFULLY GENERATED. ***\n")
        else:
            print(f"\n *** PDF GENERATION FAILED. Log tail:\n{res_final.stdout[-800:]}\n ***\n")
    except Exception as e:
        print(f"\n *** COMPILATION ERROR: {e} ***\n")

    print("Done.")


if __name__ == '__main__':
    main()
