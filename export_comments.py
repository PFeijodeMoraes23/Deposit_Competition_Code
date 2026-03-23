"""
export_comments.py
==================
Exports the estimation results to a standalone LaTeX document, including
both First Stage and Second Stage results, ready to be compiled natively in TeXStudio.

Redundant markdown generation has been strictly removed here so it's clean and native.
"""

import os
import sys
import json
import pickle
from pathlib import Path

# Paths
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "SLEEPINESS_NEW"
RESULTS_PICKLE = OUTPUT_DIR / "estimation_results.pkl"
CLUSTER_JSON = OUTPUT_DIR / "cluster_diagnostics.json"

OUT_DIR = (r"C:\Users\pedro\OneDrive\Documentos\Yale"
           r"\Year 3 (2024 - 2025)\Open Finance\Open-Finance"
           r"\Drafts\Deposit Competition")


def build_reg_table(res, spec_name, G_nominal, G_star, is_first_stage=False):
    params    = res.params
    bse       = res.bse
    tvalues   = res.tvalues
    pvalues   = res.pvalues
    nobs      = int(res.nobs)
    rsq       = res.rsquared

    def stars(p):
        if p < 0.01: return '***'
        elif p < 0.05: return '**'
        elif p < 0.10: return '*'
        return ''

    def pretty(name):
        name = name.replace('_', '\\_').replace('interaction\\_', '')
        return name

    rows = []
    for var in params.index:
        coef = params[var]
        se   = bse[var]
        p    = pvalues[var]
        s    = stars(p)
        rows.append(f"    {pretty(var)} & ${coef:.4f}^{{{s}}}$ & $({se:.4f})$ & ${p:.3f}$ \\\\")

    body = '\n'.join(rows)
    
    fs_indicator = " (First Stage)" if is_first_stage else " (Second Stage)"
    title = f"{spec_name}{fs_indicator}"
    safe_name = title.replace(' ', '_').replace('/', '').replace(':', '')

    table = (
        f"\\begin{{table}}[htbp]\n"
        f"\\centering\n"
        f"\\caption{{{title}}}\n"
        f"\\label{{tab:{safe_name}}}\n"
        f"\\begin{{tabular}}{{lccc}}\n"
        f"\\toprule\n"
        f"Variable & Coefficient & (Std.\\ Error) & $p$-value \\\\\n"
        f"\\midrule\n"
        f"{body}\n"
        f"\\midrule\n"
        f"\\multicolumn{{4}}{{l}}{{\\footnotesize Obs: {nobs:,} "
        f"\\quad $G = {G_nominal}$ "
        f"\\quad $G^* = {G_star:.2f}$ "
        f"\\quad $R^2 = {rsq:.4f}$}} \\\\\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )
    return table

def main():
    if not RESULTS_PICKLE.exists() or not CLUSTER_JSON.exists():
        print("Results not found. Run estimation_1_sleep.py first.")
        return

    print("Reading pickled model estimates...")
    with open(RESULTS_PICKLE, 'rb') as f:
        results_dict = pickle.load(f)
        
    with open(CLUSTER_JSON, 'r') as f:
        cluster_data = json.load(f)

    G_nominal = cluster_data['G_nominal']
    G_star = cluster_data['G_star']

    print("Generating latex table code...")
    tables_tex_fs = ""
    tables_tex_ss = ""
    for spec_name, res_dict in results_dict.items():
        if isinstance(res_dict, dict):
            # new dict format
            res_ss = res_dict.get('second_stage')
            res_fs = res_dict.get('first_stage')

            if res_fs is not None:
                tables_tex_fs += build_reg_table(res_fs, spec_name, G_nominal, G_star, True) + "\n\n"
            if res_ss is not None:
                tables_tex_ss += build_reg_table(res_ss, spec_name, G_nominal, G_star, False) + "\n\n"
        else:
            # fallback for old object directly containing second stage res
            tables_tex_ss += build_reg_table(res_dict, spec_name, G_nominal, G_star, False) + "\n\n"

    # Pre-amble
    preamble = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{letterpaper, margin=1in}
\usepackage{caption}
\usepackage{longtable}
\usepackage{float}
\usepackage{hyperref}
\begin{document}

\section*{Estimation Summary --- Deposit Competition}
This document aggregates all auto-generated regressions including both First Stage (Control Function) and Second Stage (Sleepiness) estimates.

"""
    
    tex_doc = (
        preamble
        + "\\subsection*{First Stage Regression Results}\n"
        + tables_tex_fs
        + "\\newpage\n\\subsection*{Second Stage Regression Results}\n"
        + tables_tex_ss
        + "\\end{document}\n"
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    temp_tex = os.path.join(OUT_DIR, "Estimation_Summary_and_SE.tex")
    print(f"Writing single LaTeX file to {temp_tex} ...")
    with open(temp_tex, 'w', encoding='utf-8') as f:
        f.write(tex_doc)
    
    print("Done. Native TeX file saved correctly and is ready for compilation in TeXStudio.")

if __name__ == '__main__':
    main()
