"""
Automated LaTeX Table generation for BLP Demand Estimation outputs.
Compiles outputs from all estimation modules and prints significance-starred
results inside tables formatted to the user's custom LaTeX preamble.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import pathlib
import json
import argparse
import numpy as np
import scipy.stats as stats
import re

ROOT = pathlib.Path(__file__).resolve().parent
# Match get_paths() logic from estimation loop scripts (locating the BCB data folder)
DATA_DIR = ROOT.parents[1] / "BCB" / "Egan_et_al_2025_Rep" / "processed"
RESULTS_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "BLP_RESULTS"
TABLES_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ── Variables Mapping ────────────────────────────────────────────────────────
VAR_MAP = {
    # Alpha params
    'alpha_1': r'$\alpha_{1}$',
    'alpha_2': r'$\alpha_{2}$',
    'alpha_4': r'$\alpha_{4}$',
    'alpha_5': r'$\alpha_{5}$',
    
    # Linear characteristics (X_COLS)
    'fgc_covered': 'FGC Covered',
    'has_ip': 'Has IP',
    'seg_S2': 'Segment S2',
    'seg_S3': 'Segment S3',
    'seg_S4': 'Segment S4',
    'seg_S5': 'Segment S5',
    'log_total_assets_lag': r'$\ln(\text{Total Assets}_{t-1})$',
    'equity_ratio_lag': r'$\text{Equity Ratio}_{t-1}$',
    
    # Non-linear characteristics (Sigma)
    # The code maps sigma variations to these parameters. 
    # Example format from pi_interactions/sigma_indices
    # Will be formatted dynamically in code as $\Sigma_{\text{param}}$
    
    # Demographics (Pi mapping)
    'gdp_per_capita': r'GDP \textit{per capita}',
    'fraction_65plus': r'Fraction 65+',
    'fraction_young': r'Fraction young',
    'pix_users_pf_per1000': r'Pix Users (per 1,000)',
    'connections_per100': r'Internet (per 100)',
    'frac_4g5g': r'Fraction 4G/5G',
    'branches_per1000': r'Branches (per 1,000)',
    'cadunico_families_per1000': r'CadUnico (per 1,000)',
}

def format_latex_val(coef, se, G_star=None):
    """Returns a tuple of two strings: (coef_string, se_string) with significance stars.
    Uses t(df=G_star) per Imbens & Koles\u00e1r (2016) if G_star is provided."""
    if np.isnan(coef) or np.isnan(se):
        return ("-", "")
    
    t_stat = coef / se
    if G_star is not None and G_star > 1:
        p_val = 2 * stats.t.sf(abs(t_stat), df=G_star)
    else:
        p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    stars = ""
    if p_val < 0.01:
        stars = "^{***}"
    elif p_val < 0.05:
        stars = "^{**}"
    elif p_val < 0.10:
        stars = "^{*}"
        
    coef_str = f"${coef:.4f}{stars}$"
    se_str = f"({se:.4f})"
    return (coef_str, se_str)

def map_var(name):
    # If the user map exists, use it
    if name in VAR_MAP:
        return VAR_MAP[name]
        
    # Attempt to catch dynamic names like 'sigma_has_ip'
    if name.startswith('sigma_'):
        inner = name.replace('sigma_', '')
        inner_mapped = map_var(inner)
        return r"$\Sigma_{" + inner_mapped.strip('$') + r"}$"
        
    if name.startswith('pi_'):
        # E.g. pi_has_ip_x_gdp_per_capita
        parts = name.replace('pi_', '').split('_x_')
        if len(parts) == 2:
            left = map_var(parts[0]).strip('$')
            right = map_var(parts[1]).strip('$')
            return r"$\Pi_{\text{" + left + r"}, \text{" + right + r"}}$"
            
    # Default to escaped clean text
    return name.replace('_', r'\_')

PREAMBLE = r"""\documentclass[12pt]{article}
\usepackage[letterpaper, margin=1in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[english]{babel}
\usepackage[dvipsnames]{xcolor}

% Math and fonts
\usepackage{amssymb, mathrsfs}
\usepackage{amsthm}
\usepackage{mathtools}
\usepackage{dsfont}
\usepackage[normalem]{ulem}
\usepackage{relsize}
\usepackage{array}

% Graphics and tables
\usepackage{graphicx}  
\usepackage{float}
\usepackage{xltabular}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\usepackage{xurl}
\usepackage{subcaption}

% Spacing and formatting
\setlength{\parindent}{1cm}
\usepackage{setspace}
\usepackage{indentfirst}
\usepackage{titlesec}
\usepackage{multirow}

% Lists
\usepackage{enumitem}

% Bibliography
\usepackage[backend=biber, style=authoryear, maxcitenames=2, sortcites=true, sorting=ynt]{biblatex}
\DeclareDelimFormat[bib,textcite,parencite]{finalnamedelim}{\addspace\&\addspace}
\DeclareDelimFormat[bib,parencite]{nameyeardelim}{\addcomma\space}
\DeclareDelimFormat[textcite]{nameyeardelim}{\addspace}
\addbibresource{../References.bib}

\usepackage{pifont}
\newcommand{\cmark}{\ding{51}}
\newcommand{\cmarkbold}{\ding{52}}

% TikZ and PGFPlots
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepgfplotslibrary{fillbetween}
\usepackage{footmisc}
\usepackage[font=small,labelfont=bf]{caption}

% Custom commands
\newcommand{\linearmap}[3]{#1:#2\to#3}
\newcommand{\myarrow}[0]{\,\to\,}
\newcommand{\mycomma}[0]{\,,\,}
\newcommand{\myspace}[0]{\,\quad\,}
\newcommand{\ida}[0]{\left(\Rightarrow\right)}
\newcommand{\volta}[0]{\left(\Leftarrow\right)}
\newcommand{\myiff}[0]{\,\longleftrightarrow\,}
\newcommand{\tabspace}[0]{\myspace\myspace\myspace}
\newcommand{\textbfit}[1]{\text{\textbf{\textit{#1}}}}
\newcommand{\openball}[3][]{\mathrm{B}_{#1}\left(#2;#3\right)}

% Calculus
\newcommand{\derivative}[2]{\frac{\mathrm{d} #1}{\mathrm{d} #2}}
\newcommand{\partialderivative}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\mylog}[1]{\ln{\left( #1 \right)}}
\newcommand{\mydet}[1]{\mathrm{det}{\left(#1\right)}}
\newcommand{\dirderi}[2]{\mathrm{D}_{\bold{#1}}{#2}}
\newcommand{\minimize}[2][ ]{\underset{#1}{\mathrm{min}} \myspace{#2}}
\newcommand{\maximize}[2][ ]{\underset{#1}{\mathrm{max}} \myspace{#2}}
\newcommand{\subjectto}[1]{\mathrm{s.t.}\myspace{#1}}

% Proper argmax/argmin definitions
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}

% Sets
\newcommand{\nnatural}[0]{\mathbb{N}}
\newcommand{\real}[0]{\mathbb{R}}
\newcommand{\zhail}[0]{\mathbb{Z}}
\newcommand{\rational}[0]{\mathbb{Q}}
\newcommand{\powerset}[1]{\mathbb{P}\left(#1\right)}

% Probability and statistics
\newcommand{\expectedvalue}[2][]{\mathbb{E}_{#1}\left[#2\right]}
\newcommand{\condexp}[3][]{\mathbb{E}_{#1}\left[#2\middle|#3\right]}
\newcommand{\indicator}[1]{\boldsymbol{1}\left\{#1\right\}}
\newcommand{\condprob}[2]{P\left(#1 \mid #2 \right)}
\newcommand{\plimarrow}[0]{\overset{p}{\longrightarrow}}
\newcommand{\convdistr}[0]{\overset{d}{\longrightarrow}}
\newcommand{\indep}[0]{\mathrel{\perp\!\!\!\perp}}

% Theorems and definitions
\theoremstyle{plain}
\newtheorem{theo}{Theorem}[subsection]
\newtheorem{prop}{Proposition}[subsection]
\newtheorem{deff}{Definition}[subsection]
\newtheorem{lemma}{Lemma}[subsection]

% Custom environments (with and without *)
\newtheorem*{theo*}{Theorem}
\newtheorem*{prop*}{Proposition}
\newtheorem*{deff*}{Definition}
\newtheorem*{lemma*}{Lemma}
\newcommand{\PFMcomment}[1]{\textcolor{MidnightBlue}{#1}}

% Allow display breaks
\allowdisplaybreaks

% Hyperlinks (loaded last)
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=RoyalBlue, urlcolor=RoyalBlue, citecolor=RoyalBlue}
\urlstyle{tt}

\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.08}
\setlength{\emergencystretch}{2em}
\AtBeginEnvironment{xltabular}{\footnotesize}
\AtBeginEnvironment{tabularx}{\footnotesize}
\AtBeginEnvironment{longtable}{\footnotesize}

% Table packages:
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{pdflscape}
\usepackage{microtype}
\usepackage{csquotes}
\usepackage{etoolbox}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{listings}
"""

def process_estimation(est_id: int):
    print(f"\n--- Processing Estimation {est_id} ---")
    
    alts = [None]
        
    tex_content = PREAMBLE + "\n\\begin{document}\n"
    tex_content += r"\section*{Estimation " + str(est_id) + " Results}" + "\n\n"
    
    stages_priority = ['extended', 'full', 'sigma', 'logit']
    
    for alt in alts:
        specs_data = {}
        all_params = []
        
        # Collect results for spec 1 to 12
        for sp in range(1, 13):
            res = None
            for stage in stages_priority:
                filename = f"blp_results_E{est_id}_spec_{sp}_{stage}.json"
                path = RESULTS_DIR / filename
                if path.exists():
                    try:
                        with open(path, 'r') as f:
                            res = json.load(f)
                            # Tag the stage found
                            res['__stage__'] = stage
                        break # Found highest priority
                    except Exception as e:
                        pass
            
            if res is not None:
                specs_data[sp] = res
                
        if not specs_data:
            print(f"No results found for Est {est_id}")
            continue
            
        print(f"Loaded {len(specs_data)} specs for Est {est_id}")
        
        # Build unified parameter list from all found specs
        # Keep consistent ordering
        theta1_names_master = []
        theta2_names_master = []
        
        for sp, r in specs_data.items():
            for p in r.get('param_names_theta1', []):
                if p not in theta1_names_master:
                    theta1_names_master.append(p)
                    
            # sigma and pi
            if 'sigma_indices' in r and r['sigma_indices']:
                for p_idx in r['sigma_indices']:
                    # if p_idx is an index, we need its name. 
                    # Normally it maps to param_names_theta1 index.
                    try:
                        name = f"sigma_{r['param_names_theta1'][p_idx]}"
                        if name not in theta2_names_master:
                            theta2_names_master.append(name)
                    except: pass
                    
            if 'pi_interactions' in r and r['pi_interactions']:
                for k_idx, d_idx in r['pi_interactions']:
                    # Assuming K_LIST mapping and D_COLS mapping
                    # To be robust, just generically name if possible
                    # Or we can look into df column names if needed.
                    # We will just label pi_{k_idx}_{d_idx} for safety
                    name = f"pi_{k_idx}_{d_idx}"
                    if name not in theta2_names_master:
                        theta2_names_master.append(name)
                        
        # Render Table
        title = f"Estimation {est_id}"
        ncols = 12
        col_format = "l" + "c" * ncols
        
        table_code = "\\begin{table}[H]\n\\centering\n\\begin{threeparttable}\n"
        table_code += "\\caption{" + title + "}\n"
        table_code += f"\\begin{{tabular}}{{{col_format}}}\n\\toprule\n"
        
        # Header row
        table_code += "Parameter & " + " & ".join([f"({sp})" for sp in range(1, 13)]) + " \\\\\n\\midrule\n"
        
        # Linear parameters
        table_code += "\\multicolumn{13}{c}{\\textit{Linear Parameters (Mean Utilities)}} \\\\\n\\midrule\n"
        for p in theta1_names_master:
            p_mapped = map_var(p)
            row_c = [p_mapped]
            row_s = [""]
            for sp in range(1, 13):
                if sp in specs_data and p in specs_data[sp].get('param_names_theta1', []):
                    idx = specs_data[sp]['param_names_theta1'].index(p)
                    coef = specs_data[sp]['theta1'][idx]
                    se = specs_data[sp]['theta1_se'][idx] if 'theta1_se' in specs_data[sp] else np.nan
                    c_str, s_str = format_latex_val(coef, se, G_star=specs_data[sp].get('G_star'))
                    row_c.append(c_str)
                    row_s.append(s_str)
                else:
                    row_c.append("-")
                    row_s.append("")
            
            table_code += " & ".join(row_c) + " \\\\\n"
            # Add SE row
            table_code += " & " + " & ".join(row_s[1:]) + " \\\\\n"
            table_code += "\\addlinespace\n"
            
        # Non-linear parameters (if any)
        if theta2_names_master:
            table_code += "\\midrule\n\\multicolumn{13}{c}{\\textit{Non-Linear Parameters}} \\\\\n\\midrule\n"
            for p in theta2_names_master:
                p_mapped = map_var(p)
                row_c = [p_mapped]
                row_s = [""]
                for sp in range(1, 13):
                    # We need to map position in theta2 array
                    r = specs_data.get(sp)
                    coef, se = np.nan, np.nan
                    if r and len(r.get('theta2', [])) > 0:
                        # Reconstruct the theta2 structure to find index
                        local_t2_names = []
                        if 'sigma_indices' in r and r['sigma_indices']:
                            for idx_ in r['sigma_indices']:
                                local_t2_names.append(f"sigma_{r['param_names_theta1'][idx_]}")
                        if 'pi_interactions' in r and r['pi_interactions']:
                            for k_i, d_i in r['pi_interactions']:
                                local_t2_names.append(f"pi_{k_i}_{d_i}")
                                
                        if p in local_t2_names:
                            idx = local_t2_names.index(p)
                            coef = r['theta2'][idx]
                            se = r['theta2_se'][idx] if 'theta2_se' in r and len(r['theta2_se']) > idx else np.nan
                            
                    c_str, s_str = format_latex_val(coef, se, G_star=specs_data.get(sp, {}).get('G_star'))
                    row_c.append(c_str)
                    row_s.append(s_str)
                
                table_code += " & ".join(row_c) + " \\\\\n"
                table_code += " & " + " & ".join(row_s[1:]) + " \\\\\n"
                table_code += "\\addlinespace\n"
                
        # Footer fits
        table_code += "\\midrule\n"
        row_stage = ["Optimization Stage"]
        row_q = ["Objective $Q$-Value"]
        row_conv = ["Converged"]
        
        for sp in range(1, 13):
            if sp in specs_data:
                row_stage.append(specs_data[sp]['__stage__'])
                row_q.append(f"{specs_data[sp]['Q_value']:.2f}")
                row_conv.append("Yes" if specs_data[sp].get('converged', False) else "No")
            else:
                row_stage.append("-")
                row_q.append("-")
                row_conv.append("-")
                
        table_code += " & ".join(row_stage) + " \\\\\n"
        table_code += " & ".join(row_q) + " \\\\\n"
        table_code += " & ".join(row_conv) + " \\\\\n"

        table_code += "\\bottomrule\n\\end{tabular}\n"
        table_code += """\\begin{tablenotes}
\\item Standard errors in parentheses (cluster-robust, Imbens \\& Koles{\'a}r 2016). Significance levels: * $p < 0.1$, ** $p < 0.05$, *** $p < 0.01$.
\\end{tablenotes}\n"""
        table_code += "\\end{threeparttable}\n\\end{table}\n\n"
        
        tex_content += table_code
        
    tex_content += "\\end{document}\n"
    
    out_file = TABLES_DIR / f"estimation_{est_id}_results.tex"
    out_file.write_text(tex_content, encoding='utf-8')
    print(f"==> Wrote {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create BLP LaTeX Tables")
    parser.add_argument('--est', type=int, choices=[1, 2, 3, 4, 5], help='Estimation routine ID to process')
    args = parser.parse_args()
    
    if args.est:
        process_estimation(args.est)
    else:
        # Fallback to sequential if no arg given
        for i in [1, 2, 3, 4, 5]:
            process_estimation(i)
