import os
import sys
import json
import pickle
from pathlib import Path
import subprocess

# ---------------------------------------------------------------------------
# Allow import of sibling module estimation_1_sleep / utils
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from utils.toon_parser import load_default_toon_context, load_markdown_artifacts
except Exception:
    load_default_toon_context = None
    load_markdown_artifacts = None


# Paths
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "SLEEPINESS_NEW"
RESULTS_PICKLE = OUTPUT_DIR / "estimation_results.pkl"
CLUSTER_JSON = OUTPUT_DIR / "cluster_diagnostics.json"

OUT_DIR = (r"C:\Users\pedro\OneDrive\Documentos\Yale"
           r"\Year 3 (2024 - 2025)\Open Finance\Open-Finance"
           r"\Drafts\Deposit Competition")

def md_to_tex_string(md_text):
    blocks = md_text.split('$$')
    processed_blocks = []

    for i, block in enumerate(blocks):
        if i % 2 == 1:
            processed_blocks.append(f"\\[ {block} \\]")
        else:
            inline_parts = block.split('$')
            processed_inline = []
            for j, ipart in enumerate(inline_parts):
                if j % 2 == 1:
                    processed_inline.append(f"${ipart}$")
                else:
                    escaped = (ipart
                               .replace('&', '\\&')
                               .replace('%', '\\%')
                               .replace('#', '\\#')
                               .replace('_', '\\_'))
                    processed_inline.append(escaped)
            processed_blocks.append("".join(processed_inline))

    tex = "".join(processed_blocks)

    lines = tex.split('\n')
    out_lines = []
    in_code = False

    for line in lines:
        if line.startswith('```'):
            if in_code:
                out_lines.append('\\end{lstlisting}')
            else:
                out_lines.append('\\begin{lstlisting}')
            in_code = not in_code
            continue

        if in_code:
            out_lines.append(line)
            continue

        if line.startswith('# '):
            out_lines.append(f"\\section*{{{line[2:]}}}")
        elif line.startswith('## '):
            out_lines.append(f"\\subsection*{{{line[3:]}}}")
        elif line.startswith('### '):
            out_lines.append(f"\\subsubsection*{{{line[4:]}}}")
        elif line.startswith('- '):
            out_lines.append(f"\\textbullet\\ {line[2:]} \\\\")
        elif line.strip() == '':
            out_lines.append("\n\n")
        else:
            out_lines.append(line + " \\\\")

    return '\n'.join(out_lines)

def stars(p):
    if p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    return ''

def clean_name(v):
    if v == 'nr_lagged_dep': return 'Lagged Deposits'
    return str(v).replace('interaction_', '').replace('_', '\\_')

def build_first_stage_table(results_dict, G, G_star):
    columns = [
        ('Spec2-IV_CostShifters', 'IV Cost'), 
        ('Spec3-IV_Wholesale', 'IV Wholesale'), 
        ('Spec4-IV_HausmanFull', 'Hausman')
    ]
    
    vs = []
    for col_key, _ in columns:
        res = results_dict[f"{col_key} x Base"]['first_stage']
        for v in res.params.index:
            if v not in vs: vs.append(v)
            
    out = []
    out.append("\\begin{table}[htbp]\\centering")
    out.append("\\caption{First Stage Estimation (Control Function)}")
    out.append("\\begin{tabular}{l" + "c"*3 + "}\\toprule")
    out.append(" & " + " & ".join([n for _, n in columns]) + " \\\\ \\midrule")
    
    for var in vs:
        if var == 'const': continue
        coef_strs, se_strs = [], []
        for col_key, _ in columns:
            res = results_dict[f"{col_key} x Base"]['first_stage']
            if var in res.params:
                c, se, p = res.params[var], res.bse[var], res.pvalues[var]
                coef_strs.append(f"${c:.4f}^{{{stars(p)}}}$")
                se_strs.append(f"$({se:.4f})$")
            else:
                coef_strs.append("")
                se_strs.append("")
        out.append(f"{clean_name(var)} & " + " & ".join(coef_strs) + " \\\\")
        out.append(f" & " + " & ".join(se_strs) + " \\\\")
        
    obs_strs = []
    rsq_strs = []
    for col_key, _ in columns:
        res = results_dict[f"{col_key} x Base"]['first_stage']
        obs_strs.append(f"{int(res.nobs):,}")
        rsq_strs.append(f"{res.rsquared:.4f}")
        
    out.append("\\midrule")
    out.append("Obs & " + " & ".join(obs_strs) + " \\\\")
    out.append("$R^2$ & " + " & ".join(rsq_strs) + " \\\\")
    out.append("Fixed Effects & No & No & No \\\\")
    out.append(f"Clusters (G) & {G} & {G} & {G} \\\\")
    out.append(f"Effective Clusters ($G^*$) & {G_star:.2f} & {G_star:.2f} & {G_star:.2f} \\\\")
    
    out.append("\\bottomrule")
    out.append("\\end{tabular}\\end{table}")
    return "\n".join(out)

def build_second_stage_table(results_dict, G, G_star):
    columns = [
        ('Spec1-OLS', 'OLS'), 
        ('Spec2-IV_CostShifters', 'IV Cost'), 
        ('Spec3-IV_Wholesale', 'IV Wholesale'), 
        ('Spec4-IV_HausmanFull', 'Hausman')
    ]
    panels = [
        ('Base', ['nr_lagged_dep']), 
        ('Macro', ['nr_lagged_dep', 'gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young']), 
        ('Tech', ['nr_lagged_dep', 'gdp_per_capita', 'cadunico_families_per1000', 'fraction_65plus', 'fraction_young', 'pix_users_pf_per1000', 'connections_per100', 'branches_per1000'])
    ]
    
    out = []
    out.append("\\begin{table}[htbp]\\centering")
    out.append("\\caption{Second Stage Estimation}")
    out.append("\\resizebox{\\textwidth}{!}{")
    out.append("\\begin{tabular}{l" + "c"*4 + "}\\toprule")
    out.append(" & " + " & ".join([n for _, n in columns]) + " \\\\ \\midrule")
    
    for p_name, p_vars in panels:
        out.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{Panel: {p_name}}}}} \\\\ \\midrule")
        
        for vshort in p_vars:
            coef_strs, se_strs = [], []
            for col_key, _ in columns:
                spec_key = f"{col_key} x {p_name}"
                res = results_dict[spec_key]['second_stage']
                
                var = vshort
                if var not in res.params and f"interaction_{var}" in res.params:
                    var = f"interaction_{var}"
                    
                if var in res.params:
                    c, se, p = res.params[var], res.bse[var], res.pvalues[var]
                    coef_strs.append(f"${c:.4f}^{{{stars(p)}}}$")
                    se_strs.append(f"$({se:.4f})$")
                else:
                    coef_strs.append("")
                    se_strs.append("")
                    
            out.append(f"{clean_name(vshort)} & " + " & ".join(coef_strs) + " \\\\")
            out.append(f" & " + " & ".join(se_strs) + " \\\\")
            
        obs_strs = []
        rsq_strs = []
        for col_key, _ in columns:
            res = results_dict[f"{col_key} x {p_name}"]['second_stage']
            obs_strs.append(f"{int(res.nobs):,}")
            rsq_strs.append(f"{res.rsquared:.4f}")
            
        out.append("\\midrule")
        out.append("Obs & " + " & ".join(obs_strs) + " \\\\")
        out.append("$R^2$ & " + " & ".join(rsq_strs) + " \\\\")
        out.append("Fixed Effects & Yes & Yes & Yes & Yes \\\\")
        out.append(f"Clusters (G) & {G} & {G} & {G} & {G} \\\\")
        out.append(f"Effective Clusters ($G^*$) & {G_star:.2f} & {G_star:.2f} & {G_star:.2f} & {G_star:.2f} \\\\")
        if p_name != 'Tech':
            out.append("\\midrule")

    out.append("\\bottomrule")
    out.append("\\end{tabular}}")
    out.append("\\end{table}")
    return "\n".join(out)

def main():
    print("=====================================================================")
    print(" INITIATING PDFLATEX COMPILATION PIPELINE")
    print("=====================================================================")

    default_brain_dir = r"C:\Users\pedro\.gemini\antigravity\brain\13d50388-cbdf-4a6d-bc55-492e6d1a04f0"

    print(" - Loading markdown artifacts...")
    toon_context = {}
    if load_default_toon_context is not None:
        try:
            toon_context = load_default_toon_context(Path(_THIS_DIR) / "utils", quiet=True)
        except Exception as e:
            print("Failed to load toon context:", e)

    try:
        if load_markdown_artifacts is not None:
            summary_text, se_text, md_meta = load_markdown_artifacts(
                toon_context,
                default_brain_dir=default_brain_dir,
            )
        else:
            brain_dir = default_brain_dir
            summary_md = os.path.join(brain_dir, 'estimation_summary.md')
            se_md = os.path.join(brain_dir, 'se_comparison.md')
            with open(summary_md, 'r', encoding='utf-8') as f:
                summary_text = f.read()
            with open(se_md, 'r', encoding='utf-8') as f:
                se_text = f.read()
    except Exception as e:
        print("Could not load markdown artifacts:", e)
        summary_text = ""
        se_text = ""

    tex_body_md = ""
    if summary_text and se_text:
        print(" - Translating markdown to LaTeX...")
        tex_body_md = (md_to_tex_string(summary_text) + "\n\n\\newpage\n" + md_to_tex_string(se_text))
    else:
        print(" - Markdown artifacts not found or empty.")

    if not RESULTS_PICKLE.exists() or not CLUSTER_JSON.exists():
        print("Results not found. Run estimation_1_sleep.py first.")
        return

    print(" - Reading pickled model estimates...")
    with open(RESULTS_PICKLE, 'rb') as f:
        results_dict = pickle.load(f)
        
    with open(CLUSTER_JSON, 'r') as f:
        cluster_data = json.load(f)

    G_nominal = cluster_data['G_nominal']
    G_star = cluster_data['G_star']

    print(" - Generating strict regression tables...")
    fs_table_tex = build_first_stage_table(results_dict, G_nominal, G_star)
    ss_table_tex = build_second_stage_table(results_dict, G_nominal, G_star)

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
\usepackage{graphicx}
\usepackage{xcolor}

\definecolor{yalegray}{RGB}{89,89,89}
\definecolor{yaleblue}{RGB}{0,53,107}
\definecolor{codebg}{RGB}{245,245,245}
\definecolor{codeframe}{RGB}{220,220,220}

\begin{document}

\title{Estimation Summary --- Deposit Competition}
\author{Autogenerated Report}
\date{\today}
\maketitle

"""
    
    tex_doc = (
        preamble
        + tex_body_md
        + "\n\\newpage\n\\section*{Regression Results}\n"
        + fs_table_tex
        + "\n\n"
        + ss_table_tex
        + "\n\\end{document}\n"
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    temp_tex = os.path.join(OUT_DIR, "Agent_Comments_Export.tex")
    print(f" - Writing single LaTeX file to {temp_tex} ...")
    with open(temp_tex, 'w', encoding='utf-8') as f:
        f.write(tex_doc)
    
    print("\n - Compiling...")
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "Agent_Comments_Export.tex"],
                       cwd=OUT_DIR, capture_output=True, text=True)
        res_final = subprocess.run(["pdflatex", "-interaction=nonstopmode", "Agent_Comments_Export.tex"],
                       cwd=OUT_DIR, capture_output=True, text=True) 

        pdf_path = os.path.join(OUT_DIR, "Agent_Comments_Export.pdf")       
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            print("\n *** PDF SUCCESSFULLY GENERATED. ***\n")
        else:
            print(f"\n *** PDF GENERATION FAILED. Log tail:\n{res_final.stdout[-800:]}\n ***\n")
    except Exception as e:
        print(f"\n *** COMPILATION ERROR: {e} ***\n")

    print("Done.")

if __name__ == '__main__':
    main()
