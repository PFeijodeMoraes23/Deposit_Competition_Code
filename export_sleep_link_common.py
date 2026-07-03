"""
export_sleep_link_common.py
================================================================================
Shared TeX table exporter for the link-based sleepiness routines Est4/Est5/Est6.
Mirrors export_3_sleep_results.py (first/second-stage longtables, standalone PDF)
but parametrised by estimation number, link description and notes, so the three
wrappers (export_4/5/6_sleep_results.py) stay one line each.

Reported second-stage estimates are Average Marginal Effects (REPORTING ONLY);
phi itself is built from native coefficients in estimation_N_sleep / demand prep.
Pickles reference utils.sleep_links.NonLinearResults, so no module-mock is needed.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import os
import sys
import json
import pickle
import subprocess
import shutil
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")

from utils.sleep_links import NonLinearResults  # noqa: F401 (needed for unpickling)
from utils import paths as _paths_mod

_DRAFTS_DIR = Path(r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Drafts\Deposit Competition")
DATA_DIR = _paths_mod.PROCESSED
TEX_OUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"


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
        'connections_per100': 'Broadband Connections (per 100 inhabitants)',
        'pix_exists': 'Pix Available',
        'const': 'Constant', 'constant': 'Constant',
        'tax_cost_ratio_lag': 'Tax Cost Ratio ($t-1$)',
        'personnel_cost_ratio_lag': 'Personnel Cost Ratio ($t-1$)',
        'admin_cost_ratio_lag': 'Admin Cost Ratio ($t-1$)',
        'indice_basileia_lag': 'Basel Index (pp, $t-1$)',
        'lci_lca_ratio_lag': 'LCI/LCA Ratio ($t-1$)',
        'wholesale_ratio_lag': 'Wholesale Ratio ($t-1$)',
        'leave_one_out_mean_spread': 'Leave-out Mean Spread',
        'v_hat_x_lagged_dep': 'CF: $\\hat{v} \\times$ Lagged Deposits',
        'time_trend': 'Time Trend (years)',
        'gdp_growth_yoy': 'GDP Growth (YoY)',
    }
    return labels.get(v, v.replace('_', '\\_'))


# Appendix caption = stage + a reference to the estimation strategy enumerated in V_Main
# (Section ref{sec:empirical:sleep}); no ad-hoc strategy names. E3/E4 (logit) are not in the
# appendix and have no enumerate label, so they fall back to a plain (Est. N) tag.
EST_LABEL = {5: "single_idx", 6: "single_idx_time", 7: "joint_sieve", 8: "joint_sieve_time"}


def _strategy_caption(stage, est_num):
    lbl = EST_LABEL.get(est_num)
    ref = rf"\ref{{estimation:{lbl}}}" if lbl else rf"(Est.\ {est_num})"
    return rf"{stage} --- Estimation Strategy~{ref}"


def _panels_for(est_num):
    # E5-E8 (single-index/joint sieve) drop the Base panel: their index excludes the
    # constant, so a constant-only Base has no index. E3/E4 (logit) keep all three.
    return ['Macro', 'Tech'] if est_num >= 5 else ['Base', 'Macro', 'Tech']


def build_first_stage_table(results_dict, est_num):
    panels = _panels_for(est_num)
    panel_labels = {'Base': 'Base Specifications', 'Macro': 'Macro Specifications', 'Tech': 'Tech Specifications'}
    panel_letters = ['A', 'B', 'C']
    fs_spec_numbers = {
        ('Base', 'IV_CostShifters'): 2, ('Base', 'IV_Wholesale'): 3, ('Base', 'IV_HausmanFull'): 4,
        ('Macro', 'IV_CostShifters'): 6, ('Macro', 'IV_Wholesale'): 7, ('Macro', 'IV_HausmanFull'): 8,
        ('Tech', 'IV_CostShifters'): 10, ('Tech', 'IV_Wholesale'): 11, ('Tech', 'IV_HausmanFull'): 12,
    }
    ivs = [('IV_CostShifters', 'IV Cost'), ('IV_Wholesale', 'IV Wholesale'), ('IV_HausmanFull', 'Hausman')]
    multispan = 4
    caption = _strategy_caption("First Stage", est_num)
    label = f"tab:est{est_num}_first_stage"
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
        rf"    \caption{{{caption}}}\label{{{label}}} \\", r"    \toprule",
        rf"    \multicolumn{{{multispan}}}{{l}}{{\textbf{{Panel {l0}: {panel_labels[p0]}}}}} \\", r"    \midrule",
        "    & " + " & ".join(il for il, _ in iv_nums_0) + r" \\",
        "    & " + " & ".join(f"({n})" for _, n in iv_nums_0) + r" \\", r"    \midrule", r"    \endfirsthead", "",
        rf"    \multicolumn{{{multispan}}}{{c}}{{{{\bfseries Table \thetable\ continued from previous page}}}} \\",
        r"    \toprule", "    & " + " & ".join(il for il, _ in iv_nums_0) + r" \\", r"    \midrule", r"    \endhead", "",
        r"    \midrule", rf"    \multicolumn{{{multispan}}}{{r}}{{\textit{{Continued on next page}}}} \\", r"    \endfoot", "",
        r"    \bottomrule", rf"    \multicolumn{{{multispan}}}{{p{{\dimexpr\textwidth-2\tabcolsep\relax}}}}{{{notes}}} \\", r"    \endlastfoot", "",
    ]
    all_vars_fs = ['tax_cost_ratio_lag', 'personnel_cost_ratio_lag', 'admin_cost_ratio_lag',
                   'indice_basileia_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'leave_one_out_mean_spread']
    for pi, panel in enumerate(panels):
        letter = panel_letters[pi]
        iv_nums = [(il, fs_spec_numbers[(panel, ik)]) for ik, il in ivs]
        if pi > 0:
            lines += [r"    \addlinespace[1.5em]", "", r"    \toprule",
                      rf"    \multicolumn{{{multispan}}}{{l}}{{\textbf{{Panel {letter}: {panel_labels[panel]}}}}} \\",
                      r"    \midrule", "    & " + " & ".join(il for il, _ in iv_nums) + r" \\",
                      "    & " + " & ".join(f"({n})" for _, n in iv_nums) + r" \\", r"    \midrule"]
        for var in all_vars_fs:
            coef_strs, se_strs, has_val = [], [], False
            for ik, _ in ivs:
                res = _get_res(ik, panel)
                if res is not None and var in res.params:
                    has_val = True
                    c, se, pval = res.params[var], res.bse[var], res.pvalues[var]
                    coef_strs.append(f"${c:.4f}^{{{stars(pval)}}}$"); se_strs.append(f"$({se:.4f})$")
                else:
                    coef_strs.append(""); se_strs.append("")
            if has_val:
                lines.append(f"    {clean_name(var)} & " + " & ".join(coef_strs) + r" \\*")
                lines.append("    & " + " & ".join(se_strs) + r" \\")
        obs_l, rsq_l, fstat_l, g_l = [], [], [], []
        for ik, _ in ivs:
            res = _get_res(ik, panel)
            if res is None:
                obs_l.append("---"); rsq_l.append("---"); fstat_l.append("---"); g_l.append("---"); continue
            obs_l.append(f"{int(res.nobs):,}"); rsq_l.append(f"{res.rsquared:.4f}")
            fv = getattr(res, 'fvalue', None); fp = getattr(res, 'f_pvalue', 1.0)
            fstat_l.append(f"${fv:.2f}^{{{stars(fp)}}}$" if fv is not None else "---")
            g_l.append(str(getattr(res, 'G_nominal', '---')))
        lines += [r"    \midrule", "    Observations & " + " & ".join(obs_l) + r" \\",
                  "    $R^2$ & " + " & ".join(rsq_l) + r" \\", "    F-Statistic & " + " & ".join(fstat_l) + r" \\",
                  "    Fixed Effects & No & No & No \\\\", "    Clusters ($G$) & " + " & ".join(g_l) + r" \\",
                  r"    \bottomrule"]
    lines += [r"\end{xltabular}", r"\doublespacing"]
    return "\n".join(lines)


def build_second_stage_table(results_dict, est_num):
    panels = _panels_for(est_num)
    panel_labels = {'Base': 'Base Specifications', 'Macro': 'Macro Specifications', 'Tech': 'Tech Specifications'}
    panel_letters = ['A', 'B', 'C']
    ss_spec_numbers = {
        ('Base', 'OLS'): 1, ('Base', 'IV_CostShifters'): 2, ('Base', 'IV_Wholesale'): 3, ('Base', 'IV_HausmanFull'): 4,
        ('Macro', 'OLS'): 5, ('Macro', 'IV_CostShifters'): 6, ('Macro', 'IV_Wholesale'): 7, ('Macro', 'IV_HausmanFull'): 8,
        ('Tech', 'OLS'): 9, ('Tech', 'IV_CostShifters'): 10, ('Tech', 'IV_Wholesale'): 11, ('Tech', 'IV_HausmanFull'): 12,
    }
    estimators = [('OLS', 'OLS'), ('IV_CostShifters', 'IV Cost'), ('IV_Wholesale', 'IV Wholesale'), ('IV_HausmanFull', 'Hausman')]
    multispan = 5
    label = f"tab:est{est_num}_second_stage"
    caption = _strategy_caption("Second Stage", est_num)
    notes = (
        r"\footnotesize \textit{Notes:} Standard errors (wild cluster bootstrap at the "
        r"conglomerate level; \textcite{cameron2008bootstrap}, \textcite{mackinnon2017wild}) "
        r"in parentheses. Reported effects are average marginal effects (AMEs). "
        r"Significance levels: *** $p<0.01$, ** $p<0.05$, * $p<0.1$."
    )

    def _get_res(ek, p):
        entry = results_dict.get(f"{ek} x {p}")
        return entry.get('second_stage') if isinstance(entry, dict) else None

    all_vars = ['constant', 'pix_exists', 'risk_free_qoq_lag', 'gdp_per_capita',
                'cadunico_families_per1000', 'fraction_65plus', 'fraction_young',
                'connections_per100', 'time_trend', 'gdp_growth_yoy', 'v_hat_x_lagged_dep']
    p0, l0 = panels[0], panel_letters[0]
    est_nums_0 = [(el, ss_spec_numbers[(p0, ek)]) for ek, el in estimators]
    lines = [
        r"\setstretch{1.0}",
        r"\begin{xltabular}{\textwidth}{>{\raggedright\arraybackslash}p{0.28\textwidth} *{4}{>{\centering\arraybackslash}X}}",
        rf"    \caption{{{caption}}}\label{{{label}}} \\", r"    \toprule",
        rf"    \multicolumn{{{multispan}}}{{l}}{{\textbf{{Panel {l0}: {panel_labels[p0]}}}}} \\", r"    \midrule",
        "     & " + " & ".join(el for el, _ in est_nums_0) + r" \\",
        "    & " + " & ".join(f"({n})" for _, n in est_nums_0) + r" \\", r"    \midrule", r"    \endfirsthead", "",
        rf"    \multicolumn{{{multispan}}}{{c}}{{{{\bfseries Table \thetable\ continued from previous page}}}} \\",
        r"    \toprule", r"    \endhead", "", r"    \midrule",
        rf"    \multicolumn{{{multispan}}}{{r}}{{\textit{{Continued on next page}}}} \\", r"    \endfoot", "",
        r"    \bottomrule", rf"    \multicolumn{{{multispan}}}{{p{{\dimexpr\textwidth-2\tabcolsep\relax}}}}{{{notes}}} \\", r"    \endlastfoot", "",
    ]
    for pi, panel in enumerate(panels):
        letter = panel_letters[pi]
        est_nums = [(el, ss_spec_numbers[(panel, ek)]) for ek, el in estimators]
        if pi > 0:
            lines += [r"    \addlinespace[1.5em]", "", r"    \toprule",
                      rf"    \multicolumn{{{multispan}}}{{l}}{{\textbf{{Panel {letter}: {panel_labels[panel]}}}}} \\",
                      r"    \midrule", "     & " + " & ".join(el for el, _ in est_nums) + r" \\",
                      "    & " + " & ".join(f"({n})" for _, n in est_nums) + r" \\", r"    \midrule"]
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
                    coef_strs.append(f"${c:.4f}^{{{stars(pval)}}}$"); se_strs.append(f"$({se:.4f})$")
                else:
                    coef_strs.append(""); se_strs.append("")
            if has_val:
                lines.append(f"    {clean_name(vshort)} & " + " & ".join(coef_strs) + r" \\")
                lines.append("    & " + " & ".join(se_strs) + r" \\")
        obs_l, rsq_l, g_l = [], [], []
        for ek, _ in estimators:
            res = _get_res(ek, panel)
            if res is None:
                obs_l.append("---"); rsq_l.append("---"); g_l.append("---"); continue
            nv = getattr(res, 'nobs', None); obs_l.append(f"{int(nv):,}" if nv is not None else "---")
            rv = getattr(res, 'rsquared', None); rsq_l.append(f"{rv:.4f}" if rv is not None else "---")
            g_l.append(str(getattr(res, 'G_nominal', '---')))
        lines += [r"    \midrule", "    Observations & " + " & ".join(obs_l) + r" \\",
                  "    $R^2$ & " + " & ".join(rsq_l) + r" \\", "    Fixed Effects & Yes & Yes & Yes & Yes \\\\",
                  "    Clusters ($G$) & " + " & ".join(g_l) + r" \\", r"    \bottomrule"]
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


def export_link_results(est_num, title):
    """Build est{est_num} first/second-stage TeX tables + standalone PDF."""
    out_dir = DATA_DIR / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / f"est{est_num}"
    results_pickle = out_dir / "estimation_results.pkl"
    print("=" * 70); print(f" EXPORT EST{est_num} SLEEP RESULTS"); print("=" * 70)
    if not results_pickle.exists():
        print(f"Results not found. Run estimation_{est_num}_sleep.py first."); return
    with open(results_pickle, 'rb') as fh:
        results_dict = pickle.load(fh)

    TEX_OUT_DIR.mkdir(parents=True, exist_ok=True)
    fs_frag = build_first_stage_table(results_dict, est_num)
    ss_frag = build_second_stage_table(results_dict, est_num)

    fs_name = f"est{est_num}_first_stage_table.tex"
    ss_name = f"est{est_num}_second_stage_table.tex"
    (TEX_OUT_DIR / fs_name).write_text(fs_frag + "\n", encoding="utf-8")
    (TEX_OUT_DIR / ss_name).write_text(ss_frag + "\n", encoding="utf-8")
    for n in (fs_name, ss_name):
        shutil.copy(TEX_OUT_DIR / n, _DRAFTS_DIR / n)
    print(f" - Fragments written and copied to {_DRAFTS_DIR}")

    tex_doc = (
        _STANDALONE_PREAMBLE + r"\begin{document}" + "\n"
        + rf"\title{{Sleep Estimation Results --- {title}}}" + "\n"
        + r"\author{Autogenerated}" + "\n" + r"\date{\today}" + "\n" + r"\maketitle" + "\n\n"
        + r"\section*{First Stage}" + "\n" + rf"\input{{{fs_name}}}" + "\n\n"
        + r"\section*{Second Stage}" + "\n" + rf"\input{{{ss_name}}}" + "\n"
        + r"\end{document}" + "\n"
    )
    wrapper_name = f"est{est_num}_sleep_results.tex"
    (TEX_OUT_DIR / wrapper_name).write_text(tex_doc, encoding="utf-8")
    shutil.copy(TEX_OUT_DIR / wrapper_name, _DRAFTS_DIR / wrapper_name)
    print(f" - Standalone wrapper written and copied to {_DRAFTS_DIR}")

    print(" - Compiling PDF...")
    try:
        for _ in range(2):
            res_final = subprocess.run(["pdflatex", "-interaction=nonstopmode", wrapper_name],
                                       cwd=str(TEX_OUT_DIR), capture_output=True, text=True)
        pdf_path = TEX_OUT_DIR / f"est{est_num}_sleep_results.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            print("\n *** PDF SUCCESSFULLY GENERATED. ***\n")
        else:
            print(f"\n *** PDF GENERATION FAILED. Log tail:\n{res_final.stdout[-800:]}\n ***\n")
    except Exception as e:
        print(f"\n *** COMPILATION ERROR: {e} ***\n")
    print("Done.")


# ── Config-driven CLI for E3-E8 tables (E1/E2 + E9 have their own export scripts) ──
#  Standalone-preview title per estimator; captions/notes are derived from est_num.
#  Run:  python export_sleep_link_common.py --est N
EXPORT_CFG = {
    3: r"E3: Pooled Logit (single-index, logistic link)",
    4: r"E4: Pooled Logit + Time block",
    5: r"E5: Pooled Single-Index (nonparametric link)",
    6: r"E6: Pooled Single-Index + Time block",
    7: r"E7: Pooled Joint Single-Index (monotone sieve)",
    8: r"E8: Pooled Joint Single-Index (sieve) + Time block",
}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Export E3-E8 sleep tables (config-driven).")
    p.add_argument("--est", type=int, required=True, choices=sorted(EXPORT_CFG),
                   help="Estimator id 3-8 (E1/E2 + E9 have their own export scripts)")
    a = p.parse_args()
    export_link_results(a.est, EXPORT_CFG[a.est])
