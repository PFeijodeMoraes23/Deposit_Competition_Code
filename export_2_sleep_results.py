
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import os
import sys
import json
import pickle
import subprocess
import shutil
import warnings
warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# est_dir/rout_dir follow SLEEP_OUT_ROOT, so a sandboxed run exports the fits it just
# produced instead of whatever sits in the production tree.
from utils import paths  # noqa: E402
from utils import routines as _routines  # noqa: E402

_DRAFTS_DIR = paths.drafts_dir()
OUTPUT_DIR = paths.est_dir(2)
RESULTS_PICKLE = OUTPUT_DIR / "estimation_results.pkl"

# The paper directory is the ONE destination: the fragments V_Main.tex \input, the
# standalone wrapper and the preview PDF are all built here, so there is no second copy
# free to diverge from the one the paper reads.
TEX_OUT_DIR = _DRAFTS_DIR
OUT_DIR = str(TEX_OUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def stars(p):
    if p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    return ''

# Row labels and display units come from the SHARED registry so that every table in the
# paper -- sleepiness, BBL policy functions, descriptives -- states the same unit for the
# same variable. There used to be four independent copies of this dict.
from sleep_export_link import clean_name, disp  # noqa: E402
from utils import state_transform as _st  # noqa: E402
from utils import se_national as _sen  # noqa: E402


def build_first_stage_table(results_dict):
    panels = ['Base', 'Macro', 'Tech']
    panel_labels = {'Base': 'Base Specifications', 'Macro': 'Macro Specifications', 'Tech': 'Tech Specifications'}
    panel_letters = ['A', 'B', 'C']
    fs_spec_numbers = {
        ('Base', 'IV_CostShifters'): 2, ('Base', 'IV_Wholesale'): 3, ('Base', 'IV_HausmanFull'): 4,
        ('Macro', 'IV_CostShifters'): 6, ('Macro', 'IV_Wholesale'): 7, ('Macro', 'IV_HausmanFull'): 8,
        ('Tech', 'IV_CostShifters'): 10, ('Tech', 'IV_Wholesale'): 11, ('Tech', 'IV_HausmanFull'): 12,
    }
    ivs = [('IV_CostShifters', 'IV Cost'), ('IV_Wholesale', 'IV Wholesale'), ('IV_HausmanFull', 'Hausman')]
    multispan = 4
    # SHARED first stage.  The first stage is the deposit-spread (price) equation and does not
    # depend on the second-stage link, so it is numerically identical to E3's (same pooled
    # sample, N=487,046, same 10 coefficients). Each routine nonetheless reports its own table
    # so the panels shown belong to that routine: E2 carries Base/Macro/Tech, while the link
    # routines have no Base specification because their index excludes the constant. E4 adds
    # one control (the Time block, 11 coefficients) and E1 runs on the local B-type sample
    # (N=486,233), so neither matches this one.
    caption = ("First Stage --- Deposit Spread on Instruments "
               f"--- Estimation Strategy~{_routines.est_ref(2)}")
    label = "tab:est2_first_stage"
    notes = (
        r"\footnotesize \textit{Notes:} Standard errors (wild cluster bootstrap at the "
        r"conglomerate level; \textcite{cameron2008bootstrap}, \textcite{mackinnon2017wild}) "
        r"in parentheses. Coefficients are in \emph{percentage points of the quarterly "
        r"deposit spread} per the unit given in the row label, matching the units of the "
        r"second-stage tables. $t$-statistics, $p$-values and significance stars are "
        r"invariant to these units. "
        r"Significance levels: *** $p<0.01$, ** $p<0.05$, * $p<0.1$."
    )

    def _get_res(iv_key, p):
        entry = results_dict.get(f"{iv_key} x {p}")
        return entry.get('first_stage') if isinstance(entry, dict) else None

    p0, l0 = panels[0], panel_letters[0]
    iv_nums_0 = [(il, fs_spec_numbers[(p0, ik)]) for ik, il in ivs]

    lines = [
        r"\begin{spacing}{1.0}",
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
        rf"    \caption[]{{{caption} (Continued)}} \\",
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

    all_vars_fs = [
        'tax_cost_ratio_lag', 'personnel_cost_ratio_lag', 'admin_cost_ratio_lag',
        'indice_basileia_lag', 'lci_lca_ratio_lag', 'wholesale_ratio_lag', 'leave_one_out_mean_spread',
    ]

    for pi, panel in enumerate(panels):
        letter = panel_letters[pi]
        plabel = panel_labels[panel]
        iv_nums = [(il, fs_spec_numbers[(panel, ik)]) for ik, il in ivs]

        if pi > 0:
            lines += [
                r"    \addlinespace[1.5em]", "",
                r"    \toprule",
                rf"    \multicolumn{{{multispan}}}{{l}}{{\textbf{{Panel {letter}: {plabel}}}}} \\",
                r"    \midrule",
                "    & " + " & ".join(il for il, _ in iv_nums) + r" \\",
                "    & " + " & ".join(f"({n})" for _, n in iv_nums) + r" \\",
                r"    \midrule",
            ]

        for var in all_vars_fs:
            coef_strs, se_strs, has_val = [], [], False
            for ik, _ in ivs:
                res = _get_res(ik, panel)
                if res is not None and var in res.params:
                    has_val = True
                    c, se, pval = res.params[var], res.bse[var], res.pvalues[var]
                    m = disp(var, lhs=_st.SPREAD_DISPLAY)   # LHS here is the spread, not phi
                    c, se = c * m, se * m
                    coef_strs.append(f"${c:.4f}^{{{stars(pval)}}}$")
                    se_strs.append(f"$({se:.4f})$")
                else:
                    coef_strs.append(""); se_strs.append("")
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
            fv = getattr(res, 'fvalue', None); fp = getattr(res, 'f_pvalue', 1.0)
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

    lines += [r"\end{xltabular}", r"\end{spacing}"]
    return "\n".join(lines)


def build_second_stage_table(results_dict):
    panels = ['Base', 'Macro', 'Tech']
    panel_labels = {'Base': 'Base Specifications', 'Macro': 'Macro Specifications', 'Tech': 'Tech Specifications'}
    panel_letters = ['A', 'B', 'C']
    ss_spec_numbers = {
        ('Base', 'OLS'): 1, ('Base', 'IV_CostShifters'): 2, ('Base', 'IV_Wholesale'): 3, ('Base', 'IV_HausmanFull'): 4,
        ('Macro', 'OLS'): 5, ('Macro', 'IV_CostShifters'): 6, ('Macro', 'IV_Wholesale'): 7, ('Macro', 'IV_HausmanFull'): 8,
        ('Tech', 'OLS'): 9, ('Tech', 'IV_CostShifters'): 10, ('Tech', 'IV_Wholesale'): 11, ('Tech', 'IV_HausmanFull'): 12,
    }
    estimators = [('OLS', 'OLS'), ('IV_CostShifters', 'IV Cost'), ('IV_Wholesale', 'IV Wholesale'), ('IV_HausmanFull', 'Hausman')]
    multispan = 5
    caption = f"Second Stage --- Estimation Strategy~{_routines.est_ref(2)}"
    label = "tab:est2_second_stage"
    notes = (
        r"\footnotesize \textit{Notes:} Standard errors (wild cluster bootstrap at the "
        r"conglomerate level; \textcite{cameron2008bootstrap}, \textcite{mackinnon2017wild}) "
        r"in parentheses. Coefficients are effects on $\phi$, expressed in "
        r"\emph{percentage points of the sleepy share} per the unit given in the row "
        r"label; shares and rates are in percentage points, and Pix Available is a discrete "
        r"$0\to1$ difference. $t$-statistics, $p$-values and significance stars are "
        r"invariant to these units. State variables are grand-mean centred at their pooled "
        r"estimation-sample means, so the Constant is $\hat{\phi}$ at the average market. "
        # Filled in at the END of this function from the schemes select_se actually returned,
        # so the note can never describe a calculation that did not run (adversarial review,
        # 2026-07-30: the previous unconditional wording claimed a quarter-clustered bootstrap
        # on tables where every national cell had fallen back to conglomerate).
        + _sen.NOTE_TOKEN +
        r"Significance levels: *** $p<0.01$, ** $p<0.05$, * $p<0.1$."
    )

    def _get_res(ek, p):
        entry = results_dict.get(f"{ek} x {p}")
        return entry.get('second_stage') if isinstance(entry, dict) else None

    all_vars = [
        'constant', 'pix_exists', 'risk_free_qoq_lag',
        'gdp_per_capita', 'cadunico_families_per1000',
        'fraction_65plus', 'fraction_young', 'connections_per100',
    ]

    p0, l0 = panels[0], panel_letters[0]
    est_nums_0 = [(el, ss_spec_numbers[(p0, ek)]) for ek, el in estimators]

    lines = [
        r"\begin{spacing}{1.0}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{xltabular}{\textwidth}{>{\raggedright\arraybackslash}p{0.34\textwidth} *{4}{>{\centering\arraybackslash}X}}",
        rf"    \caption{{{caption}}}\label{{{label}}} \\",
        r"    \toprule",
        rf"    \multicolumn{{{multispan}}}{{l}}{{\textbf{{Panel {l0}: {panel_labels[p0]}}}}} \\",
        r"    \midrule",
        "     & " + " & ".join(el for el, _ in est_nums_0) + r" \\",
        "    & " + " & ".join(f"({n})" for _, n in est_nums_0) + r" \\",
        r"    \midrule",
        r"    \endfirsthead",
        "",
        rf"    \caption[]{{{caption} (Continued)}} \\",
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
                r"    \addlinespace[1.5em]", "",
                r"    \toprule",
                rf"    \multicolumn{{{multispan}}}{{l}}{{\textbf{{Panel {letter}: {plabel}}}}} \\",
                r"    \midrule",
                "     & " + " & ".join(el for el, _ in est_nums) + r" \\",
                "    & " + " & ".join(f"({n})" for _, n in est_nums) + r" \\",
                r"    \midrule",
            ]

        _nat_schemes = set()

        for vshort in all_vars:
            coef_strs, se_strs, dk_strs, has_val = [], [], [], False
            for ek, _ in estimators:
                res = _get_res(ek, panel)
                var = vshort
                if vshort in ('constant',) and res is not None and 'nr_lagged_dep' in res.params:
                    var = 'nr_lagged_dep'
                elif res is not None and var not in res.params and f"interaction_{var}" in res.params:
                    var = f"interaction_{var}"
                if res is not None and var in res.params:
                    has_val = True
                    c = res.params[var]
                    # National rows (Pix, Selic) report the quarter-clustered bootstrap; every
                    # other row keeps the conglomerate one. See utils/se_national.select_se.
                    se, pval, scheme = _sen.select_se(res, var)
                    if _sen.is_national(var):
                        _nat_schemes.add(scheme)
                    m = disp(vshort)          # coefficient and SE only; pval/stars unchanged
                    c, se = c * m, se * m
                    mark = f"^{{{_sen.SE_MARK}}}" if scheme != "congl" else ""
                    # Driscoll-Kraay, in brackets, on the national rows only. Every column of
                    # this table is linear, so DK is defined throughout and nothing is mixed.
                    dkv = _sen.dk_se(res, var) if _sen.is_national(var) else None
                    dk_strs.append(f"$[{dkv * m:.4f}]$" if dkv is not None else "")
                    coef_strs.append(f"${c:.4f}^{{{stars(pval)}}}$")
                    se_strs.append(f"$({se:.4f}){mark}$")
                else:
                    coef_strs.append(""); se_strs.append(""); dk_strs.append("")
            if has_val:
                lines.append(f"    {clean_name(vshort)} & " + " & ".join(coef_strs) + r" \\")
                lines.append("    & " + " & ".join(se_strs) + r" \\")
                if any(dk_strs):
                    lines.append("    & " + " & ".join(dk_strs) + r" \\")

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

    lines += [r"\end{xltabular}", r"\end{spacing}"]
    return "\n".join(lines).replace(
        _sen.NOTE_TOKEN, _sen.national_note(_nat_schemes, dk_bracket=True))


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
    print(" EXPORT EST2 SLEEP RESULTS (Pooled B+D Linear)")
    print("=" * 70)

    # A missing pickle is a hard failure, not a skip: without it this script writes no
    # table at all, and a silent return leaves the orchestrator reporting success over a
    # step that produced nothing.
    if not RESULTS_PICKLE.exists():
        print(f"Results not found at {RESULTS_PICKLE}. Run sleep_est_e2.py first.")
        sys.exit(1)

    print(" - Reading pickled model estimates...")
    with open(RESULTS_PICKLE, 'rb') as fh:
        results_dict = pickle.load(fh)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(" - Building table fragments...")
    fs_frag = build_first_stage_table(results_dict)
    ss_frag = build_second_stage_table(results_dict)

    # One first-stage table per routine, matching est1/est3/est4.
    FS_NAME = "est2_first_stage_table.tex"
    fs_path = os.path.join(OUT_DIR, FS_NAME)
    ss_path = os.path.join(OUT_DIR, "est2_second_stage_table.tex")
    with open(fs_path, 'w', encoding='utf-8') as fh: fh.write(fs_frag + "\n")
    with open(ss_path, 'w', encoding='utf-8') as fh: fh.write(ss_frag + "\n")
    print(f" - Fragments written to {_DRAFTS_DIR}")

    tex_doc = (
        _STANDALONE_PREAMBLE
        + r"\begin{document}" + "\n"
        + r"\title{Sleep Estimation Results --- Est 2 (Pooled B+D Linear)}" + "\n"
        + r"\author{Autogenerated}" + "\n"
        + r"\date{\today}" + "\n"
        + r"\maketitle" + "\n\n"
        + r"\section*{First Stage}" + "\n"
        + rf"\input{{{FS_NAME}}}" + "\n\n"
        + r"\section*{Second Stage}" + "\n"
        + r"\input{est2_second_stage_table.tex}" + "\n"
        + r"\end{document}" + "\n"
    )

    wrapper_path = os.path.join(OUT_DIR, "est2_sleep_results.tex")
    with open(wrapper_path, 'w', encoding='utf-8') as fh: fh.write(tex_doc)
    print(f" - Standalone wrapper written to {_DRAFTS_DIR}")

    print(" - Compiling PDF...")
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "est2_sleep_results.tex"],
                       cwd=OUT_DIR, capture_output=True, text=True)
        res_final = subprocess.run(["pdflatex", "-interaction=nonstopmode", "est2_sleep_results.tex"],
                                   cwd=OUT_DIR, capture_output=True, text=True)
        pdf_path = os.path.join(OUT_DIR, "est2_sleep_results.pdf")
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            print("\n *** PDF SUCCESSFULLY GENERATED. ***\n")
        else:
            print(f"\n *** PDF GENERATION FAILED. Log tail:\n{res_final.stdout[-800:]}\n ***\n")
    except Exception as e:
        print(f"\n *** COMPILATION ERROR: {e} ***\n")

    print("Done.")


if __name__ == '__main__':
    main()
