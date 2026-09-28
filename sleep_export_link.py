"""
sleep_export_link.py
================================================================================
Shared TeX table exporter for the link-based sleepiness routines E3/E4 (see
EXPORT_CFG at the bottom). Mirrors sleep_export_e2.py (first/second-stage
longtables, standalone PDF) but parametrised by estimation number, so one
`--est N` invocation covers each of them.

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
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")

from utils.sleep_links import NonLinearResults  # noqa: F401 (needed for unpickling)
from utils import paths as _paths_mod
from utils import state_transform as _st
from utils import se_national as _sen
from utils import sleep_notes as _notes

_DRAFTS_DIR = _paths_mod.drafts_dir()
# rout_dir/est_dir follow SLEEP_OUT_ROOT, so a sandboxed run exports the fits it just
# produced instead of whatever sits in the production tree.
# The paper directory is the ONE destination: the fragments V_Main.tex \input, the
# standalone wrapper and the preview PDF are all built here, so there is no second copy
# free to diverge from the one the paper reads.
TEX_OUT_DIR = _DRAFTS_DIR


def stars(p):
    if p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    return ''


# Base row labels. The UNIT in each label is appended from utils/state_transform.DISPLAY
# (see clean_name below), so the sleepiness tables, the BBL policy functions and the
# descriptive tables cannot state different units for the same variable.
_BASE_LABELS = {
    'nr_lagged_dep': 'Lagged Deposits',
    'gdp_per_capita': 'GDP \\textit{per capita}',
    'cadunico_families_per1000': 'CadUnico Families',
    'fraction_65plus': 'Fraction 65+',
    'fraction_young': 'Fraction Young',
    'risk_free_qoq_lag': 'Lagged Selic Rate',
    # ANATEL "Acessos em Telefonia Movel": every active mobile-telephony access (2G-5G),
    # end-of-quarter stock per 100 inhabitants (scrape_anatel_mobile.py) -- not fixed broadband,
    # which is why it exceeds 100 in the median market. Full label in _FULL_LABELS below.
    'connections_per100': 'Mobile Lines',
    'pix_exists': 'Pix Available',
    'const': 'Constant', 'constant': 'Constant',
    'tax_cost_ratio_lag': 'Tax Cost Ratio ($t-1$)',
    'personnel_cost_ratio_lag': 'Personnel Cost Ratio ($t-1$)',
    'admin_cost_ratio_lag': 'Admin Cost Ratio ($t-1$)',
    'indice_basileia_lag': 'Basel Index ($t-1$)',
    'lci_lca_ratio_lag': 'LCI/LCA Ratio ($t-1$)',
    'wholesale_ratio_lag': 'Wholesale Ratio ($t-1$)',
    'leave_one_out_mean_spread': 'Leave-out Mean Spread',
    'v_hat_x_lagged_dep': 'CF: $\\hat{v} \\times$ Lagged Deposits',
    'time_trend': 'Time Trend',
    'gdp_growth_yoy': 'GDP Growth',
    'pix_users_pf_per1000': 'Pix Users',
    'branches_per1000': 'Branches',
    'estban_rival_branches_lag': 'Rival Branches ($t-1$)',
}

# Rows whose label must NOT carry a state-block unit: they are not state variables.
_NO_UNIT = {'nr_lagged_dep', 'const', 'constant', 'v_hat_x_lagged_dep'}

# Rows printed with the registry's unit spelled out, in the wording the demand tables use for
# the same variable. Same unit as utils/state_transform.DISPLAY ('per 100 inhab.').
_FULL_LABELS = {'connections_per100': 'Mobile Lines (per 100 inhabitants)'}


def clean_name(v, with_unit=True):
    v = str(v).replace('interaction_', '')
    if with_unit and v in _FULL_LABELS:
        return _FULL_LABELS[v]
    base = _BASE_LABELS.get(v)
    if base is None:
        return v.replace('_', '\\_')
    return base if (not with_unit or v in _NO_UNIT) else _st.label_with_unit(base, v)


# SECOND-STAGE APPENDIX LAYOUT, shared by sleep_export_e1/_e2 and this module. Every data
# cell's second line is a bias-corrected interval, which at four decimals plus a dagger is wider
# than a quarter of what a 0.34\textwidth label column leaves (up to 17pt overfull on the E4
# Macro panel). The label column is narrowed instead, and each label spans its coefficient AND
# interval rows via \multirow[t]{2}, so a label that wraps uses the interval row rather than
# opening a gap between a coefficient and its interval. The width is passed explicitly because
# multirow's `=` misjudges it under V_Main's \doublespacing (see sleep_export_spec12_compare).
SS_LABEL_W = r"0.22\textwidth"
SS_TABCOLSEP = "2pt"


def ss_label_cell(label):
    return rf"\multirow[t]{{2}}{{{SS_LABEL_W}}}{{\raggedright {label}}}"


def ss_colspec():
    return (r"\begin{xltabular}{\textwidth}{>{\raggedright\arraybackslash}p{" + SS_LABEL_W
            + r"} *{4}{>{\centering\arraybackslash}X}}")


def disp(v, lhs=None):
    """Display multiplier for a coefficient AND its standard error. Never applied to
    t-statistics, p-values or stars, which are invariant to a change of units."""
    return _st.display_mult(v, lhs=_st.PHI_DISPLAY if lhs is None else lhs)


# Appendix caption = stage + a reference to the estimation strategy enumerated in V_Main
# (Section ref{sec:empirical:sleep}); no ad-hoc strategy names. An estimator with no
# enumerate label falls back to a plain (Est. N) tag.
EST_LABEL = {3: "single_idx", 4: "single_idx_time"}

# FIRST STAGE: ONE TABLE PER ROUTINE, est{N}_first_stage_table.tex.
#
# The first stage is the deposit-spread (price) equation and does not depend on the
# second-stage link, so routines sharing a sample AND a control set produce numerically
# identical coefficients (verified on the saved pickles, spec 'IV_CostShifters x Macro'):
#     est2 == est3           (10 coefficients, no Time block)
#     est4                   (11 coefficients: the Time block adds one first-stage control)
#     est1                    stands alone (the pooled sample less the digital-bank D firms)
# The tables are nonetheless emitted per routine rather than shared. Sharing saved a page but
# forced a reader to hold the mapping in their head, and it put a Base panel that exists only
# for the linear estimator into a table captioned as covering a link routine too -- E2 carries
# Base/Macro/Tech while the link routines have no Base specification (their index excludes the
# constant). One table per routine states its own panels and its own strategy reference.
TIME_ESTS = {4}


def _strategy_caption(stage, est_num):
    lbl = EST_LABEL.get(est_num)
    ref = rf"\ref{{estimation:{lbl}}}" if lbl else rf"(Est.\ {est_num})"
    return rf"{stage} --- Estimation Strategy~{ref}"


def _panels_for(est_num):
    # E3/E4 (single-index) drop the Base panel: their index excludes the
    # constant, so a constant-only Base has no index.
    return ['Macro', 'Tech'] if est_num >= 3 else ['Base', 'Macro', 'Tech']


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
    caption = _strategy_caption("First Stage --- Deposit Spread on Instruments", est_num)
    label = f"tab:est{est_num}_first_stage"
    notes = (
        r"\footnotesize \textit{Notes:} Standard errors (WCB at the "
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
        r"\setstretch{1.0}",
        r"\footnotesize",
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
                    m = disp(var, lhs=_st.SPREAD_DISPLAY)   # LHS here is the spread, not phi
                    c, se = c * m, se * m
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


AME_CI_TOKEN = "%%AME_CI_NOTE%%"


# The band lookup is utils.sleep_links.band_row -- one reader for the single-index and linear
# families alike, since both bands come from `_ame_band_from_draws` with the same columns.
# Kept as a thin alias so this module's call sites read as before.
from utils.sleep_links import band_row as _ame_band_row  # noqa: E402


def _ame_ci_note(ci_cols, results_dict, est_num):
    """LaTeX sentence for a panel in which some columns print an interval and others an SE."""
    if not ci_cols:
        return ""
    B = scheme = None
    for entry in results_dict.values():
        r = entry.get("second_stage") if isinstance(entry, dict) else None
        m = getattr(r, "ame_2s_meta", None)
        if m:
            B, scheme = m.get("B"), m.get("scheme")
            break
    b_s = (f" $B={B}$ {scheme} draws," if B else "")
    return (r"The " + ", ".join(sorted(ci_cols)) + r" column(s) report a 95\% bias-corrected "
            r"percentile interval in brackets, not a standard error: their inference is a "
            r"TWO-STAGE WCB in which the index direction is re-solved and "
            r"the link re-profiled at every draw," + b_s + r" and the reported object is a "
            r"tangent-cone interval rather than a Wald statistic, because the link's shape "
            r"constraints are active at the estimate. "
            + _notes.reversed_draws_note(
                e.get("second_stage") for e in results_dict.values() if isinstance(e, dict)))


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
        r"\footnotesize \textit{Notes:} " + _notes.second_stage_note(columns="specifications")
    )

    def _get_res(ek, p):
        entry = results_dict.get(f"{ek} x {p}")
        return entry.get('second_stage') if isinstance(entry, dict) else None

    all_vars = ['constant', 'pix_exists', 'risk_free_qoq_lag', 'gdp_per_capita',
                'cadunico_families_per1000', 'fraction_65plus', 'fraction_young',
                'connections_per100', 'time_trend', 'gdp_growth_yoy', 'v_hat_x_lagged_dep']
    _nat_schemes = set()   # what select_se ACTUALLY returned on the national rows
    _sen.reset_ame_se_realised()   # and which AME variance it actually delivered
    _ci_cols = set()       # columns whose second line is an interval rather than an SE
    _n_se_cells = 0        # cells that printed an SE: the two-stage band exists for spec 12 only
    p0, l0 = panels[0], panel_letters[0]
    est_nums_0 = [(el, ss_spec_numbers[(p0, ek)]) for ek, el in estimators]
    # \begingroup keeps the narrow \tabcolsep local, so the paper's own setting is back in
    # force for whatever table follows.
    lines = [
        r"\begingroup",
        r"\setstretch{1.0}",
        r"\footnotesize",
        rf"\setlength{{\tabcolsep}}{{{SS_TABCOLSEP}}}",
        ss_colspec(),
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
            for ek, _ek_label in estimators:
                res = _get_res(ek, panel)
                var = vshort
                if vshort in ('const', 'constant') and res is not None and 'nr_lagged_dep' in res.params:
                    var = 'nr_lagged_dep'
                elif res is not None and var not in res.params and f"interaction_{var}" in res.params:
                    var = f"interaction_{var}"
                if res is not None and var in res.params:
                    has_val = True
                    c = res.params[var]
                    # National rows (Pix, Selic) lead with the wider time-robust SE; every other
                    # row keeps the conglomerate bootstrap. See utils/se_national.select_se.
                    se, pval, scheme = _sen.select_se(res, var)
                    if _sen.is_national(var):
                        _nat_schemes.add(scheme)
                    # The display multiplier was MISSING here (fixed 2026-07-30): these E3-E6
                    # tables printed raw phi units while export_1/export_2 printed percentage
                    # points, so Pix / GDP per capita / CadUnico were 100x too small and not
                    # comparable with the linear columns next to them. pval and stars are
                    # unit-invariant and are deliberately left alone.
                    m = disp(vshort)
                    c, se = c * m, se * m
                    mark = f"^{{{_sen.SE_MARK}}}" if scheme != "congl" else ""
                    # Where a two-stage AME bootstrap is attached, the second line is its
                    # bias-corrected interval instead of the standard error, with the display
                    # multiplier on both endpoints and stars from that same interval. A cell with
                    # no attached band prints its SE, and the note opening then switches to the
                    # mixed wording (see _bands below).
                    _bd = _ame_band_row(res, var)
                    if _bd is not None:
                        _ci_cols.add(_ek_label)
                        coef_strs.append(f"${c:.4f}^{{{_bd[2]}}}$")
                        se_strs.append(f"$[{_bd[0]*m:.4f}, {_bd[1]*m:.4f}]{mark}$")
                    else:
                        _n_se_cells += 1
                        coef_strs.append(f"${c:.4f}^{{{stars(pval)}}}$")
                        se_strs.append(f"$({se:.4f}){mark}$")
                else:
                    coef_strs.append(""); se_strs.append("")
            if has_val:
                lines.append(f"    {ss_label_cell(clean_name(vshort))} & "
                             + " & ".join(coef_strs) + r" \\*")
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
    lines += [r"\end{xltabular}", r"\endgroup", r"\doublespacing"]
    # The opening describes what the cells printed. A column label is not enough: the band is
    # attached to the spec-12 fit only, so the same Hausman column prints an interval in one
    # panel and an SE in another. Count cells.
    _bands = (True if not _n_se_cells else (False if not _ci_cols else "mixed"))
    return "\n".join(lines).replace(
        _notes.OPEN_TOKEN, _notes.note_open(_bands) + (_notes.reversed_draws_note(
            e.get("second_stage") for e in results_dict.values() if isinstance(e, dict))
            if _bands else "")).replace(
        _sen.NOTE_TOKEN, _sen.national_note(_nat_schemes, dk_bracket=False)).replace(
        AME_CI_TOKEN, _ame_ci_note(_ci_cols, results_dict, est_num)).replace(
        _sen.AME_SE_TOKEN, _sen.ame_se_note())


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
    out_dir = _paths_mod.est_dir(est_num)
    results_pickle = out_dir / "estimation_results.pkl"
    print("=" * 70); print(f" EXPORT EST{est_num} SLEEP RESULTS"); print("=" * 70)
    # A missing pickle is a hard failure, not a skip: without it this routine writes no
    # table at all, and a silent return leaves the orchestrator reporting success over a
    # step that produced nothing.
    if not results_pickle.exists():
        print(f"Results not found at {results_pickle}. "
              f"Run sleep_est_single.py --est {est_num} first.")
        sys.exit(1)
    with open(results_pickle, 'rb') as fh:
        results_dict = pickle.load(fh)

    ss_frag = build_second_stage_table(results_dict, est_num)
    ss_name = f"est{est_num}_second_stage_table.tex"
    (TEX_OUT_DIR / ss_name).write_text(ss_frag + "\n", encoding="utf-8")

    # First stage: every routine writes its own, alongside its second stage.
    fs_frag = build_first_stage_table(results_dict, est_num)
    fs_name = f"est{est_num}_first_stage_table.tex"
    (TEX_OUT_DIR / fs_name).write_text(fs_frag + "\n", encoding="utf-8")
    print(f" - First-stage table written ({fs_name})")
    print(f" - Fragments written to {_DRAFTS_DIR}")

    fs_section = r"\section*{First Stage}" + "\n" + rf"\input{{{fs_name}}}" + "\n\n"
    tex_doc = (
        _STANDALONE_PREAMBLE + r"\begin{document}" + "\n"
        + rf"\title{{Sleep Estimation Results --- {title}}}" + "\n"
        + r"\author{Autogenerated}" + "\n" + r"\date{\today}" + "\n" + r"\maketitle" + "\n\n"
        + fs_section
        + r"\section*{Second Stage}" + "\n" + rf"\input{{{ss_name}}}" + "\n"
        + r"\end{document}" + "\n"
    )
    wrapper_name = f"est{est_num}_sleep_results.tex"
    (TEX_OUT_DIR / wrapper_name).write_text(tex_doc, encoding="utf-8")
    print(f" - Standalone wrapper written to {_DRAFTS_DIR}")

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


# ── Config-driven CLI for E3/E4 tables (E1/E2 have their own export scripts) ──
#  Standalone-preview title per estimator; captions/notes are derived from est_num.
#  Run:  python sleep_export_link.py --est N
EXPORT_CFG = {
    3: r"E3: Pooled Single-Index (nonparametric link)",
    4: r"E4: Pooled Single-Index + Time block",
}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Export E3/E4 sleep tables (config-driven).")
    p.add_argument("--est", type=int, required=True, choices=sorted(EXPORT_CFG),
                   help="Estimator id 3 or 4 (E1/E2 have their own export scripts)")
    a = p.parse_args()
    export_link_results(a.est, EXPORT_CFG[a.est])
