"""
report_entry_paths_estban.py
============================
Writes entry_paths_estban_report.tex next to V_Main.tex and compiles it to PDF.

Every number in the report is read from the outputs of sleep_ident_entry_dynamics.py
(DIAG_PHI_SEPARATION, with the previous vintage in PREV_DIR), diagnostics/diag_entry_dynamics_estban.py
(DIAG_ENTRY_ESTBAN), the demand-preparation parquet and the code itself (the estimator's grid,
the type-4 rate ceiling); none is typed here.  V_Main.aux supplies the figure's number and page.
pdflatex runs twice in a build folder and only the PDF is copied back, so the Drafts folder
receives the .tex and the .pdf.

Usage:  python diagnostics/report_entry_paths_estban.py [--build-dir DIR] [--no-compile]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/ and pipeline modules
try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except Exception:
    pass

try:                                          # Windows console defaults to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import datetime as dt
import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile

import pandas as pd

from utils import paths
import sleep_ident_entry_dynamics as d6

D6_DIR = d6.OUT_DIR
PREV_DIR = D6_DIR / "_D6_PREWINDOW_20261012"     # the figure's outputs before the window rule
EST_DIR = paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_ENTRY_ESTBAN"
DRAFTS = paths.drafts_dir()
STEM = "entry_paths_estban_report"
FIG_LABEL = "fig:phi_entry_dyn_fixed_phi"

FLAG_TEXT = {
    "f_trunc": "Present in the panel's first four quarters",
    "f_mkt_young": "Market observed for fewer than four quarters at entry",
    "f_short": "Fewer than eight quarters observed after entry",
    "f_gappy": "A quarter without deposits inside the path",
    "f_jump": "Share moves by more than 0.150 in one quarter",
    "f_big_entry": "Share above 0.100 at entry",
    "f_churn": "Same CNPJ root already in the market",
    "f_branch": "Deposits two or more quarters before the first branch",
    "f_window": "Entry before the sample starts",
}


# ----------------------------------------------------------------------------- formatting
def f3(x: float) -> str:
    s = f"{float(x):.3f}"
    return "0.000" if s == "-0.000" else s


def ni(x) -> str:
    return f"{int(round(float(x))):,}"


def yq(q: int) -> str:
    return f"{q // 4}Q{q % 4 + 1}"


def day(p: Path) -> str:
    t = dt.datetime.fromtimestamp(p.stat().st_mtime)
    return f"{t:%B} {t.day}, {t.year}"


def sci(x: float) -> str:
    if x == 0:
        return "machine precision"
    m, e = f"{x:.0e}".split("e")
    return f"${m}\\times 10^{{{int(e)}}}$"


def span(s: pd.Series) -> str:
    """'a to b' over the displayed values, or one value when they all display alike."""
    lo, hi = f3(s.min()), f3(s.max())
    return lo if lo == hi else f"{lo} to {hi}"


def ci(lo, hi) -> str:
    return f"[{f3(lo)}, {f3(hi)}]"


def pct(x: float) -> str:
    return f3(100 * float(x))


# ----------------------------------------------------------------------------- inputs
def figure_ref() -> tuple[str, str]:
    aux = (DRAFTS / "V_Main.aux").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\\newlabel\{" + re.escape(FIG_LABEL) + r"\}\{\{(\d+)\}\{(\d+)\}", aux)
    if not m:
        raise SystemExit(f"{FIG_LABEL} not found in V_Main.aux; compile V_Main first")
    return m.group(1), m.group(2)


def waterfall(ev: pd.DataFrame) -> tuple[int, list[tuple[str, int]], int]:
    flags = [c for c in FLAG_TEXT if c in ev.columns]
    alive = pd.Series(True, index=ev.index)
    steps = []
    for f in flags:
        hit = ev[f].astype(bool)
        steps.append((f, int((alive & hit).sum())))
        alive &= ~hit
    return len(ev), steps, int(alive.sum())


def only_branch(ev: pd.DataFrame) -> pd.DataFrame:
    others = [c for c in FLAG_TEXT if c in ev.columns and c != "f_branch"]
    return ev[~ev[others].astype(bool).any(axis=1) & ev["f_branch"].astype(bool)]


def events(folder: Path, kind: str) -> pd.DataFrame:
    ev = pd.read_csv(folder / "d6_entry_events.csv", low_memory=False)
    return ev[ev["kind"] == kind].copy()


def g_sources(ev: pd.DataFrame, H: int) -> dict:
    k = ev[ev["keep"]]
    tot = len(k) * (H + 1)
    return {s: float(k[c].sum()) / tot for s, c in
            (("own", "n_g_cells_own"), ("entity", "n_g_fallback_entity"),
             ("national", "n_g_fallback_national"), ("carried", "n_g_extrapolated"))}


def g_frame() -> pd.DataFrame:
    d = pd.read_parquet(paths.demand_parquet_dir() / "demand_2_spec_12.parquet",
                        columns=["CodConglomeradoPrudencial", "mca_code", "deposit_type", "year",
                                 "quarter", "gross_return_lag", "deposit_balance"])
    return d[d["deposit_type"].astype(int).isin(d6.G_TYPES)]


def g_by_type(d: pd.DataFrame) -> pd.DataFrame:
    out = d.groupby("deposit_type").agg(med=("gross_return_lag", "median"),
                                        bal=("deposit_balance", "sum"))
    out["share"] = out["bal"] / out["bal"].sum()
    return out


def own_cell_at_entry(d: pd.DataFrame, ev: pd.DataFrame) -> float:
    """Share of kept entrants with a cell of their own in the entry quarter."""
    keys = set(zip(d["CodConglomeradoPrudencial"].astype(str), d["mca_code"].astype(str),
                   d["year"].astype(int) * 4 + d["quarter"].astype(int) - 1))
    k = ev[ev["keep"]]
    hits = [(str(c), str(m), int(q)) in keys for c, m, q in zip(k["congl"], k["mkt"], k["q_entry"])]
    return sum(hits) / len(hits)


def a4_ceiling() -> str:
    """The type-4 rate ceiling, read from panel_deposit_rates.py's default (importing that
    pipeline module would run its setup)."""
    src = (Path(__file__).resolve().parents[1] / "panel_deposit_rates.py").read_text(encoding="utf-8")
    m = re.search(r'RATE_A4_MAX_MULT", ([\d.]+)\)', src)
    return f"{float(m.group(1)):g}"


# ----------------------------------------------------------------------------- tables
def table_screens(ev6, evR) -> str:
    c6, s6, k6 = waterfall(ev6)
    cR, sR, kR = waterfall(evR)
    dR = dict(sR)
    neg = lambda n: "0" if n == 0 else f"$-${ni(n)}"
    rows = [f"Candidate entries & {ni(c6)} & {ni(cR)} \\\\", "\\midrule"]
    for f, n in s6:
        rows.append(f"{FLAG_TEXT[f]} & {neg(n)} & {neg(dR.get(f, 0))} \\\\")
    rows += ["\\midrule", f"Entries kept & {ni(k6)} & {ni(kR)} \\\\"]
    return "\n".join(rows)


def table_phi(imp6, impR, impP) -> str:
    def row(label, r):
        return (f"{label} & {ni(r['n_events'])} & {f3(r['phi_entry'])} & "
                f"{ci(r['lo'], r['hi'])} & {f3(r['phi_g'])} & {f3(r['phi_pooled_abs'])} \\\\")
    d = imp6.loc["D"]
    return "\n".join([
        "\\multicolumn{6}{l}{\\textit{B firms, local markets}} \\\\",
        row("\\quad Paper: market panel, per-entrant $g$", imp6.loc["B"]),
        row("\\quad Same entries, fixed $g$ (control)", impR.loc["panel_fixed_g"]),
        row("\\quad Raw ESTBAN, fixed $g$", impR.loc["window"]),
        row("\\quad Paper, previous sample rule", impP.loc["B"]),
        "\\addlinespace",
        "\\multicolumn{6}{l}{\\textit{D firms, national market}} \\\\",
        (f"\\quad Paper: market panel, per-entrant $g$ & {ni(d['n_events'])} & "
         f"{f3(d['phi_entry'])} & [{f3(d['lo'])}, top of grid] & {f3(d['phi_g'])} & "
         f"{f3(d['phi_pooled_abs'])} \\\\"),
    ])


def table_paths(paths6, pathsR, H) -> tuple[str, float]:
    def col(df, norm):
        return df[df["norm"] == norm].set_index("h")
    p6 = paths6[paths6["kind"] == "B"]
    pR = pathsR[pathsR["variant"] == "window"]
    a6, b6, aR, bR = col(p6, "main"), col(p6, "alt"), col(pR, "main"), col(pR, "alt")
    rows = [f"{h} & {f3(a6.loc[h, 'median'])} & {f3(aR.loc[h, 'median'])} & "
            f"{f3(b6.loc[h, 'median'])} & {f3(bR.loc[h, 'median'])} \\\\" for h in range(H + 1)]
    rows += ["\\midrule", f"Entries & {ni(a6['n'].iloc[0])} & {ni(aR['n'].iloc[0])} & "
                          f"{ni(b6['n'].iloc[0])} & {ni(bR['n'].iloc[0])} \\\\"]
    dmax = float(max((aR["median"] - a6["median"]).abs().max(),
                     (bR["median"] - b6["median"]).abs().max()))
    return "\n".join(rows), dmax


def table_cohorts(evP, ev6, evR) -> str:
    year = lambda df: (df["q_entry"] // 4).astype(int)
    kp = year(evP[evP["keep"]]).value_counts()
    bp = year(only_branch(evP)).value_counts()
    k6 = year(ev6[ev6["keep"]]).value_counts()
    kr = year(evR[evR["keep"]]).value_counts()
    years = sorted(set(kp.index) | set(bp.index) | set(k6.index) | set(kr.index))
    rows = [f"{y} & {ni(kp.get(y, 0))} & {ni(bp.get(y, 0))} & {ni(k6.get(y, 0))} & "
            f"{ni(kr.get(y, 0))} \\\\" for y in years]
    rows += ["\\midrule", f"Total & {ni(kp.sum())} & {ni(bp.sum())} & {ni(k6.sum())} & "
             f"{ni(kr.sum())} \\\\"]
    return "\n".join(rows)


def table_continuity(cont) -> str:
    return "\n".join(f"{r.quarter} & {ni(r.raw_all / 1e9)} & {ni(r.panel / 1e9)} & "
                     f"{f3(r.panel_over_raw_with_mun)} \\\\" for r in cont.itertuples())


# ----------------------------------------------------------------------------- document
TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[letterpaper, margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage[font=small, labelfont=bf]{caption}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\setlist{itemsep=2pt, topsep=4pt}
\setlength{\emergencystretch}{3em}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.8}
\makeatletter\setlength{\@fptop}{0pt}\makeatother
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcommand{\tnote}[1]{\par\vspace{3pt}\begin{minipage}{\linewidth}\footnotesize #1\end{minipage}}

\title{Entry Paths and the Entry-Implied Sleeping Share:\\[2pt]
\large Inputs, the Accrual Factor, the Estimator, and a Check on Raw ESTBAN Data}
\author{Deposit Competition project, research note}
\date{@@today@@}

\begin{document}
\maketitle

\section*{Summary}
\begin{itemize}
\item Figure @@fignum@@ of the current draft compares the median share path of entrants with the path the sleepiness model implies. Its data paths and the entry-implied sleeping share use deposit balances and a constructed deposit rate only. The one estimated input is the level of $\hat\phi$ behind the two model curves, taken from the sleepiness stage. Nothing comes from the demand (logit or BLP) stage.
\item The figure's $\mathrm{B}$-firm share is a conditional share: the entrant's ESTBAN deposits over the ESTBAN deposits of all branch-network institutions in the market. $\mathrm{D}$ firms are booked nationally and enter neither the numerator nor the denominator.
\item Entries are now dated inside the estimation sample, @@win_lo@@ to @@win_hi@@, and share paths are read through @@win_hi@@. The $\mathrm{B}$-firm paths imply a sleeping share of @@phi_paper@@ (95\% CI @@ci_paper@@, @@n_paper@@ entries), against @@phi_prev@@ (@@ci_prev@@, @@n_prev@@ entries) under the previous rule.
\item The branch-timing screen now judges an entry only where the branch data cover it, and drops @@n_branch_new@@. Under the previous rule it dropped @@n_branch_prev@@, of which @@n_branch_early@@ were entries from 2014Q1 to 2015Q3 that had no branch data at all.
\item The accrual factor $g$ is constructed, not estimated: one plus last quarter's deposit rate, averaged over the entrant's own mix of deposit types (Section~\ref{sec:g}).
\item Rebuilt from the raw ESTBAN files under the same rule, the median paths stay within @@dmax@@ of the paper's and imply @@phi_raw@@ (@@ci_raw@@, @@n_raw@@ entries).
\item The market panel still holds pre-2016 ESTBAN balances at about twice their raw level, a problem known since the demand repair of October 5. Under the new rule no pre-2016 quarter enters any share path, so the figure is unaffected.
\end{itemize}

\section{The figure and where it comes from}

Figure @@fignum@@ of the current draft (page @@figpage@@, label \texttt{fig:phi\_entry\_dyn\_fixed\_phi}) follows Figure~3 of Egan, Horta\c{c}su, Kaplan, Sunderam, and Yao (2025). If every depositor re-optimized each quarter, an entrant's share would jump to its long-run level at once. If a fraction $\phi$ of balances stays where it is each quarter, the share builds up gradually, at a speed set by $\phi$. An entry is the first quarter in which an institution holds deposits in a market: a Minimum Comparable Area (MCA) for brick-and-mortar ($\mathrm{B}$) firms, the national market for digital ($\mathrm{D}$) firms. Each path runs over the @@H1@@ quarters $h = 0, \dots, @@H@@$ after entry.

The figure, and the entry-implied share the text quotes (macro \texttt{phi.entry.point}), come from \texttt{sleep\_ident\_entry\_dynamics.py}. The current outputs were computed on @@date_paths@@ from the market panel built on @@date_panel@@.

\paragraph{Sample rule.} Entries are dated inside the estimation window of \texttt{utils/window.py}, @@win_lo@@ to @@win_hi@@, the bounds every estimation stage uses, and share paths are read only through @@win_hi@@. The panel's earlier quarters, from 2013Q1, enter only to show that an entrant had no deposits in the market before its entry quarter. An entry needs eight observed quarters after it, and is plotted only if at least one of quarters @@plat_lo@@ to @@plat_hi@@ is observed, so the latest plotted entries date from @@last_plotted@@. Under the previous rule, entries ran from 2014 and paths through the panel's last quarter, @@prev_qmax@@.

\section{Inputs}

Table~\ref{tab:inputs} lists what each element of the figure draws on.

\begin{table}[htbp]
\centering
\caption{Inputs to Figure @@fignum@@}\label{tab:inputs}
\begin{tabularx}{\linewidth}{@{}P{3.6cm}LP{2.6cm}@{}}
\toprule
Element & Source & Estimated? \\
\midrule
Entrant share paths (dots and bands) & Market panel: demand, savings and time deposits, @@win_lo@@ to @@win_hi@@ & No \\
Entry screens & Market panel; ESTBAN branch counts & No \\
Accrual $g_{e,h}$ & One plus last quarter's deposit rate, over the entrant's own deposit mix (Section~\ref{sec:g}) & No (constructed) \\
Model curves & Mean national $\hat\phi_t$ over @@nq@@ quarters: @@phi_E3@@ in routine (III), @@phi_E4@@ in routine (IV) & Yes: sleepiness stage \\
Demand estimates & Not used & --- \\
\bottomrule
\end{tabularx}
\tnote{\textit{Notes:} Prepaid accounts (type 5) are not in ESTBAN and are not part of any path.}
\end{table}

\paragraph{What the shares condition on.}
For a $\mathrm{B}$ entrant $e$ in market $m$, the share at horizon $h$ is
\begin{equation*}
s_{e,h} = \frac{D_{e,m,t_e+h}}{\sum_{j \in \mathcal{B}_m} D_{j,m,t_e+h}},
\end{equation*}
where $D$ is the sum of demand, savings and time deposits and $\mathcal{B}_m$ is every institution with deposits booked in $m$. Every local row of the market panel comes from ESTBAN (@@n_local@@ rows), and every $\mathrm{D}$ row is booked at the national code, so $s_{e,h}$ is the share conditional on banking with a $\mathrm{B}$ firm. The @@n_booked@@ conglomerates that report to ESTBAN but are classified $\mathrm{D}$, among them Banco Inter, BTG Pactual and C6, are moved to national booking and leave every local denominator. The $\mathrm{D}$-firm share is the same object at the national level: the entrant's deposits over those of all $\mathrm{D}$ firms.

\paragraph{Screens.}
Each candidate entry passes the screens of Table~\ref{tab:screens}, applied in the order listed, so each count is the drop given the screens above it. Most candidates are incumbents present from the start of the panel, which the first screen removes. Of the @@kept_paper@@ $\mathrm{B}$ entries kept, @@n_paper@@ are plotted; the other @@n_cens_B@@ entered in @@cens_B@@ and are not observed in quarters @@plat_lo@@ to @@plat_hi@@.

\begin{table}[htbp]
\centering
\caption{Screens on candidate $\mathrm{B}$-firm entries}\label{tab:screens}
\begin{tabularx}{\linewidth}{@{}Lrr@{}}
\toprule
 & Paper (market panel) & Raw ESTBAN \\
\midrule
@@table_screens@@
\bottomrule
\end{tabularx}
\tnote{\textit{Notes:} The paper's firm is the prudential conglomerate and the raw-ESTBAN firm is the institution (8-digit CNPJ), so the candidate counts differ. ``Raw ESTBAN'' is the check of Section~\ref{sec:raw}. The branch screen judges entries from the first quarter its branch data cover (@@branch_cover@@ in the paper) and also flags an entrant that never shows a branch.}
\end{table}

\section{The accrual factor \texorpdfstring{$g$}{g}}\label{sec:g}

\paragraph{Definition.} For entrant $e$ at horizon $h$, in calendar quarter $t = t_e + h$,
\begin{equation*}
g_{e,h} = \sum_{k \in \{1,2,4\}} w_{e,k,t}\, \bigl(1 + r_{j,k,t-1}\bigr), \qquad r_{j,k,t-1} = r^f_{t-1} - \mathit{spread}_{j,k,t-1},
\end{equation*}
where $k$ is the deposit type (1 demand, 2 savings, 4 time), $w_{e,k,t}$ is the entrant's share of its own balances in type $k$ in that market and quarter, $j$ is the entrant's conglomerate, and $r^f$ is the Selic rate compounded over the quarter. The term $1 + r_{j,k,t-1}$ is the column \texttt{gross\_return\_lag} of the demand-preparation files, the same factor that multiplies last quarter's balances in the sleepiness stage.

\paragraph{The deposit rates.} Nothing here is estimated; the rates are built in \texttt{panel\_deposit\_rates.py}.
\begin{itemize}
\item Demand deposits pay no interest, so the factor is one.
\item Savings deposits pay the regulated rate, the same at every bank: the reference rate (TR) plus 0.5 percent a month when the Selic target is above 8.5 percent, and TR plus 70 percent of the target otherwise, compounded over the quarter.
\item Time deposits pay the conglomerate's implicit rate from its COSIF statements, interest expense over the previous balance. Where that rate is missing, or above @@a4_mult@@ times the Selic, the CDI rate replaces it.
\end{itemize}

\paragraph{Why last quarter's rate.} The law of motion carries last quarter's balances forward at $\phi g$: a balance held through quarter $t$ earns the rate in force when the quarter starts. The entry paths use the same timing as the sleepiness estimates.

\paragraph{Weights and fallbacks.} Where the entrant has no cell of its own in a quarter, $g$ falls back to its conglomerate's deposit-weighted average across markets in that quarter, then to the national median, and at the edges of the sample the nearest value is carried flat. The entry quarter usually has no cell of its own, because the first quarter of a cell has no lagged rate: @@no_own0_B@@ percent of $\mathrm{B}$ entrants lack one. Over the $\mathrm{B}$ entrants' quarters, @@gB_own@@ percent use the entrant's own cell, @@gB_ent@@ percent the conglomerate average, @@gB_nat@@ percent the national median, and @@gB_car@@ percent a carried value. $\mathrm{D}$ entrants hold a single market, so they have no conglomerate average to fall back on: @@gD_nat@@ percent of their quarters use the national median.

\paragraph{Magnitudes.} In the demand sample the median factor is @@g_t1@@ for demand, @@g_t2@@ for savings and @@g_t4@@ for time deposits, which hold @@w_t1@@, @@w_t2@@ and @@w_t4@@ percent of balances. Averaged over each entrant's path, $g$ has median @@ge_med_B@@ for $\mathrm{B}$ entrants (10th to 90th percentile @@ge_p10_B@@ to @@ge_p90_B@@) and @@ge_med_D@@ for $\mathrm{D}$ entrants.

\paragraph{What $g$ does to $\hat\phi$.} The curvature of the path pins down the product $\phi g$, not $\phi$: @@phig_paper@@ for $\mathrm{B}$ entrants. The share $\phi$ is that product divided by the accrual, so the accrual assumption carries into its level. Holding $g$ at the sample median, @@g_fixed@@, for every entrant and quarter gives @@phi_control@@, a difference of @@accrual_effect@@ (Table~\ref{tab:phi}, control row).

\section{How the implied sleeping share is estimated}\label{sec:est}

\paragraph{Model path.}
With spreads held fixed, the entrant's deposits follow
\begin{equation*}
\mathrm{Dep}_{h} = \phi\, g_{h}\, \mathrm{Dep}_{h-1} + A, \qquad A = (1-\phi)\, M\, s,
\end{equation*}
where $A$ is the inflow from awake depositors: market balances $M$ times the entrant's choice probability $s$. Starting from zero before entry and setting $A = 1$, the level path is $\ell_0 = 1$ and $\ell_h = \phi\, g_{e,h}\, \ell_{h-1} + 1$; with a constant $g$ it is $\ell_h = \bigl(1-(\phi g)^{h+1}\bigr)/(1-\phi g)$.

\paragraph{Normalization.}
Data and model are normalized the same way, which cancels $A$:
\begin{equation*}
n_{e,h} = \frac{s_{e,h}-s_{e,0}}{\bar s_e - s_{e,0}}, \qquad
m_{e,h}(\phi) = \frac{\ell_{e,h}-\ell_{e,0}}{\bar\ell_e - \ell_{e,0}},
\end{equation*}
with $\bar s_e$ and $\bar \ell_e$ the means over $h = @@plat_list@@$. This is the left panel of the figure; the right panel plots $s_{e,h}/\bar s_e$.

\paragraph{Estimator.}
\begin{equation*}
\hat\phi = \arg\min_{\phi \in \Phi} \sum_{h=1}^{@@fit_hi@@} \Bigl[\operatorname*{med}_e m_{e,h}(\phi) - \operatorname*{med}_e n_{e,h}\Bigr]^2,
\end{equation*}
where $\Phi$ is a grid of @@n_grid@@ equally spaced points from @@grid_lo@@ to @@grid_hi@@ and each entrant's model path uses its own accrual path. Quarter 0 is zero for every path and quarters @@plat_lo@@ to @@plat_hi@@ average to one, so they carry no information. The interval resamples entries with replacement @@n_draws@@ times, each with its own accrual path, refits $\phi$ on each draw, and takes the 2.5th and 97.5th percentiles. For $\mathrm{B}$ firms this gives @@phi_paper@@ (95\% CI @@ci_paper@@) from @@n_paper@@ entries. Minimizing instead the summed absolute deviations over every entry and horizon gives @@phi_paper_pooled@@.

\paragraph{Two caveats.}
First, the model describes the entrant's deposit level while the data are shares; the normalization cancels the inflow level only if the market's total deposits are roughly flat over the @@H@@ quarters. Second, the fit uses only the left panel, which subtracts the entry share. In the right panel the model starts at @@model_alt_h0@@ at quarter 0 while the data start at @@data_alt_h0@@, so the first quarter after entry is slower than a constant inflow implies, and that gap does not enter $\hat\phi$.

\paragraph{D firms.}
The $\mathrm{D}$-firm paths give @@phi_D@@ from @@n_D@@ entries, with an interval whose upper end is the top of the grid. The paper quotes only the $\mathrm{B}$-firm value.

\section{Rebuilding the paths from raw ESTBAN}\label{sec:raw}

\paragraph{Construction.}
The check reads the BCB's monthly ESTBAN files directly: @@n_files@@ quarter-end months from @@q_first@@ to @@q_last@@, @@n_rows@@ institution $\times$ municipality rows. Deposits are the demand verbetes 401--419, savings (420) and time deposits (432), the same accounts as the panel's types 1, 2 and 4. They are summed to institution $\times$ MCA $\times$ quarter over @@n_inst@@ institutions and @@n_mca@@ MCAs. The municipality-to-MCA map is the panel's, the one input that is not ESTBAN; @@n_no_mca@@ rows, @@pct_no_mca@@ percent of deposits, fall outside it. The sample rule, screens, normalizations, bootstrap and estimator are the figure script's own, run unchanged. Table~\ref{tab:diff} lists what differs.

\begin{table}[htbp]
\centering
\caption{The raw-ESTBAN construction against the paper's}\label{tab:diff}
\begin{tabularx}{\linewidth}{@{}P{2.6cm}LL@{}}
\toprule
 & Paper & Raw ESTBAN \\
\midrule
Firm & Prudential conglomerate & Institution (8-digit CNPJ) \\
Denominator & $\mathrm{B}$-classified conglomerates in the MCA & Every ESTBAN reporter in the MCA, including the @@n_booked@@ booked nationally in the panel \\
Branch data & From \texttt{ESTBAN.csv}, from @@branch_cover@@ & From the raw files, from @@q_first@@ \\
Accrual $g$ & Each entrant's own path & Fixed at the median, @@g_fixed@@ \\
\bottomrule
\end{tabularx}
\end{table}

\paragraph{Results.}
Table~\ref{tab:phi} reports the implied sleeping share, Figure~\ref{fig:raw} overlays the raw-ESTBAN paths on the paper's, and Table~\ref{tab:paths} lists the medians. The raw-ESTBAN medians stay within @@dmax@@ of the paper's at every horizon in both normalizations. Two comparisons separate the sources of difference in $\hat\phi$:
\begin{itemize}
\item Accrual path: the paper's entries at a fixed accrual give @@phi_control@@, against @@phi_paper@@ with per-entrant accrual, a difference of @@accrual_effect@@.
\item Data source: raw ESTBAN at the same fixed accrual gives @@phi_raw@@ from @@n_raw@@ entries, against the control's @@phi_control@@ from @@n_control@@. The interval is @@w_raw@@ wide, against @@w_paper@@ in the paper.
\end{itemize}

\begin{table}[!htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Entry-implied sleeping share}\label{tab:phi}
\begin{tabular}{@{}lrcccc@{}}
\toprule
 & Entries & $\hat\phi$ & 95\% CI & $\hat\phi g$ & Absolute loss \\
\midrule
@@table_phi@@
\bottomrule
\end{tabular}
\tnote{\textit{Notes:} $\hat\phi$ minimizes the squared distance between the median model path and the median data path over quarters 1 to @@fit_hi@@ (Section~\ref{sec:est}); the interval resamples entries @@n_draws@@ times. ``Absolute loss'' instead minimizes summed absolute deviations over every entry and horizon. $\hat\phi g$ is evaluated at the median accrual of the entries. The fixed accrual is the sample median, @@g_fixed@@. The control row rebuilds the paper's entries from the market panel; its median path matches the published one to @@control_dev@@. The previous sample rule dated entries from 2014 and read paths through @@prev_qmax@@. The grid runs from @@grid_lo@@ to @@grid_hi@@.}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{@@fig_path@@}
\caption{Median normalized share paths of $\mathrm{B}$-firm entrants, raw ESTBAN (with the 95\% bootstrap interval of the median) and the paper's figure, both under the current sample rule. The model line is evaluated at the raw-ESTBAN implied share and the fixed accrual. Panel (a) is the normalization the fit uses; panel (b) divides by the plateau share, where instant sorting is the flat line at one.}\label{fig:raw}
\end{figure}

\begin{table}[!htbp]
\centering
\caption{Median normalized share path by quarter since entry}\label{tab:paths}
\begin{tabular}{@{}lcccc@{}}
\toprule
 & \multicolumn{2}{c}{(a) $(s_h - s_0)/(\bar s - s_0)$} & \multicolumn{2}{c}{(b) $s_h/\bar s$} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
$h$ & Paper & Raw ESTBAN & Paper & Raw ESTBAN \\
\midrule
@@table_paths@@
\bottomrule
\end{tabular}
\tnote{\textit{Notes:} Entries count the paths at quarter 0. The right panel keeps entries whose plateau share is within $10^{-6}$ of their entry share, which the left panel cannot normalize.}
\end{table}

\clearpage
\section{What changed, and what remains in the data}

\subsection{The sample rule and the branch screen}
The branch screen flags an entrant whose deposits appear two or more quarters before its first branch in the market. It reads the first branch from \texttt{estban\_own\_branches.csv}, built from \texttt{ESTBAN.csv}, which starts in @@branch_cover@@, while the panel's ESTBAN deposits start in 2013Q1. Under the previous rule every entry from 2014Q1 to 2015Q3 therefore had no recorded branch before 2016 and was dropped: @@n_branch_early@@ of the screen's @@n_branch_prev@@ drops. The screen now judges an entry only from the first quarter its branch data cover. ESTBAN lists an institution in a municipality only where it has a branch, so an entrant from that quarter on had no earlier branch there and is judged correctly; the screen now drops @@n_branch_new@@ (Table~\ref{tab:screens}). With entries dated from @@win_lo@@, every entry is covered. Table~\ref{tab:cohorts} shows the entries by year under both rules.

\begin{table}[htbp]
\centering
\caption{$\mathrm{B}$-firm entries kept, by year of entry}\label{tab:cohorts}
\begin{tabular}{@{}lrrrr@{}}
\toprule
 & \multicolumn{2}{c}{Previous rule} & \multicolumn{2}{c}{Current rule} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Year & Kept & Branch-screen drops & Paper & Raw ESTBAN \\
\midrule
@@table_cohorts@@
\bottomrule
\end{tabular}
\tnote{\textit{Notes:} ``Branch-screen drops'' counts entries that pass every other screen and fail only the branch screen. Kept entries include those not observed in the plateau quarters, which the figure does not plot.}
\end{table}

\subsection{Pre-2016 ESTBAN balances are doubled in the market panel}
\texttt{panel\_deposits.py} lists the raw ESTBAN files for 2013--2015 with two patterns, \texttt{*.CSV} and \texttt{*.csv}. File names are case-insensitive on Windows, so both patterns match every file and each file is read twice; the conglomerate $\times$ municipality $\times$ quarter sum doubles every pre-2016 balance. From 2016 the panel reads \texttt{ESTBAN.csv}, once. Table~\ref{tab:cont} shows the break: the panel's national ESTBAN deposits run at @@ratio_pre@@ times the raw files in 2015 and at @@ratio_post@@ times from 2016Q1. The problem was traced during the demand repair of October 5. The demand repair halves the 2015Q4 balance in its own repair file, and the one-line source fix was deferred to the next full rebuild; applied now, it would halve that balance a second time.

The figure does not depend on it. Within ESTBAN the doubling is uniform, so local shares are unchanged; and under the current rule no pre-2016 quarter enters any path, so the pre-2016 rows only date entries, which doubling cannot change. That second point also covers $\mathrm{D}$ firms, whose pre-2016 shares would be distorted: the $\mathrm{D}$ firms that report to ESTBAN carry doubled balances, while those sourced from IF.Data do not.

The panel's ESTBAN total after 2016 is about @@pct_post@@ percent below the raw files. This is not traced here: \texttt{ESTBAN.csv} matches the raw files, and raw rows without a municipality carry no deposits in these quarters, so the loss happens inside the panel build.

\begin{table}[htbp]
\centering
\caption{National ESTBAN deposits around the 2016 break (R\$ billion)}\label{tab:cont}
\begin{tabular}{@{}lrrc@{}}
\toprule
Quarter & Raw files & Market panel & Panel / raw \\
\midrule
@@table_cont@@
\bottomrule
\end{tabular}
\tnote{\textit{Notes:} Demand, savings and time deposits of every ESTBAN reporter, summed nationally. The market panel total includes the ESTBAN rows it moves to national booking.}
\end{table}

\section{Open items}
\begin{enumerate}
\item \textbf{V\_Main text.} The figure, its counts and the quoted share update through the macros. Two phrases of the caption are typed and no longer hold (the entrants ``of 2023Q3'', and the plural for the $\mathrm{D}$-firm entrants without a plateau path); replacement lines are supplied separately.
\item \textbf{Double read.} Apply the source fix in \texttt{panel\_deposits.py} at the next full rebuild, and remove the repair file's 2015Q4 halving in the same step.
\item \textbf{Post-2016 shortfall.} Trace the @@pct_post@@ percent gap between the panel's ESTBAN total and the raw files.
\item \textbf{D-firm accrual.} @@gD_nat@@ percent of $\mathrm{D}$ entrants' quarters use the national median $g$; the paper does not quote the $\mathrm{D}$-firm share.
\item \textbf{Reporting.} Whether to cite the raw-ESTBAN check in the paper.
\end{enumerate}

\appendix
\section{Reproduction}
From the code folder, with the project environment:
\begin{itemize}
\item \texttt{python sleep\_ident\_entry\_dynamics.py -{}-routine-curves 1 2 3 4} writes the paper's figure, and its numbers go to the folder \texttt{DIAG\_PHI\_SEPARATION}; \texttt{-{}-no-window-only} restores every panel quarter.
\item \texttt{python diagnostics/diag\_entry\_dynamics\_estban.py} reads the raw ESTBAN files, rebuilds the paths and the control row, and writes \texttt{ESTIMATION\_OUTPUT/DIAG\_ENTRY\_ESTBAN}. It changes no panel and no paper exhibit.
\item \texttt{python diagnostics/report\_entry\_paths\_estban.py} writes and compiles this note from those outputs; no number in it is typed by hand.
\end{itemize}

\section*{Reference}
\noindent\textsc{Egan, M. L., A. Horta\c{c}su, N. A. Kaplan, A. Sunderam, and V. Yao} (2025): ``Dynamic Competition for Sleepy Deposits,'' Working Paper 34267, National Bureau of Economic Research.

\end{document}
"""


def build() -> str:
    meta6 = json.loads((D6_DIR / "d6_meta.json").read_text(encoding="utf-8"))
    metaR = json.loads((EST_DIR / "estban_meta.json").read_text(encoding="utf-8"))
    imp6 = pd.read_csv(D6_DIR / "d6_implied_phi.csv").set_index("kind")
    impP = pd.read_csv(PREV_DIR / "d6_implied_phi.csv").set_index("kind")
    paths6 = pd.read_csv(D6_DIR / "d6_entry_paths.csv")
    ev6, evP = events(D6_DIR, "B"), events(PREV_DIR, "B")
    ev6D = events(D6_DIR, "D")
    impR = pd.read_csv(EST_DIR / "estban_implied_phi.csv").set_index("variant")
    pathsR = pd.read_csv(EST_DIR / "estban_entry_paths.csv")
    evR = pd.read_csv(EST_DIR / "estban_entry_events_window.csv", low_memory=False)
    cont = pd.read_csv(EST_DIR / "estban_continuity.csv")
    booked = pd.read_csv(EST_DIR / "estban_nationally_booked.csv")

    if not meta6.get("window_q"):
        raise SystemExit("d6_meta.json records no window: rerun sleep_ident_entry_dynamics.py")
    win_lo, win_hi = (int(q) for q in meta6["window_q"])
    H, plateau_w = int(meta6["horizon"]), list(meta6["plateau_w"])
    src = inspect.getsource(d6.implied_phi)
    gm = re.search(r"np\.linspace\(([\d.]+),\s*([\d.]+),\s*(\d+)\)", src)
    grid_lo, grid_hi, grid_n = float(gm.group(1)), float(gm.group(2)), int(gm.group(3))
    draw_cap = int(re.search(r"min\(boot,\s*(\d+)\)", src).group(1))
    g_fixed = float(metaR["g"])
    fignum, figpage = figure_ref()
    vint = {lab: (phi, s) for lab, phi, s in meta6["vintages"]}
    nq = re.search(r"over (\d+) quarters", vint["E3"][1]).group(1)

    b, d, bP = imp6.loc["B"], imp6.loc["D"], impP.loc["B"]
    raw, ctl = impR.loc["window"], impR.loc["panel_fixed_g"]
    paths_tex, dmax = table_paths(paths6, pathsR, H)
    alt6 = paths6[(paths6["kind"] == "B") & (paths6["norm"] == "alt")].set_index("h")["median"]
    model_alt_h0 = d6.model_curve(float(b["phi_entry"]), float(b["g_e_median"]), H, "alt",
                                  plateau_w)[0]
    pre = cont[cont["qidx"] < 2016 * 4]["panel_over_raw_with_mun"]
    post = cont[cont["qidx"] >= 2016 * 4]["panel_over_raw_with_mun"]
    n_local = pd.read_parquet(paths.PROCESSED / "market_panel.parquet",
                              columns=["CODMUN_IBGE"])["CODMUN_IBGE"]
    n_local = int((~n_local.isin(["0", "0.0"])).sum())
    branch_cover = int(pd.read_csv(d6.BRANCH_SIDECAR, usecols=["year", "quarter"])
                       .eval("year * 4 + quarter - 1").min())
    obP = only_branch(evP)
    _, steps6, _ = waterfall(ev6)
    n_branch_new = dict(steps6)["f_branch"]
    kept6 = ev6[ev6["keep"]]
    cens = kept6[kept6["q_entry"] + plateau_w[0] > win_hi]["q_entry"]
    cens_q = sorted({yq(int(q)) for q in cens})
    gB, gD = g_sources(ev6, H), g_sources(ev6D, H)
    keB, keD = kept6["g_e_mean"], ev6D[ev6D["keep"]]["g_e_mean"]
    gd = g_frame()
    gt = g_by_type(gd)
    own0 = own_cell_at_entry(gd, ev6)
    prev_qmax = int(pd.read_csv(PREV_DIR / "d6_entry_events.csv", low_memory=False,
                                usecols=["q_entry", "n_obs_post"]).eval("q_entry + n_obs_post - 1").max())
    accrual = abs(float(b["phi_entry"]) - float(ctl["phi_entry"]))
    fig_path = os.path.relpath(EST_DIR / "fig_entry_estban_window.pdf", DRAFTS).replace("\\", "/")
    if " " in fig_path:
        raise SystemExit(f"figure path has a space, which \\includegraphics will not take: {fig_path}")

    v = {
        "today": f"{dt.date.today():%B} {dt.date.today().day}, {dt.date.today().year}",
        "fignum": fignum, "figpage": figpage, "H": str(H), "H1": str(H + 1),
        "win_lo": yq(win_lo), "win_hi": yq(win_hi), "prev_qmax": yq(prev_qmax),
        "last_plotted": yq(win_hi - plateau_w[0]),
        "date_paths": day(D6_DIR / "d6_entry_paths.csv"),
        "date_panel": day(paths.PROCESSED / "market_panel.parquet"),
        "nq": nq, "phi_E3": f3(vint["E3"][0]), "phi_E4": f3(vint["E4"][0]),
        "n_local": ni(n_local), "n_booked": ni(len(booked)),
        "kept_paper": ni(len(kept6)), "n_cens_B": ni(len(cens)),
        "cens_B": " and ".join(cens_q) if cens_q else "none",
        "branch_cover": yq(branch_cover),
        "n_branch_new": ni(n_branch_new), "n_branch_prev": ni(len(obP)),
        "n_branch_early": ni(int((obP["q_entry"] < win_lo).sum())),
        "phi_paper": f3(b["phi_entry"]), "ci_paper": ci(b["lo"], b["hi"]),
        "n_paper": ni(b["n_events"]), "phi_paper_pooled": f3(b["phi_pooled_abs"]),
        "phig_paper": f3(b["phi_g"]), "w_paper": f3(b["hi"] - b["lo"]),
        "phi_prev": f3(bP["phi_entry"]), "ci_prev": ci(bP["lo"], bP["hi"]),
        "n_prev": ni(bP["n_events"]),
        "phi_raw": f3(raw["phi_entry"]), "ci_raw": ci(raw["lo"], raw["hi"]),
        "n_raw": ni(raw["n_events"]), "w_raw": f3(raw["hi"] - raw["lo"]),
        "phi_control": f3(ctl["phi_entry"]), "n_control": ni(ctl["n_events"]),
        "control_dev": sci(float(ctl["max_abs_diff_vs_published_path"])),
        "accrual_effect": "less than 0.001" if accrual < 0.0005 else f3(accrual),
        "phi_D": f3(d["phi_entry"]), "n_D": ni(d["n_events"]),
        "g_fixed": f3(g_fixed),
        "a4_mult": a4_ceiling(), "no_own0_B": pct(1 - own0),
        "gB_own": pct(gB["own"]), "gB_ent": pct(gB["entity"]), "gB_nat": pct(gB["national"]),
        "gB_car": pct(gB["carried"]), "gD_nat": pct(gD["national"]),
        "g_t1": f3(gt.loc[1, "med"]), "g_t2": f3(gt.loc[2, "med"]), "g_t4": f3(gt.loc[4, "med"]),
        "w_t1": pct(gt.loc[1, "share"]), "w_t2": pct(gt.loc[2, "share"]),
        "w_t4": pct(gt.loc[4, "share"]),
        "ge_med_B": f3(keB.median()), "ge_p10_B": f3(keB.quantile(0.1)),
        "ge_p90_B": f3(keB.quantile(0.9)), "ge_med_D": f3(keD.median()),
        "n_grid": ni(grid_n), "grid_lo": f3(grid_lo), "grid_hi": f"{grid_hi:g}",
        "n_draws": ni(min(int(meta6["boot"]), draw_cap)),
        "plat_lo": str(plateau_w[0]), "plat_hi": str(plateau_w[-1]),
        "plat_list": ", ".join(str(h) for h in plateau_w), "fit_hi": str(plateau_w[0] - 1),
        "model_alt_h0": f3(model_alt_h0), "data_alt_h0": f3(alt6[0]),
        "dmax": f3(dmax),
        "n_files": ni(metaR["n_files"]), "q_first": yq(int(metaR["q_first"])),
        "q_last": yq(int(metaR["q_last"])), "n_rows": ni(metaR["n_rows"]),
        "n_inst": ni(metaR["n_institutions"]), "n_mca": ni(metaR["n_mca"]),
        "n_no_mca": ni(metaR["n_rows_no_mca"]),
        "pct_no_mca": pct(metaR["share_deposits_no_mca"]),
        "ratio_pre": span(pre), "ratio_post": span(post), "pct_post": pct(1 - post.mean()),
        "fig_path": fig_path,
        "table_screens": table_screens(ev6, evR),
        "table_phi": table_phi(imp6, impR, impP),
        "table_paths": paths_tex,
        "table_cohorts": table_cohorts(evP, ev6, evR),
        "table_cont": table_continuity(cont),
    }

    tex = TEMPLATE
    for k, val in v.items():
        tex = tex.replace(f"@@{k}@@", val)
    left = sorted(set(re.findall(r"@@(\w+)@@", tex)))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    return tex


def compile_pdf(tex_path: Path, build_dir: Path | None) -> None:
    exe = shutil.which("pdflatex")
    if exe is None:
        raise SystemExit("pdflatex not on PATH")
    tmp = Path(build_dir) if build_dir else Path(tempfile.mkdtemp(prefix="entry_report_"))
    tmp.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        r = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error",
                            f"-output-directory={tmp}", tex_path.name],
                           cwd=tex_path.parent, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise SystemExit(f"pdflatex failed; log in {tmp}")
    log = (tmp / f"{STEM}.log").read_text(encoding="utf-8", errors="replace")
    issues = [ln for ln in log.splitlines()
              if re.search(r"Overfull|Underfull|undefined|Warning", ln)]
    print(f"  pdflatex: {len(issues)} warning line(s)")
    for ln in issues[:30]:
        print("    " + ln)
    shutil.copy2(tmp / f"{STEM}.pdf", tex_path.with_suffix(".pdf"))
    print(f"  -> {tex_path.with_suffix('.pdf')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    tex_path = DRAFTS / f"{STEM}.tex"
    tex_path.write_text(build(), encoding="utf-8", newline="\n")
    print(f"  -> {tex_path}")
    if not a.no_compile:
        compile_pdf(tex_path, a.build_dir)


if __name__ == "__main__":
    main()
