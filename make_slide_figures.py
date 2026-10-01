"""Slide visuals for the Deposit Competition deck, drawn from the outputs the paper is built on.

No number is typed here: every value comes from an estimation output or from the cached input
of a paper table. Files go to the paper folder (Drafts/Deposit Competition), which the deck's
\\graphicspath and \\input@path already point at:

  fig_market_structure_by_year_{a..f}.pdf  each panel of the paper's Figure 1, drawn alone
  fig_slide_selic_timedep_all.pdf          Selic, the B time-deposit rate and its spread
  fig_slide_selic_timedep_spread.pdf       Selic and the B time-deposit spread
  fig_slide_selic_timedep_rate.pdf         Selic and the B time-deposit rate
  fig_slide_selic_savings.pdf              Selic and the savings (type 2) rate, by quarter
  fig_slide_sleep_ame_E3.pdf               Table 4's average marginal effects, 95% intervals
  fig_slide_demand_coef_E3.pdf             Table 6, Panel A: coefficients, Student-t(G*) 95% intervals
  fig_slide_elasticity_wedge_E3.pdf        active against on-impact own-price elasticity
  fig_slide_sleepy_flow.tex                TikZ flow diagram of equation (9-B)
  fig_slide_roadmap.tex                    TikZ strip of the three estimation steps
  tab_slide_deposit_types.tex              the four deposit types: rate setter, spread, share

SOURCES
  Rout/Compressed_MarketStructure_by_Year_dep_weighted.csv   Table A.4 and Figure 1
                                                             (make_desc_compressed_tables.py)
  PANEL_INTERMED/quarterly_macro_rates.csv                   Selic and the savings rate
                                                             (panel_deposit_rates.py)
  Rout/est1-4_spec12_stage2_comparison.json                  Table 4, unrounded
                                                             (sleep_export_spec12_compare.py)
  market_panel.parquet                                       balances by deposit type, cached
                                                             in Rout/slide_deposit_type_shares.csv
  BLP_RESULTS/cluster_raw/blp_results_E*_spec_12_ext1.json   Table 6, unrounded
  PAPER_NUMBERS/paper_numbers.json                           the elasticities and Table 6's stars
                                                             (make_paper_numbers.py)

The Figure 1 panels reuse the paper figure's own drawing functions, so a panel on a slide and
the panel in the paper cannot drift apart.

Usage:
  python make_slide_figures.py
  python make_slide_figures.py --only ame,flow
  python make_slide_figures.py --routine 4 --refresh-shares
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")          # venv lives in OneDrive; Agg keeps batch runs safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_desc_compressed_tables as mdt
from utils import paths as _paths
from utils import routines as _routines
from utils.window import apply_window, MIN_YEAR, MAX_YEAR

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

T2_CSV = mdt.OUTPUT_DIR / "Compressed_MarketStructure_by_Year_dep_weighted.csv"
MACRO_CSV = mdt.MACRO_RATES_CSV
T4_SIDECAR = _paths.rout_dir() / "est1-4_spec12_stage2_comparison.json"
PANEL_PARQUET = mdt.DATA_DIR / "market_panel.parquet"
SHARES_CSV = mdt.OUTPUT_DIR / "slide_deposit_type_shares.csv"

# A beamer 4:3 text block is about 4.3 in wide: one chart per slide at this size keeps the
# fonts near the deck's footnotesize once the figure is set at \linewidth.
SLIDE_FIGSIZE = (5.2, 3.3)
FS = 1.15
SAVINGS_COLOR = "#00897B"   # the regulated savings rate: neither a bank type nor the policy rate
# Pix launched on 16 November 2020. The quarterly axis centres each year's four quarters on
# the year's tick, so a calendar date sits half a year to the left of its decimal year.
PIX_X_QUARTERLY = 2020 + 10.5 / 12 - 0.5
DEP_COLS = ["dep_a1", "dep_a2", "dep_a4", "dep_a5"]
MINUS = "−"

VISUALS = ("panels", "timedep", "savings", "ame", "coef", "wedge", "flow", "roadmap", "table")


def _save(fig, out: pathlib.Path, stem: str, png: bool):
    p = out / f"{stem}.pdf"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    if png:
        fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}")


def _dump(path: pathlib.Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    print(f"  wrote {path}")


def _legend_below(ax, ncol: int):
    """Legend under the plot, in place of the 'Year' label: these lines cross the whole panel,
    so no corner inside it is free."""
    ax.set_xlabel("")
    ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=ncol,
              frameon=False, columnspacing=1.4, handlelength=1.8)


def _plain(v: float, d: int) -> str:
    r = round(float(v), d)
    if r == 0:
        return f"{0:.{d}f}"
    return (MINUS if r < 0 else "") + f"{abs(r):.{d}f}"


def _signed(v: float, d: int) -> str:
    s = _plain(v, d)
    return s if s.startswith(MINUS) or round(float(v), d) == 0 else "+" + s


# --------------------------------------------------------------------------------------------
# Figure 1, one panel per file
# --------------------------------------------------------------------------------------------
def figure1_panels(out: pathlib.Path, png: bool):
    S = mdt.ms_series(pd.read_csv(T2_CSV))
    for key, draw in mdt.MS_PANELS.items():
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        # Alone, every panel names the Pix rule. In the concentration panel the label sits
        # below the 2,500 threshold's own label, which occupies the top of the rule there.
        draw(ax, S, fs=FS, lettered=False, pix_label=0.86 if key == "a" else True)
        mdt.ms_year_axis(ax, S["years"], fs=FS)
        _save(fig, out, f"fig_market_structure_by_year_{key}", png)


# --------------------------------------------------------------------------------------------
# Selic against the B time deposit (Table A.4, Panel B)
# --------------------------------------------------------------------------------------------
def timedep_charts(out: pathlib.Path, png: bool):
    """Three versions of one chart. The B time-deposit rate is Selic minus the B spread, both
    as Table A.4 prints them: the spread is Selic minus the deposit rate by construction."""
    S = mdt.ms_series(pd.read_csv(T2_CSV))
    years, selic, spread = S["years"], S["selic"], S["sp4_b"]
    rate = selic - spread
    lines = {
        "selic":  (selic, mdt.NATL_COLOR, "-", "Selic", "{:.1f}%"),
        "rate":   (rate, mdt.B_COLOR, "-", "B time-deposit rate", "{:.1f}%"),
        "spread": (spread, mdt.B_COLOR, (0, (5, 2)), "B time-deposit spread", "{:.1f} pp"),
    }
    variants = {
        "all":    (("selic", "rate", "spread"), "Selic and B time deposits", "% p.a. (spread: pp p.a.)"),
        "spread": (("selic", "spread"), "Selic and the B time-deposit spread", "% p.a. (spread: pp p.a.)"),
        "rate":   (("selic", "rate"), "Selic and the B time-deposit rate", "% p.a."),
    }
    for name, (keys, title, ylabel) in variants.items():
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        if "spread" in keys:
            ax.axhline(0, color="#BFBFBA", lw=0.8, zorder=1)
        for k in keys:
            y, color, ls, label, fmt = lines[k]
            ax.plot(years, y, color=color, label=label, **{**mdt.MS_LINE_KW, "linestyle": ls})
            mdt.ms_endlabel(ax, years, y, fmt.format(y[-1]), FS)
        mdt.ms_style(ax, title, ylabel, pix_label=True, fs=FS)
        mdt.ms_year_axis(ax, years, fs=FS)
        _legend_below(ax, len(keys))
        _save(fig, out, f"fig_slide_selic_timedep_{name}", png)


# --------------------------------------------------------------------------------------------
# Selic against the savings rate (type 2)
# --------------------------------------------------------------------------------------------
def macro_quarterly() -> pd.DataFrame:
    """Quarterly Selic and savings rate over the estimation window, compounded to annual
    percent as the paper's tables do. The file stores quarterly rates in percent."""
    df = pd.read_csv(MACRO_CSV)
    df["year"] = df["AnoMes"] // 100
    df["quarter"] = (df["AnoMes"] % 100) // 3
    df = apply_window(df, label="slide macro rates")
    df["x"] = df["year"] + (df["quarter"] - 2.5) / 4
    df["selic_ann"] = mdt.annualized_rate_pct(df["selic_qoq"] / 100.0)
    df["savings_ann"] = mdt.annualized_rate_pct(df["savings_rate_qoq"] / 100.0)
    return df.sort_values("x").reset_index(drop=True)


def savings_chart(out: pathlib.Path, png: bool):
    q = macro_quarterly()
    x = q["x"].to_numpy()
    kw = dict(linewidth=2, solid_capstyle="round", zorder=3)
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    ax.fill_between(x, q["savings_ann"], q["selic_ann"], color=mdt.NATL_COLOR, alpha=0.07,
                    linewidth=0, zorder=1)
    for col, color, label in (("selic_ann", mdt.NATL_COLOR, "Selic"),
                              ("savings_ann", SAVINGS_COLOR, "Savings rate (type 2)")):
        y = q[col].to_numpy()
        ax.plot(x, y, color=color, label=label, **kw)
        mdt.ms_endlabel(ax, x, y, f"{y[-1]:.1f}%", FS)
    mdt.ms_style(ax, "Selic and the regulated savings rate", "% p.a.", pix_label=True, fs=FS,
                 pix_x=PIX_X_QUARTERLY)
    mdt.ms_year_axis(ax, np.arange(MIN_YEAR, MAX_YEAR + 1, dtype=float), fs=FS)
    _legend_below(ax, 2)
    _save(fig, out, "fig_slide_selic_savings", png)


# --------------------------------------------------------------------------------------------
# Table 4's average marginal effects
# --------------------------------------------------------------------------------------------
# Table 4's rows in its order, with plain-text labels (the sidecar's are LaTeX).
AME_ROWS = (
    ("interaction_cadunico_families_per1000", "CadÚnico families\n(100s per 1,000 inhab.)"),
    ("interaction_fraction_65plus", "Share aged 65+\n(pp)"),
    ("interaction_risk_free_qoq_lag", "Lagged Selic\n(pp per quarter)"),
    ("interaction_connections_per100", "Mobile lines\n(per 100 inhab.)"),
    ("interaction_pix_exists", "Pix available\n(0 to 1)"),
)


def ame_chart(out: pathlib.Path, png: bool, E: int):
    sc = json.loads(T4_SIDECAR.read_text(encoding="utf-8"))
    col = next((c for c, e in (sc.get("column_est_id") or {}).items() if int(e) == E), None)
    if col is None:
        sys.exit(f"{T4_SIDECAR.name}: no column for routine E{E}")
    cells = {r["var"]: r["cells"][col] for r in sc["rows"]}
    rows = [(lab, cells[var]) for var, lab in AME_ROWS
            if var in cells and cells[var].get("estimate") is not None]

    texts = []
    for _, c in rows:
        d = 2 if abs(c["estimate"]) >= 0.05 else 3
        dagger = "†" if c.get("dagger") else ""
        texts.append(f"{_signed(c['estimate'], d)}{c.get('stars') or ''}  "
                     f"[{_plain(c['ci_lo'], d)}, {_plain(c['ci_hi'], d)}]{dagger}")
    note = (f"Routine {_routines.est_roman(E)}. Bars: estimates. Lines: 95% intervals.\n"
            "*/**/***: the 90/95/99% interval excludes zero.")
    if any(c.get("dagger") for _, c in rows):
        note += " †: quarter-clustered."
    _forest(out, png, f"fig_slide_sleep_ame_E{E}", [lab for lab, _ in rows],
            [c["estimate"] for _, c in rows], [c["ci_lo"] for _, c in rows],
            [c["ci_hi"] for _, c in rows], texts, "Effect on the sleepy share (pp)", note)


def _forest(out, png, stem, labels, est, lo, hi, texts, xlabel, note):
    """Horizontal bars with interval whiskers, one row per estimate, the printed value at the
    right. Row labels and values sit inside the figure, so the saved width is the figure's own
    and the fonts keep their size on the slide."""
    est, lo, hi = (np.asarray(a, dtype=float) for a in (est, lo, hi))
    ypos = np.arange(len(labels))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    fig.subplots_adjust(left=0.27, right=0.63, top=0.98, bottom=0.27)
    ax.axvline(0, color="#8A8A85", lw=0.9, zorder=1)
    ax.barh(ypos, est, height=0.42, color=mdt.NATL_COLOR, zorder=2)
    ax.errorbar(est, ypos, xerr=np.vstack([est - lo, hi - est]), fmt="none", ecolor="#111111",
                elinewidth=1.1, capsize=3, capthick=1.1, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8 * FS)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    for y, t in zip(ypos, texts):
        ax.text(1.04, y, t, transform=ax.get_yaxis_transform(), fontsize=8 * FS,
                color=mdt.AXIS_INK, va="center", ha="left")
    ax.set_xlabel(xlabel, fontsize=8.5 * FS, color=mdt.AXIS_INK)
    ax.grid(True, axis="x", color=mdt.GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(mdt.AXIS_INK)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", labelsize=9 * FS, colors=mdt.AXIS_INK, length=3)
    ax.tick_params(axis="y", length=0, colors=mdt.AXIS_INK)
    fig.text(0.02, 0.11, note, fontsize=7.5 * FS, color=mdt.MUTED_INK, ha="left", va="top")
    _save(fig, out, stem, png)


# --------------------------------------------------------------------------------------------
# Table 6, Panel A, and the on-impact elasticity
# --------------------------------------------------------------------------------------------
# Panel A's rows in its order: (name in the BLP results, id slug in paper_numbers.json, label).
COEF_ROWS = (
    ("alpha", "alpha", "Price coefficient"),
    ("fgc_covered", "fgc", "FGC covered"),
    ("has_ip", "ip", "Group contains IP"),
    ("log_total_assets_lag", "lnassets", "ln(total assets),\nlagged"),
    ("is_state_owned", "state_owned", "State-owned"),
)


def _paper_numbers():
    """({id: entry} of paper_numbers.json, the make_paper_numbers module): the values that
    script derives, and its constants."""
    import make_paper_numbers as mpn
    p = mpn.OUT_DIR / "paper_numbers.json"
    return {e["id"]: e for e in json.loads(p.read_text(encoding="utf-8"))["entries"]}, mpn


def demand_coef_plot(out: pathlib.Path, png: bool, E: int):
    """Table 6, Panel A: the mean-utility coefficients with their 95% intervals. Table 6 prints
    standard errors and takes its stars from Student-t(G*), so the interval drawn is the one that
    inference implies: estimate +/- t_{0.975}(G*) standard errors. A whisker then crosses zero
    exactly when the coefficient is not significant at the 5 percent level in the table.
    Estimates, standard errors, p-values and G* come unrounded from the BLP results; the stars
    are the ones Table 6 prints."""
    from scipy import stats
    pn, mpn = _paper_numbers()
    import make_blp_rc_table as rc
    blp = rc.load_stage(E, mpn.BLP_STAGE)
    if not blp:
        sys.exit(f"no BLP results for routine E{E}, stage {mpn.BLP_STAGE}")
    names = blp["param_names_theta1"]
    g_star = float(blp["G_star"])
    crit = float(stats.t.ppf(0.975, g_star))
    labels, est, se, texts = [], [], [], []
    for name, slug, lab in COEF_ROWS:
        if name not in names:
            continue
        i = names.index(name)
        b, s = float(blp["theta1"][i]), float(blp["theta1_se"][i])
        stars = "*" * int(((pn.get(f"demand.t6.{slug}.E{E}") or {}).get("value_full") or {})
                          .get("stars") or 0)
        labels.append(lab)
        est.append(b)
        se.append(s)
        texts.append(f"{_signed(b, 2)}{stars}  ({_plain(s, 2)})")
        # The interval and the table's p-value must tell the same story at the 5 percent level.
        p_tab = float(blp["theta1_pval"][i])
        p_t = float(2.0 * stats.t.sf(abs(b / s), g_star))
        print(f"  {name}: table p {p_tab:.4f}, Student-t(G*) p {p_t:.4f}, interval "
              f"[{b - crit * s:.3f}, {b + crit * s:.3f}]"
              + ("" if (p_tab < 0.05) == (abs(b / s) > crit) else "  <-- DISAGREE at 5%"))
    est, se = np.array(est), np.array(se)
    note = (f"Routine {_routines.est_roman(E)}. Bars: estimates. Lines: 95% intervals, estimate "
            f"± {crit:.2f} standard errors\n(Student-t, {g_star:.1f} effective clusters). "
            "Standard errors in parentheses.")
    _forest(out, png, f"fig_slide_demand_coef_E{E}", labels, est, est - crit * se, est + crit * se,
            texts, "Coefficient in mean utility", note)


def elasticity_bars(out: pathlib.Path, png: bool, E: int):
    """The active-demand elasticity against the on-impact elasticity of the deposit base, on
    one axis: the second bar is barely visible, which is the point."""
    pn, _ = _paper_numbers()
    active = float(pn[f"demand.elast.blp.E{E}"]["value_full"])
    impact = float(pn[f"demand.impact.blp.E{E}"]["value_full"])
    rows = (("Active depositors", active, 2), ("Deposit base,\non impact", impact, 3))
    ypos = np.array([1.0, 0.0])
    fig, ax = plt.subplots(figsize=(5.2, 2.1))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.97, bottom=0.30)
    ax.axvline(0, color="#8A8A85", lw=0.9, zorder=1)
    ax.barh(ypos, [v for _, v, _ in rows], height=0.5, color=mdt.NATL_COLOR, zorder=2)
    for y, (_, v, d) in zip(ypos, rows):
        ax.annotate(_plain(v, d), xy=(v, y), xytext=(-5, 0), textcoords="offset points",
                    fontsize=9 * FS, color=mdt.AXIS_INK, va="center", ha="right")
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for lab, _, _ in rows], fontsize=8.5 * FS)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(min(active, impact) * 1.25, 0.0)
    ax.set_xlabel(f"Own-price elasticity, routine {_routines.est_roman(E)}", fontsize=8.5 * FS,
                  color=mdt.AXIS_INK)
    ax.grid(True, axis="x", color=mdt.GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(mdt.AXIS_INK)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", labelsize=9 * FS, colors=mdt.AXIS_INK, length=3)
    ax.tick_params(axis="y", length=0, colors=mdt.AXIS_INK)
    _save(fig, out, f"fig_slide_elasticity_wedge_E{E}", png)


# --------------------------------------------------------------------------------------------
# Estimation roadmap
# --------------------------------------------------------------------------------------------
ROADMAP_TEX = r"""% fig_slide_roadmap.tex -- GENERATED by make_slide_figures.py. Do not edit.
% The three estimation steps and the object each recovers, 10.6 cm wide. \input it inside a
% frame. Needs tikz, and paper_equations_macros.tex for the equation numbers.
\begin{tikzpicture}[font=\footnotesize, >=latex,
    stage/.style={draw, rounded corners=2pt, align=flush center, inner sep=4pt, text width=2.7cm,
                  fill=black!6},
    flow/.style={->, line width=0.6pt}]
  \node[stage] (s1) at (0, 0)   {\textbf{Step 1: Sleepiness}\\[3pt] $\phi(\boldsymbol{S}_{mt})$\\[3pt] deposit persistence, \eqref{eq:13}};
  \node[stage] (s2) at (3.8, 0) {\textbf{Step 2: Demand}\\[3pt] $(\bar{\alpha},\bar{\beta},\Pi,\Sigma)$\\[3pt] active shares, \eqref{eq:14-B}};
  \node[stage] (s3) at (7.6, 0) {\textbf{Step 3: Costs}\\[3pt] $(\omega^{\kappa},\zeta^{\kappa},\boldsymbol{\gamma}^{\kappa})$\\[3pt] BBL inequalities, \eqref{eq:16}};
  \draw[flow] (s1) -- node[above, font=\scriptsize] {$\hat{\phi}$} (s2);
  \draw[flow] (s2) -- node[above, font=\scriptsize] {$\hat{\alpha},\hat{\delta}$} (s3);
\end{tikzpicture}
"""


def roadmap_strip(out: pathlib.Path):
    _dump(out / "fig_slide_roadmap.tex", ROADMAP_TEX)


# --------------------------------------------------------------------------------------------
# Flow diagram of equation (9-B)
# --------------------------------------------------------------------------------------------
FLOW_TEX = r"""% fig_slide_sleepy_flow.tex -- GENERATED by make_slide_figures.py. Do not edit.
% Equation (9-B) as a flow, 10.6 cm wide: it fills a beamer frame. \input it inside a frame.
% Needs tikz, and the paper's macro files for \pn and \ref.
\begin{tikzpicture}[font=\footnotesize, >=latex,
    box/.style={draw, rounded corners=2pt, align=center, inner sep=4pt, text width=#1},
    state/.style={box=#1, fill=black!6},
    flow/.style={->, line width=0.6pt}]
  \node[box=1.9cm]   (last)   at (0, 0)       {Balance at bank $j$,\\ end of $t-1$\\[3pt] $\mathrm{Dep}_{jkmt-1}$};
  \node[state=4.0cm] (asleep) at (4.2, 1.25)  {\textbf{Asleep}, share $\phi_{mt}$:\\ stay and earn interest\\[3pt] \mbox{$\phi_{mt}\bigl(1+r^{\mathrm{dep}}_{jkmt-1}\bigr)\mathrm{Dep}_{jkmt-1}$}};
  \node[state=4.0cm] (awake)  at (4.2, -1.25) {\textbf{Awake}, share $1-\phi_{mt}$:\\ join the market's active depositors and choose again\\[3pt] \mbox{$(1-\phi_{mt})\,M_{mt}\,s^{\mathrm{Act}}_{jkmt}$}};
  \node[box=1.9cm]   (now)    at (8.4, 0)     {Balance at bank $j$,\\ end of $t$\\[3pt] $\mathrm{Dep}_{jkmt}$};
  \draw[flow] (last.east) -- (asleep.west);
  \draw[flow] (last.east) -- (awake.west);
  \draw[flow] (asleep.east) -- (now.west);
  \draw[flow] (awake.east) -- (now.west);
  \node[font=\scriptsize, anchor=north] at (4.2, -2.6) {Routine __REF__: on average \pn[1]{sleep.meanphi.E__E__} percent of balances are asleep in a quarter};
\end{tikzpicture}
"""


def flow_diagram(out: pathlib.Path, E: int):
    _dump(out / "fig_slide_sleepy_flow.tex",
          FLOW_TEX.replace("__REF__", _routines.est_ref(E)).replace("__E__", str(E)))


# --------------------------------------------------------------------------------------------
# The four deposit types
# --------------------------------------------------------------------------------------------
def deposit_type_totals(refresh: bool) -> pd.DataFrame:
    """Balances by deposit type, quarter and firm type, on the paper's sample (prepare_panel).
    Cached: the panel is read again only when it is newer than the cache."""
    if (SHARES_CSV.is_file() and not refresh
            and SHARES_CSV.stat().st_mtime >= PANEL_PARQUET.stat().st_mtime):
        return pd.read_csv(SHARES_CSV)
    cols = ["year", "quarter", "CodConglomeradoPrudencial", "CODMUN_IBGE", "is_B"] + DEP_COLS
    print(f"Loading {PANEL_PARQUET.name} ...")
    df = mdt.prepare_panel(pd.read_parquet(PANEL_PARQUET, columns=cols))
    tot = df.groupby(["year", "quarter", "bank_type"], as_index=False)[DEP_COLS].sum()
    tot.to_csv(SHARES_CSV, index=False)
    print(f"  wrote {SHARES_CSV}")
    return tot


TYPES_TEX = r"""% tab_slide_deposit_types.tex -- GENERATED by make_slide_figures.py. Do not edit.
% Needs booktabs. Spreads: Selic minus the deposit rate, compounded to annual rates, in pp;
% means over __Y0__--__Y1__ of the annual values of Table A.4 (types 4 and 5, deposit-weighted;
% type 5 from 2020) and of the quarterly Selic and savings rate (types 1 and 2).
% Shares: each type's share of the four types' balances, B and D firms pooled, mean over the
% __NQ__ quarters of __Y0__--__Y1__.
\begin{tabular}{@{}llrrr@{}}
\toprule
 & & \multicolumn{2}{c}{Spread (pp p.a.)} & Share of \\
\cmidrule(lr){3-4}
Deposit type & Rate set by & B firms & D firms & balances (\%) \\
\midrule
__ROWS__
\bottomrule
\end{tabular}
"""


def deposit_types_table(out: pathlib.Path, refresh: bool, decimals: int):
    t2 = pd.read_csv(T2_CSV).set_index("var")
    q = macro_quarterly()
    f = lambda v: _plain(v, decimals).replace(MINUS, "$-$")

    def t2_mean(var):
        return float(pd.to_numeric(t2.loc[var], errors="coerce").mean())

    selic = float(q["selic_ann"].mean())
    savings_spread = float((q["selic_ann"] - q["savings_ann"]).mean())

    tot = deposit_type_totals(refresh)
    by_q = tot.groupby(["year", "quarter"])[DEP_COLS].sum()
    share = (by_q.div(by_q.sum(axis=1), axis=0) * 100.0).mean()

    both = lambda v: r"\multicolumn{2}{c}{%s}" % f(v)
    rows = [
        ("Demand (1)", "Law: zero rate", both(selic), share["dep_a1"]),
        ("Savings (2)", "Law: formula", both(savings_spread), share["dep_a2"]),
        ("Time, CDB (4)", "Bank",
         f"{f(t2_mean('spread_ann_a4_w'))} & {f(t2_mean('spread_ann_a4_d_w'))}", share["dep_a4"]),
        ("Prepaid (5)", "Bank",
         f"{f(t2_mean('spread_ann_a5_w'))} & {f(t2_mean('spread_ann_a5_d_w'))}", share["dep_a5"]),
    ]
    body = "\n".join(f"{name} & {who} & {spread} & {f(sh)} \\\\" for name, who, spread, sh in rows)
    tex = (TYPES_TEX.replace("__ROWS__", body).replace("__Y0__", str(MIN_YEAR))
           .replace("__Y1__", str(MAX_YEAR)).replace("__NQ__", str(len(by_q))))
    _dump(out / "tab_slide_deposit_types.tex", tex)
    # The Selic of the macro file and of Table A.4 are the same series on two routes.
    print(f"  Selic mean {MIN_YEAR}-{MAX_YEAR}: macro file {selic:.3f}, Table A.4 "
          f"{t2_mean('risk_free_ann'):.3f}")
    print("  shares (%): " + ", ".join(f"{c} {share[c]:.3f}" for c in DEP_COLS))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated subset of: " + ", ".join(VISUALS))
    ap.add_argument("--routine", type=int, default=_routines.LINK_ESTS[0],
                    help="routine of the estimate charts and of the flow diagram's note")
    ap.add_argument("--decimals", type=int, default=3, help="decimals of the deposit-types table")
    ap.add_argument("--refresh-shares", action="store_true",
                    help="recompute the balances by deposit type from the market panel")
    ap.add_argument("--out", type=pathlib.Path, default=mdt.DRAFTS_DIR,
                    help="output folder (default: the paper folder)")
    ap.add_argument("--png", action="store_true", help="also write a PNG next to each PDF")
    args = ap.parse_args()

    todo = VISUALS if args.only is None else tuple(s.strip() for s in args.only.split(","))
    unknown = [s for s in todo if s not in VISUALS]
    if unknown:
        sys.exit(f"unknown visual(s): {', '.join(unknown)}; choose from {', '.join(VISUALS)}")
    args.out.mkdir(parents=True, exist_ok=True)

    if "panels" in todo:
        figure1_panels(args.out, args.png)
    if "timedep" in todo:
        timedep_charts(args.out, args.png)
    if "savings" in todo:
        savings_chart(args.out, args.png)
    if "ame" in todo:
        ame_chart(args.out, args.png, args.routine)
    if "coef" in todo:
        demand_coef_plot(args.out, args.png, args.routine)
    if "wedge" in todo:
        elasticity_bars(args.out, args.png, args.routine)
    if "flow" in todo:
        flow_diagram(args.out, args.routine)
    if "roadmap" in todo:
        roadmap_strip(args.out)
    if "table" in todo:
        deposit_types_table(args.out, args.refresh_shares, args.decimals)


if __name__ == "__main__":
    main()
