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
  fig_slide_bbl_violations.pdf             Table C.14: violated inequalities, up against down
  fig_slide_sleepy_flow.tex                TikZ flow diagram of equation (9-B)
  fig_slide_roadmap.tex                    TikZ strip of the three estimation steps
  tab_slide_deposit_types.tex              the four deposit types: rate setter, spread, share

Motivation charts (prices against balances); their unrounded numbers go to
Rout/slide_motivation.json:
  fig_slide_selic_timedep_all_{sd,iqr}.pdf the Selic chart with a band across B banks around the
                                           B spread: +/-1 SD, or the interquartile range
  fig_slide_motiv_spreads_by_type.pdf      spread of each deposit type, by quarter
  fig_slide_motiv_shares_by_type.pdf       each deposit type's share of balances, by quarter
  fig_slide_motiv_rates_by_type.pdf        each deposit type's rate next to Selic, by quarter
  fig_slide_motiv_reprice_binscatter.pdf   B banks: change in own spread against deposit growth
                                           (_points: every bank-quarter behind the bins)
  fig_slide_motiv_dfirm_premium.pdf        D firms' rate premium over B firms and their share
  fig_slide_motiv_entry_by_gap.pdf         B entrants' share paths, by price at entry
                                           (_level: the share of B deposits, not normalised)

SOURCES
  Rout/Compressed_MarketStructure_by_Year_dep_weighted.csv   Table A.4 and Figure 1
                                                             (make_desc_compressed_tables.py)
  PANEL_INTERMED/quarterly_macro_rates.csv                   Selic and the savings rate
                                                             (panel_deposit_rates.py)
  Rout/est1-4_spec12_stage2_comparison.json                  Table 4, unrounded
                                                             (sleep_export_spec12_compare.py)
  market_panel.parquet                                       balances by deposit type, cached
                                                             in Rout/slide_deposit_type_shares.csv;
                                                             spreads and balances by bank and market
  DIAG_PHI_SEPARATION/d6_entry_{events,paths}.csv, d6_meta.json   the entry events of Figure 2
                                                             (sleep_ident_entry_dynamics.py)
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
TYPES_JSON = mdt.OUTPUT_DIR / "slide_deposit_types.json"

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

VISUALS = ("panels", "timedep", "savings", "ame", "coef", "wedge", "violations", "flow", "roadmap",
           "table", "band", "pricevol", "reprice", "dpremium", "entrygap")


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
def _timedep_lines(S: dict) -> dict:
    """Selic, the B time-deposit rate and the B spread, each with its style. The B time-deposit
    rate is Selic minus the B spread, both as Table A.4 prints them: the spread is Selic minus
    the deposit rate by construction."""
    selic, spread = S["selic"], S["sp4_b"]
    return {
        "selic":  (selic, mdt.NATL_COLOR, "-", "Selic", "{:.1f}%"),
        "rate":   (selic - spread, mdt.B_COLOR, "-", "B time-deposit rate", "{:.1f}%"),
        "spread": (spread, mdt.B_COLOR, (0, (5, 2)), "B time-deposit spread", "{:.1f} pp"),
    }


def timedep_charts(out: pathlib.Path, png: bool):
    """Three versions of one chart, from Table A.4's annual series."""
    S = mdt.ms_series(pd.read_csv(T2_CSV))
    years = S["years"]
    lines = _timedep_lines(S)
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
# Table C.14: violated inequalities by the direction of the deviation
# --------------------------------------------------------------------------------------------
UP_COLOR = mdt.NATL_COLOR
DOWN_COLOR = "#B8B8B3"


def violation_bars(out: pathlib.Path, png: bool):
    """The share of the BBL inequalities violated at theta-hat, upward against downward
    deviations, per routine and firm type. While the solve behind the numbers is not the paper's
    target run, the chart says so and prints the discounting it was solved at."""
    pn, mpn = _paper_numbers()
    groups, up, down, pending, runs = [], [], [], False, set()
    for E in mpn.ROUTINES:
        for k in mpn.BLOCKS:
            eu, ed = pn.get(f"bbl.ud.up.E{E}.{k}"), pn.get(f"bbl.ud.down.E{E}.{k}")
            if not eu or not ed or eu.get("value_full") is None or ed.get("value_full") is None:
                continue
            groups.append(f"{k} firms\n{_routines.est_roman(E)}")
            up.append(100.0 * float(eu["value_full"]))
            down.append(100.0 * float(ed["value_full"]))
            for e in (eu, ed):
                pending = pending or e.get("status") != "ok"
                runs |= {(r.get("beta"), r.get("T")) for r in (e.get("run") or [])}
    if not groups:
        sys.exit("paper_numbers.json has no bbl.ud.* entries: run make_paper_numbers.py first")

    x = np.arange(len(groups), dtype=float)
    w = 0.36
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    ax.axhline(50, color="#8A8A85", lw=0.8, ls=(0, (1, 3)), zorder=1)
    for xs, vals, color, label in ((x - w / 2 - 0.01, up, UP_COLOR, "Upward: raise the spread"),
                                   (x + w / 2 + 0.01, down, DOWN_COLOR, "Downward: lower the spread")):
        ax.bar(xs, vals, width=w, color=color, label=label, zorder=2)
        for xi, v in zip(xs, vals):
            ax.annotate(f"{v:.0f}%", xy=(xi, v), xytext=(0, 2), textcoords="offset points",
                        fontsize=8 * FS, color=mdt.AXIS_INK, ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8.5 * FS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Inequalities violated (%)", fontsize=9 * FS, color=mdt.AXIS_INK)
    ax.grid(True, axis="y", color=mdt.GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(mdt.AXIS_INK)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(axis="y", labelsize=9 * FS, colors=mdt.AXIS_INK, length=3)
    ax.tick_params(axis="x", length=0, colors=mdt.AXIS_INK)
    ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              frameon=False, columnspacing=1.4, handlelength=1.4)
    if pending:
        solved = "; ".join(f"β = {b}, T = {t}" for b, t in sorted(runs) if b is not None)
        design = (f"β = {pn['bbl.design.beta']['value_full']}, "
                  f"T = {pn['bbl.design.T']['value_full']}")
        fig.text(0.02, -0.19, f"Provisional: solved at {solved}. The paper's design is {design}.",
                 fontsize=7.5 * FS, color=mdt.MUTED_INK, ha="left", va="top")
    _save(fig, out, "fig_slide_bbl_violations", png)


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
  \node[font=\scriptsize, anchor=north] at (4.2, -2.6) {In the preferred estimate, on average \pn[1]{sleep.meanphi.E__E__} percent of balances are asleep in a quarter.};
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
% means over __Y0__--__Y1__ of the annual values of Table A.4 (types 4 and 5, balance-weighted:
% the spread on the average real, not Table 1's average bank; type 5 from 2020) and of the
% quarterly Selic and savings rate (types 1 and 2). The values have keys: \pn{slide.types.*}.
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
    """The table and, next to the paper's other table inputs, a sidecar of its unrounded values:
    make_paper_numbers.py turns the sidecar into the slide.types.* keys, so the prose of the
    slide quotes the numbers the table prints."""
    t2 = pd.read_csv(T2_CSV).set_index("var")
    q = macro_quarterly()

    def f(v):
        s = _plain(v, decimals)
        return f"${s.replace(MINUS, '-')}$" if s.startswith(MINUS) else s

    def t2_mean(var):
        return float(pd.to_numeric(t2.loc[var], errors="coerce").mean())

    selic = float(q["selic_ann"].mean())
    savings_spread = float((q["selic_ann"] - q["savings_ann"]).mean())

    tot = deposit_type_totals(refresh)
    by_q = tot.groupby(["year", "quarter"])[DEP_COLS].sum()
    share = (by_q.div(by_q.sum(axis=1), axis=0) * 100.0).mean()

    # The time-deposit and prepaid spreads are Table A.4's balance-weighted ones (the spread on
    # the average real of deposits), which is also what the regulated rows and the share column
    # measure. Table 1's means over conglomerate-quarters weigh every bank equally.
    types = {
        "demand": dict(label="Demand (1)", setter="Law: zero rate", spread=selic,
                       share=float(share["dep_a1"])),
        "savings": dict(label="Savings (2)", setter="Law: formula", spread=savings_spread,
                        share=float(share["dep_a2"])),
        "time": dict(label="Time, CDB (4)", setter="Bank",
                     spread=dict(B=t2_mean("spread_ann_a4_w"), D=t2_mean("spread_ann_a4_d_w")),
                     share=float(share["dep_a4"])),
        "prepaid": dict(label="Prepaid (5)", setter="Bank",
                        spread=dict(B=t2_mean("spread_ann_a5_w"), D=t2_mean("spread_ann_a5_d_w")),
                        share=float(share["dep_a5"])),
    }

    def spread_cells(sp):
        if isinstance(sp, dict):
            return f"{f(sp['B'])} & {f(sp['D'])}"
        return r"\multicolumn{2}{c}{%s}" % f(sp)

    body = "\n".join(f"{r['label']} & {r['setter']} & {spread_cells(r['spread'])} & {f(r['share'])} \\\\"
                     for r in types.values())
    tex = (TYPES_TEX.replace("__ROWS__", body).replace("__Y0__", str(MIN_YEAR))
           .replace("__Y1__", str(MAX_YEAR)).replace("__NQ__", str(len(by_q))))
    _dump(out / "tab_slide_deposit_types.tex", tex)
    _dump(TYPES_JSON, json.dumps(dict(
        schema=1, generator="make_slide_figures.py:deposit_types_table",
        tex_file="tab_slide_deposit_types.tex", window=[MIN_YEAR, MAX_YEAR],
        n_quarters=int(len(by_q)), types=types), indent=2) + "\n")
    # The Selic of the macro file and of Table A.4 are the same series on two routes.
    print(f"  Selic mean {MIN_YEAR}-{MAX_YEAR}: macro file {selic:.3f}, Table A.4 "
          f"{t2_mean('risk_free_ann'):.3f}")
    print("  shares (%): " + ", ".join(f"{c} {share[c]:.3f}" for c in DEP_COLS))


# --------------------------------------------------------------------------------------------
# Motivation: prices against balances
# --------------------------------------------------------------------------------------------
MOTIV_JSON = mdt.OUTPUT_DIR / "slide_motivation.json"
D6_DIR = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_PHI_SEPARATION"
MOTIV_COLS = ["year", "quarter", "CodConglomeradoPrudencial", "CODMUN_IBGE", "mca_code", "is_B",
              "dep_a1", "dep_a2", "dep_a4", "dep_a5", "spread_a4", "spread_a5",
              "spread_a4_national", "rate_a4_national", "rate_a4_guarded", "risk_free_qoq"]
# Demand deposits (type 1). With the savings, B and D hues it passes the dataviz palette checks;
# its closest pair under deuteranopia (savings, Delta E 7.2) is told apart by the end labels.
DEMAND_COLOR = "#AD1457"
# Prepaid accounts (type 5), B and D pooled. Passes the same checks against the four hues above;
# under deuteranopia it sits 7.2 from the D hue and its contrast is below 3:1, so the line is
# dotted and carries an end label. Figure 2's model lines use purple and cyan, so neither is
# reused here.
PREPAID_COLOR = "#7CB342"
PREPAID_LS = (0, (1, 1.6))
# Intervals on the plotted means: conglomerate wild cluster bootstrap, as in the paper's tables.
BOOT_B = 999
# B entrants that do not pay more than incumbents: a lighter step of the B hue, dashed. Its
# contrast with the white surface is below 3:1, so the line also carries an end label.
GAP_REST_COLOR = "#64B5F6"
# A bank's time-deposit growth is read only from a type-4 book of at least this size (R$).
TYPE4_FLOOR = 1e8
# The repricing scatter drops changes in spread and growth rates outside these percentiles.
TRIM_PCT = (1.0, 99.0)
N_BINS = 20
GROWTH_H = 4
# An entrant "pays more" when its type-4 spread is at least this far (pp) below the incumbents'.
ENTRY_GAP_PP = 1.0

_MOTIV_CACHE: dict = {}


def motiv_panel() -> pd.DataFrame:
    """The paper's sample (prepare_panel) with the columns the motivation charts read, and the
    row-level type-4 and type-5 spreads annualized as build_table2 does it. prepare_panel leaves
    no type-5 balance or spread before Pix. Read once per run."""
    if "df" not in _MOTIV_CACHE:
        print(f"Loading {PANEL_PARQUET.name} (motivation columns) ...")
        df = mdt.prepare_panel(pd.read_parquet(PANEL_PARQUET, columns=MOTIV_COLS))
        df["sp4"] = mdt.annualized_spread_pp(df["spread_a4"], df["risk_free_qoq"])
        df["sp5"] = mdt.annualized_spread_pp(df["spread_a5"], df["risk_free_qoq"])
        _MOTIV_CACHE["df"] = df
    return _MOTIV_CACHE["df"]


def b_spread_cells(df: pd.DataFrame) -> pd.DataFrame:
    """B rows with the weight under which the weighted mean of the row-level type-4 spread is
    Table A.4's B line (build_table2): the bank's share of the market-quarter's type-4 balance
    times the market-quarter's balance of all four types."""
    b = df[df["bank_type"] == "B"]
    keys = [b["mca_code"], b["year"], b["quarter"]]
    dep_tot = b[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].sum(axis=1, min_count=1).fillna(0.0)
    dep_total_mq = dep_tot.groupby(keys).transform("sum")
    w4 = b["dep_a4"].where(b["dep_a4"] > 0, 0.0).where(b["sp4"].notna(), 0.0)
    den = w4.groupby(keys).transform("sum")
    w = np.where(den > 0, w4 / den.where(den > 0, 1.0) * dep_total_mq, 0.0)
    return pd.DataFrame({"year": b["year"], "quarter": b["quarter"],
                         "congl": b["CodConglomeradoPrudencial"], "sp4": b["sp4"], "w": w})


def d_spread(df: pd.DataFrame, by: list) -> pd.Series:
    """D firms' type-4 spread weighted by their type-4 balances, as build_table2 weights it."""
    d = df[df["bank_type"] == "D"]
    w = d["dep_a4"].where(d["dep_a4"] > 0, 0.0).where(d["sp4"].notna(), 0.0)
    g = [d[k] for k in by]
    return (d["sp4"].fillna(0.0) * w).groupby(g).sum() / w.groupby(g).sum().replace(0, np.nan)


def spread_cells(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Rows behind one plotted mean spread, with the bank (conglomerate) as the cluster:
    time_B, Table A.4's B cells; time_D, D firms weighted by type-4 balance; prepaid, B and D
    pooled, weighted by type-5 balance (Table A.4's national type-5 line)."""
    if kind == "time_B":
        c = b_spread_cells(df)
        return pd.DataFrame({"year": c["year"], "quarter": c["quarter"], "congl": c["congl"],
                             "x": c["sp4"], "w": c["w"]})
    d = df[df["bank_type"] == "D"] if kind == "time_D" else df
    x, dep = ("sp4", "dep_a4") if kind == "time_D" else ("sp5", "dep_a5")
    return pd.DataFrame({"year": d["year"], "quarter": d["quarter"],
                         "congl": d["CodConglomeradoPrudencial"], "x": d[x],
                         "w": d[dep].where(d[dep] > 0, 0.0)})


def _mean_ci(parts, seed: int = 0) -> dict:
    """A weighted mean, or a signed sum of weighted means over disjoint sets of banks, with the
    conglomerate wild cluster bootstrap (Webb weights, BOOT_B draws) and its bias-corrected 95%
    interval, the paper's construction. Each part is (frame with x, w, congl; sign). The draws
    perturb the linearised mean: a conglomerate contributes sum of w (x - mean) / sum of w."""
    lin = _linearize(parts)
    if lin is None:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, se=np.nan, n_cl=0)
    return _wcb(*lin, seed=seed)


def _linearize(parts):
    """(estimate, per-conglomerate influence Series) of a signed sum of weighted means; None
    when a part is empty."""
    theta, contrib = 0.0, []
    for f, sign in parts:
        f = f[np.isfinite(f["x"]) & (f["w"] > 0)]
        if f.empty:
            return None
        W = float(f["w"].sum())
        mu = float((f["w"] * f["x"]).sum() / W)
        theta += sign * mu
        contrib.append((sign * f["w"] * (f["x"] - mu) / W).groupby(f["congl"]).sum())
    return theta, pd.concat(contrib).groupby(level=0).sum()


def _wcb(theta: float, IF_s: pd.Series, seed: int = 0) -> dict:
    """Webb wild cluster bootstrap of a linearised estimate; bias-corrected 95% interval."""
    from utils.sleep_links import cluster_wild_bootstrap, _ame_band_from_draws

    IF = IF_s.to_numpy()[:, None]
    draws = {}
    cluster_wild_bootstrap(np.array([theta]), IF, lambda th: {"m": float(th[0])}, {"m": theta},
                           B=BOOT_B, scheme="webb", rng=np.random.default_rng(seed),
                           mode="normal", draws_out=draws)
    band = _ame_band_from_draws(["m"], {"m": theta}, draws["m"][:, None]).iloc[0]
    return dict(mean=theta, lo=float(band["lo_bc"]), hi=float(band["hi_bc"]),
                se=float(band["se"]), n_cl=int(IF.shape[0]))


def _by_quarter(q: pd.DataFrame, parts):
    """`parts` (over all quarters) cut into one list of parts per quarter of q."""
    by = [({k: g for k, g in f.groupby(["year", "quarter"])}, s, f.iloc[:0]) for f, s in parts]
    return [[(g.get((y, k), empty), s) for g, s, empty in by]
            for y, k in zip(q["year"], q["quarter"])]


def quarterly_ci(q: pd.DataFrame, parts) -> pd.DataFrame:
    """_mean_ci for every quarter of q; `parts` as in _mean_ci, over all quarters."""
    return pd.DataFrame([_mean_ci(sub) for sub in _by_quarter(q, parts)], index=q.index)


def average_ci(q: pd.DataFrame, parts) -> dict:
    """The mean over the quarters of q of a quarterly estimate, with one bootstrap weight per
    conglomerate across all quarters (a bank's draws are not independent over time)."""
    lins = [lin for lin in map(_linearize, _by_quarter(q, parts)) if lin is not None]
    T = len(lins)
    theta = sum(t for t, _ in lins) / T
    IF = pd.concat([s for _, s in lins]).groupby(level=0).sum() / T
    return dict(_wcb(theta, IF), n_quarters=T)


def _wstats(x, w) -> dict:
    return dict(mean=mdt._weighted_mean(x, w), sd=mdt._weighted_std(x, w),
                p25=mdt._weighted_quantile(x, 0.25, w), p75=mdt._weighted_quantile(x, 0.75, w))


def _store(section: str, payload: dict):
    """Merge one chart's numbers into the motivation sidecar, keeping the other charts'."""
    data = json.loads(MOTIV_JSON.read_text(encoding="utf-8")) if MOTIV_JSON.is_file() else {}
    data.update(schema=1, generator="make_slide_figures.py (motivation charts)",
                window=[MIN_YEAR, MAX_YEAR])
    data[section] = payload
    _dump(MOTIV_JSON, json.dumps(data, indent=2, default=float) + "\n")


def band_charts(out: pathlib.Path, png: bool):
    """Slide 2's Selic chart with a band across B banks around the B spread. The band and the
    line share Table A.4's weights, so the weighted mean of the band's distribution is the line
    itself (checked)."""
    S = mdt.ms_series(pd.read_csv(T2_CSV))
    years = S["years"]
    cells = b_spread_cells(motiv_panel())
    st = {int(y): _wstats(g["sp4"].to_numpy(), g["w"].to_numpy())
          for y, g in cells.groupby("year")}
    mean = np.array([st[int(y)]["mean"] for y in years])
    gap = float(np.nanmax(np.abs(mean - S["sp4_b"])))
    if not gap < 1e-9:
        sys.exit(f"band: weighted mean differs from Table A.4's B spread by {gap:.3e}")
    print(f"  band: weighted mean = Table A.4 B spread (max |diff| {gap:.1e})")
    spread = S["sp4_b"]
    sd = np.array([st[int(y)]["sd"] for y in years])
    variants = {
        "sd": (spread - sd, spread + sd, "±1 SD across B banks"),
        "iqr": (np.array([st[int(y)]["p25"] for y in years]),
                np.array([st[int(y)]["p75"] for y in years]), "25th–75th percentile, B banks"),
    }
    lines = _timedep_lines(S)
    for name, (lo, hi, band_label) in variants.items():
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        ax.axhline(0, color="#BFBFBA", lw=0.8, zorder=1)
        for k in ("selic", "rate", "spread"):
            y, color, ls, label, fmt = lines[k]
            ax.plot(years, y, color=color, label=label, **{**mdt.MS_LINE_KW, "linestyle": ls})
            mdt.ms_endlabel(ax, years, y, fmt.format(y[-1]), FS)
        ax.fill_between(years, lo, hi, color=mdt.B_COLOR, alpha=0.13, linewidth=0, zorder=2,
                        label=band_label)
        mdt.ms_style(ax, "Selic and B time deposits", "% p.a. (spread: pp p.a.)",
                     pix_label=True, fs=FS)
        mdt.ms_year_axis(ax, years, fs=FS)
        _legend_below(ax, 2)
        _save(fig, out, f"fig_slide_selic_timedep_all_{name}", png)
    _store("band", {"weights": "Table A.4 (build_table2): bank's type-4 share of the "
                               "market-quarter times the market-quarter's balance",
                    "by_year": {str(y): v for y, v in st.items()}})
    print("  band SD by year (pp): " + ", ".join(f"{y} {v['sd']:.3f}" for y, v in st.items()))


def quarterly_motivation() -> pd.DataFrame:
    """One row per quarter: Selic and the savings rate (macro file); the B and D type-4 spreads
    and the pooled type-5 spread (Table A.4's weights within the quarter), each with its 95%
    interval (columns <name>_lo, <name>_hi, <name>_se); balances by type and firm type.
    Memoised: two charts read it."""
    if "q" in _MOTIV_CACHE:
        return _MOTIV_CACHE["q"]
    df = motiv_panel()
    q = macro_quarterly()
    for name, kind in (("sp4_b", "time_B"), ("sp4_d", "time_D"), ("sp5", "prepaid")):
        ci = quarterly_ci(q, [(spread_cells(df, kind), 1)])
        q[name] = ci["mean"]
        for c in ("lo", "hi", "se"):
            q[f"{name}_{c}"] = ci[c]
    # The B and D means are the ones the annual check below ties to Table A.4.
    sp_b = b_spread_cells(df).groupby(["year", "quarter"]).apply(
        lambda g: mdt._weighted_mean(g["sp4"].to_numpy(), g["w"].to_numpy()))
    b_gap = float(np.nanmax(np.abs(sp_b.reindex(pd.MultiIndex.from_frame(
        q[["year", "quarter"]])).to_numpy() - q["sp4_b"].to_numpy())))
    if not b_gap < 1e-9:
        sys.exit(f"pricevol: bootstrapped B mean differs from the B spread by {b_gap:.3e}")
    t2raw = pd.read_csv(T2_CSV).set_index("var")
    p5 = spread_cells(df, "prepaid")
    p5 = p5[np.isfinite(p5["x"]) & (p5["w"] > 0)]
    p5_yr = (p5["x"] * p5["w"]).groupby(p5["year"]).sum() / p5["w"].groupby(p5["year"]).sum()
    t2_p5 = pd.to_numeric(t2raw.loc["spread_ann_a5_natl_w"], errors="coerce")
    t2_p5.index = t2_p5.index.astype(int)
    p5_gap = float(np.nanmax(np.abs(p5_yr - t2_p5.reindex(p5_yr.index))))
    if not p5_gap < 1e-9:
        sys.exit(f"pricevol: prepaid spread differs from Table A.4's national line by {p5_gap:.3e}")
    # The panel's quarterly Selic and the macro file's are one series on two routes.
    rf = df.groupby(["year", "quarter"])["risk_free_qoq"].first()
    rf_gap = float(np.max(np.abs(mdt.annualized_rate_pct(rf.reindex(
        pd.MultiIndex.from_frame(q[["year", "quarter"]])).to_numpy()) - q["selic_ann"].to_numpy())))
    print(f"  Selic, panel against macro file: max |diff| {rf_gap:.1e} pp")
    # Annual D spread from the quarterly weights must be Table A.4's D line.
    t2 = mdt.ms_series(pd.read_csv(T2_CSV))
    d_yr = d_spread(df, ["year"]).reindex(t2["years"].astype(int)).to_numpy()
    d_gap = float(np.nanmax(np.abs(d_yr - t2["sp4_d"])))
    if not d_gap < 1e-9:
        sys.exit(f"pricevol: D spread differs from Table A.4's by {d_gap:.3e}")

    tot = deposit_type_totals(refresh=False)
    wide = tot.pivot_table(index=["year", "quarter"], columns="bank_type", values=DEP_COLS,
                           aggfunc="sum")
    wide.columns = [f"{c}_{t}" for c, t in wide.columns]
    q = q.join(wide, on=["year", "quarter"])
    all_cols = [f"{c}_{t}" for c in DEP_COLS for t in ("B", "D")]
    q["dep_all"] = q[all_cols].sum(axis=1)
    q["sh_demand"] = 100 * (q["dep_a1_B"] + q["dep_a1_D"]) / q["dep_all"]
    q["sh_savings"] = 100 * (q["dep_a2_B"] + q["dep_a2_D"]) / q["dep_all"]
    q["sh_time_B"] = 100 * q["dep_a4_B"] / q["dep_all"]
    q["sh_time_D"] = 100 * q["dep_a4_D"] / q["dep_all"]
    q["sh_prepaid"] = 100 * (q["dep_a5_B"] + q["dep_a5_D"]) / q["dep_all"]
    q["d_share_time"] = 100 * q["dep_a4_D"] / (q["dep_a4_B"] + q["dep_a4_D"])
    q["sp_demand"] = q["selic_ann"]
    q["sp_savings"] = q["selic_ann"] - q["savings_ann"]
    sh_sum = q[["sh_demand", "sh_savings", "sh_time_B", "sh_time_D", "sh_prepaid"]].sum(axis=1)
    if not float(np.max(np.abs(sh_sum - 100.0))) < 1e-9:
        sys.exit("pricevol: shares do not sum to 100")
    # Prepaid balances exist only from Pix on (prepare_panel); before that the share is undefined
    # for the chart, not zero.
    q["sh_prepaid_plot"] = q["sh_prepaid"].where(q["dep_a5_B"] + q["dep_a5_D"] > 0)
    _MOTIV_CACHE["q"] = q
    return q


def tightening(q: pd.DataFrame) -> tuple[int, int]:
    """Rows of the last quarter at the window's lowest Selic target and of the first quarter at
    the highest target reached after it: the hiking cycle."""
    m = q["meta_selic"].to_numpy(dtype=float)
    i0 = int(np.flatnonzero(m == m.min())[-1])
    return i0, i0 + int(np.argmax(m[i0:]))


def _shade(ax, q, i0, i1, label=True):
    x = q["x"].to_numpy()
    ax.axvspan(x[i0], x[i1], color=mdt.NATL_COLOR, alpha=0.07, linewidth=0, zorder=0)
    if label:
        ax.text((x[i0] + x[i1]) / 2, 0.97, "Selic hikes", transform=ax.get_xaxis_transform(),
                fontsize=8 * FS, color=mdt.MUTED_INK, va="top", ha="center")


def _quarter_axis(ax):
    mdt.ms_year_axis(ax, np.arange(MIN_YEAR, MAX_YEAR + 1, dtype=float), fs=FS)


def _end_labels(ax, x_end, items, min_gap=0.07):
    """ms_endlabel for several lines at once: labels whose ends sit closer than `min_gap` (a
    fraction of the axis height) are pushed apart symmetrically, and a text that another line
    already prints (two lines ending at the same value) is printed once. `items`: (y_end, text)."""
    lo, hi = ax.get_ylim()
    h_pts = ax.figure.get_size_inches()[1] * 72 * ax.get_position().height
    pts = sorted({t: (float(y), t) for y, t in items if np.isfinite(y)}.values())
    fr = [(y - lo) / (hi - lo) for y, _ in pts]
    adj = list(fr)
    # Repeat until no pair is closer than min_gap: pushing one pair apart can close the next.
    for _ in range(100):
        moved = False
        for i in range(1, len(adj)):
            short = min_gap - (adj[i] - adj[i - 1])
            if short > 1e-9:
                adj[i - 1] -= short / 2
                adj[i] += short / 2
                moved = True
        if not moved:
            break
    for (y, text), f0, f1 in zip(pts, fr, adj):
        ax.annotate(text, xy=(x_end, y), xytext=(5, (f1 - f0) * h_pts),
                    textcoords="offset points", fontsize=8.5 * FS, color=mdt.MUTED_INK,
                    va="center", ha="left")


def _style_no_pix(ax, title, ylabel, xlabel):
    """ms_style's styling for a chart whose x-axis is not calendar time, so without the Pix
    rule."""
    ax.set_title(title, fontsize=10.5 * FS, loc="left", pad=6, color="#111111")
    ax.set_ylabel(ylabel, fontsize=9 * FS, color=mdt.AXIS_INK)
    ax.set_xlabel(xlabel, fontsize=9 * FS, color=mdt.MUTED_INK)
    ax.grid(True, axis="y", color=mdt.GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(mdt.AXIS_INK)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(labelsize=9 * FS, colors=mdt.AXIS_INK, length=3)


def pricevol_charts(out: pathlib.Path, png: bool):
    """Three charts for the prices-against-balances slide, on the same quarters and the same
    shading of the hiking cycle: the spread of each deposit type (2a), each type's share of
    balances (2b), and the deposit rates themselves next to Selic (2c). The bank means (time
    deposits of B and of D firms, prepaid) carry their 95% intervals; demand and savings pay
    national, rule-set rates and carry none. The shares are totals over every institution, not
    estimates, so they carry none either."""
    q = quarterly_motivation()
    x = q["x"].to_numpy()
    i0, i1 = tightening(q)
    kw = dict(linewidth=2, solid_capstyle="round", zorder=3)
    selic = q["selic_ann"].to_numpy(dtype=float)
    # key, colour, line style, label, spread column, spread has an interval, share column
    series = (
        ("demand", DEMAND_COLOR, "-", "Demand", "sp_demand", False, "sh_demand"),
        ("savings", SAVINGS_COLOR, "-", "Savings", "sp_savings", False, "sh_savings"),
        ("time_B", mdt.B_COLOR, "-", "Time, B firms", "sp4_b", True, "sh_time_B"),
        ("time_D", mdt.D_COLOR, (0, (5, 2)), "Time, D firms", "sp4_d", True, "sh_time_D"),
        ("prepaid", PREPAID_COLOR, PREPAID_LS, "Prepaid (from 2020Q4)", "sp5", True,
         "sh_prepaid_plot"),
    )

    def col(c):
        return q[c].to_numpy(dtype=float)

    def spread(s):
        return col(s[4])

    def spread_ci(s):
        return col(f"{s[4]}_lo"), col(f"{s[4]}_hi")

    def rate(s):
        return selic - col(s[4])

    def rate_ci(s):
        lo, hi = spread_ci(s)
        return selic - hi, selic - lo

    pct1 = lambda v: f"{_plain(v, 1)}%"
    charts = (
        ("fig_slide_motiv_spreads_by_type", "Spread by deposit type", "pp p.a.", spread,
         spread_ci, lambda v: _plain(v, 1), True, False),
        ("fig_slide_motiv_shares_by_type", "Share of balances by deposit type", "% of balances",
         lambda s: col(s[6]), None, lambda v: f"{v:.0f}%", False, False),
        ("fig_slide_motiv_rates_by_type", "Deposit rate by type", "% p.a.", rate, rate_ci, pct1,
         True, True),
    )
    for stem, title, ylabel, value, ci, fmt, zero, with_selic in charts:
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        if zero:
            ax.axhline(0, color="#BFBFBA", lw=0.8, zorder=1)
        _shade(ax, q, i0, i1)
        ends = []
        if with_selic:
            ax.plot(x, selic, color=mdt.NATL_COLOR, label="Selic", **kw)
            ends.append((selic[-1], fmt(selic[-1])))
        for s in series:
            y = value(s)
            if ci is not None and s[5]:
                lo, hi = ci(s)
                ax.fill_between(x, lo, hi, color=s[1], alpha=0.16, linewidth=0, zorder=2)
            ax.plot(x, y, color=s[1], label=s[3], **{**kw, "linestyle": s[2]})
            ends.append((y[-1], fmt(y[-1])))
        if ci is not None:
            ax.fill_between([], [], [], color=mdt.NATL_COLOR, alpha=0.16, linewidth=0,
                            label="95% interval, bank means")
        _style_no_pix(ax, title, ylabel, "")
        _end_labels(ax, x[-1], ends)
        _quarter_axis(ax)
        _legend_below(ax, 3)
        _save(fig, out, stem, png)

    def at(c, i):
        return float(q[c].iloc[i])

    def ci_at(c, i):
        return [at(f"{c}_lo", i), at(f"{c}_hi", i)]
    qlab = [f"{int(y)}Q{int(k)}" for y, k in zip(q["year"], q["quarter"])]
    sav_bal = (q["dep_a2_B"] + q["dep_a2_D"]).to_numpy()
    p0 = int(np.flatnonzero(np.isfinite(col("sh_prepaid_plot")))[0])
    n_q = i1 - i0
    payload = {
        "hikes": {
            "start": qlab[i0], "end": qlab[i1], "n_quarters": n_q,
            "selic_target": [at("meta_selic", i0), at("meta_selic", i1)],
            "spread_demand": [at("sp_demand", i0), at("sp_demand", i1)],
            "spread_savings": [at("sp_savings", i0), at("sp_savings", i1)],
            "spread_time_B": [at("sp4_b", i0), at("sp4_b", i1)],
            "spread_time_D": [at("sp4_d", i0), at("sp4_d", i1)],
            "spread_prepaid": [at("sp5", i0), at("sp5", i1)],
            "share_demand": [at("sh_demand", i0), at("sh_demand", i1)],
            "share_savings": [at("sh_savings", i0), at("sh_savings", i1)],
            "share_time_B": [at("sh_time_B", i0), at("sh_time_B", i1)],
            "share_time_D": [at("sh_time_D", i0), at("sh_time_D", i1)],
            "share_prepaid": [at("sh_prepaid", i0), at("sh_prepaid", i1)],
            "ci": {k: {"start": ci_at(c, i0), "end": ci_at(c, i1)}
                   for k, c in (("spread_time_B", "sp4_b"), ("spread_time_D", "sp4_d"),
                                ("spread_prepaid", "sp5"))},
            "savings_share_change_per_quarter_pp":
                (at("sh_savings", i1) - at("sh_savings", i0)) / n_q,
            "savings_share_change_per_quarter_pp_before":
                (at("sh_savings", i0) - at("sh_savings", 0)) / i0,
            "savings_share_change_per_quarter_pp_after":
                (at("sh_savings", -1) - at("sh_savings", i1)) / (len(q) - 1 - i1),
            "demand_share_change_per_quarter_pp":
                (at("sh_demand", i1) - at("sh_demand", i0)) / n_q,
            "savings_balance_change_pct": 100 * (sav_bal[i1] / sav_bal[i0] - 1),
            "savings_balance_min_change_pct": 100 * (sav_bal[i0:].min() / sav_bal[i0] - 1),
        },
        "window": {
            "start": qlab[0], "end": qlab[-1],
            "share_demand": [at("sh_demand", 0), at("sh_demand", -1)],
            "share_savings": [at("sh_savings", 0), at("sh_savings", -1)],
            "share_time_B": [at("sh_time_B", 0), at("sh_time_B", -1)],
            "share_time_D": [at("sh_time_D", 0), at("sh_time_D", -1)],
        },
        "prepaid": {
            "first": qlab[p0],
            "share": [at("sh_prepaid", p0), at("sh_prepaid", -1)],
            "spread_mean": float(np.nanmean(col("sp5"))),
            "spread_abs_max": float(np.nanmax(np.abs(col("sp5")))),
        },
        "ci_halfwidth_mean": {k: float(np.nanmean((col(f"{c}_hi") - col(f"{c}_lo")) / 2))
                              for k, c in (("time_B", "sp4_b"), ("time_D", "sp4_d"),
                                           ("prepaid", "sp5"))},
        "ci_method": "conglomerate wild cluster bootstrap, Webb, B=999, bias-corrected 95% "
                     "percentile interval of the weighted mean, per quarter",
        "series": {"quarter": qlab, **{c: [None if not np.isfinite(v) else float(v)
                                          for v in col(c)]
                                      for c in ("sp4_b", "sp4_b_lo", "sp4_b_hi", "sp4_d",
                                                "sp4_d_lo", "sp4_d_hi", "sp5", "sp5_lo",
                                                "sp5_hi")}},
    }
    _store("pricevol", payload)
    h = payload["hikes"]
    print(f"  hikes {h['start']}-{h['end']} ({n_q} quarters): savings spread "
          f"{h['spread_savings'][0]:.3f} -> {h['spread_savings'][1]:.3f} pp, savings share "
          f"{h['share_savings'][0]:.3f} -> {h['share_savings'][1]:.3f}%, savings balance "
          f"{h['savings_balance_change_pct']:.3f}%")
    print("  mean 95% half-width (pp): " + ", ".join(
        f"{k} {v:.3f}" for k, v in payload["ci_halfwidth_mean"].items())
          + f"; prepaid spread |max| {payload['prepaid']['spread_abs_max']:.4f} pp")


def bank_quarter_b() -> pd.DataFrame:
    """B conglomerates by quarter: type-4 balance summed over markets and the bank's national
    type-4 spread, flagged when the rate is the CDI fallback or the 3 x Selic guard."""
    df = motiv_panel()
    b = df[df["bank_type"] == "B"]
    bq = b.groupby(["CodConglomeradoPrudencial", "year", "quarter"], as_index=False).agg(
        dep_a4=("dep_a4", "sum"), sp=("spread_a4_national", "first"),
        r4=("rate_a4_national", "first"), guard=("rate_a4_guarded", "first"),
        rf=("risk_free_qoq", "first"))
    mq = macro_quarterly()[["year", "quarter", "cdi_qoq"]]
    bq = bq.merge(mq, on=["year", "quarter"], how="left")
    bq["own_rate"] = ~(np.abs(bq["r4"] - bq["cdi_qoq"] / 100.0) < 1e-9) & (bq["guard"] != 1)
    bq["sp_pp"] = mdt.annualized_spread_pp(bq["sp"], bq["rf"])
    bq["t"] = bq["year"] * 4 + bq["quarter"] - 1
    return bq


def _fe_slope(d: pd.DataFrame, ycol: str, xcol: str, seed: int = 0) -> dict:
    """OLS of y on x with bank and quarter fixed effects; the conglomerate wild cluster
    bootstrap (Webb, B = 999) and its bias-corrected percentile interval, as in the paper's
    linear columns. Also returns x and y net of the fixed effects (Frisch-Waugh)."""
    import statsmodels.api as sm
    from utils.sleep_links import linear_wild_cluster_bootstrap

    d = d.reset_index(drop=True)
    fe = pd.get_dummies(d[["bank", "tq"]].astype(str), drop_first=True, dtype=float)
    Z = sm.add_constant(fe)
    X = pd.concat([d[[xcol]].astype(float), Z], axis=1)
    y = d[ycol].astype(float)
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["bank"].to_numpy()})
    band = {}
    linear_wild_cluster_bootstrap(res, B=999, scheme="webb", seed=seed, mode="normal",
                                  band_out=band)
    row = band["band"].set_index("name").loc[xcol]
    Zn = Z.to_numpy()
    proj = Zn @ np.linalg.lstsq(Zn, np.column_stack([d[xcol], y]), rcond=None)[0]
    xr = d[xcol].to_numpy(dtype=float) - proj[:, 0]
    yr = y.to_numpy() - proj[:, 1]
    slope = float(res.params[xcol])
    fwl = float(xr @ yr / (xr @ xr))
    if not abs(fwl - slope) < 1e-8:
        sys.exit(f"repricing: FWL slope {fwl:.6f} differs from the OLS slope {slope:.6f}")
    return dict(slope=slope, lo_bc=float(row["lo_bc"]), hi_bc=float(row["hi_bc"]),
                se_boot=float(row["se"]), p_bc=float(row["p_bc"]), stars=str(row["stars"]),
                n=int(len(d)), banks=int(d["bank"].nunique()), xr=xr, yr=yr)


def reprice_binscatter(out: pathlib.Path, png: bool):
    """B banks: the change in a bank's own type-4 spread from one quarter to the next against
    the growth of its time deposits over the following four quarters. The rate is interest
    expense over the lagged balance, so growth within the quarter of the change would move the
    measured rate itself; reading growth from that quarter onward avoids it."""
    bq = bank_quarter_b().sort_values(["CodConglomeradoPrudencial", "t"]).reset_index(drop=True)
    prev = bq.groupby("CodConglomeradoPrudencial")[["t", "sp_pp", "own_rate", "dep_a4"]].shift(1)
    consecutive = prev["t"] == bq["t"] - 1
    bq["d_sp"] = np.where(consecutive, bq["sp_pp"] - prev["sp_pp"], np.nan)
    bq["own_prev"] = consecutive & (prev["own_rate"] == True)  # noqa: E712 (object after shift)
    with np.errstate(divide="ignore", invalid="ignore"):
        bq["growth_now"] = 100 * np.log(bq["dep_a4"] / prev["dep_a4"].where(consecutive))
    ahead = bq[["CodConglomeradoPrudencial", "t", "dep_a4"]].rename(
        columns={"dep_a4": "dep_ahead"})
    ahead["t"] = ahead["t"] - GROWTH_H
    bq = bq.merge(ahead, on=["CodConglomeradoPrudencial", "t"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        bq["growth_ahead"] = 100 * np.log(bq["dep_ahead"] / bq["dep_a4"])
    base = bq[bq["own_rate"] & bq["own_prev"] & (bq["dep_a4"] >= TYPE4_FLOOR)
              & np.isfinite(bq["d_sp"])].copy()
    base["bank"] = base["CodConglomeradoPrudencial"]
    base["tq"] = base["t"]

    def trimmed(d, ycol):
        d = d[np.isfinite(d[ycol])]
        lo, hi = np.percentile(d["d_sp"], TRIM_PCT)
        ylo, yhi = np.percentile(d[ycol], TRIM_PCT)
        return d[d["d_sp"].between(lo, hi) & d[ycol].between(ylo, yhi)]

    full = base[np.isfinite(base["growth_ahead"])]
    main = trimmed(base, "growth_ahead")
    r = _fe_slope(main, "growth_ahead", "d_sp")
    r_untrim = _fe_slope(full, "growth_ahead", "d_sp")
    r_now = _fe_slope(trimmed(base, "growth_now"), "growth_now", "d_sp")

    # Where the B conglomerate-quarters of the window go, filter by filter.
    win = bq[bq["year"].between(MIN_YEAR, MAX_YEAR)]
    steps = []
    for label, d in (
            ("B conglomerate-quarters with time deposits", win[win["dep_a4"] > 0]),
            ("own rate this quarter (no CDI fallback, no 3 x Selic guard)",
             win[(win["dep_a4"] > 0) & win["own_rate"]]),
            ("own rate in the previous quarter too (spread change defined)",
             win[(win["dep_a4"] > 0) & win["own_rate"] & win["own_prev"]
                 & np.isfinite(win["d_sp"])]),
            (f"type-4 book of at least R$ {TYPE4_FLOOR / 1e6:,.0f} million", base),
            (f"balance observed {GROWTH_H} quarters later", full),
            (f"changes and growth inside the {TRIM_PCT[0]:g}st-{TRIM_PCT[1]:g}th percentiles",
             main)):
        steps.append(dict(step=label, rows=int(len(d)),
                          banks=int(d["CodConglomeradoPrudencial"].nunique())))

    # Each dot is a bin: the bank-quarters sorted by the spread change net of the fixed effects
    # and cut into N_BINS groups of equal size; the dot sits at the group's mean change and mean
    # growth, both net of the fixed effects with the sample means added back.
    xr = r["xr"] + main["d_sp"].mean()
    yr = r["yr"] + main["growth_ahead"].mean()
    bins = pd.qcut(xr, N_BINS, labels=False, duplicates="drop")
    bx = pd.Series(xr).groupby(bins).mean().to_numpy()
    by = pd.Series(yr).groupby(bins).mean().to_numpy()
    per_bin = pd.Series(xr).groupby(bins).size()

    for stem, points in (("fig_slide_motiv_reprice_binscatter", False),
                         ("fig_slide_motiv_reprice_binscatter_points", True)):
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        ax.axvline(0, color="#BFBFBA", lw=0.8, zorder=1)
        if points:
            ax.scatter(xr, yr, s=6, color=mdt.B_COLOR, alpha=0.18, linewidth=0, zorder=2,
                       label="Each bank-quarter")
        xx = np.array([bx.min(), bx.max()])
        # The statistics ride in the legend under the plot: any spot inside it can hold a bin.
        ax.scatter(bx, by, s=34, color=mdt.B_COLOR, edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"{r['n']:,} bank-quarters of {r['banks']} banks, in {len(bx)} "
                         f"bins of about {int(round(per_bin.mean()))}")
        ax.plot(xx, yr.mean() + r["slope"] * (xx - xr.mean()), color=mdt.NATL_COLOR, lw=1.6,
                zorder=4, label=f"Slope {_plain(r['slope'], 3)}{r['stars']} % per pp, 95% "
                                f"interval [{_plain(r['lo_bc'], 3)}, {_plain(r['hi_bc'], 3)}]")
        _style_no_pix(ax, "Repricing and deposit growth, B banks",
                      f"Time-deposit growth, next {GROWTH_H} quarters (%)",
                      "Change in own time-deposit spread (pp p.a.)")
        ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1,
                  frameon=False, handlelength=1.8)
        _save(fig, out, stem, png)

    def clean(d):
        return {k: v for k, v in d.items() if k not in ("xr", "yr")}
    _store("reprice", {
        "sample": "B conglomerate-quarters with an own type-4 rate in both quarters (not the CDI "
                  "fallback, not the 3 x Selic guard), type-4 balance at least TYPE4_FLOOR",
        "type4_floor": TYPE4_FLOOR, "trim_pct": list(TRIM_PCT), "growth_h": GROWTH_H,
        "units": "slope in percent of time deposits per pp of annual spread",
        "fe": "bank and quarter", "inference": "conglomerate WCB, Webb, B=999, BC percentile",
        "main": clean(r), "untrimmed": clean(r_untrim), "same_quarter": clean(r_now),
        "n_before_trim": int(len(full)),
        "waterfall": steps,
        "bins": {"n": int(len(bx)), "rows_min": int(per_bin.min()),
                 "rows_max": int(per_bin.max())},
        "within_bank_sd_median": {
            "d_sp": float(main.groupby("bank")["d_sp"].std().median()),
            "growth_ahead": float(main.groupby("bank")["growth_ahead"].std().median())},
    })
    print(f"  repricing slope {r['slope']:.4f} [{r['lo_bc']:.4f}, {r['hi_bc']:.4f}] "
          f"N={r['n']} banks={r['banks']}; untrimmed {r_untrim['slope']:.4f} "
          f"(N={r_untrim['n']}); same quarter {r_now['slope']:.4f}")
    for s in steps:
        print(f"    {s['rows']:>6,} rows, {s['banks']:>3} banks  {s['step']}")


def dpremium_chart(out: pathlib.Path, png: bool):
    """D firms' type-4 premium over B firms (B spread minus D spread) above their share of
    time-deposit balances, quarter by quarter. The premium is a difference of two bank means and
    carries its 95% interval (B and D conglomerates bootstrapped together); the share is a ratio
    of totals over every institution, not an estimate, and carries none."""
    q = quarterly_motivation()
    df = motiv_panel()
    x = q["x"].to_numpy()
    parts = [(spread_cells(df, "time_B"), 1), (spread_cells(df, "time_D"), -1)]
    ci = quarterly_ci(q, parts)
    prem = (q["sp4_b"] - q["sp4_d"]).to_numpy(dtype=float)
    gap = float(np.nanmax(np.abs(ci["mean"].to_numpy() - prem)))
    if not gap < 1e-9:
        sys.exit(f"dpremium: bootstrapped premium differs from B minus D by {gap:.3e}")
    avg = average_ci(q, parts)
    share = q["d_share_time"].to_numpy(dtype=float)
    kw = dict(linewidth=2, solid_capstyle="round", zorder=3, color=mdt.D_COLOR)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(SLIDE_FIGSIZE[0], SLIDE_FIGSIZE[1] * 1.25),
                                 sharex=True)
    a1.axhline(0, color="#BFBFBA", lw=0.8, zorder=1)
    a1.fill_between(x, ci["lo"], ci["hi"], color=mdt.D_COLOR, alpha=0.16, linewidth=0, zorder=2,
                    label="95% interval")
    a1.plot(x, prem, **kw)
    a1.legend(fontsize=8 * FS, loc="upper right", frameon=False)
    mdt.ms_endlabel(a1, x, prem, f"{_plain(prem[-1], 1)} pp", FS)
    mdt.ms_style(a1, "D firms' time-deposit premium over B firms", "pp p.a.", pix_label=True,
                 fs=FS, pix_x=PIX_X_QUARTERLY)
    a2.plot(x, share, **kw)
    mdt.ms_endlabel(a2, x, share, f"{share[-1]:.1f}%", FS)
    mdt.ms_style(a2, "D firms' share of time-deposit balances", "%", fs=FS,
                 pix_x=PIX_X_QUARTERLY)
    a2.set_ylim(bottom=0)
    _quarter_axis(a2)
    fig.tight_layout(h_pad=1.0)
    _save(fig, out, "fig_slide_motiv_dfirm_premium", png)

    d = df[df["bank_type"] == "D"]
    w = d["dep_a4"].where(d["dep_a4"] > 0, 0.0)
    cdi = macro_quarterly().set_index(["year", "quarter"])["cdi_qoq"] / 100.0
    cdi_row = cdi.reindex(pd.MultiIndex.from_frame(d[["year", "quarter"]])).to_numpy()
    fallback = np.abs(d["rate_a4_national"].to_numpy() - cdi_row) < 1e-9
    yrs = q["year"].to_numpy()
    _store("dpremium", {
        "premium_mean_pp": float(np.nanmean(prem)),
        "premium_mean_ci": [avg["lo"], avg["hi"]],
        "premium_mean_se": avg["se"],
        "premium_ci_halfwidth_mean": float(np.nanmean((ci["hi"] - ci["lo"]) / 2)),
        "quarters_ci_excludes_zero": int(((ci["lo"] > 0) | (ci["hi"] < 0)).sum()),
        "n_quarters": int(len(q)),
        "premium_mean_by_year_pp": {str(int(y)): float(np.nanmean(prem[yrs == y]))
                                    for y in np.unique(yrs)},
        "d_share_time_pct": [float(share[0]), float(share[-1])],
        "d_share_time_change_pp_per_year": float((share[-1] - share[0]) / (len(share) / 4)),
        "d_balance_share_with_cdi_rate": float(w[fallback].sum() / w.sum()),
        "d_rowshare_with_cdi_rate": float(fallback.mean()),
    })
    print(f"  D premium mean {np.nanmean(prem):.3f} pp [{avg['lo']:.3f}, {avg['hi']:.3f}]; "
          f"quarters whose interval excludes zero "
          f"{int(((ci['lo'] > 0) | (ci['hi'] < 0)).sum())} of {len(q)}; D share of time deposits "
          f"{share[0]:.3f}% -> {share[-1]:.3f}%; D type-4 balance at the CDI rate "
          f"{100 * w[fallback].sum() / w.sum():.3f}%")


def entry_gap_chart(out: pathlib.Path, png: bool):
    """Figure 2's B entry events, split by the entrant's type-4 price against the incumbents' at
    entry. Share paths are rebuilt from the market panel as sleep_ident_entry_dynamics builds
    them (deposits of types 1, 2 and 4 over the B firms of the market) and summarised by its
    own event_paths with its cached settings; the pooled events must reproduce its CSV."""
    import sleep_ident_entry_dynamics as sed

    meta = json.loads((D6_DIR / "d6_meta.json").read_text(encoding="utf-8"))
    H, plateau_w, boot, seed = meta["horizon"], meta["plateau_w"], meta["boot"], meta["seed"]
    ev = pd.read_csv(D6_DIR / "d6_entry_events.csv", dtype={"congl": str, "mkt": str})

    raw = pd.read_parquet(PANEL_PARQUET, columns=["CodConglomeradoPrudencial", "CODMUN_IBGE",
                                                  "mca_code", "year", "quarter", "dep_a1",
                                                  "dep_a2", "dep_a4", "spread_a4",
                                                  "risk_free_qoq"])
    raw["qidx"] = raw["year"].astype(int) * 4 + raw["quarter"].astype(int) - 1
    if meta.get("window_q"):                    # paths stop where Figure 2's stop
        raw = raw[raw["qidx"] <= int(meta["window_q"][1])]
    raw["dep"] = raw[sed.DEP_COLS].fillna(0).sum(axis=1)
    raw = raw[raw["dep"] > 0].copy()
    raw["congl"] = raw["CodConglomeradoPrudencial"].astype(str)
    raw["sp4"] = mdt.annualized_spread_pp(raw["spread_a4"], raw["risk_free_qoq"])
    is_d = raw["CODMUN_IBGE"].astype(str).str.strip().isin(["0", "0.0"])
    b = raw[~is_d].copy()
    b["mkt"] = b["mca_code"].astype(str)
    b["share"] = b["dep"] / b.groupby(["mkt", "qidx"])["dep"].transform("sum")
    share = b.set_index(["congl", "mkt", "qidx"])["share"]
    qmax = int(b["qidx"].max())

    evb = ev[ev["kind"] == "B"].copy()
    paths, gaps = {}, []
    for e in evb.itertuples():
        q0 = int(e.q_entry)
        idx = pd.MultiIndex.from_tuples([(e.congl, e.mkt, q0 + h) for h in range(H + 1)
                                         if q0 + h <= qmax])
        paths[(e.congl, e.mkt)] = pd.Series(share.reindex(idx).to_numpy(),
                                            index=idx.get_level_values(2))
        m = b[(b["mkt"] == e.mkt) & (b["qidx"] == q0)] if e.keep else None
        gap = np.nan
        if m is not None:
            own = m[m["congl"] == e.congl]
            riv = m[(m["congl"] != e.congl) & (m["dep_a4"] > 0) & m["sp4"].notna()]
            if len(own) and len(riv):
                gap = float(own["sp4"].iloc[0] - np.average(riv["sp4"], weights=riv["dep_a4"]))
        gaps.append(gap)
    evb["gap"] = gaps

    ref = pd.read_csv(D6_DIR / "d6_entry_paths.csv")
    ref = ref[(ref["kind"] == "B") & (ref["norm"] == "main")].reset_index(drop=True)
    pooled = sed.event_paths(evb, paths, H, "main", plateau_w, boot, seed)
    cols = ["h", "n", "median", "p25", "p75", "ci_lo", "ci_hi"]
    diff = float(np.max(np.abs(pooled[cols].to_numpy(float) - ref[cols].to_numpy(float))))
    if not diff < 1e-12:
        sys.exit(f"entrygap: rebuilt pooled paths differ from d6_entry_paths.csv by {diff:.3e}")
    print(f"  entrygap: pooled B paths = d6_entry_paths.csv (max |diff| {diff:.1e}, "
          f"n = {int(pooled['n'].iloc[0])})")

    kept = evb[evb["keep"]]
    more = kept["gap"] <= -ENTRY_GAP_PP
    groups = {"more": kept[more], "rest": kept[~more & kept["gap"].notna()]}
    med = float(kept["gap"].median())
    groups_med = {"below_median": kept[kept["gap"] <= med], "above_median": kept[kept["gap"] > med]}
    res = {k: sed.event_paths(v.assign(keep=True), paths, H, "main", plateau_w, boot, seed)
           for k, v in {**groups, **groups_med}.items()}

    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    style = {"more": (mdt.B_COLOR, "-", f"Pays ≥ {ENTRY_GAP_PP:.0f} pp more than incumbents"),
             "rest": (GAP_REST_COLOR, (0, (5, 2)), "Pays less, or under 1 pp more")}
    for k, (color, ls, label) in style.items():
        p = res[k]
        ax.fill_between(p["h"], p["ci_lo"], p["ci_hi"], color=color, alpha=0.15, linewidth=0,
                        zorder=1)
        ax.plot(p["h"], p["median"], color=color, linestyle=ls, linewidth=2, marker="o",
                markersize=4, markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                label=f"{label} (n = {int(p['n'].iloc[0])})")
    _style_no_pix(ax, "B entrants' market share, by price at entry", "Normalized market share",
                  "Quarters since entry")
    ax.set_xticks(range(0, H + 1, 2))
    ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1,
              frameon=False, handlelength=1.8)
    _save(fig, out, "fig_slide_motiv_entry_by_gap", png)

    # The same events without any normalisation: the entrant's share of the B firms' deposits in
    # its market (percent), median by quarter since entry with the same bootstrap of the median.
    # Only the events the normalised chart keeps are used, so both charts describe one sample.
    level = {}
    for k in ("more", "rest"):
        reg = groups[k].assign(keep=True)
        _, rows = sed._path_matrix(reg, paths, H, "main", plateau_w, want_rows=True)
        M = []
        for _, e in reg.iloc[rows].iterrows():
            s = paths[(e["congl"], e["mkt"])].astype(float).to_numpy()
            v = np.full(H + 1, np.nan)
            v[:min(len(s), H + 1)] = s[:H + 1]
            M.append(100.0 * v)
        med, lo, hi, n = sed._bootstrap_median_ci(np.vstack(M), H, boot, seed)
        level[k] = pd.DataFrame({"h": np.arange(H + 1), "median": med, "ci_lo": lo,
                                 "ci_hi": hi, "n": n})
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    for k, (color, ls, label) in style.items():
        p = level[k]
        ax.fill_between(p["h"], p["ci_lo"], p["ci_hi"], color=color, alpha=0.15, linewidth=0,
                        zorder=1)
        ax.plot(p["h"], p["median"], color=color, linestyle=ls, linewidth=2, marker="o",
                markersize=4, markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                label=f"{label} (n = {int(p['n'].iloc[0])})")
    _style_no_pix(ax, "B entrants' share of B deposits, by price at entry",
                  "Share of B deposits in the market (%)", "Quarters since entry")
    ax.set_xticks(range(0, H + 1, 2))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1,
              frameon=False, handlelength=1.8)
    _save(fig, out, "fig_slide_motiv_entry_by_gap_level", png)

    def at_h(p, hs=(4, 8, 12)):
        return {str(h): float(p.loc[p["h"] == h, "median"].iloc[0]) for h in hs
                if (p["h"] == h).any()}

    # D entries: national, priced against the B firms' balance-weighted spread of the quarter.
    evd = ev[(ev["kind"] == "D") & ev["keep"]]
    dnat = raw[is_d].set_index(["congl", "qidx"])["sp4"]
    bw = b[(b["dep_a4"] > 0) & b["sp4"].notna()]
    b_mean = (bw["sp4"] * bw["dep_a4"]).groupby(bw["qidx"]).sum() / bw.groupby("qidx")[
        "dep_a4"].sum()
    d_gap = np.array([dnat.get((e.congl, int(e.q_entry)), np.nan) - b_mean.get(int(e.q_entry),
                                                                               np.nan)
                      for e in evd.itertuples()], dtype=float)
    d_gap = d_gap[np.isfinite(d_gap)]
    g = kept["gap"].dropna()
    _store("entrygap", {
        "threshold_pp": ENTRY_GAP_PP, "norm": "main (s_h - s_0)/(s_end - s_0)",
        "d6_settings": {"horizon": H, "plateau_w": plateau_w, "boot": boot, "seed": seed},
        "b_events_kept": int(len(kept)), "b_events_with_gap": int(len(g)),
        "b_gap_pp": {"mean": float(g.mean()), "median": float(g.median()),
                     "p25": float(g.quantile(0.25)), "p75": float(g.quantile(0.75)),
                     "share_pays_more": float((g < 0).mean()),
                     "share_pays_1pp_more": float((g <= -ENTRY_GAP_PP).mean())},
        "share_definition": "deposits of types 1, 2 and 4 of the entrant over those of every B "
                            "firm in the market (MCA) and quarter, from the panel's balances",
        "bands": "95% bootstrap interval of the median: events resampled with replacement, "
                 "999 draws, seed 0 (sleep_ident_entry_dynamics.event_paths)",
        "groups": {k: {"n": int(p["n"].iloc[0]), "median_at_h": at_h(p)} for k, p in res.items()},
        "pooled_median_at_h": at_h(pooled),
        "level_pct": {k: {"n": int(p["n"].iloc[0]), "median_at_h": at_h(p, (0, 4, 8, 12)),
                          "ci_at_h12": [float(p["ci_lo"].iloc[-1]), float(p["ci_hi"].iloc[-1])]}
                      for k, p in level.items()},
        "d_events": int(len(d_gap)),
        "d_gap_vs_b_mean_pp": {"median": float(np.median(d_gap)),
                               "share_pays_more": float((d_gap < 0).mean())},
    })
    for k, p in res.items():
        print(f"  entrygap {k}: n={int(p['n'].iloc[0])}, median at h=4/8/12: "
              + ", ".join(f"{v:.3f}" for v in at_h(p).values()))
    for k, p in level.items():
        print(f"  entrygap level {k}: n={int(p['n'].iloc[0])}, median share (%) at h=0/4/8/12: "
              + ", ".join(f"{v:.3f}" for v in at_h(p, (0, 4, 8, 12)).values()))


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
    if "violations" in todo:
        violation_bars(args.out, args.png)
    if "flow" in todo:
        flow_diagram(args.out, args.routine)
    if "roadmap" in todo:
        roadmap_strip(args.out)
    if "table" in todo:
        deposit_types_table(args.out, args.refresh_shares, args.decimals)
    if "band" in todo:
        band_charts(args.out, args.png)
    if "pricevol" in todo:
        pricevol_charts(args.out, args.png)
    if "reprice" in todo:
        reprice_binscatter(args.out, args.png)
    if "dpremium" in todo:
        dpremium_chart(args.out, args.png)
    if "entrygap" in todo:
        entry_gap_chart(args.out, args.png)


if __name__ == "__main__":
    main()
