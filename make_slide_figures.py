"""Slide visuals for the Deposit Competition deck, drawn from the outputs the paper is built on.

No number is typed here: every value comes from an estimation output or from the cached input
of a paper table. Files go to the paper folder (Drafts/Deposit Competition), which the deck's
\\graphicspath and \\input@path already point at:

  fig_market_structure_by_year_{a..f}.pdf  each panel of the paper's Figure 1, drawn alone
  fig_slide_selic_timedep_all.pdf          Selic, the B time-deposit rate and its spread
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
  fig_slide_selic_timedep_all_sd.pdf       the Selic chart with a band of +/-1 SD across B banks
                                           around the B spread
  fig_slide_motiv_spreads_by_type.pdf      spread of each deposit type, by quarter
  fig_slide_motiv_shares_by_type.pdf       each deposit type's share of balances, by quarter
  fig_slide_motiv_rates_by_type.pdf        each deposit type's rate next to Selic, by quarter
  fig_slide_motiv_reprice_binscatter.pdf   B banks: change in own spread against deposit growth
                                           (_points: every bank-quarter behind the bins)
  fig_slide_motiv_dfirm_premium.pdf        D firms' rate premium over B firms and their share
  fig_slide_motiv_entry_by_gap.pdf         B entrants' share paths, by price at entry
                                           (_level: the share of B deposits, not normalised)
                                           (_bankboot: bands from resampling banks)
  fig_slide_motiv_reprice_binscatter_d.pdf the repricing chart for D firms with an own rate
  fig_slide_relprice_*.pdf                 relative prices across B banks: decomposition,
                                           paths of the five largest, persistence, share change
  tab_motiv_*.tex                          the motivation note's ten tables, booktabs bodies
                                           built from the sidecar (--only motivtables)
  fig_paper_*.pdf                          paper copies of four motivation charts, drawn at a
                                           0.49\textwidth panel's printed size (--only paper)

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
from utils import prepaid as _prepaid
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
           "table", "band", "pricevol", "reprice", "dpremium", "entrygap", "relprice", "anova",
           "motivtables", "paper", "slidesize", "slidesize2", "slidesize3", "slidesize4", "xi", "slidesize5", "slidesize6", "depanova", "ref", "followups", "refame", "pixexposure")

# Paper copies of four motivation charts, for a 2x2 figure of 0.49\textwidth panels in V_Main
# (12pt, letterpaper, 1in margins: 0.49 * 6.5in = 3.19in). As with fig_entry_dynamics_a/b, each
# panel is drawn AT its printed width with fonts sized for it, not shrunk from the slide size.
# `--only paper` redraws the four charts with these settings; every other chart the same
# functions draw is discarded, and the sidecar is not rewritten.
PAPER_FIGSIZE = (3.19, 2.45)   # starting size; _save fits the saved panel to PAPER_PRINT
PAPER_PRINT = (3.19, 2.9)
PAPER_FS = 0.9
PAPER_STEMS = {
    "fig_slide_motiv_rates_by_type_noshade": "fig_paper_motiv_rates_by_type",   # panel (a)
    "fig_slide_relprice_share_xs_pp": "fig_paper_relprice_share_xs",            # panel (b)
    "fig_slide_motiv_entry_by_gap_bankboot": "fig_paper_entry_by_gap",          # panel (c)
    "fig_slide_motiv_dfirm_premium": "fig_paper_dfirm_premium",                 # panel (d)
}
PAPER = False
MARK = 1.0       # scatter-area multiplier (dot sizes scale with the figure's area)

# Deck copies (`_slide`) of five motivation charts, each drawn AT its printed size in the
# 10pt beamer deck (text block 307.29 x 244.69 pt = 4.27 x 3.40 in), so text prints at its
# nominal size. FS 1.18 sets legends and end labels to 10.0 pt; _save then sets titles to
# SLIDE_PT["title"], axis labels and ticks to theirs, and wraps titles and legend entries to
# the slot. `--only slidesize` redraws the five; every other chart the same functions draw is
# discarded, and the sidecar is not rewritten. Printed size per file (in):
DECK_TW, DECK_TH = 307.29 / 72, 244.69 / 72
SLOT = (8.0 / 2.54, 5.5 / 2.54)
SLIDE_STEMS = {
    "fig_slide_motiv_rates_by_type_noshade": ("fig_slide_motiv_rates_by_type_slide", SLOT),
    "fig_slide_motiv_shares_by_type": ("fig_slide_motiv_shares_by_type_slide", SLOT),
    "fig_slide_relprice_share_xs_pp": ("fig_slide_relprice_share_xs_pp_slide",
                                       (5.8 / 2.54, 7.0 / 2.54)),
    "fig_slide_motiv_entry_by_gap_bankboot": ("fig_slide_motiv_entry_by_gap_bankboot_slide", SLOT),
    "fig_slide_motiv_dfirm_premium": ("fig_slide_motiv_dfirm_premium_slide", SLOT),
}
SLIDE_FS = 1.18
SLIDE_PT = dict(title=10.5, label=10.0, tick=9.5)
SLIDE = False
# Second deck layout (`_slide2`, `--only slidesize2`): the same fonts in the room a slide leaves
# beside one or two lines of text (at most 10.8 x 5.0 cm). Layout changes, set where each chart
# is drawn: the cross-section keeps only the fitted line in its legend, inside the plot; the
# entry chart hangs its legend to the right; the D-premium panels sit side by side; and
# Figure 1's panels (a), (c), (d) form one strip.
SLIDE2_STEMS = {
    "fig_slide_relprice_share_xs_pp": ("fig_slide_relprice_share_xs_pp_slide2",
                                       (6.2 / 2.54, 5.0 / 2.54)),
    "fig_slide_motiv_entry_by_gap_bankboot": ("fig_slide_motiv_entry_by_gap_bankboot_slide2",
                                              (10.8 / 2.54, 5.0 / 2.54)),
    "fig_slide_motiv_dfirm_premium": ("fig_slide_motiv_dfirm_premium_slide2",
                                      (10.8 / 2.54, 4.6 / 2.54)),
    "fig_market_structure_by_year": ("fig_market_structure_by_year_slide2",
                                     (10.8 / 2.54, 4.4 / 2.54)),
}
SLIDE2 = False
# Third deck layout, one chart per file at 8.0 x 4.6 cm beside a bullet: Figure 1's panels (a),
# (c), (d) (`--only slidesize3`) and the demand-residual variance split (`--only xi`).
SLIDE3_STEMS = {
    "fig_market_structure_hhi": ("fig_market_structure_hhi_slide", (8.0 / 2.54, 4.6 / 2.54)),
    "fig_market_structure_dcount": ("fig_market_structure_dcount_slide",
                                    (8.0 / 2.54, 4.6 / 2.54)),
    "fig_market_structure_bmca": ("fig_market_structure_bmca_slide", (8.0 / 2.54, 4.6 / 2.54)),
    # Two across the text width (slide 4): 5.3 x 4.2 cm each.
    "fig_market_structure_dcount_half": ("fig_market_structure_dcount_slide_half",
                                         (5.3 / 2.54, 4.2 / 2.54)),
    "fig_market_structure_bmca_half": ("fig_market_structure_bmca_slide_half",
                                       (5.3 / 2.54, 4.2 / 2.54)),
    "fig_slide_xi_anova": ("fig_slide_xi_anova_slide", (8.0 / 2.54, 4.6 / 2.54)),
    "fig_slide_dep_anova": ("fig_slide_dep_anova_slide", (8.0 / 2.54, 4.6 / 2.54)),
}
DEP_ANOVA_JSON = mdt.OUTPUT_DIR / "dep_anova.json"
XI_JSON = mdt.OUTPUT_DIR / "xi_diagnostics.json"
SLIDE3 = False
# Fourth deck layout (`--only slidesize4`): the two results figures of routine (III) at the
# size the deck gives them.
SLIDE4_STEMS = {
    "fig_slide_sleep_ame_E3": ("fig_slide_sleep_ame_E3_slide", (9.0 / 2.54, 4.8 / 2.54)),
    "fig_slide_elasticity_wedge_E3": ("fig_slide_elasticity_wedge_E3_slide",
                                      (8.0 / 2.54, 3.2 / 2.54)),
}
SLIDE4 = False
# Fifth deck layout (`--only slidesize5`), both 10.8 x 4.6 cm: the AME chart with its labels in
# a column left of the plot, and the D-premium panels side by side with one-line titles.
SLIDE5_STEMS = {
    "fig_slide_sleep_ame_E3": ("fig_slide_sleep_ame_E3_slide2", (10.8 / 2.54, 4.6 / 2.54)),
    "fig_slide_motiv_dfirm_premium": ("fig_slide_motiv_dfirm_premium_slide3",
                                      (10.8 / 2.54, 4.6 / 2.54)),
}
SLIDE5 = False
# Sixth deck layout (`--only slidesize6`): spreads and shares by type, two across the text width.
SLIDE6_STEMS = {
    "fig_slide_motiv_spreads_by_type": ("fig_slide_motiv_spreads_by_type_slide_half",
                                        (5.3 / 2.54, 4.6 / 2.54)),
    "fig_slide_motiv_shares_by_type": ("fig_slide_motiv_shares_by_type_slide_half",
                                       (5.3 / 2.54, 4.6 / 2.54)),
}
SLIDE6 = False
# Reference-style deck copies (`--only ref`, suffix `_ref`): the paper-style charts' own look
# (thin lines, end labels in the line's colour, the hike band, legend below, no markers),
# drawn at their printed size with no text under REF_MIN_PT. Printed size per file (in):
REF_STEMS = {
    "fig_slide_motiv_spreads_by_type": ("fig_slide_motiv_spreads_by_type_ref",
                                        (6.3 / 2.54, 4.7 / 2.54)),
    "fig_slide_motiv_shares_by_type": ("fig_slide_motiv_shares_by_type_ref",
                                       (6.3 / 2.54, 4.7 / 2.54)),
    "fig_market_structure_hhi": ("fig_market_structure_hhi_ref", (10.8 / 2.54, 3.8 / 2.54)),
    "fig_slide_motiv_dfirm_premium": ("fig_slide_motiv_dfirm_premium_ref",
                                      (6.5 / 2.54, 5.0 / 2.54)),
    "fig_slide_relprice_decomp_time": ("fig_slide_relprice_decomp_time_ref",
                                       (10.8 / 2.54, 4.6 / 2.54)),
    "fig_slide_motiv_entry_by_gap_bankboot": ("fig_slide_motiv_entry_by_gap_bankboot_ref",
                                              (10.8 / 2.54, 3.9 / 2.54)),
    # The 2026-10-20 follow-ups (`--only followups`, Rout/followups_1020.json).
    "fig_slide_phi_anova": ("fig_slide_phi_anova_ref", (8.0 / 2.54, 4.6 / 2.54)),
    "fig_slide_entry_carry": ("fig_slide_entry_carry_ref", (10.8 / 2.54, 4.6 / 2.54)),
    "fig_slide_fee_ratio": ("fig_slide_fee_ratio_ref", (10.8 / 2.54, 4.6 / 2.54)),
    # `--only refame`: the routine (III) AME chart, thin whiskers and a dot, no bars.
    "fig_slide_sleep_ame_E3": ("fig_slide_sleep_ame_E3_ref", (10.8 / 2.54, 4.6 / 2.54)),
    # Pix usage on pre-launch exposure (`--only pixexposure`, Rout/pix_exposure.json).
    "fig_slide_pix_exposure": ("fig_slide_pix_exposure_ref", (10.8 / 2.54, 4.6 / 2.54)),
}
FOLLOWUPS_JSON = mdt.OUTPUT_DIR / "followups_1020.json"
PIX_EXPOSURE_JSON = mdt.OUTPUT_DIR / "pix_exposure.json"
REF = False
REF_FS = 1.0
REF_MIN_PT = 8.0
# Deck files drawn without the Pix rule and its label.
NO_PIX = ("fig_market_structure_hhi", "fig_market_structure_dcount_half",
          "fig_market_structure_bmca_half")
AME_ROWS_SLIDE5 = {
    "interaction_cadunico_families_per1000": "CadÚnico families\n(100 per 1,000)",
    "interaction_fraction_65plus": "Share aged 65+\n(pp)",
    "interaction_risk_free_qoq_lag": "Lagged Selic\n(pp per quarter)",
    "interaction_connections_per100": "Mobile lines\n(per 100)",
    "interaction_pix_exists": "Pix available\n(0 to 1)",
}
# One-line row labels of the AME chart's deck copy (each sits above its bar).
AME_ROWS_SLIDE = {
    "interaction_cadunico_families_per1000": "CadÚnico families (100 per 1,000)",
    "interaction_fraction_65plus": "Share aged 65+ (pp)",
    "interaction_risk_free_qoq_lag": "Lagged Selic (pp per quarter)",
    "interaction_connections_per100": "Mobile lines (per 100 inhabitants)",
    "interaction_pix_exists": "Pix available (0 to 1)",
}
SLIDE_MIN_PT = 9.0   # no text in a deck copy prints smaller


def paper_figures(out: pathlib.Path, png: bool):
    global SLIDE_FIGSIZE, FS, PAPER, MARK
    SLIDE_FIGSIZE, FS, PAPER = PAPER_FIGSIZE, PAPER_FS, True
    MARK = (PAPER_FIGSIZE[0] * PAPER_FIGSIZE[1]) / (5.2 * 3.3)
    pricevol_charts(out, png)
    relprice_charts(out, png)
    entry_gap_chart(out, png)
    dpremium_chart(out, png)


def slide_size_figures(out: pathlib.Path, png: bool):
    """The `_slide` copies (SLIDE_STEMS). Same data, samples, weights, intervals and colours as
    the charts they copy; the rates chart drops the prepaid line and the bands, the shares chart
    the prepaid line."""
    global SLIDE_FIGSIZE, FS, PAPER, SLIDE, MARK
    FS, PAPER, SLIDE = SLIDE_FS, False, True
    # Each chart starts at its slot size; _save trims the margins to fit the slot exactly.
    for draw, stem in ((pricevol_charts, "fig_slide_motiv_shares_by_type"),
                       (relprice_charts, "fig_slide_relprice_share_xs_pp"),
                       (entry_gap_chart, "fig_slide_motiv_entry_by_gap_bankboot"),
                       (dpremium_chart, "fig_slide_motiv_dfirm_premium")):
        w, h = SLIDE_STEMS[stem][1]
        SLIDE_FIGSIZE = (w, h / 1.25) if draw is dpremium_chart else (w, h)
        MARK = (w * h) / (5.2 * 3.3)
        draw(out, png)


def slide2_figures(out: pathlib.Path, png: bool):
    """The `_slide2` layouts (SLIDE2_STEMS), at the `_slide` fonts. Same data, samples,
    weights, intervals and colours as the charts they copy."""
    global SLIDE_FIGSIZE, FS, PAPER, SLIDE, SLIDE2, MARK
    FS, PAPER, SLIDE, SLIDE2 = SLIDE_FS, False, True, True
    for draw, stem in ((relprice_charts, "fig_slide_relprice_share_xs_pp"),
                       (entry_gap_chart, "fig_slide_motiv_entry_by_gap_bankboot"),
                       (dpremium_chart, "fig_slide_motiv_dfirm_premium"),
                       (market_structure_strip, "fig_market_structure_by_year")):
        SLIDE_FIGSIZE = SLIDE2_STEMS[stem][1]
        MARK = (SLIDE_FIGSIZE[0] * SLIDE_FIGSIZE[1]) / (5.2 * 3.3)
        draw(out, png)
    SLIDE2 = False


def market_structure_strip(out: pathlib.Path, png: bool):
    """Figure 1's panels (a), (c) and (d) side by side, with short titles, years at the ends
    and the middle of the window, and the HHI legend inside its panel."""
    S = mdt.ms_series(pd.read_csv(T2_CSV))
    # The HHI panel is a little wider: its legend lives inside it.
    fig, axes = plt.subplots(1, 3, figsize=SLIDE_FIGSIZE,
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    titles = {"a": "Deposit HHI", "c": "Digital firms", "d": "B firms per MCA"}
    yrs = S["years"]
    ticks = [yrs[0], yrs[len(yrs) // 2], yrs[-1]]
    for ax, key in zip(axes, ("a", "c", "d")):
        mdt.MS_PANELS[key](ax, S, fs=FS, lettered=False, pix_label=key == "a")
        ax.set_title(titles[key], loc="left", fontsize=SLIDE_PT["title"], color="#111111")
        ax.set_xticks(ticks)
        # Three year labels touch at this width: the ends are labelled, 2020 is the Pix rule.
        ax.set_xticklabels([str(int(ticks[0])), "", str(int(ticks[-1]))])
        # A third of the slide holds the plot only: the titles name each series, so the y
        # labels go, and so do the end-of-line values (the y axis gives the levels).
        ax.set_ylabel("")
        for t in list(ax.texts):
            if isinstance(t, matplotlib.text.Annotation):
                t.remove()
    a = axes[0]
    # The threshold names do not fit either; the dotted rules stay at their levels.
    for t in list(a.texts):
        if t.get_text().startswith(("2,500", "1,500")):
            t.remove()
    # Room under both lines for the legend, inside the panel.
    lo_, hi_ = a.get_ylim()
    a.set_ylim(lo_ - 0.55 * (hi_ - lo_), hi_)
    handles, _ = a.get_legend_handles_labels()
    leg = a.legend(handles, ["B, local", "B+D, national"], fontsize=SLIDE_MIN_PT,
                   loc="lower left", frameon=False, handlelength=0.8, handletextpad=0.3,
                   borderaxespad=0.2, labelspacing=0.2)
    # It sits inside the panel, so the layout should not reserve room for it beside the plot.
    leg.set_in_layout(False)
    _save(fig, out, "fig_market_structure_by_year", png)


def slide3_figures(out: pathlib.Path, png: bool):
    """The single market-structure panels (SLIDE3_STEMS), at the `_slide` fonts."""
    global SLIDE_FIGSIZE, FS, PAPER, SLIDE, SLIDE3, MARK
    FS, PAPER, SLIDE, SLIDE3 = SLIDE_FS, False, True, True
    SLIDE_FIGSIZE = SLIDE3_STEMS["fig_market_structure_hhi"][1]
    MARK = (SLIDE_FIGSIZE[0] * SLIDE_FIGSIZE[1]) / (5.2 * 3.3)
    S = mdt.ms_series(pd.read_csv(T2_CSV))
    for key, stem, title in (("a", "fig_market_structure_hhi", "Deposit concentration (HHI)"),
                             ("c", "fig_market_structure_dcount", "Digital (D) firms, national"),
                             ("d", "fig_market_structure_bmca", "B firms per MCA (mean)"),
                             ("c", "fig_market_structure_dcount_half", "D firms, national"),
                             ("d", "fig_market_structure_bmca_half", "B firms per MCA (mean)")):
        SLIDE_FIGSIZE = SLIDE3_STEMS[stem][1]
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        # Pix label: at the top of the rule, except in the HHI panel, where it sits in the gap
        # between the two lines (the top holds the B line and the 2,500 rule).
        mdt.MS_PANELS[key](ax, S, fs=FS, lettered=False, pix_label=0.72 if key == "a" else True)
        ax.set_title(title, loc="left", fontsize=SLIDE_PT["title"], color="#111111")
        mdt.ms_year_axis(ax, S["years"], fs=FS)
        yrs = S["years"][::4] if stem.endswith("_half") else S["years"][::2]
        ax.set_xticks(yrs)
        ax.set_xticklabels([str(int(y)) for y in yrs])
        ax.set_xlabel("")
        if stem.endswith("_half"):   # half the text width: the title names the series
            ax.set_ylabel("")
        if stem in NO_PIX:
            _drop_pix(ax)
        if key == "a":
            # Short names for the two thresholds, and room under both lines for the legend.
            for t in ax.texts:
                if t.get_text().startswith("2,500"):
                    t.set_text("2,500 high")
                elif t.get_text().startswith("1,500"):
                    t.set_text("1,500 moderate")
            lo_, hi_ = ax.get_ylim()
            ax.set_ylim(lo_ - 0.5 * (hi_ - lo_), hi_)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=8.5 * FS, loc="lower left", frameon=False,
                      handlelength=1.2, handletextpad=0.4, borderaxespad=0.2, labelspacing=0.2)
        _save(fig, out, stem, png)
    SLIDE3 = False


def slide4_figures(out: pathlib.Path, png: bool):
    """The routine (III) results figures at deck size (SLIDE4_STEMS), at the `_slide` fonts."""
    global SLIDE_FIGSIZE, FS, PAPER, SLIDE, SLIDE4, MARK
    FS, PAPER, SLIDE, SLIDE4 = SLIDE_FS, False, True, True
    SLIDE_FIGSIZE = SLIDE4_STEMS["fig_slide_sleep_ame_E3"][1]
    ame_chart(out, png, 3)
    SLIDE_FIGSIZE = SLIDE4_STEMS["fig_slide_elasticity_wedge_E3"][1]
    elasticity_bars(out, png, 3)
    SLIDE4 = False


def ref_figures(out: pathlib.Path, png: bool):
    """The reference-style deck copies (REF_STEMS). Same data, samples, weights, intervals and
    colours as the charts they copy."""
    global SLIDE_FIGSIZE, FS, PAPER, SLIDE, REF, MARK
    FS, PAPER, SLIDE, REF = REF_FS, False, False, True
    for draw, stem in ((pricevol_charts, "fig_slide_motiv_spreads_by_type"),
                       (hhi_ref, "fig_market_structure_hhi"),
                       (dpremium_chart, "fig_slide_motiv_dfirm_premium"),
                       (relprice_charts, "fig_slide_relprice_decomp_time"),
                       (entry_gap_chart, "fig_slide_motiv_entry_by_gap_bankboot")):
        w, h = REF_STEMS[stem][1]
        SLIDE_FIGSIZE = (w, h / 1.25) if draw is dpremium_chart else (w, h)
        MARK = (w * h) / (5.2 * 3.3)
        draw(out, png)
    REF = False


def followups_ref(out: pathlib.Path, png: bool, E: int):
    """The 2026-10-20 follow-up charts in the reference style (REF_STEMS), from
    Rout/followups_1020.json (make_followups_1020.py): the variance split of phi-hat_mt for
    routine E (bars, as fig_slide_xi_anova_slide); entry and the incumbents' carry (event
    study, all entries and without Santander's); the fee ratio of incumbents and D firms."""
    global SLIDE_FIGSIZE, FS, PAPER, SLIDE, REF, MARK
    if not FOLLOWUPS_JSON.is_file():
        sys.exit(f"{FOLLOWUPS_JSON} not found: run make_followups_1020.py first")
    sc = json.loads(FOLLOWUPS_JSON.read_text(encoding="utf-8"))
    FS, PAPER, SLIDE, REF = REF_FS, False, False, True

    # Variance split of phi-hat_mt.
    c1 = sc[f"E{E}"]["C1"]
    SLIDE_FIGSIZE = REF_STEMS["fig_slide_phi_anova"][1]
    MARK = (SLIDE_FIGSIZE[0] * SLIDE_FIGSIZE[1]) / (5.2 * 3.3)
    parts = (("quarter", "Quarter", mdt.NATL_COLOR), ("market", "Market (MCA)", SAVINGS_COLOR),
             ("rest", "Rest", "#E0E0E0"))
    rows = (("Population\nweights", c1["pop"]), ("Deposit\nweights", c1["dep"]),
            ("Without\n" + r"$\hat{\phi}=1$", c1["pop_noceil"]))
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    for i, (_lab, vals) in enumerate(rows):
        left = 0.0
        for k, name, color in parts:
            v = 100.0 * vals[k]
            ax.barh(i, v, left=left, color=color, edgecolor="white", height=0.6,
                    label=name if i == 0 else None)
            if v >= 7:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5 * FS,
                        color="#111111" if color == "#E0E0E0" else "white")
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    _style_no_pix(ax, "Variance of the sleepy share, %", "", "")
    ax.grid(False)
    ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=False, handlelength=1.0, handletextpad=0.4, columnspacing=0.8)
    _save(fig, out, "fig_slide_phi_anova", png)

    # Entry and the incumbents' carry.
    c2 = sc["C2"]
    SLIDE_FIGSIZE = REF_STEMS["fig_slide_entry_carry"][1]
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    for tag, lab, color, dx in (("all", f"All entries ({c2['all']['n_markets']} markets)",
                                 mdt.B_COLOR, -0.12),
                                ("no_santander", "Without Santander's "
                                 f"({c2['no_santander']['n_markets']} markets)", mdt.D_COLOR, 0.12)):
        hs = sorted(c2[tag]["h"], key=int)
        x = np.array([int(h) for h in hs]) + dx
        est = np.array([c2[tag]["h"][h]["estimate"] for h in hs])
        lo = np.array([c2[tag]["h"][h]["ci"]["lo"] for h in hs])
        hi = np.array([c2[tag]["h"][h]["ci"]["hi"] for h in hs])
        ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="o", ms=2.5, lw=0.8, capsize=0,
                    color=color, label=lab)
    ax.axhline(0, color=mdt.AXIS_INK, lw=0.6)
    ax.axvline(-0.5, color=mdt.MUTED_INK, lw=0.6, ls="--")
    _style_no_pix(ax, "Incumbents' carry around entry", "Change in carry",
                  "Quarters since entry (8 = 8 or later; omitted: 5 or more before)")
    ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2,
              frameon=False, handlelength=1.0)
    _save(fig, out, "fig_slide_entry_carry", png)

    # Fee ratio.
    c6 = sc["C6"]
    SLIDE_FIGSIZE = REF_STEMS["fig_slide_fee_ratio"][1]
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    for key, lab, color in (("incumbent", "Incumbents", mdt.B_COLOR),
                            ("digital", "D firms", mdt.D_COLOR)):
        ys = sorted(c6[key], key=int)
        x = np.array([int(y) for y in ys])
        v = np.array([c6[key][y] for y in ys])
        ax.plot(x, v, color=color, lw=1.2)
        ax.text(x[-1] + 0.15, v[-1], lab, color=color, fontsize=8.5 * FS, va="center")
    ax.axvline(2020 + 10.5 / 12, color=mdt.MUTED_INK, lw=0.6, ls="--")
    ax.set_xlim(2015.7, 2026.3)
    ax.set_xticks(range(2016, 2025, 2))
    _style_no_pix(ax, "Fee revenue over deposits, median, pp a year", "", "")
    # The note for the slide (no room in the slot): the effective price of a deposit is the
    # spread plus fees, and the demand model's price omits fee changes.
    _save(fig, out, "fig_slide_fee_ratio", png)
    REF = False


def ref_ame(out: pathlib.Path, png: bool):
    """The routine (III) AME chart in the reference style (REF_STEMS)."""
    global SLIDE_FIGSIZE, FS, PAPER, SLIDE, REF
    FS, PAPER, SLIDE, REF = REF_FS, False, False, True
    SLIDE_FIGSIZE = REF_STEMS["fig_slide_sleep_ame_E3"][1]
    ame_chart(out, png, 3)
    REF = False


def pix_exposure_ref(out: pathlib.Path, png: bool, measure: str = "payers"):
    """Yearly first-stage slopes of Pix usage (distinct individual payers per inhabitant) on each
    standardized 2019 exposure, with 95% intervals (MCA clusters), in the reference style
    (Rout/pix_exposure.json, written by make_pix_exposure.py with tab_pix_exposure_fs.tex)."""
    global SLIDE_FIGSIZE, FS, PAPER, SLIDE, REF, MARK
    if not PIX_EXPOSURE_JSON.is_file():
        sys.exit(f"{PIX_EXPOSURE_JSON} not found: run make_pix_exposure.py first")
    fs = json.loads(PIX_EXPOSURE_JSON.read_text(encoding="utf-8"))["fs"][measure]
    FS, PAPER, SLIDE, REF = REF_FS, False, False, True
    SLIDE_FIGSIZE = REF_STEMS["fig_slide_pix_exposure"][1]
    MARK = (SLIDE_FIGSIZE[0] * SLIDE_FIGSIZE[1]) / (5.2 * 3.3)
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    for (e, lab, color), dx in zip((("mandated", "Mandated share", mdt.B_COLOR),
                                    ("mobile", "Mobile lines", SAVINGS_COLOR),
                                    ("g45", "4G/5G share", mdt.D_COLOR)), (-0.15, 0.0, 0.15)):
        ys = sorted(fs[e]["yearly"], key=int)
        x = np.array([int(y) for y in ys]) + dx
        v = fs[e]["yearly"]
        est = np.array([v[y]["slope"] for y in ys])
        lo = np.array([v[y]["lo"] for y in ys])
        hi = np.array([v[y]["hi"] for y in ys])
        ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="o", ms=2.5, lw=0.8, capsize=0,
                    color=color, label=lab)
    ax.axhline(0, color=mdt.AXIS_INK, lw=0.6)
    ys = sorted(fs["mandated"]["yearly"], key=int)
    ax.set_xticks([int(y) for y in ys])
    ax.set_xticklabels(["2020Q4" if y == "2020" else y for y in ys])
    _style_no_pix(ax, "Pix payers per inhabitant on 2019 exposure", "Slope, 1 SD", "")
    ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
              frameon=False, handlelength=1.0)
    _save(fig, out, "fig_slide_pix_exposure", png)
    REF = False


def hhi_ref(out: pathlib.Path, png: bool):
    """Figure 1's concentration panel alone, without the Pix rule and without markers."""
    S = mdt.ms_series(pd.read_csv(T2_CSV))
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    mdt.MS_PANELS["a"](ax, S, fs=FS, lettered=False, pix_label=False)
    _drop_pix(ax)
    for ln in ax.lines:
        ln.set_marker("")
        if ln.get_linestyle() == "-":
            ln.set_linewidth(1.2)
    # Between the two lines on the right, clear of both and of the 1,500 rule's label.
    ax.legend(fontsize=8.5 * FS, loc="center right", bbox_to_anchor=(1.0, 0.6), frameon=False,
              handlelength=1.2, handletextpad=0.4, labelspacing=0.25)
    mdt.ms_year_axis(ax, S["years"], fs=FS)
    ax.set_xlabel("")
    _save(fig, out, "fig_market_structure_hhi", png)


def slide6_figures(out: pathlib.Path, png: bool):
    """The sixth deck layout (SLIDE6_STEMS), at the `_slide` fonts."""
    global SLIDE_FIGSIZE, FS, PAPER, SLIDE, SLIDE6, MARK
    FS, PAPER, SLIDE, SLIDE6 = SLIDE_FS, False, True, True
    SLIDE_FIGSIZE = SLIDE6_STEMS["fig_slide_motiv_spreads_by_type"][1]
    pricevol_charts(out, png)
    SLIDE6 = False


def _drop_pix(ax):
    """Remove the dashed Pix rule and its label."""
    for ln in list(ax.lines):
        x = np.unique(np.asarray(ln.get_xdata(), dtype=float))
        if len(x) == 1 and ln.get_linestyle() == "--":
            ln.remove()
    for t in list(ax.texts):
        if t.get_text() == "Pix":
            t.remove()


def slide5_figures(out: pathlib.Path, png: bool):
    """The fifth deck layout (SLIDE5_STEMS), at the `_slide` fonts."""
    global SLIDE_FIGSIZE, FS, PAPER, SLIDE, SLIDE2, SLIDE5, MARK
    FS, PAPER, SLIDE, SLIDE5 = SLIDE_FS, False, True, True
    SLIDE_FIGSIZE = SLIDE5_STEMS["fig_slide_sleep_ame_E3"][1]
    ame_chart(out, png, 3)
    # The D-premium panels use the side-by-side layout of the second deck layout.
    SLIDE2 = True
    SLIDE_FIGSIZE = SLIDE5_STEMS["fig_slide_motiv_dfirm_premium"][1]
    dpremium_chart(out, png)
    SLIDE2 = SLIDE5 = False


def _forest_slide(out, png, stem, labels, est, lo, hi, values, intervals, xlabel):
    """_forest's deck copy: each row's label above its bar, inside the plot, so the plot takes
    the width a label column would; the estimate and its interval on one line at the right."""
    est, lo, hi = (np.asarray(a, dtype=float) for a in (est, lo, hi))
    ypos = np.arange(len(labels))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    ax.axvline(0, color="#8A8A85", lw=0.9, zorder=1)
    ax.barh(ypos, est, height=0.26, color=mdt.NATL_COLOR, zorder=2)
    ax.errorbar(est, ypos, xerr=np.vstack([est - lo, hi - est]), fmt="none", ecolor="#111111",
                elinewidth=2.0, capsize=3, capthick=2.0, zorder=3)
    ax.set_yticks([])
    # The top row's label must stay inside the axes: the limit leaves a label's height above it.
    ax.set_ylim(-0.4, len(labels) - 1 + 0.78)
    for y, lab in zip(ypos, labels):
        ax.text(0.0, y + 0.16, lab, transform=ax.get_yaxis_transform(), fontsize=8.5 * FS,
                color=mdt.AXIS_INK, va="bottom", ha="left", zorder=4)
    for y, v, i in zip(ypos, values, intervals):
        ax.text(1.02, y, f"{v} {i}", transform=ax.get_yaxis_transform(), fontsize=SLIDE_MIN_PT,
                color=mdt.AXIS_INK, va="center", ha="left")
    # Set from the plot's left end, the label stays inside the slot.
    ax.set_xlabel(xlabel, fontsize=8.5 * FS, color=mdt.AXIS_INK, loc="left")
    ax.grid(True, axis="x", color=mdt.GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(mdt.AXIS_INK)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", labelsize=9 * FS, colors=mdt.AXIS_INK, length=3)
    _save(fig, out, stem, png)


def _slide_text(fig, target):
    """Deck copies: titles, axis labels and ticks at SLIDE_PT; every title, x label or legend
    entry wider than the slot, and every y label taller than about half of it, broken at the
    space (or ', ') nearest its middle."""
    width_in, height_in = target
    r = fig.canvas.get_renderer()

    def fit(t, size, room_in, vertical=False):
        t.set_fontsize(size)
        s = t.get_text()
        ext = t.get_window_extent(r)
        span = (ext.height if vertical else ext.width) / fig.dpi
        if s and "\n" not in s and " " in s and span > room_in:
            cuts = [i for i, c in enumerate(s) if c == " "]
            c = min(cuts, key=lambda i: abs(i - len(s) / 2))
            t.set_text(s[:c] + "\n" + s[c + 1:])

    # Panels side by side share the width, panels stacked share the height.
    n_cols = len({round(ax.get_position().x0, 2) for ax in fig.axes})
    n_rows = len({round(ax.get_position().y0, 2) for ax in fig.axes})
    for ax in fig.axes:
        # The charts title with loc="left", which matplotlib keeps apart from ax.title. A left
        # title runs from the axes' left edge, so it gets the slot less the y-axis margin.
        for t in (ax.title, ax._left_title, ax._right_title):
            if SLIDE5:
                t.set_fontsize(SLIDE_PT["label"])
            else:
                fit(t, SLIDE_PT["title"], 0.85 * width_in / n_cols)
        fit(ax.xaxis.label, SLIDE_PT["label"], 0.9 * width_in / n_cols)
        fit(ax.yaxis.label, SLIDE_PT["label"], 0.5 * height_in / n_rows, vertical=True)
        ax.tick_params(labelsize=SLIDE_PT["tick"])
    _wrap_legend(fig, width=int(width_in * 72 / (0.52 * 8.5 * FS)))
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text() and t.get_fontsize() < SLIDE_MIN_PT:
            t.set_fontsize(SLIDE_MIN_PT)


def _clear_legends(fig, gap_pt: float = 4.0):
    """A legend hung below its axes (loc 'upper center', anchored under the x-axis) is placed
    with its top `gap_pt` under the tick labels and the x label."""
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or leg._loc != 9:   # 9 = 'upper center'
            continue
        floor = ax.xaxis.get_tightbbox(r).y0 - gap_pt * fig.dpi / 72
        leg.set_bbox_to_anchor((0.5, (floor - ax.bbox.y0) / ax.bbox.height),
                               transform=ax.transAxes)


def _save(fig, out: pathlib.Path, stem: str, png: bool):
    pad = 0.1
    if PAPER or SLIDE or REF:
        table = (REF_STEMS if REF else SLIDE6_STEMS if SLIDE6 else SLIDE5_STEMS if SLIDE5 else SLIDE4_STEMS if SLIDE4 else
                 SLIDE3_STEMS if SLIDE3 else
                 SLIDE2_STEMS if SLIDE2 else SLIDE_STEMS)
        stems = PAPER_STEMS if PAPER else {k: v[0] for k, v in table.items()}
        if stem not in stems:
            plt.close(fig)
            return
        target = PAPER_PRINT if PAPER else table[stem][1]
        stem = stems[stem]
        # The tight crop adds the legend and end labels outside the axes. Shrink the figure by
        # that excess (text keeps its size) until the saved file is exactly `target` inches,
        # so it prints at 100%: the paper's four share one height in the 2x2 grid, and each
        # deck copy fills its slot at its nominal font sizes.
        pad = 0.02
        if SLIDE:
            _slide_text(fig, target)
        elif REF:
            # Entries wrap to the width of one of the two legend columns.
            _wrap_legend(fig, width=int(target[0] * 72 / (2 * 0.55 * 8.5 * FS)))
            for t in fig.findobj(matplotlib.text.Text):
                if t.get_text() and t.get_fontsize() < REF_MIN_PT:
                    t.set_fontsize(REF_MIN_PT)
            # A title wider than most of the slot breaks at the space nearest its middle.
            r = fig.canvas.get_renderer()
            for ax in fig.axes:
                for t in (ax.title, ax._left_title):
                    txt = t.get_text()
                    if (txt and "\n" not in txt and " " in txt
                            and t.get_window_extent(r).width / fig.dpi > 0.92 * target[0]):
                        cuts = [i for i, c in enumerate(txt) if c == " "]
                        c = min(cuts, key=lambda i: abs(i - len(txt) / 2))
                        t.set_text(txt[:c] + "\n" + txt[c + 1:])
                yl = ax.yaxis.label
                txt = yl.get_text()
                if (txt and "\n" not in txt and " " in txt
                        and yl.get_window_extent(r).height / fig.dpi > 0.55 * target[1]):
                    cuts = [i for i, c in enumerate(txt) if c == " "]
                    c = min(cuts, key=lambda i: abs(i - len(txt) / 2))
                    yl.set_text(txt[:c] + "\n" + txt[c + 1:])
        else:
            _wrap_legend(fig)
        for _ in range(8 if SLIDE or REF else 4):
            if len(fig.axes) > 1:
                fig.tight_layout(h_pad=1.0)
            if SLIDE or REF:
                _clear_legends(fig)
            bb = fig.get_tightbbox(fig.canvas.get_renderer())
            w, h = fig.get_size_inches()
            w, h = w - (bb.width + 2 * pad - target[0]), h - (bb.height + 2 * pad - target[1])
            if w <= 0 or h <= 0:
                sys.exit(f"{stem}: the text outside the axes is larger than the {target[0]:.2f} x "
                         f"{target[1]:.2f} in slot")
            fig.set_size_inches(w, h)
        if SLIDE or REF:   # the last resize moved the axes: hang the legends again
            if len(fig.axes) > 1:
                fig.tight_layout(h_pad=1.0)
            _clear_legends(fig)
    p = out / f"{stem}.pdf"
    fig.savefig(p, bbox_inches="tight", pad_inches=pad, facecolor="white")
    if png:
        fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight", pad_inches=pad,
                    facecolor="white")
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
    if PAPER or REF:   # a ~3in panel holds two legend columns, not three
        ncol = min(ncol, 2)
    if REF:   # short swatches, so two columns fit a third of the page
        ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=ncol,
                  frameon=False, columnspacing=0.8, handlelength=1.2, handletextpad=0.4,
                  labelspacing=0.25, borderaxespad=0.0, borderpad=0.0)
        return
    if SLIDE:   # 10pt entries in a 3.15in slot (9 pt at half width): two columns
        ax.legend(fontsize=SLIDE_MIN_PT if SLIDE6 else 8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                  ncol=min(ncol, 2), frameon=False, columnspacing=0.6 if SLIDE6 else 1.0,
                  handlelength=0.9 if SLIDE6 else 1.4,
                  handletextpad=0.4, labelspacing=0.25, borderaxespad=0.0, borderpad=0.0)
        return
    ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=ncol,
              frameon=False, columnspacing=1.4, handlelength=1.8)


def _wrap_legend(fig, width=34):
    """Paper panels: a legend entry longer than `width` characters breaks into two lines at the
    ', ' or ' (' nearest its middle, so no entry is wider than the panel."""
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        for t in leg.get_texts():
            s = t.get_text()
            if len(s) <= width or "\n" in s:
                continue
            cuts = [i + 1 for i in range(len(s) - 1) if s[i:i + 2] == ", "]
            cuts += [i for i in range(len(s) - 1) if s[i:i + 2] == " ("]
            if cuts:
                c = min(cuts, key=lambda i: abs(i - len(s) / 2))
                t.set_text(s[:c].rstrip() + "\n" + s[c:].lstrip())


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
    """Slide 2's chart without a band, from Table A.4's annual series (the 10-06 and 10-19 decks
    include it; band_charts draws the version with the band)."""
    S = mdt.ms_series(pd.read_csv(T2_CSV))
    years = S["years"]
    lines = _timedep_lines(S)
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    ax.axhline(0, color="#BFBFBA", lw=0.8, zorder=1)
    for k in ("selic", "rate", "spread"):
        y, color, ls, label, fmt = lines[k]
        ax.plot(years, y, color=color, label=label, **{**mdt.MS_LINE_KW, "linestyle": ls})
        mdt.ms_endlabel(ax, years, y, fmt.format(y[-1]), FS)
    mdt.ms_style(ax, "Selic and B time deposits", "% p.a. (spread: pp p.a.)", pix_label=True, fs=FS)
    mdt.ms_year_axis(ax, years, fs=FS)
    _legend_below(ax, 3)
    _save(fig, out, "fig_slide_selic_timedep_all", png)


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
    if SLIDE5 or REF:   # label column at the left, values on one line at the right, no footnote
        keep = [v for v, _ in AME_ROWS if v in cells and cells[v].get("estimate") is not None]
        vals = []
        for v in keep:
            c = cells[v]
            d = 2 if abs(c["estimate"]) >= 0.05 else 3
            vals.append(f"{_signed(c['estimate'], d)}{c.get('stars') or ''}"
                        f"{'†' if c.get('dagger') else ''} "
                        f"[{_plain(c['ci_lo'], d)}, {_plain(c['ci_hi'], d)}]")
        _forest(out, png, f"fig_slide_sleep_ame_E{E}", [AME_ROWS_SLIDE5[v] for v in keep],
                [cells[v]["estimate"] for v in keep], [cells[v]["ci_lo"] for v in keep],
                [cells[v]["ci_hi"] for v in keep], vals, "Effect on the sleepy share (pp)", "")
        return
    if SLIDE4:   # the deck copy: one-line labels above the bars, no footnote (the slide has it)
        vals, ints = [], []
        for var, c in [(v, cells[v]) for v, _ in AME_ROWS
                       if v in cells and cells[v].get("estimate") is not None]:
            d = 2 if abs(c["estimate"]) >= 0.05 else 3
            vals.append(f"{_signed(c['estimate'], d)}{c.get('stars') or ''}"
                        f"{'†' if c.get('dagger') else ''}")
            ints.append(f"[{_plain(c['ci_lo'], d)}, {_plain(c['ci_hi'], d)}]")
        keep = [v for v, _ in AME_ROWS if v in cells and cells[v].get("estimate") is not None]
        _forest_slide(out, png, f"fig_slide_sleep_ame_E{E}", [AME_ROWS_SLIDE[v] for v in keep],
                      [c["estimate"] for _, c in rows], [c["ci_lo"] for _, c in rows],
                      [c["ci_hi"] for _, c in rows], vals, ints,
                      "Effect on the sleepy share (pp)")
        return
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
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE if SLIDE5 or REF else (5.6, 3.1))
    fig.subplots_adjust(left=0.27, right=0.63, top=0.98, bottom=0.27)
    ax.axvline(0, color="#8A8A85", lw=0.9, zorder=1)
    if REF:   # the reference look: the estimate as a dot on a thin whisker, no bar
        ax.plot(est, ypos, "o", color=mdt.NATL_COLOR, markersize=4, zorder=4)
    else:
        ax.barh(ypos, est, height=0.42, color=mdt.NATL_COLOR, zorder=2)
    lw = 1.0 if REF else 2.0 if SLIDE5 else 1.1
    ax.errorbar(est, ypos, xerr=np.vstack([est - lo, hi - est]), fmt="none", ecolor="#111111",
                elinewidth=lw, capsize=3, capthick=lw, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8 * FS, linespacing=0.95)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    for y, t in zip(ypos, texts):
        ax.text(1.03 if SLIDE5 or REF else 1.04, y, t, transform=ax.get_yaxis_transform(),
                fontsize=SLIDE_MIN_PT if SLIDE5 else 8.5 * FS if REF else 8 * FS,
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
    if note:
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
        if blp["theta1_se"][i] is None:
            # A stage without joint SEs stores them as null (se_status says why); the chart's
            # intervals cannot be drawn from it.
            sys.exit(f"no joint standard error for {name} in routine E{E}, stage {mpn.BLP_STAGE} "
                     f"(se_status: {blp.get('se_status', 'not recorded')}): the coefficient chart needs it")
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
    # The deck copy's plot is narrow: more room left of the long bar for its value label, and
    # the axis label set from the right end so it stays inside the slot.
    ax.set_xlim(min(active, impact) * (1.75 if SLIDE4 else 1.25), 0.0)
    ax.set_xlabel(f"Own-price elasticity, routine {_routines.est_roman(E)}", fontsize=8.5 * FS,
                  color=mdt.AXIS_INK, loc="right" if SLIDE4 else "center")
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
% the spread on the average real, not Table 1's average bank; type 5 from __T5__) and of the
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
__NOTE__\end{tabular}
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
    pf = prepaid_facts()
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
        "prepaid": dict(label="Prepaid (5)", setter="Bank" + (r"$^{\dagger}$" if pf["fallback"] else ""),
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
           .replace("__Y1__", str(MAX_YEAR)).replace("__NQ__", str(len(by_q)))
           .replace("__T5__", pf["start"])
           .replace("__NOTE__", PREPAID_TYPES_NOTE if pf["fallback"] else ""))
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


PREPAID_TYPES_NOTE = (r"\multicolumn{5}{@{}l}{\scriptsize $^{\dagger}$No bank-level prepaid rate is "
                      r"observed: the CDI fallback, so the spread is Selic $-$ CDI.} \\" + "\n")


def prepaid_facts() -> dict:
    """Type-5 start quarter and whether every in-window type-5 rate is the common CDI fallback
    (utils/prepaid.py), from the panel. Read once per run."""
    if "prepaid" not in _MOTIV_CACHE:
        d = apply_window(pd.read_parquet(PANEL_PARQUET, columns=["year", "quarter", "dep_a5", "rate_a5"]),
                         label="slide prepaid", verbose=False)
        _MOTIV_CACHE["prepaid"] = dict(start=_prepaid.start_label(d), fallback=_prepaid.common_rate(d))
    return _MOTIV_CACHE["prepaid"]


def motiv_panel() -> pd.DataFrame:
    """The paper's sample (prepare_panel) with the columns the motivation charts read, and the
    row-level type-4 and type-5 spreads annualized as build_table2 does it. prepare_panel leaves
    no type-5 balance or spread before the first reported type-5 balance. Read once per run."""
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
    pooled, weighted by type-5 balance (Table A.4's national type-5 line); prepaid_B and
    prepaid_D, the same for one firm type."""
    if kind == "time_B":
        c = b_spread_cells(df)
        return pd.DataFrame({"year": c["year"], "quarter": c["quarter"], "congl": c["congl"],
                             "x": c["sp4"], "w": c["w"]})
    d = df if kind == "prepaid" else df[df["bank_type"] == kind[-1]]
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
    return dict(mean=mdt._weighted_mean(x, w), sd=mdt._weighted_std(x, w))


def _store(section: str, payload: dict):
    """Merge one chart's numbers into the motivation sidecar, keeping the other charts'."""
    if PAPER or SLIDE or REF:
        return
    data = json.loads(MOTIV_JSON.read_text(encoding="utf-8")) if MOTIV_JSON.is_file() else {}
    data.update(schema=1, generator="make_slide_figures.py (motivation charts)",
                window=[MIN_YEAR, MAX_YEAR])
    data[section] = payload
    _dump(MOTIV_JSON, json.dumps(data, indent=2, default=float) + "\n")


def band_charts(out: pathlib.Path, png: bool):
    """Slide 2's Selic chart with a band of +/-1 standard deviation across B banks around the B
    spread. The band and the line share Table A.4's weights, so the weighted mean of the band's
    distribution is the line itself (checked)."""
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
    lines = _timedep_lines(S)
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    ax.axhline(0, color="#BFBFBA", lw=0.8, zorder=1)
    for k in ("selic", "rate", "spread"):
        y, color, ls, label, fmt = lines[k]
        ax.plot(years, y, color=color, label=label, **{**mdt.MS_LINE_KW, "linestyle": ls})
        mdt.ms_endlabel(ax, years, y, fmt.format(y[-1]), FS)
    ax.fill_between(years, spread - sd, spread + sd, color=mdt.B_COLOR, alpha=0.13, linewidth=0,
                    zorder=2, label="±1 SD across B banks")
    mdt.ms_style(ax, "Selic and B time deposits", "% p.a. (spread: pp p.a.)",
                 pix_label=True, fs=FS)
    mdt.ms_year_axis(ax, years, fs=FS)
    _legend_below(ax, 2)
    _save(fig, out, "fig_slide_selic_timedep_all_sd", png)
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
    for name, kind in (("sp4_b", "time_B"), ("sp4_d", "time_D"), ("sp5", "prepaid"),
                       ("sp5_b", "prepaid_B"), ("sp5_d", "prepaid_D")):
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
    # Prepaid balances are reported from 2020Q2 (prepare_panel); before that the share is
    # undefined for the chart, not zero.
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
    years = np.arange(MIN_YEAR, MAX_YEAR + 1, dtype=float)
    mdt.ms_year_axis(ax, years, fs=FS)
    if SLIDE2 or SLIDE6:   # half a slide's width per panel: the window's ends and middle
        ticks = years[[0, len(years) // 2, -1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(y)) for y in ticks])
    elif PAPER or SLIDE or REF:   # nine year labels do not fit a ~3in panel: label every other year
        ax.set_xticks(years[::2])
        ax.set_xticklabels([str(int(y)) for y in years[::2]])


def _end_labels(ax, x_end, items, min_gap=0.07):
    """ms_endlabel for several lines at once: labels whose ends sit closer than `min_gap` (a
    fraction of the axis height) are pushed apart symmetrically, and a text that another line
    already prints (two lines ending at the same value) is printed once. `items`: (y_end, text)."""
    lo, hi = ax.get_ylim()
    h_pts = ax.figure.get_size_inches()[1] * 72 * ax.get_position().height
    if SLIDE or REF:   # labels on a short axis: keep a line height between them once _save trims
        min_gap = max(min_gap, 1.2 * 8.5 * FS / (0.65 * h_pts))
    # An item may carry its line's colour, (y_end, text, colour); the deck copies print each
    # value in it, so a value pushed away from its line still reads as that line's.
    items = [(it[0], it[1], it[2] if len(it) > 2 and (SLIDE or REF) else mdt.MUTED_INK)
             for it in items]
    pts = sorted({t: (float(y), t, c) for y, t, c in items if np.isfinite(y)}.values())
    fr = [(y - lo) / (hi - lo) for y, _, _ in pts]
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
    for (y, text, color), f0, f1 in zip(pts, fr, adj):
        ax.annotate(text, xy=(x_end, y), xytext=(5, (f1 - f0) * h_pts),
                    textcoords="offset points", fontsize=8.5 * FS, color=color,
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
    kw = dict(linewidth=1.2 if REF else 2, solid_capstyle="round", zorder=3)
    selic = q["selic_ann"].to_numpy(dtype=float)
    # key, colour, line style, label, spread column, spread has an interval, share column
    series = (
        ("demand", DEMAND_COLOR, "-", "Demand", "sp_demand", False, "sh_demand"),
        ("savings", SAVINGS_COLOR, "-", "Savings", "sp_savings", False, "sh_savings"),
        ("time_B", mdt.B_COLOR, "-", "Time, B firms", "sp4_b", True, "sh_time_B"),
        ("time_D", mdt.D_COLOR, (0, (5, 2)), "Time, D firms", "sp4_d", True, "sh_time_D"),
        ("prepaid", PREPAID_COLOR, PREPAID_LS, f"Prepaid (from {prepaid_facts()['start']})", "sp5", True,
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
         spread_ci, lambda v: _plain(v, 1), True, False, True),
        ("fig_slide_motiv_shares_by_type", "Share of balances by deposit type", "% of balances",
         lambda s: col(s[6]), None, lambda v: f"{v:.0f}%", False, False, True),
        ("fig_slide_motiv_rates_by_type", "Deposit rate by type", "% p.a.", rate, rate_ci, pct1,
         True, True, True),
        # Selic is drawn in this chart, so the cycle can be read without the shading.
        ("fig_slide_motiv_rates_by_type_noshade", "Deposit rate by type", "% p.a.", rate,
         rate_ci, pct1, True, True, False),
    )
    for stem, title, ylabel, value, ci, fmt, zero, with_selic, shade in charts:
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        if zero:
            ax.axhline(0, color="#BFBFBA", lw=0.8, zorder=1)
        if shade:
            _shade(ax, q, i0, i1)
        ends = []
        if with_selic:
            ax.plot(x, selic, color=mdt.NATL_COLOR, label="Selic", **kw)
            ends.append((selic[-1], fmt(selic[-1]), mdt.NATL_COLOR))
        for s in series:
            if SLIDE and s[0] == "prepaid":   # the deck copies show the four other types
                continue
            y = value(s)
            if ci is not None and s[5] and not SLIDE:
                lo, hi = ci(s)
                ax.fill_between(x, lo, hi, color=s[1], alpha=0.16, linewidth=0, zorder=2)
            ax.plot(x, y, color=s[1], label=s[3], **{**kw, "linestyle": s[2]})
            ends.append((y[-1], fmt(y[-1]), s[1]))
        if SLIDE6:   # half width: short titles
            title = {"fig_slide_motiv_spreads_by_type": "Spread by type",
                     "fig_slide_motiv_shares_by_type": "Share of balances"}.get(stem, title)
        _style_no_pix(ax, title, ylabel, "")
        if (SLIDE or REF) and shade:   # headroom so the 'Selic hikes' label clears the top line
            lo_, hi_ = ax.get_ylim()
            ax.set_ylim(lo_, hi_ + 0.15 * (hi_ - lo_))
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
            "spread_prepaid_B": [at("sp5_b", i0), at("sp5_b", i1)],
            "spread_prepaid_D": [at("sp5_d", i0), at("sp5_d", i1)],
            "ci": {k: {"start": ci_at(c, i0), "end": ci_at(c, i1)}
                   for k, c in (("spread_time_B", "sp4_b"), ("spread_time_D", "sp4_d"),
                                ("spread_prepaid", "sp5"), ("spread_prepaid_B", "sp5_b"),
                                ("spread_prepaid_D", "sp5_d"))},
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
    return bank_quarter("B")


def bank_quarter(kind: str) -> pd.DataFrame:
    """Conglomerates of one firm type (B or D) by quarter: type-4 balance summed over markets
    and the bank's national type-4 spread, flagged when the rate is the CDI fallback or the
    3 x Selic guard."""
    df = motiv_panel()
    b = df[df["bank_type"] == kind]
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
    """The repricing chart for B banks and, on the D firms that have an own rate, for D."""
    _reprice_one(out, png, "B")
    _reprice_one(out, png, "D")


def _reprice_one(out: pathlib.Path, png: bool, kind: str):
    """One firm type: the change in a bank's own type-4 spread from one quarter to the next against
    the growth of its time deposits over the following four quarters. The rate is interest
    expense over the lagged balance, so growth within the quarter of the change would move the
    measured rate itself; reading growth from that quarter onward avoids it."""
    sfx, color = ("", mdt.B_COLOR) if kind == "B" else ("_d", mdt.D_COLOR)
    unit = "bank" if kind == "B" else "firm"
    bq = bank_quarter(kind).sort_values(["CodConglomeradoPrudencial", "t"]).reset_index(drop=True)
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
            (f"{kind} conglomerate-quarters with time deposits", win[win["dep_a4"] > 0]),
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

    for stem, points in ((f"fig_slide_motiv_reprice_binscatter{sfx}", False),
                         (f"fig_slide_motiv_reprice_binscatter_points{sfx}", True)):
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        ax.axvline(0, color="#BFBFBA", lw=0.8, zorder=1)
        if points:
            ax.scatter(xr, yr, s=6, color=color, alpha=0.18, linewidth=0, zorder=2,
                       label=f"Each {unit}-quarter")
        xx = np.array([bx.min(), bx.max()])
        # The statistics ride in the legend under the plot: any spot inside it can hold a bin.
        ax.scatter(bx, by, s=34, color=color, edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"{r['n']:,} {unit}-quarters of {r['banks']} {unit}s, in {len(bx)} "
                         f"bins of about {int(round(per_bin.mean()))}")
        ax.plot(xx, yr.mean() + r["slope"] * (xx - xr.mean()), color=mdt.NATL_COLOR, lw=1.6,
                zorder=4, label=f"Slope {_plain(r['slope'], 3)}{r['stars']} % per pp, 95% "
                                f"interval [{_plain(r['lo_bc'], 3)}, {_plain(r['hi_bc'], 3)}]")
        _style_no_pix(ax, f"Repricing and deposit growth, {kind} {unit}s",
                      f"Time-deposit growth, next {GROWTH_H} quarters (%)",
                      "Change in own time-deposit spread (pp p.a.)")
        ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1,
                  frameon=False, handlelength=1.8)
        _save(fig, out, stem, png)

    def clean(d):
        return {k: v for k, v in d.items() if k not in ("xr", "yr")}
    _store("reprice" + sfx, {
        "sample": f"{kind} conglomerate-quarters with an own type-4 rate in both quarters (not the CDI "
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
    print(f"  repricing ({kind}) slope {r['slope']:.4f} [{r['lo_bc']:.4f}, {r['hi_bc']:.4f}] "
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
    kw = dict(linewidth=1.2 if REF else 2, solid_capstyle="round", zorder=3, color=mdt.D_COLOR)
    if SLIDE2:   # a full-width, short slot: the two panels side by side
        fig, (a1, a2) = plt.subplots(1, 2, figsize=SLIDE_FIGSIZE)
    else:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(SLIDE_FIGSIZE[0], SLIDE_FIGSIZE[1] * 1.25),
                                     sharex=True)
    a1.axhline(0, color="#BFBFBA", lw=0.8, zorder=1)
    a1.fill_between(x, ci["lo"], ci["hi"], color=mdt.D_COLOR, alpha=0.16, linewidth=0, zorder=2,
                    label="95% interval")
    a1.plot(x, prem, **kw)
    if not PAPER and not SLIDE and not REF:   # the caption or the copy's title states the band
        a1.legend(fontsize=8 * FS, loc="upper right", frameon=False)
    mdt.ms_endlabel(a1, x, prem, f"{_plain(prem[-1], 1)} pp", FS)
    # The deck copy keeps each title on one line, so the two short panels keep their height.
    mdt.ms_style(a1, ("D premium over B (95% band)" if SLIDE5 or REF else
                      "D premium over B, with 95% band" if SLIDE
                      else "D firms' time-deposit premium over B firms"), "pp p.a.",
                 pix_label=True, fs=FS, pix_x=PIX_X_QUARTERLY)
    a2.plot(x, share, **kw)
    mdt.ms_endlabel(a2, x, share, f"{share[-1]:.1f}%", FS)
    mdt.ms_style(a2, ("D share of time deposits" if SLIDE5 or REF else
                      "D firms' share of time deposits" if SLIDE
                      else "D firms' share of time-deposit balances"), "%", fs=FS,
                 pix_x=PIX_X_QUARTERLY)
    if SLIDE5 or REF:   # one-line titles wider than the plots: each starts over its y label
        for a in (a1, a2):
            a._left_title.set_x(-0.2)
    a2.set_ylim(bottom=0)
    _quarter_axis(a2)
    if SLIDE2:   # side by side, each panel carries its own year axis
        _quarter_axis(a1)
        a1.set_xlabel("")
        a2.set_xlabel("")
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


def _cluster_median_ci(M: np.ndarray, clusters, boot: int, seed: int):
    """Median of each column of M over its finite rows, with a 95% percentile interval from a
    bootstrap that resamples clusters (banks) with replacement and keeps all their rows."""
    cl = np.asarray(clusters)
    ids = np.unique(cl)
    rows_of = {c: np.flatnonzero(cl == c) for c in ids}
    rng = np.random.default_rng(seed)
    draws = np.full((boot, M.shape[1]), np.nan)
    for b in range(boot):
        pick = rng.choice(ids, len(ids), replace=True)
        draws[b] = np.nanmedian(M[np.concatenate([rows_of[c] for c in pick])], axis=0)
    return (np.nanmedian(M, axis=0), np.nanpercentile(draws, 2.5, axis=0),
            np.nanpercentile(draws, 97.5, axis=0))


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

    # The same medians with the bootstrap resampling banks instead of events: one bank entering
    # several markets contributes correlated paths, which the event bootstrap treats as
    # independent. The medians are the event charts' own (checked); only the bands change.
    bankboot = {}
    for norm_key, stem, ylabel, title, scale in (
            ("main", "fig_slide_motiv_entry_by_gap_bankboot", "Normalized market share",
             "B entrants' market share, by price at entry", 1.0),
            ("level", "fig_slide_motiv_entry_by_gap_level_bankboot",
             "Share of B deposits in the market (%)",
             "B entrants' share of B deposits, by price at entry", 100.0)):
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        for k, (color, ls, label) in style.items():
            reg = groups[k].assign(keep=True)
            Mn, rows = sed._path_matrix(reg, paths, H, "main", plateau_w, want_rows=True)
            ev_rows = reg.iloc[rows]
            if norm_key == "main":
                M = Mn
            else:
                M = np.vstack([np.pad(paths[(e["congl"], e["mkt"])].astype(float).to_numpy()[:H + 1],
                                      (0, max(0, H + 1 - len(paths[(e["congl"], e["mkt"])]))),
                                      constant_values=np.nan) for _, e in ev_rows.iterrows()]) * scale
            med, lo, hi = _cluster_median_ci(M, ev_rows["congl"].to_numpy(), boot, seed)
            ref = (res[k] if norm_key == "main" else level[k])["median"].to_numpy()
            gap = float(np.nanmax(np.abs(med - ref)))
            if not gap < 1e-12:
                sys.exit(f"entrygap: bank-bootstrap medians differ from the event medians by {gap:.3e}")
            h = np.arange(H + 1)
            ax.fill_between(h, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=1)
            # The deck copy's legend: two short lines, same groups and counts.
            short = {"more": f"Pays ≥ {ENTRY_GAP_PP:.0f} pp more",
                     "rest": f"Pays < {ENTRY_GAP_PP:.0f} pp more"}[k]
            ax.plot(h, med, color=color, linestyle=ls, linewidth=1.2 if REF else 2, marker="" if REF else "o", markersize=4,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                    label=(f"{short}:\n{len(ev_rows)} entries, {ev_rows['congl'].nunique()} banks"
                           if SLIDE2 or REF else
                           f"{short}: {len(ev_rows)} entries, {ev_rows['congl'].nunique()} banks"
                           if SLIDE else
                           f"{label} (n = {len(ev_rows)}, {ev_rows['congl'].nunique()} banks)"))
            bankboot.setdefault(norm_key, {})[k] = dict(
                n=int(len(ev_rows)), banks=int(ev_rows["congl"].nunique()),
                ci_at_h={str(x): [float(lo[x]), float(hi[x])] for x in (4, 8, 12)})
        _style_no_pix(ax, title, ylabel, "Quarters since entry")
        ax.set_xticks(range(0, H + 1, 2))
        if norm_key == "level":
            ax.set_ylim(bottom=0)
        if SLIDE2 or REF:   # a full-width slide: the legend beside the plot, entries stacked
            ax.legend(fontsize=8.5 * FS, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=False, handlelength=1.4, handletextpad=0.4, labelspacing=0.8,
                      borderaxespad=0.0)
        else:
            _legend_under(ax)
        _save(fig, out, stem, png)

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
        "bankboot": bankboot,
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


# --------------------------------------------------------------------------------------------
# Relative prices across B banks
# --------------------------------------------------------------------------------------------
# The five largest B banks are drawn by size rank, darkest = largest: size is ordered, so one
# hue in steps of lightness, with an end label on every line.
RANK_COLORS = ("#2a78d6", "#e34948", "#008300", "#eda100", "#1baf7a")
# Second encoding for the five lines: aqua and yellow are below 3:1 against the surface, and the
# closest pair under deuteranopia (red and aqua) is 6.9 apart, so every rank also has its own dash.
RANK_STYLES = ("-", (0, (6, 2)), (0, (2, 1.5)), (0, (6, 2, 1.5, 2)), (0, (1, 1.2)))
# Periods of the decomposition over time: easing (2016-2018), low rates and the turn (2019-2021),
# high rates (2022-2024).
DECOMP_PERIODS = ((2016, 2018), (2019, 2021), (2022, 2024))
N_TOP = 5
RELPRICE_NOTE = ("The rate is interest expense over each bank's stock of time deposits, so part "
                 "of any gap between banks is book composition and accounting noise.")


def relprice_frames():
    """B bank-quarters with an own type-4 rate (no CDI fallback, no 3 x Selic guard), each with
    three prices: the spread (pp p.a.), the rate as % of CDI, and their gaps to the balance-
    weighted B mean of the quarter; the bank's share of all B time deposits. And B bank x market
    x quarter cells with the local gap: own spread minus the time-deposit-weighted spread of the
    other own-rate B banks in the market, and the bank's share of the market's B time deposits.
    Weights are type-4 balances throughout. Memoised."""
    if "rel" in _MOTIV_CACHE:
        return _MOTIV_CACHE["rel"]
    allb = bank_quarter_b()
    tot = allb.groupby("t")["dep_a4"].sum()
    bq = allb[allb["own_rate"] & (allb["dep_a4"] > 0)].copy()
    bq["pcdi"] = 100.0 * bq["r4"] / (bq["cdi_qoq"] / 100.0)
    for c in ("sp_pp", "pcdi"):
        sw = (bq[c] * bq["dep_a4"]).groupby(bq["t"]).transform("sum")
        bq[f"{c}_gap"] = bq[c] - sw / bq["dep_a4"].groupby(bq["t"]).transform("sum")
    bq["share"] = bq["dep_a4"] / bq["t"].map(tot)
    bq["bank"] = bq["CodConglomeradoPrudencial"]

    df = motiv_panel()
    b = df[df["bank_type"] == "B"]
    b = pd.DataFrame({"bank": b["CodConglomeradoPrudencial"], "mkt": b["mca_code"].astype(str),
                      "t": b["year"] * 4 + b["quarter"] - 1, "dep": b["dep_a4"], "sp": b["sp4"]})
    b = b[b["dep"] > 0]
    mkt_tot = b.groupby(["mkt", "t"])["dep"].transform("sum")
    b["lshare"] = b["dep"] / mkt_tot
    own = pd.MultiIndex.from_frame(bq[["bank", "t"]])
    v = b[pd.MultiIndex.from_frame(b[["bank", "t"]]).isin(own) & np.isfinite(b["sp"])].copy()
    keys = [v["mkt"], v["t"]]
    W = v["dep"].groupby(keys).transform("sum")
    SW = (v["sp"] * v["dep"]).groupby(keys).transform("sum")
    riv_w = W - v["dep"]
    v["lgap"] = np.where(riv_w > 0, v["sp"] - (SW - v["sp"] * v["dep"]) / riv_w.where(riv_w > 0),
                         np.nan)
    cells = v[np.isfinite(v["lgap"])].copy()
    _MOTIV_CACHE["rel"] = (bq, cells)
    return bq, cells


def _wdemean(cols: pd.DataFrame, groups, w, iters=200, tol=1e-11) -> pd.DataFrame:
    """Weighted within transformation over one or more sets of fixed effects (alternating
    projections); with no groups, the weighted mean is removed."""
    out = cols.astype(float).copy()
    w = np.asarray(w, dtype=float)
    if not groups:
        return out - (out.mul(w, axis=0).sum() / w.sum())
    for _ in range(iters):
        before = out.to_numpy().copy()
        for g in groups:
            m = out.mul(w, axis=0).groupby(g).transform("sum").div(
                pd.Series(w, index=out.index).groupby(g).transform("sum"), axis=0)
            out = out - m
        if np.max(np.abs(out.to_numpy() - before)) < tol:
            break
    return out


def _wls_slope(d: pd.DataFrame, x: str, y: str, w: str, cl: str, fe=()) -> dict:
    """Slope of y on x weighted by w, net of the fixed effects in `fe` (column names; tuples are
    interacted), with the conglomerate wild cluster bootstrap (Webb, BOOT_B draws) and its
    bias-corrected 95% interval. Returns the residualised x and y for a binned plot."""
    import statsmodels.api as sm
    from utils.sleep_links import linear_wild_cluster_bootstrap

    d = d.reset_index(drop=True)
    groups = [d[f[0]].astype(str) + "|" + d[f[1]].astype(str) if isinstance(f, tuple) else d[f]
              for f in fe]
    r = _wdemean(d[[x, y]], groups, d[w])
    sw = np.sqrt(d[w].to_numpy(dtype=float))
    X = pd.DataFrame({x: r[x].to_numpy() * sw})
    res = sm.OLS(r[y].to_numpy() * sw, X).fit(cov_type="cluster",
                                             cov_kwds={"groups": d[cl].to_numpy()})
    band = {}
    linear_wild_cluster_bootstrap(res, B=BOOT_B, scheme="webb", seed=0, mode="normal",
                                  band_out=band)
    row = band["band"].set_index("name").loc[x]
    return dict(slope=float(res.params[x]), lo=float(row["lo_bc"]), hi=float(row["hi_bc"]),
                p=float(row["p_bc"]), stars=str(row["stars"]), n=int(len(d)),
                banks=int(d[cl].nunique()), xr=r[x].to_numpy(), yr=r[y].to_numpy(),
                w=d[w].to_numpy(dtype=float))


def _trim(d: pd.DataFrame, cols) -> pd.DataFrame:
    d = d[np.isfinite(d[list(cols)]).all(axis=1)]
    keep = np.ones(len(d), dtype=bool)
    for c in cols:
        lo, hi = np.percentile(d[c], TRIM_PCT)
        keep &= d[c].between(lo, hi).to_numpy()
    return d[keep]


def _binned(ax, r: dict, xmean: float, ymean: float, color=mdt.B_COLOR, points=False,
            label_n="observations"):
    """Bins of equal count along the residualised x (weighted means within a bin), the fitted
    line, and the slope in the legend."""
    xr, yr, w = r["xr"] + xmean, r["yr"] + ymean, r["w"]
    if points:
        ax.scatter(xr, yr, s=6, color=color, alpha=0.15, linewidth=0, zorder=2)
    bins = pd.qcut(pd.Series(xr).rank(method="first"), N_BINS, labels=False)
    wx = pd.Series(xr * w).groupby(bins).sum() / pd.Series(w).groupby(bins).sum()
    wy = pd.Series(yr * w).groupby(bins).sum() / pd.Series(w).groupby(bins).sum()
    ax.scatter(wx, wy, s=34, color=color, edgecolor="white", linewidth=0.8, zorder=3,
               label=f"{r['n']:,} {label_n} of {r['banks']} banks, {N_BINS} bins")
    xx = np.array([wx.min(), wx.max()])
    ax.plot(xx, ymean + r["slope"] * (xx - xmean), color=mdt.NATL_COLOR, lw=1.6, zorder=4,
            label=f"Slope {_plain(r['slope'], 3)}{r['stars']}, 95% interval "
                  f"[{_plain(r['lo'], 3)}, {_plain(r['hi'], 3)}]")


def _legend_under(ax):
    if REF:   # a wide, short slot: the entries in three columns
        ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
                  frameon=False, columnspacing=0.8, handlelength=1.2, handletextpad=0.4,
                  labelspacing=0.25, borderaxespad=0.0, borderpad=0.0)
        return
    if SLIDE:   # deck copies: short swatches, and the scatter's biggest dot scaled down
        ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
                  frameon=False, handlelength=1.0, handletextpad=0.4, markerscale=0.5,
                  labelspacing=0.3, borderaxespad=0.0, borderpad=0.0)
        return
    ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1,
              frameon=False, handlelength=1.8)


def _wr2(y, w, groups) -> float:
    """Weighted R2 of y on the fixed effects in `groups`."""
    d = pd.DataFrame({"y": y})
    r = _wdemean(d, groups, w)["y"].to_numpy()
    w = np.asarray(w, dtype=float)
    yc = np.asarray(y, dtype=float) - np.average(y, weights=w)
    return float(1.0 - np.sum(w * r ** 2) / np.sum(w * yc ** 2))


def _decomp_bars(out, png, stem, title, rows, tint):
    """Stacked bars of a variance decomposition: common to all banks (grey), the bank's
    persistent position and the rest, in the firm type's colour and its light tint. rows:
    (label, {common, bank, rest}, firm type)."""
    fig, ax = plt.subplots(figsize=(SLIDE_FIGSIZE[0], 0.9 + 0.42 * len(rows)))
    seen = set()
    for i, (lab, vals, kind) in enumerate(rows):
        left = 0.0
        dark, light = tint[kind]
        # The reference copy's short names keep the legend to two rows in a 4.6 cm slot.
        parts = (("common", "Common to all banks", mdt.NATL_COLOR),
                 ("bank", f"Persistent, {kind}" if REF else
                  f"Bank's persistent position, {kind} firms", dark),
                 ("rest", f"Rest, {kind}" if REF else f"Rest, {kind} firms", light))
        for k, name, color in parts:
            v = 100.0 * vals[k]
            ax.barh(i, v, left=left, color=color, edgecolor="white", height=0.6,
                    label=None if name in seen else name)
            seen.add(name)
            if v >= 8:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8.5 * FS, color="white" if color != light else "#111111")
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    _style_no_pix(ax, title, "", "Share of balance-weighted variance (%)")
    ax.grid(False)
    _legend_under(ax)
    _save(fig, out, stem, png)


def relprice_charts(out: pathlib.Path, png: bool):
    """Relative prices across B banks: A, how much of the spread is common to all banks; B, the
    five largest banks' relative positions; C, whether a relative position persists a year
    later; D, whether a bank's deposit share follows its relative price, nationally and across
    the markets one bank serves. Three prices: the gap in pp, the rate as % of CDI, the local
    gap. RELPRICE_NOTE states the measurement caveat that goes with every chart."""
    bq, cells = relprice_frames()
    q = macro_quarterly()
    xq = dict(zip(q["year"] * 4 + q["quarter"] - 1, q["x"]))
    payload = {"note": RELPRICE_NOTE}

    # A. Variance decomposition -------------------------------------------------------------
    dec = {}
    for key, col in (("pp", "sp_pp"), ("pcdi", "pcdi")):
        r2q = _wr2(bq[col], bq["dep_a4"], [bq["t"]])
        r2qb = _wr2(bq[col], bq["dep_a4"], [bq["t"], bq["bank"]])
        dec[key] = dict(common=r2q, bank=r2qb - r2q, rest=1.0 - r2qb)
    r2 = _wr2(cells["lgap"], cells["dep"], [cells["bank"].astype(str) + "|" + cells["t"].astype(str)])
    dec["local"] = dict(national=r2, rival_mix=1.0 - r2)
    payload["decomposition"] = dec
    payload["sample"] = dict(bank_quarters=int(len(bq)), banks=int(bq["bank"].nunique()),
                             cells=int(len(cells)), cell_banks=int(cells["bank"].nunique()),
                             markets=int(cells["mkt"].nunique()))
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    # One legend for the three bars: blue is the bank's own position (persistent across quarters
    # for the spread; its national position for the local gap), light blue the remainder (for
    # the local gap, which rivals the bank meets in each market).
    common = ("common", "Common to all banks", mdt.NATL_COLOR)
    bank = "Bank's own position (local gap: national)"
    rest = "Rest (local gap: rival mix across markets)"
    rows = (("Spread, pp", (common, ("bank", bank, mdt.B_COLOR), ("rest", rest, "#BBDEFB")),
             dec["pp"]),
            ("% of CDI", (common, ("bank", bank, mdt.B_COLOR), ("rest", rest, "#BBDEFB")),
             dec["pcdi"]),
            ("Local gap, pp", (("national", bank, mdt.B_COLOR), ("rival_mix", rest, "#BBDEFB")),
             dec["local"]))
    seen = set()
    for i, (lab, parts, vals) in enumerate(rows):
        left = 0.0
        for k, name, color in parts:
            v = 100.0 * vals[k]
            ax.barh(i, v, left=left, color=color, edgecolor="white", height=0.6,
                    label=None if name in seen else name)
            seen.add(name)
            if v >= 8:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5 * FS,
                        color="white" if color != "#BBDEFB" else "#111111")
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    _style_no_pix(ax, "Where spread differences come from, B banks", "",
                  "Share of balance-weighted variance (%)")
    ax.grid(False)
    _legend_under(ax)
    _save(fig, out, "fig_slide_relprice_decomp", png)

    # A'. The same decomposition for D firms, and for B and D by period ------------------------
    def own_rate_frame(kind):
        f = bank_quarter(kind)
        f = f[f["own_rate"] & (f["dep_a4"] > 0)].copy()
        f["pcdi"] = 100.0 * f["r4"] / (f["cdi_qoq"] / 100.0)
        f["bank"] = f["CodConglomeradoPrudencial"]
        return f

    def decomp(f):
        res = {}
        for key, col in (("pp", "sp_pp"), ("pcdi", "pcdi")):
            r2q = _wr2(f[col], f["dep_a4"], [f["t"]])
            r2qb = _wr2(f[col], f["dep_a4"], [f["t"], f["bank"]])
            res[key] = dict(common=r2q, bank=r2qb - r2q, rest=1.0 - r2qb)
        res["n"] = int(len(f))
        res["banks"] = int(f["bank"].nunique())
        return res

    frames = {"B": own_rate_frame("B"), "D": own_rate_frame("D")}
    chk = decomp(frames["B"])["pp"]["common"]
    if not abs(chk - dec["pp"]["common"]) < 1e-12:
        sys.exit(f"relprice: B decomposition rebuilt differs ({chk} vs {dec['pp']['common']})")
    dec_d = decomp(frames["D"])
    dec_t = {k: {f"{a}-{b}": decomp(f[f["year"].between(a, b)]) for a, b in DECOMP_PERIODS}
             for k, f in frames.items()}
    payload["decomposition_d"] = dec_d
    payload["decomposition_by_period"] = dec_t
    tint = {"B": (mdt.B_COLOR, "#BBDEFB"), "D": (mdt.D_COLOR, "#FFCCBC")}
    _decomp_bars(out, png, "fig_slide_relprice_decomp_d",
                 "Where spread differences come from, D firms",
                 [("Spread, pp", dec_d["pp"], "D"), ("% of CDI", dec_d["pcdi"], "D")], tint)
    _decomp_bars(out, png, "fig_slide_relprice_decomp_time",
                 "Spread differences by period, B and D firms",
                 [(f"{k}, {p.replace('-', '–')}", dec_t[k][p]["pp"], k)
                  for k in ("B", "D") for p in dec_t[k]], tint)

    # B. Paths of the five largest B banks ---------------------------------------------------
    size = bank_quarter_b().groupby("CodConglomeradoPrudencial")["dep_a4"].sum().sort_values(
        ascending=False)
    top = list(size.index[:N_TOP])
    payload["top5_share_of_B_time_deposits"] = float(size.iloc[:N_TOP].sum() / size.sum())
    lcell = cells.copy()
    lcell["lw"] = lcell["lgap"] * lcell["dep"]
    lbank = (lcell.groupby(["bank", "t"])["lw"].sum() / lcell.groupby(["bank", "t"])["dep"].sum()
             ).rename("lgap_bank").reset_index()
    paths = bq.merge(lbank, on=["bank", "t"], how="left")
    bmean_pcdi = ((bq["pcdi"] * bq["dep_a4"]).groupby(bq["t"]).sum()
                  / bq["dep_a4"].groupby(bq["t"]).sum())
    tt = np.array(sorted(xq))
    for key, col, title, ylabel in (
            ("pp", "sp_pp_gap", "Spread gap to the B average", "pp p.a."),
            ("pcdi", "pcdi", "Rate as % of CDI", "% of CDI"),
            ("local", "lgap_bank", "Gap to local rivals, averaged over markets", "pp p.a.")):
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        ax.axhline(100 if key == "pcdi" else 0, color="#BFBFBA", lw=0.8, zorder=1)
        if key == "pcdi":
            ax.plot([xq[t] for t in bmean_pcdi.index], bmean_pcdi.to_numpy(),
                    color=mdt.NATL_COLOR, lw=1.4, ls=(0, (4, 2)), zorder=2, label="B average")
        ends = []
        for rank, (bank, color, ls) in enumerate(zip(top, RANK_COLORS, RANK_STYLES), start=1):
            p = paths[paths["bank"] == bank].set_index("t")[col].reindex(tt)
            ax.plot([xq[t] for t in tt], p.to_numpy(), color=color, lw=1.8, ls=ls, zorder=3,
                    label=f"{_ordinal(rank)} largest")
            last = p.dropna()
            if len(last):
                ends.append((float(last.iloc[-1]), _ordinal(rank)))
        _style_no_pix(ax, title, ylabel, "")
        _end_labels(ax, xq[tt[-1]], ends)
        _quarter_axis(ax)
        _legend_below(ax, 3)
        _save(fig, out, f"fig_slide_relprice_paths_{key}", png)

    # C. Persistence, t against t + 4 --------------------------------------------------------
    pers = {}
    nxt = bq[["bank", "t", "sp_pp_gap", "pcdi_gap"]].copy()
    nxt["t"] -= GROWTH_H
    pq = bq.merge(nxt, on=["bank", "t"], suffixes=("", "_4"))
    lnxt = cells[["bank", "mkt", "t", "lgap"]].copy()
    lnxt["t"] -= GROWTH_H
    pl = cells.merge(lnxt, on=["bank", "mkt", "t"], suffixes=("", "_4"))
    for key, d, x, y, w, unit, lab in (
            ("pp", pq, "sp_pp_gap", "sp_pp_gap_4", "dep_a4", "pp", "bank-quarters"),
            ("pcdi", pq, "pcdi_gap", "pcdi_gap_4", "dep_a4", "CDI points", "bank-quarters"),
            ("local", pl, "lgap", "lgap_4", "dep", "pp", "bank-market-quarters")):
        d = _trim(d, [x, y])
        r = _wls_slope(d, x, y, w, "bank")
        xm, ym = np.average(d[x], weights=d[w]), np.average(d[y], weights=d[w])
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        _binned(ax, r, xm, ym, points=(key != "local"), label_n=lab)
        lim = np.array(ax.get_xlim())
        ax.plot(lim, lim, color="#BFBFBA", lw=0.9, ls=(0, (4, 2)), zorder=1,
                label="45-degree line: gaps never close")
        _style_no_pix(ax, f"Does a bank's relative price persist? ({unit})",
                      f"Gap {GROWTH_H} quarters later", "Gap this quarter")
        _legend_under(ax)
        _save(fig, out, f"fig_slide_relprice_persist_{key}", png)
        pers[key] = {k: v for k, v in r.items() if k not in ("xr", "yr", "w")}
    payload["persistence"] = pers

    # D. Price against share change -----------------------------------------------------------
    shr = {}
    # Cross-section: one point per bank, average gap against the yearly log change in share.
    xs = []
    for bank, g in bq.sort_values("t").groupby("bank"):
        if g["t"].iloc[-1] - g["t"].iloc[0] < 8:
            continue
        yrs = (g["t"].iloc[-1] - g["t"].iloc[0]) / 4.0
        xs.append(dict(bank=bank, pp=float(g["sp_pp_gap"].mean()), pcdi=float(g["pcdi_gap"].mean()),
                       dshare=100.0 * np.log(g["share"].iloc[-1] / g["share"].iloc[0]) / yrs,
                       size=float(g["dep_a4"].mean())))
    xs = pd.DataFrame(xs)
    for key, unit in (("pp", "pp"), ("pcdi", "CDI points")):
        d = _trim(xs, [key, "dshare"])
        r = _wls_slope(d, key, "dshare", "size", "bank")
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        ax.axvline(0, color="#BFBFBA", lw=0.8, zorder=1)
        ax.axhline(0, color="#BFBFBA", lw=0.8, zorder=1)
        # The deck copy breaks both entries by hand to fit half the slide's width.
        ax.scatter(d[key], d["dshare"], s=MARK * (8 + 300 * d["size"] / d["size"].max()),
                   color=mdt.B_COLOR, alpha=0.55, edgecolor="white", linewidth=0.6, zorder=3,
                   label=("_nolegend_" if SLIDE2 else
                          f"{r['n']} B banks,\ndot size = time deposits" if SLIDE else
                          f"{r['n']} B banks, dot size = time deposits"))
        xm, ym = np.average(d[key], weights=d["size"]), np.average(d["dshare"], weights=d["size"])
        xx = np.array([d[key].min(), d[key].max()])
        ax.plot(xx, ym + r["slope"] * (xx - xm), color=mdt.NATL_COLOR, lw=1.6, zorder=4,
                label=(f"Slope {_plain(r['slope'], 2)},\n95% CI "
                       f"[{_plain(r['lo'], 2)}, {_plain(r['hi'], 2)}]" if SLIDE2 else
                       f"Weighted slope {_plain(r['slope'], 3)}{r['stars']},\n95% CI "
                       f"[{_plain(r['lo'], 3)}, {_plain(r['hi'], 3)}]" if SLIDE else
                       f"Weighted slope {_plain(r['slope'], 3)}{r['stars']}, 95% interval "
                       f"[{_plain(r['lo'], 3)}, {_plain(r['hi'], 3)}]"))
        # The gap is the bank's spread minus the B average, so a positive gap means the bank pays
        # less; the deck copy says so on the axis (same values).
        _style_no_pix(ax, ("Price gap and share growth" if SLIDE
                           else "Relative price and share growth, by bank"),
                      "Share growth (% a year)",
                      f"Pays less than the B average ({unit})" if SLIDE and key == "pp"
                      else f"Average gap to the B average ({unit})")
        if SLIDE2:
            # Headroom at the top, so the legend sits above every bank's dot.
            lo_, hi_ = ax.get_ylim()
            ax.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))
            # 9 pt (the deck copies' floor) and a short swatch keep the entry inside the plot.
            ax.legend(fontsize=SLIDE_MIN_PT, loc="upper right", frameon=False,
                      handlelength=0.8, handletextpad=0.3, borderaxespad=0.2)
            # One line of title is wider than the plot: it starts left of the axes, over the
            # y-axis label.
            ax._left_title.set_x(-0.32)
        else:
            _legend_under(ax)
        _save(fig, out, f"fig_slide_relprice_share_xs_{key}", png)
        shr[f"xs_{key}"] = {k: v for k, v in r.items() if k not in ("xr", "yr", "w")}
    # Panel: gap at t against the log change in share from t to t + 4, bank and quarter effects.
    nx = bq[["bank", "t", "share"]].copy()
    nx["t"] -= GROWTH_H
    pn = bq.merge(nx, on=["bank", "t"], suffixes=("", "_4"))
    pn["dshare"] = 100.0 * np.log(pn["share_4"] / pn["share"])
    for key, col, unit in (("pp", "sp_pp_gap", "pp"), ("pcdi", "pcdi_gap", "CDI points")):
        d = _trim(pn, [col, "dshare"])
        r = _wls_slope(d, col, "dshare", "dep_a4", "bank", fe=("bank", "t"))
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        _binned(ax, r, np.average(d[col], weights=d["dep_a4"]),
                np.average(d["dshare"], weights=d["dep_a4"]), label_n="bank-quarters")
        _style_no_pix(ax, "Relative price and share growth, bank-quarters",
                      f"Change in share, next {GROWTH_H} quarters (%)",
                      f"Gap to B average ({unit}), net of fixed effects")
        _legend_under(ax)
        _save(fig, out, f"fig_slide_relprice_share_panel_{key}", png)
        shr[f"panel_{key}"] = {k: v for k, v in r.items() if k not in ("xr", "yr", "w")}
    # Local: gap to local rivals at t against the log change in the local share.
    ln = cells[["bank", "mkt", "t", "lshare"]].copy()
    ln["t"] -= GROWTH_H
    pc = cells.merge(ln, on=["bank", "mkt", "t"], suffixes=("", "_4"))
    pc["dshare"] = 100.0 * np.log(pc["lshare_4"] / pc["lshare"])
    d = _trim(pc, ["lgap", "dshare"])
    for key, fe, xlab in (("local_raw", (), "Gap to local rivals (pp)"),
                          ("local_fe", (("bank", "t"), "mkt"),
                           "Gap to local rivals (pp), net of fixed effects")):
        r = _wls_slope(d, "lgap", "dshare", "dep", "bank", fe=fe)
        fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
        _binned(ax, r, np.average(d["lgap"], weights=d["dep"]),
                np.average(d["dshare"], weights=d["dep"]), label_n="bank-market-quarters")
        _style_no_pix(ax, "Local relative price and local share growth"
                      + (", within bank" if key == "local_fe" else ""),
                      f"Change in local share, next {GROWTH_H} quarters (%)", xlab)
        _legend_under(ax)
        _save(fig, out, f"fig_slide_relprice_share_{key}", png)
        shr[key] = {k: v for k, v in r.items() if k not in ("xr", "yr", "w")}
    payload["share"] = shr
    _store("relprice", payload)
    print("  decomposition (%): " + "; ".join(
        f"{k} " + ", ".join(f"{n} {100 * v:.1f}" for n, v in d_.items()) for k, d_ in dec.items()))
    print(f"  top {N_TOP} B banks hold {100 * payload['top5_share_of_B_time_deposits']:.1f}% of "
          "B time deposits")
    for k, r in {**{f"persist_{k}": v for k, v in pers.items()}, **shr}.items():
        print(f"  {k}: slope {r['slope']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] N={r['n']} "
              f"banks={r['banks']}")


def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


# --------------------------------------------------------------------------------------------
# Variance and co-movement tables for relative prices, shares and volumes
# --------------------------------------------------------------------------------------------
ANOVA_MD = mdt.OUTPUT_DIR / "slide_anova_tables.md"
COMOVE_H = (1, 4, 8, 12)


def _shapley2(y, w, ga, gb) -> dict:
    """Two-factor split of the weighted variance of y: factor a (e.g. quarter), factor b (e.g.
    bank), each with the order-free Shapley share, the sequential share with a first, and the
    rest."""
    ra, rb = _wr2(y, w, [ga]), _wr2(y, w, [gb])
    rab = _wr2(y, w, [ga, gb])
    return dict(a=0.5 * (ra + rab - rb), b=0.5 * (rb + rab - ra), rest=1.0 - rab,
                a_first=ra, b_after_a=rab - ra)


def _bank_panel(kind: str) -> pd.DataFrame:
    """Conglomerate-quarters of one firm type with time deposits: prices where the bank has an own
    rate (gap to the firm type's balance-weighted mean, pp and points of CDI), log balance, log
    share of the firm type's time deposits, and quarterly growth."""
    f = bank_quarter(kind)
    f = f[f["dep_a4"] > 0].copy()
    f["bank"] = f["CodConglomeradoPrudencial"]
    f["share"] = f["dep_a4"] / f.groupby("t")["dep_a4"].transform("sum")
    f["log_dep"] = np.log(f["dep_a4"])
    f["log_share"] = np.log(f["share"])
    f = f.sort_values(["bank", "t"])
    prev = f.groupby("bank")[["t", "dep_a4"]].shift(1)
    f["growth"] = np.where(prev["t"] == f["t"] - 1, 100 * np.log(f["dep_a4"] / prev["dep_a4"]),
                           np.nan)
    f["pcdi"] = 100.0 * f["r4"] / (f["cdi_qoq"] / 100.0)
    own = f["own_rate"]
    for c in ("sp_pp", "pcdi"):
        g = f[own]
        m = (g[c] * g["dep_a4"]).groupby(g["t"]).sum() / g["dep_a4"].groupby(g["t"]).sum()
        f[f"{c}_gap"] = np.where(own, f[c] - f["t"].map(m), np.nan)
    return f.reset_index(drop=True)


def anova_tables(out: pathlib.Path, png: bool):
    """Four tables, B and D firms, balance-weighted (type-4 balances):
    1. variance of prices and volumes split into common to all banks (quarter), persistent by
       bank, and the rest, order-free (Shapley);
    2. co-movement of the change in a bank's relative price and in its log share over 1, 4, 8 and
       12 quarters (correlation, and the slope with the conglomerate wild bootstrap);
    3. the covariance of the relative price and the log share split into between-bank and
       within-bank parts;
    4. for B firms' local positions (bank x market x quarter), the variance of the log local
       share, the log local balance and the local gap split into market, bank, the bank's
       persistent position in that market, and the rest."""
    payload, md = {}, []

    # 1. Levels -----------------------------------------------------------------------------
    rows1 = []
    variables = (("sp_pp", "Spread, pp", True), ("pcdi", "Rate, % of CDI", True),
                 ("log_dep", "Log time deposits", False), ("log_share", "Log share of time deposits", False),
                 ("growth", "Quarterly growth of time deposits, %", False))
    panels = {k: _bank_panel(k) for k in ("B", "D")}
    for kind, f in panels.items():
        for col, label, price in variables:
            d = f[f["own_rate"]] if price else f
            d = d[np.isfinite(d[col])]
            s = _shapley2(d[col], d["dep_a4"], d["t"], d["bank"])
            payload.setdefault("levels", {}).setdefault(kind, {})[col] = dict(
                s, n=int(len(d)), banks=int(d["bank"].nunique()))
            rows1.append(f"| {kind} | {label} | {s['a']:.3f} | {s['b']:.3f} | {s['rest']:.3f} | "
                         f"{len(d):,} | {d['bank'].nunique()} |")
    md += ["Table: Sources of variation in prices and volumes, share of the balance-weighted "
           "variance (order-free split).", "",
           "| Firms | Variable | Common to all banks | Persistent by bank | Rest | Obs. | Banks |",
           "| --------- | ---------------------------------------- | ------------ | ------------ | --------- | ----------- | -------- |",
           *rows1, ""]

    # 2. Co-movement by horizon ---------------------------------------------------------------
    rows2 = []
    for kind, f in panels.items():
        g = f[f["own_rate"]][["bank", "t", "sp_pp_gap", "log_share", "dep_a4"]]
        for k in COMOVE_H:
            lag = g[["bank", "t", "sp_pp_gap", "log_share"]].copy()
            lag["t"] += k
            d = g.merge(lag, on=["bank", "t"], suffixes=("", "_k"))
            d["dgap"] = d["sp_pp_gap"] - d["sp_pp_gap_k"]
            d["dshare"] = 100 * (d["log_share"] - d["log_share_k"])
            d = _trim(d, ["dgap", "dshare"])
            r = _wls_slope(d, "dgap", "dshare", "dep_a4", "bank", fe=("t",))
            # Balance-weighted, like the slope, so the two share a sign.
            c = np.cov(r["xr"], r["yr"], aweights=r["w"])
            corr = float(c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]))
            if np.sign(corr) != np.sign(r["slope"]):
                sys.exit(f"anova: weighted correlation {corr:.3f} and slope {r['slope']:.3f} "
                         f"differ in sign ({kind}, h={k})")
            payload.setdefault("comove", {}).setdefault(kind, {})[f"h{k}"] = dict(
                corr=corr, slope=r["slope"], lo=r["lo"], hi=r["hi"], stars=r["stars"],
                n=r["n"], banks=r["banks"])
            rows2.append(f"| {kind} | {k} | {_plain(corr, 3)} | {_plain(r['slope'], 3)}{r['stars']} | "
                         f"[{_plain(r['lo'], 3)}, {_plain(r['hi'], 3)}] | {r['n']:,} | {r['banks']} |")
    md += ["Table: Co-movement of relative price and share over different horizons. Change in the "
           "spread gap (pp) against the change in the log share of time deposits (percent), net of "
           "quarter effects.", "",
           "| Firms | Horizon, quarters | Correlation | Slope | 95% interval | Obs. | Banks |",
           "| --------- | ---------- | ------------ | ------------ | --------------------- | ----------- | -------- |",
           *rows2, ""]

    # 3. Covariance decomposition ---------------------------------------------------------------
    rows3 = []
    for kind, f in panels.items():
        d = f[f["own_rate"] & np.isfinite(f["sp_pp_gap"])].copy()
        w = d["dep_a4"].to_numpy(float)
        # Quarter-demeaned log share (the gap is already relative to the quarter's mean).
        d["ls"] = _wdemean(d[["log_share"]], [d["t"]], w)["log_share"]
        tot = _wdemean(d[["sp_pp_gap", "ls"]], [], w)
        cov = float(np.average(tot["sp_pp_gap"] * tot["ls"], weights=w))
        within = _wdemean(d[["sp_pp_gap", "ls"]], [d["bank"]], w)
        cov_w = float(np.average(within["sp_pp_gap"] * within["ls"], weights=w))
        between = tot - within
        cov_b = cov - cov_w

        def corr(a, b):
            return float(np.average(a * b, weights=w)
                         / np.sqrt(np.average(a ** 2, weights=w) * np.average(b ** 2, weights=w)))
        res = dict(cov=cov, between=cov_b, within=cov_w,
                   corr=corr(tot["sp_pp_gap"], tot["ls"]),
                   corr_between=corr(between["sp_pp_gap"], between["ls"]),
                   corr_within=corr(within["sp_pp_gap"], within["ls"]),
                   n=int(len(d)), banks=int(d["bank"].nunique()))
        payload.setdefault("covdecomp", {})[kind] = res
        rows3.append(f"| {kind} | {_plain(cov, 3)} | {_plain(cov_b, 3)} | {_plain(cov_w, 3)} | "
                     f"{cov_b / cov:.3f} | {_plain(res['corr'], 3)} | "
                     f"{_plain(res['corr_between'], 3)} | {_plain(res['corr_within'], 3)} | "
                     f"{res['n']:,} | {res['banks']} |")
    md += ["Table: Covariance of the spread gap (pp) and the log share of time deposits "
           "(quarter-demeaned), split into a between-bank part (persistent differences) and a "
           "within-bank part (changes over time).", "",
           "| Firms | Cov. | Between | Within | Share between | Corr. | Corr. between | Corr. within | Obs. | Banks |",
           "| --------- | -------- | ------------ | --------- | ------------ | -------- | ------------ | ---------- | --------- | --------- |",
           *rows3, ""]

    # 4. Local nested variance (B firms) ------------------------------------------------------
    _, cells = relprice_frames()
    df = motiv_panel()
    b = df[df["bank_type"] == "B"]
    loc = pd.DataFrame({"bank": b["CodConglomeradoPrudencial"], "mkt": b["mca_code"].astype(str),
                        "t": b["year"] * 4 + b["quarter"] - 1, "dep": b["dep_a4"]})
    loc = loc[loc["dep"] > 0].copy()
    loc["lshare"] = np.log(loc["dep"] / loc.groupby(["mkt", "t"])["dep"].transform("sum"))
    loc["ldep"] = np.log(loc["dep"])
    loc = loc.merge(cells[["bank", "mkt", "t", "lgap"]], on=["bank", "mkt", "t"], how="left")
    loc["pair"] = loc["bank"].astype(str) + "|" + loc["mkt"]
    rows4 = []
    for col, label in (("lshare", "Log local share of B time deposits"),
                       ("ldep", "Log local time deposits"), ("lgap", "Gap to local rivals, pp")):
        d = loc[np.isfinite(loc[col])]
        w = d["dep"]
        r_m = _wr2(d[col], w, [d["mkt"]])
        r_mb = _wr2(d[col], w, [d["mkt"], d["bank"]])
        r_p = _wr2(d[col], w, [d["pair"]])
        res = dict(market=r_m, bank=r_mb - r_m, bank_in_market=r_p - r_mb, rest=1.0 - r_p,
                   n=int(len(d)), banks=int(d["bank"].nunique()), markets=int(d["mkt"].nunique()))
        payload.setdefault("local", {})[col] = res
        rows4.append(f"| {label} | {res['market']:.3f} | {res['bank']:.3f} | "
                     f"{res['bank_in_market']:.3f} | {res['rest']:.3f} | {res['n']:,} |")
    md += ["Table: Sources of variation in B firms' local positions (bank by market by quarter), "
           "share of the balance-weighted variance, added in sequence.", "",
           "| Variable | Market | Bank | Bank in that market | Rest (over time) | Obs. |",
           "| ------------------------------------------ | ---------- | ---------- | ------------ | ------------ | ------------ |",
           *rows4, ""]

    _store("anova", payload)
    _dump(ANOVA_MD, "\n".join(md))
    print("\n".join(md))


# --------------------------------------------------------------------------------------------
# The motivation note's tables as LaTeX inputs
# --------------------------------------------------------------------------------------------
def _tn(v: float, d: int = 3) -> str:
    """A number for a LaTeX table: negative values in math mode, so the minus is a true minus."""
    s = _plain(v, d)
    return f"${s.replace(MINUS, '-')}$" if s.startswith(MINUS) else s


def _tci(lo: float, hi: float, d: int = 3) -> str:
    return f"[{_tn(lo, d)}, {_tn(hi, d)}]"


def _tstars(s: str) -> str:
    return f"$^{{{s}}}$" if s else ""


def _panel_row(text: str, ncol: int) -> str:
    return r"\multicolumn{%d}{@{}l}{\textit{%s}} \\" % (ncol, text)


def _tab_tex(name: str, comment: list, colspec: str, header: list, body: list) -> str:
    head = [f"% {name}.tex -- GENERATED by make_slide_figures.py (--only motivtables). Do not edit.",
            "% Needs booktabs. Table body only: \\input it inside your own table environment."]
    head += [f"% {c}" for c in comment]
    return "\n".join(head + [r"\begin{tabular}{%s}" % colspec, r"\toprule", *header, r"\midrule",
                             *body, r"\bottomrule", r"\end{tabular}", ""])


def motivation_tables(out: pathlib.Path):
    """The motivation note's ten tables (motivation_figures_review.md) as booktabs bodies,
    tab_motiv_*.tex, built from the unrounded values of Rout/slide_motivation.json. Nothing is
    re-estimated: run the charts first (band, pricevol, relprice, anova, reprice, entrygap)."""
    s = json.loads(MOTIV_JSON.read_text(encoding="utf-8"))
    src = f"Values: {MOTIV_JSON.name}"
    tabs = {}

    # Dispersion of the B-firm time-deposit spread by year ------------------------------------
    rows = [f"{y} & {_tn(v['mean'])} & {_tn(v['sd'])} \\\\"
            for y, v in s["band"]["by_year"].items()]
    tabs["tab_motiv_spread_dispersion"] = (
        [f"{src}, band.by_year; keys \\pn{{slide.motiv.band.*}}. Time-deposit spread across B",
         "firms, pp a year, weighted as in Table A.4 (the mean is Table A.4's B line)."],
        "@{}lrr@{}", [r"Year & Mean & Std.\ dev. \\"], rows)

    # Prices and balances over the tightening ------------------------------------------------
    h = s["pricevol"]["hikes"]
    ci = h["ci"]

    def two(key, scale=1.0, interval=True):
        out_ = f"{_tn(h[key][0] * scale)} & {_tn(h[key][1] * scale)} \\\\"
        if interval and key in ci:
            c = ci[key]
            out_ += (f"\n & {_tci(c['start'][0] * scale, c['start'][1] * scale)} & "
                     f"{_tci(c['end'][0] * scale, c['end'][1] * scale)} \\\\")
        return out_
    bp = 100.0     # pp to basis points
    pmark = " (CDI rate)" if prepaid_facts()["fallback"] else ""
    rows = [_panel_row("Panel A. Spreads", 3),
            r"\quad Demand (equal to Selic), pp & " + two("spread_demand"),
            r"\quad Savings, pp & " + two("spread_savings"),
            r"\quad Time deposits, B firms, pp & " + two("spread_time_B"),
            r"\quad Time deposits, D firms, pp & " + two("spread_time_D"),
            rf"\quad Prepaid accounts, B firms, bp{pmark} & " + two("spread_prepaid_B", bp),
            rf"\quad Prepaid accounts, D firms, bp{pmark} & " + two("spread_prepaid_D", bp),
            r"\addlinespace",
            _panel_row(r"Panel B. Share of total balances, \%", 3),
            r"\quad Demand deposits & " + two("share_demand"),
            r"\quad Savings deposits & " + two("share_savings"),
            r"\quad Time deposits, B firms & " + two("share_time_B"),
            r"\quad Time deposits, D firms & " + two("share_time_D"),
            r"\quad Prepaid accounts & " + two("share_prepaid")]
    tabs["tab_motiv_tightening"] = (
        [f"{src}, pricevol.hikes; keys \\pn{{slide.motiv.hikes.*}}. Brackets: 95% intervals of the",
         "balance-weighted bank means (conglomerate wild cluster bootstrap, Webb, 999 draws, bias-",
         "corrected percentile). Prepaid spreads in basis points (pp x 100). (CDI rate): no",
         "institution-level prepaid rate is observed, so the prepaid spread is the Selic-CDI gap."],
        "@{}lrr@{}", [f" & {h['start']} & {h['end']} \\\\"], rows)

    # Relative prices and share changes -------------------------------------------------------
    sh = s["relprice"]["share"]

    def slope_row(label, r):
        return (f"\\quad {label} & {_tn(r['slope'])} & {_tci(r['lo'], r['hi'])} & {r['n']:,} & "
                f"{r['banks']} \\\\")
    rows = [_panel_row("Per bank: average gap, yearly change in share", 5),
            slope_row("Gap in pp", sh["xs_pp"]), slope_row("Gap in CDI points", sh["xs_pcdi"]),
            r"\addlinespace",
            _panel_row("Bank-quarter: gap, change in share over 4 quarters", 5),
            slope_row("Gap in pp", sh["panel_pp"]), slope_row("Gap in CDI points", sh["panel_pcdi"]),
            r"\addlinespace",
            _panel_row("Local: gap to rivals (pp), change in local share", 5),
            slope_row("All variation", sh["local_raw"]),
            slope_row("Within bank-quarter and market", sh["local_fe"])]
    tabs["tab_motiv_relprice_share"] = (
        [f"{src}, relprice.share; keys \\pn{{slide.motiv.relprice.share.*}}. Slopes in percent of",
         "the share per unit of gap; 95% intervals from the conglomerate wild cluster bootstrap."],
        "@{}lrcrr@{}", [r" & Slope & 95\% interval & Obs. & Banks \\"], rows)

    # Analysis of variance: levels ------------------------------------------------------------
    a = s["anova"]
    labels = {"sp_pp": "Spread, pp", "pcdi": r"Rate, \% of CDI", "log_dep": "Log time deposits",
              "log_share": "Log share of time deposits",
              "growth": r"Quarterly growth of time deposits, \%"}
    rows = []
    for kind in ("B", "D"):
        if rows:
            rows.append(r"\addlinespace")
        rows.append(_panel_row(f"{kind} firms", 6))
        for col, lab in labels.items():
            v = a["levels"][kind][col]
            rows.append(f"\\quad {lab} & {v['a']:.3f} & {v['b']:.3f} & {v['rest']:.3f} & "
                        f"{v['n']:,} & {v['banks']} \\\\")
    tabs["tab_motiv_anova_levels"] = (
        [f"{src}, anova.levels; keys \\pn{{slide.motiv.anova.levels.*}}. Shares of the",
         "balance-weighted variance, order-free (Shapley) split between quarter and bank effects."],
        "@{}lrrrrr@{}",
        [r" & Common & Persistent & & & \\",
         r"Variable & to all banks & by bank & Rest & Obs. & Banks \\"], rows)

    # Co-movement by horizon ------------------------------------------------------------------
    rows = []
    for kind in ("B", "D"):
        if rows:
            rows.append(r"\addlinespace")
        rows.append(_panel_row(f"{kind} firms", 6))
        for k in COMOVE_H:
            v = a["comove"][kind][f"h{k}"]
            rows.append(f"\\quad {k} & {_tn(v['corr'])} & {_tn(v['slope'])}{_tstars(v['stars'])} & "
                        f"{_tci(v['lo'], v['hi'])} & {v['n']:,} & {v['banks']} \\\\")
    tabs["tab_motiv_anova_comove"] = (
        [f"{src}, anova.comove; keys \\pn{{slide.motiv.anova.comove.*}}. Change in the spread gap",
         "(pp) against the change in the log share (percent) over k quarters, net of quarter",
         "effects; stars from the conglomerate wild cluster bootstrap (* 10%, ** 5%, *** 1%)."],
        "@{}lrrcrr@{}",
        [r"Horizon, quarters & Correlation & Slope & 95\% interval & Obs. & Banks \\"], rows)

    # Covariance between and within banks -----------------------------------------------------
    rows = []
    for kind in ("B", "D"):
        v = a["covdecomp"][kind]
        rows.append(f"{kind} & {_tn(v['cov'])} & {_tn(v['between'])} & {_tn(v['within'])} & "
                    f"{_tn(v['between'] / v['cov'])} & {_tn(v['corr'])} & {_tn(v['corr_between'])} & "
                    f"{_tn(v['corr_within'])} & {v['n']:,} & {v['banks']} \\\\")
    tabs["tab_motiv_anova_covdecomp"] = (
        [f"{src}, anova.covdecomp; keys \\pn{{slide.motiv.anova.covdecomp.*}}. Spread gap (pp)",
         "and quarter-demeaned log share of time deposits; between = bank means, within = deviations."],
        "@{}lrrrrrrrrr@{}",
        [r" & \multicolumn{3}{c}{Covariance} & Between & \multicolumn{3}{c}{Correlation} & & \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){6-8}",
         r"Firms & Total & Between & Within & share & Total & Between & Within & Obs. & Banks \\"],
        rows)

    # Local positions of B firms --------------------------------------------------------------
    rows = []
    for col, lab in (("lshare", "Log local share of B time deposits"),
                     ("ldep", "Log local time deposits"), ("lgap", "Gap to local rivals, pp")):
        v = a["local"][col]
        rows.append(f"{lab} & {v['market']:.3f} & {v['bank']:.3f} & {v['bank_in_market']:.3f} & "
                    f"{v['rest']:.3f} & {v['n']:,} \\\\")
    tabs["tab_motiv_anova_local"] = (
        [f"{src}, anova.local; keys \\pn{{slide.motiv.anova.local.*}}. Bank by market by quarter,",
         "B firms; shares of the balance-weighted variance, added in sequence."],
        "@{}lrrrrr@{}",
        [r" & & & Bank in & Rest & \\",
         r"Variable & Market & Bank & that market & (over time) & Obs. \\"], rows)

    # Own repricing ---------------------------------------------------------------------------
    rows = []
    for kind, sec in (("B", "reprice"), ("D", "reprice_d")):
        r = s[sec]
        if rows:
            rows.append(r"\addlinespace")
        rows.append(_panel_row(f"{kind} firms", 5))
        for key, lab in (("main", f"Next {r['growth_h']} quarters"),
                         ("untrimmed", f"Next {r['growth_h']} quarters, untrimmed"),
                         ("same_quarter", "The same quarter")):
            v = r[key]
            rows.append(f"\\quad {lab} & {_tn(v['slope'])} & {_tci(v['lo_bc'], v['hi_bc'])} & "
                        f"{v['n']:,} & {v['banks']} \\\\")
    tabs["tab_motiv_reprice"] = (
        [f"{src}, reprice and reprice_d; keys \\pn{{slide.motiv.reprice.*}} and",
         "\\pn{slide.motiv.reprice_d.*}. Percent of time deposits per pp of spread change; bank and",
         "quarter effects; conglomerate wild cluster bootstrap. The same-quarter slope is",
         "mechanically biased (the rate is expense over the lagged stock)."],
        "@{}lrcrr@{}",
        [r"Growth measured over & Slope & 95\% interval & Obs. & Banks \\"], rows)

    # Entrants by price at entry --------------------------------------------------------------
    e = s["entrygap"]
    thr = e["threshold_pp"]
    rows = []
    for title, groups, boot in (("Normalized share path", e["groups"], e["bankboot"]["main"]),
                                (r"Share of B deposits, \%", e["level_pct"], e["bankboot"]["level"])):
        if rows:
            rows.append(r"\addlinespace")
        rows.append(_panel_row(title, 7))
        for g, lab in (("more", f"Pays at least {thr:g} pp more"), ("rest", "The rest")):
            m, b = groups[g]["median_at_h"], boot[g]
            rows.append(f"\\quad {lab} & {_tn(m['4'])} & {_tn(m['8'])} & {_tn(m['12'])} & "
                        f"{_tci(*b['ci_at_h']['12'])} & {b['n']} & {b['banks']} \\\\")
    tabs["tab_motiv_entry_by_gap"] = (
        [f"{src}, entrygap; keys \\pn{{slide.motiv.entrygap.*}}. Medians of the entrants' share",
         "paths; the interval resamples banks (999 draws). Normalization (s_h - s_0)/(s_end - s_0)."],
        "@{}lrrrcrr@{}",
        [r" & \multicolumn{3}{c}{Median at quarter} & 95\% interval & & \\",
         r"\cmidrule(lr){2-4}",
         r" & 4 & 8 & 12 & at quarter 12 & Entries & Banks \\"], rows)

    # Construction of the repricing samples ---------------------------------------------------
    wb, wd = s["reprice"]["waterfall"], s["reprice_d"]["waterfall"]
    assert len(wb) == len(wd) == 6, "the repricing waterfall changed: update the row labels"
    y0, y1 = (str(x)[:4] for x in s["window"])
    floor = s["reprice"]["type4_floor"] / 1e6
    lo_p, hi_p = s["reprice"]["trim_pct"]
    steps = (f"Holds time deposits, {y0}--{y1}", "Own measured rate",
             "Own rate in the previous quarter too", f"Book of at least R\\$ {floor:,.0f} million",
             f"Balance observed {s['reprice']['growth_h']} quarters later",
             f"Inside the {lo_p:g}st--{hi_p:g}th percentiles")
    rows = [f"{lab} & {b['rows']:,} & {b['banks']} & {d['rows']:,} & {d['banks']} \\\\"
            for lab, b, d in zip(steps, wb, wd)]
    tabs["tab_motiv_reprice_samples"] = (
        [f"{src}, reprice.waterfall and reprice_d.waterfall. Each row applies one more filter;",
         "observations are conglomerate-quarters."],
        "@{}lrrrr@{}",
        [r" & \multicolumn{2}{c}{B firms} & \multicolumn{2}{c}{D firms} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Filter & Obs. & Banks & Obs. & Firms \\"], rows)

    for name, (comment, colspec, header, body) in tabs.items():
        _dump(out / f"{name}.tex", _tab_tex(name, comment, colspec, header, body))


def xi_anova_slide(out: pathlib.Path, png: bool, E: int):
    """The order-free split of the demand residual's balance-weighted variance for routine E, time
    deposits and all deposit types, as stacked bars (Rout/xi_diagnostics.json, written by
    make_xi_diagnostics.py with tab_xi_diagnostics.tex), styled as the motivation decomposition
    bars: the deck copy fig_slide_xi_anova_slide, 8.0 x 4.6 cm at the `_slide` fonts."""
    global SLIDE_FIGSIZE, FS, PAPER, SLIDE, SLIDE3, MARK
    if not XI_JSON.is_file():
        sys.exit(f"{XI_JSON} not found: run make_xi_diagnostics.py first")
    res = json.loads(XI_JSON.read_text(encoding="utf-8")).get(f"E{E}")
    if res is None:
        sys.exit(f"{XI_JSON.name} has no E{E}: run make_xi_diagnostics.py --routines {E}")
    FS, PAPER, SLIDE, SLIDE3 = SLIDE_FS, False, True, True
    SLIDE_FIGSIZE = SLIDE3_STEMS["fig_slide_xi_anova"][1]
    MARK = (SLIDE_FIGSIZE[0] * SLIDE_FIGSIZE[1]) / (5.2 * 3.3)
    parts = (("quarter", "Quarter", mdt.NATL_COLOR), ("bank", "Bank", mdt.B_COLOR),
             ("bank_type", "Bank × type", "#64B5F6"), ("market", "Market", SAVINGS_COLOR),
             ("rest", "Rest", "#E0E0E0"))
    rows = (("Time\ndeposits", res["k4"]["anova"]), ("All\ntypes", res["all"]["anova"]))
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    for i, (_lab, vals) in enumerate(rows):
        left = 0.0
        for k, name, color in parts:
            v = 100.0 * vals[k]
            ax.barh(i, v, left=left, color=color, edgecolor="white", height=0.6,
                    label=name if i == 0 else None)
            # Percent without the sign (the title carries it): a 9-pt "11%" overruns a 10% bar.
            if v >= 7:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5 * FS,
                        color="#111111" if color in ("#E0E0E0", "#64B5F6") else "white")
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    _style_no_pix(ax, "Residual variance, %", "", "")
    ax.grid(False)
    ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              frameon=False, handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
              labelspacing=0.3, borderaxespad=0.0, borderpad=0.0)
    _save(fig, out, "fig_slide_xi_anova", png)
    SLIDE3 = False


def dep_anova_slide(out: pathlib.Path, png: bool):
    """The order-free split of the balance-weighted variance of B firms' log deposits
    (Rout/dep_anova.json, written by make_dep_anova.py with tab_dep_anova.tex): main effects and
    with bank x product and market x product, as stacked bars styled as fig_slide_xi_anova_slide,
    8.0 x 4.6 cm at the `_slide` fonts."""
    global SLIDE_FIGSIZE, FS, PAPER, SLIDE, SLIDE3, MARK
    if not DEP_ANOVA_JSON.is_file():
        sys.exit(f"{DEP_ANOVA_JSON} not found: run make_dep_anova.py first")
    res = json.loads(DEP_ANOVA_JSON.read_text(encoding="utf-8"))["B"]
    FS, PAPER, SLIDE, SLIDE3 = SLIDE_FS, False, True, True
    SLIDE_FIGSIZE = SLIDE3_STEMS["fig_slide_dep_anova"][1]
    MARK = (SLIDE_FIGSIZE[0] * SLIDE_FIGSIZE[1]) / (5.2 * 3.3)
    parts = (("time", "Time", mdt.NATL_COLOR), ("bank", "Bank", mdt.B_COLOR),
             ("bank_product", "Bank×type", "#64B5F6"), ("product", "Type", "#9575CD"),
             ("market", "Market", SAVINGS_COLOR), ("market_product", "Mkt×type", "#80CBC4"),
             ("rest", "Rest", "#E0E0E0"))
    rows = (("Main", res["base"]), ("With\ninter.", res["int"]))
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    for i, (_lab, vals) in enumerate(rows):
        left = 0.0
        for k, name, color in parts:
            v = 100.0 * vals.get(k, 0.0)
            ax.barh(i, v, left=left, color=color, edgecolor="white", height=0.6,
                    label=name if i == 1 else None)
            if v >= 7:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5 * FS,
                        color="#111111" if color in ("#E0E0E0", "#64B5F6", "#80CBC4")
                        else "white")
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    _style_no_pix(ax, "Log deposits, variance, %", "", "")
    ax.grid(False)
    ax.legend(fontsize=8.5 * FS, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4,
              frameon=False, handlelength=0.8, handletextpad=0.3, columnspacing=0.6,
              labelspacing=0.3, borderaxespad=0.0, borderpad=0.0)
    _save(fig, out, "fig_slide_dep_anova", png)
    SLIDE3 = False


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
    if "relprice" in todo:
        relprice_charts(args.out, args.png)
    if "anova" in todo:
        anova_tables(args.out, args.png)
    if "motivtables" in todo:
        motivation_tables(args.out)
    # Last: each switches the figure size and fonts for the rest of the run, and sets its own.
    if "paper" in todo:
        paper_figures(args.out, args.png)
    if "slidesize" in todo:
        slide_size_figures(args.out, args.png)
    if "slidesize2" in todo:
        slide2_figures(args.out, args.png)
    if "slidesize3" in todo:
        slide3_figures(args.out, args.png)
    if "xi" in todo:
        xi_anova_slide(args.out, args.png, args.routine)
    if "depanova" in todo:
        dep_anova_slide(args.out, args.png)
    if "slidesize4" in todo:
        slide4_figures(args.out, args.png)
    if "slidesize5" in todo:
        slide5_figures(args.out, args.png)
    if "slidesize6" in todo:
        slide6_figures(args.out, args.png)
    if "ref" in todo:
        ref_figures(args.out, args.png)
    if "refame" in todo:
        ref_ame(args.out, args.png)
    if "followups" in todo:
        followups_ref(args.out, args.png, args.routine)
    if "pixexposure" in todo:
        pix_exposure_ref(args.out, args.png)


if __name__ == "__main__":
    main()
