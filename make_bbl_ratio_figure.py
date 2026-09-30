#!/usr/bin/env python
"""make_bbl_ratio_figure.py -- replaces Table 10 (tab_bbl_ridge_by_start) with a figure.

The table printed the launch-quarter median of dpsi_4/dpsi_2 -- the forward rate a
deviation's change in deposits is priced at -- for routines (III) and (IV), 36 rows plus a
pooled row, and buried the one thing worth seeing in a column of numbers: the median moves
with the rate cycle. This script draws that movement directly, overlaid with the two rate
series that drive it -- the realized Selic at each launch quarter and that vintage's
expected neutral rate (the flat tail of its Focus curve) -- so the identifying variation
that separates omega from zeta across the cycle is visible at a glance.

Reads (never recomputed -- these are make_bbl_cost_tables.py's own numbers, verified equal
to Table 10 to 3dp for every quarter and both pooled rows):
  EO/COST_FWD/bbl_table_numbers{tag}.json  -- sections.tab_bbl_ridge_by_start.routines.{E3,E4}
                                              [quarter]['ratio_median_ann'], the same
                                              compounded-annual median the table prints.
  EO/COST_FWD/forward_rf_vintages.csv      -- realized Selic at launch = each start_q's h=0
                                              row; expected neutral rate = its terminal
                                              (flat) row. Both already compounded-annual
                                              (selic_ann_pct), verified against
                                              100*((1+rf_qoq)**4-1) to <5e-5pp.

Writes:
  Drafts/Deposit Competition/fig_bbl_ratio_by_start.pdf  -- \\includegraphics-ready, no tag
  Drafts/Deposit Competition/fig_bbl_ratio_by_start.png  -- same, for inspection
  EO/COST_FWD/fig_bbl_ratio_by_start{tag}.csv            -- the exact plotted series

Usage:
  python make_bbl_ratio_figure.py --tag _ms1
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import paths as _paths

STEP_DIR = _paths.cost_fwd_dir()
DRAFTS = _paths.drafts_dir()

QUARTERS = [f"{y}Q{q}" for y in range(2016, 2025) for q in range(1, 5)]

# Palette: E3/E4 hues match MODEL_COLORS in sleep_ident_entry_dynamics.py (Figure 2's
# routine colors), so a routine reads the same color across every figure in the paper.
# Grid/spine/ink greys are that same figure's convention.
INK, GRID, SPINE = "#4F4F4F", "#D5D5D0", "#BFBFBA"
E3_COLOR, E4_COLOR = "#6A1B9A", "#00838F"
SELIC_COLOR, NEUTRAL_COLOR = "#424242", "#9E9E9E"
ROMAN = {"E3": "(III)", "E4": "(IV)"}


def load_ratio_series(tag: str):
    """Median dpsi_4/dpsi_2 per launch quarter for E3 and E4, already compounded-annual pp,
    straight from the sidecar make_bbl_cost_tables.py renders Table 10 from."""
    path = STEP_DIR / f"bbl_table_numbers{tag}.json"
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    sec = d["sections"]["tab_bbl_ridge_by_start"]
    out = {}
    for est in ("E3", "E4"):
        routine = sec["routines"][est]
        missing = [q for q in QUARTERS if q not in routine]
        if missing:
            raise KeyError(f"{est}: sidecar is missing launch quarters {missing}")
        out[est] = np.array([routine[q]["ratio_median_ann"] for q in QUARTERS])
    return out


def load_rate_series():
    """Realized Selic at launch (h=0) and each vintage's expected neutral rate (its
    terminal, flat h), both read straight from selic_ann_pct -- verified equal to
    100*((1+rf_qoq)**4-1), the paper's compounding rule, to within rounding."""
    path = STEP_DIR / "forward_rf_vintages.csv"
    df = pd.read_csv(path)
    realized, neutral = [], []
    for q in QUARTERS:
        grp = df.loc[df["start_q"] == q].sort_values("h")
        if grp.empty:
            raise KeyError(f"forward_rf_vintages.csv has no rows for start_q={q}")
        realized.append(float(grp.iloc[0]["selic_ann_pct"]))
        neutral.append(float(grp.iloc[-1]["selic_ann_pct"]))
    return np.array(realized), np.array(neutral)


def build_frame(tag: str) -> pd.DataFrame:
    ratios = load_ratio_series(tag)
    realized, neutral = load_rate_series()
    return pd.DataFrame({
        "start_q": QUARTERS,
        "e3_median_ratio_ann_pct": ratios["E3"],
        "e4_median_ratio_ann_pct": ratios["E4"],
        "selic_realized_ann_pct": realized,
        "selic_neutral_ann_pct": neutral,
    })


def draw_figure(frame: pd.DataFrame, out_stem):
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(6.5, 2.8))

    ax.step(x, frame["selic_realized_ann_pct"], where="mid", color=SELIC_COLOR, lw=1.3,
            zorder=2, label="Realized Selic at launch")
    ax.plot(x, frame["selic_neutral_ann_pct"], color=NEUTRAL_COLOR, lw=1.3, ls="--",
            zorder=2, label="Vintage's expected neutral rate")
    # E3 and E4 sit within ~0.3pp of each other almost everywhere, so color alone would not
    # keep both visible where they coincide: E3 is drawn thick and solid, E4 thin and dashed
    # on top, so the dash texture of E4 remains legible even where it overlaps E3 exactly.
    ax.plot(x, frame["e3_median_ratio_ann_pct"], color=E3_COLOR, lw=2.1, ls="-", marker="o",
            ms=3.0, markeredgecolor="white", markeredgewidth=0.4, zorder=3,
            label=r"Median $\Delta\psi_4/\Delta\psi_2$, " + ROMAN["E3"])
    ax.plot(x, frame["e4_median_ratio_ann_pct"], color=E4_COLOR, lw=1.2, ls=(0, (4, 1.5)),
            marker="^", ms=3.2, markeredgecolor="white", markeredgewidth=0.4, zorder=4,
            label=r"Median $\Delta\psi_4/\Delta\psi_2$, " + ROMAN["E4"])

    ax.set_ylabel("Percent per Year (Compounded)", fontsize=8.5)
    ax.set_xlabel("Forward-Curve Launch Quarter", fontsize=8.5)
    ax.tick_params(axis="both", which="major", labelsize=8)

    year_starts = [i for i, q in enumerate(frame["start_q"]) if q.endswith("Q1")]
    ax.set_xticks(year_starts)
    ax.set_xticklabels([frame["start_q"].iloc[i][:4] for i in year_starts])
    ax.set_xlim(-0.5, len(frame) - 0.5)

    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(SPINE)
        ax.spines[sp].set_linewidth(0.8)

    # Placed over the 2019Q4-2021Q1 trough, where the Selic path falls to its sample low and
    # every series it and the neutral/ratio lines occupy is well below y~9: the only large
    # empty rectangle in the axes, so the box sits without covering any plotted series.
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.44, 0.98), fontsize=7.8,
                     frameon=True, framealpha=1.0, facecolor="white", edgecolor=SPINE,
                     borderpad=0.5, labelspacing=0.35, handlelength=1.9)
    leg.set_zorder(10)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_ms1",
                     help="psi vintage tag; reads bbl_table_numbers{tag}.json (default _ms1)")
    args = ap.parse_args()

    frame = build_frame(args.tag)

    csv_path = STEP_DIR / f"fig_bbl_ratio_by_start{args.tag}.csv"
    frame.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    out_stem = DRAFTS / "fig_bbl_ratio_by_start"
    draw_figure(frame, out_stem)
    print(f"wrote {out_stem}.pdf and .png")


if __name__ == "__main__":
    main()
