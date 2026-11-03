"""make_ccs_flows_report.py -- descriptive tables and two figures of the CCS relationship flows
(LAI 18810.025400/2026-04), PF only, for Pedro. No paper numbers.

Reads utils.paths.rout_dir() / ccs_flows_institution_month_mapped.csv (make_ccs_flows_conglomerate.py).
Firm groups: B = panel conglomerate with is_B = 1, D = panel conglomerate with is_B = 0, "not in panel"
= every other institution (unmapped roots included). Caixa (C0080738, a B firm) is drawn on its own.
"Excl. events" removes the PF closures of the 25 flagged institution-months (event_closed).

Outputs (rout_dir()):
  ccs_flows_report_tables.md            the five tables of the report
  fig_ccs_flows_monthly_pf.pdf          monthly gross openings and closures, one panel per group
  fig_ccs_flows_dshare_retention_pf.pdf D share of gross openings; cohort retention by firm type

Reading cautions: a relationship is one CPF at one institution (not a person, not an account);
"active" is measured at an unstated extraction date after 2024-12; dormant accounts count as active;
closures mix client choices with corporate events; 2020 openings are partly policy-driven.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import paths  # noqa: E402

ROUT = paths.rout_dir()
SRC = ROUT / "ccs_flows_institution_month_mapped.csv"
TABLES = ROUT / "ccs_flows_report_tables.md"
FIG1 = ROUT / "fig_ccs_flows_monthly_pf.pdf"
FIG2 = ROUT / "fig_ccs_flows_dshare_retention_pf.pdf"
CAIXA = "C0080738"
# Categorical slots 1-3 of the reference palette (dataviz skill), validated light-mode.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GROUPS = ["B", "D", "not in panel"]


def group(d: pd.DataFrame) -> pd.Series:
    return np.select([(d["in_panel"] == 1) & (d["is_B"] == 1), (d["in_panel"] == 1) & (d["is_B"] == 0)],
                     ["B", "D"], "not in panel")


def md_table(df: pd.DataFrame, fmt="{:,.2f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                 else ("" if (isinstance(v, float) and not np.isfinite(v)) else str(v)) for v in r]
        lines.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(lines)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, width=0.6)


def main():
    d = pd.read_csv(SRC, dtype={"cnpj8": str})
    d = d[d["type"] == "PF"].copy()
    d["group"] = group(d)
    d["year"] = d["ym"].str[:4].astype(int)
    d["closed_ex"] = d["closed"] - d["event_closed"]
    out = ["# CCS relationship flows, PF (LAI 18810.025400/2026-04)", "",
           "Millions of relationships unless stated. A relationship is one CPF at one institution. "
           "\"Excl. events\" drops the PF closures of the 25 flagged institution-months.", ""]

    # 1. gross openings, closures, net by year and group
    t = d.groupby(["year", "group"])[["opened_gross", "closed", "closed_ex"]].sum() / 1e6
    t["net"] = t["opened_gross"] - t["closed"]
    t["net_excl_events"] = t["opened_gross"] - t["closed_ex"]
    w = t.unstack("group")
    t1 = pd.DataFrame({f"{g}: {lab}": w[(m, g)] for g in GROUPS
                       for m, lab in (("opened_gross", "gross"), ("closed", "closed"),
                                      ("closed_ex", "closed excl. ev."), ("net", "net"))})
    t1.index.name = "year"
    cx = d[d["conglomerate"] == CAIXA].groupby("year")[["opened_gross", "closed"]].sum() / 1e6
    out += ["## 1. Gross openings, closures and net flows by year and firm type", "", md_table(t1), "",
            "Memo, Caixa (inside B): gross " + ", ".join(f"{y} {v:.2f}" for y, v in cx["opened_gross"].items())
            + "; closed " + ", ".join(f"{y} {v:.2f}" for y, v in cx["closed"].items()), ""]

    # 2. D share of gross openings
    g = d.groupby(["year", "group"])["opened_gross"].sum().unstack("group")
    t2 = pd.DataFrame({"D / all": g["D"] / g.sum(axis=1), "D / (B + D)": g["D"] / (g["B"] + g["D"]),
                       "not in panel / all": g["not in panel"] / g.sum(axis=1)})
    t2.index.name = "year"
    out += ["## 2. D share of gross openings", "", md_table(t2, "{:.3f}"), ""]

    # 3. cohort retention by opening year and group
    r = d.groupby(["year", "group"])[["opened_active", "opened_gross"]].sum()
    t3 = (r["opened_active"] / r["opened_gross"]).unstack("group")[GROUPS]
    t3["all"] = d.groupby("year")["opened_active"].sum() / d.groupby("year")["opened_gross"].sum()
    t3.index.name = "opening year"
    out += ["## 3. Cohort retention: opened and still active / gross openings", "",
            "Active at the BCB's unstated extraction date (after 2024-12), so later cohorts have had less "
            "time to close.", "", md_table(t3, "{:.3f}"), ""]

    # 5. top 15 conglomerates by gross openings in 2024
    y24 = d[d["year"] == 2024]
    names = (y24.groupby(["conglomerate", "name"])["opened_gross"].sum().reset_index()
             .sort_values("opened_gross", ascending=False).drop_duplicates("conglomerate")
             .set_index("conglomerate")["name"])
    top = y24.groupby("conglomerate").agg(opened_gross=("opened_gross", "sum"), closed=("closed", "sum"),
                                          opened_active=("opened_active", "sum"), group=("group", "first"),
                                          n_institutions=("cnpj8", "nunique"))
    top = top.sort_values("opened_gross", ascending=False).head(15)
    t5 = pd.DataFrame({"largest institution": names.reindex(top.index), "group": top["group"],
                       "institutions": top["n_institutions"].astype(str),
                       "gross": top["opened_gross"] / 1e6, "closed": top["closed"] / 1e6,
                       "retention": top["opened_active"] / top["opened_gross"]})
    t5.index.name = "conglomerate"
    out += ["## 5. The 15 conglomerates with the largest PF gross openings in 2024", "", md_table(t5), "",
            "A conglomerate's sum counts a client once per institution (Nu Pagamentos IP and Nu Financeira "
            "are one conglomerate).", ""]
    TABLES.write_text("\n".join(out), encoding="utf-8")

    # ── Figure 1: monthly gross openings and closures, one panel per group ───────────────────────
    d["date"] = pd.to_datetime(d["ym"] + "-01")
    panels = [("Caixa", d["conglomerate"] == CAIXA),
              ("B firms excl. Caixa", (d["group"] == "B") & (d["conglomerate"] != CAIXA)),
              ("D firms", d["group"] == "D"),
              ("Not in the panel", d["group"] == "not in panel")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    for ax, (title, mask) in zip(axes.flat, panels):
        s = d[mask].groupby("date")[["opened_gross", "closed", "closed_ex"]].sum() / 1e6
        ax.plot(s.index, s["opened_gross"], color=BLUE, lw=1.6)
        ax.plot(s.index, s["closed"], color=ORANGE, lw=1.6)
        if (s["closed"] - s["closed_ex"]).abs().max() > 0.01:
            ax.plot(s.index, s["closed_ex"], color=ORANGE, lw=1.2, ls=(0, (3, 2)))
        # The two reference lines are seven months apart: label the first on its left, the second on its right.
        for x, lab, ha in ((pd.Timestamp("2020-04-01"), "2020-04 ", "right"), (pd.Timestamp("2020-11-01"), " Pix", "left")):
            ax.axvline(x, color=INK2, lw=0.6, ls=":")
            ax.text(x, 1.0, lab, transform=ax.get_xaxis_transform(), color=INK2, fontsize=7,
                    va="top", ha=ha)
        style(ax)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("millions per month", fontsize=8, color=INK2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, lw=1.6), Line2D([], [], color=ORANGE, lw=1.6),
               Line2D([], [], color=ORANGE, lw=1.2, ls=(0, (3, 2)))]
    fig.legend(handles, ["Gross openings", "Closures", "Closures excl. flagged events"], loc="upper center",
               ncol=3, frameon=False, fontsize=8.5)
    fig.text(0.01, 0.005, "PF relationships (one CPF at one institution). Panels have their own scales. "
             "Dotted lines: 2020-04 (Caixa's digital savings for the emergency aid) and 2020-11 (Pix launch).",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(FIG1)
    plt.close(fig)

    # ── Figure 2: D share of gross openings; cohort retention by firm type ────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(t2.index, t2["D / all"], color=BLUE, lw=1.8, marker="o", ms=4)
    for x, v in t2["D / all"].items():
        if x in (t2.index.min(), 2020, t2.index.max()):
            a1.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=7.5, color=INK)
    style(a1)
    a1.set_ylim(0, max(0.8, float(t2["D / all"].max()) * 1.15))
    a1.set_title("D firms' share of PF gross openings", loc="left", fontsize=9.5, color=INK)
    # End labels, pushed apart to at least 0.05 so that close final values do not overprint.
    ends = sorted(((float(t3[gname].iloc[-1]), gname) for gname in GROUPS), reverse=True)
    pos, last = {}, None
    for v, gname in ends:
        y = v if last is None else min(v, last - 0.05)
        pos[gname], last = y, y
    for gname, col in zip(GROUPS, (BLUE, ORANGE, AQUA)):
        a2.plot(t3.index, t3[gname], color=col, lw=1.8, marker="o", ms=4)
        a2.annotate(gname, (t3.index.max(), pos[gname]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    style(a2)
    a2.set_ylim(0, 1)
    a2.set_xlim(t3.index.min() - 0.3, t3.index.max() + 1.6)
    a2.set_title("Cohort retention: still active / gross openings", loc="left", fontsize=9.5, color=INK)
    fig.legend([Line2D([], [], color=c, lw=1.8) for c in (BLUE, ORANGE, AQUA)], GROUPS,
               loc="lower right", ncol=3, frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Retention is measured at the BCB's unstated extraction date after 2024-12, so "
             "recent cohorts have had less time to close.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG2)
    plt.close(fig)
    print(f"wrote {TABLES.name}, {FIG1.name}, {FIG2.name}")


if __name__ == "__main__":
    main()
