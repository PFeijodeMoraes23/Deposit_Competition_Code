## analysis_3_additional_figures.py
# Author: Pedro Feijó de Moraes
# Last edited: 2026-06-02
#
# Produces four supplemental figures for the extensive-vs-intensive deposit
# margin discussion (V_Main.tex sec:inst_setting:desc + app:extensions:intensive).
# All outputs saved to processed/DESCRIPTIVES/ AND mirrored next to V_Main.tex.
#
#   Figure A — Client growth trajectories (the extensive margin story)
#       D-firms: Nubank, Inter, PagSeguro, Stone (EDGAR) + Pan, BMG (CVM-IPE)
#       B-firms: BB, Banrisul (CVM-IPE)
#       Shows the explosive account-base growth 2013-2026.
#
#   Figure B — Deposits per customer over time (intensive margin gap)
#       Cleaner standalone Panel B from analysis_2; adds Pix event line (2020)
#       and segment shading. Log scale. D vs B contrast.
#
#   Figure C — Market structure: volume-share dynamics 2013-2025
#       Top-4 incumbents (Itaú/Bradesco/BB/Santander) aggregate volume share
#       vs. all digital/payment firms, from market_panel.csv. Shows how much
#       incumbents still dominate volume despite D-firm account growth.
#
#   Figure D — Findex demographic account-ownership gradients (the Pi motivation)
#       Bar chart: account ownership rate by demographic group (income, gender,
#       age, education), for Brazil survey years 2011-2025.
#       Motivates heterogeneous demographic loadings Pi/Pi^q.
###────────────────────────────────────────────────────────────────────────────

import os, sys, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

from utils.disclosure_common import data_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("analysis_3")

BASE = data_root(__file__)
DESC = os.path.join(BASE, "BCB", "Egan_et_al_2025_Rep", "processed", "DESCRIPTIVES")
DRAFTS = os.path.normpath(os.path.join(BASE, "Drafts", "Deposit Competition"))
os.makedirs(DESC, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
D_COLOURS = {
    "nubank":      "#8B00FF",
    "inter":       "#FF6B00",
    "pagseguro":   "#009C3B",
    "stone":       "#00B4D8",
    "xp":          "#F4A261",
    "mercadopago": "#003087",
    "pan":         "#E63946",
    "bmg":         "#A8DADC",
}
B_COLOURS = {
    "bb":          "#1D3557",
    "itau":        "#E9C46A",
    "bradesco":    "#CC0000",
    "santander_br":"#D62828",
    "banrisul":    "#457B9D",
    "caixa":       "#2D6A4F",
}
DIGITAL_C = "#E64A19"
INCUMB_C  = "#1565C0"
PIX_YEAR  = 2020.5      # Pix launched Nov 2020

def _save(fig, stem: str) -> None:
    for d in (DESC, DRAFTS):
        if os.path.isdir(d):
            fig.savefig(os.path.join(d, f"{stem}.png"), dpi=160, bbox_inches="tight")
            fig.savefig(os.path.join(d, f"{stem}.pdf"),          bbox_inches="tight")
    log.info(f"saved {stem}.{{png,pdf}} to DESC + DRAFTS")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_client_series() -> pd.DataFrame:
    """Merge IPE + EDGAR client-count time series into a single long frame."""
    rows = []

    # IPE (Pan, BMG, BB, Banrisul) — metric = clients_active, value = absolute
    ipe = pd.read_csv(os.path.join(BASE, "FirmDisclosures", "Incumbents",
                                   "incumbent_client_counts.csv"))
    for _, r in ipe.iterrows():
        if r["metric"] == "clients_active" and pd.notna(r["value"]):
            rows.append(dict(firm_key=r["firm_key"], segment=r["segment"],
                             year=r["period_year"], quarter=r.get("period_quarter",2),
                             clients=float(r["value"])))

    # EDGAR (Nubank, Inter, PagSeguro, Stone, XP, Bradesco, Santander …)
    # prefer customers_total Brazil-scope; fall back to broad
    edgar = pd.read_csv(os.path.join(BASE, "FirmDisclosures", "SEC",
                                     "edgar_disclosures.csv"))
    edgar = edgar[edgar["metric"].isin(["customers_total","customers_active"])]
    edgar = edgar[edgar["geo_scope"].isin(["brazil","consolidated","latam"])]
    # rank: brazil > consolidated/latam; total > active
    edgar["rank"] = (edgar["geo_scope"].map({"brazil":0,"consolidated":1,"latam":1}).fillna(2)*2
                   + edgar["metric"].map({"customers_total":0,"customers_active":1}).fillna(1))
    edgar = edgar.sort_values("rank")
    for (fk, yr, q), g in edgar.groupby(["firm_key","period_year","period_quarter"]):
        if pd.isna(q): continue
        best = g.iloc[0]
        rows.append(dict(firm_key=fk, segment=best["segment"],
                         year=int(yr), quarter=int(q),
                         clients=float(best["value"])))

    df = pd.DataFrame(rows)
    df["t"] = df["year"] + (df["quarter"].fillna(2)-1)/4
    # keep per (firm, year, quarter) the largest value (plausibility: suppress <500k noise)
    df = df[df["clients"] >= 500_000]
    df = df.sort_values("clients", ascending=False).drop_duplicates(["firm_key","year","quarter"])
    return df.sort_values(["firm_key","t"])


def load_join() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DESC, "account_vs_volume_panel.csv"))


def load_market() -> pd.DataFrame:
    cols = ["CodConglomeradoPrudencial","year","quarter","dep_a1","dep_a2","dep_a4","dep_a5"]
    df = pd.read_csv(os.path.join(BASE,"BCB","Panel","market_panel.csv"), usecols=cols)
    df["total"] = df[["dep_a1","dep_a2","dep_a4","dep_a5"]].sum(axis=1,min_count=1)
    return df


def load_findex() -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE,"WorldBank","Findex","findex_brazil_demographics.csv"))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A — client growth trajectories
# ─────────────────────────────────────────────────────────────────────────────

def figure_a(clients: pd.DataFrame) -> None:
    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    digital_firms = [f for f in D_COLOURS if f in clients.firm_key.values]
    incumb_firms  = [f for f in B_COLOURS  if f in clients.firm_key.values]

    for fk in digital_firms:
        g = clients[clients.firm_key == fk].sort_values("t")
        if len(g) < 2: continue
        ax_d.plot(g["t"], g["clients"]/1e6, marker="o", ms=3, lw=1.8,
                  color=D_COLOURS[fk], label=fk)

    ax_d.axvline(PIX_YEAR, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax_d.text(PIX_YEAR+0.1, ax_d.get_ylim()[1]*0.95, "Pix", fontsize=7, color="grey")
    ax_d.set_ylabel("Clients (millions)")
    ax_d.set_xlabel("Year")
    ax_d.set_title("A. Digital / payment firms (D-type)", fontsize=10, loc="left")
    ax_d.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax_d.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}M"))

    for fk in incumb_firms:
        g = clients[clients.firm_key == fk].sort_values("t")
        if len(g) < 2: continue
        ax_b.plot(g["t"], g["clients"]/1e6, marker="o", ms=3, lw=1.8,
                  color=B_COLOURS[fk], label=fk)

    ax_b.set_ylabel("Clients (millions)")
    ax_b.set_xlabel("Year")
    ax_b.set_title("B. Incumbents (B-type)", fontsize=10, loc="left")
    ax_b.legend(fontsize=7, framealpha=0.9)
    ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}M"))

    fig.suptitle("Client base trajectories: D vs. B firms, Brazil 2013–2026",
                 fontsize=12, y=1.02)
    fig.text(0.5,-0.03,
             "Sources: SEC EDGAR (Nubank, Inter, PagSeguro, Stone, XP) and CVM-IPE (Pan, BMG, BB, Banrisul). "
             "D-firm client counts are 'active' or 'total' depending on disclosure. "
             "Pix launched November 2020.",
             ha="center", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_client_growth")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B — deposits per customer (intensive margin gap)
# ─────────────────────────────────────────────────────────────────────────────

def figure_b(join: pd.DataFrame) -> None:
    df = join.dropna(subset=["deposits_per_customer_brl"]).copy()
    df["t"] = df["year"] + (df["quarter"]-1)/4

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for fk, g in df.sort_values("t").groupby("firm_key"):
        seg = g["segment"].iloc[0]
        color = INCUMB_C if seg == "incumbent" else DIGITAL_C
        alpha = 0.9 if seg == "incumbent" else 0.75
        lw    = 2.0 if seg == "incumbent" else 1.4
        ax.plot(g["t"], g["deposits_per_customer_brl"], marker="o", ms=3,
                lw=lw, color=color, alpha=alpha,
                label=f"{fk} ({'B' if seg=='incumbent' else 'D'})")

    ax.axvline(PIX_YEAR, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.text(PIX_YEAR+0.1, ax.get_ylim()[1]*0.9 if ax.get_yscale()!="log" else 10**
            (np.log10(ax.get_ylim()[1])*0.97), "Pix", fontsize=7, color="grey")
    ax.set_yscale("log")
    ax.set_ylabel("Deposits per customer (R$, log scale)")
    ax.set_xlabel("Year")
    ax.set_title("Intensive margin: deposits per customer — B vs. D firms", fontsize=11, loc="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x,_: f"R${x:,.0f}" if x < 1000 else f"R${x/1000:.0f}k"))
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)

    # segment shading in legend
    from matplotlib.lines import Line2D
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [
        Line2D([0],[0],color=INCUMB_C,lw=2,label="── B-type (incumbent)"),
        Line2D([0],[0],color=DIGITAL_C,lw=1.4,label="── D-type (digital/payment)"),
    ], fontsize=6.5, ncol=2, framealpha=0.9)

    fig.text(0.5,-0.02,
             "Deposits per customer = disclosed BRL deposits ÷ Brazil customer count. "
             "USD deposits converted at R$5 = US$1. Plausibility-clamped to R$50–R$200k. "
             "Source: SEC EDGAR / CVM disclosures joined to BCB deposit panel.",
             ha="center", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_intensive_margin_gap")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE C — market structure: volume-share dynamics
# ─────────────────────────────────────────────────────────────────────────────

def figure_c(market: pd.DataFrame, join: pd.DataFrame) -> None:
    # Identify D-firm conglomerate codes from the join table
    d_congs = set(join[join.segment.isin(["digital","payment"])]["cong_prud"].dropna())
    # Top-4 B-firm codes
    top4_keys = ["itau","bradesco","bb","santander_br"]
    top4_congs = set(join[join.firm_key.isin(top4_keys)]["cong_prud"].dropna())

    mkt = market.groupby(["year","quarter","CodConglomeradoPrudencial"])["total"].sum().reset_index()

    def share(congs, df=mkt):
        subset = df[df["CodConglomeradoPrudencial"].isin(congs)]
        tot    = df.groupby(["year","quarter"])["total"].sum().reset_index().rename(columns={"total":"tot"})
        agg    = subset.groupby(["year","quarter"])["total"].sum().reset_index()
        m = agg.merge(tot, on=["year","quarter"])
        m["share"] = m["total"]/m["tot"]
        m["t"] = m["year"] + (m["quarter"]-1)/4
        return m

    d_share   = share(d_congs)
    top4_share = share(top4_congs)
    # annual for cleaner plot
    d_ann   = d_share.groupby("year")["share"].mean().reset_index()
    t4_ann  = top4_share.groupby("year")["share"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(d_ann["year"],   d_ann["share"]*100,  marker="o", ms=4, lw=2,
            color=DIGITAL_C, label="All digital/payment D-firms (aggregate)")
    ax.plot(t4_ann["year"],  t4_ann["share"]*100, marker="s", ms=4, lw=2,
            color=INCUMB_C,  label="Top-4 incumbents (Itaú/Bradesco/BB/Santander)")
    ax.axvline(PIX_YEAR, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.text(PIX_YEAR+0.15, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1]>1 else 0.95,
            "Pix", fontsize=7, color="grey")
    ax.set_ylabel("Deposit volume share (%)")
    ax.set_xlabel("Year")
    ax.set_title("Market structure: deposit volume share — D-firms vs. top-4 incumbents",
                 fontsize=11, loc="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}%"))
    ax.legend(fontsize=8, framealpha=0.9)
    fig.text(0.5,-0.02,
             "Retail deposit volume share = (dep_a1+dep_a2+dep_a4+dep_a5) of firm set / national total, "
             "annual average. Source: BCB ESTBAN deposit panel 2013–2025.",
             ha="center", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_market_structure")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE D — Findex demographic ownership gradients (Pi motivation)
# ─────────────────────────────────────────────────────────────────────────────

def figure_d(findex: pd.DataFrame) -> None:
    # Bar chart: for each survey year, show ownership rates by demographic group
    # Focus on the gap-revealing groups (not 'all')
    GAP_PAIRS = [
        ("poorest_40","richest_60","Income\n(poor vs rich)"),
        ("female","male","Gender\n(F vs M)"),
        ("young_15_24","older_25p","Age\n(15-24 vs 25+)"),
        ("primary_or_less","secondary_or_more","Education\n(primary vs secondary+)"),
    ]
    years = sorted(findex["year"].unique())
    piv = findex.set_index(["year","group"])["value_pct"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), sharey=True)
    colors_lo = ["#F4D35E","#EE6C4D","#98C1D9","#7EC8A4"]
    colors_hi = ["#083D77","#C1121F","#1D3557","#1B4332"]

    for ax, (lo, hi, label), c_lo, c_hi in zip(axes, GAP_PAIRS, colors_lo, colors_hi):
        lo_vals = [piv.get((y,lo), np.nan) for y in years]
        hi_vals = [piv.get((y,hi), np.nan) for y in years]
        x = np.arange(len(years)); w = 0.38
        bars_lo = ax.bar(x - w/2, lo_vals, width=w, color=c_lo, alpha=0.85,
                         label=lo.replace("_"," "))
        bars_hi = ax.bar(x + w/2, hi_vals, width=w, color=c_hi, alpha=0.85,
                         label=hi.replace("_"," "))
        # gap annotation on most recent year
        if pd.notna(lo_vals[-1]) and pd.notna(hi_vals[-1]):
            gap = hi_vals[-1] - lo_vals[-1]
            ax.annotate(f"Δ{gap:.1f}pp", xy=(x[-1]+w/2, hi_vals[-1]+1),
                        fontsize=7, ha="center", color=c_hi)
        ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years], fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=6, loc="upper left", framealpha=0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))

    axes[0].set_ylabel("Account ownership (% age 15+)")
    fig.suptitle(
        "Findex: account ownership by demographic group — Brazil 2011–2025\n"
        "(motivates heterogeneous loadings $\\Pi$, $\\Pi^q$ in the extensive/intensive-choice model)",
        fontsize=11, y=1.04)
    fig.text(0.5,-0.02,
             "Source: World Bank Global Findex via WB API. "
             "Account ownership = % age 15+ with a financial institution or mobile-money account. "
             "Δ = gap between high and low subgroup in 2025.",
             ha="center", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_findex_demographics")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading data...")
    clients = load_client_series()
    join    = load_join()
    market  = load_market()
    findex  = load_findex()

    log.info(f"Client series: {len(clients)} obs, {clients.firm_key.nunique()} firms, "
             f"years {int(clients.year.min())}-{int(clients.year.max())}")

    log.info("Generating Figure A: client growth trajectories...")
    figure_a(clients)

    log.info("Generating Figure B: intensive margin gap...")
    figure_b(join)

    log.info("Generating Figure C: market structure...")
    figure_c(market, join)

    log.info("Generating Figure D: Findex demographics...")
    figure_d(findex)

    log.info("All figures done.")
    print(f"\nOutputs in:\n  {DESC}\n  {DRAFTS}")
    print("\nFiles generated:")
    for stem in ["fig_client_growth","fig_intensive_margin_gap",
                 "fig_market_structure","fig_findex_demographics"]:
        for ext in ("png","pdf"):
            p = os.path.join(DRAFTS, f"{stem}.{ext}")
            print(f"  {'OK' if os.path.exists(p) else 'MISSING':6s} {stem}.{ext}")


if __name__ == "__main__":
    main()
