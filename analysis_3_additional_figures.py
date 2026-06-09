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

import os, sys, re, logging
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
    """Merge IPE + EDGAR client-count time series into a single long frame.

    Data-quality rules (targeted, not blunt QoQ filters):

    EDGAR:
    - Use customers_total only; do not fall back to customers_active. Quarters
      where only customers_active was disclosed are left blank rather than
      showing a mismatched metric (e.g. PagSeguro 2024-Q1/Q2: active≈17M vs
      total≈33M).
    - Global ceiling of 150M: values above this are deposits/other figures
      misread by the text extractor (e.g. PagSeguro 2026-Q2 = 6M is below
      ceiling but still removed via IPE-style ceiling on outliers below).

    IPE (BB, Pan, BMG, Banrisul):
    - Source preference: for any (firm, year, quarter) with both tier4 and
      tier2 values, prefer tier4 (Claude-vision extraction is more reliable
      for structured figures than tier2 text-regex).
    - Global ceiling of 150M for the same reason.
    - BB-specific floor: BB has had >50M clients since at least 2016; values
      below 20M are segment mentions extracted from PDF prose, not total client
      counts. Applied only to BB.
    """
    _BB_FLOOR    = 20_000_000  # BB: anything <20M is a segment mention
    _GLOBAL_CEIL = 150_000_000  # deposits or other figures mis-tagged as clients

    rows = []

    # ── IPE (BB, Pan, BMG, Banrisul) ─────────────────────────────────────────
    ipe = pd.read_csv(os.path.join(BASE, "FirmDisclosures", "Incumbents",
                                   "incumbent_client_counts.csv"))
    for _, r in ipe.iterrows():
        if r["metric"] != "clients_active" or pd.isna(r["value"]):
            continue
        val = float(r["value"])
        fk  = r["firm_key"]
        if val > _GLOBAL_CEIL:
            log.debug(f"IPE ceiling: {fk} {r['period_year']}-Q{r.get('period_quarter','')} "
                      f"value={val/1e6:.0f}M > {_GLOBAL_CEIL/1e6:.0f}M, skipped")
            continue
        if fk == "bb" and val < _BB_FLOOR:
            log.debug(f"IPE BB floor: {r['period_year']}-Q{r.get('period_quarter','')} "
                      f"value={val/1e6:.1f}M < {_BB_FLOOR/1e6:.0f}M, skipped")
            continue
        src = str(r.get("source", "ipe"))
        src_rank = 0 if "tier4" in src else 1  # prefer tier4 (Claude vision)
        q_raw = r.get("period_quarter", 2)
        q_int = int(float(q_raw)) if pd.notna(q_raw) else 2
        rows.append(dict(firm_key=fk, segment=r["segment"],
                         year=int(r["period_year"]), quarter=q_int,
                         clients=val, src_rank=src_rank))

    # ── EDGAR (Nubank, Inter, PagSeguro, Stone, XP, incumbents) ──────────────
    edgar = pd.read_csv(os.path.join(BASE, "FirmDisclosures", "SEC",
                                     "edgar_disclosures.csv"))
    # customers_total ONLY — do not fall back to customers_active
    edgar = edgar[edgar["metric"] == "customers_total"]
    edgar = edgar[edgar["geo_scope"].isin(["brazil", "consolidated", "latam"])]
    edgar["value"] = pd.to_numeric(edgar["value"], errors="coerce")
    edgar = edgar[edgar["value"] <= _GLOBAL_CEIL]

    # Context-based exclusion: reject rows where raw_context reveals the number
    # is NOT the firm's current total customer count.
    # Context-based exclusion filters (applied before dedup/QoQ).
    # These reject rows where raw_context reveals the number is NOT the firm's
    # current total customer count — no firm-specific number thresholds needed.
    _CTX_EXCLUDE = [
        # Forward-looking strategic targets, e.g. Inter's "60-30-30 plan aiming
        # to reach 60 million clients by 2027" (target, not current count)
        r"\bto\s+(?:reach|achieve|attain)\b",
        r"\b(?:north\s+star|pathway\s+toward|quest\s+to|aiming\s+to)\b",
        r"\baims?\s+to\b",
        # "strategic objective/framework" — also matches "trategic framework" when
        # the leading 's' is truncated at the left edge of the raw_context window
        r"trategic\s+(?:objective|framework|goal|target)s?",
        r"\bby\s+202[6-9]\b",           # future-year anchor ("by 2027")
        # Microfinance/microcredit sub-division — e.g. Santander's microcredit
        # portfolio serves ~1.1M customers; this is not total bank customers
        r"\bmicro(?:credit|finance|cr\xe9dit|financ)\b",
        # Microfinance programs described by number of municipalities/communities
        # served — bank-wide customer counts do not carry this qualifier
        r"\b\d+\s+(?:municipalities|communities)\b",
        # Credit-portfolio outstanding balance (not total customer count)
        r"\boutstanding\s+balance\b",
        # App/platform migration counts and brokerage sub-segments (Itaú: "migration
        # of 15 million customers to the Super App", "retail brokerage services to
        # over 531,000 clients" — neither is the bank's total customer count)
        r"\bmigrat(?:ion\s+of|e)\s+\d",
        r"\bbrokerage\b",
    ]
    ctx = edgar["raw_context"].fillna("").str.lower()
    ok = pd.Series(True, index=edgar.index)
    for pat in _CTX_EXCLUDE:
        ok &= ~ctx.str.contains(pat, regex=True, na=False)

    # Competitor-name-as-subject filter: reject rows where a DIFFERENT firm's
    # name appears as the grammatical subject ("Banco Inter has X million users"
    # in Stone's 6-K is Inter's count, not Stone's).  Applied row-wise so that
    # a firm's OWN filings saying "Nubank has 110M customers" are NOT excluded.
    _COMP_NAMES = {
        "banco inter": "inter",
        "nubank": "nubank",
        "pagbank": "pagseguro",
        "pagseguro digital": "pagseguro",
        "banco bradesco": "bradesco",
        "itau unibanco": "itau",
        "banco do brasil": "bb",
        "caixa econom": "caixa",
    }
    for idx, row in edgar[ok].iterrows():
        fk  = str(row.get("firm_key", ""))
        ctx_low = str(row.get("raw_context", "")).lower()
        for name, owner_fk in _COMP_NAMES.items():
            if owner_fk != fk and re.search(r"\b" + re.escape(name) + r"\s+has\b", ctx_low):
                ok[idx] = False
                break

    excluded = (~ok).sum()
    if excluded:
        for _, bad in edgar[~ok].iterrows():
            log.warning(f"  EDGAR ctx-excl: {bad.get('firm_key','')} "
                        f"{bad.get('period_year','')}Q{bad.get('period_quarter','')} "
                        f"value={float(bad['value'])/1e6:.1f}M  "
                        f"ctx: {str(bad.get('raw_context',''))[:80]}")
    edgar = edgar[ok]
    # rank: brazil-scope best; within same scope, larger value preferred
    edgar["scope_rank"] = edgar["geo_scope"].map({"brazil": 0, "consolidated": 1, "latam": 2}).fillna(3)
    edgar = edgar.sort_values(["scope_rank", "value"], ascending=[True, False])
    for (fk, yr, q), g in edgar.groupby(["firm_key", "period_year", "period_quarter"]):
        if pd.isna(q): continue
        best = g.iloc[0]
        rows.append(dict(firm_key=fk, segment=best["segment"],
                         year=int(yr), quarter=int(float(q)),
                         clients=float(best["value"]), src_rank=2))

    df = pd.DataFrame(rows)
    df["t"] = df["year"] + (df["quarter"].fillna(2) - 1) / 4
    df = df[df["clients"] >= 500_000]
    # for each (firm, quarter): prefer tier4 IPE, then tier2 IPE, then EDGAR;
    # within the same source tier, take the largest value
    df = (df.sort_values(["src_rank", "clients"], ascending=[True, False])
            .drop_duplicates(["firm_key", "year", "quarter"]))

    # QoQ plausibility: drop if >50% decline from the prior observation.
    # Applied AFTER the 150M ceiling, so BB's 242M outlier no longer poisons
    # the filter for subsequent legitimate 80M+ values.
    keep = []
    for fk, grp in df.groupby("firm_key"):
        grp = grp.sort_values("t").copy()
        prev = grp["clients"].shift(1)
        bad  = prev.notna() & (grp["clients"] < prev * 0.50)
        if bad.sum():
            log.warning(f"  QoQ drop: {fk}: dropping {bad.sum()} obs "
                        f"({', '.join(str(int(r.year))+'Q'+str(int(r.quarter)) for _, r in grp[bad].iterrows())})")
        keep.append(grp[~bad])
    df = pd.concat(keep, ignore_index=True) if keep else df

    log.info(f"Client series loaded: {len(df)} obs, {df.firm_key.nunique()} firms, "
             f"years {int(df.year.min())}-{int(df.year.max())}")
    return df.sort_values(["firm_key", "t"])


def load_join() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DESC, "account_vs_volume_panel.csv"))


def load_market() -> pd.DataFrame:
    cols = ["CodConglomeradoPrudencial","year","quarter","dep_a1","dep_a2","dep_a4","dep_a5"]
    _new = os.path.join(BASE,"BCB","Egan_et_al_2025_Rep","processed","market_panel.csv")
    _leg = os.path.join(BASE,"BCB","Panel","market_panel.csv")
    df = pd.read_csv(_new if os.path.isfile(_new) else _leg, usecols=cols)
    df["total"] = df[["dep_a1","dep_a2","dep_a4","dep_a5"]].sum(axis=1,min_count=1)
    return df


def load_findex() -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE,"WorldBank","Findex","findex_brazil_demographics.csv"))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A — client growth trajectories
# ─────────────────────────────────────────────────────────────────────────────

def figure_a(clients: pd.DataFrame) -> None:
    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    digital_firms = [f for f in D_COLOURS if f in clients.firm_key.values]
    incumb_firms  = [f for f in B_COLOURS  if f in clients.firm_key.values]

    for fk in digital_firms:
        g = clients[clients.firm_key == fk].sort_values("t")
        if len(g) < 2: continue
        ax_d.plot(g["t"], g["clients"]/1e6, marker="o", ms=3, lw=1.8,
                  color=D_COLOURS[fk], label=fk)

    ax_d.axvline(PIX_YEAR, color="grey", lw=0.8, ls="--", alpha=0.6)
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

    ax_b.axvline(PIX_YEAR, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax_b.set_xlabel("Year")
    ax_b.set_title("B. Incumbents (B-type)", fontsize=10, loc="left")
    ax_b.legend(fontsize=7, framealpha=0.9)
    ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}M"))

    # shared y-limit: max across both panels, with headroom
    all_vals = clients["clients"].dropna() / 1e6
    ymax = all_vals.max() * 1.12
    for ax in (ax_d, ax_b):
        ax.set_ylim(0, ymax)
        ax.text(PIX_YEAR + 0.1, ymax * 0.97, "Pix", fontsize=7, color="grey", va="top")

    fig.suptitle("Client base trajectories: D vs. B firms, Brazil 2015–2026",
                 fontsize=12, y=1.02)
    fig.text(
        0.5, -0.03,
        "Sources: SEC EDGAR (Nubank, Inter, PagSeguro, Stone, XP) and CVM-IPE (Pan, BMG, BB, Banrisul). "
        "Pix launched Nov 2020. Both panels share the same y-axis. "
        "Observations with >50% QoQ client-count drops excluded as scraping artefacts. "
        "Incumbent EDGAR counts excluded (too sparse / segment-level).",
        ha="center", fontsize=7,
    )
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
    """Deposit volume share over time.

    Shows:
      - Aggregate D-firm share (all digital/payment conglomerates)
      - Individual major incumbents: Itaú, Bradesco, BB, Santander, Caixa
    Y-axis spans 0-100% so shares are contextually readable.
    """
    from utils.firm_registry import load_registry
    reg = load_registry(BASE, enrich=True)
    cong_for = {f["firm_key"]: f.get("cong_prud", "") for f in reg}

    mkt = market.groupby(["year","quarter","CodConglomeradoPrudencial"])["total"].sum().reset_index()
    tot_q = mkt.groupby(["year","quarter"])["total"].sum().reset_index(name="nat_total")

    def firm_share(firm_key: str) -> pd.DataFrame:
        cong = cong_for.get(firm_key, "")
        if not cong:
            return pd.DataFrame()
        sub = mkt[mkt.CodConglomeradoPrudencial == cong]
        agg = sub.groupby(["year","quarter"])["total"].sum().reset_index()
        m = agg.merge(tot_q, on=["year","quarter"])
        m["share"] = m["total"] / m["nat_total"]
        return m

    def agg_share(firm_keys) -> pd.DataFrame:
        congs = {cong_for.get(fk,"") for fk in firm_keys} - {""}
        if not congs:
            return pd.DataFrame()
        sub = mkt[mkt.CodConglomeradoPrudencial.isin(congs)]
        agg = sub.groupby(["year","quarter"])["total"].sum().reset_index()
        m = agg.merge(tot_q, on=["year","quarter"])
        m["share"] = m["total"] / m["nat_total"]
        return m

    d_keys = [f["firm_key"] for f in reg if f.get("segment") in ("digital","payment")]
    d_ann = agg_share(d_keys).groupby("year")["share"].mean().reset_index()

    INCUMB_LINES = [
        ("itau",        "Itaú",      "#E9C46A"),
        ("bradesco",    "Bradesco",  "#CC0000"),
        ("bb",          "BB",        "#1D3557"),
        ("caixa",       "Caixa",     "#2D6A4F"),
        ("santander_br","Santander", "#D62828"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(d_ann["year"], d_ann["share"]*100, marker="o", ms=4, lw=2,
            color=DIGITAL_C, label="All D-firms (aggregate)", zorder=5)

    for fk, label, col in INCUMB_LINES:
        fs = firm_share(fk)
        if fs.empty: continue
        ann = fs.groupby("year")["share"].mean().reset_index()
        ax.plot(ann["year"], ann["share"]*100, marker="s", ms=3, lw=1.6,
                color=col, label=label, alpha=0.85)

    ax.axvline(PIX_YEAR, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.text(PIX_YEAR+0.15, 98, "Pix", fontsize=7, color="grey", va="top")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Deposit volume share (% of national total)")
    ax.set_xlabel("Year")
    ax.set_title("Market structure: individual deposit volume shares 2013–2025",
                 fontsize=11, loc="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    fig.text(0.5, -0.02,
             "Retail deposit share = (dep_a1+dep_a2+dep_a4+dep_a5) / national total, annual average. "
             "Source: BCB ESTBAN deposit panel 2013–2025. "
             "Remaining ~30% held by mid-size incumbents not shown.",
             ha="center", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_market_structure")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE D — Findex demographic ownership gradients (Pi motivation)
# ─────────────────────────────────────────────────────────────────────────────

def _findex_panel(ax, lo, hi, label, c_lo, c_hi, piv, years, show_ylabel=False):
    """Draw one Findex demographic panel onto ax. Returns the gap for annotation."""
    lo_vals = [piv.get((y, lo), np.nan) for y in years]
    hi_vals = [piv.get((y, hi), np.nan) for y in years]
    x = np.arange(len(years)); w = 0.38
    ax.bar(x - w/2, lo_vals, width=w, color=c_lo, alpha=0.85, label=lo.replace("_", " "))
    ax.bar(x + w/2, hi_vals, width=w, color=c_hi, alpha=0.85, label=hi.replace("_", " "))
    # gap bracket on most recent non-nan pair
    for idx in range(len(years)-1, -1, -1):
        if pd.notna(lo_vals[idx]) and pd.notna(hi_vals[idx]):
            gap = hi_vals[idx] - lo_vals[idx]
            top = max(lo_vals[idx], hi_vals[idx])
            ax.annotate("", xy=(x[idx]+w/2, top+2), xytext=(x[idx]-w/2, top+2),
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
            ax.text(x[idx], top+3.5, f"+{gap:.1f}pp", fontsize=7, ha="center", color="black")
            break
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=7)
    ax.set_ylim(0, 105)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    if show_ylabel:
        ax.set_ylabel("Account ownership (% age 15+)")


def figure_d(findex: pd.DataFrame) -> None:
    """Four Findex demographic panels: combined figure + 4 individual files."""
    GAP_PAIRS = [
        ("poorest_40", "richest_60",       "Income (poorest 40% vs richest 60%)",
         "#F4D35E", "#083D77", "fig_findex_income"),
        ("female",     "male",             "Gender (female vs male)",
         "#EE6C4D", "#C1121F", "fig_findex_gender"),
        ("young_15_24","older_25p",        "Age (15–24 vs 25+)",
         "#98C1D9", "#1D3557", "fig_findex_age"),
        ("primary_or_less","secondary_or_more","Education (primary vs secondary+)",
         "#7EC8A4", "#1B4332", "fig_findex_education"),
    ]
    years = sorted(findex["year"].unique())
    piv   = findex.set_index(["year","group"])["value_pct"]
    src_note = (
        "Source: World Bank Global Findex via WB API. "
        "Account ownership = % age 15+ with a financial institution or mobile-money account."
    )

    # — individual figures (one per dimension) ——————————————————————————————
    for lo, hi, label, c_lo, c_hi, stem in GAP_PAIRS:
        fig, ax = plt.subplots(figsize=(5, 4))
        _findex_panel(ax, lo, hi, label, c_lo, c_hi, piv, years, show_ylabel=True)
        fig.suptitle(f"Findex Brazil: {label}", fontsize=10, y=1.01)
        fig.text(0.5, -0.04, src_note, ha="center", fontsize=6.5)
        fig.tight_layout()
        _save(fig, stem)
        plt.close(fig)

    # — combined 1×4 panel (for the appendix) ————————————————————————————————
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)
    for ax, (lo, hi, label, c_lo, c_hi, _) in zip(axes, GAP_PAIRS):
        _findex_panel(ax, lo, hi, label, c_lo, c_hi, piv, years,
                      show_ylabel=(ax is axes[0]))
    fig.suptitle(
        "Findex: account ownership by demographic group — Brazil 2011–2024\n"
        r"(motivates heterogeneous loadings $\Pi$, $\Pi^q$)",
        fontsize=11, y=1.04,
    )
    fig.text(0.5, -0.02, src_note, ha="center", fontsize=7)
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
    for stem in ["fig_client_growth", "fig_intensive_margin_gap",
                 "fig_market_structure",
                 "fig_findex_demographics",
                 "fig_findex_income", "fig_findex_gender",
                 "fig_findex_age",    "fig_findex_education"]:
        for ext in ("png", "pdf"):
            p = os.path.join(DRAFTS, f"{stem}.{ext}")
            print(f"  {'OK' if os.path.exists(p) else 'MISSING':6s} {stem}.{ext}")


if __name__ == "__main__":
    main()
