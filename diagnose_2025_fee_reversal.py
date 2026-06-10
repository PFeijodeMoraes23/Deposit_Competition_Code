"""
diagnose_2025_fee_reversal.py
==============================
Diagnostic for the 2025 fee ratio uptick in Brazilian incumbents.

Three hypotheses tested:
  H1  Real pricing:       svc_revenue grew in 2025 H2 (level shift)
  H2  Deposit effect:     deposits stagnated / fell while revenue rose
  H3  Reclassification:   PF/PJ sub-account split still migrating post-2023

Outputs -> Drafts/Deposit Competition/:
  fee_reversal_fig1_revenue_monthly.png
  fee_reversal_fig2_decomposition.png
  fee_reversal_fig3_pf_pj_split.png
  fee_reversal_fig4_fee_ratio_quarterly.png
  fee_reversal_summary.csv
"""

from __future__ import annotations
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ─────────────────────────────────────────────────────────────────────
from utils import paths
_REPO     = Path(__file__).resolve().parents[2]
TARIF_DIR = paths.TARIFAS_PROC
OUT_DIR   = _REPO / "Drafts" / "Deposit Competition"

COSIF_CSV = TARIF_DIR / "cosif_service_fees_institution.csv"

# ── institution universe ───────────────────────────────────────────────────────
INSTITUTIONS = {
    "00000000": "BB",
    "00360305": "Caixa",
    "60746948": "Bradesco",
    "60701190": "Itaú",
    "90400888": "Santander",
    "00416968": "Inter",
    "18236120": "Nubank",
    "31872495": "C6",
    "10573521": "Mercado Pago",
}
INCUMBENTS  = ["BB", "Caixa", "Bradesco", "Itaú", "Santander"]
DIGITALS    = ["Inter", "Nubank", "C6", "Mercado Pago"]

# colour palette: incumbent=blue family, digital=orange family
COLORS = {
    "BB":          "#003f88",
    "Caixa":       "#0077b6",
    "Bradesco":    "#0096c7",
    "Itaú":        "#00b4d8",
    "Santander":   "#90e0ef",
    "Inter":       "#e85d04",
    "Nubank":      "#f48c06",
    "C6":          "#faa307",
    "Mercado Pago":"#ffba08",
}

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     9,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":    150,
})

# ── helpers ───────────────────────────────────────────────────────────────────
def _ym(data_base_series: pd.Series) -> pd.Series:
    """YYYYMM int -> pandas Period('M')."""
    s = data_base_series.astype(str).str[:6]
    return pd.PeriodIndex(s, freq="M").to_timestamp()


def _fmt_billions(x, pos):
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    return f"{x:.0f}"


# ── load & filter ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("Loading COSIF institution panel …")
    df = pd.read_csv(COSIF_CSV, dtype={"cnpj": str}, low_memory=False)
    df["cnpj"] = df["cnpj"].str.zfill(8)
    df = df[df["cnpj"].isin(INSTITUTIONS)].copy()
    df["bank"] = df["cnpj"].map(INSTITUTIONS)
    df["date"] = _ym(df["data_base"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["half"]  = df["month"].apply(lambda m: "H1" if m <= 6 else "H2")
    df["ym_int"] = df["data_base"].astype(int)
    print(f"  -> {len(df):,} rows | {df['bank'].nunique()} institutions")
    print(f"  -> {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    return df


# ── Fig 1: monthly svc_revenue time series ─────────────────────────────────────
def fig1_revenue_monthly(df: pd.DataFrame) -> None:
    """
    Monthly svc_revenue (absolute R$) for incumbents + Inter, 2019-2026.
    The Dec/Jun spikes are expected (cumulative 6-month flows). Visual
    inspection checks whether H2 2025 is structurally higher than H2 2024.
    """
    sub = df[(df["year"] >= 2019) & df["bank"].isin(INCUMBENTS + ["Inter"])].copy()

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    axes = axes.flatten()
    banks = INCUMBENTS + ["Inter"]

    for ax, bank in zip(axes, banks):
        bdf = sub[sub["bank"] == bank].sort_values("date")
        ax.plot(bdf["date"], bdf["svc_revenue"], color=COLORS[bank], linewidth=1.4)
        # shade H2 periods lightly
        for yr in range(2019, 2027):
            ax.axvspan(pd.Timestamp(f"{yr}-07-01"),
                       pd.Timestamp(f"{yr}-12-31"),
                       alpha=0.04, color="grey", linewidth=0)
        # Mark PIX launch and plan change
        ax.axvline(pd.Timestamp("2020-11-01"), color="#888", lw=0.8, ls="--")
        ax.axvline(pd.Timestamp("2023-01-01"), color="#888", lw=0.8, ls=":")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_billions))
        ax.set_title(bank, fontsize=9, fontweight="bold", color=COLORS[bank])
        ax.set_ylabel("svc_revenue (R$)")

    axes[-1].set_visible(False)  # 6 panels for 6 institutions: 3×2 grid is fine, last is spare

    # Shared legend
    fig.text(0.5, 0.01,
             "Bars: cumulative 6-month COSIF flows (peak at Dec/Jun by construction).\n"
             "Dashed = PIX launch Nov 2020.  Dotted = new COSIF plan Jan 2023.",
             ha="center", fontsize=7.5, color="#555")

    fig.suptitle("Fig 1 — Monthly COSIF Service Revenue (absolute R$), 2019–2026",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = OUT_DIR / "fee_reversal_fig1_revenue_monthly.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 1 saved -> {out.name}")


# ── Fig 2: revenue vs deposit decomposition ────────────────────────────────────
def fig2_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    For Dec 2022–Dec 2025, decompose the year-over-year change in fee_ratio_total_deposits
    into revenue effect and deposit effect.

    Δratio = revenue_effect + deposit_effect  (+ small cross-term)
    revenue_effect  = Δ(svc/6) / dep_prior
    deposit_effect  = (svc_curr/6) × (1/dep_curr - 1/dep_prior)
    """
    dec = df[df["month"] == 12].copy()
    dec = dec.sort_values(["bank", "year"])

    # Use dep_total because it works for all institution types
    dec["svc_m"] = dec["svc_revenue"] / 6
    dec = dec[["bank", "year", "svc_m", "dep_total",
               "fee_ratio_total_deposits",
               "svc_revenue_pf", "svc_revenue_pj"]].copy()

    rows = []
    for bank in INCUMBENTS + ["Inter"]:
        bdf = dec[dec["bank"] == bank].set_index("year")
        for yr in range(2021, 2026):
            if yr not in bdf.index or yr - 1 not in bdf.index:
                continue
            curr  = bdf.loc[yr]
            prior = bdf.loc[yr - 1]

            ratio_curr  = curr["fee_ratio_total_deposits"]
            ratio_prior = prior["fee_ratio_total_deposits"]
            delta_ratio = ratio_curr - ratio_prior

            rev_eff  = (curr["svc_m"] - prior["svc_m"]) / prior["dep_total"]
            dep_eff  = curr["svc_m"] * (1 / curr["dep_total"] - 1 / prior["dep_total"])

            rows.append({
                "bank": bank,
                "year": yr,
                "fee_ratio":      ratio_curr,
                "fee_ratio_prior": ratio_prior,
                "delta_ratio":    delta_ratio,
                "revenue_effect": rev_eff,
                "deposit_effect": dep_eff,
                "cross_term":     delta_ratio - rev_eff - dep_eff,
                "svc_revenue_curr":  curr["svc_m"] * 6,
                "dep_total_curr":    curr["dep_total"],
                "svc_revenue_prior": prior["svc_m"] * 6,
                "dep_total_prior":   prior["dep_total"],
                "rev_growth_pct":    (curr["svc_m"] / prior["svc_m"] - 1) * 100
                                     if prior["svc_m"] > 0 else np.nan,
                "dep_growth_pct":    (curr["dep_total"] / prior["dep_total"] - 1) * 100
                                     if prior["dep_total"] > 0 else np.nan,
            })

    summary = pd.DataFrame(rows)

    # ── figure ─────────────────────────────────────────────────────────────────
    years_to_show = [2022, 2023, 2024, 2025]
    banks_show    = INCUMBENTS + ["Inter"]
    n_banks = len(banks_show)

    fig, axes = plt.subplots(1, n_banks, figsize=(13, 5), sharey=False)
    bar_width = 0.35

    for ax, bank in zip(axes, banks_show):
        bsub = summary[summary["bank"] == bank].set_index("year")
        xs = np.arange(len(years_to_show))
        rev_vals = [bsub.loc[y, "revenue_effect"] if y in bsub.index else 0 for y in years_to_show]
        dep_vals = [bsub.loc[y, "deposit_effect"] if y in bsub.index else 0 for y in years_to_show]

        bars_r = ax.bar(xs - bar_width/2, rev_vals, bar_width,
                        label="Revenue effect", color=COLORS[bank], alpha=0.85)
        bars_d = ax.bar(xs + bar_width/2, dep_vals, bar_width,
                        label="Deposit effect", color=COLORS[bank], alpha=0.38)

        # Mark 2025 bars
        ax.axvline(x=3 - bar_width/2, color="red", lw=0.6, alpha=0.5)

        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(y) for y in years_to_show], fontsize=8)
        ax.set_title(bank, fontsize=9, fontweight="bold", color=COLORS[bank])
        if ax == axes[0]:
            ax.set_ylabel("Δ fee_ratio_total_deposits")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.tick_params(axis="y", labelsize=7)

    # Legend on last panel
    rev_patch = mpatches.Patch(color="#555", alpha=0.85, label="Revenue effect")
    dep_patch = mpatches.Patch(color="#555", alpha=0.38, label="Deposit effect")
    axes[-1].legend(handles=[rev_patch, dep_patch], fontsize=7.5, loc="upper left")

    fig.suptitle(
        "Fig 2 — Year-on-Year Change in fee_ratio_total_deposits Decomposed\n"
        "(Revenue effect = Δrevenue / prior deposits; Deposit effect = current_revenue × Δ(1/deposits))",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    out = OUT_DIR / "fee_reversal_fig2_decomposition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 2 saved -> {out.name}")
    return summary


# ── Fig 3: PF vs PJ split evolution ───────────────────────────────────────────
def fig3_pf_pj_split(df: pd.DataFrame) -> None:
    """
    Monthly PF and PJ revenue shares (2023-2026) for incumbents.
    A stable PF/PJ ratio over 2023-2025 rules out ongoing reclassification migration.
    A sudden shift in 2025 would indicate H3 (reclassification artifact).
    """
    sub = df[(df["year"] >= 2023) & df["bank"].isin(INCUMBENTS)].copy()
    sub = sub[(sub["svc_revenue_pf"].notna()) & (sub["svc_revenue_pj"].notna())]
    sub = sub[(sub["svc_revenue_pf"] > 0) | (sub["svc_revenue_pj"] > 0)]
    sub["pf_share"] = sub["svc_revenue_pf"] / (sub["svc_revenue_pf"] + sub["svc_revenue_pj"])

    fig, axes = plt.subplots(1, 5, figsize=(14, 4), sharey=True)
    for ax, bank in zip(axes, INCUMBENTS):
        bdf = sub[sub["bank"] == bank].sort_values("date")
        if bdf.empty:
            ax.set_title(f"{bank}\n(no data)", fontsize=8)
            continue
        ax.plot(bdf["date"], bdf["pf_share"] * 100,
                color=COLORS[bank], linewidth=1.5, label="PF share")
        ax.fill_between(bdf["date"], bdf["pf_share"] * 100,
                        alpha=0.15, color=COLORS[bank])
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of revenue from PF" if ax == axes[0] else "")
        ax.set_title(bank, fontsize=9, fontweight="bold", color=COLORS[bank])
        ax.axhline(50, color="grey", lw=0.6, ls="--")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        # Shade H2 of each year
        for yr in range(2023, 2027):
            ax.axvspan(pd.Timestamp(f"{yr}-07-01"), pd.Timestamp(f"{yr}-12-31"),
                       alpha=0.05, color="grey", linewidth=0)

    fig.suptitle(
        "Fig 3 — PF Share of COSIF Service Revenue (2023–2026, monthly)\n"
        "Stable series -> no ongoing reclassification migration. Sudden 2025 shift -> artifact.",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    out = OUT_DIR / "fee_reversal_fig3_pf_pj_split.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 3 saved -> {out.name}")


# ── Fig 4: quarterly fee ratio 2018-2026 ──────────────────────────────────────
def fig4_fee_ratio_quarterly(df: pd.DataFrame) -> None:
    """
    End-of-quarter (Mar/Jun/Sep/Dec) fee_ratio_total_deposits for incumbents + Inter.
    One panel with all 6 lines, annotated with PIX launch and COSIF plan change.
    """
    eom = df[df["month"].isin([3, 6, 9, 12])].copy()
    eom = eom[eom["year"] >= 2018].sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    banks = INCUMBENTS + ["Inter"]
    for bank in banks:
        bdf = eom[eom["bank"] == bank]
        ls = "-" if bank in INCUMBENTS else "--"
        ax.plot(bdf["date"], bdf["fee_ratio_total_deposits"],
                color=COLORS[bank], linewidth=1.8, linestyle=ls, label=bank)

    # Key events
    ax.axvline(pd.Timestamp("2020-11-01"), color="black", lw=1, ls="--", alpha=0.6)
    ax.text(pd.Timestamp("2020-11-01"), ax.get_ylim()[1] * 0.95,
            " PIX\n Nov'20", fontsize=7, color="black", va="top")

    ax.axvline(pd.Timestamp("2023-01-01"), color="#555", lw=1, ls=":", alpha=0.7)
    ax.text(pd.Timestamp("2023-01-01"), ax.get_ylim()[1] * 0.88,
            " New COSIF\n plan Jan'23", fontsize=7, color="#555", va="top")

    ax.set_ylabel("fee_ratio_total_deposits (svc_rev / 6 / dep_total)")
    ax.set_title("Fig 4 — Quarterly Fee Ratio Trajectory, 2018–2026\n"
                 "(end-of-quarter observations: Mar / Jun / Sep / Dec)",
                 fontsize=10, fontweight="bold")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    out = OUT_DIR / "fee_reversal_fig4_fee_ratio_quarterly.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 4 saved -> {out.name}")


# ── Fig 5: H1 vs H2 revenue symmetry check ─────────────────────────────────────
def fig5_h1_h2_ratio(df: pd.DataFrame) -> None:
    """
    For each year, compare H1 (Jun) vs H2 (Dec) svc_revenue level.
    Under normal accounting, H2/H1 ≈ 1.0 (comparable business periods).
    A 2025 spike only in H2 (Dec) would suggest year-end adjustment, not structural.
    A spike in both H1 and H2 2025 confirms a structural level shift.
    """
    peaks = df[df["month"].isin([6, 12])].copy()
    peaks["half"] = peaks["month"].map({6: "H1 (Jun)", 12: "H2 (Dec)"})
    peaks = peaks[peaks["year"] >= 2018]

    fig, axes = plt.subplots(1, len(INCUMBENTS + ["Inter"]),
                              figsize=(14, 4), sharey=False)

    for ax, bank in zip(axes, INCUMBENTS + ["Inter"]):
        bdf = peaks[peaks["bank"] == bank]
        for half, ls, mk in [("H1 (Jun)", "--", "o"), ("H2 (Dec)", "-", "s")]:
            sub = bdf[bdf["half"] == half].sort_values("year")
            ax.plot(sub["year"], sub["fee_ratio_total_deposits"],
                    ls=ls, marker=mk, markersize=4,
                    color=COLORS[bank], alpha=0.7 if half == "H1 (Jun)" else 1.0,
                    label=half)
        ax.axvline(2020.9, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(2023.0, color="#555", lw=0.8, ls=":",  alpha=0.6)
        ax.set_title(bank, fontsize=9, fontweight="bold", color=COLORS[bank])
        ax.set_xticks(range(2018, 2026))
        ax.tick_params(axis="x", rotation=60, labelsize=6.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.tick_params(axis="y", labelsize=7)
        if ax == axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle(
        "Fig 5 — H1 (Jun) vs H2 (Dec) Fee Ratio by Year\n"
        "Both halves rising in 2025 -> structural shift. Only H2 rising -> year-end accounting.",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    out = OUT_DIR / "fee_reversal_fig5_h1_h2_check.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 5 saved -> {out.name}")


# ── Table: annual December snapshot for all key institutions ──────────────────
def build_annual_table(df: pd.DataFrame) -> pd.DataFrame:
    dec = df[df["month"] == 12].copy()
    dec = dec[dec["year"].between(2018, 2025)]
    tbl = dec.pivot_table(
        index="year", columns="bank",
        values="fee_ratio_total_deposits", aggfunc="first"
    )
    # Reorder columns
    ordered = [b for b in (INCUMBENTS + DIGITALS) if b in tbl.columns]
    tbl = tbl[ordered]
    return tbl


# ── Revenue level check table ─────────────────────────────────────────────────
def build_revenue_level_table(df: pd.DataFrame) -> pd.DataFrame:
    dec = df[df["month"] == 12].copy()
    dec = dec[dec["year"].between(2022, 2025)]
    tbl = dec.pivot_table(
        index="year", columns="bank",
        values="svc_revenue", aggfunc="first"
    )
    ordered = [b for b in (INCUMBENTS + ["Inter"]) if b in tbl.columns]
    return tbl[ordered]


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    df = load_data()

    # ── Step 1: run the figures
    fig1_revenue_monthly(df)
    summary = fig2_decomposition(df)
    fig3_pf_pj_split(df)
    fig4_fee_ratio_quarterly(df)
    fig5_h1_h2_ratio(df)

    # ── Step 2: annual tables
    ratio_tbl   = build_annual_table(df)
    revenue_tbl = build_revenue_level_table(df)

    # ── Step 3: save summary CSV
    csv_out = OUT_DIR / "fee_reversal_summary.csv"
    summary.to_csv(csv_out, index=False, float_format="%.6f")
    print(f"Summary CSV -> {csv_out.name}")

    # ── Step 4: print key numbers for notes
    print("\n=== Annual fee_ratio_total_deposits (December snapshot) ===")
    print(ratio_tbl.round(5).to_string())

    print("\n=== H2 svc_revenue in R$ (December cumulative 6-month flow) ===")
    print(revenue_tbl.apply(lambda c: c.map(lambda v: f"R${v/1e9:.2f}B" if pd.notna(v) else "—"))
          .to_string())

    print("\n=== 2024->2025 decomposition (December observations) ===")
    y25 = summary[(summary["year"] == 2025) & summary["bank"].isin(INCUMBENTS + ["Inter"])]
    print(y25[["bank","fee_ratio_prior","fee_ratio","delta_ratio",
               "revenue_effect","deposit_effect",
               "rev_growth_pct","dep_growth_pct"]].round(5).to_string(index=False))

    # ── Step 5: H3 check — PF/PJ share stability
    dec_pf = df[(df["month"] == 12) & (df["year"].between(2023, 2025))].copy()
    dec_pf = dec_pf[dec_pf["bank"].isin(INCUMBENTS)]
    dec_pf["pf_share"] = dec_pf["svc_revenue_pf"] / (dec_pf["svc_revenue_pf"] + dec_pf["svc_revenue_pj"])
    pf_tbl = dec_pf.pivot_table(index="year", columns="bank", values="pf_share", aggfunc="first")
    print("\n=== PF share (svc_revenue_pf / (PF+PJ), December) — reclassification check ===")
    print(pf_tbl.round(3).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
