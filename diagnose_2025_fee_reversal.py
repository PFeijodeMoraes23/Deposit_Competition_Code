"""
diagnose_2025_fee_reversal.py
==============================
Diagnostic and validation figures for COSIF bank service-fee data.

Figures produced (Drafts/Deposit Competition/):
  fee_reversal_fig1_revenue_monthly.png        Monthly cumulative svc_revenue (raw sawtooth)
  fee_reversal_fig2_corrected_quarterly.png    Quarterly fee ratio AFTER increment correction
  fee_reversal_fig3_pf_pj_split.png            PF vs PJ revenue share (2023-2026)
  fee_reversal_fig4_decomposition.png          YoY decomposition: revenue vs deposit effect
  fee_reversal_fig5_h1_h2_check.png            H1 (Jun) vs H2 (Dec) corrected ratio by year
  fee_reversal_fig6_cosif_vs_bcb.png           Cross-validation: COSIF ratio vs BCB listed prices
  fee_reversal_fig7_atm_vs_service.png         ATM vs service-fee decomposition (Open Finance)
  fee_reversal_fig8_pix_did.png                PIX structural-break DiD (incumbents vs digitals)
  fee_reversal_fig9_event_study.png            Formal quarterly event study around PIX launch
  fee_reversal_fig10_correction_factor.png     2025 COSIF reclassification correction factors
  fee_reversal_fig11_cvm_vs_cosif.png          CVM DRE vs COSIF cross-validation + deposit-acct share
  fee_reversal_fig12_fee_ratio_per_bank.png    Annualised fee ratio (%/yr) per bank — all 9 institutions
  fee_reversal_summary.csv                     YoY decomposition data
  fee_reversal_correction_factors.csv          Per-bank 2025 correction factors

Key correction applied throughout:
  COSIF income-statement accounts accumulate within each half-year.  The
  Nakane /6 divisor understates Q1/Q3 by ~50%.  All fee ratios here use
  monthly increments (diff within H1/H2) then sum three increments per
  quarter.  See Section 6d of service_fees_notes.md for details.
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

COSIF_CSV     = TARIF_DIR / "cosif_service_fees_institution.csv"
TARIFF_LONG   = TARIF_DIR / "tarifas_panel_institution_long.csv"
OFB_WIDE_CSV  = TARIF_DIR / "openfinance_fees_panel_wide.csv"
CVM_FEE_CSV   = _REPO / "FirmDisclosures" / "CVM" / "cvm_fee_income.csv"


def _savefig(fig, name: str, **kwargs) -> None:
    kw = dict(dpi=150, bbox_inches="tight") | kwargs
    fig.savefig(OUT_DIR / name, **kw)
    print(f"  saved -> {OUT_DIR / name}")


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
INCUMBENTS = ["BB", "Caixa", "Bradesco", "Itaú", "Santander"]
DIGITALS   = ["Inter", "Nubank", "C6", "Mercado Pago"]

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

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})


# ── helpers ───────────────────────────────────────────────────────────────────
def _ym(s: pd.Series) -> pd.DatetimeIndex:
    return pd.PeriodIndex(s.astype(str).str[:6], freq="M").to_timestamp()


def _fmt_billions(x, _pos):
    if abs(x) >= 1e9:
        return f"{x/1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"{x/1e6:.0f}M"
    return f"{x:.0f}"


# ── data loading & increment correction ───────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Load COSIF institution panel and apply the half-year increment correction.

    Returns monthly rows with:
      svc_revenue_inc  — true monthly revenue increment (not cumulative)
      fee_ratio_td_corr — corrected monthly ratio = svc_revenue_inc / dep_total
    """
    print("Loading COSIF institution panel …")
    df = pd.read_csv(COSIF_CSV, dtype={"cnpj": str}, low_memory=False)
    df["cnpj"] = df["cnpj"].str.zfill(8)
    df = df[df["cnpj"].isin(INSTITUTIONS)].copy()
    df["bank"]   = df["cnpj"].map(INSTITUTIONS)
    df["date"]   = _ym(df["data_base"])
    df["year"]   = df["date"].dt.year
    df["month"]  = df["date"].dt.month
    df["half_yr"] = np.where(df["month"] <= 6, 1, 2)
    df["ym_int"] = df["data_base"].astype(int)

    # Sort for differencing
    df = df.sort_values(["bank", "year", "half_yr", "month"]).reset_index(drop=True)

    def _inc(col: str) -> pd.Series:
        d = df.groupby(["bank", "year", "half_yr"])[col].diff()
        return d.fillna(df[col])

    df["svc_revenue_inc"]    = _inc("svc_revenue")
    df["svc_revenue_pf_inc"] = _inc("svc_revenue_pf")
    df["svc_revenue_pj_inc"] = _inc("svc_revenue_pj")

    # Corrected monthly ratio (increment / deposit stock)
    df["fee_ratio_td_corr"] = df["svc_revenue_inc"] / df["dep_total"]

    print(f"  -> {len(df):,} rows | {df['bank'].nunique()} institutions")
    print(f"  -> {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    return df


def quarterly_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to quarterly fee ratios by summing 3 monthly increments.
    Q_ratio = sum(svc_revenue_inc for 3 months) / dep_total at end of quarter.
    """
    df = df.copy()
    df["quarter"] = (df["month"] - 1) // 3 + 1
    df = df.sort_values(["bank", "year", "quarter", "month"])
    q = (
        df.groupby(["bank", "year", "quarter"], as_index=False)
        .agg(
            svc_revenue_q    =("svc_revenue_inc",    "sum"),
            svc_revenue_pf_q =("svc_revenue_pf_inc", "sum"),
            svc_revenue_pj_q =("svc_revenue_pj_inc", "sum"),
            dep_total        =("dep_total",           "last"),
            date             =("date",                "last"),
        )
    )
    q["fee_ratio_q"]    = q["svc_revenue_q"] / q["dep_total"]
    q["fee_ratio_q_ann"] = q["fee_ratio_q"] * 4 * 100   # annualized pp/yr
    # Quarter end date label
    q["date"] = pd.to_datetime(q["date"])
    return q


# ── Fig 1: raw monthly svc_revenue — shows the expected sawtooth ─────────────
def fig1_revenue_monthly(df: pd.DataFrame) -> None:
    """
    Monthly svc_revenue (absolute R$, raw cumulative) for incumbents + Inter.
    The Jun/Dec peaks are expected (6-month accumulation). Shown here as
    reference to motivate the increment correction.
    """
    sub = df[(df["year"] >= 2019) & df["bank"].isin(INCUMBENTS + ["Inter"])].copy()

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    axes = axes.flatten()
    banks = INCUMBENTS + ["Inter"]

    for ax, bank in zip(axes, banks):
        bdf = sub[sub["bank"] == bank].sort_values("date")
        ax.plot(bdf["date"], bdf["svc_revenue"], color=COLORS[bank], linewidth=1.4)
        for yr in range(2019, 2027):
            ax.axvspan(pd.Timestamp(f"{yr}-07-01"), pd.Timestamp(f"{yr}-12-31"),
                       alpha=0.04, color="grey", linewidth=0)
        ax.axvline(pd.Timestamp("2020-11-01"), color="#888", lw=0.8, ls="--")
        ax.axvline(pd.Timestamp("2023-01-01"), color="#888", lw=0.8, ls=":")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_billions))
        ax.set_title(bank, fontsize=9, fontweight="bold", color=COLORS[bank])
        ax.set_ylabel("svc_revenue (R$, cumulative)")

    axes[-1].set_visible(False)
    fig.text(0.5, 0.01,
             "Cumulative 6-month flows reset at Jan and Jul → Jun/Dec peaks are expected.\n"
             "Dashed = PIX launch Nov 2020.  Dotted = new COSIF plan Jan 2023.",
             ha="center", fontsize=7.5, color="#555")
    fig.suptitle("Fig 1 — Raw Cumulative COSIF Service Revenue (R$), 2019–2026",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    _savefig(fig, "fee_reversal_fig1_revenue_monthly.png")
    plt.close(fig)


# ── Fig 2: corrected quarterly fee ratio ──────────────────────────────────────
def fig2_corrected_quarterly(df: pd.DataFrame) -> None:
    """
    Quarterly fee_ratio_total_deposits after the increment correction.
    Eliminates the saw pattern visible in the raw /6 ratio.
    """
    q = quarterly_agg(df)
    q = q[q["year"] >= 2018]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    banks = INCUMBENTS + ["Inter"]
    for bank in banks:
        bdf = q[q["bank"] == bank].sort_values("date")
        ls = "-" if bank in INCUMBENTS else "--"
        ax.plot(bdf["date"], bdf["fee_ratio_q"] * 100,
                color=COLORS[bank], linewidth=1.8, linestyle=ls, label=bank)

    ax.axvline(pd.Timestamp("2020-11-01"), color="black", lw=1, ls="--", alpha=0.6)
    ax.text(pd.Timestamp("2020-11-01"), ax.get_ylim()[1] * 0.96,
            " PIX\n Nov'20", fontsize=7.5, color="black", va="top")
    ax.axvline(pd.Timestamp("2023-01-01"), color="#555", lw=1, ls=":", alpha=0.7)
    ax.text(pd.Timestamp("2023-01-01"), ax.get_ylim()[1] * 0.88,
            " New COSIF\n Jan'23", fontsize=7.5, color="#555", va="top")
    ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31"),
               alpha=0.06, color="red", label="2025+ COSIF break")

    ax.set_ylabel("Quarterly fee ratio × 100  (svc_revenue_q / dep_total, %)")
    ax.set_title(
        "Fig 2 — Corrected Quarterly Fee Ratio (sum of 3 monthly increments / EOM deposits)\n"
        "Saw pattern eliminated.  Red band = 2025+ COSIF account reclassification.",
        fontsize=9, fontweight="bold",
    )
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    _savefig(fig, "fee_reversal_fig2_corrected_quarterly.png")
    plt.close(fig)


# ── Fig 3: PF vs PJ split evolution ───────────────────────────────────────────
def fig3_pf_pj_split(df: pd.DataFrame) -> None:
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
                color=COLORS[bank], linewidth=1.5)
        ax.fill_between(bdf["date"], bdf["pf_share"] * 100, alpha=0.15, color=COLORS[bank])
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of revenue from PF" if ax == axes[0] else "")
        ax.set_title(bank, fontsize=9, fontweight="bold", color=COLORS[bank])
        ax.axhline(50, color="grey", lw=0.6, ls="--")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        for yr in range(2023, 2027):
            ax.axvspan(pd.Timestamp(f"{yr}-07-01"), pd.Timestamp(f"{yr}-12-31"),
                       alpha=0.05, color="grey", linewidth=0)

    fig.suptitle(
        "Fig 3 — PF Share of COSIF Service Revenue (2023–2026, monthly)\n"
        "Stable → no ongoing reclassification migration.  Sudden 2025 shift → artifact.",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, "fee_reversal_fig3_pf_pj_split.png")
    plt.close(fig)


# ── Fig 4: YoY decomposition (revenue vs deposit effect) ──────────────────────
def fig4_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    YoY change in corrected fee ratio decomposed into revenue and deposit effects.
    Uses December observations (H2 end, corrected ratio).
    """
    # Use Dec (end of H2) corrected ratio
    dec = df[df["month"] == 12].copy()
    dec["svc_revenue_inc_dec"] = dec["svc_revenue_inc"]
    dec = dec.sort_values(["bank", "year"])

    rows = []
    for bank in INCUMBENTS + ["Inter"]:
        bdf = dec[dec["bank"] == bank].set_index("year")
        for yr in range(2021, 2026):
            if yr not in bdf.index or yr - 1 not in bdf.index:
                continue
            curr  = bdf.loc[yr]
            prior = bdf.loc[yr - 1]
            r_c = curr["svc_revenue_inc"] / curr["dep_total"]
            r_p = prior["svc_revenue_inc"] / prior["dep_total"]
            delta = r_c - r_p
            rev_eff = (curr["svc_revenue_inc"] - prior["svc_revenue_inc"]) / prior["dep_total"]
            dep_eff = curr["svc_revenue_inc"] * (1 / curr["dep_total"] - 1 / prior["dep_total"])
            rows.append({
                "bank": bank, "year": yr,
                "fee_ratio": r_c, "fee_ratio_prior": r_p,
                "delta_ratio": delta,
                "revenue_effect": rev_eff, "deposit_effect": dep_eff,
                "cross_term": delta - rev_eff - dep_eff,
                "rev_growth_pct": (curr["svc_revenue_inc"] / prior["svc_revenue_inc"] - 1) * 100
                                  if prior["svc_revenue_inc"] > 0 else np.nan,
                "dep_growth_pct": (curr["dep_total"] / prior["dep_total"] - 1) * 100
                                  if prior["dep_total"] > 0 else np.nan,
            })

    summary = pd.DataFrame(rows)
    years_to_show = [2022, 2023, 2024, 2025]
    banks_show    = INCUMBENTS + ["Inter"]

    fig, axes = plt.subplots(1, len(banks_show), figsize=(13, 5), sharey=False)
    bar_width = 0.35
    for ax, bank in zip(axes, banks_show):
        bsub = summary[summary["bank"] == bank].set_index("year")
        xs = np.arange(len(years_to_show))
        rev_v = [bsub.loc[y, "revenue_effect"] if y in bsub.index else 0 for y in years_to_show]
        dep_v = [bsub.loc[y, "deposit_effect"] if y in bsub.index else 0 for y in years_to_show]
        ax.bar(xs - bar_width/2, rev_v, bar_width, color=COLORS[bank], alpha=0.85,
               label="Revenue effect")
        ax.bar(xs + bar_width/2, dep_v, bar_width, color=COLORS[bank], alpha=0.38,
               label="Deposit effect")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(y) for y in years_to_show], fontsize=8)
        ax.set_title(bank, fontsize=9, fontweight="bold", color=COLORS[bank])
        if ax == axes[0]:
            ax.set_ylabel("Δ quarterly fee ratio (Dec increment)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.tick_params(axis="y", labelsize=7)

    rev_p = mpatches.Patch(color="#555", alpha=0.85, label="Revenue effect")
    dep_p = mpatches.Patch(color="#555", alpha=0.38, label="Deposit effect")
    axes[-1].legend(handles=[rev_p, dep_p], fontsize=7.5, loc="upper left")

    fig.suptitle(
        "Fig 4 — YoY Change in Corrected Fee Ratio Decomposed (December observations)\n"
        "Revenue effect = Δrevenue_inc / prior_dep;  Deposit effect = rev_curr × Δ(1/dep)",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, "fee_reversal_fig4_decomposition.png")
    plt.close(fig)
    return summary


# ── Fig 5: H1 vs H2 corrected ratio by year ───────────────────────────────────
def fig5_h1_h2_ratio(df: pd.DataFrame) -> None:
    peaks = df[df["month"].isin([6, 12])].copy()
    peaks["half_label"] = peaks["month"].map({6: "H1 (Jun)", 12: "H2 (Dec)"})
    peaks["ratio_corr"] = peaks["fee_ratio_td_corr"]
    peaks = peaks[peaks["year"] >= 2018]

    fig, axes = plt.subplots(1, len(INCUMBENTS + ["Inter"]),
                              figsize=(14, 4), sharey=False)
    for ax, bank in zip(axes, INCUMBENTS + ["Inter"]):
        bdf = peaks[peaks["bank"] == bank]
        for half, ls, mk in [("H1 (Jun)", "--", "o"), ("H2 (Dec)", "-", "s")]:
            sub = bdf[bdf["half_label"] == half].sort_values("year")
            ax.plot(sub["year"], sub["ratio_corr"] * 100,
                    ls=ls, marker=mk, markersize=4,
                    color=COLORS[bank], alpha=0.7 if "H1" in half else 1.0,
                    label=half)
        ax.axvline(2020.9, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(2023.0, color="#555", lw=0.8, ls=":",  alpha=0.6)
        ax.set_title(bank, fontsize=9, fontweight="bold", color=COLORS[bank])
        ax.set_xticks(range(2018, 2026))
        ax.tick_params(axis="x", rotation=60, labelsize=6.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="y", labelsize=7)
        if ax == axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle(
        "Fig 5 — H1 (Jun) vs H2 (Dec) Corrected Monthly Ratio × 100\n"
        "After increment fix H1 ≈ H2 within year; any remaining gap is seasonal, not artifact.",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, "fee_reversal_fig5_h1_h2_check.png")
    plt.close(fig)


# ── Fig 6: Cross-validation — COSIF corrected ratio vs BCB listed prices ──────
def fig6_cosif_vs_bcb(df: pd.DataFrame) -> None:
    """
    Scatter: COSIF annualized fee ratio (2022-2024 Dec average, %) vs
    sum of BCB listed priority-service prices (BRL) per institution.
    Validates that both measures capture the same cross-section of 'bank expensiveness'.
    """
    if not TARIFF_LONG.exists():
        print("  [Fig 6 skipped] tariff panel not found at", TARIFF_LONG)
        return

    # ── COSIF side: annual equivalent of corrected Dec ratio ──────────────
    dec_corr = (
        df[(df["month"] == 12) & df["year"].between(2022, 2024) & (df["year"] < 2025)]
        .groupby("bank", as_index=False)
        .agg(cosif_ann=("fee_ratio_td_corr", "mean"))
    )
    dec_corr["cosif_ann_pct"] = dec_corr["cosif_ann"] * 12 * 100   # monthly → pp/yr

    # ── BCB Tarifas side: sum of listed max prices for priority services ──
    svc_map = {1101: "cadastro", 1201: "card2nd", 1203: "ccf_excl",
               1204: "cheque_stop", 1213: "statement", 1501: "atm_pkg"}

    tlong = pd.read_csv(TARIFF_LONG, dtype={"cnpj": str}, low_memory=False,
                        encoding="latin-1")
    tlong["cnpj"] = tlong["cnpj"].str.strip().str.zfill(8)
    tlong_pf = tlong[(tlong["customer_type"] == "F") &
                     (tlong["codigo_servico"].isin(svc_map.keys()))].copy()

    # Map CNPJ → bank label using INSTITUTIONS dict
    tlong_pf["bank"] = tlong_pf["cnpj"].map(INSTITUTIONS)
    tlong_pf = tlong_pf[tlong_pf["bank"].notna()].copy()

    bcb_agg = (
        tlong_pf.groupby(["bank", "codigo_servico"], as_index=False)
        .agg(val=("valor_maximo", "median"))
        .groupby("bank", as_index=False)
        .agg(bcb_listed_sum=("val", "sum"))
    )

    merged = dec_corr.merge(bcb_agg, on="bank", how="inner")
    if merged.empty:
        print("  [Fig 6 skipped] No overlapping institutions between COSIF and BCB Tarifas")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in merged.iterrows():
        bank = row["bank"]
        c = COLORS.get(bank, "#888")
        mk = "o" if bank in INCUMBENTS else "^"
        ax.scatter(row["bcb_listed_sum"], row["cosif_ann_pct"],
                   color=c, marker=mk, s=80, zorder=3)
        ax.annotate(bank, (row["bcb_listed_sum"], row["cosif_ann_pct"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8, color=c)

    if len(merged) >= 3:
        coef = np.polyfit(merged["bcb_listed_sum"], merged["cosif_ann_pct"], 1)
        xfit = np.linspace(merged["bcb_listed_sum"].min(), merged["bcb_listed_sum"].max(), 50)
        ax.plot(xfit, np.polyval(coef, xfit), "--", color="#888", lw=1, alpha=0.7,
                label=f"OLS slope = {coef[0]:.4f}")
        corr = merged["bcb_listed_sum"].corr(merged["cosif_ann_pct"])
        ax.text(0.97, 0.05, f"r = {corr:.2f}", transform=ax.transAxes,
                fontsize=9, ha="right", va="bottom", color="#333")

    inc_p = mpatches.Patch(color="#003f88", label="Incumbent (circle)")
    dig_p = mpatches.Patch(color="#e85d04", label="Digital (triangle)")
    ax.legend(handles=[inc_p, dig_p], fontsize=8)

    ax.set_xlabel("BCB listed price sum (BRL) — priority services (1101-1501), PF")
    ax.set_ylabel("COSIF annualized fee ratio (% of deposits/yr) — 2022-2024 Dec avg")
    ax.set_title(
        "Fig 6 — Cross-validation: COSIF realized ratio vs BCB listed prices\n"
        "Positive correlation validates using either as a price instrument for the other.",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, "fee_reversal_fig6_cosif_vs_bcb.png")
    plt.close(fig)


# ── Fig 7: ATM vs service-fee decomposition / time series from Open Finance ────
def fig7_atm_vs_service(df: pd.DataFrame) -> None:
    """
    Cross-institution comparison of ATM withdrawal vs service/maintenance fees.

    Mode selection:
      - If OFB long panel has < 2 unique snapshot dates spanning > 30 days → bar chart
        (latest snapshot, cross-institution comparison).
      - If ≥ 2 dates spanning > 30 days → time-series panel per fee type by institution
        type (incumbent avg vs digital avg).
    """
    if not OFB_WIDE_CSV.exists():
        print("  [Fig 7 skipped] OFB wide panel not found")
        return

    wide = pd.read_csv(OFB_WIDE_CSV, dtype={"cnpj8": str}, low_memory=False)

    # ── canonical bank label ─────────────────────────────────────────────────
    wide["bank"] = wide["cnpj8"].map(INSTITUTIONS)
    if wide["bank"].isna().all() and "org_name" in wide.columns:
        known_names = {
            "BANCO DO BRASIL S.A.": "BB",
            "CAIXA ECONOMICA FEDERAL": "Caixa",
            "BANCO BRADESCO S.A.": "Bradesco",
            "ITAU UNIBANCO S.A.": "Itaú",
            "BANCO INTER": "Inter",
            "NU PAGAMENTOS S.A. - INSTITUICAO DE PAGAMENTO": "Nubank",
        }
        wide["bank"] = wide["org_name"].map(known_names)

    # ── fee columns ──────────────────────────────────────────────────────────
    atm_col = "fee_SAQUE_TERMINAL"
    svc_candidates = ["fee_CADASTRO", "fee_2_VIA_CARTAO_DEBITO",
                      "fee_EXTRATO_MOVIMENTO_E", "fee_FOLHA_CHEQUE"]
    svc_cols = [c for c in svc_candidates if c in wide.columns]

    # ── filter to PF demand accounts ────────────────────────────────────────
    pf_all = wide[
        wide["bank"].notna() &
        wide.get("customer_type", pd.Series(["PF"] * len(wide), index=wide.index)).eq("PF") &
        wide.get("account_type", pd.Series(["CORRENTE"] * len(wide), index=wide.index))
            .str.contains("CORRENTE|VISTA", case=False, na=True)
    ].copy()

    if pf_all.empty:
        pf_all = wide[wide["bank"].notna()].copy()

    # ── decide mode ──────────────────────────────────────────────────────────
    dates = pd.Series(dtype="object")
    if "data_coleta" in pf_all.columns:
        dates = pd.to_datetime(pf_all["data_coleta"].dropna().unique())

    span_days = (dates.max() - dates.min()).days if len(dates) >= 2 else 0
    time_series_mode = (len(dates) >= 2) and (span_days > 30)

    # ── helper: extract fee value from a single-row sub-df ──────────────────
    def _get_val(subdf, col):
        if col not in subdf.columns or subdf[col].isna().all():
            return 0.0
        v = subdf[col].dropna()
        return float(v.iloc[0]) if not v.empty else 0.0

    if time_series_mode:
        # ── TIME SERIES MODE ─────────────────────────────────────────────────
        # For each snapshot date × institution, pull ATM fee and svc fee sum
        records = []
        for dt in sorted(dates):
            snap = pf_all[pd.to_datetime(pf_all["data_coleta"]) == dt]
            for bank in snap["bank"].dropna().unique():
                brow = snap[snap["bank"] == bank]
                atm_v = _get_val(brow, atm_col)
                svc_v = sum(_get_val(brow, c) for c in svc_cols)
                grp = ("Incumbent" if bank in INCUMBENTS
                       else "Digital" if bank in DIGITALS else None)
                records.append({"date": dt, "bank": bank, "group": grp,
                                 "atm_fee": atm_v, "svc_fee": svc_v})
        ts = pd.DataFrame(records)
        ts = ts[ts["group"].notna()]
        grp_ts = ts.groupby(["date", "group"], as_index=False).agg(
            atm_fee=("atm_fee", "mean"), svc_fee=("svc_fee", "mean")
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, col, label in [
            (axes[0], "atm_fee", "ATM withdrawal fee (SAQUE_TERMINAL, R$)"),
            (axes[1], "svc_fee", "Service/maintenance fee sum (R$)"),
        ]:
            for grp, color, ls in [("Incumbent","#003f88","-"),("Digital","#e85d04","--")]:
                sub = grp_ts[grp_ts["group"] == grp].sort_values("date")
                ax.plot(sub["date"], sub[col], color=color, linestyle=ls,
                        linewidth=1.8, marker="o", markersize=4, label=grp)
            ax.set_ylabel(label, fontsize=8)
            ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=30, labelsize=7)

        n_dates = len(dates)
        fig.suptitle(
            f"Fig 7 — ATM vs Service-Fee Trends by Institution Type ({n_dates} OFB snapshots)\n"
            "Group average listed fees; extend as scrape_13 accumulates more snapshots.",
            fontsize=9, fontweight="bold",
        )
        fig.tight_layout()
        _savefig(fig, "fee_reversal_fig7_atm_vs_service.png")
        plt.close(fig)
        return

    # ── BAR CHART MODE (cross-institution, latest snapshot) ─────────────────
    pf = pf_all.copy()
    if "data_coleta" in pf.columns:
        pf = pf.sort_values("data_coleta").groupby("cnpj8", as_index=False).last()

    banks_with_data = pf["bank"].dropna().unique()
    banks_ordered   = [b for b in (INCUMBENTS + DIGITALS) if b in banks_with_data]
    if not banks_ordered:
        print("  [Fig 7 skipped] No recognised banks in OFB wide panel")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
    x = np.arange(len(banks_ordered))

    ax = axes[0]
    vals_atm = [_get_val(pf[pf["bank"] == b], atm_col) for b in banks_ordered]
    ax.barh(x, vals_atm, color=[COLORS.get(b, "#888") for b in banks_ordered])
    ax.set_yticks(x); ax.set_yticklabels(banks_ordered, fontsize=8)
    ax.set_xlabel("Weighted-average ATM withdrawal fee (R$)")
    ax.set_title("ATM Withdrawal\n(SAQUE_TERMINAL)", fontsize=9)
    ax.axvline(0, color="black", lw=0.6)

    ax = axes[1]
    vals_svc = [sum(_get_val(pf[pf["bank"] == b], c) for c in svc_cols) for b in banks_ordered]
    ax.barh(x, vals_svc, color=[COLORS.get(b, "#888") for b in banks_ordered])
    ax.set_yticks(x); ax.set_yticklabels(banks_ordered, fontsize=8)
    ax.set_xlabel("Sum of weighted-average service fees (R$)")
    ax.set_title(f"Service/Maintenance Fees\n({', '.join(c.replace('fee_','') for c in svc_cols)})",
                 fontsize=9)
    ax.axvline(0, color="black", lw=0.6)

    latest = dates.max().strftime("%b %Y") if len(dates) else "latest snapshot"
    fig.suptitle(
        f"Fig 7 — ATM vs Service-Fee Decomposition: Open Finance ({latest})\n"
        "Digital banks charge near-zero for most services; incumbents charge for ATM + statements.",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, "fee_reversal_fig7_atm_vs_service.png")
    plt.close(fig)


# ── Fig 8: PIX structural-break DiD ───────────────────────────────────────────
def fig8_pix_did(df: pd.DataFrame) -> None:
    """
    PIX structural-break DiD: incumbents (high TED/DOC exposure) vs digitals.
    Incumbents charged R$8-15 per TED; PIX made peer-to-peer transfers free
    from Nov 2020, reducing their 717xxx revenue. Digitals charged nothing
    pre-PIX → smaller or no decline.

    Shows quarterly averages for incumbents and digitals with 95% band,
    and the simple 2x2 DiD estimate in the subtitle.
    """
    q = quarterly_agg(df)
    q = q[q["year"].between(2018, 2024)].copy()   # exclude 2025+ COSIF break

    pix_date = pd.Timestamp("2020-10-01")   # Q4 2020
    q["post_pix"] = q["date"] >= pix_date
    q["group"]    = q["bank"].apply(
        lambda b: "Incumbent" if b in INCUMBENTS else "Digital" if b in DIGITALS else None
    )
    q = q[q["group"].notna()].copy()

    # Group averages per quarter
    grp_q = (
        q.groupby(["date", "group", "post_pix"], as_index=False)
        .agg(mean_ratio=("fee_ratio_q", "mean"),
             std_ratio =("fee_ratio_q",  "std"),
             n         =("fee_ratio_q", "count"))
    )
    grp_q["se"] = grp_q["std_ratio"] / np.sqrt(grp_q["n"].clip(lower=1))

    # Simple 2x2 DiD
    def _avg(group, post):
        return q[(q["group"] == group) & (q["post_pix"] == post)]["fee_ratio_q"].mean()

    did = (_avg("Incumbent", True) - _avg("Incumbent", False)) - \
          (_avg("Digital", True)   - _avg("Digital", False))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    group_styles = {
        "Incumbent": ("#003f88", "-",  "o"),
        "Digital":   ("#e85d04", "--", "^"),
    }
    for grp, (color, ls, mk) in group_styles.items():
        sub = grp_q[grp_q["group"] == grp].sort_values("date")
        ax.plot(sub["date"], sub["mean_ratio"] * 100,
                color=color, linestyle=ls, linewidth=1.8, marker=mk,
                markersize=3, label=grp)
        ax.fill_between(
            sub["date"],
            (sub["mean_ratio"] - 1.96 * sub["se"]) * 100,
            (sub["mean_ratio"] + 1.96 * sub["se"]) * 100,
            color=color, alpha=0.12,
        )

    ax.axvline(pix_date, color="black", lw=1.2, ls="--", alpha=0.7)
    ax.text(pix_date, ax.get_ylim()[1] * 0.97,
            " PIX launch\n Nov 2020", fontsize=8, va="top", color="black")

    ax.set_ylabel("Quarterly fee ratio × 100  (sum of 3 monthly increments / dep_total, %)")
    ax.set_title(
        f"Fig 8 — PIX Structural-Break DiD: Incumbents vs Digitals (2018–2024)\n"
        f"2×2 DiD estimate = {did*100:.3f} pp  (Incumbent post-pre minus Digital post-pre).\n"
        f"Incumbents with high pre-PIX TED/DOC exposure show larger fee ratio decline.",
        fontsize=9, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    _savefig(fig, "fee_reversal_fig8_pix_did.png")
    plt.close(fig)


# ── Fig 9: Formal quarterly event study around PIX launch ─────────────────────
def fig9_event_study(df: pd.DataFrame) -> None:
    """
    Quarterly event-study centred on Q4 2020 (PIX launch).

    Event time t=0 → Q4 2020.  t=-1 is omitted (normalisation period).
    Two series:
      - incumbent mean fee ratio (bank FE removed, relative to own t=-1 level)
      - digital mean fee ratio (same)
    95% bands from cross-bank dispersion (std / sqrt(n)).

    Persistence of the revenue shock is visible as how long β_t for incumbents
    remains below zero relative to the pre-PIX trend.
    """
    q = quarterly_agg(df)
    q = q[q["year"].between(2016, 2024)].copy()

    # Assign event time (quarters since Q4 2020)
    pix_yr, pix_qtr = 2020, 4
    q["event_t"] = (q["year"] - pix_yr) * 4 + (q["quarter"] - pix_qtr)

    # Keep a symmetric window: t in [-16, +16]
    q = q[q["event_t"].between(-16, 16)].copy()

    # Group label
    q["group"] = q["bank"].apply(
        lambda b: "Incumbent" if b in INCUMBENTS else "Digital" if b in DIGITALS else None
    )
    q = q[q["group"].notna()].copy()

    # Normalise each bank to its own t=-1 value (remove bank FE)
    base = q[q["event_t"] == -1].set_index("bank")["fee_ratio_q"]
    q["fee_ratio_norm"] = q.apply(
        lambda r: r["fee_ratio_q"] - base.get(r["bank"], r["fee_ratio_q"]), axis=1
    )

    # Group-quarter means and SE
    grp_evnt = (
        q.groupby(["event_t", "group"], as_index=False)
        .agg(
            mean_norm=("fee_ratio_norm", "mean"),
            std_norm =("fee_ratio_norm", "std"),
            n        =("fee_ratio_norm", "count"),
        )
    )
    grp_evnt["se"] = grp_evnt["std_norm"] / np.sqrt(grp_evnt["n"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    group_styles = {
        "Incumbent": ("#003f88", "-",  "o"),
        "Digital":   ("#e85d04", "--", "^"),
    }
    for grp, (color, ls, mk) in group_styles.items():
        sub = grp_evnt[grp_evnt["group"] == grp].sort_values("event_t")
        ax.plot(sub["event_t"], sub["mean_norm"] * 100,
                color=color, linestyle=ls, linewidth=1.8,
                marker=mk, markersize=4, label=grp)
        ax.fill_between(
            sub["event_t"],
            (sub["mean_norm"] - 1.96 * sub["se"]) * 100,
            (sub["mean_norm"] + 1.96 * sub["se"]) * 100,
            color=color, alpha=0.12,
        )

    ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.7,
               label="t=0: Q4 2020 (PIX launch)")
    ax.axvline(-1, color="#888", lw=0.8, ls=":", alpha=0.6,
               label="t=-1: normalisation period (omitted)")
    ax.axhline(0, color="black", lw=0.6)

    # Label every 4 quarters
    xticks = range(-16, 17, 4)
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([f"t={t}" for t in xticks], fontsize=8)

    ax.set_xlabel("Event time (quarters relative to PIX launch, Q4 2020)")
    ax.set_ylabel("Normalised fee ratio × 100 (pp relative to own t=-1)")
    ax.set_title(
        "Fig 9 — Quarterly Event Study: Fee Ratio around PIX Launch\n"
        "Each bank normalised to its t=-1 level; band = 95% CI from cross-bank dispersion.\n"
        "Incumbents show persistent post-PIX decline; digitals show near-zero pre- and post.",
        fontsize=9, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    _savefig(fig, "fee_reversal_fig9_event_study.png")
    plt.close(fig)


# ── Fig 10: 2025 COSIF reclassification correction factors ────────────────────
def fig10_correction_factor(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each bank, fit a linear trend in log(fee_ratio) over 2018-2024 using
    December increments (end-of-H2 increments, most reliable observation per year).
    Project the expected 2025 level and compute:
        correction_factor = projected_2025 / actual_2025

    A factor > 1 → actual 2025 understates the true level (reclassification deflated).
    A factor < 1 → actual 2025 overstates (reclassification inflated 717xxx scope).

    Saves fee_reversal_correction_factors.csv.
    """
    # Use Dec observations (H2 end, most stable increment)
    dec = df[df["month"] == 12].copy()
    dec["ratio_ann"] = dec["fee_ratio_td_corr"] * 12 * 100   # pp/yr annualised

    rows = []
    for bank in INCUMBENTS + DIGITALS:
        bdf = dec[dec["bank"] == bank].set_index("year").sort_index()
        # Fit on 2018-2024
        train = bdf[bdf.index.isin(range(2018, 2025)) & (bdf["ratio_ann"] > 0)]
        if len(train) < 3:
            continue
        # Log-linear trend
        yrs = train.index.values.astype(float)
        log_r = np.log(train["ratio_ann"].values)
        coef = np.polyfit(yrs, log_r, 1)          # [slope, intercept]
        expected_log_2025 = np.polyval(coef, 2025.0)
        expected_2025 = np.exp(expected_log_2025)

        actual_2025 = bdf.loc[2025, "ratio_ann"] if 2025 in bdf.index else np.nan
        actual_2024 = bdf.loc[2024, "ratio_ann"] if 2024 in bdf.index else np.nan
        factor = expected_2025 / actual_2025 if (not np.isnan(actual_2025) and actual_2025 > 0) else np.nan

        rows.append({
            "bank":            bank,
            "trend_slope_pct": coef[0] * 100,         # log-slope × 100 ≈ % per year
            "expected_2025":   expected_2025,
            "actual_2025":     actual_2025,
            "actual_2024":     actual_2024,
            "correction_factor": factor,
            "r_train_2018_24": train["ratio_ann"].values,
            "yrs_train":       yrs,
        })

    cf = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("r_train_2018_24", "yrs_train")} for r in rows])
    csv_out = OUT_DIR / "fee_reversal_correction_factors.csv"
    cf.to_csv(csv_out, index=False, float_format="%.4f")
    print(f"  saved -> {csv_out.name}")

    # ── Figure ──────────────────────────────────────────────────────────────
    banks_plot = [r["bank"] for r in rows if not np.isnan(r.get("correction_factor", np.nan))]
    n = len(rows)
    if n == 0:
        print("  [Fig 10 skipped] no correction factors computed")
        return cf

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.5))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, r in zip(axes, rows):
        bank = r["bank"]
        yrs_all = r["yrs_train"]
        log_r_all = np.log(r["r_train_2018_24"])
        coef_v = np.polyfit(yrs_all, log_r_all, 1)

        # Plot actual 2018-2024
        actual_series = dec[(dec["bank"] == bank) & dec["year"].between(2018, 2025)]
        ax.scatter(actual_series["year"], actual_series["ratio_ann"],
                   color=COLORS.get(bank, "#888"), s=30, zorder=3, label="Actual")

        # Plot trend line 2018→2025
        fit_x = np.linspace(2018, 2025.5, 60)
        fit_y = np.exp(np.polyval(coef_v, fit_x))
        ax.plot(fit_x, fit_y, "--", color=COLORS.get(bank, "#888"), lw=1.2, alpha=0.7,
                label=f"Trend ({coef_v[0]*100:+.1f}%/yr)")

        # Mark 2025 actual vs expected
        exp = r["expected_2025"]
        act = r["actual_2025"]
        ax.axvline(2025, color="#555", lw=0.7, ls=":")
        if not np.isnan(act):
            ax.scatter([2025], [act], marker="x", s=60, color="red", zorder=4, label="Actual 2025")
        ax.scatter([2025], [exp], marker="*", s=80, color="black", zorder=4, label="Expected 2025")

        cf_val = r["correction_factor"]
        subtitle = f"CF = {cf_val:.2f}" if not np.isnan(cf_val) else "CF = N/A"
        ax.set_title(f"{bank}\n{subtitle}", fontsize=8.5, fontweight="bold",
                     color=COLORS.get(bank, "#888"))
        ax.set_xlabel("Year", fontsize=7.5)
        ax.set_ylabel("Ann. fee ratio (%/yr)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5)

    for ax in axes[len(rows):]:
        ax.set_visible(False)

    fig.suptitle(
        "Fig 10 — 2025 COSIF Reclassification Correction Factors\n"
        "CF = expected_2025 (log-linear trend on 2018-2024) / actual_2025.\n"
        "Red x = actual 2025; black star = projected.  CF > 1 means actual understates.",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "fee_reversal_fig10_correction_factor.png")
    plt.close(fig)
    return cf


# ── Fig 11: CVM DRE vs COSIF cross-validation ─────────────────────────────────
def fig11_cvm_vs_cosif(df: pd.DataFrame) -> None:
    """
    Time-series comparison of annual fee/service revenue (CVM DRE, BRL bn)
    against annualised COSIF fee ratio × total deposits (also BRL bn).

    CVM = consolidated holding company revenue (all business lines).
    COSIF = bank-entity 717xxx (deposit account services only).

    The ratio CVM/COSIF measures how much of total service revenue comes from
    deposit-account fees vs. other fee lines (insurance, asset mgmt, cards).
    Expects ~2–5× for large diversified banks (only a fraction is 717xxx).

    Uses December COSIF observation (H2-end increment × 12 months) × dep_total.
    """
    if not CVM_FEE_CSV.exists():
        print(f"  [Fig 11 skipped] {CVM_FEE_CSV.name} not found — run scrape_24 first")
        return

    cvm = pd.read_csv(CVM_FEE_CSV, low_memory=False)
    cvm = cvm[cvm["firm_key"] != "firm_key"].copy()
    cvm["value"] = pd.to_numeric(cvm["value"], errors="coerce")
    cvm["period_year"] = pd.to_numeric(cvm["period_year"], errors="coerce")
    cvm["period_quarter"] = pd.to_numeric(cvm["period_quarter"], errors="coerce")

    # Take DFP (full-year Q4) best metric per firm-year
    dfp = cvm[(cvm["filing_form"] == "DFP") & (cvm["period_quarter"] == 4)].copy()
    dfp["mrank"] = dfp["metric"].map({"fee_revenue": 0, "service_revenue": 1})
    dfp = dfp.sort_values(["firm_key", "period_year", "mrank"])
    dfp_best = dfp.groupby(["firm_key", "period_year"], as_index=False).first()

    # Map CVM firm_key → bank label used in COSIF
    cvm_to_bank = {
        "bb":           "BB",
        "bradesco":     "Bradesco",
        "itau":         "Itaú",
        "santander_br": "Santander",
        "inter":        "Inter",
    }

    # COSIF: annualised Dec increment (rev × 12) in BRL bn
    dec = df[df["month"] == 12].copy()
    dec["cosif_rev_ann_bn"] = dec["svc_revenue_inc"] * 12 / 1e9   # H2-end increment → annual
    dec["dep_total_bn"]     = dec["dep_total"] / 1e9

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: absolute BRL bn comparison ───────────────────────────────────
    ax = axes[0]
    for fkey, bank in cvm_to_bank.items():
        color = COLORS.get(bank, "#888")
        ls_c = "-" if bank in INCUMBENTS else "--"
        # CVM
        sub_cvm = dfp_best[dfp_best["firm_key"] == fkey].sort_values("period_year")
        ax.plot(sub_cvm["period_year"], sub_cvm["value"] / 1e9,
                color=color, linestyle=ls_c, linewidth=1.8, marker="o", markersize=3,
                label=f"{bank} CVM")
        # COSIF
        sub_cos = dec[dec["bank"] == bank].sort_values("year")
        ax.plot(sub_cos["year"], sub_cos["cosif_rev_ann_bn"],
                color=color, linestyle=":", linewidth=1.2, marker="^", markersize=3,
                label=f"{bank} COSIF×12")

    ax.set_ylabel("Annual fee/service revenue (BRL bn)")
    ax.set_title("Left: CVM DRE (solid) vs COSIF annualised (dotted)\n"
                 "CVM includes all fee lines; COSIF = 717xxx (deposit acct only).", fontsize=8.5)
    ax.legend(ncol=2, fontsize=6.5, loc="upper left")
    ax.axvline(2020.9, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Year")

    # ── Right: CVM/COSIF ratio — how much of CVM is deposit-acct fees ──────
    ax = axes[1]
    for fkey, bank in cvm_to_bank.items():
        color = COLORS.get(bank, "#888")
        sub_cvm = dfp_best[dfp_best["firm_key"] == fkey].set_index("period_year")["value"]
        sub_cos = dec[dec["bank"] == bank].set_index("year")["cosif_rev_ann_bn"] * 1e9

        common_yrs = sorted(set(sub_cvm.index) & set(sub_cos.index))
        if len(common_yrs) < 2:
            continue
        ratio = pd.Series(
            {y: sub_cos.get(y, np.nan) / sub_cvm.get(y, np.nan)
             for y in common_yrs}
        ).dropna()
        ax.plot(ratio.index, ratio.values * 100,
                color=color, linewidth=1.8, marker="o", markersize=4, label=bank)

    ax.set_ylabel("COSIF 717xxx / CVM total fee rev (%) — deposit-acct share")
    ax.set_title("Right: fraction of total CVM fee income attributable\n"
                 "to deposit-account services (COSIF 717xxx / CVM DRE).", fontsize=8.5)
    ax.axhline(100, color="#555", lw=0.5, ls="--")
    ax.set_ylim(0, 120)
    ax.legend(fontsize=8)
    ax.axvline(2020.9, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Year")

    fig.suptitle(
        "Fig 11 — CVM DRE vs COSIF: Total Fee Revenue and Deposit-Account Share\n"
        "Validates both series; right panel shows what fraction of CVM revenue is 717xxx.",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, "fee_reversal_fig11_cvm_vs_cosif.png")
    plt.close(fig)


# ── Fig 12: annualised fee ratio per bank — all 9 institutions ────────────────
def fig12_fee_ratio_per_bank(df: pd.DataFrame) -> None:
    """
    3×3 panel: one subplot per institution, Y = annualised fee ratio (%/yr).
    Uses quarterly aggregation (sum of 3 monthly increments / EOM dep_total × 4 × 100).
    Each subplot has its own Y scale — digitals (Nubank, Mercado Pago) run 30-400%/yr
    due to tiny deposit bases; incumbents run 2-7%/yr.
    """
    q = quarterly_agg(df)
    q = q[q["year"] >= 2013].copy()

    all_banks = INCUMBENTS + DIGITALS   # BB, Caixa, Bradesco, Itaú, Santander, Inter, Nubank, C6, MP
    ncols, nrows = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 11), sharex=False, sharey=False)
    axes_flat = axes.flatten()

    pix_line   = pd.Timestamp("2020-11-01")
    reclass_lo = pd.Timestamp("2025-01-01")
    reclass_hi = pd.Timestamp("2026-12-31")

    for ax, bank in zip(axes_flat, all_banks):
        bdf = q[q["bank"] == bank].sort_values("date")
        color = COLORS[bank]

        ax.plot(bdf["date"], bdf["fee_ratio_q_ann"],
                color=color, linewidth=1.6, zorder=3)
        ax.fill_between(bdf["date"], 0, bdf["fee_ratio_q_ann"],
                        color=color, alpha=0.10)

        # PIX launch
        ymax = bdf["fee_ratio_q_ann"].max() if not bdf.empty else 1
        ax.axvline(pix_line, color="#444", lw=0.9, ls="--", alpha=0.7)

        # 2025 reclassification band
        ax.axvspan(reclass_lo, reclass_hi, alpha=0.08, color="red", linewidth=0)

        # Annotations only if there is room (skip for very noisy digitals)
        if ymax < 20:
            ax.text(pix_line, ymax * 0.92, " PIX", fontsize=6.5, color="#444", va="top")

        ax.set_title(bank, fontsize=9, fontweight="bold", color=color)
        ax.set_ylabel("%/yr", fontsize=7.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.tick_params(axis="x", rotation=30, labelsize=6.5)
        ax.tick_params(axis="y", labelsize=7)

        # Y floor at 0 unless bank has negative quarters
        if bdf["fee_ratio_q_ann"].min() >= -0.1:
            ax.set_ylim(bottom=0)

    # Hide the unused 9th cell (we have 9 banks in 3×3 so nothing hidden)
    for ax in axes_flat[len(all_banks):]:
        ax.set_visible(False)

    # Shared legend for reference lines
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#444", lw=1, ls="--", label="PIX launch (Nov 2020)"),
        mpatches.Patch(color="red", alpha=0.15, label="2025+ COSIF reclassification"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Fig 12 — Annualised Fee Ratio per Bank (%/yr)\n"
        "= sum of 3 monthly increments / EOM dep_total × 400.  Each subplot has its own Y scale.\n"
        "Digitals (Nubank, Mercado Pago) have tiny deposit bases → ratios in hundreds of %/yr.",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    _savefig(fig, "fee_reversal_fig12_fee_ratio_per_bank.png")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    df = load_data()

    print("\n=== Generating figures ===")
    fig1_revenue_monthly(df)
    fig2_corrected_quarterly(df)
    fig3_pf_pj_split(df)
    summary = fig4_decomposition(df)
    fig5_h1_h2_ratio(df)
    fig6_cosif_vs_bcb(df)
    fig7_atm_vs_service(df)
    fig8_pix_did(df)
    fig9_event_study(df)
    cf = fig10_correction_factor(df)
    fig11_cvm_vs_cosif(df)
    fig12_fee_ratio_per_bank(df)

    # ── Save summary CSV ────────────────────────────────────────────────────
    csv_out = OUT_DIR / "fee_reversal_summary.csv"
    summary.to_csv(csv_out, index=False, float_format="%.6f")
    print(f"\nSummary CSV -> {csv_out}")

    print("\n=== Correction factors (2025 COSIF reclassification) ===")
    if not cf.empty:
        print(cf[["bank", "trend_slope_pct", "expected_2025", "actual_2025",
                   "correction_factor"]].round(3).to_string(index=False))

    # ── Print key tables ────────────────────────────────────────────────────
    q = quarterly_agg(df)
    dec = df[df["month"] == 12].copy()
    dec["ratio_ann_pct"] = dec["fee_ratio_td_corr"] * 12 * 100
    ratio_tbl = dec[dec["year"].between(2018, 2025)].pivot_table(
        index="year", columns="bank",
        values="ratio_ann_pct", aggfunc="first"
    )
    ordered = [b for b in (INCUMBENTS + DIGITALS) if b in ratio_tbl.columns]
    print("\n=== Annualized fee ratio %/yr (Dec increment / dep_total × 12 × 100) ===")
    print(ratio_tbl[ordered].round(3).to_string())

    print("\n=== 2024->2025 decomposition (Dec observations) ===")
    y25 = summary[(summary["year"] == 2025) & summary["bank"].isin(INCUMBENTS + ["Inter"])]
    if not y25.empty:
        print(y25[["bank", "fee_ratio_prior", "fee_ratio", "delta_ratio",
                   "revenue_effect", "deposit_effect",
                   "rev_growth_pct", "dep_growth_pct"]].round(5).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
