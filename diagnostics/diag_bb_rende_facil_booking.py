"""
diag_bb_rende_facil_booking.py
==============================
Where is BB Rende Fácil booked? Checks 2 and 3 of task BB3 (claude/bb_handoff_2026-11-03.md).

Rende Fácil sweeps the positive checking balance into an application with daily liquidity and a
721-day maturity, paying 10-50% of the DI for the first 360 days (adhesion term V1, in force from
March 2022, delivered under LAI 18882.002114). If the swept money sits in BB's time deposits, BB's
time-deposit stock rises and its demand deposits fall from 2022-03, against its own trend and
against other large banks, and BB's measured time-deposit rate, as a share of the Selic, falls from
2022Q2.

Inputs (read only):
  - processed/COSIF_PROCESSED/custos_implicitos_{BANCO_DO_BRASIL_-_BANCO_MULTIPLO,BANCOS_MULTIPLOS,
    CAIXA_ECONOMICA_FEDERAL}.csv (panel_cosif_extract.py): monthly DOC 4010 stocks per institution
    (CNPJ): stk_time (4.1.5), stk_demand (4.1.1), stk_savings (4.1.2), stk_repos (4.2,
    compromissadas).
  - processed/market_panel.parquet, read with pyarrow (load_panel_cached may rewrite the sidecar):
    dep_a4 summed over markets, rate_a4_national, risk_free_qoq (Selic), spread_ann_a4_national and
    rate_a4_guarded per conglomerate-quarter.
Outputs (Rout):
  - fig_bb_rende_facil_cosif.pdf: check 2, BB against Itaú, Bradesco, Santander and Caixa, monthly.
  - bb_rende_facil_cosif_monthly.csv: the data behind the figure.
  - bb_rende_facil_booking.csv: check 3, conglomerate x quarter, 2020Q1-2024Q4.
  - a summary printed to the screen.
Tests (check 2): with g_t = 100 x (ln BB_t - mean over controls of ln C_t), rebased to 2022-02 = 0,
a linear trend fitted on 2021-01..2022-02 is extrapolated, and the mean deviation over 2022-03..06
is reported with the pre-period residual SD; the same for ln BB_t against its own trend. Each
control against the other three controls is the placebo.
Usage:  python diagnostics/diag_bb_rende_facil_booking.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import paths

COSIF_DIR = paths.PROCESSED / "COSIF_PROCESSED"
COSIF_FILES = ["custos_implicitos_BANCO_DO_BRASIL_-_BANCO_MULTIPLO.csv",
               "custos_implicitos_BANCOS_MULTIPLOS.csv",
               "custos_implicitos_CAIXA_ECONOMICA_FEDERAL.csv"]
PANEL = paths.PROCESSED / "market_panel.parquet"
OUT = paths.rout_dir()

# bank: (Cosif institution CNPJ, panel conglomerate code)
BANKS = {"BB": ("00000000", "C0080329"), "Itaú": ("60701190", "C0080099"),
         "Bradesco": ("60746948", "C0080075"), "Santander": ("90400888", "C0080185"),
         "Caixa": ("00360305", "C0080738")}
CONTROLS = ["Itaú", "Bradesco", "Santander", "Caixa"]
SERIES = {"stk_time": "Time deposits (4.1.5)", "stk_demand": "Demand deposits (4.1.1)",
          "stk_savings": "Savings (4.1.2)", "stk_repos": "Repurchase agreements (4.2)"}
BASE = pd.Timestamp("2022-02-01")
PRE = (pd.Timestamp("2021-01-01"), BASE)
EVENT = (pd.Timestamp("2022-03-01"), pd.Timestamp("2022-06-01"))
M0, M1 = pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-01")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREY = "#2a78d6", "#8a8984"


def load_cosif():
    cnpj2bank = {c: b for b, (c, _) in BANKS.items()}
    fr = []
    for f in COSIF_FILES:
        x = pd.read_csv(COSIF_DIR / f, dtype={"CNPJ": str}, parse_dates=["DATA_BASE"])
        fr.append(x[x["CNPJ"].isin(cnpj2bank)])
    x = pd.concat(fr)
    x = x[(x["DATA_BASE"] >= M0) & (x["DATA_BASE"] <= M1)]
    x["bank"] = x["CNPJ"].map(cnpj2bank)
    long = x.melt(id_vars=["bank", "DATA_BASE"], value_vars=list(SERIES), var_name="series",
                  value_name="reais")
    long["bn"] = long["reais"] / 1e9
    assert long.groupby(["bank", "series"]).size().eq(36).all(), "expected 36 months per bank-series"
    assert long["bn"].gt(0).all(), "non-positive Cosif stock"
    return long.drop(columns="reais")


def trend_deviation(t, y):
    """Fit y on t over PRE, extrapolate; return (mean deviation over EVENT, pre residual SD, fit)."""
    pre = ((t >= PRE[0]) & (t <= PRE[1])).to_numpy()
    ev = ((t >= EVENT[0]) & (t <= EVENT[1])).to_numpy()
    k = ((t - PRE[0]).dt.days / 30.4375).to_numpy()
    b = np.polyfit(k[pre], y[pre], 1)
    fit = np.polyval(b, k)
    resid = y - fit
    return resid[ev].mean(), resid[pre].std(ddof=2), fit


def check2(long):
    w = long.pivot_table(index=["series", "DATA_BASE"], columns="bank", values="bn").reset_index()
    rows, out = [], []
    for s in SERIES:
        d = w[w["series"] == s].sort_values("DATA_BASE").reset_index(drop=True)
        t = d["DATA_BASE"]
        at0 = (t == BASE).to_numpy()
        lb = np.log(d["BB"].to_numpy())
        lc = np.log(d[CONTROLS].to_numpy()).mean(axis=1)
        g = 100 * (lb - lc)
        g = g - g[at0][0]
        dev_g, sd_g, fit_g = trend_deviation(t, g)
        own = 100 * (lb - lb[at0][0])
        dev_o, sd_o, _ = trend_deviation(t, own)
        bb0 = d.loc[t == BASE, "BB"].iloc[0]
        plac = []
        for c in CONTROLS:
            oth = [o for o in CONTROLS if o != c]
            gc = 100 * (np.log(d[c].to_numpy()) - np.log(d[oth].to_numpy()).mean(axis=1))
            gc = gc - gc[at0][0]
            plac.append((c, trend_deviation(t, gc)[0]))
        rows.append(dict(series=s, bb_feb22_bn=bb0,
                         bb_jun22_bn=d.loc[t == EVENT[1], "BB"].iloc[0],
                         dev_vs_controls_lp=dev_g, pre_sd_vs_controls_lp=sd_g,
                         dev_vs_controls_bn=bb0 * (np.exp(dev_g / 100) - 1),
                         dev_own_trend_lp=dev_o, pre_sd_own_lp=sd_o,
                         placebo_lp="; ".join(f"{c} {v:+.1f}" for c, v in plac)))
        out.append(d.assign(gap_vs_controls_lp=g, gap_trend_lp=fit_g))
    return pd.DataFrame(rows), pd.concat(out)


def figure(w, path):
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.35, 1]))
    for j, s in enumerate(SERIES):
        d = w[w["series"] == s].sort_values("DATA_BASE")
        t = d["DATA_BASE"]
        top, bot = axes[0, j], axes[1, j]
        for ax in (top, bot):
            ax.axvspan(EVENT[0], EVENT[1] + pd.offsets.MonthEnd(0), color="#f1f0ec", zorder=0, lw=0)
            ax.axvline(EVENT[0], color=INK2, lw=0.8, ls=":", zorder=1)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8, length=0)
        base = d[t == BASE]
        idx = lambda col: 100 * d[col].to_numpy() / base[col].iloc[0]
        ends = []
        for c in CONTROLS:
            top.plot(t, idx(c), color=GREY, lw=1.0, zorder=2)
            ends.append([idx(c)[-1], c])
        cm = 100 * np.exp(np.log(d[CONTROLS].to_numpy()).mean(axis=1) - np.log(base[CONTROLS].to_numpy()).mean())
        top.plot(t, cm, color=INK2, lw=1.4, ls="--", zorder=3)
        top.plot(t, idx("BB"), color=BLUE, lw=2.0, zorder=4)
        ends.append([idx("BB")[-1], "BB"])
        ends.sort()
        span = np.ptp(top.get_ylim())
        for k in range(1, len(ends)):  # keep end labels at least 4% of the axis span apart
            if ends[k][0] - ends[k - 1][0] < 0.04 * span:
                ends[k][0] = ends[k - 1][0] + 0.04 * span
        for y, c in ends:
            bb = c == "BB"
            top.annotate(c, (t.iloc[-1], y), xytext=(4, 0), textcoords="offset points", va="center",
                         fontsize=7.5 if bb else 7, color=INK if bb else INK2,
                         fontweight="bold" if bb else "normal")
        top.set_title(SERIES[s], fontsize=9.5, color=INK, loc="left")
        upto = (t <= EVENT[1]).to_numpy()
        bot.plot(t[upto], d["gap_trend_lp"].to_numpy()[upto], color=GREY, lw=1.0, ls="--", zorder=2)
        bot.plot(t, d["gap_vs_controls_lp"], color=BLUE, lw=2.0, marker="o", ms=3, zorder=3)
        bot.axhline(0, color=INK2, lw=0.6)
        bot.xaxis.set_major_locator(mdates.YearLocator())
        bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        top.set_xlim(t.iloc[0], t.iloc[-1] + pd.Timedelta(days=150))
    axes[0, 0].set_ylabel("Index, 2022-02 = 100", fontsize=8.5, color=INK2)
    axes[1, 0].set_ylabel("BB minus controls, log points\n(2022-02 = 0; dashed: 2021-01..2022-02 trend)", fontsize=8.5, color=INK2)
    axes[0, 0].annotate("Rende Fácil V1\n(from 2022-03)", (EVENT[0], axes[0, 0].get_ylim()[1]),
                        xytext=(3, -2), textcoords="offset points", va="top", fontsize=7, color=INK2)
    handles = [plt.Line2D([], [], color=BLUE, lw=2.0, label="Banco do Brasil"),
               plt.Line2D([], [], color=INK2, lw=1.4, ls="--", label="Control mean (geometric)"),
               plt.Line2D([], [], color=GREY, lw=1.0, label="Controls: Itaú, Bradesco, Santander, Caixa")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.995), ncol=3, frameon=False,
               fontsize=8.5, labelcolor=INK)
    fig.suptitle("BB against four large banks: Cosif month-end balances, 2021-01 to 2023-12", x=0.06,
                 y=1.045, ha="left", fontsize=11, color=INK)
    fig.text(0.06, 0.005, "Source: BCB Cosif balancetes (DOC 4010), individual institution (BB Banco Múltiplo, "
             "Itaú Unibanco S.A., Banco Bradesco, Banco Santander (Brasil), Caixa), via panel_cosif_extract.py. "
             "Shaded: 2022-03 to 2022-06.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def check3(cosif_long):
    cols = ["CodConglomeradoPrudencial", "year", "quarter", "dep_a4", "rate_a4_national",
            "risk_free_qoq", "spread_ann_a4_national", "rate_a4_guarded"]
    code2bank = {k: b for b, (_, k) in BANKS.items()}
    p = pq.read_table(PANEL, columns=cols,
                      filters=[("CodConglomeradoPrudencial", "in", list(code2bank)),
                               ("year", ">=", 2020), ("year", "<=", 2024)]).to_pandas()
    chk = p.groupby(["CodConglomeradoPrudencial", "year", "quarter"])[["rate_a4_national", "risk_free_qoq"]].nunique()
    assert chk.le(1).all().all(), "rate_a4_national or risk_free_qoq varies within a conglomerate-quarter"
    q = (p.groupby(["CodConglomeradoPrudencial", "year", "quarter"])
         .agg(dep_a4=("dep_a4", "sum"), n_markets=("dep_a4", "size"),
              rate_a4_national_qoq=("rate_a4_national", "first"), risk_free_qoq=("risk_free_qoq", "first"),
              spread_ann_a4_national=("spread_ann_a4_national", "first"),
              rate_a4_guarded=("rate_a4_guarded", "max"))
         .reset_index())
    q["bank"] = q["CodConglomeradoPrudencial"].map(code2bank)
    q["period"] = q["year"].astype(str) + "Q" + q["quarter"].astype(str)
    q["dep_a4_bn"] = q["dep_a4"] / 1e9
    q["rate_a4_national_ann"] = (1 + q["rate_a4_national_qoq"]) ** 4 - 1
    q["risk_free_ann"] = (1 + q["risk_free_qoq"]) ** 4 - 1
    q["rate_share_of_selic"] = q["rate_a4_national_qoq"] / q["risk_free_qoq"]
    cm = q[q["bank"].isin(CONTROLS)].groupby("period")["rate_share_of_selic"].mean()
    q["share_minus_control_mean"] = q["rate_share_of_selic"] - q["period"].map(cm)
    # Cosif time-deposit stock of the bank entity at the quarter's last month, for scale (2021-2023 only)
    cs = cosif_long[cosif_long["series"] == "stk_time"].copy()
    cs = cs[cs["DATA_BASE"].dt.month.isin([3, 6, 9, 12])]
    cs["period"] = cs["DATA_BASE"].dt.year.astype(str) + "Q" + ((cs["DATA_BASE"].dt.month - 1) // 3 + 1).astype(str)
    q = q.merge(cs[["bank", "period", "bn"]].rename(columns={"bn": "cosif_stk_time_entity_bn"}),
                on=["bank", "period"], how="left")
    order = ["bank", "CodConglomeradoPrudencial", "period", "year", "quarter", "dep_a4_bn", "n_markets",
             "cosif_stk_time_entity_bn", "rate_a4_national_qoq", "rate_a4_national_ann", "risk_free_qoq",
             "risk_free_ann", "rate_share_of_selic", "share_minus_control_mean", "spread_ann_a4_national",
             "rate_a4_guarded"]
    q = q[order].sort_values(["bank", "year", "quarter"]).reset_index(drop=True)
    assert len(q) == 5 * 20, f"expected 100 conglomerate-quarters, got {len(q)}"
    return q


def main():
    long = load_cosif()
    summ, w = check2(long)
    figure(w, OUT / "fig_bb_rende_facil_cosif.pdf")
    w.to_csv(OUT / "bb_rende_facil_cosif_monthly.csv", index=False, float_format="%.6f")
    q = check3(long)
    q.to_csv(OUT / "bb_rende_facil_booking.csv", index=False, float_format="%.6f")

    pd.set_option("display.width", 220)
    print(f"Outputs in {OUT}")
    print("\nCHECK 2 (Cosif, monthly). Deviation over 2022-03..06 from the 2021-01..2022-02 trend, log points:")
    print(summ.round(2).to_string(index=False))
    bb = w[w["DATA_BASE"].between(pd.Timestamp("2021-10-01"), pd.Timestamp("2022-09-01"))]
    print("\nBB month-end balances, R$ bn:")
    print(bb.pivot_table(index="DATA_BASE", columns="series", values="BB").round(1).to_string())
    print("\nCHECK 3 (panel, quarterly). rate_a4_national as a share of the Selic quarterly rate:")
    print(q.pivot_table(index="period", columns="bank", values="rate_share_of_selic").round(3).to_string())
    print("\nBB dep_a4 (R$ bn, sum over markets) against the Cosif entity stock:")
    print(q[q["bank"] == "BB"].set_index("period")[["dep_a4_bn", "cosif_stk_time_entity_bn", "rate_a4_guarded"]].round(1).to_string())
    pre = q["year"].lt(2022) | (q["year"].eq(2022) & q["quarter"].eq(1))
    bbq = q[q["bank"] == "BB"].assign(pre=pre)
    print("\nBB share minus control mean: pre (2020Q1-2022Q1) mean %.3f, post (2022Q2-2024Q4) mean %.3f, "
          "difference %.3f" % (bbq.loc[bbq["pre"], "share_minus_control_mean"].mean(),
                               bbq.loc[~bbq["pre"], "share_minus_control_mean"].mean(),
                               bbq.loc[~bbq["pre"], "share_minus_control_mean"].mean()
                               - bbq.loc[bbq["pre"], "share_minus_control_mean"].mean()))


if __name__ == "__main__":
    main()
