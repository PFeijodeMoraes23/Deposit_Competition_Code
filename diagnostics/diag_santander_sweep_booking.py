"""
diag_santander_sweep_booking.py
===============================
Does Santander's default sweep move its measured time deposits? Task BB5 (claude/bb_handoff_2026-11-03.md),
checks 2 and 3, in the layout of diag_bb_rende_facil_booking.py.

Santander's PF general conditions, form 1614-506 (05/2022), make "Investimento Automático" a built-in feature of
the checking account (clauses 80-82; clause 80 names the instrument as a CDB of the bank, the only document
found that does); from 2022-05-13 the rate is progressive, and from 2022-09-30 the tiers are 2/30/60/100% of
the CDI. A default sweep that books into CDBs lowers demand deposits and raises time deposits from 2022-05,
against Santander's own trend and against other large banks.

Inputs (read only): the same Cosif files as diag_bb_rende_facil_booking.py (monthly DOC 4010 stocks per
institution); with --with-panel, processed/market_panel.parquet through pyarrow (dep_a4, rate_a4_national,
risk_free_qoq per conglomerate-quarter).
Outputs (Rout): fig_santander_sweep_cosif.pdf; santander_sweep_cosif_monthly.csv; with --with-panel,
santander_sweep_booking.csv (conglomerate x quarter, 2020Q1-2024Q4).
Tests (check 2): g_t = 100 x (ln Santander_t - mean over controls of ln C_t), rebased to 2022-04 = 0; a linear
trend on 2021-01..2022-04 is extrapolated and the mean deviation over 2022-05..08 is reported with the pre-period
residual SD, for the handoff's controls (BB, Itaú, Bradesco, Caixa) and without BB, whose own time deposits
surged in 2022-03..08 (task BB3). Each control against the others is the placebo.
Usage:  python diagnostics/diag_santander_sweep_booking.py [--with-panel]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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

BANKS = {"Santander": ("90400888", "C0080185"), "BB": ("00000000", "C0080329"),
         "Itaú": ("60701190", "C0080099"), "Bradesco": ("60746948", "C0080075"),
         "Caixa": ("00360305", "C0080738")}
TREATED = "Santander"
CONTROL_SETS = {"BB, Itaú, Bradesco, Caixa": ["BB", "Itaú", "Bradesco", "Caixa"],
                "Itaú, Bradesco, Caixa (no BB)": ["Itaú", "Bradesco", "Caixa"]}
SERIES = {"stk_time": "Time deposits (4.1.5)", "stk_demand": "Demand deposits (4.1.1)",
          "stk_savings": "Savings (4.1.2)", "stk_repos": "Repurchase agreements (4.2)"}
BASE = pd.Timestamp("2022-04-01")
PRE = (pd.Timestamp("2021-01-01"), BASE)
EVENT = (pd.Timestamp("2022-05-01"), pd.Timestamp("2022-08-01"))
TIERS = pd.Timestamp("2022-10-01")       # first month-end under the 2022-09-30 tiers
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
    long = x.melt(id_vars=["bank", "DATA_BASE"], value_vars=list(SERIES), var_name="series", value_name="reais")
    long["bn"] = long["reais"] / 1e9
    assert long.groupby(["bank", "series"]).size().eq(36).all(), "expected 36 months per bank-series"
    assert long["bn"].gt(0).all(), "non-positive Cosif stock"
    return long.drop(columns="reais")


def trend_deviation(t, y):
    pre = ((t >= PRE[0]) & (t <= PRE[1])).to_numpy()
    ev = ((t >= EVENT[0]) & (t <= EVENT[1])).to_numpy()
    k = ((t - PRE[0]).dt.days / 30.4375).to_numpy()
    b = np.polyfit(k[pre], y[pre], 1)
    fit = np.polyval(b, k)
    resid = y - fit
    return resid[ev].mean(), resid[pre].std(ddof=2), fit


def gap(d, treated, controls, at0):
    g = 100 * (np.log(d[treated].to_numpy()) - np.log(d[controls].to_numpy()).mean(axis=1))
    return g - g[at0][0]


def check2(long):
    w = long.pivot_table(index=["series", "DATA_BASE"], columns="bank", values="bn").reset_index()
    rows, out = [], []
    main_controls = CONTROL_SETS["BB, Itaú, Bradesco, Caixa"]
    for s in SERIES:
        d = w[w["series"] == s].sort_values("DATA_BASE").reset_index(drop=True)
        t = d["DATA_BASE"]
        at0 = (t == BASE).to_numpy()
        s0 = d.loc[at0, TREATED].iloc[0]
        lt = np.log(d[TREATED].to_numpy())
        own = 100 * (lt - lt[at0][0])
        dev_o, sd_o, _ = trend_deviation(t, own)
        for lab, ctl in CONTROL_SETS.items():
            g = gap(d, TREATED, ctl, at0)
            dev_g, sd_g, fit_g = trend_deviation(t, g)
            plac = [(c, trend_deviation(t, gap(d, c, [o for o in ctl if o != c], at0))[0]) for c in ctl]
            rows.append(dict(series=s, controls=lab, treated_apr22_bn=s0,
                             treated_aug22_bn=d.loc[t == EVENT[1], TREATED].iloc[0],
                             dev_vs_controls_lp=dev_g, pre_sd_vs_controls_lp=sd_g,
                             dev_vs_controls_bn=s0 * (np.exp(dev_g / 100) - 1),
                             dev_own_trend_lp=dev_o, pre_sd_own_lp=sd_o,
                             placebo_lp="; ".join(f"{c} {v:+.1f}" for c, v in plac)))
            if ctl == main_controls:
                d = d.assign(gap_vs_controls_lp=g, gap_trend_lp=fit_g)
        d = d.assign(gap_vs_controls_nobb_lp=gap(d, TREATED, CONTROL_SETS["Itaú, Bradesco, Caixa (no BB)"], at0))
        out.append(d)
    return pd.DataFrame(rows), pd.concat(out)


def figure(w, path):
    controls = CONTROL_SETS["BB, Itaú, Bradesco, Caixa"]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
    for j, s in enumerate(SERIES):
        d = w[w["series"] == s].sort_values("DATA_BASE")
        t = d["DATA_BASE"]
        top, bot = axes[0, j], axes[1, j]
        for ax in (top, bot):
            ax.axvspan(EVENT[0], EVENT[1] + pd.offsets.MonthEnd(0), color="#f1f0ec", zorder=0, lw=0)
            ax.axvline(EVENT[0], color=INK2, lw=0.8, ls=":", zorder=1)
            ax.axvline(TIERS, color=INK2, lw=0.6, ls=(0, (1, 3)), zorder=1)
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
        for c in controls:
            top.plot(t, idx(c), color=GREY, lw=1.0, zorder=2)
            ends.append([idx(c)[-1], c])
        cm = 100 * np.exp(np.log(d[controls].to_numpy()).mean(axis=1) - np.log(base[controls].to_numpy()).mean())
        top.plot(t, cm, color=INK2, lw=1.4, ls="--", zorder=3)
        top.plot(t, idx(TREATED), color=BLUE, lw=2.0, zorder=4)
        ends.append([idx(TREATED)[-1], TREATED])
        ends.sort()
        span = np.ptp(top.get_ylim())
        for k in range(1, len(ends)):
            if ends[k][0] - ends[k - 1][0] < 0.04 * span:
                ends[k][0] = ends[k - 1][0] + 0.04 * span
        for y, c in ends:
            tr = c == TREATED
            top.annotate(c, (t.iloc[-1], y), xytext=(4, 0), textcoords="offset points", va="center",
                         fontsize=7.5 if tr else 7, color=INK if tr else INK2, fontweight="bold" if tr else "normal")
        top.set_title(SERIES[s], fontsize=9.5, color=INK, loc="left")
        upto = (t <= EVENT[1]).to_numpy()
        bot.plot(t[upto], d["gap_trend_lp"].to_numpy()[upto], color=GREY, lw=1.0, ls="--", zorder=2)
        bot.plot(t, d["gap_vs_controls_nobb_lp"], color=BLUE, lw=1.0, ls=(0, (3, 2)), zorder=3)
        bot.plot(t, d["gap_vs_controls_lp"], color=BLUE, lw=2.0, marker="o", ms=3, zorder=4)
        bot.axhline(0, color=INK2, lw=0.6)
        bot.xaxis.set_major_locator(mdates.YearLocator())
        bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        top.set_xlim(t.iloc[0], t.iloc[-1] + pd.Timedelta(days=170))
    axes[0, 0].set_ylabel("Index, 2022-04 = 100", fontsize=8.5, color=INK2)
    axes[1, 0].set_ylabel("Santander minus controls, log points\n(2022-04 = 0; dashed grey: 2021-01..2022-04 trend)",
                          fontsize=8.5, color=INK2)
    axes[0, 0].annotate("Default sweep\n(from 2022-05)", (EVENT[0], axes[0, 0].get_ylim()[1]), xytext=(3, -2),
                        textcoords="offset points", va="top", fontsize=7, color=INK2)
    handles = [plt.Line2D([], [], color=BLUE, lw=2.0, label="Santander minus BB, Itaú, Bradesco, Caixa (bottom row)"),
               plt.Line2D([], [], color=BLUE, lw=1.0, ls=(0, (3, 2)), label="Santander minus Itaú, Bradesco, Caixa"),
               plt.Line2D([], [], color=INK2, lw=1.4, ls="--", label="Control mean (geometric)"),
               plt.Line2D([], [], color=GREY, lw=1.0, label="Controls")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.995), ncol=4, frameon=False, fontsize=8,
               labelcolor=INK)
    fig.suptitle("Santander against four large banks: Cosif month-end balances, 2021-01 to 2023-12", x=0.06, y=1.045,
                 ha="left", fontsize=11, color=INK)
    fig.text(0.06, 0.005, "Source: BCB Cosif balancetes (DOC 4010), individual institution, via panel_cosif_extract.py. "
             "Shaded: 2022-05 to 2022-08. Fine dotted line: 2022-10, the first month-end under the 2022-09-30 tiers.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def check3(cosif_long):
    cols = ["CodConglomeradoPrudencial", "year", "quarter", "dep_a4", "rate_a4_national", "risk_free_qoq",
            "spread_ann_a4_national", "rate_a4_guarded"]
    code2bank = {k: b for b, (_, k) in BANKS.items()}
    p = pq.read_table(PANEL, columns=cols, filters=[("CodConglomeradoPrudencial", "in", list(code2bank)),
                                                     ("year", ">=", 2020), ("year", "<=", 2024)]).to_pandas()
    chk = p.groupby(["CodConglomeradoPrudencial", "year", "quarter"])[["rate_a4_national", "risk_free_qoq"]].nunique()
    assert chk.le(1).all().all(), "rate_a4_national or risk_free_qoq varies within a conglomerate-quarter"
    q = (p.groupby(["CodConglomeradoPrudencial", "year", "quarter"])
         .agg(dep_a4=("dep_a4", "sum"), n_markets=("dep_a4", "size"),
              rate_a4_national_qoq=("rate_a4_national", "first"), risk_free_qoq=("risk_free_qoq", "first"),
              spread_ann_a4_national=("spread_ann_a4_national", "first"), rate_a4_guarded=("rate_a4_guarded", "max"))
         .reset_index())
    q["bank"] = q["CodConglomeradoPrudencial"].map(code2bank)
    q["period"] = q["year"].astype(str) + "Q" + q["quarter"].astype(str)
    q["dep_a4_bn"] = q["dep_a4"] / 1e9
    q["rate_a4_national_ann"] = (1 + q["rate_a4_national_qoq"]) ** 4 - 1
    q["risk_free_ann"] = (1 + q["risk_free_qoq"]) ** 4 - 1
    q["rate_share_of_selic"] = q["rate_a4_national_qoq"] / q["risk_free_qoq"]
    controls = [b for b in BANKS if b != TREATED]
    cm = q[q["bank"].isin(controls)].groupby("period")["rate_share_of_selic"].mean()
    cm_nobb = q[q["bank"].isin([b for b in controls if b != "BB"])].groupby("period")["rate_share_of_selic"].mean()
    q["share_minus_control_mean"] = q["rate_share_of_selic"] - q["period"].map(cm)
    q["share_minus_control_mean_nobb"] = q["rate_share_of_selic"] - q["period"].map(cm_nobb)
    q["fallback_rate"] = ((q["rate_share_of_selic"] - 1).abs() < 0.002) | (q["rate_a4_national_qoq"].abs() < 1e-9)
    cs = cosif_long[(cosif_long["series"] == "stk_time") & cosif_long["DATA_BASE"].dt.month.isin([3, 6, 9, 12])].copy()
    cs["period"] = cs["DATA_BASE"].dt.year.astype(str) + "Q" + ((cs["DATA_BASE"].dt.month - 1) // 3 + 1).astype(str)
    q = q.merge(cs[["bank", "period", "bn"]].rename(columns={"bn": "cosif_stk_time_entity_bn"}), on=["bank", "period"], how="left")
    order = ["bank", "CodConglomeradoPrudencial", "period", "year", "quarter", "dep_a4_bn", "n_markets",
             "cosif_stk_time_entity_bn", "rate_a4_national_qoq", "rate_a4_national_ann", "risk_free_qoq", "risk_free_ann",
             "rate_share_of_selic", "share_minus_control_mean", "share_minus_control_mean_nobb", "spread_ann_a4_national",
             "rate_a4_guarded", "fallback_rate"]
    q = q[order].sort_values(["bank", "year", "quarter"]).reset_index(drop=True)
    assert len(q) == 5 * 20, f"expected 100 conglomerate-quarters, got {len(q)}"
    return q


def did(q, col, pre, post, drop_fallback):
    x = q[q["bank"] == TREATED].copy()
    if drop_fallback:
        x = x[~x["fallback_rate"]]
    o = x["year"] * 4 + x["quarter"]
    a = x.loc[(o >= pre[0]) & (o <= pre[1]), col].mean()
    b = x.loc[(o >= post[0]) & (o <= post[1]), col].mean()
    return a, b, b - a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-panel", action="store_true", help="also run check 3 on market_panel.parquet")
    args = ap.parse_args()
    long = load_cosif()
    summ, w = check2(long)
    figure(w, OUT / "fig_santander_sweep_cosif.pdf")
    w.to_csv(OUT / "santander_sweep_cosif_monthly.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 240)
    print(f"Outputs in {OUT}")
    print("\nCHECK 2 (Cosif, monthly). Deviation over 2022-05..08 from the 2021-01..2022-04 trend, log points:")
    print(summ.round(2).to_string(index=False))
    s = w[w["DATA_BASE"].between(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-01"))]
    print("\nSantander month-end balances, R$ bn:")
    print(s.pivot_table(index="DATA_BASE", columns="series", values=TREATED).round(1).to_string())
    if not args.with_panel:
        print("\ncheck 3 not run (needs --with-panel and an idle machine)")
        return
    q = check3(long)
    q.to_csv(OUT / "santander_sweep_booking.csv", index=False, float_format="%.6f")
    print("\nCHECK 3 (panel, quarterly). rate_a4_national as a share of the Selic:")
    print(q.pivot_table(index="period", columns="bank", values="rate_share_of_selic").round(3).to_string())
    print("\nSantander dep_a4 against the Cosif entity stock (R$ bn):")
    print(q[q["bank"] == TREATED].set_index("period")[["dep_a4_bn", "cosif_stk_time_entity_bn", "fallback_rate"]].round(1).to_string())
    o = lambda y, k: y * 4 + k
    for lab, pre, post in (("2020Q1-2022Q1 vs 2022Q2-2024Q4", (o(2020, 1), o(2022, 1)), (o(2022, 2), o(2024, 4))),
                           ("2021Q3-2022Q1 vs 2022Q2-2023Q4", (o(2021, 3), o(2022, 1)), (o(2022, 2), o(2023, 4)))):
        for col in ("share_minus_control_mean", "share_minus_control_mean_nobb"):
            for drop in (False, True):
                a, b, dd = did(q, col, pre, post, drop)
                print(f"{lab} | {col} | {'drop Santander fallback quarters' if drop else 'all quarters'}: "
                      f"pre {a:.3f}, post {b:.3f}, difference {dd:+.3f}")


if __name__ == "__main__":
    main()
