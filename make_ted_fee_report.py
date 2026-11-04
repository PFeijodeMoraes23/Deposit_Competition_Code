"""
make_ted_fee_report.py
======================
Task TED1 (coding handoff 2026-11-02, Update 11-02k): the listed PF TED and DOC fees by
conglomerate and quarter, and the source pool's TED fee by B market and quarter. A side analysis:
nothing is merged into market_panel and nothing is regressed on the deposit panel.

Fees. fees/panel_fee_merge.py (listed_fee_institution_quarters, load_tarifa_listed_prices), read
only: listed_fee_ted_pf and listed_fee_doc_pf are the head institution's listed maximum in force
on the quarter's last day, by internet where the conglomerate lists it, else at the branch
(TRANSFER_FEE_CHANNELS); R$ per transfer.

Pool fee, for B destination j in market m and quarter t (rivals j' are the other B conglomerates
with savings in m):
    fbar_jmt = sum_{j' != j} Dep_{j'2m,t-1} f_{j't} / sum_{j' != j} Dep_{j'2m,t-1}
over the rivals with a listed TED fee; rival_cover is their share of the rivals' savings.
  (c) unscaled, R$ per transfer;
  (a) scaled: each rival's fee divided by its national savings per client (savings balance summed
      over markets / N, the CCS client stock of make_ccs_openings_pix.stock(): the complaints
      ranking's count, of the conglomerate or its member institutions, carried by net flows), in
      basis points of that balance, then weighted the same way. Rivals with no ranking count
      drop out of (a); rival_cover_a is the share of rival savings left. N counts every PF and
      PJ client, savers or not.
The market's own pool fee (all B conglomerates, no leave-one-out) is reported beside it.

Report items (Rout/ted_fee_report.json):
  1. the distribution of listed TED fees across conglomerates by year (fourth quarter), B and D,
     the share listing 0.00, and the channel the value comes from;
  2. the cross-sectional correlation of a B conglomerate's TED fee with its time-deposit spread
     (spread_ann_a4, balance-weighted over markets) and with its share of truncated cells
     (spec-12 E3 cells with active balance at or below zero before the (14-B) cut,
     utils.sleep_cells.pre_cut), 2019Q1-2020Q3 means, unweighted and savings-weighted;
  3. the conglomerates whose TED fee went to 0.00 before Pix (a positive fee, then 0.00, by
     2020Q3), with the head institution's vigencia of the zero price;
  4. the pool fee at 2020Q3 across B destination-market cells and across markets, both scalings.

Outputs
  BCB/Tarifas/processed/transfer_fee_listed_conglomerate_quarter.csv   the fee panel, 2016Q1-2024Q4
  Rout/ted_pool_fee_market_quarter.csv                                 B destination x market x quarter
  Rout/ted_fee_report.json

Usage:  python make_ted_fee_report.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import datetime
import json
import logging

import numpy as np
import pandas as pd

from utils import paths as _paths

logging.disable(logging.INFO)

PANEL = _paths.PROCESSED / "market_panel.parquet"
FEE_PANEL = _paths.TARIFAS_PROC / "transfer_fee_listed_conglomerate_quarter.csv"
POOL = _paths.rout_dir() / "ted_pool_fee_market_quarter.csv"
REPORT = _paths.rout_dir() / "ted_fee_report.json"
FIRST_Y, LAST_Y = 2016, 2024
DOSE_Q = (2020, 3)
CORR_WINDOW = ((2019, 1), (2020, 3))
CHANNEL_COLS = ["listed_fee_ted_internet_pf", "listed_fee_ted_branch_pf", "listed_fee_doc_internet_pf",
                "listed_fee_doc_branch_pf"]


def qi(y, q):
    return np.asarray(y) * 4 + np.asarray(q) - 1


def fee_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    from fees.panel_fee_merge import listed_fee_institution_quarters, load_tarifa_listed_prices
    inst = listed_fee_institution_quarters()
    w = load_tarifa_listed_prices(inst)
    w = w[w["year"].between(FIRST_Y, LAST_Y)].copy()
    cols = (["cod_cong_prudencial", "year", "quarter", "listed_fee_ted_pf", "listed_fee_doc_pf"]
            + CHANNEL_COLS + [c.replace("listed_fee_", "listed_fee_n_inst_") for c in CHANNEL_COLS])
    w = w[cols]
    w["ted_channel"] = np.where(w["listed_fee_ted_internet_pf"].notna(), "internet",
                                np.where(w["listed_fee_ted_branch_pf"].notna(), "branch", ""))
    w["doc_channel"] = np.where(w["listed_fee_doc_internet_pf"].notna(), "internet",
                                np.where(w["listed_fee_doc_branch_pf"].notna(), "branch", ""))
    w = w[(w["ted_channel"] != "") | (w["doc_channel"] != "")]
    return w.rename(columns={"cod_cong_prudencial": "conglomerate"}), inst


def panel_balances() -> pd.DataFrame:
    p = pd.read_parquet(PANEL, columns=["CodConglomeradoPrudencial", "NomeInstituicao", "mca_code", "year",
                                        "quarter", "is_B", "dep_a2", "dep_a4", "spread_ann_a4"])
    p = p[p["year"].between(FIRST_Y - 1, LAST_Y)].rename(columns={"CodConglomeradoPrudencial": "conglomerate"})
    p["mca_code"] = p["mca_code"].astype(str)
    return p


def clients_by_quarter() -> pd.Series:
    """N at each quarter's last month, by conglomerate (make_ccs_openings_pix.stock), where a
    ranking count anchors it; flows cumulated from 2016 are not a stock and are left out."""
    from make_ccs_openings_pix import flows_monthly, stock
    s = stock(flows_monthly())
    s = s[s["month"].dt.month.isin([3, 6, 9, 12]) & s["N_source"].str.startswith("ranking")]
    return s.set_index(["conglomerate", s["month"].dt.year.rename("year"),
                        s["month"].dt.quarter.rename("quarter")])["N"]


def pool_fees(p: pd.DataFrame, fees: pd.DataFrame, N: pd.Series) -> pd.DataFrame:
    b = p[p["is_B"] == 1].groupby(["conglomerate", "mca_code", "year", "quarter"], as_index=False)["dep_a2"].sum()
    b["t"] = qi(b["year"], b["quarter"])
    lag = b[["conglomerate", "mca_code", "t", "dep_a2"]].assign(t=lambda x: x["t"] + 1).rename(
        columns={"dep_a2": "w"})
    d = b.merge(lag, on=["conglomerate", "mca_code", "t"], how="left").fillna({"w": 0.0})
    d = d[d["year"].between(FIRST_Y, LAST_Y)]
    nat = p[p["is_B"] == 1].groupby(["conglomerate", "year", "quarter"])["dep_a2"].sum()
    f = fees.set_index(["conglomerate", "year", "quarter"])["listed_fee_ted_pf"]
    d = d.join(f.rename("f"), on=["conglomerate", "year", "quarter"])
    d = d.join(nat.rename("sav_nat"), on=["conglomerate", "year", "quarter"])
    d = d.join(N.rename("N"), on=["conglomerate", "year", "quarter"])
    per_client = d["sav_nat"] / d["N"].where(d["N"] > 0)
    d["f_bp"] = d["f"] / per_client.where(per_client > 0) * 1e4
    out = []
    for lab, col in (("c", "f"), ("a", "f_bp")):
        has = d[col].notna() & (d["w"] > 0)
        d["_wf"] = np.where(has, d["w"] * d[col].fillna(0), 0.0)
        d["_wh"] = np.where(has, d["w"], 0.0)
        g = d.groupby(["mca_code", "t"])
        S, W, Wall = g["_wf"].transform("sum"), g["_wh"].transform("sum"), g["w"].transform("sum")
        num, den = S - d["_wf"], W - d["_wh"]
        d[f"pool_{lab}"] = np.where(den > 0, num / den.where(den > 0), np.nan)
        d[f"market_{lab}"] = np.where(W > 0, S / W.where(W > 0), np.nan)
        d[f"rival_cover_{lab}"] = np.where((Wall - d["w"]) > 0, den / (Wall - d["w"]).where((Wall - d["w"]) > 0), np.nan)
    d = d.drop(columns=["_wf", "_wh", "t"])
    return d.rename(columns={"dep_a2": "dep_a2_t", "w": "dep_a2_lag", "f": "ted_fee", "f_bp": "ted_fee_bp_per_client_savings",
                             "pool_c": "pool_fee_brl", "pool_a": "pool_fee_bp", "market_c": "market_fee_brl",
                             "market_a": "market_fee_bp"})


def dist(s: pd.Series, w: pd.Series | None = None) -> dict:
    s = s.dropna()
    out = {"n": int(len(s)), "mean": round(float(s.mean()), 2), "p10": round(float(s.quantile(.1)), 2),
           "median": round(float(s.median()), 2), "p90": round(float(s.quantile(.9)), 2),
           "max": round(float(s.max()), 2), "share_zero": round(float((s == 0).mean()), 3)}
    if w is not None:
        ww = w.reindex(s.index).fillna(0)
        if ww.sum() > 0:
            out["savings_weighted_mean"] = round(float(np.average(s, weights=ww)), 2)
    return out


def main():
    fees, inst = fee_panel()
    FEE_PANEL.parent.mkdir(parents=True, exist_ok=True)
    fees.to_csv(FEE_PANEL, index=False)
    p = panel_balances()
    isb = p.groupby("conglomerate")["is_B"].first()
    names = p.groupby("conglomerate")["NomeInstituicao"].last()
    fees["is_B"] = fees["conglomerate"].map(isb)
    N = clients_by_quarter()

    rep = {"fee_panel": {"rows": int(len(fees)), "conglomerates": int(fees["conglomerate"].nunique()),
                         "in_market_panel": int(fees["conglomerate"].isin(isb.index).sum())}}

    # 1. distribution by year (fourth quarter), B and D
    sav = p[p["is_B"] == 1].groupby(["conglomerate", "year", "quarter"])["dep_a2"].sum()
    rep["distribution_q4"] = {}
    for y in range(FIRST_Y, LAST_Y + 1):
        f = fees[(fees["year"] == y) & (fees["quarter"] == 4)].set_index("conglomerate")
        row = {}
        for lab, s in (("B", f[f["is_B"] == 1]), ("D", f[f["is_B"] == 0]), ("not_in_panel", f[f["is_B"].isna()])):
            w = sav.xs((y, 4), level=("year", "quarter")) if lab == "B" else None
            row[lab] = dist(s["listed_fee_ted_pf"], w)
            row[lab]["channel"] = s["ted_channel"].value_counts().to_dict()
        rep["distribution_q4"][str(y)] = row

    # 2. correlations, B, 2019Q1-2020Q3 means
    lo, hi = qi(*CORR_WINDOW[0]), qi(*CORR_WINDOW[1])
    pb = p[(p["is_B"] == 1) & (qi(p["year"], p["quarter"]) >= lo) & (qi(p["year"], p["quarter"]) <= hi)].copy()
    pb["sw"] = pb["spread_ann_a4"] * pb["dep_a4"]
    spr = pb.groupby("conglomerate")["sw"].sum() / pb[pb["spread_ann_a4"].notna()].groupby("conglomerate")["dep_a4"].sum()
    from utils.sleep_cells import pre_cut
    cells, _u, _k = pre_cut(3, False)
    c = cells[cells["is_B"].astype(bool)]
    c = c[(qi(c["year"], c["quarter"]) >= lo) & (qi(c["year"], c["quarter"]) <= hi)]
    trunc = (c["active_raw"] <= 1e-6).groupby(c["CodConglomeradoPrudencial"]).mean()
    trunc2 = (c.loc[c["deposit_type"] == 2, "active_raw"] <= 1e-6).groupby(
        c.loc[c["deposit_type"] == 2, "CodConglomeradoPrudencial"]).mean()
    fw = fees[(qi(fees["year"], fees["quarter"]) >= lo) & (qi(fees["year"], fees["quarter"]) <= hi)
              & (fees["is_B"] == 1)].groupby("conglomerate")["listed_fee_ted_pf"].mean()
    savw = pb.groupby("conglomerate")["dep_a2"].sum()
    x = pd.DataFrame({"fee": fw, "spread_a4": spr, "trunc_all": trunc, "trunc_savings": trunc2, "sav": savw})
    corr = {}
    for v in ("spread_a4", "trunc_all", "trunc_savings"):
        z = x[["fee", v, "sav"]].dropna()
        cw = np.cov(z["fee"], z[v], aweights=z["sav"]) if z["sav"].sum() > 0 else None
        corr[v] = {"n": int(len(z)), "pearson": round(float(z["fee"].corr(z[v])), 3),
                   "spearman": round(float(z["fee"].corr(z[v], method="spearman")), 3),
                   "savings_weighted": None if cw is None else round(float(cw[0, 1] / np.sqrt(cw[0, 0] * cw[1, 1])), 3)}
    rep["correlations_B_2019Q1_2020Q3"] = corr
    rep["correlation_inputs_B"] = x.assign(name=x.index.map(names)).round(4).reset_index().rename(
        columns={"index": "conglomerate"}).to_dict("records")

    # 3. TED fee to 0.00 before Pix
    f = fees.sort_values(["conglomerate", "year", "quarter"]).copy()
    f["t"] = qi(f["year"], f["quarter"])
    pre = f[f["t"] <= qi(*DOSE_Q)]
    zero = []
    for cg, g in pre.groupby("conglomerate"):
        s = g.set_index("t")["listed_fee_ted_pf"]
        pos = s[s > 0]
        if pos.empty:
            continue
        after = s[(s.index > pos.index.min()) & (s == 0)]
        if after.empty:
            continue
        t0 = int(after.index.min())
        y0, q0 = t0 // 4, t0 % 4 + 1
        head = inst[(inst["cod_cong_prudencial"] == cg) & (inst["year"] == y0) & (inst["quarter"] == q0)
                    & inst["codigo_servico"].isin(["1315", "1313"]) & (inst["price"] == 0)]
        zero.append({"conglomerate": cg, "name": names.get(cg, ""), "is_B": None if pd.isna(isb.get(cg)) else int(isb.get(cg)),
                     "first_zero_quarter": f"{y0}Q{q0}", "last_positive_fee": float(pos.iloc[-1]),
                     "zero_vigencia": sorted(str(v.date()) for v in head["vigencia"].dropna().unique()),
                     "back_above_zero_by_2020Q3": bool((s[s.index > t0] > 0).any())})
    rep["ted_to_zero_before_pix"] = zero
    always0 = pre.groupby("conglomerate")["listed_fee_ted_pf"].apply(lambda s: bool(len(s.dropna()) and (s.dropna() == 0).all()))
    rep["ted_zero_throughout_2016_2020Q3"] = {"n": int(always0.sum()),
                                              "B": sorted(c for c in always0.index[always0] if isb.get(c) == 1),
                                              "D_n": int(sum(isb.get(c) == 0 for c in always0.index[always0]))}
    changes = pre.assign(prev=pre.groupby("conglomerate")["listed_fee_ted_pf"].shift())
    ch = changes[changes["prev"].notna() & changes["listed_fee_ted_pf"].notna()
                 & (changes["prev"] != changes["listed_fee_ted_pf"])]
    rep["ted_changes_2016_2020Q3"] = {"n_changes": int(len(ch)), "conglomerates": int(ch["conglomerate"].nunique()),
                                      "B_conglomerates": int(ch.loc[ch["conglomerate"].map(isb) == 1, "conglomerate"].nunique()),
                                      "increases": int((ch["listed_fee_ted_pf"] > ch["prev"]).sum()),
                                      "decreases": int((ch["listed_fee_ted_pf"] < ch["prev"]).sum())}

    # 4. pool fee
    pool = pool_fees(p, fees, N)
    pool.to_csv(POOL, index=False)
    d0 = pool[(pool["year"] == DOSE_Q[0]) & (pool["quarter"] == DOSE_Q[1])]
    mk = d0.drop_duplicates("mca_code")
    rep["pool_fee_2020Q3"] = {
        "cells": {"brl": dist(d0["pool_fee_brl"]), "bp": dist(d0["pool_fee_bp"]),
                  "rival_cover_brl_mean": round(float(d0["rival_cover_c"].mean()), 3),
                  "rival_cover_bp_mean": round(float(d0["rival_cover_a"].mean()), 3)},
        "markets": {"brl": dist(mk["market_fee_brl"]), "bp": dist(mk["market_fee_bp"]),
                    "sd_brl": round(float(mk["market_fee_brl"].std()), 3),
                    "sd_bp": round(float(mk["market_fee_bp"].std()), 3)},
        "within_market_sd_brl": round(float(d0.groupby("mca_code")["pool_fee_brl"].std().mean()), 3),
        "n_markets": int(mk["mca_code"].nunique()), "n_cells": int(len(d0)),
    }
    s = pd.DataFrame({"sav_nat": pool.drop_duplicates(["conglomerate", "year", "quarter"]).set_index(
        ["conglomerate", "year", "quarter"])["sav_nat"]})
    s = s.join(N.rename("N"))
    s = s.xs(DOSE_Q, level=("year", "quarter")).dropna()
    s = s[(s["sav_nat"] > 0) & (s["N"] > 0)]
    s["sav_per_client"] = s["sav_nat"] / s["N"]
    s["ted_fee"] = fees.set_index(["conglomerate", "year", "quarter"])["listed_fee_ted_pf"].xs(
        DOSE_Q, level=("year", "quarter")).reindex(s.index)
    s["ted_fee_bp"] = s["ted_fee"] / s["sav_per_client"] * 1e4
    s["name"] = s.index.map(names)
    rep["savings_per_client_2020Q3_B"] = {
        "n_savings_holders_with_ranking_N": int(len(s)),
        "median_brl": round(float(s["sav_per_client"].median()), 1),
        "by_conglomerate": s.sort_values("sav_nat", ascending=False).round(2).reset_index().to_dict("records")}
    rep["_meta"] = {"written": datetime.datetime.now().isoformat(timespec="seconds"), "dose_quarter": f"{DOSE_Q[0]}Q{DOSE_Q[1]}",
                    "fee_panel": str(FEE_PANEL), "pool_panel": str(POOL)}
    REPORT.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("fee_panel", "correlations_B_2019Q1_2020Q3", "ted_changes_2016_2020Q3",
                                          "pool_fee_2020Q3", "savings_per_client_2020Q3_B")}, indent=1, default=str))
    print(f"wrote {FEE_PANEL}\nwrote {POOL}\nwrote {REPORT}")


if __name__ == "__main__":
    main()
