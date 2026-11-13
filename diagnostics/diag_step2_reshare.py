"""
diag_step2_reshare.py
=====================
Step 2 of the market size from measured new money (MAIN_IMPR_CHANNEL.md, A1.26-A1.30, log 11-13 10:27; Pedro's
go): the demand frames re-shared against the awake pool P^B under the decided set, and the local logit on them.
A sandbox run: the frames go to %TEMP%/step2_1113/<variant>/E<k>/demand_prep in the production schema, the logit
(blp_logit.jl, full_dtype, WCB) to .../out. Nothing in production is written.

The decided set (A1.26-A1.29):
  - credit unions (single, Tcb B3S) are in the outside good;
  - the panel's STOCKS are brought to the public's universe by a uniform factor by deposit type and quarter on the
    comparable institutions, f_kt = (aggregate_k - B3S_k) / panel_k (types 1, 2, 4; type 5 keeps 1), and (11-B) is
    recomputed: Dep_Act = max{0, f_t D_t - phi R f_t-1 D_t-1};
  - booking moves inside a bank are netted symmetrically (diag_new_money.netting);
  - the main pool P^B (diag_new_money.national_universe with the credit unions as outside components).
The local pool of B market m: P_mt = sum over its B cells of max{0, A_c - l_c} + a_mt X_t, with A_c = max{(1 - min(phi, c))
C, C - D}, l_c the netted loss, X_t = P^B_t - sum over all B cells of A_c, and a_mt the allocation rule: L1 population x
access, or L3 the mean of L1 and the lagged local booked stock.
Local floor (the analogue of production's TOTAL_SHARE_CAP): P'_mt = max{P_mt, ACT_mt / 0.95}, ACT_mt the local active
balances over every B cell; the markets where it binds are counted.
The shares: s_j = Dep_Act'_j / P'_mt on B rows; s_j = Dep_Act'_j / P'_t on D rows, P'_t = sum_m P'_mt (the D firms
compete for the national pool). blp_logit.jl aggregates D shares with weights proportional to the per-market mean of
banked_correction x pop_total over the B rows, so B rows get banked_correction = P'_mt / pop_total (pop_total kept;
where pop_total <= 0, pop_total = 1 and banked_correction = P'_mt); then w_mt = P'_mt / P'_t and the closed form's
G_t = 1 - (sum of active balances) / P'_t, the national outside share. M_mt and M_nat hold P'_mt and P'_t.

Variants (per routine E3, E4):
  src                 the source frame written unchanged (gate S: the logit reproduces the source's)
  L{1,3}_c{099,0999}_{tv,fz}   rule, cap on phi inside A, factor time-varying or frozen at its 2016-24 mean by type
  t5zero_L1_c099_tv   the base with the type-5 spread set to zero (type 5 kept; its spread out of the price term)
Source rows: the fix agent's full-prep frames under both fixes (%TEMP%/dmkt_fix_1112/phi_1112/both/demand_prep). phi-hat is
read from those frames (phi_mt on B rows, phi_t on D rows; the national phi_t over the B markets is recomputed from
phi_mt and compared with est{k}/national_phi_t.csv). After the sleepiness re-estimation the run is repeated with
--src-dp <the new prep's demand_prep> --src-logit <its logit summary> --sandbox <a new folder>.

Steps: --step frames,logit,summary   Usage: python diagnostics/diag_step2_reshare.py --step frames --routines 3,4
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import paths
from utils import macro_rates
import diag_new_money as dnm
import diag_market_size_alt as alt
import diag_cap_sensitivity as dcs
import diag_market_size_a as msa

REPO = Path(__file__).resolve().parents[1]
SRC_DP = Path(r"C:\Users\pedro\AppData\Local\Temp\dmkt_fix_1112\phi_1112\both\demand_prep")
SRC_LOGIT = Path(r"C:\Users\pedro\AppData\Local\Temp\dmkt_fix_1112\phi_1112\both\out\logit\logit_summary_spec_12.json")
SANDBOX = Path(r"C:\Users\pedro\AppData\Local\Temp\step2_1113")
OUT = paths.rout_dir() / "diag_step2_reshare"
FRAMES = dcs.FRAMES
BANK = dnm.BANK
SHARE_CAP = 0.95
GRID = [(rule, cap, fac) for rule in ("L1", "L3") for cap in ("0.99", "0.999") for fac in ("tv", "fz")]
BASE_VARIANTS = ["src"] + [f"{r}_c{c.replace('0.', '0')}_{f}" for r, c, f in GRID] + ["t5zero_L1_c099_tv"]
# The L1-L3 decomposition (main thread, 11-13 10:48): both frames without the B rows of the market-quarters whose
# inside share of the pool is above 0.80 under L1 or under L3 (the union, so the two samples are the same)
EXTRA_VARIANTS = ["L1_c099_tv_drop80u", "L3_c099_tv_drop80u"]
VARIANTS = BASE_VARIANTS + EXTRA_VARIANTS


def vcfg(v):
    if v == "src":
        return None
    t5 = v.startswith("t5zero_")
    core = v[len("t5zero_"):] if t5 else v
    rule, c, fac = core.split("_")
    cap = {"c099": "0.99", "c0999": "0.999"}[c]
    return {"rule": rule, "cap": cap, "factor": fac, "t5zero": t5}


def read_src(k):
    f = SRC_DP / FRAMES[k]
    tab = pq.read_table(f)
    d = tab.to_pandas()
    d[BANK] = d[BANK].astype(str)
    d["mca_code"] = d["mca_code"].astype(str)
    d["is_B"] = d["is_B"].astype(bool)
    d["k"] = d["deposit_type"].astype(int)
    d["q"] = d["time_id"].str[:4].astype(int) * 4 + d["time_id"].str[-1].astype(int) - 1
    return d, tab.schema, {"path": str(f), "md5": dnm.md5_file(f), "rows": int(len(d))}


def berry(df):
    """blp_logit.jl berry_inversion's closed form (l.360-429), in numpy: delta per row, G_t, K_t."""
    isb = df["is_B"].to_numpy(bool)
    t = df["time_id"].to_numpy()
    sB = df["share_B_cond"].to_numpy(float)
    sD = df["share_D"].to_numpy(float)
    bcpop = df["banked_correction"].fillna(1.1).to_numpy(float) * df["pop_total"].fillna(0.0).to_numpy(float)
    b = pd.DataFrame({"m": df["mca_code"].to_numpy()[isb], "t": t[isb], "s": np.nan_to_num(sB[isb]), "w": bcpop[isb]})
    pr = b.groupby(["m", "t"]).agg(SB=("s", "sum"), w=("w", "mean")).reset_index()
    pr["w"] = pr["w"] / pr.groupby("t")["w"].transform("sum")
    K = (pr["w"] * (1 - pr["SB"])).groupby(pr["t"]).sum()
    SD = pd.Series(np.nan_to_num(sD[~isb]), index=t[~isb]).groupby(level=0).sum().reindex(K.index).fillna(0.0)
    G = K - SD
    pr["g"] = ((1 - pr["SB"]) * pr["t"].map(G) / pr["t"].map(K)).clip(1e-12, 1.0)
    gp = pd.Series(pr["g"].to_numpy(), index=pd.MultiIndex.from_frame(pr[["m", "t"]]))
    delta = np.full(len(df), np.nan)
    gB = gp.reindex(pd.MultiIndex.from_arrays([df["mca_code"].to_numpy()[isb], t[isb]])).to_numpy()
    delta[isb] = np.log(np.clip(sB[isb], 1e-15, None)) - np.log(gB)
    delta[~isb] = np.log(np.clip(sD[~isb], 1e-15, None)) - np.log(pd.Series(t[~isb]).map(G).to_numpy())
    return delta, G, K


def step_frames(routines):
    SANDBOX.mkdir(parents=True, exist_ok=True)
    q_idx = pd.Index(np.arange(dnm.Q0, dnm.Q1 + 1), name="q")
    dnm.stamp("panel cells")
    cells, mk, prec = dnm.read_cells()
    cells = cells[~((cells["k"] == 5) & (cells["q"] <= dnm.K5_FIRST_FLOW_Q - 1))].reset_index(drop=True)
    macro = macro_rates.make_quarterly_macro(macro_rates.load_macro_series())
    macro["q"] = (macro["AnoMes"] // 100) * 4 + (macro["AnoMes"] % 100) // 3 - 1
    rate = macro.set_index("q")
    agg, agg_info = dnm.read_aggregates(dnm.AGG_DIR)
    inst = dnm.ifdata_by_institution()
    cdiR = pd.Series(1 + rate["cdi_qoq"], index=rate.index).reindex(range(dnm.Q0 - 1, dnm.Q1 + 1))
    selR = pd.Series(1 + rate["selic_qoq"], index=rate.index).reindex(range(dnm.Q0 - 1, dnm.Q1 + 1))
    savR = pd.Series(1 + rate["savings_rate_qoq"], index=rate.index).reindex(range(dnm.Q0 - 1, dnm.Q1 + 1))
    zero = pd.Series(1.0, index=range(dnm.Q0 - 1, dnm.Q1 + 1))
    ret = {"zero": zero, "cdi": cdiR, "selic": selR}
    out_comps = {dnm.AGG[cc][0]: (agg[dnm.AGG[cc][0]], ret[dnm.AGG[cc][2]]) for cc in dnm.AGG if dnm.AGG[cc][1] == "outside"}
    acc = mk.copy()
    acc["year"] = acc["q"] // 4
    acc["acc"] = (acc["gdppc"] / acc.groupby("year")["gdppc"].transform("median")).clip(0.35, 2.5).fillna(1.0)
    acc["L1w"] = acc["pop"].fillna(0.0) * acc["acc"]
    for k in routines:
        t0 = time.time()
        d, schema0, sinfo = read_src(k)
        if schema0.metadata[b"demand_chars.market_panel_md5"].decode() != prec["panel_md5"]:
            raise SystemExit("source frame's panel is not market_panel.parquet")
        fr = d[[BANK, "mca_code", "k", "q", "is_B", "deposit_balance", "lagged_deposits", "gross_return_lag", "phi_mt",
                "phi_t", "Dep_Act", "pop_total"]].copy()
        phiN, pinfo = dnm.national_phi(fr, k)
        c = dnm.build_cells(k, fr, cells, phiN)
        g = dnm.gates_cells(c)
        res = {"source": sinfo, "panel": prec, "phi": pinfo, "gates_cells": g, "aggregates": agg_info, "variants": {}}
        res["phi_t_B_minus_frame_phi_t_max_abs"] = float((phiN["phi_t_B"] - phiN["phi_t_stored"]).abs().max())
        # credit unions (B3S) by type, the comparable factors (time-varying and frozen)
        codes = set(c[BANK].astype(str))
        nip = inst[~inst["CodInst"].isin(codes) & (inst["Tcb"] == "B3S") & inst["Conta"].isin(["78282", "78283", "78286"])]
        cu = nip.pivot_table(index="q", columns="Conta", values="v", aggfunc="sum").reindex(range(dnm.Q0 - 1, dnm.Q1 + 1)).fillna(0.0)
        cu = {kk: (cu[a] if a in cu.columns else pd.Series(0.0, index=cu.index)) for kk, a in ((1, "78282"), (2, "78283"), (4, "78286"))}
        tgt = {1: agg["demand"] - cu[1].reindex(agg.index).fillna(0.0), 2: agg["savings"] - cu[2].reindex(agg.index).fillna(0.0),
               4: agg["time"] - cu[4].reindex(agg.index).fillna(0.0)}
        fac_tv = dnm.factors(c, tgt, q_idx)
        fmean = fac_tv.groupby("k")["f"].mean()
        fac_fz = fac_tv.assign(f=fac_tv["k"].map(fmean), f_lag=fac_tv["k"].map(fmean))
        res["factor_mean_by_type"] = {int(kk): float(v) for kk, v in fmean.items()}
        res["factor_tv_range_by_type"] = {int(kk): [float(gg["f"].min()), float(gg["f"].max())] for kk, gg in fac_tv.groupby("k")}
        comps = {**out_comps, "cu_demand": (cu[1], zero), "cu_savings": (cu[2], savR), "cu_time": (cu[4], cdiR)}
        # the source frame written unchanged: gate S
        alt.write_frame(d.drop(columns=["k", "q"]), schema0, SANDBOX / "src" / f"E{k}" / "demand_prep" / FRAMES[k],
                        {"variant": "src", "source_md5": sinfo["md5"]})
        delta_src, G_src, _ = berry(d)
        key = [BANK, "mca_code", "k", "q"]
        views = {}
        for fac_lab, fac in (("tv", fac_tv), ("fz", fac_fz)):
            v = dnm.cell_view(c, fac)
            R = dnm.national_universe(v, comps, phiN["phi_t_B"], q_idx)
            gc, lc, Bq = dnm.netting(v)
            views[fac_lab] = (v, R, gc, lc, Bq)
        for vname in BASE_VARIANTS[1:]:
            t = time.time()
            cf = vcfg(vname)
            v, R, gc, lc, Bq = views[cf["factor"]]
            cap = cf["cap"]
            b = v["is_B"].to_numpy(bool)
            DA = np.maximum(0.0, v["Dep_Act"].to_numpy() - gc)
            A = np.maximum(0.0, v[f"A_{cap}"].to_numpy() - lc)
            cb = pd.DataFrame({"mca_code": v["mca_code"].to_numpy()[b], "q": v["q"].to_numpy()[b], "bank": v[BANK].to_numpy()[b],
                               "DA": DA[b], "A": A[b], "C": v["C"].to_numpy()[b], "D_lag": v["D_lag"].to_numpy()[b]})
            m = cb.groupby(["mca_code", "q"]).agg(ACT=("DA", "sum"), AWK=("A", "sum"), S_lag=("D_lag", "sum")).reset_index()
            m = m.merge(acc[["mca_code", "q", "L1w"]], on=["mca_code", "q"], how="left")
            m["L1w"] = m["L1w"].fillna(0.0)
            Xt = R[f"PB_{cap}"] - R[f"A_B_{cap}"]
            Dawk = R[f"A_in_{cap}"] - R[f"A_B_{cap}"]
            m["X"] = m["q"].map(Xt)
            m["Dawk"] = m["q"].map(Dawk)
            w1 = m["L1w"] / m.groupby("q")["L1w"].transform("sum")
            w2 = m["S_lag"] / m.groupby("q")["S_lag"].transform("sum")
            m["a"] = w1 if cf["rule"] == "L1" else 0.5 * (w1 + w2)
            m["P"] = m["AWK"] + m["a"] * m["X"]
            m["P_floor"] = np.maximum(m["P"], m["ACT"] / SHARE_CAP)
            Pnat = m.groupby("q")["P_floor"].sum()
            # rows: the source frame's, with the re-computed and netted active balance
            vv = v[key].assign(DA_new=DA)
            y = d.merge(vv, on=key, how="left")
            if len(y) != len(d):
                raise SystemExit("row merge changed the row count")
            miss = int(y["DA_new"].isna().sum())
            keep = y["DA_new"].fillna(0.0) > 1e-6
            r_ = {"cfg": cf, "rows_source": int(len(d)), "rows_without_cell": miss, "rows_dropped_zero_active": int((~keep).sum())}
            y = y[keep].copy()
            pk = pd.MultiIndex.from_frame(m[["mca_code", "q"]])
            Pm = pd.Series(m["P_floor"].to_numpy(), index=pk)
            yb = y["is_B"].to_numpy(bool)
            Pm_row = Pm.reindex(pd.MultiIndex.from_arrays([y["mca_code"].to_numpy(), y["q"].to_numpy()])).to_numpy()
            y["Dep_Act"] = y["DA_new"]
            y["share_B_cond"] = np.where(yb, y["Dep_Act"] / Pm_row, np.nan)
            y["share_D"] = np.where(~yb, y["Dep_Act"] / y["q"].map(Pnat).to_numpy(), np.nan)
            y["M_mt"] = np.where(yb, Pm_row, np.nan)
            y["M_nat"] = y["q"].map(Pnat).to_numpy()
            popv = y["pop_total"].to_numpy(float)
            nopop = yb & ~(popv > 0)
            y.loc[nopop, "pop_total"] = 1.0
            y["banked_correction"] = np.where(yb, Pm_row / y["pop_total"].to_numpy(float), y["banked_correction"])
            r_["B_rows_pop_reset"] = int(nopop.sum())
            if cf["t5zero"]:
                t5 = y["k"] == 5
                y.loc[t5, "spread_ann"] = 0.0
                y.loc[t5, "spread_qoq"] = 0.0
                r_["t5_rows_spread_zeroed"] = int(t5.sum())
            if not np.isfinite(y.loc[yb, "share_B_cond"]).all() or not np.isfinite(y.loc[~yb, "share_D"]).all():
                raise SystemExit(f"{vname} E{k}: a share is not finite")
            delta, G, K = berry(y)
            r_["G_min"] = float(G.min())
            r_["G_nonpositive_quarters"] = int((G <= 0).sum())
            r_["G_t_quantiles"] = [float(G.quantile(qq)) for qq in (0.0, 0.5, 1.0)]
            # the tail: |delta_new - delta_old| > 10 on the common rows (logit side; ext1's needs the RC rerun)
            dd = pd.DataFrame({"i": y.index.to_numpy(), "dn": delta})
            ds = pd.Series(delta_src, index=d.index)
            dd["do"] = ds.reindex(dd["i"]).to_numpy()
            diff = (dd["dn"] - dd["do"]).abs()
            r_["tail_share_abs_ddelta_gt_10"] = float((diff > 10).mean())
            r_["tail_share_abs_ddelta_gt_5"] = float((diff > 5).mean())
            r_["ddelta_median"] = float((dd["dn"] - dd["do"]).median())
            # the local counts and shares
            ins = m["ACT"] / m["P"]
            r_["share_mq_ACT_gt_0.80P"] = float(((m["ACT"] > 0.80 * m["P"]) | (m["P"] <= 0)).mean())
            r_["share_mq_ACT_gt_0.95P_floor_binds"] = float(((m["ACT"] > 0.95 * m["P"]) | (m["P"] <= 0)).mean())
            r_["stockw_share_ACT_gt_0.80P"] = float(m.loc[(m["ACT"] > 0.80 * m["P"]) | (m["P"] <= 0), "S_lag"].sum() / m["S_lag"].sum())
            insf = m["ACT"] / m["P_floor"]
            r_["inside_share_median"] = float(insf.median())
            r_["outside_share_median"] = float((1 - insf).median())
            r_["inside_share_stockw_median"] = dnm.wquant(insf, m["S_lag"], 0.5)
            r_["outside_share_stockw_median"] = dnm.wquant(1 - insf, m["S_lag"], 0.5)
            r_["D_share_of_national_pool_median"] = float((y[~yb].groupby("q")["Dep_Act"].sum() / Pnat).median())
            r_["national_outside_share_G_median"] = float(G.median())
            lagpos = (v["D_lag"].to_numpy() > 0) & b
            r_["truncation_share_B_cells"] = float(((DA <= 1e-6) & lagpos).sum() / lagpos.sum())
            r_["truncation_share_B_lagstock"] = float(v["D_lag"].to_numpy()[(DA <= 1e-6) & lagpos].sum() / v["D_lag"].to_numpy()[lagpos].sum())
            r_["netted_share_of_B_carry"] = float(Bq.sum() / v.loc[v["is_B"], "C"].sum())
            # W4 weights at destination bank-markets with a positive active balance (shares of P'_mt)
            jm = cb.groupby(["bank", "mca_code", "q"]).agg(DA=("DA", "sum"), A=("A", "sum"), C=("C", "sum")).reset_index()
            jm = jm[jm["DA"] > 1e-6].merge(m[["mca_code", "q", "AWK", "X", "Dawk", "a", "P_floor"]], on=["mca_code", "q"], how="left")
            jm = jm[jm["P_floor"] > 0]
            own = jm["A"] / jm["P_floor"]
            riv = (jm["AWK"] - jm["A"] + jm["a"] * jm["Dawk"]) / jm["P_floor"]
            new = jm["a"] * (jm["X"] - jm["Dawk"]) / jm["P_floor"]
            wts = jm["C"].to_numpy()
            r_["w4"] = {nm: {"p50": float(np.median(val)), "carryw_p50": dnm.wquant(val, wts, 0.5)}
                        for nm, val in (("own", own.to_numpy()), ("rivals", riv.to_numpy()), ("new", new.to_numpy()))}
            r_["mean_rho_1ms"] = dcs.mean_rho_1ms(y)
            out = y.drop(columns=["k", "q", "DA_new"])
            alt.write_frame(out, schema0, SANDBOX / vname / f"E{k}" / "demand_prep" / FRAMES[k],
                            {"variant": vname, "rule": cf["rule"], "cap": cap, "factor": cf["factor"], "t5zero": cf["t5zero"],
                             "shares": "Dep_Act / P_mt (B), Dep_Act / P_t (D); M_mt, M_nat = the pools; banked_correction = P_mt / pop_total (B)",
                             "source_md5": sinfo["md5"]})
            res["variants"][vname] = r_
            dnm.stamp(f"E{k} {vname}: rows {len(y):,} (dropped {r_['rows_dropped_zero_active']}); G_min {r_['G_min']:.4f} "
                      f"({r_['G_nonpositive_quarters']} q <= 0); ACT>0.80P {r_['share_mq_ACT_gt_0.80P']:.3f}; floor "
                      f"{r_['share_mq_ACT_gt_0.95P_floor_binds']:.3f}; inside p50 {r_['inside_share_median']:.3f}; tail>10 "
                      f"{r_['tail_share_abs_ddelta_gt_10']:.4f}; {time.time() - t:.0f} s")
        res["mean_rho_1ms_src"] = dcs.mean_rho_1ms(d)
        res["seconds"] = round(time.time() - t0, 1)
        (SANDBOX / f"diag_E{k}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        dnm.stamp(f"E{k} frames done in {time.time() - t0:.0f} s")


def step_extras(routines):
    """The L1-L3 decomposition frames: L1_c099_tv and L3_c099_tv without the B rows of the market-quarters whose inside
    share of the pool (the sum of share_B_cond over the frame's B rows) is above 0.80 under L1 or under L3."""
    for k in routines:
        dpath = SANDBOX / f"diag_E{k}.json"
        dj = json.loads(dpath.read_text(encoding="utf-8"))
        fr = {}
        for rule in ("L1", "L3"):
            tab = pq.read_table(SANDBOX / f"{rule}_c099_tv" / f"E{k}" / "demand_prep" / FRAMES[k])
            fr[rule] = (tab.to_pandas(), tab.schema)
        flags = {}
        for rule, (y, _) in fr.items():
            b = y[y["is_B"].astype(bool)]
            SB = b.groupby(["mca_code", "time_id"])["share_B_cond"].sum()
            flags[rule] = set(SB.index[SB > 0.80])
            st = b.groupby(["mca_code", "time_id"])["lagged_deposits"].sum()
            dj.setdefault("decomposition_80", {})[rule] = {
                "mq_inside_share_gt_0.80": int((SB > 0.80).sum()), "mq": int(len(SB)),
                "lagstock_share_gt_0.80": float(st[SB > 0.80].sum() / st.sum())}
        union = flags["L1"] | flags["L3"]
        dj["decomposition_80"]["union_mq"] = len(union)
        dj["decomposition_80"]["L1_only"] = len(flags["L1"] - flags["L3"])
        dj["decomposition_80"]["L3_only"] = len(flags["L3"] - flags["L1"])
        for rule, (y, schema0) in fr.items():
            name = f"{rule}_c099_tv_drop80u"
            isb = y["is_B"].astype(bool).to_numpy()
            mq = pd.MultiIndex.from_arrays([y["mca_code"].astype(str), y["time_id"].astype(str)])
            drop = isb & mq.isin(list(union))
            z = y[~drop]
            delta, G, K = berry(z)
            r_ = {"cfg": {"rule": rule, "cap": "0.99", "factor": "tv", "t5zero": False, "drop80u": True},
                  "rows_source": int(len(y)), "rows_dropped_zero_active": int(drop.sum()), "G_min": float(G.min()),
                  "G_nonpositive_quarters": int((G <= 0).sum()), "national_outside_share_G_median": float(G.median()),
                  "mean_rho_1ms": dcs.mean_rho_1ms(z)}
            dj["variants"][name] = r_
            alt.write_frame(z, schema0, SANDBOX / name / f"E{k}" / "demand_prep" / FRAMES[k],
                            {"variant": name, "rule": rule, "drop": "B rows of mq with inside share > 0.80 under L1 or L3"})
            dnm.stamp(f"E{k} {name}: dropped {int(drop.sum()):,} B rows in {len(union)} market-quarters; G_min {r_['G_min']:.4f}")
        dpath.write_text(json.dumps(dj, indent=1, default=float), encoding="utf-8")


def step_logit(routines, threads, variants):
    jl = shutil.which("julia")
    for k in routines:
        dj = json.loads((SANDBOX / f"diag_E{k}.json").read_text(encoding="utf-8"))
        for v in variants:
            base = SANDBOX / v / f"E{k}"
            dp, od = base / "demand_prep", base / "out"
            fr = dp / FRAMES[k]
            if not fr.exists():
                dnm.stamp(f"logit {v} E{k}: no frame")
                continue
            summ = od / "logit" / "logit_summary_spec_12.json"
            if summ.exists() and summ.stat().st_mtime > fr.stat().st_mtime:
                dnm.stamp(f"logit {v} E{k}: done earlier, skipped")
                continue
            if v != "src" and dj["variants"][v]["G_nonpositive_quarters"]:
                dnm.stamp(f"logit {v} E{k}: G_t <= 0, not run")
                continue
            msa.wait_light()
            t = time.time()
            with open(base / "logit.log", "w", encoding="utf-8") as fh:
                r = subprocess.run([jl, "--project=.", f"--threads={threads}", "blp_logit.jl", "--demand-dir", str(dp),
                                    "--out-dir", str(od)], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=dcs.julia_env())
            dnm.stamp(f"logit {v} E{k}: exit {r.returncode} in {time.time() - t:.0f} s")


def step_summary(routines, out):
    out.mkdir(parents=True, exist_ok=True)
    src = json.loads(SRC_LOGIT.read_text(encoding="utf-8"))
    rows = []
    for k in routines:
        dj = json.loads((SANDBOX / f"diag_E{k}.json").read_text(encoding="utf-8"))
        sref = src[f"E{k}_full_dtype"]
        nm = sref["param_names"]
        ref = {"alpha": float(sref["theta1"][nm.index("alpha")]), "se": float(sref["se"][nm.index("alpha")]),
               "dummy_D": float(sref["theta1"][nm.index("dummy_D_type")])}
        for v in VARIANTS:
            lg = msa.logit_read(SANDBOX / v / f"E{k}" / "out" / "logit" / "logit_summary_spec_12.json", k) \
                if (SANDBOX / v / f"E{k}" / "out" / "logit" / "logit_summary_spec_12.json").exists() else None
            r = {"routine": f"E{k}", "variant": v}
            if v != "src" and v not in dj["variants"]:
                continue                      # not built (the decomposition variants are optional)
            if v == "src":
                r["mean_rho_1ms"] = dj["mean_rho_1ms_src"]
            else:
                dv = dj["variants"][v]
                r.update({kk: dv.get(kk) for kk in ("rows_dropped_zero_active", "G_min", "G_nonpositive_quarters", "share_mq_ACT_gt_0.80P",
                                                "stockw_share_ACT_gt_0.80P", "share_mq_ACT_gt_0.95P_floor_binds", "inside_share_median",
                                                "outside_share_median", "inside_share_stockw_median", "outside_share_stockw_median",
                                                "D_share_of_national_pool_median", "national_outside_share_G_median",
                                                "truncation_share_B_cells", "truncation_share_B_lagstock", "tail_share_abs_ddelta_gt_10",
                                                "tail_share_abs_ddelta_gt_5", "mean_rho_1ms")})
                for comp in ("own", "rivals", "new"):
                    if "w4" in dv:
                        r[f"w4_{comp}_p50"] = dv["w4"][comp]["p50"]
                        r[f"w4_{comp}_carryw_p50"] = dv["w4"][comp]["carryw_p50"]
            if lg:
                r.update({"alpha": lg["alpha"], "se": lg["se"], "p": lg["p"], "ci95_lo": lg["ci95"][0], "ci95_hi": lg["ci95"][1],
                          "dummy_D": lg["dummy_D"], "dummy_D_se": lg["dummy_D_se"], "n_obs": lg["n_obs"], "se_method": lg["se_method"],
                          "elasticity": lg["alpha"] * r["mean_rho_1ms"]})
                if v == "src":
                    r["gate_S_abs_d_alpha"] = abs(lg["alpha"] - ref["alpha"])
                    r["gate_S_abs_d_dummy"] = abs(lg["dummy_D"] - ref["dummy_D"])
                    r["gate_S_pass"] = bool(abs(lg["alpha"] - ref["alpha"]) <= 0.05 * ref["se"])
            rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(out / "step2_summary.csv", index=False, lineterminator="\n")
    files = [out / "step2_summary.csv"]
    for k in routines:
        shutil.copyfile(SANDBOX / f"diag_E{k}.json", out / f"step2_diag_E{k}.json")
        files.append(out / f"step2_diag_E{k}.json")
    with open(out / "MD5SUMS.txt", "w", encoding="utf-8", newline="\n") as fh:
        for p in files:
            fh.write(f"{dnm.md5_file(p, lf=True)}  {p.name}\n")
    dnm.stamp(f"summary: {len(S)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="frames,logit,summary")
    ap.add_argument("--routines", default="3,4")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--lsp-pid", type=int, default=None, help="the editor's Julia language server, ignored by the heavy-process wait")
    ap.add_argument("--src-dp", default=None, help="folder of the source demand frames (phi-hat comes from their phi_mt and "
                    "phi_t columns); default: the fix agent's both-fix frames. After the sleepiness re-estimation, point it "
                    "at the new prep's demand_prep folder")
    ap.add_argument("--src-logit", default=None, help="the source frames' own logit summary, for gate S")
    ap.add_argument("--sandbox", default=None, help="sandbox for the Step 2 frames and fits (default %%TEMP%%/step2_1113)")
    a = ap.parse_args()
    global SRC_DP, SRC_LOGIT, SANDBOX
    if a.src_dp:
        SRC_DP = Path(a.src_dp)
    if a.src_logit:
        SRC_LOGIT = Path(a.src_logit)
    if a.sandbox:
        SANDBOX = Path(a.sandbox)
    if a.lsp_pid:
        msa.LSP_PID = a.lsp_pid
    routines = [int(x) for x in a.routines.split(",")]
    steps = a.step.split(",")
    if "frames" in steps:
        step_frames(routines)
    if "extras" in steps:
        step_extras(routines)
    if "logit" in steps:
        step_logit(routines, a.threads, a.variants.split(","))
    if "summary" in steps:
        step_summary(routines, Path(a.out))


if __name__ == "__main__":
    main()
