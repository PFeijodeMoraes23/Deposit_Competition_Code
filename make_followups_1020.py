"""
make_followups_1020.py
======================
Numbers, a table and figure inputs for the follow-ups of the 2026-10-20 brief (IO Tea and advisor
meeting). One sidecar, Rout/followups_1020.json, read by make_paper_numbers.py and
make_slide_figures.py; one table, Drafts/Deposit Competition/tab_negative_active.tex.

  C1  the sleepy share phi-hat_mt (routines E3, E4, specification 12): order-free (Shapley)
      split of its variance across MCAs and quarters, with population and lagged-deposit
      weights, with and without the cells at the ceiling phi-hat = 1; the covariance split of
      the native index Upsilon-hat'S_mt by state.
  C3  cells (14-B) truncates at zero (Dep < phi-hat g Dep_lag): shares of cells and of balances,
      by deposit type, firm type, market-size quintile and year; among rows with phi-hat g >= 1;
      and the market-quarters where phi-hat exceeds min_j (or the 10th percentile of) Dep / (g
      Dep_lag) across B rows.
  C5  the change in the D firms' national deposit share, 2016 to 2024, split into a deposit-type
      mix part and a within-type part (two-fold, 2016 and 2024 weights).

phi-hat, the index and every cell before the (14-B) cut come from the prep's own function
(sleep_demand_prep_link.process_specification with pre_cut=True) on the stored sleep fit, so
nothing is re-derived or re-estimated. phi-hat is a deterministic function of the states, so C1
describes how the estimated function spreads the states, not separate evidence on sleepiness.

Usage:  python make_followups_1020.py [--only C1,C3,C5] [--refresh]
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import datetime
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths as _paths
from make_xi_diagnostics import _codes, _shapley, _stamp, f3, compile_check

SPEC = 12
ROUTINES = (3, 4)
CACHE = _paths.rout_dir() / "followups_cache"
SIDECAR = _paths.rout_dir() / "followups_1020.json"
TEX_NEG = _paths.drafts_dir() / "tab_negative_active.tex"
PANEL = _paths.PROCESSED / "market_panel.parquet"
STATES = {"cadunico": "cadunico_families_per1000", "age65": "fraction_65plus",
          "selic": "risk_free_qoq_lag", "mobile": "connections_per100", "pix": "pix_exists"}
CEIL = 1.0 - 1e-9


def pre_cut(est: int, refresh: bool) -> tuple[pd.DataFrame, dict, dict]:
    """Every spec-12 cell before the (14-B) cut, with phi-hat, the native index and the active
    balance before the floor; the index coefficients; the input stamps. Cached on the inputs."""
    import sleep_demand_prep_link as P
    pkl = _paths.est_dir(est) / "estimation_results.pkl"
    key = dict(pkl=_stamp(pkl), panel=_stamp(Path(P.PANEL_CSV)))
    CACHE.mkdir(parents=True, exist_ok=True)
    out, meta_p = CACHE / f"precut_E{est}.parquet", CACHE / f"precut_E{est}.json"
    if out.is_file() and meta_p.is_file() and not refresh:
        m = json.loads(meta_p.read_text())
        if m["key"] == key:
            return pd.read_parquet(out), m["upsilon"], key
    import gc
    link, _tag, time_block = P.DEMAND_CFG[est]
    df_base = P.build_base_panel(P.PANEL_CSV, time_block=time_block)
    with open(pkl, "rb") as f:
        rd = pickle.load(f)
    name = next(k for k in rd if P.SPEC_MAP[SPEC] in k)
    ups = P.extract_upsilon_terms(rd[name]["second_stage"])
    d, _s, _n = P.process_specification(name, rd[name], df_base, link, pre_cut=True)
    del df_base, rd
    gc.collect()
    keep = ["CodConglomeradoPrudencial", "mca_code", "year", "quarter", "time_id", "deposit_type",
            "is_B", "deposit_balance", "lagged_deposits", "gross_return_lag", "pop_total",
            "phi_mt", "phi_t", "phi_index", "active_raw", *STATES.values()]
    d = d[[c for c in dict.fromkeys(keep) if c in d.columns]].reset_index(drop=True)
    d["mca_code"] = d["mca_code"].astype(str)
    d.to_parquet(out)
    meta_p.write_text(json.dumps(dict(key=key, upsilon=ups), indent=2))
    return d, ups, key


# ── C1 ───────────────────────────────────────────────────────────────────────
def c1(d: pd.DataFrame, ups: dict) -> dict:
    b = d[d["is_B"].astype(bool)]
    g = b.groupby(["mca_code", "time_id"])
    mq = g.agg(phi=("phi_mt", "first"), phi_sd=("phi_mt", "std"), index=("phi_index", "first"),
               pop=("pop_total", "first"), dep=("lagged_deposits", "sum"),
               **{k: (v, "first") for k, v in STATES.items()}).reset_index()
    if float(mq["phi_sd"].fillna(0).max()) > 1e-9:
        sys.exit("phi-hat varies within an MCA-quarter: it is not a market-quarter object")
    out = dict(n_cells=int(len(mq)), n_mca=int(mq["mca_code"].nunique()),
               n_quarters=int(mq["time_id"].nunique()))
    at_ceiling = mq["phi"] >= CEIL
    out["ceiling"] = dict(share=float(at_ceiling.mean()),
                          share_pop=float(np.average(at_ceiling, weights=mq["pop"])),
                          share_dep=float(np.average(at_ceiling, weights=mq["dep"])))
    for tag, frame in (("", mq), ("_noceil", mq[~at_ceiling])):
        for wname in ("pop", "dep"):
            f = frame[frame[wname] > 0]
            phi, r2 = _shapley(f["phi"].to_numpy(), f[wname].to_numpy().astype(float),
                               {"market": _codes(f["mca_code"]), "quarter": _codes(f["time_id"])})
            out[f"{wname}{tag}"] = dict(market=phi["market"], quarter=phi["quarter"],
                                        rest=1.0 - r2, n=int(len(f)))
    # Covariance split of the native index (linear in the states), population weights.
    f = mq[mq["pop"] > 0]
    w = f["pop"].to_numpy().astype(float)
    idx = f["index"].to_numpy()
    rebuilt = ups.get("constant", 0.0) + sum(ups[v] * f[k].to_numpy() for k, v in
                                             ((k, v) for k, v in STATES.items()))
    if float(np.max(np.abs(rebuilt - idx))) > 1e-8:
        sys.exit("the index is not constant + sum of the five states: the state list differs")
    ic = idx - np.average(idx, weights=w)
    var = float(np.average(ic ** 2, weights=w))
    split = {}
    for k, v in STATES.items():
        s = f[k].to_numpy() - np.average(f[k].to_numpy(), weights=w)
        split[k] = float(ups[v] * np.average(s * ic, weights=w) / var)
    assert abs(sum(split.values()) - 1.0) < 1e-9, split
    out["index"] = split
    out["upsilon"] = {k: ups[v] for k, v in STATES.items()}
    return out


# ── C3 ───────────────────────────────────────────────────────────────────────
def c3(d: pd.DataFrame) -> dict:
    d = d.copy()
    d["trunc"] = ~(d["active_raw"] > 1e-6)
    d["firm"] = np.where(d["is_B"].astype(bool), "B", "D")
    d["phi_row"] = np.where(d["is_B"].astype(bool), d["phi_mt"], d["phi_t"])

    def shares(f):
        return dict(share_cells=float(f["trunc"].mean()),
                    share_bal=float(f.loc[f["trunc"], "deposit_balance"].sum()
                                    / f["deposit_balance"].sum()),
                    n=int(len(f)))

    out = shares(d)
    out["by_firm"] = {k: shares(f) for k, f in d.groupby("firm")}
    out["by_type"] = {f"k{int(k)}": shares(f) for k, f in d.groupby("deposit_type")}
    out["by_year"] = {str(int(k)): shares(f) for k, f in d.groupby("year")}
    b = d[d["firm"] == "B"]
    pop = b.groupby("mca_code")["pop_total"].mean()
    q = pd.qcut(pop.rank(method="first"), 5, labels=[f"q{i}" for i in range(1, 6)])
    bq = b.assign(size=b["mca_code"].map(q))
    out["by_size_B"] = {str(k): shares(f) for k, f in bq.groupby("size", observed=True)}
    pg = d["phi_row"] * d["gross_return_lag"] >= 1.0
    out["phig_ge1"] = dict(share_rows=float(pg.mean()), **shares(d[pg]))
    # Market bound: phi_mt <= min_j Dep / (g Dep_lag) over the B rows of a market-quarter.
    den = b["gross_return_lag"] * b["lagged_deposits"]
    r = (b["deposit_balance"] / den).where(den > 0)
    mb = b.assign(r=r).dropna(subset=["r"]).groupby(["mca_code", "time_id"]).agg(
        phi=("phi_mt", "first"), rmin=("r", "min"), rp10=("r", lambda x: x.quantile(0.10)),
        nrows=("r", "size"))
    out["bound_violation"] = dict(min=float((mb["phi"] > mb["rmin"]).mean()),
                                  p10=float((mb["phi"] > mb["rp10"]).mean()),
                                  n_market_quarters=int(len(mb)))
    return out


def table_negative(res: dict) -> str:
    ests = [e for e in ROUTINES if f"E{e}" in res]

    def row(label, get):
        cells = []
        for e in ests:
            s = get(res[f"E{e}"]["C3"])
            cells += [f3(s["share_cells"]), f3(s["share_bal"])]
        return f"\\quad {label} & " + " & ".join(cells) + r" \\"

    def panel(t):
        return rf"\multicolumn{{{1 + 2 * len(ests)}}}{{@{{}}l}}{{\textit{{{t}}}}} \\"

    r0 = res[f"E{ests[0]}"]["C3"]
    body = [row("All cells", lambda s: s), r"\addlinespace", panel("By firm type")]
    body += [row(f"{k} firms", lambda s, k=k: s["by_firm"][k]) for k in ("B", "D")]
    body += [r"\addlinespace", panel("By deposit type")]
    names = {"k1": "Demand", "k2": "Savings", "k4": "Time", "k5": "Prepaid"}
    body += [row(names[k], lambda s, k=k: s["by_type"][k]) for k in names if k in r0["by_type"]]
    body += [r"\addlinespace", panel("B firms, by market population quintile")]
    body += [row(f"Quintile {k[1]}", lambda s, k=k: s["by_size_B"][k]) for k in sorted(r0["by_size_B"])]
    body += [r"\addlinespace", panel("By year")]
    body += [row(y, lambda s, y=y: s["by_year"][y]) for y in sorted(r0["by_year"])]
    body += [r"\addlinespace", row(r"Rows with $\hat{\phi}g \geq 1$", lambda s: s["phig_ge1"])]
    viol = []
    for e in ests:
        v = res[f"E{e}"]["C3"]["bound_violation"]
        viol += [f3(v["min"]), f3(v["p10"])]
    shr = []
    for e in ests:
        shr += [f3(res[f"E{e}"]["C3"]["phig_ge1"]["share_rows"]), ""]
    body += [r"\quad Share of rows with $\hat{\phi}g \geq 1$ & " + " & ".join(shr) + r" \\",
             r"\midrule",
             r"Market-quarters with $\hat{\phi}_{mt}$ above the bound & \multicolumn{"
             + str(2 * len(ests)) + r"}{c}{(minimum, 10th percentile)} \\",
             r"\quad Share & " + " & ".join(viol) + r" \\"]
    head1 = " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{E{e}}}" for e in ests)
    head1 = " & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{_rt(e)}}}" for e in ests)
    rules = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ests)))
    head2 = " & " + " & ".join("Cells & Balances" for _ in ests)
    note = (r"\textit{Notes:} Cells (conglomerate $\times$ deposit type $\times$ minimum comparable "
            r"area (MCA) $\times$ quarter, specification 12) whose active balance in "
            r"\eqref{eq:14-B}, the balance minus the carried-forward stock $\hat{\phi}\,g\,"
            r"\text{Dep}_{t-1}$, is at most zero. \eqref{eq:14-B} truncates them at zero and they "
            r"leave the demand sample. Shares of all cells and of all balances in each group. "
            r"$\hat{\phi}$ is $\hat{\phi}_{mt}$ for brick-and-mortar (B) firms and the national "
            r"$\hat{\phi}_t$ for digital (D) firms. Market population quintiles: B firms' MCAs by "
            r"mean population. Last panel: share of B market-quarters where $\hat{\phi}_{mt}$ "
            r"exceeds the bound the model implies, $\min_j \text{Dep}_{jkmt}/(g_{jkm,t-1}\,"
            r"\text{Dep}_{jkm,t-1})$ across the market's B cells, and the same share with the "
            r"10th percentile in place of the minimum. Diagnostic only: the demand estimation "
            r"keeps \eqref{eq:14-B} as it is.")
    return "\n".join([
        "% tab_negative_active.tex -- GENERATED by make_followups_1020.py. Do not edit.",
        r"\begin{spacing}{1.0}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\par\addvspace{\LTpre}\noindent\begin{minipage}{\textwidth}\captionsetup{position=top}",
        r"\captionof{table}{Cells with Negative Active Demand in \eqref{eq:14-B}}"
        r"\label{tab:negative_active}",
        r"\begin{tabularx}{\textwidth}{@{}X"
        + r">{\raggedleft\arraybackslash}p{1.7cm}" * (2 * len(ests)) + r"@{}}",
        r"\toprule", head1 + r" \\", rules, head2 + r" \\", r"\midrule", *body, r"\bottomrule",
        rf"\multicolumn{{{1 + 2 * len(ests)}}}{{@{{}}p{{\dimexpr\textwidth-2\tabcolsep\relax}}@{{}}}}"
        rf"{{\footnotesize {note}}} \\",
        r"\end{tabularx}", r"\end{minipage}\par\prevdepth=0pt\addvspace{\LTpost}",
        r"\end{spacing}", ""])


def _rt(e: int) -> str:
    from utils import routines as R
    r = R.ROUTINES[e]
    return f"{r.tex_name} {r.tex_ref}"


# ── C5 ───────────────────────────────────────────────────────────────────────
def c5() -> dict:
    cols = {"dep_a1": "k1", "dep_a2": "k2", "dep_a4": "k4", "dep_a5": "k5"}
    p = pd.read_parquet(PANEL, columns=["is_B", "year", "quarter", *cols])
    p = p[p["year"].isin([2016, 2024])]
    # Annual mean over quarters of the national balance by firm type and deposit type.
    q = p.groupby(["year", "quarter", "is_B"])[list(cols)].sum().groupby(["year", "is_B"]).mean()
    out = {}
    for y in (2016, 2024):
        D, B = q.loc[(y, 0)], q.loc[(y, 1)]
        tot = D + B
        out[y] = dict(w=(tot / tot.sum()).rename(cols).to_dict(),
                      s=(D / tot.where(tot > 0)).fillna(0.0).rename(cols).to_dict(),
                      S=float(D.sum() / tot.sum()))
    a, z = out[2016], out[2024]
    ks = list(cols.values())
    dS = z["S"] - a["S"]
    within16 = sum(a["w"][k] * (z["s"][k] - a["s"][k]) for k in ks)
    between16 = sum((z["w"][k] - a["w"][k]) * z["s"][k] for k in ks)
    within24 = sum(z["w"][k] * (z["s"][k] - a["s"][k]) for k in ks)
    between24 = sum((z["w"][k] - a["w"][k]) * a["s"][k] for k in ks)
    for x, y_ in ((within16 + between16, dS), (within24 + between24, dS)):
        assert abs(x - y_) < 1e-12
    return dict(total=dS, share2016=a["S"], share2024=z["S"],
                between=dict(w2016=between16, w2024=between24),
                within=dict(w2016=within16, w2024=within24), weights=dict(y2016=a["w"], y2024=z["w"]),
                within_type_share=dict(y2016=a["s"], y2024=z["s"]))


# ── C6 ───────────────────────────────────────────────────────────────────────
FEES = _paths.PROCESSED / "market_panel_with_fees.parquet"
INCUMBENTS = {"C0080329": "BB", "C0080738": "Caixa", "C0080075": "Bradesco", "C0080099": "Itau",
              "C0080185": "Santander"}


def c6() -> dict:
    """Fee revenue over total deposits (COSIF, quarterly increments), the December quarter x 4 x
    100 (pp a year), as service_fees_notes.md Section 5: the median of the five incumbents and of
    the D firms by year, 2016-2024 (2025 is the COSIF reclassification and is dropped)."""
    d = pd.read_parquet(FEES, columns=["CodConglomeradoPrudencial", "is_B", "year", "quarter",
                                       "cosif_fee_ratio_total_deposits", "cosif_fee_valid"])
    d = d.drop_duplicates(["CodConglomeradoPrudencial", "year", "quarter"])
    d = d[(d["quarter"] == 4) & d["year"].between(2016, 2024) & (d["cosif_fee_valid"] == 1)]
    d = d.dropna(subset=["cosif_fee_ratio_total_deposits"])
    d["fee_pp"] = 400.0 * d["cosif_fee_ratio_total_deposits"]
    inc = d[d["CodConglomeradoPrudencial"].isin(INCUMBENTS)].groupby("year")["fee_pp"]
    dig = d[d["is_B"] == 0].groupby("year")["fee_pp"]
    out = dict(incumbent={str(y): float(v) for y, v in inc.median().items()},
               digital={str(y): float(v) for y, v in dig.median().items()},
               n_incumbent={str(y): int(v) for y, v in inc.size().items()},
               n_digital={str(y): int(v) for y, v in dig.size().items()})
    out["incumbent_change_2019_2024"] = out["incumbent"]["2024"] - out["incumbent"]["2019"]
    out["digital_change_2019_2024"] = out["digital"]["2024"] - out["digital"]["2019"]
    return out


# ── C4 ───────────────────────────────────────────────────────────────────────
PIX_MANDATORY = _paths.OPEN_FINANCE / "shared" / "PIX" / "pix_mandatory_institutions.csv"


def c4() -> dict:
    """Pix exposure by MCA: the 2019 mobile measures, and the 2019 share of B deposits held by
    conglomerates with an institution mandated into Pix; their spread and correlation; and the
    first stage of Pix payer transactions per inhabitant on each standardized exposure x post
    (2020Q4 on), with MCA and quarter effects and the 2019 Cadunico rate x post, clustered by
    MCA (F = t squared, one excluded instrument)."""
    import statsmodels.api as sm
    from utils import conglomerates as cc
    cols = ["CodConglomeradoPrudencial", "mca_code", "is_B", "year", "quarter", "dep_a1", "dep_a2",
            "dep_a4", "dep_a5", "connections_per100", "frac_4g5g", "cadunico_families_per1000",
            "pop_total", "pix_txns_pf"]
    p = pd.read_parquet(PANEL, columns=cols)
    p = p[p["is_B"] == 1]
    p["mca_code"] = p["mca_code"].astype(str)
    p["dep"] = p[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].fillna(0).sum(axis=1)
    mand = pd.read_csv(PIX_MANDATORY, dtype=str)
    roots = set(mand["cnpj"].str.replace(r"\D", "", regex=True).str.zfill(14).str[:8])
    m = cc.build_cnpj_to_congl()
    m = m[m["AnoMes_list"].astype(float).between(201901, 201912)] if "AnoMes_list" in m else m
    mand_congl = set(m.loc[m["CNPJ"].astype(str).str.zfill(8).isin(roots), "CodConglomeradoPrudencial"])
    y19 = p[p["year"] == 2019]
    mq = y19.groupby(["mca_code"]).agg(mobile=("connections_per100", "mean"),
                                        g45=("frac_4g5g", "mean"),
                                        cad=("cadunico_families_per1000", "mean"),
                                        pop=("pop_total", "mean"))
    dm = y19.assign(mand=y19["CodConglomeradoPrudencial"].isin(mand_congl))
    sh = (dm[dm["mand"]].groupby("mca_code")["dep"].sum() / dm.groupby("mca_code")["dep"].sum())
    mq["mandated"] = sh.reindex(mq.index).fillna(0.0)
    mq = mq.dropna()
    out = dict(n_mandated_institutions=int(len(mand)), n_mandated_congl=int(len(mand_congl)),
               n_mca=int(len(mq)))
    # MCA-quarter panel of Pix payer transactions per inhabitant.
    pq_ = (p.groupby(["mca_code", "year", "quarter"])
           .agg(txns=("pix_txns_pf", "first"), pop=("pop_total", "first")).reset_index())
    pq_ = pq_[pq_["mca_code"].isin(mq.index) & (pq_["pop"] > 0)]
    pq_["t"] = pq_["year"] * 4 + pq_["quarter"] - 1
    pq_["post"] = (pq_["t"] >= 2020 * 4 + 3).astype(float)
    pq_["y"] = pq_["txns"] / pq_["pop"]
    for k in ("mobile", "g45", "mandated"):
        z = (mq[k] - mq[k].mean()) / mq[k].std()
        r = dict(sd=float(mq[k].std()), mean=float(mq[k].mean()),
                 corr_mandated=float(np.corrcoef(mq[k], mq["mandated"])[0, 1]))
        f = pq_.assign(x=pq_["mca_code"].map(z) * pq_["post"],
                       c=pq_["mca_code"].map(mq["cad"]) * pq_["post"]).dropna(subset=["y", "x", "c"])
        w = np.ones(len(f))
        R = _resid_fe(f[["y", "x", "c"]].to_numpy(), w,
                      [_codes(f["mca_code"]), _codes(f["t"])])
        res = sm.OLS(R[:, 0], R[:, 1:]).fit(cov_type="cluster",
                                            cov_kwds={"groups": f["mca_code"].to_numpy()})
        r["fs"] = dict(slope=float(res.params[0]), se=float(res.bse[0]),
                       f=float(res.tvalues[0] ** 2), n=int(len(f)))
        out[k] = r
    return out


def _resid_fe(Y, w, groups):
    from make_xi_diagnostics import _resid
    return _resid(Y, w, groups)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="C1,C3,C5")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    todo = set(a.only.split(","))
    res = json.loads(SIDECAR.read_text()) if SIDECAR.is_file() else {}
    meta = res.setdefault("_meta", {})
    for e in ROUTINES:
        if not todo & {"C1", "C3"}:
            break
        d, ups, key = pre_cut(e, a.refresh)
        r = res.setdefault(f"E{e}", {})
        meta[f"E{e}_inputs"] = key
        if "C1" in todo:
            r["C1"] = c1(d, ups)
            c = r["C1"]
            print(f"E{e} C1 pop {c['pop']} | dep {c['dep']} | ceiling {c['ceiling']['share']:.3f}"
                  f" | index {c['index']}")
        if "C3" in todo:
            r["C3"] = c3(d)
            c = r["C3"]
            print(f"E{e} C3 cells {c['share_cells']:.3f} bal {c['share_bal']:.3f} "
                  f"bound {c['bound_violation']} phig>=1 {c['phig_ge1']['share_rows']:.3f}"
                  f"/{c['phig_ge1']['share_cells']:.3f}")
    if "C5" in todo:
        res["C5"] = c5()
        print("C5", {k: res["C5"][k] for k in ("total", "between", "within")})
        meta["C5_panel"] = _stamp(PANEL)
    if "C6" in todo:
        res["C6"] = c6()
        meta["C6_fees"] = _stamp(FEES)
        print("C6 incumbent", {k: round(v, 2) for k, v in res["C6"]["incumbent"].items()})
        print("C6 digital  ", {k: round(v, 2) for k, v in res["C6"]["digital"].items()})
    if "C4" in todo:
        res["C4"] = c4()
        meta["C4_inputs"] = dict(panel=_stamp(PANEL), mandatory=_stamp(PIX_MANDATORY))
        for k in ("mobile", "g45", "mandated"):
            v = res["C4"][k]
            print(f"C4 {k}: sd {v['sd']:.3f} corr_mand {v['corr_mandated']:.3f} fs "
                  f"{v['fs']['slope']:.3f} (se {v['fs']['se']:.3f}) F {v['fs']['f']:.1f}")
        print("C4", {k: res["C4"][k] for k in ("n_mandated_institutions", "n_mandated_congl", "n_mca")})
    meta["written"] = datetime.datetime.now().isoformat(timespec="seconds")
    SIDECAR.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"  wrote {SIDECAR}")
    if "C3" in todo:
        TEX_NEG.write_text(table_negative(res), encoding="utf-8")
        print(f"  wrote {TEX_NEG}")
        import make_xi_diagnostics as X
        X._STANDALONE = X._STANDALONE.replace("tab_xi_diagnostics.tex", TEX_NEG.name)
        if not compile_check(TEX_NEG.parent):
            sys.exit(f"{TEX_NEG.name} did not compile cleanly on its own")


if __name__ == "__main__":
    main()
