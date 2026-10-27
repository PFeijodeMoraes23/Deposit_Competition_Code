"""
make_pix_exposure.py
====================
Pix usage by MCA and quarter, three pre-launch exposures, and the cross-market first stage of
usage on each exposure (brief C4, 2026-10-20).

Usage (C4a). The BCB Pix statistics by municipality (DATA_RAW/BCB/pix_municipal/
raw/pix_mun_YYYYMM.csv) mapped to MCAs with the IBGE crosswalk the panel uses
(each municipality's latest MCA), per MCA and quarter, over the panel's population:
  tx      transactions sent by individuals (QT_PagadorPF), summed over the quarter's months
  payers  distinct individual payers (QT_PES_PagadorPF): monthly counts cannot be summed across
          months, so the quarter takes the mean of its monthly counts
  value   value sent by individuals (VL_PagadorPF), in 2024Q4 reais (the IPCA deflator of
          build_advertising_panel.ipca_index), summed over the quarter's months
2020Q4 has two months (November and December). The sample is 2020Q4 to 2024Q4, the panel's
window: months after 2024-12 are not read.

Exposures (C4b), fixed at 2019, per MCA:
  mandated  the 2019 share of B deposits (types 1, 2, 4, 5) held by conglomerates with an
            institution mandated into Pix at launch (shared/PIX/pix_mandatory_institutions.csv,
            CNPJ roots mapped to conglomerates with the 2019 IF-Data lists; only institutions
            whose first mandatory roster is dated by the 2020-11-16 launch, so 34 of the 51)
  mobile    mobile lines per 100 inhabitants (connections_per100), 2019 mean
  g45       the 4G/5G share of mobile accesses (frac_4g5g), 2019 mean
Moments: population-weighted mean, standard deviation and correlations across MCAs.

First stage, cross-market (2020Q4 to 2024Q4 only, Pix has no pre-period):
  usage_mt = beta * z_m + quarter effects + 2019 Cadunico x quarter effects + log population,
with z_m the exposure standardized to population-weighted mean 0 and standard deviation 1,
population weights, MCA clusters (state clusters as a robustness check). The yearly version
replaces beta * z_m with sum_y beta_y z_m 1{year = y}. F: the squared t of beta (one exposure),
or the Wald F of the three betas together. Also without the Cadunico x quarter effects.

Pooled post-launch first stage (pooled_post_fs, a check on the panel's own Pix series): Pix payer
transactions per inhabitant from the market panel on each 2019 exposure x post (2020Q4 on), with
MCA and quarter effects and the 2019 Cadunico rate x post, unweighted, MCA clusters; its mandated
share uses the same 34 institutions mandated at launch as the exposure above. Nothing reads it yet.

Outputs: Rout/pix_usage_mca_quarter.parquet,
Rout/pix_exposure.json (read by make_paper_numbers.py, pix.exposure.* and pix.usage.*, and by
make_slide_figures.py --only pixexposure; key pooled_post_fs as above),
Drafts/Deposit Competition/tab_pix_exposure_fs.tex.

Usage:  python make_pix_exposure.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils import raw_paths
from make_xi_diagnostics import _stamp, f3, compile_check

RAW = raw_paths.BCB_PIX_MUNICIPAL_RAW
PANEL = _paths.PROCESSED / "market_panel.parquet"
XWALK = _paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"
MANDATORY = _paths.OPEN_FINANCE / "shared" / "PIX" / "pix_mandatory_institutions.csv"
OUT = _paths.rout_dir() / "pix_usage_mca_quarter.parquet"
SIDECAR = _paths.rout_dir() / "pix_exposure.json"
TEX = _paths.drafts_dir() / "tab_pix_exposure_fs.tex"
LAST_MONTH = 202412                 # the panel window ends in 2024Q4
PIX_LAUNCH = pd.Timestamp("2020-11-16")
EXPOSURES = ("mandated", "mobile", "g45")
MEASURES = ("tx", "payers", "value")
LABELS = {"mandated": "Mandated institutions' share", "mobile": "Mobile lines per 100",
          "g45": "4G/5G share", "tx": "Transactions", "payers": "Payers", "value": "Value"}


# ── C4a ──────────────────────────────────────────────────────────────────────
def usage_panel() -> tuple[pd.DataFrame, dict]:
    from build_advertising_panel import ipca_index
    frames = []
    for p in sorted(RAW.glob("pix_mun_*.csv")):
        ym = int(p.stem[-6:])
        if ym > LAST_MONTH:
            continue
        d = pd.read_csv(p, dtype={"Municipio_Ibge": str},
                        usecols=["AnoMes", "Municipio_Ibge", "VL_PagadorPF", "QT_PagadorPF",
                                 "QT_PES_PagadorPF"])
        d["VL_PagadorPF"] = pd.to_numeric(d["VL_PagadorPF"].astype(str).str.replace(",", "."))
        frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    m["municipality_code"] = pd.to_numeric(m["Municipio_Ibge"], errors="coerce")
    xw = pd.read_csv(XWALK, usecols=["municipality_code", "year", "mca_code", "state_code"])
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    xw["mca_code"] = xw["mca_code"].astype(str)
    m = m.merge(xw[["municipality_code", "mca_code", "state_code"]], on="municipality_code",
                how="left")
    unm = m["mca_code"].isna()
    coverage = dict(muni_months=int(len(m)), unmapped_muni_months=int(unm.sum()),
                    unmapped_share=float(unm.mean()),
                    unmapped_tx_share=float(m.loc[unm, "QT_PagadorPF"].sum() / m["QT_PagadorPF"].sum()),
                    unmapped_codes=sorted({str(c) for c in m.loc[unm, "Municipio_Ibge"].tolist()}))
    m = m[~unm]
    m["year"], m["month"] = m["AnoMes"] // 100, m["AnoMes"] % 100
    m["quarter"] = (m["month"] - 1) // 3 + 1
    defl, base = ipca_index(False)
    m = m.merge(defl[["year", "quarter", "deflator"]], on=["year", "quarter"], how="left")
    if m["deflator"].isna().any():
        sys.exit("the IPCA deflator misses a quarter of the Pix months")
    m["value_real"] = m["VL_PagadorPF"] * m["deflator"]
    # MCA x month, then MCA x quarter.
    mm = (m.groupby(["mca_code", "year", "quarter", "AnoMes"], as_index=False)
          .agg(tx=("QT_PagadorPF", "sum"), payers=("QT_PES_PagadorPF", "sum"),
               value=("value_real", "sum")))
    q = (mm.groupby(["mca_code", "year", "quarter"], as_index=False)
         .agg(tx=("tx", "sum"), payers=("payers", "mean"), value=("value", "sum"),
              n_months=("AnoMes", "nunique")))
    st = (m.groupby("mca_code")["state_code"].agg(lambda s: s.mode().iloc[0]).rename("state"))
    q = q.merge(st, left_on="mca_code", right_index=True, how="left")
    pop = pd.read_parquet(PANEL, columns=["mca_code", "is_B", "year", "quarter", "pop_total"])
    pop = pop[pop["is_B"] == 1].astype({"mca_code": str})
    pop = pop.groupby(["mca_code", "year", "quarter"], as_index=False)["pop_total"].first()
    q = q.merge(pop, on=["mca_code", "year", "quarter"], how="left")
    last_pop = pop.sort_values(["year", "quarter"]).groupby("mca_code")["pop_total"].last()
    q["pop_total"] = q["pop_total"].fillna(q["mca_code"].map(last_pop))
    for k in MEASURES:
        q[f"{k}_pc"] = q[k] / q["pop_total"]
    q["t"] = q["year"] * 4 + q["quarter"] - 1
    est = q.reset_index(drop=True)
    coverage.update(deflator_base=base, n_months_2020q4=int(
        est.loc[(est["year"] == 2020) & (est["quarter"] == 4), "n_months"].max()))
    return est, coverage


# ── C4b ──────────────────────────────────────────────────────────────────────
def exposures() -> pd.DataFrame:
    from utils import conglomerates as cc
    cols = ["CodConglomeradoPrudencial", "mca_code", "is_B", "year", "dep_a1", "dep_a2", "dep_a4",
            "dep_a5", "connections_per100", "frac_4g5g", "cadunico_families_per1000", "pop_total"]
    p = pd.read_parquet(PANEL, columns=cols)
    p = p[(p["is_B"] == 1) & (p["year"] == 2019)].astype({"mca_code": str})
    p["dep"] = p[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].fillna(0).sum(axis=1)
    # Mandated AT LAUNCH: the export lists every institution ever flagged mandatory on a BCB
    # roster, dated by its first mandatory roster; later designations (2021-2025) are left out.
    mand_all = pd.read_csv(MANDATORY, dtype=str)
    mand = mand_all[pd.to_datetime(mand_all["mandatory_since"]) <= PIX_LAUNCH]
    roots = set(mand["cnpj"].str.replace(r"\D", "", regex=True).str.zfill(14).str[:8])
    m = cc.build_cnpj_to_congl()
    m = m[m["AnoMes_list"].astype(float).between(201901, 201912)]
    congl = set(m.loc[m["CNPJ"].astype(str).str.zfill(8).isin(roots), "CodConglomeradoPrudencial"])
    x = p.groupby("mca_code").agg(mobile=("connections_per100", "mean"), g45=("frac_4g5g", "mean"),
                                  cad2019=("cadunico_families_per1000", "mean"),
                                  pop2019=("pop_total", "mean"))
    dm = p.assign(mand=p["CodConglomeradoPrudencial"].isin(congl))
    x["mandated"] = (dm[dm["mand"]].groupby("mca_code")["dep"].sum()
                     / dm.groupby("mca_code")["dep"].sum()).reindex(x.index).fillna(0.0)
    n_all = len(x)
    x = x.dropna(subset=["mobile", "g45", "cad2019", "pop2019", "mandated"])
    x = x[x["pop2019"] > 0]
    x.attrs.update(n_mandated_institutions=int(len(mand)), n_mandated_ever=int(len(mand_all)),
                   n_mandated_congl=int(len(congl)),
                   n_mca_2019=int(n_all), n_mca_missing_exposure=int(n_all - len(x)))
    return x


def wstats(x: pd.DataFrame) -> dict:
    w = x["pop2019"].to_numpy()
    out = {}
    for e in EXPOSURES:
        v = x[e].to_numpy()
        mu = np.average(v, weights=w)
        out[e] = dict(mean=float(mu), sd=float(np.sqrt(np.average((v - mu) ** 2, weights=w))))
    for e in EXPOSURES:
        out[e]["corr"] = {}
        for o in EXPOSURES:
            a = x[e].to_numpy() - out[e]["mean"]
            b = x[o].to_numpy() - out[o]["mean"]
            out[e]["corr"][o] = float(np.average(a * b, weights=w) / (out[e]["sd"] * out[o]["sd"]))
    return out


def first_stage(d: pd.DataFrame, zs: list, measure: str, yearly: bool, cad: bool,
                cluster: str) -> dict:
    """WLS of usage per inhabitant on the standardized exposures `zs` (pooled, or by year),
    quarter effects, optionally 2019 Cadunico x quarter effects, and log population."""
    import statsmodels.api as sm
    qd = pd.get_dummies(d["t"], prefix="q", drop_first=False).astype(float)
    X = [qd]
    if cad:
        X.append(qd.iloc[:, 1:].mul(d["cad2019"].to_numpy(), axis=0).add_prefix("cad_"))
    X.append(pd.DataFrame({"log_pop": np.log(d["pop_total"].to_numpy())}, index=d.index))
    names = []
    for z in zs:
        if yearly:
            for y in sorted(d["year"].unique()):
                c = f"{z}_y{y}"
                X.append(pd.DataFrame({c: d[f"z_{z}"] * (d["year"] == y)}, index=d.index))
                names.append(c)
        else:
            X.append(pd.DataFrame({z: d[f"z_{z}"]}, index=d.index))
            names.append(z)
    X = pd.concat(X, axis=1)
    res = sm.WLS(d[f"{measure}_pc"].to_numpy(), X, weights=d["pop_total"].to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster].to_numpy()})
    ci = res.conf_int()
    out = {c: dict(slope=float(res.params[c]), lo=float(ci.loc[c, 0]), hi=float(ci.loc[c, 1]),
                   f=float(res.tvalues[c] ** 2)) for c in names}
    if not yearly and len(zs) > 1:
        R = np.zeros((len(zs), len(res.params)))
        for i, c in enumerate(names):
            R[i, list(res.params.index).index(c)] = 1.0
        out["joint_f"] = float(np.squeeze(res.f_test(R).fvalue))
    out["n"] = int(len(d))
    return out


def pooled_post_fs() -> dict:
    """The pooled post-launch first stage on the market panel's own Pix series: Pix payer
    transactions per inhabitant (pix_txns_pf / pop_total, MCA x quarter) on each standardized 2019
    exposure x post (2020Q4 on), with MCA and quarter effects and the 2019 Cadunico rate x post,
    unweighted, clustered by MCA (F = t squared, one excluded instrument). Exposures are
    standardized with unweighted moments across MCAs. The mandated share counts the institutions
    mandated at launch (mandatory_since <= PIX_LAUNCH), mapped with the 2019 IF-Data lists, as in
    exposures(). Also the unweighted mean and SD of each exposure and its correlation with the
    mandated share."""
    import statsmodels.api as sm
    from utils import conglomerates as cc
    from make_xi_diagnostics import _codes, _resid
    cols = ["CodConglomeradoPrudencial", "mca_code", "is_B", "year", "quarter", "dep_a1", "dep_a2",
            "dep_a4", "dep_a5", "connections_per100", "frac_4g5g", "cadunico_families_per1000",
            "pop_total", "pix_txns_pf"]
    p = pd.read_parquet(PANEL, columns=cols)
    p = p[p["is_B"] == 1]
    p["mca_code"] = p["mca_code"].astype(str)
    p["dep"] = p[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].fillna(0).sum(axis=1)
    mand = pd.read_csv(MANDATORY, dtype=str)
    mand = mand[pd.to_datetime(mand["mandatory_since"]) <= PIX_LAUNCH]
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
        R = _resid(f[["y", "x", "c"]].to_numpy(), w, [_codes(f["mca_code"]), _codes(f["t"])])
        res = sm.OLS(R[:, 0], R[:, 1:]).fit(cov_type="cluster",
                                            cov_kwds={"groups": f["mca_code"].to_numpy()})
        r["fs"] = dict(slope=float(res.params[0]), se=float(res.bse[0]),
                       f=float(res.tvalues[0] ** 2), n=int(len(f)))
        out[k] = r
    return out


def run() -> dict:
    est, cov = usage_panel()
    est.to_parquet(OUT)
    x = exposures()
    mom = wstats(x)
    d = est.merge(x, left_on="mca_code", right_index=True, how="inner")
    d = d.dropna(subset=["pop_total", "cad2019", *EXPOSURES, *(f"{k}_pc" for k in MEASURES)])
    d = d[d["pop_total"] > 0].reset_index(drop=True)
    for e in EXPOSURES:
        d[f"z_{e}"] = (d[e] - mom[e]["mean"]) / mom[e]["sd"]
    fs = {}
    for k in MEASURES:
        fs[k] = {}
        for e in EXPOSURES:
            r = first_stage(d, [e], k, False, True, "mca_code")[e]
            ry = first_stage(d, [e], k, True, True, "mca_code")
            r["yearly"] = {str(y): ry[f"{e}_y{y}"] for y in sorted(d["year"].unique())}
            r["no_cad"] = first_stage(d, [e], k, False, False, "mca_code")[e]
            r["state_cl"] = first_stage(d, [e], k, False, True, "state")[e]
            fs[k][e] = r
        fs[k]["joint_f"] = first_stage(d, list(EXPOSURES), k, False, True, "mca_code")["joint_f"]
        fs[k]["joint_f_state"] = first_stage(d, list(EXPOSURES), k, False, True, "state")["joint_f"]
    return dict(moments=mom, fs=fs, coverage=cov,
                sample=dict(n_mca=int(d["mca_code"].nunique()), n_quarters=int(d["t"].nunique()),
                            n_obs=int(len(d)), n_states=int(d["state"].nunique()),
                            n_mca_exposure=int(len(x)), **x.attrs),
                pooled_post_fs=pooled_post_fs(),
                _meta=dict(written=datetime.datetime.now().isoformat(timespec="seconds"),
                           raw=str(RAW), raw_last=_stamp(sorted(RAW.glob("pix_mun_*.csv"))[-1]),
                           panel=_stamp(PANEL), xwalk=_stamp(XWALK), mandatory=_stamp(MANDATORY)))


# ── C4c ──────────────────────────────────────────────────────────────────────
def m3(v) -> str:
    return f"${f3(v)}$"


def table(r: dict) -> str:
    mom, fs = r["moments"], r["fs"]
    a = [f"\\quad {LABELS[e]} & {m3(mom[e]['mean'])} & {m3(mom[e]['sd'])} & "
         + " & ".join(m3(mom[e]["corr"][o]) for o in EXPOSURES) + r" \\" for e in EXPOSURES]
    b = []
    for e in EXPOSURES:
        b.append(f"\\quad {LABELS[e]} & " + " & ".join(
            f"{m3(fs[k][e]['slope'])} & {f3(fs[k][e]['f'])}" for k in MEASURES) + r" \\")
        b.append(r"\quad & " + " & ".join(
            f"[{m3(fs[k][e]['lo'])}, {m3(fs[k][e]['hi'])}] &" for k in MEASURES)[:-1] + r" \\")
    b.append(r"\quad Joint $F$, three exposures & " + " & ".join(
        f" & {f3(fs[k]['joint_f'])}" for k in MEASURES) + r" \\")
    b.append(r"\quad Joint $F$, state clusters & " + " & ".join(
        f" & {f3(fs[k]['joint_f_state'])}" for k in MEASURES) + r" \\")
    s = r["sample"]
    note = (r"\textit{Notes:} Usage: Pix transactions sent by individuals, distinct individual "
            r"payers, and real value sent by individuals (2024Q4 reais), per inhabitant, by "
            r"minimum comparable area (MCA) and quarter, 2020Q4 to 2024Q4, from the Banco Central "
            r"do Brasil Pix statistics by municipality. 2020Q4 has two months. Payers: the mean of "
            r"the quarter's monthly counts. Exposures are fixed at 2019. Mandated institutions' "
            r"share: the 2019 share of brick-and-mortar deposits held by conglomerates with an "
            r"institution mandated into Pix at launch. Panel A: population-weighted means, standard "
            r"deviations and correlations across MCAs. Panel B: slope of usage on the exposure, "
            r"per one standard deviation of the exposure. Regressions have quarter effects, "
            r"Cadastro \'{U}nico (Cad\'{U}nico) registration in 2019 interacted with quarter "
            r"effects, and log population, with population weights. $F$: the squared $t$ statistic "
            r"of the slope, and the joint $F$ of the three exposures in one regression. Intervals: "
            r"95 percent, clustered by MCA. "
            f"{s['n_mca']:,} MCAs, {s['n_quarters']} quarters.")
    head_a = (r" & Mean & SD & \multicolumn{3}{c}{Correlation with} & \\" + "\n"
              + r" & & & Mandated & Mobile & 4G/5G & \\")
    head_b = (r" & \multicolumn{2}{c}{Transactions} & \multicolumn{2}{c}{Payers} & "
              r"\multicolumn{2}{c}{Value} \\" + "\n" + r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} "
              r"\cmidrule(lr){6-7}" + "\n" + r" & Slope & $F$ & Slope & $F$ & Slope & $F$ \\")
    col = r"@{}X" + r">{\raggedleft\arraybackslash}p{1.65cm}" * 6 + r"@{}"
    return "\n".join([
        "% tab_pix_exposure_fs.tex -- GENERATED by make_pix_exposure.py. Do not edit.",
        r"\begin{spacing}{1.0}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\par\addvspace{\LTpre}\noindent\begin{minipage}{\textwidth}\captionsetup{position=top}",
        r"\captionof{table}{Pix Usage and Pre-Launch Exposure}\label{tab:pix_exposure_fs}",
        rf"\begin{{tabularx}}{{\textwidth}}{{{col}}}", r"\toprule",
        r"\multicolumn{7}{@{}l}{\textit{A. Exposure moments}} \\",
        head_a,
        r"\midrule", *[x.replace(r" \\", r" & \\") for x in a], r"\midrule",
        r"\multicolumn{7}{@{}l}{\textit{B. First stage}} \\", head_b, r"\midrule", *b,
        r"\bottomrule",
        rf"\multicolumn{{7}}{{@{{}}p{{\dimexpr\textwidth-2\tabcolsep\relax}}@{{}}}}"
        rf"{{\footnotesize {note}}} \\",
        r"\end{tabularx}", r"\end{minipage}\par\prevdepth=0pt\addvspace{\LTpost}",
        r"\end{spacing}", ""])


def main():
    r = run()
    SIDECAR.write_text(json.dumps(r, indent=2, default=float), encoding="utf-8")
    c, s = r["coverage"], r["sample"]
    print(f"  coverage: {c['unmapped_share']:.4%} of municipality-months unmapped "
          f"({c['unmapped_muni_months']:,} of {c['muni_months']:,}; {c['unmapped_tx_share']:.4%} of "
          f"transactions), codes {c['unmapped_codes'][:6]}; deflator base {c['deflator_base']}")
    print(f"  sample: {s}")
    for e in EXPOSURES:
        mo = r["moments"][e]
        print(f"  {e}: mean {mo['mean']:.3f} sd {mo['sd']:.3f} corr "
              + ", ".join(f"{o} {v:.3f}" for o, v in mo["corr"].items()))
    for k in MEASURES:
        for e in EXPOSURES:
            v = r["fs"][k][e]
            print(f"  fs {k:6s} {e:8s} {v['slope']:.4g} [{v['lo']:.4g}, {v['hi']:.4g}] F {v['f']:.1f}"
                  f" | no cad F {v['no_cad']['f']:.1f} | state F {v['state_cl']['f']:.1f}")
        print(f"  fs {k} joint F {r['fs'][k]['joint_f']:.1f} (state {r['fs'][k]['joint_f_state']:.1f})")
    TEX.write_text(table(r), encoding="utf-8")
    print(f"  wrote {SIDECAR}\n  wrote {TEX}\n  wrote {OUT}")
    import make_xi_diagnostics as X
    X._STANDALONE = X._STANDALONE.replace("tab_xi_diagnostics.tex", TEX.name)
    if not compile_check(TEX.parent):
        sys.exit(f"{TEX.name} did not compile cleanly on its own")


if __name__ == "__main__":
    main()
