"""
make_ccs_openings_pix.py
========================
Task AW1 (coding handoff 2026-11-02, Update 11-02k): did PF account openings at B banks jump at the
Pix launch? A side analysis: nothing here enters market_panel, an estimator or a paper number.

Opening rate, per conglomerate j and month t (PF relationships):
    o_jt = Openings_jt / (Adults_t - N_{j,t-1})
  Openings  PF gross openings, summed over the conglomerate's institutions (CC3,
            Rout/ccs_flows_institution_month_mapped.csv); a client who opens at two member
            institutions counts twice, as in CC3
  Adults    resident population aged 18 or more, PNAD Continua (SIDRA 6407, annual), placed at
            each July and interpolated linearly by month
  N         the conglomerate's stock of relationships. Where the BCB complaints ranking lists the
            conglomerate (BCB/Quality/processed/complaints_firm_quarterly.csv, "Quantidade de
            clientes - CCS", distinct clients, PF and PJ together), or else its member
            institutions (complaints_institution_quarterly.csv, summed), N is the ranking count at
            the end of the quarter's second month (the reference date CC4 found), carried to
            other months with the conglomerate's net flows (PF and PJ gross openings less
            closures). Elsewhere N is the net flows cumulated from 2016-01, floored at zero, which
            counts only relationships opened since 2016 (N_source says which). N includes firm
            clients (the ranking does not split them).

Regressions, B conglomerates (panel is_B = 1), 2018-01 to 2022-12, excluding flagged months:
    ln o_jt = theta_j + theta_moy + b1 post_t on_jt + b2 post_t + e_jt
  post      1 from 2020-11
  on_jt     1 when the conglomerate is on Pix that month: from the first month any member holds
            Pix keys (pix_participant_keys_panel.csv), or from the first month of the quarter R1
            first finds it on when its roster entry comes first; later quarters follow R1's
            quarterly on_pix (Rout/pix_participation_conglomerate_quarter.csv)
  theta_moy calendar-month (January..December) effects, so b2 is identified
  flagged   conglomerate-months with a flagged closure event (CC3 event_closed > 0), and Caixa
            2020-04..2020-08 (the emergency-aid accounts)
  Months with no PF opening drop out of the log; their count is reported.
Variants: b1 split by mandated and voluntary joiners; D conglomerates for scale; N dropped from the
denominator; Caixa dropped; a linear trend per conglomerate (B and D), which the event study's
pre-launch path calls for; b1 interacted with Pix use in the conglomerate's 2019 deposit footprint
(Route 3: Pix payers per inhabitant, 2021 mean, by MCA, Rout/pix_usage_mca_quarter.parquet,
weighted by the conglomerate's 2019 B deposits of types 1, 2, 4 and 5; standardized across the
sample's conglomerates). Event study: B conglomerates on Pix at launch (first month 2020-11 or
2020-12) against B conglomerates never on Pix by 2022-12, late joiners left out; event months
k = -12..12 around 2020-11, k = -1 omitted, k < -12 and k > 12 binned; the common path and the
launch group's difference.

Inference: conglomerate-clustered (CR1) standard errors and 95% intervals; for b1 also a wild
cluster restricted bootstrap p-value (Webb weights, 999 draws).

Outputs (Rout): ccs_openings_pix_panel.csv (the regression panel), ccs_openings_pix_aw1.json.

Usage:  python make_ccs_openings_pix.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import datetime
import json

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils import raw_paths

ROUT = _paths.rout_dir()
FLOWS = ROUT / "ccs_flows_institution_month_mapped.csv"
R1 = ROUT / "pix_participation_conglomerate_quarter.csv"
USAGE = ROUT / "pix_usage_mca_quarter.parquet"
RANKING = _paths.OPEN_FINANCE / "BCB" / "Quality" / "processed" / "complaints_firm_quarterly.csv"
RANKING_INST = _paths.OPEN_FINANCE / "BCB" / "Quality" / "processed" / "complaints_institution_quarterly.csv"
POP = raw_paths.DATA_RAW / "IBGE" / "pnadc_population" / "sidra_6407_n1_all_years.json"
KEYS = _paths.PIX_DIR / "pix_participant_keys_panel.csv"
XWALK = raw_paths.BCB_PIX_PARTICIPANTS / "compiled" / "ispb_conglomerate_map.csv"
PANEL = _paths.PROCESSED / "market_panel.parquet"
OUT_PANEL = ROUT / "ccs_openings_pix_panel.csv"
OUT_JSON = ROUT / "ccs_openings_pix_aw1.json"

CAIXA = "C0080738"
LAUNCH = pd.Period("2020-11", "M")
FIRST, LAST = pd.Period("2018-01", "M"), pd.Period("2022-12", "M")
ADULT_GROUPS = ("18 a 19 anos", "20 a 24 anos", "25 a 29 anos", "30 a 39 anos", "40 a 49 anos",
                "50 a 59 anos", "60 anos ou mais")
EVENT_K = 12
N_BOOT = 999
SEED = 20261103


# ── inputs ───────────────────────────────────────────────────────────────────
def adults_monthly() -> pd.Series:
    """18+ population by month: annual PNAD Continua values placed at July, linear in between."""
    raw = json.loads(POP.read_text(encoding="utf-8"))
    d = pd.DataFrame(raw[1:])
    d = d[(d["D5N"] == "Total") & (d["D2C"] == "606") & d["D4N"].isin(ADULT_GROUPS)]
    if d["D4N"].nunique() != len(ADULT_GROUPS):
        raise SystemExit("SIDRA 6407: an adult age group is missing")
    a = d.assign(V=pd.to_numeric(d["V"])).groupby("D3N")["V"].sum() * 1000.0
    s = pd.Series(a.values, index=pd.PeriodIndex([f"{y}-07" for y in a.index], freq="M"))
    full = pd.period_range(s.index.min(), s.index.max(), freq="M")
    return s.reindex(full).interpolate().rename("adults")


def flows_monthly() -> pd.DataFrame:
    """Conglomerate x month: PF openings, PF+PJ net flow, is_B, flagged."""
    f = pd.read_csv(FLOWS, dtype={"cnpj8": str})
    f = f[(f["mapped"] == 1) & (f["in_panel"] == 1) & f["is_B"].notna()]
    f["net"] = f["opened_gross"] - f["closed"]
    pf = f[f["type"] == "PF"].groupby(["conglomerate", "ym"]).agg(
        openings=("opened_gross", "sum"), event=("event_closed", "sum"), is_B=("is_B", "first"))
    net = f.groupby(["conglomerate", "ym"])["net"].sum()
    d = pf.join(net).reset_index()
    d["month"] = pd.PeriodIndex(d["ym"], freq="M")
    d["flagged"] = (d["event"] > 0) | ((d["conglomerate"] == CAIXA)
                                       & d["month"].between(pd.Period("2020-04", "M"),
                                                            pd.Period("2020-08", "M")))
    return d.drop(columns=["ym", "event"])


def _ranking(path, key: str) -> pd.DataFrame:
    """Ranking client counts (n > 0) at their reference month, the quarter's second month."""
    r = pd.read_csv(path, dtype=str)
    r = r[r[key].notna()].copy()
    r["n"] = pd.to_numeric(r["n_clients_ccs"], errors="coerce")
    r = r[r["n"] > 0]
    r["month"] = pd.PeriodIndex(r["year"].str[:4] + "-" + ((r["quarter"].astype(float).astype(int) - 1) * 3 + 2)
                                .astype(str), freq="M")
    return r.sort_values([key, "month"]).drop_duplicates([key, "month"], keep="last")[[key, "month", "n"]]


def stock(d: pd.DataFrame) -> pd.DataFrame:
    """N_jt: the ranking count at its reference month (the quarter's second month), carried by net
    flows. The count is the conglomerate's row of the ranking where it has one
    (N_source ranking_conglomerate), else the sum of its member institutions' rows
    (ranking_institutions); with neither, net flows cumulated from 2016-01, floored at zero
    (cumulated), which counts only relationships opened since 2016 and is not a stock."""
    rc = _ranking(RANKING, "cod_cong_prudencial")
    anchors = {c: (g.set_index("month")["n"], "ranking_conglomerate") for c, g in rc.groupby("cod_cong_prudencial")}
    ri = _ranking(RANKING_INST, "cnpj")
    f = pd.read_csv(FLOWS, dtype={"cnpj8": str}, usecols=["cnpj8", "conglomerate"]).drop_duplicates()
    ri = ri.merge(f, left_on="cnpj", right_on="cnpj8")
    for c, g in ri.groupby("conglomerate"):
        if c not in anchors:
            anchors[c] = (g.groupby("month")["n"].sum(), "ranking_institutions")
    out = []
    for c, g in d.sort_values("month").groupby("conglomerate"):
        g = g.set_index("month")
        cum = g["net"].cumsum()
        a, src = anchors.get(c, (pd.Series(dtype=float), "cumulated"))
        a = a[a.index.isin(cum.index)]
        if len(a):
            # for each month, the nearest anchor at or before it (else the first after it)
            ref = pd.Series(a.index, index=a.index).reindex(cum.index).ffill().bfill()
            n = a.reindex(ref.values).values + cum.values - cum.reindex(ref.values).values
        else:
            n = np.maximum(cum.values, 0.0)
            src = "cumulated"
        out.append(pd.DataFrame({"conglomerate": c, "month": cum.index, "N": n, "N_source": src}))
    return d.merge(pd.concat(out), on=["conglomerate", "month"])


def on_pix_monthly(conglomerates) -> pd.DataFrame:
    """on_jt for the months from 2020-11 to 2022-12, first_on_month, mandated, at_launch."""
    r1 = pd.read_csv(R1)
    r1["q"] = pd.PeriodIndex(r1["year"].astype(str) + "Q" + r1["quarter"].astype(str), freq="Q")
    xw = pd.read_csv(XWALK, dtype=str).set_index("ispb")["conglomerate"]
    k = pd.read_csv(KEYS, dtype={"ISPB": str})
    k = k[k["qtd_chaves"] > 0]
    k["conglomerate"] = k["ISPB"].str.zfill(8).map(xw)
    k["month"] = pd.PeriodIndex(k["AnoMes"].astype(str).str[:4] + "-" + k["AnoMes"].astype(str).str[4:], freq="M")
    kfirst = k.groupby("conglomerate")["month"].min().clip(lower=LAUNCH)
    g = r1.sort_values("q").groupby("conglomerate")
    info = pd.DataFrame({"mandated": g["mandated"].first(),
                         "first_q": r1[r1["on_pix"] == 1].groupby("conglomerate")["q"].min()})
    info["keys_first"] = kfirst.reindex(info.index)
    first = []
    for c, row in info.iterrows():
        fq, kf = row["first_q"], row["keys_first"]
        if pd.isna(fq):
            first.append(pd.NaT)
        elif pd.notna(kf) and kf.asfreq("Q") <= fq:
            first.append(kf)
        else:
            first.append(max(fq.asfreq("M", how="start"), LAUNCH))
    info["first_on_month"] = first
    info["first_source"] = np.where(info["first_q"].isna(), "never",
                                    np.where(info["first_on_month"] == info["keys_first"], "keys", "roster"))
    qon = r1.set_index(["conglomerate", "q"])["on_pix"]
    months = pd.period_range(LAUNCH, LAST, freq="M")
    rows = []
    for c in conglomerates:
        fm = info["first_on_month"].get(c, pd.NaT)
        for m in months:
            on = pd.notna(fm) and m >= fm
            if on and m.asfreq("Q") > fm.asfreq("Q"):
                on = bool(qon.get((c, m.asfreq("Q")), 1))
            rows.append((c, m, int(on)))
    on = pd.DataFrame(rows, columns=["conglomerate", "month", "on"])
    info["at_launch"] = info["first_on_month"].isin([LAUNCH, LAUNCH + 1])
    return on, info


def footprint_usage() -> pd.Series:
    """Pix payers per inhabitant in 2021 (mean over quarters), weighted by each B conglomerate's
    2019 deposits (types 1, 2, 4, 5) across MCAs."""
    u = pd.read_parquet(USAGE)
    u = u[u["year"] == 2021].groupby("mca_code")["payers_pc"].mean()
    p = pd.read_parquet(PANEL, columns=["CodConglomeradoPrudencial", "mca_code", "is_B", "year",
                                        "dep_a1", "dep_a2", "dep_a4", "dep_a5"])
    p = p[(p["is_B"] == 1) & (p["year"] == 2019)].copy()
    p["dep"] = p[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].fillna(0).sum(axis=1)
    p["u"] = p["mca_code"].astype(str).map(u.rename(index=str))
    p = p[p["u"].notna() & (p["dep"] > 0)]
    p["uw"] = p["u"] * p["dep"]
    g = p.groupby("CodConglomeradoPrudencial")[["uw", "dep"]].sum()
    return (g["uw"] / g["dep"]).rename("usage_2021")


# ── estimation ───────────────────────────────────────────────────────────────
def _design(d: pd.DataFrame, regs: list[str], trends: bool = False) -> tuple[np.ndarray, list[str]]:
    fe_j = pd.get_dummies(d["conglomerate"], prefix="j", drop_first=True, dtype=float)
    fe_m = pd.get_dummies(d["month"].dt.month, prefix="m", drop_first=True, dtype=float)
    parts = [d[regs].astype(float).reset_index(drop=True), fe_j.reset_index(drop=True),
             fe_m.reset_index(drop=True)]
    if trends:
        # one linear trend per conglomerate, in years from the launch
        t = np.array([(m - LAUNCH).n for m in d["month"]], dtype=float) / 12.0
        tr = pd.get_dummies(d["conglomerate"], prefix="tr", dtype=float).mul(t, axis=0)
        parts.append(tr.reset_index(drop=True))
    X = pd.concat(parts, axis=1)
    X.insert(0, "const", 1.0)
    keep = X.columns[(X != 0).any(axis=0)]
    return X[keep].to_numpy(), list(keep)


def _ols_cr1(X, y, g):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    G = np.unique(g)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in G:
        s = X[g == c].T @ e[g == c]
        meat += np.outer(s, s)
    n, k = X.shape
    adj = len(G) / (len(G) - 1) * (n - 1) / (n - k)
    V = adj * XtX_inv @ meat @ XtX_inv
    return b, np.sqrt(np.clip(np.diag(V), 0, None)), e


def _wcr_p(X, y, g, j, rng) -> float:
    """Wild cluster restricted bootstrap p-value for H0: b_j = 0 (Webb weights)."""
    b, se, _ = _ols_cr1(X, y, g)
    t0 = b[j] / se[j]
    Xr = np.delete(X, j, axis=1)
    br = np.linalg.pinv(Xr.T @ Xr) @ Xr.T @ y
    fit, er = Xr @ br, y - Xr @ br
    G, inv = np.unique(g, return_inverse=True)
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    ts = np.empty(N_BOOT)
    for r in range(N_BOOT):
        w = rng.choice(webb, size=len(G))[inv]
        bs, ses, _ = _ols_cr1(X, fit + er * w, g)
        ts[r] = bs[j] / ses[j]
    return float(np.mean(np.abs(ts) >= abs(t0)))


def fit(d: pd.DataFrame, regs: list[str], boot: list[str] | None = None, lhs="ln_o", rng=None,
        trends: bool = False) -> dict:
    d = d.reset_index(drop=True)
    X, names = _design(d, regs, trends)
    y = d[lhs].to_numpy(float)
    g = d["conglomerate"].to_numpy()
    b, se, _ = _ols_cr1(X, y, g)
    res = {"n_obs": int(len(d)), "n_conglomerates": int(d["conglomerate"].nunique()), "coef": {}}
    for r in regs:
        if r not in names:
            res["coef"][r] = None
            continue
        j = names.index(r)
        res["coef"][r] = {"b": round(float(b[j]), 4), "se": round(float(se[j]), 4),
                          "ci95": [round(float(b[j] - 1.96 * se[j]), 4), round(float(b[j] + 1.96 * se[j]), 4)]}
        if boot and r in boot:
            res["coef"][r]["wcr_p"] = round(_wcr_p(X, y, g, j, rng), 3)
    return res


def event_study(d: pd.DataFrame) -> dict:
    d = d.copy()
    k = np.array([(m - LAUNCH).n for m in d["month"]])
    d["k"] = np.clip(k, -EVENT_K - 1, EVENT_K + 1)
    ks = [x for x in range(-EVENT_K - 1, EVENT_K + 2) if x != -1]
    regs = []
    for x in ks:
        d[f"D{x}"] = (d["k"] == x).astype(float)
        d[f"L{x}"] = d[f"D{x}"] * d["at_launch"].astype(float)
        regs += [f"D{x}", f"L{x}"]
    r = fit(d, regs)
    path = [{"k": x, "month": str(LAUNCH + x) if abs(x) <= EVENT_K else ("<=" if x < 0 else ">=") + str(LAUNCH + x),
             "common": r["coef"][f"D{x}"], "launch_minus_never": r["coef"][f"L{x}"]} for x in ks]
    return {"n_obs": r["n_obs"], "n_conglomerates": r["n_conglomerates"],
            "n_launch": int(d.loc[d["at_launch"], "conglomerate"].nunique()),
            "n_never": int(d.loc[~d["at_launch"], "conglomerate"].nunique()), "path": path}


def main():
    rng = np.random.default_rng(SEED)
    adults = adults_monthly()
    d = stock(flows_monthly())
    d = d.sort_values(["conglomerate", "month"])
    d["N_lag"] = d.groupby("conglomerate")["N"].shift(1)
    d["adults"] = d["month"].map(adults)
    d = d[d["month"].between(FIRST, LAST)].copy()
    d["den"] = d["adults"] - d["N_lag"]
    d["den_share"] = d["den"] / d["adults"]
    on, info = on_pix_monthly(d["conglomerate"].unique())
    d = d.merge(on, on=["conglomerate", "month"], how="left").fillna({"on": 0})
    d["post"] = (d["month"] >= LAUNCH).astype(int)
    d["post_on"] = d["post"] * d["on"]
    d = d.join(info[["mandated", "first_on_month", "first_source", "at_launch"]], on="conglomerate")
    d["mandated"] = d["mandated"].fillna(0).astype(int)
    d["at_launch"] = d["at_launch"].fillna(False).astype(bool)
    d["post_on_mand"] = d["post_on"] * d["mandated"]
    d["post_on_vol"] = d["post_on"] * (1 - d["mandated"])
    use = footprint_usage()
    d["usage_2021"] = d["conglomerate"].map(use)

    ok = (d["openings"] > 0) & ~d["flagged"] & (d["den"] > 0)
    d["ln_o"] = np.where(ok, np.log(d["openings"].clip(lower=1)) - np.log(d["den"].where(d["den"] > 0, np.nan)), np.nan)
    d["ln_o_noN"] = np.where(ok, np.log(d["openings"].clip(lower=1)) - np.log(d["adults"]), np.nan)
    d.assign(month=d["month"].astype(str), first_on_month=d["first_on_month"].astype(str)).to_csv(
        OUT_PANEL, index=False)

    B = d[(d["is_B"] == 1) & ok].copy()
    D = d[(d["is_B"] == 0) & ok].copy()
    rep = {"sample": {
        "B_conglomerates": int(d.loc[d["is_B"] == 1, "conglomerate"].nunique()),
        "B_months_total": int((d["is_B"] == 1).sum()),
        "B_dropped_zero_openings": int(((d["is_B"] == 1) & (d["openings"] <= 0)).sum()),
        "B_dropped_flagged": int(((d["is_B"] == 1) & d["flagged"]).sum()),
        "B_dropped_nonpositive_den": int(((d["is_B"] == 1) & (d["den"] <= 0)).sum()),
        "B_used": int(len(B)), "D_used": int(len(D)), "D_conglomerates": int(D["conglomerate"].nunique()),
        "B_by_group": info.reindex(B["conglomerate"].unique())["first_source"].value_counts().to_dict(),
        "B_never_on": sorted(set(B["conglomerate"]) - set(info.index[info["first_q"].notna()])),
        "den_share_min_B": {c: round(float(s.min()), 3) for c, s in B.groupby("conglomerate")["den_share"]
                            if s.min() < 0.6},
        "N_source_B": B.drop_duplicates("conglomerate")["N_source"].value_counts().to_dict(),
        "adults_2020_07": float(adults[pd.Period("2020-07", "M")]),
    }}
    rep["main_B"] = fit(B, ["post_on", "post"], boot=["post_on"], rng=rng)
    rep["split_B"] = fit(B, ["post_on_mand", "post_on_vol", "post"], boot=["post_on_mand", "post_on_vol"], rng=rng)
    rep["D_for_scale"] = fit(D, ["post_on", "post"], boot=["post_on"], rng=rng)
    rep["B_no_N"] = fit(B, ["post_on", "post"], lhs="ln_o_noN")
    rep["B_no_caixa"] = fit(B[B["conglomerate"] != CAIXA], ["post_on", "post"], boot=["post_on"], rng=rng)
    rep["B_conglomerate_trends"] = fit(B, ["post_on", "post"], boot=["post_on"], rng=rng, trends=True)
    rep["D_conglomerate_trends"] = fit(D, ["post_on", "post"], trends=True)
    Bu = B[B["usage_2021"].notna()].copy()
    zu = Bu.drop_duplicates("conglomerate")["usage_2021"]
    Bu["z_usage"] = (Bu["usage_2021"] - zu.mean()) / zu.std()
    Bu["post_on_usage"] = Bu["post_on"] * Bu["z_usage"]
    rep["route3_B"] = fit(Bu, ["post_on", "post_on_usage", "post"], boot=["post_on_usage"], rng=rng)
    rep["route3_B"]["usage_2021_payers_pc"] = {"mean": round(float(zu.mean()), 4), "sd": round(float(zu.std()), 4),
                                               "n_conglomerates": int(len(zu))}
    ev = B[B["at_launch"] | B["first_on_month"].isna()]
    rep["event_B_launch_vs_never"] = event_study(ev)
    means = B.groupby(["at_launch", "post"])["ln_o"].mean()
    rep["pre_post_means_B"] = {f"{'launch' if a else 'late_or_never'}_post{p}": round(float(v), 4)
                               for (a, p), v in means.items()}
    rep["late_joiners_B"] = {c: str(info.loc[c, "first_on_month"]) for c in B["conglomerate"].unique()
                             if c in info.index and pd.notna(info.loc[c, "first_on_month"])
                             and not info.loc[c, "at_launch"]}
    rep["_meta"] = {"written": datetime.datetime.now().isoformat(timespec="seconds"), "seed": SEED,
                    "n_boot": N_BOOT, "window": [str(FIRST), str(LAST)], "launch": str(LAUNCH)}
    OUT_JSON.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    for k in ("main_B", "split_B", "D_for_scale", "B_no_N", "B_no_caixa", "B_conglomerate_trends",
              "D_conglomerate_trends", "route3_B"):
        print(k, {r: v for r, v in rep[k]["coef"].items()}, rep[k]["n_obs"], rep[k]["n_conglomerates"])
    print(f"wrote {OUT_PANEL}\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
