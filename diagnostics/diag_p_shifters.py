"""
diag_p_shifters.py
==================
Tasks P1 to P3 of claude/pix_handoff_2026-11-03c.md: the shifters and doses that the net-flow
regression (K1) of the design note (Drafts/Deposit Competition/kappa_design_2026-11-02.md) needs.
Nothing in the pipeline is edited and nothing here is a paper number.

  p1  The Pix-use dose and the exposures at the MCA, on the sample of the paper's Section 7.1.2
      (make_pix_exposure.py, tab_pix_exposure_fs).
        U_m     the 2021 mean of Pix payers per inhabitant (payers_pc of
                Rout/pix_usage_mca_quarter.parquet: distinct individual payers, the quarter's mean
                of monthly counts, over the MCA's population), and its path 2020Q4-2024Q4
        Mob_m   2019 mobile lines per 100 inhabitants (connections_per100, 2019 mean)
        G45_m   the 2019 4G/5G share of mobile accesses (frac_4g5g, 2019 mean)
      Each is standardized across the sample's B markets with population-weighted moments, the
      weights being 2019 population (make_pix_exposure.wstats). Mob and G45 use the moments over
      every MCA with a 2019 exposure, as 7.1.2 does. First stages:
        (i)  the 7.1.2 regression, quarterly payers per inhabitant on the standardized exposure,
             reproduced with make_pix_exposure.first_stage as the check that the sample is the same
        (ii) z(U_m) on z(Mob_m) across MCAs, with 2019 Cadunico and log 2019 population, 2019
             population weights; heteroskedasticity-robust (one row per MCA, so MCA clusters) and
             clustered by state. Because U_m Post_t and Mob_m Post_t are both constant within the
             post period, (ii) is the first stage of U_m Post_t on Mob_m Post_t with market and
             quarter effects, which is how K1 uses it.
  p2  The awareness shifters.
        adv_jt  accrued advertising expense (Cosif account 81745009, LAI 18810.023084 for
                2013-2024) of the conglomerate's member institutions summed: source
                cosif_institution, measure adv, of the advertising panel (build_advertising_panel.py),
                the series of the paper's 7.3 and appendix. The conglomerate document
                (cosif_conglomerate) sits beside it and is never spliced in. Nominal and in 2024Q4
                reais; over the conglomerate's deposits in the panel (types 1, 2, 4 and 5 summed
                over its cells, and the same without type 4, which holds judicial deposits until
                task I9). A quarter counts as observed under the statuses diag_advertising_in_
                estimation.py accepts, and not when flagged incomplete_quarter.
        branches_jm,t-1  ESTBAN branches of the conglomerate in the MCA one calendar quarter
                earlier (PANEL_INTERMED/estban_own_branches.csv, the sidecar of
                panel_estban_instrument.py), on the panel's B cells: zero where the conglomerate
                files ESTBAN at t-1 but not in that MCA, missing where it files nowhere at t-1 or
                where t-1 precedes the sidecar (2016Q1).
  p3  The stakes of the design note's eq. (11), by MCA-quarter, over the market's B banks:
        S_mt = sd_w(rho_j4mt) x bal_mt
      sd_w   the standard deviation of the banks' annual time-deposit spreads (spread_ann_a4 of
             the cell), weighted by their type-4 balances at t; banks with a positive type-4
             balance and a spread; missing with fewer than two such banks
      bal_mt the market's mean balance per relationship, sum_j Dep_jmt / sum_j (Dep_jmt / b_jt),
             with Dep types 1, 2 and 4 and b_jt = Dep_jt / N_jt the conglomerate's deposits
             (national, summed over its cells) per CCS relationship. N_jt is the stock AW1 built
             (make_ccs_openings_pix.stock: the BCB complaints-ranking client count, PF and PJ,
             carried by net flows) at the quarter's last month, used only where it rests on a
             ranking anchor; elsewhere b_jt is the median of the anchored B banks that quarter.
             The ranking starts after 2015Q4, so the 2015Q4 lag takes N of 2016-01.
      bal_cell_mt  the literal "per cell" reading: the market's deposits over its number of B
             banks, with S_cell_mt = sd_w x bal_cell_mt beside S
      Instrument: S~_mt = sd_w(rho~_jt) x bal_m,t-1, weights the type-4 balances at t-1, with
      rho~_jt = z_jt' beta the spread fitted from the paper's cost shifters (utils.demand_chars.
      COST_IV: personnel, administrative and tax expenses over assets, lagged) in a
      conglomerate-quarter WLS of spread_ann_a4_national on them with quarter effects, weights the
      national type-4 balance (the quarter effects cancel inside a market-quarter). The
      dispersions of each cost shifter are written beside it. The share of the type-4 weight in
      cells whose type-4 rate is a fallback (RT1: national rate equal to the Selic to 1e-6, or
      zero) is carried as a flag. A diagnostic, not a K1 coefficient: S on S~ with market and
      quarter effects, MCA clusters, in levels and logs, and log sd_w on log sd_w(rho~) alone,
      since S and S~ share the mean balance; and the same with rho~ shuffled across conglomerates
      within the quarter (a placebo that keeps the lagged weights and drops the cost shifters).
      S uses type-4 stocks and spreads, so it changes with the rebuild (I1, I6, I8, I9). Rerun
      `--only p3` when the new panel lands; the md5 of the panel used is written in every row.

Inputs (read only): processed/market_panel.parquet, read once with pyarrow, only the columns the
selected tasks need (load_panel_cached may rewrite its sidecar, so it is not used); for 2015 rows,
processed/demand_chars.parquet's r__dep_a1/2/4 (the raw-ESTBAN scale the preps use for 2016Q1's
lag, utils/demand_chars.restore_2016q1_lags), only when it was built from the same panel;
Rout/pix_usage_mca_quarter.parquet; Awareness/processed/advertising_panel_quarterly.parquet;
PANEL_INTERMED/estban_own_branches.csv; the CCS inputs of make_ccs_openings_pix.py.

Outputs (Rout): p1_pix_dose_exposure_mca.csv, p1_pix_use_path_mca_quarter.csv;
p2_advertising_conglomerate_quarter.csv, p2_advertising_coverage_B.csv,
p2_branches_conglomerate_mca_quarter.csv; p3_stakes_mca_quarter.csv, p3_cost_shifter_fit.csv;
p_shifters.json (the numbers the notes quote, one block per task).

Usage:  python -I diagnostics/diag_p_shifters.py [--only p1,p2,p3]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/ and pipeline modules
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import paths
from utils import demand_chars

PANEL = paths.PROCESSED / "market_panel.parquet"
USAGE = paths.rout_dir() / "pix_usage_mca_quarter.parquet"
ADV = paths.AWARENESS_PROC / "advertising_panel_quarterly.parquet"
BRANCHES = paths.PROCESSED / "PANEL_INTERMED" / "estban_own_branches.csv"
OUT = paths.rout_dir()
JSON = OUT / "p_shifters.json"
T0, T1 = (2016, 1), (2024, 4)
COST = list(demand_chars.COST_IV)
# As diag_advertising_in_estimation.OBSERVED: an imputed cell counts as observed.
OBSERVED = ("observed_positive", "observed_zero", "observed_negative", "imputed_from_row_total")
EXPOSURES = ("mobile", "g45")
SEED = 20261103
KEYS_OUT = {"j": "CodConglomeradoPrudencial", "m": "mca_code"}   # the panel's names, in the files
COLS = {
    "p1": ["mca_code", "is_B", "year", "connections_per100", "frac_4g5g",
           "cadunico_families_per1000", "pop_total"],
    "p2": ["CodConglomeradoPrudencial", "NomeInstituicao", "mca_code", "is_B", "year", "quarter",
           "dep_a1", "dep_a2", "dep_a4", "dep_a5"],
    "p3": ["CodConglomeradoPrudencial", "mca_code", "is_B", "year", "quarter", "dep_a1", "dep_a2",
           "dep_a4", "spread_ann_a4", "spread_ann_a4_national", "rate_a4_national",
           "risk_free_qoq", *COST],
}


def stamp(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def qi(year, quarter):
    return np.asarray(year, dtype=int) * 4 + np.asarray(quarter, dtype=int) - 1


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def wmoments(x, w):
    mu = float(np.average(x, weights=w))
    return mu, float(np.sqrt(np.average((x - mu) ** 2, weights=w)))


def wsd_by(df, keys, x, w):
    """Weighted mean and standard deviation of column x within groups, weights w (population
    formula), and the number of rows that enter."""
    d = df[keys].copy()
    d["sw"], d["swx"], d["swx2"] = df[w], df[w] * df[x], df[w] * df[x] ** 2
    d["n"] = 1
    g = d.groupby(keys)[["sw", "swx", "swx2", "n"]].sum()
    mu = g["swx"] / g["sw"]
    var = (g["swx2"] / g["sw"] - mu ** 2).clip(lower=0.0)
    return pd.DataFrame({"mean": mu, "sd": np.sqrt(var), "n": g["n"]})


# ------------------------------------------------------------------------------------------ inputs
def load_panel(cols):
    got = md5(PANEL)
    df = pq.read_table(PANEL, columns=cols, filters=[("year", ">=", 2015)]).to_pandas()
    df = df.rename(columns={"CodConglomeradoPrudencial": "j", "mca_code": "m"})
    df["m"] = df["m"].astype(str)
    if "j" in df:
        df["j"] = df["j"].astype(str)
        if df.duplicated(["j", "m", "year", "quarter"]).any():
            raise SystemExit("panel not unique on conglomerate x market x quarter")
    note = "no deposit columns read"
    if "dep_a1" in df:
        side = (pq.read_schema(demand_chars.chars_path()).metadata or {}).get(
            demand_chars.PANEL_MD5_KEY, b"").decode()
        if side == got:
            r = pq.read_table(demand_chars.chars_path(),
                              columns=["CodConglomeradoPrudencial", "mca_code", "year", "quarter",
                                       "r__dep_a1", "r__dep_a2", "r__dep_a4"],
                              filters=[("year", "==", 2015)]).to_pandas()
            r = r.rename(columns={"CodConglomeradoPrudencial": "j", "mca_code": "m"})
            r["j"], r["m"] = r["j"].astype(str), r["m"].astype(str)
            df = df.merge(r, on=["j", "m", "year", "quarter"], how="left", validate="one_to_one")
            pre = (df["year"] < 2016) & df[["r__dep_a1", "r__dep_a2", "r__dep_a4"]].notna().any(axis=1)
            for k in (1, 2, 4):
                df[f"dep_a{k}"] = np.where(pre, df[f"r__dep_a{k}"], df[f"dep_a{k}"])
            df = df.drop(columns=["r__dep_a1", "r__dep_a2", "r__dep_a4"])
            note = f"{int(pre.sum()):,} 2015 rows take the side-file balances"
        else:
            # Without a matching side file the 2015 balances are on another scale: drop them, so
            # every 2016Q1 lag is missing rather than wrong.
            for k in (1, 2, 4, 5):
                if f"dep_a{k}" in df:
                    df.loc[df["year"] < 2016, f"dep_a{k}"] = np.nan
            note = (f"side file built from {side[:8] or 'unknown'}, not {got[:8]}: "
                    "2015 balances dropped, 2016Q1 lags missing")
    df["qi"] = qi(df["year"], df["quarter"]) if "quarter" in df else np.nan
    stamp(f"panel {got[:8]}: {len(df):,} rows (2015 on), {len(cols)} columns; {note}")
    return df, got


# ---------------------------------------------------------------------------------------------- P1
def p1(df):
    import statsmodels.api as sm
    from make_pix_exposure import first_stage
    u = pd.read_parquet(USAGE)
    u["mca_code"] = u["mca_code"].astype(str)
    y19 = df[(df["is_B"] == 1) & (df["year"] == 2019)]
    x = y19.groupby("m").agg(mobile=("connections_per100", "mean"), g45=("frac_4g5g", "mean"),
                             cad2019=("cadunico_families_per1000", "mean"),
                             pop2019=("pop_total", "mean"))
    x = x.dropna(subset=["mobile", "g45", "cad2019", "pop2019"])
    x = x[x["pop2019"] > 0]
    mom = {e: wmoments(x[e].to_numpy(), x["pop2019"].to_numpy()) for e in EXPOSURES}
    d = u.merge(x, left_on="mca_code", right_index=True, how="inner")
    d = d.dropna(subset=["pop_total", "cad2019", *EXPOSURES, "tx_pc", "payers_pc", "value_pc"])
    d = d[d["pop_total"] > 0].reset_index(drop=True)
    for e in EXPOSURES:
        d[f"z_{e}"] = (d[e] - mom[e][0]) / mom[e][1]
    repro = {}
    for cl in ("mca_code", "state"):
        for e in EXPOSURES:
            r = first_stage(d, [e], "payers", False, True, cl)[e]
            repro[f"{e}_{cl}"] = dict(slope=r["slope"], f=r["f"], n=int(len(d)))
        repro[f"joint_{cl}"] = first_stage(d, list(EXPOSURES), "payers", False, True, cl)["joint_f"]
    stamp(f"P1 7.1.2 check: payers on mobile F {repro['mobile_mca_code']['f']:.2f} (MCA), "
          f"{repro['mobile_state']['f']:.2f} (state); {d['mca_code'].nunique():,} MCAs, "
          f"{d['t'].nunique()} quarters, {len(d):,} rows")

    # U_m and the cross-section.
    U = d[d["year"] == 2021].groupby("mca_code")["payers_pc"].agg(["mean", "size"])
    xs = (x.loc[x.index.isin(d["mca_code"])].join(U.rename(columns={"mean": "U", "size": "n_q2021"}))
          .join(d.groupby("mca_code")["state"].first()))
    if xs["U"].isna().any() or (xs["n_q2021"] != 4).any():
        raise SystemExit("an MCA of the 7.1.2 sample lacks a 2021 quarter")
    w = xs["pop2019"].to_numpy()
    mu_u, sd_u = wmoments(xs["U"].to_numpy(), w)
    xs["z_U"] = (xs["U"] - mu_u) / sd_u
    for e in EXPOSURES:
        xs[f"z_{e}"] = (xs[e] - mom[e][0]) / mom[e][1]
    xs["log_pop2019"] = np.log(xs["pop2019"])

    def cs(zs, cad=True, cluster=None):
        X = sm.add_constant(xs[[*(f"z_{z}" for z in zs), *(["cad2019"] if cad else []), "log_pop2019"]])
        kw = (dict(cov_type="HC1") if cluster is None
              else dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(xs[cluster])[0]}))
        res = sm.WLS(xs["z_U"], X, weights=w).fit(**kw)
        out = {z: dict(slope=float(res.params[f"z_{z}"]), se=float(res.bse[f"z_{z}"]),
                       f=float(res.tvalues[f"z_{z}"] ** 2)) for z in zs}
        if len(zs) > 1:
            R = np.zeros((len(zs), len(res.params)))
            for i, z in enumerate(zs):
                R[i, list(res.params.index).index(f"z_{z}")] = 1.0
            out["joint_f"] = float(np.squeeze(res.f_test(R).fvalue))
        out["n"], out["r2"] = int(res.nobs), float(res.rsquared)
        return out

    fs = {"mobile_robust": cs(["mobile"]), "mobile_state": cs(["mobile"], cluster="state"),
          "mobile_robust_nocad": cs(["mobile"], cad=False),
          "g45_robust": cs(["g45"]), "g45_state": cs(["g45"], cluster="state"),
          "joint_robust": cs(list(EXPOSURES)), "joint_state": cs(list(EXPOSURES), cluster="state")}
    stamp(f"P1 U on Mob: slope {fs['mobile_robust']['mobile']['slope']:.3f}, F "
          f"{fs['mobile_robust']['mobile']['f']:.2f} (MCA) {fs['mobile_state']['mobile']['f']:.2f} "
          f"(state); joint with G45 {fs['joint_robust']['joint_f']:.2f} / {fs['joint_state']['joint_f']:.2f}")

    dose = xs.reset_index().rename(columns={"index": "mca_code", "m": "mca_code"})
    dose = dose[["mca_code", "state", "pop2019", "U", "z_U", "mobile", "z_mobile", "g45", "z_g45",
                 "cad2019"]].rename(columns={"pop2019": "w_pop2019", "U": "U_payers_pc_2021",
                                             "mobile": "mob_lines_per100_2019",
                                             "g45": "g45_share_2019"})
    dose.to_csv(OUT / "p1_pix_dose_exposure_mca.csv", index=False)
    path = d[["mca_code", "state", "year", "quarter", "n_months", "payers_pc", "pop_total"]].copy()
    zq = []
    for (y, q), g in path.groupby(["year", "quarter"]):
        m_, s_ = wmoments(g["payers_pc"].to_numpy(), g["pop_total"].to_numpy())
        zq.append(((g["payers_pc"] - m_) / s_).rename("z_payers_pc_q"))
    path = path.join(pd.concat(zq)).rename(columns={"pop_total": "w_pop_quarter"})
    path = path.merge(dose[["mca_code", "U_payers_pc_2021", "z_U"]], on="mca_code", how="left")
    path.sort_values(["mca_code", "year", "quarter"]).to_csv(
        OUT / "p1_pix_use_path_mca_quarter.csv", index=False)
    # How the cross-market ordering of use holds over the path (the timing check of the note).
    piv = path.pivot_table(index="mca_code", columns=["year", "quarter"], values="payers_pc")
    rank_corr = {f"{y}Q{q}": float(piv[(y, q)].rank().corr(dose.set_index("mca_code")["U_payers_pc_2021"]
                                                          .reindex(piv.index).rank()))
                 for (y, q) in piv.columns}
    nat = (path.assign(pw=path["payers_pc"] * path["w_pop_quarter"])
           .groupby(["year", "quarter"])[["pw", "w_pop_quarter"]].sum())
    nat_path = {f"{y}Q{q}": float(r["pw"] / r["w_pop_quarter"]) for (y, q), r in nat.iterrows()}
    return dict(sample=dict(n_mca=int(len(xs)), n_quarters=int(d["t"].nunique()), n_rows=int(len(d)),
                            n_states=int(xs["state"].nunique()), n_mca_exposure=int(len(x))),
                moments=dict(U=dict(mean=mu_u, sd=sd_u),
                             **{e: dict(mean=mom[e][0], sd=mom[e][1]) for e in EXPOSURES}),
                U_quantiles={str(q): float(xs["U"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                corr_U_mobile=float(np.corrcoef(xs["U"], xs["mobile"])[0, 1]),
                corr_U_g45=float(np.corrcoef(xs["U"], xs["g45"])[0, 1]),
                fs_712_payers=repro, fs_U_cross_section=fs,
                rank_corr_quarter_with_U=rank_corr, payers_pc_popweighted=nat_path)


# ---------------------------------------------------------------------------------------------- P2
def p2(df, got):
    a = pd.read_parquet(ADV)
    win = a["year"].between(T0[0], T1[0])

    def pick(source):
        s = a[(a["source"] == source) & (a["measure"] == "adv") & win].copy()
        if s.duplicated(["panel_code", "year", "quarter"]).any():
            raise SystemExit(f"advertising panel: {source} repeats a conglomerate-quarter")
        ok = s["value_status"].isin(OBSERVED) & ~s["flag_notes"].fillna("").str.contains("incomplete_quarter")
        s["amt"] = np.where(ok, s["amount_brl"], np.nan)
        s["amt_real"] = np.where(ok, s["amount_brl_real"], np.nan)
        return s[["panel_code", "year", "quarter", "amt", "amt_real", "value_status", "flag_notes"]]

    inst = pick("cosif_institution").rename(columns={
        "panel_code": "j", "amt": "adv_brl", "amt_real": "adv_brl_real_2024q4",
        "value_status": "adv_status", "flag_notes": "adv_flags"})
    cong = pick("cosif_conglomerate").rename(columns={
        "panel_code": "j", "amt": "adv_congl_doc_brl", "amt_real": "adv_congl_doc_brl_real_2024q4",
        "value_status": "adv_congl_doc_status", "flag_notes": "adv_congl_doc_flags"})
    w = df[(df["qi"] >= qi(*T0)) & (df["qi"] <= qi(*T1))].copy()
    for k in (1, 2, 4, 5):
        w[f"d{k}"] = w[f"dep_a{k}"]
    cq = w.groupby(["j", "year", "quarter"]).agg(
        name=("NomeInstituicao", "first"), is_B=("is_B", "max"),
        d1=("d1", "sum"), d2=("d2", "sum"), d4=("d4", "sum"), d5=("d5", "sum"),
        n_cells=("m", "size")).reset_index()
    cq["dep_brl"] = cq[["d1", "d2", "d4", "d5"]].sum(axis=1)
    cq["dep_ex4_brl"] = cq[["d1", "d2", "d5"]].sum(axis=1)
    cq = (cq.drop(columns=["d1", "d2", "d4", "d5"])
          .merge(inst, on=["j", "year", "quarter"], how="left")
          .merge(cong, on=["j", "year", "quarter"], how="left"))
    cq["adv_over_dep"] = cq["adv_brl"] / cq["dep_brl"].where(cq["dep_brl"] > 0)
    cq["adv_over_dep_ex4"] = cq["adv_brl"] / cq["dep_ex4_brl"].where(cq["dep_ex4_brl"] > 0)
    cq["panel_md5"] = got[:8]
    cq.rename(columns=KEYS_OUT).to_csv(OUT / "p2_advertising_conglomerate_quarter.csv", index=False)
    orphan = sorted(set(inst.loc[inst["adv_brl"].notna(), "j"]) - set(cq["j"]))

    # Coverage of the B conglomerates.
    B = cq[cq["is_B"] == 1].copy()
    B["obs"] = B["adv_brl"].notna()
    by_year = {}
    for y, g in B.groupby("year"):
        by_year[int(y)] = dict(n_congl=int(g["j"].nunique()),
                               n_congl_observed_all_quarters=int(g.groupby("j")["obs"].all().sum()),
                               n_congl_never_observed=int((~g.groupby("j")["obs"].any()).sum()),
                               dep_share_observed=float(g.loc[g["obs"], "dep_brl"].sum() / g["dep_brl"].sum()),
                               dep_share_observed_congl_doc=float(
                                   g.loc[g["adv_congl_doc_brl"].notna(), "dep_brl"].sum() / g["dep_brl"].sum()))
    dep19 = B[B["year"] == 2019].groupby("j")["dep_brl"].mean()
    rank = dep19.rank(ascending=False)
    cov = B.groupby("j").agg(name=("name", "first"), n_quarters=("obs", "size"),
                             n_observed=("obs", "sum"), n_positive=("adv_brl", lambda s: int((s > 0).sum())),
                             n_zero=("adv_brl", lambda s: int((s == 0).sum())),
                             n_congl_doc=("adv_congl_doc_brl", lambda s: int(s.notna().sum())),
                             median_adv_over_dep=("adv_over_dep", "median"))
    cov["dep_2019_bn"] = dep19.reindex(cov.index) / 1e9
    cov["rank_2019"] = rank.reindex(cov.index)
    cov["small_B"] = ~(cov["rank_2019"] <= 5)
    cov = cov.sort_values("dep_2019_bn", ascending=False)
    cov.rename_axis(KEYS_OUT["j"]).to_csv(OUT / "p2_advertising_coverage_B.csv")
    sm_ = cov[cov["small_B"]]
    small = dict(n=int(len(sm_)), n_full=int((sm_["n_observed"] == sm_["n_quarters"]).sum()),
                 n_none=int((sm_["n_observed"] == 0).sum()),
                 n_all_zero=int(((sm_["n_positive"] == 0) & (sm_["n_observed"] > 0)).sum()),
                 dep_share_2019_of_B=float(sm_["dep_2019_bn"].sum() / cov["dep_2019_bn"].sum()))
    stamp(f"P2 advertising: {len(cq):,} conglomerate-quarters; small B {small}")

    # Branches at t-1.
    br = pd.read_csv(BRANCHES, dtype={"CodConglomeradoPrudencial": str, "mca_code": str})
    br = br.rename(columns={"CodConglomeradoPrudencial": "j", "mca_code": "m"})
    br["qi"] = qi(br["year"], br["quarter"])
    first_qi = int(br["qi"].min())
    filers = set(map(tuple, br[["j", "qi"]].drop_duplicates().to_numpy()))
    cells = w[w["is_B"] == 1][["j", "m", "year", "quarter", "qi", "dep_a4"]].copy()
    lag = br[["j", "m", "qi", "own_br"]].assign(qi=br["qi"] + 1).rename(columns={"own_br": "branches_lag"})
    cells = cells.merge(lag, on=["j", "m", "qi"], how="left", validate="one_to_one")
    filed = np.array([(j, q - 1) in filers for j, q in zip(cells["j"], cells["qi"])])
    cells["source_quarter_missing"] = (cells["qi"] - 1 < first_qi).astype(int)
    cells["congl_not_filing_lag"] = ((~filed) & (cells["source_quarter_missing"] == 0)).astype(int)
    cells["branches_lag"] = np.where(cells["branches_lag"].notna(), cells["branches_lag"],
                                     np.where(filed, 0.0, np.nan))
    cells["panel_md5"] = got[:8]
    cells.drop(columns=["qi", "dep_a4"]).rename(columns=KEYS_OUT).to_csv(
        OUT / "p2_branches_conglomerate_mca_quarter.csv", index=False)
    c2 = cells[cells["source_quarter_missing"] == 0]
    nf = (c2[c2["congl_not_filing_lag"] == 1].groupby("j")["qi"].nunique().sort_values(ascending=False))
    bstats = dict(n_cells=int(len(cells)), first_sidecar_quarter=f"{first_qi // 4}Q{first_qi % 4 + 1}",
                  share_positive=float((c2["branches_lag"] > 0).mean()),
                  share_zero=float((c2["branches_lag"] == 0).mean()),
                  share_missing=float(c2["branches_lag"].isna().mean()),
                  type4_share_in_zero_branch_cells=float(c2.loc[c2["branches_lag"] == 0, "dep_a4"].sum()
                                                         / c2["dep_a4"].sum()),
                  not_filing_congl_quarters={j: int(n) for j, n in nf.head(15).items()},
                  by_year={int(y): dict(positive=float((g["branches_lag"] > 0).mean()),
                                        mean_branches=float(g["branches_lag"].mean()))
                           for y, g in c2.groupby("year")})
    stamp(f"P2 branches: {bstats['n_cells']:,} B cells; positive {bstats['share_positive']:.3f}, "
          f"zero {bstats['share_zero']:.3f}, missing {bstats['share_missing']:.4f}")
    return dict(advertising=dict(by_year=by_year, small_B=small, orphan_conglomerates=orphan,
                                 n_rows=int(len(cq))),
                branches=bstats)


# ---------------------------------------------------------------------------------------------- P3
def p3(df, got):
    import statsmodels.api as sm
    from make_ccs_openings_pix import flows_monthly, stock
    from make_xi_diagnostics import _codes, _resid
    b = df[df["is_B"] == 1].copy()
    b["dep"] = b[["dep_a1", "dep_a2", "dep_a4"]].fillna(0.0).sum(axis=1)
    b["dep4"] = b["dep_a4"].fillna(0.0)

    # b_jt: deposits per CCS relationship.
    s = stock(flows_monthly())
    s = s[(s["month"].dt.month % 3 == 0) | (s["month"] == pd.Period("2016-01", "M"))].copy()
    s["qi"] = np.where(s["month"] == pd.Period("2016-01", "M"), qi(2015, 4),
                       qi(s["month"].dt.year, (s["month"].dt.month - 1) // 3 + 1))
    s = s[s["N_source"] != "cumulated"].rename(columns={"conglomerate": "j"})[["j", "qi", "N", "N_source"]]
    nat = b.groupby(["j", "qi"])["dep"].sum().rename("dep_nat").reset_index()
    nat = nat.merge(s, on=["j", "qi"], how="left")
    nat["b_jt"] = nat["dep_nat"] / nat["N"].where(nat["N"] > 0)
    med = nat.groupby("qi")["b_jt"].median()
    nat["b_imputed"] = nat["b_jt"].isna()
    nat["b_jt"] = nat["b_jt"].fillna(nat["qi"].map(med))
    anchored = nat[(nat["qi"] >= qi(*T0)) & (nat["qi"] <= qi(*T1))]
    ccs_cov = {int(y): float(g.loc[~g["b_imputed"], "dep_nat"].sum() / g["dep_nat"].sum())
               for y, g in anchored.assign(year=anchored["qi"] // 4).groupby("year")}

    b = b.merge(nat[["j", "qi", "b_jt", "b_imputed"]], on=["j", "qi"], how="left")
    pos = b[b["dep"] > 0].copy()
    pos["n_hat"] = pos["dep"] / pos["b_jt"]
    pos["dep_imp"] = pos["dep"] * pos["b_imputed"]
    mk = pos.groupby(["m", "qi"]).agg(dep=("dep", "sum"), n_hat=("n_hat", "sum"),
                                      dep_imp=("dep_imp", "sum"), n_banks=("j", "nunique"))
    mk["bal"] = mk["dep"] / mk["n_hat"]
    mk["bal_imputed_share"] = mk["dep_imp"] / mk["dep"]
    mk["bal_cell"] = mk["dep"] / mk["n_banks"]

    # sd_w of the type-4 spread at t, weights the type-4 balance at t.
    c4 = b[(b["dep4"] > 0) & b["spread_ann_a4"].notna()].copy()
    rf = (c4["rate_a4_national"] - c4["risk_free_qoq"]).abs() < 1e-6
    c4["fallback_w"] = c4["dep4"] * (rf | (c4["rate_a4_national"] == 0)).astype(float)
    sd = wsd_by(c4, ["m", "qi"], "spread_ann_a4", "dep4").rename(
        columns={"mean": "mean_spread", "sd": "sd_spread", "n": "n_banks4"})
    sd["fallback_weight_share"] = (c4.groupby(["m", "qi"])["fallback_w"].sum()
                                   / c4.groupby(["m", "qi"])["dep4"].sum())

    # rho~_jt from the cost shifters, conglomerate x quarter.
    cqv = b.groupby(["j", "qi"]).agg(spread=("spread_ann_a4_national", "first"),
                                     dep4=("dep4", "sum"),
                                     **{c: (c, "first") for c in COST}).reset_index()
    fit = cqv[(cqv["qi"] >= qi(*T0)) & (cqv["qi"] <= qi(*T1)) & (cqv["dep4"] > 0)].dropna(
        subset=["spread", *COST])
    X = pd.concat([fit[COST], pd.get_dummies(fit["qi"], prefix="q", dtype=float)], axis=1)
    res = sm.WLS(fit["spread"], X, weights=fit["dep4"]).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(fit["j"])[0]})
    R = np.zeros((len(COST), len(res.params)))
    for i, c in enumerate(COST):
        R[i, list(res.params.index).index(c)] = 1.0
    beta = res.params[COST]
    cqv["rho_fit"] = cqv[COST].to_numpy() @ beta.to_numpy()
    # Placebo: rho~ shuffled across conglomerates within the quarter. Its dispersion keeps the
    # market's lagged weights and loses the cost shifters, so what it predicts is market structure.
    rng = np.random.default_rng(SEED)

    def shuffle(s):
        v = s.to_numpy(copy=True)
        k = ~np.isnan(v)
        v[k] = rng.permutation(v[k])
        return pd.Series(v, index=s.index)

    cqv["rho_perm"] = cqv.groupby("qi", group_keys=False)["rho_fit"].apply(shuffle)
    fitstats = dict(n=int(res.nobs), n_congl=int(fit["j"].nunique()),
                    coef={c: dict(b=float(res.params[c]), se=float(res.bse[c])) for c in COST},
                    joint_f=float(np.squeeze(res.f_test(R).fvalue)))
    pd.DataFrame({"shifter": COST, "coef": beta.to_numpy(), "se": res.bse[COST].to_numpy()}).to_csv(
        OUT / "p3_cost_shifter_fit.csv", index=False)

    # S~: dispersion of rho~ and of each shifter across the market's B banks, weights Dep_j4m,t-1.
    lag4 = b[["j", "m", "qi", "dep4"]].assign(qi=b["qi"] + 1).rename(columns={"dep4": "dep4_lag"})
    t = b[["j", "m", "qi"]].merge(lag4, on=["j", "m", "qi"], how="left").merge(
        cqv[["j", "qi", "rho_fit", "rho_perm", *COST]], on=["j", "qi"], how="left")
    t = t[t["dep4_lag"] > 0]
    tw = t.dropna(subset=["rho_fit"])
    sdt = wsd_by(tw, ["m", "qi"], "rho_fit", "dep4_lag").rename(
        columns={"mean": "mean_rho_fit", "sd": "sd_rho_fit", "n": "n_banks4_lag"})
    sdt["rho_fit_weight_share"] = (tw.groupby(["m", "qi"])["dep4_lag"].sum()
                                   / t.groupby(["m", "qi"])["dep4_lag"].sum())
    for c in COST:
        sdt[f"disp_{c}"] = wsd_by(t.dropna(subset=[c]), ["m", "qi"], c, "dep4_lag")["sd"]
    sdt["sd_rho_perm"] = wsd_by(t.dropna(subset=["rho_perm"]), ["m", "qi"], "rho_perm", "dep4_lag")["sd"]
    sh = t["dep4_lag"] / t.groupby(["m", "qi"])["dep4_lag"].transform("sum")
    sdt["hhi4_lag"] = (sh ** 2).groupby([t["m"], t["qi"]]).sum()

    out = mk.join(sd, how="outer").join(sdt, how="outer").reset_index()
    lagb = out[["m", "qi", "bal"]].assign(qi=out["qi"] + 1).rename(columns={"bal": "bal_lag"})
    out = out.merge(lagb, on=["m", "qi"], how="left")
    two = out["n_banks4"] >= 2
    out["S"] = np.where(two, out["sd_spread"] * out["bal"], np.nan)
    out["S_cell"] = np.where(two, out["sd_spread"] * out["bal_cell"], np.nan)
    out["S_tilde"] = np.where(out["n_banks4_lag"] >= 2, out["sd_rho_fit"] * out["bal_lag"], np.nan)
    out = out[(out["qi"] >= qi(*T0)) & (out["qi"] <= qi(*T1))].copy()
    out["year"], out["quarter"] = out["qi"] // 4, out["qi"] % 4 + 1
    out["sd_spread_pp"] = 100 * out["sd_spread"]
    out["sd_rho_fit_pp"] = 100 * out["sd_rho_fit"]
    out["panel_md5"] = got[:8]
    cols = ["m", "year", "quarter", "n_banks", "n_banks4", "sd_spread_pp", "mean_spread", "bal",
            "bal_imputed_share", "S", "bal_cell", "S_cell", "fallback_weight_share",
            "n_banks4_lag", "hhi4_lag", "bal_lag", "sd_rho_fit_pp", "rho_fit_weight_share", "S_tilde",
            *(f"disp_{c}" for c in COST), "panel_md5"]
    out = out.rename(columns={"m": "mca_code"})
    cols[0] = "mca_code"
    out[cols].sort_values(["mca_code", "year", "quarter"]).to_csv(OUT / "p3_stakes_mca_quarter.csv", index=False)

    # Diagnostic: S on S~, MCA and quarter effects, MCA clusters (levels and logs); and the
    # dispersion alone (log sd on log sd~), since S and S~ share the mean balance.
    # The last spec adds the lagged market structure (log HHI of type-4 balances, log number of
    # banks) as controls: what the cost shifters add beyond it.
    diag = {}
    for lab, (a_, b_, f, ctrl) in (
            ("level", ("S", "S_tilde", lambda v: v, False)), ("log", ("S", "S_tilde", np.log, False)),
            ("sd_only_log", ("sd_spread", "sd_rho_fit", np.log, False)),
            ("sd_only_log_placebo", ("sd_spread", "sd_rho_perm", np.log, False)),
            ("sd_only_log_ctrl_structure", ("sd_spread", "sd_rho_fit", np.log, True))):
        d = out[(out[a_] > 0) & (out[b_] > 0) & (out["S"] > 0) & (out["S_tilde"] > 0)]
        y, z = f(d[a_].to_numpy()), f(d[b_].to_numpy())
        cols_ = [y, z] + ([np.log(d["hhi4_lag"].to_numpy()), np.log(d["n_banks4_lag"].to_numpy())]
                          if ctrl else [])
        w = np.ones(len(d))
        Rr = _resid(np.column_stack(cols_), w, [_codes(d["mca_code"]), _codes(d["qi"])])
        r = sm.OLS(Rr[:, 0], Rr[:, 1:]).fit(cov_type="cluster",
                                            cov_kwds={"groups": pd.factorize(d["mca_code"])[0]})
        diag[lab] = dict(slope=float(r.params[0]), se=float(r.bse[0]), f=float(r.tvalues[0] ** 2),
                         n=int(len(d)), corr_raw=float(np.corrcoef(y, z)[0, 1]))
    q = lambda s: {str(p): float(s.quantile(p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
    ok = out[out["S"].notna()]
    by_year = {int(y): dict(sd_spread_pp_median=float(g["sd_spread_pp"].median()),
                            bal_median=float(g["bal"].median()),
                            S_median=float(g["S"].median()),
                            S_weighted_mean=float(np.average(g["S"], weights=g["dep"])))
               for y, g in ok.groupby("year")}
    stats = dict(n_mq=int(len(out)), n_mq_S=int(out["S"].notna().sum()),
                 n_mq_one_bank=int((out["n_banks4"] < 2).sum()),
                 S_quantiles=q(ok["S"]), sd_spread_pp_quantiles=q(ok["sd_spread_pp"]),
                 bal_quantiles=q(ok["bal"]), bal_cell_quantiles=q(ok["bal_cell"]),
                 S_cell_quantiles=q(ok["S_cell"]),
                 corr_S_S_cell=float(np.corrcoef(np.log(ok.loc[ok["S"] > 0, "S"]),
                                                 np.log(ok.loc[ok["S"] > 0, "S_cell"]))[0, 1]),
                 bal_imputed_share_mean=float(ok["bal_imputed_share"].mean()),
                 ccs_coverage_of_B_deposits=ccs_cov,
                 fallback_weight_share_mean=float(ok["fallback_weight_share"].mean()),
                 mq_with_fallback_weight=int((ok["fallback_weight_share"] > 0).sum()),
                 rho_fit_weight_share_mean=float(out["rho_fit_weight_share"].mean()),
                 by_year=by_year, cost_fit=fitstats, S_on_S_tilde=diag)
    stamp(f"P3: {stats['n_mq_S']:,} of {stats['n_mq']:,} market-quarters with S; median S "
          f"R$ {stats['S_quantiles']['0.5']:.1f}/yr; cost fit joint F {fitstats['joint_f']:.2f}; "
          f"S on S~ F {diag['level']['f']:.2f} (level), {diag['log']['f']:.2f} (log), "
          f"{diag['sd_only_log']['f']:.2f} (log sd on log sd~), "
          f"{diag['sd_only_log_placebo']['f']:.2f} (placebo, rho~ shuffled), "
          f"{diag['sd_only_log_ctrl_structure']['f']:.2f} (controlling for HHI and n banks at t-1)")
    return stats


# -------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="p1,p2,p3")
    tasks = [t.strip() for t in ap.parse_args().only.split(",") if t.strip()]
    if not tasks or set(tasks) - set(COLS):
        raise SystemExit(f"--only takes a comma list of {sorted(COLS)}")
    cols = list(dict.fromkeys(c for t in tasks for c in COLS[t]))
    df, got = load_panel(cols)
    prev = json.loads(JSON.read_text(encoding="utf-8")) if JSON.exists() else {}
    if "p1" in tasks:
        prev["p1"] = p1(df)
    if "p2" in tasks:
        prev["p2"] = p2(df, got)
    if "p3" in tasks:
        prev["p3"] = p3(df, got)
    for t in tasks:
        prev[t]["_meta"] = dict(written=datetime.datetime.now().isoformat(timespec="seconds"),
                                panel_md5=got, panel_columns=cols)
    JSON.write_text(json.dumps(prev, indent=2, default=float), encoding="utf-8")
    stamp(f"wrote {JSON.name} ({', '.join(tasks)})")


if __name__ == "__main__":
    main()
