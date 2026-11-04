"""
diag_k0_netflow_objects.py
==========================
Task K0 of claude/pix_handoff_2026-11-03.md: the objects of the net-flow regression of the design
note (Drafts/Deposit Competition/kappa_design_2026-11-02.md, fourteenth version), built and
described. Nothing is regressed. B cells, 2016Q1-2024Q4.

  a  the net flow NF_jkmt = Dep_jkmt - (1 + r_{t-1} - rho_jkm,t-1) Dep_jkm,t-1 for k = 1, 2, 4, with
     the lagged rate the panel carries (risk_free_qoq - spread_ak of the same cell one calendar
     quarter earlier, as the demand preps form gross_return_lag); its moving share
     |NF| / ((1 + r - rho) Dep_{t-1}), the same share over the awake carry (1 - phi_mt) of the
     adopted fit, and the paper's truncated active balance (11-B) for comparison.
  b  the pool weights of the destination (j, m, t) from lagged stocks, B rivals only:
     Q1 = Dep_j1 / Dep_j2, Q4 = Dep_j4 / Dep_j2, Rk = sum_{j' != j} Dep_j'k / Dep_j2 (k = 1, 2, 4),
     and the variance decomposition of log R2.
  c  the savings spread rho_j2t by conglomerate and quarter, its dispersion across the
     savings-holding B conglomerates, and the gap between a bank's own savings spread and the
     rival-pool-weighted one (the vintage of the savings stock, design note Step 3). The panel's
     rate_a2 is the regulated new-rule rate of utils/macro_rates, one number per quarter, so both
     are zero by construction; `vintage_bias` sizes where an old-rule premium goes instead (the
     type-4 rate, through the residual it is built from).
  d  the outside pool O_mt = M_hat_mt - sum over B banks and types of Dep_lkm,t-1, with M_hat_mt the
     market size the adopted demand prep stored (M_mt), and o_mt = O_mt / sum_l Dep_l2m,t-1.
  e  omega_jt, the share of time deposits maturing within 90 days, from the banks' statements, with
     the fill rule for the rest and the coverage of the B type-4 stock.

Inputs (read only):
  - processed/market_panel.parquet, read once with pyarrow, the columns in COLS (load_panel_cached
    may rewrite its sidecar, so it is not used). Its md5 must start with PANEL_MD5: d01b8bae, the
    10-27 panel, unless the environment variable K0_PANEL_MD5 names another (the rebuilt panel).
  - processed/demand_chars.parquet: the 2013-2015 balances on the scale of the raw ESTBAN files
    (r__dep_a1/2/4), which give 2016Q1 its lag exactly as utils/demand_chars.restore_2016q1_lags
    does in the preps. The file records the md5 of the panel it was built from; it must match.
  - ESTIMATION_OUTPUT/DEMAND_PREP/demand_3_index_spec_12.parquet: phi_mt and M_mt of routine (III)
    (E3, the single index; spec 12, config/routines.toml), both constant within market-quarter.
    The frame records no panel md5, so load_demand refuses a frame older than the panel file: a
    rebuilt panel needs a rerun demand prep before K0 or K2 reads phi_mt and M_mt.
  - raw/DEPOSIT_MATURITY/deposit_maturity_summary.csv (scrape_deposit_maturity.py summary).
Outputs (Rout):
  - k0_netflow_panel.parquet, k0_netflow_summary.csv                          (a)
  - k0_pool_weights.parquet, k0_pool_weights_summary.csv,
    k0_logR2_variance_decomposition.csv                                       (b)
  - k0_savings_spread_dispersion.csv, k0_savings_spread_congl_quarter.csv,
    k0_savings_vintage_bias.csv                                               (c)
  - k0_outside_pool.csv                                                       (d)
  - k0_omega.csv, k0_omega_coverage.csv                                       (e)
  - k0_numbers.json: every number the report quotes.
Flags carried on the cell files (nothing is corrected): rate_fallback (the conglomerate's national
type-k rate equals the Selic to 1e-6 or is zero, at t or t-1; RT1), judicial (BB and Caixa type-4
cells; BB3, JD1), own_sav_lag_ok (Dep_j2m,t-1 > 0), one_B_bank (one B bank in the market at t-1).
Usage:  python -I diagnostics/diag_k0_netflow_objects.py
        K0_PANEL_MD5=<first 8 of the new md5> python -I diagnostics/diag_k0_netflow_objects.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import paths
from utils import demand_chars

PANEL = paths.PROCESSED / "market_panel.parquet"
PANEL_MD5 = os.environ.get("K0_PANEL_MD5", "d01b8bae")   # checked in load_panel; K2 records it
DEMAND = paths.demand_parquet_dir() / "demand_3_index_spec_12.parquet"
MATURITY = paths.RAW / "DEPOSIT_MATURITY" / "deposit_maturity_summary.csv"
OUT = paths.rout_dir()

COLS = ["CodConglomeradoPrudencial", "NomeInstituicao", "mca_code", "is_B", "year", "quarter",
        "dep_a1", "dep_a2", "dep_a4", "dep_a5", "spread_a1", "spread_a2", "spread_a4",
        "spread_ann_a2", "spread_ann_a4", "risk_free_qoq", "risk_free_ann", "rate_a2",
        "rate_a4_national"]
T0, T1 = (2016, 1), (2024, 4)
TYPES = (1, 2, 4)
BB, CAIXA = "C0080329", "C0080738"
# Statement-note firm keys -> prudential conglomerate (the map of switch_cost_tools.FIRM_TO_CONGL;
# that module is not imported because it points the output tree at its sandbox on import).
FIRM_TO_CONGL = {
    "abc": "C0080312", "banrisul": "C0080154", "bb": "C0080329", "bmg": "C0080178",
    "bradesco": "C0080075", "btg": "C0080336", "daycoval": "C0081744", "itau": "C0080099",
    "nubank": "C0084693", "pan": "C0080257", "santander_br": "C0080185",
}
QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def stamp(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def qi(year, quarter):
    return np.asarray(year, dtype=int) * 4 + np.asarray(quarter, dtype=int) - 1


def tid(q):
    q = np.asarray(q, dtype=int)
    return pd.Series([f"{a}Q{b}" for a, b in zip(q // 4, q % 4 + 1)])


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ inputs
def load_panel():
    got = md5(PANEL)
    side = (pq.read_schema(demand_chars.chars_path()).metadata or {}).get(demand_chars.PANEL_MD5_KEY, b"").decode()
    if not got.startswith(PANEL_MD5) or side != got:
        raise SystemExit(f"panel md5 {got[:8]} (want {PANEL_MD5}); side file built from {side[:8]}")
    df = pq.read_table(PANEL, columns=COLS).to_pandas()
    df = df.rename(columns={"CodConglomeradoPrudencial": "j", "mca_code": "m"})
    df["j"], df["m"] = df["j"].astype(str), df["m"].astype(str)
    if df.duplicated(["j", "m", "year", "quarter"]).any():
        raise SystemExit("panel not unique on conglomerate x market x quarter")
    df["qi"] = qi(df["year"], df["quarter"])
    # The 2013-2015 balances on the raw ESTBAN scale (utils/demand_chars.restore_2016q1_lags).
    r = pq.read_table(demand_chars.chars_path(),
                      columns=["CodConglomeradoPrudencial", "mca_code", "year", "quarter",
                               "r__dep_a1", "r__dep_a2", "r__dep_a4"]).to_pandas()
    r = r[r["year"] < 2016].rename(columns={"CodConglomeradoPrudencial": "j", "mca_code": "m"})
    r["j"], r["m"] = r["j"].astype(str), r["m"].astype(str)
    df = df.merge(r, on=["j", "m", "year", "quarter"], how="left", validate="one_to_one")
    pre = (df["year"] < 2016) & df[["r__dep_a1", "r__dep_a2", "r__dep_a4"]].notna().any(axis=1)
    n_restored = int(pre.sum())
    for k in TYPES:
        df[f"dep_a{k}"] = np.where(pre, df[f"r__dep_a{k}"], df[f"dep_a{k}"])
    df = df.drop(columns=["r__dep_a1", "r__dep_a2", "r__dep_a4"])
    stamp(f"panel {got[:8]}: {len(df):,} rows; {n_restored:,} pre-2016 rows take the side-file balances")
    return df


def rate_flags(df):
    """Conglomerate-quarter fallback flags (RT1): the national type-k rate equals the Selic to 1e-6
    or is zero. k = 2 from rate_a2, k = 4 from rate_a4_national; checking pays zero by construction."""
    cq = df.groupby(["j", "qi"]).agg(rf=("risk_free_qoq", "first"), r2=("rate_a2", "first"),
                                     r4=("rate_a4_national", "first"),
                                     dep2=("dep_a2", "sum"), dep4=("dep_a4", "sum")).reset_index()
    out = {}
    for k, c, d in ((2, "r2", "dep2"), (4, "r4", "dep4")):
        f = ((cq[c] - cq["rf"]).abs() < 1e-6) | (cq[c] == 0)
        out[k] = set(map(tuple, cq.loc[f & (cq[d] > 0), ["j", "qi"]].to_numpy()))
    return out


def load_demand():
    if DEMAND.stat().st_mtime < PANEL.stat().st_mtime:
        raise SystemExit(f"{DEMAND.name} ({datetime.fromtimestamp(DEMAND.stat().st_mtime):%Y-%m-%d %H:%M}) "
                         f"is older than {PANEL.name} ({datetime.fromtimestamp(PANEL.stat().st_mtime):%Y-%m-%d %H:%M}): "
                         "rerun the demand prep on this panel before reading phi_mt and M_mt")
    d = pq.read_table(DEMAND, columns=["mca_code", "time_id", "is_B", "phi_mt", "M_mt", "Dep_Act",
                                       "_1mphi"]).to_pandas()
    d = d[d["is_B"]].rename(columns={"mca_code": "m"})
    d["m"] = d["m"].astype(str)
    floor = (d.assign(x=d["Dep_Act"] / d["_1mphi"]).groupby(["m", "time_id"])["x"].sum() / 0.95)
    mk = d.groupby(["m", "time_id"]).agg(phi_mt=("phi_mt", "first"), M_hat=("M_mt", "first"))
    mk["floor_binds"] = (mk["M_hat"] <= floor.reindex(mk.index) * (1 + 1e-9)).astype(int)
    return mk.reset_index()


# ------------------------------------------------------------------------------------------- (a)
def netflow(b, flags, mk):
    parts = []
    for k in TYPES:
        x = b[["j", "m", "qi", "year", "quarter", f"dep_a{k}", f"spread_a{k}", "risk_free_qoq"]]
        x = x.rename(columns={f"dep_a{k}": "Dep", f"spread_a{k}": "spread"})
        x = x[x["Dep"].notna()].assign(k=k)
        parts.append(x)
    L = pd.concat(parts, ignore_index=True)
    lag = L[["j", "m", "k", "qi", "Dep", "spread", "risk_free_qoq"]].copy()
    lag["qi"] += 1
    L = L.merge(lag.rename(columns={"Dep": "Dep_lag", "spread": "spread_lag", "risk_free_qoq": "rf_lag"}),
                on=["j", "m", "k", "qi"], how="left")
    L = L[(L["qi"] >= qi(*T0)) & (L["qi"] <= qi(*T1))].copy()
    L["rate_lag"] = L["rf_lag"] - L["spread_lag"]
    L["carry"] = (1 + L["rate_lag"]) * L["Dep_lag"]
    L["NF"] = L["Dep"] - L["carry"]
    L["t"] = tid(L["qi"]).to_numpy()
    L = L.merge(mk[["m", "time_id", "phi_mt"]].rename(columns={"time_id": "t"}), on=["m", "t"], how="left")
    L["act_raw"] = L["Dep"] - L["phi_mt"] * L["carry"]
    L["Dep_Act"] = np.maximum(0.0, L["act_raw"])
    fl = np.zeros(len(L), dtype=bool)
    for k in (2, 4):
        s = flags[k]
        on_k = (L["k"] == k).to_numpy()
        hit_t = np.fromiter(((a, q) in s for a, q in zip(L["j"], L["qi"])), bool, len(L))
        hit_l = np.fromiter(((a, q - 1) in s for a, q in zip(L["j"], L["qi"])), bool, len(L))
        fl |= on_k & (hit_t | hit_l)
    L["rate_fallback"] = fl.astype(int)
    L["judicial"] = ((L["k"] == 4) & L["j"].isin([BB, CAIXA])).astype(int)
    return L


def netflow_summary(L):
    rows = []
    ok = L["NF"].notna() & (L["Dep_lag"] > 0) & (L["carry"] > 0)
    for (k, y), g in list(L[ok].groupby(["k", "year"])) + [((k, "all"), L[ok & (L["k"] == k)]) for k in TYPES]:
        ms = g["NF"].abs() / g["carry"]
        ph = g["phi_mt"].notna()
        aph = ph & (g["phi_mt"] < 1)
        awake = (1 - g["phi_mt"]) * g["carry"]
        aw = (g["NF"].abs() / awake)[aph]
        out = aph & (g["NF"] < 0)
        ar = g["Dep_Act"] / g["carry"]
        rows.append({
            "k": k, "year": y, "n_cells": len(g),
            "share_NF_negative": (g["NF"] < 0).mean(),
            "moving_mean": ms.mean(), "moving_median": ms.median(),
            "moving_p90": ms.quantile(0.9), "moving_p99": ms.quantile(0.99),
            "moving_stock_weighted": g["NF"].abs().sum() / g["carry"].sum(),
            "net_stock_weighted": g["NF"].sum() / g["carry"].sum(),
            "awake_moving_median": aw.median(), "awake_moving_p90": aw.quantile(0.9),
            "awake_moving_stock_weighted": g.loc[aph, "NF"].abs().sum() / awake[aph].sum(),
            "outflow_exceeds_awake_stock_share": (g.loc[out, "NF"].abs() > awake[out]).mean(),
            "outflow_exceeds_awake_stock_wshare": (g.loc[out & (g["NF"].abs() > awake), "carry"].sum()
                                                   / g.loc[out, "carry"].sum()),
            "n_phi": int(ph.sum()), "n_phi_one": int((g["phi_mt"] >= 1).sum()),
            "act11B_share_truncated": (g.loc[ph, "Dep_Act"] <= 1e-6).mean(),
            "act11B_ratio_mean": ar[ph].mean(), "act11B_ratio_median": ar[ph].median(),
            "act11B_ratio_p90": ar[ph].quantile(0.9), "act11B_ratio_p99": ar[ph].quantile(0.99),
            "act11B_stock_weighted": g.loc[ph, "Dep_Act"].sum() / g.loc[ph, "carry"].sum(),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- (b)
def pool_weights(b, L):
    w = b[["j", "m", "qi", "dep_a1", "dep_a2", "dep_a4"]].copy()
    for k in TYPES:
        w[f"d{k}"] = w[f"dep_a{k}"].fillna(0.0)
    own = w[["j", "m", "qi", "d1", "d2", "d4", "dep_a2"]].copy()
    own["qi"] += 1
    own = own.rename(columns={"d1": "own1", "d2": "own2", "d4": "own4", "dep_a2": "own2_raw"})
    tot = w.groupby(["m", "qi"]).agg(tot1=("d1", "sum"), tot2=("d2", "sum"), tot4=("d4", "sum"),
                                    nB=("j", "nunique")).reset_index()
    tot["qi"] += 1
    P = w[(w["qi"] >= qi(*T0)) & (w["qi"] <= qi(*T1))][["j", "m", "qi"]]
    P = P.merge(own, on=["j", "m", "qi"], how="left").merge(tot, on=["m", "qi"], how="left")
    P["has_own_lag"] = P["own2_raw"].notna() | P["own1"].notna() | P["own4"].notna()
    for c in ("own1", "own2", "own4", "tot1", "tot2", "tot4"):
        P[c] = P[c].fillna(0.0)
    P["nB"] = P["nB"].fillna(0).astype(int)
    P["own_sav_lag_ok"] = (P["own2"] > 0).astype(int)
    den = P["own2"].where(P["own2"] > 0)
    P["Q1"], P["Q4"] = P["own1"] / den, P["own4"] / den
    for k in TYPES:
        P[f"R{k}"] = (P[f"tot{k}"] - P[f"own{k}"]) / den
    P["one_B_bank"] = (P["nB"] == 1).astype(int)
    k4 = L[(L["k"] == 4) & L["NF"].notna()][["j", "m", "qi"]].assign(has_NF4=1)
    P = P.merge(k4, on=["j", "m", "qi"], how="left")
    P["has_NF4"] = P["has_NF4"].fillna(0).astype(int)
    P["t"] = tid(P["qi"]).to_numpy()
    P["year"] = P["qi"] // 4
    return P


def _demean(v, codes, n):
    s = np.bincount(codes, weights=v, minlength=n)
    c = np.bincount(codes, minlength=n)
    return v - (s / c)[codes]


def variance_decomposition(y, g_cell, g_mt, tol=1e-9, maxit=2000):
    y = np.asarray(y, float)
    c1, n1 = pd.factorize(g_cell)[0], g_cell.nunique()
    c2, n2 = pd.factorize(g_mt)[0], g_mt.nunique()
    V = y.var()
    within_cell = _demean(y, c1, n1)
    within_mt = _demean(y, c2, n2)
    r = y - y.mean()
    for it in range(maxit):
        r_new = _demean(_demean(r, c1, n1), c2, n2)
        if np.abs(r_new - r).max() < tol:
            break
        r = r_new
    return {"n": len(y), "n_cells": int(n1), "n_market_quarters": int(n2), "var_total": V,
            "share_between_cells": 1 - within_cell.var() / V,
            "share_within_cell_over_time": within_cell.var() / V,
            "share_within_market_quarter": within_mt.var() / V,
            "share_left_after_cell_and_mq_fe": r_new.var() / V, "iterations": it + 1}


def weights_summary(P):
    rows = []
    S = P[(P["has_NF4"] == 1)]
    for c in ("Q1", "Q4", "R1", "R2", "R4"):
        v = S[c]
        pos = v[v > 0]
        lv = np.log(pos)
        row = {"weight": c, "n_K1_rows": len(S), "n_defined": int(v.notna().sum()),
               "share_zero_or_negative": float((v.dropna() <= 0).mean())}
        row.update({f"ln_p{int(q * 100):02d}": lv.quantile(q) for q in QS})
        row.update({f"p{int(q * 100):02d}": pos.quantile(q) for q in (0.1, 0.5, 0.9)})
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- (c)
def savings_spread(b, flags, P):
    cq = b.groupby(["j", "qi"]).agg(name=("NomeInstituicao", "last"), sav=("dep_a2", "sum"),
                                    rho2=("spread_ann_a2", "first"), rf_ann=("risk_free_ann", "first"),
                                    year=("year", "first")).reset_index()
    cq = cq[(cq["qi"] >= qi(*T0)) & (cq["qi"] <= qi(*T1))].copy()
    cq["rho2_pp"] = 100 * cq["rho2"]
    cq["selic_pp"] = 100 * cq["rf_ann"]
    cq["rate_fallback"] = [int((a, q) in flags[2]) for a, q in zip(cq["j"], cq["qi"])]
    cq["t"] = tid(cq["qi"]).to_numpy()
    H = cq[(cq["sav"] > 0) & cq["rho2_pp"].notna()]
    rows = []
    for (q, fb), g in [((q, "all"), g) for q, g in H.groupby("qi")] + \
                      [((q, "no_fallback"), g[g["rate_fallback"] == 0]) for q, g in H.groupby("qi")]:
        x, wt = g["rho2_pp"], g["sav"]
        mu = np.average(x, weights=wt)
        rows.append({"t": tid([q])[0], "year": q // 4, "sample": fb, "n_congl": len(g),
                     "selic_pp": g["selic_pp"].iloc[0], "mean": x.mean(), "sd": x.std(),
                     "p25": x.quantile(0.25), "p75": x.quantile(0.75),
                     "iqr": x.quantile(0.75) - x.quantile(0.25),
                     "savings_weighted_mean": mu,
                     "savings_weighted_sd": np.sqrt(np.average((x - mu) ** 2, weights=wt))})
    D = pd.DataFrame(rows)
    # Own savings spread against the rival-pool-weighted one, on the K1 rows: rival weights are the
    # lagged savings stocks of every other B bank in the market, rates at t.
    rho = cq.set_index(["j", "qi"])["rho2_pp"]
    lagrows = b[["j", "m", "qi", "dep_a2"]].copy()
    lagrows["qi"] += 1
    lagrows["rho2_t"] = rho.reindex(pd.MultiIndex.from_frame(lagrows[["j", "qi"]])).to_numpy()
    lagrows = lagrows[(lagrows["dep_a2"].fillna(0) > 0) & lagrows["rho2_t"].notna()]
    lagrows["w"] = lagrows["dep_a2"] * lagrows["rho2_t"]
    S = lagrows.groupby(["m", "qi"]).agg(Sw=("w", "sum"), S=("dep_a2", "sum")).reset_index()
    K = P[(P["has_NF4"] == 1) & (P["own_sav_lag_ok"] == 1)][["j", "m", "qi", "own2"]]
    K = K.merge(S, on=["m", "qi"], how="left")
    K["rho2_own"] = rho.reindex(pd.MultiIndex.from_frame(K[["j", "qi"]])).to_numpy()
    K["rho2_rival"] = (K["Sw"] - K["own2"] * K["rho2_own"]) / (K["S"] - K["own2"])
    K["d_vintage"] = K["rho2_own"] - K["rho2_rival"]
    rho4 = b.set_index(["j", "m", "qi"])["spread_ann_a4"] * 100
    K["rho4_pp"] = rho4.reindex(pd.MultiIndex.from_frame(K[["j", "m", "qi"]])).to_numpy()
    K["g_own"] = K["rho2_own"] - K["rho4_pp"]
    K = K.replace([np.inf, -np.inf], np.nan)
    ok = K["d_vintage"].notna() & K["g_own"].notna()
    mt = pd.factorize(K.loc[ok, "m"] + "|" + K.loc[ok, "qi"].astype(str))
    vint = {"n": int(ok.sum()),
            "sd_d_vintage_pp": float(K.loc[ok, "d_vintage"].std()),
            "p05_d": float(K.loc[ok, "d_vintage"].quantile(0.05)),
            "p95_d": float(K.loc[ok, "d_vintage"].quantile(0.95)),
            "sd_g_own_pp": float(K.loc[ok, "g_own"].std()),
            "sd_d_within_mq": float(_demean(K.loc[ok, "d_vintage"].to_numpy(), mt[0], len(mt[1])).std()),
            "sd_g_within_mq": float(_demean(K.loc[ok, "g_own"].to_numpy(), mt[0], len(mt[1])).std())}
    by_year = {}
    for y, g in K[ok].groupby(K.loc[ok, "qi"] // 4):
        by_year[int(y)] = {"sd_d": float(g["d_vintage"].std()), "sd_g": float(g["g_own"].std())}
    vint["by_year"] = by_year
    return cq, D, vint


def vintage_bias(b):
    """Where an old-rule premium on savings would go. rate_a2 is the regulated NEW-rule rate
    (utils/macro_rates: TR + 70% of the Selic target when the target is at most 8.5%, else TR +
    0.5% a month), and the pre-2025 type-4 rate is the lumped Cosif funding expense less savings at
    that rate, over the lagged time-deposit stock (panel_deposit_rates.py, panel_cosif_calibrate.py
    strip). Deposits made before 2012-05-04 earn TR + 0.5% a month whatever the Selic, so a bank
    with old-rule share s of its savings has its type-4 rate overstated by
        s x premium x Dep_j2,t-1 / Dep_j4,t-1,   premium = old-rule rate - new-rule rate.
    premium_q = max(0, 1.005^3 - 1 - rate_a2_q), exact when TR = 0 (TR was zero from late 2017 to
    2021) and short by about 3 TR otherwise. Reported per unit of s, in pp a year."""
    cq = b.groupby(["j", "qi"]).agg(name=("NomeInstituicao", "last"), sav=("dep_a2", "sum"),
                                    time=("dep_a4", "sum"), r2=("rate_a2", "first"),
                                    sel=("risk_free_ann", "first")).reset_index()
    lag = cq[["j", "qi", "sav", "time"]].copy()
    lag["qi"] += 1
    cq = cq.merge(lag.rename(columns={"sav": "sav_lag", "time": "time_lag"}), on=["j", "qi"], how="left")
    cq = cq[(cq["qi"] >= qi(*T0)) & (cq["qi"] <= qi(*T1))].copy()
    cq["premium_ann_pp"] = 100 * np.clip(1.005 ** 12 - (1 + cq["r2"]) ** 4, 0, None)
    cq["sav_over_time_lag"] = cq["sav_lag"] / cq["time_lag"].where(cq["time_lag"] > 0)
    cq["bias_per_unit_old_share_pp"] = cq["premium_ann_pp"] * cq["sav_over_time_lag"]
    cq["t"] = tid(cq["qi"]).to_numpy()
    cq["year"] = cq["qi"] // 4
    return cq[["j", "name", "t", "year", "sav_lag", "time_lag", "sav_over_time_lag", "premium_ann_pp",
               "bias_per_unit_old_share_pp"]]


# ------------------------------------------------------------------------------------------- (d)
def outside_pool(b, mk):
    x = b[["m", "qi", "dep_a1", "dep_a2", "dep_a4", "dep_a5"]].copy()
    x["all"] = x[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].fillna(0.0).sum(axis=1)
    s = x.groupby(["m", "qi"]).agg(sum_dep=("all", "sum"), sum_sav=("dep_a2", "sum"),
                                   n_B=("all", "size")).reset_index()
    s["qi"] += 1
    s["t"] = tid(s["qi"]).to_numpy()
    O = mk.rename(columns={"time_id": "t"}).merge(s.drop(columns="qi"), on=["m", "t"], how="left")
    O["year"] = O["t"].str[:4].astype(int)
    O["O"] = O["M_hat"] - O["sum_dep"]
    O["o"] = O["O"] / O["sum_sav"].where(O["sum_sav"] > 0)
    O["O_over_M"] = O["O"] / O["M_hat"]
    return O[["m", "t", "year", "M_hat", "sum_dep", "sum_sav", "O", "o", "O_over_M", "floor_binds",
              "phi_mt", "n_B"]]


# ------------------------------------------------------------------------------------------- (e)
def omega(df):
    s = pd.read_csv(MATURITY)
    s["_pref"] = (s["source"] != "cvm").astype(int)
    s = s.sort_values(["firm_key", "fiscal_year", "_pref"]).drop_duplicates(["firm_key", "fiscal_year"])
    s["j"] = s["firm_key"].map(FIRM_TO_CONGL)
    if s["j"].isna().any():
        raise SystemExit(f"unmapped firm keys {sorted(s.loc[s['j'].isna(), 'firm_key'].unique())}")
    s = s.rename(columns={"fiscal_year": "year", "share_le90": "omega"})
    s["judicial_share"] = s["excluded_judicial"] / (s["total"] + s["excluded_judicial"])
    obs = s[["j", "firm_key", "year", "omega", "share_no_maturity", "judicial_share", "source"]]
    # Type-4 stock at the statement date (Q4) per conglomerate, B and D, and the firm type.
    q4 = df[(df["quarter"] == 4) & (df["year"].between(2016, 2024))]
    st = q4.groupby(["j", "year"]).agg(dep4=("dep_a4", "sum"), is_B=("is_B", "max"),
                                       name=("NomeInstituicao", "last")).reset_index()
    st = st[st["dep4"] > 0]
    ftype = df.groupby("j")["is_B"].max()
    obs = obs.assign(is_B=obs["j"].map(ftype))
    years = list(range(2016, 2025))
    med = {}
    for tb in (0, 1):
        o = obs[obs["is_B"] == tb].groupby("year")["omega"].median()
        o_all = obs.groupby("year")["omega"].median()
        for y in years:
            if y in o.index:
                med[(tb, y)] = (float(o[y]), f"median_{'B' if tb else 'D'}_{y}")
            elif len(o):
                near = min(o.index, key=lambda z: (abs(z - y), z))
                med[(tb, y)] = (float(o[near]), f"median_{'B' if tb else 'D'}_{near}")
            else:
                med[(tb, y)] = (float(o_all[y]), f"median_all_{y}")
    own = {j: g.set_index("year") for j, g in obs.groupby("j")}
    rows = []
    for _, r in st.iterrows():
        j, y, tb = r["j"], int(r["year"]), int(r["is_B"])
        if j in own and y in own[j].index:
            o = own[j].loc[y]
            rows.append((j, y, o["omega"], o["share_no_maturity"], o["judicial_share"], f"own_{o['source']}", 0))
        elif j in own:
            near = min(own[j].index, key=lambda z: (abs(z - y), z))
            o = own[j].loc[near]
            rows.append((j, y, o["omega"], o["share_no_maturity"], o["judicial_share"], f"own_nearest_{near}", 1))
        else:
            v, src = med[(tb, y)]
            rows.append((j, y, v, np.nan, np.nan, src, 1))
    W = pd.DataFrame(rows, columns=["conglomerate", "year", "omega", "share_no_maturity",
                                    "judicial_share", "source", "imputed"])
    W = W.merge(st.rename(columns={"j": "conglomerate"}), on=["conglomerate", "year"], how="left")
    cov = []
    for (y, tb), g in W.groupby(["year", "is_B"]):
        cov.append({"year": y, "firm_type": "B" if tb else "D", "n_congl": len(g),
                    "n_own_year": int((g["imputed"] == 0).sum()),
                    "stock_bn": g["dep4"].sum() / 1e9,
                    "share_stock_own_year": g.loc[g["imputed"] == 0, "dep4"].sum() / g["dep4"].sum(),
                    "share_stock_own_any_year": g.loc[g["source"].str.startswith("own"), "dep4"].sum() / g["dep4"].sum(),
                    "share_stock_caixa": g.loc[g["conglomerate"] == CAIXA, "dep4"].sum() / g["dep4"].sum(),
                    "stock_weighted_omega": np.average(g["omega"], weights=g["dep4"])})
    return obs, W, pd.DataFrame(cov), {f"{'B' if k[0] else 'D'}_{k[1]}": v for k, v in med.items()}


# ------------------------------------------------------------------------------------------ main
def main():
    stamp("start")
    df = load_panel()
    flags = rate_flags(df[df["is_B"] == 1])
    b = df[(df["is_B"] == 1) & (df["qi"] >= qi(*T0) - 1) & (df["qi"] <= qi(*T1))].copy()
    mk = load_demand()
    stamp(f"B rows 2015Q4-2024Q4: {len(b):,}; market-quarters with phi and M_hat: {len(mk):,}")
    N = {"panel_md5": PANEL_MD5, "columns_read": COLS, "demand_parquet": DEMAND.name,
         "rate_fallback_cq": {k: sorted(f"{a}:{tid([q])[0]}" for a, q in v) for k, v in flags.items()}}

    # (a)
    L = netflow(b, flags, mk)
    NS = netflow_summary(L)
    L[["j", "m", "t", "year", "quarter", "k", "Dep", "Dep_lag", "rate_lag", "NF", "phi_mt", "Dep_Act",
       "rate_fallback", "judicial"]].to_parquet(OUT / "k0_netflow_panel.parquet", index=False)
    NS.to_csv(OUT / "k0_netflow_summary.csv", index=False)
    tot = L.groupby("k").agg(n=("NF", "size"), n_nf=("NF", lambda v: int(v.notna().sum())),
                             n_neg_dep=("Dep", lambda v: int((v < 0).sum())),
                             n_neg_lag=("Dep_lag", lambda v: int((v < 0).sum())),
                             n_zero_lag=("Dep_lag", lambda v: int((v == 0).sum())),
                             n_no_lag=("Dep_lag", lambda v: int(v.isna().sum())),
                             n_no_rate=("rate_lag", lambda v: int(v.isna().sum())),
                             n_no_phi=("phi_mt", lambda v: int(v.isna().sum())),
                             n_fallback=("rate_fallback", "sum"))
    N["a_counts"] = {int(k): {c: int(v) for c, v in r.items()} for k, r in tot.iterrows()}
    f4 = L[(L["k"] == 4) & L["NF"].notna()]
    # How much of the cell-level churn survives aggregation: |sum over markets of NF| over the sum of
    # |NF| within a bank-quarter (booking moves across municipalities net out at the bank), and
    # |sum over banks| within a market-quarter (moves between banks of one market net out there).
    net = {}
    for k in TYPES:
        g = L[(L["k"] == k) & L["NF"].notna()]
        gross = g["NF"].abs().groupby(g["year"]).sum()
        bank = g.groupby(["j", "qi", "year"])["NF"].sum().abs().groupby(level="year").sum()
        mkt = g.groupby(["m", "qi", "year"])["NF"].sum().abs().groupby(level="year").sum()
        nat = g.groupby(["qi", "year"])["NF"].sum().abs().groupby(level="year").sum()
        net[k] = {int(y): {"bank_net_over_gross": float(bank[y] / gross[y]),
                           "market_net_over_gross": float(mkt[y] / gross[y]),
                           "national_net_over_gross": float(nat[y] / gross[y])} for y in gross.index}
        net[k]["all"] = {"bank_net_over_gross": float(bank.sum() / gross.sum()),
                         "market_net_over_gross": float(mkt.sum() / gross.sum()),
                         "national_net_over_gross": float(nat.sum() / gross.sum())}
    N["a_netting"] = net
    ph = L.drop_duplicates(["m", "t"])["phi_mt"]
    N["a_phi_quantiles"] = {f"p{int(q * 100):02d}": float(ph.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    N["a_fallback_k4_stock_share"] = float(f4.loc[f4["rate_fallback"] == 1, "Dep_lag"].sum() / f4["Dep_lag"].sum())
    N["a_judicial_k4_stock_share"] = float(f4.loc[f4["judicial"] == 1, "Dep_lag"].sum() / f4["Dep_lag"].sum())
    stamp("(a) net flows done")

    # (b)
    P = pool_weights(b, L)
    P[["j", "m", "t", "year", "own2", "Q1", "Q4", "R1", "R2", "R4", "nB", "own_sav_lag_ok", "one_B_bank",
       "has_NF4"]].rename(columns={"own2": "Dep_j2_lag", "nB": "n_B_banks_lag"}) \
        .to_parquet(OUT / "k0_pool_weights.parquet", index=False)
    WS = weights_summary(P)
    WS.to_csv(OUT / "k0_pool_weights_summary.csv", index=False)
    K = P[(P["has_NF4"] == 1)]
    dep4lag = L[(L["k"] == 4) & L["NF"].notna()].set_index(["j", "m", "qi"])["Dep_lag"]
    K = K.assign(dep4lag=dep4lag.reindex(pd.MultiIndex.from_frame(K[["j", "m", "qi"]])).to_numpy())
    st4 = K["dep4lag"].clip(lower=0).sum()
    N["b_K1_rows"] = int(len(K))
    N["b_no_own_savings"] = {"n": int((K["own_sav_lag_ok"] == 0).sum()),
                             "share_rows": float((K["own_sav_lag_ok"] == 0).mean()),
                             "share_B_type4_stock": float(K.loc[K["own_sav_lag_ok"] == 0, "dep4lag"].clip(lower=0).sum() / st4)}
    N["b_one_B_bank"] = {"n_rows": int(K["one_B_bank"].sum()),
                         "n_market_quarters": int(K.loc[K["one_B_bank"] == 1, ["m", "qi"]].drop_duplicates().shape[0]),
                         "n_market_quarters_total": int(K[["m", "qi"]].drop_duplicates().shape[0]),
                         "share_B_type4_stock": float(K.loc[K["one_B_bank"] == 1, "dep4lag"].clip(lower=0).sum() / st4)}
    N["b_no_own_savings_by_year"] = {int(y): float((g["own_sav_lag_ok"] == 0).mean())
                                     for y, g in K.groupby("year")}
    V = K[(K["own_sav_lag_ok"] == 1) & (K["R2"] > 0)]
    vd = variance_decomposition(np.log(V["R2"]), V["j"] + "|" + V["m"], V["m"] + "|" + V["t"])
    vd_lvl = variance_decomposition(V["R2"].clip(upper=V["R2"].quantile(0.99)), V["j"] + "|" + V["m"],
                                    V["m"] + "|" + V["t"])
    pd.DataFrame([{"variable": "log R2", **vd}, {"variable": "R2 (level, top-coded at p99)", **vd_lvl}]) \
        .to_csv(OUT / "k0_logR2_variance_decomposition.csv", index=False)
    N["b_variance_logR2"], N["b_variance_R2_level_p99"] = vd, vd_lvl
    N["b_median_R2_by_year"] = {int(y): float(g["R2"].median()) for y, g in V.groupby("year")}
    names = df.groupby("j")["NomeInstituicao"].last()
    z = K[K["own_sav_lag_ok"] == 0].groupby("j")["dep4lag"].sum().sort_values(ascending=False)
    N["b_no_own_savings_top"] = [{"j": j, "name": names.get(j), "share_of_these_cells_type4": float(v / z.sum())}
                                 for j, v in z.head(6).items()]
    fb = f4[f4["rate_fallback"] == 1].groupby("j").agg(stock=("Dep_lag", "sum"), quarters=("t", "nunique"))
    fb = fb.sort_values("stock", ascending=False)
    N["a_fallback_k4_top"] = [{"j": j, "name": names.get(j), "quarters": int(r["quarters"]),
                               "share_of_flagged_stock": float(r["stock"] / fb["stock"].sum())}
                              for j, r in fb.head(8).iterrows()]
    stamp("(b) pool weights done")

    # (c)
    cq, D, vint = savings_spread(b, flags, P)
    D.to_csv(OUT / "k0_savings_spread_dispersion.csv", index=False)
    cq[["j", "name", "t", "year", "sav", "rho2_pp", "selic_pp", "rate_fallback"]] \
        .to_csv(OUT / "k0_savings_spread_congl_quarter.csv", index=False)
    Dy = D.groupby(["sample", "year"]).agg(n_congl=("n_congl", "mean"), selic_pp=("selic_pp", "mean"),
                                           sd=("sd", "mean"), iqr=("iqr", "mean"),
                                           wsd=("savings_weighted_sd", "mean")).reset_index()
    N["c_by_year"] = Dy.to_dict(orient="records")
    N["c_rho2_distinct_values_per_quarter_max"] = int(cq[cq["sav"] > 0].groupby("qi")["rho2_pp"]
                                                      .apply(lambda v: v.round(10).nunique()).max())
    N["c_vintage"] = vint
    VB = vintage_bias(b)
    VB.to_csv(OUT / "k0_savings_vintage_bias.csv", index=False)
    vb = VB[(VB["sav_lag"] > 0) & (VB["time_lag"] > 0)]
    N["c_premium_by_year"] = {int(y): float(g["premium_ann_pp"].mean()) for y, g in vb.groupby("year")}
    N["c_premium_quarters"] = sorted(vb.loc[vb["premium_ann_pp"] > 0, "t"].unique().tolist())
    w = vb[vb["year"].between(2018, 2020)]
    top = w.groupby("j").agg(name=("name", "last"), n_q=("t", "count"),
                             sav_over_time=("sav_over_time_lag", "mean"),
                             bias_per_unit_share_pp=("bias_per_unit_old_share_pp", "mean"),
                             sav_bn=("sav_lag", lambda v: v.mean() / 1e9),
                             time_bn=("time_lag", lambda v: v.mean() / 1e9)).reset_index()
    top = top[(top["n_q"] >= 8) & (top["time_bn"] >= 1)]
    N["c_top5_vintage_bias_2018_2020"] = top.nlargest(5, "bias_per_unit_share_pp").to_dict(orient="records")
    big = top[top["time_bn"] >= 10].sort_values("time_bn", ascending=False)
    N["c_vintage_bias_large_banks_2018_2020"] = big.to_dict(orient="records")
    N["c_vintage_bias_2018_2020_time_weighted"] = float(np.average(top["bias_per_unit_share_pp"],
                                                                   weights=top["time_bn"]))
    stamp("(c) savings spread done")

    # (d)
    O = outside_pool(b, mk)
    O.to_csv(OUT / "k0_outside_pool.csv", index=False)
    rows = []
    for y, g in list(O.groupby("year")) + [("all", O)]:
        ok = g["sum_dep"].notna()
        rows.append({"year": y, "n_mq": int(len(g)), "n_with_lag": int(ok.sum()),
                     "share_O_negative": float((g.loc[ok, "O"] < 0).mean()),
                     "share_O_below_5pct_of_M": float((g.loc[ok, "O_over_M"] < 0.05).mean()),
                     "share_paper_floor_binds": float(g["floor_binds"].mean()),
                     **{f"o_p{int(q * 100):02d}": float(g["o"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                     "O_over_M_median": float(g["O_over_M"].median()),
                     "pop_weighted_inside_ratio": float(g.loc[ok, "sum_dep"].sum() / g.loc[ok, "M_hat"].sum())})
    N["d_by_year"] = rows
    stamp("(d) outside pool done")

    # (e)
    obs, W, cov, med = omega(df)
    W[["conglomerate", "name", "is_B", "year", "omega", "source", "imputed", "share_no_maturity",
       "judicial_share", "dep4"]].rename(columns={"dep4": "dep_a4_q4"}) \
        .to_csv(OUT / "k0_omega.csv", index=False)
    cov.to_csv(OUT / "k0_omega_coverage.csv", index=False)
    N["e_statement_obs"] = obs.assign(is_B=obs["is_B"].astype(int)).to_dict(orient="records")
    N["e_fill_medians"] = med
    N["e_coverage"] = cov.to_dict(orient="records")
    stamp("(e) omega done")

    with open(OUT / "k0_numbers.json", "w", encoding="utf-8") as fh:
        json.dump(N, fh, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(NS[NS["year"].astype(str) == "all"].round(4).to_string())
    print(NS[NS["k"] == 4].round(4).to_string())
    print(WS.round(3).to_string())
    print(json.dumps({k: N[k] for k in ("b_no_own_savings", "b_one_B_bank", "b_variance_logR2",
                                        "b_variance_R2_level_p99", "b_no_own_savings_top",
                                        "a_fallback_k4_top")}, indent=1, default=str))
    print(Dy.round(3).to_string())
    print(N["c_rho2_distinct_values_per_quarter_max"], N["c_premium_by_year"], N["c_premium_quarters"])
    print(pd.DataFrame(N["c_top5_vintage_bias_2018_2020"]).round(3).to_string())
    print(pd.DataFrame(N["c_vintage_bias_large_banks_2018_2020"]).round(3).to_string())
    print(N["c_vintage_bias_2018_2020_time_weighted"])
    print(json.dumps({k: v for k, v in vint.items() if k != "by_year"}, indent=1, default=str))
    print(pd.DataFrame(rows).round(3).to_string())
    print(cov.round(3).to_string())
    print(json.dumps(N["a_counts"], indent=1), N["a_fallback_k4_stock_share"], N["a_judicial_k4_stock_share"])
    print({k: len(v) for k, v in N["rate_fallback_cq"].items()})
    print(json.dumps({k: v["all"] for k, v in N["a_netting"].items()}, indent=1), N["a_phi_quantiles"])
    stamp("done")


if __name__ == "__main__":
    main()
