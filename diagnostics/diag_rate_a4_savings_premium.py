"""
diag_rate_a4_savings_premium.py
===============================
Does the panel's type-4 rate (rate_a4) carry part of the old-rule savings premium in 2017-2021?
(claude/MAIN_IMPR_CHANNEL.md, A1.50, log 11-13 14:06 MAIN item 2.)  Read-only: it writes no
pipeline file and no paper key; the outputs go to --out.

Mechanism.  Before 2025 the type-4 rate of a strip bank is k_i x strip / lagged ordinary time stock,
strip = lump 8.1.1 - repo - bills - subordinated - SAVINGS - interbank - judicial
(panel_cosif_calibrate.strip_components), with savings = savings_cost_qoq(congl, OLD_RULE_SHARE) x
lagged savings stock and OLD_RULE_SHARE = 0 (utils/macro_rates.py): the new-rule yield on the whole
stock.  Where a bank's statement prints a savings line the savings component is rescaled to it
(premium included); where it prints a time-deposit line the strip is not used at all.

Steps
  1. build_corrected() in memory with the share 0 (the baseline; checked against the stored
     cosif_cdb_rate_corrected.csv).
  2. The old-rule share implied by each statement savings line, bank-year:
        s = (E - sum_q new_q S_lag,q) / sum_q prem_q S_lag,q,   prem = savings_old_qoq - savings_rate_qoq
     with E the statement's savings expense and S_lag Cosif's lagged savings stock (4.1.2).  In years
     without premium E / sum new S_lag checks the scope (should be near 1).
  3. Scenarios for the share of every conglomerate-year (statement banks: own clipped s):
        private  premium-weighted mean s of the private statement banks (Bradesco, Santander)
        all      the same over every statement bank
        one      s = 1 (upper bound: the whole stock on the old rule)
     2017 takes the 2018 value (one premium quarter in 2017).  build_corrected() is rerun with
     savings_cost_qoq patched to use the share; k_i is measured on 2025+ quarters, where the premium
     is zero, so it does not move.
  4. The change is carried to the panel's conglomerate-quarter rates (Rout/diag_lai025440/
     panel_conglomerate_quarterly.csv) as the factor r_scen / r_base for measured rows; carried rows
     take the mean factor of the bank's measured quarters within +-4.  Segment means (dep_a4
     weights) for S1-S5, ratios to the CDI and their correlation with the BCB CDI-indexed issuance
     ratio (compare.csv), as in diag_lai025440_rates.py.

Usage:  python diagnostics/diag_rate_a4_savings_premium.py --out <dir>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging

import numpy as np
import pandas as pd

import panel_cosif_calibrate as pcc
from utils import paths
from utils.macro_rates import savings_cost_qoq as _base_cost

LAI = paths.rout_dir() / "diag_lai025440"
SEGS = ["S1", "S2", "S3", "S4", "S5"]
Q0, Q1 = 2017 * 4 + 2, 2023 * 4 + 3          # 2017Q3 .. 2023Q4 as year*4 + quarter-1
BANKS = {"C0080738": "Caixa", "C0080329": "BB", "C0080185": "Santander", "C0080099": "Itau",
         "C0080075": "Bradesco"}
PRIVATE = ["C0080075", "C0080185"]

_captured = {}
_orig_build = pcc.build_congl_panel
SHARE = None   # Series indexed by (code, year), or None for the baseline


def _capture(found, macro):
    c = _orig_build(found, macro)
    _captured["congl"] = c.copy()
    return c


def _patched_cost(d, share):
    if SHARE is None:
        return _base_cost(d, share)
    key = pd.MultiIndex.from_arrays([d["CodConglomeradoPrudencial"].astype(str), d["year"]])
    s = pd.Series(SHARE.reindex(key).to_numpy(), index=d.index).fillna(0.0)
    return _base_cost(d, s)


pcc.build_congl_panel = _capture
pcc.savings_cost_qoq = _patched_cost


def run(share):
    global SHARE
    SHARE = share
    out, _ = pcc.build_corrected()
    out = out.copy()
    out["CodConglomeradoPrudencial"] = out["CodConglomeradoPrudencial"].astype(str)
    return out


def implied_shares(congl):
    c = congl[congl["year"].between(2016, 2024)].copy()
    c["code"] = c["CodConglomeradoPrudencial"].astype(str)
    c["prem"] = (c["savings_old_qoq"] - c["savings_rate_qoq"]).fillna(0.0)
    c["A"] = c["savings_rate_qoq"] * c["stk_savings_lag"]
    c["P"] = c["prem"] * c["stk_savings_lag"]
    g = c.groupby(["code", "year"]).agg(A=("A", lambda s: s.sum(min_count=4)), P=("P", "sum"),
                                         n=("A", "count"), S_lag_mean=("stk_savings_lag", "mean"))
    st = pcc.load_dfp_benchmarks({"savings": ["savings"]})
    st["code"] = st["CodConglomeradoPrudencial"].astype(str)
    E = st.groupby(["code", "year"])["value"].sum().rename("E")
    # the lines DFP_EXCLUDE leaves out (BB 2018-19), read raw for the record
    raw = pd.concat([pd.read_csv(Path(pcc.DFP_FUNDING_DIR) / n, dtype={"cnpj": str}) for n in pcc.DFP_FUNDING_FILES])
    raw = raw[raw["line"] == "savings"].copy()
    raw["code"] = raw["cnpj"].astype(int).map(pcc._cnpj_to_code())
    if "period" in raw.columns:
        raw = raw[raw["period"].fillna("FY") == "FY"]
    Eraw = raw.dropna(subset=["code"]).groupby(["code", "year"])["value_brl"].sum().rename("E_raw")
    t = g.join(E, how="inner").join(Eraw, how="left").reset_index()
    t = t[t["n"] == 4]
    t["E_over_newrule"] = t["E"] / t["A"]
    t["premium_over_newrule"] = t["P"] / t["A"]
    t["s_implied"] = np.where(t["P"] > 0.05 * t["A"], (t["E"] - t["A"]) / t["P"], np.nan)
    t["s_implied_raw"] = np.where(t["P"] > 0.05 * t["A"], (t["E_raw"] - t["A"]) / t["P"], np.nan)
    t["bank"] = t["code"].map(BANKS).fillna(t["code"])
    return t


def share_scenarios(congl, t):
    codes = congl["CodConglomeradoPrudencial"].astype(str).unique()
    years = range(2016, 2025)
    ok = t.dropna(subset=["s_implied"])
    own = ok.set_index(["code", "year"])["s_implied"].clip(0, 1)

    def wmean_year(sub):
        m = sub.groupby("year").apply(lambda d: np.average(d["s_implied"].clip(0, 1), weights=d["P"]))
        return m

    pools = {"private": wmean_year(ok[ok["code"].isin(PRIVATE)]), "all": wmean_year(ok)}
    scen = {}
    for name, m in pools.items():
        m = m.reindex(years)
        m.loc[2017] = m.loc[2018] if pd.notna(m.loc[2018]) else m.loc[2017]
        m = m.fillna(0.0)
        idx = pd.MultiIndex.from_product([codes, years], names=["code", "year"])
        s = pd.Series(np.repeat(m.reindex(years).to_numpy()[None, :], len(codes), axis=0).ravel(), index=idx)
        o = own.reindex(idx).dropna()
        s.loc[o.index] = o
        # 2017 of a statement bank: its 2018 value
        for c in own.index.get_level_values(0).unique():
            if (c, 2018) in own.index:
                s.loc[(c, 2017)] = own.loc[(c, 2018)]
        scen[name] = s
    idx = pd.MultiIndex.from_product([codes, years], names=["code", "year"])
    scen["one"] = pd.Series(1.0, index=idx)
    return scen, pools


def qi(y, q):
    return y * 4 + (q - 1)


def ann(r):
    return 100.0 * ((1.0 + r) ** 4 - 1.0)


def map_to_panel(p, base, scen_out):
    key = ["CodConglomeradoPrudencial", "AnoMes"]
    b = base[key + ["rate_a4_corrected_qoq", "rate_flag"]].rename(columns={"rate_a4_corrected_qoq": "r0", "rate_flag": "f0"})
    s = scen_out[key + ["rate_a4_corrected_qoq", "rate_flag"]].rename(columns={"rate_a4_corrected_qoq": "r1", "rate_flag": "f1"})
    m = b.merge(s, on=key, how="outer")
    m["fac"] = np.where((m["f0"] == 0) & (m["f1"] == 0) & (m["r0"] > 0), m["r1"] / m["r0"], np.nan)
    m["lost"] = (m["f0"] == 0) & (m["f1"] != 0)
    m["code"] = m["CodConglomeradoPrudencial"]
    m["year"] = m["AnoMes"] // 100
    m["quarter"] = (m["AnoMes"] % 100) // 3
    m["qi"] = qi(m["year"], m["quarter"])
    x = p.merge(m[["code", "year", "quarter", "r0", "fac", "lost"]], on=["code", "year", "quarter"], how="left")
    meas = x["rate_a4_carried"].fillna(0) == 0
    # carried rows: mean factor of the bank's measured quarters within +-4
    fm = m.dropna(subset=["fac"])[["code", "qi", "fac"]]
    car = x[~meas & x["rate_a4_national_qoq"].notna()]
    cf = {}
    for i, r in car.iterrows():
        f = fm[(fm["code"] == r["code"]) & (fm["qi"] - r["qi"]).abs().le(4)]["fac"]
        cf[i] = f.mean() if len(f) else 1.0
    fac = x["fac"].where(meas, pd.Series(cf)).fillna(1.0)
    x["r_scen"] = x["rate_a4_national_qoq"] * fac
    x.loc[meas & x["lost"].fillna(False).astype(bool), "r_scen"] = np.nan
    x["fac_used"] = fac
    return x


def seg_means(x, col):
    rows = []
    for (seg, y, q), g in x[x["segment"].isin(SEGS)].groupby(["segment", "year", "quarter"]):
        pos = g[(g["dep_a4_bn"] > 0) & g[col].notna()]
        r = np.average(pos[col], weights=pos["dep_a4_bn"]) if len(pos) else np.nan
        rows.append(dict(segment=seg, year=y, quarter=q, r=r))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger("panel_cosif_calibrate").setLevel(logging.WARNING)

    base = run(None)
    congl = _captured["congl"]
    stored = pd.read_csv(pcc.OUT_CORRECTED, dtype={"CodConglomeradoPrudencial": str})
    chk = stored.merge(base, on=["CodConglomeradoPrudencial", "AnoMes"], suffixes=("_s", "_b"))
    d = (chk["rate_a4_corrected_qoq_s"] - chk["rate_a4_corrected_qoq_b"]).abs()
    print(f"baseline vs stored corrected csv: {len(chk)}/{len(stored)} rows matched, max |diff| {d.max():.3g}, "
          f"flag mismatches {(chk['rate_flag_s'] != chk['rate_flag_b']).sum()}")

    t = implied_shares(congl)
    t.to_csv(out_dir / "implied_old_rule_share.csv", index=False, lineterminator="\n", float_format="%.6g")
    print("\nImplied old-rule share by statement bank-year (E / new-rule cost; s):")
    print(t.pivot(index="bank", columns="year", values="E_over_newrule").round(3).to_string())
    print(t.pivot(index="bank", columns="year", values="s_implied").round(3).to_string())
    print("raw (DFP_EXCLUDE lines kept):")
    print(t.pivot(index="bank", columns="year", values="s_implied_raw").round(3).to_string())
    print("premium / new-rule cost:", t.groupby("year")["premium_over_newrule"].mean().round(3).to_dict())

    scen, pools = share_scenarios(congl, t)
    print("\npooled shares by year:", {k: v.round(3).to_dict() for k, v in pools.items()})

    p = pd.read_csv(LAI / "panel_conglomerate_quarterly.csv", dtype={"code": str})
    p["qi"] = qi(p["year"], p["quarter"])
    cmp_ = pd.read_csv(LAI / "compare.csv")
    # does the panel's measured rate equal the calibrate baseline?
    bb = base.assign(code=base["CodConglomeradoPrudencial"], year=base["AnoMes"] // 100,
                     quarter=(base["AnoMes"] % 100) // 3)
    z = p.merge(bb[["code", "year", "quarter", "rate_a4_corrected_qoq"]], on=["code", "year", "quarter"])
    z = z[(z["rate_a4_carried"] == 0) & z["rate_a4_src"].isin(["statement", "benchmarked", "imputed"])]
    dd = (z["rate_a4_national_qoq"] - z["rate_a4_corrected_qoq"]).abs()
    print(f"\npanel measured rate vs calibrate baseline: {len(z)} rows, share equal to 1e-10 {(dd < 1e-10).mean():.3f}")

    base_seg = seg_means(p, "rate_a4_national_qoq")
    results, bank_rows, seg_rows = [], [], []
    for name, sh in scen.items():
        so = run(sh)
        x = map_to_panel(p, base, so)
        x["scenario"] = name
        results.append(x)
        sm = seg_means(x, "r_scen").rename(columns={"r": "r_scen"}).merge(
            base_seg.rename(columns={"r": "r_base"}), on=["segment", "year", "quarter"])
        sm = sm.merge(cmp_[["segment", "year", "quarter", "cdi_ann", "panel_ann", "bcb_cdi_ratio_cdi"]],
                      on=["segment", "year", "quarter"])
        sm["ratio_base"] = ann(sm["r_base"]) / sm["cdi_ann"]
        sm["ratio_scen"] = ann(sm["r_scen"]) / sm["cdi_ann"]
        sm["scenario"] = name
        seg_rows.append(sm)
        xb = x[x["code"].isin(BANKS)]
        xb = xb.assign(bank=xb["code"].map(BANKS), ratio_base=xb["panel_ann"] / xb["cdi_ann"],
                       ratio_scen=ann(xb["r_scen"]) / xb["cdi_ann"])
        bank_rows.append(xb.groupby(["bank", "year"])[["ratio_base", "ratio_scen", "fac_used"]].mean()
                         .reset_index().assign(scenario=name))
        lost = x["lost"].fillna(False).astype(bool)
        print(f"\n=== scenario {name}: {int(lost.sum())} panel rows lose their rate (flag changes); "
              f"rows with factor != 1: {int((x['fac_used'].sub(1).abs() > 1e-9).sum())}")
        for seg in ["S1", "S3"]:
            g = sm[sm["segment"] == seg].sort_values(["year", "quarter"])
            c0 = np.corrcoef(g["ratio_base"], g["bcb_cdi_ratio_cdi"])[0, 1]
            okk = g["ratio_scen"].notna()
            c1 = np.corrcoef(g.loc[okk, "ratio_scen"], g.loc[okk, "bcb_cdi_ratio_cdi"])[0, 1]
            check = (g["ratio_base"] - g["panel_ann"] / g["cdi_ann"]).abs().max()
            print(f"{seg}: corr(ratio, BCB ratio) base {c0:.3f} -> {c1:.3f}  (base reproduces compare.csv to {check:.2g})")
            print(g.groupby("year")[["ratio_base", "ratio_scen", "bcb_cdi_ratio_cdi"]].mean().round(3).to_string())
        print(bank_rows[-1].pivot(index="year", columns="bank", values="ratio_scen").round(3).to_string())

    pd.concat(seg_rows).to_csv(out_dir / "segment_quarter_scenarios.csv", index=False, lineterminator="\n", float_format="%.6g")
    pd.concat(bank_rows).to_csv(out_dir / "bank_year_scenarios.csv", index=False, lineterminator="\n", float_format="%.6g")
    pd.concat(results).to_csv(out_dir / "conglomerate_quarter_scenarios.csv", index=False, lineterminator="\n", float_format="%.8g")
    print("\nbase bank ratios:")
    print(bank_rows[0].pivot(index="year", columns="bank", values="ratio_base").round(3).to_string())


if __name__ == "__main__":
    main()
