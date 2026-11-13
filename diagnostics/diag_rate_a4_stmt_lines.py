"""
diag_rate_a4_stmt_lines.py
==========================
Are Santander's and BB's 2020 type-4 rates (1.515 and 1.245 x CDI) real, or made by how
panel_cosif_calibrate.py builds them from the statements?  (claude/MAIN_IMPR_CHANNEL.md, A1.51,
log 11-13 15:06 MAIN.)  Companion of diag_rate_a4_savings_premium.py.  Read-only: outputs go to --out.

1. build_corrected() in memory, capturing the conglomerate frame at the last step, to decompose the
   statement rate of each bank-year: the printed line, the netted judicial cost, lambda_i, the
   expense used, sum_q CDI_q x lagged ordinary stock, the rate over the CDI; next to the Cosif
   route the statement replaces (lump, strip x k over the same stock).
2. BB with each year's own foreign/special share: lambda_i is measured on 2025 (Cosif 8113 over the
   2025 'Demais' line).  The BB individual deposit note splits 'Depositos a prazo' into Judiciais,
   Moeda nacional, Moedas estrangeiras and special items; lambda_y = lambda_2025 x d_y / d_2025, with
   d the domestic-currency share of the non-judicial balance (BB_DOMESTIC below, from the
   statements).  The expense is linear in lambda, so the rate scales by lambda_y / lambda_2025.
3. S1's rate and its correlation with the Doc 5005 CDI-indexed issuance ratio again, with BB's
   rates so scaled (Rout/diag_lai025440 machinery, as in the companion).

Usage:  python diagnostics/diag_rate_a4_stmt_lines.py --out <dir>
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

LAI = paths.rout_dir() / "diag_lai025440"
BB, SAN = "C0080329", "C0080185"
SEGS = ["S1", "S2", "S3", "S4", "S5"]

# BB Banco Multiplo (individual), note 'Recursos de clientes / Depositos', R$ thousand, year end:
# (moeda nacional, moedas estrangeiras, non-judicial total = Depositos a prazo - Judiciais).
# DFP 2020 p338 (2019, 2020); DFP 2021 p343 (2021); DFP 2025 p133 (2025).
BB_DEMAIS = {
    2019: (52_257_226, 9_741_385, 223_969_826 - 153_943_823),
    2020: (82_329_375, 14_435_524, 269_408_816 - 164_388_100),
    2021: (101_213_791, 13_202_917, 307_572_235 - 183_172_537),
    2025: (269_494_033, 14_155_857, 560_668_565 - 263_446_160),
}

_cap = {}
_orig_shape = pcc.benchmarked_semester_shape
_orig_lambda = pcc.td_lambda


def _shape(congl, pre):
    _cap["congl"] = congl.copy()
    _cap["pre"] = pre.copy()
    return _orig_shape(congl, pre)


def _lam(congl, td, scope):
    lam = _orig_lambda(congl, td, scope)
    _cap["lambda"] = lam.copy()
    _cap["td"] = td.copy()
    return lam


pcc.benchmarked_semester_shape = _shape
pcc.td_lambda = _lam


def ann(r):
    return 100.0 * ((1.0 + r) ** 4 - 1.0)


def bank_decomp(c, code):
    g = c[(c["CodConglomeradoPrudencial"].astype(str) == code) & c["year"].between(2017, 2024)].copy()
    g["w"] = g["cdi_qoq"] * g["stk_time_ord_lag"]
    g["strip_k"] = g["strip"] * g["k_shrunk"]
    y = g.groupby("year").agg(lump=("exp_lump", lambda s: s.abs().sum()), strip_k=("strip_k", "sum"),
                              stmt=("stmt_cdb", lambda s: s.sum(min_count=1)), w=("w", "sum"),
                              stk_ord_lag=("stk_time_ord_lag", "mean"), J_lag=("J_lag", "mean"),
                              savings_lag=("stk_savings_lag", "mean"), cdi=("cdi_qoq", lambda s: (1 + s).prod() - 1),
                              judicial=("exp_judicial_used", "sum"))
    y["stmt_over_cdi"] = y["stmt"] / y["w"]
    y["strip_k_over_cdi"] = y["strip_k"] / y["w"]
    for col in ("lump", "strip_k", "stmt", "judicial"):
        y[col] = y[col] / 1e9
    y["stk_ord_lag"] = y["stk_ord_lag"] / 1e9
    y["J_lag"] = y["J_lag"] / 1e9
    y["savings_lag"] = y["savings_lag"] / 1e9
    return y.reset_index()


def seg_means(x, col):
    rows = []
    for (seg, yy, q), g in x[x["segment"].isin(SEGS)].groupby(["segment", "year", "quarter"]):
        pos = g[(g["dep_a4_bn"] > 0) & g[col].notna()]
        rows.append(dict(segment=seg, year=yy, quarter=q,
                         r=np.average(pos[col], weights=pos["dep_a4_bn"]) if len(pos) else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("panel_cosif_calibrate").setLevel(logging.WARNING)

    base, _ = pcc.build_corrected()
    c = _cap["congl"]
    lam = _cap["lambda"]
    print("lambda_i (2025):", {k: round(v, 4) for k, v in lam.items()})
    td = _cap["td"]
    print("statement time-deposit lines, R$ bn:")
    print(td[td["CodConglomeradoPrudencial"].isin([BB, SAN])]
          .pivot_table(index=["CodConglomeradoPrudencial", "period"], columns="year", values="value", aggfunc="sum")
          .div(1e9).round(3).to_string())

    dec = []
    for code, name in ((BB, "BB"), (SAN, "Santander")):
        d = bank_decomp(c, code).assign(bank=name)
        dec.append(d)
        print(f"\n{name}: statement route vs the Cosif route it replaces (R$ bn; ratios = expense / sum CDI x lagged ordinary stock)")
        print(d[["year", "lump", "strip_k", "stmt", "judicial", "stk_ord_lag", "J_lag", "cdi",
                 "stmt_over_cdi", "strip_k_over_cdi"]].round(3).to_string(index=False))
    pd.concat(dec).to_csv(out / "stmt_decomposition.csv", index=False, lineterminator="\n", float_format="%.6g")

    # BB: each year's own domestic-currency share
    dshare = {y: v[0] / v[2] for y, v in BB_DEMAIS.items()}
    fshare = {y: v[1] / v[2] for y, v in BB_DEMAIS.items()}
    avg = {2019: dshare[2019], 2020: (dshare[2019] + dshare[2020]) / 2, 2021: (dshare[2020] + dshare[2021]) / 2}
    fac = {y: avg[y] / dshare[2025] for y in avg}
    print("\nBB non-judicial time deposits, year end: domestic share", {k: round(v, 4) for k, v in dshare.items()},
          "foreign share", {k: round(v, 4) for k, v in fshare.items()})
    print("BB lambda_y / lambda_2025 (mean of opening and closing domestic share; 2019 its closing):",
          {k: round(v, 4) for k, v in fac.items()})

    p = pd.read_csv(LAI / "panel_conglomerate_quarterly.csv", dtype={"code": str})
    cmp_ = pd.read_csv(LAI / "compare.csv")
    p["f"] = 1.0
    m = (p["code"] == BB) & p["year"].isin(fac.keys()) & (p["rate_a4_src"] == "statement")
    p.loc[m, "f"] = p.loc[m, "year"].map(fac)
    p["r_fix"] = p["rate_a4_national_qoq"] * p["f"]
    print(f"BB rows rescaled: {int(m.sum())}; BB rows 2019-21 not on the statement source: "
          f"{int(((p['code'] == BB) & p['year'].isin(fac.keys()) & ~m).sum())}")
    b0 = seg_means(p, "rate_a4_national_qoq").rename(columns={"r": "r0"})
    b1 = seg_means(p, "r_fix").rename(columns={"r": "r1"})
    s = b0.merge(b1, on=["segment", "year", "quarter"]).merge(
        cmp_[["segment", "year", "quarter", "cdi_ann", "panel_ann", "bcb_cdi_ratio_cdi"]], on=["segment", "year", "quarter"])
    s["ratio0"] = ann(s["r0"]) / s["cdi_ann"]
    s["ratio1"] = ann(s["r1"]) / s["cdi_ann"]
    s.to_csv(out / "segment_quarter_bb_own_share.csv", index=False, lineterminator="\n", float_format="%.6g")
    g = s[s["segment"] == "S1"].sort_values(["year", "quarter"])
    print("\nS1, BB at its own-year share: check vs compare.csv", float((g["ratio0"] - g["panel_ann"] / g["cdi_ann"]).abs().max()))
    print("corr(ratio, BCB ratio):", round(np.corrcoef(g["ratio0"], g["bcb_cdi_ratio_cdi"])[0, 1], 3), "->",
          round(np.corrcoef(g["ratio1"], g["bcb_cdi_ratio_cdi"])[0, 1], 3))
    print(g.groupby("year")[["ratio0", "ratio1", "bcb_cdi_ratio_cdi"]].mean().round(3).to_string())
    bb = p[p["code"] == BB].groupby("year").apply(
        lambda d: pd.Series({"ratio0": (d["panel_ann"] / d["cdi_ann"]).mean(),
                             "ratio1": (ann(d["r_fix"]) / d["cdi_ann"]).mean()}))
    print("BB by year:\n", bb.round(3).to_string())
    # S1 without Santander, and without both, for reference
    for drop in ([SAN], [SAN, BB]):
        q = p[~p["code"].isin(drop)]
        h = seg_means(q, "r_fix").merge(cmp_[["segment", "year", "quarter", "cdi_ann", "bcb_cdi_ratio_cdi"]],
                                         on=["segment", "year", "quarter"])
        h = h[h["segment"] == "S1"]
        h["ratio"] = ann(h["r"]) / h["cdi_ann"]
        print(f"S1 without {drop}: corr {np.corrcoef(h['ratio'], h['bcb_cdi_ratio_cdi'])[0, 1]:.3f};",
              h.groupby("year")["ratio"].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
