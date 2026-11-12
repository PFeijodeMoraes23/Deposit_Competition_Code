"""
diag_cap_sensitivity.py
=======================
Part B of the net-flow-gap and caps handoff (claude/MAIN_REPLY_2026-11-05.md, the two sections
"Update 11-11 19:44"): how the logit moves when the cap on phi-hat in the active market (PHI_CAP) and
the cap on a market's total inside share (TOTAL_SHARE_CAP) move. Stage 1, the screen: the stored I2
frames of routines (III) and (IV), spec 12, re-shared or filtered in sandboxes outside OneDrive, and the
logit run locally on each. No production file is written.

Variants (PHI_CAP, TOTAL_SHARE_CAP, rows):
  base_star  0.99     0.95  the stored rows re-shared
  f          0.99     0.95  the stored rows re-shared with the Global Findex series by year
                            (bcb_banked_mca_panel.csv findex_banked_frac, merged on mca_code x year) in
                            place of FINDEX_FALLBACK, which the production frames carry on every row
                            because process_specification's keep_cols drops the column; the merged column
                            is dropped again before the frame is written
  p1         0.98     0.95  re-shared
  p2         0.995    0.95  re-shared
  p4         0.999    0.95  re-shared
  p5         0.9999   0.95  re-shared
  p6         0.99999  0.95  re-shared
  s1         0.99     0.90  re-shared
  s2         0.99     0.98  re-shared
  p3a        0.99     0.95  the stored frame (production's shares) without the B rows with phi-hat >= 0.99
  s3a        0.99     0.95  the stored frame without the B rows of the market-quarters where the floor
                            binds: M_mt above the anchor R Pop / access, R the median of M access / Pop
                            (the share sum marks fewer of them, since finalize dropped some rows after
                            production formed the shares)
  p3b        0.99     0.95  the stored rows without the B rows with phi-hat >= 0.99, re-shared (r-hat
                            without those markets)
A re-share calls sleep_demand_prep_link.build_market_size_and_shares on the stored rows with PHI_CAP and
TOTAL_SHARE_CAP set as module attributes (no environment variable; an exported DEMAND_* knob stops the
script). The stored frames are what finalize left, so base_star is not production; gate B-S3 compares
the two logits.

Steps (--step, comma list):
  env      B-S0: julia --project=. -e "using Parquet2, DataFrames, JSON3, Distributions".
  frames   B-S1 and B-S4. Each variant's two frames to <sandbox>/<v>/demand_prep/ under the stored file
           names, with the stored frame's schema and its panel md5 (demand_chars.market_panel_md5) in the
           metadata, written as .part and renamed when complete; each frame's diagnostics to
           <sandbox>/<v>/frame_diag.json and the cells that set r-hat to <sandbox>/<v>/top_cells_E{k}.csv;
           the stored frames' own diagnostics to <sandbox>/production_frame_diag.json. B-S4: the first
           stage of blp_logit.jl project_spreads per endogenous type (4, 5), the spread on [X, Z] with X
           its X_COLS and the D-type dummy and Z its 16 excluded instruments: the conglomerate-clustered
           Wald F of Z (CR1) and the Montiel Olea-Pflueger effective F, on the stored rows, on the rows of
           p3a / p3b and on the rows of s3a (the other variants keep the stored rows, and the spread, X
           and Z do not depend on the caps) -> <sandbox>/first_stage.json. The step ends by writing
           <sandbox>/FRAMES_COMPLETE.json (every frame's md5); without it the frame set is partial.
  logit    B-S2: julia --project=. --threads=N blp_logit.jl --demand-dir <sandbox>/<v>/demand_prep
           --out-dir <sandbox>/<v>/out, one variant at a time, log <sandbox>/<v>/logit.log, with the BLP_*,
           SLEEP_ACTIVE_ESTS and DEMAND_* variables removed from its environment (the defaults: WCB, 999
           draws, no intercept, routines 1-4). A variant whose summary is newer than its frames is skipped.
  summary  B-S3 and B-S5 -> --out-dir (default Rout/diag_cap_sensitivity/): cap_screen_summary.csv (one row
           per variant and routine, the full_dtype column: alpha, se (WCB), p, the mean own-price
           elasticity alpha mean rho (1 - s) as blp_logit.jl forms it, the median outside share of the
           closed form (cross-checked against the logit log's s0 median), n, r-hat and its anchor, the
           shares at the caps, min G_t, the first-stage F, and the B.6 rule as the plan proposes it,
           M0-M3, against production); rows in the order production, base_star, f, then the rest;
           cap_screen_summary.json (the same, the frame diagnostics, the first stage, gate B-S3: |alpha
           base_star - alpha production| < 0.05 se production in E3 and E4); top_cells_E{k}_{v}.csv for
           base_star, f, p4, p5, p6; MD5SUMS.txt (LF md5 of each output).

Usage:  python diagnostics/diag_cap_sensitivity.py --stage screen --step frames [--variants a,b,...]
        python diagnostics/diag_cap_sensitivity.py --stage screen --step logit [--threads 4]
        python diagnostics/diag_cap_sensitivity.py --stage screen --step summary
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils import paths
from utils import demand_chars

REPO = Path(__file__).resolve().parents[1]
SANDBOX = Path(r"C:\Users\pedro\AppData\Local\Temp\capsens_1111")
OUT_DEFAULT = paths.rout_dir() / "diag_cap_sensitivity"
DP = paths.demand_parquet_dir()
FRAMES = {3: "demand_3_index_spec_12.parquet", 4: "demand_4_index_time_spec_12.parquet"}
PROD_LOGIT = paths.PROCESSED / "ESTIMATION_OUTPUT" / "BLP_RESULTS" / "logit" / "logit_summary_spec_12.json"
BANKED_CSV = paths.INCLUSION_DIR / "bcb_banked_mca_panel.csv"
SUBMODEL = "full_dtype"
# The module constants the stored frames were built with (sleep_demand_prep_link defaults).
PROD = {"PHI_CAP": 0.99, "TOTAL_SHARE_CAP": 0.95, "RMAX_RULE": "q95", "BC_RULE": "findex_gdp",
        "ACCESS_LO": 0.35, "ACCESS_HI": 2.5, "FINDEX_FALLBACK": 0.77}
ENV_KNOBS = ("DEMAND_PHI_CAP", "DEMAND_RMAX_RULE", "DEMAND_BC_RULE", "DEMAND_ACCESS_LO", "DEMAND_ACCESS_HI",
             "DEMAND_FINDEX_FALLBACK", "DEMAND_TOTAL_SHARE_CAP")
VARIANTS = {  # name: (construction, PHI_CAP, TOTAL_SHARE_CAP); the summary keeps this order after production
    "base_star": ("reshare", 0.99, 0.95),
    "f": ("reshare_findex", 0.99, 0.95),
    "p1": ("reshare", 0.98, 0.95),
    "p2": ("reshare", 0.995, 0.95),
    "p4": ("reshare", 0.999, 0.95),
    "p5": ("reshare", 0.9999, 0.95),
    "p6": ("reshare", 0.99999, 0.95),
    "s1": ("reshare", 0.99, 0.90),
    "s2": ("reshare", 0.99, 0.98),
    "p3a": ("drop_phi", 0.99, 0.95),
    "s3a": ("drop_floor", 0.99, 0.95),
    "p3b": ("reshare_drop_phi", 0.99, 0.95),
}
TOP_CELL_VARIANTS = ("base_star", "f", "p4", "p5", "p6")
# blp_logit.jl: X_COLS, then the D-type dummy (full_dtype); the instruments in build_matrices' order.
X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5", "log_total_assets_lag", "is_state_owned"]
IVS = ["loo_log_assets", "mean_loo_log_assets", "loo_equity_ratio", "mean_loo_equity_ratio", "loo_basileia",
       "mean_loo_basileia", "loo_borrow_assets", "mean_loo_borrow_assets", "loo_npl_provision",
       "mean_loo_npl_provision", "n_rivals", "estban_rival_branches_lag", "personnel_cost_ratio_lag",
       "admin_cost_ratio_lag", "tax_cost_ratio_lag", "indice_basileia_lag"]
DROP_PHI = 0.99          # p3: the B rows at or above the production cap


def stamp(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def md5_file(p, lf=False):
    b = Path(p).read_bytes()
    return hashlib.md5(b.replace(b"\r", b"") if lf else b).hexdigest()


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)),
                          encoding="utf-8", newline="\n")


def prep_module():
    """sleep_demand_prep_link with its constants checked against the production defaults."""
    set_knobs = {k: os.environ[k] for k in ENV_KNOBS if k in os.environ}
    if set_knobs:
        raise SystemExit(f"demand-prep knobs exported in the environment: {set_knobs}; unset them")
    import sleep_demand_prep_link as P
    got = {k: getattr(P, k) for k in PROD}
    if got != PROD:
        raise SystemExit(f"sleep_demand_prep_link constants {got} are not the production defaults {PROD}")
    return P


# ------------------------------------------------------------------------------------ the frames
def floor_markets(d, fx):
    """B market-quarters of a frame: Pop, access, M, phi-hat, the share sum, R = the median of
    M fx access / Pop (r-hat wherever the floor does not bind) and the floor flag (M above R Pop / (fx access))."""
    b = d[d["is_B"]]
    key = [b["mca_code"], b["time_id"]]
    mk = pd.DataFrame({"pop": b["pop_total"].groupby(key).first(), "acc": b["_access"].groupby(key).first(),
                       "M": b["M_mt"].groupby(key).first(), "phi": b["phi_mt"].groupby(key).first(),
                       "SB": b["share_B_cond"].groupby(key).sum(), "fx": fx[d["is_B"].to_numpy()].groupby(key).first(),
                       "dep": b["Dep_Act"].groupby(key).sum()})
    rec = mk["M"] * mk["fx"] * mk["acc"] / mk["pop"]
    R = float(rec.median())
    mk["rec"] = rec
    mk["floor"] = rec > R * (1 + 1e-9)
    return mk, R


def anchor(y, phi_cap):
    """r-hat exactly as build_market_size_and_shares forms it, from the frame it returned."""
    key = [y["mca_code"], y["time_id"]]
    mkt = pd.DataFrame({"pop": y["pop_total"].groupby(key).first(), "acc": y["_access"].groupby(key).first()})
    b = y[y["is_B"]]
    kb = [b["mca_code"], b["time_id"]]
    mkt = mkt.join(b["_1mphi"].groupby(kb).first().rename("onem")).join(b["Dep_Act"].groupby(kb).sum().rename("dep"))
    mkt["onem"] = mkt["onem"].fillna(1.0 - phi_cap)
    mkt["dep"] = mkt["dep"].fillna(0.0)
    local = (mkt["dep"] / (mkt["onem"] * mkt["pop"]) * mkt["acc"]).replace([np.inf, -np.inf], np.nan)
    dep1m = (y["Dep_Act"] / y["_1mphi"]).replace([np.inf, -np.inf], np.nan)
    nat = (dep1m.groupby(y["time_id"]).sum() / mkt.groupby(level=1)["pop"].sum()).replace([np.inf, -np.inf], np.nan)
    ql, qn = float(local.dropna().quantile(0.95)), float(nat.dropna().quantile(0.95))
    return max(ql, qn) or 1.0, ql, qn, local


def closed_form(d):
    """blp_logit.jl berry_inversion: the outside share g_mt of each B market-quarter, G_t and K_t."""
    B = d["is_B"].to_numpy(bool)
    b = d[B]
    key = [b["mca_code"], b["time_id"]]
    SB = b["share_B_cond"].groupby(key).sum()
    bcpop = (b["banked_correction"].fillna(1.1) * b["pop_total"].fillna(0)).groupby(key).mean()
    tq = SB.index.get_level_values(1)
    w = bcpop / bcpop.groupby(level=1).transform("sum")
    K = (w * (1 - SB)).groupby(level=1).sum()
    SD = d.loc[~B, "share_D"].groupby(d.loc[~B, "time_id"]).sum().reindex(K.index).fillna(0.0)
    G = K - SD
    g = ((1 - SB) * G.reindex(tq).to_numpy() / K.reindex(tq).to_numpy()).clip(1e-12, 1.0)
    return g, G, K, SB


def mean_rho_1ms(d):
    """blp_logit.jl _mean_rho_one_minus_s: mean of rho (1 - s), rho = spread_ann / 100, s the share of the row."""
    rho = d["spread_ann"].to_numpy(float) / 100.0
    s = np.where(d["is_B"].to_numpy(bool), d["share_B_cond"].to_numpy(float), d["share_D"].to_numpy(float))
    m = np.isfinite(rho) & np.isfinite(s) & (s >= 0) & (s < 1)
    return float(np.mean(rho[m] * (1 - s[m])))


def frame_diag(y, phi_cap, share_cap, fx, recompute_anchor, top_csv=None):
    out = {"rows": int(len(y)), "B_rows": int(y["is_B"].sum()), "D_rows": int((~y["is_B"]).sum()),
           "phi_cap": phi_cap, "share_cap": share_cap}
    mk, R_from_M = floor_markets(y, fx)
    out["B_market_quarters"] = int(len(mk))
    out["rhat_from_M"] = R_from_M
    if recompute_anchor:
        rhat, ql, qn, local = anchor(y, phi_cap)
        out.update(rhat=rhat, rhat_local_q95=ql, rhat_national_q95=qn, anchor="local" if ql >= qn else "national",
                   rhat_vs_from_M_rel=abs(R_from_M / rhat - 1))
        top = local[(local >= ql) & local.index.isin(mk.index)]
        tm = mk.loc[top.index]
        out["rhat_cells"] = {"n": int(len(top)), "phi_median": float(tm["phi"].median()),
                             "phi_p10": float(tm["phi"].quantile(0.10)),
                             "share_phi_ge_cap": float((tm["phi"] >= phi_cap).mean()),
                             "share_phi_ge_099": float((tm["phi"] >= 0.99).mean()),
                             "share_phi_lt_098": float((tm["phi"] < 0.98).mean()),
                             "pop_share": float(tm["pop"].sum() / mk["pop"].sum())}
        if top_csv is not None:
            (tm.assign(ratio=top).sort_values("ratio", ascending=False).head(25)
             .reset_index().rename(columns={"level_0": "mca_code", "level_1": "time_id"})
             [["mca_code", "time_id", "phi", "pop", "acc", "dep", "ratio"]]
             .to_csv(top_csv, index=False, lineterminator="\n"))
    else:
        out.update(rhat=R_from_M, anchor="production's (not recomputed)")
    out["B_mq_floor"] = int(mk["floor"].sum())
    out["B_mq_floor_share"] = float(mk["floor"].mean())
    out["B_mq_share_sum_at_cap"] = int(((mk["SB"] - share_cap).abs() <= 1e-9).sum())
    out["B_share_sum_max"] = float(mk["SB"].max())
    b = y[y["is_B"]]
    out["B_rows_at_phi_cap_share"] = float((b["phi_mt"] >= phi_cap).mean())
    g, G, K, SB = closed_form(y)
    out.update(outside_share_median=float(g.median()), outside_share_p10=float(g.quantile(0.10)),
               G_min=float(G.min()), G_nonpositive_quarters=int((G <= 0).sum()), quarters=int(len(G)),
               inside_sum_median=float(SB.median()))
    low = mk["phi"] < 0.98
    rows_low = b.loc[b["phi_mt"] < 0.98, "share_B_cond"]
    out.update(inside_sum_phi_lt_098_median=float(SB[low.reindex(SB.index).to_numpy()].median()) if low.any() else np.nan,
               inside_sum_phi_lt_098_p10=float(SB[low.reindex(SB.index).to_numpy()].quantile(0.10)) if low.any() else np.nan,
               row_share_phi_lt_098_median=float(rows_low.median()) if len(rows_low) else np.nan,
               row_share_phi_lt_098_p10=float(rows_low.quantile(0.10)) if len(rows_low) else np.nan,
               mean_rho_1ms=mean_rho_1ms(y),
               B_rows_share_below_1e15=int((b["share_B_cond"] < 1e-15).sum()),
               D_rows_share_below_1e15=int((y.loc[~y["is_B"], "share_D"] < 1e-15).sum()))
    return out


def findex_series():
    bk = pd.read_csv(BANKED_CSV, usecols=["mca_code", "year", "findex_banked_frac"], dtype={"mca_code": str, "year": int})
    bk["findex_banked_frac"] = pd.to_numeric(bk["findex_banked_frac"], errors="coerce")
    if bk.duplicated(["mca_code", "year"]).any():
        raise SystemExit(f"{BANKED_CSV.name} is not unique on mca_code x year")
    return bk


def reshare(P, d, phi_cap, share_cap, findex=None):
    """build_market_size_and_shares on the rows of d at the given caps; the frame keeps d's row order,
    index and columns. Returns (frame, fx), fx the Findex fraction each row's market size used."""
    x = d.copy()
    idx = x.index
    if findex is not None:
        x = x.merge(findex, on=["mca_code", "year"], how="left", validate="many_to_one")
        x.index = idx
    keep = (P.PHI_CAP, P.TOTAL_SHARE_CAP)
    try:
        P.PHI_CAP, P.TOTAL_SHARE_CAP = phi_cap, share_cap
        y = P.build_market_size_and_shares(x)
    finally:
        P.PHI_CAP, P.TOTAL_SHARE_CAP = keep
    if len(y) != len(d) or not (y["entity_id"].to_numpy() == d["entity_id"].to_numpy()).all() \
            or not (y["time_id"].to_numpy() == d["time_id"].to_numpy()).all():
        raise SystemExit("build_market_size_and_shares changed the row order")
    y.index = idx
    if findex is not None:
        fx = pd.to_numeric(y["findex_banked_frac"], errors="coerce").fillna(P.FINDEX_FALLBACK).clip(0.3, 1.0)
        y = y.drop(columns=["findex_banked_frac"])
    else:
        fx = pd.Series(P.FINDEX_FALLBACK, index=y.index)
    return y[d.columns], fx


def write_frame(y, schema0, path, meta):
    t = pa.Table.from_pandas(y)
    if t.schema.names != schema0.names:
        raise SystemExit(f"{path.name}: columns differ from the stored frame's")
    if not t.schema.remove_metadata().equals(schema0.remove_metadata()):
        t = t.cast(schema0.remove_metadata())
    md = dict(t.schema.metadata or {})
    md[demand_chars.PANEL_MD5_KEY] = schema0.metadata[demand_chars.PANEL_MD5_KEY]
    md.update({f"capsens.{k}".encode(): str(v).encode() for k, v in meta.items()})
    t = t.replace_schema_metadata(md)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    pq.write_table(t, part)
    os.replace(part, path)


def first_stage(d):
    """blp_logit.jl project_spreads, k = 4 and 5: spread (pp) on [X_COLS, D dummy, the 16 instruments]
    within the type, no intercept (as the logit), instruments with no variation over the rows dropped as
    build_matrices does; X columns that are zero throughout the type dropped (the fitted values do not
    change). Conglomerate-clustered (CR1) Wald F of the instruments, the Montiel Olea-Pflueger effective
    F = pi' Zt'Zt pi / tr(V_pi Zt'Zt) with Zt the instruments net of X, and the homoskedastic F."""
    X = d[X_COLS].to_numpy(float)
    X = np.column_stack([X, (~d["is_B"].to_numpy(bool)).astype(float)])
    xn = X_COLS + ["dummy_D_type"]
    Z = d[IVS].to_numpy(float)
    if not (np.isfinite(X).all() and np.isfinite(Z).all() and np.isfinite(d["spread_ann"].to_numpy(float)).all()):
        raise SystemExit("first stage: a missing price, characteristic or instrument")
    zk = Z.std(axis=0) > 1e-10
    Z = Z[:, zk]
    y = d["spread_ann"].to_numpy(float) / 100.0
    cl = pd.factorize(d["CodConglomeradoPrudencial"].astype(str))[0]
    dt = d["deposit_type"].to_numpy()
    out = {"instruments_kept": [n for n, k in zip(IVS, zk) if k]}
    for k in (4, 5):
        r = dt == k
        Xk, Zk, yk, ck = X[r], Z[r], y[r], cl[r]
        xkeep = np.abs(Xk).max(axis=0) > 0
        Xk = Xk[:, xkeep]
        H = np.column_stack([Xk, Zk])
        beta, *_ = np.linalg.lstsq(H, yk, rcond=None)
        u = yk - H @ beta
        cz, *_ = np.linalg.lstsq(Xk, Zk, rcond=None)
        Zt = Zk - Xk @ cz
        ZZ = Zt.T @ Zt
        ZZi = np.linalg.inv(ZZ)
        pi = ZZi @ (Zt.T @ yk)
        n, L, K = len(yk), Zt.shape[1], H.shape[1]
        G = int(ck.max() + 1) if n else 0
        cids = pd.factorize(ck)[0]
        G = int(cids.max() + 1)
        Sg = np.zeros((G, L))
        np.add.at(Sg, cids, Zt * u[:, None])
        corr = G / (G - 1) * (n - 1) / (n - K)
        V = corr * ZZi @ (Sg.T @ Sg) @ ZZi
        wald = float(pi @ np.linalg.solve(V, pi) / L)
        eff = float(pi @ ZZ @ pi / np.trace(V @ ZZ))
        s2 = float(u @ u / (n - K))
        homo = float(pi @ ZZ @ pi / L / s2)
        out[f"k{k}"] = {"n": int(n), "clusters": G, "instruments": int(L), "regressors": int(K),
                        "x_dropped_zero": [c for c, kk in zip(xn, xkeep) if not kk],
                        "pi_equals_full_regression_max_abs": float(np.max(np.abs(pi - beta[Xk.shape[1]:]))),
                        "wald_F_cluster": wald, "effective_F_MOP": eff, "F_homoskedastic": homo}
    return out


def step_frames(variants, sb):
    P = prep_module()
    bk = findex_series()
    sb.mkdir(parents=True, exist_ok=True)
    marker = sb / "FRAMES_COMPLETE.json"
    if marker.exists():
        os.replace(marker, sb / "FRAMES_COMPLETE.previous.json")
    manifest = {"started": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "variants": {}, "sources": {}}
    diag = {v: {} for v in variants}
    prod_diag, fs = {}, {}
    for E, name in FRAMES.items():
        src = DP / name
        t0 = pq.read_table(src)
        schema0 = t0.schema
        d = t0.to_pandas()
        del t0
        d["is_B"] = d["is_B"].astype(bool)
        manifest["sources"][f"E{E}"] = {"path": str(src), "md5": md5_file(src),
                                        "panel_md5": schema0.metadata[demand_chars.PANEL_MD5_KEY].decode()}
        fx0 = pd.Series(P.FINDEX_FALLBACK, index=d.index)
        mk0, R0 = floor_markets(d, fx0)
        floor_keys = set(mk0.index[mk0["floor"]])
        pair = pd.MultiIndex.from_arrays([d["mca_code"], d["time_id"]])
        drop_phi = d["is_B"].to_numpy() & (d["phi_mt"].to_numpy() >= DROP_PHI)
        drop_floor = d["is_B"].to_numpy() & pair.isin(floor_keys)
        prod_diag[f"E{E}"] = frame_diag(d, PROD["PHI_CAP"], PROD["TOTAL_SHARE_CAP"], fx0, recompute_anchor=False)
        fs[f"E{E}"] = {"stored": first_stage(d), "p3": first_stage(d[~drop_phi]), "s3a": first_stage(d[~drop_floor])}
        stamp(f"E{E}: stored frame {len(d):,} rows; floor markets {len(floor_keys):,}; first stage k4 effective F "
              f"{fs[f'E{E}']['stored']['k4']['effective_F_MOP']:.2f}")
        for v in variants:
            kind, pc, sc = VARIANTS[v]
            if kind == "reshare":
                y, fx = reshare(P, d, pc, sc)
            elif kind == "reshare_findex":
                y, fx = reshare(P, d, pc, sc, findex=bk)
            elif kind == "reshare_drop_phi":
                y, fx = reshare(P, d[~drop_phi], pc, sc)
            elif kind == "drop_phi":
                y, fx = d[~drop_phi], fx0[~drop_phi]
            elif kind == "drop_floor":
                y, fx = d[~drop_floor], fx0[~drop_floor]
            out = sb / v / "demand_prep" / name
            write_frame(y, schema0, out, {"variant": v, "construction": kind, "phi_cap": pc, "share_cap": sc,
                                          "source_md5": manifest["sources"][f"E{E}"]["md5"]})
            top = sb / v / f"top_cells_E{E}.csv" if kind.startswith("reshare") else None
            diag[v][f"E{E}"] = frame_diag(y, pc, sc, fx, recompute_anchor=kind.startswith("reshare"), top_csv=top)
            if kind == "reshare_findex":
                diag[v][f"E{E}"]["findex_rows_matched_share"] = float(
                    d["mca_code"].astype(str).isin(set(bk["mca_code"])).mean())
                diag[v][f"E{E}"]["findex_by_year_B"] = {int(yy): float(g) for yy, g in
                                                        fx[y["is_B"].to_numpy()].groupby(y.loc[y["is_B"], "year"]).median().items()}
            manifest["variants"].setdefault(v, {})[f"E{E}"] = {"path": str(out), "md5": md5_file(out),
                                                              "rows": int(len(y)), "bytes": out.stat().st_size}
            r = diag[v][f"E{E}"]
            stamp(f"E{E} {v}: {len(y):,} rows, r-hat {r['rhat']:.6g} ({r['anchor']}), floor {r['B_mq_floor_share']:.4f}, "
                  f"outside median {r['outside_share_median']:.4f}, G_min {r['G_min']:.4f}")
            del y, fx
        del d
    for v in variants:
        jdump(diag[v], sb / v / "frame_diag.json")
    jdump(prod_diag, sb / "production_frame_diag.json")
    jdump(fs, sb / "first_stage.json")
    manifest["finished"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    jdump(manifest, marker)
    stamp(f"frames complete: {marker}")


# ------------------------------------------------------------------------------------- the logit
def julia_env():
    return {k: v for k, v in os.environ.items()
            if not (k.startswith("BLP_") or k.startswith("DEMAND_") or k == "SLEEP_ACTIVE_ESTS")}


def step_env():
    jl = shutil.which("julia")
    if jl is None:
        raise SystemExit("julia is not on PATH")
    t = time.time()
    r = subprocess.run([jl, "--project=.", "-e", "using Parquet2, DataFrames, JSON3, Distributions; println(VERSION)"],
                       cwd=REPO, capture_output=True, text=True, env=julia_env())
    stamp(f"B-S0: {jl} exit {r.returncode} in {time.time() - t:.0f} s; {r.stdout.strip()} {r.stderr.strip()[-400:]}")
    if r.returncode:
        raise SystemExit("B-S0 failed")


def step_logit(variants, sb, threads):
    if not (sb / "FRAMES_COMPLETE.json").exists():
        raise SystemExit(f"{sb / 'FRAMES_COMPLETE.json'} is missing: the frame set is partial; run --step frames")
    jl = shutil.which("julia")
    for v in variants:
        dp, od = sb / v / "demand_prep", sb / v / "out"
        summ = od / "logit" / "logit_summary_spec_12.json"
        newest = max(p.stat().st_mtime for p in dp.glob("*.parquet"))
        if summ.exists() and summ.stat().st_mtime > newest:
            stamp(f"B-S2 {v}: summary newer than its frames, skipped")
            continue
        t = time.time()
        with open(sb / v / "logit.log", "w", encoding="utf-8") as fh:
            r = subprocess.run([jl, "--project=.", f"--threads={threads}", "blp_logit.jl", "--demand-dir", str(dp),
                                "--out-dir", str(od)], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=julia_env())
        stamp(f"B-S2 {v}: exit {r.returncode} in {time.time() - t:.0f} s")


# ----------------------------------------------------------------------------------- the summary
def logit_row(summ, E):
    r = summ.get(f"E{E}_{SUBMODEL}")
    if r is None:
        return None
    i = r["param_names"].index("alpha")
    return {"alpha": float(r["theta1"][i]), "se": float(r["se"][i]), "p": float(r["pval"][i]),
            "G_star": float(r["G_star"]), "Q": float(r["Q_value"]), "n_obs": int(r["n_obs"]),
            "se_method": r.get("se_method")}


def log_s0(log):
    """The logit log's s0 median per routine."""
    out, cur = {}, None
    for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"Estimation E(\d+)", line)
        if m:
            cur = int(m.group(1))
        m = re.search(r"median\s+([0-9.]+)", line)
        if m and cur is not None and "s₀" in line:
            out[cur] = float(m.group(1))
    return out


def step_summary(variants, sb, out):
    out.mkdir(parents=True, exist_ok=True)
    prod_summ = json.loads(PROD_LOGIT.read_text(encoding="utf-8"))
    prod_diag = json.loads((sb / "production_frame_diag.json").read_text(encoding="utf-8"))
    fs = json.loads((sb / "first_stage.json").read_text(encoding="utf-8"))
    rows, missing = [], []
    order = ["production"] + [v for v in VARIANTS if v in variants]
    sample_of = {"p3a": "p3", "p3b": "p3", "s3a": "s3a"}
    for E in FRAMES:
        base = logit_row(prod_summ, E)
        base["mean_rho_1ms"] = prod_diag[f"E{E}"]["mean_rho_1ms"]
        base["elasticity"] = base["alpha"] * base["mean_rho_1ms"]
        for v in order:
            if v == "production":
                lr, fd, s0 = base, prod_diag[f"E{E}"], {}
                kind, pc, sc = "production", PROD["PHI_CAP"], PROD["TOTAL_SHARE_CAP"]
            else:
                kind, pc, sc = VARIANTS[v]
                fd = json.loads((sb / v / "frame_diag.json").read_text(encoding="utf-8"))[f"E{E}"]
                sp = sb / v / "out" / "logit" / "logit_summary_spec_12.json"
                lr = logit_row(json.loads(sp.read_text(encoding="utf-8")), E) if sp.exists() else None
                s0 = log_s0(sb / v / "logit.log") if (sb / v / "logit.log").exists() else {}
                if lr is None:
                    missing.append(f"{v} E{E}")
                    lr = {}
                lr["mean_rho_1ms"] = fd["mean_rho_1ms"]
                lr["elasticity"] = lr["alpha"] * fd["mean_rho_1ms"] if "alpha" in lr else np.nan
            f1 = fs[f"E{E}"][sample_of.get(v, "stored")]
            row = {"routine": f"E{E}", "variant": v, "construction": kind, "phi_cap": pc, "share_cap": sc,
                   "alpha": lr.get("alpha", np.nan), "se_wcb": lr.get("se", np.nan), "p": lr.get("p", np.nan),
                   "elasticity_mean": lr.get("elasticity", np.nan), "mean_rho_1ms": lr.get("mean_rho_1ms", np.nan),
                   "outside_share_median": fd["outside_share_median"], "outside_share_median_logit_log": s0.get(E, np.nan),
                   "n_obs": lr.get("n_obs", fd["rows"]), "G_star": lr.get("G_star", np.nan), "Q": lr.get("Q", np.nan),
                   "rhat": fd["rhat"], "rhat_anchor": fd["anchor"], "B_rows_at_phi_cap": fd["B_rows_at_phi_cap_share"],
                   "B_mq_floor_share": fd["B_mq_floor_share"], "G_min": fd["G_min"],
                   "inside_sum_phi_lt_098_median": fd["inside_sum_phi_lt_098_median"],
                   "inside_sum_phi_lt_098_p10": fd["inside_sum_phi_lt_098_p10"],
                   "row_share_phi_lt_098_median": fd["row_share_phi_lt_098_median"],
                   "rhat_cells_n": fd.get("rhat_cells", {}).get("n", np.nan),
                   "rhat_cells_share_phi_ge_cap": fd.get("rhat_cells", {}).get("share_phi_ge_cap", np.nan),
                   "rhat_cells_share_phi_lt_098": fd.get("rhat_cells", {}).get("share_phi_lt_098", np.nan),
                   "rhat_cells_pop_share": fd.get("rhat_cells", {}).get("pop_share", np.nan),
                   "fs_k4_wald_F": f1["k4"]["wald_F_cluster"], "fs_k4_effective_F": f1["k4"]["effective_F_MOP"],
                   "fs_k5_wald_F": f1["k5"]["wald_F_cluster"], "fs_k5_effective_F": f1["k5"]["effective_F_MOP"]}
            a, s_, e0 = row["alpha"], base["se"], base["elasticity"]
            row["d_alpha_vs_production"] = a - base["alpha"]
            row["d_alpha_over_se_production"] = (a - base["alpha"]) / s_
            row["M0_admissible"] = bool(row["G_min"] > 0 and row["inside_sum_phi_lt_098_median"] >= 0.01)
            row["M1_elasticity_10pct"] = bool(abs(row["elasticity_mean"] / e0 - 1) >= 0.10) if np.isfinite(a) else False
            row["M2_half_se"] = bool(abs(a - base["alpha"]) >= 0.5 * s_) if np.isfinite(a) else False
            row["M3_5pct_verdict"] = bool((row["p"] < 0.05) != (base["p"] < 0.05)) if np.isfinite(row["p"]) else False
            row["triggers_B6"] = bool(row["M0_admissible"] and (row["M1_elasticity_10pct"] or row["M2_half_se"]
                                                                or row["M3_5pct_verdict"])) and v != "production"
            rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(out / "cap_screen_summary.csv", index=False, lineterminator="\n")
    gate = {}
    for E in FRAMES:
        b = S[(S["routine"] == f"E{E}") & (S["variant"] == "production")].iloc[0]
        st = S[(S["routine"] == f"E{E}") & (S["variant"] == "base_star")]
        if len(st) and np.isfinite(st["alpha"].iloc[0]):
            dd = abs(st["alpha"].iloc[0] - b["alpha"])
            gate[f"E{E}"] = {"abs_d_alpha": float(dd), "bound": float(0.05 * b["se_wcb"]), "pass": bool(dd < 0.05 * b["se_wcb"])}
    trig = sorted(set(S.loc[S["triggers_B6"], "variant"]))
    J = {"generated": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "gate_B_S3": gate, "missing_logits": missing,
         "B6_rule": "as the plan proposes it (netflow_gap_cap_plan_1111.md B.6), against production: M0 G_t > 0 in "
                    "every quarter and the median inside sum of markets with phi-hat < 0.98 >= 0.01; M1 the mean "
                    "elasticity moves >= 10%; M2 |d alpha| >= 0.5 se; M3 the 5% verdict changes",
         "variants_meeting_B6": trig, "rows": S.to_dict(orient="records"),
         "first_stage": fs, "production_frame_diag": prod_diag,
         "frame_diag": {v: json.loads((sb / v / "frame_diag.json").read_text(encoding="utf-8")) for v in variants},
         "frames_manifest": json.loads((sb / "FRAMES_COMPLETE.json").read_text(encoding="utf-8"))}
    jdump(J, out / "cap_screen_summary.json")
    for v in TOP_CELL_VARIANTS:
        for E in FRAMES:
            p = sb / v / f"top_cells_E{E}.csv"
            if v in variants and p.exists():
                shutil.copyfile(p, out / f"top_cells_E{E}_{v}.csv")
    lines = [f"{md5_file(p, lf=p.suffix in ('.csv', '.json', '.txt'))}  {p.name}" for p in sorted(out.iterdir())
             if p.is_file() and p.name != "MD5SUMS.txt"]
    (out / "MD5SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(S[["routine", "variant", "alpha", "se_wcb", "p", "elasticity_mean", "outside_share_median",
             "outside_share_median_logit_log", "rhat", "B_mq_floor_share", "G_min", "triggers_B6"]].round(4).to_string())
    stamp(f"summary: gate {json.dumps(gate)}; missing {missing}; meeting B.6 {trig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["screen"])
    ap.add_argument("--step", default="frames")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--sandbox", default=str(SANDBOX))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    variants = [v for v in a.variants.split(",") if v]
    bad = [v for v in variants if v not in VARIANTS]
    steps = a.step.split(",")
    if bad or set(steps) - {"env", "frames", "logit", "summary"}:
        raise SystemExit(f"unknown variants {bad} or steps {steps}")
    sb = Path(a.sandbox).resolve()
    stamp(f"stage {a.stage}, steps {steps}, variants {variants}, sandbox {sb}")
    for s in steps:
        t = time.time()
        if s == "env":
            step_env()
        elif s == "frames":
            step_frames(variants, sb)
        elif s == "logit":
            step_logit(variants, sb, a.threads)
        elif s == "summary":
            step_summary(variants, sb, Path(a.out_dir).resolve())
        stamp(f"step {s} done in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
