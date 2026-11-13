"""
diag_market_size_a.py
=====================
Variant (a) of the market-size follow-up (claude/MAIN_REPLY_2026-11-12.md, "Update 11-12 20:05", item 1;
Pedro's go), built on both fixes of 11-12 (the D market from the B markets only; the national phi_t over
the B markets only). A benchmark for the new-money construction of Step 1. Local and read-only on
production: the frames are re-shared in a sandbox outside OneDrive and the logit is run there.

Source rows: the fix agent's full-prep frames under both fixes (dmkt_fix_1112/phi_1112/both/demand_prep).
They are post-finalize rows: demand_chars.finalize drops rows after the shares are formed (E3: 45 B rows
and 65 D rows), so a re-share of them is not the full prep. Gate P measures the difference at production's
settings, and gate L its effect on the logit.

Notation (diag_market_size_alt.py's): market m = mca_code, quarter t, row j = bank x market x type;
acc_m the access index; fx_t the Findex fraction by year (0.77 at production's settings); bc = 1/(fx acc);
S_mt the booked B stock (panel dep_a1+a2+a4+a5, diag_market_size_alt.panel_stock);
X_m = sum over the B rows of m of Dep_Act_j / (1 - min(phi_mt, c_phi)); X_D,t the same over the D rows of t
with phi_t.

Variants:
  prod     production's function (sleep_demand_prep_link) at production's constants: the pooled density
           anchor, phi cap 0.99, floor 0.95, fx 0.77. Frames and logit: gates P and L.
  A1q95f   the stock anchor by quarter, r_t = Q95 over the B markets of t of (S_mt / Pop_mt) acc_m, fx by
           year, phi cap 0.99, floor 0.95: M_m = max(r_t bc_m Pop_m, X_m / 0.95); the D market
           M_nat,t = max(r_t sum_m bc_m Pop_m, X_D,t / 0.95) over the B markets (the fixed rule).
           In memory only: gate A (its B side equals msize_1112's A1q95 frames) and W4.
  a080, a090, a095   variant (a): the stock anchor as A1q95f, phi cap 0.999, NO floor:
           M_m = r_t bc_m Pop_m and M_nat,t = r_t sum_m bc_m Pop_m. In every B market with X_m > c M_m,
           every active balance is scaled by k_m = c M_m / X_m, so the inside share is exactly c; the D market
           likewise with k_D,t = c M_nat,t / X_D,t where X_D,t > c M_nat,t. Elsewhere k = 1. Where phi is
           constant within the market, k_m = c (1 - phi_c) M_m / sum_j Dep_Act_j and s_j = c Dep_Act_j /
           sum_j Dep_Act_j. The frame's Dep_Act is the scaled balance. Frames and logit.
Gates:
  G0  my build at production's settings equals production's function on the same rows, every column exactly
  P   production's function on the source rows against the source frames: share_B_cond, share_D, M_mt, M_nat
  A   A1q95f's B side (M_mt, share_B_cond) equals msize_1112's A1q95 frames (old D rule) row for row; the
      D market's M_nat old over new, by quarter
  L   (summary) the logit on prod's frames against the fix agent's: |d alpha| <= 0.05 se; the D dummy
Steps (--step, comma list; --routines 3,4):
  frames   per routine: the builds, gates G0/P/A, the frames of prod and a0xx to <sandbox>/<v>/E<k>/demand_prep,
           diagnostics to <sandbox>/diag_E<k>.json, the B market tables (M per variant) to <sandbox>/mkt_E<k>.parquet,
           and the 19:55 table X / S by bins of the uncapped phi-hat to <sandbox>/xs_bins_E<k>.csv
  logit    julia blp_logit.jl per variant and routine (skipped where G_t <= 0 in some quarter, or where the
           summary is newer than the frame); waits while another python/julia process holds > 1 GB
  w4       W4.1's pools and the composition of W (diag_market_size_alt.W4, "cost_only") for the E3 market
           sizes of production with both fixes (the source frame), A1q95f and (a) -> <sandbox>/w4.json
  summary  -> --out-dir (Rout/diag_market_size_a/): market_size_a_summary.csv/.json, the 19:55 tables, MD5SUMS.txt
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import csv
import io
import json
import shutil
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from utils import paths
from utils import demand_chars
from utils import state_transform as _st
import diag_cap_sensitivity as dcs
import diag_market_size_alt as alt

REPO = Path(__file__).resolve().parents[1]
SANDBOX = Path(r"C:\Users\pedro\AppData\Local\Temp\msize_a_1112")
SRC = Path(r"C:\Users\pedro\AppData\Local\Temp\dmkt_fix_1112\phi_1112")
SRC_DP = SRC / "both" / "demand_prep"
SRC_LOGIT = SRC / "both" / "out" / "logit" / "logit_summary_spec_12.json"
SRC_VERIFY = {3: SRC / "verify_phi_E3.json", 4: SRC / "verify_phi_E4.json"}
ALT_A1 = Path(r"C:\Users\pedro\AppData\Local\Temp\msize_1112\A1q95\demand_prep")
OUT_DEFAULT = paths.rout_dir() / "diag_market_size_a"
FRAMES = dcs.FRAMES
SUBMODEL = dcs.SUBMODEL
CS = (0.80, 0.90, 0.95)
A_VARIANTS = {f"a{int(round(c * 100)):03d}": c for c in CS}
AK_VARIANTS = {f"a{int(round(c * 100)):03d}K": c for c in CS}
LOGIT_VARIANTS = ("prod",) + tuple(A_VARIANTS) + tuple(AK_VARIANTS)
PHI_BINS = [(-np.inf, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 0.99), (0.99, 0.995), (0.995, 0.999), (0.999, np.inf)]
BIN_LABELS = ["<0.900", "0.900-0.950", "0.950-0.980", "0.980-0.990", "0.990-0.995", "0.995-0.999", ">=0.999"]
LSP_PID = 19088            # the editor's Julia language server (coordinator, 11-12)
HEAVY_KB = 1024 * 1024     # 1 GB in tasklist's K

stamp = dcs.stamp
md5_file = dcs.md5_file
jdump = dcs.jdump


def cfg_of(v):
    if v == "prod":
        return dict(anchor="prod", findex="const", phi_cap=0.99, mode="floor", share_cap=0.95)
    if v == "A1q95f":
        return dict(anchor="stock", findex="year", q=0.95, phi_cap=0.99, mode="floor", share_cap=0.95)
    if v in A_VARIANTS:
        return dict(anchor="stock", findex="year", q=0.95, phi_cap=0.999, mode="rescale", c=A_VARIANTS[v])
    if v in AK_VARIANTS:
        return dict(anchor="stock", findex="year", q=0.95, phi_cap=0.999, mode="rescale", c=AK_VARIANTS[v],
                    d_rule="K")
    raise SystemExit(f"variant {v}")


# ------------------------------------------------------------------------------- the market size
def build(P, d, cfg, fx_year, S_full):
    """sleep_demand_prep_link.build_market_size_and_shares (both fixes) with the anchor and the floor
    generalised. At production's settings every operation is production's, in production's order (gate G0).
    Returns (frame in d's row order and columns, info)."""
    x = alt._with_findex(d, {"findex": cfg["findex"]}, fx_year)
    phi_cap = cfg["phi_cap"]
    x['_1mphi'] = 1.0 - np.where(x['is_B'], x['phi_mt'], x['phi_t']).clip(0.0, phi_cap)
    _fx = pd.to_numeric(x.get('findex_banked_frac'), errors='coerce').fillna(P.FINDEX_FALLBACK) \
        if 'findex_banked_frac' in x.columns else pd.Series(P.FINDEX_FALLBACK, index=x.index)
    _fx = _fx.clip(0.3, 1.0)
    if P.BC_RULE != "findex_gdp":
        raise SystemExit(f"BC_RULE {P.BC_RULE}: build mirrors findex_gdp only")
    _x = pd.to_numeric(x['gdp_per_capita'], errors='coerce')
    if _st.centering_enabled():
        _x = _x + _st.load_transform().mean_scaled['gdp_per_capita']
    _med = _x.groupby(x['year']).transform('median')
    _acc = (_x / _med).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(P.ACCESS_LO, P.ACCESS_HI)
    x['_access'] = _acc
    x['_bc'] = 1.0 / (_fx * _acc)
    keys = ['mca_code', 'time_id']
    _b = x[x['is_B']]
    mkt = _b.groupby(keys).agg(_pop=('pop_total', 'first'), _bc_mt=('_bc', 'first'),
                               _acc_mt=('_access', 'first'), _1mphi=('_1mphi', 'first'),
                               _dep_B=('Dep_Act', 'sum'))
    quarters = pd.Index(sorted(set(mkt.index.get_level_values('time_id'))), name='time_id')
    tq = mkt.index.get_level_values('time_id')
    info = {"cfg": cfg}
    FALLBACK_BC = 1.1
    if cfg["anchor"] == "prod":
        if P.RMAX_RULE != "q95":
            raise SystemExit(f"RMAX_RULE {P.RMAX_RULE}")
        local_ratio = ((mkt['_dep_B'] / (mkt['_1mphi'] * mkt['_pop'])) * mkt['_acc_mt']) \
            .replace([np.inf, -np.inf], np.nan)
        dep_over_1mphi = (x['Dep_Act'] / x['_1mphi']).replace([np.inf, -np.inf], np.nan)
        all_dep_t = dep_over_1mphi.groupby(x['time_id']).sum()
        nat_pop_t = mkt.groupby('time_id')['_pop'].sum()
        nat_ratio = (all_dep_t / nat_pop_t).replace([np.inf, -np.inf], np.nan)
        rl, rn = alt._quant(local_ratio, 0.95), alt._quant(nat_ratio, 0.95)
        max_ratio = cfg.get("r_fixed") or (max(rl, rn) or 1.0)
        info.update(r_local=rl, r_national=rn)
        r_t = pd.Series(max_ratio, index=quarters)
        mkt['_b_mkt'] = mkt['_bc_mt'] * max_ratio * mkt['_pop']
    elif cfg["anchor"] == "stock":
        S = S_full.reindex(mkt.index)
        dens_s = ((S / mkt['_pop']) * mkt['_acc_mt']).replace([np.inf, -np.inf], np.nan)
        if dens_s.isna().any():
            raise SystemExit(f"stock anchor: {int(dens_s.isna().sum())} B market-quarters without a panel stock")
        r_t = dens_s.groupby(level='time_id').quantile(cfg["q"]).reindex(quarters)
        mkt['_b_mkt'] = mkt['_bc_mt'] * r_t.reindex(tq).to_numpy() * mkt['_pop']
    else:
        raise SystemExit(f"anchor {cfg['anchor']}")
    if not (np.isfinite(r_t.to_numpy()).all() and (r_t > 0).all()):
        raise SystemExit("an anchor r_t is missing or not positive")
    info["r_t"] = r_t
    Xm = (_b.assign(_x=lambda z: z['Dep_Act'] / z['_1mphi']).groupby(keys)['_x'].sum()) \
        .reindex(mkt.index).replace([np.inf, -np.inf], np.nan)
    M_anchor = mkt['_b_mkt'].copy()
    if cfg["mode"] == "floor":
        _floor_B = Xm / cfg["share_cap"]
        bind = _floor_B > mkt['_b_mkt']
        mkt['_b_mkt'] = np.maximum(mkt['_b_mkt'], _floor_B.fillna(0.0))
        k_m = pd.Series(1.0, index=mkt.index)
    elif cfg["mode"] == "rescale":
        c = cfg["c"]
        bind = Xm > c * mkt['_b_mkt']
        k_m = pd.Series(np.where(bind, c * mkt['_b_mkt'] / Xm, 1.0), index=mkt.index)
    else:
        raise SystemExit(f"mode {cfg['mode']}")
    info["mkt"] = pd.DataFrame({"M": mkt['_b_mkt'], "M_anchor": M_anchor, "X": Xm, "bind": bind, "k": k_m,
                                "pop": mkt['_pop'], "acc": mkt['_acc_mt'], "bc": mkt['_bc_mt'],
                                "dep": mkt['_dep_B'], "phi_first": 1.0 - mkt['_1mphi']})
    x = x.merge(mkt[['_b_mkt']].reset_index(), on=keys, how='left')
    mkt_r = mkt.reset_index()
    mkt_r['_w'] = mkt_r['_bc_mt'] * mkt_r['_pop']
    if cfg["anchor"] == "prod":                                         # production's lines
        nat_dbar = (mkt_r.groupby('time_id')
                    .apply(lambda g: (g['_w'].sum() / g['_pop'].sum()) * max_ratio if g['_pop'].sum() > 0
                           else FALLBACK_BC * max_ratio, include_groups=False).rename('_dbar_nat'))
    else:
        sw = mkt_r.groupby('time_id')['_w'].sum()
        sp = mkt_r.groupby('time_id')['_pop'].sum()
        nat_dbar = ((sw / sp).where(sp > 0, FALLBACK_BC) * r_t.reindex(sw.index)).rename('_dbar_nat')
    nat_pop = mkt_r.groupby('time_id')['_pop'].sum().rename('_pop_nat')
    d_mkt = (nat_dbar * nat_pop).rename('_d_mkt').reset_index()
    _d_num = (x.loc[~x['is_B']].assign(_x=lambda z: z['Dep_Act'] / z['_1mphi'])
              .groupby('time_id')['_x'].sum().rename('_dX'))
    d_mkt = d_mkt.merge(_d_num.reset_index(), on='time_id', how='left')
    d_anchor = d_mkt['_d_mkt'].copy()
    if cfg["mode"] == "floor":
        _dfl = d_mkt['_dX'] / cfg["share_cap"]
        d_bind = (_dfl > d_mkt['_d_mkt']).to_numpy()
        d_mkt['_d_mkt'] = np.maximum(d_mkt['_d_mkt'], _dfl.fillna(0.0))
        d_mkt['_kD'] = 1.0
    else:
        c = cfg["c"]
        d_bind = (d_mkt['_dX'] > c * d_mkt['_d_mkt']).to_numpy()
        d_mkt['_kD'] = np.where(d_bind, c * d_mkt['_d_mkt'] / d_mkt['_dX'], 1.0)
        if cfg.get("d_rule") == "K":       # the D rule is applied below, against K_t
            d_bind = np.zeros(len(d_mkt), bool)
            d_mkt['_kD'] = 1.0
    info["d_mkt"] = pd.DataFrame({"M_nat": d_mkt['_d_mkt'].to_numpy(), "M_nat_anchor": d_anchor.to_numpy(),
                                  "X_D": d_mkt['_dX'].to_numpy(), "bind": d_bind, "k": d_mkt['_kD'].to_numpy()},
                                 index=d_mkt['time_id'].to_numpy())
    x = x.merge(d_mkt[['time_id', '_d_mkt', '_kD']], on='time_id', how='left')
    if cfg["mode"] == "rescale":
        x = x.merge(k_m.rename('_kB').reset_index(), on=keys, how='left')
        kk = np.where(x['is_B'], x['_kB'], x['_kD'])
        if not np.isfinite(kk).all():
            raise SystemExit("a scale factor is missing")
        x['Dep_Act'] = x['Dep_Act'] * kk
        x = x.drop(columns=['_kB'])
    if cfg.get("d_rule") == "K":
        # (a') the D market's inside share bounded in the engine's own pool: the D products hold at most c of
        # the consumers the B products leave, S^D_t <= c K_t, K_t = sum_m w_mt (1 - S^B_mt) with blp_logit.jl's
        # weights (per-market mean of banked_correction x pop over the B rows, normalised in the quarter).
        # Then G_t = K_t - S^D_t >= (1 - c) K_t > 0.
        c = cfg["c"]
        bm = x['is_B'].to_numpy(bool)
        kb = [x.loc[bm, 'mca_code'], x.loc[bm, 'time_id']]
        SBm = (x.loc[bm, 'Dep_Act'] / (x.loc[bm, '_1mphi'] * x.loc[bm, '_b_mkt'])).groupby(kb).sum()
        bcpop = (x.loc[bm, 'banked_correction'].fillna(1.1) * x.loc[bm, 'pop_total'].fillna(0.0)).groupby(kb).mean()
        wq_ = bcpop / bcpop.groupby(level=1).transform('sum')
        K = (wq_ * (1.0 - SBm)).groupby(level=1).sum()
        sD0 = x.loc[~bm, 'Dep_Act'] / (x.loc[~bm, '_1mphi'] * x.loc[~bm, '_d_mkt'])
        SD0 = sD0.groupby(x.loc[~bm, 'time_id']).sum()
        Kd = K.reindex(SD0.index)
        bindK = SD0 > c * Kd
        kD = pd.Series(np.where(bindK, c * Kd / SD0, 1.0), index=SD0.index)
        x.loc[~bm, 'Dep_Act'] = x.loc[~bm, 'Dep_Act'].to_numpy() * x.loc[~bm, 'time_id'].map(kD).to_numpy()
        dmi = info["d_mkt"]
        dmi["bind"] = bindK.reindex(dmi.index).fillna(False).to_numpy(bool)
        dmi["k"] = kD.reindex(dmi.index).fillna(1.0).to_numpy()
        dmi["K_t"] = K.reindex(dmi.index).to_numpy()
        dmi["S_D_before"] = SD0.reindex(dmi.index).to_numpy()
    x['share_B_cond'] = np.where(x['is_B'], x['Dep_Act'] / (x['_1mphi'] * x['_b_mkt']), np.nan)
    x['share_D'] = np.where(~x['is_B'], x['Dep_Act'] / (x['_1mphi'] * x['_d_mkt']), np.nan)
    x['M_mt'] = x['_b_mkt']
    x['M_nat'] = x['_d_mkt']
    x = x.drop(columns=['_b_mkt', '_d_mkt', '_bc', '_kD'])
    return alt._restore(x, d), info


def prod_function(P, d):
    """Production's own function on a copy of d at production's constants (fx 0.77)."""
    return alt.prod_build(P, d, alt.PROD_CFG, None)


# ----------------------------------------------------------------------------------- diagnostics
def qtab(v, w=None, qs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if w is not None:
        w = np.asarray(w, float)
        ok &= np.isfinite(w) & (w > 0)
    v = v[ok]
    if not len(v):
        return {"n": 0}
    if w is None:
        out = {f"p{int(round(q * 100)):02d}": float(np.quantile(v, q)) for q in qs}
        out["mean"] = float(v.mean())
    else:
        w = w[ok]
        o = np.argsort(v)
        cw = np.cumsum(w[o]) / w.sum()
        out = {f"p{int(round(q * 100)):02d}": float(v[o][min(np.searchsorted(cw, q), len(v) - 1)]) for q in qs}
        out["mean"] = float(np.average(v, weights=w))
    out["min"] = float(v.min())
    out["max"] = float(v.max())
    out["n"] = int(len(v))
    return out


def diag(y, info, S_full):
    cfg = info["cfg"]
    B = y["is_B"].to_numpy(bool)
    mk = info["mkt"].copy()
    mk["S"] = S_full.reindex(mk.index)
    mk["year"] = mk.index.get_level_values("time_id").str[:4].astype(int)
    b = y[B]
    key = [b["mca_code"], b["time_id"]]
    SBm = b["share_B_cond"].groupby(key).sum().reindex(mk.index)
    phin = b["phi_mt"].groupby(key).nunique().reindex(mk.index)
    out = {"cfg": cfg, "rows": int(len(y)), "B_rows": int(B.sum()), "D_rows": int((~B).sum()),
           "B_market_quarters": int(len(mk)),
           "r_t": {k: float(v) for k, v in info["r_t"].items()},
           "phi_varies_within_market_quarters": int((phin > 1).sum())}
    for k in ("r_local", "r_national"):
        if k in info:
            out[k] = info[k]
    bind = mk["bind"].astype(bool)
    out["B_bound_share"] = float(bind.mean())
    out["B_bound_n"] = int(bind.sum())
    out["B_bound_share_S_weighted"] = float(mk.loc[bind, "S"].sum() / mk["S"].sum())
    out["B_bound_share_pop_weighted"] = float(mk.loc[bind, "pop"].sum() / mk["pop"].sum())
    out["B_bound_share_by_year"] = {int(yy): float(g["bind"].astype(bool).mean()) for yy, g in mk.groupby("year")}
    kb = mk.loc[bind, "k"]
    out["k_B_bound"] = {"unweighted": qtab(kb), "S_weighted": qtab(kb, mk.loc[bind, "S"]),
                        "active_weighted": qtab(kb, mk.loc[bind, "dep"])}
    out["inside_sum_B"] = qtab(SBm)
    out["inside_sum_B_bound_max_abs_dev_from_c"] = (float((SBm[bind] - cfg["c"]).abs().max())
                                                    if cfg["mode"] == "rescale" and bind.any() else None)
    out["inside_sum_B_max"] = float(SBm.max())
    bs = mk["S"] / mk["M"]
    out["B_stock_over_M"] = {"median": float(bs.median()), "S_weighted_median": alt.wmedian(bs, mk["S"]),
                             "aggregate": float(mk["S"].sum() / mk["M"].sum())}
    out["X_over_S"] = qtab(mk["X"] / mk["S"])
    dm = info["d_mkt"]
    out["D_market"] = {"quarters_bound": int(dm["bind"].sum()), "quarters": int(len(dm)),
                       "bound_list": [str(t) for t in dm.index[dm["bind"].to_numpy(bool)]],
                       "k_D_bound": qtab(dm.loc[dm["bind"].to_numpy(bool), "k"]),
                       "X_D_over_M_nat_anchor": qtab(dm["X_D"] / dm["M_nat_anchor"], qs=(0.0, 0.5, 1.0)),
                       "M_nat": {str(t): float(v) for t, v in dm["M_nat"].items()}}
    g, G, K, SB = dcs.closed_form(y)
    SD = y.loc[~B, "share_D"].groupby(y.loc[~B, "time_id"]).sum()
    out.update(G_min=float(G.min()), G_argmin=str(G.idxmin()), G_nonpositive_quarters=int((G <= 0).sum()),
               G_nonpositive_list=[str(t) for t in G.index[G <= 0]], K_min=float(K.min()), K_median=float(K.median()),
               S_D={"min": float(SD.min()), "median": float(SD.median()), "max": float(SD.max())},
               outside_share_median=float(pd.Series(np.asarray(g)).median()),
               G_t={str(t): float(v) for t, v in G.items()}, K_t={str(t): float(v) for t, v in K.items()},
               S_D_t={str(t): float(v) for t, v in SD.items()})
    out["mean_rho_1ms"] = dcs.mean_rho_1ms(y)
    sB = b["share_B_cond"]
    out["rows_share_above_0p5"] = float((sB > 0.5).mean())
    return out


def compare_cols(a, b, cols, B=None):
    """Max abs / rel difference of a and b on cols (same row order), NaN = NaN."""
    out = {}
    for c in cols:
        x, z = a[c].to_numpy(float), b[c].to_numpy(float)
        if B is not None:
            x, z = x[B], z[B]
        same = np.array_equal(x, z, equal_nan=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(x / z - 1)
            ab = np.abs(x - z)
        rel = rel[np.isfinite(rel)]
        ab = ab[np.isfinite(ab)]
        nan_mismatch = int((np.isnan(x) != np.isnan(z)).sum())
        out[c] = {"exact": bool(same), "max_abs": float(ab.max()) if len(ab) else 0.0,
                  "max_rel": float(rel.max()) if len(rel) else 0.0,
                  "median_rel": float(np.median(rel)) if len(rel) else 0.0,
                  "rows_rel_gt_1e-12": int((rel > 1e-12).sum()), "nan_mismatch": nan_mismatch}
    return out


def xs_bins(d, S_full, E):
    """The 19:55 table: X / S (X at the phi caps 0.99 and 0.999) by bins of the market's uncapped phi-hat."""
    b = d[d["is_B"].to_numpy(bool)]
    key = [b["mca_code"], b["time_id"]]
    mk = pd.DataFrame({"phi": b["phi_mt"].groupby(key).first(), "phi_max": b["phi_mt"].groupby(key).max(),
                       "phi_min": b["phi_mt"].groupby(key).min()})
    mk.index.names = ["mca_code", "time_id"]
    for c in (0.99, 0.999):
        mk[f"X{c}"] = (b["Dep_Act"] / (1 - b["phi_mt"].clip(0, c))).groupby(key).sum()
    mk["S"] = S_full.reindex(mk.index)
    rows = []
    for (lo, hi), lab in zip(PHI_BINS, BIN_LABELS):
        sel = (mk["phi"] >= lo) & (mk["phi"] < hi)
        s = mk[sel]
        for c in (0.99, 0.999):
            r = s[f"X{c}"] / s["S"]
            q = qtab(r)
            rows.append({"routine": f"E{E}", "phi_cap": c, "phi_bin": lab, "n_market_quarters": int(len(s)),
                         "share_of_market_quarters": float(sel.mean()),
                         "share_of_B_stock": float(s["S"].sum() / mk["S"].sum()),
                         "p10": q.get("p10"), "p25": q.get("p25"), "p50": q.get("p50"), "p75": q.get("p75"),
                         "p90": q.get("p90"), "S_weighted_median": alt.wmedian(r, s["S"]) if len(s) else np.nan,
                         "aggregate_X_over_S": float(s[f"X{c}"].sum() / s["S"].sum()) if len(s) else np.nan,
                         "share_above_1": float((r > 1).mean()) if len(s) else np.nan})
    return pd.DataFrame(rows), int((mk["phi_max"] > mk["phi_min"]).sum())


# ---------------------------------------------------------------------------------- the frames step
def step_frames(routines):
    P = dcs.prep_module()
    fx_year = alt.findex_by_year()
    SANDBOX.mkdir(parents=True, exist_ok=True)
    S_full = None
    for E in routines:
        t0 = time.time()
        src = SRC_DP / FRAMES[E]
        tab = pq.read_table(src)
        schema0 = tab.schema
        d = tab.to_pandas()
        del tab
        d["is_B"] = d["is_B"].astype(bool)
        pmd5 = schema0.metadata[demand_chars.PANEL_MD5_KEY].decode()
        if S_full is None:
            S_full, prec = alt.panel_stock(pmd5)
            stamp(f"panel stock {prec}")
        res = {"source": {"path": str(src), "md5": md5_file(src), "rows": int(len(d)), "panel_md5": pmd5,
                          "prep_lf_md5": md5_file(REPO / "sleep_demand_prep_link.py", lf=True)},
               "gates": {}, "variants": {}}
        Bmask = d["is_B"].to_numpy(bool)
        # gate P: production's function on the source rows against the source frame
        yP = prod_function(P, d)
        res["gates"]["P_prod_function_vs_source_frame"] = {
            "B": compare_cols(yP, d, ["M_mt", "share_B_cond"], Bmask),
            "D": compare_cols(yP, d, ["M_nat", "share_D"], ~Bmask),
            "all_columns_exact": alt.compare(yP, d)["exact"]}
        # implied anchor of the source frame: M fx acc / Pop on the B markets where the floor does not bind
        bb = d[Bmask]
        rec_src = (bb["M_mt"] * 0.77 * bb["_access"] / bb["pop_total"])
        rec_new = (yP.loc[Bmask, "M_mt"] * 0.77 * yP.loc[Bmask, "_access"] / yP.loc[Bmask, "pop_total"])
        res["gates"]["P_prod_function_vs_source_frame"]["implied_r_median"] = {
            "source": float(rec_src.median()), "reshare": float(rec_new.median())}
        # gate G0: my build at production's settings equals production's function exactly
        yG, infoP = build(P, d, cfg_of("prod"), fx_year, S_full)
        res["gates"]["G0_build_equals_prod_function"] = alt.compare(yG, yP)
        res["variants"]["prod"] = diag(yP, infoP, S_full)
        res["variants"]["prod"]["r_hat"] = float(infoP["r_t"].iloc[0])
        stamp(f"E{E} gate P: {json.dumps({k: (v['exact'], v['max_rel']) for k, v in {**res['gates']['P_prod_function_vs_source_frame']['B'], **res['gates']['P_prod_function_vs_source_frame']['D']}.items()})}; "
              f"G0 exact {res['gates']['G0_build_equals_prod_function']['exact']}; r-hat {infoP['r_t'].iloc[0]:,.1f} "
              f"(local {infoP['r_local']:,.1f} / national {infoP['r_national']:,.1f})")
        del yG
        # gate P, decomposed: the same build with r-hat fixed at the source frame's (the national ratio there
        # still counts the rows finalize dropped), so what is left is the B floor and the D market's market set
        r_src = float(rec_src.median())
        yR, _ = build(P, d, {**cfg_of("prod"), "r_fixed": r_src}, fx_year, S_full)
        res["gates"]["P_with_source_r_hat"] = {"r_src": r_src,
            "B": compare_cols(yR, d, ["M_mt", "share_B_cond"], Bmask),
            "D": compare_cols(yR, d, ["M_nat", "share_D"], ~Bmask)}
        _bk = d.loc[Bmask, ["mca_code", "time_id"]].assign(
            _rel=np.abs(yR.loc[Bmask, "M_mt"].to_numpy() / d.loc[Bmask, "M_mt"].to_numpy() - 1))
        _mq = _bk.groupby(["mca_code", "time_id"])["_rel"].max()
        res["gates"]["P_with_source_r_hat"]["B_market_quarters_rel_gt_1e-9"] = int((_mq > 1e-9).sum())
        del yR, _bk
        alt.write_frame(yP, schema0, SANDBOX / "prod" / f"E{E}" / "demand_prep" / FRAMES[E],
                        {"variant": "prod", "source_md5": res["source"]["md5"]})
        del yP
        mtab = {"prod_source_frame": bb.groupby(["mca_code", "time_id"])["M_mt"].first()}
        # A1q95f: gate A against msize_1112's A1q95 frames (B side; D market old over new)
        yA, infoA = build(P, d, cfg_of("A1q95f"), fx_year, S_full)
        res["variants"]["A1q95f"] = diag(yA, infoA, S_full)
        mtab["A1q95f"] = infoA["mkt"]["M"]
        altf = ALT_A1 / FRAMES[E]
        if altf.exists():
            o = pq.read_table(altf, columns=["entity_id", "time_id", "is_B", "M_mt", "share_B_cond", "M_nat",
                                             "share_D"]).to_pandas()
            ob = o[o["is_B"].astype(bool)].set_index(["entity_id", "time_id"])
            nb = yA[Bmask].set_index(["entity_id", "time_id"])
            j = nb[["M_mt", "share_B_cond"]].join(ob[["M_mt", "share_B_cond"]], rsuffix="_old", how="outer")
            gA = {"B_rows_new": int(len(nb)), "B_rows_old": int(len(ob)),
                  "B_rows_unmatched": int(j[["M_mt", "M_mt_old"]].isna().any(axis=1).sum())}
            for c in ("M_mt", "share_B_cond"):
                with np.errstate(invalid="ignore", divide="ignore"):
                    rel = (j[c] / j[f"{c}_old"] - 1).abs()
                gA[c] = {"max_rel": float(rel.max()), "exact_rows": int((j[c] == j[f"{c}_old"]).sum())}
            mo = o[~o["is_B"].astype(bool)].groupby("time_id")["M_nat"].first()
            mn = infoA["d_mkt"]["M_nat"]
            ratio = (mo / mn.reindex(mo.index))
            gA["M_nat_old_over_new"] = qtab(ratio, qs=(0.0, 0.5, 1.0))
            gA["D_floor_quarters_new"] = int(infoA["d_mkt"]["bind"].sum())
            ma = infoA["d_mkt"]["M_nat_anchor"]
            gA["M_nat_old_over_new_anchor_part_quarters_without_new_D_floor"] = qtab(
                ratio[~infoA["d_mkt"]["bind"].reindex(mo.index).to_numpy(bool)], qs=(0.0, 0.5, 1.0))
            gA["note"] = ("old = msize_1112 A1q95 (old D rule, pre-phi_t-fix D rows); new = this build on the "
                          "both-fixes rows; the B side must agree, the D market not")
            res["gates"]["A_A1q95_B_side_vs_msize_1112"] = gA
            stamp(f"E{E} gate A: B unmatched {gA['B_rows_unmatched']}, M_mt max rel {gA['M_mt']['max_rel']:.2e}, "
                  f"share max rel {gA['share_B_cond']['max_rel']:.2e}; M_nat old/new {gA['M_nat_old_over_new']}")
            del o, ob, nb, j
        del yA
        # variant (a) and (a')
        for v, c in {**A_VARIANTS, **AK_VARIANTS}.items():
            t = time.time()
            y, info = build(P, d, cfg_of(v), fx_year, S_full)
            r = diag(y, info, S_full)
            res["variants"][v] = r
            if v == "a080":
                mtab["a_noflo"] = info["mkt"]["M"]
            alt.write_frame(y, schema0, SANDBOX / v / f"E{E}" / "demand_prep" / FRAMES[E],
                            {"variant": v, "c": c, "phi_cap": 0.999, "anchor": "stock_q95_by_quarter",
                             "findex": "year", "source_md5": res["source"]["md5"]})
            stamp(f"E{E} {v}: B bound {r['B_bound_share']:.4f} (S-w {r['B_bound_share_S_weighted']:.4f}); "
                  f"k p50 {r['k_B_bound']['unweighted'].get('p50', float('nan')):.4f}; D bound "
                  f"{r['D_market']['quarters_bound']} q; G_min {r['G_min']:.4f} ({r['G_nonpositive_quarters']} q <= 0); "
                  f"{time.time() - t:.0f} s")
            del y
        pd.DataFrame(mtab).reset_index().to_parquet(SANDBOX / f"mkt_E{E}.parquet")
        xs, nvar = xs_bins(d, S_full, E)
        xs.to_csv(SANDBOX / f"xs_bins_E{E}.csv", index=False, lineterminator="\n")
        res["xs_bins_phi_varies_within_market_quarters"] = nvar
        vj = json.loads(SRC_VERIFY[E].read_text(encoding="utf-8"))
        res["truncation_11B"] = vj["summaries"]["both"]["Active_truncated"]
        res["finalize_drops"] = vj["summaries"]["both"].get("demand_chars")
        res["seconds"] = round(time.time() - t0, 1)
        res["peak_working_set_GB"] = alt.peak_gb()
        jdump(res, SANDBOX / f"diag_E{E}.json")
        stamp(f"E{E} frames done in {time.time() - t0:.0f} s")
        del d


# ------------------------------------------------------------------------------------ the logit
def heavy_others():
    """python/julia processes other than this one, its parent and the language server, above 1 GB."""
    r = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True)
    me = {os.getpid(), os.getppid(), LSP_PID}
    out = []
    for row in csv.reader(io.StringIO(r.stdout)):
        if len(row) < 5:
            continue
        name, pid = row[0], int(row[1])
        if not any(s in name.lower() for s in ("python", "julia")) or pid in me:
            continue
        kb = int(row[4].replace(",", "").replace(".", "").replace("K", "").strip() or 0)
        if kb > HEAVY_KB:
            out.append((name, pid, kb))
    return out


def wait_light(max_wait_s=1800):
    t = time.time()
    while True:
        h = heavy_others()
        if not h:
            return
        if time.time() - t > max_wait_s:
            raise SystemExit(f"another heavy process for {max_wait_s} s: {h}")
        stamp(f"waiting: {h}")
        time.sleep(20)


def step_logit(variants, routines, threads):
    jl = shutil.which("julia")
    for E in routines:
        dj = json.loads((SANDBOX / f"diag_E{E}.json").read_text(encoding="utf-8"))
        for v in variants:
            base = SANDBOX / v / f"E{E}"
            dp, od = base / "demand_prep", base / "out"
            fr = dp / FRAMES[E]
            if not fr.exists():
                stamp(f"logit {v} E{E}: no frame")
                continue
            summ = od / "logit" / "logit_summary_spec_12.json"
            if summ.exists() and summ.stat().st_mtime > fr.stat().st_mtime:
                stamp(f"logit {v} E{E}: summary newer than its frame, skipped")
                continue
            nq = dj["variants"][v]["G_nonpositive_quarters"]
            if nq:
                stamp(f"logit {v} E{E}: G_t <= 0 in {nq} quarter(s), not run")
                continue
            wait_light()
            t = time.time()
            with open(base / "logit.log", "w", encoding="utf-8") as fh:
                r = subprocess.run([jl, "--project=.", f"--threads={threads}", "blp_logit.jl", "--demand-dir",
                                    str(dp), "--out-dir", str(od)], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                                   env=dcs.julia_env())
            stamp(f"logit {v} E{E}: exit {r.returncode} in {time.time() - t:.0f} s")


def logit_read(path, E):
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    r = s.get(f"E{E}_{SUBMODEL}")
    if r is None:
        return None
    names = r["param_names"]
    ia, idd = names.index("alpha"), names.index("dummy_D_type")
    a, se, gs = float(r["theta1"][ia]), float(r["se"][ia]), float(r["G_star"])
    tq = float(stats.t.ppf(0.975, gs)) if gs > 1 else float(stats.norm.ppf(0.975))
    return {"alpha": a, "se": se, "p": float(r["pval"][ia]), "G_star": gs, "ci95": [a - tq * se, a + tq * se],
            "t_crit": tq, "dummy_D": float(r["theta1"][idd]), "dummy_D_se": float(r["se"][idd]),
            "n_obs": int(r["n_obs"]), "se_method": r.get("se_method")}


# ---------------------------------------------------------------------------------------- W4
def w4_block(w, M, top_mt):
    mt, dest = w.mt, w.dest
    okM = np.isfinite(M[mt]) & (M[mt] > 0)
    dd = dest & okM
    bo = w.Wst / M[mt]
    brv = (w.Wtot[mt] - w.Wst) / M[mt]
    bn = 1.0 - w.Wtot[mt] / M[mt]
    _, om, rho, nu = w.lnW(M, **alt.W41_CASE)
    wt = np.maximum(w.Wst, 1e-12)
    out = {"matched_share": float(np.isfinite(M).mean()), "dest_cells": int(dd.sum()),
           "share_dest_new_pool_negative": float((bn[dd] < 0).mean())}
    for lab, sel in (("all", dd), ("top_decile_markets", dd & top_mt[mt])):
        blk = {"cells": int(sel.sum())}
        for nm, v in (("pool_own", bo), ("pool_rivals", brv), ("pool_new", bn), ("W_own", om), ("W_rivals", rho),
                      ("W_new", nu)):
            blk[nm] = {"unweighted": qtab(v[sel]), "own_stock_weighted": qtab(v[sel], wt[sel])}
        out[lab] = blk
    return out


def step_w4(ctx_path):
    t = time.time()
    wait_light()
    w = alt.W4(ctx_path)
    stamp(f"w4: context loaded ({w.n:,} units, {w.nmt:,} market-quarters) in {time.time() - t:.0f} s")
    tq = np.asarray(w.keys.get_level_values(1))
    Wt = pd.Series(w.Wtot)
    p90 = Wt.groupby(tq).transform(lambda s: s.quantile(0.90)).to_numpy()
    top_mt = w.Wtot >= p90
    mk = pd.read_parquet(SANDBOX / "mkt_E3.parquet").set_index(["mca_code", "time_id"])
    res = {"inputs": w.inputs, "case": alt.W41_CASE, "top_decile_rule": "market-quarters with total stock W at or "
           "above the quarter's 90th percentile (W4's MT, all quarters)", "weights": "own stock W (W4.1's)",
           "blocks": {}}
    # gate W: W4.1's production numbers from the context's own M_hat
    m0 = w.metrics(w.M_hat)
    ref = json.loads(alt.W41_JSON.read_text(encoding="utf-8"))
    res["gate_W"] = {"b_new_dest_p50": [m0["b_new_dest"]["p50"], ref["pools"]["base_cap099"]["b_new"]["unweighted"]["p50"]],
                     "new_share_W_p50": [m0["W_composition_availability_0p5"]["new"]["p50"],
                                         ref["shifters"]["base_cap099"]["cost_only"]["composition_new"]["p50"]]}
    for lab, col in (("production_both_fixes", "prod_source_frame"), ("A1q95_floor095_cap099", "A1q95f"),
                     ("a_anchor_no_floor", "a_noflo")):
        M = mk[col].reindex(w.keys).to_numpy(float)
        res["blocks"][lab] = w4_block(w, M, top_mt)
        b = res["blocks"][lab]["all"]
        stamp(f"w4 {lab}: pool_new p50 {b['pool_new']['unweighted']['p50']:.4f} (w {b['pool_new']['own_stock_weighted']['p50']:.4f}); "
              f"W_new p50 {b['W_new']['unweighted']['p50']:.4f} (w {b['W_new']['own_stock_weighted']['p50']:.4f})")
    res["seconds"] = round(time.time() - t, 1)
    jdump(res, SANDBOX / "w4.json")


# ------------------------------------------------------------------------------------ the summary
def step_summary(routines, out):
    out.mkdir(parents=True, exist_ok=True)
    ref = {E: logit_read(SRC_LOGIT, E) for E in routines}
    rows, js = [], {"generated": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "reference_logit": ref, "routines": {}}
    for E in routines:
        dj = json.loads((SANDBOX / f"diag_E{E}.json").read_text(encoding="utf-8"))
        rec = {"gates": dj["gates"], "truncation_11B": dj["truncation_11B"], "finalize_drops": dj["finalize_drops"],
               "variants": {}}
        for v in ("prod", "A1q95f") + tuple(A_VARIANTS) + tuple(AK_VARIANTS):
            r = dj["variants"][v]
            summ = SANDBOX / v / f"E{E}" / "out" / "logit" / "logit_summary_spec_12.json"
            lr = logit_read(summ, E) if summ.exists() else None
            if lr is not None:
                lr["elasticity"] = lr["alpha"] * r["mean_rho_1ms"]
            rec["variants"][v] = {"diag": r, "logit": lr}
            kb = r["k_B_bound"]
            rows.append({"routine": f"E{E}", "variant": v, "c": A_VARIANTS.get(v, AK_VARIANTS.get(v)),
                         "B_bound_share": r["B_bound_share"], "B_bound_share_S_weighted": r["B_bound_share_S_weighted"],
                         "D_quarters_bound": r["D_market"]["quarters_bound"],
                         **{f"k_{q}": kb["unweighted"].get(q) for q in ("p10", "p25", "p50", "p75", "p90", "min")},
                         **{f"kw_{q}": kb["S_weighted"].get(q) for q in ("p10", "p25", "p50", "p75", "p90", "min")},
                         "G_min": r["G_min"], "G_nonpositive_quarters": r["G_nonpositive_quarters"],
                         "K_min": r["K_min"], "S_D_max": r["S_D"]["max"],
                         "alpha": lr["alpha"] if lr else None, "se_wcb": lr["se"] if lr else None,
                         "ci95_lo": lr["ci95"][0] if lr else None, "ci95_hi": lr["ci95"][1] if lr else None,
                         "p": lr["p"] if lr else None, "elasticity": lr["elasticity"] if lr else None,
                         "dummy_D": lr["dummy_D"] if lr else None, "dummy_D_se": lr["dummy_D_se"] if lr else None,
                         "B_stock_over_M_median": r["B_stock_over_M"]["median"],
                         "inside_sum_B_median": r["inside_sum_B"].get("p50")})
        lp = rec["variants"]["prod"]["logit"]
        if lp and ref[E]:
            rec["gate_L"] = {"d_alpha": lp["alpha"] - ref[E]["alpha"], "bound": 0.05 * ref[E]["se"],
                             "pass": abs(lp["alpha"] - ref[E]["alpha"]) <= 0.05 * ref[E]["se"],
                             "d_dummy_D": lp["dummy_D"] - ref[E]["dummy_D"]}
        js["routines"][f"E{E}"] = rec
    if (SANDBOX / "w4.json").exists():
        js["w4"] = json.loads((SANDBOX / "w4.json").read_text(encoding="utf-8"))
    pd.DataFrame(rows).to_csv(out / "market_size_a_summary.csv", index=False, lineterminator="\n")
    jdump(js, out / "market_size_a_summary.json")
    xs = [pd.read_csv(SANDBOX / f"xs_bins_E{E}.csv") for E in routines if (SANDBOX / f"xs_bins_E{E}.csv").exists()]
    if xs:
        pd.concat(xs).to_csv(out / "market_size_a_xs_bins.csv", index=False, lineterminator="\n")
    lines = []
    for p in sorted(out.glob("*")):
        if p.name != "MD5SUMS.txt" and p.is_file():
            lines.append(f"{md5_file(p, lf=True)}  {p.name}")
    (out / "MD5SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    stamp(f"summary -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    ap.add_argument("--routines", default="3")
    ap.add_argument("--variants", default=",".join(LOGIT_VARIANTS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", default=str(alt.CTX_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    a = ap.parse_args()
    routines = [int(x) for x in a.routines.split(",")]
    for s in a.step.split(","):
        t = time.time()
        if s == "frames":
            step_frames(routines)
        elif s == "logit":
            step_logit(a.variants.split(","), routines, a.threads)
        elif s == "w4":
            step_w4(Path(a.ctx))
        elif s == "summary":
            step_summary(routines, Path(a.out_dir))
        else:
            raise SystemExit(f"step {s}")
        stamp(f"step {s} done in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
