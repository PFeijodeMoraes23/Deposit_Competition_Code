"""
diag_market_size_alt.py
=======================
Two alternative anchors for the market size M_mt of the demand frames, screened on the stored I2 frames
(IMPR handoff of 2026-11-12, about 17:30; Pedro: "try both"). Same approach as diag_cap_sensitivity.py
Stage 1: the stored frames of routines (III) and (IV), spec 12, re-shared in sandboxes outside OneDrive, and
the logit run locally on each. No production file is written. sleep_demand_prep_link is imported; its
build_market_size_and_shares makes the base frames and the gates on copies, with PHI_CAP and TOTAL_SHARE_CAP
set as module attributes and restored. The alternative anchors come from build_ms below, which is that
function with the anchor generalised, line for line where the configuration is today's.

The Findex fix is in every variant except `base`: the Global Findex banked fraction by year
(findex_banked_frac of bcb_banked_mca_panel.csv, one national value per year, 0.694 in 2016 to 0.864 in
2024) on every row, B and D, in place of FINDEX_FALLBACK = 0.77. The production frames carry 0.77 on every
row because process_specification's keep_cols drops the column.

Notation (market m = mca_code, quarter t; a row j is bank x market x deposit type):
  phi_j   phi_mt on B rows, phi_t on D rows; c the share-side cap (PHI_CAP), c_a the anchor-side cap
  acc_m   production's access index: GDP per capita (centred in the frame, the persisted mean added back)
          over its median in the year, clipped to [0.35, 2.5]
  fx_t    the Findex fraction of the year (0.77 in base)
  X_m     sum over the B rows of Dep_Act_j / (1 - min(phi_j, c))      (the floor's and the shares' numerator)
  S_m     the booked B stock: the sum over the panel's B cells of dep_a1 + dep_a2 + dep_a4 + dep_a5 at t
          (market_panel.parquet, the frames' panel, pyarrow column subset; md5 checked against the frames'
          metadata). The stored rows hold only part of it (the (14-B) cut drops every cell whose balance
          fell by more than phi R), and that part moves with the season, so it would carry the cut into
          the anchor; their sum is reported as S_stored.
  M_m     max{ r_t Pop_m / (fx_t acc_m) , X_m / TOTAL_SHARE_CAP }        (production's form)
  M_nat,t max{ r_t sum_m Pop_m / (fx_t acc_m) , sum_D Dep_Act / (1 - min(phi_t, c)) / TOTAL_SHARE_CAP }
          (production's form; the sum runs over every market of the frame, the D rows' NATIONAL one too)
  s_j     Dep_Act_j / ((1 - min(phi_j, c)) M)                              (production's form)
The anchor r_t:
  dens  (today; anchor 2) the density sum_B Dep_Act / ((1 - min(phi_mt, c_a)) Pop) acc
        pooled (today): r = max(Q_q over every market-quarter of the frame, Q_q over the quarters of the
                        national ratio sum_all Dep_Act / (1 - min(phi, c_a)) / sum Pop), one r for all t
        by quarter:     r_t = max(Q_q over the B markets of quarter t, the national ratio of t)
  stock (anchor 1)      r_t = Q_q over the B markets of quarter t of (S_m / Pop_m) acc_m
        Where the floor does not bind, S_m / M_m = fx_t (S_m / Pop_m) acc_m / r_t: the market at the
        q-quantile holds a booked B stock of exactly fx_t of its market (the banked fraction), a market at
        half that density half of it. That is the target read in the stationary state: with no net flow
        (deposit = R x lagged deposit) Dep_Act / (1 - phi) = deposit, so the active inside share equals
        the stock share. Out of it the active inside share is (X_m / S_m) (S_m / M_m), the amplification
        X / S times the stock share, and the floor binds where that product exceeds TOTAL_SHARE_CAP.

Variants (logit = the local logit is run on them):
  base          production's function, production's constants (fx 0.77)       logit; gate L
  F             production's function, fx by year (today's anchor)            logit
  A1q{95,75,50} stock, by quarter, q 0.95 / 0.75 / 0.50, c 0.99, cap 0.95       logit
  A1q95c999     stock, by quarter, q 0.95, c 0.999, cap 0.95                   logit
  ...s80        the same four with TOTAL_SHARE_CAP 0.80                        logit
  A2pool        dens, c_a 0.95, pooled, c 0.99, cap 0.95                       logit
  A2qtr         dens, c_a 0.95, by quarter, c 0.99, cap 0.95                   logit
  A2pools80 / A2qtrs80  the same with TOTAL_SHARE_CAP 0.80                     logit
  X_A1q95c95    extra (not in the handoff): A1q95 with c 0.95 on the share side   logit
  X_A2all95     extra (not in the handoff): dens with c_a = c = 0.95, pooled      logit
  A logit is not run where the closed form has G_t <= 0 in some quarter of both routines (step logit).
Gates (frames step, in memory):
  G1  build_ms at base's configuration equals production's function on the same rows, every column exactly
  G2  build_ms at F's configuration equals production's function with fx by year, exactly
  G3  anchor 2 at the anchor cap 0.99, pooled (build_ms, c_a = 0.99 through the anchor's own (1 - phi)
      column) equals the base exactly: production's function with the same fx, both with fx 0.77 and by year
  L   (summary) the base re-share's logit alpha within 0.05 se of production's, E3 and E4
  W   (w4 step) the W4.1 numbers reproduce: b_empty and new money's share of W at the median destination cell
      for production's M and for the cap-0.999 frame of diag_cap_sensitivity (capsens_1111/p4)

Steps (--step, comma list):
  frames   each variant's two frames to <sandbox>/<v>/demand_prep/ under the stored names, with the stored
           schema and panel md5 (written as .part and renamed); frame diagnostics to <sandbox>/<v>/frame_diag.json,
           the stored frames' own to <sandbox>/production_frame_diag.json, the gates to <sandbox>/gates.json,
           and <sandbox>/FRAMES_COMPLETE.json last (every frame's md5).
  logit    julia --project=. --threads=N blp_logit.jl --demand-dir <sandbox>/<v>/demand_prep --out-dir
           <sandbox>/<v>/out, one variant at a time, log <sandbox>/<v>/logit.log, BLP_* / DEMAND_* /
           SLEEP_ACTIVE_ESTS removed from its environment. Skipped where the summary is newer than the frames,
           and where the closed form has G_t <= 0 in some quarter of both routines (the logit cannot invert).
  w4       W4.1's pools and composition (scratchpad w4_feasibility.py, the "cost_only" case: availability
           0.5 at the mean shifters, gamma1 0.17, gamma_P 1, gamma_P0 0, kappa_c 1) for every variant's E3 M,
           from the a216 K2 context -> <sandbox>/w4.json (gate W first).
  frontier market level only (B markets, no D market, no logit): the by-quarter anchors (stock; density
           with c_a 0.95), each with production's access index and without it (acc = 1 in the anchor and in
           M), over q in 0.99 ... 0.25, c in {0.99, 0.95}, TOTAL_SHARE_CAP in {0.95, 0.80}: floor incidence
           and B stock over M -> <sandbox>/frontier.csv; and the ceiling the floor puts on any anchor,
           S / M <= TOTAL_SHARE_CAP S / X -> <sandbox>/frontier_ceiling.csv.
  cellgroups  Part A's cell-group table (diag_netflow_gap.a1_run's type-4 cells and splits: on the menu with
           phi-hat <= cap, on the menu with phi-hat > cap, off the menu = cut by (14-B)) at each variant's
           cap, with the data's median |NF| / carry; per variant the floor incidence by group and by phi-hat
           bin; the K2 world rebuilt on the frames of --world-variants (production first: gate CG, Part A's
           numbers exactly) for the W0 and W3 medians; and W1.1's effective F of the type-4 first stage on
           production, base and A1q95 with the md5 of the first stage's inputs in every frame ->
           <sandbox>/cellgroups.json, <sandbox>/cellgroups.csv (see step_cellgroups).
  summary  --out-dir (default Rout/diag_market_size_alt/): market_size_alt_summary.csv (one row per variant
           and routine), market_size_alt_summary.json (the rows, gates, frame diagnostics, W4 numbers,
           frontier, cell groups, manifest), the frontier and cell-group CSVs, MD5SUMS.txt (LF md5 of each
           output).

Usage:  python diagnostics/diag_market_size_alt.py --step frames [--variants a,b] [--routines 3,4]
        python diagnostics/diag_market_size_alt.py --step logit [--threads 4]
        python diagnostics/diag_market_size_alt.py --step w4
        python diagnostics/diag_market_size_alt.py --step frontier
        python diagnostics/diag_market_size_alt.py --step cellgroups [--world-variants production,A1q95,...]
        python diagnostics/diag_market_size_alt.py --step summary
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import json
import pickle
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
from utils import state_transform as _st
import diag_cap_sensitivity as dcs

REPO = Path(__file__).resolve().parents[1]
SANDBOX = Path(r"C:\Users\pedro\AppData\Local\Temp\msize_1112")
OUT_DEFAULT = paths.rout_dir() / "diag_market_size_alt"
DP = paths.demand_parquet_dir()
FRAMES = dcs.FRAMES
PROD_LOGIT = dcs.PROD_LOGIT
BANKED_CSV = dcs.BANKED_CSV
PANEL = paths.PROCESSED / "market_panel.parquet"
SCRATCH = Path(r"C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad")
CTX_DEFAULT = SCRATCH / "kappa_1109" / "k2n_ctx_fa38794d_a216.pkl"
W41_JSON = SCRATCH / "w4_1112" / "w4_feasibility.json"
CAPSENS_P4_E3 = Path(r"C:\Users\pedro\AppData\Local\Temp\capsens_1111\p4\demand_prep") / FRAMES[3]
ROUT = paths.rout_dir()
PROD = dcs.PROD
DEP_COLS = ["dep_a1", "dep_a2", "dep_a4", "dep_a5"]
PHI_ORD = 0.98            # "ordinary" markets: phi-hat below this
B_SHARE_GOAL, FLOOR_GOAL = 0.20, 0.02   # the verdict's conditions (the handoff's examples)
EXTREME = 0.5             # a row's inside share above this is "extreme"


def _v(findex="year", anchor="dens", anchor_cap=0.99, pool="pooled", q=0.95, phi_cap=0.99, share_cap=0.95,
       builder="alt", logit=True, kind="requested"):
    return dict(findex=findex, anchor=anchor, anchor_cap=anchor_cap, pool=pool, q=q, phi_cap=phi_cap,
                share_cap=share_cap, builder=builder, logit=logit, kind=kind)


VARIANTS = {
    "base": _v(findex="const", builder="prod"),
    "F": _v(builder="prod"),
    "A1q95": _v(anchor="stock", anchor_cap=None, pool="quarter", q=0.95),
    "A1q75": _v(anchor="stock", anchor_cap=None, pool="quarter", q=0.75),
    "A1q50": _v(anchor="stock", anchor_cap=None, pool="quarter", q=0.50),
    "A1q95c999": _v(anchor="stock", anchor_cap=None, pool="quarter", q=0.95, phi_cap=0.999),
    "A1q95s80": _v(anchor="stock", anchor_cap=None, pool="quarter", q=0.95, share_cap=0.80),
    "A1q75s80": _v(anchor="stock", anchor_cap=None, pool="quarter", q=0.75, share_cap=0.80),
    "A1q50s80": _v(anchor="stock", anchor_cap=None, pool="quarter", q=0.50, share_cap=0.80),
    "A1q95c999s80": _v(anchor="stock", anchor_cap=None, pool="quarter", q=0.95, phi_cap=0.999, share_cap=0.80),
    "A2pool": _v(anchor_cap=0.95, pool="pooled"),
    "A2qtr": _v(anchor_cap=0.95, pool="quarter"),
    "A2pools80": _v(anchor_cap=0.95, pool="pooled", share_cap=0.80),
    "A2qtrs80": _v(anchor_cap=0.95, pool="quarter", share_cap=0.80),
    "X_A1q95c95": _v(anchor="stock", anchor_cap=None, pool="quarter", q=0.95, phi_cap=0.95, kind="extra"),
    "X_A2all95": _v(anchor_cap=0.95, pool="pooled", phi_cap=0.95, kind="extra"),
}
# Gate configurations (build_ms against production's function); never written.
GATE_CFG = {
    "G1_base": (_v(findex="const"), "base"),
    "G2_F": (_v(), "F"),
    "G3_A2_c099_pooled_fx_const": (_v(findex="const", anchor_cap=0.99, pool="pooled"), "base"),
    "G3_A2_c099_pooled_fx_year": (_v(anchor_cap=0.99, pool="pooled"), "F"),
}
QUARTERS_SHOWN = ["2016Q1", "2020Q1", "2020Q2", "2020Q3", "2020Q4", "2024Q4"]

stamp = dcs.stamp
md5_file = dcs.md5_file
jdump = dcs.jdump


def peak_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().peak_wset / 2 ** 30, 2)
    except Exception:
        return None


# ------------------------------------------------------------------------------------ the inputs
def findex_by_year():
    bk = dcs.findex_series()
    g = bk.groupby("year")["findex_banked_frac"]
    if (g.nunique() > 1).any():
        raise SystemExit(f"{BANKED_CSV.name}: findex_banked_frac varies across markets within a year")
    return {int(y): float(v) for y, v in g.first().items()}


def panel_stock(frame_panel_md5):
    """The booked B stock S_mt: sum over the panel's B cells (conglomerate x market x quarter, the prep's
    de-duplication) of dep_a1 + dep_a2 + dep_a4 + dep_a5, 2016-2024. The panel must be the frames' own."""
    got = md5_file(PANEL)
    if got != frame_panel_md5:
        raise SystemExit(f"market_panel.parquet md5 {got[:8]} is not the frames' panel {frame_panel_md5[:8]}")
    cols = ["CodConglomeradoPrudencial", "mca_code", "is_B", "year", "quarter"] + DEP_COLS
    p = pq.read_table(PANEL, columns=cols).to_pandas()
    p = p.drop_duplicates(subset=["CodConglomeradoPrudencial", "mca_code", "year", "quarter"])
    p = p[(pd.to_numeric(p["is_B"], errors="coerce") == 1) & p["year"].between(2016, 2024)]
    p["mca_code"] = p["mca_code"].astype(str)
    p["time_id"] = p["year"].astype(int).astype(str) + "Q" + p["quarter"].astype(int).astype(str)
    p["S"] = p[DEP_COLS].fillna(0.0).sum(axis=1)
    S = p.groupby(["mca_code", "time_id"])["S"].sum()
    return S, {"panel_md5": got, "B_cells": int(len(p)), "market_quarters": int(len(S))}


# ------------------------------------------------------------------------------- the market size
def _quant(s, q):
    s = s.dropna()
    return 0.0 if s.empty else float(s.quantile(q))


def build_ms(P, df_spec, cfg, S_full=None):
    """sleep_demand_prep_link.build_market_size_and_shares with the anchor generalised (module docstring).
    Where cfg is today's (anchor dens, c_a = c, pooled, q 0.95) every operation is production's, in
    production's order. Returns (frame, info)."""
    phi_cap, share_cap, q = cfg["phi_cap"], cfg["share_cap"], cfg["q"]
    anchor_cap = cfg["anchor_cap"] if cfg["anchor_cap"] is not None else phi_cap
    if P.BC_RULE != "findex_gdp":
        raise SystemExit(f"BC_RULE {P.BC_RULE}: build_ms mirrors findex_gdp only")
    FALLBACK_BC = 1.1
    phi_row = np.where(df_spec['is_B'], df_spec['phi_mt'], df_spec['phi_t'])
    df_spec['_1mphi'] = 1.0 - phi_row.clip(0.0, phi_cap)
    df_spec['_1mphi_a'] = 1.0 - phi_row.clip(0.0, anchor_cap)      # the anchor's own (1 - phi)
    _fx = pd.to_numeric(df_spec.get('findex_banked_frac'), errors='coerce').fillna(P.FINDEX_FALLBACK) \
        if 'findex_banked_frac' in df_spec.columns else pd.Series(P.FINDEX_FALLBACK, index=df_spec.index)
    _fx = _fx.clip(0.3, 1.0)
    _x = pd.to_numeric(df_spec['gdp_per_capita'], errors='coerce')
    if _st.centering_enabled():
        _x = _x + _st.load_transform().mean_scaled['gdp_per_capita']
    _med = _x.groupby(df_spec['year']).transform('median')
    _acc = (_x / _med).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(P.ACCESS_LO, P.ACCESS_HI)
    df_spec['_access'] = _acc
    df_spec['_bc'] = 1.0 / (_fx * _acc)

    keys = ['mca_code', 'time_id']
    mkt = df_spec.groupby(keys).agg(_pop=('pop_total', 'first'), _bc_mt=('_bc', 'first'),
                                    _acc_mt=('_access', 'first'))
    _b = df_spec[df_spec['is_B']]
    mkt = mkt.join(_b.groupby(keys)['_1mphi'].first().rename('_1mphi'), how='left')
    mkt['_1mphi'] = mkt['_1mphi'].fillna(1.0 - phi_cap)
    b_dep_mt = _b.groupby(keys)['Dep_Act'].sum().rename('_dep_B')
    mkt = mkt.join(b_dep_mt, how='left').fillna({'_dep_B': 0.0})
    mkt = mkt.join(_b.groupby(keys)['_1mphi_a'].first().rename('_1mphi_a'), how='left')
    mkt['_1mphi_a'] = mkt['_1mphi_a'].fillna(1.0 - anchor_cap)
    isBm = mkt.index.isin(_b.groupby(keys).size().index)
    tq = mkt.index.get_level_values('time_id')
    quarters = pd.Index(sorted(set(tq)), name='time_id')

    info = {"anchor": cfg["anchor"], "pool": cfg["pool"], "q": q, "anchor_cap": anchor_cap}
    pooled_dens = cfg["anchor"] == "dens" and cfg["pool"] == "pooled"
    if cfg["anchor"] == "dens":
        local_ratio = ((mkt['_dep_B'] / (mkt['_1mphi_a'] * mkt['_pop'])) * mkt['_acc_mt']) \
            .replace([np.inf, -np.inf], np.nan)
        dep_over_1mphi = (df_spec['Dep_Act'] / df_spec['_1mphi_a']).replace([np.inf, -np.inf], np.nan)
        all_dep_t = dep_over_1mphi.groupby(df_spec['time_id']).sum()
        nat_pop_t = mkt.groupby('time_id')['_pop'].sum()
        nat_ratio = (all_dep_t / nat_pop_t).replace([np.inf, -np.inf], np.nan)
        info["adens"] = local_ratio[isBm]
        if cfg["pool"] == "pooled":
            r_loc, r_nat = _quant(local_ratio, q), _quant(nat_ratio, q)
            max_ratio = max(r_loc, r_nat) or 1.0
            r_t = pd.Series(max_ratio, index=quarters)
            info.update(r_local=r_loc, r_national=r_nat, binds="local" if r_loc >= r_nat else "national")
        elif cfg["pool"] == "quarter":
            r_loc_t = local_ratio[isBm].dropna().groupby(level='time_id').quantile(q)
            r_nat_t = nat_ratio.reindex(r_loc_t.index)
            r_t = pd.concat([r_loc_t, r_nat_t], axis=1).max(axis=1)
            info.update(r_local_t=r_loc_t, r_national_t=r_nat_t,
                        national_binds_quarters=int((r_nat_t > r_loc_t).sum()))
        else:
            raise SystemExit(f"pool {cfg['pool']}")
    elif cfg["anchor"] == "stock":
        if cfg["pool"] != "quarter":
            raise SystemExit("the stock anchor is by quarter")
        S = S_full.reindex(mkt.index)
        dens_s = ((S / mkt['_pop']) * mkt['_acc_mt']).replace([np.inf, -np.inf], np.nan)[isBm]
        if dens_s.isna().any():
            raise SystemExit(f"stock anchor: {int(dens_s.isna().sum())} B market-quarters without a panel stock")
        r_t = dens_s.groupby(level='time_id').quantile(q)
        info["adens"] = dens_s
    else:
        raise SystemExit(f"anchor {cfg['anchor']}")
    r_t = r_t.reindex(quarters)
    if not (np.isfinite(r_t.to_numpy()).all() and (r_t > 0).all()):
        raise SystemExit("an anchor r_t is missing or not positive")
    info["r_t"] = r_t

    if pooled_dens:
        mkt['_b_mkt'] = mkt['_bc_mt'] * max_ratio * mkt['_pop']          # production's line
    else:
        mkt['_b_mkt'] = mkt['_bc_mt'] * r_t.reindex(tq).to_numpy() * mkt['_pop']
    _floor_B = ((_b.assign(_x=lambda x: x['Dep_Act'] / x['_1mphi'])
                 .groupby(keys)['_x'].sum()) / share_cap)
    _floor_B = _floor_B.reindex(mkt.index).replace([np.inf, -np.inf], np.nan)
    M_anchor = mkt['_b_mkt'].copy()
    bind = _floor_B > mkt['_b_mkt']
    mkt['_b_mkt'] = np.maximum(mkt['_b_mkt'], _floor_B.fillna(0.0))
    info["mkt"] = pd.DataFrame({"M_anchor": M_anchor[isBm], "floor": _floor_B[isBm], "bind": bind[isBm],
                                "M": mkt.loc[isBm, '_b_mkt']})
    info["n_bind"] = int(bind[isBm].sum())

    df_spec = df_spec.merge(mkt[['_b_mkt']].reset_index(), on=keys, how='left')
    mkt_r = mkt.reset_index()
    mkt_r['_w'] = mkt_r['_bc_mt'] * mkt_r['_pop']
    sw = mkt_r.groupby('time_id')['_w'].sum()
    if pooled_dens:                                                    # production's lines
        nat_dbar = (mkt_r.groupby('time_id')
                    .apply(lambda g: (g['_w'].sum() / g['_pop'].sum()) * max_ratio if g['_pop'].sum() > 0
                           else FALLBACK_BC * max_ratio, include_groups=False).rename('_dbar_nat'))
    else:
        sp = mkt_r.groupby('time_id')['_pop'].sum()
        nat_dbar = ((sw / sp).where(sp > 0, FALLBACK_BC) * r_t.reindex(sw.index)).rename('_dbar_nat')
    nat_pop = mkt_r.groupby('time_id')['_pop'].sum().rename('_pop_nat')
    d_mkt = (nat_dbar * nat_pop).rename('_d_mkt').reset_index()
    _d_num = (df_spec.loc[~df_spec['is_B']]
              .assign(_x=lambda x: x['Dep_Act'] / x['_1mphi'])
              .groupby('time_id')['_x'].sum().rename('_dfloor'))
    d_mkt = d_mkt.merge((_d_num / share_cap).reset_index(), on='time_id', how='left')
    dm = pd.DataFrame({"M_nat_anchor": d_mkt['_d_mkt'].to_numpy(), "floor": d_mkt['_dfloor'].to_numpy(),
                       "bind": (d_mkt['_dfloor'] > d_mkt['_d_mkt']).to_numpy()}, index=d_mkt['time_id'].to_numpy())
    # The B markets' part of the D market's anchor sum, sum_m Pop_m / (fx acc_m); the rest is the D rows'
    # NATIONAL market (Pop = the national population), which the sum also runs over.
    swB = mkt_r.loc[isBm, '_w'].groupby(mkt_r.loc[isBm, 'time_id']).sum()
    dm["B_markets_share_of_anchor_sum"] = (swB / sw).reindex(dm.index).to_numpy()
    info["d_mkt"] = dm
    d_mkt['_d_mkt'] = np.maximum(d_mkt['_d_mkt'], d_mkt['_dfloor'].fillna(0.0))
    d_mkt = d_mkt.drop(columns=['_dfloor'])
    df_spec = df_spec.merge(d_mkt, on='time_id', how='left')

    df_spec['share_B_cond'] = np.where(
        df_spec['is_B'], df_spec['Dep_Act'] / (df_spec['_1mphi'] * df_spec['_b_mkt']), np.nan)
    df_spec['share_D'] = np.where(
        ~df_spec['is_B'], df_spec['Dep_Act'] / (df_spec['_1mphi'] * df_spec['_d_mkt']), np.nan)
    df_spec['M_mt'] = df_spec['_b_mkt']
    df_spec['M_nat'] = df_spec['_d_mkt']
    df_spec.drop(columns=['_b_mkt', '_d_mkt', '_bc', '_1mphi_a'], inplace=True)
    return df_spec, info


def _with_findex(d, cfg, fx_year):
    x = d.copy()
    if cfg["findex"] == "year":
        x["findex_banked_frac"] = d["year"].map(fx_year).to_numpy()
        if x["findex_banked_frac"].isna().any():
            raise SystemExit("a frame year has no Findex value")
    elif cfg["findex"] != "const":
        raise SystemExit(f"findex {cfg['findex']}")
    return x


def _restore(y, d):
    if len(y) != len(d) or not (y["entity_id"].to_numpy() == d["entity_id"].to_numpy()).all() \
            or not (y["time_id"].to_numpy() == d["time_id"].to_numpy()).all():
        raise SystemExit("the market-size build changed the row order")
    y.index = d.index
    return y[d.columns]


def prod_build(P, d, cfg, fx_year):
    """Production's build_market_size_and_shares on a copy of d (PHI_CAP and TOTAL_SHARE_CAP patched)."""
    if not (cfg["anchor"] == "dens" and cfg["pool"] == "pooled" and cfg["q"] == 0.95
            and cfg["anchor_cap"] in (None, cfg["phi_cap"])):
        raise SystemExit("production's function builds today's anchor only")
    x = _with_findex(d, cfg, fx_year)
    keep = (P.PHI_CAP, P.TOTAL_SHARE_CAP)
    try:
        P.PHI_CAP, P.TOTAL_SHARE_CAP = cfg["phi_cap"], cfg["share_cap"]
        y = P.build_market_size_and_shares(x)
    finally:
        P.PHI_CAP, P.TOTAL_SHARE_CAP = keep
    return _restore(y, d)


def alt_build(P, d, cfg, fx_year, S_full):
    x = _with_findex(d, cfg, fx_year)
    y, info = build_ms(P, x, cfg, S_full)
    return _restore(y, d), info


def compare(a, b):
    """Every column of two frames: exactly equal (NaN = NaN)?"""
    out, worst = {}, 0.0
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
            eq = bool(np.array_equal(x, y, equal_nan=True))
            if not eq:
                with np.errstate(invalid="ignore", divide="ignore"):
                    diff = np.nanmax(np.abs(x.astype(float) - y.astype(float)))
                    rel = np.nanmax(np.abs(x.astype(float) / y.astype(float) - 1))
                out[c] = {"max_abs": float(diff), "max_rel": float(rel)}
                worst = max(worst, float(rel))
        else:
            if not a[c].equals(b[c]):
                out[c] = {"max_abs": None, "max_rel": None}
                worst = np.inf
    return {"exact": not out, "columns_differing": out, "worst_rel": worst}


# ----------------------------------------------------------------------------- the diagnostics
def wmedian(v, w):
    v, w = np.asarray(v, float), np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if not len(v):
        return np.nan
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return float(v[o][np.searchsorted(cw, 0.5)])


def qd(s, qs=(0.10, 0.25, 0.50, 0.75, 0.90, 0.99)):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    return {f"p{int(round(q * 100)):02d}": float(s.quantile(q)) for q in qs} if len(s) else {}


def frame_diag(y, info, cfg, S_full):
    """The handoff's numbers for one frame. info: build_ms's (or the stored frame's reconstruction)."""
    share_cap, phi_cap = cfg["share_cap"], cfg["phi_cap"]
    B = y["is_B"].to_numpy(bool)
    b = y[B]
    keys = [b["mca_code"], b["time_id"]]
    gb = b.groupby(keys)
    mk = pd.DataFrame({"pop": gb["pop_total"].first(), "acc": gb["_access"].first(), "phi": gb["phi_mt"].first(),
                       "M": gb["M_mt"].first(), "S_stored": gb["deposit_balance"].sum(), "act": gb["Dep_Act"].sum(),
                       "SB": gb["share_B_cond"].sum(), "year": gb["year"].first(),
                       "n_rows": gb.size()})
    mk.index.names = ["mca_code", "time_id"]
    mk["X"] = (b["Dep_Act"] / b["_1mphi"]).groupby(keys).sum()
    mk["S"] = S_full.reindex(mk.index)
    mi = info["mkt"]
    mk = mk.join(mi[["M_anchor", "bind"]])
    if "note" not in info and not np.allclose(mk["M"], np.maximum(mk["M_anchor"], mk["X"] / share_cap),
                                              rtol=1e-12, atol=0):
        raise SystemExit("frame_diag: M is not max(anchor, floor)")
    mk["b"] = mk["S"] / mk["M"]
    mk["b_stored"] = mk["S_stored"] / mk["M"]
    yrs = sorted(mk["year"].unique())
    out = {"rows": int(len(y)), "B_rows": int(B.sum()), "D_rows": int((~B).sum()), "B_market_quarters": int(len(mk)),
           "cfg": {k: v for k, v in cfg.items()}}
    r_t = info["r_t"]
    out["rhat_t"] = {k: float(v) for k, v in r_t.items()}
    out["rhat_shown"] = {k: float(r_t.get(k, np.nan)) for k in QUARTERS_SHOWN}
    out["rhat_last_over_first"] = float(r_t.iloc[-1] / r_t.iloc[0])
    out["rhat_pooled"] = float(r_t.iloc[0]) if r_t.nunique() == 1 else None
    for k in ("r_local", "r_national", "binds", "national_binds_quarters"):
        if k in info:
            out[k] = info[k]
    # B stocks over M
    by = {}
    for yy in yrs:
        s = mk[mk["year"] == yy]
        by[int(yy)] = {"median": float(s["b"].median()), "dep_weighted_median": wmedian(s["b"], s["S"]),
                       "aggregate": float(s["S"].sum() / s["M"].sum()),
                       "stored_rows_aggregate": float(s["S_stored"].sum() / s["M"].sum()),
                       "stored_rows_median": float(s["b_stored"].median()),
                       "floor_share": float(s["bind"].mean()),
                       "stock_coverage_of_stored_rows": float(s["S_stored"].sum() / s["S"].sum())}
    out["B_stock_over_M_by_year"] = by
    out["B_stock_over_M"] = {"median": float(mk["b"].median()), "dep_weighted_median": wmedian(mk["b"], mk["S"]),
                             "aggregate": float(mk["S"].sum() / mk["M"].sum()), "quantiles": qd(mk["b"]),
                             "stored_rows_median": float(mk["b_stored"].median()),
                             "stored_rows_aggregate": float(mk["S_stored"].sum() / mk["M"].sum())}
    first_q, last_q = sorted(set(mk.index.get_level_values(1)))[0], sorted(set(mk.index.get_level_values(1)))[-1]
    for lab, qq in (("first_quarter", first_q), ("last_quarter", last_q), ("2020Q4", "2020Q4")):
        s = mk.xs(qq, level=1)
        out["B_stock_over_M"][lab] = {"quarter": qq, "aggregate": float(s["S"].sum() / s["M"].sum()),
                                      "stored_rows_aggregate": float(s["S_stored"].sum() / s["M"].sum()),
                                      "median": float(s["b"].median())}
    # the floor
    out["floor_share"] = float(mk["bind"].mean())
    out["floor_n"] = int(mk["bind"].sum())
    out["floor_share_ordinary"] = float(mk.loc[mk["phi"] < PHI_ORD, "bind"].mean())
    out["floor_share_pop_weighted"] = float(np.average(mk["bind"], weights=mk["pop"]))
    # the amplification X / S (active-equivalent balance over the booked stock)
    out["amplification_X_over_S"] = qd(mk["X"] / mk["S"])
    out["amplification_X_over_S_stored"] = qd(mk["X"] / mk["S_stored"])
    # inside and outside shares
    g, G, K, SB = dcs.closed_form(y)
    gq = pd.Series(np.asarray(g), index=SB.index)
    ordm = (mk["phi"] < PHI_ORD).reindex(SB.index).to_numpy()
    out.update(inside_sum_median=float(SB.median()), inside_sum_median_ordinary=float(SB[ordm].median()),
               inside_sum_p90=float(SB.quantile(0.90)),
               one_minus_inside_median=float((1 - SB).median()),
               outside_share_median=float(gq.median()), outside_share_median_ordinary=float(gq[ordm].median()),
               outside_share_p10=float(gq.quantile(0.10)),
               G_min=float(G.min()), G_nonpositive_quarters=int((G <= 0).sum()),
               G_nonpositive_list=[str(t) for t in G.index[G <= 0]], quarters=int(len(G)),
               K_min=float(K.min()))
    SD = y.loc[~B, "share_D"].groupby(y.loc[~B, "time_id"]).sum()
    out["D_share_sum"] = {"min": float(SD.min()), "median": float(SD.median()), "max": float(SD.max()),
                          "quarter_of_max": str(SD.idxmax())}
    dm = info.get("d_mkt")
    if dm is not None:
        out["D_floor_quarters"] = int(dm["bind"].sum())
        if "B_markets_share_of_anchor_sum" in dm:
            out["D_market_B_markets_share_of_anchor_sum"] = qd(dm["B_markets_share_of_anchor_sum"], (0.0, 0.5, 1.0))
    # extreme shares
    sB = b["share_B_cond"]
    out.update(extreme_rows_share=float((sB > EXTREME).mean()), extreme_rows_n=int((sB > EXTREME).sum()),
               extreme_mq_share=float((SB > EXTREME).mean()),
               row_share_median=float(sB.median()), row_share_p99=float(sB.quantile(0.99)),
               B_mq_at_share_cap=int(((SB - share_cap).abs() <= 1e-9).sum()),
               B_rows_at_phi_cap_share=float((b["phi_mt"] >= phi_cap).mean()),
               mean_rho_1ms=dcs.mean_rho_1ms(y))
    # the markets at the anchor's quantile (+-1 percentile within the quarter, or over the pool)
    ad = info.get("adens")
    if ad is not None:
        ad = ad.reindex(mk.index)
        if cfg["pool"] == "quarter":
            rk = ad.groupby(level="time_id").rank(pct=True)
        else:
            rk = ad.rank(pct=True)
        band = (rk - cfg["q"]).abs() <= 0.01
        s = mk[band.to_numpy()]
        out["anchor_band"] = {"n": int(len(s)), "B_stock_over_M_median": float(s["b"].median()),
                              "inside_sum_median": float(s["SB"].median()),
                              "amplification_median": float((s["X"] / s["S"]).median()),
                              "floor_share": float(s["bind"].mean())}
    return out


def stored_info(d, fx):
    """build_ms's info for a stored frame, reconstructed: r = the median of M fx acc / Pop over the B markets
    (the anchor wherever the floor does not bind), the floor binding where M exceeds r Pop / (fx acc)."""
    b = d[d["is_B"].to_numpy(bool)]
    key = [b["mca_code"], b["time_id"]]
    mk = pd.DataFrame({"pop": b["pop_total"].groupby(key).first(), "acc": b["_access"].groupby(key).first(),
                       "M": b["M_mt"].groupby(key).first()})
    mk.index.names = ["mca_code", "time_id"]
    R = float((mk["M"] * fx * mk["acc"] / mk["pop"]).median())
    anchor_M = R * mk["pop"] / (fx * mk["acc"])
    bind = mk["M"] > anchor_M * (1 + 1e-9)
    mk["M_anchor"] = np.where(bind, anchor_M, mk["M"])
    q = pd.Index(sorted(set(mk.index.get_level_values(1))), name="time_id")
    return {"r_t": pd.Series(R, index=q), "mkt": pd.DataFrame({"M_anchor": mk["M_anchor"], "bind": bind}),
            "note": "reconstructed from the stored M_mt"}


def write_frame(y, schema0, path, meta):
    t = pa.Table.from_pandas(y)
    if t.schema.names != schema0.names:
        raise SystemExit(f"{path.name}: columns differ from the stored frame's")
    if not t.schema.remove_metadata().equals(schema0.remove_metadata()):
        t = t.cast(schema0.remove_metadata())
    md = dict(t.schema.metadata or {})
    md[demand_chars.PANEL_MD5_KEY] = schema0.metadata[demand_chars.PANEL_MD5_KEY]
    md.update({f"msize.{k}".encode(): str(v).encode() for k, v in meta.items()})
    t = t.replace_schema_metadata(md)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    pq.write_table(t, part)
    os.replace(part, path)


def step_frames(variants, routines, sb):
    P = dcs.prep_module()
    fx_year = findex_by_year()
    sb.mkdir(parents=True, exist_ok=True)
    marker = sb / "FRAMES_COMPLETE.json"
    if marker.exists():
        os.replace(marker, sb / "FRAMES_COMPLETE.previous.json")
    manifest = {"started": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "variants": {}, "sources": {},
                "findex_by_year": fx_year}
    diag = {v: {} for v in variants}
    prod_diag, gates = {}, {}
    S_full, panel_rec = None, None
    for E in routines:
        name = FRAMES[E]
        src = DP / name
        t0 = pq.read_table(src)
        schema0 = t0.schema
        d = t0.to_pandas()
        del t0
        d["is_B"] = d["is_B"].astype(bool)
        pmd5 = schema0.metadata[demand_chars.PANEL_MD5_KEY].decode()
        manifest["sources"][f"E{E}"] = {"path": str(src), "md5": md5_file(src), "panel_md5": pmd5}
        if S_full is None:
            S_full, panel_rec = panel_stock(pmd5)
            manifest["panel_stock"] = panel_rec
            stamp(f"panel stock: {panel_rec}")
        prod_diag[f"E{E}"] = frame_diag(d, stored_info(d, P.FINDEX_FALLBACK), dict(PROD_CFG), S_full)
        stamp(f"E{E}: stored frame {len(d):,} rows; floor {prod_diag[f'E{E}']['floor_share']:.4f}; "
              f"B stock / M median {prod_diag[f'E{E}']['B_stock_over_M']['median']:.4f}")
        built = {}
        for v in variants:
            cfg = VARIANTS[v]
            t = time.time()
            if cfg["builder"] == "prod":
                y = prod_build(P, d, cfg, fx_year)
                y_alt, info = alt_build(P, d, cfg, fx_year, S_full)
                gates[f"E{E}_{v}_build_ms_equals_production"] = compare(y_alt, y)
                del y_alt
                built[v] = y
            else:
                y, info = alt_build(P, d, cfg, fx_year, S_full)
            out = sb / v / "demand_prep" / name
            if cfg["logit"]:
                write_frame(y, schema0, out, {"variant": v, **{k: cfg[k] for k in
                                                                ("findex", "anchor", "anchor_cap", "pool", "q",
                                                                 "phi_cap", "share_cap")},
                                              "source_md5": manifest["sources"][f"E{E}"]["md5"]})
                manifest["variants"].setdefault(v, {})[f"E{E}"] = {"path": str(out), "md5": md5_file(out),
                                                                  "rows": int(len(y)), "bytes": out.stat().st_size}
            diag[v][f"E{E}"] = frame_diag(y, info, cfg, S_full)
            r = diag[v][f"E{E}"]
            stamp(f"E{E} {v}: r {r['rhat_shown']['2016Q1']:,.0f} -> {r['rhat_shown']['2024Q4']:,.0f}; floor "
                  f"{r['floor_share']:.4f}; B stock/M median {r['B_stock_over_M']['median']:.4f} (dep-w "
                  f"{r['B_stock_over_M']['dep_weighted_median']:.4f}); inside median {r['inside_sum_median']:.4f}; "
                  f"G_min {r['G_min']:.4f} ({r['G_nonpositive_quarters']} q <= 0); {time.time() - t:.0f} s; "
                  f"peak {peak_gb()} GB")
            if v not in ("base", "F"):
                del y
        # the gates G1-G3: build_ms against production's function
        for gname, (gcfg, ref) in GATE_CFG.items():
            if ref not in built:
                continue
            y_alt, _ = alt_build(P, d, gcfg, fx_year, S_full)
            gates[f"E{E}_{gname}"] = compare(y_alt, built[ref])
            stamp(f"E{E} gate {gname} vs {ref}: exact {gates[f'E{E}_{gname}']['exact']}")
            del y_alt
        del built, d
    for v in variants:
        p = sb / v / "frame_diag.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        old.update(diag[v])
        jdump(old, p)
    p = sb / "production_frame_diag.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    old.update(prod_diag)
    jdump(old, p)
    p = sb / "gates.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    old.update(gates)
    jdump(old, p)
    prev = sb / "FRAMES_COMPLETE.previous.json"
    if prev.exists():
        pm = json.loads(prev.read_text(encoding="utf-8"))
        for v, rec in pm.get("variants", {}).items():
            for E, r in rec.items():
                manifest["variants"].setdefault(v, {}).setdefault(E, r)
        for E, r in pm.get("sources", {}).items():
            manifest["sources"].setdefault(E, r)
    manifest["finished"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    manifest["peak_working_set_GB"] = peak_gb()
    jdump(manifest, marker)
    bad = [k for k, g in gates.items() if not g["exact"]]
    stamp(f"frames complete: {marker}; gates not exact: {bad}")


PROD_CFG = _v(findex="const", builder="prod")


# ------------------------------------------------------------------------------------- the logit
def step_logit(variants, sb, threads):
    if not (sb / "FRAMES_COMPLETE.json").exists():
        raise SystemExit(f"{sb / 'FRAMES_COMPLETE.json'} is missing: the frame set is partial; run --step frames")
    jl = shutil.which("julia")
    for v in variants:
        if not VARIANTS[v]["logit"]:
            continue
        dp, od = sb / v / "demand_prep", sb / v / "out"
        summ = od / "logit" / "logit_summary_spec_12.json"
        newest = max(p.stat().st_mtime for p in dp.glob("*.parquet"))
        if summ.exists() and summ.stat().st_mtime > newest:
            stamp(f"logit {v}: summary newer than its frames, skipped")
            continue
        fd = json.loads((sb / v / "frame_diag.json").read_text(encoding="utf-8"))
        ok = [E for E, r in fd.items() if r["G_nonpositive_quarters"] == 0]
        if not ok:
            stamp(f"logit {v}: G_t <= 0 in some quarter of every routine ({ {E: r['G_nonpositive_quarters'] for E, r in fd.items()} }); not run")
            continue
        t = time.time()
        with open(sb / v / "logit.log", "w", encoding="utf-8") as fh:
            r = subprocess.run([jl, "--project=.", f"--threads={threads}", "blp_logit.jl", "--demand-dir", str(dp),
                                "--out-dir", str(od)], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                               env=dcs.julia_env())
        stamp(f"logit {v}: exit {r.returncode} in {time.time() - t:.0f} s (routines with G_t > 0: {ok})")


# ---------------------------------------------------------------------------------------- W4.1
QS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
POST_QI = 2020 * 4 + 4 - 1
LAM = lambda z: 1.0 / (1.0 + np.exp(-z))
W41_CASE = dict(g0=0.0, g0n=0.0, g1=0.17, gP=1.0, gPn=0.0, kc=1.0)   # w4_feasibility.py "cost_only"


def wq(v, w=None):
    """w4_feasibility.py's quantile table."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    v = v[ok]
    if w is None:
        out = {f"p{int(round(q * 100)):02d}": float(np.quantile(v, q)) for q in QS}
        out["mean"] = float(v.mean())
    else:
        w = np.asarray(w, float)[ok]
        o = np.argsort(v)
        cw = np.cumsum(w[o]) / w.sum()
        out = {f"p{int(round(q * 100)):02d}": float(v[o][np.searchsorted(cw, q)]) for q in QS}
        out["mean"] = float(np.average(v, weights=w))
    out["n"] = int(len(v))
    return out


class W4:
    """The pieces of w4_feasibility.py that the pools and the composition of W need (the branches term
    enters with gamma2 = 0 and is left out; it adds exactly zero there)."""

    def __init__(self, ctx_path):
        with open(ctx_path, "rb") as fh:
            ctx = pickle.load(fh)
        U, MT = ctx["U"].copy(), ctx["MT"].copy()
        self.inputs = {"ctx": str(ctx_path), "ctx_md5": md5_file(ctx_path), "ctx_N0_panel": ctx["N0"].get("panel_md5"),
                       "ctx_frame_panel": ctx["N0"].get("frame_panel_md5")}
        mt = U["mt"].to_numpy()
        self.mt, self.nmt, self.MT = mt, len(MT), MT
        U["TD"] = np.where(U["W"] > 0, U["carry4"] / U["W"].where(U["W"] > 0), np.nan)
        a = pd.read_csv(ROUT / "p2_advertising_conglomerate_quarter.csv", dtype={"CodConglomeradoPrudencial": str})
        self.inputs["p2_advertising_md5"] = md5_file(ROUT / "p2_advertising_conglomerate_quarter.csv", lf=True)
        obs = ("observed_positive", "observed_zero", "observed_negative", "imputed_from_row_total")
        a["adv"] = a["adv_brl_real_2024q4"].where(a["adv_status"].isin(obs)).clip(lower=0.0)
        a["x"] = np.log1p(a["adv"])
        a["qi"] = a["year"] * 4 + a["quarter"] - 1
        a = a.rename(columns={"CodConglomeradoPrudencial": "j"})
        U = U.merge(a[["j", "qi", "x"]], on=["j", "qi"], how="left")
        assert len(U) == len(ctx["U"]) and (U["mt"].to_numpy() == mt).all()
        self.dest = (U[["e1", "e2", "e4", "e5"]].sum(axis=1) > 0).to_numpy()
        self.post = (U["qi"].to_numpy() >= POST_QI).astype(float)
        src = (U["W"] > 0).to_numpy()
        td = U.loc[src, "TD"].to_numpy()
        Ud = pd.DataFrame({"r": np.arange(len(U)), "mt": mt, "j": U["j"].to_numpy()})
        Us = pd.DataFrame({"s": np.arange(len(U))[src], "mt": mt[src], "js": U["j"].to_numpy()[src]})
        pr = Ud.merge(Us, on="mt")
        pr = pr[pr["j"] != pr["js"]]
        self.pr_r, self.pr_s = pr["r"].to_numpy(), pr["s"].to_numpy()
        self.Wst = U["W"].to_numpy(float)
        self.Wtot = np.bincount(mt, weights=self.Wst, minlength=self.nmt)
        xc = U["x"].to_numpy(float) - np.nanmean(U.loc[self.dest, "x"])
        self.xc = np.where(np.isfinite(xc), xc, 0.0)
        tdbar = float(np.average(td, weights=U.loc[src, "W"].to_numpy()))
        tdc_all = U["TD"].to_numpy(float)
        self.tdc = np.where(np.isfinite(tdc_all), tdc_all - tdbar, 0.0)
        self.n = len(U)
        self.year_u = U["qi"].to_numpy() // 4
        self.keys = pd.MultiIndex.from_arrays([MT["m"].astype(str), MT["t"].astype(str)])
        self.M_hat = MT["M_hat"].to_numpy(float)

    def lnW(self, M, g0, g0n, g1, gP, gPn, kc):
        mt = self.mt
        bo = self.Wst / M[mt]
        bs = self.Wst / M[mt]
        bn = np.clip(1.0 - self.Wtot / M, 0.0, None)[mt]
        eta_d = g0 + g1 * self.xc + gP * self.post
        A_pair = LAM(eta_d[self.pr_r] - kc * self.tdc[self.pr_s])
        R = np.bincount(self.pr_r, weights=bs[self.pr_s] * A_pair, minlength=self.n)
        N = bn * LAM(g0n + g1 * self.xc + gPn * self.post)
        W = bo + R + N
        return np.log(W), bo / W, R / W, N / W

    def M_from_frame(self, frame):
        f = pq.read_table(frame, columns=["mca_code", "time_id", "is_B", "M_mt"]).to_pandas()
        f = f[f["is_B"].astype(bool)].groupby(["mca_code", "time_id"])["M_mt"].first()
        M = f.reindex(self.keys).to_numpy(float)
        return M, float(np.isfinite(M).mean())

    def metrics(self, M):
        mt, dest = self.mt, self.dest
        okM = np.isfinite(M[mt]) & (M[mt] > 0)
        dd = dest & okM
        bo = self.Wst / M[mt]
        brv = (self.Wtot[mt] - self.Wst) / M[mt]
        bn_raw = 1.0 - self.Wtot[mt] / M[mt]
        bn_mt_raw = 1.0 - self.Wtot / M
        fin = np.isfinite(bn_mt_raw)
        w = self.Wst[dd]
        out = {"b_new_dest": wq(bn_raw[dd]), "b_new_dest_own_stock_weighted": wq(bn_raw[dd], np.maximum(w, 1e-12)),
               "b_own_dest": wq(bo[dd]), "b_rivals_dest": wq(brv[dd]),
               "b_new_market_quarter": wq(bn_mt_raw[fin]),
               "b_new_aggregate_M_weighted": float(np.clip(1 - self.Wtot.sum() / M[np.isfinite(M)].sum(), 0, 1)),
               "share_mt_negative_new": float((bn_mt_raw < 0).mean()),
               "b_new_dest_median_by_year": {int(yy): float(np.median(bn_raw[dd & (self.year_u == yy)]))
                                             for yy in sorted(set(self.year_u[dd]))}}
        L, om, rho, nu = self.lnW(M, **W41_CASE)
        out["W_composition_availability_0p5"] = {"own": wq(om[dd]), "rivals": wq(rho[dd]), "new": wq(nu[dd])}
        return out


def step_w4(variants, sb, ctx_path):
    t = time.time()
    w = W4(ctx_path)
    stamp(f"w4: context loaded ({w.n:,} units, {w.nmt:,} market-quarters, {len(w.pr_r):,} pairs) in {time.time() - t:.0f} s")
    ref = json.loads(W41_JSON.read_text(encoding="utf-8"))
    res = {"inputs": w.inputs, "case": W41_CASE, "variants": {}}
    # gate W: W4.1's own numbers, production's M and the cap-0.999 frame
    gate = {}
    Mp4, cov = w.M_from_frame(CAPSENS_P4_E3)
    for lab, M, key in (("production_M_hat", w.M_hat, "base_cap099"), ("capsens_p4", Mp4, "cap0999")):
        m = w.metrics(M)
        mine = {"b_new_dest_p50": m["b_new_dest"]["p50"], "b_new_dest_p10": m["b_new_dest"]["p10"],
                "b_new_dest_stock_w_p50": m["b_new_dest_own_stock_weighted"]["p50"],
                "new_share_W_p50": m["W_composition_availability_0p5"]["new"]["p50"],
                "new_share_W_p10": m["W_composition_availability_0p5"]["new"]["p10"]}
        theirs = {"b_new_dest_p50": ref["pools"][key]["b_new"]["unweighted"]["p50"],
                  "b_new_dest_p10": ref["pools"][key]["b_new"]["unweighted"]["p10"],
                  "b_new_dest_stock_w_p50": ref["pools"][key]["b_new"]["own_stock_weighted"]["p50"],
                  "new_share_W_p50": ref["shifters"][key]["cost_only"]["composition_new"]["p50"],
                  "new_share_W_p10": ref["shifters"][key]["cost_only"]["composition_new"]["p10"]}
        dmax = max(abs(mine[k] - theirs[k]) for k in mine)
        gate[lab] = {"mine": mine, "w4_1": theirs, "max_abs_diff": dmax, "pass": bool(dmax < 1e-12)}
        res.setdefault("reference", {})[lab] = m
    res["gate_W"] = gate
    res["capsens_p4_matched"] = cov
    stamp(f"w4 gate W: { {k: (g['pass'], g['max_abs_diff']) for k, g in gate.items()} }")
    for v in variants:
        if not VARIANTS[v]["logit"]:
            continue
        fr = sb / v / "demand_prep" / FRAMES[3]
        if not fr.exists():
            continue
        M, cov = w.M_from_frame(fr)
        res["variants"][v] = {"matched": cov, **w.metrics(M)}
        m = res["variants"][v]
        stamp(f"w4 {v}: b_new dest p50 {m['b_new_dest']['p50']:.4f}; new share of W p50 "
              f"{m['W_composition_availability_0p5']['new']['p50']:.4f}")
    res["seconds"] = round(time.time() - t, 1)
    res["peak_working_set_GB"] = peak_gb()
    jdump(res, sb / "w4.json")


# ---------------------------------------------------------------------------------- the frontier
FRONTIER_Q = (0.99, 0.95, 0.90, 0.75, 0.50, 0.25)


def step_frontier(routines, sb):
    """Market level only (B markets; no D market, no logit): for the by-quarter anchors over a grid of q,
    the share-side cap c and TOTAL_SHARE_CAP, the floor incidence and the booked B stock over M. Whether any
    q meets both conditions. The stored frames' _access is production's (the gates find it unchanged). Also
    the ceiling the floor puts on any anchor: S / M <= cap S / X, so the median B stock over M cannot
    exceed the median of cap / (X / S) whatever the anchor."""
    fx_year = findex_by_year()
    rows, ceil, S_full = [], [], None
    for E in routines:
        src = DP / FRAMES[E]
        cols = ["mca_code", "time_id", "year", "is_B", "Dep_Act", "phi_mt", "pop_total", "_access"]
        d = pq.read_table(src, columns=cols).to_pandas()
        d = d[d["is_B"].astype(bool)]
        if S_full is None:
            S_full, _ = panel_stock(pq.read_schema(src).metadata[demand_chars.PANEL_MD5_KEY].decode())
        key = [d["mca_code"], d["time_id"]]
        g = d.groupby(key)
        mk = pd.DataFrame({"pop": g["pop_total"].first(), "acc": g["_access"].first(), "year": g["year"].first(),
                           "phi": g["phi_mt"].first()})
        mk.index.names = ["mca_code", "time_id"]
        for c in (0.99, 0.95, 0.999):
            mk[f"X{c}"] = (d["Dep_Act"] / (1 - d["phi_mt"].clip(0, c))).groupby(key).sum()
        mk["S"] = S_full.reindex(mk.index)
        mk["fx"] = mk["year"].map(fx_year)
        t = mk.index.get_level_values("time_id")
        for c in (0.99, 0.95, 0.999):
            for kap in (0.95, 0.80):
                bound = kap * mk["S"] / mk[f"X{c}"]
                ceil.append({"routine": f"E{E}", "phi_cap": c, "share_cap": kap,
                             "ceiling_median_Bstock_M": float(bound.median()),
                             "ceiling_p25": float(bound.quantile(0.25)), "ceiling_p10": float(bound.quantile(0.10)),
                             "amplification_median": float((mk[f"X{c}"] / mk["S"]).median())})
        for anchor, use_acc in (("stock", True), ("dens_c095", True), ("stock_no_access", False),
                                ("dens_c095_no_access", False)):
            acc = mk["acc"] if use_acc else 1.0
            dens = (mk["S"] if anchor.startswith("stock") else mk["X0.95"]) / mk["pop"] * acc
            for q in FRONTIER_Q:
                r_t = dens.groupby(level="time_id").quantile(q)
                Ma = r_t.reindex(t).to_numpy() * mk["pop"] / (mk["fx"] * acc)
                for c in (0.99, 0.95):
                    for kap in (0.95, 0.80):
                        fl = mk[f"X{c}"] / kap
                        M = np.maximum(Ma, fl)
                        b = mk["S"] / M
                        bind = fl > Ma
                        rows.append({"routine": f"E{E}", "anchor": anchor, "access": use_acc, "q": q, "phi_cap": c,
                                     "share_cap": kap,
                                     "floor_share": float(bind.mean()), "Bstock_M_median": float(b.median()),
                                     "Bstock_M_depw_median": wmedian(b, mk["S"]),
                                     "Bstock_M_aggregate": float(mk["S"].sum() / M.sum()),
                                     "Bstock_M_median_2016": float(b[mk["year"] == 2016].median()),
                                     "Bstock_M_median_2024": float(b[mk["year"] == 2024].median()),
                                     "inside_sum_median": float((mk[f"X{c}"] / M).median()),
                                     "meets_both": bool(b.median() > B_SHARE_GOAL and bind.mean() < FLOOR_GOAL)})
        del d
    F = pd.DataFrame(rows)
    C = pd.DataFrame(ceil)
    sb.mkdir(parents=True, exist_ok=True)
    F.to_csv(sb / "frontier.csv", index=False, lineterminator="\n")
    C.to_csv(sb / "frontier_ceiling.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(C.round(4).to_string())
    print(F[F["routine"] == "E3"].round(4).to_string())
    stamp(f"frontier: {int(F['meets_both'].sum())} of {len(F)} grid points meet both conditions: "
          f"{F.loc[F['meets_both'], ['routine', 'anchor', 'q', 'phi_cap', 'share_cap']].to_dict(orient='records')}")


# --------------------------------------------------------------- Part A's cell groups; the first stage
NETFLOW_GAP_A1 = ROUT / "diag_netflow_gap" / "a1_world_moving_share.csv"
W11_JSON = SCRATCH / "w1_1112" / "w1_logit_iv.json"
WORLD_VARIANTS = ("production", "A1q95", "A2qtr", "X_A1q95c95")
PHI_BINS = (0.0, 0.95, 0.98, 0.99, 0.999, 1.0 + 1e-12)
GROUPS = ("on_menu_phi_le_cap", "on_menu_phi_gt_cap", "off_menu_truncated")


def cell_groups(C, cap):
    """Part A's type-4 groups (diag_netflow_gap.a1_run's splits) at the share-side cap."""
    k4 = C["k"].to_numpy() == 4
    on = C["on_menu"].to_numpy()
    phi = C["phi_mt"].to_numpy()
    return {"on_menu_phi_le_cap": k4 & on & (phi <= cap), "on_menu_phi_gt_cap": k4 & on & (phi > cap),
            "off_menu_truncated": k4 & ~on}


def frame_floor_mq(frame, share_cap):
    """Per B market-quarter of a built frame: phi-hat and whether the floor binds (M = X / TOTAL_SHARE_CAP,
    X from the frame's own rows, which are the rows the floor was built on)."""
    d = pq.read_table(frame, columns=["mca_code", "time_id", "is_B", "Dep_Act", "_1mphi", "M_mt", "phi_mt"]).to_pandas()
    d = d[d["is_B"].astype(bool)]
    k = [d["mca_code"].astype(str), d["time_id"].astype(str)]
    X = (d["Dep_Act"] / d["_1mphi"]).groupby(k).sum()
    M = d["M_mt"].groupby(k).first()
    out = pd.DataFrame({"phi": d["phi_mt"].groupby(k).first(), "bind": (X / share_cap) >= M * (1 - 1e-12)})
    out.index.names = ["m", "t"]
    return out


def mop_feff(Z, X, y, cl):
    """W1.1's Montiel Olea-Pflueger effective F (scratchpad w1_1112/w1_logit_iv.py mop: no small-sample
    factor, and the CR1 version), on the first stage of one type: y on [X, Z] within the type."""
    keep = np.abs(X).max(axis=0) > 0
    X = X[:, keep]
    Q, _ = np.linalg.qr(X)
    Zt, yt = Z - Q @ (Q.T @ Z), y - Q @ (Q.T @ y)
    n, L = Zt.shape
    ev, Uv = np.linalg.eigh(Zt.T @ Zt / n)
    Zs = Zt @ (Uv @ np.diag(ev ** -0.5) @ Uv.T)
    pis = Zs.T @ yt / n
    v = yt - Zs @ pis
    c_, _ = pd.factorize(cl)
    G = int(c_.max() + 1)
    Sg = np.zeros((G, L))
    np.add.at(Sg, c_, Zs * v[:, None])
    W2 = Sg.T @ Sg / n
    feff = float(n * pis @ pis / np.trace(W2))
    cr1 = G / (G - 1) * (n - 1) / (n - (X.shape[1] + L))
    return {"n": int(n), "L": int(L), "G": G, "F_eff": feff, "F_eff_cr1": feff / cr1}


def first_stage_inputs(frame):
    """The type-4 first stage's inputs as blp_logit.jl builds them (W1.1's load): spread (pp / 100), X_COLS
    and the D dummy, the 16 instruments, the conglomerate; and an md5 of those arrays."""
    cols = dcs.X_COLS + dcs.IVS + ["is_B", "spread_ann", "deposit_type", "CodConglomeradoPrudencial"]
    d = pq.read_table(frame, columns=cols).to_pandas()
    X = np.column_stack([d[dcs.X_COLS].to_numpy(float), (~d["is_B"].to_numpy(bool)).astype(float)])
    Z = d[dcs.IVS].to_numpy(float)
    p = d["spread_ann"].to_numpy(float) / 100.0
    k = d["deposit_type"].to_numpy()
    cl = d["CodConglomeradoPrudencial"].astype(str).to_numpy()
    h = hashlib.md5()
    for a in (X, Z, p, k.astype(np.int64)):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update("|".join(cl).encode())
    return X, Z, p, k, cl, h.hexdigest()


def step_cellgroups(variants, sb, world_variants):
    """Part A's cell-group table per variant (types 4; diag_netflow_gap.a1_run's cells and splits) and the
    type-4 first stage's effective F. -> <sandbox>/cellgroups.json, <sandbox>/cellgroups.csv.
      data side  the cells of K0's mask with a world unit, built as Part A does (the a216 inputs, the
                 production frame); on the menu = a type-4 product in the frame; off the menu = cut by
                 (14-B). The data's |NF| / carry does not depend on M; the cap moves the membership.
      floor      per variant, the share of each group's cells whose market-quarter has the floor binding
                 (among the cells whose market-quarter is a B market of the frame), and the market-level
                 floor incidence by phi-hat bin.
      world      for world_variants, the K2 world rebuilt on the variant's frame (k0.DEMAND and k2.DEMAND
                 pointed at it; delta from the closed form, as Part A's worlds): W0 (stored DGP, kappa 0.5,
                 raw phi) and W3 (the stored delta at kappa 0, phi capped at the variant's cap) medians.
                 Gate CG: on the production frame the groups and both worlds reproduce Part A's
                 a1_world_moving_share.csv exactly.
      F_eff      W1.1's MOP F_eff of the type-4 first stage on production, base and A1q95, and the md5 of
                 the first stage's inputs in every variant's frame (identical inputs give identical F)."""
    import diag_netflow_gap as dng      # pins K0 to the I2 panel; its K2 import reads alpha (read-only)
    k0, k2 = dng.k0, dng.k2
    t0 = time.time()
    res = {"world_variants": list(world_variants), "groups": {}, "floor_by_group": {}, "floor_by_phi_bin": {},
           "world": {}, "first_stage": {}, "gates": {}}
    k0.COLS = k0.COLS + ["spread_a5"]
    df = k0.load_panel()
    flags = k0.rate_flags(df[df["is_B"] == 1])
    keep_dem = (k0.DEMAND, k2.DEMAND)
    mk = k0.load_demand()
    b = df[(df["is_B"] == 1) & (df["qi"] >= k0.qi(*k0.T0) - 1) & (df["qi"] <= k0.qi(*k0.T1))].copy()
    Lin = k0.netflow(b, flags, mk)
    del b
    ref = pd.read_csv(NETFLOW_GAP_A1)
    ref = ref[(ref["k"] == 4)]
    caps = sorted({VARIANTS[v]["phi_cap"] for v in variants} | {0.99})
    C = None
    for wv in [w for w in world_variants]:
        frame = DP / FRAMES[3] if wv == "production" else sb / wv / "demand_prep" / FRAMES[3]
        cap = 0.99 if wv == "production" else VARIANTS[wv]["phi_cap"]
        t1 = time.time()
        try:
            k0.DEMAND = k2.DEMAND = frame
            mk_v = k0.load_demand()
            ctx, menu, extra = dng.build_world(df, flags, mk_v, "closed")
        finally:
            k0.DEMAND, k2.DEMAND = keep_dem
        U, MT, SD, SD4, K = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["K"]
        Cv, cinfo = dng.a1_cells(Lin, U, K)
        if wv == "production":
            C = Cv[["j", "m", "t", "k", "carry", "NF", "phi_mt", "on_menu"]].copy()
        phi_raw = MT["phi_mt"].to_numpy()
        e4_delta = np.where(U["delta4"].notna(), np.exp(U["delta4"].to_numpy()), 0.0)
        _, nf0, _, _ = dng.simulate_omega(U, MT, SD, SD4, extra["e4"], dng.KAPPA_BASE, phi_raw)
        _, nf3, _, _ = dng.simulate_omega(U, MT, SD, SD4, e4_delta, 0.0, np.minimum(phi_raw, cap))
        u = Cv["u"].to_numpy()
        carry, nfd = Cv["carry"].to_numpy(), Cv["NF"].to_numpy()
        rec = {"cap": cap, "frame": str(frame), "delta_source": ctx["N"].get("delta_source"),
               "outside_pool_negative_mq": int((MT["O_raw"] < 0).sum()) if "O_raw" in MT else None}
        for g, s in cell_groups(Cv, cap).items():
            rec[g] = {"n_cells": int(s.sum()), "data_p50": float(np.median(np.abs(nfd[s]) / carry[s])),
                      "W0_p50": float(np.median(np.abs(nf0[u][s]) / carry[s])),
                      "W3_p50": float(np.median(np.abs(nf3[u][s]) / carry[s]))}
        res["world"][wv] = rec
        stamp(f"cellgroups world {wv} (cap {cap}): " + "; ".join(
            f"{g} n {rec[g]['n_cells']:,} data {rec[g]['data_p50']:.4f} W0 {rec[g]['W0_p50']:.4f} W3 {rec[g]['W3_p50']:.4f}"
            for g in GROUPS) + f"; {time.time() - t1:.0f} s; peak {peak_gb()} GB")
        if wv == "production":
            gate = {}
            for g, split in zip(GROUPS, ("on_menu_phi_le_cap", "on_menu_phi_gt_cap", "off_menu")):
                for world, col in (("W0", "W0_p50"), ("W3", "W3_p50")):
                    r = ref[(ref["world"] == world) & (ref["split"] == split)].iloc[0]
                    gate[f"{g}_{world}"] = {"n": [rec[g]["n_cells"], int(r["n_cells"])],
                                            "data_p50": [rec[g]["data_p50"], float(r["data_p50"])],
                                            "world_p50": [rec[g][col], float(r["world_p50"])]}
            dmax = max(max(abs(v["data_p50"][0] - v["data_p50"][1]), abs(v["world_p50"][0] - v["world_p50"][1]))
                       for v in gate.values())
            nok = all(v["n"][0] == v["n"][1] for v in gate.values())
            res["gates"]["CG_production_reproduces_part_A"] = {"rows": gate, "max_abs": dmax, "n_equal": nok,
                                                               "pass": bool(nok and dmax <= 1e-12)}
            stamp(f"gate CG: n equal {nok}, max abs {dmax:.1e}")
        del ctx, menu, extra, U, MT, SD, SD4, K, Cv, nf0, nf3
    if C is None:
        raise SystemExit("cellgroups needs the production world for the cells (world_variants must start with production)")
    # the data side at every cap, and the floor by group / by phi bin for every variant
    k4 = C["k"].to_numpy() == 4
    absr = np.abs(C["NF"].to_numpy()) / C["carry"].to_numpy()
    for cap in caps:
        res["groups"][str(cap)] = {g: {"share_of_type4_cells": float(s.sum() / k4.sum()), "n_cells": int(s.sum()),
                                       "data_p50": float(np.median(absr[s])),
                                       "data_p25_p75": [float(np.quantile(absr[s], 0.25)), float(np.quantile(absr[s], 0.75))]}
                                   for g, s in cell_groups(C, cap).items()}
    keys = pd.MultiIndex.from_arrays([C["m"].astype(str), C["t"].astype(str)])
    rows = []
    for v in variants:
        cfg = VARIANTS[v]
        fr = sb / v / "demand_prep" / FRAMES[3]
        if not fr.exists():
            continue
        fm = frame_floor_mq(fr, cfg["share_cap"])
        bind = fm["bind"].reindex(keys).to_numpy()
        has = ~pd.isna(bind)
        bind_b = np.where(has, bind, False).astype(bool)
        out_v = {}
        for g, s in cell_groups(C, cfg["phi_cap"]).items():
            sh = s & has
            out_v[g] = {"share_of_type4_cells": float(s.sum() / k4.sum()),
                        "share_with_a_frame_market": float(sh.sum() / max(s.sum(), 1)),
                        "floor_incidence": float(bind_b[sh].mean()) if sh.any() else np.nan,
                        "data_p50": float(np.median(absr[s])),
                        "data_p50_floor_markets": float(np.median(absr[sh & bind_b])) if (sh & bind_b).any() else np.nan,
                        "data_p50_other_markets": float(np.median(absr[sh & ~bind_b])) if (sh & ~bind_b).any() else np.nan}
            rows.append({"variant": v, "phi_cap": cfg["phi_cap"], "share_cap": cfg["share_cap"], "group": g, **out_v[g]})
        res["floor_by_group"][v] = out_v
        b_ = pd.cut(fm["phi"], PHI_BINS, right=False)
        res["floor_by_phi_bin"][v] = {str(iv): {"mq": int(len(s)), "floor_share": float(s.mean())}
                                      for iv, s in fm["bind"].groupby(b_, observed=False)}
    pd.DataFrame(rows).to_csv(sb / "cellgroups.csv", index=False, lineterminator="\n")
    # the first stage (type 4): W1.1's F_eff on three frames; the inputs' md5 in every variant
    fs_frames = {"production": DP / FRAMES[3], "base": sb / "base" / "demand_prep" / FRAMES[3],
                 "A1q95": sb / "A1q95" / "demand_prep" / FRAMES[3]}
    for lab, fr in fs_frames.items():
        X, Z, p, k, cl, h = first_stage_inputs(fr)
        r = k == 4
        res["first_stage"][lab] = {"inputs_md5": h, "type4": mop_feff(Z[r], X[r], p[r], cl[r])}
    w11 = json.loads(W11_JSON.read_text(encoding="utf-8"))["E3"]["mop_type4"]
    res["first_stage"]["w1_1_stored_production_E3_type4"] = {"F_eff": w11["F_eff"], "F_eff_cr1": w11["F_eff_cr1"],
                                                            "n": w11["n"]}
    md5s = {}
    for v in variants:
        fr = sb / v / "demand_prep" / FRAMES[3]
        if fr.exists():
            md5s[v] = first_stage_inputs(fr)[-1]
    res["first_stage"]["inputs_md5_by_variant"] = md5s
    res["first_stage"]["all_variants_same_inputs_as_production"] = bool(
        set(md5s.values()) == {res["first_stage"]["production"]["inputs_md5"]})
    fp = res["first_stage"]["production"]["type4"]["F_eff"]
    res["gates"]["FS_production_equals_W1_1"] = {"mine": fp, "w1_1": w11["F_eff"], "abs_diff": abs(fp - w11["F_eff"]),
                                                 "pass": bool(abs(fp - w11["F_eff"]) <= 1e-9 * max(1.0, abs(fp)))}
    res["seconds"] = round(time.time() - t0)
    res["peak_working_set_GB"] = peak_gb()
    jdump(res, sb / "cellgroups.json")
    stamp(f"first stage type 4 F_eff: production {fp:.4f} (W1.1 {w11['F_eff']:.4f}); base "
          f"{res['first_stage']['base']['type4']['F_eff']:.4f}; A1q95 {res['first_stage']['A1q95']['type4']['F_eff']:.4f}; "
          f"same inputs in every variant {res['first_stage']['all_variants_same_inputs_as_production']}")


# ----------------------------------------------------------------------------------- the summary
def step_summary(variants, sb, out):
    out.mkdir(parents=True, exist_ok=True)
    prod_summ = json.loads(PROD_LOGIT.read_text(encoding="utf-8"))
    prod_diag = json.loads((sb / "production_frame_diag.json").read_text(encoding="utf-8"))
    gates = json.loads((sb / "gates.json").read_text(encoding="utf-8"))
    w4 = json.loads((sb / "w4.json").read_text(encoding="utf-8")) if (sb / "w4.json").exists() else {}
    manifest = json.loads((sb / "FRAMES_COMPLETE.json").read_text(encoding="utf-8"))
    rows, missing, fdiag = [], [], {}
    order = ["production"] + [v for v in VARIANTS if v in variants]
    for E in FRAMES:
        base = dcs.logit_row(prod_summ, E)
        base["elasticity"] = base["alpha"] * prod_diag[f"E{E}"]["mean_rho_1ms"]
        for v in order:
            if v == "production":
                fd, lr, cfg = prod_diag[f"E{E}"], dict(base), PROD_CFG
                w = w4.get("reference", {}).get("production_M_hat") if E == 3 else None
                status = "production"
            else:
                cfg = VARIANTS[v]
                allfd = json.loads((sb / v / "frame_diag.json").read_text(encoding="utf-8"))
                fdiag[v] = allfd
                fd = allfd.get(f"E{E}")
                if fd is None:
                    continue
                sp = sb / v / "out" / "logit" / "logit_summary_spec_12.json"
                lr = dcs.logit_row(json.loads(sp.read_text(encoding="utf-8")), E) if sp.exists() else None
                if lr is None:
                    status = "not estimable (G_t <= 0)" if fd["G_nonpositive_quarters"] else "missing"
                    if status == "missing":
                        missing.append(f"{v} E{E}")
                    lr = {}
                else:
                    status = "estimated"
                lr["elasticity"] = lr["alpha"] * fd["mean_rho_1ms"] if "alpha" in lr else np.nan
                w = w4.get("variants", {}).get(v) if E == 3 else None
            bs, by = fd["B_stock_over_M"], fd["B_stock_over_M_by_year"]
            rh = fd["rhat_shown"]
            row = {"routine": f"E{E}", "variant": v, "kind": "production" if v == "production" else cfg["kind"],
                   "findex": cfg["findex"], "anchor": cfg["anchor"],
                   "anchor_phi_cap": cfg["anchor_cap"] if cfg["anchor_cap"] is not None else
                   (np.nan if cfg["anchor"] == "stock" else cfg["phi_cap"]),
                   "pool": cfg["pool"], "q": cfg["q"], "phi_cap": cfg["phi_cap"], "share_cap": cfg["share_cap"],
                   "rhat_pooled": fd["rhat_pooled"] if fd["rhat_pooled"] is not None else np.nan,
                   **{f"rhat_{k}": rh[k] for k in QUARTERS_SHOWN},
                   "rhat_last_over_first": fd["rhat_last_over_first"],
                   "Bstock_M_median": bs["median"], "Bstock_M_depw_median": bs["dep_weighted_median"],
                   "Bstock_M_aggregate": bs["aggregate"],
                   "Bstock_M_median_2016": by["2016"]["median"], "Bstock_M_median_2024": by["2024"]["median"],
                   "Bstock_M_depw_2016": by["2016"]["dep_weighted_median"],
                   "Bstock_M_depw_2024": by["2024"]["dep_weighted_median"],
                   "Bstock_M_agg_2016": by["2016"]["aggregate"], "Bstock_M_agg_2024": by["2024"]["aggregate"],
                   "Bstock_M_stored_agg_first_q": bs["first_quarter"]["stored_rows_aggregate"],
                   "Bstock_M_stored_agg_2020Q4": bs["2020Q4"]["stored_rows_aggregate"],
                   "Bstock_M_stored_agg_last_q": bs["last_quarter"]["stored_rows_aggregate"],
                   "b_new_dest_p50": w["b_new_dest"]["p50"] if w else np.nan,
                   "b_new_dest_stockw_p50": w["b_new_dest_own_stock_weighted"]["p50"] if w else np.nan,
                   "b_new_mq_p50": w["b_new_market_quarter"]["p50"] if w else np.nan,
                   "new_share_of_W_p50": w["W_composition_availability_0p5"]["new"]["p50"] if w else np.nan,
                   "new_share_of_W_p10": w["W_composition_availability_0p5"]["new"]["p10"] if w else np.nan,
                   "floor_share": fd["floor_share"], "floor_share_2016": by["2016"]["floor_share"],
                   "floor_share_2024": by["2024"]["floor_share"],
                   "floor_share_ordinary": fd["floor_share_ordinary"],
                   "D_floor_quarters": fd.get("D_floor_quarters", np.nan),
                   "inside_median": fd["inside_sum_median"], "inside_median_ordinary": fd["inside_sum_median_ordinary"],
                   "outside_median": fd["outside_share_median"],
                   "outside_median_ordinary": fd["outside_share_median_ordinary"],
                   "one_minus_inside_median": fd["one_minus_inside_median"],
                   "D_share_sum_max": fd["D_share_sum"]["max"],
                   "G_min": fd["G_min"], "G_nonpositive_quarters": fd["G_nonpositive_quarters"],
                   "extreme_rows_share": fd["extreme_rows_share"], "extreme_mq_share": fd["extreme_mq_share"],
                   "anchor_band_inside_median": fd.get("anchor_band", {}).get("inside_sum_median", np.nan),
                   "anchor_band_Bstock_M_median": fd.get("anchor_band", {}).get("B_stock_over_M_median", np.nan),
                   "logit_status": status,
                   "alpha": lr.get("alpha", np.nan), "se_wcb": lr.get("se", np.nan), "p": lr.get("p", np.nan),
                   "elasticity_mean": lr.get("elasticity", np.nan), "mean_rho_1ms": fd["mean_rho_1ms"],
                   "n_obs": lr.get("n_obs", np.nan), "G_star": lr.get("G_star", np.nan)}
            row["d_alpha_over_se_production"] = (row["alpha"] - base["alpha"]) / base["se"] \
                if np.isfinite(row["alpha"]) else np.nan
            row["converged_G_pos_all_quarters"] = bool(row["G_nonpositive_quarters"] == 0)
            row["meets_B_share_20pct_median"] = bool(row["Bstock_M_median"] > B_SHARE_GOAL)
            row["meets_B_share_20pct_depw"] = bool(row["Bstock_M_depw_median"] > B_SHARE_GOAL)
            row["meets_floor_2pct"] = bool(row["floor_share"] < FLOOR_GOAL)
            row["verdict_pass"] = bool(row["meets_B_share_20pct_median"] and row["meets_floor_2pct"]
                                       and row["converged_G_pos_all_quarters"])
            rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(out / "market_size_alt_summary.csv", index=False, lineterminator="\n")
    gate_L = {}
    for E in FRAMES:
        b0 = S[(S["routine"] == f"E{E}") & (S["variant"] == "production")].iloc[0]
        st = S[(S["routine"] == f"E{E}") & (S["variant"] == "base")]
        if len(st) and np.isfinite(st["alpha"].iloc[0]):
            dd = abs(st["alpha"].iloc[0] - b0["alpha"])
            gate_L[f"E{E}"] = {"abs_d_alpha": float(dd), "bound": float(0.05 * b0["se_wcb"]),
                               "pass": bool(dd < 0.05 * b0["se_wcb"])}
    J = {"generated": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
         "gates": {"L_base_logit_vs_production": gate_L, "G_build_ms_vs_production": gates,
                   "W_w41_reproduction": w4.get("gate_W")},
         "missing_logits": missing,
         "not_estimable": sorted(set(S.loc[S["logit_status"].str.startswith("not estimable"), "variant"])),
         "verdict_rule": f"B stock over M (panel booked B stock, all types) at the median B market-quarter above "
                         f"{B_SHARE_GOAL}, floor binding in fewer than {FLOOR_GOAL} of the B market-quarters, and "
                         f"G_t > 0 in every quarter",
         "variants_passing": sorted(set(S.loc[S["verdict_pass"], "variant"])),
         "rows": S.to_dict(orient="records"), "production_frame_diag": prod_diag, "frame_diag": fdiag,
         "w4": w4, "frames_manifest": manifest}
    for f in ("frontier.csv", "frontier_ceiling.csv", "cellgroups.csv"):
        if (sb / f).exists():
            shutil.copyfile(sb / f, out / f"market_size_alt_{f}")
            J[f.replace(".csv", "")] = pd.read_csv(sb / f).to_dict(orient="records")
    if (sb / "cellgroups.json").exists():
        cg = json.loads((sb / "cellgroups.json").read_text(encoding="utf-8"))
        J["cellgroups_detail"] = cg
        J["gates"]["CG_cellgroups_production_reproduces_part_A"] = cg["gates"].get("CG_production_reproduces_part_A")
        J["gates"]["FS_first_stage_production_equals_W1_1"] = cg["gates"].get("FS_production_equals_W1_1")
    jdump(J, out / "market_size_alt_summary.json")
    lines = [f"{md5_file(p, lf=p.suffix in ('.csv', '.json', '.txt'))}  {p.name}" for p in sorted(out.iterdir())
             if p.is_file() and p.name != "MD5SUMS.txt"]
    (out / "MD5SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    show = ["routine", "variant", "rhat_2016Q1", "rhat_2024Q4", "Bstock_M_median", "Bstock_M_depw_median",
            "b_new_dest_p50", "new_share_of_W_p50", "floor_share", "inside_median", "outside_median",
            "G_nonpositive_quarters", "extreme_rows_share", "alpha", "se_wcb", "p", "elasticity_mean"]
    print(S[show].round(4).to_string())
    stamp(f"summary: gate L {json.dumps(gate_L)}; G not exact {[k for k, g in gates.items() if not g['exact']]}; "
          f"W {None if not w4 else {k: g['pass'] for k, g in w4['gate_W'].items()}}; missing {missing}; "
          f"passing {J['variants_passing']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="frames")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--routines", default="3,4")
    ap.add_argument("--sandbox", default=str(SANDBOX))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", default=str(CTX_DEFAULT))
    ap.add_argument("--world-variants", default=",".join(WORLD_VARIANTS))
    a = ap.parse_args()
    variants = [v for v in a.variants.split(",") if v]
    routines = [int(r) for r in a.routines.split(",") if r]
    steps = a.step.split(",")
    bad = [v for v in variants if v not in VARIANTS]
    if bad or set(steps) - {"frames", "logit", "w4", "frontier", "cellgroups", "summary"} or set(routines) - set(FRAMES):
        raise SystemExit(f"unknown variants {bad}, steps {steps} or routines {routines}")
    sb = Path(a.sandbox).resolve()
    stamp(f"steps {steps}, variants {variants}, routines {routines}, sandbox {sb}")
    for s in steps:
        t = time.time()
        if s == "frames":
            step_frames(variants, routines, sb)
        elif s == "logit":
            step_logit(variants, sb, a.threads)
        elif s == "w4":
            step_w4(variants, sb, Path(a.ctx))
        elif s == "frontier":
            step_frontier(routines, sb)
        elif s == "cellgroups":
            step_cellgroups(variants, sb, [w for w in a.world_variants.split(",") if w])
        elif s == "summary":
            step_summary(variants, sb, Path(a.out_dir).resolve())
        stamp(f"step {s} done in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
