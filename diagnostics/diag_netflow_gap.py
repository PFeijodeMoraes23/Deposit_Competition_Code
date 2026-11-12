"""
diag_netflow_gap.py
===================
Four read-only diagnostics on the gap between the net flows of the K2 world and the data's
(claude/MAIN_REPLY_2026-11-05.md, "Update 11-11 19:44", Part A). Panel I2 (fa38794d), routine
(III) (E3, spec 12), B cells, 2016Q1-2024Q4. Section 7.3 reads the K2 world's awake money (2.1% of
deposits, 14% of it staying) against a median time-deposit net flow of 10% of the carried stock.

  a1  The world's own net flows next to the data's. The world is rebuilt with K2's own functions
      (diag_k2_netflow_simulation: read_menu, build, k1_rows, nonprice, e4_of, checks) on the inputs
      the promoted a216 run read. Gate G-A1: the rebuild reproduces Rout/k2_numbers.json to 1e-12
      (the kappa = 0 check against the prep, the pool's fate, the awake share, the simulated moving
      share, the build counts), for the delta source --delta names and for the other one. Each
      world's net flow per cell is NF = A - (1 - phi) omega carry, with A the awake inflow of K2's
      equation (20) through the one bracket for types 4 and 2, on the cells of
      kappa.k0.moving.median (K0's netflow_summary mask: NF known, Dep_lag > 0, carry > 0) that have
      a unit in the world (phi-hat known):
        W0      DGP stored, kappa 0.5, awake share 1 - phi-hat (raw)          the K2 world
        W1      the stored delta as it is, kappa 0, raw                        replication world
        W2      DGP stored, kappa 0.5, awake share 1 - min(phi-hat, PHI_CAP)   the shares' cap
        W3      the stored delta, kappa 0, capped
        W4      DGP stored, kappa 0.5, raw, a type-4 maturing share omega_jt   the statements' clock
        W0_k0, W2_k0, W4_k0  the same three at kappa 0
        Wfe, Wideal          K2's other two DGPs at kappa 0.5, raw
      In W4 the movable pool counts omega of the accrued type-4 stock and NF4 = A4 - (1 - phi)
      omega carry4; omega_jt follows K1's rule (kappa/k1_netflow_nls.py omega_lambda): the statement
      of year y (Rout/k0_omega.csv) applies to the quarters of year y + 1, the 2016 quarters take
      2016, and a conglomerate-year without a value takes the nearest year's B median of the file.
      Types 1, 2 and 5 keep the stored delta in every world (K2's docstring), so type 2's inflow is
      e_j2 times the same bracket and its carry is r2_acc D2.
      simulate_omega is a copy of k2.simulate with the awake share's phi and omega as arguments;
      with the stored phi and omega = 1 it returns k2.simulate's arrays bit for bit (checked).
  a2  Common against reallocation (data): the variance of y = NF / carry on K0's cells split into
      its market-quarter part and the part within a market-quarter across banks, and nested into
      quarter, market within quarter and bank within market-quarter; types 1, 2, 4; carry-weighted
      (the weighted mean of y in a market-quarter is its own aggregate flow rate), unweighted on y
      winsorised at p1/p99 within type, and carry-weighted on the winsorised y; all market-quarters
      and those where two or more B banks hold the type.
  a3  Rebooking: the share of top-decile cells (|NF| / carry above the type's p90, pooled years)
      with an institutional event at (j, m, t) against t - 1, next to the share among all cells:
        E_br_open/close/resize  own ESTBAN branches (PANEL_INTERMED/estban_own_branches.csv, keyed by
                                the as-reported code, canonicalised and summed) 0 -> >0, >0 -> 0, or
                                another positive count; missing = 0; not evaluated at 2016Q1 (the
                                file starts in 2016) nor where E_unfiled holds
        E_br_static             the same count from ESTBAN.csv with the static CNPJ map the deposits
                                are filed under (panel_deposits.build_estban_panel)
        E_cnpj                  the set of ESTBAN CNPJ roots with positive deposits (V400_401, V420,
                                V432) in (j, m) changes; E_cnpj_k: on the cell's own verbete only;
                                both from ESTBAN.csv (2016 on), not evaluated at 2016Q1
        E_congl                 j's member set in the IF.data List (Td = I, Situacao = A, codes
                                canonicalised) changes between the two quarters' snapshots, applied
                                to all of j's cells; E_congl_estban: members that file in ESTBAN
        E_unfiled               the cell or its lag holds an ESTBAN non-filing cell
                                (PANEL_INTERMED/estban_unfiled_cells.csv, any kind)
        E_2016q1, E_fallback, E_judicial  t = 2016Q1 (lag from the 2015Q4 side file); K0's
                                rate_fallback and judicial flags; E_tech is their union
        any_estban = E_br | E_cnpj | E_unfiled; any_event = any_estban | E_congl | E_tech
  a4  Feasibility under the accounting Dep = (1 - omega) carry + omega phi carry + A with A >= 0 the
      awake inflow (stayers included), so A = Dep - (1 - omega (1 - phi)) carry:
        L1  A <= 1e-6, raw phi, omega = 1 (the (14-B) truncation's own inequality)
        L2  A <= 1e-6, raw phi, omega_jt for type 4
        U1  A > (1 - min(phi, PHI_CAP)) M-hat (stored, floored)
        U2  A > (1 - phi) M-hat, raw phi
        U3  A > (1 - min(phi, PHI_CAP)) M-anchor, the market size before the floor (R Pop / access,
            R the constant M Pop^-1 access of the non-floor market-quarters)
        U4  A(omega) > (1 - phi) Pool, Pool = M-hat - sum_l (1 - omega_l) carry_l4 over the market's
            B units, raw phi
      shares of cells by type and pooled, carry-weighted, and within the top decile; then the
      reconciliation of L1 with demand.flows.trunc_share.E3 (prep cells, all types -> types 1, 2, 4
      -> K0 cells with phi and a carry -> K0's mask).

Gate G-A0 (before anything): market_panel.parquet md5 starts with PANEL_MD5, the demand_chars side
file and the E3 frame record it, the K0 netflow panel's market-quarter phi-hat quantiles equal the
frame's and k0_numbers.json's, and (when the world is built) K0's net flow rebuilt here equals
Rout/k0_netflow_panel.parquet.

Inputs (read only): processed/market_panel.parquet (pyarrow column subset, through
diag_k0_netflow_objects.load_panel), processed/demand_chars.parquet, DEMAND_PREP
demand_3_index_spec_12.parquet, BLP_RESULTS/logit/logit_delta_E3_spec_12.bin, the ext1 results json
k2 reads its alpha from, Rout/k0_netflow_panel.parquet, k0_omega.csv, k0_numbers.json,
k0_netflow_summary.csv, k2_numbers.json, xi_diagnostics.json, xi_cache/trunc_E3_spec_12.json,
PANEL_INTERMED/estban_own_branches.csv and estban_unfiled_cells.csv, shared/ESTBAN/ESTBAN.csv (column
subset), shared/IF_DATA/List, IBGE muni_mca_regions_2010_2024_panel.csv.
Outputs (--out-dir, default Rout/diag_netflow_gap/): a1_world_moving_share.csv,
a2_variance_split.csv, a3_rebooking.csv, a4_feasibility.csv, a4_trunc_reconciliation.csv,
netflow_gap_numbers.json (every number, the inputs' md5s, the gates), MD5SUMS.txt (md5 of each
output; text outputs are written with LF), and a1_world_cells.parquet with --write-cells.

Usage:  python diagnostics/diag_netflow_gap.py [--parts a1,a2,a3,a4] [--delta closed|binary]
                                               [--out-dir DIR] [--write-cells]
        --parts gate runs G-A0 and G-A1 only.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import glob
import hashlib
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import paths
from utils import demand_chars
from utils import conglomerates as cc
import diag_k0_netflow_objects as k0
import diag_k2_netflow_simulation as k2   # its import reads alpha from the ext1 json (read_alpha's guard)

PANEL_MD5 = "fa38794d"           # I2
k0.PANEL_MD5 = PANEL_MD5          # load_panel refuses another panel
ROUT = paths.rout_dir()
OUT_DEFAULT = ROUT / "diag_netflow_gap"
FRAME = k2.DEMAND                 # DEMAND_PREP/demand_3_index_spec_12.parquet
K0_PANEL = ROUT / "k0_netflow_panel.parquet"
K0_OMEGA = ROUT / "k0_omega.csv"
K0_NUMBERS = ROUT / "k0_numbers.json"
K0_SUMMARY = ROUT / "k0_netflow_summary.csv"
K2_NUMBERS = ROUT / "k2_numbers.json"
XI_JSON = ROUT / "xi_diagnostics.json"
XI_TRUNC = ROUT / "xi_cache" / "trunc_E3_spec_12.json"
BRANCHES = paths.PROCESSED / "PANEL_INTERMED" / "estban_own_branches.csv"
UNFILED = paths.PROCESSED / "PANEL_INTERMED" / "estban_unfiled_cells.csv"
ESTBAN = paths.ESTBAN_CSV
MCA_XWALK = paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"
PANEL_CSV = paths.PROCESSED / "market_panel.csv"

PHI_CAP = 0.99                    # sleep_demand_prep_link.PHI_CAP of the frames; checked on _1mphi
SHARE_CAP = 0.95                  # sleep_demand_prep_link.TOTAL_SHARE_CAP; checked on the share sums
KAPPA_BASE = k2.KAPPA_BASE
TOL_GATE = 1e-12
QS = (0.25, 0.50, 0.75, 0.90)
TYPES = (1, 2, 4)
Q2016 = k0.qi(2016, 1)
DEP_VERBETE = {1: "V400_401", 2: "V420", 4: "V432"}   # panel_deposits.build_estban_panel: a1, a2, a4

WORLDS = [  # id, DGP ('delta': the stored delta as it is), kappa, awake share's phi, type-4 omega
    ("W0", "stored", KAPPA_BASE, "raw", "one"),
    ("W0_k0", "stored", 0.0, "raw", "one"),
    ("W1", "delta", 0.0, "raw", "one"),
    ("W2", "stored", KAPPA_BASE, "capped", "one"),
    ("W2_k0", "stored", 0.0, "capped", "one"),
    ("W3", "delta", 0.0, "capped", "one"),
    ("W4", "stored", KAPPA_BASE, "raw", "statements"),
    ("W4_k0", "stored", 0.0, "raw", "statements"),
    ("Wfe", "fe", KAPPA_BASE, "raw", "one"),
    ("Wideal", "ideal", KAPPA_BASE, "raw", "one"),
]

stamp = k0.stamp


def md5_raw(p):
    return k0.md5(p)


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)),
                          encoding="utf-8", newline="\n")


def qd(x, qs=QS):
    x = np.asarray(x, float)
    return {f"p{int(round(q * 100)):02d}": (float(np.quantile(x, q)) if len(x) else float("nan")) for q in qs}


# ------------------------------------------------------------------------------------------ gate
def max_abs_diff(got, want):
    """Largest absolute difference between two json-like structures; inf on a missing key, a length
    or type mismatch, or unequal strings."""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return float("inf")
        return max([max_abs_diff(got.get(k), v) for k, v in want.items()] or [0.0])
    if isinstance(want, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(got) != len(want):
            return float("inf")
        return max([max_abs_diff(g, w) for g, w in zip(got, want)] or [0.0])
    if isinstance(want, str) or want is None:
        return 0.0 if got == want else float("inf")
    try:
        g, w = float(got), float(want)
    except (TypeError, ValueError):
        return float("inf")
    if np.isnan(w):
        return 0.0 if np.isnan(g) else float("inf")
    return abs(g - w)


def gate_a0(N):
    """G-A0: the panel and the frame are I2's; the K0 netflow panel's phi-hat is the frame's."""
    got = md5_raw(k0.PANEL)
    side = (pq.read_schema(demand_chars.chars_path()).metadata or {}).get(demand_chars.PANEL_MD5_KEY, b"").decode()
    frame = (pq.read_schema(FRAME).metadata or {}).get(demand_chars.PANEL_MD5_KEY, b"").decode()
    fr = pq.read_table(FRAME, columns=["is_B", "mca_code", "time_id", "phi_mt"]).to_pandas()
    fr = fr[fr["is_B"]]
    mq_frame = fr.groupby(["mca_code", "time_id"])["phi_mt"].first()
    kp = pq.read_table(K0_PANEL, columns=["m", "t", "phi_mt"]).to_pandas()
    mq_k0 = kp.drop_duplicates(["m", "t"])["phi_mt"]
    want = json.loads(K0_NUMBERS.read_text(encoding="utf-8"))["a_phi_quantiles"]
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    q_frame = {f"p{int(q * 100):02d}": float(mq_frame.quantile(q)) for q in qs}
    q_k0 = {f"p{int(q * 100):02d}": float(mq_k0.quantile(q)) for q in qs}
    g = {"panel_md5": got, "side_file_panel_md5": side, "frame_panel_md5": frame,
         "phi_mq_quantiles_frame": q_frame, "phi_mq_quantiles_k0_panel": q_k0, "phi_mq_quantiles_k0_numbers": want,
         "max_abs_frame_vs_k0_numbers": max_abs_diff(q_frame, want),
         "max_abs_k0_panel_vs_k0_numbers": max_abs_diff(q_k0, want)}
    fails = []
    if not got.startswith(PANEL_MD5):
        fails.append(f"panel md5 {got[:8]} is not {PANEL_MD5}")
    if side != got:
        fails.append(f"demand_chars side file built from {side[:8]}")
    if not frame.startswith(PANEL_MD5):
        fails.append(f"frame built on {frame[:8]}")
    if g["max_abs_frame_vs_k0_numbers"] > TOL_GATE or g["max_abs_k0_panel_vs_k0_numbers"] > TOL_GATE:
        fails.append("market-quarter phi-hat quantiles differ between the frame, the K0 panel and k0_numbers.json")
    g["pass"] = not fails
    g["fails"] = fails
    N["gate_A0"] = g
    stamp(f"G-A0 {'PASS' if g['pass'] else 'FAIL'}: panel {got[:8]}, side {side[:8]}, frame {frame[:8]}; "
          f"phi quantiles frame vs k0_numbers {g['max_abs_frame_vs_k0_numbers']:.1e}")
    return g["pass"]


def read_menu_as(source):
    """k2.read_menu with its delta source fixed: 'binary' reads the stored logit binary (the path
    K2 takes for this frame now), 'closed' takes the closed form of blp_logit.jl berry_inversion
    (the path the promoted a216 run recorded). The choice is made through the module attribute
    k2.DELTA_BIN_PANEL_MD5, restored afterwards."""
    keep = k2.DELTA_BIN_PANEL_MD5
    try:
        if source == "closed":
            k2.DELTA_BIN_PANEL_MD5 = "not-a-panel-md5"
        menu, rec = k2.read_menu()
    finally:
        k2.DELTA_BIN_PANEL_MD5 = keep
    rec = dict(rec, delta_binary_panel_md5=keep)
    return menu, rec


def awake_share(U, MT, phi):
    W = U["W"].to_numpy()
    w = W > 0
    a = 1 - phi[U["mt"].to_numpy()]
    return float(np.sum((a * W)[w]) / np.sum(W[w])), int(w.sum())


def build_world(df, flags, mk, source):
    menu, rec = read_menu_as(source)
    U, MT, SD, SD4, info = k2.build(df, flags, mk, menu)
    K, kinfo = k2.k1_rows(U, MT, df, flags, mk)
    Nk = {**rec, **info, **kinfo}
    ctx = {"U": U, "MT": MT, "SD": SD, "SD4": SD4, "K": K, "N": Nk}
    k2.checks(ctx)
    aw, n_w = awake_share(U, MT, MT["phi_mt"].to_numpy())
    Nk["pool_fate_kappa_base"]["awake_money_share_of_deposits"] = aw
    Nk["awake_units_W_positive"] = n_w
    dinfo = {}
    n, ks = k2.nonprice(U, "stored", dinfo)
    e4 = k2.e4_of(U, n, ks)
    dinfo["K1_rows_on_menu"] = int((e4[K["u"].to_numpy()] > 0).sum())
    Nk["dgp_stored"] = dinfo
    return ctx, menu, {"n": n, "ks": ks, "e4": e4}


GATE_PLAN = {"check_kappa0_awake_over_prep_dep_act": None, "pool_fate_kappa_base": ["stay_at_own_bank",
             "awake_money_share_of_deposits"], "simulated_moving_share_kappa_base": None}
GATE_ALSO = ["check_kappa0_n", "pool_fate_kappa_base", "simulated_entry_composition_kappa_base",
             "simulated_y_quantiles", "units", "units_dropped_no_phi", "market_quarters",
             "pools_with_stock_but_no_lagged_rate", "D_products_per_quarter", "K0_type4_netflow_cells", "K1_rows",
             "dropped_no_own_savings_or_fallback_or_gap", "dropped_negative_outside_pool", "prep_rows",
             "frame_panel_md5", "dgp_stored"]


def gate_a1(Nk, want):
    """G-A1 on one build: the plan's keys and every other recorded number of the K2 world."""
    out = {"plan_keys": {}, "other_keys": {}}
    for key, sub in GATE_PLAN.items():
        w = want.get(key)
        if sub:
            w = {s: w[s] for s in sub}
            g = {s: Nk.get(key, {}).get(s) for s in sub}
        else:
            g = Nk.get(key)
        out["plan_keys"][key] = max_abs_diff(g, w)
    for key in GATE_ALSO:
        w = want.get(key)
        if key == "pool_fate_kappa_base":
            w = {s: v for s, v in w.items() if not s.endswith("_definition")}
        out["other_keys"][key] = max_abs_diff(Nk.get(key), w)
    vals = list(out["plan_keys"].values()) + list(out["other_keys"].values())
    out["max_abs"] = float(max(vals))
    out["pass"] = bool(out["max_abs"] <= TOL_GATE)
    return out


# ------------------------------------------------------------------------------------- the worlds
def simulate_omega(U, MT, SD, SD4, e4, kappa, phi, omega=None, h=1.0):
    """k2.simulate with the awake share's phi and a type-4 maturing share as arguments.

    phi    per market-quarter, in MT's order: the phi whose complement is awake.
    omega  per unit, or None (omega = 1): the movable pool counts omega of the accrued type-4 stock,
           W - (1 - omega) carry4, and NF4 = A4 - (1 - phi) omega carry4; the locked rest stays put.
    Returns (A4, NF4, A2, NF2): the awake inflows of (20) into (j, 4) and (j, 2), both through the
    same bracket, and the net flows; NF2 = A2 - (1 - phi) carry2, carry2 = r2_acc D2. With
    phi = MT['phi_mt'] and omega None, (A4, NF4) is k2.simulate's (A, NF) bit for bit."""
    ek = np.exp(-kappa)
    mt = U["mt"].to_numpy()
    nmt = len(MT)
    O = MT["O"].to_numpy()
    W = U["W"].to_numpy()
    carry4 = U["carry4"].to_numpy()
    if omega is not None:
        W = W - (1 - omega) * carry4
    E = U["e1"].to_numpy() + U["e2"].to_numpy() + U["e5"].to_numpy() + e4
    S = np.bincount(mt, weights=E, minlength=nmt) + SD[k2.MT_t(MT, U)]
    S4 = np.bincount(mt, weights=e4, minlength=nmt) + SD4[k2.MT_t(MT, U)]
    Dj = 1 + E + ek * ((S[mt] - E) - (1 - h) * (S4[mt] - e4))
    D0 = 1 + S
    Wd = W / Dj
    T = np.bincount(mt, weights=Wd, minlength=nmt)
    awake = 1 - phi[mt]
    bracket = Wd + h * ek * (T[mt] - Wd) + O[mt] / D0[mt]
    A4 = awake * e4 * bracket
    NF4 = A4 - awake * carry4 if omega is None else A4 - awake * omega * carry4
    A2 = awake * U["e2"].to_numpy() * bracket
    NF2 = A2 - awake * (U["r2_acc"].to_numpy() * U["D2"].to_numpy())
    return A4, NF4, A2, NF2


def omega_rule():
    """omega by conglomerate-year from K0 (e), with K1's timing and fill (kappa/k1_netflow_nls.py
    clock_table / omega_lambda): (table, fill by year)."""
    W = pd.read_csv(K0_OMEGA, dtype={"conglomerate": str})
    med = W[W["is_B"] == 1].groupby("year")["omega"].median()
    tab = {(j, int(y)): float(o) for j, y, o in zip(W["conglomerate"], W["year"], W["omega"])}
    fill = {y: float(med.loc[min(med.index, key=lambda z: (abs(z - y), z))]) for y in range(2016, 2025)}
    return tab, fill


def omega_of(j, qidx, tab, fill):
    y = np.clip(np.asarray(qidx) // 4 - 1, 2016, 2024)
    return np.array([tab.get((jj, int(yy)), fill[int(yy)]) for jj, yy in zip(j, y)], float)


def a1_cells(L, U, K):
    """K0's mask cells of types 4 and 2 (kappa.k0.moving.median's cells) with their world unit."""
    ok = L["NF"].notna() & (L["Dep_lag"] > 0) & (L["carry"] > 0)
    C = L.loc[ok & L["k"].isin([4, 2]), ["j", "m", "qi", "t", "year", "k", "Dep", "Dep_lag", "carry", "NF",
                                        "phi_mt", "rate_fallback", "judicial"]].copy()
    n_all = {int(k): int((C["k"] == k).sum()) for k in (4, 2)}
    Uk = U[["j", "m", "qi"]].reset_index().rename(columns={"index": "u"})
    C = C.merge(Uk, on=["j", "m", "qi"], how="left")
    C = C[C["u"].notna()].copy()
    C["u"] = C["u"].astype(int)
    u = C["u"].to_numpy()
    carry_u = np.where(C["k"].to_numpy() == 4, U["carry4"].to_numpy()[u],
                       U["r2_acc"].to_numpy()[u] * U["D2"].to_numpy()[u])
    gap = float(np.max(np.abs(C["carry"].to_numpy() - carry_u)))
    on4 = U["delta4"].notna().to_numpy()[u]
    on2 = (U["e2"].to_numpy()[u] > 0)
    C["on_menu"] = np.where(C["k"].to_numpy() == 4, on4, on2)
    C["K1"] = (C["k"].to_numpy() == 4) & np.isin(u, K["u"].to_numpy())
    info = {"cells_key": n_all, "cells_with_unit": {int(k): int((C["k"] == k).sum()) for k in (4, 2)},
            "carry_k0_vs_world_max_abs": gap, "carry_scale_max": float(C["carry"].max())}
    return C, info


def a1_run(ctx, extra, L, N, write_cells, out):
    U, MT, SD, SD4, K = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["K"]
    C, cinfo = a1_cells(L, U, K)
    if cinfo["carry_k0_vs_world_max_abs"] > 1e-6 * max(cinfo["carry_scale_max"], 1.0):
        raise SystemExit(f"K0's carry and the world's pools differ by {cinfo['carry_k0_vs_world_max_abs']:g}")
    phi_raw = MT["phi_mt"].to_numpy()
    phis = {"raw": phi_raw, "capped": np.minimum(phi_raw, PHI_CAP)}
    tab, fill = omega_rule()
    om_u = omega_of(U["j"].to_numpy(), U["qi"].to_numpy(), tab, fill)
    e4s = {"stored": extra["e4"], "delta": np.where(U["delta4"].notna(), np.exp(U["delta4"].to_numpy()), 0.0)}
    dgp_info = {}
    for dgp in ("fe", "ideal"):
        info = {}
        n, ks = k2.nonprice(U, dgp, info)
        e4s[dgp] = k2.e4_of(U, n, ks)
        dgp_info[dgp] = info
    # simulate_omega against k2.simulate: the K2 world (W0) and the kappa = 0 replication world (W1)
    chk = {}
    for name, e4, kap in (("W0", e4s["stored"], KAPPA_BASE), ("W1", e4s["delta"], 0.0)):
        A_ref, NF_ref = k2.simulate(U, MT, SD, SD4, e4, kap)
        A4, NF4, _, _ = simulate_omega(U, MT, SD, SD4, e4, kap, phi_raw)
        chk[name] = {"A_max_abs": float(np.max(np.abs(A4 - A_ref))), "NF_max_abs": float(np.max(np.abs(NF4 - NF_ref)))}
    A4, NF4, _, _ = simulate_omega(U, MT, SD, SD4, e4s["stored"], KAPPA_BASE, phi_raw, omega=np.ones(len(U)))
    A_ref, NF_ref = k2.simulate(U, MT, SD, SD4, e4s["stored"], KAPPA_BASE)
    chk["W0_omega_ones"] = {"A_max_abs": float(np.max(np.abs(A4 - A_ref))),
                            "NF_max_abs": float(np.max(np.abs(NF4 - NF_ref)))}
    bitwise = all(v["A_max_abs"] == 0.0 and v["NF_max_abs"] == 0.0 for k, v in chk.items() if k != "W0_omega_ones")
    stamp(f"simulate_omega vs k2.simulate: {json.dumps(chk)}")
    if not bitwise:
        raise SystemExit("simulate_omega does not reproduce k2.simulate")
    nf = {}
    agg = {}
    u = C["u"].to_numpy()
    k_arr = C["k"].to_numpy()
    W_all = U["W"].to_numpy()
    wpos = W_all > 0
    for wid, dgp, kap, ph, om in WORLDS:
        A4, NF4, A2, NF2 = simulate_omega(U, MT, SD, SD4, e4s[dgp], kap, phis[ph],
                                          omega=om_u if om == "statements" else None)
        nf[wid] = np.where(k_arr == 4, NF4[u], NF2[u])
        aw, _ = awake_share(U, MT, phis[ph])
        Wm = W_all - ((1 - om_u) * U["carry4"].to_numpy() if om == "statements" else 0.0)
        agg[wid] = {"dgp": dgp, "kappa": kap, "phi": ph, "omega": om,
                    "awake_money_share_of_deposits": aw,
                    "movable_share_of_deposits": float(Wm[wpos].sum() / W_all[wpos].sum()),
                    "type4_cells_zero_inflow_share": float(np.mean(A4[u][k_arr == 4] <= 0)),
                    "type2_cells_zero_inflow_share": float(np.mean(A2[u][k_arr == 2] <= 0))}
    rows = []
    phi = C["phi_mt"].to_numpy()
    carry = C["carry"].to_numpy()
    nfd = C["NF"].to_numpy()
    for k in (4, 2):
        base = k_arr == k
        on = C["on_menu"].to_numpy()
        splits = {"all": base, "on_menu": base & on, "off_menu": base & ~on, "phi_le_cap": base & (phi <= PHI_CAP),
                  "phi_gt_cap": base & (phi > PHI_CAP), "phi_eq_1": base & (phi >= 1.0),
                  "on_menu_phi_le_cap": base & on & (phi <= PHI_CAP), "on_menu_phi_gt_cap": base & on & (phi > PHI_CAP)}
        if k == 4:
            splits["K1_rows"] = base & C["K1"].to_numpy()
        for sname, s in splits.items():
            ms_d = np.abs(nfd[s]) / carry[s]
            dq = qd(ms_d)
            for wid, dgp, kap, ph, om in WORLDS:
                ms_w = np.abs(nf[wid][s]) / carry[s]
                wq = qd(ms_w)
                rows.append({"world": wid, "dgp": dgp, "kappa": kap, "phi": ph, "omega": om, "k": k, "split": sname,
                             "n_cells": int(s.sum()), "share_of_type_cells": float(s.sum() / base.sum()),
                             **{f"data_{q}": v for q, v in dq.items()},
                             "data_stock_weighted": float(np.abs(nfd[s]).sum() / carry[s].sum()) if s.any() else np.nan,
                             **{f"world_{q}": v for q, v in wq.items()},
                             "world_stock_weighted": float(np.abs(nf[wid][s]).sum() / carry[s].sum()) if s.any() else np.nan,
                             "world_minus_data_abs_over_carry_p50": float(np.median(np.abs(nf[wid][s] - nfd[s]) / carry[s]))
                             if s.any() else np.nan,
                             "world_net_stock_weighted": float(nf[wid][s].sum() / carry[s].sum()) if s.any() else np.nan,
                             "data_net_stock_weighted": float(nfd[s].sum() / carry[s].sum()) if s.any() else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(out / "a1_world_moving_share.csv", index=False, lineterminator="\n")
    data_med = {int(k): float(np.median(np.abs(L.loc[(L["k"] == k) & L["NF"].notna() & (L["Dep_lag"] > 0) & (L["carry"] > 0), "NF"])
                                     / L.loc[(L["k"] == k) & L["NF"].notna() & (L["Dep_lag"] > 0) & (L["carry"] > 0), "carry"]))
                for k in (4, 2)}
    want = pd.read_csv(K0_SUMMARY, dtype={"year": str})
    want = want[want["year"] == "all"].set_index("k")["moving_median"]
    N["a1"] = {"cells": cinfo, "simulate_omega_check": chk, "worlds": agg, "dgp_info": dgp_info,
               "omega_rule": "statement of year y -> quarters of y + 1, 2016 quarters take 2016; missing "
                             "conglomerate-years take the nearest year's B median of k0_omega.csv (K1's omega_lambda)",
               "omega_units_mean": float(om_u.mean()),
               "omega_units_type4_stock_weighted": float(np.average(om_u, weights=np.maximum(U["carry4"].to_numpy(), 0) + 1e-300)),
               "data_key_median_check": {int(k): {"here": data_med[k], "k0_netflow_summary": float(want.loc[k]),
                                                  "abs_diff": abs(data_med[k] - float(want.loc[k]))} for k in (4, 2)}}
    if write_cells:
        cells = C[["j", "m", "t", "k", "carry", "NF", "phi_mt", "on_menu", "K1"]].copy()
        for wid, *_ in WORLDS:
            cells[f"NF_{wid}"] = nf[wid]
        cells.to_parquet(out / "a1_world_cells.parquet", index=False)
    stamp("a1 done: " + "; ".join(f"{r['world']} k{r['k']} p50 {r['world_p50']:.4f} (data {r['data_p50']:.4f})"
                                  for _, r in R[R["split"] == "all"].iterrows()))
    return R, om_u


# ------------------------------------------------------------------------------------------- a2
def wsplit(y, w, mq, q):
    """Weighted variance split of y: quarter, market within quarter, bank within market-quarter."""
    W = w.sum()
    ybar = (w * y).sum() / W

    def gm(c):
        sw = np.bincount(c, weights=w)
        return (np.bincount(c, weights=w * y) / sw)[c]
    ymq, yq = gm(mq), gm(q)
    tot = float((w * (y - ybar) ** 2).sum())
    ss_q = float((w * (yq - ybar) ** 2).sum())
    ss_m = float((w * (ymq - yq) ** 2).sum())
    ss_b = float((w * (y - ymq) ** 2).sum())
    return {"var_total": tot / W, "mean_y": float(ybar),
            "share_between_mq": (ss_q + ss_m) / tot, "share_within_mq": ss_b / tot,
            "share_quarter": ss_q / tot, "share_market_within_quarter": ss_m / tot,
            "share_bank_within_mq": ss_b / tot, "identity_residual": (ss_q + ss_m + ss_b - tot) / tot}


def load_k0_panel():
    L = pq.read_table(K0_PANEL).to_pandas()
    L["carry"] = (1 + L["rate_lag"]) * L["Dep_lag"]
    L["qi"] = k0.qi(L["year"], L["quarter"])
    L["mask"] = L["NF"].notna() & (L["Dep_lag"] > 0) & (L["carry"] > 0)
    L["ms"] = L["NF"].abs() / L["carry"]
    return L


def top_decile(L):
    """|NF| / carry above the type's p90 over K0's mask cells, pooled years."""
    p90 = {k: float(L.loc[L["mask"] & (L["k"] == k), "ms"].quantile(0.9)) for k in TYPES}
    top = L["mask"] & (L["ms"] > L["k"].map(p90))
    return top, p90


def a2_run(L, N, out):
    rows = []
    for k in TYPES:
        g = L[L["mask"] & (L["k"] == k)].copy()
        g["y"] = g["NF"] / g["carry"]
        lo, hi = g["y"].quantile(0.01), g["y"].quantile(0.99)
        g["yw"] = g["y"].clip(lo, hi)
        g["mqk"] = g["m"] + "|" + g["t"]
        nb = g.groupby("mqk")["j"].transform("size")
        n_single = int(g.loc[nb == 1, "mqk"].nunique())
        for sname, s in (("all", np.ones(len(g), bool)), ("mq_two_or_more_banks", (nb >= 2).to_numpy())):
            h = g[s]
            mq = pd.factorize(h["mqk"])[0]
            q = pd.factorize(h["t"])[0]
            for wname, y, w in (("carry", h["y"], h["carry"]), ("unweighted_winsor", h["yw"], np.ones(len(h))),
                                ("carry_winsor", h["yw"], h["carry"])):
                r = wsplit(np.asarray(y, float), np.asarray(w, float), mq, q)
                rows.append({"k": k, "weighting": wname, "sample": sname, "n_cells": int(len(h)),
                             "n_mq": int(mq.max() + 1), "n_quarters": int(q.max() + 1),
                             "n_single_bank_mq_type": n_single, "winsor_p01": float(lo), "winsor_p99": float(hi), **r})
    R = pd.DataFrame(rows)
    R.to_csv(out / "a2_variance_split.csv", index=False, lineterminator="\n")
    net = json.loads(K0_NUMBERS.read_text(encoding="utf-8"))["a_netting"]
    N["a2"] = {"k0_netting_all_years": {k: v["all"] for k, v in net.items()}}
    stamp("a2 done: " + "; ".join(f"k{r['k']} {r['weighting']} {r['sample']}: within {r['share_within_mq']:.3f}"
                                  for _, r in R.iterrows() if r["sample"] == "all"))
    return R


# ------------------------------------------------------------------------------------------- a3
def mca_map():
    xw = pd.read_csv(MCA_XWALK, usecols=["municipality_code", "year", "mca_code"])
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    return dict(zip(xw["municipality_code"].astype(int), xw["mca_code"].astype(str)))


def static_codes(cnpj_int, cmap):
    """panel_deposits.build_estban_panel's code for an ESTBAN CNPJ root: the static map's canonical
    code, else 'CNPJ_<root>'."""
    u = pd.unique(cnpj_int)
    look = {c: (cmap[c][0] if c in cmap and cmap[c][0] is not None else f"CNPJ_{c}") for c in u}
    return pd.Series(cnpj_int).map(look).to_numpy()


def read_estban(mca):
    """Quarter-end rows of ESTBAN.csv (column subset, chunked), with the static code and the MCA;
    also every CNPJ root seen in any month and the CODMUN -> CODMUN_IBGE pairs."""
    cols = ["CNPJ", "CODMUN", "CODMUN_IBGE", "YEAR", "MONTH", "AGEN_PROCESSADAS"] + list(DEP_VERBETE.values())
    parts, roots, pairs = [], set(), []
    for ch in pd.read_csv(ESTBAN, usecols=cols, dtype={"CNPJ": str, "CODMUN": str}, encoding="latin-1",
                          chunksize=1_000_000, low_memory=False):
        ch["CNPJ"] = ch["CNPJ"].str.strip().str.zfill(8)
        roots |= set(ch["CNPJ"].dropna().unique())
        ch["CODMUN"] = pd.to_numeric(ch["CODMUN"], errors="coerce")
        ch["CODMUN_IBGE"] = pd.to_numeric(ch["CODMUN_IBGE"], errors="coerce")
        pairs.append(ch[["CODMUN", "CODMUN_IBGE"]].dropna().drop_duplicates())
        parts.append(ch[ch["MONTH"].isin([3, 6, 9, 12])])
    est = pd.concat(parts, ignore_index=True)
    est = est.dropna(subset=["CNPJ"])
    pairs = pd.concat(pairs).drop_duplicates()
    n_multi = int(pairs.groupby("CODMUN")["CODMUN_IBGE"].nunique().gt(1).sum())
    codmun = pairs.drop_duplicates("CODMUN", keep="last")
    codmun = dict(zip(codmun["CODMUN"].astype(int), codmun["CODMUN_IBGE"].astype(int)))
    est = est.dropna(subset=["CODMUN_IBGE"])
    est = est[est["CODMUN_IBGE"] != 0].copy()
    est["m"] = est["CODMUN_IBGE"].astype(int).map(mca)
    n_nomca = int(est["m"].isna().sum())
    est = est[est["m"].notna()].copy()
    est["qi"] = k0.qi(est["YEAR"], est["MONTH"] // 3)
    est["cnpj_int"] = est["CNPJ"].astype(int)
    for v in DEP_VERBETE.values():
        est[v] = pd.to_numeric(est[v], errors="coerce")
    est["AGEN_PROCESSADAS"] = pd.to_numeric(est["AGEN_PROCESSADAS"], errors="coerce").fillna(0.0)
    info = {"rows_quarter_end": int(len(est)), "rows_without_mca": n_nomca, "years": [int(est["YEAR"].min()), int(est["YEAR"].max())],
            "cnpj_roots_any_month": len(roots), "codmun_with_several_ibge": n_multi}
    return est, roots, codmun, info


def lag_values(cells, tab, col, fill):
    """tab's `col` at (j, m, qi) and at (j, m, qi - 1) for every cell (missing = fill)."""
    cur = cells[["j", "m", "qi"]].merge(tab, on=["j", "m", "qi"], how="left")[col]
    prev = cells[["j", "m", "qi"]].assign(qi=cells["qi"] - 1).merge(tab, on=["j", "m", "qi"], how="left")[col]
    if isinstance(fill, tuple):
        cur = cur.apply(lambda v: v if isinstance(v, tuple) else fill)
        prev = prev.apply(lambda v: v if isinstance(v, tuple) else fill)
        return cur.to_numpy(), prev.to_numpy()
    return cur.fillna(fill).to_numpy(), prev.fillna(fill).to_numpy()


def branch_events(b_t, b_l):
    return {"open": (b_l == 0) & (b_t > 0), "close": (b_l > 0) & (b_t == 0),
            "resize": (b_l > 0) & (b_t > 0) & (b_l != b_t)}


def a3_events(L, N):
    """Event flags on K0's mask cells of types 1, 2, 4."""
    cells = L.loc[L["mask"] & L["k"].isin(TYPES), ["j", "m", "qi", "t", "year", "k", "NF", "carry", "ms",
                                                   "rate_fallback", "judicial"]].reset_index(drop=True)
    info = {}
    mca = mca_map()
    cmap = cc.build_cnpj_conglomerate_map()
    est, roots, codmun, info["estban"] = read_estban(mca)
    stamp(f"a3: ESTBAN quarter-end rows {len(est):,}")
    not2016 = (cells["qi"] > Q2016).to_numpy()
    # E_unfiled: the cell or its lag holds a non-filing cell
    uf = pd.read_csv(UNFILED)
    uf["CODMUN_IBGE_map"] = uf["CODMUN"].astype(int).map(codmun)
    ibge = uf["CODMUN_IBGE_map"].fillna(pd.to_numeric(uf["CODMUN_IBGE"], errors="coerce"))
    uf["m"] = ibge.map(lambda v: mca.get(int(v)) if pd.notna(v) else None)
    uf["j"] = static_codes(uf["CNPJ"].astype(int).to_numpy(), cmap)
    uf["qi"] = k0.qi(uf["YEAR"], uf["MONTH"] // 3)
    info["unfiled"] = {"rows": int(len(uf)), "rows_mapped": int(uf["m"].notna().sum()),
                       "kinds": uf["kind"].value_counts().to_dict(),
                       "months_not_quarter_end": int((~uf["MONTH"].isin([3, 6, 9, 12])).sum())}
    ufs = uf.loc[uf["m"].notna(), ["j", "m", "qi"]].drop_duplicates().assign(x=1.0)
    u_t, u_l = lag_values(cells, ufs, "x", 0.0)
    E = {"E_unfiled": (u_t > 0) | (u_l > 0)}
    # E_br: the as-reported sidecar, canonicalised and summed
    br = pd.read_csv(BRANCHES, dtype={"CodConglomeradoPrudencial": str, "mca_code": str})
    br["j"] = cc.canonicalize_codes(br["CodConglomeradoPrudencial"])
    br["qi"] = k0.qi(br["year"], br["quarter"])
    br = br.rename(columns={"mca_code": "m"}).groupby(["j", "m", "qi"], as_index=False)["own_br"].sum()
    info["branches_sidecar"] = {"rows_canonical": int(len(br)), "first_quarter": k0.tid([br["qi"].min()])[0]}
    b_t, b_l = lag_values(cells, br, "own_br", 0.0)
    ev = branch_events(b_t, b_l)
    keep = not2016 & ~E["E_unfiled"]
    for k_, v in ev.items():
        E[f"E_br_{k_}"] = v & keep
    E["E_br"] = E["E_br_open"] | E["E_br_close"] | E["E_br_resize"]
    # E_br_static: branches under the static map the deposits are filed under
    est["j"] = static_codes(est["cnpj_int"].to_numpy(), cmap)
    bs = est.groupby(["j", "m", "qi"], as_index=False)["AGEN_PROCESSADAS"].sum()
    s_t, s_l = lag_values(cells, bs, "AGEN_PROCESSADAS", 0.0)
    evs = branch_events(s_t, s_l)
    E["E_br_static"] = (evs["open"] | evs["close"] | evs["resize"]) & keep
    # E_cnpj: the set of CNPJ roots with positive deposits; any verbete, and the cell's own
    dep_any = est[list(DEP_VERBETE.values())].fillna(0).sum(axis=1) > 0
    sets = (est.loc[dep_any, ["j", "m", "qi", "cnpj_int"]].drop_duplicates().sort_values(["j", "m", "qi", "cnpj_int"])
            .groupby(["j", "m", "qi"])["cnpj_int"].agg(tuple).rename("s").reset_index())
    c_t, c_l = lag_values(cells, sets, "s", ())
    E["E_cnpj"] = np.array([a != b for a, b in zip(c_t, c_l)]) & not2016
    ek = np.zeros(len(cells), bool)
    for k, v in DEP_VERBETE.items():
        sk = (est.loc[est[v].fillna(0) > 0, ["j", "m", "qi", "cnpj_int"]].drop_duplicates()
              .sort_values(["j", "m", "qi", "cnpj_int"]).groupby(["j", "m", "qi"])["cnpj_int"].agg(tuple)
              .rename("s").reset_index())
        rows = (cells["k"] == k).to_numpy()
        a, b = lag_values(cells[rows], sk, "s", ())
        ek[rows] = np.array([x != y for x, y in zip(a, b)])
    E["E_cnpj_k"] = ek & not2016
    # E_congl: the List member set of j (canonical codes), each quarter's snapshot against the last
    lm = cc.build_cnpj_to_congl().dropna(subset=["AnoMes_list", "CodConglomeradoPrudencial"]).copy()
    lm["AnoMes_list"] = lm["AnoMes_list"].astype(int)
    lm = lm[(lm["AnoMes_list"] % 100).isin([3, 6, 9, 12])]
    lm["j"] = cc.canonicalize_codes(lm["CodConglomeradoPrudencial"])
    lm["qi"] = k0.qi(lm["AnoMes_list"] // 100, (lm["AnoMes_list"] % 100) // 3)
    snaps = set(lm["qi"].unique())
    info["congl"] = {"snapshots": len(snaps), "first": k0.tid([min(snaps)])[0], "last": k0.tid([max(snaps)])[0]}
    have = np.array([(q in snaps) and (q - 1 in snaps) for q in cells["qi"]])
    for name, sub in (("E_congl", lm), ("E_congl_estban", lm[lm["CNPJ"].isin(roots)])):
        ms = (sub[["j", "qi", "CNPJ"]].drop_duplicates().sort_values(["j", "qi", "CNPJ"])
              .groupby(["j", "qi"])["CNPJ"].agg(tuple).rename("s").reset_index())
        cur = cells[["j", "qi"]].merge(ms, on=["j", "qi"], how="left")["s"]
        prev = cells[["j", "qi"]].assign(qi=cells["qi"] - 1).merge(ms, on=["j", "qi"], how="left")["s"]
        cur = [v if isinstance(v, tuple) else () for v in cur]
        prev = [v if isinstance(v, tuple) else () for v in prev]
        E[name] = np.array([a != b for a, b in zip(cur, prev)]) & have
    # E_tech
    E["E_2016q1"] = ~not2016
    E["E_fallback"] = (cells["rate_fallback"] == 1).to_numpy()
    E["E_judicial"] = (cells["judicial"] == 1).to_numpy()
    E["E_tech"] = E["E_2016q1"] | E["E_fallback"] | E["E_judicial"]
    E["any_estban"] = E["E_br"] | E["E_cnpj"] | E["E_unfiled"]
    E["any_event"] = E["any_estban"] | E["E_congl"] | E["E_tech"]
    N["a3_info"] = info
    return cells, E


def a3_table(cells, E, top, N, out):
    rows = []
    t_all = top
    for k in TYPES:
        base = (cells["k"] == k).to_numpy()
        tp = base & t_all
        nf = cells["NF"].abs().to_numpy()
        for yname, ysel in [("all", np.ones(len(cells), bool))] + [(str(y), (cells["year"] == y).to_numpy())
                                                                   for y in range(2016, 2025)]:
            b, t = base & ysel, tp & ysel
            evs = E.items() if yname == "all" else [(n, E[n]) for n in ("any_event", "any_estban")]
            for name, e in evs:
                sa = float(e[b].mean()) if b.any() else np.nan
                st = float(e[t].mean()) if t.any() else np.nan
                rows.append({"k": k, "year": yname, "event": name, "n_cells": int(b.sum()), "n_top": int(t.sum()),
                             "n_event_all": int(e[b].sum()), "n_event_top": int(e[t].sum()),
                             "share_all": sa, "share_top": st, "lift": st / sa if sa else np.nan,
                             "share_top_absNF_in_event": float(nf[t & e].sum() / nf[t].sum()) if t.any() else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(out / "a3_rebooking.csv", index=False, lineterminator="\n")
    stamp("a3 done: " + "; ".join(f"k{r['k']} {r['event']}: top {r['share_top']:.3f} all {r['share_all']:.3f}"
                                  for _, r in R[(R["year"] == "all") & R["event"].isin(["any_event", "any_estban"])].iterrows()))
    return R


# ------------------------------------------------------------------------------------------- a4
def frame_markets():
    """B market-quarters of the frame: M-hat, Pop, access, the share sum, and the anchor market size
    R Pop / access, R the median of M access / Pop (one constant wherever the floor does not bind).
    The floor binds where M-hat is above the anchor. The stored share sum marks fewer of them: where
    finalize dropped a row after the shares were formed, a floor market's sum is below the cap."""
    d = pq.read_table(FRAME, columns=["is_B", "mca_code", "time_id", "pop_total", "_access", "M_mt",
                                      "share_B_cond", "_1mphi"]).to_pandas()
    d = d[d["is_B"]]
    cap_seen = float((1 - d["_1mphi"]).max())
    mk = d.groupby(["mca_code", "time_id"]).agg(pop=("pop_total", "first"), acc=("_access", "first"),
                                                M=("M_mt", "first"), SB=("share_B_cond", "sum")).reset_index()
    r = mk["M"] * mk["acc"] / mk["pop"]
    R = float(r.median())
    x = r / R - 1
    mk["floor"] = x > 1e-9
    floor_sb = (mk["SB"] - SHARE_CAP).abs() <= 1e-9
    mk["M_anchor"] = R * mk["pop"] / mk["acc"]
    info = {"phi_cap_seen_on_B_rows": cap_seen, "share_sum_max": float(mk["SB"].max()),
            "mq": int(len(mk)), "mq_at_anchor": int((x.abs() <= 1e-9).sum()), "mq_below_anchor": int((x < -1e-9).sum()),
            "mq_floor": int(mk["floor"].sum()), "mq_floor_share": float(mk["floor"].mean()),
            "mq_share_sum_at_cap": int(floor_sb.sum()), "mq_floor_with_share_sum_below_cap": int((mk["floor"] & ~floor_sb).sum()),
            "mq_share_sum_at_cap_not_floor": int((floor_sb & ~mk["floor"]).sum()),
            "R_M_access_over_pop": R, "rhat_at_findex_077": R * 0.77}
    if (abs(cap_seen - PHI_CAP) > 1e-12 or info["share_sum_max"] > SHARE_CAP + 1e-9 or info["mq_below_anchor"]
            or info["mq_share_sum_at_cap_not_floor"]):
        raise SystemExit(f"frame caps or anchor not as expected: {info}")
    return mk.rename(columns={"mca_code": "m", "time_id": "t"}), info


def a4_run(L, U, MT, om_u, top, N, out):
    mk, finfo = frame_markets()
    M2 = MT[["m", "t", "M_hat"]].reset_index().rename(columns={"index": "mt"}).merge(
        mk[["m", "t", "M", "M_anchor", "floor"]], on=["m", "t"], how="left")
    if M2["M"].isna().any() or float(np.max(np.abs(M2["M"] - M2["M_hat"]) / M2["M_hat"])) > 1e-12:
        raise SystemExit("the frame's M_mt and K0's M_hat disagree")
    mt = U["mt"].to_numpy()
    locked = np.bincount(mt, weights=(1 - om_u) * U["carry4"].to_numpy(), minlength=len(MT))
    pool = MT["M_hat"].to_numpy() - locked
    sel = L["mask"] & L["k"].isin(TYPES) & L["phi_mt"].notna()
    C = L.loc[sel, ["j", "m", "qi", "t", "k", "Dep", "carry", "NF", "phi_mt", "Dep_Act"]].copy()
    C["top"] = top[sel].to_numpy()
    Uk = U[["j", "m", "qi", "mt"]].reset_index().rename(columns={"index": "u"})
    C = C.merge(Uk, on=["j", "m", "qi"], how="left")
    n_nounit = int(C["u"].isna().sum())
    C = C[C["u"].notna()].copy()
    u = C["u"].astype(int).to_numpy()
    c_mt = C["mt"].astype(int).to_numpy()
    phi = C["phi_mt"].to_numpy()
    phic = np.minimum(phi, PHI_CAP)
    carry = C["carry"].to_numpy()
    dep = C["Dep"].to_numpy()
    k_arr = C["k"].to_numpy()
    om = np.where(k_arr == 4, om_u[u], 1.0)
    A1 = dep - phi * carry
    Aw = dep - (1 - om * (1 - phi)) * carry
    Mh = MT["M_hat"].to_numpy()[c_mt]
    Ma = M2["M_anchor"].to_numpy()[c_mt]
    V = {"L1": A1 <= 1e-6, "L2": Aw <= 1e-6, "U1": A1 > (1 - phic) * Mh, "U2": A1 > (1 - phi) * Mh,
         "U3": A1 > (1 - phic) * Ma, "U4": Aw > (1 - phi) * pool[c_mt]}
    l1_vs_k0 = int(np.sum(V["L1"] != (C["Dep_Act"].to_numpy() <= 1e-6)))
    rows = []
    for vname, v in V.items():
        for kname, s in [(str(k), k_arr == k) for k in TYPES] + [("pooled", np.ones(len(C), bool))]:
            t = s & C["top"].to_numpy()
            rows.append({"row": vname, "k": kname, "n_cells": int(s.sum()), "share": float(v[s].mean()),
                         "carry_weighted_share": float(carry[s & v].sum() / carry[s].sum()),
                         "n_top": int(t.sum()), "share_in_top_decile": float(v[t].mean()) if t.any() else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(out / "a4_feasibility.csv", index=False, lineterminator="\n")
    # reconciliation with demand.flows.trunc_share.E3
    tr = json.loads(XI_TRUNC.read_text(encoding="utf-8"))
    cnt = tr["counts"]
    xi = json.loads(XI_JSON.read_text(encoding="utf-8"))["E3"]
    rec = []
    B = {k: cnt[f"B_k{k}"] for k in (1, 2, 4, 5)}
    rec.append({"step": "1 prep B cells, all types (demand.flows.trunc_share.E3)", "k": "all",
                "truncated": sum(v[0] for v in B.values()), "cells": sum(v[1] for v in B.values())})
    rec.append({"step": "2 prep B cells, types 1, 2, 4", "k": "124",
                "truncated": sum(B[k][0] for k in TYPES), "cells": sum(B[k][1] for k in TYPES)})
    for k in TYPES:
        rec.append({"step": "2 prep B cells", "k": str(k), "truncated": B[k][0], "cells": B[k][1]})
    k0c = L[L["k"].isin(TYPES) & L["phi_mt"].notna() & L["Dep_lag"].notna() & L["rate_lag"].notna()]
    a_raw = k0c["Dep"] - k0c["phi_mt"] * k0c["carry"]
    rec.append({"step": "3 K0 cells with phi-hat and a carry", "k": "124", "truncated": int((a_raw <= 1e-6).sum()),
                "cells": int(len(k0c))})
    for k in TYPES:
        s = k0c["k"] == k
        rec.append({"step": "3 K0 cells with phi-hat and a carry", "k": str(k), "truncated": int((a_raw[s] <= 1e-6).sum()),
                    "cells": int(s.sum())})
    rec.append({"step": "4 K0 mask cells (Dep_lag > 0, carry > 0) with phi-hat: L1", "k": "124",
                "truncated": int(V["L1"].sum()), "cells": int(len(C))})
    for k in TYPES:
        s = k_arr == k
        rec.append({"step": "4 K0 mask cells (Dep_lag > 0, carry > 0) with phi-hat: L1", "k": str(k),
                    "truncated": int(V["L1"][s].sum()), "cells": int(s.sum())})
    RC = pd.DataFrame(rec)
    RC["share"] = RC["truncated"] / RC["cells"]
    RC.to_csv(out / "a4_trunc_reconciliation.csv", index=False, lineterminator="\n")
    pst = os.stat(PANEL_CSV)
    N["a4"] = {"frame_markets": finfo, "cells_without_unit": n_nounit, "L1_vs_k0_Dep_Act_mismatches": l1_vs_k0,
               "trunc_share_key_E3": xi["trunc_share_B"], "trunc_share_key_E3_k4": xi["k4"]["trunc_share"],
               "trunc_cache_key": tr["key"],
               "market_panel_csv_now": {"size": pst.st_size,
                                        "mtime": datetime.fromtimestamp(pst.st_mtime).strftime("%Y-%m-%dT%H:%M:%S")}}
    stamp("a4 done: " + "; ".join(f"{r['row']} {r['k']} {r['share']:.4f}" for _, r in R[R["k"] == "pooled"].iterrows()))
    return R, RC


# ------------------------------------------------------------------------------------------ main
def input_md5s(parts):
    files = {"market_panel.parquet": k0.PANEL, "demand_chars.parquet": demand_chars.chars_path(),
             "demand_3_index_spec_12.parquet": FRAME, "k0_netflow_panel.parquet": K0_PANEL,
             "k0_numbers.json": K0_NUMBERS, "k0_netflow_summary.csv": K0_SUMMARY}
    if {"a1", "a4", "gate"} & parts:
        files.update({"k2_numbers.json": K2_NUMBERS, "logit_delta_E3_spec_12.bin": k2.DELTA_BIN,
                      "ext1_alpha_json": k2.ALPHA_SOURCE, "k0_omega.csv": K0_OMEGA})
    if "a4" in parts:
        files.update({"xi_diagnostics.json": XI_JSON, "xi_cache_trunc_E3_spec_12.json": XI_TRUNC})
    if "a3" in parts:
        files.update({"estban_own_branches.csv": BRANCHES, "estban_unfiled_cells.csv": UNFILED,
                      "ESTBAN.csv": ESTBAN, "muni_mca_regions_2010_2024_panel.csv": MCA_XWALK})
    out = {k: {"path": str(p), "md5": md5_raw(p), "bytes": os.path.getsize(p)} for k, p in files.items()}
    if "a3" in parts:
        lst = sorted(glob.glob(str(paths.IF_DATA_LIST / "IF_DATA_List_*.csv")))
        h = hashlib.md5("".join(md5_raw(p) for p in lst).encode()).hexdigest()
        out["IF_DATA_List"] = {"path": str(paths.IF_DATA_LIST), "files": len(lst), "md5_of_file_md5s": h}
    return out


def write_md5s(out):
    lines = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "MD5SUMS.txt":
            b = p.read_bytes()
            if p.suffix in (".csv", ".json", ".txt"):
                b = b.replace(b"\r", b"")
            lines.append(f"{hashlib.md5(b).hexdigest()}  {p.name}")
    (out / "MD5SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="a1,a2,a3,a4")
    ap.add_argument("--delta", choices=("closed", "binary"), default="closed")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--write-cells", action="store_true")
    a = ap.parse_args()
    parts = set(a.parts.split(","))
    bad = parts - {"a1", "a2", "a3", "a4", "gate"}
    if bad:
        raise SystemExit(f"unknown parts {sorted(bad)}")
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    T = {}
    N = {"script": "diagnostics/diag_netflow_gap.py", "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
         "parts": sorted(parts), "delta_for_worlds": a.delta, "panel_md5_want": PANEL_MD5,
         "alpha_bar": k2.ALPHA_BAR, "kappa_base": KAPPA_BASE, "phi_cap": PHI_CAP, "share_cap": SHARE_CAP}
    stamp(f"start: parts {sorted(parts)}, delta {a.delta}, out {out}")
    N["inputs"] = input_md5s(parts)
    T["input_md5s"] = round(time.time() - t0)
    if not gate_a0(N):
        jdump(N, out / "netflow_gap_numbers.json")
        raise SystemExit(f"G-A0 failed: {N['gate_A0']['fails']}")
    L = load_k0_panel()
    top, p90 = top_decile(L)
    N["top_decile_p90"] = {str(k): v for k, v in p90.items()}
    om_u = None
    if {"a1", "a4", "gate"} & parts:
        t1 = time.time()
        k0.COLS = k0.COLS + ["spread_a5"]
        df = k0.load_panel()
        flags = k0.rate_flags(df[df["is_B"] == 1])
        mk = k0.load_demand()
        want = json.loads(K2_NUMBERS.read_text(encoding="utf-8"))
        gates, worlds = {}, {}
        for source in (a.delta, "binary" if a.delta == "closed" else "closed"):
            ctx, menu, extra = build_world(df, flags, mk, source)
            gates[source] = gate_a1(ctx["N"], want)
            gates[source]["delta_source"] = ctx["N"]["delta_source"]
            gates[source]["delta_vs_closed_form_max_abs"] = ctx["N"].get("delta_vs_closed_form_max_abs")
            stamp(f"G-A1 [{source}] {'PASS' if gates[source]['pass'] else 'FAIL'}: max abs {gates[source]['max_abs']:.2e}; "
                  f"plan keys {json.dumps(gates[source]['plan_keys'])}")
            if source == a.delta:
                worlds = (ctx, extra)
            else:
                del ctx, menu, extra
        N["gate_A1"] = gates
        T["build_and_gate"] = round(time.time() - t1)
        if not gates[a.delta]["pass"]:
            jdump(N, out / "netflow_gap_numbers.json")
            raise SystemExit(f"G-A1 failed on the {a.delta} delta: {gates[a.delta]}")
        ctx, extra = worlds
        b = df[(df["is_B"] == 1) & (df["qi"] >= k0.qi(*k0.T0) - 1) & (df["qi"] <= k0.qi(*k0.T1))].copy()
        Lin = k0.netflow(b, flags, mk)
        cols = ["j", "m", "t", "k", "Dep", "Dep_lag", "rate_lag", "NF", "phi_mt", "Dep_Act", "rate_fallback", "judicial"]
        x = Lin[cols].sort_values(["j", "m", "t", "k"]).reset_index(drop=True)
        y = L[cols].sort_values(["j", "m", "t", "k"]).reset_index(drop=True)
        same_keys = len(x) == len(y) and bool((x[["j", "m", "t", "k"]].to_numpy() == y[["j", "m", "t", "k"]].to_numpy()).all())
        dif = {c: float(np.nanmax(np.abs(x[c].to_numpy(float) - y[c].to_numpy(float)))) if same_keys else float("inf")
               for c in ["Dep", "Dep_lag", "rate_lag", "NF", "phi_mt", "Dep_Act"]}
        nan_same = same_keys and all(bool((x[c].isna() == y[c].isna()).all()) for c in ["NF", "phi_mt", "Dep_lag"])
        N["gate_A0"]["k0_panel_rebuilt_here"] = {"rows_here": int(len(x)), "rows_stored": int(len(y)),
                                                 "same_keys": same_keys, "same_nan": nan_same, "max_abs": dif}
        if not (same_keys and nan_same and dif["NF"] <= 1e-6 and dif["phi_mt"] <= TOL_GATE):
            jdump(N, out / "netflow_gap_numbers.json")
            raise SystemExit(f"K0's net flow rebuilt here differs from {K0_PANEL.name}: {N['gate_A0']['k0_panel_rebuilt_here']}")
        stamp(f"K0 panel rebuilt here = stored: {len(x):,} rows, max abs NF {dif['NF']:.1e}")
        if "a1" in parts:
            t1 = time.time()
            _, om_u = a1_run(ctx, extra, Lin, N, a.write_cells, out)
            T["a1"] = round(time.time() - t1)
        if "a4" in parts:
            if om_u is None:
                tab, fill = omega_rule()
                om_u = omega_of(ctx["U"]["j"].to_numpy(), ctx["U"]["qi"].to_numpy(), tab, fill)
            t1 = time.time()
            a4_run(L, ctx["U"], ctx["MT"], om_u, top, N, out)
            T["a4"] = round(time.time() - t1)
        del ctx, extra, worlds, df, b, Lin, x, y
    if "a2" in parts:
        t1 = time.time()
        a2_run(L, N, out)
        T["a2"] = round(time.time() - t1)
    if "a3" in parts:
        t1 = time.time()
        cells, E = a3_events(L, N)
        a3_table(cells, E, top[L["mask"] & L["k"].isin(TYPES)].to_numpy(), N, out)
        T["a3"] = round(time.time() - t1)
    T["total"] = round(time.time() - t0)
    N["seconds"] = T
    N["finished"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    jdump(N, out / "netflow_gap_numbers.json")
    for line in write_md5s(out):
        print(line)
    stamp(f"done in {T['total']} s")


if __name__ == "__main__":
    main()
