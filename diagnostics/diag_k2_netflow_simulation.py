"""
diag_k2_netflow_simulation.py
=============================
Task K2 of claude/kappa_handoff_2026-11-03.md: the simulation test of the switching-cost design
(Drafts/Deposit Competition/kappa_design_2026-11-02.md, seventeenth version, Section 7, Group B).
The balance equation (20) of the note is simulated on the panel's own pools, the net flow (21) is
built from it, and the regression (30) and its PPML form are run on the simulated net flow, to see
what the ratios (31) return when the model holds and when one assumption is broken on purpose.

The simulated world (equation numbers are the note's):
  - B cells, 2016Q1-2024Q4, bank x market x quarter. The pools are the panel's lagged stocks by
    bank, type (1, 2, 4, 5) and market, accrued at the lagged rate of the same cell (K0's carry),
    and the outside pool O_mt = M_hat_mt - sum of the B lagged stocks (K0 (d)). D firms are on the
    menu of every market, as in the paper's market structure, and their depositors sit in O.
  - Mean utilities: the stored logit deltas of E3 spec 12 (blp_logit.jl berry_inversion, the
    10-29 local prep; checked here against the closed form) when the prep's metadata names the
    binary's panel (DELTA_BIN_PANEL_MD5), else the closed form itself (read_menu records which),
    A0 (no mu), one phi_mt (the stored phi-hat), A = 1, no clock (omega = 1, lambda = 0, so
    nothing is locked).
  - The share of money sitting at bank j' is (6): exp(delta_lk - kappa 1{l != j'}) over
    1 + sum of the same over the market's menu; the outside pool pays no kappa. The awake flow
    into (j,4) is (1 - phi) e_j4 [W_j/D_j + e^-kappa sum_{j' != j} W_j'/D_j' + O/D_0], with W the
    accrued pool of a source bank and D its denominator, the own pool inside the sum; the balance
    is (20) and NF is (21).
  - The price. The stored delta_j4 has no within relation to the gap (its slope on g within the two
    sets of effects is about -0.03 per pp), so with the stored delta as it is the simulated flows
    would not respond to the regressor at all. The simulation therefore writes
    delta_j4 = n_j4 - ALPHA_BAR * g_jmt, with g in pp a year and the non-price part n taken from
    the stored delta in one of three ways (DGP):
        stored  n = delta + ALPHA_BAR g with its within slope on g removed: the stored delta's
                bank x market and market x quarter means and its residual variation are kept, its
                within slope on g is set to -ALPHA_BAR. The realistic case.
        fe      n = the two-way fixed-effect fit of delta + ALPHA_BAR g: the within-cell variation
                of the non-price part is dropped, the cross-cell heterogeneity kept.
        ideal   n = one constant (the median of delta + ALPHA_BAR g), and every K1 destination is on
                the menu: the note's world, in which C_jmt varies only with phi_mt.
    Types 1, 2 and 5 of B banks and every D product keep their stored delta. Products absent from
    the prep (the cells (11-B) truncates) are not on the menu, except in `ideal`.

Steps (--steps, default 2):
  2  recovery: kappa in KAPPAS, the three DGPs; OLS of (30) (no clock: the exit block is Q4 g) on
     the K1 sample and on its destinations on the menu, the note's controls with R4 gbar4 and a
     variant with R4 g; the PPML form (below) on the destinations on the menu at KAPPA_BASE.
                                                                                  k2_recovery.csv
     Then the decomposition at KAPPA_BASE: each estimator on an outcome generated from its own
     mean (exact, with a market-quarter scale, with cell effects, with within-cell noise), and the
     ideal world with the inside mass, the outside pool and the variation of phi switched off one
     at a time.                                                     k2_recovery_decomposition.csv
  3  A1 broken: one bank per market (its largest type-4 holder among the banks on the menu) has
     delta_j4 raised in each quarter until its time deposits take 30% and 60% of the market's
     awake money.                                                                  k2_break_a1.csv
  4  linearization: gaps multiplied by GAP_SCALES (in the price of the simulation and in every
     price regressor).                                                          k2_break_linear.csv
  5  sorting: exp(delta_l4) multiplied by h for money sitting at any other bank.     k2_sorting.csv
  6  the outside pool left out of the regression while it is in the simulation.     k2_outside.csv
Steps 3-6 run on DGP `stored` at KAPPA_BASE (step 5 also at kappa = 0, step 6 at every kappa).

The PPML form. The note's remedy (ii) of A6 fits (28) with a log link, but the net flow is entry
minus exit and is negative in about half of the cells, so (28) cannot be fitted by Poisson PML as it
stands. What carries (28)'s multiplicative structure and is non-negative is the awake inflow with
the stayers, y_A = (Dep_j4 - phi carry_j4) / Dep_j2,t-1, which in the simulation uses the true phi:
    E[y_A] = exp(FE_jm + FE_mt + b g) (1 + th_s Q4 + th_1 Q1 + th_x R2 + th_x1 R1 + th_4 R4 + th_O O),
own savings normalized to one, so th_x and th_x1 / th_1 read e^-kappa and th_1 and th_x1 / th_x read
e^a. On the data this object needs phi-hat (it is (11-B) without the truncation); K1 needs a
decision there (k2_report.md).

Inputs (read only):
  - processed/market_panel.parquet (md5 d01b8bae), once, through diag_k0_netflow_objects.load_panel
    with K0's columns plus spread_a5; the 2013-2015 balances from processed/demand_chars.parquet.
  - ESTIMATION_OUTPUT/DEMAND_PREP/demand_3_index_spec_12.parquet (phi_mt, M_mt, the menu).
  - ESTIMATION_OUTPUT/BLP_RESULTS/logit/logit_delta_E3_spec_12.bin (Int64 n, then n Float64, the
    prep's row order; blp_logit.jl), read only for a prep whose metadata names its panel.
Outputs (Rout): k2_recovery.csv, k2_recovery_decomposition.csv, k2_break_a1.csv,
  k2_break_linear.csv, k2_sorting.csv, k2_outside.csv, k2_numbers.json. Standard errors are clustered by conglomerate; the simulation has
  no sampling noise, so they measure how much of the simulated net flow (30) leaves unexplained,
  not sampling error.
Usage:  python -I diagnostics/diag_k2_netflow_simulation.py [--steps 2|2-6|3,5]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from utils import demand_chars
from utils import paths
import diag_k0_netflow_objects as k0

# ------------------------------------------------------------------------------------- parameters
SEED = 20261103          # printed; no step draws a random number (A0 has no mu draws)
ALPHA_BAR = -0.117       # mean price coefficient per pp a year (the note's adopted value under (III))
KAPPAS = (0.0, 0.5, 1.0, 2.0)
KAPPA_BASE = 0.5         # steps 3-6
A_AWARE = 1.0            # awareness: every bank in every choice set
OMEGA, LAMBDA = 1.0, 0.0 # no clock: the whole type-4 stock can move, none of it locked
A1_TARGETS = (0.30, 0.60)
GAP_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
SORT_H = (0.9, 0.7, 0.5)
DGPS = ("ideal", "fe", "stored")
assert A_AWARE == 1.0 and OMEGA == 1.0 and LAMBDA == 0.0, "the simulator implements A = 1 and no clock only"

DEMAND = paths.demand_parquet_dir() / "demand_3_index_spec_12.parquet"
DELTA_BIN = paths.PROCESSED / "ESTIMATION_OUTPUT" / "BLP_RESULTS" / "logit" / "logit_delta_E3_spec_12.bin"
DELTA_BIN_PANEL_MD5 = "d01b8bae"   # the 10-27 panel, which the prep behind DELTA_BIN was built on
OUT = paths.rout_dir()
T0, T1 = k0.T0, k0.T1
TYPES5 = (1, 2, 4, 5)

# Regressors of (30) without the clock (omega = 1: Q4 omega g = Q4 g, the lambda term is zero).
SLOPES = ["g", "R2g", "Q1g", "R1g", "Q4g"]
LEVELS = ["R2", "Q1", "R1", "Q4", "O", "Og", "R4", "R4gb4"]
PRICE_COLS = {"g", "R2g", "Q1g", "R1g", "Q4g", "Og", "R4gb4", "R4g"}   # scaled with the gaps (step 4)
PPML_POOLS = ["Q4", "Q1", "R2", "R1", "R4", "O"]

stamp = k0.stamp


# ------------------------------------------------------------------------------- two-way demeaning
class TwoWay:
    """Alternating projections on two sets of effects, weighted or not, all columns at once."""

    def __init__(self, a, b):
        self.c1, u1 = pd.factorize(a)
        self.c2, u2 = pd.factorize(b)
        self.n1, self.n2, self.n = len(u1), len(u2), len(self.c1)
        r = np.arange(self.n)
        self.D1 = sp.csr_matrix((np.ones(self.n), (r, self.c1)), shape=(self.n, self.n1))
        self.D2 = sp.csr_matrix((np.ones(self.n), (r, self.c2)), shape=(self.n, self.n2))
        self.D1t, self.D2t = self.D1.T.tocsr(), self.D2.T.tocsr()

    def demean(self, X, w=None, tol=1e-10, maxit=5000):
        X = np.asarray(X, float)
        one = X.ndim == 1
        X = X[:, None] if one else X
        w = np.ones(self.n) if w is None else np.asarray(w, float)
        w1 = np.maximum(self.D1t @ w, 1e-300)[:, None]
        w2 = np.maximum(self.D2t @ w, 1e-300)[:, None]
        wX = w[:, None] * X
        a = np.zeros((self.n1, X.shape[1]))
        b = np.zeros((self.n2, X.shape[1]))
        scale = np.maximum(np.abs(X).max(axis=0), 1e-300)
        for it in range(maxit):
            a_new = (self.D1t @ (wX - w[:, None] * (self.D2 @ b))) / w1
            b_new = (self.D2t @ (wX - w[:, None] * (self.D1 @ a_new))) / w2
            ch = max((np.abs(a_new - a).max(axis=0) / scale).max(), (np.abs(b_new - b).max(axis=0) / scale).max())
            a, b = a_new, b_new
            if ch < tol:
                break
        R = X - self.D1 @ a - self.D2 @ b
        self.last_iter = it + 1
        return R[:, 0] if one else R


def ols_fe(y, Xd, names, cl, fe):
    """OLS of y on the demeaned design Xd (columns `names`) with the two sets of effects; standard
    errors clustered on `cl`, with the usual small-sample factor."""
    yd = fe.demean(y)
    s = Xd.std(axis=0)
    keep = s > 1e-12
    Z = Xd[:, keep] / s[keep]
    bz, *_ = np.linalg.lstsq(Z, yd, rcond=None)
    u = yd - Z @ bz
    XtX_inv = np.linalg.pinv(Z.T @ Z)
    cg = pd.factorize(cl)[0]
    G = cg.max() + 1
    Sg = np.zeros((G, Z.shape[1]))
    np.add.at(Sg, cg, Z * u[:, None])
    n, k = Z.shape
    adj = G / (G - 1) * (n - 1) / max(n - k - fe.n1 - fe.n2, 1)
    V = adj * XtX_inv @ (Sg.T @ Sg) @ XtX_inv
    b = np.full(len(names), np.nan)
    b[keep] = bz / s[keep]
    Vf = np.full((len(names), len(names)), np.nan)
    Vf[np.ix_(keep, keep)] = V / np.outer(s[keep], s[keep])
    r2w = 1 - (u @ u) / (yd @ yd) if (yd @ yd) > 0 else np.nan
    return dict(zip(names, b)), pd.DataFrame(Vf, index=names, columns=names), {"n": int(n), "G": int(G), "r2_within": r2w}


def ratio(b, V, num, den):
    if num not in b or den not in b or not np.isfinite(b[num]) or not np.isfinite(b[den]) or b[den] == 0:
        return np.nan, np.nan
    r = b[num] / b[den]
    gr = np.array([1 / b[den], -b[num] / b[den] ** 2])
    v = V.loc[[num, den], [num, den]].to_numpy()
    return r, float(np.sqrt(max(gr @ v @ gr, 0)))


def drop_zero_groups(y, a, b):
    """Rows of the effect groups whose outcome is zero throughout: their effect is minus infinity
    under the log link and they carry no information on the slopes (as ppmlhdfe drops them)."""
    keep = np.ones(len(y), bool)
    while True:
        s1 = pd.Series(y[keep]).groupby(a[keep]).transform("sum").to_numpy()
        s2 = pd.Series(y[keep]).groupby(b[keep]).transform("sum").to_numpy()
        bad = (s1 <= 0) | (s2 <= 0)
        if not bad.any():
            return keep
        idx = np.flatnonzero(keep)
        keep[idx[bad]] = False


def fe_project(fe, X, w):
    """X minus its weighted least-squares projection on the two sets of effects, solved exactly:
    the market-quarter block of the normal equations is diagonal, so it is eliminated and the
    bank x market block (the Schur complement) is factorized once for every column of X. A ridge
    of 1e-12 of the largest diagonal entry fixes the one free level per connected component; the
    projection itself does not depend on it."""
    from scipy.sparse.linalg import splu
    X = np.asarray(X, float)
    X = X[:, None] if X.ndim == 1 else X
    W1 = fe.D1.multiply(w[:, None]).tocsc()           # n x n1, rows scaled by w
    d1 = np.asarray(W1.sum(axis=0)).ravel()
    d2 = np.maximum(fe.D2t @ w, 1e-300)
    C = (W1.T @ fe.D2).tocsr()                          # n1 x n2: D1' W D2
    S = sp.diags(d1) - C @ sp.diags(1 / d2) @ C.T
    S = (S + sp.identity(fe.n1) * 1e-12 * d1.max()).tocsc()
    lu = splu(S)
    wX = w[:, None] * X
    r2 = fe.D2t @ wX
    ra = fe.D1t @ wX - C @ (r2 / d2[:, None])
    a = lu.solve(ra)
    b = (r2 - C.T @ a) / d2[:, None]
    return X - fe.D1 @ a - fe.D2 @ b


def ppml_fe(y, g, X, names, a, b, cl, maxit=200, tol=1e-9):
    """Poisson PML with mean exp(FE_a + FE_b + beta g) (1 + X theta), by Fisher scoring on the
    linearized index, the two sets of effects partialled out at every step by the exact weighted
    projection (fe_project); step halving keeps 1 + X theta positive and the deviance falling.
    Errors: the cluster sandwich on the effect-partialled scores."""
    keep = drop_zero_groups(y, a, b)
    y, g, X, cl = y[keep], g[keep], X[keep], cl[keep]
    y = y / y.mean()                                   # scale: the slopes and theta do not move
    fe = TwoWay(a[keep], b[keep])
    p = X.shape[1]
    theta, beta = np.zeros(p), 0.0
    P = np.ones(len(y))
    mu = 0.5 * (y + 1.0)                               # ppmlhdfe's start
    eta = np.log(mu)

    def dev(mu_):
        mu_ = np.maximum(mu_, 1e-300)
        t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu_), 0.0)
        return 2 * np.sum(t - (y - mu_))

    D0 = np.inf
    conv, stalled = False, False
    for it in range(maxit):
        J = np.column_stack([g, X / P[:, None]])
        c = np.log(P) - (X @ theta) / P
        z = eta - c + (y - mu) / mu
        M = fe_project(fe, np.column_stack([z, J]), mu)
        zd, Jd = M[:, 0], M[:, 1:]
        sw = np.sqrt(mu)
        Z = Jd * sw[:, None]
        cs = np.maximum(np.sqrt((Z ** 2).sum(axis=0)), 1e-300)
        psi, *_ = np.linalg.lstsq(Z / cs, zd * sw, rcond=None)
        psi = psi / cs
        if not np.isfinite(psi).all():
            raise FloatingPointError(f"PPML step not finite at iteration {it}")
        e = zd - Jd @ psi
        lin_new = (z - e) - (X @ psi[1:]) / P          # FE + beta g at the new values
        lin_old = eta - np.log(P)
        th_old, be_old = theta.copy(), beta
        step = 1.0
        while True:
            th = th_old + step * (psi[1:] - th_old)
            Pn = 1 + X @ th
            if (Pn > 0).all():
                eta_n = lin_old + step * (lin_new - lin_old) + np.log(Pn)
                if np.isfinite(eta_n).all() and eta_n.max() < 700:
                    mu_n = np.exp(eta_n)
                    Dn = dev(mu_n)
                    if np.isfinite(Dn) and (Dn <= D0 * (1 + 1e-12) or not np.isfinite(D0)):
                        break
            step /= 2
            if step < 1e-10:
                stalled = True                          # no step lowers the deviance: stop here
                break
        if stalled:
            break
        dpsi = np.abs(step * np.r_[psi[0] - be_old, psi[1:] - th_old]).max()
        theta, beta = th, be_old + step * (psi[0] - be_old)
        P, eta, mu = Pn, eta_n, mu_n
        conv = np.isfinite(D0) and abs(D0 - Dn) / max(abs(Dn), 1e-300) < 1e-12 and dpsi < tol
        D0 = Dn
        if conv:
            break
    J = np.column_stack([g, X / P[:, None]])
    Jd = fe_project(fe, J, mu)
    H = (Jd * mu[:, None]).T @ Jd
    cg = pd.factorize(cl)[0]
    G = cg.max() + 1
    Sg = np.zeros((G, Jd.shape[1]))
    np.add.at(Sg, cg, Jd * (y - mu)[:, None])
    Hi = np.linalg.pinv(H)
    nm = ["g"] + names
    V = pd.DataFrame(G / (G - 1) * Hi @ (Sg.T @ Sg) @ Hi, index=nm, columns=nm)
    est = dict(zip(nm, np.r_[beta, theta]))
    return est, V, {"n": int(len(y)), "G": int(G), "iterations": it + 1, "converged": bool(conv),
                    "stalled": bool(stalled), "deviance": float(D0), "at_zero_bound": [],
                    "dropped_zero_groups": int((~keep).sum())}


# ------------------------------------------------------------------------------------------ inputs
def read_menu():
    """The prep's menu with each product's mean utility delta, and a record of where delta came from.

    The key is the prep's schema metadata demand_chars.market_panel_md5, the md5 of the panel the
    prep was built on. When it starts with DELTA_BIN_PANEL_MD5 (the panel the stored binary's prep
    was built on), delta is the stored logit binary (DELTA_BIN), which must have the prep's row count
    and equal the closed form of blp_logit.jl berry_inversion, computed here from the prep's shares,
    to 1e-8. Otherwise (another panel's prep, such as the interim frames on a rebuilt panel, or a
    prep without the key), delta is that closed form, which must be finite on every row. The record
    (delta_source, frame_panel_md5, delta_binary_panel_md5, delta_binary_rows, prep_rows,
    delta_vs_closed_form_max_abs) goes into the callers' numbers."""
    cols = ["CodConglomeradoPrudencial", "mca_code", "time_id", "deposit_type", "is_B", "share_B_cond",
            "share_D", "pop_total", "banked_correction", "phi_mt", "Dep_Act"]
    d = pq.read_table(DEMAND, columns=cols).to_pandas()
    frame_md5 = (pq.read_schema(DEMAND).metadata or {}).get(demand_chars.PANEL_MD5_KEY, b"").decode() or None
    # The closed form of blp_logit.jl berry_inversion.
    B = d["is_B"].to_numpy(bool)
    db = d[B].assign(pair=(d.loc[B, "mca_code"].astype(str) + "|" + d.loc[B, "time_id"].astype(str)).to_numpy())
    SB = db.groupby("pair")["share_B_cond"].sum()
    bcpop = db.assign(x=db["banked_correction"].fillna(1.1) * db["pop_total"].fillna(0)).groupby("pair")["x"].mean()
    pt = db.groupby("pair")["time_id"].first()
    w = bcpop / bcpop.groupby(pt).transform("sum")
    K = (w * (1 - SB)).groupby(pt).sum()
    G = K - d.loc[~B].groupby("time_id")["share_D"].sum().reindex(K.index).fillna(0)
    gmt = ((1 - SB) * G.reindex(pt.values).values / K.reindex(pt.values).values).clip(1e-12, 1)
    chk = np.empty(len(d))
    chk[B] = np.log(db["share_B_cond"].clip(lower=1e-15)).to_numpy() - np.log(gmt.reindex(db["pair"]).to_numpy())
    dd = d[~B]
    chk[~B] = np.log(dd["share_D"].clip(lower=1e-15)).to_numpy() - np.log(G.reindex(dd["time_id"]).to_numpy())
    rec = {"frame_panel_md5": frame_md5, "delta_binary_panel_md5": DELTA_BIN_PANEL_MD5, "prep_rows": len(d)}
    if frame_md5 is not None and frame_md5.startswith(DELTA_BIN_PANEL_MD5):
        raw = DELTA_BIN.read_bytes()
        n = int(np.frombuffer(raw[:8], "<i8")[0])
        delta = np.frombuffer(raw[8:], "<f8")
        if len(delta) != n or n != len(d):
            raise SystemExit(f"delta binary has {n} rows ({len(delta)} values); the prep, built on the binary's "
                             f"panel {DELTA_BIN_PANEL_MD5}, has {len(d)}")
        err = float(np.nanmax(np.abs(chk - delta)))
        if err > 1e-8:
            raise SystemExit(f"stored delta differs from the closed form by {err:g}: not this prep's binary")
        d["delta"] = delta
        rec.update(delta_source="binary", delta_binary_rows=n, delta_vs_closed_form_max_abs=err)
        note = f"frame panel {frame_md5[:8]}: delta binary = closed form to {err:.1e}"
    else:
        bad = int((~np.isfinite(chk)).sum())
        if bad:
            raise SystemExit(f"the closed form is not finite on {bad:,} prep rows")
        d["delta"] = chk
        rec.update(delta_source="closed form", delta_binary_rows=None, delta_vs_closed_form_max_abs=None)
        note = (f"frame panel {frame_md5[:8] if frame_md5 else 'not recorded'}, the binary's {DELTA_BIN_PANEL_MD5}: "
                "delta from the closed form")
    d = d.rename(columns={"CodConglomeradoPrudencial": "j", "mca_code": "m", "time_id": "t", "deposit_type": "k"})
    d["j"], d["m"], d["t"] = d["j"].astype(str), d["m"].astype(str), d["t"].astype(str)
    stamp(f"menu: {len(d):,} products ({int(B.sum()):,} B); {note}")
    return d, rec


def build(df, flags, mk, menu):
    """The unit table (bank x market x quarter, B) with pools, menu masses, prices and the K1
    regressors, and the market-quarter and quarter tables the simulator needs."""
    b = df[(df["is_B"] == 1) & (df["qi"] >= k0.qi(*T0) - 1) & (df["qi"] <= k0.qi(*T1))].copy()
    # pools at t: stocks at t-1 accrued at the cell's own lagged rate (K0's carry)
    lag = b[["j", "m", "qi", "risk_free_qoq"] + [f"dep_a{k}" for k in TYPES5] + [f"spread_a{k}" for k in TYPES5]].copy()
    lag["qi"] += 1
    n_noacc = 0
    for k in TYPES5:
        D = lag[f"dep_a{k}"].fillna(0.0)
        acc = 1 + lag["risk_free_qoq"] - lag[f"spread_a{k}"]
        n_noacc += int(((D > 0) & acc.isna()).sum())
        lag[f"D{k}"] = D
        lag[f"A{k}"] = acc.where(acc.notna(), 1.0)
    lag["W"] = sum(lag[f"A{k}"] * lag[f"D{k}"] for k in TYPES5)
    lag["carry4"] = lag["A4"] * lag["D4"]
    lag["r1_acc"], lag["r2_acc"], lag["r4_acc"] = lag["A1"], lag["A2"], lag["A4"]
    pools = lag[["j", "m", "qi", "W", "carry4", "D2", "D4", "r1_acc", "r2_acc", "r4_acc"]]
    # prices at t
    cur = b[["j", "m", "qi", "spread_ann_a2", "spread_ann_a4"]].copy()
    cur["g"] = 100 * (cur["spread_ann_a2"] - cur["spread_ann_a4"])
    cur["rho4"] = 100 * cur["spread_ann_a4"]
    # products at t
    mb = menu[menu["is_B"]].copy()
    mb["qi"] = mb["t"].str[:4].astype(int) * 4 + mb["t"].str[-1].astype(int) - 1
    mb["e"] = np.exp(mb["delta"])
    prod = mb.pivot_table(index=["j", "m", "qi"], columns="k", values="e", aggfunc="sum").reset_index()
    prod.columns = ["j", "m", "qi"] + [f"e{int(c)}" for c in prod.columns[3:]]
    for k in TYPES5:
        if f"e{k}" not in prod:
            prod[f"e{k}"] = 0.0
    dl4 = mb[mb["k"] == 4][["j", "m", "qi", "delta", "Dep_Act"]].rename(columns={"delta": "delta4", "Dep_Act": "dep_act4_prep"})
    U = pools.merge(prod, on=["j", "m", "qi"], how="outer").merge(cur[["j", "m", "qi", "g", "rho4"]], on=["j", "m", "qi"], how="left")
    U = U.merge(dl4, on=["j", "m", "qi"], how="left")
    U = U[(U["qi"] >= k0.qi(*T0)) & (U["qi"] <= k0.qi(*T1))].copy()
    for c in ["W", "carry4", "D2", "D4"] + [f"e{k}" for k in TYPES5]:
        U[c] = U[c].fillna(0.0)
    U["t"] = k0.tid(U["qi"]).to_numpy()
    # market-quarter: phi and M from the prep, O from K0 (d)
    O = k0.outside_pool(b, mk)
    MT = O[["m", "t", "phi_mt", "M_hat", "sum_dep", "O", "o"]].copy()
    # a market-quarter with no lagged B stock has its whole M_hat outside; a negative O (M_hat below
    # the inside stocks, 34 market-quarters in K0) is a pool of zero in the simulation, and its
    # destinations leave the regression samples (k1_rows)
    MT["O_raw"] = MT["M_hat"] - MT["sum_dep"].fillna(0.0)
    MT["O"] = MT["O_raw"].clip(lower=0.0)
    n_units0 = len(U)
    U = U.merge(MT[["m", "t"]], on=["m", "t"], how="inner")
    U = U.sort_values(["m", "qi", "j"]).reset_index(drop=True)
    mt_key = U["m"] + "|" + U["t"]
    U["mt"], mt_u = pd.factorize(mt_key)
    MT["key"] = MT["m"] + "|" + MT["t"]
    MT = MT.set_index("key").loc[mt_u].reset_index()
    U["tc"], t_u = pd.factorize(U["t"])
    md = menu[~menu["is_B"]].assign(e=lambda x: np.exp(x["delta"]))
    SD = md.groupby("t")["e"].sum().reindex(t_u).fillna(0.0).to_numpy()
    SD4 = md[md["k"] == 4].groupby("t")["e"].sum().reindex(t_u).fillna(0.0).to_numpy()
    info = {"units": int(len(U)), "units_dropped_no_phi": int(n_units0 - len(U)), "market_quarters": int(len(MT)),
            "pools_with_stock_but_no_lagged_rate": n_noacc,
            "D_products_per_quarter": [int(x) for x in md.groupby("t").size().reindex(t_u).fillna(0)]}
    return U, MT, SD, SD4, info


def k1_rows(U, MT, df, flags, mk):
    """The destinations of K1, with K0's definitions: B type-4 cells with a net flow, positive own
    lagged savings, no rate fallback at t or t-1, a gap, and the pool weights of (27)."""
    b = df[(df["is_B"] == 1) & (df["qi"] >= k0.qi(*T0) - 1) & (df["qi"] <= k0.qi(*T1))].copy()
    L = k0.netflow(b, flags, mk)
    P = k0.pool_weights(b, L)
    L4 = L[(L["k"] == 4) & L["NF"].notna()][["j", "m", "qi", "carry", "NF", "rate_fallback", "judicial"]]
    K = L4.merge(P[["j", "m", "qi", "own2", "Q1", "Q4", "R1", "R2", "R4", "one_B_bank", "own_sav_lag_ok"]],
                 on=["j", "m", "qi"], how="left")
    K = K.merge(U.reset_index()[["index", "j", "m", "qi", "mt", "g", "rho4", "D4", "carry4", "r1_acc", "r2_acc"]],
                on=["j", "m", "qi"], how="inner").rename(columns={"index": "u"})
    # the carry of (21) and the simulator's own type-4 pool are the same number
    gap = (K["carry"] - K["carry4"]).abs().max()
    if gap > 1e-6 * max(K["carry"].abs().max(), 1):
        raise SystemExit(f"K0's carry and the simulator's own type-4 pool differ by {gap:g}")
    # pool-weighted rivals' time-deposit spread (lagged type-4 stocks as weights), minus own
    W4 = U[["mt", "D4", "rho4"]].copy()
    ok = W4["rho4"].notna() & (W4["D4"] > 0)
    num = np.bincount(W4.loc[ok, "mt"], weights=(W4["D4"] * W4["rho4"])[ok], minlength=U["mt"].max() + 1)
    den = np.bincount(W4.loc[ok, "mt"], weights=W4["D4"][ok], minlength=U["mt"].max() + 1)
    own_ok = K["rho4"].notna() & (K["D4"] > 0)
    rn = num[K["mt"]] - np.where(own_ok, K["D4"] * K["rho4"], 0)
    rd = den[K["mt"]] - np.where(own_ok, K["D4"], 0)
    K["gb4"] = np.where(rd > 0, rn / np.where(rd > 0, rd, 1) - K["rho4"], np.nan)
    n_all = len(K)
    sel = (K["own_sav_lag_ok"] == 1) & (K["rate_fallback"] == 0) & K["g"].notna()
    K = K[sel].copy()
    n_sel = len(K)
    K = K[MT["O_raw"].to_numpy()[K["mt"].to_numpy()] >= 0].copy()
    info = {"K0_type4_netflow_cells": int(n_all), "K1_rows": int(len(K)),
            "dropped_no_own_savings_or_fallback_or_gap": int(n_all - n_sel),
            "dropped_negative_outside_pool": int(n_sel - len(K))}
    return K, info


def design(K, MT, scale=1.0, extra=()):
    O = MT["O"].to_numpy()[K["mt"]] / K["own2"].to_numpy()
    g = scale * K["g"].to_numpy()
    R4gb4 = np.where(K["R4"].to_numpy() > 0, K["R4"].to_numpy() * scale * np.nan_to_num(K["gb4"].to_numpy()), 0.0)
    cols = {"g": g, "R2g": K["R2"].to_numpy() * g, "Q1g": K["Q1"].to_numpy() * g, "R1g": K["R1"].to_numpy() * g,
            "Q4g": K["Q4"].to_numpy() * g, "R2": K["R2"].to_numpy(), "Q1": K["Q1"].to_numpy(), "R1": K["R1"].to_numpy(),
            "Q4": K["Q4"].to_numpy(), "O": O, "Og": O * g, "R4": K["R4"].to_numpy(), "R4gb4": R4gb4,
            "R4g": K["R4"].to_numpy() * g}
    return cols


# --------------------------------------------------------------------------------------- simulator
def nonprice(U, dgp, info):
    """n_j4, the non-price part of delta_j4 under each DGP, on the units of U (NaN = not on the menu)."""
    has = U["delta4"].notna() & U["g"].notna()
    n_prime = (U["delta4"] + ALPHA_BAR * U["g"]).to_numpy()
    out = {}
    fe = TwoWay((U.loc[has, "j"] + "|" + U.loc[has, "m"]).to_numpy(), U.loc[has, "mt"].to_numpy())
    M = fe.demean(np.column_stack([U.loc[has, "g"].to_numpy(), U.loc[has, "delta4"].to_numpy(), n_prime[has]]), tol=1e-11)
    gt, dt, nt = M[:, 0], M[:, 1], M[:, 2]
    b_delta = float(gt @ dt / (gt @ gt))
    b_n = float(gt @ nt / (gt @ gt))
    info.update({"menu_type4_products_with_gap": int(has.sum()),
                 "menu_type4_products_without_gap": int((U["delta4"].notna() & U["g"].isna()).sum()),
                 "within_slope_stored_delta_on_g": b_delta, "within_slope_nonprice_on_g": b_n,
                 "within_sd_g_pp": float(gt.std()), "within_sd_stored_delta": float(dt.std())})
    n = np.full(len(U), np.nan)
    if dgp == "stored":
        n[has] = n_prime[has] - b_n * gt
    elif dgp == "fe":
        n[has] = n_prime[has] - nt
    elif dgp == "ideal":
        c = float(np.median(n_prime[has]))
        info["ideal_constant"] = c
        n[U["g"].notna().to_numpy()] = c
    # stored type-4 products without a gap keep their stored delta
    keep_stored = (U["delta4"].notna() & U["g"].isna()).to_numpy()
    return n, keep_stored


def e4_of(U, n, keep_stored, scale=1.0):
    e4 = np.where(np.isfinite(n), np.exp(n - ALPHA_BAR * scale * np.nan_to_num(U["g"].to_numpy())), 0.0)
    e4 = np.where(keep_stored, np.exp(U["delta4"].to_numpy()), e4)
    return e4


def simulate(U, MT, SD, SD4, e4, kappa, h=1.0, details=False):
    """Equation (20) for (j,4) at every unit: the awake inflow with the stayers (A) and NF (21)."""
    ek = np.exp(-kappa)
    mt, tc = U["mt"].to_numpy(), U["tc"].to_numpy()
    nmt = len(MT)
    phi = MT["phi_mt"].to_numpy()
    O = MT["O"].to_numpy()
    W = U["W"].to_numpy()
    E = U["e1"].to_numpy() + U["e2"].to_numpy() + U["e5"].to_numpy() + e4
    S = np.bincount(mt, weights=E, minlength=nmt) + SD[MT_t(MT, U)]
    S4 = np.bincount(mt, weights=e4, minlength=nmt) + SD4[MT_t(MT, U)]
    Dj = 1 + E + ek * ((S[mt] - E) - (1 - h) * (S4[mt] - e4))
    D0 = 1 + S
    Wd = W / Dj
    T = np.bincount(mt, weights=Wd, minlength=nmt)
    awake = 1 - phi[mt]
    A = awake * e4 * (Wd + h * ek * (T[mt] - Wd) + O[mt] / D0[mt])
    NF = A - awake * U["carry4"].to_numpy()
    if not details:
        return A, NF
    return A, NF, {"Dj": Dj, "D0": D0, "S": S, "E": E, "Wd": Wd, "T": T}


_MT_T = {}


def MT_t(MT, U):
    """Quarter code of each market-quarter (cached)."""
    key = id(MT)
    if key not in _MT_T:
        t = np.zeros(len(MT), int)
        t[U["mt"].to_numpy()] = U["tc"].to_numpy()
        _MT_T[key] = t
    return _MT_T[key]


def struct_ratio(U, MT, SD, SD4, e4, kappa, K, h=1.0):
    """The ratio the simulation implies with its exact denominators, per destination: probability
    of a real of rivals' savings reaching (j,4) over that of a real of own savings, savings-weighted
    over the K1 rows with entry."""
    _, _, d = simulate(U, MT, SD, SD4, e4, kappa, h, details=True)
    mt = U["mt"].to_numpy()
    sav = U["D2"].to_numpy()
    num = np.bincount(mt, weights=sav / d["Dj"], minlength=len(MT))
    den = np.bincount(mt, weights=sav, minlength=len(MT))
    u = K["u"].to_numpy()
    riv_inv = (num[mt[u]] - sav[u] / d["Dj"][u]) / np.maximum(den[mt[u]] - sav[u], 1e-300)
    r = h * np.exp(-kappa) * d["Dj"][u] * riv_inv
    w = sav[u] * (e4[u] > 0) * (den[mt[u]] - sav[u] > 0)
    return float(np.sum(r * w) / np.sum(w)) if w.sum() > 0 else np.nan


# ----------------------------------------------------------------------------------- regressions
class Sample:
    def __init__(self, K, rows, MT):
        self.K = K[rows].reset_index(drop=True)
        self.fe = TwoWay((self.K["j"] + "|" + self.K["m"]).to_numpy(), self.K["mt"].to_numpy())
        self.cl = self.K["j"].to_numpy()
        self.MT = MT
        self.cache = {}

    def Xd(self, names, scale=1.0):
        key = (tuple(names), scale)
        if key not in self.cache:
            cols = design(self.K, self.MT, scale)
            X = np.column_stack([cols[c] for c in names])
            self.cache[key] = self.fe.demean(X)
        return self.cache[key]


def run_ols(S, NF, label, scale=1.0, controls="note"):
    u = S.K["u"].to_numpy()
    y = NF[u] / S.K["own2"].to_numpy()
    lv = list(LEVELS)
    if controls == "r4g":
        lv = lv + ["R4g"]
    elif controls == "no_outside":
        lv = [c for c in lv if c not in ("O", "Og")]
    names = SLOPES + lv
    b, V, st = ols_fe(y, S.Xd(names, scale), names, S.cl, S.fe)
    rs, ses = ratio(b, V, "R2g", "g")
    rc, sec = ratio(b, V, "R1g", "Q1g")
    ea1, sea1 = ratio(b, V, "Q1g", "g")
    ea2, sea2 = ratio(b, V, "R1g", "R2g")
    return {**label, "estimator": f"ols30_{controls}", "n": st["n"], "clusters": st["G"], "r2_within": st["r2_within"],
            "beta_w": b["g"] / scale if scale else np.nan, "beta_x": b["R2g"] / scale,
            "ratio_sav": rs, "se_sav": ses, "ratio_chk": rc, "se_chk": sec,
            "ea_own": ea1, "se_ea_own": sea1, "ea_riv": ea2, "se_ea_riv": sea2,
            "exit_over_w": b["Q4g"] / b["g"] if b["g"] else np.nan,
            "outside_over_w": (b.get("Og", np.nan) / b["g"]) if b["g"] else np.nan}


def run_ppml(S, A, label, scale=1.0, pools=PPML_POOLS):
    u = S.K["u"].to_numpy()
    y = A[u] / S.K["own2"].to_numpy()
    cols = design(S.K, S.MT, scale)
    X = np.column_stack([cols[c] for c in pools])
    a = (S.K["j"] + "|" + S.K["m"]).to_numpy()
    est, V, st = ppml_fe(y, scale * S.K["g"].to_numpy(), X, list(pools), a, S.K["mt"].to_numpy(), S.cl)
    se = {c: float(np.sqrt(max(V.loc[c, c], 0))) for c in est}
    rc, sec = ratio(est, V, "R1", "Q1") if "Q1" in est else (np.nan, np.nan)
    ea2, sea2 = ratio(est, V, "R1", "R2") if "R2" in est else (np.nan, np.nan)
    return {**label, "estimator": "ppml_awake" + ("" if "O" in pools else "_no_outside"), "n": st["n"],
            "clusters": st["G"], "r2_within": np.nan, "beta_w": est["g"] / scale, "beta_x": np.nan,
            "ratio_sav": est.get("R2", np.nan), "se_sav": se.get("R2", np.nan), "ratio_chk": rc, "se_chk": sec,
            "ea_own": est.get("Q1", np.nan), "se_ea_own": se.get("Q1", np.nan), "ea_riv": ea2, "se_ea_riv": sea2,
            "exit_over_w": est.get("Q4", np.nan), "outside_over_w": est.get("O", np.nan),
            "ppml_iterations": st["iterations"], "ppml_converged": st["converged"],
            "ppml_at_zero_bound": ";".join(st["at_zero_bound"]), "ppml_dropped_zero_groups": st["dropped_zero_groups"]}


def finish(rows, kappa_col="kappa"):
    R = pd.DataFrame(rows)
    R["exp_neg_kappa"] = np.exp(-R[kappa_col])
    R["kappa_hat_sav"] = -np.log(R["ratio_sav"].where(R["ratio_sav"] > 0))
    R["kappa_hat_chk"] = -np.log(R["ratio_chk"].where(R["ratio_chk"] > 0))
    R["bias_kappa_sav"] = R["kappa_hat_sav"] - R[kappa_col]
    return R


# ------------------------------------------------------------------------------------------- steps
def step2(ctx):
    U, MT, SD, SD4, K, N = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["K"], ctx["N"]
    rows = []
    for dgp in DGPS:
        info = {}
        n, ks = nonprice(U, dgp, info)
        e4 = e4_of(U, n, ks)
        N[f"dgp_{dgp}"] = info
        on_menu = e4[K["u"].to_numpy()] > 0
        N[f"dgp_{dgp}"]["K1_rows_on_menu"] = int(on_menu.sum())
        # on_menu: the destinations with a type-4 product under the DGP (every K1 row in `ideal`); the
        # PPML form runs there only, since a destination off the menu has an entry scale of zero
        samples = {"K1": Sample(K, np.ones(len(K), bool), MT), "on_menu": Sample(K, on_menu, MT)}
        for kappa in KAPPAS:
            A, NF = simulate(U, MT, SD, SD4, e4, kappa)
            sr = struct_ratio(U, MT, SD, SD4, e4, kappa, K)
            for sname, S in samples.items():
                lab = {"dgp": dgp, "kappa": kappa, "sample": sname, "struct_ratio": sr,
                       "ea_true": float(np.average(S.K["r1_acc"] / S.K["r2_acc"], weights=S.K["own2"]))}
                o1 = run_ols(S, NF, lab, controls="note")
                o2 = run_ols(S, NF, lab, controls="r4g")
                rows += [o1, o2]
                msg = f"step 2 {dgp} kappa {kappa} {sname}: OLS {o1['ratio_sav']:.4f} / {o1['ratio_chk']:.4f} (R4g {o2['ratio_sav']:.4f})"
                if sname == "on_menu" and kappa == KAPPA_BASE:
                    p = run_ppml(S, A, lab)
                    rows.append(p)
                    msg += f", PPML {p['ratio_sav']:.4f} / {p['ratio_chk']:.4f} (converged {p['ppml_converged']})"
                stamp(msg + f"; e^-k {np.exp(-kappa):.4f}, struct {sr:.4f}")
    R = finish(rows)
    R.to_csv(OUT / "k2_recovery.csv", index=False)
    return R


def decompose(ctx):
    """Why step 2 fails, at KAPPA_BASE: unit tests of each estimator on an outcome generated from
    its own mean (exact, then with one ingredient of the simulated world added), and the ideal
    world with its ingredients switched off one at a time: the inside mass in the denominators,
    the outside pool, the variation of phi across market-quarters."""
    U, MT, SD, SD4, K = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["K"]
    kappa, ek = KAPPA_BASE, np.exp(-KAPPA_BASE)
    rng = np.random.default_rng(SEED)
    S = Sample(K, np.ones(len(K), bool), MT)
    cols = design(K, MT)
    g = K["g"].to_numpy()
    phi = MT["phi_mt"].to_numpy()[K["mt"].to_numpy()]
    names = SLOPES + LEVELS
    Xd = S.Xd(names)
    pools = ["Q4", "Q1", "R2", "R1", "R4"]
    X = np.column_stack([cols[c] for c in pools])
    a = (K["j"] + "|" + K["m"]).to_numpy()
    ca, cb = pd.factorize(a)[0], pd.factorize(K["mt"].to_numpy())[0]
    rows = []

    def ols_row(label, y):
        b, V, st = ols_fe(y, Xd, names, S.cl, S.fe)
        rs, ses = ratio(b, V, "R2g", "g")
        rc, sec = ratio(b, V, "R1g", "Q1g")
        rows.append({"case": label, "estimator": "ols30_note", "n": st["n"], "ratio_sav": rs, "se_sav": ses,
                     "ratio_chk": rc, "se_chk": sec, "beta_w": b["g"]})
        stamp(f"decompose {label}: OLS {rs:.4f} / {rc:.4f}")

    def ppml_row(label, y, Ssub=S, pl=pools):
        u = Ssub.K
        Xs = np.column_stack([design(u, MT)[c] for c in pl])
        est, V, st = ppml_fe(y, u["g"].to_numpy(), Xs, list(pl), (u["j"] + "|" + u["m"]).to_numpy(),
                             u["mt"].to_numpy(), Ssub.cl)
        rc = est["R1"] / est["Q1"] if est.get("Q1", 0) > 0 else np.nan
        rows.append({"case": label, "estimator": "ppml", "n": st["n"], "ratio_sav": est["R2"],
                     "se_sav": float(np.sqrt(max(V.loc["R2", "R2"], 0))), "ratio_chk": rc, "beta_w": est["g"],
                     "theta_Q1": est["Q1"], "theta_O": est.get("O", np.nan), "ppml_converged": st["converged"],
                     "ppml_iterations": st["iterations"], "ppml_at_zero_bound": ";".join(st["at_zero_bound"])})
        stamp(f"decompose {label}: PPML {est['R2']:.4f} / {rc:.4f}, beta {est['g']:.4f}, converged {st['converged']}")

    lin = (1 + ek * cols["R2"] + cols["Q1"] + ek * cols["R1"]) * (1 - ALPHA_BAR * g)
    ols_row("unit: y linear in (30), constant scale", 0.01 * lin)
    ols_row("unit: y linear in (30), scale (1 - phi_mt)", (1 - phi) * lin)
    mean_ppml = np.exp(-ALPHA_BAR * g) * (1 + cols["Q4"] + cols["Q1"] + ek * (cols["R2"] + cols["R1"] + cols["R4"]))
    ppml_row("unit: y = PPML mean, constant scale", 0.01 * mean_ppml)
    fe_draw = np.exp(rng.normal(0, 1, ca.max() + 1)[ca] + rng.normal(0, 1, cb.max() + 1)[cb])
    ppml_row("unit: y = PPML mean x exp(FE_jm + FE_mt), sd 1 each", 0.01 * fe_draw * mean_ppml)
    ppml_row("unit: same x within-cell noise, sd 0.5", 0.01 * fe_draw * np.exp(rng.normal(0, 0.5, len(K))) * mean_ppml)
    n, ks = nonprice(U, "ideal", {})
    e4 = e4_of(U, n, ks)
    U0 = U.copy()
    for c in ("e1", "e2", "e5"):
        U0[c] = 0.0
    phi_med = float(np.median(MT["phi_mt"]))
    for inside, O_on, ph in ((False, False, "constant"), (False, False, "stored"), (True, False, "constant"),
                             (True, True, "constant"), (True, True, "stored")):
        M2 = MT.copy()
        if not O_on:
            M2["O"] = 0.0
        if ph == "constant":
            M2["phi_mt"] = phi_med
        Uu, SDu, SD4u = (U, SD, SD4) if inside else (U0, np.zeros_like(SD), np.zeros_like(SD4))
        A, NF = simulate(Uu, M2, SDu, SD4u, e4, kappa)
        S2 = Sample(K, np.ones(len(K), bool), M2)
        lab = f"ideal world: inside mass {'in' if inside else 'out of'} the denominators, outside pool {'on' if O_on else 'off'}, phi {ph}"
        u = K["u"].to_numpy()
        b, V, st = ols_fe(NF[u] / K["own2"].to_numpy(), S2.Xd(names), names, S2.cl, S2.fe)
        rs, ses = ratio(b, V, "R2g", "g")
        rows.append({"case": lab, "estimator": "ols30_note", "n": st["n"], "ratio_sav": rs, "se_sav": ses,
                     "ratio_chk": ratio(b, V, "R1g", "Q1g")[0], "beta_w": b["g"]})
        ppml_row(lab, A[u] / K["own2"].to_numpy(), S2, PPML_POOLS if O_on else pools)
        stamp(f"decompose {lab}: OLS {rs:.4f}")
    R = pd.DataFrame(rows)
    R["exp_neg_kappa"] = ek
    R.to_csv(OUT / "k2_recovery_decomposition.csv", index=False)
    return R


def step3(ctx):
    U, MT, SD, SD4, K, N = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["K"], ctx["N"]
    kappa = KAPPA_BASE
    n, ks = nonprice(U, "stored", {})
    e4 = e4_of(U, n, ks)
    # the bank to scale: per market, the largest lagged type-4 holder among banks on the menu there
    on = e4 > 0
    tot = U[on].groupby(["m", "j"])["D4"].sum().reset_index().sort_values(["m", "D4", "j"], ascending=[True, False, True])
    dom = tot.drop_duplicates("m").set_index("m")["j"]
    isb = (U["j"].to_numpy() == U["m"].map(dom).to_numpy()) & on
    mt = U["mt"].to_numpy()
    nmt = len(MT)
    has_b = np.bincount(mt, weights=isb.astype(float), minlength=nmt) > 0
    phi = MT["phi_mt"].to_numpy()
    O = MT["O"].to_numpy()
    Wtot = np.bincount(mt, weights=U["W"].to_numpy(), minlength=nmt) + O

    def share(Delta):
        e = np.where(isb, e4 * np.exp(Delta[mt]), e4)
        Aw, _ = simulate(U, MT, SD, SD4, e, kappa)
        return np.bincount(mt, weights=np.where(isb, Aw, 0.0), minlength=nmt) / np.maximum((1 - phi) * Wtot, 1e-300), e

    rows = []
    base_share, _ = share(np.zeros(nmt))
    for tau in (0.0,) + A1_TARGETS:
        if tau == 0.0:
            Delta = np.zeros(nmt)
        else:
            lo, hi = np.zeros(nmt), np.full(nmt, 60.0)
            for _ in range(70):
                mid = (lo + hi) / 2
                s, _ = share(mid)
                up = s < tau
                lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
            Delta = np.where(has_b & (base_share < tau) & (phi < 1), hi, 0.0)
        s, e = share(Delta)
        A, NF = simulate(U, MT, SD, SD4, e, kappa)
        sr = struct_ratio(U, MT, SD, SD4, e, kappa, K)
        ok = has_b & (phi < 1)
        info = {"target": tau, "mq_with_bank": int(ok.sum()),
                "share_achieved_median": float(np.median(s[ok])), "share_before_median": float(np.median(base_share[ok])),
                "mq_already_above": int((ok & (base_share >= tau)).sum()) if tau else 0,
                "exp_Delta_median": float(np.median(np.exp(Delta[ok]))), "exp_Delta_p90": float(np.quantile(np.exp(Delta[ok]), 0.9))}
        dest_b = isb[K["u"].to_numpy()]
        for sname, rows_sel in (("K1", np.ones(len(K), bool)), ("K1_without_scaled_bank", ~dest_b)):
            S = Sample(K, rows_sel, MT)
            lab = {"target_share": tau, "kappa": kappa, "sample": sname, "struct_ratio": sr, **{k: v for k, v in info.items() if k != "target"}}
            rows.append(run_ols(S, NF, lab, controls="note"))
            rows.append(run_ppml(S, A, lab))
            stamp(f"step 3 target {tau} {sname}: OLS {rows[-2]['ratio_sav']:.4f}, PPML {rows[-1]['ratio_sav']:.4f}, struct {sr:.4f}")
    R = finish(rows)
    R.to_csv(OUT / "k2_break_a1.csv", index=False)
    return R


def step4(ctx):
    U, MT, SD, SD4, K = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["K"]
    kappa = KAPPA_BASE
    n, ks = nonprice(U, "stored", {})
    S = Sample(K, np.ones(len(K), bool), MT)
    rows = []
    for lam in GAP_SCALES:
        e4 = e4_of(U, n, ks, scale=lam)
        A, NF = simulate(U, MT, SD, SD4, e4, kappa)
        sr = struct_ratio(U, MT, SD, SD4, e4, kappa, K)
        lab = {"gap_scale": lam, "kappa": kappa, "sample": "K1", "struct_ratio": sr,
               "alpha_g_sd": float(np.std(-ALPHA_BAR * lam * K["g"])), "alpha_g_p99": float(np.quantile(-ALPHA_BAR * lam * K["g"], 0.99))}
        rows.append(run_ols(S, NF, lab, scale=lam, controls="note"))
        rows.append(run_ols(S, NF, lab, scale=lam, controls="r4g"))
        rows.append(run_ppml(S, A, lab, scale=lam))
        stamp(f"step 4 scale {lam}: OLS {rows[-3]['ratio_sav']:.4f}, PPML {rows[-1]['ratio_sav']:.4f}")
    R = finish(rows)
    R.to_csv(OUT / "k2_break_linear.csv", index=False)
    return R


def step5(ctx):
    U, MT, SD, SD4, K = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["K"]
    n, ks = nonprice(U, "stored", {})
    e4 = e4_of(U, n, ks)
    S = Sample(K, np.ones(len(K), bool), MT)
    rows = []
    for kappa in (0.0, KAPPA_BASE):
        for h in (1.0,) + SORT_H:
            A, NF = simulate(U, MT, SD, SD4, e4, kappa, h=h)
            sr = struct_ratio(U, MT, SD, SD4, e4, kappa, K, h=h)
            lab = {"h": h, "kappa": kappa, "sample": "K1", "struct_ratio": sr, "minus_ln_h": -np.log(h)}
            rows.append(run_ols(S, NF, lab, controls="note"))
            rows.append(run_ppml(S, A, lab))
            stamp(f"step 5 kappa {kappa} h {h}: OLS {rows[-2]['ratio_sav']:.4f}, PPML {rows[-1]['ratio_sav']:.4f}")
    R = finish(rows)
    R["ln_bw_over_bx"] = R["kappa_hat_sav"]
    R["sorting_bias"] = R["ln_bw_over_bx"] - R["kappa"]
    R.to_csv(OUT / "k2_sorting.csv", index=False)
    return R


def step6(ctx):
    U, MT, SD, SD4, K = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["K"]
    n, ks = nonprice(U, "stored", {})
    e4 = e4_of(U, n, ks)
    S = Sample(K, np.ones(len(K), bool), MT)
    rows = []
    for kappa in KAPPAS:
        A, NF = simulate(U, MT, SD, SD4, e4, kappa)
        sr = struct_ratio(U, MT, SD, SD4, e4, kappa, K)
        lab = {"kappa": kappa, "sample": "K1", "struct_ratio": sr}
        rows.append(run_ols(S, NF, lab, controls="note"))
        rows.append(run_ols(S, NF, lab, controls="no_outside"))
        rows.append(run_ppml(S, A, lab))
        rows.append(run_ppml(S, A, lab, pools=[c for c in PPML_POOLS if c != "O"]))
        stamp(f"step 6 kappa {kappa}: OLS with O {rows[-4]['ratio_sav']:.4f}, without {rows[-3]['ratio_sav']:.4f}; "
              f"PPML with {rows[-2]['ratio_sav']:.4f}, without {rows[-1]['ratio_sav']:.4f}")
    R = finish(rows)
    R.to_csv(OUT / "k2_outside.csv", index=False)
    return R


def checks(ctx):
    """The simulator against the prep: at kappa = 0 with the stored delta unchanged, the awake
    inflow of (20) equals the prep's active balance (14-B) up to the pools' accrual and the phi
    cap of the shares; and the menu structure every pool faces."""
    U, MT, SD, SD4, N = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["N"]
    e4 = np.where(U["delta4"].notna(), np.exp(U["delta4"].to_numpy()), 0.0)
    A, NF, d = simulate(U, MT, SD, SD4, e4, 0.0, details=True)
    phi = MT["phi_mt"].to_numpy()[U["mt"].to_numpy()]
    ok = U["dep_act4_prep"].notna().to_numpy() & (phi <= 0.99) & (U["dep_act4_prep"].to_numpy() > 0)
    r = A[ok] / U["dep_act4_prep"].to_numpy()[ok]
    N["check_kappa0_awake_over_prep_dep_act"] = {f"p{int(q * 100):02d}": float(np.quantile(r, q)) for q in (0.01, 0.5, 0.99)}
    N["check_kappa0_n"] = int(ok.sum())
    # what a real of pool money does at kappa = KAPPA_BASE (stored DGP)
    n, ks = nonprice(U, "stored", {})
    e4s = e4_of(U, n, ks)
    _, _, d = simulate(U, MT, SD, SD4, e4s, KAPPA_BASE, details=True)
    W = U["W"].to_numpy()
    ek = np.exp(-KAPPA_BASE)
    S = d["S"][U["mt"].to_numpy()]
    stay = d["E"] / d["Dj"]
    out = 1 / d["Dj"]
    move = ek * (S - d["E"]) / d["Dj"]
    w = W > 0
    N["pool_fate_kappa_base"] = {"stay_at_own_bank": float(np.average(stay[w], weights=W[w])),
                                 "to_outside_option": float(np.average(out[w], weights=W[w])),
                                 "to_other_banks": float(np.average(move[w], weights=W[w])),
                                 "denominator_p50": float(np.quantile(d["Dj"][w], 0.5)),
                                 "denominator_p99": float(np.quantile(d["Dj"][w], 0.99)),
                                 "outside_share_D0_p05_p50_p95": [float(np.quantile(1 / d["D0"], q)) for q in (0.05, 0.5, 0.95)]}
    K = ctx["K"]
    A_, NF_ = simulate(U, MT, SD, SD4, e4s, KAPPA_BASE)
    u = K["u"].to_numpy()
    carry = K["carry4"].to_numpy()
    ms = np.abs(NF_[u])[carry > 0] / carry[carry > 0]
    N["simulated_moving_share_kappa_base"] = {f"p{int(q * 100):02d}": float(np.quantile(ms, q)) for q in (0.5, 0.9)}
    yv = NF_[u] / K["own2"].to_numpy()
    own = (1 - MT["phi_mt"].to_numpy()[K["mt"]]) * e4s[u] * d["Wd"][u]
    riv = (1 - MT["phi_mt"].to_numpy()[K["mt"]]) * e4s[u] * ek * (d["T"][K["mt"]] - d["Wd"][u])
    outs = (1 - MT["phi_mt"].to_numpy()[K["mt"]]) * e4s[u] * MT["O"].to_numpy()[K["mt"]] / d["D0"][K["mt"]]
    tot = own.sum() + riv.sum() + outs.sum()
    N["simulated_entry_composition_kappa_base"] = {"own_pools_incl_stayers": float(own.sum() / tot),
                                                   "rival_pools": float(riv.sum() / tot), "outside_pool": float(outs.sum() / tot)}
    N["simulated_y_quantiles"] = {f"p{int(q * 100):02d}": float(np.quantile(yv, q)) for q in (0.01, 0.5, 0.99)}


def parse_steps(s):
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = map(int, part.split("-"))
            out |= set(range(a, b + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="2")
    steps = parse_steps(ap.parse_args().steps)
    t0 = time.time()
    np.random.seed(SEED)
    stamp(f"start; seed {SEED} (no step draws a random number); ALPHA_BAR {ALPHA_BAR}; kappas {KAPPAS}; "
          f"A {A_AWARE}; omega {OMEGA}, lambda {LAMBDA}; steps {steps}")
    k0.COLS = k0.COLS + ["spread_a5"]
    df = k0.load_panel()
    flags = k0.rate_flags(df[df["is_B"] == 1])
    mk = k0.load_demand()
    menu, menu_rec = read_menu()
    U, MT, SD, SD4, info = build(df, flags, mk, menu)
    K, kinfo = k1_rows(U, MT, df, flags, mk)
    N = {"seed": SEED, "alpha_bar": ALPHA_BAR, "kappas": KAPPAS, "kappa_base": KAPPA_BASE, "A": A_AWARE,
         "omega": OMEGA, "lambda": LAMBDA, "panel_md5": k0.PANEL_MD5, "panel_columns": k0.COLS,
         "demand_parquet": DEMAND.name, "delta_binary": str(DELTA_BIN.relative_to(paths.PROCESSED)),
         **menu_rec, **info, **kinfo, "steps": steps}
    ctx = {"U": U, "MT": MT, "SD": SD, "SD4": SD4, "K": K, "N": N}
    stamp(f"units {len(U):,}; market-quarters {len(MT):,}; K1 rows {len(K):,}")
    checks(ctx)
    stamp(f"checks: {json.dumps({k: N[k] for k in ('check_kappa0_awake_over_prep_dep_act', 'pool_fate_kappa_base', 'simulated_entry_composition_kappa_base')}, default=str)}")
    fn = {2: step2, 3: step3, 4: step4, 5: step5, 6: step6}
    res = {}
    for s in steps:
        res[s] = fn[s](ctx)
        if s == 2:
            res["2 decomposition"] = decompose(ctx)
    path = OUT / "k2_numbers.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update({k: v for k, v in N.items()})
    old["last_run"] = {"steps": steps, "finished": f"{datetime.now():%Y-%m-%d %H:%M}", "seconds": round(time.time() - t0)}
    path.write_text(json.dumps(old, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["dgp", "kappa", "sample", "estimator", "n", "exp_neg_kappa", "struct_ratio", "ratio_sav", "se_sav",
            "ratio_chk", "se_chk", "ea_true", "ea_own", "ea_riv", "beta_w"]
    for s, R in res.items():
        print(f"\n== step {s}")
        print(R[[c for c in show + ["target_share", "gap_scale", "h", "sorting_bias"] if c in R.columns]].round(4).to_string())
    stamp(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
