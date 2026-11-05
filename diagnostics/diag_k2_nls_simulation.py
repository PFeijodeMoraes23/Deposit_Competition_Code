"""
diag_k2_nls_simulation.py
=========================
Task K2, second part (claude/kappa_handoff_2026-11-03.md, update 21:50): the estimator of the
switching-cost design (Drafts/Deposit Competition/kappa_design_2026-11-02.md, twenty-third version,
eq. (28), label eq:reg) on the simulated net flows of diag_k2_netflow_simulation.py, then K2b (the
Pix change simulated at 2020Q4) and steps 3 to 7. The simulator, the panel's pools and the K1 rows
are those of diag_k2_netflow_simulation.py, imported unchanged; this script adds the estimator and
the steps.

The estimating equation, with no clock (omega = 1, lambda = 0, so c_jt = 1 and the exit factor is 1):
    NF/Dep_j2 = e^{lambda_jm + lambda_mt} e^{-alpha g} {1 + e^a Q1 + Q4 + e^{-kappa} [R2 + e^a R1 + R4]
                + e^{a_null} O} - e^{gamma_mt} (1 + r - rho_j4,t-1) Q4 + u.
The effects are carried in levels, w_jm = e^{lambda_jm}, s_mt = e^{lambda_mt}, c_mt = e^{gamma_mt},
all kept non-negative; the structural parameters are alpha, kappa-tilde, a, a_null (`note`), or
the same with one kappa-tilde per pair (savings R2, checking R1, time deposits R4; `split`).

Estimation: nonlinear least squares by variable projection.
  - Effects, for given structural parameters: Gauss-Newton on (w_jm, s_mt, c_mt) with the exact
    least-squares solve of the effect block (every effect is nested in a market: the two
    market-quarter columns of a cell are orthonormalized in closed form, the bank x market
    columns residualized on them have a normal matrix that is block diagonal by market and is
    pseudo-inverted per market), a line search per market, and alternating non-negative least
    squares (closed form per cell and per bank) as the start and before every re-solve.
  - Form (D1 of 11-04): the per-pair form (`split`) is the estimator; the savings pair's kappa-tilde
    is the gate object and the headline; one kappa-tilde for all pairs (`note`) is step 2's record.
  - Structural parameters: Levenberg-Marquardt on the effect-partialled Jacobian, a floor on the
    damping diagonal (at kappa-tilde = 0 the bracket is a market-quarter total, so a and a_null
    carry no information there), a box (kappa-tilde in [-4, 10], a in [-8, 8], a_null in [-12, 12],
    alpha in [-3, 3]), one fresh restart of the effects on a stall, and two starts (kappa-tilde 0 and
    1.5, the rest at zero); the fit with the smaller sum of squares is kept and the two starts'
    kappa-tilde are reported.
  - Weighting. As written, (28) divides NF by own lagged savings, and the top 0.1 percent of rows
    then carry the whole sum of squares (simulated and real NF alike; k2_nls_numbers.json). Every
    row is therefore divided by its total pool T = 1 + Q1 + Q4 + R1 + R2 + R4 + O (all the money
    that can enter j's time deposits, in own-savings units): the same mean and parameters, the
    error reweighted. The unweighted fit as written is one row of the decomposition.
  - Sample: the K1 rows whose destination is on the menu under the DGP (every row in `ideal`) and
    whose market-quarter has awake money (phi-hat < 1; with phi-hat = 1 the simulated net flow is
    zero and the cell is dropped, as ppmlhdfe drops all-zero groups), then rows in market-quarter
    cells with fewer than three rows or bank x market cells with fewer than two are dropped
    iteratively (two effects per cell fit two rows exactly).
  - Errors: the cluster sandwich by conglomerate on the effect-partialled Jacobian. The simulation
    has no sampling noise, so they measure misfit, not sampling error.

Steps (--steps, default 2):
  2   recovery: kappa in KAPPAS in the three worlds (ideal, fe, stored; diag_k2_netflow_simulation),
      `note` at every kappa, `split` at KAPPA_BASE.                         k2_nls_recovery.csv
      Decomposition at KAPPA_BASE: the unit test (y generated from (28) itself with random effects,
      with and without the outside pool), the ideal world with its ingredients switched off one at
      a time (the two non-convergences of the PPML form: the outside pool, phi varying), the fe
      world's profile of the sum of squares in kappa-tilde, and the fe world fitted unweighted as
      written and on the p99-trimmed sample of the real NF.               k2_nls_decomposition.csv
  2b  K2b: kappa_mt = KAPPA0 + (KAPPA1 - KAPPA0) Post_t (and + KAPPA_U Post_t z_U,m, the Pix-use
      dose of P1), Post_t = 1{t >= 2020Q4}; the per-pair form with each pair's kappa-tilde
      k0 + dk Post (+ kU Post z_U) and alpha + alpha_P Post (+ alpha_PU Post z_U). The fe world first,
      with the own-savings weighting beside it (before-after); the gate: the savings pair's change
      within GATE_CHANGE of the truth, else the run stops; then the ideal world.   k2b_pix_change.csv
  2bs K2b in the stored world (STORED_MAXIT outer iterations), run after steps 3 to 6.
  3   A2 broken: one bank per market (its largest type-4 holder on the menu) has delta_j4 raised in
      each quarter until its time deposits take 30% and 60% of the market's awake money (the
      construction of diag_k2_netflow_simulation step 3).                         k2_break_a1.csv
  4   large gaps: the gaps multiplied by GAP_SCALES in the simulation and in the regressor; the
      linearized OLS of the earlier (30) beside it, for the record.             k2_break_linear.csv
  5   sorting: exp(delta_l4) multiplied by h for money sitting at any other bank; the bias in
      kappa-tilde against -ln h.                                                    k2_sorting.csv
  6   the outside pool left out of the estimation while it is in the simulation.    k2_outside.csv
  7   the paper's random coefficients as the data-generating process, estimated under A1.  k2_rc.csv
Steps 3 to 7 run in BASE_WORLD at KAPPA_BASE (step 5 also at kappa = 0, step 6 at every kappa), on
the per-pair form, steps 3 to 6 from step 2's per-pair fe fit as the start (k2_nls_recovery.csv).
Order of a run: 2, 2b, 3, 4, 5, 6, 2bs, 7; every output is rewritten after each fit.

Inputs (read only): those of diag_k2_netflow_simulation.py (market_panel.parquet once, through
diag_k0_netflow_objects.load_panel, with K0's columns plus spread_a5; the E3 spec-12 demand prep;
the stored logit delta or its closed form, as read_menu decides), and Rout/p1_pix_dose_exposure_mca.csv
(z_U) for K2b.
Outputs (Rout): the files above and k2_nls_numbers.json.
Usage:  python -I diagnostics/diag_k2_nls_simulation.py [--steps 2|2b|3-6|2,2b,3] [--ctx PATH] [--panel PATH]
        --ctx: a pickle of the built inputs; read if it exists, written otherwise (development).
        --panel: build the inputs from this copy of market_panel.parquet instead of the production
        path (K0's md5 pin, K0_PANEL_MD5, still applies; the side file follows DEMAND_CHARS_FILE); the
        path and the md5 read go into the numbers. Not with an existing --ctx file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils import paths
import diag_k0_netflow_objects as k0
import diag_k2_netflow_simulation as k2

# ------------------------------------------------------------------------------------- parameters
SEED = 20261104
ALPHA_BAR = k2.ALPHA_BAR          # -0.117 per pp a year: the simulated price response
KAPPAS = (0.0, 0.5, 1.0, 2.0)
KAPPA_BASE = 0.5
WORLDS = ("ideal", "fe", "stored")
BASE_WORLD = "fe"                 # steps 3-7: the world in which step 2 recovers kappa
STARTS = (0.0, 1.5)               # kappa-tilde starts in step 2 and K2b; alpha, a, a_null start at zero
ONE_START = (0.0,)                # the decomposition: one start, convergence reported
GATE_CHANGE = 0.05                # K2b: the savings pair's change in the fe world must be within this of the truth
STORED_MAXIT = 60                 # K2b in the stored world: the level fits there never converge (step 2)
KAPPA0, KAPPA1, KAPPA_U = 1.0, 0.5, -0.15   # K2b: before, after, and the slope on the dose after
POST_FROM = (2020, 4)
A1_TARGETS = k2.A1_TARGETS
GAP_SCALES = k2.GAP_SCALES
SORT_H = k2.SORT_H
PROFILE_KAPPAS = (0.3, 0.4, 0.45, 0.5, 0.6, 0.8)
P1_FILE = paths.rout_dir() / "p1_pix_dose_exposure_mca.csv"
OUT = paths.rout_dir()
# step 7: the stored random-coefficients fit and its draws (blp_engine_cpu.jl: X_COLS, D_COLS, the
# "extended" stage of build_theta2_structure, load_precomputed_draws)
RC_RESULT = paths.PROCESSED / "ESTIMATION_OUTPUT" / "BLP_RESULTS" / "cluster_processed" / "blp_E3_spec_12.json"
RC_DRAWS = paths.PROCESSED / "ESTIMATION_OUTPUT" / "BLP_DRAWS" / "demo_draws_R2000_seed42.jls"
RC_KEYS = paths.rout_dir() / "k2_rc_draw_keys.csv"       # diag_k2_export_draw_keys.jl
RC_NU = paths.rout_dir() / "k2_rc_halton_nu.csv"         # diag_k2_export_draw_keys.jl
RC_X = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5", "log_total_assets_lag", "is_state_owned"]
RC_D = ["gdp_per_capita", "fraction_65plus", "fraction_young", "pix_users_pf_per1000", "connections_per100",
        "frac_4g5g", "branches_per1000", "cadunico_families_per1000"]
RC_PI = [(1, 1), (1, 2), (1, 5), (8, 1), (2, 2), (2, 8)]   # (prod_vec column, demographic), 1-based
RC_NKEYS, RC_RALL, RC_NDIM, RC_OFFSET = 109040, 2000, 8, 24   # the array (padding row + 109,039 keys) after a 24-byte header
RC_R = 200                        # consumers per market in the DGP: the first RC_R of the paper's 2000 draws
RC_KAPPAS = (0.5, 1.0)
stamp = k0.stamp


# ------------------------------------------------------------------------------- the effect block
def informative_rows(a, b, min_b=3, min_a=2):
    """Iteratively drop rows in market-quarter cells with fewer than min_b rows and in bank x
    market cells with fewer than min_a rows."""
    a = pd.factorize(a)[0]
    b = pd.factorize(b)[0]
    keep = np.ones(len(a), bool)
    while True:
        na = np.bincount(a[keep], minlength=a.max() + 1)[a]
        nb = np.bincount(b[keep], minlength=b.max() + 1)[b]
        bad = keep & ((na < min_a) | (nb < min_b))
        if not bad.any():
            return keep
        keep &= ~bad


class Effects:
    """Least squares on the effect block of the Jacobian: column ma_i 1{a_i = a} for each bank x
    market effect, columns mb_i 1{b_i = b} and -E_i 1{b_i = b} for each market-quarter cell. The two
    cell columns are orthonormalized in closed form (Gram-Schmidt twice); the bank x market
    columns, residualized on them, have a normal matrix that is block diagonal by market (a bank
    has one row per cell), equilibrated and pseudo-inverted by eigendecomposition per market,
    batched by block size. Directions with no information (the one free level per market, an exit
    coefficient in a cell with no own time deposits) get a zero step. Rows sorted by market."""

    def __init__(self, a, b, market, rtol=1e-10):
        self.a, ua = pd.factorize(a)
        self.b, ub = pd.factorize(b)
        self.n1, self.n2, self.n = len(ua), len(ub), len(self.a)
        r = np.arange(self.n)
        chk = sp.csr_matrix((np.ones(self.n), (self.b, self.a)), shape=(self.n2, self.n1))
        if (chk.data > 1).any():
            raise ValueError("a bank appears twice in a market-quarter cell")
        self.Dat = sp.csr_matrix((np.ones(self.n), (self.a, r)), shape=(self.n1, self.n))
        mkf = pd.factorize(market)[0]
        if (np.diff(mkf) < 0).any():
            raise ValueError("rows must be sorted by market")
        self.mk, self.nm = mkf, mkf.max() + 1
        self.mk_a = pd.Series(mkf).groupby(self.a).first().reindex(range(self.n1)).to_numpy()
        self.mk_b = pd.Series(mkf).groupby(self.b).first().reindex(range(self.n2)).to_numpy()
        order = np.argsort(self.mk_a, kind="stable")
        cnt = np.bincount(self.mk_a)
        starts = np.r_[0, np.cumsum(cnt)[:-1]]
        self.blocks = {int(size): np.stack([order[starts[mm]:starts[mm] + size] for mm in np.flatnonzero(cnt == size)])
                       for size in np.unique(cnt[cnt > 0])}
        self.rtol = rtol

    def prepare(self, ma, mb, E):
        b, n2 = self.b, self.n2
        self.ma, self.mb, self.e = ma, mb, -E
        e = self.e
        r11 = np.sqrt(np.bincount(b, mb * mb, n2))
        q1 = mb / np.where(r11 > 0, r11, 1.0)[b]
        r12 = np.bincount(b, q1 * e, n2)
        ep = e - r12[b] * q1
        c = np.bincount(b, q1 * ep, n2)
        ep -= c[b] * q1
        r12 += c
        r22 = np.sqrt(np.bincount(b, ep * ep, n2))
        en = np.sqrt(np.bincount(b, e * e, n2))
        ok2 = (en > 0) & (r22 > 1e-9 * en)
        self.q1 = q1
        self.q2 = np.where(ok2[b], ep / np.where(r22 > 0, r22, 1.0)[b], 0.0)
        self.r11, self.r12, self.r22, self.ok2 = r11, r12, np.where(ok2, r22, 0.0), ok2
        self.Uu = sp.csr_matrix((ma * self.q1, (self.b, self.a)), shape=(self.n2, self.n1))
        self.Ww = sp.csr_matrix((ma * self.q2, (self.b, self.a)), shape=(self.n2, self.n1))
        S = (sp.diags(self.Dat @ (ma * ma)) - (self.Uu.T @ self.Uu) - (self.Ww.T @ self.Ww)).tocsr()
        self.Sinv = {}
        for size, idx in self.blocks.items():
            k = idx.shape[0]
            flat = idx.ravel()
            blk = S[flat][:, flat].toarray().reshape(k, size, k, size)[np.arange(k), :, np.arange(k), :]
            dg = np.sqrt(np.maximum(np.einsum("kii->ki", blk), 0))
            live = dg > 1e-150 * max(dg.max(), 1e-300)
            dgs = np.where(live, dg, 1.0)
            Bq = blk / dgs[:, :, None] / dgs[:, None, :]
            Bq = np.where(live[:, :, None] & live[:, None, :], Bq, 0.0)
            Bq = np.nan_to_num(0.5 * (Bq + np.transpose(Bq, (0, 2, 1))), nan=0.0, posinf=0.0, neginf=0.0)
            lam, Q = np.linalg.eigh(Bq)
            top = np.maximum(lam[:, -1:], 1e-300)
            inv = np.where(lam > self.rtol * top, 1.0 / np.where(lam > 0, lam, 1.0), 0.0)
            self.Sinv[size] = (np.einsum("kij,kj,klj->kil", Q, inv, Q), dgs, live)

    def _cellsum(self, w, V):
        return np.stack([np.bincount(self.b, w * V[:, j], self.n2) for j in range(V.shape[1])], axis=1)

    def project(self, V):
        """Least squares of each column of V (n x k) on the prepared effect block: the fitted part
        and the coefficients (on the bank x market columns, the cell entry and exit columns)."""
        V = V[:, None] if V.ndim == 1 else V
        k = V.shape[1]
        t1, t2 = self._cellsum(self.q1, V), self._cellsum(self.q2, V)
        AtV = self.Dat @ (self.ma[:, None] * V) - self.Uu.T @ t1 - self.Ww.T @ t2
        xa = np.zeros((self.n1, k))
        for size, idx in self.blocks.items():
            Pq, dgs, live = self.Sinv[size]
            rhs = np.where(live[:, :, None], AtV[idx] / dgs[:, :, None], 0.0)
            xa[idx] = np.einsum("kil,klj->kij", Pq, rhs) / dgs[:, :, None]
        Ax = self.ma[:, None] * xa[self.a]
        w = V - Ax
        s1, s2 = self._cellsum(self.q1, w), self._cellsum(self.q2, w)
        beta2 = np.where(self.ok2[:, None], s2 / np.where(self.r22 > 0, self.r22, 1.0)[:, None], 0.0)
        beta1 = (s1 - self.r12[:, None] * beta2) / np.where(self.r11 > 0, self.r11, 1.0)[:, None]
        fit = Ax + self.mb[:, None] * beta1[self.b] + self.e[:, None] * beta2[self.b]
        return fit, xa, beta1, beta2


def als_effects(y, h, E, fe, w, sweeps=200, tol=1e-12):
    """Alternating least squares on the effects with the scales kept non-negative: for given w_jm,
    each cell's (s_mt, c_mt) is a two-variable non-negative least squares in closed form; for given
    (s_mt, c_mt), each w_jm is a one-variable one. The sum of squares never rises."""
    b, a, n2 = fe.b, fe.a, fe.n2
    E2 = np.bincount(b, E * E, n2)
    Ey = np.bincount(b, -E * y, n2)
    prev = np.inf
    for k in range(sweeps):
        x = w[a] * h
        s11 = np.bincount(b, x * x, n2)
        s12 = np.bincount(b, -x * E, n2)
        t1 = np.bincount(b, x * y, n2)
        det = s11 * E2 - s12 ** 2
        ok = det > 1e-12 * np.maximum(s11 * E2, 1e-300)
        sb = np.where(ok, (E2 * t1 - s12 * Ey) / np.where(ok, det, 1), 0.0)
        cb = np.where(ok, (s11 * Ey - s12 * t1) / np.where(ok, det, 1), 0.0)
        s_only = np.maximum(t1 / np.maximum(s11, 1e-300), 0.0)
        c_only = np.where(E2 > 0, np.maximum(Ey / np.maximum(E2, 1e-300), 0.0), 0.0)
        bad = ~ok | (sb < 0) | (cb < 0)
        use_s = (-2 * t1 * s_only + s11 * s_only ** 2) <= (-2 * Ey * c_only + E2 * c_only ** 2)
        sb = np.where(bad, np.where(use_s, s_only, 0.0), sb)
        cb = np.where(bad, np.where(use_s, 0.0, c_only), cb)
        xs = sb[b] * h
        num = fe.Dat @ (xs * (y + cb[b] * E))
        den = fe.Dat @ (xs * xs)
        w = np.where(den > 0, np.maximum(num / np.where(den > 0, den, 1), 0.0), w)
        r = y - (w[a] * sb[b] * h - cb[b] * E)
        ssr = r @ r
        if prev - ssr < tol * max(prev, 1e-300) and k > 3:
            break
        prev = ssr
    return w, sb, cb, k + 1


class Model:
    """The bracket of (28) and its price index, h = exp(-g (Za @ alpha)) B.
       B = 1 + e^a Q1 + Q4 + e^{-k} (R2 + e^a R1 + R4) + e^{a_null} O, k = Zk @ kappa (Zk: 1, Post,
       Post z_U); `split` (the estimator's form, D1 of 11-04): one kappa-tilde per pair, each with the
       same Zk interactions, B = 1 + e^a Q1 + Q4 + e^{-k} R2 + e^{a - k_chk} R1 + e^{-k_td} R4
       + e^{a_null} O; `outside` False drops the O term."""

    def __init__(self, cols, g, Za=None, Zk=None, za_names=("alpha",), zk_names=("kappa",), outside=True, split=False):
        self.c, self.g = cols, g
        n = len(g)
        self.Za = np.ones((n, 1)) if Za is None else Za
        self.Zk = np.ones((n, 1)) if Zk is None else Zk
        self.za_names, self.zk_names = list(za_names), list(zk_names)
        self.outside, self.split = outside, split
        pairs = ([f"{n}_chk" for n in self.zk_names] + [f"{n}_td" for n in self.zk_names]) if split else []
        self.psi_names = self.zk_names + pairs + ["a"] + (["a_null"] if outside else [])
        self.names = self.za_names + self.psi_names
        self.p = len(self.names)

    def start(self, kappa0=0.0):
        s = np.zeros(self.p)
        nz = len(self.za_names)
        nk = len(self.zk_names)
        s[nz] = kappa0
        if self.split:
            s[nz + nk] = s[nz + 2 * nk] = kappa0
        return s

    def box(self):
        lo, hi = [], []
        for nm in self.names:
            b = (-3.0, 3.0) if nm.startswith("alpha") else (-4.0, 10.0) if nm.startswith("kappa") else \
                (-8.0, 8.0) if nm == "a" else (-12.0, 12.0)
            lo.append(b[0]), hi.append(b[1])
        return np.array(lo), np.array(hi)

    def h(self, s):
        """h and dh/ds (n x p)."""
        nz = len(self.za_names)
        psi = s[nz:]
        c = self.c
        nk = len(self.zk_names)
        ek = np.exp(-(self.Zk @ psi[:nk]))
        if self.split:
            e1 = np.exp(-(self.Zk @ psi[nk:2 * nk]))
            e4 = np.exp(-(self.Zk @ psi[2 * nk:3 * nk]))
            ea = np.exp(psi[3 * nk])
            B = 1.0 + ea * c["Q1"] + c["Q4"] + ek * c["R2"] + e1 * ea * c["R1"] + e4 * c["R4"]
            parts = [(-ek * c["R2"])[:, None] * self.Zk, (-e1 * ea * c["R1"])[:, None] * self.Zk,
                     (-e4 * c["R4"])[:, None] * self.Zk, (ea * c["Q1"] + e1 * ea * c["R1"])[:, None]]
            j0 = 3 * nk + 1
        else:
            ea = np.exp(psi[nk])
            Rsum = c["R2"] + ea * c["R1"] + c["R4"]
            B = 1.0 + ea * c["Q1"] + c["Q4"] + ek * Rsum
            parts = [(-ek * Rsum)[:, None] * self.Zk, (ea * c["Q1"] + ek * ea * c["R1"])[:, None]]
            j0 = nk + 1
        if self.outside:
            e0 = np.exp(psi[j0])
            B = B + e0 * c["O"]
            parts.append((e0 * c["O"])[:, None])
        pr = np.exp(np.clip(-self.g * (self.Za @ s[:nz]), -700, 700))
        dh = np.column_stack([(-self.g * pr * B)[:, None] * self.Za, pr[:, None] * np.column_stack(parts)])
        return pr * B, dh


def nls_fit(y, E, a, b, market, cl, model, s0, rowscale=None, box=None, maxit=200, tol_cos=1e-9, tol_step=1e-9,
            inner_max=200, inner_tol=1e-13):
    """Nonlinear least squares on (28) by variable projection (module docstring). Returns the
    estimates, the cluster sandwich on the effect-partialled Jacobian, and diagnostics."""
    fe = Effects(a, b, market)
    mk, nm = fe.mk, fe.nm
    n = len(y)
    d = np.ones(n) if rowscale is None else np.asarray(rowscale, float)
    y = np.asarray(y, float) * d
    ysc = np.mean(np.abs(y))
    y = y / ysc
    E = np.asarray(E, float) * d
    yy = y @ y
    steps = 0.5 ** np.arange(12)

    def hfun(s_):
        h_, dh_ = model.h(s_)
        return h_ * d, dh_ * d[:, None]

    def inner(h_, w_, sb_, cb_, sweeps=10):
        if sweeps:
            w_, sb_, cb_, _ = als_effects(y, h_, E, fe, w_.copy(), sweeps=sweeps)
        r_ = y - (w_[fe.a] * sb_[fe.b] * h_ - cb_[fe.b] * E)
        ssr_m = np.bincount(mk, r_ * r_, nm)
        if not np.isfinite(ssr_m.sum()):
            return w_, sb_, cb_, r_, np.inf, 0
        for k in range(inner_max):
            fe.prepare(sb_[fe.b] * h_, w_[fe.a] * h_, E)
            _, xa, x1, x2 = fe.project(r_)
            dw, dsb, dcb = (np.nan_to_num(v[:, 0], nan=0.0, posinf=0.0, neginf=0.0) for v in (xa, x1, x2))
            best = ssr_m.copy()
            tb = np.zeros(nm)
            for t in steps:
                rt = y - (np.maximum(w_ + t * dw, 0)[fe.a] * np.maximum(sb_ + t * dsb, 0)[fe.b] * h_
                          - np.maximum(cb_ + t * dcb, 0)[fe.b] * E)
                sm = np.bincount(mk, rt * rt, nm)
                better = np.isfinite(sm) & (sm < best)
                best = np.where(better, sm, best)
                tb = np.where(better, t, tb)
            w_ = np.maximum(w_ + tb[fe.mk_a] * dw, 0.0)
            sb_ = np.maximum(sb_ + tb[fe.mk_b] * dsb, 0.0)
            cb_ = np.maximum(cb_ + tb[fe.mk_b] * dcb, 0.0)
            r_ = y - (w_[fe.a] * sb_[fe.b] * h_ - cb_[fe.b] * E)
            new = np.bincount(mk, r_ * r_, nm)
            dec = (ssr_m.sum() - new.sum()) / max(ssr_m.sum(), 1e-300)
            ssr_m = new
            if dec < inner_tol or ssr_m.sum() <= 1e-28 * yy:
                break
        return w_, sb_, cb_, r_, float(ssr_m.sum()), k + 1

    s = np.asarray(s0, float).copy()
    h, dh = hfun(s)
    w, sb, cb, n_als = als_effects(y, h, E, fe, np.ones(fe.n1))
    w, sb, cb, r, ssr, inner_total = inner(h, w, sb, cb, sweeps=0)
    lam = 1e-3
    conv, reason, last_step, cos, it, restarts = False, "maxit", np.inf, np.nan, -1, 0
    for it in range(maxit):
        Js = (w[fe.a] * sb[fe.b])[:, None] * dh
        fe.prepare(sb[fe.b] * h, w[fe.a] * h, E)
        fit, *_ = fe.project(Js)
        Jt = Js - fit
        gvec = Jt.T @ r
        cos = float(np.max(np.abs(gvec) / np.maximum(np.sqrt((Jt * Jt).sum(axis=0)) * np.sqrt(ssr), 1e-300)))
        if ssr <= 1e-26 * yy:
            conv, reason = True, "exact fit"
            break
        if cos < tol_cos or (last_step < tol_step and cos < 1e-6):
            conv, reason = True, "stationary"
            break
        H = Jt.T @ Jt
        dH = np.maximum(np.diag(H), 1e-6 * max(np.diag(H).max(), 1e-300))
        accepted = False
        for _ in range(14):
            try:
                ds = np.linalg.solve(H + lam * np.diag(dH), gvec)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            if box is not None:
                ds = np.clip(s + ds, box[0], box[1]) - s
            h_n, dh_n = hfun(s + ds)
            out = inner(h_n, w, sb, cb)
            inner_total += out[-1]
            if np.isfinite(out[4]) and out[4] < ssr:
                accepted = True
                break
            lam *= 10
        if not accepted:
            wf, sbf, cbf, _ = als_effects(y, h, E, fe, np.ones(fe.n1))
            fresh = inner(h, wf, sbf, cbf, sweeps=0)
            inner_total += fresh[-1]
            restarts += 1
            if np.isfinite(fresh[4]) and fresh[4] < ssr * (1 - 1e-9) and restarts <= 5:
                w, sb, cb, r, ssr, _ = fresh
                lam = 1e-3
                continue
            reason = "no descent step"
            conv = cos < 1e-5
            break
        last_step = float(np.max(np.abs(ds) / np.maximum(np.abs(s), 1e-1)))
        rel = (ssr - out[4]) / max(ssr, 1e-300)
        s, h, dh = s + ds, h_n, dh_n
        w, sb, cb, r, ssr, _ = out
        lam = max(lam / 10, 1e-10)
        if rel < 1e-15 and last_step < tol_step:
            conv, reason = True, "no further decrease"
            break
    Js = (w[fe.a] * sb[fe.b])[:, None] * dh
    fe.prepare(sb[fe.b] * h, w[fe.a] * h, E)
    fit, *_ = fe.project(Js)
    Jt = Js - fit
    Hm = Jt.T @ Jt
    cg = pd.factorize(cl)[0]
    G = cg.max() + 1
    Sg = np.zeros((G, model.p))
    np.add.at(Sg, cg, Jt * r[:, None])
    Hi = np.linalg.pinv(Hm)
    V = G / (G - 1) * Hi @ (Sg.T @ Sg) @ Hi
    dn = np.sqrt(np.maximum(np.diag(Hm), 1e-300))
    ev = np.linalg.eigvalsh(Hm / np.outer(dn, dn))
    at_box = [] if box is None else [nm_ for nm_, v, lo, hi in zip(model.names, s, box[0], box[1]) if v <= lo + 1e-9 or v >= hi - 1e-9]
    diag = {"n": int(n), "G": int(G), "markets": int(fe.nm), "cells": int(fe.n2), "bank_markets": int(fe.n1),
            "iterations": it + 1, "inner_iterations": int(inner_total), "converged": bool(conv), "reason": reason,
            "ssr_over_yy": float(ssr / yy), "gradient_cosine": float(cos),
            "hessian_condition": float(ev.max() / max(ev.min(), 1e-300)), "at_box": at_box, "restarts": int(restarts),
            "zero_exit_cells": int((cb <= 0).sum()), "zero_entry_cells": int((sb <= 0).sum()), "zero_banks": int((w <= 0).sum()),
            "exit_coef": cb * ysc}
    return dict(zip(model.names, s)), pd.DataFrame(V, index=model.names, columns=model.names), diag


def nls_multistart(y, E, a, b, market, cl, model, starts=STARTS, rowscale=None, box=None, maxit=200):
    """nls_fit from each kappa-tilde start; the fit with the smaller sum of squares, with the
    starts' kappa-tilde and sums of squares."""
    box = model.box() if box is None else box
    fits = [nls_fit(y, E, a, b, market, cl, model, model.start(k0), rowscale=rowscale, box=box, maxit=maxit) for k0 in starts]
    ssr = [f[2]["ssr_over_yy"] for f in fits]
    i = int(np.argmin(ssr))
    est, V, d = fits[i]
    kn = model.names[len(model.za_names)]
    ks = [float(f[0][kn]) for f in fits]
    d = {**d, "start_kappas": ks, "start_ssr": ssr, "start_converged": [f[2]["converged"] for f in fits],
         "starts_spread": float(max(ks) - min(ks)), "best_start": starts[i]}
    return est, V, d


# ------------------------------------------------------------------------------------- the samples
def world_e4(ctx, world, scale=1.0):
    n, ks = k2.nonprice(ctx["U"], world, {})
    return k2.e4_of(ctx["U"], n, ks, scale=scale)


def sample(ctx, e4, extra=None):
    """The estimation rows (module docstring) for a world with type-4 utilities e4."""
    K, MT = ctx["K"], ctx["MT"]
    phi = MT["phi_mt"].to_numpy()
    keep = (e4[K["u"].to_numpy()] > 0) & (phi[K["mt"].to_numpy()] < 1 - 1e-12)
    if extra is not None:
        keep &= extra
    if ctx.get("markets") is not None:
        keep &= K["m"].isin(ctx["markets"]).to_numpy()
    Kw = K[keep]
    a = (Kw["j"] + "|" + Kw["m"]).to_numpy()
    Kw = Kw[informative_rows(a, Kw["mt"].to_numpy())].sort_values(["m", "qi", "j"]).reset_index(drop=True)
    cols = k2.design(Kw, MT)
    T = 1 + cols["Q1"] + cols["Q4"] + cols["R1"] + cols["R2"] + cols["R4"] + cols["O"]
    return {"K": Kw, "cols": cols, "E": (Kw["carry4"] / Kw["own2"]).to_numpy(), "g": Kw["g"].to_numpy(),
            "a": (Kw["j"] + "|" + Kw["m"]).to_numpy(), "b": Kw["mt"].to_numpy(), "m": Kw["m"].to_numpy(),
            "cl": Kw["j"].to_numpy(), "d": 1 / T, "u": Kw["u"].to_numpy(), "own2": Kw["own2"].to_numpy()}


def fit_row(S, y, label, outside=True, split=False, scale=1.0, Za=None, Zk=None, za_names=("alpha",),
            zk_names=("kappa",), weight="pool", starts=STARTS, box=None, s0=None, maxit=200):
    """One NLS fit and its row. s0 (a dict of estimates) is a warm start, used instead of `starts`."""
    t0 = time.time()
    model = Model(S["cols"], scale * S["g"], Za=Za, Zk=Zk, za_names=za_names, zk_names=zk_names, outside=outside, split=split)
    rs = S["d"] if weight == "pool" else None
    if s0 is None and len(starts) > 1:
        est, V, d = nls_multistart(y, S["E"], S["a"], S["b"], S["m"], S["cl"], model, starts=starts, rowscale=rs, box=box,
                                   maxit=maxit)
    else:
        start = model.start(starts[0]) if s0 is None else np.array([s0.get(nm, 0.0) for nm in model.names], float)
        est, V, d = nls_fit(y, S["E"], S["a"], S["b"], S["m"], S["cl"], model, start, rowscale=rs,
                            box=model.box() if box is None else box, maxit=maxit)
        d = {**d, "start_kappas": [float(est[model.names[len(model.za_names)]])], "start_ssr": [d["ssr_over_yy"]],
             "start_converged": [d["converged"]], "starts_spread": 0.0,
             "best_start": "warm: step-2 per-pair fe fit" if s0 is not None else starts[0]}
    se = {k: float(np.sqrt(max(V.loc[k, k], 0))) for k in est}
    row = {**label, "model": ("split" if split else "note") + ("" if outside else "_no_outside"), "weight": weight}
    for k, v in est.items():
        row[k], row[f"se_{k}"] = float(v), se[k]
    for k in ("n", "G", "markets", "cells", "bank_markets", "converged", "reason", "iterations", "inner_iterations",
              "gradient_cosine", "hessian_condition", "ssr_over_yy", "restarts", "zero_exit_cells", "zero_entry_cells",
              "zero_banks", "starts_spread", "best_start"):
        row[k] = d[k]
    row["at_box"] = ";".join(d["at_box"])
    row["start_kappas"] = ";".join(f"{v:.6f}" for v in d["start_kappas"])
    row["start_converged"] = ";".join(str(v) for v in d["start_converged"])
    row["seconds"] = round(time.time() - t0, 1)
    if "kappa" in est:
        row["exp_neg_kappa_hat"] = float(np.exp(-est["kappa"]))
    stamp(f"  {label}: {row['model']} " + ", ".join(f"{k} {v:.4f}" for k, v in est.items())
          + f"; converged {d['converged']} ({d['reason']}), starts {row['start_kappas']}, {row['seconds']}s")
    return row


def truth(S, kappa, extra=None):
    """Simulation truths for the row labels: e^-kappa, the accrual ratio of checking to savings
    (the a of the simulation), and the A2 ratio the simulator implies with its exact denominators."""
    K = S["K"]
    ea = float(np.average(K["r1_acc"] / K["r2_acc"], weights=K["own2"]))
    out = {"kappa_true": kappa, "exp_neg_kappa": float(np.exp(-kappa)), "a_true": float(np.log(ea)), "alpha_true": ALPHA_BAR}
    if extra:
        out.update(extra)
    return out


def finish(rows):
    R = pd.DataFrame(rows)
    if "kappa" in R and "kappa_true" in R:
        R["bias_kappa"] = R["kappa"] - R["kappa_true"]
        R["bias_exp_neg_kappa"] = R["exp_neg_kappa_hat"] - R["exp_neg_kappa"]
    return R


# ------------------------------------------------------------------------------------------- steps
def step2(ctx):
    U, MT, SD, SD4, N = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["N"]
    rows = []
    for world in WORLDS:
        e4 = world_e4(ctx, world)
        S = sample(ctx, e4)
        N[f"nls_sample_{world}"] = {"rows": int(len(S["K"])), "markets": int(len(np.unique(S["m"]))),
                                    "cells": int(len(np.unique(S["b"]))), "clusters": int(len(np.unique(S["cl"])))}
        for kappa in KAPPAS:
            A, NF = k2.simulate(U, MT, SD, SD4, e4, kappa)
            y = NF[S["u"]] / S["own2"]
            lab = {"world": world, **truth(S, kappa, {"struct_ratio": k2.struct_ratio(U, MT, SD, SD4, e4, kappa, S["K"])})}
            r = fit_row(S, y, lab)
            ctx["cache"][(world, kappa, "note")] = r
            rows.append(r)
            if kappa == KAPPA_BASE:
                rows.append(fit_row(S, y, lab, split=True))
    R = finish(rows)
    R.to_csv(OUT / "k2_nls_recovery.csv", index=False)
    return R


def decompose(ctx):
    """At KAPPA_BASE: the unit test, the ideal world's ingredients one at a time, the fe world's
    profile in kappa-tilde, and the fe world unweighted as written and on the p99-trimmed sample."""
    U, MT, SD, SD4 = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"]
    kappa, ek = KAPPA_BASE, np.exp(-KAPPA_BASE)
    rng = np.random.default_rng(SEED)
    rows = []
    # unit test: y from (28) itself, random effects, on the ideal world's rows
    e4i = world_e4(ctx, "ideal")
    S = sample(ctx, e4i)
    c = S["cols"]
    ca, cb = pd.factorize(S["a"])[0], pd.factorize(S["b"])[0]
    la = rng.normal(0, 1, ca.max() + 1)[ca]
    lb = rng.normal(-4, 1, cb.max() + 1)[cb]
    cx = (0.01 * np.exp(rng.normal(0, 0.5, cb.max() + 1)))[cb]
    for O_on in (False, True):
        B = 1 + c["Q1"] + c["Q4"] + ek * (c["R2"] + c["R1"] + c["R4"]) + (c["O"] if O_on else 0.0)
        y = np.exp(la + lb - ALPHA_BAR * S["g"]) * B - cx * S["E"]
        lab = {"case": f"unit: y = (28) exactly, random effects, outside pool {'in' if O_on else 'out'}", **truth(S, kappa)}
        rows.append(fit_row(S, y, {**lab, "a_true": 0.0}, outside=O_on))
    # the ideal world, ingredients switched off one at a time
    U0 = U.copy()
    for col in ("e1", "e2", "e5"):
        U0[col] = 0.0
    phi_med = float(np.median(MT["phi_mt"]))
    for inside, O_on, ph in ((False, False, "constant"), (False, False, "stored"), (True, False, "constant"),
                             (True, True, "constant"), (True, True, "stored")):
        M2 = MT.copy()
        if not O_on:
            M2["O"] = 0.0
        if ph == "constant":
            M2["phi_mt"] = phi_med
        Uu, SDu, SD4u = (U, SD, SD4) if inside else (U0, np.zeros_like(SD), np.zeros_like(SD4))
        A, NF = k2.simulate(Uu, M2, SDu, SD4u, e4i, kappa)
        ctx2 = {**ctx, "MT": M2}
        S2 = sample(ctx2, e4i)
        y = NF[S2["u"]] / S2["own2"]
        lab = {"case": f"ideal world: inside mass {'in' if inside else 'out of'} the denominators, outside pool "
                       f"{'on' if O_on else 'off'}, phi {ph}", **truth(S2, kappa)}
        rows.append(fit_row(S2, y, lab, outside=O_on, starts=ONE_START))
    # the fe world: profile in kappa-tilde, the note's weighting, the p99 trim
    e4f = world_e4(ctx, "fe")
    S = sample(ctx, e4f)
    A, NF = k2.simulate(U, MT, SD, SD4, e4f, kappa)
    y = NF[S["u"]] / S["own2"]
    for kp in PROFILE_KAPPAS:
        model = Model(S["cols"], S["g"])
        lo, hi = model.box()
        lo[1] = hi[1] = kp
        rows.append(fit_row(S, y, {"case": f"fe world: profile, kappa-tilde fixed at {kp}", **truth(S, kappa)}, box=(lo, hi), starts=(kp,)))
    rows.append(fit_row(S, y, {"case": "fe world: unweighted, NF over own savings as written", **truth(S, kappa)}, weight="none",
                        starts=ONE_START))
    yr = np.abs(S["K"]["NF"].to_numpy() / S["own2"])
    trim = yr <= np.quantile(np.abs(ctx["K"]["NF"].to_numpy() / ctx["K"]["own2"].to_numpy()), 0.99)
    St = sample(ctx, e4f, extra=np.isin(ctx["K"]["u"].to_numpy(), S["u"][trim]))
    yt = NF[St["u"]] / St["own2"]
    rows.append(fit_row(St, yt, {"case": "fe world: rows with |real NF / own savings| above its p99 dropped", **truth(St, kappa)},
                        starts=ONE_START))
    R = finish(rows)
    R.to_csv(OUT / "k2_nls_decomposition.csv", index=False)
    return R


def simulate_mt(U, MT, SD, SD4, e4, kappa_mt, h=1.0):
    """k2.simulate with a switching cost per market-quarter (kappa_mt, aligned with MT)."""
    ek = np.exp(-np.asarray(kappa_mt, float))
    mt = U["mt"].to_numpy()
    nmt = len(MT)
    phi, O, W = MT["phi_mt"].to_numpy(), MT["O"].to_numpy(), U["W"].to_numpy()
    E = U["e1"].to_numpy() + U["e2"].to_numpy() + U["e5"].to_numpy() + e4
    tq = k2.MT_t(MT, U)
    S = np.bincount(mt, weights=E, minlength=nmt) + SD[tq]
    S4 = np.bincount(mt, weights=e4, minlength=nmt) + SD4[tq]
    Dj = 1 + E + ek[mt] * ((S[mt] - E) - (1 - h) * (S4[mt] - e4))
    Wd = W / Dj
    T = np.bincount(mt, weights=Wd, minlength=nmt)
    awake = 1 - phi[mt]
    A = awake * e4 * (Wd + h * ek[mt] * (T[mt] - Wd) + O[mt] / (1 + S)[mt])
    return A, A - awake * U["carry4"].to_numpy()


def k2b_world(ctx, world, maxit=200, weights=("pool",)):
    """K2b in one world, on the per-pair form: each pair's kappa-tilde is k0 + dk Post (+ kU Post z_U),
    alpha is alpha + alpha_P Post (+ alpha_PU Post z_U). The rows accumulate in ctx["k2b_rows"] and the
    CSV is rewritten after every fit."""
    U, MT, SD, SD4, N = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["N"]
    q_post = POST_FROM[0] * 4 + POST_FROM[1] - 1
    mt_q = U.groupby("mt")["qi"].first().reindex(range(len(MT))).to_numpy()
    post_mt = (mt_q >= q_post).astype(float)
    z_mt = MT["m"].map(ctx["zU"]).to_numpy(dtype=float)
    N["k2b_markets_without_zU"] = int(pd.Series(MT.loc[np.isnan(z_mt), "m"]).nunique())
    rows = ctx.setdefault("k2b_rows", [])
    e4 = world_e4(ctx, world)
    for version in ("before_after", "dose"):
        if version == "before_after":
            kap = KAPPA0 + (KAPPA1 - KAPPA0) * post_mt
            extra = None
        else:
            kap = KAPPA0 + (KAPPA1 - KAPPA0) * post_mt + KAPPA_U * post_mt * np.nan_to_num(z_mt)
            extra = np.isfinite(z_mt[ctx["K"]["mt"].to_numpy()])
        A, NF = simulate_mt(U, MT, SD, SD4, e4, kap)
        S = sample(ctx, e4, extra=extra)
        y = NF[S["u"]] / S["own2"]
        post = post_mt[S["b"]]
        if version == "before_after":
            Z = np.column_stack([np.ones(len(y)), post])
            zn_a, zn_k = ("alpha", "alpha_P"), ("kappa", "dkappa")
        else:
            Z = np.column_stack([np.ones(len(y)), post, post * z_mt[S["b"]]])
            zn_a, zn_k = ("alpha", "alpha_P", "alpha_PU"), ("kappa", "dkappa", "kappa_U")
        lab = {"world": world, "version": version, "kappa0_true": KAPPA0, "dkappa_true": KAPPA1 - KAPPA0,
               "kappa_U_true": KAPPA_U if version == "dose" else 0.0, "alpha_true": ALPHA_BAR, "alpha_P_true": 0.0,
               "post_share_rows": float(post.mean()), "kappa_post_min": float(np.min(kap[S["b"]][post > 0])),
               "kappa_post_max": float(np.max(kap[S["b"]][post > 0]))}
        for wt in weights:
            if wt == "none" and version == "dose":
                continue
            rows.append(fit_row(S, y, lab, split=True, Za=Z, Zk=Z, za_names=zn_a, zk_names=zn_k, weight=wt,
                                starts=STARTS if wt == "pool" else ONE_START, maxit=maxit))
            write_k2b(rows)
    return [r for r in rows if r["world"] == world]


def write_k2b(rows):
    R = pd.DataFrame(rows)
    for pr in ("", "_chk", "_td"):
        if f"kappa{pr}" in R:
            R[f"bias_kappa0{pr}"] = R[f"kappa{pr}"] - R["kappa0_true"]
            R[f"bias_dkappa{pr}"] = R[f"dkappa{pr}"] - R["dkappa_true"]
        if f"kappa_U{pr}" in R:
            R[f"bias_kappa_U{pr}"] = R[f"kappa_U{pr}"] - R["kappa_U_true"]
    R.to_csv(OUT / "k2b_pix_change.csv", index=False)
    return R


def step2b(ctx):
    """K2b in the fe world first (the gate: the savings pair's change within GATE_CHANGE of the
    truth, before-after version, total-pool weighting), with the own-savings weighting beside it,
    then the ideal world. The stored world is step 2bs."""
    N = ctx["N"]
    fe_rows = k2b_world(ctx, "fe", weights=("pool", "none"))
    ba = [r for r in fe_rows if r["version"] == "before_after" and r["weight"] == "pool"][0]
    err = abs(ba["dkappa"] - (KAPPA1 - KAPPA0))
    N["k2b_gate"] = {"world": "fe", "dkappa_savings": ba["dkappa"], "true": KAPPA1 - KAPPA0, "abs_error": err,
                     "threshold": GATE_CHANGE, "passed": bool(err <= GATE_CHANGE)}
    stamp(f"K2b gate (fe, savings pair, before-after): change {ba['dkappa']:.4f} against {KAPPA1 - KAPPA0}, "
          f"error {err:.4f}: {'passed' if err <= GATE_CHANGE else 'FAILED'}")
    if err > GATE_CHANGE:
        raise GateFailed(f"K2b gate failed: the savings pair's change in the fe world is {ba['dkappa']:.4f}")
    k2b_world(ctx, "ideal")
    return write_k2b(ctx["k2b_rows"])


def step2bs(ctx):
    k2b_world(ctx, "stored", maxit=STORED_MAXIT)
    return write_k2b(ctx["k2b_rows"])


class GateFailed(SystemExit):
    pass


def base_start(ctx):
    """The warm start for steps 3-6: the per-pair fit of the fe world at KAPPA_BASE (step 2's row)."""
    rec = OUT / "k2_nls_recovery.csv"
    R = pd.read_csv(rec)
    r = R[(R["world"] == BASE_WORLD) & (R["kappa_true"] == KAPPA_BASE) & (R["model"] == "split")]
    if not len(r):
        raise SystemExit(f"{rec.name} has no per-pair {BASE_WORLD} row at kappa {KAPPA_BASE}: run step 2 first")
    r = r.iloc[0].to_dict()
    return {k: float(r[k]) for k in ("alpha", "kappa", "kappa_chk", "kappa_td", "a", "a_null")}, r


def step3(ctx):
    U, MT, SD, SD4 = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"]
    kappa = KAPPA_BASE
    e4 = world_e4(ctx, BASE_WORLD)
    on = e4 > 0
    tot = U[on].groupby(["m", "j"])["D4"].sum().reset_index().sort_values(["m", "D4", "j"], ascending=[True, False, True])
    dom = tot.drop_duplicates("m").set_index("m")["j"]
    isb = (U["j"].to_numpy() == U["m"].map(dom).to_numpy()) & on
    mt = U["mt"].to_numpy()
    nmt = len(MT)
    has_b = np.bincount(mt, weights=isb.astype(float), minlength=nmt) > 0
    phi, O = MT["phi_mt"].to_numpy(), MT["O"].to_numpy()
    Wtot = np.bincount(mt, weights=U["W"].to_numpy(), minlength=nmt) + O

    def share(Delta):
        e = np.where(isb, e4 * np.exp(Delta[mt]), e4)
        Aw, _ = k2.simulate(U, MT, SD, SD4, e, kappa)
        return np.bincount(mt, weights=np.where(isb, Aw, 0.0), minlength=nmt) / np.maximum((1 - phi) * Wtot, 1e-300), e

    base_share, _ = share(np.zeros(nmt))
    rows = []
    for tau in (0.0,) + A1_TARGETS:
        if tau == 0.0:
            Delta = np.zeros(nmt)
        else:
            lo, hi = np.zeros(nmt), np.full(nmt, 60.0)
            for _ in range(70):
                mid = (lo + hi) / 2
                s_, _ = share(mid)
                up = s_ < tau
                lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
            Delta = np.where(has_b & (base_share < tau) & (phi < 1), hi, 0.0)
        s_, e = share(Delta)
        A, NF = k2.simulate(U, MT, SD, SD4, e, kappa)
        ok = has_b & (phi < 1)
        dest_b = isb[ctx["K"]["u"].to_numpy()]
        for sname, extra in (("K1", None), ("K1_without_scaled_bank", ~dest_b)):
            S = sample(ctx, e, extra=extra)
            y = NF[S["u"]] / S["own2"]
            info = {"target_share": tau, "sample": sname, "world": BASE_WORLD, "mq_with_bank": int(ok.sum()),
                    "share_achieved_median": float(np.median(s_[ok])), "share_before_median": float(np.median(base_share[ok])),
                    "exp_Delta_median": float(np.median(np.exp(Delta[ok]))),
                    "struct_ratio": k2.struct_ratio(U, MT, SD, SD4, e, kappa, S["K"])}
            rows.append(fit_row(S, y, {**info, **truth(S, kappa)}, split=True, s0=base_start(ctx)[0]))
            finish(rows).to_csv(OUT / "k2_break_a1.csv", index=False)
    R = finish(rows)
    R.to_csv(OUT / "k2_break_a1.csv", index=False)
    return R


def step4(ctx):
    U, MT, SD, SD4, K = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["K"]
    kappa = KAPPA_BASE
    rows = []
    for lam in GAP_SCALES:
        e4 = world_e4(ctx, BASE_WORLD, scale=lam)
        A, NF = k2.simulate(U, MT, SD, SD4, e4, kappa)
        S = sample(ctx, e4)
        y = NF[S["u"]] / S["own2"]
        lab = {"gap_scale": lam, "world": BASE_WORLD, "alpha_g_sd": float(np.std(-ALPHA_BAR * lam * S["g"])),
               "alpha_g_p99": float(np.quantile(-ALPHA_BAR * lam * S["g"], 0.99)), **truth(S, kappa)}
        r = fit_row(S, y, lab, scale=lam, split=True, s0=base_start(ctx)[0])
        # the linearized OLS of the earlier (30), on the same rows, for the record
        Sk = k2.Sample(K, np.isin(K["u"].to_numpy(), S["u"]), MT)
        o = k2.run_ols(Sk, NF, {}, scale=lam, controls="note")
        r.update({"ols30_ratio_sav": o["ratio_sav"], "ols30_se_sav": o["se_sav"], "ols30_ratio_chk": o["ratio_chk"],
                  "ols30_n": o["n"]})
        rows.append(r)
        finish(rows).to_csv(OUT / "k2_break_linear.csv", index=False)
    R = finish(rows)
    R.to_csv(OUT / "k2_break_linear.csv", index=False)
    return R


def step5(ctx):
    U, MT, SD, SD4 = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"]
    e4 = world_e4(ctx, BASE_WORLD)
    S = sample(ctx, e4)
    rows = []
    for kappa in (0.0, KAPPA_BASE):
        for h in (1.0,) + SORT_H:
            A, NF = k2.simulate(U, MT, SD, SD4, e4, kappa, h=h)
            y = NF[S["u"]] / S["own2"]
            lab = {"h": h, "minus_ln_h": float(-np.log(h)), "world": BASE_WORLD,
                   "struct_ratio": k2.struct_ratio(U, MT, SD, SD4, e4, kappa, S["K"], h=h), **truth(S, kappa)}
            rows.append(fit_row(S, y, lab, split=True, s0=base_start(ctx)[0]))
            finish(rows).to_csv(OUT / "k2_sorting.csv", index=False)
    R = finish(rows)
    R["kappa_tilde_true"] = R["kappa_true"] - np.log(R["h"])
    R["bias_vs_kappa_tilde"] = R["kappa"] - R["kappa_tilde_true"]
    R.to_csv(OUT / "k2_sorting.csv", index=False)
    return R


def step6(ctx):
    U, MT, SD, SD4 = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"]
    e4 = world_e4(ctx, BASE_WORLD)
    S = sample(ctx, e4)
    rows = []
    for kappa in KAPPAS:
        A, NF = k2.simulate(U, MT, SD, SD4, e4, kappa)
        y = NF[S["u"]] / S["own2"]
        lab = {"world": BASE_WORLD, **truth(S, kappa)}
        s0, row2 = base_start(ctx)
        if kappa == KAPPA_BASE:
            rows.append({**row2, "source": "step 2 (k2_nls_recovery.csv)"})
        else:
            rows.append(fit_row(S, y, lab, split=True, s0=s0))
        rows.append(fit_row(S, y, lab, split=True, outside=False, s0=s0))
        finish(rows).to_csv(OUT / "k2_outside.csv", index=False)
    R = finish(rows)
    R.to_csv(OUT / "k2_outside.csv", index=False)
    return R


def rc_inputs(ctx):
    """theta_2 of the stored fit, the draws (memory map, standardized and centered over every real
    key and all 2000 draws as load_precomputed_draws does), the nu draws, and the product design
    (spread in pp and X_COLS, row-aligned with the menu) with each product's draw key."""
    res = json.loads(RC_RESULT.read_text(encoding="utf-8"))
    if res.get("stage") != "extended" or list(res.get("sigma_indices", [])) != [1] or \
            [tuple(x) for x in res.get("pi_interactions", [])] != RC_PI:
        raise SystemExit(f"{RC_RESULT.name}: stage {res.get('stage')}, sigma {res.get('sigma_indices')}, "
                         f"pi {res.get('pi_interactions')}: not the structure step 7 implements")
    th2 = np.array(res["theta2_sigma_pi"], float)
    if not (RC_KEYS.exists() and RC_NU.exists()):
        raise SystemExit("run: julia --startup-file=no diagnostics/diag_k2_export_draw_keys.jl <BLP_DRAWS> <Rout>")
    if RC_DRAWS.stat().st_size != RC_OFFSET + RC_NKEYS * RC_RALL * RC_NDIM * 8:
        raise SystemExit(f"{RC_DRAWS.name}: size {RC_DRAWS.stat().st_size} is not the (109040, 2000, 8) array")
    mm = np.memmap(RC_DRAWS, dtype="<f8", mode="r", offset=RC_OFFSET, shape=(RC_NKEYS, RC_RALL, RC_NDIM), order="F")
    if float(np.abs(mm[0, :, :]).max()) != 0.0:
        raise SystemExit("row 1 of the draws is not the zero padding row: layout not as assumed")
    mean, sd = np.zeros(RC_NDIM), np.ones(RC_NDIM)
    for d in range(RC_NDIM):
        s1 = s2 = 0.0
        for r0 in range(0, RC_RALL, 100):
            blk = np.asarray(mm[1:, r0:r0 + 100, d])
            s1 += blk.sum()
            s2 += (blk * blk).sum()
        nobs = (RC_NKEYS - 1) * RC_RALL
        mean[d] = s1 / nobs
        v = (s2 - nobs * mean[d] ** 2) / (nobs - 1)
        sd[d] = np.sqrt(v) if v > 1e-20 else 1.0
    nu = pd.read_csv(RC_NU, header=None).to_numpy(float)
    keys = pd.read_csv(RC_KEYS, dtype={"mca_code": str, "time_id": str})
    kmap = dict(zip(zip(keys["mca_code"], keys["time_id"]), keys["idx"] - 1))
    import pyarrow.parquet as pq
    pr = pq.read_table(k2.DEMAND, columns=["mca_code", "time_id", "spread_ann"] + RC_X).to_pandas()
    menu, _ = k2.read_menu()
    if len(pr) != len(menu):
        raise SystemExit("prep rows and menu rows differ")
    prod = np.column_stack([pr["spread_ann"].to_numpy(float) / 100.0] + [pr[c].to_numpy(float) for c in RC_X])
    if not np.isfinite(prod).all():
        raise SystemExit("missing spread or X_COLS in the prep")
    key = np.array([kmap.get((m, t), 0) for m, t in zip(pr["mca_code"].astype(str), pr["time_id"].astype(str))])
    return {"theta2": th2, "mm": mm, "mean": mean, "sd": sd, "nu": nu, "kmap": kmap, "prod": prod, "key": key,
            "menu": menu, "B_rows_on_a_key": float((key[menu["is_B"].to_numpy(bool)] > 0).mean())}


def rc_simulate(ctx, rc, e4_base, kappa, theta2=None, n_draws=None):
    """(20) with the paper's random coefficients: consumers r = 1..n_draws in each market, the
    holders of every pool distributed as the market's consumers (no sorting), each draw's
    utilities delta + mu_r on the whole local menu (the market's B products and the quarter's D
    products, mu_r from the draw of that market-quarter), the flows averaged over the draws. The
    type-4 B utilities are the base world's (e4_base); every other product keeps its stored delta."""
    U, MT = ctx["U"], ctx["MT"]
    n_draws = RC_R if n_draws is None else n_draws
    th = rc["theta2"] if theta2 is None else theta2
    sig, pis = th[0], th[1:]
    menu, prod, key = rc["menu"], rc["prod"], rc["key"]
    isB = menu["is_B"].to_numpy(bool)
    k = menu["k"].to_numpy()
    # each B product's unit in U
    mb = menu[isB]
    qi = mb["t"].str[:4].astype(int).to_numpy() * 4 + mb["t"].str[-1].astype(int).to_numpy() - 1
    ukey = pd.MultiIndex.from_arrays([U["j"], U["m"], U["qi"]])
    uidx = ukey.get_indexer(pd.MultiIndex.from_arrays([mb["j"].to_numpy(), mb["m"].to_numpy(), qi]))
    delta_B = mb["delta"].to_numpy().copy()
    kB = k[isB]
    t4 = (kB == 4) & (uidx >= 0)
    e4u = e4_base[np.maximum(uidx, 0)]
    keep = (uidx >= 0) & ~((kB == 4) & (e4u <= 0))
    delta_B[t4 & (e4u > 0)] = np.log(e4u[t4 & (e4u > 0)])
    prodB, keyB = prod[isB], key[isB]
    # D products by quarter; each market-quarter's draw key
    md = menu[~isB]
    prodD, deltaD, kD = prod[~isB], md["delta"].to_numpy(), k[~isB]
    t_u = pd.Index(U.groupby("tc")["t"].first().sort_index().to_numpy())
    tD = t_u.get_indexer(md["t"].to_numpy())
    mt_t = k2.MT_t(MT, U)
    mt_key = np.array([rc["kmap"].get((m, t), 0) for m, t in zip(MT["m"], MT["t"])])
    nU, nmt = len(U), len(MT)
    mt = U["mt"].to_numpy()
    W, carry4 = U["W"].to_numpy(), U["carry4"].to_numpy()
    phi, O = MT["phi_mt"].to_numpy(), MT["O"].to_numpy()
    awake = 1 - phi[mt]
    ek = np.exp(-kappa)
    A_sum, NF_sum = np.zeros(nU), np.zeros(nU)
    for r in range(n_draws):
        Dt = (np.asarray(rc["mm"][:, r, :]) - rc["mean"]) / rc["sd"]
        Dt[0] = 0.0
        muB = sig * rc["nu"][r, 0] * prodB[:, 0]
        for (c, dd), p in zip(RC_PI, pis):
            muB = muB + p * prodB[:, c - 1] * Dt[keyB, dd - 1]
        ex = np.where(keep, np.exp(delta_B + muB), 0.0)
        u_ = np.maximum(uidx, 0)
        e = {kk: np.bincount(u_, weights=np.where(kB == kk, ex, 0.0), minlength=nU) for kk in (1, 2, 4, 5)}
        # D products: utility of each for the consumer of each market-quarter
        SD, SD4 = np.zeros(nmt), np.zeros(nmt)
        for tq in range(len(t_u)):
            cells = np.flatnonzero(mt_t == tq)
            dsel = np.flatnonzero(tD == tq)
            if len(cells) == 0 or len(dsel) == 0:
                continue
            P = prodD[dsel]
            Dc = Dt[mt_key[cells]]
            mu = sig * rc["nu"][r, 0] * P[None, :, 0] + np.zeros((len(cells), 1))
            for (c, dd), p in zip(RC_PI, pis):
                mu = mu + p * Dc[:, dd - 1][:, None] * P[None, :, c - 1]
            exD = np.exp(deltaD[dsel][None, :] + mu)
            SD[cells] = exD.sum(axis=1)
            SD4[cells] = exD[:, kD[dsel] == 4].sum(axis=1)
        Ej = e[1] + e[2] + e[5] + e[4]
        S = np.bincount(mt, weights=Ej, minlength=nmt) + SD
        Dj = 1 + Ej + ek * (S[mt] - Ej)
        Wd = W / Dj
        T = np.bincount(mt, weights=Wd, minlength=nmt)
        A = awake * e[4] * (Wd + ek * (T[mt] - Wd) + O[mt] / (1 + S)[mt])
        A_sum += A
        NF_sum += A - awake * carry4
    return A_sum / n_draws, NF_sum / n_draws


def step7(ctx):
    U, MT, SD, SD4, N = ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], ctx["N"]
    rc = rc_inputs(ctx)
    N["rc"] = {"theta2": rc["theta2"].tolist(), "draws_used": RC_R, "demo_mean": rc["mean"].tolist(),
               "demo_sd": rc["sd"].tolist(), "B_products_on_a_draw_key": rc["B_rows_on_a_key"]}
    e4 = world_e4(ctx, BASE_WORLD)
    # the simulator with theta_2 = 0 is k2.simulate: a check on the bookkeeping
    A0, NF0 = rc_simulate(ctx, rc, e4, KAPPA_BASE, theta2=np.zeros_like(rc["theta2"]), n_draws=1)
    Ab, NFb = k2.simulate(U, MT, SD, SD4, e4, KAPPA_BASE)
    err = float(np.max(np.abs(A0 - Ab)) / max(np.max(np.abs(Ab)), 1e-300))
    N["rc"]["check_theta2_zero_vs_base_max_rel"] = err
    stamp(f"step 7: theta_2 = 0 reproduces the base simulator to {err:.1e}")
    if err > 1e-8:
        raise SystemExit("the RC simulator with theta_2 = 0 does not reproduce k2.simulate")
    S = sample(ctx, e4)
    base = {}
    rec = OUT / "k2_nls_recovery.csv"
    if rec.exists():
        R2 = pd.read_csv(rec)
        for kp in RC_KAPPAS:
            r = R2[(R2["world"] == BASE_WORLD) & (R2["kappa_true"] == kp) & (R2["model"] == "note")]
            if len(r):
                base[kp] = r.iloc[0].to_dict()
    rows = []
    for kp in RC_KAPPAS:
        A, NF = rc_simulate(ctx, rc, e4, kp)
        Ab, NFb = k2.simulate(U, MT, SD, SD4, e4, kp)
        u = S["u"]
        ratio = A[u] / np.where(Ab[u] > 0, Ab[u], np.nan)
        y = NF[u] / S["own2"]
        lab = {"world": BASE_WORLD, "dgp": f"random coefficients (theta_2 of {RC_RESULT.name}, {RC_R} draws)",
               "rc_over_base_inflow_p10": float(np.nanquantile(ratio, 0.1)), "rc_over_base_inflow_p50": float(np.nanquantile(ratio, 0.5)),
               "rc_over_base_inflow_p90": float(np.nanquantile(ratio, 0.9)), **truth(S, kp)}
        rows.append(fit_row(S, y, lab, starts=ONE_START))
        if kp in base:
            rows.append({**base[kp], "dgp": "logit (step 2, same world)"})
        else:
            yb = NFb[u] / S["own2"]
            rows.append(fit_row(S, yb, {"world": BASE_WORLD, "dgp": "logit (refit)", **truth(S, kp)}, starts=ONE_START))
    R = finish(rows)
    R.to_csv(OUT / "k2_rc.csv", index=False)
    return R


# -------------------------------------------------------------------------------------------- main
def build_ctx(panel=None):
    """The inputs, from one panel read. panel: a copy of market_panel.parquet to read instead of
    K0's production path; K0's md5 pin and its load_demand time check apply to the copy."""
    if panel is not None:
        k0.PANEL = Path(panel)
    k0.COLS = k0.COLS + ["spread_a5"]
    df = k0.load_panel()
    panel_rec = {"panel_file": str(k0.PANEL), "panel_md5_read": k0.md5(k0.PANEL)} if panel is not None else {}
    flags = k0.rate_flags(df[df["is_B"] == 1])
    mk = k0.load_demand()
    menu, menu_rec = k2.read_menu()
    U, MT, SD, SD4, info = k2.build(df, flags, mk, menu)
    K, kinfo = k2.k1_rows(U, MT, df, flags, mk)
    p1 = pd.read_csv(P1_FILE, dtype={"mca_code": str})
    zU = p1.set_index("mca_code")["z_U"]
    return {"U": U, "MT": MT, "SD": SD, "SD4": SD4, "K": K, "zU": zU,
            "N0": {"panel_md5": k0.PANEL_MD5, **panel_rec, "panel_columns": k0.COLS, **menu_rec, **info, **kinfo}}


def parse_steps(s):
    out = []
    for part in s.split(","):
        if part == "2b":
            out.append("2b")
        elif "-" in part:
            a, b = map(int, part.split("-"))
            out += [str(i) for i in range(a, b + 1)]
        else:
            out.append(part)
    order = ["2", "2b", "3", "4", "5", "6", "2bs", "7"]
    return [s for s in order if s in out]


def main():
    global OUT, RC_R
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="2")
    ap.add_argument("--ctx", default=None)
    ap.add_argument("--markets", type=int, default=0, help="smoke test: a random subset of markets")
    ap.add_argument("--out", default=None, help="output folder (default Rout)")
    ap.add_argument("--rc-draws", type=int, default=RC_R, help="step 7: consumers per market (smoke tests use fewer)")
    ap.add_argument("--panel", default=None, help="build the inputs from this copy of market_panel.parquet")
    args = ap.parse_args()
    if args.panel and args.ctx and Path(args.ctx).exists():
        raise SystemExit(f"--panel applies to the build of the inputs, but {args.ctx} exists and would be read instead")
    if args.panel and not Path(args.panel).is_file():
        raise SystemExit(f"--panel: no file {args.panel}")
    RC_R = args.rc_draws
    if args.out:
        OUT = Path(args.out)
        OUT.mkdir(parents=True, exist_ok=True)
    steps = parse_steps(args.steps)
    t0 = time.time()
    stamp(f"start; seed {SEED}; alpha {ALPHA_BAR}; kappas {KAPPAS}; worlds {WORLDS}; base world {BASE_WORLD}; "
          f"K2b {KAPPA0} -> {KAPPA1}, kappa_U {KAPPA_U}; steps {steps}")
    if args.ctx and Path(args.ctx).exists():
        with open(args.ctx, "rb") as fh:
            ctx = pickle.load(fh)
        stamp(f"inputs from {args.ctx}")
    else:
        ctx = build_ctx(args.panel)
        if args.ctx:
            with open(args.ctx, "wb") as fh:
                pickle.dump(ctx, fh)
    zU_cov = ctx["K"]["m"].isin(ctx["zU"].index).mean()
    ctx["N"] = {"seed": SEED, "alpha_bar": ALPHA_BAR, "kappas": KAPPAS, "kappa_base": KAPPA_BASE, "worlds": WORLDS,
                "base_world": BASE_WORLD, "starts": STARTS, "k2b": {"kappa0": KAPPA0, "kappa1": KAPPA1, "kappa_U": KAPPA_U,
                                                                     "post_from": POST_FROM},
                "K1_rows_with_zU": float(zU_cov), **ctx["N0"], "steps": steps}
    ctx["cache"] = {}
    N = ctx["N"]
    if args.markets:
        ctx["markets"] = set(pd.Series(ctx["K"]["m"].unique()).sample(args.markets, random_state=3))
        N["smoke_test_markets"] = args.markets
    # the weighting: share of the sum of squares in the top rows, own-savings normalization vs pool
    K = ctx["K"]
    yr = K["NF"].to_numpy() / K["own2"].to_numpy()
    cols = k2.design(K, ctx["MT"])
    T = 1 + cols["Q1"] + cols["Q4"] + cols["R1"] + cols["R2"] + cols["R4"] + cols["O"]
    for lab, v in (("own_savings", yr), ("pool", yr / T)):
        s2 = np.sort(v[np.isfinite(v)] ** 2)[::-1]
        N[f"real_NF_sum_sq_share_top_0p1pct_{lab}"] = float(s2[:len(s2) // 1000].sum() / s2.sum())
        N[f"real_NF_sum_sq_share_top_1pct_{lab}"] = float(s2[:len(s2) // 100].sum() / s2.sum())
    fn = {"2": step2, "2b": step2b, "3": step3, "4": step4, "5": step5, "6": step6, "2bs": step2bs, "7": step7}
    res = {}
    path = OUT / "k2_nls_numbers.json"
    for s in steps:
        stop = False
        try:
            res[s] = fn[s](ctx)
        except GateFailed as e:
            stamp(str(e) + "; the remaining steps are not run")
            stop = True
        if s == "2":
            res["2 decomposition"] = decompose(ctx)
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        old.update(N)
        old["last_run"] = {"steps": steps, "last_step_done": s, "finished": f"{datetime.now():%Y-%m-%d %H:%M}",
                           "seconds": round(time.time() - t0)}
        path.write_text(json.dumps(old, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")
        if stop:
            break
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["world", "case", "version", "kappa_true", "model", "n", "alpha", "kappa", "dkappa", "kappa_U", "kappa_chk",
            "kappa_td", "a", "a_null", "exp_neg_kappa", "exp_neg_kappa_hat", "struct_ratio", "converged", "starts_spread",
            "target_share", "gap_scale", "h", "ols30_ratio_sav"]
    for s, R in res.items():
        print(f"\n== step {s}")
        print(R[[c for c in show if c in R.columns]].round(4).to_string())
    stamp(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
