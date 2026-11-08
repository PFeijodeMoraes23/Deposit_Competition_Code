"""
kappa/k1_netflow_nls.py
=======================
K1 of the switching-cost design: the net-flow equation eq:reg of Drafts/Deposit Competition/
kappa_design_2026-11-02.md (claude/review_fix_spec_2026-11-05.md sections 3 and 4, with the 19:45
amendment) on the observed net flows, by nonlinear least squares. Decisions: Pedro, 11-05 12:30
(claude/MAIN_REPLY_2026-11-05a.md, Kappa), D1 and D2 of 15:58, and 19:45 (claude/MAIN_REPLY_2026-11-05.md).

The equation, one kappa-tilde per source pair (savings R2, the headline; checking R1 + R4m; time
deposits R4d), pools lagged and divided by own lagged savings Dep_j2 as in K0:
    NF/Dep_j2 = e^{psi_jm + psi_mt} e^{-g (Za alpha) + Xe beta}
                  {1 + e^{a + Zq theta} (Q1 + Q4m) + Q4d + e^{-k} R2 + e^{a - k_chk} (R1 + R4m)
                   + e^{-k_td} R4d + e^{a_null} O}
                - (1 + r - rho_j4,t-1) [e^{theta_w} Q4m + e^{vartheta_mt} Q4d] + u.
    The savings forfeit is not in the model (it falls on every use of the money, the outside option
    included, and cancels under A2): every source has scale one, there is no source-type exit cost, and
    a is a free nuisance scale for checking money only. Q4m = omega_jt Q4 (own matured money: it lands in
    checking at j, so it has the checking scale and the sweep shifters), Q4d = (1 - omega_jt) lambda_jt Q4
    (own daily-liquidity money: staying pays no clock, so weight 1 and no E_c), R4m = sum_{j' != j}
    omega_j't Dep_j'4 / Dep_j2 (rivals' matured money: checking money, in the checking pair), R4d =
    sum_{j' != j} (1 - omega_j't) lambda_j't Dep_j'4 / Dep_j2 (rivals' daily-liquidity money: it pays the
    clock, so e^{-k_td} = e^{-kappa} E_c). Locked money, (1 - omega)(1 - lambda), neither leaves nor
    re-enters. k = Zk kappa for each pair; Xe holds the control-function residual v (cf_v) and, in the A6
    test, the scale bins; Zq the default-sweep indicators. The matured exit coefficient e^{theta_w} is one constant (1 in the model,
    estimated and reported as a check); the daily-liquidity exit coefficient e^{vartheta_mt} is free by
    market-quarter (the effect block's exit column) and drops where Q4d = 0. --exit-by-cell (the headline,
    Pedro 11-06; the constant theta_w is reported as its check): no theta_w, one market-quarter exit
    coefficient on the whole movable stock (Q4m + Q4d), which where Q4d = 0 is theta_w free by market-quarter.
The clock: there is no E_c parameter. Under A4, ln E_c = kappa (savings) - kappa_td, read after the fit
in the lambda = 1 row only (--lambda-one, D2). In the headline, lambda from the statements is about
zero, so Q4d = R4d = 0 below DAILY_MIN and the time-deposit pair is not estimable: it is dropped, and the
output says so.

Estimation: variable projection as in diagnostics/diag_k2_nls_simulation.py, whose effect block
(kn.Effects, kn.als_effects) is imported unchanged. nls_fit_k1 is kn.nls_fit's Levenberg-Marquardt loop
with one addition, the constant matured exit term: for given parameters the effects are solved on
y + e^{theta_w} E_m, and theta_w's Jacobian column is -e^{theta_w} E_m, which the effects do not
multiply. Without theta_w it is kn.nls_fit step for step (test 1). Four kappa-tilde starts (0, 0.5, 1,
1.5) and an iteration cap of 500 (--starts, --maxit; Pedro 11-06), the fit with the smallest sum of
squares kept and every start's end point reported; every row is divided by its total pool
1 + Q1 + Q4 + R1 + R2 + R4 + O. --trim p99 (the headline, the 11-03 left-side decision) drops the rows
with |NF/Dep_j2| above the 99th percentile of each spec's rows; untrimmed is the robustness row.

Inputs.
  - The K1 rows of diag_k2_netflow_simulation.k1_rows (K0's net flow and pool weights, own lagged
    savings > 0, a gap, the outside pool non-negative), from a context built by kn.build_ctx on the
    panel and its demand frame, or read from a --ctx pickle. Real NF; no phi-hat in the outcome.
  - The panel (pyarrow, column subsets): rate_a4_fallback at t and t-1 (decision 3: the headline drops
    rows with a carried or fallback rate; `carried_kept` keeps them), the national type-4 spread and the
    cost shifters utils.demand_chars.COST_IV for the control function.
  - omega_jt and lambda_jt: Rout/k0_omega.csv (K0 (e); rerun on the final panel, without judicial
    deposits). omega is the share maturing in the quarter; automatic reinvestment is banned in the
    window (Res. CMN 3.454/2007, 5.005/2022), so matured money lands in checking. The statement of year y
    applies to the quarters of year y + 1, 2016 quarters take 2016. lambda = share_no_maturity for B
    banks with a statement, the year's B median of it otherwise (about zero), 1 for D banks;
    --lambda-one sets lambda = 1 for every bank (D2).
  - Control function: the conglomerate-quarter type-4 spread (spread_ann_a4_national, pp) on COST_IV
    with quarter effects, weighted by the type-4 stock, on the quarters without a carried rate (P3's
    cost_fit, measured version); v = the residual, in the exponent (`cf_v`).
  - Pix: Post = 1{t >= 2020Q4}; U-hat, the fitted value of P1's first stage (z_U on z_mobile and z_g45
    jointly, with 2019 CadUnico and log 2019 population as controls, one row per MCA, 2019 population
    weights), plugged in; the two controls enter beside it (Post x control in kappa-tilde and alpha).
  - Stakes: S_tilde of Rout/p3_stakes_mca_quarter.csv (P3, same panel md5), standardized on the rows;
    it shifts kappa-tilde and alpha only (the stakes enter phi through tau_1, step S).
  - Advertising: P2's adv_brl_real_2024q4 on the observed quarters only (P4's statuses; negatives
    clipped at zero; `no_negative_adv` drops those quarters), ln(1 + adv), with separate slopes for P2's
    2019 big five and for the others.
  - Default sweeps on the own-checking term (Q1 + Q4m): Santander from 2022Q2 (B1), BB's opt-in from
    2022Q1 (BB3).

Specs (--spec, comma-separated; `name+sample` applies a robustness sample to a spec):
  level, before_after (headline), dose, pretrend, placebo_2018q4, placebo_2019q4, stakes, advertising,
  sweep, a6_bins; samples: zero_premium, carried_kept, no_bb_caixa, no_dominant, p99, no_one_bank, no_cf,
  no_negative_adv.
--savings-only (pilot, Pedro 11-07; off by default): the 11-03 outline's savings pair, "equation (33) with
the savings pools only (Q1 = R1 = 0)": the own and rivals' checking and matured pools (Q1, Q4m, R1, R4m) leave
the bracket, so a and the checking pair (kappa_chk and its shifters) are not estimated and their columns are
absent; the bracket keeps 1 + Q4d + e^{-k} R2 (+ e^{-k_td} R4d) + e^{a_null} O and the exit term is unchanged.
--matured-scale (pilot, Pedro 11-06; off by default): own matured money gets its own scale,
e^{a + Zq theta} Q1 + e^{a_m + Zq theta} Q4m in the own bracket (a roll-over re-applied at j is a stay in
substance); rivals' matured money R4m keeps the checking scale, since it leaves a rival like rival checking.
--smooth-effects (K1-S, Pedro 11-07; off by default): the effects are exp(psi), so positivity is built in;
for given parameters they are solved by Gauss-Newton in psi (kn.Effects on the log-tangent columns), with a
line search by market, warm-started from the previous solution, to a tight tolerance. An effect whose log
falls below PSI_MIN is set to zero (floored), and the exit coefficient is capped at C_MAX (data units): in
a cell whose entry and exit columns are collinear the two effects otherwise grow without bound and cancel.
An effect at either bound leaves the step and the parameters' projection, and comes off it when that lowers
the sum of squares; floored_effects and capped_exit_cells count them. The bank x market effects are
re-centred within each component after every step, every effect of the component shifting (the fit is
unchanged). A step with no descent ends the fit, with no fresh ALS restart. The profiled objective is then continuous and
differentiable in the parameters, and Jt' r is its gradient (SmoothProfile, nls_fit_smooth).
Reported solutions (Pedro, 11-06): INTERIOR, the lowest-SSR start with no parameter at its box, and
BOUNDARY, the lowest-SSR start overall when that start has a box hit; one output row each (column
`solution`), each with its own inference, restricted fits and convergence tail (grad_norm_end, cos_end,
ssr_rel_change_last100).
Inference: the cluster sandwich by destination conglomerate and the score wild cluster bootstrap
(Kline and Santos 2012) on the converged fit's own effect-partialled Jacobian and residual: Webb
weights, symmetric percentile-t intervals; tests with the null imposed (the restricted fit, then the
efficient-score statistic). G and the effective number of clusters G* = G / (1 + CV^2) of the
per-cluster row counts (utils.cluster_stats.effective_cluster_stats, the repo's single source) are
reported as descriptive statistics, as in the other WCB tables.

Guard. Real-data fits need the final panel (FINAL_PANEL), a demand frame stamped with it and K0 rerun
on it (k0_numbers.json); any other input runs only as a smoke test (--allow-interim, at most 100
markets). Tests: --test 1 (nls_fit_k1 with K2's Model on simulated y equals step 2's per-pair fe fit of
k2_nls_recovery.csv to 1e-8, standard errors included); --test 2 (y generated from the equation above
with known parameters, lambda > 0: recovered to 1e-6 from a zero start).

Usage: python -I kappa/k1_netflow_nls.py --test 1 --ctx <k2n_ctx_full.pkl>
       python -I kappa/k1_netflow_nls.py --test 2 --ctx <ctx.pkl>
       python -I kappa/k1_netflow_nls.py --spec before_after --ctx <ctx.pkl> --panel <panel> --markets 50
              --allow-interim [--lambda-one] [--exit-by-cell] [--trim p99] --out <dir>
       python -I kappa/k1_netflow_nls.py --spec before_after,dose --ctx <ctx.pkl> --panel <panel>
              --exit-by-cell --trim p99   (final inputs, the headline form)
Outputs (--out, default Rout/k1): k1_estimates.csv (one row per spec and solution, rewritten after each spec),
k1_numbers.json (inputs, md5s, first stages, samples, tests).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "diagnostics"))

import argparse
import json
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from utils import demand_chars
from utils import paths
from utils.cluster_stats import effective_cluster_stats
import diag_k0_netflow_objects as k0
import diag_k2_netflow_simulation as k2
import diag_k2_nls_simulation as kn

ROUT = paths.rout_dir()
OUT = ROUT / "k1"
FINAL_PANEL = "fa38794d"            # I17, 11-05 10:39
OMEGA_FILE = ROUT / "k0_omega.csv"
K0_NUMBERS = ROUT / "k0_numbers.json"
P1_FILE = ROUT / "p1_pix_dose_exposure_mca.csv"
P2_ADV = ROUT / "p2_advertising_conglomerate_quarter.csv"
P2_COV = ROUT / "p2_advertising_coverage_B.csv"
P3_STAKES = ROUT / "p3_stakes_mca_quarter.csv"
RECOVERY = ROUT / "k2_nls_recovery.csv"
COST = list(demand_chars.COST_IV)
POST_FROM = (2020, 4)
PLACEBOS = {"placebo_2018q4": (2018, 4), "placebo_2019q4": (2019, 4)}
SWEEPS = {"sweep_santander": ("C0080185", (2022, 2)), "sweep_bb": ("C0080329", (2022, 1))}
ADV_OBSERVED = ("observed_positive", "observed_zero", "observed_negative", "imputed_from_row_total")
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
DRAWS = 9999
SEED = 20261105
STARTS = (0.0, 0.5, 1.0, 1.5)     # kappa-tilde starts (Pedro, 11-06): every start is reported
MAXIT = 500                       # iteration cap of every fit (Pedro, 11-06)
MAX_INTERIM_MARKETS = 100
DAILY_MIN = 1e-3                  # the daily-liquidity share of the type-4 pools below which Q4d = R4d = 0
PSI_MIN = -30.0                   # --smooth-effects: the floor of the log effects (e^-30, about 1e-13 of the scale)
STEP_UP = 10.0                    # --smooth-effects: the largest rise of a free log effect in one inner step
SMOOTH_INNER_MAX = 500            # --smooth-effects: inner Gauss-Newton iterations at most
SMOOTH_INNER_TOL = 1e-14          # --smooth-effects: relative fall of the sum of squares below which the inner solve stops
SMOOTH_STEP_TOL = 1e-10           # --smooth-effects: ... when the largest log-effect move is also below this
C_MAX = 100.0                     # --smooth-effects: the exit coefficient's cap, data units (the model's value is 1)
stamp = k0.stamp


def qi(y, q):
    return k0.qi(y, q)


def zero_premium(q):
    """2016Q1-2017Q3 and 2021Q4-2024Q4 (the quarters with no old-rule savings premium)."""
    q = np.asarray(q)
    return ((q >= qi(2016, 1)) & (q <= qi(2017, 3))) | ((q >= qi(2021, 4)) & (q <= qi(2024, 4)))


# ------------------------------------------------------------------------------------------ the model
class KModel(kn.Model):
    """The entry index of the K1 equation (module docstring), one kappa-tilde per pair. Parameter
    order: za; kappa (savings) by Zk; kappa_chk by Zk; kappa_td by Zk (td only); a; a_null (outside
    only); xe_names; zq_names; theta_w (exit_const only). theta_w is not in the entry index: its column
    of dh is zero and nls_fit_k1 adds its exit term. td=False drops the time-deposit pair."""

    def __init__(self, cols, g, Za=None, Zk=None, za_names=("alpha",), zk_names=("kappa",), outside=True, td=True,
                 Xe=None, xe_names=(), Zq=None, zq_names=(), exit_const=True, matured_scale=False,
                 savings_only=False):
        super().__init__(cols, g, Za=Za, Zk=Zk, za_names=za_names, zk_names=zk_names, outside=outside, split=True)
        missing = [c for c in ("Q1", "Q4m", "Q4d", "R1", "R2", "R4m", "R4d", "O") if c not in cols]
        if missing:
            raise ValueError(f"KModel needs the pools {missing}")
        n = len(g)
        self.td, self.exit_const, self.matured_scale, self.savings_only = td, exit_const, matured_scale, savings_only
        if savings_only and (zq_names or matured_scale):
            raise ValueError("--savings-only removes the own checking and matured pools: no sweep shifters, no a_m")
        self.Xe = np.zeros((n, 0)) if Xe is None else np.asarray(Xe, float).reshape(n, -1)
        self.Zq = np.zeros((n, 0)) if Zq is None else np.asarray(Zq, float).reshape(n, -1)
        self.xe_names, self.zq_names = list(xe_names), list(zq_names)
        if self.Xe.shape[1] != len(self.xe_names) or self.Zq.shape[1] != len(self.zq_names):
            raise ValueError("shifter columns and names differ in number")
        if savings_only:     # the 11-03 savings pair: no checking scale a, no checking pair
            pairs = [f"{n_}_td" for n_ in self.zk_names] if td else []
            self.base_names = self.za_names + self.zk_names + pairs + (["a_null"] if outside else [])
        else:
            pairs = [f"{n_}_chk" for n_ in self.zk_names] + ([f"{n_}_td" for n_ in self.zk_names] if td else [])
            self.base_names = (self.za_names + self.zk_names + pairs + ["a"] + (["a_null"] if outside else []))
        self.names = (self.base_names + self.xe_names + self.zq_names + (["a_m"] if matured_scale else [])
                      + (["theta_w"] if exit_const else []))
        self.p = len(self.names)
        self.iw = self.names.index("theta_w") if exit_const else None

    def start(self, kappa0=0.0):
        s = np.zeros(self.p)
        nz, nk = len(self.za_names), len(self.zk_names)
        if self.savings_only:
            s[nz] = kappa0
            if self.td:
                s[nz + nk] = kappa0
            return s
        s[nz] = s[nz + nk] = kappa0
        if self.td:
            s[nz + 2 * nk] = kappa0
        return s

    def box(self):
        lo, hi = [], []
        for nm in self.names:
            if nm in self.base_names:      # kn.Model's rule
                b = (-3.0, 3.0) if nm.startswith("alpha") else (-4.0, 10.0) if nm.startswith("kappa") else \
                    (-8.0, 8.0) if nm == "a" else (-12.0, 12.0)
            else:
                b = (-3.0, 3.0) if nm.startswith("cf_") else (-8.0, 8.0)
            lo.append(b[0]), hi.append(b[1])
        return np.array(lo), np.array(hi)

    def h(self, s):
        nz, nk = len(self.za_names), len(self.zk_names)
        psi = s[nz:]
        c = self.c
        if self.savings_only:
            return self._h_savings(s, nz, nk, psi, c)
        ek = np.exp(-(self.Zk @ psi[:nk]))
        e1 = np.exp(-(self.Zk @ psi[nk:2 * nk]))
        ia = (3 if self.td else 2) * nk
        e4 = np.exp(-(self.Zk @ psi[2 * nk:3 * nk])) if self.td else None
        ea = np.exp(psi[ia])
        j = ia + 1 + (1 if self.outside else 0)
        xe = psi[j:j + len(self.xe_names)]
        j += len(self.xe_names)
        zq = psi[j:j + len(self.zq_names)]
        j += len(self.zq_names)
        ea_own = np.exp(psi[ia] + self.Zq @ zq) if len(self.zq_names) else ea
        own_chk = c["Q1"] + c["Q4m"]
        if self.matured_scale:     # own matured money at its own scale e^{a_m}, the sweep shifters kept on it
            eam = np.exp(psi[j] + self.Zq @ zq) if len(self.zq_names) else np.exp(psi[j])
            own_q1, own_m = ea_own * c["Q1"], eam * c["Q4m"]
        riv_chk = c["R1"] + c["R4m"]
        own_term = (own_q1 + own_m) if self.matured_scale else ea_own * own_chk
        B = 1.0 + own_term + c["Q4d"] + ek * c["R2"] + e1 * ea * riv_chk
        d_a = (own_q1 if self.matured_scale else ea_own * own_chk) + e1 * ea * riv_chk
        if self.td:
            B = B + e4 * c["R4d"]
        if self.outside:
            e0 = np.exp(psi[ia + 1])
            B = B + e0 * c["O"]
        parts = [(-ek * c["R2"])[:, None] * self.Zk, (-e1 * ea * riv_chk)[:, None] * self.Zk]
        if self.td:
            parts.append((-e4 * c["R4d"])[:, None] * self.Zk)
        parts.append(d_a[:, None])
        if self.outside:
            parts.append((e0 * c["O"])[:, None])
        if len(self.xe_names):
            parts.append(B[:, None] * self.Xe)
        if len(self.zq_names):
            parts.append(own_term[:, None] * self.Zq)
        if self.matured_scale:
            parts.append(own_m[:, None])
        if self.exit_const:
            parts.append(np.zeros((len(B), 1)))
        expo = -self.g * (self.Za @ s[:nz])
        if len(self.xe_names):
            expo = expo + self.Xe @ xe
        pr = np.exp(np.clip(expo, -700, 700))
        dh = np.column_stack([(-self.g * pr * B)[:, None] * self.Za, pr[:, None] * np.column_stack(parts)])
        return pr * B, dh

    def _h_savings(self, s, nz, nk, psi, c):
        """--savings-only: B = 1 + Q4d + e^{-k} R2 [+ e^{-k_td} R4d] [+ e^{a_null} O]; the own checking and
        matured pools (Q1, Q4m) and the rivals' (R1, R4m) are out, so there is no a and no checking pair."""
        ek = np.exp(-(self.Zk @ psi[:nk]))
        j = nk
        B = 1.0 + c["Q4d"] + ek * c["R2"]
        parts = [(-ek * c["R2"])[:, None] * self.Zk]
        if self.td:
            e4 = np.exp(-(self.Zk @ psi[nk:2 * nk]))
            B = B + e4 * c["R4d"]
            parts.append((-e4 * c["R4d"])[:, None] * self.Zk)
            j = 2 * nk
        if self.outside:
            e0 = np.exp(psi[j])
            B = B + e0 * c["O"]
            parts.append((e0 * c["O"])[:, None])
            j += 1
        xe = psi[j:j + len(self.xe_names)]
        if len(self.xe_names):
            parts.append(B[:, None] * self.Xe)
        if self.exit_const:
            parts.append(np.zeros((len(B), 1)))
        expo = -self.g * (self.Za @ s[:nz])
        if len(self.xe_names):
            expo = expo + self.Xe @ xe
        pr = np.exp(np.clip(expo, -700, 700))
        dh = np.column_stack([(-self.g * pr * B)[:, None] * self.Za, pr[:, None] * np.column_stack(parts)])
        return pr * B, dh


# ---------------------------------------------------------------------------------------- the fit
def nls_fit_k1(y, E, a, b, market, cl, model, s0, rowscale=None, box=None, maxit=200, Em=None, tol_cos=1e-9,
               tol_step=1e-9, inner_max=200, inner_tol=1e-13, smooth=False):
    """kn.nls_fit with the constant matured exit term -e^{theta_w} Em (module docstring), when the model
    has theta_w (model.iw); otherwise kn.nls_fit step for step. E is the effects' exit column (the
    daily-liquidity part, or the whole movable stock with --exit-by-cell). Returns the estimates, the
    cluster sandwich on the effect-partialled Jacobian, the diagnostics, and that Jacobian and the
    residual (in the fit's scaling) for the score bootstrap. smooth: nls_fit_smooth (--smooth-effects)."""
    if smooth:
        return nls_fit_smooth(y, E, a, b, market, cl, model, s0, rowscale=rowscale, box=box, maxit=maxit, Em=Em,
                              tol_cos=tol_cos, tol_step=tol_step)
    iw = getattr(model, "iw", None)
    fe = kn.Effects(a, b, market)
    mk, nm = fe.mk, fe.nm
    n = len(y)
    d = np.ones(n) if rowscale is None else np.asarray(rowscale, float)
    y = np.asarray(y, float) * d
    ysc = np.mean(np.abs(y))
    y = y / ysc
    E = np.asarray(E, float) * d
    Ew = None if iw is None else np.asarray(Em, float) * d / ysc
    yy = y @ y
    steps = 0.5 ** np.arange(12)

    def hfun(s_):
        h_, dh_ = model.h(s_)
        return h_ * d, dh_ * d[:, None]

    def ytil(s_):
        return y if iw is None else y + np.exp(s_[iw]) * Ew

    def jac(w_, sb_, dh_, s_):
        Js_ = (w_[fe.a] * sb_[fe.b])[:, None] * dh_
        if iw is not None:
            Js_[:, iw] = -np.exp(s_[iw]) * Ew
        return Js_

    def inner(h_, w_, sb_, cb_, yt, sweeps=10):
        if sweeps:
            w_, sb_, cb_, _ = kn.als_effects(yt, h_, E, fe, w_.copy(), sweeps=sweeps)
        r_ = yt - (w_[fe.a] * sb_[fe.b] * h_ - cb_[fe.b] * E)
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
                rt = yt - (np.maximum(w_ + t * dw, 0)[fe.a] * np.maximum(sb_ + t * dsb, 0)[fe.b] * h_
                           - np.maximum(cb_ + t * dcb, 0)[fe.b] * E)
                sm_ = np.bincount(mk, rt * rt, nm)
                better = np.isfinite(sm_) & (sm_ < best)
                best = np.where(better, sm_, best)
                tb = np.where(better, t, tb)
            w_ = np.maximum(w_ + tb[fe.mk_a] * dw, 0.0)
            sb_ = np.maximum(sb_ + tb[fe.mk_b] * dsb, 0.0)
            cb_ = np.maximum(cb_ + tb[fe.mk_b] * dcb, 0.0)
            r_ = yt - (w_[fe.a] * sb_[fe.b] * h_ - cb_[fe.b] * E)
            new = np.bincount(mk, r_ * r_, nm)
            dec = (ssr_m.sum() - new.sum()) / max(ssr_m.sum(), 1e-300)
            ssr_m = new
            if dec < inner_tol or ssr_m.sum() <= 1e-28 * yy:
                break
        return w_, sb_, cb_, r_, float(ssr_m.sum()), k + 1

    s = np.asarray(s0, float).copy()
    h, dh = hfun(s)
    yt = ytil(s)
    w, sb, cb, n_als = kn.als_effects(yt, h, E, fe, np.ones(fe.n1))
    w, sb, cb, r, ssr, inner_total = inner(h, w, sb, cb, yt, sweeps=0)
    lam = 1e-3
    conv, reason, last_step, cos, it, restarts = False, "maxit", np.inf, np.nan, -1, 0
    hist = []
    for it in range(maxit):
        Js = jac(w, sb, dh, s)
        fe.prepare(sb[fe.b] * h, w[fe.a] * h, E)
        fit, *_ = fe.project(Js)
        Jt = Js - fit
        gvec = Jt.T @ r
        cos = float(np.max(np.abs(gvec) / np.maximum(np.sqrt((Jt * Jt).sum(axis=0)) * np.sqrt(ssr), 1e-300)))
        hist.append((ssr, float(np.linalg.norm(gvec))))
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
            yt_n = ytil(s + ds)
            out = inner(h_n, w, sb, cb, yt_n)
            inner_total += out[-1]
            if np.isfinite(out[4]) and out[4] < ssr:
                accepted = True
                break
            lam *= 10
        if not accepted:
            wf, sbf, cbf, _ = kn.als_effects(yt, h, E, fe, np.ones(fe.n1))
            fresh = inner(h, wf, sbf, cbf, yt, sweeps=0)
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
        s, h, dh, yt = s + ds, h_n, dh_n, yt_n
        w, sb, cb, r, ssr, _ = out
        lam = max(lam / 10, 1e-10)
        if rel < 1e-15 and last_step < tol_step:
            conv, reason = True, "no further decrease"
            break
    Js = jac(w, sb, dh, s)
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
    back = hist[-100] if len(hist) >= 100 else hist[0] if hist else (ssr, np.nan)
    diag = {"n": int(n), "G": int(G), "markets": int(fe.nm), "cells": int(fe.n2), "bank_markets": int(fe.n1),
            "iterations": it + 1, "inner_iterations": int(inner_total), "converged": bool(conv), "reason": reason,
            "ssr_over_yy": float(ssr / yy), "gradient_cosine": float(cos),
            "grad_norm_start": float(hist[0][1]) if hist else np.nan, "grad_norm_end": float(np.linalg.norm(Jt.T @ r)),
            "ssr_rel_change_last100": float((back[0] - ssr) / max(back[0], 1e-300)),
            "ssr_change_window": int(min(len(hist), 100)),
            "hessian_condition": float(ev.max() / max(ev.min(), 1e-300)), "at_box": at_box, "restarts": int(restarts),
            "zero_exit_cells": int((cb <= 0).sum()), "zero_entry_cells": int((sb <= 0).sum()), "zero_banks": int((w <= 0).sum()),
            "exit_coef": cb * ysc}
    return dict(zip(model.names, s)), pd.DataFrame(V, index=model.names, columns=model.names), diag, Jt, r


class SmoothProfile:
    """--smooth-effects: the profiled objective with the effects as exp(psi), w_jm = e^{psi_jm},
    s_mt = e^{psi_mt}, c_mt = e^{psi^c_mt}, so positivity is built in and the inner solve is
    unconstrained. For given parameters, Gauss-Newton in psi: kn.Effects on the log-tangent columns
    f 1{a}, f 1{b}, -c E 1{b} (f = w s h), a line search by market (each market's direction scaled so that
    no free log effect rises by more than STEP_UP), from the previous solution, until the sum of squares
    stops falling. The data are scaled as in nls_fit_k1 (rowscale, then y / mean |y|).
    Floor: an effect whose log falls below PSI_MIN is set to zero (psi = -inf, `floored`). A floored effect
    is at its bound: its column leaves the Gauss-Newton step and the parameters' projection, and it is
    raised to its one-variable least-squares level when raising it lowers the sum of squares.
    Exit cap (C_MAX): the exit coefficient c_mt is at most C_MAX in data units (the model's value is 1).
    In a cell whose entry and exit columns are (near) collinear the least-squares entry and exit effects
    grow without bound and cancel; the cap stops the exit effect there, and the entry effect follows. A
    capped exit effect is at its bound in the same way as a floored one (it leaves the step and the
    projection while lowering it would raise the sum of squares); `capped_exit_cells` counts them.
    Normalization: w s is unchanged when a constant is added to every psi_jm and subtracted from every
    psi_mt of a connected component of the bank x market / market-quarter graph, one flat direction per
    component (c_mt multiplies data, not another effect). kn.Effects' pseudo-inverse gives the step no
    component along it, and after every step psi_jm is re-centred to mean zero (over the effects above
    the floor) within each component: every psi_jm and psi_mt of the component shifts, floored ones
    included (-inf stays -inf), with no clamp, so the fit is unchanged.
    The parameters' effect-partialled Jacobian projects on the level columns of the effects off their
    bounds, which span the same space as their log-tangent columns, so -2 Jt' r is the gradient of the
    profiled sum of squares (envelope theorem)."""

    def __init__(self, y, E, a, b, market, model, rowscale=None, Em=None):
        self.model, self.iw = model, getattr(model, "iw", None)
        self.fe = kn.Effects(a, b, market)
        fe = self.fe
        graph = csr_matrix((np.ones(fe.n), (fe.a, fe.n1 + fe.b)), shape=(fe.n1 + fe.n2, fe.n1 + fe.n2))
        self.ncomp, lab = connected_components(graph, directed=False)
        self.comp_a, self.comp_b = lab[:fe.n1], lab[fe.n1:]
        n = len(y)
        self.d = np.ones(n) if rowscale is None else np.asarray(rowscale, float)
        y = np.asarray(y, float) * self.d
        self.ysc = np.mean(np.abs(y))
        self.y = y / self.ysc
        self.E = np.asarray(E, float) * self.d
        self.Ew = None if self.iw is None else np.asarray(Em, float) * self.d / self.ysc
        self.yy = self.y @ self.y
        self.c_informative = np.bincount(self.fe.b, self.E * self.E, self.fe.n2) > 0
        self.lc_max = float(np.log(C_MAX) - np.log(self.ysc))

    def hfun(self, s):
        h, dh = self.model.h(s)
        return h * self.d, dh * self.d[:, None]

    def ytil(self, s):
        return self.y if self.iw is None else self.y + np.exp(s[self.iw]) * self.Ew

    @staticmethod
    def floored(psi):
        return tuple(np.isneginf(v) for v in psi)

    def capped(self, lc):
        return lc >= self.lc_max - 1e-12

    def clean(self, lw, ls, lc):
        """Logs below PSI_MIN to -inf, the exit cap; copies."""
        out = [np.where(np.asarray(v, float) < PSI_MIN, -np.inf, np.asarray(v, float)) for v in (lw, ls, lc)]
        out[2] = np.minimum(out[2], self.lc_max)
        return out

    def normalize(self, lw, ls):
        """Mean-zero bank x market log effects (over those above the floor) within each component; every
        log effect of the component shifts, floored ones included, with no clamp."""
        live = np.isfinite(lw)
        cnt = np.bincount(self.comp_a, live.astype(float), self.ncomp)
        mu = np.bincount(self.comp_a, np.where(live, lw, 0.0), self.ncomp) / np.maximum(cnt, 1.0)
        return lw - mu[self.comp_a], ls + mu[self.comp_b]

    def start(self, s):
        """The non-negative ALS solution in levels at s (kn.als_effects from ones), in logs, floored,
        capped and normalized."""
        h, _ = self.hfun(s)
        w, sb, cb, _ = kn.als_effects(self.ytil(s), h, self.E, self.fe, np.ones(self.fe.n1))
        with np.errstate(divide="ignore"):
            lw, ls, lc = (np.log(np.maximum(np.nan_to_num(v, nan=0.0, posinf=0.0), 0.0)) for v in (w, sb, cb))
        lw, ls, lc = self.clean(lw, ls, lc)
        return (*self.normalize(lw, ls), lc)

    def resid(self, h, yt, lw, ls, lc):
        fe = self.fe
        with np.errstate(over="ignore", invalid="ignore"):
            return yt - (np.exp(lw)[fe.a] * np.exp(ls)[fe.b] * h - np.exp(lc)[fe.b] * self.E)

    @staticmethod
    def _moved(new, old):
        """The largest move of a log effect between finite values; inf when an effect enters or leaves
        the floor."""
        m = 0.0
        for v1, v0 in zip(new, old):
            f1, f0 = np.isfinite(v1), np.isfinite(v0)
            if (f1 != f0).any():
                return np.inf
            if f1.any():
                m = max(m, float(np.max(np.abs(v1[f1] - v0[f1]))))
        return m

    def solve(self, s, psi, inner_max=SMOOTH_INNER_MAX, tol=SMOOTH_INNER_TOL, step_tol=SMOOTH_STEP_TOL):
        """The effects at s by Gauss-Newton in psi, from psi: (psi, r, ssr, iterations). It stops when
        the relative fall of the sum of squares is below tol and the largest move below step_tol, or when
        no market's sum of squares falls; psi is normalized at entry and after every step."""
        fe, mk, nm = self.fe, self.fe.mk, self.fe.nm
        h, _ = self.hfun(s)
        yt = self.ytil(s)
        lw, ls, lc = self.clean(*psi)
        lw, ls = self.normalize(lw, ls)
        r = self.resid(h, yt, lw, ls, lc)
        ssr_m = np.bincount(mk, r * r, nm)
        if not np.isfinite(ssr_m.sum()):
            return (lw, ls, lc), r, np.inf, 0
        steps = 0.5 ** np.arange(12)
        k = -1
        for k in range(inner_max):
            w, sb, cb = np.exp(lw), np.exp(ls), np.exp(lc)
            fw, fs, fc = self.floored((lw, ls, lc))
            xw, xs = sb[fe.b] * h, w[fe.a] * h
            gw, qw = fe.Dat @ (xw * r), fe.Dat @ (xw * xw)
            gs, qs = np.bincount(fe.b, xs * r, fe.n2), np.bincount(fe.b, xs * xs, fe.n2)
            gc, qc = np.bincount(fe.b, -self.E * r, fe.n2), np.bincount(fe.b, self.E * self.E, fe.n2)
            cap = self.capped(lc) & (gc >= 0)
            fix_c = fc | cap
            f = w[fe.a] * sb[fe.b] * h
            fe.prepare(f * ~fw[fe.a], f * ~fs[fe.b], cb[fe.b] * self.E * ~fix_c[fe.b])
            _, xa, x1, x2 = fe.project(r)
            dw, dsb, dcb = (np.nan_to_num(v[:, 0], nan=0.0, posinf=0.0, neginf=0.0) for v in (xa, x1, x2))
            dw[fw], dsb[fs], dcb[fix_c] = 0.0, 0.0, 0.0
            up = np.zeros(nm)
            np.maximum.at(up, fe.mk_a, dw)
            np.maximum.at(up, fe.mk_b, dsb)
            np.maximum.at(up, fe.mk_b, dcb)
            sc = np.minimum(1.0, STEP_UP / np.maximum(up, 1e-300))
            dw, dsb, dcb = dw * sc[fe.mk_a], dsb * sc[fe.mk_b], dcb * sc[fe.mk_b]
            rel = [(fl & (g > 0) & (q > 0), np.where(q > 0, g / np.where(q > 0, q, 1.0), 0.0))
                   for fl, g, q in ((fw, gw, qw), (fs, gs, qs), (fc, gc, qc))]

            def trial(t_w, t_b):
                out = []
                for lv, dv, (rm, lev), tt in ((lw, dw, rel[0], t_w), (ls, dsb, rel[1], t_b), (lc, dcb, rel[2], t_b)):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        v = np.where(rm, np.log(tt * lev), lv + tt * dv)
                    out.append(v)
                return self.clean(*out)

            best = ssr_m.copy()
            tb = np.zeros(nm)
            for t in steps:
                rt = self.resid(h, yt, *trial(t, t))
                sm_ = np.bincount(mk, rt * rt, nm)
                better = np.isfinite(sm_) & (sm_ < best)
                best = np.where(better, sm_, best)
                tb = np.where(better, t, tb)
            if not tb.any():
                break
            new = trial(tb[fe.mk_a], tb[fe.mk_b])
            moved = self._moved(new, (lw, ls, lc))
            lw, ls = self.normalize(new[0], new[1])
            lc = new[2]
            r = self.resid(h, yt, lw, ls, lc)
            ssr_new = np.bincount(mk, r * r, nm)
            dec = (ssr_m.sum() - ssr_new.sum()) / max(ssr_m.sum(), 1e-300)
            ssr_m = ssr_new
            if (dec < tol and moved < step_tol) or ssr_m.sum() <= 1e-28 * self.yy:
                break
        return (lw, ls, lc), r, float(ssr_m.sum()), k + 1

    def tangent(self, s, psi, h, dh):
        """The parameters' Jacobian Js at (s, psi) and Jt, Js less its projection on the level columns
        of the effects off their bounds (floored, or the exit cap)."""
        fe, iw = self.fe, self.iw
        w, sb, cb = (np.exp(v) for v in psi)
        fw, fs, fc = self.floored(psi)
        fix_c = fc | self.capped(psi[2])
        Js = (w[fe.a] * sb[fe.b])[:, None] * dh
        if iw is not None:
            Js[:, iw] = -np.exp(s[iw]) * self.Ew
        fe.prepare(sb[fe.b] * h * ~fw[fe.a], w[fe.a] * h * ~fs[fe.b], self.E * ~fix_c[fe.b])
        fit, *_ = fe.project(Js)
        return Js, Js - fit

    def collinear_cells(self, s, psi, tol=1e-6):
        """Cells (with both effects off their bounds) whose entry and exit columns have |cosine| above 1 - tol."""
        fe = self.fe
        h, _ = self.hfun(s)
        fw, fs, fc = self.floored(psi)
        x = np.exp(psi[0])[fe.a] * h
        num = np.bincount(fe.b, x * self.E, fe.n2)
        den = np.sqrt(np.bincount(fe.b, x * x, fe.n2) * np.bincount(fe.b, self.E * self.E, fe.n2))
        cos = np.abs(num) / np.maximum(den, 1e-300)
        return int(((cos > 1 - tol) & ~fs & ~fc & self.c_informative).sum())


def nls_fit_smooth(y, E, a, b, market, cl, model, s0, rowscale=None, box=None, maxit=200, Em=None, tol_cos=1e-9,
                   tol_step=1e-9):
    """nls_fit_k1's Levenberg-Marquardt loop on the smooth profile (SmoothProfile): the effects start
    from the levels' ALS solution at s0, and every trial step re-solves them from the current ones. When
    no damping (14 increases) gives a lower sum of squares, the fit ends ("no descent step"), with no
    fresh ALS restart. Returns what nls_fit_k1 returns; the diagnostics add the floored effects (count
    and split by kind; the exit coefficients of cells with no exit column are not counted) and psi."""
    P = SmoothProfile(y, E, a, b, market, model, rowscale=rowscale, Em=Em)
    fe, yy = P.fe, P.yy
    n = len(P.y)
    s = np.asarray(s0, float).copy()
    psi, r, ssr, inner_total = P.solve(s, P.start(s))
    inner = [inner_total]
    h, dh = P.hfun(s)
    lam = 1e-3
    conv, reason, last_step, cos, it = False, "maxit", np.inf, np.nan, -1
    hist = []
    for it in range(maxit):
        _, Jt = P.tangent(s, psi, h, dh)
        gvec = Jt.T @ r
        cos = float(np.max(np.abs(gvec) / np.maximum(np.sqrt((Jt * Jt).sum(axis=0)) * np.sqrt(ssr), 1e-300)))
        hist.append((ssr, float(np.linalg.norm(gvec))))
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
            out = P.solve(s + ds, psi)
            inner_total += out[3]
            inner.append(out[3])
            if np.isfinite(out[2]) and out[2] < ssr:
                accepted = True
                break
            lam *= 10
        if not accepted:
            reason = "no descent step"
            conv = cos < 1e-5
            break
        last_step = float(np.max(np.abs(ds) / np.maximum(np.abs(s), 1e-1)))
        rel = (ssr - out[2]) / max(ssr, 1e-300)
        s = s + ds
        psi, r, ssr = out[0], out[1], out[2]
        h, dh = P.hfun(s)
        lam = max(lam / 10, 1e-10)
        if rel < 1e-15 and last_step < tol_step:
            conv, reason = True, "no further decrease"
            break
    _, Jt = P.tangent(s, psi, h, dh)
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
    back = hist[-100] if len(hist) >= 100 else hist[0] if hist else (ssr, np.nan)
    fw, fs, fc = P.floored(psi)
    fci = fc & P.c_informative
    diag = {"n": int(n), "G": int(G), "markets": int(fe.nm), "cells": int(fe.n2), "bank_markets": int(fe.n1),
            "iterations": it + 1, "inner_iterations": int(inner_total), "converged": bool(conv), "reason": reason,
            "ssr_over_yy": float(ssr / yy), "gradient_cosine": float(cos),
            "grad_norm_start": float(hist[0][1]) if hist else np.nan, "grad_norm_end": float(np.linalg.norm(Jt.T @ r)),
            "ssr_rel_change_last100": float((back[0] - ssr) / max(back[0], 1e-300)),
            "ssr_change_window": int(min(len(hist), 100)),
            "hessian_condition": float(ev.max() / max(ev.min(), 1e-300)), "at_box": at_box, "restarts": 0,
            "zero_exit_cells": int(fc.sum()), "zero_entry_cells": int(fs.sum()), "zero_banks": int(fw.sum()),
            "exit_coef": np.where(fc, 0.0, np.exp(psi[2])) * P.ysc, "smooth_effects": True,
            "floored_effects": int(fw.sum() + fs.sum() + fci.sum()),
            "floored_split": f"w {int(fw.sum())} of {fe.n1}, s {int(fs.sum())} of {fe.n2}, "
                             f"c {int(fci.sum())} of {int(P.c_informative.sum())}",
            "capped_exit_cells": int(P.capped(psi[2]).sum()), "collinear_cells": P.collinear_cells(s, psi),
            "components": int(P.ncomp), "inner_solves": len(inner), "inner_iterations_first": int(inner[0]),
            "inner_iterations_median": float(np.median(inner)), "inner_iterations_max": int(max(inner)),
            "inner_cap_hits": int(sum(v >= SMOOTH_INNER_MAX for v in inner)), "psi": psi}
    return dict(zip(model.names, s)), pd.DataFrame(V, index=model.names, columns=model.names), diag, Jt, r


def multistart_k1(y, E, a, b, market, cl, model, starts=STARTS, rowscale=None, Em=None, maxit=200, smooth=False):
    """kn.nls_multistart's selection (the smallest sum of squares), with nls_fit_k1 from each start;
    every start's end point is kept for the report, and every start's own fit (estimates, V,
    diagnostics, Jacobian, residual) is returned in d["fits"]."""
    box = model.box()
    fits, secs = [], []
    for k0 in starts:
        t0 = time.time()
        fits.append(nls_fit_k1(y, E, a, b, market, cl, model, model.start(k0), rowscale=rowscale, box=box, Em=Em,
                               maxit=maxit, smooth=smooth))
        secs.append(round(time.time() - t0, 1))
    ssr = [f[2]["ssr_over_yy"] for f in fits]
    i = int(np.argmin(ssr))
    est, V, d, Jt, r = fits[i]
    kn_ = model.names[len(model.za_names)]
    ks = [float(f[0][kn_]) for f in fits]
    per = {"start_kappas": ks, "start_ssr": ssr, "start_converged": [f[2]["converged"] for f in fits],
           "start_iterations": [f[2]["iterations"] for f in fits], "start_reasons": [f[2]["reason"] for f in fits],
           "start_at_box": ["|".join(f[2]["at_box"]) for f in fits], "start_seconds": secs}
    if "dkappa" in model.names:
        per["start_dkappas"] = [float(f[0]["dkappa"]) for f in fits]
    if smooth:
        per["start_floored"] = [f[2]["floored_effects"] for f in fits]
    d = {**d, **per, "starts": list(starts), "starts_spread": float(max(ks) - min(ks)), "best_start": starts[i],
         "fits": [dict(zip(("est", "V", "d", "Jt", "r"), f), start=k0) for f, k0 in zip(fits, starts)]}
    return est, V, d, Jt, r


# --------------------------------------------------------------------------------------- inference
def cluster_scores(Jt, r, cl):
    cg = pd.factorize(cl)[0]
    S = np.zeros((cg.max() + 1, Jt.shape[1]))
    np.add.at(S, cg, Jt * r[:, None])
    return S


def score_wcb(Jt, r, cl, draws=DRAWS, seed=SEED, contrasts=None):
    """Score wild cluster bootstrap: the one-step draws H^-1 sum_g v_g S_g with Webb weights v_g,
    each studentized by its own cluster sandwich; symmetric percentile-t half-widths at 95 percent,
    for each parameter and for each contrast c'theta in `contrasts` (name -> vector)."""
    S = cluster_scores(Jt, r, cl)
    G = S.shape[0]
    Hi = np.linalg.pinv(Jt.T @ Jt)
    c = G / (G - 1)
    A = S @ Hi
    names = list(contrasts) if contrasts else []
    if names:
        A = np.column_stack([A] + [A @ np.asarray(contrasts[k], float) for k in names])
    se = np.sqrt(np.maximum(c * (A * A).sum(axis=0), 0))
    v = np.random.default_rng(seed).choice(WEBB, size=(draws, G))
    D = v @ A
    se_b = np.sqrt(np.maximum(c * (v * v) @ (A * A), 1e-300))
    q = np.quantile(np.abs(D / se_b), 0.95, axis=0)
    p = Jt.shape[1]
    out = {"se_cr1": se[:p], "se_boot": D.std(axis=0)[:p], "half_width": (q * se)[:p], "q95_abs_t": q[:p], "G": G}
    out["contrasts"] = {k: {"se_cr1": float(se[p + i]), "se_boot": float(D[:, p + i].std()),
                            "half_width": float(q[p + i] * se[p + i])} for i, k in enumerate(names)}
    return out


def score_test(Jt0, r0, cl, k, draws=DRAWS, seed=SEED):
    """Score test of theta_k = its restricted value, the null imposed: the efficient cluster scores
    e_g = S_gk - H_k,-k H_-k,-k^-1 S_g,-k at the restricted fit; T = sum e_g / sqrt(sum e_g^2) against
    its Webb-weighted draws."""
    S = cluster_scores(Jt0, r0, cl)
    H = Jt0.T @ Jt0
    o = [i for i in range(H.shape[0]) if i != k]
    e = S[:, k] - S[:, o] @ (np.linalg.pinv(H[np.ix_(o, o)]) @ H[o, k])
    T = e.sum() / np.sqrt(max((e * e).sum(), 1e-300))
    v = np.random.default_rng(seed).choice(WEBB, size=(draws, len(e)))
    Tb = (v @ e) / np.sqrt(np.maximum((v * v) @ (e * e), 1e-300))
    return {"T": float(T), "p": float(np.mean(np.abs(Tb) >= abs(T)))}


# ------------------------------------------------------------------------------------------- inputs
def clock_table(lambda_one=False):
    """omega and lambda by conglomerate-year from K0 (e), with the rules of the module docstring."""
    W = pd.read_csv(OMEGA_FILE, dtype={"conglomerate": str})
    own = W[(W["imputed"] == 0) & (W["is_B"] == 1)]
    med_l = own.groupby("year")["share_no_maturity"].median()
    med_o = W[W["is_B"] == 1].groupby("year")["omega"].median()

    def nearest(s, y):
        return float(s.loc[min(s.index, key=lambda z: (abs(z - y), z))])

    lam = np.where(W["is_B"] == 1, W["share_no_maturity"], 1.0)
    miss = np.isnan(lam)
    lam[miss] = [nearest(med_l, y) for y in W.loc[miss, "year"]]
    if lambda_one:
        lam = np.ones(len(W))
    W = W.assign(lam=lam)
    tab = {(j, int(y)): (float(o), float(l)) for j, y, o, l in zip(W["conglomerate"], W["year"], W["omega"], W["lam"])}
    fill = {int(y): (nearest(med_o, y), 1.0 if lambda_one else nearest(med_l, y)) for y in range(2016, 2025)}
    return tab, fill, {"rows": int(len(W)), "B_with_statement_lambda": int(own["share_no_maturity"].notna().sum()),
                       "lambda_filled_with_B_median": int(miss.sum()), "lambda_one": bool(lambda_one)}


def omega_lambda(j, qidx, tab, fill):
    y = np.clip(np.asarray(qidx) // 4 - 1, 2016, 2024)
    out = np.array([tab.get((jj, int(yy)), fill[int(yy)]) for jj, yy in zip(j, y)], float).reshape(-1, 2)
    return out[:, 0], out[:, 1]


def control_function(px):
    """First stage of the conglomerate-quarter type-4 spread on the cost shifters (module docstring)."""
    agg = dict(spread=("spread_ann_a4_national", "first"), dep4=("dep_a4", "sum"), carried=("rate_a4_carried", "max"),
               **{c: (c, "first") for c in COST})
    cq = px[px["is_B"] == 1].groupby(["j", "qi"]).agg(**agg).reset_index()
    cq["spread"] = 100.0 * cq["spread"]
    ok = cq["spread"].notna() & cq[COST].notna().all(axis=1)
    win = (cq["qi"] >= qi(*k0.T0)) & (cq["qi"] <= qi(*k0.T1))
    f = cq[ok & win & (cq["dep4"] > 0) & (cq["carried"].fillna(0) == 0)]
    X = pd.concat([f[COST], pd.get_dummies(f["qi"], prefix="q", dtype=float)], axis=1)
    r = sm.WLS(f["spread"], X, weights=f["dep4"]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(f["j"])[0]})
    R = np.zeros((len(COST), len(r.params)))
    for i, c in enumerate(COST):
        R[i, list(r.params.index).index(c)] = 1.0
    qfe = r.params[[p for p in r.params.index if p.startswith("q_")]]
    qfe.index = [int(p[2:]) for p in qfe.index]
    cq["v"] = np.where(ok, cq["spread"] - cq[COST].to_numpy() @ r.params[COST].to_numpy() - cq["qi"].map(qfe).to_numpy(), np.nan)
    info = {"n": int(r.nobs), "conglomerates": int(f["j"].nunique()), "joint_F": float(np.squeeze(r.f_test(R).fvalue)),
            "coef": {c: {"b": float(r.params[c]), "se": float(r.bse[c])} for c in COST}}
    return cq[["j", "qi", "v"]], info


def use_dose():
    """U-hat: P1's first stage, z_U on z_mobile and z_g45 jointly with 2019 CadUnico and log 2019
    population as controls, one row per MCA, 2019 population weights (diag_p_shifters `cs`). The
    controls are returned standardized: they enter the second stage beside U-hat (Post x control in
    kappa-tilde and alpha), so that only the two exposures are excluded."""
    p = pd.read_csv(P1_FILE, dtype={"mca_code": str}).dropna(subset=["z_U", "z_mobile", "z_g45", "w_pop2019", "cad2019"])
    p["log_pop2019"] = np.log(p["w_pop2019"])
    X = sm.add_constant(p[["z_mobile", "z_g45", "cad2019", "log_pop2019"]])
    r = sm.WLS(p["z_U"], X, weights=p["w_pop2019"]).fit(cov_type="HC1")
    rs = sm.WLS(p["z_U"], X, weights=p["w_pop2019"]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p["state"])[0]})
    F = float(np.squeeze(r.f_test("z_mobile = 0, z_g45 = 0").fvalue))
    Fs = float(np.squeeze(rs.f_test("z_mobile = 0, z_g45 = 0").fvalue))
    out = pd.DataFrame({"U_hat": r.fittedvalues.to_numpy(),
                        "cad_z": ((p["cad2019"] - p["cad2019"].mean()) / p["cad2019"].std()).to_numpy(),
                        "logpop_z": ((p["log_pop2019"] - p["log_pop2019"].mean()) / p["log_pop2019"].std()).to_numpy()},
                       index=p["mca_code"].to_numpy())
    return out, {"n": int(r.nobs), "joint_F_mca": F, "joint_F_state": Fs,
                 "slopes": {k: float(r.params[k]) for k in ("z_mobile", "z_g45")}, "r2": float(r.rsquared)}


def advertising():
    a = pd.read_csv(P2_ADV, dtype={"CodConglomeradoPrudencial": str}).rename(columns={"CodConglomeradoPrudencial": "j"})
    raw = a["adv_brl_real_2024q4"].where(a["adv_status"].isin(ADV_OBSERVED))
    a["adv_negative"] = (raw < 0).astype(int)
    a["ladv"] = np.log1p(raw.clip(lower=0.0))
    a["qi"] = qi(a["year"], a["quarter"])
    cov = pd.read_csv(P2_COV, dtype={"CodConglomeradoPrudencial": str})
    big5 = cov.sort_values("rank_2019")["CodConglomeradoPrudencial"].head(5).tolist()
    return a[["j", "qi", "ladv", "adv_negative"]], big5


def panel_extras(panel, md5_expect):
    got = k0.md5(panel)
    if not got.startswith(md5_expect):
        raise SystemExit(f"{panel}: md5 {got[:8]}, the context was built on {md5_expect}")
    cols = ["CodConglomeradoPrudencial", "mca_code", "is_B", "year", "quarter", "dep_a4", "spread_ann_a4_national",
            "rate_a4_fallback", "rate_a4_carried"] + COST
    px = pq.read_table(panel, columns=cols).to_pandas().rename(columns={"CodConglomeradoPrudencial": "j", "mca_code": "m"})
    px["j"], px["m"] = px["j"].astype(str), px["m"].astype(str)
    px["qi"] = qi(px["year"], px["quarter"])
    return px, got


def assemble(ctx, panel, allow_interim, lambda_one=False):
    """The K1 rows with every column a spec reads (module docstring)."""
    K, MT, U = ctx["K"].copy(), ctx["MT"], ctx["U"]
    N0 = ctx["N0"]
    md5 = str(N0.get("panel_md5_read") or N0["panel_md5"])
    px, md5_full = panel_extras(panel, md5[:8])
    info = {"panel_md5": md5_full, "frame_panel_md5": N0.get("frame_panel_md5"), "K1_rows_context": int(len(K))}
    # rate flags at t and t-1
    fb = px[["j", "m", "qi", "rate_a4_fallback"]].copy()
    fb["fb"] = (fb["rate_a4_fallback"].fillna(0) != 0).astype(int)
    lag = fb[["j", "m", "qi", "fb"]].assign(qi=fb["qi"] + 1).rename(columns={"fb": "fb_lag"})
    K = K.merge(fb[["j", "m", "qi", "fb"]], on=["j", "m", "qi"], how="left").merge(lag, on=["j", "m", "qi"], how="left")
    K["carried"] = ((K["fb"].fillna(0) + K["fb_lag"].fillna(0)) > 0).astype(int)
    # omega, lambda: own and rivals' matured and daily-liquidity pools
    tab, fill, info["omega_lambda"] = clock_table(lambda_one)
    om, la = omega_lambda(K["j"].to_numpy(), K["qi"].to_numpy(), tab, fill)
    uo, ul = omega_lambda(U["j"].to_numpy(), U["qi"].to_numpy(), tab, fill)
    mt, nmt = U["mt"].to_numpy(), len(MT)
    D4u = U["D4"].to_numpy()
    S4 = np.bincount(mt, D4u, nmt)
    Sm = np.bincount(mt, D4u * uo, nmt)
    Sd = np.bincount(mt, D4u * (1 - uo) * ul, nmt)
    km, D4, own2 = K["mt"].to_numpy(), K["D4"].to_numpy(), K["own2"].to_numpy()
    r4 = (S4[km] - D4) / own2
    gap = float(np.max(np.abs(r4 - K["R4"].to_numpy()) / np.maximum(np.abs(K["R4"].to_numpy()), 1.0)))
    if gap > 1e-8:
        raise SystemExit(f"the rivals' type-4 pool from U differs from K0's R4 by {gap:g}")
    K["omega"], K["lam"] = om, la
    K["Q4m"], K["Q4d"] = om * K["Q4"], (1 - om) * la * K["Q4"]
    K["R4m"], K["R4d"] = (Sm[km] - om * D4) / own2, (Sd[km] - (1 - om) * la * D4) / own2
    accr = (K["carry4"] / K["own2"]).to_numpy()
    K["Em"], K["Ed"] = accr * om, accr * (1 - om) * la
    # control function
    v, info["control_function"] = control_function(px)
    K = K.merge(v, on=["j", "qi"], how="left")
    # Pix
    K["post"] = (K["qi"] >= qi(*POST_FROM)).astype(float)
    uh, info["use_dose"] = use_dose()
    for c in ("U_hat", "cad_z", "logpop_z"):
        K[c] = K["m"].map(uh[c])
    K["post_U"] = K["post"] * K["U_hat"]
    K["post_cad"], K["post_pop"] = K["post"] * K["cad_z"], K["post"] * K["logpop_z"]
    K["trend_pre"] = (K["qi"] - qi(2020, 3)) / 4.0
    for nm_, yq in PLACEBOS.items():
        K[nm_] = (K["qi"] >= qi(*yq)).astype(float)
    # stakes
    st = pd.read_csv(P3_STAKES, dtype={"mca_code": str}, usecols=["mca_code", "year", "quarter", "S_tilde", "panel_md5"])
    info["stakes_panel_md5"] = sorted(st["panel_md5"].dropna().astype(str).unique().tolist())
    if not all(s.startswith(md5[:8]) for s in info["stakes_panel_md5"]) and not allow_interim:
        raise SystemExit(f"{P3_STAKES.name} is on {info['stakes_panel_md5']}, the context on {md5[:8]}")
    st["qi"] = qi(st["year"], st["quarter"])
    K = K.merge(st[["mca_code", "qi", "S_tilde"]].rename(columns={"mca_code": "m"}), on=["m", "qi"], how="left")
    # advertising
    ad, big5 = advertising()
    K = K.merge(ad, on=["j", "qi"], how="left")
    K["adv_negative"] = K["adv_negative"].fillna(0).astype(int)
    K["big5"] = K["j"].isin(big5).astype(float)
    K["ladv_other"], K["ladv_big5"] = K["ladv"] * (1 - K["big5"]), K["ladv"] * K["big5"]
    info["big5"] = big5
    # sweeps
    for nm_, (jj, yq) in SWEEPS.items():
        K[nm_] = ((K["j"] == jj) & (K["qi"] >= qi(*yq))).astype(float)
    # sample flags
    K["zero_premium"] = zero_premium(K["qi"]).astype(int)
    K["bb_caixa"] = K["j"].isin([k0.BB, k0.CAIXA]).astype(int)
    tot = U.groupby(["m", "j"])["D4"].sum().reset_index().sort_values(["m", "D4", "j"], ascending=[True, False, True])
    dom = tot.drop_duplicates("m").set_index("m")["j"]
    K["dominant"] = (K["j"].to_numpy() == K["m"].map(dom).to_numpy()).astype(int)
    K["y"] = K["NF"] / K["own2"]
    info["rows"] = {"all": int(len(K)), "headline (no carried rate at t or t-1)": int((K["carried"] == 0).sum()),
                    "with control-function residual": int(K["v"].notna().sum()),
                    "with U_hat": int(K["U_hat"].notna().sum()), "with S_tilde": int(K["S_tilde"].notna().sum()),
                    "with advertising": int(K["ladv"].notna().sum()),
                    "negative advertising quarters": int(K["adv_negative"].sum())}
    info["pools"] = {"omega_mean": float(K["omega"].mean()), "lambda_mean": float(K["lam"].mean()),
                     "daily_share_of_type4_pools": float((K["Q4d"].sum() + K["R4d"].sum()) / (K["Q4"].sum() + K["R4"].sum()))}
    return K, info


def guard(ctx, allow_interim, n_markets):
    N0 = ctx["N0"]
    k0md5 = str(json.loads(K0_NUMBERS.read_text(encoding="utf-8")).get("panel_md5", ""))
    final = (str(N0.get("panel_md5", "")).startswith(FINAL_PANEL) and str(N0.get("frame_panel_md5", "")).startswith(FINAL_PANEL)
             and k0md5.startswith(FINAL_PANEL))
    rec = {"final_inputs": bool(final), "context_panel": N0.get("panel_md5"), "frame_panel": N0.get("frame_panel_md5"),
           "k0_numbers_panel": k0md5}
    if final:
        return rec
    if not allow_interim:
        raise SystemExit(f"not the final inputs ({rec}): real-data fits wait for I2's frames and the K0 rerun on "
                         f"{FINAL_PANEL}; --allow-interim runs a smoke test only")
    if not 0 < n_markets <= MAX_INTERIM_MARKETS:
        raise SystemExit(f"interim inputs: smoke tests only, --markets 1 to {MAX_INTERIM_MARKETS}")
    return rec


# -------------------------------------------------------------------------------------------- specs
ZK_NAME = {"post": "dkappa", "post_U": "kappa_U", "post_cad": "kappa_Pcad", "post_pop": "kappa_Ppop",
           "trend_pre": "kappa_trend", "U_hat": "kappa_Upre", "cad_z": "kappa_cad", "logpop_z": "kappa_pop",
           "placebo_2018q4": "dkappa", "placebo_2019q4": "dkappa", "S_tilde_z": "kappa_S",
           "ladv_other": "kappa_adv", "ladv_big5": "kappa_adv_big5"}
ZA_NAME = {"post": "alpha_P", "post_U": "alpha_PU", "post_cad": "alpha_Pcad", "post_pop": "alpha_Ppop",
           "placebo_2018q4": "alpha_P", "placebo_2019q4": "alpha_P", "S_tilde_z": "alpha_S"}
DOSE = ["post", "post_U", "post_cad", "post_pop"]
SPECS = {
    "level": dict(),
    "before_after": dict(za=["post"], zk=["post"], tests=["dkappa", "alpha_P"]),
    "dose": dict(za=DOSE, zk=DOSE, need=["U_hat"], tests=["kappa_U", "alpha_PU"]),
    "pretrend": dict(zk=["trend_pre", "U_hat", "cad_z", "logpop_z"], rows="pre", need=["U_hat"],
                     tests=["kappa_trend", "kappa_Upre"]),
    "placebo_2018q4": dict(za=["placebo_2018q4"], zk=["placebo_2018q4"], rows="pre", tests=["dkappa", "alpha_P"]),
    "placebo_2019q4": dict(za=["placebo_2019q4"], zk=["placebo_2019q4"], rows="pre", tests=["dkappa", "alpha_P"]),
    "stakes": dict(za=["S_tilde_z"], zk=["S_tilde_z"], need=["S_tilde"], tests=["kappa_S", "alpha_S"]),
    "advertising": dict(zk=["ladv_other", "ladv_big5"], need=["ladv"], tests=["kappa_adv_big5", "kappa_adv"]),
    "sweep": dict(zq=["sweep_santander", "sweep_bb"], tests=["a_sweep_santander", "a_sweep_bb"]),
    "a6_bins": dict(xe=["R2_t2", "R2_t3"], tests=["scale_R2_t2", "scale_R2_t3"]),
}
SAMPLES = ("zero_premium", "carried_kept", "no_bb_caixa", "no_dominant", "p99", "no_one_bank", "no_cf", "no_negative_adv")


def rows_for(K, spec, sample, markets=None, trim=None):
    """The estimation rows of a spec: the headline rule, the spec's needs, the robustness sample, the
    p99 trim (trim="p99": |NF/Dep_j2| at or below the 99th percentile of these rows), the smoke-test
    markets, then kn.informative_rows; sorted by market as kn.Effects requires."""
    sp = SPECS[spec]
    keep = np.ones(len(K), bool) if sample == "carried_kept" else np.array(K["carried"].to_numpy() == 0, dtype=bool)
    cf = sample != "no_cf"
    if cf:
        keep &= K["v"].notna().to_numpy()
    for c in sp.get("need", []):
        keep &= K[c].notna().to_numpy()
    if sp.get("rows") == "pre":
        keep &= (K["qi"] < qi(*POST_FROM)).to_numpy()
    if sample == "zero_premium":
        keep &= K["zero_premium"].to_numpy() == 1
    elif sample == "no_bb_caixa":
        keep &= K["bb_caixa"].to_numpy() == 0
    elif sample == "no_dominant":
        keep &= K["dominant"].to_numpy() == 0
    elif sample == "no_one_bank":
        keep &= K["one_B_bank"].to_numpy() == 0
    elif sample == "no_negative_adv":
        keep &= K["adv_negative"].to_numpy() == 0
    elif sample == "p99":
        ay = np.abs(K["y"].to_numpy())
        keep &= ay <= np.quantile(ay[keep], 0.99)
    if trim == "p99" and sample != "p99":
        ay = np.abs(K["y"].to_numpy())
        keep &= ay <= np.quantile(ay[keep], 0.99)
    if markets is not None:
        keep &= K["m"].isin(markets).to_numpy()
    Kw = K[keep]
    a = (Kw["j"] + "|" + Kw["m"]).to_numpy()
    Kw = Kw[kn.informative_rows(a, Kw["mt"].to_numpy())].sort_values(["m", "qi", "j"]).reset_index(drop=True)
    return Kw, cf


def design(Kw, MT, spec, cf, exit_by_cell=False, matured_scale=False, savings_only=False):
    """The model and the arrays of one spec. Below DAILY_MIN the daily-liquidity pools are set to zero
    (Q4d = R4d = 0, no daily exit) and the time-deposit pair is dropped: in the headline lambda is about
    zero and kappa_td is not estimable."""
    sp = SPECS[spec]
    cols = k2.design(Kw, MT)
    T = 1 + cols["Q1"] + cols["Q4"] + cols["R1"] + cols["R2"] + cols["R4"] + cols["O"]
    for c in ("Q4m", "Q4d", "R4m", "R4d"):
        cols[c] = Kw[c].to_numpy(float)
    daily = float((cols["Q4d"].sum() + cols["R4d"].sum()) / max(cols["Q4"].sum() + cols["R4"].sum(), 1e-300))
    td = daily >= DAILY_MIN
    Em, Ed = Kw["Em"].to_numpy(float), Kw["Ed"].to_numpy(float)
    if not td:
        cols["Q4d"], cols["R4d"], Ed = np.zeros(len(Kw)), np.zeros(len(Kw)), np.zeros(len(Kw))
    n = len(Kw)
    X = Kw.copy()
    if "S_tilde_z" in sp.get("za", []) + sp.get("zk", []):
        s = X["S_tilde"]
        X["S_tilde_z"] = (s - s.mean()) / s.std()
    if sp.get("xe"):
        t = pd.qcut(X["R2"], 3, labels=False)
        X["R2_t2"], X["R2_t3"] = (t == 1).astype(float), (t == 2).astype(float)
    Za = np.column_stack([np.ones(n)] + [X[c].to_numpy(float) for c in sp.get("za", [])])
    Zk = np.column_stack([np.ones(n)] + [X[c].to_numpy(float) for c in sp.get("zk", [])])
    xe_cols = (["v"] if cf else []) + sp.get("xe", [])
    xe_names = (["cf_v"] if cf else []) + [f"scale_{c}" for c in sp.get("xe", [])]
    Xe = np.column_stack([X[c].to_numpy(float) for c in xe_cols]) if xe_cols else None
    zq = sp.get("zq", [])
    if zq and savings_only:
        raise SystemExit(f"spec {spec}: the sweep shifters act on own checking, which --savings-only removes")
    Zq = np.column_stack([X[c].to_numpy(float) for c in zq]) if zq else None
    model = KModel(cols, X["g"].to_numpy(float), Za=Za, Zk=Zk,
                   za_names=["alpha"] + [ZA_NAME[c] for c in sp.get("za", [])],
                   zk_names=["kappa"] + [ZK_NAME[c] for c in sp.get("zk", [])], td=td,
                   Xe=Xe, xe_names=xe_names, Zq=Zq, zq_names=[f"a_{c}" for c in zq], exit_const=not exit_by_cell,
                   matured_scale=matured_scale, savings_only=savings_only)
    S = {"y": X["y"].to_numpy(float), "E": (Em + Ed) if exit_by_cell else Ed, "Em": None if exit_by_cell else Em,
         "a": (X["j"] + "|" + X["m"]).to_numpy(), "b": X["mt"].to_numpy(), "m": X["m"].to_numpy(),
         "cl": X["j"].to_numpy(), "d": (1 / T), "daily_share": daily, "td": td}
    return model, S


def run_spec(K, MT, label, markets, draws, exit_by_cell=False, trim=None, starts=STARTS, maxit=MAXIT,
             matured_scale=False, savings_only=False, smooth=False):
    """One spec: the multistart fit, then up to two reported solutions (Pedro, 11-06): INTERIOR, the
    lowest-SSR start with no parameter at its box, and BOUNDARY, the lowest-SSR start overall when that
    start has a box hit. Each solution gets its own score bootstrap (at its own Jacobian and residual)
    and its own restricted fits (started from it). Returns one row per solution (column `solution`)."""
    spec, _, sample = label.partition("+")
    if spec not in SPECS or (sample and sample not in SAMPLES):
        raise SystemExit(f"unknown spec {label}")
    t0 = time.time()
    Kw, cf = rows_for(K, spec, sample or None, markets, trim=trim)
    model, S = design(Kw, MT, spec, cf, exit_by_cell, matured_scale, savings_only)
    _, _, d, _, _ = multistart_k1(S["y"], S["E"], S["a"], S["b"], S["m"], S["cl"], model, starts=starts,
                                  rowscale=S["d"], Em=S["Em"], maxit=maxit, smooth=smooth)
    fits = d["fits"]
    ssr = [f["d"]["ssr_over_yy"] for f in fits]
    interior = [i for i, f in enumerate(fits) if not f["d"]["at_box"]]
    best = int(np.argmin(ssr))
    sols = []
    if interior:
        sols.append(("interior", min(interior, key=lambda i: ssr[i])))
    if fits[best]["d"]["at_box"]:
        sols.append(("boundary", best))
    common = {"spec": label, "n": d["n"], "G": d["G"], "markets": d["markets"], "cells": d["cells"],
              "starts_spread": d["starts_spread"], "start_kappas": ";".join(f"{v:.6f}" for v in d["start_kappas"]),
              "daily_share": S["daily_share"],
              "exit": "theta_w free by market-quarter (whole movable stock)" if exit_by_cell else "theta_w one constant",
              "trim": trim or "none", "maxit": maxit, "starts": ";".join(f"{v:g}" for v in d["starts"]),
              "best_start": d["best_start"], "matured_scale": bool(matured_scale),
              "savings_only": bool(savings_only), "smooth_effects": bool(smooth)}
    for key in ("start_dkappas", "start_ssr", "start_floored"):
        if key in d:
            common[key] = ";".join(f"{v:.6g}" for v in d[key])
    for key in ("start_converged", "start_iterations", "start_reasons", "start_at_box", "start_seconds"):
        common[key] = ";".join(str(v) for v in d[key])
    cs = effective_cluster_stats(pd.Series(S["cl"]).value_counts().to_numpy())
    common["Gstar"], common["cluster_size_cv"] = cs["G_star"], cs["cv"]
    if not interior:
        common["interior_note"] = "every start ends with a parameter at its box"
    contrasts = {}
    if S["td"]:
        c_ = np.zeros(model.p)
        c_[model.names.index("kappa")], c_[model.names.index("kappa_td")] = 1.0, -1.0
        contrasts["ln_Ec"] = c_
    rows = []
    for sol, i in sols:
        t1 = time.time()
        f = fits[i]
        V, dd, Jt, r = f["V"], f["d"], f["Jt"], f["r"]
        s = np.array([f["est"][n] for n in model.names])
        wcb = score_wcb(Jt, r, S["cl"], draws=draws, contrasts=contrasts)
        se_fit = np.sqrt(np.maximum(np.diag(V.to_numpy()), 0))
        row = {**common, "solution": sol, "solution_start": f["start"], "converged": dd["converged"],
               "reason": dd["reason"], "iterations": dd["iterations"], "at_box": ";".join(dd["at_box"]),
               "ssr_over_yy": dd["ssr_over_yy"], "grad_norm_start": dd["grad_norm_start"],
               "grad_norm_end": dd["grad_norm_end"], "cos_end": dd["gradient_cosine"],
               "ssr_rel_change_last100": dd["ssr_rel_change_last100"], "ssr_change_window": dd["ssr_change_window"]}
        if smooth:
            for key in ("floored_effects", "floored_split", "capped_exit_cells", "collinear_cells", "components",
                        "inner_solves", "inner_iterations_first",
                        "inner_iterations_median", "inner_iterations_max", "inner_cap_hits"):
                row[key] = dd[key]
        for j, nm_ in enumerate(model.names):
            row[nm_], row[f"se_{nm_}"] = float(s[j]), float(se_fit[j])
            row[f"se_boot_{nm_}"] = float(wcb["se_boot"][j])
            row[f"ci_lo_{nm_}"], row[f"ci_hi_{nm_}"] = float(s[j] - wcb["half_width"][j]), float(s[j] + wcb["half_width"][j])
        if S["td"]:
            lc = float(s[model.names.index("kappa")] - s[model.names.index("kappa_td")])
            cw = wcb["contrasts"]["ln_Ec"]
            row.update({"ln_Ec": lc, "se_ln_Ec": cw["se_cr1"], "se_boot_ln_Ec": cw["se_boot"],
                        "ci_lo_ln_Ec": lc - cw["half_width"], "ci_hi_ln_Ec": lc + cw["half_width"],
                        "ln_Ec_note": "kappa (savings) - kappa_td at the base level of Zk, under A4"})
        else:
            row["kappa_td_note"] = (f"not estimable: the daily-liquidity share of the type-4 pools is {S['daily_share']:.1e} "
                                    f"(below {DAILY_MIN:g}; statement lambda about zero), so Q4d = R4d = 0; see --lambda-one")
        for nm_ in SPECS[spec].get("tests", []):
            k = model.names.index(nm_)
            lo, hi = model.box()
            lo[k] = hi[k] = 0.0
            s0 = s.copy()
            s0[k] = 0.0
            _, _, d0, Jt0, r0 = nls_fit_k1(S["y"], S["E"], S["a"], S["b"], S["m"], S["cl"], model, s0, rowscale=S["d"],
                                           box=(lo, hi), Em=S["Em"], maxit=maxit, smooth=smooth)
            t = score_test(Jt0, r0, S["cl"], k, draws=draws)
            row[f"p_{nm_}"], row[f"T_{nm_}"], row[f"restricted_converged_{nm_}"] = t["p"], t["T"], d0["converged"]
        row["seconds"] = round(time.time() - t1, 1)
        row["spec_seconds"] = round(time.time() - t0, 1)
        stamp(f"  {label} [{sol}, start {f['start']:g}]: " + ", ".join(f"{n} {row[n]:.4f}" for n in model.names) +
              (f", ln_Ec {row['ln_Ec']:.4f}" if S["td"] else "; kappa_td not estimable") +
              f"; converged {dd['converged']} ({dd['reason']}, {dd['iterations']} it), box [{row['at_box']}], "
              + (f"floored effects {dd['floored_effects']}, " if smooth else "") +
              f"SSR/yy {dd['ssr_over_yy']:.5f}, n {d['n']}, G {d['G']}")
        rows.append(row)
    return rows


# -------------------------------------------------------------------------------------------- tests
def test1(ctx, smooth=False):
    """nls_fit_k1 with K2's Model (no theta_w) on simulated y, fe world at kappa 0.5, per pair: the
    two-start fit equals step 2's row of k2_nls_recovery.csv to 1e-8, standard errors included."""
    e4 = kn.world_e4(ctx, "fe")
    S = kn.sample(ctx, e4)
    A, NF = k2.simulate(ctx["U"], ctx["MT"], ctx["SD"], ctx["SD4"], e4, kn.KAPPA_BASE)
    y = NF[S["u"]] / S["own2"]
    m0 = kn.Model(S["cols"], S["g"], split=True)
    est, V, d, Jt, r = multistart_k1(y, S["E"], S["a"], S["b"], S["m"], S["cl"], m0, starts=kn.STARTS, rowscale=S["d"],
                                     maxit=200, smooth=smooth)
    R = pd.read_csv(RECOVERY)
    ref = R[(R["world"] == "fe") & (R["kappa_true"] == kn.KAPPA_BASE) & (R["model"] == "split")].iloc[0]
    diff = {n: float(est[n] - ref[n]) for n in m0.names}
    se_f = np.sqrt(np.maximum(np.diag(V.to_numpy()), 0))
    se_ref = np.array([ref[f"se_{n}"] for n in m0.names])
    se_s = score_wcb(Jt, r, S["cl"], draws=1)["se_cr1"]
    out = {"estimates": {n: float(est[n]) for n in m0.names}, "stored": {n: float(ref[n]) for n in m0.names},
           "max_abs_diff_vs_stored": max(abs(v) for v in diff.values()),
           "se_max_rel_diff_vs_stored": float(np.max(np.abs(se_f - se_ref) / np.maximum(se_ref, 1e-300))),
           "score_se_max_rel_gap": float(np.max(np.abs(se_s - se_f) / np.maximum(se_f, 1e-300))),
           "converged": d["converged"], "n": d["n"], "smooth_effects": bool(smooth)}
    out["passed"] = bool(out["max_abs_diff_vs_stored"] <= 1e-8 and out["se_max_rel_diff_vs_stored"] <= 1e-6
                         and out["score_se_max_rel_gap"] <= 1e-6)
    stamp(f"test 1: max |estimate - stored| {out['max_abs_diff_vs_stored']:.2e}; se against stored "
          f"{out['se_max_rel_diff_vs_stored']:.1e}; the bootstrap's (Jt, r) reproduce V to {out['score_se_max_rel_gap']:.1e}; "
          f"{'passed' if out['passed'] else 'FAILED'}")
    return out


def test2(ctx, savings_only=False, smooth=False):
    """y from the K1 equation (both exit terms, all three pairs with lambda > 0, the control function,
    Pix shifters, a sweep, scale bins, the own matured scale a_m) on the context's K1 rows with synthetic omega, lambda, rivals'
    split, v, U and random effects; one start at zero."""
    rng = np.random.default_rng(SEED)
    K = ctx["K"].copy()
    a0 = (K["j"] + "|" + K["m"]).to_numpy()
    K = K[kn.informative_rows(a0, K["mt"].to_numpy())].sort_values(["m", "qi", "j"]).reset_index(drop=True)
    jy = pd.factorize(K["j"] + "|" + (K["qi"] // 4).astype(str))[0]
    om, la = rng.uniform(0.1, 0.9, jy.max() + 1)[jy], rng.uniform(0.2, 0.8, jy.max() + 1)[jy]
    K["Q4m"], K["Q4d"] = om * K["Q4"], (1 - om) * la * K["Q4"]
    sh = rng.uniform(0.2, 0.8, len(K))
    K["R4m"], K["R4d"] = sh * K["R4"], (1 - sh) * rng.uniform(0.2, 1, len(K)) * K["R4"]
    accr = (K["carry4"] / K["own2"]).to_numpy()
    Em, Ed = accr * om, accr * (1 - om) * la
    jq = pd.factorize(K["j"] + "|" + K["qi"].astype(str))[0]
    K["v"] = rng.normal(0, 0.5, jq.max() + 1)[jq]
    mm = pd.factorize(K["m"])[0]
    K["post"] = (K["qi"] >= qi(*POST_FROM)).astype(float)
    K["post_U"] = K["post"] * rng.normal(0, 1, mm.max() + 1)[mm]
    K["sweep_santander"] = ((K["j"] == SWEEPS["sweep_santander"][0]) & (K["qi"] >= qi(2022, 2))).astype(float)
    t = pd.qcut(K["R2"], 3, labels=False)
    K["R2_t2"], K["R2_t3"] = (t == 1).astype(float), (t == 2).astype(float)
    cols = k2.design(K, ctx["MT"])
    for c in ("Q4m", "Q4d", "R4m", "R4d"):
        cols[c] = K[c].to_numpy()
    n = len(K)
    Z = np.column_stack([np.ones(n), K["post"], K["post_U"]])
    model = KModel(cols, K["g"].to_numpy(float), Za=Z, Zk=Z, za_names=("alpha", "alpha_P", "alpha_PU"),
                   zk_names=("kappa", "dkappa", "kappa_U"), td=True,
                   Xe=K[["v", "R2_t2", "R2_t3"]].to_numpy(), xe_names=("cf_v", "scale_R2_t2", "scale_R2_t3"),
                   Zq=None if savings_only else K[["sweep_santander"]].to_numpy(),
                   zq_names=() if savings_only else ("a_sweep_santander",), exit_const=True,
                   matured_scale=not savings_only, savings_only=savings_only)
    truth = {"alpha": -0.117, "alpha_P": 0.01, "alpha_PU": -0.005, "kappa": 1.0, "dkappa": -0.5, "kappa_U": -0.15,
             "kappa_chk": 0.6, "dkappa_chk": -0.2, "kappa_U_chk": -0.05, "kappa_td": 1.3, "dkappa_td": -0.4,
             "kappa_U_td": -0.1, "a": 0.3, "a_null": -0.2, "cf_v": 0.05, "scale_R2_t2": 0.2, "scale_R2_t3": -0.1,
             "a_sweep_santander": 0.5, "a_m": 0.8, "theta_w": 0.15}
    if savings_only:
        truth = {k: v for k, v in truth.items() if not (k.endswith("_chk") or k in ("a", "a_sweep_santander", "a_m"))}
    if sorted(truth) != sorted(model.names):
        raise SystemExit(f"test 2 names: {model.names}")
    s_true = np.array([truth[nm_] for nm_ in model.names])
    h, _ = model.h(s_true)
    ca, cb = pd.factorize(K["j"] + "|" + K["m"])[0], pd.factorize(K["mt"])[0]
    la_, lb_ = rng.normal(0, 1, ca.max() + 1)[ca], rng.normal(-4, 1, cb.max() + 1)[cb]
    cx = (0.01 * np.exp(rng.normal(0, 0.5, cb.max() + 1)))[cb]
    y = np.exp(la_ + lb_) * h - cx * Ed - np.exp(truth["theta_w"]) * Em
    T = 1 + cols["Q1"] + cols["Q4"] + cols["R1"] + cols["R2"] + cols["R4"] + cols["O"]
    est, V, d, Jt, r = nls_fit_k1(y, Ed, (K["j"] + "|" + K["m"]).to_numpy(), K["mt"].to_numpy(), K["m"].to_numpy(),
                                  K["j"].to_numpy(), model, model.start(0.0), rowscale=1 / T, box=model.box(), Em=Em,
                                  smooth=smooth)
    err = {nm_: float(est[nm_] - truth[nm_]) for nm_ in model.names}
    out = {"n": int(n), "truth": truth, "estimates": {k: float(v) for k, v in est.items()},
           "max_abs_error": max(abs(v) for v in err.values()), "converged": d["converged"], "reason": d["reason"],
           "iterations": d["iterations"], "ln_Ec_true": truth["kappa"] - truth["kappa_td"],
           "ln_Ec_hat": float(est["kappa"] - est["kappa_td"]), "smooth_effects": bool(smooth)}
    out["passed"] = bool(out["max_abs_error"] <= 1e-6)
    stamp(f"test 2: {n:,} rows, {len(model.names)} parameters; max |estimate - truth| {out['max_abs_error']:.2e} "
          f"({d['reason']}, {d['iterations']} iterations); theta_w {est['theta_w']:.6f} against 0.15, ln_Ec "
          f"{out['ln_Ec_hat']:.6f} against {out['ln_Ec_true']:.2f}: {'passed' if out['passed'] else 'FAILED'}")
    return out


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=("1", "2", "2s"), default=None, help="2s: test 2 in the --savings-only form")
    ap.add_argument("--spec", default=None, help="comma-separated specs, e.g. before_after,dose,before_after+no_bb_caixa")
    ap.add_argument("--ctx", default=None, help="a context pickle (kn.build_ctx); built from --panel when absent")
    ap.add_argument("--panel", default=None, help="the market panel the context was built on")
    ap.add_argument("--markets", type=int, default=0, help="smoke test: a random subset of markets (kn's draw)")
    ap.add_argument("--allow-interim", action="store_true", help="non-final inputs, smoke tests only")
    ap.add_argument("--lambda-one", action="store_true", help="lambda = 1 for every bank (D2): the time-deposit pair and ln E_c")
    ap.add_argument("--exit-by-cell", action="store_true", help="robustness: the matured exit coefficient free by market-quarter")
    ap.add_argument("--trim", choices=("none", "p99"), default="none",
                    help="p99: drop rows with |NF/Dep_j2| above the 99th percentile of each spec's rows (the headline)")
    ap.add_argument("--starts", default=",".join(f"{v:g}" for v in STARTS), help="comma-separated kappa-tilde starts")
    ap.add_argument("--maxit", type=int, default=MAXIT, help="iteration cap of every fit")
    ap.add_argument("--matured-scale", action="store_true",
                    help="own matured money (Q4m) at its own scale a_m, apart from own checking a (pilot of 11-06)")
    ap.add_argument("--savings-only", action="store_true",
                    help="the 11-03 savings pair: own and rivals' checking and matured pools out of the bracket (no a, no checking pair)")
    ap.add_argument("--smooth-effects", action="store_true",
                    help="K1-S (Pedro 11-07): effects as exp(psi), Gauss-Newton inner solve, warm-started (SmoothProfile)")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.ctx and Path(args.ctx).exists():
        with open(args.ctx, "rb") as fh:
            ctx = pickle.load(fh)
        stamp(f"context from {args.ctx}: panel {ctx['N0'].get('panel_md5')}")
    elif args.panel:
        ctx = kn.build_ctx(args.panel)
    else:
        raise SystemExit("give --ctx or --panel")
    N = {"script": "kappa/k1_netflow_nls.py", "run": f"{datetime.now():%Y-%m-%d %H:%M}", "args": vars(args),
         "context_N0_panel": ctx["N0"].get("panel_md5")}
    path = out / "k1_numbers.json"
    if args.test:
        key = f"test_{args.test}" + ("_smooth" if args.smooth_effects else "")
        N[key] = (test1(ctx, smooth=args.smooth_effects) if args.test == "1"
                  else test2(ctx, savings_only=args.test == "2s", smooth=args.smooth_effects))
    else:
        if not args.spec or not args.panel:
            raise SystemExit("--spec and --panel are needed for a fit")
        N["guard"] = guard(ctx, args.allow_interim, args.markets)
        markets = None
        if args.markets:
            markets = set(pd.Series(ctx["K"]["m"].unique()).sample(args.markets, random_state=3))
            N["smoke_test_markets"] = args.markets
        K, N["inputs"] = assemble(ctx, args.panel, args.allow_interim, lambda_one=args.lambda_one)
        rows = []
        for label in args.spec.split(","):
            rows.extend(run_spec(K, ctx["MT"], label, markets, args.draws, exit_by_cell=args.exit_by_cell,
                                 trim=None if args.trim == "none" else args.trim,
                                 starts=tuple(float(v) for v in args.starts.split(",")), maxit=args.maxit,
                                 matured_scale=args.matured_scale, savings_only=args.savings_only,
                                 smooth=args.smooth_effects))
            pd.DataFrame(rows).to_csv(out / "k1_estimates.csv", index=False)
    N["seconds"] = round(time.time() - t0)
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(N)
    path.write_text(json.dumps(old, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")
    stamp(f"done in {N['seconds']}s; {path}")


if __name__ == "__main__":
    main()
