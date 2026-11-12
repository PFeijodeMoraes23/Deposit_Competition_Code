"""
diag_p4_adv_group_diff.py
=========================
The test of the five largest B conglomerates' advertising elasticity against the rest's, on P4's
groups_B regression (diag_p4_openings_advertising.py, fit g1; panel file and JSON of P8):
    ln o_jt = a_j + t_q + beta x_jt + delta x_jt 1[j in S] + p rho_j,t-1 + e_jt,     x = ln(1 + adv),
S the five largest by 2019 deposits (P2's rank_2019), unweighted, clusters by conglomerate. The same
on the intensive margin (x = ln adv on quarters with adv > 0; P4's intensive_B, fit i1). Both
specifications are P4's, unchanged. SIGN: the reported contrast is rest minus five largest = -delta
(P4 stores delta as adv_big5_minus_rest); its intervals are delta's flipped; p-values and degrees of
freedom do not depend on the sign.

Inference, all two-sided:
  CR1   P4's Fit (CR1 by conglomerate, P4's small-sample factor); p from t(G - 1).
  WCR   P4's wild cluster restricted bootstrap with P4's own draws: default_rng(SEED) advanced by the
        Webb draws of every fit P4's main() makes before the one replayed (each fit draws
        choice(WEBB, (G, N_BOOT)) once), so g1, g2, i1 and i2 get the draws P4 used. p at zero and the
        95 percent interval by P4's inversion (P4 recorded no interval for the intensive margin).
  RI-t  randomization inference over which five conglomerates carry the label (MacKinnon and Webb
        2020), on the CR1 t of the relabelled regression. Pool: the B conglomerates of the sample that
        advertise in every sample quarter, with x varying and at least POOL_MIN_QUARTERS sample
        quarters (the design alone defines it; the same codes for both margins). Every five-set of the
        pool is enumerated; p = the share of five-sets with |t_S| >= |t_S0|, the real label S0
        included. The 95 percent set inverts the sharp null delta = d0 (outcome y - d0 x 1[j in S0]):
        a grid over d0, each change of verdict refined by bisection. The RI on |delta-hat| is reported
        beside it, with a divergence warning when the two cross ALPHA differently. Secondary pools
        (no keys): every always-advertiser, the top 15 by 2019 deposits, and every conglomerate with x
        varying (N_RANDOM random five-sets plus S0; not exchangeable with the label).
  CR2   Bell and McCaffrey (2002): A_g the symmetric square root of the Moore-Penrose inverse of
        I - H_gg with both sets of effects explicit (Pustejovsky and Tipton 2018), V = sum_g
        (l'(X'X)^-1 X_g' A_g e_g)^2 and the Bell-McCaffrey degrees of freedom under the working model
        Omega = I; for the difference and for each group's level.
Diagnostics (no keys): each cluster's share of the partial leverage of x 1[j in S0] (its squared mass
after x, rho and both sets of effects), the effective clusters 1 / sum(w^2) and Caixa's share; the
difference with each of the five dropped from the sample (Caixa first; the four left keep the label)
and with the big four labelled and Caixa kept in the sample among the rest, each with CR1, a wild
bootstrap p (fresh draws from default_rng(SEED)) and CR2, and, without Caixa and for the big four,
the RI-t over every four-set of the pool (less Caixa when Caixa is out of the sample); where S0 sits
among the pool's five-sets (effective clusters, squared mass); the share of the RI distribution that
the members of a set explain.

Gates, run before anything is written; a failure prints MISMATCH, writes the JSON with the gates and
no results, and exits with status 1:
  G1  the panel: the P4 CSV's panel_md5 is the first 8 characters of the P4 JSON's _meta.panel_md5
  G2  n and G of every P4 fit replayed equal the P4 JSON's; the label is P2's rank_2019 top five; the
      five largest and the rest hold 5 and 63 clusters (big5_alone, rest_alone); the pool's members
      keep the same quarters on the intensive margin
  G3  the label's Frisch-Waugh estimate and CR1 se (the RI's own statistic) equal P4's Fit to TOL_FIT
      and the P4 JSON's at 4 dp, on both margins; the closed-form t of N_RELABEL_CHECK random five-sets
      of the pool, under the nulls 0 and the estimate +- 3 se, equals P4's Fit on the relabelled
      design to TOL_T
  G4  the replayed bootstrap reproduces every P4 JSON p-value of g1, g2, i1 and i2 and the intervals
      of the difference (g1) and of the five largest's level (g2)
  G5  every five-set of the primary pool appears once, the real label among them
  G6  CR2 unit tests: an intercept with equal clusters (G / (G - 1) CR0, df G - 1); singletons (HC2);
      a two-group dummy with singletons (Welch variance and its Satterthwaite df); explicit against
      absorbed cluster effects (Pustejovsky and Tipton's invariance) on a simulated panel and on both
      margins of the data; the degrees of freedom against a brute-force n x n computation (simulated
      and data); the five largest's level from the difference design against P4's level design;
      A_g = I reproducing P4's CR1
Inputs (read only): Rout/p4_openings_advertising.csv and .json (P4), Rout/p2_advertising_coverage_B.csv
(P2). No panel read. Output: Rout/p4_adv_group_diff.json, with the raw and LF md5 of each input;
make_paper_numbers.py registers pix.adv.diff.*, pix.adv.diff_int.* and pix.adv.big5.elast_p_cr2 from
its keys block only while those md5s are the inputs' and the gates passed.

Usage (from the repo root):  python diagnostics/diag_p4_adv_group_diff.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from itertools import combinations
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))        # repo root: utils/
sys.path.insert(0, str(HERE))               # diagnostics/: P4's module, also under python -I
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import numpy as np
import pandas as pd
from scipy import stats

from diag_p4_openings_advertising import (COVERAGE, N_BOOT, OUT_CSV, OUT_JSON, RENAME, SEED, WEBB, Fit,
                                          design, md5, qi, stamp)

OUT = OUT_JSON.with_name("p4_adv_group_diff.json")
CAIXA = "C0080738"
POOL_MIN_QUARTERS = 12
ALPHA = 0.05
SEED_RI = 20261111          # the random five-sets of the every-conglomerate-with-variation pool
SEED_CHECK = SEED_RI + 1    # the five-sets G3 refits
N_RANDOM = 99_999
N_GRID = 10_001             # grid of the RI set's inversion
N_RELABEL_CHECK = 25
TOL_FIT = 1e-10             # G3, estimate and se of the label
TOL_T = 1e-8                # G3, t of the relabelled sets
CR2_TOL = 1e-12             # eigenvalues of I - H_gg at or below it count as zero (clubSandwich's rule)
TOL_CR2_TEST = 1e-8         # G6, relative
SIGN = ("rest minus five largest = -delta, delta the coefficient on x * 1[five largest] (P4's "
        "adv_big5_minus_rest); its intervals are delta's flipped; p-values and df do not depend on "
        "the sign")
POOL_RULE = (f"B conglomerates of the margin's sample with no zero-advertising sample quarter, x "
             f"varying and at least {POOL_MIN_QUARTERS} sample quarters (pooled sample); the same "
             f"codes on the intensive margin")
# The two margins: P4's fit of the difference and of the five largest's level, its JSON block.
MARGINS = {"pooled": dict(x="x", diff_fit="g1", level_fit="g2", block="groups_B"),
           "intensive": dict(x="x_pos", diff_fit="i1", level_fit="i2", block="intensive_B")}
REPLAYED = ("g1", "g2", "i1", "i2")
# P4 JSON entries the replay must reproduce: (fit, coefficient index, block, entry, interval too).
REPLAY_CHECKS = [("g1", 0, "groups_B", "adv_rest", False),
                 ("g1", 1, "groups_B", "adv_big5_minus_rest", True),
                 ("g1", 2, "groups_B", "price", False),
                 ("g2", 0, "groups_B", "adv_big5", True),
                 ("i1", 0, "intensive_B", "adv_rest", False),
                 ("i1", 1, "intensive_B", "adv_big5_minus_rest", False),
                 ("i2", 0, "intensive_B", "adv_big5", False)]
# P4 JSON blocks whose n and G the replayed samples must equal.
SIZE_CHECKS = {"main_B": "main_B", "main_B_no_price": "main_B_no_price", "g1": "groups_B",
               "big5_alone": "big5_alone", "rest_alone": "rest_alone", "lags_plain": "lags_B",
               "D_main": "D_main", "D_no_price": "D_no_price"}


def lf_md5(p: Path) -> str:
    """md5 with CRLF read as LF (make_paper_numbers._lf_md5)."""
    return hashlib.md5(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def dump(obj):
    """OUT, LF, written whole: a temporary file beside it, then replaced."""
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, OUT)


def flip(iv):
    """An interval of delta as one of -delta (None marks an unbounded end)."""
    return [None if iv[1] is None else -iv[1], None if iv[0] is None else -iv[0]]


# ------------------------------------------------------------------------------------------- data
def load() -> pd.DataFrame:
    """P4's panel file with P4's derived columns (main(), as its --only p5 path reads it)."""
    d = pd.read_csv(OUT_CSV, dtype={"CodConglomeradoPrudencial": str, "panel_md5": str}).rename(
        columns={v: k for k, v in RENAME.items()})
    d["qi"] = qi(d["year"], d["quarter"])
    d["x_pos"] = np.log(d["adv"].where(d["adv"] > 0))
    for v in ("x", "x_l1", "x_l2", "x_ratio", "x_pos"):
        d[f"{v}_b5"], d[f"{v}_rest"] = d[v] * d["big5"], d[v] * (1 - d["big5"])
    d["dl1"], d["dl2"] = d["x_l1"] - d["x"], d["x_l2"] - d["x"]
    for v in ("dl1", "dl2"):
        d[f"{v}_b5"], d[f"{v}_rest"] = d[v] * d["big5"], d[v] * (1 - d["big5"])
    return d


def p4_fits(d: pd.DataFrame) -> list:
    """P4 main()'s fits in the order they draw from default_rng(SEED): (name, frame, regressors)."""
    B, D = d[d["is_B"] == 1], d[d["is_B"] == 0]
    lags = B.dropna(subset=["x_l1", "x_l2"])
    return [("main_B", B, ["x", "rho_lag_pp"]),
            ("main_B_no_price", B.dropna(subset=["rho_lag_pp"]), ["x"]),
            ("g1", B, ["x", "x_b5", "rho_lag_pp"]),
            ("g2", B, ["x", "x_rest", "rho_lag_pp"]),
            ("big5_alone", B[B["big5"] == 1], ["x", "rho_lag_pp"]),
            ("rest_alone", B[B["big5"] == 0], ["x", "rho_lag_pp"]),
            ("lags_plain", lags, ["x_b5", "x_l1_b5", "x_l2_b5", "x_rest", "x_l1_rest", "x_l2_rest",
                                  "rho_lag_pp"]),
            ("lags_sum", lags, ["x_b5", "dl1_b5", "dl2_b5", "x_rest", "dl1_rest", "dl2_rest", "rho_lag_pp"]),
            ("D_main", D, ["x", "rho_lag_pp"]),
            ("D_no_price", D, ["x"]),
            ("ratio_1", B, ["x_ratio", "x_ratio_b5", "rho_lag_pp"]),
            ("ratio_2", B, ["x_ratio", "x_ratio_rest", "rho_lag_pp"]),
            ("i1", B, ["x_pos", "x_pos_b5", "rho_lag_pp"]),
            ("i2", B, ["x_pos", "x_pos_rest", "rho_lag_pp"])]


def replay(d: pd.DataFrame):
    """The Fits of REPLAYED with P4's bootstrap draws; every other fit only advances the generator by
    its own draw. -> ({name: (Fit, sample, regressors)}, {name: (n, G)})."""
    rng = np.random.default_rng(SEED)
    fits, sizes = {}, {}
    for name, frame, regs in p4_fits(d):
        s = frame.dropna(subset=["ln_o", *regs]).reset_index(drop=True)
        sizes[name] = (len(s), int(s["j"].nunique()))
        if name in REPLAYED:
            fits[name] = (Fit(design(s, regs), s["ln_o"].to_numpy(float), s["j"].to_numpy(), rng), s, regs)
        else:
            rng.choice(WEBB, size=(sizes[name][1], N_BOOT))
    return fits, sizes


# ------------------------------------------------------------------------------ randomization (RI)
class Relabel:
    """The difference regression with the label as the only moving part (Frisch-Waugh). The common
    block W = [x, rho, constant, conglomerate and quarter effects] is partialled out of ln o and of
    each candidate h's own column x 1[j = h] once (z_h); a labelled set S has z_S = sum over S of z_h,
    estimate z_S'y~ / z_S'z_S and CR1 t from the cluster sums of z_S (y~ - b z_S), with P4's
    small-sample factor (k = rank W + 1). Per cluster g it keeps sum_g z_h y~ and sum_g z_h z_h', so
    every set's statistics, and its t under the sharp null delta = d0 of the real label (outcome
    y - d0 x 1[j in S0]), are closed-form."""

    def __init__(self, s: pd.DataFrame, xcol: str, candidates: list, label: list):
        W = design(s, [xcol, "rho_lag_pp"])
        if np.linalg.matrix_rank(W) < W.shape[1]:
            raise SystemExit("the common block of the relabelled design is rank deficient")
        Q, _ = np.linalg.qr(W)
        self.codes, inv = np.unique(s["j"].to_numpy(), return_inverse=True)
        n, G = len(s), len(self.codes)
        pos = {c: i for i, c in enumerate(self.codes)}
        self.cand = list(candidates)
        x = s[xcol].to_numpy(float)
        Z = np.zeros((n, len(self.cand)))
        for k, c in enumerate(self.cand):
            m = inv == pos[c]
            Z[m, k] = x[m]
        y = s["ln_o"].to_numpy(float)
        yt, Zt = y - Q @ (Q.T @ y), Z - Q @ (Q.T @ Z)
        self.A = np.zeros((G, len(self.cand)))
        self.C = np.zeros((G, len(self.cand), len(self.cand)))
        for g in range(G):
            m = inv == g
            self.A[g], self.C[g] = Zt[m].T @ yt[m], Zt[m].T @ Zt[m]
        self.s0 = np.array(sorted(self.cand.index(c) for c in label))
        self.CL = self.C[:, :, self.s0].sum(axis=2)            # sum_g z_h z_S0
        self.adj = G / (G - 1) * (n - 1) / (n - (W.shape[1] + 1))
        self.n, self.G, self.k = n, G, W.shape[1] + 1

    def positions(self, codes) -> np.ndarray:
        return np.array(sorted(self.cand.index(c) for c in codes))

    def stats(self, sets: np.ndarray, chunk: int = 4000) -> np.ndarray:
        """Per five-set (rows, positions in self.cand): columns a = z_S'y~, c = z_S'z_S0, D = z_S'z_S,
        the cluster-score moments Suu, Suv, Svv (score_g(d0) = u_g + d0 v_g), effective clusters
        1 / sum(w^2) and the largest share w of z_S'z_S."""
        out = []
        for k in range(0, len(sets), chunk):
            S = sets[k:k + chunk]
            a = self.A[:, S].sum(axis=2)
            b = self.C[:, S[:, :, None], S[:, None, :]].sum(axis=(2, 3))
            e = self.CL[:, S].sum(axis=2)
            D, aS, cS = b.sum(0), a.sum(0), e.sum(0)
            u, v, w = a - (aS / D) * b, (cS / D) * b - e, b / D
            out.append(np.column_stack([aS, cS, D, (u * u).sum(0), (u * v).sum(0), (v * v).sum(0),
                                        1.0 / (w * w).sum(0), w.max(0)]))
        return np.vstack(out)

    def shares(self) -> np.ndarray:
        """Each cluster's share of the real label's z_S0'z_S0."""
        b = self.C[:, self.s0[:, None], self.s0[None, :]].sum(axis=(1, 2))
        return b / b.sum()


def t2(st: np.ndarray, d0) -> np.ndarray:
    """adj x the squared CR1 t of every set under the sharp null delta = d0 (rows: d0)."""
    d0 = np.atleast_1d(np.asarray(d0, float))[:, None]
    num = st[:, 0] - d0 * st[:, 1]
    return num ** 2 / (st[:, 3] + 2 * d0 * st[:, 4] + d0 ** 2 * st[:, 5])


def ri_p(st: np.ndarray, i0: int, d0) -> np.ndarray:
    """RI-t p of the real label (row i0) at each d0: the share of sets with |t_S| >= |t_S0|."""
    d0 = np.atleast_1d(np.asarray(d0, float))
    out = np.empty(len(d0))
    for k in range(0, len(d0), 64):
        t = t2(st, d0[k:k + 64])
        out[k:k + 64] = (t >= t[:, [i0]]).mean(axis=1)
    return out


def ri_set(st: np.ndarray, i0: int, b0: float, se0: float) -> dict:
    """{d0 : p(d0) > ALPHA}: the grid b0 +- h, h doubled from 8 se until both ends reject and then
    once more, N_GRID points; each change of verdict between neighbouring points refined by 60
    bisections. An end of a piece that reaches the grid's end is None (no rejection found there)."""
    h = 8.0 * se0
    for _ in range(12):
        if (ri_p(st, i0, [b0 - h, b0 + h]) <= ALPHA).all():
            break
        h *= 2.0
    h *= 2.0
    grid = np.linspace(b0 - h, b0 + h, N_GRID)
    acc = ri_p(st, i0, grid) > ALPHA

    def edge(rej, acp):
        for _ in range(60):
            m = 0.5 * (rej + acp)
            rej, acp = (rej, m) if ri_p(st, i0, m)[0] > ALPHA else (m, acp)
        return 0.5 * (rej + acp)

    pieces, i = [], 0
    while i < N_GRID:
        if not acc[i]:
            i += 1
            continue
        k = i
        while k + 1 < N_GRID and acc[k + 1]:
            k += 1
        pieces.append([None if i == 0 else edge(grid[i - 1], grid[i]),
                       None if k == N_GRID - 1 else edge(grid[k + 1], grid[k])])
        i = k + 1
    return {"pieces": pieces, "grid": [float(grid[0]), float(grid[-1])], "grid_points": N_GRID,
            "grid_step": float(grid[1] - grid[0])}


def label_r2(sets: np.ndarray, n_cand: int, y: np.ndarray, col=None) -> float:
    """R^2 of y across sets on the membership indicators (all of them, or the one of column col)."""
    Mb = np.zeros((len(sets), n_cand))
    Mb[np.arange(len(sets))[:, None], sets] = 1.0
    X = np.column_stack([np.ones(len(sets)), Mb if col is None else Mb[:, col]])
    r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1.0 - (r @ r) / ((y - y.mean()) @ (y - y.mean())))


def ri_pool(rl: Relabel, sets: np.ndarray, i0: int, exact: bool, members: list) -> tuple[dict, np.ndarray]:
    """RI over the rows of `sets` (real label at row i0): p at zero by |t| and by |delta-hat|, where
    the label sits in the design (percentile of effective clusters and of squared mass), and the
    statistics' share explained by membership. -> (block, statistics)."""
    st = rl.stats(sets)
    tz = st[:, 0] / np.sqrt(rl.adj * st[:, 3])
    bz = st[:, 0] / st[:, 2]
    p_t = float(np.mean(np.abs(tz) >= abs(tz[i0])))
    p_b = float(np.mean(np.abs(bz) >= abs(bz[i0])))
    k = sets.shape[1]
    expl = {"t_all": label_r2(sets, len(rl.cand), tz), "beta_all": label_r2(sets, len(rl.cand), bz)}
    if CAIXA in rl.cand:
        cx = rl.cand.index(CAIXA)
        expl.update(t_caixa=label_r2(sets, len(rl.cand), tz, cx), beta_caixa=label_r2(sets, len(rl.cand), bz, cx))
    blk = {"members": members, "pool_size": len(members), "set_size": k, "N": int(len(sets)),
           "enumeration": f"all {k}-sets" if exact else f"{len(sets) - 1:,} random {k}-sets and the label",
           "t_label": float(tz[i0]), "p_t": p_t, "p_beta": p_b,
           "diverge": bool((p_t <= ALPHA) != (p_b <= ALPHA)),
           "typicality": {"pct_effective_clusters": float(np.mean(st[:, 6] <= st[i0, 6])),
                          "pct_squared_mass": float(np.mean(st[:, 2] <= st[i0, 2])),
                          "label_effective_clusters": float(st[i0, 6]),
                          "median_effective_clusters": float(np.median(st[:, 6])),
                          "label_squared_mass": float(st[i0, 2]),
                          "median_squared_mass": float(np.median(st[:, 2]))},
           "share_explained_by_members": expl}
    return blk, st


def variant(ss: pd.DataFrame, xcol: str, label: list, pool=None) -> dict:
    """The difference regression on a variant sample or label: x and x 1[j in label] with rho and
    both sets of effects. CR1 (P4's Fit; p from t(G - 1)); the wild cluster restricted bootstrap p
    at zero (P4's Fit, its draws from a fresh default_rng(SEED)); CR2 with its df; with a pool, the
    RI over every len(label)-set of it (|t| headline, |delta-hat| beside). label minus rest and rest
    minus label."""
    s2 = ss.assign(x_lab=ss[xcol] * ss["j"].isin(label))
    f = Fit(design(s2, [xcol, "x_lab", "rho_lag_pp"]), s2["ln_o"].to_numpy(float), s2["j"].to_numpy(),
            np.random.default_rng(SEED))
    b, se = float(f.b[1]), f.cr1(1)
    L = np.zeros((f.X.shape[1], 1))
    L[1] = 1.0
    c2 = cr2(f.X, f.y, s2["j"].to_numpy(), L)[0]
    out = {"n": f.n, "G": f.G, "G_label": int(s2.loc[s2["j"].isin(label), "j"].nunique()),
           "label_minus_rest": b, "rest_minus_label": -b, "se_cr1": se,
           "p_cr1": float(2 * stats.t.sf(abs(b / se), f.G - 1)), "wcr_p": f.wcr_p(1),
           "cr2": {k: c2[k] for k in ("se", "df", "p")}}
    if pool is not None:
        rl = Relabel(s2, xcol, pool, label)
        st0 = rl.stats(rl.s0[None, :])[0]
        if abs(st0[0] / st0[2] - b) > TOL_FIT or abs(np.sqrt(rl.adj * st0[3]) / st0[2] - se) > TOL_FIT:
            raise SystemExit(f"variant {label}: the RI statistic of the label is not P4's Fit")
        sets = np.array(list(combinations(range(len(pool)), len(label))))
        i0 = int(np.flatnonzero((sets == rl.s0[None, :]).all(axis=1))[0])
        out["ri"], _ = ri_pool(rl, sets, i0, True, list(pool))
    return out


# ----------------------------------------------------------------------------------------------- CR2
def cr2(X: np.ndarray, y: np.ndarray, g: np.ndarray, L: np.ndarray, adjust: bool = True) -> list:
    """CR2 with Bell-McCaffrey degrees of freedom for each column l of L (k x m), OLS with the
    working model Omega = I. A_g = (I - H_gg)^{+1/2}, the symmetric square root of the Moore-Penrose
    inverse (eigenvalues <= CR2_TOL set to zero), so the effects can be explicit columns of X
    (Pustejovsky and Tipton 2018); with adjust=False A_g = I (CR0). With X = QR and
    w_g = A_g Q_g R^-T l: V = sum_g (w_g'e_g)^2 and df = (tr Gamma)^2 / tr(Gamma^2), Gamma_gh =
    w_g'(I - H)_gh w_h = 1[g = h] w_g'w_g - (Q_g'w_g)'(Q_h'w_h). -> per column: estimate, se, t, df, p."""
    n, k = X.shape
    if np.linalg.matrix_rank(X) < k:
        raise SystemExit("CR2: the design is rank deficient")
    Q, R = np.linalg.qr(X)
    Ri = np.linalg.inv(R)
    b = Ri @ (Q.T @ y)
    e = y - X @ b
    Lt = Ri.T @ L
    _, inv = np.unique(g, return_inverse=True)
    G, m = int(inv.max()) + 1, L.shape[1]
    sc, ww, qq = np.zeros((G, m)), np.zeros((G, m)), np.zeros((G, k, m))
    for c in range(G):
        r = np.flatnonzero(inv == c)
        Qg = Q[r]
        if adjust:
            lam, U = np.linalg.eigh(np.eye(len(r)) - Qg @ Qg.T)
            keep = lam > CR2_TOL
            wg = (U[:, keep] / np.sqrt(lam[keep])) @ (U[:, keep].T @ (Qg @ Lt))
        else:
            wg = Qg @ Lt
        sc[c], ww[c], qq[c] = wg.T @ e[r], (wg * wg).sum(0), Qg.T @ wg
    out = []
    for i in range(m):
        gam = np.diag(ww[:, i]) - qq[:, :, i] @ qq[:, :, i].T
        df = float(np.trace(gam) ** 2 / (gam * gam).sum())
        est, se = float(L[:, i] @ b), float(np.sqrt((sc[:, i] ** 2).sum()))
        out.append({"estimate": est, "se": se, "t": est / se, "df": df,
                    "p": float(2 * stats.t.sf(abs(est / se), df))})
    return out


def bm_df_bruteforce(X, g, l) -> float:
    """The Bell-McCaffrey df from the n x n matrices: gamma_g = (I - H)[:, g] A_g X_g (X'X)^-1 l, with
    H = X X^+ and (X'X)^-1 = X^+ X^+' from the SVD pseudo-inverse X^+."""
    n = len(X)
    Xp = np.linalg.pinv(X)
    IH = np.eye(n) - X @ Xp
    m = Xp @ (Xp.T @ l)
    cols = []
    for c in np.unique(g):
        r = np.flatnonzero(g == c)
        lam, U = np.linalg.eigh(IH[np.ix_(r, r)])
        keep = lam > CR2_TOL
        A = (U[:, keep] / np.sqrt(lam[keep])) @ U[:, keep].T
        cols.append(IH[:, r] @ (A @ (X[r] @ m)))
    P = np.column_stack(cols)
    gam = P.T @ P
    return float(np.trace(gam) ** 2 / (gam * gam).sum())


def within(M: np.ndarray, g: np.ndarray) -> np.ndarray:
    """M demeaned within the groups g (the absorbed cluster effects)."""
    _, inv = np.unique(g, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    out = np.array(M, float, copy=True)
    for j in range(out.shape[1]):
        out[:, j] -= (np.bincount(inv, weights=out[:, j]) / cnt)[inv]
    return out


def absorbed_design(s: pd.DataFrame, regs: list) -> np.ndarray:
    """regs and the quarter dummies, demeaned within conglomerate: the explicit design less its
    constant and conglomerate dummies, built independently of design()."""
    fe_q = pd.get_dummies(s["qi"], prefix="q", drop_first=True, dtype=float)
    return within(np.column_stack([s[regs].to_numpy(float), fe_q.to_numpy()]), s["j"].to_numpy())


def rel(a, b) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def cr2_unit_tests(real: dict) -> dict:
    """G6. `real`: {margin: (X explicit, X absorbed, y, g, L, P4 Fit of the difference, P4 Fit of the
    five largest's level)} of the data; L's columns: the difference, the five largest's level, the
    rest's level."""
    rng = np.random.default_rng(7)
    res = {}
    # (a) intercept, G equal clusters: CR2 = G / (G - 1) CR0 and df = G - 1.
    G, m = 6, 4
    g = np.repeat(np.arange(G), m)
    y = rng.normal(size=G * m)
    r = cr2(np.ones((G * m, 1)), y, g, np.eye(1))[0]
    e = y - y.mean()
    v0 = G / (G - 1) * (np.bincount(g, weights=e) ** 2).sum() / (G * m) ** 2
    res["intercept_equal_clusters"] = {"var_rel_err": rel(r["se"] ** 2, v0), "df": r["df"], "df_expected": G - 1,
                                       "pass": rel(r["se"] ** 2, v0) < TOL_CR2_TEST and abs(r["df"] - (G - 1)) < 1e-8}
    # (b) singletons, three regressors: CR2 = HC2.
    n = 40
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(size=n)])
    y = X @ np.array([1.0, 0.5, -0.3]) + rng.normal(size=n) * (1 + X[:, 2])
    l = np.array([0.0, 1.0, 0.0])
    r = cr2(X, y, np.arange(n), l[:, None])[0]
    XtXi = np.linalg.inv(X.T @ X)
    e = y - X @ (XtXi @ X.T @ y)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    hc2 = float((((X @ XtXi @ l) ** 2) * e ** 2 / (1 - h)).sum())
    dfb = bm_df_bruteforce(X, np.arange(n), l)
    res["singletons_hc2"] = {"var_rel_err": rel(r["se"] ** 2, hc2), "df": r["df"], "df_bruteforce": dfb,
                             "pass": rel(r["se"] ** 2, hc2) < TOL_CR2_TEST and rel(r["df"], dfb) < TOL_CR2_TEST}
    # (c) a two-group dummy, singletons: the Welch variance and its Satterthwaite df at equal variances.
    n1, n0 = 7, 12
    dd = np.r_[np.ones(n1), np.zeros(n0)]
    y = rng.normal(size=n1 + n0) * np.where(dd == 1, 2.0, 0.5) + dd
    r = cr2(np.column_stack([np.ones(n1 + n0), dd]), y, np.arange(n1 + n0), np.array([[0.0], [1.0]]))[0]
    welch = y[dd == 1].var(ddof=1) / n1 + y[dd == 0].var(ddof=1) / n0
    dfw = (1 / n1 + 1 / n0) ** 2 / (1 / (n1 ** 2 * (n1 - 1)) + 1 / (n0 ** 2 * (n0 - 1)))
    res["two_group_welch"] = {"var_rel_err": rel(r["se"] ** 2, welch), "df": r["df"], "df_expected": dfw,
                              "pass": rel(r["se"] ** 2, welch) < TOL_CR2_TEST and rel(r["df"], dfw) < TOL_CR2_TEST}
    # (d) a simulated unbalanced panel: explicit against absorbed cluster effects, and the df against
    # the brute force.
    Gp, T = 9, 7
    gg, tt = np.repeat(np.arange(Gp), T), np.tile(np.arange(T), Gp)
    keep = rng.uniform(size=Gp * T) > 0.15
    gg, tt = gg[keep], tt[keep]
    x1 = rng.normal(size=len(gg)) + 0.3 * gg
    x2 = x1 * (gg < 3)
    y = 0.4 * x1 - 0.2 * x2 + rng.normal(size=len(gg)) * (1 + (gg == 0)) + 0.1 * gg
    fe_g = (gg[:, None] == np.arange(1, Gp)[None, :]).astype(float)
    fe_t = (tt[:, None] == np.arange(1, T)[None, :]).astype(float)
    Xe = np.column_stack([x1, x2, np.ones(len(gg)), fe_g, fe_t])
    Xa = within(np.column_stack([x1, x2, fe_t]), gg)
    Le = np.zeros((Xe.shape[1], 2))
    Le[1, 0], Le[0, 1], Le[1, 1] = 1.0, 1.0, 1.0
    La = np.zeros((Xa.shape[1], 2))
    La[1, 0], La[0, 1], La[1, 1] = 1.0, 1.0, 1.0
    re, ra = cr2(Xe, y, gg, Le), cr2(Xa, y, gg, La)
    dfb = bm_df_bruteforce(Xe, gg, Le[:, 0])
    err = max(max(rel(a["se"], b["se"]), rel(a["df"], b["df"]), rel(a["estimate"], b["estimate"]))
              for a, b in zip(re, ra))
    res["panel_explicit_vs_absorbed"] = {"max_rel_err": err, "df_bruteforce_rel_err": rel(re[0]["df"], dfb),
                                         "df": [x["df"] for x in re],
                                         "pass": err < TOL_CR2_TEST and rel(re[0]["df"], dfb) < TOL_CR2_TEST}
    # (e) the data, each margin: explicit against absorbed effects; the five largest's level from the
    # difference design (x + x * big5) against P4's level design (x alone, x * rest beside it); A_g = I
    # with P4's factor reproduces P4's CR1 of the difference.
    for name, (Xe, Xa, y, g, L, f, fl) in real.items():
        La = np.zeros((Xa.shape[1], L.shape[1]))
        La[:3] = L[:3]                       # both designs start with the three regressors
        re, ra = cr2(Xe, y, g, L), cr2(Xa, y, g, La)
        err = max(max(rel(a["se"], b["se"]), rel(a["df"], b["df"]), rel(a["estimate"], b["estimate"]))
                  for a, b in zip(re, ra))
        e0 = np.zeros((fl.X.shape[1], 1))
        e0[0] = 1.0
        lv = cr2(fl.X, fl.y, g, e0)[0]
        e_lv = max(rel(lv["se"], re[1]["se"]), rel(lv["df"], re[1]["df"]), rel(lv["estimate"], re[1]["estimate"]))
        c0 = cr2(Xe, y, g, L[:, :1], adjust=False)[0]
        e_cr1 = rel(np.sqrt(f.adj) * c0["se"], f.cr1(1))
        e_df = max(rel(re[i]["df"], bm_df_bruteforce(Xe, g, L[:, i])) for i in range(L.shape[1]))
        res[f"data_{name}"] = {"explicit_vs_absorbed_max_rel_err": err, "level_two_designs_rel_err": e_lv,
                               "cr0_times_adj_vs_p4_cr1_rel_err": e_cr1, "df_bruteforce_max_rel_err": e_df,
                               "pass": max(err, e_lv, e_cr1, e_df) < TOL_CR2_TEST}
    return res


# ------------------------------------------------------------------------------------------- main
def main():
    stamp("diag_p4_adv_group_diff: reading P4's panel file and JSON, P2's coverage")
    J = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    cov = pd.read_csv(COVERAGE, dtype={"CodConglomeradoPrudencial": str})
    rank = cov.sort_values("rank_2019")["CodConglomeradoPrudencial"].tolist()
    names = cov.set_index("CodConglomeradoPrudencial")["name"].to_dict()
    d = load()
    inputs = {k: {"file": p.name, "md5": md5(p), "lf_md5": lf_md5(p)}
              for k, p in (("p4_csv", OUT_CSV), ("p4_json", OUT_JSON), ("p2_coverage", COVERAGE))}
    gates = {}

    # G1 the panel.
    csv_md5 = sorted(d["panel_md5"].dropna().astype(str).unique())
    gates["G1_panel"] = {"csv": csv_md5, "json": J["_meta"]["panel_md5"],
                         "pass": csv_md5 == [J["_meta"]["panel_md5"][:8]]}

    # The replay (P4's draws) and the samples. G2 sizes, the label, the pool.
    fits, sizes = replay(d)
    label = rank[:5]
    s = {mg: fits[MARGINS[mg]["diff_fit"]][1] for mg in MARGINS}
    per = s["pooled"].groupby("j")["x"].agg(n="size", zero=lambda v: int((v == 0).sum()), nd="nunique")
    pool = sorted(per.index[(per["zero"] == 0) & (per["nd"] >= 2) & (per["n"] >= POOL_MIN_QUARTERS)])
    n_int = s["intensive"].groupby("j").size()
    gsplit = {mg: (int(ss.loc[ss["big5"] == 1, "j"].nunique()), int(ss.loc[ss["big5"] == 0, "j"].nunique()))
              for mg, ss in s.items()}
    bad = [f"{mg}: the big5 flags are not P2's rank_2019 top five" for mg, ss in s.items()
           if sorted(ss.loc[ss["big5"] == 1, "j"].unique()) != sorted(label)]
    for name, blk in SIZE_CHECKS.items():
        want = (J[blk]["n_obs"], J[blk]["n_conglomerates"])
        if tuple(sizes[name]) != want:
            bad.append(f"{name}: {sizes[name]} vs P4 {blk} {want}")
    if sizes["i1"][0] != J["intensive_B"]["n_obs"]:
        bad.append(f"i1: n {sizes['i1'][0]} vs P4 intensive_B {J['intensive_B']['n_obs']}")
    if gsplit["pooled"] != (J["big5_alone"]["n_conglomerates"], J["rest_alone"]["n_conglomerates"]):
        bad.append(f"pooled split {gsplit['pooled']} vs big5_alone/rest_alone")
    if not set(label) <= set(pool):
        bad.append("the five largest are not all in the pool")
    if any(n_int.get(c, 0) != per.loc[c, "n"] for c in pool):
        bad.append("a pool member's quarters differ on the intensive margin")
    gates["G2_sizes"] = {"sizes": {k: list(v) for k, v in sizes.items()}, "label": label, "split": gsplit,
                         "pool": pool, "problems": bad, "pass": not bad}

    # G3 the RI's statistic for the real label is P4's.
    rl, g3 = {}, {}
    for mg, spec in MARGINS.items():
        xcol, ss = spec["x"], s[mg]
        varying = sorted(ss.groupby("j")[xcol].nunique().pipe(lambda v: v.index[v >= 2]))
        rl[mg] = Relabel(ss, xcol, varying, label)
        st0 = rl[mg].stats(rl[mg].s0[None, :])[0]
        b_fwl, se_fwl = st0[0] / st0[2], np.sqrt(rl[mg].adj * st0[3]) / st0[2]
        f = fits[spec["diff_fit"]][0]
        jb = J[spec["block"]]["adv_big5_minus_rest"]
        # The closed form for other five-sets and shifted nulls against P4's Fit on the relabelled
        # design and the outcome y - d0 x 1[j in S0].
        rs, err_t = np.random.default_rng(SEED_CHECK), 0.0
        x0 = ss[xcol].to_numpy(float) * ss["big5"].to_numpy(float)
        for _ in range(N_RELABEL_CHECK):
            S = np.sort(rs.choice(rl[mg].positions(pool), 5, replace=False))
            sm = ss.assign(x_S=ss[xcol] * ss["j"].isin([rl[mg].cand[i] for i in S]))
            fS = Fit(design(sm, [xcol, "x_S", "rho_lag_pp"]), f.y, sm["j"].to_numpy(), np.random.default_rng(0))
            stS = rl[mg].stats(S[None, :])
            for d0 in (0.0, b_fwl - 3 * se_fwl, b_fwl + 3 * se_fwl):
                fd = fS.refit(f.y - d0 * x0)
                t_cf = np.sign(stS[0, 0] - d0 * stS[0, 1]) * np.sqrt(t2(stS, d0)[0, 0] / rl[mg].adj)
                err_t = max(err_t, abs(fd.b[1] / fd.cr1(1) - t_cf))
        g3[mg] = {"b_fwl": b_fwl, "se_fwl": se_fwl, "b_fit": float(f.b[1]), "se_fit": f.cr1(1),
                  "k_fit": f.k, "k_fwl": rl[mg].k, "json": [jb["b"], jb["se"]],
                  "relabel_sets_checked": N_RELABEL_CHECK, "relabel_max_abs_t_err": err_t,
                  "pass": (abs(b_fwl - f.b[1]) < TOL_FIT and abs(se_fwl - f.cr1(1)) < TOL_FIT and f.k == rl[mg].k
                           and [round(b_fwl, 4), round(se_fwl, 4)] == [jb["b"], jb["se"]] and err_t < TOL_T)}
    gates["G3_estimate"] = {**g3, "pass": all(v["pass"] for v in g3.values())}
    stamp(f"G1 {gates['G1_panel']['pass']}, G2 {gates['G2_sizes']['pass']}, G3 {gates['G3_estimate']['pass']} "
          f"(pooled b {g3['pooled']['b_fwl']:.6f} se {g3['pooled']['se_fwl']:.6f}; "
          f"FWL - Fit {g3['pooled']['b_fwl'] - g3['pooled']['b_fit']:.1e}; relabelled t, max error "
          f"{max(v['relabel_max_abs_t_err'] for v in g3.values()):.1e})")

    # G4 the replayed bootstrap reproduces P4's.
    wcr, g4 = {}, []
    for fit, i, blk, entry, with_ci in REPLAY_CHECKS:
        f = fits[fit][0]
        jb = J[blk][entry]
        p = f.wcr_p(i)
        row = {"fit": fit, "coef": i, "p4": f"{blk}.{entry}", "b": float(f.b[i]), "se": f.cr1(i), "wcr_p": p,
               "json_b": jb["b"], "json_se": jb["se"], "json_p": jb.get("wcr_p0")}
        ok = [round(float(f.b[i]), 4), round(f.cr1(i), 4), round(p, 3)] == [jb["b"], jb["se"], jb.get("wcr_p0")]
        if with_ci:
            row["wcr_ci"], row["json_ci"] = f.wcr_ci(i), jb["ci95_wcr"]
            ok = ok and row["wcr_ci"] == jb["ci95_wcr"]
        row["pass"] = ok
        g4.append(row)
        wcr[(fit, i)] = row
        stamp(f"G4 {fit}[{i}] {blk}.{entry}: p {p:.4f} (P4 {jb.get('wcr_p0')})"
              + (f", ci {row['wcr_ci']} (P4 {jb['ci95_wcr']})" if with_ci else "") + f" -> {ok}")
    gates["G4_wcr_replay"] = {"checks": g4, "pass": all(r["pass"] for r in g4)}

    # G5 the primary pool's enumeration.
    sets, i0 = {}, {}
    for mg in MARGINS:
        P = rl[mg].positions(pool)
        sets[mg] = np.array(list(combinations(P, 5)))
        hit = np.flatnonzero((sets[mg] == rl[mg].s0[None, :]).all(axis=1))
        i0[mg] = int(hit[0]) if len(hit) == 1 else -1
    gates["G5_enumeration"] = {mg: {"N": len(sets[mg]), "C": comb(len(pool), 5),
                                    "unique": len({tuple(r) for r in sets[mg]}), "label_row": i0[mg]}
                               for mg in MARGINS}
    gates["G5_enumeration"]["pass"] = all(
        v["N"] == v["C"] == v["unique"] and v["label_row"] >= 0 for v in gates["G5_enumeration"].values())

    # G6 CR2 unit tests.
    real, cr2_in = {}, {}
    for mg, spec in MARGINS.items():
        f, ss, regs = fits[spec["diff_fit"]]
        L = np.zeros((f.X.shape[1], 3))
        L[1, 0], L[0, 1], L[1, 1], L[0, 2] = 1.0, 1.0, 1.0, 1.0       # delta, five largest, rest
        cr2_in[mg] = (f.X, f.y, ss["j"].to_numpy(), L)
        real[mg] = (f.X, absorbed_design(ss, regs), f.y, ss["j"].to_numpy(), L, f, fits[spec["level_fit"]][0])
    g6 = cr2_unit_tests(real)
    gates["G6_cr2_tests"] = {**g6, "pass": all(v["pass"] for v in g6.values())}
    stamp("G4 " + str(gates["G4_wcr_replay"]["pass"]) + ", G5 " + str(gates["G5_enumeration"]["pass"])
          + ", G6 " + str(gates["G6_cr2_tests"]["pass"]) + " " + json.dumps(
              clean({k: v["pass"] for k, v in g6.items()})))

    passed = all(v["pass"] for v in gates.values())
    meta = {"written": datetime.datetime.now().isoformat(timespec="seconds"),
            "script": "diagnostics/diag_p4_adv_group_diff.py", "script_lf_md5": lf_md5(Path(__file__)),
            "inputs": inputs, "panel_md5": J["_meta"]["panel_md5"], "p4_written": J["_meta"]["written"],
            "sign": SIGN, "pool_rule": POOL_RULE, "alpha": ALPHA,
            "seeds": {"p4_wcr": SEED, "ri_random_pool": SEED_RI}, "n_boot": N_BOOT,
            "cr2": "Moore-Penrose (I - H_gg)^{+1/2}, eigenvalues <= 1e-12 zero; Bell-McCaffrey df, Omega = I",
            "gates_passed": passed}
    if not passed:
        dump({"_meta": meta, "gates": gates})
        print("MISMATCH: a gate failed: " + ", ".join(k for k, v in gates.items() if not v["pass"])
              + f"; {OUT.name} holds the gates and no results")
        sys.exit(1)

    # ------------------------------------------------------------------------------ results
    out = {"_meta": meta, "gates": gates}
    keys = {}
    always = sorted(per.index[(per["zero"] == 0) & (per["nd"] >= 2)])
    top15 = [c for c in rank if c in set(per.index) and per.loc[c, "nd"] >= 2][:15]
    for mg, spec in MARGINS.items():
        r, ss = rl[mg], s[mg]
        f = fits[spec["diff_fit"]][0]
        fl = fits[spec["level_fit"]][0]
        G, n = f.G, f.n
        b, se = float(f.b[1]), f.cr1(1)
        dfree = G - 1

        def cr1_row(est, sd):
            return {"b": est, "se": sd, "t": est / sd, "p_t_G_minus_1": float(2 * stats.t.sf(abs(est / sd), dfree)),
                    "ci95_cr1": [est - 1.96 * sd, est + 1.96 * sd]}
        # WCR: the replayed p, the interval (replayed on g1, computed here on i1).
        wd = wcr[(spec["diff_fit"], 1)]
        ci = wd["wcr_ci"] if "wcr_ci" in wd else f.wcr_ci(1)
        wl = wcr[(spec["level_fit"], 0)]
        # RI, primary pool, its set; secondary pools.
        prim, st = ri_pool(r, sets[mg], i0[mg], True, pool)
        prim["set95"] = ri_set(st, i0[mg], b, se)
        prim["set95_rest_minus_big5"] = [flip(pc) for pc in prim["set95"]["pieces"]]
        sec = {}
        P_always = r.positions(always)
        P_top = r.positions([c for c in top15 if c in r.cand])     # with x varying on this margin
        for nm, P in (("always_advertisers", P_always), ("top15_2019_deposits", P_top)):
            if not set(r.s0) <= set(P):
                continue
            ss_ = np.array(list(combinations(P, 5)))
            k0 = int(np.flatnonzero((ss_ == r.s0[None, :]).all(axis=1))[0])
            sec[nm], _ = ri_pool(r, ss_, k0, True, [r.cand[i] for i in P])
        rng = np.random.default_rng(SEED_RI)
        Pall = np.arange(len(r.cand))
        rnd = np.sort(np.argsort(rng.random((N_RANDOM, len(Pall))), axis=1)[:, :5], axis=1)
        sec["all_with_variation"], _ = ri_pool(r, np.vstack([r.s0[None, :], Pall[rnd]]), 0, False, r.cand)
        sec["all_with_variation"]["exchangeable"] = False
        # CR2.
        c2 = cr2(*cr2_in[mg])
        c2 = dict(zip(("diff", "big5_level", "rest_level"), c2))
        # Diagnostics: leverage, leave one of the five out, big four labelled.
        sh = r.shares()
        order = np.argsort(-sh)
        lev = {"by_cluster": {r.codes[i]: float(sh[i]) for i in order},
               "five_largest": {c: float(sh[list(r.codes).index(c)]) for c in label},
               "five_largest_total": float(sum(sh[list(r.codes).index(c)] for c in label)),
               "G_eff": float(1.0 / (sh ** 2).sum()), "caixa_share": float(sh[list(r.codes).index(CAIXA)])}
        xc = spec["x"]
        big4_codes = [c for c in label if c != CAIXA]
        loo = []
        for c in [CAIXA] + big4_codes:       # each of the five out of the sample; the others keep the label
            loo.append({"dropped": c, "name": names.get(c),
                        **variant(ss[ss["j"] != c].reset_index(drop=True), xc, [x for x in label if x != c],
                                  [p for p in pool if p != c] if c == CAIXA else None)})
        big4 = {"note": "the big four labelled, Caixa in the sample among the rest",
                **variant(ss, xc, big4_codes, pool)}
        for tag, v in (("without Caixa", loo[0]), ("big four, Caixa in the rest", big4)):
            stamp(f"{mg} {tag}: rest - label {v['rest_minus_label']:.4f} (CR1 se {v['se_cr1']:.4f}, p "
                  f"{v['p_cr1']:.3f}; WCR p {v['wcr_p']:.3f}; RI-t p {v['ri']['p_t']:.3f} over "
                  f"{v['ri']['N']:,} four-sets; CR2 p {v['cr2']['p']:.3f} df {v['cr2']['df']:.2f})")
        out[mg] = {
            "spec": f"ln o on {xc}, {xc}_b5, rho_lag_pp, conglomerate and quarter effects (P4 {spec['diff_fit']})",
            "sample": {"n": n, "G": G, "G_big5": gsplit[mg][0], "G_rest": gsplit[mg][1],
                       "quarters": int(ss["qi"].nunique())},
            "big5_minus_rest": {**cr1_row(b, se), "wcr_p": wd["wcr_p"], "wcr_ci95": ci},
            "rest_minus_big5": {"b": -b, "ci95_cr1": [-b - 1.96 * se, -b + 1.96 * se], "wcr_ci95": flip(ci),
                                "ri_set95": prim["set95_rest_minus_big5"]},
            "levels": {"big5": {**cr1_row(float(fl.b[0]), fl.cr1(0)), "wcr_p": wl["wcr_p"],
                                **({"wcr_ci95": wl["wcr_ci"]} if "wcr_ci" in wl else {}), "cr2": c2["big5_level"]},
                       "rest": {**cr1_row(float(f.b[0]), f.cr1(0)), "wcr_p": wcr[(spec["diff_fit"], 0)]["wcr_p"],
                                "cr2": c2["rest_level"]}},
            "ri": {"primary": prim, "secondary": sec},
            "cr2": c2,
            "diagnostics": {"leverage": lev, "leave_one_out": loo, "big_four_labelled": big4,
                            "sign": "label_minus_rest as delta; rest_minus_label = -delta"},
        }
        pcs = prim["set95_rest_minus_big5"]
        keys["diff" if mg == "pooled" else "diff_int"] = {
            "elast": -b, "p": wd["wcr_p"],
            **({"ci": flip(ci)} if None not in ci else {}),
            "G": {"big5": gsplit[mg][0], "rest": gsplit[mg][1]},
            "p_ri": prim["p_t"],
            **({"ri": pcs[0]} if len(pcs) == 1 and None not in pcs[0] else {}),
            "ri_n": prim["N"], "ri_pool": len(pool),
            "p_cr2": c2["diff"]["p"], "df_cr2": c2["diff"]["df"],
            "G_eff": lev["G_eff"], "caixa_share": lev["caixa_share"]}
        stamp(f"{mg}: rest - big5 {-b:.4f} (CR1 se {se:.4f}, p t(G-1) {cr1_row(b, se)['p_t_G_minus_1']:.3f}); "
              f"WCR p {wd['wcr_p']:.3f} ci {flip(ci)}; RI-t p {prim['p_t']:.3f} (RI-beta {prim['p_beta']:.3f}) "
              f"set {pcs}; CR2 p {c2['diff']['p']:.3f} df {c2['diff']['df']:.2f}; "
              f"G_eff {lev['G_eff']:.2f}, Caixa {lev['caixa_share']:.3f}")
        if prim["diverge"]:
            stamp(f"{mg}: DIVERGENCE: RI-t and RI-beta cross {ALPHA} differently; read the RI-t row")
    keys["big5_elast_p_cr2"] = out["pooled"]["levels"]["big5"]["cr2"]["p"]
    out["keys"] = keys
    dump(out)
    stamp(f"wrote {OUT}")
    for k, v in keys.items():
        stamp(f"keys {k}: {json.dumps(clean(v))}")


if __name__ == "__main__":
    main()
