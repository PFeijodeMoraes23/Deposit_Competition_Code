"""
diag_w1_followups.py
====================
Read-only follow-ups to W1, the main thread's two requests of 11-12 17:13 (claude/MAIN_IMPR_CHANNEL.md,
A5): where the sign of the price coefficient comes from, by deposit type.

Inputs (read only):
  - the production I2 frames of routines (III) and (IV), spec 12, in paths.demand_parquet_dir();
  - the logit's mean utility BLP_RESULTS/logit/logit_delta_E{k}_spec_12.bin, written by blp_logit.jl
    (berry_inversion: ln s_j - ln s_0 in the engine's market structure), and logit_summary_spec_12.json;
  - ext1's checkpoint BLP_RESULTS/cluster_raw/blp_checkpoint_E{k}_spec_12_ext1.jls (the engines'
    Dict("theta2_star", "delta_star", ...): delta_star is the contraction at theta_2*), its result
    blp_results_E{k}_spec_12_ext1.json and blp_summary_E{k}_ext1_gpu_ift.json. The .jls is read as
    bytes: the key, the Float64 array tag and the length are matched literally, and gate G2 checks the
    vector by reproducing ext1's theta_1.

The linear step (blp_engine_cpu.jl estimate_theta1, the one every engine and blp_logit.jl use): the
spread of types 4 and 5 is projected within type on H = [X1, Z] (the 16 instruments); types 1-2 keep
their own spread; theta_1 = OLS of delta on X_hat = [spread_hat, X1], X1 = X_COLS then the D dummy, no
constant; xi = delta - [spread, X1] theta_1. Standard errors: compute_cluster_se (CR1, clusters =
CodConglomeradoPrudencial) and the score wild cluster bootstrap of blp_se_common.jl wcb_se (Webb,
B = 999; numpy draws, so it differs from the stored Julia draws by Monte Carlo error only). For ext1
both are conditional on theta_2* (delta_star taken as data): ext1's joint standard error needs the
theta_2 Jacobian (GPU IFT) and is not formed here.

Gates (a failure stops the script before anything is written):
  G1  the logit: pooled alpha = the stored logit_summary value to 1e-7 (E3, E4); W1's types 1-2 OLS and
      types 4-5 IV alphas (scratchpad w1_1112/w1_logit_iv.json) reproduced to 1e-8; the effective F of
      the type-4 first stage = W1's to 1e-8.
  G2  ext1: the checkpoint parses (header, one match of each key, a Float64 vector of the frame's
      length ending at the end of the file, theta2_star = the result's theta2), and the linear step on
      delta_star reproduces alpha of the summary to 1e-9, the other nine theta_1 of the result to 2e-7
      (relative; the result rounds to 8 significant digits) and theta1_se_linear_iv to 2e-6.

Request 2 (logit, the spread exogenous): OLS of the logit delta on [spread, X1] on all types, types 1-2,
  types 4-5, type 4 and type 5; and on types 4-5 with quarter fixed effects (within quarter, outside the
  production design).
Request 1 (ext1): the linear step on ext1's delta on types 1-2 (the spread is its own instrument there),
  types 4-5, type 4 and type 5 (each with the engine's within-type projection), the pooled
  decomposition, and the Montiel Olea-Pflueger effective F of each subset's first stage. The first stage
  does not involve delta, so it is the logit's. Supplementary: OLS on ext1's delta (spread exogenous);
  the ext1 fits without the rows where ext1's delta is more than 10 from the logit's (a sensitivity: it
  selects on the outcome); and Anderson-Rubin sets for alpha on types 4-5 with the logit's and ext1's delta.

Writes OUT = paths.rout_dir()/diag_w1_followups/: w1_followups.json and MD5SUMS (LF md5 of each output).
Usage: python -I diagnostics/diag_w1_followups.py [--E 3,4]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "diagnostics"))

import argparse
import hashlib
import json
import struct
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from utils import paths
import diag_cap_sensitivity as D

T0 = time.time()
BLP = paths.PROCESSED / "ESTIMATION_OUTPUT" / "BLP_RESULTS"
LOGIT = BLP / "logit"
RAW = BLP / "cluster_raw"
OUT = paths.rout_dir() / "diag_w1_followups"
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
B, SEED = 999, 20261112          # W1's draws (w1_logit_iv.py), so its gates reproduce exactly
GRID = np.round(np.arange(-3.0, 3.0 + 1e-12, 0.001), 6)
# W1 (scratchpad w1_1112/w1_logit_iv.json, 11-12 15:50): alpha by OLS on types 1-2, by the logit's IV on
# types 4-5, and the type-4 MOP effective F.
W1 = {3: {"types_12_ols": -0.11847687349251182, "types_45_iv": 0.8928743233866905, "feff_type4": 1.3631370761021666},
      4: {"types_12_ols": -0.12161889059835618, "types_45_iv": 0.9292075967458922, "feff_type4": 1.3509163319337993}}


def say(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def md5_lf(p):
    """md5 of the file with CR removed (the cluster convention for text); binary files as they are."""
    b = Path(p).read_bytes()
    if Path(p).suffix in (".json", ".py", ".csv", ".md", ".txt"):
        b = b.replace(b"\r", b"")
    return hashlib.md5(b).hexdigest()


def gstar(cl):
    s = np.bincount(cl).astype(float)
    s = s[s > 0]
    m = s.mean()
    return float(len(s) / (1 + ((s - m) ** 2).mean() / m ** 2))


# ---------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------
def read_bin(path, n):
    raw = path.read_bytes()
    m = int(np.frombuffer(raw[:8], "<i8")[0])
    v = np.frombuffer(raw[8:], "<f8").copy()
    if not (m == n == len(v)):
        raise SystemExit(f"{path.name}: length {m} / {len(v)}, frame {n}")
    return v


def jls_float_vector(raw, key):
    """The Float64 vector stored under String key `key` in a Julia-serialized Dict, read from the bytes:
    the string tag, the key's length and bytes, then the array tag, the Float64 type tag and the length
    (a literal 0..32 or an Int32). Exactly one match is required."""
    kb = key.encode()
    pat = b"!" + bytes([len(kb)]) + kb + b"\x15\x00\x0e"
    hits, i = [], raw.find(pat)
    while i >= 0:
        hits.append(i)
        i = raw.find(pat, i + 1)
    if len(hits) != 1:
        return None, {"key": key, "matches": len(hits)}
    j = hits[0] + len(pat)
    tag = raw[j]
    if tag == 0x31:                                   # Int32 length
        n = struct.unpack("<i", raw[j + 1:j + 5])[0]
        j += 5
    elif 0xDF <= tag <= 0xDF + 32:                    # a small literal length
        n = tag - 0xDF
        j += 1
    else:
        return None, {"key": key, "matches": 1, "length_tag": int(tag)}
    if j + 8 * n > len(raw):
        return None, {"key": key, "matches": 1, "n": n, "overruns_file": True}
    v = np.frombuffer(raw[j:j + 8 * n], "<f8").copy()
    return v, {"key": key, "matches": 1, "n": int(n), "data_start": int(j), "data_end": int(j + 8 * n),
               "file_bytes": len(raw)}


def load(E):
    cols = D.X_COLS + D.IVS + ["is_B", "spread_ann", "deposit_type", "CodConglomeradoPrudencial", "time_id"]
    path = paths.demand_parquet_dir() / D.FRAMES[E]
    d = pq.read_table(path, columns=cols).to_pandas()
    n = len(d)
    X = np.column_stack([d[D.X_COLS].to_numpy(float), (~d["is_B"].to_numpy(bool)).astype(float)])
    Z = d[D.IVS].to_numpy(float)
    codes, names = pd.factorize(d["CodConglomeradoPrudencial"].astype(str))
    q = pd.factorize(d["time_id"].astype(str))[0]
    return dict(path=path, X=X, Z=Z, p=d["spread_ann"].to_numpy(float) / 100.0, k=d["deposit_type"].to_numpy(),
                cl=codes, q=q, n=n, delta_logit=read_bin(LOGIT / f"logit_delta_E{E}_spec_12.bin", n))


# ---------------------------------------------------------------------------------------------------
# The linear step and its inference (W1's functions, extended to return the whole theta)
# ---------------------------------------------------------------------------------------------------
def spread_hat(X, Z, p, k):
    sh = p.copy()
    H = np.column_stack([X, Z])
    for kv in (4, 5):
        r = k == kv
        b, *_ = np.linalg.lstsq(H[r], p[r], rcond=None)
        sh[r] = H[r] @ b
    return sh


def iv(delta, p, X, sh, cl, seed=SEED):
    """OLS of delta on [sh, X]; xi on [p, X]; CR1 as compute_cluster_se (G/(G-1) N/(N-K)); the score WCB."""
    keep = np.abs(X).max(axis=0) > 0
    X = X[:, keep]
    Xh, Xf = np.column_stack([sh, X]), np.column_stack([p, X])
    N, K = Xh.shape
    rank = int(np.linalg.matrix_rank(Xh))
    XhX = Xh.T @ Xh
    Xi = np.linalg.inv(XhX)
    th = Xi @ (Xh.T @ delta)
    xi = delta - Xf @ th
    c = pd.factorize(cl)[0]
    G = int(c.max() + 1)
    S = np.zeros((G, K))
    np.add.at(S, c, Xh * xi[:, None])
    corr = G / (G - 1) * N / (N - K)
    V = Xi @ (S.T @ S) @ Xi * corr
    IF = S @ Xi.T
    rng = np.random.default_rng(seed)
    W = WEBB[rng.integers(0, 6, size=(B, G))]
    se_w = (th[None, :] + W @ IF).std(axis=0, ddof=1)
    gs = gstar(c)
    a, s_c, s_w = th[0], np.sqrt(V[0, 0]), se_w[0]
    return dict(alpha=float(a), se_cr1=float(s_c), se_wcb=float(s_w), n=int(N), G=G, G_star=gs, K=int(K),
                rank=rank, dropped_zero_cols=int((~keep).sum()),
                p_wcb_tGstar=float(2 * stats.t.sf(abs(a / s_w), gs)),
                p_cr1_tGstar=float(2 * stats.t.sf(abs(a / s_c), gs)),
                theta=[float(x) for x in th], se_cr1_all=[float(x) for x in np.sqrt(np.diag(V))])


def ols_qfe(delta, p, X, q, cl, seed=SEED):
    """OLS of delta on [p, X] with quarter fixed effects (within quarter); CR1 with K counting the
    quarter effects; the score WCB on the demeaned design (Frisch-Waugh-Lovell)."""
    qq = pd.factorize(q)[0]
    T = int(qq.max() + 1)

    def dm(v):
        v = np.atleast_2d(v.T).T
        s = np.zeros((T, v.shape[1]))
        np.add.at(s, qq, v)
        cnt = np.bincount(qq, minlength=T).astype(float)
        return v - (s / cnt[:, None])[qq]
    A = dm(np.column_stack([p, X]))
    y = dm(delta)[:, 0]
    nrm = np.abs(A).max(axis=0)
    keep = nrm > 1e-12
    keep[0] = True
    A = A[:, keep]
    # drop columns collinear after demeaning (pivoted rank by sequential QR)
    cols = [0]
    for j in range(1, A.shape[1]):
        if np.linalg.matrix_rank(A[:, cols + [j]]) == len(cols) + 1:
            cols.append(j)
    A = A[:, cols]
    N, K = A.shape
    Xi = np.linalg.inv(A.T @ A)
    th = Xi @ (A.T @ y)
    e = y - A @ th
    c = pd.factorize(cl)[0]
    G = int(c.max() + 1)
    S = np.zeros((G, K))
    np.add.at(S, c, A * e[:, None])
    corr = G / (G - 1) * N / (N - K - T)
    V = Xi @ (S.T @ S) @ Xi * corr
    IF = S @ Xi.T
    rng = np.random.default_rng(seed)
    W = WEBB[rng.integers(0, 6, size=(B, G))]
    se_w = (th[None, :] + W @ IF).std(axis=0, ddof=1)
    gs = gstar(c)
    a, s_c, s_w = th[0], np.sqrt(V[0, 0]), se_w[0]
    return dict(alpha=float(a), se_cr1=float(s_c), se_wcb=float(s_w), n=int(N), G=G, G_star=gs, K=int(K),
                n_quarters=T, x_cols_kept=int(K - 1),
                p_wcb_tGstar=float(2 * stats.t.sf(abs(a / s_w), gs)),
                p_cr1_tGstar=float(2 * stats.t.sf(abs(a / s_c), gs)),
                within_share_of_spread_var=float(A[:, 0].var() / np.var(p)))


def mx(X):
    Q, _ = np.linalg.qr(X[:, np.abs(X).max(axis=0) > 0])
    return lambda v: v - Q @ (Q.T @ v)


def mop(Zx, X, y, cl):
    """MOP effective F (no small-sample factor; and with CR1), K_eff and the simplified critical values
    (TSLS, x = 1/tau, 5 %). Z is standardised through its SVD (any rank-deficient direction dropped),
    which leaves F_eff and K_eff unchanged (both are invariant to a nonsingular transform of Z)."""
    keep = np.abs(X).max(axis=0) > 0
    X = X[:, keep]
    M = mx(X)
    Zt, yt = M(Zx), M(y)
    n = len(yt)
    U, s, _ = np.linalg.svd(Zt, full_matrices=False)
    r = s > 1e-9 * s.max()
    Zs = U[:, r] * np.sqrt(n)
    L = Zs.shape[1]
    pis = Zs.T @ yt / n
    v = yt - Zs @ pis
    c_ = pd.factorize(cl)[0]
    G = int(c_.max() + 1)
    Sg = np.zeros((G, L))
    np.add.at(Sg, c_, Zs * v[:, None])
    W2 = Sg.T @ Sg / n
    feff = float(n * pis @ pis / np.trace(W2))
    K = X.shape[1] + L
    cr1 = G / (G - 1) * (n - 1) / (n - K)
    lam = np.sort(np.linalg.eigvalsh(W2))[::-1]
    cv = {}
    for tau in (0.05, 0.10, 0.20, 0.30):
        x = 1 / tau
        keff = np.trace(W2) ** 2 * (1 + 2 * x) / (np.trace(W2 @ W2) + 2 * x * np.trace(W2) * lam[0])
        cv[f"tau{int(tau * 100):02d}"] = {"K_eff": float(keff), "crit": float(stats.ncx2.ppf(0.95, keff, x * keff) / keff)}
    return dict(n=int(n), L=int(L), L_supplied=int(Zx.shape[1]), G=G, G_star=gstar(c_), F_eff=feff,
                F_eff_cr1=feff / cr1, crit=cv)


def intervals(acc, grid):
    out, i = [], 0
    while i < len(acc):
        if acc[i]:
            j = i
            while j + 1 < len(acc) and acc[j + 1]:
                j += 1
            out.append([float(grid[i]) if i > 0 else -np.inf, float(grid[j]) if j < len(acc) - 1 else np.inf])
            i = j + 1
        else:
            i += 1
    return out


def ar(delta, p, X, H, cl, W):
    """W1's Anderson-Rubin sets: the CR1 Wald of the instruments in the regression of delta - a0 p on
    [X, H] (chi2(L)), and the fixed-weight statistic with its restricted wild cluster bootstrap."""
    M = mx(X)
    yt, pt = M(delta), M(p)
    Ht = M(H)
    nrm = np.linalg.norm(Ht, axis=0)
    Ht = Ht[:, nrm > 1e-8 * max(nrm.max(), 1)]
    _, r = np.linalg.qr(Ht)
    rk = np.abs(np.diag(r)) > 1e-10 * np.abs(np.diag(r)).max()
    Ht = Ht[:, rk]
    n, L = Ht.shape
    c_ = pd.factorize(cl)[0]
    G = int(c_.max() + 1)
    Kx = int((np.abs(X).max(axis=0) > 0).sum())
    corr = G / (G - 1) * (n - 1) / (n - Kx - L)
    Qi = np.linalg.inv(Ht.T @ Ht)
    Ty, Tp = Ht.T @ yt, Ht.T @ pt
    a, b = Qi @ Ty, Qi @ Tp
    ey, ep = yt - Ht @ a, pt - Ht @ b
    sy, sp = np.zeros((G, L)), np.zeros((G, L))
    np.add.at(sy, c_, Ht * ey[:, None]); np.add.at(sp, c_, Ht * ep[:, None])
    A, Bm, C = sy.T @ sy, sy.T @ sp, sp.T @ sp
    wald = np.empty(len(GRID))
    for i, a0 in enumerate(GRID):
        t = Ty - a0 * Tp
        Om = A - a0 * (Bm + Bm.T) + a0 * a0 * C
        wald[i] = t @ np.linalg.solve(Om, t) / corr
    crit = stats.chi2.ppf(0.95, L)
    ty, tp = np.zeros((G, L)), np.zeros((G, L))
    np.add.at(ty, c_, Ht * yt[:, None]); np.add.at(tp, c_, Ht * pt[:, None])
    Wg = W[:, :G]
    Uy, Up = Wg @ ty, Wg @ tp
    qyy = np.einsum("bi,ij,bj->b", Uy, Qi, Uy)
    qyp = np.einsum("bi,ij,bj->b", Uy, Qi, Up)
    qpp = np.einsum("bi,ij,bj->b", Up, Qi, Up)
    Syy, Syp, Spp = Ty @ Qi @ Ty, Ty @ Qi @ Tp, Tp @ Qi @ Tp
    S0 = Syy - 2 * GRID * Syp + GRID ** 2 * Spp
    Sb = qyy[:, None] - 2 * GRID[None, :] * qyp[:, None] + GRID[None, :] ** 2 * qpp[:, None]
    pw = (1 + (Sb >= S0[None, :]).sum(axis=0)) / (B + 1)
    return dict(L=int(L), G=G, grid=[float(GRID[0]), float(GRID[-1]), 0.001],
                alpha_2sls_these_instruments=float(Syp / Spp),
                wald_chi2_set=intervals(wald <= crit, GRID), wald_min=float(wald.min()),
                wald_argmin=float(GRID[wald.argmin()]), wald_crit=float(crit),
                wcb_fixed_weight_set=intervals(pw >= 0.05, GRID), wcb_p_max=float(pw.max()),
                wcb_argmax=float(GRID[pw.argmax()]))


def decomposition(delta, p, X, sh, g12, g45):
    """The pooled alpha as a weighted average of the two groups' contributions (FWL on the pooled X)."""
    zt = mx(X)(sh)
    num12, num45 = float(zt[g12] @ delta[g12]), float(zt[g45] @ delta[g45])
    den12, den45 = float(zt[g12] @ p[g12]), float(zt[g45] @ p[g45])
    return {"weight_types_12": den12 / (den12 + den45), "alpha_12_part": num12 / den12,
            "alpha_45_part": num45 / den45, "alpha_recombined": (num12 + num45) / (den12 + den45)}


# ---------------------------------------------------------------------------------------------------
def run(E, Wmat):
    S = load(E)
    X, Z, p, k, cl, q, n = S["X"], S["Z"], S["p"], S["k"], S["cl"], S["q"], S["n"]
    say(f"E{E}: frame {S['path'].name}, {n} rows; types {dict(zip(*np.unique(k, return_counts=True)))}")
    sh = spread_hat(X, Z, p, k)
    g12, g45 = np.isin(k, [1, 2]), np.isin(k, [4, 5])
    g4, g5 = k == 4, k == 5
    groups = {"all": np.ones(n, bool), "types_12": g12, "types_45": g45, "type_4": g4, "type_5": g5}
    r = {"frame": S["path"].name, "n_rows": n}

    # ---- G1: the logit, W1's numbers ------------------------------------------------------------
    dl = S["delta_logit"]
    prod = json.loads((LOGIT / "logit_summary_spec_12.json").read_text(encoding="utf-8"))[f"E{E}_full_dtype"]
    pooled = iv(dl, p, X, sh, cl)
    t12, t45 = iv(dl[g12], p[g12], X[g12], p[g12], cl[g12]), iv(dl[g45], p[g45], X[g45], sh[g45], cl[g45])
    f4 = mop(Z[g4], X[g4], p[g4], cl[g4])
    g1 = {"pooled_alpha": pooled["alpha"], "pooled_alpha_stored": prod["theta1"][0],
          "types_12_ols": t12["alpha"], "types_12_ols_W1": W1[E]["types_12_ols"],
          "types_45_iv": t45["alpha"], "types_45_iv_W1": W1[E]["types_45_iv"],
          "feff_type4": f4["F_eff"], "feff_type4_W1": W1[E]["feff_type4"]}
    g1["pass"] = bool(abs(pooled["alpha"] - prod["theta1"][0]) < 1e-7
                      and abs(t12["alpha"] - W1[E]["types_12_ols"]) < 1e-8
                      and abs(t45["alpha"] - W1[E]["types_45_iv"]) < 1e-8
                      and abs(f4["F_eff"] - W1[E]["feff_type4"]) < 1e-8)
    say(f"E{E} G1: pooled {pooled['alpha']:.9f} (stored {prod['theta1'][0]}); types 1-2 OLS {t12['alpha']:.9f} "
        f"(W1 {W1[E]['types_12_ols']:.9f}); types 4-5 IV {t45['alpha']:.9f} (W1 {W1[E]['types_45_iv']:.9f}); "
        f"F_eff type 4 {f4['F_eff']:.6f} (W1 {W1[E]['feff_type4']:.6f}) -> {'PASS' if g1['pass'] else 'FAIL'}")
    if not g1["pass"]:
        raise SystemExit(f"G1 failed for E{E}: {g1}")
    r["gate_G1"] = g1

    # ---- Request 2: the logit with the spread exogenous ------------------------------------------
    r["logit_exog"] = {g: iv(dl[m], p[m], X[m], p[m], cl[m]) for g, m in groups.items()}
    r["logit_exog"]["types_45_quarter_fe"] = ols_qfe(dl[g45], p[g45], X[g45], q[g45], cl[g45])
    r["logit_iv"] = {g: iv(dl[m], p[m], X[m], sh[m], cl[m]) for g, m in groups.items()}
    r["logit_iv"]["decomposition"] = decomposition(dl, p, X, sh, g12, g45)
    for g in ("all", "types_12", "types_45", "type_4", "type_5", "types_45_quarter_fe"):
        a_ = r["logit_exog"][g]
        say(f"E{E} logit OLS {g:<20} alpha {a_['alpha']:+.4f} se CR1 {a_['se_cr1']:.4f} WCB {a_['se_wcb']:.4f} "
            f"N {a_['n']} G {a_['G']}")

    # ---- G2: ext1's delta ------------------------------------------------------------------------
    res = json.loads((RAW / f"blp_results_E{E}_spec_12_ext1.json").read_text(encoding="utf-8"))
    summ = json.loads((RAW / f"blp_summary_E{E}_ext1_gpu_ift.json").read_text(encoding="utf-8"))["12"]
    ck_path = RAW / f"blp_checkpoint_E{E}_spec_12_ext1.jls"
    raw = ck_path.read_bytes()
    if raw[:3] != b"7JL":
        raise SystemExit(f"{ck_path.name}: not a Julia serialization header")
    de, info_d = jls_float_vector(raw, "delta_star")
    t2, info_t = jls_float_vector(raw, "theta2_star")
    ok_parse = (de is not None and t2 is not None and len(de) == n and info_d["data_end"] == len(raw)
                and np.all(np.isfinite(de)) and len(t2) == len(res["theta2"])
                and np.max(np.abs(t2 - np.array(res["theta2"]))) < 1e-7)
    if not ok_parse:
        raise SystemExit(f"G2 parse failed for E{E}: {info_d} {info_t}")
    e_pool = iv(de, p, X, sh, cl)
    th, th_res = np.array(e_pool["theta"]), np.array(res["theta1"])
    se_lin = np.array(res["theta1_se_linear_iv"])
    rel_t = np.abs(th - th_res) / np.maximum(np.abs(th_res), 1e-3)
    rel_s = np.abs(np.array(e_pool["se_cr1_all"]) - se_lin) / se_lin
    d_alpha = abs(e_pool["alpha"] - summ["theta1_alpha"][0])
    g2 = {"checkpoint": ck_path.name, "checkpoint_bytes": len(raw), "delta_star": info_d, "theta2_star": info_t,
          "theta2_star_values": [float(x) for x in t2], "alpha_reproduced": e_pool["alpha"],
          "alpha_summary": summ["theta1_alpha"][0], "abs_diff_alpha": d_alpha,
          "max_rel_diff_theta1": float(rel_t.max()), "max_rel_diff_se_linear_iv": float(rel_s.max()),
          "se_cr1_alpha_reproduced": e_pool["se_cr1"], "se_linear_iv_alpha_stored": float(se_lin[0]),
          "se_joint_alpha_stored": res["theta1_se"][0], "se_method_stored": res["se_method"],
          "delta_minus_logit_delta": {"mean": float((de - dl).mean()), "sd": float((de - dl).std()),
                                      "max_abs": float(np.abs(de - dl).max())}}
    g2["pass"] = bool(d_alpha < 1e-9 and rel_t.max() < 2e-7 and rel_s.max() < 2e-6)
    say(f"E{E} G2: ext1 alpha {e_pool['alpha']:.12f} (summary {summ['theta1_alpha'][0]}); max rel diff theta1 "
        f"{rel_t.max():.2e}, se {rel_s.max():.2e} -> {'PASS' if g2['pass'] else 'FAIL'}")
    if not g2["pass"]:
        raise SystemExit(f"G2 failed for E{E}: {g2}")
    r["gate_G2"] = g2
    # the result .jls (delta and xi), a second copy: compared when it parses, reported either way
    try:
        raw_r = (RAW / f"blp_results_E{E}_spec_12_ext1.jls").read_bytes()
        dr, info_r = jls_float_vector(raw_r, "delta")
        xr, info_x = jls_float_vector(raw_r, "xi")
        cmp = {"delta_parse": info_r, "xi_parse": info_x}
        if dr is not None and len(dr) == n:
            cmp["max_abs_delta_result_minus_checkpoint"] = float(np.abs(dr - de).max())
        if xr is not None and len(xr) == n:
            xi_me = de - np.column_stack([p, X]) @ th
            cmp["max_abs_xi_result_minus_recomputed"] = float(np.abs(xr - xi_me).max())
        r["ext1_result_jls_check"] = cmp
        say(f"E{E} result .jls check: {({k_: v for k_, v in cmp.items() if k_.startswith('max')})}")
    except Exception as ex:                             # informational only
        r["ext1_result_jls_check"] = {"error": repr(ex)}

    # ---- Request 1: ext1's linear step by deposit type -------------------------------------------
    e1 = {"all": e_pool}
    e1["types_12"] = iv(de[g12], p[g12], X[g12], p[g12], cl[g12])     # the estimator: the spread is its own instrument
    for g in ("types_45", "type_4", "type_5"):
        m = groups[g]
        e1[g] = iv(de[m], p[m], X[m], sh[m], cl[m])
    e1["decomposition"] = decomposition(de, p, X, sh, g12, g45)
    e1["ols_spread_exogenous"] = {g: iv(de[m], p[m], X[m], p[m], cl[m]) for g, m in groups.items()}
    e1["ols_spread_exogenous"]["types_45_quarter_fe"] = ols_qfe(de[g45], p[g45], X[g45], q[g45], cl[g45])
    for g in ("all", "types_12", "types_45", "type_4", "type_5"):
        say(f"E{E} ext1 linear step {g:<9} alpha {e1[g]['alpha']:+.4f} se CR1 {e1[g]['se_cr1']:.4f} "
            f"WCB {e1[g]['se_wcb']:.4f} N {e1[g]['n']}; OLS {e1['ols_spread_exogenous'][g]['alpha']:+.4f}")
    say(f"E{E} ext1 decomposition {e1['decomposition']}")
    # sensitivity only (it selects on the outcome): the same fits without the rows where ext1's delta is
    # more than 10 from the logit's (the heavy tails of delta_star)
    tr = np.abs(de - dl) <= 10.0
    e1["sensitivity_trim10"] = {"rule": "|delta_ext1 - delta_logit| <= 10; selects on the outcome, a sensitivity only",
                                "rows_dropped": {g: int((groups[g] & ~tr).sum()) for g in groups}}
    for g in ("all", "types_12", "types_45", "type_4", "type_5"):
        m = groups[g] & tr
        e1["sensitivity_trim10"][g] = {"iv": iv(de[m], p[m], X[m], (p if g == "types_12" else sh)[m], cl[m]),
                                       "ols": iv(de[m], p[m], X[m], p[m], cl[m])}
        s_ = e1["sensitivity_trim10"][g]
        say(f"E{E} ext1 trim10 {g:<9} IV {s_['iv']['alpha']:+.4f} ({s_['iv']['se_cr1']:.4f}) OLS {s_['ols']['alpha']:+.4f} "
            f"({s_['ols']['se_cr1']:.4f}) N {s_['iv']['n']}")
    r["ext1"] = e1

    # ---- the first stages: MOP effective F by subset ---------------------------------------------
    x5 = X[g45] * g5[g45][:, None]
    z4, z5 = Z[g45] * g4[g45][:, None], Z[g45] * g5[g45][:, None]
    r["feff"] = {"types_12": "not applicable: the spread is exogenous on types 1-2 (its own instrument)",
                 "type_4": f4, "type_5": mop(Z[g5], X[g5], p[g5], cl[g5]),
                 "types_45_production": mop(np.column_stack([x5, z4, z5]), X[g45], p[g45], cl[g45]),
                 "types_45_external_only": mop(np.column_stack([z4, z5]), np.column_stack([X[g45], x5]), p[g45], cl[g45])}
    for g in ("type_4", "type_5", "types_45_production", "types_45_external_only"):
        f = r["feff"][g]
        say(f"E{E} F_eff {g:<24} {f['F_eff']:.3f} (L {f['L']}, crit tau10 {f['crit']['tau10']['crit']:.2f}, "
            f"tau20 {f['crit']['tau20']['crit']:.2f})")

    # ---- supplementary: AR sets for alpha on types 4-5 -------------------------------------------
    r["ar_types_45"] = {}
    Hs = {"H_ext": np.column_stack([z4, z5]), "H_prod": np.column_stack([x5, z4, z5])}
    for dn, dv in (("logit", dl), ("ext1", de)):
        for hn, H in Hs.items():
            a_ = ar(dv[g45], p[g45], X[g45], H, cl[g45], Wmat)
            r["ar_types_45"][f"{dn}_{hn}"] = a_
            say(f"E{E} AR types 4-5 {dn} {hn}: L {a_['L']}, Wald set {a_['wald_chi2_set']}, "
                f"WCB set {a_['wcb_fixed_weight_set']}")
    return r


def top(s, k=10):
    s = s.sort_values(ascending=False).head(k)
    return [[str(i), float(v)] for i, v in s.items()]


def tails(E):
    """Step `tails` (coordinator 23:24): the rows where ext1's delta is more than 10 from the logit's. By
    deposit type, B/D and quarter; the top markets and conglomerates by count and by deposits; the overlap
    with the market-quarters where the inside-share cap binds (make_paper_numbers.collect_share_cap: B
    markets whose share_B_cond sum to TOTAL_SHARE_CAP within 1e-9, and national D quarters whose share_D
    do); their deposits; and, as sensitivities, the linear step without the cap-bound market-quarters and
    without the tail rows, each with the projection re-run on the kept rows (theta_2 is not re-estimated)."""
    S = load(E)
    X, Z, p, k, cl, n = S["X"], S["Z"], S["p"], S["k"], S["cl"], S["n"]
    d = pq.read_table(S["path"], columns=["mca_code", "time_id", "is_B", "deposit_type", "CodConglomeradoPrudencial",
                                          "deposit_balance", "Dep_Act", "share_B_cond", "share_D"]).to_pandas()
    dl = S["delta_logit"]
    raw = (RAW / f"blp_checkpoint_E{E}_spec_12_ext1.jls").read_bytes()
    de, info = jls_float_vector(raw, "delta_star")
    if de is None or len(de) != n:
        raise SystemExit(f"E{E}: delta_star did not parse: {info}")
    tail = np.abs(de - dl) > 10.0
    isB = d["is_B"].fillna(False).astype(bool).to_numpy()
    cap, tol = float(D.PROD["TOTAL_SHARE_CAP"]), 1e-9
    g = d.groupby(["mca_code", "time_id"])
    bsum = d["share_B_cond"].where(isB, 0.0).groupby([d["mca_code"], d["time_id"]]).transform("sum").to_numpy()
    bcnt = pd.Series(isB.astype(int)).groupby([d["mca_code"], d["time_id"]]).transform("sum").to_numpy()
    dsum = d["share_D"].where(~isB, 0.0).groupby(d["time_id"]).transform("sum").to_numpy()
    at_b = isB & (np.abs(bsum - cap) < tol)
    at_d = (~isB) & (np.abs(dsum - cap) < tol)
    at = at_b | at_d
    single = at_b & (bcnt == 1)
    dep = d["deposit_balance"].astype(float).fillna(0.0).to_numpy()
    act = d["Dep_Act"].astype(float).fillna(0.0).to_numpy()
    dd = pd.DataFrame({"tail": tail, "at": at, "single": single, "isB": isB, "k": k, "q": d["time_id"].astype(str),
                       "mkt": d["mca_code"].astype(str), "cong": d["CodConglomeradoPrudencial"].astype(str),
                       "dep": dep, "act": act, "diff": de - dl})
    T = dd[dd["tail"]]
    r = {"rows": n, "tail_rows": int(tail.sum()), "tail_share_rows": float(tail.mean()),
         "tail_share_deposit_balance": float(dep[tail].sum() / dep.sum()),
         "tail_share_Dep_Act": float(act[tail].sum() / act.sum()),
         "tail_sign": {"below_minus10": int((dd["diff"] < -10).sum()), "above_plus10": int((dd["diff"] > 10).sum())}}
    r["by_type"] = {str(t): {"rows": int(len(x)), "tail": int(x["tail"].sum()), "share": float(x["tail"].mean()),
                             "tail_dep_share_of_type": float(x.loc[x["tail"], "dep"].sum() / max(x["dep"].sum(), 1e-30))}
                    for t, x in dd.groupby("k")}
    r["by_BD"] = {("B" if b_ else "D"): {"rows": int(len(x)), "tail": int(x["tail"].sum()), "share": float(x["tail"].mean())}
                  for b_, x in dd.groupby("isB")}
    r["by_quarter_tail_rows"] = {q_: int(v) for q_, v in T.groupby("q").size().sort_index().items()}
    r["top_markets_by_count"] = top(T.groupby("mkt").size().astype(float))
    r["top_markets_by_tail_deposits_share_of_all"] = top(T.groupby("mkt")["dep"].sum() / dep.sum())
    r["n_markets_with_tail"] = int(T["mkt"].nunique())
    r["n_market_quarters_with_tail"] = int(T.groupby(["mkt", "q"]).ngroups)
    r["top_conglomerates_by_count"] = top(T.groupby("cong").size().astype(float))
    r["top_conglomerates_by_tail_deposits_share_of_all"] = top(T.groupby("cong")["dep"].sum() / dep.sum())
    r["n_conglomerates_with_tail"] = int(T["cong"].nunique())
    r["cap_overlap"] = {"cap": cap, "rows_at_cap": int(at.sum()), "B_rows_at_cap": int(at_b.sum()),
                        "D_rows_at_cap": int(at_d.sum()), "B_single_row_markets_at_cap_rows": int(single.sum()),
                        "B_market_quarters_at_cap": int(dd[at_b].groupby(["mkt", "q"]).ngroups),
                        "tail_and_at_cap": int((tail & at).sum()),
                        "share_of_tail_at_cap": float((tail & at).sum() / max(tail.sum(), 1)),
                        "share_of_at_cap_in_tail": float((tail & at).sum() / max(at.sum(), 1)),
                        "share_of_tail_in_single_row_cap_markets": float((tail & single).sum() / max(tail.sum(), 1)),
                        "tail_rate_at_cap": float(tail[at].mean()) if at.any() else None,
                        "tail_rate_not_at_cap": float(tail[~at].mean())}
    r["contraction_failure_markets_val1112_D0"] = ("not on disk locally: no D0/val1112 market list found under "
                                                   "BLP_RESULTS, Rout or CLUSTER_META, or in the scratchpad")
    # sensitivities: the linear step on the kept rows, the projection re-run on them
    def fit(keep):
        shk = spread_hat(X[keep], Z[keep], p[keep], k[keep])
        return {"ext1": iv(de[keep], p[keep], X[keep], shk, cl[keep]), "logit": iv(dl[keep], p[keep], X[keep], shk, cl[keep]),
                "rows_kept": int(keep.sum())}
    r["sensitivity"] = {"note": "selects rows; theta_2 not re-estimated; a sensitivity only",
                        "drop_cap_bound_market_quarters": fit(~at),
                        "drop_single_row_cap_markets": fit(~single),
                        "drop_tail_rows_reprojected": fit(~tail)}
    for nm, v in r["sensitivity"].items():
        if isinstance(v, dict):
            say(f"E{E} sens {nm:<34} ext1 {v['ext1']['alpha']:+.4f} ({v['ext1']['se_cr1']:.4f}) logit "
                f"{v['logit']['alpha']:+.4f} ({v['logit']['se_cr1']:.4f}) kept {v['rows_kept']}")
    say(f"E{E} tails: {r['tail_rows']} rows ({r['tail_share_rows']:.4f}), deposits {r['tail_share_deposit_balance']:.4f}, "
        f"Dep_Act {r['tail_share_Dep_Act']:.4f}; at cap {r['cap_overlap']['rows_at_cap']}, tail&cap "
        f"{r['cap_overlap']['tail_and_at_cap']} ({r['cap_overlap']['share_of_tail_at_cap']:.3f} of tail)")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", default="3,4")
    ap.add_argument("--step", default="main", choices=["main", "tails"])
    args = ap.parse_args()
    Es = [int(x) for x in args.E.split(",")]
    if args.step == "tails":
        out = {"meta": {"script": Path(__file__).name, "script_md5_lf": md5_lf(__file__), "step": "tails",
                        "run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "rule": "|delta_ext1 - delta_logit| > 10"}}
        for E in Es:
            out[f"E{E}"] = tails(E)
        OUT.mkdir(parents=True, exist_ok=True)
        js = OUT / "w1_tails.json"
        js.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8", newline="\n")
        lines = [f"{md5_lf(f)}  {f.name}\n" for f in sorted(OUT.glob("*.json"))]
        (OUT / "MD5SUMS").write_text("".join(lines), encoding="utf-8", newline="\n")
        say(f"wrote {js} and MD5SUMS")
        return
    rng = np.random.default_rng(SEED)
    Wmat = WEBB[rng.integers(0, 6, size=(B, 400))]
    out = {"meta": {"script": Path(__file__).name, "script_md5_lf": md5_lf(__file__),
                    "run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "B": B, "seed": SEED,
                    "units": "spread in pp (spread_ann / 100); alpha per pp",
                    "inputs": {}}}
    for E in Es:
        for f in (paths.demand_parquet_dir() / D.FRAMES[E], LOGIT / f"logit_delta_E{E}_spec_12.bin",
                  RAW / f"blp_checkpoint_E{E}_spec_12_ext1.jls", RAW / f"blp_results_E{E}_spec_12_ext1.json",
                  RAW / f"blp_summary_E{E}_ext1_gpu_ift.json"):
            out["meta"]["inputs"][f.name] = md5_lf(f)
        out[f"E{E}"] = run(E, Wmat)
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / "w1_followups.json"
    js.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8", newline="\n")
    (OUT / "MD5SUMS").write_text(f"{md5_lf(js)}  {js.name}\n", encoding="utf-8", newline="\n")
    say(f"wrote {js} and MD5SUMS")


if __name__ == "__main__":
    main()
