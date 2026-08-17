"""
diag_moment_reduction.py
================================================================================
LOCAL test of the two moment-reduction fixes for the weight-matrix problem (weakiv_methods.md §7.6,
options 1 and 2) — no GPU, no Julia, no cluster time.

The finding this responds to: CUE and the Stock–Wright S-set fail on the 16-moment design because
G* ≈ 5.5 effective clusters cannot support a 16×16 clustered Ω̂ (~6 of 16 eigen-directions truncated
in every deposit-type block). The instruments hold only K_eff ≈ 3.5 independent directions, so
shrinking the moment count should discard almost nothing while making Ω̂ estimable. This script
measures whether that is true, at FIXED θ₂ on the structural δ(θ̂₂):

  option 2 — reduce to r moments: principal components of the partialled instrument block
             (r = 2..5), plus the 5 leave-one-out rival characteristics as a named alternative.
             For each variant: MOP effective F, 2SLS α, the 1-parameter CUE over an α grid
             (closed form per point — argmin and convergence are read off the curve, no optimiser),
             the 95% S-set {α : S(α) ≤ χ²₀.₉₅(r)}, Hansen J at the argmin (df r−1), and the
             Ω̂ condition number at the argmin.
  option 1 — just-identified optimal IV: z* = fitted first stage on all 16. The point estimate is
             numerically 2SLS; what changes is inference — one moment ⇒ no weight matrix and no
             overidentification, so the AR set is the inversion of a t-test and is non-empty by
             construction (the sample moment is exactly zero at α_JI). Reported against χ²₁, a
             few-cluster t(G*−1)² critical value, and (--wcb) wild-cluster-bootstrap criticals
             mirroring weak_iv_analysis._battery exactly (Webb 6-point, shared draws, null imposed,
             Ω̂ held at its observed value per grid point — the non-studentised score bootstrap).
             Two further variants:
               ji_engine — z* = the ENGINE's own generated instrument (per-type projected spread,
             types 1–2 raw, no intercept). By FWL the engine's θ₁ step IS just-identified IV with
             this instrument, so α must reproduce the engine's reported θ₁[1] to ~5e-6 (hard assert)
             and the inverted set is the identification-robust CI for the paper's own estimator.
               ji_split — 2-fold CLUSTER-level cross-fit z* (first stage fitted on the complement
             fold) purging the generated-regressor optimism of the in-sample z*; the in-sample-vs-
             split eff-F gap is the measurement, and a collapsed fold is a finding, not a bug.

Everything is in the PARTIALLED representation: X (product characteristics + constant) is projected
out of δ, the spread and Z, leaving a 1-parameter problem in α. This matches the weak-IV battery's
conventions (uniform instrumenting — NOT the engine's per-type projection; the engine-replication
self-check of diag_cue_linear_step.py is re-run here only to validate row alignment). The full
16-moment 1-parameter CUE is computed as the within-script baseline the reductions are judged
against.

Inputs (already on disk): the demand parquets and cluster_processed/rc_delta_E{k}_spec_12_{stage}.bin.

Usage
-----
  python diag_moment_reduction.py                       # E5-E8, stage ext1, r = 2..5
  python diag_moment_reduction.py --routines 6 --grid=-2:0.005:2
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scipy.stats import chi2, t as student_t

from diag_cue_linear_step import (X_COLS, IV_COLS, _build_matrices, _load_delta,
                                  _cluster_codes, _dirs)
# _wild_weights imported from the battery so the WCB draws are BIT-IDENTICAL to the ji_ci_*/ar_ci_*
# sets in weak_iv*.json (same generator, same seed, same (B×G) shape) — that is what makes the
# cross-checks against the battery exact rather than approximate.
from weak_iv_analysis import _wild_weights
from utils.cluster import effective_cluster_stats

# The 5 leave-one-out rival characteristics — the "parsimonious" set that raised eff-F when the
# 16-instrument dilution was first diagnosed. Named columns, in engine order.
LOO5 = ["loo_log_assets", "loo_equity_ratio", "loo_basileia",
        "loo_credit_assets", "loo_npl_provision"]

ALPHA_TOL = 5e-6


def _resid(Y, X):
    """Project the columns of Y off X (least squares; X assumed full column rank here)."""
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ beta


def _cluster_sums(V, codes, G):
    """G×L matrix of within-cluster column sums."""
    L = V.shape[1]
    M = np.empty((G, L))
    for l in range(L):
        M[:, l] = np.bincount(codes, weights=V[:, l], minlength=G)
    return M


def _eff_F(p_t, Z_t, codes, G):
    """Montiel-Olea–Pflueger effective F of the partialled first stage p̃ on Z̃, clustered.

    Battery-exact form (weak_iv_analysis._battery): eff-F = ESS / tr(Ŵ₂Q⁻¹) with
    Ŵ₂ = (1/n)Σ_g(Z̃_g'v̂_g)(Z̃_g'v̂_g)' and Q = Z̃'Z̃/n — no CR1 small-sample factor, so the
    full-16 value reproduces the published 4.20/3.95/4.23/4.11 (E5–E8) exactly.
    """
    N = len(p_t)
    ZtZ = Z_t.T @ Z_t
    pi, *_ = np.linalg.lstsq(Z_t, p_t, rcond=None)
    u = p_t - Z_t @ pi
    Mv = _cluster_sums(Z_t * u[:, None], codes, G)
    W2 = (Mv.T @ Mv) / N
    Q_inv = np.linalg.pinv(ZtZ / N)
    return float((pi @ ZtZ @ pi) / np.trace(W2 @ Q_inv))


def _2sls(d_t, p_t, Z_t, codes, G):
    """Partialled 2SLS α and CR1 cluster SE."""
    ph = Z_t @ np.linalg.lstsq(Z_t, p_t, rcond=None)[0]
    a = float((ph @ d_t) / (ph @ p_t))
    xi = d_t - a * p_t
    S = np.bincount(codes, weights=ph * xi, minlength=G)
    N, L = len(d_t), Z_t.shape[1]
    crfac = (G / (G - 1)) * ((N - 1) / (N - 2))
    se = float(np.sqrt(crfac * (S @ S)) / abs(ph @ p_t))
    return a, se


def _cue_curve(a_del, a_spr, grid, N, pinv_rtol):
    """S(α) = N·ḡ'Ω̂(α)⁻¹ḡ over the grid, with per-cluster moments M_g(α) = aδ_g − α·ap_g.

    One parameter, so the whole CUE is a curve scan: the argmin IS the estimator, and convexity /
    set structure are read directly — no optimiser, no convergence question. Returns the curve plus
    the Ω̂ spectrum at the argmin.
    """
    L = a_del.shape[1]
    S = np.empty(len(grid))
    for i, a0 in enumerate(grid):
        M = a_del - a0 * a_spr
        g = M.sum(axis=0) / N
        Om = (M.T @ M) / N
        if L <= 6:
            try:
                S[i] = float(N * g @ np.linalg.solve(Om, g))
            except np.linalg.LinAlgError:
                S[i] = float(N * g @ np.linalg.pinv(Om, rcond=pinv_rtol, hermitian=True) @ g)
        else:
            S[i] = float(N * g @ np.linalg.pinv(Om, rcond=pinv_rtol, hermitian=True) @ g)
    i0 = int(np.argmin(S))
    M = a_del - grid[i0] * a_spr
    ev = np.linalg.eigvalsh((M.T @ M) / N)
    return S, i0, ev


def _set_from_curve(grid, S, crit):
    """{α : S ≤ crit} with edge/disconnection flags. `crit` may be scalar (χ², t²) or a length-J
    vector (per-point WCB criticals) — both broadcast in the comparison."""
    inside = S <= crit
    if not inside.any():
        return {"empty": True}
    idx = np.flatnonzero(inside)
    gaps = int((np.diff(idx) > 1).sum())
    return {"empty": False, "lo": float(grid[idx[0]]), "hi": float(grid[idx[-1]]),
            "touches_edge": bool(idx[0] == 0 or idx[-1] == len(grid) - 1),
            "disconnected": bool(gaps > 0), "n_segments": gaps + 1,
            "contains_zero": bool(inside[int(np.argmin(np.abs(grid)))])}


def _wcb_sets(a_del, a_spr, Wb, grid, N):
    """Wild-cluster-bootstrap S-curve and per-point criticals, mirroring _battery EXACTLY: shared
    draws across the grid, null imposed (the bootstrap score is w_g·(A_g − α₀B_g)), and Ω̂ held at
    its OBSERVED value per grid point (non-studentised score bootstrap — no per-draw Ω̂*, which is
    the battery convention; a fully studentised version would cost B pinvs per point).

    Returns (S_obs, crit) both length-J, on the S scale, ready for _set_from_curve. NB the observed
    S here uses the battery's default-rcond pinv, NOT _cue_curve's 1e-10 rtol — at cond(Ω̂)≈1e13
    the truncation differs and full16's S values shift; the battery convention is what makes the
    cross-check against weak_iv_ext1.json's ar_ci_* exact, so it wins inside this function while
    the χ² set keeps the old convention.
    """
    L = a_del.shape[1]
    if L == 1:
        # Scalar fast path: with Ω̂ observed, the O1 factor multiplies both sides of the acceptance
        # inequality and cancels, so the whole grid needs two matmuls and one broadcast.
        A1, B1 = a_del[:, 0], a_spr[:, 0]
        bA1, bB1 = Wb @ A1, Wb @ B1                                  # (B,)
        M2 = (bA1[:, None] - grid[None, :] * bB1[:, None]) ** 2      # (B, J)
        q = np.quantile(M2, 0.95, axis=0)                            # (J,)
        m_obs = A1.sum() - grid * B1.sum()                           # (J,)
        O1 = (A1 @ A1) - 2.0 * grid * (A1 @ B1) + grid ** 2 * (B1 @ B1)   # n·Ω̂(α), closed form
        O1 = np.where(O1 > 0, O1, np.nan)
        S_obs = m_obs ** 2 / O1
        crit = q / O1
        return np.nan_to_num(S_obs, nan=np.inf), np.nan_to_num(crit, nan=0.0)
    bA, bB = Wb @ a_del, Wb @ a_spr                                  # (B, L) each
    S_obs = np.empty(len(grid)); crit = np.empty(len(grid))
    for j, a0 in enumerate(grid):
        Mg = a_del - a0 * a_spr
        g = Mg.sum(axis=0) / N
        Oi = np.linalg.pinv((Mg.T @ Mg) / N)                         # battery's default-rcond pinv
        S_obs[j] = N * g @ Oi @ g
        gs = (bA - a0 * bB) / N
        crit[j] = np.quantile(N * np.einsum("bk,kl,bl->b", gs, Oi, gs), 0.95)
    return S_obs, crit


def _fold_assign(sizes, top, rng):
    """2-fold cluster partition for the cross-fit z*. Deposits are extremely top-heavy (top cluster
    ≈28% of rows, top-5 ≈88%), so a plain random split routinely produces a fold with no giant —
    whose first stage is a qualitatively different object. Fix the part that matters
    deterministically: the `top` largest clusters go greedy largest-first onto the lighter fold, so
    each fold always holds 2–3 giants; the ~300 small clusters are seed-shuffled and continue the
    greedy fill, keeping the split honest rather than hand-picked."""
    G = len(sizes)
    order_top = list(np.argsort(sizes)[::-1][:top])
    rest = np.setdiff1d(np.arange(G), order_top)
    rng.shuffle(rest)
    fold = np.empty(G, int)
    load = [0.0, 0.0]
    for c in order_top + list(rest):
        f = 0 if load[0] <= load[1] else 1
        fold[c] = f
        load[f] += float(sizes[c])
    return fold


def _fmt_set(s):
    if s.get("empty"):
        return "EMPTY"
    out = f"[{s['lo']:+.2f},{s['hi']:+.2f}]"
    if s["touches_edge"]:
        out += "^"
    if s["disconnected"]:
        out += f" ({s['n_segments']} seg)"
    return out


def _variant(name, Zr, d_t, p_t, codes, G, grid, args, jdf_offset=1, wcb=None):
    """Full report for one instrument set: strength, 2SLS, 1-parameter CUE curve, 95% S-set, J,
    and (when `wcb` carries the shared (B×G) weight matrix) the WCB-critical set."""
    N, L = len(d_t), Zr.shape[1]
    effF = _eff_F(p_t, Zr, codes, G)
    a2, se2 = _2sls(d_t, p_t, Zr, codes, G)
    a_del = _cluster_sums(Zr * d_t[:, None], codes, G)
    a_spr = _cluster_sums(Zr * p_t[:, None], codes, G)
    S, i0, ev = _cue_curve(a_del, a_spr, grid, N, args.pinv_rtol)
    crit = float(chi2.ppf(0.95, L))
    sset = _set_from_curve(grid, S, crit)
    wcb_extra = {}
    if wcb is not None:
        S_w, crit_w = _wcb_sets(a_del, a_spr, wcb, grid, N)
        wcb_extra = {"sset_wcb": _set_from_curve(grid, S_w, crit_w),
                     "curve_wcb": {"S": [float(x) for x in S_w[::args.curve_thin]],
                                   "crit": [float(x) for x in crit_w[::args.curve_thin]]}}
    n_trunc = int((ev < args.pinv_rtol * ev.max()).sum())
    # local minima count on the curve interior — the convexity read
    dS = np.sign(np.diff(S))
    n_min = int(((dS[:-1] < 0) & (dS[1:] > 0)).sum()) + 1
    jdf = max(L - jdf_offset, 0)
    return {"variant": name, "L": L, "eff_F": effF, "alpha_2sls": a2, "alpha_se": se2,
            "alpha_cue": float(grid[i0]), "cue_at_grid_edge": bool(i0 in (0, len(grid) - 1)),
            "S_min": float(S[i0]), "J_df": jdf,
            "J_p": (float(chi2.sf(S[i0], jdf)) if jdf > 0 else None),
            "crit_chi2_95": crit, "sset": sset, "n_local_minima": n_min,
            "omega_cond_at_min": float(ev.max() / max(ev.min(), 1e-300)),
            "omega_n_truncated": n_trunc,
            "_S_full": S,      # stripped before JSON; kept so callers can re-invert at other criticals
            "curve": {"grid": [float(g) for g in grid[::args.curve_thin]],
                      "S": [float(x) for x in S[::args.curve_thin]]},
            **wcb_extra}


def analyse(k, stage, args, dp, raw, cp):
    fs = sorted(glob.glob(str(dp / f"demand_{k}_*spec_12.parquet")), key=os.path.getmtime)
    if not fs:
        print(f"  E{k}: no demand parquet — skipped"); return None
    delta, err = _load_delta(cp, k, stage)
    if delta is None:
        print(f"  E{k}: {err}"); return None

    need = list(dict.fromkeys(["spread_ann", "deposit_type", "CodConglomeradoPrudencial"]
                              + X_COLS + IV_COLS))
    df = pd.read_parquet(fs[-1], columns=need)
    if len(df) != len(delta):
        print(f"  E{k}: delta length {len(delta):,} != parquet rows {len(df):,} — REFUSING to align")
        return None

    spread, x_mat, z_mat, dtype, X_full, X_hat = _build_matrices(df)
    codes, G = _cluster_codes(df["CodConglomeradoPrudencial"].astype(str).to_numpy())
    N = len(delta)

    # Row-alignment self-check: the engine's own 2SLS must reproduce before anything else is trusted.
    valid = np.all(np.isfinite(X_hat), axis=1) & np.isfinite(delta)
    th, *_ = np.linalg.lstsq(X_hat[valid], delta[valid], rcond=None)
    rj = raw / f"blp_results_E{k}_spec_12_{stage}.json"
    a_eng = None
    if rj.is_file():
        try:
            t1 = json.load(open(rj)).get("theta1") or []
            a_eng = float(t1[0]) if t1 else None
        except Exception:
            pass
    if a_eng is not None and abs(th[0] - a_eng) > ALPHA_TOL:
        print(f"  E{k}: *** REPLICATION MISMATCH {th[0]:+.6f} vs engine {a_eng:+.6f} — skipped ***")
        return None

    # Partialled 1-parameter problem (battery conventions: constant + X, uniform instrumenting).
    Xc = np.column_stack([np.ones(N), x_mat])
    keep = z_mat.std(axis=0) > 1e-10
    Zk, iv_kept = z_mat[:, keep], [c for c, kp in zip(IV_COLS, keep) if kp]
    d_t = _resid(delta.reshape(-1, 1), Xc).ravel()
    p_t = _resid(spread.reshape(-1, 1), Xc).ravel()
    Z_t = _resid(Zk, Xc)

    a, b, c = (float(x) for x in args.grid.split(":"))
    grid = np.arange(a, c + b / 2, b)

    # Shared WCB draws: ONE (B×G) matrix per routine, reused across the grid and every variant —
    # exactly the battery's sharing pattern (and the same rng(0) seed at the same G, so the draws
    # are bit-identical to the ones behind weak_iv*.json's ji_ci_*/ar_ci_* sets).
    Wb = (_wild_weights(args.wcb, G, args.wcb_scheme, np.random.default_rng(0))
          if args.wcb > 0 else None)

    # Effective clusters: computed from the actual sample (Carter–Schnepel–Steigerwald), --gstar
    # overrides for sensitivity.
    gstats = effective_cluster_stats(np.bincount(codes, minlength=G))
    gstar = args.gstar if args.gstar is not None else gstats["G_star"]
    crit_t = float(student_t.ppf(0.975, max(gstar - 1, 1)) ** 2)

    variants = [_variant("full16", Z_t, d_t, p_t, codes, G, grid, args, wcb=Wb)]

    # option 2a — principal components of the standardized partialled instruments
    sd = Z_t.std(axis=0)
    Zs = Z_t / np.where(sd > 0, sd, 1.0)
    U, sv, Vt = np.linalg.svd(Zs, full_matrices=False)
    var_share = (sv ** 2) / (sv ** 2).sum()
    for r in args.pcs:
        variants.append(_variant(f"pc{r}", Zs @ Vt[:r].T, d_t, p_t, codes, G, grid, args, wcb=Wb))
        variants[-1]["pc_var_share"] = float(var_share[:r].sum())

    # option 2b — the 5 named leave-one-out characteristics
    idx5 = [iv_kept.index(cn) for cn in LOO5 if cn in iv_kept]
    if len(idx5) == len(LOO5):
        variants.append(_variant("loo5", Z_t[:, idx5], d_t, p_t, codes, G, grid, args, wcb=Wb))

    # option 1 — just-identified optimal IV: z* = fitted first stage on all 16
    zstar = (Z_t @ np.linalg.lstsq(Z_t, p_t, rcond=None)[0]).reshape(-1, 1)
    ji = _variant("ji_opt", zstar, d_t, p_t, codes, G, grid, args, jdf_offset=1, wcb=Wb)
    variants.append(ji)

    # option 1, engine-grade — z* = the engine's own generated instrument. The engine's θ₁ step
    # (OLS of δ on X_hat = [spread_hat, x_mat], NO constant) IS just-identified IV with excluded
    # instrument spread_hat: the per-type projection makes (spread − spread_hat) ⊥ {spread_hat,
    # x_mat} block-by-block, so by FWL the α below must equal the engine's reported θ₁[1]. The
    # inverted set is therefore the identification-robust CI for the PAPER'S OWN estimator. On
    # types 1–2 rows z* is the raw spread itself — the engine's exogeneity assumption made explicit.
    m_ok = valid if not valid.all() else slice(None)
    Xe = x_mat[m_ok]
    d_e = _resid(delta[m_ok].reshape(-1, 1), Xe).ravel()
    p_e = _resid(spread[m_ok].reshape(-1, 1), Xe).ravel()
    z_e = _resid(X_hat[m_ok, 0].reshape(-1, 1), Xe)
    codes_e, G_e = (codes, G) if valid.all() else _cluster_codes(
        df["CodConglomeradoPrudencial"].astype(str).to_numpy()[valid])
    Wb_e = Wb if (Wb is None or G_e == G) else _wild_weights(
        args.wcb, G_e, args.wcb_scheme, np.random.default_rng(0))
    ji_e = _variant("ji_engine", z_e, d_e, p_e, codes_e, G_e, grid, args, jdf_offset=1, wcb=Wb_e)
    if a_eng is not None and abs(ji_e["alpha_2sls"] - a_eng) > ALPHA_TOL:
        raise AssertionError(
            f"E{k}: ji_engine alpha {ji_e['alpha_2sls']:+.6f} != engine {a_eng:+.6f} — the FWL "
            f"identity failed, so the partialling/no-constant convention is wrong; do not trust "
            f"any ji_engine output")
    ji_e["alpha_matches_engine"] = bool(a_eng is not None)
    variants.append(ji_e)

    # option 1, cross-fit — z* fitted on the complement cluster fold, purging the in-sample
    # (generated-regressor) optimism of ji_opt. Partialling stays global (X exogenous,
    # low-dimensional — standard cross-fitting practice); cluster-level WCB weights ignore the
    # cross-fold dependence through π̂ (the standard 2-fold argument).
    sizes = np.bincount(codes, minlength=G)
    fold = _fold_assign(sizes, args.split_top, np.random.default_rng(args.split_seed))
    fr = fold[codes]
    zcf = np.empty(N)
    fold_meta = {"seed": args.split_seed, "top": args.split_top,
                 "row_share": [float(sizes[fold == f].sum() / N) for f in (0, 1)],
                 "n_clusters": [int((fold == f).sum()) for f in (0, 1)], "eff_F_fold": []}
    for f in (0, 1):
        tr, te = fr != f, fr == f
        pi_f, *_ = np.linalg.lstsq(Z_t[tr], p_t[tr], rcond=None)
        zcf[te] = Z_t[te] @ pi_f
        codes_f, G_f = _cluster_codes(df["CodConglomeradoPrudencial"].astype(str).to_numpy()[tr])
        fold_meta["eff_F_fold"].append(float(_eff_F(p_t[tr], Z_t[tr], codes_f, G_f)))
    ji_s = _variant("ji_split", zcf.reshape(-1, 1), d_t, p_t, codes, G, grid, args,
                    jdf_offset=1, wcb=Wb)
    ji_s["fold"] = fold_meta
    variants.append(ji_s)

    # Few-cluster t(G*-1)^2 reading for all three JI variants (needs _S_full, so before the strip).
    for v in (ji, ji_e, ji_s):
        v["crit_t_gstar_sq"] = crit_t
        v["sset_t_gstar"] = _set_from_curve(grid, v["_S_full"], crit_t)
    for v in variants:
        del v["_S_full"]

    print(f"\n  ── E{k} / {stage} ── N={N:,}  G={G}  G*={gstats['G_star']:.2f}  grid={args.grid}"
          + (f"  wcb={args.wcb}x{args.wcb_scheme}" if Wb is not None else "  wcb=off")
          + ("" if keep.all() else f"  (dropped: {','.join(set(IV_COLS)-set(iv_kept))})"))
    hdr = f"     {'variant':9s} {'L':>2s} {'eff-F':>6s} {'2SLS a (SE)':>16s} {'a_cue':>7s} " \
          f"{'S_min':>7s} {'J(p)':>6s} {'cond(Om)':>9s}  95% S-set (chi2)     WCB set"
    print(hdr)
    for v in variants:
        jp = f"{v['J_p']:.2f}" if v.get("J_p") is not None else "  --"
        print(f"     {v['variant']:9s} {v['L']:2d} {v['eff_F']:6.2f} "
              f"{v['alpha_2sls']:+8.3f} ({v['alpha_se']:.3f}) {v['alpha_cue']:+7.3f} "
              f"{v['S_min']:7.2f} {jp:>6s} {v['omega_cond_at_min']:9.2e}  "
              f"{_fmt_set(v['sset']):20s} "
              + (_fmt_set(v["sset_wcb"]) if "sset_wcb" in v else "--"))
    if "fold" in ji_s:
        fm = ji_s["fold"]
        print(f"     ji_split folds: shares {fm['row_share'][0]:.2f}/{fm['row_share'][1]:.2f}, "
              f"G {fm['n_clusters'][0]}/{fm['n_clusters'][1]}, "
              f"per-fold first-stage eff-F {fm['eff_F_fold'][0]:.2f}/{fm['eff_F_fold'][1]:.2f}")
    for v in (ji, ji_e, ji_s):
        print(f"     {v['variant']:9s} t(G*-1)^2 reading: {_fmt_set(v['sset_t_gstar'])}")
    return {"routine": k, "stage": stage, "n_obs": N, "n_clusters": G,
            "grid": args.grid, "gstar_used": float(gstar),
            "gstar_computed": float(gstats["G_star"]),
            "wcb_reps": (int(args.wcb) if Wb is not None else 0), "wcb_scheme": args.wcb_scheme,
            "pc_var_share_first5": [float(x) for x in var_share[:5]],
            "variants": variants}


def main():
    ap = argparse.ArgumentParser(description="Moment-reduction fixes (weakiv_methods.md §7.6) — local test")
    ap.add_argument("--routines", default="5,6,7,8")
    ap.add_argument("--stage", default="ext1")
    ap.add_argument("--grid", default="-2:0.005:2", help="start:step:stop for the alpha scan "
                    "(use --grid=-2:... — a leading '-' needs the '=' form)")
    ap.add_argument("--pcs", default="2,3,4,5", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--pinv-rtol", type=float, default=1e-10, dest="pinv_rtol")
    ap.add_argument("--gstar", type=float, default=None,
                    help="override the computed Carter-Schnepel-Steigerwald G* for the t(G*-1)^2 "
                         "critical (default: computed from the actual cluster sizes)")
    ap.add_argument("--wcb", type=int, default=int(os.environ.get("BLP_WCB_REPS", "999")),
                    help="wild-cluster-bootstrap draws for the WCB criticals (0 disables); "
                         "conventions mirror weak_iv_analysis._battery exactly")
    ap.add_argument("--wcb-scheme", default=os.environ.get("BLP_WCB_SCHEME", "webb"),
                    choices=["webb", "rademacher"], dest="wcb_scheme")
    ap.add_argument("--split-seed", type=int, default=0, dest="split_seed",
                    help="seed for the small-cluster shuffle in the ji_split fold assignment")
    ap.add_argument("--split-top", type=int, default=10, dest="split_top",
                    help="how many largest clusters get deterministic greedy fold balancing")
    ap.add_argument("--curve-thin", type=int, default=8, dest="curve_thin",
                    help="store every k-th grid point of each S-curve in the JSON")
    args = ap.parse_args()

    dp, raw, cp = _dirs()
    routines = [int(x) for x in args.routines.split(",") if x.strip()]
    print(f"Moment-reduction diagnostic | stage={args.stage} | routines={routines} | pcs={args.pcs}")

    out = [r for r in (analyse(k, args.stage, args, dp, raw, cp) for k in routines) if r]
    if not out:
        print("\nnothing computed."); return
    path = cp / "diag_moment_reduction.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
