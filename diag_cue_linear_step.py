"""
diag_cue_linear_step.py
================================================================================
LOCAL preview + self-check of the CUE (continuously-updated GMM) linear step, at FIXED θ₂ — no GPU,
no Julia, no cluster time.

Why: the cluster CUE run re-optimises θ₂ against a continuously-updated weight, which is a multi-hour
GPU job. Almost all of the interesting behaviour, though, lives in the *linear step* — how much α
moves when W = (Z'Z/N)⁻¹ is replaced by W(θ) = pinv(Ω̂(θ)) — and that can be evaluated exactly at the
already-estimated θ̂₂ using the exported δ(θ̂₂). So this script:

  1. Replicates the engine's linear step EXACTLY (`build_regressor_matrices` +
     `project_endogenous_spreads` + `estimate_theta1`, blp_1_estimation.jl) and ASSERTS that the
     resulting α matches the engine's reported θ₁[1] to ~5 decimals. That single assertion validates
     row alignment, instrument order, the per-type spread projection and W₀ in one shot — if it
     passes, the replication is trustworthy.
  2. Runs the same CUE fixed point as `cue_fixed_point` (blp_1_estimation.jl §7b) and reports α_cue,
     the iteration trace, Q₀ vs Q_cue, and the Ω̂ spectrum (which decides whether the pinv rtol / ridge
     defaults need tuning BEFORE cluster time).
  3. `--sset-grid`: a fixed-θ₂ Stock–Wright S-curve over α₀. NOT the true S-set (that re-optimises θ₂
     at every α₀ — the cluster job), but it previews the curve's shape and calibrates SSET_GRID.

Inputs (both already on disk):
  DEMAND_PREP/demand_{k}_*spec_12.parquet             the estimation sample
  cluster_processed/rc_delta_E{k}_spec_12_{stage}.bin δ(θ̂₂), written by export_rc_delta.jl
  cluster_raw/blp_results_E{k}_spec_12_{stage}.json   the engine's own θ₁/Q, for the assertions

Usage
-----
  python diag_cue_linear_step.py                              # E3-E4, stage ext1
  python diag_cue_linear_step.py --routines 4 --sset-grid -1.5:0.125:1.5
  python diag_cue_linear_step.py --max-witer 25 --ridge 1e-8  # knob sweep
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

from utils import paths

# ── Column sets: must match the ENGINE, not weak_iv_analysis.py ────────────────────────────────
# blp_1_estimation.jl:800 builds Z as vcat(IV_BLP_LOO, IV_ESTBAN, IV_COST, IV_CAPITAL) — note ESTBAN
# sits 12th, BEFORE the cost block. weak_iv_analysis.py lists it last; using that order here would
# silently permute Z's columns (harmless for 2SLS, but it would scramble the per-instrument reporting).
X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
          "log_total_assets_lag", "is_state_owned"]
IV_BLP_LOO = ["loo_log_assets", "mean_loo_log_assets",
              "loo_equity_ratio", "mean_loo_equity_ratio",
              "loo_basileia", "mean_loo_basileia",
              "loo_credit_assets", "mean_loo_credit_assets",
              "loo_npl_provision", "mean_loo_npl_provision",
              "n_rivals"]
IV_ESTBAN  = ["estban_rival_branches_lag"]
IV_COST    = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag", "tax_cost_ratio_lag"]
IV_CAPITAL = ["indice_basileia_lag"]
IV_COLS    = IV_BLP_LOO + IV_ESTBAN + IV_COST + IV_CAPITAL   # engine order

ENDOG_TYPES = (4, 5)          # the engine projects the spread for these types only
ALPHA_TOL   = 5e-6            # engine-vs-replication tolerance on α


# ── Paths ──────────────────────────────────────────────────────────────────────────────────────
def _dirs():
    est = paths.PROCESSED / "ESTIMATION_OUTPUT"
    return est / "DEMAND_PREP", est / "BLP_RESULTS" / "cluster_raw", est / "BLP_RESULTS" / "cluster_processed"


def _load_delta(cp_dir, k, stage):
    """δ(θ̂₂) as written by export_rc_delta.jl: Int64 n, then n Float64 (little-endian)."""
    p = cp_dir / f"rc_delta_E{k}_spec_12_{stage}.bin"
    if not p.is_file():
        return None, f"{p.name} not found — run: julia --project=. export_rc_delta.jl --stage {stage}"
    with open(p, "rb") as f:
        n = int(np.frombuffer(f.read(8), dtype="<i8")[0])
        d = np.frombuffer(f.read(n * 8), dtype="<f8")
    if len(d) != n:
        return None, f"{p.name} truncated ({len(d)} of {n})"
    return d.astype(float).copy(), None


# ── Engine replication ─────────────────────────────────────────────────────────────────────────
def _build_matrices(df):
    """Mirror of build_regressor_matrices + project_endogenous_spreads (blp_1_estimation.jl:774-831).
    Missing values coalesce to 0.0 and ±Inf to 0.0, exactly as the engine does."""
    spread = np.nan_to_num(df["spread_ann"].to_numpy(float), nan=0.0) / 100.0
    x_mat = np.column_stack([np.nan_to_num(df[c].to_numpy(float), nan=0.0) if c in df.columns
                             else np.zeros(len(df)) for c in X_COLS])
    z_mat = np.column_stack([np.nan_to_num(df[c].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
                             if c in df.columns else np.zeros(len(df)) for c in IV_COLS])
    dtype = df["deposit_type"].fillna(0).astype(int).to_numpy()

    H = np.column_stack([x_mat, z_mat])
    spread_hat = spread.copy()
    for kv in ENDOG_TYPES:                       # per-type first stage; types 1/2 keep the raw spread
        m = dtype == kv
        if m.sum() > H.shape[1]:
            ok = np.all(np.isfinite(H[m]), 1) & np.isfinite(spread[m])
            if ok.sum() > H.shape[1]:
                Hk, sk = H[m], spread[m]
                beta, *_ = np.linalg.lstsq(Hk[ok], sk[ok], rcond=None)
                fit = sk.copy(); fit[ok] = Hk[ok] @ beta
                spread_hat[m] = fit
    X_full = np.column_stack([spread, x_mat])
    X_hat  = np.column_stack([spread_hat, x_mat])
    return spread, x_mat, z_mat, dtype, X_full, X_hat


def _cluster_codes(cl):
    _, codes = np.unique(cl, return_inverse=True)
    return codes, int(codes.max()) + 1


def _omega(xi, Z, codes, G, ridge=0.0):
    """Clustered Ω̂ = (1/N) Σ_g (Z_g'ξ_g)(Z_g'ξ_g)' — mirrors cue_weight().

    Uses one np.bincount per instrument column rather than np.add.at: the latter is unbuffered and
    takes ~1 s here (755k×16 scattered adds), which made a direct minimisation over θ₁ — thousands of
    criterion evaluations — effectively non-terminating. bincount is the same arithmetic in C.
    """
    L = Z.shape[1]
    M = np.empty((G, L))
    ZX = Z * xi[:, None]
    for l in range(L):
        M[:, l] = np.bincount(codes, weights=ZX[:, l], minlength=G)
    Om = (M.T @ M) / len(xi)
    if ridge > 0:
        Om = Om + (ridge * np.trace(Om) / L) * np.eye(L)
    return Om


def _pinv_sym(Om, rtol):
    return np.linalg.pinv((Om + Om.T) / 2.0, rcond=rtol, hermitian=True)


def _theta1_gmm(delta, X, Z, valid, W):
    """θ₁(W) = (X'Z W Z'X)⁻¹ X'Z W Z'δ — mirrors estimate_theta1_gmm()."""
    Zv, Xv = Z[valid], X[valid]
    ZtX, Ztd = Zv.T @ Xv, Zv.T @ delta[valid]
    A, b = ZtX.T @ W @ ZtX, ZtX.T @ W @ Ztd
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def _cluster_moment_parts(target, design, Z, codes, G):
    """Precompute the per-cluster moment as an AFFINE function of θ₁ — the key to making a direct
    minimisation of the CU criterion cheap.

        M_g(θ₁) = Z_g'ξ_g = Z_g'(t_g − X_g θ₁) = a_g − B_g θ₁

    `a` is G×L and `B` is G×L×K, both independent of θ₁. Every subsequent criterion evaluation is then
    O(G·L·K) ≈ 44k flops instead of O(N·L) ≈ 12M — a ~5-order-of-magnitude speedup that turns the
    minimisation from hours into milliseconds. (In the engine, `B` is fixed for the whole run and only
    `a` needs recomputing when δ changes with θ₂.)
    """
    L, K = Z.shape[1], design.shape[1]
    a = np.empty((G, L))
    B = np.empty((G, L, K))
    for l in range(L):
        a[:, l] = np.bincount(codes, weights=Z[:, l] * target, minlength=G)
        for k in range(K):
            B[:, l, k] = np.bincount(codes, weights=Z[:, l] * design[:, k], minlength=G)
    return a, B


def _cu_criterion_fast(theta1, a, B, N, args):
    """Q(θ₁) = N·ḡ' pinv(Ω̂) ḡ from the precomputed affine parts — moment AND weight at the same θ₁."""
    M = a - B @ theta1                      # (G,L)
    Om = (M.T @ M) / N
    if args.ridge > 0:
        Om = Om + (args.ridge * np.trace(Om) / Om.shape[0]) * np.eye(Om.shape[0])
    if not np.all(np.isfinite(Om)):
        return 1e12
    W = _pinv_sym(Om, args.pinv_rtol)
    g = M.sum(axis=0) / N
    return float(N * g @ W @ g)


def _cu_criterion(theta1, target, design, Z, codes, G, args):
    """Reference (slow) form of the CU criterion, kept as the correctness anchor for the fast one."""
    xi = target - design @ theta1
    if not np.all(np.isfinite(xi)):
        return 1e12
    W = _pinv_sym(_omega(xi, Z, codes, G, args.ridge), args.pinv_rtol)
    g = (Z * xi[:, None]).mean(axis=0)
    return float(len(xi) * g @ W @ g)


def _cue_minimize(delta, X_full, X_hat, Z, valid, codes, G, args, alpha0=None):
    """TRUE CUE: minimise the CU criterion over θ₁ directly (Hansen-Heaton-Yaron 1996).

    The alternating "update W, re-solve θ₁" scheme is ITERATED GMM, a different estimator that is
    known to cycle — and does here (see --method fixedpoint). CUE is defined as a joint minimiser, so
    we minimise. Started from the production 2SLS θ₁ (the natural, engine-consistent start).
    """
    from scipy.optimize import minimize
    if alpha0 is None:
        th0, *_ = np.linalg.lstsq(X_hat[valid], delta[valid], rcond=None)
        design, target = X_full, delta
    else:
        design = X_full[:, 1:]
        target = delta - alpha0 * X_full[:, 0]
        th0, *_ = np.linalg.lstsq(design[valid], target[valid], rcond=None)

    N = len(target)
    a, B = _cluster_moment_parts(target, design, Z, codes, G)
    f = lambda t: _cu_criterion_fast(t, a, B, N, args)
    # Sanity: the fast affine form must agree with the reference O(N·L) form at the start point.
    q_fast, q_ref = f(th0), _cu_criterion(th0, target, design, Z, codes, G, args)
    if not np.isclose(q_fast, q_ref, rtol=1e-8, atol=1e-10):
        raise AssertionError(f"fast/reference CU criterion disagree: {q_fast:.10g} vs {q_ref:.10g}")

    r = minimize(f, th0, method="Nelder-Mead",
                 options={"maxiter": args.min_maxiter, "maxfev": args.min_maxiter * 4,
                          "xatol": 1e-10, "fatol": 1e-12, "adaptive": True})
    # Polish with BFGS (FD gradient) — Nelder-Mead is robust in 9-D but slow to high precision.
    r2 = minimize(f, r.x, method="BFGS", options={"maxiter": 500, "gtol": 1e-10})
    th = r2.x if r2.fun <= r.fun else r.x
    Qs = float(min(r.fun, r2.fun))
    xi = target - design @ th
    W  = _pinv_sym(_omega(xi, Z, codes, G, args.ridge), args.pinv_rtol)
    return {"theta1": th, "xi": xi, "W": W, "Q": Qs,
            "iters": int(r.nit + r2.nit), "converged": bool(r.success or r2.success),
            "trace": [float(th0[0]), float(th[0])] if alpha0 is None else [],
            "Q_at_2sls": float(q_fast)}


def _cue_fixed_point(delta, X_full, X_hat, Z, valid, codes, G, args, alpha0=None):
    """Mirrors cue_fixed_point / cue_fixed_point_alpha0. Returns a dict of the trace + results."""
    N = len(delta)
    if alpha0 is None:
        th, *_ = np.linalg.lstsq(X_hat[valid], delta[valid], rcond=None)   # iter 0 = production 2SLS
        xi = delta - X_full @ th
        trace = [th[0]]
        design, target = X_full, delta
    else:
        Xt   = X_full[:, 1:]                                              # β only; α pinned
        dtil = delta - alpha0 * X_full[:, 0]
        th, *_ = np.linalg.lstsq(Xt[valid], dtil[valid], rcond=None)
        xi = dtil - Xt @ th
        trace = []
        design, target = Xt, dtil

    iters, converged = 0, False
    for it in range(1, args.max_witer + 1):
        W = _pinv_sym(_omega(xi, Z, codes, G, args.ridge), args.pinv_rtol)
        tn = _theta1_gmm(target, design, Z, valid, W)
        if not np.all(np.isfinite(tn)):
            break
        drel = np.max(np.abs(tn - th)) / max(1.0, np.max(np.abs(th)))
        th, xi, iters = tn, target - design @ tn, it
        if alpha0 is None:
            trace.append(th[0])
        if drel < args.wtol:
            converged = True
            break

    Wf = _pinv_sym(_omega(xi, Z, codes, G, args.ridge), args.pinv_rtol)
    g  = (Z * xi[:, None]).mean(axis=0)
    return {"theta1": th, "xi": xi, "W": Wf, "Q": float(N * g @ Wf @ g),
            "iters": iters, "converged": bool(converged), "trace": trace}


def analyse(k, stage, args, dp, raw, cp):
    fs = sorted(glob.glob(str(dp / f"demand_{k}_*spec_12.parquet")), key=os.path.getmtime)
    if not fs:
        print(f"  E{k}: no demand parquet — skipped"); return None
    delta, err = _load_delta(cp, k, stage)
    if delta is None:
        print(f"  E{k}: {err}"); return None

    need = list(dict.fromkeys(["spread_ann", "deposit_type", "CodConglomeradoPrudencial"]
                              + X_COLS + IV_COLS))
    df = pd.read_parquet(fs[-1], columns=[c for c in need])
    if len(df) != len(delta):
        print(f"  E{k}: delta length {len(delta):,} != parquet rows {len(df):,} — REFUSING to align")
        return None

    spread, x_mat, z_mat, dtype, X_full, X_hat = _build_matrices(df)
    clus_all = df["CodConglomeradoPrudencial"].astype(str).to_numpy()

    # ── Optional deposit-type block (--types) ──────────────────────────────────────────────────
    # Default `all` = the engine's own estimation sample, which is what θ₁ is fitted on and the only
    # block where the self-check below can hold. The blocks exist to test whether Ω̂'s rank deficiency
    # is a property of the weak instrumented products (4+5) or of the clustering geometry generally:
    # G* ≈ 5 is a feature of the conglomerate concentration, not of any product, so if the deficiency
    # is general it should persist in 1+2 where the first stage is ~3x stronger.
    # NOTE the spread projection inside _build_matrices already ran per type on the FULL sample, exactly
    # as the engine does; subsetting afterwards keeps each row's X_hat identical to the engine's.
    _TYPE_SETS = {"all": None, "12": [1, 2], "45": [4, 5], "4": [4], "5": [5]}
    tsel = getattr(args, "types", "all")
    if tsel not in _TYPE_SETS:
        print(f"  E{k}: unknown --types {tsel!r} (all|12|45|4|5)"); return None
    if _TYPE_SETS[tsel] is not None:
        m = np.isin(dtype, _TYPE_SETS[tsel])
        if m.sum() < 100:
            print(f"  E{k}/{tsel}: only {m.sum()} rows — skipped"); return None
        delta, spread, x_mat, z_mat = delta[m], spread[m], x_mat[m], z_mat[m]
        X_full, X_hat, dtype, clus_all = X_full[m], X_hat[m], dtype[m], clus_all[m]

    # The engine drops zero-variance instruments before building Z (blp_gpu_engine.jl:1568-1574).
    keep = z_mat.std(axis=0) > 1e-10
    Z, iv_kept = z_mat[:, keep], [c for c, kp in zip(IV_COLS, keep) if kp]
    valid = np.all(np.isfinite(X_hat), axis=1) & np.isfinite(delta)
    codes, G = _cluster_codes(clus_all)      # recomputed on the block: G must not count empty clusters
    N = len(delta)

    # ── Self-check against the engine's own reported θ₁/Q ──────────────────────────────────────
    th_2sls, *_ = np.linalg.lstsq(X_hat[valid], delta[valid], rcond=None)
    xi0 = delta - X_full @ th_2sls
    W0  = np.linalg.inv(Z.T @ Z / N)
    g0  = (Z * xi0[:, None]).mean(axis=0)
    Q0  = float(g0 @ W0 @ g0)

    rj = raw / f"blp_results_E{k}_spec_12_{stage}.json"
    a_eng = q_eng = None
    if rj.is_file():
        try:
            j = json.load(open(rj))
            t1 = j.get("theta1") or []
            a_eng = float(t1[0]) if t1 else None
            q_eng = j.get("Q_value")
        except Exception as e:
            print(f"  E{k}: could not read {rj.name} ({e})")

    # The engine fits θ₁ on the FULL sample, so its α is only reproducible when tsel == "all".
    # On a block the comparison is meaningless and must not be reported as a mismatch.
    if tsel != "all":
        a_eng = q_eng = None
    ok_alpha = a_eng is not None and abs(th_2sls[0] - a_eng) < ALPHA_TOL
    if a_eng is not None and not ok_alpha:
        print(f"  E{k}: *** REPLICATION MISMATCH: alpha_2sls={th_2sls[0]:+.6f} vs engine "
              f"{a_eng:+.6f} (|d|={abs(th_2sls[0]-a_eng):.2e} > {ALPHA_TOL:.0e}) ***")

    # α under the GMM formula at W₀ — isolates the formula switch from the re-weighting
    a_gmm_w0 = _theta1_gmm(delta, X_full, Z, valid, W0)[0]

    solver = _cue_minimize if args.method == "minimize" else _cue_fixed_point
    res = solver(delta, X_full, X_hat, Z, valid, codes, G, args)
    ev  = np.linalg.eigvalsh((_omega(res["xi"], Z, codes, G, args.ridge)
                              + _omega(res["xi"], Z, codes, G, args.ridge).T) / 2)
    n_trunc = int((ev < args.pinv_rtol * ev.max()).sum())

    print(f"\n  ── E{k} / {stage} ── N={N:,}  G={G}  L={Z.shape[1]}"
          f"{'' if Z.shape[1] == len(IV_COLS) else ' (dropped: ' + ','.join(set(IV_COLS)-set(iv_kept)) + ')'}")
    print(f"     alpha_2sls (engine repl.) : {th_2sls[0]:+.6f}"
          + (f"   engine={a_eng:+.6f}  {'OK' if ok_alpha else 'MISMATCH'}" if a_eng is not None else ""))
    print(f"     alpha_gmm(W0)             : {a_gmm_w0:+.6f}   (formula switch only, no re-weighting)")
    print(f"     alpha_cue                 : {res['theta1'][0]:+.6f}   "
          f"[{res['iters']} iters, converged={res['converged']}]")
    if len(res["trace"]) > 1:
        print(f"     alpha trace               : " + " → ".join(f"{a:+.4f}" for a in res["trace"][:8])
              + (" …" if len(res["trace"]) > 8 else ""))
    print(f"     Q0 (fixed W, engine units): {Q0:.6g}"
          + (f"   engine Q={q_eng:.6g}" if isinstance(q_eng, (int, float)) else ""))
    print(f"     Q_cue (N-scaled, Hansen J): {res['Q']:.6g}   df_conservative={Z.shape[1]}")
    print(f"     Omega spectrum            : min={ev.min():.3e} max={ev.max():.3e} "
          f"cond={ev.max()/max(ev.min(), 1e-300):.3e}  truncated_by_pinv={n_trunc}")

    out = {"routine": k, "stage": stage, "types": tsel,
           "n_obs": N, "n_clusters": G, "n_iv": int(Z.shape[1]),
           "alpha_2sls": float(th_2sls[0]), "alpha_engine": a_eng,
           "alpha_2sls_matches_engine": bool(ok_alpha),
           "alpha_gmm_W0": float(a_gmm_w0), "alpha_cue": float(res["theta1"][0]),
           "alpha_trace": [float(a) for a in res["trace"]],
           "cue_iters": res["iters"], "cue_converged": res["converged"],
           "Q0_fixed_W": Q0, "Q_engine": q_eng, "Q_cue_Nscaled": res["Q"],
           "omega_eig_min": float(ev.min()), "omega_eig_max": float(ev.max()),
           "omega_n_truncated": n_trunc,
           "knobs": {"max_witer": args.max_witer, "wtol": args.wtol,
                     "pinv_rtol": args.pinv_rtol, "ridge": args.ridge}}

    # ── Optional fixed-θ₂ S-curve (grid calibration only; the true set re-optimises θ₂) ────────
    if args.sset_grid:
        a, b, c = (float(x) for x in args.sset_grid.split(":"))
        grid = np.arange(a, c + b / 2, b)
        # Hoist the per-cluster moment parts out of the grid loop. Only the TARGET moves with α₀, and
        # it moves affinely:  a(α₀) = Z_g'(δ − α₀·p) = a_δ − α₀·a_p.  The design (β columns) and hence
        # B are identical at every grid point. Recomputing them per point costs ~3 s × |grid| and was
        # what made the first S-curve attempt appear to hang.
        from scipy.optimize import minimize
        Xt = X_full[:, 1:]
        a_del, Bm = _cluster_moment_parts(delta, Xt, Z, codes, G)
        a_spr, _  = _cluster_moment_parts(X_full[:, 0], Xt, Z, codes, G)
        b0, *_ = np.linalg.lstsq(Xt[valid], delta[valid], rcond=None)
        N = len(delta)
        pts = []
        for a0 in grid:
            av  = a_del - float(a0) * a_spr
            fun = lambda bb: _cu_criterion_fast(bb, av, Bm, N, args)
            r1  = minimize(fun, b0, method="Nelder-Mead",
                           options={"maxiter": args.min_maxiter, "maxfev": args.min_maxiter * 2,
                                    "xatol": 1e-9, "fatol": 1e-11, "adaptive": True})
            r2  = minimize(fun, r1.x, method="BFGS", options={"maxiter": 300, "gtol": 1e-9})
            pts.append({"alpha0": float(a0), "S": float(min(r1.fun, r2.fun)),
                        "iters": int(r1.nit + r2.nit),
                        "converged": bool(r1.success or r2.success)})
        S = np.array([p["S"] for p in pts])
        try:
            from scipy.stats import chi2
            crit = float(chi2.ppf(0.95, Z.shape[1]))
        except Exception:
            crit = float("nan")
        inside = [p["alpha0"] for p, s in zip(pts, S) if s <= crit]
        print(f"     S-curve (FIXED theta2)    : min S={S.min():.4g} at "
              f"alpha0={pts[int(S.argmin())]['alpha0']:+.3f} | chi2_95({Z.shape[1]})={crit:.3g}")
        print(f"       -> fixed-theta2 set     : "
              + (f"[{min(inside):+.3f}, {max(inside):+.3f}]"
                 + ("  (touches a grid edge — widen)" if (min(inside) <= a or max(inside) >= c) else "")
                 if inside else "EMPTY at this grid/crit")
              + "   NOTE: previews shape only; the true S-set re-optimises theta2 per point.")
        out["sset_fixed_theta2"] = {"grid": args.sset_grid, "crit_chi2_95": crit, "points": pts}
    return out


def main():
    ap = argparse.ArgumentParser(description="Local CUE linear-step preview + engine self-check")
    # The single-index pair: the routines with an RC-BLP delta export (blp_2_rc.jl DEFAULT_ROUTINES).
    ap.add_argument("--routines", default="3,4")
    ap.add_argument("--stage", default="ext1")
    ap.add_argument("--max-witer", type=int, default=10, dest="max_witer")
    ap.add_argument("--wtol", type=float, default=1e-10)
    ap.add_argument("--pinv-rtol", type=float, default=1e-10, dest="pinv_rtol")
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--types", choices=["all", "12", "45", "4", "5"], default="all",
                    help="deposit-type block. 'all' (default) is the engine's estimation sample and the "
                         "only one where the alpha self-check applies; '12' is the un-instrumented "
                         "regulated block, '45' the instrumented one. Use to test whether Omega's rank "
                         "deficiency is block-specific or a general consequence of G*~5.")
    ap.add_argument("--method", choices=["minimize", "fixedpoint"], default="minimize",
                    help="minimize = TRUE CUE (joint minimiser of the CU criterion, Hansen-Heaton-"
                         "Yaron 1996); fixedpoint = iterated GMM (alternate W and theta1) — a "
                         "DIFFERENT estimator that cycles here, kept for diagnosis.")
    ap.add_argument("--min-maxiter", type=int, default=4000, dest="min_maxiter")
    ap.add_argument("--sset-grid", default=None, dest="sset_grid",
                    help="start:step:stop — fixed-theta2 S-curve, for calibrating SSET_GRID")
    args = ap.parse_args()

    dp, raw, cp = _dirs()
    routines = [int(x) for x in args.routines.split(",") if x.strip()]
    print(f"CUE linear-step diagnostic | stage={args.stage} | routines={routines}")
    print(f"method={args.method} | knobs: max_witer={args.max_witer} wtol={args.wtol} "
          f"pinv_rtol={args.pinv_rtol} ridge={args.ridge}")

    out = [r for r in (analyse(k, args.stage, args, dp, raw, cp) for k in routines) if r]
    if not out:
        print("\nnothing computed."); return

    bad = [r["routine"] for r in out if r["alpha_engine"] is not None
           and not r["alpha_2sls_matches_engine"]]
    print("\n" + "=" * 78)
    print(f"{'':4s} {'alpha_2sls':>11s} {'alpha_gmm(W0)':>14s} {'alpha_cue':>11s} {'iters':>6s} {'conv':>6s}")
    for r in out:
        print(f"E{r['routine']}   {r['alpha_2sls']:+11.4f} {r['alpha_gmm_W0']:+14.4f} "
              f"{r['alpha_cue']:+11.4f} {r['cue_iters']:6d} {str(r['cue_converged']):>6s}")
    print("=" * 78)
    if getattr(args, "types", "all") != "all":
        print(f"NOTE: --types {args.types} is a BLOCK; the engine fits theta1 on the full sample, so "
              "the alpha self-check is not applicable and was skipped.")
    print("SELF-CHECK: " + ("all alpha_2sls match the engine ✓" if not bad
                            else f"*** MISMATCH for routines {bad} — replication is NOT trustworthy ***"))

    path = cp / "diag_cue_linear_step.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
