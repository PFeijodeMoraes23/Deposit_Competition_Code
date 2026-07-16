"""
estimation_1_cost_3_solve.py
============================
CF2 — BBL Step 2, part 2: recover marginal-cost parameters (ω, ζ, γ)^κ for
κ ∈ {B, D} by minimizing the sum of squared FOC-inequality violations, using the
firm-level ψ basis produced by cost_2_fwd_sim.jl.

Moment inequality (per firm j, deviation σ̃) — the unnumbered display above
V_Main \\label{eq:17}:

    g_j(σ̂, σ̃) = (ψ_eq,j − ψ_dev,j,σ̃)' · [ 1 , −ω , −γ' , −(1+ζ) ]  ≥ 0

with the ψ block layout written by cost_2_fwd_sim.jl:
    [ psi1 ,  psi2_omega ,  psi3_gamma_<z>… ,  psi4_zeta ]
so, writing Δ = ψ_eq − ψ_dev,

    g = Δ·psi1  −  ω·Δ·psi2_omega  −  Σ_z γ_z·Δ·psi3_gamma_z  −  (1+ζ)·Δ·psi4_zeta.

Deviations should make the bank worse off, so g ≥ 0 in equilibrium; we minimize

    F^κ(ω, ζ, γ) = Σ_{j∈κ, σ̃}  min{ g_j , 0 }²                      (V_Main eq:17)

separately for B- and D-type firms. χ (private cost shock) is dropped, as in the draft.

Inputs (from cost_2_fwd_sim.jl):
    COST_FWD/psi_eq_{tag}.parquet   — firm, is_B, psi1, psi2_omega, psi3_gamma_*, psi4_zeta
    COST_FWD/psi_dev_{tag}.parquet  — shock, firm, is_B, <same blocks>
where tag = E{estim}_spec_{spec}_{stage}{suffix}.

Outputs:
    COST_FWD/cost_params_{tag}.json — (ω, ζ, γ) per type with bootstrap SEs, plus
    optimizer-health diagnostics and (with --profile) 1-D identified-set profiles.

⚠ Run only after cost_2_fwd_sim.jl has produced its parquets (which itself is gated
on the cluster data download and the author's go-ahead).

Usage:
  python estimation_1_cost_3_solve.py --estim 6 --spec 12 --stage extended \\
      --bootstrap 200 --profile
"""
try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except ModuleNotFoundError:
    pass  # utils/ is a local-dev convenience (re-exec into the Windows .venv); absent + a no-op on the cluster

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parents[2]
# COST_FWD holds the ψ parquets (from cost_2_fwd_sim) + the cost_params json. Default is the local
# processed-data layout; on the cluster the Julia writes them to data/COST_FWD, so set CF_COST_FWD
# to that path (submit_cf.sh does this) — the script's own dir doesn't contain the BCB tree there.
COST_FWD = Path(os.environ.get("CF_COST_FWD") or
                _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "ESTIMATION_OUTPUT" / "COST_FWD")


# ==========================================================================
# Assemble Δψ matrices per firm type
# ==========================================================================
def _block_columns(df: pd.DataFrame):
    """Return (omega_col, gamma_cols, zeta_col, psi1_col) from the block layout."""
    psi1 = "psi1"
    omega = "psi2_omega"
    zeta = "psi4_zeta"
    gamma = [c for c in df.columns if c.startswith("psi3_gamma_")]
    for c in (psi1, omega, zeta):
        if c not in df.columns:
            raise ValueError(f"Missing ψ block column '{c}' in {list(df.columns)}")
    return psi1, omega, gamma, zeta


def build_delta(eq: pd.DataFrame, dev: pd.DataFrame):
    """Δψ = ψ_eq − ψ_dev, broadcast eq to each (firm, shock) row of dev.

    Returns a dict with, for each firm type κ ∈ {'B','D'}, the arrays
    (d1, d_omega, d_gamma [n×nZ], d_zeta) over all (firm, shock) rows of that type.
    """
    psi1, omega, gamma, zeta = _block_columns(eq)
    eq_idx = eq.set_index("firm")
    # Align eq to dev rows by firm.
    eq_aligned = eq_idx.loc[dev["firm"].values]
    out = {}
    for kappa, mask in (("B", dev["is_B"].values.astype(bool)),
                        ("D", ~dev["is_B"].values.astype(bool))):
        if mask.sum() == 0:
            continue
        d1 = (eq_aligned[psi1].values - dev[psi1].values)[mask]
        d_om = (eq_aligned[omega].values - dev[omega].values)[mask]
        d_ze = (eq_aligned[zeta].values - dev[zeta].values)[mask]
        d_ga = (eq_aligned[gamma].values - dev[gamma].values)[mask]  # (n, nZ)
        firms = dev["firm"].values[mask]
        out[kappa] = dict(d1=d1, d_omega=d_om, d_gamma=d_ga, d_zeta=d_ze,
                          firms=firms, gamma_names=gamma)
    return out


# ==========================================================================
# Objective (V_Main eq:17) and analytic gradient
# ==========================================================================
def _rms(x):
    """Root-mean-square scale of a column; floored so a degenerate block → 1.0."""
    r = float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))
    return r if r > 1e-30 else 1.0


def _scaled_design(blk):
    """RMS-condition the Δψ blocks.

    Raw ψ blocks differ by orders of magnitude (Σdep ~1e12 vs Σdep·spread ~1e10), so
    the L-BFGS-B objective is badly scaled and the recovered ω is a scale artifact
    (and collinear regressors starve ζ). We map to unit-RMS regressors and solve for
    scaled coefficients b, then unscale. Design columns (order): [ω, γ_1..γ_nZ, (1+ζ)].

        g/s1 = d1/s1 − b_ω·(d_ω/s_ω) − Σ_z b_γz·(d_γz/s_γz) − b_ζ·(d_ζ/s_ζ)
        with  b_ω = ω·s_ω/s1,  b_γz = γ_z·s_γz/s1,  b_ζ = (1+ζ)·s_ζ/s1.

    Note the ψ4 design coefficient is b_ζ = (1+ζ)·s_ζ/s1 — it loads the FULL (1+ζ),
    not ζ; solve_kappa unscales it and subtracts 1 to report ζ (see that docstring).

    Returns (c1, Xs, scales) for the problem  min_b Σ min{c1 − Xs·b, 0}².
    """
    nZ = blk["d_gamma"].shape[1]
    s1 = _rms(blk["d1"]); s_om = _rms(blk["d_omega"]); s_ze = _rms(blk["d_zeta"])
    s_ga = np.array([_rms(blk["d_gamma"][:, z]) for z in range(nZ)])
    c1 = blk["d1"] / s1
    cols = [blk["d_omega"] / s_om]
    cols += [blk["d_gamma"][:, z] / s_ga[z] for z in range(nZ)]
    cols += [blk["d_zeta"] / s_ze]
    Xs = np.column_stack(cols) if cols else np.zeros((len(c1), 0))
    return c1, Xs, dict(s1=s1, s_om=s_om, s_ga=s_ga, s_ze=s_ze, nZ=nZ)


def _obj(b, c1, Xs):
    v = np.minimum(c1 - Xs @ b, 0.0)
    return float(v @ v)


def _grad(b, c1, Xs):
    v = np.minimum(c1 - Xs @ b, 0.0)        # 0 where g>=0; dg/db = −Xs
    return -2.0 * (Xs.T @ v)


def solve_kappa(blk):
    """Solve the eq:17 argmin in RMS-scaled coordinates, then unscale.

    PARAMETRIZATION. ψ1 carries the GROSS asset return r^j (V_Main eq:16 / appendix
    B-6-B / B-6-D, lines 874–876): it does NOT subtract the funding rate r^f. In the
    value function the single −r^f·Dep funding term is therefore delivered entirely by
    the ψ4 loading, whose coefficient is −(1+ζ): the "+1" IS that −r^f·Dep charge (gross
    ψ1 never subtracts it), and ζ is the incremental funding wedge on top. Concretely the
    inequality display above eq:17 reads

        g = Δψ1 − ω·Δψ2 − γ′·Δψ3 − (1+ζ)·Δψ4,

    so the fitted ψ4 design coefficient (unscaled: b[1+nZ]·s1/s_ze) equals (1+ζ). We
    SUBTRACT 1 to report ζ itself, matching V_Main eq:17 / B-6 and the downstream Julia
    theta_c in cf_0_psi_basis.jl, which reconstructs the loading as −(1+ζ) and also uses ζ
    as the r^f cost coefficient. (2026-07-16: this is a fix. Prior JSONs stored b[1+nZ]·… with
    NO −1, i.e. they reported 1+ζ mislabeled as ζ; the old solve_kappa docstring claiming
    "gross ψ1 ⇒ ψ4 loading IS ζ" had the algebra backwards. All cost_params_*.json written
    before 2026-07-16 are ζ-too-high-by-1.)

    NO SIGN RESTRICTIONS (eq:17 is an unconstrained argmin; ω, ζ, γ are all free).

    Earlier revisions of this file imposed ω ≥ 0 ("non-negative intercept cost") and
    1+ζ ≥ 0 ("non-negative funding base"). Those bounds are NOT in V_Main and were a
    code-side choice. They are removed, because with a hinge objective a binding bound does
    not yield a conservative estimate — it yields a CORNER that silently absorbs
    misspecification. That is what happened: ψ1 was double-charging r^f and the deposit
    franchise had no asset-side margin (r^j − r^f), so the moments wanted ω < 0; the bound
    reported ω = 0 with a quiet flag instead of exposing the error.

    In BBL (bajari2007estimating) the inequality set IS the identifying discipline; sign
    bounds substitute assumption for it. Where positivity of marginal cost matters, the
    standard device is a log-cost reparametrization (mc = exp(w′γ)), not box-clamping.

    So: estimate free, and READ THE SIGN AS A DIAGNOSTIC. ω < 0 means the value function is
    misspecified (most likely a missing/incorrect asset return r^j), not that cost is negative.
    """
    c1, Xs, sc = _scaled_design(blk)
    nZ = sc["nZ"]
    bounds = None                      # unconstrained — matches V_Main eq:17
    res = minimize(_obj, np.zeros(2 + nZ), args=(c1, Xs), jac=_grad,
                   method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=5000, ftol=1e-14, gtol=1e-10))
    b = res.x
    omega = float(b[0] * sc["s1"] / sc["s_om"])
    gamma = b[1:1 + nZ] * sc["s1"] / sc["s_ga"]
    # ψ4 design coef unscales to (1+ζ); subtract 1 to report ζ (V_Main eq:17 / B-6).
    zeta = float(b[1 + nZ] * sc["s1"] / sc["s_ze"] - 1.0)
    g = c1 - Xs @ b
    grad_inf = float(np.max(np.abs(_grad(b, c1, Xs)))) if b.size else 0.0
    return dict(omega=omega, zeta=zeta,
                gamma=dict(zip(blk["gamma_names"], gamma.tolist())),
                objective=float(res.fun), success=bool(res.success),
                theta=np.concatenate([[omega, zeta], gamma]),
                frac_bind=float(np.mean(g < 0.0)),
                opt_success=bool(res.success), opt_status=int(res.status),
                opt_message=str(res.message), opt_nit=int(res.nit),
                opt_grad_inf=grad_inf,
                # private handles for profiling / multistart (not serialized)
                _b=b, _c1=c1, _Xs=Xs, _scales=sc)


# ==========================================================================
# Firm-block bootstrap for inference
# ==========================================================================
def bootstrap_kappa(blk, n_boot, seed=42):
    """Resample FIRMS with replacement (block bootstrap), refit, collect params.

    Refits go through the (fixed) solve_kappa, so bootstrap draws use the same
    ζ = (unscaled ψ4 coef) − 1 convention; a constant −1 shift leaves the SE unchanged.
    """
    rng = np.random.default_rng(seed)
    firms = blk["firms"]
    uniq = np.unique(firms)
    idx_by_firm = {f: np.where(firms == f)[0] for f in uniq}
    draws = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_firm[f] for f in chosen])
        sub = dict(d1=blk["d1"][rows], d_omega=blk["d_omega"][rows],
                   d_gamma=blk["d_gamma"][rows], d_zeta=blk["d_zeta"][rows],
                   firms=firms[rows], gamma_names=blk["gamma_names"])
        draws.append(solve_kappa(sub)["theta"])
    arr = np.vstack(draws)
    return arr.std(axis=0)


# ==========================================================================
# Diagnostics: 1-D identified-set profiles and multistart smoke check
# ==========================================================================
def profile_param(fit, which, opt_val, se_val, npts=81, span=5.0):
    """Profile the objective along one scalar param (ω or ζ), other fitted b's held fixed.

    Grid runs over opt ± span·(bootstrap SE if given, else |opt|+1) in the REPORTED
    parameter, mapped to the scaled b-coordinate for evaluation (sweep b_i, unscale for
    reporting). Returns {grid, F, flat_lo, flat_hi} where [flat_lo, flat_hi] is the interval
    over which F ≤ F_min·(1+1e-6)+1e-12 — the near-flat (set-identified) width.
    """
    b0 = fit["_b"].copy(); c1 = fit["_c1"]; Xs = fit["_Xs"]; sc = fit["_scales"]
    nZ = sc["nZ"]
    if which == "omega":
        idx = 0
        to_b = lambda v: v * sc["s_om"] / sc["s1"]
    elif which == "zeta":
        idx = 1 + nZ
        to_b = lambda v: (v + 1.0) * sc["s_ze"] / sc["s1"]   # reported ζ ↦ (1+ζ) ↦ b
    else:
        raise ValueError(which)
    spread = span * (se_val if (se_val is not None and np.isfinite(se_val) and se_val > 0)
                     else abs(opt_val) + 1.0)
    grid = np.linspace(opt_val - spread, opt_val + spread, npts)
    F = np.empty(npts)
    for k, v in enumerate(grid):
        b = b0.copy(); b[idx] = to_b(v)
        F[k] = _obj(b, c1, Xs)
    fmin = float(F.min())
    flat = grid[F <= fmin * (1.0 + 1e-6) + 1e-12]
    return dict(grid=grid.tolist(), F=F.tolist(),
                flat_lo=float(flat.min()), flat_hi=float(flat.max()))


def multistart_check(fit, f_opt, n=5, seed=12345):
    """Re-solve from n random N(0,1) inits (scaled b-space). The objective is a convex
    squared hinge, so every restart must reach the same F as the main optimum. Returns the
    max |F_restart − f_opt| — a nonzero value flags a gradient/scaling bug, not multimodality.
    """
    c1 = fit["_c1"]; Xs = fit["_Xs"]; sc = fit["_scales"]
    dim = 2 + sc["nZ"]
    rng = np.random.default_rng(seed)
    max_dF = 0.0
    for _ in range(n):
        x0 = rng.standard_normal(dim)
        res = minimize(_obj, x0, args=(c1, Xs), jac=_grad, method="L-BFGS-B",
                       bounds=None, options=dict(maxiter=5000, ftol=1e-14, gtol=1e-10))
        max_dF = max(max_dF, abs(float(res.fun) - f_opt))
    return max_dF


def main():
    ap = argparse.ArgumentParser(description="BBL Step 2 cost solver (V_Main eq:17).")
    ap.add_argument("--estim", type=int, default=6)
    ap.add_argument("--spec", type=int, default=12)
    ap.add_argument("--stage", type=str, default="extended")
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="firm-block bootstrap reps for SEs (0 to skip)")
    ap.add_argument("--profile", action="store_true",
                    help="compute 1-D ω/ζ objective profiles (identified-set widths)")
    args = ap.parse_args()

    tag = f"E{args.estim}_spec_{args.spec}_{args.stage}{args.suffix}"
    eq_path = COST_FWD / f"psi_eq_{tag}.parquet"
    if not eq_path.exists():
        raise FileNotFoundError(f"Missing {eq_path.name} — run cost_2_fwd_sim.jl first.")
    # Deviation ψ may be a single file (n-shards=1) or several shard files; merge all.
    dev_files = sorted(COST_FWD.glob(f"psi_dev_{tag}*.parquet"))
    if not dev_files:
        raise FileNotFoundError(f"No psi_dev_{tag}*.parquet — run cost_2_fwd_sim.jl (all shards) first.")

    eq = pd.read_parquet(eq_path)
    dev = pd.concat([pd.read_parquet(f) for f in dev_files], ignore_index=True)
    # Drop any accidental duplicate (firm, shock) rows from re-runs.
    if "shock" in dev.columns:
        dev = dev.drop_duplicates(subset=["firm", "shock"]).reset_index(drop=True)
    print(f"  Loaded ψ_eq ({len(eq)} firms) and ψ_dev ({len(dev)} firm×shock rows) "
          f"from {len(dev_files)} shard file(s)")

    blocks = build_delta(eq, dev)
    results = {}
    for kappa, blk in blocks.items():
        nZ = blk["d_gamma"].shape[1]
        fit = solve_kappa(blk)
        se = (bootstrap_kappa(blk, args.bootstrap).tolist()
              if args.bootstrap > 0 else [None] * (2 + nZ))
        gamma_se = dict(zip(blk["gamma_names"], se[2:2 + nZ]))

        # Multistart smoke check (cheap, always on).
        ms_max_dF = multistart_check(fit, fit["objective"])

        rec = dict(
            omega=fit["omega"], omega_se=se[0],
            zeta=fit["zeta"], zeta_se=se[1],
            gamma=fit["gamma"], gamma_se=gamma_se,
            objective=fit["objective"], success=fit["success"],
            frac_bind=fit["frac_bind"],
            opt_success=fit["opt_success"], opt_status=fit["opt_status"],
            opt_message=fit["opt_message"], opt_nit=fit["opt_nit"],
            opt_grad_inf=fit["opt_grad_inf"],
            multistart_max_dF=float(ms_max_dF),
            n_firms=int(np.unique(blk["firms"]).size),
            n_rows=int(blk["d1"].size),
        )

        if args.profile:
            rec["profile_omega"] = profile_param(fit, "omega", fit["omega"], se[0])
            rec["profile_zeta"] = profile_param(fit, "zeta", fit["zeta"], se[1])

        results[kappa] = rec

        se0 = f"{se[0]:.3g}" if se[0] is not None else "—"
        se1 = f"{se[1]:.3g}" if se[1] is not None else "—"
        diag = " | ω<0 (misspecification diagnostic — see docstring)" if fit["omega"] < 0 else ""
        print(f"  [{kappa}] ω={fit['omega']:.4g} (se {se0}) | "
              f"ζ={fit['zeta']:.4g} (se {se1}) | obj={fit['objective']:.4g} | "
              f"bind={100*fit['frac_bind']:.0f}% | firms={rec['n_firms']}")
        print(f"        opt: success={fit['opt_success']} status={fit['opt_status']} "
              f"nit={fit['opt_nit']} ‖∇‖∞={fit['opt_grad_inf']:.2e} "
              f"msΔF={ms_max_dF:.2e}{diag}")
        if ms_max_dF > 1e-6:
            print(f"  !!!! [{kappa}] MULTISTART DISAGREEMENT max ΔF={ms_max_dF:.3e} > 1e-6 — "
                  f"convex hinge should have a unique optimum; suspect a gradient/scaling bug !!!!")
        if args.profile:
            po, pz = rec["profile_omega"], rec["profile_zeta"]
            print(f"        profile ω: F_min={min(po['F']):.4g} flat=[{po['flat_lo']:.4g},"
                  f"{po['flat_hi']:.4g}] width={po['flat_hi']-po['flat_lo']:.4g}")
            print(f"        profile ζ: F_min={min(pz['F']):.4g} flat=[{pz['flat_lo']:.4g},"
                  f"{pz['flat_hi']:.4g}] width={pz['flat_hi']-pz['flat_lo']:.4g}")

    out_path = COST_FWD / f"cost_params_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved cost parameters → {out_path.name}")


if __name__ == "__main__":
    main()
