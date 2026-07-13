"""
estimation_1_cost_3_solve.py
============================
CF2 — BBL Step 2, part 2: recover marginal-cost parameters (ω, ζ, γ)^κ for
κ ∈ {B, D} by minimizing the sum of squared FOC-inequality violations (eq 18),
using the firm-level ψ basis produced by cost_2_fwd_sim.jl.

Moment inequality (per firm j, deviation σ̃):

    g_j(σ̂, σ̃) = (ψ_eq,j − ψ_dev,j,σ̃)' · [ 1 , −ω , −γ' , −(1+ζ) ]  ≥ 0

with the ψ block layout written by cost_2_fwd_sim.jl:
    [ psi1 ,  psi2_omega ,  psi3_gamma_<z>… ,  psi4_zeta ]
so, writing Δ = ψ_eq − ψ_dev,

    g = Δ·psi1  −  ω·Δ·psi2_omega  −  Σ_z γ_z·Δ·psi3_gamma_z  −  (1+ζ)·Δ·psi4_zeta.

Deviations should make the bank worse off, so g ≥ 0 in equilibrium; we minimize

    F^κ(ω, ζ, γ) = Σ_{j∈κ, σ̃}  min{ g_j , 0 }²                      (eq 18)

separately for B- and D-type firms. χ (private cost shock) is dropped, as in the draft.

Inputs (from cost_2_fwd_sim.jl):
    COST_FWD/psi_eq_{tag}.parquet   — firm, is_B, psi1, psi2_omega, psi3_gamma_*, psi4_zeta
    COST_FWD/psi_dev_{tag}.parquet  — shock, firm, is_B, <same blocks>
where tag = E{estim}_spec_{spec}_{stage}{suffix}.

Outputs:
    COST_FWD/cost_params_{tag}.json — (ω, ζ, γ) per type with bootstrap SEs.

⚠ Run only after cost_2_fwd_sim.jl has produced its parquets (which itself is gated
on the cluster data download and the author's go-ahead).

Usage:
  python estimation_1_cost_3_solve.py --estim 6 --spec 12 --stage extended \\
      --bootstrap 200
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
# Objective (eq 18) and analytic gradient
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
    """Solve eq-18 in RMS-scaled coordinates, then unscale.

    PARAMETRIZATION. ψ1 carries the GROSS r^j (V_Main eq 16), so the ψ4 loading is ζ
    itself — NOT (1+ζ). (Before the cf_0_psi_basis fix, ψ1 held the NET (r^j − r^f) while
    θ_c still applied −(1+ζ)ψ4, double-charging r^f.) So:
        g = Δψ1 − ω·Δψ2 − γ′·Δψ3 − ζ·Δψ4.

    NO SIGN RESTRICTIONS (eq-17 is an unconstrained argmin; ω, ζ, γ are all free).

    Earlier revisions of this file imposed ω ≥ 0 ("non-negative intercept cost") and
    1+ζ ≥ 0 ("non-negative funding base"). Those bounds are NOT in V_Main and were a
    code-side choice. They are removed, because with a hinge objective a binding bound does
    not yield a conservative estimate — it yields a CORNER that silently absorbs
    misspecification. That is what happened: ψ1 was double-charging r^f and the deposit
    franchise had no asset-side margin (r^j − r^f), so the moments wanted ω < 0; the bound
    reported ω = 0 with a quiet `omega_at_bound` flag instead of exposing the error.

    In BBL (bajari2007estimating) the inequality set IS the identifying discipline; sign
    bounds substitute assumption for it. Where positivity of marginal cost matters, the
    standard device is a log-cost reparametrization (mc = exp(w′γ)), not box-clamping.

    So: estimate free, and READ THE SIGN AS A DIAGNOSTIC. ω < 0 means the value function is
    misspecified (most likely a missing/incorrect asset return r^j), not that cost is negative.
    `omega_at_bound` is retained only as a legacy flag and should now always be False.
    """
    c1, Xs, sc = _scaled_design(blk)
    nZ = sc["nZ"]
    bounds = None                      # unconstrained — matches V_Main eq 17
    res = minimize(_obj, np.zeros(2 + nZ), args=(c1, Xs), jac=_grad,
                   method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=5000, ftol=1e-14, gtol=1e-10))
    b = res.x
    omega = float(b[0] * sc["s1"] / sc["s_om"])
    gamma = b[1:1 + nZ] * sc["s1"] / sc["s_ga"]
    zeta = float(b[1 + nZ] * sc["s1"] / sc["s_ze"])      # ψ4 loading IS ζ (no −1)
    g = c1 - Xs @ b
    return dict(omega=omega, zeta=zeta,
                gamma=dict(zip(blk["gamma_names"], gamma.tolist())),
                objective=float(res.fun), success=bool(res.success),
                theta=np.concatenate([[omega, zeta], gamma]),
                omega_at_bound=bool(b[0] <= 1e-9),
                frac_bind=float(np.mean(g < 0.0)))


# ==========================================================================
# Firm-block bootstrap for inference
# ==========================================================================
def bootstrap_kappa(blk, n_boot, seed=42):
    """Resample FIRMS with replacement (block bootstrap), refit, collect params."""
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


def main():
    ap = argparse.ArgumentParser(description="BBL Step 2 cost solver (eq 18).")
    ap.add_argument("--estim", type=int, default=6)
    ap.add_argument("--spec", type=int, default=12)
    ap.add_argument("--stage", type=str, default="extended")
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="firm-block bootstrap reps for SEs (0 to skip)")
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
        results[kappa] = dict(
            omega=fit["omega"], omega_se=se[0],
            zeta=fit["zeta"], zeta_se=se[1],
            gamma=fit["gamma"], gamma_se=gamma_se,
            objective=fit["objective"], success=fit["success"],
            omega_at_bound=fit["omega_at_bound"], frac_bind=fit["frac_bind"],
            n_firms=int(np.unique(blk["firms"]).size),
            n_rows=int(blk["d1"].size),
        )
        se0 = f"{se[0]:.3g}" if se[0] is not None else "—"
        se1 = f"{se[1]:.3g}" if se[1] is not None else "—"
        print(f"  [{kappa}] ω={fit['omega']:.4g} (se {se0}){' [BOUND]' if fit['omega_at_bound'] else ''} | "
              f"ζ={fit['zeta']:.4g} (se {se1}) | obj={fit['objective']:.4g} | "
              f"bind={100*fit['frac_bind']:.0f}% | firms={results[kappa]['n_firms']}")

    out_path = COST_FWD / f"cost_params_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved cost parameters → {out_path.name}")


if __name__ == "__main__":
    main()
