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
      --suffix _coherence --bootstrap 200
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parents[2]
COST_FWD = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "ESTIMATION_OUTPUT" / "COST_FWD"


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
def _unpack(theta, nZ):
    """theta = [omega, zeta, gamma_1..gamma_nZ]."""
    return theta[0], theta[1], theta[2:2 + nZ]


def g_values(theta, blk):
    omega, zeta, gamma = _unpack(theta, blk["d_gamma"].shape[1])
    return (blk["d1"]
            - omega * blk["d_omega"]
            - blk["d_gamma"] @ gamma
            - (1.0 + zeta) * blk["d_zeta"])


def objective(theta, blk):
    g = g_values(theta, blk)
    viol = np.minimum(g, 0.0)
    return float(np.sum(viol ** 2))


def gradient(theta, blk):
    g = g_values(theta, blk)
    viol = np.minimum(g, 0.0)              # 0 where g>=0
    # dF/dθ = Σ 2·viol · dg/dθ ;  dg/dω=−d_omega, dg/dγ=−d_gamma, dg/dζ=−d_zeta
    two_v = 2.0 * viol
    dF_omega = np.sum(two_v * (-blk["d_omega"]))
    dF_zeta = np.sum(two_v * (-blk["d_zeta"]))
    dF_gamma = (two_v[:, None] * (-blk["d_gamma"])).sum(axis=0)
    return np.concatenate([[dF_omega, dF_zeta], dF_gamma])


def solve_kappa(blk, x0=None):
    nZ = blk["d_gamma"].shape[1]
    if x0 is None:
        x0 = np.zeros(2 + nZ)
    res = minimize(objective, x0, args=(blk,), jac=gradient, method="L-BFGS-B",
                   options=dict(maxiter=2000, ftol=1e-12))
    omega, zeta, gamma = _unpack(res.x, nZ)
    return dict(omega=float(omega), zeta=float(zeta),
                gamma=dict(zip(blk["gamma_names"], gamma.tolist())),
                objective=float(res.fun), success=bool(res.success), x=res.x)


# ==========================================================================
# Firm-block bootstrap for inference
# ==========================================================================
def bootstrap_kappa(blk, n_boot, seed=42):
    """Resample FIRMS with replacement (block bootstrap), refit, collect params."""
    rng = np.random.default_rng(seed)
    firms = blk["firms"]
    uniq = np.unique(firms)
    # Precompute row indices per firm for fast resampling.
    idx_by_firm = {f: np.where(firms == f)[0] for f in uniq}
    draws = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_firm[f] for f in chosen])
        sub = dict(d1=blk["d1"][rows], d_omega=blk["d_omega"][rows],
                   d_gamma=blk["d_gamma"][rows], d_zeta=blk["d_zeta"][rows],
                   firms=firms[rows], gamma_names=blk["gamma_names"])
        fit = solve_kappa(sub)
        draws.append(fit["x"])
    arr = np.vstack(draws)
    return arr.std(axis=0)


def main():
    ap = argparse.ArgumentParser(description="BBL Step 2 cost solver (eq 18).")
    ap.add_argument("--estim", type=int, default=6)
    ap.add_argument("--spec", type=int, default=12)
    ap.add_argument("--stage", type=str, default="extended")
    ap.add_argument("--suffix", type=str, default="_coherence")
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
            n_firms=int(np.unique(blk["firms"]).size),
            n_rows=int(blk["d1"].size),
        )
        print(f"  [{kappa}] ω={fit['omega']:.4g} (se {se[0]}) | "
              f"ζ={fit['zeta']:.4g} (se {se[1]}) | obj={fit['objective']:.4g} | "
              f"firms={results[kappa]['n_firms']}")

    out_path = COST_FWD / f"cost_params_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved cost parameters → {out_path.name}")


if __name__ == "__main__":
    main()
