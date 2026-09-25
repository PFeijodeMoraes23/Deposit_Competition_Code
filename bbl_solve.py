"""
bbl_solve.py
============================
BBL Step 2, part 2: recover marginal-cost parameters (ω, ζ, γ)^κ for
κ ∈ {B, D} by minimizing the sum of squared FOC-inequality violations, using the
firm-level ψ basis produced by bbl_fwd_sim.jl.

Moment inequality (per firm j, deviation σ̃) — the unnumbered display above
V_Main \\label{eq:17}:

    g_j(σ̂, σ̃) = (ψ_eq,j − ψ_dev,j,σ̃)' · [ 1 , −ω , −γ' , −(1+ζ) ]  ≥ 0

with the ψ block layout written by bbl_fwd_sim.jl:
    [ psi1 ,  psi2_omega ,  psi3_gamma_<z>… ,  psi4_zeta ]
so, writing Δ = ψ_eq − ψ_dev,

    g = Δ·psi1  −  ω·Δ·psi2_omega  −  Σ_z γ_z·Δ·psi3_gamma_z  −  (1+ζ)·Δ·psi4_zeta.

Deviations should make the bank worse off, so g ≥ 0 in equilibrium; we minimize

    F^κ(ω, ζ, γ) = Σ_{j∈κ, σ̃}  min{ g_j , 0 }²                      (V_Main eq:17)

separately for B- and D-type firms. χ (private cost shock) is dropped, as in the draft.

Inputs (from bbl_fwd_sim.jl):
    COST_FWD/psi_eq_{tag}.parquet   — firm, start_q, rf_source, rf_bar_beta, is_B,
                                      psi1, psi2_omega, psi3_gamma_*, psi4_zeta
    COST_FWD/psi_dev_{tag}.parquet  — shock, firm, start_q, is_B, <same blocks>
where tag = E{estim}_spec_{spec}_{stage}{suffix}{psi_tag} and --psi-tag carries the
multi-start marker stamped on its filenames ("_ms{P}", P = rate paths per deviation, as
bbl_run.sh sets it; empty for a single-start run).

Outputs:
    COST_FWD/cost_params_{tag}.json — (ω, ζ, γ) per type, optimizer-health diagnostics, and
    inference: `subsample` (critical values + √n-rate comparison CI) and, with --profile,
    `ci_omega`/`ci_zeta` (the criterion inverted at the subsampled critical value).
    `omega_se`/`zeta_se`/`gamma_se` are the legacy firm-block bootstrap SDs, RETAINED FOR
    COMPARISON ONLY.  Two extra top-level keys, `run` and `identification`, carry the
    multi-start provenance and the gate verdict (see PROMOTION GATE below); the per-type
    blocks stay "B"/"D" so cf_psi_basis.jl:load_cost_params and make_bbl_cost_tables.py
    read them unchanged.

THE START-QUARTER DIMENSION.  ω and ζ separate only if Δψ₄/Δψ₂ — the β-weighted mean forward
r^f — MOVES across observations.  One launch quarter means one forward curve for every firm,
so that ratio is a near-constant and only c̄ = ω + r̄^f·ζ is identified (corr(Δψ₂,Δψ₄) =
0.99997 measured on the single-curve production run).  Simulating each firm from SEVERAL
launch quarters, each discounted by the Focus curve of ITS OWN vintage, is what puts spread in
the ratio: the sample Selic runs 14.25 → 2 → 13.75%, and crossing 36 vintage curves with the
deposit paths was measured at corr 0.852.  Every row therefore carries a `start_q`, and
alignment, diagnostics and the promotion gate are all organised around it.

PROMOTION GATE.  The split ω vs ζ is a CLAIM about the design, so it is tested, not assumed.
ridge_by_start reports the BKW condition index of the two unit-length-scaled columns per start
and pooled; `identified_split` is (pooled index ≤ --ridge-cond-max, default 30 — the BKW rule
of thumb) for EVERY firm type present.  The tagged json is always written.  --promote copies
it onto the UNTAGGED cost_params name (the one cf1_net/cf3/cf5/cf6 read) only if the gate
passes, or if --force-ridge overrides it; otherwise the per-start table is printed and the
process EXITS 3.  A nonzero exit fails the SLURM solve job, and the cost-consuming CF jobs
chain afterok on it (cf_run.sh --cost-afterok), so they are CANCELLED rather than left to
consume a stale untagged file.  A cancelled dependent writes NO log at all — an absent
cf1_net .out is the expected symptom of a refused promotion, and the reason is in THIS job's
log, not in a missing one.

INFERENCE (2026-07-17).  F is a squared hinge over moment INEQUALITIES: kinked and possibly
SET-identified, so the nonparametric bootstrap is inconsistent here — it neither handles a
boundary/partially-identified parameter nor reproduces the non-standard limit of a kinked
criterion.  BBL propose SUBSAMPLING; Chernozhukov–Hong–Tamer invert the criterion with a
subsampled critical value.  Both are implemented (subsample_kappa, profile_ci) and are the
headline; the bootstrap is kept only so the two can be compared.  Cheap, because ψ is
precomputed — only the fast hinge minimisation is re-run.

⚠ TWO VARIANCE COMPONENTS ARE STILL MISSING.  (1) FIRST-STAGE: δ̂, θ̂₂, α̂, the Step-1 policy,
φ̂ and M are held FIXED and ψ is never re-simulated, so everything here is inference
*conditional on demand*; with the demand stage's few effective clusters that term may
dominate.  (2) SIMULATION: ψ carries Monte-Carlo error from the R draws.  See §2.8 of
counterfactuals_plan.md for the propagation recipe.

⚠ ζ SPECIFICALLY: within one launch quarter every firm is simulated against the SAME forward
r^f path, so ζ (a pass-through parameter) is identified off macro variation — across the
launch quarters — not off n independent firms.  With S starts the effective sample for ζ is
closer to S than to n, and the launch quarters are themselves serially dependent draws of one
Selic cycle.  No firm-resampling scheme — bootstrap or subsample — delivers honest uncertainty
for that; the firm-level intervals below understate it, and `identification.ratio_cv` is the
statistic that says how much rate variation the design actually bought.

⚠ Run only after bbl_fwd_sim.jl has produced its parquets, AND only with σ̂ from
the FITTED policy (--policy-csv upstream): with deviations around raw observed spreads,
frac_bind ≈ ½ mechanically and nothing here is interpretable.

Usage:
  python bbl_solve.py --estim 6 --spec 12 --stage extended \\
      --subsample 200 --profile --ci-level 0.95      # headline
  ... --bootstrap 200                                 # firm-block SEs, for comparison
  ... --psi-tag _ms8 --promote                        # multi-start psi, gated promotion
"""
try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except ModuleNotFoundError:
    pass  # utils/ is a local-dev convenience (re-exec into the Windows .venv); absent + a no-op on the cluster

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# COST_FWD holds the ψ parquets (from estimation_bbl_2_fwd_sim) + the cost_params json.
# cost_fwd_dir() is the ONE implementation of that location: it defaults to
# estimation_output()/COST_FWD and honours CF_COST_FWD, the env the BBL jobs export to point the
# whole step at the cluster's data/output/bbl folder. Reading the env here as well would be a
# second implementation of the same rule, free to drift from the writer's.
#
# The location is pinned to estimation_output(), NOT SLEEP_OUT_ROOT: bbl_fwd_sim.jl
# writes these parquets and knows nothing of the sleepiness sandbox, so redirecting the reader
# alone would aim it at a directory the writer never fills.
#
# utils/ is not part of the cluster code payload in every deployment, so the import sits under the
# same ModuleNotFoundError guard as the venv helper above. That branch spells the rule out by
# hand — the parents[2] walk assumes the repo sits two levels inside the data tree, which is why
# it is the last resort and not the primary.
try:
    from utils import paths as _paths
    COST_FWD = _paths.cost_fwd_dir()
except ModuleNotFoundError:
    _COST_FWD_DEFAULT = (Path(__file__).resolve().parents[2] / "BCB" / "Egan_et_al_2025_Rep"
                         / "processed" / "ESTIMATION_OUTPUT" / "COST_FWD")
    COST_FWD = Path(os.environ.get("CF_COST_FWD") or _COST_FWD_DEFAULT)


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
    """Δψ = ψ_eq − ψ_dev, broadcast eq to each (firm, start_q, shock) row of dev.

    ALIGNMENT KEY = (firm, start_q), NOT firm.  With vintage-matched forward curves a firm
    carries one equilibrium ψ per launch quarter, and the whole point of the multi-start design
    is that those rows sit at different points of the Selic cycle.  Aligning on firm alone
    would difference a 2016Q1 deviation against a 2024Q4 equilibrium and book the rate cycle as
    a deviation effect — a firm with S starts would also silently fan out into S copies of every
    deviation row.  Single-start parquets get start_q = "all" injected by main(), so the key is
    uniform and this reduces to the firm-only alignment there.

    A firm therefore contributes one row per (start × deviation).  THE DEPENDENCE UNIT IS STILL
    THE FIRM — see bootstrap_kappa / subsample_kappa.

    Returns a dict with, for each firm type κ ∈ {'B','D'}, the arrays
    (d1, d_omega, d_gamma [n×nZ], d_zeta) over all (firm, start, shock) rows of that type, plus
    the parallel label arrays `firms` and `starts`.
    """
    psi1, omega, gamma, zeta = _block_columns(eq)
    for nm, df in (("psi_eq", eq), ("psi_dev", dev)):
        if "start_q" not in df.columns:
            raise ValueError(f"{nm} has no start_q column — main() injects 'all' before this")

    eq_idx = eq.set_index(["firm", "start_q"])
    if eq_idx.index.has_duplicates:
        dup = eq_idx.index[eq_idx.index.duplicated()].unique().tolist()[:5]
        raise ValueError(f"psi_eq has duplicate (firm, start_q) rows, e.g. {dup} — the "
                         f"equilibrium ψ must be one row per firm per launch quarter")
    key = pd.MultiIndex.from_arrays([dev["firm"].values, dev["start_q"].values])
    missing = ~key.isin(eq_idx.index)
    if missing.any():
        ex = sorted(set(map(tuple, np.column_stack(
            [dev["firm"].values[missing], dev["start_q"].values[missing]]).tolist())))[:5]
        raise ValueError(f"{int(missing.sum())} psi_dev rows have no matching psi_eq "
                         f"(firm, start_q), e.g. {ex} — psi_eq and the psi_dev shards are "
                         f"different vintages or were built over different launch quarters")
    eq_aligned = eq_idx.reindex(key)

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
        starts = dev["start_q"].values[mask]
        out[kappa] = dict(d1=d1, d_omega=d_om, d_gamma=d_ga, d_zeta=d_ze,
                          firms=firms, starts=starts, gamma_names=gamma)
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
    theta_c in cf_psi_basis.jl, which reconstructs the loading as −(1+ζ) and also uses ζ
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
                # The criterion is a sum of squared SHORTFALLS, so it is exactly 0.0 whenever
                # every inequality holds, and then the gradient is 0 too. An optimum with F == 0
                # is one point inside a set of equally good theta, not an estimate; if the
                # optimiser also never moved (nit == 0), the point it reports is its start,
                # b = 0, i.e. omega = 0, zeta = -1, gamma = 0.
                objective_zero=bool(float(res.fun) == 0.0), at_init=bool(int(res.nit) == 0),
                # private handles for profiling / multistart (not serialized)
                _b=b, _c1=c1, _Xs=Xs, _scales=sc)


# ==========================================================================
# The identified combination: marginal cost at the mean forward r^f
# ==========================================================================
def rbar_of_block(blk):
    r"""r̄^f and the ω/ζ ridge diagnostics for one firm-type block.

    ω loads on Δψ₂ (`d_omega`) and (1+ζ) on Δψ₄ (`d_zeta`). Structurally
    Δψ₄/Δψ₂ is the β^t-weighted mean of r^f_t, which is COMMON to every firm, so it varies
    only through the timing of ΔDep. With the BCB Focus curve nearly flat inside the
    discounted window (β=0.9 puts ~90% of the weight in the first ~22 quarters, where rf moves
    only 0.0333 → ~0.0285) that ratio is a near-constant and the two columns are collinear:
    measured corr > 0.99997 on the 2026-08-03 production run, i.e. ≤0.005% of Δψ₄ is
    independent of Δψ₂.

    Consequence: only c̄ = ω + r̄^f·ζ — marginal cost at the mean forward risk-free rate — is
    identified. The reported ω̂/ζ̂ split is wherever the optimizer stopped on the ridge
    ω = −r̄^f·ζ, which is why ω̂ < 0 appears without implying negative marginal cost.
    V_Main:584 states the failure condition ("a flat rate would leave it unidentified"); these
    numbers show it binds WITH the market curve loaded. See identification_notes.md §9.
    """
    d2 = np.asarray(blk["d_omega"], float)
    d4 = np.asarray(blk["d_zeta"], float)
    ok = np.isfinite(d2) & np.isfinite(d4) & (d2 != 0.0)
    d2, d4 = d2[ok], d4[ok]
    if d2.size == 0:
        return dict(rbar_f=float("nan"))
    ratio = d4 / d2
    b = float((d2 @ d4) / (d2 @ d2))
    resid = d4 - b * d2
    denom = float(((d4 - d4.mean()) ** 2).sum())
    return dict(
        rbar_f=float(np.median(ratio)),
        rbar_f_mean=float(ratio.mean()),
        ridge_corr=float(np.corrcoef(d2, d4)[0, 1]),
        ridge_cond=float(np.linalg.cond(np.column_stack([d2, d4]))),
        ridge_one_minus_R2=float(1.0 - (1.0 - (resid @ resid) / denom)) if denom > 0 else float("nan"),
    )


def _bkw_cond(d2, d4):
    """Belsley–Kuh–Welsch condition index of the two-column design [Δψ₂, Δψ₄].

    BKW scale each column to unit LENGTH (2-norm) and take σ_max/σ_min of the scaled,
    UNCENTERED matrix; their rule of thumb calls >30 damaging collinearity, which is where
    --ridge-cond-max defaults. Uncentered is both the BKW definition and the right object here:
    the eq:17 design has no intercept, so the near-dependence that starves ζ is the raw
    proportionality Δψ₄ ≈ r̄^f·Δψ₂, not a correlation about column means.

    Unit-LENGTH scaling (not RMS) is what makes the index scale-free, so it is comparable
    across starts and across blocks of different size — unlike `ridge_cond`, which is taken on
    the raw columns and therefore also carries the ~1/r̄^f units gap between them.
    """
    X = np.column_stack([np.asarray(d2, float), np.asarray(d4, float)])
    nrm = np.sqrt((X ** 2).sum(axis=0))
    nrm[nrm <= 0.0] = 1.0
    s = np.linalg.svd(X / nrm, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0.0 else float("inf")


def _ridge_diag_rows(blk, rows, fit=None):
    """Ridge geometry + a hinge refit on one row subset (one start, or all rows pooled).

    `fit` short-circuits the refit when the caller already solved exactly these rows (the
    pooled case), so the printed pooled line is the SAME optimum the json reports rather than a
    second solve that could differ in the last digits.
    """
    rows = np.asarray(rows)
    if rows.dtype == bool:          # positions, so `n` counts rows and not the mask length
        rows = np.flatnonzero(rows)
    d2 = np.asarray(blk["d_omega"], float)[rows]
    d4 = np.asarray(blk["d_zeta"], float)[rows]
    out = dict(n=int(rows.size), n_firms=int(np.unique(blk["firms"][rows]).size))
    ok = np.isfinite(d2) & np.isfinite(d4) & (d2 != 0.0)
    out["n_ratio"] = int(ok.sum())
    if ok.sum() >= 2:
        ratio = d4[ok] / d2[ok]
        with np.errstate(invalid="ignore"):
            cc = float(np.corrcoef(d2[ok], d4[ok])[0, 1])
        out.update(corr=cc if np.isfinite(cc) else float("nan"),
                   bkw_cond=_bkw_cond(d2[ok], d4[ok]),
                   rbar_f=float(np.median(ratio)),
                   rbar_f_mean=float(ratio.mean()))
    else:
        out.update(corr=float("nan"), bkw_cond=float("inf"),
                   rbar_f=float("nan"), rbar_f_mean=float("nan"))

    # A within-start refit is what makes c̄ per start meaningful: inside one start the two
    # columns are collinear again (one curve for every firm), so ω_s and ζ_s are arbitrary
    # points on that start's ridge, but c̄_s = ω_s + r̄^f_s·ζ_s — marginal cost at THAT
    # quarter's mean forward rate — is pinned. Those c̄_s are the quantities whose movement
    # across starts is the ζ identification, which cbar_slope_across_starts then reads off.
    ndim = 2 + int(blk["d_gamma"].shape[1])
    if rows.size > ndim:
        f = fit if fit is not None else solve_kappa(_sub_block(blk, rows))
        out.update(omega=float(f["omega"]), zeta=float(f["zeta"]),
                   frac_bind=float(f["frac_bind"]), objective=float(f["objective"]),
                   c_bar=float(f["omega"] + out["rbar_f"] * f["zeta"]))
    return out


def ridge_by_start(blk, fit=None, cond_max=30.0):
    r"""Per-start and pooled ω/ζ ridge diagnostics — the identification claim, tested.

    For every launch quarter and for the pool it reports n, corr(Δψ₂,Δψ₄), the BKW condition
    index of the two unit-length-scaled columns, r̄^f = median(Δψ₄/Δψ₂), frac_bind and c̄.

    WHAT THE NUMBERS SHOULD LOOK LIKE. Within ONE start every firm discounts the same forward
    curve, so each per-start line is expected to stay on the ridge (corr ≈ 0.99997, index in
    the hundreds) — that is not a failure. The identification comes from POOLING starts whose
    curves sit at different points of the Selic cycle: crossing 36 vintage curves with the
    deposit paths was measured at corr 0.852, index ≈ 3.5. So the pooled line is the one the
    gate reads, and a pooled corr that matches the per-start corr on multi-start input means
    the starts are not actually differing — a driver-side wiring failure (row_start / vintage
    curves not reaching the sim), not a marginal result. main() says so loudly.

    `ratio_cv` is the coefficient of variation of the per-start MEAN ratio across starts: the
    spread in r̄^f the multi-start design bought, in the units ζ is identified from. It is
    reported next to the condition index because they answer different questions — the index
    says whether the pooled columns are separable, the CV says how much of the rate cycle the
    launch quarters actually span.

    `identified_split` = (pooled BKW index ≤ cond_max).
    """
    starts = np.asarray(blk["starts"])
    uniq = sorted(set(starts.tolist()))
    per = {s: _ridge_diag_rows(blk, np.where(starts == s)[0]) for s in uniq}
    pooled = _ridge_diag_rows(blk, np.arange(starts.size), fit=fit)

    means = np.array([per[s].get("rbar_f_mean", np.nan) for s in uniq], float)
    m = means[np.isfinite(means)]
    ratio_cv = (float(m.std(ddof=0) / abs(m.mean()))
                if m.size >= 2 and m.mean() != 0.0 else float("nan"))

    # c̄_s against r̄^f_s across starts: OLS slope ≈ ζ, intercept ≈ ω, by the definition of c̄.
    # It uses only per-start quantities that are identified WITHIN a start, so it is an
    # independent read on the split that never touches the pooled hinge. Agreement with the
    # pooled ω̂/ζ̂ is the check that the pooled solve is using the cross-start variation and not
    # some within-start residual.
    slope = dict(n_starts=int(len(uniq)))
    rv = np.array([per[s].get("rbar_f", np.nan) for s in uniq], float)
    cv_ = np.array([per[s].get("c_bar", np.nan) for s in uniq], float)
    good = np.isfinite(rv) & np.isfinite(cv_)
    if good.sum() >= 3 and np.ptp(rv[good]) > 0.0:
        A = np.column_stack([np.ones(good.sum()), rv[good]])
        coef, *_ = np.linalg.lstsq(A, cv_[good], rcond=None)
        resid = cv_[good] - A @ coef
        sst = float(((cv_[good] - cv_[good].mean()) ** 2).sum())
        slope.update(omega_from_cbar=float(coef[0]), zeta_from_cbar=float(coef[1]),
                     r2=float(1.0 - (resid @ resid) / sst) if sst > 0 else float("nan"),
                     rbar_f_span=float(np.ptp(rv[good])))

    identified = bool(np.isfinite(pooled["bkw_cond"]) and pooled["bkw_cond"] <= float(cond_max))
    return dict(starts=uniq, n_starts=int(len(uniq)), per_start=per, pooled=pooled,
                pooled_corr=pooled["corr"], pooled_bkw_cond=pooled["bkw_cond"],
                ratio_cv=ratio_cv, ratio_cv_pct=100.0 * ratio_cv,
                ridge_cond_max=float(cond_max), identified_split=identified,
                cbar_slope_across_starts=slope)


def cbar_stats(fit, rbar, boot_draws=None, sub_thetas=None, n_firms=None, b_firms=None,
               levels=(0.90, 0.95), by_start=None):
    r"""Point estimate and inference for c̄ = ω + r̄^f·ζ.

    Two inference routes, mirroring exactly what this file already does for ω and ζ separately:
      * `c_bar_se_boot` — SD over firm-block bootstrap draws. Directly comparable to
        `omega_se`/`zeta_se`, and like them RETAINED FOR COMPARISON ONLY: the nonparametric
        bootstrap is inconsistent for a set-identified, kinked criterion (Tamer 2003).
      * `c_bar_sd_rate_adj` / `c_bar_ci_sqrtn` — the subsampling route, rate-adjusted by
        √(b/n), matching `theta_sd_rate_adj` / `param_ci_sqrtn`. This is the headline.

    Because c̄ is a scalar LINEAR functional of θ, both are exact transformations of the draws
    already computed — no extra refits. The gain over quoting ω̂ and ζ̂ separately is real:
    their marginal SDs are inflated by the ridge, whereas c̄ is the direction the data pins.

    `by_start` (a ridge_by_start()["per_start"] map) adds `c_bar_by_start`: the POOLED (ω̂, ζ̂)
    evaluated at each launch quarter's own r̄^f, i.e. the fitted marginal cost across the rate
    cycle. It is deliberately NOT ridge_by_start's per-start c̄, which refits inside the start;
    comparing the two is how a reader sees whether one pooled (ω, ζ) reproduces the level each
    start identifies on its own.
    """
    out = dict(c_bar=float(fit["omega"] + rbar * fit["zeta"]))
    if by_start:
        out["c_bar_by_start"] = {
            s: float(fit["omega"] + d["rbar_f"] * fit["zeta"])
            for s, d in by_start.items() if np.isfinite(d.get("rbar_f", np.nan))}
    if not np.isfinite(rbar):
        return out

    if boot_draws is not None and len(boot_draws):
        cb = boot_draws[:, 0] + rbar * boot_draws[:, 1]
        out["c_bar_se_boot"] = float(cb.std())
        with np.errstate(invalid="ignore"):
            cc = np.corrcoef(boot_draws[:, 0], boot_draws[:, 1])[0, 1]
        out["omega_zeta_corr_boot"] = float(cc) if np.isfinite(cc) else None

    if sub_thetas is not None and len(sub_thetas) and n_firms and b_firms:
        TH = np.asarray(sub_thetas, float)
        cb = TH[:, 0] + rbar * TH[:, 1]
        cn = out["c_bar"]
        out["c_bar_sd_rate_adj"] = float(cb.std() * np.sqrt(b_firms / n_firms))
        scaled = np.sqrt(b_firms) * (cb - cn)
        ci = {}
        for lv in levels:
            a = 1.0 - lv
            lo = cn - float(np.quantile(scaled, 1.0 - a / 2.0)) / np.sqrt(n_firms)
            hi = cn - float(np.quantile(scaled, a / 2.0)) / np.sqrt(n_firms)
            ci[f"{lv:.2f}"] = dict(lo=lo, hi=hi)
        out["c_bar_ci_sqrtn"] = ci
    return out


# ==========================================================================
# Firm-block bootstrap for inference
# ==========================================================================
def bootstrap_kappa(blk, n_boot, seed=42):
    """Resample FIRMS with replacement (block bootstrap), refit, collect params.

    THE DEPENDENCE UNIT IS THE FIRM, INCLUDING ACROSS STARTS. `idx_by_firm` collects every row
    of a firm — all of its (start × deviation) pairs — so a firm moves in or out whole. Starts
    are NEVER resampled independently: a firm's 2016Q1 and 2016Q3 rows are the same firm walked
    forward through one Selic cycle, not two independent draws, and treating them as such would
    manufacture precision out of the serial correlation the multi-start design deliberately
    introduces. It would also break the design in the other direction — the cross-start spread
    in r̄^f is exactly what identifies ζ, and a start-level resample would shuffle it.

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
        draws.append(solve_kappa(_sub_block(blk, rows))["theta"])
    arr = np.vstack(draws)
    # Return the DRAWS as well as the SDs. ω and ζ are near-perfectly collinear in this design
    # (see rbar_of_block), so their marginal SDs say nothing about the precision of the one
    # combination that IS identified, ω + r̄^f·ζ. That needs the joint draws.
    return arr.std(axis=0), arr


# ==========================================================================
# Diagnostics: 1-D identified-set profiles and multistart smoke check
# ==========================================================================
def _theta_to_b(omega, zeta, gamma, sc):
    """Map REPORTED (ω, ζ, γ) into a given design's scaled b-coordinates.

    Inverse of solve_kappa's unscaling: b = [ω·s_om/s1, γ_z·s_ga_z/s1 …, (1+ζ)·s_ze/s1].
    Needed because every subsample re-conditions its own design, so the full-sample θ̂ has
    to be re-expressed in that subsample's scale before its criterion can be evaluated.
    """
    nZ = sc["nZ"]
    b = np.empty(2 + nZ)
    b[0] = omega * sc["s_om"] / sc["s1"]
    if nZ:
        b[1:1 + nZ] = np.asarray(gamma, float) * sc["s_ga"] / sc["s1"]
    b[1 + nZ] = (1.0 + zeta) * sc["s_ze"] / sc["s1"]
    return b


def _rows_for_firms(idx_by_firm, sel):
    return np.concatenate([idx_by_firm[f] for f in sel])


def _sub_block(blk, rows):
    """Row subset of a Δψ block. `starts` rides along so any sub-block stays a full block —
    ridge_by_start refits on subsets and needs the labels to line up with the rows."""
    return dict(d1=blk["d1"][rows], d_omega=blk["d_omega"][rows],
                d_gamma=blk["d_gamma"][rows], d_zeta=blk["d_zeta"][rows],
                firms=blk["firms"][rows], starts=blk["starts"][rows],
                gamma_names=blk["gamma_names"])


def subsample_kappa(blk, fit, n_sub=200, b_firms=None, seed=4242, levels=(0.90, 0.95)):
    """Subsampling inference for the eq:17 criterion (BBL 2007; Chernozhukov–Hong–Tamer 2007).

    WHY NOT THE BOOTSTRAP.  F is a squared hinge over moment INEQUALITIES: kinked, and
    potentially SET-identified.  The nonparametric bootstrap is inconsistent for
    set-identified / boundary parameters and does not reproduce the non-standard limit of a
    kinked criterion.  BBL propose subsampling; CHT invert the criterion with a subsampled
    critical value.  This is cheap here because ψ is PRECOMPUTED — only the fast,
    linear-in-θ hinge minimisation is re-run.

    PROCEDURE.  Draw `n_sub` subsamples of `b_firms` FIRMS without replacement — the firm is
    the dependence unit, and it stays the dependence unit under multi-start ψ: a firm's ψ_eq is
    reused across all of its deviations, and its rows at different launch quarters are one firm
    followed through the rate cycle, not independent draws.  Subsampling STARTS instead (or as
    well) would both understate dependence and destroy the cross-start spread in r̄^f that
    identifies ζ.  With
    Q(θ) = F(θ)/n_rows the mean squared violation (comparable across sample sizes),

        T_s = b · [ Q_b(θ̂_n) − min_θ Q_b(θ) ]

    i.e. how much worse the FULL-sample estimate does on subsample s than that subsample's
    own optimum.  The (1−α) quantile of {T_s} is the critical value that profile_ci inverts.

    Also returns a parameter-quantile CI built under a √n rate, FOR COMPARISON ONLY — that
    rate assumption is exactly what set identification can break, so it is not the headline.

    ⚠ This is inference CONDITIONAL ON THE FIRST STAGE.  δ̂, θ̂₂, α̂, the Step-1 policy, φ̂ and
    M are held fixed; ψ is not re-simulated.  Demand uncertainty is NOT propagated here.
    """
    firms = blk["firms"]
    uniq = np.unique(firms)
    n = int(len(uniq))
    if n < 8:
        return dict(skipped=f"only {n} firms — subsampling not meaningful")
    if b_firms is None:                       # b = n^(2/3): b → ∞, b/n → 0
        b_firms = int(round(n ** (2.0 / 3.0)))
    b_firms = int(np.clip(b_firms, 5, n - 1))

    idx_by_firm = {f: np.where(firms == f)[0] for f in uniq}
    rng = np.random.default_rng(seed)

    om_n, ze_n = fit["omega"], fit["zeta"]
    ga_n = np.asarray(list(fit["gamma"].values()), float)
    theta_n = np.asarray(fit["theta"], float)

    T, thetas = [], []
    for _ in range(n_sub):
        sel = rng.choice(uniq, size=b_firms, replace=False)
        rows = _rows_for_firms(idx_by_firm, sel)
        fb = solve_kappa(_sub_block(blk, rows))
        nrb = len(rows)
        b_full = _theta_to_b(om_n, ze_n, ga_n, fb["_scales"])
        Q_at_full = _obj(b_full, fb["_c1"], fb["_Xs"]) / nrb
        Q_min = fb["objective"] / nrb
        T.append(b_firms * max(Q_at_full - Q_min, 0.0))
        thetas.append(fb["theta"])

    T = np.asarray(T, float)
    TH = np.vstack(thetas)
    crit = {f"{lv:.2f}": float(np.quantile(T, lv)) for lv in levels}

    # √n-rate parameter-quantile CI (comparison only)
    scaled = np.sqrt(b_firms) * (TH - theta_n)
    param_ci = {}
    for lv in levels:
        a = 1.0 - lv
        q_lo = np.quantile(scaled, a / 2.0, axis=0)
        q_hi = np.quantile(scaled, 1.0 - a / 2.0, axis=0)
        param_ci[f"{lv:.2f}"] = dict(lo=(theta_n - q_hi / np.sqrt(n)).tolist(),
                                     hi=(theta_n - q_lo / np.sqrt(n)).tolist())

    return dict(n_firms=n, b_firms=b_firms, n_sub=int(n_sub),
                crit=crit, param_ci_sqrtn=param_ci,
                theta_sd_rate_adj=(TH.std(axis=0) * np.sqrt(b_firms / n)).tolist(),
                T_median=float(np.median(T)), T_max=float(T.max()),
                # kept (not serialized) so cbar_stats can form the subsampling CI for the
                # identified combination without re-running the n_sub refits.
                _thetas=TH)


def _profile_min_F(b_idx, b_val, c1, Xs, b0):
    """min F over the OTHER coefficients holding b[b_idx] fixed — a TRUE profile."""
    dim = len(b0)
    free = np.array([i for i in range(dim) if i != b_idx], dtype=int)
    if free.size == 0:
        b = b0.copy(); b[b_idx] = b_val
        return _obj(b, c1, Xs)

    def f(bf):
        b = b0.copy(); b[b_idx] = b_val; b[free] = bf
        return _obj(b, c1, Xs)

    def gr(bf):
        b = b0.copy(); b[b_idx] = b_val; b[free] = bf
        return _grad(b, c1, Xs)[free]

    res = minimize(f, b0[free], jac=gr, method="L-BFGS-B", bounds=None,
                   options=dict(maxiter=5000, ftol=1e-14, gtol=1e-10))
    return float(res.fun)


def profile_ci(blk, fit, which, crit_value, npts=81, max_expand=6, min_inside=11, max_refine=4):
    """TRUE profile of the criterion in ω or ζ, inverted at a subsampled critical value.

    Two fixes over the earlier `profile_param`:
      * it RE-MINIMISES over the nuisance coefficients at each grid point (a profile). The
        old routine held them fixed — a SLICE, which makes F rise faster and therefore
        UNDERSTATES the identified width.
      * the window no longer scales with the bootstrap SE it is meant to replace; it starts
        from |opt| and expands geometrically until the region closes inside the grid.

    Reports `truncated_lo/hi` so a window-limited interval is never mistaken for a set
    boundary.  Statistic matches subsample_kappa:  T(θ) = n_firms·[Q(θ) − Q(θ̂)].
    """
    sc = fit["_scales"]; nZ = sc["nZ"]
    c1 = fit["_c1"]; Xs = fit["_Xs"]; b0 = np.asarray(fit["_b"], float)
    n_rows = len(c1)
    n_firms = int(len(np.unique(blk["firms"])))
    F_hat = float(fit["objective"])

    if which == "omega":
        idx, opt = 0, fit["omega"]
        to_b = lambda v: v * sc["s_om"] / sc["s1"]
    elif which == "zeta":
        idx, opt = 1 + nZ, fit["zeta"]
        to_b = lambda v: (v + 1.0) * sc["s_ze"] / sc["s1"]
    else:
        raise ValueError(which)

    def T_of(v):
        F = _profile_min_F(idx, to_b(v), c1, Xs, b0)
        return n_firms * (F - F_hat) / n_rows

    hw = max(abs(float(opt)), 1.0)
    grid = T = None
    for _ in range(max_expand):
        grid = np.linspace(opt - hw, opt + hw, npts)
        T = np.array([T_of(v) for v in grid])
        inside = grid[T <= crit_value]
        if inside.size == 0:
            break
        lo, hi = float(inside.min()), float(inside.max())
        if lo > grid[0] + 1e-12 and hi < grid[-1] - 1e-12:
            break                                   # region closed inside the window
        hw *= 3.0

    inside = grid[T <= crit_value]
    if inside.size == 0:
        return dict(which=which, crit=float(crit_value), ci_lo=None, ci_hi=None,
                    empty=True, half_width=float(hw),
                    grid=grid.tolist(), T=T.tolist())
    lo, hi = float(inside.min()), float(inside.max())
    truncated_lo = bool(lo <= grid[0] + 1e-12)
    truncated_hi = bool(hi >= grid[-1] - 1e-12)
    # REFINE. The loop above only ever WIDENS the window, so its resolution is 2*hw/(npts-1) and
    # never finer than 2/(npts-1), because hw is floored at 1.0. A region narrower than one step
    # contains a single grid point -- the optimum, which sits at the grid's centre -- and came
    # back as a ZERO-WIDTH interval, exactly when the design identifies the split well. So
    # re-grid on the bracket between the nearest outside points until the region spans at least
    # `min_inside` points. A truncated region is left alone: its problem is the window, not the
    # resolution.
    refined = 0
    while (inside.size < min_inside and refined < max_refine
           and not (truncated_lo or truncated_hi)):
        i_lo = int(np.flatnonzero(grid >= lo)[0])
        i_hi = int(np.flatnonzero(grid <= hi)[-1])
        g2 = np.linspace(grid[max(i_lo - 1, 0)], grid[min(i_hi + 1, grid.size - 1)], npts)
        T2 = np.array([T_of(v) for v in g2])
        in2 = g2[T2 <= crit_value]
        refined += 1
        if in2.size == 0:
            break
        grid, T, inside = g2, T2, in2
        lo, hi = float(inside.min()), float(inside.max())
    return dict(which=which, crit=float(crit_value), ci_lo=lo, ci_hi=hi, empty=False,
                truncated_lo=truncated_lo, truncated_hi=truncated_hi,
                half_width=float(hw), grid_step=float(grid[1] - grid[0]),
                n_inside=int(inside.size), refined=int(refined), zero_width=bool(hi <= lo),
                grid=grid.tolist(), T=T.tolist())


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


# ==========================================================================
# Reporting: everything below must survive in the SLURM log
# ==========================================================================
# NOTHING IS DOWNLOADED FROM THE CLUSTER, so a number that exists only inside a parquet or a
# json on the compute node's filesystem does not exist for the paper. Every diagnostic this
# file computes is therefore printed as well as stored, and printed in full (one line per
# launch quarter, not a summary), so the run can be read back from the .out alone.
def _jsonable(x):
    """Recursively turn non-finite floats into None so the json stays strict JSON.

    Applied to the multi-start blocks only. The older per-type fields keep their NaNs: their
    readers (make_bbl_cost_tables.py's float(blk.get(...)) and cf_psi_basis.jl's JSON3 +
    Float64) accept a NaN and would fail on a null, so silently changing their type here would
    break the table build rather than protect it.
    """
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def _fmt(v, spec):
    """Format a possibly-missing number; a missing one still fills the column width, so the
    per-start table stays aligned when a start has too few rows to diagnose."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = float("nan")
    if np.isfinite(x):
        return format(x, spec)
    w = re.match(r"^[<>^]?(\d+)", spec)
    return format("-", f">{w.group(1)}") if w else "-"


def _print_start_table(kappa, rbs, rf_bar_beta=None, rf_source=None):
    """The per-start ridge table, one line per launch quarter plus the pooled line."""
    rf_bar_beta = rf_bar_beta or {}
    rf_source = rf_source or {}
    print(f"  [{kappa}] per-start ridge diagnostics -- rows are (firm x start x deviation); "
          f"the dependence unit is the FIRM, starts are never resampled")
    print(f"    {'start':<9}{'n':>8}{'firms':>7}{'corr(d2,d4)':>14}{'BKW cond':>11}"
          f"{'rbar_f':>11}{'rf_bar_beta':>13}{'frac_bind':>11}{'c_bar':>15}  rf_source")
    for s in rbs["starts"] + ["POOLED"]:
        d = rbs["pooled"] if s == "POOLED" else rbs["per_start"][s]
        src = "" if s == "POOLED" else ",".join(rf_source.get(s, []))
        rfb = None if s == "POOLED" else rf_bar_beta.get(s)
        print(f"    {s:<9}{d['n']:>8d}{d['n_firms']:>7d}{_fmt(d.get('corr'), '>14.6f')}"
              f"{_fmt(d.get('bkw_cond'), '>11.4g')}{_fmt(d.get('rbar_f'), '>11.6f')}"
              f"{_fmt(rfb, '>13.6f')}{_fmt(d.get('frac_bind'), '>11.3f')}"
              f"{_fmt(d.get('c_bar'), '>15.6g')}  {src}")
    print(f"    ratio CV of the per-start mean Dpsi4/Dpsi2 across starts = "
          f"{_fmt(rbs.get('ratio_cv_pct'), '.2f')}%"
          f"  |  pooled BKW cond = {_fmt(rbs.get('pooled_bkw_cond'), '.4g')}"
          f" (max {rbs['ridge_cond_max']:.4g})  ->  identified_split="
          f"{rbs['identified_split']}")
    sl = rbs.get("cbar_slope_across_starts") or {}
    if "zeta_from_cbar" in sl:
        print(f"    cross-check, OLS of c_bar_s on rbar_f_s over {sl['n_starts']} starts: "
              f"omega={sl['omega_from_cbar']:.6g} zeta={sl['zeta_from_cbar']:.6g} "
              f"R2={sl['r2']:.4f} (rbar_f span {sl['rbar_f_span']:.6f}) -- these use only "
              f"within-start-identified quantities, so they should track the pooled fit")


def _print_final(kappa, rec, level):
    """Point estimates and intervals, spelled out. This is the deliverable of the whole step."""
    print(f"  [{kappa}] FINAL PARAMETERS -- estimand={rec['estimand']} "
          f"(identified_split={rec['identified_split']}, "
          f"n_firms={rec['n_firms']}, n_rows={rec['n_rows']}, n_starts={rec['n_starts']})")
    if rec.get("objective_zero"):
        print(f"        !! criterion == 0: set-identified, the point below is arbitrary"
              f"{' (the optimiser start)' if rec.get('at_init') else ''}")
    print(f"        omega  = {rec['omega']:.10g}   (se {_fmt(rec.get('omega_se'), '.4g')})")
    print(f"        zeta   = {rec['zeta']:.10g}   (se {_fmt(rec.get('zeta_se'), '.4g')})")
    for z, v in (rec.get("gamma") or {}).items():
        print(f"        gamma[{z}] = {float(v):.10g}   "
              f"(se {_fmt((rec.get('gamma_se') or {}).get(z), '.4g')})")
    print(f"        rbar_f = {_fmt(rec.get('rbar_f'), '.10g')}   "
          f"c_bar = {rec['c_bar']:.10g}   "
          f"(sd_rate_adj {_fmt(rec.get('c_bar_sd_rate_adj'), '.4g')})")
    lvl = f"{level:.2f}"
    ci = (rec.get("c_bar_ci_sqrtn") or {}).get(lvl)
    if ci:
        print(f"        c_bar CI({lvl}) = [{ci['lo']:.10g}, {ci['hi']:.10g}]")
    for nm, sym in (("omega", "omega"), ("zeta", "zeta")):
        p = rec.get(f"ci_{nm}")
        if p and not p["empty"]:
            print(f"        {sym} profile CI({lvl}) = [{p['ci_lo']:.10g}, {p['ci_hi']:.10g}]")
    if rec["estimand"] == "c_bar":
        print(f"        omega and zeta above are a point ON the ridge, not separately "
              f"identified -- report c_bar.")


# (N, present, missing) for a list of psi_dev paths, read from their names alone. ONE
# implementation, shared with the sweep job that re-runs missing shards (bbl_job.sh
# BBL_STEP=sweep -> bbl_shards.py coverage), so "complete" means the same thing to the step that
# repairs a gap and to the step that refuses one. The caller refuses on `missing`: E1's costs were
# once estimated from 293 of 300 shards (2026-09-19) with nothing on screen to show it.
from bbl_shards import shard_coverage as _shard_coverage, compress_ranges as _compress_ranges
# (beta, T, source) the psi were simulated under -- from psi_starts_<tag>.json, else the
# bbl_run.sh launch record. Recorded in the run block of cost_params so every table built from it
# can state its discounting from provenance rather than from a constant in the table code.
from bbl_shards import psi_discount as _psi_discount, read_bbl_discount as _read_bbl_discount


def main():
    ap = argparse.ArgumentParser(description="BBL Step 2 cost solver (V_Main eq:17).")
    ap.add_argument("--estim", type=int, default=6)
    ap.add_argument("--spec", type=int, default=12)
    ap.add_argument("--stage", type=str, default="extended")
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--psi-tag", type=str, default="",
                    help="marker the forward sim stamped on the psi filenames for a multi-start "
                         "run (bbl_run.sh stamps _ms{P}, P = rate paths); appended to the tag when LOCATING the "
                         "parquets and kept on the tagged cost_params name. Empty = the "
                         "single-start naming.")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="firm-block bootstrap reps (COMPARISON ONLY -- the bootstrap is not "
                         "valid for this kinked/possibly set-identified criterion; 0 to skip)")
    ap.add_argument("--subsample", type=int, default=200,
                    help="subsampling reps for the HEADLINE inference (BBL/CHT); 0 to skip")
    ap.add_argument("--subsample-b", type=int, default=None,
                    help="subsample size in FIRMS (default n^(2/3))")
    ap.add_argument("--ci-level", type=float, default=0.95,
                    help="confidence level for the subsampled criterion inversion")
    ap.add_argument("--profile", action="store_true",
                    help="invert the criterion into omega/zeta confidence intervals (true "
                         "profile, re-minimising nuisance coefficients; needs --subsample > 0)")
    ap.add_argument("--promote", action="store_true",
                    help="copy the tagged cost_params onto the UNTAGGED name the "
                         "counterfactuals read, but only if the identification gate passes; "
                         "otherwise exit 3 so the afterok dependents are cancelled")
    ap.add_argument("--ridge-cond-max", type=float, default=30.0,
                    help="gate threshold: the pooled BKW condition index of [Dpsi2, Dpsi4] "
                         "must be <= this for the omega/zeta split to count as identified "
                         "(30 is the BKW rule of thumb; 0.99 corr is about 20)")
    ap.add_argument("--force-ridge", action="store_true",
                    help="promote even if the gate fails. The json records forced=true; the "
                         "downstream counterfactuals then run on an omega/zeta split the "
                         "design does not identify, and only c_bar is defensible.")
    args = ap.parse_args()

    # base_tag names what the counterfactuals read; tag names this run's psi and its json.
    base_tag = f"E{args.estim}_spec_{args.spec}_{args.stage}{args.suffix}"
    tag = f"{base_tag}{args.psi_tag}"
    eq_path = COST_FWD / f"psi_eq_{tag}.parquet"
    if not eq_path.exists():
        raise FileNotFoundError(f"Missing {eq_path.name} -- run bbl_fwd_sim.jl first.")

    # Deviation psi may be a single file (n-shards=1) or several shard files; merge all. The
    # glob keeps its trailing wildcard, but only the writer's two spellings are accepted after
    # the tag ("" and "_shard{i}of{N}"): with --psi-tag empty the wildcard would otherwise also
    # swallow the multi-start files sitting in the same folder (psi_dev_<base>_ms8_shard*),
    # mixing two vintages of psi into one solve with nothing on screen to show for it.
    shard_ok = re.compile(r"^(?:_shard\d+of\d+)?$")
    pre = f"psi_dev_{tag}"
    all_dev = sorted(COST_FWD.glob(f"{pre}*.parquet"))
    dev_files = [p for p in all_dev if shard_ok.match(p.stem[len(pre):])]
    if not dev_files:
        raise FileNotFoundError(f"No psi_dev_{tag}*.parquet -- run bbl_fwd_sim.jl "
                                f"(all shards) first.")
    other = [p.name for p in all_dev if p not in dev_files]
    if other:
        print(f"  Ignoring {len(other)} psi_dev file(s) carrying a different --psi-tag: "
              f"{', '.join(other[:4])}{' ...' if len(other) > 4 else ''}")

    # ONE run's shards, never two. The tag carries P (rate paths) but not the start set or the
    # shard count, so two designs sharing a tag write into the same names; a different
    # --n-shards leaves BOTH shard families on disk, and the glob above would pool them.
    # Refuse that here rather than let the dedup below quietly average two designs.
    n_of = set()
    for p in dev_files:
        m = re.search(r"_shard\d+of(\d+)$", p.stem)
        n_of.add(int(m.group(1)) if m else 0)
    if len(n_of) > 1:
        raise SystemExit(
            f"REFUSING: psi_dev_{tag}* holds shards from more than one run (shard counts "
            f"{sorted(n_of)}; 0 = an unsharded file). Two designs share the tag "
            f"'{args.psi_tag}'. Remove the stale family or re-run under a distinct --psi-tag.")

    # EVERY shard, or none of them. See _shard_coverage: a shard file appears only when its whole
    # slice finished, so a gap is missing SIMULATION, not a missing file, and averaging over what
    # survived silently reports a subsample as the estimate.
    n_shards, shards_present, shards_missing = _shard_coverage(dev_files)
    if shards_missing:
        miss = _compress_ranges(shards_missing)
        raise SystemExit(
            f"REFUSING: psi_dev_{tag}* is missing {len(shards_missing)} of {n_shards} shards: "
            f"{miss}\n"
            f"  Those (firm x shock) deviations were never simulated -- the solve would report a "
            f"subsample as the estimate.\n"
            f"  The sweep job re-runs gaps by itself; a solve that reaches this line ran without "
            f"it (--no-sweep) or after its retries were spent. Fill exactly these shards and "
            f"re-chain solve -> tables -> archive with one command (BBL_RUNBOOK.md, recovery):\n"
            f"    bash bbl_run.sh --routines {args.estim} --shards {n_shards} "
            f"--psi-tag '{args.psi_tag}' --repair")

    eq = pd.read_parquet(eq_path)
    dev = pd.concat([pd.read_parquet(f) for f in dev_files], ignore_index=True)
    # A single-start run has no start_q column. Injecting "all" here rather than branching later
    # is what keeps ONE code path: alignment, the per-start table, the gate and the json all see
    # a start dimension that happens to have size 1, and report it as such.
    for df in (eq, dev):
        if "start_q" not in df.columns:
            df["start_q"] = "all"
        df["start_q"] = df["start_q"].astype(str)
    # Same shard count but a different start set is the other way two runs collide under one tag.
    extra = sorted(set(dev["start_q"]) - set(eq["start_q"]))
    if extra:
        raise SystemExit(
            f"REFUSING: psi_dev rows carry launch quarters psi_eq_{tag} does not "
            f"({', '.join(extra[:6])}{' ...' if len(extra) > 6 else ''}). They come from "
            f"different runs under one tag; re-run the forward simulation.")
    # Drop any accidental duplicate (firm, start, shock) rows from re-runs.
    dedup = [c for c in ("firm", "start_q", "shock") if c in dev.columns]
    dev = dev.drop_duplicates(subset=dedup).reset_index(drop=True)

    starts_all = sorted(eq["start_q"].unique().tolist())
    rf_source_by_start, rf_bar_beta_by_start = {}, {}
    if "rf_source" in eq.columns:
        rf_source_by_start = {str(s): sorted(set(g.astype(str)))
                              for s, g in eq.groupby("start_q")["rf_source"]}
    if "rf_bar_beta" in eq.columns:
        rf_bar_beta_by_start = {str(s): float(np.nanmedian(g.astype(float)))
                                for s, g in eq.groupby("start_q")["rf_bar_beta"]}
    rf_sources = sorted({v for vs in rf_source_by_start.values() for v in vs})

    print(f"  tag={tag}  (psi_tag={args.psi_tag or '(none)'}, promoted name "
          f"cost_params_{base_tag}.json)")
    print(f"  Loaded psi_eq ({len(eq)} firm x start rows) and psi_dev ({len(dev)} "
          f"firm x start x shock rows) from {len(dev_files)} shard file(s)"
          + (f" -- shards {len(shards_present)}/{n_shards}, complete" if n_shards > 1 else ""))
    print(f"  starts={len(starts_all)}: {', '.join(starts_all[:8])}"
          f"{' ...' if len(starts_all) > 8 else ''}"
          f"  |  rf_source(s): {', '.join(rf_sources) if rf_sources else '(not recorded)'}")
    # Provenance only: the solve never uses beta or T (the psi are already discounted sums).
    psi_beta, psi_T, disc_src = _psi_discount(COST_FWD, tag)
    print(f"  psi simulated at beta={psi_beta if psi_beta is not None else '?'} "
          f"T={psi_T if psi_T is not None else '?'}  <- {disc_src}")
    try:
        _reg = _read_bbl_discount()
        if psi_beta is not None and (abs(psi_beta - _reg["BBL_BETA"]) > 1e-12
                                     or psi_T != _reg["BBL_HORIZON"]):
            print(f"  NOTE: bbl_discount.env now says beta={_reg['BBL_BETA']} "
                  f"T={_reg['BBL_HORIZON']}; these psi were simulated under beta={psi_beta} "
                  f"T={psi_T}. The json records the psi's own values.")
    except (OSError, KeyError, ValueError):
        pass

    blocks = build_delta(eq, dev)
    results = {}
    ident = {}
    for kappa, blk in blocks.items():
        nZ = blk["d_gamma"].shape[1]
        fit = solve_kappa(blk)
        if args.bootstrap > 0:
            _se, boot_draws = bootstrap_kappa(blk, args.bootstrap)
            se = _se.tolist()
        else:
            se, boot_draws = [None] * (2 + nZ), None
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
            objective_zero=fit["objective_zero"], at_init=fit["at_init"],
            multistart_max_dF=float(ms_max_dF),
            n_firms=int(np.unique(blk["firms"]).size),
            n_rows=int(blk["d1"].size),
        )

        # ---- Subsampling: the headline inference for this criterion --------------------
        sub = subsample_kappa(blk, fit, n_sub=args.subsample,
                              b_firms=args.subsample_b) if args.subsample > 0 else None
        if sub is not None:
            # `_thetas` is a private ndarray handle for cbar_stats -- never serialize it.
            rec["subsample"] = {k: v for k, v in sub.items() if not k.startswith("_")}

        # ---- Is the omega/zeta split identified in THIS design? ------------------------
        # Pooled ridge geometry as before, plus the per-start decomposition that says whether
        # the launch quarters bought any spread in rbar_f. The verdict drives which estimand is
        # reported and whether --promote is allowed to overwrite the counterfactuals' input.
        rbs = ridge_by_start(blk, fit=fit, cond_max=args.ridge_cond_max)
        rec["n_starts"] = rbs["n_starts"]
        rec["by_start"] = _jsonable(rbs)
        rec["identified_split"] = rbs["identified_split"]
        rec["estimand"] = "omega_zeta" if rbs["identified_split"] else "c_bar"
        ident[kappa] = dict(identified_split=rbs["identified_split"],
                            pooled_bkw_cond=rbs["pooled_bkw_cond"],
                            pooled_corr=rbs["pooled_corr"],
                            ratio_cv=rbs["ratio_cv"], n_starts=rbs["n_starts"],
                            n_firms=rec["n_firms"], n_rows=rec["n_rows"],
                            objective_zero=rec["objective_zero"], at_init=rec["at_init"])
        if rec["objective_zero"]:
            print(f"  [{kappa}] !!!! the criterion is EXACTLY 0 at the optimum: every inequality "
                  f"holds, so omega/zeta are SET-identified and the reported point is arbitrary"
                  f"{' -- it is the optimiser start b=0 (omega=0, zeta=-1, gamma=0)' if rec['at_init'] else ''}"
                  f". It will not be promoted. !!!!")

        # ---- The identified combination c_bar = omega + rbar_f*zeta --------------------
        # When the split is NOT identified, c_bar is the estimand: it is the direction the data
        # pins whatever the design does with the split, so it is computed either way.
        ridge = rbar_of_block(blk)
        rec.update(ridge)
        rec.update(cbar_stats(
            fit, ridge["rbar_f"], boot_draws=boot_draws,
            sub_thetas=(sub or {}).get("_thetas"),
            n_firms=(sub or {}).get("n_firms"), b_firms=(sub or {}).get("b_firms"),
            by_start=rbs["per_start"]))

        # ---- Invert the criterion into omega/zeta confidence intervals -----------------
        if args.profile:
            lvl = f"{args.ci_level:.2f}"
            crit = (sub or {}).get("crit", {}).get(lvl)
            if crit is None:
                print(f"  [{kappa}] --profile needs --subsample > 0 for a critical value; "
                      f"skipped.")
            else:
                rec["ci_omega"] = profile_ci(blk, fit, "omega", crit)
                rec["ci_zeta"] = profile_ci(blk, fit, "zeta", crit)
        elif rbs["identified_split"]:
            print(f"  [{kappa}] the omega/zeta split IS identified here but --profile was not "
                  f"passed, so the headline parameters ship without their criterion-inversion "
                  f"intervals. Re-run with --profile --subsample N.")

        results[kappa] = rec

        se0 = f"{se[0]:.3g}" if se[0] is not None else "-"
        se1 = f"{se[1]:.3g}" if se[1] is not None else "-"
        diag = " | omega<0 (misspecification diagnostic -- see docstring)" if fit["omega"] < 0 else ""
        print(f"  [{kappa}] omega={fit['omega']:.4g} (se {se0}) | "
              f"zeta={fit['zeta']:.4g} (se {se1}) | obj={fit['objective']:.4g} | "
              f"bind={100*fit['frac_bind']:.0f}% | firms={rec['n_firms']} | "
              f"starts={rbs['n_starts']}")
        print(f"        opt: success={fit['opt_success']} status={fit['opt_status']} "
              f"nit={fit['opt_nit']} grad_inf={fit['opt_grad_inf']:.2e} "
              f"ms_dF={ms_max_dF:.2e}{diag}")
        if ms_max_dF > 1e-6:
            print(f"  !!!! [{kappa}] MULTISTART DISAGREEMENT max dF={ms_max_dF:.3e} > 1e-6 -- "
                  f"convex hinge should have a unique optimum; suspect a gradient/scaling bug !!!!")
        if sub is not None and "skipped" not in sub:
            print(f"        subsample: b={sub['b_firms']}/{sub['n_firms']} firms, "
                  f"{sub['n_sub']} reps | crit({args.ci_level:.2f})="
                  f"{sub['crit'][f'{args.ci_level:.2f}']:.4g}")
        elif sub is not None:
            print(f"        subsample: {sub['skipped']}")

        _print_start_table(kappa, rbs, rf_bar_beta_by_start, rf_source_by_start)

        # Multi-start psi whose pooled columns are still as collinear as one curve is a WIRING
        # failure upstream, not a marginal result: 36 vintage curves crossed with the deposit
        # paths measured corr 0.852, against 0.99997 for a single curve. Report it here rather
        # than letting the gate refusal below be read as "the idea did not work".
        pc = rbs["pooled_corr"]
        if rbs["n_starts"] > 1 and np.isfinite(pc) and abs(pc) > 0.999:
            print(f"  !!!! [{kappa}] {rbs['n_starts']} starts BUT pooled corr(Dpsi2,Dpsi4)="
                  f"{pc:.6f} -- as collinear as a single curve (reference: 36 vintage curves "
                  f"= 0.852). The starts are not carrying different forward curves. Suspect the "
                  f"driver (row_start / rf_curve_paths not reaching psi_under, or one curve "
                  f"reused for every start) before reading anything below as a result. !!!!")
        if rbs["n_starts"] > 1 and np.isfinite(rbs["ratio_cv"]) and rbs["ratio_cv"] < 0.02:
            print(f"  !!!! [{kappa}] the per-start mean Dpsi4/Dpsi2 varies by only "
                  f"{100*rbs['ratio_cv']:.3f}% across {rbs['n_starts']} starts -- the launch "
                  f"quarters span almost no rate variation, so zeta has nothing to load on !!!!")

        for nm in ("omega", "zeta"):
            ci = rec.get(f"ci_{nm}")
            if not ci:
                continue
            if ci["empty"]:
                print(f"        CI {nm}: EMPTY at the {args.ci_level:.0%} level "
                      f"(criterion never within crit -- check identification)")
            else:
                trunc = ("  !! WINDOW-TRUNCATED (not a set boundary)"
                         if (ci["truncated_lo"] or ci["truncated_hi"]) else "")
                print(f"        CI {nm}: [{ci['ci_lo']:.4g}, {ci['ci_hi']:.4g}] "
                      f"width={ci['ci_hi']-ci['ci_lo']:.4g}{trunc}")
        if 0.45 <= fit["frac_bind"] <= 0.55:
            print(f"  !!!! [{kappa}] frac_bind={fit['frac_bind']:.3f} -- suspiciously close to "
                  f"1/2. A symmetric +/- grid around a sigma-hat that is NOT a turning point of "
                  f"the simulated value makes exactly one of each +/- pair bind for ANY theta, "
                  f"so the moments carry little/no identifying content and NOTHING below (point "
                  f"estimates, SEs, CIs) should be read as a result. Deviate around the FITTED "
                  f"policy (--policy-csv). !!!!")

    # ---- Gate verdict, json, promotion -------------------------------------------------
    # The gate is ALL-TYPES: cf3 solves the equilibrium with B and D costs together, so a split
    # identified for one type and not the other is not a usable promotion.
    # ...and a type whose criterion is exactly 0 has no point estimate to promote, whatever the
    # ridge geometry says: the geometry tests the DESIGN, not whether the moments bound theta.
    gate_pass = (bool(ident) and all(v["identified_split"] for v in ident.values())
                 and not any(v["objective_zero"] for v in ident.values()))
    promoted_path = COST_FWD / f"cost_params_{base_tag}.json"
    out_path = COST_FWD / f"cost_params_{tag}.json"
    same_name = out_path.name == promoted_path.name
    do_promote = bool(args.promote and (gate_pass or args.force_ridge))

    results["run"] = _jsonable(dict(
        tag=tag, base_tag=base_tag, psi_tag=args.psi_tag,
        starts=starts_all, n_starts=len(starts_all),
        rf_sources=rf_sources, rf_source_by_start=rf_source_by_start,
        rf_bar_beta_by_start=rf_bar_beta_by_start,
        n_dev_files=len(dev_files), n_eq_rows=int(len(eq)), n_dev_rows=int(len(dev)),
        n_shards_expected=int(n_shards), n_shards_found=len(shards_present),
        beta=psi_beta, T=psi_T, discount_source=disc_src,
        bootstrap=int(args.bootstrap), subsample=int(args.subsample),
        ci_level=float(args.ci_level), profile=bool(args.profile)))
    results["identification"] = _jsonable(dict(
        per_type=ident, gate_pass=gate_pass,
        ridge_cond_max=float(args.ridge_cond_max),
        promote_requested=bool(args.promote), forced=bool(args.force_ridge),
        promoted=do_promote, promoted_name=promoted_path.name if do_promote else None))

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved cost parameters -> {out_path.name}")

    for kappa in blocks:
        _print_final(kappa, results[kappa], args.ci_level)

    conds = ", ".join(f"{k}: {v['pooled_bkw_cond']:.4g}{' (criterion==0)' if v['objective_zero'] else ''}"
                      for k, v in ident.items())
    print(f"  GATE: pooled BKW cond <= {args.ridge_cond_max:.4g} and criterion > 0 for every "
          f"type? {gate_pass}  ({conds})")
    if not args.promote:
        print(f"  --promote not passed: cost_params_{base_tag}.json left untouched.")
        return
    if gate_pass or args.force_ridge:
        note = "" if gate_pass else "  (FORCED past a failed gate -- only c_bar is defensible)"
        if same_name:
            print(f"  Promotion is a no-op: --psi-tag is empty, so the file just written IS "
                  f"{promoted_path.name}.{note}")
        else:
            shutil.copyfile(out_path, promoted_path)
            print(f"  Promoted {out_path.name} -> {promoted_path.name}{note}")
        return

    # Refused. Exit 3 rather than 0: the counterfactual jobs chain afterok on this job
    # (cf_run.sh --cost-afterok), so a failure CANCELS them instead of letting them consume a
    # stale untagged cost_params. A cancelled job writes NO log at all -- if cf1_net/cf3/cf5/cf6
    # produced no .out, this refusal is why, and the reason is here rather than in a file that
    # was never created.
    print(f"  REFUSING TO PROMOTE: for at least one firm type the omega/zeta split is not "
          f"identified at --ridge-cond-max {args.ridge_cond_max:.4g}, or the criterion is "
          f"exactly 0 so there is no point estimate.")
    for k, v in ident.items():
        print(f"    [{k}] pooled BKW cond={_fmt(v['pooled_bkw_cond'], '.4g')} "
              f"corr={_fmt(v['pooled_corr'], '.6f')} "
              f"ratio CV={_fmt(100 * v['ratio_cv'], '.2f')}% starts={v['n_starts']}"
              f"{'  CRITERION == 0 (set-identified)' if v['objective_zero'] else ''}")
    if same_name:
        print(f"    NOTE: --psi-tag is empty, so {out_path.name} IS BOTH this run's json and "
              f"the promoted name -- it was already written and cannot be un-written. The exit "
              f"code, not the file, is what stops the dependents.")
    else:
        print(f"    cost_params_{base_tag}.json is untouched -- whatever the counterfactuals "
              f"were going to read, they still would.")
    print(f"    Report c_bar, or re-run with --force-ridge to promote anyway.")
    print(f"    Dependent CF jobs chained afterok on this one are now CANCELLED and will "
          f"leave no log of their own.")
    raise SystemExit(3)


if __name__ == "__main__":
    main()
