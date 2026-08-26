"""
estimation_bbl_3_solve.py
============================
BBL Step 2, part 2: recover marginal-cost parameters (ω, ζ, γ)^κ for
κ ∈ {B, D} by minimizing the sum of squared FOC-inequality violations, using the
firm-level ψ basis produced by estimation_bbl_2_fwd_sim.jl.

Moment inequality (per firm j, deviation σ̃) — the unnumbered display above
V_Main \\label{eq:17}:

    g_j(σ̂, σ̃) = (ψ_eq,j − ψ_dev,j,σ̃)' · [ 1 , −ω , −γ' , −(1+ζ) ]  ≥ 0

with the ψ block layout written by estimation_bbl_2_fwd_sim.jl:
    [ psi1 ,  psi2_omega ,  psi3_gamma_<z>… ,  psi4_zeta ]
so, writing Δ = ψ_eq − ψ_dev,

    g = Δ·psi1  −  ω·Δ·psi2_omega  −  Σ_z γ_z·Δ·psi3_gamma_z  −  (1+ζ)·Δ·psi4_zeta.

Deviations should make the bank worse off, so g ≥ 0 in equilibrium; we minimize

    F^κ(ω, ζ, γ) = Σ_{j∈κ, σ̃}  min{ g_j , 0 }²                      (V_Main eq:17)

separately for B- and D-type firms. χ (private cost shock) is dropped, as in the draft.

Inputs (from estimation_bbl_2_fwd_sim.jl):
    COST_FWD/psi_eq_{tag}.parquet   — firm, is_B, psi1, psi2_omega, psi3_gamma_*, psi4_zeta
    COST_FWD/psi_dev_{tag}.parquet  — shock, firm, is_B, <same blocks>
where tag = E{estim}_spec_{spec}_{stage}{suffix}.

Outputs:
    COST_FWD/cost_params_{tag}.json — (ω, ζ, γ) per type, optimizer-health diagnostics, and
    inference: `subsample` (critical values + √n-rate comparison CI) and, with --profile,
    `ci_omega`/`ci_zeta` (the criterion inverted at the subsampled critical value).
    `omega_se`/`zeta_se`/`gamma_se` are the legacy firm-block bootstrap SDs, RETAINED FOR
    COMPARISON ONLY.

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

⚠ ζ SPECIFICALLY: every firm is simulated against the SAME realised forward r^f path, so ζ
(a pass-through parameter) is identified off one macro path, not off n independent firms.
No firm-resampling scheme — bootstrap or subsample — delivers honest uncertainty for it.

⚠ Run only after estimation_bbl_2_fwd_sim.jl has produced its parquets, AND only with σ̂ from
the FITTED policy (--policy-csv upstream): with deviations around raw observed spreads,
frac_bind ≈ ½ mechanically and nothing here is interpretable.

Usage:
  python estimation_bbl_3_solve.py --estim 6 --spec 12 --stage extended \\
      --subsample 200 --profile --ci-level 0.95      # headline
  ... --bootstrap 200                                 # legacy SEs, for comparison
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

# COST_FWD holds the ψ parquets (from estimation_bbl_2_fwd_sim) + the cost_params json.
# cost_fwd_dir() is the ONE implementation of that location: it defaults to
# estimation_output()/COST_FWD and honours CF_COST_FWD, the env the BBL jobs export to point the
# whole step at the cluster's data/output/bbl folder. Reading the env here as well would be a
# second implementation of the same rule, free to drift from the writer's.
#
# The location is pinned to estimation_output(), NOT SLEEP_OUT_ROOT: estimation_bbl_2_fwd_sim.jl
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
    theta_c in foundation_psi_basis.jl, which reconstructs the loading as −(1+ζ) and also uses ζ
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


def cbar_stats(fit, rbar, boot_draws=None, sub_thetas=None, n_firms=None, b_firms=None,
               levels=(0.90, 0.95)):
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
    """
    out = dict(c_bar=float(fit["omega"] + rbar * fit["zeta"]))
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
    return dict(d1=blk["d1"][rows], d_omega=blk["d_omega"][rows],
                d_gamma=blk["d_gamma"][rows], d_zeta=blk["d_zeta"][rows],
                firms=blk["firms"][rows], gamma_names=blk["gamma_names"])


def subsample_kappa(blk, fit, n_sub=200, b_firms=None, seed=4242, levels=(0.90, 0.95)):
    """Subsampling inference for the eq:17 criterion (BBL 2007; Chernozhukov–Hong–Tamer 2007).

    WHY NOT THE BOOTSTRAP.  F is a squared hinge over moment INEQUALITIES: kinked, and
    potentially SET-identified.  The nonparametric bootstrap is inconsistent for
    set-identified / boundary parameters and does not reproduce the non-standard limit of a
    kinked criterion.  BBL propose subsampling; CHT invert the criterion with a subsampled
    critical value.  This is cheap here because ψ is PRECOMPUTED — only the fast,
    linear-in-θ hinge minimisation is re-run.

    PROCEDURE.  Draw `n_sub` subsamples of `b_firms` FIRMS without replacement (the firm is
    the dependence unit: a firm's ψ_eq is reused across all of its deviations).  With
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


def profile_ci(blk, fit, which, crit_value, npts=81, max_expand=6):
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
    return dict(which=which, crit=float(crit_value), ci_lo=lo, ci_hi=hi, empty=False,
                truncated_lo=bool(lo <= grid[0] + 1e-12),
                truncated_hi=bool(hi >= grid[-1] - 1e-12),
                half_width=float(hw), grid=grid.tolist(), T=T.tolist())


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
                    help="firm-block bootstrap reps (COMPARISON ONLY — the bootstrap is not "
                         "valid for this kinked/possibly set-identified criterion; 0 to skip)")
    ap.add_argument("--subsample", type=int, default=200,
                    help="subsampling reps for the HEADLINE inference (BBL/CHT); 0 to skip")
    ap.add_argument("--subsample-b", type=int, default=None,
                    help="subsample size in FIRMS (default n^(2/3))")
    ap.add_argument("--ci-level", type=float, default=0.95,
                    help="confidence level for the subsampled criterion inversion")
    ap.add_argument("--profile", action="store_true",
                    help="invert the criterion into ω/ζ confidence intervals (true profile, "
                         "re-minimising nuisance coefficients; needs --subsample > 0)")
    args = ap.parse_args()

    tag = f"E{args.estim}_spec_{args.spec}_{args.stage}{args.suffix}"
    eq_path = COST_FWD / f"psi_eq_{tag}.parquet"
    if not eq_path.exists():
        raise FileNotFoundError(f"Missing {eq_path.name} — run estimation_bbl_2_fwd_sim.jl first.")
    # Deviation ψ may be a single file (n-shards=1) or several shard files; merge all.
    dev_files = sorted(COST_FWD.glob(f"psi_dev_{tag}*.parquet"))
    if not dev_files:
        raise FileNotFoundError(f"No psi_dev_{tag}*.parquet — run estimation_bbl_2_fwd_sim.jl (all shards) first.")

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
            multistart_max_dF=float(ms_max_dF),
            n_firms=int(np.unique(blk["firms"]).size),
            n_rows=int(blk["d1"].size),
        )

        # ---- Subsampling: the headline inference for this criterion --------------------
        sub = subsample_kappa(blk, fit, n_sub=args.subsample,
                              b_firms=args.subsample_b) if args.subsample > 0 else None
        if sub is not None:
            # `_thetas` is a private ndarray handle for cbar_stats — never serialize it.
            rec["subsample"] = {k: v for k, v in sub.items() if not k.startswith("_")}

        # ---- The identified combination c̄ = ω + r̄^f·ζ ---------------------------------
        # ω and ζ are not separately identified here (Δψ₂ and Δψ₄ collinear at corr > 0.99997);
        # c̄ is the direction the data does pin down, so it carries the interpretable SE.
        ridge = rbar_of_block(blk)
        rec.update(ridge)
        rec.update(cbar_stats(
            fit, ridge["rbar_f"], boot_draws=boot_draws,
            sub_thetas=(sub or {}).get("_thetas"),
            n_firms=(sub or {}).get("n_firms"), b_firms=(sub or {}).get("b_firms")))

        # ---- Invert the criterion into ω/ζ confidence intervals -----------------------
        if args.profile:
            lvl = f"{args.ci_level:.2f}"
            crit = (sub or {}).get("crit", {}).get(lvl)
            if crit is None:
                print(f"  [{kappa}] --profile needs --subsample > 0 for a critical value; skipped.")
            else:
                rec["ci_omega"] = profile_ci(blk, fit, "omega", crit)
                rec["ci_zeta"] = profile_ci(blk, fit, "zeta", crit)

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
        if sub is not None and "skipped" not in sub:
            print(f"        subsample: b={sub['b_firms']}/{sub['n_firms']} firms, "
                  f"{sub['n_sub']} reps | crit({args.ci_level:.2f})="
                  f"{sub['crit'][f'{args.ci_level:.2f}']:.4g}")
        elif sub is not None:
            print(f"        subsample: {sub['skipped']}")

        for nm in ("omega", "zeta"):
            ci = rec.get(f"ci_{nm}")
            if not ci:
                continue
            sym = "ω" if nm == "omega" else "ζ"
            if ci["empty"]:
                print(f"        CI {sym}: EMPTY at the {args.ci_level:.0%} level "
                      f"(criterion never within crit — check identification)")
            else:
                trunc = ("  ⚠ WINDOW-TRUNCATED (not a set boundary)"
                         if (ci["truncated_lo"] or ci["truncated_hi"]) else "")
                print(f"        CI {sym}: [{ci['ci_lo']:.4g}, {ci['ci_hi']:.4g}] "
                      f"width={ci['ci_hi']-ci['ci_lo']:.4g}{trunc}")
        if 0.45 <= fit["frac_bind"] <= 0.55:
            print(f"  !!!! [{kappa}] frac_bind={fit['frac_bind']:.3f} — suspiciously close to ½. "
                  f"A symmetric ± grid around a σ̂ that is NOT a turning point of the simulated "
                  f"value makes exactly one of each ± pair bind for ANY θ, so the moments carry "
                  f"little/no identifying content and NOTHING below (point estimates, SEs, CIs) "
                  f"should be read as a result. Deviate around the FITTED policy (--policy-csv). !!!!")

    out_path = COST_FWD / f"cost_params_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved cost parameters → {out_path.name}")


if __name__ == "__main__":
    main()
