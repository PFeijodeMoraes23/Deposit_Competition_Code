"""Exact projection of a confidence REGION for theta onto phi_t, under the constrained link.

THE PROBLEM IT ADDRESSES. Every bootstrap band here is Wald-type: it centres on theta-hat and
propagates uncertainty through INFLUENCE FUNCTIONS, a first-order expansion. The measured
theta-draw cosines run down to 0.80 at the 5th percentile, so draws routinely land where a
first-order expansion is no longer obviously reliable, and the band's own tails are the part
being extrapolated. This script removes the linearization: it samples a joint confidence region
for theta and, at EVERY sampled theta, RE-RUNS the estimator's own link stage exactly -- knots
re-placed, the shape-constrained QP re-solved -- then aggregates phi_t. The theta -> phi_t map
is evaluated, never linearized.

WHAT IT IS AND IS NOT. The region is a WALD ellipsoid built from the cluster-robust covariance
of theta-hat, so this is not weak-identification-robust in the Anderson-Rubin sense (a genuine
AR set inverts a moment test and can be unbounded). What it removes is the linearization of
theta -> phi_t, which is the part the cosine spread indicts. The report must say which of the
two it is rather than borrow AR's authority.

Two readings come out:
  * ENVELOPE vs the tangent-cone bootstrap band -- how much the linearization understated;
  * the CRITERION PROFILE: how much the logit-link NLLS objective that actually chose theta-hat
    rises as phi_t swings. A criterion that barely moves while phi_t swings widely is
    identification weakness stated in phi_t units.

Coverage: projecting a JOINT 95% region gives SIMULTANEOUS coverage across quarters, which is
conservative against the pointwise bootstrap bands. Flagged, not hidden. The envelope is also a
LOWER bound at finite NSAMP -- it can only grow with more samples -- so a reported widening
factor understates rather than overstates.

Assembly is the estimator's, exactly: knots from _pix_shifted_range, ONE _twoway_demean call
over the whole design block, two-way entity+quarter FE for BOTH E3 and E4, and vmu/vsd
recomputed from the sample. Each of those is a trap that has cost measurable error before.

READ-ONLY. Writes test_inversion_est{E}_{loss}.csv/.pkl into <demand_prep_root>/DIAGNOSTICS.

Usage:  python diag_test_inversion_band.py [--est 3] [--loss robust] [--nsamp 200]
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from estimation_2_sleep import (build_pooled_data, define_specifications,  # noqa: E402
                                run_pooled_first_stage)
from utils import paths as _paths_mod                                      # noqa: E402
from utils.sleep_links import (_agg_phi_t, _build_phi_X, _bspline_design,  # noqa: E402
                               _phi_t_group_struct, _pix_shifted_range,
                               _ramp_design, _twoway_demean, ispline_constraints,
                               nlls_direction_if, solve_ispline_qp, SI_N_INTERIOR)

SPEC = "IV_HausmanFull x Tech"
FE_TIME_COL = "time_id"
LOSS_OF = {"robust": "cauchy", "ls": "linear"}
KEY_OF = {"robust": "second_stage", "ls": "second_stage_ls"}
DEGREE = 3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--est", type=int, default=3, choices=(3, 4))
    ap.add_argument("--loss", default="robust", choices=("robust", "ls"))
    ap.add_argument("--nsamp", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20240624)
    a = ap.parse_args()
    tb = (a.est == 4)
    root = _paths_mod.demand_prep_root()
    # Artifacts live with the DATA, beside the bands they cross-check -- never in the
    # code repo. demand_prep_root() honours SLEEP_OUT_ROOT, so they follow the vintage.
    OUT = root / "DIAGNOSTICS"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    si = pickle.load(open(root / f"est{a.est}" / "estimation_results.pkl",
                          "rb"))[SPEC][KEY_OF[a.loss]]
    constrained = (getattr(si, "si_constrained", False)
                   or getattr(si, "link", None) == "index_sieve")
    if not constrained:
        raise RuntimeError(f"est{a.est}/{a.loss} is not a constrained fit "
                           f"(link={getattr(si,'link',None)!r}); this script targets the "
                           f"shape-constrained estimator")

    df = build_pooled_data(time_block=tb)
    _, iv_specs, state_blocks = define_specifications(time_block=tb)
    s_cols = state_blocks["Tech"]
    iv_act = [c for c in iv_specs["IV_HausmanFull"]
              if c in df.columns and df[c].notnull().sum() > 0]
    exog = [c for c in s_cols if c in df.columns and df[c].notnull().sum() > 0]
    dft, _ = run_pooled_first_stage(df, iv_act, exog)
    del df

    idx_expect = [f"interaction_{sv}" if sv != "constant" else "nr_lagged_dep"
                  for sv in s_cols]
    theta_ser = si.params_native.reindex(idx_expect)
    # the SAME direction + influence functions the band uses, so the comparison is like-for-like
    dirif = nlls_direction_if(dft, s_cols, True, theta_ser, LOSS_OF[a.loss],
                              link="logit", fe_time_col=FE_TIME_COL)
    IF_th, theta = dirif["IF_cl"], dirif["theta"]
    d = len(theta)
    V = IF_th.T @ IF_th
    w_eig, U = np.linalg.eigh((V + V.T) / 2)
    L = U @ np.diag(np.sqrt(np.clip(w_eig, 0, None)))
    crit = float(chi2.ppf(0.95, d))

    # ---- rebuild the link stage exactly as unconditional_phi_t_band does -------------------
    cols = s_cols + ["deposit_balance", "nr_lagged_dep", "entity_id"]
    ss = dft.dropna(subset=cols + ["v_hat_x_lagged_dep"]).copy()
    X = _build_phi_X(ss, idx_expect)
    Z = ss["nr_lagged_dep"].values.astype(float)
    CF_raw = ss["v_hat_x_lagged_dep"].values.astype(float)
    _, einv = np.unique(ss["entity_id"].values, return_inverse=True)
    ec = np.bincount(einv).astype(float)
    # BOTH E3 and E4 carry two-way entity+quarter FE: estimation_sleep_common passes
    # fe_time_col unconditionally, and `time_block` selects the STATE block, not the FE
    # structure. Branching the demeaner on the time block shifts si_b by 2.2e-04.
    _, tinv = np.unique(ss[FE_TIME_COL].values, return_inverse=True)
    tc = np.bincount(tinv).astype(float)
    dm = lambda M: _twoway_demean(M, einv, ec, tinv, tc)
    y_dm = dm(ss["deposit_balance"].values.astype(float))

    v0 = X @ theta
    vmu, vsd = float(v0.mean()), float(v0.std()) or 1.0
    gs = _phi_t_group_struct(ss)
    n_basis = len(np.asarray(si.si_knots, float)) - DEGREE - 1
    A_c, lb_c, ub_c = ispline_constraints(n_basis, n_extra=1)

    def knots_at(vs_loc):
        lo, hi = _pix_shifted_range(X, vs_loc, theta, idx_expect, vsd)
        _, t = _bspline_design(np.concatenate([vs_loc, np.array([lo, hi])]),
                               n_interior=SI_N_INTERIOR, degree=DEGREE)
        return t

    def profile_phi(th):
        """Exact re-profile of the CONSTRAINED link at direction th -> (phi_t, ssr)."""
        v = X @ th
        sd = float(v.std()) or 1.0
        vs = (v - float(v.mean())) / sd
        kn = knots_at(vs)
        P = _ramp_design(vs, kn, DEGREE)
        # ONE demean call over the whole block: _twoway_demean's convergence test is joint
        # across columns, so demeaning a column on its own lands elsewhere (~2e-3 in beta).
        D = dm(np.column_stack([P * Z[:, None], CF_raw]))
        b, _info = solve_ispline_qp(D, y_dm, n_basis)
        r = y_dm - D @ b
        phi = np.clip(P @ b[:n_basis], 0.0, 1.0)   # clip is a guard; the QP already bounds it
        return _agg_phi_t(phi, gs), float(r @ r)

    # criterion that actually selected theta-hat (logit-link NLLS on deposits)
    _g = np.atleast_1d(np.asarray(dirif["gamma_hat"], float)).ravel()
    if _g.size != 1:
        raise ValueError(f"expected one CF coefficient, got {_g.size}")
    gamma_hat = float(_g[0])

    def crit_logit(th):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ th, -500, 500)))
        r = dm(ss["deposit_balance"].values.astype(float) - p * Z - gamma_hat * CF_raw)
        if a.loss == "robust":
            s = 1.4826 * np.median(np.abs(r)) + 1e-12
            return float(np.sum(np.log1p((r / (2.385 * s)) ** 2)))
        return float(r @ r)

    pt0, ssr0 = profile_phi(theta)
    q0 = crit_logit(theta)
    # ---- certification, in BASIS space (the band's own fingerprint) ----------------------
    # Comparing against the stored si_ggrid instead would measure the 200-point grid's linear
    # interpolation error (~1e-4 in phi), not whether this rebuild is the estimator.
    _vs0 = (X @ theta - vmu) / vsd
    _kn0 = knots_at(_vs0)
    _d_kn = float(np.max(np.abs(_kn0 - np.asarray(si.si_knots, float))))
    _P0 = _ramp_design(_vs0, _kn0, DEGREE)
    _b0, _ = solve_ispline_qp(dm(np.column_stack([_P0 * Z[:, None], CF_raw])), y_dm, n_basis)
    _d_phi = float(np.max(np.abs(_P0 @ _b0[:n_basis]
                                 - _P0 @ np.asarray(si.si_beta, float))))
    print(f"[ti] certification: knots {_d_kn:.2e} | link {_d_phi:.2e} in phi "
          f"(band gate 1e-6) | level {100*pt0.mean():.2f}pp "
          f"range {100*(pt0.max()-pt0.min()):.2f}pp | d={d}")
    if _d_phi > 1e-6:
        raise RuntimeError(f"rebuild is not the stored estimator ({_d_phi:.2e} in phi) -- "
                           f"refusing to project a region around a different fit")

    lo, hi = pt0.copy(), pt0.copy()
    rng = np.random.default_rng(a.seed)
    qs, cos, rngs = [], [], []
    for i in range(a.nsamp):
        u = rng.normal(size=d)
        u /= np.linalg.norm(u)
        rad = np.sqrt(crit) * (1.0 if i % 4 else rng.uniform(0.3, 1.0))
        th_b = theta + rad * (L @ u)
        try:
            pt_b, _ = profile_phi(th_b)
        except Exception as e:
            print(f"    [ti] sample {i} failed ({type(e).__name__})")
            continue
        lo, hi = np.minimum(lo, pt_b), np.maximum(hi, pt_b)
        qs.append(crit_logit(th_b))
        rngs.append(100 * float(pt_b.max() - pt_b.min()))
        cos.append(float(theta @ th_b / (np.linalg.norm(theta) * np.linalg.norm(th_b))))
        if (i + 1) % 25 == 0:
            print(f"    [ti] {i+1}/{a.nsamp}  envelope {100*float((hi-lo).mean()):.2f}pp",
                  flush=True)

    qs = np.asarray(qs, float)
    env_w = 100 * float((hi - lo).mean())

    bands = {}
    bf = root / "Rout" / f"ts_link_band_est{a.est}_uncond_{a.loss}.pkl"
    if bf.exists():
        b = pickle.load(open(bf, "rb"))
        for key, lbl in (("band_tangent_cone_total", "tangent-cone (total)"),
                         ("band_tangent_cone", "tangent-cone (link only)"),
                         ("band_theta_only", "theta only"),
                         ("band_total", "projected total")):
            if key in b:
                bands[lbl] = 100 * float((b[key]["hi"] - b[key]["lo"]).mean())

    row = dict(est=f"E{a.est}", loss=a.loss, d=d, nsamp=len(qs), chi2_crit=crit,
               level_pp=100 * float(pt0.mean()),
               range_pp=100 * float(pt0.max() - pt0.min()),
               envelope_pp=env_w,
               crit_rise_med_pct=100 * float(np.median(qs - q0) / abs(q0)),
               crit_rise_max_pct=100 * float((qs - q0).max() / abs(q0)),
               range_p95_pp=float(np.percentile(rngs, 95)) if rngs else np.nan,
               cos_med=float(np.median(cos)), cos_p05=float(np.percentile(cos, 5)),
               runtime_min=(time.time() - t0) / 60,
               **{f"band_{k}": v for k, v in bands.items()})
    pd.DataFrame([row]).to_csv(OUT / f"test_inversion_est{a.est}_{a.loss}.csv", index=False)
    pickle.dump(dict(time_id=list(gs["tuniq"]), phi_t=list(map(float, pt0)),
                     lo=list(map(float, lo)), hi=list(map(float, hi)), row=row),
                open(OUT / f"test_inversion_est{a.est}_{a.loss}.pkl", "wb"))

    print(f"\n===== EXACT PROJECTION OF THE 95% THETA REGION — E{a.est}/{a.loss} =====")
    print(f"  phi_t level {row['level_pp']:.2f}pp, path range {row['range_pp']:.2f}pp")
    print(f"  ENVELOPE (simultaneous, {len(qs)} samples): {env_w:.2f}pp")
    for k, v in bands.items():
        print(f"    vs {k:26s} {v:7.2f}pp   -> {env_w/v:5.2f}x")
    print(f"  criterion rise across the region: median {row['crit_rise_med_pct']:.4f}%, "
          f"max {row['crit_rise_max_pct']:.4f}%")
    print(f"  theta cosine: median {row['cos_med']:.3f}, 5th pct {row['cos_p05']:.3f}")
    print("\n  A large envelope-to-band ratio says the IF linearization understated the")
    print("  direction channel; a tiny criterion rise beside a wide phi_t swing is")
    print("  identification weakness measured in phi_t units. Simultaneous vs pointwise")
    print("  coverage makes the ratio conservative; the envelope is a lower bound at finite n.")
    print(f"\nwrote test_inversion_est{a.est}_{a.loss}.csv  ({row['runtime_min']:.1f} min)")


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
