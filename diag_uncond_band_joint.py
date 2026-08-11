"""Unconditional national phi_t band for the JOINT-SIEVE estimators (E7/E8).

WHY. E5/E6 have an unconditional band -- every draw perturbs the direction AND re-profiles the
link. E7/E8 have only the CONDITIONAL band the estimator stores: theta is perturbed while the
link is held frozen at its point estimate. For the joint sieve that omission is the awkward one,
because the link is estimated JOINTLY with theta there, so holding it fixed conditions on the
very object the estimator was choosing. The consequence is visible: E8's stored band averages
0.38pp, which asserts near-certainty about a flat phi_t path -- and E8 spec 12 is the paper's
preferred routine (V_Main:456).

HOW, WITHOUT RE-RUNNING THE JULIA SEARCH. theta-hat is on disk; only the link stage and the
influence functions need rebuilding, and both are deterministic given theta. This script
replicates fit_joint_single_index's sieve profiling exactly -- the same 0.1/99.9 percentile
knot hull, the same interior quantile knots, the same 1+2 IRLS passes under the robust loss,
the same ceiling on the final refit -- and then reuses it per draw.

TWO CERTIFICATIONS, both blocking, because a rebuild that is not the estimator's is worthless:
  1. profiling at theta-hat must reproduce the STORED link grid (si_ggrid);
  2. the CONDITIONAL draws (link frozen) must reproduce the STORED phi_t_boot band.
Only then is the unconditional variant -- the same draws with the link re-profiled -- reported.

READ-ONLY: writes ts_uncond_joint_est{E}.pkl + a CSV into <demand_prep_root>/DIAGNOSTICS,
beside the vintage it describes; never to Rout and never into the code repo.

Usage:  python diag_uncond_band_joint.py --est 7 [--B 400]
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from estimation_2_sleep import (build_pooled_data, define_specifications,  # noqa: E402
                                run_pooled_first_stage)
from utils import paths as _paths_mod                                      # noqa: E402
from utils.sleep_links import (_agg_phi_t, _band_from_draws, _cauchy_weights,  # noqa: E402
                               _fit_link_sieve, _phi_t_group_struct, _ramp_design,
                               _twoway_demean, _wild_weights, link_constrained)

SPEC = "IV_HausmanFull x Tech"
FE_TIME_COL = "time_id"
DEGREE, N_INTERIOR = 3, 5


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--est", type=int, default=7, choices=(7, 8))
    ap.add_argument("--B", type=int, default=400)
    ap.add_argument("--loss", default="robust", choices=("robust", "ls"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tb = (a.est == 8)
    loss = "robust" if a.loss == "robust" else "ls"
    root = _paths_mod.demand_prep_root()
    # Artifacts live with the DATA, beside the bands they cross-check -- never in the
    # code repo. demand_prep_root() honours SLEEP_OUT_ROOT, so they follow the vintage.
    OUT = root / "DIAGNOSTICS"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    key = "second_stage" if loss == "robust" else "second_stage_ls"
    res = pickle.load(open(root / f"est{a.est}" / "estimation_results.pkl", "rb"))[SPEC][key]
    stored_band = getattr(res, "phi_t_boot", None)

    df = build_pooled_data(time_block=tb)
    _, iv_specs, state_blocks = define_specifications(time_block=tb)
    s_cols = state_blocks["Tech"]
    iv_act = [c for c in iv_specs["IV_HausmanFull"]
              if c in df.columns and df[c].notnull().sum() > 0]
    exog = [c for c in s_cols if c in df.columns and df[c].notnull().sum() > 0]
    dft, _ = run_pooled_first_stage(df, iv_act, exog)
    del df

    # ---- rebuild the estimator's frame, exactly ------------------------------------------
    cols = s_cols + ["deposit_balance", "nr_lagged_dep", "entity_id", "v_hat_x_lagged_dep"]
    ss = dft.dropna(subset=cols).copy()
    idx_cols = [c for c in s_cols if c != "constant"]
    S = ss[idx_cols].values.astype(float)
    S_mu, S_sd = S.mean(0), S.std(0)
    S_sd[S_sd <= 0] = 1.0
    Sn = (S - S_mu) / S_sd
    Z = ss["nr_lagged_dep"].values.astype(float)
    y = ss["deposit_balance"].values.astype(float)
    cf = ss["v_hat_x_lagged_dep"].values.astype(float)
    _, einv = np.unique(ss["entity_id"].values, return_inverse=True)
    counts = np.bincount(einv).astype(float)
    _, tinv = np.unique(ss[FE_TIME_COL].values, return_inverse=True)
    tcounts = np.bincount(tinv).astype(float)
    demean = lambda M: _twoway_demean(M, einv, counts, tinv, tcounts)
    y_dm, cf_dm = demean(y), demean(cf)
    cl_u, cl_inv = np.unique(ss["CodConglomeradoPrudencial"].astype(str).values,
                             return_inverse=True)
    n_cl = len(cl_u)

    # theta-hat: params_native holds theta/S_sd, and the estimator constrains ||theta|| = 1
    th_nat = res.params_native.reindex([f"interaction_{c}" for c in idx_cols]).values.astype(float)
    theta = th_nat * S_sd
    nrm = float(np.linalg.norm(theta))
    print(f"[joint] n={len(ss):,} d={len(theta)} clusters={n_cl} | ||theta||={nrm:.6f} "
          f"(estimator constrains it to 1)")
    theta = theta / nrm

    _cap = bool(link_constrained())

    def profile(th, want_grid=False):
        """fit_joint_single_index._fit_link, sieve branch, replicated exactly."""
        v = Sn @ th
        lo, hi = np.quantile(v, [0.001, 0.999])
        qs = np.linspace(0, 1, N_INTERIOR + 2)[1:-1]
        interior = np.clip(np.quantile(v, qs), lo + 1e-9, hi - 1e-9)
        t = np.concatenate(([lo] * (DEGREE + 1), np.sort(interior), [hi] * (DEGREE + 1)))
        R = _ramp_design(v, t, DEGREE)
        if loss == "robust":
            beta, gamma, resid, ssr, c = _fit_link_sieve(R, Z, cf_dm, y_dm, einv, counts,
                                                         None, dm=demean, cap_sum=_cap)
            for _ in range(2):
                w = _cauchy_weights(resid)
                beta, gamma, resid, ssr, c = _fit_link_sieve(R, Z, cf_dm, y_dm, einv, counts,
                                                             w, dm=demean, cap_sum=_cap)
        else:
            beta, gamma, resid, ssr, c = _fit_link_sieve(R, Z, cf_dm, y_dm, einv, counts, None,
                                                         dm=demean, cap_sum=_cap)
        if not want_grid:
            return v, resid
        spl = BSpline(t, c, DEGREE, extrapolate=True)
        vgrid = np.linspace(t[0], t[-1], 200)
        ggrid = np.clip(np.maximum.accumulate(spl(vgrid)), 0.0, 1.0)
        gp = np.clip(spl.derivative()(vgrid), 0.0, None)
        return v, resid, vgrid, ggrid, gp

    # ---- CERTIFICATION 1: the stored link grid --------------------------------------------
    v0, resid0, vgrid0, ggrid0, gp0 = profile(theta, want_grid=True)
    offset = float(np.sum(theta * S_mu / S_sd))
    d_g = float(np.max(np.abs(ggrid0 - np.asarray(res.si_ggrid, float))))
    d_v = float(np.max(np.abs((vgrid0 + offset) - np.asarray(res.si_vgrid, float))))
    print(f"[cert 1] link grid vs stored: G {d_g:.3e} | index {d_v:.3e}")
    if d_g > 1e-9 or d_v > 1e-6:
        raise RuntimeError("profiling does not reproduce the stored link -- the rebuild is "
                           "not this estimator; refusing to report a band")

    # ---- influence functions for theta, as the estimator builds them ----------------------
    gp_obs = np.interp(v0, vgrid0, gp0)
    gS_dm = demean((gp_obs * Z)[:, None] * Sn)
    Minv = np.linalg.pinv(gS_dm.T @ gS_dm)
    IF = (resid0[:, None] * gS_dm) @ Minv.T
    IF = IF - np.outer(IF @ theta, theta)                 # project to the sphere tangent
    IF_cl = np.zeros((n_cl, len(theta)))
    for k in range(len(theta)):
        IF_cl[:, k] = np.bincount(cl_inv, weights=IF[:, k], minlength=n_cl)

    gs = _phi_t_group_struct(ss)
    phi_pt = np.clip(np.interp(v0, vgrid0, ggrid0), 0.0, 1.0)
    pt = _agg_phi_t(phi_pt, gs)

    B = int(min(a.B, getattr(res, "boot_B", 400) or 400))
    scheme = getattr(res, "boot_scheme", "webb") or "webb"
    rng = np.random.default_rng(a.seed + 12345)           # the estimator's band seed
    nt = gs["nt"]
    dr_cond, dr_unc = np.empty((B, nt)), np.empty((B, nt))
    cos = np.empty(B)

    # CHECKPOINT. The per-draw re-profile is the whole cost (~10-25 s), so an interrupted run
    # would otherwise throw away hours. Completed rows are flushed every 25 draws and reloaded
    # on restart. The RNG is NOT pickled: the loop always re-draws the weight sequence from the
    # seed and merely skips the expensive profile for rows already done, so a resumed run
    # traces the identical draw sequence -- which is what certification 2 compares against.
    ck = OUT / f".ckpt_joint_est{a.est}_{loss}_B{B}.npz"
    done = 0
    if ck.exists():
        try:
            z = np.load(ck)
            if z["dr_cond"].shape == (B, nt):
                dr_cond, dr_unc, cos = z["dr_cond"], z["dr_unc"], z["cos"]
                done = int(z["done"])
                print(f"[ckpt] resuming at draw {done}/{B} from {ck.name}")
        except Exception as e:
            print(f"[ckpt] unreadable ({type(e).__name__}), starting fresh")

    for ib in range(B):
        w = _wild_weights(n_cl, scheme, rng)              # consumed even when skipping
        if ib < done:
            continue
        th_b = theta + w @ IF_cl
        th_b = th_b / (np.linalg.norm(th_b) + 1e-12)
        cos[ib] = float(theta @ th_b)
        v_b = Sn @ th_b
        dr_cond[ib] = _agg_phi_t(np.clip(np.interp(v_b, vgrid0, ggrid0), 0.0, 1.0), gs)
        _, _, vg_b, gg_b, _ = profile(th_b, want_grid=True)
        dr_unc[ib] = _agg_phi_t(np.clip(np.interp(v_b, vg_b, gg_b), 0.0, 1.0), gs)
        if (ib + 1) % 25 == 0:
            np.savez(ck, dr_cond=dr_cond, dr_unc=dr_unc, cos=cos, done=ib + 1)
            print(f"    [joint-band] {ib+1}/{B}  (checkpointed)", flush=True)

    b_cond = _band_from_draws(pt, dr_cond, gs["tuniq"], label="joint conditional")
    b_unc = _band_from_draws(pt, dr_unc, gs["tuniq"], label="joint unconditional")
    W = lambda b: 100 * float((b["hi"] - b["lo"]).mean())

    # ---- CERTIFICATION 2: the stored conditional band --------------------------------------
    cert2 = None
    if stored_band is not None:
        s = stored_band.sort_values("time_id").reset_index(drop=True)
        c_ = b_cond.sort_values("time_id").reset_index(drop=True)
        if len(s) == len(c_):
            cert2 = max(float(np.max(np.abs(c_["lo"].values - s["lo"].values))),
                        float(np.max(np.abs(c_["hi"].values - s["hi"].values))))
            print(f"[cert 2] conditional draws vs stored phi_t_boot: {cert2:.3e} "
                  f"(stored width {W(s):.2f}pp, rebuilt {W(b_cond):.2f}pp)")

    out = dict(band_conditional=b_cond, band_unconditional=b_unc, pt=pt,
               draws=dict(conditional=dr_cond, unconditional=dr_unc),
               meta=dict(est=a.est, loss=loss, B=B, scheme=scheme, spec=SPEC,
                         cert_link=d_g, cert_cond_band=cert2, cap=_cap,
                         cos_med=float(np.median(cos)), cos_p05=float(np.percentile(cos, 5)),
                         n=len(ss), n_cl=n_cl, runtime_min=(time.time() - t0) / 60))
    pickle.dump(out, open(OUT / f"ts_uncond_joint_est{a.est}_{loss}.pkl", "wb"))
    ck.unlink(missing_ok=True)          # the run completed; the checkpoint is now dead weight
    pd.DataFrame([dict(est=f"E{a.est}", loss=loss, B=B,
                       width_conditional_pp=W(b_cond), width_unconditional_pp=W(b_unc),
                       ratio=W(b_unc) / max(W(b_cond), 1e-12),
                       level_pp=100 * float(pt.mean()),
                       range_pp=100 * float(pt.max() - pt.min()),
                       cert_link=d_g, cert_cond_band=cert2,
                       cos_med=float(np.median(cos)))]).to_csv(
        OUT / f"uncond_joint_est{a.est}_{loss}.csv", index=False)

    print(f"\n===== JOINT-SIEVE UNCONDITIONAL BAND — E{a.est}/{loss} =====")
    print(f"  phi_t level {100*pt.mean():.2f}pp, range {100*(pt.max()-pt.min()):.2f}pp")
    print(f"  conditional   (link frozen)     {W(b_cond):6.2f}pp   <- what the paper reports")
    print(f"  UNCONDITIONAL (link re-profiled){W(b_unc):6.2f}pp   "
          f"-> {W(b_unc)/max(W(b_cond),1e-12):.2f}x")
    print(f"  theta-draw cosine: median {np.median(cos):.4f}, 5th pct "
          f"{np.percentile(cos,5):.4f}")
    print(f"\nwrote uncond_joint_est{a.est}_{loss}.csv  "
          f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
