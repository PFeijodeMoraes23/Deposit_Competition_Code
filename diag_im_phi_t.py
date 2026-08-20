"""Ibragimov-Muller (2010) group-t band for national phi_t under the shape-constrained link.

WHY THIS EXISTS. Every phi_t band in the project is bootstrap machinery: wild cluster draws,
influence functions, and -- under the shape constraints -- a projection onto the constraint set
or its tangent cone. That machinery is exactly what the constrained link put in question
(Andrews 2000), so a check that shares none of it is worth more than another variant of it.
IM needs no influence function, no resampling, and no projection: partition the clusters into q
groups, run the WHOLE estimator separately in each, and read a t-interval with q-1 degrees of
freedom off the q estimates. Validity asks the group estimates to be approximately independent
and Gaussian; sizes may differ arbitrarily, which is the point under this cluster imbalance
(G = 453 nominal, G* ~ 6).

DESIGN, and why each choice is the honest one:

  * groups are unions of whole conglomerates -- the same independence assumption the WCB makes;
  * SNAKE assignment by conglomerate deposit volume, so the giants land one per group instead
    of stacking in one;
  * each group runs the FULL pipeline -- its own first stage, its own logit direction, its own
    constrained I-spline link. Sharing a full-sample first stage would couple the groups
    through v_hat and quietly break the independence IM needs;
  * every group's fitted model is then EVALUATED ON THE FULL SAMPLE. Aggregating each group's
    phi_t over only its own markets would have the q estimates target q different estimands
    (different market weights); predicting on the common sample makes them q independent
    estimators of ONE phi_t, which is what a cross-check requires.

Reported against the tangent-cone band from the same cell. READ-ONLY: writes its CSV/PKL
into <demand_prep_root>/DIAGNOSTICS, beside the vintage it describes, and touches no
estimation output.

Usage:  python diag_im_phi_t.py [--est 5] [--loss robust] [--q 8] [--qs 6,8,12]
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from estimation_2_sleep import (build_pooled_data, define_specifications,  # noqa: E402
                                run_pooled_first_stage)
from utils import paths as _paths_mod                                      # noqa: E402
from utils import routines as _routines                                    # noqa: E402
from utils.sleep_links import (fit_nlls_link, fit_single_index, phi_from_native,  # noqa: E402
                               _phi_t_group_struct, _agg_phi_t, _build_phi_X)

SPEC = _routines.SPEC12
FE_TIME_COL = _routines.FE_TIME_COL
LOSS_OF = {"robust": "cauchy", "ls": "linear"}
KEY_OF = {"robust": "second_stage", "ls": "second_stage_ls"}
CLUSTER = "CodConglomeradoPrudencial"


def snake(congl_by_size, q):
    """1..q, q..1, 1..q, ... so the largest conglomerates spread one per group."""
    order = list(range(q)) + list(range(q - 1, -1, -1))
    return {c: order[i % (2 * q)] for i, c in enumerate(congl_by_size)}


def fit_cell(df, s_cols, loss, tb):
    """One full estimation of the constrained single index: first stage, logit direction,
    shape-constrained link. Mirrors estimation_sleep_common._exec_spec's single_index path."""
    _, iv_specs, _ = define_specifications(time_block=tb)
    iv_act = [c for c in iv_specs["IV_HausmanFull"]
              if c in df.columns and df[c].notnull().sum() > 0]
    exog_act = [c for c in s_cols if c in df.columns and df[c].notnull().sum() > 0]
    dft, _ = run_pooled_first_stage(df, iv_act, exog_act)
    lg = fit_nlls_link(dft, s_cols, has_cf=True, link="logit", loss=LOSS_OF[loss],
                       fe_time_col=FE_TIME_COL, bootstrap=False)
    if lg is None:
        return None
    return fit_single_index(dft, s_cols, has_cf=True, logit_res=lg, degree=3,
                            fe_time_col=FE_TIME_COL, phi_band=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--est", type=int, default=_routines.LINK_ESTS[0],
                    choices=tuple(_routines.LINK_ESTS))
    ap.add_argument("--loss", default="robust", choices=("robust", "ls"))
    ap.add_argument("--qs", default="6,8,12", help="group counts; the primary is --q")
    ap.add_argument("--q", type=int, default=8, help="primary group count (reported band)")
    a = ap.parse_args()
    qs = sorted({int(x) for x in a.qs.split(",")} | {a.q})
    tb = (a.est == 4)
    root = _paths_mod.demand_prep_root()
    # Artifacts live with the DATA, beside the bands they cross-check -- never in the
    # code repo. demand_prep_root() honours SLEEP_OUT_ROOT, so they follow the vintage.
    OUT = root / "DIAGNOSTICS"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = build_pooled_data(time_block=tb)
    _, _, state_blocks = define_specifications(time_block=tb)
    s_cols = state_blocks["Tech"]

    # ---- the common evaluation sample and the production aggregation --------------------
    full = fit_cell(df, s_cols, a.loss, tb)
    if full is None:
        raise RuntimeError("full-sample fit failed")
    need = s_cols + ["deposit_balance", "nr_lagged_dep", "entity_id", CLUSTER]
    ev = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    gs = _phi_t_group_struct(ev)
    phi_full = phi_from_native(ev, full, getattr(full, "link", None) or "index_sieve")
    pt_full = _agg_phi_t(phi_full, gs)
    print(f"\n[im] full-sample phi_t: level {100*pt_full.mean():.2f}pp "
          f"range {100*(pt_full.max()-pt_full.min()):.2f}pp | "
          f"link={getattr(full,'link',None)} n_eval={len(ev):,}")

    size = ev.groupby(CLUSTER)["deposit_balance"].apply(
        lambda s: float(np.abs(s).sum())).sort_values(ascending=False)
    congl = list(size.index)
    print(f"[im] clusters: {len(congl)} | top-5 share {100*size.iloc[:5].sum()/size.sum():.1f}%")

    rows, paths, diag = [], {}, []
    for q in qs:
        grp = snake(congl, q)
        ev["_g"] = ev[CLUSTER].map(grp)
        df["_g"] = df[CLUSTER].map(grp)
        ests, used = [], []
        for g in range(q):
            d = df[df["_g"] == g]
            if d[CLUSTER].nunique() < 2 or d["entity_id"].nunique() < 3:
                print(f"  [q={q}] group {g}: too thin, skipped")
                continue
            try:
                r = fit_cell(d, s_cols, a.loss, tb)
            except Exception as e:
                print(f"  [q={q}] group {g}: FAILED ({type(e).__name__}: {e})")
                continue
            if r is None:
                print(f"  [q={q}] group {g}: direction failed")
                continue
            # evaluate the GROUP's model on the COMMON sample -> one estimand
            phi_g = phi_from_native(ev, r, getattr(r, "link", None) or "index_sieve")
            ests.append(_agg_phi_t(phi_g, gs))
            used.append(g)
            sb = (getattr(r, "si_qp", None) or {}).get("sum_beta")
            # EXTRAPOLATION DIAGNOSTIC. Evaluating on the common sample is what makes the q
            # estimates target one estimand, but a group whose index support is narrower than
            # the full sample's has its link extended flat beyond its own hull -- and the
            # value it is held at is the endpoint of ITS grid. If the level gap tracks this
            # share, the gap is an artifact of the evaluation, not the estimator.
            _pp = [p for p in r.params_native.index if not str(p).startswith("v_hat")]
            _ix = _build_phi_X(ev, _pp) @ r.params_native[_pp].values.astype(float)
            _gv = np.asarray(r.si_vgrid, float)
            below = float(np.mean(_ix < _gv[0]))
            above = float(np.mean(_ix > _gv[-1]))
            diag.append(dict(q=q, g=g, n=len(d), level=100 * float(ests[-1].mean()),
                             sum_beta=sb, below=100 * below, above=100 * above,
                             g_lo=float(r.si_ggrid[0]), g_hi=float(r.si_ggrid[-1])))
            print(f"  [q={q}] group {g}: n={len(d):,} clusters={d[CLUSTER].nunique()} "
                  f"level {100*ests[-1].mean():.2f}pp"
                  + (f" sum(beta)={sb:.4f}" if sb is not None else "")
                  + f" | outside hull: {100*below:.1f}% below (G={r.si_ggrid[0]:.3f}), "
                    f"{100*above:.1f}% above (G={r.si_ggrid[-1]:.3f})")
        if len(ests) < 3:
            print(f"  [q={q}] only {len(ests)} usable groups -- no interval")
            continue
        E = np.vstack(ests)                       # (n_used x nt)
        n = E.shape[0]
        m, sd = E.mean(axis=0), E.std(axis=0, ddof=1)
        tcrit = float(stats.t.ppf(0.975, n - 1))
        half = tcrit * sd / np.sqrt(n)
        rows.append(dict(est=f"E{a.est}", loss=a.loss, q=q, groups_used=n,
                         t_crit=tcrit, width_pp=200 * float(half.mean()),
                         level_group_mean=100 * float(m.mean()),
                         level_full=100 * float(pt_full.mean()),
                         offset_pp=100 * float((m - pt_full).mean()),
                         max_abs_offset_pp=100 * float(np.abs(m - pt_full).max()),
                         covers_full=int(np.all((pt_full >= m - half) & (pt_full <= m + half))),
                         n_quarters=len(pt_full)))
        paths[q] = dict(time_id=list(gs["tuniq"]), phi_t_full=list(map(float, pt_full)),
                        group_mean=list(map(float, m)), lo=list(map(float, m - half)),
                        hi=list(map(float, m + half)), groups_used=used,
                        per_group=[list(map(float, e)) for e in ests])
        print(f"  [q={q}] IM band: mean width {200*float(half.mean()):.2f}pp | "
              f"group-mean level {100*float(m.mean()):.2f}pp vs full {100*pt_full.mean():.2f}pp")

    # ---- comparison with the bootstrap bands from the same cell --------------------------
    cmp_rows = {}
    bf = root / "Rout" / f"ts_link_band_est{a.est}_uncond_{a.loss}.pkl"
    if bf.exists():
        b = pickle.load(open(bf, "rb"))
        for key, lbl in (("band_tangent_cone", "tangent-cone (link only)"),
                         ("band_tangent_cone_total", "tangent-cone (total)"),
                         ("band_total", "projected total"),
                         ("band_theta_only", "theta only")):
            if key in b:
                bb = b[key]
                cmp_rows[lbl] = 100 * float((bb["hi"] - bb["lo"]).mean())

    D = pd.DataFrame(rows)
    out_csv = OUT / f"im_phi_t_band_est{a.est}_{a.loss}.csv"
    D.to_csv(out_csv, index=False)
    if diag:
        pd.DataFrame(diag).to_csv(OUT / f"im_phi_t_groups_est{a.est}_{a.loss}.csv", index=False)
    pickle.dump(dict(paths=paths, cmp=cmp_rows, diag=diag, spec=SPEC, est=a.est, loss=a.loss),
                open(OUT / f"im_phi_t_band_est{a.est}_{a.loss}.pkl", "wb"))

    print(f"\n=========== IM GROUP-t BAND FOR phi_t — E{a.est}/{a.loss}, {SPEC} ===========")
    print(f"  {'q':>3s} {'groups':>7s} {'t(q-1)':>7s} {'IM width':>10s} "
          f"{'group mean':>11s} {'vs full':>9s} {'covers':>7s}")
    for _, r in D.iterrows():
        print(f"  {int(r.q):3d} {int(r.groups_used):7d} {r.t_crit:7.2f} {r.width_pp:9.2f}pp "
              f"{r.level_group_mean:10.2f}pp {r.offset_pp:+8.2f}pp "
              f"{'yes' if r.covers_full else 'NO':>7s}")
    if cmp_rows:
        print("\n  bootstrap bands, same cell (mean width):")
        for k, v in cmp_rows.items():
            print(f"    {k:26s} {v:8.2f}pp")
    print("\n  Read: IM shares no machinery with the bootstrap bands -- no influence functions,")
    print("  no resampling, no projection. Agreement in WIDTH is the cross-check; 'covers' asks")
    print("  whether the full-sample path lies inside the group-t interval at every quarter.")
    print(f"\nwrote {out_csv}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
