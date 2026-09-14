"""
sleep_wcb_band.py
=================
Wild-cluster-bootstrap PERCENTILE BANDS for the linear sleepiness second stage (E1, E2).

WHY THIS EXISTS.  The comparison and appendix tables report a bias-corrected percentile
interval in every column.  E3/E4 get theirs from the two-stage AME bootstrap
(sleep_ame_twostage.py).  The linear estimators stored only a standard error, and with G* of
roughly six effective clusters the t(G*) reference behind `coef +/- t*se` is itself an
approximation -- the bootstrap distribution is the object to read.  This writes that
distribution's interval to a sidecar the exporters read.

WHY A CLUSTER JOB.  The division is: the cluster computes, the exporters render.  A band is
computation, so it is produced here and travels in the sleep set like any other artifact.  No
exporter bootstraps.

WHAT IT DOES NOT DO.  It does not re-estimate.  It rebuilds the spec-12 second stage through
each estimator's OWN functions and then ASSERTS that the rebuilt coefficients reproduce the
stored ones before bootstrapping; a mismatch is a hard error, never a silently-published band
from a different sample.  The point estimates in the tables keep coming from the stored fit.

Two arms, matching how the tables read per row (utils/se_national.select_se):
  congl    -- conglomerate-clustered, the default for firm-level regressors
  quarter  -- quarter-clustered, for the national regressors (Pix, lagged Selic), which are
              constant within a quarter and whose conglomerate-clustered SE is anti-conservative

Output: <demand_prep_root>/Rout/wcb_band_est{K}_spec12.pkl
    {"meta": {est, spec, B, scheme, seed, n_obs, n_congl, n_quarter, created, counters_ok},
     "congl":   {"band": DataFrame},
     "quarter": {"band": DataFrame}}
`band` columns match the two-stage file exactly (name, ame, se, lo, hi, lo_bc, hi_bc,
p_below, p_bc, stars), so one reader serves both.

Usage
-----
    python sleep_wcb_band.py --est 1
    python sleep_wcb_band.py --est 2 --B 999 --scheme webb
"""
import argparse
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from utils.venv_guard import ensure_project_venv

ensure_project_venv(__file__)

from utils import paths as _paths                                      # noqa: E402
from utils import routines as _routines                                # noqa: E402
from utils import se_national as _sen                                  # noqa: E402
from utils.sleep_links import boot_cfg, linear_wild_cluster_bootstrap  # noqa: E402

SPEC = _routines.SPEC12
IV_NAME, S_NAME = "IV_HausmanFull", "Tech"
PARAM_TOL = 1e-10


def _estimator_api(est):
    """-> (frame, first_stage, second_stage) for a LINEAR routine.

    E1 and E2 differ in the frame (E1 drops digital banks) and in which pair of stage functions
    runs, so each is driven through its OWN module rather than a shared reimplementation that
    could drift from either.
    """
    import sleep_est_e2 as e2
    if est == 1:
        import sleep_est_e1 as e1
        return e1.build_unified_frame(), e1.run_first_stage, e1.run_second_stage
    if est == 2:
        return e2.build_pooled_data(), e2.run_pooled_first_stage, e2.run_pooled_second_stage
    raise SystemExit(f"--est {est}: this script covers the LINEAR estimators (1, 2). "
                     "E3/E4 bands come from sleep_ame_twostage.py.")


def _rebuild_one_spec(est, df0, first_stage, second_stage, iv_name, iv_cols, s_name, s_cols):
    """-> (res, periods) for ONE spec, through the estimator's own stage functions, or None.

    The appendix tables print the whole IV x state grid, not just spec 12, so a band is needed
    per cell or most of the table keeps a standard error while one column gets an interval.
    """
    has_cf = len(iv_cols) > 0
    df = df0.copy()
    if has_cf:
        iv_act = [c for c in iv_cols if c in df.columns and df[c].notnull().sum() > 0]
        exog_act = [c for c in s_cols if c in df.columns and df[c].notnull().sum() > 0]
        if not iv_act:
            return None
        df, res_fs = first_stage(df, iv_act, exog_act)
        if res_fs is None:
            return None

    # The second stage's own dropna decides the sample; recover it the same way so `periods`
    # lines up row-for-row with the fit that comes back.
    X_cols = []
    for sv in s_cols:
        col = f"interaction_{sv}" if sv != "constant" else "nr_lagged_dep"
        df[col] = df[sv] * df["nr_lagged_dep"] if sv != "constant" else df["nr_lagged_dep"]
        X_cols.append(col)
    if has_cf:
        X_cols.append("v_hat_x_lagged_dep")
    df_ss = df.dropna(subset=X_cols + ["deposit_balance"])

    kw = {"has_cf": has_cf}
    if est == 1:
        kw["spec_name"] = f"{iv_name} x {s_name}"
    res = second_stage(df, s_cols, **kw)
    if res is None:
        return None
    if len(df_ss) != int(res.nobs):
        raise SystemExit(f"est{est} / {iv_name} x {s_name}: recovered sample {len(df_ss):,} "
                         f"!= fit nobs {int(res.nobs):,}; periods would not line up.")
    return res, df_ss["time_id"]


def _load_stored(est):
    pkl = _paths.demand_prep_root() / f"est{est}" / "estimation_results.pkl"
    if not pkl.exists():
        raise SystemExit(f"est{est}: {pkl} absent -- nothing to verify the rebuild against.")
    with open(pkl, "rb") as fh:
        return pickle.load(fh)


def _assert_matches_stored(est, spec_key, stored, res):
    """The rebuilt fit must BE the stored fit, or the band describes another sample."""
    entry = stored.get(spec_key)
    if not entry or entry.get("second_stage") is None:
        return None
    a = pd.Series(entry["second_stage"].params).astype(float)
    b = pd.Series(res.params).astype(float)
    if list(a.index) != list(b.index):
        raise SystemExit(f"est{est} / {spec_key}: parameter names differ.\n"
                         f"  stored:  {list(a.index)}\n  rebuilt: {list(b.index)}")
    dev = float(np.max(np.abs(a.values - b.values)))
    rel = dev / max(1e-300, float(np.max(np.abs(a.values))))
    if rel > PARAM_TOL:
        raise SystemExit(f"est{est} / {spec_key}: rebuilt coefficients differ from the stored "
                         f"fit (rel {rel:.3e} > {PARAM_TOL:.0e}). The band would describe a "
                         "different sample; refusing to write one.")
    return dev


def main():
    ap = argparse.ArgumentParser(
        description="WCB percentile bands for the linear sleepiness second stage (E1/E2).")
    ap.add_argument("--est", type=int, required=True, choices=[1, 2])
    ap.add_argument("--B", type=int, default=None, help="default: SLEEP_BOOT_B (999)")
    ap.add_argument("--scheme", default=None, choices=["webb", "rademacher"],
                    help="default: SLEEP_BOOT_SCHEME (webb)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    B_def, sch_def = boot_cfg()
    B, scheme = (a.B or B_def), (a.scheme or sch_def)
    t0 = datetime.now()
    print("=" * 78)
    print(f"  WCB percentile bands -- est{a.est} / {SPEC} | B={B} scheme={scheme} seed={a.seed}")
    print("=" * 78)

    import sleep_est_e2 as e2
    _, iv_specs, state_blocks = e2.define_specifications(time_block=False)
    stored = _load_stored(a.est)
    df0, first_stage, second_stage = _estimator_api(a.est)

    specs = {}
    skipped = []
    for iv_name, iv_cols in iv_specs.items():
        for s_name, s_cols in state_blocks.items():
            key = f"{iv_name} x {s_name}"
            if key not in stored:
                continue                       # the estimator did not run this cell
            built = _rebuild_one_spec(a.est, df0, first_stage, second_stage,
                                      iv_name, iv_cols, s_name, s_cols)
            if built is None:
                skipped.append(key)
                continue
            res, periods = built
            dev = _assert_matches_stored(a.est, key, stored, res)
            names = list(res.params.index)

            bo = {}
            linear_wild_cluster_bootstrap(res, B=B, scheme=scheme, seed=a.seed, band_out=bo)
            if "band" not in bo:
                raise SystemExit(f"{key}: congl arm produced no band "
                                 "(non-'normal' SLEEP_WCB_MODE?).")
            X = np.asarray(res.model.exog, float)
            u = np.asarray(res.resid, float)
            qo = {}
            _sen.time_clustered_wcb(X * u[:, None], np.linalg.pinv(X.T @ X), periods,
                                    np.asarray(res.params, float), names,
                                    B=B, scheme=scheme, seed=a.seed, band_out=qo)
            if "band" not in qo:
                raise SystemExit(f"{key}: quarter arm produced no band.")

            specs[key] = {"congl": {"band": bo["band"]},
                          "quarter": {"band": qo["band"]},
                          "n_obs": int(res.nobs),
                          "n_congl": int(pd.Series(np.asarray(res.cov_kwds["groups"]))
                                         .astype(str).nunique()),
                          "n_quarter": int(qo.get("n_t", 0))}
            print(f"  [{key:34s}] n={int(res.nobs):>9,}  rebuild dev={dev:.2e}  "
                  f"rows={len(bo['band'])}")

    if not specs:
        raise SystemExit(f"est{a.est}: no spec rebuilt; nothing to write.")
    if skipped:
        print(f"  [skip] no fit rebuilt for: {', '.join(skipped)}")

    out = {"meta": dict(est=a.est, B=B, scheme=scheme, seed=a.seed,
                        specs=sorted(specs), headline=SPEC, skipped=skipped,
                        created=t0.isoformat(timespec="seconds"), counters_ok=True),
           "specs": specs}

    rout = _paths.demand_prep_root() / "Rout"
    rout.mkdir(parents=True, exist_ok=True)
    fp = rout / f"wcb_band_est{a.est}.pkl"
    with open(fp, "wb") as fh:
        pickle.dump(out, fh)

    hl = specs.get(SPEC)
    if hl:
        print(f"\n  headline {SPEC}: n_obs={hl['n_obs']:,}  congl clusters={hl['n_congl']:,}  "
              f"quarters={hl['n_quarter']}")
        for arm in ("congl", "quarter"):
            print(f"\n  --- {arm} ---")
            for _, r in hl[arm]["band"].iterrows():
                print(f"    {r['name']:42s} {r['ame']:+.6e}  "
                      f"BC[{r['lo_bc']:+.4e},{r['hi_bc']:+.4e}]  "
                      f"p_bc={r['p_bc']:.4f}{r['stars']}")
    print(f"\n  {len(specs)} spec(s) -> {fp}   "
          f"({(datetime.now() - t0).total_seconds() / 60:.1f} min)")


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
