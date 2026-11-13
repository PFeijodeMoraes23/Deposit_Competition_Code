"""
sleep_pix_genreg.py
================================================================================
The first stage of Pix use as a GENERATED REGRESSOR, under SLEEP_PIX_STATE=use_cf (utils/pix_state.py).

Under (A) the specification-12 index carries Pix use and pix_use_cf, the residual of use in its first
stage on the three 2019 exposures. The two-stage AME bootstrap of E3/E4 (sleep_ame_twostage.py) and
the linear wild cluster bootstrap of E1/E2 (sleep_wcb_band.py) treat pix_use_cf as data. This driver
adds the first stage's sampling error, nested in them:

  --refit --est N --draws a-b
        For each first-stage draw b (a wild bootstrap of the use first stage by state: 27 clusters,
        Webb weights, utils.pix_state.wild_y), refit the first stage, rebuild pix_use_cf with its
        coefficients (use itself is data), and re-estimate specification 12 with routine N's
        production estimator: E1/E2 the linear fit (sleep_est_e1.execute_specification /
        sleep_est_e2.exec_pooled_spec), E3/E4 the spread control function, the Cauchy NLLS
        direction and the monotone link (sleep_est_single._exec_spec's single-index path). One row
        of AMEs per draw, appended to <rout>/pix_genreg/genreg_E{N}_d{a}-{b}.csv (resumable).
        Draw 0 is the fit at the estimated first stage: the reproduction check.
  --combine --est N
        The 99 first-stage refits are combined with the second stage's conditional draws, the two
        stages being independent (Pix transactions and deposits are different data): draw i of the
        conditional cloud plus the deviation of refit 1 + (i mod 99) from the estimate. The BC band,
        the standard errors and the p-values are recomputed from the combined draws
        (utils.sleep_links._ame_band_from_draws).
          E3/E4: <rout>/ame_twostage_est{N}_robust_if_gen.pkl, in the structure of
                 sleep_ame_twostage.py's output, so its --attach-from attaches it unchanged.
          E1/E2: <rout>/wcb_band_est{N}_gen.pkl, sleep_wcb_band.py's sidecar with the
                 specification-12 band replaced by the combined one (both arms); the other cells are
                 copied unchanged. utils.sleep_links reads it in preference to the plain sidecar.

The market panel is read through pyarrow (sleep_attention_stakes.patch_panel_loader); nothing calls
load_panel_cached. Outputs follow SLEEP_OUT_ROOT through utils.paths.rout_dir().

    python sleep_pix_genreg.py --refit --est 3 --draws 0-9
    python sleep_pix_genreg.py --combine --est 3
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SLEEP_BOOT_B", "19")      # the link-only bootstrap of a refit; no AME depends on it

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils import pix_state as P
from utils import routines as R

SPEC = R.SPEC12
NLLS_N_STARTS = int(os.environ.get("SLEEP_NLLS_N_STARTS", "4"))
T0 = time.time()


def stamp(msg):
    print(f"[{datetime.now():%H:%M:%S} +{time.time() - T0:7.0f}s] {msg}", flush=True)


def out_dir() -> Path:
    d = _paths.rout_dir() / "pix_genreg"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _frame(est):
    """Routine est's estimation frame at the estimated first stage (production builders)."""
    from sleep_attention_stakes import patch_panel_loader
    patch_panel_loader()
    if est == 1:
        import sleep_est_e1 as E1
        df = E1.build_unified_frame()
        df["constant"] = 1.0
        s_tech, _, _ = E1.define_specifications()
        for col in s_tech:
            if col != "constant" and col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        return df
    from sleep_est_e2 import build_pooled_data
    return build_pooled_data(time_block=(est == 4))


def _fit(est, df):
    """Specification 12 of routine est on frame df -> (AMEs, the result, the second stage's rows)."""
    if est == 1:
        import sleep_est_e1 as E1
        s_tech, iv, _ = E1.define_specifications()
        _, res, _, _, _ = E1.execute_specification((df.copy(), "IV_RivalSpread", iv["IV_RivalSpread"],
                                                   "Tech", s_tech, 12))
        return res.params.copy(), res
    from sleep_est_e2 import define_specifications
    s_cols = define_specifications(time_block=(est == 4))[2]["Tech"]
    iv = define_specifications()[1]["IV_RivalSpread"]
    if est == 2:
        from sleep_est_e2 import exec_pooled_spec
        res, _, _ = exec_pooled_spec((df.copy(), "IV_RivalSpread", iv, "Tech", s_cols))
        return res.params.copy(), res
    from sleep_est_e2 import run_pooled_first_stage
    from utils.sleep_links import fit_nlls_link, fit_single_index
    d = df.copy()
    iv_act = [c for c in iv if c in d.columns and d[c].notnull().sum() > 0]
    ex_act = [c for c in s_cols if c in d.columns and d[c].notnull().sum() > 0]
    d, _ = run_pooled_first_stage(d, iv_act, ex_act)
    lg = fit_nlls_link(d, s_cols, has_cf=True, link="logit", loss="cauchy", fe_time_col=R.FE_TIME_COL,
                       bootstrap=False, n_starts=NLLS_N_STARTS)
    si = fit_single_index(d, s_cols, has_cf=True, logit_res=lg, degree=3, fe_time_col=R.FE_TIME_COL,
                          phi_band=False)
    return si.params.copy(), si


def run_refit(est, a, b):
    if P.mode() != "use_cf":
        raise SystemExit("sleep_pix_genreg.py needs SLEEP_PIX_STATE=use_cf")
    path = out_dir() / f"genreg_E{est}_d{a:03d}-{b:03d}.csv"
    done = set(pd.read_csv(path)["draw"]) if path.exists() else set()
    df = _frame(est)
    stamp(f"E{est}: frame {len(df):,} rows; draws {a}-{b} ({len(done)} already done)")
    for k in range(a, b + 1):
        if k in done:
            continue
        t0 = time.time()
        beta = P.first_stage_hat()["beta"] if k == 0 else P.first_stage(P.wild_y(k))["beta"]
        P.add_columns(df, beta=beta, where=f"genreg draw {k}")
        ame, _ = _fit(est, df)
        row = {"draw": k, "est": est, "seconds": round(time.time() - t0, 1),
               **{f"pi_{e}": float(v) for e, v in zip(P.EXPOSURES, beta[-3:])},
               **{str(n): float(v) for n, v in ame.items()}}
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
        stamp(f"E{est} draw {k}: pix_use AME {row.get('interaction_pix_use', row.get('pix_use', np.nan)):+.5g}; "
              f"{row['seconds']:.0f}s")


def _gen_cloud(est, names):
    """The 99 refits (draws 1-99) and the refit at the estimated first stage (draw 0), which centres
    the deviations: the same procedure on both sides, so a basin or thread-count difference between
    the stored fit and a refit cannot pass for first-stage noise."""
    parts = sorted(out_dir().glob(f"genreg_E{est}_d*.csv"))
    g = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).drop_duplicates("draw")
    g0 = g[g["draw"] == 0]
    g = g[g["draw"] >= 1].sort_values("draw")
    if len(g) < P.N_DRAWS or g0.empty:
        raise SystemExit(f"E{est}: {len(g)} of {P.N_DRAWS} first-stage refits on disk, draw 0 "
                         f"{'present' if not g0.empty else 'MISSING'} (run --refit --draws 0-0)")
    miss = [n for n in names if n not in g.columns]
    if miss:
        raise SystemExit(f"E{est}: refits lack {miss}")
    return g[names].to_numpy(float), g0[names].to_numpy(float)[0], [p.name for p in parts]


def _combine_cloud(D, G, G0, ame_hat, names):
    """Conditional draws D (B x K, NaN rows dropped) plus the first-stage deviations G - G0."""
    from utils.sleep_links import _ame_band_from_draws
    from scipy import stats
    D = D[np.all(np.isfinite(D), axis=1)]
    dev = G - np.asarray(G0, float)[None, :]
    C = D + dev[np.arange(len(D)) % len(dev)]
    bse = {n: float(np.std(C[:, j], ddof=1)) for j, n in enumerate(names)}
    pv = {n: (float(2 * stats.norm.sf(abs(ame_hat[n]) / bse[n])) if bse[n] > 0 else 0.0) for n in names}
    band = _ame_band_from_draws(names, ame_hat, C)
    sd_gen = {n: float(np.std(G[:, j], ddof=1)) for j, n in enumerate(names)}
    return C, band, bse, pv, sd_gen


def _linear_draws(res, groups, B, seed, scheme="webb"):
    """The conditional wild-cluster draws of a linear fit, clustered by `groups` (the normal-mode
    construction of utils.sleep_links.linear_wild_cluster_bootstrap, with the draws kept)."""
    from utils.sleep_links import cluster_wild_bootstrap, gram_pinv
    X = np.asarray(res.model.exog, float)
    u = np.asarray(res.resid, float)
    names = list(res.params.index)
    beta = np.asarray(res.params, float)
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    n_cl = int(inv.max() + 1)
    bread = gram_pinv(X.T @ X, "genreg linear")
    score = X * u[:, None]
    s_cl = np.column_stack([np.bincount(inv, weights=score[:, k], minlength=n_cl) for k in range(X.shape[1])])
    IF = s_cl @ bread.T
    ame_hat = {n: float(b) for n, b in zip(names, beta)}
    dr = {}
    cluster_wild_bootstrap(beta, IF, lambda th: {n: float(th[i]) for i, n in enumerate(names)}, ame_hat,
                           B=B, scheme=scheme, rng=np.random.default_rng(seed), mode="normal", draws_out=dr)
    return np.column_stack([np.asarray(dr[n], float) for n in names]), ame_hat


def run_combine(est):
    rout = _paths.rout_dir()
    meta_gen = dict(n_refits=P.N_DRAWS, seed=P.SEED, clusters="state (27)", weights="webb",
                    created=datetime.now().isoformat(timespec="seconds"))
    if est in (3, 4):
        src = rout / f"ame_twostage_est{est}_robust_if.pkl"
        with open(src, "rb") as fh:
            out = pickle.load(fh)
        names = list(out["meta"]["names"])
        G, G0, files = _gen_cloud(est, names)
        for s in ("congl", "quarter"):
            blk = out[s]
            if "draws_cone" not in blk:
                raise SystemExit(f"{src.name} kept no draws (run sleep_ame_twostage.py with --keep-draws)")
            D = np.column_stack([np.asarray(blk["draws_cone"][n], float) for n in names])
            C, band, bse, pv, sd_gen = _combine_cloud(D, G, G0, blk["ame"], names)
            blk.update(band=band, bse=bse, pvalues=pv, cov=np.cov(C, rowvar=False, ddof=1),
                       draws_cone={n: C[:, j] for j, n in enumerate(names)}, sd_first_stage=sd_gen)
        out["meta"]["genreg"] = dict(meta_gen, source=src.name, refit_files=files)
        dst = rout / f"ame_twostage_est{est}_robust_if_gen.pkl"
        with open(dst, "wb") as fh:
            pickle.dump(out, fh)
        r = out["congl"]["band"].set_index("name").loc["interaction_pix_use"]
        stamp(f"E{est}: wrote {dst.name}; pix_use AME {r['ame']:+.5g} BC [{r['lo_bc']:+.4g}, {r['hi_bc']:+.4g}] "
              f"p_bc {r['p_bc']:.3f}; first-stage SD {out['congl']['sd_first_stage']['interaction_pix_use']:.4g}")
        return
    src = _paths.demand_prep_root() / "Rout" / f"wcb_band_est{est}.pkl"
    with open(src, "rb") as fh:
        side = pickle.load(fh)
    df = _frame(est)
    P.add_columns(df, where="genreg combine")
    _, res = _fit(est, df)
    names = list(res.params.index)
    G, G0, files = _gen_cloud(est, names)
    B, seed = int(side["meta"].get("B", 999)), int(side["meta"].get("seed", 0))
    cl = np.asarray(res.cov_kwds["groups"])
    # The second stage's rows: statsmodels keeps the frame index of the demeaned design.
    lab = res.model.data.row_labels
    if lab is None or len(lab) != int(res.nobs):
        raise SystemExit(f"E{est}: cannot recover the second stage's rows")
    periods = df.loc[lab, "time_id"].to_numpy()
    new = {}
    for arm, groups in (("congl", cl), ("quarter", periods)):
        D, ame_hat = _linear_draws(res, groups, B, seed)
        _, band, _, _, sd_gen = _combine_cloud(D, G, G0, ame_hat, names)
        new[arm] = {"band": band, "sd_first_stage": sd_gen}
    side["specs"][SPEC] = dict(side["specs"].get(SPEC, {}), **new)
    side["meta"]["genreg"] = dict(meta_gen, source=src.name, refit_files=files, spec=SPEC)
    dst = src.with_name(f"wcb_band_est{est}_gen.pkl")
    with open(dst, "wb") as fh:
        pickle.dump(side, fh)
    stamp(f"E{est}: wrote {dst.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--refit", action="store_true")
    g.add_argument("--combine", action="store_true")
    p.add_argument("--est", type=int, required=True, choices=(1, 2, 3, 4))
    p.add_argument("--draws", default="0-2", help="--refit: a-b inclusive; 0 is the estimate, 1-99 the draws")
    a = p.parse_args()
    pd.options.mode.chained_assignment = None
    stamp(f"sleep_pix_genreg | SLEEP_PIX_STATE={P.mode()} | out {out_dir()}")
    if a.refit:
        lo, hi = (int(x) for x in a.draws.split("-"))
        if not (0 <= lo <= hi <= P.N_DRAWS):
            p.error(f"--draws must lie in 0-{P.N_DRAWS}")
        run_refit(a.est, lo, hi)
    else:
        run_combine(a.est)


if __name__ == "__main__":
    main()
