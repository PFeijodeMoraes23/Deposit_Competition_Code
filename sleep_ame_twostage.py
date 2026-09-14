"""
sleep_ame_twostage.py
================================================================================
TWO-STAGE (direction + link) wild cluster bootstrap of the E3/E4 average marginal
effects, from STORED fits -- no re-estimation. Spec 12 (the headline, `--spec` default)
or any other IV x state cell of the appendix grid (`--spec KEY|others|all`).

Why: the SEs on the estimator pickles bootstrap the LINK ONLY. The index direction
theta-hat is fitted by the Cauchy NLLS-logit and then frozen, so every continuous
AME is (theta_j/sd_j) * mean_slope(beta) -- one fixed scalar times one random
object. The scalar cancels out of the t-ratio and every continuous row of a column
reports the same |t| (1.313121091 on est3 spec 12, 1.209007720 on est4). This
driver perturbs BOTH stages with one shared weight vector per draw -- the direction
by its cluster influence functions, the link by an exact re-profile at the drawn
index -- so each row gets a t that includes direction uncertainty. See
utils.sleep_links.twostage_ame_boot, and _stage_a_perturbed_solve for what
`--theta-mode newton` measures and why it is a diagnostic rather than the report.

THE OFF SWITCH: `--theta-off` short-circuits the direction channel and reproduces
the stored conditional numbers. It is the regression guard for everything here and
should be run on every change to the touched functions (~5 min per routine).

Reads the estimator pickle from utils.paths.demand_prep_root() -- set
SLEEP_OUT_ROOT to run against the sandbox. Writes standalone result pickles to
    <root>/Rout/ame_twostage_est{N}_{robust,ls}[_{off,if}].pkl              (spec 12)
    <root>/Rout/ame_twostage_est{N}_{robust,ls}[_{off,if}]__{IV}_x_{S}.pkl  (other cells)
and never touches the estimator pickle unless --attach is passed (off by default;
makes a timestamped .bak first, and writes NEW attributes plus cov_ame only).

Usage:
    python sleep_ame_twostage.py --est 3 --loss robust --B 999 --workers 6
    python sleep_ame_twostage.py --est 3 --loss robust --theta-off
    python sleep_ame_twostage.py --est 3 --spec others          # the seven non-headline cells
    python sleep_ame_twostage.py --est 3 --spec "OLS x Macro" --theta-off
"""
import argparse
import os

# Pin the BLAS pools BEFORE numpy is imported: with `--workers N` the children inherit this
# environment, and N workers x a full BLAS pool oversubscribes the 10 physical cores and runs
# SLOWER than serial. Pinning to 1 also fixes the reduction order inside D'D, which is what lets
# the parallel run reproduce the serial one bit-for-bit.
# SLEEP_AME_BLAS_THREADS exists because the stored fits were produced under the pipeline's
# 2-thread setting, and a multithreaded reduction inside the first stage changes v_hat in the
# last ulp -- which is the difference between reproducing the stored SEs exactly and to ~1e-10.
_THR = os.environ.get("SLEEP_AME_BLAS_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _THR
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import pickle
import shutil
from pathlib import Path
import time
from datetime import datetime

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd

from utils import paths as _paths_mod
from utils import routines as _routines
from utils.sleep_links import boot_cfg, twostage_ame_boot

SPEC = _routines.SPEC12                 # spec 12, the headline cell and the --spec default
FE_TIME_COL = _routines.FE_TIME_COL
LOSS_OF = {"robust": "cauchy", "ls": "linear"}
KEY_OF = {"robust": "second_stage", "ls": "second_stage_ls"}

# Section-1 fixtures: the shared conditional |t| of the promoted local fit, and the Pix dummy's,
# which escapes the cancellation. They cross-check that these constants were transcribed right --
# they are NOT the gate's verdict, which is measured against whatever pickle was loaded. The
# lineup can be re-estimated on the cluster with a wider multistart, which may settle in another
# basin and carry its own conditional |t| while the two-stage machinery is still exact;
# off_path_gate applies the fixtures only when the stored values are already this vintage.
FIXTURES = {
    3: dict(congl_shared=1.3131210909941156, congl_pix=-1.3745817099672621,
            quarter_shared=1.2327854424803466, quarter_pix=-1.2669177460429395),
    4: dict(congl_shared=1.2090077201757856, congl_pix=-1.266188013333496,
            quarter_shared=1.1616393366109388, quarter_pix=-1.195428109199919),
}


def default_workers():
    """Draws are independent and each costs ~8.3 s serially (measured, est3 spec 12, n=487k,
    --theta-mode if), so the draw loop is the whole job. Leave two logical cores for the OS and
    the parent; SLEEP_AME_BOOT_JOBS overrides. Peak RSS is ~1.5-3 GB per worker.

    Workers pay a large one-off startup -- each child re-imports this module's graph from the
    OneDrive-backed repo and unpickles the ~55 MB context -- so pass `--workers 1` for anything
    short. Measured at B=6 the 4-worker run was 0.87x, i.e. SLOWER than serial; the startup is
    amortised only at production B. The number is printed at run time."""
    env = os.environ.get("SLEEP_AME_BOOT_JOBS")
    if env:
        return max(1, int(env))
    return max(1, min(6, (os.cpu_count() or 2) - 2))


def _spec_slug(spec):
    """'OLS x Macro' -> 'OLS_x_Macro', the suffix a non-headline cell's result file carries."""
    return spec.replace(" x ", "_x_").replace(" ", "_")


def _out_path(rout, est, loss_lbl, tag, spec):
    """Spec 12 keeps its original file name -- gate G3, the archive and --attach-from read it
    by that name -- and every other cell is suffixed, so the two can never overwrite each other."""
    stem = f"ame_twostage_est{est}_{loss_lbl}" + (f"_{tag}" if tag else "")
    if spec != SPEC:
        stem += f"__{_spec_slug(spec)}"
    return rout / f"{stem}.pkl"


def _prep_frame(time_block, spec, cache):
    """Rebuild the estimation frame for one IV x state cell exactly as sleep_est_single._exec_spec:
    pooled panel -> (IV cells only) first stage with that cell's instruments and active exogenous
    controls. Returns (df_t, s_cols, has_cf).

    The pooled panel is the expensive part and is identical across cells, so it is built once
    and held in `cache`; the first stage is per cell. run_pooled_first_stage writes its v_hat
    columns onto the frame it is given, so each IV cell works on a COPY of the cached panel.
    Whether the rebuilt frame is the one the fit was estimated on is not assumed: twostage_ame_boot
    refuses on any vmu/vsd, link or AME fingerprint mismatch."""
    from sleep_est_e2 import (build_pooled_data, define_specifications,
                                    run_pooled_first_stage)
    if "df" not in cache:
        cache["df"] = build_pooled_data(time_block=time_block)
    df = cache["df"]
    iv_name, s_name = (p.strip() for p in spec.split(" x "))
    _, iv_specs, state_blocks = define_specifications(time_block=time_block)
    s_cols = state_blocks[s_name]
    iv_cols = iv_specs[iv_name]
    has_cf = len(iv_cols) > 0
    if not has_cf:
        return df, s_cols, False
    iv_act = [c for c in iv_cols if c in df.columns and df[c].notnull().sum() > 0]
    exog_act = [c for c in s_cols if c in df.columns and df[c].notnull().sum() > 0]
    if not iv_act:
        raise SystemExit(f"{spec}: no active instrument in the rebuilt panel")
    df_t, _ = run_pooled_first_stage(df.copy(), iv_act, exog_act)
    return df_t, s_cols, True


def off_path_gate(est, si_res, out, tol=1e-8, spec=SPEC):
    """BLOCKING GATE for `--theta-off`: reproduce the loaded fit's OWN four stored series.

    The OFF path is a structural short-circuit of the draw loop, so it should land bit-for-bit
    on the numbers the conditional bootstrap wrote onto this pickle; the achieved maximum
    relative deviation is reported rather than rounded up. `bse`/`pvalues` come from the
    conglomerate loop, `bse_time`/`pvalues_time` from the quarter loop that follows it and
    inherits its projection warm cell. The derived t the tables print is params/bse, which is
    exactly the `tvalues` stored on the conglomerate scheme; the quarter scheme stores no t and
    forms it off `bse_time` the same way the exporters do.

    Every reference is read off `si_res`, so the verdict is vintage-independent: it says the
    two-stage machinery reduces to whatever conditional bootstrap produced the fit, and it holds
    on a cluster-estimated pickle as well as on the promoted local one. FIXTURES then cross-check
    the hardcoded constants, but only on the vintage they were taken from."""
    gate = dict(checked=True, ok=True, worst=0.0, detail={}, fixtures={})
    pairs = (("bse", "congl", "bse"), ("pvalues", "congl", "pvalues"),
             ("bse_time", "quarter", "bse"), ("pvalues_time", "quarter", "pvalues"))
    for attr, scheme, key in pairs:
        stored = getattr(si_res, attr, None)
        if stored is None:
            gate["detail"][attr] = None
            continue
        got = out[scheme][key]
        d = max(abs(float(got[k]) - float(stored[k])) / max(1e-300, abs(float(stored[k])))
                for k in got)
        gate["detail"][attr] = d
        gate["worst"] = max(gate["worst"], d)
    # derived t: what the tables actually print
    for scheme, attr in (("congl", "bse"), ("quarter", "bse_time")):
        stored = getattr(si_res, attr, None)
        if stored is None:
            continue
        t_got = {k: float(si_res.params[k]) / float(out[scheme]["bse"][k]) for k in out[scheme]["bse"]}
        t_ref = {k: float(si_res.params[k]) / float(stored[k]) for k in t_got}
        d = max(abs(t_got[k] - t_ref[k]) / max(1e-300, abs(t_ref[k])) for k in t_got)
        gate["detail"][f"t[{scheme}]"] = d
        gate["worst"] = max(gate["worst"], d)
        if spec != SPEC:                   # the fixtures were read off spec 12 and only there
            continue
        fx = FIXTURES[est]
        cont = sorted((abs(t_got[k]) for k in t_got if "pix" not in k))
        pix = [t_got[k] for k in t_got if "pix" in k][0]
        ref_shared = fx["congl_shared"] if scheme == "congl" else fx["quarter_shared"]
        ref_pix = fx["congl_pix"] if scheme == "congl" else fx["quarter_pix"]
        # `applies` is decided from the STORED t, never from the recomputed one: the fixtures
        # can only certify the constants on the fit they were read from, and a fit that carries
        # different conditional |t| is a different vintage, not a regression. When they do not
        # apply their deviations are still recorded, they just stay out of the verdict.
        cont_stored = sorted(abs(t_ref[k]) for k in t_ref if "pix" not in k)
        pix_stored = [t_ref[k] for k in t_ref if "pix" in k][0]
        stored_dev = max(max(abs(c - ref_shared) / ref_shared for c in cont_stored),
                         abs(pix_stored - ref_pix) / abs(ref_pix))
        gate["fixtures"][scheme] = dict(
            applies=bool(stored_dev <= tol), stored_dev=stored_dev,
            shared=cont, shared_spread=float(cont[-1] - cont[0]),
            shared_dev=max(abs(c - ref_shared) / ref_shared for c in cont),
            pix=pix, pix_dev=abs(pix - ref_pix) / abs(ref_pix))
    gate["ok"] = bool(gate["worst"] <= tol
                      and all(v["shared_dev"] <= tol and v["pix_dev"] <= tol
                              for v in gate["fixtures"].values() if v["applies"]))
    print(f"  [gate OFF] max relative deviation vs stored: {gate['worst']:.3e}  "
          f"{'REPRODUCED' if gate['ok'] else 'MISMATCH'}")
    for k, v in gate["detail"].items():
        if v is not None:
            print(f"      {k:12s} {v:.3e}")
    for s, v in gate["fixtures"].items():
        print(f"      fixture[{s}] shared |t| = "
              + ", ".join(f"{c:.9f}" for c in v["shared"])
              + f"  (dev {v['shared_dev']:.2e}); pix {v['pix']:.11f} (dev {v['pix_dev']:.2e})"
              + ("" if v["applies"] else "   [not applied]"))
        if not v["applies"]:
            print(f"      [gate OFF] fixture[{s}] SKIPPED (not a failure): the stored fit's own "
                  f"|t| sits {v['stored_dev']:.2e} from the est{est} constants, so this pickle is "
                  f"a different vintage; the stored-value check above is the verdict.")
    return gate


# Share of B draws whose re-estimated index may point against the point estimate's before the
# cloud is refused. Those draws are RETAINED -- dropping them would select the interval.
COS_IDX_NEG_MAX_SHARE = 0.01
# A draw SD at or below this multiple of |AME| is no sampling variation (see counters_ok).
DEGENERATE_SE_REL = 1e-10


def counters_ok(out, B):
    """G10: refuse to write or attach when the draw cloud is contaminated.

    The direction test is `n_cos_idx_neg` -- draws whose INDEX (X @ theta) is anti-correlated
    with the point index. That is the object the link and every AME see, and the monotone link
    cannot represent a reversed index faithfully. A handful is sampling noise in the direction:
    a 95% percentile interval is set by roughly the B/40 most extreme draws in each tail, so a
    few reversed draws move an endpoint by a few order statistics at most. Up to
    COS_IDX_NEG_MAX_SHARE of B therefore passes, with the draws kept in the cloud and their count
    reported in the table note; more than that means the index direction is badly determined and
    the cloud is refused. `n_vsd_fail` and `n_drop` stay zero-tolerance.

    A DEGENERATE arm -- some AME whose draws do not vary -- is refused as well. Its standard
    error is zero, its p-value zero and its stars three, so it would print as the most precise
    number in the table while describing no sampling variation at all. The conditional bootstrap
    stored on the E3/E4 `OLS x Macro` fits does exactly that on its quarter arm (bse_time ~1e-20
    on every row); this keeps a two-stage cloud from doing the same silently.

    `n_cos_neg` -- the cosine between the raw COEFFICIENT vectors -- is reported but does not
    gate. Its norm is dominated by the largest-unit loading (the lagged Selic rate holds ~87%
    of ||theta||^2 on spec 12), so it fires when one weakly-determined national coefficient
    changes sign while the index is intact. That is uncertainty the interval should carry, not
    contamination. See the kernel comment in utils/sleep_links.py for the measured evidence.

    A cloud produced before `n_cos_idx_neg` existed has no such key; it is refused as `missing`
    rather than silently passing on a test that never ran.
    """
    bad = []
    for s in ("congl", "quarter"):
        r = out.get(s)
        if r is None:
            bad.append(f"{s}: missing")
            continue
        if r["n_newton_fail"] > 0.01 * B:
            bad.append(f"{s}: n_newton_fail={r['n_newton_fail']} > 1% of B")
        if r["n_fail"] > 0.01 * B:
            bad.append(f"{s}: n_fail={r['n_fail']} > 1% of B")
        if "n_cos_idx_neg" not in r:
            bad.append(f"{s}: n_cos_idx_neg missing (cloud predates the index-space "
                       "direction test; re-run the bootstrap)")
        if r.get("n_cos_idx_neg", 0) > COS_IDX_NEG_MAX_SHARE * B:
            bad.append(f"{s}: n_cos_idx_neg={r['n_cos_idx_neg']} > {COS_IDX_NEG_MAX_SHARE:.0%} of B")
        for k in ("n_vsd_fail", "n_drop"):
            if r.get(k):
                bad.append(f"{s}: {k}={r[k]}")
        flat = [n for n, se in (r.get("bse") or {}).items()
                if not se > DEGENERATE_SE_REL * max(1e-300, abs(float(r["ame"][n])))]
        if flat:
            bad.append(f"{s}: degenerate draws (sd <= {DEGENERATE_SE_REL:.0e} x |AME|) on {flat}")
    return bad


def run_cell(est, loss_lbl, B, theta_off, theta_mode, workers, attach, keep_draws,
             spec=SPEC, cache=None, seed=0):
    t0 = time.time()
    cache = {} if cache is None else cache
    root = _paths_mod.demand_prep_root()
    pkl_path = root / f"est{est}" / "estimation_results.pkl"
    print(f"\n=== est{est} / {spec} / {loss_lbl} | B={B} | "
          f"{'theta OFF (conditional)' if theta_off else f'theta ON ({theta_mode})'} | "
          f"workers={1 if theta_off else workers} | root={root} ===")
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    si_res = (d.get(spec) or {}).get(KEY_OF[loss_lbl])
    if si_res is None:
        print(f"  no {KEY_OF[loss_lbl]} on {spec}; skipped")
        return None

    df_t, s_cols, has_cf = _prep_frame(time_block=(est == 4), spec=spec, cache=cache)

    out = twostage_ame_boot(df_t, s_cols, has_cf, si_res, LOSS_OF[loss_lbl], degree=3,
                            fe_time_col=FE_TIME_COL, B=B, seed=seed,
                            theta_channel=not theta_off, theta_mode=theta_mode,
                            keep_draws=keep_draws, workers=(1 if theta_off else workers))
    out["meta"].update(est=est, loss_label=loss_lbl, spec=spec,
                       driver_runtime_s=time.time() - t0,
                       created=datetime.now().isoformat(timespec="seconds"))

    if theta_off:
        out["meta"]["off_path_gate"] = off_path_gate(est, si_res, out, spec=spec)
    else:
        bad = counters_ok(out, B)
        out["meta"]["counters_ok"] = not bad
        if bad:
            print("  [gate] DEGRADED draw cloud: " + "; ".join(bad))
        for s in ("congl", "quarter"):
            bnd = out[s]["band"]
            print(f"  --- {s} ---")
            for _, r in bnd.iterrows():
                t = r["ame"] / r["se"] if r["se"] > 0 else np.nan
                print(f"    {r['name']:42s} ame={r['ame']:+.6e} se={r['se']:.3e} "
                      f"|t|={abs(t):.9f}  BC[{r['lo_bc']:+.4e},{r['hi_bc']:+.4e}] "
                      f"p_bc={r['p_bc']:.4f}{r['stars']}")

    rout = root / "Rout"
    rout.mkdir(parents=True, exist_ok=True)
    tag = "off" if theta_off else ("if" if theta_mode == "if" else "")
    fp = _out_path(rout, est, loss_lbl, tag, spec)
    with open(fp, "wb") as fh:
        pickle.dump(out, fh)
    print(f"  saved -> {fp}   ({(time.time()-t0)/60:.1f} min)")

    if attach and not theta_off:
        if _attach_into(d, spec, si_res, loss_lbl, out, B):
            _save_with_backup(pkl_path, d)
    elif attach:
        print("  [attach] refused: --theta-off produces the conditional numbers already stored")
    return out


def _attach_into(d, spec, si_res, loss_lbl, out, B):
    """Put the two-stage results onto one stored fit held in `d` (in memory; the caller saves).
    NEW attributes only, plus cov_ame -- the one pre-existing attribute this touches, and None on
    every stored E3/E4 fit before its first attach. Returns whether it attached.

    bse/pvalues/bse_time/pvalues_time are deliberately NOT overwritten: `tvalues` was frozen at
    construction as params/bse, so every downstream consumer of the stored object would silently
    change meaning. The exporters read the two-stage numbers through
    utils.se_national.select_se under SLEEP_AME_SE=twostage instead."""
    bad = counters_ok(out, B)
    gates_ok = (out["meta"].get("invariance_gate", {}).get("verdict") == "PASS"
                and out["meta"].get("sign_gate", {}).get("verdict") == "OK")
    if bad or not gates_ok:
        print(f"  [attach] {spec}: refused: gates={gates_ok} counters={bad}")
        return False
    names = out["meta"]["names"]
    cov = np.asarray(out["congl"]["cov"], float)
    bse2 = pd.Series(out["congl"]["bse"])
    d_cov = float(np.max(np.abs(np.sqrt(np.diag(cov)) - bse2[names].values)))
    if d_cov > 1e-12 * float(bse2.max()):
        print(f"  [attach] {spec}: refused: cov_ame and bse_2s describe different runs ({d_cov:.3e})")
        return False
    si_res.ame_boot = {s: {k: v for k, v in out[s].items()
                           if k not in ("draws_cone", "draws_level")}
                       for s in ("congl", "quarter")}
    si_res.bse_2s = bse2
    si_res.pvalues_2s = pd.Series(out["congl"]["pvalues"])
    si_res.bse_time_2s = pd.Series(out["quarter"]["bse"])
    si_res.pvalues_time_2s = pd.Series(out["quarter"]["pvalues"])
    si_res.ame_2s_meta = out["meta"]
    si_res.cov_ame = pd.DataFrame(cov, index=names, columns=names)
    d[spec][KEY_OF[loss_lbl]] = si_res
    print(f"  [attach] {spec}: bse_2s / pvalues_2s / bse_time_2s / pvalues_time_2s / ame_boot / "
          f"cov_ame")
    return True


def _save_with_backup(pkl_path, d):
    """One timestamped backup, then one write, however many cells were attached."""
    bak = pkl_path.with_suffix(f".pkl.bak_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(pkl_path, bak)
    with open(pkl_path, "wb") as fh:
        pickle.dump(d, fh)
    print(f"  [attach] wrote {pkl_path.name} (backup: {bak.name})")


def attach_from(est, loss_lbl, src_paths):
    """Attach PREVIOUS runs' saved results to the estimator pickle, without recomputing.

    Each file names its own cell in `meta.spec`, so the headline file and the grid files can be
    passed together; the pickle is backed up once and written once, after every file has been
    through the gates.

    The bootstrap is expensive, and its conglomerate arm is worker-count dependent: the shape
    projection is warm-started from the previous draw, so the QP's active-set path -- and the
    numbers -- depend on how the draws were chunked across workers. Recomputing locally to
    attach results that already exist would therefore spend hours to land on a DIFFERENT draw
    path than the run being reported. This reads the saved run instead and puts it through the
    SAME gates --attach applies, so nothing reaches a table that a live run would have refused.
    """
    root = _paths_mod.demand_prep_root()
    pkl_path = root / f"est{est}" / "estimation_results.pkl"
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    n_ok = 0
    seen = set()
    for src_path in src_paths:
        src = Path(src_path)
        with open(src, "rb") as fh:
            out = pickle.load(fh)
        m = out.get("meta", {})
        spec = m.get("spec", SPEC)
        # Refuse a mismatched pairing loudly: silently attaching E3's bootstrap to E4's fit, or
        # one cell's to another's, would produce a table that looks entirely normal.
        if int(m.get("est", est)) != int(est):
            raise SystemExit(f"--attach-from: {src.name} holds est{m.get('est')}, not est{est}")
        if m.get("loss_label", loss_lbl) != loss_lbl:
            raise SystemExit(f"--attach-from: {src.name} is loss={m.get('loss_label')!r}, "
                             f"not {loss_lbl!r}")
        if not m.get("theta_channel", True):
            raise SystemExit(f"--attach-from: {src.name} was produced with the direction channel "
                             "OFF (--theta-off); those are the conditional numbers already stored")
        if spec in seen:
            raise SystemExit(f"--attach-from: two files for {spec!r}")
        seen.add(spec)
        si_res = (d.get(spec) or {}).get(KEY_OF[loss_lbl])
        if si_res is None:
            raise SystemExit(f"--attach-from: no {KEY_OF[loss_lbl]} on {spec!r} in {pkl_path}")
        B_used = int(out["congl"].get("B_used", m.get("B", 0)))
        print(f"=== attach-from {src.name} -> est{est}/{spec}/{KEY_OF[loss_lbl]} | B={B_used} "
              f"theta_mode={m.get('theta_mode')!r} workers={m.get('workers')} "
              f"created={m.get('created')} ===")
        n_ok += bool(_attach_into(d, spec, si_res, loss_lbl, out, B_used))
    if n_ok:
        _save_with_backup(pkl_path, d)
    print(f"  [attach] {n_ok}/{len(src_paths)} file(s) attached")


def _resolve_specs(est, loss_lbl, spec_arg):
    """--spec -> the list of cells to run. A cell key runs that cell; `all` every IV x state cell
    the stored pickle carries a fit for; `others` the same minus the headline."""
    if spec_arg not in ("all", "others"):
        return [spec_arg]
    pkl_path = _paths_mod.demand_prep_root() / f"est{est}" / "estimation_results.pkl"
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    cells = [k for k, e in d.items()
             if " x " in str(k) and isinstance(e, dict) and e.get(KEY_OF[loss_lbl]) is not None]
    if spec_arg == "others":
        cells = [k for k in cells if k != SPEC]
    print(f"  --spec {spec_arg}: {len(cells)} cell(s): {cells}")
    return cells


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    p.add_argument("--est", type=int, required=True, choices=tuple(_routines.LINK_ESTS))
    p.add_argument("--loss", choices=("robust", "ls", "both"), default="robust")
    p.add_argument("--spec", default=SPEC,
                   help=f"cell to bootstrap: a key such as 'OLS x Macro', 'all', or 'others' "
                        f"(every stored cell but the headline). Default: {SPEC!r}")
    p.add_argument("--B", type=int, default=None,
                   help="default: SLEEP_BOOT_B (999), matching the stored conditional numbers")
    p.add_argument("--theta-mode", choices=("if", "newton"), default="if",
                   help="'if' (reported) perturbs the direction by its cluster influence "
                        "functions; 'newton' iterates the perturbed score equation by "
                        "Levenberg-Marquardt, which does not converge at these perturbation "
                        "sizes -- a diagnostic, not a reported path")
    p.add_argument("--theta-off", action="store_true",
                   help="structural short-circuit of the direction channel: reproduces the "
                        "stored conditional SEs. The regression guard.")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-draws", action="store_true", default=True)
    p.add_argument("--no-keep-draws", dest="keep_draws", action="store_false")
    p.add_argument("--attach", action="store_true")
    p.add_argument("--attach-from", metavar="PKL", nargs="+", default=None,
                   help="attach previous runs' Rout/ame_twostage_*.pkl to the estimator pickle "
                        "instead of recomputing; each file's own meta names its cell. Same gates "
                        "as --attach. Use this for cluster results: the canonical run is B=999 "
                        "on 64 workers and a local recompute would land on a different draw path.")
    a = p.parse_args()
    B = a.B if a.B is not None else boot_cfg()[0]
    workers = a.workers if a.workers is not None else default_workers()
    losses = ("robust", "ls") if a.loss == "both" else (a.loss,)
    if a.attach_from:
        for ll in losses:
            attach_from(a.est, ll, a.attach_from)
        return
    cache = {}
    failed = []
    for ll in losses:
        specs = _resolve_specs(a.est, ll, a.spec)
        for spec in specs:
            try:
                run_cell(a.est, ll, B, a.theta_off, a.theta_mode, workers, a.attach,
                         a.keep_draws, spec=spec, cache=cache, seed=a.seed)
            except (RuntimeError, SystemExit) as exc:
                # One cell's fingerprint refusal must not cost the others their run when the grid
                # is requested; the job still exits nonzero below, so the gate sees it.
                if len(specs) == 1:
                    raise
                print(f"  [!] {spec} / {ll} FAILED: {type(exc).__name__}: {exc}")
                failed.append(f"{spec}/{ll}")
    if failed:
        raise SystemExit(f"{len(failed)} cell(s) failed: {failed}")


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
