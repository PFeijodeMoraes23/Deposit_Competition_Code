"""
sleep_est_single.py
================================================================================
Shared runner for the pooled sleepiness estimators of the lineup. Each estimator is
one choice of (kind, time_block), listed in EST_CONFIG at the bottom:

    E3  single-index          (kind="single_index", time_block=False)
    E4  single-index + Time   (kind="single_index", time_block=True)

E1 (local linear, B-type) and E2 (pooled linear) are separate estimators with their
own scripts, sleep_est_e1.py / sleep_est_e2.py.

The "+Time" variant adds the time block to every state block via
estimation_2_sleep.define_specifications(time_block=True). That block is gdp_growth_yoy
alone -- add_time_variables also builds time_trend on the frame, but TIME_VARS excludes
it, because a pure-time linear trend is collinear with the quarter fixed effects that
absorb aggregate time additively. phi is
always built from the native index + link (phi_from_native); AMEs are reporting-only.
Inference: score/multiplier wild cluster bootstrap (utils.sleep_links).

THREE RUN MODES (see the CLI block at the bottom):

  default              the whole grid in one process, serially, merging into
                       est{N}/estimation_results.pkl after every state block.
  --spec-id S          exactly ONE spec, written to est{N}/_specs/spec_{S}.pkl and
                       NOTHING else. Never reads or writes the shared pickle, so an
                       arbitrary number of these can run concurrently (one SLURM
                       array task per spec).
  --merge-specs        fold every est{N}/_specs/spec_*.pkl into the shared pickle and
                       rebuild the derived outputs. THE ONLY writer of
                       estimation_results.pkl, ts_link_band_est{N}.pkl and the phi CSVs
                       in the split workflow.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pickle
import tempfile
from datetime import datetime, timezone
from functools import reduce

try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except Exception:
    pass

import numpy as np
import pandas as pd

from sleep_est_e2 import build_pooled_data, define_specifications, run_pooled_first_stage
from utils.sleep_links import fit_nlls_link, fit_single_index, phi_from_native
from utils import paths as _paths_mod
from utils import routines as R

# kind -> phi_from_native link key. Fallback only: _calculate_phis prefers each
# result's own .link tag, which distinguishes the constrained single-index
# ("index_sieve") from the cubic ("index") within the same kind.
LINK_OF = {"logit": "logit", "single_index": "index"}

_IV_ORDER = ["OLS", "IV_CostShifters", "IV_Wholesale", "IV_HausmanFull"]

# Two-way fixed effects: the sleepiness model now carries BOTH an entity FE
# (bank x deposit-type x market) and a quarter time FE (time_id). The time FE
# absorbs aggregate time shocks additively, outside the link. Set to None to
# fall back to the entity-only within estimator.
FE_TIME_COL = R.FE_TIME_COL

# Starts for the NLLS logit that supplies the single-index direction. It matters because
# fit_single_index never re-optimises theta, so E3/E4 take whatever direction this fit lands
# on -- and scored on the full sample against alternative index directions, the zeros-start
# logit direction comes last of four (164,913 against 160,940). Sequential, so each extra
# start costs one more least_squares solve; 4 is a reasonable default.
NLLS_N_STARTS = int(os.environ.get("SLEEP_NLLS_N_STARTS", "4"))
# Opt 8: drop the LS loss during the grid (robust feeds phi). Set DROP_LS=0 to keep it.
DROP_LS = os.environ.get("SLEEP_DROP_LS", "1") != "0"
# SLEEP_LS_ONLY=1: compute ONLY the LS variant (res_robust=None); the saved robust results are
# preserved by merge-on-save. Requires SLEEP_DROP_LS=0 to have any effect. See _exec_spec.
LS_ONLY = os.environ.get("SLEEP_LS_ONLY", "0") == "1"
# SLEEP_AME_TWOSTAGE=1 runs the two-stage (direction + link) AME bootstrap inside the estimation
# pass, on the frame the fit was just made on, and attaches its results to the stored object --
# the same numbers sleep_ame_twostage.py produces post hoc from the pickle, without the
# frame rebuild. OFF by default: it costs ~2 x B re-profiled draws per cell (hours at B=999),
# whereas the estimator's own link-only bootstrap costs minutes. Spec 12 only unless
# SLEEP_AME_TWOSTAGE_ALL=1. Workers come from SLEEP_AME_BOOT_JOBS.
TWOSTAGE_AME = os.environ.get("SLEEP_AME_TWOSTAGE", "0") == "1"
TWOSTAGE_AME_ALL = os.environ.get("SLEEP_AME_TWOSTAGE_ALL", "0") == "1"


def _attach_twostage_ame(si_res, df_target, s_cols, has_cf, loss, spec_name):
    """Two-stage AME bootstrap on a fit that was just made, attached in place.

    Writes NEW attributes only (plus cov_ame, which is None on every single-index fit): the
    stored bse/pvalues/bse_time/pvalues_time keep their link-only meaning, because `tvalues` is
    frozen at construction as params/bse and every downstream consumer reads it. The exporters
    pick the two-stage numbers up through utils.se_national.select_se under SLEEP_AME_SE."""
    from utils.sleep_links import twostage_ame_boot
    jobs = int(os.environ.get("SLEEP_AME_BOOT_JOBS", "1"))
    print(f"  [2s-AME] {spec_name} / {loss}: two-stage AME bootstrap (workers={jobs})")
    try:
        out = twostage_ame_boot(df_target, s_cols, has_cf, si_res, loss, degree=3,
                                fe_time_col=FE_TIME_COL, workers=jobs, keep_draws=False)
    except Exception as e:
        print(f"  [2s-AME] SKIPPED ({type(e).__name__}: {e})")
        return
    si_res.ame_boot = {s: out[s] for s in ("congl", "quarter")}
    si_res.bse_2s = pd.Series(out["congl"]["bse"])
    si_res.pvalues_2s = pd.Series(out["congl"]["pvalues"])
    si_res.bse_time_2s = pd.Series(out["quarter"]["bse"])
    si_res.pvalues_time_2s = pd.Series(out["quarter"]["pvalues"])
    si_res.ame_2s_meta = out["meta"]
    _nm = out["meta"]["names"]
    si_res.cov_ame = pd.DataFrame(out["congl"]["cov"], index=_nm, columns=_nm)


def _out_dir(est_num):
    # demand_prep_root() honours SLEEP_OUT_ROOT, so a whole grid can be re-estimated into a
    # sandbox without touching production results. Unset => the historical path exactly.
    d = _paths_mod.demand_prep_root() / f"est{est_num}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Spec identity ────────────────────────────────────────────────────────────────
# The grid is BLOCK-MAJOR: the state blocks in define_specifications order (Base, Macro,
# Tech), and inside each block the four instrument sets in _IV_ORDER. That is exactly the
# order the serial runner visits them in (_exec_block over block_tasks, _IV_ORDER inside),
# and numbering it 1-based reproduces the repo-wide spec ids of
# estimation_demand_link_common.SPEC_MAP -- 1-4 Base, 5-8 Macro, 9-12 Tech, so "spec 12" is
# IV_HausmanFull x Tech everywhere: here, in the demand parquets, and in the CF exports.
# _check_spec_ids() enforces that agreement rather than trusting it.
#
# The single-index kinds do not estimate the Base block (a constant-only index has no
# direction), so ids 1-4 are absent from their grid; the remaining ids keep their canonical
# value instead of being renumbered 1-8.
_SPECS_SUBDIR = "_specs"


def spec_grid(kind, time_block=False):
    """-> [(spec_id, iv_name, state_block_name)] in the serial runner's own order."""
    _, _iv, state_blocks = define_specifications(time_block=time_block)
    grid = [(bi * len(_IV_ORDER) + ii + 1, iv, s_name)
            for bi, s_name in enumerate(state_blocks)
            for ii, iv in enumerate(_IV_ORDER)]
    return [g for g in grid if not (g[2] == "Base" and kind != "logit")]


def spec_name_of(iv_name, s_name):
    """The dict key a spec is stored under, identical to _exec_spec's spec_name."""
    return f"{iv_name} x {s_name}"


def _check_spec_ids():
    """Fail loudly if the local numbering ever drifts from the repo-wide SPEC_MAP.

    A silent drift would send `--spec-id 12` to a different cell than the one the demand
    parquets, the phi_nopix exports and every "spec 12" in the notes mean."""
    try:
        from sleep_demand_prep_link import SPEC_MAP
    except Exception:
        return                                   # not importable here: nothing to check against
    mine = {sid: spec_name_of(iv, s) for sid, iv, s in spec_grid("logit")}
    bad = {sid: (mine[sid], SPEC_MAP[sid]) for sid in mine
           if SPEC_MAP.get(sid) != mine[sid]}
    if bad:
        raise RuntimeError(
            f"spec-id numbering disagrees with estimation_demand_link_common.SPEC_MAP: {bad}")


def resolve_spec_id(token, kind, time_block=False):
    """Accept a 1-based spec id OR a spec name -> (spec_id, iv_name, s_name, spec_name).

    Names may be given in any case and with loose spacing ("iv_hausmanfull x tech")."""
    _check_spec_ids()
    grid = spec_grid(kind, time_block=time_block)
    by_id = {sid: (sid, iv, s) for sid, iv, s in grid}
    tok = str(token).strip()
    if tok.isdigit():
        sid = int(tok)
        if sid not in by_id:
            raise SystemExit(
                f"--spec-id {sid} is not part of the {kind} grid. Runnable ids: "
                + ", ".join(f"{i}={spec_name_of(iv, s)}" for i, iv, s in grid))
        sid, iv, s_name = by_id[sid]
        return sid, iv, s_name, spec_name_of(iv, s_name)
    key = " ".join(tok.lower().split())
    for sid, iv, s_name in grid:
        if " ".join(spec_name_of(iv, s_name).lower().split()) == key:
            return sid, iv, s_name, spec_name_of(iv, s_name)
    raise SystemExit(
        f"--spec-id {token!r} matches no spec of the {kind} grid. Runnable specs: "
        + ", ".join(f"{i}={spec_name_of(iv, s)}" for i, iv, s in grid))


def _specs_dir(est_num):
    d = _out_dir(est_num) / _SPECS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _spec_artifact(est_num, spec_id):
    return _specs_dir(est_num) / f"spec_{spec_id}.pkl"


def _atomic_pickle_dump(obj, path):
    """Write a pickle that is either complete or absent, never truncated.

    A killed --spec-id task (SLURM timeout, preemption) must not leave a half-written
    spec_{S}.pkl behind, because --merge-specs would either crash on it or, worse, load a
    partial object. The temp file lives in the SAME directory so os.replace is a rename
    within one filesystem, which is atomic on Windows and POSIX alike. Its name cannot match
    the spec_*.pkl glob, so even a temp file orphaned by a hard kill is invisible to the merge."""
    path = str(path)
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix="." + os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _exec_spec(args):
    df, iv_name, iv_cols, s_name, s_cols, kind = args[:6]
    has_cf = len(iv_cols) > 0
    spec_name = f"{iv_name} x {s_name}"
    # The single-index link comparison in the time-series report uses spec 12
    # (IV_HausmanFull x Tech); compute its national phi_t CI band in the main routine.
    is_spec12 = (iv_name == "IV_HausmanFull" and s_name == "Tech")
    # SLEEP_PHI_BAND_ALL=1: compute the national phi_t bootstrap band for EVERY spec, not just
    # spec 12. Off by default because the band costs min(boot_B,400) extra draws per spec and
    # only spec 12 is reported; on when a full CI grid is wanted.
    want_band = is_spec12 or os.environ.get("SLEEP_PHI_BAND_ALL", "0") == "1"
    df_target, res_fs = df, None
    if has_cf:
        iv_act = [c for c in iv_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        exog_act = [c for c in s_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        if not iv_act:
            return None, None, spec_name, None
        df_target, res_fs = run_pooled_first_stage(df_target, iv_act, exog_act)

    if kind == "logit":
        res = fit_nlls_link(df_target, s_cols, has_cf=has_cf, link="logit", loss="cauchy",
                            fe_time_col=FE_TIME_COL, n_starts=NLLS_N_STARTS)
        return res, None, spec_name, res_fs

    if kind == "single_index":
        # warm start only: fit_single_index reads params_native (+ the index), never the
        # logit's AMEs/SEs -> skip its wild bootstrap.
        # n_starts matters HERE: fit_single_index never re-optimises theta, so E3/E4
        # inherit exactly the direction this call returns.
        #
        # E3/E4 are a HYBRID: the LINK is already least squares (profiled sieve OLS inside
        # fit_single_index) but the DIRECTION comes from this Cauchy NLLS logit. The LS
        # variant below refits only the direction, making the column fully LS and directly
        # comparable with the linear (OLS) E1/E2.
        def _single_index(loss, band):
            lg = fit_nlls_link(df_target, s_cols, has_cf=has_cf, link="logit", loss=loss,
                               fe_time_col=FE_TIME_COL, bootstrap=False,
                               n_starts=NLLS_N_STARTS)
            if lg is None:                      # fit_single_index would AttributeError on None
                print(f"  [single-index/{loss}] logit direction failed -- skipped")
                return None
            _si = fit_single_index(df_target, s_cols, has_cf=has_cf, logit_res=lg, degree=3,
                                   fe_time_col=FE_TIME_COL, phi_band=band)
            if _si is not None and TWOSTAGE_AME and (is_spec12 or TWOSTAGE_AME_ALL):
                _attach_twostage_ame(_si, df_target, s_cols, has_cf, loss, spec_name)
            return _si

        res = None if LS_ONLY else _single_index("cauchy", want_band)
        # scipy's plain least squares is loss="linear", NOT "ls": a stray "ls" here falls
        # through least_squares' default and silently re-runs Cauchy.
        res_ls = None if DROP_LS else _single_index("linear", want_band)
        return res, res_ls, spec_name, res_fs

    raise ValueError(f"unknown kind {kind!r}")


def _exec_block(block_args):
    """Run the 4 instrument specs of ONE state block sequentially. Blocks run in
    parallel (opt 3). Every spec is fitted cold, so a block is exactly its four specs.

    Under SLEEP_RESUME a spec already on disk (and computed under the SAME link setting) is
    skipped."""
    df, s_name, s_cols, kind, iv_specs = block_args[:5]
    done = block_args[5] if len(block_args) > 5 else (lambda _s: None)
    df = df.copy()      # thread-local copy (run_pooled_first_stage adds v_hat columns in place)
    out = []
    for iv in _IV_ORDER:
        spec_name = f"{iv} x {s_name}"
        if done(spec_name) is not None:
            print(f"  [resume] {spec_name}: already on disk, skipping")
            continue
        out.append(_exec_spec((df, iv, iv_specs[iv], s_name, s_cols, kind)))
    return out


# ── Resume support ───────────────────────────────────────────────────────────────
# A full grid is 9-12 fits, so an interrupted run that merged and wrote results ONCE, after
# every spec had finished, would lose everything. Two things make a run restartable:
#   * the pickle is merged and written after EACH state block, so a stop costs at most the
#     block in flight (<= 4 specs) rather than the whole estimator;
#   * with SLEEP_RESUME=1 a spec already on disk is skipped, PROVIDED it was computed under
#     the same link setting -- reusing a spec fitted under a different SLEEP_LINK_CONSTRAINED
#     would silently mix two estimators inside one pickle, which is exactly the failure the
#     fingerprint gates elsewhere exist to prevent.
# Default is OFF: an unqualified re-run still recomputes everything, which is what a
# reproduction run should do. Set SLEEP_RESUME=1 to opt in.
def _resume_on():
    return os.environ.get("SLEEP_RESUME", "0") == "1"


def _load_existing(est_num):
    pkl_path = _out_dir(est_num) / "estimation_results.pkl"
    if not pkl_path.exists():
        return {}
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"existing {pkl_path} unreadable ({e}); refusing to resume "
                           f"against it -- move it aside or run without SLEEP_RESUME") from e


def _entry_reusable(entry, kind, spec_name):
    """True if a stored {second_stage, ...} entry can be reused instead of recomputed.

    Shared by the in-process resume (_spec_done) and the per-spec artifact resume
    (run_one_spec), so both judge a stored fit by exactly the same rules."""
    from utils.sleep_links import link_constrained
    want_constrained = link_constrained()
    if not entry or entry.get("second_stage") is None:
        return False
    if not DROP_LS and entry.get("second_stage_ls") is None:
        return False                         # an LS column was requested but is missing
    if kind == "single_index":
        for k in ("second_stage", "second_stage_ls"):
            r = entry.get(k)
            if r is None:
                continue
            if bool(getattr(r, "si_constrained", False)) != want_constrained:
                print(f"  [resume] {spec_name}: stored under a different link setting "
                      f"(si_constrained={getattr(r, 'si_constrained', False)}, "
                      f"want {want_constrained}) -- recomputing")
                return False
    return True


def _spec_done(existing, kind):
    """-> callable(spec_name) returning the stored entry if it can be reused, else None."""

    def _f(spec_name):
        if not _resume_on():
            return None
        entry = existing.get(spec_name)
        return entry if _entry_reusable(entry, kind, spec_name) else None

    return _f


def _merge_into_pickle(est_num, results):
    """Merge freshly computed specs into estimation_results.pkl and write it out.
    Returns (merged dict, list of specs whose robust fit was recomputed)."""
    out = _out_dir(est_num)
    out.mkdir(parents=True, exist_ok=True)
    pkl_path = out / "estimation_results.pkl"
    results_dict = {}
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                results_dict = pickle.load(f)
        except Exception as e:
            raise RuntimeError(
                f"existing {pkl_path} unreadable ({e}); refusing to overwrite blindly") from e

    fresh_robust = []
    for res_main, res_ls, spec_name, res_fs in results:
        if res_main is not None:
            entry = results_dict.setdefault(spec_name, {})
            entry["second_stage"] = res_main
            entry["first_stage"] = res_fs
            if res_ls is not None:
                entry["second_stage_ls"] = res_ls
            fresh_robust.append(spec_name)
            print(f"Computed [{spec_name}]")
        elif res_ls is not None:
            if spec_name not in results_dict or "second_stage" not in results_dict[spec_name]:
                raise RuntimeError(
                    f"LS-only result for [{spec_name}] but no stored robust fit to attach to -- "
                    f"run the full estimator first")
            results_dict[spec_name]["second_stage_ls"] = res_ls
            print(f"Computed [{spec_name}] (LS only; robust preserved)")

    with open(pkl_path, "wb") as f:
        pickle.dump(results_dict, f)
    return results_dict, fresh_robust


# ── Split execution: one spec per process, one merge ─────────────────────────────
# WHY THIS EXISTS. _merge_into_pickle is load -> merge -> write on ONE file. Two processes
# doing that concurrently interleave as read(A) read(B) write(A) write(B) and B's write wins,
# silently dropping every spec that only A computed. That failure mode is not hypothetical
# here: a --spec12 run destroyed the other seven specs of a full-grid pickle on 2026-07-31.
# So the split does not make the shared write safe -- it removes it. A --spec-id task writes
# exactly one file that only it can name, and the merge, which is the sole writer of every
# shared artifact, runs once, alone, after the tasks are done.
def run_one_spec(est_num, kind, time_block=False, spec_token=None):
    """Estimate ONE spec and write est{N}/_specs/spec_{S}.pkl. Touches nothing else.

    Returns the artifact path. This function must never open estimation_results.pkl,
    ts_link_band_est{N}.pkl or the phi CSVs -- for reading or for writing. Everything shared
    is built later by merge_spec_artifacts()."""
    sid, iv, s_name, spec_name = resolve_spec_id(spec_token, kind, time_block=time_block)
    path = _spec_artifact(est_num, sid)
    tflag = " + Time" if time_block else ""
    print(f"\n=== ESTIMATION {est_num}: {kind}{tflag} | spec {sid} [{spec_name}] ===")

    if _resume_on() and path.exists():
        try:
            with open(path, "rb") as f:
                prev = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"existing {path} unreadable ({e}); delete it and re-run") from e
        if not isinstance(prev, dict) or "result" not in prev:
            raise RuntimeError(f"existing {path} is not a per-spec artifact; delete it and re-run")
        res_main, res_ls = prev["result"][0], prev["result"][1]
        if _entry_reusable({"second_stage": res_main, "second_stage_ls": res_ls},
                           kind, spec_name):
            print(f"  [resume] spec {sid}: already on disk -> {path}")
            return path

    df = build_pooled_data(time_block=time_block)
    _, iv_specs, state_blocks = define_specifications(time_block=time_block)
    # .copy() matches _exec_block: run_pooled_first_stage adds v_hat columns in place.
    result = _exec_spec((df.copy(), iv, iv_specs[iv], s_name, state_blocks[s_name], kind))
    res_main, res_ls, spec_name_out, res_fs = result
    if res_main is None and res_ls is None:
        print(f"  [!] spec {sid} [{spec_name}] produced no fit; recording the empty result")

    from utils.sleep_links import link_constrained
    payload = {
        "schema": 1,
        "est_num": est_num,
        "kind": kind,
        "time_block": bool(time_block),
        "spec_id": sid,
        "spec_name": spec_name_out,
        "si_constrained": link_constrained(),
        "drop_ls": DROP_LS,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # exactly the 4-tuple _merge_into_pickle consumes, so the merge applies the SAME
        # semantics to a per-spec artifact as the serial runner applies in-process.
        "result": result,
    }
    _atomic_pickle_dump(payload, path)
    print(f"Saved spec {sid} [{spec_name_out}] -> {path}")
    return path


def _load_spec_artifacts(est_num, kind, time_block):
    """-> ({spec_id: payload}, [(spec_id, spec_name) expected]). Fails loudly on a bad file."""
    sdir = _specs_dir(est_num)
    expected = [(sid, spec_name_of(iv, s)) for sid, iv, s in spec_grid(kind, time_block=time_block)]
    exp_names = dict(expected)
    found = {}
    for p in sorted(sdir.glob("spec_*.pkl")):
        stem = p.stem[len("spec_"):]
        if not stem.isdigit():
            raise RuntimeError(f"unparseable per-spec artifact name: {p}")
        sid = int(stem)
        if sid not in exp_names:
            raise RuntimeError(
                f"{p} is spec {sid}, which is not part of the {kind} grid "
                f"({sorted(exp_names)}) -- move it aside before merging")
        try:
            with open(p, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            raise RuntimeError(
                f"per-spec artifact {p} is unreadable ({e}) -- delete it and re-run that spec; "
                f"merging around it would silently drop the spec") from e
        if not isinstance(payload, dict) or "result" not in payload:
            raise RuntimeError(f"{p} is not a per-spec artifact written by --spec-id")
        for key, want in (("est_num", est_num), ("kind", kind), ("time_block", bool(time_block))):
            if payload.get(key) != want:
                raise RuntimeError(
                    f"{p} was written for {key}={payload.get(key)!r} but the merge is for "
                    f"{key}={want!r} -- the _specs dir holds artifacts of another estimator")
        if payload.get("spec_id") != sid or payload.get("spec_name") != exp_names.get(sid):
            raise RuntimeError(
                f"{p} carries spec_id={payload.get('spec_id')!r} / "
                f"spec_name={payload.get('spec_name')!r}, which does not match the grid entry "
                f"{sid}={exp_names.get(sid)!r}")
        found[sid] = payload
    return found, expected


def merge_spec_artifacts(est_num, kind, time_block=False, allow_partial=False,
                         write_phi_csv=True):
    """Fold every est{N}/_specs/spec_*.pkl into the shared pickle; rebuild the derived outputs.

    THE ONLY writer of estimation_results.pkl in the split workflow, and the only writer of
    the spec-12 band and the phi CSVs -- so no per-spec task ever contends for them. It reuses
    _merge_into_pickle verbatim, which means a pre-existing pickle's other specs survive
    exactly as they do in a serial run.

    A missing spec is an ERROR, not a shortfall to be papered over: the whole point of the
    split is that a task can die silently, and a merge that quietly wrote 7 of 8 specs would
    reproduce the 07-31 loss with extra steps. --allow-partial is the explicit opt-out."""
    _check_spec_ids()
    found, expected = _load_spec_artifacts(est_num, kind, time_block)
    missing = [(sid, name) for sid, name in expected if sid not in found]
    if missing:
        try:
            existing = _load_existing(est_num)
        except Exception as e:
            existing, _why = {}, f"could not read the stored pickle ({e})"
        else:
            _why = None
        lines = []
        for sid, name in missing:
            if _why:
                covered = _why
            else:
                covered = "already in estimation_results.pkl" if existing.get(name) else "NOT in the pickle either"
            lines.append(f"    spec {sid:>2} [{name}] -- {covered}")
        msg = (f"[!] {len(missing)} of {len(expected)} per-spec artifacts are MISSING from "
               f"{_specs_dir(est_num)}:\n" + "\n".join(lines))
        if not allow_partial:
            raise RuntimeError(
                msg + "\n    Re-run those specs with --spec-id, or pass --allow-partial to merge "
                      "what is there anyway.")
        print(msg + "\n    --allow-partial given: merging the rest anyway.", flush=True)

    results = [found[sid]["result"] for sid in sorted(found)]
    print(f"[merge] folding {len(results)} per-spec artifact(s) into "
          f"est{est_num}/estimation_results.pkl", flush=True)
    results_dict, fresh_robust = _merge_into_pickle(est_num, results)
    print(f"[merge] pickle now holds {len(results_dict)} spec(s); "
          f"{len(fresh_robust)} robust fit(s) written this pass")

    out = _out_dir(est_num)
    if not write_phi_csv:
        print(f"[merge] --skip-phi-csv: band and phi CSVs left untouched -> {out}")
        return results_dict
    # Same rule as the serial runner: nothing new merged and the CSVs already there means the
    # stored CSVs and band already describe this pickle, so leave their content AND mtimes
    # alone rather than spending minutes rewriting them identically.
    if not fresh_robust and (out / "market_panel_phis.csv").exists():
        print(f"[merge] nothing recomputed; phi CSVs/band untouched -> {out}")
        return results_dict
    df = build_pooled_data(time_block=time_block)
    _write_derived_outputs(est_num, kind, results_dict, df)
    return results_dict


def _write_derived_outputs(est_num, kind, results_dict, df):
    """Spec-12 phi_t band + the phi CSVs, from the MERGED dict. One writer, always."""
    out = _out_dir(est_num)
    # Integrate the time-series report's link-comparison CI band into the main routine:
    # the spec-12 single-index fit carries a national phi_t bootstrap band
    # (phi_t_boot); persist it where estimation_timeseries_test._link_band reads it.
    # It is written HERE, from the merged dict, and never by a per-spec task: the band is one
    # file describing one cell, so 12 concurrent tasks must not each have a claim on it. The
    # spec-12 task still computes phi_t_boot (want_band keys off the spec name) and carries it
    # inside its own artifact; this step just unpacks it.
    if kind == "single_index":
        sp12 = results_dict.get(R.SPEC12, {}).get("second_stage")
        boot = getattr(sp12, "phi_t_boot", None) if sp12 is not None else None
        if boot is not None:
            # rout_dir() honours SLEEP_OUT_ROOT like _out_dir above, so the band lands with
            # the fit it came from rather than overwriting the production one.
            rout = _paths_mod.rout_dir()
            with open(rout / f"ts_link_band_est{est_num}.pkl", "wb") as f:
                pickle.dump(boot, f)
            print(f"Saved spec-12 phi_t CI band -> ts_link_band_est{est_num}.pkl")

    # NOTE: _calculate_phis runs over the MERGED dict, so even a partial robust re-run
    # rebuilds the phi columns of every stored spec -- the CSVs stay complete.
    df["year_quarter"] = df["time_id"]
    df, national_phis = _calculate_phis(df, results_dict, LINK_OF[kind])
    df.to_csv(out / "market_panel_phis.csv", index=False)
    if national_phis:
        agg = reduce(lambda l, r: pd.merge(l, r, on="year_quarter", how="outer"), national_phis.values())
        agg.to_csv(out / "national_phi_t.csv", index=False)
    print(f"Saved results and Phis -> {out}")


def _calculate_phis(df, results_dict, link):
    df["market_size"] = df["pop_total"].fillna(0) if "pop_total" in df.columns else 1.0
    phi_results = {}
    for model_key, item in results_dict.items():
        ss = item["second_stage"]
        if ss is None:
            continue
        safe_key = model_key.replace(" ", "_").replace(".", "")
        # The result's own link tag decides the branch: a shape-constrained single-index fit
        # stores link="index_sieve" (I-spline grid over the full native index) while the
        # kind-level key says "index" (cubic in si_b), and the cubic branch would read the
        # I-spline betas as monomial coefficients. `link` is the fallback for results
        # that carry no tag.
        df[f"phi_mt_{safe_key}"] = phi_from_native(df, ss, getattr(ss, "link", None) or link)
        # WEIGHT = MARKET POPULATION, i.e. the cell MEAN of market_size, matching
        # phi_t = sum_m phi_mt*M_mt / sum_m M_mt over MARKETS m (V_Main.tex:311). market_size
        # (= pop_total) is CONSTANT within a (quarter, market) cell, so aggregating it with
        # "sum" instead would weight each market by pop x n_banks -- a product with no
        # counterpart in the model, and worth 1.90pp of LEVEL (0.9720 vs 0.9531 on est4
        # spec 12) at nearly unchanged shape. The same convention carries through the
        # bootstrap bands (_phi_t_group_struct) and demand prep's own phi_t
        # (estimation_demand_link_common), so every reported series is on one weighting.
        # MARKET = MCA, not municipality: V_Main.tex:182 defines the local market as the
        # Minimal Comparable Area and :406 sums over that market set; E1/E2 group by
        # mca_code too, so the whole lineup shares one market key (worth ~0.21 pp).
        _mkey = "mca_code" if "mca_code" in df.columns else "CODMUN_IBGE"
        agg = df.groupby(["year_quarter", _mkey], observed=True).agg(
            phi_mt=(f"phi_mt_{safe_key}", "mean"), M_mt=("market_size", "mean")).reset_index()
        num = (agg["phi_mt"] * agg["M_mt"]).groupby(agg["year_quarter"]).sum()
        den = agg["M_mt"].groupby(agg["year_quarter"]).sum().replace(0, np.nan)
        phi_results[safe_key] = (num / den).fillna(0).reset_index(name=f"phi_t_{safe_key}")
    return df, phi_results


def run_sleep_estimator(est_num, kind, time_block=False, spec12_only=False, n_jobs=4):
    """Estimate sleepiness routine E{est_num} of the given kind/time_block, over the
    full 12-spec grid (or spec 12 only). Saves estimation_results.pkl + market_panel_phis.csv
    + national_phi_t.csv to ESTIMATION_OUTPUT/DEMAND_PREP/est{est_num}."""
    tflag = " + Time" if time_block else ""
    print(f"\n=== ESTIMATION {est_num}: {kind}{tflag} "
          f"({'spec 12 only' if spec12_only else 'full 12-spec grid'}) ===")
    df = build_pooled_data(time_block=time_block)
    _, iv_specs, state_blocks = define_specifications(time_block=time_block)

    existing = _load_existing(est_num) if _resume_on() else {}
    _done = _spec_done(existing, kind)
    if _resume_on():
        print(f"[resume] SLEEP_RESUME=1 | {len(existing)} spec(s) already on disk")

    if spec12_only:
        _sp12 = R.SPEC12
        if _done(_sp12) is not None:
            print(f"  [resume] {_sp12}: already on disk, skipping")
            results = []
        else:
            results = [_exec_spec(
                (df, "IV_HausmanFull", iv_specs["IV_HausmanFull"], "Tech",
                 state_blocks["Tech"], kind))]
    else:
        # Opt 3: parallelise over the 3 state blocks; within each block the 4 instrument
        # specs run sequentially.
        # The single-index strategies drop Base: a constant-only index has no direction.
        blocks = [s for s in state_blocks if not (s == "Base" and kind != "logit")]
        block_tasks = [(df, s, state_blocks[s], kind, iv_specs, _done) for s in blocks]
        # Default SEQUENTIAL: concurrent statsmodels/scipy/numpy calls across threads
        # segfault (0xC0000005) on this stack, and the loky/process backend pickles the
        # 400k-row df (WinError 1450). Sequential is the safe default. SLEEP_BLOCK_JOBS>1
        # opts into the (risky) threading backend.
        nblk = int(os.environ.get("SLEEP_BLOCK_JOBS", "1"))
        if nblk <= 1:
            # CHECKPOINT PER BLOCK: merge and write after each one, so an interrupted run keeps
            # every completed block instead of discarding the estimator's whole grid.
            block_results = []
            for bt in block_tasks:
                br = _exec_block(bt)
                block_results.append(br)
                if br:
                    _merge_into_pickle(est_num, br)
                    print(f"[checkpoint] {bt[1]} block saved "
                          f"({len(br)} spec(s)) -> est{est_num}/estimation_results.pkl",
                          flush=True)
        else:
            from joblib import Parallel, delayed
            block_results = Parallel(n_jobs=min(nblk, len(block_tasks)), backend="threading")(
                delayed(_exec_block)(bt) for bt in block_tasks)
        results = [r for block in block_results for r in block]

    # MERGE-ON-SAVE (2026-07-31). Rebuilding results_dict from scratch and pickle.dump'ing it
    # meant a `--spec12` run silently DESTROYED the other 7 specs of a full-grid pickle (this
    # happened on 07-31; restored from _PRE_LSTEST_ backup). _merge_into_pickle loads
    # the existing pickle and updates only the specs computed in THIS run:
    #   * res_main (robust) present  -> overwrite second_stage/first_stage as before.
    #   * res_main None, res_ls set  -> LS-only run: attach second_stage_ls to the EXISTING
    #     entry, leaving the stored robust results byte-untouched. The entry must already
    #     exist (an LS variant without its robust counterpart is meaningless) -- fail loudly.
    # It also runs after every block above, so this final call is normally a no-op that just
    # returns the merged dict.
    out = _out_dir(est_num)
    results_dict, fresh_robust = _merge_into_pickle(est_num, results)

    # phi CSVs + the spec-12 CI band derive ONLY from second_stage (robust). If this run
    # recomputed no robust fit (LS-only), the stored CSVs/band are already correct for the
    # merged pickle -- skip the rebuild so their content AND mtimes stay untouched (which the
    # verification step uses as evidence that an LS run disturbed nothing downstream).
    # A RESUMED run is the exception: it may legitimately compute nothing (everything already
    # on disk) while the CSVs were never written, because the interruption landed between the
    # last block and the CSV step. Rebuild when they are missing.
    _csvs_missing = not (out / "market_panel_phis.csv").exists()
    if not fresh_robust and not (_resume_on() and _csvs_missing):
        print(f"Saved results (nothing recomputed; phi CSVs/band untouched) -> {out}")
        return
    if not fresh_robust and _csvs_missing:
        print("[resume] every spec was already on disk but the phi CSVs are missing "
              "-- rebuilding them from the stored fits")

    _write_derived_outputs(est_num, kind, results_dict, df)


# ── Config-driven CLI for the lineup E3/E4 ───────────────────────────────────────
# (E1/E2 are separate estimators with their own scripts.) Run one estimator with:
#   python sleep_est_single.py --est N
#
# SPLIT ACROSS A SLURM ARRAY. The grid is a serial loop in one process (E3 2901 s, E4 4724 s
# measured locally), but the specs are independent, so on a 128-core node the grid costs one
# spec's wall-clock instead of eight:
#
#   sbatch --array=5-12 ... --wrap 'python sleep_est_single.py --est 3 \
#                                     --spec-id $SLURM_ARRAY_TASK_ID'
#   python sleep_est_single.py --est 3 --merge-specs        # afterok the array
#
# The array indices ARE the spec ids (5-12 for the single-index kinds, which skip the Base
# block) -- see spec_grid/`--list-specs`. Each task writes only est{N}/_specs/spec_{S}.pkl;
# the merge is the only step that opens estimation_results.pkl, the band or the phi CSVs.
#
# The lineup is the SINGLE-INDEX pair under the shape-constrained link: the direction comes
# from the Cauchy NLLS logit and only the monotone link is profiled, so the reported band
# conditions on one estimated object rather than two, and sleep_band_uncond.py can widen
# it to the joint (direction + link) band from the same stored fit.
EST_CONFIG = {
    3: ("single_index", False), 4: ("single_index", True),
}

if __name__ == "__main__":
    import argparse
    pd.options.mode.chained_assignment = None
    p = argparse.ArgumentParser(description="Pooled sleepiness estimators E3/E4 (config-driven).")
    p.add_argument("--est", type=int, required=True, choices=sorted(EST_CONFIG),
                   help="Estimator id 3 or 4 (single-index, without / with the Time block)")
    p.add_argument("--spec12", action="store_true", help="Only run spec 12 (Tech[+Time] x IV_HausmanFull)")
    p.add_argument("--spec-id", metavar="S",
                   help="Run ONE spec into est{N}/_specs/spec_{S}.pkl and touch nothing else. "
                        "S is a 1-based grid id (SPEC_MAP numbering: 1-4 Base, 5-8 Macro, "
                        "9-12 Tech) or a spec name such as 'IV_HausmanFull x Tech'.")
    p.add_argument("--merge-specs", action="store_true",
                   help="Fold est{N}/_specs/spec_*.pkl into estimation_results.pkl and rebuild "
                        "the band + phi CSVs. The only writer of those files.")
    p.add_argument("--allow-partial", action="store_true",
                   help="--merge-specs: merge even though some per-spec artifacts are missing "
                        "(they are reported either way; without this the merge refuses).")
    p.add_argument("--skip-phi-csv", action="store_true",
                   help="--merge-specs: stop after the pickle; leave the band and phi CSVs alone.")
    p.add_argument("--list-specs", action="store_true",
                   help="Print the spec id -> spec name grid for --est N and exit.")
    args = p.parse_args()
    _kind, _tb = EST_CONFIG[args.est]

    if args.list_specs:
        _check_spec_ids()
        for _sid, _iv, _s in spec_grid(_kind, time_block=_tb):
            print(f"{_sid:>2}  {spec_name_of(_iv, _s)}")
        raise SystemExit(0)
    if args.spec_id is not None and args.merge_specs:
        p.error("--spec-id and --merge-specs are separate steps; run the specs, then the merge")
    if args.spec_id is not None and args.spec12:
        p.error("--spec-id and --spec12 both select specs; use --spec-id 12 for the spec-12 cell")

    if args.spec_id is not None:
        run_one_spec(args.est, _kind, time_block=_tb, spec_token=args.spec_id)
        print(f"\n--- Spec {args.spec_id} of {args.est} "
              f"({_kind}{' + Time' if _tb else ''}) Completed ---")
    elif args.merge_specs:
        merge_spec_artifacts(args.est, _kind, time_block=_tb,
                             allow_partial=args.allow_partial,
                             write_phi_csv=not args.skip_phi_csv)
        print(f"\n--- Merge {args.est} ({_kind}{' + Time' if _tb else ''}) Completed ---")
    else:
        run_sleep_estimator(args.est, _kind, time_block=_tb, spec12_only=args.spec12)
        print(f"\n--- Pipeline {args.est} ({_kind}{' + Time' if _tb else ''}) Completed ---")
