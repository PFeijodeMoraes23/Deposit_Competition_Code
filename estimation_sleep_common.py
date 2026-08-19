"""
estimation_sleep_common.py
================================================================================
Shared runner for the pooled sleepiness estimators of the lineup. Each estimator is
one choice of (kind, time_block), listed in EST_CONFIG at the bottom:

    E3  single-index          (kind="single_index", time_block=False)
    E4  single-index + Time   (kind="single_index", time_block=True)

E1 (local linear, B-type) and E2 (pooled linear) are separate estimators with their
own scripts, estimation_1_sleep.py / estimation_2_sleep.py.

The "+Time" variant adds the time block (time_trend, gdp_growth_yoy) to every
state block via estimation_2_sleep.define_specifications(time_block=True). phi is
always built from the native index + link (phi_from_native); AMEs are reporting-only.
Inference: score/multiplier wild cluster bootstrap (utils.sleep_links).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pickle
from functools import reduce

try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except Exception:
    pass

import numpy as np
import pandas as pd

from estimation_2_sleep import build_pooled_data, define_specifications, run_pooled_first_stage
from utils.sleep_links import (fit_nlls_link, fit_single_index, fit_joint_single_index,
                               phi_from_native)
from utils import paths as _paths_mod

# kind -> phi_from_native link key. Fallback only: _calculate_phis prefers each
# result's own .link tag, which distinguishes the constrained single-index
# ("index_sieve") from the cubic ("index") within the same kind.
LINK_OF = {"logit": "logit", "single_index": "index",
           "joint_sieve": "sieve", "joint_kernel": "kernel"}

_IV_ORDER = ["OLS", "IV_CostShifters", "IV_Wholesale", "IV_HausmanFull"]

# Two-way fixed effects: the sleepiness model now carries BOTH an entity FE
# (bank x deposit-type x market) and a quarter time FE (time_id). The time FE
# absorbs aggregate time shocks additively, outside the link. Set to None to
# fall back to the entity-only within estimator.
FE_TIME_COL = "time_id"

# Opt 1/4: route the joint-sieve theta-search through the Julia engine on a
# subsample of entities (the link/phi/inference stay exact on full N in Python).
# All env-overridable; USE_JULIA_SIEVE=0 falls back to the pure-Python search.
USE_JULIA_SIEVE = os.environ.get("USE_JULIA_SIEVE", "1") != "0"
# FIX 1 (2026-07-29): was 0.2. The Julia theta search ran on a 20% subsample and its answer
# was then FROZEN (theta_fixed) with no full-sample refinement, which drove E7 to a corner
# solution loading 0.988 of a unit-norm theta on risk_free_qoq_lag -- a national series with
# ~36 distinct values that a subsample makes look maximally explanatory. The saturated link
# that produced collapsed every E7 AME by ~700x. Search on the full sample.
SUBSAMPLE_FRAC = float(os.environ.get("SLEEP_SUBSAMPLE_FRAC", "1.0"))
# NOTE: a SLEEP_SIEVE_REFINE_MAXITER knob (a Python-side Nelder-Mead polish of the Julia
# direction) was added on 2026-07-29 and reverted the same day -- see the long comment in
# utils/sleep_links.fit_joint_single_index. It cost 3-5 HOURS per spec because each of the
# ~180 function evaluations is 3 full-sample sieve solves on 487k rows, and it selected a
# WORSE direction than simply trusting the engine. Do not reintroduce it without counting
# function evaluations first.
MAXITER_MULT = int(os.environ.get("SLEEP_MAXITER_MULT", "40"))
# The Julia engine parallelises the multistart across starts (`@threads for s in 1:nst`,
# sleep_joint_sieve.jl:294), so ADDITIONAL STARTS ARE NEARLY FREE IN WALL-CLOCK as long as
# threads >= starts. This box has 12 logical cores and we were using 2, i.e. the search was
# 3x narrower than it could be at the same elapsed time -- which mattered because the
# objective turned out to be flat and multimodal (E7 landed in a Selic corner at an R2 of
# 0.95230 vs 0.95235 for the good direction).
JULIA_THREADS = int(os.environ.get("SLEEP_JULIA_THREADS",
                                   str(max(2, min(8, (os.cpu_count() or 4) - 4)))))
# Starts for the joint sieve. Was hardcoded 2; now env-overridable and matched to threads.
SIEVE_N_STARTS = int(os.environ.get("SLEEP_SIEVE_N_STARTS", str(max(2, JULIA_THREADS))))
# Starts for the NLLS logit that supplies the single-index direction. Was a single start
# from zeros. That matters because fit_single_index never re-optimises theta, so E3/E4
# take whatever direction this fit lands on -- and on the joint-sieve full-sample candidate
# scan the logit direction scored WORST of four (164,913 vs 160,940). Sequential, so each
# extra start costs one more least_squares solve; 4 is a reasonable default.
NLLS_N_STARTS = int(os.environ.get("SLEEP_NLLS_N_STARTS", "4"))
# Opt 8: drop the LS loss during the grid (robust feeds phi). Set DROP_LS=0 to keep it.
DROP_LS = os.environ.get("SLEEP_DROP_LS", "1") != "0"
# SLEEP_LS_ONLY=1: compute ONLY the LS variant (res_robust=None); the saved robust results are
# preserved by merge-on-save. Requires SLEEP_DROP_LS=0 to have any effect. See _exec_spec.
LS_ONLY = os.environ.get("SLEEP_LS_ONLY", "0") == "1"


def _out_dir(est_num):
    # demand_prep_root() honours SLEEP_OUT_ROOT, so a whole grid can be re-estimated into a
    # sandbox without touching production results. Unset => the historical path exactly.
    d = _paths_mod.demand_prep_root() / f"est{est_num}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _init_theta(logit_res, s_cols):
    if logit_res is None:
        return None
    nat = logit_res.params_native
    idx_cols = [c for c in s_cols if c != "constant"]
    return np.array([float(nat.get(f"interaction_{sv}", 0.0)) for sv in idx_cols], float)


def _exec_spec(args):
    df, iv_name, iv_cols, s_name, s_cols, kind = args[:6]
    warm_theta = args[6] if len(args) > 6 else None   # opt 9: cross-spec warm start (native theta)
    has_cf = len(iv_cols) > 0
    spec_name = f"{iv_name} x {s_name}"
    # The single-index/joint-sieve link comparison in the time-series report uses spec 12
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
        # E3/E4 are a HYBRID: the LINK is already least squares (sieve OLS inside
        # fit_single_index) but the DIRECTION comes from this Cauchy NLLS logit. The LS
        # variant below refits only the direction, making the column fully LS and directly
        # comparable with the linear (OLS) E1/E2 and with the LS joint sieve.
        def _single_index(loss, band):
            lg = fit_nlls_link(df_target, s_cols, has_cf=has_cf, link="logit", loss=loss,
                               fe_time_col=FE_TIME_COL, bootstrap=False,
                               n_starts=NLLS_N_STARTS)
            if lg is None:                      # fit_single_index would AttributeError on None
                print(f"  [single-index/{loss}] logit direction failed -- skipped")
                return None
            return fit_single_index(df_target, s_cols, has_cf=has_cf, logit_res=lg, degree=3,
                                    fe_time_col=FE_TIME_COL, phi_band=band)

        res = None if LS_ONLY else _single_index("cauchy", want_band)
        # scipy's plain least squares is loss="linear" (NOT "ls"); the Julia engine spells the
        # same thing "ls". Mapping them wrongly silently re-runs Cauchy.
        res_ls = None if DROP_LS else _single_index("linear", want_band)
        return res, res_ls, spec_name, res_fs

    if kind in ("joint_sieve", "joint_kernel"):
        link = "sieve" if kind == "joint_sieve" else "kernel"
        fe_tc = FE_TIME_COL if link == "sieve" else None   # kernel two-way FE not yet wired
        # warm start only: _init_theta reads params_native -> skip its wild bootstrap.
        logit_res = fit_nlls_link(df_target, s_cols, has_cf=has_cf, link="logit", loss="cauchy",
                                  fe_time_col=fe_tc, bootstrap=False, n_starts=NLLS_N_STARTS)
        init = _init_theta(logit_res, s_cols)
        # kernel: multistart impractical at full N. sieve: starts run in PARALLEL threads
        # in the Julia engine, so widening the search costs wall-clock only when
        # starts > threads.
        n_starts = 1 if link == "kernel" else SIEVE_N_STARTS
        warm = init if warm_theta is None else warm_theta   # opt 9: cross-spec warm start
        # Opt 1+4: Julia subsample theta-search for the sieve (kernel stays Python).
        def _julia_search(loss_name):
            """One Julia direction search; returns theta or None (Python fallback)."""
            try:
                from sleep_joint_julia import julia_theta
                # n_starts MUST be passed: julia_theta defaults to 2 and it is the JULIA
                # search that actually picks the direction here (the Python multistart is
                # skipped whenever theta_fixed is returned). Raising only the Python-side
                # n_starts changed nothing -- the engine still reported "starts=2".
                return julia_theta(df_target, s_cols, has_cf=has_cf, loss=loss_name,
                                   fe_time_col=fe_tc, subsample_frac=SUBSAMPLE_FRAC,
                                   maxiter_mult=MAXITER_MULT, init_theta=warm, seed=0,
                                   threads=JULIA_THREADS, n_starts=SIEVE_N_STARTS)
            except Exception as e:
                print(f"  [julia_theta/{loss_name}] error, Python fallback: {e}")
                return None

        # SLEEP_LS_ONLY=1: skip the robust search + fit entirely (res_robust=None) and let
        # merge-on-save leave the stored robust results untouched. Used for retro-fitting the
        # LS variant onto an already-computed grid without re-paying the robust cost.
        theta_jl = None
        res_robust = None
        if not LS_ONLY:
            if link == "sieve" and USE_JULIA_SIEVE:
                theta_jl = _julia_search("robust")
            res_robust = fit_joint_single_index(df_target, s_cols, has_cf=has_cf, link=link,
                                                loss="robust", init_theta=warm,
                                                n_starts=n_starts, boot_B=999, boot_scheme="webb", seed=0,
                                                label=f"{spec_name}/robust", fe_time_col=fe_tc,
                                                theta_fixed=theta_jl, phi_band=want_band)
        # Opt 8: drop the LS loss during the grid (robust feeds phi).
        if DROP_LS:
            res_ls = None
        else:
            # Two LS modes, mutually exclusive:
            #   SLEEP_LS_FIXED_THETA=1  (diagnostic): hold theta at the ROBUST Julia direction,
            #       so robust and LS differ ONLY in the loss. Lower bound on the LS link span.
            #   default: LS finds its OWN theta through the Julia engine (loss="ls" -- the .jl
            #       else-branch is plain dot(r,r); ~3x cheaper per eval than robust). Without
            #       the Julia call the LS branch runs the full Python-side Nelder-Mead search,
            #       measured at ~14h/spec -- never let it fall through silently, which is why
            #       the fallback prints loudly above.
            if os.environ.get("SLEEP_LS_FIXED_THETA", "0") == "1":
                _ls_theta = theta_jl          # None under LS_ONLY: nothing to hold fixed
            elif link == "sieve" and USE_JULIA_SIEVE:
                _ls_theta = _julia_search("ls")
                if _ls_theta is None and LS_ONLY:
                    # Without a Julia direction the LS fit falls into the full Python-side
                    # Nelder-Mead search (~14h/spec, measured 07-30/31). In the retrofit mode
                    # that is never what was asked for -- fail instead of silently grinding.
                    raise RuntimeError(
                        f"[{spec_name}] Julia LS theta search failed and SLEEP_LS_ONLY=1; "
                        f"refusing the ~14h/spec Python fallback")
            else:
                _ls_theta = None
            _ls_boot = int(os.environ.get("SLEEP_LS_BOOT_B", "999"))
            # SLEEP_LS_POLISH_EVALS>0: Nelder-Mead polish of the Julia direction on the PYTHON
            # objective. The two engines minimise DIFFERENT functions (the .jl bins the ramp and
            # solves the bounded LS another way), so Julia's theta is a good start but not this
            # objective's optimum: measured on E7 spec 12, Julia 26,054 vs 25,528 for a 33h cold
            # Python search, and ~100 polish evals recover ~80% of that gap. Needs a start, so it
            # only fires when _ls_theta exists.
            _ls_polish = int(os.environ.get("SLEEP_LS_POLISH_EVALS", "0")) if _ls_theta is not None else 0
            res_ls = fit_joint_single_index(df_target, s_cols, has_cf=has_cf, link=link,
                                            loss="ls", init_theta=warm, n_starts=n_starts,
                                            boot_B=_ls_boot, boot_scheme="webb", seed=0,
                                            label=f"{spec_name}/ls", fe_time_col=fe_tc,
                                            theta_fixed=_ls_theta, polish_evals=_ls_polish, phi_band=want_band)
        return res_robust, res_ls, spec_name, res_fs

    raise ValueError(f"unknown kind {kind!r}")


def _warm_from(res, s_cols):
    """Extract the native theta from a joint/single-index result as a warm start for
    the next spec in the same state block (same index regressors)."""
    if res is None or not hasattr(res, "params_native"):
        return None
    nat = res.params_native
    idx_cols = [c for c in s_cols if c != "constant"]
    arr = np.array([float(nat.get(f"interaction_{sv}", 0.0)) for sv in idx_cols], float)
    return arr if np.isfinite(arr).all() and np.linalg.norm(arr) > 0 else None


def _exec_block(block_args):
    """Run the 4 instrument specs of ONE state block sequentially, warm-starting each
    joint/single-index fit from the previous spec's theta (opt 9). Blocks run in
    parallel (opt 3), so warm-start stays within a block where the index is shared.

    Under SLEEP_RESUME a spec already on disk (and computed under the SAME link setting) is
    skipped, and its stored fit still feeds the warm-start chain -- so resuming produces the
    same sequence of starting values as an uninterrupted run rather than a colder one."""
    df, s_name, s_cols, kind, iv_specs = block_args[:5]
    done = block_args[5] if len(block_args) > 5 else (lambda _s: None)
    df = df.copy()      # thread-local copy (run_pooled_first_stage adds v_hat columns in place)
    out, warm = [], None
    for iv in _IV_ORDER:
        spec_name = f"{iv} x {s_name}"
        stored = done(spec_name)
        if stored is not None:
            print(f"  [resume] {spec_name}: already on disk, skipping")
            w = _warm_from(stored.get("second_stage") or stored.get("second_stage_ls"), s_cols)
            if w is not None:
                warm = w
            continue
        res_main, res_ls, spec_name, res_fs = _exec_spec(
            (df, iv, iv_specs[iv], s_name, s_cols, kind, warm))
        out.append((res_main, res_ls, spec_name, res_fs))
        if kind in ("joint_sieve", "single_index"):
            # In LS-only mode res_main is None; chain the LS theta instead so an LS grid
            # warm-starts itself exactly as the robust grid does.
            w = _warm_from(res_main if res_main is not None else res_ls, s_cols)
            if w is not None:
                warm = w
    return out


# ── Resume support ───────────────────────────────────────────────────────────────
# A full grid is 9-12 fits and the joint sieve runs ~50 min per spec, so an interrupted run
# used to lose everything: results were merged and written ONCE, after every spec finished.
# Two changes make a run restartable:
#   * the pickle is merged and written after EACH state block, so a stop costs at most the
#     block in flight (<= 4 specs) rather than the whole estimator;
#   * with SLEEP_RESUME=1 a spec already on disk is skipped, PROVIDED it was computed under
#     the same link setting -- reusing a spec fitted under a different SLEEP_LINK_CONSTRAINED
#     would silently mix two estimators inside one pickle, which is exactly the failure the
#     fingerprint gates elsewhere exist to prevent.
# Default is OFF: an unqualified re-run still recomputes everything, which is what a
# reproduction run should do. run_sleep_constrained.sh opts in.
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


def _spec_done(existing, kind):
    """-> callable(spec_name) returning the stored entry if it can be reused, else None."""
    from utils.sleep_links import link_constrained
    want_constrained = link_constrained()

    def _f(spec_name):
        if not _resume_on():
            return None
        entry = existing.get(spec_name)
        if not entry or entry.get("second_stage") is None:
            return None
        if not DROP_LS and entry.get("second_stage_ls") is None:
            return None                      # an LS column was requested but is missing
        if kind in ("single_index", "joint_sieve"):
            for k in ("second_stage", "second_stage_ls"):
                r = entry.get(k)
                if r is None:
                    continue
                if bool(getattr(r, "si_constrained", False)) != want_constrained:
                    print(f"  [resume] {spec_name}: stored under a different link setting "
                          f"(si_constrained={getattr(r, 'si_constrained', False)}, "
                          f"want {want_constrained}) -- recomputing")
                    return None
        return entry

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
        _sp12 = "IV_HausmanFull x Tech"
        if _done(_sp12) is not None:
            print(f"  [resume] {_sp12}: already on disk, skipping")
            results = []
        else:
            results = [_exec_spec(
                (df, "IV_HausmanFull", iv_specs["IV_HausmanFull"], "Tech",
                 state_blocks["Tech"], kind, None))]
    else:
        # Opt 3+9: parallelise over the 3 state blocks; within each block the 4 instrument
        # specs run sequentially with a cross-spec theta warm start. With JULIA_THREADS=2
        # this packs the ~6 fast cores (3 blocks x 2 threads).
        # single-index/joint strategies drop Base: a constant-only index has no direction.
        blocks = [s for s in state_blocks if not (s == "Base" and kind != "logit")]
        block_tasks = [(df, s, state_blocks[s], kind, iv_specs, _done) for s in blocks]
        # Default SEQUENTIAL: concurrent statsmodels/scipy/numpy calls across threads
        # segfault (0xC0000005) on this stack, and the loky/process backend pickles the
        # 400k-row df (WinError 1450). Sequential is the safe default; the per-fit Julia
        # + sysimage + subsample optimizations keep it fast. SLEEP_BLOCK_JOBS>1 opts into
        # the (risky) threading backend.
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

    # Integrate the time-series report's link-comparison CI band into the main routine:
    # the spec-12 single-index/joint-sieve fit carries a national phi_t bootstrap band
    # (phi_t_boot); persist it where estimation_timeseries_test._link_band reads it.
    if kind in ("single_index", "joint_sieve"):
        sp12 = results_dict.get("IV_HausmanFull x Tech", {}).get("second_stage")
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


# ── Config-driven CLI for the lineup E3/E4 ───────────────────────────────────────
# (E1/E2 are separate estimators with their own scripts.) Run one estimator with:
#   python estimation_sleep_common.py --est N
#
# The lineup is the SINGLE-INDEX pair under the shape-constrained link. The joint sieve was
# dropped: it estimates the direction and the link together, and its reported band conditions
# on a link chosen jointly with that direction, so re-profiling the link per bootstrap draw
# widens the band 1.71x (no-Time) and 7.46x (+Time) -- the latter to roughly six times its own
# fitted phi_t range. Its direction was also barely identified (bootstrap draws nearly
# orthogonal to theta-hat) and its link solve was not reproducible across runs.
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
    args = p.parse_args()
    _kind, _tb = EST_CONFIG[args.est]
    run_sleep_estimator(args.est, _kind, time_block=_tb, spec12_only=args.spec12)
    print(f"\n--- Pipeline {args.est} ({_kind}{' + Time' if _tb else ''}) Completed ---")
