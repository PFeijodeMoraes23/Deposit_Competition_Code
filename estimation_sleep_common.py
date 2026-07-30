"""
estimation_sleep_common.py
================================================================================
Shared runner for the pooled sleepiness estimators E3-E9. Each estimator is one
choice of (kind, time_block):

    E3  logit                 (kind="logit",        time_block=False)
    E4  logit + Time          (kind="logit",        time_block=True)
    E5  single-index          (kind="single_index", time_block=False)
    E6  single-index + Time   (kind="single_index", time_block=True)
    E7  joint single-index, monotone sieve  (kind="joint_sieve",  time_block=False)
    E8  joint single-index, sieve + Time    (kind="joint_sieve",  time_block=True)
    E9  joint single-index, kernel (optional/deferred)  (kind="joint_kernel", spec12)

The "+Time" variants add the time block (time_trend, gdp_growth_yoy) to every
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

# kind -> phi_from_native link key
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
# Starts for the NLLS logit (E3/E4, and the warm start E5-E8 inherit). Was a single start
# from zeros. That matters beyond E3/E4: fit_single_index never re-optimises theta, so E5/E6
# take whatever direction this fit lands on -- and on the joint-sieve full-sample candidate
# scan the logit direction scored WORST of four (164,913 vs 160,940). Sequential, so each
# extra start costs one more least_squares solve; 4 is a reasonable default.
NLLS_N_STARTS = int(os.environ.get("SLEEP_NLLS_N_STARTS", "4"))
# Opt 8: drop the LS loss during the grid (robust feeds phi). Set DROP_LS=0 to keep it.
DROP_LS = os.environ.get("SLEEP_DROP_LS", "1") != "0"


def _out_dir(est_num):
    d = _paths_mod.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / f"est{est_num}"
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
        # n_starts matters HERE as much as for E3/E4: fit_single_index never re-optimises
        # theta, so E5/E6 inherit exactly the direction this call returns.
        logit_res = fit_nlls_link(df_target, s_cols, has_cf=has_cf, link="logit", loss="cauchy",
                                  fe_time_col=FE_TIME_COL, bootstrap=False,
                                  n_starts=NLLS_N_STARTS)
        res = fit_single_index(df_target, s_cols, has_cf=has_cf, logit_res=logit_res, degree=3,
                               fe_time_col=FE_TIME_COL, phi_band=is_spec12)
        return res, None, spec_name, res_fs

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
        theta_jl = None
        if link == "sieve" and USE_JULIA_SIEVE:
            try:
                from sleep_joint_julia import julia_theta
                # n_starts MUST be passed: julia_theta defaults to 2 and it is the JULIA
                # search that actually picks the direction here (the Python multistart is
                # skipped whenever theta_fixed is returned). Raising only the Python-side
                # n_starts changed nothing -- the engine still reported "starts=2".
                theta_jl = julia_theta(df_target, s_cols, has_cf=has_cf, loss="robust", fe_time_col=fe_tc,
                                       subsample_frac=SUBSAMPLE_FRAC, maxiter_mult=MAXITER_MULT,
                                       init_theta=warm, seed=0, threads=JULIA_THREADS,
                                       n_starts=SIEVE_N_STARTS)
            except Exception as e:
                print(f"  [julia_theta] error, Python fallback: {e}")
                theta_jl = None
        res_robust = fit_joint_single_index(df_target, s_cols, has_cf=has_cf, link=link,
                                            loss="robust", init_theta=warm,
                                            n_starts=n_starts, boot_B=999, boot_scheme="webb", seed=0,
                                            label=f"{spec_name}/robust", fe_time_col=fe_tc,
                                            theta_fixed=theta_jl, phi_band=is_spec12)
        # Opt 8: drop the LS loss during the grid (robust feeds phi).
        if DROP_LS:
            res_ls = None
        else:
            res_ls = fit_joint_single_index(df_target, s_cols, has_cf=has_cf, link=link,
                                            loss="ls", init_theta=warm, n_starts=n_starts,
                                            boot_B=999, boot_scheme="webb", seed=0,
                                            label=f"{spec_name}/ls", fe_time_col=fe_tc)
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
    parallel (opt 3), so warm-start stays within a block where the index is shared."""
    df, s_name, s_cols, kind, iv_specs = block_args
    df = df.copy()      # thread-local copy (run_pooled_first_stage adds v_hat columns in place)
    out, warm = [], None
    for iv in _IV_ORDER:
        res_main, res_ls, spec_name, res_fs = _exec_spec(
            (df, iv, iv_specs[iv], s_name, s_cols, kind, warm))
        out.append((res_main, res_ls, spec_name, res_fs))
        if kind in ("joint_sieve", "single_index"):
            w = _warm_from(res_main, s_cols)
            if w is not None:
                warm = w
    return out


def _calculate_phis(df, results_dict, link):
    df["market_size"] = df["pop_total"].fillna(0) if "pop_total" in df.columns else 1.0
    phi_results = {}
    for model_key, item in results_dict.items():
        ss = item["second_stage"]
        if ss is None:
            continue
        safe_key = model_key.replace(" ", "_").replace(".", "")
        df[f"phi_mt_{safe_key}"] = phi_from_native(df, ss, link)
        agg = df.groupby(["year_quarter", "CODMUN_IBGE"], observed=True).agg(
            phi_mt=(f"phi_mt_{safe_key}", "mean"), M_mt=("market_size", "sum")).reset_index()
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

    if spec12_only:
        results = [_exec_spec(
            (df, "IV_HausmanFull", iv_specs["IV_HausmanFull"], "Tech", state_blocks["Tech"], kind, None))]
    else:
        # Opt 3+9: parallelise over the 3 state blocks; within each block the 4 instrument
        # specs run sequentially with a cross-spec theta warm start. With JULIA_THREADS=2
        # this packs the ~6 fast cores (3 blocks x 2 threads).
        # single-index/joint strategies drop Base: a constant-only index has no direction.
        blocks = [s for s in state_blocks if not (s == "Base" and kind != "logit")]
        block_tasks = [(df, s, state_blocks[s], kind, iv_specs) for s in blocks]
        # Default SEQUENTIAL: concurrent statsmodels/scipy/numpy calls across threads
        # segfault (0xC0000005) on this stack, and the loky/process backend pickles the
        # 400k-row df (WinError 1450). Sequential is the safe default; the per-fit Julia
        # + sysimage + subsample optimizations keep it fast. SLEEP_BLOCK_JOBS>1 opts into
        # the (risky) threading backend.
        nblk = int(os.environ.get("SLEEP_BLOCK_JOBS", "1"))
        if nblk <= 1:
            block_results = [_exec_block(bt) for bt in block_tasks]
        else:
            from joblib import Parallel, delayed
            block_results = Parallel(n_jobs=min(nblk, len(block_tasks)), backend="threading")(
                delayed(_exec_block)(bt) for bt in block_tasks)
        results = [r for block in block_results for r in block]

    results_dict = {}
    for res_main, res_ls, spec_name, res_fs in results:
        if res_main is not None:
            entry = {"second_stage": res_main, "first_stage": res_fs}
            if res_ls is not None:
                entry["second_stage_ls"] = res_ls
            results_dict[spec_name] = entry
            print(f"Computed [{spec_name}]")

    out = _out_dir(est_num)
    with open(out / "estimation_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)

    # Integrate the time-series report's link-comparison CI band into the main routine:
    # the spec-12 single-index/joint-sieve fit carries a national phi_t bootstrap band
    # (phi_t_boot); persist it where estimation_timeseries_test._link_band reads it.
    if kind in ("single_index", "joint_sieve"):
        sp12 = results_dict.get("IV_HausmanFull x Tech", {}).get("second_stage")
        boot = getattr(sp12, "phi_t_boot", None) if sp12 is not None else None
        if boot is not None:
            rout = _paths_mod.PROCESSED / "ESTIMATION_OUTPUT" / "Rout"
            rout.mkdir(parents=True, exist_ok=True)
            with open(rout / f"ts_link_band_est{est_num}.pkl", "wb") as f:
                pickle.dump(boot, f)
            print(f"Saved spec-12 phi_t CI band -> ts_link_band_est{est_num}.pkl")

    df["year_quarter"] = df["time_id"]
    df, national_phis = _calculate_phis(df, results_dict, LINK_OF[kind])
    df.to_csv(out / "market_panel_phis.csv", index=False)
    if national_phis:
        agg = reduce(lambda l, r: pd.merge(l, r, on="year_quarter", how="outer"), national_phis.values())
        agg.to_csv(out / "national_phi_t.csv", index=False)
    print(f"Saved results and Phis -> {out}")


# ── Config-driven CLI for the default lineup E3-E8 ───────────────────────────────
# (E1/E2 are separate estimators; E9 joint-kernel is an optional, separate routine:
#  estimation_9_sleep.py.) Run one estimator with:  python estimation_sleep_common.py --est N
EST_CONFIG = {
    3: ("logit", False), 4: ("logit", True),
    5: ("single_index", False), 6: ("single_index", True),
    7: ("joint_sieve", False), 8: ("joint_sieve", True),
}

if __name__ == "__main__":
    import argparse
    pd.options.mode.chained_assignment = None
    p = argparse.ArgumentParser(description="Pooled sleepiness estimators E3-E8 (config-driven).")
    p.add_argument("--est", type=int, required=True, choices=sorted(EST_CONFIG),
                   help="Estimator id 3-8 (E9 kernel is a separate routine: estimation_9_sleep.py)")
    p.add_argument("--spec12", action="store_true", help="Only run spec 12 (Tech[+Time] x IV_HausmanFull)")
    args = p.parse_args()
    _kind, _tb = EST_CONFIG[args.est]
    run_sleep_estimator(args.est, _kind, time_block=_tb, spec12_only=args.spec12)
    print(f"\n--- Pipeline {args.est} ({_kind}{' + Time' if _tb else ''}) Completed ---")
