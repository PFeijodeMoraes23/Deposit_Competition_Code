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
    df, iv_name, iv_cols, s_name, s_cols, kind = args
    has_cf = len(iv_cols) > 0
    spec_name = f"{iv_name} x {s_name}"
    df_target, res_fs = df, None
    if has_cf:
        iv_act = [c for c in iv_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        exog_act = [c for c in s_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        if not iv_act:
            return None, None, spec_name, None
        df_target, res_fs = run_pooled_first_stage(df_target, iv_act, exog_act)

    if kind == "logit":
        res = fit_nlls_link(df_target, s_cols, has_cf=has_cf, link="logit", loss="cauchy")
        return res, None, spec_name, res_fs

    if kind == "single_index":
        logit_res = fit_nlls_link(df_target, s_cols, has_cf=has_cf, link="logit", loss="cauchy")
        res = fit_single_index(df_target, s_cols, has_cf=has_cf, logit_res=logit_res, degree=3)
        return res, None, spec_name, res_fs

    if kind in ("joint_sieve", "joint_kernel"):
        link = "sieve" if kind == "joint_sieve" else "kernel"
        logit_res = fit_nlls_link(df_target, s_cols, has_cf=has_cf, link="logit", loss="cauchy")
        init = _init_theta(logit_res, s_cols)
        n_starts = 1 if link == "kernel" else 2   # kernel: multistart impractical at full N
        res_robust = fit_joint_single_index(df_target, s_cols, has_cf=has_cf, link=link,
                                            loss="robust", init_theta=init, n_starts=n_starts,
                                            boot_B=999, boot_scheme="webb", seed=0,
                                            label=f"{spec_name}/robust")
        res_ls = fit_joint_single_index(df_target, s_cols, has_cf=has_cf, link=link,
                                        loss="ls", init_theta=init, n_starts=n_starts,
                                        boot_B=999, boot_scheme="webb", seed=0,
                                        label=f"{spec_name}/ls")
        return res_robust, res_ls, spec_name, res_fs

    raise ValueError(f"unknown kind {kind!r}")


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
        tasks = [(df, "IV_HausmanFull", iv_specs["IV_HausmanFull"], "Tech", state_blocks["Tech"], kind)]
    else:
        tasks = [(df, iv, iv_specs[iv], s, state_blocks[s], kind)
                 for s in state_blocks.keys() for iv in _IV_ORDER]

    if len(tasks) == 1:
        results = [_exec_spec(tasks[0])]
    else:
        from joblib import Parallel, delayed
        nw = max(1, (os.cpu_count() or 4) // max(1, int(os.environ.get("SLEEP_PIPELINE_NSLOTS", "1"))))
        results = Parallel(n_jobs=min(n_jobs, nw))(delayed(_exec_spec)(t) for t in tasks)

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
