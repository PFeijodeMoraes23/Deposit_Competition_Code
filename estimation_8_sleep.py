"""
estimation_8_sleep.py
==============================
Strategy 8: Pooled B+D JOINT single-index (Ichimura 1993 SLS) with a KERNEL
local-linear link. Like Est 7, the index direction theta (||theta||=1) and the
link G are estimated TOGETHER by semiparametric least squares; the difference is
the link smoother. The link is profiled by a binned weighted (proportional to
D-tilde^2) local-linear regression with BACKFITTING for the entity FE +
multiplicative-Z structure (Fan 1992; Fan-Marron 1994). Monotonicity is imposed
EX POST by rearrangement (Chernozhukov-Fernandez-Val-Galichon 2009), then clipped
to [0,1].

Supports the full 12-spec grid like Est 1-7 (omit --spec12), but the kernel
profiling is expensive (~1h+ at n_starts=1), so the pipeline default and the
cheap path is spec 12 only (--spec12, IV_HausmanFull x Tech, the headline spec).
Two losses are stored (LS and robust); the official phi uses the ROBUST fit via
phi_from_native. The index search is warm-started from the logit direction.
Inference: score/multiplier wild cluster bootstrap.

Outputs -> ESTIMATION_OUTPUT/DEMAND_PREP/est8

CLI Options:
  --spec12   Only run spec 12 (omit for the full 12-spec grid)
"""
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import pickle
from functools import reduce

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd

from estimation_2_sleep import build_pooled_data, define_specifications, run_pooled_first_stage
from utils.sleep_links import fit_nlls_link, fit_joint_single_index, phi_from_native
from utils import paths as _paths_mod

LINK = "kernel"
OUTPUT_DIR = _paths_mod.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "est8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _init_theta(logit_res, s_cols):
    if logit_res is None:
        return None
    nat = logit_res.params_native
    idx_cols = [c for c in s_cols if c != "constant"]
    return np.array([float(nat.get(f"interaction_{sv}", 0.0)) for sv in idx_cols], float)


def exec_spec(args):
    df, iv_name, iv_cols, s_name, s_cols = args
    has_cf = len(iv_cols) > 0
    spec_name = f"{iv_name} x {s_name}"
    df_target = df
    res_fs = None
    if has_cf:
        iv_act = [c for c in iv_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        exog_act = [c for c in s_cols if c in df_target.columns and df_target[c].notnull().sum() > 0]
        if not iv_act:
            return None, None, spec_name, None
        df_target, res_fs = run_pooled_first_stage(df_target, iv_act, exog_act)

    logit_res = fit_nlls_link(df_target, s_cols, has_cf=has_cf, link="logit", loss="cauchy")
    init = _init_theta(logit_res, s_cols)

    res_robust = fit_joint_single_index(df_target, s_cols, has_cf=has_cf, link=LINK,
                                        loss="robust", init_theta=init, n_starts=1,
                                        boot_B=999, boot_scheme="webb", seed=0, label=f"{spec_name}/robust")
    res_ls = fit_joint_single_index(df_target, s_cols, has_cf=has_cf, link=LINK,
                                    loss="ls", init_theta=init, n_starts=1,
                                    boot_B=999, boot_scheme="webb", seed=0, label=f"{spec_name}/ls")
    return res_robust, res_ls, spec_name, res_fs


def calculate_phis(df, results_dict):
    df["market_size"] = df["pop_total"].fillna(0) if "pop_total" in df.columns else 1.0
    phi_results = {}
    for model_key, item in results_dict.items():
        ss = item["second_stage"]
        if ss is None:
            continue
        safe_key = model_key.replace(" ", "_").replace(".", "")
        df[f"phi_mt_{safe_key}"] = phi_from_native(df, ss, LINK)
        agg = df.groupby(["year_quarter", "CODMUN_IBGE"], observed=True).agg(
            phi_mt=(f"phi_mt_{safe_key}", "mean"), M_mt=("market_size", "sum")).reset_index()
        num = (agg["phi_mt"] * agg["M_mt"]).groupby(agg["year_quarter"]).sum()
        den = agg["M_mt"].groupby(agg["year_quarter"]).sum().replace(0, np.nan)
        phi_results[safe_key] = (num / den).fillna(0).reset_index(name=f"phi_t_{safe_key}")
    return df, phi_results


def run_phase(spec12_only=False):
    print("\n=== ESTIMATION 8: POOLED B+D JOINT SINGLE-INDEX (kernel local-linear) ===")
    df = build_pooled_data()
    _, iv_specs, state_blocks = define_specifications()

    if spec12_only:
        tasks = [(df, "IV_HausmanFull", iv_specs["IV_HausmanFull"], "Tech", state_blocks["Tech"])]
    else:
        # Full 12-spec grid. The kernel backfit is expensive, so this is heavy
        # (~1h+ at n_starts=1); spec-12-only (--spec12) is the cheap default in
        # the pipeline.
        tasks = [(df, iv, iv_specs[iv], s, state_blocks[s])
                 for s in state_blocks.keys()
                 for iv in ["OLS", "IV_CostShifters", "IV_Wholesale", "IV_HausmanFull"]]

    from joblib import Parallel, delayed
    _nw = max(1, (os.cpu_count() or 4) // max(1, int(os.environ.get("SLEEP_PIPELINE_NSLOTS", "1"))))
    results = Parallel(n_jobs=min(4, _nw))(delayed(exec_spec)(t) for t in tasks)

    results_dict = {}
    for res_robust, res_ls, spec_name, res_fs in results:
        if res_robust is not None:
            results_dict[spec_name] = {"second_stage": res_robust,
                                       "second_stage_ls": res_ls,
                                       "first_stage": res_fs}
            print(f"Computed [{spec_name}]")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "estimation_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)

    df["year_quarter"] = df["time_id"]
    df, national_phis = calculate_phis(df, results_dict)
    df.to_csv(OUTPUT_DIR / "market_panel_phis.csv", index=False)
    if national_phis:
        agg = reduce(lambda l, r: pd.merge(l, r, on="year_quarter", how="outer"), national_phis.values())
        agg.to_csv(OUTPUT_DIR / "national_phi_t.csv", index=False)
    print(f"Saved results and Phis -> {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimation 8: Pooled B+D Joint Single-Index (kernel)")
    parser.add_argument("--spec12", action="store_true", help="Only run spec 12 (Tech x IV_HausmanFull); omit for the full 12-spec grid")
    args = parser.parse_args()
    pd.options.mode.chained_assignment = None
    run_phase(spec12_only=args.spec12)
    print("\n--- Pipeline 8 (Joint Single-Index, kernel) Completed ---")
