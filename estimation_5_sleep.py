"""
estimation_5_sleep.py
==============================
Strategy 5: Pooled B+D Probit (normal sleepiness shock).  phi_mt = Phi(S' theta),
the normal-CDF analogue of estimation_3_sleep.py's logistic link.  Identical
variable set, control function and clustering; reports Average Marginal Effects.

phi (market_panel_phis.csv and downstream demand prep) is built from the NATIVE
index coefficients via utils.sleep_links.phi_from_native; AMEs are reporting-only.

Outputs -> ESTIMATION_OUTPUT/DEMAND_PREP/est5

CLI Options:
  --spec12   Only run spec 12 (IV_HausmanFull x Tech)
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
from utils.sleep_links import fit_nlls_link, phi_from_native
from utils import paths as _paths_mod

LINK = "probit"
LOSS = "cauchy"   # match estimation_3_sleep's robust NLLS loss
OUTPUT_DIR = _paths_mod.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "est5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
            return None, spec_name, None
        df_target, res_fs = run_pooled_first_stage(df_target, iv_act, exog_act)
    res_ss = fit_nlls_link(df_target, s_cols, has_cf=has_cf, link=LINK, loss=LOSS)
    return res_ss, spec_name, res_fs


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
    print("\n=== ESTIMATION 5: POOLED B+D PROBIT (AME) ===")
    df = build_pooled_data()
    _, iv_specs, state_blocks = define_specifications()

    if spec12_only:
        tasks = [(df, "IV_HausmanFull", iv_specs["IV_HausmanFull"], "Tech", state_blocks["Tech"])]
    else:
        tasks = [(df, iv, iv_specs[iv], s, state_blocks[s])
                 for s in state_blocks.keys()
                 for iv in ["OLS", "IV_CostShifters", "IV_Wholesale", "IV_HausmanFull"]]

    from joblib import Parallel, delayed
    _nw = max(1, (os.cpu_count() or 4) // max(1, int(os.environ.get("SLEEP_PIPELINE_NSLOTS", "1"))))
    results = Parallel(n_jobs=min(4, _nw))(delayed(exec_spec)(t) for t in tasks)

    results_dict = {}
    for res_ss, spec_name, res_fs in results:
        if res_ss is not None:
            results_dict[spec_name] = {"second_stage": res_ss, "first_stage": res_fs}
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
    parser = argparse.ArgumentParser(description="Estimation 5: Pooled B+D Probit (AME)")
    parser.add_argument("--spec12", action="store_true", help="Only run spec 12 (Tech x IV_HausmanFull)")
    args = parser.parse_args()
    pd.options.mode.chained_assignment = None
    run_phase(spec12_only=args.spec12)
    print("\n--- Pipeline 5 (Probit) Completed ---")
