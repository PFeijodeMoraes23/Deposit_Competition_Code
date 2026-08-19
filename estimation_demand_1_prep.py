
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import paths

_LINK_DEMAND = {3, 4}   # config-driven via estimation_demand_link_common.py --est N

def _count_parquets(est, since):
    """Demand parquets routine `est` (re)wrote in this run, counted on disk.

    E1/E2 write demand_{est}_spec_{sid}.parquet and E3/E4 demand_{est}_{tag}_spec_{sid}.parquet,
    both covered by the glob. Counting the artifacts rather than trusting the child's own
    report is the point: a child that reads the wrong tree finds no fit, writes nothing and
    still exits 0, and it is these files the demand estimation consumes next.
    """
    root = paths.demand_prep_root()
    if not root.exists():
        return 0
    return sum(1 for p in root.glob(f"demand_{est}_*spec_*.parquet")
               if p.stat().st_mtime >= since)

def _run_script(est, spec):
    if est in _LINK_DEMAND:
        cmd = [sys.executable, "estimation_demand_link_common.py", "--est", str(est), "--spec", str(spec)]
    else:                            # E1/E2 have their own demand-prep scripts
        script_name = f"estimation_{est}_demand_1_prep.py"
        if not (Path(__file__).parent / script_name).exists():
            return est, None, f"[!] Warning: {script_name} not found. Skipping.", 0
        cmd = [sys.executable, script_name, "--spec", str(spec)]
    started = time.time() - 1      # 1s of slack for filesystem timestamp granularity
    result = subprocess.run(cmd, capture_output=True, text=True)
    return est, result.returncode, result.stdout + result.stderr, _count_parquets(est, started)

def main():
    parser = argparse.ArgumentParser(description="Unified Demand 1 Prep Orchestrator")
    parser.add_argument("--estimation", choices=['1', '2', '3', '4', 'all'], required=True,
                        help="Estimation strategy (1=Local Linear, 2=Pooled Linear, "
                             "3=Single-Index, 4=Single-Index+Time) or 'all' (=1-4).")
    parser.add_argument("--spec", default="all", help="Specification ID or 'all'")
    args = parser.parse_args()

    # 'all' = the whole lineup E1-E4. SLEEP_ACTIVE_ESTS narrows it; an explicit --estimation N
    # always works, so nothing is unreachable.
    #
    # Demand prep discovers routines BY FILE PRESENCE downstream, so an artifact left behind by
    # a routine that is no longer in the lineup re-enters a run on its own: parquets for a
    # dropped routine were once rebuilt from the NEW panel while carrying a STALE phi, which
    # looks current by mtime and then feeds the estimator. Archive the artifacts of any routine
    # you remove from this list, do not just stop listing it.
    _active = os.environ.get("SLEEP_ACTIVE_ESTS", "1 2 3 4").split()
    EST_LIST = [int(x) for x in _active] if args.estimation == 'all' else [int(args.estimation)]

    # Each child parses its own ~700 MB market_panel_phis.csv, so running all 8 at once
    # exhausts RAM (pandas "C error: out of memory"). Cap concurrency; DEMAND_PREP_JOBS overrides.
    n_jobs = max(1, min(len(EST_LIST), int(os.environ.get("DEMAND_PREP_JOBS", "2"))))
    print(f"[Demand Prep] Launching {len(EST_LIST)} script(s), {n_jobs} at a time...")

    failed, empty = [], []
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(_run_script, est, args.spec): est for est in EST_LIST}
        for future in as_completed(futures):
            est, returncode, output, n_parquets = future.result()
            print(f"\n{'='*50}\nEstimation {est} output:\n{'='*50}\n{output}")
            if returncode is None:       # script absent; the warning above is the report
                continue
            if returncode != 0:
                print(f"[!] Error: estimation_{est}_demand_1_prep.py failed (exit {returncode})")
                failed.append(est)
            elif n_parquets == 0:
                # A requested routine that writes nothing is a failure however the child
                # exited: the next step reads these parquets by file presence and would
                # otherwise silently run on an older vintage, or on nothing at all.
                print(f"[!] Error: estimation {est} exited 0 but wrote no demand parquet")
                empty.append(est)
            else:
                print(f"[OK] Estimation {est}: {n_parquets} demand parquet(s) written")

    if failed or empty:
        if failed:
            print(f"\n[FAILED] Estimations {failed} had errors.")
        if empty:
            print(f"\n[FAILED] Estimations {empty} produced no demand parquets in "
                  f"{paths.demand_prep_root()}. Check that the sleep estimators wrote "
                  f"estimation_results.pkl to the SAME tree (SLEEP_OUT_ROOT).")
        sys.exit(1)

    print("\n[SUCCESS] Universal Demand Prep Orchestrator Finished successfully.")

if __name__ == "__main__":
    main()

