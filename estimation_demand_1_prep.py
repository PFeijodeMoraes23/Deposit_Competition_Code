
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import argparse
import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_LINK_DEMAND = {3, 4, 5, 6}   # config-driven via estimation_demand_link_common.py --est N

def _run_script(est, spec):
    if est in _LINK_DEMAND:
        cmd = [sys.executable, "estimation_demand_link_common.py", "--est", str(est), "--spec", str(spec)]
    else:                            # E1/E2 have their own demand-prep scripts
        script_name = f"estimation_{est}_demand_1_prep.py"
        if not (Path(__file__).parent / script_name).exists():
            return est, None, f"[!] Warning: {script_name} not found. Skipping."
        cmd = [sys.executable, script_name, "--spec", str(spec)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return est, result.returncode, result.stdout + result.stderr

def main():
    parser = argparse.ArgumentParser(description="Unified Demand 1 Prep Orchestrator")
    parser.add_argument("--estimation", choices=['1', '2', '3', '4', '5', '6', 'all'], required=True,
                        help="Estimation strategy (1=Local Linear, 2=Pooled Linear, "
                             "3=Single-Index, 4=Single-Index+Time, 5=Joint SI sieve, "
                             "6=Joint SI sieve+Time) or 'all' (=1-6).")
    parser.add_argument("--spec", default="all", help="Specification ID or 'all'")
    args = parser.parse_args()

    # 'all' = the whole lineup E1-E6. SLEEP_ACTIVE_ESTS narrows it; an explicit --estimation N
    # always works, so nothing is unreachable.
    #
    # Demand prep discovers routines BY FILE PRESENCE downstream, so an artifact left behind by
    # a routine that is no longer in the lineup re-enters a run on its own: parquets for a
    # dropped routine were once rebuilt from the NEW panel while carrying a STALE phi, which
    # looks current by mtime and then feeds the estimator. Archive the artifacts of any routine
    # you remove from this list, do not just stop listing it.
    _active = os.environ.get("SLEEP_ACTIVE_ESTS", "1 2 3 4 5 6").split()
    EST_LIST = [int(x) for x in _active] if args.estimation == 'all' else [int(args.estimation)]

    # Each child parses its own ~700 MB market_panel_phis.csv, so running all 8 at once
    # exhausts RAM (pandas "C error: out of memory"). Cap concurrency; DEMAND_PREP_JOBS overrides.
    n_jobs = max(1, min(len(EST_LIST), int(os.environ.get("DEMAND_PREP_JOBS", "2"))))
    print(f"[Demand Prep] Launching {len(EST_LIST)} script(s), {n_jobs} at a time...")

    failed = []
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(_run_script, est, args.spec): est for est in EST_LIST}
        for future in as_completed(futures):
            est, returncode, output = future.result()
            print(f"\n{'='*50}\nEstimation {est} output:\n{'='*50}\n{output}")
            if returncode is not None and returncode != 0:
                print(f"[!] Error: estimation_{est}_demand_1_prep.py failed (exit {returncode})")
                failed.append(est)

    if failed:
        print(f"\n[FAILED] Estimations {failed} had errors.")
        sys.exit(1)

    print("\n[SUCCESS] Universal Demand Prep Orchestrator Finished successfully.")

if __name__ == "__main__":
    main()

