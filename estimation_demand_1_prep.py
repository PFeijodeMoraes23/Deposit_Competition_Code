
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import argparse
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_LINK_DEMAND = {3, 4, 5, 6, 7, 8}   # config-driven via estimation_demand_link_common.py --est N

def _run_script(est, spec):
    if est in _LINK_DEMAND:
        cmd = [sys.executable, "estimation_demand_link_common.py", "--est", str(est), "--spec", str(spec)]
    else:                            # E1/E2 + optional E9 have their own demand-prep scripts
        script_name = f"estimation_{est}_demand_1_prep.py"
        if not (Path(__file__).parent / script_name).exists():
            return est, None, f"[!] Warning: {script_name} not found. Skipping."
        cmd = [sys.executable, script_name, "--spec", str(spec)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return est, result.returncode, result.stdout + result.stderr

def main():
    parser = argparse.ArgumentParser(description="Unified Demand 1 Prep Orchestrator")
    parser.add_argument("--estimation", choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', 'all'], required=True,
                        help="Estimation strategy (1=Local Linear, 2=Pooled Linear, 3=Logit, "
                             "4=Logit+Time, 5=Single-Index, 6=Single-Index+Time, 7=Joint SI sieve, "
                             "8=Joint SI sieve+Time, 9=Joint SI kernel [optional]) or 'all' (=1-8).")
    parser.add_argument("--spec", default="all", help="Specification ID or 'all'")
    args = parser.parse_args()

    # 'all' = the default lineup E1-E8; E9 (optional kernel) must be requested explicitly.
    EST_LIST = [1, 2, 3, 4, 5, 6, 7, 8] if args.estimation == 'all' else [int(args.estimation)]

    print(f"[Demand Prep] Launching {len(EST_LIST)} script(s) in parallel...")

    failed = []
    with ThreadPoolExecutor(max_workers=len(EST_LIST)) as pool:
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

