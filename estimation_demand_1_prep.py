
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import argparse
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def _run_script(est, spec):
    script_name = f"estimation_{est}_demand_1_prep.py"
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        return est, None, f"[!] Warning: {script_name} not found. Skipping."
    cmd = [sys.executable, script_name, "--spec", str(spec)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return est, result.returncode, result.stdout + result.stderr

def main():
    parser = argparse.ArgumentParser(description="Unified Demand 1 Prep Orchestrator")
    parser.add_argument("--estimation", choices=['1', '2', '3', '4', '5', '6', 'all'], required=True,
                        help="Estimation step number (1 to 6) or 'all' to run demand prep.")
    parser.add_argument("--spec", default="all", help="Specification ID or 'all'")
    args = parser.parse_args()

    EST_LIST = [1, 2, 3, 4, 5] if args.estimation == 'all' else [int(args.estimation)]

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

