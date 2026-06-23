"""
export_results.py
=================
Orchestrator: dispatches to individual export_*_sleep_results.py scripts in parallel.

Each individual script writes:
  - Data/summaries (txt, csv): DEMAND_PREP/rout_*/EXPORTS/
  - TeX tables:                ESTIMATION_OUTPUT/Rout/

CLI Usage Examples:
-------------------
  python export_results.py --estimation 1
  python export_results.py --estimation all

Estimation map:
  1 -> export_1_sleep_results.py  (Local B-type)
  2 -> export_2_sleep_results.py  (Pooled B+D Linear)
  3 -> export_3_sleep_results.py  (Pooled B+D Logistic, AME)
"""

import sys
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to respect the project's venv guard
try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except ImportError:
    pass

def _run_export(est):
    script_name = f"export_{est}_sleep_results.py"
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        return est, None, f"[!] Warning: {script_name} not found. Skipping."
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    return est, result.returncode, result.stdout + result.stderr

def main():
    parser = argparse.ArgumentParser(description="Export estimation results for steps 1-8.")
    parser.add_argument("--estimation", choices=['1', '2', '3', '4', '5', '6', '7', '8', 'all'], required=True,
                        help="Specify the estimation step number (1 to 8) or 'all'.")
    args = parser.parse_args()

    est_list = [1, 2, 3, 4, 5, 6, 7, 8] if args.estimation == 'all' else [int(args.estimation)]

    print(f"[Export] Launching {len(est_list)} export script(s) in parallel...")

    failed = []
    with ThreadPoolExecutor(max_workers=len(est_list)) as pool:
        futures = {pool.submit(_run_export, est): est for est in est_list}
        for future in as_completed(futures):
            est, returncode, output = future.result()
            print(f"\n{'='*50}\nExport {est} output:\n{'='*50}\n{output}")
            if returncode is not None and returncode != 0:
                print(f"[!] Error: export_{est}_sleep_results.py failed (exit {returncode})")
                failed.append(est)

    if failed:
        print(f"\n[FAILED] Exports {failed} had errors.")
        sys.exit(1)

    print("\nExtraction and export complete!")

if __name__ == "__main__":
    main()


