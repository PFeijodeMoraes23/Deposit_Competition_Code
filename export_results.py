"""
export_results.py
=================
Orchestrator: dispatches exports in parallel. E1/E2 (+ optional E9) have their own
export_{N}_sleep_results.py; E3-E8 route to the shared export_sleep_link_common.py --est N.

Each individual script writes:
  - Data/summaries (txt, csv): DEMAND_PREP/rout_*/EXPORTS/
  - TeX tables:                ESTIMATION_OUTPUT/Rout/

CLI Usage Examples:
-------------------
  python export_results.py --estimation 1
  python export_results.py --estimation all

Estimation map:
  1, 2  -> export_1_sleep_results.py / export_2_sleep_results.py  (E1 Local B-type, E2 Pooled Linear)
  3-8   -> export_sleep_link_common.py --est N                    (E3 Logit ... E8 Joint Sieve + Time)
  9     -> export_9_sleep_results.py                              (optional joint-kernel robustness)
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

_LINK_EXPORT = {3, 4, 5, 6, 7, 8}   # config-driven via export_sleep_link_common.py --est N

def _run_export(est):
    if est in _LINK_EXPORT:
        cmd = [sys.executable, "export_sleep_link_common.py", "--est", str(est)]
    else:                            # E1/E2 + optional E9 have their own export scripts
        script_name = f"export_{est}_sleep_results.py"
        if not (Path(__file__).parent / script_name).exists():
            return est, None, f"[!] Warning: {script_name} not found. Skipping."
        cmd = [sys.executable, script_name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return est, result.returncode, result.stdout + result.stderr

def main():
    parser = argparse.ArgumentParser(description="Export estimation results for steps 1-9.")
    parser.add_argument("--estimation", choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', 'all'], required=True,
                        help="Estimation step 1-8, '9' (optional kernel), or 'all' (=1-8).")
    args = parser.parse_args()

    # 'all' = the default lineup E1-E8; E9 (optional kernel) must be requested explicitly.
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


