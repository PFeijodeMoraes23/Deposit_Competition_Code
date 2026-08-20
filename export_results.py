"""
export_results.py
=================
Orchestrator: dispatches exports in parallel. E1/E2 have their own
export_{N}_sleep_results.py; E3/E4 route to the shared export_sleep_link_common.py --est N.

Each individual script writes:
  - Data/summaries (txt, csv): DEMAND_PREP/rout_*/EXPORTS/
  - TeX tables:                ESTIMATION_OUTPUT/Rout/

CLI Usage Examples:
-------------------
  python export_results.py --estimation 1
  python export_results.py --estimation all

Estimation map:
  1, 2  -> export_1_sleep_results.py / export_2_sleep_results.py  (E1 Local B-type, E2 Pooled Linear)
  3-4   -> export_sleep_link_common.py --est N                    (E3 Single-Index, E4 +Time)
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

from utils import routines

_LINK_EXPORT = set(routines.LINK_ESTS)   # config-driven via export_sleep_link_common.py --est N

def _run_export(est):
    if est in _LINK_EXPORT:
        cmd = [sys.executable, "export_sleep_link_common.py", "--est", str(est)]
    else:                            # E1/E2 have their own export scripts
        script_name = f"export_{est}_sleep_results.py"
        if not (Path(__file__).parent / script_name).exists():
            return est, None, f"[!] Warning: {script_name} not found. Skipping."
        cmd = [sys.executable, script_name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return est, result.returncode, result.stdout + result.stderr

def main():
    _span = f"{routines.ACTIVE[0]}-{routines.ACTIVE[-1]}"
    parser = argparse.ArgumentParser(description=f"Export estimation results for steps {_span}.")
    parser.add_argument("--estimation", choices=routines.id_choices(('all',)), required=True,
                        help=f"Estimation step {_span}, or 'all' (={_span}).")
    args = parser.parse_args()

    # 'all' = the ACTIVE lineup, read through the SAME registry helper as demand prep — same
    # config/routines.toml, same SLEEP_ACTIVE_ESTS override — so the two cannot drift apart.
    est_list = routines.active_from_env() if args.estimation == 'all' else [int(args.estimation)]

    print(f"[Export] Launching {len(est_list)} export script(s) in parallel...")

    failed = []
    with ThreadPoolExecutor(max_workers=len(est_list)) as pool:
        futures = {pool.submit(_run_export, est): est for est in est_list}
        for future in as_completed(futures):
            est, returncode, output = future.result()
            print(f"\n{'='*50}\nExport {est} output:\n{'='*50}\n{output}")
            if returncode is not None and returncode != 0:
                what = (f"export_sleep_link_common.py --est {est}" if est in _LINK_EXPORT
                        else f"export_{est}_sleep_results.py")
                print(f"[!] Error: {what} failed (exit {returncode})")
                failed.append(est)

    if failed:
        print(f"\n[FAILED] Exports {failed} had errors.")
        sys.exit(1)

    print("\nExtraction and export complete!")

if __name__ == "__main__":
    main()


