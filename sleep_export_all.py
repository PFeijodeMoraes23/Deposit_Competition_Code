"""
sleep_export_all.py
=================
Orchestrator: dispatches exports in parallel. E1/E2 have their own
export_{N}_sleep_results.py; E3/E4 route to the shared sleep_export_link.py --est N.

Each individual script writes:
  - Data/summaries (txt, csv): DEMAND_PREP/rout_*/EXPORTS/
  - TeX tables:                ESTIMATION_OUTPUT/Rout/

CLI Usage Examples:
-------------------
  python sleep_export_all.py --estimation 1
  python sleep_export_all.py --estimation all

Estimation map:
  1, 2  -> sleep_export_e1.py / sleep_export_e2.py  (E1 Local B-type, E2 Pooled Linear)
  3-4   -> sleep_export_link.py --est N                    (E3 Single-Index, E4 +Time)
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

_LINK_EXPORT = set(routines.LINK_ESTS)   # config-driven via sleep_export_link.py --est N

def _run_export(est):
    if est in _LINK_EXPORT:
        cmd = [sys.executable, "sleep_export_link.py", "--est", str(est)]
    else:                            # E1/E2 have their own export scripts
        script_name = f"sleep_export_e{est}.py"
        if not (Path(__file__).parent / script_name).exists():
            # 127 (command not found), NOT a skip. A skipped routine never reaches failed[],
            # so the orchestrator prints its success line and exits 0 with no table
            # regenerated -- and because the previous .tex files are still on disk the paper
            # then compiles against stale numbers with no signal at all.
            return est, 127, f"[!] {script_name} not found in {Path(__file__).parent}."
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
            if returncode != 0:
                what = (f"sleep_export_link.py --est {est}" if est in _LINK_EXPORT
                        else f"sleep_export_e{est}.py")
                print(f"[!] Error: {what} failed (exit {returncode})")
                failed.append(est)

    if failed:
        print(f"\n[FAILED] Exports {failed} had errors.")
        sys.exit(1)

    print("\nExtraction and export complete!")

if __name__ == "__main__":
    main()


