"""
export_results.py
=================
Orchestrator: dispatches to individual export_*_sleep_results.py scripts in parallel.

Each individual script writes:
  - Data/summaries (txt, csv): DEMAND_PREP/rout_*/EXPORTS/
  - TeX tables:                Drafts/Deposit Competition/Rout/

CLI Usage Examples:
-------------------
  python export_results.py --estimation 1
  python export_results.py --estimation all

Estimation map:
  1 -> export_1_sleep_results.py
  2 -> export_2_sleep_results.py (if present)
  3 -> export_3_sleep_results.py
  4 -> export_4_sleep_results.py
  5 -> export_5_sleep_results.py
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
    parser = argparse.ArgumentParser(description="Export estimation results for steps 1-5.")
    parser.add_argument("--estimation", choices=['1', '2', '3', '4', '5', 'all'], required=True,
                        help="Specify the estimation step number (1 to 5) or 'all'.")
    args = parser.parse_args()

    est_list = [1, 2, 3, 4, 5] if args.estimation == 'all' else [int(args.estimation)]

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
    
    for target_dir in target_dirs:
        pkl_path = target_dir / "estimation_results.pkl"
        
        if not pkl_path.exists():
            print(f"Warning: No estimation_results.pkl found at {pkl_path}")
            continue
            
        found_data = True
        print(f"\nLoading pickle file: {pkl_path}")
        
        with open(pkl_path, "rb") as f:
            try:
                results_dict = pickle.load(f)
                export_specification_results(results_dict, target_dir)
            except Exception as e:
                print(f"Failed to load or process pickle file: {pkl_path}\nError: {e}")
                
    if not found_data:
        print(f"\nNo `.pkl` results were found for estimation step {args.estimation}.")
        print("Ensure you have successfully run the regression pipelines first.")
        sys.exit(1)
        
    print("\nExtraction and export complete!")

if __name__ == "__main__":
    main()






