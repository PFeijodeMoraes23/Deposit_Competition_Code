import subprocess
import sys
import time
from pathlib import Path

def main():
    print("=====================================================================")
    print(" INITIATING SLEEPINESS ESTIMATION PIPELINE")
    print("=====================================================================")

    # Define the scripts and their arguments precisely as requested
    scripts_to_run = [
        {"file": "estimation_1_sleep_1_cfa.py", "desc": "National Level First-Stage & Second-Stage (CFA)"},
        {"file": "estimation_1_sleep_2_phi.py", "desc": "National Level Implied Phi Generation"},
        {"file": "estimation_2_sleep_1_cfa.py", "args": ["--run-all"], "desc": "Firm Level First-Stage & Second-Stage (CFA)"},
        {"file": "estimation_2_sleep_2_phi.py", "args": ["--all-options"], "desc": "Firm Level Direct Phi Estimations and PCA Options"},
        {"file": "export_1_sleep_results.py", "desc": "Export National Sleepiness LaTeX Tables & PDF"},
        {"file": "export_2_sleep_results.py", "desc": "Export Firm Level Sleepiness LaTeX Tables & PDF"}
    ]

    cwd = Path(__file__).resolve().parent
    start_time_all = time.time()

    for idx, step in enumerate(scripts_to_run, 1):
        script = step["file"]
        desc = step["desc"]
        args = step.get("args", [])
        
        print("\n" + "-"*70)
        print(f"[{idx}/{len(scripts_to_run)}] Executing: {script}")
        print(f"Task: {desc}")
        print("-" * 70)

        cmd = [sys.executable, script] + args
        start_time_script = time.time()
        
        result = subprocess.run(cmd, cwd=cwd)

        elapsed_script = time.time() - start_time_script
        print(f"-> Finished {script} in {elapsed_script:.2f} seconds.")

        if result.returncode != 0:
            print(f"\n[ERROR] Pipeline aborted. Script '{script}' failed with exit code: {result.returncode}")
            sys.exit(result.returncode)

    elapsed_all = time.time() - start_time_all
    print("\n=====================================================================")
    print(f" SLEEPINESS PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_all:.2f} SECONDS")
    print("=====================================================================")

if __name__ == "__main__":
    main()
