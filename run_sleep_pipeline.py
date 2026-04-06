"""
run_sleep_pipeline.py

CLI Options:
------------
usage: run_sleep_pipeline.py [-h] [--only-spec-12]

Run the full Sleepiness Estimation Pipeline.

options:
  -h, --help      show this help message and exit
  --only-spec-12  Only run specification 12 for the demand prep scripts
                  instead of all specifications.

This script sequentially runs the following steps:
  1. estimation_1_sleep.py         (Local Estimation of Sleepness)
  2. estimation_2_sleep.py         (National Level Phi for D firms + Plots)
  3. estimation_3_sleep.py         (Robustness bounds for B-firms)
  4. estimation_4_sleep.py         (Robustness bounds for pooled B and D firms)
  5. export_1_sleep_results.py     (Export National Sleepiness LaTeX Tables & PDF)
  6. export_2_sleep_results.py     (Export Firm Level Sleepiness LaTeX Tables & PDF)
  7. estimation_1_demand_1_prep.py (Demand Data Preparation & Panel Serialization)
  8. estimation_2_demand_1_prep.py (Hybrid Demand Prep: B-type Local Phi + D-type PCA Phi)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Run the full Sleepiness Estimation Pipeline.",
        epilog="""
This script sequentially runs the following steps:
  1. estimation_1_sleep.py         (Local Estimation of Sleepness)
  2. estimation_2_sleep.py         (National Level Phi for D firms + Plots)
  3. estimation_3_sleep.py         (Robustness bounds for B-firms)
  4. estimation_4_sleep.py         (Robustness bounds for pooled B and D firms)
  5. export_1_sleep_results.py     (Export National Sleepiness LaTeX Tables & PDF)
  6. export_2_sleep_results.py     (Export Firm Level Sleepiness LaTeX Tables & PDF)
  7. estimation_1_demand_1_prep.py (Demand Data Preparation & Panel Serialization)
  8. estimation_2_demand_1_prep.py (Hybrid Demand Prep: B-type Local Phi + D-type PCA Phi)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--only-spec-12",
        action="store_true",
        help="Only run specification 12 for the demand prep scripts instead of all specifications."
    )
    
    args = parser.parse_args()

    print("=====================================================================")
    print(" INITIATING SLEEPINESS ESTIMATION PIPELINE")
    print("=====================================================================")

    spec_arg = "12" if args.only_spec_12 else "all"
    spec12_arg = ["--spec12"] if args.only_spec_12 else []

    # Define the scripts and their arguments precisely as requested
    scripts_to_run = [
        {"file": "estimation_1_sleep.py", "desc": "Local Estimation of Sleepness"},
        {"file": "estimation_2_sleep.py", "args": ["--run-all", "--all-options"], "desc": "National Level Phi for D firms + Plots"},
        {"file": "estimation_3_sleep.py", "args": spec12_arg, "desc": "Robustness bounds for B-firms"},
        {"file": "estimation_4_sleep.py", "args": spec12_arg, "desc": "Robustness bounds for pooled B and D firms"},
        {"file": "export_1_sleep_results.py", "desc": "Export National Sleepiness LaTeX Tables & PDF"},
        {"file": "export_2_sleep_results.py", "desc": "Export Firm Level Sleepiness LaTeX Tables & PDF"},
        {"file": "estimation_1_demand_1_prep.py", "args": ["--spec", spec_arg], "desc": "Demand Data Preparation & Panel Serialization"},
        {"file": "estimation_2_demand_1_prep.py", "args": ["--spec", spec_arg], "desc": "Hybrid Demand Prep: B-type Local Phi + D-type PCA Phi"}
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
