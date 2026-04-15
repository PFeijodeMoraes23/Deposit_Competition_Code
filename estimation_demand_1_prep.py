import argparse
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Unified Demand 1 Prep Orchestrator")
    parser.add_argument("--estimation", choices=['1', '2', '3', '4', '5', '6', 'all'], required=True,
                        help="Estimation step number (1 to 6) or 'all' to run demand prep.")
    parser.add_argument("--spec", default="all", help="Specification ID or 'all'")
    
    args = parser.parse_args()
    
    EST_LIST = [1, 2, 3, 4, 5] if args.estimation == 'all' else [int(args.estimation)]
    
    for est in EST_LIST:
        script_name = f"estimation_{est}_demand_1_prep.py"
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            print(f"[!] Warning: {script_name} not found. Skipping estimation {est}.")
            continue
            
        print(f"\n" + "="*50)
        print(f"Running Demand Prep for Estimation {est}: {script_name}")
        print("="*50)
        
        cmd = [sys.executable, script_name, "--spec", str(args.spec)]
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"[!] Error: {script_name} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print("\n[SUCCESS] Universal Demand Prep Orchestrator Finished successfully.")

if __name__ == "__main__":
    main()
