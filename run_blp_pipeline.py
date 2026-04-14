"""
Parallel executing pipeline that compiles the BLP estimation results into LaTeX tables.
Launches the generation of tables for routines 1, 3, 4, 5, and 6 concurrently.
Output tables are exported back to the BLP_RESULTS directory.
"""
import subprocess
import pathlib
import sys
import concurrent.futures

ROOT = pathlib.Path(__file__).resolve().parent
PYTHON_EXE = sys.executable
LATEX_SCRIPT = ROOT / "make_blp_latex_tables.py"

def run_latex_for_est(est_id: int):
    print(f"Launching LaTeX builder for Estimation {est_id}...")
    cmd = [PYTHON_EXE, str(LATEX_SCRIPT), "--est", str(est_id)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (est_id, True, res.stdout)
    except subprocess.CalledProcessError as e:
        return (est_id, False, f"Error:\n{e.stderr}\n{e.stdout}")

def main():
    routines = [1, 3, 4, 5, 6]
    
    if not LATEX_SCRIPT.exists():
        print(f"Cannot find script at: {LATEX_SCRIPT}")
        sys.exit(1)

    print(f"=== Beginning Parallel BLP LaTeX Pipeline for Routines {routines} ===")
    
    # Run concurrently using ProcessPool (avoids GIL constraints and runs securely on Windows)
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(routines)) as executor:
        futures = {executor.submit(run_latex_for_est, r): r for r in routines}
        
        for future in concurrent.futures.as_completed(futures):
            est_id, success, output = future.result()
            print(f"\n--- Output from Estimation {est_id} ---")
            print(output.strip())
            if not success:
                print(f"[!] Estimation {est_id} LaTeX generation FAILED.")
            else:
                print(f"[+] Estimation {est_id} LaTeX generation SUCCESSFUL.")
                
    print("\n=== BLP LaTeX Pipeline Complete ===")

if __name__ == "__main__":
    main()
