"""
export_results.py
=================
Exports the first and second stage regression summaries for all specifications
from a specific sleepiness estimation step (1 through 6).

CLI Usage Examples:
-------------------
  python export_results.py --estimation 5
  python export_results.py --estimation all

Estimation Procedures Overview:
-------------------------------
  Estimation 1 (Sleepiness New): Computes baseline consumer inertia (sleepiness) for traditional B-type banks at the national level.
  Estimation 2 (Sleepiness National): Computes baseline inertia for national/digital D-type banks using a post-2020 structural break dummy.
  Estimation 3 (Sleepiness No Break): Acts as a robustness bound by calculating inertia locally for traditional banks natively using a risk-free rate lag instead of a break dummy.
  Estimation 4 (Sleepiness Pooled 4): Pools estimations to calculate cross-sectional inertia with additional controls and IV strategies.
  Estimation 5 (Sleepiness Pooled 5 / NLLS): Implements a Non-Linear Least Squares (NLLS) optimization to explicitly model parameters structurally, capturing non-linear interactions natively.
  Estimation 6 (Sleepiness Pooled 6 / Alt Controls): Runs pooled robustness checks specifically interacting base variables with state-owned and cooperative bank indicators (Alt 1 and Alt 2).
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

# Try to respect the project's venv guard
try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except ImportError:
    pass

def get_estimation_directories(estimation_number: int, base_dir: Path) -> list[Path]:
    """Returns the list of output directories for the given estimation step."""
    sleep_dir = base_dir / "ESTIMATION_OUTPUT" / "SLEEPINESS"
    
    mapping = {
        1: [sleep_dir / "SLEEPINESS_NEW"],
        2: [sleep_dir / "SLEEPINESS_NATIONAL"],
        3: [sleep_dir / "SLEEPINESS_NO_BREAK" / "LOCAL", 
            sleep_dir / "SLEEPINESS_NO_BREAK" / "NATIONAL"],
        4: [sleep_dir / "SLEEPINESS_4" / "POOLED"],
        5: [sleep_dir / "SLEEPINESS_5" / "POOLED"],
        6: [sleep_dir / "SLEEPINESS_6" / "POOLED" / "ALT_1",
            sleep_dir / "SLEEPINESS_6" / "POOLED" / "ALT_2"],
    }
    
    if estimation_number not in mapping:
        raise ValueError(f"Invalid estimation number: {estimation_number}. Must be between 1 and 6.")
        
    return mapping[estimation_number]

def get_fallback_summary_text(res):
    """Extract standard errors and p-values for statsmodels-like NonLinearResults lacking .summary()"""
    try:
        lines = ["--- NonLinear Optimization Results ---"]
        lines.append(f"{'Parameter':<35} {'Coef.':>15} {'Std.Err':>15} {'t':>15} {'P>|t|':>15}")
        lines.append("-" * 100)
        for i, name in enumerate(res.params.index):
            coef = res.params.iloc[i]
            se = res.bse[i] if i < len(res.bse) else float('nan')
            t = res.tvalues[i] if i < len(res.tvalues) else float('nan')
            p = res.pvalues[i] if i < len(res.pvalues) else float('nan')
            lines.append(f"{name:<35} {coef:>15.4f} {se:>15.4f} {t:>15.4f} {p:>15.4f}")
        if hasattr(res, 'G_star'):
            lines.append(f"\nDegrees of Freedom (G_star): {res.G_star}")
        return "\n".join(lines)
    except Exception as e:
        return f"Unable to generate summary text: {e}"

def get_fallback_summary_csv(res):
    try:
        lines = ["Parameter,Coef,Std.Err,t,P>|t|"]
        for i, name in enumerate(res.params.index):
            coef = res.params.iloc[i]
            se = res.bse[i] if i < len(res.bse) else float('nan')
            t = res.tvalues[i] if i < len(res.tvalues) else float('nan')
            p = res.pvalues[i] if i < len(res.pvalues) else float('nan')
            lines.append(f"{name},{coef},{se},{t},{p}")
        return "\n".join(lines)
    except Exception as e:
        return f"Unable to generate summary csv: {e}"

def export_specification_results(results_dict, output_folder: Path):
    """Iterates through all specifications in the dictionary and exports their 1st/2nd stages."""
    
    # Create an exports subdirectory to keep things clean
    export_dir = output_folder / "EXPORTS"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing results in: {output_folder}")
    print(f"Exporting isolated summaries to: {export_dir}")
    
    for spec_name, data in results_dict.items():
        
        # Build dynamic safe name matching underlying dictionary features
        prefixes = []
        if "spec_number" in data: prefixes.append(f"Spec_{data['spec_number']:02d}")
        if "option" in data: prefixes.append(f"Opt_{data['option']}")
        if "block" in data: prefixes.append(f"Block_{data['block']}")
        
        prefix_str = "_".join(prefixes) + "_" if prefixes else ""
        safe_name = f"{prefix_str}{spec_name}".replace(" ", "_").replace("/", "").replace(":", "").replace("__", "_")
        
        # 1. Export Second Stage
        ss_res = data.get("second_stage")
        if ss_res:
            ss_path_txt = export_dir / f"{safe_name}_SecondStage.txt"
            ss_path_csv = export_dir / f"{safe_name}_SecondStage.csv"
            
            with open(ss_path_txt, "w") as f:
                f.write(ss_res.summary().as_text() if hasattr(ss_res, 'summary') else get_fallback_summary_text(ss_res))
                
            with open(ss_path_csv, "w") as f:
                f.write(ss_res.summary().as_csv() if hasattr(ss_res, 'summary') else get_fallback_summary_csv(ss_res))
                
            print(f"  -> Exported 2nd Stage: {ss_path_txt.name}")
        
        # 2. Export First Stage (if applicable)
        fs_res = data.get("first_stage")
        if fs_res:
            fs_path_txt = export_dir / f"{safe_name}_FirstStage.txt"
            fs_path_csv = export_dir / f"{safe_name}_FirstStage.csv"
            
            with open(fs_path_txt, "w") as f:
                f.write(fs_res.summary().as_text() if hasattr(fs_res, 'summary') else get_fallback_summary_text(fs_res))
                
            with open(fs_path_csv, "w") as f:
                f.write(fs_res.summary().as_csv() if hasattr(fs_res, 'summary') else get_fallback_summary_csv(fs_res))
                
            print(f"  -> Exported 1st Stage: {fs_path_txt.name}")

def main():
    parser = argparse.ArgumentParser(description="Export estimation results for steps 1-6.")
    parser.add_argument("--estimation", choices=['1', '2', '3', '4', '5', '6', 'all'], required=True,
                        help="Specify the estimation step number (1 to 6) or 'all' to export.")
    
    args = parser.parse_args()
    
    # Resolve the data directory dynamically based on the script's location
    _ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
    
    est_list = [1, 2, 3, 4, 5, 6] if args.estimation == 'all' else [int(args.estimation)]
    
    target_dirs = []
    try:
        for est in est_list:
            target_dirs.extend(get_estimation_directories(est, DATA_DIR))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    found_data = False
    
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
