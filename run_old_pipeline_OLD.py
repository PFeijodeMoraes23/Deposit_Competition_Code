## run_full_pipeline.py
# Author: Pedro Feijo de Moraes
# Last edited: 2026-03-06
# Run the complete Egan et al. (2025) replication pipeline for BOTH the
# conglomerate-level and the individual-institution-level analyses.
#
# Pipeline stages
#---------------
#  Stage 1 -- COSIF processing (shared input for both pipelines)
#    1a. cosif_process_1.py      Monthly institution COSIF -> implicit funding rates
#
#  Stage 2 -- Institution-level quarterly aggregation
#    2a. cosif_process_2.py      Monthly COSIF -> quarterly institution rates
#
#  Stage 3 -- Panel construction
#   3a. egan_panel_build.py     Conglomerate-level deposit panel
#   3b. egan_panel_build_2.py   Institution-level deposit panel
#
#  Stage 4 -- Structural estimation  (10 specs each: OLS 1-4 + CF 5-10)
#    4a. estimation_1.py         Conglomerate  (PanelOLS / statsmodels)
#    4b. estimation_2.py         Institution   (pyfixest / statsmodels)
#
#  Stage 5 -- Demand estimation  (Berry 1994, active market shares)
#    5a. estimation_demand_1.py  Conglomerate: log(s_active) = alpha_k*rho + FE
#    5b. estimation_demand_2.py  Institution:  log(s_active) = alpha_k*rho + FE
## ---------------------------------------------------------------------------

"""
run_full_pipeline.py
Run the complete Egan et al. (2025) replication pipeline for BOTH the
conglomerate-level and the individual-institution-level analyses.

Pipeline stages
---------------
  Stage 1 -- COSIF processing (shared input for both pipelines)
    1a. cosif_process_1.py      Monthly institution COSIF -> implicit funding rates

  Stage 2 -- Institution-level quarterly aggregation
    2a. cosif_process_2.py      Monthly COSIF -> quarterly institution rates

  Stage 3 -- Panel construction
    3a. egan_panel_build.py     Conglomerate-level deposit panel
    3b. egan_panel_build_2.py   Institution-level deposit panel
    3b. egan_panel_build_2.py   Institution-level deposit panel

  Stage 4 -- Structural estimation  (10 specs each: OLS 1-4 + CF 5-10)
    4a. estimation_1.py         Conglomerate  (PanelOLS / statsmodels)
    4b. estimation_2.py         Institution   (pyfixest / statsmodels)

  Stage 5 -- Demand estimation  (Berry 1994, active market shares)
    5a. estimation_demand_1.py  Conglomerate: log(s_active) = alpha_k*rho + FE
    5b. estimation_demand_2.py  Institution:  log(s_active) = alpha_k*rho + FE

Usage
-----
    python run_full_pipeline.py            # run everything
    python run_full_pipeline.py --from 3   # restart from stage 3 onwards
"""

import subprocess
import sys
import time
import os
import argparse
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)


# Force UTF-8 output so box-drawing characters print on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON     = sys.executable

STEPS = [
    # (stage, script, label)
    (1, "cosif_process_1.py",        "1a  COSIF processing (monthly -> institution rates)"),
    (2, "cosif_process_2.py",        "2a  COSIF quarterly collapse (institution level)"),
    (2, "scr_panel_build.py",        "2b  SCR segment credit-structure panel (instruments)"),
    (3, "egan_panel_build.py",       "3a  Conglomerate panel construction"),
    (3, "egan_panel_build_2.py",     "3b  Institution panel construction"),
    (3, "tarifas_panel_build_1.py",  "3c  Tariff enrichment (institution + conglomerate panels)"),
    (3, "build_loo_instruments.py",  "3d  BLP LOO rival-characteristic instruments"),
    (4, "estimation_1.py",           "4a  Structural estimation -- CONGLOMERATE level"),
    (4, "estimation_2.py",           "4b  Structural estimation -- INSTITUTION level"),
    (5, "estimation_demand_1.py",    "5a  Demand estimation (Berry 1994) -- CONGLOMERATE level"),
    (5, "estimation_demand_2.py",    "5b  Demand estimation (Berry 1994) -- INSTITUTION level"),
]


def run_step(script: str, label: str) -> float:
    path = os.path.join(SCRIPT_DIR, script)
    print(f"\n{'-' * 70}")
    print(f"  {label}")
    print(f"  Script : {script}")
    print(f"{'-' * 70}")
    t0 = time.perf_counter()
    result = subprocess.run([PYTHON, path], cwd=SCRIPT_DIR)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n*** FAILED: {script} exited with code {result.returncode} "
              f"after {elapsed:.1f}s ***")
        sys.exit(result.returncode)
    print(f"  OK  completed in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from", dest="from_stage", type=int, default=1, metavar="STAGE",
        help="Start from this stage number (1-5).  Useful to skip already-run stages.",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    print("=" * 70)
    print("  Egan et al. (2025) -- Full Replication Pipeline")
    print(f"  Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Running from stage {args.from_stage}")
    print("=" * 70)

    timings = {}
    for stage, script, label in STEPS:
        if stage < args.from_stage:
            print(f"  (skipping stage {stage}: {script})")
            continue
        elapsed = run_step(script, label)
        timings[script] = elapsed

    total = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print("  Summary")
    print(f"{'-' * 70}")
    for script, t in timings.items():
        print(f"  {script:<35} {t:>7.1f}s")
    print(f"{'-' * 70}")
    print(f"  Total:  {total:.1f}s  ({total / 60:.1f} min)")
    print(f"  Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
