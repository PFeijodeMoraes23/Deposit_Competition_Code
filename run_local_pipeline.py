"""
run_local_pipeline.py
=====================
Orchestrates the full local pipeline in three sequential stages:

  1. run_data_pipeline.py      — download & build the master panel
  2. run_sleep_pipeline.py     — sleepiness / inertia estimations
  3. run_blp_pipeline.py       — non-RC logit (all routines, spec 12) + LaTeX tables,
                                 OR, with --rc, the RC-BLP estimation (default E5-E8,
                                 spec 12) instead of the logit stage.

All CLI args passed to this script are forwarded to run_data_pipeline.py, except --rc
which is consumed here to select the BLP stage. The sleep and BLP stages are otherwise
run with default arguments.

Usage
-----
  python run_local_pipeline.py               # run all three stages (BLP = non-RC logit)
  python run_local_pipeline.py --rc          # final stage = RC-BLP (GPU)
  python run_local_pipeline.py --from 2      # restart data pipeline from stage 2
  python run_local_pipeline.py --only 3      # run only stage 3 of data pipeline
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import subprocess
import sys
import time
from pathlib import Path

ROOT       = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="latin-1", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="latin-1", errors="replace", line_buffering=True)


def _run(label: str, cmd: list):
    print(f"\n{'=' * 60}")
    print(f"  STARTING: {label}")
    print(f"{'=' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    if result.returncode != 0:
        print(f"\n[!] FAILED: {label}  (exit code {result.returncode})")
        print(f"    Elapsed: {mins}m {secs}s")
        sys.exit(result.returncode)
    print(f"\n[+] DONE:  {label}  ({mins}m {secs}s)")


def main():
    # Forward all CLI args to the data pipeline only. `--rc` is consumed here (not a
    # data-pipeline flag): it swaps the final BLP stage to the RC-BLP estimation
    # (default E5-E8, spec 12; needs a CUDA GPU). Without it the final stage is the
    # non-RC logit (all routines, spec 12; local).
    extra_args = sys.argv[1:]
    rc = "--rc" in extra_args
    if rc:
        extra_args = [a for a in extra_args if a != "--rc"]

    _run(
        "Data Pipeline (run_data_pipeline.py)",
        [PYTHON_EXE, str(ROOT / "run_data_pipeline.py")] + extra_args,
    )

    _run(
        "Sleep Pipeline (run_sleep_pipeline.py)",
        [PYTHON_EXE, str(ROOT / "run_sleep_pipeline.py")],
    )

    if rc:
        _run(
            "BLP RC-BLP (run_blp_pipeline.py --rc)",
            [PYTHON_EXE, str(ROOT / "run_blp_pipeline.py"), "--rc"],
        )
    else:
        _run(
            "BLP Logit + LaTeX (run_blp_pipeline.py --logit)",
            [PYTHON_EXE, str(ROOT / "run_blp_pipeline.py"), "--logit"],
        )

    print(f"\n{'=' * 60}")
    print("  ALL STAGES COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
