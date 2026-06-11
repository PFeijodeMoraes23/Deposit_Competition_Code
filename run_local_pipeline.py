"""
run_local_pipeline.py
=====================
Orchestrates the full local pipeline in three sequential stages:

  1. run_data_pipeline.py      — download & build the master panel
  2. run_sleep_pipeline.py     — sleepiness / inertia estimations
  3. run_blp_pipeline.py       — logit sanity check, then LaTeX tables
                                 (equivalent to --logit-then-latex)
                                 OR, with --coherence, the coherence RC-BLP
                                 (E1/E2/E3, spec 12) instead of the logit stage,
                                 OR, with --logit-coherence, the post-fix non-RC
                                 logit (E1/E2/E3, spec 12) instead of legacy logit.

All CLI args passed to this script are forwarded to run_data_pipeline.py, except
--coherence and --logit-coherence which are consumed here to select the BLP stage.
The sleep and BLP stages are otherwise run with default arguments.

Usage
-----
  python run_local_pipeline.py                    # run all three stages (BLP = legacy logit)
  python run_local_pipeline.py --logit-coherence  # final stage = post-fix non-RC logit (3 routines)
  python run_local_pipeline.py --coherence        # final stage = coherence RC-BLP (GPU)
  python run_local_pipeline.py --from 2           # restart data pipeline from stage 2
  python run_local_pipeline.py --only 3           # run only stage 3 of data pipeline
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
    # Forward all CLI args to the data pipeline only. `--coherence` and
    # `--logit-coherence` are consumed here (not data-pipeline flags): they swap the
    # final BLP stage. `--coherence` → RC-BLP (E1/E2/E3, spec 12; needs a CUDA GPU).
    # `--logit-coherence` → post-fix non-RC logit (E1/E2/E3, spec 12; local).
    extra_args = sys.argv[1:]
    coherence = "--coherence" in extra_args
    logit_coherence = "--logit-coherence" in extra_args
    if coherence or logit_coherence:
        extra_args = [a for a in extra_args
                      if a not in ("--coherence", "--logit-coherence")]
    if coherence and logit_coherence:
        print("[ERROR] Use only one of --coherence / --logit-coherence.")
        sys.exit(1)

    _run(
        "Data Pipeline (run_data_pipeline.py)",
        [PYTHON_EXE, str(ROOT / "run_data_pipeline.py")] + extra_args,
    )

    _run(
        "Sleep Pipeline (run_sleep_pipeline.py)",
        [PYTHON_EXE, str(ROOT / "run_sleep_pipeline.py")],
    )

    if coherence:
        _run(
            "BLP Coherence RC-BLP (run_blp_pipeline.py --coherence)",
            [PYTHON_EXE, str(ROOT / "run_blp_pipeline.py"), "--coherence"],
        )
    elif logit_coherence:
        _run(
            "BLP Logit Coherence + LaTeX (run_blp_pipeline.py --logit-coherence)",
            [PYTHON_EXE, str(ROOT / "run_blp_pipeline.py"), "--logit-coherence"],
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
