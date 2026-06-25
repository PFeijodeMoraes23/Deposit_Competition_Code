"""estimation_9_sleep.py — E9 (OPTIONAL / robustness): Pooled JOINT Single-Index
with a KERNEL local-linear link (joint SLS; link monotonised ex post by
rearrangement). The kernel backfit is expensive, so E9 defaults to SPEC 12 ONLY and
is NOT part of the default sleep pipeline (run explicitly). Thin config over
estimation_sleep_common.run_sleep_estimator.

  python estimation_9_sleep.py            # spec 12 only (default)
  python estimation_9_sleep.py --full     # full 12-spec grid (very slow)
"""
import argparse
from estimation_sleep_common import run_sleep_estimator

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="E9 (optional): Pooled Joint Single-Index (kernel)")
    p.add_argument("--full", action="store_true", help="Full 12-spec grid (default: spec 12 only)")
    args = p.parse_args()
    run_sleep_estimator(9, "joint_kernel", time_block=False, spec12_only=not args.full)
    print("\n--- Pipeline 9 (Joint Single-Index, kernel; optional) Completed ---")
