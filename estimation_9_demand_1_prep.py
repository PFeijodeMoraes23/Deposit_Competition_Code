"""estimation_9_demand_1_prep.py — E9 (OPTIONAL: Joint Single-Index, kernel) demand prep.
phi = rearranged kernel link at the native joint index (si_vgrid/si_ggrid); spec 12 by default.
CLI: --spec ID|all."""
import argparse
import pandas as pd
from estimation_demand_link_common import run

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    p = argparse.ArgumentParser(description="E9 (optional kernel) demand prep")
    p.add_argument("--spec", type=str, default="12", help="Specification ID (default 12) or 'all'")
    a = p.parse_args()
    run(9, "kernel", "sikernel", time_block=False, spec=a.spec)
