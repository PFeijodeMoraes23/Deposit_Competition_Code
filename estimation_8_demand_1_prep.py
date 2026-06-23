"""estimation_8_demand_1_prep.py — Est8 (Joint Single-Index, kernel) demand prep.
phi = rearranged kernel link evaluated at the NATIVE joint index (si_vgrid/si_ggrid),
identical to estimation_8_sleep.calculate_phis. Spec 12 only (kernel cost).
CLI: --spec ID|all (only spec 12 has a fitted result)."""
from estimation_demand_link_common import run

if __name__ == "__main__":
    import pandas as pd
    pd.options.mode.chained_assignment = None
    run(est_num=8, link="kernel", tag="sikernel")
