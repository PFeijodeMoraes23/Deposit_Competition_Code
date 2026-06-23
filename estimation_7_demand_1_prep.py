"""estimation_7_demand_1_prep.py — Est7 (Joint Single-Index, monotone sieve) demand prep.
phi = stored monotone I-spline link evaluated at the NATIVE joint index (si_vgrid/si_ggrid),
identical to estimation_7_sleep.calculate_phis. CLI: --spec ID|all."""
from estimation_demand_link_common import run

if __name__ == "__main__":
    import pandas as pd
    pd.options.mode.chained_assignment = None
    run(est_num=7, link="sieve", tag="sijoint")
