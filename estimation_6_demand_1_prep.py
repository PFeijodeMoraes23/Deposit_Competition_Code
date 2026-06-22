"""estimation_6_demand_1_prep.py — Est6 (Single-Index, nonparametric eta) demand prep.
phi = clip(cubic-sieve link of native index, 0, 1), using the stored si_b/si_vmu/si_vsd.
CLI: --spec ID|all."""
from estimation_demand_link_common import run

if __name__ == "__main__":
    import pandas as pd
    pd.options.mode.chained_assignment = None
    run(est_num=6, link="index", tag="index")
