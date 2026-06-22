"""estimation_5_demand_1_prep.py — Est5 (Probit, normal eta) demand prep.
phi = Phi(native index). CLI: --spec ID|all."""
from estimation_demand_link_common import run

if __name__ == "__main__":
    import pandas as pd
    pd.options.mode.chained_assignment = None
    run(est_num=5, link="probit", tag="probit")
