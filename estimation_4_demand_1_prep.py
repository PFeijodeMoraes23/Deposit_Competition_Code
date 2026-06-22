"""estimation_4_demand_1_prep.py — Est4 (Constrained Linear, uniform eta) demand prep.
phi = clip(native index, 0, 1). CLI: --spec ID|all."""
from estimation_demand_link_common import run

if __name__ == "__main__":
    import pandas as pd
    pd.options.mode.chained_assignment = None
    run(est_num=4, link="uniform", tag="constrained")
