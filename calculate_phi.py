def compute_phi(df, res, s_cols):
    import numpy as np
    import pandas as pd
    
    phi = np.zeros(len(df))
    for v in s_cols:
        col_name = f"interaction_{v}" if v != "constant" else "nr_lagged_dep"
        c = res.params[col_name]
        phi += c if v == "constant" else c * df[v]
