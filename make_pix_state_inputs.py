"""
make_pix_state_inputs.py
========================
Writes the two inputs of utils/pix_state.py (the Pix state of the sleepiness index under
SLEEP_PIX_STATE=use_cf) into processed/PANEL_INTERMED, where the sleepiness frame builder and the
demand preps read them (on the cluster they are data/input files linked there by
cluster_lib.sh cl_bootstrap_tree):

  pix_exposures_2019_mca.csv   one row per MCA: the three 2019 exposures of make_pix_exposure.py
                               (mandated, mobile, g45), cad2019 and pop2019 (exposures())
  pix_use_mca_quarter.csv      one row per MCA x quarter, 2020Q4 to 2024Q4: t, payers_pc (Pix payers
                               per inhabitant), pop_total and the state (Rout/pix_usage_mca_quarter.parquet)

Check before writing: the first stage that utils/pix_state.py fits on these two files gives the
stored first stage of Rout/pix_exposure.json (the joint F of the three exposures, state and MCA
clusters, make_pix_exposure.first_stage). Refuses to write otherwise.

Usage:  python make_pix_state_inputs.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import json
import sys

import numpy as np
import pandas as pd

from utils import paths as _paths
from make_pix_exposure import EXPOSURES, exposures, first_stage, wstats

OUT = _paths.PROCESSED / "PANEL_INTERMED"


def main():
    x = exposures()
    mom = wstats(x)
    u = pd.read_parquet(_paths.rout_dir() / "pix_usage_mca_quarter.parquet")
    u["mca_code"] = u["mca_code"].astype(str)
    # The stored first stage, reproduced on exactly the sample utils/pix_state.py will use.
    d = u.merge(x, left_on="mca_code", right_index=True, how="inner").dropna(
        subset=["pop_total", "cad2019", *EXPOSURES, "payers_pc"])
    d = d[d["pop_total"] > 0].reset_index(drop=True)
    for e in EXPOSURES:
        d[f"z_{e}"] = (d[e] - mom[e]["mean"]) / mom[e]["sd"]
    got = {cl: first_stage(d, list(EXPOSURES), "payers", False, True, cl)["joint_f"]
           for cl in ("state", "mca_code")}
    stored = json.loads((_paths.rout_dir() / "pix_exposure.json").read_text(encoding="utf-8"))["fs"]["payers"]
    ref = {"state": stored["joint_f_state"], "mca_code": stored["joint_f"]}
    dev = max(abs(got[k] - ref[k]) / abs(ref[k]) for k in got)
    print(f"joint F of the exposures: state {got['state']:.3f} (stored {ref['state']:.3f}), "
          f"MCA {got['mca_code']:.3f} (stored {ref['mca_code']:.3f}); max relative deviation {dev:.2e}")
    if dev > 1e-8:
        sys.exit("the first stage does not reproduce Rout/pix_exposure.json; nothing written")
    OUT.mkdir(parents=True, exist_ok=True)
    ex = x.reset_index()[["mca_code", "mandated", "mobile", "g45", "cad2019", "pop2019"]]
    ex.to_csv(OUT / "pix_exposures_2019_mca.csv", index=False)
    us = u[["mca_code", "year", "quarter", "t", "payers_pc", "pop_total", "state"]].copy()
    us["state"] = us["state"].astype("Int64").astype(str)
    us.sort_values(["mca_code", "t"]).to_csv(OUT / "pix_use_mca_quarter.csv", index=False)
    print(f"wrote {OUT / 'pix_exposures_2019_mca.csv'} ({len(ex):,} MCAs) and "
          f"{OUT / 'pix_use_mca_quarter.csv'} ({len(us):,} MCA-quarters)")


if __name__ == "__main__":
    main()
