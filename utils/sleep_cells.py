"""
utils/sleep_cells.py
====================
Every specification-12 demand cell before the active-deposit cut (14-B), with phi-hat, the native
index Upsilon-hat'S_mt and the active balance before its floor at zero, for routines E3 and E4.
Read by make_phi_variance.py (the variance split of phi-hat) and make_active_truncation.py (the
cells the cut truncates).

The cells come from the demand prep's own function (sleep_demand_prep_link.process_specification
with pre_cut=True) on the stored sleep fit (est{k}/estimation_results.pkl), so nothing is
re-derived or re-estimated. Building them loads the fit and the market panel (several GB); the
result is cached in Rout/sleep_cells_cache/, keyed on the size and modification time of both
inputs, and rebuilt only when either changes or --refresh is passed.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from utils import paths as _paths

SPEC = 12
ROUTINES = (3, 4)
CACHE = _paths.rout_dir() / "sleep_cells_cache"
# The five states of S_mt under their short names.
STATES = {"cadunico": "cadunico_families_per1000", "age65": "fraction_65plus",
          "selic": "risk_free_qoq_lag", "mobile": "connections_per100", "pix": "pix_exists"}


def pre_cut(est: int, refresh: bool) -> tuple[pd.DataFrame, dict, dict]:
    """Every spec-12 cell before the (14-B) cut, with phi-hat, the native index and the active
    balance before the floor; the index coefficients; the input stamps. Cached on the inputs."""
    import sleep_demand_prep_link as P
    from make_xi_diagnostics import _stamp
    pkl = _paths.est_dir(est) / "estimation_results.pkl"
    key = dict(pkl=_stamp(pkl), panel=_stamp(Path(P.PANEL_CSV)))
    CACHE.mkdir(parents=True, exist_ok=True)
    out, meta_p = CACHE / f"precut_E{est}.parquet", CACHE / f"precut_E{est}.json"
    if out.is_file() and meta_p.is_file() and not refresh:
        m = json.loads(meta_p.read_text())
        if m["key"] == key:
            return pd.read_parquet(out), m["upsilon"], key
    import gc
    link, _tag, time_block = P.DEMAND_CFG[est]
    df_base = P.build_base_panel(P.PANEL_CSV, time_block=time_block)
    with open(pkl, "rb") as f:
        rd = pickle.load(f)
    name = next(k for k in rd if P.SPEC_MAP[SPEC] in k)
    ups = P.extract_upsilon_terms(rd[name]["second_stage"])
    d, _s, _n = P.process_specification(name, rd[name], df_base, link, pre_cut=True)
    del df_base, rd
    gc.collect()
    keep = ["CodConglomeradoPrudencial", "mca_code", "year", "quarter", "time_id", "deposit_type",
            "is_B", "deposit_balance", "lagged_deposits", "gross_return_lag", "pop_total",
            "phi_mt", "phi_t", "phi_index", "active_raw", *STATES.values(),
            *(k for k in ups if k != "constant")]
    d = d[[c for c in dict.fromkeys(keep) if c in d.columns]].reset_index(drop=True)
    d["mca_code"] = d["mca_code"].astype(str)
    d.to_parquet(out)
    meta_p.write_text(json.dumps(dict(key=key, upsilon=ups), indent=2))
    return d, ups, key
