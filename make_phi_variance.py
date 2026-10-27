"""
make_phi_variance.py
====================
How the estimated sleepy share phi-hat_mt spreads across markets and quarters (routines E3, E4,
specification 12), from the spec-12 cells of utils/sleep_cells.py (the stored sleep fit; needs the
cluster's sleepiness run).

  variance split  order-free (Shapley) shares of the variance of phi-hat over MCA-quarters among
                  MCA effects, quarter effects and the rest, with population and lagged-deposit
                  weights, with and without the MCA-quarters at the ceiling phi-hat = 1
  index split     Upsilon_k cov(S_k, index) / var(index): the covariance split of the native index
                  Upsilon-hat'S_mt by state, population weights (sums to one)
  ceiling         the share of MCA-quarters at phi-hat = 1

phi-hat is a deterministic function of the states, so this describes how the estimated function
spreads the states, not separate evidence on sleepiness.

Output: Rout/phi_variance.json (keys E3, E4), read by make_paper_numbers.py (sleep.phi_anova.*,
sleep.phi_index.*, sleep.phi_ceiling.*) and make_slide_figures.py --only followups
(fig_slide_phi_anova_ref).

Usage:  python make_phi_variance.py [--refresh]    (--refresh rebuilds the cached cells)
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import datetime
import json
import sys

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils.sleep_cells import ROUTINES, STATES, pre_cut
from make_xi_diagnostics import _codes, _shapley

SIDECAR = _paths.rout_dir() / "phi_variance.json"
CEIL = 1.0 - 1e-9


def _terms(ups: dict) -> dict:
    """Index terms: the five states under their short names, then any other term of the fit
    (E4's time block) under its own column name."""
    t = dict(STATES)
    t.update({k: k for k in ups if k != "constant" and k not in STATES.values()})
    return t


def phi_variance(d: pd.DataFrame, ups: dict) -> dict:
    terms = _terms(ups)
    b = d[d["is_B"].astype(bool)]
    g = b.groupby(["mca_code", "time_id"])
    # phi-hat is a market-quarter object except where the lagged Selic is lagged within a
    # bank x type x market series that skips a quarter (it then holds an older Selic): the
    # market-quarter takes the median, and the count of such cells is recorded.
    mq = g.agg(phi=("phi_mt", "median"), phi_rng=("phi_mt", lambda x: x.max() - x.min()),
               index=("phi_index", "median"), pop=("pop_total", "first"),
               dep=("lagged_deposits", "sum"),
               **{k: (v, "median") for k, v in terms.items()}).reset_index()
    vary = mq["phi_rng"] > 1e-9
    out = dict(n_cells=int(len(mq)), n_mca=int(mq["mca_code"].nunique()),
               n_quarters=int(mq["time_id"].nunique()), n_cells_varying=int(vary.sum()),
               max_within_range=float(mq["phi_rng"].max()))
    at_ceiling = mq["phi"] >= CEIL
    out["ceiling"] = dict(share=float(at_ceiling.mean()),
                          share_pop=float(np.average(at_ceiling, weights=mq["pop"])),
                          share_dep=float(np.average(at_ceiling, weights=mq["dep"])))
    for tag, frame in (("", mq), ("_noceil", mq[~at_ceiling])):
        for wname in ("pop", "dep"):
            f = frame[frame[wname] > 0]
            phi, r2 = _shapley(f["phi"].to_numpy(), f[wname].to_numpy().astype(float),
                               {"market": _codes(f["mca_code"]), "quarter": _codes(f["time_id"])})
            out[f"{wname}{tag}"] = dict(market=phi["market"], quarter=phi["quarter"],
                                        rest=1.0 - r2, n=int(len(f)))
    # Covariance split of the native index (linear in the states), population weights.
    f = mq[mq["pop"] > 0]
    w = f["pop"].to_numpy().astype(float)
    idx = f["index"].to_numpy()
    rebuilt = ups.get("constant", 0.0) + sum(ups[v] * f[k].to_numpy() for k, v in terms.items())
    exact = ~(f["phi_rng"] > 1e-9).to_numpy()
    if float(np.max(np.abs(rebuilt - idx)[exact])) > 1e-8:
        sys.exit("the index is not constant + sum of its terms: the term list differs")
    # Split the index rebuilt from the market-quarter medians of the terms, which is exactly
    # linear in them (equal to the fitted index wherever no term varies within the cell).
    ic = rebuilt - np.average(rebuilt, weights=w)
    var = float(np.average(ic ** 2, weights=w))
    split = {}
    for k, v in terms.items():
        s = f[k].to_numpy() - np.average(f[k].to_numpy(), weights=w)
        split[k] = float(ups[v] * np.average(s * ic, weights=w) / var)
    assert abs(sum(split.values()) - 1.0) < 1e-9, split
    out["index"] = split
    out["upsilon"] = {k: ups[v] for k, v in terms.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--refresh", action="store_true", help="rebuild the cached spec-12 cells")
    a = ap.parse_args()
    res, meta = {}, {}
    for e in ROUTINES:
        d, ups, key = pre_cut(e, a.refresh)
        meta[f"E{e}_inputs"] = key
        c = res[f"E{e}"] = phi_variance(d, ups)
        print(f"E{e} pop {c['pop']} | dep {c['dep']} | ceiling {c['ceiling']['share']:.3f}"
              f" | index {c['index']}")
    meta["written"] = datetime.datetime.now().isoformat(timespec="seconds")
    SIDECAR.write_text(json.dumps(dict(_meta=meta, **res), indent=2, default=float),
                       encoding="utf-8")
    print(f"  wrote {SIDECAR}")


if __name__ == "__main__":
    main()
