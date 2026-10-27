"""
make_entry_carry.py
===================
Entry and the incumbents' carry: does the carried-forward stock of B incumbents change when a B
firm enters their market? Event study on the E1 frame (sleep_est_e2.build_pooled_data, built from
the market panel) for the B incumbents of markets with a kept B entry.

  Y   deposit balance
  X   the carry (gross return x lagged balance), the five Tech states x carry, event-time dummies
      alone and x carry (h = t - first entry quarter, -4..8 with 8 = 8 or later; h <= -5 omitted)
  FE  bank x type x market and quarter effects; OLS (no control function); conglomerate wild
      cluster bootstrap (Webb, 999 draws) 95% intervals
Incumbents: conglomerates in the market before the entry that are not themselves entrants there.
Run on all kept entries and without Santander's (C0080185, the majority of B entries).

Inputs: the market panel; the entry registry DIAG_PHI_SEPARATION/d6_entry_events.csv (columns
congl, mkt, q_entry, keep, kind), written by sleep_ident_entry_dynamics.py. That script reads the
cluster's demand parquet, so the registry, and with it this file, is refreshed after a cluster run.

Output: Rout/entry_carry.json (keys all, no_santander), read by make_paper_numbers.py
(entry.carry.*) and make_slide_figures.py --only followups (fig_slide_entry_carry_ref).

Usage:  python make_entry_carry.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import datetime
import json

import numpy as np
import pandas as pd

from utils import paths as _paths
from make_xi_diagnostics import _codes, _resid, _stamp

SIDECAR = _paths.rout_dir() / "entry_carry.json"
EVENTS = _paths.rout_dir().parent / "DIAG_PHI_SEPARATION" / "d6_entry_events.csv"
SANTANDER = "C0080185"
H_MIN, H_MAX = -4, 8          # event times reported; h <= -5 omitted, h >= 8 pooled into 8
TECH = ["pix_exists", "cadunico_families_per1000", "fraction_65plus", "risk_free_qoq_lag",
        "connections_per100"]


def entry_carry() -> dict:
    import statsmodels.api as sm
    from utils.sleep_links import linear_wild_cluster_bootstrap
    from sleep_est_e2 import build_pooled_data
    ev = pd.read_csv(EVENTS, dtype={"congl": str, "mkt": str})
    ev = ev[ev["keep"].astype(bool) & (ev["kind"] == "B")]
    df = build_pooled_data()
    df = df[df["is_B"] == 1] if "is_B" in df.columns else df
    df["mca_code"] = df["mca_code"].astype(str)
    df["t"] = df["year"].astype(int) * 4 + df["quarter"].astype(int) - 1
    out = {}
    for tag, e in (("all", ev), ("no_santander", ev[ev["congl"] != SANTANDER])):
        first = e.groupby("mkt")["q_entry"].min()
        entrants = e.groupby("mkt")["congl"].agg(set)
        d = df[df["mca_code"].isin(first.index)].copy()
        d["q0"] = d["mca_code"].map(first)
        d = d[[c not in entrants[m] for c, m in zip(d["CodConglomeradoPrudencial"], d["mca_code"])]]
        pre = d[d["t"] < d["q0"]].groupby(["CodConglomeradoPrudencial", "mca_code"]).size()
        inc = set(pre.index)
        d = d[[k in inc for k in zip(d["CodConglomeradoPrudencial"], d["mca_code"])]]
        h = (d["t"] - d["q0"]).clip(upper=H_MAX)
        carry = d["nr_lagged_dep"].to_numpy()
        X = {"carry": carry}
        for s in TECH:
            X[f"{s}_x_carry"] = d[s].to_numpy() * carry
        hs = list(range(H_MIN, H_MAX + 1))
        for k in hs:
            ind = (h == k).to_numpy().astype(float)
            X[f"h{k}"] = ind
            X[f"h{k}_x_carry"] = ind * carry
        Xd = pd.DataFrame(X)
        cols = list(Xd.columns)
        Y = np.column_stack([d["deposit_balance"].to_numpy(), Xd.to_numpy()])
        w = np.ones(len(d))
        R = _resid(Y, w, [_codes(d["entity_id"]), _codes(d["t"])])
        res = sm.OLS(R[:, 0], pd.DataFrame(R[:, 1:], columns=cols)).fit(
            cov_type="cluster", cov_kwds={"groups": d["CodConglomeradoPrudencial"].to_numpy()})
        band = {}
        linear_wild_cluster_bootstrap(res, B=999, scheme="webb", seed=0, mode="normal",
                                      band_out=band)
        bb = band["band"].set_index("name")
        r = {str(k): dict(estimate=float(res.params[f"h{k}_x_carry"]),
                          ci=dict(lo=float(bb.loc[f"h{k}_x_carry", "lo_bc"]),
                                  hi=float(bb.loc[f"h{k}_x_carry", "hi_bc"])),
                          stars=str(bb.loc[f"h{k}_x_carry", "stars"])) for k in hs}
        out[tag] = dict(h=r, carry=float(res.params["carry"]), n_markets=int(d["mca_code"].nunique()),
                        n_events=int(len(e)), n_obs=int(len(d)),
                        n_banks=int(d["CodConglomeradoPrudencial"].nunique()))
    return out


def main():
    res = entry_carry()
    for tag, v in res.items():
        print(f"{tag}: carry {v['carry']:.3f}, {v['n_markets']} markets, {v['n_obs']:,} obs, "
              f"{v['n_banks']} banks")
        print("   " + ", ".join(f"h{k} {x['estimate']:+.3f}{x['stars']}" for k, x in v["h"].items()))
    meta = dict(events=_stamp(EVENTS), written=datetime.datetime.now().isoformat(timespec="seconds"))
    SIDECAR.write_text(json.dumps(dict(_meta=meta, **res), indent=2, default=float),
                       encoding="utf-8")
    print(f"  wrote {SIDECAR}")


if __name__ == "__main__":
    main()
