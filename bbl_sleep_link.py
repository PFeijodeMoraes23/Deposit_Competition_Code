"""
bbl_sleep_link.py
=================
Export the fitted sleepiness model in a form the BBL forward simulation can evaluate in Julia,
so the sleepy share can move along each simulated path:

    phi_{i,t} = G( sum_k theta_k * S_{k,i,t} )

G is the routine's own link and theta its own index direction (params_native: AMEs never enter
phi). The forward simulation (cf_phi_path.jl) re-evaluates this every period with the states
advanced along the path; this script only writes down the fitted object and proves that it
reproduces the demand parquet's phi_mt at the launch quarter.

OUTPUT (one small JSON per routine, ~20 kB)
  COST_FWD/sleep_link_E{k}_spec_{s}.json
    index.terms   the index terms in the estimator's summation order (params_native order):
                  param, state column, coefficient, and the ROLE that fixes how the state moves
                  along a simulated path:
                    constant  the intercept (nr_lagged_dep -> a column of ones)
                    held      frozen at the launch quarter: pix_exists (no foresight of the
                              2020Q4 launch) and the +Time block (E4's gdp_growth_yoy)
                    selic     risk_free_qoq_lag: the launch vintage's forward curve, lagged one
                              quarter, in estimation units (level / scale - grand mean)
                    market    a market demographic that mean-reverts within its MCA at the
                              market_states AR(1) of bbl_transitions.json
    link          kind "index_sieve": the monotone I-spline stored as a grid over the FULL
                  native index. phi = clip(numpy.interp(v, vgrid, ggrid), 0, 1), i.e. linear
                  between grid points and flat beyond both ends -- exactly what
                  utils.sleep_links.phi_from_native and the demand prep evaluate. The I-spline
                  itself (knots, beta, degree, vmu, vsd) is recorded for provenance; the grid is
                  what is evaluated.
    states        per state: scale divisor and grand mean from utils/state_transform.py (the one
                  source of both), display units, and the parquet column it is read from.
    time_block    E4 only: the time variable, its coefficient, the hold rule, and the mean
                  contribution theta_g * gdp_growth_yoy by launch quarter (information only).
    t0_check      the index rebuilt from the parquet's own state columns, in the estimator's
                  summation order, against the parquet's phi_mt: max |difference| over every row.
    provenance    pickle / parquet / state-transform / transitions paths, sizes, mtimes, sha256.

THE HARD CHECK. A link that does not reproduce phi_mt at t=0 would simulate a different
sleepiness model from the one the demand block was estimated with, so the export REFUSES
(exit 1) when max |phi_0 - phi_mt| > --tol (1e-10). bbl_fwd_sim.jl repeats the same check on the
cluster against the parquet it actually reads.

WHERE IT RUNS. Locally, from the est{k} pickles and the demand parquets that were ingested from
the cluster; the JSONs are then uploaded to data/input. It runs unchanged on the cluster too
(est_dir / demand_parquet_dir / cost_fwd_dir follow SLEEP_OUT_ROOT / DEMAND_PREP_DIR /
CF_COST_FWD), which is the route if the cluster's sleep fit is ever re-estimated.

Usage:
  python bbl_sleep_link.py                  # E3 and E4, spec 12, into COST_FWD
  python bbl_sleep_link.py --est 3 --spec 12 --out-dir <dir>
"""
from __future__ import annotations

try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except Exception:
    pass

import argparse
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils import routines as _rt
from utils import state_transform as _st
from utils.sleep_links import NonLinearResults, phi_from_native  # noqa: F401 (unpickling)

SCHEMA = "bbl_sleep_link/1"
TOL_T0 = 1e-10

# How each index state moves along a simulated path. `market` is not listed: a state is a
# market state exactly when bbl_transitions.json carries a numeric AR(1) for it under
# market_states -- the same file, and the same entry, the demand block's StateEvolution reads.
ROLE_OF = {
    "constant": "constant",
    "pix_exists": "held",           # user decision 2026-09-28: no foresight of the 2020Q4 launch
    "gdp_growth_yoy": "held",       # E4's +Time block, held at the launch quarter
    "time_trend": "held",
    "risk_free_qoq_lag": "selic",
}
UNITS = {
    "constant": "1",
    "pix_exists": "0/1 indicator, grand-mean centred",
    "cadunico_families_per1000": "hundreds of CadUnico families per 1,000 inhabitants, grand-mean centred",
    "fraction_65plus": "share of population aged 65+ (decimal), grand-mean centred",
    "connections_per100": "mobile connections per inhabitant (per-100 / 100), grand-mean centred",
    "risk_free_qoq_lag": "quarterly Selic (decimal), lagged one quarter, grand-mean centred",
    "gdp_growth_yoy": "year-over-year growth of GDP per capita (decimal), grand-mean centred",
    "gdp_per_capita": "10k R$ per capita, grand-mean centred",
    "fraction_young": "share of population young (decimal), grand-mean centred",
}


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _file_meta(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "sha256": _sha256(path),
    }


def _floats(x) -> list:
    return [float(v) for v in np.asarray(x, dtype=float).ravel()]


def _spec_key(results: dict, spec: int) -> str:
    from sleep_demand_prep_link import SPEC_MAP     # the repo-wide spec numbering
    target = SPEC_MAP.get(spec)
    if target is None:
        raise SystemExit(f"spec {spec} is not in SPEC_MAP")
    key = next((k for k in results if target in k), None)
    if key is None:
        raise SystemExit(f"no '{target}' entry in the pickle (keys: {list(results)})")
    return key


def _demand_parquet(est: int, spec: int) -> Path:
    p = _paths.demand_parquet_dir() / f"{_rt.demand_prefix(est)}_spec_{spec}.parquet"
    if not p.is_file():
        raise SystemExit(f"demand parquet not found: {p}")
    return p


def _index_sequential(df: pd.DataFrame, terms: list) -> np.ndarray:
    """The native index in the estimator's own summation order, as the demand prep builds it
    (sleep_demand_prep_link.process_specification: phi = 0; phi += beta * column)."""
    v = np.zeros(len(df))
    for t in terms:
        col = df[t["state"]].to_numpy(dtype=float)
        v = v + t["coef"] * col
    return v


def export_one(est: int, spec: int, out_dir: Path, transitions: Path, tol: float,
               check: bool = True) -> Path:
    pkl = _paths.est_dir(est) / "estimation_results.pkl"
    if not pkl.is_file():
        raise SystemExit(f"no sleepiness fit at {pkl}")
    with open(pkl, "rb") as fh:
        results = pickle.load(fh)
    key = _spec_key(results, spec)
    res = results[key]["second_stage"]
    link = getattr(res, "link", None)
    if link != "index_sieve":
        raise SystemExit(f"E{est} spec {spec}: link '{link}' -- only the constrained single-index "
                         f"link (index_sieve) is exported; phi_from_native is the reference for "
                         f"the others")

    trans = json.loads(transitions.read_text(encoding="utf-8"))
    market_states = trans.get("market_states", {})

    native = res.params_native
    terms = []
    for p, c in native.items():
        if str(p).startswith("v_hat"):
            continue
        state = "constant" if p == "nr_lagged_dep" else str(p).replace("interaction_", "")
        role = ROLE_OF.get(state)
        if role is None:
            ent = market_states.get(state)
            if not (isinstance(ent, dict) and isinstance(ent.get("rho"), (int, float))):
                raise SystemExit(
                    f"E{est}: state '{state}' has no market_states transition in "
                    f"{transitions.name}. Add it to bbl_transitions.py S_COLS and re-run "
                    f"`python bbl_transitions.py`, then this export.")
            role = "market"
        terms.append({"param": str(p), "state": state, "coef": float(c), "role": role})
    n_selic = sum(t["role"] == "selic" for t in terms)
    if n_selic != 1:
        raise SystemExit(f"E{est}: expected exactly one Selic state, found {n_selic}")

    tf = _st.load_transform()
    tf_path = _st.json_path()
    states = {}
    for t in terms:
        s = t["state"]
        states[s] = {
            "role": t["role"],
            "parquet_col": s,
            "scale": float(tf.scale.get(s, 1.0)),
            "mean_scaled": float(tf.mean_scaled.get(s, 0.0)) if s != "constant" else 0.0,
            "units": UNITS.get(s, ""),
        }
        if t["role"] == "market":
            states[s]["transition"] = {k: market_states[s].get(k) for k in
                                       ("rho", "rho_nickell_corrected", "half_life_quarters", "n_pairs")}

    vgrid = np.asarray(res.si_vgrid, dtype=float)
    ggrid = np.asarray(res.si_ggrid, dtype=float)
    if vgrid.shape != ggrid.shape or vgrid.size < 2 or not np.all(np.diff(vgrid) > 0):
        raise SystemExit(f"E{est}: si_vgrid must be strictly increasing and match si_ggrid")

    payload = {
        "schema": SCHEMA,
        "estim": int(est),
        "spec": int(spec),
        "spec_key": key,
        "routine": {"name": _rt.est_name(est), "time_block": bool(_rt.ROUTINES[est].time_block)},
        "index": {
            "terms": terms,
            "note": ("v = sum_k coef_k * S_k in THIS order (params_native order, the demand "
                     "prep's own summation order); S_k is the parquet column in estimation "
                     "units (scaled by `scale`, minus `mean_scaled`)."),
        },
        "link": {
            "kind": "index_sieve",
            "eval": ("phi = clip(interp(v, vgrid, ggrid), 0, 1) with numpy.interp semantics: "
                     "linear between grid points, flat beyond both ends"),
            "vgrid": _floats(vgrid),
            "ggrid": _floats(ggrid),
            "sim_clamp": [0.0, 0.999],
            "ispline": {
                "knots": _floats(res.si_knots),
                "beta": _floats(res.si_beta),
                "degree": int(res.si_degree),
                "vmu": float(res.si_vmu),
                "vsd": float(res.si_vsd),
                "note": ("ggrid = clip(ramp_design((vgrid - vmu)/vsd, knots, degree) @ beta, 0, 1); "
                         "beta >= 0 and sum(beta) <= 1 (monotone CDF). Provenance only: the grid "
                         "is what every consumer evaluates."),
            },
        },
        "states": states,
        "state_transform": {"version": tf.version, "built_at": tf.built_at,
                            **_file_meta(tf_path)},
    }

    held_time = [t for t in terms if t["role"] == "held" and t["state"] != "pix_exists"]
    df = None
    pq = None
    if check or held_time:
        pq = _demand_parquet(est, spec)
        cols = sorted({t["state"] for t in terms} | {"phi_mt", "time_id"})
        df = pd.read_parquet(pq, columns=cols)

    if held_time:
        tb = {}
        for t in held_time:
            contrib = (t["coef"] * df[t["state"]].astype(float)).groupby(df["time_id"].astype(str)).mean()
            tb[t["state"]] = {
                "coef": t["coef"],
                "rule": "held at each row's launch-quarter value for the whole horizon",
                "mean_contribution_by_quarter": {str(q): float(v) for q, v in contrib.items()},
            }
        payload["time_block"] = {
            "vars": tb,
            "note": ("The +Time routine's time component inside the index. The quarter fixed "
                     "effects of the two-way estimator are concentrated out additively, outside "
                     "the link, and are not part of phi."),
        }

    if check:
        v0 = _index_sequential(df, terms)
        phi0 = np.clip(np.interp(v0, vgrid, ggrid), 0.0, 1.0)
        phi_mt = df["phi_mt"].to_numpy(dtype=float)
        d = np.abs(phi0 - phi_mt)
        ref = phi_from_native(df, res, "index_sieve")
        d_ref = np.abs(ref - phi_mt)
        payload["t0_check"] = {
            "parquet": pq.name,
            "n_rows": int(len(df)),
            "max_abs_diff": float(np.nanmax(d)),
            "n_nonfinite": int((~np.isfinite(d)).sum()),
            "n_exact": int((d == 0.0).sum()),
            "max_abs_diff_phi_from_native": float(np.nanmax(d_ref)),
            "tol": float(tol),
            "pass": bool(np.all(np.isfinite(d)) and float(np.max(d)) <= tol),
            "phi_mt_mean": float(phi_mt.mean()),
        }
        payload["provenance_parquet"] = _file_meta(pq)

    payload["provenance"] = {
        "script": "bbl_sleep_link.py",
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pickle": _file_meta(pkl),
        "transitions": _file_meta(transitions),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"sleep_link_E{est}_spec_{spec}.json"
    tmp = out.with_name(f".tmp_{out.name}.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, out)

    tc = payload.get("t0_check")
    line = (f"  E{est} spec {spec}: {len(terms)} index terms "
            f"({', '.join(t['state'] + '=' + t['role'] for t in terms)}) | grid {vgrid.size} pts "
            f"v in [{vgrid[0]:.4g}, {vgrid[-1]:.4g}]")
    print(line)
    if tc:
        print(f"    t=0 check vs {tc['parquet']}: {tc['n_rows']:,} rows, max|phi0 - phi_mt| = "
              f"{tc['max_abs_diff']:.3e} ({tc['n_exact']:,} bit-identical; phi_from_native "
              f"{tc['max_abs_diff_phi_from_native']:.3e}) -> {'PASS' if tc['pass'] else 'FAIL'}")
    print(f"    wrote {out}  ({out.stat().st_size:,} bytes, sha256 {_sha256(out)[:16]}...)")
    if tc and not tc["pass"]:
        raise SystemExit(
            f"REFUSING: E{est}'s link does not reproduce phi_mt at t=0 "
            f"(max |diff| {tc['max_abs_diff']:.3e} > {tol:g}). The pickle and the demand parquet "
            f"are different vintages; re-run the demand prep from this fit, or export from the "
            f"fit the parquet was built with. The JSON was written for inspection only.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export the sleepiness link for the BBL forward sim.")
    ap.add_argument("--est", type=int, nargs="+", default=list(_rt.LINK_ESTS))
    ap.add_argument("--spec", type=int, default=_rt.SPEC12_ID)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: COST_FWD")
    ap.add_argument("--transitions", type=Path, default=None,
                    help="bbl_transitions.json (default: COST_FWD/bbl_transitions.json)")
    ap.add_argument("--tol", type=float, default=TOL_T0)
    ap.add_argument("--no-check", action="store_true",
                    help="skip the t=0 reproduction of the parquet's phi_mt")
    a = ap.parse_args(argv)
    out_dir = a.out_dir or _paths.cost_fwd_dir()
    trans = a.transitions or (_paths.cost_fwd_dir() / "bbl_transitions.json")
    if not trans.is_file():
        raise SystemExit(f"bbl_transitions.json not found: {trans} (python bbl_transitions.py)")
    print(f"bbl_sleep_link: est {a.est}, spec {a.spec} -> {out_dir}")
    for e in a.est:
        export_one(e, a.spec, out_dir, trans, a.tol, check=not a.no_check)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
