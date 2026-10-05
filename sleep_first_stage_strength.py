"""
sleep_first_stage_strength.py
================================================================================
Strength of the sleepiness first stage (the deposit-spread equation), for every routine and
every instrumented specification: the numbers behind the "Effective $F$" row of the
first-stage tables.

READS
  est{N}/estimation_results.pkl   the stored first-stage fits of every routine in
                                  config/routines.toml (utils.paths.est_dir)
  BLP_RESULTS/cluster_processed/weak_iv_sleep.json
                                  the weak-instrument battery's stored values, used only as
                                  the reference of the identity check below
WRITES
  Rout/sleep_first_stage_strength.json   (utils.paths.rout_dir)

Nothing is estimated. Each stored first stage is taken as it was run -- the spread, the
constant and state controls, the excluded instruments and the conglomerate clusters are read
off the fit object -- and handed to blp_weak_iv._fs_F, the function sleep_weak_iv.py uses for
its "pooled" first-stage strength. Per routine (E1, E2, ...) and specification
("IV_HausmanFull x Tech", ...) the file holds

  eff_F         effective F of the excluded instruments (Montiel Olea and Pflueger 2013),
                conglomerate-clustered, after partialling out the constant and the controls
  excl_F        cluster-robust joint F of the excluded instruments
  F_all_slopes  the fit's own F: the joint test of every slope, instruments and controls
                together (the "$F$, all slopes" row of the tables)
  mop_cv        effective-F critical values for a Nagar bias of 5/10/20/30% of the benchmark,
                TSLS, 5% level (blp_weak_iv._mop_cv)
  N, G, n_iv, n_controls, k_eff, partial_R2, instruments, controls

IDENTITY CHECK. weak_iv_sleep.json stores the same statistic for the headline state block of
the pooled design without the Time block, which is the first stage of the pooled routines
that carry no Time block. The values computed here for those routines must equal the stored
ones to 1e-9; otherwise the script stops before writing, because the fits and the stored
battery no longer describe the same regression.

The exporters read the file through `eff_f` below and print a blank cell when a value is
absent.

Run:  python sleep_first_stage_strength.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import os
import sys
import json
import pickle
import argparse
from datetime import datetime, timezone

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from utils import paths as _paths  # noqa: E402
from utils import routines as _routines  # noqa: E402

JSON_NAME = "sleep_first_stage_strength.json"
CHECK_TOL = 1e-9


def json_path():
    """Where the strength file lives: beside the other sleepiness sidecars, in Rout."""
    return _paths.rout_dir() / JSON_NAME


def routine_key(est) -> str:
    """Routine id -> its key in the file, e.g. 3 -> 'E3'."""
    return f"E{int(est)}"


# ---------------------------------------------------------------------------
# Reader (used by the table exporters)
# ---------------------------------------------------------------------------
_CACHE = None


def load() -> dict:
    """The stored strength file as a dict, or {} when it has not been written."""
    global _CACHE
    if _CACHE is None:
        p = json_path()
        if p.is_file():
            with open(p, "r", encoding="utf-8") as fh:
                _CACHE = json.load(fh)
        else:
            _CACHE = {}
    return _CACHE


def eff_f(est, spec_key):
    """Effective F of routine `est` under specification `spec_key`
    (e.g. 'IV_HausmanFull x Tech'), or None when the file holds no finite value for it."""
    try:
        cell = load().get(routine_key(est), {}).get(spec_key, {})
    except (TypeError, ValueError):       # `est` is not a routine id
        return None
    v = cell.get("eff_F") if isinstance(cell, dict) else None
    if isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
        return float(v)
    return None


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _cell(fs, iv_cols):
    """Strength of one stored first-stage fit, from the regression as it was run."""
    import numpy as np
    import blp_weak_iv as wiv

    names = list(fs.model.exog_names)
    exog = np.asarray(fs.model.exog, float)
    spread = np.asarray(fs.model.endog, float)
    ivs = [c for c in iv_cols if c in names]
    controls = [c for c in names if c not in ivs and c != "const"]
    if not ivs:
        return None
    X = np.column_stack([np.ones(len(spread))] + [exog[:, names.index(c)] for c in controls])
    Z = np.column_stack([exog[:, names.index(c)] for c in ivs])
    clusters = np.asarray(fs.cov_kwds["groups"]).astype(str)
    if len(clusters) != len(spread):
        raise SystemExit(f"cluster vector ({len(clusters):,}) and first-stage sample "
                         f"({len(spread):,}) differ in length")
    uniq, codes = np.unique(clusters, return_inverse=True)
    r = wiv._fs_F(spread, X, Z, codes, int(len(uniq)))
    return {
        "eff_F": r["eff_F"],
        "excl_F": r["kp_F"],
        "F_all_slopes": float(fs.fvalue),
        "mop_cv": r["mop_cv"],
        "N": int(len(spread)),
        "G": int(len(uniq)),
        "n_iv": int(r["n_iv"]),
        "n_controls": len(controls),
        "k_eff": r["k_eff"],
        "partial_R2": r["partial_R2"],
        "instruments": ivs,
        "controls": controls,
    }


def _reference_ests():
    """Routines whose first stage is the regression weak_iv_sleep.json describes: the pooled
    frame (not the local one), without the Time block."""
    return [e for e in _routines.ACTIVE
            if _routines.ROUTINES[e].kind != "local_linear"
            and not _routines.ROUTINES[e].time_block]


def _identity_check(out):
    """Compare the cells weak_iv_sleep.json also stores; stop unless every one agrees."""
    ref_path = _paths.blp_results_dir() / "cluster_processed" / "weak_iv_sleep.json"
    if not ref_path.is_file():
        raise SystemExit(f"Identity check impossible: {ref_path} not found.")
    with open(ref_path, "r", encoding="utf-8") as fh:
        ref = json.load(fh)
    state = str(ref.get("_meta", {}).get("headline_spec", _routines.SPEC12)).split(" x ")[-1]
    cells, worst = [], 0.0
    for iv_name, block in ref.items():
        stored = (block.get("type45", {}).get("first_stage_strength", {}).get("pooled", {})
                  .get("eff_F") if isinstance(block, dict) else None)
        if stored is None:
            continue
        spec_key = f"{iv_name} x {state}"
        for est in _reference_ests():
            mine = out.get(routine_key(est), {}).get(spec_key, {}).get("eff_F")
            if mine is None:
                raise SystemExit(f"Identity check: no computed value for {routine_key(est)} "
                                 f"[{spec_key}] to set against {ref_path.name}.")
            diff = abs(mine - stored)
            worst = max(worst, diff)
            cells.append({"routine": routine_key(est), "spec": spec_key,
                          "stored": stored, "computed": mine, "abs_diff": diff})
    if not cells:
        raise SystemExit(f"Identity check: {ref_path} holds no pooled effective F to compare.")
    if worst > CHECK_TOL:
        bad = [c for c in cells if c["abs_diff"] > CHECK_TOL]
        raise SystemExit("Identity check FAILED (tolerance %g): %s" % (CHECK_TOL, json.dumps(bad)))
    return {"reference": str(ref_path), "reference_path_in_file":
            "[<IV set>]['type45']['first_stage_strength']['pooled']['eff_F']",
            "tolerance": CHECK_TOL, "max_abs_diff": worst, "passed": True, "cells": cells}


def main():
    argparse.ArgumentParser(
        description="First-stage strength of every sleepiness routine and specification, "
                    "from the stored fits (writes Rout/%s)." % JSON_NAME).parse_args()

    # The instrument sets are named in the estimation code; read them there.
    from sleep_est_e2 import define_specifications
    _, iv_specs, _ = define_specifications()

    out, inputs = {}, {}
    for est in _routines.ACTIVE:
        pkl = _paths.est_dir(est) / "estimation_results.pkl"
        # A missing pickle is a hard failure: a file that silently lacks a routine would
        # print blank cells in that routine's table.
        if not pkl.is_file():
            raise SystemExit(f"Results not found at {pkl}.")
        with open(pkl, "rb") as fh:
            results = pickle.load(fh)
        st = pkl.stat()
        inputs[routine_key(est)] = {
            "path": str(pkl),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                                 .isoformat(timespec="seconds"),
            "size_bytes": st.st_size,
        }
        cells = {}
        for spec_key, entry in results.items():
            fs = entry.get("first_stage") if isinstance(entry, dict) else None
            if fs is None:
                continue
            c = _cell(fs, iv_specs.get(spec_key.split(" x ")[0], []))
            if c is not None:
                cells[spec_key] = c
        out[routine_key(est)] = cells
        del results
        print(f"{routine_key(est)} ({_routines.est_label(est)}): {len(cells)} specification(s)")
        for spec_key, c in cells.items():
            print(f"    {spec_key:26s} N={c['N']:,} G={c['G']} K={c['n_iv']} | "
                  f"effective F {c['eff_F']:.3f} | excluded-instrument F {c['excl_F']:.3f} | "
                  f"F, all slopes {c['F_all_slopes']:.3f}")

    check = _identity_check(out)
    print(f"Identity check against {os.path.basename(check['reference'])}: "
          f"{len(check['cells'])} cells, max |diff| = {check['max_abs_diff']:.3e} (passed)")

    payload = {"_meta": {
        "written_by": os.path.basename(__file__),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "statistic": "blp_weak_iv._fs_F on each stored first-stage regression as it was run "
                     "(constant and state controls partialled out; no fixed effects)",
        "eff_F": "Montiel Olea-Pflueger (2013) effective F of the excluded instruments, "
                 "clustered by CodConglomeradoPrudencial",
        "excl_F": "cluster-robust joint F of the excluded instruments",
        "F_all_slopes": "the fit's fvalue: joint F of every slope, instruments and controls",
        "mop_cv": "effective-F critical values, Nagar bias of TSLS at 5/10/20/30% of the "
                  "benchmark, 5% level (blp_weak_iv._mop_cv)",
        "inputs": inputs,
        "identity_check": check,
    }}
    payload.update(out)
    path = json_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
