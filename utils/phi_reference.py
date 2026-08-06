"""utils/phi_reference.py -- self-refreshing phi constants for the diagnostics.

Author: Pedro Feijo de Moraes

THE PROBLEM THIS SOLVES. Several diagnostics need a phi-hat they did not themselves
estimate: D10 draws the model accumulation curve at the linear spec-12 level, D11 compares
a flow rate against the awake margin 1-phi, D8 simulates its band at the production phi.
Each of those started life as a literal in the script (`PHI_E2_AVG = 0.918`), which is
correct on the day it is written and silently wrong after the next re-estimation -- and
wrong in the worst way, because the figure still builds and still looks reasonable while
comparing new data against an old model.

THE RULE HERE. Never a bare literal. Every value is READ from the estimation artefact that
owns it; on every successful read the value is written back to a small JSON cache with its
source and timestamp. If the artefact is missing or mid-write, the cache supplies the LAST
KNOWN GOOD value together with its age, and the caller is told loudly. The frozen literals
survive only as a third-tier seed for a fresh clone, and they announce themselves as such.

So the "hard-coded" numbers now maintain themselves: any run that can see the estimation
output refreshes them, and any run that cannot says exactly how stale what it used is.

Cache: <PROCESSED>/ESTIMATION_OUTPUT/DIAG_PHI_SEPARATION/phi_reference_cache.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from utils import paths as _paths

_OUT = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_PHI_SEPARATION"
CACHE = _OUT / "phi_reference_cache.json"

# Third-tier seeds ONLY: used when neither the estimation artefact nor the cache exists
# (a fresh clone). Values as of 2026-08-03; every one of them is superseded the first time
# a real read succeeds. Do not "update" these by hand -- fix the read instead.
_SEED = {
    "phi_e2_avg": 0.918,
    "phi_k": {"1": 0.7244, "2": 0.8504, "4": 0.9093, "5": 0.9687},
}

STALE_DAYS = 14      # a cached value older than this is called out, not just reported


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return {}


def _save(key, value, source):
    """Persist a freshly-read value. Best-effort: a locked file must never break a run."""
    d = _load_cache()
    d[key] = {"value": value, "source": source, "stamp": time.time(),
              "stamp_h": time.strftime("%Y-%m-%d %H:%M")}
    try:
        _OUT.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except OSError:
        pass
    return value


def _fallback(key, label):
    """Last known good from the cache, else the seed. Reports age either way."""
    ent = _load_cache().get(key)
    if ent is not None:
        age = (time.time() - float(ent.get("stamp", 0))) / 86400.0
        flag = "  <-- STALE" if age > STALE_DAYS else ""
        print(f"  [phi_ref] {label}: using LAST KNOWN GOOD from "
              f"{ent.get('stamp_h', '?')} ({age:.1f} days old){flag}")
        return ent["value"]
    print(f"  [phi_ref] {label}: no cache -- using the frozen seed (fresh clone?); "
          "re-run the estimator that owns this value")
    return _SEED[key]


def phi_e2_avg(verbose=True):
    """E2 spec-12 phi-hat at the AVERAGE MARKET.

    Source of truth: est2/estimation_results.pkl, spec-12 second stage, coefficient on
    `nr_lagged_dep`. The state block is grand-mean centred (utils/state_transform.py), so
    that coefficient IS phi at the average market -- no reconstruction needed.
    """
    pkl = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "est2" / "estimation_results.pkl"
    if pkl.exists():
        try:
            import gc
            import pickle
            with open(pkl, "rb") as fh:
                d = pickle.load(fh)
            entry = d.get("IV_HausmanFull x Tech")
            res = entry.get("second_stage") if isinstance(entry, dict) else entry
            v = float(pd.Series(res.params)["nr_lagged_dep"])
            del d, entry, res
            gc.collect()
            if not 0.0 < v < 1.5:
                raise ValueError(f"implausible phi-hat {v}")
            ts = pd.Timestamp(pkl.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
            if verbose:
                print(f"  [phi_ref] phi_e2_avg = {v:.4f}  <- est2 spec-12 fit [{ts}]")
            return _save("phi_e2_avg", v, f"est2 spec-12 nr_lagged_dep [{ts}]")
        except Exception as e:                                     # noqa: BLE001
            print(f"  [phi_ref] phi_e2_avg: read failed ({type(e).__name__}: {e})")
    return _fallback("phi_e2_avg", "phi_e2_avg")


def phi_k(verbose=True):
    """Type-specific carries {k: phi_k} from D5's output (OLSxTech covers all four types).

    Source of truth: DIAG_PHI_SEPARATION/d_interactions_types.csv, written by
    diag_phi_interaction_tests.py --arm types.
    """
    fp = _OUT / "d_interactions_types.csv"
    if fp.exists():
        try:
            t = pd.read_csv(fp)
            t = t[t["spec"].astype(str).str.startswith("OLSxTech")]
            out = {}
            base = t[t["param"].astype(str).str.fullmatch(r"phi_k\d")]
            if len(base):
                out[int(str(base.iloc[0]["param"])[-1])] = float(base.iloc[0]["coef"])
            for _, r in t[t["param"].astype(str).str.startswith("Zdiff_k")].iterrows():
                if pd.notna(r.get("phi_level")):
                    out[int(str(r["param"])[-1])] = float(r["phi_level"])
            if len(out) >= 2:
                ts = pd.Timestamp(fp.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
                if verbose:
                    print("  [phi_ref] phi_k = "
                          + ", ".join(f"k{k}:{v:.4f}" for k, v in sorted(out.items()))
                          + f"  <- {fp.name} [{ts}]")
                return {int(k): v for k, v in
                        _save("phi_k", {str(k): v for k, v in out.items()},
                              f"{fp.name} [{ts}]").items()}
            raise ValueError(f"only {len(out)} types parsed")
        except Exception as e:                                     # noqa: BLE001
            print(f"  [phi_ref] phi_k: read failed ({type(e).__name__}: {e})")
    return {int(k): v for k, v in _fallback("phi_k", "phi_k").items()}


if __name__ == "__main__":
    print("=== phi reference (reads source of truth, refreshes the cache) ===")
    phi_e2_avg()
    phi_k()
    print(f"\ncache -> {CACHE}")
