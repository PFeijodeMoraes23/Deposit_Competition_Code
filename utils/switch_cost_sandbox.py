"""Output tree of the switch-cost investigation (deposit-type sleepiness and switching costs).

Every ``switch_cost_*`` script imports this module before it resolves any path. It points
``SLEEP_OUT_ROOT`` at the investigation's own tree, so ``paths.demand_prep_root``,
``est_dir``, ``rout_dir`` and ``demand_parquet_dir`` all land there, and it stops the
process when that tree could be a production one.

Rule: the last component of ``SLEEP_OUT_ROOT`` (and of ``DEMAND_PREP_DIR`` when set) must
contain ``switch_cost``, in any case. Production folders (``ESTIMATION_OUTPUT/DEMAND_PREP``,
the cluster's ``data/output/sleep`` and ``data/output/demand_prep``) never do, so a missing
or inherited export fails here instead of overwriting them.

Two fit variants, chosen by ``SWITCH_COST_TYPE_SHIFTS``:

  0 (default)  pooled      the production specification, re-estimated by the copies;
                           the reference for every comparison and the source of phi_0.
  1            typeshift   the same specification plus deposit-type shifts in the phi index.

Each variant has its own tree, and the tree's name must say which one it is (``typeshift``
in the name exactly when the switch is on), so the two fits can never share a pickle.

Locally, with nothing exported, the tree is
``PROCESSED/ESTIMATION_OUTPUT/SWITCH_COST/switch_cost_<variant>``. On the cluster
(``CL_DATA_OUT`` set) nothing is defaulted: ``switch_cost_sleep_job.sh`` exports
``SLEEP_OUT_ROOT`` (and ``DEMAND_PREP_DIR``) before Python starts.

Inputs are unaffected: the market panel, ``demand_chars.parquet`` and the state-centering
JSON resolve through the other anchors of ``utils.paths`` and ``utils.state_transform``,
so a switch-cost run reads exactly what production reads.
"""
from __future__ import annotations

import os
from pathlib import Path

from utils import paths as _paths

TAG = "switch_cost"
LOCAL_BASE = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "SWITCH_COST"
# The deposit-type shifts: demand deposits (k=1) are the base type.
TYPE_DUMMIES = ["dtype_2", "dtype_4", "dtype_5"]


def type_shifts_on() -> bool:
    return os.environ.get("SWITCH_COST_TYPE_SHIFTS", "0").strip() == "1"


def fit_variant() -> str:
    return "typeshift" if type_shifts_on() else "pooled"


def local_fit_root(variant: str | None = None) -> Path:
    return LOCAL_BASE / f"{TAG}_{variant or fit_variant()}"


def check_dir(p: Path | str, what: str) -> Path:
    """Refuse a write destination whose name is not a switch-cost folder."""
    p = Path(p).expanduser()
    if TAG not in p.name.lower():
        raise SystemExit(
            f"[switch_cost_sandbox] {what}={p} is not a switch-cost folder (its name must "
            f"contain '{TAG}'). Refusing to run: this could overwrite production output.")
    return p


def activate() -> Path:
    """Set and validate the output tree; return it. Idempotent."""
    if not os.environ.get("SLEEP_OUT_ROOT", "").strip():
        if os.environ.get("CL_DATA_OUT", "").strip():
            raise SystemExit(
                "[switch_cost_sandbox] on the cluster SLEEP_OUT_ROOT must be exported by "
                "switch_cost_sleep_job.sh; it is unset. Refusing to run.")
        os.environ["SLEEP_OUT_ROOT"] = str(local_fit_root())
    root = check_dir(os.environ["SLEEP_OUT_ROOT"], "SLEEP_OUT_ROOT")
    if ("typeshift" in root.name.lower()) != type_shifts_on():
        raise SystemExit(
            f"[switch_cost_sandbox] SLEEP_OUT_ROOT={root} does not match "
            f"SWITCH_COST_TYPE_SHIFTS={int(type_shifts_on())}: the tree name must contain "
            "'typeshift' exactly when the type shifts are on. Refusing to mix the two fits.")
    if os.environ.get("DEMAND_PREP_DIR", "").strip():
        check_dir(os.environ["DEMAND_PREP_DIR"], "DEMAND_PREP_DIR")
    root.mkdir(parents=True, exist_ok=True)
    return root


ROOT = activate()

__all__ = ["TAG", "LOCAL_BASE", "TYPE_DUMMIES", "ROOT", "activate", "check_dir",
           "fit_variant", "local_fit_root", "type_shifts_on"]
