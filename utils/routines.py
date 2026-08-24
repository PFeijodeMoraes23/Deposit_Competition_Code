"""utils/routines.py — the Python view of ``config/routines.toml``.

Author: Pedro Feijó de Moraes

The estimator lineup, the ``\\ref{estimation:*}`` map, the headline spec key and the
panel's time-FE column all live in ``config/routines.toml``; this module parses it once and
hands the pipeline the shapes it already uses (``ESTIMATION_REF`` as an id -> ref dict,
``LINK_ESTS`` as a list, ``est_ref``/``est_label`` as lookups with an ``E{id}`` fallback).
``routines.jl`` is the Julia view of the same file, so the two languages cannot disagree
about which routines exist.

Import it rather than restating a lineup literal::

    from utils import routines as rt

    for est in rt.ACTIVE:            ...   # the reported lineup, in order
    for est in rt.active_from_env(): ...   # narrowed by SLEEP_ACTIVE_ESTS
    caption = rt.est_ref(est)              # \\ref{estimation:single_idx}

Parsing is with the 3.11 stdlib ``tomllib``, so nothing is added to requirements.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

# utils/routines.py -> utils -> Egan_et_al_2025_Rep
_REPO = Path(__file__).resolve().parents[1]
TOML_PATH: Path = _REPO / "config" / "routines.toml"


@dataclass(frozen=True)
class Routine:
    """One row of the lineup. Field meanings are documented in config/routines.toml."""
    id: int
    slug: str
    name: str
    tex_name: str
    diag_label: str
    tex_ref: str
    demand_prefix: str
    time_block: bool
    kind: str


def _load() -> dict:
    """Parse the registry, naming the fix when the file is not where it should be.

    On the cluster this module lands at ``scripts/utils/routines.py`` and looks for
    ``scripts/config/routines.toml`` -- the same copy ``routines.jl`` reads. It gets there
    only if the upload manifest carries the file, so a missing registry means the manifest
    has not been updated, not that a lineup constant should be pasted back in here.
    """
    if not TOML_PATH.is_file():
        raise FileNotFoundError(
            f"The routine registry is missing: {TOML_PATH}. Every consumer of the lineup "
            "reads it, so nothing runs without it. If this is a cluster run, add "
            "config/routines.toml to the upload manifest with destination scripts/config "
            "(routines.jl reads the same copy).")
    with open(TOML_PATH, "rb") as fh:
        return tomllib.load(fh)


_CFG = _load()

#: id -> :class:`Routine`, in the order the toml lists them.
ROUTINES: dict[int, Routine] = {
    int(r["id"]): Routine(
        id=int(r["id"]), slug=r["slug"], name=r["name"], tex_name=r["tex_name"],
        diag_label=r["diag_label"], tex_ref=r["tex_ref"],
        demand_prefix=r["demand_prefix"], time_block=bool(r["time_block"]),
        kind=r["kind"],
    )
    for r in _CFG["routines"]
}

#: The reported lineup, in order.
ACTIVE: list[int] = [int(e) for e in _CFG["active"]]

#: The routines estimated through the shared single-index link machinery. Also the default
#: routine set for the cluster BLP/BBL runs and the tables built from their artifacts.
LINK_ESTS: list[int] = [int(e) for e in _CFG["link_ests"]]

#: The routines carrying the Time block.
TIME_ESTS: list[int] = [e for e in ACTIVE if ROUTINES[e].time_block]

#: id -> the V_Main enumerate \ref, for call sites that want the whole map.
ESTIMATION_REF: dict[int, str] = {e: r.tex_ref for e, r in ROUTINES.items()}

SPEC12: str = _CFG["spec12"]
SPEC12_TAG: str = _CFG["spec12_tag"]
SPEC12_ID: int = int(_CFG["spec12_id"])
FE_TIME_COL: str = _CFG["fe_time_col"]

#: The SLEEP_ACTIVE_ESTS default, as the env var itself spells it.
ACTIVE_ENV_DEFAULT: str = " ".join(str(e) for e in ACTIVE)

_ROMAN = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))


def active_from_env() -> list[int]:
    """The active lineup, narrowed by ``SLEEP_ACTIVE_ESTS``.

    The env var is a space-separated id list ("3 4"); unset means the whole lineup. Set it
    to the empty string and you get an empty lineup, which is how a caller asks for nothing
    -- the demand-prep and export orchestrators both treat that as "run no routine" rather
    than falling back to the default.
    """
    return [int(x) for x in os.environ.get("SLEEP_ACTIVE_ESTS", ACTIVE_ENV_DEFAULT).split()]


def est_ref(est: int) -> str:
    """Estimator id -> V_Main enumerate ``\\ref`` (fallback ``E{id}`` for an id with no live label)."""
    r = ROUTINES.get(int(est))
    return r.tex_ref if r is not None else rf"E{est}"


def est_label(est: int) -> str:
    """Estimator id -> its description for logs, JSON metadata and cluster output."""
    r = ROUTINES.get(int(est))
    return r.name if r is not None else f"E{est}"


def est_name(est: int) -> str:
    """Estimator id -> its .tex column header."""
    r = ROUTINES.get(int(est))
    return r.tex_name if r is not None else f"E{est}"


def est_diag_label(est: int) -> str:
    """Estimator id -> its row label in the identification-battery tables."""
    r = ROUTINES.get(int(est))
    return r.diag_label if r is not None else f"E{est}"


def est_slug(est: int) -> str:
    """Estimator id -> the V_Main enumerate label, without the ``\\ref`` wrapper."""
    r = ROUTINES.get(int(est))
    return r.slug if r is not None else ""


def demand_prefix(est: int) -> str:
    """Estimator id -> demand parquet stem, completed as ``{prefix}_spec_{S}.parquet``."""
    r = ROUTINES.get(int(est))
    return r.demand_prefix if r is not None else f"demand_{est}"


def demand_tag(est: int) -> str:
    """The link tag carried inside the demand parquet stem ``demand_{est}_{tag}``.

    Empty for the linear routines, whose stem is just ``demand_{est}``. This is the same
    string the Julia routine discovery keys on, so it is read from the stem rather than
    stored twice."""
    stem, head = demand_prefix(est), f"demand_{int(est)}_"
    return stem[len(head):] if stem.startswith(head) else ""


def est_roman(est: int) -> str:
    """Estimator id -> its parenthesised Roman numeral, e.g. ``(III)``."""
    n, out = int(est), []
    for value, sym in _ROMAN:
        while n >= value:
            out.append(sym)
            n -= value
    return f"({''.join(out)})"


def id_choices(extra: tuple[str, ...] = ()) -> list[str]:
    """argparse ``choices`` over the lineup as strings, plus any extra tokens ('all')."""
    return [str(e) for e in ACTIVE] + list(extra)


def csv(ests: list[int]) -> str:
    """An id list as the comma-separated form the ``--routines`` flags take."""
    return ",".join(str(e) for e in ests)


__all__ = [
    "TOML_PATH", "Routine", "ROUTINES", "ACTIVE", "LINK_ESTS", "TIME_ESTS",
    "ESTIMATION_REF", "SPEC12", "SPEC12_TAG", "SPEC12_ID", "FE_TIME_COL",
    "ACTIVE_ENV_DEFAULT", "active_from_env", "est_ref", "est_label", "est_name",
    "est_diag_label", "est_slug", "demand_prefix", "est_roman", "id_choices", "csv",
]
