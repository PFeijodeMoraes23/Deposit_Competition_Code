"""utils/blp_status.py — whether the random-coefficients (BLP) results are final, per routine.

One switch, config/blp_status.toml, for make_paper_numbers.py (the status of every BLP-derived
entry) and the random-coefficients table generators (a provisional sentence in their notes, worded
in config/table_notes.toml). See the config file for what each value does. A missing config file
reads as pending: releasing the BLP numbers takes an explicit "ok".

The file holds a default `status` and, under [routines], one status per routine (E3 = ..., E4 =
...); a routine not listed takes the default. A routine's fits are released by setting its own
entry to "ok" while another routine stays pending. Asked without a routine (a table or entry that
spans routines), the answer is pending while ANY routine, or the default, is pending.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

PENDING = "pending-blp-ext1"
STATUSES = ("ok", PENDING)
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "blp_status.toml"
TABLE_NOTES = ROOT / "config" / "table_notes.toml"


def _config() -> tuple[str, dict]:
    """(default status, {"E3": status, ...}) from config/blp_status.toml, each value checked."""
    if not CONFIG.is_file():
        return PENDING, {}
    with open(CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    default, per = cfg.get("status"), dict(cfg.get("routines", {}))
    for name, s in [("status", default)] + [(f"routines.{k}", v) for k, v in per.items()]:
        if s not in STATUSES:
            raise SystemExit(f"{CONFIG}: {name} {s!r} is not one of {STATUSES}")
    return default, per


def _key(routine) -> str:
    """3, "3" or "E3" -> "E3"."""
    r = str(routine).strip().upper()
    return r if r.startswith("E") else f"E{r}"


def status(routine=None) -> str:
    """The status of `routine` ("ok" or "pending-blp-ext1"): its [routines] entry, else the
    default. Without a routine: pending while the default or any routine entry is pending."""
    default, per = _config()
    if routine is None:
        return "ok" if default == "ok" and all(s == "ok" for s in per.values()) else PENDING
    return per.get(_key(routine), default)


def pending(routine=None) -> bool:
    """True while the random-coefficients results of `routine` (None: any routine) are provisional."""
    return status(routine) != "ok"


def provisional_note(routine=None) -> str:
    """The `provisional` sentence of config/table_notes.toml [demand] while `routine` (None: any
    routine) is pending, else ""."""
    if not pending(routine):
        return ""
    with open(TABLE_NOTES, "rb") as f:
        return tomllib.load(f)["demand"]["provisional"]
