"""utils/blp_status.py — whether the random-coefficients (BLP) results are final.

One switch, config/blp_status.toml, for make_paper_numbers.py (the status of every BLP-derived
entry) and the random-coefficients table generators (a provisional sentence in their notes, worded
in config/table_notes.toml). See the config file for what each value does. A missing config file
reads as pending: releasing the BLP numbers takes an explicit "ok".
"""
from __future__ import annotations

import tomllib
from pathlib import Path

PENDING = "pending-blp-ext1"
STATUSES = ("ok", PENDING)
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "blp_status.toml"
TABLE_NOTES = ROOT / "config" / "table_notes.toml"


def status() -> str:
    """The status in config/blp_status.toml: "ok" or "pending-blp-ext1"."""
    if not CONFIG.is_file():
        return PENDING
    with open(CONFIG, "rb") as f:
        s = tomllib.load(f).get("status")
    if s not in STATUSES:
        raise SystemExit(f"{CONFIG}: status {s!r} is not one of {STATUSES}")
    return s


def pending() -> bool:
    """True while the random-coefficients results are provisional."""
    return status() != "ok"


def provisional_note() -> str:
    """The `provisional` sentence of config/table_notes.toml [demand] while pending, else ""."""
    if not pending():
        return ""
    with open(TABLE_NOTES, "rb") as f:
        return tomllib.load(f)["demand"]["provisional"]
