"""utils/paper.py — the paper the generators read: V_Full or V_Friction.

One choice, set in config/paper.toml, for every script that reads the paper's .tex or .aux
(make_paper_equations.py, make_paper_numbers.py, diagnostics/report_entry_paths_estban.py).
Precedence: an explicit name (a script's --paper flag), then the DEPOSIT_PAPER environment
variable, then the file. Both papers live in paths.drafts_dir() and share every generated
exhibit, so this decides what is read, never where anything is written.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

from utils import paths

PAPERS = ("V_Full", "V_Friction")
CONFIG = Path(__file__).resolve().parents[1] / "config" / "paper.toml"
ENV = "DEPOSIT_PAPER"


def current(name: str | None = None) -> str:
    """The chosen paper's stem: `name` if given, else $DEPOSIT_PAPER, else config/paper.toml."""
    if name is None:
        name = os.environ.get(ENV) or None
    if name is None:
        with open(CONFIG, "rb") as f:
            name = tomllib.load(f).get("paper")
    name = str(name).removesuffix(".tex")
    if name not in PAPERS:
        raise SystemExit(f"paper {name!r} is not one of {PAPERS} (config/paper.toml, ${ENV}, --paper)")
    return name


def tex(name: str | None = None) -> Path:
    """Drafts/Deposit Competition/<paper>.tex."""
    return paths.drafts_dir() / f"{current(name)}.tex"


def aux(name: str | None = None) -> Path:
    """Drafts/Deposit Competition/<paper>.aux, written by the paper's last compile."""
    return paths.drafts_dir() / f"{current(name)}.aux"
