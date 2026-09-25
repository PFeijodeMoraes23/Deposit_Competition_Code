"""
utils/bbl_discount.py
=====================
Reader for bbl_discount.env, the one file that sets the BBL discount factor and the
forward-simulation horizon. Standard library only, so the curve builder, the table generator and
the checks can all import it without pulling in pandas or numpy.

The file is plain KEY=VALUE so bash can `source` it. This parser accepts exactly that subset and
nothing looser: blank lines and `#` comment lines are skipped, every other line must be
KEY=VALUE with a shell identifier as the key and no spaces around `=`, and a value may be wrapped
in one pair of single or double quotes. Anything else raises, because a line that bash and Python
read differently is a silent disagreement about beta.

    read(path=None)        -> {KEY: VALUE} as strings, in file order
    beta(path=None)        -> BBL_BETA as float
    horizon(path=None)     -> BBL_HORIZON as int
    rf_mean_q(path=None)   -> BBL_RF_MEAN_Q as float, or None when the file does not carry it
    rf_window(path=None)   -> BBL_RF_WINDOW as (first, last) quarter labels, or None
    annual(b)              -> b**4, the annual equivalent of a quarterly discount factor
    coverage(b, T)         -> 1 - b**T, the share of the infinite-horizon discount weight
                              sum_{t>=1} b^t that the first T quarters carry
    min_horizon(b, share)  -> the shortest T with coverage(b, T) >= share
    quarters_in(first, last) -> number of quarters from `first` to `last` inclusive

BBL_BETA is not re-derived here from BBL_RF_MEAN_Q: the registry states the value the runs use,
and the mean rate is recorded next to it as its documented derivation (1 / (1 + mean), rounded).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / "bbl_discount.env"

_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_QUARTER = re.compile(r"(\d{4})Q([1-4])\Z")


def read(path: str | Path | None = None) -> dict[str, str]:
    """Parse the registry into {KEY: VALUE}. Raises ValueError on a line bash would not read the
    same way, and FileNotFoundError when the file is absent."""
    p = Path(path) if path is not None else ENV_PATH
    out: dict[str, str] = {}
    for i, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition("=")
        if not sep or not _KEY.match(key) or key != key.strip() or val != val.lstrip():
            raise ValueError(f"{p.name}:{i}: not a KEY=VALUE line: {raw!r}")
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        elif any(c in val for c in " \t'\"#$`\\;"):
            raise ValueError(f"{p.name}:{i}: unquoted value with shell metacharacters: {raw!r}")
        if key in out:
            raise ValueError(f"{p.name}:{i}: {key} is set twice")
        out[key] = val
    return out


def _get(key: str, path=None) -> str:
    d = read(path)
    if key not in d:
        raise KeyError(f"{key} missing from {Path(path) if path is not None else ENV_PATH}")
    return d[key]


def beta(path=None) -> float:
    b = float(_get("BBL_BETA", path))
    if not 0.0 < b < 1.0:
        raise ValueError(f"BBL_BETA={b} is not a discount factor in (0, 1)")
    return b


def horizon(path=None) -> int:
    t = int(_get("BBL_HORIZON", path))
    if t < 1:
        raise ValueError(f"BBL_HORIZON={t} must be a positive number of quarters")
    return t


def rf_mean_q(path=None) -> float | None:
    v = read(path).get("BBL_RF_MEAN_Q")
    return None if v is None else float(v)


def rf_window(path=None) -> tuple[str, str] | None:
    v = read(path).get("BBL_RF_WINDOW")
    if v is None:
        return None
    first, sep, last = v.partition("-")
    if not sep or not _QUARTER.match(first) or not _QUARTER.match(last):
        raise ValueError(f"BBL_RF_WINDOW={v!r} is not YYYYQn-YYYYQn")
    return first, last


def quarters_in(first: str, last: str) -> int:
    (y0, q0), (y1, q1) = ((int(m.group(1)), int(m.group(2)))
                          for m in (_QUARTER.match(first), _QUARTER.match(last)))
    n = (y1 * 4 + q1) - (y0 * 4 + q0) + 1
    if n < 1:
        raise ValueError(f"window {first}-{last} is empty")
    return n


def annual(b: float) -> float:
    return float(b) ** 4


def coverage(b: float, T: int) -> float:
    return 1.0 - float(b) ** int(T)


def min_horizon(b: float, share: float = 0.995) -> int:
    """Shortest T with 1 - b**T >= share. Computed in closed form, then nudged by one step either
    way so floating-point error in the logarithm cannot move the answer."""
    t = max(1, math.ceil(math.log(1.0 - share) / math.log(float(b))))
    while t > 1 and coverage(b, t - 1) >= share:
        t -= 1
    while coverage(b, t) < share:
        t += 1
    return t
