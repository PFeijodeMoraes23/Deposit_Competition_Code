"""utils/refresh.py — rolling-refresh window for the period-keyed scrapers.

Author: Pedro Feijó de Moraes

Government agencies (BCB, IBGE, ANATEL, MDS/SAGI, …) routinely *revise* recently
published periods after the fact — a month/quarter that was downloaded last run
may be restated this run.  A pure skip-if-exists guard would keep the stale copy
forever.

The fix used across the pipeline: always re-download the trailing **~1 year** of
data even when a local copy exists, since revisions almost always land in the
most recent periods.  Older periods are still skipped if present (they are
effectively final).

Granularity follows the data source:
    monthly series   -> last 12 months   (REFRESH_MONTHS)
    quarterly series -> last 4 quarters   (REFRESH_QUARTERS)
    annual series    -> current + last year (REFRESH_YEARS)

Usage in a download loop (replace ``if os.path.exists(fp): continue``):

    from utils import refresh
    ...
    if os.path.exists(fp) and not refresh.is_recent_month(year, month):
        continue            # cached AND final -> skip
    # else (missing, or within the rolling window) -> (re)download

This is always-on (the project default); there is no flag to disable it.
"""
from __future__ import annotations

import datetime as _dt

# Trailing window sizes ("the past year" per granularity).
REFRESH_MONTHS = 12     # monthly series re-fetch the last 12 months
REFRESH_QUARTERS = 4    # quarterly series re-fetch the last 4 quarters
REFRESH_YEARS = 1       # annual series re-fetch the current year + previous year


def _today(today: _dt.date | None = None) -> _dt.date:
    return today if today is not None else _dt.date.today()


def is_recent_month(year: int, month: int, *, today: _dt.date | None = None) -> bool:
    """True if (year, month) falls in the trailing REFRESH_MONTHS window
    (current month and the 11 before it) -> should be re-downloaded even if cached.
    Future periods also return True (treat as not-yet-final)."""
    t = _today(today)
    months_ago = (t.year - int(year)) * 12 + (t.month - int(month))
    return months_ago < REFRESH_MONTHS


def is_recent_quarter(year: int, quarter: int, *, today: _dt.date | None = None) -> bool:
    """True if (year, quarter) falls in the trailing REFRESH_QUARTERS window."""
    t = _today(today)
    tq = (t.month - 1) // 3 + 1
    quarters_ago = (t.year - int(year)) * 4 + (tq - int(quarter))
    return quarters_ago < REFRESH_QUARTERS


def is_recent_year(year: int, *, today: _dt.date | None = None) -> bool:
    """True if `year` is the current year or within REFRESH_YEARS before it."""
    t = _today(today)
    return (t.year - int(year)) <= REFRESH_YEARS


def should_download(path_exists: bool, *, recent: bool) -> bool:
    """Convenience: download when the file is missing OR within the rolling window."""
    return (not path_exists) or recent


__all__ = [
    "REFRESH_MONTHS", "REFRESH_QUARTERS", "REFRESH_YEARS",
    "is_recent_month", "is_recent_quarter", "is_recent_year", "should_download",
]
