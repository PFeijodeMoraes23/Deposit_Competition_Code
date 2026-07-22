"""
utils/br_calendar.py
====================
Minimal Brazilian business-day calendar — no external dependencies beyond numpy.

Why this exists
---------------
Open Finance Brasil open-data fee values are published on the **10th business day**
("décimo dia útil") of each month (BCB Instrução Normativa 32/2020, art. 6), NOT the
10th calendar day. The 10th business day lands anywhere from ~calendar day 12 to 16
depending on weekends and national holidays, so a fixed calendar-15 scrape is
borderline. This module computes the nth business day of a month (for scheduling the
collector safely after the update) and the business-day index of an arbitrary date
(for the empirical schedule verifier, diag_openfinance_freshness.py, which reports on
which business day each bank's values actually changed).

Holidays covered: Brazilian NATIONAL holidays (the ANBIMA banking-calendar basis),
including the Easter-derived movable feasts (Good Friday, Carnival Mon/Tue, Corpus
Christi). State/municipal holidays are intentionally out of scope — the OFB publication
deadline is national.

API
---
    from utils.br_calendar import nth_business_day, business_day_index, is_business_day
    nth_business_day(2026, 7, 10)   -> datetime.date  (10th business day of Jul 2026)
    business_day_index(date(2026,7,15)) -> int         (which business day of its month)
    is_business_day(date(2026,7,15)) -> bool
"""

from __future__ import annotations

import datetime as _dt
from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------------
# Easter (Anonymous Gregorian algorithm) and the movable feasts
# ---------------------------------------------------------------------------

def _easter(year: int) -> _dt.date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return _dt.date(year, month, day)


@lru_cache(maxsize=None)
def national_holidays(year: int) -> frozenset[_dt.date]:
    """Brazilian national (banking-calendar) holidays for `year`."""
    easter = _easter(year)
    fixed = [
        _dt.date(year, 1, 1),    # Confraternização Universal
        _dt.date(year, 4, 21),   # Tiradentes
        _dt.date(year, 5, 1),    # Dia do Trabalho
        _dt.date(year, 9, 7),    # Independência
        _dt.date(year, 10, 12),  # Nossa Senhora Aparecida
        _dt.date(year, 11, 2),   # Finados
        _dt.date(year, 11, 15),  # Proclamação da República
        _dt.date(year, 12, 25),  # Natal
    ]
    # Consciência Negra — national holiday from 2024 (Lei 14.759/2023).
    if year >= 2024:
        fixed.append(_dt.date(year, 11, 20))
    movable = [
        easter - _dt.timedelta(days=48),  # Carnival Monday
        easter - _dt.timedelta(days=47),  # Carnival Tuesday
        easter - _dt.timedelta(days=2),   # Good Friday (Sexta-feira Santa)
        easter + _dt.timedelta(days=60),  # Corpus Christi
    ]
    return frozenset(fixed + movable)


def _holidays_array(*years: int) -> np.ndarray:
    hs: set[_dt.date] = set()
    for y in years:
        hs |= set(national_holidays(y))
    return np.array(sorted(str(h) for h in hs), dtype="datetime64[D]")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_business_day(d: _dt.date) -> bool:
    arr = _holidays_array(d.year)
    return bool(np.is_busday(np.datetime64(str(d)), holidays=arr))


def nth_business_day(year: int, month: int, n: int = 10) -> _dt.date:
    """Return the date of the n-th business day of (year, month)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    hol = _holidays_array(year - 1, year, year + 1)
    first = np.datetime64(f"{year:04d}-{month:02d}-01")
    # busday_offset with roll='forward' lands on the 1st business day for offset 0.
    result = np.busday_offset(first, n - 1, roll="forward", holidays=hol)
    return result.astype("O")


def business_day_index(d: _dt.date) -> int:
    """
    Which business day of its own month `d` is (1-based). Returns 0 if `d` is not
    itself a business day (weekend/holiday).
    """
    if not is_business_day(d):
        return 0
    hol = _holidays_array(d.year - 1, d.year, d.year + 1)
    month_start = np.datetime64(f"{d.year:04d}-{d.month:02d}-01")
    # Count business days in [month_start, d] inclusive.
    return int(np.busday_count(month_start, np.datetime64(str(d)), holidays=hol)) + 1


if __name__ == "__main__":
    # Quick self-check / reference table.
    import calendar
    for (y, m) in [(2026, 7), (2026, 5), (2026, 1), (2025, 11)]:
        bd10 = nth_business_day(y, m, 10)
        print(f"{y}-{m:02d}: 10th business day = {bd10} "
              f"(calendar day {bd10.day}, {calendar.day_name[bd10.weekday()]})")
    for ds in ["2026-07-15", "2026-07-18", "2026-07-20"]:
        d = _dt.date.fromisoformat(ds)
        print(f"{ds}: business_day_index = {business_day_index(d)}, "
              f"is_business_day = {is_business_day(d)}")
