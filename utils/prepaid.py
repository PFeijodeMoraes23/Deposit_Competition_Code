"""prepaid.py

Type-5 (prepaid payment account) facts that every exhibit showing type 5 states, measured from
the market panel so the wording follows the data.

- Start. IF-Data reports the prepaid-account line (account 110560) from 2020Q2: `dep_a5` is NaN
  on every row before that quarter. Prepaid accounts predate 2020 (Lei 12.865/2013); the start
  is when the balance is reported, not when the product appeared.
- Rate. `panel_deposit_rates` sets `rate_a5` to the institution's payment-institution (IP) rate,
  else the quarter's median IP rate, else the CDI. IP rates enter the panel only from 2025, so
  inside the 2016-2024 window every institution carries the same fallback rate in every quarter
  and the type-5 spread is the Selic-CDI gap (|spread_a5| < 3e-7 per quarter), not a price set
  by a bank. `common_rate` checks that cross-institution dispersion is zero in every quarter,
  and the note sentence is printed only when it is.
"""

from __future__ import annotations

import pandas as pd

FALLBACK_NOTE = ("No institution-level prepaid rate is observed in the sample, so every "
                 "institution carries the same CDI-based type-5 rate and the type-5 spread is "
                 "the Selic--CDI gap, not a price set by banks.")
# The tables drop the type-5 spread when it is the common fallback (omit_spread) and say so once.
# The deposit-types table keeps its prepaid row, marked with FALLBACK_NOTE's content.
OMITTED_NOTE = ("Prepaid (type 5) spreads are not shown: no institution-level prepaid rate is "
                "observed in the sample, so every institution carries the CDI rate and the "
                "type-5 spread is the Selic--CDI gap.")


def start_quarter(df: pd.DataFrame) -> tuple[int, int]:
    """First (year, quarter) with a positive type-5 balance."""
    yq = df.loc[df["dep_a5"] > 0, ["year", "quarter"]]
    first = yq.sort_values(["year", "quarter"]).iloc[0]
    return int(first["year"]), int(first["quarter"])


def start_label(df: pd.DataFrame) -> str:
    y, q = start_quarter(df)
    return f"{y}Q{q}"


def before_start(df: pd.DataFrame) -> pd.Series:
    """True on rows dated before the first reported type-5 balance."""
    y, q = start_quarter(df)
    return (df["year"] < y) | ((df["year"] == y) & (df["quarter"] < q))


def common_rate(df: pd.DataFrame, tol: float = 1e-12) -> bool:
    """True when, in every quarter with type-5 balances, all institutions share one rate_a5."""
    held = df.loc[df["dep_a5"] > 0, ["year", "quarter", "rate_a5"]].dropna()
    if held.empty:
        return False
    spread = held.groupby(["year", "quarter"])["rate_a5"].agg(lambda s: s.max() - s.min())
    return bool((spread <= tol).all())


def omit_spread(df: pd.DataFrame) -> bool:
    """True when the type-5 spread carries no price information, so tables leave it out."""
    return common_rate(df)


def fallback_note(df: pd.DataFrame) -> str:
    """FALLBACK_NOTE when every in-sample type-5 rate is the common fallback, else ''."""
    if common_rate(df):
        return FALLBACK_NOTE
    print("  [prepaid] type-5 rates vary across institutions in some quarter: "
          "fallback sentence omitted from the notes")
    return ""
