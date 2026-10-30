"""utils/calendar_lags.py — lags by calendar quarter.

An entity (conglomerate x deposit type x market) can miss a quarter: a firm absent from a market for
a quarter, a filing gap that could not be filled. A positional lag (groupby(entity).shift(k)) then
hands the row after the hole a value from more than k quarters back, with nothing to flag it. lag()
looks the value up at exactly k quarters earlier and returns NaN when the entity has no row there,
so the row leaves any regression that needs the lag (decided 2026-10-27; sleep_est_e2.py and the
demand preps).
"""
from __future__ import annotations

import pandas as pd


def lag(df: pd.DataFrame, col: str, k: int = 1, entity: str = "entity_id") -> pd.Series:
    """`col` of the same entity exactly k quarters earlier, aligned on df's index; NaN where the
    entity has no row in that quarter. df needs `year`, `quarter` and `entity`, unique on the three."""
    q = df["year"].astype(int) * 4 + df["quarter"].astype(int)
    here = pd.MultiIndex.from_arrays([df[entity], q])
    if here.has_duplicates:
        raise ValueError(f"lag({col!r}): {int(here.duplicated().sum())} rows repeat an "
                         f"({entity}, quarter) pair; a calendar lag needs one row per pair")
    values = pd.Series(df[col].to_numpy(), index=here)
    back = pd.MultiIndex.from_arrays([df[entity], q - k])
    return pd.Series(values.reindex(back).to_numpy(), index=df.index)
