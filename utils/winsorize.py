"""Outlier control for the accounting ratios, applied once at the panel source.

The accounting ratios are stock/stock or flow/stock quotients, so a near-zero denominator
produces a number that is not economically interpretable: `indice_basileia` (capital over
risk-weighted assets) reaches 16,069.73 where a real Basel ratio sits near 0.16. Left in
place these dominate everything built from them -- `indice_basileia_lag` draws 100.0% of its
sum of squares from 0.1% of rows, and because one bank with near-zero RWA enters the
leave-one-out mean of every rival in its market, 0.4% of corrupt source rows become 6.2% of
corrupt `loo_basileia` rows.

Applying the rule in each consumer is what let the panel and its instruments disagree, so it
belongs to the panel build (panel_7_instruments, before the LOO columns are constructed) and
every consumer inherits identical columns from market_panel.csv.

TWO JOBS, DELIBERATELY SEPARATE -- they answer different questions and a single percentile
rule cannot do both:

``apply_validity_bounds``   a HARD economic bound, stated as a threshold a reader can argue
                            with rather than buried in a quantile. Percentiles cannot do this
                            job here: 14.8% of D-firm rows exceed a Basel ratio of 1.0, and
                            no 1% trim removes a 15% mass. The bound is TWO-SIDED, which
                            matters more than it looks -- the panel reaches -32.87, and a
                            ratio of -32.87 squares as large as one of +32.87, so an
                            upper-only clip leaves most of the leverage in place (measured:
                            upper-only left 27-67% of the sum of squares in the top 0.1%,
                            two-sided leaves 0.4-7%).

``winsorize_within_type``   the ordinary tail trim at [pct, 1-pct], SEPARATELY WITHIN each
                            firm type. Within-type is the point: B and D have genuinely
                            different balance sheets (D equity ratios run ~4x B ones), so
                            pooled percentiles would clip real cross-type variation rather
                            than the tail.

Winsorising an INSTRUMENT is a modelling judgement, not housekeeping: a valid instrument
needs variation and clipping removes some. The argument for doing it here is that the
variation being removed is a divide-by-tiny artifact rather than dispersion in bank
capitalisation. That is why the bound is stated as an economic threshold that a reader can
argue with, instead of being buried in a quantile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["VALIDITY_BOUNDS", "apply_validity_bounds", "winsorize_within_type"]


# Hard economic bounds, keyed by the UNLAGGED base name (the helpers also match `<name>_lag`).
# A value outside the bound is not a measurement of the quantity the column claims to hold.
#
# indice_basileia = regulatory capital / risk-weighted assets.
#   UPPER 2.0. Above 1.0 is NOT automatically corrupt: the rows that breach it are genuinely
#   tiny, almost entirely equity-funded institutions (median total assets R$3.9M against
#   R$214M for the rest, median equity/assets 0.94 against 0.22), and for them a ratio above
#   one is real. The median of that group is 1.39, so a bound of 1.0 would winsorize ~1,280
#   real observations. What is not measurable at any firm size is capital at several hundred
#   times risk-weighted assets -- the column reaches 16,069.73 -- so the bound sits above the
#   legitimate band and below the implausible one. It clips 360 bank-quarters (3.4%).
#   LOWER -1.0. Negative regulatory capital is real (an insolvent institution) and the mildly
#   negative values carry no leverage, so they are kept; -1.0 removes only the 24 whose
#   magnitude is itself a denominator artifact.
VALIDITY_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "indice_basileia": (-1.0, 2.0),
}


def _targets(df: pd.DataFrame, base: str) -> list[str]:
    """The columns in `df` that carry `base`: the base itself and its `_lag` sibling."""
    return [c for c in (base, f"{base}_lag") if c in df.columns]


def _type_codes(df: pd.DataFrame, type_key) -> pd.Series | None:
    """Resolve `type_key` to a Series aligned with df, or None if it is unavailable."""
    if type_key is None:
        return None
    if isinstance(type_key, pd.Series):
        return type_key.reindex(df.index)
    if isinstance(type_key, str) and type_key in df.columns:
        return df[type_key]
    return None


def apply_validity_bounds(df: pd.DataFrame, bounds: dict | None = None,
                          verbose: bool = True) -> pd.DataFrame:
    """Clip each column named in `bounds` (and its `_lag` sibling) to its economic range.

    Returns the same frame, modified in place. Prints one audit line per column that had
    anything to clip, so the effect is visible in the panel build log.
    """
    bounds = VALIDITY_BOUNDS if bounds is None else bounds
    for base, (lo, hi) in bounds.items():
        for col in _targets(df, base):
            s = pd.to_numeric(df[col], errors="coerce")
            n_lo = int((s < lo).sum()) if lo is not None else 0
            n_hi = int((s > hi).sum()) if hi is not None else 0
            if not (n_lo or n_hi):
                continue
            before_max = s.max()
            df[col] = s.clip(lower=lo, upper=hi)
            if verbose:
                print(f"    [bound] {col:28s} below {lo}: {n_lo:>6,d}   above {hi}: "
                      f"{n_hi:>6,d}   max {before_max:>12.4f} -> {df[col].max():.4f}")
    return df


def winsorize_within_type(df: pd.DataFrame, cols, pct: float = 0.01, type_key="is_B",
                          dedup_keys=None, verbose: bool = True) -> pd.DataFrame:
    """Clip `cols` to their [pct, 1-pct] quantiles separately within each firm type.

    type_key    column name or Series splitting the sample (B vs D). If it cannot be
                resolved the quantiles are computed on the pooled sample, which is the
                right fallback for a frame that holds one type only.
    dedup_keys  when given, the quantiles are computed on ``df.drop_duplicates(dedup_keys)``
                so each unit votes once no matter how many rows it occupies, and the
                resulting cutoffs are then applied to every row. This matters on
                market_panel.csv, where a bank present in 500 MCAs would otherwise cast 500
                identical votes and the branch networks would set the percentile: on the B
                block the two populations put p99 at 0.21 against 3.15.

    Returns the same frame, modified in place, with an audit line per clipped column.
    """
    cols = [c for c in cols if c in df.columns]
    lo_q, hi_q = pct, 1.0 - pct
    codes = _type_codes(df, type_key)
    groups = ([(f"type={v!r}", codes == v) for v in pd.unique(codes.dropna())]
              if codes is not None else [("pooled", pd.Series(True, index=df.index))])

    quant_src = df.drop_duplicates(dedup_keys) if dedup_keys else df
    if verbose:
        pop = (f"{len(quant_src):,} de-duplicated units" if dedup_keys
               else f"{len(quant_src):,} rows")
        print(f"  Winsorizing at {pct:.0%}/{1-pct:.0%} within firm type "
              f"(cutoffs from {pop}):")

    for v in cols:
        n_clip, before_max = 0, pd.to_numeric(df[v], errors="coerce").max()
        for _, mask in groups:
            s = pd.to_numeric(df.loc[mask, v], errors="coerce")
            # Cutoffs come from the (possibly de-duplicated) quantile population, restricted
            # to the same firm type; the clip is then applied to every row of that type.
            q_mask = mask.reindex(quant_src.index, fill_value=False)
            qs = pd.to_numeric(quant_src.loc[q_mask, v], errors="coerce")
            if qs.notna().sum() < 100:      # too few to form stable percentiles
                continue
            lo, hi = qs.quantile(lo_q), qs.quantile(hi_q)
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
                continue
            clipped = s.clip(lo, hi)
            # NaN != NaN is True in pandas, so guard with notna() or the count degenerates
            # into the missing-value count and badly overstates the clipping.
            n_clip += int(((clipped != s) & s.notna()).sum())
            df.loc[mask, v] = clipped
        if verbose and n_clip:
            after_max = pd.to_numeric(df[v], errors="coerce").max()
            print(f"    {v:28s} clipped {n_clip:>7,d} obs   max {before_max:>12.4f} "
                  f"-> {after_max:.4f}")
    return df
