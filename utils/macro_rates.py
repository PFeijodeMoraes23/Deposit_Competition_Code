"""utils/macro_rates.py — quarterly macro rates (Selic, CDI, Selic target) from the staged SGS
file raw/SGS_RAW/macro_series_full.csv (no script in the repo downloads it), and the savings
rates from the credited yields in raw/SGS_RAW/savings_yields/ (scrape_bcb_savings_yields.py).

load_macro_series()          daily series, DatetimeIndex
make_quarterly_macro(df)     one row per quarter: AnoMes, selic_qoq, cdi_qoq, meta_selic,
                             savings_rate_qoq, savings_old_qoq (rates as FRACTIONS)
quarterly_savings_yields()   the two savings columns alone
savings_cost_qoq(d, share)   a bank's savings cost rate given its old-rule share

Moved verbatim from panel_deposit_rates.py (2026-10-15). Readers: panel_deposit_rates (step 16,
which also writes PANEL_INTERMED/quarterly_macro_rates.csv in percent) and panel_cosif_calibrate
(step 13, which computes the rates itself instead of reading that later step's CSV).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from utils import paths


def load_macro_series() -> pd.DataFrame:
    """
    Load daily macro series from the pre-downloaded SGS_RAW/macro_series_full.csv
    (staged out of band; no script in this repo produces it).

    Returns a DataFrame indexed by date with columns:
        Selic_Over, Meta_Selic, CDI_Anualizado, TR
    """
    macro_file = os.path.join(str(paths.SGS_RAW), "macro_series_full.csv")

    if not os.path.exists(macro_file):
        raise FileNotFoundError(
            f"Macro series file not found: {macro_file}\n"
            "No script in this repo downloads it: stage the SGS macro series there."
        )

    df = pd.read_csv(macro_file, index_col=0, parse_dates=True)

    if df.empty:
        raise ValueError(
            f"Macro series file is empty: {macro_file}\n"
            "Re-stage the SGS macro series; no in-repo script regenerates it."
        )

    # Ensure the index is a proper DatetimeIndex (handles CSV round-trip issues)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    logging.info(f"Macro series loaded from {macro_file} ({len(df)} days)")
    return df


def make_quarterly_macro(df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Compound daily macro rates into quarter-over-quarter aggregates aligned
    with IF-Data quarter-end months (3, 6, 9, 12).

    Returns one row per quarter with:
        AnoMes, selic_qoq, cdi_qoq, meta_selic, savings_rate_qoq, savings_old_qoq
    The savings columns come from quarterly_savings_yields().
    """
    df = df_macro.copy()
    df["year"]    = df.index.year
    df["month"]   = df.index.month
    df["q_month"] = ((df["month"] - 1) // 3 + 1) * 3   # 3, 6, 9, 12

    rows = []
    for (yr, qm), grp in df.groupby(["year", "q_month"]):
        ano_mes = yr * 100 + qm

        # ---- 1) Selic QoQ: compound daily overnight rates ----
        selic = grp["Selic_Over"].dropna()
        selic_qoq = (1 + selic / 100).prod() - 1 if len(selic) else np.nan

        # ---- 2) CDI QoQ ----
        # CDI_Anualizado is % p.a.  ->  daily rate = (1 + r_aa)^(1/252) - 1
        cdi = grp["CDI_Anualizado"].dropna()
        if len(cdi):
            cdi_daily = (1 + cdi / 100) ** (1 / 252) - 1
            cdi_qoq = (1 + cdi_daily).prod() - 1
        else:
            cdi_qoq = np.nan

        # ---- 3) Meta Selic (end-of-quarter policy rate) ----
        meta = grp["Meta_Selic"].dropna()
        meta_selic = meta.iloc[-1] if len(meta) else np.nan

        rows.append({
            "AnoMes":           ano_mes,
            "selic_qoq":        selic_qoq,
            "cdi_qoq":          cdi_qoq,
            "meta_selic":       meta_selic,
        })

    df_rates = pd.DataFrame(rows).merge(quarterly_savings_yields(), on="AnoMes", how="left")
    logging.info(f"Quarterly macro rates computed: {len(df_rates)} quarters "
                 f"({int(df_rates['savings_rate_qoq'].notna().sum())} with savings yields)")
    return df_rates


# Share of a bank's savings stock deposited up to 03.05.2012, which keeps the old rule (0.5% a
# month + TR; Lei 12.703/2012, withdrawals drain the new-rule balance first).  No source gives it
# by bank, so the rate steps use 0; where a bank's statements give its savings expense,
# panel_cosif_calibrate's statement benchmark replaces the imputed savings cost and needs no share.
OLD_RULE_SHARE = 0.0


def savings_cost_qoq(d: pd.DataFrame, old_rule_share) -> pd.Series:
    """The savings cost rate of a bank whose savings stock holds old_rule_share (scalar or
    Series) of deposits made up to 03.05.2012: savings_rate_qoq plus that share of the old-rule
    premium (savings_old_qoq - savings_rate_qoq, zero when the Selic target is above 8.5%)."""
    premium = (d["savings_old_qoq"] - d["savings_rate_qoq"]).fillna(0.0)
    return d["savings_rate_qoq"] + old_rule_share * premium


SAVINGS_YIELDS_DIR = paths.SGS_RAW / "savings_yields"
SAVINGS_SGS = {195: "savings_rate_qoq", 25: "savings_old_qoq"}


def quarterly_savings_yields() -> pd.DataFrame:
    """Quarterly savings rates (fractions) from the credited yields scrape_bcb_savings_yields.py
    saves: savings_rate_qoq from SGS 195 (deposits from 04.05.2012, the rate a new deposit
    earns, which is the panel's type-2 price) and savings_old_qoq from SGS 25 (deposits up to
    03.05.2012, kept at 0.5% a month + TR by Lei 12.703/2012).

    A month's yield is the mean over its anniversary dates of the yield credited on that date;
    a quarter compounds its three months, and a quarter missing a month is NaN."""
    frames = []
    for code, col in SAVINGS_SGS.items():
        files = sorted(SAVINGS_YIELDS_DIR.glob(f"sgs_{code}_*.json"))
        if not files:
            raise FileNotFoundError(f"No SGS {code} file in {SAVINGS_YIELDS_DIR}: "
                                    "run scrape_bcb_savings_yields.py")
        d = pd.concat([pd.read_json(f, dtype={"valor": str}) for f in files], ignore_index=True)
        d["date"] = pd.to_datetime(d["data"], dayfirst=True)
        d = d.drop_duplicates("date")
        d["y"] = pd.to_numeric(d["valor"], errors="coerce") / 100
        m = d.groupby(d["date"].dt.to_period("M"))["y"].mean().dropna()
        q = pd.DataFrame({"g": 1 + m.to_numpy(),
                          "AnoMes": [p.year * 100 + ((p.month - 1) // 3 + 1) * 3 for p in m.index]})
        g = q.groupby("AnoMes")["g"]
        frames.append((g.prod() - 1).where(g.count() == 3).rename(col))
    return pd.concat(frames, axis=1).rename_axis("AnoMes").reset_index()
