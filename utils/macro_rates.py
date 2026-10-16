"""utils/macro_rates.py — quarterly macro rates (Selic, CDI, savings, Selic target) from the
staged SGS file raw/SGS_RAW/macro_series_full.csv (no script in the repo downloads it).

load_macro_series()        daily series, DatetimeIndex
make_quarterly_macro(df)   one row per quarter: AnoMes, selic_qoq, cdi_qoq, savings_rate_qoq,
                           meta_selic (rates as FRACTIONS)

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
        AnoMes, selic_qoq, cdi_qoq, savings_rate_qoq, meta_selic
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

        # ---- 4) TR (Taxa Referencial, % per month) ----
        tr_col = "TR" if "TR" in grp.columns else "TR_Diaria"
        tr = grp[tr_col].dropna() if tr_col in grp.columns else pd.Series(dtype=float)
        avg_tr_monthly = tr.mean() if len(tr) else 0.0

        # ---- 5) Savings-deposit rate (poupança) ----
        #   Old rule (Selic > 8.5 %): TR + 0.5 % per month
        #   New rule (Selic <= 8.5 %): TR + 70 % x Selic_target / 12
        if pd.notna(meta_selic):
            if meta_selic > 8.5:
                monthly_component = 0.005               # fixed 0.5 % p.m.
            else:
                monthly_component = 0.70 * (meta_selic / 100) / 12
            monthly_savings = avg_tr_monthly / 100 + monthly_component
            savings_qoq = (1 + monthly_savings) ** 3 - 1
        else:
            savings_qoq = np.nan

        rows.append({
            "AnoMes":           ano_mes,
            "selic_qoq":        selic_qoq,
            "cdi_qoq":          cdi_qoq,
            "savings_rate_qoq": savings_qoq,
            "meta_selic":       meta_selic,
        })

    df_rates = pd.DataFrame(rows)
    logging.info(f"Quarterly macro rates computed: {len(df_rates)} quarters")
    return df_rates
