"""utils/mca.py — the municipality -> MCA crosswalk and the MCA population, shared by the
MCA-level panel scrapers (scrape_pix_municipal, scrape_anatel_mobile, scrape_bcb_inclusion,
scrape_bcb_banked, scrape_cadunico).

load_mca_crosswalk(path)    municipality_code (int, 7-digit IBGE), mca_code (str), year (int)
load_mca_population(path)   mca_code (str), year (int), pop_total; None if the file is absent
per1000(count, pop)         count / (pop / 1000), NaN where pop <= 0 or missing
per100(count, pop)          count / pop * 100,    NaN where pop <= 0 or missing

The two rate helpers keep the operation order of the code they replace, so the floats are
bit-identical: the per-1000 rates divide by (pop / 1000), the per-100 rate multiplies last.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def load_mca_crosswalk(path) -> pd.DataFrame:
    return pd.read_csv(path,
                       usecols=["municipality_code", "mca_code", "year"],
                       dtype={"municipality_code": int, "mca_code": str, "year": int})


def load_mca_population(path) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path,
                       usecols=["mca_code", "year", "pop_total"],
                       dtype={"mca_code": str, "year": int})


def per1000(count, pop):
    return np.where(pop > 0, count / (pop / 1000), np.nan)


def per100(count, pop):
    return np.where(pop > 0, count / pop * 100, np.nan)
