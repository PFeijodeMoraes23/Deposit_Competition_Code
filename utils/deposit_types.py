"""
utils/deposit_types.py
======================
Deposit-type variants of the estimation sample (task TR5: how much the estimates depend on the
prepaid product, deposit type 5).

    ""        production: every row the calling step keeps (the default)
    "no5"     drop every type-5 row
    "d5only"  drop the type-5 rows of brick-and-mortar firms (is_B), keeping the digital firms'

The variant comes from the environment variable DEPOSIT_TYPE_VARIANT or, when that is unset, from
config/deposit_type_variant.txt next to the scripts. Only a TR5 sandbox (tr5_variants.sh) writes
that file, so the repository and the production cluster tree never carry one; a sandbox job that
does not inherit the environment still reads its variant from the file. The two must agree when
both are set.

apply_variant() is called by the sleepiness sample builder (sleep_est_e2.build_pooled_data, which
E1, E3 and E4 also use) and by the demand preps right after they reshape the panel to one row per
deposit type. Under the default it returns its input unchanged (the same object).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

VARIANTS = ("", "no5", "d5only")
ENV = "DEPOSIT_TYPE_VARIANT"
FILE = Path(__file__).resolve().parent.parent / "config" / "deposit_type_variant.txt"


def variant() -> str:
    """The active variant: "" (production), "no5" or "d5only"."""
    env = os.environ.get(ENV)
    env = None if env is None else env.strip().lower()
    disk = FILE.read_text(encoding="utf-8").strip().lower() if FILE.is_file() else None
    if env is not None and disk is not None and env != disk:
        raise ValueError(f"{ENV}={env!r} disagrees with {FILE} ({disk!r})")
    v = env if env is not None else (disk or "")
    if v not in VARIANTS:
        raise ValueError(f"unknown deposit-type variant {v!r}; expected one of {VARIANTS}")
    return v


def _is_b(df: pd.DataFrame) -> pd.Series:
    if "is_B" in df.columns:
        return df["is_B"].astype(bool)
    if "CODMUN_IBGE" in df.columns:
        return df["CODMUN_IBGE"].astype(str) != "0"
    raise KeyError("deposit-type variant d5only needs is_B or CODMUN_IBGE to tell B rows from D rows")


def drop_mask(df: pd.DataFrame, v: str | None = None, type_col: str = "deposit_type") -> pd.Series:
    """True for the rows the variant drops (all False under the default)."""
    v = variant() if v is None else v
    if not v:
        return pd.Series(False, index=df.index)
    t5 = df[type_col] == 5
    return t5 if v == "no5" else t5 & _is_b(df)


def apply_variant(df: pd.DataFrame, where: str = "", type_col: str = "deposit_type") -> pd.DataFrame:
    """df without the rows the active variant drops; under the default, df itself."""
    v = variant()
    if not v:
        return df
    drop = drop_mask(df, v, type_col)
    print(f"  [deposit-type variant {v}] {where}: dropped {int(drop.sum()):,} of {len(df):,} rows "
          f"(type 5{'' if v == 'no5' else ', B firms'})", flush=True)
    return df[~drop].copy()
