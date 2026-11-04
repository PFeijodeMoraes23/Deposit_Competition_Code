"""utils/judicial_deposits.py: judicial and administrative deposits held at the institution (Cosif
4.1.5.50), J_jt, and their removal from the type-4 (time-deposit) stock.

Judicial deposits sit inside Cosif 4.1.5 and ESTBAN verbete 432 (Annex I of IN BCB 429/2023), so
dep_a4 carries them. They move with court outcomes rather than rates, and their expense
(Cosif 8114000008) is not a CDB cost, so the type-4 stock and the type-4 rate both exclude them.

Source: Rout/judicial_deposits_share.csv, one row per conglomerate-quarter for the 14 banks whose
statements report the line (year-end DFP stocks interpolated by quarter, Cosif 4155000009 observed
from 2025-01; columns judicial_inside_415_bn, share_conglomerate, flag). Banks absent from the
file, and quarters it leaves empty, keep their dep_a4 unchanged (judicial_src = "none").

load()                       J per conglomerate-quarter in reais, with judicial_src and the
                             conglomerate share, plus a 2015Q4 row held from 2016Q1 where the
                             file has none (the 2016Q1 lags need it)
remove_from_dep_a4(df, ...)  dep_a4 net of J, spread over the bank's rows in proportion to their
                             dep_a4 (levels, floored at 0); keeps dep_a4_incl_judicial and adds
                             judicial_src and judicial_cap_flag
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from utils import paths

JUDICIAL_CSV = paths.rout_dir() / "judicial_deposits_share.csv"

# A bank's state-capital rows are flagged where its judicial share of 4.1.5 exceeds this: courts
# sit in the capitals, so the booked judicial share there is above the bank's national share
# that the proportional removal uses.
CAP_SHARE = 0.10

# IBGE codes of the 26 state capitals and Brasília.
STATE_CAPITALS_IBGE = frozenset({
    1100205, 1302603, 1200401, 5002704, 1600303, 5300108, 1400100, 5103403, 1721000, 3550308,
    2211001, 3304557, 1501402, 5208707, 2927408, 4205407, 2111300, 2704302, 4314902, 4106902,
    3106200, 2304400, 2611606, 2507507, 2800308, 2408102, 3205309,
})


def load() -> pd.DataFrame:
    """CodConglomeradoPrudencial, AnoMes, J (reais), judicial_share, judicial_src."""
    md5 = hashlib.md5(JUDICIAL_CSV.read_bytes().replace(b"\r", b"")).hexdigest()
    j = pd.read_csv(JUDICIAL_CSV, dtype={"CodConglomeradoPrudencial": str})
    logging.info(f"Judicial deposits: {JUDICIAL_CSV.name}, {len(j)} rows, "
                 f"{j['CodConglomeradoPrudencial'].nunique()} conglomerates, md5 {md5} (LF)")
    j = j.dropna(subset=["judicial_inside_415_bn"])
    out = pd.DataFrame({
        "CodConglomeradoPrudencial": j["CodConglomeradoPrudencial"],
        "AnoMes": j["AnoMes"].astype(int),
        "J": j["judicial_inside_415_bn"] * 1e9,
        "judicial_share": j["share_conglomerate"],
        "judicial_src": j["flag"].fillna("file").astype(str),
    })
    first = out.sort_values("AnoMes").groupby("CodConglomeradoPrudencial").head(1)
    held = first[first["AnoMes"] == 201603].assign(AnoMes=201512, judicial_src="held")
    held = held[~held.set_index(["CodConglomeradoPrudencial", "AnoMes"]).index.isin(
        out.set_index(["CodConglomeradoPrudencial", "AnoMes"]).index)]
    if len(held):
        logging.info(f"Judicial deposits: 2015Q4 held from 2016Q1 for {len(held)} conglomerates")
    return pd.concat([out, held], ignore_index=True)


def remove_from_dep_a4(df: pd.DataFrame, code_col: str = "CodConglomeradoPrudencial",
                       year_col: str = "Year", quarter_col: str = "Quarter",
                       mun_col: str = "CODMUN_IBGE") -> pd.DataFrame:
    """dep_a4 net of the bank's judicial stock, in place on a bank x municipality x quarter frame.

    Every row of bank j in quarter t is scaled by max(0, 1 - J_jt / sum of the bank's dep_a4 that
    quarter), so the removal is in levels (the panel's national dep_a4 differs from the 4.1.5
    total by a few percent) and spread in proportion to each row's dep_a4."""
    j = load()
    am = df[year_col].astype(int) * 100 + df[quarter_col].astype(int) * 3
    key = pd.MultiIndex.from_arrays([df[code_col].astype(str), am])
    jj = j.set_index(["CodConglomeradoPrudencial", "AnoMes"])
    J = pd.Series(jj["J"].reindex(key).to_numpy(), index=df.index)
    share = pd.Series(jj["judicial_share"].reindex(key).to_numpy(), index=df.index)
    src = pd.Series(jj["judicial_src"].reindex(key).to_numpy(), index=df.index)

    tot = df["dep_a4"].groupby([df[code_col], am]).transform("sum")
    factor = (1.0 - J / tot.where(tot > 0)).clip(lower=0.0)
    hit = J.notna() & factor.notna()
    df["dep_a4_incl_judicial"] = df["dep_a4"]
    df.loc[hit, "dep_a4"] = df.loc[hit, "dep_a4"] * factor[hit]
    df["judicial_src"] = src.where(hit, "none")
    capital = pd.to_numeric(df[mun_col], errors="coerce").isin(STATE_CAPITALS_IBGE)
    df["judicial_cap_flag"] = (hit & capital & (share > CAP_SHARE)).astype("int8")

    n_cq = pd.Series(list(zip(df.loc[hit, code_col], am[hit]))).nunique()
    removed = (df["dep_a4_incl_judicial"] - df["dep_a4"]).sum()
    logging.info(f"Judicial deposits out of dep_a4: {n_cq:,} conglomerate-quarters, "
                 f"R${removed / 1e9:,.1f} bn removed in total; "
                 f"{int(df['judicial_cap_flag'].sum()):,} capital rows flagged; "
                 f"{int((hit & (factor == 0)).sum()):,} rows floored at 0")
    return df
