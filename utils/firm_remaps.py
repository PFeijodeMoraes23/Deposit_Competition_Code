"""utils/firm_remaps.py — a firm the panel keys on one conglomerate code across a change of its
prudential code.

PicPay is one firm, C0088022, in every quarter (Pedro, 2026-10-27, "route 2"). The IF.data List
consolidates PicPay's payment institution (CNPJ 22896431, from 2019-12) and PicPay Bank (09516419)
inside Banco Original's prudential conglomerate C0080903 until 2024-09; the code C0088022 exists
from 2024-12. The panel already files PicPay Bank's ESTBAN balances under C0088022, the latest List
code. Three things follow PicPay to C0088022:
  balances  C0080903's prepaid (type 5) balance, which is PicPay's wallet    (panel_deposits.py)
  rate      C0080903's payment-institution prepaid rate                       (panel_deposit_rates.py)
  chars     C0080903's consolidated characteristics, the filing that consolidated PicPay (PicPay Bank
            from 2014Q4), for every quarter through 2024Q3 where C0088022 has no filing of its own;
            Original keeps them too
                                                                              (panel_bank_chars.py)
The posted-rate rules of PicPay's wallet sit on C0088022 in config/type5_balance_rates.csv.
"""
from __future__ import annotations

import pandas as pd

PICPAY_FROM = "C0080903"   # Banco Original's conglomerate, which consolidated PicPay until 2024-09
PICPAY_TO = "C0088022"     # PicPay's own conglomerate code, from 2024-12
PICPAY_FIRST = (2019, 4)       # the payment institution's first List quarter in C0080903
PICPAY_BANK_FIRST = (2014, 4)  # PicPay Bank's first List quarter in C0080903
PICPAY_LAST = (2024, 3)        # the last quarter C0080903 consolidated either


def picpay_window(year, quarter, first=PICPAY_FIRST) -> pd.Series:
    """True for the quarters, from `first` through PICPAY_LAST, in which C0080903 consolidated
    PicPay: from PICPAY_FIRST its prepaid balance and rate are PicPay's wallet; from
    PICPAY_BANK_FIRST its filing covers PicPay Bank, whose balances the panel holds on C0088022."""
    q = pd.Series(year).astype(int).to_numpy() * 4 + pd.Series(quarter).astype(int).to_numpy()
    lo, hi = first[0] * 4 + first[1], PICPAY_LAST[0] * 4 + PICPAY_LAST[1]
    return pd.Series((q >= lo) & (q <= hi), index=getattr(year, "index", None))


def picpay_window_anomes(anomes) -> pd.Series:
    """picpay_window for a YYYYMM quarter-end month column (AnoMes)."""
    a = pd.Series(anomes).astype(int)
    return picpay_window(a // 100, (a % 100 - 1) // 3 + 1).set_axis(getattr(anomes, "index", a.index))
