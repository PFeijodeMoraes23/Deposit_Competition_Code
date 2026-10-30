"""
diag_estban_unfiled.py
======================
ESTBAN cells that did not file (task E1, 2026-10-27). A cell is one institution (CNPJ) in one
municipality in one month. It is "unfiled" when AGEN_ESPERADAS > 0, AGEN_PROCESSADAS = 0 and every
balance is blank: the BCB expected the branch and received nothing, so the balances are unknown,
not zero.

For each unfiled cell in a quarter-end month of 2013-2024 it records:
  donor          the nearest filed month of the same cell inside the quarter (one month back,
                 then two), 0 if neither filed
  next_filed     the first later month the cell filed again (any month the files hold, 2025-2026
                 included, as a diagnostic only), 0 if never
  whole_bank     every cell of that CNPJ in that month is unfiled
  kind           gap      filed in the quarter and files again later: a missed report, fillable
                 closure  never files again: the branch closed, an exit, not filled
                 nodonor  files again later but no month of the quarter filed: stays missing
  a1, a2, a3, a4 the donor month's balances (R$), when there is a donor

Inputs (read only): processed ESTBAN.csv (2016-), raw monthly ESTBAN files (2013-2015).
Output: Rout/estban_unfiled_cells.csv, and a summary on stdout.
Usage:  python diagnostics/diag_estban_unfiled.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import paths  # noqa: E402
from utils.estban import process_raw_estban_csv  # noqa: E402

DEP = {"V400_401": "a1", "V420": "a2", "V431": "a3", "V432": "a4"}
COLS = ["CNPJ", "CODMUN", "NOME_INSTITUICAO", "YEAR", "MONTH", "AGEN_ESPERADAS", "AGEN_PROCESSADAS"] + list(DEP)
OUT = paths.rout_dir() / "estban_unfiled_cells.csv"


def load() -> pd.DataFrame:
    parts = []
    for ch in pd.read_csv(paths.ESTBAN_CSV, usecols=COLS, encoding="latin1", low_memory=False,
                          chunksize=1_000_000):
        parts.append(ch)
    files = {}
    for p in list(paths.ESTBAN_RAW_MUN.glob("*.CSV")) + list(paths.ESTBAN_RAW_MUN.glob("*.csv")):
        m = re.match(r"^(\d{4})(\d{2})_ESTBAN", p.name, re.IGNORECASE)
        if m and int(m.group(1)) < 2016:
            files[str(p.resolve()).lower()] = p
    for p in sorted(files.values()):
        d = process_raw_estban_csv(str(p))
        if d is None:
            continue
        for c in COLS:
            if c not in d.columns:
                d[c] = np.nan
        parts.append(d[COLS])
    df = pd.concat(parts, ignore_index=True)
    df["CNPJ"] = pd.to_numeric(df["CNPJ"], errors="coerce").astype("Int64")
    df["CODMUN"] = pd.to_numeric(df["CODMUN"], errors="coerce").astype("Int64")
    for c in ["AGEN_ESPERADAS", "AGEN_PROCESSADAS"] + list(DEP):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["t"] = df["YEAR"].astype(int) * 12 + df["MONTH"].astype(int) - 1
    return df


def main() -> None:
    df = load()
    dep = list(DEP)
    unf = (df["AGEN_ESPERADAS"].fillna(0) > 0) & (df["AGEN_PROCESSADAS"].fillna(0) == 0) & df[dep].isna().all(axis=1)
    filed = df[~unf]
    key = ["CNPJ", "CODMUN"]
    if filed.duplicated(key + ["t"]).any():
        print(f"[!] {int(filed.duplicated(key + ['t']).sum())} duplicate filed cells (kept first)")
    filed = filed.drop_duplicates(key + ["t"])
    fidx = filed.set_index(key + ["t"])

    qe = df["MONTH"].isin([3, 6, 9, 12]) & df["YEAR"].between(2013, 2024)
    tg = df[unf & qe][key + ["NOME_INSTITUICAO", "YEAR", "MONTH", "t"]].copy()
    tg["donor"] = 0
    for c in DEP.values():
        tg[c] = np.nan
    for lag in (1, 2):
        open_ = tg["donor"] == 0
        k = pd.MultiIndex.from_arrays([tg.loc[open_, "CNPJ"], tg.loc[open_, "CODMUN"], tg.loc[open_, "t"] - lag])
        hit = fidx.reindex(k)
        ok = hit["AGEN_PROCESSADAS"].notna().to_numpy()
        rows = tg.index[open_][ok]
        for v, c in DEP.items():
            tg.loc[rows, c] = hit[v].to_numpy()[ok]
        t_d = tg.loc[rows, "t"] - lag
        tg.loc[rows, "donor"] = (t_d // 12) * 100 + t_d % 12 + 1

    # First later filing of the same cell, anywhere in the files (2025-2026 as a diagnostic only).
    later = filed[key + ["t"]].sort_values("t")
    m = tg[key + ["t"]].reset_index().merge(later, on=key, how="left", suffixes=("", "_f"))
    m = m[m["t_f"] > m["t"]].groupby("index")["t_f"].min()
    tg["next_filed"] = 0
    tg.loc[m.index, "next_filed"] = ((m // 12) * 100 + m % 12 + 1).astype(int)

    cells = df[qe].groupby(["CNPJ", "t"]).size()
    unf_n = tg.groupby(["CNPJ", "t"]).size()
    tg["whole_bank"] = [unf_n[(c, t)] == cells.get((c, t), -1) for c, t in zip(tg["CNPJ"], tg["t"])]
    tg["kind"] = np.where(tg["next_filed"] == 0, "closure", np.where(tg["donor"] > 0, "gap", "nodonor"))
    tg = tg.drop(columns="t").sort_values(["YEAR", "MONTH", "CNPJ", "CODMUN"])
    tg.to_csv(OUT, index=False)

    pd.set_option("display.width", 220)
    print(f"unfiled quarter-end cells 2013-2024: {len(tg)}  ->  {tg['kind'].value_counts().to_dict()}")
    print("\nby quarter:")
    tg["yq"] = tg["YEAR"].astype(str) + "Q" + (tg["MONTH"] // 3).astype(str)
    print(tg.pivot_table(index="yq", columns="kind", values="CODMUN", aggfunc="size", fill_value=0).to_string())
    big = tg.assign(dep_bn=(tg["a1"].fillna(0) + tg["a4"].fillna(0)) / 1e9)
    print("\nby bank-quarter with R$ (a1 + a4 of the donor month) >= 0.05 bn, or whole-bank, or a closure:")
    g = big.groupby(["yq", "CNPJ", "NOME_INSTITUICAO"]).agg(cells=("CODMUN", "size"), kinds=("kind", lambda s: dict(s.value_counts())),
                                                         whole=("whole_bank", "all"), dep_bn=("dep_bn", "sum"),
                                                         next_filed=("next_filed", "min")).reset_index()
    print(g[(g["dep_bn"] >= 0.05) | g["whole"] | g["kinds"].astype(str).str.contains("closure")].to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
