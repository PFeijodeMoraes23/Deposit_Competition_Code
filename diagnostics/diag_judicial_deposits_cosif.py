"""
diag_judicial_deposits_cosif.py
===============================
Judicial deposits inside Cosif 4.1.5 (time deposits), steps 2 and 3 of task JD1
(claude/bb_handoff_2026-11-03.md).

Cosif books judicial and administrative deposits held by the institution in their own title,
4.1.5.50.00-7 (IN BCB 271/2022 as amended by IN 459/2024; 4.1.5.50.00.00-9 in IN BCB 429/2023),
and ESTBAN maps that title to verbete 432 with the ordinary time deposits, so dep_a4 carries
them. The public monthly balancetes publish 4.1.5 only as a total before 2025 (41500002); from
2025-01 they publish the titles 4.1.5.10 (4151000003), 4.1.5.20 (4152000002), 4.1.5.50
(4155000009) and 4.1.5.99 (4159900002), and the expense titles 8.1.1 lump (8110000002),
time deposits (8113000009) and judicial deposits (8114000008).

Inputs (read only): shared/COSIF/<YYYYMM>BANCOS.* (DOC 4010 rows), 2015-12 (the lag of 2016Q1) to the latest month on
disk; the IF-Data List crosswalk (utils.conglomerates), as panel_cosif_calibrate.py uses it;
processed/COSIF_PROCESSED/custos_implicitos_BANCO_DO_BRASIL_-_BANCO_MULTIPLO.csv (a check that
the 4.1.5 total read here equals the extractor's stk_time); cosif_cdb_rate_corrected.csv (the
pipeline's type-4 rate, for comparison in step 3).
Outputs (Rout):
  - cosif_415_subaccounts_monthly.csv: institution and conglomerate rows x month; the 4.1.5 total,
    its titles and shares where published, and the semester-cumulative expense titles as published.
    Months from 2025-01 lie outside the 2016-2024 window and are kept because they are the only
    months in which the public data separate judicial deposits.
  - cosif_judicial_rates_2025.csv: step 3, quarter x bank with a judicial balance: the time-deposit
    rate over its own stock (4.1.5.10) and over the 4.1.5 total, the judicial rate, and the
    pipeline's rate_a4_corrected_qoq.
Expense titles are cumulative within each semester (they reset in January and July); quarterly
expense is the difference within the semester.
Usage:  python diagnostics/diag_judicial_deposits_cosif.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from utils import paths
from utils.conglomerates import build_cnpj_to_congl, deposit_panel_codes, map_foundation_to_congl

RAW = Path(paths.COSIF_RAW)
PROC = paths.PROCESSED / "COSIF_PROCESSED"
OUT = paths.rout_dir()
M0 = 201512

STOCK = {"41500002": "stk_415_total", "4150000004": "stk_415_total", "4151000003": "stk_4151_time",
         "4152000002": "stk_4152_fx", "4155000009": "stk_4155_judicial", "4159900002": "stk_4159_hedge"}
EXPENSE = {"81100008": "exp_811_lump_semcum", "8110000002": "exp_811_lump_semcum",
           "8113000009": "exp_8113_time_semcum", "8114000008": "exp_8114_judicial_semcum"}
CODES = {**STOCK, **EXPENSE}
SUBS = ["stk_4151_time", "stk_4152_fx", "stk_4155_judicial", "stk_4159_hedge"]


def month_files():
    rx = re.compile(r"^(\d{6})BANCOS\.(zip|csv\.zip|csv)$", re.I)
    best = {}
    for f in os.listdir(RAW):
        m = rx.match(f)
        if not m or int(m.group(1)) < M0:
            continue
        rank = 0 if m.group(2).lower() in ("zip", "csv.zip") else 1
        if m.group(1) not in best or rank < best[m.group(1)][0]:
            best[m.group(1)] = (rank, RAW / f)
    return {int(k): v[1] for k, v in sorted(best.items())}


def read_month(item):
    ym, p = item
    if str(p).lower().endswith(".zip"):
        with zipfile.ZipFile(p) as z:
            n = [x for x in z.namelist() if x.lower().endswith(".csv")][0]
            raw = z.read(n)
    else:
        raw = p.read_bytes()
    df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1", skiprows=3, dtype=str,
                     on_bad_lines="skip")
    df = df[(df["DOCUMENTO"].str.strip() == "4010") & df["CONTA"].isin(CODES)]
    s = df["SALDO"].str.strip()
    # SALDO uses a decimal comma; any thousands dot is dropped first
    df = df.assign(value=pd.to_numeric(s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                                       errors="coerce"),
                   col=df["CONTA"].map(CODES), AnoMes=ym)
    return df[["AnoMes", "CNPJ", "NOME_INSTITUICAO", "TAXONOMIA", "col", "value"]]


def build_monthly():
    files = month_files()
    with ThreadPoolExecutor(max_workers=4) as ex:
        long = pd.concat(list(ex.map(read_month, files.items())), ignore_index=True)
    long["CNPJ"] = long["CNPJ"].str.strip().str.zfill(8)
    names = (long.sort_values("AnoMes").groupby("CNPJ")[["NOME_INSTITUICAO", "TAXONOMIA"]].last())
    w = long.pivot_table(index=["CNPJ", "AnoMes"], columns="col", values="value", aggfunc="sum").reset_index()
    for c in list(STOCK.values()) + list(EXPENSE.values()):
        if c not in w.columns:
            w[c] = np.nan
    w = w.merge(names, left_on="CNPJ", right_index=True, how="left")
    # conglomerate code, as panel_cosif_calibrate keys COSIF rows
    dm = build_cnpj_to_congl()
    dm["CodConglomeradoPrudencial"] = deposit_panel_codes(dm["CNPJ"], dm["CodConglomeradoPrudencial"])
    codes = map_foundation_to_congl(w[["CNPJ", "AnoMes"]], dm)[["CNPJ", "AnoMes", "CodConglomeradoPrudencial"]]
    w = w.merge(codes, on=["CNPJ", "AnoMes"], how="left")   # institutions without a code are kept
    return w, files


def add_shares(d):
    d = d.copy()
    d["subaccounts_published"] = d["AnoMes"] >= 202501
    for c in SUBS:
        d["share_" + c[4:]] = np.where(d["subaccounts_published"] & (d["stk_415_total"] > 0),
                                       d[c].fillna(0) / d["stk_415_total"], np.nan)
    d["window"] = np.select([d["AnoMes"] < 201601, d["AnoMes"] <= 202412], ["2015-12 (lag of 2016Q1)", "2016-2024"], "2025+ (outside the window)")
    d["month"] = (d["AnoMes"] // 100).astype(str) + "-" + (d["AnoMes"] % 100).astype(str).str.zfill(2)
    return d


def quarterly_expense(g, col):
    """Quarterly expense from a semester-cumulative series observed at quarter ends."""
    g = g.set_index("AnoMes")[col]
    out = {}
    for ym in g.index:
        mm = ym % 100
        if mm in (3, 9):
            out[ym] = g[ym]
        elif mm in (6, 12):
            prev = ym - 3
            out[ym] = g[ym] - g[prev] if prev in g.index else np.nan
    return pd.Series(out)


def step3(inst):
    jd = inst[(inst["AnoMes"] >= 202501) & (inst["stk_4155_judicial"] > 0)]
    banks = jd.groupby("CNPJ")["stk_4155_judicial"].max().sort_values(ascending=False).index
    rows = []
    for c in banks:
        g = inst[(inst["CNPJ"] == c) & (inst["AnoMes"] >= 202412)].sort_values("AnoMes")
        g = g[g["AnoMes"] % 100 % 3 == 0]
        e13, e14 = quarterly_expense(g, "exp_8113_time_semcum"), quarterly_expense(g, "exp_8114_judicial_semcum")
        s = g.set_index("AnoMes")
        for ym in s.index:
            if ym < 202503:
                continue
            lag = ym - 3 if ym % 100 != 3 else ym - 100 + 9
            if lag not in s.index:
                continue
            L = s.loc[lag]
            x13 = abs(e13.get(ym, np.nan)); x14 = abs(e14.get(ym, np.nan))
            rows.append(dict(CNPJ=c, bank=s.loc[ym, "NOME_INSTITUICAO"], CodConglomeradoPrudencial=s.loc[ym, "CodConglomeradoPrudencial"],
                             AnoMes=ym, judicial_share_lag=(L["stk_4155_judicial"] / L["stk_415_total"]) if L["stk_415_total"] else np.nan,
                             rate_time_over_own_stock=x13 / L["stk_4151_time"] if L["stk_4151_time"] > 0 else np.nan,
                             rate_time_over_415_total=x13 / L["stk_415_total"] if L["stk_415_total"] > 0 else np.nan,
                             rate_judicial=x14 / L["stk_4155_judicial"] if L["stk_4155_judicial"] > 0 else np.nan))
    r = pd.DataFrame(rows)
    corr = pd.read_csv(PROC / "cosif_cdb_rate_corrected.csv", dtype={"CodConglomeradoPrudencial": str})
    r = r.merge(corr[["CodConglomeradoPrudencial", "AnoMes", "rate_a4_corrected_qoq"]],
                on=["CodConglomeradoPrudencial", "AnoMes"], how="left")
    r["understatement_factor"] = r["rate_time_over_415_total"] / r["rate_time_over_own_stock"]
    return r


def main():
    inst, files = build_monthly()
    print(f"months read: {len(files)} ({min(files)}..{max(files)}); institutions: {inst['CNPJ'].nunique()}")
    # check: the 4.1.5 total equals the extractor's stk_time for BB
    bb = pd.read_csv(PROC / "custos_implicitos_BANCO_DO_BRASIL_-_BANCO_MULTIPLO.csv", parse_dates=["DATA_BASE"])
    bb["AnoMes"] = bb["DATA_BASE"].dt.year * 100 + bb["DATA_BASE"].dt.month
    chk = inst[inst["CNPJ"] == "00000000"].merge(bb[["AnoMes", "stk_time"]], on="AnoMes")
    dev = (chk["stk_415_total"] - chk["stk_time"]).abs().max()
    print(f"check vs extractor stk_time (BB, {len(chk)} months): max abs difference R$ {dev:,.2f}")
    assert dev < 1.0, "4.1.5 total read here differs from the extractor's stk_time"
    pub = inst[inst["AnoMes"] >= 202501]
    gap = (pub[SUBS].fillna(0).sum(axis=1) - pub["stk_415_total"]).abs() / pub["stk_415_total"].where(pub["stk_415_total"] > 0)
    print(f"2025+: titles sum to the 4.1.5 total within 0.1% in {int((gap < 1e-3).sum())} of {int(gap.notna().sum())} rows")

    inst = add_shares(inst.assign(level="institution"))
    cg = (inst.groupby(["CodConglomeradoPrudencial", "AnoMes"])
          [list(dict.fromkeys(STOCK.values())) + list(dict.fromkeys(EXPENSE.values()))]
          .sum(min_count=1).reset_index())
    cg = add_shares(cg.assign(level="conglomerate", CNPJ="", NOME_INSTITUICAO="", TAXONOMIA=""))
    cols = ["level", "CodConglomeradoPrudencial", "CNPJ", "NOME_INSTITUICAO", "TAXONOMIA", "month", "AnoMes", "window",
            "subaccounts_published", "stk_415_total"] + SUBS + ["share_" + c[4:] for c in SUBS] + \
           list(dict.fromkeys(EXPENSE.values()))
    out = pd.concat([inst[cols], cg[cols]], ignore_index=True).sort_values(["level", "CodConglomeradoPrudencial", "CNPJ", "AnoMes"])
    out.to_csv(OUT / "cosif_415_subaccounts_monthly.csv", index=False, float_format="%.2f")
    print(f"wrote {OUT / 'cosif_415_subaccounts_monthly.csv'}: {len(out):,} rows")

    pd.set_option("display.width", 220)
    j = cg[(cg["AnoMes"] >= 202501)].groupby("CodConglomeradoPrudencial")["share_4155_judicial"].max()
    print("\nConglomerates with a judicial share of 4.1.5 above 10% in any month from 2025-01:")
    print(j[j > 0.10].sort_values(ascending=False).round(3).to_string())
    r = step3(inst)
    r.to_csv(OUT / "cosif_judicial_rates_2025.csv", index=False, float_format="%.6f")
    print("\nStep 3, quarterly rates (fractions per quarter), banks with a judicial balance:")
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
