###─────────────────────────────────────────────────────────────────────────────
# fees/build_consumidor_fee_complaints.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Fee disputes by institution and municipality, 2016-2024, from the consumidor.gov.br
#   complaint files (one row per closed complaint; the `Total` column of the 2016-2020 files
#   is always 1). Only firms in the two financial segments are kept:
#     "Bancos, Financeiras e Administradoras de Cartão" and "Empresas de Pagamento Eletrônico".
#
#   Counts per cell:
#     n_fin          every complaint against the firm
#     n_fee          problem naming a tariff: "Cobrança de tarifas, taxas, valores não
#                    previstos / não informados" or "Negativação indevida referente a tarifa,
#                    taxas não previstas contrato / serviço" (the only two problems that do,
#                    in all 78 files)
#     n_account      subject "Conta corrente / Salário / Poupança / Conta aposentadoria"
#     n_fee_account  both: a disputed fee on an account, the closest match to account-keeping
#                    fees (card annuities and loan fees fall in n_fee only)
#   Labels are matched on an accent-free upper-case key, so the files that lost accented
#   letters (U+FFFD, 2019-05..2021-11) match as well.
#
#   Dates: `Data Finalização` (the date the complaint was closed); dd/mm/yyyy in the older
#   files, yyyy-mm-dd in the recent ones. Cities get IBGE codes from
#   DATA_RAW/consumidor_gov/crosswalk/consumidor_cidade_ibge.csv (make_consumidor_ibge_crosswalk.py);
#   the files are decoded the same way as there (UTF-8, else Latin-1), so (UF, Cidade) keys agree.
#
#   Firms are kept under their consumidor.gov.br label (`Nome Fantasia`); no conglomerate is
#   assigned. The platform grew and firms joined it at different dates, so shares
#   (n_fee / n_fin) compare better than counts.
#
# OUTPUT  (utils.paths.TARIFAS_PROC)
#   consumidor_fee_complaints_firm_quarter.csv          firm x segment x quarter
#   consumidor_fee_complaints_municipality_quarter.csv  ibge_code x quarter (all firms)
#   consumidor_fee_complaints_firm_municipality_year.csv firm x ibge_code x year
#   consumidor_fee_complaints_firms.csv                 one row per firm label: segment(s),
#       first and last quarter, totals 2016-2024
#
# USAGE
#   python -I fees/build_consumidor_fee_complaints.py     (-I: the complaint files are downloaded data)
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import glob
import io
import os
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/

from utils import paths
from utils import raw_paths

BASE = raw_paths.CONSUMIDOR_GOV
CROSSWALK = BASE / "crosswalk" / "consumidor_cidade_ibge.csv"
OUT = paths.TARIFAS_PROC

COLS = ["UF", "Cidade", "Data Finalização", "Nome Fantasia", "Segmento de Mercado", "Assunto",
        "Problema"]
SEGMENTS = ("BANCOS FINANCEIRAS E ADMINISTRADORAS DE CARTAO", "EMPRESAS DE PAGAMENTO ELETRONICO")
ACCOUNT_SUBJECT = "CONTA CORRENTE SALARIO POUPANCA CONTA APOSENTADORIA"
COUNTS = ["n_fin", "n_fee", "n_account", "n_fee_account"]
YEARS = range(2016, 2025)


def key(s: str) -> str:
    """Accent-free upper-case key, every non-alphanumeric run as one space; a letter the
    source lost (U+FFFD) is dropped."""
    s = unicodedata.normalize("NFKD", str(s).replace("�", "")).encode("ascii", "ignore")
    return re.sub(r"[^A-Z0-9]+", " ", s.decode("ascii").upper()).strip()


def read_zip(path: str) -> pd.DataFrame:
    """One complaint file, decoded as make_consumidor_ibge_crosswalk.read_zip does."""
    z = zipfile.ZipFile(path)
    name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
    raw = z.read(name)
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    d = pd.read_csv(io.StringIO(text), sep=";", dtype=str, on_bad_lines="skip", low_memory=False)
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    return d[COLS]


def closing_date(s: pd.Series) -> pd.Series:
    """`Data Finalização` in either of its two formats."""
    s = s.str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.to_datetime(s.where(~iso), format="%d/%m/%Y", errors="coerce")
    return out.fillna(pd.to_datetime(s.where(iso).str[:10], format="%Y-%m-%d", errors="coerce"))


def flags(d: pd.DataFrame) -> pd.DataFrame:
    """Financial-segment rows with the four count flags, date and keys."""
    seg = d["Segmento de Mercado"].map(key)
    d = d[seg.isin(SEGMENTS)].copy()
    d["segment"] = seg[seg.isin(SEGMENTS)]
    problem, subject = d["Problema"].map(key), d["Assunto"].map(key)
    d["n_fin"] = 1
    d["n_fee"] = problem.str.contains("TARIF").astype(int)
    d["n_account"] = (subject == ACCOUNT_SUBJECT).astype(int)
    d["n_fee_account"] = d["n_fee"] * d["n_account"]
    date = closing_date(d["Data Finalização"])
    d["year"] = date.dt.year
    d["quarter"] = date.dt.quarter
    d["uf"] = d["UF"].astype(str).str.strip().str.upper()
    d["firm"] = d["Nome Fantasia"].str.strip()
    return d.rename(columns={"Cidade": "cidade"})


def with_shares(t: pd.DataFrame) -> pd.DataFrame:
    t["share_fee"] = t["n_fee"] / t["n_fin"]
    t["share_fee_account"] = (t["n_fee_account"] / t["n_account"]).where(t["n_account"] > 0)
    return t


def main() -> None:
    xw = pd.read_csv(CROSSWALK, dtype={"uf": str, "cidade": str})[["uf", "cidade", "ibge_code"]]
    files = sorted(glob.glob(str(BASE / "data" / "*.zip")))
    print(f"{len(files)} complaint files")

    firm_q, mun_q, firm_mun_y = [], [], []
    n_rows = n_fin = n_bad_date = n_no_code = 0
    for f in files:
        raw = read_zip(f)
        n_rows += len(raw)
        d = flags(raw)
        n_fin += len(d)
        n_bad_date += int(d["year"].isna().sum())
        d = d[d["year"].isin(YEARS)]
        d = d.merge(xw, on=["uf", "cidade"], how="left")
        n_no_code += int(d["ibge_code"].isna().sum())
        firm_q.append(d.groupby(["firm", "segment", "year", "quarter"])[COUNTS].sum())
        mun_q.append(d.groupby(["ibge_code", "year", "quarter"])[COUNTS].sum())
        firm_mun_y.append(d.groupby(["firm", "ibge_code", "year"])[COUNTS].sum())
        print(f"{os.path.basename(f)}: {len(raw):,} rows, {len(d):,} financial, "
              f"{int(d['n_fee'].sum()):,} fee", flush=True)

    # a quarter can span two files (semester files, late closings), so cells are summed again
    firm_q = with_shares(pd.concat(firm_q).groupby(level=[0, 1, 2, 3]).sum().reset_index())
    mun_q = pd.concat(mun_q).groupby(level=[0, 1, 2]).sum().reset_index()
    mun_q["ibge_code"] = mun_q["ibge_code"].astype(int)
    mun_q = with_shares(mun_q)
    firm_mun_y = pd.concat(firm_mun_y).groupby(level=[0, 1, 2]).sum().reset_index()
    firm_mun_y["ibge_code"] = firm_mun_y["ibge_code"].astype(int)
    firm_mun_y = with_shares(firm_mun_y)

    firm_q["yq"] = firm_q["year"] * 10 + firm_q["quarter"]
    firms = (firm_q.groupby("firm")
             .agg(segments=("segment", lambda s: "|".join(sorted(set(s)))),
                  first_quarter=("yq", "min"), last_quarter=("yq", "max"),
                  **{c: (c, "sum") for c in COUNTS})
             .reset_index().sort_values("n_fin", ascending=False))
    firms = with_shares(firms)
    firm_q = firm_q.drop(columns="yq")

    OUT.mkdir(parents=True, exist_ok=True)
    firm_q.to_csv(OUT / "consumidor_fee_complaints_firm_quarter.csv", index=False, float_format="%.6g")
    mun_q.to_csv(OUT / "consumidor_fee_complaints_municipality_quarter.csv", index=False,
                 float_format="%.6g")
    firm_mun_y.to_csv(OUT / "consumidor_fee_complaints_firm_municipality_year.csv", index=False,
                      float_format="%.6g")
    firms.to_csv(OUT / "consumidor_fee_complaints_firms.csv", index=False, float_format="%.6g")

    print(f"rows read {n_rows:,}; financial {n_fin:,}; undated {n_bad_date:,}; "
          f"without IBGE code {n_no_code:,}")
    tot = firm_q[COUNTS].sum()
    print(f"2016-2024: n_fin {tot['n_fin']:,}, n_fee {tot['n_fee']:,} "
          f"({100 * tot['n_fee'] / tot['n_fin']:.2f}%), n_fee_account {tot['n_fee_account']:,} "
          f"({100 * tot['n_fee_account'] / tot['n_account']:.2f}% of account complaints)")
    by_year = firm_q.groupby("year")[COUNTS].sum()
    print(with_shares(by_year).to_string(float_format=lambda x: f"{x:.3f}"))
    print(firms.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
