###─────────────────────────────────────────────────────────────────────────────
# fees/build_pof_bank_fees.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   What households report paying in bank fees, from the POF 2017-2018 (IBGE) microdata: one
#   cross-section, a household benchmark for the listed account-keeping prices of
#   build_account_fee_history.py. Three measures per family (consumption unit, UC):
#     acct_keeping       5-digit item 26005, the fee for keeping an account. Its 7-digit items
#                        2600501 TARIFA DE CONTA BANCARIA, 2600502 MANUTENCAO DE CONTA BANCARIA,
#                        2600503 MANUTENCAO DE CHEQUE ESPECIAL and 2600504 PACOTE DE SERVICO DE
#                        CONTA BANCARIA only record the respondent's wording; IBGE works with
#                        the 5-digit code
#     bank_fees          IBGE's category 61 "TARIFAS_SERVICOS_BANCARIOS" (Tradutor_Despesa_Perfil
#                        dimensions 6-9): 26001-26003 and 26005-26014. It includes overdraft and
#                        credit-card interest (26001, 26002)
#     bank_fees_ex_int   bank_fees without 26001 and 26002
#
#   Method of IBGE's memo (Memoria_de_Calculo/R/Tabela de Despesa Geral.R): quadro 26 of the
#   POF 4 questionnaire (30-day recall) in DESPESA_INDIVIDUAL, plus the few quadro-26 records of
#   SERVICO_NAO_MONETARIO_POF4, as IBGE's "Perfil das despesas" tables count monetary and
#   non-monetary spending. Monthly value = V8000_DEFLA * FATOR_ANUALIZACAO / 12 (V8000_DEFLA is
#   at prices of 2018-01-15; quadro 26 does not take the V9011 months factor), expanded with
#   PESO_FINAL. A family's value is the sum over its members. Every UC, paying or not, comes
#   from MORADOR, the only register that holds them all.
#
#   Income deciles are person-weighted, on RENDA_DISP_PC (disposable income per capita), as in
#   the published tables.
#
#   CHECK: the script recomputes the Brasil, region and decile figures of IBGE's published
#   tables 6.2 (share of persons in families with the expense) and 6.3 (monthly spending per
#   capita) for "Taxas bancárias, juros de cheque especial e de cartão de crédito", which is
#   bank_fees. The published subgroup rows are contributions to the national figure (they sum
#   to Brasil), so the check uses that form; the outputs below use within-group means.
#
# OUTPUT  (utils.paths.TARIFAS_PROC)
#   pof2017_bank_fees_by_group.csv   group_type (brasil, region, uf, income_decile) x group x
#       measure: n_uc_sample, n_uc_paying_sample, families, persons, share_families_paying,
#       mean_per_family, mean_per_paying_family, mean_per_capita, share_persons_paying
#       (R$ per month, 2018-01-15 prices; shares in %)
#   pof2017_bank_fees_items.csv      Brasil, one row per item code (7-digit inside 26005,
#       5-digit for the rest of category 61): records, n_uc_paying_sample, families_paying,
#       share_families_paying, mean_per_family, mean_per_paying_family
#   pof2017_bank_fees_check.csv      published against computed, per row of tables 6.2 and 6.3
#
# USAGE
#   python -I fees/build_pof_bank_fees.py      (-I: the POF files are downloaded data)
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/

from utils import paths
from utils import raw_paths

POF = raw_paths.IBGE_POF_2017_18
DADOS = POF / "Dados_20230713.zip"
TABLES = POF / "perfil_despesas_inclusao_fin" / "dim06"
OUT = paths.TARIFAS_PROC

# Fixed-width layouts, from IBGE's "Leitura dos Microdados - R.R" (Programas_de_Leitura).
DESPESA_W = [2, 4, 1, 9, 2, 1, 2, 2, 2, 7, 2, 10, 2, 2, 1, 1, 1, 12, 10, 1, 2, 14, 14, 10, 5]
DESPESA_N = ["UF", "ESTRATO_POF", "TIPO_SITUACAO_REG", "COD_UPA", "NUM_DOM", "NUM_UC",
             "COD_INFORMANTE", "QUADRO", "SEQ", "V9001", "V9002", "V8000", "V9010", "V9011",
             "V9012", "V4104", "V4105", "DEFLATOR", "V8000_DEFLA", "COD_IMPUT_VALOR",
             "FATOR_ANUALIZACAO", "PESO", "PESO_FINAL", "RENDA_TOTAL", "V9004"]
NAO_MONET_W = [2, 4, 1, 9, 2, 1, 2, 2, 2, 7, 2, 10, 2, 2, 1, 1, 12, 10, 1, 2, 14, 14, 10, 5]
NAO_MONET_N = [c for c in DESPESA_N if c != "V9012"]
MORADOR_W = [2, 4, 1, 9, 2, 1, 2, 2, 1, 2, 2, 4, 3, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 14, 14, 10, 1, 20, 20,
             20, 20]
MORADOR_N = ["UF", "ESTRATO_POF", "TIPO_SITUACAO_REG", "COD_UPA", "NUM_DOM", "NUM_UC",
             "COD_INFORMANTE", "V0306", "V0401", "V04021", "V04022", "V04023", "V0403", "V0404",
             "V0405", "V0406", "V0407", "V0408", "V0409", "V0410", "V0411", "V0412", "V0413",
             "V0414", "V0415", "V0416", "V041711", "V041712", "V041721", "V041722", "V041731",
             "V041732", "V041741", "V041742", "V0418", "V0419", "V0420", "V0421", "V0422",
             "V0423", "V0424", "V0425", "V0426", "V0427", "V0428", "V0429", "V0430",
             "ANOS_ESTUDO", "PESO", "PESO_FINAL", "RENDA_TOTAL", "NIVEL_INSTRUCAO",
             "RENDA_DISP_PC", "RENDA_MONET_PC", "RENDA_NAO_MONET_PC", "DEDUCAO_PC"]
QUADRO_COLS = slice(21, 23)  # byte position of QUADRO in both expense registers

UC_KEY = ["COD_UPA", "NUM_DOM", "NUM_UC"]
INTEREST = [26001, 26002]
CAT61 = [26001, 26002, 26003] + list(range(26005, 26015))
MEASURES = {
    "acct_keeping": [26005],
    "bank_fees": CAT61,
    "bank_fees_ex_int": [c for c in CAT61 if c not in INTEREST],
}
REGIONS = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste"}
UF_SIGLA = {11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO", 21: "MA",
            22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA",
            31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR", 42: "SC", 43: "RS", 50: "MS",
            51: "MT", 52: "GO", 53: "DF"}


def read_quadro(z: zipfile.ZipFile, member: str, widths: list[int], names: list[str],
                quadro: bytes = b"26") -> pd.DataFrame:
    """One quadro of an expense register; lines are filtered before parsing (the register
    is 1.8 M lines, quadro 26 about 36 thousand)."""
    with z.open(member) as fh:
        lines = [raw.decode("ascii") for raw in fh if raw[QUADRO_COLS] == quadro]
    return pd.read_fwf(io.StringIO("".join(lines)), widths=widths, names=names, header=None)


def load_expenses(z: zipfile.ZipFile) -> pd.DataFrame:
    """Quadro-26 records, monetary and non-monetary, with the monthly value of IBGE's memo."""
    mon = read_quadro(z, "DESPESA_INDIVIDUAL.txt", DESPESA_W, DESPESA_N).assign(source="monetary")
    nao = read_quadro(z, "SERVICO_NAO_MONETARIO_POF4.txt", NAO_MONET_W, NAO_MONET_N).assign(
        source="non_monetary")
    d = pd.concat([mon, nao], ignore_index=True)
    d["code5"] = d["V9001"] // 100
    d["mensal"] = d["V8000_DEFLA"] * d["FATOR_ANUALIZACAO"] / 12
    return d


def load_families(z: zipfile.ZipFile) -> pd.DataFrame:
    """One row per UC: UF, weight, members, disposable income per capita."""
    with z.open("MORADOR.txt") as fh:
        m = pd.read_fwf(fh, widths=MORADOR_W, names=MORADOR_N, header=None,
                        usecols=["UF"] + UC_KEY + ["PESO_FINAL", "RENDA_DISP_PC"])
    uc = (m.groupby(UC_KEY)
          .agg(UF=("UF", "first"), weight=("PESO_FINAL", "first"),
               members=("PESO_FINAL", "size"), renda_disp_pc=("RENDA_DISP_PC", "first"))
          .reset_index())
    uc["region"] = (uc["UF"] // 10).map(REGIONS)
    uc["uf"] = uc["UF"].map(UF_SIGLA)
    uc["income_decile"] = person_deciles(uc["renda_disp_pc"].to_numpy(),
                                         (uc["weight"] * uc["members"]).to_numpy())
    return uc


def person_deciles(income: np.ndarray, person_weight: np.ndarray) -> np.ndarray:
    """Decile 1-10 of each UC on the person-weighted distribution of income; a UC's members
    share its decile."""
    order = np.argsort(income, kind="stable")
    cum = np.cumsum(person_weight[order]) / person_weight.sum()
    dec = np.empty(len(income), dtype=int)
    dec[order] = np.minimum((cum * 10 - 1e-12).astype(int) + 1, 10)
    return dec


def family_values(d: pd.DataFrame, uc: pd.DataFrame) -> pd.DataFrame:
    """Monthly value per UC for each measure (0 when nothing is reported)."""
    for name, codes in MEASURES.items():
        s = d[d["code5"].isin(codes)].groupby(UC_KEY)["mensal"].sum().rename(name)
        uc = uc.merge(s, on=UC_KEY, how="left")
        uc[name] = uc[name].fillna(0.0)
    return uc


def summarise(g: pd.DataFrame, measure: str) -> dict:
    """Within-group statistics of one measure."""
    w, persons, v = g["weight"], g["weight"] * g["members"], g[measure]
    pay = v > 0
    return {
        "n_uc_sample": len(g),
        "n_uc_paying_sample": int(pay.sum()),
        "families": w.sum(),
        "persons": persons.sum(),
        "share_families_paying": 100 * w[pay].sum() / w.sum(),
        "mean_per_family": (v * w).sum() / w.sum(),
        "mean_per_paying_family": (v * w)[pay].sum() / w[pay].sum() if pay.any() else np.nan,
        "mean_per_capita": (v * w).sum() / persons.sum(),
        "share_persons_paying": 100 * persons[pay].sum() / persons.sum(),
    }


def by_group(uc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groupings = [("brasil", None), ("region", "region"), ("uf", "uf"),
                 ("income_decile", "income_decile")]
    for gtype, col in groupings:
        parts = [("Brasil", uc)] if col is None else uc.groupby(col, sort=True)
        for gname, g in parts:
            for measure in MEASURES:
                rows.append({"group_type": gtype, "group": gname, "measure": measure,
                             **summarise(g, measure)})
    return pd.DataFrame(rows)


def by_item(d: pd.DataFrame, uc: pd.DataFrame) -> pd.DataFrame:
    """Brasil, per item: 7-digit codes inside 26005, 5-digit codes for the rest of category 61."""
    d = d[d["code5"].isin(CAT61)].copy()
    d["item"] = np.where(d["code5"] == 26005, d["V9001"], d["code5"])
    total_w = uc["weight"].sum()
    w = uc.set_index(UC_KEY)["weight"]
    rows = []
    for item, g in d.groupby("item"):
        per_uc = g.groupby(UC_KEY)["mensal"].sum()
        per_uc = per_uc[per_uc > 0]
        uw = w.reindex(per_uc.index)
        rows.append({
            "item": int(item),
            "records": len(g),
            "n_uc_paying_sample": len(per_uc),
            "families_paying": uw.sum(),
            "share_families_paying": 100 * uw.sum() / total_w,
            "mean_per_family": (per_uc * uw).sum() / total_w,
            "mean_per_paying_family": (per_uc * uw).sum() / uw.sum(),
        })
    return pd.DataFrame(rows)


def published(table: str) -> dict[str, float]:
    """Brasil, region and decile rows of a published dim06 table, column "Taxas bancárias,
    juros de cheque especial e de cartão de crédito" (the second value column)."""
    ws = openpyxl.load_workbook(TABLES / f"Tabela {table}.xlsx", read_only=True,
                                data_only=True).worksheets[0]
    out, in_deciles = {}, False
    for row in ws.iter_rows(values_only=True):
        label = str(row[0]).strip() if row[0] is not None else ""
        if label.startswith("Décimos"):
            in_deciles = True
            continue
        if label == "Brasil" or label in REGIONS.values():
            out[label] = row[2]
        elif in_deciles and label.endswith("º"):
            out[f"decile_{int(label[:-1])}"] = row[2]
    return out


def check(uc: pd.DataFrame) -> pd.DataFrame:
    """Tables 6.3 (R$ per capita) and 6.2 (% of persons): contribution of each subgroup to the
    national figure, for bank_fees."""
    persons = uc["weight"] * uc["members"]
    total = persons.sum()
    spend = uc["bank_fees"] * uc["weight"]
    paying = persons.where(uc["bank_fees"] > 0, 0.0)
    masks = {"Brasil": pd.Series(True, index=uc.index)}
    masks.update({r: uc["region"] == r for r in REGIONS.values()})
    masks.update({f"decile_{k}": uc["income_decile"] == k for k in range(1, 11)})
    rows = []
    for table, num, unit in [("6.3", spend, "R$ per capita per month"),
                             ("6.2", paying, "% of persons")]:
        scale = 1.0 if table == "6.3" else 100.0
        pub = published(table)
        for row, mask in masks.items():
            computed = scale * num[mask].sum() / total
            rows.append({"table": table, "unit": unit, "row": row, "published": pub.get(row),
                         "computed": computed,
                         "diff": computed - pub[row] if row in pub else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    with zipfile.ZipFile(DADOS) as z:
        d = load_expenses(z)
        uc = load_families(z)
    print(f"quadro-26 records: {len(d):,} ({(d['source'] == 'non_monetary').sum()} "
          f"non-monetary); UCs: {len(uc):,}; families {uc['weight'].sum():,.0f}")
    uc = family_values(d, uc)

    groups = by_group(uc)
    items = by_item(d, uc)
    chk = check(uc)

    OUT.mkdir(parents=True, exist_ok=True)
    groups.to_csv(OUT / "pof2017_bank_fees_by_group.csv", index=False, float_format="%.6g")
    items.to_csv(OUT / "pof2017_bank_fees_items.csv", index=False, float_format="%.6g")
    chk.to_csv(OUT / "pof2017_bank_fees_check.csv", index=False, float_format="%.6g")

    print(chk.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"largest |diff|: {chk['diff'].abs().max():.3f}")
    brasil = groups[groups["group_type"] == "brasil"].set_index("measure")
    print(brasil[["share_families_paying", "mean_per_family", "mean_per_paying_family"]]
          .to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
