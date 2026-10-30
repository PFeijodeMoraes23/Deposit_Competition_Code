###─────────────────────────────────────────────────────────────────────────────
# fees/build_fee_paper_numbers.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   The fee numbers V_Friction Section 7.2 quotes, written as a sidecar for
#   make_paper_numbers.py (which turns them into \pn keys; the collector is the coding
#   agent's). Three sources, all built earlier by fees/ scripts:
#     POF 2017-18         pof2017_bank_fees_by_group.csv (build_pof_bank_fees.py): Brasil,
#                         account-keeping fee (item 26005)
#     consumidor.gov.br   consumidor_fee_complaints_conglomerate_quarter.csv
#                         (build_consumidor_firm_conglomerates.py): fee disputes over account
#                         complaints, POOLED = total disputes / total account complaints
#     listed prices       stage 7's conglomerate value of Package I (F 1501) in Q4 (price in
#                         force on 31 December), from fees/panel_fee_merge.py's own functions
#                         (listed_fee_institution_quarters, load_tarifa_listed_prices), read
#                         only: the head institution's price, or the median of the members'
#                         positive prices when the head does not list it
#
#   Bank groups are fixed by utils.firm_registry keys and put on prudential codes with
#   utils.conglomerates.build_cnpj_conglomerate_map, the map the complaint tables use:
#     big five   caixa, bradesco, bb, itau, santander_br
#     digital    nubank, inter, c6 (PicPay and Mercado Pago are left out: almost none of
#                their complaints are filed under accounts)
#
# OUTPUT  utils.paths.TARIFAS_PROC / fee_paper_numbers.json
#   {key: {value, definition, unit, source, script, run_at}}; values unrounded, counts as
#   integers. The 2016 big-five share is printed for context and has no key.
#
# USAGE
#   python -I fees/build_fee_paper_numbers.py
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/, fees/

from utils import paths
from utils.conglomerates import build_cnpj_conglomerate_map
from utils.firm_registry import FIRMS
from fees.panel_fee_merge import (conglomerate_heads, listed_fee_institution_quarters,
                                  load_tarifa_listed_prices)

PROC = paths.TARIFAS_PROC
OUT = PROC / "fee_paper_numbers.json"
POF = PROC / "pof2017_bank_fees_by_group.csv"
DISPUTES = PROC / "consumidor_fee_complaints_conglomerate_quarter.csv"
LISTED = PROC / "account_fee_listed_history.csv"
SCRIPT = "fees/build_fee_paper_numbers.py"

BIG5 = ["caixa", "bradesco", "bb", "itau", "santander_br"]
DIGITAL = ["nubank", "inter", "c6"]
PKG1 = ("F", "1501")
YEARS = (2016, 2024)


def rel(p: Path) -> str:
    """Path relative to the Open-Finance folder, as the sidecar records it."""
    return p.resolve().relative_to(paths.OPEN_FINANCE.resolve()).as_posix()


def codes_of(firm_keys: list[str]) -> dict[str, str]:
    """firm_key -> prudential code, through the registry's CNPJ root."""
    cmap = build_cnpj_conglomerate_map()
    by_key = {f["firm_key"]: f for f in FIRMS}
    out = {}
    for k in firm_keys:
        hit = cmap.get(int(by_key[k]["cnpj_root"]))
        code = hit[0] if hit and hit[0] is not None else by_key[k].get("cong_prud")
        if not code:
            raise ValueError(f"no prudential code for {k}")
        out[k] = code
    return out


def pof_numbers() -> dict:
    g = pd.read_csv(POF)
    r = g[(g["group_type"] == "brasil") & (g["measure"] == "acct_keeping")].iloc[0]
    return {"share_paying": r["share_families_paying"] / 100,
            "mean_payer": r["mean_per_paying_family"],
            "mean_family": r["mean_per_family"]}


def dispute_numbers(big5: dict, digital: dict) -> dict:
    q = pd.read_csv(DISPUTES)
    q = q[q["year"].between(2016, 2024)]
    out = {}
    for name, codes in (("big5", big5), ("digital", digital)):
        d = q[q["cod_cong_prudencial"].isin(codes.values())]
        missing = set(codes.values()) - set(d["cod_cong_prudencial"])
        if missing:
            raise ValueError(f"{name}: codes absent from {DISPUTES.name}: {missing}")
        out[name] = (int(d["n_fee_account"].sum()), int(d["n_account"].sum()))
        by_year = d.groupby("year")[["n_fee_account", "n_account"]].sum()
        out[f"{name}_by_year"] = {int(y): (int(r.n_fee_account), int(r.n_account))
                                  for y, r in by_year.iterrows()}
    return out


def listed_pkg1(big5: dict) -> dict:
    """Package I of each big-five conglomerate in Q4 of YEARS (price in force on 31 Dec),
    computed by stage 7's own load_tarifa_listed_prices: the head institution's price, or
    the median of the members' positive prices when the head does not list it. The rule
    each bank took, and the price at the registry's operating bank, are kept for the log."""
    iq = listed_fee_institution_quarters()
    sub = iq[(iq["pessoa"] == PKG1[0]) & (iq["codigo_servico"] == PKG1[1])
             & iq["cod_cong_prudencial"].isin(big5.values())
             & (iq["quarter"] == 4) & iq["year"].isin(YEARS)].copy()
    wide = load_tarifa_listed_prices(sub).set_index(["cod_cong_prudencial", "year"])
    heads = dict(zip(big5, conglomerate_heads(pd.Series(list(big5.values()))).tolist()))
    roots = {f["firm_key"]: f["cnpj_root"] for f in FIRMS}
    out = {}
    for y in YEARS:
        prices, detail = {}, {}
        for k, code in big5.items():
            if (code, y) not in wide.index:
                raise ValueError(f"{y}: no Package I price for {k} ({code})")
            prices[k] = float(wide.at[(code, y), "listed_fee_pkg1_pf"])
            m = sub[(sub["cod_cong_prudencial"] == code) & (sub["year"] == y)]
            by_cnpj = dict(zip(m["cnpj8"], m["price"].astype(float)))
            detail[k] = {"rule": "head" if heads[k] in by_cnpj else "positive_median",
                         "head": heads[k], "operating_bank": roots[k],
                         "operating_bank_price": by_cnpj.get(roots[k]),
                         "members_listing": by_cnpj}
        out[y] = {"median": float(pd.Series(prices).median()), "prices": prices,
                  "detail": detail}
    return out


def entry(value, definition: str, unit: str, source: Path, run_at: str) -> dict:
    return {"value": value, "definition": definition, "unit": unit, "source": rel(source),
            "script": SCRIPT, "run_at": run_at}


def main() -> None:
    run_at = datetime.now().isoformat(timespec="seconds")
    big5, digital = codes_of(BIG5), codes_of(DIGITAL)
    pof = pof_numbers()
    dis = dispute_numbers(big5, digital)
    lst = listed_pkg1(big5)

    b_fee, b_n = dis["big5"]
    d_fee, d_n = dis["digital"]
    b24_fee, b24_n = dis["big5_by_year"][2024]
    b16_fee, b16_n = dis["big5_by_year"][2016]
    pof_note = "POF 2017-18 (IBGE), one cross-section of what families report paying"
    dis_note = ("consumidor.gov.br complaints: disputes naming a tariff, on the subject "
                "'conta corrente / salário / poupança / conta aposentadoria', over all "
                "complaints on that subject; labels mapped to prudential conglomerates (10-27)")
    lst_note = ("BCB monthly tariff census, price in force on 31 December, stage 7's "
                "conglomerate value (the head institution's price, or the median of the "
                "members' positive prices when the head does not list it); a listed price "
                "is a ceiling, not the price charged")

    keys = {
        "fees.pof.acct.share_paying": entry(
            pof["share_paying"], f"Share of families with any expense on the account-keeping fee "
            f"(item 26005), Brasil; {pof_note}", "share (0-1)", POF, run_at),
        "fees.pof.acct.mean_payer": entry(
            pof["mean_payer"], f"Mean monthly expense on the account-keeping fee per paying "
            f"family, Brasil; {pof_note}", "R$ per month, 2018-01-15 prices", POF, run_at),
        "fees.pof.acct.mean_family": entry(
            pof["mean_family"], f"Mean monthly expense on the account-keeping fee per family "
            f"(all families), Brasil; {pof_note}", "R$ per month, 2018-01-15 prices", POF, run_at),
        "fees.disputes.acct_share.big5": entry(
            b_fee / b_n, f"Fee disputes over account complaints, pooled 2016-2024 (total "
            f"disputes / total account complaints), Caixa, Bradesco, BB, Itaú and Santander; "
            f"{dis_note}", "share (0-1)", DISPUTES, run_at),
        "fees.disputes.acct_share.big5.n": entry(
            b_n, "Account complaints behind fees.disputes.acct_share.big5, 2016-2024",
            "complaints", DISPUTES, run_at),
        "fees.disputes.acct_share.digital": entry(
            d_fee / d_n, f"Fee disputes over account complaints, pooled 2016-2024, Nubank, "
            f"Inter and C6 (PicPay and Mercado Pago excluded: almost none of their complaints "
            f"are filed under accounts); small counts in the early years; {dis_note}",
            "share (0-1)", DISPUTES, run_at),
        "fees.disputes.acct_share.digital.n": entry(
            d_n, "Account complaints behind fees.disputes.acct_share.digital, 2016-2024",
            "complaints", DISPUTES, run_at),
        "fees.listed.pkg1.big5.median.2016": entry(
            lst[2016]["median"], f"Median over Caixa, Bradesco, BB, Itaú and Santander of the "
            f"listed Package I price (code 1501, individuals), December 2016; {lst_note}",
            "R$ per month", LISTED, run_at),
        "fees.listed.pkg1.big5.median.2024": entry(
            lst[2024]["median"], f"Same as fees.listed.pkg1.big5.median.2016, December 2024; "
            f"{lst_note}", "R$ per month", LISTED, run_at),
        "fees.disputes.acct_share.big5.2024": entry(
            b24_fee / b24_n, f"Fee disputes over account complaints in 2024 alone, Caixa, "
            f"Bradesco, BB, Itaú and Santander; {dis_note}", "share (0-1)", DISPUTES, run_at),
    }
    OUT.write_text(json.dumps(keys, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"codes: big five {big5}; digital {digital}")
    for k, e in keys.items():
        print(f"{k:40s} {e['value']}")
    print(f"counts: big five {b_fee}/{b_n}; digital {d_fee}/{d_n}; big five 2024 {b24_fee}/{b24_n}")
    print(f"context (no key): big-five share 2016 = {b16_fee / b16_n:.4f} ({b16_fee}/{b16_n})")
    for y in YEARS:
        print(f"Package I, 31 Dec {y}: {lst[y]['prices']}")
        for k, dt in lst[y]["detail"].items():
            print(f"    {k}: {dt}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
