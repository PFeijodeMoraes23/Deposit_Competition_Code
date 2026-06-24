## scrape_17_cvm_disclosures.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-05-31
#
# Purpose: Structured deposit balances (BRL) for the Brazilian-listed firms in our
#          registry from CVM Dados Abertos (DFP annual + ITR quarterly financial
#          statements). This is the clean route for institutions WITHOUT a SEC
#          filer — above all Banco do Brasil — and a BRL cross-check on the SEC
#          XBRL figures from scrape_17.
#
#   Source: https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{DFP,ITR}/DADOS/
#       *_cia_aberta_YYYY.zip  ->  member *_BPP_con_YYYY.csv  (consolidated
#       balance-sheet liabilities). Deposit lines:
#           CD_CONTA 2.02.01    DS_CONTA 'Depósitos'            (total)
#           CD_CONTA 2.02.01.01 DS_CONTA 'Depósitos de Clientes' (customer)
#       VL_CONTA is in BRL at scale ESCALA_MOEDA ('MIL' => thousands).
#       ORDEM_EXERC 'ÚLTIMO' is the reference-date value (we keep only that).
#
#   NOTE on counts: CVM financial statements do NOT contain client/account counts;
#   those appear only in the free-text Formulário de Referência (item 7) and in
#   earnings releases (scrape_18/11). This script delivers the DEPOSIT side.
#
#   Output: FirmDisclosures/CVM/cvm_deposits.csv  (long format, schema = disclosure_common)
#   Cache:  FirmDisclosures/CVM/raw/<doc>_<year>.zip
#
#   CLI:
#     python scrape_17_cvm_disclosures.py                 # 2018..current, DFP+ITR
#     python scrape_17_cvm_disclosures.py --since 2015 --docs DFP
#     python scrape_17_cvm_disclosures.py --firms bb,itau,bradesco
###────────────────────────────────────────────────────────────────────────────

import os
import sys
import io
import csv
import re
import zipfile
import argparse
import logging
import datetime as dt

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

from utils.disclosure_common import http_get, new_record, write_panel, data_root
from utils.firm_registry import load_registry, _norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("scrape_23")

BASE = data_root(__file__)
OUT_DIR = os.path.join(BASE, "FirmDisclosures", "CVM")
RAW_DIR = os.path.join(OUT_DIR, "raw")
OUT_CSV = os.path.join(OUT_DIR, "cvm_deposits.csv")

CVM_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{doc}/DADOS/{doc_l}_cia_aberta_{year}.zip"
SCALE = {"MIL": 1e3, "UNIDADE": 1.0, "MILHAO": 1e6, "MILHÃO": 1e6}
# Deposits are matched by account DESCRIPTION, not code: banks use different
# liability trees (BB/Bradesco book 'Depósitos' at 2.02.01, Itaú at 2.03.01).


def download_zip(doc: str, year: int) -> bytes | None:
    os.makedirs(RAW_DIR, exist_ok=True)
    cache = os.path.join(RAW_DIR, f"{doc}_{year}.zip")
    if os.path.exists(cache) and os.path.getsize(cache) > 1000:
        with open(cache, "rb") as fh:
            return fh.read()
    url = CVM_BASE.format(doc=doc, doc_l=doc.lower(), year=year)
    code, raw = http_get(url, binary=True, timeout=180)
    if not raw:
        log.warning(f"{doc} {year}: HTTP {code}")
        return None
    with open(cache, "wb") as fh:
        fh.write(raw)
    return raw


def _classify_deposit(ds_norm: str) -> str | None:
    """Map a normalised DS_CONTA to a metric, or None if not a deposit line.
    'depositos' (exact) -> total; '...clientes' -> customer; sub-types kept too."""
    if ds_norm == "depositos":
        return "deposits"
    if "deposit" in ds_norm and "client" in ds_norm:
        return "deposits_customer"
    if "deposit" in ds_norm and any(k in ds_norm for k in
                                    ("a vista", "a prazo", "poupanca", "poupança")):
        return "deposits_" + ds_norm.replace("depositos", "").strip().replace(" ", "_")[:18]
    return None


def parse_bpp(raw: bytes, doc: str, firms: list[dict]) -> list[dict]:
    """Extract deposit lines for our firms from the consolidated BPP member(s).
    Matching is by account DESCRIPTION (code-agnostic). For each
    (firm, period, metric) we keep the largest value seen in the file (the
    aggregate line), which is robust to differing bank liability trees."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    members = [n for n in z.namelist() if "BPP_con" in n]
    best: dict[tuple, dict] = {}
    for mem in members:
        data = z.read(mem).decode("latin-1")
        rdr = csv.DictReader(io.StringIO(data), delimiter=";")
        for row in rdr:
            if row.get("ORDEM_EXERC", "").upper() not in ("ÚLTIMO", "ULTIMO"):
                continue
            metric = _classify_deposit(_norm(row.get("DS_CONTA", "")))
            if not metric:
                continue
            cnpj_digits = re.sub(r"\D", "", row.get("CNPJ_CIA", ""))
            if len(cnpj_digits) < 8:
                continue  # malformed CNPJ; cannot key to a conglomerate
            cnpj_root = cnpj_digits[:8].zfill(8)
            # Match strictly on the prudential-conglomerate CNPJ root: the CVM
            # filer for each listed incumbent is the holding/operating bank whose
            # root we validated against IF Data. A loose DENOM_CIA match would
            # wrongly grab same-name leasing/card subsidiaries (e.g. Itaú Leasing).
            firm = next((f for f in firms if cnpj_root == f["cnpj_root"]), None)
            if not firm:
                continue
            dtref = row.get("DT_REFER", "")  # YYYY-MM-DD
            try:
                y = int(dtref[:4]); mo = int(dtref[5:7]); q = (mo - 1) // 3 + 1
            except ValueError:
                continue
            scale = SCALE.get((row.get("ESCALA_MOEDA") or "MIL").upper(), 1e3)
            try:
                val = float(row.get("VL_CONTA", "")) * scale
            except ValueError:
                continue
            key = (firm["firm_key"], f"{q}Q{y}", metric)
            if key in best and best[key]["value"] >= val:
                continue
            best[key] = new_record(
                firm_key=firm["firm_key"], firm_name=row.get("DENOM_CIA", ""),
                cnpj_root=firm["cnpj_root"], segment=firm["segment"], source="cvm",
                filing_form=doc, period_year=y, period_quarter=q,
                period_label=f"{q}Q{y}", geo_scope="brazil", metric=metric,
                value=val, unit="BRL_absolute", currency="BRL", confidence="high",
                raw_context=f"{row.get('CD_CONTA','')} {row.get('DS_CONTA','')} v{row.get('VERSAO','')}",
                source_url=CVM_BASE.format(doc=doc, doc_l=doc.lower(), year=y),
                accession=f"CVM{row.get('CD_CVM','')}",
            )
    return list(best.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2018)
    ap.add_argument("--until", type=int, default=dt.date.today().year)
    ap.add_argument("--docs", type=str, default="DFP,ITR", help="DFP and/or ITR")
    ap.add_argument("--firms", type=str, default="")
    args = ap.parse_args()

    firms = load_registry(BASE)
    if args.firms:
        want = {x.strip() for x in args.firms.split(",")}
        firms = [f for f in firms if f["firm_key"] in want]
    docs = [d.strip().upper() for d in args.docs.split(",") if d.strip()]

    records, seen = [], set()
    for doc in docs:
        for year in range(args.since, args.until + 1):
            raw = download_zip(doc, year)
            if not raw:
                continue
            recs = parse_bpp(raw, doc, firms)
            # DFP (annual, Q4) shouldn't overwrite ITR Q4; dedup on (firm,period,metric,doc-priority)
            for r in recs:
                key = (r["firm_key"], r["period_label"], r["metric"])
                if key in seen:
                    continue
                seen.add(key)
                records.append(r)
            log.info(f"{doc} {year}: +{len(recs)} deposit rows "
                     f"({len({r['firm_key'] for r in recs})} firms)")

    n = write_panel(records, OUT_CSV)
    log.info(f"Wrote {n} records -> {OUT_CSV}")
    if records:
        import pandas as pd
        df = pd.DataFrame(records)
        print("\n--- latest TOTAL deposits (2.02.01) by firm, BRL bn ---")
        for fk, g in df[df.metric == "deposits"].groupby("firm_key"):
            last = g.sort_values("period_label").iloc[-1]
            print(f"   {fk:14s} {last['period_label']}: {last['value']/1e9:8.1f}  ({last['firm_name']})")
        cust = df[df.metric == "deposits_customer"]
        if not cust.empty:
            print("\n--- customer deposits (2.02.01.01) where broken out, BRL bn ---")
            for fk, g in cust.groupby("firm_key"):
                last = g.sort_values("period_label").iloc[-1]
                print(f"   {fk:14s} {last['period_label']}: {last['value']/1e9:8.1f}")


if __name__ == "__main__":
    main()
