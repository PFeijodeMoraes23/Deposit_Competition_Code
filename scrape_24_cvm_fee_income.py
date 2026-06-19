## scrape_24_cvm_fee_income.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-06-19
#
# Purpose: Fee and service-income lines from consolidated CVM DRE (income
#          statements) for the Brazilian-listed banks in our firm registry.
#          Mirrors scrape_23 but reads *DRE_con* members instead of *BPP_con*.
#
# Why:
#   COSIF covers all institutions but only via aggregated account 717xxx (total
#   service fees).  CVM DFP/ITR DRE filings break revenue into sub-lines at the
#   firm's own discretion: "Receitas de Tarifas", "Receitas de Prestação de
#   Serviços", "Tarifas Bancárias", etc.  Matching both on ~8 banks × 40 quarters
#   ≈ 320 filing-quarters.
#
#   Key DRE patterns (DS_CONTA, normalised to lower-case, accent-free):
#     "receita de tarifas"               -> metric = fee_revenue
#     "prestacao de servicos"            -> metric = service_revenue (broader)
#     "receitas de servicos" / similar   -> metric = service_revenue
#     "tarifas bancarias"                -> metric = fee_revenue
#     sum of DRE lines that sum to total revenue
#
#   Output: FirmDisclosures/CVM/cvm_fee_income.csv
#   Cache:  FirmDisclosures/CVM/raw/<doc>_<year>.zip  (shared with scrape_23)
#
#   CLI:
#     python scrape_24_cvm_fee_income.py                 # 2018..current, DFP+ITR
#     python scrape_24_cvm_fee_income.py --since 2015 --docs DFP
#     python scrape_24_cvm_fee_income.py --firms bb,itau,bradesco --dump-accounts
#
#   --dump-accounts: instead of saving the panel, print all unique DRE account
#   descriptions found for the matched firms (for exploratory mapping).
###─────────────────────────────────────────────────────────────────────────────

import os
import sys
import io
import csv
import re
import zipfile
import argparse
import logging
import datetime as dt
import unicodedata

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
log = logging.getLogger("scrape_24")

BASE = data_root(__file__)
OUT_DIR = os.path.join(BASE, "FirmDisclosures", "CVM")
RAW_DIR = os.path.join(OUT_DIR, "raw")
OUT_CSV = os.path.join(OUT_DIR, "cvm_fee_income.csv")

CVM_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{doc}/DADOS/{doc_l}_cia_aberta_{year}.zip"
SCALE = {"MIL": 1e3, "UNIDADE": 1.0, "MILHAO": 1e6, "MILHÃO": 1e6}


def _accent_free(s: str) -> str:
    """Remove diacritics and lower-case."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


# Keywords for fee / service-income DRE lines (applied to accent-free, lower-case DS_CONTA)
_FEE_KEYWORDS = [
    "receita de tarifa",
    "receitas de tarifa",
    "tarifas bancaria",
    "tarifa bancaria",
    "receita de prestacao de servico",
    "receitas de prestacao de servico",
    "prestacao de servico",        # broader; classified separately
    "receita de servico",
    "receitas de servico",
    "comissoes e tarifas",
    "rendas de tarifas",
    "rendas de servico",
]

_NARROW_KEYWORDS = [k for k in _FEE_KEYWORDS if "tarifa" in k]
_BROAD_KEYWORDS  = [k for k in _FEE_KEYWORDS if "tarifa" not in k]


def _classify_fee(ds_norm: str) -> str | None:
    """
    Map an accent-free lower-case DS_CONTA to a metric, or None if not relevant.
    Returns one of:
      'fee_revenue'      — narrowly tagged as tariff/fee line
      'service_revenue'  — broader service/commission income
    """
    for kw in _NARROW_KEYWORDS:
        if kw in ds_norm:
            return "fee_revenue"
    for kw in _BROAD_KEYWORDS:
        if kw in ds_norm:
            return "service_revenue"
    return None


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


def parse_dre(raw: bytes, doc: str, firms: list[dict],
              dump_accounts: bool = False) -> tuple[list[dict], set[str]]:
    """
    Extract fee/service income lines for our firms from DRE_con member(s).

    Returns (records, account_descriptions_seen).
    If dump_accounts=True, records is empty but account_descriptions_seen is
    populated with every unique DS_CONTA found for any matched firm (for mapping).
    """
    z = zipfile.ZipFile(io.BytesIO(raw))
    members = [n for n in z.namelist() if "DRE_con" in n]
    if not members:
        return [], set()

    best: dict[tuple, dict] = {}
    all_accounts: set[str] = set()

    for mem in members:
        data = z.read(mem).decode("latin-1")
        rdr = csv.DictReader(io.StringIO(data), delimiter=";")
        for row in rdr:
            if row.get("ORDEM_EXERC", "").upper() not in ("ÚLTIMO", "ULTIMO"):
                continue

            cnpj_digits = re.sub(r"\D", "", row.get("CNPJ_CIA", ""))
            if len(cnpj_digits) < 8:
                continue
            cnpj_root = cnpj_digits[:8].zfill(8)
            firm = next((f for f in firms if cnpj_root == f["cnpj_root"]), None)
            if not firm:
                continue

            ds_raw = row.get("DS_CONTA", "")
            ds_norm = _accent_free(ds_raw)

            if dump_accounts:
                all_accounts.add(f"{row.get('CD_CONTA','')} | {ds_raw}")
                continue

            metric = _classify_fee(ds_norm)
            if not metric:
                continue

            dtref = row.get("DT_REFER", "")
            try:
                y = int(dtref[:4]); mo = int(dtref[5:7]); q = (mo - 1) // 3 + 1
            except ValueError:
                continue

            scale = SCALE.get((row.get("ESCALA_MOEDA") or "MIL").upper(), 1e3)
            try:
                val = float(row.get("VL_CONTA", "")) * scale
            except ValueError:
                continue

            # DRE values can be negative (expenses); keep sign, filter near-zero noise
            if abs(val) < 1e4:
                continue

            key = (firm["firm_key"], f"{q}Q{y}", metric, ds_norm[:40])
            if key in best:
                continue
            best[key] = new_record(
                firm_key=firm["firm_key"], firm_name=row.get("DENOM_CIA", ""),
                cnpj_root=firm["cnpj_root"], segment=firm["segment"], source="cvm_dre",
                filing_form=doc, period_year=y, period_quarter=q,
                period_label=f"{q}Q{y}", geo_scope="brazil", metric=metric,
                value=val, unit="BRL_absolute", currency="BRL", confidence="medium",
                raw_context=f"{row.get('CD_CONTA','')} {ds_raw} v{row.get('VERSAO','')}",
                source_url=CVM_BASE.format(doc=doc, doc_l=doc.lower(), year=y),
                accession=f"CVM{row.get('CD_CVM','')}",
            )

    return list(best.values()), all_accounts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2018)
    ap.add_argument("--until", type=int, default=dt.date.today().year)
    ap.add_argument("--docs", type=str, default="DFP,ITR")
    ap.add_argument("--firms", type=str, default="")
    ap.add_argument("--dump-accounts", action="store_true",
                    help="Print all DRE account descriptions for matched firms (for mapping).")
    args = ap.parse_args()

    firms = load_registry(BASE)
    if args.firms:
        want = {x.strip() for x in args.firms.split(",")}
        firms = [f for f in firms if f["firm_key"] in want]
    docs = [d.strip().upper() for d in args.docs.split(",") if d.strip()]

    log.info(f"Firms: {[f['firm_key'] for f in firms]}")
    log.info(f"Docs: {docs}  |  Years: {args.since}-{args.until}")

    records, seen, all_accounts = [], set(), set()
    for doc in docs:
        for year in range(args.since, args.until + 1):
            raw = download_zip(doc, year)
            if not raw:
                continue
            recs, accts = parse_dre(raw, doc, firms, dump_accounts=args.dump_accounts)
            all_accounts.update(accts)
            for r in recs:
                key = (r["firm_key"], r["period_label"], r["metric"],
                       r["raw_context"][:30])
                if key in seen:
                    continue
                seen.add(key)
                records.append(r)
            log.info(f"{doc} {year}: +{len(recs)} fee-income rows "
                     f"({len({r['firm_key'] for r in recs})} firms)")

    if args.dump_accounts:
        print(f"\n=== DRE account descriptions for matched firms ({len(all_accounts)} unique) ===")
        for a in sorted(all_accounts):
            print(f"  {a}")
        return

    n = write_panel(records, OUT_CSV)
    log.info(f"Wrote {n} records -> {OUT_CSV}")

    if records:
        try:
            import pandas as pd
            df = pd.DataFrame(records)
            print("\n=== Fee revenue (fee_revenue) — latest quarter per firm, BRL bn ===")
            fee_df = df[df["metric"] == "fee_revenue"]
            if not fee_df.empty:
                for fk, g in fee_df.groupby("firm_key"):
                    last = g.sort_values("period_label").iloc[-1]
                    print(f"   {fk:14s} {last['period_label']}: "
                          f"{last['value']/1e9:7.3f} bn  | {last['raw_context'][:60]}")
            print("\n=== Service revenue (service_revenue) — latest quarter per firm ===")
            svc_df = df[df["metric"] == "service_revenue"]
            if not svc_df.empty:
                for fk, g in svc_df.groupby("firm_key"):
                    last = g.sort_values("period_label").iloc[-1]
                    print(f"   {fk:14s} {last['period_label']}: "
                          f"{last['value']/1e9:7.3f} bn  | {last['raw_context'][:60]}")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
