## scrape_18_parent_disclosures.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-05-31
#
# Purpose: Customer/active-user counts for the digital/payment firms that have NO
#          standalone SEC or CVM filer and instead surface inside a PARENT's
#          filings or only in press releases:
#            - Mercado Pago      -> inside MercadoLibre (MELI) SEC filings  [AUTOMATED]
#            - C6 Bank           -> JPMorgan-controlled; press / IR only    [MANUAL]
#            - PicPay            -> J&F/Original group; press / IR only      [MANUAL]
#
#   Mercado Pago is recoverable: MELI's 10-K/10-Q report 'fintech monthly active
#   users' and break Brazil out in the segment discussion. We extract those with
#   the geo tag (Brazil vs consolidated/Latam), exactly as scrape_17 does for NU.
#   The reduced-form deposit object for Mercado Pago is 'funds payable to
#   customers' (not a deposit liability) — flagged as such.
#
#   C6 / PicPay publish customer counts only in press releases and the financial
#   pages of their controllers; there is no machine-readable feed. We therefore
#   seed a manual-augmentation CSV (panel schema) with example rows + the IR/press
#   source URLs; fill it as those numbers are published.
#
#   Outputs (under FirmDisclosures/Parent/):
#     parent_disclosures.csv          — automated (Mercado Pago) panel rows
#     raw/mercadopago/<accn>.txt       — archived MELI filing text
#     manual_parent_counts.csv         — seeded manual template (C6, PicPay)
#
#   CLI:
#     python scrape_18_parent_disclosures.py
#     python scrape_18_parent_disclosures.py --since 2019 --max-filings 30
###────────────────────────────────────────────────────────────────────────────

import os
import sys
import re
import argparse
import logging

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

from utils.disclosure_common import (
    http_get, strip_html, parse_number, new_record, write_panel, data_root,
)
from utils.firm_registry import load_registry
# reuse EDGAR plumbing
from scrape_17_edgar_disclosures import fetch_submissions, _pick_documents, EARNINGS_FORMS

# MELI publishes its KPI tables in 8-K earnings exhibits (ex99.1), not in the
# 10-K/10-Q body, so the parent scan must include 8-K.
PARENT_FORMS = set(EARNINGS_FORMS) | {"8-K", "8-K/A"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("scrape_18")

BASE = data_root(__file__)
OUT_DIR = os.path.join(BASE, "FirmDisclosures", "Parent")
RAW_DIR = os.path.join(OUT_DIR, "raw")
OUT_CSV = os.path.join(OUT_DIR, "parent_disclosures.csv")
MANUAL_CSV = os.path.join(OUT_DIR, "manual_parent_counts.csv")

# MELI KPI table line, e.g. "Fintech monthly active users 78 61 78 61" — bare
# integers in MILLIONS, first integer = current period, consolidated (Latam).
_MAU_RX = re.compile(
    r"fintech\s+monthly\s+active\s+users[^\d]{0,40}(\d{2,4})(?:[.,](\d))?", re.I)
# generic fallback for prose like "60.8 million unique fintech active users"
_USER_RX = re.compile(
    r"(\d[\d.,]*)\s*(million|millions|mn)\s*"
    r"(unique\s+)?(fintech\s+)?(monthly\s+)?active\s+(users|customers)\b", re.I)


def _reported_period(form: str, rep: str) -> tuple[int, int]:
    """Map a filing's reference date to the (year, quarter) it REPORTS.
    For 8-K earnings the date is the announcement (≈6 weeks after quarter-end),
    so a Feb/May/Aug/Nov release reports Q4(prior)/Q1/Q2/Q3. For period filings
    (10-Q/10-K/6-K) the date is the period end and is used directly."""
    y = int(rep[:4]); mo = int(rep[5:7])
    if form.startswith("8-K"):
        idx = (mo - 1) // 3            # 0:Jan-Mar 1:Apr-Jun 2:Jul-Sep 3:Oct-Dec
        if idx == 0:
            return y - 1, 4            # Feb release -> Q4 prior year
        return y, idx                  # May->Q1, Aug->Q2, Nov->Q3
    return y, (mo - 1) // 3 + 1


def extract_mercadopago(firm: dict, since: int, max_filings: int) -> list[dict]:
    sub = fetch_submissions(firm["sec_cik"])
    if not sub:
        return []
    rows = [r for r in sub["_rows"]
            if r.get("form") in PARENT_FORMS
            and (r.get("reportDate") or r.get("filingDate", ""))[:4].isdigit()
            and int((r.get("reportDate") or r.get("filingDate"))[:4]) >= since]
    rows.sort(key=lambda r: r.get("reportDate") or r.get("filingDate"), reverse=True)
    rows = rows[:max_filings]
    os.makedirs(os.path.join(RAW_DIR, "mercadopago"), exist_ok=True)
    best = {}  # period_label -> record (one fintech-MAU obs per quarter, best confidence)
    for r in rows:
        accn = r.get("accessionNumber")
        rep = r.get("reportDate") or r.get("filingDate")
        y, q = _reported_period(r.get("form", ""), rep)
        # Prefer the press-release exhibit (ex99/ex991); it carries the KPI table.
        docs = _pick_documents(firm["sec_cik"], accn)
        docs.sort(key=lambda d: (("ex99" not in d[0].lower()), -len(d[0])))
        plain = ""
        for name, url in docs:
            c, body = http_get(url)
            if body:
                plain = strip_html(body)
                if "fintech monthly active users" in plain.lower():
                    break  # found the KPI exhibit
        if not plain:
            continue
        try:
            with open(os.path.join(RAW_DIR, "mercadopago", f"{accn}_{rep}.txt"), "w",
                      encoding="utf-8") as fh:
                fh.write(plain)
        except Exception:
            pass
        val, conf, ctx = None, None, ""
        m = _MAU_RX.search(plain)
        if m:
            whole = m.group(1); frac = m.group(2)
            num = float(f"{whole}.{frac}") if frac else float(whole)
            if 10 <= num <= 500:                      # millions, sane range
                val, conf = num * 1e6, "medium"
                ctx = re.sub(r"\s+", " ", plain[m.start():m.end() + 20]).strip()[:160]
        if val is None:                                # prose fallback
            mm = _USER_RX.search(plain)
            if mm:
                val = parse_number(mm.group(1)) * 1e6
                conf = "low"
                ctx = re.sub(r"\s+", " ", plain[max(0, mm.start()-30):mm.end()+20]).strip()[:160]
        if val is None or not (1e6 <= val <= 3e8):
            continue
        # MELI reports fintech MAU CONSOLIDATED (Latam); Brazil MAU is not broken
        # out separately, so scope='latam'. Combine with the Brazil revenue share
        # from the 10-Q segment note to localise (documented limitation).
        rec = new_record(
            firm_key=firm["firm_key"], firm_name="MercadoLibre / Mercado Pago (fintech MAU)",
            cnpj_root=firm["cnpj_root"], segment=firm["segment"], source="parent_filing",
            filing_form=r.get("form", ""), period_year=y, period_quarter=q,
            period_label=f"{q}Q{y}", geo_scope="latam", metric="customers_active",
            value=val, unit="count", currency="", confidence=conf, raw_context=ctx,
            source_url=f"https://www.sec.gov/Archives/edgar/data/{int(firm['sec_cik'])}/{accn.replace('-','')}/",
            accession=accn)
        # keep one per period, preferring higher confidence then larger value
        prev = best.get(rec["period_label"])
        if prev is None or (conf == "medium" and prev["confidence"] != "medium") \
           or (conf == prev["confidence"] and val > prev["value"]):
            best[rec["period_label"]] = rec
    out = list(best.values())
    log.info(f"[mercadopago] {len(out)} fintech-MAU obs from {len(rows)} MELI filings")
    return out


def seed_manual_template() -> None:
    if os.path.exists(MANUAL_CSV):
        log.info(f"manual template present (not overwritten): {MANUAL_CSV}")
        return
    reg = {f["firm_key"]: f for f in load_registry(BASE)}
    examples = []
    for key, url in [("c6", "https://www.c6bank.com.br/relacoes-com-investidores"),
                     ("picpay", "https://www.picpay.com/imprensa")]:
        f = reg.get(key, {"firm_key": key, "display_name": key, "cnpj_root": "", "segment": "digital"})
        examples.append(new_record(
            firm_key=key, firm_name=f.get("display_name", key), cnpj_root=f.get("cnpj_root", ""),
            segment=f.get("segment", "digital"), source="press", filing_form="press_release",
            period_year=2024, period_quarter=4, period_label="4Q2024", geo_scope="brazil",
            metric="customers_total", value=None, unit="count", currency="", confidence="manual",
            raw_context="EXAMPLE — fill total customers from press release", source_url=url,
            accession=""))
    write_panel(examples, MANUAL_CSV)
    log.info(f"seeded manual template (C6, PicPay) -> {MANUAL_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2019)
    ap.add_argument("--max-filings", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)

    reg = {f["firm_key"]: f for f in load_registry(BASE)}
    records = []
    mp = reg.get("mercadopago")
    if mp and mp.get("sec_cik"):
        records += extract_mercadopago(mp, args.since, args.max_filings)

    n = write_panel(records, OUT_CSV)
    log.info(f"Wrote {n} automated records -> {OUT_CSV}")
    seed_manual_template()
    log.info("DONE. Mercado Pago automated from MELI; C6/PicPay via manual_parent_counts.csv.")


if __name__ == "__main__":
    main()
