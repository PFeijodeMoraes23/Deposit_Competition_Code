## scrape_17_edgar_disclosures.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-05-31
#
# Purpose: Harvest customer/account counts and deposit volumes for the SEC-listed
#          firms in our prudential-conglomerate set, to build the account-share-
#          vs-volume-share (extensive-vs-intensive margin) diagnostic motivating
#          the V_Main.tex appendix `app:extensions:intensive`.
#
#   Firms with a SEC filer (foreign private issuers / domestic):
#       Nu Holdings (NU), PagSeguro (PAGS), MercadoLibre/Mercado Pago (MELI),
#       Inter & Co (INTR), StoneCo (STNE), Itaú (ITUB), Bradesco (BBD),
#       Santander Brasil (BSBR).
#
#   GEOGRAPHIC SCOPE — the key caveat you flagged:
#     SEC financial statements are CONSOLIDATED. For NU and MELI that means
#     Latam-wide (Brazil + Mexico + Colombia / multi-country). These firms break
#     Brazil out only in segment notes / investor decks. PAGS, STNE, INTR and the
#     Brazilian ADRs (ITUB/BBD/BSBR) are effectively Brazil-centric, so
#     consolidated ≈ Brazil. Every record is tagged `geo_scope` accordingly; for
#     latam firms the consolidated deposit/customer figure is an UPPER BOUND on
#     the Brazilian object and must be combined with the Brazil-share from the
#     segment data (parent-filing scraper / investor decks).
#
#   STRATEGY (hybrid, because the two metrics live in different places):
#     - DEPOSITS  -> SEC XBRL companyfacts (structured, reliable). We prefer
#                    'DepositsFromCustomers'; fall back to other deposit tags.
#     - CUSTOMERS -> text extraction from 6-K/20-F/10-K/10-Q exhibits (the count
#                    is never an XBRL fact). Best-effort + raw text archived to
#                    disk for notebook refinement; flagged confidence='low'.
#
#   Output:
#     FirmDisclosures/SEC/edgar_disclosures.csv     (long-format panel)
#     FirmDisclosures/SEC/raw/<firm>/<accession>.txt (archived filing text)
#
#   CLI:
#     python scrape_17_edgar_disclosures.py                 # all SEC firms, since 2018
#     python scrape_17_edgar_disclosures.py --firms nubank,pagseguro
#     python scrape_17_edgar_disclosures.py --since 2020 --max-filings 40
#     python scrape_17_edgar_disclosures.py --deposits-only # XBRL only, no text
#     python scrape_17_edgar_disclosures.py --list
###────────────────────────────────────────────────────────────────────────────

import os
import sys
import re
import json
import argparse
import logging

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

from utils.disclosure_common import (
    http_get, get_json, strip_html, parse_number, new_record, write_panel,
    data_root, now_iso,
)
from utils.firm_registry import load_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("scrape_17")

BASE = data_root(__file__)
OUT_DIR = os.path.join(BASE, "FirmDisclosures", "SEC")
RAW_DIR = os.path.join(OUT_DIR, "raw")
OUT_CSV = os.path.join(OUT_DIR, "edgar_disclosures.csv")

# Forms that can carry earnings tables / financials for these filers.
EARNINGS_FORMS = {"6-K", "20-F", "40-F", "10-K", "10-Q", "10-K/A", "20-F/A"}

# Deposit XBRL tags in priority order (IFRS for NU/ITUB/BBD/BSBR; US-GAAP for the
# domestic-style filers). First match with USD/BRL units wins.
DEPOSIT_TAGS = [
    "DepositsFromCustomers", "CurrentDepositsFromCustomers",
    "Deposits", "InterestBearingDepositLiabilities",
    "NoninterestBearingDepositLiabilities", "DepositsFromBanks",
]

# Customer/account count patterns (en + pt). Number then unit then noun.
_CUST_RX = re.compile(
    r"(\d[\d.,]*)\s*(million|millions|mn|mm|mi|thousand|bn|billion)?\s*"
    r"(active\s+|monthly\s+active\s+|total\s+)?"
    r"(customers|clients|accountholders|account\s?holders|users|active\s+users|"
    r"correntistas|clientes|contas)\b",
    re.I,
)
_GEO_RX = re.compile(r"\b(brazil|brasil|mexico|méxico|colombia|colômbia|consolidated|latam|group)\b", re.I)


# ----------------------------------------------------------------------------
def fetch_submissions(cik: str) -> dict | None:
    """Full submissions history (recent block + any overflow files)."""
    cik10 = str(int(cik)).zfill(10)
    code, d = get_json(f"https://data.sec.gov/submissions/CIK{cik10}.json")
    if not d:
        log.warning(f"submissions {cik10}: HTTP {code}")
        return None
    recent = d["filings"]["recent"]
    cols = list(recent.keys())
    rows = [dict(zip(cols, vals)) for vals in zip(*[recent[c] for c in cols])]
    for extra in d["filings"].get("files", []):
        _, ed = get_json(f"https://data.sec.gov/submissions/{extra['name']}")
        if ed:
            rows += [dict(zip(ed.keys(), v)) for v in zip(*ed.values())]
    d["_rows"] = rows
    return d


def deposits_from_xbrl(firm: dict) -> list[dict]:
    """Pull deposit balances from XBRL companyfacts. Returns long records."""
    cik10 = str(int(firm["sec_cik"])).zfill(10)
    code, facts = get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")
    if not facts:
        log.warning(f"[{firm['firm_key']}] companyfacts HTTP {code}")
        return []
    allfacts = facts.get("facts", {})
    # locate the best deposit tag across taxonomies
    chosen = None
    for tag in DEPOSIT_TAGS:
        for tax in allfacts:
            if tag in allfacts[tax]:
                units = allfacts[tax][tag].get("units", {})
                cur = "USD" if "USD" in units else ("BRL" if "BRL" in units else None)
                if cur:
                    chosen = (tax, tag, cur, units[cur]); break
        if chosen:
            break
    if not chosen:
        log.info(f"[{firm['firm_key']}] no deposit XBRL tag found")
        return []
    tax, tag, cur, entries = chosen
    geo = "consolidated" if firm["country_scope"] == "latam" else "brazil"
    seen, out = set(), []
    for e in entries:
        end = e.get("end"); val = e.get("val")
        if end is None or val is None:
            continue
        y = int(end[:4]); mo = int(end[5:7]); q = (mo - 1) // 3 + 1
        fp = e.get("fp", "")
        key = (end, e.get("form"))
        if key in seen:
            continue
        seen.add(key)
        out.append(new_record(
            firm_key=firm["firm_key"], firm_name=firm["display_name"],
            cnpj_root=firm["cnpj_root"], segment=firm["segment"], source="sec_edgar",
            filing_form=e.get("form", ""), period_year=y,
            period_quarter=(q if fp != "FY" else None),
            period_label=(f"{q}Q{y}" if fp != "FY" else f"FY{y}"),
            geo_scope=geo, metric="deposits", value=float(val),
            unit=f"{cur}_absolute", currency=cur, confidence="high",
            raw_context=f"{tax}:{tag} fp={fp} end={end}",
            source_url=f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik10}/{tax}/{tag}.json",
            accession=e.get("accn", ""),
        ))
    log.info(f"[{firm['firm_key']}] deposits: {len(out)} obs from {tax}:{tag} ({cur})")
    return out


def _pick_documents(cik: str, accession: str) -> list[tuple[str, str]]:
    """Return [(filename, url)] of text-bearing exhibits (htm/txt, skip images)."""
    accn = accession.replace("-", "")
    code, txt = http_get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn}/index.json")
    if not txt:
        return []
    try:
        items = json.loads(txt)["directory"]["item"]
    except Exception:
        return []
    docs = []
    for it in items:
        name = it["name"]
        if re.search(r"\.(htm|html|txt)$", name, re.I) and not name.endswith("-index.html") \
           and "index-headers" not in name:
            size = int(it.get("size") or 0)
            docs.append((size, name,
                         f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn}/{name}"))
    # largest first (financials / earnings release tend to be the big docs)
    docs.sort(reverse=True)
    return [(n, u) for _, n, u in docs[:3]]


def customers_from_text(firm: dict, since: int, max_filings: int) -> list[dict]:
    """Best-effort customer-count extraction from filing exhibits, archiving text."""
    sub = fetch_submissions(firm["sec_cik"])
    if not sub:
        return []
    rows = [r for r in sub["_rows"]
            if r.get("form") in EARNINGS_FORMS and (r.get("reportDate") or r.get("filingDate", ""))[:4].isdigit()
            and int((r.get("reportDate") or r.get("filingDate"))[:4]) >= since]
    rows.sort(key=lambda r: r.get("reportDate") or r.get("filingDate"), reverse=True)
    rows = rows[:max_filings]
    firm_raw = os.path.join(RAW_DIR, firm["firm_key"])
    os.makedirs(firm_raw, exist_ok=True)

    out = []
    for r in rows:
        accn = r.get("accessionNumber")
        rep = r.get("reportDate") or r.get("filingDate")
        y = int(rep[:4]); q = (int(rep[5:7]) - 1) // 3 + 1
        text_parts = []
        for name, url in _pick_documents(firm["sec_cik"], accn):
            code, body = http_get(url)
            if body:
                text_parts.append(strip_html(body))
        if not text_parts:
            continue
        plain = "\n".join(text_parts)
        # archive
        try:
            with open(os.path.join(firm_raw, f"{accn}_{rep}.txt"), "w", encoding="utf-8") as fh:
                fh.write(plain)
        except Exception:
            pass
        # extract candidate customer counts
        best = {}  # (metric, scope) -> (value, context)
        for m in _CUST_RX.finditer(plain):
            num = parse_number(m.group(1))
            if num is None:
                continue
            unit = (m.group(2) or "").lower()
            mult = 1e6 if unit in ("million", "millions", "mn", "mm", "mi") else \
                   (1e9 if unit in ("bn", "billion") else (1e3 if unit == "thousand" else 1.0))
            val = num * mult
            # plausibility: customer counts between 100k and 200m
            if not (1e5 <= val <= 2e8):
                continue
            active = bool(m.group(3) and "active" in m.group(3).lower())
            metric = "customers_active" if active else "customers_total"
            ctx = plain[max(0, m.start() - 60):m.end() + 40]
            geo = "consolidated" if firm["country_scope"] == "latam" else "brazil"
            gm = _GEO_RX.search(ctx)
            if gm:
                g = gm.group(1).lower()
                if g in ("brazil", "brasil"):
                    geo = "brazil"
                elif g in ("mexico", "méxico", "colombia", "colômbia"):
                    geo = "other"
            key = (metric, geo)
            # keep the largest plausible figure for a (metric,scope) in this filing
            if key not in best or val > best[key][0]:
                best[key] = (val, ctx.replace("\n", " ").strip()[:160])
        for (metric, geo), (val, ctx) in best.items():
            out.append(new_record(
                firm_key=firm["firm_key"], firm_name=firm["display_name"],
                cnpj_root=firm["cnpj_root"], segment=firm["segment"], source="sec_edgar",
                filing_form=r.get("form", ""), period_year=y, period_quarter=q,
                period_label=f"{q}Q{y}", geo_scope=geo, metric=metric, value=val,
                unit="count", currency="", confidence="low", raw_context=ctx,
                source_url=f"https://www.sec.gov/Archives/edgar/data/{int(firm['sec_cik'])}/{accn.replace('-','')}/",
                accession=accn,
            ))
    log.info(f"[{firm['firm_key']}] customers: {len(out)} candidate obs from {len(rows)} filings")
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2018)
    ap.add_argument("--max-filings", type=int, default=48, help="filings/firm for text extraction")
    ap.add_argument("--firms", type=str, default="", help="comma-separated firm_keys")
    ap.add_argument("--deposits-only", action="store_true", help="XBRL deposits only, skip text")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    reg = [f for f in load_registry(BASE) if f.get("sec_cik")]
    if args.firms:
        want = {x.strip() for x in args.firms.split(",")}
        reg = [f for f in reg if f["firm_key"] in want]

    if args.list:
        for f in reg:
            print(f"{f['firm_key']:14s} CIK={f['sec_cik']} scope={f['country_scope']} cnpj={f['cnpj_root']}")
        return

    os.makedirs(RAW_DIR, exist_ok=True)
    records = []
    for f in reg:
        log.info(f"=== {f['firm_key']} (CIK {f['sec_cik']}, scope={f['country_scope']}) ===")
        records += deposits_from_xbrl(f)
        if not args.deposits_only:
            records += customers_from_text(f, args.since, args.max_filings)

    n = write_panel(records, OUT_CSV)
    log.info(f"Wrote {n} records -> {OUT_CSV}")
    # quick summary
    import pandas as pd
    if records:
        df = pd.DataFrame(records)
        print("\n--- records by firm x metric ---")
        print(df.pivot_table(index="firm_key", columns="metric", values="value",
                             aggfunc="count", fill_value=0).to_string())


if __name__ == "__main__":
    main()
