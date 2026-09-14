"""
scrape_sec_advertising.py
=========================
Advertising and marketing expense facts from the SEC 20-F XBRL instance documents of the
Brazilian banks and payment companies that file with the SEC. An ANNUAL panel, kept separate
from the quarterly advertising panel.

Filers
------
  1132597  Itau Unibanco Holding S.A.
  1160330  Banco Bradesco S.A.
  1471055  Banco Santander (Brasil) S.A.
  1691493  Nu Holdings Ltd.           reports in USD for the whole group (Brazil, Mexico, Colombia)
  1712807  PagSeguro Digital Ltd.
  1745431  StoneCo Ltd.
  1787425  XP Inc.
  1864163  Inter & Co, Inc.

Why the instance documents
--------------------------
The companyfacts API omits company-specific extension elements (for example Bradesco's
bbd:OtherAdministrativeExpensesAdvertisingPromotionsAndPublicRelations and Nu's
nu:BrandingAndAdvertising), so each 20-F's own XBRL instance is read. Inline filings ship an
extracted instance named *_htm.xml; older filings ship a plain instance .xml.

What is kept
------------
Every fact, in any taxonomy, whose element local name contains Advertis, Marketing, Branding,
Publicity, Propaganda or Promotion, with its context period, unit and dimensions. Which element
is the right measure differs by filer (Santander and StoneCo tag SalesAndMarketingExpense, which
is broader than advertising) and is decided downstream, not here.

Units and vintages
------------------
  - Values are as filed, in the unit's currency (BRL or USD), in units (XBRL values are not
    scaled by a multiplier). value_abs is the absolute amount; sign_raw keeps the filed sign,
    which flips between filings for some filers.
  - Every 20-F carries prior-year comparatives, so one fiscal year appears in several filings.
    For each filer, element, dimensions and period the annual panel keeps value_first (the
    earliest filing) and value_latest (the most recent filing), both unaltered.
  - Amendments (20-F/A) count as filings with their own filing date.
  - Currency conversion is not done here.

Outputs (paths.AWARENESS_PROC)
------------------------------
  sec_advertising_facts.{parquet,csv}    one row per fact per filing
  sec_advertising_annual.{parquet,csv}   filer x element x dimensions x fiscal period
    is_annual       the context spans 350-380 days
    scale_suspect   first and latest values differ by a factor near 1000
    restated        first and latest differ by more than 0.5% otherwise
    sign_flip       first and latest carry different signs

Validation (any failure aborts before writing)
----------------------------------------------
  1. Every filer has at least one 20-F with an XBRL instance.
  2. Every filer has at least one matching element in its latest 20-F.
  3. Fiscal-2025 anchors read directly from the instance documents on 2026-09-14 are
     reproduced within 0.5% (absolute values, value_latest, matching element, year and
     currency under any dimensions: Itau tags its figure on
     ifrs-full:AttributionOfExpensesByNatureToTheirFunctionAxis).
  Some matching elements are not advertising expense at all (Santander's
  bsbr:MarketingOfNonbankingFinancialProducts* are revenue lines; Bradesco's
  bbd:OtherOperatingIncomeexpensesCardMarketingExpenses is card-programme marketing); they are
  kept so the element choice is made downstream with everything visible.

Usage
-----
  python scrape_sec_advertising.py
  python scrape_sec_advertising.py --refresh      # refetch submissions, indexes and instances
  python scrape_sec_advertising.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree

from utils import paths
from utils.disclosure_common import http_get

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RAW_DIR = paths.AWARENESS_RAW / "sec"
OUT_DIR = paths.AWARENESS_PROC

FILERS = {
    "1132597": "Itau Unibanco Holding S.A.",
    "1160330": "Banco Bradesco S.A.",
    "1471055": "Banco Santander (Brasil) S.A.",
    "1691493": "Nu Holdings Ltd.",
    "1712807": "PagSeguro Digital Ltd.",
    "1745431": "StoneCo Ltd.",
    "1787425": "XP Inc.",
    "1864163": "Inter & Co, Inc.",
}
FORMS = {"20-F", "20-F/A"}
ELEMENT_RE = re.compile(r"advertis|marketing|branding|publicity|propaganda|promotion", re.I)
XBRLI = "http://www.xbrl.org/2003/instance"
XBRLDI = "http://xbrl.org/2006/xbrldi"
SKIP_NS = {XBRLI, "http://www.xbrl.org/2003/linkbase", "http://www.w3.org/1999/xlink"}

# (cik, element, abs value, currency) for fiscal 2025, no dimensions.
ANCHORS_FY2025 = [
    ("1471055", "ifrs-full:SalesAndMarketingExpense", 482.9e6, "BRL"),
    ("1132597", "ifrs-full:SalesAndMarketingExpense", 1740e6, "BRL"),
    ("1160330", "bbd:OtherAdministrativeExpensesAdvertisingPromotionsAndPublicRelations", 1287.2e6, "BRL"),
    ("1691493", "nu:BrandingAndAdvertising", 271.1e6, "USD"),
    ("1691493", "ifrs-full:SalesAndMarketingExpense", 302.8e6, "USD"),
    ("1864163", "ifrs-full:AdvertisingExpense", 285.0e6, "BRL"),
    ("1787425", "ifrs-full:AdvertisingExpense", 294.5e6, "BRL"),
    ("1712807", "ifrs-full:AdvertisingExpense", 868.8e6, "BRL"),
    ("1745431", "ifrs-full:SalesAndMarketingExpense", 1030.9e6, "BRL"),
]
ANCHOR_TOL = 0.005


# ---------------------------------------------------------------------------
# Fetching with a local cache
# ---------------------------------------------------------------------------
def cached(url: str, path: Path, refresh: bool, binary: bool = False) -> bytes | str | None:
    if path.exists() and not refresh:
        return path.read_bytes() if binary else path.read_text(encoding="utf-8")
    code, body = http_get(url, binary=binary, timeout=180)
    if body is None:
        log.warning("fetch failed (HTTP %s): %s", code, url)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        path.write_bytes(body)
    else:
        path.write_text(body, encoding="utf-8")
    return body


def annual_filings(cik: str, refresh: bool) -> list[dict]:
    """All 20-F and 20-F/A filings of a filer, including the paged older submissions files."""
    cik10 = cik.zfill(10)
    base = RAW_DIR / cik
    main = cached(f"https://data.sec.gov/submissions/CIK{cik10}.json",
                  base / "submissions.json", refresh)
    if main is None:
        raise SystemExit(f"{cik}: cannot fetch the submissions index")
    doc = json.loads(main)
    blocks = [doc["filings"]["recent"]]
    for extra in doc["filings"].get("files", []):
        page = cached(f"https://data.sec.gov/submissions/{extra['name']}",
                      base / extra["name"], refresh)
        if page is not None:
            blocks.append(json.loads(page))
    rows = []
    for b in blocks:
        for i, form in enumerate(b["form"]):
            if form in FORMS:
                rows.append({"cik": cik, "form": form,
                             "accession": b["accessionNumber"][i],
                             "filing_date": b["filingDate"][i],
                             "report_date": b["reportDate"][i]})
    return sorted(rows, key=lambda r: r["filing_date"])


def instance_document(cik: str, accession: str, refresh: bool) -> tuple[str, bytes] | None:
    acc = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/"
    idx = cached(base + "index.json", RAW_DIR / cik / acc / "index.json", refresh)
    if idx is None:
        return None
    names = [it["name"] for it in json.loads(idx)["directory"]["item"]]
    candidates = [n for n in names if n.endswith("_htm.xml")]
    if not candidates:
        candidates = [n for n in names if n.lower().endswith(".xml")
                      and not re.search(r"(_cal|_def|_lab|_pre|filingsummary)", n, re.I)]
    if not candidates:
        return None
    name = sorted(candidates, key=len)[0]
    body = cached(base + name, RAW_DIR / cik / acc / name, refresh, binary=True)
    return (name, body) if body is not None else None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_instance(xml: bytes) -> pd.DataFrame:
    root = etree.fromstring(xml, parser=etree.XMLParser(huge_tree=True, recover=True))
    prefixes = {uri: prefix for prefix, uri in root.nsmap.items() if prefix}

    contexts = {}
    for ctx in root.iter(f"{{{XBRLI}}}context"):
        period = ctx.find(f"{{{XBRLI}}}period")
        start = period.findtext(f"{{{XBRLI}}}startDate")
        end = period.findtext(f"{{{XBRLI}}}endDate") or period.findtext(f"{{{XBRLI}}}instant")
        dims = sorted(f"{m.get('dimension')}={(m.text or '').strip()}"
                      for m in ctx.iter(f"{{{XBRLDI}}}explicitMember", f"{{{XBRLDI}}}typedMember"))
        contexts[ctx.get("id")] = (start, end, ";".join(dims))
    units = {}
    for u in root.iter(f"{{{XBRLI}}}unit"):
        measure = u.findtext(f"{{{XBRLI}}}measure") or ""
        units[u.get("id")] = measure.split(":")[-1]

    rows = []
    for el in root:
        if not isinstance(el.tag, str) or el.get("contextRef") is None:
            continue
        qname = etree.QName(el)
        if qname.namespace in SKIP_NS or not ELEMENT_RE.search(qname.localname):
            continue
        text = (el.text or "").strip()
        try:
            value = float(text)
        except ValueError:
            continue
        start, end, dims = contexts.get(el.get("contextRef"), (None, None, ""))
        rows.append({"element": f"{prefixes.get(qname.namespace, qname.namespace)}:{qname.localname}",
                     "period_start": start, "period_end": end, "dimensions": dims,
                     "unit": units.get(el.get("unitRef"), ""), "decimals": el.get("decimals"),
                     "value": value})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_facts(refresh: bool) -> pd.DataFrame:
    frames = []
    for cik, name in FILERS.items():
        filings = annual_filings(cik, refresh)
        with_instance = 0
        for f in filings:
            inst = instance_document(cik, f["accession"], refresh)
            if inst is None:
                log.info("  %s %s %s: no XBRL instance", name, f["form"], f["filing_date"])
                continue
            facts = parse_instance(inst[1])
            with_instance += 1
            if facts.empty:
                log.info("  %s %s %s: instance %s has no matching element", name, f["form"],
                         f["filing_date"], inst[0])
                continue
            for k, v in f.items():
                facts[k] = v
            facts["entity_name"] = name
            facts["instance_file"] = inst[0]
            frames.append(facts)
            log.info("  %s %s %s: %d matching facts (%s)", name, f["form"], f["filing_date"],
                     len(facts), ", ".join(sorted(facts["element"].unique())))
        if with_instance == 0:                                                   # check 1
            raise AssertionError(f"{name} ({cik}): no 20-F with an XBRL instance")
    facts = pd.concat(frames, ignore_index=True)
    facts = facts[facts["period_start"].notna()].copy()          # durations only
    facts["value_abs"] = facts["value"].abs()
    facts["sign_raw"] = np.sign(facts["value"]).astype(int)
    days = (pd.to_datetime(facts["period_end"]) - pd.to_datetime(facts["period_start"])).dt.days
    facts["period_days"] = days
    facts["is_annual"] = days.between(350, 380)
    facts["fiscal_year"] = pd.to_datetime(facts["period_end"]).dt.year
    return facts


def annual_panel(facts: pd.DataFrame) -> pd.DataFrame:
    key = ["cik", "entity_name", "element", "dimensions", "unit", "period_start", "period_end"]
    f = facts.sort_values(key + ["filing_date", "accession"])
    first = f.groupby(key, dropna=False).first()
    last = f.groupby(key, dropna=False).last()
    out = pd.DataFrame({
        "fiscal_year": first["fiscal_year"], "is_annual": first["is_annual"],
        "value_first": first["value"], "value_latest": last["value"],
        "accession_first": first["accession"], "filing_date_first": first["filing_date"],
        "accession_latest": last["accession"], "filing_date_latest": last["filing_date"],
        "n_filings": f.groupby(key, dropna=False)["accession"].nunique(),
    }).reset_index()
    ratio = out["value_first"].abs() / out["value_latest"].abs().replace(0, np.nan)
    out["scale_suspect"] = ratio.between(0.00095, 0.00105) | ratio.between(950, 1050)
    rel = (out["value_first"].abs() - out["value_latest"].abs()).abs() \
        / out["value_latest"].abs().replace(0, np.nan)
    out["restated"] = (rel > 0.005) & ~out["scale_suspect"]
    out["sign_flip"] = np.sign(out["value_first"]) * np.sign(out["value_latest"]) < 0
    out["value_first_abs"] = out["value_first"].abs()
    out["value_latest_abs"] = out["value_latest"].abs()
    return out.sort_values(["cik", "element", "dimensions", "period_end"]).reset_index(drop=True)


def validate(facts: pd.DataFrame, annual: pd.DataFrame) -> None:
    for cik, name in FILERS.items():                                             # check 2
        mine = facts[facts["cik"] == cik]
        if mine.empty:
            raise AssertionError(f"{name}: no matching advertising or marketing element in any 20-F")
        latest = mine["filing_date"].max()
        if mine[mine["filing_date"] == latest].empty:
            raise AssertionError(f"{name}: latest 20-F carries no matching element")

    base = annual[annual["is_annual"]]                                           # check 3
    for cik, element, expected, currency in ANCHORS_FY2025:
        rows = base[(base["cik"] == cik) & (base["element"] == element)
                    & (base["fiscal_year"] == 2025) & (base["unit"] == currency)]
        hit = rows[(rows["value_latest_abs"] / expected - 1).abs() <= ANCHOR_TOL]
        if hit.empty:
            raise AssertionError(f"anchor {FILERS[cik]} {element} FY2025 {currency}: expected "
                                 f"{expected:,.0f}, found {rows['value_latest_abs'].tolist()}")
        log.info("  anchor %s %s FY2025 = %s %s (dimensions: %s)", FILERS[cik], element,
                 f"{hit['value_latest_abs'].iloc[0]:,.0f}", currency,
                 hit["dimensions"].iloc[0] or "none")
    log.info("fiscal-2025 anchors reproduced: %d", len(ANCHORS_FY2025))


def main() -> None:
    ap = argparse.ArgumentParser(description="SEC 20-F advertising and marketing facts")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()

    facts = build_facts(a.refresh)
    annual = annual_panel(facts)
    validate(facts, annual)

    base = annual[annual["is_annual"]]
    log.info("flags on annual facts: scale_suspect %d, restated %d, sign_flip %d",
             int(base["scale_suspect"].sum()), int(base["restated"].sum()),
             int(base["sign_flip"].sum()))
    flagged = base[base["scale_suspect"] | base["restated"] | base["sign_flip"]]
    if len(flagged):
        log.info("flagged annual facts:\n%s", flagged[["entity_name", "element", "fiscal_year",
                 "unit", "value_first", "value_latest", "scale_suspect", "restated",
                 "sign_flip"]].to_string(index=False))
    cover = (base.assign(dims=base["dimensions"].ne(""))
                 .groupby(["entity_name", "element", "dims", "unit"])["fiscal_year"]
                 .agg(["min", "max", "nunique"]))
    log.info("coverage of annual facts (dims = carries dimensions):\n%s", cover.to_string())

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("sec_advertising_facts", facts), ("sec_advertising_annual", annual)):
        frame.to_parquet(a.out_dir / f"{name}.parquet", index=False)
        frame.to_csv(a.out_dir / f"{name}.csv", index=False)
        log.info("%s: %d rows -> %s", name, len(frame), a.out_dir / f"{name}.parquet")


if __name__ == "__main__":
    main()
