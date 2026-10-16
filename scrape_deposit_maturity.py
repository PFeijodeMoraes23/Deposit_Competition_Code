###─────────────────────────────────────────────────────────────────────────────
# scrape_deposit_maturity.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Time deposits by remaining maturity bucket, from the banks' own filings, 2016-2024, in
#   three subcommands that share nothing but this file:
#     sec      SEC Form 20-F filings (Itaú, Bradesco, Santander Brasil, Nu, Inter, XP)
#     cvm      CVM DFP statements (ten listed Brazilian banks), text layer with OCR fallback
#     summary  one bank-year summary of the two extractions, the input of Section 4 of
#              deposit_types_report_2026-10-15.md and of switch_cost_tools.py locked
#   The two extractors keep their own conventions; for instance, the SEC part converts months
#   to days with a 365-day year and the CVM part with a 360-day year.
#
# CLI
#   python scrape_deposit_maturity.py sec [--firms itau,bradesco] [--refresh]
#   python scrape_deposit_maturity.py cvm [--firms bb,banrisul] [--years 2016-2024] [--refresh]
#   python scrape_deposit_maturity.py summary [FOLDER] [--out FILE]
#   python scrape_deposit_maturity.py <subcommand> -h
###─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import csv
import http.cookiejar
import io
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

try:
    from utils.venv_guard import ensure_project_venv
except Exception:  # pragma: no cover
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from lxml import html as lhtml

from utils import paths
from utils.disclosure_common import get_json, http_get
from utils.firm_registry import FIRMS
from utils.paths import RAW


# ================================================================================ sec
###─────────────────────────────────────────────────────────────────────────────
# sec: SEC Form 20-F filings
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Time deposits (and, where the same table prints them, demand, savings and total
#   deposits) by remaining maturity bucket, read from the SEC Form 20-F filings of
#   the Brazilian banks and the SEC-listed digital platforms, as of each fiscal year
#   end 2016-2024. The maturity profile is the observable counterpart of how much of
#   the deposit base can leave within a quarter (deposit_types_report_2026-10-15.md).
#
# DATA SOURCE
#   SEC EDGAR. Filing lists: https://data.sec.gov/submissions/CIK##########.json
#   (plus the older-filings pages it points to). Main document of each 20-F and
#   20-F/A: https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<document>.
#   Requests go through utils.disclosure_common.http_get (SEC fair-access user agent
#   and throttle). Firms and CIKs come from utils.firm_registry: Itaú Unibanco,
#   Bradesco, Santander Brasil, Nu Holdings, Inter & Co, XP Inc.
#
# TABLE KINDS (column table_kind)
#   all_time_deposits         time deposits by remaining maturity ("Deposits by
#                             maturity", contractual-obligation and maturity tables)
#   guide3_large              SEC Industry Guide 3 table: time deposits of US$100,000
#                             or more by remaining maturity (large deposits only)
#   s1405_uninsured           Regulation S-K Item 1405 table: uninsured deposits by
#                             remaining maturity (uninsured deposits only)
#   ifrs_liabilities_maturity IFRS liquidity-risk note: financial liabilities (deposit
#                             lines) by remaining contractual maturity, often
#                             undiscounted flows and with different bucket edges
#   other                     any other deposit-by-maturity table
#   Buckets are converted to days with 1 month = 30 days and 1 year = 365 days;
#   lo_days/hi_days follow the printed edges, hi_days is empty for open-ended buckets
#   and "no maturity"/"on demand" buckets are lo 0, hi 0. A table is kept once per
#   filing for each distinct bucket set; the instrument is normalized (time_deposits,
#   total_deposits, demand, savings, interbank, uninsured_deposits, other_deposits)
#   and the printed row label is kept in notes.
#
# COVERAGE
#   Fiscal years 2016-2024 for each filer, limited to the filings that exist: Nu
#   Holdings from FY2021, Inter & Co from FY2021, XP Inc from FY2019, Itaú, Bradesco
#   and Santander Brasil from FY2016. The FY2025 20-F is read as well because its
#   prior-year columns repeat FY2024. Where several filings report the same year the
#   latest filing wins in the main table (among the filings whose buckets add up to
#   the printed total within 0.5%, when any do) and the agreement is recorded in the
#   QA table.
#
# OUTPUTS (raw data area, nothing in the estimation pipeline reads them)
#   RAW/DEPOSIT_MATURITY/sec/<cik>_<accession>.html   main document of each filing
#   RAW/DEPOSIT_MATURITY/deposit_maturity_sec.csv     one row per firm x fiscal year x
#                                                     table x instrument x bucket
#   RAW/DEPOSIT_MATURITY/deposit_maturity_sec_qa.csv  firm, fiscal_year, n_sources,
#                                                     max relative difference across
#                                                     filings reporting the same year
#
# CLI
#   python scrape_deposit_maturity.py sec                       # all firms
#   python scrape_deposit_maturity.py sec --firms itau,bradesco # subset
#   python scrape_deposit_maturity.py sec --refresh             # re-download filings
###─────────────────────────────────────────────────────────────────────────────
sec_log = logging.getLogger("deposit_maturity_sec")
OUT = paths.RAW / "DEPOSIT_MATURITY"
HTML_DIR = OUT / "sec"
FIRM_KEYS = ["itau", "bradesco", "santander_br", "nubank", "inter", "xp"]
FY_MIN, FY_MAX = 2016, 2024
FILING_YEARS = (2016, 2025)   # report-date years whose 20-F is downloaded
SEC_SUM_TOL = 0.005             # buckets must add up to the printed total within 0.5%
MIN_DOC_CHARS = 200_000      # shorter documents are cover-page amendments

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "eighteen": 18}
UNIT_DAYS = {"day": 1, "month": 30, "year": 365}
DASHES = set("-–—−�")
YEAR_CELL = re.compile(r"(19|20)\d\d")


# ───────────────────────── filings ─────────────────────────
def registry_ciks() -> dict[str, str]:
    return {f["firm_key"]: str(int(f["sec_cik"])) for f in FIRMS
            if f.get("firm_key") in FIRM_KEYS and f.get("sec_cik")}


def list_filings(cik: str) -> list[dict]:
    """20-F and 20-F/A filings with report dates 2016-2025, from the recent list and
    the older-filings pages of the submissions API."""
    code, j = get_json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
    if j is None:
        sec_log.warning("submissions unavailable for %s (HTTP %s)", cik, code)
        return []
    blocks = [j["filings"]["recent"]]
    for f in j["filings"].get("files", []):
        _, jj = get_json("https://data.sec.gov/submissions/" + f["name"])
        if jj:
            blocks.append(jj)
    rows = []
    for b in blocks:
        for i, form in enumerate(b["form"]):
            rep = b["reportDate"][i]
            if form.startswith("20-F") and rep and FILING_YEARS[0] <= int(rep[:4]) <= FILING_YEARS[1]:
                acc, doc = b["accessionNumber"][i], b["primaryDocument"][i]
                rows.append(dict(form=form, filed=b["filingDate"][i], report_date=rep, accession=acc, doc=doc,
                                 url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                                     f"{acc.replace('-', '')}/{doc}"))
    return rows


def fetch_filing(cik: str, fl: dict, refresh: bool) -> str | None:
    fn = HTML_DIR / f"{cik}_{fl['accession']}.html"
    if fn.exists() and not refresh:
        return fn.read_text(encoding="utf-8", errors="replace")
    code, body = http_get(fl["url"], timeout=180)
    if not body:
        sec_log.warning("download failed %s (HTTP %s)", fl["url"], code)
        return None
    fn.write_text(body, encoding="utf-8")
    return body


# ───────────────────────── numbers, dates, buckets ─────────────────────────
def sec_parse_num(s: str):
    """'1,234' -> 1234; '(1,234)' -> -1234; dashes -> 0; anything else -> None."""
    t = s.replace("\xa0", " ").strip()
    if t and all(c in DASHES or c in " ()" for c in t):
        return 0.0
    t = re.sub(r"\s*\(\d\)$", "", t)                       # footnote marker after a number
    m = re.fullmatch(r"\(?\s*(-?)\s*(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*\)?", t)
    if not m:
        return None
    v = float((m.group(2) + (m.group(3) or "")).replace(",", ""))
    return -v if (m.group(1) or t.startswith("(")) else v


def find_dates(s: str) -> list[str]:
    out = []
    for m in re.finditer(r"(\d{1,2})/(\d{1,2})/(\d{4})", s):
        out.append(f"{int(m.group(3))}-{int(m.group(1)):02d}-{int(m.group(2)):02d}")
    for m in re.finditer(r"(january|february|march|april|may|june|july|august|september|october|"
                         r"november|december)\s+(\d{1,2}),?\s+(\d{4})", s.lower()):
        out.append(f"{int(m.group(3))}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}")
    return out


def _n(tok: str) -> int | None:
    tok = tok.lower().strip()
    return WORDNUM.get(tok, int(tok) if tok.isdigit() else None)


NUM_RE = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|eighteen)"
UNIT_RE = r"(days?|months?|years?)"


def _days(n: int, unit: str) -> int:
    u = unit.lower().rstrip("s")
    return 365 if (u == "month" and n == 12) else n * UNIT_DAYS[u]


def sec_parse_bucket(label: str):
    """Printed bucket label -> (lo_days, hi_days|None), or None when it is not a bucket."""
    s = re.sub("[–—−�]", "-", label.lower().replace("\xa0", " "))
    s = re.sub(r"\(\d\)|\*", "", s)
    s = re.sub(r"\s+", " ", s).strip(" :.")
    if s.startswith("total") or "average" in s:
        return None
    if re.search(r"indefinite|no stated maturity|no maturity|on demand|not sensitive|without maturity|"
                 r"^demand|no contractual", s):
        return (0, 0)
    m = re.search(rf"(?:from |after |over |between )?{NUM_RE}\s*{UNIT_RE}?\s*(?:-|to|through|and)\s*{NUM_RE}\s*{UNIT_RE}?", s)
    if m:
        a, ua, b, ub = _n(m.group(1)), m.group(2), _n(m.group(3)), m.group(4)
        ub = ub or ua or ("days" if max(a, b) > 12 else None)
        if ub:
            return (_days(a, ua or ub), _days(b, ub))
    m = re.search(rf"(?:up to|within|under|less than|not more than|maturity in|due in)\s*{NUM_RE}\s*{UNIT_RE}", s) \
        or re.search(rf"{NUM_RE}\s*{UNIT_RE}\s*or less", s)
    if m:
        return (0, _days(_n(m.group(1)), m.group(2)))
    m = re.search(rf"(?:over|more than|after|greater than|beyond|>)\s*{NUM_RE}\s*{UNIT_RE}?", s) \
        or re.search(rf"{NUM_RE}\s*{UNIT_RE}?\s*or more", s)
    if m:
        a = _n(m.group(1))
        u = m.group(2) or ("days" if a > 12 else None)
        if u:
            return (_days(a, u), None)
    return None


def sec_instrument_of(label: str, deposits_table: bool = False):
    s = re.sub(r"\(\d\)|\*", "", label.lower()).strip(" :")
    if re.search(r"compulsory|central bank|from banks|credit institutions|guarantee|escrow|judicial", s):
        return None
    if "time deposit" in s or "term deposit" in s or re.search(r"certificates? of deposit", s):
        return "time_deposits"
    if "uninsured" in s:
        return "uninsured_deposits"
    if re.search(r"non-?\s?interest[- ]bearing", s):
        return "non_interest_bearing_deposits"
    if re.search(r"interest[- ]bearing", s):
        return "interest_bearing_deposits"
    if "savings" in s:
        return "savings"
    if "demand deposit" in s:
        return "demand"
    if "interbank" in s:
        return "interbank"
    if re.fullmatch(r"(total )?(customer )?deposits?( from customers)?|deposits? from customers|customer deposits?", s):
        return "total_deposits"
    if "deposit" in s:
        return "other_deposits"
    if deposits_table and s == "total":
        return "total_deposits"
    return None


# ───────────────────────── table reading ─────────────────────────
def cell_texts(tr) -> list[str]:
    out = []
    for c in tr.xpath("./td|./th"):
        t = re.sub(r"\s+", " ", c.text_content().replace("\xa0", " ")).strip()
        if t and t not in ("$", "R$", "US$", ")", "%", "("):
            out.append(t)
    return out


def detect_unit(text: str) -> str:
    s = text.lower()
    cur = "US$" if re.search(r"us\$|u\.s\. dollars", s) and not re.search(r"r\$|reais", s) else "R$"
    if re.search(r"in millions|millions of|r\$ million|million of reais|millions\)|\bmillions\b", s):
        return f"{cur} million"
    if re.search(r"in thousands|thousands of|r\$ thousand|thousand of reais|thousands\)|\bthousand\b", s):
        return f"{cur} thousand"
    if re.search(r"in billions|billions of", s):
        return f"{cur} billion"
    return ""


def classify(ctx: str, as_of_year: int | None) -> str:
    s = ctx.lower()
    if "100,000" in s:
        return "guide3_large"
    if re.search(r"uninsured|exceed\w* the (maximum )?insured|exceeding the maximum insured", s):
        return "s1405_uninsured" if (as_of_year or 2021) >= 2021 else "guide3_large"
    if re.search(r"undiscounted|contractual maturit|liquidity risk|residual maturity|"
                 r"financial assets and liabilities|schedule of (financial|assets)", s):
        return "ifrs_liabilities_maturity"
    if re.search(r"deposits by maturity|contractual obligations|maturities of|time deposits|maturity of", s):
        return "all_time_deposits"
    return "other"


def extract_tables(doc: str, fy_filing: int, meta: dict) -> list[dict]:
    root = lhtml.fromstring(doc.encode("utf-8", "replace"))
    recs = []
    for ti, t in enumerate(root.iter("table")):
        txt = t.text_content()
        if not re.search(r"deposit", txt, re.I) or not re.search(r"days|months|year|maturit|demand", txt, re.I):
            continue
        rows = [r for r in (cell_texts(tr) for tr in t.xpath(".//tr")) if r]
        if not 3 <= len(rows) <= 60:
            continue
        if re.search(r"yield|average (interest )?rate|accounts subject to interest|interest rate sensitiv", txt, re.I):
            continue
        if re.search(r"not sensitive|\bgap\b|repric", txt, re.I) and not re.search(r"deposits? by maturity", txt, re.I):
            continue
        prev = " ".join(x.strip() for x in t.xpath(
            "preceding::*[self::p or self::div or self::span][normalize-space(text())!=''][position()<=10]/text()")[-10:])
        recs += parse_table(rows, prev, ti, fy_filing, meta)
    return recs


def parse_table(rows, prev, ti, fy_filing, meta) -> list[dict]:
    full = " ".join(" ".join(r) for r in rows)
    unit = detect_unit(" ".join(" ".join(r) for r in rows[:6])) or detect_unit(prev) or detect_unit(full)
    if not unit:
        unit = "unit not printed"
    title_row = next((" ".join(r) for r in rows if not any(sec_parse_num(x) is not None for x in r)
                      and not find_dates(" ".join(r)) and len(" ".join(r)) > 8
                      and not any(sec_parse_bucket(x) for x in r)), "")
    ctx_title = title_row or prev[-160:]
    ctx = f"{title_row} {prev[-300:]} {' '.join(rows[0])}"
    deposits_table = bool(re.search(r"deposits by maturity|maturity of deposits", ctx.lower()))
    recs: list[dict] = []
    pending: list[str] = []
    header = None
    brows = []
    section = ""
    for r in rows:
        dates = find_dates(" ".join(r)) + [f"{c}-12-31" for c in r if YEAR_CELL.fullmatch(c)]
        bcells = [sec_parse_bucket(x) for x in r]
        numeric = [sec_parse_num(x) is not None and not YEAR_CELL.fullmatch(x) for x in r[1:]]
        if sum(b is not None for b in bcells) >= 3 and not any(numeric):
            cols = []
            for x, b in zip(r, bcells):
                if b is not None:
                    cols.append(("b", x, b))
                elif cols:
                    cols.append(("t" if x.lower().startswith("total") else "o", x, None))
            header = cols
            if dates:
                pending = list(dict.fromkeys(dates))
            continue
        if dates and not any(numeric):
            pending = list(dict.fromkeys(dates))
            continue
        if len(r) == 1 and re.search(r"domestic|international|foreign", r[0], re.I):
            section = "_international" if re.search(r"international|foreign", r[0], re.I) else "_domestic"
            continue
        vals = [sec_parse_num(x) for x in r[1:]]
        ok = bool(vals) and all(v is not None for v in vals)
        instr = sec_instrument_of(r[0], deposits_table)
        if header and instr and ok and not any(v < 0 for v in vals):
            recs += emit_header_row(r[0], instr + section, vals, header, pending, ctx, ctx_title, unit, ti, fy_filing, meta)
        elif ok and sec_parse_bucket(r[0]) is not None:
            brows.append((r, vals))
        elif ok and brows and r[0].lower().startswith("total"):
            brows.append((r, vals))
    if not recs and sum(sec_parse_bucket(r[0]) is not None for r, _ in brows) >= 3:
        recs += emit_bucket_rows(brows, pending, ctx, ctx_title, unit, ti, fy_filing, meta, full)
    return recs


def base_rec(meta, as_of, ctx, ctx_title, unit, ti, note):
    return dict(firm_key=meta["firm_key"], cik=meta["cik"], fiscal_year=int(as_of[:4]), as_of=as_of,
                source="sec_20f", doc_url=meta["url"], table_title=ctx_title[:200], _ctx=ctx, unit=unit,
                extraction_method="html_table", notes=note, _table=ti, _filing=meta["accession"],
                _filed=meta["filed"])


def emit_header_row(label, instr, vals, header, pending, ctx, ctx_title, unit, ti, fy_filing, meta):
    """Buckets in the column header, instruments in the row labels. A header holding
    several 'bucket ... Total' groups (one per year) is split at each Total column."""
    segs, cur = [], []
    for c in header:
        cur.append(c)
        if c[0] == "t":
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    if len(segs) <= 1:
        segs = [header]
    out, pos = [], 0
    for si, seg in enumerate(segs):
        buckets = [c for c in seg if c[0] == "b"]
        h = len(buckets)
        chunk = vals if len(segs) == 1 else vals[pos:pos + len(seg)]
        pos += len(seg)
        total = None
        if len(chunk) == len(seg):
            bvals = [v for v, c in zip(chunk, seg) if c[0] == "b"]
            tv = [v for v, c in zip(chunk, seg) if c[0] == "t"]
            total = tv[0] if tv else None
        elif len(chunk) > h:
            lead = chunk[0]
            if lead and abs(sum(chunk[1:h + 1]) - lead) / abs(lead) < 0.005 and abs(sum(chunk[:h]) - lead) / abs(lead) >= 0.005:
                total, bvals = lead, chunk[1:h + 1]
            else:
                bvals, total = chunk[:h], chunk[h]
        elif len(chunk) == h:
            bvals = chunk
        else:
            continue
        note = ""
        if len(segs) > 1 and len(pending) == len(segs):
            as_of = pending[si]
        elif len(segs) == 1 and pending:
            as_of = pending[-1]
        elif len(segs) == 1 and find_dates(ctx):
            as_of = find_dates(ctx)[-1]
        elif si == 0:
            as_of, note = f"{fy_filing}-12-31", "as_of assumed = filing fiscal year end"
        else:
            continue
        chk = abs(sum(bvals) - total) / abs(total) if total else None
        for (_, lab, (lo, hi)), v in zip(buckets, bvals):
            rec = base_rec(meta, as_of, ctx, ctx_title, unit, ti,
                           "; ".join(x for x in [note, f"row label: {label}"] if x))
            rec.update(instrument=instr, bucket_label=lab, lo_days=lo, hi_days=hi, value=v,
                       total_reported=total, sum_check=chk)
            out.append(rec)
    return out


def emit_bucket_rows(brows, pending, ctx, ctx_title, unit, ti, fy_filing, meta, full):
    """Buckets as row labels (time deposits in uninsured accounts, Guide 3 tables). Where
    each year has Domestic/Foreign/Total columns the Total column is read."""
    s = (ctx_title + " " + ctx + " " + full[:400]).lower()
    instr = "time_deposits" if "time deposit" in s else "total_deposits"
    buckets = [(r, v) for r, v in brows if sec_parse_bucket(r[0]) is not None]
    totals = [(r, v) for r, v in brows if r[0].lower().startswith("total")]
    k, n = len(pending), len(buckets[0][1])
    if k and n == 3 * k:
        idx = list(range(2, n, 3))
    elif k and n == k:
        idx = list(range(n))
    elif not k and n == 1:
        idx = [0]
    elif not k and n == 3:
        idx = [2]
    else:
        return []
    years = pending if k else [f"{fy_filing}-12-31"]
    out = []
    for j, ci in enumerate(idx):
        vals = [v[ci] for _, v in buckets]
        total = totals[0][1][ci] if totals and len(totals[0][1]) > ci else None
        chk = abs(sum(vals) - total) / abs(total) if total else None
        note = "" if k else "as_of assumed = filing fiscal year end"
        for (r, _), val in zip(buckets, vals):
            lo, hi = sec_parse_bucket(r[0])
            rec = base_rec(meta, years[j], ctx, ctx_title, unit, ti, note)
            rec.update(instrument=instr, bucket_label=r[0], lo_days=lo, hi_days=hi, value=val,
                       total_reported=total, sum_check=chk)
            out.append(rec)
    return out


# ───────────────────────── assembly ─────────────────────────
COLS = ["firm_key", "cik", "fiscal_year", "as_of", "source", "doc_url", "table_title", "table_kind",
        "instrument", "bucket_label", "lo_days", "hi_days", "value", "unit", "extraction_method",
        "total_reported", "sum_check", "notes"]


UNIT_SCALE = {"thousand": 1e-3, "million": 1.0, "billion": 1e3}


def infer_units(df: pd.DataFrame) -> pd.DataFrame:
    """Tables whose unit is not printed take the unit of the same filing's printed-unit
    tables (same instrument first) that puts their printed total closest in size."""
    df = df.copy()
    for fil, g in df.groupby("_filing"):
        known = g[g["unit"] != "unit not printed"].drop_duplicates(["_table", "fiscal_year", "instrument"])
        units = sorted(known["unit"].unique())
        if not len(known):
            continue
        for key, h in g[g["unit"] == "unit not printed"].groupby(["_table", "fiscal_year", "instrument"]):
            tot = h["total_reported"].dropna()
            if not len(tot) or tot.iloc[0] <= 0:
                continue
            same = known[known["instrument"] == key[2]]
            ref = same if len(same) else known
            best, best_d = None, 1e9
            for u in units:
                sc = UNIT_SCALE[u.split()[-1]]
                for _, k in ref.iterrows():
                    if not k["total_reported"] or k["total_reported"] <= 0:
                        continue
                    d = abs(np.log(tot.iloc[0] * sc / (k["total_reported"] * UNIT_SCALE[k["unit"].split()[-1]])))
                    if d < best_d:
                        best, best_d = u, d
            if best:
                m = (df["_filing"] == fil) & (df["_table"] == key[0]) & (df["fiscal_year"] == key[1]) \
                    & (df["instrument"] == key[2])
                df.loc[m, "unit"] = best
                df.loc[m, "notes"] = df.loc[m, "notes"].map(lambda n: f"{n}; " if n else "") + "unit inferred from the filing's other tables"
    return df


def assemble(allrecs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = infer_units(pd.DataFrame(allrecs))
    df["table_kind"] = [classify(c, int(a[:4])) for c, a in zip(df["_ctx"], df["as_of"])]
    grp = ["_filing", "_table", "fiscal_year", "instrument"]
    df["hi_key"] = df["hi_days"].fillna(-1).astype(int)
    bset = df.groupby(grp).apply(lambda g: str(sorted(zip(g["lo_days"], g["hi_key"]))), include_groups=False)
    df = df.merge(bset.rename("bset").reset_index(), on=grp)
    ser = ["firm_key", "fiscal_year", "table_kind", "instrument", "bset"]
    # a filing can hold the same series in two tables: keep the first
    df = df[df["_table"] == df.groupby(["_filing"] + ser)["_table"].transform("min")].copy()
    df = df[(df["fiscal_year"] >= FY_MIN) & (df["fiscal_year"] <= FY_MAX)]
    qa_rows = []
    for k, g in df.groupby(ser + ["lo_days", "hi_key"]):
        if g["_filing"].nunique() > 1 and g["value"].abs().max() > 0:
            qa_rows.append(dict(firm=k[0], fiscal_year=k[1], n=g["_filing"].nunique(),
                                d=(g["value"].max() - g["value"].min()) / g["value"].abs().max(),
                                worst=f"{k[2]}|{k[3]}|{k[5]}-{k[6]}"))
    n_src = df.groupby(ser)["_filing"].transform("nunique")
    # latest filing wins, among the filings whose buckets add up to the printed total
    bad = df.groupby(ser + ["_filing"])["sum_check"].transform("max").fillna(0) > SEC_SUM_TOL
    best = df["_filed"].where(~bad).groupby([df[c] for c in ser]).transform("max")
    best = best.fillna(df.groupby(ser)["_filed"].transform("max"))
    df = df[df["_filed"] == best].copy()
    df["notes"] = [f"{n}{'; ' if n else ''}reported in {c} filings" if c > 1 else n
                   for n, c in zip(df["notes"], n_src[df.index])]
    df = df.sort_values(["firm_key", "fiscal_year", "table_kind", "instrument", "bset", "lo_days", "hi_key"])
    qa = pd.DataFrame(qa_rows, columns=["firm", "fiscal_year", "n", "d", "worst"])
    if len(qa):
        top = qa.sort_values("d").groupby(["firm", "fiscal_year"]).tail(1).set_index(["firm", "fiscal_year"])["worst"]
        qa = qa.groupby(["firm", "fiscal_year"]).agg(n_sources=("n", "max"), max_rel_diff=("d", "max")).reset_index()
        qa["worst_series"] = [top[(f, y)] for f, y in zip(qa["firm"], qa["fiscal_year"])]
    else:
        qa = pd.DataFrame(columns=["firm", "fiscal_year", "n_sources", "max_rel_diff", "worst_series"])
    return df[COLS], qa


def sec_main() -> None:
    ap = argparse.ArgumentParser(description="Deposit maturity buckets from SEC 20-F filings")
    ap.add_argument("--refresh", action="store_true", help="re-download every filing")
    ap.add_argument("--firms", default=",".join(FIRM_KEYS), help="comma-separated firm keys")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    ciks = registry_ciks()
    allrecs: list[dict] = []
    for fk in [x.strip() for x in args.firms.split(",") if x.strip()]:
        cik = ciks.get(fk)
        if not cik:
            sec_log.warning("firm %s not in registry", fk)
            continue
        for fl in sorted(list_filings(cik), key=lambda d: d["filed"]):
            doc = fetch_filing(cik, fl, args.refresh)
            if not doc or len(doc) < MIN_DOC_CHARS:
                sec_log.info("%s %s %s: skipped (%d chars)", fk, fl["form"], fl["report_date"], len(doc or ""))
                continue
            meta = dict(firm_key=fk, cik=cik, url=fl["url"], accession=fl["accession"], filed=fl["filed"])
            recs = extract_tables(doc, int(fl["report_date"][:4]), meta)
            sec_log.info("%s %s FY%s: %d rows", fk, fl["form"], fl["report_date"][:4], len(recs))
            allrecs += recs
    if not allrecs:
        sec_log.warning("no rows extracted")
        return
    df, qa = assemble(allrecs)
    df.to_csv(OUT / "deposit_maturity_sec.csv", index=False, encoding="utf-8", lineterminator="\n")
    qa.to_csv(OUT / "deposit_maturity_sec_qa.csv", index=False, encoding="utf-8", lineterminator="\n")
    sec_log.info("wrote %d rows and %d QA rows to %s", len(df), len(qa), OUT)


# ================================================================================ cvm
###─────────────────────────────────────────────────────────────────────────────
# cvm: CVM DFP statements
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Deposits by remaining maturity ("Depósitos a prazo" split into "Sem vencimento",
#   "Até 3 meses", "De 3 a 12 meses", "De 1 a 3 anos", ... buckets) as of 31 December
#   of each year, read from the notes of the banks' annual financial statements
#   filed with the CVM. The maturity split of the time-deposit stock is the input
#   to the contractual-maturity share of the deposit base (how much of the stock
#   can leave within a quarter), which the Pillar 3 LIQ2 template covers only for
#   2020-2023.
#
# DATA SOURCE
#   Discovery (route a, the one that works for every listed bank):
#     CVM open data DFP index
#       https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_YYYY.zip
#     dfp_cia_aberta_YYYY.csv has one row per filing (CNPJ_CIA, DT_REFER, VERSAO,
#     LINK_DOC). LINK_DOC points to the old ENETCONSULTA host, which resets the
#     connection; the same document id served by
#       https://www.rad.cvm.gov.br/ENET/frmDownloadDocumento.aspx?CodigoInstituicao=1
#       &NumeroSequencialDocumento=<ID_DOC>
#     (with a browser-like User-Agent) returns a ZIP holding the full filing: the
#     DFP XML, DadosDocumento.xlsx and the complete financial statements with
#     notes as one PDF. The highest VERSAO per CNPJ and reference date is used.
#   Fallback (route b, only for firm-years whose DFP PDF yields no time-deposit
#   maturity row): CVM IPE index
#       https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_YYYY.zip
#   rows with Categoria "Dados Econômico-Financeiros", Data_Referencia 31 December
#   and Tipo starting "Demonstrações Financeiras"; Link_Download is fetched as a PDF.
#   Route c (bank investor-relations sites) is not implemented; source "ir_site"
#   is reserved in the output schema.
#   Each DFP covers the current and the prior year, so a fiscal year is seen in up
#   to two filings (filing years Y and Y+1); among equally ranked tables the filing
#   of fiscal year Y itself is kept over the comparative column of the Y+1 filing,
#   and the discrepancy between the two is written to the QA file. DFP filings are
#   read for filing years first_year .. last_year+1 for that reason.
#
# EXTRACTION
#   Per PDF page, in this order:
#     1. text layer (PyMuPDF words with coordinates). A page is a candidate when it
#        mentions deposits together with maturity vocabulary. Words are grouped
#        into visual rows; a header block made of maturity phrases is located above
#        a data row whose label contains "depósitos" and its words are assigned to
#        the numeric columns by horizontal position. Column headers are parsed
#        into (lo_days, hi_days): "Sem vencimento" = 0/0, "Até 3 meses" = 0/90,
#        "De 3 a 12 meses" = 90/360, "Acima de 5 anos" = 1800/null (month = 30
#        days, year = 360 days). A "Total" column, when printed, is the check
#        total.
#     2. OCR, only when the document has no text layer on most pages: pages are
#        rendered at 300 dpi and read with Tesseract (lang por, --psm 6); the word
#        boxes feed the same row parser. extraction_method = "ocr".
#     3. Validation: sum_check is the relative gap |sum(buckets) - printed total| /
#        |printed total| as a float (empty when the table prints no total). The 0.5%
#        rule: a row passes when sum_check <= 0.005; larger gaps are kept and should be
#        treated as failed. Plausibility guard: a firm-year whose time-deposit total is
#        negative, or below 10% of the same bank's total deposits (the total_deposits row
#        of the same table, else the sum of the table's deposit rows, else a table of the
#        same document and statement), is flagged ("FLAG ..." in notes, status FLAGGED and
#        flagged=True in the log) and ranked below every unflagged candidate. So is a table
#        with fewer than three nonzero time-deposit buckets, or one below 1% of the firm's
#        median level in the neighbouring years (under both the stated unit and the raw
#        figures).
#   Layouts handled: instruments as rows and buckets as columns, bucket columns wrapped
#   over several header lines ("Sem" / "vencimento"), the total in the first column (BTG),
#   unitless day ranges ("0-30", "31-180", "Acima de 365"), rows named only "A Prazo" /
#   "À Vista" under a "Depósitos" heading (Banrisul, Itaú), and notes whose tables are
#   dated only by a bare year above the header (Daycoval, "2017" over "Banco" or
#   "Consolidado"). The statement of each table is read from its own header block or the
#   short row above it ("BB Consolidado", "BB Banco Múltiplo", "Banco", "Consolidado") before
#   the page text (order below).
#   Unit cross-check: units not stated on the table's page are inherited from earlier
#   pages and can be wrong; a firm-year whose stated unit lies more than 30x from the
#   firm's median level over its R$-thousand years while R$ thousand lies within 30x is
#   corrected, with "UNIT CORRECTED a -> b" in notes and in the log.
#   Tables printed on a page that speaks of "não descontados" (undiscounted cash flows of
#   the liquidity-risk note, e.g. Itaú's "31-365 / 366-720 dias" tables, which add future
#   interest) rank below balance tables of the deposits note.
#   Tables with maturity buckets as ROWS and instruments as columns (Banco BMG's
#   "Vencimento de depósitos interfinanceiros e a prazo") are read by rowwise_tables(),
#   which needs a header row of four or more bare years and exactly one "prazo" column
#   title in the title row; each year token takes the title whose x-span covers it.
#   One table is kept per firm and fiscal year across ALL documents that report it:
#   candidates with a time-deposit row, then unflagged, then balance tables before
#   undiscounted flows, then statement priority (STMT_ORDER: consolidated > prudential >
#   ifrs > individual by default; individual first for ABC and BTG, whose consolidated
#   statements lack buckets in some years or, from 2021, include Banco Pan; prudential
#   first for BMG, whose "Conglomerado Financeiro" table is its group table), then a passing
#   sum check, then the filing of the fiscal year itself over a later filing's comparative
#   column, then the number of buckets. Output column table_id (pdf | page | statement |
#   as-of | header text) identifies the chosen table; the QA file compares, after the unit
#   correction, only balance tables of the same statement across filings.
#   Several tables on one page (a "Banco" and a "Consolidado" block with identical headers)
#   are kept apart by the vertical position of their header. A table's statement is read
#   from its own header block (first cell "Banco" / "Consolidado", or a short heading row
#   inside the block), then from the nearest heading above it on the page, then from the
#   last heading on the previous page (stacked year blocks continuing on the next page),
#   then from the page text. "Conglomerado Financeiro" / "Prudencial" headings are prudential.
#   Bucket integrity: every value column left of the last bucket needs its own bucket header
#   (a wrapped header such as "De 90 a 365" / "dias" is joined over its lines), otherwise the
#   table fails; and a row with a printed total is kept only when its buckets add to that
#   total within rounding (one unit per printed number, or 0.01% of the total if larger).
#   A heading row naming both "Banco" and "Consolidado" gives the statement whose word sits
#   nearest the bucket columns; a joint "Banco e Consolidado" heading is individual.
#   Brazilian numbers: "1.234.567" thousands, "(1.234)" negative, "-"/"--" zero.
#   Units are read from explicit statements ("em milhares", "R$ mil", "(Reais Mil)", "em
#   milhões") on the page and carried forward through the document when a page does
#   not state them; prose mentioning "milhões de reais" does not count.
#
# COVERAGE
#   Banco do Brasil, Banrisul, BTG Pactual, Santander Brasil, Bradesco, Itaú
#   Unibanco Holding, Banco Pan, BMG, Banco ABC Brasil, Daycoval; fiscal years
#   2016-2024 (default). Whatever a bank does not disclose in this note layout
#   appears in the log file with the reason.
#
# OUTPUTS (raw data area, nothing in the estimation pipeline reads them)
#   RAW/DEPOSIT_MATURITY/cvm/<firm_key>/<doc>.pdf        downloaded statements
#   RAW/DEPOSIT_MATURITY/cvm/_index/                      DFP / IPE index CSVs
#   RAW/DEPOSIT_MATURITY/cvm/_parsed/<firm_key>_<doc>.json parsed tables per PDF
#   RAW/DEPOSIT_MATURITY/deposit_maturity_cvm.csv         firm x year x instrument x bucket
#   RAW/DEPOSIT_MATURITY/deposit_maturity_cvm_qa.csv      cross-filing agreement
#   RAW/DEPOSIT_MATURITY/deposit_maturity_cvm_log.csv     per firm-year status
#
# CLI
#   python scrape_deposit_maturity.py cvm                       # all firms, 2016-2024
#   python scrape_deposit_maturity.py cvm --firms bb,banrisul --years 2020-2022
#   python scrape_deposit_maturity.py cvm --refresh             # re-download and re-parse
###─────────────────────────────────────────────────────────────────────────────
cvm_log = logging.getLogger("scrape_deposit_maturity_cvm")

OUT_DIR = RAW / "DEPOSIT_MATURITY"
CACHE = OUT_DIR / "cvm"
IDX_DIR = CACHE / "_index"
PARSED = CACHE / "_parsed"
OUT_CSV = OUT_DIR / "deposit_maturity_cvm.csv"
QA_CSV = OUT_DIR / "deposit_maturity_cvm_qa.csv"
LOG_CSV = OUT_DIR / "deposit_maturity_cvm_log.csv"

DFP_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{y}.zip"
IPE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_{y}.zip"
ENET_DFP = ("https://www.rad.cvm.gov.br/ENET/frmDownloadDocumento.aspx"
            "?CodigoInstituicao=1&NumeroSequencialDocumento={id}")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
THROTTLE_S = 2.0

CVM_FIRMS = {  # firm_key -> (CNPJ root, display)
    "bb": ("00000000", "Banco do Brasil"),
    "banrisul": ("92702067", "Banrisul"),
    "btg": ("30306294", "BTG Pactual"),
    "santander_br": ("90400888", "Santander Brasil"),
    "bradesco": ("60746948", "Bradesco"),
    "itau": ("60872504", "Itaú Unibanco Holding"),
    "pan": ("59285411", "Banco Pan"),
    "bmg": ("61186680", "Banco BMG"),
    "abc": ("28195667", "Banco ABC Brasil"),
    "daycoval": ("62232889", "Daycoval"),
}
MONTH_D, YEAR_D = 30, 360
PARSER_VERSION = 19   # bump when page parsing changes; cached parses of other versions are redone
STMT_ORDER_DEFAULT = ("consolidated", "prudential", "ifrs", "individual", "unknown")
STMT_ORDER = {   # firm-specific statement priority (user's decisions of 2026-10-14)
    "abc": ("individual", "consolidated", "prudential", "ifrs", "unknown"),
    "btg": ("individual", "consolidated", "prudential", "ifrs", "unknown"),
    "bmg": ("prudential", "consolidated", "ifrs", "individual", "unknown"),
}


def stmt_rank(firm, stmt):
    order = STMT_ORDER.get(firm, STMT_ORDER_DEFAULT)
    return order.index(stmt) if stmt in order else len(order)

_last_req = [0.0]
_jar = http.cookiejar.CookieJar()
_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_jar))


# ── helpers ─────────────────────────────────────────────────────────────────
def _fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def fetch(url, timeout=180, retries=3):
    """Polite GET with a browser User-Agent and cookies; bytes or None."""
    for a in range(1, retries + 1):
        wait = THROTTLE_S - (time.time() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with _opener.open(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            cvm_log.warning(f"GET failed ({a}/{retries}) {url[:110]}: {e}")
            time.sleep(3 * a)
    return None


def load_index(kind, year, refresh):
    """Cached CVM index rows for DFP or IPE; list of dicts."""
    IDX_DIR.mkdir(parents=True, exist_ok=True)
    p = IDX_DIR / f"{kind}_{year}.csv"
    if refresh or not p.exists():
        raw = fetch((DFP_URL if kind == "dfp" else IPE_URL).format(y=year))
        if raw is None:
            return []
        try:
            z = zipfile.ZipFile(io.BytesIO(raw))
            name = [n for n in z.namelist() if re.fullmatch(rf"(dfp|ipe)_cia_aberta_{year}\.csv", n)][0]
            p.write_bytes(z.read(name))
        except Exception as e:  # noqa: BLE001
            cvm_log.warning(f"{kind} {year} index unreadable: {e}")
            return []
    with open(p, encoding="latin-1", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def _atomic_write(path, data):
    """Write via a temporary name so an interrupted run never leaves a short PDF."""
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def cnpj_root(s):
    return re.sub(r"\D", "", s)[:8]


# ── column header and value parsing ─────────────────────────────────────────
_NUM = re.compile(r"^\(?-?\d{1,3}(?:\.\d{3})+(?:,\d+)?\)?$|^\(?-?\d+(?:,\d+)?\)?$")
_DASH = re.compile(r"^(?:-{1,3}|—|–)$")
_DATE = re.compile(r"^\d{1,2}[./]\d{1,2}[./]\d{2,4}$")


def cvm_parse_num(tok):
    """Brazilian number token -> float, None if not a value token."""
    t = tok.strip()
    if _DASH.match(t):
        return 0.0
    if t.endswith("%") or _DATE.match(t) or re.fullmatch(r"\(\d\)", t) or not _NUM.match(t):
        return None
    neg = t.startswith("(") or t.startswith("-")
    t = t.strip("()-").replace(".", "").replace(",", ".")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def cvm_parse_bucket(label):
    """Header text -> ('total'|'pct'|'bucket', lo_days, hi_days) or None."""
    s = _fold(label).strip(" :-")
    s = re.sub(r"\s+", " ", s)
    if s in ("total", "total geral", "saldo", "saldo contabil"):
        return ("total", None, None)
    if re.fullmatch(r"(part\.?|%|part %|participacao.*)", s):
        return ("pct", None, None)
    if re.search(r"sem vencimento|sem prazo|a vista|indeterminad|sem venc", s):
        return ("bucket", 0, 0)

    def days(n, unit):
        n = int(n)
        if unit.startswith("dia"):
            return n
        if unit.startswith("mes") or unit.startswith("m"):
            return n * MONTH_D
        return n * YEAR_D

    unit_rx = r"(dias?|mes(?:es)?|meses|anos?)"
    m = re.search(rf"ate (\d+) {unit_rx}", s) or re.search(rf"ate (\d+)\s*({unit_rx})?", s)
    if m and not re.search(r"\d+ a \d+|de \d+", s):
        u = m.group(2) if m.lastindex and m.lastindex >= 2 and m.group(2) else None
        if u:
            return ("bucket", 0, days(m.group(1), u))
    m = re.search(rf"(?:de |entre )?(\d+)\s*(?:a|e|ate|-|–) ?(\d+) {unit_rx}", s)
    if m:
        return ("bucket", days(m.group(1), m.group(3)), days(m.group(2), m.group(3)))
    m = re.search(rf"(?:acima de|mais de|superior a|acima|apos|> ?) ?(\d+) {unit_rx}", s)
    if m:
        return ("bucket", days(m.group(1), m.group(2)), None)
    m = re.search(rf"(?:inferior a|menos de|ate) (\d+) {unit_rx}", s)
    if m:
        return ("bucket", 0, days(m.group(1), m.group(2)))
    # unitless day ranges ("0-30", "31-180", "acima de 365"; the unit sits on a wrapped line)
    m = re.fullmatch(r"(\d+) ?[-–] ?(\d+)(?: dias)?", s)
    if m:
        return ("bucket", int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"(?:acima de|acima|mais de) ?(\d+)(?: dias)?", s)
    if m:
        return ("bucket", int(m.group(1)), None)
    return None


_VOCAB = re.compile(
    r"^(-|–|ate|de|a|e|entre|acima|apos|mais|que|sem|vencimento|prazo|indeterminad[oa]|vista|à|"
    r"dias?|m[eê]s(?:es)?|anos?|total|part\.?|%|geral|superior|inferior|menos|\d{1,3}|\d{1,3} ?[-–] ?\d{1,3})$", re.I)
_DEPO = re.compile(r"dep[oó]sitos?", re.I)


def cvm_instrument_of(label):
    f = _fold(label)
    if "interfinanc" in f:
        return "interbank"
    if "prazo" in f:
        return "time_deposits"
    if "poupanca" in f:
        return "savings"
    if re.search(r"\bavista\b|a vista|a vista", f):
        return "demand"
    if re.fullmatch(r"(total de )?depositos( totais)?( \(.*\))?|total depositos|depositos totais", f.strip()):
        return "total_deposits"
    return label.strip()


# ── page -> tables ──────────────────────────────────────────────────────────
def page_words_text(page):
    """Words in displayed orientation (pages with /Rotate 90 or 270 are landscape
    tables whose text runs vertically in the unrotated page space)."""
    out = []
    for w in page.get_text("words"):
        r = fitz.Rect(w[0], w[1], w[2], w[3])
        if page.rotation:
            r = (r * page.rotation_matrix).normalize()
        out.append((r.x0, r.y0, r.x1, r.y1, w[4]))
    return out


def group_rows(words):
    """Group (x0,y0,x1,y1,txt) words into visual rows sorted top to bottom."""
    ws = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows, cur, cy = [], [], None
    for w in ws:
        yc = (w[1] + w[3]) / 2
        h = max(w[3] - w[1], 4.0)
        if cur and abs(yc - cy) <= 0.45 * h:
            cur.append(w)
            cy = (cy * (len(cur) - 1) + yc) / len(cur)
        else:
            if cur:
                rows.append(sorted(cur, key=lambda z: z[0]))
            cur, cy = [w], yc
    if cur:
        rows.append(sorted(cur, key=lambda z: z[0]))
    return rows


def row_text(r):
    return " ".join(w[4] for w in r)


def split_label_values(r):
    """A data row -> (label, [(xc, value)]) with value tokens after the label."""
    label_parts, vals = [], []
    for w in r:
        v = cvm_parse_num(w[4])
        if v is not None and (label_parts or vals):
            if not vals and re.fullmatch(r"\(?\d{1,2}\)?", w[4]) and len(w[4]) <= 3 and w[4].startswith("("):
                continue  # footnote marker
            vals.append(((w[0] + w[2]) / 2, v))
        elif v is None and not vals:
            label_parts.append(w[4])
        elif v is None and vals and w[4].endswith("%"):
            continue
    return " ".join(label_parts).strip(), vals


def segment_header(words, with_x=False):
    """Reading-order header words -> list of (kind, lo, hi, text) phrases. `words` are
    strings, or word tuples (x0, y0, x1, y1, text) when with_x is True; then each phrase
    also carries its x-centre as a fifth element."""
    phrases, cur, cur_x = [], [], []
    closers = re.compile(r"^(dias?|mes|meses|ano|anos|vencimento|total|%|vista|geral)$")
    for w in words:
        t = w[4] if with_x else w
        f = _fold(t)
        if not (_VOCAB.match(t) or _VOCAB.match(f)):
            cur, cur_x = [], []
            continue
        cur.append(f)
        if with_x:
            cur_x.append((w[0], w[2]))
        if closers.match(f) or (f == "prazo" and len(cur) >= 2 and cur[-2] == "sem"):
            pb = cvm_parse_bucket(" ".join(cur))
            if pb:
                ph = pb + (" ".join(cur),)
                if with_x:
                    ph += ((cur_x[0][0] + cur_x[-1][1]) / 2,)
                phrases.append(ph)
            cur, cur_x = [], []
    return phrases


def is_data_row(r):
    """True when the row carries at least two value-formatted numbers (thousands
    grouping, parentheses, decimals or dashes), as opposed to header numerals."""
    n = 0
    for i, w in enumerate(r):
        t = w[4]
        if _DASH.match(t):
            nb = [r[j][4] for j in (i - 1, i + 1) if 0 <= j < len(r)]
            if len(nb) == 2 and all(re.fullmatch(r"\d{1,3}", q) for q in nb):
                continue  # range dash in a header such as "0 - 30"
            n += 1
        elif cvm_parse_num(t) is not None and re.search(r"[.,()]", t):
            n += 1
    return n >= 2


def header_info(rows, ri):
    """Locate the header block above data row `ri`.
    Returns None or dict(anchor, top, block, asof_ctx)."""
    anchor = None
    for k in range(ri - 1, max(ri - 45, -1), -1):
        ft = _fold(row_text(rows[k]))
        if is_data_row(rows[k]):
            continue
        if re.search(r"ate \d|\d a \d+ (mes|ano|dia)|acima de|sem vencimento|mais de \d|vencimento", ft):
            anchor = k
            break
    if anchor is None:
        return None
    top = anchor
    while top - 1 >= 0 and rows[anchor][0][1] - rows[top - 1][-1][3] < 16 and anchor - top < 5:
        if is_data_row(rows[top - 1]):
            break
        top -= 1
    block = list(rows[top:anchor + 1])
    k = anchor + 1
    while k < ri and k - anchor <= 2:
        t = _fold(row_text(rows[k]))
        if is_data_row(rows[k]):
            break
        if re.fullmatch(r"[a-z0-9 .%()/\-–]*", t) and re.search(r"mes|ano|dia|total|%", t):
            block.append(rows[k])
        k += 1
    return dict(anchor=anchor, top=top, block=block)


def header_columns(rows, ri, val_x):
    """Per-column headers for data row `ri` (numeric columns at val_x): the buckets
    in order plus the total column when present. Returns (cols, ctx) with cols a
    list of (kind, lo, hi, text) aligned to the values, or None."""
    hi_ = header_info(rows, ri)
    if hi_ is None:
        return None, ""
    block, anchor, top = hi_["block"], hi_["anchor"], hi_["top"]
    ctx = " ".join(row_text(r) for r in rows[max(top - 8, 0):anchor + 1])
    # 1. reading order on the anchor row
    seq = [p for p in segment_header(rows[anchor], with_x=True) if p[0] != "pct"]
    nb = sum(1 for p in seq if p[0] == "bucket")
    n_seq = len(seq)
    if nb >= 2 and len(val_x) >= nb and len(val_x) - n_seq <= 2:
        # the phrases cover every value column but at most the total and a prior-year
        # total. Each phrase goes to the value column nearest its centre; when that
        # assignment is one-to-one and in order, a value column left of the last bucket
        # without a phrase means a wrapped bucket header ("De 90 a 365" / "dias") whose
        # first line is not on the anchor row: the wrapped-header pass below reads it.
        near = [min(range(len(val_x)), key=lambda q: abs(val_x[q] - p[4])) for p in seq]
        positional = all(b > a for a, b in zip(near, near[1:]))
        last_b = max(i for i, p in enumerate(seq) if p[0] == "bucket")
        if positional:
            cols = [None] * len(val_x)
            for p, j in zip(seq, near):
                cols[j] = p[:4]
            hole = any(c is None for c in cols[:near[last_b]])
            if not hole:
                j = near[last_b] + 1
                if j < len(cols) and cols[j] is None:
                    cols[j] = ("total", None, None, "inferred total")
                return cols, ctx
        else:
            cols = [p[:4] for p in seq][:len(val_x)]
            while len(cols) < len(val_x):
                cols.append(("total", None, None, "inferred total") if len(cols) == nb else None)
            return cols, ctx
    # 2. horizontal assignment for headers wrapped over several rows
    cols = [[] for _ in val_x]
    for r in block:
        # header words of one line separated by a word space form one phrase ("De 3 a 12"); a
        # phrase goes to the value column nearest its centre, so a centred header over
        # right-aligned numbers does not split between two columns
        phrase = []
        for w in [w for w in r if _VOCAB.match(w[4]) or _VOCAB.match(_fold(w[4]))] + [None]:
            if phrase and (w is None or w[0] - phrase[-1][2] > 0.29 * max(phrase[-1][3] - phrase[-1][1], 4.0)):
                xc = (phrase[0][0] + phrase[-1][2]) / 2
                j = min(range(len(val_x)), key=lambda q: abs(val_x[q] - xc))
                for q in phrase:
                    cols[j].append(((q[1] + q[3]) / 2, q[0], q[4]))
                phrase = []
            if w is not None:
                phrase.append(w)
    out = []
    for c in cols:
        c.sort()
        text = " ".join(t for _, _, t in c)
        if len(re.findall(r"\d+", text)) > 2 or sum(1 for q in segment_header(text.split()) if q[0] == "bucket") > 1:
            return None, ctx   # one column collected several bucket phrases: header from another table
        pb = cvm_parse_bucket(text)
        out.append(pb + (text,) if pb else None)
    last = max((j for j, c in enumerate(out) if c and c[0] == "bucket"), default=-1)
    if last < 0 or any(c is None for c in out[:last]):
        return None, ctx
    out = out[:last + 1] + [c for c in out[last + 1:]]
    if last + 1 < len(out) and out[last + 1] is None:
        out[last + 1] = ("total", None, None, "inferred total")
    return out, ctx


_DATE_RX = re.compile(r"(\d{1,2})[./](\d{1,2})[./](\d{4})|(\d{1,2}) de (dezembro|junho|setembro|marco|março) de (\d{4})",
                      re.I)


def asof_from(ctx):
    """Latest 31 December date mentioned in `ctx` -> ISO string or None."""
    found = []
    for m in _DATE_RX.finditer(ctx):
        if m.group(1):
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            d, y = int(m.group(4)), int(m.group(6))
            mo = {"dezembro": 12, "junho": 6, "setembro": 9, "marco": 3, "março": 3}[m.group(5).lower()]
        found.append((y, mo, d))
    dec = [f for f in found if f[1] == 12 and f[2] == 31]
    if dec:
        y = dec[0][0]
        return f"{y}-12-31"
    return None


def unit_of(txt):
    """Unit stated on a page. Only explicit statements in short lines count ("(Em milhares
    de reais)", "R$ mil", "(Reais Mil)", "Valores expressos em milhões de reais"); prose
    sentences that mention "milhões de reais" or "em reais" do not."""
    for line in txt.splitlines():
        f = _fold(line).strip()
        if not f or len(f.split()) > 12:
            continue
        if re.search(r"em milhares|milhares de reais|r\$ mil\b|r\$ thousand|in thousands|\(reais mil\)|\(em r\$ mil\)", f):
            return "R$ thousand"
        if re.search(r"em milhoes|r\$ milhoes|in millions|\(reais milhoes\)|\(em r\$ milhoes\)|\(r\$ milhoes\)|r\$ mi\b", f):
            return "R$ million"
        if re.search(r"\(em reais\)|expressos em reais|\(reais\)", f):
            return "R$"
    return None


def stmt_of(txt, inherited):
    """Statement a page belongs to. Headings count ("DFs Consolidadas", "Demonstrações
    contábeis consolidadas", "demonstrações financeiras individuais"); a mere mention of
    "controladora", "banco múltiplo" or "individual" in prose does not change a known
    statement (BTG's note 1 says "banco múltiplo" and "controladora indireta" in a
    consolidated filing)."""
    f = _fold(txt[:1800])
    if re.search(r"ifrs|padroes internacionais|international financial", f):
        return "ifrs"
    if re.search(r"conglomerado prudencial|prudencial", f):
        return "prudential"
    if re.search(r"individua\w* e consolidad", f):
        return "consolidated"      # footer "Demonstrações Financeiras Individuais e Consolidadas"
    m = re.search(r"\b(?:dfs|demonstracoes|balancos? patrimoniais?|notas explicativas)\b[^.\n]{0,40}?"
                  r"(consolidad|individua)", f)
    if m:
        return "consolidated" if m.group(1) == "consolidad" else "individual"
    if re.search(r"consolidad", f):
        return "consolidated"
    if inherited == "unknown" and re.search(r"individua|banco multiplo|controladora", f):
        return "individual"
    return inherited


_BARE_YEAR = re.compile(r"^(20[12]\d)$")


def year_tokens(rows, lo, hi):
    """Bare four-digit year tokens ("2017") in rows lo..hi -> [(x_center, 'YYYY-12-31')];
    notes whose tables carry only a year above the header (Daycoval) use them."""
    out = []
    for rr in rows[max(lo, 0):hi]:
        for w in rr:
            if _BARE_YEAR.match(w[4]):
                out.append(((w[0] + w[2]) / 2, f"{w[4]}-12-31"))
    return out


def stmt_above(rows, lo, hi, bx=None):
    """Statement named by a short row just above a table ("BB Consolidado", "Banco",
    "BB Banco Múltiplo", "Conglomerado Financeiro"); None when there is none. A heading row
    naming both statements over different columns (ABC: "Banco" over the buckets and the
    Banco totals, "Consolidado" over two more total columns) gives the statement whose word
    lies nearest `bx`, the x-centre of the table's bucket columns."""
    found = None
    for rr in rows[max(lo, 0):hi]:
        if bx is not None:
            wb = [w for w in rr if _fold(w[4]) == "banco"]
            wc = [w for w in rr if _fold(w[4]).startswith("consolidad")]
            if wb and wc and len(rr) <= 8:
                d = lambda w: abs((w[0] + w[2]) / 2 - bx)   # noqa: E731
                joint = any(0 <= c[0] - b_[2] < 25 for b_ in wb for c in wc)   # "Banco e Consolidado"
                found = ("individual" if joint or min(map(d, wb)) < min(map(d, wc)) else "consolidated")
                continue
        t = re.sub(r"\d{1,2}[./]\d{1,2}[./]\d{4}|\b20[12]\d\b", " ", _fold(row_text(rr)))   # dates, bare years
        t = re.sub(r"\s+", " ", t).strip(" :-")
        toks = t.split()
        if len(toks) > 14:
            continue
        if "prudencial" in t or "conglomerado financeiro" in t:
            found = "prudential"
        elif re.search(r"banco e consolidado", t):
            found = "individual"     # one table for both (ABC): the Banco statement
        elif "consolidad" in t:
            found = "consolidated"
        elif re.fullmatch(r"(bb )?banco( multiplo)?|controladora|individual|banco [a-z ]+", t) or (
                toks and toks[0] == "banco" and len(toks) <= 4
                and not re.match(r"banco (central|do|de|bmg|pan|abc|btg|daycoval|santander|bradesco|itau)", t)
                and not any(re.search(r"\d", q) for q in toks[1:2])):
            found = "individual"
    return found


_OWN_STMT = (("conglomerado prudencial", "prudential"), ("conglomerado financeiro", "prudential"),
             ("prudencial", "prudential"), ("consolidado", "consolidated"), ("controladora", "individual"),
             ("individual", "individual"), ("banco", "individual"), ("bb banco", "individual"))


def stmt_own(rows, top, ri, bx=None):
    """Statement named inside a table's own header block (rows top .. ri-1 that are not data
    rows): the first cell of a header row ("Banco  Até 30 dias ...", "Consolidado
    31/12/2019" on a header line below the anchor, Pan), i.e. the words before the first
    maturity word, date or number, or a short heading row inside the block. The row nearest
    the data row wins; None otherwise."""
    found = None
    for rr in rows[max(top, 0):ri]:
        if is_data_row(rr):
            continue
        lead = []
        for w in rr:
            if _VOCAB.match(w[4]) or _VOCAB.match(_fold(w[4])) or re.search(r"\d", w[4]):
                break
            lead.append(w[4])
        f = re.sub(r"\s+", " ", _fold(" ".join(lead))).strip(" :-")
        hit = None
        for key, st in _OWN_STMT:
            if f == key or f.startswith(key + " ") and len(f.split()) <= 3:
                hit = st
                break
        if hit is None:
            hit = stmt_above([rr], 0, 1, bx) if len(rr) <= 4 else None
        if hit:
            found = hit
    return found


_SEC_ROW = re.compile(r"^(a prazo|prazo|a vista|vista|poupanca|de poupanca|interfinanceiros?|outros depositos)$")


def foots(s, total, n):
    """True when a bucket sum `s` equals the printed `total` within rounding: one unit per
    printed number (n buckets and the total), or 0.01% of the total when that is larger
    (BB 2022 prints buckets that add to 10 thousand more than its 431.6 billion total). A
    lost bucket is far larger (BTG 2022: 49,004 of 88.6 million, 0.055%)."""
    return abs(s - total) <= max(n + 1, 1e-4 * abs(total))


def rowwise_tables(rows, stmt, unit, method, undisc):
    """Tables with maturity buckets as ROWS and instruments as columns (Banco BMG: "Até 30
    dias" ... "Após 360 dias" down the side; "Depósitos interfinanceiros", "Depósitos a
    prazo", "Total" across, each over a 2018 / 2017 year pair). Columns are located by the
    bare-year tokens of the header row; blank cells are zero. Returns table dicts for the
    time-deposit column of every year."""
    out = []
    for yi, yr in enumerate(rows):
        ys = [w for w in yr if _BARE_YEAR.match(w[4])]
        if len(ys) < 4 or len(ys) != len(yr):
            continue
        # the title row: the nearest short row above the years that names time deposits, its
        # "Depósitos" possibly wrapped onto the line above (prose above it, such as "...
        # depósitos a prazo e interfinanceiros:", is not read)
        trow = None
        for k in range(yi - 1, max(yi - 4, -1), -1):
            fk_ = _fold(row_text(rows[k]))
            up = _fold(row_text(rows[k - 1])) if k >= 1 and len(rows[k - 1]) <= 6 else ""
            if len(rows[k]) <= 12 and "prazo" in fk_ and ("deposito" in fk_ or "deposito" in up):
                trow = rows[k]
                break
        if trow is None:
            continue   # the header must name time deposits, not (say) a loan-portfolio maturity table
        # title phrases: words of the title row split at wide gaps ("Depósitos a prazo(i)")
        phrases, cur = [], []
        for w in trow + [None]:
            if cur and (w is None or w[0] - cur[-1][2] > 12):
                ft = re.sub(r"\(.*?\)", "", _fold(" ".join(q[4] for q in cur)))   # footnote "prazo(i)"
                kind = ("prazo" if "prazo" in ft else "interfinanceiros" if "interfinanc" in ft
                        else "total" if "total" in ft else "vista" if "vista" in ft else "?")
                phrases.append((cur[0][0], cur[-1][2], kind))
                cur = []
            if w is not None:
                cur.append(w)
        if sum(1 for p in phrases if p[2] == "prazo") != 1:
            continue   # exactly one time-deposit title over one year pair
        cols = []
        for w in ys:
            xc = (w[0] + w[2]) / 2
            # the title whose x-span covers the year token; else the nearest span edge
            # (a right-aligned "Total" title spans only the last of its two year columns)
            cover = [p for p in phrases if p[0] - 4 <= xc <= p[1] + 4]
            p = cover[0] if cover else min(phrases, key=lambda p: min(abs(xc - p[0]), abs(xc - p[1])))
            cols.append((xc, p[2], w[4]))
        if sum(1 for c in cols if c[1] == "prazo") != 2:
            continue
        data = []
        for rr in rows[yi + 1:yi + 16]:
            xmin = min(c[0] for c in cols) - 35
            label = " ".join(w[4] for w in rr if (w[0] + w[2]) / 2 < xmin)
            vals = [((w[0] + w[2]) / 2, cvm_parse_num(w[4])) for w in rr
                    if (w[0] + w[2]) / 2 >= xmin and cvm_parse_num(w[4]) is not None]
            lf = _fold(label).strip()
            if lf.startswith("total"):
                data.append(("total", None, None, vals))
                break
            if vals and "circulante" not in lf:
                pb = cvm_parse_bucket(label)
                if pb and pb[0] == "bucket":
                    data.append(("bucket", pb[1], pb[2], vals, label))
        if len(data) < 4 or data[-1][0] != "total":
            continue
        def at(vals, xc):
            c = [v for x, v in vals if abs(x - xc) <= 26]
            return c[0] if c else 0.0
        stmt_t = stmt_above(rows, yi - 10, yi) or stmt
        for xc, title, yr_s in cols:
            if title != "prazo":
                continue
            bks = [dict(label=d[4], lo=d[1], hi=d[2], value=at(d[3], xc)) for d in data[:-1]]
            if not foots(sum(b["value"] for b in bks), at(data[-1][3], xc), len(bks)):
                continue
            out.append(dict(row_y=rows[yi][0][1], hdr_y=rows[yi][0][1],
                            label="Depósitos a prazo (maturity rows)", asof=f"{yr_s}-12-31",
                            stmt=stmt_t, unit=unit, buckets=bks, total=at(data[-1][3], xc),
                            ctx=" ".join(row_text(r_) for r_ in rows[max(yi - 6, 0):yi + 1])[-160:],
                            method=method, undisc=undisc))
    return out


def parse_page(page, words, page_text, ctx_state, method):
    """All parsed tables on one page: list of dicts."""
    rows = group_rows(words)
    undisc = bool(re.search(r"nao descontad", _fold(page_text)))   # undiscounted-cash-flow (liquidity) table
    stmt = stmt_of(page_text, ctx_state["stmt"])
    unit = unit_of(page_text) or ctx_state["unit"]
    tables = []
    in_depo = False
    for ri, r in enumerate(rows):
        label, vals = split_label_values(r)
        fl = re.sub(r"[^a-z ]", "", _fold(label)).strip()
        fl = re.sub(r"\s+", " ", fl)
        fl = re.sub(r"^[a-z] (?=depositos)", "", fl)   # note sub-heading "b) Depósitos"
        if fl.startswith("depositos"):
            in_depo = True
        elif re.match(r"(captac|recursos de aceite|obrigac|relacoes|divid|instrumentos)", fl):
            in_depo = False
        # rows named only "A Prazo" / "À Vista" / "Poupança" under a "Depósitos" heading
        if in_depo and _SEC_ROW.match(fl) and not _DEPO.search(label):
            label = "Depósitos " + re.sub(r"\s*\(\d\)\s*$", "", label).strip()
        if len(vals) < 3 or not _DEPO.search(label):
            if not (len(vals) >= 3 and re.search(r"poupan|a vista|à vista", _fold(label))):
                continue
        hdr, ctx = header_columns(rows, ri, [x for x, _ in vals])
        if hdr is None or sum(1 for h in hdr if h and h[0] == "bucket") < 2:
            continue
        vals_h = list(vals)
        lead_tot = None
        if hdr[0] is not None and hdr[0][0] == "total":
            # layout with the total in the first column (BTG): that total belongs to the
            # buckets that follow; a later "Total" is the prior-year column
            lead_tot = vals_h[0][1]
            hdr, vals_h = hdr[1:], vals_h[1:]
        # split into groups at each "total" column (e.g. two year-end blocks side by side)
        groups, cur = [], []
        for h, (xc, v) in zip(hdr, vals_h):
            if h is None:
                if cur:
                    groups.append(cur)
                cur = []
                continue
            cur.append((h, xc, v))
            if h[0] == "total":
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        hi_ = header_info(rows, ri)
        anc = hi_["anchor"] if hi_ else ri
        top = hi_["top"] if hi_ else ri
        date_toks = []
        if hi_:
            for rr in rows[max(hi_["top"] - 3, 0):hi_["anchor"] + 1]:
                for w in rr:
                    if _DATE.match(w[4]) and asof_from(w[4]):
                        date_toks.append(((w[0] + w[2]) / 2, asof_from(w[4])))
        yr_toks = year_tokens(rows, top - 3, anc + 1) if hi_ else []
        # statement: the table's own header block ("Banco  Até 30 dias", a "Consolidado" row
        # inside the block), then the nearest heading above it on the page (a heading sits above
        # the FIRST of several stacked year blocks), then the last heading of the previous page
        # (stacked blocks continuing on the next page), then the page text
        bxs = [xc for h, (xc, _) in zip(hdr, vals_h) if h and h[0] == "bucket"]
        bx = sum(bxs) / len(bxs) if bxs else None
        stmt_t = (stmt_own(rows, top, ri, bx) or stmt_above(rows, top - 4, top, bx)
                  or stmt_above(rows, top - 60, top - 4, bx))
        if stmt_t:
            ctx_state["heading"] = (stmt_t, ctx_state.get("page"))
        else:
            hd = ctx_state.get("heading")
            carry = hd and hd[1] is not None and ctx_state.get("page") is not None \
                and ctx_state["page"] - hd[1] in (0, 1)
            stmt_t = hd[0] if carry else stmt
        # a total column standing alone after the bucket group is a prior-year total, not a
        # second year block
        n_bgroups = sum(1 for g in groups if sum(1 for h, _, _ in g if h[0] == "bucket") >= 2)
        if n_bgroups == 1:
            groups = [g for g in groups if sum(1 for h, _, _ in g if h[0] == "bucket") >= 2]
        for gi, g in enumerate(groups):
            buckets = [(h, v) for h, _, v in g if h[0] == "bucket"]
            if len(buckets) < 2:
                continue
            total = [v for h, _, v in g if h[0] == "total"]
            if lead_tot is not None and gi == 0:
                total = [lead_tot]
            gx = sum(x for _, x, _ in g) / len(g)
            asof = None
            toks = date_toks or yr_toks
            if len(groups) > 1 and toks:
                asof = min(toks, key=lambda t: abs(t[0] - gx))[1]
            elif len(groups) > 1 and lead_tot is None:
                continue
            else:
                # header lines first (Pan: "31/12/2019 31/12/2018" sits on the middle line of the
                # header block, below the anchor; the bucket columns belong to the first date),
                # then upwards
                for k in list(range(anc + 1, ri)) + list(range(anc, max(anc - 9, -1), -1)):
                    if re.match(r"^DFP", row_text(rows[k])) or (k > anc and is_data_row(rows[k])):
                        continue
                    asof = asof_from(row_text(rows[k]))
                    if asof:
                        break
                if not asof and yr_toks:
                    asof = max(t[1] for t in yr_toks)
            tot_v = total[0] if total else None
            if tot_v is not None and not foots(sum(v for _, v in buckets), tot_v, len(buckets)):
                continue   # buckets do not add to the printed total: a lost or shifted column
            tables.append(dict(row_y=rows[ri][0][1], hdr_y=rows[anc][0][1], label=label, asof=asof,
                               stmt=stmt_t, unit=unit,
                               buckets=[dict(label=h[3], lo=h[1], hi=h[2], value=v) for h, v in buckets],
                               total=total[0] if total else None, ctx=ctx[-160:], method=method,
                               undisc=undisc))
    tables += rowwise_tables(rows, stmt, unit, method, undisc)
    return tables


def ocr_words(page, dpi=300):
    import pytesseract
    from PIL import Image
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    d = pytesseract.image_to_data(img, lang="por", config="--psm 6", output_type=pytesseract.Output.DICT)
    sc = 72.0 / dpi
    words = []
    for i, t in enumerate(d["text"]):
        if t.strip() and float(d["conf"][i]) >= 0:
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            words.append((x * sc, y * sc, (x + w) * sc, (y + h) * sc, t.strip()))
    return words


_SCREEN_DEP = re.compile(r"dep[oó]sitos?|poupan", re.I)
_SCREEN_MAT = re.compile(r"acima de|\d+ ?[-–] ?\d+ dias|vencimento|at[eé] \d+ (dias|m[eê]s|meses|ano)|de \d+ a \d+|acima de \d|\d+ a \d+ (meses|anos|dias)",
                         re.I)


def parse_pdf(path):
    """Parse every candidate page of one PDF. Returns list of table dicts with page."""
    doc = fitz.open(str(path))
    n = len(doc)
    sample = [len(doc[i].get_text()) for i in range(0, n, max(n // 25, 1))]
    scanned = sum(1 for c in sample if c < 80) > 0.6 * len(sample)
    state = {"stmt": "unknown", "unit": None, "heading": None, "page": None}
    out = []
    for i in range(n):
        page = doc[i]
        state["page"] = i
        if scanned:
            if i >= 400:
                break
            words = ocr_words(page, 150)
            txt = " ".join(w[4] for w in words)
            method = "ocr"
        else:
            txt = page.get_text()
            words = None
            method = "text_layer"
        st = stmt_of(txt, state["stmt"])
        if re.search(r"demonstra[cç][oõ]es? (cont[aá]beis|financeiras)", txt[:600], re.I) or st != "unknown":
            state["stmt"] = st
        u = unit_of(txt)
        if u:
            state["unit"] = u
        if not (_SCREEN_DEP.search(txt) and _SCREEN_MAT.search(txt)):
            continue
        if scanned:
            words = ocr_words(page, 300)
        else:
            words = page_words_text(page)
        for t in parse_page(page, words, txt, state, method):
            t["page"] = i + 1
            out.append(t)
    doc.close()
    return out, scanned


# ── table selection, validation ─────────────────────────────────────────────
def group_tables(tables):
    """Collapse per-row detections into tables: same page, header position, asof, bucket
    labels and statement. The header's vertical position keeps apart two blocks with
    identical headers on one page (a "Banco" and a "Consolidado" table)."""
    groups = {}
    for t in tables:
        sig = (t["page"], round(t.get("hdr_y", 0.0), 1), t["asof"], tuple((b["label"]) for b in t["buckets"]),
               t["stmt"])
        g = groups.setdefault(sig, dict(page=t["page"], asof=t["asof"], stmt=t["stmt"], unit=t["unit"],
                                        method=t["method"], rows=[], ctx=t["ctx"],
                                        undisc=t.get("undisc", False)))
        instr = cvm_instrument_of(t["label"])
        if any(r["instrument"] == instr for r in g["rows"]):
            continue  # a second block under the same header (e.g. compulsory-deposit deductions)
        g["rows"].append(dict(label=t["label"], instrument=instr, buckets=t["buckets"], total=t["total"]))
    return list(groups.values())


CVM_SUM_TOL = 0.005       # buckets must add to the printed total within 0.5%
PLAUS_FLOOR = 0.10    # time deposits below 10% of the bank's total deposits are implausible
_UNIT_SCALE = {"R$": 1e-3, "R$ thousand": 1.0, "R$ million": 1e3}


def row_value(r):
    """A row's printed total, else the sum of its buckets."""
    return r["total"] if r["total"] is not None else sum(b["value"] for b in r["buckets"])


def check_row(row):
    """(sum_gap, total): sum_gap = |sum(buckets) - total| / |total| as a float (the table
    passes at <= SUM_TOL); None when the table prints no total."""
    tot = row["total"]
    s = sum(b["value"] for b in row["buckets"])
    if tot is None:
        return None, None
    if tot == 0:
        return (0.0 if s == 0 else float("inf")), tot
    return abs(s - tot) / abs(tot), tot


def deposit_reference(g):
    """Total deposits implied by one table, in the table's unit: the total_deposits row,
    else the sum of the deposit rows (demand, savings, interbank, other deposits, time)."""
    for r in g["rows"]:
        if r["instrument"] == "total_deposits":
            return row_value(r)
    oth = [r for r in g["rows"] if r["instrument"] != "time_deposits"
           and re.search(r"dep|poupan|vista|interbank|savings|demand", _fold(r["label"] + r["instrument"]))]
    if len(oth) >= 2 or any(r["instrument"] in ("savings", "demand") for r in oth):
        return sum(row_value(r) for r in oth) + sum(row_value(r) for r in g["rows"]
                                                    if r["instrument"] == "time_deposits")
    return None


def plausibility(x, doc_tables):
    """Flag text for a candidate table x (dict with g) or '' when plausible. The reference
    is the same table's total deposits, else another table of the same document and
    fiscal year (unit-converted)."""
    g = x["g"]
    td = [r for r in g["rows"] if r["instrument"] == "time_deposits"]
    if not td:
        return ""
    v = row_value(td[0])
    if v <= 0:
        return f"negative or zero time-deposit total ({v:,.0f})"
    ref, where = deposit_reference(g), "same table"
    sc = _UNIT_SCALE.get(g["unit"] or "", None)
    if ref is None:
        for y in doc_tables:
            if y is x:
                continue
            r2 = deposit_reference(y["g"])
            s2 = _UNIT_SCALE.get(y["g"]["unit"] or "", None)
            if r2 and s2 and sc and y["g"]["stmt"] == g["stmt"] and r2 > 0 and y["g"]["unit"] == g["unit"]:
                ref, where = r2 * s2 / sc, "same document"
                break
    if ref is None or ref <= 0:
        return ""
    if v < PLAUS_FLOOR * ref:
        return (f"implausible: time deposits {v:,.0f} are {100 * v / ref:.1f}% of total deposits "
                f"{ref:,.0f} ({where}); below {100 * PLAUS_FLOOR:.0f}%")
    return ""


def table_score(firm, fy, x):
    """Sort key of a candidate x (dict with g, doc_year, flag); smaller is better."""
    g = x["g"]
    td = [r for r in g["rows"] if r["instrument"] == "time_deposits"]
    td_ok = any((check_row(r)[0] is not None and check_row(r)[0] <= CVM_SUM_TOL) for r in td)
    nb = max(len(r["buckets"]) for r in g["rows"])
    return (not td, bool(x["flag"]), g.get("undisc", False), stmt_rank(firm, g["stmt"]), not td_ok,
            x["doc_year"] != fy, -nb, x["doc_year"], g["page"])


def table_id(x):
    g = x["g"]
    title = re.sub(r"\s+", " ", g["ctx"]).strip()[-70:]
    return f"{x['pdf']}|p{g['page']}|{g['stmt']}|{g['asof']}|{title}"


# ── main ────────────────────────────────────────────────────────────────────
def download_dfp_pdf(firm, row, refresh):
    d = CACHE / firm
    d.mkdir(parents=True, exist_ok=True)
    fy_file = f"dfp_{row['DT_REFER'][:4]}_v{row['VERSAO']}_{row['ID_DOC']}.pdf"
    p = d / fy_file
    if p.exists() and not refresh:
        return p, ENET_DFP.format(id=row["ID_DOC"]), None
    raw = fetch(ENET_DFP.format(id=row["ID_DOC"]), timeout=600)
    if raw is None:
        return None, ENET_DFP.format(id=row["ID_DOC"]), "download failed"
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
        pdfs = [n for n in z.namelist() if n.lower().endswith(".pdf")]
        if not pdfs:
            return None, ENET_DFP.format(id=row["ID_DOC"]), "zip has no PDF"
        _atomic_write(p, z.read(max(pdfs, key=lambda n: z.getinfo(n).file_size)))
    except zipfile.BadZipFile:
        if raw[:4] == b"%PDF":
            _atomic_write(p, raw)
        else:
            return None, ENET_DFP.format(id=row["ID_DOC"]), "response is not a zip/PDF"
    return p, ENET_DFP.format(id=row["ID_DOC"]), None


def download_ipe_pdf(firm, row, refresh):
    d = CACHE / firm
    d.mkdir(parents=True, exist_ok=True)
    m = re.search(r"numSequencia=(\d+)&numVersao=(\d+)", row["Link_Download"])
    tag = f"{m.group(1)}_v{m.group(2)}" if m else re.sub(r"\W", "", row["Protocolo_Entrega"])
    p = d / f"ipe_{row['Data_Referencia'][:4]}_{tag}.pdf"
    if p.exists() and not refresh:
        return p, None
    raw = fetch(row["Link_Download"], timeout=600)
    if raw is None:
        return None, "download failed"
    if raw[:4] == b"%PDF":
        _atomic_write(p, raw)
        return p, None
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
        pdfs = [n for n in z.namelist() if n.lower().endswith(".pdf")]
        if pdfs:
            _atomic_write(p, z.read(max(pdfs, key=lambda n: z.getinfo(n).file_size)))
            return p, None
    except zipfile.BadZipFile:
        pass
    return None, "response is not a PDF"


def parsed_for(pdf, firm, refresh):
    PARSED.mkdir(parents=True, exist_ok=True)
    jp = PARSED / f"{firm}_{pdf.stem}.json"
    if jp.exists() and not refresh:
        cached = json.loads(jp.read_text(encoding="utf-8"))
        if cached.get("version") == PARSER_VERSION:
            return cached
    try:
        tables, scanned = parse_pdf(pdf)
        res = dict(tables=group_tables(tables), scanned=scanned, error=None, version=PARSER_VERSION)
    except Exception as e:  # noqa: BLE001
        res = dict(tables=[], scanned=False, error=f"{type(e).__name__}: {e}", version=PARSER_VERSION)
    jp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


def cvm_main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--firms", default=",".join(CVM_FIRMS))
    ap.add_argument("--years", default="2016-2024")
    a = ap.parse_args()
    firms = [f.strip() for f in a.firms.split(",") if f.strip()]
    y0, y1 = (int(x) for x in a.years.split("-")) if "-" in a.years else (int(a.years),) * 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dfp = {y: load_index("dfp", y, a.refresh) for y in range(y0, y1 + 2)}
    # filing per (firm, reference year): highest VERSAO, latest receipt
    filings = {}
    for fy_file, rows in dfp.items():
        for r in rows:
            for fk in firms:
                if cnpj_root(r["CNPJ_CIA"]) == CVM_FIRMS[fk][0] and r["DT_REFER"].endswith("12-31"):
                    k = (fk, int(r["DT_REFER"][:4]))
                    if k not in filings or int(r["VERSAO"]) > int(filings[k]["VERSAO"]):
                        filings[k] = r

    # documents to read: filing for reference year Y-1.. covers Y and Y-1; read ref years y0..y1+1
    results = {}   # (firm, fy) -> list of dict(doc_year, source, url, pdf, group)
    status = {}    # (firm, doc_year) -> (found, msg)
    for fk in firms:
        for ref in range(y0, y1 + 2):
            r = filings.get((fk, ref))
            if r is None:
                status[(fk, ref)] = (False, "no DFP filing in CVM index")
                continue
            pdf, url, err = download_dfp_pdf(fk, r, a.refresh)
            if pdf is None:
                status[(fk, ref)] = (False, err)
                continue
            res = parsed_for(pdf, fk, a.refresh)
            status[(fk, ref)] = (True, res["error"] or ("scanned" if res["scanned"] else "ok"))
            cvm_log.info(f"[{fk}] DFP {ref}: {len(res['tables'])} candidate tables"
                     f"{' (scanned)' if res['scanned'] else ''}")
            for g in res["tables"]:
                fy = int(g["asof"][:4]) if g["asof"] else None
                if fy is None:
                    fy = ref
                    g["asof"] = f"{ref}-12-31"
                    g["_asof_inferred"] = True
                results.setdefault((fk, fy), []).append(
                    dict(doc_year=ref, source="cvm_dfp", url=url, pdf=pdf.name, g=g))

    # IPE fallback for firm-years without a time-deposit table
    def has_td(k):
        return any(any(r["instrument"] == "time_deposits" for r in x["g"]["rows"]) for x in results.get(k, []))
    need = [(fk, fy) for fk in firms for fy in range(y0, y1 + 1) if not has_td((fk, fy))]
    ipe_cache = {}
    for fk, fy in need:
        for fil_year in (fy, fy + 1):
            rows = ipe_cache.setdefault(fil_year, load_index("ipe", fil_year, a.refresh))
            cands = [r for r in rows if cnpj_root(r["CNPJ_Companhia"]) == CVM_FIRMS[fk][0]
                     and r["Data_Referencia"] == f"{fy}-12-31"
                     and r["Categoria"].startswith("Dados Econ")
                     and _fold(r["Tipo"]).startswith("demonstracoes financeiras")
                     and "rad.cvm.gov.br" in r["Link_Download"]]
            cands.sort(key=lambda r: (r["Data_Entrega"], r["Versao"]), reverse=True)
            seen = set()
            for r in cands[:4]:
                key = re.search(r"numSequencia=(\d+)", r["Link_Download"])
                if key and key.group(1) in seen:
                    continue
                seen.add(key.group(1) if key else r["Protocolo_Entrega"])
                pdf, err = download_ipe_pdf(fk, r, a.refresh)
                if pdf is None:
                    cvm_log.warning(f"[{fk}] IPE {fy}: {err}")
                    continue
                res = parsed_for(pdf, fk, a.refresh)
                for g in res["tables"]:
                    fy_g = int(g["asof"][:4]) if g["asof"] else fy
                    g["asof"] = g["asof"] or f"{fy}-12-31"
                    results.setdefault((fk, fy_g), []).append(
                        dict(doc_year=fil_year, source="cvm_ipe", url=r["Link_Download"], pdf=pdf.name, g=g))
            if has_td((fk, fy)):
                break

    # selection: one table per (firm, fy) over all documents, ranked by table_score():
    # a time-deposit row, plausible, balances before undiscounted flows, the firm's statement
    # priority, a passing sum check, the fiscal year's own filing over a later comparative,
    # the number of buckets.
    def has_td_row(x):
        return any(r["instrument"] == "time_deposits" for r in x["g"]["rows"])

    def td_row(x):
        rs = [r for r in x["g"]["rows"] if r["instrument"] == "time_deposits"]
        return rs[0] if rs else None

    def neighbour_ref(fk, fy):
        """Median time-deposit level of the balance tables of years fy-1 and fy+1: (raw, in
        R$ thousand by the stated unit), or None. Both readings are kept because a stated unit
        can be wrong (corrected only later)."""
        raw, thou = [], []
        for y in (fy - 1, fy + 1):
            for x in results.get((fk, y), []):
                r = td_row(x)
                if r is None or x["g"].get("undisc") or row_value(r) <= 0:
                    continue
                raw.append(row_value(r))
                thou.append(row_value(r) * _UNIT_SCALE.get(x["g"]["unit"] or "", 1.0))
        if not raw:
            return None
        return sorted(raw)[len(raw) // 2], sorted(thou)[len(thou) // 2]

    def size_flag(fk, fy, x):
        """Guard against small unrelated tables (Daycoval 2020 p.60, total 131): fewer than
        three nonzero buckets, or a level below 1% of the neighbouring years' median under
        both the raw and the unit-scaled reading."""
        r = td_row(x)
        if r is None:
            return ""
        nz = sum(1 for b in r["buckets"] if b["value"])
        if nz < 3:
            return f"implausible: only {nz} nonzero time-deposit buckets"
        ref = neighbour_ref(fk, fy)
        v = row_value(r)
        if ref and v > 0:
            vs = v * _UNIT_SCALE.get(x["g"]["unit"] or "", 1.0)
            if v < 0.01 * ref[0] and vs < 0.01 * ref[1]:
                return (f"implausible: time deposits {v:,.0f} are below 1% of the neighbouring years' "
                        f"median {ref[0]:,.0f}")
        return ""

    chosen, log_rows = {}, []   # (firm, fy) -> (best, ranked candidates with a time-deposit row)
    for fk in firms:
        for fy in range(y0, y1 + 1):
            cand = results.get((fk, fy), [])
            for x in cand:
                x["flag"] = (plausibility(x, [y for y in cand if y["pdf"] == x["pdf"]])
                             or size_flag(fk, fy, x))
            tds = [x for x in cand if has_td_row(x)]
            if not tds:
                dfp_ok = [status.get((fk, ry), (False, "")) for ry in (fy, fy + 1)]
                msg = "; ".join(f"DFP{ry}: {m}" for ry, (_, m) in zip((fy, fy + 1), dfp_ok))
                log_rows.append(dict(firm=fk, year=fy, document_found=any(f for f, _ in dfp_ok),
                                     page_found=bool(cand), method="",
                                     status="no time-deposit maturity table: " + msg,
                                     table_id="", sum_check="", flagged=""))
                continue
            tds.sort(key=lambda x: table_score(fk, fy, x))
            chosen[(fk, fy)] = (tds[0], tds)

    # unit cross-check, before the QA comparison: a page without an explicit unit inherits the
    # last one stated in the document, which can be wrong (a "R$ milhões" page earlier in the
    # filing). Each firm's reference level is the median time-deposit total of its chosen
    # tables stated in R$ thousand (three or more years); a table whose stated unit puts it more
    # than 30x away from that level while R$ thousand lands within 30x is read in R$ thousand.
    # Applied to the chosen table (written to notes and the log) and to every QA candidate.
    def td_level(x):
        rs = [r for r in x["g"]["rows"] if r["instrument"] == "time_deposits"]
        return row_value(rs[0]) if rs else None

    med = {}
    for fk in firms:
        thou = sorted(td_level(b) for (f, _), (b, _) in chosen.items()
                      if f == fk and b["g"]["unit"] == "R$ thousand" and td_level(b) and td_level(b) > 0)
        if len(thou) >= 3:
            med[fk] = thou[len(thou) // 2]

    def unit_fix(fk, x):
        """Corrected unit for candidate x, or None when its stated unit stands."""
        u, v = x["g"]["unit"], td_level(x)
        if fk not in med or u == "R$ thousand" or v is None or v <= 0:
            return None
        if (abs(math.log(v * _UNIT_SCALE.get(u or "", 1.0) / med[fk])) > math.log(30)
                and abs(math.log(v / med[fk])) <= math.log(30)):
            return "R$ thousand"
        return None

    def eff_unit(fk, x):
        return unit_fix(fk, x) or x["g"]["unit"]

    out_rows, qa_rows = [], []
    for fk in firms:
        for fy in range(y0, y1 + 1):
            if (fk, fy) not in chosen:
                continue
            best, tds = chosen[(fk, fy)]
            g = best["g"]
            stmt = g["stmt"]
            note_flag = best["flag"]
            tid = table_id(best)
            fix = unit_fix(fk, best)
            unit_note = (f"; UNIT CORRECTED {g['unit']} -> {fix} (stated unit inconsistent with the firm's series)"
                         if fix else "")
            pref = STMT_ORDER.get(fk, STMT_ORDER_DEFAULT)[0]
            stmt_note = (f"; statement {stmt} used: no {pref} table with a time-deposit row"
                         if stmt != pref and pref != "consolidated" else "")
            for r in g["rows"]:
                chk, tot = check_row(r)
                for b in r["buckets"]:
                    out_rows.append(dict(
                        firm_key=fk, cnpj_root=CVM_FIRMS[fk][0], fiscal_year=fy, as_of=f"{fy}-12-31",
                        source=best["source"], doc_url=best["url"], page=g["page"], statement=stmt,
                        table_id=tid,
                        table_title=g["ctx"].strip(), instrument=r["instrument"], bucket_label=b["label"],
                        lo_days=b["lo"], hi_days=b["hi"], value=b["value"], unit=(fix or g["unit"] or "unknown"),
                        extraction_method=g["method"], total_reported=tot,
                        sum_check="" if chk is None else round(chk, 6),
                        notes=f"row={r['label']}; filing_year={best['doc_year']}; pdf={best['pdf']}"
                              + ("; as_of inferred from filing year" if g.get("_asof_inferred") else "")
                              + (f"; FLAG {note_flag}" if note_flag else "") + stmt_note + unit_note))
            td = [r for r in g["rows"] if r["instrument"] == "time_deposits"][0]
            gap = check_row(td)[0]
            gap_s = "no_total" if gap is None else f"{gap:.6f}"
            log_rows.append(dict(firm=fk, year=fy, document_found=True, page_found=True,
                                 method=g["method"],
                                 status=(f"{'FLAGGED' if note_flag else 'ok'}; {best['source']} filing "
                                         f"{best['doc_year']} p.{g['page']} {stmt}; time-deposit sum_check={gap_s}"
                                         + (f"; {note_flag}" if note_flag else "") + stmt_note
                                         + (f"; UNIT CORRECTED {g['unit']} -> {fix}" if fix else "")),
                                 table_id=tid, sum_check="" if gap is None else round(gap, 6),
                                 flagged=bool(note_flag)))
            # QA across filings: same statement, balance tables only (undiscounted cash flows
            # add future interest), one table per document, in R$ thousand after the unit fix
            by_doc = {}
            for x in tds:
                if x["g"]["stmt"] != stmt or x["g"].get("undisc"):
                    continue
                k = (x["source"], x["doc_year"], x["pdf"])
                if k not in by_doc:
                    by_doc[k] = x

            def key_vals(x):
                rs = [r for r in x["g"]["rows"] if r["instrument"] == "time_deposits"]
                sc_ = _UNIT_SCALE.get(eff_unit(fk, x) or "", 1.0)   # compare in R$ thousand
                return {(b["lo"], b["hi"]): b["value"] * sc_ for b in rs[0]["buckets"]} if rs else {}
            vals = [key_vals(x) for x in by_doc.values() if key_vals(x)]
            mx = 0.0
            if len(vals) > 1:
                for kk in set().union(*[v.keys() for v in vals]):
                    xs = [v[kk] for v in vals if kk in v]
                    if len(xs) > 1 and max(abs(q) for q in xs) > 0:
                        mx = max(mx, (max(xs) - min(xs)) / max(abs(q) for q in xs))
            qa_rows.append(dict(firm=fk, fiscal_year=fy, n_sources=len(vals),
                                max_rel_diff=round(mx, 6) if len(vals) > 1 else ""))
    log_rows.sort(key=lambda r: (firms.index(r["firm"]), r["year"]))

    cols = ["firm_key", "cnpj_root", "fiscal_year", "as_of", "source", "doc_url", "page", "statement",
            "table_id", "table_title", "instrument", "bucket_label", "lo_days", "hi_days", "value", "unit",
            "extraction_method", "total_reported", "sum_check", "notes"]
    for path, rows, cs in ((OUT_CSV, out_rows, cols),
                           (QA_CSV, qa_rows, ["firm", "fiscal_year", "n_sources", "max_rel_diff"]),
                           (LOG_CSV, log_rows, ["firm", "year", "document_found", "page_found", "method", "status",
                                                    "table_id", "sum_check", "flagged"])):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cs)
            w.writeheader()
            w.writerows(rows)
        cvm_log.info(f"wrote {path} ({len(rows)} rows)")


# ================================================================================ summary
###─────────────────────────────────────────────────────────────────────────────
# summary: one bank-year summary of the two extractions
#
#   from the two statement extractions (the sec and cvm subcommands).
#
#   For each bank and fiscal year: the total, the share with no stated maturity, the shares maturing
#   within 90 and within 360 days, and the share beyond 360 days.
#   - Buckets that straddle a cut-off are split linearly in days.
#   - Only series whose buckets add up to the printed total within 0.5 percent are used.
#   - When a year has several tables with a time-deposit line, the largest is kept (the all-client
#     table rather than a sub-portfolio), and `n_tables` records the ambiguity. The CVM extraction
#     already selects one table per bank-year, so for it `n_tables` is always 1.
#   - Banco do Brasil's "Sem vencimento" bucket is judicial deposits, not time deposits with daily
#     liquidity: the note's footnote keeps CDBs with a liquidity clause in their original-maturity
#     buckets, and the bucket's size matches the "Judiciais" line of the deposits note (user decision
#     of 2026-10-14). It leaves the total; its amount is kept in `excluded_judicial`.
#
#   Reads  raw/DEPOSIT_MATURITY/deposit_maturity_sec.csv and deposit_maturity_cvm.csv.
#   Writes raw/DEPOSIT_MATURITY/deposit_maturity_summary.csv, the input of the deposit-type notes'
#   Section 4 and of `switch_cost_tools.py locked` (the upper bound of the locked CDB share).
#
#   usage: python scrape_deposit_maturity.py summary [FOLDER] [--out FILE]
###─────────────────────────────────────────────────────────────────────────────
DEFAULT_DIR = paths.RAW / "DEPOSIT_MATURITY"
# Banks whose no-maturity bucket is judicial deposits, dropped from the time-deposit total.
JUDICIAL_NO_MATURITY = {"bb"}


def load(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sec = pd.read_csv(folder / "deposit_maturity_sec.csv")
    cvm = pd.read_csv(folder / "deposit_maturity_cvm.csv")
    sec = sec[(sec["instrument"] == "time_deposits") & (sec["table_kind"] == "all_time_deposits")].copy()
    cvm = cvm[cvm["instrument"] == "time_deposits"].copy()
    for d in (sec, cvm):
        sc = d["sum_check"].astype(str).str.strip().str.lower()
        num = pd.to_numeric(d["sum_check"], errors="coerce")
        for c in ("lo_days", "hi_days", "value"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        # The relative gap between the buckets and the printed total (both sources); "ok" is
        # also accepted.
        d["ok"] = (num <= 0.005) | (sc == "ok")
        d["table_id"] = d[[c for c in ("doc_url", "page", "table_title") if c in d.columns]].astype(str).agg("|".join, axis=1)
    return sec, cvm


def shares(g: pd.DataFrame, drop_no_maturity: bool = False) -> pd.Series:
    nomat_rows = (g["lo_days"].fillna(-1) == 0) & (g["hi_days"].fillna(-1) == 0)
    excluded = 0.0
    if drop_no_maturity:
        excluded = float(g.loc[nomat_rows, "value"].sum())
        g, nomat_rows = g[~nomat_rows], nomat_rows[~nomat_rows]
    tot = g["value"].sum()
    out = {"total": tot, "n_buckets": len(g), "excluded_judicial": excluded}
    nomat = g.loc[nomat_rows, "value"].sum()
    out["share_no_maturity"] = nomat / tot if tot else np.nan
    rest = g[~nomat_rows]
    for cut in (90, 360):
        v = 0.0
        for _, r in rest.iterrows():
            lo = 0 if pd.isna(r["lo_days"]) else r["lo_days"]
            hi = np.inf if pd.isna(r["hi_days"]) else r["hi_days"]
            if hi <= cut:
                v += r["value"]
            elif lo < cut < hi and np.isfinite(hi):
                v += r["value"] * (cut - lo) / (hi - lo)
        out[f"share_le{cut}"] = v / tot if tot else np.nan
    out["share_gt360"] = 1 - out["share_no_maturity"] - out["share_le360"]
    return pd.Series(out)


def summarise(sec: pd.DataFrame, cvm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, d in (("sec_20f", sec), ("cvm", cvm)):
        d = d[d["ok"]]
        for (f, y), gg in d.groupby(["firm_key", "fiscal_year"]):
            tables = gg.groupby("table_id")["value"].sum()
            g = gg[gg["table_id"] == tables.idxmax()]
            s = shares(g, drop_no_maturity=f in JUDICIAL_NO_MATURITY)
            s["n_tables"] = len(tables)
            s["firm_key"], s["fiscal_year"], s["source"], s["unit"] = f, y, src, g["unit"].iloc[0]
            rows.append(s)
    out = pd.DataFrame(rows)[["firm_key", "fiscal_year", "source", "unit", "total", "excluded_judicial", "n_buckets", "n_tables",
                              "share_no_maturity", "share_le90", "share_le360", "share_gt360"]]
    return out.sort_values(["firm_key", "fiscal_year", "source"])


def summary_main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Bank-year summary of time deposits by remaining maturity")
    p.add_argument("folder", nargs="?", default=str(DEFAULT_DIR))
    p.add_argument("--out", default=None, help="write here instead of FOLDER/deposit_maturity_summary.csv")
    a = p.parse_args()
    folder = Path(a.folder)
    out = summarise(*load(folder))
    dest = Path(a.out) if a.out else folder / "deposit_maturity_summary.csv"
    out.to_csv(dest, index=False, encoding="utf-8")
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    print(f"-> {dest}")


# ==================================================================================== CLI
SUBCOMMANDS = {"sec": sec_main, "cvm": cvm_main, "summary": summary_main}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        print("usage: python scrape_deposit_maturity.py {sec,cvm,summary} [options]  (-h after a "
              "subcommand for its options)")
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    cmd = sys.argv[1]
    # each subcommand parses its own options, as the three scripts did
    sys.argv = [f"scrape_deposit_maturity.py {cmd}"] + sys.argv[2:]
    SUBCOMMANDS[cmd]()


if __name__ == "__main__":
    main()
