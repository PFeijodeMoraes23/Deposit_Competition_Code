###─────────────────────────────────────────────────────────────────────────────
# scrape_deposit_maturity_sec.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Time deposits (and, where the same table prints them, demand, savings and total
#   deposits) by remaining maturity bucket, read from the SEC Form 20-F filings of
#   the Brazilian banks and the SEC-listed digital platforms, as of each fiscal year
#   end 2016-2024. The maturity profile is the observable counterpart of how much of
#   the deposit base can leave within a quarter (deposit_type_heterogeneity_notes.md).
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
#   python scrape_deposit_maturity_sec.py                       # all firms
#   python scrape_deposit_maturity_sec.py --firms itau,bradesco # subset
#   python scrape_deposit_maturity_sec.py --refresh             # re-download filings
###─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import logging
import re

try:
    from utils.venv_guard import ensure_project_venv
except ImportError:  # pragma: no cover
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd
from lxml import html as lhtml

from utils import paths
from utils.disclosure_common import get_json, http_get
from utils.firm_registry import FIRMS

log = logging.getLogger("deposit_maturity_sec")
OUT = paths.RAW / "DEPOSIT_MATURITY"
HTML_DIR = OUT / "sec"
FIRM_KEYS = ["itau", "bradesco", "santander_br", "nubank", "inter", "xp"]
FY_MIN, FY_MAX = 2016, 2024
FILING_YEARS = (2016, 2025)   # report-date years whose 20-F is downloaded
SUM_TOL = 0.005             # buckets must add up to the printed total within 0.5%
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
        log.warning("submissions unavailable for %s (HTTP %s)", cik, code)
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
        log.warning("download failed %s (HTTP %s)", fl["url"], code)
        return None
    fn.write_text(body, encoding="utf-8")
    return body


# ───────────────────────── numbers, dates, buckets ─────────────────────────
def parse_num(s: str):
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


def parse_bucket(label: str):
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


def instrument_of(label: str, deposits_table: bool = False):
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
    title_row = next((" ".join(r) for r in rows if not any(parse_num(x) is not None for x in r)
                      and not find_dates(" ".join(r)) and len(" ".join(r)) > 8
                      and not any(parse_bucket(x) for x in r)), "")
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
        bcells = [parse_bucket(x) for x in r]
        numeric = [parse_num(x) is not None and not YEAR_CELL.fullmatch(x) for x in r[1:]]
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
        vals = [parse_num(x) for x in r[1:]]
        ok = bool(vals) and all(v is not None for v in vals)
        instr = instrument_of(r[0], deposits_table)
        if header and instr and ok and not any(v < 0 for v in vals):
            recs += emit_header_row(r[0], instr + section, vals, header, pending, ctx, ctx_title, unit, ti, fy_filing, meta)
        elif ok and parse_bucket(r[0]) is not None:
            brows.append((r, vals))
        elif ok and brows and r[0].lower().startswith("total"):
            brows.append((r, vals))
    if not recs and sum(parse_bucket(r[0]) is not None for r, _ in brows) >= 3:
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
    buckets = [(r, v) for r, v in brows if parse_bucket(r[0]) is not None]
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
            lo, hi = parse_bucket(r[0])
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
    bad = df.groupby(ser + ["_filing"])["sum_check"].transform("max").fillna(0) > SUM_TOL
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


def main() -> None:
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
            log.warning("firm %s not in registry", fk)
            continue
        for fl in sorted(list_filings(cik), key=lambda d: d["filed"]):
            doc = fetch_filing(cik, fl, args.refresh)
            if not doc or len(doc) < MIN_DOC_CHARS:
                log.info("%s %s %s: skipped (%d chars)", fk, fl["form"], fl["report_date"], len(doc or ""))
                continue
            meta = dict(firm_key=fk, cik=cik, url=fl["url"], accession=fl["accession"], filed=fl["filed"])
            recs = extract_tables(doc, int(fl["report_date"][:4]), meta)
            log.info("%s %s FY%s: %d rows", fk, fl["form"], fl["report_date"][:4], len(recs))
            allrecs += recs
    if not allrecs:
        log.warning("no rows extracted")
        return
    df, qa = assemble(allrecs)
    df.to_csv(OUT / "deposit_maturity_sec.csv", index=False, encoding="utf-8", lineterminator="\n")
    qa.to_csv(OUT / "deposit_maturity_sec_qa.csv", index=False, encoding="utf-8", lineterminator="\n")
    log.info("wrote %d rows and %d QA rows to %s", len(df), len(qa), OUT)


if __name__ == "__main__":
    main()
