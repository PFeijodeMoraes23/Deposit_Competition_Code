"""
scrape_caixa_advertising.py
===========================
CAIXA ECONOMICA FEDERAL's own published advertising outlays, monthly from January 2013, and its
published sponsorship contracts (a separate robustness series), 2019 onward.

Caixa files with no securities regulator, so its transparency page is the only pre-2025
advertising series for it. The documents are titled "Custos CAIXA <mes>" or "Despesas com
Publicidade CAIXA - <mes> (R$)". No footnote says whether the amounts are billed, committed or
paid, and no agency-commission line is shown; April 2019 prints "Nao existem pagamentos
registrados no periodo". Every amount is therefore tagged basis = "custos_undocumented".

Sources (www.caixa.gov.br, SharePoint list API; plain clients loop on a 302 without a cookie jar)
------------------------------------------------------------------------------------------------
  _api/web/lists/GetByTitle('despesas-publicidade')/Items   one item per year, links in ColExtra2
      "<Mes>"             the PDF (2019: a ZIP holding the summary PDF and a supplier PDF)
      "<Mes> - Editavel"  from April 2020, a ZIP holding an XLSX (some months .xls, from November
                          2025 .xltx templates, May 2025 a PDF, January 2022 a PDF and an XLSX)
      "Todos os meses"    2010-2016 year ZIPs; every member is byte-identical to the monthly PDF
                          (checked 2026-09-14), so they are not downloaded
  _api/web/lists/GetByTitle('patrocinio')/Items             one item per year 2019-2026, a ZIP with
                          "Contratados" (general programme) and, from 2023, "TransparenciaPOCC"
                          (Programa de Ocupacao dos espacos da CAIXA Cultural; the 2023 one is an
                          HTML table saved as .xls)
Downloads are cached under paths.AWARENESS_RAW / "caixa"; --refresh downloads again.

Document layouts
----------------
  2013-01 .. 2014-06, 2016-04, 2016-05  summary page is an image scan (no text layer; only the
                          supplier list pages carry text). No OCR engine on this machine:
                          status unreadable_scan, never estimated.
  2014-07 .. 2015-04, 2015-07 .. 2018-12  unruled layout: category labels on the left, one column
                          per agency. Parsed from word positions: each amount is attached to the
                          nearest label line (flag ambiguous_rows when that line is more than
                          4.5pt away), labels wrapped onto a value-less line are joined.
  2015-05, 2015-06, 2019-01 .. 2026-06  ruled table. Parsed from the table cells; merged cells
                          carry their label down; vertical (rotated) labels are read bottom-to-top
                          from the characters; a table split across pages continues.
  Spreadsheets            the first sheet with an agency header row; merged ranges carry labels.
Only the summary table is read (section > category > subcategory x agency). The supplier lists
(names and CNPJs, no amounts) are not.

Rules
-----
  - Amounts are reais as published. Blank, "-" and 0,00 cells are not lines (the 2014-15 layout
    prints 0,00 in every empty cell, later layouts leave them blank).
  - Labels are kept as published, with whitespace collapsed. Nothing is reclassified: events,
    fees (cachês), research, copyright and similar rows stay in, each under its own label.
  - One source per month (parse_method): the spreadsheet where one parses (xlsx, or template for
    .xltx), else the PDF found in the spreadsheet slot (pdf_in_xlsx_slot), else the PDF
    (text_pdf). Where both a spreadsheet and a PDF parse, they are compared.
  - stated_total_brl is the grand total ("TOTAL GERAL") printed in the chosen document; when that
    document has none (a formula without a stored value, or no row), the other document's is used
    and stated_total_source says which. Per-agency "TOTAL" rows are checked as a flag.

Outputs (paths.AWARENESS_PROC)
------------------------------
  caixa_advertising_lines.{parquet,csv}     one row per published nonzero cell
  caixa_advertising_monthly.{parquet,csv}   one row per month 2013-01 .. latest published month
  caixa_sponsorship_contracts.{parquet,csv} one row per contract
All carry bank_key = "caixa", cnpj8 = "00360305" and panel_key, the prudential conglomerate code
established from the IF.data lists (CnpjInstituicaoLider 00360305) and confirmed against the
conglomerate rows of cosif_advertising_monthly.parquet.

Monthly flags
-------------
  agency_totals_match      lines per agency sum to the printed per-agency TOTAL row
  pdf_xlsx_agree           spreadsheet and PDF agree on every agency sum (NaN if not both parsed)
  pdf_xlsx_lines_match     share of spreadsheet (agency, amount) lines also found in the PDF
  repeated_agency_block    agencies whose whole block (labels and amounts) equals the previous month
  title_month_mismatch     the document title names another month or year
  ambiguous_rows           unruled layout: amounts attached to a label line more than 4.5pt away
  no_payments_stated       the document says no payments were registered in the period
  pdf_slot_broken          the PDF link returns a web page instead of a PDF

Validation (a hard failure aborts before writing)
-------------------------------------------------
  1. Every month from 2013-01 to the latest published month has a status.
  2. Where a stated grand total exists, lines sum to it within max(R$1, 0.1%); at least 95% of
     months with a stated total must match.
  3. Spreadsheet and PDF agree on every agency sum wherever both parse.
  4. No negative amount.
  5. April, May and June 2020 reproduce R$ 2.4m, 17.2m and 35.6m.
  6. The panel key is unique and confirmed by COSIF.
  7. Sponsorship: every contract has a positive amount and a date.
Printed, not gated: 2025 ratios of Caixa's quarterly, half-year and annual sums to the COSIF
conglomerate flows adv and all3 (cosif_advertising_quarterly.parquet, entity C0080738).

Usage
-----
  python scrape_caixa_advertising.py
  python scrape_caixa_advertising.py --refresh           # download the lists and files again
  python scrape_caixa_advertising.py --out-dir <dir>     # write somewhere else (smoke runs)
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import io
import json
import logging
import re
import time
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pdfplumber
import requests
import xlrd
from bs4 import BeautifulSoup

from utils import paths
from utils.disclosure_common import http_get

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

RAW_DIR = paths.AWARENESS_RAW / "caixa"
OUT_DIR = paths.AWARENESS_PROC
BASE = "https://www.caixa.gov.br"
LIST_API = BASE + "/_api/web/lists/GetByTitle('{title}')/Items/?$top=5000"
AD_LIST, SPONSOR_LIST = "despesas-publicidade", "patrocinio"

BANK_KEY, CNPJ8 = "caixa", "00360305"
BASIS_AD, BASIS_SPONSOR = "custos_undocumented", "contracted"
FIRST_MONTH = "201301"

AGENCIES = {"ARTPLAN", "BORGHI", "HEADS", "NOVA/SB", "NOVASB", "PROPEG", "PROPER", "BINDER",
            "CALIA"}
AGENCY_CANON = {"NOVASB": "NOVA/SB", "PROPER": "PROPEG"}   # October 2020 header misprint
MONTHS_PT = {"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
             "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11,
             "dezembro": 12}
MONEY_RE = re.compile(r"^-?\d{1,3}(?:\.\d{3})*,\d{1,4}$|^-?\d+,\d{1,4}$")
TOL_ABS, TOL_REL = 1.0, 0.001
MIN_MATCH_SHARE = 0.95
AMBIGUOUS_PT = 4.5
KNOWN_TOTALS_M = {"202004": 2.4, "202005": 17.2, "202006": 35.6}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def clean(s) -> str:
    if s is None:
        return ""
    return re.sub(r"[\s​\xa0]+", " ", str(s)).strip()


def agency_name(s) -> str | None:
    t = re.sub(r"\s+", "", clean(s)).upper()
    return AGENCY_CANON.get(t, t) if t in AGENCIES else None


def parse_brl(s) -> float | None:
    """Brazilian-format amount -> float; None for a blank or dash cell; ValueError otherwise."""
    if s is None:
        return None
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return None if pd.isna(s) else float(s)
    t = re.sub(r"[\s​\xa0]+", "", str(s)).replace("R$", "")
    if t in ("", "-", "–", "—"):
        return None
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    if re.fullmatch(r"-?[\d.]*,\d{1,4}", t) or re.fullmatch(r"-?\d{1,3}(\.\d{3})+", t):
        v = float(t.replace(".", "").replace(",", "."))
    elif re.fullmatch(r"-?\d+(\.\d+)?", t):
        v = float(t)
    else:
        raise ValueError(f"not an amount: {s!r}")
    return -v if neg else v


def title_period(text: str) -> str | None:
    """YYYYMM named in a document title such as 'Custos CAIXA MAIO/2015'."""
    f = fold(text)
    m = re.search(r"(janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|"
                  r"novembro|novemro|dezembro)\s*(?:de\s*)?[/-]?\s*(\d{4})", f)
    if not m:
        return None
    month = MONTHS_PT.get(m.group(1), 11 if m.group(1) == "novemro" else None)
    return f"{m.group(2)}{month:02d}"


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------
class Fetcher:
    """utils.disclosure_common.http_get first. www.caixa.gov.br answers a client without cookies
    with a 302 loop (and file URLs with 429), so on failure a requests session takes over; it is
    primed on the list API, which sets the cookie."""

    def __init__(self) -> None:
        self.session: requests.Session | None = None

    def _session(self) -> requests.Session:
        if self.session is None:
            self.session = requests.Session()
            self.session.headers["User-Agent"] = "Mozilla/5.0 (research; deposit-competition)"
            self.session.get(LIST_API.format(title=AD_LIST), timeout=90,
                             headers={"Accept": "application/json;odata=verbose"})
        return self.session

    def get(self, url: str, accept: str = "*/*") -> bytes:
        if self.session is None:
            status, body = http_get(url, binary=True, retries=1, extra_headers={"Accept": accept})
            if status == 200 and body is not None:
                return body
            log.info("http_get gave status %s for %s; using a cookie session", status, url)
        s = self._session()
        for attempt in range(1, 6):
            try:
                r = s.get(url, headers={"Accept": accept}, timeout=120)
                if r.status_code == 200:
                    time.sleep(0.3)
                    return r.content
                if r.status_code in (403, 404, 410):
                    raise FileNotFoundError(f"HTTP {r.status_code} for {url}")
                wait = float(r.headers.get("Retry-After") or 5 * attempt)
                log.warning("HTTP %s on %s (attempt %d/5), waiting %.0fs", r.status_code, url,
                            attempt, wait)
            except requests.RequestException as e:
                wait = 3 * attempt
                log.warning("GET failed %s on %s (attempt %d/5)", type(e).__name__, url, attempt)
            time.sleep(wait)
        raise RuntimeError(f"could not download {url}")


def list_items(fetcher: Fetcher, title: str, refresh: bool) -> list[dict]:
    cache = RAW_DIR / f"list_{title}.json"
    if refresh or not cache.exists():
        body = fetcher.get(LIST_API.format(title=title), accept="application/json;odata=verbose")
        json.loads(body.decode("utf-8"))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(body)
    return json.loads(cache.read_text(encoding="utf-8"))["d"]["results"]


def link_url(href: str) -> str:
    href = html.unescape(href).strip()
    return BASE + href if href.startswith("/") else re.sub(r"^http://", "https://", href)


def item_links(item: dict) -> list[tuple[str, str]]:
    soup = BeautifulSoup(item.get("ColExtra2") or "", "lxml")
    return [(clean(a.get_text(" ")).strip("• "), link_url(a["href"]))
            for a in soup.find_all("a", href=True)]


def advertising_documents(items: list[dict]) -> list[dict]:
    docs = []
    for it in items:
        m = re.search(r"(\d{4})", it.get("Title") or "")
        if not m:
            raise ValueError(f"list item without a year: {it.get('Title')!r}")
        year = int(m.group(1))
        for label, url in item_links(it):
            f = fold(label)
            if "todos os meses" in f:
                continue
            month = next((n for k, n in MONTHS_PT.items() if f.startswith(k)), None)
            if month is None:
                raise ValueError(f"unrecognised link label {label!r} in {it['Title']}")
            period = f"{year}{month:02d}"
            if period < FIRST_MONTH:
                continue
            docs.append(dict(period=period, slot="editable" if "editavel" in f else "pdf",
                             label=label, url=url))
    dup = pd.DataFrame(docs).duplicated(["period", "slot"])
    if dup.any():
        raise ValueError(f"two links for the same month and slot: "
                         f"{pd.DataFrame(docs)[dup].to_dict('records')}")
    return docs


def sponsorship_documents(items: list[dict]) -> list[dict]:
    docs = []
    for it in items:
        m = re.search(r"(\d{4})", it.get("Title") or "")
        if not m or "patroc" not in fold(it.get("Title") or ""):
            raise ValueError(f"unexpected sponsorship list item {it.get('Title')!r}")
        for _, url in item_links(it):
            docs.append(dict(year=int(m.group(1)), url=url))
    return docs


def download(fetcher: Fetcher, url: str, dest: Path, refresh: bool) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = fetcher.get(url)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    log.info("  downloaded %s (%d bytes)", dest.name, len(body))
    return dest


def ad_path(doc: dict) -> Path:
    name = Path(requests.utils.urlparse(doc["url"]).path).name
    return RAW_DIR / "despesas" / doc["period"][:4] / f"{doc['period']}_{doc['slot']}__{name}"


def sponsor_path(doc: dict) -> Path:
    name = Path(requests.utils.urlparse(doc["url"]).path).name
    return RAW_DIR / "patrocinio" / f"{doc['year']}__{name}"


# ---------------------------------------------------------------------------
# Document containers
# ---------------------------------------------------------------------------
@dataclass
class Parsed:
    method: str                       # text_pdf | xlsx | template | pdf_in_xlsx_slot
    member: str                       # file name inside the downloaded file
    lines: list[dict] = field(default_factory=list)
    stated_total: float | None = None
    agency_totals: dict[str, float] = field(default_factory=dict)
    agencies: list[str] = field(default_factory=list)
    status: str = "parsed"            # parsed | unreadable_scan
    title_period: str | None = None
    ambiguous_rows: int = 0
    no_payments_stated: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return float(sum(x["amount_brl"] for x in self.lines))

    def agency_sums(self) -> dict[str, float]:
        out = {a: 0.0 for a in self.agencies}
        for x in self.lines:
            out[x["agency"]] = out.get(x["agency"], 0.0) + x["amount_brl"]
        return out


def unpack(path: Path) -> list[tuple[str, bytes]]:
    """(member name, bytes) for a downloaded file: ZIP members, or the file itself."""
    b = path.read_bytes()
    if b[:2] == b"PK" and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            return [(m, z.read(m)) for m in z.namelist() if not m.endswith("/")]
    return [(path.name, b)]


def kind_of(name: str, b: bytes) -> str:
    low = name.lower()
    if b[:4] == b"%PDF":
        return "pdf"
    if b[:2] == b"PK" and low.endswith((".xlsx", ".xltx", ".xlsm")):
        return "template" if low.endswith(".xltx") else "xlsx"
    if b[:8] == bytes.fromhex("d0cf11e0a1b11ae1"):
        return "xls"
    if b.lstrip()[:1] == b"<":
        return "html"
    return "unknown"


# ---------------------------------------------------------------------------
# Resolving label hierarchies (tables and spreadsheets)
# ---------------------------------------------------------------------------
TOTAL_GERAL_RE = re.compile(r"^total\s*geral", re.I)


def row_kind(labels: list[str]) -> str:
    texts = [fold(t) for t in labels if t]
    if any(TOTAL_GERAL_RE.match(t) for t in texts):
        return "grand_total"
    if any(t.strip(" :") == "total" or t.startswith("total por") for t in texts):
        return "agency_total"
    return "data"


def resolve_sections(rows: list[dict], n_section_cols: int) -> None:
    """Fill row['section'] from the section columns. A column whose vertical label was split
    into several stacked unmerged cells (e.g. January 2019) is joined bottom-to-top and each row
    takes the nearest such label; otherwise merged cells carry down."""
    data_idx = [i for i, r in enumerate(rows) if r["kind"] == "data"]
    parts = []
    for c in range(n_section_cols):
        cells = [rows[i]["sec_raw"][c] for i in data_idx]
        texts = [x["text"] if x else None for x in cells]
        stacked = any(texts[k] and texts[k + 1] for k in range(len(texts) - 1))
        values = [""] * len(data_idx)
        if stacked:
            runs, k = [], 0
            while k < len(texts):
                if texts[k]:
                    j = k
                    while j + 1 < len(texts) and texts[j + 1]:
                        j += 1
                    pieces = [cells[t] for t in range(k, j + 1)]
                    vertical = any(p["rotated"] for p in pieces) or all(
                        len(p["text"]) <= 3 for p in pieces)
                    word = ("".join(p["text"].replace(" ", "") for p in reversed(pieces))
                            if vertical else " ".join(p["text"] for p in pieces))
                    runs.append(((k + j) / 2, word))
                    k = j + 1
                else:
                    k += 1
            for pos in range(len(data_idx)):
                if runs:
                    values[pos] = min(runs, key=lambda r: (abs(r[0] - pos), r[0]))[1]
        else:
            current = ""
            for pos, cell in enumerate(cells):
                if cell is None:
                    values[pos] = current
                else:
                    current = cell["text"]
                    values[pos] = current
        parts.append(values)
    for pos, i in enumerate(data_idx):
        rows[i]["section"] = " > ".join(p[pos] for p in parts if p[pos])


def lines_from_rows(rows: list[dict], agencies: list[str], parsed: Parsed) -> None:
    """rows: dicts with kind, section, group, item, values {agency: raw}, where."""
    for r in rows:
        if r["kind"] == "grand_total":
            vals = [parse_brl(v) for v in r["values"].values()]
            vals = [v for v in vals if v is not None]
            if not vals:
                m = re.search(r"R?\$?\s*([\d.]+,\d{2})", " ".join(r["label_texts"]))
                vals = [parse_brl(m.group(1))] if m else []
            if len(vals) > 1:
                parsed.stated_total = float(sum(vals))
                parsed.notes.append("grand total row printed per agency; stated total is its sum")
            elif vals:
                parsed.stated_total = vals[0]
            elif any(r.get("uncached")):
                parsed.notes.append("grand total is a formula without a stored value")
            continue
        if r["kind"] == "agency_total":
            for a, v in r["values"].items():
                v = parse_brl(v)
                if v is not None:
                    parsed.agency_totals[a] = v
            continue
        for a, raw in r["values"].items():
            if isinstance(raw, str) and not re.search(r"\d", raw):
                if raw.strip(" -–R$"):
                    parsed.notes.append(f"text in an amount cell: {clean(raw)!r}")
                continue
            v = parse_brl(raw)
            if v is None or v == 0:
                continue
            parsed.lines.append(dict(agency=a, section=r["section"], category=r["group"],
                                     subcategory=r["item"], amount_brl=v,
                                     page_or_sheet=r["where"]))


# ---------------------------------------------------------------------------
# PDF: ruled tables
# ---------------------------------------------------------------------------
def rotated_text(page, bbox) -> tuple[str, bool]:
    """Text of a table cell written vertically, read bottom-to-top with spaces at wide gaps.
    Covers rotated glyphs and upright letters stacked one per line. ('', False) otherwise."""
    x0, top, x1, bottom = bbox
    chars = [c for c in page.chars if c["text"].strip()
             and x0 <= (c["x0"] + c["x1"]) / 2 <= x1 and top <= (c["top"] + c["bottom"]) / 2 <= bottom]
    if len(chars) < 2:
        return "", False
    rot = [c for c in chars if not c.get("upright", True)]
    if len(rot) >= 0.5 * len(chars):
        seq = rot
    else:
        width = max(c["x1"] - c["x0"] for c in chars)
        tops = sorted(round(c["top"]) for c in chars)
        stacked = (len(chars) >= 3 and max(c["x0"] for c in chars) - min(c["x0"] for c in chars)
                   < 1.2 * width and len(set(tops)) >= len(chars) - 1)
        if not stacked:
            return "", False
        seq = chars
    seq = sorted(seq, key=lambda c: -(c["top"] + c["bottom"]) / 2)
    gaps = [prev["top"] - cur["bottom"] for prev, cur in zip(seq, seq[1:])]
    base = float(np.median(gaps)) if gaps else 0.0
    out = seq[0]["text"]
    for gap, cur in zip(gaps, seq[1:]):
        out += (" " if gap > base + 0.35 * max(cur.get("size", 8), 1) else "") + cur["text"]
    return clean(out), True


def page_is_supplier_list(text: str) -> bool:
    head = fold(text[:200])
    return "fornecedores" in head and "cnpj" in head


def _is_amount_text(c: str | None) -> bool:
    return bool(c) and bool(MONEY_RE.match(re.sub(r"[\s​R$]+", "", c) or "x"))


def parse_pdf_ruled(pdf, parsed: Parsed) -> bool:
    """True when the document has a ruled summary table with an agency header.

    Geometry, not column indices, links the pieces: amount cells go to the agency whose header
    cell is horizontally nearest (some 2019 tables put amounts in columns offset from the header
    cells), and label cells go to the label column of the header table whose left edge is
    nearest (a continuation table on a later page can have a different column grid)."""
    raw_rows: list[dict] = []
    centers: list[tuple[float, str]] | None = None
    canon_lefts: list[float] = []
    carry: list[str] = []
    done = False
    for pno, page in enumerate(pdf.pages):
        if done:
            break
        text = page.extract_text() or ""
        if page_is_supplier_list(text):
            if centers:
                break
            continue
        for table in page.find_tables():
            grid = table.extract()
            if not grid:
                continue
            ncols = max(len(r) for r in grid)
            lefts = []
            for j in range(ncols):
                xs = [row.cells[j][0] for row in table.rows
                      if j < len(row.cells) and row.cells[j] is not None]
                lefts.append(min(xs) if xs else float("inf"))
            header = next((i for i, row in enumerate(grid)
                           if sum(agency_name(c) is not None for c in row) >= 2), None)
            continuation = bool(centers)
            if header is not None:
                row, boxes = grid[header], table.rows[header].cells
                heads = [j for j, c in enumerate(row) if agency_name(c)]
                centers = [((boxes[j][0] + boxes[j][2]) / 2, agency_name(row[j])) for j in heads]
                agency_left = min(boxes[j][0] for j in heads)
                start = header + 1
                if parsed.title_period is None:
                    parsed.title_period = title_period(text) or title_period(clean(row[0]))
            elif centers and any(_is_amount_text(c) or clean(c) for r in grid for c in r):
                start = 0
            else:
                continue
            amount_cols = [j for i in range(start, len(grid)) for j, c in enumerate(grid[i])
                           if _is_amount_text(c)]
            n_label = min([j for j in range(ncols) if lefts[j] >= agency_left - 2] + amount_cols
                          + [ncols])
            if not canon_lefts:
                canon_lefts = lefts[:n_label]
                carry = [""] * n_label
            slot_of = {j: min(range(len(canon_lefts)), key=lambda k: abs(canon_lefts[k] - lefts[j]))
                       for j in range(n_label)}
            for i in range(start, len(grid)):
                raw, boxes = grid[i], table.rows[i].cells
                cells: list[dict | None] = [None] * len(canon_lefts)
                covered = [False] * len(canon_lefts)
                for j in range(n_label):
                    if raw[j] is None:
                        # a merged cell from the left spans this column: blank, not carried
                        if any(boxes[k] is not None and boxes[k][2] > lefts[j] + 1
                               for k in range(j)):
                            covered[slot_of[j]] = True
                        continue
                    t, rot = rotated_text(page, boxes[j]) if boxes[j] else ("", False)
                    t = t if rot else clean(raw[j])
                    k = slot_of[j]
                    if cells[k] is not None and cells[k]["text"]:
                        t = f"{cells[k]['text']} {t}".strip()
                    cells[k] = {"text": t, "rotated": rot}
                labels = []
                for k, c in enumerate(cells):
                    if c is None:
                        labels.append("" if covered[k] else carry[k])
                    elif c["text"] == "" and continuation and i == start:
                        labels.append(carry[k])
                        cells[k] = None
                    else:
                        labels.append(c["text"])
                carry = labels
                values: dict[str, str] = {}
                for j in range(n_label, len(raw)):
                    if raw[j] is None or not clean(raw[j]) or not boxes[j]:
                        continue
                    xc = (boxes[j][0] + boxes[j][2]) / 2
                    agency = min(centers, key=lambda c: abs(c[0] - xc))[1]
                    if agency in values:
                        raise ValueError(f"p{pno + 1}: two cells for {agency} in row {raw}")
                    values[agency] = raw[j]
                own = [c["text"] for c in cells if c and c["text"]]
                raw_rows.append(dict(kind=row_kind(own), own_label=bool(own), cells=cells,
                                     labels=labels, values=values, where=f"p{pno + 1}",
                                     label_texts=own, uncached=[]))
                if raw_rows[-1]["kind"] == "grand_total":
                    done = True
                    break
            if done:
                break
    if not centers:
        return False
    parsed.agencies = [a for _, a in sorted(centers)]

    # Rows without any label of their own that repeat the running sums are total rows printed
    # without a label (February 2019).
    running = {a: 0.0 for a in parsed.agencies}
    for r in raw_rows:
        vals = {a: parse_brl(v) for a, v in r["values"].items() if re.search(r"\d", v)}
        vals = {a: v for a, v in vals.items() if v}
        if r["kind"] == "data" and not r["own_label"] and vals:
            if len(vals) >= 2 and all(within_tol(v, running[a]) for a, v in vals.items()):
                r["kind"] = "agency_total"
                parsed.notes.append("unlabeled per-agency total row")
                continue
            if len(vals) == 1 and within_tol(next(iter(vals.values())), sum(running.values())):
                r["kind"] = "grand_total"
                parsed.notes.append("unlabeled grand total row")
                continue
        if r["kind"] == "data":
            for a, v in vals.items():
                running[a] += v

    width = len(canon_lefts)
    used = [k for k in range(width)
            if any(r["cells"][k] and r["cells"][k]["text"] for r in raw_rows if r["kind"] == "data")]
    item_c = used[-1] if used else None
    group_c = used[-2] if len(used) >= 2 else None
    section_cs = used[:-2]
    for r in raw_rows:
        r["group"] = r["labels"][group_c] if group_c is not None else ""
        r["item"] = r["labels"][item_c] if item_c is not None else ""
        r["sec_raw"] = [r["cells"][k] for k in section_cs]
    resolve_sections(raw_rows, len(section_cs))
    lines_from_rows(raw_rows, parsed.agencies, parsed)
    return True


# ---------------------------------------------------------------------------
# PDF: unruled layout (2014-2018)
# ---------------------------------------------------------------------------
def _words(page) -> list[dict]:
    upright = page.filter(lambda o: o.get("object_type") != "char" or o.get("upright", True))
    words = upright.extract_words(x_tolerance=1.5, y_tolerance=2)
    merged: list[dict] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if merged:
            p = merged[-1]
            numeric = re.fullmatch(r"[\d.,]+", w["text"]) and re.fullmatch(r"[\d.,]+", p["text"])
            if numeric and abs(p["top"] - w["top"]) < 2 and 0 <= w["x0"] - p["x1"] < 2.5:
                p.update(text=p["text"] + w["text"], x1=w["x1"])
                continue
        merged.append(dict(w))
    return merged


def _cluster_lines(words: list[dict], tol: float = 2.5) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: w["top"]):
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def parse_pdf_unruled(pdf, parsed: Parsed) -> bool:
    cols: list[tuple[float, str]] | None = None
    first_x0 = 0.0
    label_lines: list[dict] = []
    done = False
    for pno, page in enumerate(pdf.pages):
        if done:
            break
        text = page.extract_text() or ""
        if page_is_supplier_list(text):
            if cols:
                break
            continue
        words = _words(page)
        lines = _cluster_lines(words)
        top_limit = 0.0
        for line in lines:
            ags = [w for w in line if agency_name(w["text"])]
            if len(ags) >= 2:
                cols = [((w["x0"] + w["x1"]) / 2, agency_name(w["text"])) for w in ags]
                first_x0 = min(w["x0"] for w in ags)
                top_limit = max(w["bottom"] for w in ags)
                if parsed.title_period is None:
                    parsed.title_period = title_period(text)
                break
        if not cols:
            continue
        body = [w for w in words if w["top"] > top_limit]
        money = [w for w in body if MONEY_RE.match(w["text"])]
        labels = [w for w in body if not MONEY_RE.match(w["text"]) and w["text"] != "R$"
                  and w["x0"] < first_x0 - 2 and not re.fullmatch(r"\d{1,2}", w["text"])]
        page_lines = []
        for line in _cluster_lines(labels):
            joined = " ".join(w["text"] for w in line)
            if re.search(r"custos|despesas com publicidade|caixa", fold(joined)):
                continue
            segs = [[line[0]]]
            for w in line[1:]:
                if w["x0"] - segs[-1][-1]["x1"] > 12:
                    segs.append([w])
                else:
                    segs[-1].append(w)
            page_lines.append(dict(
                top=min(w["top"] for w in line), bottom=max(w["bottom"] for w in line),
                x0=line[0]["x0"], segs=[(s[0]["x0"], " ".join(w["text"] for w in s)) for s in segs],
                values={}, where=f"p{pno + 1}", dist=0.0))
        if not page_lines:
            continue
        for w in money:
            yc = (w["top"] + w["bottom"]) / 2
            best = min(page_lines, key=lambda ln: abs((ln["top"] + ln["bottom"]) / 2 - yc))
            d = abs((best["top"] + best["bottom"]) / 2 - yc)
            xc = (w["x0"] + w["x1"]) / 2
            agency = min(cols, key=lambda c: abs(c[0] - xc))[1]
            if agency in best["values"]:
                best["values"][agency] += " + " + w["text"]
            else:
                best["values"][agency] = w["text"]
            best["dist"] = max(best["dist"], d)
        label_lines.extend(page_lines)
        if any(row_kind([s for _, s in ln["segs"]]) == "grand_total" for ln in page_lines):
            done = True
    if not cols:
        return False
    parsed.agencies = [a for _, a in sorted(cols)]

    group_x = min(ln["x0"] for ln in label_lines)
    rows, group, prev = [], "", None
    for ln in label_lines:
        g = " ".join(t for x, t in ln["segs"] if x < group_x + 60)
        it = " ".join(t for x, t in ln["segs"] if x >= group_x + 60)
        kind = row_kind([g, it])
        if kind == "data" and not ln["values"] and not g and prev is not None \
                and prev["kind"] == "data" and prev["item"] and prev["where"] == ln["where"] \
                and ln["top"] - prev["bottom"] < 6:
            prev["item"] = f"{prev['item']} {it}"
            continue
        if kind == "data" and g:
            group = g
        if any(" + " in v for v in ln["values"].values()):
            raise ValueError(f"two amounts for one agency on the line {g} {it}: {ln['values']}")
        row = dict(kind=kind, section="", group=group if kind == "data" else "", item=it,
                   values=ln["values"], where=ln["where"], label_texts=[g, it], uncached=[],
                   bottom=ln["bottom"])
        if kind == "data" and ln["values"] and ln["dist"] > AMBIGUOUS_PT:
            parsed.ambiguous_rows += 1
        rows.append(row)
        prev = row
        if kind == "grand_total":
            break
    lines_from_rows(rows, parsed.agencies, parsed)
    return True


def parse_pdf(b: bytes, member: str, method: str) -> Parsed:
    parsed = Parsed(method=method, member=member)
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        full = " ".join((pg.extract_text() or "") for pg in pdf.pages[:6])
        parsed.no_payments_stated = "nao existem pagamentos" in fold(full)
        if parse_pdf_ruled(pdf, parsed):
            return parsed
        parsed = Parsed(method=method, member=member,
                        no_payments_stated=parsed.no_payments_stated)
        if parse_pdf_unruled(pdf, parsed):
            return parsed
        first = pdf.pages[0]
        if len(first.chars) < 20 and first.images:
            parsed.status = "unreadable_scan"
            return parsed
    raise ValueError(f"{member}: no summary table with an agency header found")


# ---------------------------------------------------------------------------
# Spreadsheets
# ---------------------------------------------------------------------------
def _grid_openpyxl(b: bytes) -> list[tuple[str, list[list], list[list[bool]], list]]:
    vals = openpyxl.load_workbook(io.BytesIO(b), data_only=True)
    forms = openpyxl.load_workbook(io.BytesIO(b), data_only=False)
    out = []
    for ws, wf in zip(vals.worksheets, forms.worksheets):
        grid = [[c.value for c in row] for row in ws.iter_rows()]
        uncached = [[(c.value is None and isinstance(f.value, str) and f.value.startswith("="))
                     for c, f in zip(row, frow)]
                    for row, frow in zip(ws.iter_rows(), wf.iter_rows())]
        merges = [(m.min_row - 1, m.max_row, m.min_col - 1, m.max_col)
                  for m in ws.merged_cells.ranges]
        out.append((ws.title, grid, uncached, merges))
    return out


def _grid_xlrd(b: bytes) -> list[tuple[str, list[list], list[list[bool]], list]]:
    book = xlrd.open_workbook(file_contents=b, formatting_info=True)
    out = []
    for sh in book.sheets():
        grid = [[(None if sh.cell_type(r, c) in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK)
                  else sh.cell_value(r, c)) for c in range(sh.ncols)] for r in range(sh.nrows)]
        uncached = [[False] * sh.ncols for _ in range(sh.nrows)]
        out.append((sh.name, grid, uncached, list(sh.merged_cells)))
    return out


def parse_spreadsheet(b: bytes, member: str, kind: str, method: str) -> Parsed:
    sheets = _grid_xlrd(b) if kind == "xls" else _grid_openpyxl(b)
    for name, grid, uncached, merges in sheets:
        header = next((i for i, row in enumerate(grid)
                       if sum(agency_name(v) is not None for v in row) >= 2), None)
        if header is None:
            continue
        parsed = Parsed(method=method, member=member)
        agency_cols = [(j, agency_name(v)) for j, v in enumerate(grid[header]) if agency_name(v)]
        parsed.agencies = [a for _, a in agency_cols]
        first_agency = agency_cols[0][0]
        for row in grid[:header + 1]:
            for v in row:
                if isinstance(v, str) and parsed.title_period is None:
                    parsed.title_period = title_period(v)

        owner: dict[tuple[int, int], tuple[int, int]] = {}
        for r0, r1, c0, c1 in merges:
            for r in range(r0, r1):
                for c in range(c0, c1):
                    owner[(r, c)] = (r0, c0)

        def row_labels(r: int) -> list[str]:
            """Label texts of row r; a cell merged with a cell to its left is blank."""
            out, seen = [], set()
            for c in range(first_agency):
                rr, cc = owner.get((r, c), (r, c))
                v = grid[rr][cc] if cc < len(grid[rr]) else None
                out.append(clean(v) if isinstance(v, str) and (rr, cc) not in seen else "")
                seen.add((rr, cc))
            return out

        body = []
        for r in range(header + 1, len(grid)):
            labels = row_labels(r)
            kind_r = row_kind(labels)
            body.append((r, labels, kind_r))
            if kind_r == "grand_total":
                break
        used = [c for c in range(first_agency)
                if any(lab[c] for _, lab, k in body if k == "data")]
        if not used:
            raise ValueError(f"{member}/{name}: no label columns")
        item_c = used[-1]
        group_c = used[-2] if len(used) >= 2 else None
        section_cs = used[:-2]
        label_merges = any(c0 < first_agency and r1 - r0 > 1 for r0, r1, c0, c1 in merges)

        rows, group, sections = [], "", [""] * len(section_cs)
        for r, labels, kind_r in body:
            values, flags = {}, []
            for c, a in agency_cols:
                v = grid[r][c] if c < len(grid[r]) else None
                if v is None and c < len(uncached[r]) and uncached[r][c]:
                    flags.append(a)
                values[a] = v
            if kind_r == "data":
                g = labels[group_c] if group_c is not None else ""
                it = labels[item_c]
                if not label_merges:
                    if g:
                        group = g
                    elif it:
                        g = group
                    for k, c in enumerate(section_cs):
                        sections[k] = labels[c] or sections[k]
                    secs = sections
                else:
                    secs = [labels[c] for c in section_cs]
                if not (g or it) and all(v in (None, "") for v in values.values()):
                    continue
                if flags:
                    raise ValueError(f"{member}/{name} row {r + 1}: formula without a stored "
                                     f"value in an amount cell for {flags}")
            else:
                g, it, secs = "", "", []
            rows.append(dict(kind=kind_r, section=" > ".join(s for s in secs if s), group=g,
                             item=it, values=values, where=name, label_texts=labels,
                             uncached=flags))
        lines_from_rows(rows, parsed.agencies, parsed)
        return parsed
    raise ValueError(f"{member}: no sheet with an agency header")


# ---------------------------------------------------------------------------
# One month
# ---------------------------------------------------------------------------
def parse_slot(path: Path, slot: str) -> tuple[list[Parsed], list[str]]:
    """Every parseable document in a downloaded file, and notes on what could not be used."""
    out, notes = [], []
    for member, b in unpack(path):
        kind = kind_of(member, b)
        if "fornecedor" in fold(member) and path.suffix.lower() == ".zip":
            continue
        if kind == "pdf":
            method = "pdf_in_xlsx_slot" if slot == "editable" else "text_pdf"
            out.append(parse_pdf(b, member, method))
        elif kind in ("xlsx", "template", "xls"):
            out.append(parse_spreadsheet(b, member, kind, "template" if kind == "template"
                                         else "xlsx"))
        elif kind == "html":
            notes.append(f"{slot} link returns a web page, not a document")
        else:
            notes.append(f"{slot} member {member} of unknown type")
    return out, notes


def within_tol(a: float, b: float) -> bool:
    return abs(a - b) <= max(TOL_ABS, TOL_REL * abs(b))


def build_month(period: str, files: dict[str, tuple[Path, str]]) -> tuple[dict, list[dict]]:
    rec = dict(period=period, amount_brl=np.nan, stated_total_brl=np.nan, stated_total_source="",
               total_matches=pd.NA, n_lines=0, n_agencies=0, agencies="", status="missing_file",
               parse_method="", source_file="", source_url="", agency_totals_match=pd.NA,
               pdf_xlsx_agree=pd.NA, pdf_xlsx_resolution="", pdf_xlsx_diff_brl=np.nan,
               pdf_xlsx_lines_match=np.nan,
               title_month_mismatch=False, ambiguous_rows=0, no_payments_stated=False,
               pdf_slot_broken=False, notes="")
    docs: list[tuple[Parsed, str, Path, str]] = []
    notes: list[str] = []
    for slot in ("editable", "pdf"):
        if slot not in files:
            continue
        path, url = files[slot]
        parsed, n = parse_slot(path, slot)
        notes += n
        if slot == "pdf" and any("web page" in x for x in n):
            rec["pdf_slot_broken"] = True
        docs += [(p, slot, path, url) for p in parsed]
    readable = [d for d in docs if d[0].status == "parsed"]
    order = {"xlsx": 0, "template": 1, "pdf_in_xlsx_slot": 2, "text_pdf": 3}
    readable.sort(key=lambda d: order[d[0].method])
    if not readable:
        if docs:
            rec["status"] = "unreadable_scan"
            rec["source_file"] = docs[0][2].name
            rec["source_url"] = docs[0][3]
            rec["parse_method"] = docs[0][0].method
        rec["notes"] = "; ".join(notes)
        return rec, []

    main, slot, path, url = readable[0]
    rec.update(amount_brl=main.total, n_lines=len(main.lines),
               agencies=",".join(main.agencies), parse_method=main.method,
               source_file=f"{path.name}::{main.member}" if main.member != path.name else path.name,
               source_url=url, ambiguous_rows=main.ambiguous_rows,
               no_payments_stated=any(d[0].no_payments_stated for d in docs))
    rec["n_agencies"] = len({x["agency"] for x in main.lines})
    stated = [(d[0].stated_total, d[0].method) for d in readable if d[0].stated_total is not None]
    if main.stated_total is None and main.no_payments_stated and not main.lines:
        stated.insert(0, (0.0, main.method))
    if not stated:
        stated = [(float(sum(d[0].agency_totals.values())), f"{d[0].method}:agency_total_row")
                  for d in readable if d[0].agency_totals]
    if stated:
        rec["stated_total_brl"], rec["stated_total_source"] = stated[0]
        rec["total_matches"] = within_tol(main.total, stated[0][0])
    rec["status"] = "parsed" if stated else "no_stored_total"
    if main.agency_totals:
        sums = main.agency_sums()
        rec["agency_totals_match"] = all(within_tol(sums.get(a, 0.0), v)
                                         for a, v in main.agency_totals.items())
    periods = {d[0].title_period for d in readable if d[0].title_period}
    rec["title_month_mismatch"] = bool(periods - {period})
    spreadsheets = [d[0] for d in readable if d[0].method in ("xlsx", "template")]
    pdfs = [d[0] for d in readable if d[0].method in ("text_pdf", "pdf_in_xlsx_slot")]
    if spreadsheets and pdfs:
        x, p = spreadsheets[0], pdfs[0]
        xs, ps = x.agency_sums(), p.agency_sums()
        rec["pdf_xlsx_agree"] = all(within_tol(xs.get(a, 0.0), ps.get(a, 0.0))
                                    for a in set(xs) | set(ps))
        if not rec["pdf_xlsx_agree"]:
            # The disagreement is explained when the PDF's own cells do not add up to the PDF's
            # own printed agency totals while the spreadsheet's cells do.
            pdf_self = bool(p.agency_totals) and all(within_tol(ps.get(a, 0.0), v)
                                                     for a, v in p.agency_totals.items())
            x_vs_pdf_totals = bool(p.agency_totals) and all(
                within_tol(xs.get(a, 0.0), v) for a, v in p.agency_totals.items())
            if not pdf_self and x_vs_pdf_totals:
                rec["pdf_xlsx_resolution"] = "pdf_cells_disagree_with_own_totals"
        rec["pdf_xlsx_diff_brl"] = x.total - p.total
        pool = [(y["agency"], round(y["amount_brl"], 2)) for y in p.lines]
        hit = 0
        for y in x.lines:
            key = (y["agency"], round(y["amount_brl"], 2))
            if key in pool:
                pool.remove(key)
                hit += 1
        rec["pdf_xlsx_lines_match"] = hit / len(x.lines) if x.lines else 1.0
    notes += main.notes
    rec["notes"] = "; ".join(dict.fromkeys(notes))
    lines = [dict(bank_key=BANK_KEY, cnpj8=CNPJ8, period=period, **ln, basis=BASIS_AD,
                  source_file=rec["source_file"], source_url=url, parse_method=main.method)
             for ln in main.lines]
    return rec, lines


def flag_repeats(monthly: pd.DataFrame, lines: pd.DataFrame) -> pd.Series:
    blocks = {}
    for (period, agency), g in lines.groupby(["period", "agency"]):
        blocks[(period, agency)] = tuple(sorted(zip(g["category"], g["subcategory"],
                                                    g["amount_brl"].round(2))))
    periods = monthly["period"].tolist()
    out = []
    for i, period in enumerate(periods):
        rep = []
        if i:
            for (p, a), block in blocks.items():
                if p == period and block and blocks.get((periods[i - 1], a)) == block:
                    rep.append(a)
        out.append(",".join(sorted(rep)))
    return pd.Series(out, index=monthly.index)


# ---------------------------------------------------------------------------
# Sponsorship
# ---------------------------------------------------------------------------
SPONSOR_COLS = {"project": ("nome do projeto", "projeto"), "beneficiary": ("patrocinado",
                "contratado"), "cnpj": ("cnpj",), "contract_date": ("data",),
                "amount_brl": ("valor",), "justification": ("justificativa", "just"),
                "uf": ("uf",), "city": ("cidade",)}


def _sponsor_tables(member: str, b: bytes) -> list[tuple[str, pd.DataFrame]]:
    kind = kind_of(member, b)
    if kind in ("xlsx", "template"):
        wb = openpyxl.load_workbook(io.BytesIO(b), data_only=True)
        return [(ws.title, pd.DataFrame([list(r) for r in ws.iter_rows(values_only=True)]))
                for ws in wb.worksheets]
    if kind == "xls":
        return [(n, pd.DataFrame(g)) for n, g, _, _ in _grid_xlrd(b)]
    if kind == "html":
        text = b.decode("iso-8859-1")
        tables = pd.read_html(io.StringIO(text), header=None)
        return [(f"html{i}", t) for i, t in enumerate(tables)]
    raise ValueError(f"{member}: unsupported sponsorship file type {kind}")


def parse_sponsorship_file(path: Path, year: int, url: str) -> tuple[list[dict], list[str]]:
    rows, notes = [], []
    for member, b in unpack(path):
        program = "caixa_cultural" if "pocc" in fold(member) else "general"
        found = False
        for sheet, df in _sponsor_tables(member, b):
            df = df.astype(object).where(pd.notna(df), None)
            header = None
            if all(isinstance(c, str) for c in df.columns) and any(
                    "valor" in fold(c) for c in df.columns):
                df = pd.concat([pd.DataFrame([list(df.columns)], columns=df.columns), df],
                               ignore_index=True)
                df.columns = range(df.shape[1])
            for i in range(min(10, len(df))):
                cells = [fold(clean(v)) for v in df.iloc[i]]
                if any(c.startswith("valor") for c in cells) and any("projeto" in c for c in cells):
                    header = i
                    break
            if header is None:
                notes.append(f"{member}/{sheet}: no header row, sheet skipped")
                continue
            found = True
            hdr = [fold(clean(v)) for v in df.iloc[header]]
            colmap = {}
            for key, keys in SPONSOR_COLS.items():
                for j, h in enumerate(hdr):
                    if h and any(h.startswith(k) for k in keys) and j not in colmap.values():
                        colmap[key] = j
                        break
            for k in ("project", "contract_date", "amount_brl"):
                if k not in colmap:
                    raise ValueError(f"{member}/{sheet}: no {k} column in header {hdr}")
            stated = None
            for i in range(header + 1, len(df)):
                r = df.iloc[i]
                raw_amount, raw_date = r[colmap["amount_brl"]], r[colmap["contract_date"]]
                filled = [v for v in r if v not in (None, "") and clean(v)]
                if raw_date in (None, "") or isinstance(raw_date, str) and not clean(raw_date):
                    if len(filled) == 1 and isinstance(filled[0], (int, float)):
                        stated = float(filled[0])
                    elif filled and not (len(filled) == 1 and isinstance(filled[0], dt.datetime)):
                        notes.append(f"{member}/{sheet} row {i + 1}: no contract date, skipped: "
                                     f"{[clean(v)[:40] for v in filled]}")
                    continue
                amount = parse_brl(raw_amount)
                if isinstance(raw_date, (dt.datetime, dt.date)):
                    date = pd.Timestamp(raw_date)
                else:
                    date = pd.to_datetime(clean(raw_date), dayfirst=True, errors="coerce")
                rec = dict(bank_key=BANK_KEY, cnpj8=CNPJ8, year_file=year,
                           contract_date=date, program=program, amount_brl=amount,
                           basis=BASIS_SPONSOR,
                           **{k: clean(r[j]) for k, j in colmap.items()
                              if k not in ("amount_brl", "contract_date")},
                           source_file=f"{path.name}::{member}", sheet=sheet, row=i + 1,
                           source_url=url)
                rows.append(rec)
            if stated is not None:
                got = sum(x["amount_brl"] or 0 for x in rows
                          if x["source_file"].endswith(member) and x["sheet"] == sheet)
                notes.append(f"{member}/{sheet}: stated total {stated:,.2f}, contracts sum "
                             f"{got:,.2f} ({'match' if within_tol(got, stated) else 'DIFFER'})")
        if not found:
            raise ValueError(f"{member}: no sheet with a sponsorship header")
    return rows, notes


# ---------------------------------------------------------------------------
# Panel key
# ---------------------------------------------------------------------------
def establish_panel_key() -> str:
    codes, names = set(), {}
    for f in sorted(paths.IF_DATA_LIST.glob("*.csv")):
        d = pd.read_csv(f, dtype=str, sep=None, engine="python")
        lead = d[d["CnpjInstituicaoLider"].fillna("").str.zfill(8) == CNPJ8]
        codes |= set(lead["CodConglomeradoPrudencial"].dropna().str.strip()) - {""}
        names.update(dict(zip(d["CodInst"].str.strip(), d["NomeInstituicao"])))
    if len(codes) != 1:
        raise AssertionError(f"IF.data lists give {sorted(codes)} prudential codes for leader "
                             f"{CNPJ8}; expected exactly one")
    code = codes.pop()
    if "caixa economica federal" not in fold(names.get(code, "")):
        raise AssertionError(f"{code} is named {names.get(code)!r} in IF.data")
    cosif = pd.read_parquet(OUT_DIR / "cosif_advertising_monthly.parquet")
    rows = cosif[(cosif["level"] == "conglomerate") & (cosif["entity_key"] == code)]
    leaders = set(rows["cnpj_leader"].dropna().astype(str).str.zfill(8))
    if rows.empty or leaders != {CNPJ8}:
        raise AssertionError(f"COSIF conglomerate {code}: leader CNPJs {leaders}, expected {CNPJ8}")
    log.info("panel_key %s = %s (IF.data), confirmed by %d COSIF conglomerate rows", code,
             names[code], len(rows))
    return code


# ---------------------------------------------------------------------------
# Validation and the COSIF comparison
# ---------------------------------------------------------------------------
def validate(monthly: pd.DataFrame, lines: pd.DataFrame, contracts: pd.DataFrame,
             latest: str) -> None:
    expected = pd.period_range(pd.Period(FIRST_MONTH, "M"), pd.Period(latest, "M"), freq="M")
    expected = [p.strftime("%Y%m") for p in expected]
    got = monthly["period"].tolist()
    if got != expected or monthly["status"].isna().any():
        raise AssertionError(f"monthly grid incomplete: missing "
                             f"{sorted(set(expected) - set(got))[:10]}")                 # check 1

    stated = monthly[monthly["stated_total_brl"].notna()]
    share = float(stated["total_matches"].astype(bool).mean()) if len(stated) else 0.0
    log.info("check 2: %d of %d months with a stated total match it (%.1f%%)",
             int(stated["total_matches"].astype(bool).sum()), len(stated), 100 * share)
    bad = stated[~stated["total_matches"].astype(bool)]
    if len(bad):
        log.warning("months whose lines do not sum to the stated total:\n%s",
                    bad.assign(diff=bad["amount_brl"] - bad["stated_total_brl"])
                    [["period", "parse_method", "amount_brl", "stated_total_brl", "diff"]]
                    .to_string(index=False))
    if share < MIN_MATCH_SHARE:
        raise AssertionError(f"only {share:.1%} of months match their stated total")

    both = monthly[monthly["pdf_xlsx_agree"].notna()]                                  # check 3
    disagree = both[~both["pdf_xlsx_agree"].astype(bool)]
    explained = disagree[disagree["pdf_xlsx_resolution"] != ""]
    if len(explained):
        log.warning("spreadsheet and PDF disagree where the PDF's cells do not add up to its own "
                    "agency totals (spreadsheet kept):\n%s", explained[
                        ["period", "amount_brl", "pdf_xlsx_diff_brl", "pdf_xlsx_resolution"]]
                    .to_string(index=False))
    disagree = disagree[disagree["pdf_xlsx_resolution"] == ""]
    log.info("check 3: spreadsheet and PDF both parsed for %d months; %d disagree on an agency "
             "sum; line-level match median %.3f, min %.3f", len(both), len(disagree),
             both["pdf_xlsx_lines_match"].median(), both["pdf_xlsx_lines_match"].min())
    if len(disagree):
        raise AssertionError("spreadsheet and PDF disagree:\n" + disagree[
            ["period", "amount_brl", "pdf_xlsx_diff_brl"]].to_string(index=False))

    if (lines["amount_brl"] < 0).any():                                                 # check 4
        raise AssertionError(f"negative amounts:\n{lines[lines['amount_brl'] < 0].head()}")

    for period, target in KNOWN_TOTALS_M.items():                                       # check 5
        v = monthly.loc[monthly["period"] == period, "amount_brl"].iloc[0] / 1e6
        if round(v, 1) != target:
            raise AssertionError(f"{period}: R$ {v:.3f}m, expected {target}m")
    log.info("check 5: April-June 2020 reproduce %s", list(KNOWN_TOTALS_M.values()))

    if contracts["amount_brl"].isna().any() or (contracts["amount_brl"] <= 0).any() \
            or contracts["contract_date"].isna().any():                                 # check 7
        bad = contracts[contracts["amount_brl"].isna() | (contracts["amount_brl"] <= 0)
                        | contracts["contract_date"].isna()]
        raise AssertionError(f"sponsorship rows without a positive amount or a date:\n"
                             f"{bad[['source_file', 'row', 'project', 'amount_brl']].head(10)}")


def print_cosif_ratios(monthly: pd.DataFrame, code: str) -> None:
    path = OUT_DIR / "cosif_advertising_quarterly.parquet"
    q = pd.read_parquet(path)
    q = q[(q["level"] == "conglomerate") & (q["entity_key"] == code) & (q["year"] == 2025)]
    m = monthly[monthly["period"].str[:4] == "2025"].copy()
    m["quarter"] = (m["period"].str[4:].astype(int) - 1) // 3 + 1
    cx = m.groupby("quarter").agg(caixa=("amount_brl", "sum"), n=("amount_brl", "count"))
    t = q.set_index("quarter")[["adv", "all3"]].join(cx)
    t.loc["H1"] = t.loc[[1, 2]].sum()
    t.loc["H2"] = t.loc[[3, 4]].sum()
    t.loc["2025"] = t.loc[[1, 2, 3, 4]].sum()
    t["ratio_adv"] = t["caixa"] / t["adv"]
    t["ratio_all3"] = t["caixa"] / t["all3"]
    for c in ("adv", "all3", "caixa"):
        t[c] = (t[c] / 1e6).round(1)
    log.info("2025: Caixa own files against COSIF conglomerate %s (R$ million; not gated):\n%s",
             code, t.round(3).to_string())
    mm = pd.read_parquet(OUT_DIR / "cosif_advertising_monthly.parquet")
    mm = mm[(mm["level"] == "conglomerate") & (mm["entity_key"] == code)
            & (mm["data_base"].str[:4] == "2025")].set_index("data_base")
    j = m.set_index("period")[["amount_brl"]].join(mm[["adv_flow", "promo_flow", "publ_flow"]])
    j["all3_flow"] = j[["adv_flow", "promo_flow", "publ_flow"]].sum(axis=1)
    log.info("2025 same-month correlation with COSIF: adv %.2f, all3 %.2f",
             j["amount_brl"].corr(j["adv_flow"]), j["amount_brl"].corr(j["all3_flow"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Caixa advertising outlays and sponsorship contracts")
    ap.add_argument("--refresh", action="store_true", help="download the lists and files again")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()

    fetcher = Fetcher()
    ad_docs = advertising_documents(list_items(fetcher, AD_LIST, a.refresh))
    sponsor_docs = sponsorship_documents(list_items(fetcher, SPONSOR_LIST, a.refresh))
    log.info("%d monthly advertising documents and %d sponsorship files listed", len(ad_docs),
             len(sponsor_docs))
    files: dict[str, dict[str, tuple[Path, str]]] = {}
    for d in ad_docs:
        files.setdefault(d["period"], {})[d["slot"]] = (download(fetcher, d["url"], ad_path(d),
                                                                 a.refresh), d["url"])
    sponsor_files = [(download(fetcher, d["url"], sponsor_path(d), a.refresh), d)
                     for d in sponsor_docs]

    latest = max(files)
    periods = [p.strftime("%Y%m") for p in pd.period_range(pd.Period(FIRST_MONTH, "M"),
                                                           pd.Period(latest, "M"), freq="M")]
    records, all_lines = [], []
    for period in periods:
        rec, lines = build_month(period, files.get(period, {}))
        records.append(rec)
        all_lines += lines
        log.info("  %s %-16s %-16s lines %3d  sum %14s  stated %14s  %s", period, rec["status"],
                 rec["parse_method"], rec["n_lines"],
                 f"{rec['amount_brl']:,.2f}" if pd.notna(rec["amount_brl"]) else "-",
                 f"{rec['stated_total_brl']:,.2f}" if pd.notna(rec["stated_total_brl"]) else "-",
                 rec["notes"])
    monthly = pd.DataFrame(records)
    lines = pd.DataFrame(all_lines)
    monthly["repeated_agency_block"] = flag_repeats(monthly, lines)
    monthly.insert(0, "cnpj8", CNPJ8)
    monthly.insert(0, "bank_key", BANK_KEY)
    monthly["basis"] = BASIS_AD

    contract_rows, sponsor_notes = [], []
    for path, d in sponsor_files:
        rows, notes = parse_sponsorship_file(path, d["year"], d["url"])
        contract_rows += rows
        sponsor_notes += notes
    contracts = pd.DataFrame(contract_rows)
    contracts["date_outside_year_file"] = contracts["contract_date"].dt.year != contracts["year_file"]
    # 2022's file was converted from a PDF: some rows repeat the project, beneficiary and CNPJ
    # run together inside the text columns. Amount and date are unaffected.
    contracts["text_fields_garbled"] = contracts["project"].str.contains(
        r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", regex=True)
    contracts["duplicate_row"] = contracts.duplicated(
        ["source_file", "project", "beneficiary", "contract_date", "amount_brl"], keep=False)
    for n in sponsor_notes:
        log.info("  sponsorship: %s", n)

    code = establish_panel_key()                                                        # check 6
    for frame in (monthly, lines, contracts):
        frame["panel_key"] = code
    validate(monthly, lines, contracts, latest)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("caixa_advertising_lines", lines),
                        ("caixa_advertising_monthly", monthly),
                        ("caixa_sponsorship_contracts", contracts)):
        frame.to_parquet(a.out_dir / f"{name}.parquet", index=False)
        frame.to_csv(a.out_dir / f"{name}.csv", index=False)
        log.info("%s: %d rows -> %s", name, len(frame), a.out_dir / f"{name}.parquet")

    monthly["year"] = monthly["period"].str[:4]
    cov = (monthly.groupby(["year", "status"]).size().unstack(fill_value=0)
           .join(monthly.groupby("year")["amount_brl"].sum().div(1e6).round(1)
                 .rename("sum_brl_m")))
    log.info("coverage by year:\n%s", cov.to_string())
    flags = monthly[(monthly["repeated_agency_block"] != "") | monthly["title_month_mismatch"]
                    | (monthly["ambiguous_rows"] > 0) | monthly["no_payments_stated"]
                    | monthly["pdf_slot_broken"] | (monthly["agency_totals_match"] == False)]  # noqa: E712
    log.info("flagged months:\n%s", flags[["period", "parse_method", "repeated_agency_block",
                                           "title_month_mismatch", "ambiguous_rows",
                                           "no_payments_stated", "pdf_slot_broken",
                                           "agency_totals_match"]].to_string(index=False))
    sp = (contracts.groupby(["year_file", "program"])
          .agg(contracts=("amount_brl", "size"), sum_brl_m=("amount_brl", "sum"),
               outside_year=("date_outside_year_file", "sum")))
    sp["sum_brl_m"] = (sp["sum_brl_m"] / 1e6).round(1)
    log.info("sponsorship contracts by file year:\n%s", sp.to_string())
    print_cosif_ratios(monthly, code)


if __name__ == "__main__":
    main()
