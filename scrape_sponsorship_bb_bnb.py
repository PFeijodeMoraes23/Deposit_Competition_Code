"""
scrape_sponsorship_bb_bnb.py
============================
Published sponsorship (patrocinio) spend of Banco do Brasil and Banco do Nordeste, line by
line, with period sums. A robustness series kept apart from advertising: sponsorship does not
enter the main awareness measure.

Sources
-------
  Banco do Brasil (bank_key bb, CNPJ8 00000000)
    Page  https://www.bb.com.br/site/compras-contratacao-e-venda-de-imoveis/servicos-de-publicidade-e-patrocinio/
    "Patrocinios - Valores Pagos": one file per half-year from 2022 (one per year for 2020 and
    2021), each in PDF, XLSX and ODS, linked from the page's embedded year accordion with a
    label such as "Janeiro a Junho de 2024". Inside, a block per month ("Mes", MM/YYYY), then
    one row per payment: Nome Projeto, CNPJ Fornecedor, Razao Social, Valor. No totals are
    published. Known defects, handled below: PatrociniosJanDez2021.xlsx is a byte-identical copy
    of PatrociniosJanJun2024.xlsx (the 2021 ODS and PDF carry 2021); the PDFs of 2022 H1 to
    2024 H1 stop before the end of their last month (4 to 23 payments missing, no "R$" printed
    for them); the 2026 H1 XLSX has a second sheet that is a pivot of June 2026 alone.
  Banco do Nordeste (bank_key bnb, CNPJ8 07237373)
    Page  https://www.bnb.gov.br/web/guest/acesso-a-informacao/licitacoes-e-contratos/patrocinios
    The page lists its files through the Liferay content API; the "Projetos Patrocinados"
    fragment names its content set (2715560 when written), read from the page at run time:
      /o/headless-delivery/v1.0/content-sets/<id>/content-set-elements
    One ODS per year (2019 in two semester files), one row per sponsored project, a
    "VALOR TOTAL" or "TOTAL" row at the foot. Stated totals are stored results of SUM formulas;
    the 2019 H2 approved total (5,140,300) is stale, its own formula over the published cells
    gives 5,240,300.

Units and amount kinds
----------------------
  Nominal reais. amount_kind comes from the document and is never pooled:
    bb   "Valores Pagos"                                        -> paid
    bnb  "VALOR DO PATROCINIO" (2012 to 2019 H1), "VALOR APROVADO" -> contracted
         "VALOR DO DESEMBOLSO", "VALOR DESEMBOLSADO"             -> disbursed
         "VALOR LIQUIDADO" (2025 file, same position as the
         disbursed column of the other years)                    -> disbursed
  amount_label keeps the column header as published. A BNB file with both an approved and a
  disbursed column gives two lines per project, one per kind. BNB lists by the year of its
  file, including disbursements on contracts signed in earlier years.

Rules
-----
  - A file's true period is read from its content (BB: the MM/YYYY month markers; BNB: the
    year in the title and "Referencia/ano" rows, and "jul a dez" for a half) and compared with
    the period it is filed under (BB: link label and file name, which must agree; BNB: the API
    "Ano" field and title). A conflict is flagged and resolved by content: the file is not used
    for the period it is filed under and is compared with the file that does cover its content.
  - BB: for each filed period the XLSX is the line source, the ODS when the XLSX conflicts, the
    PDF only when neither spreadsheet can be used. The other spreadsheet must match the line
    source month by month (count and sum). PDF amounts are read from the Valor column in
    content-stream order (see _pdf_runs); by month they must be a sub-multiset of the line
    source's amounts, and the amounts the PDF lacks are counted in pdf_missing_lines. Every
    PDF page's count of "R$" marks must equal its count of parsed amounts (logged otherwise).
  - A stated total that is the stored result of a single-column SUM formula is also
    re-evaluated on the published cells (stated_formula_recomputed_brl) when the two differ.
  - Total rows are excluded from lines, counted, and their values kept as stated totals.
  - Rows with neither an amount nor a total label (notes, repeated preambles) are skipped and
    logged. Sheets without the expected header are ignored and logged.
  - A blank amount cell gives no line for that kind (it is not a zero).

Outputs (paths.AWARENESS_PROC)
------------------------------
  sponsorship_lines.{parquet,csv}    one row per published amount: bank_key, cnpj8,
      panel_key, period (YYYYMM for BB, YYYY for BNB), period_part (H1/H2 for BNB's 2019
      semester files, blank otherwise), project, beneficiary, beneficiary_cnpj, amount_brl,
      amount_kind, amount_label, source_file, source_url, parse_method, sheet_or_page,
      is_total_row (always False)
  sponsorship_periods.{parquet,csv}  bank x period x period_part x amount_kind: amount_brl,
      stated_total_brl, stated_formula_recomputed_brl, total_diff_brl, total_matches,
      n_lines, n_total_rows_excluded, status (parsed / unreadable / missing_file),
      filed_period, content_period, period_conflict (True for every month of a filed period
      with a conflicting file), formats_parsed (line source first), formats_agree,
      pdf_missing_lines, amount_labels, source_file, source_url
  BB total rows are counted on the first month of their file; BNB's on every kind of the
  file.
  panel_key is the prudential conglomerate code of the bank in the IF.data lists
  (paths.IF_DATA_LIST), confirmed against the COSIF advertising panel.

Validation (a failure aborts before writing)
--------------------------------------------
  1. Every expected period has a status: BB every month from the first to the last filed
     half-year, BNB every year from the first to the last listed file (both halves for a
     year filed by semester).
  2. Lines sum to the stated total within R$1 in at least 95% of period-kinds with a stated
     total, a stale formula result whose re-evaluation equals the line sum counting as a
     match (the one case: BNB 2019 H2 contracted; 16 of 17 match outright).
  3. Every file's content period matches its filed period, or the conflict is flagged and
     another file covers the filed period; a filed period covered by no usable file aborts.
  4. BB: the other spreadsheet of a filed period agrees with the line source month by month
     (same count of amounts, sums within R$1), and every PDF amount is among the line
     source's amounts of the same month.
  5. Amounts are non-negative.
  6. panel_key: the IF.data prudential code for the bank's CNPJ8 is unique and the COSIF
     conglomerate row with that code has the same leader CNPJ8.
  Printed, not gated: yearly totals by bank and kind, and BB paid sponsorship over BB's
  individual DVA advertising line (cvm_advertising_quarterly, CD_CVM 001023).

Usage
-----
  python scrape_sponsorship_bb_bnb.py
  python scrape_sponsorship_bb_bnb.py --refresh        # refetch the index pages and files
  python scrape_sponsorship_bb_bnb.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import urllib.parse
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pdfplumber
from lxml import etree

from utils import paths
from utils.disclosure_common import http_get, parse_number

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

OUT_DIR = paths.AWARENESS_PROC
BB_RAW = paths.AWARENESS_RAW / "bb_sponsorship"
BNB_RAW = paths.AWARENESS_RAW / "bnb_sponsorship"

BB_PAGE = ("https://www.bb.com.br/site/compras-contratacao-e-venda-de-imoveis/"
           "servicos-de-publicidade-e-patrocinio/")
BNB_SITE = "https://www.bnb.gov.br"
BNB_PAGE = BNB_SITE + "/web/guest/acesso-a-informacao/licitacoes-e-contratos/patrocinios"
BNB_FRAGMENT_TITLE = "Projetos Patrocinados"

BANKS = {"bb": {"cnpj8": "00000000", "name_token": "BB"},
         "bnb": {"cnpj8": "07237373", "name_token": "NORDESTE"}}

TOL_BRL = 1.0
MIN_TOTAL_MATCH_SHARE = 0.95

MONTHS_PT = {"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
             "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11,
             "dezembro": 12}
FILE_TOKENS = {"jan": 1, "jun": 6, "jul": 7, "dez": 12}
MONTH_RE = re.compile(r"^(0[1-9]|1[0-2])/(20\d{2})$")
PDF_AMOUNT_RE = re.compile(r"(?<![\d.,])(-?\d{1,3}(?:\.\d{3})*,\d{2})$")
CNPJ_RE = re.compile(r"\d{1,3}(?:\.\d{3}){0,2}/\d{4}-\d{2}")

T_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
O_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
X_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"


def _deaccent(s: str) -> str:
    table = str.maketrans("áàâãäéêèíìóôõòúùüçÁÀÂÃÄÉÊÈÍÌÓÔÕÒÚÙÜÇ",
                          "aaaaaeeeiioooouuucAAAAAEEEIIOOOOUUUC")
    return s.translate(table)


# ---------------------------------------------------------------------------
# Downloads (cached under the raw folders)
# ---------------------------------------------------------------------------
def fetch_text(url: str, cache: Path, refresh: bool) -> str:
    if cache.exists() and not refresh:
        return cache.read_text(encoding="utf-8")
    status, body = http_get(url, timeout=90)
    if body is None:
        raise SystemExit(f"GET {url} failed with status {status}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(body, encoding="utf-8", newline="")
    log.info("  fetched %s -> %s", url, cache.name)
    return body


def fetch_file(url: str, dest: Path, refresh: bool) -> bool:
    """True when the file is on disk after the call."""
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return True
    status, body = http_get(url, binary=True, timeout=120)
    if not body:
        log.warning("  download failed (%s): %s", status, url)
        return dest.exists()
    dest.write_bytes(body)
    log.info("  downloaded %s (%d bytes)", dest.name, len(body))
    return True


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def _ods_inline(el) -> str:
    out = [el.text or ""]
    for ch in el:
        tag = etree.QName(ch).localname
        if tag == "s":
            out.append(" " * int(ch.get(f"{{{X_NS}}}c", "1")))
        elif tag == "tab":
            out.append("\t")
        elif tag == "line-break":
            out.append("\n")
        elif tag != "annotation":
            out.append(_ods_inline(ch))
        out.append(ch.tail or "")
    return "".join(out)


def _ods_value(cell):
    vtype = cell.get(f"{{{O_NS}}}value-type")
    if vtype in ("float", "currency", "percentage"):
        return float(cell.get(f"{{{O_NS}}}value"))
    if vtype == "date":
        return cell.get(f"{{{O_NS}}}date-value")
    return "\n".join(_ods_inline(p) for p in cell.findall(f"{{{X_NS}}}p")).strip()


def read_ods(path: Path, formulas: dict | None = None) -> dict[str, list[tuple[int, list]]]:
    """Sheet -> [(1-based row number, cells)] for non-empty rows. Repeated columns and rows are
    expanded; an empty repeated run only advances the counters. When `formulas` is given it is
    filled with {(sheet, row number, 0-based column): table:formula}."""
    root = etree.fromstring(zipfile.ZipFile(path).read("content.xml"))
    sheets = {}
    for table in root.iter(f"{{{T_NS}}}table"):
        rows, rownum = [], 0
        for row in table.iter(f"{{{T_NS}}}table-row"):
            rrep = int(row.get(f"{{{T_NS}}}number-rows-repeated", "1"))
            values, col = {}, 0
            for cell in row:
                if etree.QName(cell).localname not in ("table-cell", "covered-table-cell"):
                    continue
                crep = int(cell.get(f"{{{T_NS}}}number-columns-repeated", "1"))
                v = _ods_value(cell)
                if formulas is not None and cell.get(f"{{{T_NS}}}formula"):
                    formulas[(table.get(f"{{{T_NS}}}name"), rownum + 1, col)] =                         cell.get(f"{{{T_NS}}}formula")
                if v != "":
                    for k in range(crep):
                        values[col + k] = v
                col += crep
            if values:
                if rrep > 1000:
                    raise ValueError(f"{path.name}: {rrep} repeated non-empty rows")
                cells = [values.get(i, "") for i in range(max(values) + 1)]
                for k in range(rrep):
                    rows.append((rownum + k + 1, cells))
            rownum += rrep
        sheets[table.get(f"{{{T_NS}}}name")] = rows
    return sheets


def read_xlsx(path: Path) -> dict[str, list[tuple[int, list]]]:
    wb = openpyxl.load_workbook(path, data_only=True)
    sheets = {}
    for ws in wb.worksheets:
        rows = []
        for i, r in enumerate(ws.iter_rows(values_only=True), start=1):
            cells = ["" if v is None else (v.strip() if isinstance(v, str) else v) for v in r]
            while cells and cells[-1] == "":
                cells.pop()
            if cells:
                rows.append((i, cells))
        sheets[ws.title] = rows
    return sheets


def to_amount(v) -> tuple[float | None, bool]:
    """(amount, parsed_from_text)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v), False
    s = str(v).strip()
    if not s:
        return None, False
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s), True
    if re.fullmatch(r"(R\$)?\s*-?[\d.]+,\d{1,2}", s):
        return parse_number(s, brazilian=True), True
    return None, False


def _text(v) -> str:
    return v.strip() if isinstance(v, str) else ("" if v == "" else str(v))


# ---------------------------------------------------------------------------
# Parse results
# ---------------------------------------------------------------------------
@dataclass
class Parsed:
    lines: list[dict] = field(default_factory=list)
    stated: dict[tuple[str, str], float] = field(default_factory=dict)  # (part, kind) -> total
    n_total_rows: int = 0
    content_periods: list[str] = field(default_factory=list)
    kinds: set = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    formula_totals: dict[tuple[str, str], float] = field(default_factory=dict)
    ignored_sums: list[tuple[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Banco do Brasil
# ---------------------------------------------------------------------------
def bb_manifest(refresh: bool) -> pd.DataFrame:
    page = fetch_text(BB_PAGE, BB_RAW / "_index_page.html", refresh)
    page = page.replace("\\/", "/").replace('\\"', '"')
    page = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), page)
    rows = []
    for m in re.finditer(r'<a[^>]*href="(https?://[^"]*/Patrocinios([A-Za-z]{3})([A-Za-z]{3})'
                         r'(20\d{2})\.(pdf|xlsx|ods))"[^>]*>(.*?)</a>', page, re.S):
        url, t0, t1, year, fmt, label = m.groups()
        label = _deaccent(html.unescape(re.sub(r"<[^>]+>", " ", label))).lower()
        lm = re.search(r"(janeiro|julho)\s+a\s+(junho|dezembro)\s+de\s+(20\d{2})", label)
        if not lm:
            raise AssertionError(f"BB link without a period label: {url} ({label!r})")
        start = f"{lm.group(3)}{MONTHS_PT[lm.group(1)]:02d}"
        end = f"{lm.group(3)}{MONTHS_PT[lm.group(2)]:02d}"
        fstart = f"{year}{FILE_TOKENS[t0.lower()]:02d}"
        fend = f"{year}{FILE_TOKENS[t1.lower()]:02d}"
        if (start, end) != (fstart, fend):
            raise AssertionError(f"BB link label {start}-{end} disagrees with file name {url}")
        rows.append({"url": url.replace("http://", "https://"), "fmt": fmt.lower(),
                     "filed_start": start, "filed_end": end,
                     "file": url.rsplit("/", 1)[1]})
    man = pd.DataFrame(rows).drop_duplicates(["file"])
    if man.empty:
        raise SystemExit("no sponsorship links found on the Banco do Brasil page")
    log.info("BB page: %d sponsorship files, %d filed periods, %s to %s", len(man),
             man[["filed_start", "filed_end"]].drop_duplicates().shape[0],
             man["filed_start"].min(), man["filed_end"].max())
    return man


def parse_bb_spreadsheet(path: Path, fmt: str, url: str) -> Parsed:
    sheets = read_ods(path) if fmt == "ods" else read_xlsx(path)
    out = Parsed(kinds={"paid"})
    method = "ods-lxml" if fmt == "ods" else "xlsx-openpyxl"
    used = 0
    for sheet, rows in sheets.items():
        joined_all = " ".join(_text(c) for _, r in rows[:5] for c in r)
        if "Valores Pagos" not in joined_all:
            if rows:
                amounts = [to_amount(r[-1])[0] for _, r in rows]
                total = sum(a for a in amounts if a is not None)
                out.ignored_sums.append((sheet, total))
                out.notes.append(f"sheet {sheet!r} ignored ({len(rows)} rows, last-column "
                                 f"sum {total:,.2f}, header {[_text(c) for c in rows[0][1]]})")
            continue
        used += 1
        month, cols = None, None
        for rn, cells in rows:
            texts = [_text(c) for c in cells if _text(c)]
            if len(texts) == 1 and MONTH_RE.match(texts[0]):
                mm, yy = MONTH_RE.match(texts[0]).groups()
                month = yy + mm
                out.content_periods.append(month)
                continue
            if "Nome Projeto" in [_text(c) for c in cells]:
                cols = {_text(c): i for i, c in enumerate(cells) if _text(c)}
                continue
            if month is None or cols is None:
                continue                                  # title block, "Mes" label
            if texts == ["Mês"]:
                continue
            if texts and re.match(r"(?i)^(valor\s+)?total\b", texts[0]):
                out.n_total_rows += 1
                amt = to_amount(cells[cols["Valor"]] if len(cells) > cols["Valor"] else "")[0]
                if amt is not None:
                    out.stated[(month, "paid")] = out.stated.get((month, "paid"), 0.0) + amt
                continue
            get = lambda name: cells[cols[name]] if len(cells) > cols[name] else ""  # noqa: E731
            amt, _ = to_amount(get("Valor"))
            if amt is None:
                raise AssertionError(f"{path.name} {sheet} row {rn}: no amount in {cells}")
            out.lines.append({
                "period": month, "period_part": "",
                "project": _text(get("Nome Projeto")),
                "beneficiary": _text(get("Razão Social")),
                "beneficiary_cnpj": _text(get("CNPJ Fornecedor")),
                "amount_brl": amt, "amount_kind": "paid",
                "amount_label": "Valores Pagos / Valor",
                "source_file": path.name, "source_url": url, "parse_method": method,
                "sheet_or_page": f"{sheet}!R{rn}"})
    if used != 1:
        raise AssertionError(f"{path.name}: {used} sheets carry 'Valores Pagos'")
    return out


def _pdf_runs(chars, gap: float = 30.0):
    """Text runs in content-stream order, blanks dropped; a run ends when the line changes, the
    pen moves left, or a gap wider than `gap` points opens. BB's PDFs write each amount's digits before
    its 'R$', and long names overprint the amount column, so stream order separates them."""
    cur, out = [], []
    for c in (ch for ch in chars if ch["text"].strip()):
        if cur:
            p = cur[-1]
            if abs(c["top"] - p["top"]) > 1.5 or c["x0"] < p["x0"] - 0.5 or c["x0"] - p["x1"] > gap:
                out.append(cur)
                cur = []
        cur.append(c)
    if cur:
        out.append(cur)
    return [("".join(ch["text"] for ch in r), r[0]["x0"], r[0]["top"]) for r in out]


def parse_bb_pdf(path: Path, url: str) -> Parsed:
    out = Parsed(kinds={"paid"})
    month, valor_x = None, None
    with pdfplumber.open(path) as pdf:
        for pn, page in enumerate(pdf.pages, start=1):
            events = []
            runs = _pdf_runs(page.chars)
            for text, x0, top in runs:
                if text == "Valor":
                    valor_x = x0
            for text, x0, top in runs:
                if MONTH_RE.match(text):
                    mm, yy = MONTH_RE.match(text).groups()
                    events.append((top, 0, yy + mm))
            # Amounts: runs rebuilt from the characters in the Valor column alone, so a name
            # that overprints the column cannot join an amount's digits; a name that runs into
            # the digits is cut at the last non-digit (a digit there drops the amount, which the
            # 'R$' count below reports).
            zone = (_pdf_runs([c for c in page.chars if c["x0"] >= valor_x - 40], gap=6.0)
                    if valor_x is not None else [])
            n_rs = sum(1 for t, _, _ in zone if "R$" in t)
            for text, x0, top in zone:
                m = PDF_AMOUNT_RE.search(text)
                if m:
                    events.append((top, 1, float(m.group(1).replace(".", "").replace(",", "."))))
                elif re.search(r"(?i)total", text):
                    out.n_total_rows += 1
            n_amt = sum(1 for e in events if e[1] == 1)
            if n_rs != n_amt:
                out.notes.append(f"page {pn}: {n_rs} 'R$' marks but {n_amt} amounts parsed")
            for top, kind, v in sorted(events, key=lambda e: (e[0], e[1])):
                if kind == 0:
                    month = v
                    out.content_periods.append(v)
                elif month is None:
                    raise AssertionError(f"{path.name} page {pn}: amount before any month")
                else:
                    out.lines.append({"period": month, "period_part": "", "amount_brl": v,
                                      "amount_kind": "paid", "source_file": path.name,
                                      "source_url": url, "parse_method": "pdf-pdfplumber",
                                      "sheet_or_page": f"page {pn}"})
    return out


def _month_summary(p: Parsed) -> pd.DataFrame:
    if not p.lines:
        return pd.DataFrame(columns=["n", "sum"])
    df = pd.DataFrame(p.lines)
    return df.groupby("period")["amount_brl"].agg(n="count", sum="sum")


def _months(start: str, end: str) -> list[str]:
    return [m.strftime("%Y%m") for m in pd.period_range(pd.Period(start, "M"),
                                                        pd.Period(end, "M"), freq="M")]


def collect_bb(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    man = bb_manifest(refresh)
    BB_RAW.mkdir(parents=True, exist_ok=True)
    parsed: dict[str, Parsed | None] = {}
    errors: dict[str, str] = {}
    for r in man.itertuples():
        dest = BB_RAW / r.file
        if not fetch_file(r.url, dest, refresh):
            parsed[r.file] = None
            errors[r.file] = "missing_file"
            continue
        try:
            parsed[r.file] = (parse_bb_pdf(dest, r.url) if r.fmt == "pdf"
                              else parse_bb_spreadsheet(dest, r.fmt, r.url))
        except Exception as e:  # noqa: BLE001
            parsed[r.file] = None
            errors[r.file] = f"unreadable: {type(e).__name__}: {e}"
            log.warning("  %s unreadable: %s", r.file, e)

    # Content period of every readable file, against its filing.
    man["content_start"] = man["file"].map(lambda f: min(parsed[f].content_periods)
                                           if parsed.get(f) and parsed[f].content_periods else None)
    man["content_end"] = man["file"].map(lambda f: max(parsed[f].content_periods)
                                         if parsed.get(f) and parsed[f].content_periods else None)
    man["conflict"] = (man["content_start"].notna()
                       & ((man["content_start"] != man["filed_start"])
                          | (man["content_end"] != man["filed_end"])))
    anomalies = []
    for r in man[man["conflict"]].itertuples():
        msg = (f"BB {r.file} is filed under {r.filed_start}-{r.filed_end} but contains "
               f"{r.content_start}-{r.content_end}; not used for its filed period")
        twin = man[(man["filed_start"] == r.content_start) & (man["filed_end"] == r.content_end)
                   & ~man["conflict"] & (man["fmt"] == r.fmt)]
        if len(twin):
            a, b = parsed[r.file], parsed[twin.iloc[0]["file"]]
            same = (sorted((l["period"], round(l["amount_brl"], 2)) for l in a.lines)
                    == sorted((l["period"], round(l["amount_brl"], 2)) for l in b.lines))
            msg += (f"; its content {'is identical to' if same else 'DIFFERS from'} "
                    f"{twin.iloc[0]['file']}, filed under that period")
        log.warning(msg)
        anomalies.append(msg)

    lines, periods = [], []
    for (fs, fe), grp in man.groupby(["filed_start", "filed_end"]):
        order = {"xlsx": 0, "ods": 1, "pdf": 2}
        grp = grp.assign(o=grp["fmt"].map(order)).sort_values("o")
        usable = grp[grp["file"].map(lambda f: parsed.get(f) is not None) & ~grp["conflict"]]
        months = _months(fs, fe)
        if usable.empty:
            status = ("missing_file" if all(errors.get(f) == "missing_file" for f in grp["file"])
                      else "unreadable")
            for m in months:
                periods.append({"bank_key": "bb", "period": m, "period_part": "",
                                "amount_kind": "paid", "status": status,
                                "filed_period": f"{fs}-{fe}", "content_period": None,
                                "period_conflict": bool(grp["conflict"].any()),
                                "formats_parsed": "", "formats_agree": pd.NA,
                                "source_file": "; ".join(grp["file"]), "source_url": ""})
            continue
        src = usable.iloc[0]
        src_p = parsed[src["file"]]
        ref = _month_summary(src_p)
        agree, checked = True, []
        pdf_missing: dict[str, int] = {}
        for r in usable.iloc[1:].itertuples():
            if r.fmt == "pdf" and src["fmt"] != "pdf":
                # A PDF may drop rows (see anomalies) but must not show an amount the
                # spreadsheet lacks.
                checked.append(r.fmt)
                for m in months:
                    a = Counter(round(x["amount_brl"], 2) for x in src_p.lines if x["period"] == m)
                    b = Counter(round(x["amount_brl"], 2) for x in parsed[r.file].lines
                                if x["period"] == m)
                    if b - a:
                        agree = False
                        log.warning("BB %s %s: PDF amounts absent from %s: %s", m, r.file,
                                    src["file"], dict(b - a))
                    if a - b:
                        pdf_missing[m] = sum((a - b).values())
                if pdf_missing:
                    msg = (f"BB {r.file}: the PDF lacks {sum(pdf_missing.values())} of "
                           f"{len(src_p.lines)} amounts in {src['file']} (by month "
                           f"{pdf_missing}); every PDF amount is in the spreadsheet")
                    anomalies.append(msg)
                continue
            other = _month_summary(parsed[r.file])
            j = ref.join(other, how="outer", lsuffix="_src", rsuffix="_oth").fillna(0)
            bad = j[(j["n_src"] != j["n_oth"]) | ((j["sum_src"] - j["sum_oth"]).abs() > TOL_BRL)]
            checked.append(r.fmt)
            if len(bad):
                agree = False
                log.warning("BB %s-%s: %s disagrees with %s:\n%s", fs, fe, r.file, src["file"],
                            bad.to_string())
        for r in grp[grp["file"].map(lambda f: parsed.get(f) is None)].itertuples():
            anomalies.append(f"BB {r.file}: {errors[r.file]}")
        for sheet, tot in src_p.ignored_sums:
            hit = [m for m in months if len(ref) and abs(ref["sum"].get(m, np.nan) - tot) <= TOL_BRL]
            src_p.notes.append(f"ignored sheet {sheet!r} sums to the main sheet's month {hit[0]}"
                               if hit else f"ignored sheet {sheet!r} matches no month sum")
        for r in usable.itertuples():
            if r.fmt == "pdf":
                for note in parsed[r.file].notes:
                    anomalies.append(f"BB {r.file}: {note}")
        for note in src_p.notes:
            msg = f"BB {src['file']}: {note}"
            log.info("  %s", msg)
            anomalies.append(msg)
        lines.extend(src_p.lines)
        for m in months:
            present = m in src_p.content_periods
            n = int(ref["n"].get(m, 0)) if len(ref) else 0
            periods.append({
                "bank_key": "bb", "period": m, "period_part": "", "amount_kind": "paid",
                "amount_brl": float(ref["sum"].get(m, 0.0)) if len(ref) else 0.0,
                "stated_total_brl": src_p.stated.get((m, "paid"), np.nan),
                "n_lines": n,
                "n_total_rows_excluded": 0,
                "status": "parsed" if present else "missing_month",
                "filed_period": f"{fs}-{fe}",
                "content_period": f"{src['content_start']}-{src['content_end']}",
                "period_conflict": bool(grp["conflict"].any()),
                "formats_parsed": ",".join([src["fmt"]] + checked),
                "formats_agree": agree if checked else pd.NA,
                "pdf_missing_lines": pdf_missing.get(m, 0) if "pdf" in checked else pd.NA,
                "source_file": src["file"], "source_url": src["url"]})
        # Total rows are counted per file; attribute them to the file's first month.
        if src_p.n_total_rows:
            periods[-len(months)]["n_total_rows_excluded"] = src_p.n_total_rows
    return pd.DataFrame(lines), pd.DataFrame(periods), anomalies


# ---------------------------------------------------------------------------
# Banco do Nordeste
# ---------------------------------------------------------------------------
def bnb_manifest(refresh: bool) -> tuple[pd.DataFrame, str]:
    page = fetch_text(BNB_PAGE, BNB_RAW / "_index_page.html", refresh)
    set_id = None
    for m in re.finditer(r'<script id="data_(\w{4})" type="application/json">(.*?)</script>', page, re.S):
        if json.loads(m.group(2)).get("title", "").strip() == BNB_FRAGMENT_TITLE:
            cfg = re.search(r"#fragment-\d+-" + m.group(1) + r"'\);\s*var configuration = (\{.*?\});",
                            page, re.S)
            if cfg:
                set_id = json.loads(cfg.group(1))["contentSetId"]
    if not set_id:
        raise SystemExit(f"no '{BNB_FRAGMENT_TITLE}' content set on {BNB_PAGE}")
    api = (f"{BNB_SITE}/o/headless-delivery/v1.0/content-sets/{set_id}/"
           f"content-set-elements?pageSize=500&page=1")
    body = fetch_text(api, BNB_RAW / f"_content_set_{set_id}.json", refresh)
    j = json.loads(body)
    if j.get("lastPage", 1) != 1:
        raise AssertionError(f"content set {set_id} has more than one page")
    rows = []
    for it in j["items"]:
        c = it["content"]
        ano = [f["contentFieldValue"].get("data") for f in c["documentType"]["contentFields"]
               if f.get("name") == "Ano"]
        url = BNB_SITE + c["contentUrl"]
        rows.append({"title": it["title"], "api_year": ano[0] if ano else None,
                     "fmt": c.get("fileExtension", "").lower(), "url": url,
                     "file": urllib.parse.unquote_plus(c["contentUrl"].split("/")[4])})
    man = pd.DataFrame(rows)
    log.info("BNB content set %s: %d files, years %s to %s", set_id, len(man),
             man["api_year"].min(), man["api_year"].max())
    return man, set_id


def _bnb_kind(label: str) -> str | None:
    s = _deaccent(label).upper()
    if not s.startswith("VALOR"):
        return None
    if "DESEMBOLS" in s or "LIQUIDADO" in s:
        return "disbursed"
    if "APROVADO" in s or "PATROCINIO" in s:
        return "contracted"
    raise AssertionError(f"unknown BNB amount column {label!r}")


def _col_index(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + ord(ch) - 64
    return n - 1


def recompute_sum(formula: str, rows: list[tuple[int, list]]) -> float | None:
    """Re-evaluate a single-column 'of:=SUM([.E14:.E212])' over the sheet's cells; None for any
    other formula."""
    m = re.fullmatch(r"of:=SUM\(\[\.([A-Z]+)(\d+):\.([A-Z]+)(\d+)\]\)", formula.strip())
    if not m or m.group(1) != m.group(3):
        return None
    col, r0, r1 = _col_index(m.group(1)), int(m.group(2)), int(m.group(4))
    total = 0.0
    for rn, cells in rows:
        if r0 <= rn <= r1 and len(cells) > col:
            a = to_amount(cells[col])[0]
            total += a or 0.0
    return total


def parse_bnb(path: Path, url: str, part: str) -> Parsed:
    formulas: dict = {}
    sheets = read_ods(path, formulas)
    out = Parsed()
    used = 0
    for sheet, rows in sheets.items():
        headers = [i for i, (_, c) in enumerate(rows)
                   if c and _text(c[0]).upper() == "UF" and "PROJETO" in [_text(x).upper() for x in c]]
        if not headers:
            if rows:
                out.notes.append(f"sheet {sheet!r} ignored ({len(rows)} rows, no UF/PROJETO "
                                 f"header; first cells {[_text(x)[:40] for x in rows[0][1]][:4]})")
            continue
        if len(headers) > 1:
            raise AssertionError(f"{path.name} {sheet}: {len(headers)} header rows")
        used += 1
        h = headers[0]
        preamble = " ".join(_text(x) for _, c in rows[:h] for x in c)
        years = set(re.findall(r"20\d{2}", preamble))
        out.content_periods.extend(sorted(years))
        dpre = _deaccent(preamble).lower()
        if re.search(r"jul\w*\s+a\s+dez", dpre):
            out.notes.append("content half H2")
            out.content_periods.append("H2")
        if re.search(r"jan\w*\s+a\s+jun", dpre):
            out.content_periods.append("H1")
        hdr = [_text(x) for x in rows[h][1]]
        up = [_deaccent(x).upper() for x in hdr]
        c_proj = up.index("PROJETO")
        c_ben = next(i for i, x in enumerate(up) if x in ("PROPONENTE", "NOME DO PATROCINADO"))
        amount_cols = {i: _bnb_kind(x) for i, x in enumerate(hdr) if _bnb_kind(x)}
        if not amount_cols:
            raise AssertionError(f"{path.name}: no amount column in {hdr}")
        out.kinds.update(amount_cols.values())
        body = rows[h + 1:]
        k = 0
        while k < len(body):
            rn, cells = body[k]
            get = lambda i: cells[i] if len(cells) > i else ""  # noqa: E731
            label_cells = [_text(get(i)) for i in range(min(len(cells), 12)) if i not in amount_cols]
            is_total = any(re.match(r"(?i)^(valor\s+)?total\b", x) for x in label_cells if x)
            if is_total and not _text(get(c_proj)):
                out.n_total_rows += 1
                vals = {i: to_amount(get(i))[0] for i in amount_cols}
                if all(v is None for v in vals.values()) and k + 1 < len(body):
                    nxt = [x for x in body[k + 1][1] if _text(x)]
                    lab = next(x for x in label_cells if x)
                    if len(nxt) == 1 and to_amount(nxt[0])[0] is not None:
                        out.n_total_rows += 1
                        out.notes.append(f"secondary total {lab!r} = {to_amount(nxt[0])[0]:,.2f}")
                        out.stated.setdefault((part, "secondary"), to_amount(nxt[0])[0])
                        k += 2
                        continue
                for i, v in vals.items():
                    if v is not None:
                        key = (part, amount_cols[i])
                        if key in out.stated:
                            raise AssertionError(f"{path.name}: two totals for {amount_cols[i]}")
                        out.stated[key] = v
                        f = formulas.get((sheet, rn, i))
                        again = recompute_sum(f, rows) if f else None
                        if again is not None and abs(again - v) > TOL_BRL:
                            out.formula_totals[key] = again
                            out.notes.append(
                                f"row {rn}: stated {amount_cols[i]} total {v:,.2f} is the "
                                f"stored result of {f}, which evaluates to {again:,.2f} "
                                f"on the published cells (stale formula result)")
                k += 1
                continue
            amts = {i: to_amount(get(i)) for i in amount_cols}
            if all(a is None for a, _ in amts.values()):
                out.notes.append(f"row {rn} skipped, no amount: {[_text(x)[:50] for x in cells if _text(x)][:3]}")
                k += 1
                continue
            if not (_text(get(c_proj)) or _text(get(c_ben))):
                raise AssertionError(f"{path.name} row {rn}: amount without project or beneficiary")
            if is_total:
                out.notes.append(f"row {rn}: 'total' in a project row kept as data")
            ben = _text(get(c_ben))
            cnpj = CNPJ_RE.search(ben)
            for i, (a, from_text) in amts.items():
                if a is None:
                    continue
                out.lines.append({
                    "period": "", "period_part": part,
                    "project": _text(get(c_proj)), "beneficiary": ben,
                    "beneficiary_cnpj": cnpj.group(0) if cnpj else "",
                    "amount_brl": a, "amount_kind": amount_cols[i], "amount_label": hdr[i],
                    "source_file": path.name, "source_url": url,
                    "parse_method": "ods-lxml" + ("-text" if from_text else ""),
                    "sheet_or_page": f"{sheet}!R{rn}"})
            k += 1
    if used != 1:
        raise AssertionError(f"{path.name}: {used} sheets with a UF/PROJETO header")
    return out


def collect_bnb(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    man, set_id = bnb_manifest(refresh)
    BNB_RAW.mkdir(parents=True, exist_ok=True)
    lines, periods, anomalies = [], [], []
    seen = {}
    for r in man.sort_values("api_year").itertuples():
        dt = _deaccent(r.title).lower()
        part = "H1" if "primeiro semestre" in dt else ("H2" if "segundo semestre" in dt else "")
        title_years = set(re.findall(r"20\d{2}", r.title))
        filed = str(r.api_year)
        if title_years != {filed}:
            raise AssertionError(f"BNB {r.title}: API year {filed} vs title years {title_years}")
        key = (filed, part)
        if key in seen:
            raise AssertionError(f"BNB two files for {key}: {seen[key]} and {r.file}")
        seen[key] = r.file
        dest = BNB_RAW / r.file
        base = {"bank_key": "bnb", "period": filed, "period_part": part,
                "filed_period": filed + part, "source_file": r.file, "source_url": r.url,
                "formats_parsed": r.fmt, "formats_agree": pd.NA}
        if not fetch_file(r.url, dest, refresh):
            periods.append({**base, "amount_kind": None, "status": "missing_file"})
            continue
        try:
            p = parse_bnb(dest, r.url, part)
        except Exception as e:  # noqa: BLE001
            log.warning("  %s unreadable: %s", r.file, e)
            anomalies.append(f"BNB {r.file}: unreadable: {e}")
            periods.append({**base, "amount_kind": None, "status": "unreadable"})
            continue
        cyears = {x for x in p.content_periods if x.startswith("20")}
        chalf = {x for x in p.content_periods if x in ("H1", "H2")}
        conflict = cyears != {filed} or (bool(chalf) and chalf != {part})
        content = ",".join(sorted(cyears)) + ",".join(sorted(chalf))
        if conflict:
            msg = f"BNB {r.file}: filed under {filed}{part}, content says {content}"
            log.warning(msg)
            anomalies.append(msg)
        if part and not chalf:
            anomalies.append(f"BNB {r.file}: half {part} from the title only; content names "
                             f"the year alone")
        for note in p.notes:
            msg = f"BNB {r.file}: {note}"
            log.info("  %s", msg)
            if not note.startswith("content half"):
                anomalies.append(msg)
        df = pd.DataFrame(p.lines)
        df["period"] = filed
        lines.append(df)
        second = p.stated.get((part, "secondary"))
        for kind in sorted(p.kinds):
            sub = df[df["amount_kind"] == kind]
            stated = p.stated.get((part, kind), np.nan)
            if second is not None and kind == "disbursed" and not np.isnan(stated) \
                    and abs(second - stated) > TOL_BRL:
                anomalies.append(f"BNB {r.file}: secondary total {second:,.2f} differs from "
                                 f"the disbursed total {stated:,.2f}")
            periods.append({**base, "amount_kind": kind, "amount_brl": float(sub["amount_brl"].sum()),
                            "stated_total_brl": stated,
                            "stated_formula_recomputed_brl": p.formula_totals.get((part, kind), np.nan),
                            "n_lines": len(sub),
                            "n_total_rows_excluded": p.n_total_rows,
                            "status": "parsed",
                            "content_period": content, "period_conflict": conflict,
                            "labels": "; ".join(sorted(sub["amount_label"].unique()))})
    lines_df = pd.concat(lines, ignore_index=True) if lines else pd.DataFrame()
    per = pd.DataFrame(periods)

    # Expected: every year from first to last; both halves for a year filed by semester.
    years = sorted(int(y) for y in man["api_year"])
    for y in range(years[0], years[-1] + 1):
        parts = set(per.loc[per["period"] == str(y), "period_part"])
        expected = {"H1", "H2"} if parts & {"H1", "H2"} else {""}
        for pp in sorted(expected - parts):
            per = pd.concat([per, pd.DataFrame([{"bank_key": "bnb", "period": str(y),
                                                 "period_part": pp, "amount_kind": None,
                                                 "status": "missing_file"}])],
                            ignore_index=True)
    return lines_df, per, anomalies


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def panel_keys() -> dict[str, str]:
    frames = []
    for f in sorted(paths.IF_DATA_LIST.glob("IF_DATA_List_*.csv")):
        frames.append(pd.read_csv(f, dtype=str, usecols=["CodInst", "NomeInstituicao",
                                                         "CodConglomeradoPrudencial",
                                                         "CnpjInstituicaoLider"]))
    ifd = pd.concat(frames, ignore_index=True)
    cosif = pd.read_parquet(paths.AWARENESS_PROC / "cosif_advertising_monthly.parquet",
                            columns=["level", "entity_key", "entity_name", "cnpj_leader"])
    cosif = cosif[cosif["level"] == "conglomerate"].drop_duplicates("entity_key")
    keys = {}
    for bank, info in BANKS.items():
        inst = ifd[ifd["CodInst"] == info["cnpj8"]]
        codes = set(inst["CodConglomeradoPrudencial"].dropna()) - {"null"}
        if len(codes) != 1:
            raise AssertionError(f"{bank}: prudential codes for CNPJ8 {info['cnpj8']}: {codes}")
        code = codes.pop()
        leaders = set(ifd.loc[ifd["CodInst"] == code, "CnpjInstituicaoLider"].dropna()) - {"null"}
        if leaders != {info["cnpj8"]}:
            raise AssertionError(f"{bank}: IF.data leader of {code} is {leaders}")
        row = cosif[cosif["entity_key"] == code]
        if len(row) != 1 or row.iloc[0]["cnpj_leader"] != info["cnpj8"] \
                or info["name_token"] not in str(row.iloc[0]["entity_name"]).upper():
            raise AssertionError(f"{bank}: COSIF conglomerate {code} does not confirm leader "
                                 f"{info['cnpj8']}: {row.to_dict('records')}")
        log.info("panel_key %s: %s (%s), leader %s, %d IF.data lists without a code",
                 bank, code, row.iloc[0]["entity_name"], info["cnpj8"],
                 int((inst["CodConglomeradoPrudencial"].isna()
                      | (inst["CodConglomeradoPrudencial"] == "null")).sum()))
        keys[bank] = code
    return keys


# ---------------------------------------------------------------------------
# Validation and reporting
# ---------------------------------------------------------------------------
def validate(lines: pd.DataFrame, periods: pd.DataFrame, bb_raw_periods: pd.DataFrame) -> None:
    # 1. Every expected period has a status.
    bb_first = bb_raw_periods["filed_period"].str[:6].min()
    bb_last = bb_raw_periods["filed_period"].str[-6:].max()
    expected = set(_months(bb_first, bb_last))
    have = set(periods.loc[periods["bank_key"] == "bb", "period"])
    if expected != have:
        raise AssertionError(f"BB months without a status: {sorted(expected - have)}; "
                             f"unexpected: {sorted(have - expected)}")
    if periods["status"].isna().any():
        raise AssertionError("period rows without a status")
    odd = periods[~periods["status"].isin(["parsed", "unreadable", "missing_file"])]
    if len(odd):
        raise AssertionError(f"periods with status outside parsed/unreadable/missing_file:\n"
                             f"{odd[['bank_key', 'period', 'status']].to_string()}")
    bnb = periods[periods["bank_key"] == "bnb"]
    years = sorted(int(y) for y in bnb["period"].unique())
    if set(years) != set(range(years[0], years[-1] + 1)):
        raise AssertionError(f"BNB years without a status: "
                             f"{sorted(set(range(years[0], years[-1] + 1)) - set(years))}")

    # 2. Stated totals.
    st = periods[periods["stated_total_brl"].notna()]
    share = float(st["total_matches"].mean()) if len(st) else float("nan")
    stale = (~st["total_matches"]) & ((st["amount_brl"] - st["stated_formula_recomputed_brl"]).abs()
                                      <= TOL_BRL)
    explained = st["total_matches"] | stale
    log.info("stated totals: %d of %d period-kinds match within R$%.0f (%.1f%%); counting stored "
             "formula results that re-evaluate to the line sum on the published cells: %.1f%%",
             int(st["total_matches"].sum()), len(st), TOL_BRL, 100 * share,
             100 * float(explained.mean()))
    cols = ["bank_key", "period", "period_part", "amount_kind", "amount_brl", "stated_total_brl",
            "stated_formula_recomputed_brl"]
    if (~st["total_matches"]).any():
        log.warning("stated totals not matched:\n%s", st[~st["total_matches"]][cols].to_string())
    if len(st) and float(explained.mean()) < MIN_TOTAL_MATCH_SHARE:
        raise AssertionError("lines do not sum to the stated total:\n" +
                             st[~explained][cols].to_string())

    # 3. Filed periods covered despite conflicts (a conflict leaves a row unreadable/missing).
    bad = periods[periods["period_conflict"].eq(True) & (periods["status"] != "parsed")]
    if len(bad):
        raise AssertionError(f"filed periods left uncovered by a period conflict:\n{bad.to_string()}")

    # 4. BB formats agree.
    dis = periods[(periods["bank_key"] == "bb") & periods["formats_agree"].eq(False)]
    if len(dis):
        raise AssertionError(f"BB formats disagree for {sorted(dis['filed_period'].unique())}")
    nchk = periods.loc[(periods["bank_key"] == "bb") & periods["formats_agree"].eq(True),
                       "filed_period"].nunique()
    log.info("BB formats agree for %d filed periods", nchk)

    # 5. Non-negative amounts.
    if (lines["amount_brl"] < 0).any():
        raise AssertionError(f"negative amounts:\n{lines[lines['amount_brl'] < 0].head().to_string()}")


def report(lines: pd.DataFrame, periods: pd.DataFrame) -> None:
    p = periods[periods["status"] == "parsed"].copy()
    p["year"] = p["period"].str[:4] + p["period_part"].fillna("")
    yearly = (p.groupby(["bank_key", "year", "amount_kind"])
               .agg(brl_m=("amount_brl", "sum"), n_lines=("n_lines", "sum"),
                    n_periods=("period", "nunique"))
               .reset_index())
    yearly["brl_m"] = (yearly["brl_m"] / 1e6).round(2)
    log.info("yearly sponsorship totals (R$ million; n_periods = months for bb):\n%s",
             yearly.pivot_table(index="year", columns=["bank_key", "amount_kind"],
                                values="brl_m").to_string())
    log.info("coverage (periods parsed per year):\n%s",
             yearly.pivot_table(index="year", columns=["bank_key", "amount_kind"],
                                values="n_periods").to_string())

    cvm_path = paths.AWARENESS_PROC / "cvm_advertising_quarterly.parquet"
    if not cvm_path.exists():
        log.warning("no %s; BB ratio not printed", cvm_path.name)
        return
    cvm = pd.read_parquet(cvm_path)
    sel = cvm[(cvm["CD_CVM"] == "001023") & (cvm["scope"] == "ind") & (cvm["statement"] == "DVA")]
    adv = (sel.groupby(["line_id", "year"])["flow_first"]
              .agg(adv_brl=lambda s: s.sum(min_count=1), n_quarters="count").reset_index())
    bb = yearly[(yearly["bank_key"] == "bb") & (yearly["amount_kind"] == "paid")].copy()
    bb["year"] = bb["year"].astype(int)
    j = bb.merge(adv, on="year", how="left")
    j["adv_brl_m"] = (j["adv_brl"] / 1e6).round(1)
    full = (j["n_periods"] == 12) & (j["n_quarters"] == 4)
    j["ratio"] = (j["brl_m"] / j["adv_brl_m"]).where(full).round(3)    # full years only
    log.info("BB paid sponsorship / BB individual DVA advertising line (flow_first):\n%s",
             j[["year", "line_id", "n_periods", "brl_m", "adv_brl_m", "n_quarters",
                "ratio"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
LINE_COLS = ["bank_key", "cnpj8", "panel_key", "period", "period_part", "project", "beneficiary",
             "beneficiary_cnpj", "amount_brl", "amount_kind", "amount_label", "source_file",
             "source_url", "parse_method", "sheet_or_page", "is_total_row"]
PERIOD_COLS = ["bank_key", "cnpj8", "panel_key", "period", "period_part", "amount_kind",
               "amount_brl", "stated_total_brl", "stated_formula_recomputed_brl", "total_diff_brl",
               "total_matches", "n_lines",
               "n_total_rows_excluded", "status", "filed_period", "content_period",
               "period_conflict", "formats_parsed", "formats_agree", "pdf_missing_lines",
               "amount_labels",
               "source_file", "source_url"]


def main() -> None:
    ap = argparse.ArgumentParser(description="BB and BNB published sponsorship spend")
    ap.add_argument("--refresh", action="store_true", help="refetch index pages and files")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()

    keys = panel_keys()
    log.info("Banco do Brasil")
    bb_lines, bb_periods, bb_anom = collect_bb(a.refresh)
    log.info("Banco do Nordeste")
    bnb_lines, bnb_periods, bnb_anom = collect_bnb(a.refresh)

    lines = pd.concat([bb_lines.assign(bank_key="bb"), bnb_lines.assign(bank_key="bnb")],
                      ignore_index=True)
    lines["cnpj8"] = lines["bank_key"].map(lambda b: BANKS[b]["cnpj8"])
    lines["panel_key"] = lines["bank_key"].map(keys)
    lines["is_total_row"] = False
    lines = lines[LINE_COLS]

    periods = pd.concat([bb_periods, bnb_periods.rename(columns={"labels": "amount_labels"})],
                        ignore_index=True)
    periods["cnpj8"] = periods["bank_key"].map(lambda b: BANKS[b]["cnpj8"])
    periods["panel_key"] = periods["bank_key"].map(keys)
    periods["total_diff_brl"] = periods["amount_brl"] - periods["stated_total_brl"]
    periods["total_matches"] = np.where(periods["stated_total_brl"].notna(),
                                        periods["total_diff_brl"].abs() <= TOL_BRL, pd.NA)
    periods.loc[periods["stated_total_brl"].isna(), "total_matches"] = pd.NA
    periods["total_matches"] = periods["total_matches"].astype("boolean")
    periods["formats_agree"] = periods["formats_agree"].astype("boolean")
    periods["period_conflict"] = periods["period_conflict"].astype("boolean")
    for c in PERIOD_COLS:
        if c not in periods.columns:
            periods[c] = pd.NA
    periods = periods[PERIOD_COLS].sort_values(["bank_key", "period", "period_part",
                                                "amount_kind"]).reset_index(drop=True)

    missing_month = periods[periods["status"] == "missing_month"]
    if len(missing_month):
        raise AssertionError("BB months inside a parsed file without a month block:\n"
                             + missing_month[["period", "source_file"]].to_string())
    validate(lines, periods, bb_periods)

    anomalies = bb_anom + bnb_anom
    if anomalies:
        log.warning("anomalies (%d):\n  %s", len(anomalies), "\n  ".join(anomalies))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("sponsorship_lines", lines), ("sponsorship_periods", periods)):
        frame.to_parquet(a.out_dir / f"{name}.parquet", index=False)
        frame.to_csv(a.out_dir / f"{name}.csv", index=False, lineterminator="\n")
        log.info("%s: %d rows -> %s", name, len(frame), a.out_dir / f"{name}.parquet")
    report(lines, periods)


if __name__ == "__main__":
    main()
