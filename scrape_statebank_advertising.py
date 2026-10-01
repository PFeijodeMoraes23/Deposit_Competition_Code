"""
scrape_statebank_advertising.py
===============================
(docstring completed below once the parsers are in place)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths
from utils.disclosure_common import http_get, now_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

RAW_DIR = paths.AWARENESS_RAW
OUT_DIR = paths.AWARENESS_PROC

BANKS = ("banestes", "basa", "banrisul", "brb")

URL_BANESTES_LIST = "https://www.banestes.com.br/acessoinformacao/ajax/contrato-publicidade.html"
URL_BANESTES_BASE = "https://www.banestes.com.br"
URL_BASA_LIST = "https://www.bancoamazonia.com.br/acesso-informacao/licitacoes-contratos"
URL_BANRISUL_LIST = "https://www.banrisul.com.br/bob/site/link/servicos-marketing.html"
URL_BANRISUL_DETAIL = "https://www.banrisul.com.br/bob/site/link/servico-marketing-detalhe_{id}.html"
URL_BANRISUL_MEDIA = "https://www.banrisul.com.br/bob/site/link/midias/"
URL_BRB_LIST = "https://novo.brb.com.br/sobre-o-brb/transparencia-publicidade-e-propaganda/"

MONTHS_PT = {"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
             "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11,
             "dezembro": 12}
MONTH_ABBR_PT = {"jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6, "jul": 7,
                 "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def fold(text: str) -> str:
    """Upper-case ASCII fold: accents removed, whitespace collapsed."""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip().upper()


def month_number(label: str) -> int | None:
    t = fold(label).lower()
    for name, n in MONTHS_PT.items():
        if t.startswith(name):
            return n
    return MONTH_ABBR_PT.get(t[:3])


def brl(token: str) -> float | None:
    """Brazilian-format amount ('1.234,56', 'R$ -', '-R$ 95,00', '(1.234,56)') -> float.
    A bare dash is a published zero; returns None when the token carries no amount."""
    s = str(token).replace("\xa0", " ").strip()
    if not s:
        return None
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = re.sub(r"[R$\s()]", "", s).lstrip("-")
    if s in ("", "-"):
        return 0.0 if "-" in token or "$" in token else None
    if not re.fullmatch(r"\d{1,3}(\.\d{3})*,\d{2}|\d+,\d{2}|\d+", s):
        return None
    v = float(s.replace(".", "").replace(",", "."))
    return -v if neg else v


def safe_name(url: str) -> str:
    """Short, ASCII, unique local file name for a URL."""
    base = urllib.parse.unquote(urllib.parse.urlparse(url).path.rstrip("/").split("/")[-1])
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode()
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "index"
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, "bin"
    digest = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"{stem[:60]}_{digest}.{ext.lower()[:5]}"


def quote_url(url: str) -> str:
    return urllib.parse.quote(url, safe=":/?=&%#~+,;@")


# ---------------------------------------------------------------------------
# Download cache
# ---------------------------------------------------------------------------
class Cache:
    """One manifest per bank under RAW_DIR/<bank>/manifest.json.

    Every URL requested is recorded with its HTTP status, the local file, its size, its
    SHA-256 and the kind of body received (pdf, xlsx, xls, csv, html, other). A URL already in
    the manifest is never requested again unless --refresh is given, whether it succeeded or
    not, so a rerun works from disk alone.
    """

    def __init__(self, bank: str, refresh: bool):
        self.dir = RAW_DIR / bank
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "manifest.json"
        self.refresh = refresh
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.write_text(json.dumps(self.entries, indent=1, ensure_ascii=False),
                             encoding="utf-8", newline="\n")

    @staticmethod
    def body_kind(body: bytes) -> str:
        head = body[:8]
        if head.startswith(b"%PDF"):
            return "pdf"
        if head.startswith(b"PK"):
            return "xlsx"
        if head.startswith(b"\xd0\xcf\x11\xe0"):
            return "xls"
        low = body[:600].lower()
        if b"<html" in low or b"<!doctype html" in low or b"<head" in low:
            return "html"
        return "text"

    def get(self, url: str, name: str | None = None) -> dict:
        entry = self.entries.get(url)
        if entry and not self.refresh and (entry.get("file") is None
                                           or (self.dir / entry["file"]).exists()):
            return entry
        status, body = http_get(quote_url(url), binary=True, timeout=90, retries=3)
        entry = {"url": url, "status": status, "file": None, "bytes": 0, "sha256": None,
                 "kind": None, "retrieved_at": now_iso()}
        if body:
            fname = name or safe_name(url)
            (self.dir / fname).write_bytes(body)
            entry.update(file=fname, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                         kind=self.body_kind(body))
        self.entries[url] = entry
        self.save()
        log.info("  GET %-4s %-5s %8d  %s", status, entry["kind"], entry["bytes"], url)
        return entry

    def file(self, entry: dict) -> Path | None:
        return self.dir / entry["file"] if entry.get("file") else None


# ---------------------------------------------------------------------------
# Discovery: what each bank's disclosure page lists
# ---------------------------------------------------------------------------
@dataclass
class Doc:
    """One file listed on a bank's disclosure page."""
    bank: str
    url: str
    role: str                      # values | supplier_names
    year: int | None = None
    month: int | None = None       # month named on the listing, when it names one
    quarter: int | None = None
    entity: str = "bank"
    agency: str | None = None
    label: str = ""
    entry: dict = field(default_factory=dict)


def html_text(cache: Cache, url: str, encoding: str = "utf-8") -> str:
    entry = cache.get(url, name="listing_" + safe_name(url))
    if entry.get("kind") != "html":
        raise SystemExit(f"listing page {url} did not return HTML (status {entry['status']})")
    return cache.file(entry).read_bytes().decode(encoding, "replace")


def discover_banestes(cache: Cache) -> list[Doc]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_text(cache, URL_BANESTES_LIST), "lxml")
    table = soup.find(id="contratos-publicidade")
    docs = []
    for tr in table.find("tbody").find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        a = tr.find("a", href=True)
        if len(cells) < 3 or a is None:
            continue
        docs.append(Doc("banestes", urllib.parse.urljoin(URL_BANESTES_BASE, a["href"]),
                        "values", year=int(cells[0]), month=month_number(cells[2]),
                        agency=cells[1], label=cells[2]))
    return docs


def _basa_rsc(text: str) -> str:
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', text, flags=re.S)
    return "".join(json.loads('"' + c + '"') for c in chunks)


# Banrisul's 2024 detail page lists that year's supplier files but omits the <a> tags for its
# twelve "Valores Pagos" files, which the CMS still serves under /midias/ by name. They cannot be
# discovered from any page, so the URLs are recorded here. The parser reads each month from the
# document's own "Mes/Ano Ref." line, so a wrong month in a file name cannot mislabel a period.
BANRISUL_UNLINKED_2024 = tuple(
    URL_BANRISUL_MEDIA + f"{i}_Banrisul-Execucao-Contratual-Publicidade-Valores-Pagos-Ref{m}2024.pdf"
    for i, m in ((47885, "JAN"), (47887, "FEV"), (47893, "MAR"), (47922, "ABR"), (47925, "MAI"),
                 (47928, "JUN"), (47930, "JUL"), (47933, "AGO"), (47935, "SET"), (47939, "OUT"),
                 (47941, "NOV"), (47943, "DEZ")))

# BASA's own site has dropped its older files. Two are held only by the Internet Archive, at the
# id_ form of the capture URL, which returns the stored bytes rather than a rewritten page. A third
# capture, fornecedores_consolidado_2014.pdf, is NOT listed: it is a bare supplier list whose period
# appears nowhere in its text, so its year would rest on the file name alone. It carries no amounts,
# so leaving it out costs nothing.
BASA_ARCHIVED = (
    ("https://web.archive.org/web/20171114222003id_/http://bancoamazonia.com.br/images/"
     "arquivos/contas_publicas/valores_pagos_consolidado_2014.pdf", 2014),
    ("https://web.archive.org/web/20171114191932id_/http://bancoamazonia.com.br/images/"
     "arquivos/institucional/pagamento_servicos/valores_pagos_2012.pdf", 2012),
)


def discover_basa(cache: Cache) -> list[Doc]:
    rsc = _basa_rsc(html_text(cache, URL_BASA_LIST))
    m = re.search(r'"title":"Pagamentos de Servi\S+ de Publicidade".{0,400}?'
                  r'"\$","div","(\d+)"', rsc, flags=re.S)
    if not m:
        raise SystemExit("BASA: section 'Pagamentos de Servicos de Publicidade' not found")
    key = re.search(r'"block-rich-text-complex-%s-content\.files-\d+-\d+",' % m.group(1), rsc)
    obj, _ = json.JSONDecoder().raw_decode(rsc[rsc.index("{", key.end()):])
    docs = []
    for f in obj["files"]:
        url = (f.get("upload") or {}).get("url") or (f["url"] if f["url"] != "/" else None)
        docs.append(Doc("basa", url or "", "values", year=int(f["tag"]) if f.get("tag") else None,
                        label=f["name"]))
    live = {d.url for d in docs}
    for url, year in BASA_ARCHIVED:
        if url not in live:
            docs.append(Doc("basa", url, "values", year=year,
                            label=f"valores pagos {year} (Internet Archive)"))
    return docs


def discover_banrisul(cache: Cache) -> list[Doc]:
    from bs4 import BeautifulSoup
    listing = html_text(cache, URL_BANRISUL_LIST, encoding="latin1")
    years = re.findall(r'carregarPaginaConteudo\("servico-marketing-detalhe",\s*(\d+).*?'
                       r'<span>\s*(\d{4})\s*</span>', listing, flags=re.S)
    if not years:
        raise SystemExit("Banrisul: no year sections found on the marketing page")
    docs = []
    for page_id, year in years:
        soup = BeautifulSoup(html_text(cache, URL_BANRISUL_DETAIL.format(id=page_id),
                                       encoding="latin1"), "lxml")
        for a in soup.find_all("a", href=True):
            if "midias/" not in a["href"]:
                continue
            text = a.get_text(" ", strip=True)
            m = re.match(r"(\w+)\s+(\d{4})", fold(text))
            role = "values" if "VALORES PAGOS" in fold(text) else "supplier_names"
            docs.append(Doc("banrisul", URL_BANRISUL_MEDIA + a["href"].split("midias/")[-1],
                            role, year=int(year), month=month_number(m.group(1)) if m else None,
                            label=text))
    listed = {d.url for d in docs}
    for url in BANRISUL_UNLINKED_2024:
        if url in listed:
            continue
        mo = re.search(r"Ref([A-Z]{3})(\d{4})\.pdf$", url)
        docs.append(Doc("banrisul", url, "values", year=int(mo.group(2)),
                        month=month_number(mo.group(1)),
                        label=f"Valores Pagos {mo.group(1)}/{mo.group(2)} (page omits the link)"))
    return docs


BRB_ENTITY = {"BRB": "bank", "DTVM": "corretora", "FINANCEIRA BRB": "financeira"}


def discover_brb(cache: Cache) -> list[Doc]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_text(cache, URL_BRB_LIST), "lxml")
    head = next(h for h in soup.find_all("h2") if "QDD" in h.get_text())
    by_href: dict[tuple[int, str], list[str]] = {}
    year = None
    for el in head.find_all_next(["h2", "h3", "a"]):
        if el.name == "h2":
            break
        if el.name == "h3":
            year = int(el.get_text(strip=True)) if el.get_text(strip=True).isdigit() else None
            continue
        if year and el.get("href", "").lower().endswith(".pdf"):
            by_href.setdefault((year, el["href"]), []).append(el.get_text(" ", strip=True))
    docs = []
    for (year, href), texts in by_href.items():
        text = fold(" ".join(texts))
        mq = re.search(r"([1-4])\S* TRIMESTRE/(\d{4})", text)
        me = re.search(r"\((BRB|DTVM|FINANCEIRA BRB)\)", text)
        if not (mq and me):
            log.info("  BRB listing: link without quarter or entity ignored: %s %s", year, href)
            continue
        part = re.search(r"PARTE\s*0?(\d)", text)
        docs.append(Doc("brb", urllib.parse.urljoin(URL_BRB_LIST, href), "values", year=int(mq.group(2)), quarter=int(mq.group(1)),
                        entity=BRB_ENTITY[me.group(1)],
                        label=" ".join(texts) + (f" [parte {part.group(1)}]" if part else "")))
    return docs


DISCOVER = {"banestes": discover_banestes, "basa": discover_basa,
            "banrisul": discover_banrisul, "brb": discover_brb}


def collect(bank: str, refresh: bool) -> list[Doc]:
    cache = Cache(bank, refresh)
    docs = DISCOVER[bank](cache)
    log.info("%s: %d files listed", bank, len(docs))
    for d in docs:
        if not d.url:
            d.entry = {"url": None, "status": None, "file": None, "kind": None}
            continue
        if bank == "banrisul" and d.role == "supplier_names":
            has_values = any(o.role == "values" and o.year == d.year and o.month == d.month
                             for o in docs)
            if has_values:              # supplier name lists are read only where no values file
                continue
        d.entry = cache.get(d.url)
    return [d for d in docs if d.entry]


# ---------------------------------------------------------------------------
# Parsed records
# ---------------------------------------------------------------------------
LINE_COLUMNS = ["bank_key", "cnpj8", "entity", "period", "frequency", "agency", "section",
                "category", "detail", "category_group", "amount_brl", "amount_kind", "basis",
                "line_id", "total_scope", "client_as_published", "period_rule", "reversal",
                "source_file", "source_url", "parse_method", "page_or_sheet",
                # which printed table the line belongs to, and (BRB) the printed row it came from:
                # page-text hash, table and row position, month - the same in every copy of a page
                "table_id", "content_key",
                # BRB: how the entity was read, that entity's CNPJ, the date of the gazette issue the
                # table was printed in, the notice's own republication or rectification sentence
                # (if any), and the earlier publication a line supersedes
                "entity_rule", "entity_cnpj8", "published", "republished", "supersedes",
                # the amount as printed; differs from amount_brl only where `imputed`
                "amount_as_printed", "imputed", "imputation_rule",
                # BRB: which later notice this row-specific text correction is from, when one
                # zeroed this line (`brb_apply_notice_corrections`); None on every other line,
                # including one changed only by `imputed_from_row_total`
                "corrected_by"]


@dataclass
class Parsed:
    """What one file yields: lines, stated totals and the (entity, period) pairs it covers.
    The BRB parser also reports which pages it skipped as already read elsewhere, the gazette
    issues it saw and the rows whose month cells do not sum to their own TOTAL cell."""
    lines: list[dict] = field(default_factory=list)
    totals: list[dict] = field(default_factory=list)
    periods: set = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    n_pages: int = 0
    pages_skipped: list[str] = field(default_factory=list)
    published: set = field(default_factory=set)
    gaps: list[dict] = field(default_factory=list)


def line(doc: Doc, **kw) -> dict:
    rec = {c: None for c in LINE_COLUMNS}
    rec.update(bank_key=doc.bank, entity=doc.entity, agency=doc.agency, source_url=doc.url,
               source_file=doc.entry.get("file"), reversal=False, extra_scopes=[], imputed=False)
    rec.update(kw)
    return rec


def rows_of(page, ytol: float = 3.0, x_tolerance: float = 1.5) -> list[list[dict]]:
    """Words of a PDF page grouped into visual rows (tops within ytol), left to right."""
    words = page.extract_words(x_tolerance=x_tolerance, y_tolerance=2, keep_blank_chars=False)
    words.sort(key=lambda w: (w["top"], w["x0"]))
    rows: list[list[dict]] = []
    for w in words:
        if rows and abs(rows[-1][0]["top"] - w["top"]) <= ytol:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [sorted(r, key=lambda w: w["x0"]) for r in rows]


def row_text(row: list[dict]) -> str:
    return " ".join(w["text"] for w in row)


def xmid(w: dict) -> float:
    return (w["x0"] + w["x1"]) / 2


def joined_amount(words: list[dict], max_gap: float = 1.0) -> float | None:
    """First amount among a row's words, rebuilding amounts that the PDF split into touching
    fragments ('3' + '2.557,95' with no gap -> 32557.95). Fragments further apart than max_gap
    points are separate numbers (invoice numbers, other columns). 'R$ -' is a published zero."""
    toks = sorted((w for w in words if re.fullmatch(r"-|R\$|R\$-?[\d.,]+|-?[\d.,]+", w["text"])),
                  key=lambda w: w["x0"])
    clusters: list[list[dict]] = []
    for w in toks:
        if w["text"] == "R$":
            clusters.append([])
            continue
        if clusters and clusters[-1] and w["x0"] - clusters[-1][-1]["x1"] <= max_gap:
            clusters[-1].append(w)
        else:
            clusters.append([w])
    dash = False
    for c in clusters:
        text = "".join(w["text"] for w in c).replace("R$", "")
        if text == "-":
            dash = True
            continue
        if re.fullmatch(r"-?\d{1,3}(\.\d{3})*,\d{2}|-?\d+,\d{2}", text):
            return brl(text)
    return 0.0 if dash else None


def year_month_from(text: str, fallback_year: int | None = None) -> tuple[int | None, int | None]:
    """(year, month) from 'MARCO/2016', 'MAR 2019', 'jun/26', 'JANEIRO DE 2016', 'xxxxx 2026'."""
    t = fold(text)
    m = re.search(r"\b([A-Z]{3,9})\s*(?:/|\bDE\b)?\s*(\d{4}|\d{2})\b", t)
    if m and month_number(m.group(1)):
        y = int(m.group(2))
        return (y + 2000 if y < 100 else y), month_number(m.group(1))
    y = re.search(r"\b(20\d{2})\b", t)
    return (int(y.group(1)) if y else fallback_year), None


# ---------------------------------------------------------------------------
# Banestes
# ---------------------------------------------------------------------------
BANESTES_MEDIA = {"JORNAL", "REVISTA", "RADIO", "TV", "INTERNET", "MOOH", "OOH", "CINEMA",
                  "OUTROS"}
BANESTES_AGENCIES = ("FIRE", "MP", "SET", "AMPLA")


def banestes_agency(text: str) -> str | None:
    t = fold(text)
    return next((k for k in BANESTES_AGENCIES if re.search(rf"\b{k}\b", t)), None)


def banestes_header(text: str, doc: Doc) -> dict:
    t = fold(text)
    ag = re.search(r"NOME DA AGENCIA:?\s*(.*?)\s*MES/ANO", t)
    comp = re.search(r"COMPETENCIA:?\s*(.*?)(?:\s*CLIENTE|\s*$)", t)
    cli = re.search(r"CLIENTE:?\s*([A-Z/][A-Z/ .]*?)\s*(?:CONTRATO|N[O\xba]|\d|$)", t)
    comp_text = comp.group(1).strip() if comp else ""
    annual = "JANEIRO A DEZEMBRO" in comp_text
    year, month = year_month_from(comp_text, doc.year)
    return {"agency": banestes_agency(ag.group(1) if ag else "") or banestes_agency(doc.agency or ""),
            "year": year, "month": None if annual else month, "annual": annual,
            "client": cli.group(1).strip() if cli else None, "competencia": comp_text}


def banestes_period(h: dict, doc: Doc) -> tuple[str, str, str]:
    """(period, frequency, period_rule): the document's competencia month; the listing's month
    only where the document's is unreadable (placeholders such as 'xxxxx 2026')."""
    if h["annual"]:
        return str(h["year"]), "annual", "document competencia (JANEIRO A DEZEMBRO)"
    if h["month"] and h["year"]:
        return f"{h['year']}{h['month']:02d}", "monthly", "document competencia"
    if doc.month and (h["year"] or doc.year):
        return (f"{h['year'] or doc.year}{doc.month:02d}", "monthly",
                f"listing month; document competencia reads '{h['competencia']}'")
    raise ValueError(f"no period in {doc.url}")


def parse_banestes_pdf(doc: Doc, path: Path) -> Parsed:
    import pdfplumber
    out = Parsed()
    mode, valor_range, header, block = None, None, None, 0
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            rows = rows_of(page)
            ftext = fold("\n".join(row_text(r) for r in rows))
            if "NOME DA AGENCIA" in ftext:
                header = banestes_header(" ".join(row_text(r) for r in rows[:6]), doc)
            if header is None:
                continue
            period, freq, rule = banestes_period(header, doc)
            common = dict(agency=header["agency"], period=period, frequency=freq,
                          period_rule=rule, client_as_published=header["client"],
                          basis="undocumented", page_or_sheet=f"p{pno}")
            if "MEIO DE DIVULGACAO" in ftext:
                mode, block = "media", block + 1
                scope = f"{path.name}#media{block}"
                stated = None
                for r in rows:
                    first = fold(r[0]["text"])
                    if first == "TOTAL" and len(r) > 1 and fold(r[1]["text"]) == "GERAL":
                        stated = joined_amount(r[2:])
                    elif first in BANESTES_MEDIA:
                        amt = joined_amount(r[1:])
                        if amt is None:
                            continue                       # blank cell: nothing published
                        out.lines.append(line(
                            doc, section="MEIO DE DIVULGACAO", category=r[0]["text"],
                            category_group="advertising", amount_brl=amt, amount_kind="total",
                            parse_method="pdfplumber words, row-aligned", total_scope=scope,
                            line_id=f"{scope}:{first}", **common))
                if stated is not None:
                    out.totals.append({"scope": scope, "stated": stated, "label": "TOTAL GERAL"})
                out.periods.add(("bank", period))
                continue
            head = next((r for r in rows if "FORNECEDOR" in [fold(w["text"]) for w in r]
                         and any(fold(w["text"]).startswith("VALOR") for w in r)), None)
            if head is not None:
                mode = "supplier"
                x_val = next(w for w in head if fold(w["text"]).startswith("VALOR"))
                nf = [w for w in head if fold(w["text"]) == "NF" and w["x0"] > x_val["x0"]]
                soc = [w for w in head if fold(w["text"]) == "SOCIAL"]
                valor_range = (((soc[0]["x1"] + x_val["x0"]) / 2) if soc else x_val["x0"] - 60,
                               (nf[0]["x0"] + 2) if nf else x_val["x1"] + 60)
            elif "NOME DA AGENCIA" in ftext:
                mode = None                                # vehicle list: names only
                continue
            if mode != "supplier":
                continue
            scope = f"{path.name}#supplier{block}"
            out.periods.add(("bank", period))
            for r in rows:
                if r is head or fold(r[0]["text"]) in ("ACOMPANHAMENTO", "NOME", "CLIENTE:",
                                                       "CLIENTE") \
                        or fold(row_text(r)).startswith(("NOME DA AGENCIA", "CLIENTE")):
                    continue
                in_col = [w for w in r if w["x0"] >= valor_range[0] and w["x1"] <= valor_range[1]]
                amt = joined_amount(in_col)
                if fold(r[0]["text"]) == "TOTAL":
                    tot = amt if amt is not None else joined_amount(r[1:])
                    if tot is not None:
                        out.totals.append({"scope": scope, "stated": tot,
                                           "label": row_text(r)[:40]})
                    continue
                name = " ".join(w["text"] for w in r if xmid(w) < valor_range[0])
                if amt is None or (amt == 0.0 and not re.search(r"[A-Za-z]", name)):
                    continue
                out.lines.append(line(
                    doc, section="FORNECEDOR", category="FORNECEDOR", detail=name[:120],
                    category_group="production", amount_brl=amt, amount_kind="net",
                    parse_method="pdfplumber words, VALOR column by header x-range",
                    total_scope=scope, line_id=f"{scope}:{pno}:{len(out.lines)}", **common))
    return out


def parse_banestes_xlsx(doc: Doc, path: Path) -> Parsed:
    out = Parsed()
    for name, df in pd.read_excel(path, sheet_name=None, header=None, dtype=str).items():
        cells = df.fillna("").astype(str)
        head_text = " ".join(" ".join(r) for r in cells.values.tolist()[:6])
        header = banestes_header(head_text, doc)
        period, freq, rule = banestes_period(header, doc)
        common = dict(agency=header["agency"], period=period, frequency=freq, period_rule=rule,
                      client_as_published=header["client"], basis="undocumented",
                      page_or_sheet=name, parse_method="openpyxl cells")
        scope = f"{path.name}#{name.strip()}"
        sheet = fold(name)
        for i, r in cells.iterrows():
            label = fold(r.iloc[0])
            if sheet.startswith("VALORES POR MEIO"):
                if label in BANESTES_MEDIA and r.iloc[1].strip():
                    out.lines.append(line(
                        doc, section="MEIO DE DIVULGACAO", category=r.iloc[0].strip(),
                        category_group="advertising", amount_brl=float(r.iloc[1]),
                        amount_kind="total", total_scope=scope, line_id=f"{scope}:{label}",
                        **common))
                elif label == "TOTAL GERAL":
                    out.totals.append({"scope": scope, "stated": float(r.iloc[1] or 0),
                                       "label": "TOTAL GERAL"})
            elif sheet.startswith("VALORES FORNECEDORES"):
                if label.startswith("TOTAL"):
                    out.totals.append({"scope": scope, "stated": float(r.iloc[3] or 0),
                                       "label": label})
                elif r.iloc[0].strip() and label != "FORNECEDOR"                         and re.fullmatch(r"-?\d+(\.\d+)?", r.iloc[3].strip()):
                    out.lines.append(line(
                        doc, section="FORNECEDOR", category="FORNECEDOR",
                        detail=r.iloc[0].strip()[:120], category_group="production",
                        amount_brl=float(r.iloc[3]), amount_kind="net", total_scope=scope,
                        line_id=f"{scope}:{i}", **common))
        out.periods.add(("bank", period))
    return out


def parse_banestes(doc: Doc, path: Path) -> Parsed:
    if doc.entry["kind"] == "xlsx":
        return parse_banestes_xlsx(doc, path)
    return parse_banestes_pdf(doc, path)


# ---------------------------------------------------------------------------
# Banco da Amazonia
# ---------------------------------------------------------------------------
BASA_KINDS = (("gross", "Valor Bruto"), ("tax", "Tributos"), ("net", "Valor Liquido"))


def basa_group(label: str) -> str:
    """Explicit rule on the published 'Meio de Divulgacao' / service label."""
    t = fold(label)
    if "VEICULA" in t or "(MIDIA)" in t.replace(" ", "") or t.endswith("/MIDIA"):
        return "advertising"
    if "PRODU" in t:
        return "production"
    return "other"


def read_table_file(path: Path, kind: str) -> dict[str, pd.DataFrame]:
    if kind in ("xls", "xlsx"):
        return pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    raw = path.read_bytes()
    for enc in ("utf-8", "cp850"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    import io
    return {"csv": pd.read_csv(io.StringIO(text), sep=";", header=None, dtype=str)}


def to_float(cell) -> float | None:
    s = str(cell).strip()
    if s in ("", "nan", "None"):
        return None
    if re.search(r",\d{1,2}$", s):
        return brl(s)
    try:
        return float(s)
    except ValueError:
        return None


def month_window(text: str) -> tuple[str, str] | None:
    """('YYYYMM', 'YYYYMM') from 'Periodo: Janeiro/2020 a Fevereiro/2022', 'Janeiro a Dezembro
    2024', 'Janeiro/19 a Junho/20', 'Janeiro 2017 a Junho 2018'."""
    t = fold(text)
    m = re.search(r"([A-Z]{3,9})\s*/?\s*(\d{2,4})?\s+A\s+([A-Z]{3,9})\s*/?\s*(\d{2,4})", t)
    if not m or not month_number(m.group(1)) or not month_number(m.group(3)):
        return None
    y2 = int(m.group(4))
    y2 = y2 + 2000 if y2 < 100 else y2
    y1 = int(m.group(2)) if m.group(2) else y2
    y1 = y1 + 2000 if y1 < 100 else y1
    return f"{y1}{month_number(m.group(1)):02d}", f"{y2}{month_number(m.group(3)):02d}"


def months_between(a: str, b: str) -> list[str]:
    return [p.strftime("%Y%m") for p in pd.period_range(pd.Period(a, "M"), pd.Period(b, "M"), freq="M")]


def parse_basa_resumo(doc: Doc, path: Path) -> Parsed:
    """LAI 'Meio de Divulgacao' files, 2019-2024: one row per contract x month x medium/type with
    gross, tax and net; closing rows FORNECEDOR/VEICULO (= the rows above), HONORARIOS/DESCONTO
    DE AGENCIA, CRIACAO AGENCIA and TOTAL, published only for the whole file."""
    out = Parsed()
    (sheet, df), = list(read_table_file(path, doc.entry["kind"]).items())[:1]
    cells = df.fillna("").astype(str)
    head = " ".join(" ".join(r) for r in cells.values.tolist()[:10])
    window = month_window(re.search(r"Per\S*odo:?(.*)", head).group(1)) if "odo" in head else None
    rows, tail = [], {}
    for i, r in cells.iterrows():
        c0 = r.iloc[0].strip()
        if re.match(r"^\d{4}/\d+", c0):
            rows.append((i, r))
        elif fold(c0) in ("FORNECEDOR/VEICULO", "HONORARIOS/DESCONTO DE AGENCIA",
                          "CRIACAO AGENCIA", "TOTAL"):
            tail[fold(c0)] = (c0, [to_float(r.iloc[k]) for k in (4, 5, 6)])
    months = []
    for i, r in rows:
        mm = re.search(r"(\d{2})/(\d{4})", r.iloc[2])
        period = f"{mm.group(2)}{mm.group(1)}"
        months.append(period)
        agency = r.iloc[0].split(" - ", 2)[-1].strip()
        label = r.iloc[3].strip()
        for k, (kind, _) in enumerate(BASA_KINDS):
            v = to_float(r.iloc[4 + k])
            out.lines.append(line(
                doc, agency=agency, period=period, frequency="monthly",
                period_rule="document Mes/Ano column", section="RELATORIO MEIO DIVULGACAO",
                category=label, detail=f"contrato {r.iloc[0].split(' - ')[0]}; vigencia {r.iloc[1]}",
                category_group=basa_group(label), amount_brl=v, amount_kind=kind, basis="paid",
                total_scope=f"{path.name}#fornecedor_veiculo:{kind}",
                extra_scopes=[f"{path.name}#file_total:{kind}"], reversal=bool(v is not None and v < 0),
                line_id=f"{path.name}:{i}", page_or_sheet=sheet, parse_method="spreadsheet cells"))
    span = f"{min(months)}-{max(months)}" if months else None
    for key, group in (("HONORARIOS/DESCONTO DE AGENCIA", "other"), ("CRIACAO AGENCIA", "other")):
        if key not in tail:
            continue
        label, vals = tail[key]
        for (kind, _), v in zip(BASA_KINDS, vals):
            out.lines.append(line(
                doc, period=span, frequency="multi_month",
                period_rule="published only as a total for the whole file", section="RESUMO",
                category=label, category_group=group, amount_brl=v, amount_kind=kind,
                basis="paid", total_scope=f"{path.name}#file_total:{kind}",
                line_id=f"{path.name}:{key}", page_or_sheet=sheet,
                parse_method="spreadsheet cells"))
    for key, scope in (("FORNECEDOR/VEICULO", "fornecedor_veiculo"), ("TOTAL", "file_total")):
        if key in tail:
            for (kind, _), v in zip(BASA_KINDS, tail[key][1]):
                out.totals.append({"scope": f"{path.name}#{scope}:{kind}", "stated": v,
                                   "label": f"{tail[key][0]} ({kind})"})
    for p in (months_between(*window) if window else sorted(set(months))):
        out.periods.add(("bank", p))
    out.notes.append(f"window {window}; line months {span}")
    return out


def parse_basa_escala(doc: Doc, path: Path) -> Parsed:
    """LAI ESCALA files, 2025 onward: one sheet per quarter and type (VEICULACAO, PRODUCAO,
    CRIACAO) with one row per invoice (NF): gross billed, total taxes, net and payment date,
    plus a RESUMO sheet with the file's totals."""
    out = Parsed()
    sheets = read_table_file(path, "xlsx")
    for name, df in sheets.items():
        cells = df.fillna("").astype(str)
        if fold(name) == "RESUMO":
            for _, r in cells.iterrows():
                if fold(r.iloc[0]) == "TOTAL":
                    vals = [v for v in (to_float(c) for c in r.iloc[1:]) if v is not None][:3]
                    for (kind, _), v in zip(BASA_KINDS, vals):
                        if v is None:
                            continue
                        out.totals.append({"scope": f"{path.name}#all_sheets:{kind}",
                                           "stated": v, "label": f"RESUMO TOTAL ({kind})"})
            continue
        head_idx = cells.index[cells.iloc[:, 0].str.strip() == "NF"]
        if len(head_idx) == 0:
            continue
        cols = [fold(c) for c in cells.loc[head_idx[0]]]
        col = {"gross": next(i for i, c in enumerate(cols) if "FATURAMENTO" in c),
               "tax": next(i for i, c in enumerate(cols) if "TOTAL DOS IMPOSTOS" in c),
               "net": next(i for i, c in enumerate(cols) if c.startswith("VALOR LIQUIDO")),
               "paid": next(i for i, c in enumerate(cols) if c.startswith("DATA PAG")),
               "service": next(i for i, c in enumerate(cols) if c.startswith("SERVICO")),
               "who": next((i for i, c in enumerate(cols) if c in ("VEICULO", "FORNECEDOR")), None)}
        sheet_type = ("VEICULACAO" if "VEICULA" in fold(name) else
                      "PRODUCAO" if "PRODU" in fold(name) else "CRIACAO")
        group = {"VEICULACAO": "advertising", "PRODUCAO": "production", "CRIACAO": "other"}[sheet_type]
        for i in cells.index[cells.index > head_idx[0]]:
            r = cells.loc[i]
            vals = {k: to_float(r.iloc[col[k]]) for k in ("gross", "tax", "net")}
            if any(fold(x) == "TOTAL" for x in r.tolist()):
                for kind in ("gross", "tax", "net"):
                    v = to_float(r.iloc[col[kind]])
                    if v is not None:
                        out.totals.append({"scope": f"{path.name}#{name.strip()}:{kind}",
                                           "stated": v, "label": f"{name.strip()} TOTAL ({kind})"})
                continue
            if not vals["gross"]:
                continue
            paid = pd.to_datetime(r.iloc[col["paid"]], errors="coerce")
            if pd.isna(paid):
                raise ValueError(f"{path.name} {name} row {i}: invoice without payment date")
            for kind in ("gross", "tax", "net"):
                out.lines.append(line(
                    doc, agency="Escala Comunicacao e Marketing", period=paid.strftime("%Y%m"),
                    frequency="monthly", period_rule="invoice payment date (Data Pagto)",
                    section=name.strip(), category=r.iloc[col["service"]].strip() or sheet_type,
                    detail=(f"NF {r.iloc[0].strip()}; "
                            + (r.iloc[col["who"]].strip() if col["who"] is not None else "")),
                    category_group=group, amount_brl=vals[kind], amount_kind=kind, basis="paid",
                    total_scope=f"{path.name}#{name.strip()}:{kind}",
                    extra_scopes=[f"{path.name}#all_sheets:{kind}"],
                    reversal=bool(vals[kind] is not None and vals[kind] < 0),
                    line_id=f"{path.name}:{name.strip()}:{i}", page_or_sheet=name.strip(),
                    parse_method="spreadsheet cells"))
                out.periods.add(("bank", paid.strftime("%Y%m")))
    return out


def basa_word_rows(path: Path) -> list[tuple[int, list[dict]]]:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return [(pno, r) for pno, p in enumerate(pdf.pages, start=1) for r in rows_of(p)]


def parse_basa_month_matrix(doc: Doc, path: Path) -> Parsed:
    """A month x item table in two halves (January-June, then July-December plus TOTAL GERAL),
    sections PRODUCAO and VEICULACAO, closing with a published month-total row.

    Two vintages share the layout: 'Relatorio Consolidado Producao e Veiculacao 2016' (sections
    with SUBTOTAL rows, closing 'TOTAL GERAL/MES') and 'RELATORIO BAMZ - UNIFICADO - ANO 2014'
    (PRODUCAO is itself a row of amounts, media are rows under VEICULACAO, closing 'TOTAL/MES').
    The year comes from the document, not from the code."""
    out = Parsed()
    year = doc.year or int(re.search(r"(20\d{2})", doc.label or path.name).group(1))
    section, headers = None, []
    for pno, r in basa_word_rows(path):
        ft = fold(row_text(r))
        if ft.startswith("ESPECIFICACAO"):
            headers = [(xmid(w), month_number(w["text"])) for w in r[1:]
                       if month_number(w["text"])]
            if "TOTAL GERAL" in ft:
                tw = [w for w in r if fold(w["text"]) == "TOTAL"][0]
                headers.append((tw["x1"] + 10, 13))
            continue
        if ft in ("PRODUCAO", "VEICULACAO"):
            section = ft
            continue
        amounts = [(xmid(w), brl(w["text"])) for w in r if re.fullmatch(r"[\d.]+,\d{2}", w["text"])]
        if not amounts or not headers:
            continue
        label = " ".join(w["text"] for w in r if not re.fullmatch(r"R\$|[\d.,]+", w["text"]))
        for x, v in amounts:
            col = min(headers, key=lambda h: abs(h[0] - x))[1]
            period = f"{year}" if col == 13 else f"{year}{col:02d}"
            flabel = fold(label)
            # 2016 closes with TOTAL GERAL/MES, 2014 with TOTAL/MES; both state the month's total.
            if flabel.startswith("TOTAL GERAL") or flabel.startswith("TOTAL/MES"):
                out.totals.append({"scope": f"{path.name}#{period}", "stated": v,
                                   "label": f"TOTAL GERAL/MES {period}"})
            elif flabel == "SUBTOTAL":
                out.totals.append({"scope": f"{path.name}#{section}:{period}", "stated": v,
                                   "label": f"SUBTOTAL {section} {period}"})
            elif col != 13:
                # 2014 files PRODUCAO as a row of its own rather than as a section header.
                group = ("advertising" if section == "VEICULACAO" else
                         "production" if flabel == "FORNECEDOR"
                         or flabel.startswith("PRODUCAO") else "other")
                out.lines.append(line(
                    doc, period=period, frequency="monthly", period_rule="document month column",
                    section=section, category=label, category_group=group, amount_brl=v,
                    amount_kind="total", basis="paid", total_scope=f"{path.name}#{period}",
                    extra_scopes=[f"{path.name}#{section}:{period}",
                                  f"{path.name}#{section}:{year}", f"{path.name}#{year}"],
                    line_id=f"{path.name}:{section}:{flabel}:{period}", page_or_sheet=f"p{pno}",
                    parse_method="pdfplumber words, month column by header x"))
                out.periods.add(("bank", period))
    return out


def parse_basa_2017(doc: Doc, path: Path) -> Parsed:
    """'Relatorio de Producao (Meio de divulgacao, tipo de servico e honorarios)', unified
    January 2017 to June 2018: rows with month, medium '(MIDIA)'/'(PRODUCAO)', gross, tax, net;
    closing FORNECEDOR/VEICULO, DESCONTO DE AGENCIA/HONORARIOS, CRIACAO AGENCIA and TOTAL."""
    out = Parsed()
    rows = basa_word_rows(path)
    head = next(r for _, r in rows if "BRUTO" in fold(row_text(r)) and "TRIBUTOS" in fold(row_text(r)))
    xs = {"gross": next(xmid(w) for w in head if fold(w["text"]) == "BRUTO") - 10,
          "tax": next(xmid(w) for w in head if fold(w["text"]) == "TRIBUTOS"),
          "net": next(xmid(w) for w in head if fold(w["text"]) == "LIQUIDO") - 10}
    window = month_window(" ".join(row_text(r) for _, r in rows[:6]))
    pending, months, tail_rows = "", [], []
    for pno, r in rows:
        text = row_text(r)
        mm = next((w for w in r if re.fullmatch(r"\d{2}/20\d{2}", w["text"])), None)
        amounts = [(xmid(w), brl(w["text"].replace("R$", ""))) for w in r
                   if re.fullmatch(r"R?\$?\d[\d.]*,\d{2}", w["text"])]
        type_words = [w["text"] for w in r if mm is not None and w["x0"] > mm["x1"]
                      and not re.search(r"\d,\d{2}", w["text"])]
        if mm is None:
            if amounts and re.search(r"FORNECEDOR|HONORARIOS|CRIACAO|TOTAL", fold(text)) or \
                    (amounts and tail_rows):
                tail_rows.append((pno, r))
            elif not amounts:
                frag = [w["text"] for w in r if 250 < w["x0"] < xs["gross"] - 20]
                pending = " ".join(frag) if frag else ""
            continue
        label = " ".join(type_words)
        if pending and "(" not in label:
            label = f"{pending} {label}".strip()
        pending = ""
        vals = {k: None for k in xs}
        for x, v in amounts:
            vals[min(xs, key=lambda k: abs(xs[k] - x))] = v
        period = f"{mm.group(0)[3:] if hasattr(mm, 'group') else mm['text'][3:]}{mm['text'][:2]}"
        months.append(period)
        for kind in ("gross", "tax", "net"):
            if vals[kind] is None:
                continue
            out.lines.append(line(
                doc, period=period, frequency="monthly", period_rule="document Mes/Ano column",
                section="RELATORIO MEIO DIVULGACAO", category=label,
                category_group=basa_group(label), amount_brl=vals[kind], amount_kind=kind,
                basis="paid", total_scope=f"{path.name}#fornecedor_veiculo:{kind}",
                extra_scopes=[f"{path.name}#file_total:{kind}"],
                line_id=f"{path.name}:{pno}:{int(r[0]['top'])}", page_or_sheet=f"p{pno}",
                parse_method="pdfplumber words, amount columns by header x"))
    # Closing block: labels and amounts are vertically offset; the i-th amount of each column
    # belongs to the i-th label (checked below through TOTAL = sum of the other three).
    labels = [fold(row_text(r)) for _, r in tail_rows]
    names = [n for n in ("FORNECEDOR/VEICULO", "HONORARIOS", "CRIACAO AGENCIA", "TOTAL")
             if any(n in lab for lab in labels)]
    cols = {k: [] for k in xs}
    for _, r in tail_rows:
        for w in r:
            if re.fullmatch(r"R?\$?\d[\d.]*,\d{2}", w["text"]):
                cols[min(xs, key=lambda k: abs(xs[k] - xmid(w)))].append(brl(w["text"].replace("R$", "")))
    if len(names) == 4 and all(len(v) == 4 for v in cols.values()):
        for kind, vals in cols.items():
            if abs(sum(vals[:3]) - vals[3]) > 0.02:
                raise ValueError(f"{path.name}: closing block does not add up for {kind}")
        span = f"{min(months)}-{max(months)}"
        for idx, key in ((1, "HONORARIOS/DESCONTO DE AGENCIA"), (2, "CRIACAO AGENCIA")):
            for kind in xs:
                out.lines.append(line(
                    doc, period=span, frequency="multi_month",
                    period_rule="published only as a total for the whole file", section="RESUMO",
                    category=key, category_group="other", amount_brl=cols[kind][idx],
                    amount_kind=kind, basis="paid", total_scope=f"{path.name}#file_total:{kind}",
                    line_id=f"{path.name}:{key}", page_or_sheet=f"p{tail_rows[0][0]}",
                    parse_method="pdfplumber words, closing block by rank"))
        for kind in xs:
            out.totals.append({"scope": f"{path.name}#fornecedor_veiculo:{kind}",
                               "stated": cols[kind][0], "label": f"FORNECEDOR/VEICULO ({kind})"})
            out.totals.append({"scope": f"{path.name}#file_total:{kind}", "stated": cols[kind][3],
                               "label": f"TOTAL ({kind})"})
    else:
        out.notes.append(f"closing block not resolved: labels {names}, columns "
                         f"{ {k: len(v) for k, v in cols.items()} }")
    for p in set(months_between(*window) if window else []) | set(months):
        out.periods.add(("bank", p))
    return out


def parse_basa_annual(doc: Doc, path: Path) -> Parsed:
    """'Execucao Contratual de Publicidade - Valores Pagos (Servicos/Meios)': one column for the
    whole year, rows PRODUCAO and VEICULACAO/<medium>, closing with TOTAL. Annual, so these lines
    reach `statebank_advertising_lines` but not the quarterly panel, which needs quarters."""
    out = Parsed()
    year = doc.year or int(re.search(r"(20\d{2})", doc.label or path.name).group(1))
    for pno, r in basa_word_rows(path):
        amounts = [brl(w["text"]) for w in r if re.fullmatch(r"[\d.]+,\d{2}", w["text"])]
        amounts = [v for v in amounts if v is not None]
        label = " ".join(w["text"] for w in r if not re.fullmatch(r"R\$|[\d.,]+", w["text"])).strip()
        if len(amounts) != 1 or not label:
            continue                      # headings, and the wrapped second line of a long label
        flab = fold(label)
        if flab.startswith("TOTAL"):
            out.totals.append({"scope": f"{path.name}#{year}", "stated": amounts[0],
                               "label": f"TOTAL {year}"})
            continue
        group = ("production" if flab.startswith("PRODUCAO") else
                 "advertising" if flab.startswith("VEICULA") else "other")
        out.lines.append(line(
            doc, period=f"{year}", frequency="annual",
            period_rule="document states one column for the year", section="VALORES PAGOS",
            category=label, category_group=group, amount_brl=amounts[0], amount_kind="total",
            basis="paid", total_scope=f"{path.name}#{year}",
            line_id=f"{path.name}:{flab}", page_or_sheet=f"p{pno}",
            parse_method="pdfplumber words, one amount column"))
        out.periods.add(("bank", f"{year}"))
    stated = next((t["stated"] for t in out.totals), None)
    got = sum(ln["amount_brl"] for ln in out.lines)
    if stated is not None and abs(stated - got) > 0.05:
        out.notes.append(f"the document's own TOTAL {stated:,.2f} exceeds its itemised rows "
                         f"{got:,.2f} by {stated - got:,.2f}; the rows are kept as published")
    return out


def parse_basa(doc: Doc, path: Path) -> Parsed:
    name = fold(doc.label)
    if doc.entry["kind"] in ("xlsx", "xls", "text") and "ESCALA" in fold(doc.url):
        return parse_basa_escala(doc, path)
    if "MEIO DE DIVULGA" in name and doc.year and doc.year >= 2019:
        return parse_basa_resumo(doc, path)
    url = fold(doc.url).replace("-", "_")
    if doc.year == 2016 or "VALORES_PAGOS_CONSOLIDADO" in url:
        return parse_basa_month_matrix(doc, path)
    if re.search(r"VALORES_PAGOS_20\d{2}\.PDF", url):
        return parse_basa_annual(doc, path)
    if doc.year == 2017 and "1_TRIMESTRE" in fold(doc.url).replace(" ", "_"):
        return parse_basa_2017(doc, path)
    out = Parsed()
    out.notes.append("not a values source for this script (cross-check or names-only file)")
    return out


# ---------------------------------------------------------------------------
# BRB
# ---------------------------------------------------------------------------
BRB_ENTITIES = (("cartao", r"CARTAO BRB"),
                ("financeira", r"CREDITO,? FINANCIAMENTO|FINANCEIRA BRB"),
                ("corretora", r"DISTRIBUIDORA DE TITULOS|\bDTVM\b"),
                ("bank", r"BANCO DE BRASILIA"))
# The same names once whitespace is removed from the folded text, so letter-spaced and run-together
# headings ("EVALORESMOBILIARIOS S.A.divulgaabaixo") read alike. "CREDITO,FINAN" stops short of the
# full word because two-column pages hyphenate it across lines ("FINAN-" ... "CIAMENTO").
BRB_ENTITY_NAMES = (("cartao", r"CARTAOBRB"),
                    ("financeira", r"CREDITO,?FINAN|FINANCEIRABRB"),
                    ("corretora", r"DISTRIBUIDORADETITULOS|DTVM"),
                    ("bank", r"BANCODEBRASILIA"))
# The CNPJ of each entity that publishes a QDD notice. The first three are the members of BRB's
# prudential conglomerate in the IF.data registry. Cartao BRB's CNPJ is printed in none of the
# cached gazette pages and the IF.data registry lists no institution of that name; it is the
# advertiser CNPJ that ANCINE's CRT register (raw/ancine/crt-obras-publicitarias.csv) records for
# "CARTAO BRB S/A". Membership is decided from the registry in `brb_membership`, not assumed here.
BRB_ENTITY_CNPJ8 = {"bank": "00000208", "financeira": "33136888", "corretora": "33850686",
                    "cartao": "01984199"}
AMOUNT_RE = r"-?R?\$?\s?-?\d[\d.]*,\d{2}"


BRB_SPONSOR_CATEGORY = re.compile(r"ESPORTE|ESPORTIVO|ARTEECULTURA|ENTRETENIMENTO"
                                  r"|RELACIONAMENTOINSTITUCIO|CAUSASSOCIAIS|PATROCIN")
BRB_ACCRUAL_RE = re.compile(r"REGIME\w*DECOMPETENCIA")
# A retification or republication marker. Plain "REPUBLIC" also matches "REPUBLICA" the noun
# ("Aberto da Republica de Tenis", a sponsored tournament's own name, or "Republica Federativa do
# Brasil"), which is not a notice about republishing anything, so the verb forms are spelled out
# instead of the bare stem.
BRB_RETIFIC_RE = re.compile(r"RETIFIC|REPUBLICAD|REPUBLICAR|REPUBLICACA")


def brb_group(classification: str, purpose: str, sponsorship_table: bool) -> str:
    """Explicit rule on the published CLASSIFICACAO DA DESPESA and FINALIDADE DA ACAO:
    'PUBLICACOES OBRIGATORIAS' or 'PUBLICIDADE LEGAL' -> legal_notice; '.../PRODUCAO', also written
    '.../PROD' -> production; '.../VEICULACAO', also written '.../VEIC', or plain 'PROPAGANDA E
    PUBLICIDADE', also written 'PROP E PUBL' -> advertising; a row whose own classification says
    none of that, in a table with no propaganda/publicidade/publicacoes row (Esporte, Arte e
    Cultura, Entretenimento, Relacionamento Institucional, ...) -> sponsorship; else other.

    The row's OWN classification outranks the table-level guess. It used to be the other way round,
    and because the 2016 files abbreviate the classification as 'PROP E PUBL/PROD' - which the
    table-level test did not recognise either - 158 production and advertising rows of 2016Q2 were
    published as sponsorship."""
    c = fold(classification).replace(" ", "")
    f = fold(purpose).replace(" ", "")
    if "PUBLICAC" in c or "LEGAL" in c or "LEGAL" in f:
        return "legal_notice"
    if "PATROCINIO" in c:
        return "sponsorship"
    # The classification cell comes out empty in the layouts that carry four label columns, and the
    # classification text then sits in the purpose column instead. Reading the markers from either
    # place recovers 24 rows worth R$16.3m that were published as "other" although they say
    # PROPAGANDA E PUBLICIDADE/VEICULACAO and /PRODUCAO.
    for text in (c, f):
        if "PRODUCAO" in text or "/PROD" in text:
            return "production"
        if ("VEICULA" in text or "/VEIC" in text or "PROPAGANDA" in text or "PUBLICIDADE" in text
                or "PROPEPUBL" in text):
            return "advertising"
    # Named sponsorship categories. The documents mix sponsorship rows into the advertising table,
    # so the table-level flag is off for them and 660 rows worth R$39.0m fell through to "other":
    # Esporte, Arte e Cultura, Entretenimento, Relacionamento Institucional, Causas Sociais. The
    # space-stripping above also folds the letter-spaced spellings ("E S P O RT E").
    if BRB_SPONSOR_CATEGORY.search(c):
        return "sponsorship"
    if sponsorship_table:
        return "sponsorship"
    return "other"


def brb_heading(text: str) -> tuple[str | None, int | None, int | None]:
    """(entity, year, quarter) from QDD heading text. The entity comes from the sentence
    '... a/o <entity> divulga abaixo ...' where present (one 2025 file's title names the finance
    company while that sentence and the listing name the brokerage), else from the title."""
    t = fold(text)
    qs = list(re.finditer(r"([1-4])\s*[O\xba\xb0]?\s*\(?(?:PRIMEIRO|SEGUNDO|TERCEIRO|QUARTO)?\)?\s*"
                          r"TRIMESTRE\s*(?:DE\s*)?/?\s*(20\d{2})", t))
    year = int(qs[-1].group(2)) if qs else None
    quarter = int(qs[-1].group(1)) if qs else None
    div = list(re.finditer(r"\b(?:A|O)\s+(BRB.{0,100}?)\s+DIVULGA\s+ABAIXO", t))
    if div:
        probe = div[-1].group(1)
    elif qs:
        probe = t[max(0, qs[-1].start() - 400):qs[-1].start()]
    else:
        probe = ""
    entity = next((e for e, pat in BRB_ENTITIES if re.search(pat, probe)), None)
    return entity, year, quarter


BRB_INT_RE = r"\d{1,3}(?:\.\d{3})+"          # thousands groups, cents omitted: 4.782.000
BRB_SHORT_CENTS_RE = r"(?:\d{1,3}(?:\.\d{3})*|\d+),\d"   # cents truncated to one digit
BRB_AMOUNT_RE = (r"-?R?\$?\s?-?(?:\d[\d.]*,\d{1,2}|" + BRB_INT_RE + r")")


def brb_cell_amount(cell) -> float | None:
    """BRB's gazette mixes three amount forms in one column: '2.094.000,00', '4.782.000' with the
    cents omitted, and '12.266.760,3' with the cents truncated to one digit. brl() takes only the
    first, so the other two were read as empty cells and 6,518,000.00 of 2025Q3 sponsorship went
    missing (caught by the published-month totals). The dot is a thousands separator in every
    Brazilian amount, never a decimal point, so a dotted integer is unambiguous."""
    s = re.sub(r"\s+", "", str(cell or "")).replace("R$", "")
    if s in ("", "None"):
        return None
    if re.fullmatch(r"-+", s):
        return 0.0
    neg = s.startswith("-")
    t = s.lstrip("-")
    if re.fullmatch(BRB_INT_RE, t) or re.fullmatch(BRB_SHORT_CENTS_RE, t):
        v = float(t.replace(".", "").replace(",", "."))
        return -v if neg else v
    return brl(s)


MONTH_KEYS = {k[:3].upper() for k in MONTHS_PT}


def brb_month_cols(row: list) -> dict[int, int]:
    cols = {}
    for j, c in enumerate(row):
        k = re.sub(r"[\s\-.]", "", str(c or ""))
        if k and len(k) <= 9 and fold(k)[:3] in MONTH_KEYS and month_number(k):
            cols[j] = month_number(k)
    return cols if len(cols) >= 2 else {}


def split_merged_amounts(row: list, cols: list[int], pattern: str = AMOUNT_RE) -> list:
    """'83.333,33 249.999,99' in one cell with the next column empty: the PDF merged two cells.
    `pattern` widens the amount form for a source that writes some amounts without cents."""
    row = list(row)
    for j in cols:
        if j >= len(row):
            continue
        parts = re.findall(pattern, str(row[j] or ""))
        if len(parts) == 2 and j + 1 < len(row) and not str(row[j + 1] or "").strip():
            row[j], row[j + 1] = parts[0], parts[1]
    return row


def brb_row_total_gap(row: list, amounts: dict[int, float | None],
                      total_col: int | None) -> float | None:
    """The row's own TOTAL cell minus the sum of its month cells, when both are published.

    A non-zero gap that no column shift explains is a defect in the SOURCE, not in the parse: in
    2018Q1 one PPR row prints 32.000,00 for January while its own TOTAL, and the published January
    total, both imply 537.258,64. Such a gap is imputed only under the strict rule in
    `brb_impute_forced_cells`; every other gap is reported and the row kept as printed."""
    if total_col is None or total_col >= len(row):
        return None
    stated = brb_cell_amount(row[total_col])
    present = [v for v in amounts.values() if v is not None]
    if stated is None or not present:
        return None
    gap = stated - sum(present)
    return gap if abs(gap) > 0.05 else None


def brb_shift_repair(row: list, amounts: dict[int, float | None],
                     month_cols: dict[int, int], total_col: int | None) -> dict[int, float | None]:
    """In the wide layouts the gazette merges label columns unevenly, so a row's amounts can land
    one cell to the right of the header's month columns (2021Q4: the header puts the months at
    5/7/9 while the publicidade legal rows and the VALOR TOTAL row carry them at 6/8/10). The
    repair is accepted only when the shifted amounts sum to the row's own TOTAL cell, which is
    what distinguishes a drifted row from a row that genuinely has empty months."""
    if total_col is None or total_col >= len(row):
        return amounts
    stated = brb_cell_amount(row[total_col])
    if stated is None:
        return amounts
    present = [v for v in amounts.values() if v is not None]
    if present and abs(sum(present) - stated) <= 0.5:
        return amounts
    alt = dict(amounts)
    for j, mo in month_cols.items():
        if alt.get(mo) is not None:
            continue
        k = j + 1
        if k < len(row) and k != total_col and k not in month_cols:
            alt[mo] = brb_cell_amount(row[k])
    shifted = [v for v in alt.values() if v is not None]
    if shifted and abs(sum(shifted) - stated) <= 0.5:
        return alt
    return amounts


BRB_SPLIT_GAP = 22.0          # pt: two printed lines belonging to one logical row
BRB_SIGN_GAP = 6.0            # pt: a minus sign this close to an amount belongs to it
# Page furniture the gazette prints on every page. It has to be excluded before lines are merged,
# not after: a 2016 page header sits within BRB_SPLIT_GAP of a real PPR row, and merging the two
# gave that row's three amounts (17,100.00 a month, 51,300.00 in total, so the arithmetic test
# passed) the header's text as its beneficiary, which also flipped it to the sponsorship series.
BRB_PAGE_FURNITURE = re.compile(r"DIARIO OFICIAL|PAGINA \d|ICP-BRASIL|DOCUMENTO ASSINADO"
                                r"|INFRAESTRUTURA DE CHAVES|WWW[.]|HTTP")
# The same furniture, but for `brb_recover_text_tables` only: plain "DIARIO OFICIAL" also matches
# "Diario Oficial do DF", a real BENEFICIARY in several 2016Q1 rows (the gazette itself, paid for
# publishing legal notices) - excluding it the way `brb_sweep_outside` does dropped three real rows
# (R$16,995.00) that each reconcile through their own TOTAL cell. The running header always
# completes "do Distrito Federal"; the beneficiary's short form never does.
BRB_TEXT_TABLE_FURNITURE = re.compile(r"DIARIO OFICIAL DO DISTRITO FEDERAL|PAGINA \d|ICP-BRASIL"
                                      r"|DOCUMENTO ASSINADO|INFRAESTRUTURA DE CHAVES|WWW[.]|HTTP")


@dataclass
class BrbContext:
    """What the BRB parser carries from one file to the next.

    BRB's listing links one gazette page under every entity that publishes on it (bank, finance
    company, brokerage), and the copies differ only in PDF metadata - the XMP packet and the trailer
    /ID - so neither the file name nor the byte hash identifies what was printed. Summing every copy
    counts BRB two or three times in seven quarters of 2016-2019. The unit is therefore the PRINTED
    ROW: the hash of its page's text, the table's position on that page and the row's position in
    the table.

    `pages` maps a page's text hash to where it was first read and whether every table on it could
    be read there. A fully read page is skipped in later files. A page read only in part - a table
    that continues from a page the first file does not contain (the 2019Q3 finance-company link
    serves page 39 of the 2018Q3 gazette alone, whose first table continues the bank's table from
    page 38) - is read again wherever the previous page is present, and only the rows not already
    emitted (`rows`) are emitted. `tables` holds each table's entity, year, publication and notes,
    keyed by where the table STARTS, so rows of one printed table share it whichever file they
    came from."""
    pages: dict[str, dict] = field(default_factory=dict)
    rows: set[str] = field(default_factory=set)
    tables: dict[str, dict] = field(default_factory=dict)
    line_scopes: set[str] = field(default_factory=set)
    imputations: list[dict] = field(default_factory=list)
    # The table state as it stood at the end of the last page read for each DODF issue, keyed by
    # that page's number: what lets a page numbered page+1 of the SAME issue continue the table
    # even when it is read from a different cached file (MAJOR A, 09-28 review). `state` is None
    # once the table has closed (its own VALOR TOTAL row was read) - there is then nothing to
    # continue, so a later page starting a new table is never mistaken for a continuation.
    continuations: dict[int, dict] = field(default_factory=dict)
    # Filing order for tables the gazette never dates (a bank-produced extract, a signature dated
    # "1o de <mes>" with no DODF issue, an "EXTRA" edition the running-header regex does not
    # parse): the order `parse_brb` is called in, i.e. the bank's own document listing order.
    file_order: int = 0
    # (entity, year, quarter) triples that already have a line or a total somewhere, checked by
    # `brb_recover_text_tables` before it registers a new table off a heading whose own table the
    # grid never read: an entity whose OTHER table on the SAME page already carries real data
    # (Cartao BRB's, e.g.) is never re-read a second way, only one that has none at all.
    observed: set[tuple[str, int, int]] = field(default_factory=set)


# A QDD notice opens with "PLANO ANUAL DE PUBLICIDADE/<year>" and the title "DEMONSTRATIVO DAS
# DESPESAS COM PROPAGANDA ...". Either line anchors a heading; other notices on the page (an
# "Extrato de Contrato" of the finance company, a licitacao of the bank) have neither, so the names
# they print never label a table.
BRB_ANCHOR = re.compile(r"DEMONSTRATIVODAS|PLANOANUALDEPUBLICIDADE")
BRB_ANCHOR_CHAIN = 6       # anchor lines within this many lines of each other form one heading block
BRB_HEAD_LOOKBACK = 4      # lines above a heading block searched for the entity's title line
BRB_TITLE_LINE = re.compile(r"(BRB|CARTAOBRB|FINANCEIRABRB)")


def brb_text_lines(page) -> list[dict]:
    """The page's printed lines, top to bottom, folded and with a whitespace-free copy."""
    out = []
    for r in rows_of(page):
        t = fold(row_text(r))
        out.append({"top": min(w["top"] for w in r), "bottom": max(w["bottom"] for w in r),
                    "text": t, "stripped": re.sub(r"\s", "", t)})
    return out


def brb_nearest_entity(text: str) -> str | None:
    """The entity named LAST in a whitespace-free text, i.e. the one nearest what follows it."""
    best = None
    for entity, pat in BRB_ENTITY_NAMES:
        for m in re.finditer(pat, text):
            if best is None or m.end() > best[0]:
                best = (m.end(), entity)
    return best[1] if best else None


def brb_heading_at(lines: list[dict], anchor_i: int, stop_top: float) -> dict:
    """Entity, year and quarter of the QDD heading whose nearest anchor line is lines[anchor_i].

    The heading block runs from its first anchor (PLANO ANUAL, the DEMONSTRATIVO title, the sentence
    that repeats it) down to `stop_top`, the top of the table it heads. It names its entity twice:
    in the sentence "... a/o <entity> divulga abaixo ..." and in the title line just above the block
    ("BRB - CREDITO, FINANCIAMENTO ...", "CARTAO BRB S.A", whose notice has no such sentence). A
    line qualifies as a title only if it starts with the name and carries no amount, so a signature
    ("Diretor Presidente da BRB-DTVM") never does. Where the two agree, or only one is printed, that
    is the entity. The source sometimes contradicts itself - the bank's 2022Q3 notice is titled
    "BRB - BANCO DE BRASILIA" and then says the finance company "divulga abaixo", and one 2025
    notice does the reverse - so a disagreement is returned as `conflict` for the caller to settle,
    never decided by a fixed preference for one of the two."""
    start = anchor_i
    while True:
        prev = [i for i in range(max(0, start - BRB_ANCHOR_CHAIN), start)
                if BRB_ANCHOR.search(lines[i]["stripped"])]
        if not prev:
            break
        start = min(prev)
    lo = max(0, start - BRB_HEAD_LOOKBACK)
    hi = next((i for i in range(anchor_i + 1, len(lines)) if lines[i]["top"] >= stop_top),
              len(lines))
    window = lines[lo:hi]
    # A name hyphenated across lines ("BANCO DE BRASI-" / "LIA S.A.", 2014Q4) is rejoined.
    joined = [re.sub(r"[-\u2013]$", "", ln["stripped"]) for ln in lines[lo:hi]]
    before = "".join(joined[:start - lo])
    text = "".join(joined)
    div = [m.start() for m in re.finditer(r"DIVULGA", text) if m.start() >= len(before)]
    said = brb_nearest_entity(text[max(0, div[-1] - 200):div[-1]]) if div else None
    titles = [ln for ln in window if BRB_TITLE_LINE.match(ln["stripped"])
              and not re.search(r"\d,\d{2}", ln["stripped"]) and brb_nearest_entity(ln["stripped"])]
    titled = brb_nearest_entity(titles[-1]["stripped"]) if titles else None
    conflict = [said, titled] if said and titled and said != titled else None
    entity = None if conflict else (said or titled)
    rule = ("heading_conflict" if conflict else "heading_divulga" if said
            else "heading_title" if titled else "heading_without_entity")
    _, year, quarter = brb_heading(" ".join(ln["text"] for ln in lines[start:hi]))
    return {"entity": entity, "conflict": conflict, "rule": rule, "year": year,
            "quarter": quarter, "text": " ".join(ln["text"] for ln in lines[start:hi])}


def brb_table_heading(lines: list[dict], anchors: list[int], table_top: float) -> dict | None:
    """The heading nearest above a table on its own page, or None where the page has none."""
    above = [i for i in anchors if lines[i]["top"] < table_top - 1]
    return brb_heading_at(lines, above[-1], table_top) if above else None


def brb_text_between(lines: list[dict], y0: float, y1: float) -> str:
    return " ".join(ln["text"] for ln in lines if y0 < ln["top"] < y1)


MONTHS_PT_RE = "|".join(MONTHS_PT)


def brb_publication(page_text: str) -> dict | None:
    """Date, issue and page of a gazette page, from its running header ("No 66, sexta-feira,
    6 de abril de 2018 PAGINA 37", the No written with an ordinal or a degree sign, an "EXTRA"
    edition's issue written "No 66 - EXTRA," before the weekday). Pages the bank produced itself
    carry none - a bank-produced extract, a signature dated "1o de <mes> de <year>" with no DODF
    issue, or the like; those go through `brb_finish`'s filing-order fallback instead, since this
    function only ever returns an actual gazette date, never a guess."""
    t = fold(page_text)
    m = re.search(r"\bN\s*[O0\xb0]\.?\s*(\d{1,4})\s*(?:[\-–]?\s*EXTRA\s*)?,\s*[A-Z]+(?:-FEIRA)?"
                  r"\s*,\s*(\d{1,2})\s*DE\s*" rf"({MONTHS_PT_RE.upper()})\s*DE\s*(20\d{{2}})", t)
    if not m:
        return None
    p = re.search(r"PAGINA\s*(\d{1,4})", t)
    date = f"{m.group(4)}-{MONTHS_PT[m.group(3).lower()]:02d}-{int(m.group(2)):02d}"
    return {"date": date, "issue": int(m.group(1)), "page": int(p.group(1)) if p else None,
            "label": f"DODF {m.group(1)}" + (f" p.{p.group(1)}" if p else "") + f" ({date})"}


# The word that must follow TOTAL/TOTAIS once the label's whitespace has been stripped, so
# "TOTALIZADOR", "TOTALMENTE" or a beneficiary named "Totalcred" never pass as a total row: either
# nothing (end of the cell), any single character that is not itself an uppercase letter -
# punctuation (a trailing "." some tables print after every label, a colon, a dash), a digit (a
# year or a "1o TRIMESTRE" heading fused onto the word by the whitespace strip) - or one of the
# words the gazette actually appends to TOTAL in this table ("TOTAL R$", "TOTAL PAGO", "TOTAL
# REALIZADO", "TOTAL CONTABILIZADO", "TOTAL GERAL", "TOTAL ORCADO", "VALOR TOTAL NO 2o TRIMESTRE",
# "Total Recebido"), which start with a letter and so are not caught by the single-character rule.
BRB_TOTAL_SUFFIX_RE = re.compile(
    r"^(?:VALOR)?(?:SUB-?)?(?:TOTAL|TOTAIS)"
    r"(?:$|[^A-Z]|R\$|GERAL|PAGO|REALIZADO|CONTABILIZADO|ORCADO|RECEBIDO|NO)")


def brb_is_total_row(labels: list[str]) -> bool:
    """A printed total or subtotal row.

    The test runs on the FIRST label cell, or on any single cell that is the row's only non-blank
    label - the gazette's one-row tables print a "TOTAL R$" cell beside a beneficiary and purpose
    that state what the total is for, with nothing else printed for that row (see
    `brb_group`); testing every cell there read the SAME word twice, but testing only the first
    cell missed the ones printed with the total in cell 2 or 3. Each candidate cell is folded and
    stripped of whitespace and leading punctuation, so the letter-spaced "TO TA L" of 2016 and the
    brokerage's "SUBTOTAL 1o TRIMESTRE" rows still read as totals. Read as detail rows instead,
    2016Q2's bank "TO TA L" row added R$5.49m and the SUBTOTAL rows doubled the brokerage's table
    in ten quarters (2021Q3-2025Q1). A required word boundary after TOTAL/TOTAIS
    (`BRB_TOTAL_SUFFIX_RE`) keeps a beneficiary or purpose that merely starts with a look-alike word
    from matching; a bare prefix match let "TOTALIZADOR ..." beneficiaries through undetected."""
    def hit(lab: str) -> bool:
        s = re.sub(r"^[^A-Z0-9]+", "", re.sub(r"\s", "", fold(lab)))
        return bool(BRB_TOTAL_SUFFIX_RE.match(s))

    if labels and hit(labels[0]):
        return True
    non_blank = [i for i, lab in enumerate(labels) if str(lab or "").strip()]
    return len(non_blank) == 1 and hit(labels[non_blank[0]])


def brb_outside_candidate(words: list[dict], ref: dict) -> dict | None:
    """Month amounts, TOTAL amount and label of a candidate row, read by column position.

    The minus sign of a negative amount is often its own word ("-R$" then "1.493,07"), so the token
    before each amount is inspected. Losing it is invisible to the reconciliation test, because a
    row's total loses the sign with its months: a 2020Q4 'Glosa ao Pagamento' of -1,493.07 passed
    the test as +1,493.07 and pushed December 2,986.14 over its published total.
    """
    amounts, stated_total, labels = {}, None, []
    ordered = sorted(words, key=lambda x: x["x0"])
    for i, w in enumerate(ordered):
        text = w["text"]
        if not re.fullmatch(r"R[$]|[0-9.,]+", text):
            labels.append(text)
        # a token with no separator (a bare year in a label) is not an amount
        if not re.search(r"[.,]", text):
            continue
        v = brb_cell_amount(text)
        if v is None:
            continue
        # Only an ADJACENT sign token counts. A lone dash far to the left is the previous column's
        # published zero, not a minus: in 2021Q3 the dash of an empty cell sits 240pt from the
        # amount, while the minus of "-R$ 1.493,07" sits 2pt from it. Negating on the dash turned
        # real rows negative, and they were then refused by the reconciliation test.
        if v > 0 and i and re.fullmatch(r"-|-R[$]|R[$]-|[(]", ordered[i - 1]["text"]) \
                and w["x0"] - ordered[i - 1]["x1"] <= BRB_SIGN_GAP:
            v = -v
        xm = (w["x0"] + w["x1"]) / 2
        for j, x in ref["col_x"].items():
            if not (x[0] - 2 <= xm <= x[1] + 2):
                continue
            if j in ref["month_cols"]:
                if ref["month_cols"][j] in amounts:
                    return None           # two amounts in one month: these are separate rows
                amounts[ref["month_cols"][j]] = v
            elif j == ref["total_col"]:
                if stated_total is not None:
                    return None
                stated_total = v
    label = " ".join(labels).strip()
    if not amounts or not label:
        return None
    return {"amounts": amounts, "total": stated_total, "label": label}


def brb_outside_ok(cand: dict) -> bool:
    """A candidate is admitted only when its TOTAL column equals the sum of its month cells.

    That test is what keeps contract prose out. The gazette pages carry notices whose figures land
    inside the month columns by accident ("Valor Total: R$ 1.443.398,16", "Nota de Empenho"), and
    four of them were being read as sponsorship in the 2016 files, which publish no month totals to
    catch it. Cents are truncated to one digit in some cells, hence the 0.5 tolerance.
    """
    if cand["total"] is None:
        return False
    if re.search(r"(^|[^A-Z])TOTAL([^A-Z]|$)|VALOR OR", fold(cand["label"])):
        return False
    return abs(cand["total"] - sum(cand["amounts"].values())) <= 0.5


def brb_cluster_rows(wrows: list[list[dict]], gap: float = BRB_SPLIT_GAP
                     ) -> list[list[tuple[float, list]]]:
    """Groups a page's printed word-lines into logical-row candidates: consecutive lines within
    `gap` of each other are tried merged before being tried alone in `brb_try_accept_row`, because
    one logical row can print as two lines (a month cell on one, the row TOTAL on the other -
    2021Q3: a Bosque Formosa row with July below and its total on the line above)."""
    tops = sorted(((min(w["top"] for w in r), r) for r in wrows), key=lambda t: t[0])
    clusters: list[list[tuple[float, list]]] = []
    for top, wrow in tops:
        if clusters and top - clusters[-1][-1][0] <= gap:
            clusters[-1].append((top, wrow))
        else:
            clusters.append([(top, wrow)])
    return clusters


def brb_try_accept_row(out: Parsed, doc: Doc, cl: list[tuple[float, list]], ref: dict, pno: int,
                       page_key: str, keep_row, key_ns: str, source: str,
                       method: str | None = None, tidy: bool = False) -> bool:
    """Tries one clustered candidate row against `ref`'s columns, admitting it only through its
    own TOTAL column (`brb_outside_ok`) and emitting its month amounts as lines on success; a
    candidate that does not reconcile is reported in a note and left out, never forced in. Shared
    by `brb_sweep_outside` (rows lying outside every table pdfplumber's finder returns) and
    `brb_recover_text_tables` (a table's own rows, read by column position because its grid never
    gave it a usable header - `key_ns` keeps their row keys from ever colliding). `source` names
    what is being read, for the notes; `method` words it for `parse_method` where that differs
    (default: `source`). `tidy` drops the dashes of unprinted month cells that trail a label."""
    merged = brb_outside_candidate([w for _, wr in cl for w in wr], ref)
    singles = [brb_outside_candidate(wr, ref) for _, wr in cl] if len(cl) > 1 else []
    accepted = ([merged] if merged and brb_outside_ok(merged)
                else [c for c in singles if c and brb_outside_ok(c)])
    if not accepted:
        for c in [merged] + singles:
            if c is None or re.search(r"(^|[^A-Z])TOTAL([^A-Z]|$)|VALOR OR", fold(c["label"])):
                continue
            shown = "no total" if c["total"] is None else f"{c['total']:,.2f}"
            out.notes.append(
                f"{ref['scope']}: a row {source} on p{pno} does not reconcile with its TOTAL "
                f"column and was left out ({c['label'][:40]}; "
                f"months {sum(c['amounts'].values()):,.2f} vs {shown})")
        return False
    for k, c in enumerate(accepted):
        where = "merged from two printed lines" if c is merged and len(cl) > 1 else "one line"
        row_key = f"{page_key}:{key_ns}{int(round(cl[0][0]))}:{k}"
        if not keep_row(row_key):
            continue
        label = re.sub(r"(?:\s+-)+$", "", c["label"]) if tidy else c["label"]
        for mo, v in c["amounts"].items():
            rec = line(
                doc, period=f"{ref['year']}{mo:02d}",
                frequency="monthly", period_rule="document month column",
                section="PATROCINIOS" if ref["sponsorship"] else "PUBLICIDADE",
                category=label[:60], detail=label,
                # the recovered row has no separate classification cell, so its label serves as
                # both, and brb_group's own rules outrank the table-level guess
                category_group=brb_group(label, label, bool(ref["sponsorship"])),
                amount_brl=v, amount_kind="total", reversal=v < 0,
                total_scope=f"{ref['scope']}:{mo:02d}",
                extra_scopes=[f"{ref['scope']}:quarter"],
                line_id=f"{row_key}:m{mo:02d}", content_key=f"{row_key}:m{mo:02d}",
                table_id=ref["scope"], page_or_sheet=f"p{pno}",
                parse_method=f"pdfplumber words, {method or source} ({where})")
            rec["_table"], rec["_month"] = ref["scope"], mo
            out.lines.append(rec)
        out.notes.append(f"{ref['scope']}: recovered a row {source} on p{pno}, {where}, "
                         f"TOTAL column reconciles ({c['label'][:40]})")
    return True


def brb_sweep_outside(out: Parsed, doc: Doc, page, pno: int, state: dict, snap: dict,
                      page_key: str, keep_row) -> None:
    """Recover rows lying outside every table the finder returns, admitting only those that prove
    themselves through their own TOTAL column (`brb_outside_ok`, via `brb_try_accept_row`).

    The ruling lines stop short of rows they belong to, and the cut falls at either end - a 2025Q3
    Capital Clube row sat ABOVE its box - so no geometric rule finds these rows.

    A row above the page's first table continues the table that ended on the previous page, so it
    takes the scope and section in force when the page began. Attributing it to the current table
    instead would move a row between the advertising and the sponsorship series. Its entity and
    period are the table's, set once every file has been read (`brb_finish`).

    `keep_row(row_key)` says whether this printed row is new to the bank (see `BrbContext`); the
    key is the page's text hash plus the row's position on the page, so a copy of the page in
    another file yields the same key.
    """
    boxes = [t.bbox for t in page.find_tables()]
    first_top = min((b[1] for b in boxes), default=None)
    wrows = []
    for wrow in rows_of(page):
        top = min(w["top"] for w in wrow)
        if any(b[1] - 2 <= top <= b[3] + 2 for b in boxes):
            continue
        if BRB_PAGE_FURNITURE.search(fold(row_text(wrow))):
            continue
        wrows.append(wrow)

    for cl in brb_cluster_rows(wrows):
        above = first_top is not None and cl[0][0] < first_top
        ref = snap if above else state
        if not (ref.get("month_cols") and ref.get("col_x") and ref.get("scope")):
            continue
        brb_try_accept_row(out, doc, cl, ref, pno, page_key, keep_row, key_ns="sweep",
                           source="outside the detected table",
                           method="row outside any detected table")


# A notice's own closing total line, its TABLE'S quarter, never the annual figure a few lines
# below it ("TOTAL PAGO EM <year>", which names no trimester and would otherwise match "TOTAL...
# PAGO" too) -- requiring TRIMESTRE as well keeps the two apart regardless of which one a search
# meets first.
BRB_TEXT_TOTAL_RE = re.compile(r"TOTAL.{0,20}(PAGO|REALIZADO|CONTABILIZADO).{0,20}TRIMESTRE")
# A single printed token that is unambiguously an amount: comma decimals or dotted-thousands
# grouping, or a bare dash for a published zero. `brb_cell_amount` alone is not used to pick these
# out of a row's words because its own last, most permissive branch also accepts a bare integer
# ("2016", a year sitting a few words away on the same or a neighbouring line) as 2016.00.
BRB_MONEY_WORD_RE = re.compile(r"-?(?:\d{1,3}(?:\.\d{3})+,\d{1,2}|\d+,\d{1,2}|\d{1,3}(?:\.\d{3})+)")


def brb_is_money_word(text: str) -> bool:
    return bool(re.fullmatch(r"-+", text) or BRB_MONEY_WORD_RE.fullmatch(text))


def brb_find_quarter_total_row(wrows: list[list[dict]]) -> list[dict] | None:
    """The FIRST row (top to bottom) whose words read 'TOTAL ... PAGO/REALIZADO/CONTABILIZADO ...
    TRIMESTRE': a notice's own closing total, which the gazette always prints in full even where
    the table around it has no column header pdfplumber's grid can read (`brb_recover_text_tables`).
    """
    for wr in wrows:
        if BRB_TEXT_TOTAL_RE.search(fold(re.sub(r"\s", "", row_text(wr)))):
            return wr
    return None


def brb_col_bands(xmids: list[float], pad: float = 45.0) -> dict[int, tuple[float, float]]:
    """Splits the gaps between a row's own amount x-midpoints in half, giving the outermost ones a
    fixed fallback pad -- a Voronoi split standing in for column boundaries a table recovered this
    way never had pdfplumber read (`brb_recover_text_tables`)."""
    bands = {}
    for i, x in enumerate(xmids):
        left = (xmids[i - 1] + x) / 2 if i > 0 else x - pad
        right = (xmids[i + 1] + x) / 2 if i + 1 < len(xmids) else x + pad
        bands[i] = (left, right)
    return bands


def brb_total_basis(label: str, basis: str | None) -> str | None:
    """The basis a total row's own label states: 'TOTAL CONTABILIZADO...' accrued, 'TOTAL
    PAGO...' paid, 'TOTAL REALIZADO...' undocumented unless the table already has a basis."""
    f = fold(label)
    if "CONTABILIZADO" in f:
        return "accrued"
    if "PAGO" in f:
        return "paid"
    if "REALIZADO" in f and basis is None:
        return "undocumented"
    return basis


BRB_CID_DASH_RE = re.compile(r"\s*\(cid:177\)\s*")


def brb_stacked_records(page, tobj, col_x: dict, first_m: int, amount_cols: list[int],
                        body_top: float) -> dict | None:
    """Labels of a table whose records print stacked over several grid rows (the 2016Q1 bank and
    brokerage tables): the amounts and the purpose cell sit on one printed line, the beneficiary and
    classification cells start a line above it and finish a line or two below it, and the grid
    cuts the printed line into rows whose label cells are blank, so a row read alone inherits the
    previous record's labels.

    A record is the printed line carrying money in the amount columns. The label words between two
    such lines are split at the widest vertical gap between them (a tie within half a point goes to
    the cut nearest the midpoint), each side going to its own record, then sorted into the label
    columns by x position. Returns `tops` (the record lines' y positions), `labels` (one list of
    `first_m` strings per record, '(cid:177)', a dash whose glyph has no Unicode map, read as a
    hyphen) and `money` (the amount words, for `brb_stacked_center`)."""
    x_lo, x_hi = tobj.bbox[0] - 1, tobj.bbox[2] + 1
    words = [w for w in page.extract_words(x_tolerance=1.5, y_tolerance=2)
             if body_top - 1 <= w["top"] <= tobj.bbox[3] + 1 and x_lo <= xmid(w) <= x_hi]
    amt_x = min((col_x[j][0] for j in amount_cols if j in col_x), default=None)
    if amt_x is None:
        return None
    amt_x -= 2
    lab = [w for w in words if xmid(w) < amt_x]
    money = sorted((w for w in words if xmid(w) >= amt_x and brb_is_money_word(w["text"])),
                   key=lambda w: w["top"])
    lines: list[list[dict]] = []
    for w in money:
        if lines and w["top"] - lines[-1][-1]["top"] <= 2.0:
            lines[-1].append(w)
        else:
            lines.append([w])
    if not lines:
        return None
    tops = [sum(w["top"] for w in ln) / len(ln) for ln in lines]
    positions = sorted({round(w["top"], 1) for w in lab} | {round(t, 1) for t in tops})
    cuts = []
    for a, b in zip(tops, tops[1:]):
        between = [p for p in positions if a - 0.6 <= p <= b + 0.6]
        gaps = [(q - p, (p + q) / 2) for p, q in zip(between, between[1:])]
        mid = (a + b) / 2
        widest = max((g for g, _ in gaps), default=0.0)
        cuts.append(min((m for g, m in gaps if g >= widest - 0.5), default=mid,
                        key=lambda m: abs(m - mid)))
    # label columns: the header's own cells, the one the grid left out (None) filled from its
    # neighbours
    bands, lo = [], x_lo
    for j in range(first_m):
        if j in col_x:
            lo_j, hi_j = col_x[j]
        else:
            lo_j = lo
            hi_j = next((col_x[k][0] for k in range(j + 1, first_m) if k in col_x), amt_x)
        bands.append((lo_j, hi_j))
        lo = hi_j
    cells: list[list[list[dict]]] = [[[] for _ in range(first_m)] for _ in tops]
    for w in lab:
        k = sum(1 for c in cuts if w["top"] > c)
        j = next((i for i, (b0, b1) in enumerate(bands) if b0 - 0.5 <= xmid(w) < b1),
                 min(range(first_m), key=lambda i: abs(xmid(w) - bands[i][0])))
        cells[k][j].append(w)

    def text_of(ws: list[dict]) -> str:
        ws = sorted(ws, key=lambda w: w["top"])
        out, run = [], []
        for w in ws:
            if run and w["top"] - run[-1]["top"] > 1.0:
                out += [x["text"] for x in sorted(run, key=lambda x: x["x0"])]
                run = []
            run.append(w)
        out += [x["text"] for x in sorted(run, key=lambda x: x["x0"])]
        t = BRB_CID_DASH_RE.sub(" - ", " ".join(out))
        return re.sub(r"\s*-\s*$", "", t).strip()

    return {"tops": tops, "money": money,
            "labels": [[text_of(ws) for ws in rec] for rec in cells]}


def brb_stacked_center(stack: dict, tobj, ridx: int, row: list, amount_cols: list[int]
                       ) -> int | None:
    """Index of the stacked record a grid row's amount cells belong to: the amount word the row's
    own first non-blank amount cell holds, matched to the nearest record line."""
    cells = tobj.rows[ridx].cells
    for j in amount_cols:
        if j >= len(row) or j >= len(cells) or cells[j] is None or not str(row[j] or "").strip():
            continue
        c = cells[j]
        for w in stack["money"]:
            mid = (w["top"] + w["bottom"]) / 2
            if c[0] <= xmid(w) <= c[2] and c[1] - 0.5 <= mid <= c[3] + 0.5:
                k = min(range(len(stack["tops"])), key=lambda i: abs(stack["tops"][i] - w["top"]))
                return k if abs(stack["tops"][k] - w["top"]) <= 2.5 else None
    return None


def brb_recover_text_tables(out: Parsed, doc: Doc, path: Path, pub: dict | None, page, pno: int,
                            lines: list[dict], anchors: list[int], page_key: str, ctx: BrbContext,
                            snap: dict, keep_row) -> bool | None:
    """Reads a page's tables by column position where pdfplumber's grid could not be trusted for
    them: a table object mixed another notice's heading or a multi-line merged cell into what
    should be one entity's own rows, and `parse_brb`'s per-table-object loop bailed out of the
    per-cell read rather than risk it (2016Q1 DODF p.12: the brokerage's closing rows and the
    finance company's WHOLE notice, header included, land in one table object; pdfplumber cuts a
    wrapped label cell's own continuation into a spurious extra non-blank cell there, which can
    defeat `brb_is_total_row`'s one-cell rule for the brokerage's own total row too).

    Two things can need this on one page:

    1. A table that was already open when this page started (`snap`, the state as of page start)
       and never closes here via the grid. Read by column position using the SAME columns the
       table's real header already gave it (`snap['col_x']`), never a guess.

    2. A QDD heading on this page whose own table has produced no line and no total anywhere in
       this run (`ctx.observed`) - an entity the grid never read at all (2016Q1's finance company:
       its own column header misprints OUTUBRO/NOVEMBRO/DEZEMBRO where JANEIRO/FEVEREIRO/MARCO
       belong, so months are taken from the heading's own quarter instead, not
       from that header). An entity that already has SOME data from elsewhere on the page
       (Cartao BRB's table, which the grid does read) is left alone; only a genuinely unread table is registered here.

    Either recovery first finds the table's own closing total row by text
    (`brb_find_quarter_total_row`); the table is left unobserved and reported, never guessed at,
    where that row cannot be found or does not print exactly one amount per month plus its own
    total. The table's basis is the one that row's label states (`brb_total_basis`), the same rule
    the grid applies to a total row it reads cell by cell. Body rows in the table's span are then read the same way `brb_sweep_outside` reads rows
    outside a detected table (`brb_try_accept_row`, shared), admitting each only through its own
    TOTAL cell.

    Returns whether every table this call was responsible for on this page is now fully accounted
    for (closed, or registered and read) - `parse_brb` ORs this into the page's own `complete`, so
    a page whose grid read bailed out but that this function fully recovers is not reported as
    unread in `documents.pages_unread` on top of being numerically correct.
    """
    wrows = rows_of(page)
    # `attempted` is set only where this call actually takes responsibility for a table (either
    # branch below); a page with nothing for either branch to do returns None, so the caller's
    # `complete` is left exactly as the grid read computed it, never forced True by a call that
    # did nothing.
    attempted, ok = False, True

    def span_rows(top: float, bottom: float) -> list[list[dict]]:
        picked = []
        for wr in wrows:
            t = min(w["top"] for w in wr)
            if top <= t < bottom and not BRB_TEXT_TABLE_FURNITURE.search(fold(row_text(wr))):
                picked.append(wr)
        return picked

    def month_amounts(total_row: list[dict], n_months: int) -> tuple[list[int], list[float]] | None:
        money = sorted((w for w in total_row if brb_is_money_word(w["text"])), key=xmid)
        if len(money) != n_months + 1:
            return None
        return [xmid(w) for w in money], [brb_cell_amount(w["text"]) for w in money]

    # Heading blocks on this page, top to bottom, one per distinct QDD notice: anchors chained
    # within BRB_ANCHOR_CHAIN belong to the SAME block (brb_heading_at walks back to a chain's own
    # start from any of its members, so the chain's LAST member is a sufficient representative),
    # each block's own window bounded by the NEXT block's top so two notices sharing a page are
    # never read into each other.
    groups: list[int] = []
    for i in anchors:
        if groups and i - groups[-1] <= BRB_ANCHOR_CHAIN:
            groups[-1] = i
        else:
            groups.append(i)
    blocks = []
    for gi, g in enumerate(groups):
        # `brb_heading_at`'s own `stop_top` is meant to be a TIGHT bound - the table it heads, a
        # few lines down - never a distant next notice's own top: passing the NEXT block's anchor
        # here let the window run all the way through this notice's own body and signature and
        # into the FOLLOWING notice's preamble title line ("SECRETARIA DE ESTADO DE FAZENDA /
        # CARTAO BRB S.A", printed just above Cartao's own heading), which brb_nearest_entity then
        # read as a conflicting title for THIS heading (financeira vs cartao) - entity resolution
        # stays capped to a fixed, generous line count a real heading's own prose never exceeds
        # (2016Q1's longest runs about 6 lines from chain start to its "divulga" sentence).
        stop = min(lines[groups[gi + 1]]["top"] if gi + 1 < len(groups) else 1e9,
                  lines[min(g + 20, len(lines) - 1)]["top"] + 1)
        blocks.append({"top": lines[g]["top"], "head": brb_heading_at(lines, g, stop)})

    # (1) close a table that was already open when this page started, if its own total sits before
    # this page's first heading (or anywhere on a page with none of its own). An open, unclosed
    # continuation is completely normal by itself (any multi-page table looks like this until its
    # own later page closes it on the grid) - gated to 2016Q1, the one quarter this was verified
    # against page by page, so it is never tried on an ordinary other-year continuation page whose
    # own later page would have closed it on the grid anyway (see `parse_brb`'s bail-out docstring).
    # `snap["closed"]` is only ever False here (a continuation is carried across files exactly
    # because it was NOT closed as of the page it was carried from - see BrbContext.continuations);
    # it says nothing about whether THIS page's own, ordinary per-cell grid read - which runs
    # before this function is called, on every tobj that does not bail out - already closed the
    # table normally a few tobjs later on the SAME page (2016Q1's bank: table0 reads perfectly
    # clean and closes with its own TOTAL row; table1 is what bails out, over DTVM's heading).
    # Recovering it again here, by column position, would double every row grid already read
    # correctly, so this checks THIS page's own `out.totals` for a quarter scope the grid already
    # closed and skips a table that closed itself that way.
    already_closed_here = any(t["scope"] == f"{snap.get('scope')}:quarter" for t in out.totals)
    if (snap.get("scope") and snap.get("col_x") and snap.get("month_cols")
            and not snap.get("closed") and not already_closed_here
            and snap.get("year") == 2016 and snap.get("quarter") == 1):
        first_top = blocks[0]["top"] if blocks else 1e9
        rows_here = span_rows(0, first_top)
        total_row = brb_find_quarter_total_row(rows_here)
        if total_row is None:
            attempted, ok = True, False
        elif keep_row(f"{page_key}:qtotal:{snap['scope']}"):
            attempted = True
            months = sorted(snap["month_cols"].values())
            got = month_amounts(total_row, len(months))
            if got is None:
                ok = False
                out.notes.append(
                    f"{snap['scope']}: found a 'TOTAL...TRIMESTRE' row on p{pno} by column "
                    f"position but it does not print exactly {len(months) + 1} amounts; left "
                    "unobserved rather than guessed")
            else:
                _, vals = got
                own = ctx.tables[snap["scope"]]
                own["basis"] = brb_total_basis(row_text(total_row), own["basis"])
                out.totals.append({
                    "scope": f"{snap['scope']}:quarter", "stated": vals[-1],
                    "label": "quarter total (read by column position)", "_table": snap["scope"],
                    "_month": None, "content_key": f"{page_key}:qtotal:{snap['scope']}:quarter"})
                for mo, v in zip(months, vals[:-1]):
                    out.totals.append({
                        "scope": f"{snap['scope']}:{mo:02d}", "stated": v,
                        "label": f"month {mo} (read by column position)", "_table": snap["scope"],
                        "_month": mo, "content_key": f"{page_key}:qtotal:{snap['scope']}:m{mo:02d}"})
                out.notes.append(
                    f"{snap['scope']}: closed by column position on p{pno} (TOTAL column "
                    f"{vals[-1]:,.2f}); its table object mixed in another notice's text or a "
                    "multi-line merged cell and could not be read cell by cell")
            # Body rows are only swept here alongside an ACTUAL close: a page in the middle of a
            # multi-page continuation, with no closing total of its own, has no bounded span to
            # read this way (the table's OTHER entity is still to come, not yet on this page), and
            # sweeping the whole page up to a same-entity running-header match would double up
            # rows already read correctly on the per-cell grid path.
            for cl in brb_cluster_rows([wr for wr in rows_here if wr is not total_row]):
                brb_try_accept_row(out, doc, cl, snap, pno, page_key, keep_row,
                                   key_ns="text", source="of a table read by column position",
                                   tidy=True)

    # (2) each heading on this page whose table the grid never read at all. Gated to 2016Q1 for the
    # same reason as (1): a heading whose table produced nothing YET on this page can be genuinely
    # unread (2016Q1's finance company), or it can simply be a table this file has not reached the
    # body of yet (its detail rows are on a LATER page still to come) - the two are told apart, for
    # every other quarter, by the existing `entity_rule` logic already on the grid path.
    for gi, b in enumerate(blocks):
        head = b["head"]
        entity = head["entity"] or (doc.entity if head["conflict"] and doc.entity in
                                    (head["conflict"] or ()) else None)
        if not (entity and head["year"] == 2016 and head["quarter"] == 1):
            continue
        key = (entity, head["year"], head["quarter"])
        if key in ctx.observed:
            continue
        bottom = blocks[gi + 1]["top"] if gi + 1 < len(blocks) else 1e9
        rows_here = span_rows(b["top"], bottom)
        total_row = brb_find_quarter_total_row(rows_here)
        months = [head["quarter"] * 3 - 2, head["quarter"] * 3 - 1, head["quarter"] * 3]
        if total_row is None:
            attempted, ok = True, False
            out.notes.append(
                f"{entity} {head['year']}Q{head['quarter']}: no table found on p{pno} by "
                "find_tables, and no 'TOTAL...TRIMESTRE' row found by column position either; "
                "left unobserved")
            continue
        got = month_amounts(total_row, len(months))
        if got is None:
            attempted, ok = True, False
            out.notes.append(
                f"{entity} {head['year']}Q{head['quarter']}: its 'TOTAL...TRIMESTRE' row on "
                f"p{pno} does not print exactly {len(months) + 1} amounts; left unobserved "
                "rather than guessed")
            continue
        new_scope = f"{page_key[:12]}#text{gi}"
        if not keep_row(f"{page_key}:qtotal:{new_scope}"):
            continue
        attempted = True
        xmids, vals = got
        body_rows = [wr for wr in rows_here if wr is not total_row]
        prose = fold(" ".join(w["text"] for wr in body_rows for w in wr))
        sponsorship = not re.search(r"PROPAGANDA|PUBLICIDADE|PUBLICAC|PROP E PUBL", prose)
        ref = {"scope": new_scope, "col_x": brb_col_bands(xmids),
              "month_cols": dict(enumerate(months)), "total_col": len(months),
              "year": head["year"], "quarter": head["quarter"], "sponsorship": sponsorship}
        brb_register_table(ctx, new_scope, {
            "entity": entity, "entity_rule": f"{head['rule']}_text_column_position",
            "year": head["year"], "quarter": head["quarter"], "heading": head["text"],
            "published": pub["date"] if pub else None,
            "publication": pub["label"] if pub else None,
            "first_read": f"{path.name} p{pno}",
            "listing_entity": doc.entity, "notice": "", "tails": set(),
            "basis": brb_total_basis(row_text(total_row), None),
            "total_is_sum": False, "listing_year": doc.year, "listing_quarter": doc.quarter,
            "file_order": ctx.file_order})
        out.totals.append({
            "scope": f"{new_scope}:quarter", "stated": vals[-1],
            "label": "quarter total (read by column position)", "_table": new_scope,
            "_month": None, "content_key": f"{page_key}:qtotal:{new_scope}:quarter"})
        for mo, v in zip(months, vals[:-1]):
            out.totals.append({
                "scope": f"{new_scope}:{mo:02d}", "stated": v,
                "label": f"month {mo} (read by column position)", "_table": new_scope,
                "_month": mo, "content_key": f"{page_key}:qtotal:{new_scope}:m{mo:02d}"})
        ctx.observed.add(key)
        out.notes.append(
            f"{new_scope}: {entity} {head['year']}Q{head['quarter']} read by column position on "
            f"p{pno} (TOTAL column {vals[-1]:,.2f}); its table object mixed another notice's "
            "heading into its own rows and could not be read cell by cell")
        for cl in brb_cluster_rows(body_rows):
            brb_try_accept_row(out, doc, cl, ref, pno, page_key, keep_row,
                               key_ns=f"text{gi}", source="of a table read by column position",
                               tidy=True)
    return ok if attempted else None


def brb_filing_order(meta: dict) -> tuple:
    """One comparable filing-order key for a table, real date or not.

    The table's own (year, quarter) - read from its QDD heading where the page has one, the
    listing's claim otherwise (`parse_brb`) - dominates the comparison, so a Q2 filing always
    outranks a Q1 filing of the same year whether or not either carries a real gazette date - the
    user's rule that an undated table (a bank-produced extract, an unparsed "1o de <mes>" signature,
    an "EXTRA" issue the running-header regex does not parse) is ordered by its quarter, not left
    off the timeline. This must be the HEADING's (year, quarter), never the raw listing's
    (`meta["listing_year"/"listing_quarter"]`): the listing can point at the wrong gazette issue (a
    2019Q3 finance-company link that serves the 2018Q3 page - `parse_brb`), and ordering by that
    raw claim made the mislabelled 2018Q3 copy outrank the genuinely later 2019-01 filing that
    actually corrects 2018Q3. The real gazette date, where read, then breaks a tie between two
    filings of the same (year, quarter), ahead of the file-read-order fallback, which breaks a tie
    between two filings that are ALSO undated."""
    return (meta.get("year") or 0, meta.get("quarter") or 0,
            meta.get("published") or "", meta.get("file_order", 0))


def brb_register_table(ctx: BrbContext, scope: str, meta: dict) -> None:
    """Record a table's metadata at its start. A table first met in a file that lacks its heading
    (entity 'unknown') takes the heading another file supplies; anything else is fixed at the
    first reading, because the same printed page gives the same reading.

    A table first met in a file that has no QDD heading anywhere prints its entity from the
    LISTING instead (`listing_document_without_heading`; see `parse_brb`), which is a guess about
    what the page holds, not a reading of it. It is treated the same way as 'unknown': a later copy
    of the SAME page that does carry a heading (or an unambiguous conflict the listing settles)
    upgrades it, so the listing's guess never outranks the page's own text."""
    have = ctx.tables.get(scope)
    if have is None:
        ctx.tables[scope] = meta
        return
    have_weak = have["entity"] == "unknown" or have["entity_rule"] == "listing_document_without_heading"
    new_strong = meta["entity"] != "unknown" and meta["entity_rule"] != "listing_document_without_heading"
    if have_weak and new_strong:
        for k in ("entity", "entity_rule", "year", "quarter", "heading"):
            have[k] = meta[k]


def parse_brb(doc: Doc, path: Path, ctx: BrbContext | None = None) -> Parsed:
    """QDD 'Demonstrativo das despesas com propaganda, publicidade, publicacoes legais e
    patrocinios'. Tables have BENEFICIARIO, CLASSIFICACAO DA DESPESA, FINALIDADE DA ACAO, month
    columns and TOTAL (R$), and close with a total row carrying month values ('TOTAL R$',
    'Valor Total', 'TOTAL PAGO NO 4o TRIMESTRE') followed by cumulative rows ('TOTAL PAGO NO
    1o TRIMESTRE', 'TOTAL CONTABILIZADO EM 2025', 'TOTAL REALIZADO ...') that state the basis.

    The table-reading logic runs over every page, since a table can continue from the previous
    page, but a row is emitted only where it is new to the bank (`BrbContext`). A table's entity
    comes from the QDD heading nearest above it on its own page, or - for a table that opens a page -
    from the heading last printed before it, which is where a gazette notice left at the foot of a
    page continues; a continuation table keeps the entity of the table it continues. Nothing is
    carried from one table to an unrelated one: a table with no heading before it is 'unknown',
    except in a document that has no QDD heading anywhere (an extract the bank produced itself, one
    notice per file), where the listing's entity is the only label and is used, flagged as such.
    Entity, year and publication are stored per table on the context and reach the lines in
    `brb_finish`, once every file has been read."""
    import pdfplumber
    ctx = ctx if ctx is not None else BrbContext()
    ctx.file_order += 1
    my_file_order = ctx.file_order
    out = Parsed()
    with pdfplumber.open(path) as pdf:
        texts = [p.extract_text() or "" for p in pdf.pages]
        out.n_pages = len(texts)
        full = fold("\n".join(texts))
        if len(re.sub(r"[^A-Z]", "", full)) < 200 or "TRIMESTRE" not in full \
                or not re.search(r"JANEIRO|ABRIL|JULHO|OUTUBRO", full):
            out.notes.append("unreadable: no text layer, or a font encoding without a Unicode map")
            return out
        entity, year, quarter = brb_heading(full)
        if entity is not None and entity != doc.entity:
            out.notes.append(f"document names entity {entity}, listing says {doc.entity}")
        year = year or doc.year
        head_quarters: set[tuple[int, int]] = set()    # what the tables' own headings say
        has_heading = bool(BRB_ANCHOR.search(re.sub(r"\s", "", full)))
        state = {"year": year, "quarter": quarter, "month_cols": None, "total_col": None,
                 "ncols": None, "scope": None, "sponsorship": None, "basis": None,
                 "closed": False}
        bases: dict[str, str | None] = {}
        tables_found = 0
        doc_heading: dict | None = None       # the QDD heading printed last before this point
        page_flags = {"skip": False, "suppressed": 0}

        def keep_row(row_key: str) -> bool:
            if page_flags["skip"] or row_key in ctx.rows:
                page_flags["suppressed"] += 1
                return False
            ctx.rows.add(row_key)
            return True

        for pno, page in enumerate(pdf.pages, start=1):
            page_key = hashlib.sha256(fold(texts[pno - 1]).encode()).hexdigest()[:16]
            # every blank page has the same (empty) text, so a blank page is never "already read"
            blank = not fold(texts[pno - 1])
            seen = None if blank else ctx.pages.get(page_key)
            page_flags.update(skip=bool(seen and seen["complete"]), suppressed=0)
            complete = True
            pub = brb_publication(texts[pno - 1])
            if pub:
                out.published.add(pub["label"])
            # A page can continue a table that a DIFFERENT file left open: BRB's listing links one
            # gazette page under each entity printed on it, so the page that opened a table (DODF
            # 127 p.68, printed under the bank's link) and the page that closes it (p.69, printed
            # under the finance company's and the brokerage's links) are not always the same file.
            # `state["month_cols"]` empty means this file has nothing of its own to continue from
            # (it is the first page read here, or the local table already closed); only then is a
            # table another file left open on the immediately preceding page of the same DODF issue
            # adopted, so a same-file continuation is never overridden by this.
            if pub and not state["month_cols"]:
                prev = ctx.continuations.get(pub["issue"])
                if prev and prev["page"] == pub["page"] - 1 and prev["state"] is not None:
                    state = dict(prev["state"])
                    out.notes.append(f"{state['scope']}: table continues {prev['label']} onto "
                                     f"{pub['label']} (read from {prev['source']})")
            lines = brb_text_lines(page)
            anchors = [i for i, ln in enumerate(lines) if BRB_ANCHOR.search(ln["stripped"])]
            # A row above this page's first table continues the table that ended on the previous
            # page, so the sweep below needs the state as it stood before this page's tables were
            # read, not after.
            at_page_start = dict(state)
            tobjs = page.find_tables()
            for tidx, tobj in enumerate(tobjs):
                rows = tobj.extract()
                if not rows:
                    continue
                top, bottom = tobj.bbox[1], tobj.bbox[3]
                # The month row can sit below blank spacer rows in the mangled wide layouts
                # (2021Q4 put it at index 4), and a table whose header is not found is
                # treated as a headerless continuation -- with no state to continue from,
                # the whole 2021Q4 advertising table was dropped. Blank rows do not count
                # against the window, so a real body row still cannot pass as a header. The
                # 2016Q1 bank and brokerage tables run TWO intro sentences plus a split
                # "FINALIDADE DA ACAO / VALORES REALIZADOS" row and a "BENEFICIARIO /
                # CLASSIFICACAO DA DESPESA" row before the month row -- 4 non-blank rows on
                # their own, one more than 2021Q4 -- so a table whose header is not found within
                # the first 4 still gets one more try before falling back to "no header here".
                probe = [i for i, r in enumerate(rows)
                         if any(str(c or "").strip() for c in r)][:8]
                hdr = next((i for i in probe if brb_month_cols(rows[i])), None)
                col_shift = 0
                # A published total row closes its table, so the next table object begins a new
                # one even where its header is unreadable. Narrow layouts hyphenate the month
                # names ("OUTU- BRO"), which defeats header detection, and without this rule the
                # advertising and sponsorship tables of one quarter merged into a single scope
                # and collected two sets of month totals.
                new_table = hdr is not None or state["closed"] or any(
                    re.search(r"BENEFICI|VALORES REALIZADOS", fold(str(c or ""))) for c in rows[0])
                if hdr is not None:
                    mc = brb_month_cols(rows[hdr])
                    tc = next((j for j, c in enumerate(rows[hdr]) if j > max(mc)
                               and "TOTAL" in fold(re.sub(r"\s", "", str(c or "")))), None)
                    state.update(month_cols=mc, total_col=tc, ncols=len(rows[hdr]))
                    # Column x-ranges from the header cells, so a row lying outside the detected
                    # table can still be assigned to the right month (see the sweep below).
                    try:
                        cells = tobj.rows[hdr].cells
                        state["col_x"] = {j: (c[0], c[2]) for j, c in enumerate(cells)
                                          if c is not None}
                    except Exception:
                        state["col_x"] = state.get("col_x")
                    body = rows[hdr + 1:]
                elif state["month_cols"]:
                    # A continuation table carries no header. Its columns are aligned from the
                    # RIGHT, where the month and total columns sit, because a continuation can
                    # come back with a different number of label columns; skipping it silently
                    # dropped rows (two Metropoles rows of 300,000 in 2025Q1, caught by the
                    # published-total check).
                    shift = len(rows[0]) - state["ncols"]
                    col_shift = shift
                    if shift and all(j + shift >= 0 for j in state["month_cols"]):
                        state = dict(state,
                                     month_cols={j + shift: mo for j, mo in state["month_cols"].items()},
                                     total_col=(state["total_col"] + shift
                                                if state["total_col"] is not None else None),
                                     ncols=len(rows[0]))
                        out.notes.append(f"{state['scope']}: continuation table has "
                                         f"{len(rows[0])} columns against {len(rows[0]) - shift}; "
                                         f"amount columns realigned from the right")
                    elif shift:
                        out.notes.append(f"{state['scope']}: continuation table with "
                                         f"{len(rows[0])} columns could not be aligned; skipped")
                        complete = False
                        continue
                    body = rows[1:] if new_table else rows
                    # A table object that folds a SECOND ENTITY's own QDD notice into what should
                    # be one entity's continuation rows (2016Q1 DODF p.12: the brokerage's closing
                    # rows share a table object with the finance company's WHOLE notice, header and
                    # all) is unreliable for per-cell reading from that point on: pdfplumber cuts a
                    # wrapped label cell's own continuation ("...DEMONSTRACOES\nFINANCEIRAS") into
                    # a spurious extra non-blank cell, which can defeat brb_is_total_row's one-cell
                    # rule for THIS table's own total row too (the brokerage's "TOTAL PAGO NO 1o
                    # TRIMESTRE" row read as a detail row beside a leftover "S FINANCEIRAS" cell,
                    # double-counting its quarter). This must NOT fire on a plain mention of
                    # another entity: BRB's tables routinely cross-reference another entity's
                    # quarter total in a footer line ("read as a cross-reference to bank Q1 2018"),
                    # which the existing quarter_cumulative path already handles correctly on the
                    # grid, and several multi-page notices (the "SUMAR" annual tables) reprint THIS
                    # SAME table's own title as a running header on every continuation page. Only a
                    # SECOND entity's own QDD heading actually opening inside the table - its name
                    # (BRB_ENTITY_NAMES) AND an anchor phrase (BRB_ANCHOR) both present - is
                    # specific enough to the real defect; bail out of the grid read for the WHOLE
                    # table object only then, and leave the region unread here so
                    # `brb_recover_text_tables` can read it by column position instead of forcing a
                    # per-cell grid that has already proven unreliable on this page. Gated to
                    # 2016Q1, the one quarter this was verified against page by page: several
                    # multi-page annual tables elsewhere in the corpus (the "SUMAR"/"GEPUP_GEPAC"
                    # documents) legitimately name every conglomerate member in their own
                    # boilerplate AND carry a reprinted anchor phrase, which reads as "foreign" by
                    # this same test without actually being a second notice folded in.
                    own_meta = ctx.tables[state["scope"]]
                    foreign = False
                    if own_meta.get("year") == 2016 and own_meta.get("quarter") == 1:
                        own_entity = own_meta["entity"]
                        cell_texts = [fold(re.sub(r"\s", "", str(c or "")))
                                     for r in body for c in r]
                        foreign = (
                            any(BRB_ANCHOR.search(t) for t in cell_texts)
                            and any(e != own_entity and re.search(pat, t)
                                   for t in cell_texts for e, pat in BRB_ENTITY_NAMES))
                    if foreign:
                        out.notes.append(
                            f"{state['scope']}: table object on p{pno} mixes in another "
                            "entity's own QDD notice; left for the column-position fallback "
                            "instead of the per-cell grid read")
                        complete = False
                        continue
                else:
                    # Nothing to continue from: the table's start is on a page this file does not
                    # hold. The page is then only partly read here, so another file that holds the
                    # previous page reads it again (other notices' tables carry no amounts).
                    if any(re.search(r"\d,\d{2}", str(c or "")) for r in rows for c in r):
                        complete = False
                    continue
                if new_table:
                    tables_found += 1
                    scope = f"{page_key[:12]}#t{tidx}"
                    head = brb_table_heading(lines, anchors, top)
                    if head is not None:
                        where = "same_page"
                    elif doc_heading is not None:
                        head, where = doc_heading, "earlier_page"
                    if head is not None and head["entity"]:
                        ent, rule = head["entity"], f"{head['rule']}_{where}"
                    elif head is not None and head["conflict"] and doc.entity in head["conflict"]:
                        # The heading names two entities; the listing, which names one of them,
                        # settles it. It never supplies an entity the page does not print.
                        ent, rule = doc.entity, f"heading_conflict_settled_by_listing_{where}"
                    elif not has_heading:
                        ent, rule = doc.entity, "listing_document_without_heading"
                    else:
                        ent, rule = "unknown", (head["rule"] if head is not None
                                                else "no_heading_before_table")
                        complete = False           # another copy may hold the heading's page
                    if head is not None and head["year"]:
                        state.update(year=head["year"], quarter=head["quarter"])
                        head_quarters.add((head["year"], head["quarter"]))
                    brb_register_table(ctx, scope, {
                        "entity": ent, "entity_rule": rule, "year": state["year"],
                        "quarter": state["quarter"], "heading": head["text"] if head else "",
                        "published": pub["date"] if pub else None,
                        "publication": pub["label"] if pub else None,
                        "first_read": f"{path.name} p{pno}", "listing_entity": doc.entity,
                        "notice": "", "tails": set(), "basis": None, "total_is_sum": False,
                        # Filing order for a table the gazette never dates (see `brb_finish`):
                        # the listing's own (year, quarter) for this file, and the order this
                        # file was read in among the bank's listed documents.
                        "listing_year": doc.year, "listing_quarter": doc.quarter,
                        "file_order": my_file_order})
                    state.update(scope=scope, sponsorship=None, basis=None, closed=False)
                scope, mc, tc = state["scope"], state["month_cols"], state["total_col"]
                meta = ctx.tables[scope]
                first_m = min(mc)
                amount_cols = sorted(mc) + ([tc] if tc is not None else [])
                classes = " ".join(fold(" ".join(str(c or "") for c in r[:first_m])) for r in body)
                if state["sponsorship"] is None:
                    # "PROP E PUBL" is the 2016 files' abbreviation of "Propaganda e Publicidade";
                    # without it their advertising table was taken for a sponsorship table. This is
                    # only a fallback for a row whose OWN classification cell is ambiguous
                    # (`brb_group`): it is not reliable enough on its own to type the whole table
                    # (a beneficiary can be named "... PUBLICIDADE ... LTDA" in a table with no
                    # advertising row at all, e.g. "OH! ARTES, PUBLICIDADE, PRODUCAO E EVENTOS
                    # LTDA" in a 2019Q4 sponsorship table), so `meta["kind"]` (used to keep a
                    # sponsorship table from being compared against an advertising one as if they
                    # were vintages of each other) is set in `brb_finish` from the table's ACTUAL
                    # row classifications instead, once every row has been read.
                    state["sponsorship"] = not re.search(
                        r"PROPAGANDA|PUBLICIDADE|PUBLICAC|P U B L|PROP E PUBL", classes)
                    meta["sponsorship_guess"] = state["sponsorship"]
                sponsorship = state["sponsorship"]
                prev = [""] * first_m
                month_stated: dict[int, float] = {}
                offset = len(rows) - len(body)
                # 2016Q1's three-label-column tables print each record stacked over several grid
                # rows (`brb_stacked_records`); the other quarters, and Cartao BRB's four-column
                # table, are read row by row as the grid gives them.
                stack = None
                if (meta.get("year") == 2016 and meta.get("quarter") == 1 and first_m == 3
                        and not col_shift and state.get("col_x") and len(tobj.rows) == len(rows)):
                    stack = brb_stacked_records(
                        page, tobj, state["col_x"], first_m, amount_cols,
                        tobj.rows[hdr].bbox[3] if hdr is not None else top)
                stacked_rows: dict[int, dict] = {}
                for ridx, r in enumerate(body, start=offset):
                    row_key = f"{page_key}:t{tidx}:r{ridx}"
                    r = split_merged_amounts(r, amount_cols, BRB_AMOUNT_RE)
                    # A footnote can print inside the SAME pdfplumber table object as the rows it
                    # annotates, in a cell of a row that is itself neither a real detail row nor
                    # recognisable as a total row (3_trimestre_brb_6b445b66.pdf: "*(*) Retificacao
                    # do valor do 2o Trimestre ..." sits in the label cell of the following row,
                    # which pdfplumber's layout puts INSIDE the table's bbox, so it never reaches
                    # `brb_text_between`'s "text below the table" capture). Read every cell, not
                    # just the label columns, so `meta["notice"]` still sees it either way.
                    if BRB_RETIFIC_RE.search(fold(" ".join(str(c or "") for c in r))):
                        meta["notice"] += " " + " ".join(str(c or "").replace("\n", " ") for c in r)
                    labels = [str(r[j] or "").replace("\n", " ").strip() for j in range(first_m)]
                    if not any(labels):
                        # A cumulative "TOTAL REALIZADO/PAGO NO No TRIMESTRE" row can print its own
                        # label inside what the header calls a MONTH column instead of the label
                        # columns, every true label column landing blank (3_trimestre_brb_
                        # 6b445b66.pdf: "Total Realizado no 2o Trimestre de 2019*" sits where
                        # January's amount would be). Read the row's own text cells instead, so
                        # `brb_is_total_row` and the quarter-reference read below still see it.
                        shifted = [str(r[j] or "").replace("\n", " ").strip() for j in sorted(mc)
                                  if j < len(r) and re.search(r"[A-Za-z]", str(r[j] or ""))]
                        labels = shifted or labels
                    flab = fold(" ".join(labels))
                    amounts = {mo: brb_cell_amount(r[j]) for j, mo in mc.items() if j < len(r)}
                    repaired = brb_shift_repair(r, amounts, mc, tc)
                    gap = brb_row_total_gap(r, repaired, tc)
                    amounts = repaired
                    if brb_is_total_row(labels):
                        state["basis"] = brb_total_basis(flab, state["basis"])
                        qv = brb_cell_amount(r[tc]) if tc is not None and tc < len(r) else None
                        if any(v is not None for v in amounts.values()):
                            state["closed"] = True
                            emit = keep_row(row_key)
                            if emit:
                                ctx.observed.add((meta["entity"], state["year"], state["quarter"]))
                            for mo, v in amounts.items():
                                if v is not None:
                                    month_stated[mo] = v
                                    if emit:
                                        out.totals.append({
                                            "scope": f"{scope}:{mo:02d}", "stated": v,
                                            "label": f"{flab[:30]} month {mo}", "_table": scope,
                                            "_month": mo, "content_key": f"{row_key}:m{mo:02d}"})
                            if qv is not None and abs(qv - sum(month_stated.values())) <= 0.05:
                                if emit:
                                    out.totals.append({
                                        "scope": f"{scope}:quarter", "stated": qv,
                                        "label": f"{flab[:30]} quarter", "_table": scope,
                                        "_month": None, "content_key": f"{row_key}:quarter"})
                                meta["total_is_sum"] = True
                            elif qv is not None and emit:
                                out.notes.append(f"{scope}: TOTAL column {qv:,.2f} is not the sum "
                                                 "of the months (year-to-date); not used as a check")
                        elif qv is not None:
                            # A total row with no month breakdown at all: Cartao BRB's tables close
                            # with a single "Total Recebido" figure and no per-month split, and the
                            # cumulative "TOTAL PAGO/REALIZADO NO No TRIMESTRE DE <year>" rows a
                            # later quarter's table prints for EARLIER quarters carry only one
                            # number too. Read `flab` for a trimester number: none, or the table's
                            # OWN quarter, means this row is closing THIS table (its quarter total,
                            # same as the per-month branch above gives when months are printed);
                            # a DIFFERENT quarter's number means this is a cross-reference to
                            # another table, kept on `quarter_cumulative` for
                            # `brb_apply_quarter_corrections` to read, never as this table's own
                            # check. "TOTAL ... EM <year>" (the annual figure, no trimester word) is
                            # neither and is dropped: it would silently pass as a quarter's own
                            # total otherwise.
                            qm = re.search(r"(\d)\s*[O\xba\xb0]?\s*TRIMESTRE", flab)
                            ref_q = int(qm.group(1)) if qm else None
                            annual_only = ref_q is None and re.search(r"\bEM\s*20\d\d\b", flab)
                            emit = keep_row(row_key)
                            if annual_only:
                                pass
                            elif ref_q is None or ref_q == state["quarter"]:
                                state["closed"] = True
                                if emit:
                                    ctx.observed.add(
                                        (meta["entity"], state["year"], state["quarter"]))
                                    out.totals.append({
                                        "scope": f"{scope}:quarter", "stated": qv,
                                        "label": f"{flab[:30]} quarter (no month breakdown)",
                                        "_table": scope, "_month": None,
                                        "content_key": f"{row_key}:quarter"})
                            elif emit:
                                meta.setdefault("quarter_cumulative", {})[ref_q] = qv
                                out.notes.append(f"{scope}: read as a cross-reference to "
                                                 f"{meta['entity']} Q{ref_q} {meta['year']} "
                                                 f"({qv:,.2f}), not this table's own total")
                        bases[scope] = state["basis"]
                        continue
                    if all(v is None for v in amounts.values()):
                        continue
                    center = (brb_stacked_center(stack, tobj, ridx, rows[ridx], amount_cols)
                              if stack else None)
                    if center is not None:
                        labels = stack["labels"][center]
                    else:
                        labels = [lab or prev[j] for j, lab in enumerate(labels)]
                    prev = labels
                    if not keep_row(row_key):
                        continue
                    beneficiary = labels[0]
                    classification = labels[1] if first_m >= 2 else ""
                    purpose = " ".join(labels[2:]) if first_m >= 3 else ""
                    group = brb_group(classification, purpose, sponsorship)
                    if center is not None:
                        # Every row of a stacked table that is not propaganda or publicacoes is a
                        # patrocinio row (Esportes, Negocios, Entretenimento, ...).
                        group = "sponsorship" if group == "other" else group
                        # The record's month cells may sit on several grid rows; its TOTAL is
                        # compared once, against all of them.
                        rec_sum = stacked_rows.setdefault(center, {
                            "sum": 0.0, "stated": None, "cells": {}, "row_key": row_key,
                            "label": f"{beneficiary} | {purpose}".strip(" |")[:60]})
                        rec_sum["row_key"] = row_key
                        for mo, v in amounts.items():
                            if v is not None:
                                rec_sum["sum"] += v
                                rec_sum["cells"][mo] = rec_sum["cells"].get(mo, 0.0) + v
                        qv_row = brb_cell_amount(r[tc]) if tc is not None and tc < len(r) else None
                        if rec_sum["stated"] is None:
                            rec_sum["stated"] = qv_row
                        gap = None
                    if gap is not None:
                        out.gaps.append({"table": scope, "row_key": row_key, "gap": gap,
                                         "label": f"{beneficiary} | {purpose}".strip(" |")[:60],
                                         "cells": dict(amounts), "file": path.name, "page": pno})
                    for mo, v in amounts.items():
                        if v is None:
                            continue
                        rec = line(
                            doc, period=f"{state['year']}{mo:02d}", frequency="monthly",
                            period_rule="document month column",
                            agency=beneficiary if group in ("advertising", "production") else None,
                            section="PATROCINIOS" if sponsorship else "PUBLICIDADE",
                            category=classification, detail=f"{beneficiary} | {purpose}".strip(" |"),
                            category_group=group, amount_brl=v, amount_kind="total",
                            reversal=v < 0, total_scope=f"{scope}:{mo:02d}",
                            extra_scopes=[f"{scope}:quarter"], line_id=f"{row_key}:m{mo:02d}",
                            content_key=f"{row_key}:m{mo:02d}", table_id=scope,
                            page_or_sheet=f"p{pno}", parse_method="pdfplumber table cells")
                        rec["_table"], rec["_month"] = scope, mo
                        out.lines.append(rec)
                        out.periods.add((meta["entity"], rec["period"]))
                        ctx.observed.add((meta["entity"], state["year"], state["quarter"]))
                for rec_sum in stacked_rows.values():
                    gap = (None if rec_sum["stated"] is None
                           else rec_sum["stated"] - rec_sum["sum"])
                    if gap is not None and abs(gap) > 0.05:
                        out.gaps.append({"table": scope, "row_key": rec_sum["row_key"],
                                         "gap": gap, "label": rec_sum["label"],
                                         "cells": rec_sum["cells"], "file": path.name,
                                         "page": pno})
                bases.setdefault(scope, state["basis"])
                # The notice's own text below this part of the table, up to the next heading or
                # table, carries any republication note ("Retificam-se ...", "Republicado por ...")
                # and the basis it states ("regime de competencia").
                below = [ln["top"] for ln in lines
                         if ln["top"] > bottom and BRB_ANCHOR.search(ln["stripped"])]
                below += [t.bbox[1] for t in tobjs if t.bbox[1] > bottom + 1]
                tail_key = f"{page_key}:{tidx}"
                if tail_key not in meta["tails"]:
                    meta["tails"].add(tail_key)
                    meta["notice"] += " " + brb_text_between(lines, bottom, min(below, default=1e9))

            brb_sweep_outside(out, doc, page, pno, state, at_page_start, page_key, keep_row)
            # Always runs, even where `complete` is already True: it does real work (closing
            # whatever was left open and registering any heading the grid never reached at all),
            # never only a completeness check, so it cannot be folded into `complete` through a
            # short-circuiting `or`. A page whose grid read bailed out is not "partly read" once
            # this call accounts for everything it was responsible for, so a fully successful
            # recovery (True, never None's "nothing to do here") sets `complete` the same way a
            # fully successful grid read would, and `documents.pages_unread` reports what is
            # actually still missing.
            recovered = brb_recover_text_tables(out, doc, path, pub, page, pno, lines, anchors,
                                                page_key, ctx, at_page_start, keep_row)
            if recovered:
                complete = True
            if pub:
                # Recorded on every visit, including a repeat copy of the same page: the state a
                # printed page leaves a table in is a fact about that page, not about which file
                # happened to hold it. `state=None` once the table has closed means the NEXT DODF
                # page of this issue starts fresh rather than adopting a finished table.
                ctx.continuations[pub["issue"]] = {
                    "page": pub["page"], "label": pub["label"], "source": path.name,
                    "state": dict(state) if state.get("scope") and not state["closed"] else None}
            if anchors:
                i = anchors[-1]
                stop = min((t.bbox[1] for t in tobjs if t.bbox[1] > lines[i]["top"]), default=1e9)
                doc_heading = brb_heading_at(lines, i, stop)
            if seen:
                how = "already read" if seen["complete"] else "partly read"
                out.pages_skipped.append(
                    f"p{pno} {how} in {seen['first']}"
                    + (f" ({page_flags['suppressed']} rows already emitted)"
                       if not seen["complete"] else ""))
            if not blank:
                # Provenance for a page cached under more than one file with byte-identical text
                # (a financeira 2018Q4 page cached under both a DTVM-named and a CFI-named file):
                # the FIRST file to reach this page_key claims "first" and every line's
                # `source_file` follows it; a later file's copy is a pure duplicate and never
                # overwrites it. "First" is deterministic because it follows `bank_docs`'s own
                # order, which is `discover_brb`'s `by_href` - a dict built by one single,
                # deterministic left-to-right walk of the bank's listing page (BeautifulSoup's
                # `find_all_next` always yields the same DOM order for the same cached HTML) - so
                # the same two files resolve to the same winner on every run against this cache;
                # `by_href` keys on `href`, so two different files can never tie for the same
                # listing position. Only the listing order decides.
                ctx.pages[page_key] = {"first": seen["first"] if seen else f"{path.name} p{pno}",
                                       "complete": bool(seen and seen["complete"]) or complete}
        for ln in out.lines:
            ctx.line_scopes.add(ln["total_scope"])
        for scope, basis in bases.items():
            if basis:
                ctx.tables[scope]["basis"] = basis
        # A table published as a total row only (no detail rows): keep the published total. The
        # detail rows may have been emitted from another copy of the page, hence the bank-wide set.
        for t in list(out.totals):
            if t["_month"] is None or t["scope"] in ctx.line_scopes:
                continue
            tscope, mo = t["_table"], t["_month"]
            table_meta = ctx.tables[tscope]
            # The row's own label is all this total-only table publishes, so it is read the same
            # way `brb_group` reads a classification cell: "PATROCINIO" -> sponsorship, "PRODUCAO"
            # -> production, and so on, instead of being dropped into "other" regardless of what it
            # says. `sponsorship_guess` (the table-level text heuristic, `parse_brb`) is only the
            # last-resort fallback within `brb_group` itself, same as for a detail row.
            label_text = t["label"].split(" month")[0]
            group = brb_group(label_text, label_text,
                              bool(table_meta.get("sponsorship_guess")))
            rec = line(
                doc, period=f"{table_meta['year']}{mo:02d}", frequency="monthly",
                period_rule="document month column",
                section="PATROCINIOS" if group == "sponsorship" else "PUBLICIDADE",
                category=label_text, detail="total published without detail rows",
                category_group=group, amount_brl=t["stated"], amount_kind="total",
                total_scope=t["scope"],
                # the quarter scope too, or the table's quarter total has nothing to sum and reads
                # as a failed check although its months are all present
                extra_scopes=[f"{tscope}:quarter"],
                line_id=f"{t['content_key']}:total_only", content_key=f"{t['content_key']}:total_only",
                table_id=tscope, page_or_sheet=None,
                parse_method="pdfplumber table cells (total row only)")
            rec["_table"], rec["_month"] = tscope, mo
            out.lines.append(rec)
            ctx.line_scopes.add(t["scope"])
        if tables_found == 0:
            out.notes.append("unreadable: text layer present but no table with month columns")
        # The listing can point at the wrong gazette issue: its 2019Q3 finance-company link serves
        # the 2018Q3 page (DODF 191, p.39), which leaves 2019Q3 unobserved for that entity.
        if doc.quarter and head_quarters and (doc.year, int(doc.quarter)) not in head_quarters:
            out.notes.append(f"listed as {doc.year}Q{int(doc.quarter)}, but its tables' headings read "
                             + ", ".join(f"{y}Q{q}" for y, q in sorted(head_quarters)))
        out.meta = {"entity": entity, "year": year, "quarter": quarter}
    return out


def brb_impute_forced_cells(lines: list[dict], totals: list[dict], gaps: list[dict],
                            ctx: BrbContext) -> list[dict]:
    """Impute a printed cell only where the table's own arithmetic forces one value.

    Within one (deduplicated) table a cell is forced when (a) the row's month cells differ from the
    row's own printed TOTAL by a gap g, in a table whose TOTAL column is the sum of the months;
    (b) exactly one month column of the table falls short of its printed column total; (c) that
    column's shortfall equals g to the cent; and the row prints an amount in that column. The cell
    becomes printed + g, keeping the printed figure in `amount_as_printed`. The one case known is
    the PPR veiculacao row of the bank's 2018Q1 table (DODF 66, p.37): January prints 32.000,00
    while the row's TOTAL (1.782.037,61) and the January column total (1.607.917,39) both imply
    537.258,64, and the published month totals sum to the quarter's 5.425.434,01 to the cent.
    A row that meets (a) alone is reported and kept as printed - including a row whose cell in the
    short column is malformed ("8.000,000"), since there is no printed amount to correct. Two rows
    of the same table can each meet (a)-(c) against the SAME month by coincidence (two separate
    gaps that happen to equal that month's shortfall to the cent); condition (d) requires the
    shortfall to have exactly one row matching it in the table, so neither is imputed when that
    happens - there would be no way to tell which row the gazette actually under-printed.
    Returns one record per row that meets (a), with its outcome."""
    stated: dict[tuple[str, int], list[float]] = {}
    for t in totals:
        if t.get("_month") is not None and not t.get("superseded"):
            stated.setdefault((t["_table"], t["_month"]), []).append(t["stated"])
    sums: dict[tuple[str, int], float] = {}
    for ln in lines:
        k = (ln["_table"], ln["_month"])
        sums[k] = sums.get(k, 0.0) + float(ln["amount_brl"])
    by_key = {ln["content_key"]: ln for ln in lines}
    live = {ln["_table"] for ln in lines}
    # Per table, the month columns that fall short of their own printed total by more than half a
    # cent - by construction every OTHER month of that table already reconciles within 0.005, since
    # it is excluded from `short` unless it does not.
    short_by_table: dict[str, dict[int, float]] = {}
    for (t, mo), vs in stated.items():
        if len({round(v, 2) for v in vs}) > 1:
            continue                              # two different printed totals; handled per-row
        d = round(vs[0] - sums.get((t, mo), 0.0), 2)
        if d > 0.005:
            short_by_table.setdefault(t, {})[mo] = d
    # How many of the table's own gap rows match each shortfall to the cent.
    match_count: dict[tuple[str, int], int] = {}
    for g in gaps:
        for mo, d in short_by_table.get(g["table"], {}).items():
            if abs(round(g["gap"], 2) - d) <= 0.005:
                match_count[(g["table"], mo)] = match_count.get((g["table"], mo), 0) + 1
    report = []
    for g in gaps:
        table = g["table"]
        rec = {"table": table, "row": g["label"], "file": g["file"], "page": g["page"],
               "gap": round(g["gap"], 2), "imputed": False}
        if table not in live:
            continue                              # the table's vintage was superseded
        if not ctx.tables[table]["total_is_sum"]:
            continue                              # TOTAL column is year-to-date: no row test
        cols = {mo: vs for (t, mo), vs in stated.items() if t == table}
        if any(len({round(v, 2) for v in vs}) > 1 for vs in cols.values()):
            rec["outcome"] = "the table states two different totals for one month"
            report.append(rec)
            continue
        short = short_by_table.get(table, {})
        if len(short) != 1:
            rec["outcome"] = (f"{len(short)} month columns fall short of their printed totals "
                              f"({ {m: d for m, d in short.items()} })")
            report.append(rec)
            continue
        (mo, d), = short.items()
        rec["month"], rec["shortfall"] = mo, d
        if abs(d - g["gap"]) > 0.005:
            rec["outcome"] = f"month {mo} falls short by {d:,.2f}, not by the row's gap"
            report.append(rec)
            continue
        if match_count.get((table, mo), 0) != 1:
            rec["outcome"] = (f"{match_count.get((table, mo), 0)} rows in the table have a gap "
                              f"matching month {mo}'s shortfall to the cent; not imputed")
            report.append(rec)
            continue
        cell = by_key.get(f"{g['row_key']}:m{mo:02d}")
        if cell is None:
            rec["outcome"] = (f"month {mo} falls short by exactly the gap, but the row prints no "
                              "readable amount in that column")
            report.append(rec)
            continue
        printed = float(cell["amount_brl"])
        cell["amount_as_printed"] = printed
        cell["amount_brl"] = round(printed + g["gap"], 2)
        cell["reversal"] = cell["amount_brl"] < 0
        cell["imputed"], cell["imputation_rule"] = True, "imputed_from_row_total"
        sums[(table, mo)] = sums.get((table, mo), 0.0) + g["gap"]
        rec.update(imputed=True, printed=printed, imputed_value=cell["amount_brl"],
                   outcome="imputed_from_row_total")
        report.append(rec)
    return report


BRB_ROW_NO_PAYMENT_RE = re.compile(
    r"RETIFICANDO O VALOR DISPOSTO NO QDD DO (\d)\s*[O\xba\xb0]?\s*TRIMESTRE"
    r".{0,400}?(?:NAO HOUVE PAGAMENTO|SEM PAGAMENTO)")


def brb_apply_notice_corrections(lines: list[dict], totals: list[dict], ctx: BrbContext,
                                 note) -> None:
    """Text-only corrections a LATER table's notice states about an EARLIER quarter, applied per
    the user's decision of 2026-09-28. Two forms, both seen in the cached gazettes:

    Row-specific ("Retificando o valor disposto no QDD do 2o trimestre da BRB Credito,
    Financiamento e Investimento S.A., informamos que nao houve pagamento e/ou provisao para a
    empresa W27 EVENTOS LTDA.", the finance company's 2022Q3 notice about its own 2022Q2 table):
    every line of the named beneficiary in the TARGET table (same entity, same year, the quarter
    the sentence names) is zeroed, keeping the printed figure in `amount_as_printed` and naming the
    correcting notice in `corrected_by`. The beneficiary is matched by searching the notice text for
    one of the target table's OWN printed beneficiary names, longest first, rather than trying to
    parse a company name out of free text.

    Total-only (the bank's 2019Q2 table: "TOTAL PAGO NO 1o TRIMESTRE DE 2019* 4.608.331,42" with a
    footnote "(*) Retificacao do valor do 1o Trimestre publicado no DODF no 65 ..."): a table's own
    cumulative reference to an earlier quarter (`quarter_cumulative`, from a "TOTAL
    PAGO/REALIZADO/CONTABILIZADO NO No TRIMESTRE" row with no month breakdown - see `parse_brb`)
    that differs from that quarter's own printed quarter total, on a table whose notice is marked as
    a retification or republication. The STATED total used in the checks is replaced; the rows are
    left exactly as printed, so the existing reconciliation test (`check_totals`) now shows the gap.
    """
    by_table: dict[str, list[dict]] = {}
    for ln in lines:
        by_table.setdefault(ln["_table"], []).append(ln)

    def find_target(meta: dict, ref_q: int) -> str | None:
        return next((t for t, m in ctx.tables.items()
                    if m["entity"] == meta["entity"] and m["year"] == meta["year"]
                    and m["quarter"] == ref_q), None)

    def find_target_same_kind(meta: dict, ref_q: int) -> str | None:
        # For a TOTAL correction only: BRB usually files an advertising table and a sponsorship
        # table each quarter, each with its OWN "TOTAL PAGO NO No TRIMESTRE" cumulative row, and
        # the two series' cumulative figures for the same quarter are entirely different amounts.
        # A sponsorship table's reference must correct only a sponsorship table, never the
        # advertising table that happens to share the entity, year and quarter number. The
        # row-specific correction has no such constraint - it is looking for one beneficiary's
        # row, wherever in that quarter's table(s) it was printed - so it keeps `find_target`.
        return next((t for t, m in ctx.tables.items()
                    if m["entity"] == meta["entity"] and m["year"] == meta["year"]
                    and m["quarter"] == ref_q and m.get("kind") == meta.get("kind")), None)

    def filed_later(later: dict, earlier: dict) -> bool:
        """Whether `later` is a filing `brb_filing_order` actually places AFTER `earlier` - never
        assumed just because `later`'s own table happens to be titled a later quarter than
        `earlier`'s. A table's forward references to a quarter it has not reached yet print a
        placeholder (a CFI 2022Q3 table gives "TOTAL PAGO NO 4o TRIMESTRE DE 2022 R$ 0,00" for the
        Q4 that had yet to be filed), and without this check that placeholder was read as a
        correction and overwrote the real Q4 total with zero once it was filed."""
        return brb_filing_order(later) > brb_filing_order(earlier)

    for scope, meta in ctx.tables.items():
        said = fold(meta["heading"] + " " + meta["notice"])

        rm = BRB_ROW_NO_PAYMENT_RE.search(said)
        if rm:
            ref_q = int(rm.group(1))
            target = find_target(meta, ref_q)
            if target and not filed_later(meta, ctx.tables[target]):
                target = None
            target_lines = by_table.get(target, []) if target else []
            candidates = sorted({ln["detail"].split(" | ")[0] for ln in target_lines
                                 if ln.get("detail")}, key=len, reverse=True)
            hit = next((b for b in candidates if len(fold(b)) >= 6 and fold(b) in said), None)
            src = meta["publication"] or meta["first_read"]
            if target and hit:
                touched = [ln for ln in target_lines if ln["detail"].split(" | ")[0] == hit
                          and float(ln["amount_brl"]) != 0]
                for ln in touched:
                    ln["amount_as_printed"] = float(ln["amount_brl"])
                    ln["amount_brl"] = 0.0
                    ln["reversal"] = False
                    ln["corrected_by"] = src
                text = (f"{target}: [{hit}] {len(touched)} line(s) zeroed - {src} says no payment "
                        f"was made to this beneficiary in {meta['entity']} Q{ref_q} {meta['year']}")
                log.info("  brb text correction (row): %s", text)
                note(meta["first_read"].rsplit(" p", 1)[0], "row correction: " + text)
            elif target:
                warn = (f"{scope}: notice says no payment to a named beneficiary in "
                        f"{meta['entity']} Q{ref_q} {meta['year']}, but no line of that table's "
                        "beneficiaries matches the notice text; not applied")
                log.warning("  brb text correction (row): %s", warn)
                note(meta["first_read"].rsplit(" p", 1)[0], warn)

        if not meta.get("republished"):
            continue
        # The retification must name the SPECIFIC quarter it corrects ("Retificacao do valor do
        # 1o Trimestre publicado no DODF..."); a table's own quarter can be a retification of ITS
        # OWN total (RETIFICACAO_BRB-4o-TRI_c3b2f246.pdf's footnote retifies its Q4, not the Q1-Q3
        # cumulative references its two tables also print, in a different, larger series - "TOTAL
        # REALIZADO", not "TOTAL PAGO" - that a blind read otherwise applied as bogus Q1-Q3
        # corrections). Without a named quarter that matches one of `quarter_cumulative`'s own
        # keys, nothing here is corrected.
        qm = re.search(r"RETIFICA\w*\s*DO\s*VALOR\s*DO\s*(\d)\s*[O\xba\xb0]?\s*TRIMESTRE",
                       meta["republished"] or "")
        named_q = int(qm.group(1)) if qm else None
        for ref_q, corrected in meta.get("quarter_cumulative", {}).items():
            if named_q is not None and ref_q != named_q:
                continue
            target = find_target_same_kind(meta, ref_q)
            if target is None or not filed_later(meta, ctx.tables[target]):
                continue
            entry = next((t for t in totals if t.get("_table") == target
                         and t.get("_month") is None and not t.get("superseded")), None)
            if entry is None or abs(entry["stated"] - corrected) <= 0.005:
                continue
            src = meta["publication"] or meta["first_read"]
            text = (f"{target}: quarter total {entry['stated']:,.2f} corrected to "
                    f"{corrected:,.2f} by {src} ({meta['republished']})")
            entry["stated_before_correction"] = entry["stated"]
            entry["stated"] = corrected
            entry["corrected_by"] = src
            log.info("  brb text correction (total): %s", text)
            note(ctx.tables[target]["first_read"].rsplit(" p", 1)[0], "total correction: " + text)


def brb_finish(lines: list[dict], totals: list[dict], status: list[dict], gaps: list[dict],
               ctx: BrbContext) -> list[dict]:
    """Complete the BRB lines once every file has been read, and return those kept.

    1. Each line takes its table's entity, CNPJ, period, basis and publication. The basis is
       'accrued' where the notice says its figures are on the accrual basis ("regime de
       competencia"), as the 2018Q4 finance-company notice does for the year it republishes.
    2. Vintages: the user's standing rule is that the LAST FILED value wins. Where tables published
       on different dates cover the same (entity, month), the later one is kept and the earlier
       dropped; the kept lines name what they supersede and the documents table records both. A
       later table whose notice does not say it is a republication ("Retifica", "Republica") is
       still preferred, but the note says so.
    3. Forced cells are imputed (`brb_impute_forced_cells`), after the dedupe and the vintages.
    """
    by_file = {row["file"]: row for row in status}

    def note(file: str, text: str) -> None:
        row = by_file.get(file)
        if row is not None:
            row["notes"] = "; ".join(x for x in (row.get("notes"), text) if x)

    for meta in ctx.tables.values():
        said = fold(meta["heading"] + " " + meta["notice"])
        # The sentence itself is kept, not a yes/no: a notice can correct an earlier quarter's
        # cumulative total ("Retificacao do valor do 1o Trimestre ...") without republishing its
        # own table, and the reader should see which.
        m = re.search(r"[^.*]{0,60}(RETIFIC|REPUBLICAD|REPUBLICAR|REPUBLICACA)[^.]{0,140}", said)
        meta["republished"] = m.group(0).strip(" *") if m else None
        # \w* between the two halves, not a literal concatenation: pdfplumber's word breaks land
        # inside "COMPETENCIA" as often as between the words, so "REGIME DE COM-\nPETENCIA" and
        # similar hyphenations still fold to something the exact string "REGIMEDECOMPETENCIA"
        # would miss.
        meta["basis_final"] = ("accrued" if BRB_ACCRUAL_RE.search(re.sub(r"\s", "", said))
                               else meta["basis"] or "undocumented")
    for ln in lines:
        meta = ctx.tables[ln["_table"]]
        ln.update(entity=meta["entity"], entity_rule=meta["entity_rule"],
                  entity_cnpj8=BRB_ENTITY_CNPJ8.get(meta["entity"]),
                  period=f"{meta['year']}{ln['_month']:02d}", basis=meta["basis_final"],
                  published=meta["published"], republished=meta["republished"])
    # A table's kind (used only to keep an advertising table and a sponsorship table that share a
    # period from being compared as if they were vintages of each other - see `covers` below), from
    # the classification ITS ROWS actually got, majority rules: reading it off the raw table text
    # instead false-positives on a beneficiary whose own NAME contains "PUBLICIDADE" (e.g. "OH!
    # ARTES, PUBLICIDADE, PRODUCAO E EVENTOS LTDA", a beneficiary of a 2019Q4 sponsorship table with
    # no advertising row at all). A table with no lines (e.g. total-only, no detail rows) gets no
    # kind, so it is never wrongly matched against a real one.
    from collections import Counter
    kind_counts: dict[str, Counter] = {}
    for ln in lines:
        kind_counts.setdefault(ln["_table"], Counter())[ln["category_group"]] += 1
    for scope, meta in ctx.tables.items():
        c = kind_counts.get(scope)
        if not c:
            meta["kind"] = None
            continue
        sp, tot = c.get("sponsorship", 0), sum(c.values())
        meta["kind"] = "sponsorship" if sp * 2 > tot else "advertising"

    covers: dict[tuple[str, str], set[str]] = {}
    for ln in lines:
        covers.setdefault((ln["entity"], ln["period"]), set()).add(ln["_table"])
    dropped: dict[tuple[str, str], list[str]] = {}
    for (entity, period), tables in sorted(covers.items()):
        if len(tables) < 2:
            continue
        # Compared only within its own kind (advertising vs. sponsorship): an advertising table
        # and a sponsorship table that happen to share an entity and a period are two SECTIONS of
        # one notice, not two vintages of the same table, and are never superseded against each
        # other, whichever is dated later. A kind of `None` (a table whose sponsorship flag was
        # never determined, e.g. it holds no detail rows at all) is its own group too, so it is
        # never silently matched against a real kind.
        by_kind: dict[object, set[str]] = {}
        for t in tables:
            by_kind.setdefault(ctx.tables[t].get("kind"), set()).add(t)
        for kind, group in by_kind.items():
            if len(group) < 2:
                continue
            order = {t: brb_filing_order(ctx.tables[t]) for t in group}
            latest = max(order.values())
            winners = sorted(t for t, k in order.items() if k == latest)
            if len(winners) != 1:
                raise SystemExit(
                    f"brb {entity} {period} ({kind}): {len(group)} tables of the same kind, from "
                    "different files, cover this period with no filing order that resolves which "
                    "to keep ("
                    f"{sorted(ctx.tables[t].get('publication') or ctx.tables[t]['first_read'] for t in group)}"
                    "); this is a hard failure by design (MAJOR B, 09-28 review) rather than "
                    "silently summing two vintages of the same table.")
            winner = winners[0]
            for t in group:
                if t != winner:
                    dropped[(t, period)] = [winner]
    amounts: dict[tuple[str, str], float] = {}
    for ln in lines:
        k = (ln["_table"], ln["period"])
        amounts[k] = amounts.get(k, 0.0) + float(ln["amount_brl"])
    summary: dict[tuple[str, str], list[str]] = {}
    for (t, period), winners in sorted(dropped.items()):
        summary.setdefault((t, winners[0]), []).append(period)
    for (old, new), periods in summary.items():
        mo, mn = ctx.tables[old], ctx.tables[new]
        before = sum(amounts.get((old, p), 0.0) for p in periods)
        after = sum(amounts.get((new, p), 0.0) for p in periods)
        text = (f"{mo['entity']} {min(periods)}-{max(periods)}: {mo['publication']} "
                f"({before:,.2f}) superseded by {mn['publication']} ({after:,.2f})"
                + ("" if mn["republished"] else "; the later notice does not say it republishes"))
        log.info("  brb vintage: %s", text)
        note(mo["first_read"].rsplit(" p", 1)[0], "superseded: " + text)
        note(mn["first_read"].rsplit(" p", 1)[0], "supersedes: " + text)
    kept = []
    for ln in lines:
        winners = dropped.get((ln["_table"], ln["period"]))
        if winners:
            continue
        older = [f"{ctx.tables[t]['publication']}" for (t, p), w in dropped.items()
                 if p == ln["period"] and ln["_table"] in w and ctx.tables[t]["entity"] == ln["entity"]]
        ln["supersedes"] = "; ".join(sorted(set(older))) or None
        kept.append(ln)
    for t in totals:
        meta = ctx.tables[t["_table"]]
        periods = ([f"{meta['year']}{t['_month']:02d}"] if t["_month"] is not None
                   else [p for (tt, p) in dropped if tt == t["_table"]])
        t["superseded"] = any((t["_table"], p) in dropped for p in periods)

    brb_apply_notice_corrections(kept, totals, ctx, note)

    report = brb_impute_forced_cells(kept, totals, gaps, ctx)
    for rec in report:
        if rec["imputed"]:
            text = (f"{rec['table']}: [{rec['row']}] month {rec['month']} printed "
                    f"{rec['printed']:,.2f}, imputed {rec['imputed_value']:,.2f} (row TOTAL gap "
                    f"{rec['gap']:,.2f} = the month column's shortfall)")
            log.info("  brb imputation: %s", text)
            note(rec["file"], "imputed_from_row_total: " + text)
        else:
            text = (f"{rec['table']}: a row's month cells do not sum to its own printed TOTAL (gap "
                    f"{rec['gap']:,.2f}); kept as printed, not imputed: {rec['outcome']} "
                    f"[{rec['row'][:40]}]")
            log.info("  brb row gap: %s", text)
            note(rec["file"], text)
    ctx.imputations = report
    return kept


def brb_membership(lines: pd.DataFrame, conglomerate: str) -> pd.Series:
    """Whether each BRB line's entity belongs to BRB's prudential conglomerate in the IF.data
    registry for the line's quarter (paths.IF_DATA_LIST: CodInst, CodConglomeradoPrudencial).

    BRB's QDD pages carry four entities' notices, and the conglomerate totals must hold only its
    members. The user's rule is that Cartao BRB counts ONLY if the registry puts it inside the
    conglomerate. The registry's members of C0080288 are the bank (00000208), the finance company
    (33136888) and the brokerage (33850686) in every quarter from 2014Q1, and Cartao BRB's CNPJ
    (01984199) appears in no list, as a member or otherwise, so its lines are kept but marked out of
    the conglomerate. The registry lists only institutions the central bank authorises; that a
    non-authorised company is absent from it is what the rule tests, not proof of how the
    prudential consolidation treats it."""
    members: dict[str, set[str]] = {}
    for f in sorted(paths.IF_DATA_LIST.glob("IF_DATA_List_*.csv")):
        d = pd.read_csv(f, dtype=str, usecols=["CodInst", "Data", "CodConglomeradoPrudencial"])
        d = d[d["CodConglomeradoPrudencial"] == conglomerate]
        for data, grp in d.groupby("Data"):
            members.setdefault(str(data), set()).update(grp["CodInst"])
    if not members:
        raise SystemExit(f"IF.data registry holds no member of {conglomerate}; cannot decide "
                         "which BRB entities belong to it")
    keys = sorted(members)

    def quarter_end(period: str) -> str:
        y, m = int(period[:4]), int(period[4:6])
        want = f"{y}{((m - 1) // 3 + 1) * 3:02d}"
        if want in members:
            return want
        # outside the registry's span, the nearest quarter it covers
        return min(keys, key=lambda k: abs((int(k[:4]) * 12 + int(k[4:])) - (y * 12 + m)))

    return pd.Series([bool(c) and c in members[quarter_end(str(p))]
                      for c, p in zip(lines["entity_cnpj8"], lines["period"])], index=lines.index)


# ---------------------------------------------------------------------------
# Banrisul
# ---------------------------------------------------------------------------
def parse_banrisul(doc: Doc, path: Path) -> Parsed:
    """'Valores Pagos - Grupo Banrisul': one block per agency with 'Midia - <type>' and
    'Producao' rows in 'Valor Bruto' and an unlabelled R$ total closing the block."""
    import pdfplumber
    out = Parsed()
    with pdfplumber.open(path) as pdf:
        pages = [(pno, rows_of(p)) for pno, p in enumerate(pdf.pages, start=1)]
    all_text = fold(" ".join(row_text(r) for _, rows in pages for r in rows))
    if doc.role == "supplier_names":
        # Names-only lists: any amount would mean values are published after all.
        if re.search(r"R\$\s*\d|\d{1,3}(\.\d{3})*,\d{2}", all_text):
            out.notes.append("supplier list carries amounts; not parsed")
        return out
    block, agency, period, scope = 0, None, None, None
    for pno, rows in pages:
        for r in rows:
            text = row_text(r)
            ft = fold(text)
            m = re.search(r"\b(\d{2})/(\d{4})\b", text)
            if m and not ft.startswith(("MIDIA", "PRODUCAO", "R$")) and "AGENCIA DE" not in ft:
                block += 1
                # The agency name is whatever sits left of the Mes/Ano token. That token is not
                # always its own word ("Ref.:03/2022"), so it is located by containment.
                at = next((w for w in r if m.group(0) in w["text"]), None)
                cut = at["x0"] if at is not None else float("inf")
                agency = " ".join(w["text"] for w in r if w["x1"] <= cut).strip(" :.-")
                period = f"{m.group(2)}{m.group(1)}"
                scope = f"{path.name}#block{block}"
                out.periods.add(("bank", period))
                continue
            if scope is None:
                continue
            amt = joined_amount(r)
            if amt is None:
                continue
            label_words = [w["text"] for w in r if not re.fullmatch(r"R\$|[\d.,-]+", w["text"])]
            label = " ".join(label_words)
            if not label:
                out.totals.append({"scope": scope, "stated": amt, "label": "unlabelled R$ total"})
                continue
            group = ("advertising" if fold(label).startswith("MIDIA")
                     else "production" if fold(label).startswith("PRODUCAO") else "other")
            out.lines.append(line(
                doc, agency=agency, period=period, frequency="monthly",
                period_rule="document Mes/Ano Ref.", section=None, category=label,
                category_group=group, amount_brl=amt, amount_kind="gross", basis="paid",
                client_as_published="Grupo Banrisul", total_scope=scope,
                line_id=f"{scope}:{fold(label)}", page_or_sheet=f"p{pno}",
                parse_method="pdfplumber words, row-aligned"))
    return out


PARSERS = {"banestes": parse_banestes, "basa": parse_basa, "banrisul": parse_banrisul,
           "brb": parse_brb}
CNPJ8 = {"banestes": "28127603", "basa": "04902979", "banrisul": "92702067", "brb": "00000208"}
MIN_TOTAL_MATCH = 0.90


def parse_all(docs: dict[str, list[Doc]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run each bank's parser over its listed files. Returns (lines, totals, doc status)."""
    lines, totals, status = [], [], []
    for bank, bank_docs in docs.items():
        # A byte-identical file is recorded and skipped: the fast path. It is not enough for BRB,
        # whose listing links one gazette page under each entity printed on it and whose copies
        # usually differ only in PDF metadata, so its parser also dedupes on the printed page and
        # row (`BrbContext`), and the entity comes from the page itself.
        seen_sha: dict[str, str] = {}
        ctx = BrbContext() if bank == "brb" else None
        bank_lines, bank_totals, bank_status, gaps = [], [], [], []
        for doc in bank_docs:
            path = RAW_DIR / bank / doc.entry["file"] if doc.entry.get("file") else None
            row = {"bank_key": bank, "url": doc.url, "file": doc.entry.get("file"),
                   "http_status": doc.entry.get("status"), "kind": doc.entry.get("kind"),
                   "role": doc.role, "listing_year": doc.year, "listing_month": doc.month,
                   "listing_quarter": doc.quarter, "entity": doc.entity, "agency": doc.agency,
                   "label": doc.label, "n_lines": 0, "n_totals": 0, "notes": None,
                   "n_pages": None, "pages_skipped": None, "published": None,
                   "pages_unread": None}
            if path is None or not path.exists():
                row["status"] = "missing_file"
                bank_status.append(row)
                continue
            sha = doc.entry.get("sha256")
            if sha and sha in seen_sha:
                row.update(status="duplicate_content",
                           notes=f"byte-identical to {seen_sha[sha]}; parsed once")
                bank_status.append(row)
                continue
            if sha:
                seen_sha[sha] = doc.entry.get("file")
            if doc.entry.get("kind") == "html":
                # A soft 404: the site answers 200 with its own HTML page instead of the file
                # (Banrisul's January-July 2025 links, 44,406 bytes each).
                row.update(status="unavailable_link",
                           notes="HTTP 200 returned the site's HTML page, not the document")
                bank_status.append(row)
                continue
            try:
                parsed = parse_brb(doc, path, ctx) if ctx is not None else PARSERS[bank](doc, path)
            except Exception as exc:                      # recorded, then raised after the pass
                row.update(status="error", notes=f"{type(exc).__name__}: {exc}")
                bank_status.append(row)
                continue
            for rec in parsed.lines:
                rec["cnpj8"] = CNPJ8[bank]
                rec["doc_role"] = doc.role
                bank_lines.append(rec)
            for t in parsed.totals:
                bank_totals.append({"bank_key": bank, "file": doc.entry["file"], **t})
            gaps += parsed.gaps
            note = "; ".join(parsed.notes) or None
            unreadable = any(n.startswith("unreadable") for n in parsed.notes)
            row.update(n_lines=len(parsed.lines), n_totals=len(parsed.totals), notes=note,
                       n_pages=parsed.n_pages or None,
                       pages_skipped="; ".join(parsed.pages_skipped) or None,
                       published="; ".join(sorted(parsed.published)) or None,
                       status=("unreadable" if unreadable
                               else "parsed" if parsed.lines
                               # every page it holds had already been read from another copy
                               else "duplicate_pages" if parsed.pages_skipped
                               else "names_only" if doc.role == "supplier_names" else "no_lines"))
            bank_status.append(row)
            log.info("  %-9s %-28s %-15s lines %4d  totals %3d  %s", bank,
                     (doc.entry["file"] or "")[:28], row["status"], row["n_lines"],
                     row["n_totals"], note or "")
        if ctx is not None:
            bank_lines = brb_finish(bank_lines, bank_totals, bank_status, gaps, ctx)
            kept = pd.Series([ln["source_file"] for ln in bank_lines]).value_counts()
            for row in bank_status:
                if row.get("status") == "parsed":
                    row["n_lines"] = int(kept.get(row["file"], 0))
            # A page that stayed "partly read" (`Doc.pages[key]["complete"] is False`) after every
            # file in the listing has been read: no copy ever supplied whatever the page's other
            # tables needed (a heading, a continuation - see `parse_brb`). Recorded on the FILE
            # that first read it, so a reader can go straight from the documents table to what is
            # still missing, quarter by quarter, rather than only learning it from a downstream
            # total mismatch.
            unread_by_file: dict[str, list[str]] = {}
            for key, seen in ctx.pages.items():
                if not seen["complete"]:
                    fname, _, pg = seen["first"].rpartition(" p")
                    unread_by_file.setdefault(fname, []).append("p" + pg)
            if unread_by_file:
                log.warning("brb: %d page(s) never fully read by any copy: %s",
                           sum(len(v) for v in unread_by_file.values()), unread_by_file)
                for row in bank_status:
                    if row.get("file") in unread_by_file:
                        row["pages_unread"] = "; ".join(sorted(unread_by_file[row["file"]]))
        lines += bank_lines
        totals += bank_totals
        status += bank_status
    return pd.DataFrame(lines), pd.DataFrame(totals), pd.DataFrame(status)


def check_totals(lines: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Each published total against the sum of the lines it covers."""
    if totals.empty:
        return totals
    sums: dict[str, float] = {}
    for rec in lines.to_dict("records"):
        for scope in [rec.get("total_scope")] + list(rec.get("extra_scopes") or []):
            if scope:
                sums[scope] = sums.get(scope, 0.0) + float(rec["amount_brl"])
    out = totals.copy()
    out["lines_sum"] = out["scope"].map(sums)
    # A published total of zero over a table with no rows agrees with its (empty) line set:
    # BASA's 2025 file prints "1o e 2o TRIM - VEICULACAO TOTAL 0,00" over empty sheets.
    out["lines_sum"] = out["lines_sum"].where(out["lines_sum"].notna() | (out["stated"] != 0), 0.0)
    out["diff"] = out["lines_sum"] - out["stated"]
    tol = np.maximum(1.0, 0.001 * out["stated"].abs())
    out["matches"] = out["lines_sum"].notna() & (out["diff"].abs() <= tol)
    return out


def period_table(lines: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    """Bank x entity x period: sums by category group, with the period's total checks.

    `in_conglomerate` rides along so the panel can leave out an entity outside the bank's
    prudential conglomerate, and `imputed_<group>` holds the part of each group's sum that was
    imputed rather than printed (`brb_impute_forced_cells`), so the panel can mark the cell."""
    keys = ["bank_key", "cnpj8", "entity", "period", "in_conglomerate"]
    groups = lines.pivot_table(index=keys, columns="category_group", values="amount_brl",
                               aggfunc="sum").reset_index()
    groups.columns.name = None
    extra = lines.assign(imputed_part=np.where(lines["imputed"].astype(bool),
                                               lines["amount_brl"] - lines["amount_as_printed"], 0.0))
    imputed = extra.pivot_table(index=keys, columns="category_group", values="imputed_part",
                                aggfunc="sum").reset_index()
    imputed.columns = [c if c in keys else f"imputed_{c}" for c in imputed.columns]
    counts = (lines.groupby(keys)
                   .agg(n_lines=("amount_brl", "size"), frequency=("frequency", "first"),
                        basis=("basis", lambda s: "/".join(sorted(set(s.dropna())))),
                        any_reversal=("reversal", "any"),
                        n_imputed=("imputed", "sum"),
                        source_files=("source_file", lambda s: "; ".join(sorted(set(s.dropna())))))
                   .reset_index())
    out = (counts.merge(groups, on=keys, how="left").merge(imputed, on=keys, how="left"))
    if not checks.empty:
        scope_period = (lines.dropna(subset=["total_scope"])
                             .groupby("total_scope")["period"].first())
        c = checks[~checks["superseded"]]
        c = c.assign(period=c["scope"].map(scope_period))
        agg = (c.dropna(subset=["period"]).groupby(["bank_key", "period"])
                .agg(n_stated_totals=("stated", "size"),
                     n_totals_matched=("matches", "sum")).reset_index())
        out = out.merge(agg, on=["bank_key", "period"], how="left")
    return out.sort_values(["bank_key", "entity", "period"]).reset_index(drop=True)


def cosif_ratios(periods: pd.DataFrame, panel_key: dict[str, str]) -> None:
    """Print 2025 own-file sums against the COSIF advertising flows. Never a gate: the own
    files are outlays and the COSIF accounts are accrued expense."""
    path = OUT_DIR / "cosif_advertising_quarterly.parquet"
    if not path.exists():
        log.warning("no %s; skipping the 2025 COSIF comparison", path.name)
        return
    cos = pd.read_parquet(path)
    cos = cos[(cos["year"] == 2025) & (cos["n_months_filed"] == 3)]
    p = periods[periods["period"].astype(str).str.startswith("2025")].copy()
    if p.empty:
        log.info("no 2025 periods parsed; no COSIF comparison")
        return
    p["quarter"] = ((p["period"].astype(str).str[4:6].astype(int) - 1) // 3 + 1)
    for bank, code in panel_key.items():
        own = p[p["bank_key"] == bank]
        if own.empty:
            continue
        for level, key in (("institution", CNPJ8[bank]), ("conglomerate", code)):
            c = cos[(cos["level"] == level) & (cos["entity_key"] == key)]
            if c.empty:
                continue
            for col in ("advertising", "sponsorship"):
                if col not in own.columns:
                    own[col] = np.nan
            mine = own.groupby("quarter")[["advertising", "sponsorship"]].sum(min_count=1)
            j = mine.join(c.set_index("quarter")[["adv", "all3"]], how="inner")
            if j.empty:
                continue
            log.info("  %s vs COSIF %s (%s), 2025 by quarter: advertising/adv %s; "
                     "(advertising+sponsorship)/adv %s; advertising/all3 %s", bank, level, key,
                     (j["advertising"] / j["adv"]).round(2).to_dict(),
                     ((j["advertising"].fillna(0) + j["sponsorship"].fillna(0))
                      / j["adv"]).round(2).to_dict(),
                     (j["advertising"] / j["all3"]).round(2).to_dict())


def validate(lines: pd.DataFrame, checks: pd.DataFrame, status: pd.DataFrame,
             unvalidated: tuple[str, ...] = ()) -> None:
    """Banks named in `unvalidated` may fall below the published-total floor. Their rows are
    written with validated = False and the shortfall is logged, so nothing passes silently."""
    errors = status[status["status"] == "error"]
    if len(errors):                                                            # check 1
        raise AssertionError(f"{len(errors)} files raised while parsing:\n"
                             f"{errors[['bank_key', 'file', 'notes']].to_string(index=False)}")
    if status["status"].isna().any():
        raise AssertionError("a listed file has no status")
    log.info("document status by bank:\n%s",
             status.groupby(["bank_key", "status"]).size().unstack(fill_value=0).to_string())

    # A total of a superseded vintage is kept in the checks for the record but judged no longer:
    # its lines were dropped in favour of the later filing.
    for bank, grp in checks[~checks["superseded"]].groupby("bank_key"):       # check 2
        share = float(grp["matches"].mean())
        log.info("  %s: %d published totals, %.1f%% match the line sums", bank, len(grp),
                 100 * share)
        if share < MIN_TOTAL_MATCH and bank in unvalidated:
            log.warning("%s: %.1f%% of published totals match, below the %.0f%% floor; its rows "
                        "are written with validated = False", bank, 100 * share,
                        100 * MIN_TOTAL_MATCH)
            continue
        if share < MIN_TOTAL_MATCH:
            worst = grp[~grp["matches"]].nlargest(5, "stated")
            raise AssertionError(f"{bank}: only {share:.1%} of published totals match "
                                 f"(floor {MIN_TOTAL_MATCH:.0%}); worst:\n"
                                 f"{worst[['file', 'scope', 'label', 'stated', 'lines_sum']].to_string(index=False)}")

    basa = lines[(lines["bank_key"] == "basa") & lines["amount_kind"].isin(["gross", "net", "tax"])]
    if not basa.empty:                                                         # check 3
        w = basa.pivot_table(index=["line_id"], columns="amount_kind", values="amount_brl",
                             aggfunc="first")
        full = w.dropna(subset=["gross", "net", "tax"])
        if len(full):
            bad = (full["gross"] - full["net"] - full["tax"]).abs() > 0.05
            log.info("  basa: gross = net + tax on %d of %d fully published lines",
                     int((~bad).sum()), len(full))
            if bad.any():
                raise AssertionError(f"basa: gross != net + tax on {int(bad.sum())} lines, "
                                     f"e.g. {full[bad].head(3).to_dict('index')}")

    neg = lines[(lines["amount_brl"] < 0) & ~lines["reversal"].astype(bool)]
    if len(neg):                                                               # check 4
        raise AssertionError(f"{len(neg)} negative amounts not marked as reversals, e.g. "
                             f"{neg[['bank_key', 'period', 'category', 'amount_brl']].head(3).to_dict('records')}")

    keyed = lines.dropna(subset=["content_key"])
    dup = keyed[keyed.duplicated(["bank_key", "content_key"], keep=False)]
    if len(dup):                                                               # check 5
        raise AssertionError(f"{len(dup)} lines share a printed-row key, so one printed row was "
                             f"emitted twice:\n{dup[['bank_key', 'content_key', 'source_file']].head(8).to_string(index=False)}")

    contained = contained_tables(lines)                                        # check 6
    log.info("  table instances whose (period, amount, detail) lines are contained in another "
             "instance of the same bank: %d", len(contained))
    if contained:
        raise AssertionError(
            f"{len(contained)} table instances repeat another table's lines, i.e. one printed "
            "table is counted more than once:\n"
            + "\n".join(f"  {b}: {i} ({n} lines) within {j}" for b, i, j, n in contained[:20]))


def contained_tables(lines: pd.DataFrame) -> list[tuple[str, str, str, int]]:
    """Table instances whose multiset of (period, amount, detail, category, agency) is contained
    in another instance's multiset for the same bank.

    This is the signature of one printed table counted twice - three copies of a BRB gazette page
    gave 19 such instances - whatever the file names or entity labels say. Where two instances are
    identical, only one of the pair is reported.

    Zero-amount lines are dropped from the signature first: a printed "R$ 0,00" cell repeats across
    many unrelated rows and months (a beneficiary paid in only one of three months still gets a
    line for each), so two DIFFERENT tables that both carry some zero cells can otherwise share
    enough of their (period, 0.0, ...) tuples to look contained in each other. `category` and
    `agency` are added to the key for every bank, not only BRB, so that two distinct single-entity
    tables whose rows happen to share a period, amount and beneficiary text do not read as one
    printed table counted twice."""
    from collections import Counter
    found = []
    for bank, grp in lines.groupby("bank_key"):
        nz = grp[grp["amount_brl"].round(2) != 0]
        ms = {t: Counter(zip(g["period"].astype(str), g["amount_brl"].round(2),
                             g["detail"].fillna(""), g["category"].fillna(""),
                             g["agency"].fillna("")))
              for t, g in nz.groupby("table_id")}
        index: dict[tuple, set[str]] = {}
        for t, c in ms.items():
            for k in c:
                index.setdefault(k, set()).add(t)
        for t, c in ms.items():
            candidates = set.intersection(*(index[k] for k in c)) - {t}
            for u in sorted(candidates):
                cu = ms[u]
                if not (c - cu) and (sum(c.values()) < sum(cu.values()) or c != cu or t > u):
                    found.append((bank, t, u, sum(c.values())))
                    break
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description="State-owned banks' published advertising spend")
    ap.add_argument("--banks", nargs="+", default=list(BANKS), choices=BANKS)
    ap.add_argument("--refresh", action="store_true", help="re-download listings and files")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--unvalidated", nargs="*", default=[], choices=BANKS,
                    help="banks allowed below the published-total floor; their rows carry "
                         "validated = False")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    docs = {b: collect(b, a.refresh) for b in a.banks}
    if a.download_only:
        return

    lines, totals, status = parse_all(docs)
    if lines.empty:
        raise SystemExit("no lines parsed from any file")
    # One table per stated-total scope for the single-entity banks; BRB's tables are named by the
    # page and position they start at.
    lines["table_id"] = lines["table_id"].fillna(lines["total_scope"])
    lines["amount_as_printed"] = lines["amount_as_printed"].fillna(lines["amount_brl"])
    lines["imputed"] = lines["imputed"].fillna(False).astype(bool)
    # BRB's entity CNPJ comes from its table's heading and stays empty where none was read
    single = lines["bank_key"] != "brb"
    lines.loc[single, "entity_cnpj8"] = lines.loc[single, "cnpj8"]
    lines = lines.drop(columns=[c for c in ("_table", "_month") if c in lines.columns])
    if "superseded" not in totals.columns:
        totals["superseded"] = False
    totals["superseded"] = totals["superseded"].fillna(False).astype(bool)
    if "_table" in totals.columns:
        totals = totals.rename(columns={"_table": "table_id"}).drop(columns=["_month"])
    checks = check_totals(lines, totals)

    # Panel codes come from the crosswalk, which verified them against the IF.data lists and
    # the market panel; nothing is hard-coded here.
    panel_key: dict[str, str] = {}
    xw_path = OUT_DIR / "advertising_entity_crosswalk.parquet"
    if xw_path.exists():
        xw = pd.read_parquet(xw_path)
        panel_key = (xw[(xw["source"] == "statebank") & xw["panel_code"].notna()]
                     .set_index("source_id")["panel_code"].to_dict())
        lines["panel_key"] = lines["bank_key"].map(panel_key)
        missing = sorted(set(lines.loc[lines["panel_key"].isna(), "bank_key"]))
        if missing:
            log.warning("no crosswalk panel code for %s", missing)
    else:
        lines["panel_key"] = None
        log.warning("no %s; panel_key left empty", xw_path.name)

    # Only the entities inside the bank's prudential conglomerate reach the panel. The four
    # single-entity banks publish for themselves; BRB's pages also carry Cartao BRB, which the
    # IF.data registry must place inside the conglomerate before it counts (`brb_membership`).
    lines["in_conglomerate"] = True
    is_brb = lines["bank_key"] == "brb"
    if is_brb.any():
        if not panel_key.get("brb"):
            raise SystemExit("no crosswalk panel code for brb; cannot test which of its entities "
                             "belong to its prudential conglomerate")
        lines.loc[is_brb, "in_conglomerate"] = brb_membership(lines[is_brb], panel_key["brb"])
    lines["in_conglomerate"] = lines["in_conglomerate"].astype(bool)
    lines["panel_exclusion"] = np.where(
        lines["in_conglomerate"], None,
        np.where(lines["entity"] == "unknown", "entity could not be read from the page",
                 "entity outside the prudential conglomerate in the IF.data registry"))
    out_of = lines[~lines["in_conglomerate"]]
    if len(out_of):
        log.info("lines kept but left out of the conglomerate totals:\n%s",
                 out_of.groupby(["bank_key", "entity", "entity_cnpj8"], dropna=False)
                       .agg(lines=("amount_brl", "size"), amount=("amount_brl", "sum"),
                            first=("period", "min"), last=("period", "max")).to_string())

    judged = checks[~checks["superseded"]] if not checks.empty else checks
    share_ok = judged.groupby("bank_key")["matches"].mean() if not judged.empty else pd.Series(dtype=float)
    lines["validated"] = lines["bank_key"].map(lambda b: bool(share_ok.get(b, 0) >= MIN_TOTAL_MATCH))
    periods = period_table(lines, checks)
    periods["validated"] = periods["bank_key"].map(
        lambda b: bool(share_ok.get(b, 0) >= MIN_TOTAL_MATCH))
    validate(lines, checks, status, tuple(a.unvalidated))
    cosif_ratios(periods, panel_key)

    log.info("coverage: months with lines per bank and year:\n%s",
             periods.assign(year=periods["period"].astype(str).str[:4])
                    .groupby(["bank_key", "year"])["period"].nunique().unstack(fill_value=0)
                    .to_string())

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out_lines = lines.drop(columns=[c for c in ("extra_scopes",) if c in lines.columns])
    for name, frame in (("statebank_advertising_lines", out_lines),
                        ("statebank_advertising_periods", periods),
                        ("statebank_advertising_documents", status),
                        ("statebank_advertising_total_checks", checks)):
        frame.to_parquet(a.out_dir / f"{name}.parquet", index=False)
        frame.to_csv(a.out_dir / f"{name}.csv", index=False)
        log.info("%s: %d rows -> %s", name, len(frame), a.out_dir / f"{name}.parquet")


if __name__ == "__main__":
    main()
