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
                "source_file", "source_url", "parse_method", "page_or_sheet"]


@dataclass
class Parsed:
    """What one file yields: lines, stated totals and the (entity, period) pairs it covers."""
    lines: list[dict] = field(default_factory=list)
    totals: list[dict] = field(default_factory=list)
    periods: set = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


def line(doc: Doc, **kw) -> dict:
    rec = {c: None for c in LINE_COLUMNS}
    rec.update(bank_key=doc.bank, entity=doc.entity, agency=doc.agency, source_url=doc.url,
               source_file=doc.entry.get("file"), reversal=False, extra_scopes=[])
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


def parse_basa_2016(doc: Doc, path: Path) -> Parsed:
    """'Relatorio Consolidado Producao e Veiculacao 2016': a month x item table in two halves
    (January-June, July-December plus TOTAL GERAL), sections PRODUCAO and VEICULACAO with
    SUBTOTAL rows and a TOTAL GERAL/MES row."""
    out = Parsed()
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
            period = "2016" if col == 13 else f"2016{col:02d}"
            flabel = fold(label)
            if flabel.startswith("TOTAL GERAL"):
                out.totals.append({"scope": f"{path.name}#{period}", "stated": v,
                                   "label": f"TOTAL GERAL/MES {period}"})
            elif flabel == "SUBTOTAL":
                out.totals.append({"scope": f"{path.name}#{section}:{period}", "stated": v,
                                   "label": f"SUBTOTAL {section} {period}"})
            elif col != 13:
                group = ("advertising" if section == "VEICULACAO" else
                         "production" if flabel == "FORNECEDOR" else "other")
                out.lines.append(line(
                    doc, period=period, frequency="monthly", period_rule="document month column",
                    section=section, category=label, category_group=group, amount_brl=v,
                    amount_kind="total", basis="paid", total_scope=f"{path.name}#{period}",
                    extra_scopes=[f"{path.name}#{section}:{period}", f"{path.name}#{section}:2016",
                                  f"{path.name}#2016"],
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


def parse_basa(doc: Doc, path: Path) -> Parsed:
    name = fold(doc.label)
    if doc.entry["kind"] in ("xlsx", "xls", "text") and "ESCALA" in fold(doc.url):
        return parse_basa_escala(doc, path)
    if "MEIO DE DIVULGA" in name and doc.year and doc.year >= 2019:
        return parse_basa_resumo(doc, path)
    if doc.year == 2016:
        return parse_basa_2016(doc, path)
    if doc.year == 2017 and "1_TRIMESTRE" in fold(doc.url).replace(" ", "_"):
        return parse_basa_2017(doc, path)
    out = Parsed()
    out.notes.append("not a values source for this script (cross-check or names-only file)")
    return out


# ---------------------------------------------------------------------------
# BRB
# ---------------------------------------------------------------------------
BRB_ENTITIES = (("financeira", r"CREDITO,? FINANCIAMENTO|FINANCEIRA BRB"),
                ("corretora", r"DISTRIBUIDORA DE TITULOS|\bDTVM\b"),
                ("bank", r"BANCO DE BRASILIA"))
AMOUNT_RE = r"-?R?\$?\s?-?\d[\d.]*,\d{2}"


def brb_group(classification: str, purpose: str, sponsorship_table: bool) -> str:
    """Explicit rule on the published CLASSIFICACAO DA DESPESA and FINALIDADE DA ACAO:
    'PUBLICACOES OBRIGATORIAS' or 'PUBLICIDADE LEGAL' -> legal_notice; a row of a table with no
    propaganda/publicidade/publicacoes row (Esporte, Arte e Cultura, Entretenimento,
    Relacionamento Institucional, ...) -> sponsorship; '.../PRODUCAO' -> production;
    '.../VEICULACAO' or plain 'PROPAGANDA E PUBLICIDADE' -> advertising; anything else -> other."""
    c = fold(classification).replace(" ", "")
    f = fold(purpose).replace(" ", "")
    if "PUBLICAC" in c or "LEGAL" in c or "LEGAL" in f:
        return "legal_notice"
    if "PATROCINIO" in c or sponsorship_table:
        return "sponsorship"
    if "PRODUCAO" in c:
        return "production"
    if "VEICULA" in c or "PROPAGANDA" in c or "PUBLICIDADE" in c:
        return "advertising"
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


def brb_cell_amount(cell) -> float | None:
    s = re.sub(r"\s+", "", str(cell or "")).replace("R$", "")
    if s in ("", "None"):
        return None
    if re.fullmatch(r"-+", s):
        return 0.0
    return brl(s)


MONTH_KEYS = {k[:3].upper() for k in MONTHS_PT}


def brb_month_cols(row: list) -> dict[int, int]:
    cols = {}
    for j, c in enumerate(row):
        k = re.sub(r"[\s\-.]", "", str(c or ""))
        if k and len(k) <= 9 and fold(k)[:3] in MONTH_KEYS and month_number(k):
            cols[j] = month_number(k)
    return cols if len(cols) >= 2 else {}


def split_merged_amounts(row: list, cols: list[int]) -> list:
    """'83.333,33 249.999,99' in one cell with the next column empty: the PDF merged two cells."""
    row = list(row)
    for j in cols:
        if j >= len(row):
            continue
        parts = re.findall(AMOUNT_RE, str(row[j] or ""))
        if len(parts) == 2 and j + 1 < len(row) and not str(row[j + 1] or "").strip():
            row[j], row[j + 1] = parts[0], parts[1]
    return row


def parse_brb(doc: Doc, path: Path) -> Parsed:
    """QDD 'Demonstrativo das despesas com propaganda, publicidade, publicacoes legais e
    patrocinios'. Tables have BENEFICIARIO, CLASSIFICACAO DA DESPESA, FINALIDADE DA ACAO, month
    columns and TOTAL (R$), and close with a total row carrying month values ('TOTAL R$',
    'Valor Total', 'TOTAL PAGO NO 4o TRIMESTRE') followed by cumulative rows ('TOTAL PAGO NO
    1o TRIMESTRE', 'TOTAL CONTABILIZADO EM 2025', 'TOTAL REALIZADO ...') that state the basis."""
    import pdfplumber
    out = Parsed()
    with pdfplumber.open(path) as pdf:
        full = fold("\n".join((p.extract_text() or "") for p in pdf.pages))
        if len(re.sub(r"[^A-Z]", "", full)) < 200 or "TRIMESTRE" not in full \
                or not re.search(r"JANEIRO|ABRIL|JULHO|OUTUBRO", full):
            out.notes.append("unreadable: no text layer, or a font encoding without a Unicode map")
            return out
        entity, year, quarter = brb_heading(full)
        if entity is None:
            entity = doc.entity
            out.notes.append("entity from the listing: the document names none")
        if entity != doc.entity:
            out.notes.append(f"document names entity {entity}, listing says {doc.entity}")
        year = year or doc.year
        state = {"entity": entity, "year": year, "quarter": quarter, "month_cols": None,
                 "total_col": None, "ncols": None, "scope": None, "sponsorship": None,
                 "basis": None}
        bases: dict[str, str | None] = {}
        tables_found = 0
        for pno, page in enumerate(pdf.pages, start=1):
            for tobj in page.find_tables():
                rows = tobj.extract()
                if not rows:
                    continue
                top = tobj.bbox[1]
                above = page.crop((0, max(0, top - 160), page.width, top)).extract_text() or ""
                e, y, q = brb_heading(above)
                if y:
                    state.update(year=y, quarter=q)
                if e:
                    state["entity"] = e
                hdr = next((i for i, r in enumerate(rows[:4]) if brb_month_cols(r)), None)
                new_table = hdr is not None or any(
                    re.search(r"BENEFICI|VALORES REALIZADOS", fold(str(c or ""))) for c in rows[0])
                if hdr is not None:
                    mc = brb_month_cols(rows[hdr])
                    tc = next((j for j, c in enumerate(rows[hdr]) if j > max(mc)
                               and "TOTAL" in fold(re.sub(r"\s", "", str(c or "")))), None)
                    state.update(month_cols=mc, total_col=tc, ncols=len(rows[hdr]))
                    body = rows[hdr + 1:]
                elif state["month_cols"] and len(rows[0]) == state["ncols"]:
                    body = rows[1:] if new_table else rows
                else:
                    continue
                if new_table:
                    tables_found += 1
                    state.update(scope=f"{path.name}#t{tables_found}", sponsorship=None, basis=None)
                scope, mc, tc = state["scope"], state["month_cols"], state["total_col"]
                first_m = min(mc)
                amount_cols = sorted(mc) + ([tc] if tc is not None else [])
                classes = " ".join(fold(" ".join(str(c or "") for c in r[:first_m])) for r in body)
                if state["sponsorship"] is None:
                    state["sponsorship"] = not re.search(r"PROPAGANDA|PUBLICIDADE|PUBLICAC|P U B L",
                                                         classes)
                sponsorship = state["sponsorship"]
                prev = [""] * first_m
                month_stated: dict[int, float] = {}
                for r in body:
                    r = split_merged_amounts(r, amount_cols)
                    labels = [str(r[j] or "").replace("\n", " ").strip() for j in range(first_m)]
                    flab = fold(" ".join(labels))
                    amounts = {mo: brb_cell_amount(r[j]) for j, mo in mc.items() if j < len(r)}
                    if re.match(r"(VALOR )?TOTAL\b", flab.strip(" .")) or re.search(r"\bTOTAL (R\$|PAGO|CONTABILIZADO|REALIZADO)", flab):
                        if "CONTABILIZADO" in flab:
                            state["basis"] = "accrued"
                        elif "PAGO" in flab:
                            state["basis"] = "paid"
                        elif "REALIZADO" in flab and state["basis"] is None:
                            state["basis"] = "undocumented"
                        if any(v is not None for v in amounts.values()):
                            for mo, v in amounts.items():
                                if v is not None:
                                    month_stated[mo] = v
                                    out.totals.append({"scope": f"{scope}:{mo:02d}", "stated": v,
                                                       "label": f"{flab[:30]} month {mo}"})
                            qv = brb_cell_amount(r[tc]) if tc is not None and tc < len(r) else None
                            if qv is not None and abs(qv - sum(month_stated.values())) <= 0.05:
                                out.totals.append({"scope": f"{scope}:quarter", "stated": qv,
                                                   "label": f"{flab[:30]} quarter"})
                            elif qv is not None:
                                out.notes.append(f"{scope}: TOTAL column {qv:,.2f} is not the sum of "
                                                 "the months (year-to-date); not used as a check")
                        bases[scope] = state["basis"]
                        continue
                    if all(v is None for v in amounts.values()):
                        continue
                    labels = [lab or prev[j] for j, lab in enumerate(labels)]
                    prev = labels
                    beneficiary = labels[0]
                    classification = labels[1] if first_m >= 2 else ""
                    purpose = " ".join(labels[2:]) if first_m >= 3 else ""
                    group = brb_group(classification, purpose, sponsorship)
                    for mo, v in amounts.items():
                        if v is None:
                            continue
                        period = f"{state['year']}{mo:02d}"
                        rec = line(
                            doc, entity=state["entity"], period=period, frequency="monthly",
                            period_rule="document month column",
                            agency=beneficiary if group in ("advertising", "production") else None,
                            section="PATROCINIOS" if sponsorship else "PUBLICIDADE",
                            category=classification, detail=f"{beneficiary} | {purpose}".strip(" |"),
                            category_group=group, amount_brl=v, amount_kind="total",
                            reversal=v < 0, total_scope=f"{scope}:{mo:02d}",
                            extra_scopes=[f"{scope}:quarter"], line_id=f"{scope}:{len(out.lines)}",
                            page_or_sheet=f"p{pno}", parse_method="pdfplumber table cells")
                        rec["_table"] = scope
                        out.lines.append(rec)
                        out.periods.add((state["entity"], period))
                bases.setdefault(scope, state["basis"])
        for ln in out.lines:
            ln["basis"] = bases.get(ln.pop("_table")) or "undocumented"
        # A table published as a total row only (no detail rows): keep the published total.
        with_lines = {ln["total_scope"] for ln in out.lines}
        for t in list(out.totals):
            if t["scope"] in with_lines or t["scope"].endswith(":quarter"):
                continue
            tscope, mo = t["scope"].rsplit(":", 1)
            period = f"{state['year']}{int(mo):02d}"
            out.lines.append(line(
                doc, entity=state["entity"], period=period, frequency="monthly",
                period_rule="document month column", section="PUBLICIDADE",
                category=t["label"].split(" month")[0], detail="total published without detail rows",
                category_group="other", amount_brl=t["stated"], amount_kind="total",
                basis=bases.get(tscope) or "undocumented", total_scope=t["scope"],
                line_id=f"{t['scope']}:total_only", page_or_sheet=None,
                parse_method="pdfplumber table cells (total row only)"))
            out.periods.add((state["entity"], period))
        if tables_found == 0:
            out.notes.append("unreadable: text layer present but no table with month columns")
        out.meta = {"entity": entity, "year": year, "quarter": quarter}
    return out


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
                agency = " ".join(w["text"] for w in r if w["x1"] < r[[w["text"] for w in r]
                                                                    .index(m.group(0))]["x0"])
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


def main() -> None:
    ap = argparse.ArgumentParser(description="State-owned banks' published advertising spend")
    ap.add_argument("--banks", nargs="+", default=list(BANKS), choices=BANKS)
    ap.add_argument("--refresh", action="store_true", help="re-download listings and files")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    docs = {b: collect(b, a.refresh) for b in a.banks}
    if a.download_only:
        return


if __name__ == "__main__":
    main()
