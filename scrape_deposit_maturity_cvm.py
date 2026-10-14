###─────────────────────────────────────────────────────────────────────────────
# scrape_deposit_maturity_cvm.py
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
#   to two filings (filing years Y and Y+1); the latest filing's values are kept
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
#        flagged=True in the log) and ranked below every unflagged candidate.
#   Layouts handled: instruments as rows and buckets as columns, bucket columns wrapped
#   over several header lines ("Sem" / "vencimento"), the total in the first column (BTG),
#   unitless day ranges ("0-30", "31-180", "Acima de 365"), rows named only "A Prazo" /
#   "À Vista" under a "Depósitos" heading (Banrisul, Itaú), and notes whose tables are
#   dated only by a bare year above the header (Daycoval, "2017" over "Banco" or
#   "Consolidado"). The statement of each table is read from the short row above it
#   ("BB Consolidado", "BB Banco Múltiplo", "Banco", "Consolidado") before the page text.
#   Unit cross-check: units not stated on the table's page are inherited from earlier
#   pages and can be wrong; a firm-year whose stated unit lies more than 30x from the
#   firm's median level over its R$-thousand years while R$ thousand lies within 30x is
#   corrected, with "UNIT CORRECTED a -> b" in notes and in the log.
#   Tables printed on a page that speaks of "não descontados" (undiscounted cash flows of
#   the liquidity-risk note, e.g. Itaú's "31-365 / 366-720 dias" tables, which add future
#   interest) rank below balance tables of the deposits note.
#   Tables with maturity buckets as ROWS and instruments as columns (Banco BMG's
#   "Vencimento de depósitos interfinanceiros e a prazo") are read by rowwise_tables(),
#   which needs a header row of four or more bare years and a "prazo" column title.
#   One table is kept per firm and fiscal year across ALL documents that report it:
#   candidates with a time-deposit row, then unflagged, then statement priority
#   consolidated > prudential > ifrs > individual, then a passing sum check, then the
#   number of buckets, then the latest filing. Output column table_id (pdf | page |
#   statement | as-of | header text) identifies the chosen table; the QA file compares only
#   tables of the same statement across filings.
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
#   python scrape_deposit_maturity_cvm.py                       # all firms, 2016-2024
#   python scrape_deposit_maturity_cvm.py --firms bb,banrisul --years 2020-2022
#   python scrape_deposit_maturity_cvm.py --refresh             # re-download and re-parse
###─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import io
import json
import logging
import os
import re
import sys
import time
import unicodedata
import urllib.request
import http.cookiejar
import zipfile

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import fitz  # PyMuPDF
from utils.paths import RAW

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("scrape_deposit_maturity_cvm")

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

FIRMS = {  # firm_key -> (CNPJ root, display)
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
PARSER_VERSION = 16   # bump when page parsing changes; cached parses of other versions are redone
STMT_RANK = {"consolidated": 0, "prudential": 1, "ifrs": 2, "individual": 3, "unknown": 4}

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
            log.warning(f"GET failed ({a}/{retries}) {url[:110]}: {e}")
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
            log.warning(f"{kind} {year} index unreadable: {e}")
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


def parse_num(tok):
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


def parse_bucket(label):
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


def instrument_of(label):
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
        v = parse_num(w[4])
        if v is not None and (label_parts or vals):
            if not vals and re.fullmatch(r"\(?\d{1,2}\)?", w[4]) and len(w[4]) <= 3 and w[4].startswith("("):
                continue  # footnote marker
            vals.append(((w[0] + w[2]) / 2, v))
        elif v is None and not vals:
            label_parts.append(w[4])
        elif v is None and vals and w[4].endswith("%"):
            continue
    return " ".join(label_parts).strip(), vals


def segment_header(words):
    """Reading-order header words -> list of (kind, lo, hi, text) phrases."""
    phrases, cur = [], []
    closers = re.compile(r"^(dias?|mes|meses|ano|anos|vencimento|total|%|vista|geral)$")
    for w in words:
        f = _fold(w)
        if not (_VOCAB.match(w) or _VOCAB.match(f)):
            cur = []
            continue
        cur.append(f)
        if closers.match(f) or (f == "prazo" and len(cur) >= 2 and cur[-2] == "sem"):
            pb = parse_bucket(" ".join(cur))
            if pb:
                phrases.append(pb + (" ".join(cur),))
            cur = []
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
        elif parse_num(t) is not None and re.search(r"[.,()]", t):
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
    seq = segment_header([w[4] for w in rows[anchor]])
    nb = sum(1 for p in seq if p[0] == "bucket")
    n_seq = sum(1 for p in seq if p[0] != "pct")
    if nb >= 2 and len(val_x) >= nb and len(val_x) - n_seq <= 2:
        # the phrases cover every value column but at most the total and a prior-year
        # total; fewer phrases means a wrapped header word ("Sem" / "vencimento") was lost
        cols = [p for p in seq if p[0] != "pct"]
        cols = cols[:len(val_x)]
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
        pb = parse_bucket(text)
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


def stmt_above(rows, lo, hi):
    """Statement named by a short row just above a table ("BB Consolidado", "Banco",
    "BB Banco Múltiplo", "Conglomerado Financeiro"); None when there is none."""
    found = None
    for rr in rows[max(lo, 0):hi]:
        t = re.sub(r"\d{1,2}[./]\d{1,2}[./]\d{4}|\b20[12]\d\b", " ", _fold(row_text(rr)))   # dates, bare years
        t = re.sub(r"\s+", " ", t).strip(" :-")
        toks = t.split()
        if len(toks) > 14:
            continue
        if "prudencial" in t:
            found = "prudential"
        elif "consolidad" in t or "conglomerado financeiro" in t:
            found = "consolidated"
        elif re.fullmatch(r"(bb )?banco( multiplo)?|controladora|individual|banco [a-z ]+", t) or (
                toks and toks[0] == "banco" and len(toks) <= 4
                and not re.match(r"banco (central|do|de|bmg|pan|abc|btg|daycoval|santander|bradesco|itau)", t)
                and not any(re.search(r"\d", q) for q in toks[1:2])):
            found = "individual"
    return found


_SEC_ROW = re.compile(r"^(a prazo|prazo|a vista|vista|poupanca|de poupanca|interfinanceiros?|outros depositos)$")


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
        above = [w for rr in rows[max(yi - 4, 0):yi] for w in rr]
        if not any("prazo" in _fold(w[4]) for w in above) or not any(_fold(w[4]).startswith("deposito") for w in above):
            continue   # the header must name time deposits, not (say) a loan-portfolio maturity table
        cols = []
        for w in ys:
            xc = (w[0] + w[2]) / 2
            # a footnote marker may be glued to the title ("prazo(i)")
            near = [(abs((a_[0] + a_[2]) / 2 - xc), re.sub(r"\(.*?\)", "", _fold(a_[4]))) for a_ in above
                    if re.sub(r"\(.*?\)", "", _fold(a_[4])) in ("interfinanceiros", "prazo", "total", "vista")]
            title = min(near)[1] if near else "?"
            cols.append((xc, title, w[4]))
        data = []
        for rr in rows[yi + 1:yi + 16]:
            xmin = min(c[0] for c in cols) - 35
            label = " ".join(w[4] for w in rr if (w[0] + w[2]) / 2 < xmin)
            vals = [((w[0] + w[2]) / 2, parse_num(w[4])) for w in rr
                    if (w[0] + w[2]) / 2 >= xmin and parse_num(w[4]) is not None]
            lf = _fold(label).strip()
            if lf.startswith("total"):
                data.append(("total", None, None, vals))
                break
            if vals and "circulante" not in lf:
                pb = parse_bucket(label)
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
            out.append(dict(row_y=rows[yi][0][1], label="Depósitos a prazo (maturity rows)", asof=f"{yr_s}-12-31",
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
        # the statement heading ("Banco" / "Consolidado") sits above the FIRST table of a note
        # that stacks several year blocks under it; the nearest heading above wins
        stmt_t = stmt_above(rows, top - 4, ri) or stmt_above(rows, top - 60, top - 4) or stmt
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
            tables.append(dict(row_y=rows[ri][0][1], label=label, asof=asof, stmt=stmt_t, unit=unit,
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
    state = {"stmt": "unknown", "unit": None}
    out = []
    for i in range(n):
        page = doc[i]
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
    """Collapse per-row detections into tables: same page, asof, bucket labels."""
    groups = {}
    for t in tables:
        sig = (t["page"], t["asof"], tuple((b["label"]) for b in t["buckets"]), t["stmt"])
        g = groups.setdefault(sig, dict(page=t["page"], asof=t["asof"], stmt=t["stmt"], unit=t["unit"],
                                        method=t["method"], rows=[], ctx=t["ctx"],
                                        undisc=t.get("undisc", False)))
        instr = instrument_of(t["label"])
        if any(r["instrument"] == instr for r in g["rows"]):
            continue  # a second block under the same header (e.g. compulsory-deposit deductions)
        g["rows"].append(dict(label=t["label"], instrument=instr, buckets=t["buckets"], total=t["total"]))
    return list(groups.values())


SUM_TOL = 0.005       # buckets must add to the printed total within 0.5%
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


def table_score(g, flagged=False):
    td = [r for r in g["rows"] if r["instrument"] == "time_deposits"]
    td_ok = any((check_row(r)[0] is not None and check_row(r)[0] <= SUM_TOL) for r in td)
    nb = max(len(r["buckets"]) for r in g["rows"])
    return (not td, flagged, g.get("undisc", False), STMT_RANK.get(g["stmt"], 4), not td_ok, -nb)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--firms", default=",".join(FIRMS))
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
                if cnpj_root(r["CNPJ_CIA"]) == FIRMS[fk][0] and r["DT_REFER"].endswith("12-31"):
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
            log.info(f"[{fk}] DFP {ref}: {len(res['tables'])} candidate tables"
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
            cands = [r for r in rows if cnpj_root(r["CNPJ_Companhia"]) == FIRMS[fk][0]
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
                    log.warning(f"[{fk}] IPE {fy}: {err}")
                    continue
                res = parsed_for(pdf, fk, a.refresh)
                for g in res["tables"]:
                    fy_g = int(g["asof"][:4]) if g["asof"] else fy
                    g["asof"] = g["asof"] or f"{fy}-12-31"
                    results.setdefault((fk, fy_g), []).append(
                        dict(doc_year=fil_year, source="cvm_ipe", url=r["Link_Download"], pdf=pdf.name, g=g))
            if has_td((fk, fy)):
                break

    # selection: one table per (firm, fy) over all documents. Candidates with a
    # time-deposit row come first, then plausible ones, statement priority, passing sum
    # check, number of buckets, and finally the latest filing.
    out_rows, qa_rows, log_rows = [], [], []
    for fk in firms:
        for fy in range(y0, y1 + 1):
            cand = results.get((fk, fy), [])
            for x in cand:
                x["flag"] = plausibility(x, [y for y in cand if y["pdf"] == x["pdf"]])

            def has_td_row(x):
                return any(r["instrument"] == "time_deposits" for r in x["g"]["rows"])
            tds = [x for x in cand if has_td_row(x)]
            if not tds:
                dfp_ok = [status.get((fk, ry), (False, "")) for ry in (fy, fy + 1)]
                msg = "; ".join(f"DFP{ry}: {m}" for ry, (_, m) in zip((fy, fy + 1), dfp_ok))
                log_rows.append(dict(firm=fk, year=fy, document_found=any(f for f, _ in dfp_ok),
                                     page_found=bool(cand), method="",
                                     status="no time-deposit maturity table: " + msg,
                                     table_id="", sum_check="", flagged=""))
                continue
            tds.sort(key=lambda x: (table_score(x["g"], bool(x["flag"])), -x["doc_year"]))
            best = tds[0]
            g = best["g"]
            stmt = g["stmt"]
            note_flag = best["flag"]
            tid = table_id(best)
            for r in g["rows"]:
                chk, tot = check_row(r)
                for b in r["buckets"]:
                    out_rows.append(dict(
                        firm_key=fk, cnpj_root=FIRMS[fk][0], fiscal_year=fy, as_of=f"{fy}-12-31",
                        source=best["source"], doc_url=best["url"], page=g["page"], statement=stmt,
                        table_id=tid,
                        table_title=g["ctx"].strip(), instrument=r["instrument"], bucket_label=b["label"],
                        lo_days=b["lo"], hi_days=b["hi"], value=b["value"], unit=g["unit"] or "unknown",
                        extraction_method=g["method"], total_reported=tot,
                        sum_check="" if chk is None else round(chk, 6),
                        notes=f"row={r['label']}; filing_year={best['doc_year']}; pdf={best['pdf']}"
                              + ("; as_of inferred from filing year" if g.get("_asof_inferred") else "")
                              + (f"; FLAG {note_flag}" if note_flag else "")))
            td = [r for r in g["rows"] if r["instrument"] == "time_deposits"][0]
            gap = check_row(td)[0]
            gap_s = "no_total" if gap is None else f"{gap:.6f}"
            log_rows.append(dict(firm=fk, year=fy, document_found=True, page_found=True,
                                 method=g["method"],
                                 status=(f"{'FLAGGED' if note_flag else 'ok'}; {best['source']} filing "
                                         f"{best['doc_year']} p.{g['page']} {stmt}; time-deposit sum_check={gap_s}"
                                         + (f"; {note_flag}" if note_flag else "")),
                                 table_id=tid, sum_check="" if gap is None else round(gap, 6),
                                 flagged=bool(note_flag)))
            # QA across filings: same statement, one table per document
            by_doc = {}
            for x in tds:
                if x["g"]["stmt"] != stmt or bool(x["g"].get("undisc")) != bool(g.get("undisc")):
                    continue   # like with like: same statement, same kind of table (balances vs undiscounted flows)
                k = (x["source"], x["doc_year"], x["pdf"])
                if k not in by_doc:
                    by_doc[k] = x

            def key_vals(x):
                rs = [r for r in x["g"]["rows"] if r["instrument"] == "time_deposits"]
                sc_ = _UNIT_SCALE.get(x["g"]["unit"] or "", 1.0)   # compare in R$ thousand
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

    # unit cross-check: a page without an explicit unit inherits the last one stated in the
    # document, which can be wrong (a "R$ milhões" page earlier in the filing). Each firm's
    # time-deposit series is compared across years in R$ thousand; where the stated unit
    # puts a year more than 30x away from the firm's median and another unit lands within
    # 30x, the unit is corrected and the correction is written to notes and the log.
    import math
    td_tot = {}
    for r in out_rows:
        if r["instrument"] == "time_deposits" and r["total_reported"] not in (None, ""):
            td_tot[(r["firm_key"], r["fiscal_year"])] = (r["total_reported"], r["unit"])
    corrected = {}
    for fk in firms:
        thou = sorted(v for (f, fy), (v, u) in td_tot.items() if f == fk and v > 0 and u == "R$ thousand")
        if len(thou) < 3:
            continue
        med = thou[len(thou) // 2]   # reference level: the years stated in R$ thousand
        for (f, fy), (v, u) in td_tot.items():
            if f != fk or v <= 0 or u == "R$ thousand":
                continue
            if abs(math.log(v * _UNIT_SCALE.get(u, 1.0) / med)) > math.log(30) and                     abs(math.log(v / med)) <= math.log(30):
                corrected[(fk, fy)] = (u, "R$ thousand")
    for r in out_rows:
        k = (r["firm_key"], r["fiscal_year"])
        if k in corrected:
            r["notes"] += f"; UNIT CORRECTED {corrected[k][0]} -> {corrected[k][1]} (stated unit inconsistent with the firm's series)"
            r["unit"] = corrected[k][1]
    for r in log_rows:
        k = (r["firm"], r["year"])
        if k in corrected:
            r["status"] += f"; UNIT CORRECTED {corrected[k][0]} -> {corrected[k][1]}"

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
        log.info(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
