"""
scrape_bank_notes_advertising.py
================================
Advertising expense read from the NOTES to Brazilian banks' audited financial statements.

WHY THIS ROUTE EXISTS
---------------------
The central bank publishes the advertising accounts per institution only from January 2025, and
the securities regulator's STRUCTURED files (`scrape_cvm_advertising.py`) expose only the DVA and
DRE faces, where most banks show no advertising line at all. Banco Santander (Brasil) is 9.5% of
panel deposits, files with the regulator every quarter, and has no advertising line in any
structured year. Its figure is in the notes: note "Outras Despesas Administrativas" itemises
"Propaganda, Promocoes e Publicidade", which is the same accounting object as COSIF account
`8174500005`, and the notes go back to 2013.

`utils/note_sources.py` is the authorisation surface: the approved hosts and the eighteen targets.
Every request this module sends goes through `fetch`, and what that guarantees is exactly this:
  * the URL requested, and the target of every redirect BEFORE it is followed, must pass
    `allowed_host()`. A typo raises before anything is sent; a redirect onto any other host raises
    `UnapprovedRedirect`, the run stops, and nothing is sent to that host;
  * the URL the answer finally came from and every redirect hop are recorded in the bank's
    manifest (`final_url`, `redirects`).
What it does NOT guarantee: the check is on host NAMES, so where an approved name leads (its DNS
records, a CDN behind a CNAME, a proxy set in the environment, which urllib honours) is not
checked; a redirect that keeps an approved host may change the scheme, https to http; and nothing
is known about other code: `utils.disclosure_common.http_get`, which the other scrapers use,
follows a redirect to any host and is not used here. Redirects written into a page (a meta refresh,
a script) are never followed at all; only a listing's `.pdf` links are requested, each checked
first like any other URL.

TWO ROUTES
----------
`cvm_rad`   A securities-regulator filer. The regulator's own document system serves the complete
            DFP/ITR delivery - notes included - as one ZIP whose PDF member is the published
            statements. The document ids come from the register CSV inside the yearly DFP/ITR zips
            already cached for `scrape_cvm_advertising.py` (`LINK_DOC` there points at exactly this
            endpoint), so no search interface is involved:

                https://www.rad.cvm.gov.br/ENET/frmDownloadDocumento.aspx
                    ?CodigoInstituicao=1&NumeroSequencialDocumento=<ID_DOC>

            The regulator's own `LINK_DOC` spelling (`/ENETCONSULTA/frmDownloadDocumento.aspx`) is
            dead: it answers 200 with a WCF error page. `/ENET/` and `/ENETWeb/` both serve, and
            `/ENET/` is used here. The search page `/ENET/frmConsultaExternaCVM.aspx` was
            discontinued on 06/07/2026 in favour of `/ENETWeb/...`, whose `ListarDocumentos` web
            method answers but returns an empty result set for a company query, which is the
            second reason the register CSVs are the index of record.

`ir_site`   Not a filer, or a filer whose own site is easier. The bank publishes the same statements
            on its own domain, because the central bank requires publication regardless of listing.

ACCOUNTING RULES OBSERVED
-------------------------
* Notes are stated in R$ THOUSANDS ("Em milhares de reais", "*Valores expressos em milhares").
  The scale is read per document and per note, never assumed, and stored beside the value.
* The figures are period-cumulative. A note column is one WINDOW, and the window is read from the
  column header verbatim ("01/01 a 30/06/2025", "2o Semestre", "Exercicio", "31/12/2024"), stored as
  `period_rule` with explicit `window_start`/`window_end`. Increments are DERIVED from windows in
  `period_table`; no figure is ever assumed to be a quarter.
* INDIVIDUAL statements are the measure (`scope = individual`, the "Banco"/"Controladora" column).
  The consolidated column is parsed too and kept, marked, so that a group where only the holding
  files can fall back to it with the fallback visible.
* Where a figure is restated, the LAST FILED value wins: documents are ordered by their delivery
  date and the highest version of a reference date supersedes the lower ones.
* The published label is kept verbatim in `label_as_published`. Advertising, promotions/public
  relations and publications are distinct accounts and are not merged; `category_group` records
  which of them a label names, including `adv_promo_publ_bundled` where the bank publishes one
  line covering several (Santander's "Propaganda, Promocoes e Publicidade").

NO OCR. A PDF page with no text layer is recorded as `unreadable_no_text_layer` and never guessed
at; the vision tier is a separate pending decision.

OUTPUTS (parquet and csv, in `paths.AWARENESS_PROC`)
  bank_notes_advertising_lines        one row per note line per column
  bank_notes_advertising_periods      one row per entity-window at the finest derivable frequency
  bank_notes_advertising_documents    one row per listed document, with its HTTP and parse status
  bank_notes_advertising_total_checks the note's own arithmetic: items against the stated total

RE-RUNNING
  python scrape_bank_notes_advertising.py                      # every route, every target
  python scrape_bank_notes_advertising.py --banks santander    # one bank, merged into the outputs
  python scrape_bank_notes_advertising.py --ref-months 6 12 --out-dir <folder>   # a partial read
  python scrape_bank_notes_advertising.py --download-only      # fill the cache, parse nothing
Run with the project interpreter (C:/venvs/egan/Scripts/python.exe).

What a run may write where. The output file names are fixed, so a partial run written over them
would silently replace the combined table with its subset. Hence:
  * an unrestricted run replaces every output, and removes an output it produced no rows for, so a
    stale file never looks current;
  * a run restricted ONLY by --banks and/or --routes merges: it replaces every row of the banks it
    processed and keeps every other bank's rows as they were;
  * a run with any other restriction (--from-year, --to-year, --ref-months, --max-docs, --versions)
    is a partial reading of the banks it touches, so it refuses to write into the default output
    folder and needs --out-dir. The default folder is recognised by identity (`os.path.samefile`),
    not by spelling.
Every file, outputs and cache alike, is written to a temporary name and renamed into place, so a
killed run leaves the previous version of each file whole. The output tables are one set:
  * a lock file in the output folder is held from reading the previous outputs (a merge) to the
    last rename, so two runs writing into one folder take turns instead of overwriting each other;
  * before the first rename every target is checked to be replaceable, so a file held open by
    another program (a csv open in Excel) stops the write before any table is replaced. A failure
    after that point is logged with the tables already replaced. A run killed during the renames
    can still leave some tables new and some old.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import gzip
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from utils import paths
# The shared User-Agent and per-host throttle, so this module's requests look and pace exactly like
# the other disclosure scrapers'. Their `http_get` is not used: see `fetch`.
from utils.disclosure_common import USER_AGENT, _throttle, now_iso
from utils.note_sources import NOTE_TARGETS, allowed_host, targets, unresolved

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

RAW_DIR = paths.AWARENESS_RAW / "bank_notes"
OUT_DIR = paths.AWARENESS_PROC
CVM_REGISTER_DIR = paths.FIRM_DISCLOSURES / "CVM" / "raw"   # the cache scrape_cvm_advertising fills

# The regulator's document endpoint. One host serves every filer.
RAD_DOC = ("https://www.rad.cvm.gov.br/ENET/frmDownloadDocumento.aspx"
           "?CodigoInstituicao=1&NumeroSequencialDocumento={seq}")

# What a good body is, per kind of request. Anything else is a failed fetch whatever status the
# server gave it, and the cache fetches it again on the next run instead of serving it.
DOC_KINDS = ("zip", "pdf")
LISTING_KINDS = ("html",)
# The regulator's document host reports "busy" with HTTP 200 and a plain-text body, not with an
# error status. Fifteen of these were cached as statements in September 2026 (Santander 2, BTG 7,
# Daycoval 6) and then reported as byte-identical duplicates of one another. They are named
# explicitly so the manifest says why the fetch failed, not only that the body was not a document.
SERVICE_ERROR_MARKERS = (
    b"The HTTP service located at",             # the back-end download service is down or saturated
    b"The underlying connection was closed",    # the front end dropped its back-end connection
)
HEAD_BYTES = 1200                   # what the kind of a body is decided from
# A PDF ends with "%%EOF". All 66 own-site PDFs cached in September 2026 (Volkswagen 47, Safra 19)
# have it within 5 bytes of their last byte other than padding; a download cut short does not, so
# a truncated PDF is refused before it is cached as good instead of failing later when opened.
# Only a PDF body is tested this way: the regulator's documents arrive as ZIPs, whose damage the
# archive's own CRC shows. The tail is read long so that padding after the marker is skipped.
PDF_EOF_WINDOW = 2048
TAIL_BYTES = 65536
PDF_PADDING = b"\x00\t\n\f\r "
# A body that failed a check is kept for inspection under the document's name plus this suffix.
# The document's own name only ever holds a body that passed, so a failure cannot overwrite it.
FAILED_SUFFIX = ".failed"
# A statement link on an own-site listing page, exactly as the discovery functions read it.
PDF_HREF_RE = re.compile(r'href=["\']([^"\'#]+\.pdf)', re.I)
# A refreshed listing linking fewer than this share of the PDFs its cached copy links is taken for
# a degraded page (a partial render, a trimmed error page), not for the bank's new listing.
LISTING_MIN_SHARE = 0.5

# The output tables, in the order they are written. Every one carries `bank_key`, which is what a
# merge run replaces by.
OUTPUT_TABLES = ("bank_notes_advertising_lines", "bank_notes_advertising_periods",
                 "bank_notes_advertising_documents", "bank_notes_advertising_total_checks",
                 "bank_notes_advertising_cosif_check", "bank_notes_advertising_structured_check")
# Held by `write_outputs` from reading the previous outputs to the last rename. A second run
# waits this long for it: another run's write takes seconds, and giving up would discard a
# finished run's work.
OUTPUT_LOCK_NAME = "bank_notes_advertising.lock"
OUTPUT_LOCK_WAIT = 600.0

# Own-site listing pages, one per ir_site bank actually reachable. Probed 2026-09-22.
URL_VWFS_LIST = ("https://www.vwfs.com.br/volkswagen-financial-services/relacionamento-investidor/"
                 "demonstracoes-financeiras.html")
URL_PAGBANK_LIST = "https://www.pagbank.com.br/demonstracoes-financeiras"
URL_PICPAY_LIST = "https://www.picpay.com/imprensa"
URL_SAFRA_LIST = ("https://www.safra.com.br/sobre/relacoes-com-investidores/"
                  "informacoes-financeiras.htm")

# Bank keys used across the outputs, mapped to the registry row that authorises them.
BANK_OF_PANEL_CODE = {
    "C0080185": "santander", "C0080336": "btg", "C0081744": "daycoval",
    "C0080202": "volkswagen", "C0084813": "pagbank", "C0088022": "picpay",
    "C0080192": "citibank", "C0080109": "safra", "C0084844": "c6",
    "C0081737": "c6_consignado", "C0080745": "sicredi", "C0080879": "sicoob",
    "C0080367": "master", "C0084820": "mercadopago", "C0083694": "agibank",
    "C0084693": "nu", "C0082475": "xp", "C0080996": "inter",
}
BANK_KEYS = tuple(dict.fromkeys(BANK_OF_PANEL_CODE.values()))

# Banks whose own-site discovery is implemented. The rest are listed in the documents table with a
# status saying why they yielded nothing, so the gap is in the output rather than only in a report.
IR_IMPLEMENTED = ("volkswagen", "pagbank", "picpay", "safra")

MONTHS_PT = {"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
             "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11,
             "dezembro": 12}
MONTH_ABBR_PT = {"jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6, "jul": 7,
                 "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12}

# The note's own heading. Banks word it differently; all of these introduce the itemised
# administrative-expense table.
ADMIN_NOTE_RE = re.compile(
    r"(OUTRAS\s+DESPESAS\s+ADMINISTRATIVAS"
    r"|DESPESAS\s+ADMINISTRATIVAS"
    r"|DESPESAS\s+DE\s+PESSOAL\s+E\s+ADMINISTRATIVAS"
    r"|OUTRAS\s+DESPESAS\s+OPERACIONAIS\s+E\s+ADMINISTRATIVAS)")

# The lines we are after, and the accounts they name. Order matters: the bundled patterns are
# tested first so that "Propaganda, Promocoes e Publicidade" is not filed as plain advertising.
LABEL_GROUPS = [
    ("adv_promo_publ_bundled", re.compile(r"PROPAGANDA.*(PROMO|PUBLICA)|"
                                          r"PUBLICIDADE.*(PROMO|PUBLICA)|"
                                          r"(PROMO|PUBLICA).*PROPAGANDA")),
    ("adv_promo_bundled", re.compile(r"(PROPAGANDA|PUBLICIDADE).*(PROMO|MARKETING)|"
                                     r"(PROMO|MARKETING).*(PROPAGANDA|PUBLICIDADE)")),
    ("advertising", re.compile(r"PROPAGANDA|PUBLICIDADE")),
    ("marketing", re.compile(r"\bMARKETING\b")),
    ("promotions", re.compile(r"PROMO(COES|CAO)|RELACOES\s+PUBLICAS")),
    ("publications", re.compile(r"PUBLICACOES")),
]
TOTAL_RE = re.compile(r"^(TOTAL|TOTAIS|TOTAL\s+GERAL)\b")
# Any numbered note heading. A block runs from one heading to the next, so the Total row of the
# NEXT note is never added to this one's items.
# A numbered note heading ("24. Outras Despesas Administrativas") or a lettered sub-item heading
# ("d) Despesas administrativas", how Safra and several others break a note up). Both are block
# boundaries, and the sub-item matters: without it one page is a single block whose header rows are
# whatever sits at the top of the page rather than the table's own column titles.
NOTE_HEAD_RE = re.compile(r"^\(?(?:(\d{1,2})\s*[.)]|[a-zA-Z]\s*\))\s*[A-Za-zÀ-ÿ]")
# The running header and footer the filing system stamps on every page. Their page numbers and
# dates are read as amounts by any geometric parser, so a footer row ends the note block: on
# Santander's page 136 the footer "... | 30 de junho de 2025 | 27" and "PAGINA: 136 de 173" added
# four spurious items and broke the note's own arithmetic by R$2.4m.
FOOTER_RE = re.compile(
    r"^(PAGINA\b|VERSAO\s*:|DFP\s*-\s*DEMONSTRACOES|ITR\s*-\s*INFORMACOES"
    r"|DEMONSTRACOES\s+FINANCEIRAS|NOTAS\s+EXPLICATIVAS|RELATORIO\s+D"
    r"|DECLARACOES\s+DOS|BALANCO\s+PATRIMONIAL)")

# An amount as the PDFs write it, including a bare dash for a published zero.
AMOUNT_TOKEN = re.compile(r"^\(?-?R?\$?\s?[\d.]{1,20}(,\d{1,2})?\)?$")
DATE_RE = re.compile(r"\b(\d{2})[/.](\d{2})[/.](\d{4})\b")
YEAR_RE = re.compile(r"\b(20\d{2})\b")
# "01/01 a 30/06/2025": a window whose start is printed without its year.
PARTIAL_WINDOW_RE = re.compile(r"\b(\d{2})[/.](\d{2})\s*(?:a|A|ate|até|AT[EÉ])\s*"
                               r"(\d{2})[/.](\d{2})[/.](\d{4})\b")
# A header token that belongs to a column header: a date, a year, a day/month, or the word "a".
HEADER_TOKEN_RE = re.compile(r"^(\d{2}[/.]\d{2}([/.]\d{4})?|20\d{2}|[aA]|at[eé]|AT[EÉ])$")
IFRS_MARKER_RE = re.compile(r"\bIFRS\b|INTERNATIONAL FINANCIAL REPORTING")

# The running title the filer stamps down the side of every page names the statement set the page
# belongs to, and that is what decides scope where the note itself labels no column. Santander's
# interim delivery carries two administrative-expense notes: the one under "Demonstracoes
# Financeiras Individuais e Consolidadas" labels its columns Banco and Consolidado, and the one
# under "Demonstracoes Financeiras Intermediarias Consolidadas Condensadas" labels nothing and is a
# narrower, consolidated aggregate (a stated total of R$4.53bn against the individual note's
# R$7.47bn). Read off the page, this is evidence; assumed, it would be a fabricated scope.
SECTION_SCOPE = [
    (re.compile(r"INDIVIDUAIS\s+E\s+CONSOLIDADAS"), None),          # the columns decide
    (re.compile(r"CONGLOMERADO\s+PRUDENCIAL"), "consolidated_prudential"),
    (re.compile(r"INTERMEDIARIAS\s+CONSOLIDADAS|CONSOLIDADAS\s+CONDENSADAS"
                r"|(DEMONSTRACOES\s+)?FINANCEIRAS\s+CONSOLIDADAS|CONTABEIS\s+CONSOLIDADAS"),
     "consolidated"),
    (re.compile(r"(DEMONSTRACOES\s+)?FINANCEIRAS\s+INDIVIDUAIS|CONTABEIS\s+INDIVIDUAIS"),
     "individual"),
]
SECTION_TITLE_RE = re.compile(
    r"DEMONSTRACOES\s+(FINANCEIRAS|CONTABEIS)[A-Z\s]{0,60}|CONGLOMERADO\s+PRUDENCIAL")

# What each published label is in the central bank's chart of accounts, so that a later reader
# cannot mistake a bundled line for advertising alone. The 2025 COSIF calibration is what fixes
# this: Santander's "Propaganda, Promocoes e Publicidade" matched the SUM of the three accounts to
# 1.0013, and the advertising account alone only to 1.53.
COSIF_CONCEPT = {
    "advertising": "cosif_adv_8174500005",
    "promotions": "cosif_promo_8174200002",
    "publications": "cosif_publ_8174800006",
    "adv_promo_bundled": "cosif_adv_plus_promo",
    "adv_promo_publ_bundled": "cosif_all3_adv_promo_publ",
    "marketing": "no_cosif_counterpart_broader_than_all3",
    "admin_expense_total": "cosif_admin_81700006",
}
# Which COSIF entity a note scope is comparable with: the individual statements are the institution,
# the consolidated ones the prudential conglomerate. Also fixed by the 2025 calibration, where the
# consolidated figure matched the conglomerate at 0.997 and the institution only at 1.34.
COSIF_LEVEL_OF_SCOPE = {"individual": "institution", "consolidated": "conglomerate",
                        "consolidated_prudential": "conglomerate"}

MIN_TOTAL_MATCH = 0.70          # floor for the note's own arithmetic, per bank
SCALE_THOUSANDS = 1_000.0
# The accounts this route is after, as `category_group` names them.
ADV_CATEGORIES = ("advertising", "adv_promo_publ_bundled", "adv_promo_bundled", "marketing",
                  "promotions", "publications")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def fold(text: str) -> str:
    """Upper-case ASCII fold: accents removed, whitespace collapsed."""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip().upper()


def brl(token: str) -> float | None:
    """Brazilian-format amount -> float. A bare dash is a published zero; None when the token
    carries no amount at all."""
    s = str(token).replace("\xa0", " ").strip()
    if not s:
        return None
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = re.sub(r"[R$\s()]", "", s).lstrip("-")
    if s in ("", "-"):
        return 0.0 if "-" in token else None
    if not re.fullmatch(r"\d{1,3}(\.\d{3})*(,\d{1,2})?|\d+(,\d{1,2})?", s):
        return None
    v = float(s.replace(".", "").replace(",", "."))
    return -v if neg else v


def safe_name(url: str) -> str:
    """Short, ASCII, unique local file name for a URL."""
    parsed = urllib.parse.urlparse(url)
    base = urllib.parse.unquote(parsed.path.rstrip("/").split("/")[-1]) or "index"
    if parsed.query:
        q = urllib.parse.parse_qs(parsed.query)
        seq = q.get("NumeroSequencialDocumento", [""])[0]
        if seq:
            base = f"doc{seq}.zip"
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
def _replace(src: Path, dst: Path, attempts: int = 6) -> None:
    """`os.replace`, retried for a moment: on Windows a sync client or a virus scanner holding the
    target open fails the rename with PermissionError, and the cache and outputs live in
    OneDrive."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.25 * (i + 1))


def write_atomic(path: Path, data: bytes, suffix: str = ".part") -> None:
    """Write `data` to `path` through a temporary sibling that is renamed into place.

    A run killed mid-write leaves the temporary file and the previous `path` untouched, never a
    truncated `path` that a later run would take for the real thing.
    """
    tmp = path.with_name(path.name + suffix)
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    _replace(tmp, path)


def _is_2xx(status) -> bool:
    try:
        return 200 <= int(status) < 300
    except (TypeError, ValueError):
        return False


class UnapprovedRedirect(RuntimeError):
    """A server redirected to a host that `utils/note_sources.py` does not approve. Raised before
    anything is sent to that host. Not an OSError on purpose: `fetch` retries network errors and
    urllib wraps some OSErrors, and neither may mistake a refusal for a transient failure."""


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """urllib's redirect handling with one change: a hop is followed only onto an approved host,
    and every hop followed is recorded."""

    def __init__(self, hops: list):
        super().__init__()
        self.hops = hops

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib calls this with the target already made absolute and before requesting it, so a
        # refusal here means the unapproved host is never contacted.
        try:
            allowed_host(newurl)
        except PermissionError:
            host = (urllib.parse.urlparse(newurl).hostname or "").lower()
            raise UnapprovedRedirect(
                f"{req.full_url} redirected (HTTP {code}) to {newurl}; {host or newurl!r} is not "
                f"an approved host in utils/note_sources.py, so it was not requested. Approve the "
                f"host there, or drop the URL.") from None
        self.hops.append([int(code), newurl])
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass
class Fetched:
    """One request's outcome: the last HTTP status seen (None when no response came back), the
    body of a 2xx answer, the URL that finally answered, and every redirect hop followed."""
    status: int | None
    body: bytes | None
    final_url: str
    redirects: list = field(default_factory=list)


def fetch(url: str, timeout: int = 180, retries: int = 3) -> Fetched:
    """GET `url`, following redirects only onto approved hosts. The one network path here.

    The request is the one `utils.disclosure_common.http_get` sends (its User-Agent and per-host
    throttle, gzip, retries with backoff, 403/404/410 taken as final), with the difference that is
    the reason this function exists: `http_get` uses urllib's default redirect handling, which
    follows a 30x to any host. Here each hop's target is checked with `allowed_host` before it is
    requested, and an unapproved one raises `UnapprovedRedirect`, which is never retried.
    """
    allowed_host(url)
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip", "Accept": "*/*"}
    status: int | None = None
    hops: list = []
    for attempt in range(1, retries + 1):
        hops = []
        opener = urllib.request.build_opener(_CheckedRedirects(hops))
        _throttle(url)
        try:
            with opener.open(urllib.request.Request(url, headers=headers),
                             timeout=timeout) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding", "") == "gzip":
                    body = gzip.decompress(body)
                return Fetched(int(resp.status), body, resp.geturl(), hops)
        except UnapprovedRedirect:
            raise
        except urllib.error.HTTPError as exc:
            status = exc.code
            if exc.code in (403, 404, 410):
                return Fetched(exc.code, None, exc.filename or url, hops)
            log.warning("HTTP %s on %s (attempt %d/%d)", exc.code, url, attempt, retries)
        except Exception as exc:                        # noqa: BLE001 - retried, then reported
            log.warning("GET failed %s on %s (attempt %d/%d)", type(exc).__name__, url, attempt,
                        retries)
        time.sleep(min(8.0, 0.8 * 2 ** attempt))
    return Fetched(status, None, url, hops)


def listing_problem(body: bytes, reference: bytes | None = None) -> str | None:
    """Why a 2xx HTML answer is still not the bank's listing page, or None.

    A firewall challenge, a maintenance notice or a soft error page arrives as HTTP 200 HTML just
    as the listing does; cached in its place it would silently empty the bank's discovery. So a
    listing must link at least one PDF, and a refreshed one at least `LISTING_MIN_SHARE` as many as
    the cached copy it would replace (`reference`). PicPay's press page links none, so it never
    counts as a listing and is requested again on every run; it lists no statement either way.
    """
    n = len(set(PDF_HREF_RE.findall(body.decode("utf-8", "replace"))))
    if n == 0:
        return ("listing page links no PDF: a challenge, maintenance or error page served as "
                "HTML, not the listing")
    if reference is not None:
        n_ref = len(set(PDF_HREF_RE.findall(reference.decode("utf-8", "replace"))))
        if n < LISTING_MIN_SHARE * n_ref:
            return f"listing page links {n} PDFs where the cached copy links {n_ref}"
    return None


ContentCheck = Callable[[bytes, "bytes | None"], "str | None"]


class FileLock:
    """A lock file naming the run that holds it: one per bank cache, one per output folder.

    It is created with O_EXCL, so of two runs starting together exactly one creates it. A lock is
    STALE, and taken over, when its holder is gone: the process table no longer has its pid, or
    has that pid for a process started after the lock (a reused pid). Without psutil no holder is
    ever declared gone. A lock file that is empty or unreadable is stale once it is older than
    `GRACE` seconds: its creator writes it right after creating it, so an empty file older than
    that was left by a run killed in between, while a younger one may be a creator mid-write.

    A takeover must not let two runs that find the same stale lock both hold it, which removing
    the file and creating it again did: the second run's removal deleted the first run's new
    lock. So each writes its claim to its own temporary file, checks that the lock still holds the
    content it judged stale, renames its claim over it, waits `SETTLE` seconds and reads it back.
    Only the run that finds its own token there holds the lock; the other sees a live holder. The
    two renames would have to land more than `SETTLE` apart after passing the same check, which is
    a window of microseconds, for both to be fooled.
    """

    GRACE = 5.0
    SETTLE = 0.3

    def __init__(self, path: Path, what: str):
        self.path = Path(path)
        self.what = what
        self.token = uuid.uuid4().hex
        self.held = False
        self.holder: dict = {}

    def _payload(self) -> bytes:
        return json.dumps({"pid": os.getpid(), "token": self.token, "started_epoch": time.time(),
                           "started_at": now_iso(), "argv": sys.argv}).encode("utf-8")

    def _read(self) -> tuple[bool, bytes | None]:
        """(whether the lock file exists, its bytes); None bytes when it exists but cannot be
        read, which on Windows is another process holding it open."""
        try:
            return True, self.path.read_bytes()
        except FileNotFoundError:
            return False, None
        except PermissionError:
            return True, None

    def _is_mine(self, raw: bytes | None) -> bool:
        try:
            return json.loads(raw.decode("utf-8")).get("token") == self.token
        except (AttributeError, UnicodeDecodeError, ValueError):
            return False

    def _judge(self, raw: bytes | None) -> tuple[dict, str | None]:
        """(what the lock says about its holder, why that holder is gone or None if it is not)."""
        try:
            holder = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, ValueError):
            holder = None
        if not isinstance(holder, dict):
            try:
                age = time.time() - self.path.stat().st_mtime
            except OSError:
                return {}, None
            if age > self.GRACE:
                return {}, (f"the lock file is empty or unreadable and {age:.0f}s old: a run "
                            f"killed between creating and writing it")
            return {}, None
        pid = holder.get("pid")
        if pid == os.getpid():
            return holder, None          # this process holds it already: a second holder is a bug
        try:
            import psutil
        except ImportError:
            return holder, None
        try:
            if not psutil.pid_exists(int(pid)):
                return holder, f"pid {pid} is not running"
            if psutil.Process(int(pid)).create_time() > float(holder["started_epoch"]) + 1.0:
                return holder, f"pid {pid} was reused by a process started after the lock"
        except (psutil.Error, TypeError, ValueError, KeyError):
            pass
        return holder, None

    def _take_over(self, stale: bytes | None) -> bool:
        claim = self.path.with_name(f"{self.path.name}.{self.token}.tmp")
        try:
            claim.write_bytes(self._payload())
            exists, now = self._read()
            if (exists, now) != (True, stale):
                # Changed since it was judged: another run got there first, and holds it now.
                self.holder = self._judge(now)[0] if exists else {}
                return False
            os.replace(claim, self.path)
        except PermissionError:
            return False                 # held open by another process: not ours to take
        finally:
            with contextlib.suppress(OSError):
                claim.unlink(missing_ok=True)
        time.sleep(self.SETTLE)
        exists, raw = self._read()
        if self._is_mine(raw):
            return True
        self.holder = self._judge(raw)[0] if exists else {}
        return False

    def _try(self) -> bool:
        for _ in range(3):               # a holder may release between our create and our read
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                             | getattr(os, "O_BINARY", 0))
            except (FileExistsError, PermissionError):
                # PermissionError: on Windows a file still being deleted cannot be created again.
                pass
            else:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(self._payload())
                return True
            exists, raw = self._read()
            if not exists:
                continue
            self.holder, stale = self._judge(raw)
            if not stale:
                return False
            log.warning("taking over the stale lock %s: %s", self.path, stale)
            return self._take_over(raw)
        return False

    def acquire(self, wait: float = 0.0) -> None:
        """Hold the lock, waiting up to `wait` seconds for a live holder; SystemExit naming the
        holder otherwise."""
        deadline = time.monotonic() + wait
        while not self._try():
            if time.monotonic() >= deadline:
                h = self.holder
                raise SystemExit(
                    f"the {self.what} is locked by {self.path} (pid {h.get('pid')}, started "
                    f"{h.get('started_at')}, {' '.join(map(str, h.get('argv') or []))}). Another "
                    f"run is using it; wait for it to finish. If no run is active, a killed run "
                    f"left the lock behind: delete that file and rerun.")
            time.sleep(0.5)
        self.held = True
        atexit.register(self.release)

    def release(self) -> None:
        """Remove the lock if this run still holds it. Never raises: it runs in `finally` blocks
        and at interpreter exit, where an exception would hide the one that matters."""
        if not self.held:
            return
        self.held = False
        exists, raw = self._read()
        if exists and raw is not None and not self._is_mine(raw):
            log.warning("%s is no longer this run's lock; left in place", self.path)
            return
        for i in range(6):
            try:
                os.remove(self.path)
                return
            except FileNotFoundError:
                return
            except OSError as exc:       # PermissionError: another process has it open
                if i == 5:
                    log.warning("could not remove the lock %s (%s). The next run will find its "
                                "holder gone and take it over.", self.path, exc)
                    return
                time.sleep(0.25 * (i + 1))


class Cache:
    """One manifest per bank under RAW_DIR/<bank>/manifest.json.

    Every URL requested is recorded with its HTTP status, the URL that finally answered and the
    redirect hops, the local file, its size, its SHA-256 and the kind of body received (zip, pdf,
    xls, html, service_error, other). A URL is served from disk only while its entry is GOOD: a 2xx
    status, a file on disk of exactly the recorded size, and a body of the kind the request expects
    (a statement is a zip or a PDF, a listing is HTML), a PDF ending in "%%EOF", and for a listing
    a page that links PDFs (`listing_problem`). Anything else is a miss and is requested again: no
    response, a 404, the regulator's "busy" text served with a 200, a file cut short by a killed
    run, a firewall page served as the listing. A transient failure is therefore never frozen into
    the cache, while a good document is still requested once only, so a rerun works from disk.

    A failed fetch never replaces a good entry, so --refresh can only improve the cache: when the
    refresh of a good document or listing fails, the good file and entry stay and the failure is
    noted beside them. A failed body is kept only under `<name>.failed`, never under the name of
    the document, so it cannot overwrite a good file on disk. Files and the manifest are written to
    a temporary name and renamed into place. A lock file per bank (`FileLock`) stops two runs from
    overwriting each other's manifest entries; the second run stops at once and says which run
    holds the lock.
    """

    LOCK_NAME = "cache.lock"

    def __init__(self, bank: str, refresh: bool):
        self.dir = RAW_DIR / bank
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "manifest.json"
        self.refresh = refresh
        self.entries: dict[str, dict] = {}
        # URLs requested by this process. Each is requested at most once per run, whether it
        # succeeded or not: a failure is retried by the next run, not by the next call.
        self._fetched: set[str] = set()
        self.lock = FileLock(self.dir / self.LOCK_NAME, f"{bank} cache")
        self.lock_path = self.lock.path
        self.lock.acquire()
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self.close()
                raise SystemExit(
                    f"{self.path} is not valid JSON ({exc}). It was probably cut short by a run "
                    f"killed while writing it; restore it from the sync client's version history, "
                    f"or delete it to have every document requested again.") from exc

    @property
    def _locked(self) -> bool:
        return self.lock.held

    def close(self) -> None:
        """Release the lock. Idempotent; also runs at interpreter exit."""
        self.lock.release()

    # -- the manifest -------------------------------------------------------------------------
    def save(self) -> None:
        write_atomic(self.path, json.dumps(self.entries, indent=1, ensure_ascii=False)
                     .encode("utf-8"), suffix=".tmp")

    @staticmethod
    def body_kind(body: bytes) -> str:
        head = body[:8]
        if head.startswith(b"%PDF"):
            return "pdf"
        if head.startswith(b"PK"):
            return "zip"
        if head.startswith(b"\xd0\xcf\x11\xe0"):
            return "xls"
        if any(body[:400].lstrip().startswith(m) for m in SERVICE_ERROR_MARKERS):
            return "service_error"
        low = body[:HEAD_BYTES].lower()
        if b"<html" in low or b"<!doctype html" in low or b"<head" in low:
            return "html"
        return "other"

    @staticmethod
    def _body_problem(head: bytes, tail: bytes, kind: str, expect: tuple[str, ...],
                      zip_source) -> str | None:
        if kind == "service_error":
            return ("regulator service error: "
                    + head[:160].decode("utf-8", "replace").strip())
        if kind not in expect:
            return f"body is {kind}, expected {' or '.join(expect)}"
        if kind == "zip" and not zipfile.is_zipfile(zip_source):
            return "zip archive is unreadable (truncated or corrupt)"
        if kind == "pdf" and b"%%EOF" not in tail.rstrip(PDF_PADDING)[-PDF_EOF_WINDOW:]:
            return f"pdf has no %%EOF in its last {PDF_EOF_WINDOW} bytes: cut short"
        return None

    def problem(self, entry: dict | None, expect: tuple[str, ...] = DOC_KINDS,
                content_check: ContentCheck | None = None) -> str | None:
        """Why `entry` cannot be served from disk, or None when it is good.

        Everything positive is re-checked against the file itself; only a recorded failure is
        taken on trust, because that can only send the URL back to the server.
        """
        if not entry:
            return "not cached"
        if entry.get("failure"):
            return str(entry["failure"])
        if entry.get("status") is None:
            return "no HTTP response"
        if not _is_2xx(entry["status"]):
            return f"HTTP {entry['status']}"
        fname = entry.get("file")
        if not fname:
            return "no body"
        path = self.dir / fname
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return f"file {fname} is missing"
        if size != entry.get("bytes"):
            return f"file {fname} is {size} bytes, the manifest says {entry.get('bytes')}"
        with open(path, "rb") as fh:
            head = fh.read(HEAD_BYTES)
            fh.seek(max(0, size - TAIL_BYTES))
            tail = fh.read()
        why = self._body_problem(head, tail, self.body_kind(head), expect, path)
        if why is None and content_check is not None:
            why = content_check(path.read_bytes(), None)
        return why

    def get(self, url: str, name: str | None = None, expect: tuple[str, ...] = DOC_KINDS,
            content_check: ContentCheck | None = None) -> dict:
        """The entry for `url`, from disk while it is good, otherwise requested through `fetch`.

        `content_check(body, reference)` is asked about a body that passed the kind checks, with
        `reference` the good cached body it would replace (None when there is none); a reason
        returned makes the body a failure like any other.
        """
        allowed_host(url)                      # authorisation, before anything is sent
        old = self.entries.get(url)
        if url in self._fetched:
            return old                         # one attempt per run; fetch already retries
        old_problem = self.problem(old, expect, content_check)
        if old_problem is None and not self.refresh:
            return old
        got = fetch(quote_url(url), timeout=180, retries=3)
        self._fetched.add(url)
        status, body = got.status, got.body
        kind = self.body_kind(body) if body else None
        entry = {"url": url, "status": status, "file": None, "bytes": len(body) if body else 0,
                 "sha256": hashlib.sha256(body).hexdigest() if body else None, "kind": kind,
                 "final_url": got.final_url, "redirects": got.redirects,
                 "retrieved_at": now_iso()}
        if status is None:
            failure = "no HTTP response"
        elif not _is_2xx(status):
            failure = f"HTTP {status}"
        elif not body:
            failure = "empty body"
        else:
            failure = self._body_problem(body[:HEAD_BYTES], body[-TAIL_BYTES:], kind, expect,
                                         io.BytesIO(body))
            if failure is None and content_check is not None:
                reference = self.file(old).read_bytes() if old_problem is None else None
                failure = content_check(body, reference)

        fname = name or safe_name(url)
        failed_name = fname + FAILED_SUFFIX
        if failure and body:
            # Kept for inspection, and only ever beside the document: its own name holds nothing
            # but a body that passed, so a failure cannot overwrite a good document on disk.
            write_atomic(self.dir / failed_name, body)
        if failure and old_problem is None:
            # A refresh of a good entry failed. The good file and entry stay; the failure is noted
            # beside them so the manifest shows the refresh was attempted.
            old["last_refresh_failure"] = {"at": entry["retrieved_at"], "status": status,
                                           "kind": kind, "reason": failure,
                                           "final_url": got.final_url,
                                           "failed_file": failed_name if body else None}
            self.save()
            log.warning("  GET %-4s %-13s refresh failed (%s); the cached copy is kept: %s",
                        status, kind, failure, url)
            return old
        if failure:
            entry["failure"] = failure
            entry["failed_file"] = failed_name if body else None
        else:
            write_atomic(self.dir / fname, body)
            entry["file"] = fname
            with contextlib.suppress(OSError):     # an earlier failed body, now superseded
                (self.dir / failed_name).unlink(missing_ok=True)
        self.entries[url] = entry
        self.save()
        log.info("  GET %-4s %-13s %9d  %s%s%s", status, kind, entry["bytes"], url,
                 f"  -> {got.final_url}" if got.redirects else "",
                 f"  FAILED: {failure}" if failure else "")
        return entry

    def invalidate(self, url: str, reason: str) -> None:
        """Mark a cached entry as failed, so the next run requests it again. Used when a body that
        passed the checks here turns out to be unreadable when the document is opened."""
        entry = self.entries.get(url)
        if entry is not None:
            entry["failure"] = reason
            self.save()

    def file(self, entry: dict) -> Path | None:
        return self.dir / entry["file"] if entry.get("file") else None


_OPEN_CACHES: dict[Path, Cache] = {}


def open_cache(bank: str, refresh: bool) -> Cache:
    """The one Cache per bank this process uses. Discovery and parsing share it, because the lock
    file makes a second Cache on the same folder a conflicting run."""
    key = RAW_DIR / bank
    cache = _OPEN_CACHES.get(key)
    if cache is None or not cache._locked:
        cache = _OPEN_CACHES[key] = Cache(bank, refresh)
    return cache


def close_caches() -> None:
    """Release every lock this process holds."""
    for cache in _OPEN_CACHES.values():
        cache.close()
    _OPEN_CACHES.clear()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
@dataclass
class Doc:
    """One statement document to read."""
    bank_key: str
    url: str
    route: str                       # cvm_rad | ir_site
    doc_kind: str                    # DFP | ITR | statement
    ref_date: str | None = None      # the reference date the filer gives the document
    ref_date_method: str | None = None   # where ref_date came from: cvm_register, a filename_*
                                         # rule, or document_text
    version: int | None = None
    filed_at: str | None = None      # delivery date; orders restatements
    entity: str = "bank"             # the legal entity the document is about
    label: str = ""
    scope_hint: str | None = None    # scope from the document's own title, where its note labels
                                     # no columns; recorded as scope_source = document_title
    member: str | None = None        # PDF member inside a ZIP delivery
    entry: dict = field(default_factory=dict)
    note: str = ""


def html_text(cache: Cache, url: str, encoding: str = "utf-8") -> str | None:
    """A listing page's text, or None when neither the cache nor the server has a good one. A good
    listing is HTML that links PDFs (`listing_problem`), so a 200 challenge page is a failure and
    never replaces a good cached listing."""
    entry = cache.get(url, name="listing_" + safe_name(url), expect=LISTING_KINDS,
                      content_check=listing_problem)
    problem = cache.problem(entry, LISTING_KINDS, listing_problem)
    if problem:
        log.warning("listing %s did not return HTML (status %s, kind %s): %s", url,
                    entry.get("status"), entry.get("kind"), problem)
        return None
    return cache.file(entry).read_bytes().decode(encoding, "replace")


def cvm_register(ref_months: tuple[int, ...], from_year: int, to_year: int) -> pd.DataFrame:
    """The regulator's own document register, read from the cached yearly DFP/ITR zips.

    One row per delivered document, with `LINK_DOC` pointing at the document system. Restatements
    are kept: `period_table` and `_lines` carry every version and the last filed one wins.
    """
    codes = {t["cvm_code"]: BANK_OF_PANEL_CODE[t["panel_code"]]
             for t in targets("cvm_rad") if t.get("cvm_code")}
    if not CVM_REGISTER_DIR.is_dir():
        raise SystemExit(f"no CVM register cache at {CVM_REGISTER_DIR}; run "
                         f"scrape_cvm_advertising.py --refresh first")
    frames = []
    for doc in ("DFP", "ITR"):
        for year in range(from_year, to_year + 1):
            zpath = CVM_REGISTER_DIR / f"{doc}_{year}.zip"
            if not zpath.exists():
                continue
            with zipfile.ZipFile(zpath) as z:
                member = f"{doc.lower()}_cia_aberta_{year}.csv"
                if member not in z.namelist():
                    log.warning("%s has no register member %s", zpath.name, member)
                    continue
                df = pd.read_csv(io.BytesIO(z.read(member)), sep=";", encoding="latin-1")
            df["CD_CVM"] = df["CD_CVM"].astype(str).str.lstrip("0")
            df = df[df["CD_CVM"].isin(codes)].copy()
            if df.empty:
                continue
            df["bank_key"] = df["CD_CVM"].map(codes)
            df["zip_year"] = year
            frames.append(df)
    if not frames:
        raise SystemExit("the CVM register cache holds no rows for the cvm_rad targets")
    reg = pd.concat(frames, ignore_index=True)
    reg["ref_month"] = pd.to_datetime(reg["DT_REFER"]).dt.month
    reg = reg[reg["ref_month"].isin(ref_months)]
    return reg.sort_values(["bank_key", "DT_REFER", "VERSAO"]).reset_index(drop=True)


def discover_cvm_rad(cache_of, ref_months, from_year, to_year, versions: str) -> list[Doc]:
    """Documents for the regulator's filers. `versions='last'` takes only the highest version of
    each reference date, which is the value that stands; `'all'` takes every delivery so a
    restatement can be measured."""
    reg = cvm_register(ref_months, from_year, to_year)
    if versions == "last":
        reg = (reg.sort_values(["bank_key", "CATEG_DOC", "DT_REFER", "VERSAO"])
                  .groupby(["bank_key", "CATEG_DOC", "DT_REFER"], as_index=False).tail(1))
    docs = []
    for r in reg.to_dict("records"):
        docs.append(Doc(bank_key=r["bank_key"], url=RAD_DOC.format(seq=int(r["ID_DOC"])),
                        route="cvm_rad", doc_kind=r["CATEG_DOC"], ref_date=str(r["DT_REFER"]),
                        ref_date_method="cvm_register",
                        version=int(r["VERSAO"]), filed_at=str(r["DT_RECEB"]),
                        entity=str(r["DENOM_CIA"]),
                        label=f"{r['CATEG_DOC']} {r['DT_REFER']} v{r['VERSAO']}"))
    return docs


MONTH_END = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30,
             12: 31}


# The four dates a Brazilian statement is drawn up at. A full date in a file name that is none of
# them is the day the statements were printed or approved, never the reference date:
# "31.12.14BRGAAP_BANCO VOLKS - BALANCO - VALOR 27.03.2015" is the 2014 annual statement as the
# newspaper Valor printed it on 27 March 2015, and was dated 2015-03-27 while the first match in
# the name won.
BALANCE_SHEET_DAYS = {(3, 31), (6, 30), (9, 30), (12, 31)}
# Digit lookarounds rather than \b throughout: the banks glue dates to letters and underscores
# ("30.06.17BRGAAP_29.08.2017", "Semestrais_2025", "Jun2025_V2"), and \b sees no boundary between a
# digit and a letter or an underscore, which left 17 of Volkswagen's 47 statements undated.
FULL_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[.\-_](\d{1,2})[.\-_](\d{4}|\d{2})(?!\d)")  # 30-6-12
COMPACT_DATE_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})((?:19|20)\d{2})(?!\d)")          # 30062020
MONTH_YEAR_RE = re.compile(r"(?<!\d)(\d{2})[.\-_]?((?:19|20)\d{2})(?!\d)")       # 12.2025, 062026
MONTH_NAME_RE = re.compile(
    r"(JANEIRO|FEVEREIRO|MARCO|ABRIL|MAIO|JUNHO|JULHO|AGOSTO|SETEMBRO|OUTUBRO|NOVEMBRO|DEZEMBRO"
    r"|JAN|FEV|MAR|ABR|MAI|JUN|JUL|AGO|SET|OUT|NOV|DEZ)"
    r"[^0-9A-Z]{0,4}((?:19|20)\d{2}|\d{2})(?!\d)")                                # Jun2025, Set 20
FILE_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def _year4(y: str) -> int:
    v = int(y)
    return v if v >= 100 else (2000 + v if v < 70 else 1900 + v)


def _period_from_filename(name: str) -> tuple[str | None, str, str]:
    """(reference date, human label, method) from an own-site statement file name.

    Banks name the same thing a dozen ways: `31.12.2023`, `30.06.19`, `30-6-12`, `30062020`,
    `12_2021`, `062026`, `BalPromoJun2025`, `Dez2017`, `Semestrais_2025`, `Set 20`, and some prefix
    the reference date and append the day the statements were printed. Each shape is matched
    explicitly, in order of how much the name commits to:

      filename_balance_sheet_date  a full date on a balance-sheet day (31/03, 30/06, 30/09, 31/12)
      filename_month_year          a numeric month and year, `12.2025`
      filename_month_name          a month name and year, `Jun2025`
      filename_year_and_word       a year with "Semestrais" (30 June) or "Anuais" (31 December)

    A full date on any other day is a publication date: it is cut out of the name before the later
    rules run, so its month and year are not read as a reference, and the method records that one
    was ignored. A name that commits to nothing - a bare year, two different balance-sheet dates, or
    both "semestral" and "anual" - returns None: the reference date then comes from the document's
    own text (`_period_from_text`) once it is downloaded, and never from a guess here.
    """
    n = urllib.parse.unquote(name)
    reference, publication, spans = set(), set(), []
    for rx, groups in ((FULL_DATE_RE, (1, 2, 3)), (COMPACT_DATE_RE, (1, 2, 3))):
        for m in rx.finditer(n):
            d, mo, y = (m.group(g) for g in groups)
            d, mo, y = int(d), int(mo), _year4(y)
            if not (1 <= mo <= 12 and 1 <= d <= 31):
                continue
            spans.append(m.span())
            (reference if (mo, d) in BALANCE_SHEET_DAYS else publication).add(
                f"{y}-{mo:02d}-{d:02d}")
    ignored = "+publication_date_ignored" if publication else ""
    if len(reference) == 1:
        return reference.pop(), n, "filename_balance_sheet_date" + ignored
    if len(reference) > 1:
        return None, n, "filename_ambiguous"
    rest = n
    for a, b in sorted(spans, reverse=True):
        rest = rest[:a] + " " + rest[b:]
    f = fold(rest)
    for m in MONTH_YEAR_RE.finditer(rest):
        if 1 <= int(m.group(1)) <= 12:
            mm, yy = int(m.group(1)), int(m.group(2))
            return f"{yy}-{mm:02d}-{MONTH_END[mm]:02d}", n, "filename_month_year" + ignored
    m = MONTH_NAME_RE.search(f)
    if m:
        mm = MONTHS_PT.get(m.group(1).lower()) or MONTH_ABBR_PT.get(m.group(1)[:3].lower())
        if mm:
            yy = _year4(m.group(2))
            return f"{yy}-{mm:02d}-{MONTH_END[mm]:02d}", n, "filename_month_name" + ignored
    years = set(FILE_YEAR_RE.findall(rest))
    half = bool(re.search(r"SEMESTRA|1S\b|PRIMEIRO SEMESTRE", f))
    annual = bool(re.search(r"ANUA|EXERCICIO", f))
    if len(years) == 1 and half != annual:
        y = years.pop()
        return (f"{y}-06-30" if half else f"{y}-12-31"), n, "filename_year_and_word" + ignored
    return None, n, "filename_unrecognised" + ignored


# The reference date as the statements state it: "Balancos patrimoniais em 30 de junho de 2025",
# "semestre findo em 30 de junho de 2025", "exercicio findo em 31 de dezembro de 2024". Only a date
# introduced by "em" counts, because the comparative dates follow "e" ("em 30 de junho de 2025 e 31
# de dezembro de 2024"). Two kinds of mention are left out because they name EARLIER dates: "saldos
# em <date>", the opening balances of the statement of changes in equity, and the numeric form
# ("15.300 (15.300 em 30.06.2020)"), which the statements use for comparatives in parentheses. On
# Safra's prudential statements the numeric comparatives outnumber the one spelled-out heading and
# dated the June 2021 document 2020-06-30.
REF_HEADING_RE = re.compile(
    r"(?<!SALDOS )(?<!SALDO )\bEM\s+(30|31)\s+DE\s+(MARCO|JUNHO|SETEMBRO|DEZEMBRO)\s+DE\s+"
    r"((?:19|20)\d{2})\b")
REF_TEXT_PAGES = 6
# How many mentions make the text's date strong enough to OVERRULE a date the file name gives.
# One mention with no other date beside it fills an undated document (it is the heading alone,
# which is how Safra's prudential statements state it), but it is too thin to overrule a name.
OVERRIDE_MIN_MENTIONS = 2


def _text_ref_counts(pdf_bytes: bytes, pages: int = REF_TEXT_PAGES) -> dict[str, int]:
    """How often each balance-sheet date introduced by "em" appears on the first pages."""
    import fitz
    counts: dict[str, int] = {}
    with fitz.open(stream=pdf_bytes, filetype="pdf") as fz:
        text = fold(" ".join(fz[i].get_text() for i in range(min(pages, fz.page_count))))
    for m in REF_HEADING_RE.finditer(text):
        d, mo, y = int(m.group(1)), MONTHS_PT[m.group(2).lower()], int(m.group(3))
        if (mo, d) not in BALANCE_SHEET_DAYS:
            continue
        key = f"{y}-{mo:02d}-{d:02d}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _ref_from_counts(counts: dict[str, int],
                     pages: int = REF_TEXT_PAGES) -> tuple[str | None, str]:
    if not counts:
        return None, f"no 'em <balance-sheet date>' on the first {pages} pages"
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    top, n_top = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0
    evidence = f"{n_top} of {sum(counts.values())} 'em <date>' mentions on the first {pages} pages"
    if n_top == runner or (n_top < 2 and len(ranked) > 1):
        return None, f"inconclusive: {dict(ranked[:3])}"
    return top, evidence


def _period_from_text(pdf_bytes: bytes, pages: int = REF_TEXT_PAGES) -> tuple[str | None, str]:
    """(reference date, evidence) from the statement's own first pages.

    The date is the MOST FREQUENT balance-sheet date introduced by "em" on the first pages - the
    cover, the auditor's report and the balance sheet all repeat it. Not the first one: on the
    newspaper-format Volkswagen files ("... VALOR 29.08.2017", three pages of several columns) the
    first "em" date in reading order is a prior year's, while the mode is the reference date on all
    47 cached Volkswagen documents. A tie is not evidence; neither is a single mention beside other
    dates. A single mention with no other date beside it is the heading alone ("periodo findo em 30
    de junho de 2021"), which is how Safra's prudential statements state it.
    """
    return _ref_from_counts(_text_ref_counts(pdf_bytes, pages), pages)


def discover_volkswagen(cache: Cache) -> list[Doc]:
    """Banco Volkswagen publishes its half-year and annual statements as PDFs on its own domain.
    Two entities share the page: BVW (the bank) and CNVW (the consortium administrator, a separate
    institution outside the deposit panel), plus IFRS consolidations which are not the measure."""
    text = html_text(cache, URL_VWFS_LIST)
    if text is None:
        return []
    docs = []
    for m in PDF_HREF_RE.finditer(text):
        url = urllib.parse.urljoin(URL_VWFS_LIST, m.group(1))
        try:
            allowed_host(url)
        except PermissionError:
            continue
        name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        f = fold(name)
        if "IFRS" in f:                       # IFRS consolidation, not the BRGAAP individual
            continue
        if not re.search(r"DEMONSTRA|\bDF\b|BRGAAP|BVW|CNVW", f):
            continue
        entity = ("Banco Volkswagen S.A." if re.search(r"BVW|BANCO\s*VOLKS|VOLKWAGEN", f)
                  else "Consorcio Nacional Volkswagen" if "CNVW" in f or "CONSORCIO" in f
                  else "unknown")
        ref, label, ref_method = _period_from_filename(name)
        prudential = "PRUDENCIAL" in f
        docs.append(Doc(bank_key="volkswagen", url=url, route="ir_site", doc_kind="statement",
                        ref_date=ref, ref_date_method=ref_method, entity=entity, label=label,
                        scope_hint="consolidated_prudential" if prudential else "individual",
                        note="conglomerado prudencial" if prudential else ""))
    return docs


def discover_pagbank(cache: Cache) -> list[Doc]:
    """PagBank's statement list is on an approved host but every PDF it links is served from
    `acq-static-pages.pagseguro.com.br`, which the user has not approved. The documents are
    recorded here with `route='blocked_host'` and never requested, so the block is visible in the
    documents table instead of only in a report."""
    text = html_text(cache, URL_PAGBANK_LIST)
    if text is None:
        return []
    docs = []
    for m in PDF_HREF_RE.finditer(text):
        url = urllib.parse.urljoin(URL_PAGBANK_LIST, m.group(1))
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        ref, label, ref_method = _period_from_filename(name)
        try:
            allowed_host(url)
            route = "ir_site"
        except PermissionError:
            route = "blocked_host"
        docs.append(Doc(bank_key="pagbank", url=url, route=route, doc_kind="statement",
                        ref_date=ref, ref_date_method=ref_method,
                        entity="BancoSeguro S.A. / PagSeguro group", label=label,
                        note=f"host {host} is not approved" if route == "blocked_host" else ""))
    return docs


def discover_picpay(cache: Cache) -> list[Doc]:
    """PicPay's approved host is `www.picpay.com`. Its press page carries no statement link; the
    candidate paths below are probed and whatever answers with a PDF is taken."""
    docs = []
    text = html_text(cache, URL_PICPAY_LIST)
    if text:
        for m in PDF_HREF_RE.finditer(text):
            url = urllib.parse.urljoin(URL_PICPAY_LIST, m.group(1))
            try:
                allowed_host(url)
            except PermissionError:
                continue
            name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
            ref, label, ref_method = _period_from_filename(name)
            docs.append(Doc(bank_key="picpay", url=url, route="ir_site", doc_kind="statement",
                            ref_date=ref, ref_date_method=ref_method, entity="PicPay Bank",
                            label=label))
    return docs


def discover_safra(cache: Cache) -> list[Doc]:
    """Banco Safra does not file with the securities regulator but publishes its half-year audited
    statements on its own domain, back to 2015. The page also carries monthly risk and LIG reports,
    English versions and the IFRS consolidation; only the statement documents are taken, and the
    IFRS ones are left out because the measure is the local individual figure. Which column is
    which is decided by the note's own "Banco"/"Consolidado" headers, not by the file name, so a
    consolidated-titled document still yields the individual column when it prints one."""
    text = html_text(cache, URL_SAFRA_LIST)
    if text is None:
        return []
    docs, seen = [], set()
    for m in PDF_HREF_RE.finditer(text):
        url = urllib.parse.urljoin(URL_SAFRA_LIST, m.group(1))
        try:
            allowed_host(url)
        except PermissionError:
            continue
        name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        f = fold(name)
        if not re.search(r"DEMONSTRA|BALCON|RESDEMCON|DEMOSNTRA|NOTAS EXPLICATIVAS|BALANCO CON", f):
            continue
        if re.search(r"IFRS|ENGLISH|_EN\b|INGLES", f):
            continue
        ref, label, ref_method = _period_from_filename(name)
        if ref is None or int(ref[5:7]) not in (6, 12):   # the audited half-year statements
            continue
        if url in seen:
            continue
        seen.add(url)
        prudential = "PRUDEN" in f
        entity = ("Banco Safra S.A. - conglomerado prudencial" if prudential
                  else "Banco Safra S.A.")
        docs.append(Doc(bank_key="safra", url=url, route="ir_site", doc_kind="statement",
                        ref_date=ref, ref_date_method=ref_method, entity=entity, label=label,
                        scope_hint="consolidated_prudential" if prudential else "consolidated"))
    return docs


IR_DISCOVERY = {"volkswagen": discover_volkswagen, "pagbank": discover_pagbank,
                "picpay": discover_picpay, "safra": discover_safra}


def route_of(bank: str) -> str:
    """The route a bank is read through: the regulator's system for its filers, the bank's own site
    for everyone else, exactly as `collect` splits them. A --routes run has processed the banks
    whose route it names, and a merge replaces those banks' rows."""
    cvm = {BANK_OF_PANEL_CODE[t["panel_code"]] for t in targets("cvm_rad")}
    return "cvm_rad" if bank in cvm else "ir_site"


def collect(bank_keys: tuple[str, ...], refresh: bool, ref_months, from_year, to_year,
            versions: str, routes: tuple[str, ...]) -> tuple[dict[str, list[Doc]], list[dict]]:
    """Documents per bank, plus one row per bank that yielded none and why."""
    docs: dict[str, list[Doc]] = {b: [] for b in bank_keys}
    blocked: list[dict] = []

    unresolved_keys = {BANK_OF_PANEL_CODE[t["panel_code"]] for t in unresolved()}
    host_of = {BANK_OF_PANEL_CODE[t["panel_code"]]: t.get("host") for t in NOTE_TARGETS}
    share_of = {BANK_OF_PANEL_CODE[t["panel_code"]]: t["share_2024q4"] for t in NOTE_TARGETS}

    if "cvm_rad" in routes:
        cvm_banks = [b for b in bank_keys
                     if b in {BANK_OF_PANEL_CODE[t["panel_code"]] for t in targets("cvm_rad")}]
        if cvm_banks:
            for d in discover_cvm_rad(None, ref_months, from_year, to_year, versions):
                if d.bank_key in docs:
                    docs[d.bank_key].append(d)

    if "ir_site" in routes:
        for bank in bank_keys:
            if bank in {BANK_OF_PANEL_CODE[t["panel_code"]] for t in targets("cvm_rad")}:
                continue
            if bank in unresolved_keys:
                blocked.append(dict(bank_key=bank, share_2024q4=share_of[bank],
                                    status="not_authorised",
                                    notes="no host pinned in utils/note_sources.py: the registry "
                                          "lists it as unresolved, so it is not authorised"))
                continue
            if bank not in IR_DISCOVERY:
                blocked.append(dict(bank_key=bank, share_2024q4=share_of[bank],
                                    status="discovery_not_implemented",
                                    notes=f"approved host {host_of[bank]} did not yield a statement "
                                          f"listing in the probe; see the report"))
                continue
            cache = open_cache(bank, refresh)
            found = IR_DISCOVERY[bank](cache)
            docs[bank].extend(found)
            if not found:
                blocked.append(dict(bank_key=bank, share_2024q4=share_of[bank],
                                    status="no_documents_listed",
                                    notes=f"approved host {host_of[bank]} served, but listed no "
                                          f"statement PDF on it"))

    for bank, ds in docs.items():
        log.info("%-14s %3d documents", bank, len(ds))
    return docs, blocked


# ---------------------------------------------------------------------------
# Parsed records
# ---------------------------------------------------------------------------
LINE_COLUMNS = [
    "bank_key", "panel_code", "cnpj8", "entity", "doc_kind", "ref_date", "ref_date_method",
    "version", "filed_at",
    "note_number", "note_heading", "section_title", "label_as_published", "category_group",
    "cosif_concept", "is_advertising", "scope", "scope_source", "cosif_level",
    "statement_basis", "column_header", "period_rule", "window_start", "window_end", "frequency",
    "amount_raw", "scale", "amount_published", "amount_brl", "sign_convention",
    "sign_convention_source", "currency", "is_total", "total_scope", "reversal",
    "column_index", "n_columns", "page", "parse_method", "source_file", "source_url", "flag_notes",
]


@dataclass
class Parsed:
    """What one document yields: note lines, the notes' stated totals, and the windows it covers."""
    lines: list[dict] = field(default_factory=list)
    totals: list[dict] = field(default_factory=list)
    periods: set = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


def line(doc: Doc, **kw) -> dict:
    rec = {c: None for c in LINE_COLUMNS}
    rec.update(bank_key=doc.bank_key, entity=doc.entity, doc_kind=doc.doc_kind,
               ref_date=doc.ref_date, ref_date_method=doc.ref_date_method,
               version=doc.version, filed_at=doc.filed_at,
               source_url=doc.url, source_file=doc.entry.get("file"), currency="BRL",
               reversal=False, is_total=False, scale=SCALE_THOUSANDS)
    rec.update(kw)
    return rec


# ---------------------------------------------------------------------------
# PDF geometry
# ---------------------------------------------------------------------------
def page_rows(page, ytol: float = 3.0) -> list[list[dict]]:
    """Words of a page grouped into visual rows (tops within ytol), left to right."""
    words = page.extract_words(x_tolerance=1.5, y_tolerance=2, keep_blank_chars=False)
    words.sort(key=lambda w: (round(w["top"], 1), w["x0"]))
    rows: list[list[dict]] = []
    for w in words:
        if rows and abs(rows[-1][0]["top"] - w["top"]) <= ytol:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [sorted(r, key=lambda w: w["x0"]) for r in rows]


def row_amounts(row: list[dict], max_gap: float = 1.2) -> list[tuple[float, float, str, bool]]:
    """(x-centre, value, raw text, looks like a year) for every amount in a row.

    PDF producers split amounts into touching fragments ("3" + "2.557,95" with no gap is 32557.95),
    so fragments closer than max_gap points are rejoined and anything further apart is a separate
    cell. A bare dash is a published zero.

    The year flag matters because several banks head their columns with a bare "2021" and print the
    heading and the years on one row ("d) Despesas administrativas   2021   2020"). Read as an
    amount, that turns the heading into a table row, and the block then has no header at all.
    """
    toks = [w for w in row if AMOUNT_TOKEN.match(w["text"]) or w["text"] == "-"]
    clusters: list[list[dict]] = []
    for w in toks:
        if clusters and w["x0"] - clusters[-1][-1]["x1"] <= max_gap:
            clusters[-1].append(w)
        else:
            clusters.append([w])
    out = []
    for c in clusters:
        text = "".join(w["text"] for w in c)
        v = brl(text)
        if v is None:
            continue
        yearlike = bool(re.fullmatch(r"(19|20)\d{2}", text)) and 1990 <= v <= 2100
        out.append(((c[0]["x0"] + c[-1]["x1"]) / 2, v, text, yearlike))
    return out


def label_of(row: list[dict], first_amount_x: float | None) -> str:
    """The row's label: the words left of its first amount."""
    if first_amount_x is None:
        return " ".join(w["text"] for w in row)
    return " ".join(w["text"] for w in row if w["x1"] <= first_amount_x + 0.5)


def cluster_columns(xs: list[float], tol: float = 14.0) -> list[float]:
    """Column centres from the x-centres of a note's numeric cells."""
    if not xs:
        return []
    xs = sorted(xs)
    groups = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= tol:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [float(np.mean(g)) for g in groups]


def category_of(label: str) -> str | None:
    f = fold(label)
    if "RECEITA" in f:                       # a revenue line with the same words is not the expense
        return None
    for name, rx in LABEL_GROUPS:
        if rx.search(f):
            return name
    return None


# ---------------------------------------------------------------------------
# Column headers -> windows
# ---------------------------------------------------------------------------
SEM1_RE = re.compile(r"1\s*[oO°º]?\s*SEMESTRE|PRIMEIRO\s+SEMESTRE")
SEM2_RE = re.compile(r"2\s*[oO°º]?\s*SEMESTRE|SEGUNDO\s+SEMESTRE")
YEAR_WORD_RE = re.compile(r"EXERCICIO|ACUMULADO|ANO|ANUAL")
SCOPE_WORDS = {
    "BANCO": "individual", "INDIVIDUAL": "individual", "CONTROLADORA": "individual",
    "BANCO MULTIPLO": "individual", "INSTITUICAO": "individual",
    "CONSOLIDADO": "consolidated", "CONSOLIDADAS": "consolidated",
    "CONGLOMERADO": "consolidated", "COMBINADO": "consolidated",
}


def window_of(header: str, ref_date: str | None) -> tuple[str, str | None, str | None, str]:
    """(period_rule, window_start, window_end, frequency) from a column header, verbatim.

    Every shape a bank statement uses is handled explicitly and nothing is inferred from position:
    an unrecognised header returns `unparsed` with no window, and the row is flagged rather than
    guessed at.
    """
    h = fold(header)
    # "01/01 a 30/06/2025": the start is printed without its year, which the end date supplies.
    m = PARTIAL_WINDOW_RE.search(header)
    if m:
        d0, m0, d1, m1, y1 = m.groups()
        y0 = int(y1) if (int(m0), int(d0)) <= (int(m1), int(d1)) else int(y1) - 1
        start, end = f"{y0}-{m0}-{d0}", f"{y1}-{m1}-{d1}"
        return "window_stated_partial_start", start, end, _freq(start, end)
    dates = DATE_RE.findall(header)
    if len(dates) >= 2:
        a, b = dates[0], dates[1]
        start = f"{a[2]}-{a[1]}-{a[0]}"
        end = f"{b[2]}-{b[1]}-{b[0]}"
        return "window_stated", start, end, _freq(start, end)
    if len(dates) == 1:
        d = dates[0]
        end = f"{d[2]}-{d[1]}-{d[0]}"
        mm = int(d[1])
        if SEM2_RE.search(h):
            return "semester_2_stated", f"{d[2]}-07-01", end, "semester"
        if SEM1_RE.search(h):
            return "semester_1_stated", f"{d[2]}-01-01", end, "semester"
        if YEAR_WORD_RE.search(h):
            return "year_stated", f"{d[2]}-01-01", end, "year"
        # A bare as-of date in an expense note is the period accumulated from 1 January to it: that
        # is what "01/01 a <date>" spells out elsewhere in the same documents.
        return "asof_date_ytd_assumed", f"{d[2]}-01-01", end, _freq(f"{d[2]}-01-01", end)
    y = YEAR_RE.search(header)
    if y and SEM2_RE.search(h):
        return "semester_2_stated", f"{y.group(1)}-07-01", f"{y.group(1)}-12-31", "semester"
    if y and SEM1_RE.search(h):
        return "semester_1_stated", f"{y.group(1)}-01-01", f"{y.group(1)}-06-30", "semester"
    if y:
        # A bare year over an expense column is the period the statement itself covers, which the
        # reference date fixes: in a 30 June statement "2021" is the FIRST SEMESTER of 2021 and its
        # comparative "2020" is the first semester of 2020, because a half-year income statement
        # has no fourth quarter in it. Reading it as a full year would double every such figure.
        yr = y.group(1)
        if ref_date and ref_date[5:7] == "06":
            return "bare_year_resolved_by_ref_date", f"{yr}-01-01", f"{yr}-06-30", "semester"
        if ref_date and ref_date[5:7] == "09":
            return "bare_year_resolved_by_ref_date", f"{yr}-01-01", f"{yr}-09-30", "nine_months"
        if ref_date and ref_date[5:7] == "03":
            return "bare_year_resolved_by_ref_date", f"{yr}-01-01", f"{yr}-03-31", "quarter"
        return "bare_year", f"{yr}-01-01", f"{yr}-12-31", "year"
    if SEM2_RE.search(h) and ref_date:
        return "semester_2_from_ref", f"{ref_date[:4]}-07-01", f"{ref_date[:4]}-12-31", "semester"
    if SEM1_RE.search(h) and ref_date:
        return "semester_1_from_ref", f"{ref_date[:4]}-01-01", f"{ref_date[:4]}-06-30", "semester"
    if YEAR_WORD_RE.search(h) and ref_date:
        # "Acumulado" beside a "2o Semestre" column: the year to the reference date.
        return "accumulated_to_ref_date", f"{ref_date[:4]}-01-01", ref_date, \
            _freq(f"{ref_date[:4]}-01-01", ref_date)
    return "unparsed", None, None, "unknown"


def _freq(start: str, end: str) -> str:
    months = (int(end[:4]) - int(start[:4])) * 12 + int(end[5:7]) - int(start[5:7]) + 1
    return {3: "quarter", 6: "semester", 9: "nine_months", 12: "year"}.get(months, f"{months}m")


def header_columns(header_rows: list[list[dict]], centres: list[float],
                   ref_date: str | None) -> tuple[list[str], list[str]]:
    """Per column: the header text assembled in reading order, and the scope it sits under."""
    texts = [[] for _ in centres]
    scopes: list[str | None] = [None for _ in centres]
    marks: list[tuple[float, str]] = []
    for row in header_rows:
        row_marks = [((w["x0"] + w["x1"]) / 2, SCOPE_WORDS[fold(w["text"])])
                     for w in row if fold(w["text"]) in SCOPE_WORDS]
        if row_marks:
            marks.extend(row_marks)
            continue
        for w in row:
            f = fold(w["text"])
            if not (HEADER_TOKEN_RE.match(w["text"]) or SEM1_RE.search(f) or SEM2_RE.search(f)
                    or YEAR_WORD_RE.search(f) or f in ("SEMESTRE", "1O", "2O", "1", "2")):
                continue
            x = (w["x0"] + w["x1"]) / 2
            j = int(np.argmin([abs(x - c) for c in centres])) if centres else None
            if j is not None and abs(x - centres[j]) <= 45:
                texts[j].append(w["text"])

    # A scope word is printed over the middle of the columns it covers ("Banco" over this year and
    # last year, "Consolidado" over the same two), so each column takes the scope word nearest it.
    if marks:
        for j, c in enumerate(centres):
            scopes[j] = min(marks, key=lambda m: abs(m[0] - c))[1]
    # Columns sharing one date row ("01/01 a" printed once over several columns) inherit the text
    # of the nearest column that has any, keeping the shape visible in `column_header`.
    joined = [" ".join(t) for t in texts]
    for j, t in enumerate(joined):
        if t:
            continue
        for k in range(len(joined)):
            if joined[k] and abs(centres[k] - centres[j]) < 200:
                joined[j] = joined[k]
                break
    return joined, [s or "unknown" for s in scopes]


# ---------------------------------------------------------------------------
# The note parser
# ---------------------------------------------------------------------------
def note_heading(row_text: str, has_amounts: bool) -> tuple[str | None, str | None, bool]:
    """(note number, heading, whether it is an administrative-expense note) for a heading row.

    Every numbered heading is a boundary, whatever it is about: a block that ran to the page end
    would otherwise absorb the next note's own Total row and its arithmetic check would fail.
    """
    txt = row_text.strip()
    f = fold(txt)
    if has_amounts or len(f) > 120:           # a table row, or a sentence mentioning the note
        return None, None, False
    m = NOTE_HEAD_RE.match(txt)
    is_admin = bool(ADMIN_NOTE_RE.search(f))
    if not m and not is_admin:
        return None, None, False
    return (m.group(1) if m else None), txt, is_admin


def parse_pdf_notes(doc: Doc, pdf_bytes: bytes) -> Parsed:
    """Every administrative-expense note in one statement PDF."""
    import fitz
    import pdfplumber
    out = Parsed()

    # A statement PDF runs to a few hundred pages and the word-geometry pass costs about a second a
    # page, so the pages are selected first with a plain text read: only those that actually name
    # one of the accounts get the geometry pass. On Santander's 174-page interim report this is two
    # pages rather than 174.
    with fitz.open(stream=pdf_bytes, filetype="pdf") as fz:
        n_pages = fz.page_count
        text_pages, candidates = 0, []
        for i in range(n_pages):
            t = fz[i].get_text()
            if t.strip():
                text_pages += 1
            if re.search(r"PROPAGANDA|PUBLICIDADE|MARKETING|PROMOCO", fold(t)):
                candidates.append(i)
    if text_pages == 0:
        out.notes.append("unreadable_no_text_layer")
        out.notes.append(f"pages={n_pages}")
        return out
    if text_pages < n_pages:
        out.notes.append(f"pages_without_text={n_pages - text_pages}")

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pno in candidates:
            page = pdf.pages[pno]
            try:
                rows = page_rows(page)
            except Exception as exc:                        # a damaged page, recorded not raised
                out.notes.append(f"page {pno}: {type(exc).__name__}")
                continue
            _parse_page(doc, out, pno, rows, page)
    out.notes.append(f"pages={n_pages}; candidate_pages={len(candidates)}")
    return out


def _scale_of(rows: list[list[dict]]) -> tuple[float, str]:
    """The note's scale, from the page's own wording."""
    flat = fold(" ".join(w["text"] for r in rows for w in r))
    if re.search(r"MILHARES|MIL REAIS|R\$ MIL\b", flat):
        return SCALE_THOUSANDS, "milhares"
    if re.search(r"MILHOES|MILHOES DE REAIS", flat):
        return 1_000_000.0, "milhoes"
    if re.search(r"EM REAIS\b|UNIDADES DE REAL", flat):
        return 1.0, "reais"
    return SCALE_THOUSANDS, "milhares_default"


def _normalise_signs(recs: list[dict], totals: list[dict]) -> None:
    """Make every amount a POSITIVE expense, column by column, and record what was done.

    Banks differ, and the same bank differs between documents: Santander prints administrative
    expenses positive, Banco Safra and Banco Volkswagen print them negative. Left as filed, the
    increments in `period_table` mix the two conventions and produce nonsense - Volkswagen's second
    half of 2025 came out as R$8.01m, exactly twice its first half, because a negative half-year was
    subtracted from a positive year.

    The convention is read off the note's OWN stated total for that column, not assumed: a negative
    administrative-expense total means the column prints expenses as negatives, so the column is
    flipped. Where a column has no stated total, the majority sign of its own items decides. A value
    whose sign disagrees with its column after the flip is a genuine reversal or recovery and keeps
    its sign, with `reversal` set.
    """
    if not recs:
        return
    total_sign = {t["scope_id"]: (-1.0 if t["stated"] < 0 else 1.0) for t in totals}
    by_col: dict[str, list[dict]] = {}
    for r in recs:
        by_col.setdefault(r["total_scope"], []).append(r)
    for col, group in by_col.items():
        if col in total_sign:
            flip, src = total_sign[col], "note_total"
        else:
            neg = sum(1 for r in group if r["amount_brl"] < 0)
            flip = -1.0 if neg > len(group) / 2 else 1.0
            src = "item_majority"
        for r in group:
            r["amount_published"] = r["amount_brl"]
            r["amount_brl"] = r["amount_brl"] * flip
            r["sign_convention"] = ("negative_expenses" if flip < 0 else "positive_expenses")
            r["sign_convention_source"] = src
            r["reversal"] = bool(r["amount_brl"] < 0)
    for t in totals:
        if total_sign.get(t["scope_id"], 1.0) < 0:
            t["stated_published"] = t["stated"]
            t["stated"] = -t["stated"]
        else:
            t["stated_published"] = t["stated"]


def _parse_page(doc: Doc, out: Parsed, pno: int, rows: list[list[dict]], page) -> None:
    """One page: find each note block, its columns, its item rows and its stated total."""
    scale, scale_src = _scale_of(rows)

    # The IFRS section of the same delivery repeats the administrative-expense note on a different
    # accounting basis. It is marked from the page's own wording, not dropped.
    page_text = fold(" ".join(w["text"] for r in rows for w in r))
    # `as_published` claims nothing: the basis is only asserted where the page says so. Santander's
    # deliveries carry a second, narrower administrative-expense note whose columns are unlabelled
    # and whose line reads "Publicidade" rather than "Propaganda, Promocoes e Publicidade"; that
    # note is separated from the measure by its unlabelled scope, not by a guess about its basis.
    basis = "ifrs" if IFRS_MARKER_RE.search(page_text) else "as_published"

    # The page's running title: the statement set this page belongs to, and the scope it implies
    # where the note labels no column.
    m_title = SECTION_TITLE_RE.search(page_text)
    section_title = m_title.group(0).strip() if m_title else None
    section_scope = None
    for rx, sc in SECTION_SCOPE:
        if rx.search(page_text):
            section_scope = sc
            break

    # Where the notes start on this page. A page may carry the tail of one note and the head of the
    # next, so a block runs from its heading to the next heading or the page end.
    # A year in a heading row is a column title, not a figure, so item detection uses only the
    # amounts that are not year-like.
    figures = [[a for a in row_amounts(r) if not a[3]] for r in rows]

    heads = []
    for i, r in enumerate(rows):
        num, head, is_admin = note_heading(" ".join(w["text"] for w in r), bool(figures[i]))
        if head:
            heads.append((i, num, head, is_admin))
    if not heads or heads[0][0] > 0:
        heads.insert(0, (0, None, "(note heading not on this page)", True))

    for bi, (start, num, head, is_admin) in enumerate(heads):
        end = heads[bi + 1][0] if bi + 1 < len(heads) else len(rows)
        block = rows[start:end]
        amounts_per_row = figures[start:end]
        if not is_admin:
            continue
        if not any(category_of(label_of(r, a[0][0] if a else None))
                   for r, a in zip(block, amounts_per_row)):
            continue

        # Item rows: those with at least one figure. The modal cell count is the column count.
        counts = [len(a) for a in amounts_per_row if a]
        if not counts:
            continue
        k = int(pd.Series(counts).mode().iat[0])
        xs = [x for a in amounts_per_row for (x, _, _, _) in a if len(a) == k]
        centres = cluster_columns(xs)
        if len(centres) != k:
            centres = cluster_columns([x for a in amounts_per_row for (x, _, _, _) in a])
        if not centres:
            continue

        first_item = next((i for i, a in enumerate(amounts_per_row) if a), 0)
        heads_rows = block[:first_item]
        col_headers, col_scopes = header_columns(heads_rows, centres, doc.ref_date)

        # Where the note labels its columns "Banco" and "Consolidado" the scope is read off the
        # page. Where it labels nothing - a document that IS one scope throughout, which is how the
        # banks that publish outside the regulator's system present them - the scope comes from the
        # document's own title and is marked as such, so a reader can tell the two apart and keep
        # only the column-labelled ones if that is what the design needs.
        scope_source = "column_label"
        if all(s == "unknown" for s in col_scopes):
            if section_scope:
                col_scopes = [section_scope] * len(col_scopes)
                scope_source = "section_title"
            elif doc.scope_hint:
                col_scopes = [doc.scope_hint] * len(col_scopes)
                scope_source = "document_title"
            else:
                scope_source = "not_labelled"

        # The key that ties items to their stated total names the DOCUMENT, not only its date: two
        # documents with the same page and note number would otherwise pool their items against
        # one total, and 17 Volkswagen statements once shared the key "ref_date None". Within the
        # page it names both the note number and the block, never one standing in for the other: a
        # key of "number, or block index when unnumbered" gave note 2 and the unnumbered third
        # block (index 2) of the same page one key, pooling their items against both totals.
        source = doc.entry.get("file") or doc.url
        scope_key = f"{doc.bank_key}|{doc.entity}|{doc.ref_date}|{source}|p{pno}|n{num}|b{bi}"
        emitted_total = False
        block_recs: list[dict] = []
        block_totals: list[dict] = []
        for r, amts in zip(block, amounts_per_row):
            if not amts:
                continue
            lab = label_of(r, amts[0][0])
            if not lab.strip():
                continue
            if FOOTER_RE.match(fold(lab)):    # the running footer: the note ends above it
                break
            cat = category_of(lab)
            is_total = bool(TOTAL_RE.match(fold(lab)))
            for (x, v, raw, _yearlike) in amts:
                j = int(np.argmin([abs(x - c) for c in centres]))
                if abs(x - centres[j]) > 45:
                    continue
                header = col_headers[j] if j < len(col_headers) else ""
                rule, w0, w1, freq = window_of(header, doc.ref_date)
                flags = []
                if rule in ("unparsed", "bare_year", "asof_date_ytd_assumed"):
                    flags.append(rule)
                if scale_src == "milhares_default":
                    flags.append("scale_defaulted_to_thousands")
                sc = col_scopes[j] if j < len(col_scopes) else "unknown"
                rec = line(doc, note_number=num, note_heading=head, section_title=section_title,
                           label_as_published=lab, category_group=cat,
                           cosif_concept=COSIF_CONCEPT.get(cat) if cat else None,
                           is_advertising=bool(cat), scope=sc, scope_source=scope_source,
                           cosif_level=COSIF_LEVEL_OF_SCOPE.get(sc),
                           statement_basis=basis, column_header=header,
                           period_rule=rule, window_start=w0, window_end=w1, frequency=freq,
                           amount_raw=raw, scale=scale, amount_brl=v * scale,
                           is_total=is_total, total_scope=f"{scope_key}|c{j}",
                           reversal=bool(v < 0), column_index=j, n_columns=len(centres),
                           page=pno, parse_method="pdfplumber_words",
                           flag_notes=";".join(flags) or None)
                if is_total:
                    rec["category_group"] = "admin_expense_total"
                    rec["cosif_concept"] = COSIF_CONCEPT["admin_expense_total"]
                    block_totals.append(dict(bank_key=doc.bank_key, entity=doc.entity,
                                             ref_date=doc.ref_date, page=pno, note_number=num,
                                             statement_basis=basis, scope=rec["scope"],
                                             column_header=header, period_rule=rule,
                                             window_start=w0, window_end=w1,
                                             stated=v * scale, label=lab, scale=scale,
                                             scope_id=f"{scope_key}|c{j}",
                                             source_file=doc.entry.get("file")))
                    emitted_total = True
                block_recs.append(rec)

        _normalise_signs(block_recs, block_totals)
        for rec in block_recs:
            out.lines.append(rec)
            if rec["window_end"] and rec["category_group"] in ADV_CATEGORIES:
                out.periods.add((doc.bank_key, doc.entity, rec["scope"], rec["window_start"],
                                 rec["window_end"]))
        out.totals.extend(block_totals)
        if not emitted_total:
            out.notes.append(f"page {pno} note {num or bi}: no stated total row found")


# ---------------------------------------------------------------------------
# Whole-document parse
# ---------------------------------------------------------------------------
def pdf_bytes_of(doc: Doc, path: Path) -> tuple[bytes | None, str | None, str]:
    """(pdf bytes, member name, note). A regulator delivery is a ZIP whose PDF member is the
    published statements; an own-site document is the PDF itself."""
    kind = doc.entry.get("kind")
    if kind == "pdf":
        return path.read_bytes(), None, ""
    if kind == "zip":
        with zipfile.ZipFile(path) as z:
            pdfs = [n for n in z.namelist() if n.lower().endswith(".pdf")]
            if not pdfs:
                return None, None, f"delivery holds no PDF member ({z.namelist()})"
            pdfs.sort(key=lambda n: -z.getinfo(n).file_size)
            return z.read(pdfs[0]), pdfs[0], ""
    return None, None, f"body kind {kind!r} is not a document"


def _resolve_ref_date(doc: Doc, pdf_bytes: bytes) -> str | None:
    """Fill or check an own-site document's reference date against its own text; returns a note.

    Where the file name gave no reference date, the text supplies it and `ref_date_method` becomes
    `document_text`. Where the file name gave one and the text CONCLUSIVELY names another - the
    mode of at least `OVERRIDE_MIN_MENTIONS` mentions, ahead of every other date - the text wins
    and `ref_date_method` becomes `document_text_overrides_filename`, with both dates in the note.
    A file name can carry a publication date that happens to fall on a balance-sheet day: 31 March
    is also the deadline for publishing the annual statements, so "31.03.2019 - Valor ..." is the
    2018 annual statement, which its text says in every heading. A thinner text date does not
    overrule the name; the disagreement is recorded in `ref_date_method` instead, so the document
    can be looked at rather than silently re-dated.
    """
    counts = _text_ref_counts(pdf_bytes)
    text_ref, evidence = _ref_from_counts(counts)
    if doc.ref_date is None:
        if text_ref:
            doc.ref_date, doc.ref_date_method = text_ref, "document_text"
            return f"ref_date {text_ref} from the document text ({evidence})"
        doc.ref_date_method = f"{doc.ref_date_method or 'filename_unrecognised'};text_{evidence}"
        return f"no reference date: the file name gives none and the text is {evidence}"
    if text_ref and text_ref != doc.ref_date:
        if counts[text_ref] >= OVERRIDE_MIN_MENTIONS:
            name_ref, name_method = doc.ref_date, doc.ref_date_method
            doc.ref_date, doc.ref_date_method = text_ref, "document_text_overrides_filename"
            return (f"ref_date {text_ref} from the document text overrides {name_ref} from the "
                    f"file name ({name_method}; text: {evidence})")
        doc.ref_date_method = f"{doc.ref_date_method};conflicts_with_text={text_ref}"
        return (f"ref_date conflict: the file name gives {doc.ref_date}, the document text "
                f"{text_ref} ({evidence}), too few mentions to overrule the name")
    return None


# Archive damage that only shows when a member is read: a bad CRC or a broken deflate stream.
ARCHIVE_ERRORS = (zipfile.BadZipFile, zlib.error, EOFError)


def _pdf_errors() -> tuple[type[BaseException], ...]:
    """What the PDF libraries raise for a PDF they cannot read: a damaged or cut-short file.

    MuPDF raises `FileDataError` when it cannot open the stream at all. pdfminer raises its own
    `PSException` family (`PDFSyntaxError`, `PSEOF` for "Unexpected EOF"), and pdfplumber wraps
    what pdfminer raises while opening in `PdfminerException`. Offline, a file MuPDF repairs
    silently (a garbled cross-reference table) still stops pdfplumber, so both families are
    document damage. Imported here because the libraries are only loaded when a PDF is read.
    """
    import fitz
    from pdfminer.psexceptions import PSException
    errors: list[type[BaseException]] = [fitz.FileDataError, PSException]
    try:
        from pdfplumber.utils.exceptions import PdfminerException
        errors.append(PdfminerException)
    except ImportError:        # a pdfplumber without the wrapper raises pdfminer's errors itself
        pass
    return tuple(errors)


def parse_all(docs: dict[str, list[Doc]], refresh: bool,
              max_docs: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lines, totals, status = [], [], []
    pdf_errors = _pdf_errors()
    for bank, bank_docs in docs.items():
        if not bank_docs:
            continue
        cache = open_cache(bank, refresh)
        seen_sha: dict[str, str] = {}
        for i, doc in enumerate(bank_docs):
            if max_docs is not None and i >= max_docs:
                break
            row = {"bank_key": bank, "url": doc.url, "route": doc.route,
                   "doc_kind": doc.doc_kind, "ref_date": doc.ref_date,
                   "ref_date_method": doc.ref_date_method, "version": doc.version,
                   "filed_at": doc.filed_at, "entity": doc.entity, "label": doc.label,
                   "file": None, "bytes": None, "http_status": None, "kind": None,
                   "member": None, "n_lines": 0, "n_totals": 0, "notes": doc.note or None,
                   "status": None}
            if doc.route == "blocked_host":
                row["status"] = "blocked_host"
                status.append(row)
                continue
            entry = cache.get(doc.url, expect=DOC_KINDS)
            doc.entry = entry
            row.update(file=entry.get("file"), bytes=entry.get("bytes"),
                       http_status=entry.get("status"), kind=entry.get("kind"))
            # The body must be a document before anything else is asked of it. The byte-duplicate
            # test used to come first, so the regulator's identical "busy" texts were reported as
            # duplicates of one another, "parsed once", rather than as failed downloads.
            problem = cache.problem(entry, DOC_KINDS)
            if problem:
                html_page = entry.get("kind") == "html" and _is_2xx(entry.get("status"))
                row.update(status="unavailable_link" if html_page else "fetch_failed",
                           notes=("HTTP 200 returned the site's HTML page, not the document"
                                  if html_page else problem))
                status.append(row)
                continue
            path = cache.file(entry)
            # A damaged delivery is one document's failure, not the run's: it is recorded, and its
            # cache entry is marked failed so the next run downloads it again.
            try:
                body, member, note = pdf_bytes_of(doc, path)
            except ARCHIVE_ERRORS as exc:
                reason = f"damaged archive: {type(exc).__name__}: {exc}"
                cache.invalidate(doc.url, reason)
                row.update(status="bad_archive", notes=reason)
                status.append(row)
                log.warning("  %-11s %-10s %s", bank, doc.ref_date, reason)
                continue
            except Exception as exc:                       # recorded, then raised in validate()
                row.update(status="error", notes=f"{type(exc).__name__}: {exc}")
                status.append(row)
                continue
            if body is None:
                row.update(status="no_pdf_member", notes=note)
                status.append(row)
                continue
            sha = entry.get("sha256")
            if sha and sha in seen_sha:
                row.update(status="duplicate_content",
                           notes=f"byte-identical to {seen_sha[sha]}; parsed once")
                status.append(row)
                continue
            if sha:
                seen_sha[sha] = entry.get("file")
            doc.member = member
            row["member"] = member
            try:
                ref_note = _resolve_ref_date(doc, body) if doc.route == "ir_site" else None
                row.update(ref_date=doc.ref_date, ref_date_method=doc.ref_date_method)
                parsed = parse_pdf_notes(doc, body)
            except pdf_errors as exc:
                # A PDF the libraries cannot read passed every cache check (it starts as a PDF and
                # ends in %%EOF), so without this it would stop every rerun from disk. It is one
                # document's failure, like a damaged archive: recorded, and its entry marked
                # failed so the next run downloads it again.
                reason = f"damaged document: {type(exc).__name__}: {exc}"
                cache.invalidate(doc.url, reason)
                row.update(status="bad_document", notes=reason)
                status.append(row)
                log.warning("  %-11s %-10s %s", bank, doc.ref_date, reason)
                continue
            except Exception as exc:                       # recorded, then raised in validate()
                row.update(status="error", notes=f"{type(exc).__name__}: {exc}")
                status.append(row)
                continue
            for rec in parsed.lines:
                lines.append(rec)
            for t in parsed.totals:
                totals.append(t)
            note_txt = "; ".join(parsed.notes + ([ref_note] if ref_note else [])) or None
            row.update(n_lines=len(parsed.lines), n_totals=len(parsed.totals), notes=note_txt,
                       status=("unreadable" if "unreadable_no_text_layer" in parsed.notes
                               else "parsed" if parsed.lines else "no_advertising_line"))
            status.append(row)
            log.info("  %-11s %-10s %-11s lines %4d  totals %3d  %s", bank, doc.ref_date,
                     row["status"], row["n_lines"], row["n_totals"], note_txt or "")
    return pd.DataFrame(lines), pd.DataFrame(totals), pd.DataFrame(status)


# ---------------------------------------------------------------------------
# The note's own arithmetic
# ---------------------------------------------------------------------------
def check_totals(lines: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Each note's stated administrative-expense total against the sum of its itemised lines.

    This is the third reconciliation: it tests that the column was read as a column, because a
    misassigned cell breaks the sum even when every individual number is right.
    """
    if totals.empty or lines.empty:
        return totals
    items = lines[~lines["is_total"].astype(bool)]
    sums = items.groupby("total_scope")["amount_brl"].sum()
    counts = items.groupby("total_scope")["amount_brl"].size()
    out = totals.copy()
    out["lines_sum"] = out["scope_id"].map(sums)
    out["n_items"] = out["scope_id"].map(counts)
    out["diff"] = out["lines_sum"] - out["stated"]
    # The published figures are exact multiples of the note's scale, so the tolerance is one unit
    # of that scale, not a percentage: a proportional tolerance on a thousands-scaled billion is
    # millions wide and hides exactly the mis-assigned cell this check exists to catch.
    tol = out["scale"].fillna(SCALE_THOUSANDS).abs().clip(lower=1.0)
    out["matches"] = out["lines_sum"].notna() & (out["diff"].abs() <= tol)
    return out


def check_items_against_total(lines: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Advertising as a share of the note's own stated administrative-expense total.

    Not a pass/fail: a number to quote per bank, and a share above 1 is an error rather than a
    large share.
    """
    if totals.empty or lines.empty:
        return pd.DataFrame()
    adv = (lines[lines["is_advertising"].astype(bool) & ~lines["is_total"].astype(bool)]
           .groupby("total_scope")["amount_brl"].sum().rename("adv_sum"))
    out = totals.join(adv, on="scope_id")
    out["adv_share_of_total"] = out["adv_sum"] / out["stated"]
    return out


# ---------------------------------------------------------------------------
# Windows -> periods
# ---------------------------------------------------------------------------
def _months(w0: str, w1: str) -> int:
    return (int(w1[:4]) - int(w0[:4])) * 12 + int(w1[5:7]) - int(w0[5:7]) + 1


def period_table(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per entity-scope-account-window, and the increments those windows imply.

    Year-to-date windows inside one calendar year are differenced against the next shorter one, so a
    half-year note yields H1 as filed and H2 by difference, and a bank filing quarterly yields four
    quarters. Nothing is spread: a window that no shorter window sits inside stays as it is, with
    `derivation = as_filed` and its own frequency.
    """
    if lines.empty:
        return pd.DataFrame()
    df = lines[lines["window_start"].notna() & lines["window_end"].notna()
               & lines["category_group"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["filed_key"] = df["filed_at"].fillna("").astype(str) + "|" + \
        df["version"].fillna(0).astype(int).astype(str)
    # `section_title` is part of the identity: a delivery can carry two administrative-expense notes
    # from two statement sets, and pooling them would mix two different aggregates.
    key = ["bank_key", "panel_code", "cnpj8", "entity", "scope", "scope_source", "cosif_level",
           "statement_basis", "section_title", "category_group", "cosif_concept",
           "window_start", "window_end"]
    # Last filed value wins: order by delivery date and version, keep the last.
    df = df.sort_values(key + ["filed_key"]).groupby(key, as_index=False, dropna=False).agg(
        amount_brl=("amount_brl", "last"), label_as_published=("label_as_published", "last"),
        period_rule=("period_rule", "last"), frequency=("frequency", "last"),
        n_filings=("amount_brl", "size"), amount_first_filed=("amount_brl", "first"),
        ref_date=("ref_date", "last"), filed_at=("filed_at", "last"), version=("version", "last"),
        source_url=("source_url", "last"), page=("page", "last"),
        flag_notes=("flag_notes", "last"))
    df["restated"] = (df["n_filings"] > 1) & (
        (df["amount_brl"] - df["amount_first_filed"]).abs() > 1.0)
    df["n_months"] = [_months(a, b) for a, b in zip(df["window_start"], df["window_end"])]
    df["derivation"] = "as_filed"

    # Increments. Within one calendar year and one series, a window starting 1 January is a
    # year-to-date figure; the shorter one immediately inside it gives the increment.
    rows = [df]
    gkey = ["bank_key", "panel_code", "cnpj8", "entity", "scope", "statement_basis",
            "category_group", "section_title"]
    for _, g in df.groupby(gkey, dropna=False):
        ytd = g[(g["window_start"].str[5:] == "01-01")].sort_values("n_months")
        for year, gy in ytd.groupby(ytd["window_start"].str[:4]):
            gy = gy.sort_values("n_months")
            prev = None
            for r in gy.to_dict("records"):
                if prev is not None and r["n_months"] > prev["n_months"]:
                    inc = dict(r)
                    inc["amount_brl"] = r["amount_brl"] - prev["amount_brl"]
                    inc["window_start"] = _add_month(prev["window_end"])
                    inc["n_months"] = r["n_months"] - prev["n_months"]
                    inc["frequency"] = {3: "quarter", 6: "semester",
                                        12: "year"}.get(inc["n_months"], f"{inc['n_months']}m")
                    inc["derivation"] = f"diff_{r['n_months']}m_minus_{prev['n_months']}m"
                    inc["period_rule"] = f"{r['period_rule']}->increment"
                    rows.append(pd.DataFrame([inc]))
                prev = r
    out = pd.concat(rows, ignore_index=True)
    out["period"] = [_period_label(a, b) for a, b in zip(out["window_start"], out["window_end"])]
    return out.sort_values(["bank_key", "entity", "scope", "category_group",
                            "window_start", "n_months"]).reset_index(drop=True)


def _add_month(date: str) -> str:
    y, m = int(date[:4]), int(date[5:7])
    m += 1
    if m > 12:
        y, m = y + 1, 1
    return f"{y}-{m:02d}-01"


def _period_label(w0: str, w1: str) -> str:
    n = _months(w0, w1)
    y, m1, m2 = w1[:4], int(w0[5:7]), int(w1[5:7])
    if n == 3 and m1 in (1, 4, 7, 10):
        return f"{y}Q{(m2 - 1) // 3 + 1}"
    if n == 6 and m1 in (1, 7):
        return f"{y}H{1 if m1 == 1 else 2}"
    if n == 12 and m1 == 1:
        return f"{y}"
    return f"{w0}:{w1}"


# ---------------------------------------------------------------------------
# Reconciliations
# ---------------------------------------------------------------------------
def attach_panel_keys(lines: pd.DataFrame) -> pd.DataFrame:
    """Panel conglomerate code and CNPJ root per bank, from the registry and the crosswalk."""
    code_of, cnpj_of = {}, {}
    for t in NOTE_TARGETS:
        b = BANK_OF_PANEL_CODE[t["panel_code"]]
        code_of[b] = t["panel_code"]
        cnpj_of[b] = t["cnpj8"]
    if lines.empty:
        return lines
    lines["panel_code"] = lines["bank_key"].map(code_of)
    lines["cnpj8"] = lines["bank_key"].map(cnpj_of)
    return lines


def reconcile_cosif(periods: pd.DataFrame) -> pd.DataFrame:
    """THE calibration: a 2025 half-year note figure against the COSIF sum for the same months.

    The note and the COSIF accounts are the same accounting object, so the ratio says whether the
    note was read right - the scale, the column and the account the label names.

    One series at a time. A delivery can carry two administrative-expense notes from two statement
    sets with the same scope, and adding them would report a ratio of 1.8 for a reading that is
    exactly right: Santander's consolidated H1 2025 is R$299.2m from the individual-and-consolidated
    statements, R$233.5m from the condensed consolidated interim ones, and R$532.7m only if the two
    are wrongly summed. So the series identity here is the full key, `section_title` included.

    `ratio_on_mapped_concept` is the number to read: the note figure over whichever COSIF aggregate
    `cosif_concept` says the published label names, at whichever level `cosif_level` says the scope
    is comparable with. The other two ratios are kept beside it because they are what the mapping
    had to be chosen between.
    """
    path = OUT_DIR / "cosif_advertising_monthly.parquet"
    if not path.exists():
        log.warning("no %s; the COSIF calibration is skipped", path.name)
        return pd.DataFrame()
    if periods.empty:
        log.warning("no periods derived; the COSIF calibration has nothing to compare")
        return pd.DataFrame()
    cos = pd.read_parquet(path)
    series_key = ["bank_key", "panel_code", "cnpj8", "entity", "scope", "scope_source",
                  "statement_basis", "section_title", "category_group", "cosif_concept"]
    rows = []
    adv_cols = ["adv_flow", "promo_flow", "publ_flow"]
    for keys, g in periods.groupby(series_key, dropna=False):
        k = dict(zip(series_key, keys))
        if k["statement_basis"] == "ifrs" or k["category_group"] not in ADV_CATEGORIES:
            continue
        for half, (m0, m1) in (("H1", (1, 6)), ("H2", (7, 12))):
            note = g[(g["window_start"] == f"2025-{m0:02d}-01")
                     & (g["window_end"].str[5:7] == f"{m1:02d}")]
            if note.empty:
                continue
            if len(note) > 1:                 # one series, one window: never summed
                note = note.sort_values("filed_at").tail(1)
            value = float(note["amount_brl"].iat[0])
            for level, ekey in (("institution", str(k["cnpj8"])),
                                ("conglomerate", str(k["panel_code"]))):
                c = cos[(cos["level"] == level) & (cos["entity_key"].astype(str) == ekey)
                        & (cos["year"] == 2025) & (cos["month"].between(m0, m1))]
                if c.empty:
                    continue
                adv = float(c["adv_flow"].sum())
                all3 = float(c[adv_cols].sum().sum())
                mapped = {"cosif_adv_8174500005": adv,
                          "cosif_all3_adv_promo_publ": all3,
                          "cosif_adv_plus_promo": float(c[["adv_flow", "promo_flow"]].sum().sum()),
                          "cosif_promo_8174200002": float(c["promo_flow"].sum()),
                          "cosif_publ_8174800006": float(c["publ_flow"].sum()),
                          }.get(k["cosif_concept"])
                rows.append(dict(
                    **k, half=half, cosif_level=level, cosif_entity=ekey,
                    level_matches_scope=(COSIF_LEVEL_OF_SCOPE.get(k["scope"]) == level),
                    note_label=note["label_as_published"].iat[0], note_brl=value,
                    cosif_adv_brl=adv, cosif_all3_brl=all3,
                    cosif_mapped_brl=mapped, cosif_months=int(c["month"].nunique()),
                    ratio_note_over_adv=value / adv if adv else np.nan,
                    ratio_note_over_all3=value / all3 if all3 else np.nan,
                    ratio_on_mapped_concept=(value / mapped) if mapped else np.nan))
    out = pd.DataFrame(rows)
    if not out.empty:
        head = out[out["level_matches_scope"]] if out["level_matches_scope"].any() else out
        log.info("COSIF 2025 calibration (scope matched to its COSIF level):\n%s", head[
            ["bank_key", "half", "scope", "cosif_level", "cosif_concept", "note_brl",
             "cosif_mapped_brl", "ratio_on_mapped_concept", "ratio_note_over_adv",
             "cosif_months"]].to_string(index=False))
    return out


def reconcile_cvm_structured(periods: pd.DataFrame) -> pd.DataFrame:
    """Where the structured filings DO carry an advertising line, the note must match it."""
    path = OUT_DIR / "cvm_advertising_quarterly.parquet"
    if not path.exists():
        log.warning("no %s; the structured-filing comparison is skipped", path.name)
        return pd.DataFrame()
    if periods.empty:
        log.warning("no periods derived; the structured-filing comparison has nothing to compare")
        return pd.DataFrame()
    q = pd.read_parquet(path)
    rows = []
    for (bank, cnpj8), g in periods.groupby(["bank_key", "cnpj8"], dropna=False):
        s = q[q["cnpj8"].astype(str).str.zfill(8) == str(cnpj8).zfill(8)]
        if s.empty:
            continue
        ann = g[(g["n_months"] == 12) & (g["scope"] == "individual")
                & (g["statement_basis"] != "ifrs")
                & g["category_group"].isin(ADV_CATEGORIES)]
        for r in ann.to_dict("records"):
            y = int(r["window_end"][:4])
            sy = s[(s["year"] == y) & (s["scope"] == "ind")]
            if sy.empty:
                continue
            filed = float(sy["flow_restated"].fillna(sy["flow_first"]).sum())
            rows.append(dict(bank_key=bank, year=y, note_brl=float(r["amount_brl"]),
                             structured_brl=filed,
                             ratio=(float(r["amount_brl"]) / filed) if filed else np.nan,
                             note_label=r["label_as_published"]))
    out = pd.DataFrame(rows)
    if not out.empty:
        log.info("structured-filing comparison:\n%s", out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(lines: pd.DataFrame, checks: pd.DataFrame, status: pd.DataFrame,
             cosif: pd.DataFrame, unvalidated: tuple[str, ...] = ()) -> None:
    """Hard assertions. Banks named in `unvalidated` may fall below the arithmetic floor; their
    rows are written with validated = False and the shortfall is logged, so nothing passes
    silently."""
    errors = status[status["status"] == "error"] if not status.empty else status
    if len(errors):                                                            # check 1
        raise AssertionError(f"{len(errors)} documents raised while parsing:\n"
                             f"{errors[['bank_key', 'ref_date', 'notes']].to_string(index=False)}")
    if not status.empty and status["status"].isna().any():
        raise AssertionError("a listed document has no status")
    if not status.empty:
        log.info("document status by bank:\n%s",
                 status.groupby(["bank_key", "status"]).size().unstack(fill_value=0).to_string())

    if lines.empty:                                                            # check 2
        raise AssertionError("no note lines parsed from any document")

    bad_scale = lines[~lines["scale"].isin([1.0, 1_000.0, 1_000_000.0])]
    if len(bad_scale):                                                         # check 3
        raise AssertionError(f"{len(bad_scale)} lines carry an unrecognised scale")

    unparsed = lines[lines["period_rule"] == "unparsed"]
    share = len(unparsed) / len(lines)
    log.info("  window read from the column header on %.1f%% of %d lines",
             100 * (1 - share), len(lines))
    if share > 0.25:                                                           # check 4
        raise AssertionError(
            f"{share:.1%} of lines have a column header no window rule recognises; worst:\n"
            f"{unparsed[['bank_key', 'ref_date', 'column_header']].head(10).to_string(index=False)}")

    if not checks.empty:                                                       # check 5
        for bank, grp in checks.groupby("bank_key"):
            m = float(grp["matches"].mean())
            log.info("  %s: %d stated note totals, %.1f%% equal the sum of the note's items",
                     bank, len(grp), 100 * m)
            if m < MIN_TOTAL_MATCH and bank not in unvalidated:
                worst = grp[~grp["matches"]].nlargest(5, "stated")
                raise AssertionError(
                    f"{bank}: only {m:.1%} of stated note totals equal their item sums "
                    f"(floor {MIN_TOTAL_MATCH:.0%}); worst:\n"
                    f"{worst[['ref_date', 'page', 'column_header', 'stated', 'lines_sum']].to_string(index=False)}")
            if m < MIN_TOTAL_MATCH:
                log.warning("%s: %.1f%% of note totals match, below the %.0f%% floor; its rows are "
                            "written with validated = False", bank, 100 * m, 100 * MIN_TOTAL_MATCH)

    shares = check_items_against_total(lines, checks)                          # check 6
    if not shares.empty and shares["adv_share_of_total"].notna().any():
        over = shares[shares["adv_share_of_total"] > 1.0]
        log.info("  advertising is %.1f%% of the note's stated administrative total at the median "
                 "(n=%d columns)", 100 * shares["adv_share_of_total"].median(),
                 int(shares["adv_share_of_total"].notna().sum()))
        if len(over):
            raise AssertionError(
                f"{len(over)} note columns make advertising exceed the note's own administrative "
                f"total, so a cell was read into the wrong column:\n"
                f"{over[['bank_key', 'ref_date', 'page', 'column_header', 'adv_sum', 'stated']].head(10).to_string(index=False)}")

    if not cosif.empty:                                                        # check 7
        strong = cosif[cosif["level_matches_scope"] & (cosif["cosif_months"] == 6)
                       & cosif["ratio_on_mapped_concept"].notna()]
        if len(strong):
            off = strong[(strong["ratio_on_mapped_concept"] < 0.8)
                         | (strong["ratio_on_mapped_concept"] > 1.25)]
            log.info("  COSIF calibration: %d of %d scope-matched half-years within [0.80, 1.25] "
                     "of the COSIF aggregate their label maps to; median ratio %.4f",
                     len(strong) - len(off), len(strong),
                     float(strong["ratio_on_mapped_concept"].median()))
            if len(off) == len(strong):
                raise AssertionError(
                    "no 2025 half-year note figure lands within a quarter of the COSIF sum for the "
                    "account its own label names, so the note reading is not calibrated:\n"
                    f"{strong[['bank_key', 'scope', 'cosif_concept', 'note_brl', 'cosif_mapped_brl', 'ratio_on_mapped_concept']].to_string(index=False)}")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def output_scope(a: argparse.Namespace,
                 ap: argparse.ArgumentParser) -> tuple[str, frozenset[str], list[str]]:
    """(mode, banks processed, restrictions other than bank and route) for this run.

    `full`     no restriction: every output is replaced.
    `merge`    restricted by --banks and/or --routes only: this run read EVERY document of the
               banks it processed, so their rows are replaced and every other bank's are kept.
    `partial`  any other restriction: a partial reading of the banks it touched, which would pass
               for their full series if merged, so it is only ever written into an --out-dir.
    """
    processed = frozenset(b for b in a.banks if route_of(b) in a.routes)
    other = [flag for flag, differs in (
        ("--from-year", a.from_year != ap.get_default("from_year")),
        ("--to-year", a.to_year != ap.get_default("to_year")),
        ("--ref-months", sorted(set(a.ref_months)) != sorted(set(ap.get_default("ref_months")))),
        ("--max-docs", a.max_docs != ap.get_default("max_docs")),
        ("--versions", a.versions != ap.get_default("versions")),
    ) if differs]
    if other:
        return "partial", processed, other
    by_bank = (set(a.banks) != set(ap.get_default("banks"))
               or set(a.routes) != set(ap.get_default("routes")))
    return ("merge" if by_bank else "full"), processed, []


def _same_dir(p: Path, q: Path) -> bool:
    """Whether two spellings name one folder.

    By identity where both exist (`os.path.samefile`: the same volume and file id), because a
    spelling can reach the default output folder without resolving to the same string: `resolve()`
    leaves an admin-share path (\\\\host\\C$\\...) as it is. Where either folder does not exist yet,
    the resolved spellings are compared.
    """
    try:
        if os.path.exists(p) and os.path.exists(q):
            return os.path.samefile(p, q)
    except OSError:
        pass

    def norm(x: Path) -> str:
        s = str(Path(x).resolve())
        return os.path.normcase(s[4:] if s.startswith("\\\\?\\") else s)
    return norm(p) == norm(q)


def _windows_delete_blocker(path: Path) -> str | None:
    """On Windows, why `path` cannot be opened with DELETE access now, or None. A rename over a
    file needs that access, and a program holding the file open without delete sharing (most
    readers, Python's own open() included) denies it while still allowing writes."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    delete, share_all, open_existing = 0x00010000, 0x7, 3
    handle = k32.CreateFileW(str(path), delete, share_all, None, open_existing, 0, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        err = ctypes.get_last_error()
        if err in (2, 3):                                  # the file or its folder is gone
            return None
        return f"WinError {err}: {ctypes.FormatError(err).strip()}"
    k32.CloseHandle(handle)
    return None


def _replace_blocker(path: Path) -> str | None:
    """Why `path` could not be replaced by a rename right now, or None when it can or is absent.

    Opening it for writing fails where another program holds it with writes denied (Excel on a
    csv) or the file is read-only; on Windows the DELETE access a rename needs is probed as well.
    """
    try:
        with open(path, "r+b"):
            pass
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"{type(exc).__name__}: {exc.strerror or exc}"
    return _windows_delete_blocker(path) if os.name == "nt" else None


def write_outputs(out_dir: Path, tables: dict[str, pd.DataFrame], mode: str,
                  processed: frozenset[str]) -> None:
    """Write every output table as parquet and csv, never leaving a half-written or stale set.

    Every file is staged under a `.tmp` name first and only then renamed over the previous one, so a
    run that fails or is killed while writing leaves the previous outputs whole and each parquet
    beside its matching csv. In `merge` mode a table is this run's rows plus the rows already on
    disk for every bank it did not process. A table left with no rows is not written, and the file a
    previous run left under its name is removed, because a stale file would pass for this run's.

    The output folder's lock is held from reading the previous outputs to the last rename, so a
    second run writing into the same folder waits instead of merging into a table the first is
    about to replace, which would drop the first run's rows. Before the first rename every target
    is checked to be replaceable, so a file held open elsewhere stops the write with every table
    still whole; a failure after that is logged with the files already replaced.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(out_dir / OUTPUT_LOCK_NAME, f"output folder {out_dir}")
    lock.acquire(wait=OUTPUT_LOCK_WAIT)
    try:
        _write_outputs_locked(out_dir, tables, mode, processed)
    finally:
        lock.release()


def _write_outputs_locked(out_dir: Path, tables: dict[str, pd.DataFrame], mode: str,
                          processed: frozenset[str]) -> None:
    def discard_staged() -> None:
        for name in OUTPUT_TABLES:
            for ext in ("parquet", "csv"):
                with contextlib.suppress(OSError):
                    (out_dir / f"{name}.{ext}.tmp").unlink(missing_ok=True)

    # Staged files a killed run left behind are this module's own and never current.
    discard_staged()
    final: dict[str, pd.DataFrame] = {}
    for name in OUTPUT_TABLES:
        new = tables.get(name)
        new = pd.DataFrame() if new is None else new
        if not new.empty and "bank_key" not in new.columns:
            raise AssertionError(f"{name} has no bank_key column, so its rows cannot be attributed "
                                 f"to a bank")
        if mode == "merge":
            stray = set(new["bank_key"]) - processed if not new.empty else set()
            if stray:
                raise AssertionError(f"{name} holds rows for banks this run did not process: "
                                     f"{sorted(stray)}")
            path = out_dir / f"{name}.parquet"
            kept = pd.DataFrame()
            if path.exists():
                old = pd.read_parquet(path)
                if not old.empty and "bank_key" not in old.columns:
                    raise SystemExit(
                        f"{path} has no bank_key column, so a --banks/--routes run cannot replace "
                        f"only its own banks' rows in it. Rerun without --banks and --routes to "
                        f"rebuild every output.")
                if not old.empty:
                    kept = old[~old["bank_key"].isin(processed)]
                    log.info("%s: keeping %d rows of %d other banks, replacing %d rows of %s",
                             name, len(kept), kept["bank_key"].nunique(), len(old) - len(kept),
                             ", ".join(sorted(processed)))
            parts = [f for f in (kept, new) if not f.empty]
            new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        final[name] = new

    staged: list[str] = []
    try:
        for name, frame in final.items():
            if frame.empty:
                continue
            frame.to_parquet(out_dir / f"{name}.parquet.tmp", index=False)
            frame.to_csv(out_dir / f"{name}.csv.tmp", index=False)
            staged.append(name)
    except BaseException:
        discard_staged()
        raise

    # Every file this write will replace or remove, checked before the first rename. A sync
    # client holds a file for a moment, so a blocked file is looked at again before giving up.
    targets = [out_dir / f"{name}.{ext}" for name in OUTPUT_TABLES for ext in ("parquet", "csv")]
    blocked: list[str] = []
    for attempt in range(4):
        if attempt:
            time.sleep(1.0)
        blocked = [f"{p.name} ({why})" for p in targets if (why := _replace_blocker(p))]
        if not blocked:
            break
    if blocked:
        discard_staged()
        raise SystemExit(
            f"cannot replace {len(blocked)} output file(s) in {out_dir}: {'; '.join(blocked)}. "
            f"Close the program holding them and rerun. No output was replaced.")

    done: list[str] = []
    try:
        for name in staged:
            for ext in ("parquet", "csv"):
                _replace(out_dir / f"{name}.{ext}.tmp", out_dir / f"{name}.{ext}")
                done.append(f"{name}.{ext}")
            log.info("%s: %d rows -> %s", name, len(final[name]), out_dir / f"{name}.parquet")
        for name, frame in final.items():
            if not frame.empty:
                continue
            stale = [p for p in (out_dir / f"{name}.parquet", out_dir / f"{name}.csv")
                     if p.exists()]
            for p in stale:
                p.unlink()
                done.append(f"{p.name} (removed)")
            if stale:
                log.warning("%s has no rows in this run; removed the previous run's %s so it does "
                            "not pass for current", name, " and ".join(p.name for p in stale))
            else:
                log.warning("%s has no rows in this run; not written", name)
    except BaseException:
        pending = [f"{n}.{e}" for n in staged for e in ("parquet", "csv")
                   if f"{n}.{e}" not in done]
        log.error("writing the outputs into %s stopped after %d file change(s). Done: %s. Not "
                  "done: %s. %s", out_dir, len(done), ", ".join(done) or "none",
                  ", ".join(pending) or "none",
                  "Nothing was replaced." if not done else
                  "The folder now mixes this run's tables with the previous run's; rerun to "
                  "write the set whole.")
        discard_staged()
        raise


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        _main()
    finally:
        close_caches()


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Advertising expense from the notes to banks' audited financial statements")
    ap.add_argument("--banks", nargs="+", default=list(BANK_KEYS), choices=BANK_KEYS)
    ap.add_argument("--routes", nargs="+", default=["cvm_rad", "ir_site"],
                    choices=["cvm_rad", "ir_site"])
    ap.add_argument("--ref-months", nargs="+", type=int, default=[3, 6, 9, 12],
                    help="reference months to fetch for the regulator's filers: 6 and 12 are the "
                         "half-year statements, 3 and 9 add the quarterly ones")
    ap.add_argument("--from-year", type=int, default=2016)
    ap.add_argument("--to-year", type=int, default=2025)
    ap.add_argument("--versions", choices=["last", "all"], default="last",
                    help="'last' keeps the highest version of each reference date, which is the "
                         "value that stands; 'all' keeps every delivery so a restatement is visible")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download listings and documents; a failed re-download keeps the "
                         "cached copy. Without it, only failed or damaged cache entries are "
                         "requested again")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="stop after this many documents per bank (for a smoke run)")
    ap.add_argument("--unvalidated", nargs="*", default=[], choices=BANK_KEYS,
                    help="banks allowed below the note-arithmetic floor; their rows carry "
                         "validated = False")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="where the outputs go. A run restricted by anything other than --banks "
                         "and --routes must name a folder other than the default one")
    a = ap.parse_args()

    # Decided before anything is fetched, so a run that could not write its result stops at once.
    mode, processed, other = output_scope(a, ap)
    if mode == "partial" and not a.download_only and _same_dir(a.out_dir, OUT_DIR):
        raise SystemExit(
            f"this run is restricted by {', '.join(other)}, so it reads only part of each bank's "
            f"documents. Written into {OUT_DIR} it would replace the combined outputs with that "
            f"subset. Pass --out-dir <another folder> for a partial run; only --banks and --routes "
            f"runs merge into the default folder.")
    log.info("output mode: %s (%s)", mode,
             "banks processed: " + ", ".join(sorted(processed)) if mode != "full" else "all banks")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs, blocked = collect(tuple(a.banks), a.refresh, tuple(a.ref_months), a.from_year,
                            a.to_year, a.versions, tuple(a.routes))
    if a.download_only:
        for bank, ds in docs.items():
            if not ds:
                continue
            cache = open_cache(bank, a.refresh)
            for d in ds:
                if d.route != "blocked_host":
                    cache.get(d.url, expect=DOC_KINDS)
        return

    lines, totals, status = parse_all(docs, a.refresh, a.max_docs)
    if not status.empty and blocked:
        status = pd.concat([status, pd.DataFrame(blocked)], ignore_index=True)
    elif blocked:
        status = pd.DataFrame(blocked)
    if lines.empty:
        raise SystemExit("no note lines parsed from any document")

    lines = attach_panel_keys(lines)
    checks = check_totals(lines, totals)
    shares = check_items_against_total(lines, totals)

    match_of = checks.groupby("bank_key")["matches"].mean() if not checks.empty else pd.Series(
        dtype=float)
    lines["validated"] = lines["bank_key"].map(
        lambda b: bool(match_of.get(b, 0) >= MIN_TOTAL_MATCH))

    periods = period_table(lines)
    if not periods.empty:
        periods["validated"] = periods["bank_key"].map(
            lambda b: bool(match_of.get(b, 0) >= MIN_TOTAL_MATCH))
    cosif = reconcile_cosif(periods)
    structured = reconcile_cvm_structured(periods)
    validate(lines, checks, status, cosif, tuple(a.unvalidated))

    if not shares.empty:
        log.info("advertising as a share of the note's stated administrative total:\n%s",
                 shares.groupby("bank_key")["adv_share_of_total"]
                       .describe()[["count", "mean", "min", "max"]].to_string())
    if not periods.empty:
        log.info("coverage: windows with an advertising line per bank and year:\n%s",
                 periods.assign(year=periods["window_end"].str[:4])
                        .groupby(["bank_key", "year"])["period"].nunique()
                        .unstack(fill_value=0).to_string())

    tables = dict(zip(OUTPUT_TABLES, (lines, periods, status, checks, cosif, structured)))
    # A partial run only ever reaches here with its own --out-dir, where it is the whole content.
    write_outputs(a.out_dir, tables, "merge" if mode == "merge" else "full", processed)


if __name__ == "__main__":
    main()
