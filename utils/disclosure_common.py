"""disclosure_common.py
# Author: Pedro Feijó de Moraes
#
# Shared helpers for the firm-disclosure scrapers (scrape_17..scrape_23):
#   - SEC-compliant, throttled, gzip-aware HTTP GET with retries
#   - HTML -> text flattening
#   - human-number / currency / period parsing
#   - the canonical LONG-format firm-disclosure panel schema + writer
#
# These scrapers collect, per prudential conglomerate (the project's unit of
# analysis), the *count* observations (customers / accounts) and the matching
# deposit volume, so that an account-share-vs-volume-share (extensive vs
# intensive margin) diagnostic can be built. See V_Main.tex appendix
# `app:extensions:intensive` for why the count moment matters.
#
# Output is LONG format (one row per metric observation) because the firm
# disclosures are heterogeneous: some report total customers, some active,
# some accounts, some deposits, some Brazil-only, some consolidated.
"""

from __future__ import annotations

import os
import re
import gzip
import time
import html
import json
import logging
import threading
import datetime as _dt
import urllib.request
import urllib.error
from typing import Iterable

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

# SEC requires a descriptive User-Agent with a contact e-mail and asks for
# <= 10 requests/second. We default to the project owner's e-mail but allow an
# override via the DISCLOSURE_CONTACT_EMAIL environment variable.
_CONTACT = os.environ.get("DISCLOSURE_CONTACT_EMAIL", "pedro.feijodemoraes@yale.edu").strip()
USER_AGENT = f"deposit-competition-research ({_CONTACT})"

_THROTTLE_LOCK = threading.Lock()
_LAST_HIT: dict[str, float] = {}
# Minimum seconds between requests to the same host (SEC is the strict one).
_MIN_INTERVAL = {
    "www.sec.gov": 0.20,
    "data.sec.gov": 0.20,
    "efts.sec.gov": 0.30,
}
_DEFAULT_INTERVAL = 0.10


def _host_of(url: str) -> str:
    m = re.match(r"https?://([^/]+)/?", url)
    return m.group(1).lower() if m else ""


def _throttle(url: str) -> None:
    host = _host_of(url)
    interval = _MIN_INTERVAL.get(host, _DEFAULT_INTERVAL)
    with _THROTTLE_LOCK:
        now = time.monotonic()
        last = _LAST_HIT.get(host, 0.0)
        wait = interval - (now - last)
        if wait > 0:
            time.sleep(wait)
        _LAST_HIT[host] = time.monotonic()


def http_get(
    url: str,
    *,
    binary: bool = False,
    timeout: int = 45,
    retries: int = 3,
    extra_headers: dict | None = None,
):
    """GET with throttle, gzip handling and exponential-backoff retries.

    Returns (status_code:int|None, body:str|bytes|None). On persistent failure
    returns (None, None) rather than raising, so a single bad URL does not abort
    a long scrape.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "*/*",
    }
    if extra_headers:
        headers.update(extra_headers)

    for attempt in range(1, retries + 1):
        _throttle(url)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding", "") == "gzip":
                    raw = gzip.decompress(raw)
                status = getattr(resp, "status", 200)
            if binary:
                return status, raw
            return status, raw.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                log.debug(f"HTTP {e.code} for {url}")
                return e.code, None
            log.warning(f"HTTP {e.code} on {url} (attempt {attempt}/{retries})")
        except Exception as e:  # noqa: BLE001
            log.warning(f"GET failed {type(e).__name__} on {url} (attempt {attempt}/{retries})")
        time.sleep(min(8.0, 0.8 * 2 ** attempt))
    return None, None


def get_json(url: str, **kw):
    code, body = http_get(url, **kw)
    if body is None:
        return code, None
    try:
        return code, json.loads(body)
    except Exception:  # noqa: BLE001
        return code, None


# ----------------------------------------------------------------------------
# HTML / text
# ----------------------------------------------------------------------------

def strip_html(text: str) -> str:
    """Flatten HTML to whitespace-normalised text (tables become spaced runs)."""
    if not text:
        return ""
    text = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", text)
    # keep row/cell boundaries as spaces, block ends as newlines
    text = re.sub(r"(?i)</(tr|p|div|h[1-6]|li|table)\s*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("‐", "-").replace("–", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


# ----------------------------------------------------------------------------
# Number / currency / period parsing
# ----------------------------------------------------------------------------

_MULT = {
    "thousand": 1e3, "thousands": 1e3, "mil": 1e3,
    "million": 1e6, "millions": 1e6, "mn": 1e6, "mm": 1e6, "mi": 1e6, "milhão": 1e6, "milhoes": 1e6, "milhões": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "bi": 1e9, "bilhão": 1e9, "bilhoes": 1e9, "bilhões": 1e9,
}


def parse_number(raw: str, *, brazilian: bool = False) -> float | None:
    """Parse a numeric token. `brazilian=True` treats '.' as thousands sep and
    ',' as decimal (pt-BR); otherwise en-US convention."""
    if raw is None:
        return None
    s = str(raw).strip().replace(" ", "").replace(" ", "")
    s = re.sub(r"[R$US]+", "", s, flags=re.I).strip()
    if not re.search(r"\d", s):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        if brazilian:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
        val = float(s)
        return -val if neg else val
    except ValueError:
        return None


def parse_human_number(text: str, *, brazilian: bool = False) -> float | None:
    """Parse '1.2 million', '42,448,121', 'R$ 8,5 bilhões' -> absolute float."""
    if not text:
        return None
    t = text.strip()
    m = re.search(r"(-?\(?[\d][\d.,]*\)?)\s*([A-Za-zãõ]+)?", t)
    if not m:
        return None
    num = parse_number(m.group(1), brazilian=brazilian)
    if num is None:
        return None
    unit = (m.group(2) or "").lower()
    return num * _MULT.get(unit, 1.0)


_QMAP = {
    "1": 1, "first": 1, "i": 1, "primeiro": 1,
    "2": 2, "second": 2, "ii": 2, "segundo": 2,
    "3": 3, "third": 3, "iii": 3, "terceiro": 3,
    "4": 4, "fourth": 4, "iv": 4, "quarto": 4,
}


def parse_period(text: str) -> tuple[int | None, int | None, str | None]:
    """Best-effort (year, quarter, label) from strings like 'Q1 2026',
    '1Q26', '1T26', 'March 31, 2026', '3M26', 'FY2025', '2025'."""
    if not text:
        return None, None, None
    t = text.strip()
    # Q1'26 / 1Q26 / 1T26
    m = re.search(r"\b([1-4])\s*[QT]\s*'?\s*(\d{2,4})\b", t, re.I) or \
        re.search(r"\b[QT]\s*([1-4])\s*'?\s*(\d{2,4})\b", t, re.I)
    if m:
        q = int(m.group(1)); y = int(m.group(2)); y = y + 2000 if y < 100 else y
        return y, q, f"{q}Q{y}"
    # Q1 2026 / first quarter 2025
    m = re.search(r"(?:Q([1-4])|(first|second|third|fourth)\s+quarter).{0,12}?(\d{4})", t, re.I)
    if m:
        q = int(m.group(1)) if m.group(1) else _QMAP[m.group(2).lower()]
        y = int(m.group(3)); return y, q, f"{q}Q{y}"
    # Month DD, YYYY -> quarter
    m = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+(\d{4})", t, re.I)
    if m:
        mo = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"].index(m.group(1)[:3].lower()) + 1
        y = int(m.group(2)); return y, (mo - 1) // 3 + 1, f"{(mo-1)//3+1}Q{y}"
    # bare FY / year
    m = re.search(r"\b(?:FY)?\s?(20\d{2})\b", t)
    if m:
        return int(m.group(1)), None, m.group(1)
    return None, None, None


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------------------
# Paths & panel schema
# ----------------------------------------------------------------------------

def data_root(script_file: str | None = None) -> str:
    """The shared data root (…/Open-Finance), matching the convention used by
    every scrape_*/panel_* script. Now delegates to the canonical
    ``utils/paths.py`` anchor; ``script_file`` is accepted for backward
    compatibility but ignored."""
    from .paths import OPEN_FINANCE
    return str(OPEN_FINANCE)


# Canonical LONG-format columns for every firm-disclosure scraper.
PANEL_COLUMNS = [
    "firm_key",        # stable key from the registry (e.g. 'nubank')
    "firm_name",       # disclosed institution / group name
    "cnpj_root",       # 8-digit CNPJ root of the lead institution (prudential conglomerate key)
    "segment",         # 'digital' | 'payment' | 'incumbent'
    "source",          # 'sec_edgar' | 'parent_filing' | 'incumbent_release' | 'bcb_*' | 'fgc' | 'cvm'
    "filing_form",     # '6-K','20-F','ITR','press_release', ...
    "period_year",
    "period_quarter",  # 1-4 or <NA>
    "period_label",    # e.g. '1Q2026'
    "geo_scope",       # 'brazil' | 'consolidated' | 'latam' | 'unknown'
    "metric",          # 'customers_total'|'customers_active'|'accounts'|'deposits'|'deposits_per_customer'
    "value",
    "unit",            # 'count' | 'BRL_thousands' | 'USD_thousands' | ...
    "currency",        # 'BRL'|'USD'|''
    "confidence",      # 'high'|'medium'|'low'
    "raw_context",     # short surrounding text for auditing
    "source_url",
    "accession",       # filing id where applicable
    "retrieved_at",
]


def new_record(**kw) -> dict:
    rec = {c: kw.get(c) for c in PANEL_COLUMNS}
    rec.setdefault("retrieved_at", now_iso())
    if rec["retrieved_at"] is None:
        rec["retrieved_at"] = now_iso()
    return rec


def write_panel(records: Iterable[dict], out_csv: str) -> int:
    """Merge new records into the existing CSV (read-modify-write), then write.

    Rows for firms present in `records` are replaced; all other firms' rows are
    preserved.  This means per-firm scraper runs accumulate correctly instead of
    overwriting each other.
    """
    import pandas as pd
    rows = list(records)
    new_df = pd.DataFrame(rows, columns=PANEL_COLUMNS) if rows else \
        pd.DataFrame(columns=PANEL_COLUMNS)

    # Merge with existing file: keep rows for firms NOT in this batch
    if os.path.isfile(out_csv) and new_df["firm_key"].notna().any():
        try:
            existing = pd.read_csv(out_csv)
            scraped_firms = set(new_df["firm_key"].dropna().unique())
            kept = existing[~existing["firm_key"].isin(scraped_firms)]
            new_df = pd.concat([kept, new_df], ignore_index=True)
        except Exception:  # noqa: BLE001 - if read fails, just use new records
            pass

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.csv as pa_csv
        pa_csv.write_csv(pa.Table.from_pandas(new_df, preserve_index=False), out_csv)
    except Exception:  # noqa: BLE001
        new_df.to_csv(out_csv, index=False)
    return len(new_df)
