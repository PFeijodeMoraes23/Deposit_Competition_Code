"""
scrape_19_openfinance_fees.py
==============================
Collect banking service-fee schedules from the Open Finance Brazil (OFB) public
open-data product APIs — no OAuth required, pure open data.

Why this beats Nakane et al.
-----------------------------
  • Listed prices (not accounting ratios): directly what each bank charges per
    service, in BRL, by account type (CONTA_CORRENTE / CONTA_PAGAMENTO_PRE_PAGA …)
  • Institution heterogeneity including digital banks: every OFB participant
    (Nubank, Inter, C6, Mercado Pago …) publishes this data.
  • Service-level granularity: ATM withdrawal, TED, maintenance, overdraft …
  • Customer distribution: each price tier comes with the share of customers
    paying it — so we can compute a weighted average price, not just a max.
  • Time variation: responses cached to disk tagged with scrape date; each run
    appends a new snapshot → free prospective historical archive.

Scope — the full open-data Products & Services family
-----------------------------------------------------
The OFB open data used to be one "products-services" API; it is now split into
per-product families. This collector is parameterized over them:

  accounts           deposit / payment accounts  →  fees.priorityServices/otherServices  (per-event BRL)
  creditcards        credit cards                →  fees.services (annuity …) + interest.rates/instalmentRates
  loans              personal/business loans     →  fees.services + interestRates
  financings         financings                  →  fees.services + interestRates
  invoicefinancings  invoice financings          →  fees.services + interestRates
  unarranged         unarranged overdraft        →  fees.services + interestRates

Two economically-distinct objects are parsed and TAGGED with `price_kind`:
  per_event_brl   a fee charged per event, in BRL   (comparable across banks; enters the fee measure)
  rate_pct        an interest-rate distribution, %  (a different object; kept separate)

IMPORTANT — temporal scope
---------------------------
The frequency-distribution values (price bands + customer shares) only exist
MONTHLY since JANUARY 2021 (BCB Instrução Normativa 32/2020, art. 6), published
on the 10th BUSINESS day referencing the previous month. This is a recent
short-panel / cross-sectional source — NOT the historical backbone. Pre-2021
history comes from scrape_20 (BCB DataVigencia listed prices) and scrape_18
(COSIF realized revenue). The response carries NO reference-period field, so
freshness must be inferred empirically (see diag_openfinance_freshness.py).

Data flow
---------
  1. Fetch participant list from the OFB Directory.
  2. Extract each institution's opendata-* API URLs from the registered
     ApiDiscoveryEndpoints, for the requested families.
  3. Concurrently fetch all URLs; parse fees + interest-rate distributions.
  4. Compute customer-weighted average price/rate from the quartile bands.
  5. Append to a long panel CSV tagged with today's scrape date.

Stack
-----
  httpx (async, HTTP/2)  +  diskcache (persistent cache)

Output files  (BCB/Tarifas/processed/)
---------------------------------------
  openfinance_fees_raw/YYYYMMDD_fees_raw.jsonl   raw rows this run (immutable archive)
  openfinance_fees_panel_long.csv                long panel (all runs, appended)
  openfinance_fees_panel_wide.csv                latest wide snapshot (per-event fees only)

Usage
-----
  python scrape_19_openfinance_fees.py                       # accounts only (default)
  python scrape_19_openfinance_fees.py --families accounts,creditcards,loans
  python scrape_19_openfinance_fees.py --families all
  python scrape_19_openfinance_fees.py --test 15             # first 15 endpoints
  python scrape_19_openfinance_fees.py --no-cache            # bypass disk cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from datetime import date
from pathlib import Path

# Enforce the project venv before importing third-party deps.
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import httpx
import pandas as pd

try:
    import diskcache
    _HAVE_DISKCACHE = True
except ImportError:
    _HAVE_DISKCACHE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from utils import paths
_REPO     = Path(__file__).resolve().parents[2]
OUT_DIR   = paths.TARIFAS_PROC
RAW_DIR   = OUT_DIR / "openfinance_fees_raw"
CACHE_DIR = paths.TARIFAS_CACHE / "of_opendata"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIRECTORY_URL  = "https://data.directory.openbankingbrasil.org.br/participants"

# The open-data product families we know how to parse. Keys are the canonical
# "product" tokens accepted by --families; values are the ApiFamilyType prefix
# (the part before the first '_') normalized to hyphen-free lowercase, so we can
# match the directory's family strings regardless of their exact hyphenation
# (e.g. "opendata-invoice-financings_personal-invoice-financings").
#   accounts           <- opendata-accounts_*
#   creditcards        <- opendata-creditcards_*
#   loans              <- opendata-loans_*
#   financings         <- opendata-financings_*
#   invoicefinancings  <- opendata-invoice-financings_*
#   unarranged         <- opendata-unarranged-accounts-overdraft_*
PRODUCT_BY_PREFIX = {
    "opendataaccounts":                    "accounts",
    "opendatacreditcards":                 "creditcards",
    "opendataloans":                       "loans",
    "opendatafinancings":                  "financings",
    "opendatainvoicefinancings":           "invoicefinancings",
    "opendataunarranged":                  "unarranged",
}
ALL_PRODUCTS = list(dict.fromkeys(PRODUCT_BY_PREFIX.values()))
DEFAULT_PRODUCTS = ["accounts"]

CONCURRENCY    = 20
TIMEOUT        = 25.0
MAX_RETRIES    = 3
TODAY          = date.today().isoformat()
TODAY_TAG      = TODAY.replace("-", "")


def _norm_token(s: str) -> str:
    """Lowercase and strip all non-alphanumerics (hyphens/underscores)."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def product_for_family(api_family: str) -> str | None:
    """Map an ApiFamilyType (e.g. 'opendata-loans_personal-loans') to a product token."""
    prefix = api_family.split("_", 1)[0]
    return PRODUCT_BY_PREFIX.get(_norm_token(prefix))


def customer_type_for_family(api_family: str) -> str:
    """PF / PJ / NA inferred from the resource segment of the ApiFamilyType."""
    resource = api_family.split("_", 1)[1] if "_" in api_family else ""
    r = resource.lower()
    if "personal" in r:
        return "PF"
    if "business" in r:
        return "PJ"
    return "NA"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cache: "diskcache.Cache | None" = None


def get_cache(use_cache: bool) -> "diskcache.Cache | None":
    global _cache
    if not use_cache or not _HAVE_DISKCACHE:
        return None
    if _cache is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache = diskcache.Cache(str(CACHE_DIR), size_limit=2 * 1024**3)
    return _cache


# ---------------------------------------------------------------------------
# Directory parsing
# ---------------------------------------------------------------------------

def extract_endpoints(participants: list[dict], products: list[str]) -> list[dict]:
    """
    Walk the participant directory and return all open-data endpoints for the
    requested product families.

    Returns list of:
      { org_name, cnpj8, customer_type ('PF'/'PJ'/'NA'), api_family, product, url }
    """
    wanted = set(products)
    endpoints: list[dict] = []
    seen: set[tuple[str, str]] = set()
    unknown_families: set[str] = set()

    for p in participants:
        if p.get("Status", "Active") not in ("Active", ""):
            continue

        name  = p.get("OrganisationName") or p.get("RegisteredName") or ""
        cnpj8 = _clean_cnpj(p.get("RegistrationId", "") or p.get("RegistrationNumber", ""))

        for auth in p.get("AuthorisationServers", []):
            for res in auth.get("ApiResources", []):
                fam = res.get("ApiFamilyType", "")
                if not fam.lower().startswith("opendata"):
                    continue

                product = product_for_family(fam)
                if product is None:
                    unknown_families.add(fam)
                    continue
                if product not in wanted:
                    continue

                ctype = customer_type_for_family(fam)

                for disc in res.get("ApiDiscoveryEndpoints") or []:
                    url = disc.get("ApiEndpoint", "").strip()
                    if not url:
                        continue
                    key = (url, fam)
                    if key in seen:
                        continue
                    seen.add(key)
                    endpoints.append({
                        "org_name":      name,
                        "cnpj8":         cnpj8,
                        "customer_type": ctype,
                        "api_family":    fam,
                        "product":       product,
                        "url":           url,
                    })

    by_product: dict[str, int] = {}
    for e in endpoints:
        by_product[e["product"]] = by_product.get(e["product"], 0) + 1
    log.info("Directory: %d open-data endpoints for %s  (%s)",
             len(endpoints), sorted(wanted),
             ", ".join(f"{k}={v}" for k, v in sorted(by_product.items())))
    if unknown_families:
        log.info("Directory: %d unrecognized opendata family types skipped: %s",
                 len(unknown_families), sorted(unknown_families))
    return endpoints


def _clean_cnpj(raw: str) -> str:
    d = re.sub(r"\D", "", str(raw))
    return d[:8].zfill(8) if d else ""


# ---------------------------------------------------------------------------
# Response parser — one unified parser over the open-data family shapes
# ---------------------------------------------------------------------------

# Fee-service lists can appear under any of these keys, depending on family.
_FEE_LIST_KEYS = ("priorityServices", "otherServices", "services")

# The four quartile bands, in order.
_FAIXAS = ("1_FAIXA", "2_FAIXA", "3_FAIXA", "4_FAIXA")


def parse_response(url: str, payload: dict, ep: dict) -> list[dict]:
    """
    Parse an open-data response into unified fee/rate rows.

    Emits two kinds of rows, tagged by `price_kind`:
      per_event_brl  from fees.{priorityServices,otherServices,services}[]
                     prices[{interval, value, currency, customers.rate}] + minimum/maximum
      rate_pct       from interestRates[] (loans/financings/…) OR
                     interest.{rates,instalmentRates}[] (credit cards)
                     applications[{interval, indexer.rate, customers.rate}] + minimum/maximumRate

    All rows share one schema so the long panel stays a single table.
    """
    org_name      = ep["org_name"]
    cnpj8         = ep["cnpj8"]
    customer_type = ep["customer_type"]
    api_family    = ep["api_family"]
    product       = ep["product"]

    rows: list[dict] = []
    items = payload.get("data", [])
    if isinstance(items, dict):
        items = [items]

    for item in items:
        if not isinstance(item, dict):
            continue
        participant = item.get("participant", {}) or {}
        resp_cnpj8  = re.sub(r"\D", "", participant.get("cnpjNumber", ""))[:8].zfill(8)
        resolved_cnpj = resp_cnpj8 if resp_cnpj8 != "00000000" else cnpj8
        resp_name   = participant.get("name", org_name)
        resp_brand  = participant.get("brand", "")
        account_type = item.get("type", "") or item.get("name", "")

        base = {
            "org_name":      org_name,
            "cnpj8":         resolved_cnpj,
            "brand":         resp_brand,
            "company_name":  resp_name,
            "customer_type": customer_type,
            "api_family":    api_family,
            "product":       product,
            "account_type":  account_type,
            "source_url":    url,
        }

        # --- per-event BRL fees -------------------------------------------
        fees_block = item.get("fees", {}) or {}
        for key in _FEE_LIST_KEYS:
            for svc in (fees_block.get(key) or []):
                if not isinstance(svc, dict):
                    continue
                row = _fee_row(svc, base)
                if row is not None:
                    rows.append(row)

        # --- interest-rate distributions ----------------------------------
        # Shape A: top-level interestRates[] (loans/financings/invoicefinancings/unarranged)
        for ir in (item.get("interestRates") or []):
            if isinstance(ir, dict):
                rows.append(_rate_row(ir, base, kind_label="INTEREST"))
        # Shape B: nested interest.{rates,instalmentRates}[] (credit cards)
        interest = item.get("interest", {}) or {}
        for ir in (interest.get("rates") or []):
            if isinstance(ir, dict):
                rows.append(_rate_row(ir, base, kind_label="INTEREST_ROTATIVO"))
        for ir in (interest.get("instalmentRates") or []):
            if isinstance(ir, dict):
                rows.append(_rate_row(ir, base, kind_label="INTEREST_PARCELADO"))

    return [r for r in rows if r is not None]


def _fee_row(svc: dict, base: dict) -> dict | None:
    name    = svc.get("name", "")
    code    = svc.get("code", "")
    if not (name or code):
        return None
    prices  = svc.get("prices", []) or []
    minimum = svc.get("minimum", {}) or {}
    maximum = svc.get("maximum", {}) or {}

    bands = _band_values(prices, value_getter=lambda pt: pt.get("value"))
    row = {
        **base,
        "price_kind":    "per_event_brl",
        "service_name":  name,
        "service_code":  str(code),
        "trigger_info":  svc.get("chargingTriggerInfo", ""),
        "price_minimum": _to_float(minimum.get("value")),
        "price_maximum": _to_float(maximum.get("value")),
    }
    row.update(bands)
    return row


def _rate_row(ir: dict, base: dict, kind_label: str) -> dict:
    indexer = ir.get("referentialRateIndexer") or ir.get("indexer") or ""
    code    = f"{kind_label}_{indexer}" if indexer else kind_label
    apps    = ir.get("applications", []) or []

    # The per-band rate lives under applications[].indexer.rate; the customer
    # share under applications[].customers.rate.
    bands = _band_values(
        apps,
        value_getter=lambda a: (a.get("indexer") or {}).get("rate") if isinstance(a.get("indexer"), dict) else a.get("rate"),
    )
    row = {
        **base,
        "price_kind":    "rate_pct",
        "service_name":  str(indexer) or kind_label,
        "service_code":  str(code),
        "trigger_info":  "",
        "price_minimum": _to_float(ir.get("minimumRate")),
        "price_maximum": _to_float(ir.get("maximumRate")),
    }
    row.update(bands)
    return row


def _band_values(items: list[dict], value_getter) -> dict:
    """
    Build the band-level columns + summary stats from a 4-quartile distribution.

    Each item has an `interval` (1_FAIXA..4_FAIXA), a value (via value_getter),
    and customers.rate (share of customers in that band).
    Returns price_weighted_avg, share_zero, price_band_spread, and per-band
    price_<FAIXA> / cust_rate_<FAIXA> columns.
    """
    q_vals: dict = {f"price_{iv}": None for iv in _FAIXAS}
    q_vals.update({f"cust_rate_{iv}": None for iv in _FAIXAS})

    total_w = 0.0
    total_wv = 0.0
    share_zero = 0.0
    for pt in items:
        if not isinstance(pt, dict):
            continue
        iv = pt.get("interval", "")
        v  = _to_float(value_getter(pt))
        r  = _to_float((pt.get("customers") or {}).get("rate"))
        if iv in _FAIXAS:
            q_vals[f"price_{iv}"] = v
            q_vals[f"cust_rate_{iv}"] = r
        if v is not None and r is not None:
            total_w  += r
            total_wv += v * r
            if v == 0.0:
                share_zero += r

    q_vals["price_weighted_avg"] = (total_wv / total_w) if total_w > 0 else None
    q_vals["share_zero"] = share_zero if total_w > 0 else None
    p1 = q_vals["price_1_FAIXA"]
    p4 = q_vals["price_4_FAIXA"]
    q_vals["price_band_spread"] = (p4 - p1) if (p1 is not None and p4 is not None) else None
    return q_vals


def _to_float(v: "str | None") -> "float | None":
    if v is None or str(v).strip() in ("", "nan"):
        return None
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------

async def fetch_endpoint(
    client: httpx.AsyncClient,
    ep: dict,
    semaphore: asyncio.Semaphore,
    cache: "diskcache.Cache | None",
) -> list[dict]:
    url = ep["url"]

    if cache is not None and url in cache:
        payload = cache[url]
        if payload:
            return _stamp(parse_response(url, payload, ep))

    async with semaphore:
        payload = await _get_json(client, url)

    if payload is None:
        return []

    if cache is not None:
        cache.set(url, payload, expire=86400 * 30)   # 30-day TTL

    return _stamp(parse_response(url, payload, ep))


def _stamp(rows: list[dict]) -> list[dict]:
    for r in rows:
        r["data_coleta"] = TODAY
    return rows


async def _get_json(client: httpx.AsyncClient, url: str) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "")
                text = resp.text.strip()
                if "json" in ct or text.startswith(("{", "[")):
                    return resp.json()
                return None
            if resp.status_code in (404, 405, 501, 400):
                return None   # endpoint doesn't exist — no retry
            await asyncio.sleep(2 ** attempt)         # 429 / 5xx → backoff
        except (httpx.TimeoutException, httpx.ConnectError,
                httpx.RemoteProtocolError, httpx.ReadError):
            if attempt == MAX_RETRIES - 1:
                return None
            await asyncio.sleep(2 ** attempt)
        except Exception:
            return None
    return None


async def run_async(endpoints: list[dict], use_cache: bool) -> list[dict]:
    cache     = get_cache(use_cache)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    limits    = httpx.Limits(max_keepalive_connections=30, max_connections=60)

    all_rows: list[dict] = []
    n = len(endpoints)

    async with httpx.AsyncClient(
        limits=limits,
        verify=False,
        follow_redirects=True,
        headers={"Accept": "application/json", "User-Agent": "research-bot/1.0"},
    ) as client:
        tasks = [fetch_endpoint(client, ep, semaphore, cache) for ep in endpoints]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            rows = await coro
            all_rows.extend(rows)
            if i % 20 == 0 or i == n:
                log.info("  [%d/%d]  rows so far: %d", i, n, len(all_rows))

    if cache is not None:
        cache.close()
    return all_rows


# ---------------------------------------------------------------------------
# Panel building
# ---------------------------------------------------------------------------

# Long-panel column order (stable, so appended snapshots stay aligned).
LONG_COLS = [
    "data_coleta", "org_name", "cnpj8", "brand", "company_name",
    "customer_type", "api_family", "product", "account_type",
    "price_kind", "service_name", "service_code", "trigger_info",
    "price_minimum", "price_maximum", "price_weighted_avg",
    "share_zero", "price_band_spread",
    "price_1_FAIXA", "price_2_FAIXA", "price_3_FAIXA", "price_4_FAIXA",
    "cust_rate_1_FAIXA", "cust_rate_2_FAIXA", "cust_rate_3_FAIXA", "cust_rate_4_FAIXA",
    "source_url",
]

DEDUP_KEY = ["cnpj8", "customer_type", "api_family", "product",
             "account_type", "service_code", "price_kind", "data_coleta"]


def build_panels(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    long = pd.DataFrame(rows)
    long = long[long["service_name"].astype(str).str.strip() != ""].copy()
    # Stable column order (tolerate any missing columns).
    ordered = [c for c in LONG_COLS if c in long.columns]
    extra   = [c for c in long.columns if c not in ordered]
    long = long[ordered + extra]

    # Wide: one column per (product, service_code) for PER-EVENT fees only —
    # interest-rate rows are a different object and must not be mixed in.
    fees = long[long["price_kind"] == "per_event_brl"].copy()
    try:
        fees["svc_key"] = fees["product"] + "_" + fees["service_code"].astype(str)
        wide = fees.pivot_table(
            index=["cnpj8", "org_name", "customer_type", "account_type", "data_coleta"],
            columns="svc_key",
            values="price_weighted_avg",
            aggfunc="mean",
        ).reset_index()
        wide.columns.name = None
        idx_cols = ("cnpj8", "org_name", "customer_type", "account_type", "data_coleta")
        wide = wide.rename(columns={c: f"fee_{c}" for c in wide.columns if c not in idx_cols})
    except Exception as exc:
        log.warning("Wide pivot failed: %s", exc)
        wide = pd.DataFrame()

    return long, wide


def append_long_panel(new_long: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 1000:
        existing = pd.read_csv(path, low_memory=False, dtype={"cnpj8": str})
        combined = pd.concat([existing, new_long], ignore_index=True)
    else:
        combined = new_long.copy()

    dedup = [c for c in DEDUP_KEY if c in combined.columns]
    combined = combined.drop_duplicates(subset=dedup, keep="last")
    combined.to_csv(path, index=False)
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def resolve_products(families_arg: str) -> list[str]:
    if not families_arg or families_arg.strip().lower() == "all":
        return ALL_PRODUCTS if (families_arg or "").strip().lower() == "all" else DEFAULT_PRODUCTS
    tokens = [_norm_token(t) for t in families_arg.split(",") if t.strip()]
    valid = {_norm_token(p): p for p in ALL_PRODUCTS}
    out: list[str] = []
    for t in tokens:
        if t in valid and valid[t] not in out:
            out.append(valid[t])
        elif t not in valid:
            log.warning("Unknown family token '%s' (valid: %s)", t, ALL_PRODUCTS)
    return out or DEFAULT_PRODUCTS


def main(test_n: int | None = None, use_cache: bool = True,
         families: str = "accounts") -> None:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    products = resolve_products(families)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: directory
    log.info("=== Step 1: fetching OFB participant directory ===")
    log.info("Families requested: %s", products)
    try:
        resp = httpx.get(DIRECTORY_URL, timeout=60, follow_redirects=True, verify=False)
        resp.raise_for_status()
        participants = resp.json()
    except Exception as exc:
        log.error("Failed to fetch directory: %s", exc)
        return
    log.info("Directory: %d participants", len(participants))

    # Step 2: extract endpoints
    endpoints = extract_endpoints(participants, products)
    if not endpoints:
        log.error("No open-data endpoints found for families %s.", products)
        return

    if test_n is not None:
        endpoints = endpoints[:test_n]
        log.info("TEST MODE — %d endpoints", len(endpoints))

    # Step 3: fetch
    log.info("=== Step 3: fetching %d endpoints ===", len(endpoints))
    rows = asyncio.run(run_async(endpoints, use_cache=use_cache))
    log.info("Collected %d rows (%d per-event, %d rate)",
             len(rows),
             sum(1 for r in rows if r.get("price_kind") == "per_event_brl"),
             sum(1 for r in rows if r.get("price_kind") == "rate_pct"))

    if not rows:
        log.warning("No rows — check endpoint URLs above.")
        return

    # Step 4: save raw
    raw_path = RAW_DIR / f"{TODAY_TAG}_fees_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    log.info("Raw JSONL: %d rows → %s", len(rows), raw_path.name)

    # Step 5: panels
    long_df, wide_df = build_panels(rows)

    if not long_df.empty:
        long_path = OUT_DIR / "openfinance_fees_panel_long.csv"
        combined  = append_long_panel(long_df, long_path)
        log.info("Long panel: %d rows → %s", len(combined), long_path.name)

        try:
            import polars as pl
            pl.from_pandas(combined).write_parquet(
                str(OUT_DIR / "openfinance_fees_panel_long.parquet"),
                compression="zstd",
            )
            log.info("Long parquet saved.")
        except ImportError:
            pass

    if not wide_df.empty:
        wide_path = OUT_DIR / "openfinance_fees_panel_wide.csv"
        wide_df.to_csv(wide_path, index=False)
        log.info("Wide panel: %d rows × %d cols → %s",
                 len(wide_df), len(wide_df.columns), wide_path.name)

    log.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open Finance Brazil open-data fee collector")
    parser.add_argument("--families", type=str, default="accounts",
                        help="Comma-separated product families, or 'all'. "
                             f"Valid: {','.join(ALL_PRODUCTS)}. Default: accounts")
    parser.add_argument("--test", type=int, default=None, metavar="N",
                        help="Collect only first N endpoints")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass disk cache")
    args = parser.parse_args()
    main(test_n=args.test, use_cache=not args.no_cache, families=args.families)
