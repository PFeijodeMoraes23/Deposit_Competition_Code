"""
scrape_19_openfinance_fees.py
==============================
Collect banking service-fee schedules from the Open Finance Brazil (OFB) public
product APIs — no OAuth required, pure open data.

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
    appends a new snapshot → free historical archive.

Data flow
---------
  1. Fetch participant list from the OFB Directory.
  2. Extract each institution's opendata-accounts_* API URLs directly from
     the registered ApiDiscoveryEndpoints (family: opendata-accounts_personal-
     accounts / opendata-accounts_business-accounts).
  3. Concurrently fetch all URLs; parse fees from data[].fees.priorityServices[].
  4. Compute weighted average price from quartile-interval distributions.
  5. Append to a long panel CSV tagged with today's scrape date.

Stack
-----
  httpx (async, HTTP/2)  +  tenacity (retry)  +  diskcache (persistent cache)

Output files  (BCB/Tarifas/processed/)
---------------------------------------
  openfinance_fees_raw/YYYYMMDD_fees_raw.jsonl   raw rows this run
  openfinance_fees_panel_long.csv                long panel (all runs, appended)
  openfinance_fees_panel_wide.csv                latest wide snapshot

Usage
-----
  python scrape_19_openfinance_fees.py
  python scrape_19_openfinance_fees.py --test 15   # first 15 endpoints
  python scrape_19_openfinance_fees.py --no-cache  # bypass disk cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import date
from pathlib import Path

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
OPENDATA_FAM_PF = "opendata-accounts_personal-accounts"
OPENDATA_FAM_PJ = "opendata-accounts_business-accounts"

CONCURRENCY    = 20
TIMEOUT        = 25.0
MAX_RETRIES    = 3
TODAY          = date.today().isoformat()
TODAY_TAG      = TODAY.replace("-", "")


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

def extract_opendata_endpoints(participants: list[dict]) -> list[dict]:
    """
    Walk the participant directory and return all open-data account endpoints.

    Returns list of:
      { org_name, cnpj, customer_type ('PF'/'PJ'), url }
    """
    endpoints: list[dict] = []
    seen: set[str] = set()

    for p in participants:
        if p.get("Status", "Active") not in ("Active", ""):
            continue

        name  = p.get("OrganisationName") or p.get("RegisteredName") or ""
        cnpj8 = _clean_cnpj(p.get("RegistrationId", "") or p.get("RegistrationNumber", ""))

        for auth in p.get("AuthorisationServers", []):
            for res in auth.get("ApiResources", []):
                fam = res.get("ApiFamilyType", "")
                if fam not in (OPENDATA_FAM_PF, OPENDATA_FAM_PJ):
                    continue

                ctype = "PF" if fam == OPENDATA_FAM_PF else "PJ"

                for disc in res.get("ApiDiscoveryEndpoints") or []:
                    url = disc.get("ApiEndpoint", "").strip()
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    endpoints.append({
                        "org_name":      name,
                        "cnpj8":         cnpj8,
                        "customer_type": ctype,
                        "url":           url,
                    })

    log.info("Directory: %d open-data account endpoints (%d PF, %d PJ)",
             len(endpoints),
             sum(1 for e in endpoints if e["customer_type"] == "PF"),
             sum(1 for e in endpoints if e["customer_type"] == "PJ"))
    return endpoints


def _clean_cnpj(raw: str) -> str:
    import re
    d = re.sub(r"\D", "", str(raw))
    return d[:8].zfill(8) if d else ""


# ---------------------------------------------------------------------------
# Response parser — handles the v1 opendata-accounts structure
# ---------------------------------------------------------------------------

def parse_opendata_response(url: str, payload: dict,
                             org_name: str, cnpj8: str,
                             customer_type: str) -> list[dict]:
    """
    Parse the open-data personal/business accounts response.

    Structure:
      data[] →
        participant.cnpjNumber
        type  (account type: CONTA_CORRENTE, CONTA_PAGAMENTO_PRE_PAGA, …)
        fees.priorityServices[] →
          name, code, chargingTriggerInfo
          prices[{interval, value, currency, customers.rate}]
          minimum.value / maximum.value
    """
    rows: list[dict] = []
    items = payload.get("data", [])
    if isinstance(items, dict):
        items = [items]

    for item in items:
        participant = item.get("participant", {})
        # Prefer the CNPJ from the response (more reliable than directory)
        resp_cnpj_full = participant.get("cnpjNumber", "")
        import re
        resp_cnpj8 = re.sub(r"\D", "", resp_cnpj_full)[:8].zfill(8)
        resolved_cnpj = resp_cnpj8 or cnpj8

        resp_name    = participant.get("name", org_name)
        resp_brand   = participant.get("brand", "")
        account_type = item.get("type", "")

        fees_block       = item.get("fees", {})
        priority_svcs    = fees_block.get("priorityServices", [])
        other_svcs       = fees_block.get("otherServices", [])

        for svc in (priority_svcs + other_svcs):
            name    = svc.get("name", "")
            code    = svc.get("code", "")
            trigger = svc.get("chargingTriggerInfo", "")

            prices  = svc.get("prices", [])
            minimum = svc.get("minimum", {})
            maximum = svc.get("maximum", {})

            price_min = _to_float(minimum.get("value"))
            price_max = _to_float(maximum.get("value"))

            # Weighted average price from quartile distribution
            wtd_avg = _weighted_avg(prices)

            # Also store individual quartile values (useful for distributional analysis)
            q_vals: dict[str, float | None] = {}
            for pt in prices:
                iv = pt.get("interval", "")
                q_vals[f"price_{iv}"] = _to_float(pt.get("value"))
                q_vals[f"cust_rate_{iv}"] = _to_float(
                    (pt.get("customers") or {}).get("rate")
                )

            row: dict = {
                "data_coleta":   TODAY,
                "org_name":      org_name,
                "cnpj8":         resolved_cnpj,
                "brand":         resp_brand,
                "company_name":  resp_name,
                "customer_type": customer_type,
                "account_type":  account_type,
                "service_name":  name,
                "service_code":  code,
                "trigger_info":  trigger,
                "price_minimum": price_min,
                "price_maximum": price_max,
                "price_weighted_avg": wtd_avg,
                "source_url":    url,
            }
            row.update(q_vals)
            rows.append(row)

    return rows


def _to_float(v: "str | None") -> "float | None":
    if v is None or str(v).strip() in ("", "nan"):
        return None
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


def _weighted_avg(prices: list[dict]) -> "float | None":
    """
    Compute the customer-weighted average price across quartile intervals.
    E.g. 3% pay R$0, 97% pay R$6.50 → weighted avg = R$6.30.
    """
    total_w = 0.0
    total_wv = 0.0
    for pt in prices:
        v = _to_float(pt.get("value"))
        r = _to_float((pt.get("customers") or {}).get("rate"))
        if v is not None and r is not None:
            total_w  += r
            total_wv += v * r
    if total_w > 0:
        return total_wv / total_w
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
    url           = ep["url"]
    org_name      = ep["org_name"]
    cnpj8         = ep["cnpj8"]
    customer_type = ep["customer_type"]

    # Disk cache
    if cache is not None and url in cache:
        payload = cache[url]
        if payload:
            return parse_opendata_response(url, payload, org_name, cnpj8, customer_type)

    async with semaphore:
        payload = await _get_json(client, url)

    if payload is None:
        return []

    if cache is not None:
        cache.set(url, payload, expire=86400 * 30)   # 30-day TTL

    return parse_opendata_response(url, payload, org_name, cnpj8, customer_type)


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
            # 429 / 5xx → exponential backoff
            await asyncio.sleep(2 ** attempt)
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

def build_panels(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    long = pd.DataFrame(rows)
    long = long[long["service_name"].str.strip() != ""].copy()

    # Wide: one column per service_code, value = weighted_avg price
    try:
        wide = long.pivot_table(
            index=["cnpj8", "org_name", "customer_type", "account_type", "data_coleta"],
            columns="service_code",
            values="price_weighted_avg",
            aggfunc="mean",
        ).reset_index()
        wide.columns.name = None
        svc_cols = [c for c in wide.columns
                    if c not in ("cnpj8", "org_name", "customer_type",
                                 "account_type", "data_coleta")]
        wide = wide.rename(columns={c: f"fee_{c}" for c in svc_cols})
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

    dedup = ["cnpj8", "customer_type", "account_type", "service_code", "data_coleta"]
    dedup = [c for c in dedup if c in combined.columns]
    combined = combined.drop_duplicates(subset=dedup, keep="last")
    combined.to_csv(path, index=False)
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(test_n: int | None = None, use_cache: bool = True) -> None:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: directory
    log.info("=== Step 1: fetching OFB participant directory ===")
    try:
        resp = httpx.get(DIRECTORY_URL, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        participants = resp.json()
    except Exception as exc:
        log.error("Failed to fetch directory: %s", exc)
        return
    log.info("Directory: %d participants", len(participants))

    # Step 2: extract endpoints
    endpoints = extract_opendata_endpoints(participants)
    if not endpoints:
        log.error("No open-data account endpoints found in directory.")
        return

    if test_n is not None:
        endpoints = endpoints[:test_n]
        log.info("TEST MODE — %d endpoints", len(endpoints))

    # Step 3: fetch
    log.info("=== Step 3: fetching %d endpoints ===", len(endpoints))
    rows = asyncio.run(run_async(endpoints, use_cache=use_cache))
    log.info("Collected %d fee rows", len(rows))

    if not rows:
        log.warning("No fee rows — check endpoint URLs above.")
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

        # Polars parquet if available
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
    parser = argparse.ArgumentParser(description="Open Finance Brazil fee collector")
    parser.add_argument("--test", type=int, default=None, metavar="N",
                        help="Collect only first N endpoints")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass disk cache")
    args = parser.parse_args()
    main(test_n=args.test, use_cache=not args.no_cache)
