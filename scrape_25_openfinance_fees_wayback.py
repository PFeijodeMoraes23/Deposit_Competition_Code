"""
scrape_openfinance_fees_wayback.py
=====================================
Assess (and, if worthwhile, execute) a HISTORICAL reconstruction of Open Finance
Brasil open-data fees from the Wayback Machine.

Why this is needed and why it's hard
------------------------------------
The OFB open-data APIs serve only the CURRENT snapshot; the monthly frequency data
exists since 2021 but is overwritten each month. The only retrospective source is the
web archive. But unlike the PIX-roster case (scrape_23), there is NO single canonical
URL: every bank exposes its own endpoint host, and both the directory and the per-bank
hosts changed over time. And JSON API endpoints are archived far less than HTML pages —
a spot check found the participants directory IS archived but Nubank's live fee endpoint
has ZERO Wayback captures. So this is expected to be sparse, big-bank-biased and
irregular. Hence the design is FEASIBILITY-GATED: measure coverage first, only backfill
if it clears a threshold.

VERDICT (2026-07-23): checked three independent archives — Internet Archive/Wayback
(this module), Common Crawl, and arquivo.pt. Per-bank fee endpoints are unarchived in
ALL THREE (only the directory is sparsely captured); the gate FAILS. Pre-collection OFB
actual prices are unrecoverable — use BCB DataVigencia (scrape_20) for listed-price
history. Kept for periodic re-checking, not because a backfill is expected.

Pipeline
--------
  Phase A (default) — FEASIBILITY PROBE
    1. Current endpoints  : reuse scrape_19.extract_endpoints on today's directory.
    2. (optional) Historical endpoints: CDX-enumerate archived snapshots of the
       directory URL, parse each with scrape_19.extract_endpoints, union in the
       endpoints as they existed at each capture (--historical-endpoints).
    3. CDX-probe each endpoint URL: count captures & distinct months since 2021.
    4. Emit openfinance_wayback_coverage.csv + a GO/NO-GO recommendation.

  Phase B (opt-in, --backfill) — RECONSTRUCTION  [only if the gate passes]
    5. For endpoints with captures, fetch the archived JSON via the raw-bytes
       Wayback URL (…/{ts}id_/{url}), parse with scrape_19.parse_response, tag
       data_coleta = capture date, and append to openfinance_fees_panel_long.csv
       (same dedup key as the live scraper).

Reuses the CDX request pattern from scrape_deposit_rate_cdx.py / scrape_pix_roster.py
(no-retry, per-endpoint resumable cache, politeness sleep, soft-ban handling).

Usage
-----
  python scrape_openfinance_fees_wayback.py                      # probe, current endpoints
  python scrape_openfinance_fees_wayback.py --families all
  python scrape_openfinance_fees_wayback.py --historical-endpoints
  python scrape_openfinance_fees_wayback.py --backfill           # probe + reconstruct (gated)
  python scrape_openfinance_fees_wayback.py --backfill --force-backfill   # ignore the gate
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlencode

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from utils import paths
import scrape_openfinance_fees as s19

CDX_URL   = "https://web.archive.org/cdx/search/cdx"     # https (port 443) — port 80 gets refused
WB_RAW    = "https://web.archive.org/web/{ts}id_/{url}"  # id_ = original bytes, no IA rewrite
FROM_YEAR = 2021                                          # OFB frequency data starts 2021

OUT_COVERAGE = paths.TARIFAS_PROC / "openfinance_wayback_coverage.csv"
CDX_CACHE    = paths.TARIFAS_CACHE / "wayback_cdx"

# --- Feasibility gate ------------------------------------------------------
# Backfill is only worthwhile if a meaningful number of banks have several monthly
# captures. Conservative defaults (env-overridable).
GATE_MIN_ENDPOINTS = 5    # >= this many endpoints with any capture
GATE_MIN_MONTHS    = 6    # ... and >= this many distinct months among them

# Politeness / ban handling (Wayback bans aggressively).
SLEEP_BETWEEN = 1.5
SOFTBAN_SLEEP = 60
MAX_SOFTBAN_RETRY = 3


def _slug(url: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_")[:150]


# ---------------------------------------------------------------------------
# CDX probe (one endpoint)
# ---------------------------------------------------------------------------

def cdx_captures(url: str, use_cache: bool = True) -> list[dict] | None:
    """
    Return the archived captures of `url` (prefix match) since FROM_YEAR, collapsed
    to one per month. Returns list of {timestamp, original} (possibly empty), or None
    on hard failure (so the caller can leave it uncached and retry later).
    """
    CDX_CACHE.mkdir(parents=True, exist_ok=True)
    cache_f = CDX_CACHE / f"{_slug(url)}.json"
    if use_cache and cache_f.exists():
        try:
            return json.loads(cache_f.read_text(encoding="utf-8"))
        except Exception:
            pass

    params = {
        "url": url,
        "matchType": "prefix",              # catch ?page= / trailing-slash variants
        "output": "json",
        "fl": "timestamp,original,statuscode",
        "filter": "statuscode:200",
        "collapse": "timestamp:6",          # one capture per YYYYMM
        "from": f"{FROM_YEAR}0101",
        "limit": 10000,
    }
    req = f"{CDX_URL}?{urlencode(params)}"

    for attempt in range(MAX_SOFTBAN_RETRY):
        try:
            r = requests.get(req, timeout=60)
        except Exception as e:
            # Connection refused / reset / timeout: Wayback throttling. Back off and
            # retry (leaving it uncached on final failure, so a later run tries again).
            if attempt < MAX_SOFTBAN_RETRY - 1:
                log.warning("  CDX connection error (%s) — backing off %ds", type(e).__name__, SOFTBAN_SLEEP)
                time.sleep(SOFTBAN_SLEEP)
                continue
            log.warning("  CDX request failed (%s): %s", url, e)
            return None
        if r.status_code == 200:
            try:
                data = r.json()
            except Exception:
                return None
            rows = data[1:] if data and len(data) > 1 else []
            out = [{"timestamp": ts, "original": orig} for ts, orig, _ in rows]
            if use_cache:
                cache_f.write_text(json.dumps(out), encoding="utf-8")
            return out
        if r.status_code in (429, 503):
            log.warning("  CDX %d (soft-ban?) — sleeping %ds", r.status_code, SOFTBAN_SLEEP)
            time.sleep(SOFTBAN_SLEEP)
            continue
        log.warning("  CDX returned %d for %s — skipping", r.status_code, url)
        return None
    return None


# ---------------------------------------------------------------------------
# Endpoint discovery
# ---------------------------------------------------------------------------

def current_endpoints(products: list[str]) -> list[dict]:
    r = requests.get(s19.DIRECTORY_URL, timeout=60, verify=False)
    r.raise_for_status()
    return s19.extract_endpoints(r.json(), products)


def historical_endpoints(products: list[str]) -> list[dict]:
    """CDX-enumerate directory snapshots, parse each, union endpoints over time."""
    caps = cdx_captures(s19.DIRECTORY_URL)
    if not caps:
        log.warning("No archived directory snapshots found — historical endpoints unavailable.")
        return []
    log.info("Directory has %d archived monthly snapshots; parsing for endpoints…", len(caps))
    seen: dict[tuple, dict] = {}
    for cap in caps:
        raw = WB_RAW.format(ts=cap["timestamp"], url=s19.DIRECTORY_URL)
        try:
            r = requests.get(raw, timeout=90)
            time.sleep(SLEEP_BETWEEN)
            if r.status_code != 200:
                continue
            parts = r.json()
        except Exception:
            continue
        for ep in s19.extract_endpoints(parts, products):
            seen.setdefault((ep["url"], ep["api_family"]), ep)
    log.info("Historical endpoint union: %d distinct endpoints", len(seen))
    return list(seen.values())


# ---------------------------------------------------------------------------
# Phase A — feasibility probe
# ---------------------------------------------------------------------------

def probe(endpoints: list[dict], use_cache: bool = True) -> "pd.DataFrame":
    import pandas as pd
    rows = []
    n = len(endpoints)
    for i, ep in enumerate(endpoints, 1):
        caps = cdx_captures(ep["url"], use_cache=use_cache)
        time.sleep(SLEEP_BETWEEN)
        if caps is None:
            n_caps, months, first, last = -1, 0, "", ""     # -1 = query failed
        else:
            ts = sorted(c["timestamp"] for c in caps)
            months = len({t[:6] for t in ts})
            n_caps = len(ts)
            first = ts[0] if ts else ""
            last = ts[-1] if ts else ""
        rows.append({
            "org_name": ep.get("org_name", ""), "cnpj8": ep.get("cnpj8", ""),
            "product": ep.get("product", ""), "customer_type": ep.get("customer_type", ""),
            "endpoint": ep["url"], "n_captures": n_caps,
            "months_covered": months, "first_capture": first, "last_capture": last,
        })
        if i % 20 == 0 or i == n:
            log.info("  probed [%d/%d]", i, n)
    return pd.DataFrame(rows)


def gate(cov: "pd.DataFrame") -> bool:
    with_caps = cov[cov["n_captures"] > 0]
    n_ep = len(with_caps)
    n_months = int(with_caps["months_covered"].sum())
    total_months = int(with_caps["months_covered"].max()) if n_ep else 0
    failed = int((cov["n_captures"] < 0).sum())
    log.info("── Wayback coverage summary ──")
    log.info("  endpoints probed        : %d", len(cov))
    log.info("  endpoints with captures : %d", n_ep)
    log.info("  CDX query failures      : %d", failed)
    log.info("  max months on one endpoint: %d", total_months)
    log.info("  sum of monthly captures : %d", n_months)
    if n_ep:
        top = with_caps.sort_values("months_covered", ascending=False).head(10)
        log.info("  top endpoints by months covered:\n%s",
                 top[["org_name", "product", "months_covered", "first_capture",
                      "last_capture"]].to_string(index=False))
    passed = (n_ep >= GATE_MIN_ENDPOINTS) and (n_months >= GATE_MIN_MONTHS)
    log.info("  GATE (>=%d endpoints AND >=%d monthly captures): %s",
             GATE_MIN_ENDPOINTS, GATE_MIN_MONTHS, "PASS → backfill worthwhile"
             if passed else "FAIL → prospective + BCB DataVigencia remain the history sources")
    return passed


# ---------------------------------------------------------------------------
# Phase B — reconstruction (opt-in, gated)
# ---------------------------------------------------------------------------

def backfill(endpoints: list[dict], cov: "pd.DataFrame") -> None:
    import pandas as pd
    have = set(cov[cov["n_captures"] > 0]["endpoint"])
    todo = [ep for ep in endpoints if ep["url"] in have]
    log.info("Backfilling %d endpoints with captures…", len(todo))
    all_rows: list[dict] = []
    for ep in todo:
        caps = cdx_captures(ep["url"])
        if not caps:
            continue
        for cap in caps:
            raw = WB_RAW.format(ts=cap["timestamp"], url=ep["url"])
            try:
                r = requests.get(raw, timeout=90)
                time.sleep(SLEEP_BETWEEN)
                if r.status_code != 200:
                    continue
                payload = r.json()
            except Exception:
                continue
            parsed = s19.parse_response(ep["url"], payload, ep)
            coleta = f"{cap['timestamp'][:4]}-{cap['timestamp'][4:6]}-{cap['timestamp'][6:8]}"
            for row in parsed:
                row["data_coleta"] = coleta
            all_rows.extend(parsed)
    if not all_rows:
        log.warning("No rows reconstructed.")
        return
    long_df, _ = s19.build_panels(all_rows)
    long_path = paths.TARIFAS_PROC / "openfinance_fees_panel_long.csv"
    combined = s19.append_long_panel(long_df, long_path)
    log.info("Backfill appended %d reconstructed rows; panel now %d rows.",
             len(long_df), len(combined))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    ap = argparse.ArgumentParser(description="Wayback feasibility probe / reconstruction for OFB fees")
    ap.add_argument("--families", default="accounts",
                    help="Families to probe (default accounts; 'all' for everything).")
    ap.add_argument("--historical-endpoints", action="store_true",
                    help="Also reconstruct historical endpoints from archived directory snapshots.")
    ap.add_argument("--backfill", action="store_true",
                    help="After probing, reconstruct history for endpoints with captures (gated).")
    ap.add_argument("--force-backfill", action="store_true",
                    help="Backfill even if the feasibility gate fails.")
    ap.add_argument("--no-cache", action="store_true", help="Bypass the CDX cache.")
    args = ap.parse_args()

    products = s19.resolve_products(args.families)
    eps = current_endpoints(products)
    if args.historical_endpoints:
        hist = historical_endpoints(products)
        merged = {(e["url"], e["api_family"]): e for e in eps}
        for e in hist:
            merged.setdefault((e["url"], e["api_family"]), e)
        eps = list(merged.values())
    log.info("Probing Wayback coverage for %d endpoints (families=%s)…", len(eps), products)

    cov = probe(eps, use_cache=not args.no_cache)
    OUT_COVERAGE.parent.mkdir(parents=True, exist_ok=True)
    cov.to_csv(OUT_COVERAGE, index=False)
    log.info("Saved coverage: %s (%d rows)", OUT_COVERAGE.name, len(cov))

    passed = gate(cov)

    if args.backfill:
        if passed or args.force_backfill:
            backfill(eps, cov)
        else:
            log.info("Gate failed — skipping backfill (use --force-backfill to override).")

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
