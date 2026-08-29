"""
scrape_deposit_rate_cdx.py
==============================
Interrogates the Wayback Machine CDX API to fetch URLs with snapshots containing
relevant keywords or PDF content, ensuring we only query valid data points 
between banks' first operating year and the present.

Resumable: each domain's response is cached under IP_SCRAPE/cdx_cache/.
Re-runs skip any domain whose cache file already exists. Use --force to
re-query a single domain (e.g. after a 503 stub was cached), or
--force-all to ignore the cache entirely.

No retry: on any error (503/429/timeout/etc.) we skip that domain and
move on. Just re-run the script later -- successful domains stay cached,
so only the failed ones get queried again.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import argparse
import json
import re
import requests
import time
from pathlib import Path
from urllib.parse import urlencode

# 1. Load targets
# Ensure we can find the targets from step 1
root = Path(__file__).resolve().parents[2]
ip_scrape_dir = root / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed' / 'IP_SCRAPE'
cache_dir = ip_scrape_dir / 'cdx_cache'

cdx_url = "http://web.archive.org/cdx/search/cdx"

# Filter keywords we care about inside the URL structure to cut down the giant response sizes.
KEYWORDS = ['tarifa', 'taxa', 'remuneracao', 'rentabilidade', 'rendimento', 'conta', 'termos', 'contrato', 'regulamento']


def _slug(domain: str) -> str:
    """Filesystem-safe cache key for a domain."""
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', domain)


def query_cdx(domain: str, from_year: int):
    """Hit the CDX API for one domain. No retry: on any non-200 we give up on
    this domain and leave it for a future run (per-domain cache makes that
    cheap). Returns list of result dicts, or None on failure."""
    start_timestamp = f"{from_year}0101000000"
    params = {
        'url': f"*.{domain}/*",
        'output': 'json',
        'fl': 'timestamp,original,mimetype,statuscode',
        'filter': 'statuscode:200',
        'collapse': 'urlkey',
        'from': start_timestamp,
        'limit': 50000,
    }
    req_url = f"{cdx_url}?{urlencode(params)}"

    try:
        response = requests.get(req_url, timeout=60)
    except Exception as e:
        print(f"  request failed: {e}")
        return None

    if response.status_code != 200:
        print(f"  CDX returned {response.status_code} - skipping (retry on next run)")
        return None

    try:
        data = response.json()
    except Exception as e:
        print(f"  bad JSON: {e}")
        return None

    if not data or len(data) <= 1:
        return []
    rows = data[1:]
    out = []
    for r in rows:
        ts, orig_url, mime, _ = r
        orig_lower = orig_url.lower()
        if any(k in orig_lower for k in KEYWORDS) or mime == 'application/pdf':
            out.append({
                'timestamp': ts,
                'original_url': orig_url,
                'mimetype': mime,
                'wayback_url': f"http://web.archive.org/web/{ts}/{orig_url}",
            })
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Query Wayback CDX per domain with resumable per-domain cache.")
    ap.add_argument('--force-all', action='store_true',
                    help="Ignore cache: re-query every domain.")
    ap.add_argument('--force', action='append', default=[],
                    help="Re-query just this domain (can repeat).")
    ap.add_argument('--sleep', type=float, default=3.0,
                    help="Politeness sleep between domains (seconds).")
    args = ap.parse_args()

    targets_file = ip_scrape_dir / 'scraper_targets.json'
    if not targets_file.exists():
        print(f"{targets_file} not found! Run scrape_deposit_rate_targets.py first.")
        return 1

    with open(targets_file, 'r', encoding='utf-8') as f:
        targets = json.load(f)

    cache_dir.mkdir(parents=True, exist_ok=True)
    force_set = {d.lower() for d in args.force}

    n_cached = n_queried = n_skipped = n_failed = 0
    result_urls = []

    for target in targets:
        domain = target['domain']
        if not domain or domain == '.com.br':
            n_skipped += 1
            continue

        cache_file = cache_dir / f"{_slug(domain)}.json"
        use_cache = (cache_file.exists()
                     and not args.force_all
                     and domain.lower() not in force_set)

        if use_cache:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                print(f"[{target['name']}] CACHED ({len(matches)} matches) - skip CDX")
                n_cached += 1
            except Exception as e:
                print(f"[{target['name']}] cache unreadable ({e}); re-querying")
                use_cache = False

        if not use_cache:
            from_year = int(float(target['first_year']))
            print(f"[{target['name']}] querying CDX *.{domain}/* from {from_year}")
            matches = query_cdx(domain, from_year)
            if matches is None:
                # Hard failure: do NOT cache. Try again next run.
                n_failed += 1
                time.sleep(args.sleep)
                continue
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(matches, f, ensure_ascii=False)
            print(f"  Found {len(matches)} candidate snapshots (cached).")
            n_queried += 1
            time.sleep(args.sleep)

        # Attach target metadata
        for m in matches:
            result_urls.append({
                'cod_conglomerado': target['cod_conglomerado'],
                'name': target['name'],
                'domain': domain,
                **m,
            })

    # 2. Save Results
    out_path = ip_scrape_dir / 'scraper_urls.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result_urls, f, indent=4, ensure_ascii=False)

    print(f"\nCDX summary: {n_cached} cached | {n_queried} freshly queried | "
          f"{n_failed} failed | {n_skipped} skipped (no domain)")
    print(f"Discovered {len(result_urls)} historical URLs to fetch.")
    print(f"Saved payload to {out_path.resolve()}")
    if n_failed:
        print(f"NOTE: {n_failed} domains failed (likely Wayback 503). "
              f"Re-run this script later to pick them up; cached domains "
              f"are skipped automatically.")
    # Always exit 0: stage 2 is incrementally resumable, downstream stages
    # can run on whatever URLs we already have.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())