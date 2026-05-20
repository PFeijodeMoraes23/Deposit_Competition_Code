"""
d_rate_scrape_3_fetch.py
==============================
Fetches historical HTML and PDF snapshots from the Internet Archive (Wayback Machine).
Uses an asynchronous queue and session pool to download the discovered URLs efficiently
while respecting rate limits, stripping IA toolbars from HTML for clean text parsing.

Resumable
---------
This script is idempotent. Before enqueuing the work list, every URL whose
target file already exists on disk is filtered out, so re-running after a
partial completion only fetches missing items.  Use `--dry-run` to preview
how many fetches would actually happen.

CLI
---
  --workers N         number of concurrent async workers (default 1; raise
                      cautiously, the Wayback Machine bans aggressively at >2)
  --sleep-min S       min seconds of jitter between successful requests (default 2.5)
  --sleep-max S       max seconds of jitter (default 4.5)
  --dry-run           report counts only, do not fetch
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import argparse
import json
import asyncio
import aiohttp
from pathlib import Path
from bs4 import BeautifulSoup
import logging
import random
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Resolve paths from this script's location, NOT the current working directory.
# This guarantees the cache is reused regardless of where the script is invoked
# from, so re-running fetch will only re-download URLs whose file does not exist
# on disk (see `if filepath.exists(): return` below).
root = Path(__file__).resolve().parents[2]
ip_scrape_dir = root / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed' / 'IP_SCRAPE'

urls_file = ip_scrape_dir / 'scraper_urls.json'

# output directories based on type inside IP_SCRAPE
PAGES_DIR = ip_scrape_dir / 'archive_html'
PDFS_DIR = ip_scrape_dir / 'archive_pdfs'
# Images (rate tables embedded as PNG/JPG; needs OCR in stage 4)
IMAGES_DIR = ip_scrape_dir / 'archive_images'

_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.gif')
_IMAGE_MIMES = ('image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif')

# Default URL-keyword denylist (substring match against lowercased
# wayback_url + original_url). Skips files that almost never carry deposit
# rate language and waste Wayback bandwidth. Override with --skip-url-patterns
# or disable with --no-url-filter.
DEFAULT_FETCH_SKIP_PATTERNS = (
    # Auto-industry brochures (Cielo/Stone POS partnerships)
    'honda', 'mercedes', 'mercedesbenz', 'automovei', 'automoveis',
    'motos', 'caminhoes', 'garantia-estendida',
    # WordPress uploads, consortium docs, post-sale
    'pos-venda', 'consorcio', 'wp-content/uploads/202',
    # Legal/compliance noise
    'politica-de-privacidade', 'termos-de-uso', 'cookie',
    'edital', 'demonstracao-financeira', 'demonstracoes-financeiras',
    'relatorio-anual', 'relatorio-de-sustentabilidade', 'ouvidoria',
    'sac/', 'imprensa/', 'press/',
    # Generic asset paths
    '/assets/', '/static/', '/dist/', '/build/', '.min.',
    # Career / careers / vagas
    'carreira', 'careers', 'vagas/', 'trabalhe-conosco',
)

# Configurable at runtime (set in main() from CLI).
MAX_CONCURRENT = 1
SLEEP_MIN = 2.5
SLEEP_MAX = 4.5
_DRY_RUN = False
_INCLUDE_IMAGES = False
_SKIP_PATTERNS = ()
_MIME_PRIORITY = 'html'  # one of: 'html', 'pdf', 'none'
timeout = aiohttp.ClientTimeout(total=60)


def _mime_rank(item):
    """Sort key for queue ordering. Lower = fetched first.
    Modes: 'html' drains HTML before images before PDFs; 'pdf' is reverse;
    'none' returns 0 for all (stable insertion order).
    """
    if _MIME_PRIORITY == 'none':
        return 0
    mime = (item.get('mimetype') or '').lower()
    wb = (item.get('wayback_url') or '').lower()
    if mime.startswith('text/html') or wb.endswith(('.html', '.htm')) or (
        not wb.endswith(('.pdf',) + _IMAGE_EXTS) and 'html' in mime
    ):
        kind = 'html'
    elif mime in _IMAGE_MIMES or wb.endswith(_IMAGE_EXTS):
        kind = 'image'
    elif mime == 'application/pdf' or wb.endswith('.pdf'):
        kind = 'pdf'
    else:
        kind = 'html'  # treat unknowns as cheap HTML-like
    order = {'html': 0, 'image': 1, 'pdf': 2}
    if _MIME_PRIORITY == 'pdf':
        order = {'pdf': 0, 'image': 1, 'html': 2}
    return order.get(kind, 3)


def _is_image(item):
    mime = (item.get('mimetype') or '').lower()
    wb_url = (item.get('wayback_url') or '').lower()
    if mime in _IMAGE_MIMES:
        return True
    return any(wb_url.endswith(ext) for ext in _IMAGE_EXTS)


def _image_ext(item):
    mime = (item.get('mimetype') or '').lower()
    wb_url = (item.get('wayback_url') or '').lower()
    for ext in _IMAGE_EXTS:
        if wb_url.endswith(ext):
            return ext
    if 'png' in mime: return '.png'
    if 'webp' in mime: return '.webp'
    if 'gif' in mime: return '.gif'
    return '.jpg'


def _target_filepath(item):
    """Return the on-disk path where this item would be cached.
    Logic duplicated from fetch_snapshot so we can pre-filter the queue."""
    ts = item['timestamp']
    firm_cod = item['cod_conglomerado']
    mime = item.get('mimetype')
    wb_url = item['wayback_url']
    sanitized_url = "".join(c for c in item['original_url']
                            if c.isalnum() or c in '-_.')[:50]
    filename = f"{firm_cod}_{ts}_{sanitized_url}"
    if mime == 'application/pdf' or wb_url.endswith('.pdf'):
        return PDFS_DIR / f"{filename}.pdf"
    if _is_image(item):
        return IMAGES_DIR / f"{filename}{_image_ext(item)}"
    return PAGES_DIR / f"{filename}.html"

async def fetch_snapshot(session, item, prefix_dir):
    ts = item['timestamp']
    firm_cod = item['cod_conglomerado']
    mime = item['mimetype']
    wb_url = item['wayback_url']
    
    # Enforce HTTPS instead of HTTP to prevent port 80 SSL redirect issues
    if wb_url.startswith("http://"):
        wb_url = wb_url.replace("http://", "https://", 1)
    
    sanitized_url = "".join(c for c in item['original_url'] if c.isalnum() or c in '-_.')[:50]
    filename = f"{firm_cod}_{ts}_{sanitized_url}"
    
    if mime == 'application/pdf' or wb_url.endswith('.pdf'):
        filepath = PDFS_DIR / f"{filename}.pdf"
    elif _is_image(item):
        filepath = IMAGES_DIR / f"{filename}{_image_ext(item)}"
    else:
        filepath = PAGES_DIR / f"{filename}.html"
    
    if filepath.exists():
        return # Skip

    for attempt in range(5): # Up to 5 retries to ride out IP bans
        try:
            async with session.get(wb_url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Binary write for images (no HTML cleaning)
                    if filepath.suffix.lower() in _IMAGE_EXTS:
                        with open(filepath, 'wb') as f:
                            f.write(content)
                        return True

                    # if html, clean out wayback toolbar bloat
                    if not filepath.name.endswith('.pdf'):
                        try:
                            soup = BeautifulSoup(content, 'html.parser')
                            for script in soup(['script', 'style', 'iframe', 'svg', 'noscript']):
                                script.decompose()
                            
                            # Remove the annoying IA toolbar injections so we only store the actual text DOM
                            for ia in soup.find_all('div', id="wm-ipp-base"):
                                ia.decompose()
                                
                            # Save
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(str(soup))
                        except Exception as e:
                            # Fallback raw write
                            print(f"Parse error for {wb_url}, falling back to raw save. {e}")
                            with open(filepath, 'wb') as f:
                                f.write(content)
                    else:
                        # PDF Binary 
                        with open(filepath, 'wb') as f:
                            f.write(content)

                    return True
                else:
                    logging.warning(f"Status {response.status} for {wb_url}")
                    return False
        except Exception as e:
            if "Connect call failed" in str(e) or "ssl" in str(e).lower() or "443" in str(e):
                logging.error(f"Ban detected on {wb_url}, sleeping 60 seconds... (Attempt {attempt+1}/5)")
                await asyncio.sleep(60) # Sleep off the soft-ban
            else:
                logging.error(f"Error fetching {wb_url}: {e}")
                return False
                
    return False # Failed all attempts

async def worker(queue, session, pbar):
    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            break
            
        res = await fetch_snapshot(session, task, PAGES_DIR)
        
        # Only sleep if we actually made a network request (res is not None).
        # We don't want to sleep when skipping files that are already downloaded!
        if res is not None:
            await asyncio.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
        
        queue.task_done()

async def main():
    queue = asyncio.Queue()
    valid_mimes = ['text/html', 'application/pdf']
    if _INCLUDE_IMAGES:
        valid_mimes = valid_mimes + list(_IMAGE_MIMES)
    def _keep(u):
        if u.get('mimetype') in valid_mimes:
            return True
        wb = str(u.get('wayback_url', '')).lower()
        if wb.endswith('.pdf'):
            return True
        if _INCLUDE_IMAGES and any(wb.endswith(ext) for ext in _IMAGE_EXTS):
            return True
        return False
    def _url_blocked(u):
        if not _SKIP_PATTERNS:
            return False
        haystack = (str(u.get('wayback_url', '')) + ' ' + str(u.get('original_url', ''))).lower()
        return any(p in haystack for p in _SKIP_PATTERNS)
    valid_urls = [u for u in urls_to_fetch if _keep(u) and not _url_blocked(u)]
    if _SKIP_PATTERNS:
        n_blocked = sum(1 for u in urls_to_fetch if _keep(u) and _url_blocked(u))
        logging.info(f"URL-pattern filter blocked {n_blocked:,} URLs "
                     f"({len(_SKIP_PATTERNS)} patterns).")

    # Pre-filter: skip URLs whose cache file already exists. This makes
    # progress reporting honest and lets the user see how much work remains.
    pending, cached = [], 0
    for u in valid_urls:
        if _target_filepath(u).exists():
            cached += 1
        else:
            pending.append(u)

    total = len(valid_urls)
    logging.info(
        f"Snapshot queue: {total:,} total | {cached:,} already cached (skip) "
        f"| {len(pending):,} to fetch"
    )

    # Order the queue by mime priority so cheap HTML rate-pages get parsed
    # first and start yielding rate mentions long before the PDF tail.
    if _MIME_PRIORITY != 'none' and pending:
        pending.sort(key=_mime_rank)
        from collections import Counter
        ranks = Counter(_mime_rank(u) for u in pending)
        logging.info(
            f"Queue ordered by mime-priority='{_MIME_PRIORITY}': "
            f"rank0={ranks.get(0,0):,} rank1={ranks.get(1,0):,} "
            f"rank2={ranks.get(2,0):,}"
        )

    if _DRY_RUN:
        logging.info("--dry-run set: exiting without fetching.")
        return
    if not pending:
        logging.info("Nothing to fetch.")
        return

    for u in pending:
        queue.put_nowait(u)

    logging.info(f"Launching {MAX_CONCURRENT} worker(s) with sleep "
                 f"[{SLEEP_MIN:.1f}, {SLEEP_MAX:.1f}]s between requests.")

    # We add None objects to stop the workers
    for _ in range(MAX_CONCURRENT):
        queue.put_nowait(None)
        
    async with aiohttp.ClientSession(timeout=timeout) as session:
        workers = [asyncio.create_task(worker(queue, session, None)) for _ in range(MAX_CONCURRENT)]
        await queue.join()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, default=1,
                    help='Concurrent async workers (default 1). Wayback Machine '
                         'bans aggressively at >2; raise cautiously.')
    ap.add_argument('--sleep-min', type=float, default=2.5,
                    help='Min seconds of jitter between requests (default 2.5).')
    ap.add_argument('--sleep-max', type=float, default=4.5,
                    help='Max seconds of jitter between requests (default 4.5).')
    ap.add_argument('--dry-run', action='store_true',
                    help='Report cached/pending counts and exit without fetching.')
    ap.add_argument('--include-images', action='store_true',
                    help='Also fetch image snapshots (png/jpg/webp/gif) into '
                         'archive_images/ for downstream OCR in stage 4.')
    ap.add_argument('--skip-url-patterns', type=str,
                    default=','.join(DEFAULT_FETCH_SKIP_PATTERNS),
                    help='Comma-separated substrings; URLs matching any are '
                         'skipped before fetching. Big win on --target-all runs.')
    ap.add_argument('--no-url-filter', action='store_true',
                    help='Disable URL-pattern filter (fetch every eligible URL).')
    ap.add_argument('--mime-priority', choices=('html', 'pdf', 'none'),
                    default='html',
                    help="Queue ordering: 'html' (default) drains HTML before "
                         "images before PDFs; 'pdf' reverses; 'none' keeps "
                         "insertion order.")
    args = ap.parse_args()
    MAX_CONCURRENT = max(1, args.workers)
    SLEEP_MIN = max(0.0, args.sleep_min)
    SLEEP_MAX = max(SLEEP_MIN, args.sleep_max)
    _DRY_RUN = bool(args.dry_run)
    _INCLUDE_IMAGES = bool(args.include_images)
    _SKIP_PATTERNS = () if args.no_url_filter else tuple(
        p.strip().lower() for p in args.skip_url_patterns.split(',') if p.strip()
    )
    _MIME_PRIORITY = args.mime_priority

    if not urls_file.exists():
        print(f"{urls_file} not found! Run d_rate_scrape_2_cdx.py first.")
        exit(1)
    with open(urls_file, 'r', encoding='utf-8') as f:
        urls_to_fetch = json.load(f)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    if _INCLUDE_IMAGES:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Fix for windows Proactor loop event
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
    print("Done fetching Archive snapshots.")