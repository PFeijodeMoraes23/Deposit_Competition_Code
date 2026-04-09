"""
d_rate_scrape_3_fetch.py
==============================
Fetches historical HTML and PDF snapshots from the Internet Archive (Wayback Machine).
Uses an asynchronous queue and session pool to download the discovered URLs efficiently 
while respecting rate limits, stripping IA toolbars from HTML for clean text parsing.
"""
import json
import asyncio
import aiohttp
from pathlib import Path
from bs4 import BeautifulSoup
import logging
import random
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

root = Path('.').resolve().parents[2]
ip_scrape_dir = root / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed' / 'IP_SCRAPE'

urls_file = ip_scrape_dir / 'scraper_urls.json'
if not urls_file.exists():
    print(f"{urls_file} not found! Run d_rate_scrape_2_cdx.py first.")
    exit(1)

with open(urls_file, 'r', encoding='utf-8') as f:
    urls_to_fetch = json.load(f)

# output directories based on type inside IP_SCRAPE
PAGES_DIR = ip_scrape_dir / 'archive_html'
PAGES_DIR.mkdir(exist_ok=True)
PDFS_DIR = ip_scrape_dir / 'archive_pdfs'
PDFS_DIR.mkdir(exist_ok=True)

MAX_CONCURRENT = 5 # don't hammer the fragile Internet Archive too hard!
timeout = aiohttp.ClientTimeout(total=45)

async def fetch_snapshot(session, item, prefix_dir):
    ts = item['timestamp']
    firm_cod = item['cod_conglomerado']
    mime = item['mimetype']
    wb_url = item['wayback_url']
    
    sanitized_url = "".join(c for c in item['original_url'] if c.isalnum() or c in '-_.')[:50]
    filename = f"{firm_cod}_{ts}_{sanitized_url}"
    
    if mime == 'application/pdf' or wb_url.endswith('.pdf'):
        filepath = PDFS_DIR / f"{filename}.pdf"
    else:
        filepath = PAGES_DIR / f"{filename}.html"
    
    if filepath.exists():
        return # Skip

    try:
        async with session.get(wb_url) as response:
            if response.status == 200:
                content = await response.read()
                
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
        logging.error(f"Error fetching {wb_url}: {e}")
        return False

async def worker(queue, session, pbar):
    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            break
            
        await fetch_snapshot(session, task, PAGES_DIR)
        await asyncio.sleep(random.uniform(0.5, 1.5)) # Random spread avoids ban
        
        queue.task_done()

async def main():
    queue = asyncio.Queue()
    for u in urls_to_fetch:
        queue.put_nowait(u)
        
    logging.info(f"Loaded {len(urls_to_fetch)} snapshot requests into queue. Launching {MAX_CONCURRENT} workers.")
        
    # We add None objects to stop the workers
    for _ in range(MAX_CONCURRENT):
        queue.put_nowait(None)
        
    async with aiohttp.ClientSession(timeout=timeout) as session:
        workers = [asyncio.create_task(worker(queue, session, None)) for _ in range(MAX_CONCURRENT)]
        await queue.join()

if __name__ == '__main__':
    # Fix for windows Proactor loop event
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
    print("Done fetching Archive snapshots.")