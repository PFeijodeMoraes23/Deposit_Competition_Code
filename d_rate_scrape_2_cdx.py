"""
d_rate_scrape_2_cdx.py
==============================
Interrogates the Wayback Machine CDX API to fetch URLs with snapshots containing
relevant keywords or PDF content, ensuring we only query valid data points 
between banks' first operating year and the present.
"""
import json
import requests
import time
from pathlib import Path
from urllib.parse import urlencode

# 1. Load targets
targets_file = Path('scraper_targets.json')
if not targets_file.exists():
    print("scraper_targets.json not found! Run d_rate_scrape_1_targets.py first.")
    exit(1)

with open(targets_file, 'r', encoding='utf-8') as f:
    targets = json.load(f)

cdx_url = "http://web.archive.org/cdx/search/cdx"

# Filter keywords we care about inside the URL structure to cut down the giant response sizes.
KEYWORDS = ['tarifa', 'taxa', 'remuneracao', 'rentabilidade', 'rendimento', 'conta', 'termos', 'contrato', 'regulamento']
result_urls = []

for target in targets:
    domain = target['domain']
    if not domain or domain == '.com.br':
        continue

    print(f"[{target['name']}] querying Wayback Machine CDX API for: *.{domain}/*")
    
    # We want snapshots spanning target's first_year until roughly today.
    from_year = int(float(target['first_year']))
    start_timestamp = f"{from_year}0101000000"

    params = {
        'url': f"*.{domain}/*",
        'output': 'json',
        'fl': 'timestamp,original,mimetype,statuscode', # Fields to return
        'filter': 'statuscode:200', # Only fetch OK snapshots
        'collapse': 'urlkey', # Deduplicate the exact exact url
        'from': start_timestamp,
        'limit': 50000 # Just in case
    }
    
    req_url = f"{cdx_url}?{urlencode(params)}"
    
    try:
        response = requests.get(req_url, timeout=30)
        if response.status_code != 200:
            print(f"  Error: CDX returned {response.status_code}")
            continue
            
        data = response.json()
        if not data or len(data) <= 1:
            print("  No snapshots found.")
            continue
            
        # First row is headers: ["timestamp", "original", "mimetype", "statuscode"]
        headers = data[0]
        rows = data[1:]
        
        matches = 0
        for r in rows:
            ts, orig_url, mime, _ = r
            orig_lower = orig_url.lower()

            # We aggressively filter URLs containing at least one relevant keyword
            if any(k in orig_lower for k in KEYWORDS) or mime == 'application/pdf':
                matches += 1
                result_urls.append({
                    'cod_conglomerado': target['cod_conglomerado'],
                    'name': target['name'],
                    'domain': domain,
                    'timestamp': ts,
                    'original_url': orig_url,
                    'mimetype': mime,
                    'wayback_url': f"http://web.archive.org/web/{ts}/{orig_url}"
                })
        print(f"  Found {matches} candidate snapshots.")
        
    except Exception as e:
        print(f"  Failed request: {e}")
    
    # Sleep to be polite to the IA API
    print("  Sleeping 3 seconds block...")
    time.sleep(3)

# 2. Save Results
out_path = Path('scraper_urls.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(result_urls, f, indent=4, ensure_ascii=False)

print(f"\nDiscovered {len(result_urls)} historical URLs to fetch.")
print(f"Saved payload to {out_path.resolve()}")