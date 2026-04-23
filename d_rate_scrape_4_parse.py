"""
d_rate_scrape_4_parse.py
==============================
Parses downloaded archival HTML and PDF files to extract historical deposit yields.
Relies on NLP heuristics matching common Brazilian CDI and Selic rate phrasing,
and outputs candidate rates with surrounding context for manual validation.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import os
import re
import json
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
import pdfplumber

root = Path('.').resolve().parents[2]
ip_scrape_dir = root / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed' / 'IP_SCRAPE'

PAGES_DIR = ip_scrape_dir / 'archive_html'
PDFS_DIR = ip_scrape_dir / 'archive_pdfs'

# Regexes for common deposit rate phrasing in Brazil
# Looks for "100% do CDI", "rende 100% CDI", "Rendimento: 105% do CDI", "100% ao CDI"
CDI_REGEX = re.compile(r'(\d{2,3})[%\s]*(?:do|ao)?\s*CDI', re.IGNORECASE)
SELIC_REGEX = re.compile(r'(\d{2,3})[%\s]*(?:da|a)?\s*SELIC', re.IGNORECASE)

def parse_html(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            
        text = soup.get_text(separator=' ', strip=True)
        return _extract_rates(text, filepath.name)
    except Exception as e:
        print(f"Error parsing HTML {filepath.name}: {e}")
        return []

def parse_pdf(filepath):
    try:
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages[:10]: # Limit to first 10 pages for speed/relevance
                t = page.extract_text()
                if t: text += t + "\n"
        return _extract_rates(text, filepath.name)
    except Exception as e:
        print(f"Error parsing PDF {filepath.name}: {e}")
        return []

def _extract_rates(text, filename):
    extracted = []
    
    # Split cod, timestamp, and url from filename (format: C00800_20191231_url)
    parts = filename.split('_', 2)
    conglomerado = parts[0]
    ts = parts[1] if len(parts) > 1 else '0000'
    date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else 'Unknown'
    
    # Check CDI matches
    for match in CDI_REGEX.finditer(text):
        rate_val = match.group(1)
        # Grab some context around the match to manually verify it isn't a loan rate!
        start = max(0, match.start() - 60)
        end = min(len(text), match.end() + 60)
        context = text[start:end].replace('\n', ' ').strip()
        
        extracted.append({
            'CodConglomerado': conglomerado,
            'Snapshot_Date': date_str,
            'Rate_Type': 'CDI',
            'Rate_Value': f"{rate_val}%",
            'Context': f"... {context} ...",
            'Source': filename
        })
        
    return extracted

def main():
    results = []
    
    # 1. Parse HTML files
    if PAGES_DIR.exists():
        for f in PAGES_DIR.glob('*.html'):
            results.extend(parse_html(f))

    # 2. Parse PDF files
    if PDFS_DIR.exists():
        for f in PDFS_DIR.glob('*.pdf'):
            results.extend(parse_pdf(f))

    # Basic deductive deduplication: if the exact same Context appears for the same cod + date
    dedup_map = {
        f"{r['CodConglomerado']}_{r['Snapshot_Date']}_{r['Rate_Value']}_{r['Context']}": r
        for r in results
    }

    final_list = list(dedup_map.values())
    df = pd.DataFrame(final_list)

    if not df.empty:
        df = df.sort_values(by=['CodConglomerado', 'Snapshot_Date'])
        
    out_path = ip_scrape_dir / 'extracted_historical_rates.csv'
    df.to_csv(out_path, index=False)
    print(f"Extraction complete. Found {len(final_list)} distinct rate mentions.")
    print(f"Saved to {out_path.resolve()}. Manual review is STRONGLY advised to filter out loan rates.")

if __name__ == '__main__':
    main()