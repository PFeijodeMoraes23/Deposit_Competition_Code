"""
d_rate_scrape_4_parse.py
==============================
Parses downloaded archival HTML and PDF files to extract historical deposit yields.
Relies on NLP heuristics matching common Brazilian CDI and Selic rate phrasing,
and outputs candidate rates with surrounding context for manual validation.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import argparse
import os
import re
import json
import logging
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed

# Primary PDF backend: pypdfium2 (Google PDFium, ~60x faster than pdfplumber).
# Fallback: pdfplumber for pages where pdfium fails.
try:
    import pypdfium2 as pdfium
    _HAVE_PDFIUM = True
except ImportError:  # pragma: no cover
    _HAVE_PDFIUM = False
import pdfplumber

# Silence pdfminer/pdfplumber stderr spam (e.g. "Cannot set non-stroke color...")
for _name in ("pdfminer", "pdfminer.pdfinterp", "pdfminer.pdfpage",
              "pdfminer.converter", "pdfminer.cmapdb", "pdfplumber"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# Resolve paths relative to this file, not CWD.
# parents[0]=Egan_et_al_2025_Rep, [1]=Code, [2]=Open-Finance, [3]=Open Finance
# Data lives under Open-Finance/BCB/..., so use parents[2].
root = Path(__file__).resolve().parents[2]
ip_scrape_dir = root / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed' / 'IP_SCRAPE'

PAGES_DIR = ip_scrape_dir / 'archive_html'
PDFS_DIR = ip_scrape_dir / 'archive_pdfs'

# Regexes for common deposit rate phrasing in Brazil.
# Word-bounded so "1000% CDI" doesn't yield a spurious "100" match.
# CDI is masculine (do/ao CDI); SELIC is feminine (da/a SELIC). "de" is allowed for both.
CDI_REGEX = re.compile(r'\b(\d{2,3})\s*%?\s*(?:do|ao|de)?\s*CDI\b', re.IGNORECASE)
SELIC_REGEX = re.compile(r'\b(\d{2,3})\s*%?\s*(?:da|a|de)?\s*SELIC\b', re.IGNORECASE)

# How many pages to scan in each PDF. Rate sheets / fund regulations often
# have tables past page 10; widen to capture them. Override via --pages.
PDF_PAGE_LIMIT = 30

# Default URL-keyword denylist for pre-filtering obviously irrelevant PDFs
# (auto-industry warranty docs, consortium booklets, WP uploads, etc.). The
# match is a substring check against the lowercased filename. Override via
# --skip-url-patterns (comma-separated) or disable with --no-url-filter.
DEFAULT_SKIP_URL_PATTERNS = (
    'honda', 'mercedes', 'mercedesbenz', 'automovei', 'automoveis',
    'pos-venda', 'consorcio', 'wp-content', 'motos', 'caminhoes',
    'garantia-estendida',
)


def _should_skip(filename, patterns):
    if not patterns:
        return False
    low = filename.lower()
    return any(p in low for p in patterns)


def parse_html(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        text = soup.get_text(separator=' ', strip=True)
        return _extract_rates(text, filepath.name)
    except Exception as e:
        print(f"Error parsing HTML {filepath.name}: {e}")
        return []


def _extract_text_pdfium(filepath):
    text = ""
    doc = pdfium.PdfDocument(filepath)
    try:
        n = min(PDF_PAGE_LIMIT, len(doc))
        for i in range(n):
            page = doc[i]
            try:
                tp = page.get_textpage()
                try:
                    t = tp.get_text_range()
                finally:
                    tp.close()
                if t:
                    text += t + "\n"
            finally:
                page.close()
    finally:
        doc.close()
    return text


def _extract_text_plumber(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages[:PDF_PAGE_LIMIT]:
            try:
                t = page.extract_text()
            except Exception:
                continue
            if t:
                text += t + "\n"
    return text


def parse_pdf(filepath):
    try:
        if _HAVE_PDFIUM:
            try:
                text = _extract_text_pdfium(filepath)
            except Exception:
                # pdfium failed (corrupt/unsupported); try pdfplumber
                text = _extract_text_plumber(filepath)
        else:
            text = _extract_text_plumber(filepath)
        return _extract_rates(text, filepath.name)
    except Exception as e:
        print(f"Error parsing PDF {filepath.name}: {e}")
        return []


def _extract_rates(text, filename):
    extracted = []

    # Strip extension before splitting so `parts[2]` doesn't carry .pdf/.html
    stem = filename.rsplit('.', 1)[0]
    parts = stem.split('_', 2)
    if not parts or not parts[0]:
        return extracted
    conglomerado = parts[0]
    ts = parts[1] if len(parts) > 1 else '0000'
    date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else 'Unknown'

    for regex, rate_type in [(CDI_REGEX, 'CDI'), (SELIC_REGEX, 'SELIC')]:
        for match in regex.finditer(text):
            rate_val = match.group(1)
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)
            context = text[start:end].replace('\n', ' ').strip()
            extracted.append({
                'CodConglomerado': conglomerado,
                'Snapshot_Date': date_str,
                'Rate_Type': rate_type,
                'Rate_Value': f"{rate_val}%",
                'Context': f"... {context} ...",
                'Source': filename
            })

    return extracted


COLS = ['CodConglomerado', 'Snapshot_Date', 'Rate_Type', 'Rate_Value', 'Context', 'Source']


def _parse_pdf_worker(path_str):
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    return parse_pdf(Path(path_str))


def _worker_init(page_limit):
    """Propagate the CLI --pages override into each worker process."""
    global PDF_PAGE_LIMIT
    PDF_PAGE_LIMIT = page_limit


def main():
    global PDF_PAGE_LIMIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages', type=int, default=PDF_PAGE_LIMIT,
                        help=f'Max pages to scan per PDF (default: {PDF_PAGE_LIMIT}).')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help='Parallel PDF workers (default: CPU-1).')
    parser.add_argument('--skip-url-patterns', type=str, default=','.join(DEFAULT_SKIP_URL_PATTERNS),
                        help='Comma-separated substrings; PDFs whose filename contains any are skipped.')
    parser.add_argument('--no-url-filter', action='store_true',
                        help='Disable URL-keyword pre-filter (parse every PDF).')
    parser.add_argument('--out', type=str, default=str(ip_scrape_dir / 'extracted_historical_rates.csv'),
                        help='Output CSV path.')
    args = parser.parse_args()

    PDF_PAGE_LIMIT = max(1, args.pages)

    skip_patterns = () if args.no_url_filter else tuple(
        p.strip().lower() for p in args.skip_url_patterns.split(',') if p.strip()
    )

    results = []

    # 1. Parse HTML files (single-process: BeautifulSoup is fast enough)
    if PAGES_DIR.exists():
        html_files = sorted(PAGES_DIR.glob('*.html'))
        print(f"Parsing {len(html_files)} HTML files...")
        for i, f in enumerate(html_files, 1):
            results.extend(parse_html(f))
            if i % 500 == 0:
                print(f"  HTML: {i}/{len(html_files)}")
    else:
        print(f"HTML dir not found: {PAGES_DIR}")

    # 2. Parse PDF files in parallel (pdfplumber is the bottleneck)
    if PDFS_DIR.exists():
        all_pdfs = sorted(PDFS_DIR.glob('*.pdf'))
        skipped = [p for p in all_pdfs if _should_skip(p.name, skip_patterns)]
        pdf_files = [p for p in all_pdfs if not _should_skip(p.name, skip_patterns)]
        n = len(pdf_files)
        if skipped:
            print(f"URL-filter skipped {len(skipped)} of {len(all_pdfs)} PDFs "
                  f"(patterns: {','.join(skip_patterns)}).")
        workers = max(1, args.workers)
        print(f"Parsing {n} PDF files with {workers} workers, page_limit={PDF_PAGE_LIMIT}...")
        done = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(PDF_PAGE_LIMIT,),
        ) as ex:
            futures = {ex.submit(_parse_pdf_worker, str(p)): p for p in pdf_files}
            for fut in as_completed(futures):
                try:
                    results.extend(fut.result())
                except Exception as e:
                    print(f"  Worker error on {futures[fut].name}: {e}")
                done += 1
                if done % 100 == 0:
                    print(f"  PDF: {done}/{n}")
    else:
        print(f"PDF dir not found: {PDFS_DIR}")

    # Basic deduplication: same cod + date + value + context
    dedup_map = {
        f"{r['CodConglomerado']}_{r['Snapshot_Date']}_{r['Rate_Value']}_{r['Context']}": r
        for r in results
    }

    final_list = list(dedup_map.values())
    df = pd.DataFrame(final_list, columns=COLS) if final_list else pd.DataFrame(columns=COLS)

    if not df.empty:
        df = df.sort_values(by=['CodConglomerado', 'Snapshot_Date'])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Extraction complete. Found {len(final_list)} distinct rate mentions.")
    print(f"Saved to {out_path.resolve()}. Manual review is STRONGLY advised to filter out loan rates.")


if __name__ == '__main__':
    main()
