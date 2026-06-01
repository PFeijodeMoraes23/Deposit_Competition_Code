"""
scrape_7e_cosif_download.py
============================
Download missing COSIF balancete ZIPs (BANCOS, SOCIEDADES, COOPERATIVAS)
for months not already on disk, then re-run scrape_7b to extend the panel.

HOW TO FIND THE URL PATTERN
-----------------------------
The BCB's informacoescontabeis page is a JS SPA that doesn't expose download
URLs in static HTML.  To find the real URL:

  1. Open Chrome DevTools (F12) → Network tab
  2. Navigate to: https://www.bcb.gov.br/estabilidadefinanceira/informacoescontabeis
  3. Click on any "Banco Múltiplo" download link
  4. In the Network tab, look for the request that downloads the ZIP
  5. Right-click → Copy → Copy as cURL → paste the URL_TEMPLATE below

The URL is typically structured as one of:
  https://www.bcb.gov.br/content/estabilidadefinanceira/.../{YYYYMM}BANCOS.ZIP
  https://cdn.bcb.gov.br/.../{YYYYMM}BANCOS.ZIP

USAGE (once URL_TEMPLATE is filled in)
---------------------------------------
  python scrape_7e_cosif_download.py
  python scrape_7e_cosif_download.py --from 202212 --to 202506
  python scrape_7e_cosif_download.py --dry-run   # just show what would download
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from datetime import datetime, date

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# *** FILL THIS IN after browser inspection ***
# ---------------------------------------------------------------------------
# Replace the {YYYYMM} placeholder and {TYPE} (BANCOS, SOCIEDADES, COOPERATIVAS)
URL_TEMPLATE = "https://FILL_IN_FROM_BROWSER/{YYYYMM}{TYPE}.ZIP"
# ---------------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[2]
COSIF_RAW = _REPO / "BCB" / "Egan_et_al_2025_Rep" / "raw" / "COSIF_RAW"
TYPES    = ["BANCOS", "SOCIEDADES", "COOPERATIVAS"]


def _yyyymm_range(from_ym: str, to_ym: str) -> list[str]:
    """Generate YYYYMM strings from from_ym to to_ym inclusive."""
    result = []
    y, m = int(from_ym[:4]), int(from_ym[4:6])
    ey, em = int(to_ym[:4]), int(to_ym[4:6])
    while (y, m) <= (ey, em):
        result.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return result


def main(from_ym: str, to_ym: str, dry_run: bool = False) -> None:
    if "FILL_IN" in URL_TEMPLATE:
        log.error(
            "URL_TEMPLATE is not set. Open Chrome DevTools on the BCB "
            "informacoescontabeis page, download any ZIP, copy the URL, "
            "and paste it into URL_TEMPLATE in this script."
        )
        return

    COSIF_RAW.mkdir(parents=True, exist_ok=True)
    periods = _yyyymm_range(from_ym, to_ym)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    to_download = []
    for ym in periods:
        for typ in TYPES:
            fname = f"{ym}{typ}.ZIP"
            dest  = COSIF_RAW / fname
            if dest.exists() and dest.stat().st_size > 1000:
                continue   # already have it
            url = URL_TEMPLATE.format(YYYYMM=ym, TYPE=typ)
            to_download.append((url, dest, fname))

    log.info("%d files to download (%d already present)",
             len(to_download), len(periods) * len(TYPES) - len(to_download))

    if dry_run:
        for url, _, fname in to_download:
            print(fname, "←", url)
        return

    for i, (url, dest, fname) in enumerate(to_download, 1):
        log.info("[%d/%d] %s", i, len(to_download), fname)
        try:
            r = session.get(url, timeout=120, stream=True)
            r.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            log.info("  → %d KB", dest.stat().st_size // 1024)
        except Exception as exc:
            log.warning("  FAILED: %s", exc)
            if dest.exists():
                dest.unlink()
        time.sleep(0.5)

    log.info("Done. Re-run scrape_7b_cosif_service_fees.py to rebuild the panel.")


if __name__ == "__main__":
    # Default: download from Dec 2022 (first month after existing data) to today
    last_existing = "202211"
    this_month = date.today().strftime("%Y%m")

    parser = argparse.ArgumentParser(description="COSIF balancete ZIP downloader")
    parser.add_argument("--from", dest="from_ym", default="202212",
                        help="First YYYYMM to download (default: 202212)")
    parser.add_argument("--to", dest="to_ym", default=this_month,
                        help=f"Last YYYYMM to download (default: {this_month})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    args = parser.parse_args()
    main(from_ym=args.from_ym, to_ym=args.to_ym, dry_run=args.dry_run)
