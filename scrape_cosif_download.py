"""
scrape_cosif_download.py
============================
Pipeline step 11. Download missing COSIF balancete CSV ZIPs (BANCOS, SOCIEDADES,
COOPERATIVAS, BLOPRUDENCIAL, and with --all-types the other categories) for months not
already on disk. Readers: panel_cosif_extract.py, panel_ip_rates.py (SOCIEDADES and
BLOPRUDENCIAL), fees/scrape_cosif_service_fees.py and the advertising scrapers.

A failed listing or download is reported at the end and the script exits 1.

DOWNLOAD MECHANISM (resolved 2026-06-02)
----------------------------------------
The legacy www4.bcb.gov.br/fis/cosif/... endpoint is dead. The modern BCB
"Balancetes e balanços patrimoniais" SPA
(https://www.bcb.gov.br/estabilidadefinanceira/balancetesbalancospatrimoniais)
lists files through a CMS API and serves the ZIPs from a /content/ path:

  List API : https://www.bcb.gov.br/api/servico/sitebcb/Documentos/byListGuid
             ?tronco=estabilidadefinanceira
             &guidLista=a11917e4-c729-4259-bd4e-0266827b6acd
             &ordem=DataDocumento desc
             &pasta=/<Folder>
  File URL : https://www.bcb.gov.br/content/estabilidadefinanceira/cosif/<Folder>/<YYYYMM><TYPE>.csv.zip

The API returns, per folder, a list of {Nome, Url, DataDocumento, Tamanho};
we download each whose YYYYMM is in range and not already on disk.

USAGE
-----
  python scrape_cosif_download.py
  python scrape_cosif_download.py --from 202212 --to 202512
  python scrape_cosif_download.py --dry-run
  python scrape_cosif_download.py --all-types   # include the other categories
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from utils import paths
from utils import refresh
from utils import bcb_http
COSIF_RAW = paths.COSIF_RAW

GUID     = "a11917e4-c729-4259-bd4e-0266827b6acd"
SITE     = bcb_http.BCB_SITE

# CMS folder -> COSIF document TYPE (filename suffix). The core three are downloaded on every
# run; the rest only with --all-types.
FOLDERS_CORE = {
    "/Bancos":                    "BANCOS",
    "/Sociedades":                "SOCIEDADES",
    "/Cooperativas-de-credito":   "COOPERATIVAS",
    "/Conglomerados-prudenciais": "BLOPRUDENCIAL",   # read by panel_ip_rates.py (step 15)
}
FOLDERS_EXTRA = {
    "/Conglomerados-financeiros":    "COMBINADOS",
    "/Administradoras-de-consorcios": "CONSORCIOS",
    "/Instituicoes-em-regime-especial": "LIQUIDACAO",
}

# Accept only canonical names: <YYYYMM><TYPE>.csv.zip  (rejects malformed
# duplicates the API occasionally lists, e.g. "...zip.csv.zip").
_NAME_RE = re.compile(r"^(\d{6})[A-Z]+\.csv\.zip$", re.I)


def main(from_ym: str, to_ym: str, dry_run: bool = False, all_types: bool = False) -> None:
    COSIF_RAW.mkdir(parents=True, exist_ok=True)
    folders = dict(FOLDERS_CORE)
    if all_types:
        folders.update(FOLDERS_EXTRA)

    session = bcb_http.retry_session()
    failures: list[str] = []

    to_download: list[tuple[str, Path, str]] = []
    for folder in folders:
        try:
            items = bcb_http.cms_documents(session, "estabilidadefinanceira", GUID, folder)
        except Exception as exc:
            log.error("Could not list %s: %s", folder, exc)
            failures.append(f"listing {folder}: {exc}")
            continue
        for it in items:
            nome = it.get("Nome", "")
            m = _NAME_RE.match(nome)
            if not m:
                continue
            ym = m.group(1)
            if not (from_ym <= ym <= to_ym):
                continue
            dest = COSIF_RAW / nome
            # Skip only if cached AND outside the rolling refresh window (monthly series).
            if (dest.exists() and dest.stat().st_size > 1000
                    and not refresh.is_recent_month(int(ym[:4]), int(ym[4:6]))):
                continue
            to_download.append((SITE + it["Url"], dest, nome))

    log.info("%d files to download", len(to_download))
    if dry_run:
        for url, _, nome in to_download:
            print(nome, "<-", url)
        return

    for i, (url, dest, nome) in enumerate(sorted(to_download, key=lambda t: t[2]), 1):
        log.info("[%d/%d] %s", i, len(to_download), nome)
        try:
            r = session.get(url, timeout=300)
            r.raise_for_status()
            if r.content[:2] != b"PK":
                log.error("  not a ZIP: %s", nome)
                failures.append(f"{nome}: not a ZIP")
                continue
            dest.write_bytes(r.content)
            log.info("  -> %d KB", dest.stat().st_size // 1024)
        except Exception as exc:
            log.error("  FAILED: %s", exc)
            failures.append(f"{nome}: {exc}")
        time.sleep(0.5)

    if failures:
        log.error("%d COSIF listing/download(s) FAILED (nothing was written for them):", len(failures))
        for f in failures:
            log.error("  - %s", f)
        sys.exit(1)
    log.info("Done.")


if __name__ == "__main__":
    this_month = date.today().strftime("%Y%m")
    parser = argparse.ArgumentParser(description="COSIF balancete CSV-ZIP downloader (BCB sitebcb API)")
    parser.add_argument("--from", dest="from_ym", default="202212",
                        help="First YYYYMM to download (default: 202212)")
    parser.add_argument("--to", dest="to_ym", default=this_month,
                        help=f"Last YYYYMM to download (default: {this_month})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    parser.add_argument("--all-types", action="store_true",
                        help="Also download COMBINADOS/CONSORCIOS/LIQUIDACAO")
    args = parser.parse_args()
    main(from_ym=args.from_ym, to_ym=args.to_ym, dry_run=args.dry_run, all_types=args.all_types)
