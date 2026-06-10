"""
scrape_15_cosif_download.py
============================
Download missing COSIF balancete CSV ZIPs (BANCOS, SOCIEDADES, COOPERATIVAS,
and optionally the other categories) for months not already on disk, then
re-run scrape_12 to extend the fee panel.

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
  python scrape_15_cosif_download.py
  python scrape_15_cosif_download.py --from 202212 --to 202512
  python scrape_15_cosif_download.py --dry-run
  python scrape_15_cosif_download.py --all-types   # include the other categories
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from datetime import date

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from utils import paths
from utils import refresh
_REPO     = Path(__file__).resolve().parents[2]
COSIF_RAW = paths.COSIF_RAW

API      = "https://www.bcb.gov.br/api/servico/sitebcb/Documentos/byListGuid"
GUID     = "a11917e4-c729-4259-bd4e-0266827b6acd"
SITE     = "https://www.bcb.gov.br"

# CMS folder -> COSIF document TYPE (filename suffix). The first three are the
# ones scrape_12 consumes; the rest are downloaded only with --all-types.
FOLDERS_CORE = {
    "/Bancos":                  "BANCOS",
    "/Sociedades":              "SOCIEDADES",
    "/Cooperativas-de-credito": "COOPERATIVAS",
}
FOLDERS_EXTRA = {
    "/Conglomerados-prudenciais":    "BLOPRUDENCIAL",
    "/Conglomerados-financeiros":    "COMBINADOS",
    "/Administradoras-de-consorcios": "CONSORCIOS",
    "/Instituicoes-em-regime-especial": "LIQUIDACAO",
}

# Accept only canonical names: <YYYYMM><TYPE>.csv.zip  (rejects malformed
# duplicates the API occasionally lists, e.g. "...zip.csv.zip").
_NAME_RE = re.compile(r"^(\d{6})[A-Z]+\.csv\.zip$", re.I)


def _list_folder(session: requests.Session, folder: str) -> list[dict]:
    r = session.get(API, params={
        "tronco": "estabilidadefinanceira", "guidLista": GUID,
        "ordem": "DataDocumento desc", "pasta": folder}, timeout=60)
    r.raise_for_status()
    return r.json().get("conteudo", [])


def main(from_ym: str, to_ym: str, dry_run: bool = False, all_types: bool = False) -> None:
    COSIF_RAW.mkdir(parents=True, exist_ok=True)
    folders = dict(FOLDERS_CORE)
    if all_types:
        folders.update(FOLDERS_EXTRA)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    to_download: list[tuple[str, Path, str]] = []
    for folder in folders:
        try:
            items = _list_folder(session, folder)
        except Exception as exc:
            log.warning("Could not list %s: %s", folder, exc)
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
                log.warning("  not a ZIP, skipping: %s", nome)
                continue
            dest.write_bytes(r.content)
            log.info("  -> %d KB", dest.stat().st_size // 1024)
        except Exception as exc:
            log.warning("  FAILED: %s", exc)
        time.sleep(0.5)

    log.info("Done. Re-run scrape_12_cosif_service_fees.py to rebuild the fee panel.")


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
                        help="Also download BLOPRUDENCIAL/COMBINADOS/CONSORCIOS/LIQUIDACAO")
    args = parser.parse_args()
    main(from_ym=args.from_ym, to_ym=args.to_ym, dry_run=args.dry_run, all_types=args.all_types)
