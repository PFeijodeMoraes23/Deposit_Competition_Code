## scrape_bcb_accounts.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-05-31
#
# Purpose: National / regional DEPOSIT-ACCOUNT COUNTS from BCB, for the
#          account-share-vs-volume-share (extensive-vs-intensive) diagnostic.
#
#   DATA-AVAILABILITY REALITY (verified 2026-05-31, important):
#     BCB does NOT expose deposit/payment-account *counts* through a clean public
#     API. Concretely:
#       - Olinda 'Estatisticas de Meios de Pagamento' (MPV) OData returns HTTP 500
#         server-side for the aggregate resources.
#       - DASFN per-institution endpoints only point to each bank's Open Finance
#         *branch* lists, not account counts.
#       - The SGS time-series API works but carries no depositor-count series.
#       - The account counts published in the Relatório de Cidadania Financeira
#         (RCF) and Relatório de Economia Bancária (REB) live in JS-rendered pages
#         / PDFs with no static download link.
#       - IF.Data (already pulled by scrape_1) is financial-statement data only;
#         it has no client/account count.
#       - Institution x municipality account counts exist only in restricted
#         supervisory documents (CADOC), available via a BCB data request.
#
#   WHAT THIS SCRIPT THEREFORE DOES:
#     (A) Best-effort: discovers + archives the latest RCF and REB PDFs from the
#         BCB pages, so the human can read the account-count tables. Archived to
#         BCB/Accounts/reports/. (Pages are JS-rendered, so this is often empty —
#         the script then logs the manual-download URLs.)
#     (B) Emits a SEEDED manual-augmentation CSV (manual_bcb_account_counts.csv)
#         in the project's long panel schema. Drop the RCF/REB national/regional
#         'número de contas de depósito / poupança / contas de pagamento pré-pagas'
#         figures here and they flow into the unified disclosure panel.
#
#   Outputs (under BCB/Accounts/):
#     reports/RCF_*.pdf, reports/REB_*.pdf — archived report PDFs (if discoverable)
#     manual_bcb_account_counts.csv        — seeded manual template (panel schema)
#
#   CLI:
#     python scrape_bcb_accounts.py
#     python scrape_bcb_accounts.py --no-reports   # skip PDF discovery
###────────────────────────────────────────────────────────────────────────────

import os
import sys
import re
import argparse
import logging

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

from utils.disclosure_common import (
    http_get, data_root, new_record, write_panel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("scrape_20")

BASE = data_root(__file__)
OUT_DIR = os.path.join(BASE, "BCB", "Accounts")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")
MANUAL_CSV = os.path.join(OUT_DIR, "manual_bcb_account_counts.csv")

RCF_PAGE = "https://www.bcb.gov.br/cidadaniafinanceira/relatoriocidadania"
REB_PAGE = "https://www.bcb.gov.br/publicacoes/relatorioeconomiabancaria"


def discover_reports() -> None:
    """Best-effort: pull any .pdf links the BCB pages expose (often none, as the
    pages are JS-rendered) and archive them. Logs the manual path if empty."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    found = 0
    for tag, page in (("RCF", RCF_PAGE), ("REB", REB_PAGE)):
        code, body = http_get(page)
        pdfs = re.findall(r'href="([^"]+\.pdf)"', body or "", re.I)
        pdfs = [p if p.startswith("http") else "https://www.bcb.gov.br" + p for p in pdfs]
        for url in list(dict.fromkeys(pdfs))[:4]:
            c, raw = http_get(url, binary=True, timeout=120)
            if raw:
                fn = os.path.join(REPORTS_DIR, f"{tag}_{os.path.basename(url.split('?')[0])}")
                with open(fn, "wb") as fh:
                    fh.write(raw)
                found += 1
                log.info(f"archived {tag} report -> {fn}")
    if not found:
        log.warning("No static RCF/REB PDF links exposed (JS-rendered pages). "
                    "Download manually for the account-count tables:")
        log.warning(f"   RCF: {RCF_PAGE}")
        log.warning(f"   REB: {REB_PAGE}")


def write_manual_template() -> None:
    """Seed the manual-augmentation CSV (panel schema) if it doesn't exist yet,
    with example rows showing the expected format. Never overwrites user edits."""
    if os.path.exists(MANUAL_CSV):
        log.info(f"manual template already present (not overwritten): {MANUAL_CSV}")
        return
    examples = [
        new_record(firm_key="ALL_BR", firm_name="Brazil (national)", cnpj_root="",
                   segment="", source="bcb_rcf", filing_form="RCF", period_year=2023,
                   period_quarter=None, period_label="2023", geo_scope="brazil",
                   metric="accounts", value=None, unit="count", currency="",
                   confidence="manual",
                   raw_context="EXAMPLE: nº contas de depósito à vista (RCF Tab X) — fill value",
                   source_url=RCF_PAGE, accession=""),
        new_record(firm_key="ALL_BR", firm_name="Brazil (national)", cnpj_root="",
                   segment="", source="bcb_rcf", filing_form="RCF", period_year=2023,
                   period_quarter=None, period_label="2023", geo_scope="brazil",
                   metric="accounts_prepaid", value=None, unit="count", currency="",
                   confidence="manual",
                   raw_context="EXAMPLE: nº contas de pagamento pré-pagas (type 5) — fill value",
                   source_url=REB_PAGE, accession=""),
    ]
    write_panel(examples, MANUAL_CSV)
    log.info(f"seeded manual template -> {MANUAL_CSV} (edit & re-run downstream merge)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-reports", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.no_reports:
        discover_reports()

    write_manual_template()
    log.info("DONE. Account counts: fill manual_bcb_account_counts.csv from RCF/REB; "
             "institution x municipality counts require a BCB CADOC data request.")


if __name__ == "__main__":
    main()
