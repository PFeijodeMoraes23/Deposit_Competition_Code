## scrape_11_incumbent_clients.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-05-31
#
# Purpose: CLIENT / ACCOUNT-HOLDER counts (correntistas, active clients) for the
#          incumbent B-firms — Itaú, Bradesco, Santander Brasil, Banco do Brasil.
#          These are the denominator for the incumbents' account-share-vs-volume-
#          share comparison against the digital entrants.
#
#   WHY THIS IS A TEMPLATE, NOT A PURE SCRAPE:
#     Incumbent DEPOSITS are already covered structurally — Itaú/Bradesco/Santander
#     via SEC XBRL (scrape_9) and CVM DFP/ITR (scrape_15), and Banco do Brasil via
#     CVM (scrape_15). What XBRL/CVM do NOT carry is the CLIENT COUNT. Those appear
#     only in quarterly earnings releases / 'Dados Suplementares' spreadsheets on
#     each bank's IR site, in non-stable per-quarter URLs and varied layouts (a
#     reliable scrape needs per-bank, per-quarter parsers).
#
#   WHAT THIS SCRIPT DOES:
#     (A) Best-effort: fetch each incumbent's IR landing page and archive any
#         linked 'Dados Suplementares' / earnings spreadsheets it exposes.
#     (B) Seed a manual-augmentation CSV (manual_incumbent_clients.csv) in the
#         project panel schema, one row per bank x quarter, to fill 'active
#         clients' / 'correntistas' from the releases. Joins to the deposit panel
#         on cnpj_root (the prudential-conglomerate key).
#
#   Outputs (under FirmDisclosures/Incumbents/):
#     reports/<firm>_*.xlsx|pdf       — archived IR files (if discoverable)
#     manual_incumbent_clients.csv     — seeded manual template (panel schema)
#
#   CLI: python scrape_11_incumbent_clients.py
###────────────────────────────────────────────────────────────────────────────

import os
import sys
import re
import logging

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

from utils.disclosure_common import http_get, data_root, new_record, write_panel
from utils.firm_registry import load_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("scrape_11")

BASE = data_root(__file__)
OUT_DIR = os.path.join(BASE, "FirmDisclosures", "Incumbents")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")
MANUAL_CSV = os.path.join(OUT_DIR, "manual_incumbent_clients.csv")

# IR landing pages for best-effort spreadsheet discovery.
IR_PAGES = {
    "itau":         "https://www.itau.com.br/relacoes-com-investidores/resultados-e-relatorios/central-de-resultados/",
    "bradesco":     "https://www.bradescori.com.br/informacoes-ao-mercado/relatorios-e-planilhas/planilhas/",
    "santander_br": "https://www.santander.com.br/ri/informacoes-financeiras/central-de-resultados",
    "bb":           "https://ri.bb.com.br/informacoes-financeiras/central-de-resultados/",
}
INCUMBENTS = ["itau", "bradesco", "santander_br", "bb"]


def harvest_ir(firm_key: str, url: str) -> int:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    code, body = http_get(url)
    if not body:
        log.info(f"[{firm_key}] IR page HTTP {code} (no archive)")
        return 0
    files = re.findall(r'href="([^"]+\.(?:xlsx|xls|pdf))"', body, re.I)
    found = 0
    for u in list(dict.fromkeys(files))[:4]:
        full = u if u.startswith("http") else re.sub(r"(https?://[^/]+).*", r"\1", url) + u
        c, raw = http_get(full, binary=True, timeout=120)
        if raw:
            fn = os.path.join(REPORTS_DIR, f"{firm_key}_{os.path.basename(full.split('?')[0])}")
            with open(fn, "wb") as fh:
                fh.write(raw)
            found += 1
            log.info(f"[{firm_key}] archived -> {fn}")
    return found


def seed_template() -> None:
    if os.path.exists(MANUAL_CSV):
        log.info(f"manual template present (not overwritten): {MANUAL_CSV}")
        return
    reg = {f["firm_key"]: f for f in load_registry(BASE)}
    rows = []
    for k in INCUMBENTS:
        f = reg.get(k, {"display_name": k, "cnpj_root": "", "segment": "incumbent"})
        rows.append(new_record(
            firm_key=k, firm_name=f.get("display_name", k), cnpj_root=f.get("cnpj_root", ""),
            segment="incumbent", source="incumbent_release", filing_form="earnings_release",
            period_year=2024, period_quarter=4, period_label="4Q2024", geo_scope="brazil",
            metric="clients_active", value=None, unit="count", currency="", confidence="manual",
            raw_context="EXAMPLE — fill active clients / correntistas from Dados Suplementares",
            source_url=IR_PAGES.get(k, ""), accession=""))
    write_panel(rows, MANUAL_CSV)
    log.info(f"seeded incumbent client template -> {MANUAL_CSV}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for k in INCUMBENTS:
        total += harvest_ir(k, IR_PAGES[k])
    if not total:
        log.warning("No static IR spreadsheets discovered (JS-rendered IR portals). "
                    "Download 'Dados Suplementares' manually and fill the template:")
        for k in INCUMBENTS:
            log.warning(f"   {k:14s} {IR_PAGES[k]}")
    seed_template()
    log.info("DONE. Incumbent deposits come from scrape_9 (SEC) / scrape_15 (CVM, incl. BB); "
             "client counts go in manual_incumbent_clients.csv.")


if __name__ == "__main__":
    main()
