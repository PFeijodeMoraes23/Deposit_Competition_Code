## scrape_22_fgc_statistics.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-05-31
#
# Purpose: FGC (Fundo Garantidor de Créditos) statistics — the number of insured
#          depositors (CPFs/CNPJs) and insured value BY DEPOSIT-SIZE BRACKET
#          (R$250k guarantee ceiling). This is the single richest COUNT source for
#          the extensive-vs-intensive margin (V_Main.tex app:extensions:intensive):
#          a depositor count AND a conditional-size distribution that would identify
#          Sigma^q, the intensive-margin heterogeneity flagged as fragile.
#
#   DATA-AVAILABILITY REALITY (verified 2026-05-31):
#     FGC publishes the bracketed insured-deposit statistics only inside JS-rendered
#     dashboards and annual-report PDFs (no static data file is linked on
#     /publicações-e-estatísticas). INSTITUTION-LEVEL granularity (number of
#     depositors per conglomerate) is reported to FGC but released only on request.
#     So the granular object is a DATA REQUEST, not a scrape.
#
#   WHAT THIS SCRIPT DOES:
#     (A) Best-effort: harvest + archive any PDF/XLS the FGC statistics/publications
#         pages expose (annual reports carry the national bracket tables).
#     (B) Emit a SEEDED manual-augmentation CSV (manual_fgc_brackets.csv) in a
#         bracket schema: drop the 'nº de CPFs/CNPJs' and 'valor garantido' per
#         R$ bracket from the FGC report/dashboard and they feed the diagnostic.
#
#   Outputs (under FGC/):
#     reports/*.pdf                  — archived FGC reports (if discoverable)
#     manual_fgc_brackets.csv        — seeded manual bracket template
#
#   CLI: python scrape_22_fgc_statistics.py
###────────────────────────────────────────────────────────────────────────────

import os
import sys
import re
import csv
import urllib.parse
import logging

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

from utils.disclosure_common import http_get, data_root, now_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("scrape_22")

BASE = data_root(__file__)
OUT_DIR = os.path.join(BASE, "FGC")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")
MANUAL_CSV = os.path.join(OUT_DIR, "manual_fgc_brackets.csv")

FGC_PAGES = [
    "https://www.fgc.org.br/" + urllib.parse.quote("publicações-e-estatísticas"),
    "https://www.fgc.org.br/" + urllib.parse.quote("pesquisa-e-base-de-dados"),
]

# Bracket schema for the manual template (one row per year x bracket x scope).
BRACKET_COLUMNS = [
    "year", "scope", "institution", "cnpj_root",
    "bracket_min_brl", "bracket_max_brl",
    "n_depositors", "value_insured_brl", "source_url", "note", "retrieved_at",
]


def harvest_reports() -> int:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    found = 0
    for page in FGC_PAGES:
        code, body = http_get(page)
        if not body:
            continue
        files = re.findall(r'href="([^"]+\.(?:pdf|xlsx|xls|csv))"', body, re.I)
        files = [f if f.startswith("http") else "https://www.fgc.org.br/" + f.lstrip("/")
                 for f in files]
        for url in list(dict.fromkeys(files))[:8]:
            c, raw = http_get(url, binary=True, timeout=120)
            if raw:
                fn = os.path.join(REPORTS_DIR, os.path.basename(url.split("?")[0]))
                with open(fn, "wb") as fh:
                    fh.write(raw)
                found += 1
                log.info(f"archived FGC file -> {fn}")
    return found


def seed_template() -> None:
    if os.path.exists(MANUAL_CSV):
        log.info(f"manual bracket template present (not overwritten): {MANUAL_CSV}")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = [
        dict(year=2023, scope="national", institution="ALL", cnpj_root="",
             bracket_min_brl=0, bracket_max_brl=5000, n_depositors="",
             value_insured_brl="", source_url=FGC_PAGES[0],
             note="EXAMPLE bracket — fill nº CPFs/CNPJs and valor garantido", retrieved_at=now_iso()),
        dict(year=2023, scope="national", institution="ALL", cnpj_root="",
             bracket_min_brl=5000, bracket_max_brl=20000, n_depositors="",
             value_insured_brl="", source_url=FGC_PAGES[0], note="EXAMPLE bracket",
             retrieved_at=now_iso()),
        dict(year=2023, scope="national", institution="ALL", cnpj_root="",
             bracket_min_brl=20000, bracket_max_brl=250000, n_depositors="",
             value_insured_brl="", source_url=FGC_PAGES[0],
             note="EXAMPLE bracket (up to R$250k ceiling)", retrieved_at=now_iso()),
    ]
    with open(MANUAL_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=BRACKET_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    log.info(f"seeded FGC bracket template -> {MANUAL_CSV}")


def main():
    n = harvest_reports()
    if not n:
        log.warning("No static FGC data files exposed (JS-rendered). Read the bracket "
                    "tables manually from the annual report / statistics dashboard:")
        for p in FGC_PAGES:
            log.warning(f"   {urllib.parse.unquote(p)}")
        log.warning("Institution-level depositor counts: request from FGC "
                    "(diretoria de estatística) — not publicly downloadable.")
    seed_template()
    log.info("DONE. FGC brackets identify Sigma^q (conditional size distribution); "
             "fill manual_fgc_brackets.csv from the report.")


if __name__ == "__main__":
    main()
