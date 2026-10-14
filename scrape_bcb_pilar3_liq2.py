###─────────────────────────────────────────────────────────────────────────────
# scrape_bcb_pilar3_liq2.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Quarterly Pillar 3 NSFR disclosures (template LIQ2, "Indicador de Liquidez de
#   Longo Prazo") of the S1 banks, from the BCB open-data catalogue. LIQ2 splits
#   each bank's funding by residual maturity, which is the closest public proxy for
#   how much of the deposit base can leave within a quarter (the maturity share mu
#   in deposit_type_heterogeneity_notes.md, Section 4).
#
# DATA SOURCE
#   Catalogue (BCB, OData):
#     https://olinda.bcb.gov.br/olinda/servico/DASFN/versao/v1/odata/Recursos
#       ?$filter=Api eq 'pilar3' and Recurso eq '/liq2/{trimestre}'&$format=json
#   Each catalogue row points to the bank's own JSON endpoint (URLDados) for one
#   quarter (Argumento = 'YYYY-Q'). Template definition (columns and rows):
#     https://www.bcb.gov.br/content/estabilidadefinanceira/regulacao_prudencial_docs/
#       Relatorio_pilar_3/Tabela_LIQ2.pdf
#   Columns: a = no maturity, b = under 6 months, c = 6 months to under 1 year,
#   d = 1 year or more, e = weighted value; amounts in R$ thousand. Rows 4-6 are
#   retail funding (total, stable, less stable); row 7 is wholesale funding. Rows
#   12, 28-30 carry a single "bcd" column for some items.
#
# COVERAGE (catalogue as of 2026-10-12)
#   Itaú and Bradesco 2020-1 to 2023-3 (and 2025-4 on); Banco do Brasil 2020-1 to
#   2023-1; Caixa 2020-1 to 2022-1; BNDES 2020-1 to 2023-2; Fibra 2020-2. Santander's
#   entries point to PDF documents, not JSON, and are listed but not parsed.
#   Between 2023-06-30 and 2025-12-31 the banks published LIQ2 only as PDF on their
#   own sites (dataset notes), so those quarters are not in the catalogue.
#
# OUTPUTS (raw data area, nothing in the estimation pipeline reads them)
#   RAW/PILAR3/liq2/<cnpj>_<quarter>.json   one file per bank-quarter, as served
#   RAW/PILAR3/liq2_catalog.csv             the catalogue rows and fetch status
#   RAW/PILAR3/liq2_long.csv                cnpj, bank, quarter, row, col, value
#   RAW/PILAR3/liq2_retail_buckets.csv      rows 4-7 by bucket, with shares
#
# CLI
#   python scrape_bcb_pilar3_liq2.py             # fetch what is missing, rebuild tables
#   python scrape_bcb_pilar3_liq2.py --refresh   # re-download every JSON
###─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import logging
import re
import time
import urllib.parse
import urllib.request

try:
    from utils.venv_guard import ensure_project_venv
except ImportError:  # pragma: no cover
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd

from utils import paths

log = logging.getLogger("pilar3_liq2")
OUT = paths.RAW / "PILAR3"
JSON_DIR = OUT / "liq2"
# OData wants %20 for spaces (urlencode's '+' gets HTTP 400) and rejects $top here.
CATALOG_URL = ("https://olinda.bcb.gov.br/olinda/servico/DASFN/versao/v1/odata/Recursos?$filter="
               + urllib.parse.quote("Api eq 'pilar3' and Recurso eq '/liq2/{trimestre}'", safe="'/{}")
               + "&$format=json")
UA = {"User-Agent": "Mozilla/5.0 (research; deposit competition)"}
TIMEOUT = 60
ROW_LABELS = {
    1: "capital", 2: "regulatory capital", 3: "other capital instruments",
    4: "retail funding", 5: "retail funding: stable", 6: "retail funding: less stable",
    7: "wholesale funding", 8: "wholesale: operational deposits", 9: "wholesale: other",
    10: "interdependent liabilities", 11: "other liabilities",
    14: "total available stable funding",
}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


def parse_json_bytes(raw: bytes) -> dict | None:
    """Tolerate a UTF-8 BOM and trailing commas (Bradesco's files have one before a closing
    brace in row 27); return None when the payload is not a JSON object."""
    txt = raw.decode("utf-8-sig", errors="replace").strip()
    # Bradesco also serves a bare "." for an empty cell (row 32, column a): read it as null.
    repaired = re.sub(r":\s*\.\s*([,}])", r": null\1", re.sub(r",\s*([}\]])", r"\1", txt))
    for candidate in (txt, repaired):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return obj if isinstance(obj, dict) else None
    return None


def quarter_label(argumento: str) -> str:
    """'2023-3' as served by most banks; Santander serves a file name such as
    '21-08-23_191556_20210630_liq2.json', whose 8-digit date is the reference date."""
    if re.fullmatch(r"\d{4}-[1-4]", argumento):
        return argumento
    m = re.search(r"(20\d{2})(\d{2})(\d{2})_liq2", argumento)
    if m:
        return f"{m.group(1)}-{(int(m.group(2)) - 1) // 3 + 1}"
    # One Santander file carries only its publication date ('20-06-03_..._liq2.json'):
    # Pillar 3 is published after the quarter closes, so the reference is the prior quarter.
    m = re.match(r"(\d{2})-(\d{2})-\d{2}_", argumento)
    if m:
        y, q = 2000 + int(m.group(1)), (int(m.group(2)) - 1) // 3 + 1
        y, q = (y, q - 1) if q > 1 else (y - 1, 4)
        return f"{y}-{q}"
    return re.sub(r"[^\w.-]", "_", argumento)


def tidy(obj: dict, cnpj: str, bank: str, quarter: str) -> list[dict]:
    rows = []
    for k, v in obj.items():
        if not k.startswith("liq2_") or not isinstance(v, dict):
            continue
        tail = k[len("liq2_"):]
        if not tail.isdigit():
            continue
        for col, val in v.items():
            if col == "outros" or not isinstance(val, (int, float)):
                continue
            rows.append({"cnpj": cnpj, "bank": bank, "quarter": quarter,
                         "row": int(tail), "col": col, "value": float(val)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    cat = json.loads(fetch(CATALOG_URL).decode("utf-8-sig"))["value"]
    log.info("catalogue rows: %d", len(cat))
    status, long_rows = [], []
    for c in sorted(cat, key=lambda r: (r["NomeInstituicao"], r["Argumento"])):
        cnpj, bank, arg, url = c["CnpjInstituicao"], c["NomeInstituicao"], c["Argumento"], c["URLDados"]
        q = quarter_label(arg)
        safe_arg = re.sub(r"[^\w.-]", "_", arg)
        dest = JSON_DIR / f"{cnpj}_{q}_{safe_arg}.json"
        st = "cached"
        if a.refresh or not dest.exists():
            try:
                raw = fetch(url)
                dest.write_bytes(raw)
                st = "downloaded"
                time.sleep(0.3)
            except Exception as e:  # network or server error: record and move on
                status.append({"cnpj": cnpj, "bank": bank, "quarter": q, "url": url,
                               "status": f"error: {type(e).__name__}: {e}"[:200]})
                continue
        obj = parse_json_bytes(dest.read_bytes())
        if obj is None:
            st += "; not JSON (document or malformed)"
        else:
            got = [dict(r, source=arg) for r in tidy(obj, cnpj, bank, q)]
            long_rows += got
            st += f"; {len(got)} cells"
        status.append({"cnpj": cnpj, "bank": bank, "quarter": q, "url": url, "status": st})

    pd.DataFrame(status).to_csv(OUT / "liq2_catalog.csv", index=False, encoding="utf-8")
    long = pd.DataFrame(long_rows)
    if not long.empty:
        # Santander can publish several files for one quarter: keep the latest publication.
        last = long.groupby(["cnpj", "quarter"])["source"].transform("max")
        long = long[long["source"] == last].drop(columns="source")
    long.to_csv(OUT / "liq2_long.csv", index=False, encoding="utf-8")
    if long.empty:
        log.warning("no LIQ2 cells parsed")
        return 1

    w = long[long["row"].isin([4, 5, 6, 7]) & long["col"].isin(["a", "b", "c", "d"])]
    w = w.pivot_table(index=["cnpj", "bank", "quarter", "row"], columns="col", values="value",
                      aggfunc="first").reset_index()
    w["total_unweighted"] = w[["a", "b", "c", "d"]].sum(axis=1)
    for c in ("a", "b", "c", "d"):
        w[f"share_{c}"] = w[c] / w["total_unweighted"]
    w["row_label"] = w["row"].map(ROW_LABELS)
    w = w.rename(columns={"a": "no_maturity", "b": "under_6m", "c": "6m_to_1y", "d": "1y_or_more",
                          "share_a": "share_no_maturity", "share_b": "share_under_6m",
                          "share_c": "share_6m_to_1y", "share_d": "share_1y_or_more"})
    w.to_csv(OUT / "liq2_retail_buckets.csv", index=False, encoding="utf-8")
    n_ok = sum("cells" in s["status"] for s in status)
    log.info("bank-quarters parsed: %d of %d; outputs in %s", n_ok, len(status), OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
