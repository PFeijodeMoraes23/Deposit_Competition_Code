## scrape_11_incumbent_clients.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-06-01
#
# Purpose: CLIENT / ACTIVE-ACCOUNT COUNTS for incumbents and mid/digital banks
#          that have CVM filings but no SEC filer (Pan, BMG, Banrisul, Banco do
#          Brasil), using a tiered extraction cascade:
#
#   Tier 1  (structural)   — CVM DFP/ITR deposits (handled by scrape_15, not here)
#   Tier 2  (text PDF)     — CVM IPE earnings-release PDFs via pdfplumber regex:
#                            category "Comunicado ao Mercado" / "Dados Econômico-
#                            Financeiros" + assunto matching "resultado/apresenta/
#                            release/institucional". Works for text-heavy releases
#                            (BB confirmed; Banrisul/Pan text docs).
#   Tier 3  (vision-LLM)   — Fallback for slide-deck PDFs that defeat text
#                            extraction (Pan APIMEC, BMG "Apresentação Institucional").
#                            Sends rendered page images to Claude claude-haiku-4-5;
#                            GATED on ANTHROPIC_API_KEY being set.
#
#   Regex patterns for Tier 2 (Brazilian banking terminology):
#     "(\d[\d.,]+)\s*(mil|mi|milhão|milhões|mn|bi)?\s*(de\s+)?(clientes|correntistas|
#       contas ativas|usuários ativos|base de clientes|clientes ativos)"
#
#   Output: FirmDisclosures/Incumbents/incumbent_client_counts.csv  (long panel)
#   Raw:    FirmDisclosures/Incumbents/raw/<firm>/<yyyymmdd>_<protocol>.pdf
#
#   CLI:
#     python scrape_11_incumbent_clients.py                   # all target firms
#     python scrape_11_incumbent_clients.py --firms bb,pan    # specific firms
#     python scrape_11_incumbent_clients.py --since 2018 --no-vision
#     python scrape_11_incumbent_clients.py --vision-test bb  # test Tier-4 on one firm
###─────────────────────────────────────────────────────────────────────────────

import os
import sys
import io
import csv
import re
import zipfile
import argparse
import logging
import datetime as dt

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pdfplumber
from utils.disclosure_common import http_get, data_root, new_record, write_panel
from utils.firm_registry import load_registry, _norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("scrape_11")

BASE = data_root(__file__)
OUT_DIR = os.path.join(BASE, "FirmDisclosures", "Incumbents")
RAW_DIR = os.path.join(OUT_DIR, "raw")
OUT_CSV = os.path.join(OUT_DIR, "incumbent_client_counts.csv")

CVM_IPE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_{year}.zip"
IPE_CACHE = os.path.join(OUT_DIR, "raw", "_ipe_index")

# CVM IPE categories that may carry client counts
RESULT_CATS = {"Comunicado ao Mercado", "Dados Econômico-Financeiros"}
RESULT_KWORDS = {"resultado", "release", "apresenta", "institucional", "desempenho",
                 "trimest", "semest", "anual", "4t", "3t", "2t", "1t"}

# Regex pattern for client counts in PT-BR releases
_COUNT_RX = re.compile(
    r"(\d[\d.,]+)\s*(mil|mi|milhão|milhões|mn|bi|bilhão|bilhões)?\s*"
    r"(?:de\s+)?(clientes\s+ativos|clientes\s+totais|clientes|correntistas|"
    r"contas\s+ativas|usuários\s+ativos|base\s+de\s+clientes|contas\s+de\s+pagamento)",
    re.I,
)
_MULT = {"mil": 1e3, "mi": 1e6, "milhão": 1e6, "milhões": 1e6, "mn": 1e6,
         "bi": 1e9, "bilhão": 1e9, "bilhões": 1e9}

# Target firms for this scraper (CVM filers without an adequate SEC count source).
# Expand by adding to utils/firm_registry.py.
TARGET_KEYS = {"bb", "banrisul", "brb", "pan", "bmg", "caixa"}


def _load_ipe_index(year: int) -> list[dict]:
    """Download and cache the CVM IPE index for `year`. Returns list of dicts."""
    os.makedirs(IPE_CACHE, exist_ok=True)
    cache_path = os.path.join(IPE_CACHE, f"ipe_{year}.csv")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        with open(cache_path, encoding="latin-1") as fh:
            return list(csv.DictReader(fh, delimiter=";"))
    code, raw = http_get(CVM_IPE_URL.format(year=year), binary=True, timeout=180)
    if not raw:
        log.warning(f"IPE index {year}: HTTP {code}")
        return []
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
        data = z.read(z.namelist()[0]).decode("latin-1")
    except Exception as e:
        log.warning(f"IPE index {year} unzip error: {e}")
        return []
    with open(cache_path, "w", encoding="latin-1") as fh:
        fh.write(data)
    return list(csv.DictReader(io.StringIO(data), delimiter=";"))


def _is_results_doc(row: dict) -> bool:
    cat = row.get("Categoria", "")
    assunto = row.get("Assunto", "").lower()
    if cat not in RESULT_CATS:
        return False
    return any(k in assunto for k in RESULT_KWORDS)


def _parse_count(line: str) -> float | None:
    """Extract a client count from a text line. Returns absolute float or None."""
    m = _COUNT_RX.search(line)
    if not m:
        return None
    raw = m.group(1).replace(".", "").replace(",", ".")
    try:
        num = float(raw)
    except ValueError:
        return None
    unit = (m.group(2) or "").lower()
    val = num * _MULT.get(unit, 1.0)
    if not (1e6 <= val <= 5e8):  # plausibility: 1M–500M (all registry banks exceed 1M clients)
        return None
    return val


def _extract_tier2(pdf_bytes: bytes) -> tuple[float | None, str]:
    """Try pdfplumber text extraction. Returns (value, context_quote)."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:12]:
                text = page.extract_text() or ""
                for line in text.split("\n"):
                    val = _parse_count(line)
                    if val is not None:
                        return val, line.strip()[:140]
    except Exception as e:
        log.debug(f"Tier-2 pdfplumber error: {e}")
    return None, ""


def _extract_tier4(pdf_bytes: bytes, firm_name: str, period: str) -> tuple[float | None, str]:
    """Vision-LLM fallback. Returns (value, source_quote)."""
    try:
        from utils.vision_extractor import extract_metric_from_pdf
    except ImportError:
        return None, ""
    result = extract_metric_from_pdf(
        pdf_bytes, firm=firm_name,
        metric="total active clients or correntistas",
        period=period)
    if result and result.get("value"):
        return float(result["value"]), result.get("source_quote", "")[:140]
    return None, ""


def _ref_to_period(dt_str: str) -> tuple[int | None, int | None, str]:
    """Parse 'YYYY-MM-DD' to (year, quarter, label)."""
    try:
        y, m = int(dt_str[:4]), int(dt_str[5:7])
        q = (m - 1) // 3 + 1
        return y, q, f"{q}Q{y}"
    except Exception:
        return None, None, ""


def scrape_firm(firm: dict, since: int, until: int, use_vision: bool) -> list[dict]:
    cnpj_root = firm["cnpj_root"]
    if not cnpj_root:
        log.info(f"[{firm['firm_key']}] no CNPJ root — skipping IPE search")
        return []
    # Note: "00000000" is Banco do Brasil's legitimate 8-digit CNPJ root
    # (CNPJ 00.000.000/0001-91), so do NOT skip it.
    firm_raw = os.path.join(RAW_DIR, firm["firm_key"])
    os.makedirs(firm_raw, exist_ok=True)
    records = []
    seen_periods = set()

    for year in range(since, until + 1):
        index = _load_ipe_index(year)
        # match by CNPJ_Companhia first 8 digits
        matches = []
        for row in index:
            cnpj_cvm = re.sub(r"\D", "", row.get("CNPJ_Companhia", ""))[:8].zfill(8)
            if cnpj_cvm == cnpj_root and _is_results_doc(row):
                matches.append(row)
        if not matches:
            continue
        log.info(f"[{firm['firm_key']}] {year}: {len(matches)} results docs in IPE")

        for row in matches:
            dt_str = row.get("Data_Referencia", "")
            yr, q, period_label = _ref_to_period(dt_str)
            if not yr:
                continue
            key = (yr, q)
            if key in seen_periods:
                continue

            url = row.get("Link_Download", "")
            if not url:
                continue
            code, pdf_bytes = http_get(url, binary=True, timeout=90)
            if not pdf_bytes or len(pdf_bytes) < 500:
                continue
            if pdf_bytes[:4] != b"%PDF":
                log.debug(f"[{firm['firm_key']}] {dt_str}: not a PDF")
                continue

            # archive
            proto = re.sub(r"\W", "_", row.get("Protocolo_Entrega", dt_str))
            cache_path = os.path.join(firm_raw, f"{dt_str}_{proto}.pdf")
            if not os.path.exists(cache_path):
                try:
                    with open(cache_path, "wb") as fh:
                        fh.write(pdf_bytes)
                except Exception:
                    pass

            # Tier 2
            val, ctx = _extract_tier2(pdf_bytes)
            tier = 2
            # Tier 4 fallback
            if val is None and use_vision:
                val, ctx = _extract_tier4(pdf_bytes, firm["display_name"], period_label)
                tier = 4
            if val is None:
                continue

            seen_periods.add(key)
            records.append(new_record(
                firm_key=firm["firm_key"], firm_name=firm["display_name"],
                cnpj_root=cnpj_root, segment=firm["segment"],
                source=f"cvm_ipe_tier{tier}",
                filing_form="Comunicado ao Mercado",
                period_year=yr, period_quarter=q,
                period_label=period_label, geo_scope="brazil",
                metric="clients_active", value=val,
                unit="count", currency="", confidence=("medium" if tier == 2 else "low"),
                raw_context=ctx, source_url=url,
                accession=row.get("Protocolo_Entrega", ""),
            ))
            log.info(f"[{firm['firm_key']}] {period_label}: {val/1e6:.1f}M clients "
                     f"(Tier {tier}) — {ctx[:60]}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2018)
    ap.add_argument("--until", type=int, default=dt.date.today().year)
    ap.add_argument("--firms", type=str, default="")
    ap.add_argument("--no-vision", action="store_true", help="Disable Tier-4 vision-LLM")
    ap.add_argument("--vision-test", type=str, default="",
                    help="firm_key: test vision tier on that firm's first PDF only")
    args = ap.parse_args()

    reg = {f["firm_key"]: f for f in load_registry(BASE)}
    want = {x.strip() for x in args.firms.split(",") if x.strip()} or TARGET_KEYS
    firms = [f for k, f in reg.items() if k in want]

    if args.vision_test:
        f = reg.get(args.vision_test)
        if not f:
            print(f"Unknown firm: {args.vision_test}"); return
        index = _load_ipe_index(dt.date.today().year)
        row = next((r for r in index
                    if re.sub(r"\D","",r.get("CNPJ_Companhia",""))[:8].zfill(8)==f["cnpj_root"]
                    and _is_results_doc(r)), None)
        if not row:
            print("No IPE results doc found"); return
        _, pdf_bytes = http_get(row["Link_Download"], binary=True, timeout=90)
        print(f"PDF bytes: {len(pdf_bytes) if pdf_bytes else 0}")
        if pdf_bytes and pdf_bytes[:4] == b"%PDF":
            from utils.vision_extractor import extract_metric_from_pdf
            r = extract_metric_from_pdf(pdf_bytes, f["display_name"],
                                         metric="total active clients or correntistas",
                                         period=row.get("Data_Referencia",""))
            print("Vision result:", r)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    use_vision = not args.no_vision
    if use_vision and not os.environ.get("ANTHROPIC_API_KEY"):
        log.warning("ANTHROPIC_API_KEY not set — Tier-4 vision disabled for this run. "
                    "Set the env var to enable vision fallback for slide-deck PDFs.")
        use_vision = False

    records = []
    for f in firms:
        log.info(f"=== {f['firm_key']} (CNPJ root {f['cnpj_root']}) ===")
        records += scrape_firm(f, args.since, args.until, use_vision)

    n = write_panel(records, OUT_CSV)
    log.info(f"Wrote {n} records -> {OUT_CSV}")

    if records:
        import pandas as pd
        df = pd.DataFrame(records)
        print("\n--- client counts by firm x period ---")
        print(df.groupby(["firm_key","segment"])["period_label"].agg(
            list).apply(lambda x: sorted(x)).to_string())


if __name__ == "__main__":
    main()
