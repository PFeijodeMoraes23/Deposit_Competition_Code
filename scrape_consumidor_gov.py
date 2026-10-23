###─────────────────────────────────────────────────────────────────────────────
# scrape_consumidor_gov.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Download the consumidor.gov.br open data: complaints finalised on the platform, one row
#   per complaint, with the consumer's UF and city and the company's trade name. It is the
#   only public source found that crosses a named financial institution with the consumer's
#   municipality (a usage proxy, not a customer count; request S4, whose NOTES.md in
#   DATA_RAW/_requests/devo_2026-10-20/S4_digital_local_presence/ documents the fields).
#
#   Not a pipeline step. Files are saved exactly as served; nothing is merged or recoded.
#
# DATA SOURCE  (found in the open-data page's own JavaScript, datatables-65064.min.js)
#   page      https://www.consumidor.gov.br/pages/dadosabertos/externo/
#   file list https://www.consumidor.gov.br/pages/publicacao/externo/publicacoes.json?indicadorTipoPublicacao=2
#   download  https://www.consumidor.gov.br/pages/publicacao/externo/<codigo>/download
#   The complaint files are the entries titled "Dados - ...": monthly zips from 2019
#   (finalizadas_YYYY-MM.zip, or YYYY-MM.zip before 2021), semester/annual zips for 2014-2018.
#   Encodings differ by year (UTF-8 recently, Latin-1 in 2019); separator ';'. The file
#   list itself is served as UTF-8 or as cp1252 (both seen), so both are accepted.
#   The site's firewall answers HTTP 503 to roughly one request in three: retried.
#
# FIELDS (dictionary v2, checked on 2019-09 and 2026-09): Regiao, UF, Cidade (city NAME, no
#   IBGE code), Sexo, Faixa Etaria, Data Finalizacao, Tempo Resposta, Nome Fantasia (trade
#   name, no CNPJ), Segmento de Mercado, Area, Assunto, Grupo Problema, Problema, ...
#
# OUTPUT  (default utils.raw_paths.CONSUMIDOR_GOV, or --out)
#   publicacoes_<date>.json   the file list as served
#   manifest.csv              codigo, titulo, nomeArquivo, period, size, dates, url
#   data/<nomeArquivo>        complaint files requested with --periods or --all
#   docs/<nomeArquivo>        the data dictionary and the open-data guide (always)
#   A file already present with the listed size is not downloaded again.
#
# USAGE
#   python scrape_consumidor_gov.py                         # list + docs only
#   python scrape_consumidor_gov.py --periods 2026-09 2019-09
#   python scrape_consumidor_gov.py --years 2016-2024       # monthly AND semester files of those years
#   python scrape_consumidor_gov.py --all                   # every complaint file (~GBs)
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import time
import zipfile

from utils import raw_paths
from utils.bcb_http import retry_session

SITE = "https://www.consumidor.gov.br/pages/publicacao/externo"
LIST_URL = SITE + "/publicacoes.json?indicadorTipoPublicacao=2"
DOWNLOAD_URL = SITE + "/{codigo}/download"
DEFAULT_OUT = raw_paths.CONSUMIDOR_GOV


def period_of(nome: str) -> str:
    """'YYYY-MM' for a monthly file, else '' (semester/annual files keep their name)."""
    m = re.search(r"(\d{4})-(\d{2})(?!\d)", nome)
    return f"{m.group(1)}-{m.group(2)}" if m and "sem" not in nome.lower() else ""


def year_of(nome: str) -> int | None:
    """The year a complaint file covers (the first 20YY in its name), for monthly and semester files."""
    m = re.search(r"(20\d\d)", nome)
    return int(m.group(1)) if m else None


def get(session, url: str, tries: int = 6):
    """GET with extra retries on the firewall's 503 (on top of the session's own retries)."""
    for t in range(tries):
        r = session.get(url, timeout=600)
        if r.status_code == 200:
            return r
        time.sleep(5 * (t + 1))
    r.raise_for_status()
    return r


def save(session, entry: dict, folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, entry["nomeArquivo"])
    if os.path.exists(target) and os.path.getsize(target) == int(entry["tamanhoArquivo"]):
        return "present"
    r = get(session, DOWNLOAD_URL.format(codigo=entry["codigo"]))
    if len(r.content) != int(entry["tamanhoArquivo"]):
        return f"FAILED size {len(r.content)} != listed {entry['tamanhoArquivo']}"
    tmp = target + ".part"
    with open(tmp, "wb") as f:
        f.write(r.content)
    if target.lower().endswith(".zip") and not zipfile.is_zipfile(tmp):
        return "FAILED not a zip (kept as .part)"
    os.replace(tmp, target)
    return "saved"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--periods", nargs="*", default=[], help="monthly periods YYYY-MM to download")
    ap.add_argument("--years", default="", help="year range YYYY-YYYY: every complaint file of those years")
    ap.add_argument("--all", action="store_true", help="download every complaint file")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    session = retry_session(total=4, backoff=3.0)
    today = dt.date.today().isoformat()

    raw = get(session, LIST_URL).content
    with open(os.path.join(args.out, f"publicacoes_{today}.json"), "wb") as f:
        f.write(raw)
    try:
        entries = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        entries = json.loads(raw.decode("cp1252"))

    data = [e for e in entries if str(e.get("titulo", "")).startswith("Dados")]
    docs = [e for e in entries if e["nomeArquivo"].lower().endswith(".pdf")]
    with open(os.path.join(args.out, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["codigo", "titulo", "nomeArquivo", "period", "tamanho_bytes",
                    "dataPublicacao", "dataAlteracao", "url_download"])
        for e in entries:
            w.writerow([e["codigo"], e["titulo"], e["nomeArquivo"], period_of(e["nomeArquivo"]),
                        e["tamanhoArquivo"], e["dataPublicacao"], e["dataAlteracao"],
                        DOWNLOAD_URL.format(codigo=e["codigo"])])
    print(f"file list: {len(entries)} entries, {len(data)} complaint files, {len(docs)} documents")

    for e in docs:
        print(f"  doc   {e['nomeArquivo']}: {save(session, e, os.path.join(args.out, 'docs'))}")

    if args.all:
        wanted = data
    elif args.years:
        y0, y1 = (int(x) for x in args.years.split("-"))
        wanted = [e for e in data if (year_of(e["nomeArquivo"]) or 0) in range(y0, y1 + 1)]
    else:
        wanted = [e for e in data if period_of(e["nomeArquivo"]) in set(args.periods)]
    missing = set(args.periods) - {period_of(e["nomeArquivo"]) for e in wanted}
    for e in wanted:
        print(f"  data  {e['nomeArquivo']}: {save(session, e, os.path.join(args.out, 'data'))}")
        time.sleep(2)
    if missing:
        print(f"periods not in the file list: {sorted(missing)}")


if __name__ == "__main__":
    main()
