###─────────────────────────────────────────────────────────────────────────────
# scrape_estban_branches.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Download the ESTBAN per-BRANCH files (Estatística Bancária, Documento 4500, "por agência"):
#   one row per branch per month, with UF, CODMUN, MUNICIPIO, CNPJ (8-digit root),
#   NOME_INSTITUICAO, AGENCIA (branch code) and the same VERBETE_* balance-sheet lines as the
#   municipality-level files the pipeline uses (scrape_bcb_estban_ifdata.py).
#
#   Not a pipeline step, and no script reads these files yet. Files are saved exactly as
#   served (zips, not extracted). CODMUN is the BCB's municipality code, not the IBGE code;
#   mapping it is a later coding step. Only institutions with branches appear, so digital
#   firms are absent or appear as one headquarters row.
#
# DATA SOURCE
#   page     https://www.bcb.gov.br/estatisticas/estatisticabancariamunicipios
#   listing  BCB CMS listing (utils.bcb_http.cms_documents), tronco "estatisticas", the
#            guidLista shared with the municipality files, pasta "agencia"
#   files    https://www.bcb.gov.br/content/estatisticas/estatistica_bancaria_estban/agencia/
#            {YYYYMM}_ESTBAN_AG.csv.zip from 2023-02, {YYYYMM}_ESTBAN_AG.ZIP before, and two
#            months served as a plain CSV (202212, 202301: {YYYYMM}_ESTBAN_AG.csv). The
#            listing's Url is used as served, never a built name.
#   A download is accepted only if its size equals the listing's Tamanho and it is a valid
#   zip (zip names) or starts with the ESTBAN header (plain CSV).
#
# OUTPUT  (default utils.raw_paths.BCB_ESTBAN_AGENCIA, or --out)
#   listing_<date>.json   the listing as served
#   <Nome>                one zip per month, as served
#   manifest.csv          yyyymm, nome, url, accessed, bytes, sha256, status
#   A month already present as a valid zip is not downloaded again unless --force.
#
# USAGE
#   python scrape_estban_branches.py                      # 201301 to the latest month
#   python scrape_estban_branches.py --start 202501 --force
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import time
import zipfile

from utils import raw_paths
from utils import bcb_http

BCB_SITE = "https://www.bcb.gov.br"
ESTBAN_GUID = "f6391806-fd85-43af-acf1-c86d5b8dd6df"   # guidLista for municipio + agencia files
REFERER = BCB_SITE + "/estatisticas/estatisticabancariamunicipios"
DEFAULT_OUT = raw_paths.BCB_ESTBAN_AGENCIA


def valid(path: str, item: dict) -> bool:
    """The file on disk is the one the listing describes: same size, and a valid zip (zip
    names) or a CSV starting with the ESTBAN header (plain-CSV months)."""
    if not os.path.exists(path):
        return False
    listed = str(item.get("Tamanho", "")).strip()
    if listed.isdigit() and os.path.getsize(path) != int(listed):
        return False
    if item["Nome"].lower().endswith(".zip"):
        return zipfile.is_zipfile(path)
    with open(path, "rb") as f:
        return f.read(6) == b"ESTBAN"


def main() -> None:
    ap = argparse.ArgumentParser(description="ESTBAN per-branch files, raw")
    ap.add_argument("--start", default="201301", help="first YYYYMM (default 201301)")
    ap.add_argument("--end", default="999912", help="last YYYYMM (default: latest listed)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true", help="re-download months already present")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    session = bcb_http.retry_session(total=4, backoff=3.0, headers={"Referer": REFERER})
    today = dt.date.today().isoformat()

    items = bcb_http.cms_documents(session, "estatisticas", ESTBAN_GUID, "agencia",
                                   timeout=180, referer=REFERER)
    with open(os.path.join(args.out, f"listing_{today}.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)

    months = {}
    for it in items:
        m = re.match(r"(\d{6})_ESTBAN_AG", it.get("Nome", ""), re.IGNORECASE)
        if m and args.start <= m.group(1) <= args.end:
            months[m.group(1)] = it
    print(f"listing: {len(items)} per-branch files; {len(months)} in {args.start}..{args.end}")

    rows = []
    for yyyymm in sorted(months):
        it = months[yyyymm]
        target = os.path.join(args.out, it["Nome"])
        status = "present"
        if args.force or not valid(target, it):
            status = "FAILED"
            for attempt in range(4):
                r = session.get(BCB_SITE + it["Url"], timeout=300)
                if r.status_code == 200 and len(r.content) > 200:
                    tmp = target + ".part"
                    with open(tmp, "wb") as f:
                        f.write(r.content)
                    if valid(tmp, it):
                        os.replace(tmp, target)
                        status = "saved"
                        break
                time.sleep(10 * (attempt + 1))
            time.sleep(1)
        size = os.path.getsize(target) if os.path.exists(target) else 0
        sha = ""
        if status != "FAILED":
            with open(target, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
        rows.append({"yyyymm": yyyymm, "nome": it["Nome"], "url": BCB_SITE + it["Url"],
                     "accessed": today, "bytes": size, "sha256": sha, "status": status})
        print(f"  {yyyymm}  {status:7s}  {size:>10,d} B")

    with open(os.path.join(args.out, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["yyyymm", "nome", "url", "accessed", "bytes", "sha256", "status"])
        w.writeheader()
        w.writerows(rows)
    bad = [r["yyyymm"] for r in rows if r["status"] == "FAILED"]
    print(f"{len(rows) - len(bad)} of {len(rows)} months on disk in {args.out}"
          + (f"; FAILED: {bad}" if bad else ""))


if __name__ == "__main__":
    main()
