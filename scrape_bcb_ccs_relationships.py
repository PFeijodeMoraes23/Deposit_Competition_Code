###─────────────────────────────────────────────────────────────────────────────
# scrape_bcb_ccs_relationships.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Download the BCB's annual counts of ADULTS WITH A BANKING RELATIONSHIP, built on the
#   Cadastro de Clientes do Sistema Financeiro Nacional (CCS): each person (CPF) counted once,
#   however many institutions they use. Brazil, the five macro-regions and six age bands,
#   2006 onward, plus the share of adults with a relationship.
#
#   Not a pipeline step. Raw files are saved untouched (request S1; its NOTES.md, in
#   DATA_RAW/_requests/devo_2026-10-20/S1_accounts_per_person/, documents the series and gaps).
#
# DATA SOURCE
#   BCB SGS API, one series per call:
#     https://api.bcb.gov.br/dados/serie/bcdata.sgs.<code>/dados?formato=csv
#   Documentation: the dataset pages on dadosabertos.bcb.gov.br (e.g. dataset 25127).
#   Values: millions of adults (counts) or percent (25126); ';'-separated, comma decimals,
#   dates dd/mm/yyyy (one observation per year, dated 01/01).
#
# OUTPUT  (default utils.raw_paths.BCB_CCS_RELATIONSHIPS, or --out)
#   sgs_<code>.csv   the series exactly as served
#   manifest.csv     code, label, url, accessed, status, n_obs, first, last, sha256
#
#   The API sometimes answers with an HTML error page instead of data. A response whose
#   first line is not the CSV header is recorded as a failure and never overwrites a good
#   file from an earlier run.
#
# USAGE
#   python scrape_bcb_ccs_relationships.py
#   python scrape_bcb_ccs_relationships.py --codes 25105 25107 --out <dir>
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import time

from utils import raw_paths
from utils.bcb_http import retry_session

SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=csv"
CSV_HEADER = '"data";"valor"'

SERIES = {
    25120: "adults with a banking relationship, age 15-24",
    25121: "adults with a banking relationship, age 25-34",
    25122: "adults with a banking relationship, age 35-44",
    25123: "adults with a banking relationship, age 45-54",
    25124: "adults with a banking relationship, age 55-64",
    25125: "adults with a banking relationship, age 65+",
    25126: "share of adults with a banking relationship (%)",
    25127: "adults with a banking relationship, Brazil",
    25128: "adults with a banking relationship, Centro-Oeste",
    25129: "adults with a banking relationship, Nordeste",
    25130: "adults with a banking relationship, Norte",
    25131: "adults with a banking relationship, Sudeste",
    25132: "adults with a banking relationship, Sul",
}

DEFAULT_OUT = raw_paths.BCB_CCS_RELATIONSHIPS


def fetch(session, code: int) -> tuple[str, bytes | None, str]:
    """(status, body, note). status is 'ok' only when the body is the SGS CSV."""
    url = SGS_URL.format(code=code)
    try:
        r = session.get(url, timeout=120)
    except Exception as e:  # noqa: BLE001 -- recorded in the manifest
        return "error", None, repr(e)
    if r.status_code != 200:
        return "error", None, f"HTTP {r.status_code}"
    first = r.content[:64].decode("utf-8", errors="replace").splitlines()[0] if r.content else ""
    if first.strip() != CSV_HEADER:
        return "error", None, f"not the SGS CSV (first line: {first[:40]!r})"
    return "ok", r.content, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", type=int, nargs="*",
                    help="SGS codes to fetch (default: the 13 relationship series)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    codes = args.codes or list(SERIES)
    os.makedirs(args.out, exist_ok=True)
    session = retry_session(total=4, backoff=3.0)
    accessed = dt.date.today().isoformat()
    rows = []
    for code in codes:
        status, body, note = fetch(session, code)
        target = os.path.join(args.out, f"sgs_{code}.csv")
        n_obs = first = last = sha = ""
        if status == "ok":
            tmp = target + ".part"
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, target)
            lines = body.decode("utf-8").strip().splitlines()[1:]
            n_obs, first, last = len(lines), lines[0].split(";")[0].strip('"'), lines[-1].split(";")[0].strip('"')
            sha = hashlib.sha256(body).hexdigest()
        print(f"{code}  {status:5s}  {n_obs!s:>4} obs  {first}..{last}  {note}")
        rows.append({"code": code, "label": SERIES.get(code, ""), "url": SGS_URL.format(code=code),
                     "accessed": accessed, "status": status, "note": note, "n_obs": n_obs,
                     "first": first, "last": last, "sha256": sha})
        time.sleep(1)

    with open(os.path.join(args.out, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r["code"] for r in rows if r["status"] != "ok"]
    print(f"{len(rows) - len(bad)} of {len(rows)} series saved to {args.out}"
          + (f"; FAILED: {bad}" if bad else ""))


if __name__ == "__main__":
    main()
