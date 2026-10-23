###─────────────────────────────────────────────────────────────────────────────
# scrape_national_series.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Download national return series for households' outside options (request S8(c); notes in
#   DATA_RAW/_requests/breakfast_2026-10-21/S8_outside_options/NOTES.md):
#     Ibovespa   Ipeadata GM366_IBVSP366 (daily close, points) and GM12_IBVSP12 (monthly %
#                change of the close), full history, plus each series' metadata
#     Selic      BCB SGS 11 (daily rate, % per business day), 4390 (accumulated in the
#                month, % per month), 4189 (accumulated in the month, annualised % per year)
#   Selic is the proxy for the Tesouro Selic return: the Tesouro Direto price/rate history
#   sits behind a bot wall (HTTP 429 + JavaScript challenge) and is not scraped.
#
#   Not a pipeline step. Responses are saved exactly as served; nothing is transformed.
#
# DATA SOURCES
#   Ipeadata OData v4  http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='<code>')
#                      http://www.ipeadata.gov.br/api/odata4/Metadados('<code>')
#   BCB SGS API        https://api.bcb.gov.br/dados/serie/bcdata.sgs.<code>/dados?formato=json
#                      &dataInicial=dd/mm/aaaa[&dataFinal=dd/mm/aaaa]
#   The SGS API serves at most 10 years of a DAILY series per request, so daily series are
#   fetched in blocks of --block-years calendar years; the last block is open-ended.
#   SGS 7 (Ibovespa) is not fetched: it stops in 2019-09.
#   The monthly series (GM12_IBVSP12, SGS 4390, SGS 4189) carry the CURRENT month as a
#   partial value until the month closes; re-run after month end for the final value.
#
# OUTPUT  one folder per source (DATA_RAW is organised by source; utils.raw_paths):
#   --out-ipea (default raw_paths.IPEADATA_IBOVESPA): ipeadata_<code>.json, ipeadata_meta_<code>.json
#   --out-sgs  (default raw_paths.BCB_SELIC_SGS):     sgs_<code>_<firstyear>-<lastyear or "">.json,
#                                                       one file per request block
#   each folder: manifest.csv   file, url, accessed, status, n_obs, first, last, sha256
#   A response that is not the expected JSON is recorded as a failure and never overwrites
#   a good file from an earlier run.
#
# USAGE
#   python scrape_national_series.py
#   python scrape_national_series.py --start 2013 --out-ipea <dir> --out-sgs <dir>
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import time

from utils import raw_paths
from utils.bcb_http import retry_session

IPEA = "http://www.ipeadata.gov.br/api/odata4"
SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={ini}"

IPEA_SERIES = {
    "GM366_IBVSP366": "Ibovespa, daily close (points)",
    "GM12_IBVSP12": "Ibovespa, monthly % change of the close",
}
SGS_SERIES = {   # code: (label, daily?)
    11: ("Selic, daily rate (% per business day)", True),
    4390: ("Selic accumulated in the month (% per month)", False),
    4189: ("Selic accumulated in the month, annualised (% per year)", False),
}


def parse(body: bytes, kind: str):
    """(ok, n_obs, first, last) for an Ipeadata or SGS JSON body."""
    try:
        j = json.loads(body.decode("utf-8"))
    except Exception:  # noqa: BLE001 -- an HTML error page, recorded as a failure
        return False, "", "", ""
    if kind == "ipea":
        v = j.get("value") if isinstance(j, dict) else None
        if not isinstance(v, list):
            return False, "", "", ""
        dates = [r.get("VALDATA", "")[:10] for r in v if "VALDATA" in r]
        return True, len(v), (dates[0] if dates else ""), (dates[-1] if dates else "")
    if kind == "ipea_meta":
        return isinstance(j, dict) and isinstance(j.get("value"), list), "", "", ""
    if not (isinstance(j, list) and all(isinstance(r, dict) and "data" in r for r in j)):
        return False, "", "", ""
    return True, len(j), (j[0]["data"] if j else ""), (j[-1]["data"] if j else "")


def fetch_save(session, url: str, target: str, kind: str, accessed: str) -> dict:
    status, n, first, last, sha, note = "error", "", "", "", "", ""
    try:
        r = session.get(url, timeout=300)
        if r.status_code != 200:
            note = f"HTTP {r.status_code}"
        else:
            ok, n, first, last = parse(r.content, kind)
            if ok:
                tmp = target + ".part"
                with open(tmp, "wb") as f:
                    f.write(r.content)
                os.replace(tmp, target)
                status, sha = "ok", hashlib.sha256(r.content).hexdigest()
            else:
                note = "not the expected JSON"
    except Exception as e:  # noqa: BLE001
        note = repr(e)
    print(f"  {status:5s}  {os.path.basename(target):38s} {n!s:>6} obs  {first}..{last}  {note}")
    return {"file": os.path.basename(target), "url": url, "accessed": accessed, "status": status,
            "note": note, "n_obs": n, "first": first, "last": last, "sha256": sha}


def main() -> None:
    ap = argparse.ArgumentParser(description="National return series (Ibovespa, Selic), raw")
    ap.add_argument("--start", type=int, default=2016, help="first year of the SGS series")
    ap.add_argument("--block-years", type=int, default=8,
                    help="calendar years per request for daily SGS series (API limit: 10)")
    ap.add_argument("--out-ipea", default=str(raw_paths.IPEADATA_IBOVESPA))
    ap.add_argument("--out-sgs", default=str(raw_paths.BCB_SELIC_SGS))
    args = ap.parse_args()
    if not 1 <= args.block_years <= 10:
        ap.error("--block-years must be between 1 and 10 (the SGS limit for daily series)")

    for d in (args.out_ipea, args.out_sgs):
        os.makedirs(d, exist_ok=True)
    session = retry_session(total=4, backoff=3.0)
    today = dt.date.today()
    rows = {args.out_ipea: [], args.out_sgs: []}

    for code in IPEA_SERIES:
        rows[args.out_ipea].append(fetch_save(session, f"{IPEA}/ValoresSerie(SERCODIGO='{code}')",
                               os.path.join(args.out_ipea, f"ipeadata_{code}.json"), "ipea", today.isoformat()))
        rows[args.out_ipea].append(fetch_save(session, f"{IPEA}/Metadados('{code}')",
                               os.path.join(args.out_ipea, f"ipeadata_meta_{code}.json"), "ipea_meta", today.isoformat()))
        time.sleep(1)

    for code, (_label, daily) in SGS_SERIES.items():
        starts = list(range(args.start, today.year + 1, args.block_years)) if daily else [args.start]
        for i, y0 in enumerate(starts):
            last_block = i == len(starts) - 1
            url = SGS.format(code=code, ini=f"01/01/{y0}")
            y1 = ""
            if not last_block:
                y1 = y0 + args.block_years - 1
                url += f"&dataFinal=31/12/{y1}"
            rows[args.out_sgs].append(fetch_save(session, url, os.path.join(args.out_sgs, f"sgs_{code}_{y0}-{y1}.json"),
                                   "sgs", today.isoformat()))
            time.sleep(1)

    for out, rr in rows.items():
        with open(os.path.join(out, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
        bad = [r["file"] for r in rr if r["status"] != "ok"]
        print(f"{len(rr) - len(bad)} of {len(rr)} files saved to {out}"
              + (f"; FAILED: {bad}" if bad else ""))


if __name__ == "__main__":
    main()
