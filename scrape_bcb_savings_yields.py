###─────────────────────────────────────────────────────────────────────────────
# scrape_bcb_savings_yields.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Download the credited yields of savings deposits (caderneta de poupança), the series
#   utils/macro_rates.py compounds into the quarterly savings rates of the panel:
#     SGS 195  deposits from 04.05.2012 (the "new rule": 70% of the Selic target + TR when the
#              target is at or below 8.5% a year, else 0.5% a month + TR)  -> savings_rate_qoq
#     SGS 25   deposits up to 03.05.2012 (the "old rule": always 0.5% a month + TR; Lei
#              12.703/2012 keeps it for the stock deposited before the change) -> savings_old_qoq
#   Each observation is the yield (% in the period) credited on one anniversary date, the
#   observation date, for the month that ends on the next month's same day.
#
#   Pipeline input of the rate steps (panel_cosif_calibrate.py, panel_deposit_rates.py).
#   Responses are saved exactly as served; utils/macro_rates.py parses them.
#
# DATA SOURCE
#   BCB SGS API  https://api.bcb.gov.br/dados/serie/bcdata.sgs.<code>/dados?formato=json
#                &dataInicial=dd/mm/aaaa[&dataFinal=dd/mm/aaaa]
#   Both series are daily (one value per anniversary date), so they are fetched in blocks of
#   --block-years calendar years (the API serves at most 10 years of a daily series per
#   request); the last block is open-ended.
#
# OUTPUT  --out (default paths.SGS_RAW / "savings_yields"):
#   sgs_<code>_<firstyear>-<lastyear or "">.json   one file per request block
#   manifest.csv                                   file, url, accessed, status, n_obs, first, last, sha256
#   A response that is not the expected JSON is recorded as a failure and never overwrites a
#   good file from an earlier run.
#
# USAGE
#   python scrape_bcb_savings_yields.py
#   python scrape_bcb_savings_yields.py --start 2013 --out <dir>
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import time

from scrape_national_series import SGS, fetch_save
from utils import paths
from utils.bcb_http import retry_session

SAVINGS_SERIES = {
    195: "Savings deposits from 04.05.2012, yield in the period (% per month), by anniversary date",
    25: "Savings deposits up to 03.05.2012, yield in the period (% per month), by anniversary date",
}
OUT_DEFAULT = paths.SGS_RAW / "savings_yields"


def main() -> None:
    ap = argparse.ArgumentParser(description="Savings-deposit yields, SGS 195 and 25, raw")
    ap.add_argument("--start", type=int, default=2013, help="first calendar year fetched")
    ap.add_argument("--block-years", type=int, default=8,
                    help="calendar years per request (API limit for daily series: 10)")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    if not 1 <= args.block_years <= 10:
        ap.error("--block-years must be between 1 and 10 (the SGS limit for daily series)")

    os.makedirs(args.out, exist_ok=True)
    session = retry_session(total=4, backoff=3.0)
    today = dt.date.today()
    rows = []
    for code in SAVINGS_SERIES:
        starts = list(range(args.start, today.year + 1, args.block_years))
        for i, y0 in enumerate(starts):
            url = SGS.format(code=code, ini=f"01/01/{y0}")
            y1 = ""
            if i < len(starts) - 1:
                y1 = y0 + args.block_years - 1
                url += f"&dataFinal=31/12/{y1}"
            rows.append(fetch_save(session, url, os.path.join(args.out, f"sgs_{code}_{y0}-{y1}.json"),
                                   "sgs", today.isoformat()))
            time.sleep(1)

    with open(os.path.join(args.out, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r["file"] for r in rows if r["status"] != "ok"]
    print(f"{len(rows) - len(bad)} of {len(rows)} files saved to {args.out}"
          + (f"; FAILED: {bad}" if bad else ""))


if __name__ == "__main__":
    main()
