###─────────────────────────────────────────────────────────────────────────────
# scrape_pix_keys.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Download the month-end stock of Pix keys by participant institution (scraping task S13):
#   the only per-participant Pix measure the BCB publishes as open data. The BCB denied the
#   LAI for Pix statistics by participant (18810.025656/2026-11), so this is the public route.
#   One row per month-end x participant (ISPB) x user type (PF/PJ) x key type, with the count
#   of registered keys.
#
#   Not a pipeline step. The response is saved exactly as served; nothing is transformed.
#
# DATA SOURCE  (BCB Olinda, open data; scripted access is allowed)
#   https://olinda.bcb.gov.br/olinda/servico/Pix_DadosAbertos/versao/v1/odata/
#       ChavesPix(Data=@Data)?@Data='<YYYY-MM-DD>'&$format=text/csv
#   Fields: Data (month-end), ISPB, Nome, NaturezaUsuario (PF/PJ), TipoChave (CPF, CNPJ,
#   Celular, e-mail, Aleatória), qtdChaves, Segmento.
#   The Data parameter is a LOWER BOUND: one request returns every month-end from that date
#   to the latest month (checked 2026-10-27: Data='2020-11-30' returned 71 month-ends,
#   2020-11-30..2026-09-30, 416,472 rows, 35 MB). The first month served is 2020-11-30.
#   The CSV is UTF-8 with a header row.
#
# CAVEATS (see NOTES.md in the output folder)
#   Keys are a stock of registrations, not usage. One person can hold keys at several
#   institutions, and a CPF key sits at one institution only.
#
# OUTPUT  (default utils.raw_paths.BCB_PIX_KEYS, or --out)
#   chavespix_from<YYYY-MM-DD>_<access date>.csv   the response as served
#   manifest.csv   file, url, accessed, status, note, n_rows, n_months, first, last,
#                  months_missing (month-ends absent between first and last), sha256
#   A response that is not the expected CSV is recorded as a failure and writes no file.
#
# USAGE
#   python scrape_pix_keys.py
#   python scrape_pix_keys.py --start 2020-11-01 --out <dir>
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import os

import pandas as pd

from utils import raw_paths
from utils.bcb_http import retry_session

URL = ("https://olinda.bcb.gov.br/olinda/servico/Pix_DadosAbertos/versao/v1/odata/"
       "ChavesPix(Data=@Data)")
FIELDS = ["Data", "ISPB", "Nome", "NaturezaUsuario", "TipoChave", "qtdChaves", "Segmento"]


def check(body: bytes) -> tuple[bool, str, pd.DataFrame | None]:
    """(ok, note, frame) for a ChavesPix CSV body."""
    try:
        df = pd.read_csv(io.BytesIO(body), dtype={"ISPB": str}, encoding="utf-8")
    except Exception as e:  # noqa: BLE001 -- an HTML error page or a broken body
        return False, f"not a CSV ({type(e).__name__})", None
    if list(df.columns) != FIELDS:
        return False, f"unexpected columns {list(df.columns)}", None
    if df.empty:
        return False, "empty response", None
    return True, "", df


def main() -> None:
    ap = argparse.ArgumentParser(description="Pix keys by participant (BCB Olinda ChavesPix), raw")
    ap.add_argument("--start", default="2020-11-01", help="lower bound of the Data parameter (YYYY-MM-DD)")
    ap.add_argument("--out", default=str(raw_paths.BCB_PIX_KEYS))
    args = ap.parse_args()
    dt.date.fromisoformat(args.start)

    os.makedirs(args.out, exist_ok=True)
    session = retry_session(total=4, backoff=3.0)
    today = dt.date.today().isoformat()
    params = {"@Data": f"'{args.start}'", "$format": "text/csv"}
    target = os.path.join(args.out, f"chavespix_from{args.start}_{today}.csv")

    row = {"file": os.path.basename(target), "url": URL + f"?@Data='{args.start}'&$format=text/csv",
           "accessed": today, "status": "error", "note": "", "n_rows": "", "n_months": "",
           "first": "", "last": "", "months_missing": "", "sha256": ""}
    try:
        r = session.get(URL, params=params, timeout=900)
        if r.status_code != 200:
            row["note"] = f"HTTP {r.status_code}"
        else:
            ok, note, df = check(r.content)
            row["note"] = note
            if ok:
                tmp = target + ".part"
                with open(tmp, "wb") as f:
                    f.write(r.content)
                os.replace(tmp, target)
                months = sorted(pd.to_datetime(df["Data"]).dt.to_period("M").unique())
                expected = pd.period_range(months[0], months[-1], freq="M")
                missing = [str(m) for m in expected if m not in set(months)]
                row.update(status="ok", n_rows=len(df), n_months=len(months), first=str(months[0]),
                           last=str(months[-1]), months_missing=";".join(missing),
                           sha256=hashlib.sha256(r.content).hexdigest())
    except Exception as e:  # noqa: BLE001
        row["note"] = repr(e)

    manifest = os.path.join(args.out, "manifest.csv")
    new = not os.path.exists(manifest)
    with open(manifest, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"{row['status']}: {row['file']}  rows={row['n_rows']} months={row['n_months']} "
          f"{row['first']}..{row['last']} missing=[{row['months_missing']}] {row['note']}")


if __name__ == "__main__":
    main()
