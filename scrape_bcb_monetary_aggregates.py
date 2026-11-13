###─────────────────────────────────────────────────────────────────────────────
# scrape_bcb_monetary_aggregates.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Download the BCB's broad monetary aggregates (the "Novo" series of the 2018 methodology,
#   Nota Técnica do BCB nº 48, novembro de 2018) and their components, for the measured
#   new money of the demand side's market size (MAIN_IMPR_CHANNEL.md, A3: N_t = Wealth_t - C_t,
#   the public's liquid financial wealth). Sixteen monthly series, 2015-01 to 2024-12, one
#   request each:
#     27789 currency held by the public       27790 demand deposits          27791 M1
#      1835 savings deposits                  27805 time deposits            27806 letras financeiras
#     27807 letras de crédito (LCA + LCI)     27808 other private securities 27809 private securities, total
#     27810 M2                                27811 quotas of money-market-type funds
#     27812 repos with private securities     1839 repos with federal securities
#     27813 M3                                27814 federal securities held by the public (Selic)
#     27815 M4
#   Monthly month-end balances, thousands of current reais (catalogue: "Milhares de unidades
#   monetárias correntes"). Holdings of RESIDENTS in the money-holding sector; non-residents,
#   the issuing institutions and investment funds are excluded (NT 48, Tabela 34-A, nota 1/).
#
#   Not a pipeline step. Responses are saved exactly as served; nothing is transformed.
#   The request, parse and save routines are scrape_national_series.py's (imported), so the
#   files, the manifest and the failure rule are the same as in BCB/selic_sgs.
#
# OUTPUT  --out (default DATA_RAW/BCB/monetary_aggregates_sgs):
#   sgs_<code>_2015-2024.json   one per series, as served by the SGS API
#   manifest.csv                file, url, accessed, status, note, n_obs, first, last, sha256
#   source_log_rows.csv         one row in DATA_RAW's master-log format (build_master_source_log.py)
#   A response that is not the expected JSON is recorded as a failure and never overwrites a good
#   file from an earlier run.
#
# USAGE
#   python scrape_bcb_monetary_aggregates.py --dry-run      the 16 URLs and targets; no network
#   python scrape_bcb_monetary_aggregates.py                download (needs Pedro's go)
#   python scrape_bcb_monetary_aggregates.py --validate     checks on the saved files; no network:
#       120 monthly observations 01/01/2015 .. 01/12/2024 per series, numeric values, sha256 equal
#       to the manifest, and the aggregation identities of NT 48 (M1 = 27789 + 27790;
#       27809 = 27805 + 27806 + 27807 + 27808; M2 = M1 + 1835 + 27809; M3 = M2 + 27811 + 27812 +
#       1839; M4 = M3 + 27814), each to 0.1 percent in every month.
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

SGS = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"
       "&dataInicial={ini}&dataFinal={fin}")
OUT_DEFAULT = getattr(raw_paths, "BCB_MONETARY_AGGREGATES_SGS",
                      raw_paths.DATA_RAW / "BCB" / "monetary_aggregates_sgs")
FIRST, LAST = (2015, 1), (2024, 12)

SERIES = {   # code: (label, role in the new-money construction)
    27789: ("Currency held by the public (PMPP)", "outside, zero return"),
    27790: ("Demand deposits", "inside"),
    27791: ("M1", "check"),
    1835: ("Savings deposits", "inside"),
    27805: ("Time deposits", "inside"),
    27806: ("Letras financeiras", "outside"),
    27807: ("Letras de credito (LCA + LCI)", "outside"),
    27808: ("Other private securities", "outside"),
    27809: ("Private securities held by the public, total", "check"),
    27810: ("M2", "check"),
    27811: ("Quotas of money-market-type funds (FX, fixed income, multimarket)", "outside"),
    27812: ("Repos with private securities", "outside"),
    1839: ("Repos with federal securities", "outside"),
    27813: ("M3", "check"),
    27814: ("Federal securities held by the public (Selic)", "outside"),
    27815: ("M4", "broad total"),
}
IDENTITIES = {   # lhs: terms, NT 48 section 6
    27791: (27789, 27790),
    27809: (27805, 27806, 27807, 27808),
    27810: (27791, 1835, 27809),
    27813: (27810, 27811, 27812, 1839),
    27815: (27813, 27814),
}


def plan():
    ini = f"01/{FIRST[1]:02d}/{FIRST[0]}"
    fin = f"31/{LAST[1]:02d}/{LAST[0]}"
    return [(code, SGS.format(code=code, ini=ini, fin=fin), f"sgs_{code}_{FIRST[0]}-{LAST[0]}.json")
            for code in SERIES]


def source_log_row(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    return {
        "round": "", "task": "Step 1 new money (IMPR 11-12)",
        "dataset": "BCB broad monetary aggregates, 2018 methodology (Novo): M1-M4 and components, "
                   "16 SGS series (" + ", ".join(str(c) for c in SERIES) + ")",
        "url": "https://api.bcb.gov.br/dados/serie/bcdata.sgs.<code>/dados?formato=json"
               f"&dataInicial=01/01/{FIRST[0]}&dataFinal=31/12/{LAST[0]}",
        "date_accessed": dt.date.today().isoformat(), "publisher": "Banco Central do Brasil (SGS)",
        "coverage_years": f"{FIRST[0]}-{FIRST[1]:02d} to {LAST[0]}-{LAST[1]:02d}", "frequency": "monthly",
        "geography": "national", "units": "thousands of current reais (month-end balance)",
        "licence": "ODbL (BCB open-data pages)",
        "local_files": f"sgs_<code>_{FIRST[0]}-{LAST[0]}.json ({len(ok)} of {len(rows)}); manifest.csv",
        "notes": "scrape_bcb_monetary_aggregates.py. Holdings of residents in the money-holding sector "
                 "(non-residents, issuing institutions and investment funds excluded; NT 48, Tabela 34-A, "
                 "nota 1/). Validate with --validate (identities M1..M4).",
        "location": "BCB/monetary_aggregates_sgs",
    }


def validate(out: str) -> int:
    man = {r["file"]: r for r in csv.DictReader(open(os.path.join(out, "manifest.csv"), encoding="utf-8"))}
    vals, bad = {}, []
    for code, _url, name in plan():
        path = os.path.join(out, name)
        if not os.path.exists(path):
            bad.append(f"{code}: missing")
            continue
        body = open(path, "rb").read()
        if man.get(name, {}).get("sha256") != hashlib.sha256(body).hexdigest():
            bad.append(f"{code}: sha256 differs from the manifest")
        j = json.loads(body.decode("utf-8"))
        dates = [r["data"] for r in j]
        v = [float(r["valor"]) for r in j]
        want_first, want_last = f"01/{FIRST[1]:02d}/{FIRST[0]}", f"01/{LAST[1]:02d}/{LAST[0]}"
        n_want = (LAST[0] - FIRST[0]) * 12 + LAST[1] - FIRST[1] + 1
        if len(j) != n_want or dates[0] != want_first or dates[-1] != want_last:
            bad.append(f"{code}: {len(j)} obs {dates[0] if dates else ''}..{dates[-1] if dates else ''}")
        vals[code] = dict(zip(dates, v))
        print(f"  {code:>6}  {len(j):4d} obs  {dates[0]}..{dates[-1]}  Dec-2016 {vals[code].get('01/12/2016', float('nan')):>18,.0f}"
              f"  Dec-2024 {vals[code].get('01/12/2024', float('nan')):>18,.0f}  ({SERIES[code][0]})")
    for lhs, terms in IDENTITIES.items():
        if lhs not in vals or any(t not in vals for t in terms):
            continue
        worst = max(abs(vals[lhs][d] - sum(vals[t][d] for t in terms)) / abs(vals[lhs][d])
                    for d in vals[lhs] if vals[lhs][d])
        print(f"  identity {lhs} = {' + '.join(map(str, terms))}: max relative gap {worst:.2e}")
        if worst > 1e-3:
            bad.append(f"identity {lhs}: {worst:.2e}")
    print("VALID" if not bad else f"PROBLEMS: {bad}")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="BCB broad monetary aggregates (16 SGS series), raw")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--dry-run", action="store_true", help="print the URLs and targets; no network")
    ap.add_argument("--validate", action="store_true", help="check the saved files; no network")
    args = ap.parse_args()
    if args.validate:
        return validate(args.out)
    if args.dry_run:
        for code, url, name in plan():
            print(f"{code:>6}  {os.path.join(args.out, name)}\n        {url}")
        print(f"{len(SERIES)} requests; nothing fetched (--dry-run)")
        return 0

    import scrape_national_series as sns          # its parse + fetch_save: same files, same failure rule
    from utils.bcb_http import retry_session
    os.makedirs(args.out, exist_ok=True)
    session = retry_session(total=4, backoff=3.0)
    today = dt.date.today().isoformat()
    rows = []
    for code, url, name in plan():
        rows.append(sns.fetch_save(session, url, os.path.join(args.out, name), "sgs", today))
        time.sleep(1)
    with open(os.path.join(args.out, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    row = source_log_row(rows)
    with open(os.path.join(args.out, "source_log_rows.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    bad = [r["file"] for r in rows if r["status"] != "ok"]
    print(f"{len(rows) - len(bad)} of {len(rows)} files saved to {args.out}" + (f"; FAILED: {bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
