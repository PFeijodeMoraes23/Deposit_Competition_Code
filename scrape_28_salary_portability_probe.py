"""scrape_salary_portability_probe.py -- does BCB publish salary-account portability counts?

Author: Pedro Feijo de Moraes

VERIFY-FIRST PROBE. Writes only a probe report.

Why this source. "Portabilidade de salario" is a worker moving their payroll credit from the
employer's bank to a bank of their choice -- an explicit, GROSS, directed switching decision.
Counts per period (ideally per institution) would give an eq-(8)-style flow measure of the
awake share 1-phi that is completely independent of deposit autocorrelation, and unlike the
Pix-key proxy in D11 it is a genuine gross flow rather than a net stock change.

Note the repo already holds CREDIT portability (shared/SCR/Portabilidade, processed by the
sibling SCR_Portability_PIX project) -- that is loan refinancing, a different margin, and
not a deposit-relationship flow. This probe is specifically about SALARY portability.

Method: probe the SGS series catalogue and the open-data (CKAN) portal for portability
series, then fetch the "Cidadania Financeira" / statistics pages and list downloadable
artifacts whose text mentions salary portability.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import json
import re
import sys

import requests

from utils import paths as _paths

OUT = _paths.RAW / "PORTABILIDADE_SALARIO" / "probe"
OUT.mkdir(parents=True, exist_ok=True)
TIMEOUT = 60

CKAN = "https://dadosabertos.bcb.gov.br/api/3/action/package_search?q={}"
PAGES = [
    "https://www.bcb.gov.br/meubc/faqs/p/portabilidade-de-salario",
    "https://www.bcb.gov.br/estatisticas",
]
QUERIES = ["portabilidade", "portabilidade de salario", "salario"]
SALARY_WORDS = ("salario", "salário", "portabilidade")


def get(url):
    try:
        r = requests.get(url, timeout=TIMEOUT,
                         headers={"User-Agent": "research-probe/1.0"})
        print(f"    GET {url[:86]:<86s} -> {r.status_code}")
        return r if r.status_code == 200 else None
    except requests.RequestException as e:
        print(f"    GET {url[:86]:<86s} -> {type(e).__name__}")
        return None


def main():
    print("=== PROBE: salary-account portability counts (gross switching flow) ===")
    hits, reachable = [], False

    print("\n  [1] open-data catalogue (CKAN)")
    for q in QUERIES:
        r = get(CKAN.format(requests.utils.quote(q)))
        if r is None:
            continue
        reachable = True
        try:
            res = r.json().get("result", {}).get("results", [])
        except ValueError:
            continue
        for d in res:
            blob = ((d.get("title") or "") + " " + (d.get("notes") or "")).lower()
            if any(w in blob for w in SALARY_WORDS):
                rec = {"query": q, "name": d.get("name"), "title": d.get("title"),
                       "resources": [x.get("url") for x in d.get("resources", [])][:5]}
                hits.append(rec)
                print(f"      {d.get('title')}  ({d.get('name')})")

    print("\n  [2] BCB pages -- downloadable artifacts mentioning salary portability")
    for url in PAGES:
        r = get(url)
        if r is None:
            continue
        reachable = True
        body = r.text
        low = body.lower()
        present = [w for w in SALARY_WORDS if w in low]
        links = re.findall(r'href="([^"]+\.(?:csv|xlsx?|zip|ods))"', body, flags=re.I)
        print(f"      keywords: {present or 'none'};  data links: {len(links)}")
        for l in links[:8]:
            print(f"        {l}")
        if links and present:
            hits.append({"page": url, "links": links[:20]})

    (OUT / "salary_portability_probe.json").write_text(
        json.dumps(hits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  report -> {OUT / 'salary_portability_probe.json'}")

    if not reachable:
        print("\nVERDICT: NOT-FOUND -- no BCB endpoint answered; re-run when online.")
        return 1
    if not hits:
        print("\n  No salary-portability statistics found in the open-data portal or the "
              "statistics pages. BCB documents the RIGHT to salary portability (regulation "
              "and FAQ) without publishing operation counts.")
        print("\nVERDICT: NOT-FOUND -- no usable series. Do not build a scraper; D11's "
              "Pix-key proxy remains the flow measure, with its net-flow caveat.")
        return 0
    print("\nVERDICT: PARTIAL -- candidates listed. Before building: confirm the series is a "
          "COUNT of portability operations (not a stock of eligible accounts), its "
          "periodicity, and whether any institution dimension exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
