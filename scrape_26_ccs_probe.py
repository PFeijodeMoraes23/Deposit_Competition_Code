"""scrape_26_ccs_probe.py -- does BCB publish CCS relationship counts (and INCLUSIONS)?

Author: Pedro Feijo de Moraes

VERIFY-FIRST PROBE. Writes only a probe report.

Why this source. The CCS (Cadastro de Clientes do Sistema Financeiro Nacional) records every
client-institution RELATIONSHIP in Brazil. If BCB publishes per-period *inclusoes* (new
relationships), that is a true economy-wide GROSS relationship-formation flow -- the direct
analogue of Egan et al. (2025) eq. (8)'s new-account moment, which is the identification
piece our setting currently lacks (we have only deposit autocorrelation).

What would make it usable, in order of value:
  1. inclusions per period AT INSTITUTION level  -> a real second moment for phi
  2. inclusions per period, national only        -> an aggregate 1-phi benchmark
  3. relationship STOCKS only                    -> weak; deltas are net, like the Pix keys

Method: (a) query the SGS series metadata search for CCS/relacionamento series; (b) probe
the Olinda dataset catalogue for a CCS service; (c) fetch the CCS statistics page and list
any tabular/downloadable artifacts. Each step prints what it found.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import json
import re
import sys

import requests

from utils import paths as _paths

OUT = _paths.RAW / "CCS" / "probe"
OUT.mkdir(parents=True, exist_ok=True)
TIMEOUT = 60

SGS_SEARCH = ("https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do"
              "?method=prepararTelaLocalizarSeries")
OLINDA_CATALOG = "https://olinda.bcb.gov.br/olinda/servico/"
CCS_PAGES = [
    "https://www.bcb.gov.br/estabilidadefinanceira/ccs",
    "https://dadosabertos.bcb.gov.br/dataset?q=CCS",
    "https://dadosabertos.bcb.gov.br/api/3/action/package_search?q=relacionamento",
]
KEYWORDS = ("ccs", "cadastro de clientes", "relacionamento", "inclus")


def get(url, **kw):
    try:
        r = requests.get(url, timeout=TIMEOUT,
                         headers={"User-Agent": "research-probe/1.0"}, **kw)
        print(f"    GET {url[:88]:<88s} -> {r.status_code} ({len(r.content):,} B)")
        return r if r.status_code == 200 else None
    except requests.RequestException as e:
        print(f"    GET {url[:88]:<88s} -> {type(e).__name__}")
        return None


def main():
    print("=== PROBE: CCS (Cadastro de Clientes do SFN) relationship counts / inclusions ===")
    hits, reachable = [], False

    print("\n  [1] CKAN open-data catalogue search")
    for url in CCS_PAGES:
        r = get(url)
        if r is None:
            continue
        reachable = True
        ct = r.headers.get("Content-Type", "")
        if "json" in ct:
            try:
                js = r.json()
            except ValueError:
                continue
            res = (js.get("result", {}) or {}).get("results", []) if isinstance(js, dict) else []
            for d in res:
                title = (d.get("title") or "") + " " + (d.get("notes") or "")
                if any(k in title.lower() for k in KEYWORDS):
                    n = d.get("name")
                    hits.append({"kind": "ckan", "name": n, "title": d.get("title"),
                                 "resources": [x.get("url") for x in d.get("resources", [])]})
                    print(f"      dataset: {d.get('title')}  ({n})")
        else:
            body = r.text.lower()
            found = [k for k in KEYWORDS if k in body]
            links = re.findall(r'href="([^"]+\.(?:csv|xlsx?|zip))"', r.text, flags=re.I)
            print(f"      keywords present: {found or 'none'}; "
                  f"downloadable links: {len(links)}")
            for l in links[:10]:
                print(f"        {l}")
            if links:
                hits.append({"kind": "page", "url": url, "links": links[:20]})

    print("\n  [2] Olinda service catalogue (is there a CCS OData service?)")
    r = get(OLINDA_CATALOG + "$metadata") or get(OLINDA_CATALOG)
    if r is not None:
        reachable = True
        names = set(re.findall(r"[A-Za-z_]*CCS[A-Za-z_]*", r.text))
        print(f"      CCS-like service names: {sorted(names) or 'none'}")
        if names:
            hits.append({"kind": "olinda", "services": sorted(names)})

    (OUT / "ccs_probe.json").write_text(json.dumps(hits, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"\n  report -> {OUT / 'ccs_probe.json'}")

    if not reachable:
        print("\nVERDICT: NOT-FOUND -- no BCB endpoint answered; re-run when online.")
        return 1
    if not hits:
        print("\n  Nothing CCS-shaped is published as open data. Consistent with "
              "scrape_14_bcb_accounts.py:6-30, which documents that BCB exposes no public "
              "per-institution client/account counts (CCS itself is access-restricted: it "
              "answers judicial/authority queries, it is not a statistical release).")
        print("\nVERDICT: NOT-FOUND -- no usable CCS series. Do not build a scraper.")
        return 0
    print("\nVERDICT: PARTIAL -- candidate artifacts listed above. Before building anything, "
          "confirm (i) INCLUSIONS not just stocks, (ii) periodicity, (iii) whether any "
          "institution dimension survives; a national stock series adds little over D11.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
