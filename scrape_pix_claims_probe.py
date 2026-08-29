"""scrape_pix_claims_probe.py -- does BCB publish Pix key CLAIMS (portability) data?

Author: Pedro Feijo de Moraes

VERIFY-FIRST PROBE. Writes nothing but a probe report; no panel is touched.

Why this source. A Pix key "reivindicacao" (claim/portability) is a depositor moving a key
from one institution to another -- a DIRECTED, GROSS switching event, which is the closest
Brazilian analogue of the new-account flows Egan et al. (2025, eq. 8) use to identify
sleepiness independently of deposit autocorrelation. The repo already consumes ChavesPix
(key STOCKS per ISPB-month, scrape_22); this probe asks whether the same Olinda dataset
exposes claim FLOWS.

Method: fetch the OData $metadata for the Pix open-data service, list every EntitySet and
its properties, then sample any entity whose name or fields look claim/portability-related.

Exit line is one of:
  VERDICT: VERIFIED   -- a claims/portability entity exists and returned rows (schema printed)
  VERDICT: PARTIAL    -- the service answers but exposes no claim entity (stocks only)
  VERDICT: NOT-FOUND  -- endpoint unreachable / no usable metadata
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import json
import re
import sys
import xml.etree.ElementTree as ET

import requests

from utils import paths as _paths

OUT = _paths.RAW / "PIX" / "probe"
OUT.mkdir(parents=True, exist_ok=True)

BASES = [
    "https://olinda.bcb.gov.br/olinda/servico/Pix_DadosAbertos/versao/v1/odata",
    "https://olinda.bcb.gov.br/olinda/servico/SPI/versao/v1/odata",
]
CLAIM_WORDS = ("reivindic", "portab", "claim", "posse", "transferenc", "doacao")
TIMEOUT = 60


def get(url, **kw):
    try:
        r = requests.get(url, timeout=TIMEOUT, headers={"Accept": "application/json"}, **kw)
        return r if r.status_code == 200 else None
    except requests.RequestException as e:
        print(f"    [net] {type(e).__name__}: {e}")
        return None


def entity_sets(base):
    """EntitySets for an Olinda service. Olinda does not always serve $metadata at the
    documented path (it 404s for Pix_DadosAbertos even though the DATA endpoints work --
    scrape_22 reads ChavesPix from this very base), so fall back to the OData service
    document, which lists the entity sets as JSON."""
    r = get(base + "/$metadata")
    if r is None:
        svc = get(base) or get(base + "/")
        if svc is not None:
            try:
                js = svc.json()
                names = [d.get("name") or d.get("url") for d in js.get("value", [])]
                names = [n for n in names if n]
                if names:
                    print(f"    (service document used; $metadata not served)")
                    return names
            except ValueError:
                pass
        return []
    (OUT / f"metadata_{re.sub(r'[^A-Za-z0-9]', '_', base)[-40:]}.xml").write_text(
        r.text, encoding="utf-8")
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        print(f"    [xml] unparseable metadata: {e}")
        return []
    out = []
    for es in root.iter():
        if es.tag.endswith("}EntitySet") or es.tag == "EntitySet":
            out.append(es.attrib.get("Name", ""))
    return [e for e in out if e]


def main():
    print("=== PROBE: Pix key claims / portability (gross switching flows) ===")
    found_any, claim_hits = False, []
    for base in BASES:
        print(f"\n  service: {base}")
        sets = entity_sets(base)
        if not sets:
            print("    no metadata / unreachable")
            continue
        found_any = True
        print(f"    {len(sets)} EntitySet(s): {', '.join(sets[:20])}"
              + (" ..." if len(sets) > 20 else ""))
        for name in sets:
            if any(w in name.lower() for w in CLAIM_WORDS):
                claim_hits.append((base, name))

    if not found_any:
        print("\nVERDICT: NOT-FOUND -- no Olinda Pix service answered; re-run when online.")
        return 1

    if not claim_hits:
        print("\n  No EntitySet name matches "
              f"{CLAIM_WORDS}. The Pix open data appears to expose key STOCKS "
              "(ChavesPix, already used by scrape_22) and transaction aggregates, not "
              "key-claim flows.")
        print("\nVERDICT: PARTIAL -- service reachable, no claim/portability entity exposed.")
        print("  Implication for D11: the key-stock delta remains the only available")
        print("  institution-level relationship-flow proxy; it is NET, so it bounds the")
        print("  gross rate from below. Do NOT build a scraper against a non-existent entity.")
        return 0

    print("\n  candidate claim entities:")
    report = []
    for base, name in claim_hits:
        print(f"    {name}")
        r = get(f"{base}/{name}?$top=5&$format=json")
        if r is None:
            r = get(f"{base}/{name}(dataBase=@dataBase)?@dataBase='202401'&$top=5&$format=json")
        if r is None:
            print("      (no sample rows returned -- parameterised entity?)")
            report.append({"base": base, "entity": name, "sample": None})
            continue
        js = r.json()
        vals = js.get("value", [])
        cols = sorted(vals[0].keys()) if vals else []
        print(f"      rows={len(vals)}  columns={cols}")
        report.append({"base": base, "entity": name, "columns": cols,
                       "sample": vals[:2]})
    (OUT / "pix_claims_probe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    print(f"\n  report -> {OUT / 'pix_claims_probe.json'}")
    print("\nVERDICT: VERIFIED -- claim/portability entity exists. Next step: confirm it")
    print("  carries an INSTITUTION dimension (ISPB) and directed counts; only then design")
    print("  the scraper + the eq-(8)-style gross-flow moment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
