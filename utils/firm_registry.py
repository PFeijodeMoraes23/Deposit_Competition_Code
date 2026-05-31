"""firm_registry.py
# Author: Pedro Feijó de Moraes
#
# Registry of the firms whose customer/account/deposit disclosures we scrape,
# keyed so the result joins onto the project's unit of analysis: the prudential
# conglomerate (8-digit CNPJ root of the lead institution).
#
# Each entry carries:
#   - identifiers for the public-disclosure sources (SEC CIK, B3 ticker, CVM code)
#   - a *seed* CNPJ root and a list of name fragments
#   - segment ('digital'/'payment'/'incumbent') matching panel_5_flag_digital.py
#   - country_scope: what geography the firm's HEADLINE figures cover. SEC filers
#     such as NU/MELI report consolidated Latam numbers but break Brazil out in
#     segment notes; PAGS/STNE/INTR and the BR ADRs are effectively Brazil-only.
#
# `load_registry()` validates/overrides each seed CNPJ root and attaches the
# prudential-conglomerate code by fuzzy-matching the name fragments against the
# BCB IF Data List files already downloaded by scrape_1, so the canonical key
# comes from the project's own data rather than a hard-coded guess.
"""

from __future__ import annotations

import os
import re
import glob
import logging

log = logging.getLogger(__name__)

# segment ∈ {digital, payment, incumbent}; country_scope ∈ {brazil, latam}
# sec_cik None  => no SEC filer (use parent_filing or CVM)
# has_sec_deposits: whether the SEC/consolidated financials carry a Deposits line
FIRMS: list[dict] = [
    # ---- digital / payment institutions (D firms; mostly type-5 prepaid) ----
    dict(firm_key="nubank", display_name="Nu Holdings (Nu Pagamentos / Nu Financeira)",
         segment="digital", sec_cik="0001691493", b3_ticker="ROXO34", cvm_code=None,
         cnpj_root="18236120", country_scope="latam", has_sec_deposits=True,
         names=["nu pagamentos", "nubank", "nu financeira", "nu holdings"]),
    dict(firm_key="inter", display_name="Inter & Co (Banco Inter)",
         segment="digital", sec_cik="0001864163", b3_ticker="INBR32", cvm_code=None,
         cnpj_root="00416968", country_scope="brazil", has_sec_deposits=True,
         names=["banco inter", "inter &", "inter dtvm"]),
    dict(firm_key="pagseguro", display_name="PagSeguro Digital / PagBank",
         segment="payment", sec_cik="0001712807", b3_ticker="PAGS34", cvm_code=None,
         cnpj_root="08561701", country_scope="brazil", has_sec_deposits=True,
         names=["pagseguro", "pagbank"]),
    dict(firm_key="stone", display_name="StoneCo / Banco Stone",
         segment="payment", sec_cik="0001745431", b3_ticker="STOC31", cvm_code=None,
         cnpj_root="16501555", country_scope="brazil", has_sec_deposits=True,
         names=["stone", "stonex", "pagar.me"]),
    dict(firm_key="mercadopago", display_name="MercadoLibre / Mercado Pago",
         segment="payment", sec_cik="0001099590", b3_ticker="MELI34", cvm_code=None,
         cnpj_root="10573521", country_scope="latam", has_sec_deposits=False,
         names=["mercado pago", "mercadolibre", "mercado livre", "mercadopago"]),
    # no SEC filer -> handled by parent_filing / press scrapers
    dict(firm_key="c6", display_name="Banco C6 (C6 Bank)",
         segment="digital", sec_cik=None, b3_ticker=None, cvm_code=None,
         cnpj_root="31872495", country_scope="brazil", has_sec_deposits=False,
         names=["banco c6", "c6 bank", "c6 ctvm"], parent="JPMorgan Chase"),
    dict(firm_key="picpay", display_name="PicPay",
         segment="payment", sec_cik=None, b3_ticker=None, cvm_code=None,
         cnpj_root="22896431", country_scope="brazil", has_sec_deposits=False,
         names=["picpay"], parent="J&F / Original"),
    dict(firm_key="bs2", display_name="Banco BS2",
         segment="digital", sec_cik=None, b3_ticker=None, cvm_code=None,
         cnpj_root="71027866", country_scope="brazil", has_sec_deposits=False,
         names=["banco bs2", "bs2"]),
    dict(firm_key="sofisa", display_name="Banco Sofisa (Sofisa Direto)",
         segment="digital", sec_cik=None, b3_ticker=None, cvm_code="SOFISA",
         cnpj_root="60889128", country_scope="brazil", has_sec_deposits=False,
         names=["sofisa"]),
    dict(firm_key="rendimento", display_name="Banco Rendimento",
         segment="digital", sec_cik=None, b3_ticker=None, cvm_code=None,
         cnpj_root="68900810", country_scope="brazil", has_sec_deposits=False,
         names=["rendimento"]),
    dict(firm_key="genial", display_name="Banco Genial",
         segment="digital", sec_cik=None, b3_ticker=None, cvm_code=None,
         cnpj_root="45246410", country_scope="brazil", has_sec_deposits=False,
         names=["genial"]),

    # ---- incumbents (B firms; full type 1-4 portfolio) ----
    dict(firm_key="itau", display_name="Itaú Unibanco",
         segment="incumbent", sec_cik="0001132597", b3_ticker="ITUB4", cvm_code="ITAUUNIBANCO",
         cnpj_root="60701190", country_scope="latam", has_sec_deposits=True,
         names=["itau unibanco", "itaú unibanco", "itau", "itaú"]),
    dict(firm_key="bradesco", display_name="Banco Bradesco",
         segment="incumbent", sec_cik="0001160330", b3_ticker="BBDC4", cvm_code="BRADESCO",
         cnpj_root="60746948", country_scope="brazil", has_sec_deposits=True,
         names=["bradesco"]),
    dict(firm_key="santander_br", display_name="Banco Santander (Brasil)",
         segment="incumbent", sec_cik="0001471055", b3_ticker="SANB11", cvm_code="SANTANDERBR",
         cnpj_root="90400888", country_scope="brazil", has_sec_deposits=True,
         names=["santander"]),
    dict(firm_key="bb", display_name="Banco do Brasil",
         segment="incumbent", sec_cik=None, b3_ticker="BBAS3", cvm_code="BBRASIL",
         cnpj_root="00000000", country_scope="brazil", has_sec_deposits=False,
         names=["banco do brasil"]),
    dict(firm_key="caixa", display_name="Caixa Econômica Federal",
         segment="incumbent", sec_cik=None, b3_ticker=None, cvm_code=None,
         cnpj_root="00360305", country_scope="brazil", has_sec_deposits=False,
         names=["caixa economica", "caixa econômica"]),
]


def _norm(s: str) -> str:
    s = str(s).lower()
    for a, b in [("á", "a"), ("â", "a"), ("ã", "a"), ("é", "e"), ("ê", "e"),
                 ("í", "i"), ("ó", "o"), ("ô", "o"), ("õ", "o"), ("ú", "u"), ("ç", "c")]:
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9 ]+", " ", s)


def _load_ifdata_namemap(base: str) -> list[dict]:
    """Build [{name_norm, cnpj_root, cong_prud}] from IF Data List CSVs."""
    import pandas as pd
    files = sorted(glob.glob(os.path.join(base, "BCB", "IF Data", "List", "IF_DATA_List*.csv")))
    if not files:
        # alternate naming used by scrape_1
        files = sorted(glob.glob(os.path.join(base, "BCB", "IF Data", "List", "*.csv")))
    rows: list[dict] = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=None, engine="python", dtype=str)
        except Exception as e:  # noqa: BLE001
            log.debug(f"firm_registry: cannot read {f}: {e}")
            continue
        name_col = next((c for c in df.columns if c.lower() in ("nomeinstituicao", "nome")), None)
        cnpj_col = next((c for c in df.columns if "cnpj" in c.lower()), None)
        cong_col = next((c for c in df.columns if "prudencial" in c.lower()), None)
        if not name_col:
            continue
        for _, r in df.iterrows():
            nm = r.get(name_col)
            if not isinstance(nm, str):
                continue
            cnpj = re.sub(r"\D", "", str(r.get(cnpj_col, "")))[:8].zfill(8) if cnpj_col else ""
            rows.append(dict(name_norm=_norm(nm), cnpj_root=cnpj,
                             cong_prud=str(r.get(cong_col, "")) if cong_col else ""))
    return rows


def load_registry(base: str, enrich: bool = True) -> list[dict]:
    """Return the firm registry. When `enrich`, validate/override each seed
    cnpj_root and attach `cong_prud` (prudential conglomerate code) by matching
    `names` fragments against BCB IF Data List files."""
    firms = [dict(f) for f in FIRMS]
    for f in firms:
        f.setdefault("cong_prud", "")
        f["cnpj_root"] = str(f.get("cnpj_root", "")).zfill(8)

    if not enrich:
        return firms

    try:
        namemap = _load_ifdata_namemap(base)
    except Exception as e:  # noqa: BLE001
        log.warning(f"firm_registry: IF Data enrichment skipped ({e})")
        return firms
    if not namemap:
        log.warning("firm_registry: no IF Data List rows found; using seed CNPJ roots only.")
        return firms

    for f in firms:
        frags = [_norm(x) for x in f.get("names", [])]
        best = None
        for rec in namemap:
            if any(fr and fr in rec["name_norm"] for fr in frags):
                # prefer a match that actually has a cnpj root
                if best is None or (not best["cnpj_root"] and rec["cnpj_root"]):
                    best = rec
        if best and best["cnpj_root"] and best["cnpj_root"] != "00000000":
            if best["cnpj_root"] != f["cnpj_root"]:
                log.info(f"firm_registry[{f['firm_key']}]: cnpj_root "
                         f"{f['cnpj_root']} -> {best['cnpj_root']} (from IF Data)")
            f["cnpj_root"] = best["cnpj_root"]
            f["cong_prud"] = best.get("cong_prud", "") or f.get("cong_prud", "")
    return firms


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.normpath(os.path.join(here, "..", "..", ".."))
    reg = load_registry(base)
    print(f"{'key':14s} {'seg':10s} {'cnpj_root':10s} {'cong':6s} {'cik':12s} scope")
    for f in reg:
        print(f"{f['firm_key']:14s} {f['segment']:10s} {f['cnpj_root']:10s} "
              f"{str(f.get('cong_prud','')):6s} {str(f.get('sec_cik')):12s} {f['country_scope']}")
