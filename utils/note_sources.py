"""note_sources.py

Where the STATEMENT-NOTE route may fetch from, and for which banks.

The route exists because the structured filings do not carry the number. Every bank publishes
audited financial statements twice a year, and the administrative-expenses note itemises
advertising ("Propaganda e publicidade") - the same accounting concept as the 2025 COSIF account
`8174500005`, but available from 2013. The securities regulator's STRUCTURED data
(`dados.cvm.gov.br`, already collected by `scrape_cvm_advertising.py`) only exposes the DVA and
DRE faces, where most banks show no advertising line at all: Santander Brasil is 9.5% of panel
deposits and files every quarter, yet has no advertising line in any structured year. Its figure
is in the notes, and so is every other missing bank's.

Two routes, by whether the bank files with the securities regulator:
  cvm_rad   a CVM filer: the full DFP/ITR document, notes included, comes from the regulator's own
            document system (`rad.cvm.gov.br`). One host serves every filer, so this is the
            cheapest route and the first to build.
  ir_site   not a CVM filer: the bank publishes the same statements on its own domain, because the
            central bank requires publication regardless of listing.

HOSTS ARE AN ALLOWLIST, NOT A SUGGESTION. `allowed_host` raises on any host not listed here. It is a
check, not a barrier: it protects exactly the URLs a caller passes through it, and importing this
module stops nothing by itself. `scrape_bank_notes_advertising.py` passes every URL it requests
through it, and the target of every redirect before following it (its `fetch`); what that does and
does not guarantee is set out in that module's docstring. The user approved the hosts below on
2026-09-22; `reachable` records whether a probe has confirmed the domain serves, and is False until
one has.

This module holds no fetching logic and no credentials: it is the authorization surface and the
target list, kept separate so that what may be contacted is reviewable in one place.
"""

from __future__ import annotations

import urllib.parse

# ---------------------------------------------------------------------------
# Approved hosts
# ---------------------------------------------------------------------------
# host -> (what it serves, reachability confirmed by a probe)
NOTE_HOSTS: dict[str, tuple[str, bool]] = {
    "rad.cvm.gov.br": ("securities regulator document system: full DFP/ITR with notes, every "
                       "CVM filer", False),
    "www.rad.cvm.gov.br": ("same system, www form", False),
    # Banks that do not file with the securities regulator publish on their own domains.
    "www.santander.com.br": ("Santander Brasil statements (also a CVM filer; kept as a fallback)",
                             False),
    "ri.santander.com.br": ("Santander Brasil investor relations", False),
    "ri.btgpactual.com": ("BTG Pactual investor relations", False),
    "www.citibank.com.br": ("Citibank Brasil statements", False),
    "www.safra.com.br": ("Banco Safra statements", False),
    "www.sicredi.com.br": ("Sicredi cooperative system statements", False),
    "www.sicoob.com.br": ("Sicoob cooperative system statements", False),
    "ri.daycoval.com.br": ("Banco Daycoval investor relations", False),
    "www.bancomaster.com.br": ("Banco Master statements", False),
    "www.agibank.com.br": ("Agibank statements", False),
    "www.mercadopago.com.br": ("Mercado Pago IP statements", False),
    "www.pagbank.com.br": ("PagBank group, including BancoSeguro", False),
    # PagBank's listing links its statement PDFs to PagSeguro's static-file server (user approved
    # 2026-09-30).
    "acq-static-pages.pagseguro.com.br": ("PagSeguro static files: PagBank statement PDFs", False),
    "www.vwfs.com.br": ("Volkswagen Financial Services Brasil, Banco Volkswagen", False),
    # Groups listed abroad (Nu Holdings, Inter & Co, XP Inc), so the Brazilian bank's statements sit
    # on no regulator path and only its own domain serves them. Approved by the user 2026-09-23.
    "www.nubank.com.br": ("Nu Pagamentos / Nu Financeira statements", False),
    "nubank.com.br": ("same, bare form", False),
    "www.xpi.com.br": ("Banco XP statements", False),
    "inter.co": ("Banco Inter statements", False),
    "www.inter.co": ("same, www form", False),
    # Already used elsewhere in this project, repeated here so the route needs no exception.
    "www.c6bank.com.br": ("Banco C6 statements; host already used by the disclosure scrapers",
                          False),
    "www.picpay.com": ("PicPay Bank statements; host already used", False),
}


def allowed_host(url: str) -> str:
    """The host of `url`, if it is approved for this route. Raises PermissionError otherwise.

    It looks at the host NAME of the one URL it is given, and nothing else: it does not fetch, so
    it cannot see a redirect, and it knows nothing of where the name resolves. A typo in a URL
    becomes an error only where a caller runs the URL through it before sending the request, and a
    redirect onto a third-party CDN only where the caller runs every redirect target through it
    before following it, as `scrape_bank_notes_advertising.fetch` does.
    """
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host not in NOTE_HOSTS:
        raise PermissionError(
            f"{host or url!r} is not an approved host for the statement-note route. Approved: "
            + ", ".join(sorted(NOTE_HOSTS)))
    return host


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
# Chosen from the data, not by reputation: these are the conglomerates present in the E3 spec-12
# estimation sample that have NO quarterly advertising series, ranked by their 2024Q4 share of panel
# deposits. The seventeen already covered are not here. `share_2024q4` is that share in per cent,
# and the eighteen below are about 26 points of it, which is most of the 31 points now uncovered.
#
# cvm_code is VERIFIED against the cached DFP 2024 filing set where present; None means the bank did
# not file that year, so it takes the ir_site route. A verified code matters because the document
# system is keyed on it.
NOTE_TARGETS: list[dict] = [
    dict(panel_code="C0080185", name="Banco Santander (Brasil) S.A.", cnpj8="90400888",
         share_2024q4=9.54, route="cvm_rad", cvm_code="20532", host="rad.cvm.gov.br",
         note="files every quarter, no advertising line in any structured year: the reason this "
              "route exists"),
    dict(panel_code="C0084693", name="Nu Pagamentos / Nu Financeira", cnpj8="18236120",
         share_2024q4=3.04, route="ir_site", cvm_code=None, host="www.nubank.com.br",
         note="SEC 20-F already gives a group-wide annual figure in USD; the local statements give "
              "the Brazil entity twice a year in BRL"),
    dict(panel_code="C0080336", name="Banco BTG Pactual S.A.", cnpj8="30306294",
         share_2024q4=1.82, route="cvm_rad", cvm_code="22616", host="rad.cvm.gov.br"),
    dict(panel_code="C0080192", name="Banco Citibank S.A.", cnpj8="33479023",
         share_2024q4=1.71, route="ir_site", cvm_code=None, host="www.citibank.com.br"),
    dict(panel_code="C0080109", name="Banco Safra S.A.", cnpj8="58160789",
         share_2024q4=1.23, route="ir_site", cvm_code=None, host="www.safra.com.br"),
    dict(panel_code="C0082475", name="Banco XP S.A.", cnpj8="33264668",
         share_2024q4=1.18, route="ir_site", cvm_code=None, host="www.xpi.com.br",
         note="XP Inc files with the SEC; the Brazilian bank's own statements carry the local line"),
    dict(panel_code="C0080996", name="Banco Inter S.A.", cnpj8="00416968",
         share_2024q4=0.94, route="ir_site", cvm_code=None, host="inter.co",
         note="not in the 2024 DFP set: the group relisted as Inter & Co on Nasdaq"),
    dict(panel_code="C0084844", name="Banco C6 S.A.", cnpj8="31872495",
         share_2024q4=0.84, route="ir_site", cvm_code=None, host="www.c6bank.com.br"),
    dict(panel_code="C0080745", name="Banco Cooperativo Sicredi S.A.", cnpj8="01181521",
         share_2024q4=0.83, route="ir_site", cvm_code=None, host="www.sicredi.com.br"),
    dict(panel_code="C0084813", name="BancoSeguro S.A. (PagBank group)", cnpj8="08561701",
         share_2024q4=0.73, route="ir_site", cvm_code=None, host="www.pagbank.com.br"),
    dict(panel_code="C0080367", name="Banco Master S.A.", cnpj8="80271455",
         share_2024q4=0.67, route="ir_site", cvm_code=None, host="www.bancomaster.com.br",
         note="in extrajudicial liquidation; statements for the years it operated still stand"),
    dict(panel_code="C0084820", name="Mercado Pago IP", cnpj8="10573521",
         share_2024q4=0.62, route="ir_site", cvm_code=None, host="www.mercadopago.com.br"),
    dict(panel_code="C0081744", name="Banco Daycoval S.A.", cnpj8="62232889",
         share_2024q4=0.54, route="cvm_rad", cvm_code="20796", host="rad.cvm.gov.br"),
    dict(panel_code="C0081737", name="Banco C6 Consignado S.A.", cnpj8="31872495",
         share_2024q4=0.53, route="ir_site", cvm_code=None, host="www.c6bank.com.br"),
    dict(panel_code="C0080879", name="Banco Cooperativo Sicoob S.A.", cnpj8="02038232",
         share_2024q4=0.46, route="ir_site", cvm_code=None, host="www.sicoob.com.br"),
    dict(panel_code="C0088022", name="PicPay Bank", cnpj8="09516419",
         share_2024q4=0.43, route="ir_site", cvm_code=None, host="www.picpay.com",
         note="its 2024 note 18 was already read by text extraction while proving the route"),
    dict(panel_code="C0080202", name="Banco Volkswagen S.A.", cnpj8="59109165",
         share_2024q4=0.40, route="ir_site", cvm_code=None, host="www.vwfs.com.br"),
    dict(panel_code="C0083694", name="Banco Agibank S.A.", cnpj8="10664513",
         share_2024q4=0.36, route="ir_site", cvm_code=None, host="www.agibank.com.br"),
]


def targets(route: str | None = None) -> list[dict]:
    """Targets, optionally one route only, largest deposit share first."""
    rows = [t for t in NOTE_TARGETS if route is None or t["route"] == route]
    return sorted(rows, key=lambda t: -t["share_2024q4"])


def unresolved() -> list[dict]:
    """Targets whose document source is not yet pinned down: no CVM code and no host.

    Kept explicit so the gap is visible in the registry rather than discovered mid-run.
    """
    return [t for t in NOTE_TARGETS if not t.get("cvm_code") and not t.get("host")]
