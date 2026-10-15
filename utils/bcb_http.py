"""utils/bcb_http.py — shared HTTP plumbing for the BCB scrapers.

retry_session()  a requests.Session that retries connection errors and 429/5xx responses with
                 exponential backoff (urllib3 Retry), so a caller does not hand-roll a loop.
cms_documents()  one folder of the www.bcb.gov.br CMS listing (Documentos/byListGuid), which
                 serves the ESTBAN and COSIF files: a list of {Nome, Url, DataDocumento, Tamanho},
                 newest first.

Used by scrape_cosif_download.py and scrape_bcb_estban_ifdata.py.
"""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BCB_SITE     = "https://www.bcb.gov.br"
CMS_LIST_API = BCB_SITE + "/api/servico/sitebcb/Documentos/byListGuid"
BROWSER_UA   = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def retry_session(total: int = 3, backoff: float = 2.0,
                  statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
                  headers: dict | None = None) -> requests.Session:
    """A Session retrying GET/HEAD `total` times on connection errors and on `statuses`.

    After the last retry the response is returned as is, so `raise_for_status()` still raises
    for the caller to record the failure.
    """
    s = requests.Session()
    retry = Retry(total=total, connect=total, read=total, status=total,
                  backoff_factor=backoff, status_forcelist=statuses,
                  allowed_methods=frozenset(["GET", "HEAD"]), raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": BROWSER_UA, **(headers or {})})
    return s


def cms_documents(session: requests.Session, tronco: str, guid: str, pasta: str,
                  timeout: int = 60, referer: str | None = None) -> list[dict]:
    """One folder of the BCB CMS listing, newest first. Raises on an HTTP error."""
    params = {"tronco": tronco, "guidLista": guid, "ordem": "DataDocumento desc", "pasta": pasta}
    headers = {"Accept": "application/json"}
    if referer:
        headers["Referer"] = referer
    r = session.get(CMS_LIST_API, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json().get("conteudo", [])
