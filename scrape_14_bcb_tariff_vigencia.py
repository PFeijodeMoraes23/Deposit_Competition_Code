"""
scrape_14_bcb_tariff_vigencia.py
==================================
Extract the full DataVigencia (effective-date) history from the BCB Tarifas
API to reconstruct a synthetic historical listed-price panel.

Why this matters
-----------------
The BCB Olinda API (`ListaTarifasPorInstituicaoFinanceira`) only returns
CURRENT tariffs — but each row carries a `DataVigencia` field: the date the
current price became effective. This encodes the *last price change* for each
(institution, service) pair, giving us:

  price_value  |  from DataVigencia  →  until next scrape that shows a change

Combined with the forward-looking panel from scrape_11_fees.py (which captures
FUTURE changes), we reconstruct the full price path:

  [pre-DataVigencia]  → unknown (pre-2018 typically)
  [DataVigencia → first scrape change]  → current price (known)
  [subsequent scrapes]  → whatever changes we capture going forward

Output
------
  BCB/Tarifas/processed/tariff_vigencia_history.csv
    cnpj | nome_instituicao | customer_type | codigo_servico | servico
    | data_vigencia | valor_maximo | data_coleta

  BCB/Tarifas/processed/tariff_vigencia_summary.csv
    Pivot: conglomerate × service_code → (data_vigencia, valor_maximo)
    Shows for each institution and service: when the current price was set.

Usage
-----
  python scrape_14_bcb_tariff_vigencia.py
  python scrape_14_bcb_tariff_vigencia.py --test 20
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants (reuse scrape_11_fees.py infrastructure)
# ---------------------------------------------------------------------------
from utils import paths
_REPO    = Path(__file__).resolve().parents[2]
OUT_DIR  = paths.TARIFAS_PROC
IF_LIST_DIR = paths.IF_DATA_LIST

API_BASE = (
    "https://olinda.bcb.gov.br/olinda/servico/"
    "Informes_ListaTarifasPorInstituicaoFinanceira/versao/v1/odata"
)
CUSTOMER_TYPES = ["F", "J"]
INTER_CALL_SLEEP = 0.4
TODAY = date.today().isoformat()

# Services most relevant to deposit competition and switching costs
KEY_SERVICES = {
    "F": ["1101", "1201", "1209", "1210", "1307", "1311", "1314", "1315",
          "1401", "1501", "1502", "1503"],
    "J": ["0402", "0403", "0404", "0501", "0502", "0506"],
}


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"Accept": "application/json"})
    return s


SESSION = _session()


def _odata_get(url: str, params: dict | None = None) -> list[dict]:
    if params is None:
        params = {}
    params.setdefault("$format", "json")
    params.setdefault("$top", 9999)
    records: list[dict] = []
    while url:
        try:
            r = SESSION.get(url, params=params, timeout=30)
            r.raise_for_status()
        except Exception as exc:
            log.debug("Request failed for %s: %s", url, exc)
            break
        data = r.json()
        records.extend(data.get("value", []))
        url = data.get("@odata.nextLink")
        params = {}
    return records


# ---------------------------------------------------------------------------
# Institution list
# ---------------------------------------------------------------------------

def fetch_all_cnpjs() -> pd.DataFrame:
    groups = _odata_get(f"{API_BASE}/GruposConsolidados")
    frames = []
    for g in groups:
        code = g["Codigo"]
        recs = _odata_get(
            f"{API_BASE}/ListaInstituicoesDeGrupoConsolidado(CodigoGrupoConsolidado='{code}')"
        )
        df = pd.DataFrame(recs)
        df["grupo_codigo"] = code
        df["grupo_nome"] = g["Nome"]
        frames.append(df)
    all_inst = pd.concat(frames, ignore_index=True)
    all_inst["Cnpj"] = all_inst["Cnpj"].astype(str).str.strip().str.zfill(8)
    all_inst = all_inst.drop_duplicates(subset="Cnpj", keep="first")
    log.info("Institutions: %d CNPJs across %d groups", len(all_inst), len(groups))
    return all_inst


# ---------------------------------------------------------------------------
# Fetch tariffs with vigencia for all institutions
# ---------------------------------------------------------------------------

def fetch_vigencia_panel(institutions: pd.DataFrame, test_n: int | None = None) -> pd.DataFrame:
    """
    For each institution × customer_type, fetch the current tariff schedule
    (which includes DataVigencia).  Returns a long DataFrame.
    """
    if test_n:
        institutions = institutions.head(test_n)
        log.info("TEST MODE: %d institutions", len(institutions))

    all_rows: list[dict] = []
    n = len(institutions)

    for i, row in enumerate(institutions.itertuples(index=False), 1):
        if i % 50 == 0 or i == 1 or i == n:
            log.info("  [%d/%d] %s (%s)", i, n, row.Nome, row.Cnpj)

        for ct in CUSTOMER_TYPES:
            url = (
                f"{API_BASE}/ListaTarifasPorInstituicaoFinanceira"
                f"(PessoaFisicaOuJuridica='{ct}',CNPJ='{row.Cnpj}')"
            )
            recs = _odata_get(url)
            for r in recs:
                all_rows.append({
                    "cnpj":             row.Cnpj,
                    "nome_instituicao": row.Nome,
                    "grupo_bcb":        row.grupo_nome,
                    "customer_type":    ct,
                    "codigo_servico":   str(r.get("CodigoServico", "")).zfill(4),
                    "servico":          r.get("Servico", ""),
                    "unidade":          r.get("Unidade", ""),
                    "tipo_valor":       r.get("TipoValor", ""),
                    "periodicidade":    r.get("Periodicidade", ""),
                    "data_vigencia":    r.get("DataVigencia"),
                    "valor_maximo":     r.get("ValorMaximo"),
                    "data_coleta":      TODAY,
                })
            time.sleep(INTER_CALL_SLEEP)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["valor_maximo"]  = pd.to_numeric(df["valor_maximo"], errors="coerce")
    df["data_vigencia"] = pd.to_datetime(df["data_vigencia"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Build summary pivot: what was the price, and since when?
# ---------------------------------------------------------------------------

def build_vigencia_summary(long_df: pd.DataFrame, cmap: pd.DataFrame) -> pd.DataFrame:
    """
    Wide pivot: for each (conglomerate, customer_type, service_code),
    report (data_vigencia, valor_maximo).

    Interpretation: institution X has charged R$Y for service Z since date D.
    This anchors the historical listed-price series.
    """
    df = long_df.copy()
    df["cnpj"] = df["cnpj"].astype(str).str.zfill(8)
    df = df.merge(cmap, on="cnpj", how="left")
    df["cod_cong_prudencial"] = df["cod_cong_prudencial"].fillna(df["cnpj"])

    # Aggregate at conglomerate level: max price, earliest vigencia across members
    grp = df.groupby(
        ["cod_cong_prudencial", "customer_type", "codigo_servico", "servico"],
        as_index=False,
    ).agg(
        valor_maximo_max    = ("valor_maximo", "max"),
        valor_maximo_median = ("valor_maximo", "median"),
        data_vigencia_min   = ("data_vigencia", "min"),   # earliest a member changed
        data_vigencia_max   = ("data_vigencia", "max"),   # most recent change
        n_institutions      = ("cnpj", "nunique"),
    )

    log.info(
        "Vigencia summary: %d rows | %d conglomerates | %d services",
        len(grp),
        grp["cod_cong_prudencial"].nunique(),
        grp["codigo_servico"].nunique(),
    )
    return grp


def build_cnpj_cong_map() -> pd.DataFrame:
    list_files = sorted(IF_LIST_DIR.glob("IF_DATA_List_*.csv"))
    list_files = [f for f in list_files if f.stat().st_size > 500]
    if not list_files:
        return pd.DataFrame(columns=["cnpj", "cod_cong_prudencial"])
    df = pd.read_csv(list_files[-1], sep=",", encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["CodInst_str"] = df["CodInst"].astype(str).str.strip()
    numeric = df["CodInst_str"].str.match(r"^\d+$")
    part_a = df.loc[numeric, ["CodInst_str", "CodConglomeradoPrudencial"]].copy()
    part_a.columns = ["cnpj", "cod_cong_prudencial"]
    part_a["cnpj"] = part_a["cnpj"].str.zfill(8)
    leader_valid = df["CnpjInstituicaoLider"].notna()
    part_b = df.loc[leader_valid, ["CnpjInstituicaoLider", "CodConglomeradoPrudencial"]].copy()
    part_b.columns = ["cnpj", "cod_cong_prudencial"]
    part_b["cnpj"] = part_b["cnpj"].astype(str).str.replace(r"\.0$","",regex=True).str.strip().str.zfill(8)
    cmap = (pd.concat([part_a, part_b], ignore_index=True)
            .dropna(subset=["cod_cong_prudencial"])
            .drop_duplicates(subset="cnpj", keep="first"))
    cmap["cod_cong_prudencial"] = cmap["cod_cong_prudencial"].astype(str).str.strip()
    return cmap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(test_n: int | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== Step 1: fetching institution list ===")
    institutions = fetch_all_cnpjs()

    log.info("=== Step 2: fetching tariffs with DataVigencia ===")
    long_df = fetch_vigencia_panel(institutions, test_n=test_n)

    if long_df.empty:
        log.error("No data fetched.")
        return

    # Save long panel
    long_path = OUT_DIR / "tariff_vigencia_history.csv"
    long_df.to_csv(long_path, index=False)
    log.info("Long panel: %d rows → %s", len(long_df), long_path.name)

    log.info("=== Step 3: building vigencia summary ===")
    cmap = build_cnpj_cong_map()
    summary = build_vigencia_summary(long_df, cmap)

    sum_path = OUT_DIR / "tariff_vigencia_summary.csv"
    summary.to_csv(sum_path, index=False)
    log.info("Vigencia summary: %d rows → %s", len(summary), sum_path.name)

    # Quick report: oldest vigencia dates (prices that haven't changed in a long time)
    old = summary.nsmallest(10, "data_vigencia_min")[
        ["cod_cong_prudencial", "customer_type", "codigo_servico",
         "servico", "valor_maximo_max", "data_vigencia_min"]
    ]
    log.info("Oldest price points (set earliest, never updated):\n%s", old.to_string(index=False))

    log.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCB Tarifas DataVigencia extractor")
    parser.add_argument("--test", type=int, default=None, metavar="N",
                        help="Test with first N institutions only")
    args = parser.parse_args()
    main(test_n=args.test)
