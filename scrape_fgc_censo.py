###─────────────────────────────────────────────────────────────────────────────
# scrape_fgc_censo.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   The FGC monthly census ("Censo"): balances and NUMBER OF CLIENTS of every
#   FGC-eligible product (demand, savings, CDB/RDB, LCI, LCA, ...), by prudential
#   segment (S1-S5), holder type and 27 value bands, December 2019 onwards. It is
#   the only public source found with client counts by deposit instrument.
#
# DATA SOURCE
#   https://www.fgc.org.br/pesquisa-e-base-de-dados links one XLSX, "Dados Censo"
#   (asset name carries the upload date, e.g. dados-censo_20250714), about 82 MB.
#   The page is scraped for the current link, so a newer upload is picked up.
#   Sheets: "Censo Produto" (segment x band x product x guarantee x holder type),
#   "Censo Titularidade" (no product; adds the covered amount), "Censo Corretora"
#   (balances distributed by brokers), "Dicionário de dados".
#   No institution dimension: the FGC receives the census by conglomerate but
#   publishes it by segment only. A person with accounts at two conglomerates is
#   counted twice (clients are counted per conglomerate).
#
# PRODUCT CODES (num_produto)
#   1 demand deposits; 2 savings; 3 time deposits without special guarantee (CDB,
#   RDB, others); 4 DPGE; 5 LC; 6 LH; 7 LCA; 8 LCI; 9 deposits not movable by
#   cheque; 10 repos on related-issuer securities; 11 inactive-account deposits; 12 LCD.
#
# OUTPUTS (raw data area; nothing in the estimation pipeline reads them)
#   RAW/FGC_CENSO/<file>.xlsx                 the download, as served
#   RAW/FGC_CENSO/censo_produto.parquet       the Produto sheet, typed
#   RAW/FGC_CENSO/censo_titularidade.parquet  the Titularidade sheet, typed
#   RAW/FGC_CENSO/censo_produto_monthly.csv   month x segment x product x holder type,
#                                             summed over value bands (clients, value)
#
# CLI
#   python scrape_fgc_censo.py            # download if the file is new, rebuild outputs
#   python scrape_fgc_censo.py --refresh  # download again
###─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import logging
import re
import urllib.parse
import urllib.request

try:
    from utils.venv_guard import ensure_project_venv
except ImportError:  # pragma: no cover
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd

from utils import paths

log = logging.getLogger("fgc_censo")
OUT = paths.RAW / "FGC_CENSO"
PAGE = "https://www.fgc.org.br/pesquisa-e-base-de-dados"
UA = {"User-Agent": "Mozilla/5.0 (research; deposit competition)"}
PRODUCTS = {1: "demand", 2: "savings", 3: "time_cdb_rdb", 4: "dpge", 5: "lc", 6: "lh", 7: "lca",
            8: "lci", 9: "non_cheque_deposits", 10: "repos_related", 11: "inactive_accounts", 12: "lcd"}
NUMERIC = ["cod_competencia", "num_faixa_valor", "val_limite_inferior", "val_limite_superior",
           "num_produto", "num_garantia", "num_titularidade", "qtd_cliente", "val_deposito", "val_coberto"]


def fetch(url: str, timeout: int = 300) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def census_link() -> str:
    html = fetch(PAGE, 60).decode("utf-8", errors="replace")
    links = re.findall(r'href="([^"]*dados-censo[^"]*)"', html, flags=re.I)
    if not links:
        raise RuntimeError(f"no 'dados-censo' link found on {PAGE}")
    link = urllib.parse.urljoin("https://www.fgc.org.br/", links[0].replace("&amp;", "&"))
    return link if "download=true" in link else link + ("&" if "?" in link else "?") + "download=true"


def typed(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # The text columns arrive cp1252-mangled; the numeric codes carry the meaning.
    for c in [c for c in df.columns if c.startswith("des_")]:
        df[c] = df[c].astype(str)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    url = census_link()
    name = re.sub(r"[^\w.-]", "_", url.split("/")[-1].split("?")[0]) + ".xlsx"
    dest = OUT / name
    if a.refresh or not dest.exists():
        log.info("downloading %s", url)
        dest.write_bytes(fetch(url))
    log.info("file: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)

    prod = typed(pd.read_excel(dest, sheet_name="Censo Produto", engine="openpyxl"))
    prod.to_parquet(OUT / "censo_produto.parquet", index=False)
    tit = typed(pd.read_excel(dest, sheet_name="Censo Titularidade", engine="openpyxl"))
    tit.to_parquet(OUT / "censo_titularidade.parquet", index=False)

    p = prod.dropna(subset=["num_produto"]).copy()
    p["product"] = p["num_produto"].astype(int).map(PRODUCTS)
    monthly = (p.groupby(["cod_competencia", "sr", "num_produto", "product", "num_titularidade"], dropna=False)
                 [["qtd_cliente", "val_deposito"]].sum().reset_index())
    monthly.to_csv(OUT / "censo_produto_monthly.csv", index=False, encoding="utf-8")
    log.info("Produto rows %d, months %d (%s to %s); Titularidade rows %d",
             len(prod), prod["cod_competencia"].nunique(), int(prod["cod_competencia"].min()),
             int(prod["cod_competencia"].max()), len(tit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
