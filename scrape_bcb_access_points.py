###─────────────────────────────────────────────────────────────────────────────
# scrape_bcb_access_points.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Per-INSTITUTION physical access points (agências, postos de atendimento,
#   correspondentes) by municipality, from the three BCB "Informes" open-data
#   services. This is the evidence base for the brick-and-mortar (B) vs digital
#   (D) firm classification consumed by panel_digital_flags.py.
#
# WHY THIS EXISTS
#   ESTBAN's AGEN_PROCESSADAS counts only full regulated *agências*. A bank that
#   serves customers through *postos de atendimento* or *correspondentes* has a
#   real retail footprint that ESTBAN cannot see. Agibank is the motivating case:
#   ESTBAN reports exactly one agência in one municipality every month since
#   2016, while the institution operates a large storefront network. Classifying
#   on ESTBAN alone therefore labels a hybrid bank "digital".
#
#   Note also that bcb_inclusion_mca_panel.csv carries correspondents = 0 for
#   every row (scrape_bcb_inclusion.py sets the column to a constant and never
#   calls an API), so access_points_total there equals branches_total exactly.
#   This script does NOT patch that panel — see the VINTAGE caveat below.
#
# DATA SOURCE
#   https://olinda.bcb.gov.br/olinda/servico/<service>/versao/v1/odata/<entity>
#     Informes_Agencias             -> Agencias           (CnpjBase, MunicipioIbge, DataInicio)
#     Informes_PostosDeAtendimento  -> PostosAtendimento  (Cnpj, MunicipioIbge, TipoPosto)
#     Informes_Correspondentes      -> Correspondentes    (CnpjContratante, MunicipioIBGE)
#
#   The services reject $filter with a type error ("The types 'Edm.Boolean' and
#   'Edm.String' are not compatible"), so the tables are paged in full with
#   $select/$top/$skip and filtered locally.
#
# VINTAGE CAVEAT — THIS IS A CROSS-SECTION, NOT A PANEL
#   Every row carries the same Posicao (the snapshot date the BCB last
#   published; 25/08/2026 at time of writing). These services report the CURRENT
#   state of the network only. Do not broadcast these counts across panel years
#   as if they were a time series: that would manufacture fake time variation in
#   a market characteristic. They are used here for a firm-level attribute
#   ("does this institution operate a physical retail network?"), which is
#   slow-moving, and the snapshot date is recorded in the output.
#
#   Agencias.DataInicio gives each branch's opening date and would support a
#   partial historical reconstruction, but only of branches still open today —
#   closures are absent from the snapshot — so it is stored, not used to build a
#   back-series.
#
# OUTPUT (paths.INCLUSION_DIR)
#   bcb_access_points_raw_<entity>.csv     one row per physical point (cached)
#   bcb_access_points_by_institution.csv   cnpj8 x counts + municipality spread
#   bcb_access_points_by_inst_muni.csv     cnpj8 x municipality x counts
#
# Usage:  python scrape_bcb_access_points.py            # cached if present
#         python scrape_bcb_access_points.py --refresh  # force re-download
###─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.parse
import urllib.request

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from utils import paths

OUT_DIR = paths.INCLUSION_DIR
os.makedirs(OUT_DIR, exist_ok=True)

BY_INST_CSV = OUT_DIR / "bcb_access_points_by_institution.csv"
BY_MUNI_CSV = OUT_DIR / "bcb_access_points_by_inst_muni.csv"

# Set from --refresh in __main__; module-level so build() works on import too.
REFRESH = False

ODATA = "https://olinda.bcb.gov.br/olinda/servico/{svc}/versao/v1/odata/{ent}"
PAGE = 10_000

# (service, entity, cnpj field, municipality field, extra fields, output tag)
SPECS = [
    ("Informes_Agencias", "Agencias",
     "CnpjBase", "MunicipioIbge", ["NomeIf", "DataInicio"], "agencias"),
    ("Informes_PostosDeAtendimento", "PostosAtendimento",
     "Cnpj", "MunicipioIbge", ["NomeIf", "TipoPosto"], "postos"),
    ("Informes_Correspondentes", "Correspondentes",
     "CnpjContratante", "MunicipioIBGE", ["NomeContratante"], "correspondentes"),
]


def _fetch_page(svc: str, ent: str, select: str, skip: int, top: int) -> list[dict]:
    url = ODATA.format(svc=svc, ent=ent) + "?" + urllib.parse.urlencode(
        {"$format": "json", "$select": select, "$top": str(top), "$skip": str(skip)}
    )
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return json.load(r)["value"]
        except Exception as e:                      # noqa: BLE001 - transient API
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{ent} skip={skip} failed after 4 attempts: {last}")


def download_entity(svc: str, ent: str, cnpj_f: str, mun_f: str,
                    extra: list[str], tag: str, refresh: bool) -> pd.DataFrame:
    """Page one Informes entity in full. Cached to CSV; --refresh forces re-download."""
    cache = OUT_DIR / f"bcb_access_points_raw_{tag}.csv"
    if cache.exists() and not refresh:
        logging.info(f"  [{tag}] cached -> {cache.name}")
        return pd.read_csv(cache, dtype=str)

    select = ",".join([cnpj_f, mun_f, *extra, "Posicao"])
    rows: list[dict] = []
    skip = 0
    t0 = time.time()
    while True:
        page = _fetch_page(svc, ent, select, skip, PAGE)
        rows.extend(page)
        if len(page) < PAGE:
            break
        skip += PAGE
        if skip % 100_000 == 0:
            logging.info(f"  [{tag}] {skip:,} rows …")
    df = pd.DataFrame(rows, dtype=str)
    logging.info(f"  [{tag}] {len(df):,} rows in {time.time() - t0:.0f}s -> {cache.name}")
    df.to_csv(cache, index=False)
    return df


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    posicao = None
    for svc, ent, cnpj_f, mun_f, extra, tag in SPECS:
        df = download_entity(svc, ent, cnpj_f, mun_f, extra, tag, REFRESH)
        if df.empty:
            logging.warning(f"  [{tag}] returned no rows")
            continue
        if "Posicao" in df.columns and df["Posicao"].notna().any():
            posicao = df["Posicao"].dropna().iloc[0]
        name_col = next((c for c in ("NomeIf", "NomeContratante") if c in df.columns), None)
        out = pd.DataFrame({
            # CnpjBase/Cnpj/CnpjContratante are all 8-digit CNPJ roots, matching
            # the CNPJ_root key used by panel_5 and panel_1.
            "cnpj8": df[cnpj_f].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8),
            "mun_code": pd.to_numeric(df[mun_f], errors="coerce"),
            "kind": tag,
            "name": df[name_col].astype(str) if name_col else "",
        })
        parts.append(out.dropna(subset=["mun_code"]))

    long = pd.concat(parts, ignore_index=True)
    long["mun_code"] = long["mun_code"].astype(int)

    by_muni = (long.groupby(["cnpj8", "mun_code", "kind"]).size()
                   .unstack("kind", fill_value=0).reset_index())
    for k in ("agencias", "postos", "correspondentes"):
        if k not in by_muni.columns:
            by_muni[k] = 0
    by_muni = by_muni[["cnpj8", "mun_code", "agencias", "postos", "correspondentes"]]

    names = (long[long["name"].astype(bool)]
             .drop_duplicates("cnpj8").set_index("cnpj8")["name"])

    def _muns(mask_col: str) -> pd.Series:
        sub = by_muni[by_muni[mask_col] > 0]
        return sub.groupby("cnpj8")["mun_code"].nunique()

    by_inst = pd.DataFrame({
        "n_agencias":        by_muni.groupby("cnpj8")["agencias"].sum(),
        "n_postos":          by_muni.groupby("cnpj8")["postos"].sum(),
        "n_correspondentes": by_muni.groupby("cnpj8")["correspondentes"].sum(),
        "n_mun_agencias":        _muns("agencias"),
        "n_mun_postos":          _muns("postos"),
        "n_mun_correspondentes": _muns("correspondentes"),
        "n_mun_any":         by_muni.groupby("cnpj8")["mun_code"].nunique(),
    }).fillna(0).astype(int).reset_index()

    # Own-network = the institution's own physical points. Correspondents are
    # contracted third-party retail agents (shops, lotéricas) and are counted
    # separately: a purely digital bank can contract a wide correspondent network
    # for cash-in/cash-out without operating any premises of its own.
    by_inst["n_own_points"] = by_inst["n_agencias"] + by_inst["n_postos"]
    by_inst["n_mun_own"] = by_muni.assign(own=lambda d: d["agencias"] + d["postos"]) \
                                  .query("own > 0").groupby("cnpj8")["mun_code"] \
                                  .nunique().reindex(by_inst["cnpj8"]).fillna(0).astype(int).values
    by_inst["name"] = by_inst["cnpj8"].map(names).fillna("")
    by_inst["posicao"] = posicao
    return by_inst, by_muni


def main() -> None:
    by_inst, by_muni = build()
    by_inst.sort_values("n_own_points", ascending=False, inplace=True)
    by_inst.to_csv(BY_INST_CSV, index=False)
    by_muni.to_csv(BY_MUNI_CSV, index=False)
    logging.info(f"Saved {len(by_inst):,} institutions -> {BY_INST_CSV}")
    logging.info(f"Saved {len(by_muni):,} institution-municipality rows -> {BY_MUNI_CSV}")

    print(f"\nBCB access points (Posicao = {by_inst['posicao'].iloc[0]})")
    print(f"  institutions:                 {len(by_inst):,}")
    print(f"  with >=1 own physical point:  {(by_inst['n_own_points'] > 0).sum():,}")
    print(f"  with >=1 correspondent:       {(by_inst['n_correspondentes'] > 0).sum():,}")
    print("\nTop 15 by own physical points:")
    print(by_inst.head(15)[["cnpj8", "name", "n_agencias", "n_postos",
                            "n_mun_own", "n_correspondentes"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="force re-download instead of using the cached raw CSVs")
    REFRESH = ap.parse_args().refresh
    main()
