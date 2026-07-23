"""
scrape_4b_mca_region_hierarchy.py
=================================
Populate the region-hierarchy columns of the municipality<->MCA crosswalk
(``IBGE/muni_mca_regions_2010_2024_panel.csv``) from the IBGE Divisão Territorial
Brasileira (DTB), replacing the ``"N/A"`` placeholders that
``scrape_4_ibge_demographics.py::build_mca_crosswalk_direct`` writes for
``micro_*/meso_*/immgr_*/intgr_*``.

WHY.  The crosswalk carries slots for four nested geographies but they were left
as ``"N/A"`` (the geobr sjoin path is ~7.6h and was skipped). Downstream nothing
read them, so a region-level "ring" of neighbouring MCAs was impossible. This
script fills them cheaply from the authoritative DTB tables:

  * micro-region + meso-region  ← DTB **2016** (`DTB_BRASIL_MUNICIPIO.xls`); the
    old hierarchy, dropped from the 2024 DTB. Full codes are built IBGE-style:
    meso = UF(2)+meso(2) (4 digits), micro = UF(2)+micro(3) (5 digits).
  * immediate + intermediate region ← DTB **2024** (`RELATORIO_DTB_BRASIL_2024_
    MUNICIPIOS.xls`); the current (2017+) hierarchy. Codes are already full
    (imediata 6-digit, intermediária 4-digit).

Both DTB zips live in ``RAW/IBGE`` (downloaded from
geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/divisao_territorial/).
Keyed on the 7-digit ``Código Município Completo`` == crosswalk ``municipality_code``.

The crosswalk is backed up (``*.bak_regions``) before it is overwritten. Idempotent.
NB: the decisive use-case (a spatial IV for the deposit spread) is dead because the
spread is national (constant across MCAs); this data is for spatial descriptives /
less-sparse rival sets / any future ring, not for identifying the price coefficient.

Usage:  python scrape_4b_mca_region_hierarchy.py
"""
from __future__ import annotations
import io
import logging
import shutil
import zipfile

import pandas as pd

from utils import paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

XWALK = paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"
DTB_2016_ZIP = paths.IBGE_RAW / "DTB_2016_v2.zip"
DTB_2024_ZIP = paths.IBGE_RAW / "DTB_2024.zip"
DTB_2016_MEMBER = "DTB_2016_v2/DTB_2016/DTB_BRASIL_MUNICIPIO.xls"
DTB_2024_MEMBER = "RELATORIO_DTB_BRASIL_2024_MUNICIPIOS.xls"


def _read_dtb(zip_path, member, header_row):
    raw = zipfile.ZipFile(zip_path).read(member)
    df = pd.read_excel(io.BytesIO(raw), engine="xlrd", header=header_row, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _col(df, *needles):
    """First column whose name contains ALL needles (case-insensitive)."""
    for c in df.columns:
        cl = c.lower()
        if all(n.lower() in cl for n in needles):
            return c
    raise KeyError(f"no column matching {needles} in {list(df.columns)}")


def build_region_map() -> pd.DataFrame:
    # ── micro / meso from the 2016 DTB (header on row 0) ──────────────────────
    d16 = _read_dtb(DTB_2016_ZIP, DTB_2016_MEMBER, 0)
    uf   = _col(d16, "UF")                      # 2-digit state
    muni = _col(d16, "Código Município")        # 7-digit IBGE code
    meso, meso_nm  = _col(d16, "Mesorregião"), _col(d16, "Nome_Meso")
    micro, micro_nm = _col(d16, "Microrregião"), _col(d16, "Nome_Micro")
    m = pd.DataFrame({"municipality_code": pd.to_numeric(d16[muni], errors="coerce").astype("Int64")})
    ufd = d16[uf].str.strip().str.zfill(2)
    m["meso_code"]  = ufd + d16[meso].str.strip().str.zfill(2)          # UF+meso  → 4-digit
    m["meso_name"]  = d16[meso_nm].str.strip()
    m["micro_code"] = ufd + d16[micro].str.strip().str.zfill(3)         # UF+micro → 5-digit
    m["micro_name"] = d16[micro_nm].str.strip()

    # ── immediate / intermediate from the 2024 DTB (header on row 6) ──────────
    d24 = _read_dtb(DTB_2024_ZIP, DTB_2024_MEMBER, 6)
    muni24 = _col(d24, "Código Município")
    intgr, intgr_nm = _col(d24, "Intermediária"), _col(d24, "Nome", "Intermediária")
    immgr, immgr_nm = _col(d24, "Imediata"),      _col(d24, "Nome", "Imediata")
    n = pd.DataFrame({"municipality_code": pd.to_numeric(d24[muni24], errors="coerce").astype("Int64")})
    n["immgr_code"] = d24[immgr].str.strip()
    n["immgr_name"] = d24[immgr_nm].str.strip()
    n["intgr_code"] = d24[intgr].str.strip()
    n["intgr_name"] = d24[intgr_nm].str.strip()

    out = m.merge(n, on="municipality_code", how="outer").dropna(subset=["municipality_code"])
    logging.info(f"region map: {len(out):,} municipalities "
                 f"(micro/meso from 2016: {m['municipality_code'].nunique():,}; "
                 f"immgr/intgr from 2024: {n['municipality_code'].nunique():,})")
    return out


def main() -> None:
    if not XWALK.exists():
        raise FileNotFoundError(f"crosswalk not found: {XWALK}")
    reg = build_region_map()

    xw = pd.read_csv(XWALK, dtype=str)
    xw["_muni"] = pd.to_numeric(xw["municipality_code"], errors="coerce").astype("Int64")
    reg_i = reg.set_index("municipality_code")
    region_cols = ["micro_code", "micro_name", "meso_code", "meso_name",
                   "immgr_code", "immgr_name", "intgr_code", "intgr_name"]

    backup = XWALK.with_suffix(".csv.bak_regions")
    if not backup.exists():
        shutil.copy2(XWALK, backup)
        logging.info(f"backed up crosswalk -> {backup.name}")

    filled = {}
    for c in region_cols:
        mapped = xw["_muni"].map(reg_i[c])
        xw[c] = mapped.where(mapped.notna(), "N/A")     # keep N/A only where truly unmatched
        filled[c] = int((xw[c] != "N/A").sum())
    xw = xw.drop(columns=["_muni"])
    xw.to_csv(XWALK, index=False)

    n = len(xw)
    logging.info(f"populated {XWALK.name} ({n:,} rows). Non-N/A coverage:")
    for c in region_cols:
        logging.info(f"    {c:12s}: {filled[c]:,}/{n:,} ({100*filled[c]/n:.1f}%)")
    still_na = [c for c in region_cols if filled[c] == 0]
    if still_na:
        logging.warning(f"columns still all-N/A: {still_na}")


if __name__ == "__main__":
    main()
