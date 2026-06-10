"""utils/paths.py — single source of truth for every data path in the
Egan_et_al_2025_Rep pipeline.

Author: Pedro Feijó de Moraes

Every scrape_*/panel_*/estimation_*/analysis_* script imports its data
locations from here instead of recomputing ``parents[2]`` /
``os.path.join(..., "..", "..")`` inline.  Move a data tree once, edit it once
here, and the whole pipeline follows.

Layout anchor
-------------
    OPEN_FINANCE = .../Open-Finance          shared data root (3 levels up)
    DATA_ROOT    = OPEN_FINANCE/BCB/Egan_et_al_2025_Rep
    RAW          = DATA_ROOT/raw             consolidated *raw downloads*
    PROCESSED    = DATA_ROOT/processed       pipeline outputs (deposits_panel,
                                             market_panel, PANEL_INTERMED, ...)

Raw downloads for every source live under ``RAW/<SOURCE>``.  Per-source
*processed* panel CSVs (anatel_mca_panel, cadunico_mca_panel, IBGE demographics
panels, BCB Inclusion panels, PIX ``pix_*_panel``) stay in their historical
per-source directories and are exposed below as ``<SOURCE>_DIR``.

The anchor can be overridden with the ``OPEN_FINANCE_ROOT`` environment variable
(useful on HPC / a different machine).  This is independent of, and composes
with, the per-script ``resolve_script_paths`` (TOON) override layer: scripts
pass the constants below as *defaults*, which TOON context may still override.

All values are :class:`pathlib.Path`.  ``os.path.join`` accepts Path objects, so
both ``os.path``-style and ``pathlib``-style call sites keep working.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Anchor
# ---------------------------------------------------------------------------
# utils/paths.py -> utils -> Egan_et_al_2025_Rep -> Code -> Open-Finance
_ENV_ROOT = os.environ.get("OPEN_FINANCE_ROOT", "").strip()
OPEN_FINANCE: Path = (
    Path(_ENV_ROOT).expanduser().resolve()
    if _ENV_ROOT
    else Path(__file__).resolve().parents[3]
)

BCB: Path       = OPEN_FINANCE / "BCB"
DATA_ROOT: Path = BCB / "Egan_et_al_2025_Rep"
RAW: Path       = DATA_ROOT / "raw"
PROCESSED: Path = DATA_ROOT / "processed"


def data_root() -> str:
    """Backwards-compatible accessor: the shared data root (…/Open-Finance) as a
    string, matching the historical ``disclosure_common.data_root(__file__)``."""
    return str(OPEN_FINANCE)


# ---------------------------------------------------------------------------
# Raw download trees (consolidated under RAW/)
# ---------------------------------------------------------------------------
ESTBAN_DIR: Path   = RAW / "ESTBAN"      # whole ESTBAN tree (raw monthly CSVs + ESTBAN.csv)
IF_DATA_ROOT: Path = RAW / "IF_DATA"     # whole IF-Data tree
COSIF_RAW: Path    = RAW / "COSIF_RAW"
SGS_RAW: Path      = RAW / "SGS_RAW"
ANATEL_RAW: Path   = RAW / "ANATEL"      # raw acessos_telefonia_movel_* downloads only
CADUNICO_RAW: Path = RAW / "CadUnico"    # raw cadunico_YYYYMM.csv downloads only
PIX_RAW: Path      = RAW / "PIX"         # raw Pix source files only
IBGE_RAW: Path     = RAW / "IBGE"        # raw IBGE downloads (zips + census raw CSVs)

# ESTBAN sub-paths
ESTBAN_CSV: Path     = ESTBAN_DIR / "ESTBAN.csv"
ESTBAN_RAW_MUN: Path = ESTBAN_DIR / "Relatório por município"
ESTBAN_RAW_AG: Path  = ESTBAN_DIR / "Relatório por município e agência"

# IF-Data sub-paths
IF_DATA_LIST: Path       = IF_DATA_ROOT / "List"
IF_DATA_PRUDENTIAL: Path = IF_DATA_ROOT / "Prudential Conglomerates"
IF_DATA_INDIVIDUAL: Path = IF_DATA_ROOT / "Individual Institutions"
IF_DATA_AGG: Path        = IF_DATA_ROOT / "Aggregated Data"

# ---------------------------------------------------------------------------
# Per-source directories that KEEP their processed panels in place
# (only the raw downloads above were consolidated into RAW/)
# ---------------------------------------------------------------------------
ANATEL_DIR: Path    = OPEN_FINANCE / "ANATEL"     # anatel_mca_panel.csv / anatel_muni_panel.csv
CADUNICO_DIR: Path  = OPEN_FINANCE / "CadUnico"   # cadunico_mca_panel.csv / cadunico_muni_panel.csv
IBGE_DIR: Path      = OPEN_FINANCE / "IBGE"       # *_demographics_panel.csv, muni/region panels
INCLUSION_DIR: Path = BCB / "Inclusion"           # bcb_inclusion_* / bcb_banked_* panels (pure processed)
PIX_DIR: Path       = BCB / "PIX"                 # pix_*_panel.csv processed outputs

# BCB Tarifas (not moved; raw/ was empty — real data is in processed/ + cache/)
TARIFAS_DIR: Path   = BCB / "Tarifas"
TARIFAS_PROC: Path  = TARIFAS_DIR / "processed"
TARIFAS_CACHE: Path = TARIFAS_DIR / "cache"
TARIFAS_RAW: Path   = TARIFAS_DIR / "raw"

# ---------------------------------------------------------------------------
# Stage-6 firm disclosures & other sources (unchanged location)
# ---------------------------------------------------------------------------
FIRM_DISCLOSURES: Path = OPEN_FINANCE / "FirmDisclosures"
WORLDBANK: Path        = OPEN_FINANCE / "WorldBank"
FGC: Path              = OPEN_FINANCE / "FGC"
ACCOUNTS: Path         = BCB / "Accounts"


__all__ = [
    "OPEN_FINANCE", "BCB", "DATA_ROOT", "RAW", "PROCESSED", "data_root",
    "ESTBAN_DIR", "ESTBAN_CSV", "ESTBAN_RAW_MUN", "ESTBAN_RAW_AG",
    "IF_DATA_ROOT", "IF_DATA_LIST", "IF_DATA_PRUDENTIAL", "IF_DATA_INDIVIDUAL", "IF_DATA_AGG",
    "COSIF_RAW", "SGS_RAW",
    "ANATEL_RAW", "CADUNICO_RAW", "PIX_RAW", "IBGE_RAW",
    "ANATEL_DIR", "CADUNICO_DIR", "IBGE_DIR", "INCLUSION_DIR", "PIX_DIR",
    "TARIFAS_DIR", "TARIFAS_PROC", "TARIFAS_CACHE", "TARIFAS_RAW",
    "FIRM_DISCLOSURES", "WORLDBANK", "FGC", "ACCOUNTS",
]
