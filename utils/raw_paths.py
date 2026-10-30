"""utils/raw_paths.py: where each research-raw dataset lives under Open-Finance/DATA_RAW.

DATA_RAW is organised by SOURCE (publisher) and then dataset: DATA_RAW/<source>/<dataset>/.
The index of every dataset, and the request that asked for it, is DATA_RAW/README.md; one row
per dataset/document is in DATA_RAW/source_log.csv. These are raw files collected for side
analyses, separate from the pipeline's own BCB/, IBGE/, WorldBank/ folders under Open-Finance
(see utils/paths.py), which this module does not touch.

Scripts that read or write DATA_RAW take their paths from here, so a future move changes one file.
"""
from __future__ import annotations

from utils import paths

DATA_RAW = paths.OPEN_FINANCE / "DATA_RAW"

# BCB
BCB_CCS_RELATIONSHIPS = DATA_RAW / "BCB" / "ccs_relationships" / "current"   # scrape_bcb_ccs_relationships.py
BCB_ESTBAN_AGENCIA = DATA_RAW / "BCB" / "estban_agencia"                       # scrape_estban_branches.py
BCB_PIX_MUNICIPAL_RAW = DATA_RAW / "BCB" / "pix_municipal" / "raw"             # pix_mun_YYYYMM.csv
BCB_PIX_PARTICIPANTS = DATA_RAW / "BCB" / "pix_participants"                   # Pix by participant (ISPB)
BCB_PIX_KEYS = BCB_PIX_PARTICIPANTS / "chaves"                                 # scrape_pix_keys.py
BCB_SELIC_SGS = DATA_RAW / "BCB" / "selic_sgs"                                 # scrape_national_series.py (SGS)
BCB_TARIFAS_MONTHLY = DATA_RAW / "BCB" / "tarifas_monthly"                     # fees/scrape_bcb_tarifas_monthly.py
BCB_TARIFAS_LAI = DATA_RAW / "BCB" / "tarifas_lai_18810_025439" / "raw"        # LAI delivery 2026-10-27 (same files)
BCB_TARIFAS_WAYBACK = DATA_RAW / "BCB" / "tarifas_wayback"                     # old tariff pages: CDX list, sample

# Ipeadata
IPEADATA_IBOVESPA = DATA_RAW / "Ipeadata" / "ibovespa"                         # scrape_national_series.py (Ipeadata)

# Tesouro Nacional
TESOURO_INVESTIDORES_LIVE = DATA_RAW / "TesouroNacional" / "investidores_td" / "live_manual"

# consumidor.gov.br (Ministério da Justiça / Senacon)
CONSUMIDOR_GOV = DATA_RAW / "consumidor_gov"                                   # scrape_consumidor_gov.py

# IBGE
IBGE_LOCALIDADES = DATA_RAW / "IBGE" / "localidades"                           # municipality and district lists
IBGE_POF_2017_18 = DATA_RAW / "IBGE" / "pof" / "pof2017_18"                    # POF microdata + tables; fees/build_pof_bank_fees.py
