# Deposit Competition Code

**Author:** Pedro Feijó de Moraes  
**Affiliation:** Yale University  
**Project:** Brazilian Open Finance — Deposit Competition & Demand Estimation

---

## Overview

This repository contains the full data pipeline and structural econometric estimation code for a research project studying deposit competition in Brazil's banking sector. The project replicates and extends the framework of **Egan, Hortaçsu & Matvos (2025)** to Brazilian prudential conglomerates, using granular municipality-level deposit data from the Central Bank of Brazil (BCB).

The codebase:

1. **Downloads and processes** raw data from multiple Brazilian government APIs (BCB, IBGE, ANATEL, SAGI/CadUnico, INSS).
2. **Builds panel datasets** at the conglomerate x municipality x quarter level.
3. **Estimates structural models** of deposit supply and demand.

---

## Key Data Sources

| Source | Description |
| ------ | ----------- |
| **BCB / ESTBAN** | Monthly bank balance sheets by municipality (COSIF format) |
| **BCB / IF Data (Olinda API)** | Prudential conglomerate reports (deposit stocks, assets, solvency) |
| **BCB / SGS** | Macro time series: Selic overnight rate, CDI, TR |
| **BCB / PIX** | PIX instant-payment adoption by municipality |
| **BCB / Financial Inclusion** | Branch and banking-correspondent counts by municipality |
| **IBGE / SIDRA** | Municipal population, GDP per capita, age structure |
| **ANATEL** | Mobile broadband (4G/5G) connections by municipality |
| **SAGI / CadUnico** | Low-income household registry (poverty indicator) |
| **INSS** | Retirement/pension beneficiary counts by municipality |

---

## Repository Structure

```text
.
├── run_data_pipeline.py          # Master pipeline runner (orchestrates all stages)
│
├── ── Stage 0: Raw Data Downloads ──
├── if_data_scrape_1.py           # Downloads ESTBAN monthly CSVs + IF Data via BCB Olinda API
│
├── ── Stage 1: Demographics ──
├── ibge_demographics_panel.py    # IBGE population, GDP, age structure → MCA-level panel
│
├── ── Stage 2: Market Characteristic Panels ──
├── scrape_pix_panel.py           # BCB PIX adoption → MCA panel
├── scrape_anatel.py              # ANATEL mobile connections → MCA panel
├── scrape_bcb_inclusion.py       # BCB banking access-points → MCA panel
├── scrape_cadunico.py            # CadUnico low-income families → MCA panel
├── scrape_inss.py                # INSS retirees → MCA panel
├── tarifas_scrape_1.py           # BCB bank fee schedules → conglomerate panel
│
├── ── Stage 3: Deposit Panel ──
├── deposits_panel_build.py       # ESTBAN + IF Data → conglomerate × municipality × quarter
├── append_rates.py               # Appends deposit rates/spreads (COSIF + SGS)
├── bank_chars_panel_build.py     # IF Data → conglomerate bank size & solvency panel
│
├── ── Stage 4: Master Analysis Panel ──
├── build_market_panel.py         # Merges all panels → single analysis-ready dataset
│
├── ── Egan et al. (2025) Replication Panel ──
├── egan_panel_build.py           # Builds CodConglomerate × Quarter × DepositType panel
├── egan_panel_build_2.py         # Extended version (municipal-level Egan panel)
│
├── ── Estimation ──
├── estimation_1_OLD.py           # Supply-side (sleepiness Υ) estimation — conglomerate level
├── estimation_2_OLD.py           # Supply-side estimation — bank level
├── estimation_demand_1_OLD.py    # Demand-side (Berry 1994) estimation — conglomerate level
├── estimation_demand_2_OLD.py    # Demand-side estimation — bank level
├── estimation_1_sleep.py         # Variant estimation script
│
├── ── Output & Utilities ──
├── table_builder.py              # Builds LaTeX regression tables from estimation results
├── generate_mca_json.py          # Generates MCA geographic JSON files
├── rectangularize_mca_json.py    # Converts MCA JSON to tabular format
├── export_comments.py            # Exports inline code comments/notes
│
├── ── COSIF Processing (legacy) ──
├── cosif_process_1_OLD.py        # COSIF raw data cleaning
├── cosif_process_2_OLD.py        # COSIF aggregation
│
├── ── Data Collection (legacy) ──
├── data_collection_1_OLD.py      # SGS macro series download
├── data_collection_2_OLD.py      # Additional macro series
│
├── ── Utilities ──
├── utils/
│   ├── venv_guard.py             # Ensures correct virtual environment is active
│   ├── toon_parser.py            # Parses Gemini AI-generated context/configurations
│   ├── toon_parser_cli.py        # CLI interface for toon_parser
│   ├── toon_runtime.py           # Loads TOON runtime context for path resolution
│   └── toon_context.json         # Cached runtime configuration (auto-generated)
│
├── requirements.txt              # Python dependencies
├── requirements_venv_full.txt    # Full venv dependency list
└── requirements_toon.txt         # Web-scraping dependencies (beautifulsoup4, lxml)
```

---

## Pipeline Architecture

The pipeline is executed via `run_data_pipeline.py`, which runs scripts as subprocesses in **dependency-ordered waves**, with independent steps parallelized within each wave:

```text
Wave 1  (serial)    : 0a — Download ESTBAN + IF Data raw files
Wave 2  (parallel)  : 1a, 2a, 2b, 2d, 2e, 2f, 3a — Demographics, market panels, deposit panel
Wave 3  (parallel)  : 3b, 3c — Deposit rates/spreads + bank characteristics
Wave 4  (serial)    : 4a — Master merge into final analysis dataset
```

### Running the Pipeline

```bash
# Run all stages
python run_data_pipeline.py

# Start from a specific stage
python run_data_pipeline.py --from 2

# Run only one stage
python run_data_pipeline.py --only 3

# Skip specific sub-steps
python run_data_pipeline.py --skip 2b,2e

# List all pipeline steps
python run_data_pipeline.py --list
```

---

## Econometric Model

The project estimates a structural model of deposit supply and demand following **Egan, Hortaçsu & Matvos (2025)**:

### Supply Side (Deposit "Sleepiness")

The structural equation in levels:

```text
Dep_jkt = φ(S_t, X_jt) · nr_t · Dep_jkt−1 + ε_jkt

where φ(S_t, X_jt) = Υ₁'S_t + Υ₂'X_jt
      nr_t = 1 + (R^F_{t−1} − ρ_{jkt−1}) / 100
```

- `Dep_jkt`: deposit balance of bank *j*, type *k*, quarter *t*
- `S_t`: market-level characteristics (PIX adoption, mobile coverage, poverty, demographics)
- `X_jt`: bank-level characteristics (assets, solvency ratio)
- `Υ`: "sleepiness" parameters to be estimated

### Identification

- Deposit types 1-3 (demand, savings, interbank): exogenous rates -> OLS
- Deposit types 4-5 (CDB, prepaid): endogenous rates -> **Control Function** approach (Petrin & Train 2010)
  - **Cost-shifter instruments**: lagged COSIF implicit rate, log assets, equity ratio, lagged CDI/Selic
  - **Hausman IV**: leave-one-out mean deposit spread (same type x quarter)

### Demand Side

Active market shares are constructed after removing the "sleeping" component, then demand is estimated via **Berry (1994)**:

```text
log(s_active_jkt) = α_k · σ_jkt + δ_j + μ_kt + e_jkt
```

where `σ_jkt` is the deposit spread (opportunity cost) and `δ_j` is a bank×type fixed effect.

---

## Geographic Unit

The primary geographic unit is the **MCA (Minimum Comparable Area)** — a time-consistent municipal grouping used to handle Brazilian municipal boundary changes from 2010–2024. Municipality codes follow the 7-digit IBGE standard (`CODMUN_IBGE`).

---

## Key Technologies

| Technology | Purpose |
| ---------- | ------- |
| **Python 3.x** | All data processing and estimation |
| **pandas** | Data manipulation and panel construction |
| **numpy** | Numerical computations |
| **requests / beautifulsoup4 / lxml** | Web scraping and API calls |
| **statsmodels** | OLS and panel regression |
| **linearmodels** | Panel OLS with fixed effects (two-way FE, clustered SE) |
| **scipy** | Statistical utilities |
| **pyarrow** | Efficient Parquet I/O |
| **concurrent.futures** | Parallel pipeline execution |

---

## Output Files

The pipeline produces CSV files organized by data source under a `BCB/` directory tree:

| File | Description |
| ---- | ----------- |
| `BCB/Panel/deposits_panel.csv` | Conglomerate × municipality × quarter deposit balances |
| `BCB/Panel/market_panel.csv` | Master analysis dataset (deposits + all market characteristics) |
| `BCB/Egan_et_al_2025_Rep/processed/PANEL_INTERMED/egan_panel_deposits.csv` | Egan-style panel for estimation |
| `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/` | Estimation results, LaTeX tables |
| `IBGE/mca_demographics_panel.csv` | MCA demographics panel |
| `ANATEL/anatel_mca_panel.csv` | Mobile connectivity panel |
| `BCB/PIX/pix_mca_panel.csv` | PIX adoption panel |

---

## References

- Egan, M., Hortaçsu, A., & Matvos, G. (2025). *Deposit Competition and Financial Fragility: Evidence from the U.S. Banking Sector.*
- Berry, S. T. (1994). *Estimating Discrete-Choice Models of Product Differentiation.* RAND Journal of Economics, 25(2), 242–262.
- Petrin, A., & Train, K. (2010). *A Control Function Approach to Endogeneity in Consumer Choice Models.* Journal of Marketing Research, 47(1), 3–13.
- Conlon, C., & Gortmaker, J. (2020). *Best Practices for Differentiated Products Demand Estimation with PyBLP.* RAND Journal of Economics, 51(4), 1108–1161.
