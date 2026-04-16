# Deposit Competition Code

**Author:** Pedro Feijó de Moraes  
**Affiliation:** Yale University  
**Project:** Brazilian Open Finance — Deposit Competition & Demand Estimation

---

## Overview

This repository contains the full data pipeline and structural econometric estimation code for a research project studying deposit competition in Brazil's banking sector. The project replicates and extends the framework of **Egan, Hortaçsu & Matvos (2025)** to Brazilian prudential conglomerates, using granular municipality-level deposit data from the Central Bank of Brazil (BCB).

The codebase:
1. **Downloads and processes** raw data from multiple Brazilian government APIs (BCB, IBGE, ANATEL, SAGI/CadUnico, INSS).
2. **Builds panel datasets** at the conglomerate × municipality × quarter level.
3. **Estimates structural models** of deposit supply (sleepiness) and demand (BLP/Berry 1994).

---

## Key Data Sources

| Source | Description |
|--------|-------------|
| **BCB / ESTBAN** | Monthly bank balance sheets by municipality (COSIF format) |
| **BCB / IF Data (Olinda API)** | Prudential conglomerate reports (deposit stocks, assets, solvency) |
| **BCB / SGS** | Macro time series: Selic overnight rate, CDI, TR |
| **BCB / PIX** | PIX instant-payment adoption by municipality |
| **BCB / Financial Inclusion** | Branch and banking-correspondent counts by municipality |
| **IBGE / SIDRA** | Municipal population, GDP per capita, age structure |
| **ANATEL** | Mobile broadband (4G/5G) connections by municipality |
| **SAGI / CadUnico** | Low-income household registry (poverty indicator) |
| **INSS** | Retirement/pension beneficiary counts by municipality |
| **Internet Archive** | Historical deposit-rate disclosures (Wayback Machine CDX API) |
| **World Bank / Findex** | Banked-population fraction (FX.OWN.TOTL.ZS) |

---

## Repository Structure

```
.
├── run_data_pipeline.py              # Master data pipeline runner (stages 0–4)
├── run_sleep_pipeline.py             # Sleepiness estimation pipeline runner (est. 1–5 + exports)
├── run_blp_pipeline.py               # BLP pipeline runner (Julia estimation + LaTeX tables)
│
├── ── Stage 0: Raw Data Downloads ──
├── scrape_1_bcb_estban_if_data.py    # ESTBAN monthly CSVs + IF Data via BCB Olinda API
│
├── ── Stage 1: Demographics ──
├── scrape_2_ibge_demographics.py     # IBGE population, GDP, age structure → MCA-level panel
│
├── ── Stage 2: Market Characteristic Panels ──
├── scrape_3_pix_panel.py             # BCB PIX adoption → MCA panel
├── scrape_4_anatel.py                # ANATEL mobile connections → MCA connectivity panel
├── scrape_5_bcb_inclusion.py         # BCB banking access-points (branches + correspondents) → MCA panel
├── scrape_6_cadunico.py              # CadUnico low-income families → MCA poverty panel
├── scrape_7_fees.py                  # BCB bank fee schedules (PF + PJ) → tarifas conglomerate panel
├── scrape_8_bcb_banked.py            # ESTBAN Dec snapshots + WB Findex → MCA banked-fraction proxy panel
├── scrape_inss.py                    # INSS retirees → MCA quarter panel
│
├── ── Stage 3: Deposit Panel, Rates & Characteristics ──
├── panel_1_deposits.py               # ESTBAN + IF Data → conglomerate × municipality × quarter deposit panel
├── panel_2_rates_ip.py               # Extract IP explicit deposit rates from raw COSIF files
├── panel_3_rates.py                  # Compute and append deposit rates/spreads (COSIF + SGS)
├── panel_4_bank_chars.py             # IF Data → conglomerate bank size & solvency characteristics panel
├── panel_5_flag_digital.py           # Identify purely digital banks from ESTBAN → PANEL_INTERMED
│
├── ── Stage 4: Master Analysis Panel ──
├── panel_6_market.py                 # Merge all MCA panels + deposit panel → master analysis dataset
├── panel_7_instruments.py            # Compute LOO instruments and FGC coverage dummy → overwrites market_panel.csv
│
├── ── Deposit Rate Scraping (Internet Archive) ──
├── d_rate_scrape_1_targets.py        # Initialise target domain list for archival rate scraping
├── d_rate_scrape_2_cdx.py            # Query Wayback Machine CDX API for candidate URLs
├── d_rate_scrape_3_fetch.py          # Async-fetch HTML/PDF snapshots from Internet Archive
├── d_rate_scrape_4_parse.py          # NLP extraction of deposit yields from HTML and PDF files
│
├── ── Sleepiness Estimation ──
├── estimation_1_sleep.py             # B-type CFA estimation (Υ); phi_mt and phi_t construction
├── estimation_2_sleep.py             # Robustness: omit post-2020 structural break dummy
├── estimation_3_sleep.py             # Robustness: pooled B + D firms with D-type dummy
├── estimation_4_sleep.py             # Robustness: NLLS logistic link function
├── estimation_5_sleep.py             # Robustness: cooperative / state-owned institution controls
│
├── ── Demand Estimation (BLP) ──
├── estimation_demand_1_prep.py       # Universal demand prep orchestrator (runs est. 1–5 in parallel)
├── estimation_1_demand_1_prep.py     # Demand prep for estimation round 1 (active shares, market sizes)
├── estimation_2_demand_1_prep.py     # Demand prep for estimation round 2
├── estimation_3_demand_1_prep.py     # Demand prep for estimation round 3
├── estimation_4_demand_1_prep.py     # Demand prep for estimation round 4
├── estimation_5_demand_1_prep.py     # Demand prep for estimation round 5
├── estimation_1_demand_2_loop.py     # Python wrapper: launches Julia BLP loop for round 1
├── estimation_2_demand_2_loop.py     # Python wrapper: launches Julia BLP loop for round 2
├── estimation_3_demand_2_loop.py     # Python wrapper: launches Julia BLP loop for round 3
├── estimation_4_demand_2_loop.py     # Python wrapper: launches Julia BLP loop for round 4
├── estimation_5_demand_2_loop.py     # Python wrapper: launches Julia BLP loop for round 5
├── estimation_1_demand_2_loop_ju.jl  # Julia BLP GMM loop (round 1)
├── estimation_2_demand_2_loop_ju.jl  # Julia BLP GMM loop (round 2)
├── estimation_3_demand_2_loop_ju.jl  # Julia BLP GMM loop (round 3)
├── estimation_4_demand_2_loop_ju.jl  # Julia BLP GMM loop (round 4)
├── estimation_5_demand_2_loop_ju.jl  # Julia BLP GMM loop (round 5)
│
├── ── Cost Estimation (BBL) ──
├── estimation_1_cost_1_polfunc.py    # BBL Step 1: parametric policy functions for endogenous rates (k=4,5)
├── estimation_1_cost_2_fwd.py        # BBL Step 2: forward simulation for cost recovery
│
├── ── Export & Results ──
├── export_results.py                 # Orchestrator: dispatches to export_*_sleep_results.py in parallel
├── export_1_sleep_results.py         # Export sleep results for estimation round 1 (tables, plots)
├── export_2_sleep_results.py         # Export sleep results for estimation round 2
├── export_3_sleep_results.py         # Export sleep results for estimation round 3
├── export_4_sleep_results.py         # Export sleep results for estimation round 4
├── export_5_sleep_results.py         # Export sleep results for estimation round 5
├── export_analyze_spec12.py          # Comparative analysis across specification 1 & 2 variants
│
├── ── Descriptives & Tables ──
├── desc_1.py                         # Summary statistics tables (CSV + LaTeX) by bank type and region
├── make_blp_latex_tables.py          # Compile BLP demand estimation results into LaTeX tables
├── table_builder.py                  # Build LaTeX regression tables from sleepiness pickle output
│
├── ── HPC Submission Scripts ──
├── submit_blp_hpc_1.sh               # SLURM submission script for BLP round 1
├── submit_blp_hpc_2.sh               # SLURM submission script for BLP round 2
├── submit_blp_hpc_3.sh               # SLURM submission script for BLP round 3
├── submit_blp_hpc_4.sh               # SLURM submission script for BLP round 4
├── submit_blp_hpc_5.sh               # SLURM submission script for BLP round 5
├── submit_blp_hpc_test_1.sh          # SLURM test (dry-run) submission for round 1
├── submit_blp_hpc_test_2.sh          # SLURM test submission for round 2
├── submit_blp_hpc_test_3.sh          # SLURM test submission for round 3
├── submit_blp_hpc_test_4.sh          # SLURM test submission for round 4
├── submit_blp_hpc_test_5.sh          # SLURM test submission for round 5
│
├── ── Utilities ──
├── utils/
│   ├── venv_guard.py                 # Ensures correct virtual environment is active
│   ├── toon_parser.py                # Parses Gemini AI-generated context/configurations
│   ├── toon_parser_cli.py            # CLI interface for toon_parser
│   ├── toon_runtime.py               # Loads TOON runtime context for path resolution
│   └── tex_preamble.py               # Shared LaTeX preamble template for export scripts
│
├── requirements_venv_full.txt        # Full Python dependency list
└── requirements_toon.txt             # Minimal scraping dependencies (beautifulsoup4, lxml)
```

---

## Pipeline Architecture

### Data Pipeline (`run_data_pipeline.py`)

Runs all download, processing, and panel-building scripts as subprocesses in dependency order. Independent steps within the same stage can be skipped individually.

```
Stage 0  (serial)    : scrape_1  — Download ESTBAN + IF Data raw files
Stage 1  (serial)    : scrape_2  — IBGE demographics → MCA panel
Stage 2  (parallel)  : scrape_3..8, scrape_inss — Market characteristic panels
Stage 3  (parallel)  : panel_1..5 — Deposit panel, rates, bank chars, digital flag
Stage 4  (serial)    : panel_6   — Master merge → market_panel.csv
                       panel_7   — Append LOO instruments + FGC dummy
```

```bash
python run_data_pipeline.py                    # Run all stages
python run_data_pipeline.py --from 3           # Resume from stage 3
python run_data_pipeline.py --skip 5b,6        # Skip specific steps
python run_data_pipeline.py --list             # Print all steps and exit
```

### Sleep Pipeline (`run_sleep_pipeline.py`)

Runs the full sleepiness estimation sequence:

```
Step 1 : estimation_1_sleep.py         B-type CFA + phi construction
Step 2 : estimation_2_sleep.py         Robustness: no break dummy
Step 3 : estimation_3_sleep.py         Robustness: pooled B+D firms
Step 4 : estimation_4_sleep.py         Robustness: NLLS logistic
Step 5 : estimation_5_sleep.py         Robustness: coop/state controls
Step 6 : export_results.py             Export 1st/2nd stage summaries
Step 7 : estimation_demand_1_prep.py   Demand prep for all 5 rounds
```

```bash
python run_sleep_pipeline.py
python run_sleep_pipeline.py --only-spec-12    # Run only specification 1 & 2
```

### BLP Pipeline (`run_blp_pipeline.py`)

Runs Julia BLP demand estimation and/or compiles LaTeX output tables:

```bash
python run_blp_pipeline.py --latex                                    # Build LaTeX tables only
python run_blp_pipeline.py --julia --est 12345 --spec 12 --R 2000 \
    --chunk-size 200 --workers 4 --stage sequence                     # Full estimation run
python run_blp_pipeline.py --julia --est 1 --spec 1 --dry-run        # Sanity-check dry run
```

---

## Econometric Model

The project estimates a structural model of deposit supply and demand following **Egan, Hortaçsu & Matvos (2025)**:

### Supply Side (Deposit "Sleepiness")
The structural equation in levels:

```
Dep_jkt = φ(S_t, X_jt) · nr_t · Dep_jkt−1 + ε_jkt

where φ(S_t, X_jt) = Υ₁'S_t + Υ₂'X_jt
      nr_t = 1 + (R^F_{t−1} − ρ_{jkt−1}) / 100
```

- `Dep_jkt`: deposit balance of bank *j*, type *k*, quarter *t*
- `S_t`: market-level characteristics (PIX adoption, mobile coverage, poverty, demographics)
- `X_jt`: bank-level characteristics (assets, solvency ratio)
- `Υ`: "sleepiness" parameters to be estimated

### Identification
- Deposit types 1–3 (demand, savings, interbank): exogenous rates → OLS
- Deposit types 4–5 (CDB, prepaid): endogenous rates → **Control Function** approach (Petrin & Train 2010)
  - **Cost-shifter instruments**: lagged COSIF implicit rate, log assets, equity ratio, lagged CDI/Selic
  - **Hausman IV**: leave-one-out mean deposit spread (same type × quarter)

### Demand Side
Active market shares are constructed after removing the "sleeping" component, then demand is estimated via **Berry (1994)**:

```
log(s_active_jkt) = α_k · σ_jkt + δ_j + μ_kt + e_jkt
```

where `σ_jkt` is the deposit spread (opportunity cost) and `δ_j` is a bank×type fixed effect.

### Cost Estimation (BBL)
Following Bajari, Benkard & Levin (2007), parametric policy functions for endogenous deposit types (k=4,5) are estimated in `estimation_1_cost_1_polfunc.py`, then used to recover marginal costs via forward simulation in `estimation_1_cost_2_fwd.py`.

---

## Geographic Unit

The primary geographic unit is the **MCA (Minimum Comparable Area)** — a time-consistent municipal grouping used to handle Brazilian municipal boundary changes from 2010–2024. Municipality codes follow the 7-digit IBGE standard (`CODMUN_IBGE`).

---

## Key Technologies

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | All data processing and estimation |
| **Julia** | BLP GMM demand estimation loops (`estimation_*_demand_2_loop_ju.jl`) |
| **pandas** | Data manipulation and panel construction |
| **numpy** | Numerical computations |
| **scipy** | Statistical utilities, NLLS optimisation, Halton draws |
| **statsmodels** | OLS and panel regression |
| **linearmodels** | Panel OLS with fixed effects (two-way FE, clustered SE) |
| **scikit-learn** | PCA for instrument construction |
| **requests / beautifulsoup4 / lxml** | Web scraping and API calls |
| **aiohttp** | Async HTTP for high-throughput Internet Archive fetching |
| **pdfplumber** | PDF text extraction for deposit rate parsing |
| **geopandas / geobr** | Brazilian geographic data and MCA boundary processing |
| **matplotlib** | Diagnostic and results plots |
| **pyarrow** | Efficient Parquet I/O |
| **concurrent.futures** | Parallel pipeline execution |

---

## Output Files

The pipeline produces CSV and pickle files organized under a `BCB/` directory tree:

| File | Description |
|------|-------------|
| `BCB/Panel/deposits_panel.csv` | Conglomerate × municipality × quarter deposit balances |
| `BCB/Panel/market_panel.csv` | Master analysis dataset (deposits + all market characteristics + instruments) |
| `BCB/Egan_et_al_2025_Rep/processed/COSIF_PROCESSED/ip_rates_quarterly.csv` | IP explicit deposit rates (quarterly) |
| `BCB/Egan_et_al_2025_Rep/processed/PANEL_INTERMED/digital_banks_diagnostic.csv` | Digital bank classification diagnostic |
| `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/SLEEPINESS/` | Sleepiness estimation results (PKL, JSON, TeX tables) |
| `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/DEMAND_PREP/rout_*/` | Active-shares panels ready for BLP (per estimation round) |
| `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/BLP_RESULTS/` | BLP demand estimation outputs |
| `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/COST_FWD/` | BBL cost recovery forward-simulation outputs |
| `BCB/Egan_et_al_2025_Rep/processed/IP_SCRAPE/` | Archival deposit-rate HTML/PDF snapshots and extracted rates |
| `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/DESCRIPTIVES/` | Descriptive statistics (CSV + LaTeX) |
| `IBGE/mca_demographics_panel.csv` | MCA demographics panel |
| `ANATEL/anatel_mca_panel.csv` | Mobile connectivity panel |
| `BCB/PIX/pix_mca_panel.csv` | PIX adoption panel |
| `BCB/Inclusion/bcb_inclusion_mca_panel.csv` | Banking access-point density panel |
| `BCB/Banked/banked_fraction_mca_panel.csv` | Banked-population fraction proxy panel |

---

## References

- Egan, M., Hortaçsu, A., & Matvos, G. (2025). *Deposit Competition and Financial Fragility: Evidence from the U.S. Banking Sector.*
- Berry, S. T. (1994). *Estimating Discrete-Choice Models of Product Differentiation.* RAND Journal of Economics, 25(2), 242–262.
- Petrin, A., & Train, K. (2010). *A Control Function Approach to Endogeneity in Consumer Choice Models.* Journal of Marketing Research, 47(1), 3–13.
- Bajari, P., Benkard, C. L., & Levin, J. (2007). *Estimating Dynamic Models of Imperfect Competition.* Econometrica, 75(5), 1331–1370.
