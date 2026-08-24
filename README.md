 Deposit Competition Code

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
4. **Recovers marginal costs** via Bajari-Benkard-Levin (BBL) forward simulation.

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
| **Internet Archive** | Historical deposit-rate disclosures (Wayback Machine CDX API) |
| **World Bank / Findex** | Banked-population fraction (FX.OWN.TOTL.ZS) |

---

## Repository Structure

```text
.
├── run_data_pipeline.py                  # Master data pipeline runner (stages 0–5)
├── run_sleep_pipeline.py                 # Sleepiness estimation pipeline runner (E1–E4 + exports, demand prep, tables)
│                                         # (run_blp_pipeline.py / run_local_pipeline.py were
│                                         #  removed in a30be201 — call the Julia directly)
│
├── ── Stage 0: Raw Data Downloads ──
├── scrape_1_bcb_estban_if_data.py        # ESTBAN monthly CSVs + IF Data via BCB Olinda API
│
├── ── Stage 1: Demographics ──
├── scrape_4_ibge_demographics.py         # IBGE population, GDP, age structure → MCA-level panel
│
├── ── Stage 2: Market Characteristic Panels (parallel) ──
├── scrape_5_pix_panel.py                 # BCB PIX adoption → MCA panel
├── scrape_6_anatel.py                    # ANATEL mobile connections → MCA connectivity panel
├── scrape_7_bcb_inclusion.py             # BCB banking access-points (branches + correspondents) → MCA panel
├── scrape_8_bcb_banked.py                 # ESTBAN Dec snapshots + WB Findex → MCA banked-fraction proxy panel
├── scrape_9_cadunico.py                   # CadUnico low-income families → MCA poverty panel
├── scrape_10_fees.py                      # BCB bank fee schedules (PF + PJ) → tarifas conglomerate panel
├── scrape_inss.py                        # INSS retirees → MCA quarter panel
│
├── ── Stage 2c: COSIF Download + Processing ──
├── scrape_21_cosif_download.py           # Download missing monthly COSIF ZIPs → shared/COSIF
├── cosif_process_1_extract.py            # Extract COSIF → custos_implicitos_<TAXONOMY>.csv + per-type foundation
├── cosif_process_2_calibrate.py          # Per-bank + segment-shrunk corrected k=4 CDB rate → cosif_cdb_rate_corrected.csv
│
├── ── Stage 3: Deposit Panel, Rates & Characteristics (parallel) ──
├── panel_1_deposits.py                   # ESTBAN + IF Data → conglomerate × municipality × quarter deposit panel
├── panel_2_rates_ip.py                   # Extract IP explicit deposit rates from raw COSIF files
├── panel_3_master_panel_build.py         # Compute and append deposit rates/spreads (COSIF + SGS; corrected k=4 CDB rate)
├── panel_4_bank_chars.py                 # IF Data → conglomerate bank size & solvency characteristics panel
├── panel_5_flag_digital.py               # Identify purely digital banks from ESTBAN → PANEL_INTERMED
│
├── ── Stage 4: Master Analysis Panel ──
├── panel_6_market.py                     # Merge all MCA panels + deposit panel → master analysis dataset
├── panel_7_instruments.py                # Compute LOO instruments and FGC coverage dummy
├── panel_8_demographics_sigma.py         # Within-MCA demographic σ for BLP parametric draws → demographics_sigma.parquet
│
├── ── Stage 5: Descriptive Statistics (parallel) ──
├── desc_1.py                             # Summary statistics tables (CSV + LaTeX) by bank type and region
├── desc_2.py                             # Compact market-structure / cross-section descriptive tables
├── desc_3.py                             # Cluster-imbalance & deposit-concentration table (justifies the wild cluster bootstrap). Reads the est7 second-stage sample, so it runs AFTER the sleep estimation (wired as the final step of run_sleep_pipeline.py).
│
├── ── Deposit Rate Scraping (Internet Archive) ──
├── d_rate_scrape_1_targets.py            # Initialise target domain list for archival rate scraping
├── d_rate_scrape_2_cdx.py                # Query Wayback Machine CDX API for candidate URLs
├── d_rate_scrape_3_fetch.py              # Async-fetch HTML/PDF snapshots from Internet Archive
├── d_rate_scrape_4_parse.py              # NLP extraction of deposit yields from HTML and PDF files
│
├── ── Sleepiness Estimation ──
├── estimation_1_sleep.py                 # B-type CFA estimation (Υ); phi_mt and phi_t construction
├── estimation_2_sleep.py                 # Robustness: omit post-2020 structural break dummy
├── estimation_3_sleep.py                 # Robustness: pooled B + D firms with D-type dummy
├── estimation_4_sleep.py                 # Robustness: NLLS logistic link function
├── estimation_5_sleep.py                 # Robustness: cooperative / state-owned institution controls
│
├── ── Demand Estimation (BLP) ──
├── estimation_demand_1_prep.py           # Universal demand prep orchestrator (runs est. 1–5 in parallel)
├── estimation_1_demand_1_prep.py         # Demand prep round 1: active shares and market sizes
├── estimation_2_demand_1_prep.py         # Demand prep round 2
├── estimation_3_demand_1_prep.py         # Demand prep round 3
├── estimation_4_demand_1_prep.py         # Demand prep round 4
├── estimation_5_demand_1_prep.py         # Demand prep round 5
├── blp_logit_local.jl                    # Julia: non-RC logit demand (sanity check, θ₂ = 0)
├── blp_draws.jl                          # Julia: pre-compute Halton/quasi-Monte Carlo simulation draws
├── blp_estimation.jl                     # Julia: full BLP GMM estimation (loads draws, solves BLP)
│
├── ── Cost Estimation (BBL) ──
├── estimation_bbl_1_polfunc.py           # BBL Step 1: parametric policy functions for endogenous rates (k=4,5)
├── estimation_bbl_2_fwd_sim.jl           # BBL Step 2a: forward-simulate the ψ value-function basis (Julia)
├── estimation_bbl_3_solve.py             # BBL Step 2b: recover (ω,ζ,γ) via the eq:17 squared-hinge solve
│   #   cluster: submit_bbl.sh (driver) + submit_bbl_all.sh (orchestrator)
│
├── ── Export & Results ──
├── export_results.py                     # Orchestrator: dispatches export_*_sleep_results.py in parallel
├── export_1_sleep_results.py             # Export sleep results for estimation round 1 (tables, plots)
├── export_2_sleep_results.py             # Export sleep results for estimation round 2
├── export_3_sleep_results.py             # Export sleep results for estimation round 3
├── export_4_sleep_results.py             # Export sleep results for estimation round 4
├── export_5_sleep_results.py             # Export sleep results for estimation round 5
├── export_analyze_spec12.py              # Comparative analysis and plots across specification 1 & 2 variants
│
├── ── Tables ──
├── make_blp_latex_tables.py              # Compile BLP demand estimation results into LaTeX tables
│
├── ── HPC Submission Scripts (Yale HPC / SLURM) ──
├── submit_blp_E1.sh                      # SLURM: BLP estimation strategy 1
├── submit_blp_E2.sh                      # SLURM: BLP estimation strategy 2
├── submit_blp_E3.sh                      # SLURM: BLP estimation strategy 3
├── submit_blp_E4.sh                      # SLURM: BLP estimation strategy 4
├── submit_blp_E5.sh                      # SLURM: BLP estimation strategy 5
├── submit_blp_draws.sh                   # SLURM: pre-compute simulation draws
│
├── ── Utilities ──
├── utils/
│   ├── venv_guard.py                     # Ensures correct virtual environment is active
│   ├── toon_parser.py                    # Parses Gemini AI-generated context/configurations
│   ├── toon_parser_cli.py                # CLI interface for toon_parser
│   ├── toon_runtime.py                   # Loads TOON runtime context for path resolution
│   └── tex_preamble.py                   # Shared LaTeX preamble template for export scripts
│
├── requirements_venv_full.txt            # Full Python dependency list (with explanations)
└── requirements_toon.txt                 # Minimal scraping dependencies (beautifulsoup4, lxml)
```

---

## Pipeline Architecture

### Data Pipeline (`run_data_pipeline.py`)

Runs all download, processing, and panel-building scripts as subprocesses in dependency order. Independent steps within the same stage run in parallel.

Stage 0  (serial)    : step 1     — Download ESTBAN + IF Data raw files
Stage 1  (serial)    : step 2     — IBGE demographics → MCA panel
Stage 2  (parallel)  : steps 3–7  — Market characteristic panels (PIX, ANATEL, Inclusion,
                                     CadUnico, fees, banked fraction)
Stage 3  (parallel)  : steps 8–11 — Deposit panel, IP rates, rates/spreads, bank chars,
                                     digital bank flag
Stage 4  (serial)    : steps 12–13b — Master merge, LOO instruments, FGC dummy,
                                       demographic sigma for BLP draws
Stage 5  (parallel)  : steps 14–15 — Descriptive statistics (unweighted + market-weighted)


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
python run_sleep_pipeline.py --skip-sleep      # Skip estimation, run exports only
```

### BLP stages (call the Julia directly)

`run_blp_pipeline.py` and `run_local_pipeline.py` were REMOVED in `a30be201` (2026-08-04) with no
replacement wrapper. Call each stage directly — every script auto-discovers its routines from the
demand parquets and writes its own LaTeX tables, which is what the wrapper was doing.

```bash
# Non-RC logit sanity check (fast) — writes est*_spec12_logit.tex to Rout/ and Drafts/
julia --project=. --threads=auto blp_1_logit.jl
julia --project=. --threads=auto blp_1_logit.jl --est 8      # a single routine

# Pre-compute simulation draws
julia --project=. blp_1_draws.jl --R 2000 --seed 42

# BLP GMM (one round; see the script header for --stage values)
julia --project=. --threads=4 blp_1_estimation.jl --estim 1 --spec 12 --stage sigma --R 50 --seed 42
```

> **Stale-doc warning.** The *Usage* headers inside `blp_1_draws.jl` and `blp_1_estimation.jl`
> still show PRE-RENAME filenames (`blp_draws.jl`, `blp_estimation.jl`). Use the `blp_1_` names
> above. Only the logit line has been re-verified end-to-end (2026-08-06: clean, 57 result files
> + 9 tables); the draws/GMM lines are transcribed from those headers with the filename
> corrected, so read the header before committing to a long run.

For the end-to-end local sequence: run `run_data_pipeline.py`, then `run_sleep_pipeline.py`
(estimators → exports → demand prep), which takes `--skip-sleep`, `--sleep-only` and
`--skip-steps` to resume partway. The full chain — logit, BLP, BBL and the counterfactuals —
runs on the cluster as a single command via `pipeline_all.sh`.

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

### Cost Estimation (BBL)
Following Bajari, Benkard & Levin (2007), parametric policy functions for endogenous deposit types (k=4,5) are estimated in `estimation_bbl_1_polfunc.py` (Step 1). The ψ value-function basis is then forward-simulated under the equilibrium and deviating strategies in `estimation_bbl_2_fwd_sim.jl` (Step 2a), and marginal costs `(ω,ζ,γ)` are recovered from the eq:17 squared-hinge minimization in `estimation_bbl_3_solve.py` (Step 2b). On the cluster the stage runs via `submit_bbl_all.sh`; it writes `cost_params_E*_spec_12_*.json`, which the counterfactuals then consume.

---

## Geographic Unit

The primary geographic unit is the **MCA (Minimum Comparable Area)** — a time-consistent municipal grouping used to handle Brazilian municipal boundary changes from 2010–2024. Municipality codes follow the 7-digit IBGE standard (`CODMUN_IBGE`).

---

## Key Technologies

| Technology | Purpose |
| ---------- | ------- |
| **Python 3.x** | All data processing, estimation, and export scripts |
| **Julia** | BLP GMM demand estimation (`blp_draws.jl`, `blp_estimation.jl`, `blp_logit_local.jl`) |
| **pandas** | Data manipulation and panel construction |
| **numpy** | Numerical arrays and computations |
| **scipy** | Statistical utilities, NLLS optimisation, Halton draws |
| **statsmodels** | OLS and panel regression (sleepiness estimation, BBL) |
| **linearmodels** | Panel OLS with two-way fixed effects and clustered standard errors |
| **scikit-learn** | PCA for LOO instrument construction |
| **requests / urllib3** | Synchronous HTTP for BCB, IBGE, ANATEL, and INSS API calls |
| **aiohttp** | Async HTTP for high-throughput Internet Archive fetching |
| **beautifulsoup4 / lxml** | HTML parsing for deposit-rate page scraping |
| **pdfplumber** | PDF text extraction for archival deposit-rate parsing |
| **geopandas / geobr** | Brazilian geographic data and MCA boundary processing |
| **matplotlib** | Diagnostic and results plots |
| **pyarrow** | Parquet I/O (BLP draws and demographics sigma) |
| **concurrent.futures** | Parallel pipeline execution |

---

## Python Dependencies

All Python dependencies are listed in `requirements_venv_full.txt` with inline comments explaining the purpose and which script(s) use each package. Install with:

```bash
pip install -r requirements_venv_full.txt
```

The `requirements_toon.txt` file contains a minimal subset (beautifulsoup4, lxml) for lightweight scraping tasks only.

---

## Julia Dependencies

BLP estimation uses Julia with the following packages (defined in `Project.toml`):

| Package | Purpose |
|---------|---------|
| **Parquet2** | Read Parquet input panels from Python pipeline |
| **DataFrames** | Panel data manipulation in Julia |
| **Optim** | Outer GMM optimisation loop |
| **QuasiMonteCarlo** | Halton sequence draws for simulation |
| **Distributions** | Random-coefficient draw sampling |
| **JSON3** | Read/write estimation configuration files |
| **ArgParse** | CLI argument parsing for BLP scripts |
| **SparseArrays** | Efficient sparse matrix operations |

---

## Output Files

The pipeline produces CSV, Parquet, and pickle files organised under a `BCB/` directory tree:

| File | Description |
| ---- | ----------- |
| `BCB/Panel/deposits_panel.csv` | Conglomerate × municipality × quarter deposit balances |
| `BCB/Panel/market_panel.csv` | Master analysis dataset (deposits + all market characteristics + instruments) |
| `BCB/Panel/demographics_sigma.parquet` | Within-MCA demographic σ for BLP parametric draws |
| `BCB/Egan_et_al_2025_Rep/processed/COSIF_PROCESSED/ip_rates_quarterly.csv` | IP explicit deposit rates (quarterly) |
| `BCB/Egan_et_al_2025_Rep/processed/PANEL_INTERMED/digital_banks_diagnostic.csv` | Digital bank classification diagnostic |
| `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/SLEEPINESS/` | Sleepiness estimation results (PKL, JSON, TeX tables) |
| `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/DEMAND_PREP/rout_*/` | Active-shares panels ready for BLP (per estimation round) |
| `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/BLP_RESULTS/` | BLP demand estimation outputs (per strategy × spec) |
| `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/COST_FWD/` | BBL cost recovery forward-simulation outputs |
| `BCB/Egan_et_al_2025_Rep/processed/IP_SCRAPE/` | Archival deposit-rate HTML/PDF snapshots and extracted rates |
| `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/DESCRIPTIVES/` | Descriptive statistics tables (CSV + LaTeX) |
| `IBGE/mca_demographics_panel.csv` | MCA demographics panel |
| `ANATEL/anatel_mca_panel.csv` | Mobile connectivity panel |
| `BCB/PIX/pix_mca_panel.csv` | PIX adoption panel |
| `BCB/Inclusion/bcb_inclusion_mca_panel.csv` | Banking access-point density panel |
| `BCB/Banked/banked_fraction_mca_panel.csv` | Banked-population fraction proxy panel |

---

## Workflow Tips

### 1. Use `--from` and `--skip` to avoid re-running completed stages

```bash
python run_data_pipeline.py --from 3        # Resume from the deposit panel stage
python run_data_pipeline.py --skip 0a,0b    # Skip downloads if raw files already exist
python run_data_pipeline.py --list          # Print all step names and exit
```

### 2. Set email credentials once for overnight-run notifications

The sleep pipeline (`run_sleep_pipeline.py`) sends progress emails between steps if credentials are present in the environment:

```powershell
$env:SYS_EMAIL_USER = "your-email@gmail.com"
$env:SYS_EMAIL_PWD  = "your-16-char-app-password"   # Gmail App Password
```

### 3. Use `TOON_CONTEXT_PATH` to switch between machines without editing scripts

Store a per-machine `toon_context.json` with local paths and point the env var at it:

```powershell
$env:TOON_CONTEXT_PATH = "C:\Users\pedro\toon_yale.json"
```

This lets the same scripts resolve data directories correctly on your laptop, on Grace HPC, and in CI — without any code changes.

### 4. Always run the logit sanity check before submitting BLP to HPC

```bash
julia --project=. --threads=auto blp_1_logit.jl     # Fast local check (~minutes); catches data issues early
julia --project=. blp_1_draws.jl --R 2000           # Pre-compute draws
sbatch submit_blp_1_E1.sh                           # Only then submit to SLURM
```

### 5. `venv_guard` must come before all heavy imports

In any new script, call `ensure_project_venv` **before** importing pandas, numpy, or any third-party library. If it is placed after heavy imports the script will crash before it can relaunch into the correct venv:

```python
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)   # ← Must be first
import pandas as pd             # ← Safe now
```

### 6. Never manually parallelize Stage 2 scrapers

`scrape_5` through `scrape_16` have per-request rate-limit protections. Running them concurrently across multiple terminals will trigger IP bans from the BCB and ANATEL APIs. Let `run_data_pipeline.py` manage the controlled parallelism.

### 7. Target a single BLP specification during development

Use `--est` and `--spec` flags to run a single round rather than all 25 combinations:

```bash
julia --project=. --threads=4 blp_1_estimation.jl --estim 1 --spec 12 \
    --stage sigma --R 50 --seed 42                       # Only round 1, spec 12
```

### 8. Check `pipeline_output.txt` for a record of the last full run

This file captures stdout/stderr from `run_data_pipeline.py` and is the fastest way to diagnose failures after an overnight run without re-executing anything.

---

## Active Development Branches

### `coherence_fix`

This branch revisits the specification of the sleepiness function φ(·). The planned changes are:

- **Remove bank-level characteristics** (`X_jt`): log total assets and the equity/solvency ratio are dropped from the φ(·) regressors, making sleepiness a function of market-level variables only.
- **Remove branch count** from the set of market-level regressors (`S_t`).
- **Remove PIX volume** from the set of market-level regressors (`S_t`).

The motivation is to achieve a cleaner separation between the supply-side inertia equation and the bank-level covariates that enter the demand side, avoiding potential collinearity and improving structural coherence across the two estimation stages.

---

## References

- Egan, M., Hortaçsu, A., & Matvos, G. (2025). *Deposit Competition and Financial Fragility: Evidence from the U.S. Banking Sector.*
- Berry, S. T. (1994). *Estimating Discrete-Choice Models of Product Differentiation.* RAND Journal of Economics, 25(2), 242–262.
- Petrin, A., & Train, K. (2010). *A Control Function Approach to Endogeneity in Consumer Choice Models.* Journal of Marketing Research, 47(1), 3–13.
- Conlon, C., & Gortmaker, J. (2020). *Best Practices for Differentiated Products Demand Estimation with PyBLP.* RAND Journal of Economics, 51(4), 1108–1161.
- Bajari, P., Benkard, C. L., & Levin, J. (2007). *Estimating Dynamic Models of Imperfect Competition.* Econometrica, 75(5), 1331–1370.

