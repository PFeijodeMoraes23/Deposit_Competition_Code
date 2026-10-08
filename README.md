# Deposit Competition in Brazil

Research code for studying deposit competition in Brazil using Central Bank and Open Finance data. The project builds municipality-level banking panels and estimates deposit supply, demand, and costs, extending deposit-competition methods to the Brazilian market, with a focus on the introduction of Pix and the dichotomy between digital and non-digital banks.

**Suggested GitHub repository description**

> Research code and data pipelines for analyzing deposit competition in Brazil using Central Bank and Open Finance data, with structural estimation of deposit supply, demand, and costs.

## What this repository does

- Collects and processes Brazilian banking, demographic, financial-inclusion, connectivity, and deposit-rate data.
- Builds panel data at the prudential conglomerate × municipality × quarter level, using Minimum Comparable Areas (MCAs) to account for changes in municipal boundaries.
- Estimates deposit-supply inertia (“sleepiness”) and demand, including Berry-logit/BLP methods.
- Recovers marginal costs using Bajari–Benkard–Levin (BBL) forward simulation and supports counterfactual analysis.

The main data sources include the Central Bank of Brazil (BCB), IBGE, ANATEL, SAGI/CadÚnico, INSS, the Internet Archive, and the World Bank. Raw and generated data are not included in this repository; access to the relevant sources and configured data paths is required to run the pipelines.

## Main workflows

### Build the analysis panel

`panel_pipeline.py` runs the data-download and panel-construction steps in dependency order, followed by descriptive statistics. Its stages are numbered 0–6. The individual download and transformation steps are serialized to respect source API rate limits.

```bash
python panel_pipeline.py --list
python panel_pipeline.py
python panel_pipeline.py --from 3
python panel_pipeline.py --only 4
python panel_pipeline.py --skip 2b,2e
```

Use `python panel_pipeline.py --help` for the current options and `--list` for step IDs.

### Estimate sleepiness and prepare demand inputs

`sleep_pipeline.py` runs the sleepiness estimators, exports, demand preparation, and post-estimation summaries sequentially.

```bash
python sleep_pipeline.py
python sleep_pipeline.py --only-spec-12
python sleep_pipeline.py --skip-sleep
python sleep_pipeline.py --sleep-only
```

The script’s help text documents `--skip-steps` and the remaining options. Sleepiness estimates can be computationally intensive.

### Run BLP and BBL estimation

BLP estimation uses Julia. The old `run_blp_pipeline.py` and `run_local_pipeline.py` wrappers are not part of the current repository; use the Julia entry points directly for local work. Check each script’s usage header before starting a long estimation.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. --threads=auto blp_logit.jl
```

The full cluster workflow is orchestrated by `pipeline_all.sh`. **Submit it with SLURM; do not run it directly on a cluster login node.** For cluster prerequisites, launch instructions, and BBL operations, see [`cluster/RUNBOOK.md`](cluster/RUNBOOK.md) and [`BBL_RUNBOOK.md`](BBL_RUNBOOK.md).

## Setup

Python dependencies are listed in [`requirements_venv_full.txt`](requirements_venv_full.txt):

```bash
python -m pip install -r requirements_venv_full.txt
```

Julia dependencies are declared in [`Project.toml`](Project.toml) and [`Manifest.toml`](Manifest.toml). Instantiate the project as shown above before running Julia scripts.

The scripts expect project data directories and may rely on machine-specific paths. Configure the data paths for your environment before running a pipeline; where applicable, `TOON_CONTEXT_PATH` points to a machine-specific TOON context file. Do not commit local credentials or private data. Email notifications in `sleep_pipeline.py` are optional and require `SYS_EMAIL_PWD` in the environment.

## Key files

| File | Purpose |
|---|---|
| `panel_pipeline.py` | Downloads and builds the analysis panel |
| `sleep_pipeline.py` | Runs sleepiness estimation, exports, and demand preparation |
| `blp_logit.jl`, `blp_draws.jl`, `blp_engine_cpu.jl` | Julia demand-estimation entry points |
| `bbl_run.sh`, `bbl_polfunc.py`, `bbl_fwd_sim.jl`, `bbl_solve.py` | BBL cost-recovery workflow |
| `pipeline_all.sh` | SLURM orchestration for the cluster estimation and counterfactual workflow |
| `cluster/RUNBOOK.md`, `BBL_RUNBOOK.md` | Cluster and BBL operating instructions |
| `requirements_venv_full.txt`, `Project.toml`, `Manifest.toml` | Python and Julia environment specifications |

The repository also contains source-specific scrapers, panel transformations, diagnostics, and result-table scripts; their filenames describe their main tasks.

## Model at a glance

- **Deposit supply:** estimates persistence/inertia in deposit balances as a function of market and bank characteristics. For deposit types with endogenous rates, the estimation uses a control-function approach and instrumental variables.
- **Deposit demand:** constructs active market shares after accounting for the sleeping component, then estimates demand using Berry-style logit methods and BLP routines.
- **Marginal costs:** estimates policy functions, forward-simulates value-function terms, and recovers cost parameters using the BBL approach.

## Software dependencies

**Python** (see `requirements_venv_full.txt` and `requirements_toon.txt`):

- Data collection: `requests`, `urllib3`, `aiohttp`, `beautifulsoup4`, `lxml`, `pdfplumber`, `PyMuPDF` (`fitz`), `rapidocr_onnxruntime`, `openpyxl`
- Data handling: `pandas`, `numpy`, `pyarrow`, `polars`
- Econometrics: `statsmodels`, `linearmodels`, `scipy`
- Geographic data: `geobr`, `geopandas`
- Machine learning: `scikit-learn`
- Visualisation: `matplotlib`

**Julia** (see `Project.toml`; Julia 1.10–1.12):

- Data and I/O: `DataFrames`, `Parquet2`, `JSON3`, `ArgParse`
- Statistics and optimisation: `Distributions`, `Optim`, `LBFGSB`, `QuasiMonteCarlo`
- Performance: `MKL`, `LoopVectorization`, `CUDA`, `PackageCompiler`, `SparseArrays`
- Standard library: `LinearAlgebra`, `Statistics`, `Random`, `Serialization`, `Printf`, `Dates`, `TOML`

## References

- Methodological background: Egan, M., Hortaçsu, A., & Matvos, G. (2025). *Deposit Competition and Financial Fragility: Evidence from the U.S. Banking Sector.*
- Berry, S. T. (1994). “Estimating Discrete-Choice Models of Product Differentiation.” *RAND Journal of Economics*, 25(2), 242–262.
- Petrin, A., & Train, K. (2010). “A Control Function Approach to Endogeneity in Consumer Choice Models.” *Journal of Marketing Research*, 47(1), 3–13.
- Conlon, C., & Gortmaker, J. (2020). “Best Practices for Differentiated Products Demand Estimation with PyBLP.” *RAND Journal of Economics*, 51(4), 1108–1161.
- Bajari, P., Benkard, C. L., & Levin, J. (2007). “Estimating Dynamic Models of Imperfect Competition.” *Econometrica*, 75(5), 1331–1370.
