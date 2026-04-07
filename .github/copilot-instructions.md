# Egan et al (2025) Replication & Extension

## Architecture & General Overview
This project implements a large-scale data pipeline and economic estimation framework focusing on the Brazilian banking sector (Open Finance, PIX, deposits, banking access). It replicates and extends models (e.g., Egan et al., BLP logit demand) to analyze consumer "sleepiness" (inertia) and deposit elasticity.

- **Data Gathering**: Scrapers (`scrape_*.py`, `if_data_scrape_1.py`) for Brazilian APIs (BCB Olinda, CadUnico, Anatel).
- **Panel Building**: Generates intermediate datasets and merges them into a master 1.4GB `market_panel.csv` (`panel_5_market.py`).
- **Sleepiness Pipeline**: Estimates consumer inertia at national and firm levels (`run_sleep_pipeline.py`).
- **BLP Estimation**: Partitioned between local prep and HPC parallel loop execution to protect against RAM exhaustion. See `HPC_README.md` for full HPC execution instructions.

## Build and Test Commands
- **Full Data Pipeline**: `python run_data_pipeline.py` (Supports `--from <stage>`, `--only <stage>`, or `--skip <steps>`). 
- **Sleepiness Estimations**: `python run_sleep_pipeline.py`
- **HPC BLP Loop**: See `HPC_README.md` for SLURM payload commands and multiprocessing notes (e.g., `sbatch submit_blp_hpc.sh`). The main loop script is `estimation_1_demand_3_loop.py`.

## Conventions & Gotchas
- **Venv Guardian**: Python scripts frequently self-relaunch into the local `.venv`. *Crucially*, the `venv_guard` must be called *before* importing heavy third-party modules (like `pandas` or `numpy`), otherwise the script might crash before successfully relaunching.
- **Python in PowerShell**: When using PowerShell, always run quoted executables with the call operator `&`, e.g., `& "path/to/python.exe" -m pip ...`.
- **Parallel processing & Memory**: BLP estimations can exhaust hundreds of GBs of RAM on HPC; adhere to the partitioned `.pkl` approach rather than loading `market_panel.csv` directly into multiprocessing workers.
- **Rate Limiting**: Do not parallelize the Stage 2 data scrapers. Doing so will trigger concurrent rate-limit bans from the BCB / ANATEL APIs.
- **Serialization**: The pipeline heavily uses `.pkl` for large internal memory buffers and plain-text `.json` output logs for HPC monitoring.
- **Terminal Encoding**: Console outputs are forced to `latin-1` (with line buffering) to handle CP1252 Windows terminal environments natively.
- **HTML Parsing**: Gemini chat HTML placeholders may be effectively empty (11-byte `<div></div>`), so parsers must fail-soft and keep defaults.
- **AI Models**: Never recommend or insert usage of OpenAI (ChatGPT) models or agents; use Gemini alternatives where applicable as per project preferences.
