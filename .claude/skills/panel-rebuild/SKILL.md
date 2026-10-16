---
name: panel-rebuild
description: Rebuild or extend the market panel with panel_pipeline.py and the panel_ and scrape_ steps, and verify it before anything is estimated on it. Use whenever a task reruns or changes a panel step, a scraper output, the municipality crosswalk, firm classification, instruments, winsorization or the fee merge, whenever the user asks whether the panel is current or why a column is missing, constant or implausible, and before any side task reads panel data, because side work must leave the data pipeline untouched.
---

# Rebuilding the market panel

`market_panel.csv` (with its parquet sidecar) feeds every estimator. A step that fails here rarely
raises: a failed merge becomes a median, a missing instrument is dropped without a word, an old
file shadows a new one. So a rebuild has three parts, and the middle one is the shortest: record
the state before, run the steps, audit the result before anyone estimates on it.

## First: is this a pipeline task?

If the task is something else that merely needs panel data (the advertising panel, a diagnostic,
a figure), treat the pipeline as read-only:

- Read from `market_panel.parquet`. Write nothing under `BCB/Egan_et_al_2025_Rep/processed`.
- Do not import `panel_*`, `sleep_*`, `make_*` or most `scrape_*` modules to borrow a helper.
  They call the venv guard at import, and under any interpreter other than the project's they
  re-launch themselves as scripts. On 2026-09-14 an import of `panel_deposits` rebuilt and
  overwrote `deposits_panel.csv`. Reimplement the small read locally.
- If the side task seems to need a pipeline write, stop and ask. The user has declined to restore
  pipeline files even from a verified copy.

The rest of this file is for a task that is explicitly a rebuild.

## Setup

- Interpreter: `C:/venvs/egan/Scripts/python.exe`, always. The venv lives outside OneDrive and
  must stay there; never create a `.venv` folder or junction inside the repo.
- `MPLBACKEND=Agg` for anything that draws.
- The chain from the deposit panel through the descriptive tables took about 40 minutes on
  2026-08-26, half of it `panel_demographics_sigma.py`; the scrapers of stages 0 to 2 come on top.
  Run it alone: two heavy jobs on this machine both finish later, and a memory spike beside a
  large worker has crashed runs.
- Judge a long step alive by its output file's modification time and by the CPU of the real worker
  (`python3.11.exe`, or `julia.exe` under it). The venv's `python.exe` is a launcher shim that
  always looks idle, and log silence for an hour is normal for some steps.

## Before

- `python panel_pipeline.py --list` shows the steps and waves as they are now.
- `python panel_audit_firm_class.py --baseline <scratchpad>/firm_class_before.json`.
- Record the panel's shape and a few totals: rows, firms, MCAs, columns, rows by `is_B`, and the
  sum of each deposit column. The CSV is about 760 MB; record numbers rather than copying it.
- Say which upstream sources will be re-fetched. The scrapers of stages 0 to 2 re-download only
  the trailing year; stage 6, when opted in, re-scrapes everything.

## Running

`panel_pipeline.py --from N`, `--only N`, `--to N`, `--skip <step ids>`. Stages: 0 raw downloads,
1 demographics, 2 market characteristics and COSIF, 3 deposits, rates, bank characteristics and
the digital verdict, 4 the master panel, 5 descriptive tables. Stage 6, the extensive-margin
disclosures frozen in `extensive_margin/` (2026-10-14), runs only with `--extensive-margin` or
`--only 6`; nothing in the paper reads it. The BCB tariff scrapers are frozen in `fees/` and are
not a step; `panel_fee_merge.py` reads their last outputs.

**Stage 4's order is load-bearing.** `panel_market.py` builds the panel; `panel_loo_instruments.py`
overwrites it with the leave-one-out instruments and the FGC dummy, after cleaning the accounting
ratios; `panel_estban_instrument.py` and `panel_local_cdb_spread.py` patch it; `panel_fee_merge.py
--patch-market` writes the separate fee panel last. Re-running `panel_market.py` alone wipes the
instruments and the local spread, and the BLP then drops the missing instruments silently and is
identified off the cost shifters alone. After any change to the base panel, run the whole stage
(`--only 4`).

Things a re-run does not do by itself:

- `scrape_ibge_demographics.py`, `scrape_anatel_mobile.py` and `scrape_bcb_inclusion.py` return
  at the top of `main()` when their output exists. The pipeline never passes `--force`, so a fix
  to one of them appears to run and changes nothing until that script is run by hand with
  `--force`. Ask before forcing one. The demographics step keeps an existing crosswalk even with
  `--force`, which preserves the set of markets and is what you want.
- `scrape_ifdata_aggregate.py` builds only the type-1 reports 1-5 that `panel_bank_chars.py` and
  `panel_deposits.py` read (`--all` builds every type x report) and leaves one timestamped backup
  of the reports it overwrites, about 2 GB. Mention it so the user can reclaim the space.
- `scrape_bcb_estban_ifdata.py` (step 1) and `scrape_cosif_download.py` (step 11) exit 1 when a
  download fails, write nothing for it and list the failures, so the pipeline stops there. A BCB
  outage therefore stops a rebuild at step 1; rerun later, or `--skip` the step to keep the cached
  files. Step 1 fetches IF-Data types 1 and 3 only (`--if-types 1,2,3` adds type 2, which nothing
  reads) and no per-branch ESTBAN files.
- `make_margin_figures.py` reads the panel and is not a step of the runner. Run it after a rebuild.
- Step 39, `panel_demand_chars.py`, writes `processed/demand_chars.parquet`: the corrections the
  demand prep applies (2016Q1 lags from 2015Q4, nearest-filing fills, segment back-fill, the
  re-timed borrowings ratio). The file records the md5 of the market panel it was built from, and
  the demand prep refuses to run on any other panel. So after any change to the panel, step 39
  must run again.

## Due at the next full rebuild (decided 2026-10-13; each changes numbers downstream)

- **The ESTBAN double read.** `panel_deposits.py` (L.624–626) lists the pre-2016 raw ESTBAN files
  with `glob("*.CSV") + glob("*.csv")`, and Windows matches each file twice, so 2013–2015 balances
  are doubled. `panel_demand_chars.py` currently halves the 2015Q4 balances to offset it. The two
  must change TOGETHER: read each file once, remove the halving, and update check T1 of
  `panel_demand_chars_checks.py`, which asserts the factor is exactly 2 in 2015-12.
- **The k calibration** (`panel_cosif_calibrate.py`, the type-4 rate's de-bias factor).
  `load_segment_map` now raises instead of silently putting every bank on the global profile. In
  the project venv, `pd.read_csv(market_panel.csv, usecols=...)` raises
  `IndexError('list index out of range')`, so fix that read first. Three more bugs, from
  `checkpoints_20260928/estban_local_rate_20261013/K_AUDIT.md`:
  - n_i counts 2026Q1, which has no CDI;
  - 59 of 116 banks with 2025 leaf data get no own k;
  - k is estimated with each bank's own repo/bills ratios but applied with the shrunk ones.

  Fixing these moves the 2016–2024 type-4 rate, so treat the result as a new vintage.

## After: the audits

Run these before any estimation. Each exists because the failure it catches was invisible for weeks.

1. `python panel_audit_mca_coverage.py --strict`. It checks coverage and cross-market variation
   of the state and demographic variables. The sleepiness prep median-fills them, so a failed
   merge never raises, and a column that is fully populated but constant is as broken as one that
   is mostly missing. When it fails, compare each `*_mca_panel.csv`'s date and distinct-MCA count
   with `IBGE/muni_mca_regions_2010_2024_panel.csv`: a source panel older than the crosswalk is
   the usual cause.
2. `python panel_audit_firm_class.py --compare <the baseline json>`. Verdict, coherence with the
   market tier, identity (no institution under two conglomerate codes in one quarter) and booking
   stability. `is_B` is the one stored verdict (1 = brick-and-mortar, 0 = digital), written by
   `panel_market.py`; downstream code reads it and never re-derives firm type from `CODMUN_IBGE`.
3. `python panel_audit_fee_coverage.py --strict`, when the fee panel was rebuilt. It reads
   `market_panel_with_fees.csv` directly, on purpose.
4. By hand, against the "before" numbers: shape and totals; the instrument columns are present
   (`loo_log_assets`, `mean_loo_log_assets`, `loo_equity_ratio`, `loo_basileia`,
   `rival_mean_spread`, `estban_rival_branches_lag`); the Basel ratio columns
   (`indice_basileia*`) lie in [-1, 2], the validity bound applied before the 1/99 winsorization
   within firm type.
5. The sidecar: every writer of `market_panel.csv` refreshes `market_panel.parquet` and logs
   `panel cache refreshed`. The freshness gate compares the recorded source-CSV size as well as
   the dates, because OneDrive can move dates.
6. `python panel_demand_chars_checks.py`, after step 39: 29 checks of the demand corrections file
   against its sources. It exits non-zero if any fails.

## Known oddities that are not new defects

- The estimators read `market_panel.csv`. The fee panel is opt-in (`USE_FEE_PANEL=1`), so a stale
  fee panel cannot shadow the base panel.
- Some conglomerate-quarters have implausible deposit jumps (Mercantil 2023Q3 to Q4, several firms
  in 2016Q1), and 2025 has rows with negative deposits. The analysis window ends in 2024Q4 and
  the user has ruled the 2025 rows not an issue; do not raise them again unless the window moves.
- A date gap between two panel files proves nothing. To claim the data changed, re-estimate or
  diff the values.
- Large percentage swings in a near-zero mean (a spread of 0.0001) are noise.

## What a rebuild leaves stale

Name these in the report; re-estimating is the user's decision, not a step of the rebuild.

- The demand parquets in `DEMAND_PREP`, until the sleepiness estimators and the demand prep run again.
- `processed/demand_chars.parquet`, until step 39 runs again. The demand prep refuses a
  corrections file built from another panel.
- The state-centering means (`state_centering_means.json`, written only by
  `sleep_est_e2.py --write-centers`); `sleep_audit_state_centering.py --tier0` verifies them.
- The policy function, which is fitted on the panel, and everything simulated from it.
- The copy of `market_panel.parquet` on the cluster.
- `DEMAND_PREP/demographics_sigma.parquet` (step 21) is an upload: the RC-BLP draws on the cluster
  are built from it, so when it changes, upload it and rebuild the draws (`blp_run.sh --draws`; a
  bare run skips them because the old ones exist).
- The descriptive tables, unless stage 5 ran, and the paper numbers built from them (the
  `post-run-refresh` skill).

## Report

A before-and-after table of shape and totals, each audit's verdict, what moved and why, the list
of stale downstream artefacts, and any step that was skipped with the reason.
