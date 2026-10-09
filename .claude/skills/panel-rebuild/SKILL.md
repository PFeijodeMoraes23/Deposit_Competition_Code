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
- Say which upstream sources will be re-fetched. The scrapers re-download only the trailing year.

## Running

`panel_pipeline.py --from N`, `--only N`, `--to N`, `--skip <step ids>`. Stages: 0 raw downloads,
1 demographics, 2 market characteristics and COSIF, 3 deposits, rates, bank characteristics and
the digital verdict, 4 the master panel, 5 descriptive tables, 6 disclosures.

**Stage 4's order is load-bearing.** `panel_market.py` builds the panel; `panel_loo_instruments.py`
overwrites it with the leave-one-out instruments and the FGC dummy, after cleaning the accounting
ratios; `panel_estban_instrument.py` and `panel_local_cdb_spread.py` patch it; `panel_fee_merge.py
--patch-market` writes the separate fee panel last. Re-running `panel_market.py` alone wipes the
instruments and the local spread, and the BLP then drops the missing instruments silently and is
identified off the cost shifters alone. After any change to the base panel, run the whole stage
(`--only 4`).

Things a re-run does not do by itself:

- `scrape_ibge_demographics.py`, `scrape_anatel_mobile.py`, `scrape_bcb_inclusion.py` and
  `scrape_bcb_tarifas.py` return at the top of `main()` when their output exists, with no flag to
  override. A fix to one of them appears to run and changes nothing until the old output is moved
  aside. Ask before moving it. The demographics step also keeps an existing crosswalk, which
  preserves the set of markets and is what you want.
- `scrape_ifdata_aggregate.py` leaves one timestamped backup of about 8 GB per rebuild. That is
  by design; mention it so the user can reclaim the space.
- `make_margin_figures.py` reads the panel and is not a step of the runner. Run it after a rebuild.

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
   `leave_one_out_mean_spread`, `estban_rival_branches_lag`); the Basel ratio columns
   (`indice_basileia*`) lie in [-1, 2], the validity bound applied before the 1/99 winsorization
   within firm type.
5. The sidecar: every writer of `market_panel.csv` refreshes `market_panel.parquet` and logs
   `panel cache refreshed`. The freshness gate compares the recorded source-CSV size as well as
   the dates, because OneDrive can move dates.

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
- The state-centering means (`state_centering_means.json`, written only by
  `sleep_est_e2.py --write-centers`); `sleep_audit_state_centering.py --tier0` verifies them.
- The policy function, which is fitted on the panel, and everything simulated from it.
- The copy of `market_panel.parquet` on the cluster.
- The descriptive tables, unless stage 5 ran, and the paper numbers built from them (the
  `post-run-refresh` skill).

## Report

A before-and-after table of shape and totals, each audit's verdict, what moved and why, the list
of stale downstream artefacts, and any step that was skipped with the reason.
