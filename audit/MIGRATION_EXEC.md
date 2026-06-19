# Migration EXECUTION plan (chosen approach, 2026-06-18)

Decisions: **physically relocate** the big shared datasets into `Open-Finance/shared/`;
**consolidate** the ~40 loose `Code/*` scripts into named subprojects; **dedupe +
resolve collisions**; **rewire** everything through `of_paths.py` / `paths.R`. Venv
move = deferred (use `MPLBACKEND=Agg`, see [[venv-onedrive-matplotlib]]).

## Safety-critical ordering
Rewire code to the **config** first (config points at CURRENT locations → no behavior
change), verify, and only then move data + flip the config. This keeps every step
individually reversible/verifiable and avoids a broken window. The Egan repo's live
pipeline reads ESTBAN/IF_DATA/COSIF, so those move LAST and with a full pipeline re-verify.

## Phases
- **P0 Foundation — DONE.** `Code/of_paths.py` (Python) + `Code/paths.R` (R), anchored
  on `OPEN_FINANCE` (env `OPEN_FINANCE_ROOT`), with `OF_USE_SHARED=1` to flip all shared
  datasets to `shared/` at once. `shared/` skeleton + README created. Config verified:
  resolves anchor; all 8 datasets point at existing current locations.

- **P1 Consolidate loose scripts** (CODE moves only — reversible via the `Code/.git` repo).
  Target subprojects (by primary dataset; cross-refs noted):
  - `Code/RAIS/` ← base_dos_dados_RAIS.py, public_RAIS_*.py (analysis/graphs/query+conc) ~17
  - `Code/SCR/` ← SCR_*.R, Analysis_3.R, Inflation.R, openbanking_4.py, openfinance_rankings_1.R
  - `Code/PIX/` ← Analysis_1/2.R, PIX_process_*.R, Portability_Presentation.r, openfinance_calls_1.R
  - `Code/Portability/` ← Portabilidade_process_*.R, Interest_1.R, SCR_time_series_by_modalidade_*.R
  - `Code/IFData_legacy/` ← if_data_scrape_1.py, if_data_process_1.py, ESTBAN_Process_1.R
    (DEPRECATED — superseded by Egan repo scrape_1/scrape_3; archive, don't rewire heavily)
  - `Code/_utils/` ← R_fix.r, clear_geobr_cache.py, openfinance_APIcalls.R
  Use `git mv`; fix any intra-script `source()`/relative refs after moving. Verify each
  script still parses (py_compile / `Rscript -e 'parse()'`).

- **P2 Rewire path literals → config.** In each consolidated script, replace hardcoded
  dataset paths with `of_paths`/`paths.R` constants (config still = current locations →
  outputs unchanged). Punch list = `_audit_output/path_refs.csv` (kind=literal). Verify a
  representative Python and R script per group runs to the same output.

- **P3 Dedupe + collisions** (low risk). `duplicates_exact.csv`: promote the cross-project
  `muni_mca_regions_2010_2024.json` (keep `IBGE/`, drop `Code/IBGE/`, repoint). The IF_DATA
  `_bak` dup is already resolved. `collisions_name.csv`: resolve `IF_DATA_Values_*` drift.

- **P4 Egan repo shared-data rewire** (delicate — touches the live pipeline). Point
  `utils/paths.py` ESTBAN/IF_DATA/COSIF at `of_paths` (still current locations), re-run the
  data pipeline `--from 3 --to 4` + a logit smoke test to confirm identical outputs.

- **P5 Physical data move — BIG / SLOW / checkpointed.** Move each dataset into `shared/`
  **smallest-first** (SEC → STR → PIX → COSIF → IF_DATA → ESTBAN → SCR 57 GB → RAIS 149 GB),
  using `robocopy /MOVE` (or Explorer). After EACH dataset: flip its config line (or set
  `OF_USE_SHARED=1` once all done) and verify its consumers. ~210 GB on OneDrive = hours of
  re-sync; do deliberately, one dataset per checkpoint.

- **P6 Final verify.** Re-run the audit (`duplicates_exact`≈0; every project shows a path
  module); smoke-run the Egan pipeline + one RAIS and one SCR script; `OPEN_FINANCE_ROOT`
  relocation test.

## Reversibility / checkpoints
P1–P3 are safe (code in git + small dedup). **P4 and P5 touch the just-fixed pipeline and
move ~210 GB → confirm with the user before each.** Pre-move manifest = `_audit_output/
inventory.csv`; deletions go via Recycle Bin.
