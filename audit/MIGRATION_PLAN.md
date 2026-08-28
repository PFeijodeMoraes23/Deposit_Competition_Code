# Open Finance reorganization — MIGRATION PLAN (draft for approval)

Second pass, following the audit (`_audit_output/SUMMARY.md`, 2026-06-18). This
is a **proposal** — nothing here has been executed. Target layout (your choice):
**per-project ownership + an `Open-Finance/shared/<DOMAIN>/` tree for genuinely
cross-project datasets.**

## What the audit established (the inputs to this plan)
- 3,357 data files, ~281 GB. Dominated by **RAIS/RAIS Publica 149 GB**,
  **BCB/Egan_et_al_2025_Rep 63 GB**, **BCB/SCR 57 GB**. These three are 95% of bytes.
- **~5.0 GB exact duplicates** (`duplicates_exact.csv`), almost all intra-repo:
  - `raw/IF_DATA/Aggregated Data/` vs sibling **`Aggregated Data_bak/`** (~4 GB).
  - `ESTIMATION_OUTPUT/COST_POLFUNC/` outputs replicated across spec folders:
    `polfunc_results_spec_*.pkl` ×11 (883 MB), `polfunc_fitted_spec_*.csv` ×12
    (238 MB), `polfunc_summary_spec_*.json` ×12.
  - Cross-project tiny: `muni_mca_regions_2010_2024.json` in both `Code/IBGE/`
    and `IBGE/`; `portability_analysis_{Denied,Granted}.csv` in `BCB/` and
    `Presentations/Images/`.
- **94 name-collisions** (`collisions_name.csv`), mostly `IF_DATA_Values_YYYY_M.csv`
  pairs inside the Egan repo (two locations, differing/unknown content = drift).
- **Grandparent stray:** `Open Finance/BCB/Egan_et_al_2025_Rep/processed/` holds 4
  stale files (10.5 MB) — misdirected writes to the wrong root.
- **Path hygiene:** only `Code/Egan_et_al_2025_Rep` centralizes paths (`utils/paths.py`).
  All loose `Code/*` scripts (R + Python) and `Drafts/` hardcode. Cross-project
  consumers point at: **SCR, ESTBAN, IF_DATA/COSIF, PIX, STR, RAIS, SEC**.

## Guiding principle: rewire, don't relocate (for the big stuff)
Moving 200 GB inside OneDrive forces a full re-upload and risks sync corruption.
So: **designate a canonical location where the data already lives** and point a
shared path config at it. Physically move only (a) duplicates we delete and (b)
small misplaced files. `shared/` is realized as the canonical anchor in the path
config — not necessarily a physical `shared/` folder for the 149 GB trees.

## Phase A — Reclaim duplicate space (~5 GB, low risk, do first)
Worklist = `duplicates_exact.csv` (already byte-verified equal by sha256).
1. **Delete `Open-Finance/BCB/Egan_et_al_2025_Rep/raw/IF_DATA/Aggregated Data_bak/`**
   (~4 GB) — confirmed identical to `Aggregated Data/`. Biggest single win.
2. **COST_POLFUNC replication:** the `polfunc_*_spec_N` files are byte-identical
   *across spec numbers* — i.e. specs 10/11/12… share one result. Investigate the
   producing script (likely writes the same artifact under each spec name). Either
   keep one canonical + symlink, or confirm they're cheaply regenerable and prune.
   Do NOT delete blind — verify they are not spec-specific outputs that merely
   happened to collide.
3. **IBGE json:** keep `Open-Finance/IBGE/muni_mca_regions_2010_2024.json`
   (data home), delete the `Code/IBGE/` copy, repoint the script that reads it.
4. **portability_analysis_*.csv:** keep the `BCB/` copy; the `Presentations/Images/`
   copies are figure-run snapshots — leave or delete per your call.
5. **Grandparent stray (10.5 MB):** diff the 4 files vs their canonical
   `Open-Finance/BCB/Egan_et_al_2025_Rep/processed/...` counterparts; if identical
   or older, delete the grandparent `Open Finance/BCB/` tree.

## Phase B — Resolve version drift (94 groups, medium risk)
Worklist = `collisions_name.csv`. For each group inspect the two locations,
pick the authoritative copy (newest mtime / matches the live pipeline output),
archive or delete the stale one, and note any script that points at the stale
path. The `IF_DATA_Values_*` clusters are one recurring pattern — resolve the
pattern once (find the two parent dirs, decide which is canonical) and apply to all.

## Phase C — Shared anchor + per-project path config (the structural fix)
1. **Decide canonical homes** for the cross-project datasets (SCR, ESTBAN,
   IF_DATA/COSIF, PIX, STR, RAIS, SEC). Default: where the largest/cleanest copy
   already sits. Record them in one table (extend `utils/paths.py`'s convention).
2. **Python:** generalize `Code/Egan_et_al_2025_Rep/utils/paths.py` into a tiny
   shared module importable by the loose `Code/*.py` scripts (e.g.
   `Code/of_paths.py` exposing the same `OPEN_FINANCE` anchor + `OPEN_FINANCE_ROOT`
   override + per-dataset constants). The Egan repo keeps its own `utils/paths.py`
   (re-exporting from the shared one) so nothing in the pipeline breaks.
3. **R:** add `Code/paths.R` mirroring the same anchors (resolve root via
   `Sys.getenv("OPEN_FINANCE_ROOT")` then `rprojroot`/`here`), `source()`-d by the
   loose `*.R` scripts. R is the bulk of the loose scripts (SCR_*, Analysis_*,
   openfinance_*, Portabilidade_*).
4. **Optionally** create a physical `Open-Finance/shared/` only for small shared
   datasets; large trees stay in place but are addressed *only* through the config.

## Phase D — Rewire scripts incrementally + verify
- Replace hardcoded literals with config references, **one project at a time**
   (start with the loose `Code/` Python, then R). `path_refs.csv` is the punch list
   (kind=`literal` rows, grouped by project, with `inferred_dataset`).
- After each project: run its script(s) and diff outputs against a pre-change run.
- **End-to-end verification:**
  1. Re-run the audit (`audit_1`→`audit_2`→`audit_report`); expect
     `duplicates_exact.csv` ≈ empty and reclaimable ≈ 0.
  2. `path_refs.csv` / SUMMARY "Path hygiene" shows **every** project
     `has path module = yes`.
  3. Smoke-run the Egan pipeline (`panel_pipeline.py`) and one representative
     loose script per language (e.g. a `public_RAIS_*.py` and an `SCR_*.R`) to
     confirm all paths resolve with no hardcoded fallbacks.
  4. Set `OPEN_FINANCE_ROOT` to a throwaway copy and confirm the whole thing
     relocates cleanly (the real test that rewiring worked).

## Open questions to settle before Phase C
- For the 149 GB RAIS and 57 GB SCR trees: canonical-in-place (recommended) or a
  real physical move into `shared/`? (Move = large OneDrive re-upload.)
- Do the loose `Code/*` scripts get consolidated into named subprojects
  (`Code/RAIS/`, `Code/SCR/`, …) as part of this, or left flat and just rewired?
- Is `Drafts/` in scope for path rewiring now, or later?

## Risks / safeguards
- All Phase A/B deletions are reversible only via OneDrive version history — take a
  manifest (`inventory.csv` is one) before deleting, and delete in small batches.
- Large data is not in git; rely on OneDrive recycle bin + the pre-change inventory.
- Keep the `OPEN_FINANCE_ROOT` override working at every step so a bad path edit
  fails loudly rather than silently writing to the wrong root (the bug that created
  the grandparent stray in the first place).
