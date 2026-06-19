# IF-Data "Aggregated Data" reconciliation — SCOPE

Prereq to the folder reorg. Diagnosis from `_audit_output/` + direct inspection,
2026-06-18. See memory `ifdata-aggregated-inconsistency`.

## What is actually wrong (and what is NOT)
`scrape_3_ifdata_aggregate.py` builds `Aggregated Data/IF_DATA_type_{t}_report_{r}.csv`
where **t = 1 Prudential / 2 Financial / 3 Individual** and r = `NumeroRelatorio`,
by concatenating rows from the per-period `IF_DATA_Values_{year}_{q}.csv` inputs in
each conglomerate folder. Per-type status:

- **type_1 Prudential — COMPLETE & CORRECT.** 40 inputs `201603–202512`. Reports 1–6
  full history; reports 7–16 are 2025-only **because BCB only launched those
  credit-portfolio reports in 2025** (`Carteira de crédito ativa…`). Not a bug.
  This is the only data any panel consumes (`panel_1_deposits`→report_3;
  `panel_4_bank_chars`→reports 1–5).
- **type_3 Individual — COMPLETE.** 40 inputs `201603–202512`.
- **type_2 Financial — BROKEN.** Only **2** input files exist (`2025_3`, `2025_6`);
  **all pre-2025 Financial raw inputs are missing** and `scrape_1` was never wired to
  download Financial (TipoInstituicao=2) at all. Because `scrape_3.save_type_report_frames`
  only writes a report when current inputs yield rows (`if frames:`) and never cleans
  the output dir, the live folder is a Frankenstein:
  - `report_1–4` = freshly overwritten with **2025-only** (Resumo/Ativo/Passivo/DRE);
  - `report_7–14` = **stale 2016–2024 leftovers** from an old build (identical to `_bak`).

## Root cause
The pre-2025 Financial aggregates were built once (legacy `if_data_process_1.py` /
an early run) and captured in `_bak` (the one-time, never-refreshed backup —
`shutil.copytree` guarded by `not os.path.isdir(bak_dir)`). The Financial raw inputs
were later removed; the last `scrape_3` run had only 2025 Financial inputs and
overwrote `report_1–4`, leaving `report_7–14` stale.

## The only irreplaceable data
`_bak` is NOT a redundant duplicate. Its **unique** content = **Financial reports 1–4
pre-2025 history** (4 files, ~1.0 GB: 69+215+282+468 MB) — overwritten in the live
folder and **not rebuildable from raw on disk**. (`_bak report_7–14` already equals the
stale live copies; those 8 files are genuinely redundant.)

Financial (type_2) is **not consumed by any current panel** — so this is preservation
of historical data, not a live-analysis fix.

## Options (pick one — see question)
1. **Preserve & retire `_bak` (recommended).** For Financial reports 1–4, write
   canonical files = `_bak` history ⊕ live 2025 (faithful to what a full raw rebuild
   would produce; schemas/columns already match). Keep `report_7–14` as-is (already
   present). Verify, then delete `_bak`. Bounded: 4 files merged. End state: single
   complete, self-consistent `Aggregated Data/`; `_bak` gone.
2. **Archive `_bak`, don't merge.** Rename `_bak` → `Aggregated Data_archive_pre2025/`
   (documented, read-only), leave the active folder as-is. Lowest risk, no data
   writing; but the active Financial set stays split/inconsistent.
3. **Full raw recovery.** Wire `scrape_1` to fetch Financial 2016–2024 from BCB
   Olinda (TipoInstituicao=2), re-run `scrape_3` cleanly. Most "correct", highest
   effort, depends on BCB still serving it, rebuilds unused data.

## Hygiene fix (do regardless of option, separate small change)
`scrape_3` silently leaves stale files. Add: (a) at start, warn if any existing output
report has no current-input source (the Financial case); (b) optionally write to a temp
dir and swap, so the output always reflects the current inputs; (c) timestamp/refresh
the backup instead of the one-time guard. Do NOT naively clean the output dir — that
would delete the stale-but-irreplaceable Financial history before it is preserved.

## Verification (for option 1)
- Merged Financial report_R: row count == `_bak` rows + live-2025 rows; period range
  `201603..202506`; no duplicate (CNPJ,Year,Month,NumeroConta) across the seam.
- `panel_1_deposits` + `panel_4_bank_chars` still run unchanged (they read type_1 only).
- Re-run the audit: `duplicates_exact` for IF_DATA drops to ~0 once `_bak` is removed.
