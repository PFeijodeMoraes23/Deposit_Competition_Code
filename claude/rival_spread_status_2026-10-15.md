# Rival spread: panel-side status (2026-10-15 17:00)

Reply from the panel agent to `rival_spread_handoff_2026-10-15.md`.

## Done (code only; the panel is NOT rebuilt yet)

- `panel_market.py`: `calculate_rival_mean_spread_wide` replaces the row-level leave-one-out. For a row of conglomerate j in quarter t: `rival_mean_spread_a{k} = (S_t - S_jt) / (N_t - N_jt)`, NaN when `N_t - N_jt == 0`. Each row has weight 1, as before. The `_COL_ORDER` entries are renamed.
- `panel_local_cdb_spread.py`: `_adjust_rival_spread_a4` re-derives the local-patch update: `R += (dS_t - dS_jt) / (N_t - N_jt)`. A deviation in j's rows moves every other conglomerate's instrument, never j's own. The columns are `rival_mean_spread_a4` and `rival_mean_spread_a4_national`, and the docstrings are updated.
- `panel_pipeline.py`: the stage-4 comment no longer lists the column among the panel_loo_instruments outputs. panel_market builds it.

## Column set

`rival_mean_spread_a1`, `_a2`, `_a4`, `_a5`. There is no `_a3`: type 3 (interbank) is not used, so the panel has no `spread_a3` (confirmed by Pedro 10-15).

## Validation on the current panel (before the rebuild)

- The old formula reproduces the stored column to 1e-16 (types 1, 2). Types 4 and 5 differ by at most 2.5e-4, because the D-type rows of 2013 are dropped after the instrument is built.
- New vs old instrument:

| type | corr(new, old) | NaN share | mean old | mean new | within-quarter SD old | within-quarter SD new |
|---|---|---|---|---|---|---|
| a1 | 1.0000 | 0% | 0.02380 | 0.02380 | ~0 | ~0 |
| a2 | 1.0000 | 0% | 0.00843 | 0.00843 | 4.9e-19 | 8.3e-19 |
| a4 | 0.9431 | 0% | 0.00414 | 0.00376 | 9.3e-07 | 2.2e-03 |
| a5 | 1.0000 | 0% | 0.00223 | 0.00223 | | |

- Types 1, 2 and 5: every firm has the same spread in a quarter, so the instrument is the quarter value under either construction. It is still a pure quarter effect and is absorbed by quarter fixed effects.
- Type 4: the instrument now varies across firms within a quarter. Before, it was essentially the quarter mean.
- The local-patch delta matches a from-scratch recompute to 3.5e-18, with the same NaN pattern. The patch changes the a4 instrument on 65% of rows.

## What else the 10-15 evening rebuild changes for downstream (all approved by Pedro)

- **ESTBAN instrument (D6). ACTION NEEDED in the preps.**
  - `estban_rival_branches_lag` now uses a calendar lag. Unmatched B rows are **NaN**, flagged in the new 0/1 column `estban_rival_branches_missing`. D rows stay 0, by construction: it is a B-only instrument.
  - About 599 B rows inside 2016Q1-2024Q4 stay NaN after the 2015Q4 refill in `demand_chars`. Every 2013-2015 B row is NaN.
  - `utils/demand_chars.finalize` asserts finite `IV_ENGINE` values, so `sleep_demand_prep_*` and `switch_cost_sleep_demand_prep_link` will abort until they drop those rows, or fill 0 and add the flag as a control. That choice is Pedro's or yours.
  - The BLP engines throw `DemandInputError` unless `BLP_ZERO_FILL_MISSING=1`.
  - Agent report, per reader: `C:\Users\pedro\AppData\Local\Temp\claude\d6\verify.log`.
- **Conglomerate key for the COSIF rates (D1).**
  - The corrected type-4 rate and the step-16 rates are now keyed the way the deposit panel files each CNPJ: an ESTBAN bank under its static code in every quarter, everyone else under the canonicalized reported code.
  - Pan, C6 Consignado and PicPay recover their corrected rate. ESTBAN type-4 volume with no corrected rate falls from 1.03% to 0.02% in 2016-2024.
  - Pre-2025 corrected rates move a little for most banks, because the shrinkage groups changed.
- **Segment source of step 13 (D3).** The List `Sr`. Step 13 no longer reads market_panel.csv.
- **Bank characteristics (D2).** 233 cells per DRE flow (personnel, admin, tax, fin_income) that were semester totals filed as one quarter are now NaN. The cost ratios follow (214 NaN each).
- **Payment-institution rates (D4).** Parents only from 2025, and no 499. There are no pre-2025 IP rates now, so pre-2025 `rate_a5` is the quarterly median or the CDI fallback.

## Rebuild DONE (2026-10-15 19:43-20:01)

- **Command:** `panel_pipeline.py --from 2 --to 4 --skip 5,6,7,8,9,11,12,38,21`.
  - Steps run: 13, 14, 15, 16, 17, 18, 19, 20, 36, 37, 39, 22. All 12 exit 0, 18.1 min.
  - Skipped: the scrapers, so the inputs are unchanged; COSIF extract (unchanged); sigma (current).
  - Stage 5 (descriptive tables) was NOT run, because its generators are being renamed on your side. Run it after your renames.
- **`market_panel.parquet` md5: `3cd0df51c013718616c926d78966a956`** (csv 760,432,619 B; parquet sidecar refreshed 19:56).
- **`demand_chars.parquet` sha256: `75f455407200b0cd45adb893f34fd5844461d8290b877acd9cbdbb11d82daa74`.** It is at `processed/demand_chars.parquet` and records panel md5 3cd0df51.
- **`build_stagec_bundle.py` pins:** `PANEL_MD5`, `CHARS_SHA256` and `MS982_MD5["market_panel.parquet"]` now refer to the old files. They are yours to update.
- **Audits:**
  - `panel_demand_chars_checks.py`: 29/29 PASS.
  - `panel_audit_mca_coverage.py --strict`: all 9 variables pass.
  - `panel_audit_firm_class.py --compare` against the 19:43 baseline: identical. That is 500,186 rows, 536 firms (79 B, 457 D), 3,737 MCAs, and dep_a4 by year to the cent.
- **Columns:**
  - `rival_mean_spread_a1`, `_a2`, `_a4`, `_a4_national` and `_a5`. No `leave_one_out*` column is left.
  - `estban_rival_branches_lag` and `estban_rival_branches_missing`.

### Instrument on the rebuilt panel

The comparison is the rival mean against the row-level leave-one-out, both computed on the same rows. Type 4 uses the national values.

| type | corr | NaN share | mean rival | mean row-LOO | within-quarter SD rival | within-quarter SD row-LOO |
|---|---|---|---|---|---|---|
| a1 | 1.0000 | 0% | 0.02380 | 0.02380 | 2.8e-17 | 6.9e-19 |
| a2 | 1.0000 | 0% | 0.00843 | 0.00843 | 2.9e-17 | 3.7e-19 |
| a4 | 0.9420 | 0% | 0.00329 | 0.00379 | 1.9e-03 | 8.9e-07 |
| a5 | 0.9998 | 0% | 0.00202 | 0.00201 | 1.6e-04 | 1.1e-07 |

- The local patch moves `rival_mean_spread_a4` on 57.0% of rows.
- Type 5 now has a little within-quarter variation (1.6e-4): the 2025 payment-institution rates differ across firms.

### ESTBAN instrument on the rebuilt panel

| | zero | NaN | positive |
|---|---|---|---|
| B rows | 42,455 | 139,448 | 309,388 |
| D rows | 8,895 | 0 | 0 |

- 11,196 B rows inside 2016Q1-2024Q4 are NaN. `demand_chars` refills 10,597 of them, all in 2016Q1, with the 2015Q4 count; check T8 passes.
- That leaves about 599 for the preps to handle. See the action item above.

### Surprises

- **The type-4 plausibility guard bites more.** Step 16 now replaces 1,109 of 19,346 bank-quarters with the CDI (5.73%), against 1,077 (5.57%) before. More corrected rates reach the merge, and some of them are implausible.
- **Pre-2025 `spread_a5` is about 0 every year:** `rate_a5` falls to the CDI fallback, so Selic minus CDI. It was the same before D4, so this is not new.
- **The step-16 COSIF time-aggregation correction takes 4.5 min.** A per-group Python lambda; the audit listed it as an efficiency item, not a correctness one.

## Also changed today, touching the bundle

- `demographics_sigma.parquet` was rebuilt at 16:09 without its 8 unread mean columns. The keys and the 8 sigma columns are bit-identical. New sha256 `d96c647298a94c8b3428302132524558c28c1a35577f746d0833fb49f7cd21e6` (md5 `06c8ddaed0325ffa915523b230916717`).
- With Pedro's OK, `SIGMA_SHA256` in `build_stagec_bundle.py` is updated to it, and the V7 wording in `verify_bundle.py` too. The 10-14 file (`9e0cfc96...`) gives the same draws.
