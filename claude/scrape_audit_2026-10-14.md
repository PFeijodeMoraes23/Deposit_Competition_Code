# Audit of the `scrape_*` scripts, 2026-10-14

This audit was read-only. Nothing in the repo or the data folders was changed, and no scraper was run.

**Scope.** There are 55 `scrape_*.py` scripts, 32,188 lines in all. Eight readers each read one family in full and searched the repo for the readers of every output. `[checked]` marks a finding that was re-verified by hand against the code or the files on disk. The other findings carry file:line references for checking.

## 0. Summary

- **Where the scrapers run.** `panel_pipeline.py` runs 19 of them: stages 0 to 3 and stage 6. The other 36 are run by hand or called by other scripts.
- **The finding that can move results.** The ANATEL sigma used in the RC-BLP draws is computed over raw records (family B) [checked]. Nothing else found here changes a number in the paper today.
- **Strong retire or freeze candidates: 15 scripts, about 7,600 lines** (including `utils/note_sources.py`).
  - the deposit-rate web scrape: 7 scripts, 2,324 lines;
  - the three probes;
  - two stage-6 template seeders;
  - the OFB Wayback probe;
  - the statement-note advertising parser;
  - `scrape_inss`.
- **Second tier, depending on what should stay maintained:**
  - the non-main advertising sources, about 10,500 lines;
  - pipeline stage 6, about 1,200 lines plus about 4 GB of raw archives;
  - the Pix timeline, 737 lines;
  - complaints, 501 lines;
  - the IBGE region fill, 134 lines.
- **Shared helpers** would remove roughly 1,000 to 1,500 lines. Most of the code is parsing specific to each source, so retiring scripts is the big lever.
- **Unread data.** About 27 GB on OneDrive has no code reader (section 6).

## 1. Family by family

### A. Raw BCB downloads (steps 1-3 and 11), COSIF fees, Pillar 3, forward curve

Every script here has a live reader, so none is a retire candidate.

**`scrape_bcb_estban_ifdata` (503 lines, step 1).** Keep.
- `--end-year` defaults to 2025 (:486), so IF.data stops at 2025-12 while ESTBAN reaches 2026-03.
- Failure stubs are cached for good and the script still exits 0 (:120, :147-149, :188, :351). Three hand-renamed `.csv.stub` files show this has happened.
- The ESTBAN per-branch (AG) files (1.0 GB) have no reader.
- Financial-conglomerate Values (3.0 GB) are read only by the aggregator.
- It has four hand-written retry loops.
- It parses 80-230 MB files with pandas only to test whether they are empty (:215).

**`scrape_estban_concat` (261 lines, step 2).** Keep. It has 10 readers.
- It has the case-duplicated glob pair, but `sorted(set(...))` (:137-140) means no double read. The double read is only in `panel_deposits.py:624-626`.
- It imports `panel_deposits` (`process_raw_estban_csv`, `coerce_cnpj`), so a scrape script depends on a panel script.
- It rebuilds in full on every run, about 1.5 min.
- It rewrites in place, non-atomically, a file that 10 scripts read.

**`scrape_ifdata_aggregate` (373 lines, step 3).** Keep, and slim it.
- Only the type-1 reports 1-5 are read [checked]: `panel_deposits.py:796` reads report 3 and `panel_bank_chars.py:123` reads reports 1-5. Types 2 and 3 and type-1 reports 6-14 (8.9 GB, 83% of the output) have no reader.
- A run takes about 70 min. The hot spot is a per-row string normalisation over about 50M rows (:151-152).
- Each rebuild leaves an 8-10 GB backup.

**`scrape_cosif_download` (153 lines, step 11).** Keep.
- It has no retry. A failed listing silently skips a type and the script still exits 0.
- Its default types leave out BLOPRUDENCIAL, which step 15 reads (`panel_ip_rates.py:118`) [checked]. The files on disk run to 202603, from an `--all-types` run on 06-02.

**`scrape_cosif_service_fees` (390 lines, manual).** Keep. Its readers are `panel_fee_merge`, `panel_audit_fee_coverage` and `panel_diag_fee_reversal`.
- **Months stored twice are counted twice** [checked]. 202508BANCOS, 202509BANCOS and 202302SOCIEDADES exist both as a plain CSV and as `.csv.zip`. `_gather` (:297-303) takes both, and `_build_panel` sums them (:206-208).
- Quarterly fee ratios are unaffected: the increments telescope within the quarter, the ratio uses the quarter-end stock, and `panel_fee_merge` excludes 2025. Only the three monthly rows are wrong.
- `--test` writes to the production filenames.
- Half-year de-accumulation is coded in six places.

**`scrape_bcb_pilar3_liq2` (200 lines, manual, 10-13).** Keep: `switch_cost_tools.py:150` (`_withdrawal_bound`) reads it.
- 66 JSON files under an old naming scheme are orphans.
- The header comments are out of date.

**`scrape_forward_rf` (718 lines, manual).** Keep as is.
- Its readers are the BBL/CF Julia code and the cluster scripts. The command line is hard-coded in at least 12 files.
- It builds its own paths instead of `paths.cost_fwd_dir()`, so it ignores CF_COST_FWD and SLEEP_OUT_ROOT.
- The Focus vintages are not cached.

### B. Market-characteristic panels (steps 4-9 and 38), INSS, IBGE regions

**`scrape_anatel_mobile` (912 lines, step 6).** Keep, and fix it: **the "municipality" file is record-level** [checked].
- `aggregate_to_mca_quarter` (:433-437) saves its rows before the groupby. So `anatel_muni_panel.csv` (1.59 GB) has one row per raw ANATEL record, e.g. `acessos` = 1.0.
- `panel_demographics_sigma.compute_anatel_sigma` (:189-214) computes `connections_per100` and `frac_4g5g` per record and takes the dispersion across records. The Pix version aggregates to municipality first (:166-167).
- The stored values show it:
  - `frac_4g5g_sigma` runs from 0.47 to 0.50 (p10 to p90), the spread of a 0/1 variable;
  - `connections_per100_sigma` has a median of 2.06, for a variable whose median is 103.
- Nearly every market is one municipality (5,476 markets out of 5,565 municipalities), so the other variables mostly get a fallback sigma. One value covers 48.8% of rows for Pix and for GDP. The two ANATEL columns get record-level noise instead.
- `blp_draws.jl:50` draws both with these sigmas, and `blp_engine_cpu.jl:304-343` interacts `connections_per100` with the random coefficients. **This can move the RC-BLP estimates.** The size is unknown until the draws are rebuilt and the model re-run on the cluster.
- Other items:
  - the early exit has no force flag;
  - files are read as `dtype=str` with no `usecols`;
  - the header describes per-year files that the code does not use.

**`scrape_ibge_demographics` (1,476 lines, step 4).** Keep.
- About 480 lines are a legacy path that runs only when `SCRAPE4_FULL_REGIONS=1`, which nothing sets. `fetch_sidra_classified` is never called.
- The early exit has no force flag.
- GDP for 2024-25 is carried forward and never refreshed.
- The crosswalk filename says 2024 but the data run to 2025. Sixteen files hard-code that name.

**`scrape_ibge_region_hierarchy` (134 lines, manual).**
- It fills region columns in the crosswalk that no code reads.
- A crosswalk rebuild reverts them, and nothing re-runs this script.
- It has no venv guard.
- Fold it into the crosswalk build, or retire it.

**`scrape_pix_municipal` (479 lines, step 5).** Keep.
- Its download loop is a near-copy of the one in `scrape_pix_participants`.
- TLS verification is off, as it is in `pix_participants` and `pix_roster` too.

**`scrape_bcb_inclusion` (281 lines, step 7).** Keep.
- `correspondents` is 0 in every row and `access_points` equals `branches` in every row. This is known and documented in the notes. Only `branches_per1000` reaches the estimation.
- The header still describes correspondents.
- It reads the 349 MB `ESTBAN.csv` as strings with all 55 columns, to use 4 of them.

**`scrape_bcb_banked` (520 lines, step 8).** Keep. `bcb_banked_mca_panel.csv` is a cluster input for the sleepiness preps.
- Its Findex fetch duplicates `scrape_worldbank_findex`.
- It is a second raw-ESTBAN reader.

**`scrape_cadunico` (399 lines, step 9).** Keep.
- The docstring describes an API the code does not call.
- SAGI data after 2025 is never fetched.

**`scrape_bcb_access_points` (228 lines, step 38).** Keep.
- `bcb_access_points_by_inst_muni.csv` has no reader.
- The pipeline never refreshes the cache.
- Paging uses `$skip` with no `$orderby`.

**`scrape_inss` (482 lines, manual). Retire candidate.**
- It was never built: its outputs are absent, and its base path resolves to `Code/` instead of the data root (:83).
- `panel_market.py:626` says nothing consumes it.
- Two orphan zips (1.69 GB) sit in `INSS/raw`.

**Duplicated helpers in this family.**
- `load_mca_crosswalk` has 6 identical copies plus inline ones.
- The per-capita merge has 6 copies.
- The HTTP retry has 6 copies.
- The pyarrow CSV writer block appears 7 times.

Together these come to about 155 lines that one `utils` module would replace.

### C. Fees and Open Finance

**`scrape_bcb_tarifas` (820 lines, step 10). Step 10 is a permanent no-op** [checked].
- It returns early whenever `tarifas_fee_summary.csv` exists (:690-699).
- The files on disk come from two test runs: 892 rows, collected on 02-26 and 03-09 [checked].
- The only reader is Fig 6 of `panel_diag_fee_reversal`.

**`scrape_bcb_tariff_vigencia` (300 lines, manual).**
- It calls the same Olinda API and entity sets as `scrape_bcb_tarifas`.
- Its "history" is a single snapshot, overwritten on every run.
- `panel_fee_merge` reads it as the sole source of `listed_fee_*`.
- `--test 20` overwrites production.

**Fee codes are mislabelled** [checked].
- `panel_fee_merge.py:92-99` maps code 1501 to "atm_withdrawal" and 1203 to "cheque_stop".
- BCB's own names: 1501 is "PACOTE PADRONIZADO DE SERVIÇOS I" and 1203 is "Exclusão do CCF". The withdrawals are 1209 and 1210.
- So `listed_fee_atm_withdrawal_pf` holds a package price.
- No estimator reads the fee columns (`panel_fee_merge.py:41-43`) [checked]. No result is affected.

**`scrape_openfinance_fees` (700), `_freshness` (205) and `_monthly` (108).** Together they are one stand-alone OFB pipeline.
- **The weekly task is dead** [checked]. `OFB_fees_weekly` is Disabled and points at `run_openfinance_monthly.py`, which no longer exists. Its last run, on 08-24, returned 0x80070002. The panel has 4 snapshots, the last on 07-22.
- These defects were not checked by hand:
  - there is no pagination, so rows past page 1 are never collected (24 of 347 cached payloads have more than one page);
  - the 30-day cache stamps old payloads with today's date;
  - `DEDUP_KEY` drops 14% of the 07-22 rows.

**`scrape_openfinance_fees_wayback` (336 lines).** Its verdict, "not viable", is recorded (script :19-23; `service_fees_notes.md:73-75`). It has no reader.

**`scrape_cvm_fee_income` (302 lines).** Its reader is Fig 11 of `panel_diag_fee_reversal`. Its `download_zip` is a verbatim copy of the one in `scrape_cvm_deposits`.

### D. Deposit-rate web scrape: 7 scripts, 2,324 lines, a 6.9 GB archive

This covers `scrape_deposit_rate_{targets,cdx,fetch,parse,format,diagnose}` and `scrape_drate_pipeline`. **No live reader** [checked].
- The only reader of their output is `panel_deposit_rates.build_panel` (:1544-1587).
- `build_panel` is reachable only from `__egan_aux_main__` (:1683), which nothing calls. `__main__` (:2047-2050) runs `append_rates_to_panel`.
- The last run was on 05-25.
- If it were ever revived, these must be fixed first:
  - a 100x unit mismatch;
  - SELIC-indexed rows multiplied by the CDI;
  - a fallback definition older than the corrected type-4 rate;
  - missing stage-6 inputs.
- Aggregator sites are 33% of the archive and produce zero output rows.

**Coupling that must be handled before deleting anything** [checked].
- The same dead function is the only writer of `PANEL_INTERMED/quarterly_macro_rates.csv` (:1767).
- Readers of that file:
  - pipeline step 13 (`panel_cosif_calibrate.py:55`);
  - `make_desc_compressed_tables.py:75`;
  - `make_slide_figures.py:44`;
  - two diagnostics.
- The file was last written on 2026-03-03 and ends at 2025Q4. That is fine for the 2016-2024 window, but it cannot be regenerated.

### E. Pix, probes, complaints

**The three probes** (`scrape_ccs_probe` 124, `scrape_pix_claims_probe` 145, `scrape_salary_portability_probe` 114).
- All three are answered and recorded in `Drafts/Deposit Competition/identification_notes.md:1181-1190` [checked].
- They have no readers.
- The CCS and salary probes cannot return NOT-FOUND by construction.

**`scrape_pix_participants` (276) and `scrape_pix_roster` (461).**
- The only reader is `make_pix_timeline.py`, whose report target no longer exists.
- In the roster script, one failed CDX regime silently writes a truncated history over all three outputs (:117-121, :426-427, :433-442).
- The roster cache has a gap from 2023-03 to 2024-11.

**`scrape_bcb_complaints` (501 lines).**
- It has no reader, and 2022Q2 is missing.
- It imports `panel_deposits.build_cnpj_conglomerate_map` (:394) [checked], against the rule of not borrowing helpers from `panel_*`. Its own venv guard runs first, so this is a coupling, not an active hazard.

### F. Disclosures (pipeline stage 6), deposit maturity, FGC

**Stage 6 feeds nothing in the paper** [checked]. Stage 6 is edgar, parent, incumbent, bcb_accounts, findex, fgc_statistics, cvm_deposits and `panel_disclosure_join`.
- Its only readers are `make_account_vs_volume_figure.py` and `make_margin_figures.py`.
- Neither those figures nor `app:extensions:intensive` appears in any `.tex` under Drafts.
- Every full pipeline run still re-scrapes:
  - about 1,750 SEC requests, about 1 h;
  - about 660 PDFs, downloaded before the cache is checked;
  - paid vision calls.

**Stage 6 can delete stored rows** [mechanism checked].
- `scrape_incumbent_clients.py:292` decides on the vision fallback from the shell environment alone.
- `.env` is loaded only inside `utils/vision_extractor.py`, which is imported only inside the fallback (:153).
- `write_panel` (`utils/disclosure_common.py:284-313`) replaces every row of each firm present in the batch.
- So if the key is not exported in the shell, a run drops the vision-derived rows of every firm that still has a text-derived row (53 of the 99 rows are vision-derived).
- The same mechanism hits edgar under `--deposits-only` or when the companyfacts call fails.

**`scrape_bcb_accounts` (139) and `scrape_fgc_statistics` (134).**
- Both only seed templates, which were never filled, and both have no readers.
- `scrape_fgc_statistics` finds 0 files by construction and is superseded by `scrape_fgc_censo`.

**`scrape_fgc_censo` (125), `scrape_deposit_maturity_cvm` (1,151) and `scrape_deposit_maturity_sec` (551).** These are live for the switch-cost work, through `deposit_maturity_summary.py` and then `switch_cost_tools.py`.
- The CVM script uses a 360-day year (:164) and the SEC script a 365-day year (:96) [checked]. So `share_le360` and `share_gt360` differ between the two sources by construction. `switch_cost_tools` uses only the 90-day cut, so it is unaffected.
- `ipe_cache.setdefault` re-parses the index on every iteration (:1005).

**CVM and SEC code is duplicated.**
- Six files carry their own CVM download and index logic, over seven separate caches.
- There are three SEC fetchers.
- `utils/disclosure_common.py` offers only HTTP.

### G. Advertising: 9 scripts, 17,100 lines

**Main measure:** `scrape_cosif_advertising` (467 lines) plus `build_cosif_lai_advertising`. Keep.

**Hard inputs that the measure never uses.** These are all hard inputs of `build_advertising_panel.py` (`needed`, :851-856), but none is the source chosen for any of the 270 sample conglomerates:

| Script | Lines | Note |
|---|---|---|
| `scrape_cvm_advertising` | 468 | |
| `scrape_caixa_advertising` | 2,162 | |
| `scrape_caixa_scan_advertising` | 1,208 | |
| `scrape_statebank_advertising` | 3,328 | |
| `scrape_sponsorship_bb_bnb` | 1,043 | |
| `scrape_sec_advertising` | 345 | annual |
| `scrape_ancine_ad_films` | 1,999 | counts, not spend |

No estimation code reads `advertising_panel_*`; only two diagnostics do.

**`scrape_bank_notes_advertising` (3,661) and `utils/note_sources.py` (164).**
- They have been a cross-check only since 10-05, and the panel does not read them.
- Check 11 of `build_cosif_lai_advertising.py:393-441` reads their periods parquet [checked]. It asserts that Santander's note is within 3% of the file, and only warns if the parquet is absent.
- A plain run cannot finish: Volkswagen fails the totals floor.

**Defects:**
- **Statebank cache.** `Cache.get` (:140-156) replaces a good cache entry with a failure under `--refresh`.
- **`--banks X` overwrites** all four combined outputs.
- **Wrong run order in the register.** In §10, the crosswalk must come before the statebank scraper.
- **Caixa OCR is not cached**, which costs 20-45 min per full rerun.
- **Shared CVM cache.** `scrape_cvm_advertising --refresh` rewrites the CVM cache that `fee_income` and `deposits` also read.

**Library code living in entry scripts.**
- `build_cosif_lai_advertising` imports `scrape_cosif_advertising` for its flow functions.
- `diag_ocr_benchmark` imports `brl` from the statebank scraper.
- `scrape_ancine_ad_films` imports the crosswalk.

## 2. Reduce: merges

| Merge | Scripts | Label |
|---|---|---|
| OFB fetch + freshness + weekly into one module with subcommands; archive the Wayback probe | 4 to 1 | NEEDS-CHECK: the scheduled task must be re-registered |
| `scrape_bcb_tarifas` + `scrape_bcb_tariff_vigencia` into one fetch (same API) | 2 to 1 | NEEDS-CHECK: step 10, the `--test` paths, SKILL.md |
| `scrape_pix_participants` + `scrape_pix_roster` (keys and roster phases) | 2 to 1 | SAFE |
| `scrape_fgc_statistics` retired in favour of `scrape_fgc_censo` | 2 to 1 | SAFE |
| Caixa main + scan into one entry point (`--scans`); the code stays separate by design | 2 to 1 | SAFE |
| Deposit-rate family into one module, only if it is kept | 7 to 1 | SAFE |
| `scrape_cvm_deposits` + `scrape_cvm_fee_income` (same zips, same parse skeleton) | 2 to 1 | NEEDS-CHECK: step 34 |

## 3. Optimize

### Shared helpers

Each new module is listed with the scripts that would call it.

- **`utils/bcb_http.py`:** a retry session, Olinda OData paging, the CMS document listing and an SGS range fetch.
  - Callers: `estban_ifdata`, `cosif_download`, `tarifas`, `tariff_vigencia`, `access_points`, `pix_*`, `pilar3`, `forward_rf`, `complaints`.
  - About -190 lines. Byte-identical files: SAFE. Making failures exit non-zero: NEEDS-CHECK.
- **`utils/cvm.py`:** the DFP/ITR/IPE zip cache with a refresh policy, a member reader and the RAD fetch.
  - Callers: `cvm_deposits`, `cvm_fee_income`, `cvm_advertising`, `bank_notes`, `deposit_maturity_cvm`, `incumbent_clients`.
  - Frees about 290 MB of duplicate index files.
- **`utils/sec.py`:** submissions, the filing index and documents, with a cache.
  - Callers: `edgar`, `parent`, `deposit_maturity_sec`, `sec_advertising`.
- **`utils/cosif_io.py`:** month-level de-duplication by vintage (this fixes the double count), `read_balancete` and half-year de-accumulation, which today has 6 copies.
- **`utils/text_br.py`:** accent folding (about 10 copies), month names (4 copies) and Brazilian number parsing.
  - There are 7 number parsers with different rules. Keep the variants explicit, or the parsed cells change.
- **Other moves:**
  - Move `process_raw_estban_csv` and `build_cnpj_conglomerate_map` out of `panel_deposits.py` into `utils/`.
  - Move the COSIF flow functions from `scrape_cosif_advertising` into `utils/cosif_flows.py`.
  - The CNPJ-to-conglomerate map is built in 5 places. `panel_fee_merge.build_cnpj_cong_map` holds the fix for the latest-snapshot defect, so start from that one.
- **`utils/wayback.py` is not worth it.** If the deposit-rate family goes, only two callers are left.
- **The TOON layer does nothing** [checked].
  - It is wired into 10 scrapers, 5 `panel_*` scripts and `panel_pipeline.py`.
  - `utils/gemini_*.html` contain only `<div></div>`, and `utils/toon_context.json` has no per-script section, so every lookup returns its default.

### Slow steps

| Step | Cost now | Fix |
|---|---|---|
| `scrape_ifdata_aggregate` | about 70 min, plus an 8-10 GB backup | normalise unique values, merge per period, stream the writes; optionally build only type 1, reports 1-5 |
| Stage 6 on every full run | about 1 h of SEC requests, ~660 PDFs, paid vision calls | make it opt-in |
| `scrape_bank_notes_advertising --offline` | about 35 min (no parse cache); `--refresh` re-downloads 1.3 GiB of immutable documents | moot if frozen |
| Caixa OCR | 20-45 min per rerun | OCR cache keyed by sha256, page and dpi |
| `scrape_bcb_tarifas` / `scrape_bcb_tariff_vigencia` | about 40 min each, same API | one fetch |
| `scrape_incumbent_clients` | re-downloads every PDF before checking the cache (:211); parses the IPE index 54 times instead of 9 | check the cache first, hoist the parse |
| `scrape_ancine_ad_films` | re-hashes 314 MB and runs a regex per row on every run | 1-3 min; low priority |
| `scrape_bcb_estban_ifdata` | downloads the unread AG files (half of every ESTBAN refresh); parses whole files to test emptiness | drop AG; size check |

## 4. Defects found along the way, by consequence

0. **ANATEL sigmas in the RC-BLP draws come from raw records**, not from municipalities. This can move the RC-BLP estimates by an amount not yet known. [checked; family B]

Nothing below changes a number in the paper today.

1. **No live writer for the macro rates.** `quarterly_macro_rates.csv` feeds step 13, but its only writer is dead code. It cannot be regenerated, and extending the window past 2025 is blocked. [checked]
2. **Stage 6 can delete client-count rows.** It runs by default and can drop stored rows (the vision gate plus `write_panel` replacing whole firms). [checked mechanism]
3. **The OFB weekly task is dead**, and the OFB scraper has pagination and snapshot-labelling defects. [task checked]
4. **Step 10 is a no-op over test data.** [checked]
5. **Fee codes 1501 and 1203 are mislabelled.** No estimator uses those columns. [checked]
6. **The service-fee panel double-counts three month files.** Quarterly values are unaffected. [checked]
7. **Failures are cached as data.** `scrape_bcb_estban_ifdata` caches failure stubs permanently and exits 0; `scrape_cosif_download` exits 0 on failures.
8. **Silent truncation.** A failed CDX regime makes `scrape_pix_roster` write a truncated history over its outputs.
9. **Deposit maturity uses two year lengths** (360 vs 365 days). [checked]
10. **Inflated volume shares.** The stage-6 join sums a repeated conglomerate total, so its volume shares are inflated. This is known from the 10-05 audit.
11. **Cache corruption under `--refresh`.** The statebank `Cache.get` replaces good cache entries with failures.
12. **`--test` writes to production** in `scrape_bcb_tariff_vigencia`, `scrape_cosif_service_fees` and `scrape_openfinance_fees`.

## 5. Stale text, all comment-only

- **`panel_pipeline.py`:**
  - :91-94 still uses numbered names;
  - the ID lists at :154-161 and :597-598 run [26]-[33], while STEPS uses 28-35;
  - `--to` is missing from the usage text.
- **`SKILL.md:47`** says the scrapers re-download only the trailing year. That is false for stage 6.
- **Old numbered names** (`scrape_N`, `panel_N`, `d_rate`, `run_openfinance_monthly`, `ESTBAN_Process_1.R`, `if_data_process_1.py`) remain in most scrapers, in `make_pix_timeline.py`, in `panel_fee_merge.py:82,316`, in `panel_openfinance_fees.py`, and in `Drafts/Deposit Competition/service_fees_notes.md`.
- **Out-of-date headers:**
  - PILAR3 (:26-34);
  - `caixa_scan` :8-18;
  - the notes docstring :6-16;
  - the crosswalk :12;
  - `build_advertising_panel.py:64`, which quotes BRB at 96.6% where the register says 94.2%.

## 6. Data with no code reader (OneDrive)

| Data | Size |
|---|---|
| IF.data aggregate types 2 and 3, type-1 reports 6-14 | 8.9 GB |
| (plus the backup each rebuild leaves) | 8-10 GB |
| Deposit-rate archive `processed/IP_SCRAPE` | 6.9 GB |
| Financial-conglomerate Values (only the aggregator reads them) | 3.0 GB |
| SEC raw filings | 2.0 GB |
| Incumbent PDFs | 1.5 GB |
| Bank-notes cache | 1.4 GiB |
| ESTBAN AG files | 1.0 GB |
| Duplicate IPE index | 172 MB |
| INSS raw zips (`scrape_inss` never built) | 1.69 GB |
| CSV twins of the advertising parquets | 106 MB |
| `tariff_vigencia_summary.csv`, the `tarifas_*` roll-ups, `cosif_service_fees_conglomerate.*`, `complaints_conglomerate_quarterly.csv`, `pix_roster_entry.csv`, 66 old-name PILAR3 JSONs, three IF.data `.csv.stub` files, three COSIF plain-CSV duplicates | small |

"No code reader" does not mean safe to delete. Some of these are raw archives kept for reproducibility.


## 7. Outcome, 2026-10-14 (the user's decisions, and what was done)

| Item | Decision | Done |
|---|---|---|
| ANATEL sigma | fix | `panel_demographics_sigma.compute_anatel_sigma` sums records to municipality x quarter first; `scrape_anatel_mobile` now writes municipality rows. `demographics_sigma.parquet` rebuilt: only the two ANATEL sigma columns changed (`connections_per100_sigma` median 2.06 -> 11.79, `frac_4g5g_sigma` 0.49 -> 0.013), all 15 other columns identical. Cluster: upload + `blp_run.sh --draws` + RC re-estimate still to run (user's timing). |
| 15 retire candidates | delete outright | Deleted by `claude/cleanup_2026-10-14.ps1` (162 targets, 24.3 GB). Kept, per "leave advertising alone": `scrape_bank_notes_advertising.py` and `utils/note_sources.py`. |
| Deposit-rate dead code | (follows) | The macro writer moved into `append_rates_to_panel` (byte-identical output); 13 unreachable functions and 5 dead constants removed (2,051 -> 933 lines). |
| Stage 6 | freeze, move, opt-in | 8 scripts in `extensive_margin/`; `panel_pipeline.py --extensive-margin` (or `--only 6`); steps 31 and 33 removed with their scripts. |
| Advertising | not now | untouched |
| OFB | delete | scripts, data, caches and the `OFB_fees_weekly` task removed |
| Tariffs | freeze, move to `fees/` | 2 scripts in `fees/`; step 10 removed from the pipeline |
| Shared helpers | yes, except CVM/SEC (user: do not touch) | `utils/bcb_http.py` (retry session + CMS listing; used by `scrape_cosif_download`, `scrape_bcb_estban_ifdata`) and `utils/mca.py` (crosswalk, population, per-1000/per-100; 5 scrapers, old-vs-new frames identical). The COSIF de-accumulation was not consolidated: its six copies sit in pipeline steps and advertising code. |
| Unread data | delete | in the same PowerShell run; the producers no longer create it (aggregator builds type-1 reports 1-5 only, `--all` for the rest; step 1 fetches IF-Data types 1 and 3, no per-branch ESTBAN) |
| Small fixes | yes | service-fee dedup (5 monthly cells moved, quarterly ratios unchanged; panel regenerated); no failure stubs and exit 1 on failure (steps 1 and 11); `--force` on the three early-exit scrapers; TOON removed everywhere; stale names in `panel_pipeline.py` |

Open: the BLOPRUDENCIAL files that step 15 reads are fetched only with `scrape_cosif_download.py --all-types` (on disk to 2026-03); the RC-BLP rerun.
