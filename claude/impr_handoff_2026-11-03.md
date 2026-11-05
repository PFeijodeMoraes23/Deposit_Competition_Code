# Handoff: the coding (IMPR) thread, 2026-11-03 (replaces `coding_handoff_2026-11-02.md`)

**To:** the coding agent, now the IMPR session. **From:** main thread, for Pedro. **Written:** 2026-11-03, 17:05.

**Answers** `IMPR_REPLY_2026-11-03.md` (15:57) and **replaces** both `coding_handoff_2026-11-02.md` (the former master, whose updates, settled facts and tasks are kept below in full) and `impr_handoff_2026-11-02.md` (removed by Pedro 11-03; its FR plan is restated in the next section). Reply in `claude/IMPR_REPLY_2026-11-03b.md`, timestamped. All paper text lives in one plan, `Drafts/Deposit Competition/paper_edits_V_Friction_2026-11-02.tex`.

## The FR plan, restated (from `IMPR_REPLY_2026-11-03.md`), and what is decided

```
fr[6]{step,state,decision}:
  FR0,"DONE 11-02 20:24: M3b and the E4 chain cancelled; zip 31716854 archived in CLUSTER_IN/20261102/, not ingested","none"
  FR1,"DONE 11-03: 2016Q1 enters the sleep frame (lags before the year cut; funding ratios spliced to 2015Q4 in panel_demand_chars.py, utils/demand_chars.py, sleep_est_e2.build_pooled_data); B2 one-quarter-set assertion; B3 real per-capita GDP growth, t-1 timing, mca_gdp_real_growth.csv; B4 re-centring; B5 per-routine BLP status. E3 demand parquet verified identical","accepted by the main thread; Pedro's OK of ~14:45 stands"
  FR2,"DONE 11-03: local sizing on the new frame (B = 9, scratch). Path tilts (2016-17 lower, 2023-24 higher); level averages unchanged; sign flips only in AMEs R3 found insignificant; E4 saturation at phi-hat = 1 doubles to 33.4%","accepted; the E4 saturation goes into PC1's disclosure"
  FR3,"now I2: the full cluster chain, every routine, with entry and the new centring file; then demand prep and logit. Package of 16:32 approved in content; sleep_demand_prep_link.py changes again with I1","WAIT: I1 first, then Pedro's go"
  FR4,"data/output/blp aside, then the RC ladder and the six-start ext1 multi-start on the new frame, with E3's archived ext1/ext2 among the starts (new EXT1_TAG)","after FR3; the adoption thresholds T1 to T3 of Update 11-02 apply unchanged; report T1 to T3 for every candidate per routine and the old against new endpoints side by side"
  FR5,"keys against R3, then M5 (SEs with the exact Jacobian, the L1 items, blp_status ok per routine, tables and figures rebuilt, every changed key with sign and significance crossings)","after FR4 adoption"
```

**One question for the IMPR session before FR3.** FR2 says the four changes (2016Q1, `demand_chars` in the sleep frame, real GDP growth, re-centring) are cumulative and not decomposed. The main thread does not need the decomposition for the paper, but the paper will state that the window is 36 quarters and that E4 uses real per-capita growth with t−1 timing; confirm that nothing in FR1 changed a number the demand side already reports (the E3 parquet was verified identical; say the same for E4 once its parquet is rebuilt).

## Update 11-04, 20:25: reply (e) accepted; nothing to run tonight; the imputed-tail rule is Pedro's call on 11-05

Your reply (e) (`IMPR_REPLY_2026-11-04e.md`, removed after processing) is accepted: Q1 (the frames' line relayed to the Kappa agent), I14 (`0b9385b5`, ten Itaú and Banrisul quarters re-carried, the 2016 and 2020 tail weight down to 0.66 and 1.39 percent), the restaged I2 package (248 files), and I15. I15 is the finding of the evening: the weighted tails are all calibrate's strip, the excess is a level of 4.5 to 11 pp a year above the CDI that no imputed component removes, concentrated at foreign and wholesale banks and some mid-size fixed-rate issuers, and the 0.5 to 1.5 ratio bites in 2018 to 2021 because the CDI was low, not because the excess grew. Pedro decides on 11-05 between the ratio rule (a) and a band in percentage points (b); the I2 launch follows. Until then nothing runs. One thing to prepare, read-only: for rule (b), the counts and type-4 weight by year at bands of +3, +4 and +5 pp above the CDI (and a symmetric or a wider band below), so the decision has numbers; post them in `IMPR_REPLY_2026-11-04f.md` when convenient, no rush. The Pix agent will rerun P3 to P5 on `0b9385b5` (its own handoff).

## Update 11-04, 19:00: reply (d) accepted; `52ea6495` pinned for tonight's local work; I2 waits for Pedro's decision on the Itaú and imputed tails

Your reply (d) (`IMPR_REPLY_2026-11-04d.md`, removed after processing) is accepted: Q4 in two passes, Q3, Q2 (new means recorded beside the 11-03 ones; the local R3 pickles are not to be re-evaluated on the new frame), I12 (455 conglomerate-quarters changed, carried tails gone, 157 tails left, 153 of them imputed and 4 Itaú), the restaged I2 package (sha256 recorded), Q1 running. `52ea6495` is the md5 the Kappa agent's step 7 and the K1 inputs pin tonight; post the interim frames' line here when they exist.

Two items you flagged go to Pedro and are decided below this line when he answers: (i) Itaú's level (2016 H1 2.35, 2018 H1 1.96, 2020 H1 3.61 times the CDI after the reshaping; the IFRS line that mixes subordinated and external debt, B8); (ii) the 153 imputed tails, where your one rule (a measured `imputed` or `benchmarked` rate outside 0.5 to 1.5 times the CDI is unmeasured and goes to the carry) reclassifies 520 of 1,813 imputed B quarters. Decided (Pedro, 19:05): **I2 holds.** Two tasks, in this order, after Q1's interim fits end (do not interrupt them):

**I14: the benchmarked tails now.** Apply your rule to `benchmarked` rates only: a benchmarked semester whose rate lies outside 0.5 to 1.5 times the CDI is unmeasured and goes to the carry (with the I12(b) donor range), no donor meaning missing. Rerun from calibrate as for I12, post rows changed by bank, the type-4 weight by source before and after, the tails left, and the new md5 as the end line; it replaces `52ea6495` for the I2 package (restage) and for the Pix reruns (the main thread reopens the Pix thread then). The Kappa agent's step 7 and the K1 inputs keep `52ea6495`; the difference is a handful of Itaú and Banrisul semesters, which step 7 does not test.

**I15: the imputed tails, diagnostic only, no change.** For the 520 imputed B quarters outside 0.5 to 1.5 times the CDI: which banks, how many quarters each, their type-4 weight by year, the rate the imputation produced against the CDI, and what in the imputation makes them (a donor group with few members, a ratio carried across years, a stock-weighted mean dominated by one bank). Pedro decides on the rule tomorrow, 11-05, after his advisor meeting; the I2 launch follows that decision.

Reply in `IMPR_REPLY_2026-11-04e.md` (the frames' line can go there or in a one-line note here).

**IMPR Q1 interim frames: ready 11-04 19:41**, on panel `52ea6495` with the new centering (`f74a37d3`). Folder (set `DEMAND_PREP_DIR` to it, or read by path): `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\rebuild_1104\interim_q1b\demand_prep\`; files `demand_1_spec_12.parquet` (620,701 rows), `demand_2_spec_12.parquet` (619,595), `demand_3_index_spec_12.parquet` (610,459), `demand_4_index_time_spec_12.parquet` (616,779); each records `demand_chars.market_panel_md5` = `52ea6495c9195ef38b83bc40fb2014fa` in its schema metadata. The fits (spec 12, point estimates, 36 quarters) are in `..\interim_q1b\est1` to `est4`; national φ̄ against R3 (35 quarters): E3 0.9904 vs 0.9854, E4 0.9881 vs 0.9845. Production fits and frames untouched. I14 (rerun from calibrate) starts now; it does not touch these frames, and step 7 and K1 keep `52ea6495`.

**IMPR I14 rerun: started 11-04 19:46** (same command as I12; calibrate LF md5 `c1d22980`: a benchmarked rate outside `BENCH_RANGE` = 0.5–1.5 × CDI is `rate_flag` 4; `panel_deposit_rates.py` LF md5 `db921580`: flag 4 → fallback code 7, to the carry with the I12(b) donor range). `processed/market_panel.parquet` and `demand_chars.parquet` are rewritten in place; the Kappa agent's 19:16 snapshot of `52ea6495` (`...\904ee55d-...\scratchpad\panel_52ea6495\`) is what step 7 and K1 read. End line below.

**IMPR I14 rerun: ended 11-04 20:02** (every step exit 0, 16.4 min; audits and report read by 20:06). **Panel md5: `market_panel.parquet` `0b9385b5410b5cb6ae4c884c32e4ef46`** (for the I2 package and the Pix reruns; it replaces `52ea6495` there, while step 7 and K1 keep `52ea6495`); `market_panel.csv` `e06da8eaf6cf503eb41f03d7fda09434`; `demand_chars.parquet` rebuilt from it. Rows, firms, markets, `is_B`, every deposit sum unchanged; audits pass; demand-chars check 29/29. **10 conglomerate-quarters changed**, all benchmarked → carried (fallback code 7): Itaú 2016Q1–Q4, 2018Q1–Q2, 2020Q1–Q2 (now 0.82–1.12 × CDI, from 1.96–3.61), Banrisul 2020Q1–Q2 (0.96, 1.10, from 2.06); 2.24% of type-4 weight. Type-4 weight by source: statement 46.87%, benchmarked 20.90% (from 23.15), imputed 26.37%, carried 5.56% (from 3.31), none 0.30%. Tails beyond ±10 pp (Pix base): **153**, all imputed (from 157); their weight in B type-4 deposits 2016 0.66% (from 8.33), 2020 1.39% (from 6.33), other years unchanged. I2 restage and I15 follow; details in `IMPR_REPLY_2026-11-04e.md`.

## Update 11-04, 17:55: I12, national type-4 spreads beyond ±10 pp on `45dbfc58`; before the I2 restage

The Pix agent's 17:25 reruns on `45dbfc58` are in (P3 to P5 hold; thread closed). Its flag, which Pedro (17:50) turns into a task that goes before the interim fits (Q1) and the I2 restage:

**I12: the ±10 pp tails of the national type-4 spread.** 185 of 2,277 B conglomerate-quarters in 2016 to 2024 have `spread_ann_a4_national` outside ±10 pp (153 `imputed`, 27 `carried`, 5 `benchmarked`; 125 negative). They hold 9 percent of type-4 weight in 2016, about 7 in 2018 and 2020, nearly none from 2021, and sit in 6.4 percent of P3's multi-bank market-quarters at 22 percent of the cell's weight. Two cases drive them:

```
i12[2]{bank,source,symptom,likely cause}:
  Itaú,"benchmarked (no time-deposit line)","28.7, 43.5, 28.2, 20.9 percent a year through 2016; 20.1 in 2018Q2, 20.3 in 2020Q1; near the Selic between","the strip × k benchmark shapes the quarters wrongly: the semester accounts are not split evenly over their quarters; the annual level may be right"
  Safra,"carried 2016Q1 to 2017Q1, 2018Q1, 2020Q1; imputed between","24 to 66 percent a year carried; up to 48.5 imputed; above 17 until 2019Q3","the carry takes the median CDI ratio of neighbours that are extreme but inside the guard band"
```

Decided: (a) Itaú and any other benchmarked bank: split the semester accounts evenly across their quarters (keep the annual level; drop the strip's quarterly shape for the benchmarked rate); (b) the carry: a neighbour outside a plausible ratio-to-CDI range, 0.5 to 1.5, is not a donor; a bank-quarter left with no donor goes to imputation; (c) rerun from step 16 as in Q4, post rows changed by bank and source, the share of type-4 weight by `rate_a4_src` before and after, the tails left (count beyond ±10 pp, and their weight), and the new panel md5 as the end line, which replaces `45dbfc58` for the Pix and Kappa agents (the Pix agent reruns P3 to P5 once more, 6 minutes; the Kappa agent's step 7 reads the new md5). Then Q3, Q2, Q1 and the restage on the I12 panel. If (a) cannot be done inside the 15-minute chain, say so and do (b) alone first. Reply in `IMPR_REPLY_2026-11-04d.md`.

**For your awareness, not a task:** the tables agent (`tables_handoff_2026-11-04.md`) adds a generator `make_tax_clock_table.py` in the paper folder and a key block `clock.t3.*` through `make_paper_numbers.py`; if I10's productionization touches `make_paper_numbers.py`, keep that block.

**One line for I10's list (from the Kappa agent, 18:22):** `load_demand` in `diagnostics/diag_k0_netflow_objects.py:156` checks that the demand frame is newer than the panel (mtime), not that the frame's metadata md5 equals the panel's; make it the md5. Not now.

**IMPR I12 rerun: started 11-04 18:03** (`panel_pipeline.py --from 2 --skip 5,6,7,8,9,11,12,14,15,17,18,38 --fees`: calibrate, rates, stage 4, step 39, stage 5, fee merge). Pedro's 18:00 answers: (a) a benchmarked bank's semester spread by CDI × lagged ordinary stock (calibrate LF md5 `f8b464ac`); (b) carry donors only within 0.5–1.5 × CDI, no donor → missing (`panel_deposit_rates.py` LF md5 `50413520`); the 153 imputed tails reported, not changed. The interim Q2/Q1 run on `45dbfc58` was stopped at E3 and restarts on the I12 panel. `market_panel` (md5 `45dbfc58`, now superseded) is being rewritten: do not read it until the end line below.

**IMPR I12 rerun: ended 11-04 18:34** (every step exit 0, 31.6 min; audits and report read by 18:41). **Panel md5: `market_panel.parquet` `52ea6495c9195ef38b83bc40fb2014fa`** (set `K0_PANEL_MD5=52ea6495`; it replaces `45dbfc58`); `market_panel.csv` `9dd98472aaf5db46bf88758952e9e006`; `demand_chars.parquet` rebuilt from it. Rows, firms, markets, `is_B`, every deposit sum unchanged. Audits: MCA, firm class, fees pass; demand-chars check 29/29. On the Pix agent's base (2,277 B conglomerate-quarters, 2016–24) the ±10 pp tails go from 185 to **157** (153 imputed, left as decided; 4 Itaú benchmarked; 0 carried, from 27). Changed: 455 conglomerate-quarters with type-4 deposits, 164 of them B (Itaú and Banrisul semesters reshaped, 72; carries re-taken from plausible neighbours, 173; carries left without a rate, 210). Type-4 weight by source: statement 46.87%, benchmarked 23.15% (from 22.46), imputed 26.37%, carried 3.31% (from 4.29). Quarters with type-4 deposits and no rate: 323 (0.30% of type-4 weight), from 113 (0.015%). Details in `IMPR_REPLY_2026-11-04d.md`. This is the md5 to pin. The interim fits and frames (Q2, Q1) start now on it.

## Update 11-04, 16:00: the rebuild accepted; Q1 to Q4 decided; Q4 first, then restage

Your reply (c) (`IMPR_REPLY_2026-11-04c.md`, removed after processing) is accepted: the rebuild, the audits, the I6 recount, the stage-5 files (Pedro recompiles), the demand-prep metadata change, the Bradesco 2016 observation (I10). Reply next in `IMPR_REPLY_2026-11-04d.md`.

```
decided[4]{item,decision}:
  Q4,"NOW, before anything else: with positive type-4 deposits, an uncorrected rate of exactly 0 takes code 4 and an uncorrected rate equal to the Selic to 1e-6 takes a code of its own; both go to the carry. Rerun from step 16 (rates, stage 4, step 39, stage 5); post the new panel md5 here as the end line (it replaces 70241f4e; the Pix and Kappa agents pin the new one); restage the I2 package on it"
  Q1,"(b): re-estimate the four sleepiness fits locally, point estimates only, in a sandbox (SLEEP_OUT_ROOT and DEMAND_PREP_DIR to scratch), on the Q4 panel, and build the interim demand frames with the panel md5 in their metadata; say where they are so the Kappa agent's step 7 and K1 read them; I2's frames replace them when ingested; production files untouched"
  Q2,"REVISED (Pedro, 16:10): re-center on the panel the fits use. After the Q4 rerun, regenerate state_centering_means.json with --write-centers on the final frame, use it for the interim fits (Q1) and put it in the I2 package in place of the 11-03 file; report the new means beside the 11-03 ones"
  Q3,"yes: T1 reads dep_a4_incl_judicial where deposits_panel.csv has it"
```

**Sequence.** Q4 rerun and the new md5; Q3; the new centering file (Q2); the interim fits and frames (Q1); the restaged I2 package with its sha256 and the seven launch lines, unchanged in form; then Pedro launches I2. Announce the Q4 end line here before the interim fits start, so the Pix agent can run its P3 and P4 reruns on the final md5 rather than twice.

**IMPR Q4 rerun: started 11-04 16:21** (`panel_pipeline.py --from 3 --skip 14,15,17,18,38 --fees`: step 16, stage 4, step 39, stage 5, fee merge; `panel_deposit_rates.py` LF md5 `8d2d9517`). `market_panel.csv`/`.parquet` (md5 `70241f4e`, now superseded) are being rewritten: do not read them until the end line below.

**IMPR Q4 rerun, second pass: started 11-04 16:48.** The first pass ended 16:39 (all audits pass, demand-chars check 29/29) but left 8 quarters of C0083632 carried at 0 from donor quarters without type-4 deposits; Pedro (16:45): fix now. An uncorrected rate of exactly 0 is no longer a carry donor (`panel_deposit_rates.py` LF md5 `a3e0324f`). Same command. The first pass's md5 `0b76b4e0` is NOT to be pinned; the end line below follows the second pass.

**IMPR Q4 rerun: ended 11-04 17:08** (every step exit 0, 19.5 min; audits read by 17:12). **Panel md5: `market_panel.parquet` `45dbfc58a5de4d25dd11fe38eee5fdd3`** (set `K0_PANEL_MD5=45dbfc58`; it replaces `70241f4e`); `market_panel.csv` `184582efdebb2e11b566d7986c6ac679`; `demand_chars.parquet` rebuilt from it. Rows, firms, markets, `is_B` and every deposit sum unchanged from `70241f4e`; only the type-4 rate chain moved. Audits: MCA coverage, firm class, fee coverage pass; `panel_demand_chars.py --check` 29/29 (T1 now reads `dep_a4_incl_judicial`, Q3). K0's test (type-4 rate = Selic to 1e-6 or 0, with type-4 deposits, 2016–24): **1** conglomerate-quarter, C0080123 2021Q1, a measured calibrate rate 3.8e-7 from the Selic (not a fill). C0083632's 35 type-4 quarters now have no rate (code 4, no non-zero neighbour to carry from). This is the md5 to pin. The interim demand frames (Q1) start now; their location follows in my next line.

## Update 11-04, 13:45: start the rebuild now, in parallel with the Kappa job; stage 5 runs

Your reply (b) (`IMPR_REPLY_2026-11-04b.md`, removed after processing) is accepted: N4 and N5 as coded (the DFP_NET amounts with pages, DFP_INLINE for the capital-instrument rows, λ 0.983 for Santander), the final smoke, and your correction of the N4 sentence (what is constant within a statement period is rate/CDI; the rate and $g$ still move with the CDI by quarter; your wording goes into the note). Reply next in `IMPR_REPLY_2026-11-04c.md`.

**Pedro (13:45): do not wait for the Kappa finish line.** Its job now ends about 20:00 (per-pair fits take 40 minutes each). Start the rebuild now, in parallel; the Kappa job is one CPU-bound process and the rebuild is I/O and single-threaded steps. Announce start and end here as planned. The one thing that still waits for both is the Kappa agent's step 7 (the 14 GB read), which it starts only after your end line and its own finish line.

**Stage 5 runs** (`panel_pipeline.py --from 2 --skip 5,6,7,8,9,11,38 --fees`, no `--to 4`): the descriptive tables in `Drafts/Deposit Competition` regenerate from the new panel. List the regenerated files in your reply; Pedro recompiles the paper himself.

**IMPR rebuild: started 11-04 14:37** (`panel_pipeline.py --from 2 --skip 5,6,7,8,9,11,38 --fees`; calibrate LF md5 `d870bbb2`; panel before: CSV md5 `e30e202a…`, 500,439 rows). Until the end line, `market_panel.csv`/`.parquet` are being rewritten: do not read them. The end line follows here.

**IMPR rebuild: ended 11-04 15:04** (every step exit 0, 26.4 min; audits read by 15:13). **Panel md5: `market_panel.parquet` `70241f4e31443c581209946d960127cc`** (set `K0_PANEL_MD5=70241f4e`); `market_panel.csv` `844c0d770634d13dedec28762f5bf1ac`; `demand_chars.parquet` rebuilt from it (its recorded md5 matches). 500,439 rows, 537 firms, 3,738 markets, `is_B` split unchanged; `dep_a4` −13.7% (judicial deposits out, I9; `dep_a4_incl_judicial` = the old `dep_a4` exactly); new columns `rate_a4_src`, `rate_a4_fallback`, `rate_a4_carried`, `rate_a4_vintage_src`, `dep_a4_incl_judicial`, `judicial_src`, `judicial_cap_flag`. Audits: MCA coverage, firm class and fee coverage pass; `panel_demand_chars.py --check` 27/29, the two T1 failures being the test still reading `dep_a4` against raw ESTBAN (with `dep_a4_incl_judicial` the identity is exact in every cell: details in `IMPR_REPLY_2026-11-04c.md`). The panel may be read now. The demand frames (φ̂, M̂) are NOT yet rebuilt: steps that read them wait for my next line.

## Update 11-04, 13:10: N4 and N5 decided; the rebuild after the Kappa finish line

Your reply (a) (`IMPR_REPLY_2026-11-04a.md`, removed after processing) is accepted: N1 to N3 as coded, B8's traps in the loader, BRB's special deposits read as negligible (confirmed), the Santander 2024 carry left as the chain does it. Reply next in `IMPR_REPLY_2026-11-04b.md`.

```
decided[2]{item,decision}:
  N4,"(b): within each statement period (the year; the semester for BB) the CDB expense is spread in proportion to CDI x lagged ordinary stock, so rate/CDI is constant within the period for the statement banks; banks without statements keep strip x k. Record in the log which banks and years are on which shape; rate_a4_src stays 'statement'"
  N5,"(b): net footnote (1), the capital-instrument interest (the external and subordinated rows), and footnote (2), the exchange-rate variation with its printed amount, from each year's Santander line and from the 2025 line before lambda (lambda' 0.983); 2019 read as an expense, DFP 2020's label; the footnote-(2) amounts into DFP_NET with pages"
```

Then the rebuild, as soon as the Kappa agent's finish line is in `kappa_handoff_2026-11-03.md`: extract to parquet, announced here at start and end; the demand frames with the panel md5 in their metadata; the E4 confirmation; the md5 table and the restaged I2 package; I2 on Pedro's go. The Pix agent reruns P3 and P4 at your end line (about four minutes), and the Kappa agent's step 7 follows.

**For the design note, one consequence of N4 to state:** for the seven statement banks the time-deposit rate, and so the gap $g$, varies by year (semester for BB) and not by quarter within 2016 to 2024; for every other bank it varies by quarter.

## Update 11-04, 12:20: the BRB fix is in; the BB session is closed and its folders are yours; the rebuild waits only for the Kappa job

`Rout/judicial_deposits_share.csv` was rerun at 12:08 with BRB re-anchored (19 rows changed, all BRB 2018Q1 to 2022Q3; the 2022 step held back to 2022Q1, dated by BRB's 4.1.5 jump of R\$6.6 bn in 2022-03). Read it again before the rebuild. The BB agent has no open task and is closed; its folders and scripts pass to you under I10: `DATA_RAW/issuers/judicial_deposits/` (DFP and ITR extracts, Caixa's statements, the two share scripts), `DATA_RAW/issuers/funding_expense/` (B7 and B8 files), `DATA_RAW/BCB/cosif_plano_contas/`, `DATA_RAW/issuers/sweep_programs/` (the registry), `diagnostics/diag_judicial_deposits_cosif.py`, `diag_judicial_deposits_share.py`, `diag_bb_rende_facil_booking.py`, `diag_santander_sweep_booking.py`. Pedro's rule from here on: as agents stop, you organize what they leave, and I10's map is where it is recorded. The BB handoff file is gone; its record is in `SHARED_MEMORY.md` (12:30), including two temporary scratchpad folders (`...\3686646d-12ca-4ddc-b088-07f872fba1f4\scratchpad\bd1` and `\b7`: the share file before and after the BRB fix, the script backup, the B8 merge tools) that you copy under `DATA_RAW/issuers/judicial_deposits/_work/` before they vanish, as the first I10 action. The rebuild starts when the Kappa agent posts its finish line in `kappa_handoff_2026-11-03.md`.

## Update 11-04, 11:30: N1 to N3 decided; B8 is in; the rebuild after the BRB fix and the Kappa job

Your reply (f) (`IMPR_REPLY_2026-11-03f.md`, removed after processing) is accepted: the D10 code, the four loader rules, the provisional-scope smoke. B8 landed at 00:45 (`compiled/dfp_funding_expense.csv` 696 rows, `subordinated_in_lump.csv`, `time_deposit_scope.csv`, `brb_deposit_balances.csv`; `dfp_funding_expense_alternatives.csv` is never to be loaded). Reply next in `IMPR_REPLY_2026-11-04a.md`.

```
decided[3]{item,decision}:
  N1,"(ii): for every bank with a 2025 time-deposit line, the pre-2025 CDB expense is lambda_i times the line net of its judicial cost, with lambda_i = Cosif 8113 over the 2025 line net of observed judicial cost (Caixa 1.000, Bradesco 0.971, BB 0.912, Santander 0.903); the scope file serves the banks without a 2025 line (BRB, Banese, Banestes); rate_a4_src = statement for all of them; the constant-excess limit is recorded in the log, with BB's 2018-19 (+0.14, +0.17 against the item-by-item imputation) as the measured size"
  N2,"BB's savings line is excluded in 2018 and 2019 only; used elsewhere"
  N3,"Santander 2024 benchmarked rate marked failed (rate_flag 3, unsplit FX in the lump); the D1 chain carries it from 2023"
```

**What B8 says that your loader must respect** (`BB_REPLY_2026-11-03e.md`, in the file's notes): Caixa's `subordinated` rows 2020-24 are a sub-line of `bills_lf` and Santander's `external` and `subordinated` rows sit inside `time_deposits`: never add them to those lines. Caixa's IHCD (`hybrid`) is not in the Cosif lump in any year (Cosif 8.1.1 equals deposits + repo + letras + other to R$1m in 2022-24 with the IHCD left out), so it is not subtracted; `subordinated_in_lump.csv` records that no bank's notes state a 2022 change and only Caixa's presentation changed (2020); Banese's and Bradesco's subordinated costs are inside throughout. BRB's time-deposit line includes judicial deposits in every year (printed 2016-17 and 2022-24, inferred 2018-21), and no judicial expense is printed; its scope row says so. Santander's time-deposit line carries the capital-instrument interest (footnote 1) and the FX variation, with 2019's sign reversed between DFP 2019 and DFP 2020. Itaú prints no time-deposit line anywhere. Banestes' balance-sheet "A Prazo" includes judicial and special deposits (a trap for anyone dividing by it).

**One rebuild input changes before you start (B-D1, Pedro 11-04):** BRB's judicial path 2018-21 in `Rout/judicial_deposits_share.csv` is being re-anchored by the BB agent (2021-12 at R$3.0 bn from the DFP 2021 management report, interpolated from 2017-12, the 2022 step kept); the straight line overstated BRB's stock by up to R$4.9 bn. Wait for the BB agent's "done" in `BB_REPLY_2026-11-04a.md`, then read the file again.

**Sequence.** The Kappa agent is running a 1.5-hour job (K2b, stored world) on the machine; it posts its finish time in `kappa_handoff_2026-11-03.md`. Start the rebuild after that line and after the BRB fix; announce start and end here; the Kappa agent's step 7 waits for your end line.

## Update 11-03k (22:10): D10 yes; one rebuild, after B8

**Pedro (22:10): D10 adopted, and the rebuild waits for B8.**

```
d10[5]{item,rule}:
  Scope,"for a bank-year with a statement time-deposit line, the year's CDB expense is that line net of the scope items the note lumps into it (judicial, special, foreign deposits, interest on capital instruments, exchange-rate effects), as recorded in the BB agent's time_deposit_scope.csv and its notes; over the ordinary stock"
  Shape,"the year's expense is spread over its quarters in proportion to the strip (H1 and H2 where the note prints the second semester)"
  Fallback,"bank-years without a time-deposit line, or without a resolved scope, keep the component benchmark (D6, D7, D9); banks without statements keep the formula"
  Column,"rate_a4_src with values statement, benchmarked, imputed, carried, so the demand step and K1 run with and without"
  Check,"the pass table against DFP_dom for every statement bank, before and after, in your reply; BRB's judicial-inclusion question settled by the scope file before its line is used"
```

**Sequence.** B8 lands (`bb_handoff_2026-11-03.md`, 22:05: the subordinated and hybrid lines, `subordinated_in_lump.csv`, `time_deposit_scope.csv`, Itaú); you add D10 and the scope reads to calibrate; then one panel rebuild (I6, I8, I9 with D6, D7, D9, D10), announced here at start and end; the demand-frame rebuild; the md5 table with the restaged I2 package; I2 on Pedro's go. Until the rebuild starts the other agents may read `market_panel.parquet`. Reply in `IMPR_REPLY_2026-11-03f.md`.

## Update 11-03j (22:05): your 21:52 reply; D10 to Pedro; B8 sent to the BB agent

Your reply (e) (`IMPR_REPLY_2026-11-03e.md`) crossed with Update 11-03i. Taken: D9 implemented (BB nine of nine within the band, 2019 at the edge); the duplicate guard on the loader; the 2025-as-calibration-window statement, which is right and is now the note's reading too; the `pre2016_scale` and column-order changes. Your three requests to the BB agent are sent as task B8 (`bb_handoff_2026-11-03.md`, 22:05): domestic `subordinated` and `hybrid` lines with foreign issues in `external`; `subordinated_in_lump.csv`; `time_deposit_scope.csv` with BRB first and Itaú's line if it exists.

**D10 was with Pedro (decided yes at 22:10, Update 11-03k).** The main thread's recommendation was yes: for a bank-year with a statement time-deposit line, the year's CDB expense is that line net of the scope items the note lumps in (judicial, special, foreign), over the ordinary stock, with the strip giving the quarterly shape; the component benchmark stays as the fallback; a column `rate_a4_src` (statement, benchmarked, imputed, carried) so that K1 and the demand step can run with and without. **Sequencing (decided 22:10):** one rebuild, after B8 and D10.

## Update 11-03i (21:55): B7 is in; start the rebuild; four rules for the loader

B7 is complete (`DATA_RAW/issuers/funding_expense/compiled/dfp_funding_expense.csv`, eight banks, 666 rows; `Rout/funding_expense_report.md`; the judicial share file now from 2015Q4 with BB's and Caixa's 2015-12 values and BB's 2023 to 2024 quarters observed). **Start the rebuild** (I6, I8, I9 with D6, D7, D9), announce start and end here, then the demand-frame rebuild and the md5 table. Pedro's rules for the loader (21:50):

```
loader[4]{rule,detail}:
  BB savings,"BB's 2018-19 savings line exceeds any lawful remuneration (6.35-6.45% on the stock against 6.17 maximum) and is not used; BB's savings term stays at the official yield with s = 0, rate_a4_vintage_src = none, while its repo, bills and judicial are benchmarked"
  Partial banks,"benchmark only the lines a bank prints (Itaú: none of savings, CDB or bills; Banrisul: no savings or time split; Bradesco, Santander, BRB partial); s = 0 flagged where savings is not printed"
  Santander FX,"its bills (2020, +R$22.2 bn), 'Outras' (2021, -R$9.2 bn) and three lines (2024, +R$42.2 bn) carry exchange-rate effects; net them out where the note separates them, skip the benchmark for that line-year where it does not; its time-deposit line also carries interest on capital instruments (2019)"
  Exclusions,"Banrisul's 2023 judicial line is inferred, not printed: exclude; BRB's BR GAAP alternatives (dfp_funding_expense_alternatives.csv) stay out; two funding_total rows on different bases in one bank-year are never summed"
```

The design note changed again (twenty-third version): its estimating equation now has the exit as a market-quarter coefficient on the awake movable stock; nothing for the rebuild.

## Update 11-03h (21:30): your 11-03d reply accepted; D9 decided; two additions to I10

Your reply of 20:32 to 21:12 (`IMPR_REPLY_2026-11-03d.md`, removed after processing) is accepted: D5 applied and diffed; D7's variant on the record, scope B with the letters-stock guard (BB 8 of 9; 2019 at +0.13); the loader reads B7's file as it is (your two requests to the BB agent are forwarded); the rebuild code in progress and not started before B7. **D9 (the subordinated cost the pre-2025 lump carries from 2022, taken out for the banks with statements): YES (Pedro, 21:30)**, with the start year per bank read from its own statements rather than inferred from BB; where a statement does not say, BB's 2022 is the default, flagged. Reply next in `IMPR_REPLY_2026-11-03e.md`.

**Two additions to I10, from the Pix agent's 21:10 reply:** (i) K0's panel pin is now an environment override (`K0_PANEL_MD5`), and `load_demand` refuses a demand frame older than the panel; the robust form is for the demand prep to write the panel md5 into its parquet metadata, as `utils/demand_chars.py` does with `PANEL_MD5_KEY`, so that readers compare md5s rather than dates: one line in the prep, yours in I10. (ii) When the rebuilt panel lands, announce the md5 here; the Kappa and Pix agents set `K0_PANEL_MD5` from it, and K2's steps that read $\hat\phi$ or $\widehat M$ wait for the demand-frame rebuild.

**The design note changed (twenty-second version):** the switching-cost regression is no longer linearized; the estimating equation is nonlinear in the gap, estimated by NLS or PPML with a control function for the spread as in (10). No consequence for the rebuild; it bears on I10's `kappa/` module when it comes.

## Update 11-03g (22:30): Pedro's decisions on your 11-03c reply; the rebuild waits for B7

Your reply of 19:07 to 20:16 (`IMPR_REPLY_2026-11-03c.md`, removed after processing) is accepted in full: I1 applied and verified (the floor and cap shares stable at 8 to 12 and 4 to 6 percent, $\widehat M$ per capita stable, Spearman +1.00 every year); I6 audited (198 in-window quarters, four mechanisms; k = 1 and 2 clean); I7 done (288 rows, three added, none lost); I8 confirmed with the timing defect found; I9 stock part ready, rate part with the BB check. Reply next in `IMPR_REPLY_2026-11-03d.md`.

```
decided[8]{item,decision}:
  D1 (I6),"(c) the guard band judged against the maximum Selic over t-7..t, then (b) the carried rate (median rate/CDI over t+-4 times CDI_t, flagged) for whatever is still flagged; (a) missing-and-drop as the robustness. New column rate_a4_fallback as you defined it. The appendix sentence on the guard is Pedro's edit, later"
  D2 (I9),"proportional removal in levels across the bank's markets, floored at 0; capital-market flag (state capital x judicial share above 10%); dep_a4 overwritten, dep_a4_incl_judicial, judicial_src and judicial_cap_flag kept; the same helper on the 2015Q4 splice balances"
  D3 (I8),"yes: savings_rate_qoq from SGS 195 (new rule) and savings_old_qoq from SGS 25, quarterly compounds of the monthly credited yields; the end-of-quarter Meta Selic rule goes; new scraper scrape_bcb_savings_yields.py and a raw file under SGS_RAW with a source-log row"
  D4 (I8, the share s),"verified 11-03: the 2012 rule applies only to deposits made from 4 May 2012, and withdrawals drain the new-rule balance first (Lei 12.703/2012), so the old-rule stock persisted; no source gives it by bank (B5). Decision: do NOT hold the rebuild for s. The premium term is coded with s = 0 and rate_a4_vintage_src = none where no statement line exists; where the bank prints its annual savings expense (BB, Bradesco, Caixa now; Itau and Santander after B7's manual read), the DFP expense replaces the imputed savings term through D6, which needs no s"
  D5,"apply the same four lines to switch_cost_sleep_demand_prep_link.py now"
  D6 (I9 rate base),"ADOPTED: judicial expense out of the strip (Cosif 8114 from 2025; before, rho_i x SGS 195 x J_lag with rho_i from the bank's 2025 quarters, the segment value where none); ordinary stock in every denominator; k_i re-measured on the 2025 strip net of 8114; and DFP benchmarking for the banks with statements (imputed repo, bills, savings and judicial terms rescaled to the annual DFP totals, Cosif giving the quarterly shape), which passed 9 of 9 for BB. BB's transcription is promoted from scratch to DATA_RAW with a source-log row; Caixa, BRB, Banese, Banestes, Banrisul, Bradesco, Itau and Santander come from the BB agent (B7)"
  D7 (bills-ratio scope),"it is an error (expense excludes subordinated-debt expense 8119100006, the 4.3 stock includes 4.3.9): fix it in this rebuild, tested as one more variant in the BB check first, so the change in the bills ratios and in k is on the record"
  D8 (I7),"leave the request logs as they are; no dedup rule"
```

**I10 (Pedro, 11-03): bring the adopted diagnostics into production.** The side scripts that found and sized the defects are not where the pipeline will live. Once the other agents are done (B7 landed, K1 run, P3 rerun), you reorganize: every adopted object gets a home in the panel build or in the estimation routine, with its raw inputs registered (source log, `DATA_RAW` folder, upload manifest where the cluster needs them) and the diagnostic retired or kept as a test under `diagnostics/`. A change summary per script before anything moves; the runner (`exhibits_post_cluster.py`, `pipeline_all.sh`) gains the steps. The map, to be confirmed against the code when you get there:

```
i10[10]{object,now,home}:
  Judicial deposits J_jt,"diagnostics/diag_judicial_deposits_cosif.py, diag_judicial_deposits_share.py; Rout/judicial_deposits_share.csv; compiled DFP extracts","a panel step (panel_judicial_deposits.py) that builds J_jt from DATA_RAW/issuers/judicial_deposits/compiled/ and the Cosif 2025 titles, feeding utils/judicial_deposits.py; the diagnostics become its tests"
  DFP funding-expense benchmarks,"coding-thread scratch (BB) and B7's compiled file","DATA_RAW/issuers/funding_expense/ with a loader in panel_cosif_calibrate.py (D6)"
  Savings yields,"I8 scraper, new","scrape_bcb_savings_yields.py in the scraping step; SGS_RAW file; utils/macro_rates.py reads it"
  Rate fallbacks,"Rout/rate_fallback_audit.csv (audit)","the guard and the carried rate inside panel_deposit_rates.py and panel_cosif_calibrate.py (D1); the audit stays as a diagnostic"
  Market size,"applied in sleep_demand_prep_link.py (I1) and the switch-cost copy (D5)","done; no further move"
  Net-flow objects K0,"diagnostics/diag_k0_netflow_objects.py","make_netflow_objects.py in the estimation routine, writing the k0_* files as pipeline outputs"
  Shifters P1 to P3,"diagnostics/diag_p_shifters.py","make_kappa_shifters.py in the estimation routine, after the panel and the CCS steps; P3 depends on the panel md5"
  K2 simulation and K1 regression,"Kappa agent's scripts, when adopted","a kappa/ module in the estimation routine, with its tables under the paper's table scripts and \\pn keys only when Pedro says"
  Booking checks (BB3, B1),"diagnostics/diag_bb_rende_facil_booking.py, diag_santander_sweep_booking.py","stay diagnostics; their verdicts live in the design note"
  Pix roster and 2019 check (R1, R1b),"Rout/pix_participation_*; diagnostics/diag_pix_mandate_2019.py","the roster becomes a panel side table if the awareness margin uses it; else stays"
```

**Sequence.** D5 and D7's variant test now. The rebuild (I6, I8, I9 with D6) starts when B7 lands, which is the BB agent's next task and the only thing it has; announce start and end here. Then the demand-frame rebuild, the md5 table, the restaged I2 package, and I2 on Pedro's go. Until the rebuild starts the other agents may read `market_panel.parquet`.

**Two inputs that reach you from the BB agent with B7:** the 2015Q4 judicial values for BB and Caixa (DFP 2016 comparatives), so the `held` flag goes; and BB's quarterly judicial stock for 2023 and 2024 from its interim statements, because the União deposited about R\$90.7 bn of precatórios at BB and Caixa in December 2023 (withdrawable from January 2024), so both banks' 2023Q4 type-4 stock carries a one-quarter spike of about R\$47 to 49 bn; Caixa's quarters are already observed.

## Update 11-03f (21:25): I9's inputs are complete (Caixa added); the order of the rebuild; where the other agents are

**I9 inputs, complete.** The BB agent's `Rout/judicial_deposits_share.csv` (574 rows, 14 banks) now carries Caixa 2016 to 2024 from its BRGAAP statements (judicial share 0.33 to 0.71; the note's time-deposit total equals Cosif 4.1.5 at every December; 2025-12 equals Cosif 4155000009). One flag inside: Caixa's judicial line falls R\$43.0 bn in 2024 and the quarters are interpolated across it until the BB agent dates it from the 2024 quarterly statements (B6). Use the file as it is; B6 updates it in place with observed quarters.

**The rebuild, in order.** I1 (market size), I6 (fallback guard), I8 (old-rule premium out of the type-4 rate; B5 supplies the bank-level old-rule stock, the SGS national share elsewhere), I9 (judicial stock out of `dep_a4` and out of the rate base, with the capital-market flag). Four change summaries, one panel rebuild, one demand-frame rebuild, a new md5 in the reply, then I2 on Pedro's go. The K1 regression of the design (Kappa agent) and P3 (Pix agent) wait for the md5.

**Other agents.** A fourth session exists from 11-03: the **Kappa agent** (`kappa_handoff_2026-11-03.md`), which runs the design's simulation (K2) now and the regression (K1) after your rebuild. The Pix agent builds the shifters (P1 to P3, `pix_handoff_2026-11-03c.md`); the BB agent has B4 to B6. All read `market_panel.parquet` through pyarrow and one at a time; say in your reply when the rebuild starts and ends so they stay off the file.

## Update 11-03e (20:00): Pedro's decisions; I9, judicial deposits out of `dep_a4` and out of the rate base; the order before I2

**Pedro (11-03, 19:45): "we should fix those things."** Decisions:

```
decided[5]{item,decision}:
  I1,"APPROVED: apply the market-size fix on your change summary (before/after floor and cap shares, median M-hat per capita by year); keep the access rule, fixed, rather than switching to findex"
  I8,"APPLY, not conditional: correct the type-4 rate for the old-rule savings premium. Input priority: the bank-level old-rule stock from the statements' notes (B5, BB agent); where a bank has none, the national old-rule share from the SGS applied to its savings stock, flagged. Change summary first"
  I6,"audit the 186, then the guard fix on a change summary; before I2 if ready"
  I9,"NEW, below: judicial deposits out of dep_a4 and out of the rate's stock base; change summary first"
  I2,"go after I1, I6, I8 and I9 are in, as ONE panel rebuild and ONE demand-frame rebuild; not before"
```

**I9: judicial deposits are inside `dep_a4` and the rate chain divides by them.** The BB agent's B3 (`BB_REPLY_2026-11-03b.md`, 17:05; files `Rout/cosif_415_subaccounts_monthly.csv`, `Rout/judicial_deposits_share.csv`, `Rout/cosif_judicial_rates_2025.csv`, `Rout/judicial_deposits_report.md`; scripts `diagnostics/diag_judicial_deposits_cosif.py`, `diagnostics/diag_judicial_deposits_share.py`; plan of accounts staged in `DATA_RAW/BCB/cosif_plano_contas/`). Verified facts, which you should re-check against the code before the summary:

```
i9[8]{item,finding}:
  Where they sit,"Cosif title 4.1.5.50 (Depósitos Judiciais ... Mantidos na Instituição) throughout 2016-2024; ESTBAN verbete 432 = dep_a4 (panel_deposits.py:315) maps 4.1.5.10, .20, .50 and .99 together (Annex I of IN BCB 429/2023). Published monthly only as the 4.1.5 total before 2025; the four titles from 2025-01"
  How much,"BB 51-76% of its 4.1.5 (December shares 2016-2024: 0.70 0.72 0.76 0.75 0.67 0.65 0.56 0.60 0.51); Caixa 0.37-0.43 (2025 only; 2016-2024 needs its statements, B6); BRB 0.20-0.56; Banese 0.29-0.49; Banestes 0.21-0.31; Banrisul 0.12 from 2024Q2 (booked outside 4.1.5 before IN 459/2024: its dep_a4 jumps +16% in 2024Q2 for accounting reasons alone); everyone else under 3.2%"
  The rate defect,"panel_cosif_extract.py sets stk_time = 4150000004 (the 4.1.5 total, judicial included) but exp_cdb = 8113000009 (ordinary time deposits only; the judicial expense is 8114000008, published from 2025-01); panel_cosif_calibrate.py divides |exp_cdb| by stk_time_lag at line 330 (the 2025+ rate) and in the de-bias factor k_i (lines 229-233), which rescales every pre-2025 rate. So BB's measured CDB rate is about half the rate on its ordinary time deposits (factor 0.49-0.51 in 2025), Caixa 0.61-0.63, BRB 0.55, Banese 0.70, Banestes 0.75, Banrisul 0.87. This explains BB3: BB at 0.45 of the Selic against 0.6-0.9 for the controls; doubled it is about 0.9"
  What they pay,"0.019-0.032 a quarter, savings-rate levels, higher at BB (0.027-0.029)"
  Local booking,"ESTBAN assigns the balance to the municipality of the booking branch, typically the court's seat, so local type-4 shares of BB, Caixa and the state banks are distorted where courts sit; not measured"
  Series available,"judicial_deposits_share.csv: 574 rows, 14 banks, year-end DFP stocks interpolated linearly and divided by the actual quarter-end 4.1.5 total, flags for interpolation, Cosif-observed from 2025; BB's DFP also splits the judicial expense annually (e.g. R$12.2 bn in 2016, DFP 2016 p297)"
  Cross-check,"BB's DFP judicial stock at 2025-12 equals Cosif 4155000009 exactly (R$263.4 bn); the 4.1.5 total read by the BB agent equals the extractor's stk_time for BB every month"
  Why it matters,"dep_a4 is the type-4 stock in the demand model (share overstated, price understated, so xi overstated at BB and Caixa), the carried stock in the sleepiness regressions (judicial money moves with court outcomes, not rates, so it reads as inertia), and the exit pool Q4 of the switching-cost design"
```

**The task.** A change summary covering: (1) `dep_a4_ex_judicial` (and the type-4 total used anywhere the rate or the shares are built) = `dep_a4` minus the bank's judicial stock from `judicial_deposits_share.csv` (Cosif 4155000009 from 2025-01; DFP interpolation before; Caixa from B6 when it lands, flagged until then); (2) the rate base: `stk_time` excluding 4.1.5.50 wherever `exp_cdb` excludes 8114, in the 2025+ rate and in `k_i`; for BB, check the rebuilt pre-2025 rate against the annual DFP rate ex judicial; (3) the local allocation: first pass, remove the bank's national judicial share proportionally from its `dep_a4` in every market, with a flag on state-capital markets where the true share is higher (Pedro's decision below); (4) Banrisul's 2024Q2 reclassification handled by the same subtraction (its share is zero before 2024Q2 by construction of the file); (5) which keys, tables and the Appendix text on deposit types change. Then the panel rebuild, with I8's correction in the same rebuild, and a new md5. Nothing runs before Pedro's OK on the summary.

**Decision for Pedro inside I9:** proportional removal across markets (first pass) with a capital-market flag, or drop the type-4 cells of BB, Caixa and the state banks in state capitals as the robustness. Main-thread recommendation: proportional plus flag; the robustness in K1 and in the demand fit.

## Update 11-03d (19:00): labels rebooted; one defect found by K0 that goes before FR3; your 16:32 reply accepted

**Labels, rebooted (Pedro, 11-03).** From here on tasks are numbered by agent: **I** for this thread, **B** for the BB agent, **P** for the Pix agent's own tasks, **K** for the switching-cost design (run by the Pix agent). Old names stay in parentheses once. Your reply names stay `IMPR_REPLY_<date>.md`; the next is `IMPR_REPLY_2026-11-03c.md`.

```
I_tasks[10]{id,was,what,when}:
  I1,"new","the market-size access index is inverted (below): decide with Pedro, then apply the one-line fix in build_market_size_and_shares as a change summary, verify on the E3 and E4 demand frames, and report the new share-at-floor and share-at-cap statistics against the stored ones","NOW, before I2; Pedro decides"
  I2,"FR3","the full cluster chain on the 36-quarter frame, with the I1 fix in the upload set (sleep_demand_prep_link.py changes again, new md5)","after I1, on Pedro's go"
  I3,"FR4","RC ladder and six-start multi-start under T1 to T3, E3's archived ext1/ext2 among the starts, data/output/blp backed up first","after I2"
  I4,"FR5 + M5","keys against R3; SEs with the exact Jacobian; the L1 items; blp_status ok per routine; tables and figures; every changed key with sign and significance crossings","after I3 adoption"
  I5,"PC1","PHI_CAP keys demand.phicap.{rows,dep}.E{3,4} on the I2 fits (E4's share at phi-hat = 1 is 33.4% in FR2)","after I2"
  I6,"RT1","rate-chain fallback audit, corrected premise: K0 finds rate_a4_guarded = 1 in five of the seven Itaú and Santander quarters (rate = Selic) and 0 only in 2021Q2 (rate = 0); its rule (national type-4 rate equal to the Selic to 1e-6, or zero, with positive stock) flags 186 B conglomerate-quarters, including Safra (8), BMG (8), Daycoval (7), BNP (8), Citi (3), Pine (4); list in Rout/k0_numbers.json -> rate_fallback_cq. Audit all 186 for k in 1, 2, 4; name the chain step; count demand and sleepiness cells; guard fix as a change summary","now, read-only"
  I7,"SL1","master source log: builder inputs catch up with CC1 and S4, rebuild, diff","now"
  I8,"new","the old-rule savings premium sits inside the type-4 rate (below): confirm the mechanics in panel_cosif_calibrate.py:142 and panel_deposit_rates.py:714; then APPLY the correction (Update 11-03e) on a change summary, with B5's bank-level old-rule stock where it exists and the national SGS share elsewhere, flagged","change summary now; apply with I9 in the one rebuild"
  I9,"new","judicial deposits out of dep_a4 and out of the rate's stock base (Update 11-03e; D6 in 11-03g)","summaries ready; one panel rebuild with I6 and I8 once B7 lands, before I2"
  I10,"new","bring the adopted diagnostics into the panel build and the estimation routine (Update 11-03g)","after the other agents are done"
```

**Your 16:32 reply: accepted.** E3 and E4 demand parquets identical under FR1's frame changes (0 columns differ), so the paper's demand-side numbers move only through the refits and E4's real growth; the window is 36 quarters; the final centring file has 0 rows without real growth. The FR3 package (11 files, md5s listed) is approved in content, but **do not upload it yet**: I1 changes `sleep_demand_prep_link.py`.

**I1: the market-size access index is built from centred GDP per capita.** Found by the Pix session in K0 (`Rout/k0_report.md`, section (d)); nothing edited. The facts, which you should verify before the change summary:

```
i1[6]{item,finding}:
  Where,"build_base_panel (sleep_demand_prep_link.py) scales gdp_per_capita by 1e4 and subtracts the grand mean (utils/state_transform.CENTER; on unless SLEEP_CENTER_STATES=0); build_market_size_and_shares then computes access = gdppc / its yearly median on that centred column (lines 364-367); the same function serves E1, E2, E3 and E4"
  Check,"recomputing access from the parquet's centred gdp_per_capita reproduces the stored _access to within 0.024"
  Sign,"Spearman(GDP per capita, access) = -0.992 in 2019: richer markets get the larger outside option, the reverse of Appendix 'Market Size'; markets above the grand mean have a negative ratio and sit at the 0.35 floor"
  Drift,"as the yearly median of the centred variable nears zero, the share at the 0.35 floor goes 19% (2016) to 56% (2023-24) and the share at the 2.5 cap 0% to 53% (2022); the median market's M-hat per capita jumps 6.7x from 2022 to 2023"
  Touches,"M_mt, M_nat, the inside shares, xi-hat, the floor statistics behind demand.sharecap.*, and every counterfactual that reads M_mt"
  Fix proposed,"in build_market_size_and_shares, add back the persisted grand mean before the ratio: x + x-bar with x-bar = _st.load_transform().mean_scaled['gdp_per_capita'] when centering_enabled(); the 1e4 scale cancels in the ratio; E1 and E2 centre in their own build_base_panel (lines 300 and 296) but import this function, so one edit covers all four preps; DEMAND_BC_RULE=findex is the existing switch for a run without the access term"
```

**Main-thread reading.** If the check reproduces, this is a bug in the paper's market-size construction, not a design choice, and it has been in every demand fit since the access rule was introduced. It must be fixed before I2, because I2 rebuilds the demand frames. Pedro decides; the change summary is yours, with the before and after of the floor and cap shares and of the median $\widehat M$ per capita by year, so the paper's Appendix text can be checked against the fixed numbers.

**I8: the vintage premium of savings is booked as time-deposit expense.** K0 section (c): `rate_a2` is the regulated new-rule rate, one value per quarter for every bank; Cosif splits savings expense from the lumped funding expense only from 2025-01; so the pre-2025 type-4 rate (raw and corrected) is the lumped expense less savings at the new-rule rate. A bank with an old-rule share $s$ of its savings stock has its time-deposit rate overstated by $s\times\text{premium}\times\mathrm{Dep}_{j2}/\mathrm{Dep}_{j4}$ in 2017Q4 to 2021Q3, when the premium was positive (1.5 pp a year in 2018, 4.8 at the 2020 trough). Per unit of $s$, 2018 to 2020 mean: Caixa 4.6 pp, BB 2.3, Itaú 1.6, Bradesco 1.5, Santander 0.7; Caixa 10.2 in 2020Q4 alone. This is a paper-level issue for $\rho_{j4}$ of the big banks in those quarters, not only a design-note one. The old-rule stock by bank is the missing input (the BB agent checks the DFP notes, task B5); the SGS has it nationally.

## Update 11-03c: what the IMPR session's FR0 to FR2 supersede here (16:50)

`IMPR_REPLY_2026-11-03.md` (15:57; its handoff `impr_handoff_2026-11-02.md` was removed by Pedro, so the FR plan now lives in that reply and here) reports: M3b and the E4 chain cancelled (FR0); the zip 31716854 archived, not ingested (E3 sigma to ext2 and E4 sigma to ext1 from the corrected engine; E3's ext1 θ₂ = [0, .831, .036, −.314, −.083] is a warm start); the sleep frame now covers 2016Q1 to 2024Q4 (36 quarters, 1,174,484 rows) with the two funding-ratio instruments spliced back to 2015Q4; real per-capita GDP growth with t−1 timing replaces the nominal series for E4 (B3; the E4G per-market nominal code is removed); re-centring done (B4); per-routine BLP status (B5); FR2 sizing on the new frame. What that changes in this file:

```
supersede[7]{item,before,now}:
  M3b,"ladder running, then the six-start multi-start","cancelled (FR0). The RC ladder and multi-start are FR4, on the new frame, with E3's archived ext1/ext2 among the starts. The adoption thresholds T1 to T3 (Update 11-02) and EXT1_TAG discipline carry over to FR4 unchanged"
  E4G,"per-market nominal gdp_growth_yoy, sizing, rerun after E3 adoption","done differently: FR1-B3 (real growth, t-1 timing, mca_gdp_real_growth.csv). Closed"
  PC1,"PHI_CAP keys for E3 and E4 on the R3 frame","still owed, but on the FR3 fits: FR2 finds E4's share of rows at phi-hat = 1 doubles to 33.4% (E3 stays 16.8%), so the E4 disclosure will be larger. Produce the keys after FR3, not before"
  M5,"after M3b adoption","after FR5 (keys against R3). The M5 spec is unchanged: SEs with the exact Jacobian, the L1 items, blp_status ok per routine (B5 now allows E3 and E4 separately), tables and figures rebuilt, every changed key with sign and significance crossings"
  Pedro's lines,"multi-start start file; squeue; grep of the 317168* logs","all stale; removed below. New lines come with FR3's submission"
  Open question 1,"adoption once M3b reports","adoption once FR4 reports which fits pass T1 to T3"
  Paper text,"Section 5.2 rewrites and the red provisional note held until M5","still held. FR2's sign flips (65+, mobile lines, E4 GDP growth) are in AMEs R3 found insignificant, so no sentence in the 11-02 plan is contradicted yet; the 36-quarter window will change every count of quarters the paper states"
```

**For the design note (fourteenth version):** nothing in its derivation depends on the sleep or demand frames. The first-stage F statistics it quotes (16.7 MCA clusters, 4.4 state clusters) are R3 values and will be refreshed after FR3; K0 reads the panel, which FR1 did not rebuild. The two funding-ratio instruments now exist for 2016Q1, which matters for K1's instrument set $\boldsymbol{z}_{jmt}$.

## Update 11-03b: two findings from the BB session, and tasks RT1 and SL1 (16:30)

The BB session (`BB_REPLY_2026-11-03.md`, 15:25, removed; answered in `bb_handoff_2026-11-03.md`) checked where BB Rende Fácil is booked. The sweep is immaterial (at most a few R$ bn, no effect on BB's measured rate), but two things it found touch the panel you own:

```
bb3[2]{finding,consequence}:
  Judicial deposits,"BB books court-ordered deposits inside Cosif 4.1.5; the DFP puts them at 60% of BB's time deposits at end-2021, 52% in 2022, 57% in 2023, 48% in 2024. They sit in dep_a4 (which tracks the 4.1.5 entity total plus R$9-11 bn) and their expense in rate_a4_national. BB's rate is the lowest of the five in 10 of 15 quarters from 2021Q2. Task JD1 (BB agent) sizes the judicial share for every bank from the Cosif subaccounts or the DFPs. Nothing for you until JD1 reports; if the share is separable in Cosif, a dep_a4_ex_judicial column will be a change summary for the prep"
  Rate fallbacks,"in the 10-27 panel (d01b8bae), rate_a4_national equals the Selic exactly for Itaú in 2020Q1, 2020Q2, 2021Q1 and for Santander in 2020Q1, 2021Q1, and is 0.000 for both in 2021Q2 (so the type-4 spread equals the Selic there); rate_a4_guarded is 0 in all seven. Task RT1 below"
```

```
tasks_new[2]{id,what,output}:
  RT1,"Rate-chain fallback audit: list every conglomerate-quarter in the 10-27 panel where rate_ak_national equals the Selic to 1e-6 or equals 0 with rate_ak_guarded 0, for k in 1, 2, 4; name the script and the step of the rate chain that produces each; count the demand-sample cells and the sleepiness cells affected; propose the guard fix as a change summary (no edit before Pedro's OK)","Rout/rate_fallback_audit.csv, and the change summary in your reply"
  SL1,"Master source log: move the CC1 row and the S4 edit (notes 'DELIVERED', location BCB/ccs_lai_18810_025400) into the builder's inputs (_extra_source_log_rows.csv or the request logs, whichever the convention in DATA_RAW/README.md names), so that _requests/build_master_source_log.py reproduces the current master plus the AW1 and BB1 rows; then rebuild and diff against the current master; report the diff. Change summary first if a script changes","the diff in your reply; DATA_RAW/source_log.csv rebuilt"
```

**Other agents, for the record.** The Pix side session's tasks now live in `pix_handoff_2026-11-03.md` (task K0: the objects of the net-flow regression of the design note's fourteenth version; no regression yet). The BB session has BB5 (Santander) and JD1. Do not read `market_panel.parquet` while they do; the rule is one heavy process at a time, and they read it through pyarrow.

## Update 11-02 (after your reply of 13:34)

```
update[7]{item,state}:
  Backup,"done 13:27: 70 files (sigma to extended, E3 and E4, the adopted ext1 included) in data/output/blp/_superseded_ladder_20261102_132721; the two _superseded_ext1_* folders untouched"
  Launch,"driver blp_run_m3, job 31716408 (blp_run.sh --routines '3 4'), queued 13:30; it submits rc_ift_E{3,4}_head (sigma to ext1), _ext2 and _ext (extended), each blp_stage_job.sh on gpu_h200, 6 CPUs, 200G, 2-day wall, on the verified engine"
  Thresholds,"APPROVED (Pedro 11-02). Accept a fit only if fd_check gives (T1) projected FD gradient norm <= 1e-3, (T2) best dQ along the FD descent direction >= -1e-7, and (T3) se_status ok with no fd_check error. Among the six ext1 starts adopt the lowest Q that passes; if two passing fits differ in Q by less than 1e-7 but materially in theta2, report both"
  CPUs,"KEEP 6 for the ladder (Pedro 11-02). Decide for the multi-start from the ladder's measured times"
  Sizing,"expected per routine: head 6-21 h, ext2 1-5 h, extended 6-16 h; a wall-kill resumes by resubmission since finished stages are skipped; the multi-start runs with SBATCH_TIMELIMIT=24:00:00"
  Step 6,"done: exhibits_post_cluster.py step 6 is make_blp_rc_table.py --ests 3 4; --dry-run lists it ok; --status shows NEVER until its next run"
  fd_check gap,"fd_check does not record whether its own contractions converged; noise would fail T1, so the gate stays conservative; add the flag later, not in the running engine"
```

**Main-thread reading:** the thresholds are sound. At a curvature of about 60 along the steepest FD line, a projected gradient of 1e-3 bounds the possible drop at about 8e-9. The old fits fail both T1 and T2 by two to four orders of magnitude.

**Rules (unchanged):**
- V_Friction and the decks are edited by Pedro only.
- Pedro owns git, and you give `Remove-Item -LiteralPath` lines rather than deleting.
- Raw files stay untouched.
- Python is `C:/venvs/egan/Scripts/python.exe -I`, with `MPLBACKEND=Agg`.
- Data window: 2016–2024.
- Cluster commands are single-line and non-blocking.
- Before you edit an existing script, summarize the change and its side effects and wait for Pedro's OK.
- Timestamp every update. "Wait" means stop.

## Update 11-02b (after your reply of 13:52)

```
update_b[6]{item,state}:
  Ladder,"submitted 13:30 by driver 31716408: six RC jobs 31716848-31716853 (gpu_h200, 6 CPUs, 200G, 2-day wall) plus blp_zip 31716854 (afterany); head jobs waiting on priority at 13:39"
  Node shape,"a gpu_h200 node has 48 CPUs and 8 H200s, so 6 CPUs is one GPU's share; QOS part_gpu_h200 caps GPUs (16) and jobs (6)"
  Multi-start,"EXT1_TAG=20261102 with the 10-29 starts (best1019, rc3, logit, rand1, rand2, rand3 per routine), so each start's old and new endpoints differ only through the gradient; launched after the ladder's ext1 exists, SBATCH_TIMELIMIT=24:00:00; with the ladder's own cold and warm_rc4 starts, eight corrected fits per routine compete under T1-T3"
  L1 job,"ready, not uploaded: blp_l1_core.jl (84ea78ef...), blp_l1_items.jl (df452ebf...), blp_l1_items_job.sh (6f82f2d7...); runs at a fit after adoption, one job per routine"
  L1 output,"per-row RC own-spread elasticity (row- and Dep_Act-weighted, B, D and all) beside the plug-in; alpha_ir mean, share >= 0 and percentiles under row-chooser, deposit-chooser and population weights; means of each D-tilde; share and contraction checks"
  L1 tests,"synthetic system: elasticities match FD of shares to 5e-9, weighted moments exact; against the engine, mu equals rho (alpha_ir - alpha) to 1.4e-14 on two real E3 quarters"
```

**Main-thread reading.** The design is sound. Reusing the 10-29 starts makes the old-against-new comparison clean, and the L1 job measures what the abstract's number should be. Two small additions for M5 (task M5b), and one old question still unanswered (task P11).

## Update 11-02c (after your reply of 14:10)

```
update_c[5]{item,state}:
  P11,"answered: a firm in ESTBAN in 2016-2024 but absent from the December 2024 cross-section defaults to B (panel_market.py attach_mca_code l. 358); 2 of 474 conglomerates take that default: C0080525 Kirton (ex-HSBC, 1,572 municipality rows, B is right) and C0081218 (9 rows), 0.19% of dep_a4 together. The paper sentence (block P11) now says so"
  M5b,"done: plug-in at row and Dep_Act weights; on-impact lines with L1_MEAN_PHI = sleep.meanphi_frac.E<k>. md5s: blp_l1_core.jl 84ea78ef... (unchanged), blp_l1_items.jl 41f8dac0..., blp_l1_items_job.sh 62dfe238..."
  Ladder sigma,"E3 and E4: projected FD norm 0.0, sigma at its lower bound with the derivative pointing out of the box: a constrained minimum, passes"
  Ladder rc2,"E3: FD norm 1.2e-6, best dQ +8.8e-9; E4: 8.1e-7, +8.5e-9; both pass T1 and T2 by about 1,000x; 5-7 min per stage"
  ETA,"ext1 about 19:00-23:00 on 11-02; extended and the multi-start about 03:00-14:00 on 11-03; adoption and M5 on the afternoon of 11-03"
```

**Main-thread reading.** The first two rungs pass the thresholds by a wide margin, which is what an exact gradient should give. The P11 default is harmless in the data and now stated in the paper.

**New task XI1 (Pedro, 11-02): shorten the note of `tab_xi_diagnostics`.** Its definitions move into the appendix text (block P19 of the paper plan, which Pedro applies). Replace the note body in `make_xi_diagnostics.py` (or wherever its wording lives, for example `config/table_notes.toml`) with the text below. Keep the "Provisional" sentence, which your code appends while the BLP family is pending.

```
\textit{Notes:} $\hat{\xi}$ is the demand residual of the random-coefficients model, specification 12, in the brick-and-mortar (B) firms' conglomerate $\times$ deposit type $\times$ minimum comparable area (MCA) $\times$ quarter cells, weighted by deposit balance. Panel A: Shapley shares of the weighted variance. Panel B: slope of $\hat{\xi}$ on its previous-quarter value. Panel C: one regression per variable, with quarter effects. Definitions: Appendix \ref{app:empirical:xi}. Brackets: 95\% wild cluster bootstrap intervals (Webb weights, 999 draws, bias-corrected), conglomerate clusters. *** $p<0.01$, ** $p<0.05$, * $p<0.1$.
```

- This edits an existing script or config, so summarize the change and wait for Pedro's OK.
- Then regenerate the table from its cache (no ξ̂ re-read) and confirm that only the note line differs.
- The main thread compiled a stand-in: the short note also removes a 21-point overfull box.

## Update 11-02d (after your reply of 14:38)

- **XI1 is done.**
  - `make_xi_diagnostics.py` writes the short note, word for word, and the 'Provisional' sentence still follows it while pending.
  - The table differs in the note line only, with 0 of 225 numbers changed and a clean standalone build.
  - The main thread checked the file: it holds `\ref{app:empirical:xi}` correctly. The "ef{" in your reply is only how the reply rendered it.
  - Pedro can apply block P19 now.
- **New task CCS1 (main thread, 11-02): how much does double counting move the CCS opening shares?**
  - A relationship is one client at one institution. A conglomerate with several institutions counts a client once per institution, and a client who adds a second institution of the same conglomerate months later shows up as a new opening.
  - Your sensitivity (one institution per conglomerate and month) handles only the same-month case.

## Update 11-02e (after your reply of 14:46)

**CCS1 is done** (`diagnostics/diag_ccs_main_entity.py`, `Rout/ccs_main_entity_ccs1.md`). The D share of PF gross openings:
- 2016: 0.099 as now, 0.099 main entities only, cross bounds 0.097 to 0.101.
- 2024: 0.747 as now, 0.704 main entities only, cross bounds 0.690 to 0.760.
- Twelve of the 20 largest conglomerates have two or more institutions. On the D side these are mostly a second active relationship of the same clients (Nubank's credit company, C6's broker). On the B side they are closed products (Itaucard, Bitz, Superdigital).

**Main-thread reading.** The rise from about 10 to about 70–75 percent survives every version, so the paper keeps "75 percent" in the text and gives the stricter figure in the footnote (block P20 of the paper plan).

**New task CCS2: one key for that footnote.**
- Add `ccs.pf.dshare_open_main.2024`: the 2024 D share of PF gross openings, main entities only (0.704). Same collector and family (local) as the other `ccs.*` keys.
- Use exactly this name. The paper block quotes it, and a leaf under the existing `ccs.pf.dshare_open.2024` could clash with the prefix matching.
- Run `make_paper_numbers.py` and report that no other value moved.

**New task SV1 (Pedro, 11-02): shorten the note of `tab_state_vector_map`.** Its definitions move into the appendix text (block P21, which Pedro applies). In `make_state_vector_table.py` (the `note` string near l. 146), replace the note body with:

```
\textit{Notes:} Panel A: the states at specification (12) and the collected candidates left out. Absorbed by FE: the share of the state's variance absorbed by bank $\times$ type $\times$ market and quarter fixed effects, on the specification (12) sample. Panel B: the two-way-demeaned regressors of the second stage of the sleepiness estimation. VIF: variance inflation factor. Definitions: Appendix \ref{app:empirical:sleep}.
```

- Pedro asked for the change, so give the one-line change summary and proceed.
- Regenerate from the sidecar or cache, and confirm that only the note line differs and that `tab_state_vector_map.json` is unchanged.

## Update 11-03a: R1b, TED1 and AW1 done (Pix side session, `PIX_REPLY_2026-11-03.md`, 15:00)

```
pix[7]{item,state}:
  R1b,"2019 B savings at non-mandated conglomerates: R$12 m of R$3,225 bn (Bank of China only); the market exposure of tab_pix_exposure_fs draws its variation from demand (5%) and time deposits (6%) at non-mandated banks. Script diagnostics/diag_pix_mandate_2019.py"
  TED1 edit,"fees/panel_fee_merge.py (approved 14:15): services F1315/F1313 (TED internet/branch) and F1312/F1310 (DOC), derived listed_fee_ted_pf and listed_fee_doc_pf (internet else branch), STICKINESS_SERVICES pinned; all 26 existing listed_fee_* columns identical on 51,512 conglomerate-quarters; --patch-market not run; separate panel BCB/Tarifas/processed/transfer_fee_listed_conglomerate_quarter.csv"
  TED1 result,"the 16 savings-holding B conglomerates list TED at R$9-11 by internet (R$20-22 at the branch), move together, none ever at zero; the pool fee's sd across markets in 2020Q3 is R$0.33 on R$10.50 (3%); 63 within-bank changes before Pix, all small increases in common quarters. NO DOSE. Listed prices are ceilings; charges are not in the census"
  TED1 scaling,"Pedro chose (c) R$ per transfer plus (a) bp of the rival's savings per client; (a) varies through balances per client (bank mix), not fees"
  AW1,"B openings at the launch: post x on-Pix 0.80 without trends, 0.15 [-0.24, 0.55] with a linear trend per conglomerate (WCR p 0.47); no break in the event path; Pix use in the footprint does not predict post-launch openings (-0.002 +- 0.22). D: 1.25 without trends, 0.01 with, common post 0.31 (the entry wave). Script make_ccs_openings_pix.py; adults from PNAD Continua (SIDRA 6407), filed in DATA_RAW/IBGE/pnadc_population/"
  AW1 caveats,"unweighted over 67 conglomerates; the comparison group for the event path is 22 wholesale banks; Caixa's client count reaches 93% of adults (CCS counts firms and minors); dropping N or Caixa leaves the estimate"
  Files,"make_ted_fee_report.py, make_ccs_openings_pix.py, diagnostics/diag_pix_mandate_2019.py (new); Rout/ted_pool_fee_market_quarter.csv, Rout/ted_fee_report.json, Rout/ccs_openings_pix_aw1.json, Rout/ccs_openings_pix_panel.csv; DATA_RAW/_extra_source_log_rows.csv gained one row (master log not rebuilt)"
```

**Main-thread reading (design note, tenth version).**
- **The fee dose is dead.** Listed TED fees do not vary where the savings are, so $\alpha_f$ is not identified and the money against hassle split is not available from data. The fee panel stays as a control.
- **Awareness of B banks did not move at the launch.** That is the empirical support the before-after reading needed (its condition I-3), with a bound of 0.15 in logs. D destinations stay out.
- **The dose is now Pix use by market,** instrumented by 2019 mobile access, with the within margin absorbing attention and AW1 answering the awareness objection for B destinations.
- The entry-side regression itself (task K1) is the next step and is ready to specify once Pedro reads the note.

**Pending Pedro:** the master source log was not rebuilt after the PNAD row; rebuild at the next convenient point.

## Update 11-02k: R1 done (side session, `impr_reply_2026-11-02.md`, 20:15), and what it changes

```
r1[8]{item,state}:
  Deliverable,"Rout/pix_participation_conglomerate_quarter.csv (5,745 rows, 456 panel conglomerates, 2020Q4-2024Q4; on_pix, first_on_pix, mandated, source, is_B, legs, members, balances by type); make_pix_participation.py (new); Rout/pix_participation_r1.json"
  Control group,"EMPTY for savings: off-Pix B conglomerates hold at most 0.0003% of B savings in every quarter (only Bank of China, at most R$2.9 m). All 16 mandated B conglomerates hold 99.9997% of B savings in 2020Q4; the 29 voluntary joiners and 22 never-joiners hold none"
  Where variation exists,"destination side, wholesale: off-Pix B hold 1-4% of B time deposits (Citibank to 2021Q2, then Pine, BNP Paribas); off-Pix D hold 14-20% of D time deposits (captive finance companies whose CDBs sell through brokers that are on Pix)"
  Roster coverage,"two gaps in the first six months (Dec 2020; Feb-May 2021) and three later (up to 643 days); all captures that exist are cached, nothing to download; the key leg dates joins to the month; no parse gaps"
  Legs,"84% agree on the quarter; the roster gaps explain 126 of 131 cases where the roster is later; 14 tiny institutions hold keys while still listed as joining (7 D firms in the panel, R$65 m)"
  Mandate,"the 34 mandated institutions map to 30 panel conglomerates (16 B, 14 D), the same 30 as make_pix_exposure.py"
  Design consequence,"the bank-pair mandate design of kappa_design_2026-11-02.md is dropped for savings; the Pix change in kappa is now read from before-after with the within margin as control, and from the pre-launch TED fee as the dose (its Sections 5.1 and 5.5, seventh version)"
  Flag,"a stale roster can keep a conglomerate on after its keys stop (Woori 2023Q3-2024Q3, Pine, BNP, Inbursa); none holds savings; keys-override-stale-roster is one condition in build() if wanted"
```

**Main-thread reading.** The result is a clean negative, and it is the right kind: it shows the pair design cannot work before anything was estimated on it. The roster itself stays useful, for the D firms and for the awareness margin. The design note is updated (seventh version): the fee is now the shifter of $\kappa$, TED1 comes first, and AW1 second.

**Pedro's decisions (11-02, after 20:15):**
1. **Dispatch TED1 and AW1 now.** TED1 first: it edits `fees/panel_fee_merge.py`, so give the change summary and wait for the OK. AW1 is new files only. Specs below.
2. **Yes to the one-line check (task R1b):** were 2019 savings (type 2) at non-mandated B banks also zero? Report the 2019 share of B savings, and of each type separately, at the 34 mandated institutions' conglomerates, with the same mapping as `make_pix_exposure.py`.
3. Keys-override-stale-roster rule: not needed, since no savings holder is affected. Leave `build()` as is.
4. `impr_reply_2026-11-02.md` was removed on Pedro's instruction. Its tables stay in `Rout/pix_participation_r1.json`.

**Specs of TED1 and AW1** (from `Drafts/Deposit Competition/kappa_design_2026-11-02.md`, Section 11; equation numbers refer to that note):

### Task TED1 (first): the TED and DOC fees by source bank

**Input:** the tariff census (`DATA_RAW/BCB/tarifas_monthly`, the same files as LAI 18810.025439), parsed by `fees/build_account_fee_history.py`.

**Build:** add to `LISTED_FEE_SERVICES` in `fees/panel_fee_merge.py` the PF services for TED (electronic transfer, the standardized code for "Transferência por meio de TED") and DOC, in the internet or application channel where the census lists channels separately, and at the branch otherwise. Output `listed_fee_ted_pf` and `listed_fee_doc_pf` by conglomerate and quarter, 2016Q1 to 2024Q4, through the existing versioned-price logic (price in force on the quarter's last day, head institution). This edits an existing script, so a change summary first.

**Then**, for each B market and quarter, the source pool's fee: the savings-balance-weighted mean of rivals' TED fees,

$$
\bar f_{jm,t-1}=\frac{\sum_{j'\neq j}\mathrm{Dep}_{j'2m,t-1}\,f_{j't}}{\sum_{j'\neq j}\mathrm{Dep}_{j'2m,t-1}},
$$

scaled by the market's mean savings balance per relationship where the CCS stock allows, else by the mean balance per cell.

**Report:**

- the distribution of listed TED fees across conglomerates and over time, and the share of conglomerates listing zero (digital banks and packages);
- the cross-sectional correlation of a bank's TED fee with its time-deposit spread and with its share of truncated cells, which is the endogeneity worry of Route 2;
- the date each conglomerate's TED fee went to zero, if before Pix: these are the within-bank fee changes that identify $\alpha_f$ in (I9) of the design note;
- $\bar f^{\,0}_{jm}$, the pool fee at 2020Q3, by market, and its spread across markets, which is the dose of (fee dose) of the design note.

Nothing is regressed on the deposit panel until Pedro reads both reports and R1.

### Task AW1: did openings at B banks jump at the Pix launch?

**Input:** `ccs_flows_institution_month_mapped.csv` (CC3), PF relationships, 2016-01 to 2024-12, with conglomerate, `is_B` and the flagged event months.

**Build**, per conglomerate $j$ and month $t$: the opening rate

$$
o_{jt}=\frac{\mathrm{Openings}_{jt}}{\mathrm{Adults}_t-N_{j,t-1}},
$$

with $N_{j,t-1}$ the cumulated stock of relationships (opened and still active plus opened and later closed, less closures, anchored at the complaints-ranking count where CC4 found it reliable, else cumulated from 2016-01) and $\mathrm{Adults}_t$ the IBGE adult population.

**Estimate**, on B conglomerates, 2018-01 to 2022-12, excluding flagged event months,

$$
\ln o_{jt}=\theta_j+\theta_{\mathrm{month}(t)}+\beta_1\,\indicator\{t\ge 2020\text{-}11\}\,\indicator\{j\in\mathrm{Pix}_t\}+\beta_2\,\indicator\{t\ge 2020\text{-}11\}+\varepsilon_{jt},
$$

with $\theta_j$ conglomerate effects and $\theta_{\mathrm{month}(t)}$ calendar-month effects, and the same with event-time dummies around 2020-11 (twelve months each side) in place of the post indicator. Also split $\beta_1$ by mandated against voluntary joiner, and report the D firms' $\beta_1$ beside it for scale.

**Report:**

- $\beta_1$ and the event-time path, with conglomerate-clustered intervals;
- the implied $\ln a=\beta_1$ for B banks, which is the awareness term of (I7) of the design note;
- the 2020-04 to 2020-08 Caixa months handled as flagged, since the emergency-aid accounts are not an awareness event;
- a one-line verdict: did B banks' openings move at the launch, by how much relative to the D firms, and does it rise with the market's Pix use (Route 3), using the usage series of 7.1.2 aggregated to the conglomerate's 2019 deposit footprint.

**Flag:** openings also respond to the option value of a cheaper future move, so $\beta_1$ is an upper bound on the awareness term. A relationship opened to receive a Pix transfer is awareness in the sense that matters here, since it puts the bank in the choice set.



## Update 11-02j: task R1, the Pix participation roster at the bank-quarter level (Pedro, 11-02)

**Why.** The main thread's design for the switching cost (`Drafts/Deposit Competition/kappa_design_2026-11-02.md`, Section 3b) uses the Pix mandate at the **bank pair**: a saver at bank $j'$ can use Pix to move to bank $j$ only if both are on Pix. The mandate rule (more than 500,000 active accounts) instruments actual participation, which was a choice for the other banks. That needs, for every conglomerate in the panel and every quarter from 2020Q4, whether it was on Pix, and whether it was mandated. The pieces exist (`scrape_pix_roster.py`, `scrape_pix_participants.py`, `make_pix_timeline.py`, `shared/PIX/pix_mandatory_institutions.csv`). R1 audits and assembles them. **Nothing is estimated**, and no key is added.

**Deliverable:** `Rout/pix_participation_conglomerate_quarter.csv`, one row per panel conglomerate and quarter, 2020Q4 to 2024Q4, with:
- `on_pix` (1 if any member institution is on the roster at the quarter's end, or holds Pix keys that month);
- `first_on_pix` (the first quarter);
- `mandated` (1 if any member institution is in the mandated list with a roster date at or before the launch, the same 34 as `make_pix_exposure.py`);
- `source` (roster, keys, or both), and `is_B`.

Map institutions to conglomerates as the Pix-keys task did (`make_pix_keys_conglomerate.py`, with the PicPay override).

**Report:**
1. **Roster coverage in the first six months.** For each roster date in `pix_roster_history.csv` from 2020-11 to 2021-05, the number of institutions and the gap in days to the previous date. Where the Wayback captures leave a gap longer than a month, say so: those are the months where a join date is uncertain, and the key-based entry month (`pix_participant_entry.csv`) is the fallback.
2. **Agreement of the two legs.** For institutions in both, the share whose roster `first_seen` and key-based `entry_date` fall in the same quarter, and the largest disagreements.
3. **The control group's size.** For B firms, by quarter from 2020Q4: the number of panel conglomerates **not** on Pix, and the share of B savings balances (type 2) they hold. This is the variation the pair-level design uses. If it is near zero after 2021Q2, say so: the design then rests on the first two or three quarters.
4. **Mandated against voluntary joiners** among panel conglomerates: how many of each, and the distribution of the voluntary joiners' first quarter.
5. Any conglomerate whose member institutions disagree (one on, one off), with the rule you applied.

**Flags to keep visible:** the 2020 to 2022 rosters key on CNPJ only, and the modern ones carry ISPB; a bank absent from a roster PDF may be a parsing gap rather than an absence. Check against the keys leg before reading an absence as "off Pix".

**Local, light.** Existing outputs are read, not rebuilt, unless a roster month is missing from the cache, in which case say which months and ask before any Wayback download.

## Update 11-02i (after your reply of 18:12: SV2, XI2, E4 finding)

```
update_i[6]{item,state}:
  SV2,"done 18:08: analogue field dropped from IN_ROWS/OUT_ROWS; Panel A four columns on the five-column grid ('Law of motion' spans 4-5); build clean; make_paper_numbers.py changed in two lines (Panel A cross-check column, _tsv_cell multicolumn regex) so the six sleep.ssel.absorbed.* checks pass; 155 cross-checks match, 388 entries unchanged, sidecar byte-identical. Accepted: the make_paper_numbers.py lines were needed for the cross-check"
  XI2 method,"option 2: no draws of the sleepiness parameters are stored (the two-stage pickles keep AME draws only); phi-hat moved to the 95% ends of the national link band and by 20 of its 400 draws; logit benchmark, closed-form Berry and linear IV, gates pass (baseline equals the stored parquet, delta 7e-15, theta1 2e-8)"
  XI2 result,"E3 time deposits: phi-hat itself persistent (within-market slope 0.92), but the induced error e is not: within slope 0.11-0.17, pooled 0.23-0.32, 3-8% of xi-hat's variance; Panel B within slope moves 0.50-0.58 (baseline 0.57), pooled 0.70-0.76 (baseline 0.75). Mechanism: with phi near one the active balance is close to the net flow, so a slow phi error becomes a fast xi error"
  XI2 truncation,"nearly memoryless: P(truncated | truncated before) 0.50, P(truncated | not before) 0.40 (unweighted); 98.6% of cells ever, 6.9% always truncated"
  PHI_CAP,"the prep caps phi at 0.99 before forming (1 - phi); at the estimate the cap binds for 40% of B rows (14% by balance) in both routines; at the band's upper end for 84% (E3) and 97% (E4), which makes e large and persistent in E4's upper end only"
  E4 gdp_growth_yoy,"built per entity in sleep_est_e2.add_time_variables, 0 when the entity has no row four quarters back: 26% of type-5 rows (99.6% in 2020, 44% in 2021) and about 0.35% of other rows; E4's phi-hat then differs across products of one market-quarter in 8% of market-quarters (up to 0.028); E3 untouched. Decision for Pedro (below)"
```

**Main-thread reading.**
- XI2 settles the direction. The φ̂ error in ξ̂ is small and noise-like, so the Panel B text gets rewritten (blocks X1 and X2 of the paper plan): if anything, the slopes understate shock persistence.
- The cap of φ̂ at 0.99 is a construction fact the paper does not state, and it binds for many rows. It should be disclosed (task PC1, pending Pedro).
- The per-entity GDP growth is a construction error: GDP growth is a market fact, and an entrant's zero is not a value. It only reaches E4.

**Pedro's decisions (11-02, after 18:12):**

1. **E4 GDP growth: fix it, and cancel E4's random-coefficients jobs now (task E4G).**
   - Give Pedro the `scancel` lines for every queued or running E4 job of M3b: the E4 ladder stages and the E4 multi-start starts. Leave every E3 job alone.
   - Then propose the fix as a change summary, since it edits `sleep_est_e2.add_time_variables`:
     - build `gdp_growth_yoy` per market (or take the market's value for any entity without its own lag);
     - in the same pass, list any other variable in `add_time_variables` that is a market fact built per entity.
   - **Sizing step first (Pedro, 11-02), after the OK on the fix and before any cluster run.** Re-estimate only E4's sleepiness point estimates with the corrected growth, with no bootstrap and no bands, locally if feasible.
     - If it is not feasible locally, say why and give the cheapest cluster line. Ask before running it.
     - Report, against the R3 E4 fit:
       - every coefficient of E4's index (time block included);
       - mean φ̂ and its quarterly path;
       - the largest change in φ̂_mt across market-quarters;
       - the share of type-5 rows whose φ̂ changes by more than 0.001.
     - No keys and no paper text. Pedro reads it before the rerun is scheduled.
   - After Pedro's OK and once E3 is adopted, rerun the E4 chain: the sleep fit with its bootstrap and bands (cluster), the demand prep, the logit, and the random-coefficients ladder and multi-start under T1–T3.
   - Report the E4 keys that move (Table 4 column (IV), the E4 entry and demand keys) with sign and significance crossings.
2. **The φ̂ cap: disclose it with keys (task PC1).**
   - Confirm where `PHI_CAP` (0.99) enters: only `(1 − φ)` in the share denominator, or also the carried stock in (14-B)/(11-B), and in the D market.
   - Add four keys, exactly these names, family cluster: `demand.phicap.rows.E3`, `demand.phicap.rows.E4` (share of B demand rows at the cap) and `demand.phicap.dep.E3`, `demand.phicap.dep.E4` (balance-weighted share). E4's change once E4G reruns.
   - The main thread then drafts one sentence for Sec. 5.2 or the appendix.
3. `coding_reply_2026-11-02g.md` was removed on Pedro's instruction. The full XI2 tables stay in `Rout/xi_phi_error.md`.

**For Pedro, from your reply:** remove the subagent's stray scratch files.
```
Remove-Item -LiteralPath 'C:\Users\pedro\AppData\Local\Temp\ed.py','C:\Users\pedro\AppData\Local\Temp\ed2.py','C:\Users\pedro\AppData\Local\Temp\ed3.py','C:\Users\pedro\AppData\Local\Temp\ed4.py','C:\Users\pedro\AppData\Local\Temp\pn.log'
```

## Update 11-02h: task SV2, drop the analogue column of `tab_state_vector_map` (Pedro, 11-02)

**What.** Remove the column "Analogue in Egan et al." from Panel A of the state-vector table, in its header and in every row (the in-vector rows and the left-out candidates). Pedro applies the matching text edits (blocks A1, A2 and P21′ of the paper plan), which remove the column's three mentions in V_Friction.

**Where (as read by the main thread).**
- In `make_state_vector_table.py`, the row strings at l. 116 and l. 126, and the header at l. 153.
- The `IN_ROWS` and `OUT_ROWS` tuples (l. 45) carry the analogue as a field. Whether you drop the field or only stop printing it is your call; say which.
- The module docstring (l. 9) mentions the analogue. Update it.

**Layout.** Panels A and B share one five-column `tabularx`, and Panel B (Block, Adds, Condition number, Effective dimension, Largest VIF) keeps its five columns. With the analogue gone, Panel A has four. Choose the layout, for example two `tabularx` environments in the same minipage, or Panel A spanning the freed width. Requirements:
- the caption, label and position stay as they are;
- the standalone build is clean, with no overfull box;
- Panel B's numbers and the note are unchanged.

**Checks.**
- `make_paper_numbers.py` reads `tab_state_vector_map.tex` (its `TSV` constant, l. 1263). Confirm that its cross-checks still parse the table after the change, and that `tab_state_vector_map.json` is byte-identical.
- Report the diff of the `.tex`.
- This edits an existing script, but Pedro asked for the change, so give the one-line summary and proceed.

## Update 11-02g: task XI2, the persistence of errors in phi-hat (Pedro, 11-02)

**Why.** The new appendix text on the demand-residual table (block P19b of the paper plan) says that ξ̂ = ξ + e, where e is the error φ̂ passes into ξ̂ through active balances (14-B). Panel B's slopes of ξ̂ on its lag (within 0.528, pooled 0.759) are therefore a mix of the persistence of the true shocks and the persistence of e.
- A persistent e raises the slopes.
- A noise-like e lowers them.
- The text currently argues that e is persistent (φ̂ is a function of slow-moving states). It does not measure it. XI2 measures it.

**Three parts.** One new script, `diagnostics/diag_xi_phi_error.py`, and one output, `Rout/xi_phi_error.md`. Use E3 and time deposits as the headline, with the other cells of the ξ table if cheap.

```
xi2[3]{part,what,report}:
  (a) phi-hat itself,"AR(1) of phi-hat_mt within market (market-demeaned) and pooled, on the E3 phi-hat panel of the R3 sleep fit; and the share of phi-hat's variance that is between markets","slopes with WCB or cluster intervals, the between share"
  (b) truncation,"within B bank x type x market cells: P(truncated at t | truncated at t-1) and P(truncated at t | not truncated at t-1), the share of cells ever truncated and always truncated, with truncation as defined by (14-B)","the two transition probabilities and the two shares, balance-weighted and unweighted"
  (c) error propagation,"perturb phi-hat; rebuild active balances (14-B) and the logit benchmark's xi by its closed-form Berry inversion and linear IV (blp_logit.jl berry_inversion, same instruments and sample rules); e = xi(perturbed) - xi(baseline) on cells present in both","for e: the AR(1) slope within cells and pooled (same weights and pair rules as Panel B), var(e)/var(xi-hat), corr(e, xi-hat); and how the truncated share changes under each perturbation"
```

**How to perturb, in order of preference.**
1. Bootstrap draws of the sleepiness parameters, if the R3 sleep outputs store them. Then e has the sampling distribution of φ̂'s estimation error.
2. Otherwise, φ̂ moved to the lower and to the upper end of its pointwise band (the ts_link band files), which bounds the error rather than sampling it.

Say which one you used. Do not guess a file layout: if neither is stored, say so and propose the cheapest way to get draws. Ask before any cluster run.

**Why the logit and not the random-coefficients ξ̂.** The logit inversion is closed form, so each perturbation costs one linear IV, and it does not depend on the RC refit that is running. Under the logit, a change in active balances maps directly into δ, so e measures how φ̂ errors enter ξ in the simplest case. Flag this as an approximation in the report.

**How the answer feeds the paper.**
- If e's within-cell slope clearly exceeds 0.53, the "raises both slopes" clause of P19b holds, and the main thread can quote a key for it.
- If e behaves like noise, the clause flips: the slopes would understate shock persistence. The main thread rewrites the clause.
- No keys and no paper text until Pedro sees the report.

## Settled (record)

```
settled[14]{item,state}:
  Diagnosis,"the diagonal IFT gave a gradient pointing uphill (cos -0.91 to -0.99 against FD); neither rc4 nor the adopted ext1 is a minimum in either routine"
  Cause,"the approximation, not a sign error: the diagonal column error rises from 0.12-0.14 (outside share 0.8-1) to 0.89-0.90 (below 0.2), so cos >= 0.99 in the high-outside-share bin; the formula's sign is right in all three functions"
  B-D coupling,"confirmed in the synthetic test: B blocks by market, then a Schur complement on the D rows, against the dense N x N Jacobian (3e-8)"
  M1,"stage 1 passed at A and B in both routines (shares 4-5e-15, 10/10 FD contractions, cos(exact, FD) = 1.0, rel. diff. 4e-5 to 2e-4)"
  M2,"the exact Jacobian, in log-share form, replaces the diagonal one in compute_ift_gradient!, compute_ift_gradient_gpu! and compute_delta_jacobian_gpu; G3 (fd_check) reported on every fit. Verified on the cluster 11-02 (jobs 31667893, 31667895): cos 1.0, rel. diff. 1e-5 to 7e-5, engine SEs = FD SEs; condition numbers 303 (market blocks) and 1.8 (D Schur)"
  Uploaded,"blp_engine_gpu.jl 16261cde..., blp_exact_ift.jl f2a9ca21..., blp_diag_ext1_exact_ift.jl ae94bcc1..."
  Speed,"77-105 s per gradient evaluation on 6 CPU threads; Pedro (11-02) keeps the verified version; the faster rewrite is parked unused"
  SEs,"at the current (non-minimum) fits, SE(alpha) is 0.074 (E3) and 0.069 (E4) with the exact Jacobian, against 0.180 and 0.297 reported; theta2 SEs about half; alpha/SE would be -1.58 and -0.80"
  M4 (a),"12 demand.xi.* entries now pending; 50 pending-blp-ext1 entries in all; V_Friction quotes 17 distinct pending keys (22 occurrences), the 10-30 deck 25"
  M4 (b),"config/blp_status.toml and utils/blp_status.py: a plain make_paper_numbers.py keeps the BLP family pending; a missing file reads as pending; release is one explicit 'ok' at M5"
  M4 (c),"a 'Provisional' sentence (config/table_notes.toml [demand] provisional) closes the notes of Table 6, the _full table, blp_rc_E*, blp_compare_E* and tab_xi_diagnostics while pending"
  M4 (d),"post-run-refresh names every RC-based exhibit as provisional; three slide figures read the fit (wedge, demand coefficients, xi anova); the wedge figure reads value_full and shows no marker"
  Stale table,"blp_rc_E4_spec12.tex was older than R3 because runner step 6 builds only E3; rebuilt with --est 4; neither paper inputs it"
  M3 scope,"(b): sigma, rc2, rc3, rc4, ext1, ext2, extended per routine, then the six-start ext1 multi-start with a new EXT1_TAG; TR5 not redone. The current RC outputs must move to a dated backup first, or the ladder skips every stage"
```

## Assessment and answers

1. **M1 and point 2: accepted.** The bound cos ≥ √(1 − r²) settles it without a follow-up job. The monotone rise of r as the outside share falls is the prediction of the approximation story. Your explanation of how the gradient reverses (the θ₁ projection leaves a residual dominated by the badly approximated rows) is consistent with all of it.
2. **M2: accepted.** The verification is what was asked: the engine's own gradient and SEs equal FD at both points in both routines. The log-share scaling was the right fix for the conditioning. Keeping the verified version over the faster rewrite is Pedro's call and the safer one.
3. **The SE finding matters for the paper even before M5.** The abstract and the introduction say the price coefficient is "imprecisely estimated", and line 213 says the state premium "is imprecise". Both rest on standard errors now known to be overstated. The 11-02 plan removes those clauses until M5 (blocks T7, T8) and adds the fact to the provisional note (T6′). No new key is needed.
4. **M4: accepted as built.** The `incumbency.censored` key going pending with its entry is harmless. The table-note wording passes the style rules.
5. **Runner step 6 (`make_blp_rc_table.py --ests 3 4`):** approved by Pedro (Decisions). The only side effect is that the E4 per-routine table stays current.
6. **M3: three conditions before launch.**
   - **Acceptance thresholds, fixed in advance.** G3 is a report, not a gate, which is right during the ladder. For adoption, propose thresholds now, before any new fit exists, so the choice is not made after seeing the numbers. Two candidates: (i) the projected FD gradient norm below a stated tolerance, relative to its value at the old fits (0.144 to 0.521); (ii) no FD descent step lowering Q by more than a stated amount (say 1e-7). Pedro approves the numbers.
   - **Wall time and threads.** At 77–105 s per gradient evaluation, a stage costs that times its evaluation count. Size each rung from the evaluation counts in the old logs, and check it against `blp_stage_job.sh`'s 2-day floor. The exact solve runs the 36 quarters in parallel threads on 6 CPUs. If the partition allows more CPUs per GPU job, raising `--cpus-per-task` is a speedup without code change. Propose the number, and say what the QOS permits.
   - **The backup before launch.** Move the current E3 and E4 RC outputs, including the adopted ext1 files, to one dated folder. Leave `_superseded_ext1_E3_20261031_172418` and `_superseded_ext1_E4_20261031_172458` untouched. At M5 the report compares the new fits with these.
7. **Pedro's line in your §5 is no longer needed.** The 11-02 check measured the cost per evaluation directly (77–105 s).

## Tasks

The live list is the I table of Update 11-03d (I1 to I8). The table below is the record; RT1 is I6, SL1 is I7, PC1 is I5, M5 is in I4.

```
tasks[20]{id,what,when}:
  RT1,"now I6: rate-chain fallback audit, 186 flagged conglomerate-quarters (Update 11-03d)","now, read-only, then the change summary"
  SL1,"now I7: master source log: builder inputs catch up with CC1 and S4, then rebuild and diff (Update 11-03b)","now"
  R1,"DONE 11-02 20:15 by a side session (see Update 11-02k): the control group holds no savings","done"
  TED1,"DONE 11-03 (Update 11-03a): no cross-market variation in listed TED fees","done"
  AW1,"DONE 11-03 (Update 11-03a): no jump in B openings at the launch once trends are in","done"
  R1b,"DONE 11-03 (Update 11-03a): 2019 savings at non-mandated B banks were zero","done"
  SV2,"DONE 11-02 18:08 (see Update 11-02i)","done"
  E4G,"CLOSED 11-03: done differently by the IMPR session as FR1-B3, real per-capita GDP growth with t-1 timing (Update 11-03c)","done"
  PC1,"where PHI_CAP enters, and keys demand.phicap.{rows,dep}.E{3,4}, on the FR3 fits (Update 11-02i, 11-03c)","after FR3"
  M3a,"DONE 11-02 (see Update 11-02): thresholds approved, 6 CPUs kept, backup done, ladder queued","done"
  M3b,"CANCELLED 11-02 20:14 (FR0); superseded by FR4 on the 36-quarter frame, same T1-T3 thresholds and EXT1_TAG discipline, E3's archived ext1/ext2 among the starts (Update 11-03c)","superseded"
  M5b,"DONE 11-02 14:10 (see Update 11-02c)","done"
  P11,"DONE 11-02 14:10 (see Update 11-02c)","done"
  XI1,"DONE 11-02 14:38 (see Update 11-02d)","done"
  CCS1,"DONE 11-02 14:46 (see Update 11-02e)","done"
  CCS2,"DONE 11-02 16:21: ccs.pf.dshare_open_main.2024 = 0.704 (p0 70), family local; 388 entries, 0 other changes; detail keeps main-entity 2016 0.0993 and cross bounds 2024 0.6902/0.7598","done"
  SV1,"DONE 11-02 16:21: make_state_vector_table.py note replaced; only the note row of tab_state_vector_map.tex differs; tab_state_vector_map.json byte-identical. Main thread checked: block P21 carries every definition the old note had","done"
  M5,"at the adopted fits: SEs with the exact Jacobian; the L1 items (share- and row-weighted RC own-price elasticity, share-weighted mean alpha_i, share- and row-weighted means of each D-tilde, the share-weighted fraction with alpha_i >= 0 and its 5th, 50th and 95th percentiles); then set config/blp_status.toml to ok, rebuild Table 6, the per-routine tables, the xi table and the three slide figures, and report every changed key with sign and significance crossings","after FR5"
  Step 6,"done 11-02","done"
  XI2,"DONE 11-02 18:10 (see Update 11-02i)","done"
```

## Decisions

- **Runner step 6 (Pedro, 11-02): yes.** It becomes `make_blp_rc_table.py --ests 3 4`.
- `coding_reply_2026-11-01.md` and `coding_handoff_2026-11-01.md` were removed on Pedro's instruction (11-02).

## Pedro's lines

None open on this file (the ladder lines of 11-02 are stale since FR0). The IMPR session supplies the FR3 submission lines. The L1 files are uploaded after M5b, with the new md5s.

## Open questions for Pedro (not tasks)

1. The adoption of the corrected fits, once FR4 reports which pass T1–T3.
2. At M5: which elasticity the paper reports. The main thread recommends the random-coefficients elasticity weighted by active deposits (it matches "the deposit base responds" in the abstract), with the row-weighted one and the plug-in in the appendix.
3. The CCS appeal and the narrower request (Pedro: this week).

## References

- Conlon, Christopher, and Jeff Gortmaker. 2020. "Best Practices for Differentiated Products Demand Estimation with PyBLP." *RAND Journal of Economics* 51 (4): 1108–1161.
- Nevo, Aviv. 2000. "A Practitioner's Guide to Estimation of Random-Coefficients Logit Models of Demand." *Journal of Economics & Management Strategy* 9 (4): 513–548.
