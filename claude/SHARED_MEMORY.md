# Shared memory for every Claude instance on Pedro's machine

Read this before any task that touches the paper. Newest entries first. Entries older than the top one are the record of how things got here; where they conflict with the top entry, the top entry wins.

## 2026-11-04, 20:30. I14 landed (`0b9385b5`); I15 says the tails are a level above the CDI; frames ready for step 7 (main thread)

IMPR (e): interim frames ready 19:41 on `52ea6495` (`...\rebuild_1104\interim_q1b\demand_prep\`, 610,459 rows in the index frame; interim national mean φ within 0.01 to 0.02 of R3); I14 done, panel `0b9385b5410b5cb6ae4c884c32e4ef46` (ten Itaú and Banrisul quarters re-carried, 2.24 percent of type-4 weight; tails' weight 2016 8.33 to 0.66, 2020 6.33 to 1.39); I2 restaged on it (248 files; holds). I15 (diagnostic): 520 imputed tails at 52 banks, 378 with deposits, all calibrate's strip (path K); the excess is a level of 4.5 to 11 pp a year above the CDI that the imputed components do not remove, at foreign and wholesale banks (Safra, BNP, Citi, Société Générale, Rabobank) and mid-size fixed-rate issuers (BMG, Paraná, Pine, Master); the 0.5 to 1.5 ratio bites in 2018 to 2021 because the CDI was low. Decision 11-05: ratio rule (a) or a band in pp (b); IMPR asked for counts at +3, +4, +5 pp. Kappa told the frames' location (step 7 after its finish line, on its `panel_52ea6495` copy). Pix reopened (`pix_handoff_2026-11-04.md`): P6 rerun P3 to P5 on `0b9385b5`; P7 write the P3 first-stage numbers to `Rout/p3_numbers.json`. Design note at the thirty-third version (clock premium $\Delta^{c}$).

## 2026-11-04, 20:10. Keys T4 to T6 in; the paper plan now runs on keys (main thread)

Tables agent: five collectors in `make_paper_numbers.py` (`collect_k2b`, `collect_k0`, `collect_k2_world`, `collect_pix_adv`, `collect_pix_stakes`), 100 keys (`kappa.k2b` 67, `kappa.k0` 8, `kappa.k2.world` 2, `pix.adv` 19, `pix.stakes` 4), `make_k2b_pairs_table.py`, `tab_k2b_pairs.tex`. Sources: K0 and K2 on `d01b8bae`, P3 and P4 on `45dbfc58`; `kappa.k0.outside.menu_share.median` from `k2_numbers.json` (0.875); `pix.stakes.fs.*` read from `p3_note.md`'s table. Missing: `kappa.k2.world.awake_share` (asked of the Kappa agent). The paper plan (`paper_edits_2026-11-04.tex`, nine blocks pending: W5 to W7, F1 to F6) now prints every number through `\pn` except that one; the F keys of the stakes are integer-format keys (`\pn{}` without a format). Test on the 19:32 save: 0 errors beyond the biblatex stub, 0 overfull, 0 undefined, 0 missing keys, 88 pages.

## 2026-11-04, 19:35. Kappa: panel copy for step 7; --panel option approved (main thread)

I14 overwrites `processed/market_panel.parquet` in place, so the Kappa agent copied the `52ea6495` panel and its side file, read-only, to `...\904ee55d-...\scratchpad\panel_52ea6495\`. Pedro approved a `--panel PATH` option in `diag_k2_nls_simulation.py` (K0 untouched) and keying `read_menu`'s closed-form fallback on the frame's `demand_chars.market_panel_md5` metadata rather than the row count. Paper: Pedro applied W0 to W2; the Figure 2 blocks (F1 to F6, panels a, c, d kept) are in `claude/paper_edits_2026-11-04_fig2.tex`, tested with W3 to W7.

## 2026-11-04, 20:10. The paper edit plan for the frictions is delivered (main thread)

`claude/paper_edits_2026-11-04.tex`: eight blocks for V_Friction.tex (W0 renames the type function so $\kappa$ is the switching cost; W1 the clock sentence in Section 2 with `clock.t3` keys; W2 the introduction; W3, W4 the opening of the Disentangling section; W5 the new body, 6.1 Four Frictions in One Model, 6.2 The Net-Flow Equation with Pix, advertising, stakes, fees and sweeps, 6.3 Evidence So Far, then `\iffalse` over the old subsections; W6 the `\fi` before the Conclusion; W7 three appendices, The Tax Clock, The Net-Flow Equation, Simulation Evidence, with the clock table and a provisional K2b table under `\IfFileExists{tab_k2b_pairs.tex}`). Provisional numbers print red through `\PNpending{key}{value}` with the key names the tables agent is producing (T4 to T6). Test in the sandbox with a biblatex stub: every SEARCH unique, 0 errors beyond the stub's, 0 overfull, 0 undefined references, 0 missing keys, 88 pages. IMPR (19:20): I2 holds, I14 coded and runs after the interim fits, I15 ready; frames' location to be posted.

## 2026-11-04, 19:40. Tax-clock table and keys in; descriptive keys moved with the rebuild (main thread)

Tables agent T1 to T3 done: `make_tax_clock_table.py` (Code folder), `tab_tax_clock.tex` and `.md` (paper folder), `Rout/tax_clock_cost.csv`, `Rout/tax_clock.json`, `collect_tax_clock` in `make_paper_numbers.py`, 36 `clock.t3.*` keys; the (800, 180) cost is 0.57, the 10-15 report's 0.58 was rounded twice. Regenerating the macros also moved 64 `desc.*` keys because the descriptive tables were rewritten by the 11-04 rebuild's stage 5 (18:25); four are quoted in V_Friction, among them `desc.t1.spread4.B.mean` from $-0.04$ to $-1.13$, which Pedro must reread. T4 to T6 (keys `kappa.k2b.*`, `kappa.k0.*`, `pix.adv.*`, plus `kappa.k2.world.*` and `pix.stakes.*`) next.

## 2026-11-04, 19:00. Final panel `52ea6495`; I12 done; two rate decisions open; the paper gets the frictions tomorrow (main thread)

IMPR (d): panel `52ea6495c9195ef38b83bc40fb2014fa` (K0_PANEL_MD5=52ea6495) replaces `45dbfc58`; Q3, Q2 (re-centred, rows 1,170,207), I12 (455 conglomerate-quarters changed, 24 percent of type-4 weight, mostly Itaú and Banrisul reshaped; carried tails gone; 157 tails left, 153 imputed, 4 Itaú; no-rate quarters 113 to 323), I2 restaged (sha256 737f7cd4... code, d78d3cf4... data), Q1 interim fits running (end 20:00 to 20:30, sandbox `...\scratchpad\rebuild_1104\interim_q1b\`). Open for Pedro: Itaú's level in three semesters (2.35, 1.96, 3.61 times the CDI), and IMPR's one rule (measured imputed or benchmarked rate outside 0.5 to 1.5 times the CDI goes to the carry; reclassifies 29 percent of imputed B quarters). Decided 19:05: I2 holds; I14 applies the 0.5 to 1.5 rule to benchmarked rates only (new md5, restage); I15 is a diagnostic of the 520 imputed tails, rule decided 11-05 after the advisor meeting. Step 7 and the K1 inputs keep `52ea6495`; Pix reruns P3 to P5 on I14's md5 (fresh handoff then). Paper: Pedro presents to his advisor 11-05 morning; the main thread drafts edit blocks for 3.3 Four Frictions, 4.2 the net-flow equation, 5.x evidence, appendices D (tax clock), E (net-flow derivation), F (simulation); numbers as red provisional literals via `\PNpending{key}{value}` until the tables agent's keys (T4 to T6) land.

## 2026-11-04, 18:40. K2b final; item 1 approved for step 7 (main thread)

K2b (Kappa, 18:22): savings pair recovers level, change and dose in the fe and ideal worlds (largest errors 0.066, 0.016, 0.009); $\alpha$ $-0.117$ and $\alpha^P\approx0$ recovered; checking pair unusable (0.65 / $-0.30$ even in the ideal world; at its bounds with the dose term, se 30.9); time-deposit pair level and change within 0.05, dose off by 5 to 7 se; own-savings normalization diverges (D2 settled). K1 reports $\tilde\kappa$ by pair, savings headline only, no second dose estimate. Steps 3 to 6 running since 18:16, end about 21:00. Pedro approved the `read_menu` change (closed-form $\delta$ when the binary's rows differ from the frame's) so step 7 can read the interim frames; `load_demand`'s mtime check goes to I10. Design note at the thirty-second version.

## 2026-11-04, 18:10. Pix handoff removed (thread closed); K2b interim; I12 sent (main thread)

Tables agent opened (`tables_handoff_2026-11-04.md`, urgent): `make_tax_clock_table.py`, `tab_tax_clock.tex` (+ `.md`, `tax_clock_cost.csv`), keys `clock.t3.*` via `make_paper_numbers.py`; reply `TABLES_REPLY_2026-11-04a.md`. Label T for its tasks.

`pix_handoff_2026-11-03c.md` deleted: the Pix thread is closed; it reopens only for a 6-minute rerun of P3 to P5 if I12 changes the type-4 chain (then a fresh handoff). I12 (the ±10 pp tails: Itaú's quarterly shape, Safra's carry donors) is in the IMPR handoff, before Q3 to Q1 and the restage. Kappa, 17:55: fe-world dose fit savings 0.934 / $-0.484$ (se 0.032) / dose $-0.1415$ (se 0.005); ideal before-after savings 0.974 / $-0.490$ (se 0.008); checking pair at its bound in the fe dose fit and off in the ideal world (D1, weak scale identification); the fe gate passed (error 0.0013); the stored-world K2b dropped with a watcher that stops the job after step 6; ideal dose fit ends 18:30 to 20:00, steps 3 to 6 and the stop about 22:00 to 23:00. The tax-clock Table 3 (deposit_types_report_2026-10-15.md, Section 3.2) reproduces from the law's equations; it has no `.tex` table or macro key yet.

## 2026-11-04, 17:50. Final panel md5 45dbfc58; the Pix reruns in; a rate flag for IMPR (main thread)

The rate fix landed at 17:08; the final panel is `45dbfc58a5de4d25dd11fe38eee5fdd3` (the 16:00 `70241f4e` is superseded). Pix reran P3 to P5 on it: advertising 0.087 (rest), 0.062 n.s. (big five), IV 0.206, AR [0.056, 0.478], group placebo $-0.76$; the stakes S halved at the median (R\$50) and their instrument is now carried by the cost shifters ($F$ 988, 709 net of structure); `S_measured` populated (corr 0.996). Pix thread closed. Flag for the coding thread: 185 B conglomerate-quarters with national type-4 spreads beyond ±10 pp (153 imputed, 27 carried, 5 benchmarked; Itaú 2016 at 28 to 44 percent a year from the strip's quarterly shape; Safra carried at 24 to 66 percent because the carry's neighbours are extreme but inside the band); 9 percent of type-4 weight in 2016, about 7 in 2018 and 2020, none from 2021. Design note at the thirty-first version (Section 2 as one model with four frictions).

## 2026-11-04, 16:30. P5 accepted; the Pix thread waits only for reruns (main thread)

Kappa job: fe-world dose fit since 13:46, machine paging (42 GB committed on 32 GB). Ideal K2b ends about 19:00 to 20:00, reply follows; steps 3 to 6 about 3 h more. Stored-world K2b dropped (D3 amended, 16:30): step 2 already gives the A6 result. Step 7 reruns its inputs with `K0_PANEL_MD5=<final md5>` once the coding thread names the interim frames.

P5 (Pix): advertising instrumented by its own four-quarter lag, first stage $F$ 24.5, elasticity 0.21 against OLS 0.085, Anderson--Rubin set [0.06, 0.48]; the big-five IV is uninformative; the literal rivals' placebo cannot be run (quarter effects absorb it); the group placebo is $-0.79$ ($p$ 0.002), negative as the model predicts; no market-level version exists. Recorded in the design note (Section 4.3, Group C, record table). Pix thread open only for the P3 to P5 reruns at the final md5; reply next in `PIX_REPLY_2026-11-04c.md`.

## 2026-11-04, 16:15. The panel is rebuilt; what follows today (main thread)

The panel rebuild ran 14:37 to 15:04 (I1, I6, I8, I9, D1 to D10, N1 to N5): md5 `70241f4e`, judicial deposits out of `dep_a4` (down 13.7 percent; the old column kept as `dep_a4_incl_judicial`), seven new columns (`rate_a4_src`, `rate_a4_fallback`, `rate_a4_carried`, `rate_a4_vintage_src`, `judicial_*`), all audits pass, stage 5 regenerated 21 descriptive files in `Drafts/Deposit Competition`. Pedro's decisions on the reply: fix the exact-zero and exact-Selic rates now (rerun from the rates step; a final md5 follows and every agent pins it); re-center on the final frame (new `state_centering_means.json`); interim sleepiness fits and demand frames locally, in a sandbox, so K1 and K2's step 7 run before I2 returns; the T1 check reads `dep_a4_incl_judicial`. The I2 package is restaged after that; Pedro launches it. Bradesco 2016 at 1.09 × CDI (the consolidated-entity year) is recorded for I10. Kappa: K2b per pair recovers the savings level (1.006) and change ($-0.501$) in the fe world; job ends about 20:00. Pix: P3 variant and P4 done (advertising moves openings at the non-big-five banks, 0.086 and 0.14 over three quarters; the big five not distinguishable from zero); P5 running; P3 and P4 rerun at the final md5. Note: twenty-ninth version.

## 2026-11-04, 12:30. The BB session is closed; its record (main thread)

**What the BB agent delivered, 11-03 to 11-04 (tasks BB1 to BB6, B1 to B8, B-D1), now owned by the coding thread under I10.**

```
bb_record[9]{item,where,what}:
  Rende Fácil terms,"DATA_RAW/issuers/sweep_programs/raw/bb/lai_18882_002114/; sweep_products.csv rows BB-4, BB-5; bib key bb_rendefacil_termo_2022 in References.bib","opt-in sweep, V1 from 2022-03 (10/20/30/50/100% of the DI by holding period), V2 from 2026-05; instrument not named; items 3 to 5 asked again"
  Booking checks,"diagnostics/diag_bb_rende_facil_booking.py, diag_santander_sweep_booking.py; Rout/bb_rende_facil_*, santander_sweep_*","neither BB's nor Santander's sweep moves type-4 stocks or rates materially; Santander's default sweep shows at most in demand deposits (about R$2 bn)"
  Judicial deposits,"diagnostics/diag_judicial_deposits_cosif.py, diag_judicial_deposits_share.py; Rout/judicial_deposits_share.csv (588 rows from 2015Q4), cosif_415_subaccounts_monthly.csv, cosif_judicial_rates_2025.csv, judicial_deposits_report.md; DATA_RAW/issuers/judicial_deposits/ (DFP and ITR extracts with pages, Caixa's BRGAAP statements, CVM DFPs for BRB, Banestes, Banese); DATA_RAW/BCB/cosif_plano_contas/","Cosif 4.1.5.50 sits inside ESTBAN verbete 432 = dep_a4; shares of type 4: BB 0.51-0.76, Caixa 0.33-0.71, BRB 0.20-0.56 (re-anchored: 0.18-0.25 in 2018-21, step to 0.46 in 2022Q1), Banese 0.29-0.49, Banestes 0.21-0.31, Banrisul 0.12 from 2024Q2; the December 2023 precatório deposits put R$45-49 bn one-quarter spikes at BB and Caixa; the rate chain divided CDB expense by a stock that included them"
  Statement funding expense,"DATA_RAW/issuers/funding_expense/compiled/dfp_funding_expense.csv (eight banks, 696 rows), dfp_funding_expense_bb.csv (coding thread), subordinated_in_lump.csv, time_deposit_scope.csv, brb_deposit_balances.csv, NOTES.md; Rout/funding_expense_report.md","the inputs of D6 to D10; traps: Caixa's subordinated rows are a sub-line of bills_lf, Santander's external rows sit inside time_deposits, Caixa's IHCD is not in the Cosif lump, Banestes' balance-sheet 'A Prazo' includes judicial and special deposits, BRB's time-deposit line includes judicial deposits in every year, Itaú prints no time-deposit line, BB's 2018-19 savings line exceeds any lawful remuneration"
  Old-rule savings,"Rout/savings_old_rule_*","no source gives the old-rule stock by bank; the printed savings expense (Caixa, Banese, Banestes, BRB, Santander, Bradesco) replaces it in the rebuild; Lei 12.703/2012 applies only to deposits made from 4 May 2012"
  LAI 025439,"DATA_RAW/BCB/tarifas_lai_18810_025439/raw/parecer_juridico_1461_2026_pgbc.pdf, decisao_2a_instancia_2026-11-03.txt; NOTES.md","second-instance appeal granted 11-03; Deati has 30 days (about 12-03) to deliver the package client counts by institution or declare them nonexistent"
  Source log,"source_log_rows.csv in each folder; DATA_RAW/_extra_source_log_rows.csv; the master rebuilt by I7","every statement used has a row with its hash"
  Scratch to copy,"C:\Users\pedro\AppData\Local\Temp\claude\c--Users-pedro-OneDrive-Documentos-Yale-Year-3--2024---2025--Open-Finance-Open-Finance-Code-Egan-et-al-2025-Rep\3686646d-12ca-4ddc-b088-07f872fba1f4\scratchpad\bd1 and \b7","the before-and-after copies of the share file, the script backup and the B8 merge tools; temporary; the coding thread copies what it wants under I10"
  Next BB handoff,"with the CDB registry answers (025399, 025440, due 11-13) or BB's follow-up on items 3 to 5","none open now"
```

The BB handoff file was removed on 11-04 at Pedro's instruction; this entry is its record.

## 2026-11-04, 11:30. The estimator is settled; the rebuild's last inputs; run order (main thread)

K2 with a purpose-built solver: nonlinear least squares on the net-flow equation converges; the level of $\kappa$ is recovered in the ideal world, biased when one rival factor is shared across pairs, exact for the savings pair, lost when the scale moves within cells; the Pix change is recovered ($-0.503$ and $-0.455$ against $-0.5$). Decisions (Pedro, 11-04): per-pair rival factor with the savings pair as the headline; total-pool weighting; the stored-world K2b and steps 3 to 6 run now, step 7 after the rebuild. Rebuild decisions: N1 (ii), the 2025 scope ratio for lines that include more than CDBs; N2, BB's savings line excluded in 2018-19 only; N3, Santander 2024 carried from 2023; B-D1, BRB's judicial path re-anchored at R$3.0 bn on 2021-12. B8 is in (statement subordinated/hybrid/external lines, `subordinated_in_lump.csv`, `time_deposit_scope.csv`). Order on the machine: the Kappa job, then the rebuild (after the BRB fix), then step 7. Handoffs: `impr_handoff_2026-11-03.md`, `bb_handoff_2026-11-03.md`, `kappa_handoff_2026-11-03.md`, `pix_handoff_2026-11-03c.md`; replies from now on dated 11-04. Note: twenty-fourth version.

## 2026-11-03, 22:10. D10 adopted; the rebuild waits for B8 (main thread)

D10 (Pedro, 22:10): for statement banks the annual CDB expense is anchored to the statements' time-deposit line net of its scope items, Cosif giving the quarterly shape, with `rate_a4_src` recording the source; the component benchmark is the fallback. The BB agent has B8 (domestic subordinated and hybrid lines, `subordinated_in_lump.csv`, `time_deposit_scope.csv`, Itaú's line); the coding thread adds D10 and runs one rebuild after B8. 2025 is outside the sample and serves only as the calibration window.

## 2026-11-03, 22:00. B7 in; K2's gate failed and changed the design (main thread)

B7 complete (eight banks' statement funding-expense lines; judicial share file from 2015Q4), so the coding thread starts the panel rebuild (I6, I8, I9 with D6, D7, D9; loader rules in `impr_handoff_2026-11-03.md`, Update 11-03i). The BB agent has no open task. K2 (Kappa agent) step 2: on the paper's $\hat\delta$ and $\widehat M$ the outside option is 87 percent of the median menu (an awake real stays with probability 0.11; 84 percent of entry comes from the outside pool), the linearized regression recovers nothing about $\kappa$, and the multiplicative form recovers it to within 0.03 when the scale is bank $\times$ market plus market $\times$ quarter. The design note (twenty-third version) restates A2 as a common denominator (the outside-dominated case operative), drops quality weights, writes the exit as a market-quarter coefficient on the awake stock, and estimates by nonlinear least squares; next for the Kappa agent: a converging solver, then K2b (the Pix change simulated). The note records that the paper's gross flows (0.11 stay, 1.2 percent awake) and the data's net flows (9 percent of stock) disagree; the design uses the data.

## 2026-11-03, 22:50. Decisions on the rebuild; one BB task left before it starts (main thread)

Pedro decided (impr_handoff_2026-11-03.md, Update 11-03g): fallback rates by the max-Selic guard band then a carried rate, missing-and-drop as robustness (D1); judicial deposits removed proportionally in levels with a capital flag, `dep_a4` overwritten and `dep_a4_incl_judicial` kept (D2); savings rate from the BCB's credited yields, SGS 195 and 25 (D3); no hold for the old-rule share, which no source gives (verified: Lei 12.703/2012 applies only to deposits made from 4 May 2012, and withdrawals drain the new-rule balance first), `s = 0` flagged where no statement line exists (D4); the I9 rate base with DFP benchmarking adopted (D6); the bills-ratio scope error fixed (D7); the switch-cost tree copy of the market-size bug fixed (D5); request logs left alone (D8). **The rebuild starts when the BB agent's B7 lands** (the DFP funding-expense lines for Caixa, BRB, Banese, Banestes, Banrisul, Bradesco, Itaú, Santander; the 2015-12 judicial values; BB's 2023 to 2024 quarterly judicial stock). Pix: P1 to P3 delivered and accepted; nothing open but the P3 rerun at the new md5. Kappa: K2 running. I10 (Pedro, 11-03): once the other agents are done, the coding thread moves every adopted diagnostic into the panel build or the estimation routine (map in Update 11-03g). Handoffs: `impr_handoff_2026-11-03.md`, `bb_handoff_2026-11-03.md`, `pix_handoff_2026-11-03c.md` (one file now), `kappa_handoff_2026-11-03.md`. All reply files processed and removed.

## 2026-11-03, 21:30. Four agents; where everything stands (main thread)

```
agents[4]{agent,handoff,reply_to_write,open}:
  coding (I),"impr_handoff_2026-11-03.md","IMPR_REPLY_2026-11-03c.md","I1 market size, I6 fallbacks, I8 old-rule premium, I9 judicial deposits: four change summaries, one panel rebuild, new md5; then I2 (FR3) on Pedro's go, I3, I4, I5, I7"
  Pix (P),"pix_handoff_2026-11-03c.md (rules in pix_handoff_2026-11-03.md)","PIX_REPLY_2026-11-03c.md","P1 Pix-use dose and exposures at the MCA; P2 advertising and branches; P3 the stakes (rebuilt at the new md5)"
  BB (B),"bb_handoff_2026-11-03.md","BB_REPLY_2026-11-03c.md","B4 stage the 025439 second-instance decision; B5 old-rule savings stock by bank; B6 timing of Caixa's 2024 judicial fall. B1, B2, B3 done"
  Kappa (K, new 11-03),"kappa_handoff_2026-11-03.md","KAPPA_REPLY_2026-11-03.md","K2 the simulation test now (current panel); K1 the regression after the rebuild and P1 to P3"
```

The design note is at its seventeenth version (`Drafts/Deposit Competition/kappa_design_2026-11-02.md`, 37 pages): model (Section 3), why the reduced form (4), derivation (5), identification (6), tests (7), order of work (8), then the dead ends (10), the record of every task with its result (11) and the data requests (12). Judicial deposits: Caixa's 2016 to 2024 shares are in (0.33 to 0.71); the five downloads Pedro was asked for are not needed.

## 2026-11-03, 19:30. Labels rebooted; three findings from K0 (main thread)

**Labels (Pedro, 11-03).** Tasks are numbered by agent: **I** coding thread (`impr_handoff_2026-11-03.md`), **B** BB agent (`bb_handoff_2026-11-03.md`), **P** Pix agent's own tasks, **K** the switching-cost design, run by the Pix agent (`pix_handoff_2026-11-03b.md`). Map: FR3→I2, FR4→I3, FR5+M5→I4, PC1→I5, RT1→I6, SL1→I7; BB5→B1, BB6→B2, JD1→B3; SIM1→K2. New: I1, I8, I9, B4, B5, B6.

**Three findings from K0 (`Rout/k0_report.md`), each with an owner:**
1. **The market size $\widehat M_{mt}$ has its access index inverted** (I1, before I2): `build_market_size_and_shares` computes access on grand-mean-centred GDP per capita, so richer markets get the larger outside option (Spearman −0.992 in 2019), 56 percent of B markets sit at the 0.35 floor in 2023–24, and the median market's $\widehat M$ per capita jumps 6.7× from 2022 to 2023. Touches shares, $\hat\xi$, `demand.sharecap.*` and every counterfactual. Fix proposed: add the persisted grand mean back before the ratio. Pedro decides.
2. **The old-rule savings premium is booked as time-deposit expense** (I8 mechanics; B5 old-rule stock from the DFP notes): the panel's savings rate is the regulated new-rule rate for every bank; Cosif splits savings expense only from 2025-01; so in 2017Q4–2021Q3 a bank's type-4 rate is overstated by (old-rule share) × premium × savings/time deposits: per unit of share, Caixa 4.6 pp a year, BB 2.3, Itaú 1.6, Bradesco 1.5 (2018–2020). A paper-level fact about $\rho_{j4}$ of the big banks.
3. **Cell-level flows against the sleepiness fit:** the median type-4 cell moves 9 percent of its carried stock in a quarter while the adopted $\hat\phi$ leaves 1.2 percent awake, so outflows exceed the whole awake stock in 86 percent of outflow cells (the reason (11-B) truncates 40.5 percent of cells); 41 percent of the gross churn nets within the bank across municipalities (booking, not choice). The ratio design is free of $\phi$; the left side of K1 needs a rule (Pedro's decision).

Also: the rate fallbacks are 186 B conglomerate-quarters, not seven (I6); LAI 18810.025439's second-instance appeal was granted 11-03 (package client counts by institution; Deati has 30 days; staged as B4).

**4. Judicial deposits (B3, 20:00).** `dep_a4` (ESTBAN verbete 432) contains Cosif 4.1.5.50, the court-ordered deposits, for every bank and year: BB 51 to 76 percent of its type-4 stock, Caixa about 40 percent, BRB, Banese, Banestes 20 to 56 percent, Banrisul 12 percent from 2024Q2 (reclassification; its `dep_a4` jumps 16 percent for accounting reasons). The rate chain divides CDB expense (8113, judicial excluded) by the 4.1.5 total (judicial included) in the 2025+ rate and in the de-bias factor $k_i$ that rescales every pre-2025 rate, so BB's measured CDB rate is about half the true one. ESTBAN books the balances at the branch by the court, so local type-4 shares are distorted where courts sit. **Pedro (11-03): fix it.** Task I9: judicial stock out of `dep_a4` and out of the rate base; one panel rebuild together with I8 (old-rule premium) and I1 (market size), then I2. Caixa's 2016 to 2024 shares need its statements (B6, Pedro downloads five).

## 2026-11-03, 17:30. State of every workstream (main thread)

**Handoffs in force** (all in `Code/Egan_et_al_2025_Rep/claude/`; one file per agent; replies go next to them with the names given):

```
handoffs[3]{agent,handoff,reply_to_write,open_items}:
  coding (IMPR),"impr_handoff_2026-11-03.md (the master; replaces coding_handoff_2026-11-02.md and impr_handoff_2026-11-02.md)","IMPR_REPLY_2026-11-03b.md","FR3 (WAIT for Pedro's go), FR4, FR5, M5, PC1 after FR3, RT1 (rate fallbacks), SL1 (source-log builder inputs)"
  Pix,"pix_handoff_2026-11-03.md","PIX_REPLY_2026-11-03b.md","K0 (the objects of the net-flow regression; no regression)"
  BB,"bb_handoff_2026-11-03.md (one file)","BB_REPLY_2026-11-03b.md","BB5 (Santander sweep booking), BB6 (bib entry into References.bib), JD1 (judicial deposits inside Cosif 4.1.5)"
```

**Estimation.**
- **The RC-BLP engine had a wrong gradient:** the diagonal IFT Jacobian pointed uphill (cos −0.91 to −0.99 against finite differences). Fixed on 11-02 with the exact log-share Jacobian (`blp_exact_ift.jl`; `compute_ift_gradient!`, `compute_ift_gradient_gpu!`, `compute_delta_jacobian_gpu`), verified on the cluster (cos 1.0; engine SEs = FD SEs). Every RC-based number in the papers is **provisional** until refit: `config/blp_status.toml` is pending per routine, tables carry a "Provisional" note, and the "imprecisely estimated" clauses on $\bar\alpha$ are out of the 11-02 edit plan because the old SEs were overstated 2.4 to 4.3 times.
- **Adoption thresholds for any new RC fit (Pedro, 11-02):** T1 projected FD gradient norm ≤ 1e-3; T2 best dQ along the FD descent direction ≥ −1e-7; T3 se_status ok. Lowest Q among passing starts is adopted.
- **The 11-02 ladder (M3b) was cancelled (FR0)**; its outputs sit in `CLUSTER_IN/20261102/` as an archive. The full rerun from 2016Q1 is the FR plan: FR1 (sleep frame now 36 quarters, 2016Q1 to 2024Q4; funding-ratio instruments spliced to 2015Q4; real per-capita GDP growth with t−1 timing for E4; re-centring; per-routine BLP status) and FR2 (local sizing) are done; **FR3 (full cluster chain) waits for Pedro's go**; FR4 is the RC ladder and multi-start under T1 to T3; FR5 keys against R3, then M5.
- **FR2 findings to carry:** the φ̂ path tilts (2016–17 lower, 2023–24 higher), level averages unchanged; sign flips only in AMEs R3 found insignificant; E4's share of rows at φ̂ = 1 doubles to 33.4 percent (E3 16.8).
- Panel `d01b8bae` (10-27) is unchanged by FR1. The paper's window becomes 36 quarters.

**Paper text.** `V_Friction.tex` only; Pedro applies every edit. All blocks of `Drafts/Deposit Competition/paper_edits_V_Friction_2026-11-02.tex` are applied (save of 11-02 18:28). Held until M5: the Section 5.2 rewrites and the removal of the red provisional note. Held until BB3's follow-up: the 7.2 sweep sentence on BB. Numbers come from `\pn` keys only.

**The switching-cost design** (`Drafts/Deposit Competition/kappa_design_2026-11-02.md` and `.pdf`, fourteenth version, 11-03): the structural model is the paper's with four additions (source-bank state and $\kappa$; consideration probability; stakes in $\phi$; the tax clock), and the reduced form is a regression of the **net flow** into a bank's time deposits, $\mathrm{NF}_{j4mt}=\mathrm{Dep}_{j4mt}-(1+r_{t-1}-\rho_{j4m,t-1})\mathrm{Dep}_{j4m,t-1}$ (no $\hat\phi$, no truncation), on the bank's own time-deposit spread gap interacted with observed pool weights; $e^{-\kappa}$ is the slope on rivals' pools over the slope on own pools. Settled by data: R1 (no off-Pix bank holds savings, so the bank-pair design is dead), TED1 (listed TED fees do not vary where savings are, so there is no fee dose), AW1 (openings at B banks did not jump at the Pix launch: 0.15 [−0.24, 0.55] with trends), R1b (2019 savings at non-mandated banks were zero). The Pix change in $\kappa$ is read before-after with the within-bank margin as control, and Pix use by market (instrumented by 2019 mobile access) as the dose.

**Data facts found this week.**
- CCS: a relationship is not a person; D firms' share of PF openings 0.747 (0.704 main entity); key `ccs.pf.dshare_open_main.2024`. CC1 to CC5 done.
- BB Rende Fácil (LAI 18882.002114): opt-in sweep; V1 from March 2022 (10/20/30/50/100 percent of the DI by holding period), V2 from May 2026; instrument not named; follow-up on items 3 to 5 sent 11-03. Not material for BB's type-4 stock or rate (BB3). **Judicial deposits are 48 to 60 percent of BB's type-4 stock**, inside `dep_a4` and its rate (JD1 sizes them for every bank).
- Seven rate fallbacks in the 10-27 panel: `rate_a4_national` equals the Selic (Itaú 2020Q1, Q2, 2021Q1; Santander 2020Q1, 2021Q1) or zero (both, 2021Q2) with `rate_a4_guarded` 0 (RT1).
- The master source log `DATA_RAW/source_log.csv` is appended by hand since 10-30 (CC1, S4 edit, BB1 rows); do not rebuild it until SL1 has moved those into the builder's inputs.
- LAI status: 025400 answered and processed; 025439 granted (byte-identical to the public census); 023084 in use; 025656 denied twice, not appealed; 025399 and 025440 (CDB registry) due 11-13; 18882.002114 (BB) answered in part.
- Bib entry `bb_rendefacil_termo_2022`: the BB agent pastes it into `References.bib` (BB6) and reports the diff.

**Rules (unchanged):** Claude never edits the paper or the decks; Pedro owns git and deletions (agents give `Remove-Item -LiteralPath` lines); raw files stay untouched; Python is `C:/venvs/egan/Scripts/python.exe -I` with `MPLBACKEND=Agg`; one heavy process at a time, and read `market_panel.parquet` through pyarrow; summarize before editing any existing script and wait for Pedro's OK; timestamp every update; "wait" means stop.

## 2026-10-27, 17:45. E1 rebuild done, R3 running (coding reply of 17:40)

- **Panel `d01b8bae`** (was `0c47f30a`); every audit passes. **R3 = job 31020411** (Sleep + Logit + RC-BLP on `d01b8bae`, calendar lags).
- **E1 changes (Pedro's decisions):**
  - ESTBAN non-filing: gaps are filled from the same quarter (18 cells, including C6 2024Q1 and BTG 2019Q4); closures are exits (79 cells); no donor stays missing (43).
  - B/D rule: no retail-deposit test; judged on the 2024-12 cross-section. PagBank, C6 Consignado and Letsbank become D; Arbi becomes B.
  - Type-5 allocation: total-deposit or equal weights when a firm has no checking footprint.
  - The source tier is decided per firm over 2016–2024.
  - **PicPay is one firm (C0088022) in every quarter** (PC1 route 2).
  - Calendar lags.
- **Paper numbers refreshed with `--cluster-pending`.** 39 panel ids moved, and every V_Friction claim still holds. The `fees.*` keys are live (task FK done).
- **Paper text:** `Drafts/Deposit Competition/paper_edits_E1_2026-10-27.tex`, V_Friction only, 9 blocks:
  - the firm-type rule, ESTBAN non-filing, type-5 allocation, the source tier, PicPay as one firm, calendar lags;
  - the 7.2 fee paragraph, with three new bib entries;
  - the 7.2 footnote and a typo.
- **Entry paths:** the charts and the `entrygap` keys redraw after R3. There are now 67 D entry events: PagBank, plus Rodobens, Randon and Andbank, which Pedro kept although their "entry" is a switch of source at 2016Q1.

## 2026-10-27, 15:40. ESTBAN non-filing fix takes precedence

- **The bug:** where a bank did not file a raw ESTBAN month (`AGEN_PROCESSADAS = 0`, blank balances), the panel reads zero deposits. C6 drops out in 2024Q1 (about R$24 bn) and BTG Pactual in 2019Q4 (about R$9 bn), plus 1–27 single cells per quarter (coding reply of 15:35).
- **Pedro is fixing it now, and it comes before every other task.** The fix fills from the nearest filed month in the same quarter, separates real closures from non-filing, and makes sure a missing quarter is not read as an exit. Then one rebuild and R3.
- **R2 (job 30985673) was stopped by Pedro (15:45).** The next cluster run is R3, on the fixed panel. Until R3 returns, the cluster-family numbers keep their old values in red, in V_Friction and in the 10-30 deck.
- **Other results of the 15:35 reply:**
  - P1 done: `Rout/pix_keys_conglomerate_quarter.csv`. D firms passed B firms in PF Pix keys in 2023Q4 (0.364 to 0.549 of PF keys).
  - FP3 was applied on 10-26.
  - The TR5 intervals stay comparable to Table 4: wild-cluster bands exist only for E1/E2, and the bootstrap stays on.
- **PicPay: checks first (Pedro, 15:45).** PC1 runs before the E1 rebuild. Pedro then picks the route, and a re-map would join that rebuild.
- **PagBank (C0084813) should be D (Pedro, 15:45).**
  - It is a digital bank without a branch network, so its B flag looks like a rule artifact.
  - The coding agent reports the classification rule and similar cases. A reclassification would join the E1 rebuild.
  - It also checks whether B firms with prepaid balances but no checking footprint lose those balances when type 5 is spread across markets.

## 2026-10-27. Pedro's decisions (recorded by the main thread, 14:00)

- **Cost estimation is out of scope for the near future (Pedro, 10-27).** Policy function, BBL and counterfactuals stay on the old panel, and nobody re-runs or audits them.
- **Paper text: V_Friction only.** Edit plans target `V_Friction.tex`. V_Full gets no more text edits unless Pedro asks. The type-5 text is applied in both papers (10-27). It quotes `type5.rules.*`, `slide.types.prepaid.*`, `desc.t1.spread5.D.mean` and `slide.motiv.dpremium.cdi_balance_share`, so keep those key names.
- **The handoffs of 10-21 to 10-26 were removed by Pedro** (coding, scraping, fees). The main thread writes new ones when needed.
- **R2 is running on the cluster** (Sleep + Logit + RC-BLP on the rebuilt panel, `--no-bbl --no-cf`). Until it returns, the cluster-family numbers print their old value in red.
- **Pix keys by participant: YES, build the intensity variable.**
  - Source: the scraping agent's S13 (`DATA_RAW/BCB/pix_participants/`, `make_pix_keys_conglomerate.py`).
  - PicPay's keys (ISPB 22896431) go to C0080903 for 2020-11..2024-09 and to C0088022 after, matching where the panel holds PicPay's balances (type-5 decision D3).
  - Use: V_Friction Section 7.1.2, the convenience channel.
- **Tesouro Direto as a local outside option: cheap check first.** Compare the registry (`make_tesouro_mca.py`, by market and quarter) with B3's state points (2019-12, 2020-03, 2023-09). Nothing reads it until Pedro sees the check.
- **TD1 result (coding agent, 15:23; note deleted after assessment).**
  - **Outputs:** script `diagnostics/diag_tesouro_vs_b3.py`; files `Rout/tesouro_vs_b3_state.csv` and `Rout/tesouro_vs_b3_national.csv`.
  - **Cross-section: the registry ranks states like B3.** Spearman 0.95–0.99; Pearson per 1,000 inhabitants 0.98–0.99, at 2019-12, 2020-03 and 2023-09.
  - **Level and trend: it does not track B3.** It is 4–6 times B3 by state, highest in AP and TO and lowest in DF, RS and RJ.
  - **The registry is cumulative since 2002 and never drops inactive investors.** Its ratio to the active TD investors of B3's RI series rises from 4 (2016) to 10 (2024). Growth correlations are only 0.43–0.64.
  - **Main-thread reading:** usable as a cross-sectional, pre-period measure (for example 2019 registrations per capita by market). Not usable as a time-varying covariate in levels.
  - **A time-varying measure of active investors would need the Tesouro Transparente "Operações do Tesouro Direto" files** (2014–2026, monthly, about 1.1 GiB compressed), joined to the registry's state and city by investor code. Whether that file carries the investor code is to be confirmed.
  - **Open bug:** at 2023-09 the MCA output totals more than the investor-level count (25.34M vs 25.11M), cause unknown. Fix it before any use of `make_tesouro_mca.py` output.
  - **PARKED (Pedro, 10-27).** No further Tesouro Direto work until Section 7 uses an outside option. If it does, build active investors from the operations files rather than using the registry.
- **B3 investors by state: dropped.** Public B3 sources give only three state points with mixed units. The national series stay as context only.
- **Fee keys: YES (Pedro, 14:40).** Ten `fees.*` keys support V_Friction 7.2: fees are a holding cost at incumbents and near zero at digital banks.
  - The Fees agent writes `BCB/Tarifas/processed/fee_paper_numbers.json` (`claude/fees_handoff_2026-10-27.md`).
  - The coding agent adds `collect_fees` (its task FK).
  - No LAI for BCB complaints by type. The complaint tables are not joined to the panel yet. F6 is skipped and F5 stays on hold.
  - **Done 15:25; the fee workstream is closed.** Sidecar values:
    - POF: 27.7% of families pay; R$38.19 a month per paying family (2018 prices);
    - fee disputes over account complaints: big five 0.249 (2016: 0.320, 2024: 0.122), against Nubank, Inter and C6 at 0.031 (C6 is 440 of the 723 disputes, after its conditional fee from 2023-04);
    - big-five median listed Package I: R$11.55 (2016) and R$15.95 (2024), nominal.
  - The Fees agent's record is `DATA_RAW/_requests/fees_2026-10-23/STATUS.md`.
- **Fees agent, state at 14:00:**
  - Every task is closed except F5 (on hold), F6 (needs Pedro's browser downloads) and a possible LAI for BCB complaints by type.
  - Outputs are in `Open-Finance/BCB/Tarifas/processed/`; the status file is `DATA_RAW/_requests/fees_2026-10-23/STATUS.md`.
  - The consumidor.gov.br complaints are now mapped to 85 conglomerates (94.7% of complaints). Panel-ready side tables exist by conglomerate × quarter and by conglomerate × municipality × year.
  - No keys, no paper text, no pipeline change.
- **Scraping agent:** no tasks left. It waits on the open LAIs (CCS 18810.025400 and its Ouvidoria, 025399, 025440, BB 18882.002114) and Pedro's tariff follow-up (NUP not recorded).
- **K8 (Pedro, 10-27): keep the national g fallback as headline.** After R2, the firm-type fallback goes in a footnote as a robustness check, quoted through keys.
- **PicPay, main-thread proposal (not yet decided):**
  - R2 runs on the current panel.
  - Three cheap local checks: PicPay's share of type-5 balances; whether C0088022's 2022Q3 appearance is a D entry event; the 2022Q3 demand residual of C0080903.
  - Then either keep the data with an appendix sentence, or re-map PicPay as one firm (C0088022 in every quarter) before TR5 is staged, with a re-run of the main estimates.
  - Defaults until then: keep the 102% rule on C0080903's wallet balance, and leave the 2022Q3 switch as data.
- **Housekeeping done (main thread, 14:30):**
  - Temp: `fee_probe`, `scrape_FEE`, `scrape_T5`, `scrape_E2`, `edel_check` and `findex_check` deleted;
  - the empty `Code/Egan_et_al_2025_Rep/other/` and `Rout/followups_cache` deleted;
  - `DATA_RAW/issuers/posted_rates/evidence/iti_termo_20201125.html` (a "page not found" capture) deleted. Its source-log row still says "29 distinct files saved", so it is now 28.
  - The BCB zips had already left Downloads.

## 2026-10-21, later. The code now follows the rename

- **Figures and tables serve both papers.** `V_Full.tex` and `V_Friction.tex` input the same exhibits (8 figures, 24 `\input` files, `paper_numbers_macros.tex`), all flat in `Drafts/Deposit Competition` with no `\graphicspath`. Every generator writes each exhibit once into that folder, never per paper.
- **Which paper the scripts read is ONE setting: `Code/Egan_et_al_2025_Rep/config/paper.toml`** (`paper = "V_Full"` now), read through `utils/paper.py`. It is Pedro's choice; do not change it without asking him.
  - The setting covers `make_paper_equations.py`, `make_paper_numbers.py` (the routine enumerate order, which V_Friction shares) and `diagnostics/report_entry_paths_estban.py`.
  - For one run: `--paper V_Friction` on the equations and report scripts, or `DEPOSIT_PAPER=V_Friction` for all three.
  - `paper_equations_macros.tex` is regenerated, so it names V_Full.
- **Still naming V_Main:**
  - comments and docstrings ("V_Main eq 16" means the same equation in V_Full);
  - the hash-pinned files `bbl_solve.py`, `make_bbl_cost_tables.py`, `utils/sleep_links.py` and `utils/switch_cost_sleep_links.py`, left alone so the bundle checks keep passing;
  - the deck line `\PEQpaperpdf`, which is Pedro's (step 12 of `deck_edits_2026-10-30.tex`).
- **The lit-review subfiles** (`sec_lit_review_full.tex`, `sec_lit_review_friction.tex`) are Pedro's to finish.

## 2026-10-21. The paper file was renamed, and a new working paper exists

- **`V_Main.tex` is now `V_Full.tex`**, in `Drafts/Deposit Competition`. It is the full paper, including the cost estimation, the Egan et al. (2025) replication for Brazil, and its extensions.
  - Any path, script, macro or subfile that points to `V_Main.tex`, `V_Main.aux` or `V_Main.pdf` must now point to `V_Full`.
  - Do not rename or delete `V_Full.tex`.
- **`V_Friction.tex` is new**, in the same folder. It is where Pedro works from now on. Edit plans for paper text target `V_Friction.tex` unless Pedro says otherwise.
- **Known references to the old name**, as of 2026-10-21. These need updating, by Pedro or by the agent that owns each file:

```
refs[4]{file,line,reference,owner}:
  Drafts/Deposit Competition/paper_equations_macros.tex,2 and 28,"generated from V_Main.tex/.aux, \PEQpaperpdf{V_Main.pdf}",coding agent (make_paper_equations.py)
  Drafts/Deposit Competition/sec_lit_review.tex,1,"\documentclass[V_Main]{subfiles}",Pedro (decide V_Full or V_Friction)
  Drafts/Presentations/Deposit Competition/2026-10-30/Presentation-2026-10-30.tex,72,"\PEQpaperpdf{../../../Deposit Competition/V_Main.pdf}",Pedro (deck edit)
  Code/... make_paper_numbers.py, make_paper_equations.py, and any script that reads the paper's .aux or .tex,—,unknown until checked,coding agent
```

- **Unchanged rules:**
  - Claude never edits the paper or the decks. Pedro applies every change from edit plans.
  - Numbers come from `\pn` keys only.
