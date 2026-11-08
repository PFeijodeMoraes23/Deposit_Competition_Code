# Shared memory for every Claude instance on Pedro's machine (condensed)

Read before any task. The logs of 10-21 to 11-05 were condensed into this file on 2026-11-05 (19:48) at Pedro's request, and the full logs were then deleted. Detail lives in the files each item points to. New entries go in a log section at the end of this file, newest first, each with its `date` time, and a newer entry wins over the condensed text.

## 1. Rules and conventions (binding)

- Notation (11-07): time-deposit parts are written $4^{\mathrm{m}}$ (matured) and $4^{\mathrm{d}}$ (daily liquidity), e.g. $Q^{4^{\mathrm{m}}}$.

- Claude never edits `V_Friction.tex`, `V_Full.tex` or the decks. Pedro applies every change from search-and-replace edit plans (each SEARCH unique, the plan test-compiled first). Only the main thread edits the design notes. Agents deliver code, outputs and reports.
- Edit plans target `V_Friction.tex`. `V_Full.tex` (formerly `V_Main.tex`) gets no text edits unless Pedro asks, and is never renamed or deleted.
- Numbers only through `\pn` keys from `paper_numbers_macros.tex`. F statistics are integer keys (`\pn{}`). Citations only through keys in `Drafts/References.bib`.
- Destructive commands are shown and confirmed first. Pedro owns git and deletions: agents give `Remove-Item -LiteralPath` lines. Only the main thread deletes in `claude/`, with Pedro's OK. Raw files stay untouched.
- "Wait" means stop. Timestamps come from the shell `date` (Eastern).
- No edit to an existing script without a one-paragraph plan and Pedro's OK. Post the LF md5 of every edited script. New scripts go in `diagnostics/` or `kappa/`.
- Never guess a schema: read the file or ask. Do not read datasets over 50 MB. Code reads `market_panel.parquet` by pyarrow column subsets, never `load_panel_cached`.
- Python `C:/venvs/egan/Scripts/python.exe -I`, `MPLBACKEND=Agg`. One heavy process on the machine at a time. Cluster jobs only on Pedro's launch lines. One panel md5 per run, posted as the end line and carried by every output row.
- `config/paper.toml` (now `paper = "V_Friction"`) is Pedro's setting: never change it without asking.
- Leave alone: the hash-pinned files naming V_Main (`bbl_solve.py`, `make_bbl_cost_tables.py`, `utils/sleep_links.py`, `utils/switch_cost_sleep_links.py`) and the deck's `\PEQpaperpdf` line. The lit-review subfiles are Pedro's.
- Keep the key names the type-5 text quotes: `type5.rules.*`, `slide.types.prepaid.*`, `desc.t1.spread5.D.mean`, `slide.motiv.dpremium.cdi_balance_share`.
- Two live reply files only (Pedro, 11-05 19:37): `IMPR_REPLY_<date><letter>.md`, one combined file in which IMPR writes its own lines and those of the Kappa and tables agents, newest on top, and `MAIN_REPLY_<date>.md`, the main thread's single file of what is in force. Old replies are deleted once recorded here. `claude/README.md` indexes the live files.
- TOON for structured data between agents. Task labels: I IMPR, K Kappa, T tables, P Pix, B BB, S data search. Decision labels (D1...) restart with each set, so cite their date.
- Prose: American English, no dashes separating ideas, no semicolons, acronyms defined at first use, short sentences. Note front matter: one date line and the version number.
- Standing instruction (Pedro, 11-05 20:06, "yes, always"): whenever the paper's notation or the edit plan changes, the main thread aligns both design notes with it and re-renders their PDFs, without asking.

## 2. Where things are

`OF` = `C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\`. `Code` = `OF\Code\Egan_et_al_2025_Rep\`. `Paper` = `OF\Drafts\Deposit Competition\`.

| What | Where |
|---|---|
| Papers, exhibits, macros | `Paper\` (`V_Friction.tex`, `V_Full.tex`, flat exhibits shared by both, `paper_numbers_macros.tex` from `Code\make_paper_numbers.py`) |
| Edit plan | `Code\claude\paper_edits_2026-11-04.tex`. The K1 equation is eq:reg of the switching-cost note and is written out in `MAIN_REPLY_2026-11-05.md` |
| Notes | `Paper\kappa_design_2026-11-02.md` (switching cost, v40), `Paper\attention_awareness_note_2026-11-05.md` (awareness, v18), PDFs beside them |
| Coordination | `Code\claude\`: `MAIN_REPLY_2026-11-05.md` (in force for the threads), `impr_master_brief_2026-11-05.md` (rules §0, I10 map §2), `IMPR_REPLY_2026-11-06a.md` |
| Outputs | `OF\BCB\Egan_et_al_2025_Rep\processed\ESTIMATION_OUTPUT\Rout\`. Panel `processed/market_panel.parquet`, overwritten by each rebuild |
| Code | `Code\kappa\k1_netflow_nls.py`, `Code\diagnostics\diag_k2_nls_simulation.py`, `Code\config\blp_status.toml` |
| Cluster | Bouchet. `blp_run.sh`, `cluster_lib.sh`, `cluster_archive.sh`, `cluster_ingest.py`, `pipeline_all.sh`. Project path not recorded |
| Scratch to keep | `C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\`: `21a30b32-...\scratchpad\rebuild_1104\` (I2 package, `I10_plan.md`, `0b9385b5` copy, until I10 ends), `904ee55d-...\scratchpad\panel_52ea6495\` |
| Raw data | `OF\DATA_RAW\` (`README.md` index, `source_log.csv` append only, 403 rows), requests in `DATA_RAW\_requests\` (brief `awareness_2026-11-05\BRIEFS.md`) |
| BB record | judicial deposits and funding expense in `DATA_RAW/issuers/`, `Rout/judicial_deposits_share.csv`, `Rout/funding_expense_report.md`. Scratch `...\Temp\claude\c--Users-pedro-...-Egan-et-al-2025-Rep\3686646d-12ca-4ddc-b088-07f872fba1f4\scratchpad\bd1`, `\b7` (temporary, IMPR copies under I10) |
| 11-05 data | `DATA_RAW/Datafolha/top_of_mind/`, `ANATEL/broadcast_register/`, `ANATEL/coverage_municipal/`, `IBGE/censo2010_bens/`, `issuers/digital_reach/inter/`, `GoogleTrends/search_interest/` (documentation only) |
| ANBIMA request | `Code\claude\anbima_request_draft_2026-11-05.md` |

## 3. Current state by thread

**Replies.** `IMPR_REPLY_2026-11-06a.md` is the one live reply for IMPR, Kappa and tables. The 11-04 and 11-05 reply files are gone.

**IMPR**
- Panel in use: `fa38794d`. I2 BLP driver 32206965 (after the 8 to 6 CPU fix). Gates G0, G1, G2, G4 passed.
- ext1: E1 and E3 passed, their ext2 run as 32207095 and 32207101. E2 (32207097) and E4 (32207103) are in random starts. `blp_zip` 32207106 and `pipe_download` 32207107 follow (1 to 2 days). AME bootstraps 32180473/4 and `sleep_gate` 32180475 wait for a Turin node.
- Early pull of the demand frames for K1: job 32235273. Pedro downloads zip, `.sha256` and `sha256SUMS` into a fresh folder, then `cluster_ingest.py --family demand_prep`, dry run first.
- I10: plan read-only, one item at a time with Pedro's OK, after I2 and the Kappa runs. It includes ext1's failed line search (cold and random starts end at θ₀).

**Kappa**
- K2 complete (steps 2 to 7, K2b, K7, the two-start check).
- K1 `kappa/k1_netflow_nls.py` LF `761a02cf` passed tests 1 and 2 before the respecification. The `fit_recorded` SE fix stands.
- Now: re-code to the respecification and the 19:45 amendment, then `py_compile`, test 1, test 2 (regenerated truth), 50-market smoke with and without `--lambda-one`. Then K0 rerun on `fa38794d`, K1 on real data, the 199-draw refit on Bouchet.
- Flags: `nls_fit_k1` copies `kn.nls_fit`'s loop (drift risk). The headline moves through the pool assignment as well as the exit term, and the full-sample run measures it.

**Tables**
- Done, 0 MISMATCH: T10, T10b, T11 (`desc.gap4.*`), Nubank basis, T12 (`kappa.k2.sort|outside.*`), T13 (five row labels, in the macros).
- T9 code ready (19:43): `make_k1_table.py` `b321d2f0`, `collect_k1` in `make_paper_numbers.py` `491a44da`, guarded to a real run on `fa38794d`. Keys as proposed by the tables agent and accepted by the main thread: `kappa.k1.{sav,chk}.{change,level,dose}` (with `.lo`, `.hi`, and `.p` for change and dose), `kappa.k1.theta_w`, `kappa.k1.{n,markets,G,converged,panel}`, `kappa.k1.rob.{carried_kept,zero_premium,no_negative_adv}.sav.change`, `kappa.k1.l1.{sav,chk,td}.{change,level}`, `kappa.k1.l1.ln_Ec`, `kappa.k1.l1.{G,converged}`. No key for the headline time-deposit pair.

**Closed.** Pix (11-05), BB (11-04), data-search round `awareness_2026-11-05`, fees.

**Data**
- Registry 18810.025399 (λ and ω by bank) and 025440 (CDB issuance and rates by issuer): due 11-13. Then a handoff, λ from 025399, T10b rerun on 025440.
- LAI 025439 (package client counts): granted on appeal 11-03, answer by about 12-03. BB 18882.002114: appeal answered 11-06 (instrument, opt-in, predecessors); second-instance appeal for the predecessors' dates and opt-in status sent 11-06, answer pending.
- ANBIMA request (named-bank answers) sent 11-05 to faleconosco@anbima.com.br, reply pending.
- Closed routes: 025656 (Pix by participant) denied twice, not appealed. 025400 answered: the BCB refuses new CCS crossings.

**Paper** (save of 19:08 read)
- Pending in the edit plan: U1 (6.1 opening without b, replaces C1), W5 (b) (Section 6 body, after U1), W7 (three appendices, `tab:k2_breaks`), S2, S4, S5 (Section 2.3), F7 (Figure 2 in one row), C2 (typos). Last test: blocks unique, keys resolve but T13's five (now in the macros).
- After applying: compile with biber, then `make_paper_equations.py`.

**Notes** (b-free, rendered without errors)
- Switching-cost note v40 (44 pages): the four-friction model, the derivation to eq:reg, assumptions A1 to A9, identification (Pix moves κ, advertising moves A, stakes move φ), tests, order of work, task record, LAI requests, notation.
- Awareness note v18 (27 pages): the model without κ, identification, estimation with the net-flow check eq:nf, simulation evidence, the Pix launch, data, hand-collected data.
- Symbols: κ̃^pre and κ̃^post (Pix), χ_jmt (multi-homing share), ν^A_ij with loading σ_A (awareness factor).
- The render header (`/tmp/kd`) is lost on restart and must be rebuilt.

## 4. Binding decisions in force

**Model and K1**
- 11-05 15:58 D1: the outside option does not pay the clock. Staying is exempt.
- D2: K1 has a λ = 1 row (`--lambda-one`), where ln E_c = κ̃₂ − κ̃₄.
- D3: the headline is the Pix change. The level of κ̃ is an upper bound (A ≤ 1, A8, A5, A6, up to the A1 and A2 errors). Say that h may move at the launch and that the change was never tested under random coefficients.
- D4: the paper's friction list (inattention, unawareness, switching cost, preferences). The clock sits in the switching-cost block.
- 11-05 19:45: the savings forfeit is out of the model, with no b_{k'} anywhere. `Q4d` at weight 1, `R4d` at `e^{-k_td}`, `a` a nuisance scale for checking money only. The text states the forfeit (`brasil_lei8177_1991`, `brasil_lei12703_2012`) and says it cancels under A2. The 18:10 exact-scale amendment (`e^{a}` on `Q4d`) is superseded.
- Respecification: matured money joins checking (own and rivals'), exit `e^{theta_w} Q4m + e^{vartheta_mt} Q4d`, no `log_Ec`. The headline row (λ≈0) cannot estimate the clock.
- 11-05 12:30: full window 2016Q1 to 2024Q4. Carried-rate rows out (`carried_kept` row). Advertising on observed quarters, no zero fill. Score wild bootstrap locally, refit on Bouchet. Stakes elasticity from the sleepiness step. λ from statements for B (B median if missing), 1 for D, until 025399.
- 11-05 18:35 choices (a) to (h) as recommended: (c) the statement of year y applies to y+1, (e) controls as Post × control, (f) first stage on measured quarters, (g) negatives clipped to 0, (h) bank × state and D destinations not coded.
- 11-04: one κ̃ per pair, savings the headline, total-pool weighting. Awake share 0.0177, printed as 1.8 percent.
- Spec 16:05: A2 rests on outside-option dominance only, never "the stay option dominates", and `outside_entry_share` is not support. The level comes from within cells, never "cross-sectional". No η in the reduced form. α^P is a price shift, not attention. The legal test compares Ê_c with E[e^{ᾱΔ^c(s,H)}]. The bootstrap holds ω, λ and M̂ fixed.

**Data and panel**
- 11-05 10:00 (I17): an imputed type-4 rate more than 4 pp above the CDI or below half of it goes to the carry, for B and D. The Pix placebo uses 99 shuffles.
- 11-04 (I14): benchmarked rates outside 0.5 to 1.5 times the CDI go to the carry.
- In the panel: `dep_a4` excludes judicial deposits (`dep_a4_incl_judicial` kept), PicPay is one firm, 2025 is calibration only.
- 11-05: Nubank's posted basis is 100 percent of the CDI.
- 11-05 13:05: the B versus D gap is quoted only with intervals (D indicator with quarter effects, conglomerate clusters, wild bootstrap with Webb weights, G and G* reported), no p-values.
- Table 2's `desc.t2.spread_ann_a4.*` is D's post-Pix change, not a B versus D gap.
- Google Trends stopped (descriptive only). The BCB research route is deferred. Tesouro Direto is parked.

**Scope**
- 10-27: cost estimation is out of scope. Policy function, BBL and counterfactuals stay on the old panel, not rerun or audited.
- 11-02: RC-BLP numbers are provisional until I2. A fit is adopted only under T1 to T3 (gradient ≤ 1e-3, best dQ ≥ −1e-7, SEs ok).

## 5. Key numbers

| Panel md5 | Use now |
|---|---|
| `fa38794dcb3bed514358d3c2406f82ab` | Current: I2, K0 rerun, K1, P8, T10, T10b. Centering `f86f041e` |
| `0b9385b5` | T10b's comparison panel (copy in IMPR scratch) |
| `52ea6495` | Kappa step 7 snapshot, output `Rout/k2_step7_52ea6495/` |
| `d01b8bae` | R3 (job 31020411, I2 gate reference), K0 and K2 records (`k2_nls_numbers.json`) |
| `45dbfc58`, `70241f4e` | Superseded |

- B versus D type-4 gap (D minus B, points of the CDI, keys `desc.gap4.*`): unweighted 12.96 [8.22, 17.77], stock-weighted 23.33 [11.91, 34.06], spread −0.92 pp [−1.28, −0.57], G/G* 163/133. Posted against measured: Nubank 100.0 vs 85.8, PagBank 138.5 vs 81.1, C6 103.5 vs 106.2, so the true gap is likely larger.
- Section 2 after T10: relative-price slopes not significant (p 0.17), own repricing predicts deposit growth −1.14 [−2.14, −0.12], D premium 2.00 [1.42, 2.63], CDI-fallback share 0.
- Step 5 (sorting): κ̃ − κ overstates −ln h by up to 0.52. Step 6: without the outside pool κ̃ ≈ 0, with it 0.507 (κ = 0.5) and 1.200 (κ = 1), box at κ = 2. Step 7 (random coefficients): κ̃ 0.388 and 0.072, not converged (logit world 0.467 and 0.875). K2b recovers the Pix change (−0.503 and −0.455 against −0.5).
- Pix P8: stakes median R$42.1, F 782, placebo median 244, 1 of 99 at or above. Advertising IV 0.206.

## 6. Open items for Pedro

1. Bradesco 2016: A (DFP 2017 comparatives, about 1.092 × CDI) plus the HSBC merger adjustment of C0080075's 2016Q4 lagged stock (about 0.95 × CDI), at the next rebuild? The compiled file is edited only after his explicit OK.
2. Download the early-pull frames (job 32235273).
3. Apply the pending paper blocks, then compile.
4. Run the scratch `Remove-Item` lines in `IMPR_REPLY_2026-11-06a.md`.
5. Are κ̃^pre, κ̃^post, χ and ν^A confirmed as the paper's symbols?
6. Does a source-log rebuild keep the rows appended by hand on 11-05?

## 7. Log (newest first)

- 2026-11-08 00:37. IMPR 1e: four pix.roster.B.* keys ok (16; 1,022; 2.9; 2.9), roster access 2026-06-09 (all lists via Internet Archive), R1b is a sum of 2019 quarterly stocks. Pedro's 00:32 save: E3a, E3b through the advertising exclusion, with slips. 'Five largest known everywhere' = the awareness note's identifying assumption A_b = 1 stated as fact; support: branch presence (no key; two desc.big5.branch.any.* keys requested from IMPR) and Datafolha Top of Mind (BB first 2016-2022; 2020: the five are the top five, 30/18/18/17/8). Plan: Y1-Y4 slips (of; 2020Q4 and pix.roster; from; 'the finding'), Z1/Z2 define the dose kappa-tilde^U and alpha^{PU}, E3c restates the five largest as an assumption with its support, E4, and a bibliography section at the end (bcb_pix_participantes, bcb_pix_chaves, datafolha_topofmind, rothenberg1971identification). Harness: 8 blocks unique, 0 undefined refs, only the 2 pending keys.
- 2026-11-08 00:12. Pix mandate sentence (Pedro asked its source): it came from task R1 (make_pix_participation.py, Rout/pix_participation_*.csv/json, 11-02: BCB dated participant rosters via Wayback, Pix keys open data, panel savings), with no key or citation. Not exact: Bank of China, off Pix and not mandated, held R$2.94 million savings in 2020Q4, of R$1,021.9 billion; the 16 mandated hold the rest. Plan: X1 to X3 fix slips in Pedro's 00:03 save; E3a rewrites the sentence with a footnote (four keys pix.roster.B.*, two new bib entries bcb_pix_participantes and bcb_pix_chaves, drafted in the block header); E3b the rest; E4 appendices. MAIN_REPLY 11-08 asks IMPR for the four keys, the roster access date, and what R1b's 'R$12 million of R$3,225 billion' sums. Harness: 6 blocks unique, 0 undefined refs, only the 4 pending keys missing.
- 2026-11-07 23:41. Pedro (23:2x): formal identification in the appendix, move the detail of 7.2 to Appendix B, shorter paragraphs, blocks in execution order. Plan rebuilt as E1 to E4 in paper order: E1 kappa-tilde-4 carries the clock; E2 replaces his five varsigma/vartheta paragraphs with two short ones; E3 rest of 7.2 (identification as intuition pointing to the propositions), Pix (exposure details and table moved), awareness, stakes, fees (evidence moved), 7.3, all paragraphs short; E4 appendices with Appendix B subsections: derivation, scales and exit, identification (Prop: local identification of the rival factors via a Jacobian rank condition, Rothenberg 1971; Prop: kappa-tilde = kappa - ln A - ln h, bound under A8, Pix change under A9, with proofs; h now defined relative to j's own savings pool), estimation and inference, Pix exposures, within-bank margin. Needs rothenberg1971identification in References.bib (BibTeX in E4 header). Harness on the 23:31 save: 4 blocks unique, 0 undefined refs, no new overfull except two stub-only cite-key boxes. Kappa note v43 (h definition aligned).
- 2026-11-07 23:05. Pedro's 22:59 save: N1a, S1, S4, S5, S7 to S10 applied, W5 (b) through the varsigma paragraph (slips). Plan: kappa-tilde now defined at first use, right after eq:20 (K5: kappa-tilde = kappa - ln A - ln h, subscripts 2/1/4); later W5 text no longer repeats it. K6 to K9 fix the slips (frees; rivals'; e6 -> e^; sec:inst_setting:switching -> sec:disentangle:switching). Notation (Pedro): time-deposit parts are 4^{\mathrm{m}} and 4^{\mathrm{d}}, not 4\mathrm{m}/4\mathrm{d}; plan replaced throughout, K10/K11 fix eq:19 and eq:20 in the paper; both notes aligned (kappa v42, aw v19) and re-rendered. Harness on the 22:59 save: 10 blocks unique, 0 undefined refs, no new overfull.
- 2026-11-07 22:39. Pedro's 22:30 save has eq:20 and the entry-and-exit paragraph. W5 (b) re-anchored there and rewritten: three paragraphs (entry scale and alpha; varsigma/varsigma_varnothing: model fixes scale one, why freed, what they absorb incl. M-hat error, identification via own checking; vartheta: model coefficients 1 and 1-phi, the constant-exit fit does not converge, cause not isolated, three candidates: annual/imputed omega, lambda~0 artifact, sweeps at maturity). W7 B aligned. Softening blocks: S1 (intro poverty + Selic), S2 extended (CadUnico 'most consistent', 95% interval includes 0), S7 abstract, S8 conclusion, S9 intro state premium, S10 logit appendix. Harness on the 22:30 save: 11 blocks unique, keys resolve, 0 undefined refs, no new overfull. Kappa note: exit claim aligned, re-rendered.
- 2026-11-07 22:18. Deleted with Pedro's confirmation: IMPR_REPLY_2026-11-06a.md (superseded by 07a) and the sandbox scratch. Its scratch-removal lines (07a item 10) must be restated by IMPR; asked in MAIN_REPLY.
- 2026-11-07 22:17. Pedro's 22:12 save: S3, S6 applied; S4 applied with 'starting value ... they' and the red SE sentence; S5 with ',,'. IMPR 22:11 (IMPR_REPLY_2026-11-07a, now live): BLP SEs use the exact IFT Jacobian, WCB, so S4 drops the red sentence. Plan: S3, S6 removed; S4, S5 rewritten as slip fixes; N1b merged into W5 (b), which now has a real SEARCH (end of the eq:19 gather plus the Conclusion heading), because 'The estimating equation is' did not yet exist in the paper. Harness on the 22:12 save: all 7 blocks unique, keys resolve, 0 undefined references, no new overfull box. Kappa note v41: SE line says WCB with the exact Jacobian.
- 2026-11-07 22:05. I2 text check (main thread, own probe on the 21:47 macros vs 11-05 20:33): 65 quoted values changed, none missing or pending. Plan Part I-s, S1 to S6: S1 intro Selic, S2 age 65 not significant / Selic significant at 5 under (IV) [-8.33, -0.02] / mobile now negative, S3 Pix significant at 10 under (IV), both 95% intervals include 0, S4 red BLP caveat replaced (FD check passes, global optimum not proven; SE sentence kept red pending IMPR), S5 Sigma(Spread) at 0 in both columns, S6 entrant bank counts (8, 9) replace the unkeyed 'one bank makes most'. W5 Selic sentence updated. N1 split into N1a/N1b (Pedro pasted the gather). Harness: all blocks unique, keys resolve, no new overfull box. Kappa note v41: alpha-bar -0.216 (was -0.117), E_c range 0.76 to 0.96, engine status. Awareness note unchanged. Asked IMPR (MAIN_REPLY): exact-Jacobian SEs, gfb.firmtype/holdout vintage and a D-firm entry-interval key, the D-share jump 30 to 46, the entry-gap regrouping, stale ssel/TR5/k2.
- 2026-11-07 21:36. Type-5 prices (Pedro): institutions set prepaid rates too; in the panel every B institution carries the CDI on prepaid balances, so among B institutions only the time-deposit rate varies, and prepaid balances (0.6 percent) are left out of the pools. Plan: W5 (b) starts after block M1 (Pedro's 21:31 sentence, corrected), M2 fixes the stakes sentence, W7 Appendix B states the same; both notes aligned. The plan's 6.3 now says the K1 pilots do not yet pin down the change. K1 pilots (11-06 a_m, 11-07 savings-only) recorded in the switching-cost note: the profiled objective is rough; the main thread recommends option (i), smoothing the effect solve. `MAIN_REPLY` asks IMPR for a table of changed, missing and stale keys quoted in the paper or the plan after refresh phase 2 (macros still 11-05 20:33), and answers the BLP-stage question (ext1). Retest: 10 blocks unique, no undefined reference once applied.
- 2026-11-06 18:42. Assumptions A1 to A9 are an enumerated list at the start of Appendix B (W7), `enumerate[label = A\arabic*, ref = A\arabic*]`, labels assum:mean_utility, common_denominator, source_cost, one_kappa, market_size, scale, awareness, sorting, launch, and every mention in W5 (b), W7 and Pedro's applied 6.2 text (blocks L1 to L3) goes through \ref, as the estimation methods do. Retest: 8 blocks unique, no undefined reference once applied.
- 2026-11-06 17:50. Pedro's 17:48 save: J1 in his wording, J2, J3 applied; 6.2 through '... not on where it sits.'. Plan: W5 (b) from 'Savings pays the regulated rate' as a new paragraph; W7; K1 to K4 (slips in the J1 paragraph). Retest: 5 blocks unique, no undefined reference once applied, 95 pages.
- 2026-11-06 17:04. Pedro's 16:59 save: I1 to I4 applied, W5 (b) pasted through '... that moves.' in 6.2 (he writes 'a R\$' for 'a real'; the plan follows). D_mt now defined (block J1: $D_{mt}=1+\sum_l\sum_k\exp\{\delta_{lkmt}\}$, the denominator of eq:4 at mean utilities, no κ or clock; also in the switching-cost note); J2 'leavers', J3 eq:18's period. Retest: 4 blocks unique, no undefined reference once applied, rendered page checked.
- 2026-11-06 16:40. From IMPR's 11-06 log, carried into the plan, both notes and `MAIN_REPLY_2026-11-05.md`: the K1 headline exit is one market-quarter coefficient on the movable stock (`--exit-by-cell`; the constant matured exit froze the overnight real fit and is reported as the failed check); headline sample p99-trimmed on |NF/Dep_j2|, untrimmed as robustness; four starts, cap 500, interior and boundary optima both reported where starts split. The void real fit gave G 16, G* 5.34, control-function F 9.85. I2: E4's ext2/extended froze (being re-fit), G3 passes under the documented 2.5% exception, R3's '6 RC' column was ext1's point, and the φ-cap shares rose under I2 (E3 0.143 to 0.235); the paper's RC and AME numbers change at the I2 refresh. Pedro applied I5 himself (16:32). Retest: 5 blocks unique.
- 2026-11-06 16:30. Second-instance appeal on BB 18882.002114 sent by Pedro (predecessors' dates and opt-in). Text in `DATA_RAW/issuers/sweep_programs/raw/bb/lai_18882_002114/bb_recurso2_18882.002114.txt`.
- 2026-11-06 16:26. Pedro's 16:16 save: H1, H2 applied; W5 (b) pasted through the first sentences of 6.2 (eq:17 is the balance). Plan: W5 (b) continues that paragraph from 'Adding the stayers'; new blocks I1 to I5 (eq:17's missing '+', 'stock by source', a leftover b and 'A\equiv_1' in the nesting conditions, 'Appendix \ref', 'awake'); the plan's own 'state' leftovers fixed. Retest: 6 blocks unique, keys and citations resolve, no duplicate label, no undefined reference once applied. LAI 18882.002114 (BB) appeal answered: Rende Fácil invests idle checking at day end in BB's own CDBs with automatic redemption; opt-in, with an opt-out default for new accounts from 05/2025; predecessors were automatic funds or savings, no dates. Filed in `DATA_RAW/issuers/sweep_programs` (answer file, NOTES section, rows BB-4, BB-5 updated, BB-6 added); switching-cost note updated. K1 pilot (IMPR 16:15): unstable across starts, boundary optima with the checking scale or κ̃ at the box; Pedro chose option (c), report interior and boundary optima side by side.
- 2026-11-06 15:36. Paper equation labels are numeric (Pedro, 15:33 save): eq:12 utility (was eq:fr_util), eq:13 share (fr_share), eq:14 awareness (fr_aware), eq:15 stakes (fr_stakes), eq:16 cost of attention. The plan follows: W5 (b) adds eq:17 balance (fr_balance), eq:18 net flow (fr_nf), eq:19 pools (fr_pools), eq:20 estimating equation (fr_reg), eq:21 Pix exposure, eq:22 openings (fr_openings). Appendix labels stay eq:app_*. Retest: 3 blocks unique, no duplicate label, no undefined reference once applied.
- 2026-11-06 15:27. Pedro's 15:24 save: G1 to G9 applied, W5 (b) pasted through eq:12, the superseded \iffalse block deleted. Compile error (`\tau_\mathrm{stakes}`, double subscript) fixed in `V_Friction.tex` by the main thread at Pedro's request (one token: `\tau_1\,\mathrm{stakes}`), line endings kept. Plan now: W5 (b) from after eq:12, W7, H1 (R\$ in math), H2 (sentence). Retest: 3 blocks unique, keys and citations resolve, no undefined reference once applied, 93 pages.
- 2026-11-05 20:40. Pedro's 20:27 save: the start of W5 (b) pasted, C3a to C3e applied. 'State' in Section 6 now means only the market states S_mt; the depositor's (j'(i),k'(i)) is her source (blocks G1 to G6; both notes aligned). The stakes are written out as the balance-weighted standard deviation of type-4 spreads times the mean balance per relationship (G8, W5 (b), both notes), replacing sd_w. eq:fr_share stays on one line in a \small display (G9, \mathsmaller alone left it 11pt wide). The plan's preamble is now eight lines. Pending: W5 (b), W7, G1 to G9.
- 2026-11-05 20:07. Both notes aligned with the paper: age $a_i$ (was $s_i$) in the clock formulas, checking and new-money scales $\varsigma$, $\varsigma_\varnothing$ (were $a$, $a_\varnothing$), and the awareness note's clock factor $E_c$ (was $e^{a_4}$). Re-rendered: switching-cost note 44 pages, awareness note 28 pages, no errors.
- 2026-11-05 20:05. Notation (Pedro, 20:01 save): a time deposit's age is $a_i$ (was $s_i$), to avoid confusion with shares. In the paper plan the checking and new-money scales are therefore $\varsigma$ and $\varsigma_\varnothing$ (were $a$, $a_\varnothing$; the code keeps `a`, `a_null`). The 20:01 save applied S2, S4, S5, F7, U1 and C2 (some reworded), all removed from the plan, which now holds W5 (b), W7 and C3a to C3e (typos in applied text, including a key typo `desc.gap.posted.nubank.measured` in S4). Retest on 20:01: 6 blocks unique, keys resolve paper-wide after applying, no new overfull. The notes were aligned at 20:07.
- 2026-11-05 19:58. `review_fix_spec_2026-11-05.md` deleted (Pedro's OK). The K1 equation lives in eq:reg of the switching-cost note and is written out in `MAIN_REPLY_2026-11-05.md`.
- 2026-11-05 19:55. Old memory replaced by this file and the October archive deleted (Pedro's OK). The paper plan and both notes record the two-start check: step 6's rows at κ = 1 and 2 reflect the form, not the start. The plan's simulation table has every key (T13 in). The notes record K1 `761a02cf`'s tests and the re-coding of its scales in progress. Retest: 7 blocks unique, 218 `\pn` uses resolve, citations present. Renders: switching-cost note 44 pages, awareness note 28 pages, no errors.
- 2026-11-05 19:48. Memory condensed. The main thread accepted the T9 key names and the combined IMPR reply file. T13 is done (18:59).
