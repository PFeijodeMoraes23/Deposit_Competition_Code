# Handoff: BB Rende Fácil and what followed (LAI 18882.002114/2026-19)

**To:** the BB agent. **From:** main thread, for Pedro. **Written:** 2026-11-03, 18:00. One file: it replaces the first handoff of 14:14 and the reply-answer of 16:10 (`bb_handoff_2026-11-03b.md`, removed), and keeps their record below.

**Answers** your reply of 15:25 (`BB_REPLY_2026-11-03.md`, removed by Pedro after processing). Reply in `claude/BB_REPLY_2026-11-03b.md`, timestamped.

## Update 11-03, 22:05: task B8, three additions to the statement file (from the coding thread's 21:52 reply)

The coding thread read your file and needs three small things before the rate rebuild can use it fully. Same format, same folder, read-only on the pipeline. Reply in `BB_REPLY_2026-11-03e.md`.

```
b8[3]{step,what,output}:
  1,"Subordinated and hybrid debt cost, domestic only, as lines subordinated and hybrid for every bank and year that prints them (Bradesco already has subordinated); perpetual or other instruments issued abroad go in external, not in hybrid","rows in compiled/dfp_funding_expense.csv"
  2,"A small file compiled/subordinated_in_lump.csv (cnpj, from_year, source, page, note): for each bank whose notes say from when the subordinated cost is presented inside funding expense, that year with the page; banks whose notes do not say get no row (the loader then uses 2022, BB's year, flagged)","the file"
  3,"The scope of each bank's time_deposits line: does it include judicial deposits, special deposits (FAT, FGTS, PIS, court funds) or foreign-currency deposits, by bank and year range, with the page; one row per bank and range in compiled/time_deposit_scope.csv (cnpj, year_from, year_to, includes_judicial, includes_special, includes_foreign, page, note). BRB first: its statement time-deposit rate runs 1.37 to 1.50 times the CDI in 2020 to 2024, which the coding thread suspects is judicial deposits inside the line. And add Itaú's time_deposits line if any of its notes prints one (its deposits expense appears only as an IFRS total; say so if that is all there is)","the file; Itaú rows or a sentence"
```

Why: Pedro adopted (D10, 22:10) the coding thread's proposal to anchor each statement bank's annual CDB expense to its time-deposit line, net of what the line includes beyond ordinary domestic CDBs, with Cosif giving the quarterly shape; the universal banks fail the component benchmark (Bradesco +1.42 × CDI in 2016, Santander between −0.81 and +0.53). Step 3 is what makes that anchoring possible, and the panel rebuild waits for it, so B8 comes before anything else. Keep Santander's exchange-rate and capital-instrument items as separate lines or notes, as you did.

## Update 11-03, 21:55: B7 accepted in full; the three additions above reopen it as B8

Your 21:35 reply (`BB_REPLY_2026-11-03d.md`, removed after processing) closes B7: eight banks in `dfp_funding_expense.csv` (666 rows), Santander's savings line found, Itaú's confirmed absent, BB's 2018 to 2019 line shown not to be pure savings remuneration, the 2015-12 judicial values in and the share file from 2015Q4 (588 rows), BB's 2023 to 2024 quarterly judicial stock from six ITRs with the December 2023 precatório step at +R$45.3 bn. Pedro's decisions on your flags (21:50), which go to the coding thread's loader: BB's savings line is not used (savings at the official yield, flagged); partial banks are benchmarked only on the lines they print; Santander's exchange-rate effects are netted out where the note separates them and the line-year is skipped where it does not; Banrisul's inferred 2023 judicial line is excluded; BRB's BR GAAP alternatives stay out. The rebuild starts now on your files.

**You have no open task.** The session can stop; if the CDB registry (025399, 025440) lands on 11-13 or the BB follow-up on items 3 to 5 arrives, a new handoff will come. Reply, if anything, in `BB_REPLY_2026-11-03e.md`.

## Update 11-03, 21:35: B7 step 1 (Caixa) accepted; two requests from the coding thread

Your 20:45 reply (`BB_REPLY_2026-11-03d.md`) is accepted: `DATA_RAW/issuers/funding_expense/compiled/dfp_funding_expense.csv`, 124 Caixa rows 2016 to 2025, individual entity, sum checks against the printed totals, special deposits as their own line. The coding thread's loader reads that file as it is, so keep the format for the other seven banks. Two requests from the coding thread, both small: (1) an optional `period` column with `FY` or `H2` where a note also prints the second semester (BB does), missing read as `FY`; (2) `bills_total` as letters only (LCA, LCI, LF and the like, COE where the note groups it there), with subordinated and hybrid debt in their own lines `subordinated` and `hybrid`, and securities issued abroad in `external`, as you already did for Caixa. BB's own transcription goes in a separate file, `dfp_funding_expense_bb.csv`, written by the coding thread; do not transcribe BB's lines except the 2018 to 2019 savings check asked in step 1. Still open: the seven banks, the 2015Q4 judicial rows, BB's 2023 to 2024 quarterly judicial stock. Reply next in `BB_REPLY_2026-11-03e.md`.

## Update 11-03, 22:40: B4 to B6 accepted; task B7, the inputs the rate rebuild needs

Your 19:28 reply (`BB_REPLY_2026-11-03c.md`, removed after processing) is accepted: B4 staged with a matching hash (the Downloads copy is deleted); B5's negative result is settled (the 2012 rule applies only to deposits made from 4 May 2012, withdrawals drain the new-rule balance first, and no source gives the old-rule stock by bank; the decision is to rebuild without the share and use printed savings expense where it exists); B6 dated Caixa's fall as the unwinding of the December 2023 precatório deposits, with 2023 and 2024 quarters now observed. Reply next in `BB_REPLY_2026-11-03d.md`.

**B7 is the only open task and the rebuild waits for it.** The coding thread's I9 benchmarking (adopted by Pedro) rescales the pipeline's imputed funding costs to each bank's annual statement totals; it passed the BB check 9 years of 9 and needs the same lines for the other banks. Read-only on the pipeline; new files only; the transcriptions go in `DATA_RAW/issuers/judicial_deposits/compiled/` (or a sibling folder `funding_expense/`, said once) with the page cited on every row and a source-log row per statement used.

```
b7[5]{step,what,output}:
  1,"The annual funding-expense note, 2016 to 2025 where the statement exists: for Caixa, BRB, Banese, Banestes, Banrisul (the high-judicial banks) and Bradesco, Itaú, Santander (the big savings banks), the lines for repurchase agreements, bills (letras financeiras, LCI, LCA), savings, judicial deposits (where printed) and time deposits, in R$, with the page. BB is already transcribed by the coding thread (promoted from its scratch), so do not redo it; check its 2018 to 2019 savings line instead, which implies an old-rule share above one, and say what else the line contains","compiled/dfp_funding_expense.csv (bank, year, line, value_brl, page, statement, note)"
  2,"The savings expense at Itaú and Santander by hand: the automated search missed the line; read the deposits and funding notes 2017 to 2021 and transcribe it, or state that it is not printed","rows in the same file; one sentence per bank in the reply"
  3,"The 2015-12 judicial stock for BB and Caixa, from the comparatives in the 2016 statements, so the 2016Q1 lag is observed rather than held back","rows appended to compiled/dfp_judicial_extracts.csv; judicial_deposits_share.csv rerun"
  4,"BB's quarterly judicial stock for 2023 and 2024 from its interim statements (CVM ITR, the same cache route as the DFPs, or BB's investor-relations site), to date the December 2023 precatório spike and its 2024 unwinding at BB as you did for Caixa","rows appended; judicial_deposits_share.csv rerun with BB's 2023 and 2024 quarters observed"
  5,"One paragraph: which banks have every line, which are partial, and anything in a note that contradicts the Cosif lump (your BB3 and B3 cross-checks are the model)","Rout/funding_expense_report.md"
```

**Flags.** Statement lines are consolidated or individual; say which you took and why, and keep it consistent with the judicial extracts (Caixa individual equals consolidated; check the others). Foreign-currency and special deposits (FAT, court funds in 4.1.6) are outside Cosif 4.1.5; keep them as separate lines rather than folding them into time deposits. When step 1 is done for Caixa, say so first: the coding thread can start the rebuild on BB and Caixa and add the rest as they land.

## Update 11-03, 21:20: your 18:36 reply (B1 check 3, Caixa) accepted; state of the B tasks

```
assess[4]{item,verdict}:
  B1 check 3 and verdict,"accepted: Santander's dep_a4 equals its Cosif stock within R$0.1 bn, no shift at the sweep; the rate change across the event is +0.086 of the Selic, inside the placebo ranges; immaterial for type 4, the same verdict as BB3. Closed"
  B3, Caixa,"accepted, and it closes the gap: Caixa's BRGAAP statements from its investor-relations file API (ten PDFs, hashes in source_log_rows.csv), the note's time-deposit total equal to Cosif 4.1.5 at every December, the 2025-12 judicial line equal to Cosif 4155000009, the 2018 share confirmed in words (0.462). Caixa 0.33 to 0.71 (0.70 at end-2021). So B6 as written at 20:10 (Pedro's five downloads) is NOT needed; it is repurposed below"
  Source log,"the six rows appended to the master (279 to 285), earlier rows byte-identical: accepted"
  Where it went,"your two B3 scripts and four Rout files are the inputs of the coding thread's task I9 (impr_handoff_2026-11-03.md, Updates 11-03e and 11-03f): judicial stock out of dep_a4 and out of the rate base, one panel rebuild. The 2024 Caixa fall is flagged there as the one open item"
```

```
B_state[6]{id,what,state}:
  B1,"Santander sweep booking","DONE 18:36, closed"
  B2,"bb_rendefacil_termo_2022 into References.bib","DONE 16:40, closed"
  B3,"judicial deposits inside type 4, every bank, Caixa included","DONE 18:36, closed; feeds I9"
  B4,"stage the second-instance decision on LAI 18810.025439 (Parecer PDF in Downloads)","OPEN, small; spec in the 19:20 update"
  B5,"old-rule savings stock by bank from the statements' notes; the SGS national series","OPEN; spec in the 19:20 update; feeds I8, which Pedro has decided to apply"
  B6,"repurposed: the timing of Caixa's 2024 fall in judicial deposits (R$43.0 bn) from Caixa's 2024 quarterly statements on the same file API: which quarter, and whether it is the transfer to the Treasury; one row per quarter added to dfp_judicial_extracts.csv with the page, and the 2024 interpolation in judicial_deposits_share.csv replaced by the observed quarters","OPEN, after B4 and B5"
```

Reply in `claude/BB_REPLY_2026-11-03c.md`. The results of BB3, B1 and B3 are now recorded, with their files, in Section 11 of the design note (`Drafts/Deposit Competition/kappa_design_2026-11-02.md`, seventeenth version).

## Update 11-03, 20:10: answers to your 17:05 reply (B1 to B3), and B6

```
assess[5]{item,verdict}:
  B2 (BB6),"accepted: 11 lines inserted after bradesco_investfacil_2020, bytes identical around the insertion, biber 0/0/0. Closed"
  B3 (JD1),"accepted, and it is the finding that changes the panel. dep_a4 (verbete 432) carries 4.1.5.50 for every bank and every year; the rate chain divides CDB expense (8113 only) by the 4.1.5 total (judicial included) in the 2025+ rate and in the de-bias factor k_i, so BB's measured CDB rate is about half the true one. Goes to the coding thread as task I9 (impr_handoff_2026-11-03.md, Update 11-03e): judicial stock out of dep_a4 and out of the rate base, one panel rebuild. Your two scripts and four Rout files are its inputs. The BB3 share is corrected to 51-76% in every record"
  B1 (BB5) checks 1 and 2,"accepted: a default sweep shows at most in demand deposits (about R$2 bn, at the edge of the placebos) and is immaterial for type 4. The instrument caution stays in the report"
  Local booking,"your remark that ESTBAN books judicial balances at the branch by the court is recorded; it is the open question inside I9 (proportional removal with a capital flag against dropping capital cells)"
  Correction to BB3,"taken: the 48-60% divided by the DFP's broader line; against the 4.1.5 total it is 51-76%"
```

**Your three questions, answered by Pedro (11-03, 19:45):**

1. **B1 check 3:** run it as soon as `julia.exe` 38692 ends; no further word needed. Five conglomerates through pyarrow, as before.
2. **Master log:** append the five new rows (`BCB/cosif_plano_contas` 1, `issuers/judicial_deposits` 4) to `DATA_RAW/source_log.csv` as with BB1, without rebuilding (I7 rebuilds later).
3. **Caixa's statements:** yes. Pedro downloads the annual statements for 2017, 2019, 2021, 2023 and 2024 from ri.caixa.gov.br into `C:\Users\pedro\Downloads\` and tells you the file names. That becomes **B6**.

### B6. Caixa's judicial deposits 2016 to 2024 (after Pedro's five files land)

Stage the five PDFs untouched under `DATA_RAW/issuers/judicial_deposits/cvm_dfp/caixa/` (or the folder convention you used for BRB, Banestes and Banese), sha256 each, source-log rows; transcribe the year-end judicial stock and the time-deposit line with the page into `dfp_judicial_extracts.csv`; extend `Rout/judicial_deposits_share.csv` for Caixa 2016 to 2024 by the same interpolation, with the 2023 to 2024 transfer of about R\$12.6 bn to the Treasury flagged as a break; report Caixa's share by year in the reply. `Remove-Item` lines for the Downloads copies after the hashes match.

**Open on your side:** B1 check 3, B4 (the Parecer staging), B5 (old-rule savings from the notes; now feeds a correction the coding thread will apply, task I8), B6.

## Update 11-03, 19:20: labels rebooted; two tasks added (B4, B5)

**Labels (Pedro, 11-03).** Tasks are numbered by agent from now on: **B** for you. The map: BB5 is **B1**, BB6 is **B2**, JD1 is **B3**; **B4** and **B5** are new. Pedro is waiting on two points from you (B1 and B2 or B3; say which in your reply). Reply in `claude/BB_REPLY_2026-11-03b.md` as before.

```
B_tasks[5]{id,was,what,when}:
  B1,"BB5","Santander Investimento Automático booking check (Decisions below)","go"
  B2,"BB6","bb_rendefacil_termo_2022 into References.bib, verified (Task BB6 below)","go"
  B3,"JD1","judicial deposits inside Cosif 4.1.5, every bank (Task JD1 below)","go"
  B4,"new","stage the second-instance decision on LAI 18810.025439 (below)","now, small"
  B5,"new","old-rule savings stock by bank from the DFP notes (below)","after B3"
```

### B4. Stage the second-instance decision on LAI 18810.025439/2026-13

The BCB's Diretora de Cidadania e Supervisão de Conduta granted Pedro's second-instance appeal on 11-03 (decision dated 03/11/2026, on Parecer Jurídico 1461/2026-BCB/PGBC of 29/10/2026). The appeal asked two things: (1) whether the BCB receives from institutions, or keeps, the number of clients with a Standardized Service Package (Pacote Padronizado de Serviços, Resoluções CMN 3.919/2010 and 4.196/2013); (2) if so, the number of natural-person clients with each package (I to IV, separately), by institution, at December of each year 2016 to 2024, in the stored format. The decision orders Deati to (a) provide the information if not already delivered, (b) say where it sits in the delivered set if it was, or (c) state expressly that it does not exist, within 30 days (so by about 12-03), individually for each item. A further appeal to the CGU is possible within 10 days.

- Copy `C:\Users\pedro\Downloads\Parecer Jurídico 14612026-BCBPGBC.pdf` (10.1 MB, 11 pages, no text layer; the decision minute is page 10) untouched to `DATA_RAW/BCB/tarifas_lai_18810_025439/raw/parecer_juridico_1461_2026_pgbc.pdf`, with sha256.
- Save the decision text below as `DATA_RAW/BCB/tarifas_lai_18810_025439/raw/decisao_2a_instancia_2026-11-03.txt` (UTF-8, the same two-line provenance header you used for the BB answer).
- One row each in the folder's `source_log_rows.csv` if it has one (else `DATA_RAW/_extra_source_log_rows.csv`), publisher "Banco Central do Brasil (via LAI, Direc)", accessed 2026-11-03, licence "not stated"; the master log is not rebuilt (I7).
- Add to `DATA_RAW/BCB/tarifas_lai_18810_025439/NOTES.md` a dated section: what the second-instance appeal asked, the decision, the deadline, and that the Fees workstream's verdict on the first delivery (byte-identical to the public census) stands.
- Give Pedro the `Remove-Item -LiteralPath` line for the Downloads copy once the hash matches.

The decision text, verbatim as Pedro pasted it:

```
Prezado senhor, Refiro-me ao seu recurso de segunda instância interposto contra a decisão proferida pelo Chefe do Departamento de Supervisão de Conduta (DECON), referente à manifestação protocolizada sob o NUP: 18810.025439/2026-13. Com fundamento no Parecer Jurídico 1461/2026-BCB/PGBC (cópia anexa), conheço do recurso e lhe dou provimento, para que o Departamento de Atendimento Institucional (Deati), em articulação com outras áreas do Banco Central (BC) competentes, providencie resposta acerca dos seguintes pleitos: 1. disponibilização ao demandante do acesso às informações solicitadas no recurso de 2ª instância, caso não tenham sido franqueadas na oportunidade anterior; 2. caso já tenham sido disponibilizadas ao demandante, a discriminação de onde se encontram as referidas informações dentro do acervo enviado; 3. se as informações não estão disponíveis no BC, a comunicação expressa ao cidadão sobre a inexistência. As informações requeridas ou a confirmação da sua inexistência, de maneira expressa e individualizada, serão fornecidas no prazo de 30 dias. Caso queira, o senhor poderá interpor recurso contra a decisão ora comunicada perante a Controladoria-Geral da União (CGU), no prazo de 10 (dez) dias. Atenciosamente, Izabela Moreira Correa Diretora de Cidadania e Supervisão de Conduta.
```

### B5. The old-rule savings stock by bank, from the DFP notes

**Why.** Savings deposits made before 4 May 2012 pay the old rule (TR plus 0.5 percent a month); later ones pay 70 percent of the Selic plus TR when the Selic is at or below 8.5 percent (Lei 12.703/2012). The panel's savings rate is the regulated new-rule rate for every bank, and Cosif does not split savings expense from the lumped funding expense before 2025-01, so any old-rule premium a bank pays is booked in the panel as time-deposit expense and overstates its time-deposit rate by (old-rule share) × (premium) × (savings over time deposits). The premium was positive in 2017Q4 to 2021Q3 (1.5 pp a year in 2018, 4.8 at the 2020 trough). Per unit of old-rule share the overstatement averages 4.6 pp a year at Caixa, 2.3 at BB, 1.6 at Itaú, 1.5 at Bradesco over 2018 to 2020 (K0, `Rout/k0_savings_vintage_bias.csv`). The missing input is the old-rule share by bank and year.

```
b5[4]{step,what,output}:
  1,"In the cached DFPs (raw/DEPOSIT_MATURITY/cvm/<bank>/) for BB, Itaú, Bradesco, Santander, Banrisul and any other cached bank with savings, and in Caixa's published statements if any are cached, search the deposits note for a split of 'depósitos de poupança' by remuneration rule ('regra antiga', 'regra nova', 'anteriores a 04/05/2012', 'TR + 0,5%', '70% da Selic'); report, per bank and year 2016 to 2024, the old-rule stock and share if printed, with the page","Rout/savings_old_rule_dfp.csv (bank, year, savings_total, old_rule_stock, old_rule_share, page, note)"
  2,"Where the notes do not split, say so per bank, and record what they do print about savings","in the reply"
  3,"The national series: the BCB's SGS stock of savings by rule (the 'poupança' series split old and new rule), monthly, 2016 to 2024, as a check on the sum of the bank figures","Rout/savings_old_rule_national.csv, with the SGS series codes"
  4,"One paragraph: for which banks the old-rule share is observable, its range, and whether a rate correction by bank is feasible","Rout/savings_old_rule_report.md"
```

No pipeline edit, no key. Read-only on the DFP cache and the SGS.

## Rules

- **V_Friction.tex and the decks are edited by Pedro only.** Deliver data, records and reports. `References.bib` is the one exception, task BB6 below, with the diff reported.
- **Pedro owns git, and you cannot delete files:** give `Remove-Item -LiteralPath` lines.
- **Raw files stay untouched.** The two PDFs are copied, never edited.
- **Python:** `C:/venvs/egan/Scripts/python.exe -I`, with `MPLBACKEND=Agg`. One heavy process at a time. The coding agent (`claude/impr_handoff_2026-11-03.md`) and the Pix session (`claude/pix_handoff_2026-11-03.md`) run on the same machine; do not read the market panel while they do, and read it through pyarrow when you do, as you did in BB3.
- **Before you edit any existing script, registry file or the bibliography, summarize the change and its side effects and wait for Pedro's OK**, except where this file already gives it. New small files are preferred.
- **Never guess a schema or a file layout.** Read the file, or ask.
- **Timestamp every update.** "Wait" means stop.

## State at 18:00

```
state[7]{id,what,state}:
  BB1,"the two PDFs, the answer and the request staged in DATA_RAW/issuers/sweep_programs/raw/bb/lai_18882_002114/ with sha256; four rows appended to source_log_rows.csv and to the master log (not rebuilt)","DONE 14:34, accepted. The Downloads copies were removed by the main thread on 11-03 after the hashes matched"
  BB2,"sweep_products.csv rows BB-4 and BB-5 after BB-3; BB-1 verification upgraded; BB-2 note; NOTES.md section and three one-line corrections","DONE 15:08 with Pedro's OK, accepted as applied"
  BB3,"where is Rende Fácil booked: DFP notes, Cosif monthly against four controls, the panel","DONE 15:25, accepted: undetermined but immaterial (below)"
  BB4,"bibliography entry bb_rendefacil_termo_2022, biber-clean","DONE 15:08, accepted; it goes into References.bib as BB6"
  BB5,"Santander Investimento Automático, the same three checks","GO (decision below)"
  BB6,"paste the entry into References.bib and verify","NEW, go"
  JD1,"judicial deposits inside Cosif 4.1.5, every bank","NEW, go"
```

**BB3, what it settled.** The 2022 jump in BB's time deposits (+R\$70 bn raw, +R\$29 bn against the controls, 2022-02 to 2022-06) is not Rende Fácil: it is too large for a sweep of PF demand deposits (R\$53 bn at end-2021), demand deposits show no mirror (flat against the controls, pre-period sd 6.1 log points, so a sweep up to about R\$12 bn could hide), and the DFP puts it in the 1-to-3-year bucket without a liquidity clause, while the liquidity-clause line grew R\$3.9 bn over 2022. BB's measured type-4 rate as a share of the Selic does not fall relative to the controls from 2022Q2 (+0.048; +0.157 without the seven fallback quarters). Back of the envelope, R\$4 to 12 bn at 10 to 50 percent of the DI inside a R\$346 bn stock lowers the measured rate by 1 to 3 pp of the Selic, below the quarter noise. Where it is booked stays item 4 of the follow-up. The jump is probably fixed-term 1-to-3-year issuance into the rising Selic; the counterparty is not identified.

**Two findings that matter more than the question asked.**
- **Judicial deposits are 48 to 60 percent of BB's type-4 stock** (DFP: 183.2 of 307.6 R\$ bn at end-2021; 52% 2022; 57% 2023; 48% 2024), inside `dep_a4` (which tracks the Cosif 4.1.5 entity total plus R\$9 to 11 bn) with their expense in `rate_a4_national`. BB's rate is the lowest of the five in 10 of 15 quarters from 2021Q2. Task JD1.
- **Seven fallback quarters:** `rate_a4_national` equals the Selic exactly (Itaú 2020Q1, 2020Q2, 2021Q1; Santander 2020Q1, 2021Q1) and is 0.000 for both in 2021Q2, with `rate_a4_guarded` 0. That is the main agent's task RT1 (`impr_handoff_2026-11-03.md`, Update 11-03b), not yours.

**For the design note** (`Drafts/Deposit Competition/kappa_design_2026-11-02.md`, fourteenth version): BB's checking pool is treated as unswept (opt-in, a few R\$ bn at most); BB's and Caixa's own time-deposit pools are flagged until JD1 sizes the judicial share.

## Decisions

- **BB5: go.** Santander's Investimento Automático, the same three checks, Santander treated and BB among the controls, 2021-01 to 2023-12 around 2022-05 (and 2022-09-30 for the tier change). It is the default sweep, so it is the one that can be large enough to see, and the one the sweep indicator of the design note needs dated. Read-only, about fifteen minutes. Output `Rout/santander_sweep_booking.csv` and the figure in the BB layout. The caution that the instrument is named as a CDB only in clause 80 of the general conditions goes in the report, not in the registry.
- **The master source log is not rebuilt** until the main agent's SL1 has moved the CC1 row and the S4 edit into the builder's inputs. Your four rows match what a rebuild would produce except `round`, left blank as in CC1 and Fees-L; fine.
- **The bib entry:** "Termo de Adesão e Ciência" with the printed heading recorded in `annotation` is the right choice.

## Task BB6: the entry into `References.bib`

Pedro's OK is given here. `Drafts/References.bib` (903 entries, UTF-8, CRLF line ends, tab indents) gets the entry below, inserted after the closing brace of `@misc{bradesco_investfacil_2020,...}` so the sweep entries stay together, with one blank line before and after, CRLF and tabs as the file has.

```bibtex
@misc{bb_rendefacil_termo_2022,
	author       = {{Banco do Brasil S.A.}},
	shortauthor  = {BB},
	title        = {Termo de Adesão e Ciência ao {BB Rende Fácil}},
	year         = {2022},
	date         = {2022-03},
	howpublished = {Adhesion term in force from March 2022, delivered under the Access to Information Law, NUP 18882.002114/2026-19},
	note         = {Second version in force from May 2026},
	annotation   = {Opt-in: the client signs to authorize BB to apply the positive checking balance on the day it becomes available and to redeem automatically to cover the account. Daily liquidity, 721-day maturity, minimum R\$0.01. Remunerated at a share of the DI that rises with the holding period: 10, 20, 30, 50 and 100 percent for deposits held up to 60, 120, 180, 360 and 721 days, the reached share applying to the whole deposit. The term does not name the instrument. Heading printed as "Termo Adesão e Ciência" (without "de"). PDF created 2022-09-22; received 2026-11-03; staged in DATA_RAW/issuers/sweep_programs/raw/bb/lai_18882_002114/.}
}
```

**Verify and report:** (i) the key occurs once in the file; (ii) line count before and after, and a diff showing only the insertion (every other byte identical, CRLF preserved, no BOM added); (iii) biber on a throwaway document with `\nocite{bb_rendefacil_termo_2022}` and the paper's `biblatex` options (copy them from `V_Friction.tex`'s preamble), 0 warnings; (iv) the rendered list entry, as in your 15:08 report. Do not run the paper's own build. Nothing else in the file changes.

## Task JD1: judicial deposits inside type 4

You found it and you have the DFP cache and the Cosif extracts open, so it is yours. Read-only on the pipeline; new files only. The Cosif extracts are enough for steps 1 to 3; do not read the market panel while the other sessions run.

**Why it matters.** `dep_a4` is the bank's time-deposit stock in the demand model and the own time-deposit pool $Q^{4}$ of the switching-cost design, and `rate_a4_national` is its price. Judicial deposits are neither chosen nor priced by the depositor. If they are half of BB's stock, BB's type-4 share is overstated by that half and its measured rate is a blend of a retail rate and a court-set remuneration. Caixa holds the federal judicial deposits by law, so the same question applies there, and to any state bank that holds state court deposits.

```
jd1[5]{step,what,output}:
  1,"The Cosif plan of accounts (bcb_cosif): list every subaccount under 4.1.5 (depósitos a prazo) with its name, and say which, if any, carries judicial deposits or other non-retail time deposits (interfinancial, court, government). Do not guess from names alone; read the plan's description of each","a table in the reply"
  2,"For every conglomerate in the panel's head-institution list and every month 2016-01 to 2024-12, the balance of each 4.1.5 subaccount from the Cosif balancetes, and its share of the 4.1.5 total","Rout/cosif_415_subaccounts_monthly.csv"
  3,"If a judicial subaccount exists: the matching interest-expense subaccount under 8.1.1 (or wherever the plan books it), so the implicit rate can be recomputed ex judicial. Report whether expense is separable","in the reply, with the account codes"
  4,"If no subaccount separates them: the judicial share from the DFP for BB (you have 2019 to 2025), Caixa, and any state bank whose DFP names 'depósitos judiciais', by year, with the page cited; a quarterly series by linear interpolation with a flag column","Rout/judicial_deposits_share.csv (conglomerate, quarter, share, source, flag)"
  5,"One paragraph: is the judicial share separable in Cosif, and for which banks does it exceed 10 percent of type 4 in any quarter. No pipeline edit, no \\pn key","Rout/judicial_deposits_report.md"
```

**Flag:** if step 1 shows that the Cosif 4.1.5 total the panel uses already excludes judicial deposits (that is, `dep_a4` tracks a subaccount rather than the group total), say so first and stop; your BB3 check 3 (`dep_a4` tracks the entity 4.1.5 stock plus R\$9 to 11 bn) suggests it does not, but check.

## Honesty flags (keep visible)

- The term does not name the instrument. Do not write "CDB" for Rende Fácil anywhere until item 4 is answered.
- The launch date comes from press (2022-03-03) and BB's phrase "a partir de março/2022", not from a launch document. The V1 PDF was created on 2022-09-22, a later copy.
- Whether a predecessor existed between 2016 and February 2022 is unknown (item 3, pending). Item 9 of both terms refers to BB's "fundos automáticos", which may be that predecessor; a reading of one clause, not a finding.
- V2 is outside the sample.
- Listed shares of the DI are the schedule, not realized rates.
- The term says the top tier ends at day 721; the press printed 720. BB-1 keeps the press table; BB-4 has the term's.

## Open for Pedro

1. The 7.2 sentence on BB's sweep: the main thread drafts it once the follow-up on items 3 to 5 is answered or BB5 reports, whichever comes first.
2. The judicial-deposit treatment in the demand model, once JD1 reports.

---

## Record

### What the answer says (main-thread reading, 11-03)

```
facts[7]{item,content}:
  Versions,"two adhesion terms: V1 from March 2022, V2 from May 2026 (outside the sample)"
  Mechanism,"the positive checking balance is applied on the day it becomes available (the term's wording; 'end of the day' is the press'); automatic redemptions cover the account; minimum R$0.01; daily liquidity; maturity 721 days; redemption from oldest to newest; income tax and IOF on redemptions"
  Adhesion,"a signed term ('Autorizo o BB a aplicar'), so opt-in; whether it is presented at account opening is item 5, pending"
  Remuneration V1,"share of the DI rising with each deposit's holding period: 10% (1-60 days), 20% (61-120), 30% (121-180), 50% (181-360), 100% (361-721); the higher share applies to the whole deposit once reached; tier changes apply only to new applications (item 8)"
  Remuneration V2,"2% (1-90), 5% (91-180), 10% (181-270), 60% (271-360), 100% (361-721)"
  Instrument,"not named in either term; item 4, pending"
  Predecessor,"BB says two versions only; whether a sweep product existed between 2016 and February 2022 is item 3, pending"
```

### The tasks as originally set (14:14), for provenance

- **BB1.** Copy both PDFs untouched to `DATA_RAW/issuers/sweep_programs/raw/bb/lai_18882_002114/` with ASCII names; sha256 of each; Appendix A as `bb_answer_18882.002114.txt` and Appendix C as `bb_request_18882.002114.txt`; one source-log row per file (publisher "Banco do Brasil (via LAI, SIC BB)", accessed 2026-11-03, licence "not stated"); master log by the convention found (append, as it turned out).
- **BB2.** Rows BB-4 (V1) and BB-5 (V2) in `sweep_products.csv` in its existing columns; BB-1's verification upgraded to "confirmed by the primary term"; a NOTES.md section. Applied 15:08 after Pedro's OK of 14:48, with options (a) to (c).
- **BB3.** Three checks on existing extracts: BB's DFP notes (and 20-F if cached; none is), BB's Cosif monthly balances 2021-01 to 2023-12 for 4.1.5, 4.1.1, 4.1.2 and repurchase agreements against Itaú, Bradesco, Santander and Caixa (figure `Rout/fig_bb_rende_facil_cosif.pdf`), and the panel's `dep_a4` and implicit rate 2020Q1 to 2024Q4 (`Rout/bb_rende_facil_booking.csv`). Script `diagnostics/diag_bb_rende_facil_booking.py`; data `Rout/bb_rende_facil_cosif_monthly.csv`.
- **BB4.** The bibliography entry, checked against the house style and the sibling entries; the handoff's draft was corrected to the legal name, a `date` field, braced product name, and "held up to" tier wording.

### Outputs on disk (from your 15:25 reply)

`diagnostics/diag_bb_rende_facil_booking.py`; `Rout/fig_bb_rende_facil_cosif.pdf`; `Rout/bb_rende_facil_cosif_monthly.csv`; `Rout/bb_rende_facil_booking.csv` (100 rows, five conglomerates × 2020Q1 to 2024Q4); `DATA_RAW/issuers/sweep_programs/sweep_products.csv` (42 rows); `NOTES.md` (new BB section); `source_log_rows.csv` (32 rows) and `DATA_RAW/source_log.csv` (279 rows). `Rout` is `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/Rout`.

### DFP Note 17, BB Banco Múltiplo, R\$ bn, 31 December (your check 1)

```
dfp_note17[10]{year,time_deposits,judicial,domestic_ccy,maturity_1to3y,liquidity_clause,dfp_page}:
  2016,194.9,—,—,20.4,51.1,p296
  2017,184.4,—,—,18.1,45.3,p300
  2018,198.1,—,—,17.4,42.4,p321
  2019,224.0,153.9,52.3,16.3,48.2,p331 / 2020 DFP p340
  2020,269.4,164.4,82.3,14.9,77.2,p340
  2021,307.6,183.2,101.2,15.9,97.1,p343
  2022,409.5,213.7,170.6,91.7,101.1,p147
  2023,478.7,272.2,181.7,89.5,114.6,p158
  2024,516.4,247.7,237.3,119.8,143.7,p130
  2025,560.7,263.4,269.5,138.4,160.9,p133-134
```

Side note you recorded: every cached DFP PDF is padded with zero bytes to 8, 16 or 32 MiB after a valid `%%EOF`; the text is complete; nothing to fix.

## Appendix A. BB's answer (verbatim, as Pedro pasted it)

```
Prezado(a) Sr(a)., Encaminhamos-lhe resposta da Unidade Captação e Investimentos do Banco do Brasil ao seu pedido de informação: "Prezado Pedro, Os termos de adesão ao produto BB Rende Fácil se encontram em anexo. Foram 2 versões: a primeira a partir de março/2022 e, a segunda, a partir de maio/2026. Atenciosamente, Rodolpho Petek Gerente" Atenciosamente, Serviço de Informação ao Cidadão do Banco do Brasil SICBB Recurso Conforme a Lei 12527/11 em seu artigo Art. 15, no caso de indeferimento de acesso a informações ou às razões da negativa do acesso, poderá o interessado interpor recurso contra a decisão no prazo de 10 (dez) dias a contar da sua ciência. Parágrafo único. O recurso será dirigido à autoridade hierarquicamente superior à que exarou a decisão impugnada, que deverá se manifestar no prazo de 5 (cinco) dias.
```

## Appendix B. The two terms (main-thread transcription of the clauses that matter; confirmed clause by clause by your text extraction)

Common to both: the client authorizes BB to apply the positive available checking balance in BB Rende Fácil on the day it becomes available and to redeem automatically to cover insufficient balances; minimum application and redemption R\$0.01; conservative profile, daily liquidity, not counted in suitability; remuneration a share of the DI (DI over, 252 business days, CETIP) adjusted by each deposit's holding period, and once a deposit reaches a new tier the share applies to the remaining capital from the initial date to the redemption date; maturity 721 days; tiers may change for new applications only (item 8); Rende Fácil redemptions come before other automatic-redemption applications except BB's "fundos automáticos" (item 9); redemptions from oldest to newest (item 13); income tax and IOF on redemptions; after a full redemption plus a rule change BB may suspend the sweep until a new term is signed (item 14); the client or BB may cancel at any time.

| holding period (days) | V1, from March 2022 | holding period (days) | V2, from May 2026 |
|---|---|---|---|
| 1 to 60 | 10% of the DI | 1 to 90 | 2% |
| 61 to 120 | 20% | 91 to 180 | 5% |
| 121 to 180 | 30% | 181 to 270 | 10% |
| 181 to 360 | 50% | 271 to 360 | 60% |
| 361 to 721 | 100% | 361 to 721 | 100% |

## Appendix C. Pedro's request (verbatim)

```
Prezados, Com base na Lei nº 12.527/2011, solicito as seguintes informações sobre o produto BB Rende Fácil (aplicação automática do saldo da conta corrente), para o período de janeiro de 2016 a dezembro de 2024: 1. Todas as versões do regulamento, termo de adesão ou contrato do BB Rende Fácil oferecidas aos clientes pessoa física no período, com a data de início de vigência de cada versão. 2. As tabelas de remuneração aplicadas em cada versão: a taxa (por exemplo, percentual do CDI) por faixa de saldo e por prazo de permanência, com as datas em que cada tabela vigorou. 3. A data de lançamento do produto e, se for o caso, a data de encerramento ou de substituição por outro produto. 4. O tipo de aplicação em que o saldo é investido (por exemplo, CDB, compromissada ou outro) em cada versão. 5. Se a adesão é automática na abertura da conta ou depende de opção do cliente, e se isso mudou no período. Os documentos dos itens 1 e 2 já foram entregues aos clientes à época, por isso entendo que não estão sujeitos a sigilo comercial. Caso alguma parte seja considerada sigilosa, peço que as demais sejam fornecidas e que a parte restrita seja indicada com o fundamento legal. Prefiro receber os arquivos em formato digital (PDF ou planilha).
```
