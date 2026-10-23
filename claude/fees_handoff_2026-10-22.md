# Handoff: account fees, 2016–2024 (new Fees agent)

**To:** Fees agent. **From:** main thread, for Pedro. **Date:** 2026-10-22.

**Replaces:** the scraping agent's note `account_fees_data_2026-10-22.md` (20:10). Its content is carried over below, and the note itself is deleted.

**Target object:** what a customer paid to **keep an account**, by bank and year, 2016–2024. That means the monthly maintenance fee:
- the "pacote padronizado de serviços" (Pacotes Padronizados I–IV);
- a bank's own "cesta";
- an inactivity fee;
- for firms, the maintenance fee of an active or inactive account.

For digital and prepaid issuers the fee is usually zero.

**Why it matters for the papers:** a fee for keeping an account is a holding cost on demand deposits, and a cost of holding a second bank relationship, so it bears on switching costs (V_Friction, Section 7.2). The existing keys `desc.fees.*` (fee revenue over deposits, from COSIF) cover only the collected side.

## Rules

- **V_Full.tex, V_Friction.tex and the decks are edited by the author only.** Deliver data, documentation, tables and keys. Write no paper text.
- **Numbers reach the papers only as `\pn` keys** through `make_paper_numbers.py`, following the existing naming (`block.table.column.spec`). Add keys only when Pedro asks.
- **Raw files stay untouched.** Every new raw dataset goes under `DATA_RAW/<source>/<dataset>/` (the by-source layout of 10-22; index in `DATA_RAW/README.md`), with a row in the master `DATA_RAW/source_log.csv` and a `NOTES.md` per dataset.
- **Collecting:**
  - official sources first;
  - no guessed endpoints;
  - respect bot walls and terms of use;
  - Wayback requests sequential and gentle (raw `id_` captures).
  - If a source does not exist or is not public, say so.
- **Code:**
  - repo scripts, standalone, PEP 8;
  - relative paths only, raw paths through `utils/raw_paths.py`;
  - commented in the style of the existing `fees/` scripts.
  - Do not change the panel pipeline. That is the coding agent's or Pedro's call.
- **The analysis window ends 2024Q4.** Later months may come along with downloads, but are never filled or extended.
- **Before any `rm` or `mv`,** state the exact command and wait for Pedro's confirmation.

## Basis: what is held (inventory of 10-22, 20:07, re-checked by the scraping agent)

```
held[10]{id,path,measures,coverage,account_keeping_content}:
  1,BCB/Tarifas/processed/tariff_vigencia_history.csv (fees/scrape_bcb_tariff_vigencia.py; BCB Olinda),"listed maximum price + data_vigencia (date in force)","one snapshot 2026-06-22: 74,310 rows, 1,279 institutions, 136 service codes","PF: Packages I-IV (codes 1501-1504; ~700 institutions; 1501 median R$15, 27% at zero). PJ: 0402 active-account maintenance (764), 0403 inactive (722). No PF maintenance code: essential services are free by rule"
  2,BCB/Tarifas/processed/tarifas_panel_institution_long.csv,same as 1,"two --test runs (2026-02/03), 38 institutions",adds nothing
  3,BCB/Tarifas/processed/cosif_service_fees_institution.parquet (fees/scrape_cosif_service_fees.py),"collected service-fee revenue (COSIF 717) + deposits","institution x month 2013-01..2026-03, ~1,400 institutions/year","total fees only; PF/PJ split only from 2025-01; the 2025 chart-of-accounts change breaks the series (cut at 2024)"
  4,BCB/Tarifas/processed/cosif_fee_quarterly_conglomerate.csv,fee revenue / deposits,"conglomerate x quarter 2013Q1..2026Q1",as 3
  5,FirmDisclosures/CVM/cvm_fee_income.csv (fees/scrape_cvm_fee_income.py),reported fee/service revenue,"9 listed banks 2018-2026 (thin before 2022); no Caixa, Nubank, C6",no account line
  6,panel market_panel_with_fees.* (stage 7; opt-in; no estimator reads it),COSIF ratios + listed_fee_* + stickiness,2013-2025,"listed_fee_* is the June-2026 snapshot repeated back to 2013 (no time variation); listed_fee_atm_withdrawal_pf is really Package I (1501) and listed_fee_cheque_stop_pf is code 1203 (README flags the mislabels)"
  7,DATA_RAW/IBGE/pof (2008-09 and 2017-18),household spending on bank fees,two cross-sections,"2017-18 item codes 2600501-2600504 (manutencao de conta, pacote de servico, tarifa de conta); microdata not yet opened"
  8,DATA_RAW/BCB/normativos_pix_ted_doc/docs/wb_bcb_tarifas_20180205.txt,"BCB consolidated listed prices, private banks (Wayback)",one date (2018-02-05),"Packages I-IV, min-max-mean"
  9,DATA_RAW/consumidor_gov (2016-2024),complaints,complaint level,"problem 'Cobranca de tarifas, taxas, valores nao previstos' (5,453 in 2024-12, all sectors); no bank filter yet"
  10,Open Finance charged prices (Drafts/Deposit Competition/service_fees_notes.md section 1A),charged per-event prices,"2026-06/07 only; scripts and data deleted 10-14",charged price ~0.30 x listed ceiling (median)
```

**Measurable today for 2016–2024:**
- **The listed Package I price** is known for 26.6% of institutions in 2016 and 62.4% in 2024. These are institutions whose current price was already in force, mostly credit cooperatives.
- **None of the large banks.** The current package prices of Itaú (2025-03), BB (2025-09), Santander (2026-03), Bradesco (2026-04) and Caixa (2026-04) all took effect after 2024. The exception is Inter, with R$20 for Package I since 2014.
- **Total fee revenue over deposits** exists for nearly every bank (COSIF), but not the account-keeping part of it.

## The binding gap

- **The BCB publishes listed tariffs only as a current snapshot.** Neither Olinda service (`Informes_ListaTarifasPorInstituicaoFinanceira`, `Informes_ListaValoresDeServicoBancario`) takes a date parameter.
- **What the snapshot implies:** the price at date T is known only when `data_vigencia` ≤ T. Every earlier price is lost.
- **The Internet Archive fills 2003–2018.**
  - The old BCB site `www.bcb.gov.br/fis/tarifas/htms/<CNPJ8>.asp` had one page per institution, with the full PF schedule, "Data de vigência" and "Valor máximo" per service, packages included.
  - Wayback holds 29,320 distinct captures: 2014: 4,221; 2016: 1,929; 2017: 2,323; 2018: 1,990. Almost nothing exists after 2018, when the site moved to the Olinda API, which was never archived.
  - **Checked example:** Itaú, "Posição em 12/05/2017" (capture 20170514140640): Package I R$12,20, II R$19,50, III R$25,40, IV R$38,00, all "vigência 01/03/2017". Itaú's current Package I is R$16,10, effective 2025-03-20.
  - Captures per bank: Itaú 118 (2016: 24, 2017: 12, 2018: 19), Caixa 181, Bradesco 109, Santander 105, BB 95.
- **What stays missing is 2019–2024** for every institution that changed prices between its last 2018 capture and its current effective date. That includes all the big banks.
- **The LAI route is under appeal.** On 10-22 the BCB denied LAI 18810.025439/2026-13 (the full tariff history) under Decreto 7.724/2012, art. 13, III ("trabalhos adicionais de ... consolidação"). Pedro filed the first-instance appeal the same day. It asks for the raw stored history, or year-end values, or only the packages, and at least for the data dictionary and whether the BCB keeps past values. If it succeeds, it covers 2019–2024 for every institution and checks F1 to F3.

## Tasks, in order of value

```
tasks[9]{id,route,output,priority}:
  F0,—,"staged files moved from Temp to permanent locations",first
  F1,A0,"tariff history from the old BCB pages on Wayback, 2003-2018",high
  F2,—,"dated listed-price series per institution x service, with the 2019-2024 holes measured",high
  F3,B,"big banks' own tariff tables on Wayback, 2019-2024",high
  F4,H,"verify and merge the 50 staged issuer account-fee rows",medium
  F5,C,"Procon-SP / IDEC package-price surveys as a validation series",medium
  F6,D+E,"feasibility note: fee revenue per customer per bank-year",medium
  F7,F,"fee disputes by institution (consumidor.gov.br, BCB complaints ranking)",low
  F8,G,"POF 2017-18 bank-fee expenditure items",low
```

### F0. Secure the staged files (first)

Three inputs sit in Pedro's temporary folder and can be lost:

```
staged[3]{path,content,move to}:
  C:\Users\pedro\AppData\Local\Temp\claude\fee_probe\cdx_bcb_tarifas.txt,"CDX list of the 29,320 captures of bcb.gov.br/fis/tarifas/htms/",DATA_RAW/BCB/tarifas_wayback/ (with the sample page)
  C:\Users\pedro\AppData\Local\Temp\claude\fee_probe\ (sample page),"Itau capture 20170514140640",DATA_RAW/BCB/tarifas_wayback/
  C:\Users\pedro\AppData\Local\Temp\claude\scrape_FEE\rows.csv + notes.md,"50 dated issuer account-fee rows (STAGED, not verified)",staging area for F4 (not DATA_RAW until verified)
```

State the copy commands and wait for Pedro's OK. Copy, then confirm the copies before anyone deletes the Temp folder. Add the source-log rows.

### F1. Old BCB tariff pages on Wayback, 2003–2018 (route A0)

- **Script:** `scrape_bcb_tarifas_wayback.py`, standalone, in the style of `scrape_estban_branches.py`. It reads the saved CDX list, fetches the raw `id_` captures sequentially, and saves each capture under `DATA_RAW/BCB/tarifas_wayback/raw/`.
- **Parse:** the schedule table (Latin-1), giving one row per institution × service × "Data de vigência" × "Valor máximo". Keep the capture timestamp and "Posição em" date.
- **Output:** `BCB/Tarifas/processed/tariff_history_wayback.csv` and a `NOTES.md` with the counts.
- **Acceptance:**
  - The Itaú example above is reproduced exactly.
  - Report the counts by year, the institutions covered, and the capture coverage of Itaú, BB, Bradesco, Santander, Caixa, Banrisul, BRB and Inter.
  - Report any page layouts that fail to parse, with an example.

### F2. Dated listed-price series, and the remaining holes

- **Join** F1 with the 2026 snapshot (held item 1) into one dated series per institution × service: an effective date and a price, with the source of each row.
- **Services:** Packages I–IV (1501–1504) for individuals, and active and inactive account maintenance (0402, 0403) for firms.
- **Holes:** for each institution and year 2016–2024, flag whether the price in force is known, and why not when it isn't (gap between the last capture and the current effective date).
- **Report:** the share of 2016–2024 institution-years known, overall and for the big banks named in F1.
- **Output:** `BCB/Tarifas/processed/account_fee_listed_history.csv`.
- **Do not** replace `listed_fee_*` in the panel. That is a pipeline change (open question 2).

### F3. The big banks' own tariff tables, 2019–2024 (route B)

- **Banks:** Itaú, Bradesco, Santander, BB, Caixa, Banrisul and BRB.
- **Sources:** their "Tabela de Tarifas Pessoa Física" pages and PDFs on Wayback, using the S6 method: CDX by URL prefix, raw captures, one row per bank × package × price version, with the capture date.
- **Output:** `DATA_RAW/issuers/bank_tariff_tables/` (raw) and a compiled table.
- **Check:** the F3 prices against F1/F2 wherever both exist, and report any disagreement.

### F4. Issuer account-fee rules (route H): verify, then merge

- **Input:** the 50 staged rows from F0. Highlights:
  - **Free accounts:** Nubank, Neon, Stone, iti, PicPay, Asaas, Caixa Poupança Social Digital, Bitz, Cora, PagBank, Mercado Pago.
  - **Conditional or priced:** C6 (conditional from 2023-04), Santander, ACG/Americanas, Alelo, BB Ourocard.
  - **No statement:** 11 issuers.
- **Verify** each row against its saved evidence file before merging into `DATA_RAW/issuers/posted_rates/account_fees.csv`.
- **Flag:** "grátis" is marketing text, and the contracts can still list tariffs. Neon's 2023 terms, for example, say tariffs may be charged and then waived by usage. Record the contract wording where it exists.

### F5. Consumer-survey package prices (route C)

- **Sources:** Procon-SP "pesquisa de tarifas bancárias" and IDEC, from the live sites or Wayback. Use press only as a pointer.
- **Deliver:** package prices for the largest banks by year, as a validation series for F2 and F3.

### F6. Fee revenue per customer: feasibility note (routes D and E)

- **Numerator:** COSIF 717 (held), plus the account-service fee lines in the 20-F (Itaú, Bradesco, Santander Brasil) and in BB and Caixa results reports where they exist. Browser downloads need Pedro.
- **Denominator:**
  - whether the BCB complaints ranking (quarterly from 2014) carries client counts per institution;
  - the CCS LAI 18810.025400/2026-04, still pending.
- **Deliver:** a note on what can be built, and for which banks and years. Do not build it yet.

### F7 and F8 (low priority)

- **F7, fee disputes:** the share of fee complaints by institution, built locally from the files held. Use consumidor.gov.br problems naming a "tarifa" (× city) and the BCB ranking categories for irregular tariff charges.
- **F8, POF 2017–18:** the household bank-fee expenditure items 2600501–2600504 by state and income. One cross-section, as a household benchmark.

## Flags to keep visible

- **Listed against charged:** listed package prices are ceilings. Many customers hold only the free essential services, and the Open Finance sample put charged prices at about 0.30 of the listed ceiling.
- **Panel mislabels:** `listed_fee_atm_withdrawal_pf` is the Package I price, and `listed_fee_cheque_stop_pf` is code 1203. Anyone reading the stage-7 panel must know this until it is fixed.
- **The 2026 snapshot is not a history.** Never use it for a year before its `data_vigencia`.
- **COSIF breaks in 2025.** Cut the series at 2024.

## Open questions for Pedro

1. **Panel `listed_fee_*`:** replace the repeated 2026 snapshot with the F2 dated series, and fix the labels? That is a pipeline change, for you or the coding agent.
2. **Scope:** is the target the PF package price alone, or also PJ maintenance (0402, 0403)?
3. **F6:** worth the browser downloads (20-F and results reports), or stop at the feasibility note?
