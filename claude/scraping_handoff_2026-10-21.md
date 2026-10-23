# Handoff: open scraping tasks (merged, 2026-10-21)

**To:** scraping agent. **From:** main thread, for Pedro.

**Replaces:**
- `breakfast_followups_scraping_handoff_2026-10-21.md` (S6 to S10)
- your reply `breakfast_followups_scraping_reply_2026-10-21.md`

Both rounds (S1 to S5 and S6 to S10) are delivered. Their status files stay the record:
- `DATA_RAW/devo_followups_2026-10-20/STATUS.md`
- `DATA_RAW/breakfast_followups_2026-10-21/STATUS.md`

**Rule:** V_Full.tex, V_Friction.tex and the decks are edited by the author only. Deliver raw files, documentation and source-log rows, with no analysis.

## Rules (unchanged)

- Official sources first. No guessed endpoints.
- Raw files untouched, one folder per task, each with `NOTES.md` and its source-log rows.
- If a source does not exist or is not public, say so.
- **Bot walls and terms of use are respected.**
  - Tesouro Direto is collected by hand, by Pedro.
  - B3 forbids automated collection.
  - Latinobarómetro files stay internal.

## Delivered

```
delivered[6]{task,where}:
  S1-S5,DATA_RAW/devo_followups_2026-10-20/
  S6-S10,DATA_RAW/breakfast_followups_2026-10-21/
  ESTBAN per branch,DATA_RAW/estban_agencia/ (2013-01..2026-06)
  national series,DATA_RAW/national_series/ (Ibovespa; Selic SGS 11/4390/4189)
  scrapers,"scrape_national_series.py, scrape_estban_branches.py, scrape_bcb_ccs_relationships.py, scrape_consumidor_gov.py"
  Tesouro Direto,S8_outside_options/tesouro/live_manual/ (Pedro's manual download)
```

## Tasks

```
tasks[2]{id,agent,target,when}:
  P1,scraping,"Pix by municipality: re-download 2026-10 once the month closes",early November 2026
  P2,scraping,"re-run scrape_national_series.py for final monthly values",after each month end
```

**P1.** 2026-10 is partial in the current files, and the last full month is 2026-09. Replace only that month's file, and log the new capture.

**P2.** The current month is partial in the monthly series. Re-run after the month closes, and log it.

## On hold: Pedro decides

None of these runs until Pedro says yes.

```
hold[4]{id,question}:
  H1,"S6: are dated rules enough, or is a monthly series per firm needed? Chase Nubank RDB rates by maturity, Inter CDB (JS-rendered, needs a browser) and the big banks' digital accounts?"
  H2,"S7: chase Itau's own regulamento and the June 2021 BCB norm that ended automatic fund sweeps? Draft a LAI request for BB Rende Facil terms?"
  H3,"S9: fill the WVS download form, and download Edelman through a browser? Clarify whether the Findex 2024 trust value (5.672) is a share of all adults or of the unbanked."
  H4,"From 10-20: a Findex World Bank account; IBGE codes and full history for consumidor.gov.br; SGS series 25105-25119; monthly snapshots of the branch and correspondent registers."
```

## Waiting on outside answers (track only)

```
pending[8]{request,protocol,subject}:
  B3,B3PF0031927,"individual investors by state x month or quarter from 2016, municipality if available, by product, with custody value"
  LAI,18810.025400/2026-04,"CCS relationships by institution x municipality (still open)"
  LAI appeal,18810.025656/2026-11,appeal filed
  Ouvidoria,18810026133202684,request about the open LAI 18810.025400/2026-04
  LAI,18810.025399/2026-18,CDB/RDB registry (B3/Cetip) by institution
  LAI appeal,18810.025439/2026-13,"tariff history per institution: denied 10-22 (Decreto 7.724, art. 13, III); first-instance appeal filed by Pedro 10-22"
  LAI,18810.025440/2026-48,CDB/RDB by issuer x investor state + ESTBAN verbete 907
  LAI,18882.002114/2026-19,BB Rende Facil terms (filed 10-22)
```

When an answer arrives, save it under the matching task folder and add a source-log row.

## Open questions for Pedro

1. Record the missing protocol numbers (the Pix LAI, and the CDB registry, tariff and CDB-by-state requests) so they can be tracked here.

## Response from the scraping agent (2026-10-21 17:50)

- **P1 and P2 are DROPPED, per Pedro (2026-10-21):** "No need for 2026, recall I need only until 2024." The analysis window ends 2024Q4. October 2026 and later months come along with downloads but are never re-run or gap-filled. Nothing is scheduled.
- **Three of the "not recorded" protocols are recorded** (the 10-21 reply omitted them by mistake):

  ```
  pending_resolved[3]{request,protocol}:
    LAI CDB/RDB registry (B3/Cetip) by institution,18810.025399/2026-18
    LAI tariff history per institution,18810.025439/2026-13
    LAI CDB/RDB by issuer x investor state + ESTBAN verbete 907,18810.025440/2026-48
  ```

  Only the Pix LAI NUP is still unrecorded, so question 1 reduces to that one number.
- **Recorded:** the LAI appeal 18810.025656/2026-11 and the Ouvidoria request 18810026133202684 (on LAI 18810.025400/2026-04).
- Holds H1–H4 are untouched until Pedro says yes.

## Update from the scraping agent (2026-10-22 16:10)

### DATA_RAW is now organised BY SOURCE (Pedro's decision; moved 16:04)

- **Layout:** `DATA_RAW/<source>/<dataset>/`. The index is `DATA_RAW/README.md`; the one master log is `DATA_RAW/source_log.csv` (207 rows, with `round`, `task` and a `location` column that exists on disk for every row); the old → new map is `DATA_RAW/_move_plan_2026-10-22.csv`.
- **Round records** (STATUS, BRIEFS, each task's NOTES) are in `DATA_RAW/_requests/`. The paths inside them are historical.
- **Code:** every script reads `DATA_RAW` through `utils/raw_paths.py`. With Pedro's OK, the path lines (only) of your three scripts were switched: `make_branch_estban.py` → `raw_paths.BCB_ESTBAN_AGENCIA`; `make_pix_exposure.py` and `make_tesouro_mca.py` → `raw_paths.BCB_PIX_MUNICIPAL_RAW` and `raw_paths.TESOURO_INVESTIDORES_LIVE`. Each target exists and holds the same files (Pix 72 months, ESTBAN 164 items, Tesouro 11 files). Re-run them at your convenience to confirm.
- The new locations of what you use:
  - `BCB/pix_municipal/raw/`
  - `BCB/estban_agencia/`
  - `TesouroNacional/investidores_td/live_manual/`
  - `issuers/posted_rates/` (S6 + H1 + H1b)
  - `issuers/sweep_programs/` (S7 + H2)
  - `IBGE/pnadc_tic/`, `IBGE/pof/`
  - `Latinobarometro/` (INTERNAL)
  - `WorldBank/findex/` (INTERNAL)
  - `Ipeadata/ibovespa/`, `BCB/selic_sgs/`

### Holds, now run (Pedro approved 10-21)

- **H1 + H1b (posted rates, `issuers/posted_rates/posted_rates.csv`, 99 rows):**
  - **Nubank RDB:** launched June 2019. The F-1/A gives 102–126% of CDI for 3–36 months (2020–21); 20-F weighted averages run 104–107% for 2021–2024. There is no public per-maturity table.
  - **Big banks' CHECKING balances:** they cannot be remunerated by law, so that is not a finding.
  - **Big banks' PREPAID accounts:**
    - Bradesco **bitz** paid 100% of CDI on balances above R$10 (2022; closed 2023-04).
    - Santander **Superdigital:** no rate on any saved page (closed 2024-11).
    - Itaú **iti:** 100% of CDI until between 2023-07 and 2023-09.
    - BB: the Ourocard prepaid contracts have no remuneration clause.
    - Caixa: no general prepaid account.
- **H2 (sweeps):**
  - The June-2021 "end of automatic fund sweeps" norm was NOT found in a full scan of BCB/CMN acts; it may be Santander-specific.
  - Itaú's % of CDI band table is not public in any Itaú document 2016–2024.
- **H3 (trust):**
  - **Findex correction:** lack of trust is **fin11f = 4.069** (2024, % of all adults ≈ 30% of the unbanked). 5.672 is fin11d = insufficient funds.
  - **Latinobarómetro:** confidence in banks in 2015, 2018, 2020, 2023; the UF is derivable in 2015, 2017 and 2018.
  - **Edelman:** an annual Brazil financial-services trust series for 2012–2021, charts only.
- **H4:**
  - SGS 25109–25132 are in `BCB/ccs_relationships/current`; 25105–25108 are not served by the API.
  - The IBGE crosswalk script for consumidor.gov.br is ready (99.81% match on samples). The site was down (HTTP 504) from 10-21 18:00 through 10-22.
  - Findex microdata 2017/2021 are in and verified; 2014/2024 come from Pedro.

## Update from the scraping agent (2026-10-22 19:55)

- **Type-5 balance rates (the coding agent's request of 16:40) are DELIVERED.** The reply is `claude/scraping_reply_2026-10-22_type5_rates.md`, addressed to the coding agent. It carries Pedro's decisions D1-D3 (19:35):
  - D1: a stated or implied zero is a nominal 0% rate (Stone, C6, Alelo, Cora, PayPal); every other issuer keeps the 100%-of-CDI fallback;
  - D2: the zeros extend back to 2020;
  - D3: Banco Original's type-5 balances are PicPay's (2020Q2-2024Q3, then C0088022).
  Data: `DATA_RAW/issuers/posted_rates/posted_rates.csv` (140 rows; new columns `conglomerate`, `entity_cnpj`).
- **Also done today:**
  - consumidor.gov.br 2016-2024 (78 files; 99.84% of complaints matched to an IBGE code);
  - Edelman Brazil trust in financial services, 2012-2025;
  - Findex 2014 and 2024 verified;
  - BB Rende Fácil LAI filed by Pedro (18882.002114/2026-19);
  - master source log at 240 rows.
- **Started: account fees** (Pedro, 19:38). The note is `claude/account_fees_data_2026-10-22.md`: what we hold, the binding gap (BCB listed tariffs exist only as a 2026 snapshot with last-change dates), and expansion routes A-H.
- **Pix LAI NUP:** Pedro dropped it ("don't worry about that"); remove it from the tracking list.
