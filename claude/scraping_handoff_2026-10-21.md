# Handoff: open scraping tasks (merged; updated 2026-10-26 after the B3 and Pix LAI answers and your status of 15:00)

**To:** scraping agent. **From:** main thread, for Pedro.

**This version folds in** your updates of 10-21 (17:50) and 10-22 (16:10 and 19:55), B3's answer to request B3PF0031927, and the BCB's denial of the Pix LAI 18810.025656/2026-11. The earlier round records stay in `DATA_RAW/_requests/`.

**Rule:** V_Full.tex, V_Friction.tex and the decks are edited by the author only. Deliver raw files, documentation and source-log rows, with no analysis.

## Rules

- Official sources first. No guessed endpoints.
- **`DATA_RAW` is organised by source** (`DATA_RAW/<source>/<dataset>/`, index `DATA_RAW/README.md`, one master `DATA_RAW/source_log.csv`). Scripts read it through `utils/raw_paths.py`.
- Raw files stay untouched. Each dataset has its own `NOTES.md` and its source-log rows. Files that an agent compiles are labelled as such and are not raw.
- If a source does not exist or is not public, say so.
- **Bot walls and terms of use are respected.**
  - Tesouro Direto and B3 are collected by hand, by Pedro. B3 forbids automated collection.
  - Latinobarómetro and Findex files stay internal.
- The analysis window ends 2024Q4. Later months come along with downloads but are never re-run or gap-filled.

## Delivered (record)

```
delivered[9]{round,content}:
  S1-S5 (10-20),"CCS adults with a bank relationship; household surveys; Pix by municipality; usage proxies; DOC/TED/Pix fee rules"
  S6-S10 (10-21),"posted rates; sweep programs; outside options (Tesouro Direto, B3, national series); trust in banks; PNAD TIC and POF"
  ESTBAN per branch,"BCB/estban_agencia/ (2013-01..2026-06)"
  Holds H1-H4 (10-22),"posted rates extended (Nubank RDB, big banks' prepaid); sweeps (no June-2021 norm found); trust (Findex fin11f = 4.069, Latinobarometro, Edelman); SGS 25109-25132, consumidor.gov.br crosswalk, Findex microdata"
  Type-5 rates (10-22),"issuers/posted_rates/posted_rates.csv, 140 rows; Pedro's decisions D1-D3 are now coding task T5"
  consumidor.gov.br,"2016-2024, 78 files, 99.84% of complaints matched to an IBGE code"
  Edelman,"Brazil trust in financial services, 2012-2025"
  Findex,"2014 and 2024 verified"
  10-23 (S12, H6),"iti evidence -> D4 final (100% of CDI); Shopee stated zero, Ame implied zero (now in T5); Neon, Stark, Safra, MagaluPay, FitBank, Celcoin GAP; bitz 2020-09..2021-05 press only (= fallback); posted_rates.csv now 162 rows; master source log 258 rows"
```

Account fees moved to the Fees agent (`claude/fees_handoff_2026-10-22.md`) on 10-22.

## Tasks

```
tasks[2]{id,target,order}:
  S13,"Pix by participant institution: the public route only",first (item 1 needs no one else)
  S11,"B3 individual investors by state: the two sources B3 pointed to",second (step 1 needs Pedro's hand download)
```

### S11. B3 individual investors: follow B3's two pointers

**B3's answer to request B3PF0031927** (Atendimento ao Investidor, shared by Pedro on 10-26):
- B3 will not produce the custom extraction. The investor profile on b3.com.br "possui uma limitação de datas".
- It points to two sources:
  1. **MUB3, Museu da B3, reference collection:** https://mub3.org.br/centro-referencia/acervo. Search "pessoa física"; the site lists historical "books" on individual investors.
  2. **B3 investor relations:** https://ri.b3.com.br/pt-br/, then *Kit do Investidor > Banco de Dados*. This downloads a spreadsheet of historical data, with the tabs **"Depositária"** and **"Listados"**.

**What we already hold** (S8(b); paths in `DATA_RAW/README.md`):
- by-UF snapshots for 2026-09 (Pedro's manual CSVs), 2023-09 (Wayback) and 2020-03 (study PDF);
- regional annual counts 2018–2025 from the quarterly Books (chart labels only);
- national annual counts 2016–2026.

**The target:** individual investors (CPF counts) by state × year, ideally by quarter, 2016–2024, with custody value where available. It measures a local outside option for depositors.

**Step 1, the RI spreadsheet (by Pedro).**
- Give Pedro the exact click path and the file name to save.
- Pedro downloads it by hand, because of the B3 automation rule.
- Save it under `DATA_RAW/B3/ri_banco_de_dados/` with its download date.
- Then document the "Depositária" and "Listados" tabs in `NOTES.md`:
  - each series, its unit (accounts, CPFs, custody value in R$), frequency, first and last date;
  - the individual / institutional split;
  - any state or regional breakdown.

**Step 2, the MUB3 collection.**
- First read MUB3's terms of use and `robots.txt`. If automated access is not allowed, list what Pedro should download by hand instead.
- Search "pessoa física" and catalogue every result: title, date, type, URL, and whether it can be downloaded.
- For the items dated 2016–2024, especially the "books" on individual investors, record which pages carry state or regional tables.
- Save the files with a state table under `DATA_RAW/B3/mub3_acervo/`.
- **Optional, labelled as agent-compiled:** transcribe each state table into one CSV, one row per state × date × measure, with the source page.

**Step 3, the coverage table.**
- Lay out every B3 source we hold, plus the new ones: source × year × geography (national, region, state) × measure (CPF counts, accounts, custody value).
- State plainly whether state × year counts for 2016–2024 are now reachable, and which years remain missing.

**Acceptance:**
- Raw files saved, with their `NOTES.md` and source-log rows.
- The catalogue and the coverage table exist.
- Nothing collected by automation from b3.com.br or ri.b3.com.br.

### S13. Pix by participant institution: public sources only

**Why.** The BCB denied LAI 18810.025656/2026-11 (aggregate Pix statistics by participant institution) after the appeal.
- **Grounds:** secrecy (Lei 12.527, art. 22, with Decreto 7.724, art. 5, § 2) and additional work (art. 13, III).
- **Pedro's decision (10-26):** no further appeal. Take only what the BCB publishes.
- **Use in the papers:** a bank-level measure of Pix adoption would let Pix intensity vary by institution. V_Friction 7.1.2 proposes intensity interacted with deposit and firm type, the convenience channel.

**What to collect:**

1. **`ChavesPix`** (Olinda `Pix_DadosAbertos`, the official API; scripting it is allowed).
   - **Content:** month-end stock of Pix keys by participant, with the fields `Data`, `ISPB`, `Nome`, `NaturezaUsuario` (PF/PJ), `TipoChave` and `qtdChaves`, per the open-data documentation.
   - **Request:** every month-end from 2020-11 to 2024-12, one request per month with the `Data` parameter.
   - **Report:** the first month the API actually serves, and any month it refuses.
2. **`EstatisticasTransacoesPix` and `EstatisticasFraudesPix`:**
   - check whether either has any participant dimension (ISPB or name);
   - as documented, `EstatisticasTransacoesPix` breaks down by payer and receiver type, region, age, initiation method and nature, with no participant field.
   - Collect them only if a participant field exists. Otherwise say so in the notes.
3. **The BCB Pix statistics pages** (estatísticas do Pix on bcb.gov.br) and any downloadable file or dashboard they link:
   - look for transactions or users by participant, such as rankings or shares;
   - if only the current version is published, look for Wayback captures across 2020-11 to 2024-12.
   - No guessed endpoints, and no scraping of a dashboard that its terms forbid.
4. **Linking participants to conglomerates:**
   - map ISPB to CNPJ and to the panel's conglomerate code (`C00…`);
   - use the Pix participant rosters already held (`pix_roster_history.csv`, from `scrape_pix_roster.py`) and the BCB's institution registers;
   - report the share of keys that map to a conglomerate in the panel.

**Deliver:**
- the raw files under `DATA_RAW/BCB/pix_participantes/` (or a name that fits the layout), with `NOTES.md` and source-log rows;
- a repo script for item 1, in the style of `scrape_national_series.py`;
- a coverage note: which per-participant Pix measures are public, for which months, and what is not public (transactions by participant, if none is).

**Flags:**
- **Keys are a stock of registrations, not usage.** One person can hold keys at several institutions, and a CPF key sits at one institution only.
- **The roster gives participation, not intensity.**

## On hold

Nothing. H6 was done on 10-23 and H7 (the WVS form) was closed by Pedro on 10-22.

## Waiting on outside answers (track only)

```
pending[6]{request,protocol,subject}:
  LAI,18810.025400/2026-04,"CCS relationships by institution x municipality (still open)"
  Ouvidoria,18810026133202684,request about the open LAI 18810.025400/2026-04
  LAI,18810.025399/2026-18,CDB/RDB registry (B3/Cetip) by institution
  LAI appeal,18810.025439/2026-13,"tariff history per institution: denied 10-22 (Decreto 7.724, art. 13, III); first-instance appeal filed by Pedro 10-22"
  LAI,18810.025440/2026-48,CDB/RDB by issuer x investor state + ESTBAN verbete 907
  LAI,18882.002114/2026-19,BB Rende Facil terms (filed 10-22)
```

```
closed[3]{request,protocol,outcome}:
  LAI appeal,18810.025656/2026-11,"Pix statistics by participant institution: denied (Lei 12.527 art. 22 + Decreto 7.724 art. 5 para. 2; art. 13 III); Pedro will not appeal again (10-26); public route = task S13"
  B3,B3PF0031927,"answered (shared 10-26): no custom extraction; pointers to MUB3 and the RI Banco de Dados (task S11)"
  LAI,Pix (NUP not recorded),"dropped by Pedro on 10-22"
```

When an answer arrives, save it under the matching source folder and add a source-log row.

## Housekeeping (for Pedro)

- **Junk file:** `DATA_RAW/issuers/posted_rates/evidence/iti_termo_20201125.html` is an icon URL saved by mistake. No row cites it, so it can be deleted.
- **Agent scratch** under `C:\Users\pedro\AppData\Local\Temp\claude\`: `scrape_T5`, `scrape_E2`, `edel_check` and `findex_check` can go. Keep `fee_probe` until the Fees agent's F0 has copied the CDX list and the sample page.

## Open questions for Pedro

1. **S11 step 1:** download the B3 RI spreadsheet yourself when the agent gives you the click path.
