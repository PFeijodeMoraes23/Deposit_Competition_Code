# Bank advertising data: sources, collection and results

Advertising expense per bank per quarter, 2013 onward, for the awareness tier of the deposit
demand model. This note records where the data can and cannot be found, how each source was
checked against the central bank's own accounts, the choices made, and what the collection
scripts produce.

Last updated: 2026-10-05.

**What changed on 2026-10-05.** The central bank answered the appeal with NAMED data: the three
advertising accounts for every institution, monthly, 2013-2024 (section 4). That file and the
public accounts from 2025 are now one source and the main measure (section 3). The statement
notes become a cross-check, and the document listings still missing for twelve banks are no
longer needed.

## Status

| Piece | Script | State |
|-------|----------|--------|
| Central bank accounts, 2013-2024 (file released on appeal) | `build_cosif_lai_advertising.py` | Built 2026-10-05, validated, independently reviewed, outputs written (5.10) |
| Central bank accounts, 2025 onward (public files) | `scrape_cosif_advertising.py` | Built, validated; reads all five institution files since 2026-10-05 (5.1) |
| Securities-regulator filings, 2013 onward | `scrape_cvm_advertising.py` | Built, validated, outputs written |
| SEC annual reports (separate annual panel) | `scrape_sec_advertising.py` | Built, validated, outputs written |
| Caixa's own disclosures and sponsorship, text months | `scrape_caixa_advertising.py` | Built; 142 months, 133 validated against their printed totals, 9 recorded as files that do not close |
| Caixa's 20 scanned months | `scrape_caixa_scan_advertising.py` | Built; 17 validated (one within 5 centavos), 3 not validated |
| Banestes, BASA, Banrisul, BRB own disclosures | `scrape_statebank_advertising.py` | Built, validated; three banks at 100% of published totals, BRB at 94.2% |
| Sponsorship robustness series, BB and BNB | `scrape_sponsorship_bb_bnb.py` | Built, validated, outputs written; open questions in 5.4 |
| Entity crosswalk to market panel codes | `build_advertising_crosswalk.py` | Built, validated, outputs written |
| Registered advertising films (a count, not spend) | `scrape_ancine_ad_films.py` | Built, seven checks pass; kept as a separate quarterly table (5.8) |
| Assembly of the quarterly and annual panels | `build_advertising_panel.py` | Rebuilt 2026-10-05 with the central bank's accounts from 2013 as the main source; checks pass; the quarter-by-quarter assignment of institutions to conglomerates was independently reviewed and its fixes applied (5.7) |
| Statement-note extraction | `scrape_bank_notes_advertising.py` | Built; six banks written 2026-10-04; a cross-check only since 2026-10-05, not a panel source (5.9) |
| Freedom-of-information request to the central bank | filed on Fala.BR | Appeal granted with institution names; file received 2026-10-05 (section 4) |

All outputs go to `paths.AWARENESS_PROC` (`Open-Finance/BCB/Awareness/processed`).

---

## 1. The search for a pre-2025 series

The central bank publishes the advertising accounts for every institution only from January
2025. Before writing any PDF or vision extraction, every public route to 2013-2024 was checked.

### 1.1 Central bank

- **The accounts always existed and were always reported.** The COSIF chart of accounts has
  carried `8.1.7.45.00-9 Despesas de Propaganda e Publicidade`, `8.1.7.42.00-2 Promocoes e
  Relacoes Publicas` and `8.1.7.48.00-6 Publicacoes` since the plan began in 1988-1989 (2017
  archived copy of the complete chart). Filing instructions for document 4010 require every
  account, and a small bank's self-published 2019 filing carries them.
- **Only publication changed.** Comunicado 44.132 (31 October 2025) set publication "com
  abertura ate o seu 4o nivel" from January 2025. No earlier act fixed the published level at
  3; it was practice. Nothing was backfilled: the 2024 public files still stop at the level-3
  total `81700006`.
- **No central bank interface exposes the detail.** IF.data's income statement stops at
  administrative expenses; ESTBAN has no expense verbete (only 710/711/712); the "ifs-balancetes"
  entry on the open-data portal points to the same level-3 files.
- **Self-published full filings are not a route.** None of Itau, Bradesco, Banco do Brasil,
  Caixa, Santander, BTG, Safra, Sicoob, Sicredi, Nubank, Inter, C6, PagBank, Mercado Pago,
  Original or Neon posts its document 4010. Legal publication follows the summary document,
  which has no expense detail.

### 1.2 Securities-regulator filings (CVM)

CVM's structured DFP/ITR files carry advertising as a bank-chosen sub-line, mostly in the DVA
under `7.03.04 Outros`, for about nine listed banks. The notes to the statements are embedded
PDFs, not tagged data, and the FRE has no expense field.

| Bank | Where | Coverage |
|---|---|---|
| Bradesco | DVA 7.03.04.03 "Propaganda, Promocoes e Publicidade" | 2013-2025 quarterly |
| Banco do Brasil | DVA 7.03.04.06 "Propaganda e Publicidade", individual only | 2013-2025 quarterly |
| Itau Unibanco Holding | DVA 7.03.04.02 "Propaganda Promocao e Publicacoes" | quarterly from 2013, Q4 only from 2018 |
| Mercantil do Brasil | DVA 7.03.04.03 | 2013-2025 quarterly |
| BMG | DVA 7.03.04.02 | 2017Q4-2025 |
| Pan | DVA 7.03.04.04/.06 | 2018-2025 |
| Banese | DRE 3.04.03.07/.08/.09, three separate lines | 2013-2024 (2025 filed as zeros) |
| Banpara | DRE 3.04.03.07/.08/.09 | 2013-2020Q2 (chart change removes the lines) |
| Banco do Nordeste | DVA 7.03.04.02 | 2013-2017 full, 2018-2024 gappy, 2025 |

No line in any year for Santander Brasil, BTG, Banrisul, BRB, Banestes, Daycoval, Pine,
Parana Banco, Banco da Amazonia, XP, Inter. Caixa, Safra, Votorantim, the cooperatives,
PagSeguro and Stone do not file with CVM.

### 1.3 SEC annual reports (20-F XBRL)

The companyfacts API omits company-specific elements, so the 20-F instance documents were read
directly. All annual; the 6-Ks carry no tagged income statement.

| Filer | Element | Currency | Years |
|---|---|---|---|
| Santander Brasil | ifrs-full:SalesAndMarketingExpense | BRL | 2015-2025 |
| Itau | ifrs-full:SalesAndMarketingExpense | BRL | 2017-2025 |
| Bradesco | bbd:OtherAdministrativeExpensesAdvertisingPromotionsAndPublicRelations | BRL | 2015-2025 |
| Nu Holdings | nu:BrandingAndAdvertising (2023-25); ifrs-full:SalesAndMarketingExpense (2019-25) | USD, whole group | see element |
| Inter & Co | ifrs-full:AdvertisingExpense | BRL | 2019-2025 |
| XP | ifrs-full:AdvertisingExpense | BRL | 2017-2025 |
| PagSeguro | ifrs-full:AdvertisingExpense | BRL | 2015-2025 |
| StoneCo | ifrs-full:SalesAndMarketingExpense | BRL | 2016-2025 |

### 1.4 State-owned banks' own disclosures

Transparency rules oblige state-owned banks to publish advertising spend. All in nominal reais,
per period (never year-to-date).

| Bank | Coverage | Frequency | Format | Scope | Basis |
|---|---|---|---|---|---|
| Banco do Brasil | 2010-2026 | monthly, half-year files | text PDF to 2021H1, XLSX/ODS after | production plus media | cash paid |
| Caixa | 2010-2026 | monthly | PDF; 2013-Jan 2014 image scans; XLSX from Apr 2020 | media, production, services by agency | "Custos", undocumented |
| Banco do Nordeste | 2013-2026 | monthly | ODS | media plus production | undocumented |
| BNDES | 2011-2025 | monthly | open-data CSV | media plus production | cash paid (not a deposit taker) |
| Banestes | 2016 onward (2015 annual) | monthly, per agency | text PDF | media by type | |
| Banco da Amazonia | 2019 onward; 2016, 2017 partial; 2013-15 and 2018 missing | line records | XLSX/CSV from 2019 | production plus media, gross/tax/net | paid |
| Banpara | 2019 onward | monthly | text PDF | media plus agency services | cash from 05/2023 |
| Banrisul | July 2021 onward only | monthly | text PDF | media plus production, group level | gross |
| BRB | 2014-2025, gaps | quarterly with monthly columns | gazette PDF | advertising, legal notices AND sponsorship bundled | accrual ("total contabilizado") |
| Banese | nothing found | | | | |

The federal audit court's Acordao 1521/2024 obliged BB and Caixa to publish the outlets that
carried paid advertising since 2019; Caixa's resulting files list names only, no amounts.

### 1.5 Federal advertising monitor and SECOM

The IAP archives (hosted by Poder360, 17 annual PDFs to 2016) itemise insertion orders by
federal entity, annually, IGPM-adjusted to 2016 prices, and state that they "nao representam
gastos". SECOM's current tool excludes state-owned companies. So this can only fill 2013-2016
annually, and only for the federal banks.

### 1.6 Mirrors and literature

- Base dos Dados: `bcb.ifdata` and `bcb.estban` mirror the raw files with no finer detail; no
  DFP/ITR or COSIF mirror.
- dados.gov.br: the API requires a gov.br key (HTTP 401 without it).
- Replication packages (Dataverse, Zenodo, GitHub, openICPSR): none found. The closest paper,
  Ribeiro et al. (Congresso Brasileiro de Custos 2019; Contabilometria 2020), hand-coded
  advertising for 19 listed banks, annual 2011-2017, with no data appendix.

---

## 2. Reconciliation against the central bank's 2025 accounts

2025 is the one year every source overlaps with COSIF, so it tells which object each source
measures. COSIF flows were de-cumulated within each half-year.

### 2.1 Filings against COSIF (ratios by quarter of 2025)

| Bank, filing line | Closest COSIF object | Ratio |
|---|---|---|
| Banco do Brasil, individual DVA | institution, advertising account alone | 1.000 every quarter |
| Banco do Nordeste, individual DVA | institution, all three accounts | 1.000 every quarter |
| BMG, individual DVA | institution, all three | 1.000, 1.000, 1.000, 0.961 |
| Bradesco, individual DVA | institution, advertising plus promotions | 1.007-1.020 |
| Bradesco, consolidated DVA | conglomerate, all three | 1.035-1.094 |
| Itau, consolidated DVA | conglomerate, advertising plus promotions | 0.898-0.978 |
| Mercantil, individual DVA | institution, advertising alone | 1.004-1.024 |
| Pan, individual DVA | institution, advertising plus promotions | 0.990-1.018 |

Units hold (R$ thousand, no half-year error), but **each bank puts a different set of accounts
into its line**, so levels are not comparable across banks. Banco do Brasil's conglomerate runs
7-9% above its institution.

### 2.2 Banks' own files against accounting expense

The own files measure outlays, not expense.

- Banco do Brasil paid / filed expense: 0.92, 0.81, 0.77, 0.86, 0.78, 0.85, 0.79 (2013-2019),
  1.04, 0.96, 1.24, 1.13, 1.13 (2020-2024). A level shift from about 0.82 to about 1.17; a
  one-quarter lag does not remove it.
- Banco do Nordeste: 0.35-0.72 (2013-2017), 0.59 (2025).
- Caixa 2025: 1.41 against the advertising account, 0.68 against all three; same-month
  correlation 0.90.
- BRB books sponsorship inside its advertising account (advertising plus sponsorship = 0.999 of
  the COSIF advertising account every quarter).

### 2.3 Quality comparison for banks with both a filing and own files

| Bank | Filing | Own files |
|---|---|---|
| Banco do Brasil | complete 2013-2025, accrual, splices at 1.000; one +16% restatement (2019Q1); Q4-heavy | complete monthly, cash; splices at 1.25-1.45; drifts 0.8 to 1.2 |
| Banco do Nordeste | splices at 1.000; 19 of 52 quarters missing (all 2018-2020, H2 2021-24) | complete monthly; undocumented basis; volatile; format break 2022; 2025 ratio 0.17-1.51 |
| Banpara | 2013-2020Q2 accrual, three lines | 2019 onward cash, four definition changes; legal notices separable only in 2019 (15.4%) |

---

## 3. Decisions (all by the user)

**Decisions of 2026-09-14.**

- Sources: all of them - filings (CVM), Caixa's and the other state banks' own files.
- SEC 20-F figures form a separate annual panel, not interpolated into the quarterly one.
- Advertising is intended for estimation, using variation within each bank only (bank fixed
  effects); levels are not harmonised across banks.
- Sponsorship is a separate robustness series, not part of the main measure.
- Banco do Brasil: the filing. Banco do Nordeste: the filing, gaps left missing. Banpara: the
  filing to 2020Q2, then a gap until the COSIF series from 2025.
- Spending spikes are kept flagged and unaltered; their treatment is decided when the
  estimation specification is built.
- A freedom-of-information request was filed (section 4).

**Decisions of 2026-10-05, after the central bank's named file arrived.**

- The released 2013-2024 file and the public accounts from 2025 are ONE source: the same
  documents and the same three accounts, with a continuous join. Every bank then has a single
  series from 2013.
- The main measure is the SUM OF MEMBER INSTITUTIONS (each institution's own document 4010),
  which runs from 2013 for everyone. The conglomerate's consolidated document (40605, from
  2014-07) is the alternative and the fallback.
- The statement notes are a cross-check only. No further work on their parser, and the document
  listings for the twelve banks without one are dropped: the file has all twelve.
- Earlier decisions that still stand: sources are stored apart and never spliced; sponsorship is
  a separate series; BRB's "Negócios" category is sponsorship in every year (2026-10-04); whoever
  reads the panel pools the `amount_kind` labels (2026-10-01).

---

## 4. Freedom-of-information request

Filed on Fala.BR on 2026-09-14. It asks for the monthly balances, January 2013 to December 2024,
of accounts 8.1.7.45.00-9, 8.1.7.42.00-2 and 8.1.7.48.00-6 per institution, as submitted in
documents 4010, 4040 and 4060, with a fallback of the same balances without institution names,
one row per reporting institution (the form granted in NUP 18810.029130/2024-31).

- Answer due in 20 days (about 2026-10-04), extendable once by 10 (about 2026-10-14).
- Each appeal within 10 days of a refusal; the reply names the available instances.
- Precedents: bank-level level-4/5 accounts refused on commercial-secrecy grounds in
  NUP 18600.000888/2015-71 (the same 8.1.7 accounts, per bank, quarterly), 18600.001135/2019-15
  and later; unnamed per-institution monthly rows granted in 18810.029130/2024-31.
- If only unnamed rows arrive, they give system totals and cross-institution dispersion.
  Matching rows back to named banks would defeat the confidentiality the release rests on.

**Outcome, 2026-09-24: granted in the unnamed form.** One file
(`BCB/Awareness/raw/lai_bcb_cosif/RDR20261349700.xlsx`), 514,175 rows, January 2013 to December
2024, the three accounts requested. Each row carries a 10-digit identifier that is neither a tax
number nor a conglomerate code but is stable over time: 2,353 identifiers, 247 of them present in
all 144 months of document 4010. So the release is a pseudonymous panel, not only a set of totals.
Balances are negative and cumulate within each half-year, the convention
`scrape_cosif_advertising.py` already converts to quarterly flows. There is no 2025 month, so the
file does not overlap the named accounts.

An appeal filed the same day asked, in order, for named data; failing that, for attributes per
identifier (segment, control, institution type, conglomerate type) with small cells suppressed;
and for the meaning of the document codes.

**Appeal outcome: named data, received 2026-10-05.** The central bank's monitoring department
(Desig) replied that the first answer followed the level of detail then available, and sent the
complementary information requested. The file is
`BCB/Awareness/raw/lai_bcb_cosif/Contas 81742-45e48 demanda recursal.zip`, one CSV with the
columns `DATA;CNPJ;INSTITUICAO;DOC;CONTA;SALDO`.

- The same 514,175 rows as the unnamed file, now with each institution's CNPJ and name: 2,353
  institutions, January 2013 to December 2024, the three accounts. Names change over time for
  1,022 of them (Banco Original do Agronegócio became PicPay Bank, Banco Gerador became Agibank).
- Every kind of institution is in it: banks, payment institutions, finance and brokerage
  companies, consortium administrators, cooperatives.
- All twelve banks for which the statement-note route had no document listing are in it.
- The central bank itself names the institutions, so the file is used with names. The unnamed
  file is superseded and its identifiers are not linked to anything.

**The file agrees with audited figures where the two can be compared.**

| Bank | Compared with | Result |
|---|---|---|
| Banco do Brasil | its securities filing, the advertising account alone, 48 quarters 2013-2024 | median ratio 1.000; all 48 within 3% |
| Santander | its statement note, individual statement, all three accounts, 10 years 2015-2024 | 0.972 to 1.017 |
| Banco Volkswagen | its statement note, 6 years between 2018 and 2024 | 1.0000 every year |
| Bradesco, Banco do Nordeste, BMG, Itaú Holding | their securities filings, all three accounts | median 1.000 to 1.001 |

One rule carries over from the unnamed file: aggregate from document 40605 (the conglomerate's
consolidated block) or 4010 (the institution), never 40604 on top of 40605. Document 40604 lists
the individual entities consolidated into a conglomerate and repeats keys 4,894 times.

Section 5.10 describes the reader built from this file.

---

## 5. Built datasets

### 5.1 Central bank accounts, 2025 onward: `scrape_cosif_advertising.py`

**Inputs.** Under `paths.COSIF_RAW`, 202501-202603:

- Document 4010, per institution, keyed by CNPJ8 (COD_CONGL is blank), from FIVE public files:
  `{YYYYMM}BANCOS`, `SOCIEDADES`, `CONSORCIOS`, `COOPERATIVAS` and `LIQUIDACAO`. They are one
  universe split by kind of institution, and a conglomerate's members are spread across them: Nu
  Pagamentos, Mercado Pago, PagSeguro and Stone are in `SOCIEDADES`. Until 2026-10-05 only
  `BANCOS` was read, which left every non-bank member out of the 2025 member sums and broke the
  join with the 2013-2024 file (Nu's conglomerate fell from R$853m in 2024 to zero in 2025).
  `CONSORCIOS` also carries document 4110, the consortium groups, which is left out.
- Document 4060, per prudential conglomerate, keyed by COD_CONGL (the panel's
  `CodConglomeradoPrudencial` code space), from `{YYYYMM}BLOPRUDENCIAL`.

| File | Institutions a month | Reporting the advertising account |
|---|---|---|
| BANCOS | 170-176 | 96-116 |
| SOCIEDADES | 672-724 | 270-351 |
| CONSORCIOS | 121-125 | 94-106 |
| COOPERATIVAS | 754-779 | 523-583 |
| LIQUIDACAO | 1-6 | 0 |

No institution appears in two files in the same month. An institution can move between files:
Avenue Securities is in `SOCIEDADES` until November 2025 and in `BANCOS` from December.

**Rules.**
- SALDO is reais, stored negative; converted once to positive reais.
- Result accounts accumulate within the half-year and restart in January and July. Monthly flow
  = bal(m) - bal(m-1) (bal(m) in January and July); quarterly Q1 = bal(Mar),
  Q2 = bal(Jun) - bal(Mar), Q3 = bal(Sep), Q4 = bal(Dec) - bal(Sep). A missing balance gives a
  missing flow.
- Zero balances are not published. The level-4 children of 8.1.7 sum exactly to the level-3
  total for every filer (checked on every file), so an absent account for a filer is a true
  zero; for a non-filer every balance is missing.
- 202508 and 202509 BANCOS exist twice on disk (.csv.zip generated 2026-05-31 and .CSV generated
  2025-11/12). Their advertising rows are identical; the latest generation date is used.
- Running totals fall within a half-year in about 1.5% of month pairs (reclassifications):
  kept, with `is_reversal`.

**Validation (all pass).** No level-4 advertising account in the 72 files of 2024; every raw
balance negative; children sum to the total for every filer and month; July median below June;
advertising reporters per month within the bounds set for each file; anchors C0080099 ITAU,
C0080738 CAIXA, C0080329 BB; no duplicate entity-account-months and no institution in two files
in a month; reversal share under 5% (0.4-0.8% of month pairs); monthly flows reproduce quarter
flows for all 9,577 complete entity-quarters.

**Spot checks.** Itau prudential 2025 quarters 298.1 / 346.6 / 299.3 / 358.1 R$m and Banco do
Brasil institution 2025 total 528.2 R$m, both matching independent figures exactly. 144 of the
179 conglomerates with an advertising account are panel conglomerate codes.

**Outputs.**
- `cosif_advertising_monthly.{parquet,csv}` (30,315 rows; 1,842 institutions and 179
  conglomerates): `level`, `entity_key`, `entity_name`,
  `cnpj_leader`, `taxonomia`, `data_base`, `year`, `month`, `half`, `quarter`, `filed`,
  `{adv,promo,publ,admin}_bal`, `{adv,promo,publ,admin}_flow`, `is_reversal`,
  `adv_flow_spike`, `panel_key`.
- `cosif_advertising_quarterly.{parquet,csv}` (10,105 rows): flows `adv`, `promo`, `publ`,
  `admin`, `all3`, `adv_share_admin`, `all3_share_admin`, `n_months_filed`, `any_reversal`,
  `any_adv_spike`, `panel_key`.

**Totals over complete entity-quarters (R$ million).**

| Level | Quarter | Entities | Advertising | All three |
|---|---|---|---|---|
| Conglomerate | 2025Q1 | 151 | 2,223.7 | 3,098.1 |
| Conglomerate | 2025Q2 | 159 | 2,455.5 | 3,323.2 |
| Conglomerate | 2025Q3 | 150 | 2,543.4 | 3,432.7 |
| Conglomerate | 2025Q4 | 162 | 3,249.8 | 4,552.5 |
| Conglomerate | 2026Q1 | 160 | 4,161.1 | 5,129.2 |
| Institution | 2025Q1 | 1,745 | 2,534.6 | 3,916.9 |
| Institution | 2025Q2 | 1,750 | 2,818.4 | 4,245.4 |
| Institution | 2025Q3 | 1,775 | 2,907.3 | 4,422.9 |
| Institution | 2025Q4 | 1,779 | 3,645.0 | 5,683.6 |
| Institution | 2026Q1 | 1,746 | 4,499.2 | 6,046.6 |

The institution rows now include every kind of institution, cooperatives among them, most of
which belong to no panel conglomerate. For scale, the 2013-2024 file's institutions total
R$16,370m in the three accounts in 2024, about R$4,090m a quarter.

**Spikes (`adv_flow_spike`, kept unaltered).** A month more than 5x the entity's median positive
monthly flow and more than R$10m above it. 19 entity-months with all five files read; the list
below is the 15 found when only `BANCOS` was read. The dominant one: NU PAGAMENTOS
prudential, 202603, R$1,545.3m in one month (administrative expenses also +R$3,170m that month),
against about R$60m a month before; it drives most of the 2026Q1 conglomerate jump. Others:
Honda 202603, Sicoob 202510 (both levels), BNDES 202510 and 202512, Celcoin 202512, Sicoob
institution 202501, Banco da Amazonia 202512 (R$29.9m), PicPay Bank 202503 and 202603,
BancoSeguro 202506, RCI Brasil 202508, 202511 and 202512.

### 5.2 Securities-regulator filings, 2013 onward: `scrape_cvm_advertising.py`

**Inputs.** CVM DFP and ITR yearly zips 2013-2026, members DVA and DRE, individual and
consolidated, from the cache shared with `scrape_cvm_fee_income.py`
(`FirmDisclosures/CVM/raw`, downloaded 2026-06-01; `--refresh` redownloads). IF.data lists
(`paths.IF_DATA_LIST`, 52 quarters) mark BCB-supervised filers.

**Rules.**
- Every listed company's lines whose description names propaganda, publicidade, promocoes,
  publicacoes, relacoes publicas or marketing (and not "receita"). No bank list, no version
  choice: first-reported and restated series are both produced.
- Highest VERSAO per company-document-reference date. Values scaled by ESCALA_MOEDA; stored as
  absolute reais with the filed sign kept.
- Year-to-date values (period starting 1 January) are differenced within the calendar year, Q4
  from the DFP annual figure. ITR DRE quarter-only figures are kept only to cross-check.
- 2025 filings carry their 2024 comparatives as zeros (the 2025 accounting change): a zero
  comparative against a non-zero first-reported value is treated as missing (81 cases).
- **Collisions** are resolved by accounting structure: parent over its own sub-line; a zero
  sibling dropped when a non-zero sibling exists (36; for example Cielo's two "Vendas e
  Marketing" accounts 3.04.02.03/.04); identical siblings kept once (57); differing siblings
  summed and marked `summed_siblings` (238, all non-financial companies).
- **Series identity.** Labels drift for the same account (Itau's line reads "...e Publicidade" in
  2013Q3 and "...e Publicacoes" in every other quarter). A company-scope-statement that never
  files two advertising lines in one period is one series (`line_id` "single", 102 of 127);
  companies filing several side by side keep component-based ids (Banese and Banpara: propaganda,
  promotions and publications separately).

**Validation (all pass).** Every year's zip with all members and columns; Banco do Brasil's
individual DVA series present with an annual value 2013-2024; quarter-only figures agree with
year-to-date differences for 98.6% of BCB-supervised cases (n=139; 87.9% for other listed
companies, not gated); Banco do Brasil 2025 individual DVA / COSIF institution advertising =
1.0004.

**Flags (not corrected).**
- `scale_suspect`: 10, one supervised: Itau consolidated DVA 2013Q3 first-reported 735,000 reais
  against restated 735,000,000 (the whole 3Q13 DVA was keyed in thousands of thousands).
- `qo_mismatch`: 2 supervised, for example Banese 2022Q1 publications filed as zero while the
  Q2 filing implies about R$149k.
- `ytd_falls`: 18 supervised company-line-quarters (a year-to-date value below the previous one).
- `sign_flip`: 0 supervised.
- Known restatements visible as first versus restated: Bradesco consolidated 2022 (1,870.4 to
  1,704.6 R$m), Banco do Brasil 2019Q1 (+16%).

**Outputs.**
- `cvm_advertising_lines.{parquet,csv}` (9,753 rows): one row per filed value after version and
  collision resolution.
- `cvm_advertising_quarterly.{parquet,csv}` (5,524 rows): `CD_CVM`, `DENOM_CIA`, `cnpj8`,
  `bcb_supervised`, `scope`, `statement`, `line_id`, `n_label_variants`, `year`, `quarter`,
  `ytd_first`, `ytd_restated`, `flow_first`, `flow_restated`, `quarter_only_first`, and the flags.

**Supervised series found.** BMG (ind, con), ABC Brasil (con DRE; ind DVA, promotions only),
Bradesco (ind, con), Banco do Brasil (ind), Banese (three DRE lines), Banpara (three DRE lines),
Mercantil do Brasil (ind, con) and Mercantil de Investimentos, Banco do Nordeste (ind), Pan (ind,
con) and Pan Financeira, Itau Unibanco Holding (ind, con), Cielo (DRE "Vendas e Marketing"), and
the leasing and finance arms of Bradesco, BV, Dibens and Mercantil.

### 5.3 SEC annual reports, separate annual panel: `scrape_sec_advertising.py`

**Inputs.** For the eight filers (Itau 1132597, Bradesco 1160330, Santander Brasil 1471055, Nu
1691493, PagSeguro 1712807, StoneCo 1745431, XP 1787425, Inter & Co 1864163): the EDGAR
submissions index including paged older files, every 20-F and 20-F/A, and each filing's XBRL
instance (`*_htm.xml` for inline filings, the plain instance otherwise). Cached under
`paths.AWARENESS_RAW/sec/<cik>/`; `--refresh` refetches. Requests use the repository's
disclosure helper and its SEC contact header.

**Rules.**
- Every fact whose element name contains Advertis, Marketing, Branding, Publicity, Propaganda or
  Promotion, in any taxonomy, with period, unit, decimals and dimensions. Durations only.
- Each fiscal year appears in several 20-Fs as a comparative. The annual panel keeps
  `value_first` (earliest filing) and `value_latest` (most recent), both unaltered, with
  `restated`, `scale_suspect` (factor near 1000) and `sign_flip`.
- No currency conversion (Nu reports USD for the whole group).
- The element choice is left downstream: some matches are not advertising expense
  (Santander's `bsbr:MarketingOfNonbankingFinancialProducts*` are revenue from selling insurance,
  capitalisation and funds; Bradesco's `bbd:OtherOperatingIncomeexpensesCardMarketingExpenses` /
  `MarketingExpenses` are card-programme marketing).

**Validation (all pass).** Every filer has a 20-F with an XBRL instance and a matching element
in its latest 20-F. The nine fiscal-2025 anchors are reproduced exactly:

| Filer | Element | FY2025 |
|---|---|---|
| Santander Brasil | ifrs-full:SalesAndMarketingExpense | 482,880,000 BRL |
| Itau | ifrs-full:SalesAndMarketingExpense, on AttributionOfExpensesByNatureToTheirFunctionAxis = SellingGeneralAndAdministrativeExpenseMember | 1,740,000,000 BRL |
| Bradesco | bbd:OtherAdministrativeExpensesAdvertisingPromotionsAndPublicRelations | 1,287,243,000 BRL |
| Nu | nu:BrandingAndAdvertising | 271,108,000 USD |
| Nu | ifrs-full:SalesAndMarketingExpense | 302,822,000 USD |
| Inter & Co | ifrs-full:AdvertisingExpense | 284,998,000 BRL |
| XP | ifrs-full:AdvertisingExpense | 294,469,000 BRL |
| PagSeguro | ifrs-full:AdvertisingExpense | 868,803,000 BRL |
| StoneCo | ifrs-full:SalesAndMarketingExpense | 1,030,925,000 BRL |

Itau's figure carries a dimension, so any use must not filter to undimensioned facts.

**Coverage of the advertising elements (fiscal years).**

| Filer | Element | Years |
|---|---|---|
| Bradesco | bbd:OtherAdministrativeExpensesAdvertisingPromotionsAndPublicRelations | 2015-2025 |
| Santander Brasil | ifrs-full:SalesAndMarketingExpense | 2015-2025 |
| Itau | ifrs-full:SalesAndMarketingExpense (dimensioned) | 2015-2025; us-gaap:AdvertisingExpense 2008-2010 |
| Nu | ifrs-full:SalesAndMarketingExpense | 2019-2025 |
| Nu | nu:BrandingAndAdvertising | 2022-2025 |
| PagSeguro | ifrs-full:AdvertisingExpense | 2015-2025 |
| StoneCo | ifrs-full:SalesAndMarketingExpense | 2016-2025 |
| XP | ifrs-full:AdvertisingExpense | 2017-2025 |
| Inter & Co | ifrs-full:AdvertisingExpense | 2019-2025 |

The 20-Fs before each filer's XBRL era (Bradesco before 2018, Santander before its 2018 amendment,
PagSeguro's and StoneCo's and XP's first 20-F) carry no instance; their years come from
comparatives in later filings.

**Flags.**
- `scale_suspect`: Santander `SalesAndMarketingExpense` FY2022 (first 540,593,000; latest
  540,593, the FY2024 20-F re-reported 2022 in the wrong scale) and FY2024 (first 516,448; latest
  516,448,000). So neither "first" nor "latest" is uniformly right.
- `restated`: Bradesco FY2022 (1,870.4 to 1,704.6 R$m, as in its CVM filing); StoneCo FY2023
  (772.9 to 668.0) and FY2024 (993.0 to 874.9); Santander's non-banking marketing revenue lines.
- `sign_flip`: Bradesco FY2017, 2019, 2020, 2021; Nu FY2020, 2021; PagSeguro FY2016, 2017; Inter
  FY2021, 2022. Use absolute values.

**Outputs.**
- `sec_advertising_facts.{parquet,csv}` (338 rows): one row per fact per filing: `cik`,
  `entity_name`, `form`, `accession`, `filing_date`, `report_date`, `instance_file`, `element`,
  `period_start`, `period_end`, `dimensions`, `unit`, `decimals`, `value`, `value_abs`,
  `sign_raw`, `period_days`, `is_annual`, `fiscal_year`.
- `sec_advertising_annual.{parquet,csv}` (159 rows): filer x element x dimensions x period with
  `value_first`, `value_latest`, their absolute values, accessions and filing dates, `n_filings`
  and the flags.

### 5.4 Sponsorship robustness series, Banco do Brasil and Banco do Nordeste: `scrape_sponsorship_bb_bnb.py`

Kept separate from advertising by decision (section 3).

**Sources.**
- Banco do Brasil: "Patrocinios - Valores Pagos", 11 filed periods (2020 and 2021 as full
  years, 2022 to 2026 H1 as half-years), each in PDF, XLSX and ODS (33 files); monthly amounts
  by project, cash paid. All 78 months 202001-202606 parsed.
- Banco do Nordeste: "Projetos Patrocinados", content set 2715560 on bnb.gov.br (read from the
  page at run time), 16 ODS files 2012-2026 (2019 split into semesters); project-level values.
  ODS files are read as zip archives (content.xml via lxml), since no ODS library is installed.
- Raw files cached under `paths.AWARENESS_RAW/bb_sponsorship` and `/bnb_sponsorship`.
- Panel keys, confirmed in the market panel for 2013-2025: Banco do Brasil `C0080329`, Banco do
  Nordeste `C0081593`.

**Validation (all pass).** Every expected period has a status (none missing or unreadable).
Stated totals match line sums in 16 of 17 periods within R$1; the 17th (BNB 2019 H2 approved,
stated 5,140,300 against lines 5,240,300) is a stale saved formula result: recomputing its own
`SUM(E14:E212)` over the published cells gives 5,240,300. Banco do Brasil's XLSX and ODS agree
month by month in all 11 periods, and every PDF amount appears in the spreadsheet for its month.
No negative or zero amounts. The agent ran only under the project interpreter and imported only
`utils.paths` and `utils.disclosure_common`.

**Totals (R$ million).**

| Year | BB paid | BNB contracted | BNB disbursed |
|---|---|---|---|
| 2012 | | 11.94 | |
| 2013 | | 6.75 | |
| 2014 | | 7.21 | |
| 2015 | | 8.17 | |
| 2016 | | 3.61 | |
| 2017 | | 6.71 | |
| 2018 | | | 7.61 |
| 2019 | | 8.23 | 4.84 |
| 2020 | 88.99 | 4.14 | 3.38 |
| 2021 | 87.86 | 4.02 | 3.18 |
| 2022 | 135.56 | 5.61 | 5.34 |
| 2023 | 155.45 | 10.21 | 9.30 |
| 2024 | 237.02 | 33.85 | 30.54 |
| 2025 | 268.74 | 57.79 | 49.61 |
| 2026 | 131.49 (H1) | 42.73 (partial) | 37.31 (partial) |

Banco do Brasil sponsorship relative to its CVM advertising line (individual DVA): 0.206 (2020),
0.180, 0.296, 0.327, 0.449, 0.509 (2025).

**Anomalies.**
- `PatrociniosJanDez2021.xlsx` is a byte copy of `PatrociniosJanJun2024.xlsx` (same MD5, content
  202401-202406). Flagged and excluded from 2021; the 2021 ODS, confirmed by the PDF, is used.
  Periods are always taken from content, never from file names.
- Five Banco do Brasil PDFs stop before the end of their last month (4, 6, 11, 16 and 23 payments
  missing in 2022 H1, 2022 H2, 2023 H1, 2023 H2, 2024 H1); the spreadsheets are complete. Recorded
  in `pdf_missing_lines`.
- BNB layouts: 2018 has a disbursed column only; approved and disbursed appear together from 2019
  H2; the 2025 file's second column is "VALOR LIQUIDADO", mapped to disbursed with the published
  header kept in `amount_label`. BNB 2015-2017 files are titled "desembolsados" but their amount
  column is "VALOR DO PATROCINIO"; tagged contracted.

**Open questions for the user.**
1. Should BNB 2015-2017 amounts count as disbursed, given the file titles?
2. Can "liquidado" (2025) stand in for "desembolsado"?
3. BNB files repeat older contracts in later years, so contracted amounts double count across
   years. Should the BNB series use disbursed amounts only, from 2019 H2?
4. Can paid, disbursed and contracted amounts ever be pooled or spliced?
5. For BNB 2019 H2 approved: line sum (5,240,300) or the stale published total (5,140,300)?
6. Keep partial years (BB 2026 H1, BNB 2026 to about August)?

**Outputs.**
- `sponsorship_lines.{parquet,csv}` (9,859 rows: BB 2,541, BNB 7,318): `bank_key`, `cnpj8`,
  `panel_key`, `period`, `period_part`, `project`, `beneficiary`, `beneficiary_cnpj`,
  `amount_brl`, `amount_kind` (paid / contracted / disbursed), `amount_label`, source columns.
- `sponsorship_periods.{parquet,csv}` (102 rows): sums by `amount_kind`, `stated_total_brl`,
  `stated_formula_recomputed_brl`, `total_matches`, `n_lines`, `n_total_rows_excluded`, `status`,
  `filed_period`, `content_period`, `period_conflict`, `formats_agree`, `pdf_missing_lines`.

### 5.5 Entity crosswalk to the market panel: `build_advertising_crosswalk.py`

Maps every source's institution identifier to the market panel's prudential conglomerate code,
without choosing anything that is not mechanical. Reads the IF.data lists and
`market_panel.parquet` only; imports no pipeline module and writes nothing outside
`paths.AWARENESS_PROC`.

**Rule.** For a CNPJ8, the candidate codes are every `CodConglomeradoPrudencial` on an IF.data
row whose `CodInst` or `CnpjInstituicaoLider` is that CNPJ8, with first and last quarter. Only
fields that hold digits are compared: Banco do Brasil's CNPJ8 is 00000000, and zero-filling a
blank leader field had matched it to unrelated conglomerates in the first run.

| Status | Meaning |
|---|---|
| mapped | exactly one candidate code is in the market panel |
| time_varying | several panel codes in non-overlapping periods (an ownership change); use each quarter's code |
| ambiguous | several panel codes with overlapping periods; nothing chosen |
| not_in_panel | no candidate code is in the market panel |

**Hand-entered identifiers**, each verified against its own IF.data name:

| Source | Identifier | CNPJ8 | Panel code |
|---|---|---|---|
| SEC | Itau Unibanco Holding 1132597 | 60872504 | C0080099 |
| SEC | Bradesco 1160330 | 60746948 | C0080075 |
| SEC | Santander Brasil 1471055 | 90400888 | C0080185 |
| SEC | Nu Holdings 1691493 (via Nu Pagamentos) | 18236120 | C0084693 |
| SEC | PagSeguro 1712807 (via PagSeguro Internet IP) | 08561701 | C0084813 |
| SEC | StoneCo 1745431 (via Stone IP) | 16501555 | C0084686 |
| SEC | XP Inc 1787425 (via XP Investimentos CCTVM) | 02332886 | C0082475 |
| SEC | Inter & Co 1864163 | 00416968 | C0080996 |
| State bank | Banco do Brasil | 00000000 | C0080329 |
| State bank | Caixa | 00360305 | C0080738 |
| State bank | Banco do Nordeste | 07237373 | C0081593 |
| State bank | Banco da Amazonia | 04902979 | C0081249 |
| State bank | BRB | 00000208 | C0080288 |
| State bank | Banestes | 28127603 | C0080147 |
| State bank | Banrisul | 92702067 | C0080154 |
| State bank | Banpara | 04913711 | C0081586 |
| State bank | Banese | 13009717 | C0081634 |

**Result.** 391 entities, 247 dated code periods.

| Source | mapped | time_varying | ambiguous | not_in_panel |
|---|---|---|---|---|
| COSIF conglomerate | 147 | | | 32 |
| COSIF institution | 126 | 22 | 1 | 28 |
| CVM supervised filers | 16 | 2 | | |
| SEC | 8 | | | |
| State banks | 9 | | | |

The CVM filers map as expected: Bradesco and its leasing arm C0080075, Banco do Brasil C0080329,
Itau holding and Dibens C0080099, Mercantil (bank, investment bank, finance company, leasing)
C0080123, BNB C0081593, Banese C0081634, Banpara C0081586, ABC C0080312, BMG C0080178, BV
leasing C0080484, Cielo C0084710. Pan and Pan Financeira are time_varying: C0080257 to 2021Q1,
C0080336 from 2021Q2. The one ambiguous row is Banco Digio (institution level), which carries
C0080075 throughout and C0084655 over 2017-2022.

**Outputs.** `advertising_entity_crosswalk.{parquet,csv}` and
`advertising_entity_crosswalk_periods.{parquet,csv}`.

### 5.6 Caixa and the state banks: `scrape_caixa_advertising.py`, `scrape_caixa_scan_advertising.py`, `scrape_statebank_advertising.py`

**Caixa.** Monthly outlays from its transparency list (SharePoint list API; plain clients loop on a
302 without a cookie jar), January 2013 to June 2026: 162 months. Two builders read them, and the
assembly merges their outputs. The two never cover the same month, and the assembly stops if they
do.

*Text months.* `scrape_caixa_advertising.py` reads the 142 months that carry a text layer and
checks each against the totals printed in its own file, in integer centavos:

| Result | Months | Written as |
|---|---|---|
| Equal to the printed totals | 132 | `validated_exact` |
| Within 5 centavos (October 2018, 3 centavos) | 1 | `validated_within_5_centavos`, `tolerance_flagged` |
| The file's own columns do not add up to its own totals | 9 | `not_validated_document`, `validated = False` |

The nine are July and August 2015, March, June and July 2016, April 2017, July 2023, and May and
July 2024. Each was confirmed by an independent reading of the page, and each is kept as printed.
They are a reviewed list held in the script with their exact gaps, and that list is what makes the
parser-health gate meaningful: a month that fails and is not on the list counts against the
parser, and the run stops when fewer than 95% of months pass. A gate without the list passed a
parser that read totals as data and produced 45 such months.

One reading rule is the user's (2026-09-28): a cell printed without decimals, such as `14.250`,
counts as money only when the printed totals include it.

*Scanned months.* `scrape_caixa_scan_advertising.py` reads the 20 months published as image scans
with no text layer: all of 2013, January to June 2014, and April and May 2016. It uses the local
OCR engine benchmarked in 9.1, rebuilds the table from the OCR boxes, and accepts a month on the
table's own arithmetic. Nothing is sent to an external service.

| Result | Months |
|---|---|
| Closes exactly | 16 |
| Within 5 centavos (May 2014, 4 centavos on the grand total), admitted and flagged | 1 |
| Does not close: April 2013 (R$2.00), December 2013, June 2014 | 3 |

For May 2014 the arithmetic cannot say whether the print or the reading is off. The three that do
not close are written with `validated = False`.

*The 5-centavo rule* is the user's and applies to both builders: an exact match is
`validated_exact`; a gap of at most 5 centavos is `validated_within_5_centavos` and carries
`tolerance_flagged`; anything larger is not validated.

Sponsorship contracts 2019-2026 are a separate file, contracted at signing. Caixa's 2025 sums run
1.36 times the COSIF advertising account for the year and 0.66 of all three accounts, with a
same-month correlation of 0.88.

**Four state banks.** 422 listed documents, parsed per bank with published-total checks:

| Bank | Parsed | Unreadable | Duplicate content | Other statuses | Published totals matching | Months with lines |
|---|---|---|---|---|---|---|
| Banestes | 178 | 0 | 2 | 1 missing file | 100.0% of 327 | 2016-2026, 12 a year |
| Banrisul | 52 | 0 | 7 | 1 unavailable link, 1 names only, 1 no lines | 100.0% of 161 | 2021H2-2024, 2025-2026 |
| BASA | 11 | 0 | 4 | 20 no lines, 1 missing file | 100.0% of 156 | 2012, 2014, 2016-2026 |
| BRB | 83 | 38 | 4 | 6 duplicate pages, 12 missing file | 94.2% of 499 | 2014-2025, partial |

BRB's 94.2% is 470 of 499 checks, leaving out 18 checks on filings that a later republication
supersedes; counting those it is 470 of 517, or 90.9%. The first build stood at 96.2% of 422
checks; the later rounds described below added tables, and with them checks.

Fourteen defects were found and fixed while validating the first build. Most were caught by the
published totals rather than by inspection, and they compound: every fix exposed the next one,
which is why BRB moved 86.3% -> 91.5% -> 95.7% -> 96.2% rather than in one step. The last two were
caught by a separate audit, because no total can find them.
- Documents linked once per entity arrive as the same file under several names. A repeated content
  hash is now recorded and parsed once; this alone lifted Banestes and Banrisul to 100%.
- A published total row closes its table, so the next table starts a new one even where its header
  is unreadable. Without this, one quarter's advertising and sponsorship tables merged and
  collected two sets of month totals.
- Continuation tables are aligned from the right, where the amount columns sit. Skipping them for
  a column-count mismatch had silently dropped rows.
- BRB's gazette writes an amount in three forms in one column: `2.094.000,00`, `4.782.000` with the
  cents omitted, and `12.266.760,3` with the cents truncated to one digit. The shared reader took
  only the first, so the other two were read as empty cells. In 2025Q3 alone this dropped
  R$6,518,000 of sponsorship, and the September published total was itself unreadable, so no check
  existed for that month. The two extra forms are now read for BRB only, leaving the three banks at
  100% untouched. A dot is a thousands separator in every Brazilian amount, never a decimal point,
  so a dotted integer is unambiguous.
- Rows can lie outside every table the finder returns, because the ruling lines stop short of the
  rows they belong to. The cut falls at either end -- one 2025Q3 row sat ABOVE its table's box --
  so no geometric rule finds them. They are recovered from the page's words and admitted only when
  the row's own TOTAL column equals the sum of its month cells. That test matters: the same gazette
  pages carry contract prose ("Valor Total: R$ 1.443.398,16", "Nota de Empenho") whose figures fall
  inside the month columns by accident, and four such lines were being read as sponsorship in the
  2016 files, which publish no totals to catch it. Refusing a genuine row is the safe direction:
  the published month totals then show a shortfall.
- A table whose month header sits below blank spacer rows was treated as a headerless continuation
  and, having no state to continue from, dropped whole. This lost BRB's entire 2021Q4 advertising
  table (R$2.16m, R$2.64m and R$4.76m in October, November and December). Blank rows no longer
  count against the header search window, so a body row still cannot pass as a header.
- In those wide layouts the gazette merges label columns unevenly, so a row's amounts can land one
  cell right of the header's month columns (2021Q4 puts the months at columns 5/7/9 while the legal
  notice rows and the total row carry them at 6/8/10). The shift is applied only when the shifted
  amounts sum to the row's own total cell, which is what separates a drifted row from a row that
  genuinely has empty months.
- One logical row can be printed as TWO lines, the month cell on one and the row total on the other
  (a 2021Q3 Bosque Formosa row has July below and its total on the line above). Consecutive outside
  lines within 22pt are therefore tried merged before being tried alone, and a merge that would put
  two amounts in one column is refused as two separate rows. This is what recovered most of the
  remaining shortfall, taking 91.5% to 95.7%.
- Page furniture has to be dropped BEFORE lines are merged, not after. A 2016 page header sits within
  the merge distance of a real row, and merging the two gave that row's amounts (17,100.00 a month,
  51,300.00 in total, so the arithmetic test passed) the header's text as its beneficiary, which also
  moved it into the sponsorship series.
- A negative amount is printed with its minus sign as a separate word ("-R$" then "1.493,07"), so a
  recovered row read a 2020Q4 reversal as positive and pushed December over its published total by
  exactly twice the amount. Sign loss is invisible to the reconciliation test, because a row's total
  loses the sign along with its months. The sign is now taken only from an ADJACENT token: a lone dash
  far to the left is the previous column's published zero, and the distances are unambiguous - 2pt for
  a real minus against 240pt for an empty-cell dash. Negating on the dash turned real rows negative
  and had them refused, which cost 1.4 points of match rate until it was gated.
- A row's OWN published classification now outranks the table-level guess about whether a table is
  advertising or sponsorship, and that guess recognises BRB's abbreviation "PROP E PUBL" for
  Propaganda e Publicidade. Before both changes, 158 production and advertising rows of 2016Q2 were
  published as sponsorship: the abbreviation defeated the table test, and the table's verdict then
  overrode each row's own classification.
- A table published as a total row with no detail rows now carries the quarter scope as well as the
  month scope, or its quarter total has nothing to sum and reads as a failed check although every
  month is present.
- Two classification defects that the published totals CANNOT catch, because the amounts reconcile
  whichever category they are filed under. This is why a separate audit compares every row's group
  against the words in its own label, for all four banks. First: in the layouts with four label
  columns the classification cell comes out empty and its text sits in the purpose column instead, so
  24 rows worth R$16.3m that say PROPAGANDA E PUBLICIDADE/VEICULACAO and /PRODUCAO were filed as
  "other"; the markers are now read from either column. Second: the documents mix sponsorship rows
  into the advertising table, so the table-level flag is off for them, and 660 rows worth R$39.0m
  whose published classification says Esporte, Arte e Cultura, Entretenimento, Relacionamento
  Institucional or Causas Sociais also fell to "other". Those categories are now recognised by name.
  BRB's "other" is down from 660 rows and R$39.0m to 13 rows and R$0.23m. After the fix the audit
  finds no row in any of the four banks whose group contradicts its own label.

The classification audit is clean for the other three banks. Banestes has no unclassified row;
Banrisul's 104 are all zero-valued; BASA's 98, worth R$9.3m gross, are agency fees
("Honorarios/Desconto de Agencia") and creative work ("Criacao"), which belong in neither placement
nor production and are deliberately left in `other`. BASA's gross equals net plus tax on all 1,172
fully published lines. BRB clears its 90% floor at 94.2% and its rows are written with
`validated = True`. Twenty-nine of its 499 published totals fail. They have not been classified one
by one since the count stood at 16 (2026-09-20). At that count most were defects in the SOURCE
rather than in the parse: `brb_row_total_gap` reported eight rows whose month cells contradict
their own printed TOTAL. The clearest is 2018Q1, published in three entity files: a PPR row prints
32.000,00 for January, while both its own row total and the published January total imply
537.258,64. The document's two totals agree with each other to the cent - the row totals sum to the
stated quarter, and the month totals sum to the stated quarter - so it is the single printed cell
that is wrong. That cell is now imputed from the row total (+505,258.64), marked `imputed`, and
counted as observed (user decision, 2026-09-28). A cell is imputed only when the published month
total falls short by exactly the row's own gap; every other such row is reported and kept as
printed, because filling it would silently rewrite published data. Nine of the sixteen failures
were of that class; the rest were a table whose TOTAL column is year-to-date, a 2014 file whose
digits arrive as `(CID:55)` glyph codes, and two files short by round amounts with no visible
cause. BRB is 0.8% of panel deposits. Its own advertising plus sponsorship equals 0.94 to 1.06 of
its COSIF advertising account, confirming it books sponsorship inside that account.

**BRB after the first build (2026-09-24 to 2026-10-04).** Reviews of the BRB series found and closed
the items below. Banestes, BASA and Banrisul stayed row-for-row identical through all of them.

- *Duplicate copies.* The same table is published in more than one entity's file. The copies are
  removed: exactly R$13,670,918.65 of double counting.
- *Cartão BRB* is read but kept out of the conglomerate's series: the registry places it outside
  prudential conglomerate C0080288.
- *Republications.* A later filing of a quarter supersedes the earlier one, and filings that carry
  no date are ordered by their position in the listing. 2018Q2 for the bank is 4,822,250.24, from
  the January 2019 republication, which prints the detail rows. The 4,815,932.74 that later notices
  print for that quarter sits in a year-to-date recap with no correction wording, and changes
  nothing.
- *Corrections printed as text.* A correction that names a row is applied to that row. A
  correction that restates only a total is used in the check, and the gap is flagged (2019Q1 and
  2019Q2).
- *2016Q1 recovered.* Bank 6,137,786.95, brokerage 117,292.40 and finance company 213,492.70, each
  equal to its printed "TOTAL PAGO". Three things had hidden the quarter: the header search was one
  row too short for that quarter's header; one gazette page packs three entities' tables into two
  table objects; and each record is printed over about three grid rows, so an amount row took the
  previous record's category. Each amount row now takes its own record's labels. This rule applies
  to 2016Q1 tables only, and a table that falls back from it writes a note. An independent reading
  of the PDF checked all 44 bank rows and every brokerage and finance-company row for category,
  beneficiary, purpose and month. At institution level the quarter is advertising 4,656,904.20,
  legal notices 992,280.85 and sponsorship 616,500.00.
- *Basis.* Each table takes its basis from its own total row: paid, accrued, or undocumented when
  the row prints only "REALIZADO". A quarter whose tables differ carries a combined kind such as
  `paid/undocumented`. Whatever reads the panel pools the kinds (user decision, 2026-10-01), so no
  quarter is dropped for its label.
- *"Negócios" is sponsorship in every year* (user decision, 2026-10-04). BRB's 2014 and 2016 notices
  print the category as "Patrocínios/Negócios", and the same fair was filed as sponsorship in 2016
  and 2025 but as "other" in 2018-2020. Six rows, R$539,000, moved from "other" to sponsorship.
- *Left as is.* Cartão BRB's 2016Q1 labels are misaligned (it is outside the series), and 38
  scanned gazette files are unread.
**Two years were recovered after the first pass, one of them from the live site.**

Banrisul's 2024 was never lost. Its detail page lists that year's supplier files and simply omits
the `<a>` tags for the twelve "Valores Pagos" files, which the CMS still serves under `/midias/` at
IDs interleaved with the listed ones. No archive was needed. The twelve URLs are recorded in
`BANRISUL_UNLINKED_2024` because no page lists them; the parser still reads each month from the
document's own "Mes/Ano Ref." line, so a wrong month in a file name cannot mislabel a period. Eleven
of the twelve parse: November's PDF carries a broken font and yields no text, so 2024Q4 stands as an
incomplete quarter of two months.

BASA's 2014 and 2012 survive only as Internet Archive captures, taken through the `id_` form of the
capture URL, which returns the stored bytes rather than a rewritten page. The 2014 file is a month
matrix - production plus media by month, with a published TOTAL/MES row - and all twelve of its
month totals reconcile to the cent against the generalised `parse_basa_month_matrix`, which now takes
its year from the document instead of having 2016 written into it. The 2012 file states one column
for the whole year, so its rows are annual and reach `statebank_advertising_lines` but neither panel;
its own published TOTAL exceeds its itemised rows by R$338.38, which the parser now reports rather
than letting the 0.1% check tolerance swallow it. A third capture,
`fornecedores_consolidado_2014.pdf`, is deliberately not ingested: its period appears nowhere in its
text and it carries no amounts.

What is confirmed gone: Banrisul's January to July 2025 (IDs 47944-47969 are dead server-side, so
even the supplier files the site still links return the same 44,406-byte "page not found" page, and
no capture of any advertising PDF or detail page exists); Banrisul before July 2021; BASA's 2015 and
2018, whose exact URLs were recovered from archived index pages but which have zero captures and are
dead live; and BASA's 2013 values, which were never published - only that year's supplier list
was.

### 5.7 Assembled panels: `build_advertising_panel.py`

Outputs keyed to market-panel conglomerate codes through the crosswalk. Rebuilt on 2026-10-05
with the central bank's accounts running from 2013.

**`advertising_panel_quarterly`** (83,192 rows; 10,680 before the central bank's file): one row
per conglomerate, quarter, SOURCE and measure. Sources are never spliced, by decision: a source
change inside a bank's history is a level jump that bank fixed effects cannot absorb.

| Source | Conglomerates | Span | Basis |
|---|---|---|---|
| cosif_institution (members summed) | 522 | 2013-2026 | accrued; THE MAIN MEASURE |
| cosif_conglomerate (consolidated document) | 210 | 2014-2026 | accrued; the alternative and the fallback |
| cvm | 12 | 2012-2026 | accrued, individual statements summed; Itau consolidated |
| caixa_own | 1 | 2013-2026 | outlays ("custos"); text months and scanned months merged |
| statebank_own | 4 | 2014-2026 | outlays; the basis printed on each table (paid, accrued, undocumented) |
| sponsorship | 1 | 2020-2026 | paid |

The two `cosif` sources are the central bank's accounts: the file it released for 2013-2024
(5.10) and the public files from 2025 (5.1). They are one source by decision (2026-10-05), the
same documents and accounts on both sides of 2025. The statement-note series (5.9) and the film
counts (5.8) are not sources of this table.

**Which conglomerate a member institution belongs to is decided quarter by quarter**
(`codes_by_quarter`), because banks change hands over 2013-2026:

- *Registry rule.* An institution is counted in the conglomerate whose registry span holds the
  quarter. Before the registry's first coded quarter (2014Q1) the code it gives then is used, and
  likewise after its last. Outside every span the institution belongs to no panel conglomerate in
  that quarter. This matters: Órama and Easynvest were bought by BTG and Nu in 2024 and 2021, and
  a single code per institution would have added their earlier spending (R$145m and R$135m) to
  BTG and Nu for every year before the purchase.
- *Own-entity rule.* Where an institution has several candidate codes and its name in that
  quarter matches one of their panel names better than any other, that code wins in every quarter
  in which the market panel holds DEPOSITS under it. The registry moved Banco Pan into BTG's
  conglomerate in 2021Q2 while the market panel keeps Banco Pan as its own entity with about
  R$20bn of deposits, and attaching Pan's advertising to BTG would contaminate BTG and empty Pan.
  Where the own code has no deposits in the quarter, the registry rule decides. Three cases show
  why the deposits matter:
  - PicPay's own code has no deposits in the market panel until 2022Q2 and R$8bn from 2022Q3,
    while Banco Original's falls from R$19.5bn to R$12bn in the same step. So PicPay's
    advertising sits under Original until 2022Q2 (R$659m over six quarters) and under PicPay
    from 2022Q3: it follows the code that holds its deposits.
  - Will Financeira's own code has no deposits from 2024Q3, when the registry places it in Banco
    Master, so its spending goes to Master from then.
  - Kirton Bank, the former HSBC, goes to Bradesco once the panel stops carrying its code.
- *Name matching.* Tokens of three letters or more count, common words (Brasil, Bank, Crédito,
  Investimento and the like) do not, and an abbreviation matches the word it begins, from four
  letters: the public files write "BCO C6 CONSIG" for the released file's "BANCO C6
  CONSIGNADO". Without these two refinements PSA Finance matched Santander on "Brasil" alone,
  and C6 Consignado changed conglomerate at the 2025 join.
- A conglomerate's consolidated document is filed by its leader, so in the released file it is
  mapped from the leader's CNPJ to the code that leader leads in that quarter.

808 of the 2,531 institutions that ever file belong to a panel conglomerate in at least one
quarter. The rest are outside the panel: cooperatives (Sicredi's and Sicoob's singles and
centrals), BNDES, consortium administrators and independent brokers. Their quarters are dropped
with a count: 13.7% of the three accounts' total over 2013-2026, rising from 8.5% in 2013 to 16%
in 2023-2024.

**The cooperatives stay out (user decision, 2026-10-05), to match the deposit panel.** The
registry lists the bank and its subsidiaries under Banco Sicredi's and Banco Sicoob's codes, and
no cooperative. The market panel does the same: its deposits for the two come from the banks' own
branch-level filings and hold what is booked at the bank, mostly savings accounts.

| | Banco Sicredi | Banco Sicoob |
|---|---|---|
| Panel deposits, 2024Q4 | R$39.4bn, of which savings R$32.7bn | R$21.9bn, of which savings R$15.4bn |
| Share of panel deposits | 0.83% | 0.46% |
| Advertising counted, 2024 (bank and subsidiaries) | R$304m | R$268m |
| The system's cooperatives, 2024 (not counted) | 134 cooperatives, R$1.02bn | 232 cooperatives, R$239m |

Members' deposits at the individual cooperatives are not in the panel, so adding the
cooperatives' advertising would set R$1.02bn of spending against deposits that exclude theirs.
One caveat stays: Banco Sicredi's own spending promotes a brand whose deposits are mostly outside
the panel, so its advertising per real of deposits is high next to other banks (about 0.8%
against about 0.1% for Banco do Brasil). Bank fixed effects absorb a constant gap of that kind.
The film counts (5.8) follow the same choice.

**Independent review of the assignment (2026-10-05).** A reviewer rebuilt the assignment from the
registry with its own code. It agrees in 98.1% of institution-quarters, and every difference is
one of the rules above. No institution-quarter is counted in two conglomerates, and no spending
is attributed to an acquirer before the registry places the institution there. Across the 4,332
conglomerate-quarters where both central-bank sources are observed, the median ratio of the
consolidated document to the member sum is 1.000 in every year. Its findings (Will Financeira,
C6 Consignado, PSA Finance) are the cases described above, fixed the same day.

Known and left as it is: an institution the registry stops listing while it still files leaves
its conglomerate from then on (22 institutions, R$117m; the largest are Avenue Securities since
2024Q4 and Caixa Consórcios since 2022Q3).

A member-sum row whose flow rests on a month taken as zero in the released file (5.10) carries
`member_zero_inferred` in `flag_notes`: 731 of the 12,468 advertising rows.

Measures are separate rows: `adv`, `adv_production` (the main own-file measure), `promo`, `publ`,
`all3`, `as_filed` (the filing's own line, with whatever accounts the bank puts in it),
`legal_notice`, `sponsorship`, plus `adv_production_plus_sponsorship`, a BRB-only extra measure
because BRB books sponsorship inside its advertising account.

`amount_kind` records the basis as published. For the state banks it can be a combined label such
as `paid/undocumented`, when the tables of one quarter print different bases. Readers of the panel
select by source, measure and `value_status` and never filter on `amount_kind` (user decision,
2026-10-01): a `paid` filter would drop real BRB quarters, 2019Q4, 2020Q3 and 2020Q4 among them.
Both diagnostics already read it this way.

A Caixa quarter is `validated` only if every month in it is. Eleven Caixa quarters are written with
`validated = False`: 2013Q2, 2013Q4, 2014Q2, 2015Q3, 2016Q1, 2016Q2, 2016Q3, 2017Q2, 2023Q3, 2024Q2
and 2024Q3. `flag_notes` says why: `month_totals_do_not_close`, `month_within_5_centavos`,
`month_read_by_ocr`.

Columns carry nominal amounts and `amount_brl_real`, deflated by the chained IPCA index from the
central bank's series 433 with the base pinned to 2024Q4, the window's end, so the real series does
not move when a new inflation month is published. Intensity is `adv_over_assets` on the
conglomerate's total assets (the headline: constant within a conglomerate-quarter and free of the
jumps that affect the deposit aggregate), `adv_over_deposits` alongside it with `deposits_suspect`
marking the 7,039 rows whose market-panel deposit denominator jumps by more than half against the
previous quarter, and `adv_over_admin` where the 2025 accounts give administrative expenses (the
released file has no administrative total, so it is empty before 2025).

`value_status` gives every cell one of five meanings, because a single NaN cannot carry them:
`observed_positive` (52,322), `observed_zero` (16,607), `observed_negative` (451, reversals and
glosas), `imputed_from_row_total` (3, the BRB January 2018 cell of 5.6, counted as observed) and
`not_observed_in_span` (13,809 rows generated inside a series' own span, so a gap inside a bank's
history is visible as a row rather than as an absence).

`segment_id` increments on any change of source, scope, bundle or key, so every join is legible
without recomputing it. `registry_code` stores the code the IF.data registry gives the filer in that
very quarter, beside the panel code, and `code_conflict` marks the 687 rows (38 conglomerates) where
they disagree (82,656 of 83,192 rows carry a registry code). When the panel held only 2025-2026
there were 151 such rows, and every one was a filer whose acquirer the registry had already
absorbed it into while the market panel still carried it as its own entity: Kirton into Bradesco,
Alfa into Safra, Modal into XP, Traton into Volkswagen, Master BI and Letsbank and Pleno into
Master, John Deere into Bradesco, Pan into BTG. The panel wins by design in those, since attaching
the filer's advertising to the acquirer's code would both contaminate the acquirer and empty the
filer. The 687 rows of the longer panel have not been listed one by one.

Remaining columns: `validated`, `n_months_observed`, `n_entities` and `flag_notes` (named so because
`flags` collides with the DataFrame attribute of that name).

**`advertising_panel_annual`** (113 rows): the SEC 20-F series, advertising-specific element where
a filer has one and the broad sales-and-marketing line otherwise, flagged; plus the sponsorship
that is filed by year rather than month (Banco do Nordeste's contract lists, Caixa's contracted
amounts), kept annual and never spread across quarters.

**`advertising_ancine_quarterly`** (5,995 rows): the film counts of 5.8, one row per conglomerate
and quarter. It is a separate table by decision (user, 2026-09-29): a count of films is not an
amount and does not belong among the measures above.

Six checks abort the build: inputs present and every source contributing; every row carrying a
panel code or being dropped with a counted reason; no duplicate keys; the deflator covering every
quarter; no conglomerate-quarter mixing a consolidated filing with an individual filing of an
entity inside it; and a printed 2025 reconciliation. That reconciliation gives a median ratio to
the COSIF conglomerate advertising account of 1.05 for the filings, 1.31 for Caixa and 0.74 for
the state banks (2026-10-05 build). The build also stops if a Caixa month is read by both
builders, if the quarterly film counts do not add up to the monthly table, or if two leaders
file a consolidated document for the same conglomerate in the same quarter.

Five defects of my own surfaced in the runs and are fixed: institution rows were dropped for want of
a conglomerate code, then double-counted once mapped, because several institutions sit inside one
conglomerate and their rows must be summed; sponsorship filed by year was silently excluded by a
month-format filter; the deflator based itself on a partly observed quarter; the x1000 repair in the
filings branch was inverted, keeping the slipped small value rather than the large one (a negative
half-billion for Itau in 2013Q3 was the symptom -- in a near-1000 ratio pair the SMALL member is the
keying slip); and the count of unvalidated rows printed negative, because `validated` arrives as
object-dtype Python booleans where `~True` is -2.

### 5.8 Registered advertising films: `scrape_ancine_ad_films.py`

The audiovisual regulator's open file of advertising-film registrations
(`crt-obras-publicitarias.csv`, 314 MB, last modified 2026-09-01) has one row per registered film,
with the advertiser's tax identifier, the product advertised and the registration date. It gives a
count of campaigns per bank and never a spend measure: the levy per film is a fixed statutory fee.
It covers banks that disclose nothing else.

9,813 registrations are attributed to a panel bank, each with the basis recorded:

| Class | Registrations | Rule |
|---|---|---|
| `own` | 7,088 | the advertiser is the bank or a member of its conglomerate |
| `affiliate` | 698 | a bank-owned insurer or affiliate, attached when the film carries the bank's brand or the bank's group controls the advertiser |
| `holding` | 66 | a parent holding (J&F for PicPay, C6 Holding, UOL for PagSeguro), only films whose product or title names the bank |
| `media_sponsored` | 81 | sponsored content registered by a media company, attributed to the bank that paid, flagged |
| `coop_system` | 1,880 | Sicoob and Sicredi singles and centrals, counted against the two cooperative banks and never in a bank's total |

The attribution rules were decided by the user on 2026-09-28:

- An affiliate's film that names another bank and not its own goes to the review list, not to
  either bank (Instituto Porto Seguro's "Cinema Itaú Teatro" films).
- Multi-bank joint ventures (Elo, Livelo, Alelo, Cielo) stay unattached, with their owners
  recorded.
- A brand word alone is not enough when the advertiser's name also carries COMUNICACAO,
  PUBLICIDADE, DROGARIA, IMOVEIS or OTICA: the agencies and shops so named advertise supermarkets
  and shoes. SAFRA also means harvest, and INTER, PAN, NEON and XP are generic words.
- A brand in a film's title may only name a tournament or a venue. 85 such registrations carry
  `naming_rights_suspect`, so they can be dropped.
- Outside the registry's span an advertiser takes its nearest code only when that span touches the
  first or the last registry list; otherwise the film has no code in that quarter.

`advertising_ancine_quarterly`, written by the assembly: 5,995 rows, 109 conglomerates, 2013-2026.
`n_crt` (6,997 in all) is own plus affiliate plus holding plus media-sponsored, each also in its
own column. `n_crt_coop_system` (1,818) is apart. `in_panel` marks the 3,960 rows whose
conglomerate is in the market panel that quarter, and `partial_quarter` the quarters the file does
not cover in full. Seven checks stop the scraper.

### 5.9 Statement notes: `scrape_bank_notes_advertising.py`

Every bank publishes audited statements whose administrative-expense note itemises advertising. It
is the same accounting concept as the 2025 COSIF accounts, and it exists for banks whose
structured filings carry no such line. Santander Brasil is the reason the route exists: 9.5% of
panel deposits, a filing every quarter, and no advertising line in any structured year.

The script reads the regulator's document system for CVM filers (`cvm_rad`) and each bank's own
site otherwise (`ir_site`). It contacts only hosts approved by the user: `utils/note_sources.py`
holds the 23 approved hosts and the eighteen targets, and `allowed_host()` is applied to every URL
and to every redirect target before it is followed. The targets are the conglomerates in the E3
spec-12 estimation sample with no quarterly advertising series, 25.9 points of 2024Q4 panel
deposits.

**The mapping, settled on Santander's ITR at 2025-06-30.** Note 25 states "Propaganda, Promoções e
Publicidade" of R$223,041 thousand for the first half on the INDIVIDUAL statement and R$299,207
thousand on the CONSOLIDATED one. Against the COSIF accounts for the same six months:

| note figure | COSIF comparison | ratio |
|---|---|---|
| individual, R$223.0m | institution 90400888, advertising account alone, R$146.2m | 1.526 |
| individual, R$223.0m | institution 90400888, **all three accounts**, R$222.76m | **1.0013** |
| consolidated, R$299.2m | **conglomerate** C0080185, all three accounts, R$300.1m | **0.997** |

So the bundled note label corresponds to COSIF `all3` (advertising plus promotions plus
publications) and not to the advertising account alone, which would overstate by about half. And
the note's individual column corresponds to the INSTITUTION while its consolidated column
corresponds to the PRUDENTIAL CONGLOMERATE.

**Outputs, written 2026-10-04.** 232 listed documents, 195 parsed; 10,660 lines; 1,200 period rows;
858 checks of a stated note total against the sum of its items.

| Bank | Route | Parsed | Years with an advertising line | Windows | Stated totals equal to their items | 2025 half-years against COSIF | `validated` |
|---|---|---|---|---|---|---|---|
| Santander | cvm_rad | 40 | 2015-2025 | quarter, half-year, nine months, year | 93.7% of 286 | 0.99-1.00 | True |
| BTG | cvm_rad | 36 | 2015-2025 | quarter, half-year, nine months, year | 99.4% of 158 | 1.14-1.20 | True |
| Daycoval | cvm_rad | 40 | 2015-2025 | same; one window a year in 2015-2018 | 99.7% of 370 | 1.00 individual, 1.06 consolidated | True |
| Safra | ir_site | 20 | 2015-2021 | half-year, year | 95.2% of 42 | no 2025 line | True |
| Volkswagen | ir_site | 37 | 2014-2026 | half-year, year | 0 of 2 | 1.00-1.05 on one set of documents, 0.37-0.71 on another | False |
| PagBank | ir_site | 22 | 2018, 2019 | year | prints no totals | no 2025 line | False |

`validated` is per bank: at least 70% of its stated note totals must equal the sum of the items
printed above them. A bank below the floor stops the run unless it is named in `--unvalidated`,
which writes its rows with `validated = False`. Volkswagen is written that way: its only two
stated totals come from one two-page 2012 document. PagBank's notes print no total, so nothing can
be checked. Two further checks pass: advertising never exceeds the note's own administrative
total (5.1% of it at the median, 782 columns), and 11 of 13 scope-matched 2025 half-years land
within [0.80, 1.25] of the COSIF sum their label maps to, median 1.0013.

**Reading rules the totals check forced.** Daycoval went from 33.1% to 99.7% through five rules,
none specific to a bank:

- A note may print several totals per column (personnel, taxes, other administrative). Each is
  checked against the items since the previous total.
- A negative printed with a detached bracket, "( 58.930)", is negative. Reading it as positive
  mixed the signs within a column, and 46 of Daycoval's advertising rows came out negative.
- A row whose label ends in a preposition is prose ("trimestre findo em 30 de junho") and is
  skipped, unless the label is an advertising one.
- A figure row with no label, followed by a text-only row, takes that row as its label.
- The filing system's running page title is skipped.

**Run behaviour.** One document that raises is recorded as `error` and the run goes on; two stop
it. A lost connection stops the run cleanly with exit code 75, and the same command resumes from
the cache. Nothing is written until every check has passed. `--offline` re-parses the cache
without contacting any host.

**Status since 2026-10-05: a cross-check, not a panel source.** The central bank's named file
(5.10) gives every bank the same accounts on one definition from 2013, so the note series are
not put in the panel, their parser gets no further work, and the eleven missing document
listings and the search for a PicPay source are dropped (user decision).

Five checks made on 2026-10-05, each against the named file or the 2025 public accounts:

| Bank | Finding |
|---|---|
| BTG | The individual statement tracks BTG's own filing: 1.000 in 2020 and 2021, 1.05 in 2022, 1.06 in 2023, 1.17 in 2024, and 1.13-1.18 in each 2025 quarter. Its note line has grown beyond the three accounts. The consolidated statement does not follow the prudential conglomerate (1.15 to 1.65 by quarter in 2025). |
| Daycoval | Every filing from 2016 to 2025 carries the advertising line; the one window a year in 2015-2018 is the parser failing to read the column headers of the 2016-2019 quarterly filings. But the note is 2 to 6 times what Daycoval reported to the central bank in 2015-2023, and equal to it in 2025: before 2025 the note line is a different object. |
| Volkswagen | Good once the entities are separated. Banco Volkswagen's note equals its filing to the thousand (1.0000 in six years), and the bank plus Consórcio Nacional Volkswagen equals the conglomerate in 2025 (0.99-1.00). The bad ratios of the table above compared the consortium administrator with the bank. The 2012 document is not a statement note. |
| Safra | The document LISTING stops at December 2021; it is not that later statements lack the line. Where it exists the note is 0.90 to 1.07 of the conglomerate's advertising account. |
| PagBank | "Marketing e publicidade" is parsed in almost every document from 2019 to 2025; only two reach the period table because scope and window are not read on the rest. |

Known limits left as they are: the 70% floor still stops a run when a bank's failures come from a
single document, so a run needs `--unvalidated volkswagen`; and the Volkswagen, Daycoval, PagBank
and Safra items above are not fixed.

### 5.10 Central bank accounts, 2013-2024: `build_cosif_lai_advertising.py`

**Input.** The named file of section 4: 514,175 rows, one per month, institution, document and
account, for the accounts 81745009 (advertising), 81742002 (promotions and public relations) and
81748006 (publications). These are the 8-digit codes in force until 2024 of the same three
accounts the public files carry from 2025. The administrative-expense total is not in the file,
so shares of it are empty before 2025.

**Two levels.**

- *Institution*: document 4010, the institution's own balancete, keyed by CNPJ8. 2,350
  institutions, from January 2013.
- *Conglomerate*: document 40605, the consolidated block of the prudential conglomerate's
  balancete, keyed by the CNPJ8 of the conglomerate's LEADER, from July 2014. 215 leaders. The
  file carries no prudential code; `cod_congl_2025` gives the code the leader has in the 2025
  public files (146 of the 157 leaders of 2024 have one), and the crosswalk maps leaders to codes
  quarter by quarter.

**Flows.** Balances are stored negative and accumulate within each half-year. Monthly and
quarterly flows come from the same functions as the 2025 script, so one method serves 2013-2026.
Among institutions with a balance in both June and July the median July/June ratio is 0.16-0.18
every year, which confirms the reset.

**What an absent row means.** The file holds only non-zero balances, and institutions file at
different frequencies: banks every month, many small institutions only at quarter ends or
half-year ends. An absent month is therefore either a zero or a month the institution does not
file. The rule:

- An institution-year is *monthly* when it has a balance in a month other than March, June,
  September or December.
- In a monthly year, a month with no row is taken as a true zero when it precedes the first
  balance of its half-year, or lies in a half-year with no balance, and falls between the
  institution's first and last month in the file. 16,621 months are filled this way, and they
  carry `zero_inferred`.
- Every other absent month stays missing, and a flow that needs it is missing. 1,699 months
  (0.87%) are left missing after a balance in the same half-year.
- **No zero is ever inferred at the conglomerate level.** A conglomerate's document is filed by
  its leader, and the leader can change: Banco Original led C0080903 until 2023Q2, PicPay from
  2023Q3 to 2024Q3, Banco Original again from 2024Q4. A first version filled zeros there too and
  wrote five zero quarters for XP's conglomerate in 2018-2019 while XP's own filing showed about
  R$55m a quarter.

Quarter flows use quarter-end balances only, so an institution that files at quarter ends still
gets quarterly flows. 6.8% of the known institution-quarters rest on an inferred zero
(`flow_uses_inferred_zero`); among the 100 largest institutions the filled months are 101, in 13
institutions.

**Outputs.**

| Table | Rows | Content |
|---|---|---|
| `cosif_lai_advertising_monthly` | 247,706 | balances, monthly flows, `zero_inferred` |
| `cosif_lai_advertising_quarterly` | 83,935 | quarterly flows in the columns of the 2025 table, plus `flow_known`, `flow_uses_inferred_zero`, `n_months_inferred`, `cod_congl_2025` |
| `cosif_lai_advertising_halfyear` | 41,873 | the June and December balances, which are the flows of each half |
| `cosif_lai_advertising_long` | 514,175 | the file as received, every document |

73,763 institution-quarters have a known flow.

**Validation.** Eleven checks stop the run before anything is written: the header, the 144
months, the CNPJ and account formats, the sign of every balance, no repeated key in the two
documents used, identity anchors (Banco do Brasil, Caixa, Itaú, Santander), the half-year reset,
falls of a running total under 5% of month pairs (0.6-0.8% found), monthly flows adding up to
quarter flows, the share of gaps, and Santander's statement note within 3% of the file in every
year both have (0.972 to 1.017 over 2015-2024).

An independent review recomputed every quarterly flow from the raw file with its own code, for
all 2,350 institutions and all 215 leaders, and found no disagreement in any of 68,717
institution-quarters and 4,984 conglomerate-quarters. It also found the conglomerate zero error
described above, which is fixed.

**Things the file shows that are worth knowing.**

- Reclassifications inside a half-year are real and are kept, flagged `any_reversal`: PagSeguro's
  advertising balance falls R$48.7m in December 2024, which leaves its 2024Q4 at R$78.9m against
  R$170-205m in the three quarters before.
- An account can be absent in a month between two months where it is positive (80 account-months,
  mostly Caixa). The file says zero, so the flow falls and recovers; these quarters carry
  `any_reversal`.
- Sem Parar, a conglomerate with R$47m in the 2025 public accounts, is not in the file at all.

---

## 6. Coverage in deposit terms

`diagnostics/diag_advertising_coverage.py` computes this, so the figures are reproducible rather than
hand-typed; it writes `advertising_coverage_by_year.csv` and `advertising_coverage_by_source.csv`.

A quarter counts as observed only when all three of its months are known; a quarter short of a
month is excluded.

Share of panel deposits (sum of `dep_a1 + dep_a2 + dep_a4 + dep_a5`, Q4 denominator) held by
conglomerates with an observed quarterly series, from the build of 2026-10-05:

| Year | Conglomerates in the panel | With all four quarters | Share of deposits, all four quarters | Share of deposits, any quarter |
|---|---|---|---|---|
| 2013 | 67 | 63 | 99.97% | 99.97% |
| 2014 | 83 | 80 | 99.97% | 99.97% |
| 2015 | 82 | 79 | 100.00% | 100.00% |
| 2016 | 197 | 156 | 99.79% | 99.79% |
| 2017 | 203 | 162 | 99.85% | 99.85% |
| 2018 | 204 | 176 | 99.92% | 99.92% |
| 2019 | 218 | 177 | 99.93% | 99.93% |
| 2020 | 248 | 195 | 99.89% | 99.93% |
| 2021 | 278 | 215 | 99.87% | 99.91% |
| 2022 | 332 | 238 | 99.60% | 99.86% |
| 2023 | 384 | 276 | 99.80% | 99.91% |
| 2024 | 428 | 283 | 99.67% | 99.88% |
| 2025 | 469 | 436 | 99.88% | 99.95% |

Before the central bank's named file the same column read 64.6% to 79.2% for 2016-2024, on 9 to
13 conglomerates a year. The conglomerates still without a full year are small: they hold under
0.4% of deposits in every year. 2022 is the lowest at 99.60%, because PicPay's own code holds
deposits only from the third quarter of that year.

By source, in share of Q4 deposits:

| Source | Share of deposits | Years |
|---|---|---|
| Central bank accounts, member institutions summed | 99.6% to 100% | 2013-2025 |
| Central bank accounts, consolidated document | 96.3% to 98.3% | 2015-2025 (the document starts in July 2014) |
| Securities filings | 37% to 53% | 2013-2025 |
| Caixa's own files | 16% to 31%, falling as its deposit share falls | 2013-2025 |
| The four state banks' own files | 0% to 3.2% | 2014-2025 |

The SEC series alone is 31-47% from 2015 but annual, so it cannot enter a quarterly regression.

### 6.1 Coverage of the estimation sample, and whether it survives fixed effects

`diagnostics/diag_advertising_in_estimation.py` answers the question the design actually poses. It reads the E3
spec-12 demand parquet, merges ONE source per conglomerate, kept for the whole window and never
spliced, and writes three CSVs. The source is the central bank's accounts wherever they cover the
window: the member institutions summed, else the consolidated document. Another source is chosen
only when it covers more than four quarters beyond them. In this build 269 conglomerates take
the member sum, one takes the consolidated document, and none takes another source.

| Year | Conglomerates in the sample | Covered | Share of rows | Share of deposits |
|---|---|---|---|---|
| 2016 | 112 | 107 | 99.99% | 99.90% |
| 2017 | 116 | 113 | 99.99% | 100.00% |
| 2018 | 122 | 119 | 99.99% | 100.00% |
| 2019 | 120 | 116 | 99.97% | 99.96% |
| 2020 | 156 | 148 | 99.98% | 99.95% |
| 2021 | 182 | 175 | 99.97% | 99.94% |
| 2022 | 210 | 197 | 99.96% | 99.94% |
| 2023 | 236 | 218 | 99.94% | 99.94% |
| 2024 | 275 | 236 | 99.85% | 99.77% |

Before the named file the sample was covered for 13 to 16 conglomerates a year, 67% to 81% of its
deposits. Now 270 conglomerates appear in total, 182 with at least 8 in-window quarters, 102 with
at least 20, and 17 with all 36. That makes 4,304 conglomerate-quarters, against about 450 before.

The identifying variation survives the two-way fixed effects the demand equation carries:

| Variable | After conglomerate FE | After conglomerate and quarter FE | Absorbed |
|---|---|---|---|
| log real advertising | 13.8% | 13.0% | 87.0% |
| advertising / total assets | 55.5% | 55.2% | 44.8% |

That is the test `frac_4g5g` failed at 98.9% absorbed. The median within-bank standard deviation
of log real advertising is 0.82. One caveat on the ratio's larger residual: its denominator moves
too, so part of what survives in `advertising / total assets` is total assets rather than
advertising. The log measure is the conservative read, and it passes.

---

## 7. Data issues found along the way

- The panel's `total_deposits` repeats each conglomerate's national total on every municipality
  row and is not constant within conglomerate-quarter for 2.7% of groups; for Banestes in 2024Q4
  it is exactly twice the sum of the deposit-type columns. Not investigated further here.
- Panel codes, deposits and spreads for the assembly step are read from `market_panel.parquet`
  only; no pipeline file is written or regenerated by any advertising script, and no pipeline
  module is imported (several re-run themselves as scripts when imported outside the project
  interpreter).
- The market panel has NEGATIVE deposits in 27 rows of 2025 and in none before: Banco Sicredi in
  2025Q1 and 2025Q2 (about -R$40bn a quarter, positive again from Q3), Banco Sofisa and Banco
  BMG. Found on 2026-10-05 while checking the cooperative banks. It is in the main data pipeline,
  which no advertising script touches, and outside the 2016-2024 estimation window.
- Parsing constraints on this machine: a local OCR engine is installed and reads Caixa's scans
  (9.1, 5.6); there is no ODS reader library (Banco do Nordeste's files).

---

## 8. IBOPE and per-market exposure

Kantar IBOPE Media's advertising monitoring is proprietary and no academic access route was found.
What is public was catalogued: sector investment shares (financial services about 9% of an R$80
billion total in 2023), the weekly top-ten programmes by audience in each of the 15 metered
markets, annual reach figures, and spend by medium from the agency association with no advertiser
split. The research arm that ran the social-trust surveys closed in 2021 and its successor
publishes no microdata. Three routes are worth pursuing.

**8.1 A free exposure layer.** The awareness tier needs advertising exposure per bank, market and
quarter, and no source sells that for Brazil. It can be constructed: national spend per bank, which
this note's panel now provides, times a market reach weight. The weights come from public data,
verified reachable: the audiovisual and telecoms regulator's access counts by municipality and
provider (2007 onward, municipality x provider x month), broadcast station licences and their
coverage per municipality, and the metered-market audience figures. Media-market ("praca")
definitions are published annually by the media association. This is the only verified route to
per-market variation and needs no licence.

**8.2 Outlet lists for the two federal banks.** The federal audit court obliged Banco do Brasil and
Caixa to publish the outlets and internet domains that carried their paid advertising from 2019.
The files name outlets without amounts, so they cannot give spend, but they say where advertising
ran, which is a direct geographic exposure proxy for two banks holding about a third of deposits.

**8.3 The advertiser ranking.** The annual yearbook of the 300 largest advertisers carries
monitored gross investment per advertiser from 2009. It is the only per-bank media series that also
covers banks disclosing nothing in their own files, such as Santander and BTG. Trade coverage makes
the bank rows free for recent years, verified for 2024: Banco do Brasil R$686.2m, Itau R$630.2m,
Bradesco R$583.4m. Earlier advertiser rows are behind the yearbook paywall. Figures are gross at
list prices, so this is a validation and ranking instrument rather than a substitute for the
accounting series.

Ruled out with measurement rather than assumption: the national procurement register returns zero
contracts for all these banks; the federal communication portal excludes state-owned enterprises by
design; and Meta, Google, TikTok and LinkedIn publish commercial advertising spend only for Europe,
not Brazil, leaving creative counts and date ranges.

A fourth route, a licence inquiry to Kantar for the monitoring product (41 open-television markets,
institution by market by month), is the ideal instrument and is left for the user to consider
separately. [^kantar]

[^kantar]: Cost unknown and likely prohibitive; no academic access route was found, and an inquiry
    commits nothing. Kantar TGI, brand usage by demographics across nine metropolitan areas
    quarterly, would also need a licence.

## 9. Sources found but not yet collected

- **Statement notes for the twelve targets with no usable listing: dropped on 2026-10-05.** The
  central bank's named file has all twelve (Nu, Citibank, XP, Inter, C6, C6 Consignado, Sicredi,
  Sicoob, Master, Mercado Pago, Agibank and PicPay), on the same accounts as everyone else, so no
  listing is written for them.
- **Scanned documents still unread.** Caixa's 20 scanned months are read (5.6). What remains is
  BRB's 38 scanned gazette files, the reason its 2024 shows 9 months rather than 12, and Banrisul's
  November 2024 PDF, whose text layer has a font with no Unicode map. The OCR engine of 9.1 reads
  such pages; the work is rebuilding each table from the OCR boxes. With the central bank's
  accounts as the main measure these files matter only as a cross-check.
- **The administrative-expense total before 2025.** The named file does not carry it. The public
  files do, at level 3, for every institution and month since 2013, so shares of administrative
  expense for 2013-2024 could be added from there.
- **News and union compilations**: a news series on state-bank advertising 2019-2024, and a union
  study on state-bank sponsorship 2018-2023 (2023: BRB R$129.3m, Banrisul R$81.4m, Banpara R$17.6m,
  Banestes R$6.2m). Useful cross-checks on the banks' own files.
- **Awareness proxies, verified.** Wikipedia article pageviews per bank, monthly from July 2015
  (use the `user` agent filter, chain renamed titles, and note that redirects are not aggregated and
  there is no geography); client counts per conglomerate per quarter from the complaints files,
  2014 onward. Ruled out: the central bank's financial-literacy survey has no bank identifier,
  Brand Finance's awareness sub-scores are paywalled and cover few banks, and social and app-store
  histories have no free history.

### 9.1 OCR benchmark, 2026-09-22: recognition is not the constraint

The scanned months were the largest remaining hole, so the OCR chain was scored before anything was
trusted, using `diagnostics/diag_ocr_benchmark.py`. `rapidocr-onnxruntime` 1.4.4 was installed into the project
venv (pure pip, no administrator rights, ~15 MB of ONNX models). It ships CHINESE+ENGLISH models
only - detection, recognition and a classifier - and **no Latin model was needed**: these tables are
digits and uppercase Latin, and every parser folds diacritics anyway (`VEICULACAO`), so the one thing
that dictionary lacks costs nothing here.

**Result on pages whose answer is already known** - 20 Caixa months spanning both the unruled
2014-2018 layout and the later ruled one, rendered to images and compared against the amounts their
own text layer yields:

| months | amount recall | missed | printed totals read | 150 vs 200 vs 300 dpi |
|---|---|---|---|---|
| 20 | **100.0%** | 0 | 20 of 20 | identical |

Resolution makes no difference at all, so the work is cheap. Three defects in the BENCHMARK, not in
the OCR, produced every apparent failure in the first run and are fixed: it demanded OCR find grand
totals that are not printed anywhere (several months print a total per agency and the stated figure
is their sum, as one month's own note says); it counted the layout's empty `0,00` cells as OCR
inventing numbers, of which one page has 122; and it silently skipped the 2019 months, which are
published as zips and recorded as `archive.zip::member.pdf`.

**On the scans themselves**, where no external truth exists - `stated_total_brl` is absent for all
twenty months and no line was ever parsed - mean character confidence is 0.980 across the twenty
months, worst single box 0.52, and the headings, agency columns and category rows all read correctly.
Once both scanned pages are read the monthly maxima land in a plausible R$15-56m band. Three things
about them shape the work that remains:

- The scanned pages are exactly those with **no text layer**, and 16 of the 20 months have TWO of
  them, so the summary spans both. The remaining pages are supplier lists, which do carry text.
- **Labels lose characters where amounts do not.** OCR returns `TOTAL GERA` for "TOTAL GERAL", and
  more often `TOTALGERAL` with no space, both above 0.97 confidence. That is the reverse of the usual
  worry about OCR and money: the digits came through exactly on every one of the 20 known-truth
  months, and it was a word that dropped a letter. Matching on whitespace-stripped text finds the
  total row on 19 of the 20 scanned months (all but January 2014) and the heading on 18; requiring
  the space had reported it missing on 16 months that had in fact read it correctly.
- **One real OCR error, and it is the kind the gate catches.** June 2014 returns R$915,422,471.37 in
  the TOTAL row, far right - absurd for a month whose neighbours are R$15-56m, and the signature of
  two adjacent cells merged into one box. It argues for identifying the total row by its ARITHMETIC
  property, the row whose values equal the column sums, rather than by its label: that is immune to
  label OCR errors and would have rejected this cell on sight.

**What remains is layout, not recognition.** RapidOCR returns boxes with coordinates, the same shape
as the pdfplumber words the Caixa parser already consumes for its unruled layouts, so a scan becomes
a table by the machinery that exists. The acceptance gate is then the table's own arithmetic -
components sum to the printed total - which is the device the BRB parser already uses, and which is
available on these pages because the total is printed on them. Note that summing every amount on the
page is NOT that gate: the summary is a matrix with per-agency totals along the top and TOTAL rows at
the foot, so it double counts and lands at two to three times the month's true spend.

Reproduce with `python diagnostics/diag_ocr_benchmark.py --months 20 --dpi 200` and
`python diagnostics/diag_ocr_benchmark.py --mode scans`.

## 10. Re-running

```text
python scrape_cosif_advertising.py              # 2025 onward; aborts on any failed check
python build_cosif_lai_advertising.py           # 2013-2024 from the central bank's named file
python scrape_cvm_advertising.py                # 2013 onward from the CVM cache
python scrape_cvm_advertising.py --refresh      # redownload the CVM yearly zips first
python scrape_sec_advertising.py                # SEC 20-F instances, cached after the first run
python scrape_sponsorship_bb_bnb.py             # BB and BNB sponsorship; --refresh redownloads
python scrape_caixa_advertising.py              # Caixa's text months and its sponsorship
python scrape_caixa_scan_advertising.py         # Caixa's 20 scanned months, by local OCR
python scrape_statebank_advertising.py          # all four state banks; see the warning below
python scrape_ancine_ad_films.py                # registered advertising films (a count)
python build_advertising_crosswalk.py           # after the scrapers; reads their outputs
python build_advertising_panel.py               # last; assembles both panels and the film table
python diagnostics/diag_advertising_coverage.py # coverage in deposit terms, section 6
python diagnostics/diag_advertising_in_estimation.py  # coverage and FE survival in the estimation sample

# Statement notes (5.9); not read by the assembly.
# First line: fetch from the approved hosts, then parse.
# Second line: re-parse the cache with no network.
python scrape_bank_notes_advertising.py
python scrape_bank_notes_advertising.py --offline --unvalidated volkswagen
```

Run these with the project interpreter (`C:/venvs/egan/Scripts/python.exe`), with `PYTHONUTF8=1`
and `MPLBACKEND=Agg` set.

The statement-note run takes about 35 minutes offline. It exits 0 when everything is written, 75
when the connection is lost (run the same command again to resume), and 1 when a check fails, in
which case nothing is written. As of 2026-10-04 it needs `--unvalidated volkswagen` to finish.

`scrape_statebank_advertising.py --banks <one>` rewrites the four combined outputs with that bank
alone, so a single-bank run is for diagnosis only and must be followed by a full run before the
assembly reads those files. The order matters in two more places. The crosswalk reads every
scraper's output, and the assembly reads the crosswalk, so a re-scrape of any source needs both
rebuilt after it. And `build_cosif_lai_advertising.py` runs after `scrape_cosif_advertising.py`,
because it takes each conglomerate leader's 2025 code from that script's output.
