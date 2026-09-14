# Bank advertising data: sources, collection and results

Advertising expense per bank per quarter, 2013 onward, for the awareness tier of the deposit
demand model. This note records where the data can and cannot be found, how each source was
checked against the central bank's own accounts, the choices made, and what the collection
scripts produce.

Last updated: 2026-09-14.

## Status

| Piece | Script | State |
|---|---|---|
| Central bank accounts, 2025 onward | `scrape_cosif_advertising.py` | Built, validated, outputs written |
| Securities-regulator filings, 2013 onward | `scrape_cvm_advertising.py` | Built, validated, outputs written |
| SEC annual reports (separate annual panel) | `scrape_sec_advertising.py` | Built, validated, outputs written |
| State-owned banks' own disclosures | not yet written | In progress |
| Sponsorship robustness series | not yet written | In progress |
| Assembly onto panel conglomerate codes | not yet written | Pending |
| Freedom-of-information request to the central bank | filed on Fala.BR | Awaiting reply |

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

## 3. Decisions (all by the user, 2026-09-14)

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

---

## 5. Built datasets

### 5.1 Central bank accounts, 2025 onward: `scrape_cosif_advertising.py`

**Inputs.** `{YYYYMM}BANCOS` (document 4010, per institution, keyed by CNPJ8; COD_CONGL is blank)
and `{YYYYMM}BLOPRUDENCIAL` (document 4060, per prudential conglomerate, keyed by COD_CONGL, the
panel's `CodConglomeradoPrudencial` code space) under `paths.COSIF_RAW`, 202501-202603.

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

**Validation (all pass).** No level-4 advertising account in the 24 files of 2024; every raw
balance negative; children sum to the total for every filer and month; July median below June;
advertising reporters per month within bounds (4010: 96-116; 4060: 111-137); anchors C0080099
ITAU, C0080738 CAIXA, C0080329 BB; no duplicate entity-account-months; reversal share under 5%;
monthly flows reproduce quarter flows for all 1,642 complete entity-quarters.

**Spot checks.** Itau prudential 2025 quarters 298.1 / 346.6 / 299.3 / 358.1 R$m and Banco do
Brasil institution 2025 total 528.2 R$m, both matching independent figures exactly. 144 of the
179 conglomerates with an advertising account are panel conglomerate codes.

**Outputs.**
- `cosif_advertising_monthly.{parquet,csv}` (5,340 rows): `level`, `entity_key`, `entity_name`,
  `cnpj_leader`, `taxonomia`, `data_base`, `year`, `month`, `half`, `quarter`, `filed`,
  `{adv,promo,publ,admin}_bal`, `{adv,promo,publ,admin}_flow`, `is_reversal`,
  `adv_flow_spike`, `panel_key`.
- `cosif_advertising_quarterly.{parquet,csv}` (1,780 rows): flows `adv`, `promo`, `publ`,
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
| Institution | 2025Q1 | 174 | 1,091.4 | 1,748.8 |
| Institution | 2025Q2 | 175 | 1,199.3 | 1,846.7 |
| Institution | 2025Q3 | 171 | 1,187.9 | 1,870.9 |
| Institution | 2025Q4 | 170 | 1,625.6 | 2,622.8 |
| Institution | 2026Q1 | 170 | 997.4 | 1,655.8 |

**Spikes (`adv_flow_spike`, kept unaltered).** A month more than 5x the entity's median positive
monthly flow and more than R$10m above it. 15 entity-months. The dominant one: NU PAGAMENTOS
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

---

## 6. Coverage in deposit terms

Share of panel deposits (sum of `dep_a1 + dep_a2 + dep_a4 + dep_a5`, Q4 of each year).

| Year | CVM quarterly, full year | plus SEC annual | Caixa alone |
|---|---|---|---|
| 2016 | 38.0% | 57.8% | 30.7% |
| 2020 | 52.7% | 65.6% | 19.4% |
| 2024 | 47.6% | 62.9% | 16.5% |

With Caixa's own files the covered share is roughly 80-88%. State banks' shares (average of the
four quarters): Banrisul 2.5 / 2.0 / 1.8%, BRB 0.6 / 0.5 / 0.8%, Banestes 0.6 / 0.5 / 0.5%,
BNB 0.6 / 0.5 / 0.3%, Banpara 0.2 / 0.3 / 0.3%, BASA 0.2 / 0.2 / 0.3%, Banese about 0.2%
(2016 / 2020 / 2024). BNDES is not in the panel.

---

## 7. Data issues found along the way

- The panel's `total_deposits` repeats each conglomerate's national total on every municipality
  row and is not constant within conglomerate-quarter for 2.7% of groups; for Banestes in 2024Q4
  it is exactly twice the sum of the deposit-type columns. Not investigated further here.
- Parsing constraints on this machine: no OCR engine (Caixa's 2013 to January 2014 scans cannot
  be read yet) and no ODS reader library (Banco do Nordeste's files).

---

## 8. Re-running

```text
python scrape_cosif_advertising.py              # 2025 onward; aborts on any failed check
python scrape_cvm_advertising.py                # 2013 onward from the CVM cache
python scrape_cvm_advertising.py --refresh      # redownload the CVM yearly zips first
python scrape_sec_advertising.py                # SEC 20-F instances, cached after the first run
```
