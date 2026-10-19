# Handoff: data to collect after the IO Tea and the advisor meeting (2026-10-20)

**To:** scraping agent. **From:** main thread, for Pedro.

**Rule:** `V_Main.tex` and the deck are edited by the author only. Deliver raw files and a source log only. No analysis.

## General rules

- **Official sources first:** the Banco Central do Brasil (BCB) open data portal and its Olinda APIs, Anatel, IBGE, ANBIMA, and the World Bank.
- **Do not guess endpoints or schemas.** Find each source's own documentation, record its URL, and save the documentation next to the data.
- **Source log.** Each dataset gets one row in `source_log.csv`, with these fields:
  - URL
  - date accessed
  - publisher
  - coverage (years, frequency, geography)
  - units
  - licence.
- **Where files go.** Save raw files untouched under `DATA_RAW/devo_followups_2026-10-20/`, one folder per task.
- **Be honest about gaps.** If a source does not exist or is not public, say so. Do not substitute a press article for official data.

## Tasks

```
tasks[5]{id,agent,target,priority}:
  S1,scraping,"accounts and banking relationships per person, Brazil, 2016 to latest",high
  S2,scraping,"surveys on liquidity demand and savings motives, Brazil",high
  S3,scraping,"Pix transactions by municipality, monthly, 2020-11 to latest",high
  S4,scraping,"local presence of digital institutions (any municipality-level measure)",medium
  S5,scraping,"Pix rules and TED/DOC facts for the deck's primer, official texts",medium
```

### S1. Accounts per person

**Targets:**

- The number of individuals with a relationship in the financial system.
- The average number of institutions or accounts per individual, by year.
- If they exist: splits by age, region, or income.

**Where to look:**

- The BCB's statistics built on the Cadastro de Clientes do Sistema Financeiro Nacional (CCS)
- the BCB's Relatório de Cidadania Financeira
- Open Finance participation statistics.

**Flag:** an account is not a person. Caixa Tem alone opened more than 105 million digital savings accounts by end-2020.

### S2. Liquidity demand and savings motives

**Find:**

- surveys with questions on emergency savings
- reasons for holding cash or checking balances
- the use of savings against time deposits
- reasons for not switching banks.

**Candidates:**

- ANBIMA, *Raio X do Investidor Brasileiro* (annual)
- the World Bank's Global Findex microdata for Brazil (2017, 2021, and later waves)
- BCB surveys on financial citizenship
- FEBRABAN's banking technology survey.

**For each survey, give:**

- the years
- the sample
- the exact question wording in Portuguese and English
- whether microdata are public.

### S3. Pix by municipality

**Targets:**

- monthly Pix transactions and values by municipality, by payer and by payee where available
- the count of Pix keys, if published by municipality.

**Where to look:** the BCB Pix statistics pages and their open data.

**Merge:** keep the IBGE municipality code, so the coding agent can map it to MCAs (minimum comparable areas) with the panel's map.

### S4. Local presence of digital institutions

**Question:** is there any public municipality-level measure of customers, accounts, or transactions by institution? D firms enter nationally, so a local measure is the only way to make their entry a local shock.

**Candidates:**

- BCB statistics by institution and municipality
- Pix statistics by participant, if they come with geography
- Open Finance consent statistics.

If nothing exists below the state level, say so. Record what exists at the state level.

### S5. Official texts for the Pix primer

**Collect:**

- Resolução BCB nº 1/2020 and its current consolidated Regulamento
- Resolução BCB nº 19/2020, the fee rules for individuals
- the BCB's statements on TED, DOC, and the end of DOC in 2024.

**On the end of DOC:** press and bank pages give January 15 or February 29, 2024. Find the official announcement.

**On TED fees:** if a BCB or FEBRABAN source documents TED fees for individuals before November 2020, collect it. If not, say so.

## Acceptance

- Each task has its raw files, the documentation, and rows in the source log, or a written note that the source does not exist.
- Nothing is analyzed or merged beyond adding identifiers.

## Open questions for Pedro

1. S2: is Global Findex microdata enough, or do you want only Brazilian surveys?
2. S4: is a state-level measure of digital presence useful, or only municipality-level?
