# fees/

The fee branch of the panel pipeline. No estimator reads it: estimators read `market_panel.csv`, and `market_panel_with_fees.csv` is read only with `USE_FEE_PANEL=1`.

- `panel_fee_merge.py --patch-market` builds `market_panel_with_fees.csv`. It is opt-in stage 7 of `panel_pipeline.py` (`--fees`, or `--only 7`) and runs after every stage-4 step.
- `panel_audit_fee_coverage.py` audits that file (`--strict` exits non-zero on a failed floor). It reads `market_panel_with_fees.csv` directly.
- `scrape_cosif_service_fees.py` builds the COSIF realized-fee inputs of the merge. It is outside the runner.
- `panel_diag_fee_reversal.py` is a diagnostic: it draws figures from the COSIF fee output, the Open Finance fee panel and the CVM fee income. Nothing reads its output.
- `scrape_cvm_fee_income.py` collects the banks' fee income from CVM filings (`FirmDisclosures/CVM/cvm_fee_income.csv`), a validation source for the diagnostic. It is outside the runner, and it shares its raw-filing cache with `scrape_cvm_advertising.py`.

The two BCB scrapers for the listed (published) bank fees were frozen on 2026-10-14. Both call the same Olinda service (`Informes_ListaTarifasPorInstituicaoFinanceira`):

- `scrape_bcb_tarifas.py` gets the current schedules. It was pipeline step 10, but its outputs are two `--test` runs and the step never ran a full scrape.
- `scrape_bcb_tariff_vigencia.py` gets the last price-change date and price per service (`tariff_vigencia_history.csv`, the Olinda snapshot of 2026-06-22). It dates a price only from its vigência on, and it is the only source after 2024.

The listed-price history comes from two scripts outside the runner. Stage 7 reads their outputs, so run them before `panel_fee_merge.py`.

- `scrape_bcb_tarifas_monthly.py` downloads the BCB's monthly census of listed tariffs (`<YYYYMM>TARINST.ZIP`, every institution × service, PF and PJ, with the date in force) into `DATA_RAW/BCB/tarifas_monthly`.
  - 2015-01..2024-12 is collected: 120 files.
  - The archived listing names the months to 2018-07; later months follow the same name pattern (`--unlisted`).
  - The script writes `tariff_history_bcb_monthly.csv`, one row per price spell.
- `build_account_fee_history.py` joins that census with the Olinda snapshot for the account-keeping services: Pacotes Padronizados I–IV (1501–1504) and PJ account maintenance (0402, 0403). It writes:
  - `account_fee_listed_history.csv`, the dated versions;
  - `account_fee_listed_by_year.csv`, months known per institution and year 2016–2024, with the reason a month is unknown;
  - `account_fee_listed_coverage.csv`, the shares known.

`panel_fee_merge.py` turns these into quarterly listed-price columns, joined on conglomerate × year × quarter:

- **Services**, labelled by code (`LISTED_FEE_SERVICES`): Pacotes Padronizados I–IV (`listed_fee_pkg1_pf`..`pkg4_pf`, codes 1501–1504), PJ account maintenance (`maint_active_pj` 0402, `maint_inactive_pj` 0403), cadastro (`acct_reg_pf` 1101), 2nd debit card (`card_2nd_debit_pf` 1201), CCF exclusion (`ccf_exclusion_pf` 1203), cheque countermand (`cheque_countermand_pf` 1204), withdrawals at the teller (`withdrawal_teller_pf` 1209) and at a terminal (`withdrawal_atm_pf` 1210), statement (`statement_pf` 1213).
- **Price:** for each institution, the version in force on the quarter's last day. For the six account-keeping services it comes from `account_fee_listed_history.csv`; the other seven go through the same `load_versions()` rule. Prices are in R$ only; percentage listings are dropped and logged.
- **Window:**
  - 2015-Q1..2024-Q4 from the census;
  - 2025-Q1 on, from the snapshot alone, where its vigência is on or before the quarter end;
  - 2013–2014 NaN.
- **Conglomerate:**
  - `listed_fee_<name>` is the head institution's price: the canonical leader, the same leader as the panel's `CNPJ_Lider`.
  - When the head does not list the service, it is the median of the members' positive prices; it is 0.00 when every listing is 0.00.
  - `listed_fee_n_inst_<name>` counts the members with a price.
  - Members are placed the way the panel places their deposits (`utils.conglomerates.deposit_panel_codes`). An ESTBAN bank keeps its static code in every quarter; every other institution follows the IF Data List of that quarter.
- **Stickiness:** `tarifa_stickiness_yrs` is the years since the vigência of the price in force, for 2015–2024. It is averaged over the head's PF services, or over every member's when the head has none. `tarifa_stickiness_n` counts the pairs averaged.
- **Cautions:**
  - Listed is a ceiling, not the price charged.
  - An institution absent from the census lists nothing: NaN, not zero.
  - A listed R$0.00 is a price. A head's 0.00 is used as it stands, but the fallback median leaves zeros out. For example, BB Banco de Investimento lists Package I at 0.00, which does not affect Banco do Brasil's value.
  - Bridged prices carry under 1% of known months.

No estimator reads the fee columns.

Two benchmarks sit outside the runner, and stage 7 does not read them:

- `build_consumidor_fee_complaints.py` counts fee disputes in the consumidor.gov.br complaint files for 2016–2024, for firms in the two financial segments.
  - **Counts:** complaints naming a tariff (`n_fee`), complaints about accounts (`n_account`), and both together (`n_fee_account`), with their shares.
  - **Outputs:**
    - `consumidor_fee_complaints_firm_quarter.csv`, by firm and quarter;
    - `consumidor_fee_complaints_municipality_quarter.csv`, by IBGE municipality and quarter, through the city crosswalk;
    - `consumidor_fee_complaints_firm_municipality_year.csv`, by firm, municipality and year;
    - `consumidor_fee_complaints_firms.csv`, the list of firm labels.
  - **Firms** keep their consumidor.gov.br label. No conglomerate is assigned here.
  - **Run it** with `python -I`.
- `build_consumidor_firm_conglomerates.py`, run after it, puts the labels on prudential conglomerates.
  - **Matching:** a label reaches a CNPJ root through a checked table, or through an exact name match with an institution on disk. The code comes from `build_cnpj_conglomerate_map`.
  - **Coverage:** 129 of 318 labels resolve, on 85 conglomerates, holding 94.7% of complaints and 99.2% of account-fee disputes.
  - **Outputs:**
    - `consumidor_firm_conglomerate_crosswalk.csv`;
    - `consumidor_fee_complaints_conglomerate_quarter.csv`;
    - `consumidor_fee_complaints_conglomerate_municipality_year.csv`.

- `build_fee_paper_numbers.py` writes `fee_paper_numbers.json`, the sidecar of the ten `fees.*` numbers for V_Friction Section 7.2. `make_paper_numbers.py` reads it; the coding agent owns that collector.
  - **Sources:** the POF, the conglomerate complaint table and stage 7's listed Package I value. The script calls `panel_fee_merge.py`'s own functions, read only.
  - **Run it** after the scripts it reads, with `python -I`.
- `build_pof_bank_fees.py` measures what families report paying in bank fees in the POF 2017–2018 (IBGE microdata in `DATA_RAW/IBGE/pof/pof2017_18`, one cross-section, R$ of 2018-01-15). It covers three measures:
  - the account-keeping fee, item 26005;
  - IBGE's bank-fee category 61;
  - that category without overdraft and credit-card interest.
- It writes three files:
  - `pof2017_bank_fees_by_group.csv`: Brasil, region, UF and per-capita income decile, with the share of families paying and the mean per family, per paying family and per capita;
  - `pof2017_bank_fees_items.csv`: one row per item code;
  - `pof2017_bank_fees_check.csv`: IBGE's published tables 6.2 and 6.3, recomputed.
- Run it as `python -I fees/build_pof_bank_fees.py`, because the POF files are downloaded data.

Run any script by hand from the repo root, for example `python fees/scrape_bcb_tariff_vigencia.py`. Each finds `utils/` through a `sys.path` line that points at the repo root.
