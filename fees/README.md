# fees/

The fee branch of the panel pipeline. No estimator reads it: estimators read `market_panel.csv`, and `market_panel_with_fees.csv` is read only with `USE_FEE_PANEL=1`.

- `panel_fee_merge.py --patch-market` builds `market_panel_with_fees.csv`. It is opt-in stage 7 of `panel_pipeline.py` (`--fees`, or `--only 7`) and runs after every stage-4 step.
- `panel_audit_fee_coverage.py` audits that file (`--strict` exits non-zero on a failed floor). It reads `market_panel_with_fees.csv` directly.
- `scrape_cosif_service_fees.py` builds the COSIF realized-fee inputs of the merge. It is outside the runner.
- `panel_diag_fee_reversal.py` is a diagnostic: it draws figures from the COSIF fee output, the Open Finance fee panel and the CVM fee income. Nothing reads its output.
- `scrape_cvm_fee_income.py` collects the banks' fee income from CVM filings (`FirmDisclosures/CVM/cvm_fee_income.csv`), a validation source for the diagnostic. It is outside the runner, and it shares its raw-filing cache with `scrape_cvm_advertising.py`.

The two BCB scrapers for the listed (published) bank fees were frozen on 2026-10-14. Both call the same Olinda service (`Informes_ListaTarifasPorInstituicaoFinanceira`):

- `scrape_bcb_tarifas.py` gets the current schedules. It was pipeline step 10, but its outputs are two `--test` runs and the step never ran a full scrape.
- `scrape_bcb_tariff_vigencia.py` gets the last price-change date and price per service. `panel_fee_merge.py` reads its `tariff_vigencia_history.csv`, which is the source of `listed_fee_*`.

No estimator reads the fee columns. Before any fee specification uses them, fix the service-code labels in `panel_fee_merge.py`:

- code 1501 is "PACOTE PADRONIZADO DE SERVIÇOS I", not an ATM withdrawal;
- the withdrawal codes are 1209 and 1210;
- code 1203 is the CCF exclusion.

Run any script by hand from the repo root, for example `python fees/scrape_bcb_tariff_vigencia.py`. Each finds `utils/` through a `sys.path` line that points at the repo root.
