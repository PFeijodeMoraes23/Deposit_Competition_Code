# fees/

Frozen on 2026-10-14. These are the two BCB scrapers for the listed (published) bank fees. Both call the same Olinda service (`Informes_ListaTarifasPorInstituicaoFinanceira`):

- `scrape_bcb_tarifas.py` gets the current schedules. It was pipeline step 10, but its outputs are two `--test` runs and the step never ran a full scrape.
- `scrape_bcb_tariff_vigencia.py` gets the last price-change date and price per service. `panel_fee_merge.py` reads its `tariff_vigencia_history.csv`, which is the source of `listed_fee_*`.

No estimator reads the fee columns. Before any fee specification uses them, fix the service-code labels in `panel_fee_merge.py`:

- code 1501 is "PACOTE PADRONIZADO DE SERVIÇOS I", not an ATM withdrawal;
- the withdrawal codes are 1209 and 1210;
- code 1203 is the CCF exclusion.

Run either script by hand from the repo root, for example `python fees/scrape_bcb_tariff_vigencia.py`. Each finds `utils/` through a `sys.path` line that points at the repo root.
