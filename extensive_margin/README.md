# extensive_margin/

Frozen on 2026-10-14. These scripts build the extensive-vs-intensive margin diagnostic: firm customer counts and deposits from SEC EDGAR, MELI filings, CVM earnings PDFs and CVM DFP/ITR, and World Bank Findex, joined into an account-vs-volume table and its figures. No estimator, table or paper exhibit reads them.

`panel_pipeline.py` runs them as stage 6 only when asked: `python panel_pipeline.py --extensive-margin`, or `--only 6`. The two `make_*` scripts are run by hand.

Each script finds `utils/` through a `sys.path` line that points at the repo root.

Known issue: `write_panel` replaces every stored row of each firm it scrapes. So if `ANTHROPIC_API_KEY` is set only in `.env` and not in the shell, `scrape_incumbent_clients.py` skips its vision step and drops the vision-derived rows.
