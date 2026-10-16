# Handoff: rival-spread instrument in the panel (2026-10-15)

From the main thread. Pedro decided that **the panel-side change and the panel rebuild belong to the panel agent**. The main thread does everything downstream (see the last section). Do not edit the downstream files listed there.

## The change (Pedro's instruction)

`panel_market.py`, function `calculate_hausman_iv_wide` (L.755–765 at 16:26 on 10-15), output columns at L.779–781:

- Today, for each deposit type k: `leave_one_out_mean_spread_a{k}` = (sum of `spread_a{k}` over all rows of the quarter − the row's own spread) / (count of non-null rows − 1). Each row is left out on its own, so conglomerate j's other markets stay in its own instrument. That makes the instrument close to the quarter mean of the dependent variable (first-stage strength partly mechanical; memory `sleep-hausman-instrument-construction`).
- **New:** for a row of conglomerate j in quarter t, the instrument is the mean of `spread_a{k}` over the rows of **all conglomerates other than j**, same quarter and same deposit type:
  `rival_mean_spread_a{k} = (S_t − S_jt) / (N_t − N_jt)`
  - S_t and N_t are the sum and the count of non-null `spread_a{k}` over all rows of quarter t.
  - S_jt and N_jt are the same over the rows of `CodConglomeradoPrudencial == j` in quarter t.
  - The result is NaN when N_t − N_jt == 0.
- **Weighting:** keep the current row weighting, each row with weight 1, unless you find a reason not to. If you change it, report the reason.
- **Rename** the column `leave_one_out_mean_spread_a{k}` to `rival_mean_spread_a{k}` (all five types, the `_COL_ORDER` entries included). Rename the function to match; the house rule is no legacy wording in comments.

`panel_local_cdb_spread.py` touches the same column 8 times: `_PATCH_COLS`, the `_national` copies, and `_adjust_loo_spread_a4`, which updates the instrument "by adding the exact" local deviation. Under the new construction j's own rows are excluded entirely, so the adjustment must be re-derived. A local change to j's spread no longer enters j's instrument; it enters every other conglomerate's instrument through S_t. Rename `leave_one_out_mean_spread_a4[_national]` to `rival_mean_spread_a4[_national]`, and update the docstrings, including L.47, which calls it "a sleep HausmanFull instrument".

`panel_pipeline.py` has one mention, in its docs; update it.

## The rebuild (yours)

1. Stage 4 in order, which is load-bearing per the panel-rebuild skill: `panel_pipeline.py --only 4`, i.e. panel_market, then panel_loo_instruments, panel_estban_instrument, panel_local_cdb_spread, and the fee merge last.
2. **Step 39**, `panel_demand_chars.py`, then `panel_demand_chars_checks.py`. The panel md5 changes, and the demand prep refuses a corrections file built from another panel.
3. The post-rebuild audits from the skill: `panel_audit_mca_coverage.py --strict`, `panel_audit_firm_class.py --compare <baseline>`, the instrument columns present (`rival_mean_spread_a1..a5` instead of the old names), the parquet sidecar refreshed.
4. Report to the main thread (through Pedro):
   - the new `market_panel.parquet` md5;
   - the `demand_chars.parquet` sha256;
   - a before/after summary of the instrument: correlation of new vs old per type, and the share of rows that became NaN;
   - anything that surprised you.

## What the main thread does (do not edit these)

- The renames everywhere downstream:
  - the sleepiness stage: `sleep_est_*`, `sleep_export_*`, `sleep_band_uncond`, `sleep_wcb_band`, `sleep_weak_iv`, `sleep_first_stage_strength`, `sleep_ident_augmented`, `sleep_desc_clusters`, `sleep_upsilon_export`, `sleep_job.sh`;
  - `config/routines.toml` (the spec `IV_HausmanFull` becomes `IV_RivalSpread`);
  - the demand preps (`sleep_demand_prep_*`);
  - `utils/state_transform.py`;
  - the table and macro generators, which print the label "Rival mean spread";
  - diagnostics, the cluster bundle and tests, the skill text;
  - the `switch_cost_*` copies (Pedro: rename them too).
- The cluster launch: the sleepiness stage is now re-estimated **on the cluster**. Point estimates come first; the bootstraps run off the critical path.
- V_Main prose stays Pedro's.
