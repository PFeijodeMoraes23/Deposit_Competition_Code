# Pipeline rework — live status

Plan: `C:\Users\pedro\.claude\plans\ok-let-s-do-the-reflective-church.md`
Started 2026-08-19. Legend: `[ ]` not started · `[~]` in flight · `[x]` done · `[!]` blocked/needs you


> **Reconciled 2026-08-24.** This file SHRINKS as work closes - finished sections are deleted, not
> ticked. Closed since the last pass: the joint-sieve removal (four files, ~1.4k lines; E3/E4
> reproduce their stored bootstrap to 1.137e-10 / 3.13e-11), W2 registries incl. phase 2, the dead
> `1 2 5 6 7 8` lineup in five `submit_*.sh`, and the two hardcoded `\ref` ranges that rendered `??`.
> **Blocked on the cluster**: BLP/BBL/CF exhibits and the E3/E4 table regeneration (W11.d), which
> needs `ame_twostage_est{3,4}_robust*.pkl` downloaded from `data/output/Rout`.

## NEXT UP

1. **W11.d** regenerate the E3/E4 tables from the downloaded AME pickles
2. **W3** orchestrator consolidation, then **W7** `diag_` -> `step_`
3. **W4.2-4.8** battery as an orchestrated stage + staleness guard
4. **W13** CIR rate-process variant; **W12** move panel_8 + polfunc to the cluster

---

## W13 — CIR rate-process option alongside the Focus path (decided 2026-08-20)

**What Egan et al. (2025) actually do** — verified from the working paper PDF (HBS 26-015), not from
memory. Appendix B.1, p.57:

> "we parameterize the short rate transition process using a discretized **Cox et al. (1985)** process
> with mean reversion, kappa, of .11, a long-term average rate, theta, of .03, and a volatility, sigma, of .08."

i.e. CIR `dr = kappa(theta - r)dt + sigma*sqrt(r)dW` — the sqrt(r) keeps r >= 0 and makes the transition
density non-central chi-squared, **not Gaussian**. The Normal in that paper is on the POLICY function,
same paragraph: *"mean-zero, normally-distributed shocks to banks' chosen spreads, sd ~21 bps"*.

Their counterfactual machinery (p.61): solving 54,000 markets x 51 periods x 1000 sims is infeasible,
so they **discretize the short rate on an equal grid 0 to .05 in 25 bp buckets**, solve equilibria at
every grid point, then forward-simulate over **1000 drawn paths**, assigning each draw to a bucket by
floor() and averaging discounted profits. The same 1000 paths are reused from their baseline two-step
estimation.

**Ours today**: `cf_forward_rf.py` builds ONE deterministic path from live BCB data (SGS 4189 anchor +
Focus/Olinda median, declining to the long-run median, held flat past the Focus horizon). Both designs
break the omega/zeta collinearity because both are time-varying — but a single path yields no rate-risk
dispersion, so franchise-value VARIANCE and tail outcomes (their >20% default probability) are not
expressible from it.

**Decision: keep the Focus path as the headline, ADD a CIR variant.**
- [ ] **W13.1** CIR simulator calibrated to the **Brazilian Selic** (estimate kappa/theta/sigma from the
      panel's own Selic series — do NOT inherit their US .11/.03/.08, which describe the fed funds rate)
- [ ] **W13.2** Rate grid + per-grid-point equilibrium solve, then N-path forward simulation and
      averaging (mirrors their p.61 design; reuse the same draws across estimation and CF as they do)
- [ ] **W13.3** `forward_rf_qoq.csv` already carries a `source` column — extend it so a CIR run is
      self-labelling and can never be confused with the live Focus curve or the offline fallback
- [ ] **W13.4** Report both: Focus path = "along the expected path"; CIR = "in expectation over rate
      risk", which is the object needed for value-under-uncertainty and tail statements

**Framing for the paper**: the Focus path is arguably the stronger choice for a Brazil study — actual
market expectations for the actual economy, rather than a US-calibrated process. The CIR variant buys
comparability with Egan et al. and the dispersion their default-probability result requires. State the
difference explicitly rather than leaving a reader to infer we simply did it differently.

## W12 — Two "local prerequisites" that need not be local (deferred 2026-08-20)

Of the three inputs the stager requires a human to refresh before every upload, only ONE is
genuinely local:

| producer | must be local? | why |
|---|---|---|
| `cf_forward_rf.py` | **YES, permanently** | hits BCB SGS 4189 + Focus/Olinda; compute nodes have no outbound internet |
| `panel_8_demographics_sigma.py` | no | pure pandas over `market_panel.csv`, which the run uploads anyway |
| `estimation_bbl_1_polfunc.py` | no | same; a cluster path already exists (`bbl_run.sh --polfunc`, off by default) |

Moving the latter two into the chain would cut ~30 min of local compute per run and ~68 MB from
the data bundle (`polfunc_fitted.csv` 59 MB + `demographics_sigma.parquet` 8.7 MB), and make them
gated cluster steps like everything else. They are uploaded inputs today only because that
classification predates the sleepiness phase moving to the cluster.

- [ ] **W12.1** Enable `bbl_run.sh --polfunc` in the chain; reclassify its manifest row `produced`
- [ ] **W12.2** Add a small `panel_8` cluster job (or a `SLEEP_STEP=demog` branch); reclassify likewise
- **Deferred by decision 2026-08-20**: ship the first cycle as-is; revisit once one full cluster run
  has succeeded end-to-end. Do not do this while the chain is still unproven.

**Why `cf_forward_rf.py` can never move** (worth recording — it is not just an API convenience):
in the BBL value basis ψ4 = Σ β^t r^f_t · Σ_k Dep_t. A FLAT r^f_t collapses ψ4 to a rescaling of
ψ2, making ω and ζ collinear and leaving **ζ (funding-cost pass-through) unidentified**. The live
Focus curve exists to break that collinearity. Its CSV carries a `source` column so an offline
fallback path can never be mistaken for the live curve.

## W11 - [!] E3/E4 AME tables still to regenerate

The two-stage bootstrap is implemented and the cluster has run it; theta is perturbed via its
cluster influence functions AND the link re-profiled per draw, so every row now carries its own t.
The off-switch gate reproduces the old conditional numbers to ~1e-10/1e-11.

- [ ] **W11.d** Regenerate `est{3,4}_second_stage_table.tex` + `est1-4_spec12_stage2_comparison*.tex`
      from the downloaded `ame_twostage_est{3,4}_robust*.pkl`; the table note gains one sentence:
      SEs include direction-estimation uncertainty

## W0b — V_Main sleepiness exhibits: audit of what today's run owes

Audit of all 41 `\input`/`\includegraphics` in V_Main.tex against disk + generators (2026-08-19).

**Produced by re-running steps 5–8 (blocked only on W0.5):**

**Needs the local Julia logit after demand parquets exist (W3.2 step 14):**
- [ ] `est3_spec12_logit.tex`, `est4_spec12_logit.tex`, `est1-4_spec12_logit_comparison.tex` ← `blp_1_logit.jl`

**Needs the cluster re-run:**
- [ ] `blp_rc_E3_spec12.tex`, `blp_rc_E4_spec12.tex` ← `make_blp_rc_table.py`
- [ ] `blp_demand_comparison_noseg_spec12.tex` (currently 08-12)

**[!] Must be REMOVED from V_Main — deleted lineup, no generator will ever emit these again:**
- [ ] `blp_rc_E7_spec12.tex`, `blp_rc_E8_spec12.tex`
- [ ] `est7_spec12_logit.tex`, `est8_spec12_logit.tex`
- [ ] `est_phi_t_joint_sieve_pair.png` (`_make_pair_plot` emits only the single-index pair)

**Valid and kept:** `est_phi_t_single_index_pair.png` (built from the genuine `ts_link_band_est{3,4}.pkl`).

## W1 — Stale disposition (~73 GB)

- **Total reclaimed 31.04 GB**; `ESTIMATION_OUTPUT` now 54.4 GB. Log: `_ARCHIVE_PRE_RELABEL_20260818/DISPOSITION_LOG.md`
- Correction from the sweep: **E7's psi_dev shard set is complete at 100/100** — the "missing" shard was `or.parquet`, a truncated-name copy of `psi_dev_E7_spec_12_extended_shard71of100.parquet` (verified by schema, 150 firms, shock 22)
- [ ] **W1.5** *(hold)* JI jsons stay until the battery re-runs under the new lineup
      series (244 KB total, est1/est2/est5–est8 from BOTH roots) were copied to
      `_ARCHIVE_PRE_RELABEL_20260818/vintage_comparators/` first, so the old-vs-new comparison stays
      reproducible at the reporting level without the multi-GB pickles

## W2 - Single-source registries

- [ ] One duplication left, MEASURED to agree on all 12 specs (2026-08-24): `SPEC_MAP` in
      `estimation_demand_link_common.py` vs the `_IV_ORDER` x state-block enumeration in
      `estimation_sleep_common.py:143`. Consolidating touches the estimator's own spec
      enumeration, so not next to a live cluster run

## W3 — Orchestrator consolidation

- [ ] **W3.1** Extract `utils/pipeline_runner.py` (waves, resume predicates, RAM gate, env hardening, RUN_MANIFEST.json)
- [ ] **W3.2** Rebuild `run_sleep_pipeline.py` on it — 15 steps through bands, diagnostics, battery, tables
- [ ] **W3.3** Delete the two dead shell drivers; wire `check_mca_coverage.py` as a data-pipeline gate

## W4 — Weak-IV integration into the BLP pipeline (hybrid)

- [ ] **W4.2** Battery becomes an orchestrated stage (sleep-side + demand-side)
- [ ] **W4.3** Staleness guard: `input_fingerprint` on every battery json
- [ ] **W4.4** `make_iv_tables.py` hard-fails instead of emitting empty tables
- [ ] **W4.5** Ingest hook: `process_blp_outputs.py` auto-runs export_rc_delta → battery → tables
- [ ] **W4.6** *(cluster batch)* `se_common.jl` emits per-cluster moment blocks (spread_hat, A_g, B_g)
- [ ] **W4.7** *(cluster batch)* Battery fast path reads blocks; cross-check vs parquet path
- [ ] **W4.8** *(after cluster re-run)* Regenerate iv tables + update `weakiv_report.tex` prose

## W5 — Upload / ingest tooling

- [ ] **W5.4** Cluster preflight blocks read the manifest *(cluster batch)*

## W10 - Residue from the joint-sieve removal

- [ ] `tab_phi_need_sleep.tex` still holds joint-sieve-vintage numbers. Its sieve arm now fits with
      `fit_single_index`, so a re-run of `weak_iv_sleep_analysis.py` + `make_iv_sleep_tables.py`
      will move the numbers. Not `\input` into V_Main, so nothing in the paper moves today
- [ ] `run_diag_matrix.py:25,133` - `--units d0:1,2,5,6,7,8` in help; the diag-matrix unit ids are
      their own namespace and must be reconciled with `run_phi_diagnostics.py` first (which also has
      the duplicate `D4b` key bug, W3.2)

**Deliberately keep:** `utils/se_national.py:249` - a dated (2026-07-30) adversarial-review finding
about which stored pickles lacked quarter blocks *at that date*; naming E8 is part of that record.

## W9 - Retire the legacy submit_*.sh

- [ ] *(after one green cluster cycle)* delete the 23 superseded `submit_*.sh`; keep
      `setup_julia_env.sh`. `pipeline_all.sh` is the proven path; these remain only as a fallback

## W8 - WCU vs WCR, narrowed to the linear routines

W11's two-stage bootstrap is its own reference for the E3/E4 AME rows, so this is now only about
E1/E2 and the battery. Measured sizes at nominal 5%: normal 0.270 / WCU 0.138 / WCR 0.083 / CRVE-t
0.145 (MC SE 0.0109), so WCR beats WCU but is still oversized.

- [ ] **W8.2** Choose WCU vs WCR for E1/E2 and the weak-IV battery - **user decision**
- [ ] **W8.4** `utils/sleep_links.py` coerces `wcr`->`wcu` on the generic score path and warns once
      per process via `_WCB_WARNED`. Record the realised mode on each results object so a table can
      state its own reference distribution

## W7 — Rename pipeline-integrated `diag_*` → `step_*`

Scripts that became pipeline steps stop advertising themselves as ad-hoc diagnostics.
**Sequenced after W3.2 (step list settled) and W4.1–4.5 (those files are being edited now)** — renaming
before the step list is final would rename the wrong set.

- [ ] **W7.1** Rename the 7 integrated scripts (`git mv`-equivalent, no commit):
  - `diag_phi_augmented_tests.py` → `step_phi_augmented_tests.py`
  - `diag_phi_interaction_tests.py` → `step_phi_interaction_tests.py`
  - `diag_phi_mc_recovery.py` → `step_phi_mc_recovery.py`
  - `diag_entry_dynamics.py` → `step_entry_dynamics.py`
  - `diag_moment_reduction.py` → `step_moment_reduction.py`
  - `diag_cue_linear_step.py` → `step_cue_linear.py` *(avoids `step_..._step`)*
  - `diag_openfinance_freshness.py` → `step_openfinance_freshness.py`
- [ ] **W7.2** Update all call sites — 18 files, ~65 refs. Load-bearing ones:
  `run_phi_diagnostics.py` (19 refs: STEPS table + docstring), `run_diag_matrix.py` (10: UNITS table + comments),
  **`diag_moment_reduction.py:68` `from diag_cue_linear_step import (X_COLS, IV_COLS, _build_matrices, _load_delta, …)`**,
  `run_openfinance_monthly.py`, `make_diag_tables.py`, `make_iv_tables.py`, `make_d2bc_cross_routine.py`,
  `panel_10_estban_instruments.py`, `scrape_19_openfinance_fees.py`, `utils/phi_reference.py`, `utils/br_calendar.py`,
  `diag_wcb_reference.py`, plus the new orchestrator's step table
- [ ] **W7.3** Verify: `grep -rn "diag_(phi_augmented|phi_interaction|phi_mc|entry_dynamics|moment_reduction|cue_linear|openfinance_freshness)"` returns nothing; import-smoke every renamed module and every caller; `run_phi_diagnostics.py --list` and `run_diag_matrix.py --dry-run` still enumerate every unit

**Stays `diag_`** (genuinely ad-hoc, not pipeline steps): `diag_im_phi_t.py`, `diag_index_identification.py`,
`diag_k4_spread_local_variation.py`, `diag_openfinance_coverage.py`, `diag_test_inversion_band.py`,
`diag_wcb_reference.py`. Being deleted: `diag_uncond_band_joint.py`.

- [!] **Open decision (yours)**: do the *output artifact* names follow the scripts?
  `diag_moment_reduction.json`, `diag_cue_linear_step.json`, `.diag_matrix_state.json`, dirs `DIAG_PHI_SEPARATION/`,
  `DIAG_WEAK_IV_SLEEP/`. **My default: no** — those are data contracts read by table generators and already
  present in archives/cluster outputs; renaming them breaks reads of existing vintages for cosmetic gain.
  Say the word if you want them renamed too.

## W6 — Docs

- [ ] **W6.1** README pipeline sections
- [ ] **W6.2** `weakiv_methods.md` §8 reproduce block
- [ ] **W6.3** Memory updates + disposition log

---

## Needs you (not mine to do)

- [ ] **V_Main.tex** dangling refs: L462 `\ref{estimation:single_index}` → `single_idx` (preferred-spec sentence); L695 `single_idx_timee`; L691 `tab:blp_rc_est4spec12`; L606 `joint_sieve` ×2 (needs rewrite); L462 prose "all six approaches" → four
- [ ] Decision on commit `d105f3bd` (made before the no-commit rule; offer stands to soft-reset)
- [ ] Delete `estimation_timeseries_test.py`
- [ ] **Security**: repo-root `.env` holds a plaintext `ANTHROPIC_API_KEY` — confirm untracked, rotate if ever shared
- [ ] Cluster re-run on Bouchet (upload staged by W5, then BLP RC → BBL → CF)
