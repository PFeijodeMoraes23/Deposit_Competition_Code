# Pipeline rework — live status

Plan: `C:\Users\pedro\.claude\plans\ok-let-s-do-the-reflective-church.md`
Started 2026-08-19. Legend: `[ ]` not started · `[~]` in flight · `[x]` done · `[!]` blocked/needs you


> **Reconciled 2026-08-20 14:40.** Today closed: local promotion + verification, the AME off-switch
> gate (~1e-11, both routines), all Python/Julia/shell cluster-migration edits, the upload tooling,
> and the paper-facing joint-sieve strings. Bundles are staged at `C:\egan_cluster_stage60820`.
> **Blocked on the cluster run**: the logit/BLP/BBL/CF exhibits, the full two-stage AME, the bands.
> **Superseded**: W9 as originally written (a one-command `pipeline_all.sh` now exists; consolidating
> the 23 legacy `submit_*.sh` is now cleanup, not a prerequisite).

## ☀️ READ THIS FIRST IN THE MORNING (2026-08-20)

**Before you upload anything to Bouchet, run these 7 locally.** The stager
(`python stage_cluster_upload.py`) prints this list itself and refuses `--stage` until they pass:

```
python panel_8_demographics_sigma.py
julia --project=. --threads=auto blp_1_logit.jl --est 1     (then --est 2, 3, 4)
python cf_forward_rf.py --horizon 50 --start 2026Q1
python estimation_bbl_1_polfunc.py
```

Then `python stage_cluster_upload.py --stage` → two zips in `C:\egan_cluster_stage\<date>\`, each
carrying an `UPLOAD_README.txt` with numbered steps. Code bundle ~252 KB, data bundle ~237 MB.

**[FIXED last night]** Those producers resolve `DEMAND_PREP` to **production**, which had zero
parquets — `blp_1_logit.jl` discovers routines by file presence, so it would have found none and
**exited 0**, silently. The 44 demand parquets (2.28 GB) are now promoted to production, so they work.
The `est*/` dirs are deliberately NOT promoted yet — the bootstrap rewrites their SEs.

**[!] On the cluster, pass `ROUTINES="1 2 3 4"` explicitly.** `submit_cf_all.sh`, `submit_bbl_all.sh`
and `submit_blp_2_rc_default.sh` still default to the dead `"1 2 5 6 7 8"`, so their preflights will
hunt for E5–E8 files this vintage does not have. W9 (running overnight) may supersede this.

**[!] Toolchain skew — check before submitting.** Scripts hardcode `module load
Julia/1.11.4-linux-x86_64`; Bouchet's docs list **1.10.4**; `Manifest.toml` was resolved under local
**1.12.6**. Run `module avail Julia` first. A mismatch triggers a manifest re-resolve, which must go
through `setup_julia_env.sh` alone (concurrent resolves corrupt the manifest over NFS), and it
invalidates the sysimage — which fails by silently falling back to CPU while holding an H200.

## NEXT UP (in order)

1. **W11.a–c** two-stage AME bootstrap — implement, then run on est3/est4 (~2–5 h). *Blocks the
   final E3/E4 tables, so it comes before promotion.*
2. **W0.5b + W11.d** regenerate step 7 tables (note-text fix + new SEs), then **W0.8 promote**
   sandbox → production and re-run 5/7/8 there
3. **W1.6** delete `C:\egan_constrained` + `C:\egan_trimmed` (**unblocked** — W0.6 closed the
   vintage comparison), 40 GB
4. **W2** registries → **W3** orchestrator (the drift-killer work; everything after gets cheaper)
5. **W4.2–4.5** battery as an orchestrated stage + staleness guard; **W10 Tier 1** (joint-sieve text
   that reaches the paper)
6. **W5** upload/ingest tooling → cluster re-run → **W4.6–4.8**, then **W9** submit-script
   consolidation *(after the cluster run, not during)*

Deferred until the above lands: **W6** docs, **W7** `diag_`→`step_` rename, **W8** (now narrowed to
E1/E2 + battery).

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

## W11 — [!] E3/E4 report one test as five rows — **inferential, affects the headline table**

**Verified independently 2026-08-19 against the new-vintage pickles.** In `est1-4_spec12_stage2_comparison.tex`,
the E3 and E4 columns show 5 and 6 state-variable rows that read as separate significance tests. They are not.

| routine | rows | distinct \|t\| | shared value |
|---|---|---|---|
| est3 | 5 | **2** | \|t\| = 1.313121091 across cadunico, fraction_65plus, risk_free_qoq_lag, connections_per100 |
| est4 | 6 | **2** | \|t\| = 1.209007720 across those four plus gdp_growth_yoy |
| est2 (linear control) | 7 | 7 | each coefficient has its own t, as expected |

**Mechanism.** `fit_single_index` does not estimate the index direction θ — it inherits it from a
separate Cauchy NLLS-logit warm start and estimates only the link G. For a continuous regressor,
`AME_j = θ_j · (slope_w · β)`. The bootstrap perturbs **β only**, never θ, so the fixed scalar θ_j
cancels out of `t = AME_j / se(AME_j)`, leaving one common \|t\| with the sign carried by sign(θ_j).
The dummy (`pix_exists`) is a discrete difference `dcols_j · β` — a different functional of β, hence
the second distinct t. So each column carries **two** tests: "is the link flat" and "does Pix move it".

**Consequences.**
**DECIDED 2026-08-19 (user): full two-stage bootstrap of the AMEs.** Each draw perturbs the
direction θ via its cluster influence functions (`nlls_direction_if`) AND re-profiles the link
(sieve OLS + QP projection) at the perturbed index, then recomputes every AME — continuous rows as
θ*_j/sd_j × mean_slope(β*), the Pix dummy as its discrete difference. Every row then carries its own
t including direction uncertainty. Pattern already proven in `unconditional_phi_t_band`; measured
cost ≈ 1–2.3 h locally at B=999. Presentation-only and θ-only options rejected.

      driver that recomputes AME SEs from a STORED fit + parquet without re-estimating
- [ ] **W11.b** Keep both clustering schemes: conglomerate WCB for market-level rows, quarter WCB for
      the national rows (pix_exists, risk_free_qoq_lag) — same dual scheme as today
- [ ] **W11.c** Run post-hoc on sandbox est3/est4 (~2×1–2.3 h); **verify per-row t's are now distinct**
      and the old conditional t (1.313/1.209) is recovered when θ-perturbation is switched off
- [ ] **W11.d** Regenerate `est{3,4}_second_stage_table.tex` + `est1-4_spec12_stage2_comparison*.tex`
      (also picks up the W0.5b note-text fix); table note gains one sentence: SEs include
      direction-estimation uncertainty
- [ ] **W11.e** Consequence for **W8**: the two-stage bootstrap replaces the WCU/WCR question for the
      E3/E4 AME rows entirely (it is its own reference). W8.2 narrows to the E1/E2 linear columns and
      the diagnostics battery
- Note kept for the record: conditional on inherited θ, all continuous-AME nulls collapse to one
  restricted fit (φ ≡ 0.9757 flat, violation 7e-16) — this is WHY the two-stage design is required

## W0 — Recover the in-flight run (steps 5–8), verify, promote


  | routine | old | **new** | Δ (pp) |
  |---|---|---|---|
  | E1 local linear | 99.9669 | **99.9945** | +0.028 |
  | E2 pooled linear | 99.5041 | **99.5400** | +0.036 |
  | E3 single-index *(headline)* | 96.5120 | **96.5627** | +0.051 |
  | E4 single-index + Time | 96.6270 | **96.6814** | +0.054 |

  All four move < 0.06pp: **winsorization-at-source + the φ_t weight unification did not move the
  headline**. E3/E4 constrained link confirmed genuine (`link='index_sieve'`, `si_constrained=True`,
  7 active constraints, Σβ = 0.973). Trend "up" preserved on all four.
  - Cross-check already passing: `desc_3` reproduces the known cluster geometry exactly — G=456, G\*=6.17, CV=8.54, top-5 80.9%. Its two `[WARN]`s (1,142,280 panel rows vs 487,046 pkl nobs; 456 vs 453 clusters) are the documented panel-vs-estimation-sample distinction, not a defect

## W0b — V_Main sleepiness exhibits: audit of what today's run owes

Audit of all 41 `\input`/`\includegraphics` in V_Main.tex against disk + generators (2026-08-19).

**Produced by re-running steps 5–8 (blocked only on W0.5):**

**Needs the local Julia logit after demand parquets exist (W3.2 step 14):**
- [ ] `est3_spec12_logit.tex`, `est4_spec12_logit.tex`, `est1-4_spec12_logit_comparison.tex` ← `blp_1_logit.jl`

**Needs the cluster re-run:**
- [ ] `blp_rc_E3_spec12.tex`, `blp_rc_E4_spec12.tex` ← `make_blp_rc_table.py`
- [ ] `blp_demand_comparison_noseg_spec12.tex` (currently 08-12)

**[!] Must be REMOVED from V_Main — deleted lineup, no generator will ever emit these again:**
- [ ] `blp_rc_E7_spec12.tex`, `blp_rc_E8_spec12.tex` (joint sieve)
- [ ] `est7_spec12_logit.tex`, `est8_spec12_logit.tex` (joint sieve)
- [ ] `est_phi_t_joint_sieve_pair.png` (joint-sieve figure; `_make_pair_plot` now emits only the single-index pair)

**Valid and kept:** `est_phi_t_single_index_pair.png` (built from the genuine `ts_link_band_est{3,4}.pkl`).

## W1 — Stale disposition (~73 GB)

- **Total reclaimed 31.04 GB**; `ESTIMATION_OUTPUT` now 54.4 GB. Log: `_ARCHIVE_PRE_RELABEL_20260818/DISPOSITION_LOG.md`
- Correction from the sweep: **E7's psi_dev shard set is complete at 100/100** — the "missing" shard was `or.parquet`, a truncated-name copy of `psi_dev_E7_spec_12_extended_shard71of100.parquet` (verified by schema, 150 firms, shock 22)
- [ ] **W1.5** *(hold)* JI jsons stay until the battery re-runs under the new lineup
      series (244 KB total, est1/est2/est5–est8 from BOTH roots) were copied to
      `_ARCHIVE_PRE_RELABEL_20260818/vintage_comparators/` first, so the old-vs-new comparison stays
      reproducible at the reporting level without the multi-GB pickles

## W2 — Single-source registries (drift killer)

- [ ] **W2.1** `config/routines.toml` — one lineup definition for Python + Julia
- [ ] **W2.2** `utils/routines.py` + migrate 13 Python declaration sites
- [ ] **W2.3** `routines.jl` + migrate Julia sites *(ships with cluster batch)*
- [ ] **W2.4** `utils/paths.py` completion: BLP/BBL/CF/COST_FWD accessors, kill 14 Drafts literals
- [ ] **W2.5** `submit_*.sh` ROUTINES defaults read the manifest *(cluster batch)*

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

## W10 — Joint-sieve residue: code, live branches, and generated table text

The joint sieve left the *lineup*, but its implementation and several user-visible strings remain.
Three tiers, in priority order.

**Tier 1 — reaches the paper (generated output, not comments):**
      `tab_alpha_weakiv_sleep.tex` / `tab_phi_need_sleep.tex`
      the E4/E6/E8 time block)`; the live time block is E4 alone

**Tier 2 — live code, needs a decision not an edit:**
- [ ] `estimation_sleep_common.py:13` vs `estimation_2_sleep.py:133` — docstring says the +Time block
      adds `(time_trend, gdp_growth_yoy)`, but `TIME_VARS = ['gdp_growth_yoy']`; `time_trend` was
      dropped. Pre-dates the relineup — decide whether the doc or the code is wrong
- [ ] `run_diag_matrix.py:25,133` — `--units d0:1,2,5,6,7,8` in help; the diag-matrix unit ids are
      their own namespace and must be reconciled with `run_phi_diagnostics.py` first (which also has
      the duplicate `D4b` key bug, W3.2)

**Tier 3 — the estimator implementation itself:**
- [ ] `fit_joint_single_index` and the `"sieve"` link path still exist in `utils/sleep_links.py`
      (~18 sites incl. `:413-431`, `:1025-1027`, `:1464`, `:2378-2394`), plus references in
      `estimation_sleep_common.py:61,64,77,234`, `diag_index_identification.py`,
      `estimation_demand_link_common.py:270,533`, `export_sleep_link_common.py:118,282`.
      **Decide: delete the joint-sieve estimator, or keep it as an unexported capability?** Comments
      describing it are correct as long as the code is there, so this is one decision, not a sweep.
      Note the parallel precedent: the logit *link* code is deliberately kept because E3/E4's
      direction step is an internal logit fit

**Deliberately keep:** `utils/se_national.py:249` — a dated (2026-07-30) adversarial-review finding
about which stored pickles lacked quarter blocks *at that date*; naming E8 is part of that record.

## W9 — SUPERSEDED 2026-08-20 (was: consolidate 23 submit_*.sh)

A single command now exists — `pipeline_all.sh` plus `sleep_run.sh` / `blp_run.sh` / `bbl_run.sh` /
`cf_run.sh` / `cf_eq_run.sh` on `cluster_lib.sh`, with gates and self-continuing phases. The 23 legacy
`submit_*.sh` were deliberately left in place as a fallback. Retiring them is CLEANUP after one
successful cluster cycle, not a prerequisite, and is not worth doing before the new path is proven.

- [ ] *(after one green cluster cycle)* delete the superseded `submit_*.sh`; keep `setup_julia_env.sh`

## W8 — LARGELY MOOTED 2026-08-20 by the two-stage bootstrap (W11)

The WCU-vs-WCR question was about how to get honest SEs for the single-index AMEs. W11's two-stage
bootstrap replaces that question for E3/E4 entirely: it resamples the direction AND re-profiles the
link per draw, and reports percentile/BC intervals rather than a symmetric SE with a normal reference.
The size measurement stands (normal 0.270 · WCU 0.138 · WCR 0.083 · CRVE-t 0.145 at nominal 5%; MC SE
0.0109, so the WCR-over-WCU gap is real but WCR is itself still oversized).

What genuinely remains, narrowed to the LINEAR routines and the battery:
- [ ] **W8.2** Choose WCU vs WCR for E1/E2 and the weak-IV battery — **user decision**, no longer
      blocking the E3/E4 headline
- [ ] **W8.4** Make the silent fallback loud: `utils/sleep_links.py:1873-1879` coerces `wcr`→`wcu` on
      the generic score path and warns ONCE PER PROCESS via `_WCB_WARNED`. Record the realised mode in
      each results object so a table can state its own reference distribution

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