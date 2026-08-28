# Pipeline rework — live status

Plan: `C:\Users\pedro\.claude\plans\ok-let-s-do-the-reflective-church.md`
Started 2026-08-19. Legend: `[ ]` not started · `[~]` in flight · `[x]` done · `[!]` blocked/needs you


> This file SHRINKS as work closes: finished sections are deleted, not ticked.

## NEXT UP

1. **W11.d** regenerate the E3/E4 tables from the downloaded AME pickles
2. **Phase 8B** repo-wide stage+role rename (after one green cluster cycle), then **W3** orchestrator consolidation
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

**Ours today**: `scrape_forward_rf.py` builds ONE deterministic path from live BCB data (SGS 4189 anchor +
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
| `scrape_forward_rf.py` | **YES, permanently** | hits BCB SGS 4189 + Focus/Olinda; compute nodes have no outbound internet |
| `panel_demographics_sigma.py` | no | pure pandas over `market_panel.csv`, which the run uploads anyway |
| `bbl_polfunc.py` | no | same; a cluster path already exists (`bbl_run.sh --polfunc`, off by default) |

Moving the latter two into the chain would cut ~30 min of local compute per run and ~68 MB from
the data bundle (`polfunc_fitted.csv` 59 MB + `demographics_sigma.parquet` 8.7 MB), and make them
gated cluster steps like everything else. They are uploaded inputs today only because that
classification predates the sleepiness phase moving to the cluster.

- [ ] **W12.1** Enable `bbl_run.sh --polfunc` in the chain; reclassify its manifest row `produced`
- [ ] **W12.2** Add a small `panel_8` cluster job (or a `SLEEP_STEP=demog` branch); reclassify likewise
- **Deferred by decision 2026-08-20**: ship the first cycle as-is; revisit once one full cluster run
  has succeeded end-to-end. Do not do this while the chain is still unproven.

**Why `scrape_forward_rf.py` can never move** (worth recording — it is not just an API convenience):
in the BBL value basis ψ4 = Σ β^t r^f_t · Σ_k Dep_t. A FLAT r^f_t collapses ψ4 to a rescaling of
ψ2, making ω and ζ collinear and leaving **ζ (funding-cost pass-through) unidentified**. The live
Focus curve exists to break that collinearity. Its CSV carries a `source` column so an offline
fallback path can never be mistaken for the live curve.

## W11 - [!] E3/E4 AME tables still to regenerate

- [ ] **W11.d** Regenerate `est{3,4}_second_stage_table.tex` + `est1-4_spec12_stage2_comparison*.tex`
      from the downloaded `ame_twostage_est{3,4}_robust*.pkl`; the table note gains one sentence:
      SEs include direction-estimation uncertainty

## W0b — V_Main sleepiness exhibits: audit of what today's run owes

**Needs the local Julia logit after demand parquets exist (W3.2 step 14):**
- [ ] `est3_spec12_logit.tex`, `est4_spec12_logit.tex`, `est1-4_spec12_logit_comparison.tex` ← `blp_logit.jl`

**Needs the cluster re-run:**
- [ ] `blp_rc_E3_spec12.tex`, `blp_rc_E4_spec12.tex` ← `make_blp_rc_table.py`
- [ ] `blp_demand_comparison_noseg_spec12.tex` (currently 08-12)

## W1 — Stale disposition (~73 GB)

- [ ] **W1.5** *(hold)* JI jsons stay until the battery re-runs under the new lineup

## W2 - Single-source registries

- [ ] One duplication left, MEASURED to agree on all 12 specs (2026-08-24): `SPEC_MAP` in
      `sleep_demand_prep_link.py` vs the `_IV_ORDER` x state-block enumeration in
      `sleep_est_single.py:143`. Consolidating touches the estimator's own spec
      enumeration, so not next to a live cluster run

## W3 — Orchestrator consolidation

- [ ] **W3.1** Extract `utils/pipeline_runner.py` (waves, resume predicates, RAM gate, env hardening, RUN_MANIFEST.json)
- [ ] **W3.2** Rebuild `sleep_pipeline.py` on it — 15 steps through bands, diagnostics, battery, tables
- [ ] **W3.3** Wire `panel_audit_mca_coverage.py` into a data-pipeline gate — today it is referenced by no orchestrator, so a failed MCA merge reaches estimation as median-filled coverage

## W4 — Weak-IV integration into the BLP pipeline (hybrid)

- [ ] **W4.2** Battery becomes an orchestrated stage (sleep-side + demand-side)
- [ ] **W4.3** Staleness guard: `input_fingerprint` on every battery json
- [ ] **W4.4** `make_iv_tables.py` hard-fails instead of emitting empty tables
- [ ] **W4.5** Ingest hook: `cluster_ingest_blp.py` auto-runs export_rc_delta → battery → tables
- [ ] **W4.6** *(cluster batch)* `blp_se_common.jl` emits per-cluster moment blocks (spread_hat, A_g, B_g)
- [ ] **W4.7** *(cluster batch)* Battery fast path reads blocks; cross-check vs parquet path
- [ ] **W4.8** *(after cluster re-run)* Regenerate iv tables + update `weakiv_report.tex` prose

## W5 — Upload / ingest tooling

- [ ] **W5.4** Cluster preflight blocks read the manifest *(cluster batch)*

## W10 - Residue from the joint-sieve removal

- [ ] `tab_phi_need_sleep.tex` still holds joint-sieve-vintage numbers. Its sieve arm now fits with
      `fit_single_index`, so a re-run of `sleep_weak_iv.py` + `make_iv_sleep_tables.py`
      will move the numbers. Not `\input` into V_Main, so nothing in the paper moves today
**Deliberately keep:** `utils/se_national.py:249` - a dated (2026-07-30) adversarial-review finding
about which stored pickles lacked quarter blocks *at that date*; naming E8 is part of that record.

## W8 - WCU vs WCR, narrowed to the linear routines

W11's two-stage bootstrap is its own reference for the E3/E4 AME rows, so this is now only about
E1/E2 and the battery. Measured sizes at nominal 5%: normal 0.270 / WCU 0.138 / WCR 0.083 / CRVE-t
0.145 (MC SE 0.0109), so WCR beats WCU but is still oversized.

- [ ] **W8.2** Choose WCU vs WCR for E1/E2 and the weak-IV battery - **user decision**
- [ ] **W8.4** `utils/sleep_links.py` coerces `wcr`->`wcu` on the generic score path and warns once
      per process via `_WCB_WARNED`. Record the realised mode on each results object so a table can
      state its own reference distribution

## Naming constraints (bind the Phase 8B stage+role rename)

- **Output artifact names do NOT follow script names.** `diag_moment_reduction.json`,
  `diag_cue_linear_step.json`, `.diag_matrix_state.json`, and the dirs `DIAG_PHI_SEPARATION/`
  and `DIAG_WEAK_IV_SLEEP/` keep their spelling — they are data contracts read by the table
  generators and present in archives and cluster outputs.
- **Stays `diag_`** (ad-hoc, not pipeline steps): `diag_im_phi_t.py`,
  `diag_index_identification.py`, `diag_k4_spread_local_variation.py`,
  `diag_openfinance_coverage.py`, `diag_test_inversion_band.py`, `diag_wcb_reference.py`.

## W6 — Docs

- [ ] **W6.1** README pipeline sections
- [ ] **W6.2** `weakiv_methods.md` §8 reproduce block
- [ ] **W6.3** Memory updates + disposition log

---

## Needs you (not mine to do)

- [ ] **V_Main.tex** dangling refs: L462 `\ref{estimation:single_index}` → `single_idx` (preferred-spec sentence); L695 `single_idx_timee`; L691 `tab:blp_rc_est4spec12`; L606 `joint_sieve` ×2 (needs rewrite); L462 prose "all six approaches" → four
- [ ] Decision on commit `d105f3bd` (made before the no-commit rule; offer stands to soft-reset)
- [ ] **Security**: the leaked `ANTHROPIC_API_KEY` is revoked and `.env` is untracked + gitignored (2026-08-27). Remaining call: whether to purge the key from git history — commit `f635883d` is on `origin/master` and `origin/coherence_fix`, so a rewrite touches both
- [ ] Cluster re-run on Bouchet (upload staged by W5, then BLP RC → BBL → CF)
