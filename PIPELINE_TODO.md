# Pipeline rework — live status

Plan: `C:\Users\pedro\.claude\plans\ok-let-s-do-the-reflective-church.md`
Started 2026-08-19. Legend: `[ ]` not started · `[~]` in flight · `[x]` done · `[!]` blocked/needs you

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

- [ ] **W11.a** Implement in `fit_single_index` (so every future run does it natively) + a post-hoc
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

- [x] Estimators E1–E4 into sandbox `C:\egan_relineup_20260819` (E1 735s, E2 803s, E3 2901s, E4 4724s)
- [x] **W0.1** `utils/paths.py`: `estimation_output()`, `est_dir(n)`, `rout_dir()`, `drafts_dir()` (the Drafts path is *derived*, verified byte-identical to the literal it replaces)
- [x] **W0.2** Migrated 9 step-5–8 scripts off hardcoded production paths; `est1-3_spec12_all_models.pkl` → `est1-4_…` (no readers found); φ_t comparison figures → `est1-4_spec12_phi_t_comparison[_ci_pastel].png`
- [x] **W0.3** Silent-success failures now hard-fail: zero demand parquets → exit 1 naming the tree + `SLEEP_OUT_ROOT`; missing pickle → exit 1; `export_analyze_spec12.py` refuses to write an empty pickle over a real one (the exact 5-byte symptom)
- [x] **W0.4** Invalid 14:11 production Rout outputs deleted *(via W1.1)*
- [ ] **W0.3b** Two path inconsistencies found but out of W0 scope: `estimation_1_sleep.py:340` hand-builds the production `Rout` (step 1 writes its Drafts copy to production even under a sandbox); `make_phi_t_band_table.py:73` reads `demand_prep_root()/Rout` — a *different* production dir than `rout_dir()`, consistent only when sandboxed. Fold into W2.4
- [x] **W0.5** Re-ran steps 5–8 into the sandbox — **exit 0, 1619s** (exports 90s · demand prep 1029s · analyze 416s · desc_3 85s). 40 demand parquets; `est1-4_spec12_all_models.pkl` **1.18 GB** (was 5 bytes when broken); comparison tables carry real coefficients, the only `-` cells being structural (Constant is linear-only; GDP Growth is Time-block-only, so E4 alone)
- [ ] **W0.5b** Re-run step 7 before promotion: table-notes wording fixed (dropped the deleted joint sieve from "single-index/joint strategies", 3 sites) — the current tables still carry the old note text
- [x] **W0.6** Vintage verified — `compare_vintage.py 1 2 3 4`, national φ̂ at spec 12:

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
- [ ] **W0.7** Record the E4 old-vs-new comparison → releases the 40 GB sandboxes
- [ ] **W0.8** Promote sandbox → production by explicit copy, re-run 5/7/8 against production

## W0b — V_Main sleepiness exhibits: audit of what today's run owes

Audit of all 41 `\input`/`\includegraphics` in V_Main.tex against disk + generators (2026-08-19).

**Produced by re-running steps 5–8 (blocked only on W0.5):**
- [ ] `est2_first_stage_table.tex` ← `export_2_sleep_results.py`
- [ ] `est3_first_stage_table.tex`, `est3_second_stage_table.tex` ← `export_sleep_link_common.py --est 3`
- [ ] `est4_first_stage_table.tex`, `est4_second_stage_table.tex` ← `export_sleep_link_common.py --est 4`
- [ ] `est1-4_spec12_stage{1,2}_comparison[_landscape].tex` ← `export_analyze_spec12.py` *(the 14:11 copies were dataless — every cell `-` — and have been deleted from Rout and Drafts)*
- [ ] `est1-4_spec12_phi_t_comparison[_ci_pastel].png` ← same *(14:11 copies carried only E3/E4, no E1/E2 series; deleted)*
- [ ] `cluster_imbalance_panelB.tex` ← `desc_3.py` (currently 08-05)
- [ ] `est1_first_stage_table.tex`, `est1_second_stage_table.tex`, `est2_second_stage_table.tex` (currently 08-12)

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

- [x] **W1.1** Production `Rout` pre-regeneration deletes — 99 items, 528.6 MB
- [x] **W1.2** Archive cluster GPU outputs → `_ARCHIVE_PRE_RELABEL_20260818/cluster_vintage/` — 536 items, 1.32 GB
- [x] **W1.3** Delete locally-regenerable derivations — 128 items, 580.3 MB
- [x] **W1.4** Delete big snapshots (`_PRECENTER`, `_PRE_LS*`, `failed_runs`, BLP_DRAWS fragments) — 30.67 GB
- [x] **W1.4b** Rout stale-lineup leftovers (joint-sieve pair fig, 8 `est_timeseries_phi_*.png`, April `estimation_{1..5}_results.tex`) — 14 files, 0.69 MB, guarded against today's outputs
- **Total reclaimed 31.04 GB**; `ESTIMATION_OUTPUT` now 54.4 GB. Log: `_ARCHIVE_PRE_RELABEL_20260818/DISPOSITION_LOG.md`
- Correction from the sweep: **E7's psi_dev shard set is complete at 100/100** — the "missing" shard was `or.parquet`, a truncated-name copy of `psi_dev_E7_spec_12_extended_shard71of100.parquet` (verified by schema, 150 firms, shock 22)
- [ ] **W1.5** *(hold)* JI jsons stay until the battery re-runs under the new lineup
- [x] **W1.6** Comparator sandboxes retired (2026-08-19, ~40 GB). The 12 reported `national_phi_t.csv`
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

- [x] **W4.1** Relabelled battery stragglers to E1–E4 (13 files); deleted dead `diag_uncond_band_joint.py`. Found `make_diag_tables.py` raising `KeyError` on **every** run (label map 1–4 vs estimator list 5–8)
- [x] **W4.1b** Straggler batch: **`export_rc_delta.jl` was writing nothing** — its `--routines` default of `5,6,7,8` meant every run printed `MISSING — skipped`, so the JI/CUE diagnostics had no `rc_delta_*.bin` input at all. Now `3,4`, and switched onto the validated `of_root.jl` resolver (no `--hpc` shape exists for that script, so the switch is safe and a wrong root now fails loudly). Plus ~30 comment relabels — two of which were *actively inverted*: `estimation_sleep_common.py:83,141,143` named E5/E6 from inside the live single-index branch, and `export_analyze_spec12.py:56` read as if live E3 were the pooled logit
- [ ] **W4.2** Battery becomes an orchestrated stage (sleep-side + demand-side)
- [ ] **W4.3** Staleness guard: `input_fingerprint` on every battery json
- [ ] **W4.4** `make_iv_tables.py` hard-fails instead of emitting empty tables
- [ ] **W4.5** Ingest hook: `process_blp_outputs.py` auto-runs export_rc_delta → battery → tables
- [ ] **W4.6** *(cluster batch)* `se_common.jl` emits per-cluster moment blocks (spread_hat, A_g, B_g)
- [ ] **W4.7** *(cluster batch)* Battery fast path reads blocks; cross-check vs parquet path
- [ ] **W4.8** *(after cluster re-run)* Regenerate iv tables + update `weakiv_report.tex` prose

## W5 — Upload / ingest tooling

- [ ] **W5.1** `cluster/upload_manifest.txt`
- [ ] **W5.2** `stage_cluster_upload.py` (staleness-checked staging + sha256 + LF rewrite)
- [ ] **W5.3** `process_cluster_outputs.py --kind bbl|cf` (the missing ingest symmetry)
- [ ] **W5.4** Cluster preflight blocks read the manifest *(cluster batch)*

## W10 — Joint-sieve residue: code, live branches, and generated table text

The joint sieve left the *lineup*, but its implementation and several user-visible strings remain.
Three tiers, in priority order.

**Tier 1 — reaches the paper (generated output, not comments):**
- [ ] `make_iv_sleep_tables.py:164,179` — LaTeX strings naming the "E7/E8 sieve link" are emitted into
      `tab_alpha_weakiv_sleep.tex` / `tab_phi_need_sleep.tex`
- [ ] `desc_2.py:1640,1642` — appendix caption text `Sleep'' = sleepiness estimation (E1--E8; …
      the E4/E6/E8 time block)`; the live time block is E4 alone
- [ ] `make_blp_rc_table.py:16,521` — `--all` help text says E1-E8

**Tier 2 — live code, needs a decision not an edit:**
- [ ] `export_selic_wakeup.py:133` — `if "E7" in set(sig["est"])` is a **dead branch**, not a comment
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

## W9 — Consolidate the SLURM submission layer (23 shell scripts)

**Assessment: yes, but as TWO layers, not one big script — and sequenced around the cluster re-run.**

Current inventory (23 `.sh`): orchestrators that call `sbatch` (`submit_bbl_all.sh` 298 L,
`submit_bbl_cf_all.sh` 207 L, `submit_cf_all.sh`, `submit_blp_rc_all.sh`, `submit_blp_rc_grouped.sh`,
`submit_blp_build_and_run_spec12.sh`, `submit_cf3_jacobi.sh`, `submit_cf5_all.sh`,
`submit_cf5_passthrough.sh`, `submit_cf6_merger.sh`); job scripts carrying `#SBATCH` headers
(`submit_bbl.sh`, `submit_blp_1_draws.sh`, `submit_blp_rc_stage.sh`, `submit_cf.sh`,
`submit_build_sysimage.sh`, `submit_build_sysimage_cpu.sh`); thin wrappers that only set defaults
(`submit_bbl_default.sh` 23 L, `submit_blp_2_rc_default.sh`); archivers (`zip_cf_outputs.sh`,
`zip_all_cf.sh`); and one-time env (`setup_julia_env.sh`).

**Why not literally one script.** A `#SBATCH` header only takes effect in the file handed to `sbatch`,
so job scripts and orchestrators are different kinds of object. The workable equivalent is to pass
resources as `sbatch` CLI flags (which override headers) and dispatch work through a generic job
script. The repo is already half-way there: `submit_blp_rc_stage.sh` receives `RC_ROUTINE/RC_ENGINE/
RC_STAGE` via `--export=ALL`, and `submit_cf.sh` dispatches on `CF_STEP`.

**Target: ~23 → ~5 files.**
- [ ] **W9.1** One orchestrator `submit.sh <stage> [--routines …] [--spec …] [--dry-run]` covering
      draws / rc / bbl / cf / sysimage / archive; owns dependency chains, preflight (reads the W5
      `upload_manifest.txt`), and routine sets (from `config/routines.toml`, W2.1)
- [ ] **W9.2** Three generic job scripts by resource profile — `job_gpu.sh`, `job_cpu_turin.sh`,
      `job_day.sh` — receiving the command via `--export`, replacing the per-task job scripts
- [ ] **W9.3** One archiver with an explicit `--move`/`--copy` flag, replacing the three different
      zip conventions (`zip -jm` moves in BLP, `zip -j` copies in BBL, `zip`+verify+conditional `rm`
      in `zip_cf_outputs.sh`). **Highest-risk item**: divergent zip semantics already destroyed
      uploaded inputs once (2026-08-01, a `cf4` glob in move-mode emptied `CF_FOUNDATION`)
- [ ] **W9.4** Delete the thin default-wrappers; their values become `submit.sh` flags

**Duplication this removes:** `ROUTINES` defaults in 6 mutually-inconsistent copies (four still say
`1 2 5 6 7 8`, two say `1 2 3 4`, `blp_2_rc.jl` says `[3,4]`); the `pick_zip()` + `CP_DIR` block
duplicated byte-for-byte between `submit_cf_all.sh:60-113` and `submit_bbl_all.sh:98-147` under a
three-way "keep in sync" comment pointing at `foundation_demand_eval.jl:157`; the staging list
triplicated as prose in three preflight blocks; two sysimage wrappers differing only by partition,
constraint and `BLP_SYSIMAGE_CPU=1`; and two front doors for the same RC sweep with different
`--mem` policies (only one has `MEM_BIG` for E1/E2, which OOM'd at 200 G).

**Two constraints that shape the plan.**
1. **Unverifiable locally** — there is no SLURM here, so a rewritten submission layer cannot be
   tested until it runs on Bouchet. Argues against a big-bang rewrite: keep the current scripts in
   place until one full consolidated cycle has succeeded, then delete.
2. **LF endings** — these files are edited on Windows and copied over; `.gitattributes` pins
   `*.sh text eol=lf` because every submit script once failed at line 1 after upload. A wholesale
   rewrite is maximum exposure to that bug class; verify CRLF=0 on every file.

**Sequencing: do W9 either clearly BEFORE the cluster re-run (with a small smoke job to validate) or
clearly AFTER it — not during.** The re-run is the first consumer of the new demand parquets and
should not be debugging a new submission layer at the same time. Recommend AFTER, with W9.1 drafted
beforehand so the re-run itself exercises the old path one last time.

## W8 — Decide the bootstrap reference distribution (WCU vs WCR) — **blocks the final tables**

Sequenced after the new vintage's bands + diagnostics exist (needs re-measured size numbers on this
vintage, not the old ones). Decision is required before the paper's inference is final.

**The constraint that decides it.** `utils/sleep_links.py:1873-1879`: requesting `SLEEP_WCB_MODE=wcr`
on the generic score path (the nonlinear AME routines) cannot impose H₀ without a restricted refit,
so it coerces `mode = "wcu"`. The notice is guarded by `_WCB_WARNED` → **prints once per process**,
then silent. So "promote WCR" delivers true WCR for E1/E2 and WCU for E3/E4.

**Why the relineup made this sharper.** The nonlinear routines are now **E3/E4** — including E3, the
paper's preferred spec — and the headline exhibit `est1-4_spec12_stage2_comparison.tex` puts E1, E2,
E3, E4 in *one table*. Under WCR that single table would carry two different reference distributions
across its columns. Previously the nonlinear block was E5–E8 and sat in its own exhibit.

**Evidence pulls both ways, and both facts are true.** Synthetic size at nominal 5%: WCR 8.3%, WCU
13.8%, normal reference 27%. But on the D4b coefficient the bootstrap 95% critical value is 4.41
under WCU vs 2.38 under WCR (p = 0.150 vs 0.021) — WCU is *far more conservative there*, because OLS
residuals are orthogonal to X by construction and one conglomerate carries >25% of rows, so the
dominant cluster's unrestricted residuals are shrunk and the studentised draws go fat-tailed.
Averages over a synthetic DGP do not determine behaviour at one realised design point.

- [x] **W8.1** Size re-measured 2026-08-19 (`--only size --reps 400`, 460s). Geometry realised
      G=456, CV=8.540, G\*=6.17 — matching `desc_3` on the new vintage exactly. Rejection at nominal 5%:
      **normal 0.270 · WCU 0.138 · WCR 0.083 · CRVE-t(G\*) 0.145**. Replicates the archived header
      values (27.0/13.8/8.3/14.5) essentially exactly. Monte Carlo SE at 5% ≈ 0.0109, so 2 SE = 0.0218:
      - the WCR-over-WCU gap (0.055) is **2.5× the noise threshold — a real difference, not sampling error**
      - but WCR at 0.083 is itself still oversized vs 0.05 by 1.5× that threshold, so **no scheme is
        correctly sized at this geometry**; the choice is between degrees of over-rejection
- [ ] **W8.2** Choose the option (see below) — **user decision**
- [ ] **W8.3** Implement + regenerate every inference-bearing table; state the choice in the methods text
- [ ] **W8.4** Make the fallback non-silent: per-routine notice, and record the realised mode in each
      results object so a table can print its own reference (kills the mixed-column hazard by construction)

**Options.** (1) WCR + disclose the mixed reference — now hits the *main* table. (2) WCU everywhere —
uniform, available for every estimator, 2× better-sized than normal; loses the D4b rejection, which is
arguably the more conservative claim. (3) Keep the normal reference at 27% — not defensible.
(4) **New, opened up by deleting the joint sieve**: implement the restricted refit for the
single-index path, spec 12 only, reported coefficients only. The cost objection was written when the
sieve routines needed a full re-optimisation per draw; E3/E4 are an NLLS-logit direction step plus a
monotone I-spline QP, materially cheaper. Rough order: ~199 refits × coefficients × 2 routines —
plausible as a cluster job, painful locally. Would need timing on one coefficient before committing.

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
- [ ] **V_Main.tex** renamed figure includes (generators now emit these): `est1-3_spec12_phi_t_comparison_ci_pastel.png` → **`est1-4_…`**, and the plain `est1-3_spec12_phi_t_comparison.png` → **`est1-4_…`**. Table `\input`s already moved to `est1-4_spec12_stage{1,2}_comparison[_landscape].tex`
- [ ] Decision on commit `d105f3bd` (made before the no-commit rule; offer stands to soft-reset)
- [ ] Delete `estimation_timeseries_test.py`
- [ ] **Security**: repo-root `.env` holds a plaintext `ANTHROPIC_API_KEY` — confirm untracked, rotate if ever shared
- [ ] Cluster re-run on Bouchet (upload staged by W5, then BLP RC → BBL → CF)
