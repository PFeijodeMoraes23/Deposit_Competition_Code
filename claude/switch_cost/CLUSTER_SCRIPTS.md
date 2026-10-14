# Switch-cost cluster scripts

Written 2026-10-13 (18:55). Three shell copies, made at 17:47 from the working tree:

| Source | Copy | Source md5 (before and after: unchanged) |
|---|---|---|
| `cluster_lib.sh` | `switch_cost_cluster_lib.sh` | `8bcd02f43cf3337c040fe0db3ba807a2` |
| `sleep_job.sh` | `switch_cost_sleep_job.sh` | `823267f9962a010a8df54dbb131b959e` |
| `sleep_run.sh` | `switch_cost_sleep_run.sh` | `bab5ac543908edda6a152e5c122de75a` |

All three are LF, like their sources (0 CR bytes), and pass `bash -n`. Nothing was submitted, no
cluster was contacted, no original, Python, Julia or TeX file was edited.

**Scope** (the coordinator's update of 2026-10-13, a user decision): E3 and E4, spec 12 only,
both fit variants (`pooled` and `typeshift`), with the two-stage AME bootstrap. The demand prep is
optional on the cluster (off by default). No E1/E2, no spec grid.

## 1. What each copy does

**`switch_cost_sleep_run.sh`** (login node; only parses flags, runs a read-only preflight, calls
`sbatch`). For each fit variant and routine it submits one estimation job, then one gate G1 per
variant. With `--prep` it also submits one demand-prep job per arm and routine, then one gate G2
per arm:

```
switch_cost_sleep_est_pooled_E3, _E4        -> switch_cost_sleep_gate_G1_pooled
switch_cost_sleep_est_typeshift_E3, _E4     -> switch_cost_sleep_gate_G1_typeshift
   (--prep) switch_cost_sleep_prep_pooled_E{3,4}   afterok G1_pooled             -> G2_pooled
            switch_cost_sleep_prep_F1_E{3,4}       afterok G1_pooled+G1_typeshift -> G2_F1
            switch_cost_sleep_prep_F3hi_E{3,4}     (same)                         -> G2_F3hi
            switch_cost_sleep_prep_F3lo_E{3,4}     (same)                         -> G2_F3lo
```

Flags: `--routines "3 4"`, `--variants "pooled typeshift"`, `--boot-b N` (SLEEP_BOOT_B, default
999, the Python default), `--ame-workers N` (SLEEP_AME_BOOT_JOBS, default 64, the original AME
job's value), `--blas-threads N` (default 2, the original job's value), `--prep`, `--arms`,
`--skip-est`, `--after`, `--skip-preflight`, `--dry-run`, `-h`. The original's flags for the
removed branches are refused with a message (`--spec-array`, `--no-spec-array`, `--spec-ids`,
`--no-ame`, `--no-wcb`, `--no-weakiv`, `--no-upsilon`, `--no-entry`). An F arm with
`--variants pooled` (its typeshift fit not estimated in the same submission) is refused.

Resources (defaults, each env-overridable): est 10:00:00 / 64 cpus / 256G (the original spec
task's 4 h plus the original AME job's 6 h, at the AME job's cpus and memory, since the bootstrap
now runs inside the estimation pass); prep 06:00:00 / 4 cpus / 64G (a quarter of the original
four-routine prep job); gates 00:10:00 / 2 / 8G. None of these is measured.

**`switch_cost_sleep_job.sh`** (the worker; dispatches on `SLEEP_STEP`):

- `est`: `switch_cost_sleep_est_single.py --est K --spec12` with `SLEEP_AME_TWOSTAGE=1`. Writes
  `est{K}/estimation_results.pkl`, `est{K}/market_panel_phis.csv`, `est{K}/national_phi_t.csv`
  and `Rout/ts_link_band_est{K}.pkl` under `data/output/switch_cost_sleep_<variant>`. `--spec12`
  is used rather than `--spec-id 12` plus `--merge-specs`, because the merge expects the whole 5-12
  grid and refuses a partial one without `--allow-partial`.
- `prep`: `switch_cost_sleep_demand_prep_link.py --est K --spec 12 --arm ARM --fits-dir
  <variant folder of the arm> --pooled-fits-dir <pooled folder> [--locked-csv
  SC_DATA_IN/r4_locked_bank_year.csv for F3hi/F3lo] --out-dir data/output/switch_cost_demand_<arm>`.
  The demand-prep orchestrator has no switch-cost copy. The job fails when no
  `demand_K_*spec_12.parquet` newer than its start appears in the out dir, because the prep exits
  0 without writing when a pickle or a spec is missing.
- `gate`: G1 (per variant) or G2 (per arm), written to `data/output/.gate_switch_cost_G1_<variant>.json`
  / `.gate_switch_cost_G2_<arm>.json` before the gate exits. G1 keeps the original checks (phi CSV
  over 1 MB, the pickle loads, the spec-12 cell and its second stage exist) and adds one: with
  `SLEEP_AME_TWOSTAGE=1`, the two-stage AME must be stored on the fit (`ame_boot`, `ame_2s_meta`)
  and pass the counters of the main project's G3. This matters because `_attach_twostage_ame`
  prints `[2s-AME] SKIPPED` and carries on when the bootstrap raises. The phi-hat comparison is
  report-only: against the production values for `pooled`, reported only for `typeshift`. G2
  keeps the original check: one parquet per routine, carrying the required columns.

**`switch_cost_cluster_lib.sh`** (the sourced library): the original, with the switch-cost folder
names, the variant/arm helpers (`sc_sleep_dir`, `sc_demand_dir`, `sc_arm_variant`), its own
skeleton, a submission guard in `cl_sbatch`, a read-only preflight (`sc_preflight`), and refusals
in every function that would touch a shared resource or a phase outside the scope.

## 2. What the copies write on the cluster

| Path (under the project root) | Written by |
|---|---|
| `data/output/switch_cost_sleep_pooled/` and `.../switch_cost_sleep_typeshift/` (`est3/`, `est4/`, `Rout/`) | est jobs |
| `data/output/switch_cost_demand_{pooled,F1,F3hi,F3lo}/` (`demand_{3,4}_*_spec_12.parquet`, `*.chars_audit.json`, `demand_prep_summary_{3,4}.json`) | prep jobs (with `--prep`) |
| `data/output/.gate_switch_cost_G1_<variant>.json`, `.gate_switch_cost_G2_<arm>.json` | gate jobs |
| `data/output/switch_cost_estimation_output/` (created empty; the target of the skeleton's `ESTIMATION_OUTPUT` link; nothing in the kept steps is expected to write there) | every job (mkdir) |
| `data/switch_cost_root/Open-Finance/` (skeleton: `shared/`, `Drafts/Deposit Competition/` empty, `BCB/Inclusion/`, `BCB/Egan_et_al_2025_Rep/processed/PANEL_INTERMED/`, and symlinks) | every job |
| `scripts/switch_cost_logs/` (`switch_cost_<job>_<jobid>.out/.err`, `switch_cost_orchestrator_*.log`) | Slurm, run script |

Names defined but never created (only refused functions use them): `data/output/switch_cost_logit`,
`switch_cost_blp`, `switch_cost_blp/draws`, `switch_cost_bbl`, `switch_cost_counterfactuals`,
`switch_cost_download`. `CF_FOUNDATION_DIR`, `CF_COST_FWD` and `COST_POLFUNC_DIR` are exported to
these, so a stray write could only land in a switch-cost folder.

## 3. Isolation edits (file:line in the copies)

### `switch_cost_cluster_lib.sh` (1855 lines)

| Line(s) | Edit |
|---|---|
| 2-3 | header lines |
| 4-35 | header rewritten for the copy (what differs, how to source it) |
| 49-86 | step tree: `CL_STEP_SLEEP_POOLED`/`_TYPESHIFT` (61-62) replace `CL_STEP_SLEEP`; demand arms by `sc_demand_dir`; `CL_STEP_LOGIT/BLP/DRAWS/BBL/CF/DOWNLOAD` renamed to `switch_cost_*` (67-72); `SC_ESTIMATION_OUTPUT` (79), `SC_DATA_IN` (82), `SC_GATE_DIR` (84), `SC_LOG_DIR` (86) |
| 92-117 | `_sc_refuse`, `_sc_tagged`, `sc_sleep_dir`, `sc_demand_dir`, `sc_arm_variant` |
| 132-158 | `cl_export_step_dirs`: requires `SC_VARIANT`; exports `SLEEP_OUT_ROOT` + `SWITCH_COST_TYPE_SHIFTS` together (138); `DEMAND_PREP_DIR` from `SC_ARM`, else unset (143); `STATE_CENTERING_JSON` and `DEMAND_CHARS_FILE` from `SC_DATA_IN` (148-149); refuses a destination without `switch_cost`; creates only the job's own folders |
| 203 | `cl_load_julia` refuses (shared `.julia_depot`) |
| 334 | `cl_sysimage_build_cmd` prints no rebuild command |
| 423, 434, 441, 498 | `cl_sysimage_missing`, `cl_draws_missing`, `cl_require_sysimage`, `cl_gpu_gate` refuse |
| 574-685 | `cl_need_draws`, `cl_need_rf_curve`, `cl_need_rf_vintages`, `cl_need_transitions`, `cl_need_sleep_link`, `cl_need_rc_jls`, `cl_need_costs`, `cl_need_polfunc`, `cl_cf4_dirs`, `cl_need_cf4_file` refuse |
| 746 | `cl_setup_python` prints `SLEEP_OUT_ROOT` (was `CF_COST_FWD`) |
| 768 | dry-run counter `SC_FAKE_COUNTER`, file `.switch_cost_dryrun_jid.<pid>` (a main-project counter is never read or advanced) |
| 834-846 | submission record under `SC_SUBMIT_LOG` (not `CL_SUBMIT_LOG`) |
| 850-873, 876 | `_sc_check_submission`, called first in `cl_sbatch`: job name must start with `switch_cost_`, every `-o/-e` must be in `SC_LOG_DIR`, the payload must be a `switch_cost_*.sh`, `--wrap` refused |
| 975 | `cl_log_dir` -> `scripts/switch_cost_logs`, not created in a dry run |
| 979-986 | `cl_run_marker` -> `data/output/.switch_cost_<prefix>_marker.*` |
| 1016-1028 | orchestrator log under `SC_ORCH_LOG`, file `switch_cost_logs/switch_cost_orchestrator_*.log` (a main `CL_ORCH_LOG` is never appended to) |
| 1047-1173 | skeleton section rewritten: `cl_of_root` -> `data/switch_cost_root/Open-Finance` (`SC_SKEL_ROOT` overrides, `CL_SKEL_ROOT` ignored) (1083); `_cl_link` refuses a link outside the skeleton, refuses a missing source, does not replace a correct link (1085-1105); `cl_bootstrap_tree` links the market panel and the banked/MCA panel from `SC_DATA_IN`, checks the by-name inputs (r4 table for F3 arms only, 1154), refuses (does not delete) a `market_panel.csv`, links `ESTIMATION_OUTPUT` to `SC_ESTIMATION_OUTPUT`; no digital-flag or ESTBAN link (E1 and D6 are out of scope) |
| 1175-1179, 1231-1232 | BBL section notes |
| 1388-1788 | `cl_bbl_event`, `cl_bbl_ctx_write`, `cl_bbl_ctx_load`, `cl_bbl_version_guard`, `cl_check_policy_units`, `_cl_bbl_claim`, `cl_bbl_dispatch_fwd`, `_cl_bbl_submit_part`, `_cl_bbl_submit_packed`, `cl_bbl_live_claims`, `cl_bbl_submit_sweep`, `cl_bbl_retarget_solve` (scontrol on a main job), `cl_bbl_check_curve` refuse |
| 1825-1855 | `sc_preflight` (new, read-only) |

### `switch_cost_sleep_job.sh` (429 lines)

| Line(s) | Edit |
|---|---|
| 2-3 | header lines (after the shebang, before the `#SBATCH` block, which sbatch still reads) |
| 15-58 | header rewritten for the copy |
| 61 | sources `switch_cost_cluster_lib.sh` |
| 66-83 | `merge`, `ame_gate`, `ame`, `wcb_band`, `weakiv`, `entry`, `upsilon` refused with what they write |
| 85-98 | `SPEC` must be 12; gates G3/G4/G7 refused; all before anything is created |
| 100 | `mkdir -p "${SC_LOG_DIR}"` (was `${CL_ROOT}/logs`) |
| 109-121 | `PYTHONDONTWRITEBYTECODE=1` (no `__pycache__` in the shared `scripts/`); BLAS threads from `SC_BLAS_THREADS` (default 2, unchanged) |
| 138-150 | `est`: `switch_cost_sleep_est_single.py --est K --spec12`, K in 3/4 |
| 152-177 | `prep`: `switch_cost_sleep_demand_prep_link.py` per arm, with the parquet count check (172) |
| 179-424 | gate: G1 and G2 only; reads `SLEEP_OUT_ROOT`/`DEMAND_PREP_DIR` (no fallback to `data/output/sleep` or `demand_prep`) and refuses a non-switch-cost folder (219); G1 adds the two-stage AME check (268-306, 333-336); the G3/G4/G7 bodies (which read `data/output/logit` and `data/output/counterfactuals`) are removed; verdict file `.gate_switch_cost_<NAME>.json` in `SC_GATE_DIR` (413) |

### `switch_cost_sleep_run.sh` (278 lines)

| Line(s) | Edit |
|---|---|
| 2-3 | header lines |
| 4-66 | header rewritten (scope, graph, flags) |
| 69 | sources `switch_cost_cluster_lib.sh` |
| 74-95 | defaults: routines `3 4`, both variants, all arms, B, workers, BLAS threads, resources |
| 116-120 | refused flags of the removed branches |
| 127-153 | scope checks (routines, variants, arms, integer options, arm-variant consistency) |
| 156-164 | logs via `cl_log_dir`; `sc_preflight` instead of `cluster_preflight.sh` |
| 178 | payload `switch_cost_sleep_job.sh` |
| 187-192 | `sub`: logs `switch_cost_logs/<name>_%j.out/.err` |
| 196-206, 213, 246 | job names `switch_cost_sleep_gate_*`, `switch_cost_sleep_est_<variant>_E<k>`, `switch_cost_sleep_prep_<arm>_E<k>` |
| 208-258 | the est and prep graphs (replace the E1/E2 array, the spec fan-out, merge, AME, WCB, weak-IV, upsilon and entry branches) |
| 262 | the disabled steps are named on every submission |
| 270-278 | handles renamed `SWITCH_COST_*_JOBID(S)` |

## 4. Disabled steps

Refused by `switch_cost_sleep_job.sh` (exit 2, message naming what the step writes) and never
submitted by `switch_cost_sleep_run.sh`. None of their scripts has a switch-cost copy.

| Step | Script | Writes in the main project |
|---|---|---|
| E1/E2 (`est`, routines 1, 2) | `sleep_est_e1.py`, `sleep_est_e2.py` | `sleep/est1`, `sleep/est2` (out of scope; the job refuses routine 1/2) |
| spec fan-out | `sleep_est_single.py --spec-id S` | `sleep/est{K}/_specs/spec_S.pkl` (replaced by `--spec12`) |
| `merge` | `sleep_est_single.py --merge-specs` | `sleep/est{K}/estimation_results.pkl`, band, phi CSVs |
| `prep` (original) | `sleep_pipeline.py --skip-sleep` | exporters to `sleep/Rout` and `drafts_dir()`, `demand_prep/`, desc_3 (replaced by the per-arm link prep) |
| `ame_gate` | `sleep_ame_twostage.py --theta-off` | regression guard against stored production numbers |
| `ame` | `sleep_ame_twostage.py` | `sleep/Rout/ame_twostage_est{K}_robust_if.pkl` (the two-stage AME runs inside `est` here) |
| `wcb_band` | `sleep_wcb_band.py` | `sleep/Rout/wcb_band_est{1,2}.pkl` |
| `weakiv` | `sleep_weak_iv.py` | `BLP_RESULTS/cluster_processed/weak_iv_sleep.json`, `DIAG_WEAK_IV_SLEEP/` (BLP family) |
| `upsilon` + G7 | `sleep_upsilon_export.py` | `counterfactuals/upsilon_pix_E{k}_spec_12.json`, `phi_nopix_E{k}_spec_12.parquet` (CF family) |
| `entry` | `sleep_ident_entry_dynamics.py --compute-only` | `DIAG_PHI_SEPARATION/` (reads `phi_nopix`) |
| gates G3, G4, G7 | embedded gate | read the AME files, `logit/`, `counterfactuals/` |
| preflight | `cluster_preflight.sh` | probe files created and removed in every main step folder; Julia on the shared depot |

Library functions that refuse: listed in section 3 (Julia/sysimage/GPU, BLP/BBL/CF checks, BBL
dispatch writers and submitters, `scontrol update`). There is no env creation, pip/conda install,
sysimage build, archive or clean function in the library; the build commands are only text
(`cl_sysimage_build_cmd`, neutralised), and `cluster_archive.sh` is not called by any copy.

## 5. Shared state the copies still touch (read-only, or create-only)

- `scripts/` code: the copies and the shared modules they import are read. `scripts/switch_cost_logs/`
  is created there. `PYTHONDONTWRITEBYTECODE=1` keeps `__pycache__` out.
- Shared Python imported, not copied: `utils/__init__.py`, `paths.py`, `state_transform.py`,
  `demand_chars.py`, `routines.py`, `window.py`, `se_national.py`, `cluster_stats.py`,
  `venv_guard.py`, `sleep_links.py` (through `se_national.py`), and `config/routines.toml`.
- The conda env `dep_comp_blp` (`module load miniconda`, `source activate`, `python -c import`).
  Nothing is installed.
- `SC_DATA_IN` (default `data/input/switch_cost/`): read through symlinks and by name, never written.
- `data/output/` itself: created if absent; the gate jsons are written at its root.
- `data/`: `data/switch_cost_root/` is created.
- The user's home caches (matplotlib font cache, conda), shared with the main project's jobs.
- SLURM: same account and fairshare, partition `day`, `--constraint=cpugen:turin` added by
  `_cl_cpugen_args` (unchanged from the original); mail to the same address.
- Still active, read-only and unused by the copies: `cl_julia_candidates`, `cl_manifest_path`,
  `cl_manifest_julia_version`, `cl_loaded_julia_version` (`julia --version`), `cl_check_julia_version`,
  `cl_sysimage_for`, `cl_manifest_sha`, `cl_sysimage_verdict`, `cl_bbl_discount_get`,
  `cl_bbl_live_named` (squeue), and the pure helpers.

Main-project actions that could reach our files: `cluster_archive.sh --set gates` globs
`output/.gate_*.json`, which matches `.gate_switch_cost_*.json` (open question 3). No other main
glob found reaches a `switch_cost_*` folder (`--set sleep` is `output/sleep/...`, `--set logs` is
`logs/*.out|*.err|orchestrator_*.log`).

## 6. Input inventory (for the data zip; nothing zipped)

Cluster paths assume the default `SC_DATA_IN = data/input/switch_cost/` under the project root
(`cluster/upload_manifest.txt` names it `/nfs/roberts/project/pi_mf2263/pf382/dep_comp`). Local
paths are relative to `Open-Finance/`.

| File | Read by | Local path | Cluster path | How the job reaches it | Bytes | md5 |
|---|---|---|---|---|---|---|
| `market_panel.parquet` | est, prep | `BCB/Egan_et_al_2025_Rep/processed/market_panel.parquet` | `data/input/switch_cost/market_panel.parquet` | link `data/switch_cost_root/Open-Finance/BCB/Egan_et_al_2025_Rep/processed/market_panel.parquet` | 40,938,984 | `6613228b2fe6b0ba95d0f16d1d5a743c` |
| `state_centering_means.json` | est, prep | `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/DEMAND_PREP/state_centering_means.json` | `data/input/switch_cost/state_centering_means.json` | `STATE_CENTERING_JSON` | 1,385 | `f43be34ed3f52b1892a14f851bd00154` |
| `demand_chars.parquet` | prep | `BCB/Egan_et_al_2025_Rep/processed/demand_chars.parquet` | `data/input/switch_cost/demand_chars.parquet` | `DEMAND_CHARS_FILE` | 14,017,312 | `cfd25d7d1c858bf0ab5d3e2692d38e9e` |
| `bcb_banked_mca_panel.csv` | prep | `BCB/Inclusion/bcb_banked_mca_panel.csv` | `data/input/switch_cost/bcb_banked_mca_panel.csv` | link `.../Open-Finance/BCB/Inclusion/bcb_banked_mca_panel.csv` | 7,067,387 | `be2115c642564f7db42e16719173112f` |
| `r4_locked_bank_year.csv` | prep, F3hi/F3lo only | `BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/SWITCH_COST/switch_cost_inputs/r4_locked_bank_year.csv` | `data/input/switch_cost/r4_locked_bank_year.csv` | `--locked-csv` | 8,339 (117 rows) | `6292f5c2d0c7b1df83c4b23d73c07c4c` |

Total about 62 MB. Checks made now (read only): `demand_chars.parquet` records the market-panel
md5 `6613228b2fe6b0ba95d0f16d1d5a743c` in its schema metadata, equal to the local
`market_panel.parquet`, so the two must travel as this pair (the prep raises otherwise). The
"Hausman" instruments are panel columns (`leave_one_out_mean_spread_a*`), not a separate file.
Without `bcb_banked_mca_panel.csv` the prep would fall back to a constant banked correction with a
warning; the bootstrap refuses instead.

Produced on the cluster and read by later steps: `switch_cost_sleep_<variant>/est{3,4}/estimation_results.pkl`
(read by the prep; under `--skip-est` they must already be there).

Not needed by the kept steps: `digital_banks_diagnostic.csv` (E1 only), `estban_own_branches.csv`
(D6 entry), `state_centering_tier0.csv` (the centering audit).

Code the kept steps run (code bundle, under `scripts/`): `switch_cost_cluster_lib.sh`,
`switch_cost_sleep_job.sh`, `switch_cost_sleep_run.sh`, `switch_cost_sleep_est_single.py`,
`switch_cost_sleep_est_e2.py`, `switch_cost_sleep_demand_prep_link.py`,
`utils/switch_cost_sandbox.py`, `utils/switch_cost_sleep_links.py`, the shared modules of section 5,
`config/routines.toml`. Local only, not needed on the cluster: `switch_cost_r4_locked.py` (builds
the r4 table from `raw/DEPOSIT_MATURITY/deposit_maturity_summary.csv` and
`raw/PILAR3/liq2_retail_buckets.csv`) and `switch_cost_prep_checks.py` (identity checks). The
unused copies (E1, the orchestrator, the E1/E2 preps, the two linear diagnostics) were removed on
2026-10-13 at 21:12.

## 7. Dry run

`TMPDIR=<scratchpad> CL_VERBOSE=1 bash switch_cost_sleep_run.sh --dry-run --prep` works in Git
Bash with no `sbatch` on the PATH: exit 0, 18 printed submissions with synthetic ids
9000001-9000018 and the dependency graph of section 1, every name `switch_cost_*`, every log in
`switch_cost_logs/`, every payload `switch_cost_sleep_job.sh`. It created nothing in the repository
or the data tree (no `switch_cost_logs/`, no `Code/data/`); the only file is the id counter in
`TMPDIR`. It is slow here (several minutes for 18 submissions, the cost of process forks in Git
Bash on this machine), not on Linux.

Refusals exercised: `--routines 1`, `--no-ame`, `--spec-ids`, `--prep --arms F1 --variants pooled`,
`--skip-est` without `--prep`, `--boot-b x` (run script, exit 2); `SLEEP_STEP=` `ame`, `upsilon`,
`weakiv`, `entry`, `wcb_band`, `ame_gate`, `merge`, gate `G7`, `SPEC=11` (job script, exit 2,
before anything is created). The job script itself has no dry-run mode; it was not run past its
refusals (it would build the skeleton and load modules).

## 8. Grep of the copies for survivors

- `data/output/<family>`: none in code. The BBL refusal messages mention `bbl/.dispatch/...` as
  text (lib 1388, 1405, 1586).
- `.gate_`: lib 254 is a comment about `.gate_G0.json` in the manifest-version helper. All
  others name `.gate_switch_cost_*`.
- `-J`: lib 838, 854, 859 are argument parsers; lib 1655, 1673, 1712, 1751 submit `bbl_fwd_*` /
  `bbl_sweep_*` inside functions that refuse on entry, and `cl_sbatch` would refuse those names and
  the `bbl_job.sh` payload in any case. The run script's only `-J` is `sub` (189), names prefixed.
- `logs/`: comments only (lib 85, 973, 1001; run 20).
- Unprefixed script names: comments of the library (`bbl_run.sh`, `blp_run.sh`, `env_job.sh`,
  `pipeline_all.sh`, `of_root.jl`, ...); `${CL_ROOT}/bbl_job.sh` payloads in the refused BBL
  submitters; the job script's refusal messages name the production scripts on purpose
  (`sleep_ame_twostage.py`, `sleep_wcb_band.py`, `sleep_weak_iv.py`, `sleep_ident_entry_dynamics.py`,
  `sleep_upsilon_export.py`, `sleep_est_single.py`); `blp_logit.jl` in a gate docstring; the
  header lines name the sources; the run script's header says it never runs `cluster_preflight.sh`.
  Every Python step the copies execute is a `switch_cost_*` script.

## 9. Findings for the Python side (not changed here)

1. `utils/se_national.py` (shared) imports the production `utils/sleep_links.py`
   (`cluster_wild_bootstrap`, `_ame_band_from_draws`). The switch-cost E2 path and
   `utils/switch_cost_sleep_links.py` call `se_national`, so national-regressor standard errors come
   from the production bootstrap code. Read-only, but a change to the switch-cost links' bootstrap
   does not reach that path.
2. `switch_cost_sleep_demand_prep_link.py` exits 0 with nothing written when a pickle or a spec is
   missing; the job checks the parquet itself.
3. `_attach_twostage_ame` swallows a bootstrap error (`[2s-AME] SKIPPED`) and the fit is saved
   without it; gate G1 now fails on that.
4. An inherited `SLEEP_RESUME=1` would make `--spec12` skip a spec already on disk, and with it the
   AME. The job prints `SLEEP_RESUME` in its banner; do not export it in the launch shell.
5. The link prep builds `dtype_2/4/5` unconditionally, so the `SWITCH_COST_TYPE_SHIFTS` value of a
   prep job (set to the arm's own variant, as the sandbox requires) does not change its output.

## 10. Open questions for the user

1. **Where the inputs come from.** The copies read `SC_DATA_IN`, default `data/input/switch_cost/`
   (a switch-cost upload). The alternative is the main project's `data/input/` read as is
   (`SC_DATA_IN=<project>/data/input`). Recommendation: the separate folder. It pins the vintage
   (the main launch may re-upload mid-run), a zip unpacked there cannot overwrite main inputs, and
   the panel and `demand_chars.parquet` must be the matched pair above.
2. **The conda env.** The copies use the main env `dep_comp_blp` read-only. Recommendation: keep
   it (it imports the six packages; nothing to install), and ask the main session not to change it
   while the run is live; a separate env would need an install, which these copies refuse to do.
3. **Gate file location.** As specified, the verdicts are `data/output/.gate_switch_cost_*.json`.
   Main's `cluster_archive.sh --set gates` (`output/.gate_*.json`) would package them into its own
   gates zip, and a `--move` archive would remove them. Recommendation: move them into a
   switch-cost folder (one line, `SC_GATE_DIR`, lib 84; for example `data/output/switch_cost_gates`).
4. **BLAS threads during the AME.** The job keeps the original 2 BLAS threads; with 64 AME workers
   on 64 cpus that is 128 threads (the production AME driver pins 1). Options: `--blas-threads 1`
   (also changes the fit's BLAS reduction order, last-ulp), `EST_CPUS=128` (a whole turin node per
   job), or `--ame-workers 32`. Recommendation: `--blas-threads 1`, since pooled and typeshift are
   compared under the same setting.
5. **Demand prep on the cluster.** Off by default (`--prep` opts in), since the local prep runs.
   If both run, the same arm parquet has two producers. Recommendation: keep it off unless the
   cluster fits differ from the local ones; then decide which set of parquets is authoritative.
6. **Wall clock and memory.** 10 h / 64 cpus / 256G per est job and 6 h / 4 / 64G per prep job are
   derived from the original budgets, not measured; four 64-cpu jobs will queue beside the main
   launch under the same fairshare. Recommendation: one smoke run first (`--routines 3 --variants
   pooled --boot-b 9`), or launch after the main sleep phase.
7. **Bringing results home.** No switch-cost archive exists and main's `cluster_archive.sh` must
   not be used (main patterns, move mode). Recommendation: a copy-mode zip of the
   `data/output/switch_cost_*` folders, the gate jsons and `scripts/switch_cost_logs/` into
   `data/output/switch_cost_download/` (named, not created).
8. **Names of the new folders.** `data/switch_cost_root/Open-Finance`,
   `data/output/switch_cost_estimation_output`, `scripts/switch_cost_logs`: to confirm against the
   cluster layout the user will describe.
9. **G1's reference phi.** G1 compares the pooled fit with 96.5627 / 96.6814 pp (the main gate's
   promoted values for E3/E4), report-only. Whether those are still the current production
   vintage, or G1 should compare with the local production `national_phi_t.csv` means.

## 11. Changes after review (coordinator, 2026-10-13 19:00–19:20)

- **Finding 1 fixed.** `utils/se_national.py` is copied to `utils/switch_cost_se_national.py`,
  whose two imports point at `utils/switch_cost_sleep_links.py`; the three import sites in the
  copies now use the copy. No copy imports a production sleepiness module any more. The functions
  involved are identical in both libraries, so the local fits are unaffected.
- **Finding 2 fixed.** The link prep exits 2 when a fit is missing and 3 when no parquet was
  written. The job's own parquet count stays as a second check.
- **Open question 3 applied.** `SC_GATE_DIR` is `data/output/switch_cost_gates` (lib 84-87), so
  main's `cluster_archive.sh --set gates` never sees the verdicts.
- **Open question 4 applied.** `SC_BLAS_THREADS` defaults to 1 in the job and the run script, as
  the local fits ran.
- **Open question 9 applied.** G1 compares each variant with the LOCAL switch-cost fit of the same
  code and data: pooled E3 98.2638 / E4 98.1471 pp (equal to the local production
  `national_phi_t.csv` to four decimals), typeshift E3 92.0371 / E4 91.1434 pp. The main gate's
  96.5627 / 96.6814 pp do not match the local production files and are not used.
- **Open question 7.** `switch_cost_collect.sh` (new, copy mode) zips the switch-cost outputs,
  gates and logs into `data/output/switch_cost_download/` with a `.sha256`; it never moves or
  deletes. `market_panel_phis.csv` (about 2 GB per fit) only with `--with-phis`.
- Open questions 1, 2, 5, 6 and 8 stay with the user.
- **Removed 21:12.** The six unused Python copies (`switch_cost_sleep_est_e1.py`, the
  demand-prep orchestrator, the E1 and E2 preps, `switch_cost_sleep_ident_augmented.py`,
  `switch_cost_sleep_ident_interaction.py`) are deleted; nothing imports them.
