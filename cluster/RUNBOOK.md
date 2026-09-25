# Bouchet runbook — the cluster scripts

One page, in order. The step-by-step launch checklist is `cluster/LAUNCH_SEQUENCE.txt`; this file is
the reference behind it — what each script is, what it decides for itself, and what to read when
something dies.

## Rule 0 — nothing runs on the login node

On the login node you upload, `unzip`, `chmod`, `sha256sum`, `sbatch`, and look (`squeue` / `ls` /
`cat`). Nothing else. Every check that needs the toolchain — the Julia module, both sysimages, the
Python import stack, the staged inputs — lives inside **gate G0**, which is a job
(`env_job.sh ENV_STEP=preflight`), and every first-tier submission waits `afterok` on it.

`pipeline_all.sh` carries its own `#SBATCH` block and is **submitted**, never `bash`-run: all it does
is parse flags and call `sbatch`. The one login-safe exception is `bash pipeline_all.sh --dry-run`,
which prints the whole graph and submits nothing.

The project root is the parent of `scripts/` and `data/`:

```
cd ~/project_pi_mf2263/pf382/dep_comp
```

`data/input` is **uploaded** and `data/output` is **cluster-produced**. The rule is by PRODUCER, not
by role.

---

## The sixteen scripts

Nine are things you reach for; six are job payloads `sbatch` reaches for you; one is the library
every other file sources first. Every driver takes `--dry-run` and `-h`.

| you type | what it does |
|---|---|
| `pipeline_all.sh` | **the** command. `sbatch pipeline_all.sh` runs the whole chain |
| `sleep_run.sh` | the sleepiness (A) phase: estimators → prep → AME + upsilon |
| `blp_run.sh` | the RC-BLP demand sweep, plus the sysimage/draws builds it decides it needs |
| `bbl_run.sh` | the BBL cost stage: polfunc → packed `fwd_sim` jobs → sweep (re-runs gaps) → solve. Operated from `BBL_RUNBOOK.md` |
| `bbl_status.sh` | one screen of BBL state and the next action (shards, fwd R/P/F, sweep retries, solve). Login-safe |
| `bbl_sizing.sh` | reads the memory probe (`bbl_run.sh --probe`) and prints MEM, PACK_H200, PACK_H100 and the launch command. Login-safe |
| `bbl_cancel.sh` | cancels a BBL chain dependents-first (tables/zip → solves → sweeps → arrays). Login-safe |
| `bbl_discount.env` | the registry of the BBL discount factor and horizon (`BBL_BETA`, `BBL_HORIZON`); every BBL/CF script reads it |
| `cf_run.sh` | the counterfactuals (`demand_eval`, `cf1`, `cf1_net`, `cf4`) |
| `cf_eq_run.sh` | the long equilibrium CFs (CF3 / CF5 / CF6), opt-in |
| `cluster_archive.sh` | the one packager: a step folder → `data/output/download` |
| `env_job.sh` | one toolchain job, four steps (below) |
| `cluster_preflight.sh` | the verdict block; submits nothing. Loads Julia, so it runs inside a job |

| `sbatch` reaches it | payload |
|---|---|
| `sleep_job.sh` | the sleepiness steps and every content gate (`SLEEP_STEP`, `SLEEP_GATE`) |
| `logit_job.sh` | `blp_logit.jl --hpc` |
| `blp_draws_job.sh` | `blp_draws.jl` |
| `blp_stage_job.sh` | one RC stage (`RC_ROUTINE` / `RC_ENGINE` / `RC_STAGE`) |
| `bbl_job.sh` | one BBL step (`BBL_STEP`) |
| `cf_job.sh` | one CF step (`CF_STEP`) |

`cluster_lib.sh` is sourced by all fifteen: module names, the step-directory seam
(`CL_STEP_SLEEP … CL_STEP_DOWNLOAD`), routine defaults, per-routine memory, the `sbatch` wrapper and
the quiet-logging helpers. Every other script fails at line ~2 with a plain
`No such file or directory` without it.

The phase drivers run `cluster_preflight.sh` at submit time, which loads Julia — so they belong
inside a job too. `pipeline_all.sh` submits them for you; by hand, wrap one:

```
sbatch --wrap "cd ~/project_pi_mf2263/pf382/dep_comp/scripts && bash blp_run.sh --routines '3 4'"
```

`bash <driver> --dry-run` is login-safe: a dry run skips the preflight and submits nothing.

---

## 0. Upload

Both zips go **into the project root** (not `~`, not `data/input`): the member paths are already
prefixed `scripts/` and `data/input/`, so one `unzip -o` from the root places every file.

Build them locally with `python cluster_upload.py --stage`; the manifest is
`cluster/upload_manifest.txt`. The data bundle is **seven** files, ~50 MB — the market panel travels
as `market_panel.parquet`, not the 724 MB CSV.

After unzipping:

```
mkdir -p scripts/logs
chmod +x scripts/*.sh
sha256sum -c sha256SUMS.txt
```

`mkdir -p scripts/logs` is load-bearing. A zip cannot carry an empty directory, every `sbatch` here
writes `--output` into `scripts/logs/`, and SLURM does not create a missing `--output` directory: the
job is killed in about a second **with no log at all**.

---

## 1. Resolve the Julia manifest — one job, alone

```
cd ~/project_pi_mf2263/pf382/dep_comp/scripts
sbatch --partition=day --time=02:00:00 --cpus-per-task=8 --mem=32G \
       --export=ALL,ENV_STEP=resolve env_job.sh
```

**Run this after every code-bundle upload, not only the first.** The bundle ships the
`Manifest.toml` resolved on the local machine (Julia 1.12.6 as staged), `unzip -o` overwrites
whatever the cluster's own resolve wrote, and Bouchet loads Julia 1.11.4 — so a re-upload puts the
mismatch back and gate G0 blocks at section (3). `grep julia_version Manifest.toml` must read
`1.11.x` before step 2.

Wait for it (`squeue -u $USER`) and submit nothing else meanwhile — concurrent `Pkg.resolve()` on NFS
corrupts `Manifest.toml`. This is the sole `Pkg.resolve()` site. It backs the manifest up to
`Manifest.toml.bak.<ver>`, derives the accepted version from the **loaded** Julia rather than a
hardcoded literal, and installs into `scripts/.julia_depot` — the depot every job reads. An
incompatible manifest is moved aside before the resolve and **restored if the resolve leaves no
readable manifest**, so a resolve that dies (a `day` node with no route to the registry, a wall
kill) cannot leave the tree with no manifest at all — the state that made `.gate_G0.json` report an
empty `manifest_julia_version`.

That depot is the point. A resolve that installs into `~/.julia` reports "all key packages loaded OK"
and the next job still dies on `Package Parquet2 … is required but does not seem to be installed`:
same `Manifest.toml`, different library. G0 probes the depot directly rather than trusting a version
match.

**If the Julia module name has drifted**, the preflight prints the parsed candidates:

```
ERROR: 'module load Julia/1.11.4-linux-x86_64' FAILED.
  --- parsed candidates ---   Julia/1.10.4-foss-2022b   Julia/1.12.0 ...
```

Pick one and keep it for the session — `export JULIA_MODULE=<candidate>`. Every script reads the name
from `cluster_lib.sh`; nothing else needs editing.

**If the conda env is wrong**, section (5) prints the one-line unblock:

```
FAILED: 'costsolve' does not import numpy, pandas, scipy, pyarrow
  ... 'dep_comp_blp' exists and imports the stack cleanly. One-line unblock:
      export CONDA_ENV=dep_comp_blp
```

Do not skip this. The import failure it catches otherwise surfaces ~14 h later, when the `fwd_sim`
array drains — and the solve's `afterok` cascade then cancels `cf1_net` with **no log at all**.

`env_job.sh` has four steps, dispatched on `ENV_STEP`:

| `ENV_STEP` | what it does |
|---|---|
| `preflight` | gate G0: build the tree, load Julia, run `cluster_preflight.sh`, write `.gate_G0.json` |
| `resolve` | the sole `Pkg.resolve()` site |
| `sysimage_gpu` | build `blp_sysimage.so` (submit with `--partition=gpu_h200 --gpus=h200:1`) |
| `sysimage_cpu` | build `blp_sysimage_cpu.so` (submit with `--partition=day --constraint=cpugen:turin`) |

The two sysimage steps are submitted for you by `blp_run.sh` when either image is absent. Both end in
a **fatal load proof** on the node type that will use the image and write a provenance sidecar
`<img>.so.json` (julia version, module, `Sys.CPU_NAME`, build host, partition, `Manifest.toml`
sha256, build time). A build whose proof fails exits non-zero and the `.so` is renamed `.so.failed`,
so a broken image never sits on disk looking done.

Two images, because a sysimage bakes its build node's CPU target: `blp_sysimage.so` is built on
gpu_h200 (sapphirerapids) and is **rejected on `day` nodes**, and `blp_sysimage_cpu.so` is built on
`day` and pins `cpugen:turin`, because `day` mixes turin (AMD) with emeraldrapids (Intel).

---

## 2. Launch

```
cd ~/project_pi_mf2263/pf382/dep_comp/scripts
bash pipeline_all.sh --dry-run       # read the graph; submits nothing
sbatch pipeline_all.sh               # do it
```

Submit from `scripts/`: `sbatch` runs a spool copy of the file, so `pipeline_all.sh` finds
`cluster_lib.sh` through `SLURM_SUBMIT_DIR`.

Defaults: routines **1 2 3 4** through *every* phase, sleepiness spec-arrays on, the equilibrium CFs
off. Both sysimages and the draws are built when absent — `blp_run.sh` decides from disk and prints
the decision, so `--blp-args` is only for forcing a rebuild, and `--skip-preflight` is not needed
because G0 is a job that passes on its own.

Useful flags (`pipeline_all.sh -h` prints them all):

```
--from <sleep|logit|blp|bbl|cf|cfeq>   resume at a phase
--routines "3 4"                       RC/BBL/CF routine set
--sleep-routines "1 2 3 4"             sleepiness routine set
--cfeq [--cfeq-modes "cf3 cf5"]        add the equilibrium CFs
--no-sleep --no-logit --no-blp --no-bbl --no-cf
--blp-args "--draws --sysimage"        force those builds
--dry-run
```

### The run stops after the logit phase — that is the design

`blp_run.sh`, `bbl_run.sh` and `cf_eq_run.sh` preflight their inputs **on disk at submit time**, and
at t=0 those files genuinely do not exist. So each phase boundary submits a thirty-minute CPU job
that re-invokes `pipeline_all.sh --from <next>`:

```
pipe_next_blp    afterok gates G2 and G4     -> the RC ladders
pipe_next_bbl    afterok the RC terminals    -> BBL costs + the CFs
pipe_next_cfeq   afterok the CF results      -> only with --cfeq
```

`squeue` therefore shows far fewer jobs than the dry run printed. Read
`scripts/logs/pipe_next_*.out` at each hand-off.

### The gates

Each gate is a ~1-minute CPU job that OPENS the artefacts, checks their CONTENT, writes
`data/output/.gate_G<n>.json`, and only then exits nonzero. Everything downstream chains `afterok`
the gate, never the producer.

| gate | checks |
|---|---|
| G0 | toolchain, uploaded inputs, step dirs writable, both sysimages probed by loading them |
| G1 | the estimators landed: `market_panel_phis.csv` per routine, and phi-hat against the promoted values |
| G2 | exactly one demand parquet per routine in `data/output/demand_prep` |
| G3 | the AME two-stage bootstrap |
| G4 | each logit delta's leading `Int64` against its routine's parquet row count |
| G7 | `upsilon_pix` + `phi_nopix` for all four routines: `exact_nopix` and full key coverage |

G1 compares against `E1 99.9945 · E2 99.5400 · E3 96.5627 · E4 96.6814`. The multistart caps at 8
threads / 4 starts on any node size, so these should match — a match validates the whole port end to
end. G1 reports a move and never fails the chain, but a move is worth stopping for.

G4 matters because the RC engine **silently** skips a mismatched delta and starts cold.

---

## 3. The output tree

```
data/output/
  .gate_G{0,1,2,3,4,7}.json
  sleep/            est{k}/, Rout/ (tex, figures, ame pickles), DIAGNOSTICS/
  demand_prep/      demand_{k}[_tag]_spec_{s}.parquet + the prep summaries
  logit/            logit_delta_E{k}_spec_12.{bin,jls}, the *.jls fits, the tex
  blp/              blp_results_*, blp_checkpoint_*, blp_summary_*
  blp/draws/        halton_nu_*, demo_draws_* (+ parts), the key index
  bbl/              polfunc_fitted.csv, psi_eq_*, psi_dev_*_shard{i}of{N}.parquet, cost_params_*
  counterfactuals/  upsilon_pix_*, phi_nopix_*, shares_elas_*, cf1_franchise_*, cf4_pix_realloc_*
  download/         every zip + .part.NN + sha256SUMS
```

One family, one directory. Every consumer reads the step folder its producer wrote, so no artefact
has a second, older copy anywhere for a search order to pick up by mistake.

The equilibrium CFs nest their sigma trees under `counterfactuals/`:

```
cf3_jacobi_E{k}_{stage}/
cf5_E{k}_{stage}/{base,shock}/
cf6_E{k}_{stage}/{base,merged}/
```

---

## 4. Archive and download

`pipeline_all.sh` submits `pipe_download` itself, `afterANY` the CF jobs and the BBL solves, and it
packages **all eight sets**. By hand:

```
bash cluster_archive.sh --set blp --copy
bash cluster_archive.sh --all --copy      # sleep demand_prep logit blp bbl counterfactuals gates logs
```

`--copy` / `--move` is **required**; there is no default and no inference, and every caller in the
pipeline passes `--copy` — the step folders stay readable for the rest of the run and the download is
meant to be complete rather than incremental. `--dry-run` lists exactly what would be taken and exits
before touching anything; `--tag <t>` makes a separate snapshot instead of appending into one zip;
`--newer <marker>` restricts to files newer than a marker.

Everything lands in `data/output/download`. Anything over `${SPLIT_BYTES:-3000000000}` becomes
`<name>.zip.part.NN` of `${CHUNK:-2G}` plus a `.sha256` of the whole zip, because a browser download
has no resume; one `sha256SUMS` covers every emitted file.

Two size knobs: `INCLUDE_DRAWS=1` adds the ~14 GB `demo_draws_*` to the `blp` set (off by default —
the local machine regenerates it from `R`/`seed` in minutes), and `INCLUDE_PSI=0` drops the `psi_*`
shards from the `bbl` set.

Archives are **one-directional**. Nothing in the cluster scripts ever unzips one back into the tree.

Then, on the local machine, download `data/output/download` whole into
`<processed>/ESTIMATION_OUTPUT/CLUSTER_IN/` and run:

```
python cluster_ingest.py --dry-run
python cluster_ingest.py
```

It reassembles parts, verifies against `sha256SUMS`, and routes all eight families into the local
tree. A checksum mismatch refuses **that family only**; the rest still land and the run exits 1.

---

## 5. If a job dies

* **A missing log is the symptom, never the cause.** A job cancelled `DependencyNeverSatisfied`
  (`--kill-on-invalid-dep=yes`) writes **no log at all**. If `cf1_net` has no log, read the
  corresponding `bbl_solve` `.err`; if a phase never started, read the gate json before it.
* **A GPU job that refuses in ~20 s** with `CUDA.functional() == false` has a sysimage / CPU-target
  mismatch. Rebuild the image (`ENV_STEP=sysimage_gpu`). The refusal is deliberate — the alternative
  is an H200 held at ~0% utilisation for a whole wall. `ALLOW_SYSIMAGE_FALLBACK=1` restores the
  degrade-to-CPU behaviour, loudly.
* **A sysimage with no provenance sidecar is UNKNOWN provenance and is refused.**
  `ALLOW_UNSTAMPED_SYSIMAGE=1` accepts it deliberately. On a fresh cluster, let the run build both
  rather than trusting an image left lying around.
* **A wall-killed BLP chain resumes by plain resubmission.** Each stage warm-starts from the previous
  stage's on-disk checkpoint and completed stages are skipped; the checkpoints sit in
  `data/output/blp` and the archiver only ever copies, so nothing has to be unpacked first.
* **An archive that prints `WARNING: N file(s) failed to archive`** has deleted nothing. Re-run the
  same command.
* **The `gpu_h200` QOS caps concurrent jobs and GPUs per user.** Read it with
  `sacctmgr -n -P show qos gpu_h200 format=MaxJobsPU`; G0 reports it against the number of chains the
  run wants. GPU arrays drain in batches of that size — slow, not broken.

---

## 6. Deliberate behaviour worth knowing

* **Routine defaults are 1 2 3 4 everywhere** — the sleepiness phase, the RC ladders, BBL and the
  counterfactuals — set once in `cluster_lib.sh` (`CL_ROUTINES_ALL` / `_RC` / `_CF`). Every run
  prints where the value came from.
* **Per-routine memory.** `cl_mem_for` gives E1/E2 `${CL_MEM_BIG:-600G}` and everything else
  `${CL_MEM_DEFAULT:-200G}`, applied on every RC submission. The BBL/CF CPU jobs use `MEM=256G` for
  all routines; the polfunc pre-step takes 64G.
* **Wall time is capped by the QOS, not by us.** `CL_WALL_DEEP` / `CL_WALL_CAP` default to 2 days
  because a 4-day request is rejected at submit time with `QOSMaxWallDurationPerJobLimit`.
* **The archiver has no default mode.** Neither `--copy` nor `--move` is reachable by omission, and
  five independent guards keep it away from `data/input` and from its own `download/` folder.
* **`polfunc` is a default pre-step of `bbl_run.sh`, not an upload.** It needs only the market panel,
  which the sleepiness phase already brings in, so fitting it on the cluster costs one short CPU job
  and removes a 59 MB upload that could drift out of step with the panel it was fitted on.
  `--no-polfunc` uses the CSV already on disk.
* **`fwd_sim` runs on the H200 by the code's own default**, and `bbl_run.sh` does not silently change
  it. `pipeline_all.sh` passes `--fwd-cpu`, because the step measures as memory-bandwidth bound at
  ~0% GPU utilisation and a job holding an H200 at ~0% costs later priority.
* **Scripts are quiet by default.** Each orchestrator prints one line per submission; the full
  transcript goes to `scripts/logs/orchestrator_<ts>_<pid>.log`. `CL_VERBOSE=1` puts it back on the
  terminal.
