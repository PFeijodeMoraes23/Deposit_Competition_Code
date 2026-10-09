---
name: cluster-run
description: Prepare, launch, monitor and recover this project's jobs on Bouchet (Yale HPC, SLURM) - the full chain through pipeline_all.sh, RC-BLP, BBL forward-simulation shards, counterfactuals and the upload bundles that feed them. Use whenever the user asks for a cluster command, a launch or relaunch line, a probe, an upload list, a reading of squeue, sacct or log output they pasted, or why a job died, stalled or never started, even when they do not say "cluster" (for example "give me the line for E3", "the solve has no log", "is the run done?").
---

# Cluster runs on Bouchet

## How the work is split

You cannot reach the cluster. The user pastes each command into the Bouchet terminal, uploads files
through Yale's web file manager, and pastes back what the terminal printed. So this skill always
produces three things: commands that survive a paste, a statement of what their output must show,
and a reading of what came back. A wrong command costs cluster hours and often fails without
leaving a log, so the care goes in before the command is handed over.

Heavy work belongs on the cluster. The local machine is a 15 W laptop CPU with 32 GB: use it for
code edits, syntax checks, internet fetches, tables and small smokes, one heavy process at a time.

## Read the authority before composing anything

| You need | Read |
|---|---|
| The whole chain, the gates, the output tree, what to read when a job dies | `cluster/RUNBOOK.md` |
| The first-launch checklist, step by step | `cluster/LAUNCH_SEQUENCE.txt` |
| The BBL stage: probe, launch, sweep, status, cancel, repair, every knob | `BBL_RUNBOOK.md` |
| A driver's flags | the header of the script, which is what `-h` prints |
| Defaults: modules, routine sets, memory, partitions, walls | `cluster_lib.sh` |
| Which inputs are uploaded and which the run produces | `cluster/upload_manifest.txt` |
| The `dx` demand variant | `dx_suite_20261001.sh`, `dx_bbl_run.sh`, `blp_dx*.jl` |

Flags and defaults move between runs. Build each command from the current file, never from memory
or from a command quoted in an old note. The sections below hold what is expensive to rediscover.
If a runbook disagrees with this file, the runbook is right, and this file should be corrected.

## Rules for every command handed to the user

1. **One line, no trailing backslash.** On 2026-09-24 a two-line `bbl_run.sh` command lost its
   continuation in the paste. Bash ran the first line alone, and `bbl_run.sh` submitted its default
   chain: an untagged run, a policy-function rebuild, a solve. Six jobs had to be cancelled before
   they overwrote results. The runbooks print a few commands over two lines; join them first.
2. **Self-locating.** Start with `cd ~/project_pi_mf2263/pf382/dep_comp/scripts && ` so the command
   does not depend on where the user's shell happens to be.
3. **Nothing runs on the login node** beyond upload, `unzip`, `chmod`, checksums, `sbatch`, looking
   (`squeue`, `sacct`, `sinfo`, `ls`, `cat`, `grep`) and the login-safe helpers `bbl_status.sh`,
   `bbl_sizing.sh`, `bbl_cancel.sh`. The drivers (`*_run.sh`) run the preflight, which loads Julia,
   so they go inside a small job: `sbatch -p day -t 00:30:00 -c 1 --mem=4G -J <name> -o logs/<name>_%j.out --wrap "bash <driver> ..."`.
   `pipeline_all.sh` carries its own `#SBATCH` block and is submitted directly.
   `bash pipeline_all.sh --dry-run` is the login-safe way to print the graph.
4. **Say what the output must show.** For a submission, name the exact job names the log must list
   (`grep -h '^BBL_JOB_NAMES=' logs/bbl_submit_<jobid>.out` and the line it must print), so a
   mis-parse is visible in seconds. For a check, state the pass and the fail.
5. **Give the cancel line with the launch line.** For BBL that is `bbl_cancel.sh`, which cancels
   dependents first. A plain `scancel` of the array satisfies every `afterany` dependency and
   releases the solve onto a partial set.
6. **One command per code block**, each complete without the others.
7. **Mark each claim** as locally tested or cluster-only, so the user knows what the first real
   run is still proving.

## Designs, tags and shard counts (BBL)

- The discount factor and the horizon come from `bbl_discount.env`. They are not on command lines,
  and every run's banner prints each value with its source. Check the banner line in the log.
- A multi-start tag has the shape `_ms<digits>`, the only shape `make_bbl_cost_tables.py` reads as
  multi-start. Its single-curve companion has used `_sc<digits>`. A probe tag contains `probe`.
- A new design, or code that changes what psi means, gets a new tag. The launcher refuses a tag
  whose `psi_starts` records another beta or T; do not suggest `--force` to get past it.
- The shard count is baked into every filename and never changes between a run and its re-runs
  (production uses 300). Re-running a subset keeps the same count.
- CPU- and GPU-computed shards of one routine agree to 3e-14 (2026-09-20), so they can be mixed.
- The cost tables report E3 and E4 (`link_ests` in `config/routines.toml`). E1 and E2 cost
  parameters feed only the counterfactuals. Do not widen `link_ests` to add columns: other code
  reads it as "routines with a single-index link".

## What can be tested where

Locally: `bash -n`, `python -m py_compile`, the Julia parse check, unit tests on fake shard files,
and dry runs (`--dry-run`, `CL_DRYRUN=1`).

Only on the cluster: anything that needs a real shard (185 to 201 GiB peak at T = 250, plus the GPU
sysimage and CUDA), the random-coefficient counterfactuals on the full panel, packing, the QOS,
and wall times. No local smoke of a shard exists, however small.

A check that needs a real shard is therefore written as a probe step: one exact `sbatch` line and
a stated pass or fail. `BBL_RUNBOOK.md` sections 1 and 6 are the model. When briefing a subagent
on cluster code, say this explicitly and ask it to label each claim.

## Uploads

- The full bundle: `cluster_upload.py` checks every input in the manifest and refuses one older
  than the artefact it derives from; `--stage` builds the two zips. `--allow-stale` is the user's
  call, never a default.
- A small code update: a per-file table with size and md5 plus the one-line `md5sum` check, as in
  `BBL_RUNBOOK.md` M0. Upload single files; never unzip the whole code bundle while jobs are queued.
- `.sh` files must have LF endings: a CRLF shebang breaks every job. Count carriage returns with
  `tr -cd '\r' < file | wc -c`. In this Git Bash, `grep $'\r'` can match the letter r and has
  produced a false alarm. When Python writes a script, open the file with `newline=""`.
- After every code-bundle upload the Julia manifest is re-resolved by one job that runs alone
  (`cluster/LAUNCH_SEQUENCE.txt` step 3), and `scripts/logs` must exist.
- Compute nodes have no internet. `scrape_forward_rf.py` runs locally and its CSVs are uploaded;
  the curve must reach the horizon the run simulates.
- The panel travels as `market_panel.parquet` only, never beside the CSV.
- `data/input` holds what was uploaded and `data/output` what the cluster produced. An archive
  pattern never names an input, and every archive call passes `--copy`: an early archiver moved
  uploaded inputs into a zip and deleted them.

## Symptoms that look like something else

| What the user sees | What it usually is | What to do |
|---|---|---|
| A job has no log file at all | It was cancelled as `DependencyNeverSatisfied` because something upstream failed, or `scripts/logs/` is missing | Read the gate json before it, then the `.err` of the job it depended on. Never debug the silent job |
| `squeue` shows far fewer jobs than the dry run printed | The design: each phase boundary submits a `pipe_next_<phase>` job that re-enters `pipeline_all.sh` once its inputs exist | Read `logs/pipe_next_*.out` at each hand-off |
| Jobs pending with `QOSMaxJobsPerUserLimit` | The per-user cap: gpu_h200 6 jobs and 16 GPUs, gpu_h100 12 jobs and 32 GPUs, shared by all routines | Slow, not broken. Fewer CPUs per job does not help; more shards per job does |
| Submission rejected with `QOSMaxWallDurationPerJobLimit` | A wall above the two-day cap | Lower the wall |
| A GPU job refuses in about 20 s with `CUDA.functional() == false` | The sysimage is missing or was built for another node type, so tasks race the shared precompile cache | Rebuild it (`ENV_STEP=sysimage_gpu`). The fallback-to-CPU switch holds a GPU idle and is the user's call |
| `Unable to find compatible target in cached code image` | An image built on one CPU type loaded on another: the gpu_h200 image is rejected on `day`, and `day` mixes two CPU generations | Use the CPU image, built and run with `--constraint=cpugen:turin` |
| A Python step dies in seconds with `ModuleNotFoundError` | The conda env is not the one that carries the stack | Preflight section (5) names the env that imports it and prints the `export CONDA_ENV=` line. Trust that over any note (`dep_comp_blp` on 2026-08-20) |
| `Package X is required but does not seem to be installed`, manifest matches | The packages were installed into `~/.julia`, not the project depot | The resolve job (`env_job.sh`, `ENV_STEP=resolve`), alone |
| G0 says `RESOLVE REQUIRED: yes` right after an upload | The bundle's manifest, resolved on the local Julia, overwrote the cluster's | The same resolve job; submit nothing else until it ends |
| The first RC stage is far slower than usual | The logit delta's length does not match the demand parquet, and the engine starts cold without saying so | Gate G4's json |
| The solve or tables start right after an array was cancelled | `afterany` counts a cancellation as done | Cancel chains with `bbl_cancel.sh` only |

## Reading what comes back

- The whole chain: the gate jsons in order (G0, G1, G2, G4, then G3 and G7). The first failing
  gate explains everything downstream of it. G1 reports the cluster's phi-hat against the promoted
  values listed in `cluster/RUNBOOK.md`; a move does not stop the chain but is worth stopping for.
- BBL: `bbl_status.sh --routines "<k>" --psi-tag <tag>` prints one line per routine. `wait (...)`
  means the chain is handling it; `ACT: ...` gives the exact command. `--timing` after about
  25 minutes gives GO or NO-GO against the probe's shard time.
- After a solve: in `cost_params_*.json`, `n_shards_found` equals `n_shards_expected`, and the
  `run` block records the beta and T the psi was simulated under. A cost estimate without that
  equality is not to be used.
- Once the archive job has finished, the `post-run-refresh` skill covers download and ingest.

## Decisions that belong to the user

GO or NO-GO after a probe; promoting cost parameters (`--promote`); any `--force`, `--allow-stale`
or `ALLOW_*` override; relaunching unpacked; which routines run; cancelling a running chain. Bring
each as a question with a recommendation, and keep preparing whatever does not depend on the answer.
