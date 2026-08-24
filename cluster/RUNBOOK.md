# Bouchet runbook — the consolidated scripts

One page, in order. Everything here runs from the code directory on the **login node**:

```
cd /nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts
```

Only **six** of the thirteen new files are things you type. The other seven are job scripts and one
library that `sbatch` or `source` reaches for you. Every one of the six takes `--dry-run` and `-h`.

| you type | what it does |
|---|---|
| `cluster_preflight.sh` | verdict block; submits nothing |
| `blp_run.sh` | the RC-BLP demand sweep |
| `bbl_run.sh` | the BBL cost stage |
| `cf_run.sh` | the counterfactuals |
| `pipeline_run.sh` | `bbl_run.sh` then `cf_run.sh`, chained by SLURM |
| `cf_eq_run.sh` | the long equilibrium CFs (CF3 / CF5 / CF6) |
| `cluster_archive.sh` | the one archiver |

> **These scripts are new.** The 21 existing `submit_*.sh` / `zip_*.sh` scripts are still present,
> still work, and **have not been modified**. They are the fallback (§Fallback) and are retired only
> after one successful cluster cycle on the new set.

---

## 0. Upload (manual, Bouchet web file browser)

Code bundle → `scripts/`, data bundle → `data/input/`, per `cluster/upload_manifest.txt`.

New shell files to include in the code bundle:

```
cluster_lib.sh        cluster_preflight.sh   env_job.sh
blp_draws_job.sh      blp_stage_job.sh       blp_run.sh
bbl_job.sh            bbl_run.sh             pipeline_run.sh
cf_job.sh             cf_run.sh              cf_eq_run.sh
cluster_archive.sh
```

`cluster_lib.sh` first — every other new script is inert without it and fails at line ~2 with a plain
`No such file or directory`. The 21 existing scripts ship too; they are the fallback. The `.jl` and
`.py` upload set is unchanged.

Cluster convention, unchanged: `data/input` = **uploaded**, `data/output` = **cluster-produced**.

---

## 1. Preflight — always first, submits nothing

```
bash cluster_preflight.sh
```

Read the verdict block top to bottom. Three things it may tell you.

**(a) The Julia module is not there.**

```
ERROR: 'module load Julia/1.11.4-linux-x86_64' FAILED.
  --- parsed candidates ---   Julia/1.10.4-foss-2022b   Julia/1.12.0 ...
```

Pick one and keep it for the session:

```
JULIA_MODULE=<candidate> bash cluster_preflight.sh
export JULIA_MODULE=<candidate>
```

That is the whole fix. Every consolidated script reads the name from `cluster_lib.sh`; nothing else
needs editing.

**(b) `RESOLVE REQUIRED: yes`.**

`Manifest.toml` was resolved under a different Julia (today it records `1.12.6`). Run the one command
it prints, **alone** — concurrent `Pkg.resolve()` on NFS corrupts the manifest:

```
sbatch --partition=day --time=00:30:00 --cpus-per-task=4 --mem=16G \
       --export=ALL,ENV_STEP=resolve env_job.sh
```

Wait for it (`squeue -u $USER`). Submit nothing else meanwhile. Then re-run the preflight.
`env_job.sh` backs the manifest up to `Manifest.toml.bak.<ver>` before touching it, and derives the
accepted version from the **loaded** Julia rather than a hardcoded literal.

If that resolve fails on a package with no release for this Julia: do **not** relax anything on the
cluster. Re-resolve locally under a matching Julia and re-upload `Manifest.toml`. `Project.toml`
compat is already permissive (`julia = "1.10, 1.11, 1.12"`), so this is unlikely — but it would cost
a second manual upload cycle.

**(c) The conda env is wrong.**

```
FAILED: 'costsolve' does not import numpy, pandas, scipy, pyarrow
  ... 'dep_comp_blp' exists and imports the stack cleanly. One-line unblock:
      export CONDA_ENV=dep_comp_blp
```

Or create the documented one:

```
module load miniconda && conda create -y -n costsolve python=3.11 numpy pandas scipy pyarrow statsmodels
```

Do not skip this. The import failure it catches otherwise surfaces ~14 h later, when the `fwd_sim`
array drains — and the solve's `afterok` cascade then cancels `cf1_net` with **no log at all**.

---

## 2. Toolchain jobs — only if step 1 asked for them

```
sbatch --partition=gpu_h200 --gpus=h200:1 --time=04:00:00 --cpus-per-task=8 --mem=64G \
       --export=ALL,ENV_STEP=sysimage_gpu env_job.sh

sbatch --partition=day --constraint=cpugen:turin --time=04:00:00 --cpus-per-task=8 --mem=64G \
       --export=ALL,ENV_STEP=sysimage_cpu env_job.sh
```

Both end with a **fatal load proof** on the node type that will use the image, and both write a
provenance sidecar `<img>.so.json` (julia version, module, `Sys.CPU_NAME`, build host, partition,
`Manifest.toml` sha256, build time). A build whose proof fails exits non-zero and the `.so` is
renamed `.so.failed`, so a broken image never sits on disk looking done.

Both must succeed before any GPU work. **On a fresh cluster, rebuild both** rather than trusting an
image left over from a previous run: an image with no sidecar is UNKNOWN provenance and is refused
(`ALLOW_UNSTAMPED_SYSIMAGE=1` accepts it deliberately).

Two images because a sysimage bakes its build node's CPU target. `blp_sysimage.so` is built on
gpu_h200 (sapphirerapids) and is **rejected on `day` nodes**; `blp_sysimage_cpu.so` is built on
`day` and pins `cpugen:turin`, because `day` mixes turin (AMD) with emeraldrapids (Intel).

---

## 3. BLP

```
bash blp_run.sh --dry-run          # prints every sbatch line and the whole dependency graph
bash blp_run.sh --draws            # first run on a fresh cluster; omit --draws afterwards
bash blp_run.sh
```

Defaults: `--layout grouped`, routines `3 4`, engines `ift`, per-routine memory (600 G for E1/E2,
200 G otherwise), terminal archive on.

```
grouped:  HEAD(sigma+rc2+rc3+rc4+ext1) --afterok--> ext2 --afterok--> extended
all:      sigma -> rc2 -> rc3 -> rc4 -> full -> ext1 -> ext2 -> extended
```

Add `--engines "ift numerical"` for the numerical cross-check at `extended`, `--engines "ift cue"`
(+ `--sset`) for the CUE / Stock-Wright variants — all seeded from the IFT checkpoints and all
written with the `_cue` / `_num` suffix, so nothing IFT wrote is touched.

Result: `data/output/blp_outputs_<jid>.zip` (+ `blp_logs_`, `blp_checkpoints_`).

---

## 4. BBL + CF

```
bash pipeline_run.sh --dry-run
bash pipeline_run.sh
```

or, separately:

```
bash bbl_run.sh
bash cf_run.sh --do "demand_eval cf1 cf1_net cf4"
```

`bbl_run.sh` auto-builds `cluster_processed/` from the newest `blp_outputs_*.zip`, so step 3's
archive does not need downloading first.

`pipeline_run.sh` runs Phase 1a (BBL) + Phase 1b (CF4 no-re-eval, needs no costs) concurrently, then
Phase 2 (CF1 gross, CF4 re-eval, CF1 net). CF1-net is chained `afterok` on the BBL solve jobs, so it
launches the instant the cost params are written. Both CF phases share ONE `cf_warmup` barrier.

`fwd_sim` runs on the H200 by default (that is the code default and it is not silently changed).
`bbl_run.sh` prints an advisory that the step measures as memory-bandwidth bound at ~0% GPU
utilisation; `--fwd-cpu` moves it to `day` / `cpugen:turin`.

---

## 5. Equilibrium CFs — optional, long

```
bash cf_eq_run.sh --mode cf3 --routine 3
bash cf_eq_run.sh --mode cf5 --routine 3 [--selic-shock 0.01] [--base-sigma <path>]
bash cf_eq_run.sh --mode cf6 --routine 3 --merge firmA,firmB [--base-sigma <path>]
```

Sigma layout (one convention, nested):

```
data/output/cf/cf3_jacobi_E{k}_{stage}/
data/output/cf/cf5_E{k}_{stage}/{base,shock}/
data/output/cf/cf6_E{k}_{stage}/{base,merged}/
```

This is the layout the `cf5` / `cf6` archive patterns match. **If FLAT `cf5_base_E*` /
`cf5_shock_E*` directories from an earlier run are on the cluster, archive them with the old
`zip_cf_outputs.sh` before running this driver.** `cluster_preflight.sh` reports their presence; it
does not move them.

---

## 6. Archive and download

```
bash cluster_archive.sh --set foundation --copy
bash cluster_archive.sh --set cf1        --copy
bash cluster_archive.sh --set cf4        --copy
bash cluster_archive.sh --all --copy        # foundation cf1 cf3 cf4 cf5 cf6
```

`--copy` / `--move` is **required**; there is no default and no inference. Use `--copy` while
anything downstream is still running (`cost_params` feeds CF3/CF5/CF6; CF3's `sig_N` feeds CF5).
Switch to `--move` only once the whole chain is finished.

**Never `--move` cf2.** That set owns `psi_dev_E*_shard{i}of{N}.parquet` — a ~100-shard GPU array
with no second copy — plus the `cost_params` four CFs read in place. It is excluded from `--all`,
and `--set cf2 --move` additionally requires `ALLOW_CF2_MOVE=1`.

`--dry-run` lists exactly what would be taken and exits before touching anything.

---

## 7. If a job dies

* **A missing `cf1_net` log** means the job was cancelled `DependencyNeverSatisfied`
  (`--kill-on-invalid-dep=yes` writes no log at all). Read the corresponding `bbl_solve` `.err`,
  not `cf1_net`.
* **A wall-killed BLP chain** resumes by plain resubmission: each stage warm-starts from the previous
  stage's on-disk checkpoint and completed stages are skipped. But the checkpoints were **MOVED**
  into `blp_checkpoints_<jid>.zip` — unzip them back into `data/output` before any resume or
  `--se-only` run.
* **A GPU job that refuses in ~20 s** with `CUDA.functional() == false` has a sysimage/CPU-target
  mismatch. Rebuild the image (step 2). This refusal is deliberate: the alternative is an H200 held
  at ~0% utilisation for the whole wall. `ALLOW_SYSIMAGE_FALLBACK=1` restores the old degrade-to-CPU
  behaviour, loudly.
* **An archive that prints `WARNING: N file(s) failed to archive`** has deleted nothing. Re-run the
  same command.

---

## Fallback — the old scripts, unmodified

| step | fallback command |
|---|---|
| 1 / 2 | `sbatch setup_julia_env.sh` ; `sbatch submit_build_sysimage.sh` ; `sbatch submit_build_sysimage_cpu.sh` |
| 3 | `bash submit_blp_build_and_run_spec12.sh --draws` (the feature-complete old front door) |
| 4 | `bash submit_bbl_cf_all.sh` |
| 5 | `bash submit_cf3_jacobi.sh` / `submit_cf5_passthrough.sh` / `submit_cf6_merger.sh` |
| 6 | `KEEP=1 bash zip_all_cf.sh` |

**Caveat on the fallback.** The old scripts hardcode `Julia/1.11.4-linux-x86_64` at **7** sites and
the `'1.11'` manifest literal at **2** more. If step 1 reports that module is gone, the fallback path
requires editing those 9 sites by hand first:

```
grep -rn 'Julia/1.11.4\|julia_version = "1.11' *.sh
```

The new path needs only `export JULIA_MODULE=...`. Preferring the new scripts is therefore the
**faster** option precisely in the scenario where the toolchain has drifted.

---

## What replaces what — nothing was lost

| old script | replaced by | notes |
|---|---|---|
| `setup_julia_env.sh` | `env_job.sh` `ENV_STEP=resolve` | still the sole `Pkg.resolve()` site; now backs up the manifest, uses the project depot, and derives the accepted version from the loaded Julia |
| `submit_build_sysimage.sh` | `env_job.sh` `ENV_STEP=sysimage_gpu` | + fatal load proof + provenance sidecar |
| `submit_build_sysimage_cpu.sh` | `env_job.sh` `ENV_STEP=sysimage_cpu` | same; its existing load proof is now on both targets and fatal |
| `submit_blp_1_draws.sh` | `blp_draws_job.sh` | same julia call, same `split` + `.sha256` block, verbatim |
| `submit_blp_rc_stage.sh` | `blp_stage_job.sh` | engine allow-list, `+`→`,` stage translation, SE knobs all preserved; the sysimage probe now **refuses** instead of clearing `JULIA_SYS` |
| `submit_blp_rc_all.sh` | `blp_run.sh --layout all` | |
| `submit_blp_rc_grouped.sh` | `blp_run.sh --layout grouped` | now also gets the SE-knob export block and a terminal archive, which it lacked |
| `submit_blp_build_and_run_spec12.sh` | `blp_run.sh` (`--sysimage` `--draws` `--se-only` `--engines` `--sset`) | |
| `submit_blp_2_rc_default.sh` | `blp_run.sh` | the `LAYOUT` switch became `--layout` |
| `submit_bbl.sh` | `bbl_job.sh` | CF_GPU→sysimage pairing, `forward_rf_qoq.csv` refusal, python import probe preserved |
| `submit_bbl_all.sh` | `bbl_run.sh` | |
| `submit_bbl_default.sh` | `bbl_run.sh` | |
| `submit_bbl_cf_all.sh` | `pipeline_run.sh` | phase split, capture pattern, `--export=ALL` inheritance of `CF4_EXACT_NOPIX` preserved; the two `cf_warmup` barriers became one |
| `submit_cf.sh` | `cf_job.sh` | all 13 `CF_STEP` branches named in the header and validated; gains sysimage handling for the `cf3_shard` GPU array |
| `submit_cf_all.sh` | `cf_run.sh` | the six `DO_*` booleans became one `--do "..."` selector |
| `submit_cf3_jacobi.sh` | `cf_eq_run.sh --mode cf3` | |
| `submit_cf5_all.sh` | `cf_eq_run.sh --mode cf5` | nested sigma layout adopted (see step 5) |
| `submit_cf5_passthrough.sh` | `cf_eq_run.sh --mode cf5 --base-sigma <path>` | `--base-sigma` now available in cf5 **and** cf6 |
| `submit_cf6_merger.sh` | `cf_eq_run.sh --mode cf6 --merge "A,B"` | |
| `zip_cf_outputs.sh` | `cluster_archive.sh --set <cf> --copy\|--move` | same add→verify→delete algorithm, verbatim; mode is now explicit |
| `zip_all_cf.sh` | `cluster_archive.sh --all --copy\|--move` | same default set list, cf2 still excluded |
| — | `cluster_lib.sh` | new: one site for `JULIA_MODULE` / `PY_MODULE` / `CONDA_ENV`, `cl_pick_zip`, `cl_cp_dir`, routine defaults, the sbatch wrapper |
| — | `cluster_preflight.sh` | new: the version/sysimage/python/staging verdict block |

21 cluster scripts → 13. `run_sleep_constrained.sh` is local-only and out of scope.

---

## Deliberate changes you should know about

* **Routine defaults.** `3 4` for the RC/CF set, `1 2 3 4` for the full lineup — matching
  `blp_2_rc.jl` `DEFAULT_ROUTINES=[3,4]` and `upload_manifest.txt`'s `#!ROUTINES 1 2 3 4`. Four old
  scripts default to `1 2 5 6 7 8` and one to `5 6 7 8`; that numbering is dead (E5/E6 became E3/E4,
  the joint sieve E7/E8 was deleted). Every run prints where the value came from.
* **Silent CPU fall-back is now a refusal.** A CUDA-mismatched sysimage used to degrade to CPU and
  finish slowly but correctly; it now exits non-zero in seconds, and under `afterok` that cancels the
  rest of the chain. That is the right trade — an H200 held at 0% for 16 h is worse — but a marginal
  image that previously "worked" now blocks the run. `ALLOW_SYSIMAGE_FALLBACK=1` restores the old
  behaviour with a loud log line.
* **The archiver has no default mode.** Under the old scheme the destructive behaviour was the
  default and the safe one needed `KEEP=1`; now neither is reachable by omission.
* **`full` stays in `--layout all` and stays out of `--layout grouped`,** matching today's split.
  `blp_1_estimation.jl:1334` documents it as reproducing `rc4` exactly since σ(ln assets) was
  dropped, and `ext1` warm-starts from `rc4` — but whether to drop it is a computational-semantics
  decision belonging to the `.jl` author, so neither behaviour was changed.
