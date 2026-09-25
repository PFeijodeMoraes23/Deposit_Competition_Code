# BBL runbook: the forward-sim cost stage on Bouchet

What to type, what runs by itself, and what to do when it stops. The general cluster reference is
`cluster/RUNBOOK.md`. This page covers only the BBL stage (`bbl_run.sh`).

Everything below runs from the scripts checkout:

```
cd ~/project_pi_mf2263/pf382/dep_comp/scripts
```

**Rule 0 still holds.** The login node runs only `sbatch`, `squeue`, `sacct`, `sinfo`, `ls`, `cat`
and the three helpers built from those commands: `bbl_status.sh`, `bbl_sizing.sh` and
`bbl_cancel.sh` (the last also runs `scancel`). `bbl_run.sh` runs `cluster_preflight.sh`, which loads
Julia, so it is always **submitted** as a small job (`sbatch --wrap`), never `bash`-run on the login
node.

---

## The commands, in order

```
# 1. the MEMORY PROBE at T=250 (two concurrent jobs), then its sizing once both have finished
sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out \
  --wrap "bash bbl_run.sh --probe --routines 3 --shards 300 --multi-start --n-paths 1"
bash bbl_sizing.sh

# 2. the full E3/E4 launch: paste the sbatch line bbl_sizing.sh printed. Its shape:
sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out \
  --wrap "bash bbl_run.sh --routines '3 4' --fwd-gpu --multi-partition --shards 300 --multi-start --n-paths 1 --psi-tag _ms2 --no-warmup --no-polfunc --mem-h200 <M>G --mem-h100 <M>G --pack-h200 <k> --pack-h100 <k>"

# 3. watching it (login-safe, one screen)
bash bbl_status.sh --routines "3 4" --psi-tag _ms2
```

`cat logs/bbl_submit_<jobid>.out` shows the submission graph each wrap job printed.

**beta and T are not on these command lines.** They come from `bbl_discount.env`
(`BBL_BETA=0.979`, `BBL_HORIZON=250`, decided 2026-09-24): the per-quarter rate-implied discount
factor and the shortest horizon holding 99.5% of the discount weight. Every script reads that one
file. `--beta` / `--horizon` (or `BETA=` / `HORIZON=`) override it, and the banner of every run
prints each value with its source, for example `beta=0.979 (bbl_discount.env)  T=250 (bbl_discount.env)`.
A run with no registry and no override is refused. The counterfactual scripts (`cf_run.sh`,
`cf_eq_run.sh`, `cf1/cf3/cf5/cf6`) take **beta** from the same file. Their horizon stays their own
(50 quarters), because it is not the BBL forward-simulation horizon.

**The new run's tag is `_ms2`.** The beta=0.9 / T=50 run is `_ms1`, and its files stay on the
cluster untouched. Every name the new run writes (psi, cost_params, tables, job names, the archive)
carries `_ms2`, and the solve, the tables and the slim archive match a tag only as the whole
remainder of a name, so the two runs can never mix.

---

## 0. Before the first launch (once per upload)

1. **Upload these files** to `scripts/` one by one (never the whole code bundle while jobs are
   queued). Run `chmod +x *.sh` afterwards. The files are `bbl_run.sh`, `bbl_job.sh`,
   `bbl_status.sh`, `bbl_sizing.sh`, `bbl_cancel.sh`, `cluster_lib.sh`, `cluster_archive.sh`,
   `bbl_fwd_sim.jl`, `cf_psi_basis.jl`, `cf1_franchise.jl`, `cf3_equilibrium.jl`, `bbl_solve.py`,
   `bbl_shards.py`, `cf_run.sh`, `cf_eq_run.sh`, **`bbl_discount.env`**, `BBL_RUNBOOK.md` and
   `cluster/RUNBOOK.md` + `cluster/upload_manifest.txt` (to `scripts/cluster/`). The sysimages do
   **not** need a rebuild: they bake packages, not these scripts.
2. **The Focus curves must reach h = 250.** `bbl_run.sh` refuses a run whose T exceeds the curve
   its sim reads (the **shortest** vintage of `forward_rf_vintages.csv` under `--multi-start`), and so
   does every fwd job, both in the shell and inside `bbl_fwd_sim.jl`. Otherwise the sim would repeat
   the last quoted rate for every quarter past the curve. Upload the h=250 builds to `data/input/`:
   `forward_rf_vintages.csv` (`python scrape_forward_rf.py --vintage-from 2016Q1 --vintage-to 2024Q4 --horizon 250`)
   and `forward_rf_qoq.csv` (`--horizon 250 --start 2026Q1`). A multi-start run does not read
   `forward_rf_qoq.csv`, so a short one there is only a one-line note. `ALLOW_RF_PAD=1` accepts
   padding deliberately.
3. **Node and QOS shapes** (read-only; the defaults in `cluster_lib.sh` §11 were taken from these
   on 2026-09-24):
   ```
   sinfo -p gpu_h100,gpu_h200 -N -o "%N %m %c %G" | sort -k2 -u
   sacctmgr -n show qos part_gpu_h200,part_gpu_h100 format=Name,MaxJobsPU,MaxTRESPU%30,MaxWall
   ```
   Expected: gpu_h200 has 2,043,833 MB, 48 CPUs and 8 H200 per node, with a cap of 6 jobs / 16
   GPUs; gpu_h100 has 1,000,000 MB, 48 CPUs and 4 H100 per node, with a cap of 12 jobs / 32 GPUs.
   If either differs, export the override named in `cl_bbl_part` (for example
   `BBL_H100_NODE_MEM_MB=...`) in the `--wrap` string and when running `bbl_sizing.sh`.

---

## 1. The memory probe (cluster-only, required before the full launch)

**Why.** Memory grows with T: `cf_shares_path` and `simulate_deposits` hold N x T paths. At T=50
the shards peaked at 164 GiB median and 179 GiB max, so `MEM=256G` held. At T=250 nobody knows,
and 600 shards should not be the ones to find out. Packing (k shards per job, one GPU each) is
what lifts GPU concurrency from 18 to 46 under the job-count QOS. It pays only if k shards on one
node are not much slower than one alone, which also depends on T.

**What it submits** (routine 3, tag `_probe250`; no sweep, solve, tables or archive). Two jobs run
**concurrently** on gpu_h200, both with generous memory so neither can OOM:

* **(a) packed:** shards 0-3 in ONE job, `PROBE_PACK=4` x `PROBE_MEM_PACKED=450G`
  (`--gpus=h200:4 --mem=1843200M --cpus-per-task=32`). Four processes, each on one device of the
  job's `CUDA_VISIBLE_DEVICES`, each with its own log `logs/bbl_fwd_E3_probe250_shard<i>_<jobid>.out`.
* **(b) unpacked:** shard 4 alone, `PROBE_MEM_UNPACKED=500G` (`--gpus=h200:1 --mem=500G`).

Both use production shard ids of N_SHARDS=300 (50 shocks over 300 shards), so a shard's elapsed
time is a real production shard time at T=250. The wall is the derived one x `BBL_PROBE_WALL_MULT`
(2): 08:30:00. The packed job needs a nearly empty h200 node (1.8 TB of 2 TB), so it may queue
longer than the unpacked one. Every shard logs its **own** peak RSS on one line,
`[BBL] shard i peak RSS X GiB`, because sacct's MaxRSS of a packed job sums its four children.

**Read it** once both jobs have left `squeue` (login-safe):

```
bash bbl_sizing.sh                  # newest probe; --psi-tag _probe250 names it
```

It lists each shard (rc, elapsed, GPU kernel yes/no, peak RSS) and derives:

| quantity | rule |
|---|---|
| `MEM` | ceil(max per-shard peak RSS over all 5 shards x **1.15**) GiB |
| cap per partition | min(GPUs per node, floor(**0.85** x node MB / MEM), floor(48 CPUs / 8 threads)) |
| `PACK_H200`, `PACK_H100` | the k <= cap that runs the most shards at once under the QOS, k x min(MaxJobsPU, floor(MaxGPU / k)); ties go to the smaller k |
| packing go / no-go | GO when the slowest packed shard <= **1.3** x the unpacked shard at T=250. NO-GO means `--pack 1` everywhere (one shard per job, the configuration validated against CPU to 3.4e-14) |
| gpu_h100 | used only if its cap >= 1 and the peak GPU memory (nvidia-smi samples) <= 85% of an H100's 80 GB; otherwise `--partitions gpu_h200` |
| `--shard-time` | added only when the measured shard x 1.5, rounded up to 30 min, exceeds the derived wall |

It ends with the exact `sbatch ... --wrap "bash bbl_run.sh ..."` line for the E3/E4 launch
(`BBL_LAUNCH_CMD=` repeats it on one line). `--launch-tag` and `--routines` change what that
command launches.

If a probe shard failed, `bbl_sizing.sh` prints `PROBE INCOMPLETE` and no sizing. Read
`sacct -j <jobid> -X -o JobID,State,Elapsed,MaxRSS`. On `OUT_OF_MEMORY`, re-probe with more memory
(`PROBE_MEM_UNPACKED=900G PROBE_MEM_PACKED=480G`, exported inside the `--wrap` string). On
`TIMEOUT`, re-probe with `BBL_PROBE_WALL_MULT=4`. A re-probe under the same tag is refused while
the first probe's jobs are alive, and it replaces the probe files when it finishes.

---

## 2. The full run: E3 and E4 first

E3 and E4 are the tabulated routines, so they go first, as one command: the line
`bbl_sizing.sh` printed (section 1). `--no-polfunc` makes the run use the `polfunc_fitted.csv`
already on disk, the same one the probe used. Drop it to re-fit the policy first.

**What it submits** (`bash bbl_run.sh --dry-run ...` prints exactly this and is login-safe; the
example is `--pack-h200 4 --pack-h100 4 --mem-h200 240G --mem-h100 240G`):

```
bbl_fwd_E3_ms2 [gpu_h200] PACK 4: shards 0-99,    25 jobs, %4 --+
bbl_fwd_E3_ms2 [gpu_h100] PACK 4: shards 100-299, 50 jobs, %8 --+--afterany--> bbl_sweep_E3_ms2 (#0, held until the solve exists)
                                                                                    | afterok
                                                             bbl_solve_E3_ms2 <-----+   (always afterok the FINAL sweep)
   ... the same for E4 ...
bbl_solve_E3_ms2 + bbl_solve_E4_ms2 --afterok--> bbl_tables_ms2 --afterok--> bbl_zip_ms2 (slim, tag _ms2)
```

* The two partitions get **disjoint** index blocks sized to their concurrent capacity (16 : 32
  shards in the example).
* The throttle is `min(MaxJobsPU, floor(MaxGPU / PACK))` per partition and is logged. The QOS caps
  are per **user**, so E3 and E4 share them: jobs beyond the cap wait as `QOSMaxJobsPerUserLimit`.
* Wall, derived and logged: 26 min (the measured max at T=50) x T/50 x shard size x 1.3 contention x
  1.5 safety, rounded up to 30 min: **04:30:00** at T=250 (253.5 min), unless `--shard-time` is given.
* Memory: without `--mem*` a T=250 launch prints a loud WARNING that the default 256G is the T=50
  sizing. Always pass what `bbl_sizing.sh` printed.
* The archive is **slim**: this tag's psi, every `cost_params_*.json`, the `tab_bbl_*` tables and
  the `polfunc_*` outputs. It leaves out `_ms1` psi, `_probe*`/bench files and dispatch state.
  `--zip-full` restores the whole-folder archive.
* `--promote` is not in the command: the solve writes `cost_params_E{k}_spec_12_extended_ms2.json`
  and leaves the untagged file the counterfactuals read alone. Promoting is a separate decision.

### E1 and E2 afterwards (a separate, later command)

They feed only the counterfactuals. Their panels are **larger** than E3's (E1 has 796,154 rows),
and RSS scales with the panel, so probe one of them first. Then launch with the sizing it prints:

```
sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out \
  --wrap "bash bbl_run.sh --probe --routines 1 --shards 300 --multi-start --n-paths 1 --psi-tag _probe250e1"
bash bbl_sizing.sh --psi-tag _probe250e1 --routines "1 2" --launch-tag _ms2
#   -> paste its sbatch line (it keeps --no-polfunc: a re-fit would rewrite the CSV that running
#      E3/E4 tasks read)
```

Their tables job re-renders the tables with all four routines on disk. Their archive holds every
`cost_params_*.json` and this tag's psi.

---

## 3. What the sweep does by itself

`bbl_sweep_E<k>_ms2` is a 30-minute, 4 GB `day` job that runs after all of a routine's fwd jobs have
ended, whatever their state:

1. It asks `bbl_shards.py coverage` which of the N shards really exist. A shard counts only if it is
   a **readable parquet with the psi columns** under the final name (a truncated file counts as
   missing), and, on a fresh full launch, only if it is newer than the launch. psi_eq and
   psi_starts must be present, and psi_starts must record this run's beta and T. If they are not,
   shard 0 (their only writer) is re-run.
2. **Complete:** it exits 0 and the solve runs.
3. **Gaps, retries left** (default 3, `--max-retries`): it re-submits **exactly** the missing
   indices, repacked as a list (for example `3 50 51 52 | 53 54 55 56 | ...`), with the same
   N_SHARDS, tag and settings (read from `data/output/bbl/.dispatch/<key>/context.env`) and a wall
   of x1.5 per attempt. A short last chunk requests only the GPUs it uses. Shards that another live
   job is already computing are waited for, never duplicated. It then chains the next sweep
   `afterany` those jobs and **moves the pending solve's `afterok` onto it**
   (`scontrol update ... Dependency=`). The solve keeps its job id, so the tables, the archive and
   any CF job chained on it stay valid.
4. **Retries spent:** it exits 1 with the indices named. The solve is cancelled rather than run on a
   partial set, and the tables and archive are cancelled with it.
5. **A problem no re-run fixes:** a second shard-count family beside this one, or psi written under
   another beta or T. It exits 2 at once.

Every step appends one line to `.dispatch/<key>/events.log`, which is what `bbl_status.sh` reads.

**Writes are atomic.** `bbl_fwd_sim.jl` writes every psi file (`psi_dev_*`, `psi_eq_*`,
`psi_starts_*`) under a temporary name `.tmp_<name>.<jobid>_<pid>` and renames it into place. A
killed or duplicated task can leave `.tmp_` debris, never a truncated file under the final name.
The debris matches no psi glob and is never archived.

**No duplicate submissions.** `bbl_run.sh` refuses to launch a routine/tag while a sweep or solve of
it is queued, or while a live fwd job claims any of the same shards (each fwd job's shards are
recorded in `.dispatch/<key>/<jobid>.map`). A live job with no claim file counts as claiming
everything. `--force` overrides the check. The sweep's own re-runs pass by construction, because
they take only unclaimed indices.

---

## 4. Watching

```
bash bbl_status.sh                               # newest non-probe run, tag auto-detected
bash bbl_status.sh --routines "3 4" --psi-tag _ms2
bash bbl_status.sh --rss                         # + MaxRSS of completed fwd jobs, per-shard peaks of packed ones
```

One line per routine: shards on disk / N, fwd jobs running/pending/failed, the sweep's state and
retries used (`r1/3`), the solve's state, and the **next action**. `wait (...)` means the chain is
handling it. `ACT: ...` gives the exact command to run. The last line shows the tables and archive
jobs.

Reading the tables: `cat logs/bbl_tables_ms2_<jobid>_*.out`. The archive lands in
`data/output/download/bbl_outputs_<jobid>.zip`. The solve's `cost_params` json records, in its
`run` block, the beta and T the psi were simulated under and where that came from
(`psi_starts_<tag>.json`).

---

## 5. Manual recovery

| status says | do |
|---|---|
| `ACT: read logs/bbl_sweep_...` (retries spent, or a problem) | `sacct --name=bbl_fwd_E3_ms2 -X -o JobID,State,Elapsed,MaxRSS` and the shard logs say why (TIMEOUT, OOM, a traceback). Fix it (for example `--shard-time`, `--mem-h200`), then `--repair` (below) |
| `ACT: scontrol release <id>` | a sweep stayed held (release failed after the solve was re-targeted): `scontrol release <id>` |
| `ACT: read logs/bbl_solve_...` | the solve failed or was cancelled: read its `.out`/`.err`, fix, then `--repair` |
| `ACT: nothing queued, n/N shards` | the chain is gone with gaps left: `--repair` |

**`--repair`** re-attaches a dead chain without relaunching anything that exists. It submits a sweep
immediately (which re-runs whatever is missing, or nothing), then solve -> tables -> archive:

```
sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out \
  --wrap "bash bbl_run.sh --routines 3 --shards 300 --psi-tag _ms2 --multi-start --n-paths 1 --repair --multi-partition --mem-h200 <M>G --mem-h100 <M>G --pack-h200 <k> --pack-h100 <k>"
```

The design (N_SHARDS, T, beta, shocks, the fwd flags) comes from the run's `context.env`, not from
`bbl_discount.env`. A value passed that disagrees with it is refused, and **N_SHARDS can never
change** between a run and its re-runs. Placement (`--multi-partition`, `--pack*`, `--mem*`,
`--shard-time`) comes from the repair's own flags. A run with no `context.env` takes its N from the
files on disk and everything else from the flags and the registry, so pass the original run's
values (`_ms1` needs `--beta 0.9 --horizon 50`). The sweep checks beta/T against `psi_starts`.

**Cancel a chain** (dependents first; a plain `scancel` of the array would release every `afterany`
job onto a partial set):

```
bash bbl_cancel.sh --routines "3 4" --psi-tag _ms2 --dry-run     # list
bash bbl_cancel.sh --routines "3 4" --psi-tag _ms2               # tables/zip -> solves -> sweeps -> fwd
```

It writes a `STOP` marker first, so a sweep caught mid-run re-submits nothing. The next launch or
`--repair` clears it. `--with-shared` also cancels `bbl_warmup` and `bbl_polfunc`.

**A launch is refused** for one of three reasons. A duplicate: wait, cancel, or pass `--force`. A
second shard-count family of the same tag on disk: move it aside or use a new `--psi-tag`. A curve
shorter than T: upload the h=250 build.

---

## 6. Cluster-only checks (none of this can run locally)

A real shard needs ~164-179 GiB at T=50 (more at T=250), the gpu sysimage and CUDA, so every item
below is verified on Bouchet. Each has a command and the result it must show.

1. **The probe itself:** `sacct -j <packed>,<unpacked> -X -o JobID,State,Elapsed` shows both
   COMPLETED, and `grep -h "peak RSS" logs/bbl_fwd_E3_probe250_*` prints **5** lines, one per shard.
   `bash bbl_sizing.sh` ends with a `LAUNCH` block.
2. **Every packed shard used its own GPU:**
   `grep -L "GPU share kernel enabled" logs/bbl_fwd_E3_probe250_shard*_<packed jobid>.out` prints
   **nothing**, and `grep -h '^\[child\] shard=' logs/bbl_fwd_E3_probe250_shard*_<packed jobid>.out`
   shows **4 different** `device=` values. The packed job's log shows `CUDA gate (fatal, ..., 4 device(s))`
   and four `CUDA OK: device` lines.
3. **Atomic writes on real shards:** after the probe, `ls data/output/bbl/.tmp_* 2>/dev/null` prints
   nothing, and `ls data/output/bbl/psi_*_E3_spec_12_extended_probe250*` lists psi_eq, psi_starts
   and psi_dev shards 0-4 of 300. A kill test (the wall expires before the shard can finish):
   ```
   sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out \
     --wrap "bash bbl_run.sh --routines 3 --shards 300 --shard-list 7 --pack 1 --mem 500G --shard-time 00:15:00 --multi-start --n-paths 1 --psi-tag _atomictest --no-warmup --no-polfunc --no-sweep --no-solve"
   ```
   PASS: `sacct` shows TIMEOUT and `data/output/bbl/psi_dev_E3_spec_12_extended_atomictest_shard7of300.parquet`
   does **not** exist. A `.tmp_...` file may exist, which is the point.
4. **psi provenance:** `cat data/output/bbl/psi_starts_E3_spec_12_extended_probe250.json` shows
   `"beta":0.979` and `"T":250`, and every probe shard log has the line
   `[BBL] discount: β=0.979 ← --beta | T=250 ← --horizon` (bbl_run.sh passes both explicitly).
5. **The sweep on a real gap, end to end** (small: 2 shocks, 20 shards, T=50, its own tag; the
   launch deliberately leaves shards 16-19 out):
   ```
   sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out \
     --wrap "bash bbl_run.sh --routines 3 --shards 20 --shocks 2 --shard-list 0-15 --horizon 50 --pack 1 --multi-start --n-paths 1 --psi-tag _sweeptest --no-warmup --no-polfunc --no-tables --no-zip --max-retries 1"
   ```
   PASS: `data/output/bbl/.dispatch/E3_spec_12_extended_sweeptest/events.log` shows
   `SWEEP retry=0 result=resubmit missing=16-19`, then `SWEEP retry=1 result=complete n=20`. While
   the second sweep is pending, `scontrol show job <solve id> | grep Dependency` shows
   `afterok:<second sweep>`, which proves `scontrol update` of a dependency is allowed for our jobs.
   The solve then runs. Its `cost_params_E3_spec_12_extended_sweeptest.json` is a test artifact
   that nothing reads.

---

## 7. Knobs (flag / env)

| knob | default | meaning |
|---|---|---|
| `bbl_discount.env` (`BBL_BETA`, `BBL_HORIZON`) | 0.979, 250 | the design; `--beta` / `--horizon` (or `BETA` / `HORIZON`) override; `BBL_DISCOUNT_ENV` points at another file |
| `--shards` / `N_SHARDS` | 100 | never change between a run and its re-runs; production is 300 |
| `--pack`, `--pack-h200`, `--pack-h100` / `PACK*` | h200 4, h100 3 | shards per job; PACK=1 is one shard per job |
| `--mem`, `--mem-h200`, `--mem-h100` / `MEM*` | 256G (T=50 sizing) | per shard; the job gets PACK x MEM |
| `--multi-partition` / `--partitions` | gpu_h200 | GPU partitions, disjoint index blocks |
| `--shard-time` / `SHARD_TIME` | derived | fixes the fwd wall |
| `BBL_SHARD_MIN_T50`, `BBL_PACK_CONTENTION`, `BBL_WALL_SAFETY`, `BBL_WALL_ROUND_MIN` | 26, 1.3, 1.5, 30 | the wall derivation |
| `--max-retries` / `BBL_MAX_RETRIES`, `BBL_RETRY_WALL_MULT` | 3, 1.5 | sweep retries and their wall growth |
| `--sweep-partitions` | the launch's | where the sweep's re-runs go (for example the other GPU partition) |
| `PROBE_PACK`, `PROBE_MEM_PACKED`, `PROBE_MEM_UNPACKED`, `BBL_PROBE_WALL_MULT` | 4, 450G, 500G, 2 | the memory probe |
| `BBL_MEM_HEADROOM`, `BBL_NODE_FRACTION`, `BBL_PACK_MAX_RATIO` | 1.15, 0.85, 1.3 | `bbl_sizing.sh` rules |
| `ARRAY_THROTTLE` | derived | overrides the per-partition throttle (empty = none) |
| `ALLOW_RF_PAD=1` | off | accept T beyond the curve (a flat tail) |
| `--zip-full` | slim | archive the whole bbl folder |
| `--no-sweep`, `--no-solve` | on | the solve goes afterany the arrays / no solve at all (tests) |
