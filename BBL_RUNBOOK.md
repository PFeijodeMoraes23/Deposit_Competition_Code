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

## Submitting and watching

This replaces the dated launch walkthroughs of 2026-09-25 (`_ms979`) and 2026-09-30 (`_ms982` and
`_sc982`). The 2026-10-13 run (`_ms983` and `_sc983`) was launched on a fresh tree by the stage C
launch package, whose BBL step calls `bbl_run.sh` with the flags shown below.

**Every command is one line.** It is submitted as a small job and never `bash`-run on the login
node. Paste it whole and never split it with a trailing backslash: on 2026-09-24 a broken
continuation made `bbl_run.sh` submit its DEFAULT chain. A submission returns at once. Read its log
afterwards with `tail -n 40 logs/<name>_<jobid>.out` and watch with `squeue` or `bbl_status.sh`;
never use a command that holds the terminal.

The order of a new run:

1. **Memory probe** (section 1) whenever the horizon, the state space or the simulation's memory
   use changes. `bbl_sizing.sh` turns it into MEM and PACK.
2. **Shard probe** of new simulation code. `--shard-probe --array-spec <i> --psi-tag _probe<name>`
   submits exactly those shards, one per job, and nothing else: no warmup, polfunc, sweep, solve,
   tables or archive. It is refused without `--array-spec`, without `probe` in the tag, or when the
   shard's file already exists under that tag (`--force` overwrites). `--repair` refuses every
   probe tag. Example (E3, shard 1 of 300, on gpu_h200):

   ```
   cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_probe_submit -o logs/bbl_probe_submit_%j.out --wrap "bash bbl_run.sh --shard-probe --routines 3 --shards 300 --array-spec 1 --multi-start --n-paths 1 --psi-tag _probe<name> --partitions gpu_h200"
   ```

3. **Node check**, right before the launch (login-safe): how many nodes could start a job now, per
   partition and PACK, at 240G and one free GPU per shard.

   ```
   scontrol -o show node | awk '{delete v; for(i=1;i<=NF;i++){j=index($i,"="); if(j) v[substr($i,1,j-1)]=substr($i,j+1)} p=(v["Partitions"]~/gpu_h100/)?"h100":((v["Partitions"]~/gpu_h200/)?"h200":""); if(p==""||v["State"]~/DOWN|DRAIN|FAIL|MAINT|RESERVED/) next; tot=(p=="h100")?4:8; ag=0; if(match(v["AllocTRES"],/gres\/gpu=[0-9]+/)) ag=substr(v["AllocTRES"],RSTART+9,RLENGTH-9)+0; fg=tot-ag; fm=(v["RealMemory"]-v["AllocMem"])/1024; for(k=1;k<=4;k++) if(fg>=k && fm>=240*k) n[p" pack="k]++} END{for(x in n) print x": "n[x]" node(s) could start a job now"}' | sort
   ```

   Use PACK 4, the code default, when both partitions list `pack=4`. Otherwise pass
   `--pack-h200 2 --pack-h100 2`.
4. **Dry run** (optional). Submit the launch line with `--dry-run` added; then
   `grep -h '^BBL_JOB_NAMES=' logs/bbl_submit_<jobid>.out` lists the jobs it would submit.
5. **Launch**, E3 and E4, PACK 2 (`--no-warmup` and `--no-polfunc` assume the sysimages and
   `polfunc_fitted.csv` are already in place):

   ```
   cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out --wrap "bash bbl_run.sh --routines '3 4' --fwd-gpu --multi-partition --shards 300 --multi-start --n-paths 1 --psi-tag <tag> --no-warmup --no-polfunc --pack-h200 2 --pack-h100 2 --mem-h200 240G --mem-h100 230G --shard-time 01:45:00"
   ```

6. **Watching**: `bash bbl_status.sh --routines "3 4" --psi-tag <tag>`.

**beta and T are not on these command lines.** They come from `bbl_discount.env`
(`BBL_BETA=0.979`, `BBL_HORIZON=250`, decided 2026-09-24): the per-quarter rate-implied discount
factor and the shortest horizon holding 99.5% of the discount weight. Every script reads that one
file. `--beta` / `--horizon` (or `BETA=` / `HORIZON=`) override it, and the banner of every run
prints each value with its source, for example `beta=0.979 (bbl_discount.env)  T=250 (bbl_discount.env)`.
A run with no registry and no override is refused. The counterfactual scripts (`cf_run.sh`,
`cf_eq_run.sh`, `cf1/cf3/cf5/cf6`) take **beta** from the same file. Their horizon stays their own
(50 quarters), because it is not the BBL forward-simulation horizon.

**Tags.** A multi-start run's tag has the shape `_ms<digits>`, the only one
`make_bbl_cost_tables.py` reads as a multi-start tag; the single-curve companion is `_sc<digits>`.
Every name a run writes (psi, cost_params, tables, job names, the archive) carries the tag, and the
solve, the tables and the archive match a tag only as the whole remainder of a name, so two runs
never mix. A launch under a tag whose `psi_starts` records another beta or T is refused. Tags so
far:

- `_ms1`: beta 0.9, T 50.
- `_ms979`: beta 0.979, T 250, 2026-09-25.
- `_ms981`.
- `_ms982` / `_sc982`: 2026-09-30, on the refitted policy function.
- `_ms983` / `_sc983`: 2026-10-13, on the demand fit with the D-type dummy and the demand data fixes.

---

## 0. Before the first launch (once per upload)

1. **Upload the files the run needs** (`cluster/upload_manifest.txt`), one by one (never the whole code bundle while
   jobs are queued), and check their md5 there. The sysimages do **not** need a rebuild: they bake
   packages, not these scripts.
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
example is the quiet-cluster `--pack-h200 4 --pack-h100 4 --mem-h200 240G --mem-h100 230G`; today's
PACK 2 graph has twice as many jobs of 2 shards each):

```
bbl_fwd_E3_ms983 [gpu_h200] PACK 4: shards 0-99,    25 jobs, %4 --+
bbl_fwd_E3_ms983 [gpu_h100] PACK 4: shards 100-299, 50 jobs, %8 --+--afterany--> bbl_sweep_E3_ms983 (#0, held until the solve exists)
                                                                                    | afterok
                                                             bbl_solve_E3_ms983 <-----+   (always afterok the FINAL sweep)
   ... the same for E4 ...
bbl_solve_E3_ms983 + bbl_solve_E4_ms983 --afterok--> bbl_tables_ms983
bbl_solve_E3_ms983 + bbl_solve_E4_ms983 --afterok--> bbl_zip_ms983 (slim, tag _ms983; it does not wait for the tables)
```

* The two partitions get **disjoint** index blocks sized to their concurrent capacity (16 : 32
  shards in the example).
* The throttle is `min(MaxJobsPU, floor(MaxGPU / PACK))` per partition and is logged. The QOS caps
  are per **user**, so E3 and E4 share them: jobs beyond the cap wait as `QOSMaxJobsPerUserLimit`.
* Wall: **01:45:00**, the default on GPU (`BBL_SHARD_TIME_DEFAULT`), from the T=250 probe: 37 min
  x 1.37 (T=50 max/median) x 1.3 (packing) x 1.5 (safety). `--shard-time` overrides it;
  `--shard-time derive` selects the T-based formula (26 min x T/50 x shard size x 1.3 x 1.5 ->
  04:30:00 at T=250), kept only as a fallback: the probe took x1.8 for 5x the horizon, not x5.
* Memory: **240G** per shard on gpu_h200 and **230G** on gpu_h100, the defaults (`BBL_MEM_DEFAULT`,
  `BBL_MEM_DEFAULT_H100`), from the probe's 200.75 GiB max; h100's is lower to leave room for
  what SLURM reserves on a packed node. Only a launch beyond T=250 without `--mem*` prints the
  WARNING.
* The archive is **slim**: this tag's psi and `cost_params_*<tag>.json`, the `tab_bbl_*` tables and
  the `polfunc_*` outputs. It leaves out other tags' psi and cost_params (`_ms1`, the untagged
  file the counterfactuals read), `_probe*`/bench files and dispatch state.
  `--zip-full` restores the whole-folder archive.
* `--promote` is not in the command: the solve writes `cost_params_E{k}_spec_12_extended_ms983.json`
  and leaves the untagged file the counterfactuals read alone. Promoting is a separate decision.

### E1 and E2 afterwards (a separate, later command)

They feed only the counterfactuals. Their panels are **larger** than E3's (E1 has 796,154 rows),
and RSS scales with the panel, so probe one of them first. Then launch with the sizing it prints:

```
sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out --wrap "bash bbl_run.sh --probe --routines 1 --shards 300 --multi-start --n-paths 1 --psi-tag _probe250e1"
bash bbl_sizing.sh --psi-tag _probe250e1 --routines "1 2" --launch-tag _ms983
#   -> paste its sbatch line (it keeps --no-polfunc: a re-fit would rewrite the CSV that running
#      E3/E4 tasks read)
```

Their tables job re-renders the tables with all four routines on disk. Their archive holds every
`cost_params_*_ms983.json` (all four routines) and this tag's psi.

---

## 3. What the sweep does by itself

`bbl_sweep_E<k>_ms983` is a 30-minute, 4 GB `day` job that runs after all of a routine's fwd jobs have
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
bash bbl_status.sh --routines "3 4" --psi-tag _ms983
bash bbl_status.sh --rss                         # + MaxRSS of completed fwd jobs, per-shard peaks of packed ones
```

One line per routine: shards on disk / N, fwd jobs running/pending/failed, the sweep's state and
retries used (`r1/3`), the solve's state, and the **next action**. `wait (...)` means the chain is
handling it. `ACT: ...` gives the exact command to run. The last line shows the tables and archive
jobs.

Reading the tables: `cat logs/bbl_tables_ms983_<jobid>_*.out`. The archive lands in
`data/output/download/bbl_outputs_<jobid>.zip`. The solve's `cost_params` json records, in its
`run` block, the beta and T the psi were simulated under and where that came from
(`psi_starts_<tag>.json`).

---

## 5. Manual recovery

| status says | do |
|---|---|
| `ACT: read logs/bbl_sweep_...` (retries spent, or a problem) | `sacct --name=bbl_fwd_E3_ms983 -X -o JobID,State,Elapsed,MaxRSS` and the shard logs say why (TIMEOUT, OOM, a traceback). Fix it (for example `--shard-time`, `--mem-h200`), then `--repair` (below) |
| `ACT: scontrol release <id>` | a sweep stayed held (release failed after the solve was re-targeted): `scontrol release <id>` |
| `ACT: read logs/bbl_solve_...` | the solve failed or was cancelled: read its `.out`/`.err`, fix, then `--repair` |
| `ACT: nothing queued, n/N shards` | the chain is gone with gaps left: `--repair` |

**`--repair`** re-attaches a dead chain without relaunching anything that exists. It submits a sweep
immediately (which re-runs whatever is missing, or nothing), then solve -> tables -> archive:

```
sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out --wrap "bash bbl_run.sh --routines 3 --shards 300 --psi-tag _ms983 --multi-start --n-paths 1 --repair --multi-partition --pack-h200 2 --pack-h100 2 --mem-h200 240G --mem-h100 230G --shard-time 01:45:00"
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
bash bbl_cancel.sh --routines "3 4" --psi-tag _ms983 --dry-run     # list
bash bbl_cancel.sh --routines "3 4" --psi-tag _ms983               # tables/zip -> solves -> sweeps -> fwd
```

It writes a `STOP` marker first, so a sweep caught mid-run re-submits nothing. The next launch or
`--repair` clears it. `--with-shared` also cancels `bbl_warmup` and `bbl_polfunc`.

**A launch is refused** for one of four reasons. A duplicate: wait, cancel, or pass `--force`. A
second shard-count family of the same tag on disk: move it aside or use a new `--psi-tag`. A curve
shorter than T: upload the h=250 build. A tag whose `psi_starts` records another beta or T (for
example `_ms1` at beta=0.979): use a new `_ms<digits>` tag, or `--force` to overwrite deliberately.

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
   sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out --wrap "bash bbl_run.sh --routines 3 --shards 300 --shard-list 7 --pack 1 --mem 500G --shard-time 00:15:00 --multi-start --n-paths 1 --psi-tag _atomictest --no-warmup --no-polfunc --no-sweep --no-solve"
   ```
   PASS: `sacct` shows TIMEOUT and `data/output/bbl/psi_dev_E3_spec_12_extended_atomictest_shard7of300.parquet`
   does **not** exist. A `.tmp_...` file may exist, which is the point.
4. **psi provenance:** `cat data/output/bbl/psi_starts_E3_spec_12_extended_probe250.json` shows
   `"beta":0.979` and `"T":250`, and every probe shard log has the line
   `[BBL] discount: β=0.979 ← --beta | T=250 ← --horizon` (bbl_run.sh passes both explicitly).
5. **The sweep on a real gap, end to end** (small: 2 shocks, 20 shards, T=50, its own tag; the
   launch deliberately leaves shards 16-19 out):
   ```
   sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out --wrap "bash bbl_run.sh --routines 3 --shards 20 --shocks 2 --shard-list 0-15 --horizon 50 --pack 1 --multi-start --n-paths 1 --psi-tag _sweeptest --no-warmup --no-polfunc --no-tables --no-zip --max-retries 1"
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
| `--pack`, `--pack-h200`, `--pack-h100` / `PACK*` | 4, 4 (pass 2, 2 when the node check lists no `pack=4`) | shards per job; PACK=1 is one shard per job |
| `--mem`, `--mem-h200`, `--mem-h100` / `MEM*` | 240G (`BBL_MEM_DEFAULT`, the T=250 sizing); 230G on gpu_h100 (`BBL_MEM_DEFAULT_H100`) | per shard; the job gets PACK x MEM; `--mem` sets every partition |
| `--shard-probe` + `--array-spec` | off | the code probe: those shards, one per job, a `probe` tag, nothing chained |
| `BBL_THREADS_PER_SHARD` | 8 | CPUs (Julia threads) per shard: `--cpus-per-task` = PACK x this, PACK=1 included |
| `--multi-partition` / `--partitions` | gpu_h200 | GPU partitions, disjoint index blocks |
| `--shard-time` / `SHARD_TIME` | 01:45:00 on GPU (`BBL_SHARD_TIME_DEFAULT`) | the fwd wall per job; `derive` = the T-based fallback formula (the CPU path always) |
| `BBL_SHARD_MIN_T50`, `BBL_PACK_CONTENTION`, `BBL_WALL_SAFETY`, `BBL_WALL_ROUND_MIN` | 26, 1.3, 1.5, 30 | the wall derivation |
| `--max-retries` / `BBL_MAX_RETRIES`, `BBL_RETRY_WALL_MULT` | 3, 1.5 | sweep retries and their wall growth |
| `--sweep-partitions` | the launch's | where the sweep's re-runs go (for example the other GPU partition) |
| `PROBE_PACK`, `PROBE_MEM_PACKED`, `PROBE_MEM_UNPACKED`, `BBL_PROBE_WALL_MULT` | 4, 450G, 500G, 2 | the memory probe |
| `BBL_MEM_HEADROOM`, `BBL_NODE_FRACTION`, `BBL_PACK_MAX_RATIO` | 1.15, 0.85, 1.3 | `bbl_sizing.sh` rules |
| `ARRAY_THROTTLE` | derived | overrides the per-partition throttle (empty = none) |
| `ALLOW_RF_PAD=1` | off | accept T beyond the curve (a flat tail) |
| `--zip-full` | slim | archive the whole bbl folder |
| `--no-sweep`, `--no-solve` | on | the solve goes afterany the arrays / no solve at all (tests) |
