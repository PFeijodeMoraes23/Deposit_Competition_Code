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

## MORNING LAUNCH 2026-09-25

This section covers the beta = 0.979, T = 250 re-run of E3 and E4 under the tag **`_ms979`**: S = 36
launch quarters, P = 1 rate path, 300 shards, both GPU partitions, **packed 2 per job**. The live
node check at 10:46 found h200 nodes that could start a PACK-1 or PACK-2 job, none that could start
a PACK-4 job, and no free h100 node (M1c has the one-liner). The order is: upload (M0), the probe of
the new rate-cap code on ONE shard and its comparison (M1b), the node check (M1c), the launch (M2).
Every command below is **one line**. Paste each one whole, and never split one with a trailing
backslash. On 2026-09-24 a broken continuation made `bbl_run.sh` submit its DEFAULT chain.

**What was tested where.** LOCAL-TESTED: `bash -n`, `py_compile`, the Julia parse, and the unit
tests on fake shards. The dry runs of this exact launch (PACK 2), one sweep retry in the same
configuration and the cancel are LOCAL-TESTED too; they are in `test_bbl_dispatch.sh`, cases B, F2
and I2. The 09-25 review fixes are in `test_bbl_launch_fixes.sh`: the h100 memory default, the
archive's dependency, `-c` on PACK=1 jobs, the shard probe (one fwd job and nothing else), and the
all-or-nothing rollback, which drives the LIVE submission path against a stub `sbatch` that fails
on its Nth call. The dead firm-quarter paths are in `test_bbl_deadfq_paths.py` (the single-curve
solve checked field for field against the git HEAD solver) and the probe comparison in
`test_bbl_probe_compare.py` (fake parquet shards in the writer's layout). The packed fwd branch of
`bbl_job.sh` ran for real against a stub `julia`: one listed device per child, per-child logs, a
failed child fails the job, and PACK=1 is unchanged. CLUSTER-ONLY: real shards, what the rate cap
does to real psi (the probe, M1b), packing on real nodes (the 09-24 probe ran one shard per job),
the QOS, `scontrol update` of a dependency, whether SLURM reserves node memory, and wall-clock
times.

**The tag is `_ms979`.** It is not `_ms1` (the beta = 0.9, T = 50 files, which stay untouched) and
not `_probe250`. `make_bbl_cost_tables.py` recognises only tags of the form `_ms<digits>`, so
`_ms1_b979` would not be read as a multi-start tag. `_ms2` would collide with a later
`--n-paths 2` run, whose default tag is `_ms<P>`. `bbl_run.sh` also refuses to launch under a tag
whose `psi_starts` on disk records another beta or T, unless `--force` is given.

**Sizing.** It comes from probe 27455290 (H100, E3 shards 1 and 8, one shard per job, T = 250):
35:06 and 37:04 elapsed, MaxRSS 185.0 and 200.75 GiB, 71-74 GB of the H100's 80 GB. Today's launch
line passes every value explicitly. PACK 4 stays the code default and the option for a quiet
cluster:

| | today: PACK 2 (the M2 launch line) | quiet cluster: PACK 4 (the code defaults) |
|---|---|---|
| MEM per shard | 240G on gpu_h200, 230G on gpu_h100 | the same |
| memory per job | 2 x 240G = 491,520 MB (h200); 2 x 230G = 471,040 MB (h100) | 4 x 240G = 983,040 MB (h200); 4 x 230G = 942,080 MB (h100) |
| throttle | 6 jobs on h200 + 12 on h100 = 36 shards at once | 4 + 8 jobs = 48 shards at once |
| wall | 01:45:00 = 37 min x 1.37 x 1.3 x 1.5 | the same |

**gpu_h100 defaults to 230G per shard, not 240G.** 230G is 1.15 x the measured peak of 200.75 GiB.
A PACK-4 job at 240G asked for 983,040 of an h100 node's 1,000,000 MB, which left 16,960 MB for
anything SLURM reserves on the node (MemSpecLimit), and sbatch refuses a job that asks for more than
the node can schedule. At 230G the margin is 57,920 MB. `--mem-h100` still overrides the default,
and `--mem` sets every partition. `--shard-time derive` selects the T-based wall formula instead,
as a fallback.

**Dead firm-quarters.** For **multi-start psi only**, the solve drops them before anything else: a
(firm, start_q) goes when max |dpsi2| <= 1e-12 x |psi2_eq|. It records `n_null_fq` /
`n_null_rows` for each firm type. A single-curve psi (no `start_q` column of its own; the solve
injects `all`) is solved on every row as loaded, identical to the git HEAD solve field for field.
`run.null_fq.path` in the json says which path ran: `multi_start_drop`, `multi_start_keep`
(`--keep-null-fq`) or `single_curve`. If the drop empties a whole firm type, the solve prints
`!!!! WARNING: EVERY <type> firm-quarter is dead ...`, lists the type in
`run.null_fq.types_emptied` and writes no block for it; it does not fail. (With `--promote`, the
gate then judges only the types that are left.) `--keep-null-fq` reproduces the previous solve.
On `_ms1` this left theta-hat unchanged to 1e-15 and moved E3 B frac_bind from 0.401 to 0.483.
`make_bbl_cost_tables.py --from-psi` (the ridge pass of the tables job) still recomputes the ratio
over all rows, so on multi-start psi with dead firm-quarters it will stop with "psi archive
does not match the solve". The tables job carries on past it (`|| echo`), and the cost tables
still render. The table-side mask is the next change.

**All or nothing.** If any `sbatch` of a `bbl_run.sh` invocation is refused, or the script refuses
anything after its first submission (for example E4 refused after E3's chain went in), it cancels
every job it had submitted, dependents first (tables + archive, solves, sweeps, fwd jobs, then
polfunc and warmup: the `bbl_cancel.sh` order), writes a STOP marker for each routine it touched,
prints `ROLLBACK: ...` with every job id and name, and exits nonzero. Nothing half-submitted stays
in the queue.

**CPUs of every GPU fwd job are explicit.** A PACK=1 job asks for `--cpus-per-task=8`
(`BBL_THREADS_PER_SHARD`) as a PACK=k job asks for 8k. Every submission made from inside a job
(the `bbl_submit` wrap and the sweep both run `-c 1 --mem=4G`) goes out with the parent's
`SLURM_CPUS_PER_TASK`, `SLURM_MEM_PER_*` and `SLURM_TRES_PER_TASK` unset, so a job's Julia thread
count comes from its own request, never from the parent's `-c 1`.

### M0. Upload (nothing running reads these yet)

Local root: `C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Code\Egan_et_al_2025_Rep\`.
Cluster root: `~/project_pi_mf2263/pf382/dep_comp/`.

<!-- md5:UPLOAD_TABLE -->
| # | local path | cluster destination (under `~/project_pi_mf2263/pf382/dep_comp/`) | size (B) | md5 | why |
|---|---|---|---|---|---|
| 1 | `bbl_run.sh` | `scripts/bbl_run.sh` | 57,770 | `df1ac5dd951b67ca00acbf69fbbcec65` | launch: sizing, tag guard, --shard-probe, cancel-on-failed-submit, zip afterok solves |
| 2 | `bbl_job.sh` | `scripts/bbl_job.sh` | 34,926 | `afc04f2c3b7348ecf1067618770ccfcd` | fwd packing (one GPU per child), the sweep step, curve check |
| 3 | `cluster_lib.sh` | `scripts/cluster_lib.sh` | 92,397 | `bdeea3bd1df149f212f36ef3038d06bc` | dispatch, sizing defaults (h200 240G, h100 230G), -c on every GPU fwd job |
| 4 | `cluster_archive.sh` | `scripts/cluster_archive.sh` | 26,898 | `1cb1ae2139e76314be1bf2f9eef5293e` | slim archive (this tag's psi + results) |
| 5 | `cluster_preflight.sh` | `scripts/cluster_preflight.sh` | 21,899 | `26fc62671658454a5c530ae766268e6c` | section (7): produced inputs looked up under their manifest dest |
| 6 | `bbl_status.sh` | `scripts/bbl_status.sh` | 15,885 | `59770d6a13f700a7f5c5551422fd1786` | status, --timing first-hour check |
| 7 | `bbl_cancel.sh` | `scripts/bbl_cancel.sh` | 4,175 | `a1599e986a5a4349be81a1930af6f928` | cancel in dependency order |
| 8 | `bbl_sizing.sh` | `scripts/bbl_sizing.sh` | 14,578 | `27464d7510c8a0c8fbe4dae9ba2215dc` | probe reader (bbl_status.sh --probe calls it) |
| 9 | `bbl_shards.py` | `scripts/bbl_shards.py` | 19,496 | `a767327b1dcc2e1ff03a5035733e22e8` | sweep coverage (truncated parquet = missing), registry reader |
| 10 | `bbl_solve.py` | `scripts/bbl_solve.py` | 79,017 | `65b3998677924f1f3003cc649e591cca` | dead firm-quarter exclusion (multi-start only), --keep-null-fq, beta/T in the json |
| 11 | `bbl_fwd_sim.jl` | `scripts/bbl_fwd_sim.jl` | 70,529 | `9856eb2fa1dc49dc1a4b8359c4aac1bc` | atomic psi writes, peak RSS line, curve refusal, registry, beta passed to the deposit sim |
| 12 | `cf_deposit_sim.jl` | `scripts/cf_deposit_sim.jl` | 29,460 | `9551a8b5404be1953ca404dd87acc6bd` | r^dep stability cap beta*phi*(1+r^dep) <= 0.995 when beta is given (never binds at beta=0.9) |
| 13 | `cf_psi_basis.jl` | `scripts/cf_psi_basis.jl` | 19,522 | `ed60db8a2a6122736f9bde7491dd2a0a` | bbl_discount() registry reader (included by the sim) |
| 14 | `bbl_probe_compare.py` | `scripts/bbl_probe_compare.py` | 14,784 | `8a6aabd8f6bc6efbe467e4fb92dcec87` | compares the capped probe with the uncapped probe and the beta=0.9 run |
| 15 | `bbl_discount.env` | `scripts/bbl_discount.env` | 1,483 | `9b7c8965930dd5d84b37a27b8740662f` | THE registry: BBL_BETA=0.979, BBL_HORIZON=250 |
| 16 | `utils/bbl_discount.py` | `scripts/utils/bbl_discount.py` | 4,897 | `004f3c769eb96b2140399e1b1d15893d` | registry reader of the tables job |
| 17 | `make_bbl_cost_tables.py` | `scripts/make_bbl_cost_tables.py` | 125,927 | `c4f428e464fca14c972c2016f4f3ed04` | the tables job (09-24 version) |
| 18 | `cf_run.sh` | `scripts/cf_run.sh` | 12,414 | `ff12d7f972521ab7d0e20c59566fc0ba` | CF side: beta from the registry (not used by this launch) |
| 19 | `cf_eq_run.sh` | `scripts/cf_eq_run.sh` | 13,263 | `37e5133ba8e1e487ee1e1a6f9f19ab63` | CF side: beta from the registry (not used by this launch) |
| 20 | `cf1_franchise.jl` | `scripts/cf1_franchise.jl` | 13,717 | `6b1cd5b647dc168803a146f4b4146999` | CF side: beta from the registry (not used by this launch) |
| 21 | `cf3_equilibrium.jl` | `scripts/cf3_equilibrium.jl` | 21,758 | `47c8e7ff961ef4e109b04242a8ae84c7` | CF side (+cf5/cf6): beta from the registry (not used by this launch) |
| 22 | `diag_cf1_franchise_dataonly.py` | `scripts/diag_cf1_franchise_dataonly.py` | 7,963 | `982f79c27f6176c62ca419e21fcd207c` | CF side diagnostic: beta from the registry |
| 23 | `cluster/upload_manifest.txt` | `scripts/cluster/upload_manifest.txt` | 21,726 | `41e7b376f8b8e7b124222036a0246ecc` | preflight (7) reads it; curves listed at h=250 |
| 24 | `cluster/RUNBOOK.md` | `scripts/cluster/RUNBOOK.md` | 17,178 | `e1a88838acbf8e649e1c177684fe76c9` | general runbook (09-24) |
| 25 | `C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\BCB\Egan_et_al_2025_Rep\processed\ESTIMATION_OUTPUT\COST_FWD\forward_rf_vintages.csv` | `data/input/forward_rf_vintages.csv` | 815,802 | `13f321168f7bbd6e29f76b2b3f6e3b37` | 36 Focus vintages, h = 0..250 |
| 26 | `C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\BCB\Egan_et_al_2025_Rep\processed\ESTIMATION_OUTPUT\COST_FWD\forward_rf_qoq.csv` | `data/input/forward_rf_qoq.csv` | 20,478 | `e544d34079c60c8169f0995faa171873` | single curve, h = 1..250 |
<!-- /md5:UPLOAD_TABLE -->

Upload each file, as it is on disk now (LF line endings), to its destination above; the sysimages
need no rebuild. Verify from the login node (one line each):

<!-- md5:MD5_DATA_CMD -->
```
cd ~/project_pi_mf2263/pf382/dep_comp && md5sum data/input/forward_rf_vintages.csv data/input/forward_rf_qoq.csv scripts/bbl_discount.env
```
<!-- /md5:MD5_DATA_CMD -->

It must print exactly:

<!-- md5:MD5_DATA_EXPECT -->
```
13f321168f7bbd6e29f76b2b3f6e3b37  data/input/forward_rf_vintages.csv
e544d34079c60c8169f0995faa171873  data/input/forward_rf_qoq.csv
9b7c8965930dd5d84b37a27b8740662f  scripts/bbl_discount.env
```
<!-- /md5:MD5_DATA_EXPECT -->

The code files, the same way (compare with the md5 column above):

<!-- md5:MD5_CODE_CMD -->
```
cd ~/project_pi_mf2263/pf382/dep_comp && md5sum scripts/bbl_run.sh scripts/bbl_job.sh scripts/cluster_lib.sh scripts/cluster_archive.sh scripts/cluster_preflight.sh scripts/bbl_status.sh scripts/bbl_cancel.sh scripts/bbl_sizing.sh scripts/bbl_shards.py scripts/bbl_solve.py scripts/bbl_fwd_sim.jl scripts/cf_deposit_sim.jl scripts/cf_psi_basis.jl scripts/bbl_probe_compare.py scripts/utils/bbl_discount.py scripts/make_bbl_cost_tables.py scripts/cf_run.sh scripts/cf_eq_run.sh scripts/cf1_franchise.jl scripts/cf3_equilibrium.jl scripts/diag_cf1_franchise_dataonly.py scripts/cluster/upload_manifest.txt scripts/cluster/RUNBOOK.md
```
<!-- /md5:MD5_CODE_CMD -->

### M1. Before launching (login-safe, one line each)

```
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && squeue -u $USER -o '%.12i %.28j %.9T %.22R' | grep -E 'bbl_|JOBID'
```

Nothing named `*_ms979` or `*_probe250cap` may be listed. `bbl_run.sh` would refuse a duplicate
anyway.

Optional dry run (it is a 1-core `day` job, because `bbl_run.sh` must not run on the login node):

```
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch -p day -t 00:15:00 -c 1 --mem=4G -J bbl_dry -o logs/bbl_dry_%j.out --wrap "bash bbl_run.sh --dry-run --routines '3 4' --fwd-gpu --multi-partition --shards 300 --multi-start --n-paths 1 --psi-tag _ms979 --no-warmup --no-polfunc --pack-h200 2 --pack-h100 2 --mem-h200 240G --mem-h100 230G --shard-time 01:45:00"
```

Once it has finished, `grep -h '^BBL_JOB_NAMES=' logs/bbl_dry_<jobid>.out` must print the line in M3.

### M1b. FIRST: probe the rate-cap code on one shard (one job, nothing chained)

The deposit-rate stability cap (`cf_deposit_sim.jl`, and `bbl_fwd_sim.jl` passing beta to
`simulate_deposits`) is new code, so one production shard runs alone before the launch: E3, shard
1 of 300, one GPU on gpu_h200 (h100 has no room today), tag `_probe250cap`, beta and T from
`bbl_discount.env`, 240G, wall 01:45:00 (the unpacked T = 250 probe took 35-37 min).
`--shard-probe` submits exactly the shards of `--array-spec`, one per job, and NOTHING else: no
warmup, polfunc, sweep, solve, tables or archive, so nothing ever simulates the other 299 shards of
the tag. It is refused without `--array-spec`, without a `--psi-tag` containing `probe`, or when
the shard's file already exists under that tag (`--force` overwrites); `--repair` refuses every
probe tag.

```
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_probe_submit -o logs/bbl_probe_submit_%j.out --wrap "bash bbl_run.sh --shard-probe --routines 3 --shards 300 --array-spec 1 --multi-start --n-paths 1 --psi-tag _probe250cap --partitions gpu_h200"
```

Then `grep -h '^BBL_JOB_NAMES=' logs/bbl_probe_submit_<jobid>.out` must print exactly
`BBL_JOB_NAMES=bbl_fwd_E3_probe250cap`, and `squeue -u $USER -n bbl_fwd_E3_probe250cap` lists ONE
job. Its log is `logs/bbl_fwd_E3_probe250cap_<fwd jobid>_1.out`: it should show
`GPU share kernel enabled`, `[BBL] shard 1 peak RSS ... GiB` and end with `[DONE]`.

**Compare** once `sacct -j <fwd jobid> -X -o JobID,State,Elapsed,MaxRSS` shows COMPLETED. The
comparison reads shard 1 of 300 of `_ms1` (beta = 0.9, T = 50, the baseline), `_probe250`
(beta = 0.979, T = 250, no cap: the 09-24 memory probe) and `_probe250cap` (the same with the cap),
plus each tag's `psi_eq` where one exists (only `_ms1`: shard 0 never ran for either probe). It runs
in the solve's Python environment, activated exactly as the solve step does it (`cl_setup_python`:
`PY_MODULE` / `CONDA_ENV`):

```
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch -p day -t 00:15:00 --mem=16G -J bbl_probe_cmp -o logs/bbl_probe_cmp_%j.out --wrap "bash -c '. ./cluster_lib.sh && cl_export_step_dirs && cl_setup_python \"\$CL_PY_REQ_SOLVE\" && \"\$PYBIN\" bbl_probe_compare.py --key E3_spec_12_extended --shard 1 --n-shards 300'"
```

`logs/bbl_probe_cmp_<jobid>.out` prints, per psi component (psi1, psi2_omega, psi4_zeta, then the
psi3_gamma_*) and tag, max / p99 / median of |psi| across the shard's firm x start x shock rows,
each also as a ratio to `_ms1`; the rows and firms with |psi2| above 100 x the tag's own median;
the 10 firms with the largest |psi2| under `_probe250` next to their `_probe250cap` and `_ms1`
values; how far the cap moved each component row for row, over all rows and over the typical rows
only; and a SUMMARY. The cap does what it is for when `_probe250cap` has no explosive rows where
`_probe250` has some, and the SUMMARY's typical-rows line says the cap moved (almost) none of them.
GO / NO-GO for M2 is your call.

### M1c. Node check: the step right before the launch (login-safe, one line)

How many nodes could start a job NOW, per partition and PACK (240G and one free GPU per shard):

```
scontrol -o show node | awk '{delete v; for(i=1;i<=NF;i++){j=index($i,"="); if(j) v[substr($i,1,j-1)]=substr($i,j+1)} p=(v["Partitions"]~/gpu_h100/)?"h100":((v["Partitions"]~/gpu_h200/)?"h200":""); if(p==""||v["State"]~/DOWN|DRAIN|FAIL|MAINT|RESERVED/) next; tot=(p=="h100")?4:8; ag=0; if(match(v["AllocTRES"],/gres\/gpu=[0-9]+/)) ag=substr(v["AllocTRES"],RSTART+9,RLENGTH-9)+0; fg=tot-ag; fm=(v["RealMemory"]-v["AllocMem"])/1024; for(k=1;k<=4;k++) if(fg>=k && fm>=240*k) n[p" pack="k]++} END{for(x in n) print x": "n[x]" node(s) could start a job now"}' | sort
```

PACK 2 starts at once where `h200 pack=2` and `h100 pack=2` are listed. A partition with no line
still takes the jobs; they wait (Resources / Priority) until a node frees up. If both partitions
list `pack=4`, the quiet-cluster PACK-4 line below M2 runs 48 shards at once instead of 36.

### M2. THE launch command (PACK 2)

```
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out --wrap "bash bbl_run.sh --routines '3 4' --fwd-gpu --multi-partition --shards 300 --multi-start --n-paths 1 --psi-tag _ms979 --no-warmup --no-polfunc --pack-h200 2 --pack-h100 2 --mem-h200 240G --mem-h100 230G --shard-time 01:45:00"
```

It runs `cluster_preflight.sh` first, then submits the graph below. `--no-polfunc` uses the
`polfunc_fitted.csv` already on disk, the one `_ms1` and the probes used.

```
bbl_fwd_E3_ms979 [gpu_h200] 50 jobs x 2 shards (0-99),     %6  --+
bbl_fwd_E3_ms979 [gpu_h100] 100 jobs x 2 shards (100-299), %12 --+--afterany--> bbl_sweep_E3_ms979 (held, then released)
bbl_solve_E3_ms979  afterok the FINAL sweep (each sweep retry moves this dependency onto the next sweep)
   ... the same for E4 ...
bbl_tables_ms979  afterok bbl_solve_E3_ms979 + bbl_solve_E4_ms979
bbl_zip_ms979     afterok bbl_solve_E3_ms979 + bbl_solve_E4_ms979, NOT the tables   (slim: _ms979 psi + cost_params, tab_bbl_*, polfunc_*)
```

The archive does not wait for the tables: they are rendered locally from the archived cost_params
as well, so a tables job that fails cannot cancel the download. The two start together once both
solves are done, so the `tab_bbl_*` files in the archive are whatever the folder holds at that
moment (possibly none, or an earlier run's). A sweep re-runs missing shards with the run's own
PACK 2 and memory, read from its `context.env`.

On a quiet cluster (M1c lists `pack=4` on both partitions), the PACK-4 launch, which is the code
default, runs 25 + 50 jobs of 4 shards, `%4` and `%8`, 48 shards at once:

```
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out --wrap "bash bbl_run.sh --routines '3 4' --fwd-gpu --multi-partition --shards 300 --multi-start --n-paths 1 --psi-tag _ms979 --no-warmup --no-polfunc --pack-h200 4 --pack-h100 4 --mem-h200 240G --mem-h100 230G --shard-time 01:45:00"
```

### M3. The one line to read straight away

```
grep -h '^BBL_JOB_NAMES=' logs/bbl_submit_<jobid>.out
```

It must print exactly this:

```
BBL_JOB_NAMES=bbl_fwd_E3_ms979,bbl_sweep_E3_ms979,bbl_solve_E3_ms979,bbl_fwd_E4_ms979,bbl_sweep_E4_ms979,bbl_solve_E4_ms979,bbl_tables_ms979,bbl_zip_ms979
```

The banner at the top of the same log must read `beta=0.979 (bbl_discount.env)  T=250 (bbl_discount.env)`.

Anything else is a mis-parse, and you should cancel at once (M5). Examples: `bbl_polfunc`,
`bbl_warmup`, E1/E2, `_ms1`, `_ms8`, or no tag. If there is no such line at all, `bbl_run.sh`
stopped early. `cat logs/bbl_submit_<jobid>.out` says why. If it had submitted anything first, it
cancelled all of it itself: the log ends with `ROLLBACK: ...` naming each job, and
`squeue -u $USER | grep ms979` shows nothing of it. Only if a ROLLBACK line reports a failed
`scancel`, cancel what is left with M5.

### M4. First-hour checks (login-safe, one line each)

1. **State of the chain:** `bash bbl_status.sh --routines "3 4" --psi-tag _ms979`. You should see
   `wait (fwd running ...)` on both lines, and fwd `R/P/F` with F = 0. Jobs beyond the QOS wait as
   `QOSMaxJobsPerUserLimit`; E3 and E4 share the per-user caps.
2. **Timing, GPU kernel and RSS in one line (after ~25 min):**
   `bash bbl_status.sh --routines "3 4" --psi-tag _ms979 --timing`. For each routine it prints:
   - `GPU n/m`: m children, n of them logged `GPU share kernel enabled`. Any `NO KERNEL: shard ...`
     took the CPU share path.
   - `RSS`: how many shards logged `[BBL] shard i peak RSS X GiB`, and the largest X (MEM is 240G on
     h200, 230G on h100).
   - `done`: finished shards, with the worst elapsed / 36 min.
   - `running`: shards past 25 simulations, with the worst seconds per simulation / 39 s.

   The line ends with **GO** when the worst ratio is <= 1.3, and **NO-GO** otherwise. The
   reference is the unpacked T = 250 probe: 35-37 min per shard, 38-40 s per simulation.
3. **The same, raw:** run this at least 5 min after the children start. It prints nothing when
   every child is on the GPU:
   `grep -L "GPU share kernel enabled" logs/bbl_fwd_E[34]_ms979_shard*_*.out`
4. **The largest peak RSS so far:** `grep -ho 'shard [0-9]* peak RSS [0-9.]* GiB' logs/bbl_fwd_E[34]_ms979_shard*_*.out | sort -k5 -n | tail -3`
5. **Node memory of the packed jobs (completed ones):**
   `sacct -u $USER -S now-12hours --name=bbl_fwd_E3_ms979,bbl_fwd_E4_ms979 -o JobID%20,Partition,State,Elapsed,MaxRSS | grep -E 'batch|JobID' | head -20`
   `MaxRSS` of a packed `.batch` step is the sum of its 2 shards, which must stay under 491,520 MB
   on h200 and 471,040 MB on h100 (at PACK 4: 983,040 and 942,080 MB).

**If NO-GO** (packed shards more than 1.3x slower than the unpacked probe), the shards still finish
while they are under 01:45:00, about 2.9x. Packing 2 per job (36 at once) also still beats one per
job (18 at once) up to 2x. The sweep re-runs any TIMEOUT at 1.5x the wall by itself. Stopping to
relaunch unpacked is your call. To do it, cancel (M5), then run M6 with `--pack-h200 1 --pack-h100 1`.

### M5. Safe cancel (dependents first; one line)

```
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && bash bbl_cancel.sh --routines "3 4" --psi-tag _ms979
```

It first writes a STOP marker per routine, so a sweep caught mid-run re-submits nothing. It then
cancels in this order: `bbl_tables_ms979` + `bbl_zip_ms979`, then the solves, then the sweeps,
then the fwd arrays. It waits for each group to leave the queue before the next. Add `--dry-run`
to list without cancelling. It touches only `*_ms979` names, never `_ms1`.

### M6. Manual recovery (one line)

These are the cases: the sweep says `exhausted` or `problem`, the chain was cancelled, or
`bbl_status.sh` says `ACT: nothing queued`. Read why first:
`sacct --name=bbl_fwd_E3_ms979 -X -o JobID,State,Elapsed,MaxRSS | grep -v COMPLETED`. Then
re-attach the chain. This runs a sweep now, which re-runs only the missing shards with the same
N_SHARDS = 300, tag and design, then solve, tables and archive:

```
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out --wrap "bash bbl_run.sh --routines 3 --shards 300 --multi-start --n-paths 1 --psi-tag _ms979 --repair --multi-partition --pack-h200 2 --pack-h100 2 --mem-h200 240G --mem-h100 230G --shard-time 01:45:00"
```

This is E3; for E4, change `--routines 3` to `--routines 4`. Repair one routine at a time: a repair
is refused while any sweep or solve of that routine is still queued.

- **More wall** (TIMEOUTs): change `--shard-time`.
- **OOM:** use `--mem-h200 300G --mem-h100 300G` (2 x 300G fits either node). At PACK 4, add
  `--pack-h100 3`, since 4 x 300G does not fit an h100 node and is refused.
- A probe tag (`_probe*`) is never repaired: `--repair` refuses it, because its sweep would
  simulate every shard the probe left out on purpose.
- **A held sweep** (`ACT: scontrol release <id>`): `scontrol release <id>`.

`--repair` takes the design (N_SHARDS, T, beta, shocks, the sim flags) from
`data/output/bbl/.dispatch/E3_spec_12_extended_ms979/context.env`, and it refuses a different
`--shards`. The placement comes from the flags above.

---

## The commands, in order

```
# 1. the MEMORY PROBE: DONE 2026-09-24 (job 27455290); its numbers are the sizing in MORNING LAUNCH
# 2. the CODE PROBE of the rate cap, one shard (MORNING LAUNCH M1b), then its comparison (M1b)
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_probe_submit -o logs/bbl_probe_submit_%j.out --wrap "bash bbl_run.sh --shard-probe --routines 3 --shards 300 --array-spec 1 --multi-start --n-paths 1 --psi-tag _probe250cap --partitions gpu_h200"
# 3. the full E3/E4 launch, PACK 2 (one line; MORNING LAUNCH M2 has it with its checks)
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out --wrap "bash bbl_run.sh --routines '3 4' --fwd-gpu --multi-partition --shards 300 --multi-start --n-paths 1 --psi-tag _ms979 --no-warmup --no-polfunc --pack-h200 2 --pack-h100 2 --mem-h200 240G --mem-h100 230G --shard-time 01:45:00"

# 4. watching it (login-safe, one screen)
bash bbl_status.sh --routines "3 4" --psi-tag _ms979
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

**The new run's tag is `_ms979`.** The beta=0.9 / T=50 run is `_ms1`, and its files stay on the
cluster untouched. The tag keeps the `_ms<digits>` shape, the only one `make_bbl_cost_tables.py`
reads as a multi-start tag (so not `_ms1_b979`), and no `--n-paths` default can produce it (unlike
`_ms2`). A launch under a tag whose `psi_starts` records another beta or T is refused. Every name
the new run writes (psi, cost_params, tables, job names, the archive) carries `_ms979`, and the solve, the tables and the slim archive match a tag only as the whole
remainder of a name, so the two runs can never mix.

---

## 0. Before the first launch (once per upload)

1. **Upload the files listed in MORNING LAUNCH M0**, one by one (never the whole code bundle while
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
PACK 2 graph is in MORNING LAUNCH M2):

```
bbl_fwd_E3_ms979 [gpu_h200] PACK 4: shards 0-99,    25 jobs, %4 --+
bbl_fwd_E3_ms979 [gpu_h100] PACK 4: shards 100-299, 50 jobs, %8 --+--afterany--> bbl_sweep_E3_ms979 (#0, held until the solve exists)
                                                                                    | afterok
                                                             bbl_solve_E3_ms979 <-----+   (always afterok the FINAL sweep)
   ... the same for E4 ...
bbl_solve_E3_ms979 + bbl_solve_E4_ms979 --afterok--> bbl_tables_ms979
bbl_solve_E3_ms979 + bbl_solve_E4_ms979 --afterok--> bbl_zip_ms979 (slim, tag _ms979; it does not wait for the tables)
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
* `--promote` is not in the command: the solve writes `cost_params_E{k}_spec_12_extended_ms979.json`
  and leaves the untagged file the counterfactuals read alone. Promoting is a separate decision.

### E1 and E2 afterwards (a separate, later command)

They feed only the counterfactuals. Their panels are **larger** than E3's (E1 has 796,154 rows),
and RSS scales with the panel, so probe one of them first. Then launch with the sizing it prints:

```
sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out --wrap "bash bbl_run.sh --probe --routines 1 --shards 300 --multi-start --n-paths 1 --psi-tag _probe250e1"
bash bbl_sizing.sh --psi-tag _probe250e1 --routines "1 2" --launch-tag _ms979
#   -> paste its sbatch line (it keeps --no-polfunc: a re-fit would rewrite the CSV that running
#      E3/E4 tasks read)
```

Their tables job re-renders the tables with all four routines on disk. Their archive holds every
`cost_params_*_ms979.json` (all four routines) and this tag's psi.

---

## 3. What the sweep does by itself

`bbl_sweep_E<k>_ms979` is a 30-minute, 4 GB `day` job that runs after all of a routine's fwd jobs have
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
bash bbl_status.sh --routines "3 4" --psi-tag _ms979
bash bbl_status.sh --rss                         # + MaxRSS of completed fwd jobs, per-shard peaks of packed ones
```

One line per routine: shards on disk / N, fwd jobs running/pending/failed, the sweep's state and
retries used (`r1/3`), the solve's state, and the **next action**. `wait (...)` means the chain is
handling it. `ACT: ...` gives the exact command to run. The last line shows the tables and archive
jobs.

Reading the tables: `cat logs/bbl_tables_ms979_<jobid>_*.out`. The archive lands in
`data/output/download/bbl_outputs_<jobid>.zip`. The solve's `cost_params` json records, in its
`run` block, the beta and T the psi were simulated under and where that came from
(`psi_starts_<tag>.json`).

---

## 5. Manual recovery

| status says | do |
|---|---|
| `ACT: read logs/bbl_sweep_...` (retries spent, or a problem) | `sacct --name=bbl_fwd_E3_ms979 -X -o JobID,State,Elapsed,MaxRSS` and the shard logs say why (TIMEOUT, OOM, a traceback). Fix it (for example `--shard-time`, `--mem-h200`), then `--repair` (below) |
| `ACT: scontrol release <id>` | a sweep stayed held (release failed after the solve was re-targeted): `scontrol release <id>` |
| `ACT: read logs/bbl_solve_...` | the solve failed or was cancelled: read its `.out`/`.err`, fix, then `--repair` |
| `ACT: nothing queued, n/N shards` | the chain is gone with gaps left: `--repair` |

**`--repair`** re-attaches a dead chain without relaunching anything that exists. It submits a sweep
immediately (which re-runs whatever is missing, or nothing), then solve -> tables -> archive:

```
sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%j.out --wrap "bash bbl_run.sh --routines 3 --shards 300 --psi-tag _ms979 --multi-start --n-paths 1 --repair --multi-partition --pack-h200 2 --pack-h100 2 --mem-h200 240G --mem-h100 230G --shard-time 01:45:00"
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
bash bbl_cancel.sh --routines "3 4" --psi-tag _ms979 --dry-run     # list
bash bbl_cancel.sh --routines "3 4" --psi-tag _ms979               # tables/zip -> solves -> sweeps -> fwd
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
| `--pack`, `--pack-h200`, `--pack-h100` / `PACK*` | 4, 4 (today's launch passes 2, 2) | shards per job; PACK=1 is one shard per job |
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
