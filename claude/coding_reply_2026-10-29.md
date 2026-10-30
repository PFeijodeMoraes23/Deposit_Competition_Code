# Coding reply: R3 partly in; ext1 multi-start, TR5 and the download fix staged

**To:** Pedro and the main thread. **From:** coding agent. **Written:** 2026-10-29, 19:41.

## Summary

- **R3 (job 31020411) is partly ingested.**
  - In: demand_prep, blp, logs, logit, gates. All six gates passed (G0, G1, G2, G3, G4, G7).
  - Still downloading: the sleep set (15 parts; 7 had landed at 19:33). Everything below that needs φ waits for it.
- **The headline RC-BLP fit (ext1) is a local optimum again.** α is E3 −0.009, E4 +0.004. This is the 10-16 failure repeating:
  - the multi-start of 10-19 found a lower Q;
  - your decision then ("more ext1 starts next full run") never reached the engine.
  - **Pedro approved the multi-start this evening.** It is staged: run it with the lines below.
- **The download fix:** one `pipeline_all.sh` submit now packs every set, sleep, logit and gates included, also with `--no-bbl --no-cf`. It ships with the next code bundle.
- **TR5 is staged** (`upload_tr5_20261029.zip`, sha256 `2fc49c5c…`). It includes the ext1 extra starts, so the variants' BLP is not stuck at the same local optimum.

## Cluster lines, in order (Pedro)

Upload both zips and their `.sha256` files to `~/project_pi_mf2263/pf382/dep_comp/`. They are in `ESTIMATION_OUTPUT\CLUSTER_OUT\`:
- `upload_ext1_20261029.zip` (sha256 `d28de3b1…`)
- `upload_tr5_20261029.zip` (sha256 `2fc49c5c…`)

The two runs are independent; launch both now. The GPU cap (6 jobs) queues the overflow. TR5 spends its first hours on CPU (sleepiness), so it overlaps the ext1 GPU jobs.

### A. ext1 multi-start (12 GPU jobs, about 1 h each)

Six starts per routine:
- `best1019`: the 10-19 adopted fit;
- `rc3`: R3's rc3, padded with zeros;
- `logit`;
- `rand1`, `rand2`, `rand3`: the same values as on 10-18.

The sandboxes go under `diag_ext1_20261029` on scratch. Production files are hashed first and never written.

1. Unpack and check:
   ```
   cd ~/project_pi_mf2263/pf382/dep_comp && unzip -o upload_ext1_20261029.zip && sha256sum -c upload_ext1_20261029.sha256
   ```
2. Build and submit as one job (returns the id):
   ```
   cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch --parsable -p day -t 12:00:00 -c 1 --mem=2G -J ext1_build -o $HOME/scratch_pi_mf2263/pf382/ext1_build_%j.out --export=ALL,EXT1_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 --wrap "bash blp_ext1_starts.sh"
   ```
3. Watch:
   ```
   squeue -u $USER -o "%.10i %.22j %.8T %.10M %R" | grep -E "ext1"
   ```
4. Results, once the 12 `ext1s_*` jobs are gone:
   ```
   cd ~/project_pi_mf2263/pf382/dep_comp/scripts && EXT1_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_ext1_starts.sh --report
   ```
5. Adopt, one line per routine, after you pick the start (**your call**). My recommendation: the lowest Q with `converged` true and SEs computed. The script refuses anything else.
   ```
   cd ~/project_pi_mf2263/pf382/dep_comp/scripts && EXT1_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_ext1_adopt.sh 3 <start>
   ```
   ```
   cd ~/project_pi_mf2263/pf382/dep_comp/scripts && EXT1_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_ext1_adopt.sh 4 <start>
   ```
6. Download, from `data/output/blp/`, the 8 files matching `*E3*ext1*` and `*E4*ext1*`, into a new folder `CLUSTER_IN\20261029_ext1\`. I place them and rebuild the BLP table and keys.

To cancel:
```
scancel <build id>; squeue -u $USER -h -o "%i %j" | awk '$2 ~ /^ext1s_/ {print $1}' | xargs -r scancel
```

### B. TR5 (both variants, full sleepiness inference)

1. Unpack and check:
   ```
   cd ~/project_pi_mf2263/pf382/dep_comp && unzip -o upload_tr5_20261029.zip && sha256sum -c upload_tr5_20261029.sha256
   ```
2. Build and submit as one job (the login node's file system is slow):
   ```
   cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch --parsable -p day -t 04:00:00 -c 1 --mem=2G -J tr5_build -o $HOME/tr5_build_%j.out --wrap "bash tr5_variants.sh --build-only && bash tr5_variants.sh --submit-built"
   ```
3. Status:
   ```
   cd ~/project_pi_mf2263/pf382/dep_comp/scripts && bash tr5_variants.sh --status
   ```
4. When both chains are done, pack as a job:
   ```
   cd ~/project_pi_mf2263/pf382/dep_comp/scripts && sbatch --parsable -p day -t 02:00:00 -c 2 --mem=8G -J tr5_pack -o $HOME/tr5_pack_%j.out --wrap "bash tr5_variants.sh --pack"
   ```
   Then download `dep_comp/tr5_results_no5.zip` and `dep_comp/tr5_results_d5only.zip`.

To cancel:
```
cd ~/project_pi_mf2263/pf382/dep_comp/scripts && bash tr5_variants.sh --cancel
```

## R3 results so far

**Gates:** all six `ok`. G3 (the AME) finished at 00:45 on 10-28, hours after the RC ladders.

**Logit** (now in Drafts, `est1-4_spec12_logit_comparison.tex` and the per-routine tables):

| | Old (10-18) | R3 |
|---|---|---|
| α E3 | −0.0638 (se 0.0240) | −0.0616 (se 0.0260), t −2.37, p 0.058 |
| α E4 | −0.0554 (se 0.0244) | −0.0529 (se 0.0260), t −2.04, p 0.092 |

**E3 likely drops from the 5% to the 10% level.** The old p is my estimate from the estimate and SE (about 0.04). The changed-key report will confirm it, and any "significant at 5%" text for the logit α needs checking.

**RC-BLP, ext1 (headline Table `blp_demand_comparison_noseg_spec12`):**

| | rc4 Q | ext1 Q | ext1 α | added π (ln assets × gdp) |
|---|---|---|---|---|
| E3, R3 | 0.0904 | 0.0904 | −0.009 | 0.000 |
| E4, R3 | 0.0966 | 0.0966 | +0.004 | 0.000 |
| E3, 10-19 adopted (old panel) | 0.0873 | 0.0813 | −0.119 | −0.109 |
| E4, 10-19 adopted (old panel) | 0.0936 | 0.0909 | −0.075 | −0.085 |

- R3's ext1 kept the warm rc4 start (Q 0.0904 against the cold start's 0.0935). It never moved the added π off zero, and its Q equals rc4's: the signature of the 10-16 local optimum.
- Every rung before ext1 matches the old run closely (sigma through rc4: α −0.062 → −0.009 for E3, as before). So the data change is not the cause.
- **I rebuilt the headline table at 18:41 from these fits, before I saw this.** The table in Drafts therefore shows α −0.009 / +0.004 until the multi-start is adopted. The BLP macro keys stay pending. Do not compile V_Friction for the BLP section until then.

## Code changed today (no git action taken)

- **`pipeline_all.sh` (download):**
  - `pipe_download` goes out from whichever invocation ends the chain, `afterANY` that invocation's last jobs and every earlier job still in the queue. Earlier ids travel in a new internal `--download-after`.
  - Sets of phases switched off with `--no-<phase>` stay home. For `--no-bbl --no-cf` the sets are sleep, demand_prep, logit, blp, gates and logs.
  - A default run is unchanged: all eight sets, after the CF jobs and the BBL solves.
  - A hand resume (`--from bbl`) still packs every set.
  - New `--no-download` flag for callers that pack themselves. `tr5_variants.sh` passes it.
  - `-h` now prints the whole header (it stopped at line 88, mid-flags).
  - Tested locally against stub `sbatch`/`squeue`/children: R3 shape, default, sleep/logit-only, hand resume, TR5 shape, dry runs, and a cfeq-only invocation (no duplicate pack).
- **`blp_engine_cpu.jl` (the 10-19 decision):**
  - ext1 now runs three extra starts after its two built-in ones: rand1–rand3 from `config/blp_ext1_extra_starts.csv`, new.
  - The lowest re-evaluated Q wins, as before. `extended` is unchanged.
  - The GPU engine includes this file, and the sysimage holds only packages, so no rebuild is needed.
  - Cost: about 1–2 GPU-h more per routine (R3's ext1 starts took 13–37 min each).
  - Julia unit test passed: start order, choice, allow-single path, malformed-row and repeated-name errors. Parses on 1.11.4.
  - New manifest row `ext1_starts_config` ships the CSV with the next bundle.
- **`blp_ext1_starts.sh` / `blp_ext1_adopt.sh`:**
  - `EXT1_TAG` (default 20261029) names the run;
  - new starts file `blp_ext1_starts_20261029.csv`;
  - adopt no longer counts TR5 sandbox jobs as live.
- **`tr5_variants.sh` / `make_tr5_upload.py`:**
  - `--no-download`;
  - the overlay adds `blp_engine_cpu.jl` and the starts CSV;
  - the sandbox check verifies both.
- **Docs:** `cluster/RUNBOOK.md` §4 and the cluster-run skill describe the new download rule and the ext1 local optimum.
- **K8, prepared (it runs once the sleep set is in):**
  - **`sleep_ident_entry_dynamics.py` writes `d6_g_holdout.csv`.** On event-quarters with an own g, it records what the national and the firm-type fallbacks would have put there: n, bias and MAE in bp. Until now the holdout existed only in my 10-18 scratch scripts.
  - **Checked on the current events with the new demand file.** For D entrants: national bias −73.7 bp and MAE 118; firm type −10.0 and 63.5. The 10-18 scratch numbers were −66/113 and −5/61.
  - **`make_paper_numbers.py` `collect_entry_gfb` (cluster family) adds three keys:**
    - `phi.entry.gfb.firmtype.{B,D}`, with `.point`, `.lo` and `.hi`; the detail flags whether `hi` sits at the 0.9985 grid cap;
    - `phi.entry.gfb.share_national`, with `.B` and `.D`: the production run's share of event-quarters filled by the national median (about 30% for D);
    - `phi.entry.gfb.holdout`, as `.{national,firmtype}.{B,D}.{bias_bp,mae_bp}`.

    Tested on scratch inputs.

## Decisions for Pedro

1. **Which ext1 start to adopt** after `--report` (recommendation above).
2. **Whether the next full run takes the five-start engine** (it is in the repo now and in the TR5 overlay). Say so if you prefer to keep the production engine at two starts.

## Next (mine, as the inputs arrive)

1. **Sleep set lands:** ingest. Then:
   - `exhibits_post_cluster.py --run`, steps 1–3 and 6–12 (4 and 5 are done);
   - the entry-path slide sets;
   - plain `make_paper_numbers.py` and the changed-key report (old vs new by family; sign and significance changes);
   - K8.
2. **ext1 files land:** place them in `BLP_RESULTS/cluster_raw` (the current ones go to legacy), rebuild Table `blp_demand_comparison_noseg_spec12`, `make_blp_rc_table.py`, the BLP keys and the slide figures.
3. **TR5 lands:** unpack into `ESTIMATION_OUTPUT/TR5/<v>/`, then `collect_tr5`.
