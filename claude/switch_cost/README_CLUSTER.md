# Switch-cost investigation: cluster package

Sleepiness by deposit type, routines III and IV (E3, E4), spec 12. Two fit variants: `pooled`
(the production specification re-estimated by the copies) and `typeshift` (deposit-type shifts
in the phi index). The cluster runs both with the two-stage AME bootstrap (B = 999), which is
the inference the local runs could not do. Full notes: `CLUSTER_SCRIPTS.md` and the local report
`switch_cost_local_report_2026-10-13.md`.

**Nothing here may be unpacked before the user has described the cluster layout.** The commands
below are the proposal.

## What the zips hold

- `switch_cost_scripts_<stamp>.zip`: only NEW file names (`switch_cost_*.py`, `switch_cost_*.sh`,
  `utils/switch_cost_*.py`), plus `SWITCH_COST_MANIFEST.md5` (the files),
  `SWITCH_COST_REQUIRES.md5` (the shared modules the copies import, with the LOCAL md5 they were
  tested against) and this README. Unpacking cannot overwrite a file of the shared project.
- `switch_cost_data_<stamp>.zip`: the five inputs (about 62 MB), meant for
  `data/input/switch_cost/`, plus `SWITCH_COST_DATA.sha256`. `market_panel.parquet` and
  `demand_chars.parquet` are a matched pair (the second records the first's md5).

## Proposed steps (each one line; never `\` continuations)

1. Scripts, from `scripts/`: `unzip -n switch_cost_scripts_<stamp>.zip` (`-n` never overwrites),
   then `md5sum -c SWITCH_COST_MANIFEST.md5`.
2. Shared modules: `md5sum -c SWITCH_COST_REQUIRES.md5`. A mismatch means the cluster's shared
   module differs from the one the local runs used. Stop and report it; do not replace it.
3. Data, from `data/input/`: `mkdir -p switch_cost && cd switch_cost && unzip -n ../switch_cost_data_<stamp>.zip && sha256sum -c SWITCH_COST_DATA.sha256`.
4. Dry run: `bash switch_cost_sleep_run.sh --dry-run`.
5. Smoke run: `bash switch_cost_sleep_run.sh --routines 3 --variants pooled --boot-b 9`, then
   read `data/output/switch_cost_gates/.gate_switch_cost_G1_pooled.json` and the log in
   `scripts/switch_cost_logs/`.
6. Full run: `bash switch_cost_sleep_run.sh` (E3 and E4, both variants, B = 999). The demand
   prep stays local unless `--prep` is added.
7. Collect: `bash switch_cost_collect.sh`. It writes one zip into
   `data/output/switch_cost_download/` and never moves or deletes anything.

## What the run writes

Only `data/output/switch_cost_*`, `data/switch_cost_root/`, `scripts/switch_cost_logs/`. Every
job name starts with `switch_cost_`, and the submission guard refuses any other name, log path or
payload. The conda env `dep_comp_blp` is used read-only; nothing is installed or built.
