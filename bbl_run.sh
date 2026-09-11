#!/bin/bash
# ==============================================================================
# bbl_run.sh — THE single front door for the BBL cost-estimation stage.
#
# WHAT MUST EXIST FIRST
#   1. gate G0 green                            (env_job.sh ENV_STEP=preflight)
#   2. the RC results, in data/output/blp: blp_results_E{k}_spec_12_{stage}.jls,
#      exactly where the RC ladder wrote them and where _result_path opens them
#   3. uploaded inputs (all from the BCB APIs, so no compute node can build them):
#        data/input/forward_rf_qoq.csv          (scrape_forward_rf.py)
#      and, for --multi-start only:
#        data/input/forward_rf_vintages.csv     (scrape_forward_rf.py --vintage-from 2016Q1)
#        data/input/bbl_transitions.json        (python bbl_transitions.py)
#      Both multi-start inputs are TINY — 1,836 rows and ~8 kB. The S simulated rate paths are
#      regenerated on the compute node from those parameters plus a fixed seed, so nothing that
#      scales with S is ever staged.
#      The fitted policy is NOT an upload: this script's polfunc pre-step builds it.
# WHAT TO RUN NEXT
#   bash cf_run.sh --do "demand_eval cf1 cf1_net cf4"
#   (pipeline_all.sh runs both in one shot, with --cost-afterok wiring the solves)
#
# WHAT IT SUBMITS
#   warmup --afterok--> polfunc --afterok--> fwd_sim ARRAY --afterANY--> solve --afterANY--> tables
#                                                                    \                    \
#                                                          afterany --> archive (COPY) <--/
#   The solve is afterANY the array deliberately: it globs whatever psi_dev shards
#   exist, so one flaky shard does not block the routine's costs. (If shard 0
#   failed there is no psi_eq, the solve errors, and --kill-on-invalid-dep cancels
#   its afterok downstream.)
#
#   THE TABLES STEP IS NOT OPTIONAL POLISH. Nothing is downloaded from this cluster, so a
#   number that exists only as a parquet or a JSON in data/output is a number nobody reads.
#   bbl_tables runs make_bbl_cost_tables.py on the node and echoes every table into its own
#   SLURM log, which is where the results are actually collected.
#
#   POLFUNC IS A DEFAULT PRE-STEP, not an option. It needs only the market panel
#   (uploaded as market_panel.parquet), which the sleepiness stage already brings
#   in, so fitting it here costs one short
#   CPU job and removes a 59 MB upload that could drift out of step with the panel
#   it was fitted on. Every routine's fwd_sim waits afterok on it because they all
#   read the one CSV it writes.
#
# Usage
#   bash bbl_run.sh --dry-run
#   bash bbl_run.sh
#   bash bbl_run.sh --routines "3" --shards 50
#   bash bbl_run.sh --multi-start --n-paths 8
#   bash bbl_run.sh --fwd-cpu
#
# Flags
#   --routines "3 4"   routine set (default from cluster_lib.sh)
#   --polfunc          accepted and a NO-OP — it names the default
#   --no-polfunc       skip the polfunc pre-step and use the CSV already on disk
#   --no-warmup        skip the pre-warm barrier
#   --starts|--multi-start   one forward curve PER LAUNCH QUARTER instead of one shared curve
#   --n-paths N        multi-start only: simulated rate paths averaged per deviation (default 8)
#   --psi-tag T        override the artifact tag (default _ms<N> with --multi-start, else empty)
#   --promote          let the solve copy its tagged cost_params onto the UNTAGGED name the
#                      counterfactuals read, but ONLY if the identification gate passes
#   --fwd-cpu|--fwd-gpu  where the fwd_sim array runs (default: GPU — see the advisory below)
#   --shards N         fwd_sim shard count (default 100)
#   --array-spec S     re-run a SUBSET of shards, e.g. 96-99 or 3,17,88
#   --no-tables        do not submit the terminal tables job
#   --no-zip           do not submit the terminal archive job
#   --skip-preflight   skip cluster_preflight.sh
#   --dry-run          print, submit nothing
#   -h                 this header
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"
set -e

ROUTINES_SRC=default
if [[ -n "${ROUTINES+set}" ]]; then ROUTINES_SRC=env; fi
ROUTINES="${ROUTINES:-${CL_ROUTINES_CF}}"
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
SHOCKS="${SHOCKS:-50}"; N_SHARDS="${N_SHARDS:-100}"
PERTURB_SCALE="${PERTURB_SCALE:-2.0}"; DEV_SCHEME="${DEV_SCHEME:-grid}"
BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}"
SHARD_TIME="${SHARD_TIME:-16:00:00}"; SOLVE_TIME="${SOLVE_TIME:-01:00:00}"
MEM="${MEM:-256G}"
DO_POLFUNC="${DO_POLFUNC:-1}"
DO_WARMUP="${DO_WARMUP:-1}"
DO_ZIP=1; DO_TABLES="${DO_TABLES:-1}"; SKIP_PREFLIGHT=0
# FWD_GPU=1 is the CODE DEFAULT and the effective default here — see the advisory printed below.
FWD_GPU="${FWD_GPU:-1}"
# Multi-start: one forward r^f curve per LAUNCH QUARTER instead of one shared curve, with
# N_PATHS simulated rate paths averaged per deviation. OFF by default — it changes the design,
# not a tuning knob — and it is the only thing that needs the two extra uploads.
MULTI_START="${MULTI_START:-0}"; N_PATHS="${N_PATHS:-8}"; PSI_TAG="${PSI_TAG:-}"
# --promote is OFF by default, which is the behaviour this script has always had. It matters
# because bbl_solve.py's identification gate is only load-bearing when it is passed: without it
# the solve returns 0 whatever the gate says (bbl_solve.py:1124-1126), so exit 3 never fires and
# cf1_net/cf3/cf5/cf6 keep reading the OLD untagged cost_params. Opting in turns the gate on in
# BOTH directions -- a pass overwrites the production file, a failure exits 3 and cancels the
# afterok dependents with no log of their own.
PROMOTE="${PROMOTE:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --routines)       ROUTINES="$2"; ROUTINES_SRC=flag; shift ;;
        --polfunc)        DO_POLFUNC=1 ;;          # names the default; kept so old command lines still parse
        --no-polfunc)     DO_POLFUNC=0 ;;
        --no-warmup)      DO_WARMUP=0 ;;
        --starts|--multi-start) MULTI_START=1 ;;
        --n-paths)        N_PATHS="$2"; shift ;;
        --psi-tag)        PSI_TAG="$2"; shift ;;
        --promote)        PROMOTE=1 ;;
        --fwd-cpu)        FWD_GPU=0 ;;
        --fwd-gpu)        FWD_GPU=1 ;;
        --shards)         N_SHARDS="$2"; shift ;;
        --array-spec)     ARRAY_SPEC="$2"; shift ;;
        --no-tables)      DO_TABLES=0 ;;
        --no-zip)         DO_ZIP=0 ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --dry-run)        CL_DRYRUN=1 ;;
        -h|--help)        sed -n '2,68p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

# The tag every artifact of this run carries: psi_eq_<tag>.parquet, psi_dev_<tag>_shard*.parquet
# and cost_params_<tag>.json, where <tag> = E{k}_spec_12_{stage}{PSI_TAG}. It has to be decided
# here, once, because three consumers derive filenames from it — bbl_fwd_sim.jl writes them,
# bbl_solve.py globs them and make_bbl_cost_tables.py matches them — and a tag that disagrees
# between any two of them looks exactly like a missing shard.
[[ "${MULTI_START}" == "1" && -z "${PSI_TAG}" ]] && PSI_TAG="_ms${N_PATHS}"

LOGD="$(cl_log_dir)"
if [[ "${SKIP_PREFLIGHT}" == "0" && "${CL_DRYRUN}" != "1" ]]; then
    bash "${CL_ROOT}/cluster_preflight.sh" --routines "${ROUTINES}" || {
        echo "bbl_run.sh: refusing to submit — the preflight found blockers (above)." >&2; exit 1; }
fi

cl_banner "BBL cost estimation$([[ "${CL_DRYRUN}" == "1" ]] && echo '  [DRY RUN — nothing is submitted]')" \
          "$(cl_routines_provenance "${ROUTINES}" "${ROUTINES_SRC}" "${CL_ROUTINES_CF}")" \
          "stage=${CF_STAGE} R=${R} seed=${SEED} shocks=${SHOCKS} shards=${N_SHARDS} mem=${MEM}" \
          "python: PY_MODULE='${PY_MODULE:-}' CONDA_ENV='${CONDA_ENV:-}' CF_PYTHON='${CF_PYTHON}'"
export PY_MODULE CONDA_ENV CF_PYTHON

# ── fwd_sim placement ────────────────────────────────────────────────────────
GPU_PARTITION="${GPU_PARTITION:-gpu_h200}"; GPUS="${GPUS:-h200:1}"
if [[ "${FWD_GPU}" == "1" ]]; then
    # The share aggregation moves to the H200 (compute_model_shares_gpu!), while
    # the 72 GB Pi products + compute_mu! stay on the HOST, so only ~30 G of share
    # buffers land on the device — no HBM-overflow risk.
    FWD_SB=(--partition="${GPU_PARTITION}" --gpus="${GPUS}"); FWD_GPU_ENV="CF_GPU=1"
    # ARRAY SIZING. gpu_h200 grants 16 GPUs but only 6 RUNNING JOBS per user, with a 2-day wall;
    # gpu_h100 grants 32 GPUs and 12 jobs. The binding constraint is therefore the JOB count, not
    # the GPU count, and an array task counts as a job: a 100-wide array on gpu_h200 runs 6 at a
    # time no matter what throttle is set, and each of those 6 pays the sysimage load and the
    # panel read again. Few LONG jobs that keep a resident array beat a wide one — the throttle
    # below caps concurrency at 8 so the array degrades to queueing rather than to thrash, and
    # SHARD_TIME (16 h) is deliberately close to the 2-day cap so a shard finishes inside one
    # allocation instead of being resubmitted.
    [[ -z "${ARRAY_THROTTLE+set}" ]] && ARRAY_THROTTLE=8   # default only when truly UNSET (empty => unlimited)
    cl_log "fwd_sim -> GPU (${GPU_PARTITION}, --gpus=${GPUS}, throttle=${ARRAY_THROTTLE:-none}); --fwd-cpu forces CPU"
    cl_log "[advisory] The forward sim runs on the GPU, which is the right target for the design"
    cl_log "           as it stands. The older note recording fwd_sim as memory-bandwidth bound at"
    cl_log "           ~0% GPU utilisation described a sim with ONE share evaluation per deviation:"
    cl_log "           the states were frozen, so the kernel launched once and the job was all host"
    cl_log "           work. With evolving states there are T share evaluations per deviation, and"
    cl_log "           cf_shares_at is the expensive kernel, so the device now carries the bulk of"
    cl_log "           the arithmetic. --fwd-cpu still forces CPU (day, --constraint=cpugen:turin)."
else
    # EXPLICIT CPU partition and NO --gpus: requesting a GPU and then leaving it
    # at ~0% util gets the job flagged by YCRC and lowers later priority. Naming
    # the partition here also stops a stray PARTITION= in the environment from
    # routing CPU fwd_sim onto gpu_h200. --constraint pins the microarchitecture:
    # `day` mixes cpugen:turin (AMD 9575f/9655) with cpugen:emeraldrapids (Intel
    # 8562Y+), and blp_sysimage_cpu.so is valid only on the cpugen it was BUILT
    # on. Must match the constraint used for ENV_STEP=sysimage_cpu.
    FWD_SB=(--partition="${CPU_PARTITION:-day}" --constraint="${CPU_CONSTRAINT:-cpugen:turin}")
    FWD_GPU_ENV="CF_GPU=0"
    cl_log "fwd_sim -> CPU (${CPU_PARTITION:-day}, ${CPU_CONSTRAINT:-cpugen:turin}, no GPU, throttle=${ARRAY_THROTTLE:-none})"
fi
THROTTLE="${ARRAY_THROTTLE:+%${ARRAY_THROTTLE}}"

# ── Preflight — every required input, INCLUDING the login-node Python probe ──
miss=0
cl_need_draws "${R}" "${SEED}" || miss=1
cl_need_rf_curve || miss=1
# The multi-start inputs are preflighted HERE, on the login node, and again inside bbl_job.sh
# before the sim starts. Twice on purpose: this check turns a missing upload into a refusal at
# submit time, while the in-job one covers a resume that skipped the preflight and an array task
# that outlived the file. Both are cheap; a 100-task GPU array that discovers the absence at
# t+10 min is not.
if [[ "${MULTI_START}" == "1" ]]; then
    cl_need_rf_vintages || miss=1
fi
# The vintages are multi-start only; the TRANSITIONS are not. psi_under calls
# assert_state_evolution on EVERY path (bbl_fwd_sim.jl), and evolving states default ON
# (cf_demand_eval.jl:133, CF_EVOLVING_STATES unset => on), so a plain single-start array
# without bbl_transitions.json does not quietly fall back to frozen states -- it dies task by
# task. Gating this check on --multi-start hid that. CF_EVOLVING_STATES=0 is the deliberate
# frozen-state opt-out, and it is honoured here so the two agree.
# Multi-start OR evolving states. Two independent reasons this file must exist, and the
# first is NOT covered by CF_EVOLVING_STATES: a multi-start run builds its rate paths from
# rate.process, so without the file --n-paths silently repeats the Focus mean N times.
# Gating on the state switch ALONE was a regression -- CF_EVOLVING_STATES=0 would have let
# a multi-start run through unchecked while still logging 'Preflight OK: ... transitions'.
if [[ "${MULTI_START}" == "1" || "${CF_EVOLVING_STATES:-1}" != "0" ]]; then
    cl_need_transitions || miss=1
fi

# The fitted policy is REQUIRED, always: deviations formed around OBSERVED spreads
# instead of the fitted policy make frac_bind ~ 0.5 mechanical and leave
# (omega, zeta, gamma) uninformative (V_Main line 551, sec 0A/9).
#
# When the polfunc job is in THIS graph, the CSV does not exist yet and must not be
# tested for — the fwd_sim jobs chain afterok that job, so SLURM enforces the
# ordering and an on-disk check here would refuse a run that was always going to
# produce it. When it is not, cl_need_polfunc resolves the CSV already on disk (bbl
# step folder first, then data/input for a hand-staged one) or refuses.
if [[ "${DO_POLFUNC}" == "1" ]]; then
    POLICY_CSV="${CL_STEP_BBL}/polfunc_fitted.csv"
else
    POLICY_CSV="${POLICY_CSV:-$(cl_need_polfunc)}" || miss=1
fi
for k in ${ROUTINES}; do cl_need_rc_jls "${k}" || miss=1; done
# The BBL solve is PYTHON. Verify the env NOW, on the login node, which shares
# modules + NFS with the compute nodes — instead of discovering it ~14 h from now
# when the fwd_sim array finally drains. On 2026-08-01 the solve died at t+4s on
# `No module named 'pandas'` and its afterok cascade auto-cancelled cf1_net with
# no log at all. PY_PREFLIGHT=0 skips.
if [[ "${PY_PREFLIGHT:-1}" == "1" ]]; then
    PY_REQ="${CL_PY_REQ_SOLVE}"
    [[ "${DO_POLFUNC}" == "1" ]] && PY_REQ="${CL_PY_REQ_POLFUNC}"
    cl_python_probe "${PY_REQ}" || {
        echo "MISSING the solve's Python stack (${PY_REQ})"
        echo "   env: PY_MODULE='${PY_MODULE:-}' CONDA_ENV='${CONDA_ENV:-}' CF_PYTHON='${CF_PYTHON}'"
        echo "   -> run  bash cluster_preflight.sh  — section (5) prints the one-line unblock"
        echo "      (PY_PREFLIGHT=0 skips this check; CONDA_ENV=\"\" opts out of conda entirely)"
        miss=1; }
fi
[[ "${miss}" == "0" ]] || { echo "Stage the missing input(s) (cluster/RUNBOOK.md step 0), then re-run."; exit 1; }
cl_log "Preflight OK: draws + forward r^f curve + RC results + solve Python env for routines: ${ROUTINES}"
if [[ "${MULTI_START}" == "1" ]]; then
    cl_log "Preflight OK: r^f vintages + transitions (multi-start, ${N_PATHS} paths, tag '${PSI_TAG}')"
fi

# ── fwd_sim flags ────────────────────────────────────────────────────────────
# Asset return r^j (V_Main eq 16, psi1 row). Both flags are read as the QUARTERLY
# NET margin (r^j - r^f); foundation_psi_basis adds r^f back so psi1 carries the
# GROSS r^j the paper requires. Left unset => r^j = r^f (zero asset margin),
# which makes a deposit worth only (rho - c) and forces omega-hat < 0 in eq:17.
#   ASSET_RETURN_COL=asset_gross_return_lag   per-obs, = 1 + asset_return_qoq_lag
#   ASSET_MARGIN=0.015                        constant quarterly net margin
# NOT `gross_return_lag` — that is 1 + the DEPOSIT rate (what the bank PAYS).
asset_flags=""
[[ -n "${ASSET_RETURN_COL:-}" ]] && asset_flags="--asset-return-col ${ASSET_RETURN_COL}"
[[ "${ASSET_MARGIN:-0}" != "0" ]] && asset_flags="${asset_flags} --asset-margin ${ASSET_MARGIN}"
policy_flag="--policy-csv ${POLICY_CSV}"
cl_log "fwd_sim sigma-hat <- FITTED policy: ${POLICY_CSV}$([[ "${DO_POLFUNC}" == "1" ]] && echo '  (produced by this run polfunc job)')"

# Multi-start flags. The two file paths point INTO data/input because that is where uploads
# live; the S rate paths themselves are built on the node from bbl_transitions.json plus the
# sim's own seed, so the upload stays at ~130 kB however large --n-paths gets. --psi-tag is
# passed to the sim so the artifacts it writes carry the tag the solve and the tables expect.
ms_flags=""
if [[ "${MULTI_START}" == "1" ]]; then
    ms_flags="--multi-start --rf-vintages ${CL_DATA_IN}/forward_rf_vintages.csv"
    ms_flags="${ms_flags} --transitions ${CL_DATA_IN}/bbl_transitions.json"
    ms_flags="${ms_flags} --n-paths ${N_PATHS} --psi-tag ${PSI_TAG}"
    cl_log "fwd_sim multi-start: one forward curve per launch quarter, ${N_PATHS} rate paths/deviation, tag '${PSI_TAG}'"
fi
bbl_extra="--shocks ${SHOCKS} --perturb-scale ${PERTURB_SCALE} --dev-scheme ${DEV_SCHEME} --beta ${BETA} --horizon ${HORIZON}${asset_flags:+ ${asset_flags}} ${policy_flag}${ms_flags:+ ${ms_flags}}"
# The solve reads the SAME tag it was written under. Passed as --psi-tag rather than folded into
# --suffix: the suffix is part of the stage name in every other consumer, and reusing it here
# would make cost_params_E3_spec_12_extended_ms8.json look like a different STAGE to anything
# that parses the tag.
# Built with an explicit `if`, never `$([[ ... ]] && echo ...)`: this file runs under
# `set -e` (line 72), and a command substitution whose test is FALSE returns nonzero,
# which makes the assignment itself nonzero and ends the script -- on the DEFAULT path,
# where PROMOTE=0.
promote_flag=""
if [[ "${PROMOTE}" == "1" ]]; then promote_flag=" --promote"; fi
# --profile inverts the subsampled criterion into omega/zeta confidence intervals; without it
# the identified branch of the cost table prints a dash in every 95% CI cell. It needs
# --subsample > 0, which bbl_solve.py defaults to 200.
solve_extra="--bootstrap 200 --profile${PSI_TAG:+ --psi-tag ${PSI_TAG}}${promote_flag}"

sub () {  # sub <jobname> <time> <extra sbatch args...>
    local name="$1" tlim="$2"; shift 2
    # CPU steps load blp_sysimage_cpu.so, which is valid ONLY on the cpugen it was built on,
    # and `day` mixes cpugen:turin (AMD 9575f/9655) with cpugen:emeraldrapids (Intel 8562Y+).
    # warmup and fwd_sim get the pin through FWD_SB; polfunc and solve did not, so they landed
    # wherever day had room and refused with "the image does not LOAD on <node>" -- and only
    # sometimes, which is worse than always. Skipped when the caller already pins a constraint
    # (the --fwd-cpu FWD_SB does) or when this is a GPU submission, whose image is the
    # gpu_h200-built one and which no cpugen constraint can satisfy.
    local cons="${CPU_CONSTRAINT:-cpugen:turin}" a
    for a in "$@"; do
        case "${a}" in --constraint*|--gpus*|--partition=gpu*) cons=""; break ;; esac
    done
    [[ "${PARTITION:-day}" == gpu* ]] && cons=""
    cl_sbatch -J "${name}" -t "${tlim}" \
        ${PARTITION:+--partition="${PARTITION}"} ${MEM:+--mem="${MEM}"} \
        ${cons:+--constraint="${cons}"} \
        -o "${LOGD}/${name}_%A_%a.out" -e "${LOGD}/${name}_%A_%a.err" "$@"
}

ALL_JIDS=""     # EVERY job id this run creates (declared BEFORE the first append)
solve_dep=""    # colon-joined solve job ids -> the archive waits on all of them

# ── Pre-warm barrier: precompile/load ONCE so the fwd_sim array does not stampede
#    the shared-NFS precompile/load lock (the cause of tasks sitting at 0% CPU).
warm_dep=""
if [[ "${DO_WARMUP}" == "1" ]]; then
    wj=$(sub "bbl_warmup" "${SOLVE_TIME}" "${FWD_SB[@]}" \
        --export=ALL,BBL_ROUTINE=${ROUTINES%% *},BBL_STAGE=${CF_STAGE},R=${R},SEED=${SEED},BBL_STEP=warmup,${FWD_GPU_ENV} \
        "${CL_ROOT}/bbl_job.sh")
    cl_say "  bbl_warmup -> ${wj}"
    warm_dep="--dependency=afterok:${wj}"
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${wj}"
fi

# BBL Step 1, the default pre-step. It writes the ONE shared policy CSV, so every
# routine's fwd_sim waits afterok on it. An explicit --mem: this is the only step in
# the chain that is a pandas fit over the whole market panel rather than a Julia
# simulation, and MEM (256G) is sized for the latter — 64G is what the fit needs and
# asking for four times that queues behind nodes it does not use.
fwd_dep="${warm_dep}"
if [[ "${DO_POLFUNC}" == "1" ]]; then
    pj=$(sub "bbl_polfunc" "${SOLVE_TIME}" ${warm_dep} --mem=64G \
        --export=ALL,BBL_STAGE=${CF_STAGE},R=${R},SEED=${SEED},BBL_STEP=polfunc \
        "${CL_ROOT}/bbl_job.sh")
    cl_say "  bbl_polfunc -> ${pj}  (--mem=64G; every fwd_sim waits afterok on it)"
    fwd_dep="--dependency=afterok:${pj}"
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${pj}"
fi

for k in ${ROUTINES}; do
    base_export="BBL_ROUTINE=${k},BBL_STAGE=${CF_STAGE},R=${R},SEED=${SEED}"
    cl_log "-- E${k} ${CF_STAGE} | R=${R} | shocks=${SHOCKS} over ${N_SHARDS} shards --"
    # --array-spec re-runs a SUBSET of shards without redoing the ones already on
    # disk: psi_dev_*_shard{i}of{N}.parquet files are independent and the solve
    # globs whatever exists (a CPU-computed shard is numerically identical to a
    # GPU one). N_SHARDS must stay the SAME as the original run — it is baked into
    # the filename and the (firm x delta) split. psi_eq is written by shard 0 only.
    arr=$(sub "bbl_fwd_E${k}" "${SHARD_TIME}" ${fwd_dep} "${FWD_SB[@]}" \
        --array="${ARRAY_SPEC:-0-$((N_SHARDS-1))}""${THROTTLE}" \
        --export=ALL,${base_export},BBL_STEP=fwd_sim,N_SHARDS=${N_SHARDS},BBL_EXTRA="${bbl_extra}",${FWD_GPU_ENV} \
        "${CL_ROOT}/bbl_job.sh")
    cl_say "  bbl_fwd_E${k} --array=${ARRAY_SPEC:-0-$((N_SHARDS-1))}${THROTTLE} -> ${arr}"
    slv=$(sub "bbl_solve_E${k}" "${SOLVE_TIME}" --dependency=afterany:"${arr}" \
        --export=ALL,${base_export},BBL_STEP=solve,BBL_EXTRA="${solve_extra}",PSI_TAG="${PSI_TAG}" \
        "${CL_ROOT}/bbl_job.sh")
    cl_say "  bbl_solve_E${k} -> ${slv}  (afterANY ${arr} — it globs the surviving psi_dev shards)"
    solve_dep="${solve_dep}:${slv}"
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${arr}:${slv}"
done

# ── Auto-archive, afterANY all solves. COPY is MANDATORY here: cf1_net/cf3/cf5/cf6
#    read cost_params_*.json IN PLACE, so moving them would strand the CF inputs.
#    The whole bbl step folder is packaged — polfunc, the psi shards and the cost
#    params are one family with one producer, and the download is complete rather
#    than incremental, so there is no --newer window to get wrong.
dep_csv="$(echo ${solve_dep} | sed 's/^://')"

# ── Tables, afterANY every solve. THE COLLECTION POINT OF THE WHOLE PHASE.
#    Nothing comes back from this cluster, so a cost_params JSON in data/output is not a result
#    anybody has. This step turns the solves into the four .tex/.md tables AND echoes them into
#    its own SLURM log, which is the copy that gets read. afterANY, like the archive: one
#    routine's solve failing must not cost the others their tables.
#    A short CPU job (a JSON read and some string formatting), so it takes the small --mem
#    rather than the 256G the simulation is sized for.
tables_dep="${dep_csv}"
if [[ "${DO_TABLES}" == "1" && -n "${dep_csv}" ]]; then
    tab_jid=$(sub "bbl_tables" "00:30:00" --dependency=afterany:${dep_csv} --mem=16G \
        --export=ALL,BBL_STAGE=${CF_STAGE},BBL_STEP=tables,PSI_TAG="${PSI_TAG}" \
        "${CL_ROOT}/bbl_job.sh")
    cl_say "  bbl_tables -> ${tab_jid}  (afterany ${dep_csv//:/, }; tables are ECHOED to its log)"
    cl_log "   -> read them with: cat ${LOGD}/bbl_tables_${tab_jid}_*.out"
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${tab_jid}"
    tables_dep="${dep_csv}:${tab_jid}"
fi

if [[ "${DO_ZIP}" == "1" && -n "${dep_csv}" ]]; then
    ZWRAP="cd '${CL_ROOT}' && bash cluster_archive.sh --set bbl --copy --tag \"\${SLURM_JOB_ID}\""
    # afterany the TABLES job as well, so the .tex/.md it writes into the step folder are inside
    # the archive rather than one job too late for it.
    zip_jid=$(cl_sbatch --dependency=afterany:${tables_dep} \
        -J bbl_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGD}/bbl_zip_%j.out" -e "${LOGD}/bbl_zip_%j.err" \
        --wrap "${ZWRAP}")
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${zip_jid}"
    cl_say "  bbl_zip -> ${zip_jid}  (afterany ${tables_dep//:/, }; set: bbl, --copy)"
    cl_log "   -> ${CL_STEP_DOWNLOAD}/bbl_outputs_<tag>.zip  (cost_params COPIED — originals stay for the CFs)"
fi

cl_log "Submitted BBL cost estimation for routines: ${ROUTINES}. Watch with: squeue -u \$USER"
# Machine-parseable handles for pipeline_all.sh. Keep BBL_ALL_JOBIDS the LAST line
# so `sed -n 's/^BBL_ALL_JOBIDS=//p' | tail -1` captures it cleanly.
echo "BBL_SOLVE_JOBIDS=${dep_csv}"
echo "BBL_ALL_JOBIDS=${ALL_JIDS}"
