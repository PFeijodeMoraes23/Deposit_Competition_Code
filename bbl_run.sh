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
#      Both multi-start inputs are TINY: 36 x (T+1) curve rows and ~8 kB. The S simulated rate
#      paths are regenerated on the compute node from those parameters plus a fixed seed, so
#      nothing that scales with S is ever staged. The curve must reach h = T (checked; see below).
#      The fitted policy is NOT an upload: this script's polfunc pre-step builds it.
#   4. scripts/bbl_discount.env, the registry of beta and T (see BETA AND HORIZON below).
# WHAT TO RUN NEXT
#   bash cf_run.sh --do "demand_eval cf1 cf1_net cf4"
#   (pipeline_all.sh runs both in one shot, with --cost-afterok wiring the solves)
# OPERATING IT: BBL_RUNBOOK.md (memory probe, full run, status, recovery). bbl_status.sh shows one
#   screen of state; bbl_sizing.sh turns the probe into MEM/PACK and the launch command;
#   bbl_cancel.sh cancels a chain in dependency order.
#
# WHAT IT SUBMITS (per routine k; warmup/polfunc once; tables and archive once)
#
#   polfunc --afterok--> fwd E_k [gpu_h200, PACK_H200] --+
#                        fwd E_k [gpu_h100, PACK_H100] --+--afterany--> sweep E_k (attempt 0)
#                                                                          |
#            complete: exit 0 --------------------------------------------+--afterok--> solve E_k
#            gaps:     re-submit EXACTLY the missing shards --> sweep E_k (attempt 1) ...
#                      and MOVE the solve's afterok onto that sweep (same solve job id)
#            retries spent: exit 1 --> the solve is CANCELLED, never run on a partial set
#
#   solve E_1..E_n --afterok (all)--> tables --afterok--> archive (slim: this tag only)
#
#   The solve is afterok its FINAL sweep, and the tables and archive are afterok EVERY solve:
#   nothing downstream ever renders or zips a partial result. (Before the sweep existed the
#   solve was afterany the array and E1 was once estimated from 293 of 300 shards.)
#
#   THE TABLES STEP IS NOT OPTIONAL POLISH. Nothing is downloaded from this cluster but the
#   archive, and bbl_tables echoes every table into its own SLURM log.
#
#   POLFUNC IS A DEFAULT PRE-STEP, not an option. It needs only the market panel, costs one short
#   CPU job and removes a 59 MB upload that could drift out of step with the panel. Every
#   routine's fwd_sim waits afterok on it because they all read the one CSV it writes. A second
#   invocation while the first one's arrays still run must pass --no-polfunc: a re-fit rewrites
#   the CSV those tasks read at startup.
#
# PACKING (GPU). The gpu QOS caps JOBS, not GPUs: gpu_h200 6 jobs / 16 GPUs, gpu_h100 (a separate
#   QOS) 12 jobs / 32 GPUs. One shard per job therefore uses 18 of the 48 GPUs allowed. PACK=k
#   runs k shards in one job, one GPU each (--gpus=<type>:k, --mem=k*MEM, --cpus-per-task=k*8),
#   and the array throttle becomes min(MaxJobsPU, floor(MaxGPU/k)), logged per partition.
#   PACK is the common default (unset: h200 4, h100 3); PACK_H200 / PACK_H100 set one partition.
#   PACK=1 is the one-shard-per-job submission, unchanged. Any PACK x MEM beyond a node, or
#   PACK beyond a node's GPUs, is refused here (cl_bbl_check_packing). MEM=256G is the T=50
#   sizing; at another T take MEM and PACK from the memory probe (bbl_sizing.sh).
#
# WALL TIME is derived unless --shard-time is given: 26 min (the measured max GPU shard at T=50)
#   x T/50 x shard size x contention 1.3 x safety 1.5, rounded up to 30 min and capped at the
#   partition limit, and logged: 04:30:00 at T=250. A sweep re-run multiplies it by 1.5 per attempt.
#
# BETA AND HORIZON come from bbl_discount.env (BBL_BETA, BBL_HORIZON), the one registry of both.
#   --beta / --horizon (or BETA / HORIZON in the environment) override it; the banner says which
#   of the three each value came from, and a missing registry with no override is refused. They
#   flow into the sim, the wall derivation and the sweep's provenance check, and the uploaded
#   Focus curve must reach h = T or the run is refused (ALLOW_RF_PAD=1 accepts padding).
#
# Usage
#   bash bbl_run.sh --dry-run
#   bash bbl_run.sh --probe --shards 300 --multi-start --n-paths 1
#   bash bbl_run.sh --routines "3 4" --fwd-gpu --multi-partition --shards 300 --multi-start \
#        --n-paths 1 --psi-tag _ms2 --no-warmup --no-polfunc --mem-h200 <M> --mem-h100 <M> \
#        --pack-h200 <k> --pack-h100 <k>                  (bbl_sizing.sh prints this line)
#   bash bbl_run.sh --routines 3 --shards 300 --psi-tag _ms2 --repair
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
#   --beta B --horizon T   override bbl_discount.env (env BETA / HORIZON do the same)
#   --shocks S         deviations per firm (env SHOCKS; default 50)
#   --fwd-cpu|--fwd-gpu  where the fwd_sim array runs (default: GPU)
#   --partitions "P.." GPU partitions for fwd_sim (default gpu_h200); indices split DISJOINTLY
#   --multi-partition  = --partitions "gpu_h200 gpu_h100"
#   --pack K           shards per job on every GPU partition (env PACK; default h200 4, h100 3)
#   --pack-h200 K --pack-h100 K --mem-h200 X --mem-h100 X   per-partition packing / memory
#   --mem X            per-shard memory on every partition (env MEM; default 256G, the T=50 sizing)
#   --shard-time HH:MM:SS   fix the fwd wall instead of deriving it
#   --shards N         fwd_sim shard count (default 100). NEVER change it between a run and its
#                      re-runs: it is baked into every file name and the round-robin split.
#   --shard-list S     launch only these shards, e.g. 0-3 or 3,50-299 (alias: --array-spec)
#   --max-retries N    sweep re-runs before it gives up and cancels the solve (default 3)
#   --sweep-partitions "P.."  where the sweep's re-runs go (default: --partitions)
#   --no-sweep         no sweep: the solve is afterany the arrays (its own gate still refuses gaps)
#   --no-solve         no solve (and so no tables/archive): fwd jobs (+ sweep) only, e.g. a test
#   --repair           re-attach a dead chain: sweep now (re-runs whatever is missing), then
#                      solve -> tables -> archive. Design settings come from the run's context.
#   --probe            the MEMORY PROBE: routine 3 (or --routines k) at this T, tag _probe<T>,
#                      two concurrent gpu_h200 jobs with generous memory so neither can OOM:
#                      shards 0..PROBE_PACK-1 PACKED in one job (PROBE_PACK=4 x PROBE_MEM_PACKED=450G)
#                      and shard PROBE_PACK alone (PROBE_MEM_UNPACKED=500G); wall x BBL_PROBE_WALL_MULT
#                      (2). No sweep/solve/tables/archive. Read it with: bash bbl_sizing.sh
#   --no-probe-control the probe without its unpacked job
#   --force            submit even though a live job of this routine/tag claims the same shards
#   --zip-full         archive the whole bbl folder (default: slim, this tag's psi + results)
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

# The design parameters the caller GAVE (flag or environment), recorded before any default fills
# them in: --repair inherits every one it was not given from the original run.
_GIVEN=" "
for _v in N_SHARDS SHOCKS HORIZON BETA; do
    if [[ -n "${!_v+set}" && -n "${!_v}" ]]; then _GIVEN="${_GIVEN}${_v} "; fi
done

ROUTINES_SRC=default
if [[ -n "${ROUTINES+set}" ]]; then ROUTINES_SRC=env; fi
ROUTINES="${ROUTINES:-${CL_ROUTINES_CF}}"
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
SHOCKS="${SHOCKS:-50}"; N_SHARDS="${N_SHARDS:-100}"
PERTURB_SCALE="${PERTURB_SCALE:-2.0}"; DEV_SCHEME="${DEV_SCHEME:-grid}"
# Empty = read from bbl_discount.env after the flags (cl_bbl_discount_fill); *_SRC records where
# each value came from, for the banner and the run's context.
BETA="${BETA:-}"; HORIZON="${HORIZON:-}"
BETA_SRC=""; HORIZON_SRC=""
if [[ -n "${BETA}" ]]; then BETA_SRC="env BETA"; fi
if [[ -n "${HORIZON}" ]]; then HORIZON_SRC="env HORIZON"; fi
# Empty SHARD_TIME = derived from HORIZON (cl_bbl_wall_minutes); a value fixes it.
SHARD_TIME="${SHARD_TIME:-}"; SOLVE_TIME="${SOLVE_TIME:-01:00:00}"
_MEM_GIVEN="${MEM:-}${MEM_H200:-}${MEM_H100:-}"
MEM="${MEM:-256G}"
DO_POLFUNC="${DO_POLFUNC:-1}"
DO_WARMUP="${DO_WARMUP:-1}"
DO_ZIP=1; DO_TABLES="${DO_TABLES:-1}"; SKIP_PREFLIGHT=0
DO_SWEEP="${DO_SWEEP:-1}"; DO_SOLVE=1
BBL_MAX_RETRIES="${BBL_MAX_RETRIES:-3}"
# FWD_GPU=1 is the CODE DEFAULT and the effective default here.
FWD_GPU="${FWD_GPU:-1}"
FWD_PARTITIONS="${FWD_PARTITIONS:-${GPU_PARTITION:-gpu_h200}}"
SWEEP_PARTITIONS="${SWEEP_PARTITIONS:-}"
SHARD_LIST="${SHARD_LIST:-${ARRAY_SPEC:-}}"
# Multi-start: one forward r^f curve per LAUNCH QUARTER instead of one shared curve, with
# N_PATHS simulated rate paths averaged per deviation. OFF by default — it changes the design,
# not a tuning knob — and it is the only thing that needs the two extra uploads.
MULTI_START="${MULTI_START:-0}"; N_PATHS="${N_PATHS:-8}"; PSI_TAG="${PSI_TAG:-}"
# --promote is OFF by default. bbl_solve.py's identification gate is only load-bearing when it
# is passed: without it the solve returns 0 whatever the gate says, so exit 3 never fires and
# cf1_net/cf3/cf5/cf6 keep reading the untagged cost_params. Opting in turns the gate on in
# BOTH directions -- a pass overwrites the production file, a failure exits 3 and cancels the
# afterok dependents (here: the tables and the archive too) with no log of their own.
PROMOTE="${PROMOTE:-0}"
FORCE=0; REPAIR=0; PROBE=0; PROBE_CONTROL=1
ZIP_SLIM="${ZIP_SLIM:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --routines)         ROUTINES="$2"; ROUTINES_SRC=flag; shift ;;
        --polfunc)          DO_POLFUNC=1 ;;          # names the default; kept so existing command lines still parse
        --no-polfunc)       DO_POLFUNC=0 ;;
        --no-warmup)        DO_WARMUP=0 ;;
        --starts|--multi-start) MULTI_START=1 ;;
        --n-paths)          N_PATHS="$2"; shift ;;
        --psi-tag)          PSI_TAG="$2"; shift ;;
        --promote)          PROMOTE=1 ;;
        --beta)             BETA="$2"; BETA_SRC="--beta"; _GIVEN="${_GIVEN}BETA "; shift ;;
        --horizon)          HORIZON="$2"; HORIZON_SRC="--horizon"; _GIVEN="${_GIVEN}HORIZON "; shift ;;
        --shocks)           SHOCKS="$2"; _GIVEN="${_GIVEN}SHOCKS "; shift ;;
        --fwd-cpu)          FWD_GPU=0 ;;
        --fwd-gpu)          FWD_GPU=1 ;;
        --partitions)       FWD_PARTITIONS="$2"; shift ;;
        --multi-partition)  FWD_PARTITIONS="gpu_h200 gpu_h100" ;;
        --sweep-partitions) SWEEP_PARTITIONS="$2"; shift ;;
        --pack)             PACK="$2"; shift ;;
        --pack-h200)        PACK_H200="$2"; shift ;;
        --pack-h100)        PACK_H100="$2"; shift ;;
        --mem)              MEM="$2"; _MEM_GIVEN="${_MEM_GIVEN}$2"; shift ;;
        --mem-h200)         MEM_H200="$2"; _MEM_GIVEN="${_MEM_GIVEN}$2"; shift ;;
        --mem-h100)         MEM_H100="$2"; _MEM_GIVEN="${_MEM_GIVEN}$2"; shift ;;
        --shard-time)       SHARD_TIME="$2"; shift ;;
        --shards)           N_SHARDS="$2"; _GIVEN="${_GIVEN}N_SHARDS "; shift ;;
        --shard-list|--array-spec) SHARD_LIST="$2"; shift ;;
        --max-retries)      BBL_MAX_RETRIES="$2"; shift ;;
        --no-sweep)         DO_SWEEP=0 ;;
        --no-solve)         DO_SOLVE=0 ;;
        --repair)           REPAIR=1 ;;
        --probe)            PROBE=1 ;;
        --no-probe-control) PROBE_CONTROL=0 ;;
        --force)            FORCE=1 ;;
        --zip-full)         ZIP_SLIM=0 ;;
        --no-tables)        DO_TABLES=0 ;;
        --no-zip)           DO_ZIP=0 ;;
        --skip-preflight)   SKIP_PREFLIGHT=1 ;;
        --dry-run)          CL_DRYRUN=1 ;;
        -h|--help)          sed -n '2,119p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

# ── beta and T: flag > environment > bbl_discount.env ────────────────────────
cl_bbl_discount_fill BETA BBL_BETA || exit 2
cl_bbl_discount_fill HORIZON BBL_HORIZON || exit 2
[[ "${BETA}" =~ ^0?\.[0-9]+$|^1(\.0*)?$ ]] || { cl_err "REFUSING: beta '${BETA}' (${BETA_SRC}) is not a number in (0, 1]."; exit 2; }
[[ "${HORIZON}" =~ ^[1-9][0-9]*$ ]] || { cl_err "REFUSING: horizon '${HORIZON}' (${HORIZON_SRC}) is not a positive integer."; exit 2; }

# ── The two special modes ────────────────────────────────────────────────────
# --probe: the MEMORY PROBE. Two concurrent jobs on the same partition at the same T, both with
# generous memory so neither can OOM: shards 0..PROBE_PACK-1 PACKED in one job (PROBE_PACK x
# PROBE_MEM_PACKED) and shard PROBE_PACK alone (PROBE_MEM_UNPACKED). Every shard logs its own peak
# RSS ("[BBL] shard i peak RSS X GiB"), because sacct's MaxRSS of a packed job sums its children;
# packed vs unpacked elapsed at the SAME T is the contention ratio. bbl_sizing.sh reads both and
# prints MEM, PACK_H200, PACK_H100 and the launch command. Its own tag (_probe<T>) keeps its files
# out of every production glob: the solve's shard filter, the tables' tag pattern and the slim
# archive all match a tag only as the whole remainder of the name.
if [[ "${PROBE}" == "1" ]]; then
    [[ "${REPAIR}" == "0" ]] || { echo "--probe and --repair are exclusive" >&2; exit 2; }
    [[ "${ROUTINES_SRC}" == "flag" ]] || ROUTINES=3
    read -r -a _r <<< "${ROUTINES}"
    (( ${#_r[@]} == 1 )) || { echo "--probe runs ONE routine (got '${ROUTINES}')" >&2; exit 2; }
    [[ "${FWD_GPU}" == "1" ]] || { echo "--probe measures the GPU shards; drop --fwd-cpu" >&2; exit 2; }
    DO_WARMUP=0; DO_POLFUNC=0; DO_SWEEP=0; DO_SOLVE=0; DO_TABLES=0; DO_ZIP=0; PY_PREFLIGHT=0
    PROBE_PARTITION="${PROBE_PARTITION:-gpu_h200}"; FWD_PARTITIONS="${PROBE_PARTITION}"
    PROBE_PACK="${PROBE_PACK:-4}"
    PROBE_MEM_PACKED="${PROBE_MEM_PACKED:-450G}"; PROBE_MEM_UNPACKED="${PROBE_MEM_UNPACKED:-500G}"
    BBL_PROBE_WALL_MULT="${BBL_PROBE_WALL_MULT:-2}"
    [[ "${PROBE_PACK}" =~ ^[2-9]$ ]] || { echo "--probe: PROBE_PACK must be 2..9 (got '${PROBE_PACK}')" >&2; exit 2; }
    _PK="$(cl_bbl_pkey "${PROBE_PARTITION}")"
    printf -v "PACK_${_PK}" '%s' "${PROBE_PACK}"; printf -v "MEM_${_PK}" '%s' "${PROBE_MEM_PACKED}"
    PSI_TAG="${PSI_TAG:-_probe${HORIZON}}"
    [[ "${PSI_TAG}" == *probe* ]] || { cl_err "REFUSING --probe: tag '${PSI_TAG}' must contain 'probe', so a probe can never write under a production tag."; exit 2; }
fi
# --repair: no new launch. A sweep starts at once and re-runs whatever is missing (or nothing),
# then solve -> tables -> archive exactly as in a full run. The policy CSV must already be on
# disk, and it must be the one the original shards were simulated with.
if [[ "${REPAIR}" == "1" ]]; then DO_WARMUP=0; DO_POLFUNC=0; DO_SWEEP=1; fi

# The tag every artifact of this run carries: psi_eq_<tag>.parquet, psi_dev_<tag>_shard*.parquet
# and cost_params_<tag>.json, where <tag> = E{k}_spec_12_{stage}{PSI_TAG}. It has to be decided
# here, once, because three consumers derive filenames from it — bbl_fwd_sim.jl writes them,
# bbl_solve.py globs them and make_bbl_cost_tables.py matches them — and a tag that disagrees
# between any two of them looks exactly like a missing shard.
if [[ "${MULTI_START}" == "1" && -z "${PSI_TAG}" ]]; then PSI_TAG="_ms${N_PATHS}"; fi

LOGD="$(cl_log_dir)"
if [[ "${SKIP_PREFLIGHT}" == "0" && "${CL_DRYRUN}" != "1" ]]; then
    bash "${CL_ROOT}/cluster_preflight.sh" --routines "${ROUTINES}" || {
        echo "bbl_run.sh: refusing to submit — the preflight found blockers (above)." >&2; exit 1; }
fi

_mode="launch"; [[ "${REPAIR}" == "1" ]] && _mode="REPAIR"; [[ "${PROBE}" == "1" ]] && _mode="PROBE"
cl_banner "BBL cost estimation (${_mode})$([[ "${CL_DRYRUN}" == "1" ]] && echo '  [DRY RUN — nothing is submitted]')" \
          "$(cl_routines_provenance "${ROUTINES}" "${ROUTINES_SRC}" "${CL_ROUTINES_CF}")" \
          "stage=${CF_STAGE} R=${R} seed=${SEED} shocks=${SHOCKS} shards=${N_SHARDS} mem=${MEM}" \
          "beta=${BETA} (${BETA_SRC})  T=${HORIZON} (${HORIZON_SRC})" \
          "python: PY_MODULE='${PY_MODULE:-}' CONDA_ENV='${CONDA_ENV:-}' CF_PYTHON='${CF_PYTHON}'"
export PY_MODULE CONDA_ENV CF_PYTHON
cl_say "BBL ${_mode}: routines ${ROUTINES} | tag '${PSI_TAG}' | ${N_SHARDS} shards | beta=${BETA} (${BETA_SRC}) T=${HORIZON} (${HORIZON_SRC}) shocks=${SHOCKS}"

# ── fwd_sim placement ────────────────────────────────────────────────────────
if [[ "${FWD_GPU}" == "1" ]]; then
    FWD_GPU_ENV="CF_GPU=1"
    read -r -a _parts <<< "${FWD_PARTITIONS}"
    (( ${#_parts[@]} > 0 )) || { echo "REFUSING: --partitions is empty" >&2; exit 2; }
    for _p in "${_parts[@]}"; do
        if [[ "$(cl_bbl_pkey "${_p}")" == "CPU" ]]; then
            cl_err "REFUSING: '${_p}' is not a GPU partition (use --fwd-cpu for the CPU path)."; exit 2
        fi
        cl_bbl_check_packing "${_p}" || exit 2
    done
    # With evolving states there are T share evaluations per deviation and cf_shares_at is the
    # expensive kernel, so the device carries the bulk of the arithmetic. GPU psi == CPU psi to
    # 3e-14 (2026-09-20).
    cl_log "fwd_sim -> GPU (${FWD_PARTITIONS}); --fwd-cpu forces CPU (day, --constraint=cpugen:turin)"
else
    # EXPLICIT CPU partition and NO --gpus: requesting a GPU and then leaving it at ~0% util gets
    # the job flagged by YCRC. --constraint pins the microarchitecture: `day` mixes cpugen:turin
    # with cpugen:emeraldrapids, and blp_sysimage_cpu.so is valid only on the cpugen it was
    # BUILT on. Packing is a GPU notion: CPU shards run one per job.
    FWD_GPU_ENV="CF_GPU=0"
    FWD_PARTITIONS="${CPU_PARTITION:-day}"
    if [[ -n "${PACK:-}${PACK_H200:-}${PACK_H100:-}" ]]; then cl_say "  (PACK is ignored on the CPU path: one shard per job)"; fi
fi
for _p in ${FWD_PARTITIONS}; do
    _k="$(_cl_bbl_pack_for "${_p}")"
    _ti="$(cl_bbl_throttle "${_p}" "${_k}")"
    _wi="$(cl_bbl_wall_minutes "${_p}" "${_k}" 1)"
    if [[ "${FWD_GPU}" == "1" ]]; then
        _mem="$(cl_bbl_part "${_p}" mem)"; _mb="$(cl_mem_mb "${_mem}")"
        cl_say "fwd_sim [${_p}] PACK=${_k} x MEM ${_mem} = $(( _k * _mb )) of $(cl_bbl_part "${_p}" node_mem_mb) MB/node, ${_k} of $(cl_bbl_part "${_p}" node_gpus) GPUs; throttle ${_ti#*|}"
    else
        cl_say "fwd_sim [${_p}] one shard per job, MEM ${MEM}; throttle ${_ti#*|}"
    fi
    cl_say "fwd_sim [${_p}] wall $(cl_min_to_wall "${_wi%%|*}") = ${_wi#*|}"
done
# MEM=256G was sized at T=50 (max shard 179 GiB). RSS grows with T, so a launch at another T that
# names no memory is flagged loudly; the memory probe and bbl_sizing.sh give the value to pass.
if [[ "${PROBE}" != "1" && -z "${_MEM_GIVEN}" && "${HORIZON}" != "50" ]]; then
    cl_err "[!] WARNING: no --mem/--mem-h200/--mem-h100 given, so every shard gets the default ${MEM}, which was"
    cl_err "    sized at T=50. This run is T=${HORIZON}; RSS grows with T. Run the memory probe and pass what"
    cl_err "    bash bbl_sizing.sh prints (BBL_RUNBOOK.md section 1), or the shards may be OOM-killed."
fi

# ── Preflight — every required input, INCLUDING the login-node Python probe ──
miss=0
cl_need_draws "${R}" "${SEED}" || miss=1
cl_need_rf_curve || miss=1
# The multi-start inputs are preflighted HERE, on the login node, and again inside bbl_job.sh
# before the sim starts. Twice on purpose: this check turns a missing upload into a refusal at
# submit time, while the in-job one covers a resume that skipped the preflight and an array task
# that outlived the file.
if [[ "${MULTI_START}" == "1" ]]; then
    cl_need_rf_vintages || miss=1
fi
# The vintages are multi-start only; the TRANSITIONS are not. psi_under calls
# assert_state_evolution on EVERY path (bbl_fwd_sim.jl), and evolving states default ON
# (cf_demand_eval.jl:133, CF_EVOLVING_STATES unset => on), so a plain single-start array
# without bbl_transitions.json does not quietly fall back to frozen states -- it dies task by
# task. CF_EVOLVING_STATES=0 is the deliberate frozen-state opt-out, and it is honoured here.
# Multi-start OR evolving states. Two independent reasons this file must exist, and the
# first is NOT covered by CF_EVOLVING_STATES: a multi-start run builds its rate paths from
# rate.process, so without the file --n-paths silently repeats the Focus mean N times.
if [[ "${MULTI_START}" == "1" || "${CF_EVOLVING_STATES:-1}" != "0" ]]; then
    cl_need_transitions || miss=1
fi
# The Focus curves must reach the horizon (cl_bbl_check_curve). The file this run's sim reads is
# a refusal; the other one (forward_rf_qoq.csv under --multi-start) is a one-line note, since the
# counterfactuals read it at their own horizon. A --repair checks against the ORIGINAL run's T,
# inside the routine loop, once its context is loaded.
if [[ "${REPAIR}" != "1" ]]; then
    cl_bbl_check_curve "${HORIZON}" "${MULTI_START}" || miss=1
    if [[ "${MULTI_START}" == "1" ]]; then cl_bbl_check_curve "${HORIZON}" 0 "" 1; fi
fi

# The fitted policy is REQUIRED, always: deviations formed around OBSERVED spreads instead of
# the fitted policy make frac_bind ~ 0.5 mechanical and leave (omega, zeta, gamma)
# uninformative (V_Main line 551, sec 0A/9). When the polfunc job is in THIS graph the CSV does
# not exist yet and must not be tested for — the fwd_sim jobs chain afterok that job.
if [[ "${DO_POLFUNC}" == "1" ]]; then
    POLICY_CSV="${CL_STEP_BBL}/polfunc_fitted.csv"
else
    POLICY_CSV="${POLICY_CSV:-$(cl_need_polfunc)}" || miss=1
fi
for k in ${ROUTINES}; do cl_need_rc_jls "${k}" || miss=1; done
# The BBL solve is PYTHON. Verify the env NOW, on the login node, instead of discovering it when
# the array drains. On 2026-08-01 the solve died at t+4s on `No module named 'pandas'` and its
# afterok cascade auto-cancelled cf1_net with no log at all. PY_PREFLIGHT=0 skips.
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
cl_log "Preflight OK: draws + forward r^f curve (h >= T=${HORIZON}) + RC results + solve Python env for routines: ${ROUTINES}"
if [[ "${MULTI_START}" == "1" ]]; then
    cl_log "Preflight OK: r^f vintages + transitions (multi-start, ${N_PATHS} paths, tag '${PSI_TAG}')"
fi

# ── fwd_sim flags ────────────────────────────────────────────────────────────
# Asset return r^j (V_Main eq 16, psi1 row). Both flags are read as the QUARTERLY
# NET margin (r^j - r^f); foundation_psi_basis adds r^f back so psi1 carries the
# GROSS r^j the paper requires. Left unset => r^j = r^f (zero asset margin).
#   ASSET_RETURN_COL=asset_gross_return_lag   per-obs, = 1 + asset_return_qoq_lag
#   ASSET_MARGIN=0.015                        constant quarterly net margin
# NOT `gross_return_lag` — that is 1 + the DEPOSIT rate (what the bank PAYS).
asset_flags=""
[[ -n "${ASSET_RETURN_COL:-}" ]] && asset_flags="--asset-return-col ${ASSET_RETURN_COL}"
[[ "${ASSET_MARGIN:-0}" != "0" ]] && asset_flags="${asset_flags} --asset-margin ${ASSET_MARGIN}"
policy_flag="--policy-csv ${POLICY_CSV}"
cl_log "fwd_sim sigma-hat <- FITTED policy: ${POLICY_CSV}$([[ "${DO_POLFUNC}" == "1" ]] && echo '  (produced by this run polfunc job)')"

# Multi-start flags. The two file paths point INTO data/input because that is where uploads
# live; --psi-tag is passed to the sim so the artifacts it writes carry the tag the solve and
# the tables expect.
ms_flags=""
if [[ "${MULTI_START}" == "1" ]]; then
    ms_flags="--multi-start --rf-vintages ${CL_DATA_IN}/forward_rf_vintages.csv"
    ms_flags="${ms_flags} --transitions ${CL_DATA_IN}/bbl_transitions.json"
    ms_flags="${ms_flags} --n-paths ${N_PATHS} --psi-tag ${PSI_TAG}"
    cl_log "fwd_sim multi-start: one forward curve per launch quarter, ${N_PATHS} rate paths/deviation, tag '${PSI_TAG}'"
elif [[ -n "${PSI_TAG}" ]]; then
    # A single-start run under an explicit tag (a probe, a test): the sim must write that tag.
    ms_flags="--psi-tag ${PSI_TAG}"
fi
# beta and T are ALWAYS passed explicitly, so the sim never falls back to its own registry read
# and the value in every job's command line is the one this banner printed.
bbl_extra="--shocks ${SHOCKS} --perturb-scale ${PERTURB_SCALE} --dev-scheme ${DEV_SCHEME} --beta ${BETA} --horizon ${HORIZON}${asset_flags:+ ${asset_flags}} ${policy_flag}${ms_flags:+ ${ms_flags}}"
# The solve reads the SAME tag it was written under. Passed as --psi-tag rather than folded into
# --suffix: the suffix is part of the stage name in every other consumer.
# Built with an explicit `if`, never `$([[ ... ]] && echo ...)`: this file runs under
# `set -e`, and a command substitution whose test is FALSE returns nonzero, which makes the
# assignment itself nonzero and ends the script -- on the DEFAULT path, where PROMOTE=0.
promote_flag=""
if [[ "${PROMOTE}" == "1" ]]; then promote_flag=" --promote"; fi
# --profile inverts the subsampled criterion into omega/zeta confidence intervals; without it
# the identified branch of the cost table prints a dash in every 95% CI cell.
solve_extra="--bootstrap 200 --profile${PSI_TAG:+ --psi-tag ${PSI_TAG}}${promote_flag}"

sub () {  # sub <jobname> <time> <extra sbatch args...>
    local name="$1" tlim="$2"; shift 2
    # CPU steps load blp_sysimage_cpu.so, valid ONLY on the cpugen it was built on; `day` mixes
    # cpugen:turin with cpugen:emeraldrapids. Skipped when the caller already pins a constraint
    # or when this is a GPU submission, whose image no cpugen constraint can satisfy.
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
_ctx_get () { ( . "$1"; printf '%s' "${!2-}" ); }   # one value out of a context.env

ALL_JIDS=""     # EVERY job id this run creates (declared BEFORE the first append)
solve_dep=""    # colon-joined solve job ids -> the tables and the archive wait on all of them

# ── Pre-warm barrier: precompile/load ONCE so the fwd_sim array does not stampede
#    the shared-NFS precompile/load lock (the cause of tasks sitting at 0% CPU).
if [[ "${FWD_GPU}" == "1" ]]; then
    _p0="${FWD_PARTITIONS%% *}"; WARM_SB=(--partition="${_p0}" --gpus="$(cl_bbl_part "${_p0}" gpu_type):1")
else
    WARM_SB=(--partition="${CPU_PARTITION:-day}" --constraint="${CPU_CONSTRAINT:-cpugen:turin}")
fi
warm_dep=""
if [[ "${DO_WARMUP}" == "1" ]]; then
    wj=$(sub "bbl_warmup" "${SOLVE_TIME}" "${WARM_SB[@]}" \
        --export=ALL,BBL_ROUTINE=${ROUTINES%% *},BBL_STAGE=${CF_STAGE},R=${R},SEED=${SEED},BBL_STEP=warmup,${FWD_GPU_ENV} \
        "${CL_ROOT}/bbl_job.sh")
    cl_say "  bbl_warmup -> ${wj}"
    warm_dep="--dependency=afterok:${wj}"
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${wj}"
fi

# BBL Step 1, the default pre-step. It writes the ONE shared policy CSV, so every routine's
# fwd_sim waits afterok on it. 64G is what the pandas fit needs.
fwd_dep="${warm_dep}"
if [[ "${DO_POLFUNC}" == "1" ]]; then
    pj=$(sub "bbl_polfunc" "${SOLVE_TIME}" ${warm_dep} --mem=64G \
        --export=ALL,BBL_STAGE=${CF_STAGE},R=${R},SEED=${SEED},BBL_STEP=polfunc \
        "${CL_ROOT}/bbl_job.sh")
    cl_say "  bbl_polfunc -> ${pj}  (--mem=64G; every fwd_sim waits afterok on it)"
    fwd_dep="--dependency=afterok:${pj}"
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${pj}"
fi

RUN_EPOCH_NOW="$(date +%s)"
for k in ${ROUTINES}; do
    BBL_ROUTINE="${k}"; BBL_STAGE="${CF_STAGE}"
    BBL_KEY="$(cl_bbl_key "${k}" "${CF_STAGE}" "${PSI_TAG}")"
    BBL_RTAG="$(cl_bbl_rtag "${k}" "${CF_STAGE}" "${PSI_TAG}")"
    DD="$(cl_bbl_dispatch_dir "${BBL_KEY}")"
    base_export="BBL_ROUTINE=${k},BBL_STAGE=${CF_STAGE},R=${R},SEED=${SEED}"
    BBL_FWD_EXTRA="${bbl_extra}"; RUN_EPOCH="${RUN_EPOCH_NOW}"; BBL_FRESH=0; want=""

    # ── (0) the design: this invocation's, or (--repair) the original run's ─────────────
    if [[ "${REPAIR}" == "1" ]]; then
        if [[ -f "${DD}/context.env" ]]; then
            for _v in N_SHARDS SHOCKS HORIZON BETA; do
                _o="$(_ctx_get "${DD}/context.env" "${_v}")"
                if [[ "${_GIVEN}" == *" ${_v} "* && "${!_v}" != "${_o}" ]]; then
                    cl_err "REFUSING --repair E${k}: ${_v}=${!_v} here, but the original run used ${_v}=${_o}."
                    cl_err "  A repair re-runs THAT design (N_SHARDS is in every shard's name and split); omit ${_v} or pass ${_o}."
                    exit 2
                fi
                printf -v "${_v}" '%s' "${_o}"
            done
            BETA_SRC="context.env of the original run"; HORIZON_SRC="${BETA_SRC}"
            BBL_FWD_EXTRA="$(_ctx_get "${DD}/context.env" BBL_FWD_EXTRA)"
            MULTI_START="$(_ctx_get "${DD}/context.env" MULTI_START)"
            RUN_EPOCH="$(_ctx_get "${DD}/context.env" RUN_EPOCH)"
            cl_say "  --repair E${k}: design from the original context (N_SHARDS=${N_SHARDS} T=${HORIZON} beta=${BETA} shocks=${SHOCKS}); placement from these flags"
        else
            _ns="$(ls "${CL_STEP_BBL}" 2>/dev/null | sed -n "s/^psi_dev_${BBL_KEY}_shard[0-9]*of\([0-9]*\)\.parquet$/\1/p" | sort -u | paste -sd' ' - || true)"
            if [[ -z "${_ns}" ]]; then
                cl_err "REFUSING --repair E${k}: no context and no psi_dev_${BBL_KEY}_shard* on disk — nothing to repair; launch it instead."
                exit 2
            fi
            if [[ "${_ns}" == *" "* ]]; then
                cl_err "REFUSING --repair E${k}: shard files of more than one count on disk (of ${_ns}); the solve refuses the mix."
                exit 2
            fi
            if [[ "${_ns}" != "${N_SHARDS}" ]]; then
                if [[ "${_GIVEN}" == *" N_SHARDS "* ]]; then
                    cl_err "REFUSING --repair E${k}: --shards ${N_SHARDS}, but the files on disk are of${_ns}."; exit 2
                fi
                N_SHARDS="${_ns}"
            fi
            cl_say "  --repair E${k}: no context (launched before the sweep existed): N_SHARDS=${N_SHARDS} from the files,"
            cl_say "     every other setting from THESE flags and bbl_discount.env — they must match the original"
            cl_say "     (the sweep checks beta/T against psi_starts)"
        fi
        cl_bbl_check_curve "${HORIZON}" "${MULTI_START}" || exit 1
    elif [[ "${PROBE}" != "1" ]]; then
        want="$(cl_expand_spec "${SHARD_LIST:-0-$(( N_SHARDS - 1 ))}")" \
            || { cl_err "REFUSING: --shard-list '${SHARD_LIST}' is not an index list"; exit 2; }
        for _s in ${want}; do
            (( _s < N_SHARDS )) || { cl_err "REFUSING: shard ${_s} is outside 0..$(( N_SHARDS - 1 ))"; exit 2; }
        done
        # A fresh FULL launch owns every index: a file older than it is a leftover, and the sweep
        # re-runs it instead of trusting it. A partial --shard-list launch trusts the rest.
        if [[ -z "${SHARD_LIST}" ]]; then BBL_FRESH=1; fi
    fi
    cl_say "-- E${k} ${CF_STAGE} | ${BBL_KEY} | ${SHOCKS} shocks over ${N_SHARDS} shards | T=${HORIZON} beta=${BETA}"

    # ── (1) no duplicate submission: nothing of this routine/tag may be alive on the same shards
    if [[ "${FORCE}" != "1" ]]; then
        _busy="$(cl_bbl_live_named "bbl_sweep_${BBL_RTAG},bbl_solve_${BBL_RTAG}")"
        if [[ -n "${_busy}" ]]; then
            cl_err "REFUSING E${k}: a sweep or solve of ${BBL_KEY} is still queued or running (job ${_busy})."
            cl_err "  That chain is alive: watch it (bash bbl_status.sh) or cancel it first"
            cl_err "  (bash bbl_cancel.sh --routines ${k} --psi-tag '${PSI_TAG}'). --force overrides."
            exit 1
        fi
        _claims="$(cl_bbl_live_claims "${BBL_KEY}" "${BBL_RTAG}")"
        if [[ -n "${_claims}" ]]; then
            if [[ "${REPAIR}" == "1" || "${PROBE}" == "1" && -z "${want}" ]]; then
                _ov="$(awk '{print $1 ":all"}' <<< "${_claims}" | paste -sd' ' -)"
            else
                _ov="$(while read -r _j _rest; do
                          if [[ "${_rest}" == "UNKNOWN" ]]; then echo "${_j}:all(unclaimed)"; continue; fi
                          _hit="$(comm -12 <(printf '%s\n' ${_rest} | sort -u) <(printf '%s\n' ${want} | sort -u) | sort -n | paste -sd' ' -)"
                          if [[ -n "${_hit}" ]]; then echo "${_j}:$(cl_compress_ranges ${_hit})"; fi
                      done <<< "${_claims}" | paste -sd' ' -)"
            fi
            if [[ -n "${_ov}" ]]; then
                cl_err "REFUSING E${k}: live fwd job(s) of ${BBL_KEY} already claim these shards: ${_ov}"
                cl_err "  Two jobs writing one shard is how overlapping arrays used to collide. Wait, cancel"
                cl_err "  (bash bbl_cancel.sh --routines ${k} --psi-tag '${PSI_TAG}'), or --force."
                exit 1
            fi
        fi
    fi

    # ── (2) files a launch would collide with (a second shard-count family) ─────────────
    if [[ "${REPAIR}" != "1" ]]; then
        _others="$(ls "${CL_STEP_BBL}" 2>/dev/null | sed -n "s/^psi_dev_${BBL_KEY}_shard[0-9]*of\([0-9]*\)\.parquet$/\1/p" \
                   | sort -u | grep -vx "${N_SHARDS}" | paste -sd, - || true)"
        if [[ -n "${_others}" && "${FORCE}" != "1" ]]; then
            cl_err "REFUSING E${k}: psi_dev_${BBL_KEY}_shard*of{${_others}} is on disk beside the of${N_SHARDS} family this run"
            cl_err "  would write. The solve refuses mixed shard counts, so the run would fail after the whole array."
            cl_err "  Move those files aside, pick a new --psi-tag, or --force."
            exit 1
        fi
        _nsame="$(ls "${CL_STEP_BBL}" 2>/dev/null | grep -c "^psi_dev_${BBL_KEY}_shard[0-9]*of${N_SHARDS}\.parquet$" || true)"
        if [[ "${BBL_FRESH}" == "1" ]] && (( ${_nsame:-0} > 0 )); then
            cl_say "   note: ${_nsame} psi_dev file(s) of ${BBL_KEY} from an earlier run are on disk. Each is replaced when"
            cl_say "         its shard finishes, and the sweep trusts only files newer than this launch."
        fi
    fi

    # ── (3) the run's context: what every later re-run of this routine is built from ──
    cl_bbl_ctx_write "${BBL_KEY}" || { cl_err "cannot write ${DD}/context.env"; exit 1; }
    if [[ "${CL_DRYRUN}" != "1" ]]; then rm -f "${DD}/STOP" "${DD}/solve.jid"; fi
    cl_bbl_event "${BBL_KEY}" "LAUNCH mode=${_mode} n=${N_SHARDS} T=${HORIZON} beta=${BETA} shards=$(cl_compress_ranges ${want})"

    # ── (4) the fwd jobs ─────────────────────────────────────────────────────────────
    FWD_JIDS=""
    if [[ "${PROBE}" == "1" ]]; then
        _plist=( $(cl_expand_spec "${SHARD_LIST:-0-${PROBE_PACK}}") )
        _need=$(( PROBE_PACK + PROBE_CONTROL ))
        (( ${#_plist[@]} >= _need )) || { cl_err "REFUSING --probe: it needs ${_need} shard ids (got ${#_plist[@]})"; exit 2; }
        for _s in "${_plist[@]:0:_need}"; do
            (( _s < N_SHARDS )) || { cl_err "REFUSING --probe: shard ${_s} is outside 0..$(( N_SHARDS - 1 ))"; exit 2; }
        done
        BBL_MULT_LABEL="probe"
        cl_say "  probe (a) PACKED: shards $(cl_compress_ranges "${_plist[@]:0:PROBE_PACK}") in ONE ${PROBE_PARTITION} job, ${PROBE_PACK} x ${PROBE_MEM_PACKED}"
        cl_bbl_dispatch_fwd "${_plist[*]:0:PROBE_PACK}" "${BBL_PROBE_WALL_MULT}" "" || exit 1
        PROBE_PACKED_JID="${CL_BBL_JIDS}"; PROBE_CONTROL_JID=""; FWD_JIDS="${PROBE_PACKED_JID}"
        if [[ "${PROBE_CONTROL}" == "1" ]]; then
            printf -v "PACK_${_PK}" '%s' 1; printf -v "MEM_${_PK}" '%s' "${PROBE_MEM_UNPACKED}"
            cl_say "  probe (b) UNPACKED: shard ${_plist[PROBE_PACK]} alone, 1 x ${PROBE_MEM_UNPACKED}"
            cl_bbl_dispatch_fwd "${_plist[PROBE_PACK]}" "${BBL_PROBE_WALL_MULT}" "" || exit 1
            PROBE_CONTROL_JID="${CL_BBL_JIDS}"; FWD_JIDS="${FWD_JIDS}:${PROBE_CONTROL_JID}"
            printf -v "PACK_${_PK}" '%s' "${PROBE_PACK}"; printf -v "MEM_${_PK}" '%s' "${PROBE_MEM_PACKED}"
        fi
        BBL_MULT_LABEL="retry"
        # What bbl_sizing.sh reads: which job is which, and what each was given.
        if [[ "${CL_DRYRUN}" != "1" ]]; then
            printf 'PACKED_JID=%s\nPACKED_SHARDS=%s\nPACKED_MEM=%s\nPACKED_K=%s\nCONTROL_JID=%s\nCONTROL_SHARD=%s\nCONTROL_MEM=%s\nPARTITION=%s\nT=%s\nBETA=%s\nN_SHARDS=%s\nRTAG=%s\n' \
                "${PROBE_PACKED_JID}" "$(cl_compress_ranges "${_plist[@]:0:PROBE_PACK}")" "${PROBE_MEM_PACKED}" "${PROBE_PACK}" \
                "${PROBE_CONTROL_JID}" "$([[ "${PROBE_CONTROL}" == "1" ]] && printf '%s' "${_plist[PROBE_PACK]}" || true)" \
                "${PROBE_MEM_UNPACKED}" "${PROBE_PARTITION}" "${HORIZON}" "${BETA}" "${N_SHARDS}" "${BBL_RTAG}" \
                > "${DD}/probe.env"
        else
            cl_log "  [dry-run] would write ${DD}/probe.env (PACKED_JID=${PROBE_PACKED_JID} CONTROL_JID=${PROBE_CONTROL_JID})"
        fi
        cl_bbl_event "${BBL_KEY}" "PROBE packed=${PROBE_PACKED_JID} control=${PROBE_CONTROL_JID:-none} T=${HORIZON}"
    elif [[ "${REPAIR}" != "1" ]]; then
        cl_bbl_dispatch_fwd "${want}" 1 "${fwd_dep}" || exit 1
        FWD_JIDS="${CL_BBL_JIDS}"
    fi

    # ── (5) the sweep, HELD until the solve it re-targets is registered ─────────────────
    # Without the hold, a sweep whose fwd jobs died in seconds could run before solve.jid exists,
    # find no solve to move, and exit 0 — releasing the solve onto a partial set.
    sweep_jid=""
    if [[ "${DO_SWEEP}" == "1" ]]; then
        _sd=""
        if [[ -n "${FWD_JIDS}" ]]; then _sd="--dependency=afterany:${FWD_JIDS}"; fi
        sweep_jid="$(cl_bbl_submit_sweep 0 --hold ${_sd})"
        cl_require_jid "${sweep_jid}" "bbl_sweep_${BBL_RTAG}" || exit 1
        cl_say "  bbl_sweep_${BBL_RTAG} attempt 0/${BBL_MAX_RETRIES} ${_sd:+afterany ${FWD_JIDS//:/, } }-> ${sweep_jid}"
    fi

    # ── (6) the solve: afterok the FINAL sweep (each re-run moves this dependency forward) ──
    slv=""
    if [[ "${DO_SOLVE}" == "1" ]]; then
        if [[ -n "${sweep_jid}" ]]; then _dsol="afterok:${sweep_jid}"; else _dsol="afterany:${FWD_JIDS}"; fi
        slv=$(sub "bbl_solve_${BBL_RTAG}" "${SOLVE_TIME}" --dependency="${_dsol}" \
            --export=ALL,${base_export},BBL_STEP=solve,BBL_EXTRA="${solve_extra}",PSI_TAG="${PSI_TAG}" \
            "${CL_ROOT}/bbl_job.sh")
        cl_require_jid "${slv}" "bbl_solve_${BBL_RTAG}" || exit 1
        if [[ "${CL_DRYRUN}" != "1" ]]; then printf '%s\n' "${slv}" > "${DD}/solve.jid"; fi
        cl_bbl_event "${BBL_KEY}" "SUBMIT solve jid=${slv} dep=${_dsol}"
        cl_say "  bbl_solve_${BBL_RTAG} ${_dsol//:/ } -> ${slv}$([[ -n "${sweep_jid}" ]] && echo '  (afterok the FINAL sweep)' || true)"
        solve_dep="${solve_dep}:${slv}"
    fi
    if [[ -n "${sweep_jid}" ]]; then
        cl_bbl_event "${BBL_KEY}" "SUBMIT sweep jid=${sweep_jid} retry=0"
        if [[ "${CL_DRYRUN}" == "1" ]]; then
            cl_log "  [dry-run] scontrol release ${sweep_jid}"
        elif ! scontrol release "${sweep_jid}"; then
            cl_err "  could not release sweep ${sweep_jid}: it stays HELD. Release it by hand:  scontrol release ${sweep_jid}"
        fi
    fi
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}$(echo "${FWD_JIDS}:${sweep_jid}:${slv}" | sed 's/::*/:/g; s/^://; s/:$//')"
done

dep_csv="$(echo ${solve_dep} | sed 's/^://')"

# ── Tables, afterOK EVERY solve. THE COLLECTION POINT OF THE WHOLE PHASE.
#    Nothing comes back from this cluster but the archive, so this step turns the solves into
#    the .tex/.md tables AND echoes them into its own SLURM log. afterok, not afterany: a
#    routine whose solve did not finish must not be rendered from whatever else is on disk.
zip_dep="${dep_csv}"
if [[ "${DO_TABLES}" == "1" && -n "${dep_csv}" ]]; then
    tab_jid=$(sub "bbl_tables${PSI_TAG}" "00:30:00" --dependency=afterok:${dep_csv} --mem=16G \
        --export=ALL,BBL_STAGE=${CF_STAGE},BBL_STEP=tables,PSI_TAG="${PSI_TAG}" \
        "${CL_ROOT}/bbl_job.sh")
    cl_say "  bbl_tables${PSI_TAG} afterok ${dep_csv//:/, } -> ${tab_jid}  (tables are ECHOED to its log)"
    cl_log "   -> read them with: cat ${LOGD}/bbl_tables${PSI_TAG}_${tab_jid}_*.out"
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${tab_jid}"
    zip_dep="${dep_csv}:${tab_jid}"
fi

# ── The archive, afterOK every solve and the tables. COPY is MANDATORY: cf1_net/cf3/cf5/cf6
#    read cost_params_*.json IN PLACE. SLIM by default: this tag's psi, cost_params, tables and
#    polfunc outputs, without the untagged single-curve psi, bench/probe files and dispatch state
#    that inflated the 2026-09-22 archive to ~1 GB. --zip-full packages the whole folder.
if [[ "${DO_ZIP}" == "1" && -n "${dep_csv}" ]]; then
    _slim=""
    if [[ "${ZIP_SLIM}" == "1" ]]; then _slim=" --slim --psi-tag '${PSI_TAG}'"; fi
    ZWRAP="cd '${CL_ROOT}' && bash cluster_archive.sh --set bbl --copy --tag \"\${SLURM_JOB_ID}\"${_slim}"
    zip_jid=$(cl_sbatch --dependency=afterok:${zip_dep} \
        -J "bbl_zip${PSI_TAG}" --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGD}/bbl_zip${PSI_TAG}_%j.out" -e "${LOGD}/bbl_zip${PSI_TAG}_%j.err" \
        --wrap "${ZWRAP}")
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${zip_jid}"
    cl_say "  bbl_zip${PSI_TAG} afterok ${zip_dep//:/, } -> ${zip_jid}  ($([[ "${ZIP_SLIM}" == "1" ]] && echo "slim, tag '${PSI_TAG}'" || echo 'whole bbl folder'), --copy)"
    cl_log "   -> ${CL_STEP_DOWNLOAD}/bbl_outputs_<jobid>.zip  (cost_params COPIED — originals stay for the CFs)"
fi

if [[ "${PROBE}" == "1" ]]; then
    cl_say "Probe submitted (tag '${PSI_TAG}'). When both jobs have left squeue:  bash bbl_sizing.sh"
else
    cl_say "Submitted BBL (${_mode}) for routines: ${ROUTINES}. Watch with:  bash bbl_status.sh --psi-tag '${PSI_TAG}'"
fi
# Machine-parseable handles for pipeline_all.sh. Keep BBL_ALL_JOBIDS the LAST line
# so `sed -n 's/^BBL_ALL_JOBIDS=//p' | tail -1` captures it cleanly.
echo "BBL_SOLVE_JOBIDS=${dep_csv}"
echo "BBL_ALL_JOBIDS=${ALL_JIDS}"
