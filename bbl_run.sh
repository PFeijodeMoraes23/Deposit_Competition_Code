#!/bin/bash
# ==============================================================================
# bbl_run.sh — THE single front door for the BBL cost-estimation stage.
#
# WHAT MUST EXIST FIRST
#   1. bash cluster_preflight.sh                (must print PREFLIGHT OK)
#   2. the RC results: data/output/blp_outputs_*.zip (step 1 auto-builds
#      cluster_processed/ from the newest one), or cluster_processed/ already there
#   3. TWO uploaded inputs, both built LOCALLY:
#        data/input/forward_rf_qoq.csv     (cf_forward_rf.py — needs internet)
#        data/input/polfunc_fitted.csv     (estimation_bbl_1_polfunc.py)
# WHAT TO RUN NEXT
#   bash cf_run.sh --do "demand_eval cf1 cf1_net cf4"
#   (or run both in one shot with pipeline_run.sh)
#
# THIS SCRIPT IS NEW. submit_bbl_all.sh / submit_bbl_default.sh are still present
# and still work; they are the fallback and are retired only after one successful
# cluster cycle. Nothing in them has been modified.
#
# WHAT IT SUBMITS
#   warmup --afterok--> [polfunc] --afterok--> fwd_sim ARRAY --afterANY--> solve
#                                                                    \
#                                                          afterany --> archive (COPY)
#   The solve is afterANY the array deliberately: it globs whatever psi_dev shards
#   exist, so one flaky shard does not block the routine's costs. (If shard 0
#   failed there is no psi_eq, the solve errors, and --kill-on-invalid-dep cancels
#   its afterok downstream.)
#
# Usage
#   bash bbl_run.sh --dry-run
#   bash bbl_run.sh
#   bash bbl_run.sh --routines "3" --shards 50
#   bash bbl_run.sh --fwd-cpu
#
# Flags
#   --routines "3 4"   routine set (default from cluster_lib.sh)
#   --polfunc          run BBL Step 1 on the cluster as a pre-step
#   --no-warmup        skip the pre-warm barrier
#   --fwd-cpu|--fwd-gpu  where the fwd_sim array runs (default: the code default, GPU)
#   --shards N         fwd_sim shard count (default 100)
#   --array-spec S     re-run a SUBSET of shards, e.g. 96-99 or 3,17,88
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
AUTO_PROCESS="${AUTO_PROCESS:-1}"
DO_POLFUNC="${DO_POLFUNC:-0}"
DO_WARMUP="${DO_WARMUP:-1}"
DO_ZIP=1; SKIP_PREFLIGHT=0
# FWD_GPU=1 is the CODE DEFAULT and is NOT silently changed here — see the
# advisory printed below.
FWD_GPU="${FWD_GPU:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --routines)       ROUTINES="$2"; ROUTINES_SRC=flag; shift ;;
        --polfunc)        DO_POLFUNC=1 ;;
        --no-warmup)      DO_WARMUP=0 ;;
        --fwd-cpu)        FWD_GPU=0 ;;
        --fwd-gpu)        FWD_GPU=1 ;;
        --shards)         N_SHARDS="$2"; shift ;;
        --array-spec)     ARRAY_SPEC="$2"; shift ;;
        --no-zip)         DO_ZIP=0 ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --dry-run)        CL_DRYRUN=1 ;;
        -h|--help)        sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

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
    # Each GPU shard holds one H200; default to ~one node's worth of concurrent
    # shards so the array does not demand N_SHARDS scarce GPUs at once.
    [[ -z "${ARRAY_THROTTLE+set}" ]] && ARRAY_THROTTLE=8   # default only when truly UNSET (empty => unlimited)
    echo "fwd_sim -> GPU (${GPU_PARTITION}, --gpus=${GPUS}, throttle=${ARRAY_THROTTLE:-none}); --fwd-cpu forces CPU"
    echo "[advisory] FWD_GPU=1 is the code default, and this run keeps it. A memory note records"
    echo "           fwd_sim as memory-bandwidth bound at ~0% GPU utilisation, i.e. it arguably"
    echo "           belongs on CPU (--fwd-cpu: day partition, --constraint=cpugen:turin)."
    echo "           Stated, not silently changed."
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
    echo "fwd_sim -> CPU (${CPU_PARTITION:-day}, ${CPU_CONSTRAINT:-cpugen:turin}, no GPU, throttle=${ARRAY_THROTTLE:-none})"
fi
THROTTLE="${ARRAY_THROTTLE:+%${ARRAY_THROTTLE}}"

# ── Step 1: build cluster_processed/ from the newest RC zip if a routine is missing
CP_DIR="$(cl_cp_dir ${ROUTINES})"
need_process=0
for k in ${ROUTINES}; do [[ -f "${CP_DIR}/blp_E${k}_spec_12.jls" ]] || need_process=1; done
if [[ "${need_process}" == "1" && "${AUTO_PROCESS}" == "1" ]]; then
    cl_build_cp_dir "${CP_DIR}" "${CF_STAGE}" ${ROUTINES}
fi

# ── Step 2: preflight — every required input, INCLUDING the login-node Python probe
miss=0
cl_need_draws "${R}" "${SEED}" || miss=1
cl_need_rf_curve || miss=1
POLICY_CSV="${POLICY_CSV:-${CL_DATA_IN}/polfunc_fitted.csv}"
# The fitted policy is REQUIRED unless we build it here: deviations formed around
# OBSERVED spreads instead of the fitted policy make frac_bind ~ 0.5 mechanical
# and leave (omega, zeta, gamma) uninformative (V_Main line 551, sec 0A/9).
if [[ "${DO_POLFUNC}" != "1" ]]; then
    cl_need_file "${POLICY_CSV}" "fitted policy" \
        "build locally then upload: python estimation_bbl_1_polfunc.py" \
        "(or pass --polfunc to run BBL Step 1 on the cluster as a pre-step)" || miss=1
fi
for k in ${ROUTINES}; do cl_need_rc_jls "${CP_DIR}" "${k}" || miss=1; done
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
echo "Preflight OK: draws + forward r^f curve + fitted policy + RC results + solve Python env for routines: ${ROUTINES}"

# ── Step 3: fwd_sim flags ────────────────────────────────────────────────────
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
echo "fwd_sim sigma-hat <- FITTED policy: ${POLICY_CSV}"
bbl_extra="--shocks ${SHOCKS} --perturb-scale ${PERTURB_SCALE} --dev-scheme ${DEV_SCHEME} --beta ${BETA} --horizon ${HORIZON}${asset_flags:+ ${asset_flags}} ${policy_flag}"

sub () {  # sub <jobname> <time> <extra sbatch args...>
    local name="$1" tlim="$2"; shift 2
    cl_sbatch -J "${name}" -t "${tlim}" \
        ${PARTITION:+--partition="${PARTITION}"} ${MEM:+--mem="${MEM}"} \
        -o "${LOGD}/${name}_%A_%a.out" -e "${LOGD}/${name}_%A_%a.err" "$@"
}

MARKER="$(cl_run_marker bbl)"
ALL_JIDS=""     # EVERY job id this run creates (declared BEFORE the first append)
solve_dep=""    # colon-joined solve job ids -> the archive waits on all of them

# ── Pre-warm barrier: precompile/load ONCE so the fwd_sim array does not stampede
#    the shared-NFS precompile/load lock (the cause of tasks sitting at 0% CPU).
warm_dep=""
if [[ "${DO_WARMUP}" == "1" ]]; then
    wj=$(sub "bbl_warmup" "${SOLVE_TIME}" "${FWD_SB[@]}" \
        --export=ALL,BBL_ROUTINE=${ROUTINES%% *},BBL_STAGE=${CF_STAGE},R=${R},SEED=${SEED},BBL_STEP=warmup,${FWD_GPU_ENV} \
        "${CL_ROOT}/bbl_job.sh")
    echo "-- pre-warm -> job ${wj} (fwd_sim waits on it)"
    warm_dep="--dependency=afterok:${wj}"
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${wj}"
fi

# Optional BBL Step 1 on the cluster. It writes the SHARED policy CSV, so every
# routine's fwd_sim waits on it.
fwd_dep="${warm_dep}"
if [[ "${DO_POLFUNC}" == "1" ]]; then
    pj=$(sub "bbl_polfunc" "${SOLVE_TIME}" ${warm_dep} \
        --export=ALL,BBL_STAGE=${CF_STAGE},R=${R},SEED=${SEED},BBL_STEP=polfunc \
        "${CL_ROOT}/bbl_job.sh")
    echo "-- BBL Step 1 (polfunc) -> job ${pj} (fwd_sim waits on it)"
    fwd_dep="--dependency=afterok:${pj}"
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${pj}"
fi

for k in ${ROUTINES}; do
    base_export="BBL_ROUTINE=${k},BBL_STAGE=${CF_STAGE},R=${R},SEED=${SEED}"
    echo "-- E${k} ${CF_STAGE} | R=${R} | shocks=${SHOCKS} over ${N_SHARDS} shards --"
    # --array-spec re-runs a SUBSET of shards without redoing the ones already on
    # disk: psi_dev_*_shard{i}of{N}.parquet files are independent and the solve
    # globs whatever exists (a CPU-computed shard is numerically identical to a
    # GPU one). N_SHARDS must stay the SAME as the original run — it is baked into
    # the filename and the (firm x delta) split. psi_eq is written by shard 0 only.
    arr=$(sub "bbl_fwd_E${k}" "${SHARD_TIME}" ${fwd_dep} "${FWD_SB[@]}" \
        --array="${ARRAY_SPEC:-0-$((N_SHARDS-1))}""${THROTTLE}" \
        --export=ALL,${base_export},BBL_STEP=fwd_sim,N_SHARDS=${N_SHARDS},BBL_EXTRA="${bbl_extra}",${FWD_GPU_ENV} \
        "${CL_ROOT}/bbl_job.sh")
    echo "  fwd_sim array -> job ${arr} (${N_SHARDS} shards${THROTTLE:+, throttled ${THROTTLE}})"
    slv=$(sub "bbl_solve_E${k}" "${SOLVE_TIME}" --dependency=afterany:"${arr}" \
        --export=ALL,${base_export},BBL_STEP=solve,BBL_EXTRA="--bootstrap 200" \
        "${CL_ROOT}/bbl_job.sh")
    echo "  solve         -> job ${slv} (afterANY:${arr} — it globs the surviving psi_dev shards)"
    solve_dep="${solve_dep}:${slv}"
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${arr}:${slv}"
done

# ── Auto-archive, afterANY all solves. COPY is MANDATORY here: cf1_net/cf3/cf5/cf6
#    read cost_params_*.json IN PLACE, so moving them would strand the CF inputs.
dep_csv="$(echo ${solve_dep} | sed 's/^://')"
if [[ "${DO_ZIP}" == "1" && -n "${dep_csv}" ]]; then
    ZWRAP="cd '${CL_ROOT}'"
    for s in bbl_costs bbl_logs; do
        ZWRAP="${ZWRAP}; bash cluster_archive.sh --set ${s} --copy --newer '${MARKER}' --tag \"\${SLURM_JOB_ID}\""
    done
    ZWRAP="${ZWRAP}; rm -f '${MARKER}'"
    zip_jid=$(cl_sbatch --dependency=afterany:${dep_csv} \
        -J bbl_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGD}/bbl_zip_%j.out" -e "${LOGD}/bbl_zip_%j.err" \
        --wrap "${ZWRAP}")
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${zip_jid}"
    echo "-- archive: ${zip_jid}  (afterany${solve_dep})"
    echo "   -> ${CL_DATA_OUT}/bbl_outputs_<${zip_jid}>.zip  (cost_params COPIED — originals stay for the CFs)"
else
    [[ "${CL_DRYRUN}" == "1" ]] || rm -f "${MARKER}"
fi

echo "Submitted BBL cost estimation for routines: ${ROUTINES}. Watch with: squeue -u \$USER"
echo "When the solve completes the CFs can run:  bash cf_run.sh"
# Machine-parseable handles for pipeline_run.sh. Keep BBL_ALL_JOBIDS the LAST line
# so `sed -n 's/^BBL_ALL_JOBIDS=//p' | tail -1` captures it cleanly.
echo "BBL_SOLVE_JOBIDS=${dep_csv}"
echo "BBL_ALL_JOBIDS=${ALL_JIDS}"
