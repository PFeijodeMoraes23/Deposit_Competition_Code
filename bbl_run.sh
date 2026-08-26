#!/bin/bash
# ==============================================================================
# bbl_run.sh — THE single front door for the BBL cost-estimation stage.
#
# WHAT MUST EXIST FIRST
#   1. gate G0 green                            (env_job.sh ENV_STEP=preflight)
#   2. the RC results, in data/output/blp: blp_results_E{k}_spec_12_{stage}.jls,
#      exactly where the RC ladder wrote them and where _result_path opens them
#   3. ONE uploaded input:
#        data/input/forward_rf_qoq.csv     (cf_forward_rf.py — needs internet, so it
#                                           cannot be produced on a compute node)
#      The fitted policy is NOT an upload: this script's polfunc pre-step builds it.
# WHAT TO RUN NEXT
#   bash cf_run.sh --do "demand_eval cf1 cf1_net cf4"
#   (pipeline_all.sh runs both in one shot, with --cost-afterok wiring the solves)
#
# WHAT IT SUBMITS
#   warmup --afterok--> polfunc --afterok--> fwd_sim ARRAY --afterANY--> solve
#                                                                    \
#                                                          afterany --> archive (COPY)
#   The solve is afterANY the array deliberately: it globs whatever psi_dev shards
#   exist, so one flaky shard does not block the routine's costs. (If shard 0
#   failed there is no psi_eq, the solve errors, and --kill-on-invalid-dep cancels
#   its afterok downstream.)
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
#   bash bbl_run.sh --fwd-cpu
#
# Flags
#   --routines "3 4"   routine set (default from cluster_lib.sh)
#   --polfunc          accepted and a NO-OP — it names the default
#   --no-polfunc       skip the polfunc pre-step and use the CSV already on disk
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
DO_POLFUNC="${DO_POLFUNC:-1}"
DO_WARMUP="${DO_WARMUP:-1}"
DO_ZIP=1; SKIP_PREFLIGHT=0
# FWD_GPU=1 is the CODE DEFAULT and is NOT silently changed here — see the
# advisory printed below.
FWD_GPU="${FWD_GPU:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --routines)       ROUTINES="$2"; ROUTINES_SRC=flag; shift ;;
        --polfunc)        DO_POLFUNC=1 ;;          # names the default; kept so old command lines still parse
        --no-polfunc)     DO_POLFUNC=0 ;;
        --no-warmup)      DO_WARMUP=0 ;;
        --fwd-cpu)        FWD_GPU=0 ;;
        --fwd-gpu)        FWD_GPU=1 ;;
        --shards)         N_SHARDS="$2"; shift ;;
        --array-spec)     ARRAY_SPEC="$2"; shift ;;
        --no-zip)         DO_ZIP=0 ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --dry-run)        CL_DRYRUN=1 ;;
        -h|--help)        sed -n '2,49p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
    cl_log "fwd_sim -> GPU (${GPU_PARTITION}, --gpus=${GPUS}, throttle=${ARRAY_THROTTLE:-none}); --fwd-cpu forces CPU"
    cl_log "[advisory] FWD_GPU=1 is the code default, and this run keeps it. A memory note records"
    cl_log "           fwd_sim as memory-bandwidth bound at ~0% GPU utilisation, i.e. it arguably"
    cl_log "           belongs on CPU (--fwd-cpu: day partition, --constraint=cpugen:turin)."
    cl_log "           Stated, not silently changed."
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
bbl_extra="--shocks ${SHOCKS} --perturb-scale ${PERTURB_SCALE} --dev-scheme ${DEV_SCHEME} --beta ${BETA} --horizon ${HORIZON}${asset_flags:+ ${asset_flags}} ${policy_flag}"

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
        --export=ALL,${base_export},BBL_STEP=solve,BBL_EXTRA="--bootstrap 200" \
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
if [[ "${DO_ZIP}" == "1" && -n "${dep_csv}" ]]; then
    ZWRAP="cd '${CL_ROOT}' && bash cluster_archive.sh --set bbl --copy --tag \"\${SLURM_JOB_ID}\""
    zip_jid=$(cl_sbatch --dependency=afterany:${dep_csv} \
        -J bbl_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGD}/bbl_zip_%j.out" -e "${LOGD}/bbl_zip_%j.err" \
        --wrap "${ZWRAP}")
    ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}${zip_jid}"
    cl_say "  bbl_zip -> ${zip_jid}  (afterany ${dep_csv//:/, }; set: bbl, --copy)"
    cl_log "   -> ${CL_STEP_DOWNLOAD}/bbl_outputs_<tag>.zip  (cost_params COPIED — originals stay for the CFs)"
fi

cl_log "Submitted BBL cost estimation for routines: ${ROUTINES}. Watch with: squeue -u \$USER"
# Machine-parseable handles for pipeline_all.sh. Keep BBL_ALL_JOBIDS the LAST line
# so `sed -n 's/^BBL_ALL_JOBIDS=//p' | tail -1` captures it cleanly.
echo "BBL_SOLVE_JOBIDS=${dep_csv}"
echo "BBL_ALL_JOBIDS=${ALL_JIDS}"
