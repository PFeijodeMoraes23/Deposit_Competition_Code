#!/bin/bash
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# (job-name + .out/.err are set per submission by cf_run.sh / cf_eq_run.sh)
# ==============================================================================
# cf_job.sh — THE generic counterfactual worker. Dispatches on CF_STEP, all
# THIRTEEN branches:
#
#   warmup       precompile the depot + JIT-load the CF stack ONCE (the barrier
#                everything else waits on afterok)
#   demand_eval  0a: reproduce in-sample shares + export the shares_elas parquet
#   cf1          CF1 gross franchise-value sleepiness decomposition
#   cf4          CF4 Pix reallocation (descriptive: no costs, no equilibrium)
#   cf1_net      CF1 net-of-cost franchise value        (needs the BBL cost params)
#   cf3          CF3 equilibrium, single-process solve
#   cf3_init     firm-sharded Jacobi: write sigma0 = rho-hat
#   cf3_shard    firm-sharded Jacobi: one shard of sweep ${SWEEP}   (THE GPU step)
#   cf3_merge    firm-sharded Jacobi: merge sweep ${SWEEP}'s shards
#   cf5          CF5 monetary pass-through, single-process solve
#   cf6          CF6 merger simulation, single-process solve
#   cf5_compare  combine the base + Selic-shocked equilibria
#   cf6_compare  combine the base + merged equilibria
#
# Also: CF_STEP=zip shells out to cluster_archive.sh with an EXPLICIT mode.
#
# Env vars: CF_ROUTINE CF_STAGE R SEED CF_EXTRA CF_GPU SIGMA_DIR SWEEP
#           N_FIRM_SHARDS SIGMA_BASE SIGMA_SCN CF_WHICH CF_TAG CF_ZIP_MODE
#
# The #SBATCH block is a FLOOR sized for the CPU steps (the R=2000 'extended'
# context peaks ~60G, so keep --mem >= ~100G). cf_run.sh / cf_eq_run.sh override
# --partition/--gpus/--mem/-t on the sbatch command line; on gpu_h200 you MUST
# request a GPU or the QOS rejects the job ("QOSMinGRES").
#
# WHAT MUST EXIST FIRST: cluster_preflight.sh green; both sysimages built; the RC
#   results in data/output/blp; for the equilibrium steps, the BBL cost params in
#   data/output/bbl from bbl_run.sh.
# WHERE ITS ARTIFACTS GO: data/output/counterfactuals — cf_out_dir(out_dir) maps
#   every CF family there, and cf_4_pix.jl reads its upsilon/phi^noPix pair from the
#   same single folder the sleepiness upsilon step wrote them to.
# WHAT TO RUN NEXT: nothing directly — cf_run.sh / cf_eq_run.sh own the chains.
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

: "${CF_STEP:?set CF_STEP (warmup|demand_eval|cf1|cf4|cf1_net|cf3|cf3_init|cf3_shard|cf3_merge|cf5|cf6|cf5_compare|cf6_compare|zip)}"
case "${CF_STEP}" in
    warmup|demand_eval|cf1|cf4|cf1_net|cf3|cf3_init|cf3_shard|cf3_merge|cf5|cf6|cf5_compare|cf6_compare|zip) ;;
    *) echo "Unknown CF_STEP='${CF_STEP}' — see the header for all thirteen." >&2; exit 2 ;;
esac
CF_ROUTINE="${CF_ROUTINE:-3}"
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
CF_EXTRA="${CF_EXTRA:-}"

# ── The zip step needs no Julia at all; handle it before loading anything. ────
if [[ "${CF_STEP}" == "zip" ]]; then
    # CF_WHICH is a cluster_archive.sh SET NAME, not a CF family: every CF step
    # writes into cf_out_dir, so 'counterfactuals' is the set that holds all of them.
    : "${CF_WHICH:?set CF_WHICH to a cluster_archive.sh set (counterfactuals)}"
    # EXPLICIT mode, always. --move at the end of a finished equilibrium, --copy
    # mid-pipeline (cost_params feeds CF3/CF5/CF6 and CF3's sig_N feeds CF5 long
    # after the orchestrator exits).
    MODE="${CF_ZIP_MODE:-copy}"
    case "${MODE}" in copy|move) ;; *) echo "CF_ZIP_MODE must be copy|move (got '${MODE}')" >&2; exit 2 ;; esac
    exec bash "${CL_ROOT}/cluster_archive.sh" --set "${CF_WHICH}" "--${MODE}" ${CF_TAG:+--tag "${CF_TAG}"}
fi

mkdir -p "${CL_ROOT}/logs"

# The skeleton, then the step dirs. CF_FOUNDATION_DIR and CF_COST_FWD come from
# cl_export_step_dirs, so the Python half of the CF stack (cf_4_upsilon_export.py,
# the BBL solve) resolves the same folders the Julia half does.
cl_bootstrap_tree
cl_export_step_dirs

cl_load_julia
# IFT results/deltas are un-suffixed (the default), so no BLP_OUTPUT_SUFFIX /
# BLP_DELTA_SUFFIX override is needed. For numerical-engine results, export
# BLP_OUTPUT_SUFFIX=_num.

# GPU share kernel: load CUDA ONLY for the sharded best-response (cf3_shard — the
# GPU workhorse). Every other step is CPU-only; keeping CUDA out of them stops
# many concurrent jobs on the shared NFS depot from stampeding the Julia
# precompile/load lock for a package they never use.
case "${CF_STEP}" in cf3_shard) : "${CF_GPU:=1}" ;; *) : "${CF_GPU:=0}" ;; esac
export CF_GPU

# ── Sysimage handling — the asymmetry this closes ────────────────────────────
# The 40-wide cf3_shard GPU array otherwise precompiles CUDA cold against ONE
# shared NFS depot, and that race is precisely what produces
# CUDA.functional()==false and the silent CPU fall-back at
# foundation_demand_eval.jl:77. So the CF worker takes a sysimage like every
# other worker, and refuses rather than degrading.
if [[ "${CF_GPU}" != "0" ]]; then cl_require_sysimage gpu; else cl_require_sysimage cpu; fi
if [[ "${CF_GPU}" != "0" && -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]]; then
    cl_gpu_gate
fi

cl_banner "CF pipeline | step=${CF_STEP} | E${CF_ROUTINE} | stage=${CF_STAGE}" \
          "R=${R} seed=${SEED} threads=${SLURM_CPUS_PER_TASK:-8} CF_GPU=${CF_GPU}" \
          "node=$(hostname) | $(date)"

run_julia () {
    local script="$1"; shift
    julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} \
        --threads="${SLURM_CPUS_PER_TASK:-8}" \
        "${CL_ROOT}/${script}" \
        --estim "${CF_ROUTINE}" --spec 12 --stage "${CF_STAGE}" \
        --R "${R}" --seed "${SEED}" --hpc ${CF_EXTRA} "$@"
}

case "${CF_STEP}" in

    warmup)
        # Serially precompile the depot + JIT-load the CF stack ONCE, so the
        # parallel CFs do not stampede the shared-NFS depot precompile/load lock
        # (the cause of jobs sitting at 0% CPU for an hour). Everything else in
        # the run depends on this, afterok. This is THE ONLY step that
        # precompiles: running Pkg.precompile() again inside demand_eval is the
        # stampede the barrier exists to prevent.
        if [[ ${#CL_JULIA_SYS[@]} -gt 0 ]]; then
            echo "Sysimage present — skipping depot precompile."
        else
            echo "Precompiling depot..."
            julia --project="${CL_ROOT}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
        fi
        echo "Loading the CF stack (CF_GPU=${CF_GPU})..."
        julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} \
            --threads="${SLURM_CPUS_PER_TASK:-8}" \
            -e "include(joinpath(\"${CL_ROOT}\", \"foundation_demand_eval.jl\"))" || true
        echo "warmup complete: depot warm + CF stack loaded" ;;

    demand_eval)  run_julia foundation_demand_eval.jl ;;

    cf1)          run_julia cf_1_franchise_value.jl ;;

    cf4)          # Descriptive Pix reallocation. Consumes the exact link-aware
                  # phi^noPix parquet + Upsilon_pix JSON from cf_4_upsilon_export.py,
                  # written by the sleepiness upsilon step into
                  # data/output/counterfactuals and verified by gate G7. That is the
                  # only place cf_4_pix.jl looks, so the pair it reads is the pair
                  # this run produced. cf_run.sh preflights both.
        run_julia cf_4_pix.jl ;;

    cf1_net)      run_julia cf_1_franchise_value.jl --net ;;

    cf3)          run_julia cf_3_equilibrium_spreads.jl ;;

    cf3_init)     run_julia cf_3_equilibrium_spreads.jl --write-sigma0 \
                      --sigma-out "${SIGMA_DIR:?set SIGMA_DIR}/sig_0.parquet" ;;

    cf3_shard)    SID="${SLURM_ARRAY_TASK_ID:-${SHARD_ID:-0}}"
        run_julia cf_3_equilibrium_spreads.jl \
            --n-firm-shards "${N_FIRM_SHARDS:?set N_FIRM_SHARDS}" --firm-shard-id "${SID}" \
            --sigma-in  "${SIGMA_DIR:?set SIGMA_DIR}/sig_$((SWEEP-1)).parquet" \
            --sigma-out "${SIGMA_DIR}/sh_${SWEEP}_${SID}.parquet" ;;

    cf3_merge)    run_julia cf_3_equilibrium_spreads.jl --jacobi-merge \
            --sigma-in   "${SIGMA_DIR:?set SIGMA_DIR}/sig_$((SWEEP-1)).parquet" \
            --sigma-glob "${SIGMA_DIR}/sh_${SWEEP}_*.parquet" \
            --sigma-out  "${SIGMA_DIR}/sig_${SWEEP}.parquet" ;;

    cf5)          run_julia cf_5_passthrough.jl ;;
    cf6)          run_julia cf_6_merger.jl ;;

    cf5_compare)  run_julia cf_5_passthrough.jl --compare \
                      --sigma-base "${SIGMA_BASE:?}" --sigma-scn "${SIGMA_SCN:?}" ;;
    cf6_compare)  run_julia cf_6_merger.jl --compare \
                      --sigma-base "${SIGMA_BASE:?}" --sigma-scn "${SIGMA_SCN:?}" ;;

esac

echo "CF step ${CF_STEP} (E${CF_ROUTINE} ${CF_STAGE}) complete: $(date)"
