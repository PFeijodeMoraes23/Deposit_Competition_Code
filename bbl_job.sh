#!/bin/bash
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# (job-name + .out/.err are set per submission by bbl_run.sh via sbatch -J/-o/-e)
# ==============================================================================
# bbl_job.sh — THE generic BBL worker. This is the marginal-cost ESTIMATION
# stage: every artifact of it — polfunc_fitted.csv, the psi_eq/psi_dev shards and
# cost_params_*.json, which the counterfactuals CONSUME — lands in ONE folder,
# data/output/bbl. of_root.jl maps both COST_FWD and COST_POLFUNC there and
# cl_export_step_dirs points the Python halves (COST_POLFUNC_DIR, CF_COST_FWD) at
# the same place, so the fit, the shards and the solve share one directory.
#
# The #SBATCH block is a FLOOR (the panel iterates ~700k demand rows, so it is
# sized above the CF driver). bbl_run.sh overrides --partition/--gpus/--mem/-t
# per step on the sbatch command line.
#
# Driven by env vars:
#   BBL_STEP     warmup | polfunc | fwd_sim | solve
#   BBL_ROUTINE  3 (headline) | 4 | ...        -> --estim
#   BBL_STAGE    extended (headline) | full | ...   -> --stage
#   R, SEED      draws (default 2000 / 42) — must match the staged BLP_DRAWS
#   BBL_EXTRA    extra CLI flags passed through
#   N_SHARDS     fwd_sim only: total deviation shards (default 1)
#   SHARD_ID     fwd_sim only: this shard (default = SLURM_ARRAY_TASK_ID, else 0)
#   CF_GPU       1 -> GPU path (gpu sysimage); 0 -> CPU path (cpu sysimage)
#
# WHAT MUST EXIST FIRST: cluster_preflight.sh green; both sysimages built; the RC
#   results in data/output/blp.
# WHAT TO RUN NEXT: nothing directly — bbl_run.sh owns the chain.
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

: "${BBL_STEP:?set BBL_STEP (warmup|polfunc|fwd_sim|solve)}"
case "${BBL_STEP}" in warmup|polfunc|fwd_sim|solve) ;;
    *) echo "Unknown BBL_STEP='${BBL_STEP}' (warmup|polfunc|fwd_sim|solve)" >&2; exit 2 ;;
esac
BBL_ROUTINE="${BBL_ROUTINE:-3}"
BBL_STAGE="${BBL_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
BBL_EXTRA="${BBL_EXTRA:-}"

mkdir -p "${CL_ROOT}/logs"

# The skeleton, then the step dirs. The polfunc branch below is PYTHON and resolves
# its output through utils/paths.polfunc_dir(), which reads COST_POLFUNC_DIR — set
# here and nowhere else, so a job that skipped this call would write the policy CSV
# into the local-layout default, exit 0, and strand every fwd_sim task behind it.
cl_bootstrap_tree
cl_export_step_dirs

cl_load_julia

# CF_GPU is set by bbl_run.sh (FWD_GPU=1 -> warmup + fwd_sim on the H200;
# solve/polfunc stay CPU). Default OFF here so a bare `BBL_STEP=... sbatch
# bbl_job.sh` stays CPU and concurrent array tasks on the shared NFS depot do not
# stampede the precompile/load lock loading CUDA.
: "${CF_GPU:=0}"; export CF_GPU

# ── Sysimage: the GPU/CPU pairing, as a REFUSAL. ─────────────────────────────
# A sysimage bakes its BUILD node's CPU target, so the GPU (gpu_h200 =
# sapphirerapids) and CPU (`day`, cpugen:turin) runs need DIFFERENT images.
# Loading the wrong one fails with "Unable to find compatible target in cached
# code image" and drops the task back to precompiling against the shared NFS
# depot; a task that loses that race gets CUDA.functional()==false, which
# cf_demand_eval.jl catches and silently downgrades to the CPU share path
# — the job then holds its H200 at 0% util. (Diagnosed 2026-07-31: 5/11 fwd_sim
# tasks fell back to CPU on FillArrays/StaticArrays cache races.)
if [[ "${CF_GPU}" != "0" ]]; then cl_require_sysimage gpu; else cl_require_sysimage cpu; fi

# Fatal CUDA gate, before the real work, only when the job actually holds a GPU.
if [[ "${CF_GPU}" != "0" && -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]]; then
    cl_gpu_gate
fi

cl_banner "BBL cost stage | step=${BBL_STEP} | E${BBL_ROUTINE} | stage=${BBL_STAGE}" \
          "R=${R} seed=${SEED} threads=${SLURM_CPUS_PER_TASK:-8} CF_GPU=${CF_GPU}" \
          "node=$(hostname) | $(date)"

run_julia () {
    local script="$1"; shift
    julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} \
        --threads="${SLURM_CPUS_PER_TASK:-8}" \
        "${CL_ROOT}/${script}" \
        --estim "${BBL_ROUTINE}" --spec 12 --stage "${BBL_STAGE}" \
        --R "${R}" --seed "${SEED}" --hpc ${BBL_EXTRA} "$@"
}

case "${BBL_STEP}" in

    warmup)
        # JIT-load the BBL stack ONCE. With a sysimage the packages are already
        # baked, so the array never precompiles and there is no shared-NFS
        # stampede. Without one (ALLOW_SYSIMAGE_FALLBACK=1), precompile the depot
        # here so the parallel fwd_sim array at least does not all precompile at
        # the same moment.
        if [[ ${#CL_JULIA_SYS[@]} -gt 0 ]]; then
            echo "Sysimage present — skipping depot precompile."
        else
            echo "Precompiling depot..."
            julia --project="${CL_ROOT}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
        fi
        echo "Loading the BBL stack (CF_GPU=${CF_GPU})..."
        julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} \
            --threads="${SLURM_CPUS_PER_TASK:-8}" \
            -e "include(joinpath(\"${CL_ROOT}\", \"bbl_fwd_sim.jl\"))" || true
        echo "warmup complete: BBL stack loaded — the array can launch warm" ;;

    polfunc)
        # BBL Step 1, and the DEFAULT pre-step of every bbl_run.sh chain: fit the
        # parametric policy function. It needs only the market panel (uploaded as
        # market_panel.parquet), which is already there for the sleepiness stage,
        # so running it here costs one
        # short CPU job and removes a 59 MB upload that could go stale against the
        # panel. Writes ${COST_POLFUNC_DIR}/polfunc_fitted.csv — data/output/bbl —
        # which the fwd_sim tasks take through --policy-csv. The policy function is
        # SPEC-INVARIANT, so it takes no --spec (unlike fwd_sim/solve, whose
        # --spec 12 selects the BLP demand specification).
        cl_setup_python "${CL_PY_REQ_POLFUNC}"
        "${PYBIN}" "${CL_ROOT}/bbl_polfunc.py" ${BBL_EXTRA} ;;

    fwd_sim)
        # BBL Step 2 part 1: psi under sigma-hat and sigma-tilde deviations
        # (shardable). The forward r^f curve is a REQUIRED input — the sim errors
        # under --hpc if it is absent, and a flat r^f makes psi4 proportional to
        # psi2 and leaves zeta unidentified. It is fetched from the BCB APIs
        # locally (compute nodes have no internet) and uploaded to data/input.
        RF_CURVE="${RF_CURVE:-${CL_DATA_IN}/forward_rf_qoq.csv}"
        if [[ ! -f "${RF_CURVE}" ]]; then
            echo "ERROR: forward r^f curve missing: ${RF_CURVE}" >&2
            echo "  Generate locally (needs internet) and upload to data/input/:" >&2
            echo "    python scrape_forward_rf.py --horizon 50 --start 2026Q1" >&2
            echo "  Refusing to run: a flat r^f leaves zeta unidentified." >&2
            exit 1
        fi
        echo "forward r^f curve OK: ${RF_CURVE}"
        SHARD_ID="${SLURM_ARRAY_TASK_ID:-${SHARD_ID:-0}}"
        N_SHARDS="${N_SHARDS:-1}"
        run_julia bbl_fwd_sim.jl --n-shards "${N_SHARDS}" --shard-id "${SHARD_ID}" ;;

    solve)
        # BBL Step 2 part 2: the eq:17 minimisation -> data/output/bbl/cost_params_*.json.
        cl_setup_python "${CL_PY_REQ_SOLVE}"
        "${PYBIN}" "${CL_ROOT}/bbl_solve.py" \
            --estim "${BBL_ROUTINE}" --spec 12 --stage "${BBL_STAGE}" ${BBL_EXTRA} ;;

esac

echo "BBL step ${BBL_STEP} (E${BBL_ROUTINE} ${BBL_STAGE}) complete: $(date)"
