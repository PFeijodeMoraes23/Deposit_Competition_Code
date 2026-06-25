#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# (job-name + .out/.err set per submission via sbatch -J/-o/-e)
#
# Counterfactual pipeline driver (foundation 0a, CF1, CF2). Driven by env vars:
#   CF_STEP     demand_eval | cf1 | cost2 | cost_solve      (which stage to run)
#   CF_ROUTINE  6 (E6 headline) | 3 (E3 robustness)          → --estim
#   CF_STAGE    extended (headline) | full | …               → --stage
#   R, SEED     draws (default 2000 / 42) — must match the downloaded BLP_DRAWS
#   CF_EXTRA    extra CLI flags passed through (e.g. "--shocks 50 --beta 0.9")
#   N_SHARDS    cost2 only: total deviation shards (default 1)
#   SHARD_ID    cost2 only: this shard (default = SLURM_ARRAY_TASK_ID, else 0)
#
# To beat Bouchet time walls, submit cost2 as a JOB ARRAY (one shard per task) and
# the cost_solve as an afterok dependency — see submit_cf_all.sh (the orchestrator).
#
# ⚠ The CF Julia scripts are CPU-only (they include blp_1_estimation.jl, NOT the
# CUDA path), so this job does NOT use a GPU. We request gpu_h200 only because it
# is the node known to have ≥200G RAM (the R=2000 'extended' stage peaks ~60G).
# If your cluster has a high-mem CPU partition, switch --partition to it and you'll
# queue faster and not reserve an H200. CF2 GPU acceleration is a future option.

set -euo pipefail
: "${CF_STEP:?set CF_STEP (demand_eval|cf1|cost2|cost_solve)}"
CF_ROUTINE="${CF_ROUTINE:-6}"
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"
SEED="${SEED:-42}"
CF_EXTRA="${CF_EXTRA:-}"

# ── Environment (match the RC-BLP estimation so paths/suffixes line up) ─────────
module reset
module load Julia/1.11.4-linux-x86_64
export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/.julia_depot:${JULIA_DEPOT_PATH:-}"
# IFT results/deltas are un-suffixed (the default), so no BLP_OUTPUT_SUFFIX/BLP_DELTA_SUFFIX
# override is needed. For numerical-engine results, export BLP_OUTPUT_SUFFIX=_num.

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

echo "======================================"
echo " CF pipeline | step=${CF_STEP} | E${CF_ROUTINE} | stage=${CF_STAGE}"
echo " R=${R} seed=${SEED} threads=${SLURM_CPUS_PER_TASK} | $(date)"
echo " node=$(hostname)"
echo "======================================"

run_julia () {
    local script="$1"; shift
    julia --project="${PROJECT_DIR}" --threads="${SLURM_CPUS_PER_TASK}" \
        "${PROJECT_DIR}/${script}" \
        --estim "${CF_ROUTINE}" --spec 12 --stage "${CF_STAGE}" \
        --R "${R}" --seed "${SEED}" --hpc ${CF_EXTRA} "$@"
}

case "${CF_STEP}" in
    demand_eval)  # 0a: reproduce in-sample shares + export shares_elas parquet
        echo "Julia packages check:"; julia --project="${PROJECT_DIR}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
        run_julia cf_0_demand_eval.jl ;;
    cf1)          # CF1: gross franchise-value sleepiness decomposition
        run_julia cf_1_franchise_value.jl ;;
    cost2)        # CF2 part 1: ψ under σ̂ and σ̃ deviations (shardable)
        SHARD_ID="${SLURM_ARRAY_TASK_ID:-${SHARD_ID:-0}}"
        N_SHARDS="${N_SHARDS:-1}"
        run_julia cost_2_fwd_sim.jl --n-shards "${N_SHARDS}" --shard-id "${SHARD_ID}" ;;
    cost_solve)   # CF2 part 2: Eq-18 minimization (Python; light, CPU)
        python "${PROJECT_DIR}/estimation_1_cost_3_solve.py" \
            --estim "${CF_ROUTINE}" --spec 12 --stage "${CF_STAGE}" \
            ${CF_EXTRA} ;;
    *) echo "Unknown CF_STEP='${CF_STEP}'"; exit 1 ;;
esac

echo "CF step ${CF_STEP} (E${CF_ROUTINE} ${CF_STAGE}) complete: $(date)"
