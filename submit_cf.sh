#!/bin/bash
#SBATCH --partition=day
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
#   CF_STEP     demand_eval | cf1 | cost2 | cost_solve       (CF1-gross + CF2 chain)
#               | cf1_net | cf3 | cf5 | cf6                   (equilibrium CFs; need cost_solve first)
#               | zip                                         (CF_WHICH=<cf>: archive that CF's outputs)
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
# PARTITIONS (Bouchet). Default is the CPU `day` partition (64–192 CPU, 990–2251G RAM,
# 1-day wall) — right for demand_eval/cf1/cost2/cost_solve/cf1_net, which are CPU-only
# (they use blp_1_estimation.jl's CPU share kernels). The orchestrators override per job:
#   • CPU steps   → PARTITION=day (or week/bigmem/mpi for longer/bigger)
#   • GPU steps   → PARTITION=gpu_h200 GPUS=h200:1  (cf3/cf5/cf6 use the GPU share path
#                   when a GPU is present; on gpu_h200 you MUST request one or the QOS
#                   rejects the job — "QOSMinGRES"). 48 CPU, 1995G RAM, 2-day wall, 8 H200/node.
# The R=2000 'extended' context peaks ~60G, so keep --mem ≥ ~100G.

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

# GPU share kernel: load CUDA ONLY for the sharded best-response (cf3_shard — the GPU workhorse).
# Every other step is CPU-only; keep CUDA out of them so many concurrent jobs on the shared NFS
# depot don't stampede the Julia precompile/load lock loading a package they never use. Override
# per-run with CF_GPU=1/0 if you deliberately want a GPU (or not) for a given step.
case "${CF_STEP}" in cf3_shard) : "${CF_GPU:=1}" ;; *) : "${CF_GPU:=0}" ;; esac
export CF_GPU

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
    warmup)       # Serially precompile the depot + JIT-load the CF stack ONCE, so the parallel
                  # cost2 array + cf1 don't stampede the shared-NFS depot precompile/load lock (the
                  # cause of jobs sitting at 0% CPU for an hour). Everything else depends on this.
        echo "Precompiling depot…"; julia --project="${PROJECT_DIR}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
        echo "Loading the CF stack (CF_GPU=${CF_GPU})…"
        julia --project="${PROJECT_DIR}" --threads="${SLURM_CPUS_PER_TASK}" \
            -e 'include(joinpath(ENV["SLURM_SUBMIT_DIR"], "cf_0_demand_eval.jl"))' || true
        echo "warmup complete: depot precompiled + CF stack loaded — array can launch warm" ;;
    demand_eval)  # 0a: reproduce in-sample shares + export shares_elas parquet
        echo "Julia packages check:"; julia --project="${PROJECT_DIR}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
        run_julia cf_0_demand_eval.jl ;;
    cf1)          # CF1: gross franchise-value sleepiness decomposition
        run_julia cf_1_franchise_value.jl ;;
    cost2)        # CF2 part 1: ψ under σ̂ and σ̃ deviations (shardable)
        # Forward r^f curve is a REQUIRED input (cost_2 errors under --hpc if absent):
        # a flat r^f makes ψ4∝ψ2 and leaves ζ unidentified. It is fetched from the BCB
        # APIs locally (compute nodes have no internet) and uploaded to data/COST_FWD/.
        RF_CURVE="${RF_CURVE:-${PROJECT_DIR}/../data/COST_FWD/forward_rf_qoq.csv}"
        if [[ ! -f "${RF_CURVE}" ]]; then
            echo "ERROR: forward r^f curve missing: ${RF_CURVE}" >&2
            echo "  Generate locally (needs internet) and upload to data/COST_FWD/:" >&2
            echo "    python cf_forward_rf.py --horizon 50 --start 2026Q1" >&2
            echo "  Refusing to run: a flat r^f leaves ζ unidentified." >&2
            exit 1
        fi
        echo "forward r^f curve OK: ${RF_CURVE}"
        SHARD_ID="${SLURM_ARRAY_TASK_ID:-${SHARD_ID:-0}}"
        N_SHARDS="${N_SHARDS:-1}"
        run_julia cost_2_fwd_sim.jl --n-shards "${N_SHARDS}" --shard-id "${SHARD_ID}" ;;
    cost_solve)   # CF2 part 2: Eq-18 minimization (Python; needs numpy/pandas/scipy, light, CPU).
        # Compute nodes have no bare `python`; load a Python env. Configure for your cluster via env:
        #   PY_MODULE=miniconda   module(s) to load (space-list ok)
        #   CONDA_ENV=<name>      conda env to activate (must have numpy/pandas/scipy)
        #   CF_PYTHON=python      interpreter to call (default python3)
        # One-time setup example:  module load miniconda && conda create -y -n costsolve numpy pandas scipy
        [[ -n "${PY_MODULE:-}" ]] && module load ${PY_MODULE}
        if [[ -n "${CONDA_ENV:-}" ]]; then
            source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"
        fi
        PYBIN="${CF_PYTHON:-python3}"
        command -v "${PYBIN}" >/dev/null 2>&1 || {
            echo "ERROR: Python '${PYBIN}' not found on the node. Set PY_MODULE / CONDA_ENV / CF_PYTHON" >&2
            echo "  e.g.  PY_MODULE=miniconda CONDA_ENV=costsolve ROUTINES=7 bash submit_cf_all.sh" >&2
            exit 127; }
        # The script's default COST_FWD is the local BCB tree, absent on the cluster. Point it at the
        # data/COST_FWD where cost2 (Julia) wrote the psi_* — same dir as the forward r^f curve.
        export CF_COST_FWD="${CF_COST_FWD:-${PROJECT_DIR}/../data/COST_FWD}"
        echo "COST_FWD = ${CF_COST_FWD}"
        "${PYBIN}" "${PROJECT_DIR}/estimation_1_cost_3_solve.py" \
            --estim "${CF_ROUTINE}" --spec 12 --stage "${CF_STAGE}" \
            ${CF_EXTRA} ;;
    cf1_net)      # CF1 net-of-cost franchise value — needs cost_solve output (cost_params_*.json)
        run_julia cf_1_franchise_value.jl --net ;;
    cf3)          # CF3 equilibrium — single-process solve (only tractable on --n-markets subsets)
        run_julia cf_3_equilibrium_spreads.jl ;;
    cf3_init)     # firm-sharded Jacobi: write σ⁰=ρ̂ (SIGMA_DIR from env)
        run_julia cf_3_equilibrium_spreads.jl --write-sigma0 --sigma-out "${SIGMA_DIR}/sig_0.parquet" ;;
    cf3_shard)    # firm-sharded Jacobi: one shard of sweep ${SWEEP} (SLURM_ARRAY_TASK_ID = shard id)
        SID="${SLURM_ARRAY_TASK_ID:-${SHARD_ID:-0}}"
        run_julia cf_3_equilibrium_spreads.jl --n-firm-shards "${N_FIRM_SHARDS}" --firm-shard-id "${SID}" \
            --sigma-in  "${SIGMA_DIR}/sig_$((SWEEP-1)).parquet" \
            --sigma-out "${SIGMA_DIR}/sh_${SWEEP}_${SID}.parquet" ;;
    cf3_merge)    # firm-sharded Jacobi: merge sweep ${SWEEP}'s shards → sig_${SWEEP}
        run_julia cf_3_equilibrium_spreads.jl --jacobi-merge \
            --sigma-in   "${SIGMA_DIR}/sig_$((SWEEP-1)).parquet" \
            --sigma-glob "${SIGMA_DIR}/sh_${SWEEP}_*.parquet" \
            --sigma-out  "${SIGMA_DIR}/sig_${SWEEP}.parquet" ;;
    cf5)          # CF5 monetary pass-through — single-process (subset) solve
        run_julia cf_5_passthrough.jl ;;
    cf6)          # CF6 merger simulation — single-process (subset) solve
        run_julia cf_6_merger.jl ;;
    cf5_compare)  # CF5: combine the base + Selic-shocked equilibria (SIGMA_BASE/SIGMA_SCN from env)
        run_julia cf_5_passthrough.jl --compare --sigma-base "${SIGMA_BASE}" --sigma-scn "${SIGMA_SCN}" ;;
    cf6_compare)  # CF6: combine the base + merged equilibria (CF_EXTRA carries --merge "A,B")
        run_julia cf_6_merger.jl --compare --sigma-base "${SIGMA_BASE}" --sigma-scn "${SIGMA_SCN}" ;;
    zip)          # archive a CF's outputs across all estimations → CF_ZIPS/<cf>_outputs.zip (moves them out)
        DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../data}" \
            bash "${PROJECT_DIR}/zip_cf_outputs.sh" "${CF_WHICH:?set CF_WHICH (foundation|cf1|cf2|cf3|cf4|cf5|cf6)}" ${CF_TAG:-} ;;
    *) echo "Unknown CF_STEP='${CF_STEP}'"; exit 1 ;;
esac

echo "CF step ${CF_STEP} (E${CF_ROUTINE} ${CF_STAGE}) complete: $(date)"
