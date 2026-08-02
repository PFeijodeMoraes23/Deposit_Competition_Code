#!/bin/bash
#SBATCH --partition=day
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# (job-name + .out/.err set per submission via sbatch -J/-o/-e)
#
# BBL cost-estimation driver (the marginal-cost stage; sibling of submit_blp_rc_stage.sh).
# This was formerly the cost2/cost_solve steps of submit_cf.sh; it is now its own
# ESTIMATION stage — it produces cost_params_*.json, which the counterfactuals CONSUME.
# Driven by env vars:
#   BBL_STEP     fwd_sim | solve                                  (BBL Step-2 chain)
#                | polfunc                                        (Step-1; usually run LOCALLY, off by default)
#                | warmup                                         (precompile depot + JIT-load the stack)
#   BBL_ROUTINE  5 | 6 (E6 headline) | 7 | 8                      → --estim
#   BBL_STAGE    extended (headline) | full | logit | …           → --stage
#   R, SEED      draws (default 2000 / 42) — must match the downloaded BLP_DRAWS
#   BBL_EXTRA    extra CLI flags passed through (e.g. "--shocks 50 --beta 0.9 --policy-csv <csv>";
#               solve: "--bootstrap 200 --profile")
#   N_SHARDS     fwd_sim only: total deviation shards (default 1)
#   SHARD_ID     fwd_sim only: this shard (default = SLURM_ARRAY_TASK_ID, else 0)
#
# To beat Bouchet time walls, submit fwd_sim as a JOB ARRAY (one shard per task) and
# the solve as an afterany dependency — see submit_bbl_all.sh (the orchestrator). All three
# steps are CPU-only (they use foundation_demand_eval.jl's CPU share kernels); the panel
# now iterates ~700k demand rows, so the header is sized larger than the CF driver
# (mem=256G, time=12:00:00). The orchestrator overrides --mem/-t per step.

set -euo pipefail
: "${BBL_STEP:?set BBL_STEP (fwd_sim|solve|polfunc)}"
BBL_ROUTINE="${BBL_ROUTINE:-6}"
BBL_STAGE="${BBL_STAGE:-extended}"
R="${R:-2000}"
SEED="${SEED:-42}"
BBL_EXTRA="${BBL_EXTRA:-}"

# ── Environment (match the RC-BLP estimation so paths/suffixes line up) ─────────
module reset
module load Julia/1.11.4-linux-x86_64
export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/.julia_depot:${JULIA_DEPOT_PATH:-}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# CF_GPU is set by the orchestrator (submit_bbl_all.sh: FWD_GPU=1 → warmup + fwd_sim on the H200,
# solve/polfunc CPU). Default OFF here so a bare `BBL_STEP=… sbatch submit_bbl.sh` stays CPU and
# concurrent array tasks on the shared NFS depot don't stampede the precompile/load lock loading CUDA.
: "${CF_GPU:=0}"; export CF_GPU

# ── Sysimage: load blp_sysimage.so if present, exactly like the RC-BLP path (submit_blp_rc_stage.sh).
# WHY THIS MATTERS FOR GPU: without it, each concurrent array task loads packages fresh from the ONE
# shared-NFS depot and stampedes the precompile cache. A task that loses that race on CUDA's dependency
# chain hits CUDA.functional()==false, which foundation_demand_eval.jl catches and silently downgrades
# to the CPU share path — the job then holds its H200 at 0% util. The sysimage bakes CUDA + the whole
# package closure, so tasks touch the precompile cache zero times → no race → the GPU engages on every
# task. (Diagnosed 2026-07-31: 5/11 fwd_sim tasks fell back to CPU on FillArrays/StaticArrays cache races.)
SYSIMAGE="${PROJECT_DIR}/blp_sysimage.so"
JULIA_SYS=()
if [ -f "${SYSIMAGE}" ]; then
    JULIA_SYS=(--sysimage "${SYSIMAGE}")
    echo "Julia sysimage: ${SYSIMAGE} (no per-task precompile)"
elif [ "${CF_GPU}" != "0" ]; then
    echo "[!] CF_GPU=1 but no blp_sysimage.so — concurrent array tasks may lose the CUDA precompile"
    echo "    race on the shared depot → silent CPU fallback (0% GPU util). Build it once, on gpu_h200:"
    echo "      sbatch submit_build_sysimage.sh"
fi

echo "======================================"
echo " BBL cost stage | step=${BBL_STEP} | E${BBL_ROUTINE} | stage=${BBL_STAGE}"
echo " R=${R} seed=${SEED} threads=${SLURM_CPUS_PER_TASK} | $(date)"
echo " node=$(hostname)"
echo "======================================"

run_julia () {
    local script="$1"; shift
    julia --project="${PROJECT_DIR}" "${JULIA_SYS[@]}" --threads="${SLURM_CPUS_PER_TASK}" \
        "${PROJECT_DIR}/${script}" \
        --estim "${BBL_ROUTINE}" --spec 12 --stage "${BBL_STAGE}" \
        --R "${R}" --seed "${SEED}" --hpc ${BBL_EXTRA} "$@"
}

# The Step-1 polfunc and the Step-3 solve are Python (numpy/pandas/scipy/pyarrow/statsmodels — pyarrow
# because the solve pd.read_parquet's the psi shards; light, CPU). Compute nodes' bare `python3` has
# NONE of that stack (2026-08-01: the solve died at t+4s on `No module named 'pandas'`), so default to
# the Bouchet conda env and VERIFY the imports before running. Override per cluster via env:
#   PY_MODULE=miniconda   module(s) to load (space-list ok; default miniconda)
#   CONDA_ENV=<name>      conda env to activate (default costsolve; CONDA_ENV="" opts out)
#   CF_PYTHON=python      interpreter to call (default python3)
# One-time setup:  module load miniconda && conda create -y -n costsolve python=3.11 numpy pandas scipy pyarrow statsmodels
[[ -z "${PY_MODULE+set}" ]]  && PY_MODULE=miniconda    # default only when truly UNSET (empty ⇒ opt out)
[[ -z "${CONDA_ENV+set}" ]]  && CONDA_ENV=costsolve
setup_python () {
    [[ -n "${PY_MODULE:-}" ]] && module load ${PY_MODULE}
    if [[ -n "${CONDA_ENV:-}" ]]; then
        source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"
    fi
    PYBIN="${CF_PYTHON:-python3}"
    command -v "${PYBIN}" >/dev/null 2>&1 || {
        echo "ERROR: Python '${PYBIN}' not found on the node. Set PY_MODULE / CONDA_ENV / CF_PYTHON" >&2
        echo "  e.g.  PY_MODULE=miniconda CONDA_ENV=costsolve ROUTINES=7 bash submit_bbl_all.sh" >&2
        exit 127; }
    # Probe the actual imports, not just the binary — bare node python3 exists but has no sci stack,
    # and a Traceback 14h into the pipeline (after the fwd_sim array drains) is the failure mode this
    # prevents. Fail loud, with the one-time fix.
    "${PYBIN}" -c 'import numpy, pandas, scipy, pyarrow, statsmodels' 2>/dev/null || {
        echo "ERROR: '${PYBIN}' lacks the solve stack (numpy/pandas/scipy/pyarrow/statsmodels)." >&2
        echo "  One-time setup:  module load miniconda && conda create -y -n costsolve python=3.11 numpy pandas scipy pyarrow statsmodels" >&2
        echo "  Then submit with the defaults (PY_MODULE=miniconda CONDA_ENV=costsolve), or override them." >&2
        exit 127; }
    # The scripts' default COST_FWD is the local BCB tree, absent on the cluster. Point it at the
    # data/output/cost where fwd_sim (Julia) writes the psi_* this solve reads.
    # Cluster layout (2026-08-02): the BBL cost family (psi_eq, psi_dev shards, cost_params) lives in
    # data/output/cost — matching cf_out_dir(out_dir,"COST_FWD") in foundation_demand_eval.jl, which is
    # where the Julia fwd_sim writes the psi this solve reads. Legacy data/COST_FWD is retired.
    export CF_COST_FWD="${CF_COST_FWD:-${PROJECT_DIR}/../data/output/cost}"
    echo "COST_FWD = ${CF_COST_FWD}"
}

case "${BBL_STEP}" in
    warmup)       # JIT-load the BBL stack ONCE. With the sysimage, packages are already baked so the
                  # array never precompiles → no shared-NFS stampede. Without it, precompile the depot
                  # here so the parallel fwd_sim array at least doesn't all precompile at once.
        if [ ${#JULIA_SYS[@]} -gt 0 ]; then
            echo "Sysimage present — skipping depot precompile."
        else
            echo "Precompiling depot…"; julia --project="${PROJECT_DIR}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
        fi
        echo "Loading the BBL stack (CF_GPU=${CF_GPU})…"
        julia --project="${PROJECT_DIR}" "${JULIA_SYS[@]}" --threads="${SLURM_CPUS_PER_TASK}" \
            -e 'include(joinpath(ENV["SLURM_SUBMIT_DIR"], "estimation_bbl_2_fwd_sim.jl"))' || true
        echo "warmup complete: BBL stack loaded — array can launch warm" ;;
    polfunc)      # BBL Step 1: fit the parametric policy function (usually run LOCALLY and uploaded;
                  # this cluster path exists for reproducibility, off by default in the orchestrator).
                  # Reads market_panel.csv → writes polfunc_fitted.csv (upload to data/input/).
                  # NOTE: the policy function is SPEC-INVARIANT — it takes no --spec (unlike the
                  # fwd_sim/solve below, whose --spec 12 selects the BLP demand specification).
        setup_python
        "${PYBIN}" "${PROJECT_DIR}/estimation_bbl_1_polfunc.py" ${BBL_EXTRA} ;;
    fwd_sim)      # BBL Step 2 part 1: ψ under σ̂ and σ̃ deviations (shardable).
        # Forward r^f curve is a REQUIRED input (the sim errors under --hpc if absent):
        # a flat r^f makes ψ4∝ψ2 and leaves ζ unidentified. It is fetched from the BCB
        # APIs locally (compute nodes have no internet) and uploaded to data/input/.
        # UPLOADED input (fetched locally from the BCB APIs — compute nodes have no internet)
        RF_CURVE="${RF_CURVE:-${PROJECT_DIR}/../data/input/forward_rf_qoq.csv}"
        if [[ ! -f "${RF_CURVE}" ]]; then
            echo "ERROR: forward r^f curve missing: ${RF_CURVE}" >&2
            echo "  Generate locally (needs internet) and upload to data/input/:" >&2
            echo "    python cf_forward_rf.py --horizon 50 --start 2026Q1" >&2
            echo "  Refusing to run: a flat r^f leaves ζ unidentified." >&2
            exit 1
        fi
        echo "forward r^f curve OK: ${RF_CURVE}"
        SHARD_ID="${SLURM_ARRAY_TASK_ID:-${SHARD_ID:-0}}"
        N_SHARDS="${N_SHARDS:-1}"
        run_julia estimation_bbl_2_fwd_sim.jl --n-shards "${N_SHARDS}" --shard-id "${SHARD_ID}" ;;
    solve)        # BBL Step 2 part 2: eq:17 minimization → data/output/cost/cost_params_*.json.
        setup_python
        "${PYBIN}" "${PROJECT_DIR}/estimation_bbl_3_solve.py" \
            --estim "${BBL_ROUTINE}" --spec 12 --stage "${BBL_STAGE}" \
            ${BBL_EXTRA} ;;
    *) echo "Unknown BBL_STEP='${BBL_STEP}'"; exit 1 ;;
esac

echo "BBL step ${BBL_STEP} (E${BBL_ROUTINE} ${BBL_STAGE}) complete: $(date)"
