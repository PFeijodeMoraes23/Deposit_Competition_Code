#!/bin/bash
#SBATCH --job-name=blp_logit
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# ==============================================================================
# logit_job.sh — the plain-logit delta warm-starts, on the cluster.
#
#   julia --project=. blp_1_logit.jl --hpc
#
# WHAT IT PRODUCES, and why the RC stage wants it
#   data/output/logit_delta_E{k}_spec_12.bin   the delta warm-start blp_2_rc.jl /
#       blp_gpu_engine.jl load before the sigma stage. Version-agnostic binary:
#       one leading Int64 length, then that many Float64. delta_dir() is
#       data/output ITSELF on the cluster — flat, which is where the RC engine
#       looks.
#   data/output/logit/logit_summary_spec_12.json  the combined 2SLS summary
#       (alpha and the X coefficients per routine x sub-model) and the per-key JLS.
#   data/output/Rout/*.tex                     the logit tables.
#
# WHY IT IS ITS OWN JOB AND NOT PART OF THE RC CHAIN
#   It reads the demand parquets and nothing else, so it can run the moment G2 is
#   green — in parallel with everything else the pipeline is doing — and the RC
#   chain then starts warm. It needs neither a GPU nor a sysimage: it is one 2SLS
#   per routine x sub-model, minutes of CPU, and pulling CUDA in would only add
#   this job to the shared-NFS precompile race for a package it never calls.
#
# WHAT MUST EXIST FIRST
#   the demand parquets — data/output/DEMAND_PREP/demand_*_spec_12.parquet from
#   the sleepiness prep step (gate G2), or uploaded to data/input. blp_1_logit.jl
#   searches data/input first, then data/output/DEMAND_PREP.
# WHAT TO RUN NEXT
#   gate G4 (pipeline_all.sh submits it afterok this job), then blp_run.sh.
#
# Env vars: SLEEP_ACTIVE_ESTS (default '1 2 3 4') — the ACTIVE lineup filter.
#   blp_1_logit.jl discovers routines by FILE PRESENCE, so a demand parquet left
#   over from a routine no longer in the lineup silently re-enters the run. This
#   is the guard, and it must agree with estimation_demand_1_prep.py and
#   export_results.py, which read the same variable.
#
# Usage
#   sbatch logit_job.sh                          # standalone, the whole lineup
#   sbatch --export=ALL,SLEEP_ACTIVE_ESTS="3 4" logit_job.sh
#   (pipeline_all.sh submits it afterok G2 and chains G4 behind it)
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

SPEC="${SPEC:-12}"
export SLEEP_ACTIVE_ESTS="${SLEEP_ACTIVE_ESTS:-${CL_ROUTINES_ALL}}"

mkdir -p "${CL_ROOT}/logs" "${CL_DATA_OUT}"

# --hpc resolves data/{input,output} directly and never calls resolve_of_root(),
# so the skeleton is not on this job's critical path. Build it anyway: it is
# idempotent and it keeps the ONE path that would otherwise throw — a run where
# the flag is dropped — resolving to the same tree as every other job.
cl_bootstrap_tree || echo "[!] skeleton bootstrap reported a problem (above) — --hpc does not need it."
export OPEN_FINANCE_ROOT="$(cl_of_root)"

cl_load_julia

# No sysimage and no CUDA gate, deliberately. cl_require_sysimage would refuse this
# job over a cpu image it has no use for, and the GPU images are for the RC/CF
# stack. The cost is one cold JIT of the logit stack, ~1 minute.
cl_banner "Logit delta warm-starts | spec ${SPEC}" \
          "active lineup: SLEEP_ACTIVE_ESTS='${SLEEP_ACTIVE_ESTS}'" \
          "julia $(julia --version 2>/dev/null | sed 's/^julia version //') | threads=${SLURM_CPUS_PER_TASK:-8}" \
          "node=$(hostname) | $(date)"

julia --project="${CL_ROOT}" --threads="${SLURM_CPUS_PER_TASK:-8}" \
      "${CL_ROOT}/blp_1_logit.jl" --hpc ${LOGIT_EXTRA:-}

echo
echo "-- what landed --"
ls -la "${CL_DATA_OUT}"/logit_delta_E*_spec_"${SPEC}".bin 2>/dev/null || echo "  (no logit_delta_E*_spec_${SPEC}.bin — gate G4 will say so)"
ls -la "${CL_DATA_OUT}"/logit/logit_summary_spec_"${SPEC}".json 2>/dev/null || true
echo "logit job complete: $(date)"
