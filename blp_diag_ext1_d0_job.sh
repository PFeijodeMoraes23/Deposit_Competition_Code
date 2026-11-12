#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=06:00:00
# ==============================================================================
# blp_diag_ext1_d0_job.sh — one GPU job running blp_diag_ext1_d0.jl (D0 of the RC-BLP engine plan,
# 2026-11-11) for one routine: the ext1 starts' first evaluation and first trial, Q along the first
# step, Q at the production theta* from delta*, the logit delta-hat and a polluted delta, the stage
# test, and the |ln 0.95| plateau. Read-only on the production data; writes
# data/output/diag_ext1_d0/diag_ext1_d0_E<k>.json and its log.
#
# Submit from scripts/ (one line per routine):
#   sbatch --parsable -J d0_ext1_E3 -o logs/%x_%j.out -e logs/%x_%j.err --export=ALL,DIAG_ROUTINE=3 blp_diag_ext1_d0_job.sh
# Optional: DIAG_PARTS (default i+ii+iii+iv; '+' since --export splits on commas), D0_CLEAN_CAP
# (10000), D0_REEVAL_CAP (20000).
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e
: "${DIAG_ROUTINE:?set DIAG_ROUTINE (3 or 4)}"
case "${DIAG_ROUTINE}" in
    3|4) ;;
    *) echo "DIAG_ROUTINE must be 3 or 4" >&2; exit 2 ;;
esac

mkdir -p "${CL_ROOT}/logs"
cl_bootstrap_tree
cl_export_step_dirs
export DIAG_OUT="${CL_ROOT}/../data/output/diag_ext1_d0"
mkdir -p "${DIAG_OUT}"

# The production inputs D0 reads, hashed into the log (the job writes none of them).
for f in "${CL_ROOT}/../data/output/blp/blp_checkpoint_E${DIAG_ROUTINE}_spec_12_ext1.jls" \
         "${CL_ROOT}/../data/output/blp/blp_checkpoint_E${DIAG_ROUTINE}_spec_12_ext1.bin" \
         "${CL_ROOT}/../data/output/blp/blp_results_E${DIAG_ROUTINE}_spec_12_ext1.json" \
         "${CL_ROOT}/../data/output/logit/logit_delta_E${DIAG_ROUTINE}_spec_12.bin" \
         "${CL_ROOT}/config/blp_ext1_extra_starts.csv"; do
    if [[ -f "${f}" ]]; then sha256sum "${f}"; else echo "missing (D0 may stop): ${f}"; fi
done

cl_load_julia
echo "GPU node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cl_require_sysimage gpu
if [[ -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]]; then
    cl_gpu_gate
fi

julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} \
    --threads="${SLURM_CPUS_PER_TASK:-6}" \
    "${CL_ROOT}/blp_diag_ext1_d0.jl"
echo "D0 ext1 E${DIAG_ROUTINE} done: $(date)"
