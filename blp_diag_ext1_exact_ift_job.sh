#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=03:00:00
# ==============================================================================
# blp_diag_ext1_exact_ift_job.sh — one GPU job running blp_diag_ext1_exact_ift.jl for one routine:
# the exact implicit-function Jacobian of delta (blp_exact_ift.jl) against the engine's diagonal one
# and against finite differences of delta, with the gradients and standard errors each implies, at
# rc4's point and at the adopted ext1 fit. Read-only on the production data; writes
# data/output/diag_ext1_grad/diag_ext1_exact_ift_E<k>.json and its log.
#
# Submit from scripts/ (one line per routine):
#   sbatch --parsable -J exact_ift_E3 -o logs/%x_%j.out -e logs/%x_%j.err --export=ALL,DIAG_ROUTINE=3 blp_diag_ext1_exact_ift_job.sh
# DIAG_ADOPTED defaults to the adopted multi-start fit of 2026-10-30 (E3 best1019, E4 rand1) in
# EXT1_DIAG_ROOT (default $HOME/scratch_pi_mf2263/pf382)/diag_ext1_20261029.
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e
: "${DIAG_ROUTINE:?set DIAG_ROUTINE (3 or 4)}"
case "${DIAG_ROUTINE}" in
    3) start=best1019 ;;
    4) start=rand1 ;;
    *) echo "DIAG_ROUTINE must be 3 or 4" >&2; exit 2 ;;
esac
export DIAG_ADOPTED="${DIAG_ADOPTED:-${EXT1_DIAG_ROOT:-${HOME}/scratch_pi_mf2263/pf382}/diag_ext1_20261029/E${DIAG_ROUTINE}_${start}/data/output/blp/blp_checkpoint_E${DIAG_ROUTINE}_spec_12_ext1.jls}"
[[ -f "${DIAG_ADOPTED}" ]] || { echo "missing ${DIAG_ADOPTED}" >&2; exit 2; }

mkdir -p "${CL_ROOT}/logs"
cl_bootstrap_tree
cl_export_step_dirs
export DIAG_OUT="${CL_ROOT}/../data/output/diag_ext1_grad"
mkdir -p "${DIAG_OUT}"

cl_load_julia
echo "GPU node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cl_require_sysimage gpu
if [[ -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]]; then
    cl_gpu_gate
fi

julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} \
    --threads="${SLURM_CPUS_PER_TASK:-6}" \
    "${CL_ROOT}/blp_diag_ext1_exact_ift.jl"
echo "exact IFT E${DIAG_ROUTINE} done: $(date)"
