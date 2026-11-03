#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=03:00:00
# ==============================================================================
# blp_l1_items_job.sh — one GPU job running blp_l1_items.jl for one routine: the L1 items (the
# random-coefficients own-spread elasticities, the distribution of the consumer's price coefficient,
# the demographic means) at an ext1 fit. Read-only on the production data; writes
# data/output/diag_ext1_grad/l1_items_E<k>_<label>.json and its log.
#
# Submit from scripts/ (one line per routine):
#   sbatch --parsable -J l1_items_E3 -o logs/%x_%j.out -e logs/%x_%j.err --export=ALL,L1_ROUTINE=3,L1_MEAN_PHI=<mean phi-hat> blp_l1_items_job.sh
# L1_CHECKPOINT defaults to the production ext1 checkpoint (data/output/blp); point it at a sandbox's
# checkpoint, with L1_LABEL naming it, to evaluate a candidate fit before adoption. L1_MEAN_PHI (the
# paper's sleep.meanphi_frac.E<k>) adds the on-impact elasticities; without it they are skipped.
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e
: "${L1_ROUTINE:?set L1_ROUTINE (3 or 4)}"

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
    "${CL_ROOT}/blp_l1_items.jl"
echo "L1 items E${L1_ROUTINE} done: $(date)"
