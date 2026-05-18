#!/bin/bash
#SBATCH --job-name=blp_E5_gpu
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --output=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_E5_gpu_%j.out
#SBATCH --error=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_E5_gpu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

# ── Environment ───────────────────────────────────────────────────────────────
module reset
module load Julia/1.10.4-linux-x86_64
module load CUDA/12.4.0

export JULIA_MKL_THREADING=tbb
# CUDA.jl reads this to locate the CUDA toolkit installed by the module
export JULIA_CUDA_USE_BINARYBUILDER=false
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

echo "GPU node: $(hostname)"
echo "CUDA devices: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── Verify environment ────────────────────────────────────────────────────────
echo "Checking Julia packages: $(date)"
julia --project="${PROJECT_DIR}" -e "using Pkg; Pkg.instantiate()"

echo "======================================"
echo " BLP Estimation E5 GPU — $(date)"
echo " stage=sigma | R=2000 | threads=${SLURM_CPUS_PER_TASK}"
echo "======================================"

julia --project="${PROJECT_DIR}" --threads=${SLURM_CPUS_PER_TASK} \
    "${PROJECT_DIR}/blp_estimation_gpu.jl" \
    --estim 5 --spec 12 --stage sigma \
    --R 2000 --seed 42 \
    --tol-inner 1e-12 --max-inner 5000 --tol-outer 1e-6 \
    --hpc

echo "E5 GPU complete: $(date)"
