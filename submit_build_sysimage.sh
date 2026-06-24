#!/bin/bash
#SBATCH --job-name=blp_build_sysimg
#SBATCH --partition=gpu_devel
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_build_sysimg_%j.out
#SBATCH --error=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_build_sysimg_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

# Build the BLP Julia sysimage ON a GPU node so CUDA's GPU-specific code is baked in.
# Produces scripts/blp_sysimage.so, which the estimation submit scripts then use
# automatically (each job starts in seconds instead of re-precompiling CUDA + packages).
#
# Set BLP_SYSIMAGE_WORKLOAD=1 to also bake the estimation hot path (needs the demand
# parquets + draws already in data/input + data/output/BLP_DRAWS):
#   BLP_SYSIMAGE_WORKLOAD=1 sbatch submit_build_sysimage.sh
#
# Run this ONCE (rebuild only if Manifest.toml / package versions change). After it
# finishes, just submit the estimation jobs normally — they detect the sysimage.

module reset
module load Julia/1.11.4-linux-x86_64
export JULIA_MKL_THREADING=tbb
export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/.julia_depot:${JULIA_DEPOT_PATH:-}"
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

echo "GPU node: $(hostname)"
echo "CUDA devices: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Building BLP sysimage: $(date)"

julia --project="${PROJECT_DIR}" -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

BLP_SYSIMAGE_WORKLOAD="${BLP_SYSIMAGE_WORKLOAD:-0}" \
    julia --project="${PROJECT_DIR}" --threads="${SLURM_CPUS_PER_TASK}" \
    "${PROJECT_DIR}/build_blp_sysimage.jl"

echo "Sysimage build complete: $(date)"
ls -la "${PROJECT_DIR}/blp_sysimage.so" || echo "[!] blp_sysimage.so not found — check the log above."
