#!/bin/bash
#SBATCH --job-name=blp_build_sysimg
#SBATCH --partition=gpu_h200
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h200:1
#SBATCH --output=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_build_sysimg_%j.out
#SBATCH --error=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_build_sysimg_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

# Build the BLP Julia sysimage ON a GPU node so CUDA's GPU-specific code is baked in.
# Produces scripts/blp_sysimage.so, which the estimation submit scripts then use
# automatically (each job starts in seconds instead of re-precompiling CUDA + packages).
#
# IMPORTANT: build on the SAME partition the estimation runs on (gpu_h200). A sysimage
# bakes the build node's CPU target; if built on a different node type (e.g. gpu_devel)
# the run node rejects it ("Unable to find compatible target in cached code image /
# Target 0 (generic): Rejecting this target due to use of runtime-disabled features")
# and every job silently falls back to no-sysimage. Building here on gpu_h200 matches the
# run CPU. (Alternative: export a portable JULIA_CPU_TARGET multiversion string before the
# build instead of pinning the node — but matching the partition is the simplest fix.)
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

# Do NOT Pkg.resolve() here: that rewrites Manifest.toml and should only happen once,
# via `sbatch setup_julia_env.sh`. Instantiate only (safe to run alongside other jobs).
julia --project="${PROJECT_DIR}" -e 'using Pkg; Pkg.instantiate()'

BLP_SYSIMAGE_WORKLOAD="${BLP_SYSIMAGE_WORKLOAD:-0}" \
    julia --project="${PROJECT_DIR}" --threads="${SLURM_CPUS_PER_TASK}" \
    "${PROJECT_DIR}/build_blp_sysimage.jl"

echo "Sysimage build complete: $(date)"
ls -la "${PROJECT_DIR}/blp_sysimage.so" || echo "[!] blp_sysimage.so not found — check the log above."
