#!/bin/bash
#SBATCH --job-name=blp_build_sysimg_cpu
#SBATCH --partition=day
#SBATCH --constraint=cpugen:turin
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
#SBATCH --output=logs/blp_build_sysimg_cpu_%j.out
#SBATCH --error=logs/blp_build_sysimg_cpu_%j.err
#
# Build the CPU-partition Julia sysimage -> scripts/blp_sysimage_cpu.so
#
# WHY A SECOND SYSIMAGE: a sysimage bakes the BUILD node's CPU target. blp_sysimage.so is built on
# gpu_h200 (sapphirerapids) and is REJECTED on `day` nodes with
#     ERROR: Unable to find compatible target in cached code image.
#     Target 0 (sapphirerapids): Rejecting this target due to use of runtime-disabled features
# (this is exactly what killed cf_demaneval on 2026-08-02 after 59 min of depot thrash). Without a
# matching image, every CPU task falls back to precompiling and they stampede the shared NFS depot.
#
# The `day` partition is HETEROGENEOUS — cpugen:turin (AMD 9575f/9655) AND cpugen:emeraldrapids
# (Intel 8562Y+). An image built on one is rejected on the other, so BOTH this build and the CPU
# fwd_sim array pin --constraint=cpugen:turin (matching CPU_CONSTRAINT in submit_bbl_all.sh).
# Turin is also the bigger hardware: 2305 GB / 128 cores vs 1015 GB / 64.
#
# So: build ON the same partition the CPU jobs run on (`day`). No GPU is requested and CUDA is
# EXCLUDED from the image — with CF_GPU=0 the CF/BBL stack never includes blp_gpu_engine.jl, so
# CUDA is dead weight. That also makes this build much faster than the GPU one.
#
# Run ONCE (rebuild only if Manifest.toml / package versions change):
#     sbatch submit_build_sysimage_cpu.sh
# Then CPU runs pick it up automatically:
#     FWD_GPU=0 bash submit_bbl_all.sh
#

module reset
module load Julia/1.11.4-linux-x86_64
export JULIA_MKL_THREADING=tbb
export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/.julia_depot:${JULIA_DEPOT_PATH:-}"
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

echo "CPU node: $(hostname)"
echo "CPU model: $(lscpu | grep -m1 'Model name' | cut -d: -f2 | xargs)"
echo "Building CPU sysimage (no CUDA): $(date)"

julia --project="${PROJECT_DIR}" -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

BLP_SYSIMAGE_CPU=1 BLP_SYSIMAGE_WORKLOAD="${BLP_SYSIMAGE_WORKLOAD:-0}" \
    julia --project="${PROJECT_DIR}" --threads="${SLURM_CPUS_PER_TASK}" \
    "${PROJECT_DIR}/build_blp_sysimage.jl"

echo "CPU sysimage build complete: $(date)"
ls -la "${PROJECT_DIR}/blp_sysimage_cpu.so" || echo "[!] blp_sysimage_cpu.so not found — check the log above."
# Prove it loads on THIS node type (the whole point of building here).
julia --project="${PROJECT_DIR}" --sysimage "${PROJECT_DIR}/blp_sysimage_cpu.so" \
    -e 'println("sysimage loads OK on ", gethostname())'
