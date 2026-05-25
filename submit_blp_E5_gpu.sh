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
# Pin Julia 1.11.4 — ships LLVM 16.0.6, which is the minimum required to
# emit PTX for sm_89 (RTX 5000 Ada) and sm_90 (H100). Julia 1.10's LLVM 15
# clamps Ada to sm_86/PTX 7.5 and triggers a PTXCompilerTarget MethodError.
module load Julia/1.11.4-linux-x86_64
# NOTE: We deliberately do NOT load a system CUDA module. CUDA.jl ships its
# own toolkit via JLL artifacts (CUDA_Runtime_jll) and chooses a version
# compatible with the detected driver. Loading a system CUDA module is only
# needed if you call `CUDA.set_runtime_version!("local")`, which we don't.
# (The previous `module load CUDA/12.4.0` failed on Bouchet — that exact
# version is not installed; run `module spider CUDA` to see what is.)

export JULIA_MKL_THREADING=tbb
# JULIA_CUDA_USE_BINARYBUILDER is deprecated since CUDA.jl 5.x — removed.
# Keep Julia depots on the project filesystem so artifacts (CUDA toolkit ~3GB)
# don't blow up the small $HOME quota.
export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/.julia_depot:${JULIA_DEPOT_PATH:-}"
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

echo "GPU node: $(hostname)"
echo "CUDA devices: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "NVIDIA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

# ── Verify environment ────────────────────────────────────────────────────────
# Pkg.resolve() recomputes the Manifest from Project.toml compat bounds, fixing
# any stale-manifest drift from the cluster environment. Pkg.update("CUDA") was
# removed: it triggered a CUDACore 6.1.1 conflict with the CUDA = "5.8-6.0" bound.
echo "Checking Julia packages: $(date)"
julia --project="${PROJECT_DIR}" -e '
    using Pkg
    Pkg.resolve()        # recompute Manifest from Project.toml compat bounds (fixes stale-manifest warning)
    Pkg.instantiate()    # install resolved versions
    Pkg.precompile()
    using CUDA
    @info "CUDA.jl version" CUDA.versioninfo()
'

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
