#!/bin/bash
# setup_julia_env.sh
# ==================
# ONE-TIME script: builds a Julia 1.10-compatible Manifest.toml on the cluster
# and precompiles all packages (including MKL and LoopVectorization).
#
# Run this ONCE after cloning/pulling changes that touch Project.toml.
# All subsequent compute jobs (submit_blp_*.sh) will just call Pkg.instantiate()
# to verify the environment is intact — no re-resolution needed.
#
# Submit: sbatch setup_julia_env.sh
# Expected runtime: 5–15 min (MKL artifact download is the slow part)
#
#SBATCH --job-name=julia_setup
#SBATCH --partition=day
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/julia_setup_%j.out
#SBATCH --error=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/julia_setup_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

# ── Environment ───────────────────────────────────────────────────────────────
module reset
module load Julia/1.10.4-linux-x86_64
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
echo "Project dir: ${PROJECT_DIR}"
echo "Julia: $(julia --version)"

# ── Remove any manifest from a different Julia version ───────────────────────
# Local machine likely uses Julia 1.12.x; cluster uses 1.10.4. The manifest
# encodes the exact Julia version and is not portable across major/minor versions.
if [ -f "${PROJECT_DIR}/Manifest.toml" ]; then
    MANIFEST_VER=$(grep 'julia_version' "${PROJECT_DIR}/Manifest.toml" | head -1 | grep -oP '"\K[^"]+' || echo "unknown")
    echo "Found Manifest.toml (julia_version = ${MANIFEST_VER})"
    if ! grep -q 'julia_version = "1.10' "${PROJECT_DIR}/Manifest.toml"; then
        echo "  → Incompatible with Julia 1.10 — removing."
        rm -f "${PROJECT_DIR}/Manifest.toml"
    else
        echo "  → Compatible — keeping."
    fi
fi

# ── Resolve and instantiate ───────────────────────────────────────────────────
# Pkg.resolve()     : re-solves dependencies from Project.toml → writes Manifest.toml
# Pkg.instantiate() : downloads + builds all packages listed in Manifest.toml
# Pkg.precompile()  : compiles all packages to .ji cache (speeds up subsequent loads)
echo ""
echo "=== Resolving packages: $(date) ==="
julia --project="${PROJECT_DIR}" -e "
    using Pkg
    println(\"  Updating package registry...\")
    Pkg.Registry.update()
    println(\"  Registry updated.\")
    Pkg.resolve()
    println(\"  Manifest written.\")
    Pkg.instantiate()
    println(\"  Packages installed.\")
    Pkg.precompile()
    println(\"  Precompilation done.\")
"

# ── Verify key packages load ──────────────────────────────────────────────────
echo ""
echo "=== Smoke test: $(date) ==="
julia --project="${PROJECT_DIR}" -e "
    using MKL               # loads Intel MKL BLAS/LAPACK
    using LoopVectorization # @turbo macro
    using LinearAlgebra, DataFrames, Optim, ArgParse
    println(\"  All key packages loaded OK.\")
    println(\"  BLAS: \", BLAS.get_config())
    println(\"  Threads: \", Threads.nthreads())
"

# ── CUDA smoke test (only runs if a GPU is present on this node) ──────────────
echo ""
echo "=== CUDA smoke test: $(date) ==="
julia --project="${PROJECT_DIR}" -e "
    using CUDA
    if CUDA.functional()
        println(\"  CUDA functional: \", CUDA.name(CUDA.device()))
        println(\"  VRAM: \", round(CUDA.totalmem(CUDA.device())/2^30, digits=1), \" GB\")
        println(\"  CUDA version: \", CUDA.version())
        # Quick kernel compile check
        x = CUDA.zeros(Float32, 128, 128)
        y = x .* 2f0
        CUDA.synchronize()
        println(\"  Kernel smoke test passed.\")
    else
        println(\"  CUDA not functional on this node (expected on CPU-only nodes).\")
    end
"

echo ""
echo "=== Julia environment ready: $(date) ==="
echo "You can now submit compute jobs:"
echo "  sbatch submit_blp_logit.sh   # warm-start delta (~5 min)"
echo "  sbatch submit_blp_draws.sh   # pre-compute draws (~20 sec)"
echo "  sbatch submit_blp_E5.sh      # sigma estimation (CPU, 32 threads)"
echo "  sbatch submit_blp_E5_gpu.sh  # sigma estimation (GPU)"
