#!/bin/bash
# setup_julia_env.sh
# ==================
# ONE-TIME script: builds a Julia 1.11-compatible Manifest.toml on the cluster
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
# MUST match the Julia module every compute job loads (submit_blp_*.sh, submit_bbl.sh,
# submit_cf.sh, submit_build_sysimage*.sh all load 1.11.4) — a mismatched Manifest.toml
# (built under one Julia minor version, read under another) plus concurrent Pkg writes
# from parallel jobs on NFS is what corrupts Manifest.toml. This script is the ONLY
# place that should call Pkg.resolve(); every other script must only Pkg.instantiate().
module reset
module load Julia/1.11.4-linux-x86_64
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
echo "Project dir: ${PROJECT_DIR}"
echo "Julia: $(julia --version)"

# ── Remove any manifest from a different Julia version ───────────────────────
# Local machine likely uses Julia 1.12.x; cluster compute jobs use 1.11.4. The manifest
# encodes the exact Julia version and is not portable across major/minor versions.
if [ -f "${PROJECT_DIR}/Manifest.toml" ]; then
    MANIFEST_VER=$(grep 'julia_version' "${PROJECT_DIR}/Manifest.toml" | head -1 | grep -oP '"\K[^"]+' || echo "unknown")
    echo "Found Manifest.toml (julia_version = ${MANIFEST_VER})"
    if ! grep -q 'julia_version = "1.11' "${PROJECT_DIR}/Manifest.toml"; then
        echo "  → Incompatible with Julia 1.11 — removing."
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
echo "  sbatch submit_blp_1_draws.sh                # pre-compute the nu + demographic draws"
echo "  bash   submit_blp_2_rc_default.sh           # RC-BLP sweep, submitted as dependency chains"
echo "  bash   submit_blp_build_and_run_spec12.sh   # same sweep, with optional --sysimage / --draws"
