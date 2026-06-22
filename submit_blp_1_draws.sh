#!/bin/bash
#SBATCH --job-name=blp_1_draws
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_1_draws_%j.out
#SBATCH --error=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_1_draws_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

# ── Environment ──────────────────────────────────────────────────────────────
# module reset is the correct command on Bouchet (module purge cannot unload StdEnv)
module reset
module load Julia/1.11.4-linux-x86_64

# Read the post-fix coherence reference panel (demand_1_spec_12.parquet) for market
# keys, matching the coherence RC-BLP estimation inputs. Draws are routine-independent.
export BLP_COHERENCE_INPUTS=1

# Strict error checking starts after module loading to avoid false failures
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ── Resolve packages for Julia 1.10 ─────────────────────────────────────────
if [ -f "${PROJECT_DIR}/Manifest.toml" ] && ! grep -q 'julia_version = "1.10' "${PROJECT_DIR}/Manifest.toml"; then
    echo "Removing incompatible Manifest.toml before instantiate: $(date)"
    rm -f "${PROJECT_DIR}/Manifest.toml"
fi
echo "Resolving Julia packages: $(date)"
julia --project="${PROJECT_DIR}" -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"

echo "======================================"
echo " BLP Draws — blp_1_draws.jl"
echo " R=2000 | seed=42 | $(date)"
echo "======================================"

julia --project="${PROJECT_DIR}" --threads=${SLURM_CPUS_PER_TASK} \
    "${PROJECT_DIR}/blp_1_draws.jl" \
    --R 2000 \
    --seed 42 \
    --spec 12 \
    --estim 1 \
    --hpc

echo "Draws complete: $(date)"
