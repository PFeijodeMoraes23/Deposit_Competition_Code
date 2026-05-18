#!/bin/bash
# DEPRECATED — DEFUNCT. USE submit_blp_E5.sh INSTEAD.
#SBATCH --job-name=blp_E4
#SBATCH --partition=day
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --output=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_E4_%j.out
#SBATCH --error=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_E4_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

# ── Environment ──────────────────────────────────────────────────────────────
# module reset is the correct command on Bouchet (module purge cannot unload StdEnv)
module reset
module load Julia/1.10.4-linux-x86_64

# Strict error checking starts after module loading to avoid false failures
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# ── Resolve packages for Julia 1.10 ─────────────────────────────────────────
if [ -f "${PROJECT_DIR}/Manifest.toml" ] && ! grep -q 'julia_version = "1.10' "${PROJECT_DIR}/Manifest.toml"; then
    echo "Removing incompatible Manifest.toml before instantiate: $(date)"
    rm -f "${PROJECT_DIR}/Manifest.toml"
fi
echo "Resolving Julia packages: $(date)"
julia --project="${PROJECT_DIR}" -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"

echo "======================================"
echo " BLP Estimation E4 sigma — $(date)"
echo " stage=sigma | R=2000 | threads=${SLURM_CPUS_PER_TASK}"
echo " (logit delta warm-start loaded if logit_delta_E4_spec_12.jls present)"
echo "======================================"

julia --project="${PROJECT_DIR}" --threads=${SLURM_CPUS_PER_TASK} \
    "${PROJECT_DIR}/blp_estimation.jl" \
    --estim 4 --spec 12 --stage sigma \
    --R 2000 --seed 42 \
    --tol-inner 1e-12 --max-inner 5000 --tol-outer 1e-6 \
    --hpc

echo "E4 complete: $(date)"
