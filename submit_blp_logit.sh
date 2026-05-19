#!/bin/bash
# submit_blp_logit.sh
# ====================
# Quick job: run the logit (θ₂=0) stage for E5 spec 12 to generate a
# version-compatible warm-start delta checkpoint (logit_delta_E5_spec_12.bin)
# before submitting the sigma job.
#
# Runtime: ~2–5 min  |  RAM: <4 GB  |  CPUs: 4
# Submit:  sbatch submit_blp_logit.sh
# Then:    sbatch submit_blp_E5.sh
#
#SBATCH --job-name=blp_logit
#SBATCH --partition=day
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_logit_%j.out
#SBATCH --error=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_logit_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

# ── Environment ───────────────────────────────────────────────────────────────
module reset
module load Julia/1.11.4-linux-x86_64

export JULIA_MKL_THREADING=tbb
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# ── Verify environment (run setup_julia_env.sh once before first submission) ──
echo "Checking Julia packages: $(date)"
julia --project="${PROJECT_DIR}" -e "using Pkg; Pkg.instantiate()"

echo "======================================"
echo " BLP Logit warm-start — E5 spec 12"
echo " $(date)"
echo " Saves: logit_delta_E5_spec_12.{jls,bin}"
echo "======================================"

julia --project="${PROJECT_DIR}" --threads=${SLURM_CPUS_PER_TASK} \
    "${PROJECT_DIR}/blp_estimation.jl" \
    --estim 5 --spec 12 --stage logit \
    --R 2000 --seed 42 \
    --hpc

echo "Logit complete — warm-start checkpoints saved: $(date)"
echo "Now submit the sigma job: sbatch submit_blp_E5.sh"
