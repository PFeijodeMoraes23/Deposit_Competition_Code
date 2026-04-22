#!/bin/bash
#SBATCH --job-name=blp_draws
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/pf382/dep_comp/scripts/logs/blp_draws_%j.out
#SBATCH --error=/home/pf382/dep_comp/scripts/logs/blp_draws_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load Julia/1.10.4-linux-x86_64

PROJECT_DIR=/home/pf382/dep_comp/scripts
LOG_DIR=/home/pf382/dep_comp/scripts/logs
mkdir -p "$LOG_DIR"

echo "======================================"
echo " BLP Draws — blp_draws.jl"
echo " R=2000 | seed=42 | $(date)"
echo "======================================"

julia --project="${PROJECT_DIR}" --threads=${SLURM_CPUS_PER_TASK} \
    "${PROJECT_DIR}/blp_draws.jl" \
    --R 2000 \
    --seed 42 \
    --spec 12 \
    --estim 1 \
    --hpc

echo "Draws complete: $(date)"
