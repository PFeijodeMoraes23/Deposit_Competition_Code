#!/bin/bash
#SBATCH --job-name=blp_E3
#SBATCH --partition=day
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=/home/pf382/dep_comp/scripts/logs/blp_E3_%j.out
#SBATCH --error=/home/pf382/dep_comp/scripts/logs/blp_E3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

module purge
module load Julia/1.10.4-linux-x86_64

PROJECT_DIR=/home/pf382/dep_comp/scripts
mkdir -p /home/pf382/dep_comp/scripts/logs

echo "======================================"
echo " BLP Estimation E3 — $(date)"
echo " stage=sequence | R=2000 | threads=${SLURM_CPUS_PER_TASK}"
echo "======================================"

julia --project="${PROJECT_DIR}" --threads=${SLURM_CPUS_PER_TASK} \
    "${PROJECT_DIR}/blp_estimation.jl" \
    --estim 3 --spec 12 --stage sequence \
    --R 2000 --seed 42 \
    --tol-inner 1e-12 --max-inner 5000 --tol-outer 1e-6 \
    --hpc

echo "E3 complete: $(date)"
