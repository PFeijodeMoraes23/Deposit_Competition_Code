#!/bin/bash
#SBATCH --job-name=blp_E5
#SBATCH --output=blp_E5_output_%j.log
#SBATCH --error=blp_E5_error_%j.log
#SBATCH --time=24:00:00
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=750G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

# ── Environment ──────────────────────────────────────────────────────────────
module load Julia/1.12.5

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SCRIPTS=/home/pf382/dep_comp/scripts
PROJECT_DIR=${SCRIPTS}

julia --project=${PROJECT_DIR} -e 'using Pkg; Pkg.instantiate()'

# ── Run ──────────────────────────────────────────────────────────────────────
julia --project=${PROJECT_DIR} \
      --threads=${SLURM_CPUS_PER_TASK} \
      ${SCRIPTS}/blp_loop.jl \
    --estim 5 \
    --spec 12 \
    --stage sequence \
    --R 2000 \
    --seed 42 \
    --tol-inner 1e-12 \
    --max-inner 5000 \
    --tol-outer 1e-6 \
    --hpc
