#!/bin/bash
#SBATCH --job-name=blp_julia_2                  # Job name
#SBATCH --output=blp_julia_2_output_%j.log      # Standard output log (%j = job id)
#SBATCH --error=blp_julia_2_error_%j.log        # Standard error log
#SBATCH --time=24:00:00                          # Wall time limit (day partition max)
#SBATCH --partition=day                          # Bouchet day partition — faster queue than week
#SBATCH --nodes=1                                # Single node
#SBATCH --ntasks=1                               # Single task
#SBATCH --cpus-per-task=32                       # 32 cores for BLAS + Julia threads
#SBATCH --mem=750G                               # 750 GB for pre-allocated HotBuffers + Pi products
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

# ── Environment ──────────────────────────────────────────────────────────────
module load Julia

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SCRIPTS=/home/pf382/dep_comp/scripts
PROJECT_DIR=${SCRIPTS}

# Instantiate Julia project (no-op if already done)
julia --project=${PROJECT_DIR} -e 'using Pkg; Pkg.instantiate()'

# ── Run ──────────────────────────────────────────────────────────────────────
julia --project=${PROJECT_DIR} \
      --threads=${SLURM_CPUS_PER_TASK} \
      ${SCRIPTS}/estimation_2_demand_2_loop_ju_hpc.jl \
    --spec 12 \
    --stage sequence \
    --R 2000 \
    --seed 42 \
    --tol-inner 1e-12 \
    --max-inner 2000 \
    --tol-outer 1e-6 \
    --hpc
