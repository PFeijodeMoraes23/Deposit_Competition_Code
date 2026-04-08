#!/bin/bash
#SBATCH --job-name=blp_estim_loop_6            # Job name
#SBATCH --output=blp_hpc_6_output_%j.log       # Standard output log (%j = job id)
#SBATCH --error=blp_hpc_6_error_%j.log         # Standard error log
#SBATCH --time=48:00:00                         # Wall time limit. Max for 'week' partition is 7d.
#SBATCH --partition=week                        # Partition (queue) name
#SBATCH --nodes=1                               # Single node
#SBATCH --ntasks=1                              # Single task
#SBATCH --cpus-per-task=12                      # 12 CPU cores; 8 workers + 4 overhead
#SBATCH --mem=750G                              # Raised from 500G: each worker allocates (N_B,R)
#                                               # and (N_D,R) float64 matrices (~3GB each at R=1000)
#SBATCH --mail-type=ALL                         # SLURM emails on BEGIN, END, FAIL (guaranteed delivery)
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

module load miniconda
source activate dep_comp_blp
cd /home/pf382/dep_comp/scripts
python -u estimation_6_demand_2_loop.py \
    --spec all \
    --alt all \
    --stage sequence \
    --R 1000 \
    --workers 8 \
    --max-inner 1500 \
    --tol-inner 1e-12 \
    --hpc
