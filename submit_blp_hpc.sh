#!/bin/bash
#SBATCH --job-name=blp_estim_loop          # Job name
#SBATCH --output=blp_hpc_output_%j.log # Standard output log (%j ensures job id is added)
#SBATCH --error=blp_hpc_error_%j.log   # Standard error log
#SBATCH --time=24:00:00               # Wall time limit (hrs:min:sec). Max for 'day' partition is 24h.
#SBATCH --partition=day               # Partition (queue) name, e.g., 'day' or 'week'
#SBATCH --nodes=1                     # Run all processes on a single node 
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=12            # Number of CPU cores to allocate (script exploits multiprocessing up to core count)
#SBATCH --mem=500G                    # Massive memory envelope to support 12 simultaneous inner loops
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu # Add your email here to be notified

module load miniconda
conda activate dep_comp_blp
cd /home/pf382/dep_comp/scripts

python -u estimation_1_demand_3_loop.py --spec all --stage sequence --R 1000 --workers 12 --hpc