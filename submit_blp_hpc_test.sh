#!/bin/bash
#SBATCH --job-name=blp_test
#SBATCH --output=blp_test_output_%j.log
#SBATCH --error=blp_test_error_%j.log
#SBATCH --time=04:00:00              # 4 hours is plenty for R=100
#SBATCH --partition=day              # 'day' queues faster for small tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # Matches --workers 4 + overhead
#SBATCH --mem=120G                   # R=100 uses ~1/10th the RAM of R=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

module load miniconda
conda activate dep_comp_blp
cd /home/pf382/dep_comp/scripts

python -u estimation_1_demand_3_loop.py \
    --spec 1 \
    --stage sigma \
    --R 100 \
    --workers 4 \
    --hpc
