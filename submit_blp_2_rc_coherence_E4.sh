#!/bin/bash
#SBATCH --job-name=blp2rc_coh_E4
#SBATCH --partition=gpu_h200
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --gpus=h200:1
#SBATCH --output=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp2rc_coh_E4_%j.out
#SBATCH --error=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp2rc_coh_E4_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

# Coherence RC-BLP — routine E4 (Pooled Constrained Linear), spec 12.
# Reads demand_4_constrained_spec_12.parquet; warm-starts from
# logit_delta_E4_spec_12_coherence.bin; writes *_coherence-suffixed outputs.

# ── Environment ───────────────────────────────────────────────────────────────
module reset
# Pin Julia 1.11.4 — ships LLVM 16.0.6 required to emit PTX for sm_90 (H100/H200).
module load Julia/1.11.4-linux-x86_64
# CUDA.jl ships its own toolkit via JLL artifacts — do NOT load a system CUDA module.
export JULIA_MKL_THREADING=tbb
# Keep Julia depots on the project filesystem so the CUDA artifact doesn't blow $HOME.
export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/.julia_depot:${JULIA_DEPOT_PATH:-}"
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

echo "GPU node: $(hostname)"
echo "CUDA devices: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── Verify environment ────────────────────────────────────────────────────────
# Sysimage-aware Julia setup (sets JULIA_SYS; skips precompile when blp_sysimage.so exists).
source "${PROJECT_DIR}/julia_sysimage_env.sh"

echo "======================================"
echo " Coherence RC-BLP E4 (Pooled Constrained Linear) — $(date)"
echo " spec=12 | stage=sequence | R=2000 | threads=${SLURM_CPUS_PER_TASK}"
echo "======================================"

julia --project="${PROJECT_DIR}" ${JULIA_SYS[@]+"${JULIA_SYS[@]}"} --threads=${SLURM_CPUS_PER_TASK} \
    "${PROJECT_DIR}/blp_2_rc_e4_coherence.jl" \
    --hpc --R 2000 --seed 42 --stage sequence \
    --tol-inner 1e-10 --max-inner 5000 --tol-outer 1e-6

echo "Coherence RC-BLP E4 complete: $(date)"
