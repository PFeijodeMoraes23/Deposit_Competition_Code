#!/bin/bash
# submit_blp_chain.sh
# ===================
# Submit BLP sigma → full → extended as three dependent SLURM jobs.
# Each stage warm-starts θ₂ from the previous stage checkpoint (already
# implemented in blp_estimation.jl) and δ from the logit checkpoint.
#
# Usage:
#   bash submit_blp_chain.sh <ESTIM_ID> [R]
#
# Examples:
#   bash submit_blp_chain.sh 1 500    # E1 preliminary (R=500, ~2/4/8 h)
#   bash submit_blp_chain.sh 1 2000   # E1 production  (R=2000, ~6/12/24 h)
#   bash submit_blp_chain.sh 3        # E3 production  (default R=2000)
#
# Prerequisites (run locally FIRST, then rsync to cluster):
#   python run_blp_pipeline.py --logit          # generates logit_delta_E*.jls
#   rsync logit_delta_E*.jls <cluster>:../data/output/
#   sbatch submit_blp_draws.sh                  # generates R=2000 draws
#
# All three jobs are submitted immediately. The full and extended jobs sit in
# PENDING state and only start once the preceding job exits with code 0.
# Cancel the chain at any time with: scancel <JID_SIGMA> <JID_FULL> <JID_EXT>

set -euo pipefail

ESTIM="${1:?Usage: bash submit_blp_chain.sh <ESTIM_ID> [R]}"
R="${2:-2000}"
SEED=42
CPUS=32
SPEC=12
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ── Resource scaling by R ────────────────────────────────────────────────────
# Memory estimates: sigma≈N*R*8B, full adds 3 Pi buffers, extended adds 6.
# Padded 2× for Julia GC + BLAS workspace.
if [ "${R}" -le 500 ]; then
    T_SIGMA="02:00:00"; MEM_SIGMA="20G"
    T_FULL="04:00:00";  MEM_FULL="40G"
    T_EXT="08:00:00";   MEM_EXT="80G"
elif [ "${R}" -le 1000 ]; then
    T_SIGMA="04:00:00"; MEM_SIGMA="40G"
    T_FULL="08:00:00";  MEM_FULL="80G"
    T_EXT="16:00:00";   MEM_EXT="120G"
else
    T_SIGMA="06:00:00"; MEM_SIGMA="80G"
    T_FULL="12:00:00";  MEM_FULL="120G"
    T_EXT="24:00:00";   MEM_EXT="200G"
fi

echo "======================================================"
echo " BLP chain submit: E${ESTIM} | R=${R} | seed=${SEED}"
echo "  sigma    : time=${T_SIGMA}, mem=${MEM_SIGMA}"
echo "  full     : time=${T_FULL},  mem=${MEM_FULL}"
echo "  extended : time=${T_EXT},   mem=${MEM_EXT}"
echo "======================================================"

# ── Common Julia preamble (expanded at submission time) ──────────────────────
JULIA_SETUP="module reset
module load Julia/1.10.4-linux-x86_64
set -euo pipefail
mkdir -p ${LOG_DIR}
if [ -f ${PROJECT_DIR}/Manifest.toml ] && ! grep -q 'julia_version = \"1.10' ${PROJECT_DIR}/Manifest.toml; then
    echo 'Removing incompatible Manifest.toml before instantiate: ' \$(date)
    rm -f ${PROJECT_DIR}/Manifest.toml
fi
echo 'Resolving Julia packages: '\$(date)
julia --project=${PROJECT_DIR} -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'"

BLP_COMMON_ARGS="--spec ${SPEC} --R ${R} --seed ${SEED} \
    --tol-inner 1e-12 --max-inner 5000 --tol-outer 1e-6 --hpc"

# ── Stage: sigma ─────────────────────────────────────────────────────────────
JID_SIGMA=$(sbatch --parsable \
    --job-name="blp_E${ESTIM}_sigma_R${R}" \
    --partition=day \
    --time="${T_SIGMA}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --mem="${MEM_SIGMA}" \
    --output="${LOG_DIR}/blp_E${ESTIM}_sigma_R${R}_%j.out" \
    --error="${LOG_DIR}/blp_E${ESTIM}_sigma_R${R}_%j.err" \
    --mail-type=END,FAIL,TIME_LIMIT_90 \
    --mail-user=pedro.feijodemoraes@yale.edu \
    --wrap="${JULIA_SETUP}
echo '=== BLP E${ESTIM} sigma | R=${R} | '\$(date)' ==='
julia --project=${PROJECT_DIR} --threads=${CPUS} \\
    ${PROJECT_DIR}/blp_estimation.jl \\
    --estim ${ESTIM} --stage sigma ${BLP_COMMON_ARGS}
echo 'E${ESTIM} sigma done: '\$(date)")

echo "  Submitted sigma:    Job ${JID_SIGMA}"

# ── Stage: full (θ₂ warm-start from sigma checkpoint) ───────────────────────
JID_FULL=$(sbatch --parsable \
    --dependency=afterok:${JID_SIGMA} \
    --job-name="blp_E${ESTIM}_full_R${R}" \
    --partition=day \
    --time="${T_FULL}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --mem="${MEM_FULL}" \
    --output="${LOG_DIR}/blp_E${ESTIM}_full_R${R}_%j.out" \
    --error="${LOG_DIR}/blp_E${ESTIM}_full_R${R}_%j.err" \
    --mail-type=END,FAIL,TIME_LIMIT_90 \
    --mail-user=pedro.feijodemoraes@yale.edu \
    --wrap="${JULIA_SETUP}
echo '=== BLP E${ESTIM} full | R=${R} | '\$(date)' ==='
julia --project=${PROJECT_DIR} --threads=${CPUS} \\
    ${PROJECT_DIR}/blp_estimation.jl \\
    --estim ${ESTIM} --stage full ${BLP_COMMON_ARGS}
echo 'E${ESTIM} full done: '\$(date)")

echo "  Submitted full:     Job ${JID_FULL} (after ${JID_SIGMA})"

# ── Stage: extended (θ₂ warm-start from full checkpoint) ─────────────────────
JID_EXT=$(sbatch --parsable \
    --dependency=afterok:${JID_FULL} \
    --job-name="blp_E${ESTIM}_extended_R${R}" \
    --partition=day \
    --time="${T_EXT}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --mem="${MEM_EXT}" \
    --output="${LOG_DIR}/blp_E${ESTIM}_extended_R${R}_%j.out" \
    --error="${LOG_DIR}/blp_E${ESTIM}_extended_R${R}_%j.err" \
    --mail-type=END,FAIL,TIME_LIMIT_90 \
    --mail-user=pedro.feijodemoraes@yale.edu \
    --wrap="${JULIA_SETUP}
echo '=== BLP E${ESTIM} extended | R=${R} | '\$(date)' ==='
julia --project=${PROJECT_DIR} --threads=${CPUS} \\
    ${PROJECT_DIR}/blp_estimation.jl \\
    --estim ${ESTIM} --stage extended ${BLP_COMMON_ARGS}
echo 'E${ESTIM} extended done: '\$(date)")

echo "  Submitted extended: Job ${JID_EXT} (after ${JID_FULL})"
echo ""
echo "  Chain: ${JID_SIGMA} → ${JID_FULL} → ${JID_EXT}"
echo "  Monitor: squeue -j ${JID_SIGMA},${JID_FULL},${JID_EXT}"
echo "  Cancel:  scancel ${JID_SIGMA} ${JID_FULL} ${JID_EXT}"
