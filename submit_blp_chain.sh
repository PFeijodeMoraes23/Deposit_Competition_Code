#!/bin/bash
# submit_blp_chain.sh
# ===================
# Submit BLP sigma → rc2 → rc3 → rc4 → full → ext1 → ext2 → extended as eight dependent SLURM jobs.
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
# All six jobs are submitted immediately. Each sits in PENDING state until the
# preceding job exits with code 0 (--dependency=afterok).
# Cancel the chain at any time with: scancel <JID_SIGMA> <JID_RC2> ... <JID_EXT>

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
# Memory is set uniformly large for all stages to avoid OOM on the heavier
# Pi-interaction buffers. Time budgets scale with R and parameter count.
MEM_ALL="200G"
if [ "${R}" -le 500 ]; then
    T_SIGMA="02:00:00"
    T_RC2="02:00:00"
    T_RC3="02:00:00"
    T_RC4="03:00:00"
    T_FULL="04:00:00"
    T_EXT1="05:00:00"
    T_EXT2="06:00:00"
    T_EXT="08:00:00"
elif [ "${R}" -le 1000 ]; then
    T_SIGMA="04:00:00"
    T_RC2="04:00:00"
    T_RC3="05:00:00"
    T_RC4="06:00:00"
    T_FULL="08:00:00"
    T_EXT1="10:00:00"
    T_EXT2="12:00:00"
    T_EXT="16:00:00"
else
    T_SIGMA="06:00:00"
    T_RC2="06:00:00"
    T_RC3="08:00:00"
    T_RC4="10:00:00"
    T_FULL="12:00:00"
    T_EXT1="14:00:00"
    T_EXT2="16:00:00"
    T_EXT="24:00:00"
fi

echo "======================================================"
echo " BLP chain submit: E${ESTIM} | R=${R} | seed=${SEED} | mem=${MEM_ALL}"
echo "  sigma    : time=${T_SIGMA}"
echo "  rc2      : time=${T_RC2}"
echo "  rc3      : time=${T_RC3}"
echo "  rc4      : time=${T_RC4}"
echo "  full     : time=${T_FULL}"
echo "  ext1     : time=${T_EXT1}"
echo "  ext2     : time=${T_EXT2}"
echo "  extended : time=${T_EXT}"
echo "======================================================"

# ── Common Julia preamble (expanded at submission time) ──────────────────────
JULIA_SETUP="module reset
module load Julia/1.11.4-linux-x86_64
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
    --mem="${MEM_ALL}" \
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

# ── Stage: rc2 (σ_spread + π×gdp; θ₂ warm-start from sigma) ────────────────
JID_RC2=$(sbatch --parsable \
    --dependency=afterok:${JID_SIGMA} \
    --job-name="blp_E${ESTIM}_rc2_R${R}" \
    --partition=day \
    --time="${T_RC2}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --mem="${MEM_ALL}" \
    --output="${LOG_DIR}/blp_E${ESTIM}_rc2_R${R}_%j.out" \
    --error="${LOG_DIR}/blp_E${ESTIM}_rc2_R${R}_%j.err" \
    --mail-type=END,FAIL,TIME_LIMIT_90 \
    --mail-user=pedro.feijodemoraes@yale.edu \
    --wrap="${JULIA_SETUP}
echo '=== BLP E${ESTIM} rc2 | R=${R} | '\$(date)' ==='
julia --project=${PROJECT_DIR} --threads=${CPUS} \\
    ${PROJECT_DIR}/blp_estimation.jl \\
    --estim ${ESTIM} --stage rc2 ${BLP_COMMON_ARGS}
echo 'E${ESTIM} rc2 done: '\$(date)")

echo "  Submitted rc2:      Job ${JID_RC2} (after ${JID_SIGMA})"

# ── Stage: rc3 (+ π×frac65; θ₂ warm-start from rc2) ────────────────────────
JID_RC3=$(sbatch --parsable \
    --dependency=afterok:${JID_RC2} \
    --job-name="blp_E${ESTIM}_rc3_R${R}" \
    --partition=day \
    --time="${T_RC3}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --mem="${MEM_ALL}" \
    --output="${LOG_DIR}/blp_E${ESTIM}_rc3_R${R}_%j.out" \
    --error="${LOG_DIR}/blp_E${ESTIM}_rc3_R${R}_%j.err" \
    --mail-type=END,FAIL,TIME_LIMIT_90 \
    --mail-user=pedro.feijodemoraes@yale.edu \
    --wrap="${JULIA_SETUP}
echo '=== BLP E${ESTIM} rc3 | R=${R} | '\$(date)' ==='
julia --project=${PROJECT_DIR} --threads=${CPUS} \\
    ${PROJECT_DIR}/blp_estimation.jl \\
    --estim ${ESTIM} --stage rc3 ${BLP_COMMON_ARGS}
echo 'E${ESTIM} rc3 done: '\$(date)")

echo "  Submitted rc3:      Job ${JID_RC3} (after ${JID_RC2})"

# ── Stage: rc4 (+ π×conn100; θ₂ warm-start from rc3) ───────────────────────
JID_RC4=$(sbatch --parsable \
    --dependency=afterok:${JID_RC3} \
    --job-name="blp_E${ESTIM}_rc4_R${R}" \
    --partition=day \
    --time="${T_RC4}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --mem="${MEM_ALL}" \
    --output="${LOG_DIR}/blp_E${ESTIM}_rc4_R${R}_%j.out" \
    --error="${LOG_DIR}/blp_E${ESTIM}_rc4_R${R}_%j.err" \
    --mail-type=END,FAIL,TIME_LIMIT_90 \
    --mail-user=pedro.feijodemoraes@yale.edu \
    --wrap="${JULIA_SETUP}
echo '=== BLP E${ESTIM} rc4 | R=${R} | '\$(date)' ==='
julia --project=${PROJECT_DIR} --threads=${CPUS} \\
    ${PROJECT_DIR}/blp_estimation.jl \\
    --estim ${ESTIM} --stage rc4 ${BLP_COMMON_ARGS}
echo 'E${ESTIM} rc4 done: '\$(date)")

echo "  Submitted rc4:      Job ${JID_RC4} (after ${JID_RC3})"

# ── Stage: full (+ σ_log_assets; θ₂ warm-start from rc4) ────────────────────
JID_FULL=$(sbatch --parsable \
    --dependency=afterok:${JID_RC4} \
    --job-name="blp_E${ESTIM}_full_R${R}" \
    --partition=day \
    --time="${T_FULL}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --mem="${MEM_ALL}" \
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

echo "  Submitted full:     Job ${JID_FULL} (after ${JID_RC4})"

# ── Stage: ext1 (+ π(log_assets×gdp); θ₂ warm-start from full) ──────────────
JID_EXT1=$(sbatch --parsable \
    --dependency=afterok:${JID_FULL} \
    --job-name="blp_E${ESTIM}_ext1_R${R}" \
    --partition=day \
    --time="${T_EXT1}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --mem="${MEM_ALL}" \
    --output="${LOG_DIR}/blp_E${ESTIM}_ext1_R${R}_%j.out" \
    --error="${LOG_DIR}/blp_E${ESTIM}_ext1_R${R}_%j.err" \
    --mail-type=END,FAIL,TIME_LIMIT_90 \
    --mail-user=pedro.feijodemoraes@yale.edu \
    --wrap="${JULIA_SETUP}
echo '=== BLP E${ESTIM} ext1 | R=${R} | '\$(date)' ==='
julia --project=${PROJECT_DIR} --threads=${CPUS} \\
    ${PROJECT_DIR}/blp_estimation.jl \\
    --estim ${ESTIM} --stage ext1 ${BLP_COMMON_ARGS}
echo 'E${ESTIM} ext1 done: '\$(date)")

echo "  Submitted ext1:     Job ${JID_EXT1} (after ${JID_FULL})"

# ── Stage: ext2 (+ π(fgc_covered×frac65); θ₂ warm-start from ext1) ──────────
JID_EXT2=$(sbatch --parsable \
    --dependency=afterok:${JID_EXT1} \
    --job-name="blp_E${ESTIM}_ext2_R${R}" \
    --partition=day \
    --time="${T_EXT2}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --mem="${MEM_ALL}" \
    --output="${LOG_DIR}/blp_E${ESTIM}_ext2_R${R}_%j.out" \
    --error="${LOG_DIR}/blp_E${ESTIM}_ext2_R${R}_%j.err" \
    --mail-type=END,FAIL,TIME_LIMIT_90 \
    --mail-user=pedro.feijodemoraes@yale.edu \
    --wrap="${JULIA_SETUP}
echo '=== BLP E${ESTIM} ext2 | R=${R} | '\$(date)' ==='
julia --project=${PROJECT_DIR} --threads=${CPUS} \\
    ${PROJECT_DIR}/blp_estimation.jl \\
    --estim ${ESTIM} --stage ext2 ${BLP_COMMON_ARGS}
echo 'E${ESTIM} ext2 done: '\$(date)")

echo "  Submitted ext2:     Job ${JID_EXT2} (after ${JID_EXT1})"

# ── Stage: extended (+ π(equity×cadunico); θ₂ warm-start from ext2) ─────────
JID_EXT=$(sbatch --parsable \
    --dependency=afterok:${JID_EXT2} \
    --job-name="blp_E${ESTIM}_extended_R${R}" \
    --partition=day \
    --time="${T_EXT}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --mem="${MEM_ALL}" \
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

echo "  Submitted extended: Job ${JID_EXT} (after ${JID_EXT2})"
echo ""
echo "  Chain: ${JID_SIGMA} → ${JID_RC2} → ${JID_RC3} → ${JID_RC4} → ${JID_FULL} → ${JID_EXT1} → ${JID_EXT2} → ${JID_EXT}"
echo "  Monitor: squeue -j ${JID_SIGMA},${JID_RC2},${JID_RC3},${JID_RC4},${JID_FULL},${JID_EXT1},${JID_EXT2},${JID_EXT}"
echo "  Cancel:  scancel ${JID_SIGMA} ${JID_RC2} ${JID_RC3} ${JID_RC4} ${JID_FULL} ${JID_EXT1} ${JID_EXT2} ${JID_EXT}"
