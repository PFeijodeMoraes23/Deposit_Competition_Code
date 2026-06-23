#!/bin/bash
# submit_cf_all.sh — orchestrator for the counterfactual pipeline on Bouchet.
#
# Splits work across MULTIPLE jobs so no single job hits a time wall, and chains the
# Eq-18 solve as an afterok dependency on the CF2 deviation shards. Submits:
#   (optional) demand_eval  — 0a in-sample share reproduction        [1 job]
#   (optional) cf1          — gross franchise-value decomposition    [1 job]
#   cost2 ARRAY             — CF2 ψ deviations, N_SHARDS tasks        [array job]
#   cost_solve              — Eq-18 minimization, afterok the array  [1 job]
#
# Each cost2 array task computes SHOCKS/N_SHARDS deviations (reproducible per-shock
# RNG → shards merge exactly), keeping each job short. The solver globs all shards.
#
# Usage:
#   sbatch-less driver — run on the login node, it calls sbatch for you:
#     bash submit_cf_all.sh
#   Tunables (env):
#     CF_ROUTINE=6  CF_STAGE=extended  R=2000  SEED=42
#     SHOCKS=50  N_SHARDS=10  PERTURB_SCALE=0.05  BETA=0.9  HORIZON=50
#     DO_DEMAND_EVAL=1  DO_CF1=1            (set to 0 to skip those legs)
#     SHARD_TIME=08:00:00  SOLVE_TIME=01:00:00
set -euo pipefail

CF_ROUTINE="${CF_ROUTINE:-6}"
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
SHOCKS="${SHOCKS:-50}"; N_SHARDS="${N_SHARDS:-10}"
PERTURB_SCALE="${PERTURB_SCALE:-0.05}"; BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}"
DO_DEMAND_EVAL="${DO_DEMAND_EVAL:-1}"; DO_CF1="${DO_CF1:-1}"
SHARD_TIME="${SHARD_TIME:-08:00:00}"; SOLVE_TIME="${SOLVE_TIME:-01:00:00}"
LOGDIR="logs"; mkdir -p "${LOGDIR}"

base_export="CF_ROUTINE=${CF_ROUTINE},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED}"
cf2_extra="--shocks ${SHOCKS} --perturb-scale ${PERTURB_SCALE} --beta ${BETA} --horizon ${HORIZON}"
cf1_extra="--beta ${BETA} --horizon ${HORIZON}"

submit () {  # submit <jobname> <time> <extra-sbatch-args...> -- <ENV exports>
    local name="$1" tlim="$2"; shift 2
    sbatch --parsable -J "${name}" -t "${tlim}" \
        -o "${LOGDIR}/${name}_%A_%a.out" -e "${LOGDIR}/${name}_%A_%a.err" "$@"
}

echo "Orchestrating CF pipeline: E${CF_ROUTINE} ${CF_STAGE} | R=${R} | shocks=${SHOCKS} over ${N_SHARDS} shards"

if [[ "${DO_DEMAND_EVAL}" == "1" ]]; then
    j=$(submit cf_demaneval "${SOLVE_TIME}" \
        --export=ALL,${base_export},CF_STEP=demand_eval submit_cf.sh)
    echo "  demand_eval  → job ${j}"
fi

if [[ "${DO_CF1}" == "1" ]]; then
    j=$(submit cf_cf1 "${SHARD_TIME}" \
        --export=ALL,${base_export},CF_STEP=cf1,CF_EXTRA="${cf1_extra}" submit_cf.sh)
    echo "  cf1          → job ${j}"
fi

# CF2 deviation shards as a job ARRAY (0 … N_SHARDS-1).
arr=$(submit cf_cost2 "${SHARD_TIME}" --array=0-$((N_SHARDS-1)) \
    --export=ALL,${base_export},CF_STEP=cost2,N_SHARDS=${N_SHARDS},CF_EXTRA="${cf2_extra}" \
    submit_cf.sh)
echo "  cost2 array  → job ${arr} (${N_SHARDS} shards)"

# Eq-18 solve: runs only after EVERY shard finishes OK.
slv=$(submit cf_solve "${SOLVE_TIME}" --dependency=afterok:"${arr}" \
    --export=ALL,${base_export},CF_STEP=cost_solve,CF_EXTRA="--bootstrap 200" submit_cf.sh)
echo "  cost_solve   → job ${slv} (afterok:${arr})"

echo "Submitted. Watch with: squeue -u \$USER"
