#!/bin/bash
# submit_cf3_jacobi.sh — full-panel CF3 equilibrium spreads on Bouchet via FIRM-SHARDED JACOBI.
#
# The best-response re-simulates the full national panel per evaluation (D-firm shares are
# national), so a single-process solve is intractable at R=2000. Instead each Jacobi SWEEP is an
# array of shard jobs — shard s best-responds firms {j : j mod N_FIRM_SHARDS == s} against the
# FROZEN previous-sweep σ — followed by a merge into the next σ. Sweeps chain via afterok. This
# uses the exact (correct) full-panel psi_under; it just parallelizes the firms.
#
# Prerequisite: the RC costs (cost_params_E{k}_spec_12_extended.json) — i.e. run submit_cf_all.sh
# (through cost_solve) first.
#
# Usage (login node, code dir):
#   CF_ROUTINE=6 N_FIRM_SHARDS=40 N_SWEEPS=6 bash submit_cf3_jacobi.sh
# Tunables (env): CF_ROUTINE CF_STAGE=extended R=2000 SEED=42 N_FIRM_SHARDS N_SWEEPS
#                 BR_GRID=7 BR_WINDOW=0.02 BETA=0.9 HORIZON=50 SHARD_TIME MERGE_TIME
set -euo pipefail

CF_ROUTINE="${CF_ROUTINE:-6}"; CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
N_FIRM_SHARDS="${N_FIRM_SHARDS:-40}"; N_SWEEPS="${N_SWEEPS:-6}"
BR_GRID="${BR_GRID:-7}"; BR_WINDOW="${BR_WINDOW:-0.02}"; BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}"
SHARD_TIME="${SHARD_TIME:-06:00:00}"; MERGE_TIME="${MERGE_TIME:-00:30:00}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/../data}"
SIGMA_DIR="${DATA_ROOT}/CF_FOUNDATION/cf3_jacobi_E${CF_ROUTINE}_${CF_STAGE}"
LOGDIR="logs"; mkdir -p "${LOGDIR}" "${SIGMA_DIR}"

COST="${DATA_ROOT}/COST_FWD/cost_params_E${CF_ROUTINE}_spec_12_${CF_STAGE}.json"
[[ -f "${COST}" ]] || { echo "MISSING costs: ${COST}"; echo "  → run submit_cf_all.sh through cost_solve first."; exit 1; }
echo "CF3 firm-sharded Jacobi: E${CF_ROUTINE} ${CF_STAGE} | ${N_FIRM_SHARDS} shards × ${N_SWEEPS} sweeps → ${SIGMA_DIR}"

base="CF_ROUTINE=${CF_ROUTINE},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED},SIGMA_DIR=${SIGMA_DIR},N_FIRM_SHARDS=${N_FIRM_SHARDS}"
eq="--beta ${BETA} --horizon ${HORIZON} --br-grid ${BR_GRID} --br-window ${BR_WINDOW}"
sub () { local n="$1" t="$2"; shift 2
    sbatch --parsable -J "${n}" -t "${t}" -o "${LOGDIR}/${n}_%A_%a.out" -e "${LOGDIR}/${n}_%A_%a.err" "$@"; }

j=$(sub cf3init_E${CF_ROUTINE} "${MERGE_TIME}" --export=ALL,${base},CF_STEP=cf3_init,CF_EXTRA="${eq}" submit_cf.sh)
echo "  init σ⁰ → ${j}"
prev="${j}"
for s in $(seq 1 "${N_SWEEPS}"); do
    arr=$(sub "cf3sw${s}_E${CF_ROUTINE}" "${SHARD_TIME}" --array=0-$((N_FIRM_SHARDS-1)) \
        --dependency=afterok:"${prev}" \
        --export=ALL,${base},SWEEP=${s},CF_STEP=cf3_shard,CF_EXTRA="${eq}" submit_cf.sh)
    mrg=$(sub "cf3mg${s}_E${CF_ROUTINE}" "${MERGE_TIME}" --dependency=afterok:"${arr}" \
        --export=ALL,${base},SWEEP=${s},CF_STEP=cf3_merge,CF_EXTRA="${eq}" submit_cf.sh)
    echo "  sweep ${s}: shard array ${arr} → merge ${mrg}"
    prev="${mrg}"
done
echo "Final equilibrium σ → ${SIGMA_DIR}/sig_${N_SWEEPS}.parquet (‖Δσ‖∞ per sweep in the cf3mg*.out logs)."
echo "Watch: squeue -u \$USER"
