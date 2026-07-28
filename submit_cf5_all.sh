#!/bin/bash
# submit_cf5_all.sh — ONE-COMMAND full-panel CF5 monetary pass-through on Bouchet.
#
# CF5 differences a BASE equilibrium against a Selic-SHOCKED one, each of which is the CF3
# full-panel fixed point (firm-sharded Jacobi — D-firm shares are national, so a market subset is
# not a valid headline). This orchestrator:
#   1. solves the BASE equilibrium   → submit_cf3_jacobi.sh (SIGMA_DIR=…cf5_base,  no shock)
#   2. solves the SHOCKED equilibrium→ submit_cf3_jacobi.sh (SIGMA_DIR=…cf5_shock, --selic-shock Δ)
#      (both run concurrently; each is its own warmup→init→sweeps chain)
#   3. submits cf5_compare gated on afterok BOTH final merges — it reads the two solved σ vectors
#      and reports ∂ρ*/∂Selic and ∂Dep/∂Selic → CF_FOUNDATION/cf5_passthrough_*.parquet
#   4. (optional) auto-zips the cf5 outputs afterany the compare (DO_ZIP=1).
#
# Prerequisite: the RC costs (cost_params_E{k}_spec_12_extended.json) — run the BBL stage first:
#     bash submit_bbl_all.sh
#
# Usage (login node, code dir):
#     CF_ROUTINE=6 bash submit_cf5_all.sh                       # E6 headline, Selic +100bp
#     CF_ROUTINE=6 SELIC_SHOCK=0.005 bash submit_cf5_all.sh     # +50bp
#   Tunables (env, forwarded to submit_cf3_jacobi.sh):
#     CF_ROUTINE=6  CF_STAGE=extended  R=2000  SEED=42
#     SELIC_SHOCK=0.01        annual Selic shock (100bp); enters the shocked leg's r^f
#     N_FIRM_SHARDS=40  N_SWEEPS=6   Jacobi width/depth (per equilibrium)
#     BETA=0.9  HORIZON=50  BR_GRID=7  BR_WINDOW=0.02   (inherited by the Jacobi)
#     GPU_PARTITION=gpu_h200  GPUS=h200:1  CPU_PARTITION=day  (shard vs init/merge partitions)
#     COMPARE_TIME=00:30:00   MEM=…      DO_ZIP=1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CF_ROUTINE="${CF_ROUTINE:-6}"
export CF_STAGE="${CF_STAGE:-extended}"
export R="${R:-2000}"
export SEED="${SEED:-42}"
export N_FIRM_SHARDS="${N_FIRM_SHARDS:-40}"
export N_SWEEPS="${N_SWEEPS:-6}"
SELIC_SHOCK="${SELIC_SHOCK:-0.01}"
DO_ZIP="${DO_ZIP:-1}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/../data}"
LOGDIR="logs"; mkdir -p "${LOGDIR}"

# Fail fast on the shared prerequisite (submit_cf3_jacobi.sh checks it too, but do it once up front
# so we never submit a half-chain).
COST="${DATA_ROOT}/COST_FWD/cost_params_E${CF_ROUTINE}_spec_12_${CF_STAGE}.json"
[[ -f "${COST}" ]] || { echo "MISSING costs: ${COST}"; \
    echo "  → run the BBL cost stage first:  bash submit_bbl_all.sh"; exit 1; }

BASE_DIR="${DATA_ROOT}/CF_FOUNDATION/cf5_base_E${CF_ROUTINE}_${CF_STAGE}"
SCN_DIR="${DATA_ROOT}/CF_FOUNDATION/cf5_shock_E${CF_ROUTINE}_${CF_STAGE}"

echo "======================================================================"
echo " CF5 pass-through | E${CF_ROUTINE} ${CF_STAGE} | Selic +${SELIC_SHOCK} | R=${R}"
echo " ${N_FIRM_SHARDS} shards × ${N_SWEEPS} sweeps per equilibrium (base + shocked)"
echo "======================================================================"

# Capture a Jacobi run's FINAL_JOB / FINAL_SIGMA from its stdout (it prints both for chaining).
# ZIP_AFTER="" suppresses the Jacobi's own cf3 auto-zip (this is a CF5 sub-solve, not a CF3 result).
run_jacobi () {  # run_jacobi <SIGMA_DIR> <EXTRA_CF>  →  echoes "<final_job> <final_sigma>"
    local sdir="$1" extra="$2" out job sig
    out="$(SIGMA_DIR="${sdir}" ZIP_AFTER="" EXTRA_CF="${extra}" bash "${HERE}/submit_cf3_jacobi.sh")"
    echo "${out}" >&2                                    # surface the full Jacobi log
    job="$(printf '%s\n' "${out}" | sed -n 's/^FINAL_JOB=//p'   | tail -1)"
    sig="$(printf '%s\n' "${out}" | sed -n 's/^FINAL_SIGMA=//p' | tail -1)"
    [[ -n "${job}" && -n "${sig}" ]] || { echo "ERROR: no FINAL_JOB/FINAL_SIGMA from Jacobi (${sdir})" >&2; exit 1; }
    printf '%s %s\n' "${job}" "${sig}"
}

echo "── (1/3) BASE equilibrium → ${BASE_DIR} ──"
base_res="$(run_jacobi "${BASE_DIR}" "")"      # $() so a failed sub-solve trips set -e (unlike < <(…))
read -r base_job base_sig <<< "${base_res}"
echo "   base:    job=${base_job}  sigma=${base_sig}"

echo "── (2/3) SHOCKED equilibrium (Selic +${SELIC_SHOCK}) → ${SCN_DIR} ──"
scn_res="$(run_jacobi "${SCN_DIR}" "--selic-shock ${SELIC_SHOCK}")"
read -r scn_job scn_sig <<< "${scn_res}"
echo "   shocked: job=${scn_job}  sigma=${scn_sig}"

# ── (3/3) cf5_compare — difference the two solved equilibria, afterok BOTH ─────────
echo "── (3/3) cf5_compare (afterok:${base_job}:${scn_job}) ──"
cmp_job=$(sbatch --parsable -J "cf5cmp_E${CF_ROUTINE}" -t "${COMPARE_TIME:-00:30:00}" \
    --kill-on-invalid-dep=yes --dependency=afterok:"${base_job}":"${scn_job}" \
    ${PARTITION:+--partition="${PARTITION}"} ${MEM:+--mem="${MEM}"} \
    -o "${LOGDIR}/cf5cmp_E${CF_ROUTINE}_%j.out" -e "${LOGDIR}/cf5cmp_E${CF_ROUTINE}_%j.err" \
    --export=ALL,CF_ROUTINE="${CF_ROUTINE}",CF_STAGE="${CF_STAGE}",R="${R}",SEED="${SEED}",CF_STEP=cf5_compare,SIGMA_BASE="${base_sig}",SIGMA_SCN="${scn_sig}",CF_EXTRA="--selic-shock ${SELIC_SHOCK}" \
    "${HERE}/submit_cf.sh")
echo "   cf5_compare → ${cmp_job}"

# Optional: archive the cf5 outputs once the compare lands (afterany, so a partial still zips).
if [[ "${DO_ZIP}" == "1" ]]; then
    zj=$(sbatch --parsable -J "cf_zip_cf5_E${CF_ROUTINE}" -t 00:30:00 --dependency=afterany:"${cmp_job}" \
        ${CPU_PARTITION:+--partition="${CPU_PARTITION}"} --mem=4G \
        -o "${LOGDIR}/cf_zip_cf5_E${CF_ROUTINE}_%j.out" -e "${LOGDIR}/cf_zip_cf5_E${CF_ROUTINE}_%j.err" \
        --export=ALL,CF_STEP=zip,CF_WHICH=cf5,CF_STAGE="${CF_STAGE}",DATA_ROOT="${DATA_ROOT}" \
        "${HERE}/submit_cf.sh")
    echo "   zip cf5 → ${zj} (afterany:${cmp_job})"
fi

echo "----------------------------------------------------------------------"
echo "Submitted CF5 pass-through: base=${base_job} shock=${scn_job} compare=${cmp_job}"
echo "Result: CF_FOUNDATION/cf5_passthrough_*.parquet + console ∂ρ*/∂Selic, ∂Dep/∂Selic (after ${cmp_job})"
echo "Watch: squeue -u \$USER"
