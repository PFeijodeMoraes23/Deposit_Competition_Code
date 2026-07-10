#!/bin/bash
# submit_cf5_passthrough.sh — CF5 monetary pass-through on Bouchet.
#
# Solves TWO full-panel equilibria via the firm-sharded Jacobi (GPU shards) — the BASE and a
# Selic-SHOCKED forward r^f — then a compare job reports ∂ρ*/∂Selic and ∂Dep/∂Selic. The two
# solves are independent (run in parallel); the compare is afterok both.
#
# Prereq: RC costs (run submit_cf_all.sh through cost_solve first).
# Usage:  CF_ROUTINE=7 SELIC_SHOCK=0.01 N_FIRM_SHARDS=40 N_SWEEPS=6 bash submit_cf5_passthrough.sh
# Tunables: everything submit_cf3_jacobi.sh takes, plus SELIC_SHOCK (annual, default 0.01) and
#   BASE_SIGMA=<path> to REUSE an already-solved base σ (e.g. the CF3 headline) instead of re-solving.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF_ROUTINE="${CF_ROUTINE:-6}"; CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"; BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}"
SELIC_SHOCK="${SELIC_SHOCK:-0.01}"; CPU_PARTITION="${CPU_PARTITION:-day}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/../data}"
ROOT="${DATA_ROOT}/CF_FOUNDATION/cf5_E${CF_ROUTINE}_${CF_STAGE}"
export CF_ROUTINE CF_STAGE R SEED BETA HORIZON DATA_ROOT   # inherited by submit_cf3_jacobi.sh
mkdir -p "${HERE}/logs"

dep=""   # afterok list for the compare
if [[ -n "${BASE_SIGMA:-}" ]]; then
    base_sig="${BASE_SIGMA}"; echo "== CF5 base: reusing ${base_sig} =="
else
    echo "== CF5 base equilibrium =="
    # ZIP_AFTER="" — the base/shock CF3 solves are internal to CF5; we zip the cf5 archive below.
    base_out=$(SIGMA_DIR="${ROOT}/base" EXTRA_CF="--selic-shock 0" ZIP_AFTER="" bash "${HERE}/submit_cf3_jacobi.sh"); echo "${base_out}"
    base_job=$(echo "${base_out}" | sed -n 's/^FINAL_JOB=//p'); base_sig=$(echo "${base_out}" | sed -n 's/^FINAL_SIGMA=//p')
    dep="${dep}:${base_job}"
fi
echo "== CF5 shocked equilibrium (Selic +${SELIC_SHOCK}) =="
shk_out=$(SIGMA_DIR="${ROOT}/shock" EXTRA_CF="--selic-shock ${SELIC_SHOCK}" ZIP_AFTER="" bash "${HERE}/submit_cf3_jacobi.sh"); echo "${shk_out}"
shk_job=$(echo "${shk_out}" | sed -n 's/^FINAL_JOB=//p'); shk_sig=$(echo "${shk_out}" | sed -n 's/^FINAL_SIGMA=//p')
dep="${dep}:${shk_job}"

echo "== CF5 compare (afterok${dep}) =="
cmp=$(sbatch --parsable -J "cf5cmp_E${CF_ROUTINE}" -t 02:00:00 --kill-on-invalid-dep=yes --partition="${CPU_PARTITION}" \
    --dependency=afterok"${dep}" \
    -o "${HERE}/logs/cf5cmp_E${CF_ROUTINE}_%j.out" -e "${HERE}/logs/cf5cmp_E${CF_ROUTINE}_%j.err" \
    --export=ALL,CF_ROUTINE=${CF_ROUTINE},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED},CF_STEP=cf5_compare,SIGMA_BASE="${base_sig}",SIGMA_SCN="${shk_sig}",CF_EXTRA="--selic-shock ${SELIC_SHOCK} --beta ${BETA} --horizon ${HORIZON}" \
    "${HERE}/submit_cf.sh")
echo "  cf5_compare → job ${cmp}"

# Auto-zip the cf5 archive (passthrough parquet + the base/shock σ dirs) after the compare.
if [[ "${DO_ZIP:-1}" == "1" ]]; then
    zj=$(sbatch --parsable -J "cf_zip_cf5_E${CF_ROUTINE}" -t 00:30:00 --kill-on-invalid-dep=yes \
        --partition="${CPU_PARTITION}" --mem=4G --dependency=afterany:"${cmp}" \
        -o "${HERE}/logs/cf_zip_cf5_E${CF_ROUTINE}_%j.out" -e "${HERE}/logs/cf_zip_cf5_E${CF_ROUTINE}_%j.err" \
        --export=ALL,CF_STEP=zip,CF_WHICH=cf5,CF_STAGE=${CF_STAGE},DATA_ROOT=${DATA_ROOT} "${HERE}/submit_cf.sh")
    echo "  zip cf5     → job ${zj} (afterany:${cmp})"
fi
echo "CF5 pass-through submitted (E${CF_ROUTINE}). Result → CF_FOUNDATION/cf5_passthrough_E${CF_ROUTINE}_spec_12_${CF_STAGE}.parquet. Watch: squeue -u \$USER"
