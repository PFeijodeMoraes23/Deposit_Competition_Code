#!/bin/bash
# submit_cf6_merger.sh — CF6 merger simulation on Bouchet.
#
# Solves TWO full-panel equilibria via the firm-sharded Jacobi (GPU shards) — the BASE and the
# MERGED market (the pair MERGE="A,B" treated as one decision-maker) — then a compare job reports
# the merger's franchise-value effect (pre V_A+V_B vs post V_AB). The two solves run in parallel;
# the compare is afterok both.
#
# Prereq: RC costs (run the BBL cost stage first:  bash submit_bbl_all.sh).
# Usage:  CF_ROUTINE=7 MERGE="C0080099,C0080329" N_FIRM_SHARDS=40 N_SWEEPS=6 bash submit_cf6_merger.sh
# Tunables: everything submit_cf3_jacobi.sh takes, plus MERGE (required, the conglomerate pair) and
#   BASE_SIGMA=<path> to REUSE an already-solved base σ instead of re-solving it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF_ROUTINE="${CF_ROUTINE:-6}"; CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"; BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}"
CPU_PARTITION="${CPU_PARTITION:-day}"
: "${MERGE:?set MERGE=\"firmA,firmB\" (the conglomerate pair to merge)}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/../data}"
ROOT="${DATA_ROOT}/CF_FOUNDATION/cf6_E${CF_ROUTINE}_${CF_STAGE}"
export CF_ROUTINE CF_STAGE R SEED BETA HORIZON DATA_ROOT   # inherited by submit_cf3_jacobi.sh
mkdir -p "${HERE}/logs"

dep=""
if [[ -n "${BASE_SIGMA:-}" ]]; then
    base_sig="${BASE_SIGMA}"; echo "== CF6 base: reusing ${base_sig} =="
else
    echo "== CF6 base equilibrium =="
    # ZIP_AFTER="" — the base/merged CF3 solves are internal to CF6; we zip the cf6 archive below.
    base_out=$(SIGMA_DIR="${ROOT}/base" EXTRA_CF="" ZIP_AFTER="" bash "${HERE}/submit_cf3_jacobi.sh"); echo "${base_out}"
    base_job=$(echo "${base_out}" | sed -n 's/^FINAL_JOB=//p'); base_sig=$(echo "${base_out}" | sed -n 's/^FINAL_SIGMA=//p')
    dep="${dep}:${base_job}"
fi
echo "== CF6 merged equilibrium (${MERGE}) =="
mg_out=$(SIGMA_DIR="${ROOT}/merged" EXTRA_CF="--merge ${MERGE}" ZIP_AFTER="" bash "${HERE}/submit_cf3_jacobi.sh"); echo "${mg_out}"
mg_job=$(echo "${mg_out}" | sed -n 's/^FINAL_JOB=//p'); mg_sig=$(echo "${mg_out}" | sed -n 's/^FINAL_SIGMA=//p')
dep="${dep}:${mg_job}"

echo "== CF6 compare (afterok${dep}) =="
cmp=$(sbatch --parsable -J "cf6cmp_E${CF_ROUTINE}" -t 02:00:00 --kill-on-invalid-dep=yes --partition="${CPU_PARTITION}" \
    --dependency=afterok"${dep}" \
    -o "${HERE}/logs/cf6cmp_E${CF_ROUTINE}_%j.out" -e "${HERE}/logs/cf6cmp_E${CF_ROUTINE}_%j.err" \
    --export=ALL,CF_ROUTINE=${CF_ROUTINE},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED},CF_STEP=cf6_compare,SIGMA_BASE="${base_sig}",SIGMA_SCN="${mg_sig}",CF_EXTRA="--merge ${MERGE} --beta ${BETA} --horizon ${HORIZON}" \
    "${HERE}/submit_cf.sh")
echo "  cf6_compare → job ${cmp}"

# Auto-zip the cf6 archive (merger parquet + the base/merged σ dirs) after the compare.
if [[ "${DO_ZIP:-1}" == "1" ]]; then
    zj=$(sbatch --parsable -J "cf_zip_cf6_E${CF_ROUTINE}" -t 00:30:00 --kill-on-invalid-dep=yes \
        --partition="${CPU_PARTITION}" --mem=4G --dependency=afterany:"${cmp}" \
        -o "${HERE}/logs/cf_zip_cf6_E${CF_ROUTINE}_%j.out" -e "${HERE}/logs/cf_zip_cf6_E${CF_ROUTINE}_%j.err" \
        --export=ALL,CF_STEP=zip,CF_WHICH=cf6,CF_STAGE=${CF_STAGE},DATA_ROOT=${DATA_ROOT} "${HERE}/submit_cf.sh")
    echo "  zip cf6     → job ${zj} (afterany:${cmp})"
fi
echo "CF6 merger submitted (E${CF_ROUTINE}, ${MERGE}). Result → CF_FOUNDATION/cf6_merger_E${CF_ROUTINE}_spec_12_${CF_STAGE}.parquet. Watch: squeue -u \$USER"
