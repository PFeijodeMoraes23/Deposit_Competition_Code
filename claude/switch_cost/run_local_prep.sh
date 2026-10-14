#!/usr/bin/env bash
# Local demand prep of the switch-cost arms, spec 12, with the identity checks (Git Bash, from
# the repo root, after run_local_fits.sh). Reads production fits and parquets, never writes
# them: every output folder is under processed/ESTIMATION_OUTPUT/SWITCH_COST.
#   usage: bash claude/switch_cost/run_local_prep.sh [3 4]
set -u
PY="${PY:-C:/venvs/egan/Scripts/python.exe}"
export MPLBACKEND=Agg
ESTS="${*:-3 4}"
EO="../../BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT"
SC="${EO}/SWITCH_COST"
LOGS="${SC}/logs"
mkdir -p "${LOGS}"
prep () {   # prep <est> <log name> <args...>
    local e="$1" name="$2"; shift 2
    echo "[$(date '+%H:%M:%S')] prep E${e} ${name}"
    "${PY}" -u switch_cost_sleep_demand_prep_link.py --est "${e}" --spec 12 "$@" > "${LOGS}/prep_${name}_est${e}.log" 2>&1
    local rc=$?   # captured first: a $(date) in the echo would reset it
    echo "[$(date '+%H:%M:%S')]   rc=${rc}"
}
check () {  # check <name> <a> <b>
    "${PY}" -u switch_cost_prep_checks.py parquet --name "$1" "$2" "$3" > "${LOGS}/check_$1.log" 2>&1
    local rc=$?
    echo "[$(date '+%H:%M:%S')] check $1 rc=${rc} (log ${LOGS}/check_$1.log)"
}
for E in ${ESTS}; do
    TAG=index; [[ "${E}" == "4" ]] && TAG=index_time
    F="demand_${E}_${TAG}_spec_12.parquet"
    "${PY}" -u switch_cost_prep_checks.py fit --est "${E}" > "${LOGS}/check_fit_est${E}.log" 2>&1
    rc=$?
    echo "[$(date '+%H:%M:%S')] check fit_est${E} rc=${rc}"
    # C2: the copied prep on the production fits reproduces the production parquet
    prep "${E}" C2_prodfits --arm pooled --fits-dir "${EO}/DEMAND_PREP" --out-dir "${SC}/switch_cost_check_prodfits"
    check "C2_est${E}" "${SC}/switch_cost_check_prodfits/${F}" "${EO}/DEMAND_PREP_20261013/${F}"
    # the pooled arm on the re-estimated pooled fit: the reference of every arm
    prep "${E}" pooled --arm pooled
    # C3 / C4: the F1 and F3 machinery collapse to the pooled arm
    prep "${E}" C3_F1_identity --arm F1 --fits-dir "${SC}/switch_cost_pooled" --out-dir "${SC}/switch_cost_check_F1_identity"
    check "C3_est${E}" "${SC}/switch_cost_check_F1_identity/${F}" "${SC}/switch_cost_demand_pooled/${F}"
    prep "${E}" C4_F3_zero --arm F3hi --zero-lock --fits-dir "${SC}/switch_cost_pooled" --out-dir "${SC}/switch_cost_check_F3_zero"
    check "C4_est${E}" "${SC}/switch_cost_check_F3_zero/${F}" "${SC}/switch_cost_demand_pooled/${F}"
    # the arms
    for ARM in F1 F3hi F3lo; do
        prep "${E}" "${ARM}" --arm "${ARM}"
    done
done
