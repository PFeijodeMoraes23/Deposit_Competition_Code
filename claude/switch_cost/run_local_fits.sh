#!/usr/bin/env bash
# Local spec-12 fits of the switch-cost investigation (Git Bash, from the repo root).
# Pooled = the production specification re-estimated by the copies; typeshift = the same
# plus the deposit-type shifts in the phi index. Each writes only to its own tree under
# processed/ESTIMATION_OUTPUT/SWITCH_COST (utils/switch_cost_sandbox.py). Sequential on
# purpose: concurrent estimators exhaust the Windows non-paged pool.
#   usage: bash claude/switch_cost/run_local_fits.sh [3 4]
set -u
PY="${PY:-C:/venvs/egan/Scripts/python.exe}"
export MPLBACKEND=Agg
ESTS="${*:-3 4}"
LOGS="../../BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/SWITCH_COST/logs"
mkdir -p "${LOGS}"
for E in ${ESTS}; do
    for V in pooled typeshift; do
        TS=0; [[ "${V}" == "typeshift" ]] && TS=1
        LOG="${LOGS}/fit_${V}_est${E}.log"
        echo "[$(date '+%H:%M:%S')] start ${V} E${E} -> ${LOG}"
        SWITCH_COST_TYPE_SHIFTS="${TS}" "${PY}" -u switch_cost_sleep_est_single.py --est "${E}" --spec12 > "${LOG}" 2>&1
        rc=$?   # captured first: a $(date) in the echo would reset it
        echo "[$(date '+%H:%M:%S')] end   ${V} E${E} rc=${rc}"
    done
done
