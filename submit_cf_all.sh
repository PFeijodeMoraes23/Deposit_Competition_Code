#!/bin/bash
# submit_cf_all.sh — ONE-COMMAND orchestrator for the counterfactual pipeline on Bouchet.
#
# Run from the code directory on the login node; it does everything:
#   1. builds cluster_processed/ from the RC zip (blp_outputs_*.zip) if it's missing
#      (process_blp_outputs.py — light, runs here on the login node), then
#   2. preflight-checks the staged inputs (draws, RC results, forward r^f curve), then
#   3. submits the CF chain for EACH routine in ROUTINES:
#        (optional) demand_eval — 0a share reproduction (first routine only; precompiles)
#        (optional) cf1         — gross franchise-value decomposition          [1 job]
#        cost2 ARRAY            — CF2 ψ deviations, N_SHARDS tasks              [array job]
#        cost_solve             — Eq-18 minimization, afterok the array         [1 job]
#   Each cost2 array task computes SHOCKS/N_SHARDS deviations (reproducible per-shock RNG
#   → shards merge exactly); the solver globs all shards.
#
# Usage:
#     bash submit_cf_all.sh                       # E6 headline, everything
#     ROUTINES="5 6 7 8" bash submit_cf_all.sh    # headline + robustness band, one shot
#   Tunables (env):
#     ROUTINES="6"          routines to run CFs for (space list; "5 6 7 8" = headline+band)
#     CF_STAGE=extended  R=2000  SEED=42
#     SHOCKS=50  N_SHARDS=10  PERTURB_SCALE=0.02  DEV_SCHEME=grid  BETA=0.9  HORIZON=50
#     DO_DEMAND_EVAL=1  DO_CF1=1
#     AUTO_PROCESS=1        auto-build cluster_processed/ from blp_outputs_*.zip if absent
#     BLP_ZIP=<path>        pin a specific RC zip (default: latest by job id across all saved)
#     PYTHON=python         interpreter for process_blp_outputs.py
#     SHARD_TIME=08:00:00   SOLVE_TIME=01:00:00
#
# The forward r^f curve (data/COST_FWD/forward_rf_qoq.csv) is the one input that must be
# built LOCALLY (cf_forward_rf.py needs internet) and uploaded; everything else is either
# already staged from the BLP run or auto-built here from the zip. See runbook §8.
set -euo pipefail

ROUTINES="${ROUTINES:-${CF_ROUTINE:-6}}"
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
SHOCKS="${SHOCKS:-50}"; N_SHARDS="${N_SHARDS:-10}"
PERTURB_SCALE="${PERTURB_SCALE:-0.02}"; DEV_SCHEME="${DEV_SCHEME:-grid}"
BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}"
DO_DEMAND_EVAL="${DO_DEMAND_EVAL:-1}"; DO_CF1="${DO_CF1:-1}"
AUTO_PROCESS="${AUTO_PROCESS:-1}"; PYTHON="${PYTHON:-python}"
SHARD_TIME="${SHARD_TIME:-08:00:00}"; SOLVE_TIME="${SOLVE_TIME:-01:00:00}"
LOGDIR="logs"; mkdir -p "${LOGDIR}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/../data}"
CP_DIR="${DATA_ROOT}/output/cluster_processed"
ROUTINES_CSV="$(echo ${ROUTINES} | tr ' ' ',')"

# ── Step 1: build cluster_processed/ from the RC zip if any routine is missing ───
# Each RC run is saved as blp_outputs_<SLURM_JOB_ID>.zip (submit_blp_rc_all.sh), so with
# several runs on disk "latest" = highest job id. pick_zip returns: BLP_ZIP override →
# highest-job-id zip → (legacy fallback) newest by mtime. We touch the winner so
# process_blp_outputs.py — which auto-discovers by newest mtime — processes exactly it.
pick_zip () {
    if [[ -n "${BLP_ZIP:-}" ]]; then printf '%s\n' "${BLP_ZIP}"; return; fi
    local best="" bestid=-1 f id
    for f in "${DATA_ROOT}"/output/blp_outputs_*.zip "${DATA_ROOT}"/output/cluster_raw/blp_outputs_*.zip; do
        [[ -f "$f" ]] || continue
        id="$(basename "$f" | sed -E 's/^blp_outputs_([0-9]+)\.zip$/\1/')"
        if [[ "$id" =~ ^[0-9]+$ ]] && (( id > bestid )); then bestid=$id; best="$f"; fi
    done
    [[ -n "$best" ]] || best="$(ls -t "${DATA_ROOT}"/output/blp_outputs*.zip "${DATA_ROOT}"/output/cluster_raw/blp_outputs*.zip 2>/dev/null | head -1 || true)"
    printf '%s\n' "${best}"
}

need_process=0
for k in ${ROUTINES}; do [[ -f "${CP_DIR}/blp_E${k}_spec_12.jls" ]] || need_process=1; done
if [[ "${need_process}" == "1" && "${AUTO_PROCESS}" == "1" ]]; then
    zip="$(pick_zip)"
    if [[ -n "${zip}" && -f "${zip}" ]]; then
        echo "── building cluster_processed/ from $(basename "${zip}") (latest of $(ls "${DATA_ROOT}"/output/blp_outputs_*.zip "${DATA_ROOT}"/output/cluster_raw/blp_outputs_*.zip 2>/dev/null | wc -l) zip(s); routines ${ROUTINES_CSV}) ──"
        touch "${zip}" 2>/dev/null || true   # make it the newest-mtime → process_blp_outputs.py picks THIS one
        # The .jls copy runs before the optional pandas heterogeneity/summary step, so the
        # CF inputs land even if that trailing step warns — tolerate a nonzero exit.
        ${PYTHON} process_blp_outputs.py "${DATA_ROOT}/output" --routines "${ROUTINES_CSV}" \
            || echo "  (process_blp_outputs.py returned nonzero — verifying .jls in preflight anyway)"
    else
        echo "── no blp_outputs_*.zip under ${DATA_ROOT}/output — cannot auto-build cluster_processed/ (has the RC run finished?)"
    fi
fi

# ── Step 2: preflight — every required input must be present ──────────────────────
RF_CURVE="${DATA_ROOT}/COST_FWD/forward_rf_qoq.csv"
DRAWS="${DATA_ROOT}/output/BLP_DRAWS/halton_nu_R${R}_seed${SEED}.jls"
miss=0
[[ -f "${DRAWS}" ]]    || { echo "MISSING R=${R} draws:  ${DRAWS}"; miss=1; }
[[ -f "${RF_CURVE}" ]] || { echo "MISSING forward curve: ${RF_CURVE}"; \
    echo "   → build locally then upload: python cf_forward_rf.py --horizon ${HORIZON} --start 2026Q1"; miss=1; }
for k in ${ROUTINES}; do
    [[ -f "${CP_DIR}/blp_E${k}_spec_12.jls" ]] || { echo "MISSING RC result: ${CP_DIR}/blp_E${k}_spec_12.jls"; \
        echo "   → drop the RC blp_outputs_*.zip in ${DATA_ROOT}/output and re-run (auto-built), or:"; \
        echo "     python process_blp_outputs.py ${DATA_ROOT}/output --routines 5,6,7,8"; miss=1; }
done
[[ "${miss}" == "0" ]] || { echo "Stage the missing input(s) (runbook §8), then re-run."; exit 1; }
echo "Preflight OK: R=${R} draws + forward r^f curve + RC results for routines: ${ROUTINES}"

# ── Step 3: submit the CF chain per routine ──────────────────────────────────────
cf2_extra="--shocks ${SHOCKS} --perturb-scale ${PERTURB_SCALE} --dev-scheme ${DEV_SCHEME} --beta ${BETA} --horizon ${HORIZON}"
cf1_extra="--beta ${BETA} --horizon ${HORIZON}"

submit () {  # submit <jobname> <time> <extra-sbatch-args...>
    local name="$1" tlim="$2"; shift 2
    sbatch --parsable -J "${name}" -t "${tlim}" \
        -o "${LOGDIR}/${name}_%A_%a.out" -e "${LOGDIR}/${name}_%A_%a.err" "$@"
}

first=1
for k in ${ROUTINES}; do
    base_export="CF_ROUTINE=${k},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED}"
    echo "── E${k} ${CF_STAGE} | R=${R} | shocks=${SHOCKS} over ${N_SHARDS} shards ──"
    # demand_eval runs once (first routine) — it also Pkg.instantiate/precompiles the depot.
    if [[ "${DO_DEMAND_EVAL}" == "1" && "${first}" == "1" ]]; then
        j=$(submit "cf_demaneval_E${k}" "${SOLVE_TIME}" \
            --export=ALL,${base_export},CF_STEP=demand_eval submit_cf.sh)
        echo "  demand_eval  → job ${j}"
    fi
    if [[ "${DO_CF1}" == "1" ]]; then
        j=$(submit "cf_cf1_E${k}" "${SHARD_TIME}" \
            --export=ALL,${base_export},CF_STEP=cf1,CF_EXTRA="${cf1_extra}" submit_cf.sh)
        echo "  cf1          → job ${j}"
    fi
    arr=$(submit "cf_cost2_E${k}" "${SHARD_TIME}" --array=0-$((N_SHARDS-1)) \
        --export=ALL,${base_export},CF_STEP=cost2,N_SHARDS=${N_SHARDS},CF_EXTRA="${cf2_extra}" \
        submit_cf.sh)
    echo "  cost2 array  → job ${arr} (${N_SHARDS} shards)"
    slv=$(submit "cf_solve_E${k}" "${SOLVE_TIME}" --dependency=afterok:"${arr}" \
        --export=ALL,${base_export},CF_STEP=cost_solve,CF_EXTRA="--bootstrap 200" submit_cf.sh)
    echo "  cost_solve   → job ${slv} (afterok:${arr})"
    first=0
done

echo "Submitted CFs for routines: ${ROUTINES}. Watch with: squeue -u \$USER"
