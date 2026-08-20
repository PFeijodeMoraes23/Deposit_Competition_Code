#!/bin/bash
# ==============================================================================
# pipeline_run.sh — one command: BBL cost estimation -> the counterfactuals, in
# order, with the cross-stage dependency handled by SLURM (fire and forget; watch
# with `squeue -u $USER`).
#
# WHAT MUST EXIST FIRST
#   1. bash cluster_preflight.sh                    (must print PREFLIGHT OK)
#   2. the RC results (blp_run.sh finished, or its blp_outputs_*.zip on disk)
#   3. data/input/forward_rf_qoq.csv and data/input/polfunc_fitted.csv
#   4. for the CF4 legs: data/input/{upsilon_pix,phi_nopix}_E{k}_spec_12.*
# WHAT TO RUN NEXT
#   bash cf_eq_run.sh --mode cf3|cf5|cf6      (the long equilibrium CFs)
#   bash cluster_archive.sh --all --copy      (then --move at the very end)
#
# THIS SCRIPT IS NEW. submit_bbl_cf_all.sh is still present and still works; it is
# the fallback and is retired only after one successful cluster cycle. Nothing in
# it has been modified.
#
# LAUNCH ORDER
#   Phase 1a  BBL cost estimation (bbl_run.sh) -> cost_params_E*_spec_12_{stage}.json
#   Phase 1b  CF4 Pix, NO re-eval (CF4_EXACT_NOPIX=0): the identity-link scalar
#             fallback phi_cf = clamp(phi-hat - Upsilon_pix*pix). Needs NO costs, so
#             it runs alongside the BBL. The baseline demand_eval (shares_elas) is
#             produced once, here.
#   Phase 2   CF1 gross + CF4 re-eval (CF4_EXACT_NOPIX=1) + CF1 net. CF1-net
#             consumes the BBL cost params, so it is submitted with afterok on the
#             BBL solve jobs and the on-disk cost preflight is skipped — it
#             launches the instant the cost params are written.
#   Both CF phases REUSE ONE cf_warmup barrier: phase 1b creates it, phase 2 is
#   handed its job id, so the depot is warmed once, not twice.
#
# GOTCHA, and it is real: a failed BBL solve silently takes cf1_net with it.
# cf1_net is chained afterok on the solve, so --kill-on-invalid-dep cancels it as
# DependencyNeverSatisfied and it produces NO LOG FILE AT ALL. A missing cf1_net
# log means "read bbl_solve's .err", not "cf1_net misbehaved". (2026-08-01: the
# solve died at t+4s on a missing pandas; the login-node env probe in
# cluster_preflight.sh / bbl_run.sh now catches that at submit time.)
#
# Usage
#   bash pipeline_run.sh --dry-run
#   bash pipeline_run.sh
#   bash pipeline_run.sh --routines "3" --no-bbl
#
# Flags
#   --routines "3 4"   routine set (default from cluster_lib.sh)
#   --no-bbl           skip the BBL stage; cost params must already be on disk and
#                      CF1-net then uses the normal on-disk preflight
#   --no-cf4           skip both CF4 legs
#   --no-cf1           skip CF1 gross and CF1 net
#   --no-final-zip     do not submit the terminal CF archive job
#   --no-log-zip       keep this run's ~800 .out/.err loose instead of bundling them
#   --skip-preflight   pass through to the children
#   --dry-run          print, submit nothing
#   -h                 this header
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"
set -e

ROUTINES_SRC=default
if [[ -n "${ROUTINES+set}" ]]; then ROUTINES_SRC=env; fi
ROUTINES="${ROUTINES:-${CL_ROUTINES_CF}}"
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
DO_BBL=1; DO_CF4=1; DO_CF1=1; DO_FINAL_ZIP=1; DO_LOG_ZIP=1
SKIP_PF=""; DRY=""
FINAL_ZIP_SETS="${FINAL_ZIP_SETS:-foundation cf1 cf4}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --routines)       ROUTINES="$2"; ROUTINES_SRC=flag; shift ;;
        --no-bbl)         DO_BBL=0 ;;
        --no-cf4)         DO_CF4=0 ;;
        --no-cf1)         DO_CF1=0 ;;
        --no-final-zip)   DO_FINAL_ZIP=0 ;;
        --no-log-zip)     DO_LOG_ZIP=0 ;;
        --skip-preflight) SKIP_PF="--skip-preflight" ;;
        --dry-run)        CL_DRYRUN=1; DRY="--dry-run" ;;
        -h|--help)        sed -n '2,56p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

LOGD="$(cl_log_dir)"
# CF_STAGE/R/SEED are shared config and are exported. ROUTINES deliberately is
# NOT: it is passed to each child as --routines, so the child reports it as a
# FLAG rather than raising the "FROM ENVIRONMENT" warning, which must stay
# reserved for a stray `export ROUTINES=` genuinely lingering in the login shell.
export CF_STAGE R SEED
CHILD_ROUTINES=(--routines "${ROUTINES}")

cl_banner "BBL -> CF orchestrator$([[ "${CL_DRYRUN}" == "1" ]] && echo '  [DRY RUN — nothing is submitted]')" \
          "$(cl_routines_provenance "${ROUTINES}" "${ROUTINES_SRC}" "${CL_ROUTINES_CF}")" \
          "stage=${CF_STAGE} R=${R} seed=${SEED}" \
          "BBL=${DO_BBL} | CF4=${DO_CF4} | CF1=${DO_CF1}"

# A full run writes ~800 log files (each fwd_sim array task emits a .out and a
# .err, x N_SHARDS x routines). The marker scopes the log archive by mtime so
# earlier runs' logs survive.
# A dry run must leave nothing behind, so it names a marker without creating it.
if [[ "${CL_DRYRUN}" == "1" ]]; then LOG_MARKER="${LOGD}/.orch_marker.DRYRUN"
else LOG_MARKER="$(mktemp "${LOGD}/.orch_marker.XXXXXX")"; fi

CF_JOBIDS=""; ALL_JOBIDS=""; WARMUP_JOBID=""
add_jobids () { [[ -n "$1" ]] && ALL_JOBIDS="${ALL_JOBIDS:+${ALL_JOBIDS}:}$1"; return 0; }
capture_cf () {   # $1 = captured cf_run.sh stdout
    local ids w
    ids="$(printf '%s\n' "$1" | sed -n 's/^CF_RESULT_JOBIDS=//p' | tail -n1)"
    w="$(printf '%s\n'   "$1" | sed -n 's/^CF_WARMUP_JOBID=//p'  | tail -n1)"
    [[ -n "${ids}" ]] && { CF_JOBIDS="${CF_JOBIDS:+${CF_JOBIDS}:}${ids}"; add_jobids "${ids}"; }
    [[ -n "${w}" && -z "${WARMUP_JOBID}" ]] && { WARMUP_JOBID="${w}"; add_jobids "${w}"; }
    return 0
}

# ── Phase 1a: BBL cost estimation ────────────────────────────────────────────
COST_AFTEROK=""
if [[ "${DO_BBL}" == "1" ]]; then
    echo; echo "=== Phase 1a: BBL cost estimation (bbl_run.sh) ==="
    # Capture WITHOUT letting set -e abort before the output is shown. The child's
    # preflight prints its MISSING diagnostics to stdout, and dying on the $(...)
    # assignment swallowed them entirely (observed 2026-08-07: Phase 1a printed
    # nothing and no job was submitted). Same pattern at every capture site below.
    set +e; bbl_out="$(bash "${CL_ROOT}/bbl_run.sh" "${CHILD_ROUTINES[@]}" ${DRY} ${SKIP_PF})"; rc=$?; set -e
    printf '%s\n' "${bbl_out}"
    if [[ ${rc} -ne 0 ]]; then
        echo "ERROR: bbl_run.sh failed (rc=${rc}) — its output is above (usually a preflight MISSING)." >&2
        rm -f "${LOG_MARKER}"; exit "${rc}"
    fi
    COST_AFTEROK="$(printf '%s\n' "${bbl_out}" | sed -n 's/^BBL_SOLVE_JOBIDS=//p' | tail -n1)"
    add_jobids "$(printf '%s\n' "${bbl_out}" | sed -n 's/^BBL_ALL_JOBIDS=//p' | tail -n1)"
    if [[ -z "${COST_AFTEROK}" && "${DO_CF1}" == "1" ]]; then
        echo "ERROR: bbl_run.sh emitted no BBL_SOLVE_JOBIDS — cannot chain CF1-net. Aborting." >&2
        echo "  (Re-run with --no-cf1, or --no-bbl once cost_params_*.json are on disk.)" >&2
        rm -f "${LOG_MARKER}"; exit 1
    fi
    echo "  -> BBL solve jobs for the CF1-net afterok: ${COST_AFTEROK:-<none>}"
else
    echo; echo "=== Phase 1a: BBL SKIPPED (--no-bbl) — CF1-net will preflight cost_params_*.json on disk ==="
fi

# ── Phase 1b: CF4 Pix, NO re-eval (identity-link fallback; needs no costs) ───
first_demand=1
if [[ "${DO_CF4}" == "1" ]]; then
    echo; echo "=== Phase 1b: CF4 no re-eval  (CF4_EXACT_NOPIX=0) ==="
    steps="cf4"; [[ "${first_demand}" == "1" ]] && steps="demand_eval cf4"
    set +e
    out="$(CF4_EXACT_NOPIX=0 bash "${CL_ROOT}/cf_run.sh" "${CHILD_ROUTINES[@]}" --do "${steps}" ${DRY} ${SKIP_PF})"
    rc=$?; set -e
    printf '%s\n' "${out}"
    [[ ${rc} -ne 0 ]] && { echo "ERROR: Phase-1b cf_run.sh failed (rc=${rc}) — output above." >&2; rm -f "${LOG_MARKER}"; exit "${rc}"; }
    capture_cf "${out}"
    first_demand=0
fi

# ── Phase 2: CF1 gross + CF4 re-eval + CF1 net (afterok the BBL solve) ──────
if [[ "${DO_CF1}" == "1" || "${DO_CF4}" == "1" ]]; then
    echo; echo "=== Phase 2: CF1 gross + CF1 net + CF4 re-eval  (CF4_EXACT_NOPIX=1) ==="
    steps=""
    [[ "${first_demand}" == "1" ]] && steps="demand_eval"
    [[ "${DO_CF1}" == "1" ]] && steps="${steps} cf1 cf1_net"
    [[ "${DO_CF4}" == "1" ]] && steps="${steps} cf4"
    set +e
    # --export=ALL in cf_run.sh's submissions is what carries CF4_EXACT_NOPIX
    # through to cf_4_pix.jl:155 by environment inheritance.
    out="$(CF4_EXACT_NOPIX=1 bash "${CL_ROOT}/cf_run.sh" "${CHILD_ROUTINES[@]}" --do "${steps# }" \
              ${WARMUP_JOBID:+--warmup-jobid "${WARMUP_JOBID}"} \
              ${COST_AFTEROK:+--cost-afterok "${COST_AFTEROK}"} ${DRY} ${SKIP_PF})"
    rc=$?; set -e
    printf '%s\n' "${out}"
    [[ ${rc} -ne 0 ]] && { echo "ERROR: Phase-2 cf_run.sh failed (rc=${rc}) — output above." >&2; rm -f "${LOG_MARKER}"; exit "${rc}"; }
    capture_cf "${out}"
fi

# ── Final CF archive: afterany every CF job. COPY, always. ──────────────────
final_zip_jid=""
if [[ "${DO_FINAL_ZIP}" == "1" && -n "${CF_JOBIDS}" ]]; then
    echo; echo "=== Final archive: ${FINAL_ZIP_SETS} -> data/output (afterany all CF jobs) ==="
    # COPY because cost_params and the sigma dirs are still read in place by
    # anything downstream, and because the cf4 set sits next to uploaded inputs.
    cmd="cd '${CL_ROOT}'"
    for s in ${FINAL_ZIP_SETS}; do cmd="${cmd} && bash cluster_archive.sh --set ${s} --copy"; done
    final_zip_jid=$(cl_sbatch --dependency=afterany:"${CF_JOBIDS}" \
        -J cf_final_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGD}/cf_final_zip_%j.out" -e "${LOGD}/cf_final_zip_%j.err" \
        --wrap "${cmd}")
    echo "  final CF archive -> job ${final_zip_jid} (afterany:${CF_JOBIDS})"
    add_jobids "${final_zip_jid}"
elif [[ "${DO_FINAL_ZIP}" == "1" ]]; then
    echo; echo "-- final archive skipped: no CF jobs were submitted --"
fi

# ── Log archive: ONE zip for the whole run, afterany every job above ────────
log_zip_jid=""
if [[ "${DO_LOG_ZIP}" == "1" && -n "${ALL_JOBIDS}" ]]; then
    echo; echo "=== Log archive: this run's logs -> data/output/run_logs_<jobid>.zip ==="
    # --newer scopes it to logs written AFTER this orchestrator started, so
    # previous runs' logs survive. cluster_archive.sh excludes the archiving job's
    # own still-open .out/.err by job id, and MOVE deletes only entries it has
    # confirmed inside the archive.
    LWRAP="cd '${CL_ROOT}'"
    LWRAP="${LWRAP}; bash cluster_archive.sh --set run_logs --move --newer '${LOG_MARKER}' --tag \"\${SLURM_JOB_ID}\""
    LWRAP="${LWRAP}; rm -f '${LOG_MARKER}'"
    log_zip_jid=$(cl_sbatch --dependency=afterany:"${ALL_JOBIDS}" \
        -J run_log_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G \
        -o "${LOGD}/run_log_zip_%j.out" -e "${LOGD}/run_log_zip_%j.err" \
        --wrap "${LWRAP}")
    echo "  run-log archive -> job ${log_zip_jid} (afterany ${ALL_JOBIDS//:/, })"
else
    [[ "${CL_DRYRUN}" == "1" ]] || rm -f "${LOG_MARKER}"
fi

echo
cl_banner "All stages submitted. Track:  squeue -u \$USER" \
          "CF1-net waits on the BBL solve (afterok) and stays PENDING until the params exist." \
          "CF4 writes two output sets: *_noeval (Phase 1b) and the exact re-eval (Phase 2)."
echo " Downloads when everything finishes:"
[[ "${DO_BBL}" == "1" ]] && echo "   - BBL cost params : data/output/bbl_outputs_<jobid>.zip"
if [[ -n "${final_zip_jid}" ]]; then
    echo "   - CF outputs      : data/output/{$(echo ${FINAL_ZIP_SETS} | tr ' ' ',')}_outputs.zip  (job ${final_zip_jid})"
else
    echo "   - CF outputs      : once squeue is empty ->  bash cluster_archive.sh --all --copy"
fi
[[ -n "${log_zip_jid}" ]] && echo "   - run logs        : data/output/run_logs_<${log_zip_jid}>.zip"
echo "PIPELINE_ALL_JOBIDS=${ALL_JOBIDS}"
