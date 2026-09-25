#!/bin/bash
# ==============================================================================
# bbl_cancel.sh — cancel BBL chains in DEPENDENCY ORDER, dependents first.
#
#   bash bbl_cancel.sh --routines "3" --psi-tag _ms1            one routine's chain
#   bash bbl_cancel.sh --routines "1 2 3 4" --psi-tag _ms1      a whole run
#   bash bbl_cancel.sh ... --with-shared                        + bbl_warmup / bbl_polfunc
#   bash bbl_cancel.sh ... --dry-run                            list, cancel nothing
#
# WHY THE ORDER. `afterany` is satisfied by a CANCELLED job. Cancel an array first and every job
# waiting afterany on it starts at once: the sweep sees a gap and re-submits it, and before the
# sweep existed the solve, tables and archive ran on the partial set. So the order is
#   1. tables + archive of the tag (afterok every solve: they could not run anyway)
#   2. solves
#   3. sweeps          (after a STOP marker, so a sweep already running re-submits nothing)
#   4. fwd arrays      (then sweeps and arrays once more, for anything a sweep submitted
#                       while it was being cancelled)
# Each step waits until its jobs have left the queue before the next begins. CF jobs chained
# afterok a solve (cf_run.sh --cost-afterok) are cancelled by SLURM itself
# (--kill-on-invalid-dep=yes) and leave no log.
#
# LOGIN-NODE SAFE: squeue, scancel, and one small STOP file per routine in the dispatch folder.
# A later bbl_run.sh launch or --repair removes the STOP file.
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"

ROUTINES=""; STAGE="${CF_STAGE:-extended}"; TAG=""; TAG_SET=0; SHARED=0; DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --routines)    ROUTINES="$2"; shift ;;
        --psi-tag)     TAG="$2"; TAG_SET=1; shift ;;
        --stage)       STAGE="$2"; shift ;;
        --with-shared) SHARED=1 ;;
        --dry-run)     DRY=1 ;;
        -h|--help)     sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done
[[ -n "${ROUTINES}" ]] || { echo "bbl_cancel.sh: --routines is required (the routines whose chain to cancel)" >&2; exit 2; }
[[ "${TAG_SET}" == "1" ]] || { echo "bbl_cancel.sh: --psi-tag is required (--psi-tag '' for the untagged run)" >&2; exit 2; }
command -v squeue >/dev/null 2>&1 || { echo "bbl_cancel.sh: no squeue here (not a SLURM host)" >&2; exit 2; }

_names () { local k out=""; for k in ${ROUTINES}; do out="${out:+${out},}$1$(cl_bbl_rtag "${k}" "${STAGE}" "${TAG}")"; done; printf '%s' "${out}"; }
_cancel () {   # _cancel <label> <comma-separated job names>
    local label="$1" ids i
    ids="$(cl_bbl_live_named "$2")"
    if [[ -z "${ids}" ]]; then printf '  %-22s none live\n' "${label}"; return 0; fi
    if [[ "${DRY}" == "1" ]]; then printf '  %-22s would scancel %s\n' "${label}" "${ids}"; return 0; fi
    printf '  %-22s scancel %s\n' "${label}" "${ids}"
    scancel ${ids} || true
    for i in $(seq 1 30); do
        [[ -z "$(squeue -h -j "${ids// /,}" -o '%i' 2>/dev/null)" ]] && return 0
        sleep 2
    done
    printf '  %-22s [!] still listed after 60 s (COMPLETING?) — continuing\n' "${label}"
}

echo "bbl_cancel: routines ${ROUTINES} | tag '${TAG}' | stage ${STAGE}$([[ "${DRY}" == "1" ]] && echo '  [DRY RUN]')"
if [[ "${DRY}" != "1" ]]; then
    for k in ${ROUTINES}; do
        d="$(cl_bbl_dispatch_dir "$(cl_bbl_key "${k}" "${STAGE}" "${TAG}")")"
        mkdir -p "${d}" && : > "${d}/STOP"
    done
    echo "  STOP markers written: a sweep that starts now re-submits nothing"
fi
_cancel "1 tables + archive" "bbl_tables${TAG},bbl_zip${TAG}"
_cancel "2 solves" "$(_names bbl_solve_)"
_cancel "3 sweeps" "$(_names bbl_sweep_)"
_cancel "4 fwd jobs" "$(_names bbl_fwd_)"
_cancel "  sweeps (again)" "$(_names bbl_sweep_)"
_cancel "  fwd jobs (again)" "$(_names bbl_fwd_)"
if [[ "${SHARED}" == "1" ]]; then _cancel "5 warmup + polfunc" "bbl_warmup,bbl_polfunc"; fi
echo "done. Files already written stay; a later  bash bbl_run.sh ... --repair  re-attaches the chain."
