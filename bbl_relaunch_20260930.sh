#!/bin/bash
# ==============================================================================
# bbl_relaunch_20260930.sh -- the 2026-09-30 relaunch of the BBL cost estimation on the refitted
# policy function (bbl_polfunc.py fitted on positive-balance rows only, re-estimated locally at
# B=999; polfunc_fitted.csv sha256 fd208b5f...). Everything that is left, in one submission from
# the scripts folder (one line; the upload's md5 list is checked first):
#
#   cd ~/project_pi_mf2263/pf382/dep_comp && unzip -o ~/bbl_update_20260930.zip && md5sum -c bbl_update_20260930.md5 && cd scripts && { j=$(sbatch --parsable --wait -p day -t 01:00:00 -c 1 --mem=4G -J bbl_relaunch -o logs/bbl_relaunch_%j.out --wrap "bash bbl_relaunch_20260930.sh"); cat "logs/bbl_relaunch_${j%%;*}.out"; }
#
# The submission waits for this job (sbatch --wait, a few minutes) and then prints its log.
# Ctrl-C stops the waiting only; the log stays in logs/bbl_relaunch_<jobid>.out.
#
#   0. CHECK THE TREE. The run executes files that this upload does not carry (the 2026-09-28 and
#      2026-09-29 uploads, and the forward curves of 2026-09-25): each must have the md5 pinned
#      below, the one the relaunch was tested with. A mismatch refuses before anything is
#      cancelled or moved. bbl_transitions.json is reported, not pinned.
#   1. CANCEL what is left of the superseded runs (_ms981, _sc981 and the probes _probe250phi,
#      _probe250sc) through bbl_cancel.sh: tables and archive, then solves, then sweeps, then fwd
#      arrays, plus any bbl_polfunc / bbl_warmup job (a polfunc job would overwrite the policy
#      installed below). A no-op when nothing is queued. Refuses if any of them is still live.
#   2. INSTALL THE POLICY. The upload data/input/polfunc_fitted_liverows_20260930.csv must have the
#      sha256 pinned below. The step folder's polfunc_* family (the 2026-09-29 cluster fit on all
#      rows) is moved to data/output/bbl/_archive/polfunc_superseded_20260930/, each file renamed
#      <name>.superseded, and the upload is copied to data/output/bbl/polfunc_fitted.csv (temporary +
#      rename), which must then hash to the same sha256. That one file is what both runs read
#      (--no-polfunc, POLICY_CSV), what the sweep compares every shard with, and what the slim
#      archive ships for cluster_ingest_bbl_cf.py to compare byte for byte with the local fit.
#      Already installed (a re-run): kept as it is.
#   3. DRY-RUN BOTH LAUNCHES (CL_DRYRUN=1), then submit them: the multi-start run _ms982 (E3 + E4;
#      sweep -> solve per routine -> tables -> archive) and the single-curve run _sc982 (E3 + E4;
#      sweep -> solve -> archive; no tables job, the tables are built locally). Same design as
#      _ms981: beta 0.979, T 250, evolving phi, mean-reverting Z, lagged carry; gpu_h100, PACK 2 x
#      230G, 300 shards, --shard-time 01:45:00. No probe.
#   4. PRINT every job id and the status commands.
#
# Re-running it is safe: nothing is cancelled twice, an installed policy is kept, and bbl_run.sh
# refuses a second launch of a tag whose jobs are live. CL_DRYRUN=1 in the environment makes the
# whole script a dry run (the local test runs it that way).
# ==============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}" || exit 2
. ./cluster_lib.sh

POLICY_SHA256="fd208b5fc57b9697eefc010ed4a1076af776d9510b89f3bbc1edca009dc29728"
POLICY_UPLOAD="${CL_DATA_IN}/polfunc_fitted_liverows_20260930.csv"
POLICY_CSV="${CL_STEP_BBL}/polfunc_fitted.csv"
SUPERSEDED_DIR="${CL_STEP_BBL}/_archive/polfunc_superseded_20260930"
OLD_TAGS="_ms981 _sc981 _probe250phi _probe250sc"
MS_TAG="_ms982"; SC_TAG="_sc982"
DESIGN=(--routines "3 4" --fwd-gpu --partitions gpu_h100 --shards 300 --beta 0.979 --horizon 250
        --phi-path evolving --z-path mean_reverting --rdep-timing lagged --no-warmup --no-polfunc
        --pack-h100 2 --mem-h100 230G --shard-time 01:45:00)
# RELAUNCH_BBL_EXTRA: extra bbl_run.sh flags for both runs (the local test passes --skip-preflight).
read -r -a EXTRA <<< "${RELAUNCH_BBL_EXTRA:-}"
MS_FLAGS=("${DESIGN[@]}" --multi-start --n-paths 1 --psi-tag "${MS_TAG}" ${EXTRA[@]+"${EXTRA[@]}"})
SC_FLAGS=("${DESIGN[@]}" --psi-tag "${SC_TAG}" --no-tables ${EXTRA[@]+"${EXTRA[@]}"})
DRY="${CL_DRYRUN:-0}"
LOGD="${RELAUNCH_LOG_DIR:-$(cl_log_dir)}"      # the local test points it at a temporary folder
STAMP="$(date +%Y%m%d_%H%M%S)"

say () { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
die () { printf '%s  RELAUNCH REFUSED: %s\n' "$(date '+%F %T')" "$*"; exit 1; }
sha () { sha256sum "$1" | cut -d' ' -f1; }
md5 () { md5sum "$1" | cut -d' ' -f1; }

# The files the run executes that this upload does not carry, as the relaunch was tested with
# them: <where>:<name>:<md5>, where = scripts (this folder) | input (data/input).
PINNED=(
    scripts:bbl_job.sh:d2942824c1041a86e94f4f6aaf9fcf5a
    scripts:cluster_lib.sh:8bcd02f43cf3337c040fe0db3ba807a2
    scripts:bbl_fwd_sim.jl:ddd2e5d179422e436879ad645da9be91
    scripts:cf_phi_path.jl:09666d856c163c64515da7ed0160d7ee
    scripts:cf_deposit_sim.jl:ac72c23a3e9b36b66bbff75f35be94ba
    scripts:cf_psi_basis.jl:1669a68ce682e0517415d4b0fa40097a
    scripts:bbl_solve.py:ebec66d2717f447aee7ee7c86ae5c71d
    input:sleep_link_E3_spec_12.json:14b73a495175b282226a23e14713649b
    input:sleep_link_E4_spec_12.json:ef594035f1d0b21a04e878415265be3b
    input:forward_rf_vintages.csv:13f321168f7bbd6e29f76b2b3f6e3b37
    input:forward_rf_qoq.csv:e544d34079c60c8169f0995faa171873
)
TRANSITIONS_MD5="621c621f40f9f615fa0864259bc4f60d"   # the local COST_FWD/bbl_transitions.json

say "bbl_relaunch_20260930: data root ${CL_DATA_ROOT}$([[ "${DRY}" == "1" ]] && echo '  [DRY RUN: nothing is cancelled, moved or submitted]')"

# ── 0. the tree the run executes ─────────────────────────────────────────────────────────────
say "0. the files this upload does not carry (${#PINNED[@]} pinned)"
bad=""
for p in "${PINNED[@]}"; do
    IFS=: read -r where name want <<< "${p}"
    f="${HERE}/${name}"; [[ "${where}" == "input" ]] && f="${CL_DATA_IN}/${name}"
    if [[ ! -f "${f}" ]]; then bad="${bad} ${name}(missing)"
    elif [[ "$(md5 "${f}")" != "${want}" ]]; then bad="${bad} ${name}(md5 $(md5 "${f}"), want ${want})"
    fi
done
[[ -z "${bad}" ]] || die "the tree is not the one the relaunch was tested with:${bad}. Nothing was cancelled or moved. Those files come from bbl_update_20260928.zip / bbl_update_20260929.zip (scripts, sleep links) and bbl_update_20260925.zip (forward curves); re-upload them, then run this again."
say "   all ${#PINNED[@]} match"
tr_md5="$( [[ -f "${CL_DATA_IN}/bbl_transitions.json" ]] && md5 "${CL_DATA_IN}/bbl_transitions.json" || echo missing)"
if [[ "${tr_md5}" == "${TRANSITIONS_MD5}" ]]; then say "   bbl_transitions.json: the local version (md5 ${tr_md5})"
else say "   [note] bbl_transitions.json md5 ${tr_md5}, the local one is ${TRANSITIONS_MD5}: every psi file records the sha256 of the one it read"; fi

# ── 1. cancel what is left of the superseded runs ────────────────────────────────────────────
say "1. superseded runs: ${OLD_TAGS}"
for t in ${OLD_TAGS}; do
    shared=""; [[ "${t}" == "_ms981" ]] && shared="--with-shared"
    cargs=(--routines "3 4" --psi-tag "${t}" ${shared})
    [[ "${DRY}" == "1" ]] && cargs+=(--dry-run)
    bash "${HERE}/bbl_cancel.sh" "${cargs[@]}" || die "bbl_cancel.sh failed for ${t} (above)"
done
if [[ "${DRY}" != "1" ]]; then
    left="$(squeue -h -u "${USER:-$(id -un)}" -t PENDING,RUNNING,CONFIGURING,SUSPENDED,REQUEUED -o '%i %j' 2>/dev/null \
            | awk '$2 ~ /(_ms981|_sc981|_probe250phi|_probe250sc)$/ || $2 == "bbl_polfunc" || $2 == "bbl_warmup"')"
    [[ -z "${left}" ]] || die "still queued after the cancel: $(tr '\n' ';' <<< "${left}"). Wait for them to leave the queue and run this again."
    say "   no job of ${OLD_TAGS// /, } (or bbl_polfunc / bbl_warmup) is queued"
fi

# ── 2. install the policy ────────────────────────────────────────────────────────────────────
say "2. policy: ${POLICY_UPLOAD##*/} -> ${POLICY_CSV}"
[[ -f "${POLICY_UPLOAD}" ]] || die "the upload ${POLICY_UPLOAD} is missing (unzip bbl_update_20260930.zip in dep_comp)"
got="$(sha "${POLICY_UPLOAD}")"
[[ "${got}" == "${POLICY_SHA256}" ]] || die "${POLICY_UPLOAD} has sha256 ${got}, not the pinned ${POLICY_SHA256}"
if [[ -f "${POLICY_CSV}" && "$(sha "${POLICY_CSV}")" == "${POLICY_SHA256}" ]]; then
    say "   already installed (sha256 ${POLICY_SHA256:0:12}); kept"
elif [[ "${DRY}" == "1" ]]; then
    say "   [dry-run] would move the step folder's polfunc_* to ${SUPERSEDED_DIR}/ and install the upload"
else
    mkdir -p "${SUPERSEDED_DIR}" || die "cannot create ${SUPERSEDED_DIR}"
    n=0
    for f in "${CL_STEP_BBL}"/polfunc_*; do
        [[ -f "${f}" ]] || continue
        mv "${f}" "${SUPERSEDED_DIR}/${f##*/}.superseded" || die "cannot move ${f}"
        n=$(( n + 1 ))
    done
    say "   moved ${n} superseded polfunc_* file(s) to ${SUPERSEDED_DIR}/"
    tmp="${CL_STEP_BBL}/.tmp_polfunc_fitted.csv.$$"
    cp "${POLICY_UPLOAD}" "${tmp}" && mv "${tmp}" "${POLICY_CSV}" || { rm -f "${tmp}"; die "cannot install ${POLICY_CSV}"; }
    got="$(sha "${POLICY_CSV}")"
    [[ "${got}" == "${POLICY_SHA256}" ]] || die "the installed ${POLICY_CSV} has sha256 ${got}, not ${POLICY_SHA256}"
    say "   installed; sha256 ${got}"
fi
if [[ "${DRY}" != "1" || -f "${POLICY_CSV}" ]]; then
    cl_check_policy_units "${POLICY_CSV}" || die "the installed policy fails the lhs_unit check (above)"
    say "   lhs_unit = spread_ann_frac on every row"
fi
# The dry run of an uninstalled root reads the upload itself (same bytes, same sha256).
RUN_POLICY="${POLICY_CSV}"
[[ -f "${RUN_POLICY}" ]] || RUN_POLICY="${POLICY_UPLOAD}"

# ── 3. dry-run both launches, then submit them ────────────────────────────────────────────────
launch () {   # launch <label> <log> <flags...>: bbl_run.sh with this run's policy; its output to <log>
    local label="$1" log="$2"; shift 2
    env POLICY_CSV="${RUN_POLICY}" CL_DRYRUN="${LAUNCH_DRY}" bash "${HERE}/bbl_run.sh" "$@" > "${log}" 2>&1
}
say "3. dry runs (CL_DRYRUN=1): ${MS_TAG}, ${SC_TAG}"
LAUNCH_DRY=1
for pair in "${MS_TAG}:MS_FLAGS" "${SC_TAG}:SC_FLAGS"; do
    tag="${pair%%:*}"; arr="${pair#*:}[@]"
    lg="${LOGD}/bbl_relaunch_${STAMP}_dry${tag}.log"
    launch "${tag}" "${lg}" "${!arr}" || { tail -25 "${lg}"; die "the dry run of ${tag} was refused (${lg})"; }
    say "   ${tag}: dry run OK ($(grep -c '\[dry-run\] sbatch' "${lg}") jobs would be submitted)"
done

LAUNCH_DRY="${DRY}"
ids_of () { sed -n 's/^BBL_ALL_JOBIDS=//p' "$1" | tail -1; }
names_of () { sed -n 's/^BBL_JOB_NAMES=//p' "$1" | tail -1; }
LOG_MS="${LOGD}/bbl_relaunch_${STAMP}${MS_TAG}.log"; LOG_SC="${LOGD}/bbl_relaunch_${STAMP}${SC_TAG}.log"
say "   submitting ${MS_TAG} (multi-start) ..."
if ! launch "${MS_TAG}" "${LOG_MS}" "${MS_FLAGS[@]}"; then
    tail -40 "${LOG_MS}"
    die "the ${MS_TAG} launch failed (${LOG_MS}); bbl_run.sh withdrew whatever it had submitted. Nothing of ${SC_TAG} was submitted."
fi
say "   ${MS_TAG}: $(names_of "${LOG_MS}")"
say "   ${MS_TAG} job ids: $(ids_of "${LOG_MS}")"
say "   submitting ${SC_TAG} (single curve) ..."
if ! launch "${SC_TAG}" "${LOG_SC}" "${SC_FLAGS[@]}"; then
    tail -40 "${LOG_SC}"
    printf 'RELAUNCH PARTIAL: %s IS SUBMITTED (ids %s); %s WAS NOT (%s).\n' "${MS_TAG}" "$(ids_of "${LOG_MS}")" "${SC_TAG}" "${LOG_SC}"
    printf '  After fixing the cause, submit it alone:\n'
    printf '  cd %s && sbatch -p day -t 00:30:00 -c 1 --mem=4G -J bbl_submit -o logs/bbl_submit_%%j.out --wrap "POLICY_CSV=%s bash bbl_run.sh %s"\n' \
        "${HERE}" "${POLICY_CSV}" "$(printf '%q ' "${SC_FLAGS[@]}")"
    exit 1
fi
say "   ${SC_TAG}: $(names_of "${LOG_SC}")"
say "   ${SC_TAG} job ids: $(ids_of "${LOG_SC}")"

# ── 4. summary ───────────────────────────────────────────────────────────────────────────────
echo
echo "RELAUNCH SUBMITTED$([[ "${DRY}" == "1" ]] && echo ' [DRY RUN]'): policy ${POLICY_CSV} (sha256 ${POLICY_SHA256:0:12})"
echo "  ${MS_TAG} multi-start  E3+E4: $(ids_of "${LOG_MS}")"
echo "  ${SC_TAG} single curve E3+E4: $(ids_of "${LOG_SC}")"
echo "  full submission logs: ${LOG_MS}"
echo "                        ${LOG_SC}"
echo "STATUS (one line):"
echo "  cd ${HERE} && bash bbl_status.sh --routines \"3 4\" --psi-tag ${MS_TAG}; bash bbl_status.sh --routines \"3 4\" --psi-tag ${SC_TAG}"
