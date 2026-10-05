#!/bin/bash
# ==============================================================================
# dx_suite_20261001.sh -- the `dx` robustness variant of the demand side on the cluster: the
# D-type dummy in the linear part of the BLP (blp_dx.jl), then the BBL cost estimation on that
# fit. Routines E3 + E4. Everything in one submission from the scripts folder (one line; the
# upload's md5 list is checked first):
#
#   cd ~/project_pi_mf2263/pf382/dep_comp && unzip -o ~/dx_update_20261001.zip && md5sum -c dx_update_20261001.md5 && cd scripts && { j=$(sbatch --parsable --wait -p day -t 01:00:00 -c 1 --mem=4G -J dx_suite -o logs/dx_suite_%j.out --wrap "bash dx_suite_20261001.sh"); cat "logs/dx_suite_${j%%;*}.out"; }
#
# The submission waits for this job (sbatch --wait, a few minutes) and then prints its log.
# Ctrl-C stops the waiting only; the log stays in logs/dx_suite_<jobid>.out.
#
# The variant ADDS files and changes none: its Julia entry point (blp_dx_rc.jl) include()s the
# engine and wraps it from outside, its worker is blp_dx_stage_job.sh, its BBL launcher is
# dx_bbl_run.sh (derived from bbl_run.sh at run time), and every file it writes carries `_dx`
# (data/output/blp/*_dx.* and blp_bblfit_*_dx_ms982.json; data/output/bbl/*_extended_dx_ms982*).
# The main specification's results, checkpoints, psi and cost parameters are read, never written.
#
# THE BBL IS BOUND TO THE FIT IT WAS SIMULATED ON. The forward simulation reads
# blp_results_E{k}_spec_12_extended_dx.jls and records nothing about it, so files of a BBL run
# cannot say which fit they came from. When the hand-off submits a fresh BBL launch it writes
#     data/output/blp/blp_bblfit_E{k}_spec_12_extended_dx_ms982.json     (the sha256 of that fit)
# and from then on every launch, hand-off and archive compares the fit on disk with that record
# whenever files of the run exist (cost parameters, psi files, a launch context). Files of a BBL
# run with no fit on disk, with no record, or with a record of another fit are REFUSED: nothing is
# submitted, nothing is deleted, and the message names what to archive and remove.
# The record also lists the job id of every archive job (dx_bbl_zip) the suite sequenced for the
# run. That id is in the archive's name (bbl_outputs_<jobid>_dx.zip), and it is the one thing a
# BBL archive carries that says which launch it comes from: the cost parameters and the psi files
# record no job id, launch time or fit. The local ingest and the tables use it to tie the cost
# parameters to the record, as the sha256 ties the record to the fit.
#
# NEVER REPAIR THE VARIANT BY HAND WITH bbl_run.sh. bbl_status.sh, the sweep job and bbl_solve.py
# print a line `bash bbl_run.sh ... --psi-tag '_dx_ms982' --repair` when shards are missing. Run
# for the variant, that line skips the binding above, submits the tables job (which overwrites the
# step folder's untagged tab_bbl_* with the variant's tables) and an archive tagged with a bare
# job id (the kind the main ingests prefer). Run the launch line again instead: the suite's own
# repair passes --no-tables --no-zip, checks the binding first and archives under <jobid>_dx.
#
#   0. CHECK THE TREE. The run executes files this upload does not carry: each must have the md5
#      pinned below, the one the variant was tested with. A mismatch refuses before anything is
#      submitted. bbl_run.sh is not in that list: dx_bbl_run.sh pins it itself.
#   1. CHECK THE QUEUE. Refuses if the queue cannot be read, if a job of the variant is live (a
#      second launch), and if a job of the main BBL runs _ms982 / _sc982 is live: the variant's BBL
#      shares their GPU partition and QOS, so it waits for them unless DX_WITH_LIVE_MAIN=1.
#   2. CHECK THE INPUTS the ladder reads and no job of this graph produces: the demand parquets,
#      the logit delta, the draws, the GPU sysimage. The logit sub-model the variant nests
#      (E{k}_full_dtype in data/output/logit/logit_summary_spec_12.json) is reported.
#   3. THE MAIN RESULTS must be on disk (ext1 and extended, both routines: the hand-off compares
#      with them); their sha256 and that of the _ms982 / _sc982 cost parameters go into
#      logs/dx_suite_<stamp>_main.sha256, which the hand-off verifies before it submits anything.
#   4. THE STATE OF THE VARIANT, per routine, the ladder and the BBL TOGETHER: which ladders are
#      left to run (a routine whose ext1 and extended results are on disk gets no ladder), and
#      whether the files of its BBL run are those of the fit on disk (the binding above; a
#      routine with BBL files and no extended fit is an error, not a finished routine). When no
#      ladder is left and the bound cost parameters exist for every routine there is nothing to
#      do and nothing is submitted.
#   5. THE BBL, CHECKED NOW rather than hours later at the hand-off: the standard errors of ext1
#      for the routines whose ladder is on disk (the rule of H1), the policy (sha256 pinned),
#      bbl_transitions.json (md5 reported), cluster_preflight.sh (what a live bbl_run.sh runs
#      first), and a dry run of the very launch or repair the hand-off will make; every
#      forward-simulation line of a launch must carry `--suffix _dx --psi-tag _ms982`. A launch
#      that would end in "BBL NOT SUBMITTED" is REFUSED here, before any GPU job, unless
#      DX_BLP_ONLY=1 asks for the ladders and the comparison alone.
#   6. SUBMIT THE BLP LADDERS, dry run first: per routine
#          rc_ift_dx_E{k}_head (sigma+rc2+rc3+rc4+ext1) -> rc_ift_dx_E{k}_ext2 -> rc_ift_dx_E{k}_ext
#      afterok chains on the GPU IFT engine, a cold ladder: no stage is seeded from the main
#      specification. rc2-rc4, ext2 and extended start theta2 from the variant's previous
#      checkpoint; ext1 starts from the seed (the engine looks for a `full` checkpoint this ladder
#      does not write), exactly as it did in the main run. Each job first runs the theta2 = 0
#      nesting check against the logit. Every rung is submitted with DX_SMOKE=0 and with this
#      launch's DX_ALLOW_NO_SE.
#   7. SUBMIT THE HAND-OFF dx_next_bbl (afterok the ladders; `bash dx_suite_20261001.sh
#      --handoff`) and the archive dx_blp_zip (afterany EVERY ladder's last rung and the
#      hand-off, so a ladder that fails early does not fire it while the other still runs; sets
#      blp + logs, --copy, tag <jobid>_dx).
#   8. PRINT every job id and the status commands.
#
# The hand-off (H):
#   H0. the pinned files again, and the recorded sha256 of the main results (unchanged).
#   H1. the variant's ext1 and extended results exist for both routines, and the files of each
#       routine's BBL run, if any, are bound to the extended fit on disk (refused otherwise,
#       before the comparison is written).
#   H2. MAIN vs DX, printed to the log (blp_dx_compare.jl): alpha, the D-type coefficient, Q,
#       every theta2, and the distribution of delta_dx - delta_main, at ext1 and extended.
#   H3. THE BBL, multi-start only, on the variant's `extended` fit, tag _dx_ms982: the _ms982
#       design (beta 0.979, T 250, evolving phi, mean-reverting Z, lagged carry; gpu_h100,
#       PACK 2 x 230G, 300 shards, --shard-time 01:45:00), --no-polfunc on the installed policy,
#       --no-tables. What is submitted depends on what is on disk, per routine:
#         cost_params_E{k}_spec_12_extended_dx_ms982.json exists   nothing: the routine is done
#         psi files or a launch context exist, no cost_params       bbl_run.sh --repair: the
#                                             sweep re-runs only the missing shards, then solves
#         neither                                                   a fresh launch; the sha256 of
#                                             the fit is recorded first (blp_bblfit_*.json)
#       The first two need the fit record to match the fit on disk (H1). THE STANDARD ERRORS OF
#       ext1: a routine that would get BBL jobs must have joint standard errors at ext1 (its
#       sidecar says se_reported). That is the rule the rung itself applies (blp_dx_rc.jl ends
#       nonzero without them), applied here to results already on disk. DX_ALLOW_NO_SE=1 accepts
#       a routine without them, in the rungs and here: it is flagged in the log and in the
#       sidecar, and the tables print no standard error for its RC + D column.
#       A fresh launch goes through dx_bbl_run.sh when it is in this folder and bbl_run.sh is the
#       shipped file, otherwise through bbl_run.sh with DEMAND_SUFFIX=_dx. Launch and repair are
#       dry-run first (the checks of step 5 again). A launcher that does not pass the suffix would
#       simulate on the MAIN fit under the variant's tag, so the hand-off then submits no BBL
#       job, says so, and exits 4. If the repair is submitted and the fresh launch then fails,
#       the archive of H4 is still sequenced on the repair's solves and the hand-off exits 5.
#   H4. The archive dx_bbl_zip, afterok the solves: first `--check-fit` (the binding once more,
#       now that the solves have written: a fit replaced while the BBL ran stops the archive),
#       then set bbl, --slim on _dx_ms982, --copy, tag <jobid>_dx (bbl_outputs_<jobid>_dx.zip: a
#       tag that is not a number, so the main ingest's newest-archive rule never prefers it to an
#       archive of the main runs).
#
# RE-RUNNING. Step 1 refuses while a job of the variant is live. Otherwise a second run submits
# only what is left: ladders for the routines without their ext1 and extended results (the engine
# skips the stages already on disk, so a fit on disk is never rewritten), a repair for a BBL that
# left psi files but no cost parameters, nothing for a routine whose cost parameters exist; the
# last two only when the files are bound to the fit on disk. It never starts a fresh BBL launch
# over existing shards, and it never overwrites cost parameters.
#   To re-SOLVE a finished routine on the shards on disk (same fit): move its
#   cost_params_*_dx_ms982.json away, then run this again (it repairs).
#   To re-ESTIMATE a routine's demand fit after its BBL was launched: archive and remove ALL of
#   that routine's BBL files first (cost_params_<key>.json, psi_eq_/psi_dev_/psi_starts_<key>*,
#   data/output/bbl/.dispatch/<key>/, data/output/blp/blp_bblfit_<key>.json, with
#   <key> = E{k}_spec_12_extended_dx_ms982), then the variant's blp_results/blp_checkpoint files
#   of the routine, then run this again. Removing only the fit, or only the cost parameters, is
#   refused: the shards on disk would be solved, or reported, under a fit they did not come from.
#
# Options: none for a launch; --handoff is the launch's own continuation; --check-fit verifies
# the binding of every BBL run on disk and submits nothing (the archive job runs it).
# Environment: CL_DRYRUN=1 makes the whole script a dry run, hand-off included; it submits
# nothing and writes nothing. DX_BLP_ONLY=1: ladders and comparison, no BBL. DX_WITH_LIVE_MAIN=1:
# queue behind live main BBL jobs. DX_BLP_PARTITION=gpu_h100: the ladders' partition.
# DX_ALLOW_NO_SE=1: accept ext1 without joint standard errors (above). DX_SMOKE must NOT be set:
# it belongs to the smoke job (blp_dx_stage_job.sh) and a launch that sees it is refused.
# ==============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}" || exit 2
CL_VERBOSE=1
. ./cluster_lib.sh

MODE=launch
case "${1:-}" in
    "")          ;;
    --handoff)   MODE=handoff ;;
    --check-fit) MODE=checkfit ;;
    *) echo "unknown option: $1 (a launch takes no argument; --handoff is its own continuation; --check-fit verifies the BBL-to-fit binding)" >&2; exit 2 ;;
esac

SELF="dx_suite_20261001.sh"
ROUTINES="3 4"
DX="_dx"
MAIN_TAGS="_ms982 _sc982"
DX_TAG="${DX}_ms982"
SE_STAGE="ext1"; BBL_STAGE_NAME="extended"
HEAD_STAGES="sigma+rc2+rc3+rc4+ext1"
BLP_PARTITION="${DX_BLP_PARTITION:-gpu_h200}"
BLP_ONLY="${DX_BLP_ONLY:-0}"
ALLOW_NO_SE="${DX_ALLOW_NO_SE:-0}"
POLICY_SHA256="fd208b5fc57b9697eefc010ed4a1076af776d9510b89f3bbc1edca009dc29728"
POLICY_CSV="${CL_STEP_BBL}/polfunc_fitted.csv"
TRANSITIONS_MD5="621c621f40f9f615fa0864259bc4f60d"   # the local COST_FWD/bbl_transitions.json
# A fresh launch: the _ms982 design. --no-zip because the suite archives under its own tag (H4).
BBL_DESIGN=(--fwd-gpu --partitions gpu_h100 --shards 300 --beta 0.979 --horizon 250
            --phi-path evolving --z-path mean_reverting --rdep-timing lagged --no-warmup --no-polfunc
            --pack-h100 2 --mem-h100 230G --shard-time 01:45:00
            --multi-start --n-paths 1 --psi-tag "${DX_TAG}" --no-tables --no-zip)
# A repair: the design comes from the run's context; these are the placement and the tag.
BBL_REPAIR_FLAGS=(--shards 300 --psi-tag "${DX_TAG}" --repair --partitions gpu_h100 --pack-h100 2
                  --mem-h100 230G --shard-time 01:45:00 --no-tables --no-zip)
# DX_BBL_EXTRA: extra bbl_run.sh flags (the local test passes --skip-preflight).
read -r -a EXTRA <<< "${DX_BBL_EXTRA:-}"
DRY="${CL_DRYRUN:-0}"
LOGD="${DX_LOG_DIR:-$(cl_log_dir)}"            # the local test points it at a temporary folder
STAMP="${DX_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MAIN_SHA_FILE="${DX_MAIN_SHA:-${LOGD}/dx_suite_${STAMP}_main.sha256}"

say () { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
die () { printf '%s  DX SUITE REFUSED: %s\n' "$(date '+%F %T')" "$*"; exit 1; }
sha () { sha256sum "$1" | cut -d' ' -f1; }
md5 () { md5sum "$1" | cut -d' ' -f1; }

# The files the suite executes that this upload does not carry, as the variant was tested with
# them: <where>:<name>:<md5>, where = scripts (this folder) | input (data/input).
PINNED=(
    scripts:cluster_lib.sh:8bcd02f43cf3337c040fe0db3ba807a2
    scripts:blp_rc.jl:3330f97fd5b0fd8f8e5ce1ad72379de5
    scripts:blp_engine_gpu.jl:e39d409886cc97c95d2fd7ee7222580f
    scripts:blp_engine_cpu.jl:e8329ee9d0036b4e1a82e97393baab45
    scripts:blp_se_common.jl:9babc942eb8d3c99319a6acf96665a3a
    scripts:of_root.jl:d2e0f74314ccb52fd6b7e3641f3afda5
    scripts:routines.jl:173021e492a1833ecbede35fcf227b80
    scripts:config/routines.toml:5b08b9906ffa68d85922044965cfec4e
    scripts:cluster_archive.sh:1cb1ae2139e76314be1bf2f9eef5293e
    scripts:cluster_preflight.sh:26fc62671658454a5c530ae766268e6c
    scripts:bbl_job.sh:d2942824c1041a86e94f4f6aaf9fcf5a
    scripts:bbl_fwd_sim.jl:ddd2e5d179422e436879ad645da9be91
    scripts:cf_demand_eval.jl:bf8fdade17cfa9a88dc80703f1de8a6f
    scripts:cf_phi_path.jl:09666d856c163c64515da7ed0160d7ee
    scripts:cf_deposit_sim.jl:ac72c23a3e9b36b66bbff75f35be94ba
    scripts:cf_psi_basis.jl:1669a68ce682e0517415d4b0fa40097a
    scripts:bbl_solve.py:ebec66d2717f447aee7ee7c86ae5c71d
    scripts:bbl_shards.py:12780a1476ade0ffc7cca880ff005c81
    input:sleep_link_E3_spec_12.json:14b73a495175b282226a23e14713649b
    input:sleep_link_E4_spec_12.json:ef594035f1d0b21a04e878415265be3b
    input:forward_rf_vintages.csv:13f321168f7bbd6e29f76b2b3f6e3b37
    input:forward_rf_qoq.csv:e544d34079c60c8169f0995faa171873
)
BBL_RUN_MD5="33c97ed1e56cdfd5368fae0187e9fb4f"     # bbl_run.sh as shipped 2026-09-30

check_pins () {   # refuses on the first tree that is not the tested one
    local bad="" p where name want f
    for p in "${PINNED[@]}"; do
        IFS=: read -r where name want <<< "${p}"
        f="${HERE}/${name}"; [[ "${where}" == "input" ]] && f="${CL_DATA_IN}/${name}"
        if [[ ! -f "${f}" ]]; then bad="${bad} ${name}(missing)"
        elif [[ "$(md5 "${f}")" != "${want}" ]]; then bad="${bad} ${name}(md5 $(md5 "${f}"), want ${want})"
        fi
    done
    [[ -z "${bad}" ]] || die "the tree is not the one the variant was tested with:${bad}. Nothing was submitted. Send this line back before changing anything on the cluster."
    say "   all ${#PINNED[@]} match"
}

# The variant's result of a routine and stage, and the main specification's.
dx_res ()   { printf '%s/blp_results_E%s_spec_12_%s%s.jls' "${CL_STEP_BLP}" "$1" "$2" "${DX}"; }
main_res () { printf '%s/blp_results_E%s_spec_12_%s.jls' "${CL_STEP_BLP}" "$1" "$2"; }

# The queue, read once and FAIL-CLOSED: when squeue fails the suite cannot tell whether jobs are
# live, and says so instead of reading an empty answer as an empty queue.
QUEUE=""
read_queue () {
    QUEUE="$(squeue -h -u "${USER:-$(id -un)}" -t PENDING,RUNNING,CONFIGURING,SUSPENDED,REQUEUED -o '%i %j')" \
        || die "squeue failed, so the queue cannot be read and the suite cannot tell whether jobs of the variant or of the main runs are live. Nothing was submitted; run it again when squeue answers."
}
live_jobs () { awk "$1" <<< "${QUEUE}"; }   # live_jobs <awk condition on the job name $2>

# The first value of a string field in a JSON file, whatever its layout (the suite's own records
# have one field per line; the Julia sidecars are one line).
json_str () {   # json_str <file> <field>
    grep -o "\"$2\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$1" 2>/dev/null | head -1 | sed 's/^"[^"]*"[[:space:]]*:[[:space:]]*"//; s/"$//'
}

# ── the BBL of the variant: what is on disk, per routine, and the fit it is bound to ─────────
bbl_key ()  { printf 'E%s_spec_12_%s%s' "$1" "${BBL_STAGE_NAME}" "${DX_TAG}"; }
fit_path () { dx_res "$1" "${BBL_STAGE_NAME}"; }                       # what the forward simulation reads
fit_rec ()  { printf '%s/blp_bblfit_%s.json' "${CL_STEP_BLP}" "$(bbl_key "$1")"; }
# The psi files of a run, by their whole names: psi_{eq,dev,starts}_<key>[_shard<i>of<n>].{parquet,json}.
# Another tag that merely starts with this one (<key>probe, <key>0) is not this run's.
bbl_psi_list () {   # bbl_psi_list <key>
    ls "${CL_STEP_BBL}" 2>/dev/null | grep -E "^psi_(eq|dev|starts)_$1(_shard[0-9]+of[0-9]+)?\.(parquet|json)$" || true
}

is_sha256 () { [[ "$1" =~ ^[0-9a-f]{64}$ ]]; }

# The record of the fit a FRESH launch is about to be simulated on. Written before anything of
# the BBL is submitted, and only for a routine with no file of the run on disk (bbl_classify:
# fresh). Every hash must be 64 hex characters, when computed and again when read back from the
# file: a record with an empty or cut hash would bind the run to nothing. Returns 1 otherwise.
fit_record_write () {   # fit_record_write <routine>
    local k="$1" f rec js h hj=""
    f="$(fit_path "${k}")"; rec="$(fit_rec "${k}")"; js="${f%.jls}.json"
    h="$(sha "${f}")"; is_sha256 "${h}" || return 1
    if [[ -f "${js}" ]]; then hj="$(sha "${js}")"; is_sha256 "${hj}" || return 1; fi
    {
        printf '{\n'
        printf ' "bbl_key": "%s",\n' "$(bbl_key "${k}")"
        printf ' "fit": "%s",\n' "${f##*/}"
        printf ' "fit_sha256": "%s",\n' "${h}"
        if [[ -n "${hj}" ]]; then
            printf ' "fit_json": "%s",\n' "${js##*/}"
            printf ' "fit_json_sha256": "%s",\n' "${hj}"
        fi
        printf ' "recorded": "%s",\n' "$(date '+%F %T')"
        printf ' "recorded_by": "%s --handoff%s",\n' "${SELF}" "${SLURM_JOB_ID:+, job ${SLURM_JOB_ID}}"
        printf ' "bbl_archive_jobs": ""\n'
        printf '}\n'
    } > "${rec}.tmp" || return 1
    mv -f "${rec}.tmp" "${rec}" || return 1
    [[ "$(json_str "${rec}" fit_sha256)" == "${h}" && "$(json_str "${rec}" bbl_key)" == "$(bbl_key "${k}")" ]] || return 1
    [[ -z "${hj}" || "$(json_str "${rec}" fit_json_sha256)" == "${hj}" ]]
}

# The archive job the suite has just sequenced for a run, added to the run's record: its job id
# is in the name of the archive it will write (bbl_outputs_<jobid>_dx.zip). The hashes stay as
# they are; a job id already listed is not listed twice. Returns 1 when the record cannot take it.
fit_record_add_archive () {   # fit_record_add_archive <routine> <archive job id>
    local rec jobs h
    rec="$(fit_rec "$1")"
    [[ -f "${rec}" && "$2" =~ ^[0-9]+$ ]] || return 1
    grep -q '^ "bbl_archive_jobs": "[0-9 ]*"$' "${rec}" || return 1
    h="$(json_str "${rec}" fit_sha256)"
    jobs="$(json_str "${rec}" bbl_archive_jobs)"
    [[ " ${jobs} " == *" $2 "* ]] && return 0
    jobs="${jobs:+${jobs} }$2"
    sed 's/^ "bbl_archive_jobs": "[0-9 ]*"$/ "bbl_archive_jobs": "'"${jobs}"'"/' "${rec}" > "${rec}.tmp" || return 1
    [[ "$(json_str "${rec}.tmp" fit_sha256)" == "${h}" && "$(json_str "${rec}.tmp" bbl_archive_jobs)" == "${jobs}" ]] || { rm -f "${rec}.tmp"; return 1; }
    mv -f "${rec}.tmp" "${rec}"
}

# bbl_classify: per routine, the BBL run on disk and whether it is the run of the fit on disk.
#   BBL_FRESH   no file of the run (no cost_params, no psi file, no launch context)
#   BBL_DONE    cost_params on disk, and the fit on disk is the recorded one
#   BBL_REPAIR  psi files or a launch context, no cost_params, and the fit on disk is the recorded one
#   BBL_BAD     files of the run that cannot be tied to the fit on disk, one message per routine:
#               the fit is not on disk, the record is not on disk, or the record names another fit
bbl_classify () {
    local k key f rec cp ctx npsi what want got
    BBL_DONE=""; BBL_REPAIR=""; BBL_FRESH=""; BBL_BAD=()
    for k in ${ROUTINES}; do
        key="$(bbl_key "${k}")"; f="$(fit_path "${k}")"; rec="$(fit_rec "${k}")"
        cp=0;  [[ -f "${CL_STEP_BBL}/cost_params_${key}.json" ]] && cp=1
        ctx=0; [[ -f "$(cl_bbl_dispatch_dir "${key}")/context.env" ]] && ctx=1
        npsi="$(bbl_psi_list "${key}" | grep -c . || true)"
        if (( cp == 0 && ctx == 0 && npsi == 0 )); then
            BBL_FRESH="${BBL_FRESH:+${BBL_FRESH} }${k}"; continue
        fi
        what=""
        (( cp ))       && what="${what}, cost_params_${key}.json"
        (( npsi > 0 )) && what="${what}, ${npsi} psi file(s)"
        (( ctx ))      && what="${what}, the launch context .dispatch/${key}/"
        what="${what#, }"
        if [[ ! -f "${f}" ]]; then
            BBL_BAD+=("E${k}: files of the BBL run ${key} are on disk (${what}) but the fit they were simulated on, ${f##*/}, is not")
        elif [[ ! -f "${rec}" ]]; then
            BBL_BAD+=("E${k}: files of the BBL run ${key} are on disk (${what}) without the fit record ${rec##*/}, so nothing shows they were simulated on the ${f##*/} now on disk")
        else
            want="$(json_str "${rec}" fit_sha256)"; got="$(sha "${f}")"
            if [[ "$(json_str "${rec}" bbl_key)" != "${key}" ]] || ! is_sha256 "${want}"; then
                BBL_BAD+=("E${k}: the fit record ${rec##*/} is not a record of the BBL run ${key} (bbl_key '$(json_str "${rec}" bbl_key)', fit_sha256 '${want:0:12}' of ${#want} characters)")
            elif [[ "${want}" != "${got}" ]]; then
                BBL_BAD+=("E${k}: ${f##*/} on disk has sha256 ${got:0:12}..., but the BBL run ${key} on disk (${what}) was simulated on the fit ${want:0:12}... (${rec##*/}, recorded $(json_str "${rec}" recorded)): the fit was re-estimated or replaced after the BBL was launched")
            elif (( cp )); then BBL_DONE="${BBL_DONE:+${BBL_DONE} }${k}"
            else BBL_REPAIR="${BBL_REPAIR:+${BBL_REPAIR} }${k}"; fi
        fi
    done
    return 0
}
bbl_report () {   # the classification, in words; needs bbl_classify
    [[ -z "${BBL_DONE}" ]]   || say "   BBL done (cost_params on disk, bound to the fit on disk; nothing is submitted for it): E${BBL_DONE// /, E}"
    [[ -z "${BBL_REPAIR}" ]] || say "   BBL to REPAIR (psi files or a launch context on disk, no cost_params; bound to the fit on disk): E${BBL_REPAIR// /, E}"
    [[ -z "${BBL_FRESH}" ]]  || say "   BBL to launch (no file of the run on disk): E${BBL_FRESH// /, E}"
    return 0
}
# Dies when files of a BBL run cannot be tied to the fit on disk. Deletes nothing, relaunches nothing.
bbl_require_bound () {   # needs bbl_classify
    local m
    (( ${#BBL_BAD[@]} == 0 )) && return 0
    for m in "${BBL_BAD[@]}"; do say "   [!] ${m}"; done
    die "the variant's BBL files on disk are not bound to the demand fit on disk (the lines above). Nothing was submitted and nothing was deleted. For each routine named, with <key> = E{k}_spec_12_${BBL_STAGE_NAME}${DX_TAG}: EITHER put back the fit its record names, OR archive and then remove the routine's BBL files, which are ${CL_STEP_BBL}/cost_params_<key>.json, ${CL_STEP_BBL}/psi_eq_<key>*, psi_dev_<key>*, psi_starts_<key>*, the folder ${CL_STEP_BBL}/.dispatch/<key>/ and the record ${CL_STEP_BLP}/blp_bblfit_<key>.json; the routine then gets a fresh BBL launch on the fit on disk. Run this again afterwards."
}

# ── the standard errors of the reported stage (ext1), for results already on disk ────────────
# A rung ends nonzero when its ext1 has no joint standard errors, which stops the chain before
# the BBL. The same rule is applied here to results on disk, from the sidecar the rung's
# post-check writes. DX_ALLOW_NO_SE=1 accepts such a routine and says so.
SE_WHY=""
se_policy () {   # se_policy "<routines>": 1 with SE_WHY set when a routine is stopped by the rule
    local k meta why
    SE_WHY=""
    for k in $1; do
        meta="${CL_STEP_BLP}/blp_meta_E${k}_spec_12_${SE_STAGE}${DX}.json"
        if [[ ! -f "${meta}" ]]; then
            why="the sidecar ${meta##*/} is missing (the rung's post-check did not finish), so whether ${SE_STAGE} has joint standard errors is not known"
        elif grep -Eq '"se_reported"[[:space:]]*:[[:space:]]*true' "${meta}"; then
            say "   E${k} ${SE_STAGE}: joint standard errors on record"; continue
        else
            why="${SE_STAGE} has NO joint standard errors ($(json_str "${meta}" se_status))"
        fi
        if [[ "${ALLOW_NO_SE}" == "1" ]]; then say "   [flag] E${k}: ${why}. ACCEPTED under DX_ALLOW_NO_SE=1: the BBL runs on this ladder and the tables print no standard error for its RC + D column"
        else SE_WHY="${SE_WHY:+${SE_WHY}; }E${k}: ${why}"; fi
    done
    [[ -z "${SE_WHY}" ]]
}

# ── the BBL pre-checks: everything a BBL submission from here needs, short of the variant's fit.
# Run at the launch (step 5) and again at the hand-off. Returns 1 with WHY_NOT set when the BBL
# could not be submitted as it should; dies on a wrong policy. Needs bbl_classify first.
LAUNCHER="bbl_run.sh"; WHY_NOT=""; RUN_POLICY="${POLICY_CSV}"
bbl_prechecks () {
    local k key ctx lg m fwd with trm
    WHY_NOT=""
    if [[ -f "${POLICY_CSV}" ]]; then
        [[ "$(sha "${POLICY_CSV}")" == "${POLICY_SHA256}" ]] || die "${POLICY_CSV} has sha256 $(sha "${POLICY_CSV}"), not the pinned ${POLICY_SHA256:0:12}... the main runs used. No BBL job was submitted."
        say "   policy ${POLICY_CSV} (sha256 ${POLICY_SHA256:0:12}), the one _ms982 was simulated around"
    elif [[ "${DRY}" == "1" ]]; then say "   [dry-run] ${POLICY_CSV} is not on disk"
    else WHY_NOT="${POLICY_CSV} is missing"; return 1; fi
    # Reported, not pinned: the main runs read whatever is in data/input, and every psi file
    # records the sha256 of the one it read (make_dx_tables.py compares main and variant).
    if [[ -f "${CL_DATA_IN}/bbl_transitions.json" ]]; then
        trm="$(md5 "${CL_DATA_IN}/bbl_transitions.json")"
        if [[ "${trm}" == "${TRANSITIONS_MD5}" ]]; then say "   bbl_transitions.json: the local version (md5 ${trm})"
        else say "   [note] bbl_transitions.json md5 ${trm}; the local one is ${TRANSITIONS_MD5}. It must be the file the main runs read: the tables compare the sha256 each run recorded"; fi
    elif [[ "${DRY}" == "1" ]]; then say "   [dry-run] ${CL_DATA_IN}/bbl_transitions.json is not on disk"
    else WHY_NOT="${CL_DATA_IN}/bbl_transitions.json is missing"; return 1; fi

    if [[ -n "${BBL_REPAIR}" ]]; then
        for k in ${BBL_REPAIR}; do
            key="$(bbl_key "${k}")"; ctx="$(cl_bbl_dispatch_dir "${key}")/context.env"
            if [[ ! -f "${ctx}" ]]; then WHY_NOT="psi files of ${key} are on disk without the launch context ${ctx}: they cannot be repaired, and a fresh launch would simulate every shard again over them"; return 1; fi
            if ! grep -q "^BBL_FWD_EXTRA=.*--suffix.*${DX}.*--psi-tag" "${ctx}" || ! grep -q "^PSI_TAG=${DX_TAG}\$" "${ctx}"; then
                WHY_NOT="the launch context ${ctx} does not carry the demand suffix (BBL_FWD_EXTRA ... --suffix ${DX}): its shards were not simulated on the variant's fit"; return 1
            fi
        done
        lg="${LOGD}/dx_suite_${STAMP}_bbl_repair_dry.log"
        if ! env POLICY_CSV="${RUN_POLICY}" CL_DRYRUN=1 CL_VERBOSE=1 bash "${HERE}/bbl_run.sh" --routines "${BBL_REPAIR}" "${BBL_REPAIR_FLAGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} > "${lg}" 2>&1; then
            tail -25 "${lg}"; WHY_NOT="the dry run of the BBL repair was refused (${lg})"; return 1
        fi
        say "   repair dry run OK: $(grep -c '\[dry-run\] sbatch' "${lg}") jobs would be submitted ($(sed -n 's/^BBL_JOB_NAMES=//p' "${lg}" | tail -1))"
    fi
    if [[ -n "${BBL_FRESH}" ]]; then
        m="$(md5 "${HERE}/bbl_run.sh")"
        say "   bbl_run.sh md5 ${m}$([[ "${m}" == "${BBL_RUN_MD5}" ]] && echo ' (as shipped 2026-09-30)')"
        # The launcher: dx_bbl_run.sh when it is there and bbl_run.sh is the shipped file it derives
        # from; otherwise bbl_run.sh itself, which then has to honour DEMAND_SUFFIX on its own.
        LAUNCHER="bbl_run.sh"
        if [[ "${m}" == "${BBL_RUN_MD5}" && -f "${HERE}/dx_bbl_run.sh" ]]; then LAUNCHER="dx_bbl_run.sh"; fi
        say "   launcher: ${LAUNCHER}"
        lg="${LOGD}/dx_suite_${STAMP}_bbl_dry.log"
        if ! env DEMAND_SUFFIX="${DX}" POLICY_CSV="${RUN_POLICY}" CL_DRYRUN=1 CL_VERBOSE=1 bash "${HERE}/${LAUNCHER}" --routines "${BBL_FRESH}" "${BBL_DESIGN[@]}" ${EXTRA[@]+"${EXTRA[@]}"} > "${lg}" 2>&1; then
            tail -25 "${lg}"; WHY_NOT="the dry run of the BBL launch through ${LAUNCHER} was refused (${lg})"; return 1
        fi
        # The forward simulation must be told the demand suffix. Each fwd submission carries the
        # sim's command line in BBL_EXTRA; the suffix has to be on every one of them, with the psi
        # tag reduced to the part after it, so the files the sim writes are the ones the sweep and
        # the solve look for under --psi-tag ${DX_TAG}.
        fwd="$(grep -c -- '-J bbl_fwd_E[0-9]*'"${DX_TAG}"' ' "${lg}")"
        with="$(grep -- '-J bbl_fwd_E[0-9]*'"${DX_TAG}"' ' "${lg}" | grep -c -- "--suffix ${DX} --psi-tag ${DX_TAG#"${DX}"}\([ ,]\|$\)")"
        if (( fwd == 0 || with != fwd )); then
            WHY_NOT="${LAUNCHER} does not pass the demand suffix to the forward simulation (${with} of ${fwd} fwd submissions carry '--suffix ${DX} --psi-tag ${DX_TAG#"${DX}"}'): the simulation would read the MAIN fit blp_results_E{k}_spec_12_${BBL_STAGE_NAME}.jls and write it under the variant's tag"
            return 1
        fi
        say "   launch dry run OK: $(grep -c '\[dry-run\] sbatch' "${lg}") jobs would be submitted ($(sed -n 's/^BBL_JOB_NAMES=//p' "${lg}" | tail -1)); every fwd submission carries '--suffix ${DX} --psi-tag ${DX_TAG#"${DX}"}' (${with} of ${fwd})"
    fi
    return 0
}

# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE HAND-OFF
# ══════════════════════════════════════════════════════════════════════════════════════════════
# The recovery line the main tools print when shards are missing is the main run's. Printed with
# every summary of the suite, because for the variant it must not be run by hand (see the header).
repair_warning () {
    echo "NEVER, FOR THE VARIANT: the line  bash bbl_run.sh ... --psi-tag '${DX_TAG}' --repair  that bbl_status.sh, the sweep job and"
    echo "  bbl_solve.py print when shards are missing. Run by hand it skips the check that the shards belong to the demand fit on disk,"
    echo "  submits the tables job (which overwrites the step folder's untagged tab_bbl_* with the variant's tables) and an archive"
    echo "  tagged with a bare job id. To repair the variant, run the launch line again: the suite checks the binding, repairs with"
    echo "  --no-tables --no-zip, and archives as bbl_outputs_<jobid>${DX}.zip."
}

bbl_submit () {   # bbl_submit <label> <log> <script> <args...>: one live (or dry) bbl launch; adds to SOLVE_IDS; 1 when it failed
    local label="$1" out="$2" script="$3"; shift 3
    say "   submitting the ${label} ..."
    if ! env DEMAND_SUFFIX="${DX}" POLICY_CSV="${RUN_POLICY}" CL_DRYRUN="${DRY}" CL_VERBOSE=1 bash "${HERE}/${script}" "$@" > "${out}" 2>&1; then
        tail -40 "${out}"
        say "   [!] the ${label} failed (${out}); bbl_run.sh withdrew whatever that call had submitted."
        return 1
    fi
    local s; s="$(sed -n 's/^BBL_SOLVE_JOBIDS=//p' "${out}" | tail -1)"
    SOLVE_IDS="${SOLVE_IDS:+${SOLVE_IDS}:}${s}"
    say "   ${label}: $(sed -n 's/^BBL_JOB_NAMES=//p' "${out}" | tail -1)"
    say "   ${label} job ids: $(sed -n 's/^BBL_ALL_JOBIDS=//p' "${out}" | tail -1)   (full log: ${out})"
}

# H4: the archive of the variant's BBL, afterok the solve ids in SOLVE_IDS. It first verifies the
# binding again (--check-fit): a fit replaced while the BBL ran must not be archived as its fit.
ZIP_BBL_JID=""
bbl_archive () {
    say "H4. the archive of the variant's BBL"
    local byhand="cd ${HERE} && bash ${SELF} --check-fit && bash cluster_archive.sh --set bbl --copy --slim --psi-tag '${DX_TAG}' --tag <jobid>${DX}"
    ZIP_BBL_JID=""
    SOLVE_IDS="$(sed 's/::*/:/g; s/^://; s/:$//' <<< "${SOLVE_IDS}")"
    if [[ -z "${SOLVE_IDS}" ]]; then say "   [!] no solve job id came back, so no archive job was sequenced. By hand later:  ${byhand}"; return 0; fi
    ZIP_BBL_JID=$(cl_sbatch --dependency=afterok:${SOLVE_IDS} -J dx_bbl_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGD}/dx_bbl_zip_%j.out" -e "${LOGD}/dx_bbl_zip_%j.err" \
        --wrap "cd '${HERE}' && bash ${SELF} --check-fit && bash cluster_archive.sh --set bbl --copy --slim --psi-tag '${DX_TAG}' --tag \"\${SLURM_JOB_ID}${DX}\"") || true
    if cl_require_jid "${ZIP_BBL_JID}" dx_bbl_zip; then say "   dx_bbl_zip afterok ${SOLVE_IDS//:/, } -> ${ZIP_BBL_JID}  (--check-fit, then slim on ${DX_TAG}, --copy -> ${CL_STEP_DOWNLOAD}/bbl_outputs_<jobid>${DX}.zip)"
    else ZIP_BBL_JID=""; say "   [!] the archive job was refused; the BBL jobs stay submitted. By hand later:  ${byhand}"; fi
}

# --check-fit: every BBL run of the variant on disk is the run of the fit on disk. Submits nothing.
check_fit () {
    local k m
    say "dx check-fit: data root ${CL_DATA_ROOT}"
    bbl_classify
    if (( ${#BBL_BAD[@]} > 0 )); then
        for m in "${BBL_BAD[@]}"; do say "   [!] ${m}"; done
        printf '%s  DX CHECK-FIT FAILED: files of the variant'"'"'s BBL on disk are not those of the demand fit on disk (above). They are not archived as its results. Nothing was deleted.\n' "$(date '+%F %T')"
        exit 1
    fi
    for k in ${BBL_DONE} ${BBL_REPAIR}; do
        say "   E${k}: $(basename "$(fit_path "${k}")") sha256 $(json_str "$(fit_rec "${k}")" fit_sha256 | cut -c1-12)... is the fit the BBL run $(bbl_key "${k}") was simulated on (recorded $(json_str "$(fit_rec "${k}")" recorded); archive job(s) on record: $(json_str "$(fit_rec "${k}")" bbl_archive_jobs))"
    done
    [[ -z "${BBL_FRESH}" ]] || say "   no BBL file on disk for E${BBL_FRESH// /, E}"
    say "DX CHECK-FIT OK"
}

handoff () {
    say "dx hand-off: data root ${CL_DATA_ROOT}$([[ "${DRY}" == "1" ]] && echo '  [DRY RUN: nothing is submitted or written]')"

    say "H0. the pinned files, and the main results recorded at launch"
    check_pins
    if [[ -f "${MAIN_SHA_FILE}" ]]; then
        if sha256sum -c --quiet "${MAIN_SHA_FILE}"; then say "   main results unchanged since launch ($(grep -c . "${MAIN_SHA_FILE}") files, ${MAIN_SHA_FILE##*/})"
        else die "a main-specification file changed since the launch (sha256sum -c ${MAIN_SHA_FILE}, above). The variant writes none of them; find out what did before going on."; fi
    elif [[ "${DRY}" == "1" ]]; then say "   [dry-run] no launch record ${MAIN_SHA_FILE##*/}: the check is skipped"
    else die "the launch record ${MAIN_SHA_FILE} is missing: the hand-off cannot show the main results are unchanged."; fi

    say "H1. the variant's results"
    local k st miss=""
    for k in ${ROUTINES}; do for st in ${SE_STAGE} ${BBL_STAGE_NAME}; do
        [[ -f "$(dx_res "${k}" "${st}")" ]] || miss="${miss} $(basename "$(dx_res "${k}" "${st}")")"
    done; done
    if [[ -n "${miss}" ]]; then
        if [[ "${DRY}" == "1" ]]; then say "   [dry-run] not on disk yet:${miss}"
        else die "the variant's results are missing:${miss}. A ladder did not finish; read logs/rc_ift_dx_E*_*.out."; fi
    else say "   $(for k in ${ROUTINES}; do printf 'E%s %s + %s  ' "${k}" "${SE_STAGE}" "${BBL_STAGE_NAME}"; done)on disk"; fi
    # The BBL files on disk against the fit on disk, before the comparison is written.
    bbl_classify
    bbl_require_bound
    for k in ${BBL_DONE} ${BBL_REPAIR}; do
        say "   E${k}: the BBL files on disk are those of the fit on disk (sha256 $(json_str "$(fit_rec "${k}")" fit_sha256 | cut -c1-12)..., recorded $(json_str "$(fit_rec "${k}")" recorded))"
    done

    say "H2. main vs dx (alpha, theta2, delta)"
    if [[ -n "${miss}" ]]; then say "   [dry-run] skipped: no variant result to compare"
    else
        local nowrite=(); [[ "${DRY}" == "1" ]] && nowrite=(--no-write)
        command -v julia >/dev/null 2>&1 || cl_load_julia || die "julia is not available for the comparison"
        julia --startup-file=no "${HERE}/blp_dx_compare.jl" --dir "${CL_STEP_BLP}" --routines "${ROUTINES}" --stages "${SE_STAGE} ${BBL_STAGE_NAME}" ${nowrite[@]+"${nowrite[@]}"} \
            || die "the main-vs-dx comparison failed (above). No BBL job was submitted."
    fi

    say "H3. the BBL on the variant's ${BBL_STAGE_NAME} fit, tag ${DX_TAG}"
    bbl_report
    if [[ -z "${BBL_REPAIR}${BBL_FRESH}" ]]; then
        echo
        echo "DX BBL ALREADY COMPLETE: cost_params_E{k}_spec_12_${BBL_STAGE_NAME}${DX_TAG}.json exist for E${BBL_DONE// /, E}, bound to the fit on disk. Nothing was submitted."
        echo "  To re-solve a routine on the shards on disk (same fit), move its cost_params file away and run this again (it then repairs)."
        return 0
    fi
    if [[ "${BLP_ONLY}" == "1" ]]; then
        echo
        echo "DX SUITE: BLP DONE, BBL NOT ASKED FOR (DX_BLP_ONLY=1). No BBL job was submitted."
        return 0
    fi
    if [[ -n "${miss}" ]]; then say "   [dry-run] the standard errors of ${SE_STAGE}: not checked, no variant result yet"
    elif ! se_policy "${BBL_REPAIR} ${BBL_FRESH}"; then
        die "no BBL job was submitted: ${SE_WHY}. The variant reports the standard errors of ${SE_STAGE}, and by default a routine without them is not taken to the BBL (the rung itself ends nonzero for the same reason). DX_ALLOW_NO_SE=1 accepts it: the routine is flagged and the tables print no standard error for its RC + D column."
    fi
    if ! bbl_prechecks; then
        echo
        echo "DX SUITE: BLP DONE, BBL NOT SUBMITTED."
        echo "  ${WHY_NOT}."
        echo "  A fresh launch needs dx_bbl_run.sh in this folder with bbl_run.sh as shipped (md5 ${BBL_RUN_MD5}), or a bbl_run.sh"
        echo "  that honours DEMAND_SUFFIX; see blocker B1 in dtype_variant_impl_20261001.md. Then submit the hand-off alone:"
        printf '  cd %s && sbatch -p day -t 01:00:00 -c 1 --mem=4G -J dx_next_bbl -o logs/dx_next_bbl_%%j.out --export=ALL,DX_MAIN_SHA=%s%s --wrap "bash %s --handoff"\n' "${HERE}" "${MAIN_SHA_FILE}" "$([[ "${ALLOW_NO_SE}" == "1" ]] && echo ',DX_ALLOW_NO_SE=1')" "${SELF}"
        echo "  That hand-off writes the fit records blp_bblfit_*.json after this launch's blp archive was built. For the tables' check, archive the blp set once more when it has run:"
        printf '  cd %s && sbatch -p day -t 00:20:00 -c 2 --mem=8G -J dx_blp_zip -o logs/dx_blp_zip_%%j.out --wrap '"'"'bash cluster_archive.sh --set blp --copy --tag "${SLURM_JOB_ID}%s"'"'"'\n' "${HERE}" "${DX}"
        [[ "${DRY}" == "1" ]] && return 0
        exit 4
    fi

    SOLVE_IDS=""
    local fresh_failed=0 in_archive=""
    # The fit each fresh routine is about to be simulated on, recorded BEFORE anything of the BBL
    # is submitted (the repair included): a record that cannot be written whole stops here with
    # nothing in the queue.
    for k in ${BBL_FRESH}; do
        if [[ "${DRY}" == "1" ]]; then say "   [dry-run] would record the sha256 of $(basename "$(fit_path "${k}")") in $(basename "$(fit_rec "${k}")")"
        else
            fit_record_write "${k}" || die "the fit record $(fit_rec "${k}") could not be written whole (each sha256 must be 64 hex characters, in the file as computed). No BBL job was submitted."
            say "   E${k}: fit $(basename "$(fit_path "${k}")") sha256 $(json_str "$(fit_rec "${k}")" fit_sha256 | cut -c1-12)... recorded in $(basename "$(fit_rec "${k}")")"
        fi
    done
    if [[ -n "${BBL_REPAIR}" ]]; then
        bbl_submit "repair of ${DX_TAG} (E${BBL_REPAIR// /, E})" "${LOGD}/dx_suite_${STAMP}_bbl_repair.log" bbl_run.sh \
            --routines "${BBL_REPAIR}" "${BBL_REPAIR_FLAGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
            || die "the repair of ${DX_TAG} failed (above). No BBL job of the variant is in the queue."
    fi
    if [[ -n "${BBL_FRESH}" ]]; then
        bbl_submit "launch of ${DX_TAG} (E${BBL_FRESH// /, E}, multi-start)" "${LOGD}/dx_suite_${STAMP}_bbl.log" "${LAUNCHER}" \
            --routines "${BBL_FRESH}" "${BBL_DESIGN[@]}" ${EXTRA[@]+"${EXTRA[@]}"} || fresh_failed=1
    fi
    if [[ "${fresh_failed}" == "1" && -z "${BBL_REPAIR}" ]]; then
        die "the launch of ${DX_TAG} for E${BBL_FRESH// /, E} failed (above). No BBL job of the variant is in the queue."
    fi

    bbl_archive
    # The archive job's id goes into the record of every routine whose cost parameters that
    # archive will hold: the routines already done, the repaired ones, and the launched ones.
    in_archive="${BBL_DONE} ${BBL_REPAIR}"; [[ "${fresh_failed}" == "1" ]] || in_archive="${in_archive} ${BBL_FRESH}"
    if [[ "${DRY}" == "1" ]]; then say "   [dry-run] would add the archive job's id to the fit records of E$(echo ${in_archive} | sed 's/ /, E/g')"
    elif [[ -n "${ZIP_BBL_JID}" ]]; then
        for k in ${in_archive}; do
            if fit_record_add_archive "${k}" "${ZIP_BBL_JID%%;*}"; then say "   E${k}: archive job ${ZIP_BBL_JID%%;*} (bbl_outputs_${ZIP_BBL_JID%%;*}${DX}.zip) added to $(basename "$(fit_rec "${k}")")"
            else say "   [!] E${k}: the archive job ${ZIP_BBL_JID} could not be added to $(basename "$(fit_rec "${k}")"); the tables will say the cost parameters are not verifiable against the record"; fi
        done
    else say "   [!] no archive job was sequenced, so no record names one: the tables will say the cost parameters are not verifiable against the record"; fi
    echo
    if [[ "${fresh_failed}" == "1" ]]; then
        # The repair's jobs are in the queue and their archive is sequenced; only the launch is missing.
        echo "DX BBL PARTLY SUBMITTED: repaired E${BBL_REPAIR// /, E}; the launch for E${BBL_FRESH// /, E} FAILED (${LOGD}/dx_suite_${STAMP}_bbl.log) and submitted nothing."
        echo "  solve job ids: ${SOLVE_IDS:-none}    archive: ${ZIP_BBL_JID:-none} (dx_bbl_zip, on the repair's solves)"
        echo "  When the repair's jobs have left the queue and the cause is fixed, run the launch line again for E${BBL_FRESH// /, E}."
        repair_warning
        [[ "${DRY}" == "1" ]] && return 0
        exit 5
    fi
    echo "DX BBL SUBMITTED$([[ "${DRY}" == "1" ]] && echo ' [DRY RUN]'): ${DX_TAG} on blp_results_E{k}_spec_12_${BBL_STAGE_NAME}${DX}.jls$([[ -n "${BBL_FRESH}" ]] && echo "; launched E${BBL_FRESH// /, E}")$([[ -n "${BBL_REPAIR}" ]] && echo "; repaired E${BBL_REPAIR// /, E}")$([[ -n "${BBL_DONE}" ]] && echo "; already done E${BBL_DONE// /, E}")"
    echo "  solve job ids: ${SOLVE_IDS:-none}    archive: ${ZIP_BBL_JID:-none} (dx_bbl_zip)"
    echo "STATUS (one line):"
    echo "  cd ${HERE} && bash bbl_status.sh --routines \"${ROUTINES}\" --psi-tag ${DX_TAG}"
    repair_warning
}

if [[ "${MODE}" == "handoff" ]]; then handoff; exit 0; fi
if [[ "${MODE}" == "checkfit" ]]; then check_fit; exit 0; fi

# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE LAUNCH
# ══════════════════════════════════════════════════════════════════════════════════════════════
say "dx_suite_20261001: data root ${CL_DATA_ROOT}$([[ "${DRY}" == "1" ]] && echo '  [DRY RUN: nothing is submitted or written]')"

# DX_SMOKE turns blp_dx_stage_job.sh into a dry run that ends with status 0 and writes no result.
# It is the smoke job's variable; inherited by a rung it would make the whole ladder a no-op.
if [[ -n "${DX_SMOKE:-}" && "${DX_SMOKE}" != "0" ]]; then
    die "DX_SMOKE='${DX_SMOKE}' is set in this environment. It belongs to the smoke job alone (blp_dx_stage_job.sh as a dry run). Run 'unset DX_SMOKE' in the shell this was launched from, then launch again. Nothing was submitted."
fi
case "${ALLOW_NO_SE}" in
    0) ;;
    1) say "   DX_ALLOW_NO_SE=1: ${SE_STAGE} without joint standard errors is ACCEPTED in the rungs and at the hand-off (flagged; the tables print no standard error for that column)" ;;
    *) die "DX_ALLOW_NO_SE='${ALLOW_NO_SE}': use 1 to accept ${SE_STAGE} without joint standard errors, or leave it unset." ;;
esac

# ── 0. the tree the suite executes ───────────────────────────────────────────────────────────
say "0. the files this upload does not carry (${#PINNED[@]} pinned)"
check_pins

# ── 1. the queue ─────────────────────────────────────────────────────────────────────────────
say "1. the queue"
if [[ "${DRY}" != "1" ]]; then
    read_queue
    own="$(live_jobs '$2 ~ /^rc_ift_dx_/ || $2 == "dx_next_bbl" || $2 == "dx_blp_zip" || $2 == "dx_bbl_zip" || $2 ~ /_dx_ms982$/')"
    [[ -z "${own}" ]] || die "jobs of the variant are live: $(tr '\n' ';' <<< "${own}"). Wait for them, or scancel them, and run this again."
    main="$(live_jobs '$2 ~ /(_ms982|_sc982)$/ && $2 !~ /_dx_/ || $2 == "bbl_relaunch" || $2 == "bbl_polfunc"')"
    if [[ -n "${main}" ]]; then
        if [[ "${DX_WITH_LIVE_MAIN:-0}" == "1" ]]; then say "   [note] main BBL jobs are live (DX_WITH_LIVE_MAIN=1): $(tr '\n' ';' <<< "${main}")"
        else die "the main BBL runs are still live: $(tr '\n' ';' <<< "${main}"). The variant waits for them (its BBL uses the same gpu_h100 QOS). Run this again when they are done, or set DX_WITH_LIVE_MAIN=1 to queue behind them."; fi
    else say "   no job of the variant and none of ${MAIN_TAGS// /, } is live"; fi
else say "   [dry-run] not checked"; fi

# ── 2. the inputs ────────────────────────────────────────────────────────────────────────────
say "2. the inputs of the ladders"
bad=""
for k in ${ROUTINES}; do
    pq="$(ls "${CL_STEP_DEMAND}"/demand_${k}_*spec_12.parquet "${CL_STEP_DEMAND}"/demand_${k}_spec_12.parquet 2>/dev/null | grep -v _final_ | head -1)"
    [[ -n "${pq}" ]] || bad="${bad} demand_${k}_*_spec_12.parquet"
    if [[ ! -f "${CL_DATA_IN}/logit_delta_E${k}_spec_12.bin" && ! -f "${CL_STEP_LOGIT}/logit_delta_E${k}_spec_12.bin" ]]; then bad="${bad} logit_delta_E${k}_spec_12.bin"; fi
    if grep -q "\"E${k}_full_dtype\"" "${CL_STEP_LOGIT}/logit_summary_spec_12.json" 2>/dev/null; then
        say "   E${k}: ${pq##*/}; logit delta on disk; logit E${k}_full_dtype in the summary (the nesting check compares with it)"
    else
        say "   E${k}: ${pq##*/}; [note] no E${k}_full_dtype in ${CL_STEP_LOGIT}/logit_summary_spec_12.json: the nesting check reports the linear step without a logit to compare with"
    fi
done
cl_draws_missing "${R:-2000}" "${SEED:-42}" && bad="${bad} draws(R=${R:-2000},seed=${SEED:-42})"
# DX_ASSUME_SYSIMAGE=1 skips the image test (the local test has no image; every stage job still
# refuses without a valid one, cl_require_sysimage).
if [[ "${DRY}" != "1" && "${DX_ASSUME_SYSIMAGE:-0}" != "1" ]]; then cl_sysimage_missing gpu && bad="${bad} blp_sysimage.so(gpu)"; fi
if [[ -n "${bad}" ]]; then
    if [[ "${DRY}" == "1" ]]; then say "   [dry-run] missing:${bad}"; else die "inputs missing:${bad}. The variant builds none of them; they are the main specification's."; fi
fi

# ── 3. the main results: required, and recorded ──────────────────────────────────────────────
say "3. the main specification's results (read-only for the variant)"
rec=(); miss=""
for k in ${ROUTINES}; do
    for st in ${SE_STAGE} ${BBL_STAGE_NAME}; do f="$(main_res "${k}" "${st}")"; if [[ -f "${f}" ]]; then rec+=("${f}"); else miss="${miss} ${f##*/}"; fi; done
    for t in ${MAIN_TAGS}; do f="${CL_STEP_BBL}/cost_params_E${k}_spec_12_${BBL_STAGE_NAME}${t}.json"; [[ -f "${f}" ]] && rec+=("${f}"); done
done
[[ -z "${miss}" ]] || die "main results missing:${miss}. The hand-off compares the variant with them (and the tables need them); nothing was submitted."
if [[ "${DRY}" == "1" ]]; then say "   [dry-run] would record the sha256 of ${#rec[@]} file(s) in ${MAIN_SHA_FILE##*/}"
else
    sha256sum "${rec[@]}" > "${MAIN_SHA_FILE}" || die "cannot write ${MAIN_SHA_FILE}"
    say "   sha256 of ${#rec[@]} file(s) -> ${MAIN_SHA_FILE}"
fi

# ── 4. the state of the variant ──────────────────────────────────────────────────────────────
say "4. the variant on disk: the ladders and the BBL, per routine"
LADDERS=""; LADDER_DONE=""
for k in ${ROUTINES}; do
    done_st=""
    for st in sigma rc2 rc3 rc4 ext1 ext2 extended; do [[ -f "$(dx_res "${k}" "${st}")" ]] && done_st="${done_st} ${st}"; done
    if [[ -f "$(dx_res "${k}" "${BBL_STAGE_NAME}")" && -f "$(dx_res "${k}" "${SE_STAGE}")" ]]; then
        LADDER_DONE="${LADDER_DONE:+${LADDER_DONE} }${k}"
        say "   E${k}: the ladder is done (results on disk:${done_st}); no ladder job"
    else
        LADDERS="${LADDERS:+${LADDERS} }${k}"
        say "   E${k}: ladder to run$([[ -n "${done_st}" ]] && echo " (results already on disk:${done_st}; the engine skips them)")"
    fi
done
# The BBL files of each routine against its fit. A routine with BBL files and no ${BBL_STAGE_NAME}
# fit, or with a fit that is not the recorded one, stops the launch here: a ladder would give
# it a fit its shards and cost parameters do not come from.
bbl_classify
bbl_require_bound
bbl_report
if [[ -z "${LADDERS}" && -z "${BBL_REPAIR}${BBL_FRESH}" ]]; then
    echo
    echo "DX SUITE: NOTHING TO DO. No ladder is left, and the variant's cost parameters exist for E${BBL_DONE// /, E} (cost_params_E{k}_spec_12_${BBL_STAGE_NAME}${DX_TAG}.json), bound to the fits on disk."
    echo "  Nothing was submitted. To re-solve a routine on the shards on disk (same fit), move its cost_params file away and run this again."
    exit 0
fi

# ── 5. the BBL the hand-off will submit, checked now ─────────────────────────────────────────
say "5. the BBL of the hand-off, checked before any GPU job"
if [[ "${BLP_ONLY}" == "1" ]]; then
    say "   DX_BLP_ONLY=1: the BLP ladders and the comparison only; the hand-off submits no BBL job"
elif [[ -z "${BBL_REPAIR}${BBL_FRESH}" ]]; then
    say "   the BBL is done for every routine: the hand-off compares main and dx and submits no BBL job"
else
    # The rule of H3 on the results already on disk: the routines whose ladder is done and that
    # would get BBL jobs. A routine whose ladder is still to run is held to it by its own rung.
    need_se=""
    for k in ${LADDER_DONE}; do [[ " ${BBL_REPAIR} ${BBL_FRESH} " == *" ${k} "* ]] && need_se="${need_se:+${need_se} }${k}"; done
    if [[ -n "${need_se}" ]]; then
        se_policy "${need_se}" || die "the hand-off would submit no BBL job: ${SE_WHY}. The variant reports the standard errors of ${SE_STAGE}, and by default a routine without them is not taken to the BBL. DX_ALLOW_NO_SE=1 in front of the launch accepts it (flagged; the tables print no standard error for its RC + D column). Nothing was submitted."
    fi
    if [[ "${DRY}" == "1" || " ${EXTRA[*]-} " == *" --skip-preflight "* ]]; then
        say "   cluster_preflight.sh: not run ($([[ "${DRY}" == "1" ]] && echo 'dry run' || echo '--skip-preflight'))"
    else
        lg="${LOGD}/dx_suite_${STAMP}_preflight.log"
        bash "${HERE}/cluster_preflight.sh" --routines "${ROUTINES}" > "${lg}" 2>&1 \
            || { tail -30 "${lg}"; die "cluster_preflight.sh found blockers (${lg}). bbl_run.sh runs it before a live BBL launch, so the hand-off would stop there. Nothing was submitted."; }
        say "   cluster_preflight.sh: no blocker (${lg##*/})"
    fi
    bbl_prechecks || die "the hand-off would end in 'BBL NOT SUBMITTED': ${WHY_NOT}. Nothing was submitted. A fresh launch needs dx_bbl_run.sh in this folder with bbl_run.sh as shipped (md5 ${BBL_RUN_MD5}); DX_BLP_ONLY=1 runs the BLP ladders and the comparison without a BBL."
fi

# ── 6. the BLP ladders ───────────────────────────────────────────────────────────────────────
# The SE knobs and the draw the engine reads, exported as blp_run.sh exports them.
export BLP_SE_METHOD="${SE_METHOD:-wcb}" BLP_WCB_REPS="${WCB_REPS:-999}" BLP_WCB_SCHEME="${WCB_SCHEME:-webb}"
export BLP_SE_ONLY=0
R="${R:-2000}"; SEED="${SEED:-42}"; export R SEED
PART_ARGS=()
if [[ "${BLP_PARTITION}" != "gpu_h200" ]]; then PART_ARGS=(--partition="${BLP_PARTITION}" --gpus="$(cl_bbl_part "${BLP_PARTITION}" gpu_type):1"); fi

# Every rung is told DX_SMOKE=0 (a real stage, whatever the environment holds) and this launch's
# DX_ALLOW_NO_SE, so the rule on ext1's standard errors is the same in the rungs and at the hand-off.
rung () {   # rung <routine> <stage-set> <wall> <jobtag> [dep] -> job id
    local k="$1" st="$2" wall="$3" jt="$4" dep="${5:-}" jid name
    name="rc_ift_dx_E${k}_${jt}"
    jid=$(cl_sbatch --time="${wall}" --mem="$(cl_mem_for "${k}")" ${PART_ARGS[@]+"${PART_ARGS[@]}"} \
        ${dep:+--dependency=afterok:${dep}} \
        --export=ALL,RC_ROUTINE=${k},RC_STAGE=${st},DX_SMOKE=0,DX_ALLOW_NO_SE=${ALLOW_NO_SE} \
        -J "${name}" -o "${LOGD}/${name}_%j.out" -e "${LOGD}/${name}_%j.err" \
        "${HERE}/blp_dx_stage_job.sh") || true
    cl_require_jid "${jid}" "${name}" || return 1
    printf '%s\n' "${jid}"
}
submit_ladders () {   # sets TERM_JIDS (':'-joined terminal ids) and ALL_BLP (every id)
    local k h e2 ex
    TERM_JIDS=""; ALL_BLP=""
    for k in ${LADDERS}; do
        h=$(rung "${k}" "${HEAD_STAGES}" "2-00:00:00" head) || return 1
        ALL_BLP="${ALL_BLP:+${ALL_BLP} }${h}"
        say "   rc_ift_dx_E${k}_head (${HEAD_STAGES}) -> ${h}  [${BLP_PARTITION}, mem $(cl_mem_for "${k}")]"
        e2=$(rung "${k}" ext2 "$(cl_stage_wall ext2)" ext2 "${h}") || return 1
        ALL_BLP="${ALL_BLP} ${e2}"
        say "   rc_ift_dx_E${k}_ext2 afterok ${h} -> ${e2}"
        ex=$(rung "${k}" extended "$(cl_stage_wall extended)" ext "${e2}") || return 1
        ALL_BLP="${ALL_BLP} ${ex}"
        say "   rc_ift_dx_E${k}_ext (extended) afterok ${e2} -> ${ex}"
        TERM_JIDS="${TERM_JIDS:+${TERM_JIDS}:}${ex}"
    done
    [[ -n "${LADDERS}" ]] || say "   no ladder to submit: every routine's results are on disk"
}
submit_tail () {   # the hand-off and the archive; sets HAND_JID, ZIP_JID
    HAND_JID=$(cl_sbatch ${TERM_JIDS:+--dependency=afterok:${TERM_JIDS}} -J dx_next_bbl --partition="${ZIP_PARTITION:-day}" --time=01:00:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGD}/dx_next_bbl_%j.out" -e "${LOGD}/dx_next_bbl_%j.err" \
        --export=ALL,DX_MAIN_SHA="${MAIN_SHA_FILE}",DX_STAMP="${STAMP}",DX_BLP_ONLY="${BLP_ONLY}",DX_ALLOW_NO_SE="${ALLOW_NO_SE}" \
        --wrap "cd '${HERE}' && bash ${SELF} --handoff") || true
    cl_require_jid "${HAND_JID}" dx_next_bbl || return 1
    say "   dx_next_bbl ${TERM_JIDS:+afterok ${TERM_JIDS//:/, } }-> ${HAND_JID}  (main vs dx, then the BBL ${DX_TAG}$([[ "${BLP_ONLY}" == "1" ]] && echo ': NOT submitted, DX_BLP_ONLY=1'))"
    # afterany EVERY terminal id, not the hand-off alone: when one ladder fails, the hand-off is
    # cancelled at once (kill-on-invalid-dep) while the other ladder still runs, and an archive
    # waiting on the hand-off only would fire before that ladder's results exist.
    ZIP_JID=$(cl_sbatch --dependency=afterany:${TERM_JIDS:+${TERM_JIDS}:}${HAND_JID} -J dx_blp_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGD}/dx_blp_zip_%j.out" -e "${LOGD}/dx_blp_zip_%j.err" \
        --wrap "cd '${HERE}' && bash cluster_archive.sh --set blp --copy --tag \"\${SLURM_JOB_ID}${DX}\" && bash cluster_archive.sh --set logs --copy --tag \"\${SLURM_JOB_ID}${DX}\"") || true
    cl_require_jid "${ZIP_JID}" dx_blp_zip || return 1
    say "   dx_blp_zip afterany ${TERM_JIDS:+${TERM_JIDS//:/, }, }${HAND_JID} -> ${ZIP_JID}  (sets blp + logs, --copy -> ${CL_STEP_DOWNLOAD}/blp_outputs_<jobid>${DX}.zip)"
}

say "6. BLP ladders, dry run (CL_DRYRUN=1)"
( CL_DRYRUN=1; submit_ladders && submit_tail ) || die "the dry run of the BLP ladders failed (above). Nothing was submitted."
if [[ "${DRY}" == "1" ]]; then
    say "7. the hand-off, dry run"
    ( MODE=handoff; handoff ) || die "the dry run of the hand-off failed (above)."
    echo
    echo "DX SUITE DRY RUN COMPLETE: nothing was submitted."
    exit 0
fi

say "   submitting ..."
submit_ladders || { [[ -z "${ALL_BLP:-}" ]] || scancel ${ALL_BLP} 2>/dev/null; die "a ladder submission was refused; the jobs already submitted were cancelled (${ALL_BLP:-none})."; }
say "7. the hand-off and the archive"
submit_tail || { scancel ${ALL_BLP:-} ${HAND_JID:-} 2>/dev/null; die "the hand-off or the archive was refused; the jobs already submitted were cancelled (${ALL_BLP:-none} ${HAND_JID:-})."; }

# ── 8. summary ───────────────────────────────────────────────────────────────────────────────
echo
echo "DX SUITE SUBMITTED: variant dx (D-type dummy in X1), routines ${ROUTINES}"
echo "  BLP ladders : ${ALL_BLP:-none (results on disk)}"
echo "  hand-off    : ${HAND_JID}  (dx_next_bbl: main vs dx, then the BBL ${DX_TAG}$([[ "${BLP_ONLY}" == "1" ]] && echo ' -- NOT submitted, DX_BLP_ONLY=1'))"
echo "  archive     : ${ZIP_JID}  (dx_blp_zip)"
echo "  main results recorded in: ${MAIN_SHA_FILE}"
echo "STATUS (one line):"
echo "  squeue -u \$USER -h -o \"%i %j %T %r\" | grep -E 'rc_ift_dx_|dx_next_bbl|dx_blp_zip|dx_bbl_zip|${DX_TAG}'"
echo "THE COMPARISON AND THE BBL JOB IDS (one line, when dx_next_bbl has run):"
echo "  cd ${HERE} && cat logs/dx_next_bbl_${HAND_JID}.out"
repair_warning
