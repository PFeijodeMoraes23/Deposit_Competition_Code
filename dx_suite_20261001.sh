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
# engine and wraps it from outside, its worker is blp_dx_stage_job.sh, and every file it writes
# carries `_dx` (data/output/blp/*_dx.*; data/output/bbl/*_extended_dx_ms982*). The main
# specification's results, checkpoints, psi and cost parameters are read, never written.
#
#   0. CHECK THE TREE. The run executes files this upload does not carry: each must have the md5
#      pinned below, the one the variant was tested with. A mismatch refuses before anything is
#      submitted. bbl_run.sh is reported, not pinned (see step H3).
#   1. CHECK THE QUEUE. Refuses if a job of the variant is live (a second launch), and if a job
#      of the main BBL runs _ms982 / _sc982 is live: the variant's BBL shares their GPU partition
#      and QOS, so it waits for them unless DX_WITH_LIVE_MAIN=1 says otherwise.
#   2. CHECK THE INPUTS the ladder reads and no job of this graph produces: the demand parquets,
#      the logit delta, the draws, the GPU sysimage. The logit sub-model the variant nests
#      (E{k}_full_dtype in data/output/logit/logit_summary_spec_12.json) is reported.
#   3. RECORD THE MAIN RESULTS: sha256 of the main specification's ext1 / extended results and
#      of its _ms982 cost parameters, into logs/dx_suite_<stamp>_main.sha256. The hand-off
#      verifies them again before it submits anything.
#   4. SUBMIT THE BLP LADDERS, dry run first: per routine
#          rc_ift_dx_E{k}_head (sigma+rc2+rc3+rc4+ext1) -> rc_ift_dx_E{k}_ext2 -> rc_ift_dx_E{k}_ext
#      afterok chains on the GPU IFT engine, a cold ladder: theta2 starts from the variant's own
#      checkpoints only. Each job first runs the theta2 = 0 nesting check against the logit.
#   5. SUBMIT THE HAND-OFF dx_next_bbl (afterok both ladders; `bash dx_suite_20261001.sh
#      --handoff`) and the archive dx_blp_zip (afterany the hand-off; sets blp + logs, --copy).
#   6. PRINT every job id and the status commands.
#
# The hand-off (H):
#   H0. the pinned files again, and the recorded sha256 of the main results (unchanged).
#   H1. the variant's ext1 and extended results exist for both routines.
#   H2. MAIN vs DX, printed to the log (blp_dx_compare.jl): alpha, the D-type coefficient, Q,
#       every theta2, and the distribution of delta_dx - delta_main, at ext1 and extended.
#   H3. THE BBL, multi-start only, on the variant's `extended` fit, tag _dx_ms982: the _ms982
#       design (beta 0.979, T 250, evolving phi, mean-reverting Z, lagged carry; gpu_h100,
#       PACK 2 x 230G, 300 shards, --shard-time 01:45:00), --no-polfunc on the installed policy
#       (sha256 pinned), --no-tables. bbl_run.sh is dry-run first with DEMAND_SUFFIX=_dx, and
#       the forward-simulation command it builds must carry `--suffix _dx`: that is what makes
#       the simulation read blp_results_E{k}_spec_12_extended_dx.jls. A bbl_run.sh that does not
#       pass it would simulate on the MAIN fit under the variant's tag, so the hand-off then
#       submits no BBL job, says so, and prints the command to run once bbl_run.sh passes it.
#
# Re-running it is safe: step 1 refuses while the variant's jobs are live, the engine skips the
# stages whose results are on disk, and bbl_run.sh refuses a second launch of a live tag.
# CL_DRYRUN=1 in the environment makes the whole script a dry run, hand-off included.
# ==============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}" || exit 2
CL_VERBOSE=1
. ./cluster_lib.sh

MODE=launch
case "${1:-}" in
    "")        ;;
    --handoff) MODE=handoff ;;
    *) echo "unknown option: $1 (the script takes no argument; --handoff is its own continuation)" >&2; exit 2 ;;
esac

SELF="dx_suite_20261001.sh"
ROUTINES="3 4"
DX="_dx"
MAIN_TAGS="_ms982 _sc982"
DX_TAG="${DX}_ms982"
SE_STAGE="ext1"; BBL_STAGE_NAME="extended"
HEAD_STAGES="sigma+rc2+rc3+rc4+ext1"
BLP_PARTITION="${DX_BLP_PARTITION:-gpu_h200}"
POLICY_SHA256="fd208b5fc57b9697eefc010ed4a1076af776d9510b89f3bbc1edca009dc29728"
POLICY_CSV="${CL_STEP_BBL}/polfunc_fitted.csv"
BBL_FLAGS=(--routines "${ROUTINES}" --fwd-gpu --partitions gpu_h100 --shards 300 --beta 0.979 --horizon 250
           --phi-path evolving --z-path mean_reverting --rdep-timing lagged --no-warmup --no-polfunc
           --pack-h100 2 --mem-h100 230G --shard-time 01:45:00
           --multi-start --n-paths 1 --psi-tag "${DX_TAG}" --no-tables)
# DX_BBL_EXTRA: extra bbl_run.sh flags (the local test passes --skip-preflight).
read -r -a EXTRA <<< "${DX_BBL_EXTRA:-}"
BBL_FLAGS+=(${EXTRA[@]+"${EXTRA[@]}"})
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

live_jobs () {   # live_jobs <awk condition on the job name $2>
    squeue -h -u "${USER:-$(id -un)}" -t PENDING,RUNNING,CONFIGURING,SUSPENDED,REQUEUED -o '%i %j' 2>/dev/null | awk "$1"
}

# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE HAND-OFF
# ══════════════════════════════════════════════════════════════════════════════════════════════
handoff () {
    say "dx hand-off: data root ${CL_DATA_ROOT}$([[ "${DRY}" == "1" ]] && echo '  [DRY RUN: nothing is submitted]')"

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

    say "H2. main vs dx (alpha, theta2, delta)"
    if [[ -n "${miss}" ]]; then say "   [dry-run] skipped: no variant result to compare"
    else
        command -v julia >/dev/null 2>&1 || cl_load_julia || die "julia is not available for the comparison"
        julia --startup-file=no "${HERE}/blp_dx_compare.jl" --dir "${CL_STEP_BLP}" --routines "${ROUTINES}" --stages "${SE_STAGE} ${BBL_STAGE_NAME}" \
            || die "the main-vs-dx comparison failed (above). No BBL job was submitted."
    fi

    say "H3. the BBL on the variant's ${BBL_STAGE_NAME} fit, tag ${DX_TAG}"
    local run_policy="${POLICY_CSV}"
    if [[ -f "${POLICY_CSV}" ]]; then
        [[ "$(sha "${POLICY_CSV}")" == "${POLICY_SHA256}" ]] || die "${POLICY_CSV} has sha256 $(sha "${POLICY_CSV}"), not the pinned ${POLICY_SHA256:0:12}... the main runs used. No BBL job was submitted."
        say "   policy ${POLICY_CSV} (sha256 ${POLICY_SHA256:0:12}), the one _ms982 was simulated around"
    elif [[ "${DRY}" == "1" ]]; then say "   [dry-run] ${POLICY_CSV} is not on disk"
    else die "${POLICY_CSV} is missing. No BBL job was submitted."; fi
    local m; m="$(md5 "${HERE}/bbl_run.sh")"
    say "   bbl_run.sh md5 ${m}$([[ "${m}" == "${BBL_RUN_MD5}" ]] && echo ' (as shipped 2026-09-30)')"

    local lg="${LOGD}/dx_suite_${STAMP}_bbl_dry.log"
    env DEMAND_SUFFIX="${DX}" POLICY_CSV="${run_policy}" CL_DRYRUN=1 CL_VERBOSE=1 bash "${HERE}/bbl_run.sh" "${BBL_FLAGS[@]}" > "${lg}" 2>&1 \
        || { tail -25 "${lg}"; die "the dry run of the BBL launch was refused (${lg}). No BBL job was submitted."; }
    local fwd; fwd="$(grep -c -- '-J bbl_fwd_E[0-9]*'"${DX_TAG}"' ' "${lg}")"
    say "   dry run OK: $(grep -c '\[dry-run\] sbatch' "${lg}") jobs would be submitted ($(sed -n 's/^BBL_JOB_NAMES=//p' "${lg}" | tail -1))"
    # The forward simulation must be told the demand suffix. Each fwd submission carries the sim's
    # command line in BBL_EXTRA; the suffix has to be on every one of them, with the psi tag
    # reduced to the part after it, so the files the sim writes are the ones the sweep and the
    # solve look for under --psi-tag ${DX_TAG}.
    local with; with="$(grep -- '-J bbl_fwd_E[0-9]*'"${DX_TAG}"' ' "${lg}" | grep -c -- "--suffix ${DX} --psi-tag ${DX_TAG#"${DX}"}\([ ,]\|$\)")"
    if (( fwd == 0 || with != fwd )); then
        echo
        echo "DX SUITE: BLP DONE, BBL NOT SUBMITTED."
        echo "  bbl_run.sh does not pass the demand suffix to the forward simulation (${with} of ${fwd} fwd submissions carry"
        echo "  '--suffix ${DX} --psi-tag ${DX_TAG#"${DX}"}'). As it stands the simulation would read the MAIN fit"
        echo "  blp_results_E{k}_spec_12_${BBL_STAGE_NAME}.jls and write it under the variant's tag."
        echo "  bbl_run.sh needs to honour DEMAND_SUFFIX (three lines; see the checkpoint dtype_variant_impl_20261001.md,"
        echo "  blocker B1). Once it does, submit the BBL alone:"
        printf '  cd %s && sbatch -p day -t 01:00:00 -c 1 --mem=4G -J dx_next_bbl -o logs/dx_next_bbl_%%j.out --export=ALL,DX_MAIN_SHA=%s --wrap "bash %s --handoff"\n' "${HERE}" "${MAIN_SHA_FILE}" "${SELF}"
        [[ "${DRY}" == "1" ]] && return 0
        exit 4
    fi
    say "   every fwd submission carries '--suffix ${DX} --psi-tag ${DX_TAG#"${DX}"}' (${with} of ${fwd})"

    local out="${LOGD}/dx_suite_${STAMP}_bbl.log"
    say "   submitting ${DX_TAG} (multi-start) ..."
    if ! env DEMAND_SUFFIX="${DX}" POLICY_CSV="${run_policy}" CL_DRYRUN="${DRY}" CL_VERBOSE=1 bash "${HERE}/bbl_run.sh" "${BBL_FLAGS[@]}" > "${out}" 2>&1; then
        tail -40 "${out}"
        die "the ${DX_TAG} launch failed (${out}); bbl_run.sh withdrew whatever it had submitted."
    fi
    say "   ${DX_TAG}: $(sed -n 's/^BBL_JOB_NAMES=//p' "${out}" | tail -1)"
    echo
    echo "DX BBL SUBMITTED$([[ "${DRY}" == "1" ]] && echo ' [DRY RUN]'): ${DX_TAG} multi-start E3+E4 on blp_results_E{k}_spec_12_${BBL_STAGE_NAME}${DX}.jls"
    echo "  job ids: $(sed -n 's/^BBL_ALL_JOBIDS=//p' "${out}" | tail -1)"
    echo "  full submission log: ${out}"
    echo "STATUS (one line):"
    echo "  cd ${HERE} && bash bbl_status.sh --routines \"${ROUTINES}\" --psi-tag ${DX_TAG}"
}

if [[ "${MODE}" == "handoff" ]]; then handoff; exit 0; fi

# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE LAUNCH
# ══════════════════════════════════════════════════════════════════════════════════════════════
say "dx_suite_20261001: data root ${CL_DATA_ROOT}$([[ "${DRY}" == "1" ]] && echo '  [DRY RUN: nothing is submitted]')"

# ── 0. the tree the suite executes ───────────────────────────────────────────────────────────
say "0. the files this upload does not carry (${#PINNED[@]} pinned)"
check_pins
m="$(md5 "${HERE}/bbl_run.sh")"
if [[ "${m}" == "${BBL_RUN_MD5}" ]]; then
    say "   [note] bbl_run.sh is the 2026-09-30 file: it does not pass the demand suffix to the forward simulation,"
    say "          so the hand-off will stop after the BLP comparison with 'BBL NOT SUBMITTED' (step H3 of the header)."
else say "   bbl_run.sh md5 ${m} (not the 2026-09-30 file; the hand-off tests what it passes to the simulation)"; fi

# ── 1. the queue ─────────────────────────────────────────────────────────────────────────────
say "1. the queue"
if [[ "${DRY}" != "1" ]]; then
    own="$(live_jobs '$2 ~ /^rc_ift_dx_/ || $2 == "dx_next_bbl" || $2 == "dx_blp_zip" || $2 ~ /_dx_ms982$/')"
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
    done_st=""
    for st in sigma rc2 rc3 rc4 ext1 ext2 extended; do [[ -f "$(dx_res "${k}" "${st}")" ]] && done_st="${done_st} ${st}"; done
    [[ -z "${done_st}" ]] || say "   E${k}: variant results already on disk for:${done_st} (the engine skips them)"
done
cl_draws_missing "${R:-2000}" "${SEED:-42}" && bad="${bad} draws(R=${R:-2000},seed=${SEED:-42})"
# DX_ASSUME_SYSIMAGE=1 skips the image test (the local test has no image; every stage job still
# refuses without a valid one, cl_require_sysimage).
if [[ "${DRY}" != "1" && "${DX_ASSUME_SYSIMAGE:-0}" != "1" ]]; then cl_sysimage_missing gpu && bad="${bad} blp_sysimage.so(gpu)"; fi
if [[ -n "${bad}" ]]; then
    if [[ "${DRY}" == "1" ]]; then say "   [dry-run] missing:${bad}"; else die "inputs missing:${bad}. The variant builds none of them; they are the main specification's."; fi
fi

# ── 3. the main results, recorded ────────────────────────────────────────────────────────────
say "3. the main specification's results (read-only for the variant)"
rec=()
for k in ${ROUTINES}; do
    for st in ${SE_STAGE} ${BBL_STAGE_NAME}; do f="$(main_res "${k}" "${st}")"; [[ -f "${f}" ]] && rec+=("${f}"); done
    for t in ${MAIN_TAGS}; do f="${CL_STEP_BBL}/cost_params_E${k}_spec_12_${BBL_STAGE_NAME}${t}.json"; [[ -f "${f}" ]] && rec+=("${f}"); done
done
if (( ${#rec[@]} == 0 )); then say "   [note] no main result on disk to record"
elif [[ "${DRY}" == "1" ]]; then say "   [dry-run] would record the sha256 of ${#rec[@]} file(s) in ${MAIN_SHA_FILE##*/}"
else
    sha256sum "${rec[@]}" > "${MAIN_SHA_FILE}" || die "cannot write ${MAIN_SHA_FILE}"
    say "   sha256 of ${#rec[@]} file(s) -> ${MAIN_SHA_FILE}"
fi

# ── 4. the BLP ladders ───────────────────────────────────────────────────────────────────────
# The SE knobs and the draw the engine reads, exported as blp_run.sh exports them.
export BLP_SE_METHOD="${SE_METHOD:-wcb}" BLP_WCB_REPS="${WCB_REPS:-999}" BLP_WCB_SCHEME="${WCB_SCHEME:-webb}"
export BLP_SE_ONLY=0
R="${R:-2000}"; SEED="${SEED:-42}"; export R SEED
PART_ARGS=()
if [[ "${BLP_PARTITION}" != "gpu_h200" ]]; then PART_ARGS=(--partition="${BLP_PARTITION}" --gpus="$(cl_bbl_part "${BLP_PARTITION}" gpu_type):1"); fi

rung () {   # rung <routine> <stage-set> <wall> <jobtag> [dep] -> job id
    local k="$1" st="$2" wall="$3" jt="$4" dep="${5:-}" jid name
    name="rc_ift_dx_E${k}_${jt}"
    jid=$(cl_sbatch --time="${wall}" --mem="$(cl_mem_for "${k}")" ${PART_ARGS[@]+"${PART_ARGS[@]}"} \
        ${dep:+--dependency=afterok:${dep}} \
        --export=ALL,RC_ROUTINE=${k},RC_STAGE=${st} \
        -J "${name}" -o "${LOGD}/${name}_%j.out" -e "${LOGD}/${name}_%j.err" \
        "${HERE}/blp_dx_stage_job.sh") || true
    cl_require_jid "${jid}" "${name}" || return 1
    printf '%s\n' "${jid}"
}
submit_ladders () {   # sets TERM_JIDS (':'-joined terminal ids) and ALL_BLP (every id)
    local k h e2 ex
    TERM_JIDS=""; ALL_BLP=""
    for k in ${ROUTINES}; do
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
}
submit_tail () {   # the hand-off and the archive; sets HAND_JID, ZIP_JID
    HAND_JID=$(cl_sbatch --dependency=afterok:${TERM_JIDS} -J dx_next_bbl --partition="${ZIP_PARTITION:-day}" --time=01:00:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGD}/dx_next_bbl_%j.out" -e "${LOGD}/dx_next_bbl_%j.err" \
        --export=ALL,DX_MAIN_SHA="${MAIN_SHA_FILE}",DX_STAMP="${STAMP}" \
        --wrap "cd '${HERE}' && bash ${SELF} --handoff") || true
    cl_require_jid "${HAND_JID}" dx_next_bbl || return 1
    say "   dx_next_bbl afterok ${TERM_JIDS//:/, } -> ${HAND_JID}  (main vs dx, then the BBL ${DX_TAG})"
    ZIP_JID=$(cl_sbatch --dependency=afterany:${HAND_JID} -J dx_blp_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGD}/dx_blp_zip_%j.out" -e "${LOGD}/dx_blp_zip_%j.err" \
        --wrap "cd '${HERE}' && bash cluster_archive.sh --set blp --copy --tag \"dx_\${SLURM_JOB_ID}\" && bash cluster_archive.sh --set logs --copy --tag \"dx_\${SLURM_JOB_ID}\"") || true
    cl_require_jid "${ZIP_JID}" dx_blp_zip || return 1
    say "   dx_blp_zip afterany ${HAND_JID} -> ${ZIP_JID}  (sets blp + logs, --copy -> ${CL_STEP_DOWNLOAD}/blp_outputs_dx_<jobid>.zip)"
}

say "4. BLP ladders, dry run (CL_DRYRUN=1)"
( CL_DRYRUN=1; submit_ladders && submit_tail ) || die "the dry run of the BLP ladders failed (above). Nothing was submitted."
if [[ "${DRY}" == "1" ]]; then
    say "5. the hand-off, dry run"
    ( MODE=handoff; handoff ) || die "the dry run of the hand-off failed (above)."
    echo
    echo "DX SUITE DRY RUN COMPLETE: nothing was submitted."
    exit 0
fi

say "   submitting ..."
submit_ladders || { [[ -z "${ALL_BLP:-}" ]] || scancel ${ALL_BLP} 2>/dev/null; die "a ladder submission was refused; the jobs already submitted were cancelled (${ALL_BLP:-none})."; }
say "5. the hand-off and the archive"
submit_tail || { scancel ${ALL_BLP} ${HAND_JID:-} 2>/dev/null; die "the hand-off or the archive was refused; the ladders were cancelled (${ALL_BLP})."; }

# ── 6. summary ───────────────────────────────────────────────────────────────────────────────
echo
echo "DX SUITE SUBMITTED: variant dx (D-type dummy in X1), routines ${ROUTINES}"
echo "  BLP ladders : ${ALL_BLP}"
echo "  hand-off    : ${HAND_JID}  (dx_next_bbl: main vs dx, then the BBL ${DX_TAG})"
echo "  archive     : ${ZIP_JID}  (dx_blp_zip)"
echo "  main results recorded in: ${MAIN_SHA_FILE}"
echo "STATUS (one line):"
echo "  squeue -u \$USER -h -o \"%i %j %T %r\" | grep -E 'rc_ift_dx_|dx_next_bbl|dx_blp_zip|${DX_TAG}'"
echo "THE COMPARISON AND THE BBL JOB IDS (one line, when dx_next_bbl has run):"
echo "  cd ${HERE} && cat logs/dx_next_bbl_${HAND_JID}.out"
