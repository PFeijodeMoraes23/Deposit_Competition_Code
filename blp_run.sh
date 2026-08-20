#!/bin/bash
# ==============================================================================
# blp_run.sh — THE single front door for the RC-BLP demand-estimation sweep.
#
# WHAT MUST EXIST FIRST
#   1. bash cluster_preflight.sh            (must print PREFLIGHT OK)
#   2. the GPU sysimage                     (RUNBOOK step 2, or pass --sysimage)
#   3. the R=2000 draws                     (or pass --draws the first time)
# WHAT TO RUN NEXT
#   bash pipeline_run.sh                    (BBL costs -> the counterfactuals)
#
# THIS SCRIPT IS NEW. submit_blp_rc_all.sh / submit_blp_rc_grouped.sh /
# submit_blp_build_and_run_spec12.sh / submit_blp_2_rc_default.sh are still
# present and still work; they are the fallback and are retired only after one
# successful cluster cycle. Nothing in them has been modified.
#
# WHAT IT SUBMITS
#   [sysimage build] ---.
#                        >-- afterok --> per-routine chain --> afterany --> archive
#   [draws build]    ---'
#
#   --layout grouped (default)   HEAD(sigma+rc2+rc3+rc4+ext1) -> ext2 -> extended
#   --layout all                 sigma -> rc2 -> rc3 -> rc4 -> full -> ext1 -> ext2 -> extended
#   Both are afterok chains, so every stage warm-starts theta2 from the previous
#   stage's on-disk checkpoint and a wall-kill is resumable by plain resubmission
#   (the engine skips stages already on disk). ext1 warm-starts from rc4.
#
# Usage
#   bash blp_run.sh --dry-run                 # print every sbatch line + the graph
#   bash blp_run.sh                           # the usual run
#   bash blp_run.sh --draws                   # first run on a fresh cluster
#   bash blp_run.sh --layout all
#   bash blp_run.sh --routines "1 2 3 4"
#   bash blp_run.sh --engines "ift numerical" # + the numerical cross-check
#   bash blp_run.sh --engines "ift cue" --sset
#   bash blp_run.sh --se-only                 # recompute SEs from the checkpoints
#
# Flags
#   --layout all|grouped   job packaging (default grouped)
#   --routines "3 4"       routine set (default from cluster_lib.sh)
#   --engines  "ift ..."   ift | numerical | cue (default ift)
#   --sysimage             rebuild blp_sysimage.so first (afterok prerequisite)
#   --draws                regenerate the nu + demographic draws first
#   --se-only              SE-only rerun: all stages in ONE short job per routine
#   --sset                 add the Stock-Wright S-set grid job (needs cue)
#   --no-zip               do not submit the terminal archive job
#   --skip-preflight       skip cluster_preflight.sh
#   --dry-run              print, submit nothing
#   -h                     this header
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"
set -e

LAYOUT="${LAYOUT:-grouped}"
ENGINES="${ENGINES:-ift}"
DO_SYSIMAGE=0; DO_DRAWS=0; SE_ONLY="${SE_ONLY:-0}"; DO_SSET="${CUE_SSET:-0}"
DO_ZIP=1; SKIP_PREFLIGHT=0
ROUTINES_SRC=default
if [[ -n "${ROUTINES+set}" ]]; then ROUTINES_SRC=env; fi
ROUTINES="${ROUTINES:-${CL_ROUTINES_RC}}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --layout)          LAYOUT="$2"; shift ;;
        --routines)        ROUTINES="$2"; ROUTINES_SRC=flag; shift ;;
        --engines)         ENGINES="$2"; shift ;;
        --sysimage)        DO_SYSIMAGE=1 ;;
        --no-sysimage)     DO_SYSIMAGE=0 ;;
        --draws)           DO_DRAWS=1 ;;
        --no-draws)        DO_DRAWS=0 ;;
        --se-only)         SE_ONLY=1 ;;
        --sset)            DO_SSET=1 ;;
        --no-zip)          DO_ZIP=0 ;;
        --skip-preflight)  SKIP_PREFLIGHT=1 ;;
        --dry-run)         CL_DRYRUN=1 ;;
        -h|--help)         sed -n '2,58p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done
case "${LAYOUT}" in all|grouped) ;; *) echo "--layout must be all|grouped (got '${LAYOUT}')" >&2; exit 2 ;; esac

LOGD="$(cl_log_dir)"
cl_warn_wall_cap

# ── Preflight (V2/V3/V5/V6). Never skipped by accident. ──────────────────────
if [[ "${SKIP_PREFLIGHT}" == "0" && "${CL_DRYRUN}" != "1" ]]; then
    bash "${CL_ROOT}/cluster_preflight.sh" --routines "${ROUTINES}" || {
        echo "" >&2
        echo "blp_run.sh: refusing to submit — the preflight found blockers (above)." >&2
        echo "  Act on them, or re-run with --skip-preflight if you know better." >&2
        exit 1; }
fi

# ── Engine allow-list (same defence-in-depth as blp_stage_job.sh) ────────────
do_ift=0; do_num=0; do_cue=0
for e in ${ENGINES}; do
    case "$e" in
        ift)       do_ift=1 ;;
        numerical) do_num=1 ;;
        cue)       do_cue=1 ;;
        *) echo "unknown engine token '$e' in --engines (valid: ift numerical cue)" >&2; exit 2 ;;
    esac
done

# ── SE knobs, read by the engine. Exported on EVERY layout — the grouped front
#    door used to omit them, so which door you picked silently chose the SE method.
SE_METHOD="${SE_METHOD:-wcb}"          # wcb (default, matches the sleepiness inference) | sandwich
WCB_REPS="${WCB_REPS:-999}"; WCB_SCHEME="${WCB_SCHEME:-webb}"
export BLP_SE_METHOD="${SE_METHOD}" BLP_WCB_REPS="${WCB_REPS}" BLP_WCB_SCHEME="${WCB_SCHEME}"
export BLP_SE_ONLY="${SE_ONLY}"

# ── CUE (continuously-updated GMM): a SIDE-BY-SIDE variant, never a replacement.
# W(theta)=pinv(Omega-hat(theta)) recomputed at every objective evaluation (the
# LIML analogue) instead of the fixed W=(Z'Z/N)^-1. Every artifact carries the
# _cue suffix (blp_2_rc.jl ENGINE_SUFFIX), so the IFT results — and every
# downstream consumer, which reads exact ift filenames — are untouched.
CUE_STAGES="${CUE_STAGES:-ext1}"       # '+'-separated; ext1 is the reported headline rung
SSET_STAGE="${SSET_STAGE:-ext1}"
SSET_GRID="${SSET_GRID:--1.5:0.125:1.5}"

GENERIC="${CL_ROOT}/blp_stage_job.sh"
STAGES_ALL=(sigma rc2 rc3 rc4 full ext1 ext2 extended)
# `full` is in the 'all' ladder and NOT in 'grouped': blp_1_estimation.jl:1334
# documents it as reproducing rc4 exactly since sigma(ln assets) was dropped, and
# ext1 warm-starts from rc4, not from full. Which of the two is right is a
# computational-semantics call belonging to the .jl author, so both are preserved
# exactly as they stand and 'grouped' (the default) simply avoids the cost.
HEAD_STAGES="sigma+rc2+rc3+rc4+ext1"
ALL_STAGES="sigma+rc2+rc3+rc4+ext1+ext2+extended"    # --se-only: one job, no chain
WALL_HEAD="${WALL_HEAD:-2-00:00:00}"
WALL_SE="${WALL_SE:-04:00:00}"

cl_banner "RC-BLP sweep${CL_DRYRUN:+ }$([[ "${CL_DRYRUN}" == "1" ]] && echo '[DRY RUN — nothing is submitted]')" \
          "$(cl_routines_provenance "${ROUTINES}" "${ROUTINES_SRC}" "${CL_ROUTINES_RC}")" \
          "layout=${LAYOUT}  engines='${ENGINES}'  se_only=${SE_ONLY}  sset=${DO_SSET}" \
          "SE: ${SE_METHOD} (WCB reps=${WCB_REPS}, scheme=${WCB_SCHEME}) — propagated via --export=ALL" \
          "mem: E1/E2=${CL_MEM_BIG}, others=${CL_MEM_DEFAULT}   wall_deep=${CL_WALL_DEEP}" \
          "prereqs: sysimage=${DO_SYSIMAGE} draws=${DO_DRAWS}   zip=${DO_ZIP}"
[[ "${LAYOUT}" == "all" ]] && \
    echo "[note] --layout all includes the 'full' stage; --layout grouped omits it (see the header)."

# ── 1. Build prerequisites, in PARALLEL; every chain head waits afterok on BOTH.
# Both are OPT-IN because the common case — editing our own .jl source — needs
# NEITHER: the sysimage bakes only third-party PACKAGES (our files are include()d
# at runtime, blp_1_estimation.jl:37), and the draws depend on the
# demographics/panel and R, not on the instruments or on our source.
sys_dep=""
if [[ "${DO_SYSIMAGE}" == "1" ]]; then
    j=$(cl_sbatch --partition=gpu_h200 --gpus=h200:1 --time=04:00:00 \
        --cpus-per-task=8 --mem=64G -J blp_build_sysimg \
        -o "${LOGD}/blp_build_sysimg_%j.out" -e "${LOGD}/blp_build_sysimg_%j.err" \
        --export=ALL,ENV_STEP=sysimage_gpu "${CL_ROOT}/env_job.sh")
    sys_dep="${j}"
    echo "-- sysimage build: ${j}  (every chain head afterok this)"
fi
if [[ "${DO_DRAWS}" == "1" ]]; then
    j=$(cl_sbatch -J blp_draws \
        -o "${LOGD}/blp_draws_%j.out" -e "${LOGD}/blp_draws_%j.err" \
        --export=ALL "${CL_ROOT}/blp_draws_job.sh")
    sys_dep="${sys_dep:+${sys_dep}:}${j}"
    echo "-- draws build:    ${j}  (every chain head afterok this; runs parallel to the sysimage)"
fi

# ── 2. One stage submission. cl_mem_for is applied on EVERY submit. ──────────
submit_one () {   # submit_one <routine> <engine> <tag> <stage> <wall> <jobtag> [dep_ids] [extra_export]
    local k="$1" eng="$2" tag="$3" st="$4" wall="$5" jt="$6" dep="${7:-}" extra="${8:-}"
    cl_sbatch --time="${wall}" --mem="$(cl_mem_for "${k}")" \
        ${dep:+--dependency=afterok:${dep}} \
        --export=ALL,RC_ROUTINE=${k},RC_ENGINE=${eng},RC_STAGE=${st}${extra:+,${extra}} \
        -J "rc_${tag}_E${k}_${jt}" \
        -o "${LOGD}/rc_${tag}_E${k}_${jt}_%j.out" \
        -e "${LOGD}/rc_${tag}_E${k}_${jt}_%j.err" \
        "${GENERIC}"
}

# ONE chain builder for both layouts. Echoes the TERMINAL job id on stdout;
# progress goes to stderr.
submit_chain () {   # submit_chain <routine> <engine> <tag> -> terminal job id
    local k="$1" eng="$2" tag="$3" prev="${sys_dep}" jid st wall
    if [[ "${LAYOUT}" == "all" ]]; then
        for st in "${STAGES_ALL[@]}"; do
            wall="$(cl_stage_wall "${st}")"
            jid=$(submit_one "${k}" "${eng}" "${tag}" "${st}" "${wall}" "${st}" "${prev}")
            echo "    ${st}: ${jid}  (${prev:+afterok ${prev}, }wall=${wall}, mem=$(cl_mem_for "${k}"))" >&2
            prev="${jid}"
        done
    else
        jid=$(submit_one "${k}" "${eng}" "${tag}" "${HEAD_STAGES}" "${WALL_HEAD}" "head" "${prev}")
        echo "    head (${HEAD_STAGES}): ${jid}  (${prev:+afterok ${prev}, }wall=${WALL_HEAD}, mem=$(cl_mem_for "${k}"))" >&2
        prev="${jid}"
        jid=$(submit_one "${k}" "${eng}" "${tag}" "ext2" "$(cl_stage_wall ext2)" "ext2" "${prev}")
        echo "    ext2: ${jid}  (afterok ${prev}, wall=$(cl_stage_wall ext2))" >&2
        prev="${jid}"
        jid=$(submit_one "${k}" "${eng}" "${tag}" "extended" "$(cl_stage_wall extended)" "ext" "${prev}")
        echo "    extended: ${jid}  (afterok ${prev}, wall=$(cl_stage_wall extended))" >&2
        prev="${jid}"
    fi
    printf '%s\n' "${prev}"
}

# The numerical cross-check, emitted by ONE function instead of three verbatim
# copies. Its value is confirming the analytical gradient AT the optimum, so it is
# ONE job at `extended`, afterok the IFT terminal and seeded from its theta2.
submit_crosscheck () {   # submit_crosscheck <routine> <ift_terminal_jid>
    local k="$1" dep="$2" ckpt jid
    ckpt="${CL_DATA_OUT}/blp_checkpoint_E${k}_spec_12_extended.jls"
    jid=$(submit_one "${k}" numerical num extended "$(cl_stage_wall extended)" xcheck "${dep}" \
                     "BLP_THETA2_INIT_FILE=${ckpt}")
    echo "    numerical xcheck: ${jid}  (afterok ${dep}; seed $(basename "${ckpt}"))" >&2
    printf '%s\n' "${jid}"
}

# ── 3. Submit ────────────────────────────────────────────────────────────────
njobs=0; term_jids=""
MARKER="$(cl_run_marker blp)"
for k in ${ROUTINES}; do
    if [[ "${SE_ONLY}" == "1" ]]; then
        # No homotopy chain: every stage reloads its own checkpoint, so one job
        # does them all. Checkpoints live in data/output — if a previous run's
        # archive MOVED them into blp_checkpoints_<jid>.zip, unzip them back first.
        jid=$(submit_one "${k}" ift se "${ALL_STAGES}" "${WALL_SE}" seonly "${sys_dep}")
        echo "-- E${k}: SE-only, all stages in ONE job: ${jid}"
        njobs=$((njobs+1)); term_jids="${term_jids} ${jid}"
        continue
    fi
    ift_term=""
    if [[ "${do_ift}" == "1" ]]; then
        echo "-- chain: E${k} / ift (${LAYOUT}) --"
        ift_term=$(submit_chain "${k}" ift ift)
        njobs=$(( njobs + $([[ "${LAYOUT}" == "all" ]] && echo 8 || echo 3) ))
        term_jids="${term_jids} ${ift_term}"
    fi
    if [[ "${do_num}" == "1" ]]; then
        if [[ -n "${ift_term}" ]]; then
            jid=$(submit_crosscheck "${k}" "${ift_term}")
            njobs=$((njobs+1)); term_jids="${term_jids} ${jid}"
        else
            echo "-- chain: E${k} / numerical (full, no IFT to seed from) --"
            jid=$(submit_chain "${k}" numerical num)
            njobs=$(( njobs + $([[ "${LAYOUT}" == "all" ]] && echo 8 || echo 3) ))
            term_jids="${term_jids} ${jid}"
        fi
    fi
    if [[ "${do_cue}" == "1" ]]; then
        if [[ -z "${ift_term}" ]]; then
            echo "[!] --engines cue needs ift in the same invocation (the cue jobs are seeded from its checkpoints) — skipped." >&2
        else
            # ONE JOB PER STAGE, not a grouped '+' list: a grouped job would seed
            # every stage from the same BLP_THETA2_INIT_FILE, and the _cue
            # prev-stage checkpoint chain does not exist. Each is seeded from the
            # IFT checkpoint of the SAME stage.
            for st in $(echo "${CUE_STAGES}" | tr '+' ' '); do
                ckpt="${CL_DATA_OUT}/blp_checkpoint_E${k}_spec_12_${st}.jls"
                jid=$(submit_one "${k}" cue cue "${st}" "$(cl_stage_wall "${st}")" "cue_${st}" "${ift_term}" \
                                 "BLP_THETA2_INIT_FILE=${ckpt}")
                echo "    CUE ${st}: ${jid}  (afterok ${ift_term}; seed $(basename "${ckpt}"))"
                njobs=$((njobs+1)); term_jids="${term_jids} ${jid}"
            done
            if [[ "${DO_SSET}" == "1" ]]; then
                # Stock-Wright S-set: S(alpha0)=min_{theta2,beta} N*g'W(theta)g on a
                # grid of pinned alpha0, inverted into an identification-robust set
                # that ACCOUNTS for theta2 being estimated (unlike conditional AR/LM).
                ckpt="${CL_DATA_OUT}/blp_checkpoint_E${k}_spec_12_${SSET_STAGE}.jls"
                jid=$(submit_one "${k}" cue sset "${SSET_STAGE}" "$(cl_stage_wall "${SSET_STAGE}")" sset "${ift_term}" \
                                 "BLP_THETA2_INIT_FILE=${ckpt},BLP_ALPHA_GRID=${SSET_GRID}")
                echo "    S-set ${SSET_STAGE} [${SSET_GRID}]: ${jid}  (afterok ${ift_term})"
                njobs=$((njobs+1)); term_jids="${term_jids} ${jid}"
            fi
        fi
    fi
done
echo "Submitted ${njobs} RC-BLP jobs (routines: ${ROUTINES}; engines: ${ENGINES}; layout: ${LAYOUT})${sys_dep:+ + build prereqs [afterok ${sys_dep}]}."

# ── 4. Terminal archive, afterANY every chain, on EVERY layout ───────────────
# afterany (not afterok) so partial results + logs still get bundled if a stage
# wall-kills. MOVE mode with --newer ${MARKER}: only files this run produced are
# taken, so an earlier run's outputs are neither re-archived nor deleted.
dep_csv="$(echo ${term_jids} | tr ' ' ':' | sed 's/^://; s/:$//')"
if [[ "${DO_ZIP}" == "1" && -n "${dep_csv}" ]]; then
    ZWRAP="cd '${CL_ROOT}'"
    for s in blp_outputs blp_logs blp_checkpoints; do
        ZWRAP="${ZWRAP}; bash cluster_archive.sh --set ${s} --move --newer '${MARKER}' --tag \"\${SLURM_JOB_ID}\""
    done
    ZWRAP="${ZWRAP}; rm -f '${MARKER}'"
    zip_jid=$(cl_sbatch --dependency=afterany:${dep_csv} \
        -J blp_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGD}/blp_zip_%j.out" -e "${LOGD}/blp_zip_%j.err" \
        --wrap "${ZWRAP}")
    echo "-- archive: ${zip_jid}  (afterany ${dep_csv//:/, })"
    echo "   -> ${CL_DATA_OUT}/blp_outputs_<${zip_jid}>.zip      (process_blp_outputs.py auto-discovers blp_outputs_*.zip)"
    echo "   -> ${CL_DATA_OUT}/blp_logs_<${zip_jid}>.zip"
    echo "   -> ${CL_DATA_OUT}/blp_checkpoints_<${zip_jid}>.zip  (warm-start .jls MOVED off data/output;"
    echo "      unzip them back into data/output before any --se-only run or stage resume)"
else
    [[ "${CL_DRYRUN}" == "1" ]] || rm -f "${MARKER}"
    [[ "${DO_ZIP}" == "1" ]] || echo "-- archive skipped (--no-zip). Archive by hand later:"
    [[ "${DO_ZIP}" == "1" ]] || echo "     bash cluster_archive.sh --set blp_outputs --copy"
fi

echo "Watch with: squeue -u \$USER"
echo "BLP_TERM_JOBIDS=${dep_csv}"
