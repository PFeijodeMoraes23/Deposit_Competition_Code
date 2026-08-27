#!/bin/bash
# ==============================================================================
# blp_run.sh — THE single front door for the RC-BLP demand-estimation sweep.
#
# WHAT MUST EXIST FIRST
#   1. gate G0 green                        (env_job.sh ENV_STEP=preflight)
#   2. the demand parquets and the logit deltas — gates G2 and G4
#   The sysimages and the draws are NOT prerequisites: this script builds whichever
#   of them is missing, which is what makes a bare run work on an empty cluster.
# WHAT TO RUN NEXT
#   bash pipeline_all.sh --from bbl         (BBL costs -> the counterfactuals),
#   which pipeline_all.sh chains for you when it runs this phase.
#
# WHAT IT SUBMITS
#   [sysimage build gpu+cpu] ---.
#                                >-- afterok --> per-routine chain --> afterany --> archive
#   [draws build]            ---'
#
#   AUTO-BUILD. With neither --sysimage nor --no-sysimage, the sysimage job is
#   submitted exactly when the image or its provenance sidecar is missing; with
#   neither --draws nor --no-draws, the draws job is submitted exactly when this
#   R/seed's halton_nu or demo_key_index is missing. An explicit flag always wins,
#   in both directions, and the decision is printed as one line before anything is
#   submitted. Nothing is rebuilt that already exists, so the common case — editing
#   our own .jl source — still submits neither: the sysimage bakes only third-party
#   PACKAGES (our files are include()d at runtime, blp_1_estimation.jl:37) and the
#   draws depend on the demographics/panel and R, not on our source.
#
#   --layout grouped (default)   HEAD(sigma+rc2+rc3+rc4+ext1) -> ext2 -> extended
#   --layout all                 sigma -> rc2 -> rc3 -> rc4 -> full -> ext1 -> ext2 -> extended
#   Both are afterok chains, so every stage warm-starts theta2 from the previous
#   stage's on-disk checkpoint and a wall-kill is resumable by plain resubmission
#   (the engine skips stages already on disk). ext1 warm-starts from rc4.
#
# Usage
#   bash blp_run.sh --dry-run                 # print every sbatch line + the graph
#   bash blp_run.sh                           # the usual run; builds what is missing
#   bash blp_run.sh --draws                   # force a rebuild of the draws
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
#   --sysimage             force the sysimage builds (gpu + cpu) as an afterok prerequisite
#   --no-sysimage          never build them, whatever is on disk
#   --draws                force regenerating the nu + demographic draws first
#   --no-draws             never build them, whatever is on disk
#                          (omit all four and the decision is made from what is on disk)
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
# R and SEED name the draw FILES the auto-build below tests for and the engine then
# reads, so they must be decided here rather than left to each consumer's own default:
# a mismatch would build halton_nu at one R and look for it at another.
R="${R:-2000}"; SEED="${SEED:-42}"
# _SET flags distinguish "the user chose" from "nobody said", which is what makes the
# auto-build possible: DO_SYSIMAGE=0 alone cannot tell --no-sysimage from silence.
SYS_SET=0; DRAWS_SET=0
ROUTINES_SRC=default
if [[ -n "${ROUTINES+set}" ]]; then ROUTINES_SRC=env; fi
ROUTINES="${ROUTINES:-${CL_ROUTINES_RC}}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --layout)          LAYOUT="$2"; shift ;;
        --routines)        ROUTINES="$2"; ROUTINES_SRC=flag; shift ;;
        --engines)         ENGINES="$2"; shift ;;
        --sysimage)        DO_SYSIMAGE=1; SYS_SET=1 ;;
        --no-sysimage)     DO_SYSIMAGE=0; SYS_SET=1 ;;
        --draws)           DO_DRAWS=1; DRAWS_SET=1 ;;
        --no-draws)        DO_DRAWS=0; DRAWS_SET=1 ;;
        --se-only)         SE_ONLY=1 ;;
        --sset)            DO_SSET=1 ;;
        --no-zip)          DO_ZIP=0 ;;
        --skip-preflight)  SKIP_PREFLIGHT=1 ;;
        --dry-run)         CL_DRYRUN=1 ;;
        -h|--help)         sed -n '2,59p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done
case "${LAYOUT}" in all|grouped) ;; *) echo "--layout must be all|grouped (got '${LAYOUT}')" >&2; exit 2 ;; esac

LOGD="$(cl_log_dir)"
cl_warn_wall_cap

# ── AUTO-BUILD: decide the two prerequisites from what is on disk ────────────
# This is what makes `bash pipeline_all.sh` work on an EMPTY cluster without anyone
# having to know which build flags a fresh tree needs. The tests are the artifacts
# themselves, at the paths their producers write:
#   sysimage  the .so AND its .json provenance sidecar — an image with no sidecar is
#             refused at run time by cl_require_sysimage, so a sidecar-less image is
#             not a usable image and must be rebuilt like a missing one.
#   draws     halton_nu AND demo_key_index for THIS R/seed. demo_draws itself is not
#             tested: it is the 14 GB member and it is written in the same job as the
#             key index, so the index standing in for it costs nothing and avoids a
#             stat on a file that may be mid-split.
# An explicit flag wins in both directions and skips the test entirely.
#
# The tests themselves are cl_sysimage_missing / cl_draws_missing in cluster_lib.sh, not
# inline stats, because gate G0 has to excuse exactly what this decides to build:
# pipeline_all.sh's submit_g0 calls the same two functions to derive --will-build, so the
# gate and the builder read one predicate instead of two copies of it.
if [[ "${SYS_SET}" == "0" ]] && cl_sysimage_missing gpu; then DO_SYSIMAGE=1; fi
if [[ "${DRAWS_SET}" == "0" ]] && cl_draws_missing "${R}" "${SEED}"; then DO_DRAWS=1; fi
cl_say "prereqs: sysimage=${DO_SYSIMAGE} draws=${DO_DRAWS}  (sysimage $([[ "${SYS_SET}" == "1" ]] && echo 'from the flag' || echo 'auto, from disk'), draws $([[ "${DRAWS_SET}" == "1" ]] && echo 'from the flag' || echo "auto, from disk at R=${R}/seed=${SEED}"))"

# ── Preflight (V2/V3/V5/V6). Never skipped by accident. ──────────────────────
if [[ "${SKIP_PREFLIGHT}" == "0" && "${CL_DRYRUN}" != "1" ]]; then
    # Jobs 1 and 2 of this run BUILD the sysimage and the draws, so the preflight must not
    # refuse on their absence. Derived from the SAME two booleans that decide whether those
    # jobs are submitted — flag or auto — so the two cannot disagree.
    _wb=""
    [[ "${DO_SYSIMAGE}" == "1" ]] && _wb="${_wb} sysimage"
    [[ "${DO_DRAWS}"    == "1" ]] && _wb="${_wb} draws"
    bash "${CL_ROOT}/cluster_preflight.sh" --routines "${ROUTINES}" ${_wb:+--will-build "${_wb# }"} || {
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
# EXPORTED so --export=ALL carries the SAME R/seed to the draws builder and to every
# stage job that then loads those draws. The auto-build test above is only meaningful
# if the value it tested is the value the jobs use.
export R SEED

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
          "R=${R} seed=${SEED}   prereqs: sysimage=${DO_SYSIMAGE} draws=${DO_DRAWS}   zip=${DO_ZIP}" \
          "results persist in ${CL_STEP_BLP}; draws in ${CL_STEP_DRAWS}"
[[ "${LAYOUT}" == "all" ]] && \
    echo "[note] --layout all includes the 'full' stage; --layout grouped omits it (see the header)."

# ── 1. Build prerequisites, in PARALLEL; every chain head waits afterok on BOTH.
# Whether either is submitted was decided above, from disk or from an explicit flag.
sys_dep=""
if [[ "${DO_SYSIMAGE}" == "1" ]]; then
    j=$(cl_sbatch --partition=gpu_h200 --gpus=h200:1 --time=04:00:00 \
        --cpus-per-task=8 --mem=64G -J blp_build_sysimg \
        -o "${LOGD}/blp_build_sysimg_%j.out" -e "${LOGD}/blp_build_sysimg_%j.err" \
        --export=ALL,ENV_STEP=sysimage_gpu "${CL_ROOT}/env_job.sh")
    sys_dep="${j}"
    cl_say "  blp_build_sysimg -> ${j}  (every chain head afterok this)"
    # The CPU twin, built in the same breath. pipeline_all.sh runs bbl_run.sh with
    # --fwd-cpu, whose jobs refuse without blp_sysimage_cpu.so — so a run that builds
    # only the GPU image cannot reach the BBL phase on a fresh cluster. Deliberately NOT
    # added to sys_dep: the RC ladders are GPU and do not read this image, and gating them
    # on a `day`-queued build would idle the H200 reservation for no reason. BBL is
    # submitted hours later by a continuation job, and bbl_run.sh preflights the image at
    # submit time, so the worst case is a clean refusal rather than a silent CPU fallback.
    jc=$(cl_sbatch --partition=day --constraint=cpugen:turin --time=04:00:00 \
        --cpus-per-task=8 --mem=64G -J blp_build_sysimg_cpu \
        -o "${LOGD}/blp_build_sysimg_cpu_%j.out" -e "${LOGD}/blp_build_sysimg_cpu_%j.err" \
        --export=ALL,ENV_STEP=sysimage_cpu "${CL_ROOT}/env_job.sh")
    cl_say "  blp_build_sysimg_cpu -> ${jc}  (for the BBL --fwd-cpu phase; runs in parallel)"
fi
if [[ "${DO_DRAWS}" == "1" ]]; then
    j=$(cl_sbatch -J blp_draws \
        -o "${LOGD}/blp_draws_%j.out" -e "${LOGD}/blp_draws_%j.err" \
        --export=ALL "${CL_ROOT}/blp_draws_job.sh")
    sys_dep="${sys_dep:+${sys_dep}:}${j}"
    cl_say "  blp_draws -> ${j}  (every chain head afterok this; runs parallel to the sysimage)"
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

# Every rung is the next rung's afterok dependency, so a refused submission cannot be
# absorbed: "afterok:" carrying no id reads to SLURM as NO dependency, and the rest of
# the ladder would run unsequenced against a sysimage that is still building. bash does
# not inherit errexit into a command substitution, so the check has to be explicit.
submit_rung () {   # same arguments as submit_one; prints the id, or fails loudly
    local jid
    jid=$(submit_one "$@") || true
    cl_require_jid "${jid}" "rc_$3_E$1_$6" || return 1
    printf '%s\n' "${jid}"
}

# ONE chain builder for both layouts. Echoes the TERMINAL job id on stdout;
# progress goes to stderr.
submit_chain () {   # submit_chain <routine> <engine> <tag> -> terminal job id
    local k="$1" eng="$2" tag="$3" prev="${sys_dep}" jid st wall
    if [[ "${LAYOUT}" == "all" ]]; then
        for st in "${STAGES_ALL[@]}"; do
            wall="$(cl_stage_wall "${st}")"
            jid=$(submit_rung "${k}" "${eng}" "${tag}" "${st}" "${wall}" "${st}" "${prev}") || return 1
            echo "    ${st}: ${jid}  (${prev:+afterok ${prev}, }wall=${wall}, mem=$(cl_mem_for "${k}"))" >&2
            prev="${jid}"
        done
    else
        jid=$(submit_rung "${k}" "${eng}" "${tag}" "${HEAD_STAGES}" "${WALL_HEAD}" "head" "${prev}") || return 1
        echo "    head (${HEAD_STAGES}): ${jid}  (${prev:+afterok ${prev}, }wall=${WALL_HEAD}, mem=$(cl_mem_for "${k}"))" >&2
        prev="${jid}"
        jid=$(submit_rung "${k}" "${eng}" "${tag}" "ext2" "$(cl_stage_wall ext2)" "ext2" "${prev}") || return 1
        echo "    ext2: ${jid}  (afterok ${prev}, wall=$(cl_stage_wall ext2))" >&2
        prev="${jid}"
        jid=$(submit_rung "${k}" "${eng}" "${tag}" "extended" "$(cl_stage_wall extended)" "ext" "${prev}") || return 1
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
    ckpt="${CL_STEP_BLP}/blp_checkpoint_E${k}_spec_12_extended.jls"
    jid=$(submit_rung "${k}" numerical num extended "$(cl_stage_wall extended)" xcheck "${dep}" \
                     "BLP_THETA2_INIT_FILE=${ckpt}") || return 1
    echo "    numerical xcheck: ${jid}  (afterok ${dep}; seed $(basename "${ckpt}"))" >&2
    printf '%s\n' "${jid}"
}

# ── 3. Submit ────────────────────────────────────────────────────────────────
njobs=0; term_jids=""
for k in ${ROUTINES}; do
    if [[ "${SE_ONLY}" == "1" ]]; then
        # No homotopy chain: every stage reloads its own checkpoint, so one job
        # does them all. Checkpoints live in data/output — if a previous run's
        # archive MOVED them into blp_checkpoints_<jid>.zip, unzip them back first.
        jid=$(submit_rung "${k}" ift se "${ALL_STAGES}" "${WALL_SE}" seonly "${sys_dep}")
        cl_say "  rc_se_E${k}_seonly -> ${jid}  (SE-only: all stages in ONE job)"
        njobs=$((njobs+1)); term_jids="${term_jids} ${jid}"
        continue
    fi
    ift_term=""
    if [[ "${do_ift}" == "1" ]]; then
        cl_log "-- chain: E${k} / ift (${LAYOUT}) --"
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
                ckpt="${CL_STEP_BLP}/blp_checkpoint_E${k}_spec_12_${st}.jls"
                jid=$(submit_rung "${k}" cue cue "${st}" "$(cl_stage_wall "${st}")" "cue_${st}" "${ift_term}" \
                                 "BLP_THETA2_INIT_FILE=${ckpt}")
                echo "    CUE ${st}: ${jid}  (afterok ${ift_term}; seed $(basename "${ckpt}"))"
                njobs=$((njobs+1)); term_jids="${term_jids} ${jid}"
            done
            if [[ "${DO_SSET}" == "1" ]]; then
                # Stock-Wright S-set: S(alpha0)=min_{theta2,beta} N*g'W(theta)g on a
                # grid of pinned alpha0, inverted into an identification-robust set
                # that ACCOUNTS for theta2 being estimated (unlike conditional AR/LM).
                ckpt="${CL_STEP_BLP}/blp_checkpoint_E${k}_spec_12_${SSET_STAGE}.jls"
                jid=$(submit_rung "${k}" cue sset "${SSET_STAGE}" "$(cl_stage_wall "${SSET_STAGE}")" sset "${ift_term}" \
                                 "BLP_THETA2_INIT_FILE=${ckpt},BLP_ALPHA_GRID=${SSET_GRID}")
                echo "    S-set ${SSET_STAGE} [${SSET_GRID}]: ${jid}  (afterok ${ift_term})"
                njobs=$((njobs+1)); term_jids="${term_jids} ${jid}"
            fi
        fi
    fi
done
cl_log "Submitted ${njobs} RC-BLP jobs (routines: ${ROUTINES}; engines: ${ENGINES}; layout: ${LAYOUT})${sys_dep:+ + build prereqs [afterok ${sys_dep}]}."

# ── 4. Terminal archive, afterANY every chain, on EVERY layout ───────────────
# afterany (not afterok) so partial results + logs still get bundled if a stage
# wall-kills.
#
# COPY, never MOVE, and no --newer filter. The RC results have to STAY in
# data/output/blp: foundation_demand_eval.jl's _result_path opens them there for the
# whole BBL and CF phase, an --se-only rerun reloads the checkpoints in place, and
# the download is meant to be complete rather than incremental — a --newer window
# would silently drop a rung an earlier resume produced. The archiver writes the zip
# into data/output/download, alongside every other set.
dep_csv="$(echo ${term_jids} | tr ' ' ':' | sed 's/^://; s/:$//')"
# pipeline_all.sh reads BLP_TERM_JOBIDS to build the BBL hand-off. Exiting nonzero is
# the honest verdict when no ladder was sequenced: a COMPLETED job publishing an empty
# terminal list strands the BBL phase with nothing to wait on and no error to explain it.
if [[ -z "${dep_csv}" ]]; then
    cl_err "FATAL: no RC-BLP job was sequenced (routines: ${ROUTINES}); see the refusals above."
    echo "BLP_TERM_JOBIDS="
    exit 1
fi
if [[ "${DO_ZIP}" == "1" ]]; then
    ZWRAP="cd '${CL_ROOT}'"
    for s in blp logs; do
        ZWRAP="${ZWRAP} && bash cluster_archive.sh --set ${s} --copy --tag \"\${SLURM_JOB_ID}\""
    done
    zip_jid=$(cl_sbatch --dependency=afterany:${dep_csv} \
        -J blp_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGD}/blp_zip_%j.out" -e "${LOGD}/blp_zip_%j.err" \
        --wrap "${ZWRAP}")
    cl_say "  blp_zip -> ${zip_jid}  (afterany ${dep_csv//:/, }; sets: blp logs, --copy)"
    cl_log "   -> ${CL_STEP_DOWNLOAD}/blp_outputs_<tag>.zip and logs_<tag>.zip"
    cl_log "      the results themselves stay in ${CL_STEP_BLP} for the BBL/CF phases"
else
    [[ "${DO_ZIP}" == "1" ]] || cl_say "  archive skipped (--no-zip). By hand later:  bash cluster_archive.sh --set blp --copy"
fi

cl_log "Watch with: squeue -u \$USER"
echo "BLP_TERM_JOBIDS=${dep_csv}"
