#!/bin/bash
# ==============================================================================
# pipeline_all.sh — THE single command. Sleepiness -> logit -> RC-BLP -> BBL costs
# -> the counterfactuals -> the equilibrium counterfactuals, in order, on the
# cluster, from one invocation.
#
#   bash pipeline_all.sh --dry-run     # print the whole graph, submit nothing
#   bash pipeline_all.sh               # do it
#   bash pipeline_all.sh --from blp    # resume at a phase
#
# WHAT MUST EXIST FIRST
#   the uploaded inputs (cluster/upload_manifest.txt) and the two sysimages.
#   Everything else this run builds for itself.
# WHAT TO RUN NEXT
#   Nothing. Watch it with `squeue -u $USER` and read the gate jsons:
#       for g in G1 G2 G3 G4 G7; do python -m json.tool ../data/output/.gate_$g.json; done
#
# ------------------------------------------------------------------------------
# THE SHAPE OF THE RUN, and why it is not one flat dependency graph
# ------------------------------------------------------------------------------
#   G0 (login node, inline — not a job)
#     |
#   sleep_run.sh ....... est array -> [G1] -> prep -> [G2] -+-> ame_gate -> ame -> [G3]
#     |                                                     +-> upsilon -> [G7]
#   logit_job.sh (afterok G2) -> [G4]
#     |
#   =====  continuation job, afterok G2:G4  ==============================
#   blp_run.sh ......... sysimage/draws -> per-routine RC ladders -> archive
#   rc_stage ........... unpack the newest blp_outputs zip into cluster_processed/
#     |
#   =====  continuation job, afterok rc_stage  ===========================
#   bbl_run.sh --fwd-cpu ....... warmup -> polfunc -> fwd_sim array -> solve
#   cf_run.sh (phase 1b) ....... demand_eval + cf4, no re-eval, needs no costs
#   cf_run.sh (phase 2) ........ cf1 + cf1_net + cf4 re-eval, --cost-afterok the solves
#     |
#   =====  continuation job, afterok the CF results  =====================
#   cf_eq_run.sh ....... the firm-sharded Jacobi equilibria
#   terminal archives .. afterany, --copy
#
# THE CONTINUATION JOBS ARE THE POINT, so they are worth stating plainly.
# blp_run.sh, bbl_run.sh and cf_eq_run.sh all preflight their inputs ON DISK at
# SUBMIT time — RC .jls, cost params, draws — and none of them takes an "assume it
# is coming, chain afterok instead" flag the way cf_run.sh's --cost-afterok does.
# Submitting all six phases at t=0 would therefore refuse at phase 3 with a
# perfectly correct MISSING report, because at t=0 the file really is missing.
#
# So each phase boundary submits a ten-minute CPU job that re-invokes THIS script
# with --from <next-phase>. The preflight then runs at the moment its inputs
# exist, which is the moment it is actually informative, and a resume by hand is
# the same command the pipeline runs itself. `--dry-run` walks every phase inline
# instead, so one command still prints the whole graph.
#
# GOTCHA: a job cancelled as DependencyNeverSatisfied writes NO LOG AT ALL. If a
# phase never starts, read the gate json and the .err of the phase before it — the
# missing log is the symptom, never the cause.
#
# Flags
#   --from <phase>     sleep | logit | blp | bbl | cf | cfeq   (default sleep)
#   --routines "3 4"   routine set for the RC/BBL/CF phases (default from cluster_lib.sh)
#   --sleep-routines   routine set for the sleepiness phase (default: all four)
#   --skip-est         pass --skip-est to sleep_run.sh (no re-estimation)
#   --no-sleep         skip the sleepiness phase entirely (same as --from logit)
#   --no-logit         skip logit_job.sh and G4
#   --no-blp           skip the RC-BLP phase (its results must be on disk)
#   --no-bbl           skip the BBL cost phase (cost params must be on disk)
#   --no-cf            skip the CF phase
#   --no-cfeq          skip the equilibrium CF phase
#   --cfeq-modes "cf3" which equilibrium CFs to run (cf3 cf5 cf6; cf6 needs CF_MERGE)
#   --blp-args "..."   extra flags forwarded verbatim to blp_run.sh (e.g. "--draws --sysimage")
#   --skip-preflight   skip G0's cluster_preflight.sh
#   --resumed          internal: set by the continuation jobs; skips G0
#   --dry-run          print every sbatch line, submit nothing
#   -h                 this header
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"
set -e

PHASES="sleep logit blp bbl cf cfeq"
FROM="sleep"
ROUTINES_SRC=default
if [[ -n "${ROUTINES+set}" ]]; then ROUTINES_SRC=env; fi
ROUTINES="${ROUTINES:-${CL_ROUTINES_CF}}"
SLEEP_ROUTINES="${SLEEP_ROUTINES:-${CL_ROUTINES_ALL}}"
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
CFEQ_MODES="${CFEQ_MODES:-cf3}"
BLP_ARGS="${BLP_ARGS:-}"
SKIP_EST=""; SKIP_PF=""; RESUMED=0; DRY=""
DO_SLEEP=1; DO_LOGIT=1; DO_BLP=1; DO_BBL=1; DO_CF=1; DO_CFEQ=1
FINAL_ZIP_SETS="${FINAL_ZIP_SETS:-foundation cf1 cf4}"
# gpu_h200 is the default and stays the default: it is the only partition whose
# 141 GB of vRAM the RC ladders are known to fit in. The cluster also offers
# gpu_h100 (12 jobs / 32 GPUs, 80 GB) and gpu_rtx6000 (16 jobs / 16 GPUs), and
# this is the one place to point the GPU-consuming children at one of them.
# blp_run.sh's sysimage build names gpu_h200 directly and does not read this.
GPU_PARTITION="${GPU_PARTITION:-gpu_h200}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)           FROM="$2"; shift ;;
        --routines)       ROUTINES="$2"; ROUTINES_SRC=flag; shift ;;
        --sleep-routines) SLEEP_ROUTINES="$2"; shift ;;
        --skip-est)       SKIP_EST="--skip-est" ;;
        --no-sleep)       DO_SLEEP=0 ;;
        --no-logit)       DO_LOGIT=0 ;;
        --no-blp)         DO_BLP=0 ;;
        --no-bbl)         DO_BBL=0 ;;
        --no-cf)          DO_CF=0 ;;
        --no-cfeq)        DO_CFEQ=0 ;;
        --cfeq-modes)     CFEQ_MODES="$2"; shift ;;
        --blp-args)       BLP_ARGS="$2"; shift ;;
        --skip-preflight) SKIP_PF="--skip-preflight" ;;
        --resumed)        RESUMED=1 ;;
        --dry-run)        CL_DRYRUN=1; DRY="--dry-run" ;;
        -h|--help)        sed -n '2,74p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done
case " ${PHASES} " in *" ${FROM} "*) ;;
    *) echo "--from must be one of: ${PHASES} (got '${FROM}')" >&2; exit 2 ;;
esac

LOGD="$(cl_log_dir)"
export CF_STAGE R SEED GPU_PARTITION
# A dry run prints the SUBMISSION GRAPH; the files each phase consumes are built by
# the phase before it and cannot exist yet on the login node at t=0. This tells the
# children's on-disk preflights to report and continue instead of refusing, and it
# is inert unless CL_DRYRUN=1 (see cluster_lib.sh section 6).
if [[ "${CL_DRYRUN}" == "1" ]]; then export CL_ASSUME_CHAINED=1; fi

phase_index () { local i=0 p; for p in ${PHASES}; do [[ "${p}" == "$1" ]] && { echo "${i}"; return; }; i=$((i+1)); done; echo 99; }
FROM_IX="$(phase_index "${FROM}")"

# WHICH PHASES CAN SHARE ONE INVOCATION. Two can, and for the same reason: the
# later one needs no file the earlier one has yet to write, only a job id.
#   A = sleep + logit  logit is one sbatch chained afterok G2; nothing to preflight
#   C = bbl   + cf     cf_run.sh --cost-afterok takes the solve job ids INSTEAD of
#                      the cost params, which is exactly pipeline_run.sh's phase 2
# B (blp) and D (cfeq) stand alone: blp_run.sh needs the parquets and deltas ON
# DISK, cf_eq_run.sh needs the cost params and RC results ON DISK, and neither
# takes a dependency flag. They get a continuation job instead.
phase_group () {
    case "$1" in sleep|logit) echo A ;; blp) echo B ;; bbl|cf) echo C ;; cfeq) echo D ;; *) echo Z ;; esac
}
FROM_GROUP="$(phase_group "${FROM}")"

# runs_now: the LIVE rule, and the one the phase boundaries below test — a dry run
# must still show where a live run would stop and hand off.
runs_now () {   # runs_now <name>
    local ix; ix="$(phase_index "$1")"
    [[ ${ix} -ge ${FROM_IX} ]] || return 1
    [[ "$(phase_group "$1")" == "${FROM_GROUP}" ]] || return 1
    return 0
}
# want_phase: what this invocation actually executes. Identical to runs_now except
# that a dry run walks every remaining phase inline, so one command prints one
# whole graph instead of stopping at the first hand-off.
want_phase () {   # want_phase <name>
    local ix; ix="$(phase_index "$1")"
    [[ ${ix} -ge ${FROM_IX} ]] || return 1
    [[ "${CL_DRYRUN}" == "1" ]] && return 0
    runs_now "$1"
}

cl_banner "FULL PIPELINE$([[ "${CL_DRYRUN}" == "1" ]] && echo '  [DRY RUN — nothing is submitted]')" \
          "from=${FROM}$([[ "${RESUMED}" == "1" ]] && echo '  (resumed by a continuation job)')" \
          "$(cl_routines_provenance "${ROUTINES}" "${ROUTINES_SRC}" "${CL_ROUTINES_CF}")" \
          "sleep routines='${SLEEP_ROUTINES}'  stage=${CF_STAGE}  R=${R}  seed=${SEED}" \
          "phases: sleep=${DO_SLEEP} logit=${DO_LOGIT} blp=${DO_BLP} bbl=${DO_BBL} cf=${DO_CF} cfeq=${DO_CFEQ}"

ALL_JIDS=""
add_jid () { [[ -n "$1" ]] && ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}$1"; return 0; }

# Capture a child WITHOUT letting set -e abort before its output is shown. The
# child's preflight prints its MISSING diagnostics on stdout, and dying on the
# $(...) assignment swallows them entirely. Same pattern at every capture site.
CHILD_OUT=""; CHILD_RC=0
run_child () {   # run_child <label> <command...>
    local label="$1"; shift
    echo; echo "=== ${label} ==="
    set +e; CHILD_OUT="$("$@")"; CHILD_RC=$?; set -e
    printf '%s\n' "${CHILD_OUT}"
    return 0
}
child_field () { printf '%s\n' "${CHILD_OUT}" | sed -n "s/^$1=//p" | tail -n1; }
die_child () {   # die_child <label> <rc>
    echo "ERROR: $1 failed (rc=$2) — its output is above (usually a preflight MISSING)." >&2
    exit "$2"
}

# One continuation: a ten-minute CPU job that re-invokes this script at the next
# phase, once <dep> is satisfied. Nothing about the pipeline's shape lives in the
# wrap string beyond the flags a human would type.
continue_at () {   # continue_at <phase> <dep_ids> -> job id
    local ph="$1" dep="$2" jid wrap
    wrap="cd '${CL_ROOT}' && bash pipeline_all.sh --from ${ph} --resumed"
    wrap="${wrap} --routines '${ROUTINES}' --sleep-routines '${SLEEP_ROUTINES}'"
    wrap="${wrap} --cfeq-modes '${CFEQ_MODES}'"
    [[ "${DO_BLP}"  == "1" ]] || wrap="${wrap} --no-blp"
    [[ "${DO_BBL}"  == "1" ]] || wrap="${wrap} --no-bbl"
    [[ "${DO_CF}"   == "1" ]] || wrap="${wrap} --no-cf"
    [[ "${DO_CFEQ}" == "1" ]] || wrap="${wrap} --no-cfeq"
    [[ -n "${BLP_ARGS}" ]] && wrap="${wrap} --blp-args '${BLP_ARGS}'"
    # 30 minutes, not 10: the resumed invocation runs the next phase's FULL preflight
    # (and the bbl+cf group runs three of them), each of which loads the Julia module
    # and probes both sysimages. It submits and exits; the wall is slack, not work.
    #
    # ASSUMPTION, and the one this whole design rests on: sbatch is callable from
    # inside a batch job. It is on Bouchet — every one of these continuations does
    # nothing else.
    jid=$(cl_sbatch --dependency=afterok:"${dep}" \
        -J "pipe_next_${ph}" --partition="${ZIP_PARTITION:-day}" --time=00:30:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G \
        -o "${LOGD}/pipe_next_${ph}_%j.out" -e "${LOGD}/pipe_next_${ph}_%j.err" \
        --wrap "${wrap}")
    # Progress on stderr, the job id on stdout — the same split submit_chain uses in
    # blp_run.sh, so `jid=$(continue_at ...)` captures an id and not a paragraph.
    echo "-- continuation: phase '${ph}' resumes in job ${jid}  (afterok ${dep})" >&2
    echo "   it runs:  ${wrap#cd * && }" >&2
    # No add_jid here: this function is always called inside $( ), i.e. a subshell,
    # so an assignment made here would not survive. defer_to records the id.
    printf '%s\n' "${jid}"
}

# defer_to: submit the continuation and stop. A dry run prints the same sbatch line
# and then keeps walking, with the hand-off marked, so the graph is both complete
# and honest about where the live run actually pauses.
defer_to () {   # defer_to <phase> <dep_ids>
    local ph="$1" dep="$2" jid
    [[ -n "${dep}" ]] || { echo "-- no dependency to chain '${ph}' on; nothing deferred."; return 0; }
    jid="$(continue_at "${ph}" "${dep}")"
    add_jid "${jid}"
    if [[ "${CL_DRYRUN}" == "1" ]]; then
        echo "   [dry-run] A LIVE RUN STOPS HERE. Everything printed below is what job ${jid}"
        echo "             submits when it resumes; the dry run walks it inline instead."
        return 0
    fi
    echo
    echo "PIPELINE_ALL_JOBIDS=${ALL_JIDS}"
    exit 0
}

# ── G0: the login-node gate. Inline, not a job — everything it checks is a ────
#    property of the login node, and a job would have to be submitted before the
#    checks that decide whether submitting is a good idea.
g0 () {
    cl_banner "G0 — login-node preflight (inline; submits nothing)"
    echo "(a) THE OPEN-FINANCE SKELETON"
    cl_bootstrap_tree || echo "  [!] skeleton bootstrap reported a problem (above)."
    echo "    OPEN_FINANCE_ROOT   = $(cl_of_root)"
    echo "    SLEEP_OUT_ROOT      = ${CL_DATA_OUT}/DEMAND_PREP"
    echo "    estimation_output() = ${CL_DATA_OUT}   (via the ESTIMATION_OUTPUT symlink)"
    echo
    echo "(b) JULIA: WHAT THE CLUSTER OFFERS vs WHAT THE MANIFEST WAS RESOLVED UNDER"
    cl_julia_candidates || true
    echo "  Manifest.toml julia_version = $(cl_manifest_julia_version || echo '<none>')"
    echo "  JULIA_MODULE                = ${JULIA_MODULE}"
    if cl_load_julia >/dev/null 2>&1; then
        echo "  loaded: $(julia --version 2>/dev/null || echo '??')"
        if cl_check_julia_version; then
            echo "  -> no resolve needed."
        else
            echo "  -> A RESOLVE IS EXPECTED, and it is not an error. The manifest was resolved"
            echo "     under a different Julia; setup_julia_env.sh deletes and re-resolves it"
            echo "     ONCE, which is its documented job. Run it ALONE (concurrent Pkg.resolve()"
            echo "     on NFS corrupts Manifest.toml), wait for it, then re-run this script:"
            echo "         sbatch --partition=day --time=00:30:00 --cpus-per-task=4 --mem=16G \\"
            echo "                --export=ALL,ENV_STEP=resolve env_job.sh"
        fi
    else
        echo "  [!] '${JULIA_MODULE}' did not load. Pick a candidate above:  export JULIA_MODULE=<candidate>"
    fi
    echo
    echo "(c) cluster_preflight.sh"
    if [[ -n "${SKIP_PF}" ]]; then
        echo "  skipped (--skip-preflight)"
    elif [[ "${CL_DRYRUN}" == "1" ]]; then
        local wb=""
        [[ "${BLP_ARGS}" == *--sysimage* ]] && wb="${wb} sysimage"
        [[ "${BLP_ARGS}" == *--draws*    ]] && wb="${wb} draws"
        echo "  [dry-run] would run:  bash cluster_preflight.sh --routines '${SLEEP_ROUTINES}'${wb:+ --will-build '${wb# }'}"
    else
        # The preflight cannot know that jobs 1 and 2 of the blp phase BUILD the sysimage
        # and the draws; without this it reports them missing and refuses a run that was
        # always going to create them. Derived from --blp-args so the two cannot disagree.
        local wb=""
        [[ "${BLP_ARGS}" == *--sysimage* ]] && wb="${wb} sysimage"
        [[ "${BLP_ARGS}" == *--draws*    ]] && wb="${wb} draws"
        set +e
        bash "${CL_ROOT}/cluster_preflight.sh" --routines "${SLEEP_ROUTINES}"              ${wb:+--will-build "${wb# }"}
        local rc=$?; set -e
        if [[ ${rc} -ne 0 ]]; then
            echo "" >&2
            echo "pipeline_all.sh: refusing to submit — the preflight found blockers (above)." >&2
            echo "  If RESOLVE REQUIRED was the ONLY one, it is the expected one-time step:" >&2
            echo "  run the env_job.sh resolve printed above, then re-run this script." >&2
            echo "  --skip-preflight submits anyway." >&2
            exit 1
        fi
    fi
}

[[ "${RESUMED}" == "1" ]] || g0

# ── Phase: sleepiness ───────────────────────────────────────────────────────
G2_JID=""; G4_JID=""; SLEEP_TERM=""
if want_phase sleep && [[ "${DO_SLEEP}" == "1" ]]; then
    run_child "Phase sleep: sleep_run.sh" \
        bash "${CL_ROOT}/sleep_run.sh" --routines "${SLEEP_ROUTINES}" ${SKIP_EST} ${DRY} --skip-preflight
    [[ ${CHILD_RC} -eq 0 ]] || die_child "sleep_run.sh" "${CHILD_RC}"
    G2_JID="$(child_field SLEEP_G2_JOBID)"
    SLEEP_TERM="$(child_field SLEEP_RESULT_JOBIDS)"
    add_jid "$(child_field SLEEP_ALL_JOBIDS)"
    if [[ -z "${G2_JID}" ]]; then
        echo "ERROR: sleep_run.sh emitted no SLEEP_G2_JOBID — nothing downstream can be chained." >&2
        exit 1
    fi
    echo "  -> G2 (demand parquets verified) = ${G2_JID};  A-phase terminals = ${SLEEP_TERM:-<none>}"

    # The demand parquets are the one A-phase product worth pulling down mid-run
    # (they are the estimation sample everything else is read against). afterANY
    # G2 and --copy: non-blocking, and it never removes what the rest of the run
    # is about to read.
    if [[ -n "${G2_JID}" ]]; then
        # --copy, never --move: the rest of the pipeline reads these parquets in place
        # for days after this zip is written.
        DPWRAP="cd '${CL_ROOT}' && bash cluster_archive.sh --set demand_prep --copy --tag \"\${SLURM_JOB_ID}\""
        j=$(cl_sbatch --dependency=afterany:"${G2_JID}" \
            -J pipe_demand_prep_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
            --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
            -o "${LOGD}/pipe_demand_prep_zip_%j.out" -e "${LOGD}/pipe_demand_prep_zip_%j.err" \
            --wrap "${DPWRAP}")
        echo "  -> demand-prep archive: job ${j} (afterANY ${G2_JID}, --copy) -> data/output/demand_prep_outputs*.zip"
        add_jid "${j}"
    fi
fi

# ── Phase: logit ────────────────────────────────────────────────────────────
if want_phase logit && [[ "${DO_LOGIT}" == "1" ]]; then
    echo; echo "=== Phase logit: blp_1_logit.jl --hpc + gate G4 ==="
    lj=$(cl_sbatch ${G2_JID:+--dependency=afterok:${G2_JID}} \
        -J blp_logit --partition="${SLEEP_PARTITION:-day}" --time=02:00:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
        --export=ALL,SLEEP_ACTIVE_ESTS="${SLEEP_ROUTINES}",SPEC=12 \
        -o "${LOGD}/blp_logit_%j.out" -e "${LOGD}/blp_logit_%j.err" \
        "${CL_ROOT}/logit_job.sh")
    echo "  logit -> job ${lj}${G2_JID:+  (afterok G2 ${G2_JID})}"
    add_jid "${lj}"
    G4_JID=$(cl_sbatch --dependency=afterok:"${lj}" \
        -J sleep_gate_G4 --partition="${SLEEP_PARTITION:-day}" --time=00:10:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        --export=ALL,SLEEP_STEP=gate,SLEEP_GATE=G4,SLEEP_GATE_ROUTINES="${SLEEP_ROUTINES}",SPEC=12 \
        -o "${LOGD}/sleep_gate_G4_%j.out" -e "${LOGD}/sleep_gate_G4_%j.err" \
        "${CL_ROOT}/sleep_job.sh")
    echo "  [G4] gate -> job ${G4_JID}  (afterok ${lj})   -> data/output/.gate_G4.json"
    add_jid "${G4_JID}"
fi

# ── Hand-off A -> B. The RC ladders read the demand parquets AND warm-start from
#    the deltas, so the continuation waits on BOTH gates.
BLP_DEP="$(echo "${G2_JID}:${G4_JID}" | sed 's/^://; s/:$//')"
if [[ -n "${BLP_DEP}" ]] && ! runs_now blp && [[ "${DO_BLP}" == "1" ]]; then
    echo
    defer_to blp "${BLP_DEP}"
elif [[ -z "${BLP_DEP}" ]] && ! runs_now blp && [[ "${DO_BLP}" == "1" ]] \
     && { runs_now sleep || runs_now logit; }; then
    # Neither gate produced a job id — both phases were switched off — so there is
    # nothing to chain the RC ladders on. Say so rather than exit silently having
    # submitted nothing.
    echo
    echo "-- nothing to chain the RC phase on: both the sleep and logit phases were skipped."
    echo "   Start there instead:  bash pipeline_all.sh --from blp"
fi

# ── Phase: RC-BLP ───────────────────────────────────────────────────────────
RC_STAGE_JID=""
if want_phase blp && [[ "${DO_BLP}" == "1" ]]; then
    run_child "Phase blp: blp_run.sh" \
        bash "${CL_ROOT}/blp_run.sh" --routines "${ROUTINES}" ${BLP_ARGS} ${DRY} ${SKIP_PF}
    [[ ${CHILD_RC} -eq 0 ]] || die_child "blp_run.sh" "${CHILD_RC}"
    blp_term="$(child_field BLP_TERM_JOBIDS)"
    add_jid "${blp_term}"
    echo "  -> RC terminals = ${blp_term:-<none>}"
    if [[ -n "${blp_term}" ]]; then
        # The RC results reach the BBL/CF stack as cluster_processed/blp_E{k}_spec_12.jls,
        # unpacked from the newest blp_outputs_*.zip. cf_run.sh/bbl_run.sh do that at
        # SUBMIT time, which at t=0 is before the zip exists — so do it here, as a job,
        # once the ladders have finished and the archive has been written.
        RCWRAP="cd '${CL_ROOT}' && . ./cluster_lib.sh && cl_build_cp_dir \"\$(cl_cp_dir ${ROUTINES})\" '${CF_STAGE}' ${ROUTINES}"
        RC_STAGE_JID=$(cl_sbatch --dependency=afterany:"${blp_term}" \
            -J pipe_rc_stage --partition="${ZIP_PARTITION:-day}" --time=00:30:00 \
            --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G \
            -o "${LOGD}/pipe_rc_stage_%j.out" -e "${LOGD}/pipe_rc_stage_%j.err" \
            --wrap "${RCWRAP}")
        echo "  -> rc_stage (unpack cluster_processed/) : job ${RC_STAGE_JID}  (afterANY ${blp_term})"
        add_jid "${RC_STAGE_JID}"
    fi
fi

# ── Hand-off B -> C ─────────────────────────────────────────────────────────
if [[ -n "${RC_STAGE_JID}" ]] && ! runs_now bbl && ! runs_now cf \
   && { [[ "${DO_BBL}" == "1" ]] || [[ "${DO_CF}" == "1" ]]; }; then
    echo
    defer_to bbl "${RC_STAGE_JID}"
fi

# ── Phase: BBL costs ────────────────────────────────────────────────────────
COST_AFTEROK=""
if want_phase bbl && [[ "${DO_BBL}" == "1" ]]; then
    # --fwd-cpu deliberately. A memory note records fwd_sim as memory-bandwidth
    # bound at ~0% GPU utilisation, and a job that holds an H200 at ~0% gets
    # flagged by YCRC and costs later priority.
    run_child "Phase bbl: bbl_run.sh --fwd-cpu" \
        bash "${CL_ROOT}/bbl_run.sh" --routines "${ROUTINES}" --fwd-cpu ${DRY} ${SKIP_PF}
    [[ ${CHILD_RC} -eq 0 ]] || die_child "bbl_run.sh" "${CHILD_RC}"
    COST_AFTEROK="$(child_field BBL_SOLVE_JOBIDS)"
    add_jid "$(child_field BBL_ALL_JOBIDS)"
    echo "  -> BBL solve jobs for the CF1-net afterok: ${COST_AFTEROK:-<none>}"
fi

# ── Phase: counterfactuals (mirrors pipeline_run.sh's two-phase logic) ──────
CF_JOBIDS=""; WARMUP_JOBID=""
capture_cf () {
    local ids w
    ids="$(child_field CF_RESULT_JOBIDS)"; w="$(child_field CF_WARMUP_JOBID)"
    [[ -n "${ids}" ]] && { CF_JOBIDS="${CF_JOBIDS:+${CF_JOBIDS}:}${ids}"; add_jid "${ids}"; }
    [[ -n "${w}" && -z "${WARMUP_JOBID}" ]] && { WARMUP_JOBID="${w}"; add_jid "${w}"; }
    return 0
}

# CF4 consumes upsilon_pix + phi^noPix, produced by our own upsilon job and gated by
# G7. cf_run.sh resolves them through cl_cf4_dirs — data/input first, then
# data/output/CF_FOUNDATION — which is the same order cf_4_pix.jl uses at run time, so
# both a cluster-produced pair and a hand-uploaded one satisfy it. Nothing to detect
# and nothing to drop.
if want_phase cf && [[ "${DO_CF}" == "1" ]]; then
    # Phase 1b: CF4 with NO re-eval + the one baseline demand_eval. Needs no costs.
    steps="demand_eval cf4"
    run_child "Phase cf (1b): CF4 no re-eval (CF4_EXACT_NOPIX=0)" \
        env CF4_EXACT_NOPIX=0 bash "${CL_ROOT}/cf_run.sh" --routines "${ROUTINES}" \
            --do "${steps}" ${DRY} ${SKIP_PF}
    [[ ${CHILD_RC} -eq 0 ]] || die_child "cf_run.sh (phase 1b)" "${CHILD_RC}"
    capture_cf

    # Phase 2: CF1 gross + CF1 net + the exact CF4 re-eval. --cost-afterok is what
    # lets this go out NOW, before the BBL solve has written anything: the params
    # do not exist yet and SLURM enforces the ordering instead of the filesystem.
    steps="cf1 cf1_net cf4"
    run_child "Phase cf (2): CF1 gross + net + CF4 re-eval (CF4_EXACT_NOPIX=1)" \
        env CF4_EXACT_NOPIX=1 bash "${CL_ROOT}/cf_run.sh" --routines "${ROUTINES}" \
            --do "${steps}" \
            ${WARMUP_JOBID:+--warmup-jobid} ${WARMUP_JOBID:+${WARMUP_JOBID}} \
            ${COST_AFTEROK:+--cost-afterok} ${COST_AFTEROK:+${COST_AFTEROK}} \
            ${DRY} ${SKIP_PF}
    [[ ${CHILD_RC} -eq 0 ]] || die_child "cf_run.sh (phase 2)" "${CHILD_RC}"
    capture_cf
    echo "  -> CF result jobs = ${CF_JOBIDS:-<none>}"
fi

# ── Hand-off C -> D ─────────────────────────────────────────────────────────
if [[ -n "${CF_JOBIDS}" ]] && ! runs_now cfeq && [[ "${DO_CFEQ}" == "1" ]]; then
    echo
    defer_to cfeq "${CF_JOBIDS}"
fi

# ── Phase: equilibrium counterfactuals ──────────────────────────────────────
CFEQ_TERM=""
if want_phase cfeq && [[ "${DO_CFEQ}" == "1" ]]; then
    for mode in ${CFEQ_MODES}; do
        case "${mode}" in cf3|cf5|cf6) ;;
            *) echo "--cfeq-modes: '${mode}' is not cf3|cf5|cf6" >&2; exit 2 ;;
        esac
        if [[ "${mode}" == "cf6" && -z "${CF_MERGE:-}" ]]; then
            echo "-- cf6 skipped: it needs the conglomerate pair, e.g. CF_MERGE='firmA,firmB'"
            continue
        fi
        for k in ${ROUTINES}; do
            run_child "Phase cfeq: cf_eq_run.sh --mode ${mode} --routine ${k}" \
                bash "${CL_ROOT}/cf_eq_run.sh" --mode "${mode}" --routine "${k}" \
                    ${CF_MERGE:+--merge} ${CF_MERGE:+${CF_MERGE}} ${DRY}
            [[ ${CHILD_RC} -eq 0 ]] || die_child "cf_eq_run.sh --mode ${mode} --routine ${k}" "${CHILD_RC}"
            fj="$(child_field FINAL_JOB)"
            [[ -n "${fj}" ]] && { CFEQ_TERM="${CFEQ_TERM:+${CFEQ_TERM}:}${fj}"; add_jid "${fj}"; }
        done
    done
    echo "  -> equilibrium CF terminals = ${CFEQ_TERM:-<none>}"
fi

# ── Terminal archives: afterANY, --copy, at the very end ────────────────────
# afterany, not afterok, so partial results still get bundled if a stage
# wall-kills. --copy, always: cost_params and the sigma directories are read IN
# PLACE by anything still running, and the cf4 set sits next to uploaded inputs.
if want_phase cfeq; then
    dep="$(echo "${CFEQ_TERM}:${CF_JOBIDS}" | sed 's/^://; s/:$//; s/::/:/g')"
    if [[ -n "${dep}" ]]; then
        sets="${FINAL_ZIP_SETS}"
        for m in ${CFEQ_MODES}; do sets="${sets} ${m}"; done
        cmd="cd '${CL_ROOT}'"
        for s in ${sets}; do cmd="${cmd} && bash cluster_archive.sh --set ${s} --copy"; done
        zj=$(cl_sbatch --dependency=afterany:"${dep}" \
            -J pipe_final_zip --partition="${ZIP_PARTITION:-day}" --time=00:30:00 \
            --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
            -o "${LOGD}/pipe_final_zip_%j.out" -e "${LOGD}/pipe_final_zip_%j.err" \
            --wrap "${cmd}")
        echo; echo "-- terminal archive: job ${zj}  (afterany ${dep})  sets: ${sets}"
        add_jid "${zj}"
    else
        echo; echo "-- terminal archive skipped: no CF/CF-eq jobs were submitted in this invocation."
    fi
fi

echo
cl_banner "Submitted. Track:  squeue -u \$USER" \
          "Gate verdicts: data/output/.gate_G{1,2,3,4,7}.json — written BEFORE a gate exits." \
          "A phase that never starts means the gate before it failed; read its json first." \
          "Resume by hand at any point:  bash pipeline_all.sh --from <sleep|logit|blp|bbl|cf|cfeq>"
echo "PIPELINE_ALL_JOBIDS=${ALL_JIDS}"
