#!/bin/bash
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --job-name=pipeline_all
# ==============================================================================
# pipeline_all.sh — THE single command. Sleepiness -> logit -> RC-BLP -> BBL costs
# -> the counterfactuals -> the equilibrium counterfactuals, in order, on the
# cluster, from one invocation.
#
#   sbatch pipeline_all.sh               # do it
#   sbatch pipeline_all.sh --from blp    # resume at a phase
#   sbatch pipeline_all.sh --cfeq        # add the equilibrium counterfactuals
#
# IT IS SUBMITTED, NOT RUN. Nothing in this project executes on the login node —
# not the estimators, not the preflight (G0 is its own job), and not this
# orchestrator. The #SBATCH block above is what lets `sbatch pipeline_all.sh`
# work directly; flags after the script name are passed through to it, and SLURM
# writes the job's own output to slurm-<jobid>.out in the submit directory.
#
# The work this job does is ONLY: parse flags, and sbatch the phase chains. It
# holds one core and 2 GB for the seconds that takes, then exits — the chains it
# submits outlive it, and each phase boundary submits a continuation job that
# re-enters this script with --from <next> --resumed.
#
#   bash pipeline_all.sh --dry-run     # the ONE exception: prints the graph and
#                                      # submits nothing, so it is safe anywhere
#
# WHAT MUST EXIST FIRST
#   the uploaded inputs (cluster/upload_manifest.txt). Everything else — the step
#   tree, both sysimages, the draws, every intermediate — this run builds for itself.
# WHAT TO RUN NEXT
#   Nothing. Watch it with `squeue -u $USER` and read the gate jsons:
#       for g in G0 G1 G2 G3 G4 G7; do python -m json.tool ../data/output/.gate_$g.json; done
#
# RULE 0: NOTHING RUNS ON THE LOGIN NODE. This script parses flags and calls sbatch;
# that is all it does. It loads no Julia, runs no preflight, creates no data
# directory and follows no symlink. Every check that needs the toolchain is inside
# gate G0, which is a job, and every first-tier submission waits afterok on it.
#
# ------------------------------------------------------------------------------
# THE SHAPE OF THE RUN, and why it is not one flat dependency graph
# ------------------------------------------------------------------------------
#   pipe_G0 (env_job.sh ENV_STEP=preflight; day 2c/8G 30m) -> .gate_G0.json
#     |  afterok
#   sleep_run.sh ..... est_lin array + spec arrays -> merges -> [G1] -> prep -> [G2]
#     |                                             [G2] -+-> ame_gate -> ame -> [G3]
#     |                                                   +-> upsilon E1-4 -> [G7]
#     |                                                   +-> pipe_demand_prep_zip
#   logit_job.sh (afterok G2) -> [G4]
#     |
#   =====  continuation job, afterok G2:G4  ==============================
#   blp_run.sh ....... [auto] sysimage gpu+cpu, draws -> per-routine RC ladders
#                      -> blp_zip (--copy; the results STAY in data/output/blp)
#     |
#   =====  continuation job, afterok the RC terminals  ===================
#   bbl_run.sh --fwd-cpu ....... warmup -> polfunc -> fwd_sim array -> solve
#   cf_run.sh (phase 1b) ....... demand_eval + cf4, no re-eval, needs no costs
#   cf_run.sh (phase 2) ........ cf1 + cf1_net + cf4 re-eval, --cost-afterok the solves
#   pipe_download .............. afterANY the CF jobs and the BBL solves: every set
#                                into data/output/download, --copy
#     |
#   =====  continuation job (--cfeq only), afterok the CF results  =======
#   cf_eq_run.sh ....... the firm-sharded Jacobi equilibria
#   pipe_download_cfeq . the sets those equilibria changed
#
# THE CONTINUATION JOBS ARE THE POINT, so they are worth stating plainly.
# blp_run.sh, bbl_run.sh and cf_eq_run.sh all preflight their inputs ON DISK at
# SUBMIT time — RC .jls, cost params, draws — and none of them takes an "assume it
# is coming, chain afterok instead" flag the way cf_run.sh's --cost-afterok does.
# Submitting all six phases at t=0 would therefore refuse at phase 3 with a
# perfectly correct MISSING report, because at t=0 the file really is missing.
#
# So each phase boundary submits a thirty-minute CPU job that re-invokes THIS script
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
#   --cfeq             ALSO run the equilibrium CFs (CF3/CF5/CF6). OFF by default:
#                      they are days of GPU array time and no headline result reads them
#   --no-cfeq          accepted and a NO-OP — it names the default
#   --cfeq-modes "cf3" which equilibrium CFs to run (cf3 cf5 cf6; cf6 needs CF_MERGE)
#   --blp-args "..."   extra flags forwarded verbatim to blp_run.sh (e.g. "--draws --sysimage");
#                      blp_run.sh builds what is missing without them
#   --skip-preflight   do not submit gate G0
#   --resumed          internal: set by the continuation jobs; skips G0
#   --dry-run          print every sbatch line, submit nothing
#   -h                 this header
#
# Env: DOWNLOAD_SETS names what pipe_download packages (default: every step).
#      CL_VERBOSE=1 restores each child orchestrator's full output.
# ==============================================================================
set -uo pipefail
# Where this script's siblings live. Two candidates, because `sbatch pipeline_all.sh` does NOT
# run the file you submitted: SLURM copies it into a spool directory first, so
# dirname "${BASH_SOURCE[0]}" then names the spool, where cluster_lib.sh does not exist and the
# source below would die before printing anything. SLURM_SUBMIT_DIR names the directory the
# submission was made FROM, which is the checkout when you `cd scripts && sbatch pipeline_all.sh`
# — but it is whatever your shell was in, so it is a candidate to be TESTED, not trusted.
# Both are probed for cluster_lib.sh and the first that has it wins; run with `bash` (the dry
# run) only the second exists, and it is correct.
_cl_pick_dir () {
    local d
    for d in "$@"; do
        [[ -n "${d}" && -f "${d}/cluster_lib.sh" ]] && { (cd "${d}" && pwd); return 0; }
    done
    return 1
}
CL_DIR="$(_cl_pick_dir "${SLURM_SUBMIT_DIR:-}" "$(dirname "${BASH_SOURCE[0]}")")" || {
    echo "pipeline_all.sh: cannot find cluster_lib.sh beside this script." >&2
    echo "  looked in SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-<unset>}'" >&2
    echo "         and '$(dirname "${BASH_SOURCE[0]}")'" >&2
    echo "  Submit from the scripts directory:  cd -P <project>/scripts && sbatch pipeline_all.sh" >&2
    exit 2; }
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
DO_SLEEP=1; DO_LOGIT=1; DO_BLP=1; DO_BBL=1; DO_CF=1
# The equilibrium CFs are OFF by default. They are days of GPU array time and nothing
# in the headline results reads them; --cfeq opts in, and --no-cfeq names the default.
DO_CFEQ=0
# What comes home. One set per step folder plus the gate jsons and the logs, i.e. the
# whole of data/output — the cluster is scratch and the local disk is the record, so a
# partial download is the one failure mode there is no recovering from once the
# allocation is cleaned. cluster_archive.sh splits anything over ~3 GB.
DOWNLOAD_SETS="${DOWNLOAD_SETS:-sleep demand_prep logit blp bbl counterfactuals gates logs}"
# What group D re-packages: only the sets an equilibrium CF run can change.
DOWNLOAD_SETS_CFEQ="${DOWNLOAD_SETS_CFEQ:-counterfactuals gates logs}"
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
        --cfeq)           DO_CFEQ=1 ;;
        --no-cfeq)        DO_CFEQ=0 ;;          # names the default; kept so old command lines still parse
        --cfeq-modes)     CFEQ_MODES="$2"; shift ;;
        --blp-args)       BLP_ARGS="$2"; shift ;;
        --skip-preflight) SKIP_PF="--skip-preflight" ;;
        --resumed)        RESUMED=1 ;;
        --dry-run)        CL_DRYRUN=1; DRY="--dry-run" ;;
        -h|--help)        sed -n '2,88p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
#                      the cost params on disk, so SLURM enforces the ordering
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
#
# The capture is UNCONDITIONAL — child_field parses it either way — but the ECHO of
# it is not, and there are three cases:
#   verbose / dry run / the child FAILED  -> print all of it. A dry run exists to show
#       the graph, so hiding the graph would make --dry-run useless; a failed child's
#       MISSING report is the only thing worth reading at that moment.
#   otherwise -> the whole transcript goes to the orchestrator log, and the terminal
#       keeps one line per submission. The filter is ' -> <jobid>', which is the shape
#       every submission line in every child has, so a run reads as its own job graph
#       rather than as six concatenated banners.
CHILD_OUT=""; CHILD_RC=0
run_child () {   # run_child <label> <command...>
    local label="$1"; shift
    cl_log ""; cl_log "=== ${label} ==="
    set +e; CHILD_OUT="$("$@")"; CHILD_RC=$?; set -e
    if [[ "${CL_VERBOSE}" == "1" || "${CL_DRYRUN}" == "1" || ${CHILD_RC} -ne 0 ]]; then
        printf '%s\n' "${CHILD_OUT}"
    else
        _cl_to_log "${CHILD_OUT}"
        printf '%s\n' "${CHILD_OUT}" | grep -E ' -> [0-9]+' || true
    fi
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
    # --cfeq is opt-IN, so the continuation carries it forward only when it was asked
    # for. The other three are opt-OUT and carry their negation instead.
    [[ "${DO_CFEQ}" == "1" ]] && wrap="${wrap} --cfeq"
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

# ── G0: a JOB, and the first tier of the graph ───────────────────────────────
# It builds the tree, loads the toolchain, LOADS both sysimages and imports the
# Python stack — none of which belongs on a shared login host, and all of which is
# more reliable on a compute node with the same CPU target the real jobs get. So it
# is submitted like everything else, and the A-phase hangs off it afterok:
# --kill-on-invalid-dep then turns a blocking verdict into "nothing downstream ever
# starts", which is exactly the semantics an inline refusal had.
#
# It echoes the job id on stdout and its narration on stderr, so
# `G0_JID="$(submit_g0)"` captures an id and not a paragraph.
submit_g0 () {
    local wb="" jid
    # The preflight cannot know that the BLP phase BUILDS the sysimage and the draws
    # when they are absent; without --will-build it reports them missing and refuses a
    # run that was always going to create them. Derived from --blp-args, the only place
    # a caller can force those builds, so the two cannot disagree. blp_run.sh's own
    # auto-build is silent here on purpose: it decides from disk, at its own submit
    # time, and by then G0 has already run.
    [[ "${BLP_ARGS}" == *--sysimage* ]] && wb="${wb} sysimage"
    [[ "${BLP_ARGS}" == *--draws*    ]] && wb="${wb} draws"
    jid=$(cl_sbatch -J pipe_G0 --partition="${SLEEP_PARTITION:-day}" --time=00:30:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        --export=ALL,ENV_STEP=preflight,PF_ROUTINES="${SLEEP_ROUTINES}",PF_WILL_BUILD="${wb# }" \
        -o "${LOGD}/pipe_G0_%j.out" -e "${LOGD}/pipe_G0_%j.err" \
        "${CL_ROOT}/env_job.sh")
    cl_say "  pipe_G0 -> ${jid}  (ENV_STEP=preflight; -> data/output/.gate_G0.json)" >&2
    printf '%s\n' "${jid}"
}

G0_JID=""
if [[ "${RESUMED}" != "1" && -z "${SKIP_PF}" ]]; then
    G0_JID="$(submit_g0)"
    add_jid "${G0_JID}"
elif [[ -n "${SKIP_PF}" ]]; then
    cl_say "  G0 skipped (--skip-preflight) — nothing verifies the toolchain or the inputs."
fi

# ── Phase: sleepiness ───────────────────────────────────────────────────────
G2_JID=""; G4_JID=""; SLEEP_TERM=""
if want_phase sleep && [[ "${DO_SLEEP}" == "1" ]]; then
    # --after ${G0_JID}: every first-tier A-phase submission waits on the gate.
    # --skip-preflight: G0 IS the preflight, and running a second one here would put
    # it back on the login node, which is the thing this design removes.
    run_child "Phase sleep: sleep_run.sh" \
        bash "${CL_ROOT}/sleep_run.sh" --routines "${SLEEP_ROUTINES}" ${SKIP_EST} ${DRY} \
             --skip-preflight ${G0_JID:+--after} ${G0_JID:+${G0_JID}}
    [[ ${CHILD_RC} -eq 0 ]] || die_child "sleep_run.sh" "${CHILD_RC}"
    G2_JID="$(child_field SLEEP_G2_JOBID)"
    SLEEP_TERM="$(child_field SLEEP_RESULT_JOBIDS)"
    add_jid "$(child_field SLEEP_ALL_JOBIDS)"
    if [[ -z "${G2_JID}" ]]; then
        echo "ERROR: sleep_run.sh emitted no SLEEP_G2_JOBID — nothing downstream can be chained." >&2
        exit 1
    fi
    cl_log "  -> G2 (demand parquets verified) = ${G2_JID};  A-phase terminals = ${SLEEP_TERM:-<none>}"

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
        cl_say "  pipe_demand_prep_zip -> ${j}  (afterANY ${G2_JID}, --copy)"
        add_jid "${j}"
    fi
fi

# ── Phase: logit ────────────────────────────────────────────────────────────
if want_phase logit && [[ "${DO_LOGIT}" == "1" ]]; then
    # Normally afterok G2 — the logit reads the demand parquets that gate verifies.
    # On `--from logit` there is no G2 in this graph, so it falls back to G0: the
    # parquets are already on disk from an earlier run, and what still has to hold
    # is that the toolchain checks passed.
    LOGIT_DEP="${G2_JID:-${G0_JID}}"
    lj=$(cl_sbatch ${LOGIT_DEP:+--dependency=afterok:${LOGIT_DEP}} \
        -J blp_logit --partition="${SLEEP_PARTITION:-day}" --time=02:00:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
        --export=ALL,SLEEP_ACTIVE_ESTS="${SLEEP_ROUTINES}",SPEC=12 \
        -o "${LOGD}/blp_logit_%j.out" -e "${LOGD}/blp_logit_%j.err" \
        "${CL_ROOT}/logit_job.sh")
    cl_say "  blp_logit -> ${lj}${LOGIT_DEP:+  (afterok ${LOGIT_DEP})}"
    add_jid "${lj}"
    G4_JID=$(cl_sbatch --dependency=afterok:"${lj}" \
        -J sleep_gate_G4 --partition="${SLEEP_PARTITION:-day}" --time=00:10:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        --export=ALL,SLEEP_STEP=gate,SLEEP_GATE=G4,SLEEP_GATE_ROUTINES="${SLEEP_ROUTINES}",SPEC=12 \
        -o "${LOGD}/sleep_gate_G4_%j.out" -e "${LOGD}/sleep_gate_G4_%j.err" \
        "${CL_ROOT}/sleep_job.sh")
    cl_say "  gate G4 -> ${G4_JID}  (afterok ${lj})"
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
    cl_say "  nothing to chain the RC phase on: both the sleep and logit phases were skipped."
    cl_say "  Start there instead:  bash pipeline_all.sh --from blp"
fi

# ── Phase: RC-BLP ───────────────────────────────────────────────────────────
BLP_TERM=""
if want_phase blp && [[ "${DO_BLP}" == "1" ]]; then
    # --skip-preflight always: G0 already ran it, as a job. blp_run.sh still decides
    # the sysimage/draws builds for itself, from disk.
    run_child "Phase blp: blp_run.sh" \
        bash "${CL_ROOT}/blp_run.sh" --routines "${ROUTINES}" ${BLP_ARGS} ${DRY} --skip-preflight
    [[ ${CHILD_RC} -eq 0 ]] || die_child "blp_run.sh" "${CHILD_RC}"
    BLP_TERM="$(child_field BLP_TERM_JOBIDS)"
    add_jid "${BLP_TERM}"
    cl_log "  -> RC terminals = ${BLP_TERM:-<none>}"
fi

# ── Hand-off B -> C ─────────────────────────────────────────────────────────
# Straight onto the RC terminals. The ladders persist their results in
# data/output/blp and nothing stages, unpacks or renames them afterwards, so the
# moment the last rung reports COMPLETED the file bbl_run.sh and cf_run.sh preflight
# is the file that is there.
if [[ -n "${BLP_TERM}" ]] && ! runs_now bbl && ! runs_now cf \
   && { [[ "${DO_BBL}" == "1" ]] || [[ "${DO_CF}" == "1" ]]; }; then
    echo
    defer_to bbl "${BLP_TERM}"
fi

# ── Phase: BBL costs ────────────────────────────────────────────────────────
COST_AFTEROK=""
if want_phase bbl && [[ "${DO_BBL}" == "1" ]]; then
    # --fwd-cpu deliberately. A memory note records fwd_sim as memory-bandwidth
    # bound at ~0% GPU utilisation, and a job that holds an H200 at ~0% gets
    # flagged by YCRC and costs later priority.
    #
    # PY_PREFLIGHT=0 with --skip-preflight: both of bbl_run.sh's checks import the
    # Python stack, and G0 has already done exactly that on a compute node. Leaving
    # them on here would put an interpreter back on the login host for no new
    # information.
    run_child "Phase bbl: bbl_run.sh --fwd-cpu" \
        env PY_PREFLIGHT=0 bash "${CL_ROOT}/bbl_run.sh" --routines "${ROUTINES}" --fwd-cpu \
            ${DRY} --skip-preflight
    [[ ${CHILD_RC} -eq 0 ]] || die_child "bbl_run.sh" "${CHILD_RC}"
    COST_AFTEROK="$(child_field BBL_SOLVE_JOBIDS)"
    add_jid "$(child_field BBL_ALL_JOBIDS)"
    cl_log "  -> BBL solve jobs for the CF1-net afterok: ${COST_AFTEROK:-<none>}"
fi

# ── Phase: counterfactuals, in two sub-phases ───────────────────────────────
CF_JOBIDS=""; WARMUP_JOBID=""
capture_cf () {
    local ids w
    ids="$(child_field CF_RESULT_JOBIDS)"; w="$(child_field CF_WARMUP_JOBID)"
    [[ -n "${ids}" ]] && { CF_JOBIDS="${CF_JOBIDS:+${CF_JOBIDS}:}${ids}"; add_jid "${ids}"; }
    [[ -n "${w}" && -z "${WARMUP_JOBID}" ]] && { WARMUP_JOBID="${w}"; add_jid "${w}"; }
    return 0
}

# CF4 consumes upsilon_pix + phi^noPix, produced by this run's own upsilon job into
# data/output/counterfactuals and gated by G7. cf_run.sh resolves them through
# cl_cf4_dirs, which names that one directory — the same one cf_4_pix.jl reads at run
# time — so the pair the preflight sees is the pair the job opens.
if want_phase cf && [[ "${DO_CF}" == "1" ]]; then
    # Phase 1b: CF4 with NO re-eval + the one baseline demand_eval. Needs no costs.
    steps="demand_eval cf4"
    run_child "Phase cf (1b): CF4 no re-eval (CF4_EXACT_NOPIX=0)" \
        env CF4_EXACT_NOPIX=0 bash "${CL_ROOT}/cf_run.sh" --routines "${ROUTINES}" \
            --do "${steps}" ${DRY} --skip-preflight
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
            ${DRY} --skip-preflight
    [[ ${CHILD_RC} -eq 0 ]] || die_child "cf_run.sh (phase 2)" "${CHILD_RC}"
    capture_cf
    cl_log "  -> CF result jobs = ${CF_JOBIDS:-<none>}"
fi

# ── THE DOWNLOAD: one job, every set, afterANY ──────────────────────────────
# It belongs to the CF phase, not to the equilibrium phase, because group C is where
# the run ends by default. Hanging it off --cfeq would mean a default run finished
# with nothing packaged and the whole point of a complete download lost.
#
# afterANY the CF jobs AND the BBL solves: a wall-killed CF still packages, and the
# bbl set is only complete once the solves have written cost_params. The sha256SUMS
# the archiver writes prove TRANSPORT, not completeness — which is why the gates and
# logs sets travel alongside, so the local side can tell a short zip from a short run.
#
# Chained with && rather than ;: one failing set then fails the job, visibly, instead
# of leaving a silently missing family to be discovered after the allocation is gone.
if want_phase cf && [[ "${DO_CF}" == "1" ]]; then
    dl_dep="$(echo "${CF_JOBIDS}:${COST_AFTEROK}" | sed 's/^://; s/:$//; s/::/:/g')"
    if [[ -n "${dl_dep}" ]]; then
        cmd="cd '${CL_ROOT}'"
        for s in ${DOWNLOAD_SETS}; do
            cmd="${cmd} && bash cluster_archive.sh --set ${s} --copy --tag \"\${SLURM_JOB_ID}\""
        done
        dj=$(cl_sbatch --dependency=afterany:"${dl_dep}" \
            -J pipe_download --partition="${ZIP_PARTITION:-day}" --time=02:00:00 \
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G \
            -o "${LOGD}/pipe_download_%j.out" -e "${LOGD}/pipe_download_%j.err" \
            --wrap "${cmd}")
        cl_say "  pipe_download -> ${dj}  (afterANY ${dl_dep})"
        cl_say "     sets: ${DOWNLOAD_SETS}  -> data/output/download"
        add_jid "${dj}"
    else
        cl_say "  pipe_download skipped: no CF or BBL job was submitted in this invocation."
    fi
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
            cl_say "  cf6 skipped: it needs the conglomerate pair, e.g. CF_MERGE='firmA,firmB'"
            continue
        fi
        for k in ${ROUTINES}; do
            # --no-zip: every mode and every routine writes into the SAME step folder,
            # and pipe_download_cfeq below packages that folder once, afterany all of
            # them. Left to itself each cf_eq_run.sh would submit its own terminal
            # archive of the whole counterfactuals set, so a "cf3 cf5" x "3 4" group
            # would re-zip the identical tree four extra times, each one a --copy pass
            # over every earlier equilibrium's output. Run by hand, cf_eq_run.sh keeps
            # its own zip: nothing else is watching that chain to package it.
            run_child "Phase cfeq: cf_eq_run.sh --mode ${mode} --routine ${k}" \
                bash "${CL_ROOT}/cf_eq_run.sh" --mode "${mode}" --routine "${k}" \
                    ${CF_MERGE:+--merge} ${CF_MERGE:+${CF_MERGE}} --no-zip ${DRY}
            [[ ${CHILD_RC} -eq 0 ]] || die_child "cf_eq_run.sh --mode ${mode} --routine ${k}" "${CHILD_RC}"
            fj="$(child_field FINAL_JOB)"
            [[ -n "${fj}" ]] && { CFEQ_TERM="${CFEQ_TERM:+${CFEQ_TERM}:}${fj}"; add_jid "${fj}"; }
        done
    done
    cl_log "  -> equilibrium CF terminals = ${CFEQ_TERM:-<none>}"

    # A SECOND download, for the sets these equilibria change. The first one has long
    # since run — group D can start days later — so the counterfactuals, gates and logs
    # zips are re-cut with this job's tag rather than overwriting the earlier ones.
    if [[ -n "${CFEQ_TERM}" ]]; then
        cmd="cd '${CL_ROOT}'"
        for s in ${DOWNLOAD_SETS_CFEQ}; do
            cmd="${cmd} && bash cluster_archive.sh --set ${s} --copy --tag \"\${SLURM_JOB_ID}\""
        done
        dj=$(cl_sbatch --dependency=afterany:"${CFEQ_TERM}" \
            -J pipe_download_cfeq --partition="${ZIP_PARTITION:-day}" --time=02:00:00 \
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G \
            -o "${LOGD}/pipe_download_cfeq_%j.out" -e "${LOGD}/pipe_download_cfeq_%j.err" \
            --wrap "${cmd}")
        cl_say "  pipe_download_cfeq -> ${dj}  (afterANY ${CFEQ_TERM}; sets: ${DOWNLOAD_SETS_CFEQ})"
        add_jid "${dj}"
    fi
fi

echo
cl_say "track:    squeue -u \$USER          (details: $(cl_orch_log))"
cl_say "gates:    data/output/.gate_G{0,1,2,3,4,7}.json — each written BEFORE its gate exits;"
cl_say "          a phase that never starts means the gate before it failed, so read that json first."
cl_say "download: data/output/download/ once pipe_download reports COMPLETED, then sha256sum -c sha256SUMS there."
echo "PIPELINE_ALL_JOBIDS=${ALL_JIDS}"
