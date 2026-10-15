#!/bin/bash
# Switch-cost investigation copy of sleep_run.sh (copied 2026-10-13 17:47, source md5 bab5ac543908edda6a152e5c122de75a).
# It writes only to the switch-cost tree; the original is untouched.
# ==============================================================================
# switch_cost_sleep_run.sh — the front door of the switch-cost sleepiness run.
#
# SCOPE (user decision, 2026-10-13): E3 and E4, spec 12 only, both fit variants, with the
# two-stage (direction + link) AME bootstrap. No grid, no other spec, no E1/E2.
#
# WHAT MUST EXIST FIRST
#   1. the inputs in SC_DATA_IN (default data/input/switch_cost/; input inventory in
#      claude/switch_cost/CLUSTER_SCRIPTS.md). Run standalone, this script runs sc_preflight
#      (read-only) itself; --skip-preflight opts out. It never runs cluster_preflight.sh,
#      which writes probe files into the main step folders and starts Julia.
#   2. the code: switch_cost_*.sh, switch_cost_*.py, utils/switch_cost_*.py beside the main
#      scripts/ checkout, plus the shared utils/*.py and config/routines.toml they import.
#   3. a Python env carrying CL_PY_REQ_SLEEP. The default, dep_comp_blp, is the main
#      project's env, used as is: this copy creates and installs nothing.
#
# WHAT IT SUBMITS (all job names start with switch_cost_; logs in scripts/switch_cost_logs/)
#
#   switch_cost_sleep_est_pooled_E3 ─┐                     one job per variant x routine:
#   switch_cost_sleep_est_pooled_E4 ─┴─> [G1_pooled]       switch_cost_sleep_est_single.py
#   switch_cost_sleep_est_typeshift_E3 ─┐                    --est K --spec12, with
#   switch_cost_sleep_est_typeshift_E4 ─┴─> [G1_typeshift]   SLEEP_AME_TWOSTAGE=1
#        |
#     afterok (only with --prep)
#        v
#   switch_cost_sleep_prep_<arm>_E{3,4} ──> [G2_<arm>]     one job per arm x routine:
#                                                           switch_cost_sleep_demand_prep_link.py
#   The pooled arm waits on G1_pooled; F1, F3hi and F3lo on G1_pooled AND G1_typeshift.
#
#   A [Gx] is a GATE: a ~1-minute CPU job that opens the artefacts and checks their CONTENT,
#   writes data/output/.gate_switch_cost_<Gx>.json, and only then exits nonzero. G1 also
#   fails when the two-stage AME is missing from a fit: the estimator skips it with only a
#   printed line when it raises.
#
# GOTCHA: a job cancelled as DependencyNeverSatisfied (--kill-on-invalid-dep=yes) writes NO
# LOG AT ALL. A missing prep log means "read the gate json and the est job's .err".
#
# DISABLED (the main project's A-phase steps outside this scope): the E1/E2 routine array,
# the spec fan-out and merge, ame_gate, ame (the post-hoc AME driver), wcb_band, weakiv,
# upsilon (G7), entry. Their flags are refused, and switch_cost_sleep_job.sh refuses the steps.
#
# Usage
#   bash switch_cost_sleep_run.sh --dry-run
#   bash switch_cost_sleep_run.sh
#   bash switch_cost_sleep_run.sh --prep                          # fits, then the four arms
#   bash switch_cost_sleep_run.sh --skip-est --prep --arms "F1"    # an arm against fits on disk
#   bash switch_cost_sleep_run.sh --collect                       # zip the results for download
#
# Flags
#   --routines "3 4"        routine set (default "3 4"; only 3 and 4 exist in this copy)
#   --variants "pooled typeshift"   fit variants (default both)
#   --boot-b N              SLEEP_BOOT_B, the bootstrap draws (default 999, the Python default)
#   --ame-workers N         SLEEP_AME_BOOT_JOBS, the AME worker processes (default 64, the
#                           original AME job's value); the est job asks for EST_CPUS cpus
#   --blas-threads N        SC_BLAS_THREADS in the job (default 1, as the local fits ran)
#   --prep                  also submit the demand prep (default OFF: the local prep runs)
#   --arms "pooled F1 F3hi F3lo"   the demand arms (default all four; only with --prep)
#   --skip-est              do not estimate; with --prep, prep against the fits on disk.
#                           G1 is skipped with it.
#   --after <jid>           make every FIRST-TIER submission --dependency=afterok this job id
#   --skip-preflight        skip sc_preflight
#   --dry-run               print every sbatch line, submit nothing
#   --collect               submit nothing: zip the switch-cost outputs, gates and logs into
#                           data/output/switch_cost_download/switch_cost_results_<stamp>[_<tag>].zip
#                           (+ .sha256). COPY MODE: it never moves or deletes anything and reads
#                           no main-project folder. With --with-phis it adds the about 2 GB
#                           market_panel_phis.csv per fit; --tag NAME suffixes the zip name.
#   -h                      this header
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/switch_cost_cluster_lib.sh"
set -e

ROUTINES_SRC=default
if [[ -n "${ROUTINES+set}" ]]; then ROUTINES_SRC=env; fi
ROUTINES="${ROUTINES:-3 4}"
VARIANTS="${SC_RUN_VARIANTS:-${SC_VARIANTS}}"
ARMS="${SC_RUN_ARMS:-${SC_ARMS}}"
SPEC=12
DO_EST=1; DO_PREP=0; SKIP_PREFLIGHT=0
# AFTER_JID: the job every first-tier submission waits on. Empty means "start now".
AFTER_JID=""
BOOT_B="${SLEEP_BOOT_B:-999}"
AME_WORKERS="${SLEEP_AME_BOOT_JOBS:-64}"
BLAS_THREADS="${SC_BLAS_THREADS:-1}"

# Per-step resources. One site, so a wall-clock surprise is fixed here and not in
# several --export lines.
#
# The est job is the original spec task (EST_TIME 04:00:00) plus the original AME job
# (AME_TIME 06:00:00, 64 cpus, 256 G) in one: the two-stage bootstrap now runs inside the
# estimation pass. Its wall is the sum of the two, its cpus and memory the AME job's.
EST_TIME="${EST_TIME:-10:00:00}";   EST_CPUS="${EST_CPUS:-64}";   EST_MEM="${EST_MEM:-256G}"
# One routine x one spec x one arm per job. The original prep held four routines side by side
# in 256 G / 16 cpus for 6 h over 12 specs and the exports; this is a quarter of the memory and
# cpus, the same wall as a ceiling.
PREP_TIME="${PREP_TIME:-06:00:00}"; PREP_CPUS="${PREP_CPUS:-4}";  PREP_MEM="${PREP_MEM:-64G}"
GATE_TIME="${GATE_TIME:-00:10:00}"; GATE_CPUS="${GATE_CPUS:-2}";  GATE_MEM="${GATE_MEM:-8G}"

DO_COLLECT=0; WITH_PHIS=0; COLLECT_TAG=""

# --collect: one zip of the switch-cost results for download, then exit. Copy mode only.
sc_collect () {
    local root dl stamp zipf list n D V E F
    root="$(cd "${CL_DIR}/.." && pwd)"
    dl="${root}/data/output/switch_cost_download"
    stamp="$(date +%Y%m%d_%H%M)"
    zipf="${dl}/switch_cost_results_${stamp}${COLLECT_TAG}.zip"
    command -v zip >/dev/null || { echo "zip not found on PATH" >&2; return 2; }
    mkdir -p "${dl}"
    [[ -e "${zipf}" ]] && { echo "${zipf} exists; not overwriting" >&2; return 2; }
    list="$(mktemp)"
    (
        cd "${root}"
        for V in pooled typeshift; do
            D="data/output/switch_cost_sleep_${V}"
            [[ -d "${D}" ]] || continue
            for E in 3 4; do
                for F in estimation_results.pkl national_phi_t.csv; do
                    [[ -f "${D}/est${E}/${F}" ]] && echo "${D}/est${E}/${F}" >> "${list}"
                done
                [[ "${WITH_PHIS}" == 1 && -f "${D}/est${E}/market_panel_phis.csv" ]] && echo "${D}/est${E}/market_panel_phis.csv" >> "${list}"
            done
            [[ -d "${D}/Rout" ]] && find "${D}/Rout" -type f >> "${list}"
        done
        for D in data/output/switch_cost_demand_* data/output/switch_cost_gates scripts/switch_cost_logs; do
            [[ -d "${D}" ]] && find "${D}" -type f >> "${list}"
        done
        n=$(wc -l < "${list}")
        [[ "${n}" -gt 0 ]] || { echo "nothing to collect" >&2; exit 1; }
        # float payloads stored, text deflated (as the main archive does)
        zip -q -n .pkl:.parquet "${zipf}" -@ < "${list}"
        zip -T -q "${zipf}"
        ( cd "${dl}" && sha256sum "$(basename "${zipf}")" > "$(basename "${zipf}").sha256" )
        echo "collected ${n} files -> ${zipf} ($(du -h "${zipf}" | cut -f1))"
    )
    local rc=$?
    rm -f "${list}"
    return ${rc}
}

refuse_flag () {
    echo "switch_cost_sleep_run.sh: $1 is not an option of the switch-cost copy -- $2 (see -h)." >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --routines)       ROUTINES="$2"; ROUTINES_SRC=flag; shift ;;
        --variants)       VARIANTS="$2"; shift ;;
        --arms)           ARMS="$2"; shift ;;
        --boot-b)         BOOT_B="$2"; shift ;;
        --ame-workers)    AME_WORKERS="$2"; shift ;;
        --blas-threads)   BLAS_THREADS="$2"; shift ;;
        --prep)           DO_PREP=1 ;;
        --after)          AFTER_JID="$2"; shift ;;
        --skip-est)       DO_EST=0 ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --dry-run)        CL_DRYRUN=1 ;;
        --collect)        DO_COLLECT=1 ;;
        --with-phis)      WITH_PHIS=1 ;;
        --tag)            COLLECT_TAG="_$2"; shift ;;
        --spec-array|--no-spec-array|--spec-ids)
                          refuse_flag "$1" "this copy runs spec 12 only, one job per routine (--spec12)" ;;
        --no-ame)         refuse_flag "$1" "the two-stage AME is the reason for this run; SLEEP_AME_TWOSTAGE=1 is always submitted" ;;
        --no-wcb|--no-weakiv|--no-upsilon|--no-entry)
                          refuse_flag "$1" "that step is disabled in this copy" ;;
        -h|--help)        sed -n '2,/^# =====/p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

if [[ "${DO_COLLECT}" == 1 ]]; then
    sc_collect
    exit $?
fi

for k in ${ROUTINES}; do
    case "${k}" in 3|4) ;; *) echo "--routines: '${k}' is outside this copy's scope (3 4)" >&2; exit 2 ;; esac
done
for v in ${VARIANTS}; do
    sc_sleep_dir "${v}" >/dev/null || { echo "--variants: '${v}' (pooled typeshift)" >&2; exit 2; }
done
for a in ${ARMS}; do
    sc_demand_dir "${a}" >/dev/null || { echo "--arms: '${a}' (${SC_ARMS})" >&2; exit 2; }
done
for n in "${BOOT_B}" "${AME_WORKERS}" "${BLAS_THREADS}"; do
    [[ "${n}" =~ ^[1-9][0-9]*$ ]] || { echo "--boot-b / --ame-workers / --blas-threads: '${n}' is not a positive integer" >&2; exit 2; }
done
if [[ "${DO_EST}" == "0" && "${DO_PREP}" == "0" ]]; then
    echo "--skip-est without --prep submits nothing." >&2; exit 2
fi
# An arm reads its fits from the variant folders. With the estimators in this submission, every
# variant an arm reads must be estimated here too, or the arm would run on whatever an earlier
# run left on disk while this one rewrites the other variant.
if [[ "${DO_PREP}" == "1" && "${DO_EST}" == "1" ]]; then
    for a in ${ARMS}; do
        for v in pooled $(sc_arm_variant "${a}"); do
            [[ " ${VARIANTS} " == *" ${v} "* ]] || {
                echo "--arms ${a} reads the ${v} fit, which this submission does not estimate (--variants '${VARIANTS}')." >&2
                echo "  Add the variant, or use --skip-est to prep against the fits on disk." >&2
                exit 2; }
        done
    done
fi

LOGD="$(cl_log_dir)"
if [[ "${SKIP_PREFLIGHT}" == "0" && "${CL_DRYRUN}" != "1" ]]; then
    if ! sc_preflight $([[ "${DO_PREP}" == "1" ]] && echo ${ARMS}); then
        echo "" >&2
        echo "switch_cost_sleep_run.sh: refusing to submit — the preflight found blockers (above)." >&2
        echo "  Act on them, or re-run with --skip-preflight if you know better." >&2
        exit 1
    fi
fi

cl_banner "Switch-cost sleepiness$([[ "${CL_DRYRUN}" == "1" ]] && echo '  [DRY RUN — nothing is submitted]')" \
          "$(cl_routines_provenance "${ROUTINES}" "${ROUTINES_SRC}" "3 4")" \
          "spec=${SPEC}  variants='${VARIANTS}'  est=${DO_EST} (SLEEP_AME_TWOSTAGE=1, B=${BOOT_B}, workers=${AME_WORKERS}, blas=${BLAS_THREADS})" \
          "prep=${DO_PREP}$([[ "${DO_PREP}" == "1" ]] && echo " (arms '${ARMS}')")" \
          "sleep folders: ${CL_STEP_SLEEP_POOLED}, ${CL_STEP_SLEEP_TYPESHIFT}" \
          "inputs: ${SC_DATA_IN}   logs: ${LOGD}" \
          "${AFTER_JID:+first-tier jobs wait afterok ${AFTER_JID}}"

# NOTHING RUNS ON THE LOGIN NODE but the read-only preflight. The switch-cost skeleton is built
# by switch_cost_sleep_job.sh itself (cl_bootstrap_tree + cl_export_step_dirs at the top of
# every branch), so this script only parses flags and calls sbatch.

JOB="${CL_ROOT}/switch_cost_sleep_job.sh"
# FIRST_DEP: the sbatch fragment every first-tier submission carries. Second-tier jobs
# inherit the wait transitively through their own afterok edges.
FIRST_DEP="${AFTER_JID:+--dependency=afterok:${AFTER_JID}}"
ALL_JIDS=""
add_jid () { [[ -n "$1" ]] && ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}$1"; return 0; }

# One sbatch call site. Caller args land last, so they win (cl_sbatch's rule). Every name
# carries the switch_cost_ prefix; cl_sbatch refuses one that does not.
sub () {   # sub <jobname> <time> <cpus> <mem> <extra sbatch args...>
    local name="$1" t="$2" c="$3" m="$4"; shift 4
    cl_sbatch -J "${name}" -t "${t}" --partition="${SLEEP_PARTITION:-day}" \
        --nodes=1 --ntasks=1 --cpus-per-task="${c}" --mem="${m}" \
        -o "${LOGD}/${name}_%j.out" -e "${LOGD}/${name}_%j.err" "$@"
}

# A gate is the same payload with SLEEP_STEP=gate; day, 10 min, 2 cpus, 8 G, never a GPU.
# gate_after echoes the gate's job id on stdout, progress on stderr.
gate_after () {   # gate_after <G1|G2> <variant> <arm|""> <dep_ids|""> -> job id
    local g="$1" v="$2" a="$3" dep="$4" jid tag
    tag="${g}_${a:-${v}}"
    jid=$(sub "switch_cost_sleep_gate_${tag}" "${GATE_TIME}" "${GATE_CPUS}" "${GATE_MEM}" \
        ${dep:+--dependency=afterok:${dep}} \
        --export=ALL,SLEEP_STEP=gate,SLEEP_GATE=${g},SC_VARIANT=${v},SC_ARM=${a},SLEEP_GATE_ROUTINES="${ROUTINES}",SPEC=${SPEC},SLEEP_AME_TWOSTAGE=1 \
        "${JOB}")
    cl_say "  gate ${tag} -> ${jid}${dep:+  (afterok ${dep})}" >&2
    printf '%s\n' "${jid}"
}

# ── The estimators: one job per variant x routine, then one G1 per variant ───────
declare -A G1_JID=()
if [[ "${DO_EST}" == "1" ]]; then
    for v in ${VARIANTS}; do
        deps=""
        for k in ${ROUTINES}; do
            j=$(sub "switch_cost_sleep_est_${v}_E${k}" "${EST_TIME}" "${EST_CPUS}" "${EST_MEM}" \
                ${FIRST_DEP} \
                --export=ALL,SLEEP_STEP=est,SC_VARIANT=${v},SLEEP_EST=${k},SPEC=${SPEC},SLEEP_AME_TWOSTAGE=1,SLEEP_BOOT_B=${BOOT_B},SLEEP_AME_BOOT_JOBS=${AME_WORKERS},SC_BLAS_THREADS=${BLAS_THREADS} \
                "${JOB}")
            cl_require_jid "${j}" "switch_cost_sleep_est_${v}_E${k}"
            cl_say "  switch_cost_sleep_est_${v}_E${k} (spec 12, two-stage AME B=${BOOT_B}, ${AME_WORKERS} workers) -> ${j}${AFTER_JID:+  (afterok ${AFTER_JID})}"
            add_jid "${j}"; deps="${deps:+${deps}:}${j}"
        done
        G1_JID[${v}]=$(gate_after G1 "${v}" "" "${deps}")
        cl_require_jid "${G1_JID[${v}]}" "gate G1_${v}"
        add_jid "${G1_JID[${v}]}"
    done
else
    cl_log "-- estimators SKIPPED (--skip-est): prep runs against the fits already on disk."
    cl_log "   G1 is skipped with them: it gates THIS run's estimators, and there are none."
fi

# ── The demand prep: one job per arm x routine, then one G2 per arm ──────────────
declare -A G2_JID=()
if [[ "${DO_PREP}" == "1" ]]; then
    for a in ${ARMS}; do
        fv="$(sc_arm_variant "${a}")"
        # The arm waits on the G1 of every variant it reads; under --skip-est it is first tier.
        pdep=""
        if [[ "${DO_EST}" == "1" ]]; then
            for v in $(printf '%s\n' pooled "${fv}" | sort -u); do
                pdep="${pdep:+${pdep}:}${G1_JID[${v}]}"
            done
        else
            pdep="${AFTER_JID}"
        fi
        deps=""
        for k in ${ROUTINES}; do
            j=$(sub "switch_cost_sleep_prep_${a}_E${k}" "${PREP_TIME}" "${PREP_CPUS}" "${PREP_MEM}" \
                ${pdep:+--dependency=afterok:${pdep}} \
                --export=ALL,SLEEP_STEP=prep,SC_ARM=${a},SC_VARIANT=${fv},SLEEP_EST=${k},SPEC=${SPEC},SC_BLAS_THREADS=${BLAS_THREADS} \
                "${JOB}")
            cl_require_jid "${j}" "switch_cost_sleep_prep_${a}_E${k}"
            cl_say "  switch_cost_sleep_prep_${a}_E${k} (fits: $([[ "${fv}" == pooled ]] && echo pooled || echo 'typeshift + pooled')) -> ${j}${pdep:+  (afterok ${pdep})}"
            add_jid "${j}"; deps="${deps:+${deps}:}${j}"
        done
        G2_JID[${a}]=$(gate_after G2 "${fv}" "${a}" "${deps}")
        cl_require_jid "${G2_JID[${a}]}" "gate G2_${a}"
        add_jid "${G2_JID[${a}]}"
    done
else
    cl_log "-- demand prep not submitted (no --prep): the local prep builds the parquets."
fi

cl_say "  DISABLED in this copy (not submitted): E1/E2, the spec fan-out + merge, ame_gate, ame, wcb_band, weakiv, upsilon (G7), entry."

echo
cl_banner "Switch-cost run submitted. Track:  squeue -u \$USER -n <name>   (names start with switch_cost_)" \
          "Gate verdicts land in data/output/.gate_switch_cost_{G1_<variant>,G2_<arm>}.json, written BEFORE a gate exits." \
          "A missing log downstream of a gate means the gate failed — read its json first."

# Machine-parseable handles. Keep SWITCH_COST_RESULT_JOBIDS the LAST line.
for v in "${!G1_JID[@]}"; do echo "SWITCH_COST_G1_${v}_JOBID=${G1_JID[${v}]}"; done
for a in "${!G2_JID[@]}"; do echo "SWITCH_COST_G2_${a}_JOBID=${G2_JID[${a}]}"; done
echo "SWITCH_COST_ALL_JOBIDS=${ALL_JIDS}"
# The terminal: every gate (each covers its own producers), so an archive that waits on this
# never packages a half-written folder.
term=""
for a in "${!G2_JID[@]}"; do term="${term:+${term}:}${G2_JID[${a}]}"; done
for v in "${!G1_JID[@]}"; do term="${term:+${term}:}${G1_JID[${v}]}"; done
echo "SWITCH_COST_RESULT_JOBIDS=${term}"
