#!/bin/bash
# ==============================================================================
# sleep_run.sh — THE single front door for the sleepiness (A-phase) stage.
#
# WHAT MUST EXIST FIRST
#   1. bash cluster_preflight.sh                     (must print PREFLIGHT OK)
#   2. FIVE uploaded inputs, all tracked in cluster/upload_manifest.txt:
#        data/input/market_panel.csv
#        data/input/digital_banks_diagnostic.csv
#        data/input/bcb_banked_mca_panel.csv
#        data/input/state_centering_means.json
#        data/input/state_centering_tier0.csv
#   3. a Python env carrying CL_PY_REQ_SLEEP (numpy pandas scipy pyarrow
#      statsmodels matplotlib). The default, dep_comp_blp, already imports all six
#      — verified on the login node; there is nothing to create or install.
# WHAT TO RUN NEXT
#   bash pipeline_all.sh          (this stage is its first phase; running both
#                                  would submit the A-phase twice)
#   or, standalone:  sbatch ... logit_job.sh   then   bash blp_run.sh
#
# WHAT IT SUBMITS
#
#   est ARRAY (1 job per routine)
#        |
#     afterok
#        v
#      [G1] --afterok--> prep --afterok--> [G2] --+--> ame_gate --> ame E3 --+--> [G3]
#                                                 |               \ ame E4 --/
#                                                 +--> upsilon --> [G7]
#
#   A [Gn] is a GATE: a ~1-minute CPU job that opens the artefacts and checks
#   their CONTENT, writes data/output/.gate_Gn.json, and only then exits nonzero.
#   Everything downstream chains afterok the gate, never afterok the producer, so
#   a job that "succeeded" while writing a half-formed artefact stops the chain
#   here instead of 14 hours downstream inside Julia. Gates never take a GPU.
#
#   The two branches off G2 are deliberately PARALLEL: the AME bootstrap is the
#   long pole (~12 h) and the CF4 upsilon export is minutes, and neither reads the
#   other's output.
#
# GOTCHA, and it is the same one the BBL/CF stack has: a job cancelled as
# DependencyNeverSatisfied (--kill-on-invalid-dep=yes) writes NO LOG AT ALL. A
# missing prep log means "read .gate_G1.json and the est array's .err", not "prep
# misbehaved".
#
# Usage
#   bash sleep_run.sh --dry-run
#   bash sleep_run.sh
#   bash sleep_run.sh --skip-est            # re-run from prep against existing est output
#   bash sleep_run.sh --routines "3 4"
#
# Flags
#   --routines "1 2 3 4"  routine set (default from cluster_lib.sh: all four)
#   --spec-array          fan E3/E4 out one SLURM task PER SPEC instead of one per routine,
#                         then one merge job per routine. Cuts the estimator phase from
#                         E4-bound (~79 min) to about one spec (~7 min). OFF by default:
#                         the default routine array is the proven path.
#   --spec-ids "5-12"     the spec grid for --spec-array (default 5-12, the eight
#                         Macro/Tech x IV cells E3/E4 run; `--list-specs` prints the map)
#   --skip-est            do not re-estimate; start at prep, against the est{k}
#                         output already on disk. G1 is skipped with it — it gates
#                         THIS run's estimators, and there are none.
#   --no-ame              skip the AME off-path guard, both AME jobs and G3
#   --no-upsilon          skip the CF4 upsilon export and G7
#   --skip-preflight      skip cluster_preflight.sh
#   --dry-run             print every sbatch line, submit nothing
#   -h                    this header
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"
set -e

ROUTINES_SRC=default
if [[ -n "${ROUTINES+set}" ]]; then ROUTINES_SRC=env; fi
ROUTINES="${ROUTINES:-${CL_ROUTINES_ALL}}"
SPEC="${SPEC:-12}"
DO_EST=1; DO_AME=1; DO_UPSILON=1; SKIP_PREFLIGHT=0
# Spec-level fan-out is OPT IN. estimation_sleep_common.py grew --spec-id/--merge-specs so a
# spec task writes only _specs/spec_{S}.pkl and one merge writes the shared pickle; that was
# verified to reproduce the serial grid to 0.000e+00 across all eight specs. It stays off
# until a cluster cycle has run green on the default path.
SPEC_ARRAY=0; SPEC_IDS="${SPEC_IDS:-5-12}"
MERGE_TIME="${MERGE_TIME:-01:00:00}"; MERGE_CPUS="${MERGE_CPUS:-8}"; MERGE_MEM="${MERGE_MEM:-128G}"

# Per-step resources. One site, so a wall-clock surprise is fixed here and not in
# five --export lines.
#
# Sized against what `day` ACTUALLY offers (scontrol, 2026-08-20): the turin nodes
# carry 128 CPUs and ~2.25 TB, the emeraldrapids ones 64 CPUs and ~991 GB. The
# earlier 32-96 G figures were guesses against a much smaller node.
#
# More cores do NOT buy a wider search on the est array:
# estimation_sleep_common.py:79-82 pins JULIA_THREADS = max(2, min(8, ncpu-4)) and
# NLLS_N_STARTS is a hardcoded 4, so 16 cores and 128 cores fit the same direction
# from the same starts. 16 is what E1/E2's internal 4-way pool plus that 8-thread
# cap can actually use; the memory is what a full market panel per process needs.
EST_TIME="${EST_TIME:-04:00:00}";   EST_CPUS="${EST_CPUS:-16}";   EST_MEM="${EST_MEM:-128G}"
PREP_TIME="${PREP_TIME:-06:00:00}"; PREP_CPUS="${PREP_CPUS:-16}"; PREP_MEM="${PREP_MEM:-256G}"
AMEG_TIME="${AMEG_TIME:-00:30:00}"; AMEG_CPUS="${AMEG_CPUS:-4}";  AMEG_MEM="${AMEG_MEM:-16G}"
AME_TIME="${AME_TIME:-06:00:00}";   AME_CPUS="${AME_CPUS:-64}";   AME_MEM="${AME_MEM:-256G}"
UPS_TIME="${UPS_TIME:-01:00:00}";   UPS_CPUS="${UPS_CPUS:-4}";    UPS_MEM="${UPS_MEM:-64G}"
GATE_TIME="${GATE_TIME:-00:10:00}"; GATE_CPUS="${GATE_CPUS:-2}";  GATE_MEM="${GATE_MEM:-8G}"
# estimation_demand_1_prep.py runs the routines concurrently and each holds a full
# market panel. Its own default of 2 is a cap against "C error: out of memory" on a
# 32 GB desktop; at 256 G all four routines fit side by side.
DEMAND_PREP_JOBS="${DEMAND_PREP_JOBS:-4}"
# The draw loop IS the AME job and the draws are independent (~1.5-3 GB RSS each),
# so this scales with cores: 64 workers on 64 cpus and 256 G. default_workers()
# would otherwise cap itself at 6. SLEEP_AME_BLAS_THREADS stays UNSET on purpose —
# the driver pins it to 1, and that fixed reduction order is what makes the
# parallel run reproduce the serial one bit-for-bit.
SLEEP_AME_BOOT_JOBS="${SLEEP_AME_BOOT_JOBS:-64}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --routines)       ROUTINES="$2"; ROUTINES_SRC=flag; shift ;;
        --spec-array)     SPEC_ARRAY=1 ;;
        --spec-ids)       SPEC_IDS="$2"; shift ;;
        --skip-est)       DO_EST=0 ;;
        --no-ame)         DO_AME=0 ;;
        --no-upsilon)     DO_UPSILON=0 ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --dry-run)        CL_DRYRUN=1 ;;
        -h|--help)        sed -n '2,62p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

for k in ${ROUTINES}; do
    case "${k}" in 1|2|3|4) ;; *) echo "--routines: '${k}' is not in the lineup (1 2 3 4)" >&2; exit 2 ;; esac
done
# The AME two-stage driver serves E3/E4 only (--est choices=(3,4)); so does the
# CF4 upsilon export, whose exact phi^noPix exists only for a nonlinear link.
AME_ROUTINES="$(for k in ${ROUTINES}; do [[ "${k}" == "3" || "${k}" == "4" ]] && printf '%s ' "${k}"; done)"
AME_ROUTINES="${AME_ROUTINES% }"

LOGD="$(cl_log_dir)"
if [[ "${SKIP_PREFLIGHT}" == "0" && "${CL_DRYRUN}" != "1" ]]; then
    bash "${CL_ROOT}/cluster_preflight.sh" --routines "${ROUTINES}" || {
        echo "" >&2
        echo "sleep_run.sh: refusing to submit — the preflight found blockers (above)." >&2
        echo "  Act on them, or re-run with --skip-preflight if you know better." >&2
        exit 1; }
fi

cl_banner "Sleepiness A-phase$([[ "${CL_DRYRUN}" == "1" ]] && echo '  [DRY RUN — nothing is submitted]')" \
          "$(cl_routines_provenance "${ROUTINES}" "${ROUTINES_SRC}" "${CL_ROUTINES_ALL}")" \
          "spec=${SPEC}  est=${DO_EST}  ame=${DO_AME} (routines '${AME_ROUTINES:-none}')  upsilon=${DO_UPSILON}" \
          "vintage: SLEEP_OUT_ROOT = ${CL_DATA_OUT}/DEMAND_PREP" \
          "skeleton: OPEN_FINANCE_ROOT = $(cl_of_root)"

# ── The Open-Finance skeleton, built ONCE here as well as in every job. ──────
# Building it at submit time means a broken/absent uploaded panel is reported on
# the login node, in seconds, instead of inside the first array task.
cl_bootstrap_tree || echo "[!] skeleton bootstrap reported a problem (above) — the jobs retry it."

JOB="${CL_ROOT}/sleep_job.sh"
ALL_JIDS=""
add_jid () { [[ -n "$1" ]] && ALL_JIDS="${ALL_JIDS:+${ALL_JIDS}:}$1"; return 0; }

# One sbatch call site. Caller args land last, so they win (cl_sbatch's rule).
sub () {   # sub <jobname> <time> <cpus> <mem> <extra sbatch args...>
    local name="$1" t="$2" c="$3" m="$4"; shift 4
    cl_sbatch -J "${name}" -t "${t}" --partition="${SLEEP_PARTITION:-day}" \
        --nodes=1 --ntasks=1 --cpus-per-task="${c}" --mem="${m}" \
        -o "${LOGD}/${name}_%A_%a.out" -e "${LOGD}/${name}_%A_%a.err" "$@"
}

# A gate is the same payload with SLEEP_STEP=gate; day, 10 min, 2 cpus, 8 G, never
# a GPU. gate_after echoes the gate's job id on stdout, progress on stderr.
gate_after () {   # gate_after <G1|G2|G3|G4|G7> <dep_ids|""> -> job id
    local g="$1" dep="$2" jid
    jid=$(sub "sleep_gate_${g}" "${GATE_TIME}" "${GATE_CPUS}" "${GATE_MEM}" \
        ${dep:+--dependency=afterok:${dep}} \
        --export=ALL,SLEEP_STEP=gate,SLEEP_GATE=${g},SLEEP_GATE_ROUTINES="${ROUTINES}",SPEC=${SPEC} \
        "${JOB}")
    echo "  [${g}] gate -> job ${jid}${dep:+  (afterok ${dep})}   -> data/output/.gate_${g}.json" >&2
    printf '%s\n' "${jid}"
}

# ── The est array ────────────────────────────────────────────────────────────
# SLURM_ARRAY_TASK_ID IS the routine id. Contiguous sets print as lo-hi (so the
# default lineup reads --array=1-4), anything else as an explicit comma list.
array_spec () {
    local ks lo hi n
    ks="$(printf '%s\n' ${ROUTINES} | sort -n | tr '\n' ' ')"
    lo="${ks%% *}"; hi="$(printf '%s\n' ${ks} | tail -1)"; n="$(printf '%s\n' ${ks} | wc -w)"
    if [[ $(( hi - lo + 1 )) -eq ${n} ]]; then printf '%s-%s' "${lo}" "${hi}"
    else printf '%s' "$(echo ${ks} | tr ' ' ',')"; fi
}

est_jid=""; g1_jid=""
if [[ "${DO_EST}" == "1" && "${SPEC_ARRAY}" == "1" ]]; then
    # SPEC MODE. E1/E2 are separate scripts with no spec selector, so they stay one task each
    # in a routine array; E3/E4 each get their own array over SPEC ids with the routine pinned
    # by SLEEP_EST, followed by one merge job apiece. G1 waits on the merges, not the arrays —
    # it reads estimation_results.pkl and the phi CSV, which only exist once a merge has run.
    lin=""; link=""
    for k in ${ROUTINES}; do
        case "${k}" in 1|2) lin="${lin} ${k}" ;; 3|4) link="${link} ${k}" ;; esac
    done
    est_deps=""
    if [[ -n "${lin// /}" ]]; then
        lin_spec="$(echo ${lin} | tr ' ' ',')"
        lin_jid=$(sub "sleep_est_lin" "${EST_TIME}" "${EST_CPUS}" "${EST_MEM}" \
            --array="${lin_spec}" \
            --export=ALL,SLEEP_STEP=est,SPEC=${SPEC} "${JOB}")
        echo "  linear est array (--array=${lin_spec}) -> job ${lin_jid}  (task id = routine id)"
        add_jid "${lin_jid}"; est_deps="${est_deps}:${lin_jid}"
    fi
    for k in ${link}; do
        sp_jid=$(sub "sleep_spec_E${k}" "${EST_TIME}" "${EST_CPUS}" "${EST_MEM}" \
            --array="${SPEC_IDS}" \
            --export=ALL,SLEEP_STEP=est,SLEEP_SPEC_MODE=1,SLEEP_EST=${k},SPEC=${SPEC} "${JOB}")
        echo "  E${k} spec array (--array=${SPEC_IDS}) -> job ${sp_jid}  (task id = SPEC id)"
        add_jid "${sp_jid}"
        mg_jid=$(sub "sleep_merge_E${k}" "${MERGE_TIME}" "${MERGE_CPUS}" "${MERGE_MEM}" \
            --dependency=afterok:"${sp_jid}" \
            --export=ALL,SLEEP_STEP=merge,SLEEP_EST=${k},SPEC=${SPEC} "${JOB}")
        echo "    E${k} merge-specs -> job ${mg_jid}  (afterok ${sp_jid}; the only writer of the shared pickle)"
        add_jid "${mg_jid}"; est_deps="${est_deps}:${mg_jid}"
    done
    est_jid="${est_deps#:}"
    g1_jid=$(gate_after G1 "${est_jid}")
    add_jid "${g1_jid}"
elif [[ "${DO_EST}" == "1" ]]; then
    echo "-- estimators: one array, task id = routine id --"
    est_jid=$(sub "sleep_est" "${EST_TIME}" "${EST_CPUS}" "${EST_MEM}" \
        --array="$(array_spec)" \
        --export=ALL,SLEEP_STEP=est,SPEC=${SPEC} "${JOB}")
    echo "  est array (--array=$(array_spec)) -> job ${est_jid}  (wall=${EST_TIME}, ${EST_CPUS} cpus, ${EST_MEM})"
    add_jid "${est_jid}"
    g1_jid=$(gate_after G1 "${est_jid}")
    add_jid "${g1_jid}"
else
    echo "-- estimators SKIPPED (--skip-est): prep runs against the est{k} output already on disk."
    echo "   G1 is skipped with them: it gates THIS run's estimators, and there are none."
fi

# ── prep: exports + the universal demand prep + the spec-12 analysis + desc_3 ─
prep_dep="${g1_jid}"
prep_jid=$(sub "sleep_prep" "${PREP_TIME}" "${PREP_CPUS}" "${PREP_MEM}" \
    ${prep_dep:+--dependency=afterok:${prep_dep}} \
    --export=ALL,SLEEP_STEP=prep,SPEC=${SPEC},DEMAND_PREP_JOBS=${DEMAND_PREP_JOBS},SLEEP_ACTIVE_ESTS="${ROUTINES}" \
    "${JOB}")
echo "-- prep (run_sleep_pipeline.py --skip-sleep, DEMAND_PREP_JOBS=${DEMAND_PREP_JOBS}) -> job ${prep_jid}${prep_dep:+  (afterok ${prep_dep})}"
add_jid "${prep_jid}"
g2_jid=$(gate_after G2 "${prep_jid}")
add_jid "${g2_jid}"

# ── AME branch: the off-path guard, then one full bootstrap per routine ──────
g3_jid=""
if [[ "${DO_AME}" == "1" && -n "${AME_ROUTINES}" ]]; then
    echo "-- AME branch (afterok G2) --"
    ameg_jid=$(sub "sleep_ame_gate" "${AMEG_TIME}" "${AMEG_CPUS}" "${AMEG_MEM}" \
        --dependency=afterok:"${g2_jid}" \
        --export=ALL,SLEEP_STEP=ame_gate,SPEC=${SPEC},SLEEP_AME_ROUTINES="${AME_ROUTINES}" \
        "${JOB}")
    echo "  off-path guard (--theta-off, routines '${AME_ROUTINES}') -> job ${ameg_jid}  (afterok ${g2_jid})"
    add_jid "${ameg_jid}"
    ame_dep=""
    for k in ${AME_ROUTINES}; do
        j=$(sub "sleep_ame_E${k}" "${AME_TIME}" "${AME_CPUS}" "${AME_MEM}" \
            --dependency=afterok:"${ameg_jid}" \
            --export=ALL,SLEEP_STEP=ame,SLEEP_EST=${k},SPEC=${SPEC},SLEEP_AME_BOOT_JOBS=${SLEEP_AME_BOOT_JOBS} \
            "${JOB}")
        echo "  AME E${k} (--loss robust, full B, ${SLEEP_AME_BOOT_JOBS} workers) -> job ${j}  (afterok ${ameg_jid})"
        ame_dep="${ame_dep:+${ame_dep}:}${j}"; add_jid "${j}"
    done
    g3_jid=$(gate_after G3 "${ame_dep}")
    add_jid "${g3_jid}"
elif [[ "${DO_AME}" == "1" ]]; then
    echo "-- AME branch skipped: no routine in '${ROUTINES}' is served by the two-stage driver (E3/E4 only)."
else
    echo "-- AME branch skipped (--no-ame)."
fi

# ── CF4 upsilon branch: parallel to the AME, both afterok G2 ────────────────
g7_jid=""
if [[ "${DO_UPSILON}" == "1" && -n "${AME_ROUTINES}" ]]; then
    echo "-- CF4 upsilon branch (afterok G2, parallel to the AME) --"
    ups_jid=$(sub "sleep_upsilon" "${UPS_TIME}" "${UPS_CPUS}" "${UPS_MEM}" \
        --dependency=afterok:"${g2_jid}" \
        --export=ALL,SLEEP_STEP=upsilon,SPEC=${SPEC},SLEEP_AME_ROUTINES="${AME_ROUTINES}" \
        "${JOB}")
    echo "  upsilon_pix + phi^noPix (routines '${AME_ROUTINES}') -> job ${ups_jid}  (afterok ${g2_jid})"
    add_jid "${ups_jid}"
    g7_jid=$(gate_after G7 "${ups_jid}")
    add_jid "${g7_jid}"
elif [[ "${DO_UPSILON}" == "1" ]]; then
    echo "-- CF4 upsilon branch skipped: it serves E3/E4 only."
else
    echo "-- CF4 upsilon branch skipped (--no-upsilon)."
fi

echo
cl_banner "A-phase submitted. Track:  squeue -u \$USER" \
          "Gate verdicts land in data/output/.gate_G{1,2,3,7}.json, written BEFORE a gate exits." \
          "A missing log downstream of a gate means the gate failed — read its json first."

# Machine-parseable handles for pipeline_all.sh. Keep SLEEP_RESULT_JOBIDS the LAST
# line so `sed -n 's/^SLEEP_RESULT_JOBIDS=//p' | tail -1` captures it cleanly.
echo "SLEEP_EST_JOBID=${est_jid}"
echo "SLEEP_PREP_JOBID=${prep_jid}"
echo "SLEEP_G1_JOBID=${g1_jid}"
echo "SLEEP_G2_JOBID=${g2_jid}"
echo "SLEEP_G3_JOBID=${g3_jid}"
echo "SLEEP_G7_JOBID=${g7_jid}"
echo "SLEEP_ALL_JOBIDS=${ALL_JIDS}"
term="$(echo "${g3_jid}:${g7_jid}" | sed 's/^://; s/:$//')"
echo "SLEEP_RESULT_JOBIDS=${term}"
