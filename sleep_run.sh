#!/bin/bash
# ==============================================================================
# sleep_run.sh — THE single front door for the sleepiness (A-phase) stage.
#
# WHAT MUST EXIST FIRST
#   1. gate G0 green — sbatch --export=ALL,ENV_STEP=preflight env_job.sh. Run standalone,
#      this script runs cluster_preflight.sh itself instead (--skip-preflight opts out);
#      under pipeline_all.sh, G0 has already run and its id arrives through --after.
#   2. FIVE uploaded inputs, all tracked in cluster/upload_manifest.txt:
#        data/input/market_panel.parquet   (the panel travels as parquet, 39 MB not 724 MB;
#                                           utils.load_panel_cached reads it as the source
#                                           because no market_panel.csv exists here)
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
#   sleep_est_lin --array=1,2          (E1/E2: one task per routine, whole grid)
#   sleep_spec_E3 --array=5-12 --> sleep_merge_E3   (E3/E4: one task per SPEC)
#   sleep_spec_E4 --array=5-12 --> sleep_merge_E4
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
#   --after <jid>         make every FIRST-TIER submission (the est/spec arrays, or prep
#                         under --skip-est) --dependency=afterok this job id. That is how
#                         pipeline_all.sh hangs the whole A-phase off gate G0 without this
#                         script having to know what G0 is.
#   --no-spec-array       fan E3/E4 out one task PER ROUTINE instead of one per SPEC, and
#                         drop the merge jobs. The spec fan-out is the default: it cuts the
#                         estimator phase from E4-bound (~79 min) to about one spec (~7 min)
#                         and reproduces the serial grid to 0.000e+00 across all eight specs.
#   --spec-array          accepted and a NO-OP — it names the default. Kept so an existing
#                         command line keeps working.
#   --spec-ids "5-12"     the spec grid for the fan-out (default 5-12, the eight
#                         Macro/Tech x IV cells E3/E4 run; `--list-specs` prints the map)
#   --skip-est            do not re-estimate; start at prep, against the est{k}
#                         output already on disk. G1 is skipped with it — it gates
#                         THIS run's estimators, and there are none.
#   --no-ame              skip the AME off-path guard, both AME jobs and G3
#   --no-wcb              skip the linear percentile bands (E1/E2); their columns then
#                         fall back to standard errors and the note says so
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
DO_EST=1; DO_AME=1; DO_UPSILON=1; DO_WCB=1; SKIP_PREFLIGHT=0
# AFTER_JID: the job every first-tier submission waits on. Empty means "start now".
AFTER_JID=""
# Spec-level fan-out is the DEFAULT. sleep_est_single.py's --spec-id/--merge-specs
# split makes a spec task's entire write set _specs/spec_{S}.pkl, with one merge job as the
# only writer of the shared pickle — so concurrent spec tasks have no shared write to race
# on, rather than a guarded one, and the result reproduces the serial grid to 0.000e+00
# across all eight specs. --no-spec-array falls back to one task per routine.
SPEC_ARRAY=1; SPEC_IDS="${SPEC_IDS:-5-12}"
MERGE_TIME="${MERGE_TIME:-01:00:00}"; MERGE_CPUS="${MERGE_CPUS:-8}"; MERGE_MEM="${MERGE_MEM:-128G}"

# Per-step resources. One site, so a wall-clock surprise is fixed here and not in
# five --export lines.
#
# Sized against what `day` ACTUALLY offers (scontrol, 2026-08-20): the turin nodes
# carry 128 CPUs and ~2.25 TB, the emeraldrapids ones 64 CPUs and ~991 GB. The
# earlier 32-96 G figures were guesses against a much smaller node.
#
# More cores do NOT buy a wider search on the est array: NLLS_N_STARTS is a
# hardcoded 4 in sleep_est_single.py, so 16 cores and 128 cores fit the same
# direction from the same starts. 16 is what E1/E2's internal 4-way pool can
# actually use; the memory is what a full market panel per process needs.
EST_TIME="${EST_TIME:-04:00:00}";   EST_CPUS="${EST_CPUS:-16}";   EST_MEM="${EST_MEM:-128G}"
PREP_TIME="${PREP_TIME:-06:00:00}"; PREP_CPUS="${PREP_CPUS:-16}"; PREP_MEM="${PREP_MEM:-256G}"
AMEG_TIME="${AMEG_TIME:-00:30:00}"; AMEG_CPUS="${AMEG_CPUS:-4}";  AMEG_MEM="${AMEG_MEM:-16G}"
AME_TIME="${AME_TIME:-06:00:00}";   AME_CPUS="${AME_CPUS:-64}";   AME_MEM="${AME_MEM:-256G}"
UPS_TIME="${UPS_TIME:-01:00:00}";   UPS_CPUS="${UPS_CPUS:-4}";    UPS_MEM="${UPS_MEM:-64G}"
# The linear bands rebuild one demeaned OLS per routine and take B draws off its influence
# functions -- no refit, no parallel draw loop. The memory is a full market panel per routine,
# the same reason the est array asks for 128 G; the wall clock is the panel read.
WCB_TIME="${WCB_TIME:-01:30:00}";   WCB_CPUS="${WCB_CPUS:-8}";    WCB_MEM="${WCB_MEM:-128G}"
GATE_TIME="${GATE_TIME:-00:10:00}"; GATE_CPUS="${GATE_CPUS:-2}";  GATE_MEM="${GATE_MEM:-8G}"
# sleep_demand_prep.py runs the routines concurrently and each holds a full
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
        --after)          AFTER_JID="$2"; shift ;;
        --spec-array)     SPEC_ARRAY=1 ;;          # names the default; kept so old command lines still parse
        --no-spec-array)  SPEC_ARRAY=0 ;;
        --spec-ids)       SPEC_IDS="$2"; shift ;;
        --skip-est)       DO_EST=0 ;;
        --no-ame)         DO_AME=0 ;;
        --no-wcb)         DO_WCB=0 ;;
        --no-upsilon)     DO_UPSILON=0 ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --dry-run)        CL_DRYRUN=1 ;;
        -h|--help)        sed -n '2,77p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

for k in ${ROUTINES}; do
    case "${k}" in 1|2|3|4) ;; *) echo "--routines: '${k}' is not in the lineup (1 2 3 4)" >&2; exit 2 ;; esac
done
# The AME two-stage driver serves E3/E4 only, by construction: sleep_ame_twostage.py
# takes --est choices=(3,4), because the two-stage AME is defined off a single-index link.
AME_ROUTINES="$(for k in ${ROUTINES}; do [[ "${k}" == "3" || "${k}" == "4" ]] && printf '%s ' "${k}"; done)"
AME_ROUTINES="${AME_ROUTINES% }"
# The mirror image: sleep_wcb_band.py serves E1/E2 only, because the linear estimators are the
# ones whose second stage stored a standard error and no percentile band. Between the two, all
# four columns of the comparison table can report an interval.
WCB_ROUTINES="$(for k in ${ROUTINES}; do [[ "${k}" == "1" || "${k}" == "2" ]] && printf '%s ' "${k}"; done)"
WCB_ROUTINES="${WCB_ROUTINES% }"
# The CF4 upsilon export carries NO such restriction, and it must not: cf4_pix.jl needs an
# upsilon_pix/phi^noPix pair for every routine that reaches the CF phase, and
# sleep_upsilon_export.py's identity branch writes phi_nopix with exact_nopix=True for the
# linear routines exactly as the single-index branch does for E3/E4. Gate G7 checks all four.
UPSILON_ROUTINES="${ROUTINES}"

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
          "spec=${SPEC}  est=${DO_EST} (spec-array=${SPEC_ARRAY})  ame=${DO_AME} (routines '${AME_ROUTINES:-none}')" \
          "upsilon=${DO_UPSILON} (routines '${UPSILON_ROUTINES}')" \
          "SLEEP_OUT_ROOT = ${CL_STEP_SLEEP}   DEMAND_PREP_DIR = ${CL_STEP_DEMAND}" \
          "${AFTER_JID:+first-tier jobs wait afterok ${AFTER_JID}}"

# NOTHING RUNS ON THE LOGIN NODE. The Open-Finance skeleton is built by sleep_job.sh
# itself (cl_bootstrap_tree + cl_export_step_dirs at the top of every branch), so this
# script only ever parses flags and calls sbatch. A broken or absent uploaded panel is
# reported by gate G0, which is a job, and pipeline_all.sh passes its id in via --after.

JOB="${CL_ROOT}/sleep_job.sh"
# FIRST_DEP: the sbatch fragment every first-tier submission carries. Second-tier jobs
# (gates, prep behind G1, the AME and upsilon branches behind G2) inherit the wait
# transitively through their own afterok edges, so it belongs here and nowhere else.
FIRST_DEP="${AFTER_JID:+--dependency=afterok:${AFTER_JID}}"
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
    cl_say "  gate ${g} -> ${jid}${dep:+  (afterok ${dep})}" >&2
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
            --array="${lin_spec}" ${FIRST_DEP} \
            --export=ALL,SLEEP_STEP=est,SPEC=${SPEC} "${JOB}")
        cl_say "  sleep_est_lin --array=${lin_spec} -> ${lin_jid}${AFTER_JID:+  (afterok ${AFTER_JID})}"
        add_jid "${lin_jid}"; est_deps="${est_deps}:${lin_jid}"
    fi
    for k in ${link}; do
        sp_jid=$(sub "sleep_spec_E${k}" "${EST_TIME}" "${EST_CPUS}" "${EST_MEM}" \
            --array="${SPEC_IDS}" ${FIRST_DEP} \
            --export=ALL,SLEEP_STEP=est,SLEEP_SPEC_MODE=1,SLEEP_EST=${k},SPEC=${SPEC} "${JOB}")
        cl_say "  sleep_spec_E${k} --array=${SPEC_IDS} -> ${sp_jid}${AFTER_JID:+  (afterok ${AFTER_JID})}"
        add_jid "${sp_jid}"
        mg_jid=$(sub "sleep_merge_E${k}" "${MERGE_TIME}" "${MERGE_CPUS}" "${MERGE_MEM}" \
            --dependency=afterok:"${sp_jid}" \
            --export=ALL,SLEEP_STEP=merge,SLEEP_EST=${k},SPEC=${SPEC} "${JOB}")
        cl_say "  sleep_merge_E${k} -> ${mg_jid}  (afterok ${sp_jid}; the only writer of the shared pickle)"
        add_jid "${mg_jid}"; est_deps="${est_deps}:${mg_jid}"
    done
    est_jid="${est_deps#:}"
    g1_jid=$(gate_after G1 "${est_jid}")
    add_jid "${g1_jid}"
elif [[ "${DO_EST}" == "1" ]]; then
    est_jid=$(sub "sleep_est" "${EST_TIME}" "${EST_CPUS}" "${EST_MEM}" \
        --array="$(array_spec)" ${FIRST_DEP} \
        --export=ALL,SLEEP_STEP=est,SPEC=${SPEC} "${JOB}")
    cl_say "  sleep_est --array=$(array_spec) -> ${est_jid}  (task id = routine id)${AFTER_JID:+  (afterok ${AFTER_JID})}"
    add_jid "${est_jid}"
    g1_jid=$(gate_after G1 "${est_jid}")
    add_jid "${g1_jid}"
else
    cl_log "-- estimators SKIPPED (--skip-est): prep runs against the est{k} output already on disk."
    cl_log "   G1 is skipped with them: it gates THIS run's estimators, and there are none."
fi

# ── prep: exports + the universal demand prep + the spec-12 analysis + desc_3 ─
# With the estimators in the graph, prep waits on G1 and inherits the --after wait
# through it. Under --skip-est there is no G1, so prep IS the first tier and takes
# --after directly — otherwise a resumed A-phase would start before its gate.
prep_dep="${g1_jid:-${AFTER_JID}}"
prep_jid=$(sub "sleep_prep" "${PREP_TIME}" "${PREP_CPUS}" "${PREP_MEM}" \
    ${prep_dep:+--dependency=afterok:${prep_dep}} \
    --export=ALL,SLEEP_STEP=prep,SPEC=${SPEC},DEMAND_PREP_JOBS=${DEMAND_PREP_JOBS},SLEEP_ACTIVE_ESTS="${ROUTINES}" \
    "${JOB}")
cl_say "  sleep_prep -> ${prep_jid}${prep_dep:+  (afterok ${prep_dep})}"
add_jid "${prep_jid}"
g2_jid=$(gate_after G2 "${prep_jid}")
add_jid "${g2_jid}"

# ── Linear percentile bands (E1/E2): beside the AME branch, not inside it ────
# They share only the requirement that the estimators exist (G2). Kept independent of DO_AME
# because a run restricted to the linear routines still needs its intervals, and folded into
# G3's dependency below when the AME branch is also running, so the phase terminal does not
# clear while half the second-stage table's inference is still missing.
wcb_jid=""
if [[ "${DO_WCB}" == "1" && -n "${WCB_ROUTINES}" ]]; then
    wcb_jid=$(sub "sleep_wcb_band" "${WCB_TIME}" "${WCB_CPUS}" "${WCB_MEM}" \
        --dependency=afterok:"${g2_jid}" \
        --export=ALL,SLEEP_STEP=wcb_band,SPEC=${SPEC},SLEEP_WCB_ROUTINES="${WCB_ROUTINES}" \
        "${JOB}")
    cl_say "  sleep_wcb_band (linear percentile bands, routines '${WCB_ROUTINES}') -> ${wcb_jid}  (afterok ${g2_jid})"
    add_jid "${wcb_jid}"
elif [[ "${DO_WCB}" == "1" ]]; then
    cl_log "-- WCB band branch skipped: no linear routine (E1/E2) in '${ROUTINES}'."
else
    cl_log "-- WCB band branch skipped (--no-wcb)."
fi

# ── AME branch: the off-path guard, then one full bootstrap per routine ──────
g3_jid=""
if [[ "${DO_AME}" == "1" && -n "${AME_ROUTINES}" ]]; then
    ameg_jid=$(sub "sleep_ame_gate" "${AMEG_TIME}" "${AMEG_CPUS}" "${AMEG_MEM}" \
        --dependency=afterok:"${g2_jid}" \
        --export=ALL,SLEEP_STEP=ame_gate,SPEC=${SPEC},SLEEP_AME_ROUTINES="${AME_ROUTINES}" \
        "${JOB}")
    cl_say "  sleep_ame_gate (--theta-off, routines '${AME_ROUTINES}') -> ${ameg_jid}  (afterok ${g2_jid})"
    add_jid "${ameg_jid}"
    ame_dep=""
    for k in ${AME_ROUTINES}; do
        j=$(sub "sleep_ame_E${k}" "${AME_TIME}" "${AME_CPUS}" "${AME_MEM}" \
            --dependency=afterok:"${ameg_jid}" \
            --export=ALL,SLEEP_STEP=ame,SLEEP_EST=${k},SPEC=${SPEC},SLEEP_AME_BOOT_JOBS=${SLEEP_AME_BOOT_JOBS} \
            "${JOB}")
        cl_say "  sleep_ame_E${k} (full B, ${SLEEP_AME_BOOT_JOBS} workers) -> ${j}  (afterok ${ameg_jid})"
        ame_dep="${ame_dep:+${ame_dep}:}${j}"; add_jid "${j}"
    done
    # Append only when the band job exists: an empty wcb_jid here would leave a trailing
    # colon and sbatch rejects `afterok:1:2:` outright.
    if [[ -n "${wcb_jid}" ]]; then
        ame_dep="${ame_dep:+${ame_dep}:}${wcb_jid}"
    fi
    g3_jid=$(gate_after G3 "${ame_dep}")
    add_jid "${g3_jid}"
elif [[ "${DO_AME}" == "1" ]]; then
    cl_say "  AME branch skipped: no routine in '${ROUTINES}' is served by the two-stage driver (E3/E4 only)."
else
    cl_log "-- AME branch skipped (--no-ame)."
fi

# ── CF4 upsilon branch: parallel to the AME, both afterok G2 ────────────────
g7_jid=""
if [[ "${DO_UPSILON}" == "1" ]]; then
    ups_jid=$(sub "sleep_upsilon" "${UPS_TIME}" "${UPS_CPUS}" "${UPS_MEM}" \
        --dependency=afterok:"${g2_jid}" \
        --export=ALL,SLEEP_STEP=upsilon,SPEC=${SPEC},SLEEP_UPSILON_ROUTINES="${UPSILON_ROUTINES}" \
        "${JOB}")
    cl_say "  sleep_upsilon (upsilon_pix + phi^noPix, routines '${UPSILON_ROUTINES}') -> ${ups_jid}  (afterok ${g2_jid})"
    add_jid "${ups_jid}"
    g7_jid=$(gate_after G7 "${ups_jid}")
    add_jid "${g7_jid}"
else
    cl_log "-- CF4 upsilon branch skipped (--no-upsilon)."
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
