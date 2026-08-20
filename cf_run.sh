#!/bin/bash
# ==============================================================================
# cf_run.sh — THE single front door for the counterfactual pipeline.
#
# WHAT MUST EXIST FIRST
#   1. bash cluster_preflight.sh                     (must print PREFLIGHT OK)
#   2. the RC results (auto-built from the newest blp_outputs_*.zip)
#   3. data/input/forward_rf_qoq.csv
#   4. for cf1_net/cf3/cf5/cf6: the BBL cost params from bbl_run.sh
#   5. for cf4: {upsilon_pix,phi_nopix}_E{k}_spec_12.*, either produced by the
#      sleepiness phase into data/output/CF_FOUNDATION (bash sleep_run.sh, gate G7)
#      or built locally and uploaded to data/input, which wins when both exist
# WHAT TO RUN NEXT
#   bash cf_eq_run.sh --mode cf3|cf5|cf6            (the long equilibrium CFs)
#   bash cluster_archive.sh --set foundation --copy (and cf1, cf4)
#
# THIS SCRIPT IS NEW. submit_cf_all.sh is still present and still works; it is the
# fallback and is retired only after one successful cluster cycle. Nothing in it
# has been modified.
#
# WHAT IT SUBMITS
#   cf_warmup --afterok--> demand_eval / cf1 / cf4        (no costs needed)
#             --afterok--> cf1_net / cf3 / cf5 / cf6      (cost-consuming)
#   ONE warmup barrier for the whole run. The cost-consuming steps take the
#   warmup AND, when the BBL solve is chained into the same orchestrated run,
#   the solve jobs — MERGED into ONE afterok list, because sbatch keeps only the
#   LAST --dependency flag and passing two would silently drop the warmup edge.
#
#   NO AUTO-ZIP. cost_params feeds CF3/CF5/CF6 and CF3's sig_N feeds CF5 long
#   after this script exits, so archiving mid-pipeline would strand those inputs.
#   Archive deliberately at the end with cluster_archive.sh.
#
# Usage
#   bash cf_run.sh --dry-run
#   bash cf_run.sh --do "demand_eval cf1 cf1_net cf4"
#   bash cf_run.sh --do "demand_eval cf1" --routines "3"
#
# Flags
#   --routines "3 4"     routine set (default from cluster_lib.sh)
#   --do "<steps>"       selector list; any of
#                          demand_eval cf1 cf4 cf1_net cf3 cf5 cf6
#                        default: "demand_eval cf1 cf1_net"
#   --no-warmup          skip the barrier
#   --warmup-jobid ID    reuse an existing warmup barrier instead of submitting one
#   --cost-afterok IDS   colon-joined BBL solve job ids; skips the on-disk cost
#                        preflight and chains afterok instead
#   --skip-preflight     skip cluster_preflight.sh
#   --dry-run            print, submit nothing
#   -h                   this header
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"
set -e

ROUTINES_SRC=default
if [[ -n "${ROUTINES+set}" ]]; then ROUTINES_SRC=env; fi
ROUTINES="${ROUTINES:-${CL_ROUTINES_CF}}"
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}"
DO_STEPS="${DO_STEPS:-demand_eval cf1 cf1_net}"
DO_WARMUP="${DO_WARMUP:-1}"
WARMUP_JOBID="${WARMUP_JOBID:-}"
CF_COST_AFTEROK="${CF_COST_AFTEROK:-}"
AUTO_PROCESS="${AUTO_PROCESS:-1}"
SHARD_TIME="${SHARD_TIME:-08:00:00}"; SOLVE_TIME="${SOLVE_TIME:-01:00:00}"
EQ_TIME="${EQ_TIME:-12:00:00}"
SKIP_PREFLIGHT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --routines)       ROUTINES="$2"; ROUTINES_SRC=flag; shift ;;
        --do)             DO_STEPS="$2"; shift ;;
        --no-warmup)      DO_WARMUP=0 ;;
        --warmup-jobid)   WARMUP_JOBID="$2"; shift ;;
        --cost-afterok)   CF_COST_AFTEROK="$2"; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --dry-run)        CL_DRYRUN=1 ;;
        -h|--help)        sed -n '2,50p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

want () { case " ${DO_STEPS} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
for s in ${DO_STEPS}; do
    case "${s}" in demand_eval|cf1|cf4|cf1_net|cf3|cf5|cf6) ;;
        *) echo "unknown step '${s}' in --do (valid: demand_eval cf1 cf4 cf1_net cf3 cf5 cf6)" >&2; exit 2 ;;
    esac
done

LOGD="$(cl_log_dir)"
if [[ "${SKIP_PREFLIGHT}" == "0" && "${CL_DRYRUN}" != "1" ]]; then
    bash "${CL_ROOT}/cluster_preflight.sh" --routines "${ROUTINES}" || {
        echo "cf_run.sh: refusing to submit — the preflight found blockers (above)." >&2; exit 1; }
fi

cl_banner "Counterfactual pipeline$([[ "${CL_DRYRUN}" == "1" ]] && echo '  [DRY RUN — nothing is submitted]')" \
          "$(cl_routines_provenance "${ROUTINES}" "${ROUTINES_SRC}" "${CL_ROUTINES_CF}")" \
          "steps: ${DO_STEPS}" \
          "stage=${CF_STAGE} R=${R} seed=${SEED} beta=${BETA} horizon=${HORIZON}"

# ── Step 1: build cluster_processed/ from the newest RC zip if a routine is missing
CP_DIR="$(cl_cp_dir ${ROUTINES})"
need_process=0
for k in ${ROUTINES}; do [[ -f "${CP_DIR}/blp_E${k}_spec_12.jls" ]] || need_process=1; done
if [[ "${need_process}" == "1" && "${AUTO_PROCESS}" == "1" ]]; then
    cl_build_cp_dir "${CP_DIR}" "${CF_STAGE}" ${ROUTINES}
fi

# ── Step 2: preflight ────────────────────────────────────────────────────────
miss=0
cl_need_draws "${R}" "${SEED}" || miss=1
cl_need_rf_curve || miss=1
for k in ${ROUTINES}; do cl_need_rc_jls "${CP_DIR}" "${k}" || miss=1; done

# The equilibrium CFs consume the BBL cost params, produced by the SEPARATE BBL
# stage. Preflight the JSON on disk rather than chaining across orchestrators —
# unless --cost-afterok says the solve jobs are queued in the same run, in which
# case the params do not exist YET and SLURM enforces the ordering instead.
need_costs=0
for s in cf1_net cf3 cf5 cf6; do want "${s}" && need_costs=1; done
if [[ -n "${CF_COST_AFTEROK}" ]]; then
    need_costs=0
    echo "cost params chained via afterok:${CF_COST_AFTEROK} — skipping the on-disk cost_params preflight"
fi
if [[ "${need_costs}" == "1" ]]; then
    for k in ${ROUTINES}; do cl_need_costs "${k}" "${CF_STAGE}" || miss=1; done
fi

# CF4 needs no BBL costs, but DOES need the exact link-aware phi^noPix + Upsilon_pix
# from cf_4_upsilon_export.py. That export is no longer local-only: the sleepiness
# estimators run on the cluster, so the sleep pickle it reads is there too, and
# sleep_job.sh SLEEP_STEP=upsilon produces the pair into data/output/CF_FOUNDATION
# (gate G7). An uploaded pair in data/input still wins — cl_cf4_dirs searches it
# first — so a hand-staged export from a local build keeps working unchanged.
#
# cl_cf4_dirs is the SAME order cf_4_pix.jl's cf4_search_dirs uses at run time.
cf4_note=""
if want cf4; then
    for k in ${ROUTINES}; do
        cl_need_cf4_file "upsilon_pix_E${k}_spec_12.json"  "CF4 Upsilon_pix E${k}" || miss=1
        cl_need_cf4_file "phi_nopix_E${k}_spec_12.parquet" "CF4 phi^noPix E${k}"   || miss=1
    done
    cf4_note=" + CF4 phi_nopix"
fi
[[ "${miss}" == "0" ]] || { echo "Stage the missing input(s) (cluster/RUNBOOK.md step 0), then re-run."; exit 1; }
# Test the VALUE, not emptiness: need_costs="0" is a NON-EMPTY string, so a
# `${need_costs:+...}` expansion would fire even when the cost preflight was
# skipped and claim "+ BBL cost params" on a run where nothing was verified.
costs_note=""; [[ "${need_costs}" == "1" ]] && costs_note=" + BBL cost params"
echo "Preflight OK: R=${R} draws + forward r^f curve + RC results${costs_note}${cf4_note} for routines: ${ROUTINES}"

# ── Step 3: submit ───────────────────────────────────────────────────────────
sub () {  # sub <jobname> <time> <extra sbatch args...>
    local name="$1" tlim="$2"; shift 2
    # --kill-on-invalid-dep=yes (from cl_sbatch): if an afterok dependency can
    # NEVER be satisfied (an upstream job failed or was cancelled, including one
    # array task), cancel this job instead of leaving it pending forever. The
    # consequence, in words: a job cancelled as DependencyNeverSatisfied writes
    # NO LOG FILE AT ALL. A missing cf1_net log therefore means "read the
    # corresponding bbl_solve .err", not "cf1_net misbehaved".
    cl_sbatch -J "${name}" -t "${tlim}" \
        ${PARTITION:+--partition="${PARTITION}"} ${GPUS:+--gpus="${GPUS}"} ${MEM:+--mem="${MEM}"} \
        -o "${LOGD}/${name}_%A_%a.out" -e "${LOGD}/${name}_%A_%a.err" "$@"
}

# ── The ONE warmup barrier ───────────────────────────────────────────────────
wj="${WARMUP_JOBID}"
if [[ -z "${wj}" && "${DO_WARMUP}" == "1" ]]; then
    wj=$(sub "cf_warmup" "${SOLVE_TIME}" \
        --export=ALL,CF_ROUTINE=${ROUTINES%% *},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED},CF_STEP=warmup \
        "${CL_ROOT}/cf_job.sh")
    echo "-- pre-warm barrier -> job ${wj} (every CF waits on it, afterok) --"
elif [[ -n "${wj}" ]]; then
    echo "-- pre-warm barrier: REUSING job ${wj} (--warmup-jobid) --"
fi
warm_dep="${wj:+--dependency=afterok:${wj}}"

# Cost-consuming steps wait on the warmup AND the BBL solves. MERGE into ONE
# afterok list: sbatch keeps only the LAST --dependency flag, so a step cannot
# take warm_dep and a second dependency flag separately — passing two silently
# drops the warmup edge and the whole point of the barrier with it.
cost_ok_ids="${wj}"
[[ -n "${CF_COST_AFTEROK}" ]] && cost_ok_ids="${cost_ok_ids:+${cost_ok_ids}:}${CF_COST_AFTEROK}"
cost_dep="${cost_ok_ids:+--dependency=afterok:${cost_ok_ids}}"

cf1_extra="--beta ${BETA} --horizon ${HORIZON}"
eq_extra="--beta ${BETA} --horizon ${HORIZON}"
found_dep=""; cf1_dep=""; cf4_dep=""; cf3_dep=""; cf5_dep=""; cf6_dep=""
for k in ${ROUTINES}; do
    base_export="CF_ROUTINE=${k},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED}"
    echo "-- E${k} ${CF_STAGE} | R=${R} --"
    # demand_eval runs PER ROUTINE: shares_elas_E{k} is the in-sample share
    # reproduction for THAT estimator, so one routine's copy validates only that one.
    if want demand_eval; then
        j=$(sub "cf_demaneval_E${k}" "${SOLVE_TIME}" ${warm_dep} \
            --export=ALL,${base_export},CF_STEP=demand_eval "${CL_ROOT}/cf_job.sh")
        echo "  demand_eval  -> job ${j}"; found_dep="${found_dep}:${j}"
    fi
    if want cf1; then
        j=$(sub "cf_cf1_E${k}" "${SHARD_TIME}" ${warm_dep} \
            --export=ALL,${base_export},CF_STEP=cf1,CF_EXTRA="${cf1_extra}" "${CL_ROOT}/cf_job.sh")
        echo "  cf1          -> job ${j}"; cf1_dep="${cf1_dep}:${j}"
    fi
    if want cf4; then
        j=$(sub "cf_cf4_E${k}" "${SHARD_TIME}" ${warm_dep} \
            --export=ALL,${base_export},CF_STEP=cf4 "${CL_ROOT}/cf_job.sh")
        echo "  cf4          -> job ${j}"; cf4_dep="${cf4_dep}:${j}"
    fi
    if want cf1_net; then
        j=$(sub "cf_cf1net_E${k}" "${SOLVE_TIME}" ${cost_dep} \
            --export=ALL,${base_export},CF_STEP=cf1_net,CF_EXTRA="${eq_extra}" "${CL_ROOT}/cf_job.sh")
        echo "  cf1_net      -> job ${j}"; cf1_dep="${cf1_dep}:${j}"   # net -> same cf1 archive
    fi
    for step in cf3 cf5 cf6; do
        want "${step}" || continue
        j=$(sub "cf_${step}_E${k}" "${EQ_TIME}" ${cost_dep} \
            --export=ALL,${base_export},CF_STEP=${step},CF_EXTRA="${eq_extra} ${CF_EQ_EXTRA:-}" \
            "${CL_ROOT}/cf_job.sh")
        echo "  ${step}          -> job ${j}"
        case "${step}" in cf3) cf3_dep="${cf3_dep}:${j}";; cf5) cf5_dep="${cf5_dep}:${j}";; cf6) cf6_dep="${cf6_dep}:${j}";; esac
    done
done

echo "Submitted CFs for routines: ${ROUTINES}. Watch with: squeue -u \$USER"
echo "When the WHOLE chain (incl. CF3/CF5/CF6) is done, archive with:"
echo "  bash cluster_archive.sh --all --copy      # then --move once nothing downstream runs"
# Machine-parseable handles for pipeline_run.sh. CF_RESULT_JOBIDS stays the LAST
# line so `sed -n 's/^CF_RESULT_JOBIDS=//p' | tail -1` captures it cleanly.
echo "CF_WARMUP_JOBID=${wj}"
cf_all_ids="$(echo "${found_dep}${cf1_dep}${cf4_dep}${cf3_dep}${cf5_dep}${cf6_dep}" | sed 's/^://')"
echo "CF_RESULT_JOBIDS=${cf_all_ids}"
