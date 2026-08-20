#!/bin/bash
# ==============================================================================
# cf_eq_run.sh — THE equilibrium-counterfactual driver: CF3, CF5 and CF6 from ONE
# implementation of the firm-sharded Jacobi.
#
# WHY SHARDED. The best-response re-simulates the full national panel per
# evaluation (D-firm shares are national), so a single-process solve is
# intractable at R=2000 and a market subset is not a valid headline. Each Jacobi
# SWEEP is an array of shard jobs — shard s best-responds firms
# {j : j mod N_FIRM_SHARDS == s} against the FROZEN previous-sweep sigma —
# followed by a merge into the next sigma. Sweeps chain afterok. This uses the
# exact full-panel psi_under; it only parallelises the firms.
#
# WHAT MUST EXIST FIRST
#   the BBL cost params:  data/output/cost/cost_params_E{k}_spec_12_{stage}.json
#   i.e. run  bash bbl_run.sh  first.
# WHAT TO RUN NEXT
#   bash cluster_archive.sh --set cf3|cf5|cf6 --copy   (--move once nothing else needs them)
#
# THIS SCRIPT IS NEW. submit_cf3_jacobi.sh / submit_cf5_all.sh /
# submit_cf5_passthrough.sh / submit_cf6_merger.sh are still present and still
# work; they are the fallback and are retired only after one successful cluster
# cycle. Nothing in them has been modified.
#
# WHAT IT SUBMITS
#   --mode cf3   warm(GPU) -> init(CPU) -> N_SWEEPS x [ shard array(GPU) -> merge(CPU) ]
#                                                                 -> [zip cf3]
#   --mode cf5   the same chain TWICE (base, Selic-shocked) -> cf5_compare afterok BOTH -> [zip cf5]
#   --mode cf6   the same chain TWICE (base, merged)        -> cf6_compare afterok BOTH -> [zip cf6]
#
# SIGMA LAYOUT (one convention, nested):
#   cf3   data/output/cf/cf3_jacobi_E{k}_{stage}/
#   cf5   data/output/cf/cf5_E{k}_{stage}/{base,shock}/
#   cf6   data/output/cf/cf6_E{k}_{stage}/{base,merged}/
#   This is the layout the cf5/cf6 archive patterns (output/cf/cf5_E*, cf6_E*)
#   actually match. If FLAT cf5_base_E*/cf5_shock_E* directories from an earlier
#   run are on the cluster, archive them with the old zip_cf_outputs.sh BEFORE
#   running this driver — cluster_preflight.sh reports them; it does not move them.
#
# Usage
#   bash cf_eq_run.sh --mode cf3 --routine 3 --dry-run
#   bash cf_eq_run.sh --mode cf3 --routine 3
#   bash cf_eq_run.sh --mode cf5 --routine 3 [--selic-shock 0.01] [--base-sigma <path>]
#   bash cf_eq_run.sh --mode cf6 --routine 3 --merge firmA,firmB [--base-sigma <path>]
#
# Flags
#   --mode cf3|cf5|cf6   required
#   --routine K          default from cluster_lib.sh (first of CL_ROUTINES_CF)
#   --sweeps N           Jacobi depth  (default 6)
#   --shards N           Jacobi width  (default 40)
#   --selic-shock X      cf5 only, annual (default 0.01 = 100bp)
#   --merge "A,B"        cf6 only, required: the conglomerate pair
#   --base-sigma PATH    reuse an already-solved base sigma instead of re-solving
#   --no-zip             suppress the terminal archive job
#   --dry-run            print, submit nothing
#   -h                   this header
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"
set -e

MODE=""
CF_ROUTINE="${CF_ROUTINE:-${CL_ROUTINES_CF%% *}}"
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
N_FIRM_SHARDS="${N_FIRM_SHARDS:-40}"; N_SWEEPS="${N_SWEEPS:-6}"
BR_GRID="${BR_GRID:-7}"; BR_WINDOW="${BR_WINDOW:-0.02}"
BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}"
SELIC_SHOCK="${SELIC_SHOCK:-0.01}"
MERGE="${MERGE:-}"
BASE_SIGMA="${BASE_SIGMA:-}"
SHARD_TIME="${SHARD_TIME:-06:00:00}"; MERGE_TIME="${MERGE_TIME:-00:30:00}"
WARM_TIME="${WARM_TIME:-01:00:00}"; COMPARE_TIME="${COMPARE_TIME:-02:00:00}"
GPU_PARTITION="${GPU_PARTITION:-gpu_h200}"; GPUS="${GPUS:-h200:1}"; CPU_PARTITION="${CPU_PARTITION:-day}"
DO_WARMUP="${DO_WARMUP:-1}"
DO_ZIP="${DO_ZIP:-1}"
# ZIP_AFTER keeps its single-dash ${ZIP_AFTER-...} semantics: UNSET means "default
# to the mode's own archive", while an explicit EMPTY value turns the terminal
# archive off mid-run. The two must stay distinguishable, hence the flag.
_ZIP_AFTER_WAS_SET=0; [[ -n "${ZIP_AFTER+set}" ]] && _ZIP_AFTER_WAS_SET=1
ZIP_AFTER="${ZIP_AFTER-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)        MODE="$2"; shift ;;
        --routine)     CF_ROUTINE="$2"; shift ;;
        --stage)       CF_STAGE="$2"; shift ;;
        --sweeps)      N_SWEEPS="$2"; shift ;;
        --shards)      N_FIRM_SHARDS="$2"; shift ;;
        --selic-shock) SELIC_SHOCK="$2"; shift ;;
        --merge)       MERGE="$2"; shift ;;
        --base-sigma)  BASE_SIGMA="$2"; shift ;;
        --no-warmup)   DO_WARMUP=0 ;;
        --no-zip)      DO_ZIP=0 ;;
        --dry-run)     CL_DRYRUN=1 ;;
        -h|--help)     sed -n '2,62p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done
case "${MODE}" in cf3|cf5|cf6) ;; *) echo "--mode must be cf3|cf5|cf6 (got '${MODE}')" >&2; exit 2 ;; esac
[[ "${MODE}" == "cf6" && -z "${MERGE}" ]] && { echo "--mode cf6 requires --merge \"firmA,firmB\"" >&2; exit 2; }
[[ "${_ZIP_AFTER_WAS_SET}" == "1" ]] || ZIP_AFTER="${MODE}"

LOGD="$(cl_log_dir)"
ROOT="${CL_DATA_OUT}/cf"

# Fail fast on the shared prerequisites, once and up front, so we never submit a
# half-chain: a cf5 run is TWO Jacobi chains plus a compare, and discovering a
# missing cost file after the first array has queued wastes GPU hours.
cl_need_costs "${CF_ROUTINE}" "${CF_STAGE}" || exit 1
cl_validate_routines "${CF_ROUTINE}" || {
    echo "  -> run the BLP stage first, or drop its blp_outputs_*.zip in ${CL_DATA_OUT}" >&2; exit 1; }

cl_banner "Equilibrium CF: ${MODE}$([[ "${CL_DRYRUN}" == "1" ]] && echo '  [DRY RUN — nothing is submitted]')" \
          "E${CF_ROUTINE} ${CF_STAGE} | R=${R} | ${N_FIRM_SHARDS} shards x ${N_SWEEPS} sweeps per equilibrium" \
          "shards -> ${GPU_PARTITION} (--gpus=${GPUS}) | init/merge/compare -> ${CPU_PARTITION}" \
          "$([[ "${MODE}" == "cf5" ]] && echo "Selic shock +${SELIC_SHOCK}")$([[ "${MODE}" == "cf6" ]] && echo "merge ${MERGE}")"

base_export="CF_ROUTINE=${CF_ROUTINE},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED},N_FIRM_SHARDS=${N_FIRM_SHARDS}"
eq_flags="--beta ${BETA} --horizon ${HORIZON} --br-grid ${BR_GRID} --br-window ${BR_WINDOW}"
CPU_SB=(--partition="${CPU_PARTITION}")
GPU_SB=(--partition="${GPU_PARTITION}" --gpus="${GPUS}")

sub () {  # sub <jobname> <time> <extra sbatch args...>
    local n="$1" t="$2"; shift 2
    cl_sbatch -J "${n}" -t "${t}" ${MEM:+--mem="${MEM}"} \
        -o "${LOGD}/${n}_%A_%a.out" -e "${LOGD}/${n}_%A_%a.err" "$@"
}

# ── ONE Jacobi implementation. Sets RJ_JOB (terminal merge job id) and RJ_SIGMA.
# The empty-capture guard is kept: an empty job id would silently produce a
# dependency on nothing and the compare would run against a half-solved sigma.
RJ_JOB=""; RJ_SIGMA=""; EQ_WARM_JID=""
run_jacobi () {   # run_jacobi <sigma_dir> <extra_cf_flags> <tag>
    local sdir="$1" extra="$2" tag="$3" wdep="" j prev arr mrg s
    # A dry run must leave nothing behind, so the sigma dir is created only for real.
    [[ "${CL_DRYRUN}" == "1" ]] || mkdir -p "${sdir}"
    if [[ "${DO_WARMUP}" == "1" ]]; then
        # GPU pre-warm: load the CUDA share path ONCE on a GPU node so the
        # N_FIRM_SHARDS shard tasks do not stampede the shared-NFS depot lock
        # loading CUDA cold — the same 0%-CPU stall the CPU chain hit. ONE
        # barrier serves both legs of a cf5/cf6 run: they share the depot, so a
        # second warmup would hold a second H200 to redo identical work.
        if [[ -z "${EQ_WARM_JID}" ]]; then
            j=$(sub "eqwarm_E${CF_ROUTINE}" "${WARM_TIME}" "${GPU_SB[@]}" \
                --export=ALL,CF_ROUTINE=${CF_ROUTINE},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED},CF_STEP=warmup,CF_GPU=1 \
                "${CL_ROOT}/cf_job.sh")
            EQ_WARM_JID="${j}"
            echo "  GPU warmup -> ${j} (${GPU_PARTITION}; init/shards wait on it)"
        else
            echo "  GPU warmup: reusing barrier ${EQ_WARM_JID}"
        fi
        wdep="--dependency=afterok:${EQ_WARM_JID}"
    fi
    j=$(sub "eqinit_${tag}_E${CF_ROUTINE}" "${MERGE_TIME}" "${CPU_SB[@]}" ${wdep} \
        --export=ALL,${base_export},SIGMA_DIR=${sdir},CF_STEP=cf3_init,CF_EXTRA="${eq_flags} ${extra}" \
        "${CL_ROOT}/cf_job.sh")
    echo "  init sigma0 -> ${j} (${CPU_PARTITION}) -> ${sdir}"
    prev="${j}"
    for s in $(seq 1 "${N_SWEEPS}"); do
        arr=$(sub "eqsw${s}_${tag}_E${CF_ROUTINE}" "${SHARD_TIME}" \
            --array=0-$((N_FIRM_SHARDS-1))${SHARD_THROTTLE:+%${SHARD_THROTTLE}} \
            "${GPU_SB[@]}" --dependency=afterok:"${prev}" \
            --export=ALL,${base_export},SIGMA_DIR=${sdir},SWEEP=${s},CF_STEP=cf3_shard,CF_EXTRA="${eq_flags} ${extra}" \
            "${CL_ROOT}/cf_job.sh")
        mrg=$(sub "eqmg${s}_${tag}_E${CF_ROUTINE}" "${MERGE_TIME}" "${CPU_SB[@]}" --dependency=afterok:"${arr}" \
            --export=ALL,${base_export},SIGMA_DIR=${sdir},SWEEP=${s},CF_STEP=cf3_merge,CF_EXTRA="${eq_flags} ${extra}" \
            "${CL_ROOT}/cf_job.sh")
        echo "  sweep ${s}: shard array ${arr} (GPU) -> merge ${mrg} (CPU)"
        prev="${mrg}"
    done
    RJ_JOB="${prev}"; RJ_SIGMA="${sdir}/sig_${N_SWEEPS}.parquet"
    [[ -n "${RJ_JOB}" && -n "${RJ_SIGMA}" ]] || {
        echo "ERROR: the Jacobi produced no terminal job / sigma path (${sdir})" >&2; exit 1; }
}

submit_zip () {   # submit_zip <which> <afterany dep> <mode copy|move>
    [[ "${DO_ZIP}" == "1" && -n "${1}" ]] || return 0
    local zj
    zj=$(sub "cf_zip_$1_E${CF_ROUTINE}" 00:30:00 "${CPU_SB[@]}" --mem=4G \
        --dependency=afterany:"$2" \
        --export=ALL,CF_STEP=zip,CF_WHICH=$1,CF_STAGE=${CF_STAGE},CF_ZIP_MODE=$3 \
        "${CL_ROOT}/cf_job.sh")
    echo "  zip $1 (${3}) -> ${zj} (afterany:$2)"
}

case "${MODE}" in

cf3)
    SDIR="${SIGMA_DIR:-${ROOT}/cf3_jacobi_E${CF_ROUTINE}_${CF_STAGE}}"
    echo "== CF3 equilibrium =="
    run_jacobi "${SDIR}" "${EXTRA_CF:-}" "cf3"
    FINAL_JOB="${RJ_JOB}"; FINAL_SIGMA="${RJ_SIGMA}"
    # COPY, not move: CF3's sig_N is the CF5 base and is read long after this exits.
    submit_zip "${ZIP_AFTER}" "${FINAL_JOB}" copy
    ;;

cf5)
    SDIR="${ROOT}/cf5_E${CF_ROUTINE}_${CF_STAGE}"
    dep=""
    if [[ -n "${BASE_SIGMA}" ]]; then
        base_sig="${BASE_SIGMA}"; echo "== CF5 base: REUSING ${base_sig} =="
    else
        echo "== CF5 base equilibrium -> ${SDIR}/base =="
        run_jacobi "${SDIR}/base" "--selic-shock 0" "cf5b"
        base_sig="${RJ_SIGMA}"; dep="${dep}:${RJ_JOB}"
    fi
    echo "== CF5 shocked equilibrium (Selic +${SELIC_SHOCK}) -> ${SDIR}/shock =="
    run_jacobi "${SDIR}/shock" "--selic-shock ${SELIC_SHOCK}" "cf5s"
    scn_sig="${RJ_SIGMA}"; dep="${dep}:${RJ_JOB}"
    dep="${dep#:}"
    echo "== CF5 compare (afterok:${dep}) =="
    FINAL_JOB=$(sub "cf5cmp_E${CF_ROUTINE}" "${COMPARE_TIME}" "${CPU_SB[@]}" \
        --dependency=afterok:"${dep}" \
        --export=ALL,CF_ROUTINE=${CF_ROUTINE},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED},CF_STEP=cf5_compare,SIGMA_BASE="${base_sig}",SIGMA_SCN="${scn_sig}",CF_EXTRA="--selic-shock ${SELIC_SHOCK} --beta ${BETA} --horizon ${HORIZON}" \
        "${CL_ROOT}/cf_job.sh")
    echo "  cf5_compare -> ${FINAL_JOB}"
    FINAL_SIGMA="${scn_sig}"
    submit_zip "${ZIP_AFTER}" "${FINAL_JOB}" copy
    ;;

cf6)
    SDIR="${ROOT}/cf6_E${CF_ROUTINE}_${CF_STAGE}"
    dep=""
    if [[ -n "${BASE_SIGMA}" ]]; then
        base_sig="${BASE_SIGMA}"; echo "== CF6 base: REUSING ${base_sig} =="
    else
        echo "== CF6 base equilibrium -> ${SDIR}/base =="
        run_jacobi "${SDIR}/base" "" "cf6b"
        base_sig="${RJ_SIGMA}"; dep="${dep}:${RJ_JOB}"
    fi
    echo "== CF6 merged equilibrium (${MERGE}) -> ${SDIR}/merged =="
    run_jacobi "${SDIR}/merged" "--merge ${MERGE}" "cf6m"
    mg_sig="${RJ_SIGMA}"; dep="${dep}:${RJ_JOB}"
    dep="${dep#:}"
    echo "== CF6 compare (afterok:${dep}) =="
    FINAL_JOB=$(sub "cf6cmp_E${CF_ROUTINE}" "${COMPARE_TIME}" "${CPU_SB[@]}" \
        --dependency=afterok:"${dep}" \
        --export=ALL,CF_ROUTINE=${CF_ROUTINE},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED},CF_STEP=cf6_compare,SIGMA_BASE="${base_sig}",SIGMA_SCN="${mg_sig}",CF_EXTRA="--merge ${MERGE} --beta ${BETA} --horizon ${HORIZON}" \
        "${CL_ROOT}/cf_job.sh")
    echo "  cf6_compare -> ${FINAL_JOB}"
    FINAL_SIGMA="${mg_sig}"
    submit_zip "${ZIP_AFTER}" "${FINAL_JOB}" copy
    ;;

esac

echo "Watch: squeue -u \$USER"
echo "FINAL_JOB=${FINAL_JOB}"
echo "FINAL_SIGMA=${FINAL_SIGMA}"
