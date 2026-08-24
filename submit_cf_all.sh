#!/bin/bash
# submit_cf_all.sh — ONE-COMMAND orchestrator for the counterfactual pipeline on Bouchet.
#
# Run from the code directory on the login node; it does everything:
#   1. builds cluster_processed/ from the RC zip (blp_outputs_*.zip) if it's missing
#      (minimal unzip + cp of the four extended .jls — no python), then
#   2. preflight-checks the staged inputs (draws, RC results, forward r^f curve), then
#   3. submits the CF chain for EACH routine in ROUTINES:
#        (optional) demand_eval — 0a share reproduction, PER ROUTINE (shares_elas_E{k})
#        (optional) cf1         — gross franchise-value decomposition          [1 job]
#        (optional) cf1_net/cf3/cf5/cf6 — equilibrium CFs; consume the BBL cost params
#   The BBL cost-estimation stage (fwd_sim ψ deviations + eq:17 solve) is now a SEPARATE
#   stage — run `bash submit_bbl_all.sh` first; it writes cost_params_E*_spec_12_*.json,
#   which the equilibrium CFs here preflight and consume.
#
# Usage:
#     bash submit_cf_all.sh                       # E6 headline, everything
#     ROUTINES="5 6 7 8" bash submit_cf_all.sh    # headline + robustness band, one shot
#   Tunables (env):
#     ROUTINES="6"          routines to run CFs for (space list; "5 6 7 8" = headline+band)
#     CF_STAGE=extended  R=2000  SEED=42  BETA=0.9  HORIZON=50
#     DO_DEMAND_EVAL=1  DO_CF1=1     (CF1-gross)
#     DO_CF4=0                       (CF4 Pix reallocation; descriptive; needs φ^noPix uploaded — opt-in)
#     DO_CF1NET=1  DO_CF3=0  DO_CF5=0  DO_CF6=0   (equilibrium CFs; need cost params; cf3/5/6 opt-in)
#     CF_EQ_EXTRA=""        extra flags for cf3/cf5/cf6 (e.g. "--min-firm-markets 50 --selic-shock 0.01")
#     AUTO_PROCESS=1        auto-build cluster_processed/ from blp_outputs_*.zip if absent (unzip+cp)
#     BLP_ZIP=<path>        pin a specific RC zip (default: latest by job id across all saved)
#     SHARD_TIME=08:00:00   SOLVE_TIME=01:00:00   EQ_TIME=12:00:00 (cf3/5/6 wall)
#     PARTITION=<name>      SBATCH partition. submit_cf.sh defaults to `day` (CPU). These CF1
#                           steps are CPU-only — keep them on a CPU partition (day/week/bigmem/mpi).
#     GPUS=h200:1           request a GPU (Bouchet syntax). Not needed here — the CF1 chain is
#                           CPU-only; the GPU is for the cf3/cf5/cf6 equilibrium solve (submit_cf3_jacobi.sh).
#     MEM=200G              override --mem (needs ≥ ~100G for the R=2000 extended context)
#
# Inputs built LOCALLY and uploaded (the compute nodes lack internet / the sleep pickle):
#   data/input/forward_rf_qoq.csv          (cf_forward_rf.py) — cf3/cf5/cf6 rebuild ψ and read it.
#   data/input/{upsilon_pix,phi_nopix}_E{k}_spec_12.*  (cf_4_upsilon_export.py) — CF4 only (DO_CF4=1).
# The BBL cost params the equilibrium CFs consume come from the SEPARATE BBL stage
# (submit_bbl_all.sh → data/output/cost/cost_params_E*_spec_12_*.json); everything else is either
# already staged from the BLP run or auto-built here from the zip. See runbook §8.
set -euo pipefail

ROUTINES="${ROUTINES:-${CF_ROUTINE:-3 4}}"   # the single-index routines; authority is CL_ROUTINES_CF in cluster_lib.sh
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}"
DO_DEMAND_EVAL="${DO_DEMAND_EVAL:-1}"; DO_CF1="${DO_CF1:-1}"
# CF4 (Pix reallocation) is descriptive like cf1 (no costs), but needs the locally-built φ^noPix
# artifacts uploaded (see preflight) — opt-in so the default run is unchanged.
DO_CF4="${DO_CF4:-0}"
# Equilibrium CFs — consume the BBL cost params (preflighted below). cf1_net is light (on by
# default); cf3/cf5/cf6 re-solve the pricing game (heavier even with the market-local best-response) — opt-in.
DO_CF1NET="${DO_CF1NET:-1}"; DO_CF3="${DO_CF3:-0}"; DO_CF5="${DO_CF5:-0}"; DO_CF6="${DO_CF6:-0}"
AUTO_PROCESS="${AUTO_PROCESS:-1}"
SHARD_TIME="${SHARD_TIME:-08:00:00}"; SOLVE_TIME="${SOLVE_TIME:-01:00:00}"
EQ_TIME="${EQ_TIME:-12:00:00}"
LOGDIR="logs"; mkdir -p "${LOGDIR}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/../data}"
# RC results are produced + placed by the BLP family, which owns their layout. Accept whichever is in
# force (first EXISTING wins); fall back to the legacy path as the build target when none exists yet.
# Keep in sync with _result_path()'s candidate list in foundation_demand_eval.jl.
if [[ -z "${CP_DIR:-}" ]]; then
    for _c in "${DATA_ROOT}/output/BLP_RESULTS/cluster_processed" "${DATA_ROOT}/output/cluster_processed"; do
        [[ -d "${_c}" ]] && { CP_DIR="${_c}"; break; }
    done
    CP_DIR="${CP_DIR:-${DATA_ROOT}/output/cluster_processed}"
fi
ROUTINES_CSV="$(echo ${ROUTINES} | tr ' ' ',')"

# ── Step 1: build cluster_processed/ from the RC zip if any routine is missing ───
# Each RC run is saved as blp_outputs_<SLURM_JOB_ID>.zip (submit_blp_rc_all.sh), so with
# several runs on disk "latest" = highest job id. pick_zip returns: BLP_ZIP override →
# highest-job-id zip → (legacy fallback) newest by mtime.
pick_zip () {
    if [[ -n "${BLP_ZIP:-}" ]]; then printf '%s\n' "${BLP_ZIP}"; return; fi
    local best="" bestid=-1 f id
    for f in "${DATA_ROOT}"/output/blp_outputs_*.zip "${DATA_ROOT}"/output/cluster_raw/blp_outputs_*.zip; do
        [[ -f "$f" ]] || continue
        id="$(basename "$f" | sed -E 's/^blp_outputs_([0-9]+)\.zip$/\1/')"
        if [[ "$id" =~ ^[0-9]+$ ]] && (( id > bestid )); then bestid=$id; best="$f"; fi
    done
    [[ -n "$best" ]] || best="$(ls -t "${DATA_ROOT}"/output/blp_outputs*.zip "${DATA_ROOT}"/output/cluster_raw/blp_outputs*.zip 2>/dev/null | head -1 || true)"
    printf '%s\n' "${best}"
}

# Minimal, robust: the CF inputs are just the four extended result .jls — extract them straight
# from the zip (unzip + cp; no python, no pandas, no file reorganization). The RC zips are FLAT
# (submit_blp_rc_all.sh writes them with `zip -jm`), so `-j` + the member name lands each .jls in
# cluster_raw/; a `find` fallback covers a dir-prefixed member too. (process_blp_outputs.py stays
# the LOCAL tool for the full reorg + summaries; it is NOT needed — or trusted — here.)
need_process=0
for k in ${ROUTINES}; do [[ -f "${CP_DIR}/blp_E${k}_spec_12.jls" ]] || need_process=1; done
if [[ "${need_process}" == "1" && "${AUTO_PROCESS}" == "1" ]]; then
    zip="$(pick_zip)"; CRAW="${DATA_ROOT}/output/cluster_raw"
    if [[ -n "${zip}" && -f "${zip}" ]]; then
        echo "── building cluster_processed/ from $(basename "${zip}") (routines ${ROUTINES}) ──"
        mkdir -p "${CP_DIR}" "${CRAW}"
        for k in ${ROUTINES}; do
            src="blp_results_E${k}_spec_12_${CF_STAGE}.jls"
            unzip -o -j "${zip}" "${src}" "*/${src}" -d "${CRAW}" >/dev/null 2>&1 || true
            found="${CRAW}/${src}"
            [[ -f "${found}" ]] || found="$(find "${CRAW}" -name "${src}" 2>/dev/null | head -1)"
            if [[ -n "${found}" && -f "${found}" ]]; then
                cp -f "${found}" "${CP_DIR}/blp_E${k}_spec_12.jls"; echo "     E${k} ✓  (${src})"
            else
                echo "     E${k} ✗  — ${src} not found in $(basename "${zip}")"
            fi
        done
    else
        echo "── no blp_outputs_*.zip under ${DATA_ROOT}/output — cannot auto-build cluster_processed/ (has the RC run finished?)"
    fi
fi

# ── Step 2: preflight — every required input must be present ──────────────────────
RF_CURVE="${DATA_ROOT}/input/forward_rf_qoq.csv"          # UPLOADED input
DRAWS="${DATA_ROOT}/output/BLP_DRAWS/halton_nu_R${R}_seed${SEED}.jls"
miss=0
[[ -f "${DRAWS}" ]]    || { echo "MISSING R=${R} draws:  ${DRAWS}"; miss=1; }
[[ -f "${RF_CURVE}" ]] || { echo "MISSING forward curve: ${RF_CURVE}"; \
    echo "   → build locally then upload: python cf_forward_rf.py --horizon ${HORIZON} --start 2026Q1"; miss=1; }
for k in ${ROUTINES}; do
    [[ -f "${CP_DIR}/blp_E${k}_spec_12.jls" ]] || { echo "MISSING RC result: ${CP_DIR}/blp_E${k}_spec_12.jls"; \
        echo "   → drop the RC blp_outputs_*.zip in ${DATA_ROOT}/output and re-run (auto-built via unzip+cp),"; \
        echo "     pin one with BLP_ZIP=<path>, or extract by hand:"; \
        echo "     unzip -o -j <zip> 'blp_results_E${k}_spec_12_${CF_STAGE}.jls' -d ${CRAW:-${DATA_ROOT}/output/cluster_raw} && cp <...>_extended.jls ${CP_DIR}/blp_E${k}_spec_12.jls"; miss=1; }
done
# The equilibrium CFs (cf1_net/cf3/cf5/cf6) consume the BBL cost params. Those come from the
# SEPARATE BBL stage (submit_bbl_all.sh), which is an independent run — so preflight the JSON on
# disk rather than chaining an sbatch dependency across the two orchestrators. cf1 (gross) and
# demand_eval need no costs, so this check is skipped when only they are requested.
need_costs=0
[[ "${DO_CF1NET}" == "1" || "${DO_CF3}" == "1" || "${DO_CF5}" == "1" || "${DO_CF6}" == "1" ]] && need_costs=1
# CF_COST_AFTEROK=<jobid[:jobid…]> — when the BBL solve jobs are chained via afterok (an orchestrated
# run, e.g. submit_bbl_cf_all.sh), the cost params do NOT exist on disk yet; they'll be written before
# these CFs run. Skip the on-disk preflight and let SLURM enforce the ordering (afterok, below).
if [[ -n "${CF_COST_AFTEROK:-}" ]]; then
    need_costs=0
    echo "cost params chained via afterok:${CF_COST_AFTEROK} — skipping the on-disk cost_params preflight"
fi
if [[ "${need_costs}" == "1" ]]; then
    for k in ${ROUTINES}; do
        cp_json="${DATA_ROOT}/output/cost/cost_params_E${k}_spec_12_${CF_STAGE}.json"
        [[ -f "${cp_json}" ]] || { echo "MISSING BBL cost params: ${cp_json}"; \
            echo "   → run the BBL cost stage first:  bash submit_bbl_all.sh"; miss=1; }
    done
fi
# CF4 needs no BBL costs, but DOES need the exact link-aware φ^noPix + Υ_pix built LOCALLY (the
# sleep pickle is not on the compute nodes) and uploaded to data/input, like the forward r^f
# curve. Preflight both per routine when CF4 is requested.
cf4_note=""
if [[ "${DO_CF4}" == "1" ]]; then
    CFF="${DATA_ROOT}/input"          # upsilon_pix / phi_nopix are UPLOADED inputs
    for k in ${ROUTINES}; do
        for pf in "${CFF}/upsilon_pix_E${k}_spec_12.json" "${CFF}/phi_nopix_E${k}_spec_12.parquet"; do
            [[ -f "${pf}" ]] || { echo "MISSING CF4 input: ${pf}"; \
                echo "   → build locally then upload: python cf_4_upsilon_export.py --estim ${k} --spec 12"; miss=1; }
        done
    done
    cf4_note=" + CF4 phi_nopix"
fi
[[ "${miss}" == "0" ]] || { echo "Stage the missing input(s) (runbook §8), then re-run."; exit 1; }
# `${need_costs:+…}` was WRONG here: need_costs="0" is a NON-EMPTY string, so the ":+" expansion fired
# even when the cost preflight had been skipped — the line claimed "+ BBL cost params" on runs where
# nothing was verified (seen 2026-08-03 Phase 1b, with no cost_params on disk at all). Test the value.
costs_note=""; [[ "${need_costs}" == "1" ]] && costs_note=" + BBL cost params"
echo "Preflight OK: R=${R} draws + forward r^f curve + RC results${costs_note}${cf4_note} for routines: ${ROUTINES}"

# ── Step 3: submit the CF chain per routine ──────────────────────────────────────
cf1_extra="--beta ${BETA} --horizon ${HORIZON}"

submit () {  # submit <jobname> <time> <extra-sbatch-args...>
    local name="$1" tlim="$2"; shift 2
    # --kill-on-invalid-dep=yes: if an afterok dependency can never be satisfied (an upstream job
    # FAILED / was cancelled, incl. one array task), cancel this job instead of leaving it pending.
    sbatch --parsable -J "${name}" -t "${tlim}" --kill-on-invalid-dep=yes \
        ${PARTITION:+--partition="${PARTITION}"} ${GPUS:+--gpus="${GPUS}"} ${MEM:+--mem="${MEM}"} \
        -o "${LOGDIR}/${name}_%A_%a.out" -e "${LOGDIR}/${name}_%A_%a.err" "$@"
}

# ── Pre-warm barrier ──────────────────────────────────────────────────────────
# Many Julia processes starting cold against ONE shared NFS depot stampede its precompile/load
# lock — they sit at 0% CPU for a long time. So precompile the depot ONCE (serial job), and make
# demand_eval / cf1 / the equilibrium CFs all wait on it (afterok): they then load a warm cache
# with no lock fight. DO_WARMUP=0 to skip.
warm_dep=""
if [[ "${DO_WARMUP:-1}" == "1" ]]; then
    wj=$(submit "cf_warmup" "${SOLVE_TIME}" \
        --export=ALL,CF_ROUTINE=${ROUTINES%% *},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED},CF_STEP=warmup submit_cf.sh)
    echo "── pre-warm depot → job ${wj} (demand_eval/cf1/equilibrium CFs wait on it) ──"
    warm_dep="--dependency=afterok:${wj}"
fi

# Cost-consuming steps (cf1_net/cf3/cf5/cf6) wait on the warmup AND — when the BBL stage is chained
# into the same orchestrated run — the BBL solve jobs (CF_COST_AFTEROK). Merge into ONE afterok list:
# sbatch keeps only the LAST --dependency flag, so a step can't take warm_dep and a second dep flag
# separately. cf1(gross)/cf4/demand_eval need no costs and keep the plain warm_dep.
cost_ok_ids=""
[[ "${DO_WARMUP:-1}" == "1" ]] && cost_ok_ids="${wj}"
[[ -n "${CF_COST_AFTEROK:-}" ]] && cost_ok_ids="${cost_ok_ids:+${cost_ok_ids}:}${CF_COST_AFTEROK}"
cost_dep="${cost_ok_ids:+--dependency=afterok:${cost_ok_ids}}"

# The BBL cost params are produced by the separate BBL stage (preflighted above), NOT chained
# here — so every step depends only on the warmup. cf1_net/cf3/cf5/cf6 read cost_params_*.json off
# disk. Per-CF afterany dependency lists (colon-joined job ids) feed the end-of-run archiver.
found_dep=""; cf1_dep=""; cf4_dep=""; cf3_dep=""; cf5_dep=""; cf6_dep=""
for k in ${ROUTINES}; do
    base_export="CF_ROUTINE=${k},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED}"
    echo "── E${k} ${CF_STAGE} | R=${R} ──"
    # demand_eval runs PER ROUTINE: shares_elas_E{k}_… is the in-sample share reproduction for THAT
    # estimator, so a single routine's copy validates only that one. (It was gated on the first
    # routine back when ROUTINES defaulted to a single estimator and this doubled as a precompile
    # warm-up; with the E5-E8 default that silently produced validation for E5 alone. 2026-08-03.)
    if [[ "${DO_DEMAND_EVAL}" == "1" ]]; then
        j=$(submit "cf_demaneval_E${k}" "${SOLVE_TIME}" ${warm_dep} \
            --export=ALL,${base_export},CF_STEP=demand_eval submit_cf.sh)
        echo "  demand_eval  → job ${j}"; found_dep="${found_dep}:${j}"
    fi
    if [[ "${DO_CF1}" == "1" ]]; then
        j=$(submit "cf_cf1_E${k}" "${SHARD_TIME}" ${warm_dep} \
            --export=ALL,${base_export},CF_STEP=cf1,CF_EXTRA="${cf1_extra}" submit_cf.sh)
        echo "  cf1          → job ${j}"; cf1_dep="${cf1_dep}:${j}"
    fi
    if [[ "${DO_CF4}" == "1" ]]; then   # descriptive Pix reallocation; consumes the uploaded φ^noPix
        j=$(submit "cf_cf4_E${k}" "${SHARD_TIME}" ${warm_dep} \
            --export=ALL,${base_export},CF_STEP=cf4 submit_cf.sh)
        echo "  cf4          → job ${j}"; cf4_dep="${cf4_dep}:${j}"
    fi

    # Equilibrium CFs — consume the BBL cost params. Normally those are preflighted on disk and these
    # depend only on the warmup (${cost_dep}==${warm_dep}); in an orchestrated run CF_COST_AFTEROK folds
    # the BBL solve jobs into ${cost_dep} so they launch the instant the params are written.
    eq_extra="--beta ${BETA} --horizon ${HORIZON}"
    if [[ "${DO_CF1NET}" == "1" ]]; then
        j=$(submit "cf_cf1net_E${k}" "${SOLVE_TIME}" ${cost_dep} \
            --export=ALL,${base_export},CF_STEP=cf1_net,CF_EXTRA="${eq_extra}" submit_cf.sh)
        echo "  cf1_net      → job ${j}"; cf1_dep="${cf1_dep}:${j}"   # net → same cf1 archive
    fi
    for step in cf3 cf5 cf6; do
        flag="DO_$(echo ${step} | tr a-z A-Z)"          # DO_CF3 / DO_CF5 / DO_CF6
        if [[ "${!flag}" == "1" ]]; then
            j=$(submit "cf_${step}_E${k}" "${EQ_TIME}" ${cost_dep} \
                --export=ALL,${base_export},CF_STEP=${step},CF_EXTRA="${eq_extra} ${CF_EQ_EXTRA:-}" submit_cf.sh)
            echo "  ${step}          → job ${j}"
            case "${step}" in cf3) cf3_dep="${cf3_dep}:${j}";; cf5) cf5_dep="${cf5_dep}:${j}";; cf6) cf6_dep="${cf6_dep}:${j}";; esac
        fi
    done
done

# ── Archiving ──────────────────────────────────────────────────────────────────
# NO auto-zip here. zip_cf_outputs.sh MOVES files out of data/output/{cost,cf}, but the
# equilibrium CFs consume them long after this script exits (cost_params_E*.json → CF3/CF5/CF6;
# CF3's sig_6 → CF5 base). Archiving mid-pipeline strands those inputs inside the .zip. So archive
# ONCE, at the very end, after CF3/CF5/CF6 have finished:  bash zip_all_cf.sh

echo "Submitted CFs for routines: ${ROUTINES}. Watch with: squeue -u \$USER"
echo "When the WHOLE chain (incl. CF3/CF5/CF6) is done, archive with:  bash zip_all_cf.sh"
# Machine-parseable handle for an orchestrator (submit_bbl_cf_all.sh): every result job id this run
# submitted, colon-joined, so a final `afterany` archive job can wait on all of them before zipping.
# Keep this the LAST line so `sed -n 's/^CF_RESULT_JOBIDS=//p' | tail -1` captures it cleanly.
cf_all_ids="$(echo "${found_dep}${cf1_dep}${cf4_dep}${cf3_dep}${cf5_dep}${cf6_dep}" | sed 's/^://')"
echo "CF_RESULT_JOBIDS=${cf_all_ids}"
