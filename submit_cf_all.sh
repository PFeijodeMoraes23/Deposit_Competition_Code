#!/bin/bash
# submit_cf_all.sh — ONE-COMMAND orchestrator for the counterfactual pipeline on Bouchet.
#
# Run from the code directory on the login node; it does everything:
#   1. builds cluster_processed/ from the RC zip (blp_outputs_*.zip) if it's missing
#      (minimal unzip + cp of the four extended .jls — no python), then
#   2. preflight-checks the staged inputs (draws, RC results, forward r^f curve), then
#   3. submits the CF chain for EACH routine in ROUTINES:
#        (optional) demand_eval — 0a share reproduction (first routine only; precompiles)
#        (optional) cf1         — gross franchise-value decomposition          [1 job]
#        cost2 ARRAY            — CF2 ψ deviations, N_SHARDS tasks              [array job]
#        cost_solve             — Eq-18 minimization, afterANY the array          [1 job]
#                                 (solves on surviving shards; a flaky shard won't block it)
#   Each cost2 array task computes SHOCKS/N_SHARDS deviations (reproducible per-shock RNG
#   → shards merge exactly); the solver globs all shards.
#
# Usage:
#     bash submit_cf_all.sh                       # E6 headline, everything
#     ROUTINES="5 6 7 8" bash submit_cf_all.sh    # headline + robustness band, one shot
#   Tunables (env):
#     ROUTINES="6"          routines to run CFs for (space list; "5 6 7 8" = headline+band)
#     CF_STAGE=extended  R=2000  SEED=42
#     SHOCKS=50  N_SHARDS=10  PERTURB_SCALE=0.02  DEV_SCHEME=grid  BETA=0.9  HORIZON=50
#     DO_DEMAND_EVAL=1  DO_CF1=1     (CF1-gross + CF2 chain)
#     DO_CF1NET=1  DO_CF3=0  DO_CF5=0  DO_CF6=0   (equilibrium CFs, afterok cost_solve; cf3/5/6 opt-in)
#     CF_EQ_EXTRA=""        extra flags for cf3/cf5/cf6 (e.g. "--min-firm-markets 50 --selic-shock 0.01")
#     AUTO_PROCESS=1        auto-build cluster_processed/ from blp_outputs_*.zip if absent (unzip+cp)
#     BLP_ZIP=<path>        pin a specific RC zip (default: latest by job id across all saved)
#     SHARD_TIME=08:00:00   SOLVE_TIME=01:00:00   EQ_TIME=12:00:00 (cf3/5/6 wall)
#     PARTITION=<name>      SBATCH partition. submit_cf.sh defaults to `day` (CPU). These CF2/CF1
#                           steps are CPU-only — keep them on a CPU partition (day/week/bigmem/mpi).
#     GPUS=h200:1           request a GPU (Bouchet syntax). Not needed here — the CF2/CF1 chain is
#                           CPU-only; the GPU is for the cf3/cf5/cf6 equilibrium solve (submit_cf3_jacobi.sh).
#     MEM=200G              override --mem (needs ≥ ~100G for the R=2000 extended context)
#
# The forward r^f curve (data/COST_FWD/forward_rf_qoq.csv) is the one input that must be
# built LOCALLY (cf_forward_rf.py needs internet) and uploaded; everything else is either
# already staged from the BLP run or auto-built here from the zip. See runbook §8.
set -euo pipefail

ROUTINES="${ROUTINES:-${CF_ROUTINE:-6}}"
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
# cost2 now runs UNILATERAL (Nash) deviations: one forward sim per (firm × Δ), not per Δ.
# With ~300 choice firms × 50 Δ that is ~15k sims (was 50) — so N_SHARDS must be much larger.
# 50 shards ⇒ ~300 sims each (~2.5h at R=2000), inside the 8h SHARD_TIME wall.
SHOCKS="${SHOCKS:-50}"; N_SHARDS="${N_SHARDS:-50}"
# 2.0 ρ-units = 200bp ≈ 54% of the median choice spread (~3.7pp); the ± grid runs graduated
# magnitudes up to that. The old 0.02 (=2bp) was pure linear regime — no curvature in g at all.
PERTURB_SCALE="${PERTURB_SCALE:-2.0}"; DEV_SCHEME="${DEV_SCHEME:-grid}"
BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}"
DO_DEMAND_EVAL="${DO_DEMAND_EVAL:-1}"; DO_CF1="${DO_CF1:-1}"
# Equilibrium CFs (afterok cost_solve). cf1_net is light (on by default); cf3/cf5/cf6 re-solve
# the pricing game (heavier even with the market-local best-response) — opt-in.
DO_CF1NET="${DO_CF1NET:-1}"; DO_CF3="${DO_CF3:-0}"; DO_CF5="${DO_CF5:-0}"; DO_CF6="${DO_CF6:-0}"
AUTO_PROCESS="${AUTO_PROCESS:-1}"
SHARD_TIME="${SHARD_TIME:-08:00:00}"; SOLVE_TIME="${SOLVE_TIME:-01:00:00}"
EQ_TIME="${EQ_TIME:-12:00:00}"
LOGDIR="logs"; mkdir -p "${LOGDIR}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/../data}"
CP_DIR="${DATA_ROOT}/output/cluster_processed"
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
RF_CURVE="${DATA_ROOT}/COST_FWD/forward_rf_qoq.csv"
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
[[ "${miss}" == "0" ]] || { echo "Stage the missing input(s) (runbook §8), then re-run."; exit 1; }
echo "Preflight OK: R=${R} draws + forward r^f curve + RC results for routines: ${ROUTINES}"

# ── Step 3: submit the CF chain per routine ──────────────────────────────────────
# Asset return r^j (V_Main eq 16, ψ1 row). Both flags are read as the QUARTERLY NET margin
# (r^j − r^f); cf_0_psi_basis adds r^f back so ψ1 carries the GROSS r^j the paper requires.
# Left unset ⇒ r^j = r^f (zero asset margin), which makes a deposit worth only (ρ − c) and,
# with the observed spreads, forces ω to its ≥0 bound in eq-18. Set one of these to give the
# deposit franchise its asset-side value:
#   ASSET_RETURN_COL=gross_return_lag   (per-obs, from the demand parquet)
#   ASSET_MARGIN=0.015                  (constant quarterly net margin, e.g. 1.5%/q ≈ 6pp/yr)
asset_flags=""
[[ -n "${ASSET_RETURN_COL:-}" ]] && asset_flags="--asset-return-col ${ASSET_RETURN_COL}"
[[ "${ASSET_MARGIN:-0}" != "0" ]] && asset_flags="${asset_flags} --asset-margin ${ASSET_MARGIN}"
cf2_extra="--shocks ${SHOCKS} --perturb-scale ${PERTURB_SCALE} --dev-scheme ${DEV_SCHEME} --beta ${BETA} --horizon ${HORIZON}${asset_flags:+ ${asset_flags}}"
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
# demand_eval / cf1 / cost2 all wait on it (afterok): they then load a warm cache with no lock
# fight. DO_WARMUP=0 to skip; ARRAY_THROTTLE=N caps concurrent cost2 tasks (belt-and-suspenders).
warm_dep=""
if [[ "${DO_WARMUP:-1}" == "1" ]]; then
    wj=$(submit "cf_warmup" "${SOLVE_TIME}" \
        --export=ALL,CF_ROUTINE=${ROUTINES%% *},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED},CF_STEP=warmup submit_cf.sh)
    echo "── pre-warm depot → job ${wj} (demand_eval/cf1/cost2 wait on it) ──"
    warm_dep="--dependency=afterok:${wj}"
fi
THROTTLE="${ARRAY_THROTTLE:+%${ARRAY_THROTTLE}}"

# Per-CF afterany dependency lists (colon-joined job ids) — one auto-zip job per CF is submitted
# after ALL routines finish, so each <cf>_outputs.zip bundles every estimation in one archive.
found_dep=""; cf1_dep=""; cf2_dep=""; cf3_dep=""; cf5_dep=""; cf6_dep=""
first=1
for k in ${ROUTINES}; do
    base_export="CF_ROUTINE=${k},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED}"
    echo "── E${k} ${CF_STAGE} | R=${R} | shocks=${SHOCKS} over ${N_SHARDS} shards ──"
    # demand_eval runs once (first routine); it waits on the warmup so the cache is already built.
    if [[ "${DO_DEMAND_EVAL}" == "1" && "${first}" == "1" ]]; then
        j=$(submit "cf_demaneval_E${k}" "${SOLVE_TIME}" ${warm_dep} \
            --export=ALL,${base_export},CF_STEP=demand_eval submit_cf.sh)
        echo "  demand_eval  → job ${j}"; found_dep="${found_dep}:${j}"
    fi
    if [[ "${DO_CF1}" == "1" ]]; then
        j=$(submit "cf_cf1_E${k}" "${SHARD_TIME}" ${warm_dep} \
            --export=ALL,${base_export},CF_STEP=cf1,CF_EXTRA="${cf1_extra}" submit_cf.sh)
        echo "  cf1          → job ${j}"; cf1_dep="${cf1_dep}:${j}"
    fi
    arr=$(submit "cf_cost2_E${k}" "${SHARD_TIME}" ${warm_dep} --array=0-$((N_SHARDS-1))${THROTTLE} \
        --export=ALL,${base_export},CF_STEP=cost2,N_SHARDS=${N_SHARDS},CF_EXTRA="${cf2_extra}" \
        submit_cf.sh)
    echo "  cost2 array  → job ${arr} (${N_SHARDS} shards${THROTTLE:+, throttled ${THROTTLE}})"
    # cost_solve depends on the cost2 array via afterANY (not afterok): the Eq-18 solve globs
    # whatever psi_dev shards exist, so a flaky shard doesn't block the routine's costs — it just
    # solves on the surviving deviations. (If shard 0 failed there's no psi_eq → the solve errors
    # and its own afterok downstream is cancelled by --kill-on-invalid-dep.)
    slv=$(submit "cf_solve_E${k}" "${SOLVE_TIME}" --dependency=afterany:"${arr}" \
        --export=ALL,${base_export},CF_STEP=cost_solve,CF_EXTRA="--bootstrap 200" submit_cf.sh)
    echo "  cost_solve   → job ${slv} (afterany:${arr})"
    cf2_dep="${cf2_dep}:${slv}"     # cost_solve implies the array is done → covers psi_* + cost_params

    # Equilibrium CFs — each runs only after this routine's costs are solved.
    eq_extra="--beta ${BETA} --horizon ${HORIZON}"
    if [[ "${DO_CF1NET}" == "1" ]]; then
        j=$(submit "cf_cf1net_E${k}" "${SOLVE_TIME}" --dependency=afterok:"${slv}" \
            --export=ALL,${base_export},CF_STEP=cf1_net,CF_EXTRA="${eq_extra}" submit_cf.sh)
        echo "  cf1_net      → job ${j} (afterok:${slv})"; cf1_dep="${cf1_dep}:${j}"   # net → same cf1 archive
    fi
    for step in cf3 cf5 cf6; do
        flag="DO_$(echo ${step} | tr a-z A-Z)"          # DO_CF3 / DO_CF5 / DO_CF6
        if [[ "${!flag}" == "1" ]]; then
            j=$(submit "cf_${step}_E${k}" "${EQ_TIME}" --dependency=afterok:"${slv}" \
                --export=ALL,${base_export},CF_STEP=${step},CF_EXTRA="${eq_extra} ${CF_EQ_EXTRA:-}" submit_cf.sh)
            echo "  ${step}          → job ${j} (afterok:${slv})"
            case "${step}" in cf3) cf3_dep="${cf3_dep}:${j}";; cf5) cf5_dep="${cf5_dep}:${j}";; cf6) cf6_dep="${cf6_dep}:${j}";; esac
        fi
    done
    first=0
done

# ── Archiving ──────────────────────────────────────────────────────────────────
# NO auto-zip here. zip_cf_outputs.sh MOVES files out of data/COST_FWD & CF_FOUNDATION, but the
# equilibrium CFs consume them long after this script exits (cost_params_E*.json → CF3/CF5/CF6;
# CF3's sig_6 → CF5 base). Archiving mid-pipeline strands those inputs inside the .zip. So archive
# ONCE, at the very end, after CF3/CF5/CF6 have finished:  bash zip_all_cf.sh

echo "Submitted CFs for routines: ${ROUTINES}. Watch with: squeue -u \$USER"
echo "When the WHOLE chain (incl. CF3/CF5/CF6) is done, archive with:  bash zip_all_cf.sh"
