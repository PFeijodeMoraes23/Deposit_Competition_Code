#!/bin/bash
# submit_bbl_all.sh — ONE-COMMAND orchestrator for the BBL cost-estimation stage on Bouchet.
#
# This is the marginal-cost ESTIMATION stage (formerly the cost2/cost_solve steps of
# submit_cf_all.sh). It produces data/COST_FWD/cost_params_E*_spec_12_*.json, which the
# counterfactuals (submit_cf_all.sh: cf1_net/cf3/cf5/cf6) then CONSUME. Structure mirrors
# submit_blp_rc_all.sh (the demand-estimation stage): a driver + this orchestrator.
#
# Run from the code directory on the login node; it does everything:
#   1. builds cluster_processed/ from the RC zip (blp_outputs_*.zip) if it's missing
#      (minimal unzip + cp of the four extended .jls — no python), then
#   2. preflight-checks the staged inputs (draws, RC results, forward r^f curve, fitted policy), then
#   3. submits the BBL Step-2 chain for EACH routine in ROUTINES:
#        (optional) polfunc     — BBL Step 1, fits the policy CSV        [1 job, DO_POLFUNC=1]
#        fwd_sim ARRAY          — ψ under σ̂ + σ̃ deviations, N_SHARDS tasks   [array job]
#        solve                  — eq:17 minimization, afterANY the array      [1 job]
#                                 (solves on surviving shards; a flaky shard won't block it)
#   4. auto-zips the per-routine cost_params_*.json into data/output/bbl_outputs_<jobid>.zip
#      (COPY, not move — the CFs consume the JSONs in place).
#
# Usage:
#     bash submit_bbl_all.sh                       # E6 headline
#     ROUTINES="5 6 7 8" bash submit_bbl_all.sh    # headline + robustness band, one shot
#   Tunables (env):
#     ROUTINES="6"          routines to estimate costs for (space list; "5 6 7 8" = headline+band)
#     CF_STAGE=extended  R=2000  SEED=42
#     SHOCKS=50  N_SHARDS=100  PERTURB_SCALE=2.0  DEV_SCHEME=grid  BETA=0.9  HORIZON=50
#     SHARD_TIME=08:00:00   SOLVE_TIME=01:00:00
#     MEM=256G              override --mem. The panel now iterates ~700k demand rows (was ~257k),
#                           so peak ≈170G at R=2000 extended — 256G gives ~1.5× headroom.
#     N_SHARDS=100          fwd_sim runs UNILATERAL deviations: one sim per (firm × Δ), ~300 firms ×
#                           50 Δ ≈ 15k sims. Per-sim cost scales with panel rows (now ~2.8× larger),
#                           so 100 shards ⇒ ~3.5h/shard at R=2000, comfortably inside the 8h wall.
#                           (Fallback if the cluster caps array width: N_SHARDS=50 SHARD_TIME=12:00:00.)
#     AUTO_PROCESS=1        auto-build cluster_processed/ from blp_outputs_*.zip if absent (unzip+cp)
#     BLP_ZIP=<path>        pin a specific RC zip (default: latest by job id across all saved)
#     DO_POLFUNC=0          if 1, run BBL Step 1 on the cluster as a pre-step (afterok warmup) and
#                           chain fwd_sim after it; default 0 → the fitted CSV is a preflighted input.
#     POLICY_CSV=<path>     BBL Step-1 fitted policy for σ̂ (default:
#                           data/COST_POLFUNC/polfunc_fitted_spec_12.csv). Built LOCALLY by
#                           estimation_bbl_1_polfunc.py and uploaded. REQUIRED (deviations must be
#                           formed around the fitted policy, not observed spreads — else frac_bind ≈ 0.5
#                           is mechanical and (ω,ζ,γ) are uninformative; V_Main line 551, §0A/§9).
#     ASSET_RETURN_COL=asset_gross_return_lag   give the deposit franchise its asset-side margin
#                           (NOT `gross_return_lag`, which is the deposit rate — see the note below).
#
# TWO inputs must be built LOCALLY and uploaded (both need data/tools the compute nodes lack):
#   data/COST_FWD/forward_rf_qoq.csv              (cf_forward_rf.py — needs internet)
#   data/COST_POLFUNC/polfunc_fitted_spec_12.csv  (estimation_bbl_1_polfunc.py — needs market_panel)
# everything else is either already staged from the BLP run or auto-built here from the zip.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTINES="${ROUTINES:-${BBL_ROUTINE:-6}}"
CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
SHOCKS="${SHOCKS:-50}"; N_SHARDS="${N_SHARDS:-100}"
PERTURB_SCALE="${PERTURB_SCALE:-2.0}"; DEV_SCHEME="${DEV_SCHEME:-grid}"
BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}"
SHARD_TIME="${SHARD_TIME:-08:00:00}"; SOLVE_TIME="${SOLVE_TIME:-01:00:00}"
MEM="${MEM:-256G}"
AUTO_PROCESS="${AUTO_PROCESS:-1}"
DO_POLFUNC="${DO_POLFUNC:-0}"
LOGDIR="logs"; mkdir -p "${LOGDIR}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/../data}"
CP_DIR="${DATA_ROOT}/output/cluster_processed"
RUN_MARKER="$(mktemp "${DATA_ROOT}/output/.bbl_run_marker.XXXXXX")"

# ── Step 1: build cluster_processed/ from the RC zip if any routine is missing ───
# (identical to submit_cf_all.sh: the fwd_sim needs the same extended RC result .jls the CFs need.)
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
POLICY_CSV="${POLICY_CSV:-${DATA_ROOT}/COST_POLFUNC/polfunc_fitted_spec_12.csv}"
miss=0
[[ -f "${DRAWS}" ]]    || { echo "MISSING R=${R} draws:  ${DRAWS}"; miss=1; }
[[ -f "${RF_CURVE}" ]] || { echo "MISSING forward curve: ${RF_CURVE}"; \
    echo "   → build locally then upload: python cf_forward_rf.py --horizon ${HORIZON} --start 2026Q1"; miss=1; }
# The fitted policy is REQUIRED unless we build it here (DO_POLFUNC=1): deviations formed around
# observed spreads make frac_bind ≈ 0.5 mechanical and the recovered costs uninformative.
if [[ "${DO_POLFUNC}" != "1" ]]; then
    [[ -f "${POLICY_CSV}" ]] || { echo "MISSING fitted policy: ${POLICY_CSV}"; \
        echo "   → build locally then upload: python estimation_bbl_1_polfunc.py --spec 12"; \
        echo "     (or set DO_POLFUNC=1 to run BBL Step 1 on the cluster as a pre-step)"; miss=1; }
fi
for k in ${ROUTINES}; do
    [[ -f "${CP_DIR}/blp_E${k}_spec_12.jls" ]] || { echo "MISSING RC result: ${CP_DIR}/blp_E${k}_spec_12.jls"; \
        echo "   → drop the RC blp_outputs_*.zip in ${DATA_ROOT}/output and re-run (auto-built via unzip+cp),"; \
        echo "     or pin one with BLP_ZIP=<path>."; miss=1; }
done
[[ "${miss}" == "0" ]] || { rm -f "${RUN_MARKER}"; echo "Stage the missing input(s) (runbook §8), then re-run."; exit 1; }
echo "Preflight OK: R=${R} draws + forward r^f curve + fitted policy + RC results for routines: ${ROUTINES}"

# ── Step 3: build the fwd_sim flags ───────────────────────────────────────────────
# Asset return r^j (V_Main eq 16, ψ1 row). Both flags are read as the QUARTERLY NET margin
# (r^j − r^f); foundation_psi_basis adds r^f back so ψ1 carries the GROSS r^j the paper requires.
# Left unset ⇒ r^j = r^f (zero asset margin), which makes a deposit worth only (ρ − c) and forces
# ω̂ < 0 in eq:17. Set one of these to give the deposit franchise its asset-side value:
#   ASSET_RETURN_COL=asset_gross_return_lag  (per-obs; = 1 + asset_return_qoq_lag, built in panel_4)
#   ASSET_MARGIN=0.015                       (constant quarterly net margin, e.g. 1.5%/q ≈ 6pp/yr)
# ⚠ NOT `gross_return_lag` — that is 1 + the DEPOSIT rate (what the bank PAYS). The asset return
#   is `asset_gross_return_lag`.
asset_flags=""
[[ -n "${ASSET_RETURN_COL:-}" ]] && asset_flags="--asset-return-col ${ASSET_RETURN_COL}"
[[ "${ASSET_MARGIN:-0}" != "0" ]] && asset_flags="${asset_flags} --asset-margin ${ASSET_MARGIN}"

# When DO_POLFUNC=0 the fitted policy is a preflighted input; feed it via --policy-csv. When
# DO_POLFUNC=1 it is produced on the cluster into the same default path, so still pass the flag.
policy_flag="--policy-csv ${POLICY_CSV}"
echo "fwd_sim σ̂ ← FITTED policy: ${POLICY_CSV}"

bbl_extra="--shocks ${SHOCKS} --perturb-scale ${PERTURB_SCALE} --dev-scheme ${DEV_SCHEME} --beta ${BETA} --horizon ${HORIZON}${asset_flags:+ ${asset_flags}} ${policy_flag}"

submit () {  # submit <jobname> <time> <extra-sbatch-args...>
    local name="$1" tlim="$2"; shift 2
    sbatch --parsable -J "${name}" -t "${tlim}" --kill-on-invalid-dep=yes \
        ${PARTITION:+--partition="${PARTITION}"} ${MEM:+--mem="${MEM}"} \
        -o "${LOGDIR}/${name}_%A_%a.out" -e "${LOGDIR}/${name}_%A_%a.err" "$@"
}

# ── Pre-warm barrier: precompile the depot ONCE so the fwd_sim array doesn't stampede the
#    shared-NFS precompile/load lock. DO_WARMUP=0 to skip; ARRAY_THROTTLE=N caps concurrent tasks.
warm_dep=""
if [[ "${DO_WARMUP:-1}" == "1" ]]; then
    wj=$(submit "bbl_warmup" "${SOLVE_TIME}" \
        --export=ALL,BBL_ROUTINE=${ROUTINES%% *},BBL_STAGE=${CF_STAGE},R=${R},SEED=${SEED},BBL_STEP=warmup submit_bbl.sh)
    echo "── pre-warm depot → job ${wj} (fwd_sim waits on it) ──"
    warm_dep="--dependency=afterok:${wj}"
fi
THROTTLE="${ARRAY_THROTTLE:+%${ARRAY_THROTTLE}}"

# Optional BBL Step 1 on the cluster (default: run locally, preflighted above). If enabled, it
# runs once (afterok warmup) and every routine's fwd_sim waits on it (it writes the shared policy CSV).
fwd_dep="${warm_dep}"
if [[ "${DO_POLFUNC}" == "1" ]]; then
    pj=$(submit "bbl_polfunc" "${SOLVE_TIME}" ${warm_dep} \
        --export=ALL,BBL_STAGE=${CF_STAGE},R=${R},SEED=${SEED},BBL_STEP=polfunc submit_bbl.sh)
    echo "── BBL Step 1 (polfunc) → job ${pj} (fwd_sim waits on it) ──"
    fwd_dep="--dependency=afterok:${pj}"
fi

solve_dep=""    # colon-joined solve job ids → the auto-zip waits on all of them
for k in ${ROUTINES}; do
    base_export="BBL_ROUTINE=${k},BBL_STAGE=${CF_STAGE},R=${R},SEED=${SEED}"
    echo "── E${k} ${CF_STAGE} | R=${R} | shocks=${SHOCKS} over ${N_SHARDS} shards ──"
    arr=$(submit "bbl_fwd_E${k}" "${SHARD_TIME}" ${fwd_dep} --array=0-$((N_SHARDS-1))${THROTTLE} \
        --export=ALL,${base_export},BBL_STEP=fwd_sim,N_SHARDS=${N_SHARDS},BBL_EXTRA="${bbl_extra}" \
        submit_bbl.sh)
    echo "  fwd_sim array → job ${arr} (${N_SHARDS} shards${THROTTLE:+, throttled ${THROTTLE}})"
    # solve depends on the fwd_sim array via afterANY: the eq:17 solve globs whatever psi_dev shards
    # exist, so a flaky shard doesn't block the routine's costs. (If shard 0 failed there's no psi_eq
    # → the solve errors and its afterok downstream is cancelled by --kill-on-invalid-dep.)
    slv=$(submit "bbl_solve_E${k}" "${SOLVE_TIME}" --dependency=afterany:"${arr}" \
        --export=ALL,${base_export},BBL_STEP=solve,BBL_EXTRA="--bootstrap 200" submit_bbl.sh)
    echo "  solve        → job ${slv} (afterany:${arr})"
    solve_dep="${solve_dep}:${slv}"
done

# ── auto-zip: one short CPU job, afterany ALL solves, bundles this run's cost_params into
#    data/output/bbl_outputs_<thisjobid>.zip. COPY (zip -j, NOT -jm): the CFs consume
#    cost_params_*.json IN PLACE (cf1_net/cf3/cf5/cf6), so moving them would strand the CF inputs.
#    Restricted to THIS run's outputs via find -newer ${RUN_MARKER}.
DATA_OUT="${DATA_ROOT}/output"
dep_csv="$(echo ${solve_dep} | sed 's/^://')"
if [[ -n "${dep_csv}" ]]; then
    ZIP_PARTITION="${ZIP_PARTITION:-day}"
    zip_jid=$(sbatch --parsable --dependency=afterany:${dep_csv} \
        --job-name=bbl_zip --partition="${ZIP_PARTITION}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${LOGDIR}/bbl_zip_%j.out" -e "${LOGDIR}/bbl_zip_%j.err" \
        --wrap "cd '${DATA_ROOT}/COST_FWD' && { \
                  find . -maxdepth 1 -newer '${RUN_MARKER}' -name 'cost_params_E*_spec_12_*.json' -print0 \
                    | xargs -0 -r zip -j \"${DATA_OUT}/bbl_outputs_\${SLURM_JOB_ID}.zip\" ; \
                  find '${HERE}'/logs -maxdepth 1 -newer '${RUN_MARKER}' \\( -name 'bbl_solve_*.out' -o -name 'bbl_solve_*.err' \\) -print0 \
                    | xargs -0 -r zip -j \"${DATA_OUT}/bbl_logs_\${SLURM_JOB_ID}.zip\" ; \
                  rm -f '${RUN_MARKER}' ; \
                  echo \"wrote \${PWD%/*}/output/bbl_outputs_\${SLURM_JOB_ID}.zip (cost_params COPIED — originals kept in place for the CFs)\"; }")
    echo "── auto-zip: ${zip_jid}  (afterany${solve_dep})"
    echo "   → ${DATA_OUT}/bbl_outputs_<${zip_jid}>.zip  (cost_params_*.json, copied)"
else
    rm -f "${RUN_MARKER}"
fi

echo "Submitted BBL cost estimation for routines: ${ROUTINES}. Watch with: squeue -u \$USER"
echo "When solve completes, the CFs can run:  bash submit_cf_all.sh  (it preflights cost_params_*.json)"
