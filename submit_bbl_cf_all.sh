#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════════════════════════
# submit_bbl_cf_all.sh — one-command orchestrator: BBL cost estimation → the requested CFs, in order,
# with the cross-stage dependency handled by SLURM (fire-and-forget; watch with `squeue -u $USER`).
#
# Launch order (matches the requested sequence):
#
#   Phase 1  (independent — submitted immediately, run concurrently):
#     • BBL cost estimation            submit_bbl_all.sh   → writes cost_params_E*_spec_12_${CF_STAGE}.json
#                                                            (fwd_sim on the H200 by default; see FWD_GPU)
#     • CF4 Pix, NO re-eval            submit_cf_all.sh  DO_CF4=1 CF4_EXACT_NOPIX=0
#         identity-link scalar fallback  φ_cf = clamp(φ̂ − Υ_pix·pix)  — needs NO costs, so it runs
#         alongside BBL. Writes the *_noeval CF4 outputs (distinct from the re-eval run below).
#         The baseline demand_eval (shares_elas) is produced once here.
#
#   Phase 2  (submitted right after Phase 1; SLURM orders them):
#     • CF1 gross    (DO_CF1=1)              franchise value, no costs      → depends only on the warmup
#     • CF4 re-eval  (CF4_EXACT_NOPIX=1)     exact index-then-link φ^noPix  → depends only on the warmup
#     • CF1 net      (DO_CF1NET=1)           net-of-cost franchise value    → consumes the BBL cost
#         params, so it is submitted with afterok on the BBL solve jobs (CF_COST_AFTEROK) and the
#         on-disk cost preflight is skipped — it launches the instant the cost params are written.
#
# Toggles (all default to the sequence above):
#   DO_BBL=1          run the BBL stage (0 → skip; cost params must already be on disk, and CF1-net
#                     then uses the normal on-disk preflight instead of afterok chaining)
#   DO_CF4_NOEVAL=1   Phase-1 CF4 identity-link fallback
#   DO_CF1_GROSS=1 / DO_CF1_NET=1 / DO_CF4_REEVAL=1     Phase-2 CFs
#   DO_DEMAND_EVAL=1  baseline shares_elas (attached to the first CF invocation)
#   DO_LOG_ZIP=1      final afterany job → ONE data/output/run_logs_<jobid>.zip with every .out/.err
#                     this run produced (~800 files at 100 shards × 4 routines), originals removed.
#                     Scoped by mtime marker, so earlier runs' logs are untouched. 0 → keep them loose.
#   DO_FINAL_ZIP=1    final afterany job → data/output/{foundation,cf1,cf4}_outputs.zip for download
#                     (FINAL_ZIP_CFS overrides which CFs; BBL cost params are auto-zipped separately)
#   ROUTINES="4"  CF_STAGE=extended  R=2000  SEED=42     (shared by both stages)
#   PY_MODULE/CONDA_ENV  (default miniconda/costsolve) — the BBL solve is PYTHON; its env is
#                     preflighted at submit time by submit_bbl_all.sh (PY_PREFLIGHT=0 skips).
#                     One-time: module load miniconda && conda create -y -n costsolve python=3.11 \
#                               numpy pandas scipy pyarrow statsmodels
#
# GOTCHA — a failed BBL solve silently takes cf1_net with it: cf1_net is chained afterok on the solve,
# so --kill-on-invalid-dep cancels it as DependencyNeverSatisfied and it produces NO LOG FILE AT ALL.
# A missing cf1_net log ⇒ look UPSTREAM at bbl_solve's .err, not at cf1_net. (Happened 2026-08-01:
# solve died at t+4s on a missing pandas; the env preflight above now catches that at submit time.)
#
# Everything the two sub-orchestrators accept still works via the environment (e.g. FWD_GPU=0 to put
# the BBL fwd_sim back on CPU, N_SHARDS=…, SHARD_TIME=…, DO_CF3/5/6 for extra equilibrium CFs).
# ══════════════════════════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── Shared config (both stages read these) ────────────────────────────────────────
# Default to the FULL lineup E1-E6 (E3 is the headline single-index estimator; a single-routine
# default silently ships one estimator only). Narrow with e.g. ROUTINES="3 4" when iterating.
export ROUTINES="${ROUTINES:-1 2 3 4 5 6}"
export CF_STAGE="${CF_STAGE:-extended}"
export R="${R:-2000}"
export SEED="${SEED:-42}"

# ── Phase toggles ─────────────────────────────────────────────────────────────────
DO_BBL="${DO_BBL:-1}"
DO_CF4_NOEVAL="${DO_CF4_NOEVAL:-1}"
DO_CF1_GROSS="${DO_CF1_GROSS:-1}"
DO_CF1_NET="${DO_CF1_NET:-1}"
DO_CF4_REEVAL="${DO_CF4_REEVAL:-1}"
ORCH_DEMAND_EVAL="${DO_DEMAND_EVAL:-1}"   # baseline shares_elas — attach to the first CF invocation
# Final archive: one short afterany job zips this run's CF outputs into data/output/ for download.
# ALWAYS in KEEP=1 COPY mode: zip_cf_outputs.sh otherwise MOVES (rm -f) what it archives, and the cf4
# pattern sweeps upsilon_pix_E*.json + phi_nopix_E*.parquet — which are INPUTS built locally from the
# sleep pickle (absent on the cluster) and uploaded. Move-mode emptied CF_FOUNDATION on 2026-08-01.
# Copy mode keeps every original on disk so re-runs and downstream CFs still find their inputs.
# 0 → skip. FINAL_ZIP_CFS scopes which CFs get archived (never add cf2: that glob owns the psi shards).
DO_FINAL_ZIP="${DO_FINAL_ZIP:-1}"
FINAL_ZIP_CFS="${FINAL_ZIP_CFS:-foundation cf1 cf4}"

# A full run writes ~800 log files (each fwd_sim array task emits a .out and a .err, ×N_SHARDS
# ×routines). DO_LOG_ZIP=1 adds ONE short job, afterany EVERYTHING, that bundles this run's logs into
# data/output/run_logs_<jobid>.zip and REMOVES the originals (zip -m: only what it archived is
# deleted). Scoped by mtime against a marker created below, so older logs are untouched.
DO_LOG_ZIP="${DO_LOG_ZIP:-1}"
LOG_MARKER="$(mktemp logs/.orch_marker.XXXXXX)"

CF_JOBIDS=""   # every CF result job id this run submits (both phases) → the final-zip afterany
ALL_JOBIDS=""  # + BBL jobs + the archive jobs → the log archiver waits on ALL of them
add_jobids() { [[ -n "$1" ]] && ALL_JOBIDS="${ALL_JOBIDS:+${ALL_JOBIDS}:}$1"; }
capture_cf_ids() {  # $1 = captured submit_cf_all.sh stdout
    local ids; ids="$(printf '%s\n' "$1" | sed -n 's/^CF_RESULT_JOBIDS=//p' | tail -n1)"
    [[ -n "${ids}" ]] && { CF_JOBIDS="${CF_JOBIDS:+${CF_JOBIDS}:}${ids}"; add_jobids "${ids}"; }
}

echo "══════════════════════════════════════════════════════════════════════════════"
echo " BBL→CF orchestrator | routines='${ROUTINES}' stage=${CF_STAGE} R=${R} seed=${SEED}"
echo " BBL=${DO_BBL} | CF4-noeval=${DO_CF4_NOEVAL} | CF1-gross=${DO_CF1_GROSS} CF1-net=${DO_CF1_NET} CF4-reeval=${DO_CF4_REEVAL}"
echo "══════════════════════════════════════════════════════════════════════════════"

# ── Phase 1a: BBL cost estimation (capture its solve job ids for the CF1-net afterok) ─────
COST_AFTEROK=""
if [[ "${DO_BBL}" == "1" ]]; then
    echo; echo "═══ Phase 1a: BBL cost estimation (submit_bbl_all.sh) ═══"
    # Capture WITHOUT letting set -e abort before the output is shown: the sub-orchestrator's
    # preflight prints its "MISSING ..." diagnostics to stdout, and dying on the $(...) assignment
    # swallowed them entirely (observed on Bouchet 2026-08-07: Phase 1a printed nothing and no job
    # was submitted). Same pattern at every capture site below.
    set +e; bbl_out="$(bash submit_bbl_all.sh)"; bbl_rc=$?; set -e
    printf '%s\n' "${bbl_out}"
    if [[ ${bbl_rc} -ne 0 ]]; then
        echo "ERROR: submit_bbl_all.sh failed (rc=${bbl_rc}) — its output is above (usually a preflight MISSING)." >&2
        rm -f "${LOG_MARKER}"
        exit "${bbl_rc}"
    fi
    COST_AFTEROK="$(printf '%s\n' "${bbl_out}" | sed -n 's/^BBL_SOLVE_JOBIDS=//p' | tail -n1)"
    add_jobids "$(printf '%s\n' "${bbl_out}" | sed -n 's/^BBL_ALL_JOBIDS=//p' | tail -n1)"
    if [[ -z "${COST_AFTEROK}" && "${DO_CF1_NET}" == "1" ]]; then
        echo "ERROR: submit_bbl_all.sh emitted no BBL_SOLVE_JOBIDS — cannot chain CF1-net. Aborting." >&2
        echo "  (Re-run with DO_CF1_NET=0, or DO_BBL=0 once cost_params_*.json are on disk.)" >&2
        exit 1
    fi
    echo "  → BBL solve jobs for the CF1-net afterok: ${COST_AFTEROK:-<none>}"
else
    echo; echo "═══ Phase 1a: BBL SKIPPED (DO_BBL=0) — CF1-net will preflight cost_params_*.json on disk ═══"
fi

# ── Phase 1b: CF4 Pix, NO re-eval (identity-link fallback; needs no costs) ─────────
first_cf_demand="${ORCH_DEMAND_EVAL}"   # run the baseline on the first CF invocation only
if [[ "${DO_CF4_NOEVAL}" == "1" ]]; then
    echo; echo "═══ Phase 1b: CF4 no re-eval  (CF4_EXACT_NOPIX=0) ═══"
    set +e
    out="$(DO_DEMAND_EVAL="${first_cf_demand}" DO_CF1=0 DO_CF1NET=0 DO_CF4=1 DO_CF3=0 DO_CF5=0 DO_CF6=0 \
           CF4_EXACT_NOPIX=0 \
           bash submit_cf_all.sh)"
    cf_rc=$?; set -e
    printf '%s\n' "${out}"
    [[ ${cf_rc} -ne 0 ]] && { echo "ERROR: Phase-1b submit_cf_all.sh failed (rc=${cf_rc}) — output above." >&2; rm -f "${LOG_MARKER}"; exit "${cf_rc}"; }
    capture_cf_ids "${out}"
    first_cf_demand=0                    # baseline already scheduled
fi

# ── Phase 2: CF1 gross + CF4 re-eval (no costs) + CF1 net (afterok BBL solve) ──────
if [[ "${DO_CF1_GROSS}" == "1" || "${DO_CF1_NET}" == "1" || "${DO_CF4_REEVAL}" == "1" ]]; then
    echo; echo "═══ Phase 2: CF1 gross + CF1 net + CF4 re-eval  (CF4_EXACT_NOPIX=1) ═══"
    p2_env=(DO_DEMAND_EVAL="${first_cf_demand}"
            DO_CF1="${DO_CF1_GROSS}" DO_CF1NET="${DO_CF1_NET}" DO_CF4="${DO_CF4_REEVAL}"
            DO_CF3=0 DO_CF5=0 DO_CF6=0 CF4_EXACT_NOPIX=1)
    [[ -n "${COST_AFTEROK}" ]] && p2_env+=(CF_COST_AFTEROK="${COST_AFTEROK}")
    set +e; out="$(env "${p2_env[@]}" bash submit_cf_all.sh)"; cf_rc=$?; set -e
    printf '%s\n' "${out}"
    [[ ${cf_rc} -ne 0 ]] && { echo "ERROR: Phase-2 submit_cf_all.sh failed (rc=${cf_rc}) — output above." >&2; rm -f "${LOG_MARKER}"; exit "${cf_rc}"; }
    capture_cf_ids "${out}"
fi

# ── Final archive: one short afterany job zips this run's CF outputs for download ──
final_zip_jid=""
if [[ "${DO_FINAL_ZIP}" == "1" && -n "${CF_JOBIDS}" ]]; then
    echo; echo "═══ Final archive: zip ${FINAL_ZIP_CFS} → data/output/ (afterany all CF jobs) ═══"
    final_zip_jid=$(sbatch --parsable --dependency=afterany:"${CF_JOBIDS}" \
        -J cf_final_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o logs/cf_final_zip_%j.out -e logs/cf_final_zip_%j.err \
        --wrap "cd '$(pwd)' && KEEP=1 CFS='${FINAL_ZIP_CFS}' bash zip_all_cf.sh")
    echo "  final CF archive → job ${final_zip_jid} (afterany:${CF_JOBIDS})"
    add_jobids "${final_zip_jid}"
elif [[ "${DO_FINAL_ZIP}" == "1" ]]; then
    echo; echo "── final archive skipped: no CF jobs were submitted ──"
fi

# ── Log archive: ONE zip for the whole run, afterany every job above ───────────────
log_zip_jid=""
if [[ "${DO_LOG_ZIP}" == "1" && -n "${ALL_JOBIDS}" ]]; then
    echo; echo "═══ Log archive: bundle this run's logs → data/output/run_logs_<jobid>.zip ═══"
    # -newer "${LOG_MARKER}" scopes this to logs written AFTER the orchestrator started, so previous
    # runs' logs survive. Excludes its own .out/.err (still open) and any existing archive. `zip -m`
    # deletes only entries it actually archived, so a skipped file is kept rather than lost.
    log_zip_jid=$(sbatch --parsable --dependency=afterany:"${ALL_JOBIDS}" \
        -J run_log_zip --partition="${ZIP_PARTITION:-day}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G \
        -o logs/run_log_zip_%j.out -e logs/run_log_zip_%j.err \
        --wrap "cd '$(pwd)' && { \
                  find logs -maxdepth 1 -type f -newer '${LOG_MARKER}' \
                       \\( -name '*.out' -o -name '*.err' \\) ! -name \"run_log_zip_\${SLURM_JOB_ID}.*\" -print0 \
                    | xargs -0 -r zip -qmj '../data/output/run_logs_'\"\${SLURM_JOB_ID}\"'.zip' ; \
                  rm -f '${LOG_MARKER}' ; \
                  echo \"bundled \$(unzip -l '../data/output/run_logs_'\"\${SLURM_JOB_ID}\"'.zip' | tail -1) — originals removed\"; }")
    echo "  run-log archive → job ${log_zip_jid} (afterany ${ALL_JOBIDS//:/, })"
else
    rm -f "${LOG_MARKER}"
fi

echo
echo "══════════════════════════════════════════════════════════════════════════════"
echo " All stages submitted. Track:  squeue -u \$USER"
echo "   • CF1-net waits on the BBL solve (afterok) — it stays PENDING until the cost params exist."
echo "   • CF4 writes two output sets: *_noeval (Phase 1b) and the exact re-eval (Phase 2)."
echo " Downloads when everything finishes:"
if [[ "${DO_BBL}" == "1" ]]; then
    echo "   • BBL cost params : data/output/bbl_outputs_<jobid>.zip   (BBL auto-zip)"
fi
if [[ -n "${final_zip_jid}" ]]; then
    echo "   • CF outputs      : data/output/{$(echo ${FINAL_ZIP_CFS} | tr ' ' ',')}_outputs.zip   (job ${final_zip_jid})"
else
    echo "   • CF outputs      : once squeue is empty, archive in COPY mode →"
    echo "                       KEEP=1 CFS='foundation cf1 cf4' bash zip_all_cf.sh"
    echo "                       (bare 'zip_all_cf.sh' MOVES files and its default CFS includes cf2,"
    echo "                        which would sweep the psi_* shards out of data/output/cost)"
fi
if [[ -n "${log_zip_jid}" ]]; then
    echo "   • run logs        : data/output/run_logs_<${log_zip_jid}>.zip   (all .out/.err, originals removed)"
fi
echo "══════════════════════════════════════════════════════════════════════════════"
