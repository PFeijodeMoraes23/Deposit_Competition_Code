#!/bin/bash
# submit_cf3_jacobi.sh — full-panel CF3 equilibrium spreads on Bouchet via FIRM-SHARDED JACOBI.
#
# The best-response re-simulates the full national panel per evaluation (D-firm shares are
# national), so a single-process solve is intractable at R=2000. Instead each Jacobi SWEEP is an
# array of shard jobs — shard s best-responds firms {j : j mod N_FIRM_SHARDS == s} against the
# FROZEN previous-sweep σ — followed by a merge into the next σ. Sweeps chain via afterok. This
# uses the exact (correct) full-panel psi_under; it just parallelizes the firms.
#
# Prerequisite: the RC costs (cost_params_E{k}_spec_12_extended.json) — i.e. run the BBL cost
# stage first:  bash submit_bbl_all.sh
#
# The shard best-responses are share-evaluation-bound, so they run on the GPU (the CF share
# kernel auto-uses the H200 when present, CF_GPU!=0). init/merge are light and stay on CPU.
#
# Usage (login node, code dir):
#   CF_ROUTINE=6 N_FIRM_SHARDS=40 N_SWEEPS=6 bash submit_cf3_jacobi.sh
# Tunables (env): CF_ROUTINE CF_STAGE=extended R=2000 SEED=42 N_FIRM_SHARDS N_SWEEPS
#                 BR_GRID=7 BR_WINDOW=0.02 BETA=0.9 HORIZON=50 SHARD_TIME MERGE_TIME MEM
#                 GPU_PARTITION=gpu_h200  GPUS=h200:1  (shard jobs)
#                 CPU_PARTITION=day       (init/merge)   CF_GPU=1 (0 ⇒ force CPU shards)
set -euo pipefail

CF_ROUTINE="${CF_ROUTINE:-6}"; CF_STAGE="${CF_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
N_FIRM_SHARDS="${N_FIRM_SHARDS:-40}"; N_SWEEPS="${N_SWEEPS:-6}"
BR_GRID="${BR_GRID:-7}"; BR_WINDOW="${BR_WINDOW:-0.02}"; BETA="${BETA:-0.9}"; HORIZON="${HORIZON:-50}"
SHARD_TIME="${SHARD_TIME:-06:00:00}"; MERGE_TIME="${MERGE_TIME:-00:30:00}"
GPU_PARTITION="${GPU_PARTITION:-gpu_h200}"; GPUS="${GPUS:-h200:1}"; CPU_PARTITION="${CPU_PARTITION:-day}"
# ZIP_AFTER: which CF archive to auto-zip into after the final sweep (default cf3). CF5/CF6 drive
# this script for their base/scenario solves and set ZIP_AFTER="" so THEY zip cf5/cf6 instead.
ZIP_AFTER="${ZIP_AFTER-cf3}"; DO_ZIP="${DO_ZIP:-1}"   # note: -, not :-, so an explicit "" suppresses
DATA_ROOT="${DATA_ROOT:-$(pwd)/../data}"
# SIGMA_DIR + EXTRA_CF are overridable so CF5/CF6 can drive this as a scenario solver (a shocked /
# merged equilibrium into its own σ dir). It prints FINAL_JOB=<id> / FINAL_SIGMA=<path> for chaining.
SIGMA_DIR="${SIGMA_DIR:-${DATA_ROOT}/output/cf/cf3_jacobi_E${CF_ROUTINE}_${CF_STAGE}}"
LOGDIR="logs"; mkdir -p "${LOGDIR}" "${SIGMA_DIR}"

COST="${DATA_ROOT}/output/cost/cost_params_E${CF_ROUTINE}_spec_12_${CF_STAGE}.json"
[[ -f "${COST}" ]] || { echo "MISSING costs: ${COST}"; echo "  → run the BBL cost stage first:  bash submit_bbl_all.sh"; exit 1; }
echo "CF3 firm-sharded Jacobi: E${CF_ROUTINE} ${CF_STAGE} | ${N_FIRM_SHARDS} shards × ${N_SWEEPS} sweeps"
echo "  shards → ${GPU_PARTITION} (--gpus=${GPUS}) | init/merge → ${CPU_PARTITION} | → ${SIGMA_DIR}"

base="CF_ROUTINE=${CF_ROUTINE},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED},SIGMA_DIR=${SIGMA_DIR},N_FIRM_SHARDS=${N_FIRM_SHARDS}"
eq="--beta ${BETA} --horizon ${HORIZON} --br-grid ${BR_GRID} --br-window ${BR_WINDOW} ${EXTRA_CF:-}"
CPU_SB=(--partition="${CPU_PARTITION}")
GPU_SB=(--partition="${GPU_PARTITION}" --gpus="${GPUS}")
sub () { local n="$1" t="$2"; shift 2
    # --kill-on-invalid-dep=yes: cancel (don't hang) if an upstream sweep/merge fails.
    sbatch --parsable -J "${n}" -t "${t}" --kill-on-invalid-dep=yes ${MEM:+--mem="${MEM}"} \
        -o "${LOGDIR}/${n}_%A_%a.out" -e "${LOGDIR}/${n}_%A_%a.err" "$@"; }

# GPU pre-warm: precompile/load the CUDA share path ONCE on a GPU node, so the N_FIRM_SHARDS shard
# tasks don't stampede the shared-NFS depot lock loading CUDA cold (the same 0%-CPU stall the CPU
# chain hit). init waits on it → the shards, which wait on init, start warm. DO_WARMUP=0 to skip.
warm_dep=""
if [[ "${DO_WARMUP:-1}" == "1" ]]; then
    wj=$(sub cf3warm_E${CF_ROUTINE} "${WARM_TIME:-01:00:00}" "${GPU_SB[@]}" \
        --export=ALL,CF_ROUTINE=${CF_ROUTINE},CF_STAGE=${CF_STAGE},R=${R},SEED=${SEED},CF_STEP=warmup,CF_GPU=1 submit_cf.sh)
    echo "  GPU warmup → ${wj} (${GPU_PARTITION}; init/shards wait on it)"
    warm_dep="--dependency=afterok:${wj}"
fi
j=$(sub cf3init_E${CF_ROUTINE} "${MERGE_TIME}" "${CPU_SB[@]}" ${warm_dep} \
    --export=ALL,${base},CF_STEP=cf3_init,CF_EXTRA="${eq}" submit_cf.sh)
echo "  init σ⁰ → ${j} (${CPU_PARTITION})"
prev="${j}"
for s in $(seq 1 "${N_SWEEPS}"); do
    arr=$(sub "cf3sw${s}_E${CF_ROUTINE}" "${SHARD_TIME}" --array=0-$((N_FIRM_SHARDS-1))${SHARD_THROTTLE:+%${SHARD_THROTTLE}} \
        "${GPU_SB[@]}" --dependency=afterok:"${prev}" \
        --export=ALL,${base},SWEEP=${s},CF_STEP=cf3_shard,CF_EXTRA="${eq}" submit_cf.sh)
    mrg=$(sub "cf3mg${s}_E${CF_ROUTINE}" "${MERGE_TIME}" "${CPU_SB[@]}" --dependency=afterok:"${arr}" \
        --export=ALL,${base},SWEEP=${s},CF_STEP=cf3_merge,CF_EXTRA="${eq}" submit_cf.sh)
    echo "  sweep ${s}: shard array ${arr} (GPU) → merge ${mrg} (CPU)"
    prev="${mrg}"
done
echo "Final equilibrium σ → ${SIGMA_DIR}/sig_${N_SWEEPS}.parquet (‖Δσ‖∞ per sweep in the cf3mg*.out logs)."

# Auto-zip cf3 outputs after the final sweep (skipped when a CF5/CF6 parent set ZIP_AFTER="").
if [[ "${DO_ZIP}" == "1" && -n "${ZIP_AFTER}" ]]; then
    zj=$(sub "cf_zip_${ZIP_AFTER}_E${CF_ROUTINE}" 00:30:00 "${CPU_SB[@]}" --mem=4G \
        --dependency=afterany:"${prev}" \
        --export=ALL,CF_STEP=zip,CF_WHICH=${ZIP_AFTER},CF_STAGE=${CF_STAGE},DATA_ROOT=${DATA_ROOT} submit_cf.sh)
    echo "  zip ${ZIP_AFTER} → ${zj} (afterany:${prev})"
fi
echo "FINAL_JOB=${prev}"
echo "FINAL_SIGMA=${SIGMA_DIR}/sig_${N_SWEEPS}.parquet"
echo "Watch: squeue -u \$USER"
