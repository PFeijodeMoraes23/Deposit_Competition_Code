#!/bin/bash
# ONE command: submit the FULL RC-BLP sweep as dependency CHAINS.
#
# For each routine in ${ROUTINES} × engine (blp_2 IFT, blp_1 numerical) we submit the
# 8 random-coefficient stages as SEPARATE jobs, each --dependency=afterok on the
# previous stage. That way:
#   • no single job has to run the whole sweep (avoids the wall-time ceiling), and
#   • each stage warm-starts θ₂ from the prior stage's checkpoint on disk.
#
#   sigma → rc2 → rc3 → rc4 → full → ext1 → ext2 → extended   (1→8 σ params)
#
# DEFAULT routine set is "5 6 7 8" (the single-index links + their +Time variants:
# E5 single-index, E6 single-index+time, E7 joint, E8 joint+time). The numerical engine
# runs as a CROSS-CHECK at `extended` only (NUMERICAL_MODE=crosscheck, seeded from the
# IFT optimum), not a full chain — so the default is per routine: 8 IFT stage-jobs + 1
# numerical extended-job = 9 jobs, × 4 routines = 36 jobs (vs 64 full). Engine per job
# (BLP_ENGINE); numerical outputs are tagged *_num so they never collide with IFT
# (un-suffixed). Set NUMERICAL_MODE=full for the full numerical chain.
#
# Prerequisite: warm-start deltas on the cluster:
#   data/output/logit_delta_E{k}_spec_12.bin  for each k in ${ROUTINES}
# (the sigma stage warns + falls back to log-share init if its delta is missing).
#
# Usage:  bash submit_blp_rc_all.sh
#         ENGINES="ift" bash submit_blp_rc_all.sh          # IFT only
#         NUMERICAL_MODE=full bash submit_blp_rc_all.sh    # full numerical chain
#         ROUTINES="1 2 3 4 5 6 7 8" bash submit_blp_rc_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/logs"

# Default routines: the single-index links + their +Time variants E5 (single-index),
# E6 (single-index+time), E7 (joint), E8 (joint+time).
ROUTINES="${ROUTINES:-5 6 7 8}"
# Engines: IFT (blp_2) + numerical (blp_1) cross-check. Override e.g. ENGINES="ift".
ENGINES="${ENGINES:-ift numerical}"
# How the numerical engine is run:
#   crosscheck (default) — ONE numerical job at `extended` only, afterok the IFT
#                          `extended` job and seeded from its θ₂ (BLP_THETA2_INIT_FILE).
#                          The cross-check's value is confirming the analytical gradient
#                          AT the optimum; running the full numerical chain just re-derives
#                          points IFT already found. Needs IFT in ENGINES.
#   full                 — the full 8-stage numerical chain (old behaviour).
NUMERICAL_MODE="${NUMERICAL_MODE:-crosscheck}"
DATA_OUT="${HERE}/../data/output"
STAGES=(sigma rc2 rc3 rc4 full ext1 ext2 extended)
GENERIC="${HERE}/submit_blp_rc_stage.sh"

# Stage-dependent wall-time (overrides #SBATCH --time in the stage script). Capped at
# the gpu_h200 QOS max wall-PER-JOB: 4-day requests are rejected (QOSMaxWallDuration-
# PerJobLimit), 2 days is accepted. Raise WALL_DEEP if your QOS allows more.
#   sigma..rc4  → 1 day   (1-param warm-start; historical convergence <6h)
#   full, ext1  → 2 days  (5-6 params; historical convergence <15h)
#   ext2/ext    → WALL_DEEP (default 2 days; IFT ~6h. Numerical may need a resubmit
#                 to resume if it wall-kills — per-stage checkpoints make that safe.)
WALL_DEEP="${WALL_DEEP:-2-00:00:00}"
stage_wall() {
    case "$1" in
        ext2|extended) echo "${WALL_DEEP}" ;;
        full|ext1)     echo "2-00:00:00" ;;
        *)             echo "1-00:00:00" ;;
    esac
}

# Submit one full 8-stage afterok chain for (routine, engine). Per-stage progress goes
# to stderr; the final (extended) job id is echoed to stdout for the caller to capture.
submit_chain () {   # $1=routine $2=engine $3=tag
    local k="$1" eng="$2" tag="$3" prev="" jid wall dep st
    for st in "${STAGES[@]}"; do
        dep=""; [ -n "${prev}" ] && dep="--dependency=afterok:${prev}"
        wall=$(stage_wall "${st}")
        jid=$(sbatch --parsable --time="${wall}" ${dep} \
            --export=ALL,RC_ROUTINE=${k},RC_ENGINE=${eng},RC_STAGE=${st} \
            -J "rc_${tag}_E${k}_${st}" \
            -o "${HERE}/logs/rc_${tag}_E${k}_${st}_%j.out" \
            -e "${HERE}/logs/rc_${tag}_E${k}_${st}_%j.err" \
            "${GENERIC}")
        echo "    ${st}: ${jid}  (${dep:+afterok ${prev}, }wall=${wall})" >&2
        prev="${jid}"
    done
    echo "${prev}"
}

# Which engines were requested?
do_ift=0; do_num=0
for e in ${ENGINES}; do
    [ "$e" = "ift" ] && do_ift=1
    [ "$e" = "numerical" ] && do_num=1
done

njobs=0
term_jids=""           # terminal job of every chain — the auto-zip waits on these
for k in ${ROUTINES}; do
    ift_ext_jid=""
    if [ "${do_ift}" = "1" ]; then
        echo "── chain: E${k} / ift ──"
        ift_ext_jid=$(submit_chain "${k}" "ift" "ift")
        njobs=$((njobs + 8))
        term_jids="${term_jids} ${ift_ext_jid}"
    fi
    if [ "${do_num}" = "1" ]; then
        if [ "${NUMERICAL_MODE}" = "crosscheck" ] && [ "${do_ift}" = "1" ]; then
            # One numerical job at `extended`, seeded from the IFT extended θ₂.
            echo "── numerical cross-check: E${k} / extended only (afterok IFT extended) ──"
            ckpt="${DATA_OUT}/blp_checkpoint_E${k}_spec_12_extended.jls"
            wall=$(stage_wall extended)
            jid=$(sbatch --parsable --time="${wall}" --dependency=afterok:${ift_ext_jid} \
                --export=ALL,RC_ROUTINE=${k},RC_ENGINE=numerical,RC_STAGE=extended,BLP_THETA2_INIT_FILE=${ckpt} \
                -J "rc_num_E${k}_xcheck" \
                -o "${HERE}/logs/rc_num_E${k}_xcheck_%j.out" \
                -e "${HERE}/logs/rc_num_E${k}_xcheck_%j.err" \
                "${GENERIC}")
            echo "    extended (xcheck): ${jid}  (afterok ${ift_ext_jid}, wall=${wall})"
            njobs=$((njobs + 1))
            term_jids="${term_jids} ${jid}"
        else
            # Full 8-stage numerical chain (NUMERICAL_MODE=full, or numerical without IFT).
            echo "── chain: E${k} / numerical (full) ──"
            num_ext_jid=$(submit_chain "${k}" "numerical" "num")
            njobs=$((njobs + 8))
            term_jids="${term_jids} ${num_ext_jid}"
        fi
    fi
done
echo "Submitted ${njobs} RC-BLP jobs (routines: ${ROUTINES}; engines: ${ENGINES}; numerical_mode: ${NUMERICAL_MODE})."

# ── auto-zip: one short CPU job, afterany ALL chains, bundles into data/output:
#     • blp_outputs_<thisjobid>.zip — the result/summary/logit files process_blp_outputs.py reads
#     • blp_logs_<thisjobid>.zip    — the per-stage SLURM .out/.err (for assessing run performance)
#    `afterany` (not afterok) so partial results + logs still get bundled if a stage wall-kills.
ZIP_PARTITION="${ZIP_PARTITION:-day}"   # any CPU partition; override if your cluster differs
dep_csv="$(echo ${term_jids} | tr ' ' ':' | sed 's/^://; s/:$//')"
if [ -n "${dep_csv}" ]; then
    zip_jid=$(sbatch --parsable --dependency=afterany:${dep_csv} \
        --job-name=blp_zip --partition="${ZIP_PARTITION}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${HERE}/logs/blp_zip_%j.out" -e "${HERE}/logs/blp_zip_%j.err" \
        --wrap "cd '${DATA_OUT}' && { \
                  zip -j \"blp_outputs_\${SLURM_JOB_ID}.zip\" \
                    blp_results_E*_spec_12_*.json blp_results_E*_spec_12_*.jls \
                    blp_summary_E*_gpu_*.json logit_* ; \
                  zip -j \"blp_logs_\${SLURM_JOB_ID}.zip\" \
                    '${HERE}'/logs/rc_*.out '${HERE}'/logs/rc_*.err ; \
                  echo \"wrote \${PWD}/blp_outputs_\${SLURM_JOB_ID}.zip + blp_logs_\${SLURM_JOB_ID}.zip\"; }")
    echo "── auto-zip: ${zip_jid}  (afterany ${term_jids# })"
    echo "   → ${DATA_OUT}/blp_outputs_<${zip_jid}>.zip  (results; process_blp_outputs.py auto-discovers blp_outputs_*.zip)"
    echo "   → ${DATA_OUT}/blp_logs_<${zip_jid}>.zip     (the rc_*.out/.err SLURM logs for performance review)"
fi
