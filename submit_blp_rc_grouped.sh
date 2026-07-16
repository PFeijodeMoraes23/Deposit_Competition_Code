#!/bin/bash
# ── Option-6 GROUPED RC-BLP submission ──────────────────────────────────────────
# Fewer, smarter jobs: run the CHEAP head stages in ONE process (amortising the
# Julia JIT + CUDA context + parquet-load startup that every separate job re-pays),
# while keeping the EXPENSIVE tail stages as their own resilient jobs.
#
# Partition is evidence-based, from the Jun-12/13 numerical runs:
#     full   9.2–15.3h   ✅ completed
#     ext1   3.9–4.8h    ✅ completed      → head = sigma..ext1 ≈ 15–25h  (fits a 2-day wall)
#     ext2   >48h        ❌ wall-killed     → ext2 keeps its own 4-day window
#     extended  (unmeasured; assume ext2-class)
#
#   HEAD : sigma+rc2+rc3+rc4+full+ext1   one job,  2-day wall
#   ext2 :                                one job,  4-day wall  (afterok HEAD)
#   ext  : extended                       one job,  4-day wall  (afterok ext2)
#
# DEFAULT routine set is "5 6 7 8" (the single-index links + their +Time variants:
# E5 single-index, E6 single-index+time, E7 joint, E8 joint+time); override with e.g.
# ROUTINES="1 2 3 4 5 6 7 8".
# numerical defaults to a cross-check at `extended` only (NUMERICAL_MODE). Per-stage
# checkpoints + the engine's skip-logic make a wall-killed HEAD fully resumable: just
# resubmit and it skips the stages already on disk.
#
# Stages are joined with '+' (NOT ',') because sbatch --export uses commas to
# separate variables; submit_blp_rc_stage.sh translates '+' → ',' before calling
# Julia (which runs a comma-separated stage subset in one process).
#
# CONCURRENCY: the routine chains are submitted independently (no cross-routine dependency, no
# shared mutable file — checkpoints are per-(routine,stage)), so all |ROUTINES| HEADs can run AT THE
# SAME TIME and the wall-clock is one chain (~1.5 h), not their sum (~5.9 h) — BUT only if the
# scheduler grants that many concurrent gpu_h200 jobs. If your QOS caps concurrent GPUs below
# |ROUTINES|, they serialise and you pay the sum. Check your limit with:
#     sacctmgr -n show qos gpu_h200 format=MaxJobsPU,MaxTRESPU%30    (or: MaxSubmitJobsPU)
# and, if needed, ask RC to raise it or stagger routine sets.
#
# Prerequisite (same as the per-stage orchestrator): warm-start deltas on the
# cluster — data/output/logit_delta_E{k}_spec_12.bin for each k in ${ROUTINES}.
#
# NOTE: validate once with a smoke test after OOD returns (e.g. submit only the E6
# numerical HEAD and confirm the startup GPU-vs-CPU guard passes) before the full set.
#
# Usage:  bash submit_blp_rc_grouped.sh
#         ROUTINES="1 2 3 4 5 6" bash submit_blp_rc_grouped.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/logs"

# Default routines: the single-index links + their +Time variants E5 (single-index),
# E6 (single-index+time), E7 (joint), E8 (joint+time).
ROUTINES="${ROUTINES:-5 6 7 8}"
# Engines: IFT (blp_2) by default. The numerical (blp_1) cross-check is the single most expensive
# block (~2.8 h across E5–E8) and only re-validates POINT estimates, which don't change run-to-run —
# so it is OFF by default. Run it ONCE after a code change, ideally on one representative routine:
#   ENGINES="ift numerical" ROUTINES=5 bash submit_blp_rc_grouped.sh
ENGINES="${ENGINES:-ift}"
# Numerical engine: crosscheck (default) = ONE job at `extended` only, afterok the IFT
# extended job and seeded from its θ₂; full = the 3-job grouped numerical chain.
NUMERICAL_MODE="${NUMERICAL_MODE:-crosscheck}"
DATA_OUT="${HERE}/../data/output"
# Per-job wall time. The gpu_h200 QOS caps wall-PER-JOB: 4-day requests are rejected
# with QOSMaxWallDurationPerJobLimit, and a 2-day job is accepted — so 2 days is the
# usable max. Override if your QOS allows more, e.g. WALL_DEEP="3-00:00:00".
WALL_HEAD="${WALL_HEAD:-2-00:00:00}"
WALL_DEEP="${WALL_DEEP:-2-00:00:00}"
HEAD_STAGES="sigma+rc2+rc3+rc4+full+ext1"   # '+'-joined; stage script → comma list
GENERIC="${HERE}/submit_blp_rc_stage.sh"

submit_one () {  # $1=routine $2=engine $3=tag $4=stage $5=wall $6=jobtag [$7=dep_jobid]
    local dep=""
    [ -n "${7:-}" ] && dep="--dependency=afterok:$7"
    sbatch --parsable --time="$5" ${dep} \
        --export=ALL,RC_ROUTINE=$1,RC_ENGINE=$2,RC_STAGE=$4 \
        -J "rcg_$3_E$1_$6" \
        -o "${HERE}/logs/rcg_$3_E$1_$6_%j.out" \
        -e "${HERE}/logs/rcg_$3_E$1_$6_%j.err" \
        "${GENERIC}"
}

# head → ext2 → extended for (routine, engine, tag). Per-stage progress → stderr;
# the extended job id → stdout for the caller to capture.
submit_grouped_chain () {  # $1=routine $2=engine $3=tag
    local k="$1" eng="$2" tag="$3" jh je2 je
    jh=$(submit_one "${k}" "${eng}" "${tag}" "${HEAD_STAGES}" "${WALL_HEAD}" "head")
    echo "    head (sigma..ext1): ${jh}  wall=${WALL_HEAD}" >&2
    je2=$(submit_one "${k}" "${eng}" "${tag}" "ext2" "${WALL_DEEP}" "ext2" "${jh}")
    echo "    ext2: ${je2}  (afterok ${jh}, wall=${WALL_DEEP})" >&2
    je=$(submit_one "${k}" "${eng}" "${tag}" "extended" "${WALL_DEEP}" "extended" "${je2}")
    echo "    extended: ${je}  (afterok ${je2}, wall=${WALL_DEEP})" >&2
    echo "${je}"
}

# Which engines were requested?
do_ift=0; do_num=0
for e in ${ENGINES}; do
    [ "$e" = "ift" ] && do_ift=1
    [ "$e" = "numerical" ] && do_num=1
done

# Concurrency check (non-fatal): the routine chains are independent, so wall-clock = one chain only
# if the QOS lets |ROUTINES| gpu_h200 jobs run at once. Report the cap so serialisation is visible.
nroutines=$(echo ${ROUTINES} | wc -w)
maxjobs="$(sacctmgr -n -P show qos gpu_h200 format=MaxJobsPU 2>/dev/null | head -1 || true)"
if [ -n "${maxjobs}" ] && [ "${maxjobs}" != "0" ] 2>/dev/null; then
    echo "[concurrency] ${nroutines} routine chains submitted; QOS gpu_h200 MaxJobsPU=${maxjobs}." \
         "$([ "${maxjobs}" -lt "${nroutines}" ] 2>/dev/null && echo '⚠ below routine count → chains will SERIALISE.' || echo 'chains can run in parallel.')"
else
    echo "[concurrency] ${nroutines} routine chains submitted; could not read QOS MaxJobsPU — verify" \
         "≥${nroutines} concurrent gpu_h200 jobs are allowed, else chains serialise (see header)."
fi

njobs=0
for k in ${ROUTINES}; do
    ift_ext_jid=""
    if [ "${do_ift}" = "1" ]; then
        echo "── grouped chain: E${k} / ift ──"
        ift_ext_jid=$(submit_grouped_chain "${k}" "ift" "ift")
        njobs=$((njobs + 3))
    fi
    if [ "${do_num}" = "1" ]; then
        if [ "${NUMERICAL_MODE}" = "crosscheck" ] && [ "${do_ift}" = "1" ]; then
            echo "── numerical cross-check: E${k} / extended only (afterok IFT extended) ──"
            ckpt="${DATA_OUT}/blp_checkpoint_E${k}_spec_12_extended.jls"
            jid=$(sbatch --parsable --time="${WALL_DEEP}" --dependency=afterok:${ift_ext_jid} \
                --export=ALL,RC_ROUTINE=${k},RC_ENGINE=numerical,RC_STAGE=extended,BLP_THETA2_INIT_FILE=${ckpt} \
                -J "rcg_num_E${k}_xcheck" \
                -o "${HERE}/logs/rcg_num_E${k}_xcheck_%j.out" \
                -e "${HERE}/logs/rcg_num_E${k}_xcheck_%j.err" \
                "${GENERIC}")
            echo "    extended (xcheck): ${jid}  (afterok ${ift_ext_jid}, wall=${WALL_DEEP})"
            njobs=$((njobs + 1))
        else
            echo "── grouped chain: E${k} / numerical (full) ──"
            submit_grouped_chain "${k}" "numerical" "num" >/dev/null
            njobs=$((njobs + 3))
        fi
    fi
done
echo "Submitted ${njobs} grouped RC-BLP jobs (routines: ${ROUTINES}; engines: ${ENGINES}; numerical_mode: ${NUMERICAL_MODE})."
