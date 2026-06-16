#!/bin/bash
# ── Option-6 GROUPED coherence RC-BLP submission ────────────────────────────────
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
# 6 chains (E1/E2/E3 × ift/numerical) × 3 jobs = 18 jobs  (vs 48 in the per-stage
# orchestrator). Per-stage checkpoints + the engine's skip-logic make a wall-killed
# HEAD fully resumable: just resubmit and it skips the stages already on disk.
#
# Stages are joined with '+' (NOT ',') because sbatch --export uses commas to
# separate variables; submit_blp_rc_coherence_stage.sh translates '+' → ',' before
# calling Julia (which runs a comma-separated stage subset in one process).
#
# Prerequisite (same as the per-stage orchestrator): warm-start deltas on the
# cluster — data/output/logit_delta_E{1,2,3}_spec_12_coherence.bin.
#
# NOTE: validate once with a smoke test after OOD returns (e.g. submit only the E1
# numerical HEAD and confirm the startup GPU-vs-CPU guard passes) before the full set.
#
# Usage:  bash submit_blp_rc_coherence_grouped.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/logs"

HEAD_STAGES="sigma+rc2+rc3+rc4+full+ext1"   # '+'-joined; stage script → comma list
GENERIC="${HERE}/submit_blp_rc_coherence_stage.sh"

submit_one () {  # $1=routine $2=engine $3=tag $4=stage $5=wall $6=jobtag [$7=dep_jobid]
    local dep=""
    [ -n "${7:-}" ] && dep="--dependency=afterok:$7"
    sbatch --parsable --time="$5" ${dep} \
        --export=ALL,COH_ROUTINE=$1,COH_ENGINE=$2,COH_STAGE=$4 \
        -J "cohg_$3_E$1_$6" \
        -o "${HERE}/logs/cohg_$3_E$1_$6_%j.out" \
        -e "${HERE}/logs/cohg_$3_E$1_$6_%j.err" \
        "${GENERIC}"
}

for k in 1 2 3; do
    for eng in ift numerical; do
        if [ "${eng}" = "numerical" ]; then tag="num"; else tag="ift"; fi
        echo "── grouped chain: E${k} / ${eng} ──"

        jid_head=$(submit_one "${k}" "${eng}" "${tag}" "${HEAD_STAGES}" "2-00:00:00" "head")
        echo "    head (sigma..ext1): ${jid_head}  wall=2d"

        jid_ext2=$(submit_one "${k}" "${eng}" "${tag}" "ext2" "4-00:00:00" "ext2" "${jid_head}")
        echo "    ext2: ${jid_ext2}  (afterok ${jid_head}, wall=4d)"

        jid_ext=$(submit_one "${k}" "${eng}" "${tag}" "extended" "4-00:00:00" "extended" "${jid_ext2}")
        echo "    extended: ${jid_ext}  (afterok ${jid_ext2}, wall=4d)"
    done
done
echo "Submitted 6 grouped chains × 3 jobs = 18 coherence RC-BLP jobs (was 48)."
