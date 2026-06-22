#!/bin/bash
# ONE command: submit the FULL coherence RC-BLP sweep as dependency CHAINS.
#
# For each routine in ${ROUTINES} × engine (blp_2 IFT, blp_1 numerical) we submit the
# 8 random-coefficient stages as SEPARATE jobs, each --dependency=afterok on the
# previous stage. That way:
#   • no single job has to run the whole sweep (avoids the wall-time ceiling), and
#   • each stage warm-starts θ₂ from the prior stage's checkpoint on disk.
#
#   sigma → rc2 → rc3 → rc4 → full → ext1 → ext2 → extended   (1→8 σ params)
#
# DEFAULT routine set is "3 6" (the two headline sleepiness links E3 logistic + E6
# index). Override with e.g. ROUTINES="1 2 3 4 5 6". With the 2 default routines:
# 4 chains × 8 stages = 32 jobs; at most 4 run at once (one per routine×engine).
# Engine is exported per job (BLP_COHERENCE_ENGINE); blp_2_rc_coherence.jl tags
# numerical outputs *_coherence_num so blp_1 and blp_2 results never collide.
#
# Prerequisite: warm-start deltas on the cluster:
#   data/output/logit_delta_E{k}_spec_12_coherence.bin  for each k in ${ROUTINES}
# (the sigma stage warns + falls back to log-share init if its delta is missing).
#
# Usage:  bash submit_blp_rc_coherence_all.sh
#         ROUTINES="1 2 3 4 5 6" bash submit_blp_rc_coherence_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/logs"

# Default headline routines: E3 (logistic) + E6 (single-index). Override via env.
ROUTINES="${ROUTINES:-3 6}"
# Engines: IFT (blp_2) + numerical (blp_1) cross-check. Override e.g. ENGINES="ift".
ENGINES="${ENGINES:-ift numerical}"
STAGES=(sigma rc2 rc3 rc4 full ext1 ext2 extended)
GENERIC="${HERE}/submit_blp_rc_coherence_stage.sh"

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

for k in ${ROUTINES}; do
    for eng in ${ENGINES}; do
        if [ "${eng}" = "numerical" ]; then tag="num"; else tag="ift"; fi
        echo "── chain: E${k} / ${eng} ──"
        prev=""
        for st in "${STAGES[@]}"; do
            dep=""
            [ -n "${prev}" ] && dep="--dependency=afterok:${prev}"
            wall=$(stage_wall "${st}")
            jid=$(sbatch --parsable --time="${wall}" ${dep} \
                --export=ALL,COH_ROUTINE=${k},COH_ENGINE=${eng},COH_STAGE=${st} \
                -J "coh_${tag}_E${k}_${st}" \
                -o "${HERE}/logs/coh_${tag}_E${k}_${st}_%j.out" \
                -e "${HERE}/logs/coh_${tag}_E${k}_${st}_%j.err" \
                "${GENERIC}")
            if [ -n "${prev}" ]; then
                echo "    ${st}: ${jid}  (afterok ${prev}, wall=${wall})"
            else
                echo "    ${st}: ${jid}  (head of chain, wall=${wall})"
            fi
            prev="${jid}"
        done
    done
done
n_routines=$(echo ${ROUTINES} | wc -w); n_engines=$(echo ${ENGINES} | wc -w)
echo "Submitted $((n_routines * n_engines)) chains × 8 stages = $((n_routines * n_engines * 8)) coherence RC-BLP jobs (routines: ${ROUTINES}; engines: ${ENGINES})."
