#!/bin/bash
# ONE command: submit the FULL coherence RC-BLP sweep as dependency CHAINS.
#
# For each routine (E1/E2/E3) × engine (blp_2 IFT, blp_1 numerical) we submit the
# 8 random-coefficient stages as SEPARATE jobs, each --dependency=afterok on the
# previous stage. That way:
#   • no single job has to run the whole sweep (avoids the wall-time ceiling), and
#   • each stage warm-starts θ₂ from the prior stage's checkpoint on disk.
#
#   sigma → rc2 → rc3 → rc4 → full → ext1 → ext2 → extended   (1→8 σ params)
#
# 6 chains × 8 stages = 48 jobs; at most 6 run at once (one per routine×engine).
# Engine is exported per job (BLP_COHERENCE_ENGINE); blp_2_rc_coherence.jl tags
# numerical outputs *_coherence_num so blp_1 and blp_2 results never collide.
#
# Prerequisite: warm-start deltas on the cluster:
#   data/output/logit_delta_E{1,2,3}_spec_12_coherence.bin
# (the sigma stage warns + falls back to log-share init if its delta is missing).
#
# Usage:  bash submit_blp_2_rc_coherence_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/logs"

STAGES=(sigma rc2 rc3 rc4 full ext1 ext2 extended)
GENERIC="${HERE}/submit_blp_rc_coherence_stage.sh"

# Stage-dependent wall-time (overrides #SBATCH --time in the stage script):
#   sigma..rc4  → 1 day   (1-param warm-start; historical convergence <6h)
#   full, ext1  → 2 days  (5-6 params; historical convergence <15h)
#   ext2        → 4 days  (7 params; numerical engine took >48h; IFT ~6h)
#   extended    → 4 days  (8 params; most expensive stage)
stage_wall() {
    case "$1" in
        ext2|extended) echo "4-00:00:00" ;;
        full|ext1)     echo "2-00:00:00" ;;
        *)             echo "1-00:00:00" ;;
    esac
}

for k in 1 2 3; do
    for eng in ift numerical; do
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
echo "Submitted 6 chains × 8 stages = 48 coherence RC-BLP jobs."
