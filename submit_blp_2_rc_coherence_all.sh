#!/bin/bash
# ONE command to submit ALL coherence RC-BLP cluster jobs:
#   E{1,2,3} × {blp_2 IFT, blp_1 numerical} = 6 independent parallel H200 jobs.
#
# The estimation engine is selected per job via BLP_COHERENCE_ENGINE, exported into
# the job environment; the orchestrator (blp_2_rc_coherence.jl) reads it and tags
# numerical-engine outputs *_coherence_num so the blp_1 and blp_2 results never
# collide (IFT → *_coherence, numerical → *_coherence_num). Job name / log paths are
# overridden per engine so each run gets its own .out/.err.
#
# Prerequisite: the logit-coherence step wrote the warm-start deltas
#   data/output/logit_delta_E{1,2,3}_spec_12_coherence.bin
# (each job warns and falls back to log-share init if its delta is missing).
#
# Usage:  bash submit_blp_2_rc_coherence_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/logs"

for k in 1 2 3; do
    for eng in ift numerical; do
        if [ "${eng}" = "numerical" ]; then tag="num"; blp="blp_1"; else tag="ift"; blp="blp_2"; fi
        jid=$(sbatch --parsable \
            --export=ALL,BLP_COHERENCE_ENGINE="${eng}" \
            -J "blp_rc_coh_${tag}_E${k}" \
            -o "${HERE}/logs/blp_rc_coh_${tag}_E${k}_%j.out" \
            -e "${HERE}/logs/blp_rc_coh_${tag}_E${k}_%j.err" \
            "${HERE}/submit_blp_2_rc_coherence_E${k}.sh")
        echo "Submitted coherence ${blp} (${eng}) E${k}: job ${jid}"
    done
done
echo "Submitted 6 coherence RC-BLP jobs (E1/E2/E3 × {IFT, numerical})."
