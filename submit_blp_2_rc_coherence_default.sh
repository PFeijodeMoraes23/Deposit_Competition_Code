#!/bin/bash
# ── DEFAULT coherence RC-BLP cluster run ────────────────────────────────────────
# Runs the two headline sleepiness routines — E3 (Pooled Logistic) and E6 (Pooled
# Single-Index), spec 12 — through the FULL extended random-coefficient sequence
# (sigma → rc2 → rc3 → rc4 → full → ext1 → ext2 → extended), submitted as RESUMABLE
# DEPENDENCY CHAINS so no single job hits the cluster wall-time limit.
#
# LAYOUT controls how stages are packaged into jobs:
#   all     (default) — ONE job per stage = 8 jobs per routine×engine chain. Safest
#                       under the gpu_h200 2-day QOS wall: every stage gets its own
#                       ≤2-day budget, so no multi-stage bundle can ever blow the wall.
#   grouped           — sigma..ext1 bundled into one "head" job, then ext2, then
#                       extended = 3 jobs per chain. Fewer job startups, but the head
#                       bundle must fit a single 2-day wall.
# Both run identical computation; each stage warm-starts θ₂ from the prior stage's
# on-disk checkpoint, so a wall-kill is fully resumable on resubmit (the engine skips
# stages already completed).
#
# Engines: IFT (blp_2, headline) + numerical (blp_1) cross-check, both by default.
# Drop to IFT-only with  ENGINES="ift".  Routine set overridable via ROUTINES.
#   default (all, IFT+numerical, E3+E6) → 2 routines × 2 engines × 8 stages = 32 jobs
#   ENGINES="ift"                       → 2 × 1 × 8 = 16 jobs
#   LAYOUT=grouped                      → 2 × 2 × 3 = 12 jobs
#
# Prereqs already on disk: data/input/ demand parquets + demographics_sigma.parquet,
# data/output/ logit_delta_E{3,6}_spec_12_coherence.bin, and the R=2000 draws
# (run `sbatch submit_blp_1_draws.sh` first).
#
# Usage:  bash submit_blp_2_rc_coherence_default.sh
#         ENGINES="ift"  bash submit_blp_2_rc_coherence_default.sh
#         LAYOUT=grouped bash submit_blp_2_rc_coherence_default.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROUTINES="${ROUTINES:-3 6}"
export ENGINES="${ENGINES:-ift numerical}"
LAYOUT="${LAYOUT:-all}"

case "${LAYOUT}" in
    all)     DELEGATE="submit_blp_rc_coherence_all.sh" ;;
    grouped) DELEGATE="submit_blp_rc_coherence_grouped.sh" ;;
    *) echo "[!] LAYOUT must be 'all' or 'grouped' (got '${LAYOUT}')." >&2; exit 1 ;;
esac

echo "Default coherence RC-BLP → ${LAYOUT} chains | routines=${ROUTINES} | engines=${ENGINES}"
exec bash "${HERE}/${DELEGATE}"
