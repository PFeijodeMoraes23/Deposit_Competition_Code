#!/bin/bash
# ── DEFAULT RC-BLP cluster run ──────────────────────────────────────────────────
# Runs the four single-index routines — E5 (Single-Index), E6 (Single-Index+Time),
# E7 (Joint Single-Index), E8 (Joint Single-Index+Time), spec 12 — through the FULL
# extended random-coefficient
# sequence (sigma → rc2 → rc3 → rc4 → full → ext1 → ext2 → extended), submitted as
# RESUMABLE DEPENDENCY CHAINS so no single job hits the cluster wall-time limit.
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
# Engines: IFT (blp_2, headline) full chain + numerical (blp_1) cross-check at `extended`
# only (NUMERICAL_MODE=crosscheck), both by default. Routine set overridable via ROUTINES.
#   default (all, E5-E8)    → 4 routines × (8 IFT + 1 numerical xcheck) = 36 jobs
#   ENGINES="ift"           → 4 × 8 = 32 jobs
#   NUMERICAL_MODE=full     → 4 × 2 × 8 = 64 jobs
#   LAYOUT=grouped          → 4 × (3 IFT + 1 num) = 16 jobs
#
# Prereqs already on disk: data/input/ demand parquets + demographics_sigma.parquet,
# data/input/ logit_delta_E{5,6,7,8}_spec_12.bin (data/output kept as a fallback), and the R=2000 draws
# (run `sbatch submit_blp_1_draws.sh` first).
#
# Usage:  bash submit_blp_2_rc_default.sh
#         ENGINES="ift"  bash submit_blp_2_rc_default.sh
#         LAYOUT=grouped bash submit_blp_2_rc_default.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROUTINES="${ROUTINES:-1 2 5 6 7 8}"
export ENGINES="${ENGINES:-ift numerical}"
LAYOUT="${LAYOUT:-all}"

case "${LAYOUT}" in
    all)     DELEGATE="submit_blp_rc_all.sh" ;;
    grouped) DELEGATE="submit_blp_rc_grouped.sh" ;;
    *) echo "[!] LAYOUT must be 'all' or 'grouped' (got '${LAYOUT}')." >&2; exit 1 ;;
esac

echo "Default RC-BLP → ${LAYOUT} chains | routines=${ROUTINES} | engines=${ENGINES}"
exec bash "${HERE}/${DELEGATE}"
