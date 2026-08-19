#!/bin/bash
# ── DEFAULT BBL cost-estimation cluster run ─────────────────────────────────────
# Runs the four single-index routines — E5 (Single-Index), E6 (Single-Index+Time),
# E7 (Joint), E8 (Joint+Time), spec 12 — through the BBL Step-2 chain
# (fwd_sim ψ-deviation array → eq:17 solve), then auto-zips the cost_params.
# Sibling of submit_blp_2_rc_default.sh; delegates to submit_bbl_all.sh.
#
# Prereqs already on disk (see runbook §8):
#   data/output/BLP_DRAWS/halton_nu_R2000_seed42.jls    (run submit_blp_1_draws.sh first)
#   data/output/blp_outputs_*.zip  OR  cluster_processed/blp_E{3,4}_spec_12.jls (RC results)
#   data/input/forward_rf_qoq.csv                    (cf_forward_rf.py, local)
#   data/input/polfunc_fitted.csv        (estimation_bbl_1_polfunc.py, local)
#
# Usage:  bash submit_bbl_default.sh
#         ROUTINES="6" bash submit_bbl_default.sh         # E6 headline only
#         DO_POLFUNC=1 bash submit_bbl_default.sh         # also fit the policy on the cluster
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROUTINES="${ROUTINES:-5 6 7 8}"

echo "Default BBL cost estimation → fwd_sim array + eq:17 solve | routines=${ROUTINES}"
exec bash "${HERE}/submit_bbl_all.sh"
