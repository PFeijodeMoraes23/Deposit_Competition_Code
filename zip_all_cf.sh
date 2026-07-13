#!/bin/bash
# zip_all_cf.sh — archive EVERY counterfactual's outputs, ONCE, at the very end.
#
# Run this only after the whole chain is done (costs + CF1 + the equilibrium CFs CF3/CF5/CF6).
# It MOVES outputs into data/CF_ZIPS/<cf>_outputs.zip, so running it earlier would strand inputs
# that later CFs still need (cost_params → CF3/CF5/CF6; CF3's sig_6 → CF5 base). That's exactly why
# auto-zip was removed from submit_cf_all.sh — archiving is a deliberate final act, done here.
#
# Zipping is light I/O — just run it on the login node:
#   bash zip_all_cf.sh                 # archive all CFs that have outputs on disk
#   CFS="cf2 cf3 cf5" bash zip_all_cf.sh   # only these
#   DRYRUN=1 bash zip_all_cf.sh        # preview what each would move, touch nothing
#
# Missing CFs are skipped quietly (zip_cf_outputs.sh prints "nothing to zip"). Safe to re-run:
# a second pass just appends any newly-produced files into the same archives.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFS="${CFS:-foundation cf1 cf2 cf3 cf4 cf5 cf6}"
echo "== zip_all_cf: archiving [${CFS}] → data/CF_ZIPS/ =="
for cf in ${CFS}; do
    echo "── ${cf} ──"
    DRYRUN="${DRYRUN:-0}" bash "${HERE}/zip_cf_outputs.sh" "${cf}" || echo "  (${cf}: skipped/failed — continuing)"
done
echo "== done. Archives in data/CF_ZIPS/ =="