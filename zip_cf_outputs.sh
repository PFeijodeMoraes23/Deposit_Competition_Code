#!/bin/bash
# zip_cf_outputs.sh — bundle ONE counterfactual's outputs (across ALL estimations E5–E8) into a
# single archive, MOVING the files out of the working dirs (zip -m), mirroring the BLP auto-zip.
# Run it after that CF's runs finish for whichever routines you ran.
#
#   bash zip_cf_outputs.sh <cf> [tag]
#     <cf> ∈ foundation | cf1 | cf2 | cf3 | cf4 | cf5 | cf6
#     [tag]  optional label. DEFAULT is EMPTY → the archive is <cf>_outputs.zip, so every
#            estimation/run of that CF APPENDS into ONE archive (zip -m updates same-named
#            entries). Pass a tag only to keep a separate snapshot (<cf>_outputs_<tag>.zip).
#   Env: SPEC=12  DATA_ROOT=<...>/data  OUT_DIR=<...>/output  DRYRUN=1 (list only, don't zip/move)
#        KEEP=1 → COPY mode: archive + verify but do NOT delete the originals. Use this to make a
#        downloadable bundle while the pipeline is still running — the equilibrium CFs (CF3/CF5/CF6)
#        still need cost_params_E*.json and CF3's sig_*.parquet on disk. Move mode is only safe once
#        the whole chain is finished (that's what zip_all_cf.sh does at the end).
#
# Archive → ${OUT_DIR}/<cf>_outputs[_<tag>].zip, paths relative to data/ (output/cf/…,
# output/cost/…). Concurrent appends (several estimations' zip jobs → one archive) are serialized
# with flock. DRYRUN=1 previews exactly what would be moved without touching anything.
set -euo pipefail
CF="${1:?usage: zip_cf_outputs.sh <foundation|cf1|cf2|cf3|cf4|cf5|cf6> [tag]}"
TAG="${2:-}"
SPEC="${SPEC:-12}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/../data}"
DATA_ROOT="$(cd "${DATA_ROOT}" 2>/dev/null && pwd)" || { echo "DATA_ROOT not found: ${DATA_ROOT:-?}"; exit 1; }
# Cluster layout (2026-08-02): archives land in data/output alongside what they bundle; the separate
# CF_ZIPS/ tree is retired. CFF/COST are the CF + BBL-cost output dirs, matching cf_out_dir() in
# foundation_demand_eval.jl. Paths stay relative to DATA_ROOT so `unzip <zip> -d data/` restores in place.
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/output}"
CFF="output/cf"; COST="output/cost"           # relative to DATA_ROOT

# Build the pattern list FROM INSIDE data/ (so the globs resolve there) with nullglob on (an
# unmatched pattern expands to nothing rather than a literal). Dirs (cf3_jacobi_E*, cf5_E*,
# cf6_E*) hold the sharded-Jacobi σ files and are archived whole.
cd "${DATA_ROOT}"
shopt -s nullglob
case "${CF}" in
  foundation) pats=("${CFF}/shares_elas_E"*"_spec_${SPEC}_"*.parquet) ;;
  cf1)        pats=("${CFF}/cf1_franchise_"*"_E"*"_spec_${SPEC}"*.parquet) ;;
  cf2)        pats=("${COST}/psi_eq_E"*"_spec_${SPEC}_"* "${COST}/psi_dev_E"*"_spec_${SPEC}_"* "${COST}/cost_params_E"*"_spec_${SPEC}_"*.json) ;;
  cf3)        pats=("${CFF}/cf3_equilibrium_E"*"_spec_${SPEC}_"*.parquet "${CFF}/cf3_jacobi_E"*) ;;
  # OUTPUTS ONLY. upsilon_pix_E*.json and phi_nopix_E*.parquet are UPLOADED INPUTS (built locally by
  # cf_4_upsilon_export.py from the sleep pickle, which does not exist on the cluster) and now live in
  # data/input. They were previously listed here, so every default (move-mode) archive DELETED them —
  # that is what emptied CF_FOUNDATION on 2026-08-01. Never glob an input from this script.
  cf4)        pats=("${CFF}/cf4_pix_realloc_E"*"_spec_${SPEC}_"*.parquet) ;;
  cf5)        pats=("${CFF}/cf5_passthrough_E"*"_spec_${SPEC}_"*.parquet "${CFF}/cf5_E"*) ;;
  cf6)        pats=("${CFF}/cf6_merger_E"*"_spec_${SPEC}_"*.parquet "${CFF}/cf6_E"*) ;;
  *) echo "Unknown CF '${CF}' (expected foundation|cf1|cf2|cf3|cf4|cf5|cf6)"; exit 1 ;;
esac
paths=(); for p in "${pats[@]}"; do [[ -e "${p}" ]] && paths+=("${p}"); done
shopt -u nullglob
if [[ ${#paths[@]} -eq 0 ]]; then
    echo "No ${CF} outputs found under ${DATA_ROOT} (spec ${SPEC}) — nothing to zip."; exit 0
fi

ZIP="${OUT_DIR}/${CF}_outputs${TAG:+_${TAG}}.zip"
echo "${CF}: ${#paths[@]} item(s) → ${ZIP}"
printf '  %s\n' "${paths[@]}"
if [[ "${DRYRUN:-0}" == "1" ]]; then echo "(DRYRUN — nothing moved)"; exit 0; fi
mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
# Flatten to leaf files (expand the σ dirs) so each entry can be CONFIRMED in the archive before it
# is deleted. We deliberately DON'T use `zip -m` (move): some Info-ZIP builds flakily skip an entry
# in move mode; plain add is reliable. So: add → verify via `unzip -Z1` → delete only what landed.
# This never deletes an unarchived file. flock serializes concurrent appends to one <cf>_outputs.zip.
files=(); for p in "${paths[@]}"; do
    if [[ -d "${p}" ]]; then while IFS= read -r f; do files+=("${f}"); done < <(find "${p}" -type f)
    else files+=("${p}"); fi
done
_add () { zip -q "${ZIP}" "$@" 2>/dev/null || true; }
# Add → confirm-in-archive → delete only what landed; re-add anything skipped (a zip build can drop
# an entry that is momentarily read-locked). On a settled tree the first pass takes everything.
KEEP="${KEEP:-0}"          # 1 = copy mode: archive + verify, keep the originals on disk
remaining=("${files[@]}"); tries=0
while ((${#remaining[@]})); do
    if command -v flock >/dev/null 2>&1; then ( flock 9; _add "${remaining[@]}" ) 9>"${OUT_DIR}/.${CF}.ziplock"
    else _add "${remaining[@]}"; fi
    declare -A have=(); while IFS= read -r n; do have["${n}"]=1; done < <(unzip -Z1 "${ZIP}" 2>/dev/null)
    ng=(); for f in "${remaining[@]}"; do
        if [[ -n "${have[${f}]:-}" ]]; then [[ "${KEEP}" == "1" ]] || rm -f "${f}"   # confirmed in archive
        else ng+=("${f}"); fi                                                        # missed → retry
    done
    unset have; remaining=("${ng[@]}")
    ((${#remaining[@]} == 0)) && break
    ((++tries >= 4)) && break
done
[[ "${KEEP}" == "1" ]] || for p in "${paths[@]}"; do [[ -d "${p}" ]] && find "${p}" -type d -empty -delete 2>/dev/null || true; done
if ((${#remaining[@]})); then
    echo "WARNING: ${#remaining[@]} file(s) failed to archive after ${tries} retries — NOT deleted; re-run 'zip_cf_outputs.sh ${CF}':" >&2
    printf '  %s\n' "${remaining[@]}" >&2
fi
if [[ "${KEEP}" == "1" ]]; then echo "Archived (COPY — originals kept on disk). $(unzip -l "${ZIP}" | tail -1)"
else echo "Archived + removed originals. $(unzip -l "${ZIP}" | tail -1)"; fi
