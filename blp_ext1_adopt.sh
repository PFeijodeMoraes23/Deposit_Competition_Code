#!/bin/bash
# ==============================================================================
# blp_ext1_adopt.sh — make one multi-start ext1 fit (blp_ext1_starts.sh) the production ext1 of
# its routine. The production ext1 files of that routine are moved to a dated backup folder first,
# then the sandbox's ext1 files are copied in; every file is hashed before and after.
#
# Run from scripts/ on the login node, once per routine (one line each):
#   EXT1_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_ext1_adopt.sh 3 rand1
#   EXT1_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_ext1_adopt.sh 4 rand1
# EXT1_TAG (default 20261029) picks the multi-start run, as in blp_ext1_starts.sh.
#
# Refuses when a BLP or BBL job is live, when the sandbox result is missing or not converged, or
# when its standard errors were not computed (the BBL hand-off requires ext1's joint SEs).
# Undo: move the files of <backup>/ back into data/output/blp/.
# ==============================================================================
set -euo pipefail
K="${1:?usage: bash blp_ext1_adopt.sh <routine 3|4> <start, e.g. rand1>}"
ST="${2:?usage: bash blp_ext1_adopt.sh <routine 3|4> <start, e.g. rand1>}"
[[ "${K}" == "3" || "${K}" == "4" ]] || { echo "routine must be 3 or 4" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${HERE}/.." && pwd)"
PROD="${PROJ}/data/output/blp"
SBB="${EXT1_DIAG_ROOT:-${PROJ}}/diag_ext1_${EXT1_TAG:-20261029}/E${K}_${ST}/data/output/blp"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="${PROD}/_superseded_ext1_E${K}_${STAMP}"

[[ -d "${SBB}" ]] || { echo "REFUSING: no sandbox ${SBB}" >&2; exit 1; }
# A job running from a TR5 sandbox (tr5_variants.sh: <root>/tr5/<variant>/scripts) writes only to its
# own data/output, so it does not block the swap.
live="$(squeue -h -u "${USER}" -o '%j|%Z' 2>/dev/null \
        | awk -F'|' '$1 ~ /^(rc_|blp_|bbl_|stagec_|ext1s_)/ && $2 !~ /\/tr5\// { print $1 }' || true)"
[[ -z "${live}" ]] || { echo "REFUSING: live jobs: $(tr '\n' ' ' <<< "${live}")" >&2; exit 1; }

python3 - "${SBB}/blp_results_E${K}_spec_12_ext1.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
ok = j.get("converged") is True and j.get("se_status") == "ok"
print(f"sandbox E{j.get('estim')} ext1: converged={j.get('converged')} se_status={j.get('se_status')} "
      f"Q={j.get('Q_value')} alpha={j['theta1'][0]:+.4f}")
sys.exit(0 if ok else 3)
PY

mapfile -t NEW < <(cd "${SBB}" && ls -1 | grep -E "E${K}.*ext1" || true)
[[ ${#NEW[@]} -gt 0 ]] || { echo "REFUSING: no E${K} ext1 files in ${SBB}" >&2; exit 1; }
mapfile -t OLD < <(cd "${PROD}" && ls -1 | grep -E "E${K}.*ext1" || true)
echo "sandbox files (${#NEW[@]}): ${NEW[*]}"
echo "production files replaced (${#OLD[@]}): ${OLD[*]:-none}"

mkdir -p "${BACKUP}"
( cd "${PROD}" && [[ ${#OLD[@]} -eq 0 ]] || sha256sum "${OLD[@]}" ) > "${BACKUP}/before.sha256"
for f in "${OLD[@]}"; do mv "${PROD}/${f}" "${BACKUP}/${f}"; done
for f in "${NEW[@]}"; do cp -p "${SBB}/${f}" "${PROD}/${f}"; done
( cd "${PROD}" && sha256sum "${NEW[@]}" ) > "${BACKUP}/after.sha256"
( cd "${SBB}" && sha256sum -c --quiet "${BACKUP}/after.sha256" ) \
    && echo "E${K}: production ext1 is now the ${ST} fit (copies verified against the sandbox). Backup: ${BACKUP}"
