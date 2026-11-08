#!/bin/bash
# ==============================================================================
# blp_ext1_adopt.sh — make one multi-start fit (blp_ext1_starts.sh) the production fit of its
# routine's rung: ext1 by default, ext2 or extended with EXT_STAGE. The production files of that
# routine and rung are moved to a dated backup folder first, then the sandbox's files are copied
# in; every file is hashed before and after.
#
# Run from scripts/ on the login node, once per routine (one line each):
#   EXT1_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_ext1_adopt.sh 3 rand1
#   EXT1_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_ext1_adopt.sh 4 rand1
#   EXT_STAGE=ext2 EXT_TAG=20261106 EXT_DIAG_ROOT=$HOME/scratch_pi_mf2263/pf382 bash blp_ext1_adopt.sh 4 x1zero
# EXT_STAGE, EXT_TAG (EXT1_TAG; ext1 defaults to 20261029) and EXT_DIAG_ROOT (EXT1_DIAG_ROOT) pick
# the multi-start run, as in blp_ext1_starts.sh.
#
# Refuses when a BLP or BBL job is live, when the sandbox result is missing or not converged, or
# when its standard errors were not computed (the BBL hand-off requires ext1's joint SEs).
# ext2 and extended also refuse a fit of another routine or rung, a fit without an FD check, and
# one whose FD check found a descent step (best dQ < -EXT_FD_TOL, default 0); they take exactly
# the rung's five files (results .jls/.json, checkpoint .jls/.bin, summary .json) and leave any
# _num/_cue files alone. A new ext2 leaves the production extended stale: it was fitted from the
# previous ext2.
# EXT_ALLOW_SE_REFUSED=1 (extended only) also accepts a fit whose standard errors the engine refused
# because the stage has more parameters than moments (se_status "refused: K > L: ..."). The engine
# states that refusal leaves the point estimates unaffected, and every routine's extended stage
# carries it, so a CF/BBL input that reads only the extended point estimates can still be adopted.
# Every other gate (converged, FD check, routine/rung, hashes, backup) stays.
# Undo: move the files of <backup>/ back into data/output/blp/.
# ==============================================================================
set -euo pipefail
K="${1:?usage: bash blp_ext1_adopt.sh <routine 3|4> <start, e.g. rand1>}"
ST="${2:?usage: bash blp_ext1_adopt.sh <routine 3|4> <start, e.g. rand1>}"
[[ "${K}" == "3" || "${K}" == "4" ]] || { echo "routine must be 3 or 4" >&2; exit 2; }
STAGE="${EXT_STAGE:-ext1}"
case "${STAGE}" in
    ext1|ext2|extended) ;;
    *) echo "EXT_STAGE='${STAGE}' must be ext1, ext2 or extended" >&2; exit 2 ;;
esac
if [[ "${STAGE}" == "ext1" ]]; then
    TAG="${EXT_TAG:-${EXT1_TAG:-20261029}}"
else
    TAG="${EXT_TAG:-${EXT1_TAG:-}}"
    [[ -n "${TAG}" ]] || { echo "EXT_STAGE=${STAGE} needs EXT_TAG (the run's name, e.g. 20261106)" >&2; exit 2; }
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${HERE}/.." && pwd)"
PROD="${PROJ}/data/output/blp"
SBB="${EXT_DIAG_ROOT:-${EXT1_DIAG_ROOT:-${PROJ}}}/diag_${STAGE}_${TAG}/E${K}_${ST}/data/output/blp"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="${PROD}/_superseded_${STAGE}_E${K}_${STAMP}"

[[ -d "${SBB}" ]] || { echo "REFUSING: no sandbox ${SBB}" >&2; exit 1; }
# A job running from a TR5 sandbox (tr5_variants.sh: <root>/tr5/<variant>/scripts) writes only to its
# own data/output, so it does not block the swap.
LIVE_RE='^(rc_|blp_|bbl_|stagec_|ext1s_)'
[[ "${STAGE}" == "ext1" ]] || LIVE_RE='^(rc_|blp_|bbl_|stagec_|ext1s_|ext2s_|extds_)'
live="$(squeue -h -u "${USER}" -o '%j|%Z' 2>/dev/null \
        | awk -F'|' -v re="${LIVE_RE}" '$1 ~ re && $2 !~ /\/tr5\// { print $1 }' || true)"
[[ -z "${live}" ]] || { echo "REFUSING: live jobs: $(tr '\n' ' ' <<< "${live}")" >&2; exit 1; }

if [[ "${STAGE}" == "ext1" ]]; then
python3 - "${SBB}/blp_results_E${K}_spec_12_ext1.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
ok = j.get("converged") is True and j.get("se_status") == "ok"
print(f"sandbox E{j.get('estim')} ext1: converged={j.get('converged')} se_status={j.get('se_status')} "
      f"Q={j.get('Q_value')} alpha={j['theta1'][0]:+.4f}")
sys.exit(0 if ok else 3)
PY
else
python3 - "${SBB}/blp_results_E${K}_spec_12_${STAGE}.json" "${STAGE}" "${K}" "${EXT_FD_TOL:-0}" "${EXT_ALLOW_SE_REFUSED:-0}" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
stage, k, tol, allow_kl = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5] == "1"
npar = {"ext2": 6, "extended": 7}[stage]
fd = j.get("fd_check") or {}
b = fd.get("best_dQ")
fd_ok = isinstance(b, (int, float)) and b >= -tol
same = j.get("stage") == stage and j.get("estim") == k and len(j.get("theta2") or []) == npar
se = str(j.get("se_status"))
kl_refused = allow_kl and stage == "extended" and se.startswith("refused: K > L:")
se_ok = se == "ok" or kl_refused
ok = j.get("converged") is True and se_ok and fd_ok and same
if kl_refused:
    print(f"  NOTE: se_status '{se}' accepted under EXT_ALLOW_SE_REFUSED=1 (point estimates only)")
print(f"sandbox E{j.get('estim')} {j.get('stage')}: converged={j.get('converged')} se_status={j.get('se_status')} "
      f"Q={j.get('Q_value')} alpha={j['theta1'][0]:+.4f} | FD norm {fd.get('grad_fd_projected_norm')} "
      f"(IFT {fd.get('grad_exact_projected_norm')}), best dQ {b} -> FD {'pass' if fd_ok else 'FAIL'}"
      + ("" if same else f" | NOT an E{k} {stage} fit with {npar} theta2 values"))
print("  theta2 = " + ", ".join(f"{x:+.6f}" for x in (j.get("theta2") or [])))
sys.exit(0 if ok else 3)
PY
fi

if [[ "${STAGE}" == "ext1" ]]; then
    PAT="E${K}.*ext1"
else
    PAT="^blp_(results|checkpoint)_E${K}_spec_12_${STAGE}[.](jls|json|bin)\$|^blp_summary_E${K}_${STAGE}_gpu_ift[.]json\$"
fi
mapfile -t NEW < <(cd "${SBB}" && ls -1 | grep -E "${PAT}" || true)
[[ ${#NEW[@]} -gt 0 ]] || { echo "REFUSING: no E${K} ${STAGE} files in ${SBB}" >&2; exit 1; }
if [[ "${STAGE}" != "ext1" && ${#NEW[@]} -ne 5 ]]; then
    echo "REFUSING: ${#NEW[@]} E${K} ${STAGE} files in ${SBB} (${NEW[*]}); a complete fit writes 5" >&2; exit 1
fi
mapfile -t OLD < <(cd "${PROD}" && ls -1 | grep -E "${PAT}" || true)
echo "sandbox files (${#NEW[@]}): ${NEW[*]}"
echo "production files replaced (${#OLD[@]}): ${OLD[*]:-none}"

mkdir -p "${BACKUP}"
( cd "${PROD}" && [[ ${#OLD[@]} -eq 0 ]] || sha256sum "${OLD[@]}" ) > "${BACKUP}/before.sha256"
for f in "${OLD[@]}"; do mv "${PROD}/${f}" "${BACKUP}/${f}"; done
for f in "${NEW[@]}"; do cp -p "${SBB}/${f}" "${PROD}/${f}"; done
( cd "${PROD}" && sha256sum "${NEW[@]}" ) > "${BACKUP}/after.sha256"
( cd "${SBB}" && sha256sum -c --quiet "${BACKUP}/after.sha256" ) \
    && echo "E${K}: production ${STAGE} is now the ${ST} fit (copies verified against the sandbox). Backup: ${BACKUP}"
if [[ "${STAGE}" == "ext2" ]]; then
    echo "NOTE: E${K}'s production extended was fitted from the previous ext2. Re-fit it (EXT_STAGE=extended) before any table or BBL run reads it."
fi
