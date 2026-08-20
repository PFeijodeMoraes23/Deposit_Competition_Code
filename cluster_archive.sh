#!/bin/bash
# ==============================================================================
# cluster_archive.sh — THE one archiver for all three stages.
#
#   bash cluster_archive.sh --set <name> (--copy | --move) \
#        [--newer <marker>] [--tag <t>] [--dry-run] [-h]
#   bash cluster_archive.sh --all --copy
#
# --copy / --move is REQUIRED. There is no default and no inference. That single
# decision replaces three divergent conventions: `zip -jm` MOVE, `zip -j` COPY,
# and add/verify/conditional-rm defaulting to MOVE-unless-KEEP=1. Under the old
# scheme the destructive behaviour was the DEFAULT and the safe behaviour needed
# an env var; here neither is reachable by omission.
#
# WHAT MUST EXIST FIRST: the outputs you want archived. Nothing else.
# WHAT TO RUN NEXT:      download the .zip from data/output.
#
# THIS SCRIPT IS NEW. zip_cf_outputs.sh / zip_all_cf.sh are still present and
# still work; they are the fallback and are retired only after one successful
# cluster cycle. Nothing in them has been modified.
#
# ALGORITHM (kept verbatim from zip_cf_outputs.sh, the only safe one of the
# three): never `zip -m`. Add with `zip -q`, read the archive back with
# `unzip -Z1`, and `rm -f` ONLY entries confirmed present, retrying up to 4
# passes and printing a WARNING naming anything it could not archive — those are
# never deleted. flock serialises concurrent appends into one archive. The empty
# default tag is kept, so repeated runs APPEND into ONE <base>.zip; a tag makes a
# separate snapshot. The trailing empty-dir sweep runs in --move mode only.
#
# FOUR INDEPENDENT MECHANISMS KEEP IT AWAY FROM data/input. All must pass:
#  (1) every pattern in the table below is RELATIVE and must start with the
#      domain prefix (output/ for data sets, logs/ for log sets). A pattern that
#      does not is a startup error, not a runtime surprise. The table is a
#      literal ~20-line block in ONE file, so "does any glob reach input?" is
#      answerable by reading it.
#  (2) after glob expansion — and again after the `find -type f` flatten of the
#      sigma directories — every resolved path goes through realpath and is
#      rejected unless it is under the domain root. This catches a symlink from
#      output/ into input/, which (1) cannot see.
#  (3) in --move mode only, a second explicit guard rejects the whole run if
#      ${DATA_ROOT}/input is a path prefix of ANY resolved path. Redundant with
#      (2) by construction; kept because the failure it prevents cost uploaded
#      phi^noPix and an emptied CF_FOUNDATION on 2026-08-01 and is unrecoverable
#      except by re-uploading over a manual browser session.
#  (4) the cf4 set globs cf4_pix_realloc_* ONLY — see the comment in the table.
#
# cf2 IS EXCLUDED from --all. Its glob owns psi_dev_E*_shard{i}of{N}.parquet, the
# product of a ~100-shard GPU array with no second copy, plus cost_params_*.json
# which four downstream CFs read in place. `--set cf2 --move` additionally
# requires ALLOW_CF2_MOVE=1.
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"
set -e

SET=""; MODE=""; NEWER=""; TAG=""; DRY=0; DO_ALL=0
SPEC="${SPEC:-12}"
ALL_SETS="${ALL_SETS:-foundation cf1 cf3 cf4 cf5 cf6}"   # cf2 deliberately absent

while [[ $# -gt 0 ]]; do
    case "$1" in
        --set)      SET="$2"; shift ;;
        --all)      DO_ALL=1 ;;
        --copy)     MODE=copy ;;
        --move)     MODE=move ;;
        --newer)    NEWER="$2"; shift ;;
        --tag)      TAG="$2"; shift ;;
        --dry-run)  DRY=1 ;;
        -h|--help)  sed -n '2,48p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

[[ -n "${MODE}" ]] || { echo "ERROR: one of --copy / --move is REQUIRED (there is no default)." >&2; exit 2; }
if [[ "${DO_ALL}" == "1" ]]; then
    [[ -z "${SET}" ]] || { echo "ERROR: --all and --set are mutually exclusive." >&2; exit 2; }
    rc=0
    for s in ${ALL_SETS}; do
        echo "-- ${s} --"
        bash "${BASH_SOURCE[0]}" --set "${s}" "--${MODE}" ${NEWER:+--newer "${NEWER}"} ${TAG:+--tag "${TAG}"} \
            $([[ "${DRY}" == "1" ]] && echo --dry-run) || { echo "  (${s}: skipped/failed — continuing)"; rc=1; }
    done
    echo "== done. Archives in ${CL_DATA_OUT} =="
    exit 0
fi
[[ -n "${SET}" ]] || { echo "ERROR: --set <name> is required (or --all). See -h." >&2; exit 2; }

OUT_DIR="${OUT_DIR:-${CL_DATA_OUT}}"

# ── THE PATTERN TABLE. Two domains, each with its own root and required prefix.
#    data  -> cd ${DATA_ROOT} ; every pattern must start with 'output/'
#    logs  -> cd ${CL_ROOT}   ; every pattern must start with 'logs/'
# Every entry is a QUOTED string: the '*' must survive to the expansion pass
# further down, which runs AFTER the cd into the domain root. Letting the shell
# glob here would resolve the patterns against the scripts dir and match nothing.
DOMAIN=data; BASE=""; pats=()
case "${SET}" in
  foundation)      BASE=foundation_outputs; pats=("output/cf/shares_elas_E*_spec_${SPEC}_*.parquet") ;;
  cf1)             BASE=cf1_outputs;        pats=("output/cf/cf1_franchise_*_E*_spec_${SPEC}*.parquet") ;;
  cf2)             BASE=cf2_outputs;        pats=("output/cost/psi_eq_E*_spec_${SPEC}_*" "output/cost/psi_dev_E*_spec_${SPEC}_*" "output/cost/cost_params_E*_spec_${SPEC}_*.json") ;;
  cf3)             BASE=cf3_outputs;        pats=("output/cf/cf3_equilibrium_E*_spec_${SPEC}_*.parquet" "output/cf/cf3_jacobi_E*") ;;
  # OUTPUTS ONLY. upsilon_pix_E*.json and phi_nopix_E*.parquet are UPLOADED
  # INPUTS (built locally by cf_4_upsilon_export.py from the sleep pickle, which
  # does not exist on the cluster) and live in data/input. They were once listed
  # in a cf4 pattern, so a default move-mode archive DELETED them — that is what
  # emptied CF_FOUNDATION on 2026-08-01. Never glob an input from this script.
  cf4)             BASE=cf4_outputs;        pats=("output/cf/cf4_pix_realloc_E*_spec_${SPEC}_*.parquet") ;;
  cf5)             BASE=cf5_outputs;        pats=("output/cf/cf5_passthrough_E*_spec_${SPEC}_*.parquet" "output/cf/cf5_E*") ;;
  cf6)             BASE=cf6_outputs;        pats=("output/cf/cf6_merger_E*_spec_${SPEC}_*.parquet" "output/cf/cf6_E*") ;;
  # The estimation sample the BLP, BBL and CF all read, produced on the cluster by the
  # sleepiness prep step into data/output/DEMAND_PREP (gate G2). Worth pulling down
  # mid-run: it is what every later result is read against.
  # COPY ONLY, in practice: pipeline_all.sh always passes --copy, because the whole
  # rest of the pipeline reads these parquets in place for days afterwards. The
  # pattern is confined to output/DEMAND_PREP, so guards (1)-(3) keep it inside the
  # output tree even if someone ever passes --move.
  demand_prep)     BASE=demand_prep_outputs; pats=("output/DEMAND_PREP/demand_*_spec_${SPEC}.parquet") ;;
  blp_outputs)     BASE=blp_outputs;        pats=("output/blp_results_E*_spec_${SPEC}_*.json" "output/blp_results_E*_spec_${SPEC}_*.jls" "output/blp_summary_E*_gpu_*.json" "output/logit_*") ;;
  blp_checkpoints) BASE=blp_checkpoints;    pats=("output/blp_checkpoint_E*_spec_*.jls") ;;
  bbl_costs)       BASE=bbl_outputs;        pats=("output/cost/cost_params_E*_spec_${SPEC}_*.json") ;;
  blp_logs)        BASE=blp_logs; DOMAIN=logs; pats=("logs/rc_*.out" "logs/rc_*.err" "logs/blp_build_sysimg_*.out" "logs/blp_build_sysimg_*.err" "logs/blp_draws_*.out" "logs/blp_draws_*.err" "logs/env_*.out" "logs/env_*.err") ;;
  bbl_logs)        BASE=bbl_logs; DOMAIN=logs; pats=("logs/bbl_*.out" "logs/bbl_*.err") ;;
  run_logs)        BASE=run_logs; DOMAIN=logs; pats=("logs/*.out" "logs/*.err") ;;
  *) echo "Unknown --set '${SET}'." >&2
     echo "  data sets: foundation cf1 cf2 cf3 cf4 cf5 cf6 demand_prep blp_outputs blp_checkpoints bbl_costs" >&2
     echo "  log  sets: blp_logs bbl_logs run_logs" >&2
     exit 2 ;;
esac

if [[ "${DOMAIN}" == "data" ]]; then ROOT="${CL_DATA_ROOT}"; PREFIX="output/"; else ROOT="${CL_ROOT}"; PREFIX="logs/"; fi
ROOT_ABS="$(cd "${ROOT}" && pwd)"
DOMAIN_ABS="$(cd "${ROOT_ABS}/${PREFIX%/}" 2>/dev/null && pwd || echo "${ROOT_ABS}/${PREFIX%/}")"
INPUT_ABS="${CL_DATA_ROOT}/input"

# ── Guard (1): every pattern must be relative and start with the domain prefix.
for p in "${pats[@]}"; do
    case "${p}" in
        /*) echo "STARTUP ERROR: absolute pattern in set '${SET}': ${p}" >&2; exit 2 ;;
        ${PREFIX}*) ;;
        *) echo "STARTUP ERROR: pattern in set '${SET}' does not start with '${PREFIX}': ${p}" >&2; exit 2 ;;
    esac
done

# cf2 move needs an explicit opt-in.
if [[ "${SET}" == "cf2" && "${MODE}" == "move" && "${ALLOW_CF2_MOVE:-0}" != "1" ]]; then
    echo "REFUSING: '--set cf2 --move' would take psi_dev_E*_shard{i}of{N}.parquet — a ~100-shard" >&2
    echo "  GPU array with no second copy — plus cost_params_*.json, which cf1_net/cf3/cf5/cf6 read" >&2
    echo "  IN PLACE. Use --copy, or set ALLOW_CF2_MOVE=1 if you really mean it." >&2
    exit 2
fi

# Expand the patterns INSIDE the domain root, with nullglob so an unmatched
# pattern vanishes instead of surviving as a literal.
cd "${ROOT_ABS}"
shopt -s nullglob
paths=()
for p in "${pats[@]}"; do
    for m in ${p}; do [[ -e "${m}" ]] && paths+=("${m}"); done
done
shopt -u nullglob
if [[ ${#paths[@]} -eq 0 ]]; then
    echo "${SET}: no matching items under ${ROOT_ABS} (spec ${SPEC}) — nothing to archive."
    exit 0
fi

# ── Flatten directories to leaf files so every entry can be CONFIRMED in the
#    archive before it is deleted, then apply the --newer scope filter.
files=()
for p in "${paths[@]}"; do
    if [[ -d "${p}" ]]; then while IFS= read -r f; do files+=("${f}"); done < <(find "${p}" -type f)
    else files+=("${p}"); fi
done

# run_logs must never take its own still-open .out/.err.
SELF_RE=""
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF_RE="${SLURM_JOB_ID}"

scoped=()
for f in "${files[@]}"; do
    [[ -n "${NEWER}" && ! "${f}" -nt "${NEWER}" ]] && continue
    [[ -n "${SELF_RE}" && "${f}" == *"${SELF_RE}"* ]] && continue
    scoped+=("${f}")
done
files=("${scoped[@]}")
if [[ ${#files[@]} -eq 0 ]]; then
    if [[ -n "${NEWER}" ]]; then
        echo "${SET}: ${#paths[@]} pattern match(es), none newer than $(basename "${NEWER}") — nothing to archive."
    else
        echo "${SET}: ${#paths[@]} pattern match(es) but they hold no files (empty directories) — nothing to archive."
    fi
    exit 0
fi

# ── Guards (2) and (3): resolve every path and prove where it lives.
for f in "${files[@]}"; do
    rp="$(realpath -m "${f}" 2>/dev/null || readlink -f "${f}" 2>/dev/null || echo "${ROOT_ABS}/${f}")"
    case "${rp}" in
        "${DOMAIN_ABS}"/*) ;;
        *) echo "REFUSING: '${f}' resolves to '${rp}', outside the ${SET} domain root ${DOMAIN_ABS}." >&2
           echo "  (a symlink out of the output tree is the usual cause)" >&2; exit 2 ;;
    esac
    if [[ "${MODE}" == "move" ]]; then
        case "${rp}" in
            "${INPUT_ABS}"/*|"${INPUT_ABS}")
                echo "REFUSING: move-mode run touches an UPLOADED INPUT: ${rp}" >&2
                echo "  data/input is uploaded by hand through the browser; deleting it is unrecoverable." >&2
                exit 2 ;;
        esac
    fi
done

ZIP="${OUT_DIR}/${BASE}${TAG:+_${TAG}}.zip"
echo "${SET}: ${#files[@]} file(s), mode=${MODE}${NEWER:+, scoped by --newer $(basename "${NEWER}")} -> ${ZIP}"
printf '  %s\n' "${files[@]}" | head -40
[[ ${#files[@]} -gt 40 ]] && echo "  ... and $(( ${#files[@]} - 40 )) more"
if [[ "${DRY}" == "1" ]]; then echo "(--dry-run — nothing added, nothing removed)"; exit 0; fi

mkdir -p "${OUT_DIR}"
_add () { zip -q "${ZIP}" "$@" 2>/dev/null || true; }
remaining=("${files[@]}"); tries=0
while ((${#remaining[@]})); do
    if command -v flock >/dev/null 2>&1; then ( flock 9; _add "${remaining[@]}" ) 9>"${OUT_DIR}/.${SET}.ziplock"
    else _add "${remaining[@]}"; fi
    declare -A have=(); while IFS= read -r n; do have["${n}"]=1; done < <(unzip -Z1 "${ZIP}" 2>/dev/null)
    ng=()
    for f in "${remaining[@]}"; do
        if [[ -n "${have[${f}]:-}" ]]; then
            [[ "${MODE}" == "move" ]] && rm -f "${f}"      # confirmed present in the archive
        else
            ng+=("${f}")                                    # missed -> retry
        fi
    done
    unset have; remaining=("${ng[@]}")
    ((${#remaining[@]} == 0)) && break
    ((++tries >= 4)) && break
done
if [[ "${MODE}" == "move" ]]; then
    for p in "${paths[@]}"; do [[ -d "${p}" ]] && find "${p}" -type d -empty -delete 2>/dev/null || true; done
fi
if ((${#remaining[@]})); then
    echo "WARNING: ${#remaining[@]} file(s) failed to archive after ${tries} retries — NOT deleted; re-run 'cluster_archive.sh --set ${SET} --${MODE}':" >&2
    printf '  %s\n' "${remaining[@]}" >&2
fi
if [[ "${MODE}" == "copy" ]]; then
    echo "Archived (COPY — every original kept on disk). $(unzip -l "${ZIP}" | tail -1)"
else
    echo "Archived + removed the originals. $(unzip -l "${ZIP}" | tail -1)"
fi
echo "-> ${ZIP}"
