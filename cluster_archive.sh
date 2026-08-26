#!/bin/bash
# ==============================================================================
# cluster_archive.sh — THE one packager. It turns a finished pipeline step into
# download-ready zips in data/output/download, and does nothing else.
#
#   bash cluster_archive.sh --set <name> (--copy | --move) \
#        [--newer <marker>] [--tag <t>] [--dry-run] [-h]
#   bash cluster_archive.sh --all --copy
#
# THE EIGHT SETS — one per step folder of the output tree, plus gates and logs:
#
#   sleep            output/sleep/est*, Rout, DIAGNOSTICS, top-level *.csv / *.json
#   demand_prep      output/demand_prep         (whole folder)
#   logit            output/logit               (whole folder)
#   blp              output/blp/*.jls, *.json + draws/halton_nu_*, demo_key_index_*,
#                    draws/*.json ; INCLUDE_DRAWS=1 adds the ~14 GB demo_draws_*
#   bbl              output/bbl                 (whole folder; INCLUDE_PSI=0 drops psi_*)
#   counterfactuals  output/counterfactuals     (whole folder)
#   gates            output/.gate_*.json
#   logs             logs/*.out, *.err, orchestrator_*.log
#
# The eight together are the WHOLE run: the cluster is scratch and the local
# machine is the system of record, so a set is defined by "everything this step
# produced", not by the few files a later cluster phase happens to read.
#
# ONE DIRECTION. Archives are for DOWNLOAD. Nothing here, and nothing else in the
# cluster scripts, ever unzips an archive back into the tree — every consumer
# reads the step folder its producer wrote, so a zip can never become a second,
# older copy of a family that some consumer resolves instead.
#
# WHAT MUST EXIST FIRST: the outputs you want archived. Nothing else.
# WHAT TO RUN NEXT:      download data/output/download/ whole, then
#                        `sha256sum -c sha256SUMS` inside it on the local side.
#
# --copy / --move is REQUIRED. There is no default and no inference; neither
# behaviour is reachable by omission. Every caller in the pipeline passes --copy,
# because the step folders stay readable for the rest of the run and the download
# is meant to be complete rather than incremental.
#
# ALGORITHM — never `zip -m`. Add with `zip -q`, read the archive back with
# `unzip -Z1`, and `rm -f` ONLY entries confirmed present, retrying up to 4
# passes and printing a WARNING naming anything it could not archive — those are
# never deleted. flock serialises concurrent appends into one archive and into
# sha256SUMS. The empty default tag is kept, so repeated runs APPEND into ONE
# <base>.zip; a tag makes a separate snapshot. The trailing empty-dir sweep runs
# in --move mode only.
#
# PACKAGING — three decisions, all about the browser download at the far end:
#   compression  `zip -n <suffixes>`: float payloads are STORED, text DEFLATEd.
#   splitting    anything over ${SPLIT_BYTES:-3000000000} becomes .part.NN of
#                ${CHUNK:-2G} plus a .sha256 of the whole zip. A browser download
#                has no resume, so one 10 GB file is one point of failure.
#   checksums    every emitted file is appended to ONE download/sha256SUMS.
#
# FIVE INDEPENDENT MECHANISMS KEEP IT AWAY FROM THE WRONG FILES. All must pass:
#  (1) every pattern in the table below is RELATIVE and must start with the
#      domain prefix (output/ for data sets, logs/ for log sets). A pattern that
#      does not is a startup error, not a runtime surprise. The table is a
#      literal ~20-line block in ONE file, so "does any glob reach input?" is
#      answerable by reading it.
#  (2) after glob expansion — and again after the `find -type f` flatten of the
#      step directories — every resolved path goes through realpath and is
#      rejected unless it is under the domain root. This catches a symlink from
#      output/ into input/, which (1) cannot see.
#  (3) in --move mode only, a second explicit guard rejects the whole run if
#      ${DATA_ROOT}/input is a path prefix of ANY resolved path. Redundant with
#      (2) by construction; kept because the failure it prevents cost uploaded
#      phi^noPix and an emptied CF_FOUNDATION on 2026-08-01 and is unrecoverable
#      except by re-uploading over a manual browser session.
#  (4) no pattern globs data/input at all: everything the sets name is
#      cluster-produced. The uploaded inputs (market_panel, demographics_sigma,
#      forward_rf_qoq, the state-centering pair, …) are never archived, because
#      the local machine already has them — they came from there.
#  (5) a resolved path under data/output/download is refused. The download folder
#      lives INSIDE the archived domain, so a whole-folder set or a future glob
#      could otherwise swallow the archives themselves and recurse.
#
# The *_ABS values below are PHYSICAL (realpath), not logical: reached through a
# symlinked home (~/project_pi_... -> /nfs/roberts/project/...) a logical root
# cannot contain the realpath'd files it is compared against, and every archive
# would refuse — or, worse for (3) and (5), silently never match.
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"
set -e

SET=""; MODE=""; NEWER=""; TAG=""; DRY=0; DO_ALL=0
ALL_SETS="${ALL_SETS:-sleep demand_prep logit blp bbl counterfactuals gates logs}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --set)      SET="$2"; shift ;;
        --all)      DO_ALL=1 ;;
        --copy)     MODE=copy ;;
        --move)     MODE=move ;;
        --newer)    NEWER="$2"; shift ;;
        --tag)      TAG="$2"; shift ;;
        --dry-run)  DRY=1 ;;
        -h|--help)  sed -n '2,81p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) cl_err "unknown option: $1 (see -h)"; exit 2 ;;
    esac
    shift
done

# Every zip, part and checksum lands in the download folder — the ONE directory
# to pull to the local machine. It sits under data/output, which is why guard (5)
# exists: the archiver would otherwise be able to reach its own output.
OUT_DIR="${OUT_DIR:-${CL_STEP_DOWNLOAD}}"
SPLIT_BYTES="${SPLIT_BYTES:-3000000000}"
CHUNK="${CHUNK:-2G}"

[[ -n "${MODE}" ]] || { cl_err "ERROR: one of --copy / --move is REQUIRED (there is no default)."; exit 2; }
if [[ "${DO_ALL}" == "1" ]]; then
    [[ -z "${SET}" ]] || { cl_err "ERROR: --all and --set are mutually exclusive."; exit 2; }
    rc=0
    for s in ${ALL_SETS}; do
        bash "${BASH_SOURCE[0]}" --set "${s}" "--${MODE}" ${NEWER:+--newer "${NEWER}"} ${TAG:+--tag "${TAG}"} \
            $([[ "${DRY}" == "1" ]] && echo --dry-run) || { cl_err "${s}: skipped/failed — continuing"; rc=1; }
    done
    cl_say "archives in ${OUT_DIR}"
    exit ${rc}
fi
[[ -n "${SET}" ]] || { cl_err "ERROR: --set <name> is required (or --all). See -h."; exit 2; }

# ── THE PATTERN TABLE. Two domains, each with its own root and required prefix.
#    data  -> cd ${DATA_ROOT} ; every pattern must start with 'output/'
#    logs  -> cd ${CL_ROOT}   ; every pattern must start with 'logs/'
# Every entry is a QUOTED string: the '*' must survive to the expansion pass
# further down, which runs AFTER the cd into the domain root. Letting the shell
# glob here would resolve the patterns against the scripts dir and match nothing.
#
# The step folders are named by cluster_lib.sh's CL_STEP_* and by of_root.jl's
# blp_dir/logit_dir/draws_dir/cf_out_dir. Renaming one means renaming it in all
# three places in the same edit.
DOMAIN=data; BASE=""; pats=(); EXCLUDES=()
case "${SET}" in
  # est{k} pickles and phi CSVs, the Rout exports (tex, figures, ame pickles) and
  # the DIAGNOSTICS tree. The two top-level globs take national_phi_t and the
  # summary jsons the merge step writes beside the est dirs.
  sleep)            BASE=sleep_outputs;   pats=("output/sleep/est*" "output/sleep/Rout" "output/sleep/DIAGNOSTICS" "output/sleep/*.csv" "output/sleep/*.json") ;;
  # The estimation sample every later phase is read against. Archived mid-run
  # (pipeline_all.sh, afterany G2) as well as in the terminal download.
  demand_prep)      BASE=demand_prep_outputs; pats=("output/demand_prep") ;;
  logit)            BASE=logit_outputs;   pats=("output/logit") ;;
  # RC results, checkpoints and summaries, plus the SMALL half of the draws.
  # demo_draws_R2000_seed42.jls is ~14 GB of Float64 randn noise that the local
  # machine can regenerate from (R, seed) in minutes, so it is opt-in: shipping it
  # by default would triple the whole download for a file nobody reads locally.
  blp)              BASE=blp_outputs;     pats=("output/blp/*.jls" "output/blp/*.json" "output/blp/draws/halton_nu_*" "output/blp/draws/demo_key_index_*" "output/blp/draws/*.json")
                    [[ "${INCLUDE_DRAWS:-0}" == "1" ]] && pats+=("output/blp/draws/demo_draws_*") ;;
  # polfunc, the psi shards and the cost params: one family, one producer.
  # INCLUDE_PSI=0 is the escape hatch for the case the ~100-shard psi_dev array
  # turns out to dominate the download; it defaults to INCLUDED because the
  # shards are the fwd_sim product and nothing recomputes them locally.
  bbl)              BASE=bbl_outputs;     pats=("output/bbl")
                    [[ "${INCLUDE_PSI:-1}" == "0" ]] && EXCLUDES+=("output/bbl/psi_*") ;;
  # upsilon_pix / phi_nopix, shares_elas, cf1_*, cf4_* and any equilibrium sigma
  # trees cf_eq_run.sh nested here.
  counterfactuals)  BASE=counterfactuals_outputs; pats=("output/counterfactuals") ;;
  # KB of json, but the pair that lets the local side tell a short zip from a
  # short RUN: the gates say which phases actually passed.
  gates)            BASE=gates;           pats=("output/.gate_*.json") ;;
  logs)             BASE=logs; DOMAIN=logs; pats=("logs/*.out" "logs/*.err" "logs/orchestrator_*.log") ;;
  *) cl_err "Unknown --set '${SET}'."
     cl_err "  data sets: sleep demand_prep logit blp bbl counterfactuals gates"
     cl_err "  log  sets: logs"
     exit 2 ;;
esac

if [[ "${DOMAIN}" == "data" ]]; then ROOT="${CL_DATA_ROOT}"; PREFIX="output/"; else ROOT="${CL_ROOT}"; PREFIX="logs/"; fi
_abs () { realpath -m "$1" 2>/dev/null || readlink -f "$1" 2>/dev/null || printf '%s' "$1"; }
ROOT_ABS="$(_abs "${ROOT}")"
DOMAIN_ABS="$(_abs "${ROOT_ABS}/${PREFIX%/}")"
INPUT_ABS="$(_abs "${CL_DATA_ROOT}/input")"
DOWNLOAD_ABS="$(_abs "${OUT_DIR}")"

# ── Guard (1): every pattern must be relative and start with the domain prefix.
for p in "${pats[@]}"; do
    case "${p}" in
        /*) cl_err "STARTUP ERROR: absolute pattern in set '${SET}': ${p}"; exit 2 ;;
        ${PREFIX}*) ;;
        *) cl_err "STARTUP ERROR: pattern in set '${SET}' does not start with '${PREFIX}': ${p}"; exit 2 ;;
    esac
done

# Expand the patterns INSIDE the domain root, with nullglob so an unmatched
# pattern vanishes instead of surviving as a literal.
cd "${ROOT_ABS}"
shopt -s nullglob
paths=()
for p in "${pats[@]}"; do
    for m in ${p}; do [[ -e "${m}" ]] && paths+=("${m}"); done
done
shopt -u nullglob
# An empty set exits 0, never 1: the download job chains all eight sets with &&,
# and a phase that legitimately produced nothing (no --cfeq run, a resume that
# skipped BBL) must not fail the packaging of the seven that did.
if [[ ${#paths[@]} -eq 0 ]]; then
    cl_say "${SET}: nothing to archive (no match under ${ROOT_ABS})."
    exit 0
fi

# ── Flatten directories to leaf files so every entry can be CONFIRMED in the
#    archive before it is deleted, then apply the exclusion / scope filters.
files=()
for p in "${paths[@]}"; do
    if [[ -d "${p}" ]]; then while IFS= read -r f; do files+=("${f}"); done < <(find "${p}" -type f)
    else files+=("${p}"); fi
done

# The logs set must never take its own still-open .out/.err.
SELF_RE=""
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF_RE="${SLURM_JOB_ID}"

scoped=()
for f in "${files[@]}"; do
    skip=0
    for x in ${EXCLUDES[@]+"${EXCLUDES[@]}"}; do [[ "${f}" == ${x} ]] && { skip=1; break; }; done
    [[ ${skip} -eq 1 ]] && continue
    [[ -n "${NEWER}" && ! "${f}" -nt "${NEWER}" ]] && continue
    [[ -n "${SELF_RE}" && "${f}" == *"${SELF_RE}"* ]] && continue
    scoped+=("${f}")
done
files=(${scoped[@]+"${scoped[@]}"})
if [[ ${#files[@]} -eq 0 ]]; then
    if [[ -n "${NEWER}" ]]; then
        cl_say "${SET}: ${#paths[@]} pattern match(es), none newer than $(basename "${NEWER}") — nothing to archive."
    else
        cl_say "${SET}: ${#paths[@]} pattern match(es) but they hold no files to package — nothing to archive."
    fi
    exit 0
fi

# ── Guards (2), (3) and (5): resolve every path and prove where it lives.
for f in "${files[@]}"; do
    rp="$(_abs "${f}")"
    case "${rp}" in
        "${DOMAIN_ABS}"/*) ;;
        *) cl_err "REFUSING: '${f}' resolves to '${rp}', outside the ${SET} domain root ${DOMAIN_ABS}."
           cl_err "  (a symlink out of the output tree is the usual cause)"; exit 2 ;;
    esac
    case "${rp}" in
        "${DOWNLOAD_ABS}"/*|"${DOWNLOAD_ABS}")
            cl_err "REFUSING: '${f}' resolves into the download folder itself: ${rp}"
            cl_err "  ${DOWNLOAD_ABS} holds the archives; packaging it would nest every earlier"
            cl_err "  zip inside the new one and grow without bound. Narrow the set's pattern." ; exit 2 ;;
    esac
    if [[ "${MODE}" == "move" ]]; then
        case "${rp}" in
            "${INPUT_ABS}"/*|"${INPUT_ABS}")
                cl_err "REFUSING: move-mode run touches an UPLOADED INPUT: ${rp}"
                cl_err "  data/input is uploaded by hand through the browser; deleting it is unrecoverable."
                exit 2 ;;
        esac
    fi
done

ZIP_NAME="${BASE}${TAG:+_${TAG}}.zip"
ZIP="${OUT_DIR}/${ZIP_NAME}"
SRC_BYTES="$(printf '%s\0' "${files[@]}" | xargs -0 stat -c%s 2>/dev/null | awk '{s+=$1} END{print s+0}')"

_hsize () {   # bytes -> "5.8 GB"
    awk -v b="${1:-0}" 'BEGIN{
        if (b>=1073741824) printf "%.1f GB", b/1073741824;
        else if (b>=1048576) printf "%.1f MB", b/1048576;
        else if (b>=1024) printf "%.1f KB", b/1024;
        else printf "%d B", b}'
}

# The per-file listing is narration: it belongs in the orchestrator log, not on a
# terminal that has to show one line per set.
cl_log "${SET}: ${#files[@]} file(s), $(_hsize "${SRC_BYTES}"), mode=${MODE}${NEWER:+, scoped by --newer $(basename "${NEWER}")} -> ${ZIP}"
for f in "${files[@]}"; do cl_log "  ${f}"; done
if [[ "${DRY}" == "1" ]]; then
    cl_say "${SET}: ${#files[@]} files, $(_hsize "${SRC_BYTES}") -> ${ZIP_NAME} (--dry-run: nothing written)"
    exit 0
fi

mkdir -p "${OUT_DIR}"

# ── COMPRESSION: store the float payloads, deflate the text ──────────────────
# `-n <suffix list>` is zip's own store-by-suffix switch, so the decision is per
# FILE inside one archive rather than per archive.
#
# STORED, because deflate buys ~nothing and costs the whole packaging wall:
#   .pkl .parquet .jls   float64 estimator state — measured: pickles barely
#                        compress, and the psi/est families are GBs of them.
#   .npz .zip .gz        already-compressed containers; re-deflating them can
#                        even grow the archive.
#   .so                  a prebuilt Julia sysimage is ~1 GB of machine code; no
#                        set globs one today, and if one ever does, a multi-minute
#                        deflate pass is not worth it for a transport archive.
#
# DEFLATED (everything else), because it is text and the win is large:
#   .csv  the phi CSVs measured ~8.4 GB -> ~2.7 GB
#   .json .tex .log .out .err  gates, summaries, exports, Slurm logs
STORE_SUFFIXES="${STORE_SUFFIXES:-.pkl:.parquet:.jls:.npz:.zip:.gz:.so}"
_add () { zip -q -n "${STORE_SUFFIXES}" "${ZIP}" "$@" 2>/dev/null || true; }
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
    unset have; remaining=(${ng[@]+"${ng[@]}"})
    ((${#remaining[@]} == 0)) && break
    ((++tries >= 4)) && break
done
if [[ "${MODE}" == "move" ]]; then
    for p in "${paths[@]}"; do [[ -d "${p}" ]] && find "${p}" -type d -empty -delete 2>/dev/null || true; done
fi
if ((${#remaining[@]})); then
    cl_err "WARNING: ${#remaining[@]} file(s) failed to archive after ${tries} retries — NOT deleted; re-run 'cluster_archive.sh --set ${SET} --${MODE}':"
    for f in "${remaining[@]}"; do cl_err "  ${f}"; done
fi

# ── SPLITTING: a browser download has no resume ──────────────────────────────
# One 10 GB file is one point of failure and one restart from zero; fixed-size
# parts fail at most a chunk at a time. The .sha256 covers the WHOLE zip, with a
# RELATIVE name, so the local side reassembles and then verifies in place:
#     cat <base>.zip.part.* > <base>.zip && sha256sum -c <base>.zip.sha256
# Stale parts and a stale whole-zip checksum are cleared on BOTH branches: a
# re-archive that crosses the threshold in either direction must not leave a
# mixture of the two shapes ON DISK behind. The matching cleanup in sha256SUMS is
# _sums_write's, because the shapes have different NAMES and a run only knows the
# names it is emitting.
ENTRIES="$(unzip -Z1 "${ZIP}" 2>/dev/null | wc -l | tr -d ' ')"
ZIP_BYTES="$(stat -c%s "${ZIP}" 2>/dev/null || echo 0)"
emitted=()
if (( ZIP_BYTES > SPLIT_BYTES )); then
    ( cd "${OUT_DIR}" \
      && rm -f "${ZIP_NAME}.part."* \
      && split -b "${CHUNK}" -d "${ZIP_NAME}" "${ZIP_NAME}.part." \
      && sha256sum "${ZIP_NAME}" > "${ZIP_NAME}.sha256" \
      && rm -f "${ZIP_NAME}" ) \
      || { cl_err "${SET}: split of ${ZIP_NAME} FAILED — the whole zip is left in place."; exit 1; }
    while IFS= read -r n; do emitted+=("$(basename "${n}")"); done < <(ls "${ZIP}.part."*)
    SHAPE="${#emitted[@]} parts"
else
    rm -f "${ZIP}.part."* "${ZIP}.sha256"
    emitted=("${ZIP_NAME}")
    SHAPE="whole"
fi

# ── CHECKSUMS: ONE sha256SUMS for the whole download folder ──────────────────
# The local side verifies everything it pulled with a single `sha256sum -c
# sha256SUMS` run inside the folder, so the names in it are RELATIVE.
#
# The append is serialised with flock, the same way blp_draws_job.sh serialises
# nothing but writes its checksum once: blp_zip, bbl_zip and pipe_download can be
# in flight together, and two unserialised appends interleave into a half-written
# line that then fails `-c` for a file that transferred perfectly.
#
# Two classes of line are dropped before the append, and the second is why the first
# is not enough.
#
#   (a) names THIS run re-emits. A re-archive under the same tag changes the zip's
#       bytes, and a stale line fails `-c` on a file that is fine.
#   (b) EVERY name this archive could go by, whatever shape it took last time:
#       ${ZIP_NAME} itself and ${ZIP_NAME}.part.*. A set's shape is not fixed —
#       INCLUDE_PSI=0 or INCLUDE_DRAWS=1 changes what goes in, so the same tag can
#       cross SPLIT_BYTES in either direction between runs. The SPLITTING block above
#       deletes the other shape's FILES, but this manifest is append-only across runs,
#       so without (b) the other shape's lines survive and `sha256sum -c sha256SUMS`
#       reports FAILED (no such file) on a download that is perfectly intact.
_sums_write () {   # runs inside the lock; reads _SUMS_LINES / _SUMS_NAMES / ZIP_NAME
    cd "${OUT_DIR}" || return 1
    if [[ -f sha256SUMS ]]; then
        # sha256sum writes 64 hex chars + two spaces, so the name starts at byte 67.
        awk -v base="${ZIP_NAME}" '
            NR==FNR{drop[$0]=1;next}
            {n=substr($0,67)}
            n==base || index(n, base ".part.")==1 {next}
            !(n in drop)' \
            <(printf '%s\n' "${_SUMS_NAMES}") sha256SUMS > ".sha256SUMS.$$" \
            && mv -f ".sha256SUMS.$$" sha256SUMS
    fi
    printf '%s\n' "${_SUMS_LINES}" >> sha256SUMS
}
_sums_append () {   # _sums_append <name relative to OUT_DIR>...
    local rc=0
    _SUMS_LINES="$( cd "${OUT_DIR}" && sha256sum "$@" )" || return 1
    _SUMS_NAMES="$(printf '%s\n' "$@")"
    if command -v flock >/dev/null 2>&1; then
        ( flock 9; _sums_write ) 9>"${OUT_DIR}/.sha256SUMS.lock" || rc=1
    else
        ( _sums_write ) || rc=1
    fi
    return ${rc}
}
if _sums_append "${emitted[@]}"; then SHA=OK; else SHA="FAILED"; cl_err "${SET}: could not update ${OUT_DIR}/sha256SUMS"; fi

# ── The ONE line the operator reads. Everything else went to the log. ────────
cl_say "${SET}: ${ENTRIES} files, $(_hsize "${SRC_BYTES}") -> ${ZIP_NAME} $(_hsize "${ZIP_BYTES}") (${SHAPE}, sha ${SHA})"
cl_log "  mode=${MODE} ($([[ "${MODE}" == "copy" ]] && echo "every original kept on disk" || echo "originals removed"))"
for n in "${emitted[@]}"; do cl_log "  -> ${OUT_DIR}/${n}"; done
