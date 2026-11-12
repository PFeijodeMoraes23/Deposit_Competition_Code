#!/bin/bash
# ==============================================================================
# blp_ext1_starts.sh — re-fit one RC-BLP rung of E3/E4 (ext1 by default; ext2 or extended with
# EXT_STAGE) from several explicit starting points, each in its own sandbox, and compare the
# solutions. The pipeline's own starts can stop where they began: ext1's at rc4's solution (the
# added pi at zero, a local optimum), and on panel fa38794d E4's ext2 and extended at ext1's
# solution with the added pi's at zero, where the FD check finds a descent step. This tests whether
# a lower-Q solution exists for the current data.
#
# Run from scripts/ on the login node:
#   bash blp_ext1_starts.sh --build-only   build and check the sandboxes, submit nothing
#   bash blp_ext1_starts.sh --submit-built one GPU job per built sandbox (once each)
#   bash blp_ext1_starts.sh                both steps at once
#   bash blp_ext1_starts.sh --report    the solutions side by side, and the production check
#
# EXT_STAGE (default ext1) is the rung: ext1 (5 theta2 values), ext2 (6: + pi fgc x 65+) or
# extended (7: + pi fgc x cadunico). EXT_TAG names the run (EXT1_TAG is read when EXT_TAG is
# unset; ext1 defaults to 20261029, ext2 and extended need one): starting points
# blp_<stage>_starts_<tag>.csv and sandboxes under diag_<stage>_<tag>/, so the sandboxes of one
# rung never share a folder with another rung's. EXT_DIAG_ROOT (EXT1_DIAG_ROOT) is where the
# diag_* folders live (default: the project folder). Any file system works: a sandbox only links
# to the real data, so a faster one (scratch) can hold them.
#
# The csv: a header line, then one row per start: routine (3 or 4), start name, the rung's theta2
# values, and optionally where the fit's delta starts: logit (the default: the engine's own start,
# the logit delta-hat) or rc4 | ext1 | ext2 | extended, the routine's production checkpoint delta*
# of that rung (blp_checkpoint_E<k>_spec_12_<rung>.bin), copied into the sandbox's data/input
# under the logit delta's name, which the engine reads before any other delta. Lines starting with
# # are skipped. Every row is checked before anything is built.
#
# Each start gets <root>/diag_<stage>_<tag>/E<k>_<start>/ with scripts/ (a copy of its top-level
# files, config/, utils/; the sysimages and the Julia depot are linked) and data/ (links to the
# real input/ and to every output/ step folder except blp/, which is its own, with the draws
# linked in). With a checkpoint delta, data/input is a folder of links to each real input except
# that delta, which is a copy; delta_start.txt and delta_start.sha256 record it. The engine writes
# to ../data relative to its own code, so its results land in the sandbox. The production files
# of the rung are hashed before the jobs start, and --report checks the hashes.
#
# Walls: ext1 jobs keep the job script's 06:00:00; ext2 and extended jobs get 1-00:00:00. An
# exported SBATCH_TIMELIMIT overrides both. --report marks a fit "pass" when its FD check found
# no descent step (best dQ >= -EXT_FD_TOL, default 0).
# ==============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${HERE}/.." && pwd)"
REAL="${PROJ}/data"
STAGE="${EXT_STAGE:-ext1}"
case "${STAGE}" in
    ext1)     NPAR=5; JPFX=ext1s ;;
    ext2)     NPAR=6; JPFX=ext2s ;;
    extended) NPAR=7; JPFX=extds ;;
    *) echo "EXT_STAGE='${STAGE}' must be ext1, ext2 or extended" >&2; exit 2 ;;
esac
if [[ "${STAGE}" == "ext1" ]]; then
    TAG="${EXT_TAG:-${EXT1_TAG:-20261029}}"
else
    TAG="${EXT_TAG:-${EXT1_TAG:-}}"
    [[ -n "${TAG}" ]] || { echo "EXT_STAGE=${STAGE} needs EXT_TAG (the run's name, e.g. 20261106)" >&2; exit 2; }
fi
ROOT="${EXT_DIAG_ROOT:-${EXT1_DIAG_ROOT:-${PROJ}}}"
DIAG="${ROOT}/diag_${STAGE}_${TAG}"
STARTS="${HERE}/blp_${STAGE}_starts_${TAG}.csv"
HASHES="${DIAG}/production_${STAGE}.sha256"
FD_TOL="${EXT_FD_TOL:-0}"
# What the printed follow-up commands carry: nothing for ext1 (its defaults), the run otherwise.
RUN_ENV=""
[[ "${STAGE}" == "ext1" ]] || RUN_ENV="EXT_STAGE=${STAGE} EXT_TAG=${TAG} EXT_DIAG_ROOT=${ROOT} "

if [[ "${1:-}" == "--report" ]]; then
    [[ -d "${DIAG}" ]] || { echo "no ${DIAG}: nothing launched yet" >&2; exit 1; }
    python3 - "${DIAG}" "${REAL}/output/blp" "${STAGE}" "${STARTS}" "${NPAR}" "${FD_TOL}" "${RUN_ENV}" <<'PY'
import json, sys
from pathlib import Path
diag, prod, stage, starts = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
npar, fd_tol, run_env = int(sys.argv[5]), float(sys.argv[6]), sys.argv[7]

# Every rung: the FD check, the move from the start, the delta start, and why the optimizer stopped
# (the engine's "outer" record, written since engine edit (b); "-" for a result without it).
names = {"ext1": "sigma, pi gdp, pi 65+, pi mobile, pi assets x gdp",
         "ext2": "sigma, pi gdp, pi 65+, pi mobile, pi assets x gdp, pi fgc x 65+",
         "extended": "sigma, pi gdp, pi 65+, pi mobile, pi assets x gdp, pi fgc x 65+, pi fgc x cadunico"}[stage]
start = {}
if starts.is_file():
    for i, line in enumerate(starts.read_text().splitlines()):
        s = line.strip()
        if i == 0 or not s or s.startswith("#"):
            continue
        f = [x.strip() for x in s.split(",")]
        start[f"E{f[0]}_{f[1]}"] = ([float(x) for x in f[2:2 + npar]],
                                     f[2 + npar] if len(f) > 2 + npar and f[2 + npar] else "logit")

def fd_cols(j):
    fd = j.get("fd_check") or {}
    n, e, b = fd.get("grad_fd_projected_norm"), fd.get("grad_exact_projected_norm"), fd.get("best_dQ")
    if not isinstance(b, (int, float)):
        return "      n/a       n/a        n/a  n/a ", False
    ok = b >= -fd_tol
    fmt = lambda x: f"{x:9.2e}" if isinstance(x, (int, float)) else f"{'n/a':>9s}"
    return f"{fmt(n)} {fmt(e)} {b:+10.2e}  {'pass' if ok else 'FAIL'}", ok

def stop_cols(j):
    o = j.get("outer") or {}
    s, ls = str(o.get("stop_reason") or "-"), str(o.get("line_search") or "")
    code = ("CONV:PG" if "PROJECTED_GRADIENT" in s else "CONV:F" if "REL_REDUCTION" in s else
            "ABNORMAL" if s.startswith("ABNORMAL") else "STOP" if s.startswith("STOP") else
            "FMINBOX" if "Fminbox" in s else "-" if s == "-" else s[:8])
    code += "+W" if ls.startswith("WARNING") else ""
    g = o.get("inner_guard") or {}
    gtxt = f"{g.get('n_fail', '-')}/{g.get('n_eval', '-')}" if g else "-"
    return f"{code:10s} {str(o.get('n_evals', '-')):>5s} {str(o.get('step', '-')):5s} {gtxt:>7s}"

def line(name, j, moved, delta):
    se = (j.get("theta1_se") or [None])[0]
    fd, ok = fd_cols(j)
    t1 = j.get("theta1") or [float("nan")]
    print(f"{name:18s} {str(j.get('converged')):5s} {j.get('Q_value', float('nan')):12.10f} "
          f"{t1[0]:+8.3f} {se if se is None else round(se, 3)!s:>7s} {str(j.get('se_status', '-'))[:6]:6s} "
          f"{fd} {moved:>8s} {delta:8s} {stop_cols(j)} " + ", ".join(f"{x:+.6f}" for x in j.get("theta2", [])))
    return ok

print(f"stage {stage}; FD pass = best dQ >= {0.0 - fd_tol:g}; moved = max |theta2* - start|; "
      "stop = the optimizer's stop (CONV:PG projected gradient, CONV:F relative reduction of Q, +W after a "
      "line-search WARNING), evals, step control, guard = failed/all inner evaluations")
print(f"{'run':18s} {'conv':5s} {'Q':>12s} {'alpha':>8s} {'se':>7s} {'se_st':6s} "
      f"{'FD norm':>9s} {'IFT norm':>9s} {'best dQ':>10s}  {'FD':4s} {'moved':>8s} {'delta':8s} "
      f"{'stop':10s} {'evals':>5s} {'step':5s} {'guard':>7s} theta2 ({names})")
routines = sorted({int(n[1:].split("_")[0]) for n in start}
                  | {int(p.name[1:].split("_")[0]) for p in diag.glob("E*_*") if p.is_dir()})
for k in routines:
    f = prod / f"blp_results_E{k}_spec_12_{stage}.json"
    if f.is_file():
        line(f"E{k} production", json.loads(f.read_text()), "-", "-")
    cands = []
    for sb in sorted(p for p in diag.glob(f"E{k}_*") if p.is_dir()):
        f = sb / "data" / "output" / "blp" / f"blp_results_E{k}_spec_12_{stage}.json"
        t0 = start.get(sb.name, (None, ""))[0]
        # The delta start as built (delta_start.txt), not as the csv now says.
        dtxt = sb / "delta_start.txt"
        delta = (dtxt.read_text().split("(rung ", 1)[-1].split(" ", 1)[0] if dtxt.is_file() else "logit")
        if not f.is_file():
            print(f"{sb.name:18s} (no result yet)  delta start: {delta}")
            continue
        j = json.loads(f.read_text())
        t2 = j.get("theta2", [])
        moved = (f"{max(abs(a - b) for a, b in zip(t2, t0)):.1e}" if t0 and len(t0) == len(t2) else "?")
        ok = line(sb.name, j, moved, delta)
        if ok and j.get("converged") is True and j.get("se_status") == "ok":
            cands.append((j.get("Q_value", float("inf")), sb.name))
    if stage == "ext1":
        continue                     # ext1: the table only, as before (adoption: blp_ext1_adopt.sh)
    if cands:
        q, nm = min(cands)
        print(f"  E{k}: lowest Q among the converged fits that pass the FD check with se_status ok: {nm} "
              f"(Q {q:.7f}). Adopting is your call: {run_env}bash blp_ext1_adopt.sh {k} {nm.split('_', 1)[1]}")
    else:
        print(f"  E{k}: no sandbox fit is converged, FD-passing and se_status ok (yet).")
PY
    echo "production ${STAGE} files unchanged?"
    ( cd / && sha256sum -c --quiet "${HASHES}" ) && echo "  yes: every hash matches" || echo "  NO: see above"
    exit 0
fi

MODE="${1:-}"
[[ -z "${MODE}" || "${MODE}" == "--build-only" || "${MODE}" == "--submit-built" ]] \
    || { echo "usage: bash blp_ext1_starts.sh [--build-only | --submit-built | --report]" >&2; exit 2; }
[[ -f "${STARTS}" ]] || { echo "missing ${STARTS}" >&2; exit 1; }
PROD_BLP="$(readlink -f "${REAL}/output/blp")"

# The rows of the csv, checked once before anything is built or submitted:
# ROW_K, ROW_ST, ROW_V (the theta2 values joined with ':'), ROW_D (the delta start).
ROW_K=(); ROW_ST=(); ROW_V=(); ROW_D=()
NUM_RE='^[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?$'
while IFS= read -r ln || [[ -n "${ln}" ]]; do
    ln="${ln%$'\r'}"
    [[ -n "${ln//[[:space:],]/}" && "${ln}" != \#* ]] || continue
    IFS=, read -r -a f <<< "${ln}"
    k="${f[0]}"; st="${f[1]:-}"; d="logit"
    if (( ${#f[@]} == NPAR + 3 )); then
        d="${f[NPAR + 2]}"
    elif (( ${#f[@]} != NPAR + 2 )); then
        echo "REFUSING ${STARTS##*/}: row '${ln}' has $(( ${#f[@]} - 2 )) values after routine,start; ${STAGE} takes ${NPAR} (+ an optional delta start)" >&2; exit 2
    fi
    [[ "${k}" == "3" || "${k}" == "4" ]] || { echo "REFUSING ${STARTS##*/}: routine '${k}' (only 3 or 4: the job asks for 200G)" >&2; exit 2; }
    [[ "${st}" =~ ^[A-Za-z0-9][A-Za-z0-9_.+-]*$ ]] || { echo "REFUSING ${STARTS##*/}: start name '${st}' (letters, digits, _ . + -)" >&2; exit 2; }
    case "${d}" in logit|rc4|ext1|ext2|extended) ;; *) echo "REFUSING ${STARTS##*/}: E${k} ${st}: delta start '${d}' (logit | rc4 | ext1 | ext2 | extended)" >&2; exit 2 ;; esac
    for i in "${!ROW_K[@]}"; do
        [[ "${ROW_K[i]}_${ROW_ST[i]}" != "${k}_${st}" ]] || { echo "REFUSING ${STARTS##*/}: E${k} ${st} appears twice" >&2; exit 2; }
    done
    v=""
    for (( i = 2; i < NPAR + 2; i++ )); do
        [[ "${f[i]}" =~ ${NUM_RE} ]] || { echo "REFUSING ${STARTS##*/}: E${k} ${st}: '${f[i]}' is not a number" >&2; exit 2; }
        v="${v:+${v}:}${f[i]}"
    done
    ROW_K+=("${k}"); ROW_ST+=("${st}"); ROW_V+=("${v}"); ROW_D+=("${d}")
done < <(tail -n +2 "${STARTS}")
(( ${#ROW_K[@]} > 0 )) || { echo "REFUSING: ${STARTS##*/} holds no start" >&2; exit 2; }

# check_sandbox <sandbox> [routine]: every path the job reads or writes, resolved; refuses on the first wrong one.
check_sandbox () {
    local sb="$1" k="${2:-}" bad="" e nm miss=0 dname
    [[ -d "${sb}/data/output/blp" && ! -L "${sb}/data/output/blp" ]] || bad="${bad} output/blp is not a plain folder;"
    [[ "$(readlink -f "${sb}/data/output/blp")" != "${PROD_BLP}" ]] || bad="${bad} output/blp resolves to production;"
    [[ "$(readlink -f "${sb}/data/output/blp/draws")" == "$(readlink -f "${REAL}/output/blp/draws")" ]] || bad="${bad} draws not linked;"
    if [[ -f "${sb}/delta_start.sha256" ]]; then
        # A checkpoint delta start: data/input is the sandbox's own folder of links, with the delta a copy.
        dname="logit_delta_E${k}_spec_12.bin"
        [[ -d "${sb}/data/input" && ! -L "${sb}/data/input" ]] || bad="${bad} input is not a plain folder;"
        [[ -f "${sb}/data/input/${dname}" && ! -L "${sb}/data/input/${dname}" ]] || bad="${bad} the delta start is not a file of its own;"
        sha256sum -c --quiet "${sb}/delta_start.sha256" >/dev/null 2>&1 || bad="${bad} the delta start differs from its recorded hash;"
        while IFS= read -r -d '' e; do
            nm="$(basename "${e}")"
            [[ "${nm}" == "${dname}" ]] && continue
            [[ -L "${sb}/data/input/${nm}" && "$(readlink -f "${sb}/data/input/${nm}")" == "$(readlink -f "${e}")" ]] || miss=$((miss + 1))
        done < <(find "${REAL}/input/" -mindepth 1 -maxdepth 1 -print0)
        (( miss == 0 )) || bad="${bad} ${miss} input(s) not linked;"
    else
        [[ "$(readlink -f "${sb}/data/input")" == "$(readlink -f "${REAL}/input")" ]] || bad="${bad} input not linked;"
    fi
    [[ -L "${sb}/scripts/.julia_depot" && "$(readlink -f "${sb}/scripts/.julia_depot")" == "$(readlink -f "${HERE}/.julia_depot")" ]] \
        || bad="${bad} .julia_depot not linked;"
    [[ -L "${sb}/scripts/blp_sysimage.so" ]] || bad="${bad} blp_sysimage.so not linked;"
    [[ ! -L "${sb}/scripts" && "$(cd "${sb}/scripts" && pwd -P)" != "$(cd "${HERE}" && pwd -P)" ]] || bad="${bad} scripts is not a copy;"
    [[ -f "${sb}/scripts/blp_ext1_start_job.sh" && -f "${sb}/scripts/blp_stage_job.sh" ]] || bad="${bad} job scripts missing;"
    if [[ -n "${bad}" ]]; then echo "REFUSING ${sb##*/}:${bad}" >&2; return 1; fi
    echo "  ${sb##*/}: sandbox OK (output/blp $(readlink -f "${sb}/data/output/blp"))"
}

if [[ "${MODE}" != "--submit-built" ]]; then
    # Resumable: a build cut short (a slow file system, a job's time limit) is completed by running
    # it again. The production hashes are taken once, before any job; a sandbox that passed its
    # check carries built.ok and is skipped.
    mkdir -p "${DIAG}"
    if [[ -s "${HASHES}" ]]; then
        echo "production ${STAGE} hashes kept from the first build: ${HASHES}"
    else
        HK=(); for k in $(printf '%s\n' "${ROW_K[@]}" | sort -u); do HK+=("${k}"); done
        HF=()
        for x in json jls; do for k in "${HK[@]}"; do HF+=("${REAL}/output/blp/blp_results_E${k}_spec_12_${STAGE}.${x}"); done; done
        sha256sum "${HF[@]}" > "${HASHES}.tmp"
        mv -f "${HASHES}.tmp" "${HASHES}"
        echo "production ${STAGE} hashed -> ${HASHES}"
    fi
    for r in "${!ROW_K[@]}"; do
        k="${ROW_K[r]}"; st="${ROW_ST[r]}"; d="${ROW_D[r]}"
        SB="${DIAG}/E${k}_${st}"
        if [[ -f "${SB}/built.ok" ]]; then echo "  ${SB##*/}: already built"; continue; fi
        mkdir -p "${SB}/data/output/blp"
        # The code the stage job runs: every top-level file of scripts/ (the .jl it includes, the
        # .sh it sources, Project/Manifest, the sysimage stamps) and the small config/ and utils/
        # folders. Nothing else is walked: scripts/ also holds very large trees (the Julia depot,
        # build folders), which a recursive copy would have to scan file by file.
        mkdir -p "${SB}/scripts/logs"
        find "${HERE}" -maxdepth 1 -type f ! -name '*.so' ! -name '*.zip' -exec cp -p -t "${SB}/scripts/" {} +
        for sub in config utils; do
            if [[ -d "${HERE}/${sub}" ]]; then
                mkdir -p "${SB}/scripts/${sub}"
                cp -rp "${HERE}/${sub}/." "${SB}/scripts/${sub}/"
            fi
        done
        # Links, never copies; ln -sfn replaces a link a cut-short build left behind.
        if [[ -e "${SB}/scripts/.julia_depot" && ! -L "${SB}/scripts/.julia_depot" ]]; then
            echo "REFUSING: ${SB}/scripts/.julia_depot is a copy, not a link" >&2; exit 2
        fi
        ln -sfn "${HERE}/.julia_depot" "${SB}/scripts/.julia_depot"
        for so in "${HERE}"/*.so; do
            if [[ -e "${so}" ]]; then ln -sfn "${so}" "${SB}/scripts/$(basename "${so}")"; fi
        done
        if [[ "${d}" == "logit" ]]; then
            ln -sfn "${REAL}/input" "${SB}/data/input"
        else
            # The fit's delta starts from a converged delta*: the routine's production checkpoint of
            # rung ${d}, copied under the name the engine reads first (data/input/logit_delta_*.bin).
            src="${REAL}/output/blp/blp_checkpoint_E${k}_spec_12_${d}.bin"
            dname="logit_delta_E${k}_spec_12.bin"
            ref="${REAL}/output/logit/${dname}"
            [[ -f "${src}" ]] || { echo "REFUSING ${SB##*/}: no delta start ${src}" >&2; exit 2; }
            if [[ -f "${ref}" && "$(stat -L -c %s "${src}")" != "$(stat -L -c %s "${ref}")" ]]; then
                echo "REFUSING ${SB##*/}: ${src##*/} is $(stat -L -c %s "${src}") bytes, the logit delta ${ref##*/} $(stat -L -c %s "${ref}"): another length, which the engine would skip" >&2; exit 2
            fi
            if [[ -L "${SB}/data/input" ]]; then
                echo "REFUSING ${SB##*/}: data/input is a link (built earlier with the logit delta); build this start under a new name" >&2; exit 2
            fi
            mkdir -p "${SB}/data/input"
            while IFS= read -r -d '' e; do
                nm="$(basename "${e}")"
                [[ "${nm}" == "${dname}" ]] && continue
                ln -sfn "${e}" "${SB}/data/input/${nm}"
            done < <(find "${REAL}/input/" -mindepth 1 -maxdepth 1 -print0)
            cp -p "${src}" "${SB}/data/input/${dname}"
            ( cd "${SB}/data/input" && sha256sum "${PWD}/${dname}" ) > "${SB}/delta_start.sha256"
            printf 'delta start: %s (rung %s of E%s), copied to %s\nsha256 %s\n' "${src}" "${d}" "${k}" \
                "${SB}/data/input/${dname}" "$(cut -d' ' -f1 "${SB}/delta_start.sha256")" > "${SB}/delta_start.txt"
            echo "  ${SB##*/}: delta start = ${src##*/} ($(cut -c1-16 "${SB}/delta_start.sha256")...)"
        fi
        for d2 in "${REAL}"/output/*; do
            b="$(basename "${d2}")"
            [[ "${b}" == "blp" ]] && continue
            ln -sfn "${d2}" "${SB}/data/output/${b}"
        done
        ln -sfn "${REAL}/output/blp/draws" "${SB}/data/output/blp/draws"
        check_sandbox "${SB}" "${k}" || exit 2
        touch "${SB}/built.ok"
    done
    if [[ "${MODE}" == "--build-only" ]]; then
        echo "built, nothing submitted. Submit with: ${RUN_ENV}bash blp_ext1_starts.sh --submit-built"
        exit 0
    fi
fi

# ext2 and extended take a longer wall than the job script's 06:00:00, unless SBATCH_TIMELIMIT is set.
WALL_ARGS=(); STAGE_EXPORT=""
if [[ "${STAGE}" != "ext1" ]]; then
    STAGE_EXPORT=",EXT_STAGE=${STAGE}"
    [[ -n "${SBATCH_TIMELIMIT:-}" ]] || WALL_ARGS=(--time=1-00:00:00)
fi
for r in "${!ROW_K[@]}"; do
    k="${ROW_K[r]}"; st="${ROW_ST[r]}"; v="${ROW_V[r]}"; d="${ROW_D[r]}"
    SB="${DIAG}/E${k}_${st}"
    [[ -d "${SB}" ]] || { echo "REFUSING: ${SB} was never built" >&2; exit 2; }
    if [[ -f "${SB}/submitted.jid" ]]; then echo "E${k} ${st}: already submitted ($(cat "${SB}/submitted.jid"))"; continue; fi
    check_sandbox "${SB}" "${k}" >/dev/null || exit 2
    # Run inside a batch job (the overnight launch), this shell carries that job's SLURM_* settings
    # (1 cpu, 2G), which --export=ALL would hand to the GPU job: they are dropped first.
    jid=$(cd "${SB}/scripts" && for vv in $(compgen -v SLURM_ || true); do unset "${vv}"; done \
          && env -u CL_ROOT -u DATA_ROOT -u CL_DATA_ROOT -u CL_DATA_OUT -u CL_DATA_IN \
          sbatch --parsable -J "${JPFX}_E${k}_${st}" ${WALL_ARGS[@]+"${WALL_ARGS[@]}"} \
                 -o "${SB}/scripts/logs/%x_%j.out" -e "${SB}/scripts/logs/%x_%j.err" \
                 --export=ALL,RC_ROUTINE="${k}",EXT1_START="${v}",EXT1_PROD_BLP="${PROD_BLP}"${STAGE_EXPORT} \
                 "${SB}/scripts/blp_ext1_start_job.sh")
    echo "${jid}" > "${SB}/submitted.jid"
    echo "E${k} ${st}: start ${v//:/, } -> job ${jid}$([[ ! -f "${SB}/delta_start.txt" ]] \
         || echo " (delta start: $(sed -n '1s#^delta start: .*/\([^/ ]*\) (rung.*#\1#p' "${SB}/delta_start.txt"))")"
done
echo "submitted. Status: squeue -u \$USER -o \"%.10i %.22j %.8T %.10M %R\" | grep ${JPFX}"
echo "results:  ${RUN_ENV}bash blp_ext1_starts.sh --report"
