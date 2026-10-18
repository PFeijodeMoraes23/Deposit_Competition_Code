#!/bin/bash
# ==============================================================================
# blp_ext1_starts.sh — re-fit the ext1 rung of E3/E4 from several explicit starting points, each
# in its own sandbox, and compare the solutions. The 10-16 run's ext1 sits at rc4's solution
# (alpha about 0, Q 0.0873 / 0.0936); the 10-05 run's ext1 reached alpha about -0.43 from
# another start. This tests whether a lower-Q ext1 solution exists for the current data.
#
# Run from scripts/ on the login node:
#   bash blp_ext1_starts.sh --build-only   build and check the sandboxes, submit nothing
#   bash blp_ext1_starts.sh --submit-built one GPU job per built sandbox (once each)
#   bash blp_ext1_starts.sh                both steps at once
#   bash blp_ext1_starts.sh --report    the solutions side by side, and the production check
#
# Starting points: blp_ext1_starts_20261018.csv (routine, start, the five ext1 theta2 values).
# Each start gets <proj>/diag_ext1_20261018/E<k>_<start>/ with scripts/ (a copy of its top-level files, config/, utils/;
# the sysimages and the Julia depot are linked) and data/ (links to the real input/ and to every
# output/ step folder except blp/, which is its own, with the draws linked in). The engine writes
# to ../data relative to its own code, so its results land in the sandbox. The production ext1
# files are hashed before the jobs start, and --report checks the hashes.
# ==============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${HERE}/.." && pwd)"
REAL="${PROJ}/data"
DIAG="${PROJ}/diag_ext1_20261018"
STARTS="${HERE}/blp_ext1_starts_20261018.csv"
HASHES="${DIAG}/production_ext1.sha256"

if [[ "${1:-}" == "--report" ]]; then
    [[ -d "${DIAG}" ]] || { echo "no ${DIAG}: nothing launched yet" >&2; exit 1; }
    python3 - "${DIAG}" "${REAL}/output/blp" <<'PY'
import json, sys
from pathlib import Path
diag, prod = Path(sys.argv[1]), Path(sys.argv[2])
print(f"{'run':16s} {'conv':5s} {'Q':>9s} {'alpha':>8s} {'se':>7s}  theta2 (sigma, pi gdp, pi 65+, pi mobile, pi assets x gdp)")
for k in (3, 4):
    f = prod / f"blp_results_E{k}_spec_12_ext1.json"
    if f.is_file():
        j = json.loads(f.read_text())
        se = (j.get("theta1_se") or [None])[0]
        print(f"{'E%d production' % k:16s} {str(j.get('converged')):5s} {j.get('Q_value', float('nan')):9.4f} "
              f"{j['theta1'][0]:+8.3f} {se if se is None else round(se, 3)!s:>7s}  "
              + ", ".join(f"{x:+.3f}" for x in j.get("theta2", [])))
    for sb in sorted(diag.glob(f"E{k}_*")):
        f = sb / "data" / "output" / "blp" / f"blp_results_E{k}_spec_12_ext1.json"
        if not f.is_file():
            print(f"{sb.name:16s} (no result yet)")
            continue
        j = json.loads(f.read_text())
        se = (j.get("theta1_se") or [None])[0]
        print(f"{sb.name:16s} {str(j.get('converged')):5s} {j.get('Q_value', float('nan')):9.4f} "
              f"{j['theta1'][0]:+8.3f} {se if se is None else round(se, 3)!s:>7s}  "
              + ", ".join(f"{x:+.3f}" for x in j.get("theta2", [])))
PY
    echo "production ext1 files unchanged?"
    ( cd / && sha256sum -c --quiet "${HASHES}" ) && echo "  yes: every hash matches" || echo "  NO: see above"
    exit 0
fi

MODE="${1:-}"
[[ -z "${MODE}" || "${MODE}" == "--build-only" || "${MODE}" == "--submit-built" ]] \
    || { echo "usage: bash blp_ext1_starts.sh [--build-only | --submit-built | --report]" >&2; exit 2; }
[[ -f "${STARTS}" ]] || { echo "missing ${STARTS}" >&2; exit 1; }
PROD_BLP="$(readlink -f "${REAL}/output/blp")"

# check_sandbox <sandbox>: every path the job reads or writes, resolved; refuses on the first wrong one.
check_sandbox () {
    local sb="$1" bad=""
    [[ -d "${sb}/data/output/blp" && ! -L "${sb}/data/output/blp" ]] || bad="${bad} output/blp is not a plain folder;"
    [[ "$(readlink -f "${sb}/data/output/blp")" != "${PROD_BLP}" ]] || bad="${bad} output/blp resolves to production;"
    [[ "$(readlink -f "${sb}/data/output/blp/draws")" == "$(readlink -f "${REAL}/output/blp/draws")" ]] || bad="${bad} draws not linked;"
    [[ "$(readlink -f "${sb}/data/input")" == "$(readlink -f "${REAL}/input")" ]] || bad="${bad} input not linked;"
    [[ -L "${sb}/scripts/.julia_depot" && "$(readlink -f "${sb}/scripts/.julia_depot")" == "$(readlink -f "${HERE}/.julia_depot")" ]] \
        || bad="${bad} .julia_depot not linked;"
    [[ -L "${sb}/scripts/blp_sysimage.so" ]] || bad="${bad} blp_sysimage.so not linked;"
    [[ ! -L "${sb}/scripts" && "$(cd "${sb}/scripts" && pwd -P)" != "$(cd "${HERE}" && pwd -P)" ]] || bad="${bad} scripts is not a copy;"
    [[ -f "${sb}/scripts/blp_ext1_start_job.sh" && -f "${sb}/scripts/blp_stage_job.sh" ]] || bad="${bad} job scripts missing;"
    if [[ -n "${bad}" ]]; then echo "REFUSING ${sb##*/}:${bad}" >&2; return 1; fi
    echo "  ${sb##*/}: sandbox OK (output/blp $(readlink -f "${sb}/data/output/blp"))"
}

if [[ "${MODE}" != "--submit-built" ]]; then
    [[ ! -e "${DIAG}" ]] || { echo "REFUSING: ${DIAG} exists (a launch already ran). Use --submit-built or --report." >&2; exit 1; }
    mkdir -p "${DIAG}"
    sha256sum "${REAL}"/output/blp/blp_results_E{3,4}_spec_12_ext1.json \
              "${REAL}"/output/blp/blp_results_E{3,4}_spec_12_ext1.jls > "${HASHES}"
    echo "production ext1 hashed -> ${HASHES}"
    while IFS=, read -r k st s p1 p2 p3 p4; do
        [[ -n "${k}" ]] || continue
        SB="${DIAG}/E${k}_${st}"
        mkdir -p "${SB}/data/output/blp"
        # The code the stage job runs: every top-level file of scripts/ (the .jl it includes, the
        # .sh it sources, Project/Manifest, the sysimage stamps) and the small config/ and utils/
        # folders. Nothing else is walked: scripts/ also holds very large trees (the Julia depot,
        # build folders), which a recursive copy would have to scan file by file.
        mkdir -p "${SB}/scripts/logs"
        find "${HERE}" -maxdepth 1 -type f ! -name '*.so' ! -name '*.zip' -exec cp -p -t "${SB}/scripts/" {} +
        for sub in config utils; do
            if [[ -d "${HERE}/${sub}" ]]; then cp -rp "${HERE}/${sub}" "${SB}/scripts/${sub}"; fi
        done
        [[ ! -e "${SB}/scripts/.julia_depot" ]] || { echo "REFUSING: ${SB}/scripts/.julia_depot was copied" >&2; exit 2; }
        ln -s "${HERE}/.julia_depot" "${SB}/scripts/.julia_depot"
        for so in "${HERE}"/*.so; do [[ -e "${so}" ]] && ln -sfn "${so}" "${SB}/scripts/$(basename "${so}")"; done
        ln -s "${REAL}/input" "${SB}/data/input"
        for d in "${REAL}"/output/*; do
            b="$(basename "${d}")"
            [[ "${b}" == "blp" ]] && continue
            ln -s "${d}" "${SB}/data/output/${b}"
        done
        ln -s "${REAL}/output/blp/draws" "${SB}/data/output/blp/draws"
        check_sandbox "${SB}" || exit 2
    done < <(tail -n +2 "${STARTS}")
    if [[ "${MODE}" == "--build-only" ]]; then
        echo "built, nothing submitted. Submit with: bash blp_ext1_starts.sh --submit-built"
        exit 0
    fi
fi

while IFS=, read -r k st s p1 p2 p3 p4; do
    [[ -n "${k}" ]] || continue
    SB="${DIAG}/E${k}_${st}"
    [[ -d "${SB}" ]] || { echo "REFUSING: ${SB} was never built" >&2; exit 2; }
    if [[ -f "${SB}/submitted.jid" ]]; then echo "E${k} ${st}: already submitted ($(cat "${SB}/submitted.jid"))"; continue; fi
    check_sandbox "${SB}" >/dev/null || exit 2
    jid=$(cd "${SB}/scripts" && env -u CL_ROOT -u DATA_ROOT -u CL_DATA_ROOT -u CL_DATA_OUT -u CL_DATA_IN \
          sbatch --parsable -J "ext1s_E${k}_${st}" \
                 -o "${SB}/scripts/logs/%x_%j.out" -e "${SB}/scripts/logs/%x_%j.err" \
                 --export=ALL,RC_ROUTINE="${k}",EXT1_START="${s}:${p1}:${p2}:${p3}:${p4}" \
                 "${SB}/scripts/blp_ext1_start_job.sh")
    echo "${jid}" > "${SB}/submitted.jid"
    echo "E${k} ${st}: start ${s}, ${p1}, ${p2}, ${p3}, ${p4} -> job ${jid}"
done < <(tail -n +2 "${STARTS}")
echo "submitted. Status: squeue -u \$USER -o \"%.10i %.22j %.8T %.10M %R\" | grep ext1s"
echo "results:  bash blp_ext1_starts.sh --report"
