#!/bin/bash
# ==============================================================================
# tr5_variants.sh — TR5, the type-5 robustness of the whole estimation chain: the sleepiness
# estimators, the demand prep, the logit and the RC-BLP ladder re-run without the prepaid product,
# each variant in its own sandbox, through pipeline_all.sh --no-bbl --no-cf.
#   no5     every type-5 row dropped
#   d5only  the type-5 rows of brick-and-mortar firms dropped; the digital firms' kept
# The variant reaches every job through config/deposit_type_variant.txt inside the sandbox's copy
# of the scripts (read by utils/deposit_types.py) and through DEPOSIT_TYPE_VARIANT in the
# submitting environment; the production tree has neither.
#
# Run from scripts/ on the login node (each mode is login-safe: copies, links, sbatch, squeue, zip):
#   bash tr5_variants.sh --build-only     build and check the sandboxes, submit nothing
#   bash tr5_variants.sh --submit-built   one pipeline_all.sh per variant (once each)
#   bash tr5_variants.sh --status         the variant's jobs in the queue and its gate verdicts
#   bash tr5_variants.sh --pack           zip each sandbox's results for download
#   bash tr5_variants.sh --cancel         cancel every job running from the sandboxes
# Sandboxes: ${TR5_ROOT:-<proj>}/tr5/<variant>/, with
#   scripts/      a copy of the top-level files, config/ and utils/ of scripts/, overlaid by
#                 ${TR5_UPLOAD:-<proj>/upload_tr5}/scripts/ (the TR5 code), the Julia depot and the
#                 sysimages linked, config/deposit_type_variant.txt = the variant
#   data/input    a link to the production data/input (the uploads, read only)
#   data/output   the sandbox's own, empty at build: the chain writes every step folder here
# The chain each variant runs (TR5_ROUTINES, default "3 4"; TR5_SLEEP_ARGS, default
# "--no-upsilon --no-entry --no-wcb"): sleepiness with its AME two-stage bootstrap (the E3/E4
# intervals of Table 4) -> demand prep -> logit -> RC-BLP with the draws rebuilt (--blp-args
# "--draws": the draws' demographic means come from the variant's demand parquets). The CF4
# upsilon export, the D6 entry moment and the linear E1/E2 bands are not needed for TR5 and are
# skipped.
# ==============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${HERE}/.." && pwd)"
REAL="${PROJ}/data"
UPLOAD="${TR5_UPLOAD:-${PROJ}/upload_tr5}"
ROOT="${TR5_ROOT:-${PROJ}}/tr5"
VARIANTS=(${TR5_VARIANTS:-no5 d5only})
ROUTINES="${TR5_ROUTINES:-3 4}"
SLEEP_ARGS="${TR5_SLEEP_ARGS:---no-upsilon --no-entry --no-wcb}"

MODE="${1:-}"
case "${MODE}" in
    --build-only|--submit-built|--status|--pack|--cancel) ;;
    *) echo "usage: bash tr5_variants.sh --build-only | --submit-built | --status | --pack | --cancel" >&2; exit 2 ;;
esac

jobs_of () {   # job ids whose working directory is inside the sandbox $1
    squeue -u "${USER}" -h -o "%i %Z" | awk -v d="$1/" 'index($2"/", d) == 1 {print $1}'
}

check_sandbox () {
    local sb="$1" v="$2" bad=""
    [[ -d "${sb}/scripts" && ! -L "${sb}/scripts" ]] || bad="${bad} scripts is not a plain copy;"
    [[ "$(cd "${sb}/scripts" 2>/dev/null && pwd -P)" != "$(cd "${HERE}" && pwd -P)" ]] || bad="${bad} scripts resolves to production;"
    [[ "$(cat "${sb}/scripts/config/deposit_type_variant.txt" 2>/dev/null)" == "${v}" ]] || bad="${bad} config/deposit_type_variant.txt is not '${v}';"
    [[ -f "${sb}/scripts/utils/deposit_types.py" ]] || bad="${bad} utils/deposit_types.py missing (TR5 overlay not applied);"
    grep -q -- "--sleep-args" "${sb}/scripts/pipeline_all.sh" || bad="${bad} pipeline_all.sh has no --sleep-args (TR5 overlay not applied);"
    [[ -L "${sb}/scripts/.julia_depot" ]] || bad="${bad} .julia_depot not linked;"
    [[ -L "${sb}/scripts/blp_sysimage.so" ]] || bad="${bad} blp_sysimage.so not linked;"
    [[ "$(readlink -f "${sb}/data/input")" == "$(readlink -f "${REAL}/input")" ]] || bad="${bad} data/input not linked to production;"
    [[ -d "${sb}/data/output" && ! -L "${sb}/data/output" ]] || bad="${bad} data/output is not the sandbox's own folder;"
    [[ "$(readlink -f "${sb}/data/output")" != "$(readlink -f "${REAL}/output")" ]] || bad="${bad} data/output resolves to production;"
    if [[ -n "${bad}" ]]; then echo "REFUSING ${v}:${bad}" >&2; return 1; fi
    echo "  ${v}: sandbox OK (${sb})"
}

if [[ "${MODE}" == "--build-only" ]]; then
    [[ -f "${UPLOAD}/scripts/utils/deposit_types.py" ]] || { echo "missing ${UPLOAD}/scripts (unzip the TR5 bundle first)" >&2; exit 1; }
    for v in "${VARIANTS[@]}"; do
        SB="${ROOT}/${v}"
        if [[ -f "${SB}/built.ok" ]]; then echo "  ${v}: already built"; continue; fi
        [[ ! -e "${SB}/data/output" || -z "$(ls -A "${SB}/data/output" 2>/dev/null)" ]] \
            || { echo "REFUSING ${v}: ${SB}/data/output is not empty (a previous run); move it away first" >&2; exit 2; }
        mkdir -p "${SB}/scripts/logs" "${SB}/data/output"
        find "${HERE}" -maxdepth 1 -type f ! -name '*.so' ! -name '*.zip' -exec cp -p -t "${SB}/scripts/" {} +
        for sub in config utils cluster; do
            if [[ -d "${HERE}/${sub}" ]]; then mkdir -p "${SB}/scripts/${sub}"; cp -rp "${HERE}/${sub}/." "${SB}/scripts/${sub}/"; fi
        done
        cp -rp "${UPLOAD}/scripts/." "${SB}/scripts/"
        printf '%s\n' "${v}" > "${SB}/scripts/config/deposit_type_variant.txt"
        ln -sfn "${HERE}/.julia_depot" "${SB}/scripts/.julia_depot"
        for so in "${HERE}"/*.so; do [[ -e "${so}" ]] && ln -sfn "${so}" "${SB}/scripts/$(basename "${so}")"; done
        ln -sfn "${REAL}/input" "${SB}/data/input"
        check_sandbox "${SB}" "${v}" || exit 2
        touch "${SB}/built.ok"
    done
    echo "built, nothing submitted. Submit with: bash tr5_variants.sh --submit-built"
    exit 0
fi

if [[ "${MODE}" == "--submit-built" ]]; then
    for v in "${VARIANTS[@]}"; do
        SB="${ROOT}/${v}"
        [[ -f "${SB}/built.ok" ]] || { echo "REFUSING: ${SB} was never built" >&2; exit 2; }
        check_sandbox "${SB}" "${v}" >/dev/null || exit 2
        if [[ -f "${SB}/submitted.jid" ]]; then echo "${v}: already submitted ($(cat "${SB}/submitted.jid"))"; continue; fi
        jid=$(cd "${SB}/scripts" && for x in $(compgen -v SLURM_ || true); do unset "${x}"; done \
              && env -u CL_ROOT -u DATA_ROOT -u CL_DATA_ROOT -u CL_DATA_OUT -u CL_DATA_IN \
              DEPOSIT_TYPE_VARIANT="${v}" \
              sbatch --parsable -J "tr5_${v}" -o "${SB}/scripts/logs/%x_%j.out" -e "${SB}/scripts/logs/%x_%j.err" \
                     --export=ALL,DEPOSIT_TYPE_VARIANT="${v}" \
                     pipeline_all.sh --no-bbl --no-cf --routines "${ROUTINES}" --sleep-routines "${ROUTINES}" \
                     --sleep-args "${SLEEP_ARGS}" --blp-args "--draws")
        echo "${jid}" > "${SB}/submitted.jid"
        echo "${v} -> pipeline_all job ${jid} (routines '${ROUTINES}', sleep args '${SLEEP_ARGS}')"
    done
    echo "status:  bash tr5_variants.sh --status"
    exit 0
fi

for v in "${VARIANTS[@]}"; do
    SB="${ROOT}/${v}"
    [[ -d "${SB}" ]] || { echo "${v}: no sandbox at ${SB}"; continue; }
    case "${MODE}" in
        --status)
            echo "== ${v} ($(cat "${SB}/submitted.jid" 2>/dev/null || echo 'not submitted'))"
            ids="$(jobs_of "${SB}" | tr '\n' ',')"
            [[ -n "${ids}" ]] && squeue -j "${ids%,}" -o "%.10i %.26j %.8T %.10M %R" || echo "  no job in the queue"
            for g in G0 G1 G2 G4; do
                f="${SB}/data/output/.gate_${g}.json"
                [[ -f "${f}" ]] && echo "  ${g}: $(tr -d '\n' < "${f}" | cut -c1-200)"
            done
            ;;
        --pack)
            out="${PROJ}/tr5_results_${v}.zip"
            # The sleep folder without its pickles (phi-hat is in the CSVs and the demand parquets).
            (cd "${SB}" && zip -qry "${out}" data/output/sleep data/output/demand_prep data/output/logit \
                 data/output/blp data/output/.gate_*.json scripts/logs scripts/config/deposit_type_variant.txt \
                 -x 'data/output/blp/draws/*' '*.pkl')
            echo "${v} -> ${out} ($(du -h "${out}" | cut -f1)); sha256 $(sha256sum "${out}" | cut -c1-16)"
            ;;
        --cancel)
            ids="$(jobs_of "${SB}")"
            if [[ -n "${ids}" ]]; then echo "${ids}" | xargs scancel; echo "${v}: cancelled $(echo "${ids}" | wc -w) job(s)"; else echo "${v}: nothing to cancel"; fi
            ;;
    esac
done
