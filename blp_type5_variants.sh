#!/bin/bash
# ==============================================================================
# blp_type5_variants.sh — RC-BLP robustness without the prepaid fallback (brief C8, 2026-10-21):
# the ladder head (sigma..ext1, joint SEs at ext1) of E3 and E4 on two sample variants of the
# specification-12 demand parquets, each in its own sandbox, compared with production ext1.
#   no5     every type-5 row dropped
#   d5only  type-5 rows kept for D firms only
# The variants and their logit warm starts are built locally (make_demand_type5_variants.py,
# then blp_logit.jl --demand-dir/--out-dir) and uploaded as ${UPLOAD}/<variant>/ holding
# demand_3_index_spec_12.parquet, demand_4_index_time_spec_12.parquet and logit/ (the variant's
# logit step folder).
#
# Run from scripts/ on the login node:
#   bash blp_type5_variants.sh --build-only   build and check the sandboxes, submit nothing
#   bash blp_type5_variants.sh --submit-built one GPU job per variant x routine (once each)
#   bash blp_type5_variants.sh --report       the ext1 fits side by side with production
# Sandboxes: ${T5_DIAG_ROOT:-<proj>}/diag_type5_20261021/<variant>/, with scripts/ (a copy of the
# top-level files, config/, utils/; the Julia depot and sysimages linked) and data/ (input and
# every output step folder linked, except blp/ (draws linked in), demand_prep/ and logit/, which
# are the sandbox's own and hold the variant).
# ==============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "${HERE}/.." && pwd)"
REAL="${PROJ}/data"
UPLOAD="${T5_UPLOAD:-${PROJ}/upload_type5_20261021}"
DIAG="${T5_DIAG_ROOT:-${PROJ}}/diag_type5_20261021"
VARIANTS=(no5 d5only)
ROUTINES=(3 4)
PARQ=("demand_3_index_spec_12.parquet" "demand_4_index_time_spec_12.parquet")
OWN=(blp demand_prep logit)

if [[ "${1:-}" == "--report" ]]; then
    [[ -d "${DIAG}" ]] || { echo "no ${DIAG}: nothing launched yet" >&2; exit 1; }
    python3 - "${DIAG}" "${REAL}/output/blp" <<'PY'
import json, sys
from pathlib import Path
diag, prod = Path(sys.argv[1]), Path(sys.argv[2])
print(f"{'run':16s} {'conv':5s} {'Q':>9s} {'alpha':>8s} {'se':>7s} {'n_obs':>9s}  theta2")
def row(name, f):
    if not f.is_file():
        print(f"{name:16s} (no result yet)"); return
    j = json.loads(f.read_text())
    se = (j.get("theta1_se") or [None])[0]
    print(f"{name:16s} {str(j.get('converged')):5s} {j.get('Q_value', float('nan')):9.4f} "
          f"{j['theta1'][0]:+8.3f} {se if se is None else round(se, 3)!s:>7s} "
          f"{j.get('n_obs', 0):9,d}  " + ", ".join(f"{x:+.3f}" for x in j.get("theta2", [])))
for k in (3, 4):
    row(f"E{k} production", prod / f"blp_results_E{k}_spec_12_ext1.json")
    for sb in sorted(diag.iterdir()):
        if sb.is_dir():
            row(f"E{k} {sb.name}", sb / "data" / "output" / "blp" / f"blp_results_E{k}_spec_12_ext1.json")
PY
    exit 0
fi

MODE="${1:-}"
[[ "${MODE}" == "--build-only" || "${MODE}" == "--submit-built" ]] \
    || { echo "usage: bash blp_type5_variants.sh --build-only | --submit-built | --report" >&2; exit 2; }

check_sandbox () {
    local sb="$1" bad="" d
    for d in "${OWN[@]}"; do
        [[ -d "${sb}/data/output/${d}" && ! -L "${sb}/data/output/${d}" ]] || bad="${bad} output/${d} is not a plain folder;"
        [[ "$(readlink -f "${sb}/data/output/${d}")" != "$(readlink -f "${REAL}/output/${d}")" ]] \
            || bad="${bad} output/${d} resolves to production;"
    done
    [[ "$(readlink -f "${sb}/data/output/blp/draws")" == "$(readlink -f "${REAL}/output/blp/draws")" ]] || bad="${bad} draws not linked;"
    [[ "$(readlink -f "${sb}/data/input")" == "$(readlink -f "${REAL}/input")" ]] || bad="${bad} input not linked;"
    [[ -L "${sb}/scripts/.julia_depot" ]] || bad="${bad} .julia_depot not linked;"
    [[ -L "${sb}/scripts/blp_sysimage.so" ]] || bad="${bad} blp_sysimage.so not linked;"
    [[ ! -L "${sb}/scripts" && "$(cd "${sb}/scripts" && pwd -P)" != "$(cd "${HERE}" && pwd -P)" ]] || bad="${bad} scripts is not a copy;"
    for f in "${PARQ[@]}"; do
        cmp -s "${sb}/data/output/demand_prep/${f}" "${UPLOAD}/${sb##*/}/${f}" || bad="${bad} ${f} differs from the upload;"
        [[ ! -e "${REAL}/input/${f}" ]] || bad="${bad} data/input holds ${f}, which the engine would read first;"
    done
    [[ -n "$(ls "${sb}/data/output/logit" 2>/dev/null)" ]] || bad="${bad} logit warm starts missing;"
    if [[ -n "${bad}" ]]; then echo "REFUSING ${sb##*/}:${bad}" >&2; return 1; fi
    echo "  ${sb##*/}: sandbox OK"
}

if [[ "${MODE}" == "--build-only" ]]; then
    for v in "${VARIANTS[@]}"; do
        [[ -d "${UPLOAD}/${v}/logit" ]] || { echo "missing ${UPLOAD}/${v}/logit" >&2; exit 1; }
        for f in "${PARQ[@]}"; do [[ -f "${UPLOAD}/${v}/${f}" ]] || { echo "missing ${UPLOAD}/${v}/${f}" >&2; exit 1; }; done
        SB="${DIAG}/${v}"
        if [[ -f "${SB}/built.ok" ]]; then echo "  ${v}: already built"; continue; fi
        mkdir -p "${SB}/scripts/logs" "${SB}/data/output"
        find "${HERE}" -maxdepth 1 -type f ! -name '*.so' ! -name '*.zip' -exec cp -p -t "${SB}/scripts/" {} +
        for sub in config utils; do
            if [[ -d "${HERE}/${sub}" ]]; then mkdir -p "${SB}/scripts/${sub}"; cp -rp "${HERE}/${sub}/." "${SB}/scripts/${sub}/"; fi
        done
        ln -sfn "${HERE}/.julia_depot" "${SB}/scripts/.julia_depot"
        for so in "${HERE}"/*.so; do [[ -e "${so}" ]] && ln -sfn "${so}" "${SB}/scripts/$(basename "${so}")"; done
        ln -sfn "${REAL}/input" "${SB}/data/input"
        for d in "${REAL}"/output/*; do
            b="$(basename "${d}")"
            case " ${OWN[*]} " in *" ${b} "*) continue ;; esac
            ln -sfn "${d}" "${SB}/data/output/${b}"
        done
        mkdir -p "${SB}/data/output/blp" "${SB}/data/output/demand_prep" "${SB}/data/output/logit"
        ln -sfn "${REAL}/output/blp/draws" "${SB}/data/output/blp/draws"
        for f in "${PARQ[@]}"; do cp -p "${UPLOAD}/${v}/${f}" "${SB}/data/output/demand_prep/${f}"; done
        # The rest of the production demand_prep folder (the other routines' parquets, the
        # per-parquet audits, the prep summaries) is linked, so nothing the stage reads is absent.
        for d in "${REAL}"/output/demand_prep/*; do
            b="$(basename "${d}")"
            case " ${PARQ[*]} " in *" ${b} "*) continue ;; esac
            ln -sfn "${d}" "${SB}/data/output/demand_prep/${b}"
        done
        cp -rp "${UPLOAD}/${v}/logit/." "${SB}/data/output/logit/"
        check_sandbox "${SB}" || exit 2
        touch "${SB}/built.ok"
    done
    echo "built, nothing submitted. Submit with: bash blp_type5_variants.sh --submit-built"
    exit 0
fi

for v in "${VARIANTS[@]}"; do
    SB="${DIAG}/${v}"
    [[ -f "${SB}/built.ok" ]] || { echo "REFUSING: ${SB} was never built" >&2; exit 2; }
    check_sandbox "${SB}" >/dev/null || exit 2
    for k in "${ROUTINES[@]}"; do
        if [[ -f "${SB}/submitted_E${k}.jid" ]]; then echo "${v} E${k}: already submitted ($(cat "${SB}/submitted_E${k}.jid"))"; continue; fi
        jid=$(cd "${SB}/scripts" && for x in $(compgen -v SLURM_ || true); do unset "${x}"; done \
              && env -u CL_ROOT -u DATA_ROOT -u CL_DATA_ROOT -u CL_DATA_OUT -u CL_DATA_IN \
              sbatch --parsable -J "t5v_${v}_E${k}" \
                     -o "${SB}/scripts/logs/%x_%j.out" -e "${SB}/scripts/logs/%x_%j.err" \
                     --export=ALL,RC_ROUTINE="${k}",T5_PROD_DATA="$(readlink -f "${REAL}/output")" \
                     "${SB}/scripts/blp_type5_variant_job.sh")
        echo "${jid}" > "${SB}/submitted_E${k}.jid"
        echo "${v} E${k} -> job ${jid}"
    done
done
echo "submitted. Status: squeue -u \$USER -o \"%.10i %.22j %.8T %.10M %R\" | grep t5v"
echo "results:  bash blp_type5_variants.sh --report"
