#!/usr/bin/env bash
# Switch-cost investigation: collect the cluster results into ONE zip for download.
# COPY MODE ONLY: it reads the switch-cost folders and writes a new zip; it never moves, deletes or
# rewrites anything, and it reads no main-project folder. Run on the login node from scripts/.
#
#   bash switch_cost_collect.sh [--with-phis] [--tag NAME]
#
# Contents (paths relative to the project root):
#   data/output/switch_cost_sleep_{pooled,typeshift}/est{3,4}/{estimation_results.pkl,national_phi_t.csv}
#   data/output/switch_cost_sleep_{pooled,typeshift}/Rout/
#   data/output/switch_cost_demand_*/                 (only if the optional prep ran)
#   data/output/switch_cost_gates/
#   scripts/switch_cost_logs/
#   --with-phis adds est{3,4}/market_panel_phis.csv (about 2 GB each; the prep never reads them).
# Output: data/output/switch_cost_download/switch_cost_results_<stamp>[_<tag>].zip and its .sha256.
set -euo pipefail
WITH_PHIS=0
TAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-phis) WITH_PHIS=1 ;;
        --tag) TAG="_$2"; shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPTS}/.." && pwd)"
OUT="${ROOT}/data/output"
DL="${OUT}/switch_cost_download"
STAMP="$(date +%Y%m%d_%H%M)"
ZIP="${DL}/switch_cost_results_${STAMP}${TAG}.zip"
command -v zip >/dev/null || { echo "zip not found on PATH" >&2; exit 2; }
mkdir -p "${DL}"
[[ -e "${ZIP}" ]] && { echo "${ZIP} exists; not overwriting" >&2; exit 2; }

cd "${ROOT}"
LIST="$(mktemp)"
trap 'rm -f "${LIST}"' EXIT
for V in pooled typeshift; do
    D="data/output/switch_cost_sleep_${V}"
    [[ -d "${D}" ]] || continue
    for E in 3 4; do
        for F in estimation_results.pkl national_phi_t.csv; do
            [[ -f "${D}/est${E}/${F}" ]] && echo "${D}/est${E}/${F}" >> "${LIST}"
        done
        [[ "${WITH_PHIS}" == 1 && -f "${D}/est${E}/market_panel_phis.csv" ]] && echo "${D}/est${E}/market_panel_phis.csv" >> "${LIST}"
    done
    [[ -d "${D}/Rout" ]] && find "${D}/Rout" -type f >> "${LIST}"
done
for D in data/output/switch_cost_demand_* data/output/switch_cost_gates scripts/switch_cost_logs; do
    [[ -d "${D}" ]] && find "${D}" -type f >> "${LIST}"
done
N=$(wc -l < "${LIST}")
[[ "${N}" -gt 0 ]] || { echo "nothing to collect" >&2; exit 1; }
# float payloads stored, text deflated (as the main archive does)
zip -q -n .pkl:.parquet "${ZIP}" -@ < "${LIST}"
zip -T -q "${ZIP}"
( cd "${DL}" && sha256sum "$(basename "${ZIP}")" > "$(basename "${ZIP}").sha256" )
echo "collected ${N} files -> ${ZIP} ($(du -h "${ZIP}" | cut -f1))"
