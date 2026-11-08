#!/bin/bash
#SBATCH --job-name=k1
#SBATCH --partition=day
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/k1_%A_%a.out
#SBATCH --error=logs/k1_%A_%a.err
# ==============================================================================
# kappa/k1_job.sh -- K1 real-data fits on Bouchet (kappa/k1_netflow_nls.py), one array task per
# invocation, each writing its own folder data/output/k1/task_<id>/ (k1_estimates.csv,
# k1_numbers.json). Merge them locally with kappa/k1_merge.py after the download.
#
# Submit from scripts/:   sbatch --parsable --array=0-5 kappa/k1_job.sh      (6 and 7 optional)
#
# Tasks (the headline form is --exit-by-cell --trim p99, Pedro 11-06; 4 starts, cap 500 by default):
#   0  before_after                      (headline)
#   1  dose
#   2  before_after+carried_kept
#   3  before_after+zero_premium
#   4  before_after+no_negative_adv
#   5  before_after --lambda-one         (D2; the time-deposit pair and ln E_c)
#   6  before_after --trim none          (untrimmed, the robustness row)
#   7  before_after, theta_w constant    (no --exit-by-cell: the failed check of reply d)
#
# Inputs:
#   data/input/market_panel.parquet (fa38794d) and data/input/demand_chars.parquet (its side file)
#   data/output/demand_prep/ (I2's frames on fa38794d; read when the context is rebuilt)
#   data/input/k1/Rout/  k0_omega.csv k0_numbers.json p1_pix_dose_exposure_mca.csv
#                        p2_advertising_conglomerate_quarter.csv p2_advertising_coverage_B.csv
#                        p3_stakes_mca_quarter.csv          (paths.rout_dir() via SLEEP_OUT_ROOT)
#   data/input/k1/k1_ctx_fa38794d.pkl  only with K1_CTX_MODE=pickle (the local pandas is 3.0.2: use
#                        it only if the cluster env's pandas is 3.0.x; otherwise the default rebuild)
# ==============================================================================
set -euo pipefail

CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
mkdir -p "${CL_ROOT}/logs"

cl_bootstrap_tree
cl_export_step_dirs
# paths.rout_dir() -> data/input/k1/Rout, where the K0 and P1-P3 inputs were uploaded. K1 writes
# only to --out; nothing is written under data/input.
export SLEEP_OUT_ROOT="${CL_DATA_IN}/k1"
export DEMAND_CHARS_FILE="${CL_DATA_IN}/demand_chars.parquet"
export K0_PANEL_MD5=fa38794d
export OPEN_FINANCE_VENV_ENFORCED=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}" OPENBLAS_NUM_THREADS="${NT}"

cl_setup_python "numpy, pandas, scipy, pyarrow, statsmodels"
cd "${CL_ROOT}"

ID="${SLURM_ARRAY_TASK_ID:?run as an array: sbatch --array=0-5 kappa/k1_job.sh}"
HEAD="--exit-by-cell --trim p99"
case "${ID}" in
    0) SPEC="before_after";                  OPTS="${HEAD}" ;;
    1) SPEC="dose";                          OPTS="${HEAD}" ;;
    2) SPEC="before_after+carried_kept";     OPTS="${HEAD}" ;;
    3) SPEC="before_after+zero_premium";     OPTS="${HEAD}" ;;
    4) SPEC="before_after+no_negative_adv";  OPTS="${HEAD}" ;;
    5) SPEC="before_after";                  OPTS="${HEAD} --lambda-one" ;;
    6) SPEC="before_after";                  OPTS="--exit-by-cell --trim none" ;;
    7) SPEC="before_after";                  OPTS="--trim p99" ;;
    *) echo "unknown K1 task ${ID}" >&2; exit 2 ;;
esac
OUT="${CL_DATA_OUT}/k1/task_${ID}"
mkdir -p "${OUT}"
MODE="${K1_CTX_MODE:-rebuild}"
CTX_ARGS=()
if [[ "${MODE}" == "pickle" ]]; then
    CTX_ARGS=(--ctx "${CL_DATA_IN}/k1/k1_ctx_fa38794d.pkl")
fi

echo "K1 task ${ID}: spec=${SPEC} opts=${OPTS} ctx_mode=${MODE} out=${OUT}"
echo "K1 task ${ID}: host=$(hostname) start=$(date '+%F %T') python=$(command -v "${PYBIN}")"
echo "K1 task ${ID}: DEMAND_PREP_DIR=${DEMAND_PREP_DIR} SLEEP_OUT_ROOT=${SLEEP_OUT_ROOT} OPEN_FINANCE_ROOT=${OPEN_FINANCE_ROOT}"
"${PYBIN}" -I -u kappa/k1_netflow_nls.py --spec "${SPEC}" ${OPTS} ${CTX_ARGS[@]+"${CTX_ARGS[@]}"} \
    --panel "${CL_DATA_IN}/market_panel.parquet" --out "${OUT}"
echo "K1 task ${ID}: end=$(date '+%F %T')"
