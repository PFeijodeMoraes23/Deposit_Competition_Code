#!/bin/bash
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# ==============================================================================
# w2_job.sh -- the payload of the W2 driver (sleep_attention_stakes.py) and of the Pix
# generated-regressor bootstrap (sleep_pix_genreg.py). Same environment as sleep_job.sh: the
# Open-Finance skeleton (cl_bootstrap_tree), the step dirs (cl_export_step_dirs: SLEEP_OUT_ROOT
# is data/output/sleep, so both drivers write under data/output/sleep/Rout), the conda env.
#
#   W2_SCRIPT   sleep_attention_stakes.py | sleep_pix_genreg.py
#   W2_ARGS     the driver's flags. In an array job the token @TASK@ becomes the task's draw range,
#               W2_CHUNK draws per task (default 9): task i -> (i-1)*CHUNK+1 .. i*CHUNK.
#   W2_INPUTS_DIR  where sleep_attention_stakes.py reads w2_inputs.parquet, w2_draws.parquet and
#               w2_prep.json (default data/input/attention_stakes, the upload's destination).
#   SLEEP_PIX_STATE  passed through (utils/pix_state.py); printed in the banner.
#
# Submitted with sbatch --export=ALL,W2_SCRIPT=...,W2_ARGS=... (see the package in the W2 note).
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

: "${W2_SCRIPT:?set W2_SCRIPT (sleep_attention_stakes.py | sleep_pix_genreg.py)}"
: "${W2_ARGS:?set W2_ARGS}"
case "${W2_SCRIPT}" in
    # sleep_demand_prep_link.py: the attention branch's prep (DEMAND_PHI_SOURCE=attention), run
    # alone in its own data root (DATA_ROOT); see the W2 note's package, step 7.
    sleep_attention_stakes.py|sleep_pix_genreg.py|sleep_demand_prep_link.py) ;;
    *) echo "W2_SCRIPT=${W2_SCRIPT}: not a W2 driver" >&2; exit 2 ;;
esac

mkdir -p "${CL_ROOT}/logs"
cl_bootstrap_tree
cl_export_step_dirs
export OPEN_FINANCE_VENV_ENFORCED=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
export W2_INPUTS_DIR="${W2_INPUTS_DIR:-${CL_DATA_IN}/attention_stakes}"
cl_setup_python "${CL_PY_REQ_SLEEP}"
cd "${CL_ROOT}"

ARGS="${W2_ARGS}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    CH="${W2_CHUNK:-9}"
    LO=$(( (SLURM_ARRAY_TASK_ID - 1) * CH + 1 ))
    HI=$(( SLURM_ARRAY_TASK_ID * CH ))
    ARGS="${ARGS//@TASK@/${LO}-${HI}}"
fi

cl_banner "W2 / Pix genreg | ${W2_SCRIPT} ${ARGS}" \
          "SLEEP_OUT_ROOT    = ${SLEEP_OUT_ROOT}" \
          "SLEEP_PIX_STATE   = ${SLEEP_PIX_STATE:-national}" \
          "W2_INPUTS_DIR     = ${W2_INPUTS_DIR}" \
          "cpus=${SLURM_CPUS_PER_TASK:-1} node=$(hostname) | $(date)"

"${PYBIN}" "${CL_ROOT}/${W2_SCRIPT}" ${ARGS}
echo "W2 job complete: $(date)"
