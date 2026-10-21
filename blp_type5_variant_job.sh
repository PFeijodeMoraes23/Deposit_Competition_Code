#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
# ==============================================================================
# blp_type5_variant_job.sh — the RC-BLP ladder head (sigma, rc2, rc3, rc4, ext1, with the joint
# standard errors at ext1) of ONE routine on ONE type-5 sample variant, inside a sandbox built by
# blp_type5_variants.sh: <sandbox>/scripts (a copy of the code) and <sandbox>/data (links to the
# real inputs and step folders, except its own output/blp, output/demand_prep and output/logit,
# which hold the variant). Submitted FROM <sandbox>/scripts, so every path resolves inside the
# sandbox: the engine reads and writes ../data relative to its own folder.
#
# Reads: RC_ROUTINE (3 or 4), T5_PROD_DATA (the production data/output, which this job must never
# write). Runs blp_stage_job.sh unchanged with the production settings (ift, R=2000, seed 42).
# ==============================================================================
set -euo pipefail
: "${RC_ROUTINE:?RC_ROUTINE is required}"
SBS="${SLURM_SUBMIT_DIR:?submit this job from <sandbox>/scripts}"
SB="$(cd "${SBS}/.." && pwd)"
PROD_OUT="${T5_PROD_DATA:?T5_PROD_DATA (the production data/output) is required}"

for v in CL_ROOT CL_DATA_ROOT DATA_ROOT CL_DATA_OUT CL_DATA_IN OPEN_FINANCE_ROOT SLEEP_OUT_ROOT \
         DEMAND_PREP_DIR CF_FOUNDATION_DIR CF_COST_FWD COST_POLFUNC_DIR STATE_CENTERING_JSON \
         BLP_THETA2_INIT_FILE BLP_DEMAND_PREFIX; do
    unset "${v}"
done
for v in $(compgen -v CL_STEP_ || true); do unset "${v}"; done
export CL_ROOT="${SBS}" DATA_ROOT="${SB}/data"
for d in blp demand_prep logit; do
    if [[ ! -d "${SB}/data/output/${d}" || -L "${SB}/data/output/${d}" \
          || "$(readlink -f "${SB}/data/output/${d}")" == "$(readlink -f "${PROD_OUT}/${d}")" ]]; then
        echo "REFUSING: ${SB}/data/output/${d} is not this sandbox's own folder." >&2
        exit 2
    fi
done

export RC_ENGINE=ift RC_STAGE="sigma+rc2+rc3+rc4+ext1" R=2000 SEED=42
exec bash "${SBS}/blp_stage_job.sh"
