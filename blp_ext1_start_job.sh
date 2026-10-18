#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=06:00:00
# ==============================================================================
# blp_ext1_start_job.sh — ONE ext1 fit of one routine from one explicit starting point, inside a
# sandbox built by blp_ext1_starts.sh: <sandbox>/scripts (a copy of the code) and <sandbox>/data
# (links to the real inputs, its own output/blp). Submitted FROM <sandbox>/scripts, so
# SLURM_SUBMIT_DIR is the sandbox's code and every path below resolves inside the sandbox: the
# engine writes to ../data relative to its own folder, never to the production data/output/blp.
#
# Reads: RC_ROUTINE (3 or 4) and EXT1_START (the five ext1 theta2 values, colon-separated:
# sigma_spread, pi spread x gdp, pi spread x 65+, pi spread x mobile, pi assets x gdp).
# Writes the starting point as <sandbox>/init_theta2.jls (BLP_THETA2_INIT_FILE, which replaces
# ext1's two built-in starts), then runs blp_stage_job.sh unchanged with the production settings.
# ==============================================================================
set -euo pipefail
: "${RC_ROUTINE:?RC_ROUTINE is required}" "${EXT1_START:?EXT1_START is required}"
SBS="${SLURM_SUBMIT_DIR:?submit this job from <sandbox>/scripts}"
SB="$(cd "${SBS}/.." && pwd)"

# Every path variable a parent shell may have exported points at the production tree: drop them
# all, then name the sandbox's own code and data.
for v in CL_ROOT CL_DATA_ROOT DATA_ROOT CL_DATA_OUT CL_DATA_IN OPEN_FINANCE_ROOT SLEEP_OUT_ROOT \
         DEMAND_PREP_DIR CF_FOUNDATION_DIR CF_COST_FWD COST_POLFUNC_DIR STATE_CENTERING_JSON; do
    unset "${v}"
done
for v in $(compgen -v CL_STEP_ || true); do unset "${v}"; done
export CL_ROOT="${SBS}" DATA_ROOT="${SB}/data"
PROD_BLP="$(readlink -f "${SB}/../../data/output/blp" 2>/dev/null || true)"
if [[ ! -d "${SB}/data/output/blp" || -L "${SB}/data/output/blp" \
      || "$(readlink -f "${SB}/data/output/blp")" == "${PROD_BLP}" ]]; then
    echo "REFUSING: ${SB}/data/output/blp is not this sandbox's own folder." >&2
    exit 2
fi

. "${SBS}/cluster_lib.sh"
cl_load_julia
export BLP_THETA2_INIT_FILE="${SB}/init_theta2.jls"
julia --startup-file=no -e 'using Serialization
v = parse.(Float64, split(ENV["EXT1_START"], ":"))
length(v) == 5 || error("EXT1_START must hold 5 values, got $(length(v))")
serialize(ENV["BLP_THETA2_INIT_FILE"], Dict{String,Any}("theta2_star" => v))
println("ext1 start theta2 = ", v, " -> ", ENV["BLP_THETA2_INIT_FILE"])'

export RC_ENGINE=ift RC_STAGE=ext1 R=2000 SEED=42
exec bash "${SBS}/blp_stage_job.sh"
