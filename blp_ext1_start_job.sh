#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=06:00:00
# ==============================================================================
# blp_ext1_start_job.sh — ONE fit of one routine's ext1 (or, with EXT_STAGE, ext2 or extended) from
# one explicit starting point, inside a sandbox built by blp_ext1_starts.sh: <sandbox>/scripts (a
# copy of the code) and <sandbox>/data (links to the real inputs, its own output/blp). Submitted
# FROM <sandbox>/scripts, so SLURM_SUBMIT_DIR is the sandbox's code and every path below resolves
# inside the sandbox: the engine writes to ../data relative to its own folder, never to the
# production data/output/blp.
#
# Reads: RC_ROUTINE (3 or 4), EXT_STAGE (ext1 | ext2 | extended; default ext1), EXT1_PROD_BLP (the
# production data/output/blp, which this job must never write) and EXT1_START (the rung's theta2
# values, colon-separated; ext1 takes 5: sigma_spread, pi spread x gdp, pi spread x 65+,
# pi spread x mobile, pi assets x gdp; ext2 adds pi fgc x 65+, extended then pi fgc x cadunico).
# Writes the starting point as <sandbox>/init_theta2.jls (BLP_THETA2_INIT_FILE, which replaces
# the rung's built-in starts), then runs blp_stage_job.sh unchanged with the production settings.
# The fit's delta starts where the engine looks first, data/input/logit_delta_E<k>_spec_12.bin:
# the logit delta-hat, or the checkpoint delta* the build copied there (<sandbox>/delta_start.txt),
# whose hash is checked here.
# ==============================================================================
set -euo pipefail
: "${RC_ROUTINE:?RC_ROUTINE is required}" "${EXT1_START:?EXT1_START is required}"
STAGE="${EXT_STAGE:-ext1}"
case "${STAGE}" in
    ext1)     NPAR=5 ;;
    ext2)     NPAR=6 ;;
    extended) NPAR=7 ;;
    *) echo "EXT_STAGE='${STAGE}' must be ext1, ext2 or extended" >&2; exit 2 ;;
esac
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
PROD_BLP="${EXT1_PROD_BLP:?EXT1_PROD_BLP (the production data/output/blp) is required}"
if [[ ! -d "${SB}/data/output/blp" || -L "${SB}/data/output/blp" \
      || "$(readlink -f "${SB}/data/output/blp")" == "${PROD_BLP}" ]]; then
    echo "REFUSING: ${SB}/data/output/blp is not this sandbox's own folder." >&2
    exit 2
fi
if [[ -f "${SB}/delta_start.txt" ]]; then
    cat "${SB}/delta_start.txt"
    sha256sum -c --quiet "${SB}/delta_start.sha256" \
        || { echo "REFUSING: the delta start in ${SB}/data/input differs from the one the build copied." >&2; exit 2; }
fi

. "${SBS}/cluster_lib.sh"
cl_load_julia
export BLP_THETA2_INIT_FILE="${SB}/init_theta2.jls"
export EXT_STAGE="${STAGE}" EXT_NPAR="${NPAR}"
julia --startup-file=no -e 'using Serialization
v = parse.(Float64, split(ENV["EXT1_START"], ":"))
n = parse(Int, ENV["EXT_NPAR"])
length(v) == n || error("EXT1_START must hold $n values, got $(length(v))")
serialize(ENV["BLP_THETA2_INIT_FILE"], Dict{String,Any}("theta2_star" => v))
println(ENV["EXT_STAGE"], " start theta2 = ", v, " -> ", ENV["BLP_THETA2_INIT_FILE"])'

export RC_ENGINE=ift RC_STAGE="${STAGE}" R=2000 SEED=42
exec bash "${SBS}/blp_stage_job.sh"
