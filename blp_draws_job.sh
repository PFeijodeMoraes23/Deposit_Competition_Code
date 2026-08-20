#!/bin/bash
#SBATCH --job-name=blp_draws
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/blp_draws_%j.out
#SBATCH --error=logs/blp_draws_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# ==============================================================================
# blp_draws_job.sh — the nu + demographic draw builder (blp_1_draws.jl).
#
# Its own file because a human runs it once, standalone, and its resource profile
# is unique: `day`, no GPU, 8 cpu, 64G, 2 h. Everything else in the BLP family is
# a GPU stage job.
#
# WHAT MUST EXIST FIRST
#   bash cluster_preflight.sh must pass (Julia module + Manifest agree), and the
#   reference demand parquet + demographics must be staged in data/input.
# WHAT TO RUN NEXT
#   bash blp_run.sh          (or let blp_run.sh --draws submit this for you and
#                             chain the estimation afterok on it)
#
# Submit:  sbatch blp_draws_job.sh
#          R=2000 SEED=42 SPEC=12 sbatch --export=ALL blp_draws_job.sh
#
# THIS SCRIPT IS NEW. submit_blp_1_draws.sh is still present and still works; it
# is the fallback and is retired only after one successful cluster cycle. Nothing
# in it has been modified.
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

mkdir -p "${CL_ROOT}/logs"
cl_load_julia

# ── Draw parameters: ONE source of truth for the julia call AND the split below.
R="${R:-2000}"
SEED="${SEED:-42}"
SPEC="${SPEC:-12}"
CHUNK_SIZE="${CHUNK_SIZE:-2G}"                    # per-part size for `split`
SPLIT_MIN_BYTES="${SPLIT_MIN_BYTES:-4000000000}"  # only split demo_draws above ~4 GB

# ── The shared version guard. No Pkg.resolve() here: this may run alongside other
# jobs sharing one PROJECT_DIR on NFS, and concurrent resolves corrupt
# Manifest.toml. The single resolve site is env_job.sh ENV_STEP=resolve.
if ! cl_check_julia_version; then
    echo "[!] Manifest.toml does not match the loaded Julia." >&2
    echo "    Run the resolve ONCE, alone:" >&2
    echo "      sbatch --partition=day --time=00:30:00 --cpus-per-task=4 --mem=16G \\" >&2
    echo "             --export=ALL,ENV_STEP=resolve env_job.sh" >&2
    exit 1
fi
echo "Instantiating Julia packages: $(date)"
julia --project="${CL_ROOT}" -e 'using Pkg; Pkg.instantiate()'

cl_banner "BLP draws — blp_1_draws.jl" \
          "R=${R} | seed=${SEED} | spec=${SPEC} | threads=${SLURM_CPUS_PER_TASK:-8}" \
          "node $(hostname) | $(date)"

# Draws are routine-independent; --estim 1 just selects the reference panel
# (demand_1_spec_12.parquet) for the market keys.
julia --project="${CL_ROOT}" --threads="${SLURM_CPUS_PER_TASK:-8}" \
    "${CL_ROOT}/blp_1_draws.jl" \
    --R "${R}" --seed "${SEED}" --spec "${SPEC}" --estim 1 --hpc

echo "Draws complete: $(date)"

# ── Package demo_draws for chunked download ───────────────────────────────────
# demo_draws_R{R}_seed{S}.jls is per-market Float64 randn noise (~14 GB at
# R=2000): incompressible (gzip/zstd save ~0%) and too big to move whole. Split
# it into fixed-size chunks and write a checksum so it can be pulled piecewise
# and reassembled losslessly:
#     cat demo_draws_R{R}_seed{S}.jls.part.* > demo_draws_R{R}_seed{S}.jls
#     sha256sum -c demo_draws_R{R}_seed{S}.sha256        # must print: OK
# halton_nu (KB) and demo_key_index (MB) stay whole. Idempotent: stale parts are
# cleared first, and the parts + checksum carry RELATIVE names so `sha256sum -c`
# is portable.
DRAWS_OUT="${CL_DATA_OUT}/BLP_DRAWS"      # matches blp_1_draws.jl get_paths(is_hpc=true)
DEMO_NAME="demo_draws_R${R}_seed${SEED}.jls"
DEMO="${DRAWS_OUT}/${DEMO_NAME}"
if [[ -f "${DEMO}" ]]; then
    SZ=$(stat -c%s "${DEMO}")
    if (( SZ > SPLIT_MIN_BYTES )); then
        echo "[split] ${DEMO_NAME} is ${SZ} bytes (> ${SPLIT_MIN_BYTES}); splitting into ${CHUNK_SIZE} chunks"
        ( cd "${DRAWS_OUT}" \
          && rm -f "${DEMO_NAME}.part."* \
          && split -b "${CHUNK_SIZE}" -d "${DEMO_NAME}" "${DEMO_NAME}.part." \
          && sha256sum "${DEMO_NAME}" > "demo_draws_R${R}_seed${SEED}.sha256" )
        NPARTS=$(ls "${DEMO}.part."* | wc -l)
        echo "[split] wrote ${NPARTS} parts + demo_draws_R${R}_seed${SEED}.sha256 in ${DRAWS_OUT}"
        echo "[split] reassemble on the laptop:"
        echo "        cat ${DEMO_NAME}.part.* > ${DEMO_NAME} && sha256sum -c demo_draws_R${R}_seed${SEED}.sha256"
    else
        echo "[split] ${DEMO_NAME} is ${SZ} bytes (<= ${SPLIT_MIN_BYTES}); transferable whole, no split"
    fi
else
    echo "[split] WARNING: expected ${DEMO} not found; skipping split"
fi
