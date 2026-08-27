#!/bin/bash
#SBATCH --job-name=blp_draws
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# No --output/--error here: see env_job.sh. blp_run.sh passes -o/-e explicitly, and it
# calls cl_log_dir first, so the directory exists by then.
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
#   bash cluster_preflight.sh must pass (Julia module + Manifest agree), plus its
#   TWO inputs, which come from opposite halves of the tree:
#     data/output/demand_prep/demand_1_spec_12.parquet   the reference panel, built
#         by the sleepiness prep step and verified by gate G2 — cluster-produced,
#         never uploaded.
#     data/input/demographics_sigma.parquet              an UPLOAD. Building it needs
#         seven muni panels (ANATEL alone is 1.59 GB), so producing 8.7 MB here would
#         cost a 1.6 GB upload; panel_8_demographics_sigma.py stays local.
# WHAT TO RUN NEXT
#   bash blp_run.sh          (which submits this itself when the draws are absent,
#                             and chains the estimation afterok on it)
#
# Submit:  sbatch blp_draws_job.sh
#          R=2000 SEED=42 SPEC=12 sbatch --export=ALL blp_draws_job.sh
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

mkdir -p "${CL_ROOT}/logs"

# The skeleton, then the step dirs. cl_export_step_dirs creates data/output/blp/draws,
# which is where blp_1_draws.jl writes under --hpc (of_root.jl's draws_dir), and the
# split block below reads back through the same CL_STEP_DRAWS.
cl_bootstrap_tree
cl_export_step_dirs

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
# The one place the draws live on the cluster: CL_STEP_DRAWS is data/output/blp/draws,
# the same path of_root.jl's draws_dir(out_dir) hands blp_1_draws.jl under --hpc, so the
# split below cannot address a directory the writer never used.
DRAWS_OUT="${CL_STEP_DRAWS}"
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
