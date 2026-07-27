#!/bin/bash
#SBATCH --job-name=blp_1_draws
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_1_draws_%j.out
#SBATCH --error=/nfs/roberts/project/pi_mf2263/pf382/dep_comp/scripts/logs/blp_1_draws_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu

# ── Environment ──────────────────────────────────────────────────────────────
# module reset is the correct command on Bouchet (module purge cannot unload StdEnv)
module reset
module load Julia/1.11.4-linux-x86_64

# Read the reference panel (demand_1_spec_12.parquet) for market keys, matching the
# RC-BLP estimation inputs. Draws are routine-independent.

# Strict error checking starts after module loading to avoid false failures
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ── Draw parameters (single source of truth: the julia call AND the post-run split use these) ──
R="${R:-2000}"
SEED="${SEED:-42}"
SPEC="${SPEC:-12}"
# Post-run packaging of the huge demo_draws file for chunked download (see split block at EOF).
CHUNK_SIZE="${CHUNK_SIZE:-2G}"                     # per-part size for `split`
SPLIT_MIN_BYTES="${SPLIT_MIN_BYTES:-4000000000}"  # only split demo_draws if larger than ~4 GB

# ── Resolve packages for Julia 1.10 ─────────────────────────────────────────
if [ -f "${PROJECT_DIR}/Manifest.toml" ] && ! grep -q 'julia_version = "1.10' "${PROJECT_DIR}/Manifest.toml"; then
    echo "Removing incompatible Manifest.toml before instantiate: $(date)"
    rm -f "${PROJECT_DIR}/Manifest.toml"
fi
echo "Resolving Julia packages: $(date)"
julia --project="${PROJECT_DIR}" -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"

echo "======================================"
echo " BLP Draws — blp_1_draws.jl"
echo " R=${R} | seed=${SEED} | spec=${SPEC} | $(date)"
echo "======================================"

julia --project="${PROJECT_DIR}" --threads=${SLURM_CPUS_PER_TASK} \
    "${PROJECT_DIR}/blp_1_draws.jl" \
    --R "${R}" \
    --seed "${SEED}" \
    --spec "${SPEC}" \
    --estim 1 \
    --hpc

echo "Draws complete: $(date)"

# ── Package demo_draws for chunked download ─────────────────────────────────────────────────
# demo_draws_R{R}_seed{S}.jls is per-market Float64 randn noise (~14 GB at R=2000): incompressible
# (gzip/zstd save ~0%) and usually too big for a single scp/rsync. Split it into fixed-size chunks
# and write a checksum so it can be pulled piecewise and reassembled losslessly on the laptop:
#     cat demo_draws_R{R}_seed{S}.jls.part.* > demo_draws_R{R}_seed{S}.jls
#     sha256sum -c demo_draws_R{R}_seed{S}.sha256        # must print: OK
# halton_nu (KB) and demo_key_index (MB) stay whole. The split is idempotent (stale parts cleared).
DRAWS_OUT="${PROJECT_DIR}/../data/output/BLP_DRAWS"       # matches blp_1_draws.jl get_paths(is_hpc=true)
DEMO_NAME="demo_draws_R${R}_seed${SEED}.jls"
DEMO="${DRAWS_OUT}/${DEMO_NAME}"
if [[ -f "${DEMO}" ]]; then
    SZ=$(stat -c%s "${DEMO}")
    if (( SZ > SPLIT_MIN_BYTES )); then
        echo "[split] ${DEMO_NAME} is ${SZ} bytes (> ${SPLIT_MIN_BYTES}); splitting into ${CHUNK_SIZE} chunks"
        # Run in the draws dir so parts + checksum carry RELATIVE names → `sha256sum -c` is portable.
        ( cd "${DRAWS_OUT}" \
          && rm -f "${DEMO_NAME}.part."* \
          && split -b "${CHUNK_SIZE}" -d "${DEMO_NAME}" "${DEMO_NAME}.part." \
          && sha256sum "${DEMO_NAME}" > "demo_draws_R${R}_seed${SEED}.sha256" )
        NPARTS=$(ls "${DEMO}.part."* | wc -l)
        echo "[split] wrote ${NPARTS} parts + demo_draws_R${R}_seed${SEED}.sha256 in ${DRAWS_OUT}"
        echo "[split] reassemble on the laptop:"
        echo "        cat ${DEMO_NAME}.part.* > ${DEMO_NAME} && sha256sum -c demo_draws_R${R}_seed${SEED}.sha256"
    else
        echo "[split] ${DEMO_NAME} is ${SZ} bytes (<= ${SPLIT_MIN_BYTES}); small enough to transfer whole, no split"
    fi
else
    echo "[split] WARNING: expected ${DEMO} not found; skipping split"
fi
