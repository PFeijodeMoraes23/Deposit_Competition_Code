#!/bin/bash
#SBATCH --job-name=env_job
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/env_%x_%j.out
#SBATCH --error=logs/env_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# ==============================================================================
# env_job.sh — ONE toolchain job. Dispatches on ENV_STEP:
#     resolve       the SOLE Pkg.resolve() site: rebuild Manifest.toml for the
#                   Julia this cluster actually has. Run ALONE, never in an array,
#                   never inside a chain (concurrent resolves on NFS corrupt it).
#     sysimage_gpu  build blp_sysimage.so     (run on gpu_h200 WITH a GPU)
#     sysimage_cpu  build blp_sysimage_cpu.so (run on day, --constraint=cpugen:turin)
#
# The #SBATCH block above is a SAFE FLOOR only. Resources arrive as sbatch CLI
# flags, which override these directives:
#   sbatch --partition=day --time=00:30:00 --cpus-per-task=4 --mem=16G \
#          --export=ALL,ENV_STEP=resolve env_job.sh
#   sbatch --partition=gpu_h200 --gpus=h200:1 --time=04:00:00 --cpus-per-task=8 \
#          --mem=64G --export=ALL,ENV_STEP=sysimage_gpu env_job.sh
#   sbatch --partition=day --constraint=cpugen:turin --time=04:00:00 \
#          --cpus-per-task=8 --mem=64G --export=ALL,ENV_STEP=sysimage_cpu env_job.sh
#
# WHAT MUST EXIST FIRST: bash cluster_preflight.sh, and act on what it says.
# WHAT TO RUN NEXT:      re-run cluster_preflight.sh; then blp_run.sh.
#
# A sysimage build ends in a FATAL LOAD PROOF and writes a provenance sidecar
# <img>.so.json. A build whose proof fails exits non-zero and the .so is renamed
# .so.failed, so a broken image never sits on disk looking done.
#
# THIS SCRIPT IS NEW. setup_julia_env.sh / submit_build_sysimage.sh /
# submit_build_sysimage_cpu.sh are still present and still work; they are the
# fallback and are retired only after one successful cluster cycle. Nothing in
# them has been modified.
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

: "${ENV_STEP:?set ENV_STEP (resolve|sysimage_gpu|sysimage_cpu)}"
case "${ENV_STEP}" in resolve|sysimage_gpu|sysimage_cpu) ;;
    *) echo "ENV_STEP='${ENV_STEP}' is not recognised (resolve|sysimage_gpu|sysimage_cpu)" >&2; exit 2 ;;
esac

mkdir -p "${CL_ROOT}/logs"
cl_load_julia
NPROC="${SLURM_CPUS_PER_TASK:-4}"

cl_banner "env_job: ENV_STEP=${ENV_STEP}" \
          "node    $(hostname)" \
          "julia   $(julia --version)" \
          "module  ${JULIA_MODULE}" \
          "depot   ${JULIA_DEPOT_PATH%%:*}" \
          "date    $(date)"

# ── The build/load workload lives in build_blp_sysimage.jl; nothing here edits it.
build_one () {   # build_one gpu|cpu
    local tgt="$1" img side cpu_only=0 rc
    img="$(cl_sysimage_for "${tgt}")"; side="${img}.json"
    [[ "${tgt}" == "cpu" ]] && cpu_only=1

    echo "CPU model: $(lscpu 2>/dev/null | grep -m1 'Model name' | cut -d: -f2 | xargs || echo '?')"
    if [[ "${tgt}" == "gpu" ]]; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
            echo "ERROR: ENV_STEP=sysimage_gpu but this node has no GPU." >&2
            echo "  Submit it with --partition=gpu_h200 --gpus=h200:1." >&2
            exit 2; }
    fi

    # Instantiate only. Pkg.resolve() belongs to ENV_STEP=resolve and nowhere else.
    julia --project="${CL_ROOT}" -e 'using Pkg; Pkg.instantiate()'

    rm -f "${img}.failed" "${side}"
    echo "── building $(basename "${img}") : $(date) ──"
    BLP_SYSIMAGE_CPU="${cpu_only}" BLP_SYSIMAGE_WORKLOAD="${BLP_SYSIMAGE_WORKLOAD:-0}" \
        julia --project="${CL_ROOT}" --threads="${NPROC}" "${CL_ROOT}/build_blp_sysimage.jl"
    [[ -f "${img}" ]] || { echo "ERROR: ${img} was not produced — see the log above." >&2; exit 1; }
    ls -la "${img}"

    # ── FATAL LOAD PROOF on THIS node type — the whole reason for building here.
    echo "── load proof : $(date) ──"
    set +e
    if [[ "${tgt}" == "gpu" ]]; then
        CPUNAME="$(julia --project="${CL_ROOT}" --sysimage "${img}" -e '
            using CUDA
            CUDA.functional() || (println(stderr, "CUDA.functional() == false"); exit(3))
            print(Sys.CPU_NAME)' 2>&1)"
    else
        CPUNAME="$(julia --project="${CL_ROOT}" --sysimage "${img}" -e 'print(Sys.CPU_NAME)' 2>&1)"
    fi
    rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
        mv -f "${img}" "${img}.failed"
        echo "ERROR: the load proof FAILED — image renamed to $(basename "${img}").failed" >&2
        echo "  proof output: ${CPUNAME}" >&2
        echo "  A broken image must never sit on disk looking done: every consumer would" >&2
        echo "  accept it and then degrade (a GPU job to CPU at ~0% util)." >&2
        exit 1
    fi
    echo "  loads OK on $(hostname); Sys.CPU_NAME=${CPUNAME}"

    # ── provenance sidecar: flat one-key-per-line JSON, read by cl_sysimage_verdict.
    cat > "${side}" <<EOF
{
  "julia_version": "$(cl_loaded_julia_version)",
  "julia_module": "${JULIA_MODULE}",
  "cpu_target": "${CPUNAME}",
  "build_host": "$(hostname)",
  "partition": "${SLURM_JOB_PARTITION:-unknown}",
  "manifest_sha256": "$(cl_manifest_sha)",
  "built_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "── provenance sidecar $(basename "${side}") ──"
    cat "${side}"
}

case "${ENV_STEP}" in

resolve)
    # ── THE ONE Pkg.resolve() SITE ────────────────────────────────────────────
    # The accepted version is derived from the LOADED julia, not from a hardcoded
    # '1.11' literal, so this keeps working whatever module the cluster offers.
    ACCEPT="$(julia -e 'print(VERSION.major, ".", VERSION.minor)')"
    MF="${CL_ROOT}/Manifest.toml"
    echo "Accepted manifest julia_version prefix: ${ACCEPT}"
    if [[ -f "${MF}" ]]; then
        MV="$(cl_manifest_julia_version || echo unknown)"
        echo "Found Manifest.toml (julia_version = ${MV})"
        if [[ "${MV}" == ${ACCEPT}.* || "${MV}" == "${ACCEPT}" ]]; then
            echo "  compatible with ${ACCEPT} — keeping; this run only instantiates + precompiles."
        else
            cp -f "${MF}" "${MF}.bak.${MV}"
            echo "  incompatible with ${ACCEPT} — backed up to $(basename "${MF}").bak.${MV}, removing before the resolve."
            rm -f "${MF}"
        fi
    else
        echo "No Manifest.toml — the resolve will create one."
    fi

    echo ""
    echo "=== Resolving packages: $(date) ==="
    julia --project="${CL_ROOT}" -e '
        using Pkg
        println("  Updating package registry...");  Pkg.Registry.update()
        Pkg.resolve();      println("  Manifest written.")
        Pkg.instantiate();  println("  Packages installed.")
        Pkg.precompile();   println("  Precompilation done.")
    '

    echo ""
    echo "=== Smoke test: $(date) ==="
    julia --project="${CL_ROOT}" -e '
        using MKL, LoopVectorization, LinearAlgebra, DataFrames, Optim, ArgParse
        println("  All key packages loaded OK.")
        println("  BLAS: ", BLAS.get_config())
        println("  Threads: ", Threads.nthreads())
    '

    # CUDA smoke test. Submit this step with --gpus if you want it to mean
    # anything: on a GPU-less node it can only ever print "not functional".
    echo ""
    echo "=== CUDA smoke test: $(date) ==="
    if [[ -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]]; then
        julia --project="${CL_ROOT}" -e '
            using CUDA
            if CUDA.functional()
                println("  CUDA functional: ", CUDA.name(CUDA.device()))
                println("  VRAM: ", round(CUDA.totalmem(CUDA.device())/2^30, digits=1), " GB")
                x = CUDA.zeros(Float32, 128, 128); y = x .* 2f0; CUDA.synchronize()
                println("  Kernel smoke test passed.")
            else
                println("  CUDA NOT functional although a GPU was allocated — investigate.")
            end'
    else
        echo "  no GPU allocated to this job — skipped."
        echo "  To make it meaningful, add --gpus=h200:1 --partition=gpu_h200 to the sbatch line."
    fi

    echo ""
    echo "Manifest.toml now records: $(cl_manifest_julia_version)"
    echo "NEXT: bash cluster_preflight.sh   (it should now print 'RESOLVE REQUIRED: no')"
    ;;

sysimage_gpu) build_one gpu ;;
sysimage_cpu) build_one cpu ;;

esac

echo "env_job ENV_STEP=${ENV_STEP} complete: $(date)"
