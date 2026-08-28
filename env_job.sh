#!/bin/bash
#SBATCH --job-name=env_job
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# No --output/--error here on purpose. SLURM will not create a missing directory, and a
# job whose log path cannot be opened is killed before it runs -- one second, no log, no
# clue. This script is submitted BY HAND on a fresh tree, where scripts/logs/ may not
# exist yet, so it falls back to slurm-<jobid>.out in the submit directory, which always
# does. pipeline_all.sh passes -o/-e explicitly when it submits this as the G0 job.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# ==============================================================================
# env_job.sh — ONE toolchain job. Dispatches on ENV_STEP:
#     preflight     GATE G0: build the tree, load Julia, run cluster_preflight.sh
#                   and write data/output/.gate_G0.json. This is the ONLY thing
#                   pipeline_all.sh's first phase waits on.
#     resolve       the SOLE Pkg.resolve() site: rebuild Manifest.toml for the
#                   Julia this cluster actually has. Run ALONE, never in an array,
#                   never inside a chain (concurrent resolves on NFS corrupt it).
#     sysimage_gpu  build blp_sysimage.so     (run on gpu_h200 WITH a GPU)
#     sysimage_cpu  build blp_sysimage_cpu.so (run on day, --constraint=cpugen:turin)
#
# WHY THE PREFLIGHT IS A JOB. It loads a Julia module, probes both sysimages by
# LOADING them, walks the upload manifest and imports the Python stack. None of that
# belongs on the login node — the shared login host has a hard memory cap and a
# different CPU target, so the checks are both antisocial and less reliable there.
# Making it a job is also what lets pipeline_all.sh be a pure submitter: it hangs
# every first-tier submission off this job id, and --kill-on-invalid-dep turns a
# blocking verdict into "nothing downstream ever starts".
#
# The #SBATCH block above is a SAFE FLOOR only. Resources arrive as sbatch CLI
# flags, which override these directives:
#   sbatch --partition=day --time=00:30:00 --cpus-per-task=2 --mem=8G \
#          --export=ALL,ENV_STEP=preflight env_job.sh
#   sbatch --partition=day --time=00:30:00 --cpus-per-task=4 --mem=16G \
#          --export=ALL,ENV_STEP=resolve env_job.sh
#   sbatch --partition=gpu_h200 --gpus=h200:1 --time=04:00:00 --cpus-per-task=8 \
#          --mem=64G --export=ALL,ENV_STEP=sysimage_gpu env_job.sh
#   sbatch --partition=day --constraint=cpugen:turin --time=04:00:00 \
#          --cpus-per-task=8 --mem=64G --export=ALL,ENV_STEP=sysimage_cpu env_job.sh
#
# Env vars read by ENV_STEP=preflight:
#   PF_ROUTINES    routine set to check (default '1 2 3 4')
#   PF_WILL_BUILD  space-separated artifacts THIS run creates before anything
#                  consumes them ('sysimage', 'draws') — reported, not blocking
#
# WHAT MUST EXIST FIRST: the uploaded code + data bundles, nothing else.
# WHAT TO RUN NEXT:      whatever .gate_G0.json / the log says; then
#                        bash pipeline_all.sh, which submits this step itself.
#
# A sysimage build ends in a FATAL LOAD PROOF and writes a provenance sidecar
# <img>.so.json. A build whose proof fails exits non-zero and the .so is renamed
# .so.failed, so a broken image never sits on disk looking done.
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

: "${ENV_STEP:?set ENV_STEP (preflight|resolve|sysimage_gpu|sysimage_cpu)}"
case "${ENV_STEP}" in preflight|resolve|sysimage_gpu|sysimage_cpu) ;;
    *) echo "ENV_STEP='${ENV_STEP}' is not recognised (preflight|resolve|sysimage_gpu|sysimage_cpu)" >&2; exit 2 ;;
esac

mkdir -p "${CL_ROOT}/logs"
# The skeleton and the step dirs, on every branch. G0 is the first job of a fresh
# cluster, so this is where data/output and its eight step folders come into
# existence — the preflight below then CHECKS them rather than creating them.
cl_bootstrap_tree
cl_export_step_dirs

# A module that does not load is FATAL for the three build steps — they have nothing
# to do without Julia — but it is the preflight's whole job to REPORT that, in the
# gate json, rather than die before writing one. So the load is captured here and the
# refusal is per step.
JULIA_LOADED=1
cl_load_julia || JULIA_LOADED=0
if [[ "${JULIA_LOADED}" != "1" && "${ENV_STEP}" != "preflight" ]]; then
    echo "ERROR: ENV_STEP=${ENV_STEP} needs Julia and '${JULIA_MODULE}' did not load (candidates above)." >&2
    exit 2
fi
NPROC="${SLURM_CPUS_PER_TASK:-4}"

cl_banner "env_job: ENV_STEP=${ENV_STEP}" \
          "node    $(hostname)" \
          "julia   $(julia --version 2>/dev/null || echo '<module did not load>')" \
          "module  ${JULIA_MODULE}" \
          "depot   ${JULIA_DEPOT_PATH%%:*}" \
          "date    $(date)"

# ── The build/load workload lives in blp_build_sysimage.jl; nothing here edits it.
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
        julia --project="${CL_ROOT}" --threads="${NPROC}" "${CL_ROOT}/blp_build_sysimage.jl"
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

preflight)
    # ── GATE G0 ───────────────────────────────────────────────────────────────
    # Two products, and the order matters: a HUMAN log (cluster_preflight.sh's own
    # numbered verdict block, verbatim) and a MACHINE verdict, .gate_G0.json, which
    # is written whatever the outcome — a gate that fails without leaving a json is
    # a gate nobody can read after the fact, and every downstream job of a failed G0
    # is cancelled as DependencyNeverSatisfied and writes no log of its own.
    PF_ROUTINES="${PF_ROUTINES:-${CL_ROUTINES_ALL}}"
    PF_WILL_BUILD="${PF_WILL_BUILD:-}"
    GATE_JSON="${CL_DATA_OUT}/.gate_G0.json"
    PF_LOG="${CL_ROOT}/logs/preflight_${SLURM_JOB_ID:-manual}.log"

    echo ""
    echo "=== (1) TOOLCHAIN IDENTITY ==="
    JV="$(cl_loaded_julia_version || echo '')"
    # The manifest read is reported as PATH + VERSION + REASON, never as a bare string.
    # .gate_G0.json once carried manifest_julia_version:"" next to a healthy julia_version
    # and a passing input check, and an empty string cannot say which of the two states it
    # meant: no manifest on disk at all (ENV_STEP=resolve moves an incompatible one aside
    # before it rewrites it, so a resolve that does not finish leaves the tree with none),
    # or a manifest present but carrying no julia_version line. cl_manifest_julia_version
    # returns 2 and 1 for those; the reason string below is what the gate records.
    MF_PATH="$(cl_manifest_path || true)"
    set +e
    MV="$(cl_manifest_julia_version)"; MV_RC=$?
    set -e
    case "${MV_RC}" in
        0) MV_WHY="read from $(basename "${MF_PATH}")" ;;
        1) MV_WHY="$(basename "${MF_PATH}") exists but has NO julia_version line" ;;
        *) MV_WHY="NO manifest file at ${MF_PATH} — the resolve did not leave one" ;;
    esac
    echo "  julia module   ${JULIA_MODULE}  -> ${JV:-<did not load>}"
    echo "  manifest file  ${MF_PATH}"
    echo "  manifest       julia_version = ${MV:-<none>}   (${MV_WHY})"
    # MANIFEST_MATCH is derived from the SAME two strings the json reports, by the same
    # major.minor comparison cl_check_julia_version makes, so the boolean and the printed
    # verdict below cannot disagree about a tree neither of them changed.
    MANIFEST_MATCH=false
    if [[ "${JULIA_LOADED}" == "1" && -n "${JV}" && -n "${MV}" \
       && "$(_cl_majmin "${MV}")" == "$(_cl_majmin "${JV}")" ]]; then MANIFEST_MATCH=true; fi
    cl_check_julia_version || true
    echo "  resolve required: $([[ "${MANIFEST_MATCH}" == "true" ]] && echo no || echo yes)"

    echo ""
    echo "=== (2) SYSIMAGE VERDICTS ==="
    SYS_GPU=ok; cl_sysimage_verdict gpu || SYS_GPU=unusable
    SYS_CPU=ok; cl_sysimage_verdict cpu || SYS_CPU=unusable

    echo ""
    echo "=== (3) cluster_preflight.sh --routines '${PF_ROUTINES}'${PF_WILL_BUILD:+ --will-build '${PF_WILL_BUILD}'} ==="
    # tee, not a redirect: the .out file stays the readable record AND the json below
    # is built by parsing what the preflight actually reported, so the two can never
    # disagree about which inputs were missing.
    set +e
    bash "${CL_ROOT}/cluster_preflight.sh" --routines "${PF_ROUTINES}" \
         ${PF_WILL_BUILD:+--will-build "${PF_WILL_BUILD}"} 2>&1 | tee "${PF_LOG}"
    PF_RC=${PIPESTATUS[0]}
    set -e

    # Every MISSING line the preflight printed, de-duplicated, with the trailing
    # "<- do this" hint stripped so the array holds paths and not prose.
    MISSING_JSON=""
    while IFS= read -r m; do
        [[ -n "${m}" ]] || continue
        m="${m//\"/\'}"
        MISSING_JSON="${MISSING_JSON:+${MISSING_JSON}, }\"${m}\""
    done < <(sed -n 's/^[[:space:]]*MISSING[[:space:]]*//p' "${PF_LOG}" \
             | sed 's/[[:space:]]*<-.*$//; s/[[:space:]]*$//' | sort -u)

    echo ""
    echo "=== (4) UPLOADED MARKET PANEL ==="
    # The panel arrives as market_panel.parquet and NOTHING on this cluster can
    # re-derive it: the 724 MB market_panel.csv it was built from stays on the local
    # machine, so utils.load_panel_cached reads this file as the source rather than as
    # a cache to validate. Section (3) only established that the file EXISTS. A
    # truncated or corrupted upload of the right name passes that and then surfaces six
    # hours later as a pyarrow traceback inside an estimator, after the sleep array has
    # burned its allocation — so open it here, decode a column, and count the rows.
    # source_csv_bytes is the stamp cluster_upload.py matched against the local
    # CSV before bundling; recording it makes the two ends of the transfer comparable.
    PANEL_PQ="${CL_DATA_IN}/market_panel.parquet"
    PANEL_ROWS=-1; PANEL_COLS=-1; PANEL_SRC=-1; PANEL_OK=false; PANEL_ERR=""
    if [[ ! -f "${PANEL_PQ}" ]]; then
        PANEL_ERR="not on disk: ${PANEL_PQ}"
    else
        PANEL_OUT="$(
            if [[ -n "${PY_MODULE:-}" ]]; then module load ${PY_MODULE} >/dev/null 2>&1 || true; fi
            if [[ -n "${CONDA_ENV:-}" ]]; then
                source activate "${CONDA_ENV}" >/dev/null 2>&1 || conda activate "${CONDA_ENV}" >/dev/null 2>&1 || true
            fi
            "${CF_PYTHON}" -c '
import sys
import pyarrow.parquet as pq
path = sys.argv[1]
handle = pq.ParquetFile(path)
schema = handle.schema_arrow
meta = schema.metadata or {}
stamp = meta.get(b"source_csv_bytes")
# Read ONE column in full: the footer alone would still parse on a file whose data
# pages were cut short, and a row count taken from the footer would then be a lie.
rows = pq.read_table(path, columns=[schema.names[0]]).num_rows
print(rows)
print(len(schema.names))
print(int(stamp) if stamp is not None else -1)
' "${PANEL_PQ}" 2>&1
        )" || true
        # The LAST three lines, not the first: a conda/module banner on stdout would
        # otherwise shift the fields and fail a perfectly good panel. A python that
        # raised has a traceback in those three, which the numeric test below rejects.
        PANEL_TAIL="$(echo "${PANEL_OUT}" | tail -n 3)"
        PANEL_ROWS="$(echo "${PANEL_TAIL}" | sed -n '1p')"
        PANEL_COLS="$(echo "${PANEL_TAIL}" | sed -n '2p')"
        PANEL_SRC="$(echo "${PANEL_TAIL}"  | sed -n '3p')"
        case "${PANEL_ROWS}${PANEL_COLS}${PANEL_SRC}" in
            *[!0-9-]*|"") PANEL_ERR="unreadable: ${PANEL_OUT}"; PANEL_ROWS=-1; PANEL_COLS=-1; PANEL_SRC=-1 ;;
            *) if [[ "${PANEL_ROWS}" -gt 0 && "${PANEL_COLS}" -gt 0 ]]; then PANEL_OK=true
               else PANEL_ERR="opened but empty (${PANEL_ROWS} rows, ${PANEL_COLS} cols)"; fi ;;
        esac
    fi
    if [[ "${PANEL_OK}" == "true" ]]; then
        PANEL_BYTES="$(stat -c%s "${PANEL_PQ}" 2>/dev/null || echo '?')"
        echo "  market_panel.parquet: ${PANEL_ROWS} rows x ${PANEL_COLS} cols, ${PANEL_BYTES} bytes on disk"
        echo "  source_csv_bytes    : ${PANEL_SRC}  (size of the market_panel.csv it was built from)"
        if [[ "${PANEL_SRC}" == "-1" ]]; then
            echo "  [!] no source_csv_bytes stamp — this parquet was not written by utils.refresh_panel_cache"
        fi
    else
        echo "  [X] market_panel.parquet is NOT usable: ${PANEL_ERR}"
        echo "      Re-upload the data bundle (cluster/upload_manifest.txt row 'market_panel')."
        echo "      Build it locally with: python panel_pipeline.py, then"
        echo "      python cluster_upload.py --stage  (it checks the stamp before bundling)."
    fi

    OK=true; [[ ${PF_RC} -eq 0 ]] || OK=false
    [[ "${PANEL_OK}" == "true" ]] || OK=false
    cat > "${GATE_JSON}" <<EOF
{
  "gate": "G0",
  "ok": ${OK},
  "job_id": "${SLURM_JOB_ID:-manual}",
  "routines": "${PF_ROUTINES}",
  "will_build": "${PF_WILL_BUILD}",
  "julia_module": "${JULIA_MODULE}",
  "julia_version": "${JV}",
  "manifest_path": "${MF_PATH}",
  "manifest_julia_version": "${MV}",
  "manifest_read": "${MV_WHY}",
  "manifest_match": ${MANIFEST_MATCH},
  "sysimage_gpu": "${SYS_GPU}",
  "sysimage_cpu": "${SYS_CPU}",
  "missing_inputs": [${MISSING_JSON}],
  "panel_ok": ${PANEL_OK},
  "panel_rows": ${PANEL_ROWS},
  "panel_cols": ${PANEL_COLS},
  "panel_source_csv_bytes": ${PANEL_SRC},
  "preflight_rc": ${PF_RC},
  "log": "${PF_LOG}",
  "checked_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo ""
    echo "── .gate_G0.json ──"
    cat "${GATE_JSON}"
    if [[ "${OK}" != "true" ]]; then
        # A non-zero preflight keeps its own code; a panel-only failure exits 1, so the
        # gate blocks on a corrupt panel exactly as it blocks on a missing input.
        G0_RC=${PF_RC}; if [[ ${G0_RC} -eq 0 ]]; then G0_RC=1; fi
        echo "" >&2
        echo "G0 FOUND BLOCKERS — exiting ${G0_RC} so nothing downstream starts." >&2
        [[ "${PANEL_OK}" == "true" ]] || echo "  the uploaded market panel could not be read (see section 4)." >&2
        echo "  Act on the REMEDIATION blocks above, then re-submit the pipeline." >&2
        echo "  --skip-preflight on pipeline_all.sh submits without this gate." >&2
        exit ${G0_RC}
    fi
    ;;

resolve)
    # ── THE ONE Pkg.resolve() SITE ────────────────────────────────────────────
    # The accepted version is derived from the LOADED julia, not from a hardcoded
    # '1.11' literal, so this keeps working whatever module the cluster offers.
    ACCEPT="$(julia -e 'print(VERSION.major, ".", VERSION.minor)')"
    MF="$(cl_manifest_path || true)"
    MF_BAK=""
    echo "Accepted manifest julia_version prefix: ${ACCEPT}"
    if [[ -f "${MF}" ]]; then
        # Two steps, not `$(f || echo unknown)`: on a manifest that exists but carries no
        # julia_version the function prints an empty line AND returns non-zero, so the
        # fallback would append to it and MV would hold an embedded newline — which then
        # goes into the backup FILENAME below.
        MV="$(cl_manifest_julia_version || true)"; MV="${MV:-unknown}"
        echo "Found $(basename "${MF}") (julia_version = ${MV})"
        if [[ "${MV}" == ${ACCEPT}.* || "${MV}" == "${ACCEPT}" ]]; then
            echo "  compatible with ${ACCEPT} — keeping; this run only instantiates + precompiles."
        else
            # The manifest has to be out of the way before the resolve — Pkg will not
            # downgrade one resolved under a newer Julia — but moving it aside is the step
            # that can leave the tree with NO manifest at all, and every consumer of
            # cl_manifest_julia_version then reports an empty version with no way to say
            # why. So the removal is paired with a restore: MF_BAK holds the original
            # until the resolve has both exited zero and left a parsable manifest behind,
            # and the trap below puts it back on any other outcome, including a wall-kill.
            MF_BAK="${MF}.bak.${MV}"
            cp -f "${MF}" "${MF_BAK}"
            echo "  incompatible with ${ACCEPT} — backed up to $(basename "${MF_BAK}") and moved aside for the resolve."
            echo "  (it is restored automatically if the resolve does not produce a usable manifest)"
            rm -f "${MF}"
            trap 'if [[ -n "${MF_BAK}" && -f "${MF_BAK}" && ! -f "${MF}" ]]; then
                      cp -f "${MF_BAK}" "${MF}"
                      echo "  [restore] the resolve left no manifest — put $(basename "${MF_BAK}") back at $(basename "${MF}")." >&2
                  fi' EXIT
        fi
    else
        echo "No manifest at ${MF} — the resolve will create one."
    fi

    echo ""
    echo "=== Resolving packages: $(date) ==="
    # Registry.update is ADVISORY here. It is the one step in this job that wants the
    # network, `day` nodes do not always have it, and an uncaught throw would abort the
    # resolve after the manifest has already been moved aside — turning a stale registry
    # into a tree with no manifest. The resolve itself then either succeeds against the
    # registry already in the depot or fails on its own terms, which is a verdict worth
    # having; a registry that could not be refreshed is not.
    julia --project="${CL_ROOT}" -e '
        using Pkg
        # A resolve needs a registry to look package UUIDs up in, and this depot is
        # project-local: a fresh one has NO registry at all. Registry.update() only
        # refreshes registries that are already installed -- on an empty depot it
        # succeeds having done nothing, and the resolve then dies on the first package
        # it cannot find ("expected package CUDA [052768ef] to be registered"), which
        # reads like a broken environment rather than a missing index. So: install
        # General when nothing is reachable, refresh it when something is.
        regs = try Pkg.Registry.reachable_registries() catch; [] end
        if isempty(regs)
            println("  No registry in this depot — installing General...")
            Pkg.Registry.add("General")
        else
            println("  Updating package registry (", length(regs), " installed)...")
            try
                Pkg.Registry.update()
            catch err
                println("  [!] registry update failed: ", sprint(showerror, err))
                println("      continuing against the registry already in this depot.")
            end
        end
        Pkg.resolve();      println("  Manifest written.")
        Pkg.instantiate();  println("  Packages installed.")
        Pkg.precompile();   println("  Precompilation done.")
    '

    # The resolve is only done when a manifest Julia can read is back on disk. Exiting
    # zero here without one is how the NEXT run's gate G0 ends up reporting an empty
    # manifest_julia_version about a resolve that was believed to have worked.
    MF="$(cl_manifest_path || true)"
    if ! cl_manifest_julia_version >/dev/null; then
        echo "ERROR: the resolve exited zero but left no readable manifest at ${MF}." >&2
        echo "  The backup is restored on exit; re-run this step once the cause is fixed." >&2
        exit 1
    fi
    # Nothing left to restore: the tree carries a usable manifest again.
    MF_BAK=""; trap - EXIT

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
    echo "$(basename "${MF}") now records: $(cl_manifest_julia_version)"
    echo "NEXT: bash cluster_preflight.sh   (it should now print 'RESOLVE REQUIRED: no')"
    ;;

sysimage_gpu) build_one gpu ;;
sysimage_cpu) build_one cpu ;;

esac

echo "env_job ENV_STEP=${ENV_STEP} complete: $(date)"
