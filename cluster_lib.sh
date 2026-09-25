#!/bin/bash
# cluster_lib.sh — the ONE sourced library for the consolidated Bouchet scripts.
# ==============================================================================
# WHAT THIS IS
#   A library. It is SOURCED, never executed; it carries no #SBATCH header and it
#   submits nothing. Sourcing it has no side effect beyond defining variables and
#   functions, so sourcing inside a compute job and on the login node is identical.
#
# WHAT MUST EXIST FIRST
#   Nothing. It needs no module, no Python, no PATH change. If the manual upload
#   drops this file, every consolidated script fails at line ~2 with a plain
#   "No such file or directory" instead of halfway through a submission.
#
# WHAT TO RUN NEXT
#   Never this. Two entry points sit above it:
#     env_job.sh       submitted, never run on the login node. ENV_STEP=resolve
#                      (Pkg.resolve, alone), sysimage_gpu / sysimage_cpu, preflight (G0).
#     pipeline_all.sh  the whole graph. It only SUBMITS; every check is a job.
#   The per-phase orchestrators it calls — sleep_run.sh, blp_run.sh, bbl_run.sh,
#   cf_run.sh, cf_eq_run.sh — each submit one phase and are also usable alone for a
#   resume. cluster_archive.sh packages a finished step into data/output/download.
#
# HOW TO SOURCE IT
#   login-node script:  CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   compute job:        CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
#   then:               . "${CL_DIR}/cluster_lib.sh"
# ==============================================================================

# ── 0. Where we are ───────────────────────────────────────────────────────────
# CL_ROOT is the scripts/ checkout: the dir holding this library and every .jl/.py.
CL_ROOT="${CL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CL_DATA_ROOT="${DATA_ROOT:-$(cd "${CL_ROOT}/.." && pwd)/data}"
DATA_ROOT="${CL_DATA_ROOT}"
CL_DATA_OUT="${CL_DATA_ROOT}/output"
CL_DATA_IN="${CL_DATA_ROOT}/input"
# EXPORTED, not merely assigned: the Python and Julia steps read the tree out of the
# environment, and sbatch propagates only exported variables, so an assignment alone
# reaches a job or a --wrap child as unset and the step silently falls back to its
# local-layout default.
export CL_ROOT CL_DATA_ROOT DATA_ROOT CL_DATA_OUT CL_DATA_IN

# ── 0b. THE STEP TREE — one folder per pipeline step under data/output ────────
# data/input is UPLOADS ONLY and data/output is everything the cluster produced, split
# so that every produced family has exactly ONE producer writing to exactly ONE
# directory. That is what makes a consumer unable to resolve a second, older copy of
# the same routine, and it is also the unit cluster_archive.sh packages.
#
# of_root.jl holds the other half of this agreement: blp_dir(out), logit_dir(out),
# draws_dir(out) and cf_out_dir(out, ...) build these same paths from out_dir ==
# data/output, so the shell and Julia sides agree by construction rather than by two
# lists that have to be kept in step by hand. Changing a folder name here means
# changing it there in the same edit.
CL_STEP_SLEEP="${CL_DATA_OUT}/sleep"
CL_STEP_DEMAND="${CL_DATA_OUT}/demand_prep"
CL_STEP_LOGIT="${CL_DATA_OUT}/logit"
CL_STEP_BLP="${CL_DATA_OUT}/blp"
CL_STEP_DRAWS="${CL_STEP_BLP}/draws"
CL_STEP_BBL="${CL_DATA_OUT}/bbl"
CL_STEP_CF="${CL_DATA_OUT}/counterfactuals"
CL_STEP_DOWNLOAD="${CL_DATA_OUT}/download"
export CL_STEP_SLEEP CL_STEP_DEMAND CL_STEP_LOGIT CL_STEP_BLP CL_STEP_DRAWS \
       CL_STEP_BBL CL_STEP_CF CL_STEP_DOWNLOAD

# cl_export_step_dirs: the ONE place the Python layer learns where each step writes.
# utils/paths.py resolves each step folder from exactly one environment variable
# (SLEEP_OUT_ROOT, DEMAND_PREP_DIR, CF_FOUNDATION_DIR, CF_COST_FWD, COST_POLFUNC_DIR)
# and falls back to the local ESTIMATION_OUTPUT layout when it is unset — so a job that
# does not call this writes a real output into a tree nobody downloads, exits 0, and the
# loss surfaces phases later. Every *_job.sh therefore calls it immediately after
# cl_bootstrap_tree, which is what makes OPEN_FINANCE_ROOT and the step dirs one
# decision instead of a per-branch export that can drift between branches.
#
# Idempotent: plain assignments plus mkdir -p, so calling it twice, or in a job whose
# tree already exists, does nothing. cl_of_root is defined in section 10; a function
# body resolves its calls at call time, so the forward reference is safe.
cl_export_step_dirs () {
    export OPEN_FINANCE_ROOT="$(cl_of_root)"
    export SLEEP_OUT_ROOT="${CL_STEP_SLEEP}"
    export DEMAND_PREP_DIR="${CL_STEP_DEMAND}"
    export CF_FOUNDATION_DIR="${CL_STEP_CF}"
    export CF_COST_FWD="${CL_STEP_BBL}"
    export COST_POLFUNC_DIR="${CL_STEP_BBL}"
    export STATE_CENTERING_JSON="${CL_DATA_IN}/state_centering_means.json"
    export CL_ROOT CL_DATA_IN CL_DATA_OUT
    # A dry run submits nothing and must leave nothing behind, so it exports the
    # paths (the point of a dry run is to show them) without creating the tree.
    [[ "${CL_DRYRUN:-0}" == "1" ]] && return 0
    mkdir -p "${CL_STEP_SLEEP}" "${CL_STEP_DEMAND}" "${CL_STEP_LOGIT}" \
             "${CL_STEP_BLP}" "${CL_STEP_DRAWS}" "${CL_STEP_BBL}" \
             "${CL_STEP_CF}" "${CL_STEP_DOWNLOAD}"
}

# ── 1. TOOLCHAIN IDENTITY — the single site for every module/env name ─────────
# Every one of these is env-overridable, so a toolchain drift discovered at login
# is fixed with ONE `export`, not by editing seven files under time pressure.
#
#   export JULIA_MODULE=Julia/1.10.4-foss-2022b     # if 1.11.4 is gone
#   export CONDA_ENV=base                           # if dep_comp_blp is gone
#
# JULIA_MODULE is the name handed to `module load`. cluster_preflight.sh prints
# what `module avail Julia` actually offers when this one does not load.
# Verified on the login node (module -t avail): 1.9.2, 1.10.4, 1.10.8, 1.11.3,
# 1.11.4-linux-x86_64, 1.12.1, 1.12.4, 1.12.5 — so the default below loads.
JULIA_MODULE="${JULIA_MODULE:-Julia/1.11.4-linux-x86_64}"
# PY_MODULE / CONDA_ENV keep the `${VAR+set}` idiom: UNSET means "apply the
# default", explicitly EMPTY means "opt out" (no module load / no conda activate).
[[ -z "${PY_MODULE+set}" ]] && PY_MODULE=miniconda
# CONDA_ENV: dep_comp_blp. Verified on the login node — `conda env list` offers
# base, dep_comp_blp and test_env, and dep_comp_blp already imports numpy, pandas,
# scipy, pyarrow, statsmodels AND matplotlib, i.e. every REQ string below. There is
# nothing to create and nothing to install.
[[ -z "${CONDA_ENV+set}" ]] && CONDA_ENV=dep_comp_blp
CF_PYTHON="${CF_PYTHON:-python3}"
# What the Python steps actually import. bbl_solve.py needs
# numpy/pandas/scipy (+pyarrow via pd.read_parquet) and NOT statsmodels — that is
# a Step-1 polfunc dependency only, so requiring it always would reject an
# otherwise-usable env.
CL_PY_REQ_SOLVE="numpy, pandas, scipy, pyarrow"
CL_PY_REQ_POLFUNC="numpy, pandas, scipy, pyarrow, statsmodels"
# The sleepiness stage (sleep_job.sh, every branch) imports the widest stack of the
# three: statsmodels for the first stages and matplotlib because the export step
# (sleep_export_all.py / sleep_export_spec12_compare.py, run through
# sleep_pipeline.py --skip-sleep) draws figures. MPLBACKEND=Agg keeps that
# headless; it does not make the import optional. dep_comp_blp already carries all
# six — this string is the preflight PROBE, not a to-do list.
CL_PY_REQ_SLEEP="numpy, pandas, scipy, pyarrow, statsmodels, matplotlib"
# The BBL sweep (bbl_job.sh BBL_STEP=sweep) runs bbl_shards.py, which is stdlib-only and opens
# the parquet shards with pyarrow to tell a complete file from a truncated one.
CL_PY_REQ_SWEEP="pyarrow"

# ── 2. Julia environment ──────────────────────────────────────────────────────
# cl_load_julia: `module reset` (the correct command on Bouchet — `module purge`
# cannot unload StdEnv) + load JULIA_MODULE + the depot prepend every compute job
# uses. ONE depot for the whole project: a job that reads a different depot from
# the one the resolve wrote precompiles from scratch every time.
cl_load_julia () {
    module reset >/dev/null 2>&1 || true
    if ! module load "${JULIA_MODULE}" 2>&1; then
        echo "ERROR: 'module load ${JULIA_MODULE}' FAILED." >&2
        cl_julia_candidates >&2
        echo "  REMEDIATION: pick one of the candidates above and re-run with" >&2
        echo "      export JULIA_MODULE=<candidate>" >&2
        return 2
    fi
    export JULIA_MKL_THREADING="${JULIA_MKL_THREADING:-tbb}"
    export JULIA_DEPOT_PATH="${CL_ROOT}/.julia_depot:${JULIA_DEPOT_PATH:-}"
    return 0
}

# cl_julia_candidates: report what the cluster ACTUALLY offers. Lmod writes
# `module avail` to stderr, hence the 2>&1.
cl_julia_candidates () {
    echo "  --- module avail Julia (verbatim) ---"
    module avail Julia 2>&1 | sed 's/^/  | /' || echo "  | (module avail failed)"
    echo "  --- parsed candidates ---"
    local cands
    cands="$(module avail Julia 2>&1 \
             | tr ' \t' '\n\n' \
             | grep -E '^Julia/[^ ]+$' \
             | sed 's/(default)//' \
             | sort -u)"
    if [[ -n "${cands}" ]]; then printf '  %s\n' ${cands}
    else echo "  (none parsed — read the verbatim block above)"; fi
}

# cl_manifest_path: the manifest file JULIA ITSELF would read for this project, printed
# on stdout. Base.manifest_names() searches four names in this order and takes the first
# that exists, so a project can legitimately carry a version-pinned manifest instead of
# (or alongside) the plain one, and a check hardcoded to Manifest.toml then reports "no
# manifest" about a project Pkg resolves fine. When none of the four exists it prints the
# path Pkg would CREATE — the caller still gets a name to put in a message — and returns 1,
# which is the one signal that distinguishes "no manifest at all" from "cannot parse it".
cl_manifest_path () {
    local d="${1:-${CL_ROOT}}" n v
    v="$(cl_loaded_julia_version 2>/dev/null || true)"; v="$(_cl_majmin "${v}")"
    for n in ${v:+"JuliaManifest-v${v}.toml"} "JuliaManifest.toml" \
             ${v:+"Manifest-v${v}.toml"} "Manifest.toml"; do
        [[ -f "${d}/${n}" ]] && { printf '%s' "${d}/${n}"; return 0; }
    done
    printf '%s' "${d}/Manifest.toml"
    return 1
}

# cl_manifest_julia_version: the julia_version the manifest records, on stdout.
# THREE outcomes, and they are three different return codes because the gate json has to
# be able to say which one happened: an empty string alone cannot tell "the resolve wiped
# the manifest" from "the manifest is there and unparsable", and .gate_G0.json once
# reported "" for the first while every other section looked healthy.
#   0  parsed        -> prints the version
#   1  no julia_version line in a manifest that does exist
#   2  no manifest file at all
cl_manifest_julia_version () {
    local mf ver rc=0
    if [[ $# -gt 0 ]]; then mf="$1"; [[ -f "${mf}" ]] || { echo ""; return 2; }
    else mf="$(cl_manifest_path)" || { echo ""; return 2; }
    fi
    ver="$(sed -n 's/^[[:space:]]*julia_version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "${mf}" | head -1)"
    [[ -n "${ver}" ]] || rc=1
    printf '%s\n' "${ver}"
    return ${rc}
}

cl_loaded_julia_version () {
    command -v julia >/dev/null 2>&1 || { echo ""; return 1; }
    julia --version 2>/dev/null | sed -n 's/.*version[[:space:]]*\([0-9][0-9A-Za-z.\-]*\).*/\1/p' | head -1
}

_cl_majmin () { printf '%s' "${1%%-*}" | cut -d. -f1,2; }

# cl_check_julia_version: 0 = manifest matches the loaded julia (no resolve),
#                         1 = mismatch (a resolve is required), 2 = cannot tell.
# Prints a one-line verdict. Never rewrites anything: the resolve itself is
# env_job.sh ENV_STEP=resolve, run ALONE, never inside an array or a chain,
# because concurrent Pkg.resolve() on NFS corrupts Manifest.toml.
cl_check_julia_version () {
    local mv lv mrc mf
    mv="$(cl_manifest_julia_version)"; mrc=$?
    mf="$(cl_manifest_path || true)"
    lv="$(cl_loaded_julia_version || true)"
    if [[ -z "${lv}" ]]; then echo "  julia not on PATH — load the module first"; return 2; fi
    if [[ ${mrc} -eq 2 ]]; then
        echo "  NO MANIFEST at ${mf} — a resolve creates it"
        echo "  (ENV_STEP=resolve moves an incompatible manifest aside and writes a new one;"
        echo "   a tree with no manifest at all means that resolve did not finish.)"
        return 1
    fi
    if [[ -z "${mv}" ]]; then echo "  $(basename "${mf}") has no julia_version line — a resolve rewrites it"; return 1; fi
    if [[ "$(_cl_majmin "${mv}")" == "$(_cl_majmin "${lv}")" ]]; then
        echo "  $(basename "${mf}") julia_version=${mv} matches loaded julia ${lv}"
        return 0
    fi
    echo "  $(basename "${mf}") julia_version=${mv} does NOT match loaded julia ${lv}"
    return 1
}

# ── 3. SYSIMAGE POLICY (a hard check, not a comment) ─────────────────────────
# A sysimage is BOTH Julia-version-specific AND CPU-target-specific. Built on
# gpu_h200 (sapphirerapids) it is REJECTED on `day` nodes, and when the load
# fails CUDA.functional() goes false and cf_demand_eval.jl (~line 77)
# takes the CPU share path while the job still holds an H200 at ~0% util.
# So: refuse, loudly, rather than degrade silently.
#
# FIVE SIGNALS, any one failing = refuse:
#   (a) the .so exists;
#   (b) Manifest.toml / Project.toml newer than the .so by MTIME — cheap and
#       usable on the login node, but NFS/OneDrive mtimes lie, so pre-filter only;
#   (c) the provenance sidecar <img>.json written at build time by env_job.sh —
#       julia_version compared exactly, manifest_sha256 compared by CONTENT
#       (not mtime), cpu_target compared against the running node;
#   (d) the FATAL load proof at build time (env_job.sh renames a failed image to
#       .so.failed, so a broken image never sits on disk looking done);
#   (e) the runtime probe below, which loads the image on the node that will use
#       it and compares Sys.CPU_NAME with the sidecar's cpu_target.
# An image with NO sidecar is UNKNOWN provenance and is refused unless
# ALLOW_UNSTAMPED_SYSIMAGE=1. ALLOW_SYSIMAGE_FALLBACK=1 restores the old degrade
# with a loud log line.
cl_sysimage_for () {   # cl_sysimage_for gpu|cpu -> path
    case "$1" in
        gpu) printf '%s' "${CL_ROOT}/blp_sysimage.so" ;;
        cpu) printf '%s' "${CL_ROOT}/blp_sysimage_cpu.so" ;;
        *)   echo "cl_sysimage_for: target must be gpu|cpu (got '$1')" >&2; return 2 ;;
    esac
}

cl_sysimage_build_cmd () {  # the exact command that rebuilds this target
    case "$1" in
        gpu) printf '%s' "sbatch --partition=gpu_h200 --gpus=h200:1 --time=04:00:00 --cpus-per-task=8 --mem=64G --export=ALL,ENV_STEP=sysimage_gpu env_job.sh" ;;
        cpu) printf '%s' "sbatch --partition=day --constraint=cpugen:turin --time=04:00:00 --cpus-per-task=8 --mem=64G --export=ALL,ENV_STEP=sysimage_cpu env_job.sh" ;;
    esac
}

_cl_json_get () {  # _cl_json_get <file> <key>  (flat one-key-per-line JSON)
    [[ -f "$1" ]] || return 1
    sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$1" | head -1
}

# The sidecar's manifest_sha256 has to be taken over the manifest JULIA reads, which is
# what cl_manifest_path resolves — a sha of a Manifest.toml that Pkg ignores in favour of
# a version-pinned twin would compare a file nothing loads against a build nothing broke.
cl_manifest_sha () {
    local mf
    mf="$(cl_manifest_path)" || { echo "none"; return 0; }
    sha256sum "${mf}" 2>/dev/null | cut -d' ' -f1
}

# cl_sysimage_verdict TARGET -> prints a verdict, returns 0 OK / 1 refuse.
# Login-node safe: runs signals (a) (b) (c-version/sha) only. The node-local
# cpu_target check needs the node, so it lives in cl_require_sysimage.
cl_sysimage_verdict () {
    local tgt="$1" img side rc=0 sv ss lv mf
    img="$(cl_sysimage_for "${tgt}")" || return 2
    side="${img}.json"
    if [[ ! -f "${img}" ]]; then
        echo "  ${tgt}: MISSING $(basename "${img}")"
        echo "     REMEDIATION: $(cl_sysimage_build_cmd "${tgt}")"
        return 1
    fi
    mf="$(cl_manifest_path || true)"
    if [[ -f "${mf}" && "${mf}" -nt "${img}" ]] \
       || [[ -f "${CL_ROOT}/Project.toml" && "${CL_ROOT}/Project.toml" -nt "${img}" ]]; then
        echo "  ${tgt}: Manifest/Project.toml is NEWER than $(basename "${img}") (mtime pre-filter)"
        rc=1
    fi
    if [[ ! -f "${side}" ]]; then
        if [[ "${ALLOW_UNSTAMPED_SYSIMAGE:-0}" == "1" ]]; then
            echo "  ${tgt}: no provenance sidecar — ACCEPTED because ALLOW_UNSTAMPED_SYSIMAGE=1"
        else
            echo "  ${tgt}: no provenance sidecar $(basename "${side}") — UNKNOWN provenance, refused"
            echo "     REMEDIATION (preferred): $(cl_sysimage_build_cmd "${tgt}")"
            echo "     override (deliberate):   export ALLOW_UNSTAMPED_SYSIMAGE=1"
            rc=1
        fi
    else
        sv="$(_cl_json_get "${side}" julia_version)"
        ss="$(_cl_json_get "${side}" manifest_sha256)"
        lv="$(cl_loaded_julia_version || true)"
        if [[ -n "${lv}" && -n "${sv}" && "${sv}" != "${lv}" ]]; then
            echo "  ${tgt}: built under julia ${sv}, loaded julia is ${lv}"
            echo "     REMEDIATION: $(cl_sysimage_build_cmd "${tgt}")"
            rc=1
        fi
        if [[ -n "${ss}" && "${ss}" != "$(cl_manifest_sha)" ]]; then
            echo "  ${tgt}: Manifest.toml CONTENT changed since the build (sha256 differs)"
            echo "     REMEDIATION: $(cl_sysimage_build_cmd "${tgt}")"
            rc=1
        fi
    fi
    [[ ${rc} -eq 0 ]] && echo "  ${tgt}: $(basename "${img}") OK ($(_cl_json_get "${side}" cpu_target 2>/dev/null || echo unstamped))"
    return ${rc}
}

# cl_require_sysimage TARGET: the in-job hard gate. On success it sets the global
# array CL_JULIA_SYS=(--sysimage <img>); on failure it either exits 1 (default) or,
# with ALLOW_SYSIMAGE_FALLBACK=1, clears CL_JULIA_SYS and continues loudly.
# ── The two BUILD PROBES the RC phase decides from, and the gate has to agree with.
#
# blp_run.sh decides whether to submit the sysimage build and the draws build by looking
# at these exact artifacts on disk; gate G0 runs BEFORE either job exists and must not
# refuse the run for the absence of the very things the run is about to create. Two
# callers, one implementation, so "what blp_run.sh will build" and "what G0 excuses" are
# the same predicate rather than two lists that drift apart.
#
# cl_sysimage_missing: 0 (true) when the image OR its provenance sidecar is absent. The
# sidecar counts: cl_require_sysimage refuses an unstamped image at run time, so an image
# without one is not a usable image and has to be rebuilt exactly like a missing one.
# cl_sysimage_missing: 0 (true) when the image cannot be used as it stands. This is the
# auto-build decision in blp_run.sh, so it asks the same question the G0 gate asks --
# cl_sysimage_verdict -- rather than testing for the files alone. A present-but-invalid
# image (Manifest.toml re-resolved under a different Julia, a rebuilt environment, a
# sidecar sha that no longer matches) is exactly the case that must trigger a rebuild:
# the RC ladders load it, fail in seconds, and leave the mismatch to be read off a stack
# trace. Diagnostics are silenced here because the caller reports its own one-line verdict.
cl_sysimage_missing () {   # cl_sysimage_missing gpu|cpu
    local img; img="$(cl_sysimage_for "$1")" || return 2
    [[ -f "${img}" && -f "${img}.json" ]] || return 0
    cl_sysimage_verdict "$1" >/dev/null 2>&1 && return 1
    return 0
}
# cl_draws_missing: 0 (true) when THIS R/seed's draws are not both on disk. demo_draws
# itself is deliberately not tested — it is the 14 GB member, written in the same job as
# the key index, so the index stands in for it and no stat lands on a file that may be
# mid-split.
cl_draws_missing () {   # cl_draws_missing <R> <SEED>
    [[ -f "${CL_STEP_DRAWS}/halton_nu_R${1}_seed${2}.jls" \
    && -f "${CL_STEP_DRAWS}/demo_key_index_R${1}_seed${2}.jls" ]] && return 1
    return 0
}

cl_require_sysimage () {
    local tgt="$1" img side want got
    CL_JULIA_SYS=()
    img="$(cl_sysimage_for "${tgt}")" || return 2
    side="${img}.json"
    if ! cl_sysimage_verdict "${tgt}"; then
        _cl_sysimage_refuse "${tgt}" "static provenance check failed"; return $?
    fi
    # Signal (e): load the image ON THIS NODE and read back its CPU target. A
    # CPU-target mismatch fails here with "Unable to find compatible target in
    # cached code image", which is exactly the failure that otherwise degrades
    # into a silent precompile stampede.
    got="$(julia --project="${CL_ROOT}" --sysimage "${img}" -e 'print(Sys.CPU_NAME)' 2>/dev/null || true)"
    if [[ -z "${got}" ]]; then
        _cl_sysimage_refuse "${tgt}" "the image does not LOAD on $(hostname)"; return $?
    fi
    want="$(_cl_json_get "${side}" cpu_target 2>/dev/null || true)"
    if [[ -n "${want}" && "${want}" != "${got}" ]]; then
        _cl_sysimage_refuse "${tgt}" "built for cpu_target=${want}, this node is ${got}"; return $?
    fi
    CL_JULIA_SYS=(--sysimage "${img}")
    echo "sysimage OK: $(basename "${img}")  (cpu_target=${got}, no per-task precompile)"
    return 0
}

_cl_sysimage_refuse () {
    local tgt="$1" why="$2"
    echo "" >&2
    echo "REFUSING TO RUN: sysimage (${tgt}) unusable — ${why}." >&2
    echo "  Running without it would precompile against the shared NFS depot; on a GPU job" >&2
    echo "  that race ends in CUDA.functional()==false and a silent CPU fall-back while the" >&2
    echo "  job still holds an H200. Rebuild it:" >&2
    echo "      $(cl_sysimage_build_cmd "${tgt}")" >&2
    if [[ "${ALLOW_SYSIMAGE_FALLBACK:-0}" == "1" ]]; then
        echo "  ALLOW_SYSIMAGE_FALLBACK=1 is set — continuing WITHOUT a sysimage (slow start)." >&2
        CL_JULIA_SYS=()
        return 0
    fi
    echo "  ALLOW_SYSIMAGE_FALLBACK=1 continues anyway (slow, and GPU jobs may run on CPU)." >&2
    exit 1
}

# cl_gpu_gate: a FATAL, cheap CUDA check run BEFORE the real work, so a mismatched
# toolchain costs seconds instead of a 16-hour H200 reservation at 0% util. Only
# call it from a job that actually holds a GPU.
#
# EVERY device the job holds is checked, not just the first: a packed BBL job (bbl_job.sh,
# PACK=k) gives each of its k shard processes one device, and a device that is not functional
# would turn exactly one shard into a silent CPU run. `cl_gpu_gate [n]` requires at least n
# visible devices (default: as many as CUDA_VISIBLE_DEVICES lists) and runs a trivial kernel on
# each. With one device this is the check it always was.
_cl_visible_gpus () {
    local v="${CUDA_VISIBLE_DEVICES:-}"
    [[ -n "${v}" ]] || { printf '1'; return 0; }
    awk -F, '{print NF}' <<< "${v}"
}
cl_gpu_gate () {
    local t="${CL_GPU_GATE_TIMEOUT:-300}" need="${1:-$(_cl_visible_gpus)}"
    echo "── CUDA gate (fatal, ${t}s budget, ${need} device(s)) ──"
    if CL_GPU_GATE_NEED="${need}" timeout "${t}" julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} -e '
        using CUDA
        CUDA.functional() || (println(stderr, "CUDA.functional() == false"); exit(3))
        need = parse(Int, get(ENV, "CL_GPU_GATE_NEED", "1"))
        devs = collect(CUDA.devices())
        length(devs) >= need || (println(stderr, "only ", length(devs), " CUDA device(s) visible, the job needs ", need); exit(4))
        for d in devs
            CUDA.device!(d)
            s = sum(CUDA.ones(Float32, 1024))
            s == 1024f0 || (println(stderr, "device ", CUDA.deviceid(d), " failed a trivial kernel"); exit(5))
            println("CUDA OK: device ", CUDA.deviceid(d), " ", CUDA.name(d), "  driver ", CUDA.driver_version())
        end
    '; then
        return 0
    fi
    echo "" >&2
    echo "REFUSING TO RUN: this job holds a GPU but CUDA is not functional." >&2
    echo "  cf_demand_eval.jl would take the CPU share path and the H200 would sit" >&2
    echo "  at ~0% util for the whole wall. Almost always a sysimage/CPU-target mismatch:" >&2
    echo "      $(cl_sysimage_build_cmd gpu)" >&2
    echo "  ALLOW_SYSIMAGE_FALLBACK=1 continues on CPU anyway." >&2
    [[ "${ALLOW_SYSIMAGE_FALLBACK:-0}" == "1" ]] && { echo "  (set — continuing on CPU)" >&2; return 0; }
    exit 1
}

# ── 5. ROUTINE DEFAULTS — one site, replacing six that disagree ──────────────
# The lineup is E1,E2,E3,E4 and all four go through EVERY phase — sleep, demand prep,
# logit, RC-BLP, BBL, CF — which is why the three lists hold the same value. They stay
# three variables because a resume narrows ONE phase (`--routines "3 4"` on bbl_run.sh
# after two ladders survive) without silently narrowing the others. An E5-E8 id
# anywhere is pre-relineup numbering.
CL_ROUTINES_ALL="${CL_ROUTINES_ALL:-1 2 3 4}"
CL_ROUTINES_RC="${CL_ROUTINES_RC:-1 2 3 4}"
CL_ROUTINES_CF="${CL_ROUTINES_CF:-1 2 3 4}"

# cl_routines_provenance: says where the value came from. No consolidated script
# does an unconditional `export ROUTINES=`, so an ENVIRONMENT warning here means
# the environment really does carry one.
cl_routines_provenance () {   # cl_routines_provenance <value> <source> <default>
    case "$2" in
        flag) echo "routines='$1'  (from --routines)" ;;
        env)  echo "routines='$1'  (FROM ENVIRONMENT — the default is '$3'; unset ROUTINES to use it)" ;;
        *)    echo "routines='$1'  (DEFAULT)" ;;
    esac
}

# ── 6. PREFLIGHT PRIMITIVES — one implementation of each check ───────────────
# CL_ASSUME_CHAINED: for the end-to-end DRY RUN only. pipeline_all.sh prints one
# graph spanning sleep -> logit -> BLP -> BBL -> CF -> CF-eq, and every stage after
# the first consumes a file an EARLIER stage of that same graph produces. On the
# login node at t=0 those files do not exist yet, so the children's on-disk
# preflights refuse and the graph stops at the first of them — which is exactly the
# thing a dry run exists to show.
#
# It is inert unless CL_DRYRUN=1 as well, and a dry run submits nothing, so this can
# never let a real job past a real check. Each softened check still prints its full
# MISSING line, tagged, so the dry run remains a staging report too.
CL_ASSUME_CHAINED="${CL_ASSUME_CHAINED:-0}"
_cl_assume_chained () { [[ "${CL_DRYRUN}" == "1" && "${CL_ASSUME_CHAINED}" == "1" ]]; }

cl_need_file () {   # cl_need_file <path> <label> [remediation...]
    local p="$1" label="$2"; shift 2
    if [[ -f "${p}" ]]; then return 0; fi
    if _cl_assume_chained; then
        echo "[dry-run] MISSING ${label}: ${p}"
        echo "          assumed produced earlier in this graph — a LIVE run REFUSES here."
        return 0
    fi
    echo "MISSING ${label}: ${p}"
    [[ $# -gt 0 ]] && printf '   -> %s\n' "$@"
    return 1
}
cl_need_draws () {  # cl_need_draws <R> <SEED>
    cl_need_file "${CL_STEP_DRAWS}/halton_nu_R${1}_seed${2}.jls" "R=${1} draws" \
        "the BLP phase builds them when they are absent:  bash blp_run.sh   (force with --draws)"
}
# The horizon a rebuilt curve must reach: the BBL forward-simulation horizon from the registry
# (bbl_discount.env), or HORIZON when that is longer. The curve files are shared by the BBL stage
# and the counterfactuals, and the BBL horizon is the longer consumer, so a hint printed from a
# counterfactual script (whose own HORIZON is shorter) must not tell anyone to truncate them.
_cl_rf_hint_h () {
    local h
    h="$(cl_bbl_discount_get BBL_HORIZON 2>/dev/null)" || h=""
    if [[ "${HORIZON:-}" =~ ^[0-9]+$ ]] && { [[ -z "${h}" ]] || (( HORIZON > h )); }; then h="${HORIZON}"; fi
    printf '%s' "${h:-<BBL_HORIZON from bbl_discount.env>}"
}
cl_need_rf_curve () {
    cl_need_file "${CL_DATA_IN}/forward_rf_qoq.csv" "forward r^f curve" \
        "build locally then upload: python scrape_forward_rf.py --horizon $(_cl_rf_hint_h) --start 2026Q1"
}
# The two multi-start inputs. Both are BCB-API products (Focus surveys), so like
# forward_rf_qoq.csv they cannot be produced on a compute node — no outbound internet — and both
# are small enough to stage every run: 36 x (T+1) rows of vintages (under 1 MB at T=250) and ~8 kB
# of transition parameters. Small is what makes the multi-start design affordable at all: the S
# simulated rate paths are REGENERATED on the node from these parameters plus a fixed seed, so
# nothing that scales with S or with the panel ever crosses the upload.
cl_need_rf_vintages () {
    local h; h="$(_cl_rf_hint_h)"
    cl_need_file "${CL_DATA_IN}/forward_rf_vintages.csv" "forward r^f curve vintages (per launch quarter)" \
        "build locally then upload: python scrape_forward_rf.py --vintage-from 2016Q1 --vintage-to 2024Q4 --horizon ${h}" \
        "36 launch quarters 2016Q1..2024Q4 x h=0..${h}; h=0 is the realised anchor rate"
}
cl_need_transitions () {
    cl_need_file "${CL_DATA_IN}/bbl_transitions.json" "BBL transition parameters" \
        "build locally then upload: python bbl_transitions.py" \
        "carries rate.process (focus_mean_plus_horizon_shock), cost_shifters and market_states"
}
# cl_need_rc_jls <routine>: the RC result the CF/BBL stack opens. It tests the exact file
# blp_dir(out_dir) names in cf_demand_eval.jl's _result_path, in the exact place
# the RC job writes it — the ladder persists its results in the blp step folder and nothing
# stages or renames them afterwards, so preflight and run resolve one path by construction.
# CF_STAGE picks which rung (the CF/BBL default is the deepest, `extended`).
cl_need_rc_jls () { # cl_need_rc_jls <routine>
    cl_need_file "${CL_STEP_BLP}/blp_results_E$1_spec_12_${CF_STAGE:-extended}.jls" \
        "RC result E$1 (stage ${CF_STAGE:-extended})" \
        "run the RC ladder first:  bash blp_run.sh --routines $1"
}
cl_need_costs () {  # cl_need_costs <routine> <stage>
    cl_need_file "${CL_STEP_BBL}/cost_params_E$1_spec_12_$2.json" "BBL cost params E$1" \
        "run the BBL cost stage first:  bash bbl_run.sh"
}

# cl_need_polfunc: the fitted policy function bbl_fwd_sim.jl reads through
# --policy-csv. Two locations, in this order: the bbl step folder, where the polfunc job
# writes it (COST_POLFUNC_DIR == CL_STEP_BBL), then data/input, which covers a resume that
# skips the polfunc job and hand-stages the CSV instead. It PRINTS the resolved path on
# stdout, so a caller both tests and captures in one call:
#     POLICY_CSV="$(cl_need_polfunc)" || exit 1
# which is why every diagnostic line here goes to stderr.
cl_need_polfunc () {
    local c
    for c in "${CL_STEP_BBL}/polfunc_fitted.csv" "${CL_DATA_IN}/polfunc_fitted.csv"; do
        [[ -f "${c}" ]] && { printf '%s\n' "${c}"; return 0; }
    done
    if _cl_assume_chained; then
        printf '%s\n' "${CL_STEP_BBL}/polfunc_fitted.csv"
        echo "[dry-run] MISSING polfunc_fitted.csv" >&2
        echo "          assumed produced earlier in this graph — a LIVE run REFUSES here." >&2
        return 0
    fi
    echo "MISSING policy function: polfunc_fitted.csv" >&2
    echo "   searched, in order:" >&2
    echo "     ${CL_STEP_BBL}" >&2
    echo "     ${CL_DATA_IN}" >&2
    echo "   -> produced by the BBL phase:  bash bbl_run.sh   (its polfunc pre-step)" >&2
    return 1
}

# ── CF4's two artefacts: ONE producer, ONE location, hence one candidate.
# upsilon_pix_E{k} and phi_nopix_E{k} are written by sleep_upsilon_export.py running on the
# cluster (sleep_job.sh SLEEP_STEP=upsilon, gated by G7) into CF_FOUNDATION_DIR, which
# cl_export_step_dirs points at the counterfactuals step folder. Nothing else writes them
# and nothing copies them afterwards, so there is exactly one place to look.
#
# The single candidate is the point, not an economy: a second candidate is how a stale
# uploaded pair and a fresh cluster-produced pair coexist and different consumers resolve
# different vintages of the same routine — the collision that aborts cf4 three phases
# later, after the RC ladders and the BBL solves. cf4_search_dirs() in cf4_pix.jl and
# gate G7 in sleep_job.sh resolve this same lone directory, so preflight, gate and run
# cannot open different files.
cl_cf4_dirs () {
    printf '%s\n' "${CL_STEP_CF}"
}

# cl_need_cf4_file <basename> <label>: 0 as soon as one candidate holds it. On failure
# it names EVERY directory searched — "missing" without "where I looked" is the report
# that sends someone hunting in the one place the code never reads.
cl_need_cf4_file () {
    local fn="$1" label="$2" d
    while IFS= read -r d; do
        [[ -f "${d}/${fn}" ]] && return 0
    done < <(cl_cf4_dirs)
    if _cl_assume_chained; then
        echo "[dry-run] MISSING ${label}: ${fn}"
        echo "          assumed produced earlier in this graph — a LIVE run REFUSES here."
        return 0
    fi
    echo "MISSING ${label}: ${fn}"
    echo "   searched:"
    while IFS= read -r d; do echo "     ${d}"; done < <(cl_cf4_dirs)
    echo "   -> produced by the sleepiness phase:  bash sleep_run.sh   (gate G7)"
    return 1
}

# cl_python_probe: LOGIN-NODE probe. Runs in a SUBSHELL so `module load` and
# `conda activate` cannot leak into the submitting shell. Returns 0/1.
cl_python_probe () {   # cl_python_probe "<import list>"
    if _cl_assume_chained; then
        echo "[dry-run] skipping the login-node Python probe (import $1)" >&2
        return 0
    fi
    (
      if [[ -n "${PY_MODULE:-}" ]]; then module load ${PY_MODULE} || true; fi
      if [[ -n "${CONDA_ENV:-}" ]]; then
          source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}" || true
      fi
      "${CF_PYTHON}" -c "import $1"
    ) >/dev/null 2>&1
}

# cl_setup_python: IN-JOB activation + verification. Sets PYBIN.
# Exits 127 on failure — bare compute-node python3 has no scientific stack, and a
# Traceback 14 h into the pipeline (after the fwd_sim array drains) is exactly the
# failure this prevents.
cl_setup_python () {   # cl_setup_python "<import list>"
    local req="$1"
    # if/fi, not `[[ ]] && cmd`: an empty PY_MODULE (the documented "opt out"
    # value) would make a trailing && list return 1 and, under set -e, kill the job.
    if [[ -n "${PY_MODULE:-}" ]]; then module load ${PY_MODULE}; fi
    if [[ -n "${CONDA_ENV:-}" ]]; then
        source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"
    fi
    PYBIN="${CF_PYTHON}"
    command -v "${PYBIN}" >/dev/null 2>&1 || {
        echo "ERROR: Python '${PYBIN}' not found on the node. Set PY_MODULE / CONDA_ENV / CF_PYTHON," >&2
        echo "  e.g.  PY_MODULE=miniconda CONDA_ENV=dep_comp_blp bash bbl_run.sh" >&2
        exit 127; }
    "${PYBIN}" -c "import ${req}" 2>/dev/null || {
        echo "ERROR: '${PYBIN}' lacks the required stack (${req})." >&2
        echo "  The cluster offers base, dep_comp_blp and test_env; dep_comp_blp carries the" >&2
        echo "  whole stack (numpy pandas scipy pyarrow statsmodels matplotlib) and is the default." >&2
        echo "  Use an existing env:  export CONDA_ENV=<name>   (conda env list)" >&2
        echo "  or an interpreter:    export CF_PYTHON=/path/to/python CONDA_ENV=''" >&2
        exit 127; }
    # Informational only: CF_COST_FWD is owned by cl_export_step_dirs, the single site
    # that decides every step folder. Printing the value this job actually received is
    # how a missing cl_export_step_dirs call shows up here, in the first ten lines of
    # the log, instead of as a psi_* file written where the solve does not read.
    echo "COST_FWD = ${CF_COST_FWD:-<unset: cl_export_step_dirs has not run>}"
}

# ── 7. THE ONE sbatch WRAPPER ────────────────────────────────────────────────
# Always --parsable (so the job id is capturable) and --kill-on-invalid-dep=yes
# (an afterok dependency that can never be satisfied CANCELS the job instead of
# leaving it pending forever). Caller args land LAST, so a caller-supplied
# --partition/--gpus/--mem/--time always wins over anything set before it — that
# is explicit here, not an accident of ordering.
#
# CL_DRYRUN=1 prints the full command and returns a synthetic MONOTONIC job id, so
# an entire dependency graph prints without a single submission.
CL_DRYRUN="${CL_DRYRUN:-0}"
CL_FAKE_JID_BASE="${CL_FAKE_JID_BASE:-9000000}"
# The counter lives in a FILE, not a shell variable: cl_sbatch is always called
# inside $( ), i.e. in a subshell, so an in-memory counter would reset on every
# call and print the same id for every job — making the dependency graph
# unreadable and unverifiable.
# It is EXPORTED so a child orchestrator (pipeline_all.sh -> bbl_run.sh /
# cf_run.sh) keeps counting from the parent's numbers instead of restarting, and
# it is keyed on the top-level PID so a second dry run starts clean. The file is
# created lazily, inside cl_sbatch, so sourcing this library still has no side
# effect on disk.
CL_FAKE_COUNTER="${CL_FAKE_COUNTER:-${TMPDIR:-/tmp}/.cl_dryrun_jid.$$}"
export CL_FAKE_COUNTER

# Quote an argument only when it needs it, so the printed command stays readable
# and still copy-pasteable.
_cl_q () {
    local a
    for a in "$@"; do
        if [[ "${a}" =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then printf ' %s' "${a}"
        else printf " '%s'" "${a//\'/\'\\\'\'}"; fi
    done
}

# _cl_cpugen_args: the cpugen pin, decided ONCE for every submission in the codebase.
#
# blp_sysimage_cpu.so is valid only on the microarchitecture it was BUILT on, and `day` mixes
# cpugen:turin (AMD 9575f/9655) with cpugen:emeraldrapids (Intel 8562Y+). An unpinned CPU job
# lands wherever day has room and refuses with "the image does not LOAD on <node>" -- but only
# when the scheduler happens to pick the wrong family, so it reads as a flaky or
# routine-specific failure. It cost three separate debug cycles in three different scripts
# before it was recognised as one bug, which is exactly why the decision belongs here rather
# than in each caller's own sub() helper.
#
# Two exemptions, both load-bearing:
#   - a GPU submission takes the gpu_h200-built image, and NO gpu node satisfies cpugen:turin;
#     pinning one makes sbatch reject the job outright with "Requested node configuration is
#     not available". (Exporting SBATCH_CONSTRAINT globally causes precisely this.)
#   - a caller that already passes its own --constraint keeps it; we never override.
# CL_NO_CPUGEN=1 opts a submission out entirely.
_cl_cpugen_args () {
    [[ "${CL_NO_CPUGEN:-0}" == "1" ]] && return 0
    local a last=""
    for a in "$@"; do
        case "${a}" in
            --constraint*|-C) return 0 ;;
            --gpus*|--gres=gpu*|--partition=gpu*|--partition=scavenge_gpu*) return 0 ;;
        esac
        last="${a}"
    done
    # A job script declares its partition in its own #SBATCH block, and that
    # declaration never reaches this function through "$@" -- blp_stage_job.sh names
    # gpu_h200 there and is submitted with neither flag on the command line. A GPU
    # submission carrying a CPU feature is refused outright ("Requested node
    # configuration is not available"), so the script named last is read before the
    # constraint is added. --wrap submissions leave a command string here, not a
    # path, and keep the pin.
    if [[ -f "${last}" ]] &&        grep -Eq '^#SBATCH[[:space:]]+--(gpus|gres=gpu|partition=(gpu|scavenge_gpu))' "${last}"; then
        return 0
    fi
    printf '%s' "--constraint=${CL_CPU_CONSTRAINT:-cpugen:turin}"
}

cl_sbatch () {
    local _cg; _cg="$(_cl_cpugen_args "$@")"
    if [[ "${CL_DRYRUN}" == "1" ]]; then
        local n jid
        n=$(( $(cat "${CL_FAKE_COUNTER}" 2>/dev/null || echo 0) + 1 ))
        printf '%s' "${n}" > "${CL_FAKE_COUNTER}"
        jid=$((CL_FAKE_JID_BASE + n))
        # Through cl_log, so the full command text is narration: on the terminal only
        # under CL_VERBOSE=1, and in the orchestrator log otherwise. Redirected to
        # stderr because every caller runs cl_sbatch inside $( ) — anything on stdout
        # but the job id lands in the caller's jobid variable and poisons the
        # dependency it is about to build.
        { cl_log "[dry-run] sbatch --parsable --kill-on-invalid-dep=yes$(_cl_q ${_cg:+"${_cg}"} "$@")"
          cl_log "          -> job id ${jid}"; } >&2
        printf '%s\n' "${jid}"
        return 0
    fi
    # The live path prints NOTHING but the job id sbatch --parsable returns: the caller
    # owns the one "name -> jobid" line, because only it knows the name and the deps.
    #
    # A refusal is reported here rather than returned as an empty string. Callers run
    # cl_sbatch inside $( ), and bash does not inherit errexit into a command
    # substitution, so an unreported failure becomes an empty job id that flows into
    # the next --dependency -- where "afterok:" with no id reads as NO dependency and
    # the rest of the ladder runs unsequenced.
    # A dependency whose id list came back empty is the most expensive typo in this
    # pipeline: SLURM reads "afterok:" with nothing after it as NO dependency, so the
    # job starts IMMEDIATELY instead of waiting, against inputs that do not exist yet.
    # Every orchestrator builds these from a captured job id, so the check lives here
    # rather than being repeated at the ~15 call sites that construct one.
    local _a _v _ids
    for _a in "$@"; do
        case "${_a}" in
            --dependency=*)
                _v="${_a#--dependency=}"
                case "${_v}" in
                    *:*)
                        _ids="${_v#*:}"
                        if [[ -z "${_ids}" || "${_ids}" == *::* || "${_ids}" == *: ]]; then
                            cl_err "REFUSING to submit: malformed dependency '${_v}' -- empty id list."
                            cl_err "  SLURM reads that as NO dependency and would start the job at once."
                            return 1
                        fi ;;
                esac ;;
        esac
    done
    local _out _rc
    _out="$(sbatch --parsable --kill-on-invalid-dep=yes ${_cg:+"${_cg}"} "$@")"; _rc=$?
    if (( _rc != 0 )) || [[ -z "${_out}" ]]; then
        cl_err "sbatch REFUSED this submission (rc=${_rc}); sbatch's own message is above."
        cl_err "  args:$(_cl_q ${_cg:+"${_cg}"} "$@")"
        return 1
    fi
    printf '%s
' "${_out}"
}

# cl_require_jid: a job id, or a hard stop naming what failed. Every dependency in
# the graph is built from a captured id, and an empty one silently detaches the job
# that was meant to wait.
cl_require_jid () {   # cl_require_jid <jobid> <what>
    if [[ -z "${1:-}" ]]; then
        cl_err "FATAL: no job id for '${2:-submission}' -- it was refused, so nothing downstream of it can be sequenced."
        return 1
    fi
    return 0
}

# ── 8. Small shared knobs ────────────────────────────────────────────────────
# Per-routine memory. E1/E2 are the LARGEST panels (E1 = 796,154 obs) and 200G
# OOM-killed them at ext1 (job 21525240): the hot buffer in
# blp_engine_cpu.jl:357 scales as n_pi*N*R and rc4 -> ext1 takes n_pi 3 -> 4.
# gpu_h200 nodes carry ~1.95 TiB, so 600G is ~30% of a node and schedules freely.
CL_MEM_DEFAULT="${CL_MEM_DEFAULT:-200G}"
CL_MEM_BIG="${CL_MEM_BIG:-600G}"
cl_mem_for () { case "$1" in 1|2) printf '%s' "${CL_MEM_BIG}" ;; *) printf '%s' "${CL_MEM_DEFAULT}" ;; esac; }

# Stage wall-time. The gpu_h200 QOS caps wall PER JOB: a 4-day request is rejected
# at submit time (QOSMaxWallDurationPerJobLimit), 2 days is accepted.
CL_WALL_DEEP="${CL_WALL_DEEP:-2-00:00:00}"
CL_WALL_CAP="${CL_WALL_CAP:-2-00:00:00}"
cl_stage_wall () {
    case "$1" in
        ext2|extended) printf '%s' "${CL_WALL_DEEP}" ;;
        full|ext1)     printf '%s' "2-00:00:00" ;;
        *)             printf '%s' "1-00:00:00" ;;
    esac
}
cl_warn_wall_cap () {
    local d="${CL_WALL_DEEP%%-*}"
    [[ "${CL_WALL_DEEP}" == *-* && "${d}" =~ ^[0-9]+$ && "${d}" -gt 2 ]] && \
        echo "[!] CL_WALL_DEEP=${CL_WALL_DEEP} exceeds the gpu_h200 QOS per-job wall (${CL_WALL_CAP}); sbatch will reject it." >&2
    return 0
}

cl_log_dir () { mkdir -p "${CL_ROOT}/logs"; printf '%s' "${CL_ROOT}/logs"; }

# cl_run_marker: an mtime marker so an archive bundles only THIS run's files.
# That filter is what stops a re-run from re-archiving — and, in move mode,
# deleting — an earlier run's output.
cl_run_marker () {   # cl_run_marker <prefix>
    # A dry run must leave nothing behind, so it names a marker without creating it.
    if [[ "${CL_DRYRUN}" == "1" ]]; then
        printf '%s\n' "${CL_DATA_OUT}/.$1_marker.DRYRUN"; return 0
    fi
    mkdir -p "${CL_DATA_OUT}"
    mktemp "${CL_DATA_OUT}/.$1_marker.XXXXXX"
}

# ── 9. QUIET OUTPUT — three sinks, one decision per line ─────────────────────
# A full submission is ~20 lines the operator has to act on: what was submitted with
# which id and dependency, which verdict a gate returned, where the download lands.
# Everything else — banners, env dumps, the dry-run command text, a child
# orchestrator's own narration — is detail that belongs in a file, because a terminal
# scrolling 800 lines hides the one line that matters and there is no way to tell
# afterwards which id belonged to which phase.
#
#   cl_say  facts the operator needs   -> terminal AND log
#   cl_log  narration                  -> log; terminal only when CL_VERBOSE=1
#   cl_err  failures                   -> stderr AND log, always visible
#
# Inside a compute job the default flips to verbose: Slurm is already capturing stdout
# to logs/<name>.out, that file IS the job's record, and a job whose banner and env
# dump went somewhere else is a job nobody can debug from its own log. The flip works
# because the orchestrator only ASSIGNS CL_VERBOSE, never exports it, so a job inherits
# the variable unset and picks the in-job default.
CL_VERBOSE="${CL_VERBOSE:-$([[ -n "${SLURM_JOB_ID:-}" ]] && echo 1 || echo 0)}"

# _cl_orch_log_init: decide the log path ONCE and leave it in CL_ORCH_LOG. It assigns in
# the CURRENT shell and is never called through $( ) from the writers below — a command
# substitution runs in a subshell, so an assignment made there is discarded and the next
# line, a second later, names a new file. It is EXPORTED, so a child orchestrator
# (pipeline_all.sh -> bbl_run.sh) appends to the parent's file and one run stays one file
# in submission order.
#
# Two sinks are /dev/null on purpose: a dry run must leave nothing behind, and inside a
# compute job stdout is already the record — a second sink would duplicate every line
# and, from a 100-task array, have a hundred tasks appending to one shared file.
_cl_orch_log_init () {
    if [[ "${CL_DRYRUN:-0}" == "1" || -n "${SLURM_JOB_ID:-}" ]]; then
        CL_ORCH_LOG="/dev/null"; export CL_ORCH_LOG; return 0
    fi
    [[ -n "${CL_ORCH_LOG:-}" ]] && return 0
    mkdir -p "${CL_ROOT}/logs" 2>/dev/null || true
    CL_ORCH_LOG="${CL_ROOT}/logs/orchestrator_$(date +%Y%m%d_%H%M%S)_$$.log"
    export CL_ORCH_LOG
    return 0
}

# cl_orch_log: the log path on stdout, for the closing "details in <file>" line.
cl_orch_log () { _cl_orch_log_init; printf '%s' "${CL_ORCH_LOG}"; }

# The log append never fails the caller: losing a narration line to a full quota must
# not take down a submission that is otherwise fine.
_cl_to_log () { _cl_orch_log_init; printf '%s\n' "$*" >> "${CL_ORCH_LOG}" 2>/dev/null || true; }

cl_say () { printf '%s\n' "$*"; _cl_to_log "$*"; return 0; }
cl_log () { [[ "${CL_VERBOSE}" == "1" ]] && printf '%s\n' "$*"; _cl_to_log "$*"; return 0; }
cl_err () { printf '%s\n' "$*" >&2;  _cl_to_log "$*"; return 0; }

# cl_banner is narration by definition — it identifies a phase, it never reports a
# result — so it goes through cl_log and the terminal keeps the ~20 lines that matter.
cl_banner () {
    local l
    cl_log "=============================================================================="
    for l in "$@"; do cl_log " ${l}"; done
    cl_log "=============================================================================="
}

# ── 10. THE OPEN-FINANCE SKELETON — what the sleepiness stage resolves against ─
# The Python sleepiness/CF stack anchors every path on utils/paths.py's
# OPEN_FINANCE root and the Julia entry points on of_root.jl. Both walk three
# levels up from the source file unless OPEN_FINANCE_ROOT is set — and on the
# cluster the code lives at HEAD/scripts, so that walk lands on the parent of HEAD,
# where the reads find nothing and an export can still exit 0.
#
# So we build a MINIMAL tree and point OPEN_FINANCE_ROOT at it. Three constraints
# make this exact rather than approximate:
#
#   (a) the BASENAME must be 'Open-Finance'. of_root.jl:resolve_of_root() rejects
#       anything else outright.
#   (b) shared/ and BCB/Egan_et_al_2025_Rep/processed/ are of_root.jl's SENTINELS.
#       It refuses — deliberately, without mkpath'ing — when either is absent,
#       because a June 2026 run resolved the root one level short and quietly
#       accumulated 8.9 GB of draws nobody read.
#   (c) 'Drafts/Deposit Competition/' must EXIST and stay EMPTY. The exporters
#       mirror .tex fragments and figures into paths.drafts_dir(), which is never
#       created on demand ("a copy into a path that does not exist should fail").
#
# The three panels the sleepiness estimators read are SYMLINKED out of data/input,
# so the uploaded copy stays the single physical copy and data/input keeps its
# "uploaded only" meaning. A dangling link is reported, never silently created-over.
#
# processed/ESTIMATION_OUTPUT IS ITSELF A SYMLINK TO data/output, and that is the
# load-bearing piece. utils/paths.estimation_output() is deliberately NOT
# redirectable by SLEEP_OUT_ROOT — "it also holds the BLP, counterfactual and cost
# trees, which a sleepiness sandbox has no business redirecting" — so without the
# link everything that resolves through it lands INSIDE the skeleton, next to nothing,
# while the rest of the stack reads data/output. With it the two coincide by
# construction:
#     estimation_output()  -> data/output
#     every accessor cl_export_step_dirs overrides (SLEEP_OUT_ROOT, DEMAND_PREP_DIR,
#     CF_FOUNDATION_DIR, CF_COST_FWD, COST_POLFUNC_DIR) -> its step folder under it
#     everything else utils/paths.py resolves (Rout/, BLP_RESULTS/) -> under it too
# and cluster-produced files land in the cluster-produced half of the tree, which
# is what the contract asks for.
#
# Idempotent: mkdir -p and ln -sfn. Safe to call from every job start AND from G0.
cl_of_root () { printf '%s' "${CL_SKEL_ROOT:-${CL_DATA_ROOT}/root/Open-Finance}"; }

_cl_link () {   # _cl_link <target> <linkname> <label>
    local tgt="$1" lnk="$2" label="$3"
    if [[ ! -e "${tgt}" ]]; then
        echo "  [!] ${label}: link source is MISSING — ${tgt}"
        echo "      upload it (cluster/upload_manifest.txt, RUNBOOK step 0); the link is still made"
    fi
    ln -sfn "${tgt}" "${lnk}" 2>/dev/null \
        || { echo "  [!] ${label}: could not link ${lnk} -> ${tgt}"; return 1; }
    echo "  link ${label}: $(basename "${lnk}") -> ${tgt}"
    return 0
}

cl_bootstrap_tree () {
    local root proc
    root="$(cl_of_root)"
    proc="${root}/BCB/Egan_et_al_2025_Rep/processed"
    if [[ "${CL_DRYRUN}" == "1" ]]; then
        echo "[dry-run] would build the Open-Finance skeleton at ${root}"
        echo "[dry-run]   dirs:  shared/, Drafts/Deposit Competition/, BCB/Inclusion/,"
        echo "[dry-run]          BCB/Egan_et_al_2025_Rep/processed/PANEL_INTERMED/"
        echo "[dry-run]   links: processed/market_panel.parquet, PANEL_INTERMED/digital_banks_diagnostic.csv,"
        echo "[dry-run]          PANEL_INTERMED/estban_own_branches.csv,"
        echo "[dry-run]          BCB/Inclusion/bcb_banked_mca_panel.csv  <- ${CL_DATA_IN}"
        echo "[dry-run]          processed/ESTIMATION_OUTPUT -> ${CL_DATA_OUT}"
        echo "[dry-run]          => estimation_output() == ${CL_DATA_OUT}, the step dirs under it"
        return 0
    fi
    mkdir -p "${root}/shared" \
             "${root}/Drafts/Deposit Competition" \
             "${root}/BCB/Inclusion" \
             "${proc}/PANEL_INTERMED" \
             "${CL_DATA_OUT}" || return 1
    echo "Open-Finance skeleton: ${root}"
    # The panel is uploaded as PARQUET (39 MB) and only as parquet — the 724 MB CSV never
    # exists on this cluster. utils.paths.market_panel_csv() still names market_panel.csv
    # and utils.sidecar_path() derives the twin from it with .with_suffix('.parquet'), so
    # linking the parquet under its own name is what the Python stack looks for, and the
    # ABSENCE of processed/market_panel.csv is what tells load_panel_cached the parquet is
    # the source rather than a cache to validate. Deliberately NO market_panel.csv link:
    # pointing that name at the parquet bytes would feed pd.read_csv a parquet file.
    _cl_link "${CL_DATA_IN}/market_panel.parquet" \
             "${proc}/market_panel.parquet"                        "market panel (parquet)"
    # A skeleton built before the panel moved to parquet holds a market_panel.csv link.
    # This function is idempotent and runs at every job start, so it heals that here:
    # a CSV that resolves keeps the loader in cache-validation mode against a file the
    # bundle no longer refreshes. Only a SYMLINK is removed — a real file at that path
    # is somebody's deliberate copy and is reported instead.
    if [[ -L "${proc}/market_panel.csv" ]]; then
        rm -f "${proc}/market_panel.csv"
        echo "  removed the stale processed/market_panel.csv link (the panel is parquet now)"
    elif [[ -e "${proc}/market_panel.csv" ]]; then
        echo "  [!] ${proc}/market_panel.csv is a REAL file. The panel ships as parquet;"
        echo "      with a CSV present the loader validates the parquet against THAT file"
        echo "      and may re-read it instead. Move it aside unless you put it there on purpose."
    fi
    _cl_link "${CL_DATA_IN}/digital_banks_diagnostic.csv" \
             "${proc}/PANEL_INTERMED/digital_banks_diagnostic.csv" "digital-bank flags"
    _cl_link "${CL_DATA_IN}/bcb_banked_mca_panel.csv" \
             "${root}/BCB/Inclusion/bcb_banked_mca_panel.csv"      "BCB banked/MCA panel"
    # The D6 event screen (sleep_ident_entry_dynamics.py BRANCH_SIDECAR) reads the ESTBAN
    # own-branch timing sidecar from PANEL_INTERMED. Absent, the screen is skipped with only a
    # printed line, and the cluster keeps entry events the local run drops for branch timing --
    # a different kept-event set, so a different D6 moment, with nothing in the log saying why.
    _cl_link "${CL_DATA_IN}/estban_own_branches.csv" \
             "${proc}/PANEL_INTERMED/estban_own_branches.csv"       "ESTBAN own-branch timing"
    # A REAL directory here would shadow the link and split the tree in half, so
    # refuse rather than overwrite: only an absent path or an existing symlink is
    # replaced. (ln -sfn on a real directory silently creates the link INSIDE it.)
    if [[ -d "${proc}/ESTIMATION_OUTPUT" && ! -L "${proc}/ESTIMATION_OUTPUT" ]]; then
        echo "  [!] ESTIMATION_OUTPUT is a REAL directory, not the expected link to data/output:"
        echo "      ${proc}/ESTIMATION_OUTPUT"
        echo "      Anything it holds was written to the wrong half of the tree. Move it aside"
        echo "      (mv ESTIMATION_OUTPUT ESTIMATION_OUTPUT.stray) and re-run."
        return 1
    fi
    _cl_link "${CL_DATA_OUT}" "${proc}/ESTIMATION_OUTPUT" "ESTIMATION_OUTPUT -> data/output"
    return 0
}

# ── 11. BBL SHARD DISPATCH — one implementation for bbl_run.sh AND the sweep job ──
# The forward-sim array is submitted from two places: bbl_run.sh at launch, and the sweep job
# (bbl_job.sh BBL_STEP=sweep) when it re-runs the shards a launch left missing. Both go through
# cl_bbl_dispatch_fwd, so a re-run is built exactly like the original — same partitions, same
# packing rule, same export list — and the only thing the sweep decides is WHICH indices.
#
# NODE AND QOS SPECS. Hard-coded defaults, each overridable by the env var named in cl_bbl_part.
#   node shape, from   sinfo -p gpu_h100,gpu_h200 -N -o "%N %m %c %G"   (2026-09-24):
#     gpu_h200   2,043,833 MB   48 CPUs   8 x H200
#     gpu_h100   1,000,000 MB   48 CPUs   4 x H100
#   per-user QOS caps (a job counts once however many GPUs it holds; an array TASK is a job):
#     gpu_h200   MaxJobsPU=6    gres/gpu=16
#     gpu_h100   MaxJobsPU=12   gres/gpu=32      a SEPARATE QOS, so the two caps add up
# At one shard per job that is 6 + 12 = 18 GPUs of the 48 allowed; packing k shards into one
# job (one GPU each) is what reaches the GPU cap instead of the job cap.
#
# MEASURED SHARD COST (the 2026-09 run: T=50, 50 shocks, 300 shards, multi-start S=36, P=1;
# COMPLETED GPU shards, whole-job ElapsedRaw and the sacct MaxRSS of the .batch step):
#     time    gpu_h200 n=398 median 19 / p90 23 / max 26 min;  gpu_h100 n=141 median 18 / max 22
#     MaxRSS  median 164 GiB, max 179 GiB  -> MEM=256G is ~30% headroom at T=50
# RSS grows with T (cf_shares_path and simulate_deposits hold N x T paths), so none of this is a
# promise at T=250: the memory probe (bbl_run.sh --probe, BBL_RUNBOOK.md section 1) measures
# per-shard peak RSS and time there, and bbl_sizing.sh turns them into MEM / PACK_H200 / PACK_H100.
#
# DISPATCH STATE lives in data/output/bbl/.dispatch/<key>/, key = E{k}_spec_12_{stage}{psi_tag}:
#   context.env   the run's settings (bbl_run.sh writes it; the sweep and --repair read it)
#   <jobid>.map   the shard indices that fwd job claims, one array task per line
#   map_*.txt     the index list a packed array reads (BBL_SHARD_MAP), one task per line
#   solve.jid     the solve the sweep re-targets when it re-runs a gap
#   probe.env     --probe only: the packed and the unpacked job ids (read by bbl_sizing.sh)
#   events.log    SUBMIT / SWEEP lines, read by bbl_status.sh
#   STOP          written by bbl_cancel.sh: a sweep that finds it re-submits nothing
# A dry run reads this tree but writes nothing to it.

# ── THE DISCOUNT REGISTRY: bbl_discount.env ─────────────────────────────────
# bbl_discount.env, beside this library, is the ONE place the BBL discount factor (BBL_BETA)
# and the forward-simulation horizon (BBL_HORIZON) are set. Every consumer reads it: bbl_run.sh
# (and through it the sim, the wall derivation and the sweep's provenance check), cf_run.sh and
# cf_eq_run.sh (beta only: their horizon is the counterfactual's own), the Julia entry points
# through bbl_discount() in cf_psi_basis.jl, and the Python ones through bbl_shards.read_bbl_discount.
# An explicit flag or environment variable still overrides it, and every consumer logs which of
# the three a value came from. BBL_DISCOUNT_ENV points the readers at another file (tests).
#
# The file is READ, never sourced: KEY=VALUE lines, '#' comments, a trailing CR tolerated, and
# nothing in it is executed.
CL_BBL_DISCOUNT_FILE="${BBL_DISCOUNT_ENV:-${CL_ROOT}/bbl_discount.env}"
cl_bbl_discount_get () {   # cl_bbl_discount_get <KEY> -> the value on stdout; rc 1 if file or key is missing
    local f="${CL_BBL_DISCOUNT_FILE}" line v=""
    [[ -f "${f}" ]] || return 1
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%$'\r'}"
        [[ "${line}" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*([^#[:space:]]+) ]] || continue
        [[ "${BASH_REMATCH[1]}" == "$1" ]] && v="${BASH_REMATCH[2]}"
    done < "${f}"
    [[ -n "${v}" ]] || return 1
    printf '%s' "${v}"
}
# cl_bbl_discount_fill <VAR> <KEY>: when VAR is empty, set it from the registry and set VAR_SRC to
# "bbl_discount.env"; otherwise leave both as the caller set them. rc 1 (with the reason on
# stderr) when VAR is empty and the registry cannot supply it -- callers refuse rather than fall
# back to a literal, which is how a beta nobody chose used to reach a run.
cl_bbl_discount_fill () {
    local var="$1" key="$2" v
    [[ -n "${!var:-}" ]] && return 0
    if v="$(cl_bbl_discount_get "${key}")"; then
        printf -v "${var}" '%s' "${v}"
        printf -v "${var}_SRC" '%s' "$(basename "${CL_BBL_DISCOUNT_FILE}")"
        return 0
    fi
    cl_err "REFUSING: ${var} is not set and ${key} cannot be read from ${CL_BBL_DISCOUNT_FILE}."
    cl_err "  That file is the single source of the BBL discount factor and horizon: upload it beside"
    cl_err "  the scripts, or pass the value explicitly (flag or ${var}=...)."
    return 1
}

cl_bbl_pkey () {   # partition -> H200 | H100 | GPU? | CPU
    case "$1" in gpu_h200) printf 'H200' ;; gpu_h100) printf 'H100' ;; gpu*) printf 'GPU?' ;; *) printf 'CPU' ;; esac
}
cl_bbl_part () {   # cl_bbl_part <partition> <field>
    local K; K="$(cl_bbl_pkey "$1")"
    case "${K}:$2" in
        H200:node_mem_mb) printf '%s' "${BBL_H200_NODE_MEM_MB:-2043833}" ;;
        H200:node_cpus)   printf '%s' "${BBL_H200_NODE_CPUS:-48}" ;;
        H200:node_gpus)   printf '%s' "${BBL_H200_NODE_GPUS:-8}" ;;
        H200:gpu_type)    printf '%s' "${BBL_H200_GPU_TYPE:-h200}" ;;
        H200:gpu_mem_mib) printf '%s' "${BBL_H200_GPU_MEM_MIB:-143771}" ;;
        H200:max_jobs)    printf '%s' "${BBL_H200_MAX_JOBS:-6}" ;;
        H200:max_gpus)    printf '%s' "${BBL_H200_MAX_GPUS:-16}" ;;
        H200:wall_cap)    printf '%s' "${BBL_H200_WALL_CAP:-2-00:00:00}" ;;
        H200:pack)        printf '%s' "${PACK_H200:-${PACK:-4}}" ;;
        H200:mem)         printf '%s' "${MEM_H200:-${MEM:-256G}}" ;;
        H100:node_mem_mb) printf '%s' "${BBL_H100_NODE_MEM_MB:-1000000}" ;;
        H100:node_cpus)   printf '%s' "${BBL_H100_NODE_CPUS:-48}" ;;
        H100:node_gpus)   printf '%s' "${BBL_H100_NODE_GPUS:-4}" ;;
        H100:gpu_type)    printf '%s' "${BBL_H100_GPU_TYPE:-h100}" ;;
        # ASSUMED 80 GB parts (nvidia-smi reports 81559 MiB). Not measured on Bouchet: read it
        # off `nvidia-smi` in any gpu_h100 job log and override if it differs.
        H100:gpu_mem_mib) printf '%s' "${BBL_H100_GPU_MEM_MIB:-81559}" ;;
        H100:max_jobs)    printf '%s' "${BBL_H100_MAX_JOBS:-12}" ;;
        H100:max_gpus)    printf '%s' "${BBL_H100_MAX_GPUS:-32}" ;;
        H100:wall_cap)    printf '%s' "${BBL_H100_WALL_CAP:-2-00:00:00}" ;;
        # PACK is the common default of both partitions; unset, h200 takes 4 and h100 3, because
        # at the default MEM=256G four shards (1,048,576 MB) do not fit a 1,000,000 MB h100 node.
        # MEM=256G is the T=50 sizing: at another T, MEM and PACK come from the memory probe
        # (bbl_sizing.sh prints --mem-h200/--mem-h100/--pack-h200/--pack-h100).
        H100:pack)        printf '%s' "${PACK_H100:-${PACK:-3}}" ;;
        H100:mem)         printf '%s' "${MEM_H100:-${MEM:-256G}}" ;;
        CPU:wall_cap)     printf '%s' "${BBL_CPU_WALL_CAP:-1-00:00:00}" ;;
        CPU:pack)         printf '1' ;;
        CPU:mem)          printf '%s' "${MEM:-256G}" ;;
        *) return 1 ;;
    esac
}

cl_mem_mb () {   # "256G" -> 262144 (SLURM units: K M G T, bare number = MB)
    local s="${1^^}"
    [[ "${s}" =~ ^([0-9]+)([KMGT]?)B?$ ]] || { echo "cl_mem_mb: cannot parse '$1'" >&2; return 2; }
    case "${BASH_REMATCH[2]}" in
        K) printf '%s' $(( (BASH_REMATCH[1] + 1023) / 1024 )) ;;
        ""|M) printf '%s' "${BASH_REMATCH[1]}" ;;
        G) printf '%s' $(( BASH_REMATCH[1] * 1024 )) ;;
        T) printf '%s' $(( BASH_REMATCH[1] * 1048576 )) ;;
    esac
}
cl_wall_min () {   # SLURM time (M, M:S, H:M:S, D-H, D-H:M, D-H:M:S) -> whole minutes, rounded up
    local w="$1" d=0 h=0 m=0 s=0 a b c
    if [[ "${w}" == *-* ]]; then
        d="${w%%-*}"; w="${w#*-}"
        IFS=: read -r a b c <<< "${w}"
        h="${a:-0}"; m="${b:-0}"; s="${c:-0}"
    else
        IFS=: read -r a b c <<< "${w}"
        if [[ -n "${c}" ]]; then h="${a}"; m="${b}"; s="${c}"
        elif [[ -n "${b}" ]]; then m="${a}"; s="${b}"
        else m="${a}"; fi
    fi
    printf '%s' $(( 10#${d} * 1440 + 10#${h} * 60 + 10#${m} + (10#${s} + 59) / 60 ))
}
cl_min_to_wall () {   # minutes -> D-HH:MM:00 | HH:MM:00
    local t="$1" d h m
    d=$(( t / 1440 )); h=$(( (t % 1440) / 60 )); m=$(( t % 60 ))
    if (( d > 0 )); then printf '%d-%02d:%02d:00' "${d}" "${h}" "${m}"; else printf '%02d:%02d:00' "${h}" "${m}"; fi
}

# Index specs. bbl_shards.py carries the Python twins (compress_ranges / expand_spec / pack) and
# the check suite runs both on the same inputs, so the sweep's Python view of a gap and the
# shell's submission of it cannot disagree about which indices "3,50-299" names.
cl_compress_ranges () {   # cl_compress_ranges 3 50 51 52 -> "3,50-52"   ("" for no input)
    printf '%s\n' "$@" | awk 'NF' | sort -n -u | awk '
        NR == 1 { a = $1; b = $1; next }
        $1 == b + 1 { b = $1; next }
        { o = o (o == "" ? "" : ",") (a == b ? a : a "-" b); a = $1; b = $1 }
        END { if (NR) o = o (o == "" ? "" : ",") (a == b ? a : a "-" b); print o }'
}
cl_expand_spec () {   # cl_expand_spec "3,50-52%8" -> "3 50 51 52"   (a %throttle suffix is ignored)
    local s="${1%%\%*}"
    printf '%s\n' "${s}" | tr ', ' '\n\n' | awk '
        !NF { next }
        /^[0-9]+-[0-9]+$/ { split($0, r, "-"); if (r[2] + 0 < r[1] + 0) { print "descending range " $0 > "/dev/stderr"; bad = 1; exit }
                             for (i = r[1] + 0; i <= r[2] + 0; i++) print i; next }
        /^[0-9]+$/ { print $0 + 0; next }
        { print "not an index or a range: " $0 > "/dev/stderr"; bad = 1; exit }
        END { exit bad }' | sort -n -u | paste -sd' ' -
}
cl_bbl_pack_tasks () {   # cl_bbl_pack_tasks <k> <ids...> -> one task per line, k ids each (last may be short)
    local k="$1"; shift
    printf '%s\n' "$@" | awk 'NF' | sort -n -u | awk -v k="${k}" '
        { t = t (t == "" ? "" : " ") $1; if (++n % k == 0) { print t; t = "" } }
        END { if (t != "") print t }'
}

cl_bbl_key ()  { printf 'E%s_spec_12_%s%s' "$1" "$2" "${3:-}"; }   # <routine> <stage> <psi_tag>
cl_bbl_rtag () {                                                     # job-name tag: E3_ms1
    if [[ "$2" == "extended" ]]; then printf 'E%s%s' "$1" "${3:-}"; else printf 'E%s_%s%s' "$1" "$2" "${3:-}"; fi
}
cl_bbl_dispatch_dir () { printf '%s/%s' "${CL_BBL_DISPATCH_ROOT:-${CL_STEP_BBL}/.dispatch}" "$1"; }

cl_bbl_event () {   # cl_bbl_event <key> <text>
    local d; d="$(cl_bbl_dispatch_dir "$1")"
    if [[ "${CL_DRYRUN:-0}" == "1" ]]; then cl_log "  [dry-run] event ${1}: ${2}"; return 0; fi
    mkdir -p "${d}" 2>/dev/null || return 0
    printf '%s %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$2" >> "${d}/events.log" 2>/dev/null || true
    return 0
}

# The run's settings, saved once at launch so every later re-run is built from the SAME values
# rather than from whatever the environment of the job that re-submits happens to hold.
CL_BBL_CTX_VARS="BBL_KEY BBL_RTAG BBL_ROUTINE BBL_STAGE R SEED N_SHARDS SHOCKS HORIZON BETA \
HORIZON_SRC BETA_SRC PSI_TAG MULTI_START N_PATHS BBL_FWD_EXTRA POLICY_CSV FWD_GPU FWD_PARTITIONS \
SWEEP_PARTITIONS PACK PACK_H200 PACK_H100 MEM MEM_H200 MEM_H100 BBL_THREADS_PER_SHARD CPU_PARTITION \
CPU_CONSTRAINT SHARD_TIME BBL_SHARD_MIN_T50 BBL_SHARD_MIN_T50_CPU BBL_PACK_CONTENTION BBL_WALL_SAFETY \
BBL_WALL_FLOOR_MIN BBL_WALL_ROUND_MIN BBL_RETRY_WALL_MULT BBL_MAX_RETRIES RUN_EPOCH BBL_FRESH PROBE"
cl_bbl_ctx_write () {   # cl_bbl_ctx_write <key>
    local d f v
    d="$(cl_bbl_dispatch_dir "$1")"
    if [[ "${CL_DRYRUN:-0}" == "1" ]]; then cl_log "  [dry-run] would write ${d}/context.env"; return 0; fi
    mkdir -p "${d}" || return 1
    f="${d}/.context.env.$$"
    {
        echo "# bbl_run.sh $(date +%Y-%m-%dT%H:%M:%S) -- read by the sweep job and by --repair; do not edit mid-run"
        for v in ${CL_BBL_CTX_VARS}; do printf '%s=%q\n' "${v}" "${!v-}"; done
        if [[ -n "${ARRAY_THROTTLE+set}" ]]; then printf 'ARRAY_THROTTLE_SET=1\nARRAY_THROTTLE=%q\n' "${ARRAY_THROTTLE}"
        else echo "ARRAY_THROTTLE_SET=0"; fi
    } > "${f}" && mv -f "${f}" "${d}/context.env"
}
cl_bbl_ctx_load () {   # cl_bbl_ctx_load <file>: sets every CL_BBL_CTX_VARS variable in THIS shell
    [[ -f "$1" ]] || return 1
    ARRAY_THROTTLE_SET=0
    . "$1"
    if [[ "${ARRAY_THROTTLE_SET}" != "1" ]]; then unset ARRAY_THROTTLE; fi
    return 0
}

# cl_bbl_check_packing <partition>: the submit-time refusal. A packed job is ONE node, so k
# shards must fit its GPUs, its memory and its CPUs; asking for more is a job that never starts
# (or, for memory, one whose k shards share an OOM kill).
cl_bbl_check_packing () {
    local p="$1" K k mem mb node gpus cpus thr rc=0
    K="$(cl_bbl_pkey "${p}")"
    [[ "${K}" == "CPU" ]] && return 0
    if [[ "${K}" == "GPU?" ]]; then
        cl_err "REFUSING: no node/QOS specs for GPU partition '${p}' (known: gpu_h200, gpu_h100); add them to cl_bbl_part."
        return 1
    fi
    k="$(cl_bbl_part "${p}" pack)"; mem="$(cl_bbl_part "${p}" mem)"
    [[ "${k}" =~ ^[1-9][0-9]*$ ]] || { cl_err "REFUSING: PACK for ${p} is '${k}'; a positive integer is required."; return 1; }
    mb="$(cl_mem_mb "${mem}")" || { cl_err "REFUSING: cannot read MEM '${mem}' for ${p}."; return 1; }
    node="$(cl_bbl_part "${p}" node_mem_mb)"; gpus="$(cl_bbl_part "${p}" node_gpus)"
    cpus="$(cl_bbl_part "${p}" node_cpus)"; thr="${BBL_THREADS_PER_SHARD:-8}"
    if (( k > gpus )); then
        cl_err "REFUSING: PACK=${k} on ${p} exceeds the ${gpus} GPUs of one node (a packed job is one node)."; rc=1
    fi
    if (( k * mb > node )); then
        cl_err "REFUSING: PACK=${k} x MEM ${mem} = $(( k * mb )) MB on ${p}, but one node schedules ${node} MB."
        cl_err "  Size MEM and PACK from the memory probe: bash bbl_sizing.sh (BBL_RUNBOOK.md section 1)."
        rc=1
    fi
    if (( k * thr > cpus )); then
        cl_err "REFUSING: PACK=${k} x ${thr} threads = $(( k * thr )) CPUs on ${p}; one node has ${cpus}."; rc=1
    fi
    return ${rc}
}

# cl_bbl_throttle <partition> <pack> -> "N|why"   (N empty = no throttle)
# The binding QOS limit is whichever runs out first, the job count or the GPU count, so the
# array concurrency is min(MaxJobsPU, floor(MaxGPU / PACK)). ARRAY_THROTTLE, when SET (even
# empty, which means none), overrides the derivation on every partition.
cl_bbl_throttle () {
    local p="$1" k="$2" mj mg t
    if [[ -n "${ARRAY_THROTTLE+set}" ]]; then
        printf '%s|ARRAY_THROTTLE=%s (explicit override)' "${ARRAY_THROTTLE}" "${ARRAY_THROTTLE:-<empty: none>}"; return 0
    fi
    if [[ "$(cl_bbl_pkey "${p}")" == "CPU" ]]; then printf '|none (CPU partition)'; return 0; fi
    mj="$(cl_bbl_part "${p}" max_jobs)"; mg="$(cl_bbl_part "${p}" max_gpus)"
    t=$(( mg / k )); (( t > mj )) && t=${mj}; (( t < 1 )) && t=1
    printf '%s|min(MaxJobsPU=%s, floor(MaxGPU=%s / PACK=%s) = %s) = %s jobs = %s shards at once' \
        "${t}" "${mj}" "${mg}" "${k}" "$(( mg / k ))" "${t}" "$(( t * k ))"
}

# cl_bbl_wall_minutes <partition> <pack> [mult] [mult_label] -> "minutes|arithmetic"
# The wall is DERIVED unless SHARD_TIME is set:
#     BBL_SHARD_MIN_T50    26 min, the slowest COMPLETED GPU shard of the 2026-09 run at T=50
#   x T/50                 evolving states re-evaluate the shares every period: linear in T
#   x shard size           (SHOCKS/50) x (300/N_SHARDS), relative to the measured shard
#   x BBL_PACK_CONTENTION  1.3, for shards sharing a node (memory bandwidth, PCIe, other jobs)
#   x BBL_WALL_SAFETY      1.5
#   x mult                 1.5^n on the sweep's n-th re-run; BBL_PROBE_WALL_MULT for the probe
# rounded UP to a multiple of BBL_WALL_ROUND_MIN (30), floored at BBL_WALL_FLOOR_MIN (30) and
# capped at the partition's per-job limit. At T=250: 26 x 5 x 1 x 1.3 x 1.5 = 253.5 min, rounded
# up to 270 min = 04:30:00. A packed job's wall is its slowest child's. The CPU base is NOT a
# measurement: the last CPU run hit its 16 h wall.
cl_bbl_wall_minutes () {
    local p="$1" k="$2" mult="${3:-1}" lab="${4:-retry}" K base capm m why
    K="$(cl_bbl_pkey "${p}")"
    capm="$(cl_wall_min "$(cl_bbl_part "${p}" wall_cap)")"
    if [[ -n "${SHARD_TIME:-}" ]]; then
        m="$(awk -v b="$(cl_wall_min "${SHARD_TIME}")" -v x="${mult}" 'BEGIN{ v = b * x; i = int(v); if (i < v) i++; print i }')"
        why="SHARD_TIME=${SHARD_TIME} (explicit)$( [[ "${mult}" == "1" ]] || printf ' x %s %s' "${lab}" "${mult}" )"
    else
        if [[ "${K}" == "CPU" ]]; then base="${BBL_SHARD_MIN_T50_CPU:-960}"; else base="${BBL_SHARD_MIN_T50:-26}"; fi
        read -r m why < <(awk -v b="${base}" -v T="${HORIZON:-50}" -v sh="${SHOCKS:-50}" -v ns="${N_SHARDS:-300}" \
            -v c="${BBL_PACK_CONTENTION:-1.3}" -v s="${BBL_WALL_SAFETY:-1.5}" -v x="${mult}" -v lab="${lab}" \
            -v fl="${BBL_WALL_FLOOR_MIN:-30}" -v rd="${BBL_WALL_ROUND_MIN:-30}" -v k="${k}" 'BEGIN{
            r = (sh / 50) * (300 / ns); v = b * (T / 50) * r * c * s * x
            i = int(v); if (i < v) i++
            if (rd > 0) i = int((i + rd - 1) / rd) * rd
            if (i < fl) i = fl
            printf "%d %g min x T/50 (%g/50 = %.3g) x shard size (%g shocks / %g shards vs 50/300 = %.3g) x contention %g x safety %g%s = %.1f min, rounded up to a multiple of %g = %d min (PACK=%d)", \
                i, b, T, T / 50, sh, ns, r, c, s, (x != 1 ? sprintf(" x %s %g", lab, x) : ""), v, rd, i, k }')
        if [[ "${K}" == "CPU" ]]; then why="${why} [CPU base ${base} min is NOT measured: the last CPU run hit its 16 h wall]"; fi
    fi
    if (( m > capm )); then
        why="${why}; CAPPED at the ${p} limit $(cl_bbl_part "${p}" wall_cap): a shard that needs longer can never finish there"
        m="${capm}"
    fi
    printf '%s|%s' "${m}" "${why}"
}

_cl_bbl_pack_for () { if [[ "${FWD_GPU:-1}" == "1" ]]; then cl_bbl_part "$1" pack; else printf '1'; fi; }

_cl_bbl_claim () {   # _cl_bbl_claim <jobid> <one task per line>: what that job computes
    cl_bbl_event "${BBL_KEY}" "SUBMIT fwd jid=$1 shards=$(cl_compress_ranges $2)"
    [[ "${CL_DRYRUN:-0}" == "1" ]] && return 0
    local d; d="$(cl_bbl_dispatch_dir "${BBL_KEY}")"
    mkdir -p "${d}" && printf '%s\n' "$2" > "${d}/$1.map"
}

# cl_bbl_dispatch_fwd "<ids>" [wall_mult] [--dependency=...]
# Submits the fwd_sim work for exactly these shard indices and leaves the job ids, colon-joined,
# in CL_BBL_JIDS. Call it directly, never inside $( ): it reports with cl_say. wall_mult scales
# the derived wall (the sweep's 1.5^n, the probe's BBL_PROBE_WALL_MULT); BBL_MULT_LABEL names it
# in the logged arithmetic (default "retry").
#   FWD_PARTITIONS  one or more GPU partitions ("gpu_h200 gpu_h100"); each gets a DISJOINT
#                   contiguous block of the indices, sized in proportion to its concurrent
#                   capacity (throttle x PACK), so both QOS caps fill at the same pace.
#   PACK=1          the one-shard-per-job submission: --array=<the indices themselves>,
#                   --gpus=<type>:1, and the task id IS the shard id.
#   PACK=k>1        an index LIST, not a range: the indices are chunked into tasks of k, written
#                   to a map file (one task per line) and the array runs 0..tasks-1 with
#                   --gpus=<type>:k. A short last chunk goes out as its own job requesting only
#                   the GPUs it uses.
# Reads: BBL_KEY BBL_RTAG BBL_ROUTINE BBL_STAGE R SEED N_SHARDS BBL_FWD_EXTRA PSI_TAG FWD_GPU
#        FWD_PARTITIONS CPU_PARTITION CPU_CONSTRAINT LOGD + the knobs cl_bbl_part/_wall/_throttle read.
cl_bbl_dispatch_fwd () {
    local -a ids parts caps
    local mult="${2:-1}" dep="${3:-}" p k thr tot=0 j n start=0 cnt
    read -r -a ids <<< "$(printf '%s\n' $1 | awk 'NF' | sort -n -u | paste -sd' ' -)"
    CL_BBL_JIDS=""
    (( ${#ids[@]} > 0 )) || return 0
    if [[ "${FWD_GPU:-1}" == "1" ]]; then read -r -a parts <<< "${FWD_PARTITIONS:-gpu_h200}"
    else parts=("${CPU_PARTITION:-day}"); fi
    n=${#ids[@]}
    for j in "${!parts[@]}"; do
        p="${parts[$j]}"
        cl_bbl_check_packing "${p}" || return 1
        k="$(_cl_bbl_pack_for "${p}")"
        thr="$(cl_bbl_throttle "${p}" "${k}")"; thr="${thr%%|*}"
        caps[$j]=$(( ${thr:-${n}} * k ))
        tot=$(( tot + caps[j] ))
    done
    for j in "${!parts[@]}"; do
        p="${parts[$j]}"; k="$(_cl_bbl_pack_for "${p}")"
        if (( j == ${#parts[@]} - 1 )); then
            cnt=$(( n - start ))
        else
            cnt=$(( n * caps[j] / tot )); cnt=$(( (cnt + k / 2) / k * k ))
            if (( cnt > n - start )); then cnt=$(( n - start )); fi
        fi
        (( cnt > 0 )) || continue
        _cl_bbl_submit_part "${p}" "${k}" "${mult}" "${dep}" "${ids[@]:start:cnt}" || return 1
        start=$(( start + cnt ))
    done
    return 0
}

_cl_bbl_submit_part () {   # <partition> <pack> <retry_mult> <dep> <ids...>
    local p="$1" k="$2" mult="$3" dep="$4"; shift 4
    local -a ids=("$@") lines last
    local wi wall ti thr thrs name ex spec jid mem type nfull
    wi="$(cl_bbl_wall_minutes "${p}" "${k}" "${mult}" "${BBL_MULT_LABEL:-retry}")"; wall="$(cl_min_to_wall "${wi%%|*}")"
    ti="$(cl_bbl_throttle "${p}" "${k}")"; thr="${ti%%|*}"; thrs="${thr:+%${thr}}"
    name="bbl_fwd_${BBL_RTAG}"
    ex="ALL,BBL_ROUTINE=${BBL_ROUTINE},BBL_STAGE=${BBL_STAGE},R=${R},SEED=${SEED},BBL_STEP=fwd_sim,N_SHARDS=${N_SHARDS},BBL_EXTRA=${BBL_FWD_EXTRA},BBL_RTAG=${BBL_RTAG},BBL_KEY=${BBL_KEY},PSI_TAG=${PSI_TAG}"
    cl_log "  ${name} [${p}] wall ${wall} = ${wi#*|}"
    cl_log "  ${name} [${p}] throttle ${ti#*|}"
    mem="$(cl_bbl_part "${p}" mem)"
    if [[ "${FWD_GPU:-1}" != "1" ]]; then
        spec="$(cl_compress_ranges "${ids[@]}")"
        jid=$(cl_sbatch -J "${name}" -t "${wall}" --mem="${mem}" \
            -o "${LOGD}/${name}_%A_%a.out" -e "${LOGD}/${name}_%A_%a.err" ${dep} \
            --partition="${p}" --constraint="${CPU_CONSTRAINT:-cpugen:turin}" \
            --array="${spec}${thrs}" --export="${ex},CF_GPU=0" "${CL_ROOT}/bbl_job.sh") || return 1
        cl_require_jid "${jid}" "${name}" || return 1
        _cl_bbl_claim "${jid}" "$(printf '%s\n' "${ids[@]}")"
        CL_BBL_JIDS="${CL_BBL_JIDS:+${CL_BBL_JIDS}:}${jid}"
        cl_say "  ${name} [${p}] ${#ids[@]} shards, 1 per job, --array=${spec}${thrs} -> ${jid}"
        return 0
    fi
    type="$(cl_bbl_part "${p}" gpu_type)"
    if (( k == 1 )); then
        spec="$(cl_compress_ranges "${ids[@]}")"
        jid=$(cl_sbatch -J "${name}" -t "${wall}" --mem="${mem}" \
            -o "${LOGD}/${name}_%A_%a.out" -e "${LOGD}/${name}_%A_%a.err" ${dep} \
            --partition="${p}" --gpus="${type}:1" \
            --array="${spec}${thrs}" --export="${ex},CF_GPU=1" "${CL_ROOT}/bbl_job.sh") || return 1
        cl_require_jid "${jid}" "${name}" || return 1
        _cl_bbl_claim "${jid}" "$(printf '%s\n' "${ids[@]}")"
        CL_BBL_JIDS="${CL_BBL_JIDS:+${CL_BBL_JIDS}:}${jid}"
        cl_say "  ${name} [${p}] ${#ids[@]} shards, 1 per job (--gpus=${type}:1), --array=${spec}${thrs} -> ${jid}"
        return 0
    fi
    mapfile -t lines < <(cl_bbl_pack_tasks "${k}" "${ids[@]}")
    read -r -a last <<< "${lines[${#lines[@]}-1]}"
    nfull=${#lines[@]}
    if (( ${#last[@]} < k )); then nfull=$(( nfull - 1 )); fi
    if (( nfull > 0 )); then
        _cl_bbl_submit_packed "${p}" "${k}" "${wall}" "${thrs}" "${dep}" "${ex}" "${lines[@]:0:nfull}" || return 1
    fi
    if (( nfull < ${#lines[@]} )); then
        _cl_bbl_submit_packed "${p}" "${#last[@]}" "${wall}" "" "${dep}" "${ex}" "${lines[@]:nfull}" || return 1
    fi
    return 0
}

_cl_bbl_submit_packed () {   # <partition> <gpus per job> <wall> <%throttle> <dep> <export> <task lines...>
    local p="$1" g="$2" wall="$3" thrs="$4" dep="$5" ex="$6"; shift 6
    local -a lines=("$@")
    local name="bbl_fwd_${BBL_RTAG}" type mem mb tps d map arr jid n=${#lines[@]} l
    type="$(cl_bbl_part "${p}" gpu_type)"; mem="$(cl_bbl_part "${p}" mem)"; mb="$(cl_mem_mb "${mem}")"
    tps="${BBL_THREADS_PER_SHARD:-8}"
    d="$(cl_bbl_dispatch_dir "${BBL_KEY}")"
    map="${d}/map_$(date +%Y%m%d_%H%M%S)_${RANDOM}${RANDOM}_${p}_x${g}.txt"
    if [[ "${CL_DRYRUN:-0}" == "1" ]]; then
        cl_log "  [dry-run] would write ${map} (${n} task line(s)):"
        for l in "${lines[@]}"; do cl_log "      ${l}"; done
    else
        mkdir -p "${d}" && printf '%s\n' "${lines[@]}" > "${map}" || { cl_err "cannot write shard map ${map}"; return 1; }
    fi
    if (( n > 1 )); then arr="0-$(( n - 1 ))${thrs}"; else arr="0"; fi
    jid=$(cl_sbatch -J "${name}" -t "${wall}" --mem="$(( g * mb ))M" --cpus-per-task="$(( g * tps ))" \
        -o "${LOGD}/${name}_pack_%A_%a.out" -e "${LOGD}/${name}_pack_%A_%a.err" ${dep} \
        --partition="${p}" --gpus="${type}:${g}" --array="${arr}" \
        --export="${ex},CF_GPU=1,BBL_SHARD_MAP=${map},BBL_PACK=${g},BBL_THREADS_PER_SHARD=${tps}" \
        "${CL_ROOT}/bbl_job.sh") || return 1
    cl_require_jid "${jid}" "${name}" || return 1
    _cl_bbl_claim "${jid}" "$(printf '%s\n' "${lines[@]}")"
    CL_BBL_JIDS="${CL_BBL_JIDS:+${CL_BBL_JIDS}:}${jid}"
    cl_say "  ${name} [${p}] PACK=${g}: ${n} job(s) x ${g} shard(s) ($(cl_compress_ranges ${lines[*]})), --gpus=${type}:${g} --mem=$(( g * mb ))M --cpus-per-task=$(( g * tps )), --array=${arr} -> ${jid}"
}

# Live jobs, by name. squeue only: safe on the login node, and absent (so: nothing live) on a
# machine without SLURM, which is what a local dry run sees. COMPLETING is not live: an array the
# sweep was chained afterany can still list its last tasks as CG for a moment, and counting those
# would make the sweep wait on work that is already over.
cl_bbl_live_named () {   # cl_bbl_live_named <name>[,<name>...] -> base job ids, space-separated
    command -v squeue >/dev/null 2>&1 || return 0
    squeue -h -u "${USER:-$(id -un)}" -n "$1" -t PENDING,RUNNING,CONFIGURING,SUSPENDED,REQUEUED -o '%F' 2>/dev/null \
        | awk 'NF' | sort -u | paste -sd' ' -
}
# cl_bbl_live_claims <key> <rtag> -> one line per live fwd job: "<jid> <shard ids...>", or
# "<jid> UNKNOWN" for a live job this tree has no claim for (submitted by hand, or before the
# dispatch registry existed) -- which callers must treat as claiming EVERY index.
cl_bbl_live_claims () {
    local d j
    d="$(cl_bbl_dispatch_dir "$1")"
    for j in $(cl_bbl_live_named "bbl_fwd_$2"); do
        if [[ -f "${d}/${j}.map" ]]; then printf '%s %s\n' "${j}" "$(tr -s ' \n' '  ' < "${d}/${j}.map")"
        else printf '%s UNKNOWN\n' "${j}"; fi
    done
}

# cl_bbl_submit_sweep <retry> [sbatch args, e.g. --hold --dependency=afterany:...] -> job id on
# stdout ONLY. A small `day` job (no Julia, no GPU); no cpugen pin, since it loads no sysimage.
# Callers submit it --hold and release it once the solve it will re-target is registered.
cl_bbl_submit_sweep () {
    local r="$1" name="bbl_sweep_${BBL_RTAG}"; shift
    CL_NO_CPUGEN=1 cl_sbatch -J "${name}" -t "${BBL_SWEEP_TIME:-00:30:00}" \
        --partition="${SWEEP_PARTITION:-day}" --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
        -o "${LOGD}/${name}_%j.out" -e "${LOGD}/${name}_%j.err" "$@" \
        --export="ALL,BBL_STEP=sweep,BBL_KEY=${BBL_KEY},BBL_RTAG=${BBL_RTAG},BBL_ROUTINE=${BBL_ROUTINE},BBL_STAGE=${BBL_STAGE},R=${R},SEED=${SEED},PSI_TAG=${PSI_TAG},BBL_RETRY=${r}" \
        "${CL_ROOT}/bbl_job.sh"
}

# cl_bbl_retarget_solve <key> <new sweep jid>: move the pending solve's afterok onto the next
# sweep. This is what keeps ONE solve job id for the life of a routine however many re-runs its
# shards need, so the tables, the archive and the CF jobs (all afterok that id) never have to be
# re-chained. Returns 1 when there is a registered solve that cannot be moved (it is gone or no
# longer pending): the caller then withdraws its re-run instead of leaving an orphan chain.
cl_bbl_retarget_solve () {
    local d s st
    d="$(cl_bbl_dispatch_dir "$1")"
    s="$(cat "${d}/solve.jid" 2>/dev/null || true)"
    if [[ -z "${s}" ]]; then cl_say "  no solve registered for $1 (--no-solve?): nothing to re-target"; return 0; fi
    if [[ "${CL_DRYRUN:-0}" == "1" ]]; then cl_say "  [dry-run] scontrol update JobId=${s} Dependency=afterok:$2"; return 0; fi
    st="$(squeue -h -j "${s}" -o '%T' 2>/dev/null | head -1 || true)"
    if [[ "${st}" != "PENDING" ]]; then
        cl_err "solve ${s} is ${st:-no longer queued}, not PENDING: its dependency cannot be moved."
        return 1
    fi
    scontrol update JobId="${s}" Dependency="afterok:$2" || return 1
    cl_say "  solve ${s}: dependency moved to afterok:$2"
}

# cl_bbl_check_curve <T> <multi_start 0|1> [csv] [note]: the Focus forward curve must reach h=T.
# Beyond its last horizon bbl_fwd_sim.jl repeats the last quoted rate (load_forward_rf,
# _load_rf_vintages), which at T=250 against a curve built to h=50 would be a flat tail over
# 200 of the 250 quarters. Refused unless ALLOW_RF_PAD=1. Multi-start reads the vintages file
# and checks its SHORTEST vintage; single-start reads forward_rf_qoq.csv. A missing file is not
# this check's business (cl_need_rf_curve / cl_need_rf_vintages report it). note=1 checks a file
# the run does NOT read: a short one is reported on one line and never refused.
cl_bbl_check_curve () {
    local T="$1" ms="${2:-0}" f="${3:-}" note="${4:-0}" hmax what
    if [[ -z "${f}" ]]; then
        if [[ "${ms}" == "1" ]]; then f="${CL_DATA_IN}/forward_rf_vintages.csv"; else f="${CL_DATA_IN}/forward_rf_qoq.csv"; fi
    fi
    [[ -f "${f}" ]] || return 0
    hmax="$(awk -F, -v ms="${ms}" '
        NR == 1 { for (i = 1; i <= NF; i++) { g = $i; gsub(/^[ \t\r"]+|[ \t\r"]+$/, "", g); c[g] = i }
                  if (!("h" in c) || (ms == "1" && !("start_q" in c))) { bad = 1; exit }
                  next }
        { h = $(c["h"]) + 0
          if (ms == "1") { s = $(c["start_q"]); if (h >= 1 && h > m[s]) m[s] = h }
          else if (h > mx) mx = h }
        END { if (bad) { print -1; exit }
              if (ms == "1") { n = 0; for (s in m) { if (n == 0 || m[s] < mn) mn = m[s]; n++ }; print (n ? mn : 0) }
              else print mx + 0 }' "${f}")"
    if [[ "${ms}" == "1" ]]; then what="$(basename "${f}") (its SHORTEST launch-quarter vintage)"; else what="$(basename "${f}")"; fi
    if (( hmax < 0 )); then cl_err "curve-length check: ${f} has no 'h'$([[ "${ms}" == "1" ]] && printf "/'start_q'" || true) column."; return 1; fi
    if (( T <= hmax )); then cl_log "  forward curve OK: T=${T} <= h_max=${hmax} in ${what}"; return 0; fi
    if [[ "${note}" == "1" ]]; then
        cl_say "  note: ${what} runs to h=${hmax} < T=${T}. This run does not read it; rebuild it to h >= ${T} before a run that does."
        return 0
    fi
    cl_err "T=${T} EXCEEDS the forward-curve length: ${what} runs to h=${hmax}."
    cl_err "  bbl_fwd_sim.jl would fill h=$(( hmax + 1 ))..${T} by repeating the last quoted rate, a flat tail nobody"
    cl_err "  forecast over $(( T - hmax )) of the ${T} quarters the discounted sums integrate."
    cl_err "  Rebuild the curve to h >= ${T} locally (needs internet) and upload it to data/input/:"
    if [[ "${ms}" == "1" ]]; then
        cl_err "     python scrape_forward_rf.py --vintage-from 2016Q1 --vintage-to 2024Q4 --horizon ${T}"
    else
        cl_err "     python scrape_forward_rf.py --horizon ${T} --start 2026Q1"
    fi
    if [[ "${ALLOW_RF_PAD:-0}" == "1" ]]; then cl_err "  ALLOW_RF_PAD=1: continuing WITH the padded tail, deliberately."; return 0; fi
    cl_err "  ALLOW_RF_PAD=1 accepts the padding deliberately."
    return 1
}
