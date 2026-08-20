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
#   Never this. Run  bash cluster_preflight.sh  first, then blp_run.sh /
#   bbl_run.sh / cf_run.sh / pipeline_run.sh / cf_eq_run.sh.
#
# THESE SCRIPTS ARE NEW. The submit_*.sh / zip_*.sh scripts they replace are still
# present and still work; they are the fallback and are retired only after one
# successful cluster cycle. Nothing in them has been modified.
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

# ── 1. TOOLCHAIN IDENTITY — the single site for every module/env name ─────────
# Every one of these is env-overridable, so a toolchain drift discovered at login
# is fixed with ONE `export`, not by editing seven files under time pressure.
#
#   export JULIA_MODULE=Julia/1.10.4-foss-2022b     # if 1.11.4 is gone
#   export CONDA_ENV=dep_comp_blp                   # if costsolve was never created
#
# JULIA_MODULE is the name handed to `module load`. cluster_preflight.sh prints
# what `module avail Julia` actually offers when this one does not load.
JULIA_MODULE="${JULIA_MODULE:-Julia/1.11.4-linux-x86_64}"
# PY_MODULE / CONDA_ENV keep the `${VAR+set}` idiom: UNSET means "apply the
# default", explicitly EMPTY means "opt out" (no module load / no conda activate).
[[ -z "${PY_MODULE+set}" ]] && PY_MODULE=miniconda
# CONDA_ENV reconciliation: costsolve is the name every human-facing remediation
# string in this repo tells the reader to create, and it matches the one-time
# setup line below, so it is the default here. If the cluster only has
# dep_comp_blp, cluster_preflight.sh detects that and prints the one-line unblock.
#   module load miniconda && conda create -y -n costsolve python=3.11 \
#          numpy pandas scipy pyarrow statsmodels
[[ -z "${CONDA_ENV+set}" ]] && CONDA_ENV=costsolve
CF_PYTHON="${CF_PYTHON:-python3}"
# What the Python steps actually import. estimation_bbl_3_solve.py needs
# numpy/pandas/scipy (+pyarrow via pd.read_parquet) and NOT statsmodels — that is
# a Step-1 polfunc dependency only, so requiring it always would reject an
# otherwise-usable env.
CL_PY_REQ_SOLVE="numpy, pandas, scipy, pyarrow"
CL_PY_REQ_POLFUNC="numpy, pandas, scipy, pyarrow, statsmodels"

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

# cl_manifest_julia_version: the julia_version line 3 of Manifest.toml records.
cl_manifest_julia_version () {
    local mf="${1:-${CL_ROOT}/Manifest.toml}"
    [[ -f "${mf}" ]] || { echo ""; return 1; }
    sed -n 's/^[[:space:]]*julia_version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "${mf}" | head -1
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
    local mv lv
    mv="$(cl_manifest_julia_version || true)"
    lv="$(cl_loaded_julia_version || true)"
    if [[ -z "${lv}" ]]; then echo "  julia not on PATH — load the module first"; return 2; fi
    if [[ -z "${mv}" ]]; then echo "  Manifest.toml absent or has no julia_version — a resolve will create it"; return 1; fi
    if [[ "$(_cl_majmin "${mv}")" == "$(_cl_majmin "${lv}")" ]]; then
        echo "  Manifest julia_version=${mv} matches loaded julia ${lv}"
        return 0
    fi
    echo "  Manifest julia_version=${mv} does NOT match loaded julia ${lv}"
    return 1
}

# ── 3. SYSIMAGE POLICY (a hard check, not a comment) ─────────────────────────
# A sysimage is BOTH Julia-version-specific AND CPU-target-specific. Built on
# gpu_h200 (sapphirerapids) it is REJECTED on `day` nodes, and when the load
# fails CUDA.functional() goes false and foundation_demand_eval.jl (~line 77)
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

cl_manifest_sha () {
    [[ -f "${CL_ROOT}/Manifest.toml" ]] || { echo "none"; return 0; }
    sha256sum "${CL_ROOT}/Manifest.toml" 2>/dev/null | cut -d' ' -f1
}

# cl_sysimage_verdict TARGET -> prints a verdict, returns 0 OK / 1 refuse.
# Login-node safe: runs signals (a) (b) (c-version/sha) only. The node-local
# cpu_target check needs the node, so it lives in cl_require_sysimage.
cl_sysimage_verdict () {
    local tgt="$1" img side rc=0 sv ss lv
    img="$(cl_sysimage_for "${tgt}")" || return 2
    side="${img}.json"
    if [[ ! -f "${img}" ]]; then
        echo "  ${tgt}: MISSING $(basename "${img}")"
        echo "     REMEDIATION: $(cl_sysimage_build_cmd "${tgt}")"
        return 1
    fi
    if [[ -f "${CL_ROOT}/Manifest.toml" && "${CL_ROOT}/Manifest.toml" -nt "${img}" ]] \
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
cl_gpu_gate () {
    local t="${CL_GPU_GATE_TIMEOUT:-300}"
    echo "── CUDA gate (fatal, ${t}s budget) ──"
    if timeout "${t}" julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} -e '
        using CUDA
        CUDA.functional() || (println(stderr, "CUDA.functional() == false"); exit(3))
        println("CUDA OK: ", CUDA.name(CUDA.device()), "  driver ", CUDA.driver_version())
    '; then
        return 0
    fi
    echo "" >&2
    echo "REFUSING TO RUN: this job holds a GPU but CUDA is not functional." >&2
    echo "  foundation_demand_eval.jl would take the CPU share path and the H200 would sit" >&2
    echo "  at ~0% util for the whole wall. Almost always a sysimage/CPU-target mismatch:" >&2
    echo "      $(cl_sysimage_build_cmd gpu)" >&2
    echo "  ALLOW_SYSIMAGE_FALLBACK=1 continues on CPU anyway." >&2
    [[ "${ALLOW_SYSIMAGE_FALLBACK:-0}" == "1" ]] && { echo "  (set — continuing on CPU)" >&2; return 0; }
    exit 1
}

# ── 4. RC-result staging: pick_zip + CP_DIR, ONE copy, Julia-consistent order ─
# Candidate order matches _result_path() in foundation_demand_eval.jl:157-159:
#   out/cluster_processed, out/BLP_RESULTS/cluster_processed, out/BLP_RESULTS
# and the test is on the LEAF FILE, not on directory existence — Julia takes the
# first candidate whose file exists, so testing dirs can preflight one file while
# Julia reads another.
cl_pick_zip () {
    if [[ -n "${BLP_ZIP:-}" ]]; then printf '%s\n' "${BLP_ZIP}"; return; fi
    local best="" bestid=-1 f id
    for f in "${CL_DATA_OUT}"/blp_outputs_*.zip "${CL_DATA_OUT}"/cluster_raw/blp_outputs_*.zip; do
        [[ -f "$f" ]] || continue
        id="$(basename "$f" | sed -E 's/^blp_outputs_([0-9]+)\.zip$/\1/')"
        if [[ "$id" =~ ^[0-9]+$ ]] && (( id > bestid )); then bestid=$id; best="$f"; fi
    done
    [[ -n "$best" ]] || best="$(ls -t "${CL_DATA_OUT}"/blp_outputs*.zip "${CL_DATA_OUT}"/cluster_raw/blp_outputs*.zip 2>/dev/null | head -1 || true)"
    printf '%s\n' "${best}"
}

cl_cp_dir () {   # cl_cp_dir <routine...>  -> the dir the CF/BBL stack will read
    if [[ -n "${CP_DIR:-}" ]]; then printf '%s\n' "${CP_DIR}"; return; fi
    local c k
    for c in "${CL_DATA_OUT}/cluster_processed" \
             "${CL_DATA_OUT}/BLP_RESULTS/cluster_processed" \
             "${CL_DATA_OUT}/BLP_RESULTS"; do
        for k in "$@"; do
            [[ -f "${c}/blp_E${k}_spec_12.jls" ]] && { printf '%s\n' "${c}"; return; }
        done
    done
    printf '%s\n' "${CL_DATA_OUT}/cluster_processed"    # build target when none exists yet
}

# cl_build_cp_dir: extract the extended RC .jls out of the newest blp_outputs zip.
# Minimal and robust: unzip + cp, no python. The RC zips are FLAT, so -j plus the
# member name lands each .jls in cluster_raw/; a find fallback covers a
# dir-prefixed member.
cl_build_cp_dir () {   # cl_build_cp_dir <cp_dir> <stage> <routine...>
    local cp="$1" stage="$2"; shift 2
    local zip craw k src found
    zip="$(cl_pick_zip)"; craw="${CL_DATA_OUT}/cluster_raw"
    if [[ -z "${zip}" || ! -f "${zip}" ]]; then
        echo "── no blp_outputs_*.zip under ${CL_DATA_OUT} — cannot auto-build cluster_processed/ (has the RC run finished?)"
        return 0
    fi
    echo "── building $(basename "${cp}")/ from $(basename "${zip}") (routines $*) ──"
    mkdir -p "${cp}" "${craw}"
    for k in "$@"; do
        src="blp_results_E${k}_spec_12_${stage}.jls"
        unzip -o -j "${zip}" "${src}" "*/${src}" -d "${craw}" >/dev/null 2>&1 || true
        found="${craw}/${src}"
        [[ -f "${found}" ]] || found="$(find "${craw}" -name "${src}" 2>/dev/null | head -1)"
        if [[ -n "${found}" && -f "${found}" ]]; then
            cp -f "${found}" "${cp}/blp_E${k}_spec_12.jls"; echo "     E${k} ok  (${src})"
        else
            echo "     E${k} --  ${src} not found in $(basename "${zip}")"
        fi
    done
}

# ── 5. ROUTINE DEFAULTS — one site, replacing six that disagree ──────────────
# The live lineup is E1,E2,E3,E4. The RC/CF default set is 3,4, matching
# blp_2_rc.jl:112 DEFAULT_ROUTINES=[3,4] and cluster/upload_manifest.txt's
# '#!ROUTINES 1 2 3 4'. The old E5-E8 numbering is dead: E5/E6 became E3/E4 and
# the joint sieve E7/E8 was deleted.
CL_ROUTINES_ALL="${CL_ROUTINES_ALL:-1 2 3 4}"
CL_ROUTINES_RC="${CL_ROUTINES_RC:-3 4}"
CL_ROUTINES_CF="${CL_ROUTINES_CF:-3 4}"

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

# cl_validate_routines: refuse at t=0 for a routine whose RC result is not on
# disk, so the failure is a 2-second login-node error instead of N chains that
# die at their first stage and take every afterok successor with them. The test
# is the exact file the CF/BBL stack opens, in the exact directory Julia picks.
cl_validate_routines () {   # cl_validate_routines <routine...>
    local k bad=0 cp
    cp="$(cl_cp_dir "$@")"
    for k in "$@"; do
        if [[ ! -f "${cp}/blp_E${k}_spec_12.jls" ]]; then
            echo "  E${k}: no RC result ${cp}/blp_E${k}_spec_12.jls" >&2
            bad=1
        fi
    done
    return ${bad}
}

# ── 6. PREFLIGHT PRIMITIVES — one implementation of each check ───────────────
cl_need_file () {   # cl_need_file <path> <label> [remediation...]
    local p="$1" label="$2"; shift 2
    if [[ -f "${p}" ]]; then return 0; fi
    echo "MISSING ${label}: ${p}"
    [[ $# -gt 0 ]] && printf '   -> %s\n' "$@"
    return 1
}
cl_need_draws () {  # cl_need_draws <R> <SEED>
    cl_need_file "${CL_DATA_OUT}/BLP_DRAWS/halton_nu_R${1}_seed${2}.jls" "R=${1} draws" \
        "build them: bash blp_run.sh --draws   (or sbatch blp_draws_job.sh)"
}
cl_need_rf_curve () {
    cl_need_file "${CL_DATA_IN}/forward_rf_qoq.csv" "forward r^f curve" \
        "build locally then upload: python cf_forward_rf.py --horizon ${HORIZON:-50} --start 2026Q1"
}
cl_need_rc_jls () { # cl_need_rc_jls <cp_dir> <routine>
    cl_need_file "$1/blp_E$2_spec_12.jls" "RC result E$2" \
        "drop the RC blp_outputs_*.zip in ${CL_DATA_OUT} and re-run (auto-built via unzip+cp)," \
        "or pin one with BLP_ZIP=<path>."
}
cl_need_costs () {  # cl_need_costs <routine> <stage>
    cl_need_file "${CL_DATA_OUT}/cost/cost_params_E$1_spec_12_$2.json" "BBL cost params E$1" \
        "run the BBL cost stage first:  bash bbl_run.sh"
}

# cl_python_probe: LOGIN-NODE probe. Runs in a SUBSHELL so `module load` and
# `conda activate` cannot leak into the submitting shell. Returns 0/1.
cl_python_probe () {   # cl_python_probe "<import list>"
    (
      if [[ -n "${PY_MODULE:-}" ]]; then module load ${PY_MODULE} || true; fi
      if [[ -n "${CONDA_ENV:-}" ]]; then
          source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}" || true
      fi
      "${CF_PYTHON}" -c "import $1"
    ) >/dev/null 2>&1
}

# cl_setup_python: IN-JOB activation + verification. Sets PYBIN and CF_COST_FWD.
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
        echo "  e.g.  PY_MODULE=miniconda CONDA_ENV=costsolve bash bbl_run.sh" >&2
        exit 127; }
    "${PYBIN}" -c "import ${req}" 2>/dev/null || {
        echo "ERROR: '${PYBIN}' lacks the required stack (${req})." >&2
        echo "  Use an existing env:  export CONDA_ENV=<name>   (conda env list)" >&2
        echo "  or an interpreter:    export CF_PYTHON=/path/to/python CONDA_ENV=''" >&2
        echo "  or create it:         module load miniconda && conda create -y -n ${CONDA_ENV:-costsolve} python=3.11 numpy pandas scipy pyarrow statsmodels" >&2
        exit 127; }
    # The scripts' default COST_FWD is the local BCB tree, absent on the cluster.
    # Point it at data/output/cost, where the Julia fwd_sim writes the psi_* this
    # solve reads (matches cf_out_dir(out_dir,"COST_FWD")).
    export CF_COST_FWD="${CF_COST_FWD:-${CL_DATA_OUT}/cost}"
    echo "COST_FWD = ${CF_COST_FWD}"
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
# It is EXPORTED so a child orchestrator (pipeline_run.sh -> bbl_run.sh /
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

cl_sbatch () {
    if [[ "${CL_DRYRUN}" == "1" ]]; then
        local n jid
        n=$(( $(cat "${CL_FAKE_COUNTER}" 2>/dev/null || echo 0) + 1 ))
        printf '%s' "${n}" > "${CL_FAKE_COUNTER}"
        jid=$((CL_FAKE_JID_BASE + n))
        { printf '[dry-run] sbatch --parsable --kill-on-invalid-dep=yes'
          _cl_q "$@"
          printf '\n          -> job id %s\n' "${jid}"; } >&2
        printf '%s\n' "${jid}"
        return 0
    fi
    sbatch --parsable --kill-on-invalid-dep=yes "$@"
}

# ── 8. Small shared knobs ────────────────────────────────────────────────────
# Per-routine memory. E1/E2 are the LARGEST panels (E1 = 796,154 obs) and 200G
# OOM-killed them at ext1 (job 21525240): the hot buffer in
# blp_1_estimation.jl:357 scales as n_pi*N*R and rc4 -> ext1 takes n_pi 3 -> 4.
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

cl_banner () {
    echo "=============================================================================="
    printf ' %s\n' "$@"
    echo "=============================================================================="
}
