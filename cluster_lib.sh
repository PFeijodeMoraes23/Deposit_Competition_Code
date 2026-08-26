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
# What the Python steps actually import. estimation_bbl_3_solve.py needs
# numpy/pandas/scipy (+pyarrow via pd.read_parquet) and NOT statsmodels — that is
# a Step-1 polfunc dependency only, so requiring it always would reject an
# otherwise-usable env.
CL_PY_REQ_SOLVE="numpy, pandas, scipy, pyarrow"
CL_PY_REQ_POLFUNC="numpy, pandas, scipy, pyarrow, statsmodels"
# The sleepiness stage (sleep_job.sh, every branch) imports the widest stack of the
# three: statsmodels for the first stages and matplotlib because the export step
# (export_results.py / export_analyze_spec12.py, run through
# run_sleep_pipeline.py --skip-sleep) draws figures. MPLBACKEND=Agg keeps that
# headless; it does not make the import optional. dep_comp_blp already carries all
# six — this string is the preflight PROBE, not a to-do list.
CL_PY_REQ_SLEEP="numpy, pandas, scipy, pyarrow, statsmodels, matplotlib"

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
cl_need_rf_curve () {
    cl_need_file "${CL_DATA_IN}/forward_rf_qoq.csv" "forward r^f curve" \
        "build locally then upload: python cf_forward_rf.py --horizon ${HORIZON:-50} --start 2026Q1"
}
# cl_need_rc_jls <routine>: the RC result the CF/BBL stack opens. It tests the exact file
# blp_dir(out_dir) names in foundation_demand_eval.jl's _result_path, in the exact place
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

# cl_need_polfunc: the fitted policy function estimation_bbl_2_fwd_sim.jl reads through
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
# upsilon_pix_E{k} and phi_nopix_E{k} are written by cf_4_upsilon_export.py running on the
# cluster (sleep_job.sh SLEEP_STEP=upsilon, gated by G7) into CF_FOUNDATION_DIR, which
# cl_export_step_dirs points at the counterfactuals step folder. Nothing else writes them
# and nothing copies them afterwards, so there is exactly one place to look.
#
# The single candidate is the point, not an economy: a second candidate is how a stale
# uploaded pair and a fresh cluster-produced pair coexist and different consumers resolve
# different vintages of the same routine — the collision that aborts cf4 three phases
# later, after the RC ladders and the BBL solves. cf4_search_dirs() in cf_4_pix.jl and
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
    local a
    for a in "$@"; do
        case "${a}" in
            --constraint*|-C) return 0 ;;
            --gpus*|--gres=gpu*|--partition=gpu*|--partition=scavenge_gpu*) return 0 ;;
        esac
    done
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
    sbatch --parsable --kill-on-invalid-dep=yes ${_cg:+"${_cg}"} "$@"
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
