#!/bin/bash
# cluster_preflight.sh — the FIRST thing you run on Bouchet. It SUBMITS NOTHING.
# ==============================================================================
# WHAT IT DOES
#   Prints a numbered verdict block: what Julia the cluster actually offers,
#   whether Manifest.toml matches it (i.e. whether a Pkg.resolve is required),
#   whether each sysimage is usable, whether the conda env exists and imports the
#   stack the BBL solve needs, whether the output step tree is writable, and which
#   staged inputs are present.
#
# WHERE IT NORMALLY RUNS: inside gate G0, i.e.
#     sbatch --export=ALL,ENV_STEP=preflight env_job.sh
#   Nothing here belongs on the login node — it loads a Julia module, LOADS both
#   sysimages and imports the Python stack — and pipeline_all.sh submits that job
#   for you. Running it by hand on the login node still works and is still useful
#   when you are diagnosing a refusal; it just does more work there than it should.
#
# WHAT MUST EXIST FIRST
#   Only the uploaded scripts/ bundle. cluster_lib.sh must sit next to this file.
#
# WHAT TO RUN NEXT
#   Whatever the verdict block tells you: a resolve (env_job.sh ENV_STEP=resolve),
#   a sysimage build, an `export`, or straight on to pipeline_all.sh.
#
# Usage:  bash cluster_preflight.sh [--routines "3 4"] [--will-build "sysimage draws"]
#                                   [--dry-run] [-h]
#         --will-build names artifacts this run creates: reported, but not blocking.
#         JULIA_MODULE=Julia/1.10.4-foss-2022b bash cluster_preflight.sh
#         CL_VERBOSE=1 adds the verbatim `module avail Julia` dump.
# --dry-run is accepted and is a NO-OP: this script never submits anything.
# ==============================================================================
set -uo pipefail
CL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${CL_DIR}/cluster_lib.sh"

ROUTINES_SRC=default
if [[ -n "${ROUTINES+set}" ]]; then ROUTINES_SRC=env; fi
ROUTINES="${ROUTINES:-${CL_ROUTINES_ALL}}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --routines) ROUTINES="$2"; ROUTINES_SRC=flag; shift ;;
        # Artifacts THIS run creates before anything consumes them. A missing sysimage is a
        # hard gate for a run that assumes one exists, and a routine no-op for a run whose
        # first two jobs build it -- the preflight cannot tell those apart on its own, so the
        # caller says which it is. Still reported either way; only the verdict changes.
        --will-build) WILL_BUILD="$2"; shift ;;
        --dry-run)  ;;                       # no-op: this script submits nothing
        -h|--help)  sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

FAIL=0
WILL_BUILD="${WILL_BUILD:-}"
note_fail () { FAIL=1; }
will_build () { [[ " ${WILL_BUILD} " == *" $1 "* ]]; }
# Anything not named in --will-build still blocks exactly as before.
note_fail_unless_building () {
    if will_build "$1"; then
        echo "    ^ not a blocker: this run builds it (--will-build $1)"
    else
        note_fail
    fi
}

cl_banner "CLUSTER PREFLIGHT — submits nothing" \
          "scripts: ${CL_ROOT}" \
          "data:    ${CL_DATA_ROOT}   (input = uploaded, output = cluster-produced)" \
          "$(cl_routines_provenance "${ROUTINES}" "${ROUTINES_SRC}" "${CL_ROUTINES_ALL}")"

# ── (0) the consolidated bundle itself + the log dir ─────────────────────────
echo ""
echo "(0) SCRIPT BUNDLE"
LOGD="$(cl_log_dir)"
echo "  log dir: ${LOGD}"
for f in cluster_lib.sh env_job.sh blp_draws_job.sh blp_stage_job.sh blp_run.sh \
         bbl_job.sh bbl_run.sh cf_job.sh cf_run.sh cf_eq_run.sh \
         cluster_archive.sh sleep_job.sh sleep_run.sh logit_job.sh pipeline_all.sh; do
    if [[ -f "${CL_ROOT}/${f}" ]]; then echo "  ok      ${f}"
    else echo "  MISSING ${f}  <- re-upload the code bundle"; note_fail; fi
done

# ── (0b) the output step tree ────────────────────────────────────────────────
# data/output is split one folder per pipeline step, and every producer writes to
# exactly one of them. A folder that is absent or read-only is a failure mode with
# no early symptom: Julia's mkpath and Python's os.makedirs both raise deep inside a
# job that has already burned its queue time, and a full-quota project directory
# fails the WRITE rather than the mkdir, hours later still. So test both here, by
# actually creating and removing a probe file. cl_export_step_dirs creates the tree;
# this section only reports on it, which is why a missing folder is a blocker rather
# than something to fix silently — its absence means no job has bootstrapped yet.
echo ""
echo "(0b) OUTPUT STEP TREE  (exists + writable)"
echo "  data/output: ${CL_DATA_OUT}"
for d in "${CL_STEP_SLEEP}" "${CL_STEP_DEMAND}" "${CL_STEP_LOGIT}" "${CL_STEP_BLP}" \
         "${CL_STEP_DRAWS}" "${CL_STEP_BBL}" "${CL_STEP_CF}" "${CL_STEP_DOWNLOAD}"; do
    rel="${d#${CL_DATA_OUT}/}"
    if [[ ! -d "${d}" ]]; then
        echo "  MISSING ${rel}/  <- no job has bootstrapped yet; ENV_STEP=preflight creates it"
        note_fail
    elif ( : > "${d}/.pf_probe" ) 2>/dev/null; then
        rm -f "${d}/.pf_probe"; echo "  ok      ${rel}/"
    else
        echo "  NOT WRITABLE ${rel}/  <- check the project quota and the directory mode"
        note_fail
    fi
done

# ── (1) what Julia does the cluster actually offer? ──────────────────────────
echo ""
echo "(1) JULIA MODULES AVAILABLE  (authoritative — nothing local can answer this)"
# The parsed candidate list IS the verdict; the verbatim `module avail Julia` dump is
# 30+ lines of Lmod formatting that only matters when the parse comes back empty.
# CL_VERBOSE is 1 inside a job (so G0's .out keeps the whole thing) and 0 on the
# login node, which is where this script is read by a human.
if [[ "${CL_VERBOSE}" == "1" ]]; then
    cl_julia_candidates
else
    cl_julia_candidates | sed -n '/--- parsed candidates ---/,$p'
fi

# ── (2) load the configured module ───────────────────────────────────────────
echo ""
echo "(2) LOADING JULIA_MODULE='${JULIA_MODULE}'"
if cl_load_julia; then
    echo "  loaded: $(julia --version 2>/dev/null || echo '??')"
    echo "  depot:  ${JULIA_DEPOT_PATH%%:*}"
else
    echo ""
    echo "  REMEDIATION -----------------------------------------------------------"
    echo "  '${JULIA_MODULE}' did not load. Pick one of the candidates printed in (1)"
    echo "  and retry with it, then keep it for the session:"
    echo "      JULIA_MODULE=<candidate> bash cluster_preflight.sh"
    echo "      export JULIA_MODULE=<candidate>"
    echo "  Every script reads it from cluster_lib.sh — that is the one site, and"
    echo "  nothing else needs editing. Inside a job, pass it through:"
    echo "      sbatch --export=ALL,ENV_STEP=preflight,JULIA_MODULE=<candidate> env_job.sh"
    echo "  -----------------------------------------------------------------------"
    exit 2
fi

# ── (3) manifest vs loaded julia — does a resolve have to happen? ────────────
echo ""
echo "(3) MANIFEST vs LOADED JULIA"
# The path first, then the verdict. Which file Julia reads is decided by
# Base.manifest_names() (a version-pinned twin wins over the plain name), and a verdict
# that does not name it cannot be checked against the tree by whoever reads this block.
echo "  manifest file: $(cl_manifest_path || true)"
cl_check_julia_version
RESOLVE_RC=$?
if [[ ${RESOLVE_RC} -eq 0 ]]; then
    echo "RESOLVE REQUIRED: no"
else
    echo "RESOLVE REQUIRED: yes"
    echo ""
    echo "  REMEDIATION -----------------------------------------------------------"
    echo "  The manifest was resolved under a different Julia — or is not there at all."
    echo "  EVERY code-bundle upload re-introduces this: the bundle carries the manifest"
    echo "  resolved on the LOCAL machine, and unzip -o overwrites whatever the cluster's"
    echo "  own resolve wrote. So this step belongs after every upload, not only the first."
    echo "  Re-resolve ONCE, ALONE, through env_job.sh — never concurrently, never inside"
    echo "  an array job (concurrent Pkg.resolve() on NFS corrupts the manifest):"
    echo ""
    echo "      sbatch --partition=day --time=00:30:00 --cpus-per-task=4 --mem=16G \\"
    echo "             --export=ALL,ENV_STEP=resolve env_job.sh"
    echo ""
    echo "  Wait for it (squeue -u \$USER), submit nothing else meanwhile, then"
    echo "  re-run this preflight. env_job.sh backs the manifest up to"
    echo "  Manifest.toml.bak.<ver> first."
    echo ""
    echo "  IF THE RESOLVE FAILS on a package with no release for this Julia:"
    echo "  do NOT relax anything on the cluster. Re-resolve LOCALLY under a"
    echo "  matching Julia and re-upload Manifest.toml (Project.toml compat is"
    echo "  already permissive: julia = \"1.10, 1.11, 1.12\")."
    echo "  -----------------------------------------------------------------------"
    note_fail
fi

# ── (3b) IS THE DEPOT ACTUALLY POPULATED? ────────────────────────────────────
# A version match is NOT evidence the packages are installed. On 2026-08-20 the manifest
# matched the loaded Julia, this section reported "RESOLVE REQUIRED: no", and the logit job
# still died on `Package Parquet2 ... is required but does not seem to be installed`: the
# instantiate had gone into the DEFAULT depot (~/.julia) while cluster_lib.sh points every
# job at ${CL_ROOT}/.julia_depot. Same Manifest.toml, different library — which is why
# env_job.sh sources cluster_lib.sh before it resolves, and why anything that installs
# packages must do the same. The only decisive test is to load a package the way a job does.
echo ""
echo "(3b) DEPOT CONTENTS  (a version match is not proof the packages exist)"
echo "  depot searched first: ${JULIA_DEPOT_PATH%%:*}"
# Resolvability, NOT a load. `using` on the LOGIN node recompiles against a CPU target the
# compute nodes never precompiled for, and the login node's memory cap kills it mid-precompile
# -- so a perfectly good depot reports as a blocker. Base.find_package resolves a package
# through the active project and depot WITHOUT compiling anything, and that is exactly the
# failure this section exists to catch: a package listed in the Manifest but absent from the
# depot the jobs actually read (which is how the logit phase once died at import).
DEPOT_PROBE="$(JULIA_PKG_PRECOMPILE_AUTO=0 julia --project="${CL_ROOT}" -e '
    miss = String[]
    for p in ("Parquet2", "DataFrames", "JSON3")
        Base.find_package(p) === nothing && push!(miss, p)
    end
    println(isempty(miss) ? "OK" : "FAIL: not installed in this depot: " * join(miss, ", "))
    ' 2>&1 | tail -3)"
if [[ "${DEPOT_PROBE}" == OK* ]]; then
    echo "  ok: Parquet2, DataFrames, JSON3 resolve in this depot"
else
    echo "  INCONCLUSIVE — could not confirm the packages from this node:"
    printf '    %s\n' "${DEPOT_PROBE}"
    echo ""
    echo "  This is ADVISORY, not a blocker. The probe runs on the LOGIN node, which has a"
    echo "  different CPU target and a hard memory cap; a package that is present and fine"
    echo "  for compute jobs can still fail to load or precompile here. Treat a failure as"
    echo "  \"go look\", not as \"the depot is broken\" — the authority is whether the jobs run."
    echo ""
    echo "  If jobs DO die at import (\"Package X is required but does not seem to be"
    echo "  installed\"), instantiate into the depot the jobs actually use. env_job.sh is"
    echo "  the way to do that: it sources cluster_lib.sh and so inherits JULIA_DEPOT_PATH,"
    echo "  which a bare \`julia -e 'Pkg.instantiate()'\` does not — that installs into"
    echo "  ~/.julia, which no job reads first:"
    echo "      sbatch --partition=day --time=02:00:00 --cpus-per-task=8 --mem=32G \\"
    echo "             --export=ALL,ENV_STEP=resolve env_job.sh"
    echo "  Then confirm:  ls .julia_depot/packages/ | head"
fi

# ── (4) sysimages ────────────────────────────────────────────────────────────
echo ""
echo "(4) SYSIMAGE PROVENANCE  (a hard gate at run time, not a warning)"
for t in gpu cpu; do cl_sysimage_verdict "${t}" || note_fail_unless_building sysimage; done
echo "  (an image with no <img>.so.json sidecar is UNKNOWN provenance and is refused;"
echo "   ALLOW_UNSTAMPED_SYSIMAGE=1 accepts it deliberately. On a fresh cluster the"
echo "   right move is to rebuild both — RUNBOOK step 2.)"

# ── (5) Python env for the BBL solve ─────────────────────────────────────────
echo ""
echo "(5) PYTHON ENV FOR THE BBL SOLVE"
echo "  PY_MODULE='${PY_MODULE:-}'  CONDA_ENV='${CONDA_ENV:-}'  CF_PYTHON='${CF_PYTHON}'"
if cl_python_probe "${CL_PY_REQ_SOLVE}"; then
    echo "  ok: '${CONDA_ENV:-<no conda>}' imports ${CL_PY_REQ_SOLVE}"
    cl_python_probe "${CL_PY_REQ_POLFUNC}" \
        && echo "  ok: it also imports statsmodels (needed only for --polfunc)" \
        || echo "  note: no statsmodels — fine unless you run bbl_run.sh --polfunc"
    # The sleepiness stage is the widest importer of the three and it is now the
    # FIRST phase of pipeline_all.sh, so a gap here stops everything, not just one
    # optional flag. Reported, not fatal: bbl_run.sh/cf_run.sh do not need it.
    if cl_python_probe "${CL_PY_REQ_SLEEP}"; then
        echo "  ok: it also imports ${CL_PY_REQ_SLEEP} (the sleepiness stage)"
    else
        echo "  [!] it does NOT import ${CL_PY_REQ_SLEEP}"
        echo "      sleep_run.sh / pipeline_all.sh would fail at their first Python step."
        echo "      dep_comp_blp carries the whole stack — export CONDA_ENV=dep_comp_blp"
    fi
else
    echo "  FAILED: '${CONDA_ENV:-<no conda>}' does not import ${CL_PY_REQ_SOLVE}"
    echo ""
    echo "  conda env list:"
    ( [[ -n "${PY_MODULE:-}" ]] && module load ${PY_MODULE}; conda env list 2>/dev/null ) | sed 's/^/    /' \
        || echo "    (conda not available)"
    ALT=""
    for cand in dep_comp_blp costsolve base; do
        [[ "${cand}" == "${CONDA_ENV:-}" ]] && continue
        if CONDA_ENV="${cand}" cl_python_probe "${CL_PY_REQ_SOLVE}"; then ALT="${cand}"; break; fi
    done
    echo ""
    echo "  REMEDIATION -----------------------------------------------------------"
    if [[ -n "${ALT}" ]]; then
    echo "  '${ALT}' exists and imports the stack cleanly. One-line unblock:"
    echo "      export CONDA_ENV=${ALT}"
    else
    echo "  No existing env imports the stack. Create the documented one:"
    echo "      module load miniconda && conda create -y -n ${CONDA_ENV:-costsolve} python=3.11 \\"
    echo "             numpy pandas scipy pyarrow statsmodels"
    fi
    echo "  Do not skip this. The import failure it catches otherwise surfaces ~14 h"
    echo "  later, when the fwd_sim array drains — and the solve's afterok cascade"
    echo "  then cancels cf1_net with NO LOG AT ALL."
    echo "  -----------------------------------------------------------------------"
    note_fail
fi

# ── (6) non-fatal cluster facts ──────────────────────────────────────────────
echo ""
echo "(6) CLUSTER FACTS  (non-fatal, for sizing the run)"
echo "  partitions:"
sinfo -o "%P" 2>/dev/null | sed 's/^/    /' | head -20 || echo "    (sinfo unavailable)"
MAXJOBS="$(sacctmgr -n -P show qos gpu_h200 format=MaxJobsPU 2>/dev/null | head -1 || true)"
NR="$(echo ${ROUTINES} | wc -w)"
if [[ -n "${MAXJOBS}" ]]; then
    echo "  gpu_h200 QOS MaxJobsPU=${MAXJOBS}; this run wants ${NR} concurrent chains."
    [[ "${MAXJOBS}" =~ ^[0-9]+$ && "${MAXJOBS}" -lt "${NR}" ]] && \
        echo "    -> below the routine count: the chains will SERIALISE (wall-clock = their sum)."
else
    echo "  could not read gpu_h200 MaxJobsPU — verify >=${NR} concurrent GPU jobs are allowed,"
    echo "  else the routine chains serialise."
fi

# ── (7) staged inputs, driven by cluster/upload_manifest.txt ─────────────────
echo ""
echo "(7) STAGED INPUTS"
MANIFEST="${CL_ROOT}/cluster/upload_manifest.txt"
if [[ -f "${MANIFEST}" ]]; then
    echo "  manifest: ${MANIFEST}  (read-only here)"
    # Nine '|'-separated fields; field 1 = name, 2 = bundle, 5 = pattern, 6 = dest.
    #
    # TWO bundle classes reach this loop, and they mean opposite things:
    #   data      UPLOADED. It must already be under data/input, and a missing one
    #             is a blocker: the run cannot start without it.
    #   produced  BUILT BY THIS RUN, into data/output. Checking it is still worth
    #             doing -- it tells you whether a stage can be resumed or skipped --
    #             but a missing one is NOT a blocker. Before the first run of a
    #             fresh cluster every single one is absent, by construction.
    MISS=0; SEEN=0; PROD_SEEN=0; PROD_HAVE=0
    while IFS= read -r line; do
        case "${line}" in \#*|"") continue ;; esac
        echo "${line}" | grep -q '|' || continue
        name="$(echo "${line}"    | awk -F'|' '{gsub(/^ +| +$/,"",$1); print $1}')"
        bundle="$(echo "${line}"  | awk -F'|' '{gsub(/^ +| +$/,"",$2); print $2}')"
        pattern="$(echo "${line}" | awk -F'|' '{gsub(/^ +| +$/,"",$5); print $5}')"
        dest="$(echo "${line}"    | awk -F'|' '{gsub(/^ +| +$/,"",$6); print $6}')"
        case "${bundle}" in data|produced) ;; *) continue ;; esac
        [[ -n "${pattern}" && -n "${dest}" ]] || continue
        for k in ${ROUTINES}; do
            p="${pattern//\{k\}/${k}}"
            case "${p}" in *"{"*) continue ;; esac       # unexpanded token: skip
            # Files land FLAT at their destination — cluster_upload.py stores
            # each member as '<dest>/<basename>'. Patterns are bare filenames today,
            # so this is a no-op; it keeps the check right if one ever is not.
            p="${p##*/}"
            case "${bundle}" in
                produced)
                    PROD_SEEN=$((PROD_SEEN+1))
                    # dest is the data/output SUBFOLDER the producing stage writes to
                    # (data/output/bbl, data/output/logit, ...), relative to the project
                    # root exactly as for uploaded data -- not the data/output top level.
                    full="${CL_ROOT}/../${dest}/${p}"
                    if ls ${full} >/dev/null 2>&1; then
                        PROD_HAVE=$((PROD_HAVE+1))
                    else
                        echo "  pending ${name//\{k\}/${k}}: ${dest}/${p}  (produced on cluster — checked, not uploaded)"
                    fi
                    ;;
                *)
                    SEEN=$((SEEN+1))
                    # dest is relative to the cluster project root (scripts/..).
                    full="${CL_ROOT}/../${dest}/${p}"
                    if ! ls ${full} >/dev/null 2>&1; then
                        echo "  MISSING ${dest}/${p}"; MISS=$((MISS+1))
                    fi
                    ;;
            esac
            case "${pattern}" in *"{k}"*) ;; *) break ;; esac
        done
    done < "${MANIFEST}"
    if [[ ${MISS} -eq 0 ]]; then echo "  ok: all ${SEEN} manifest data entries present for routines ${ROUTINES}"
    else echo "  ${MISS} of ${SEEN} manifest data entries missing (upload them, RUNBOOK step 0)"; note_fail; fi
    if [[ ${PROD_SEEN} -gt 0 ]]; then
        echo "  produced-on-cluster entries: ${PROD_HAVE}/${PROD_SEEN} already in data/output"
        echo "    (never uploaded, never a blocker — pipeline_all.sh builds them and gates G4/G7 check them)"
    fi
else
    echo "  cluster/upload_manifest.txt not found — falling back to the core checks:"
    cl_need_draws "${R:-2000}" "${SEED:-42}" || note_fail_unless_building draws
    # forward_rf_qoq.csv is the one BBL input that can only be built off-cluster
    # (scrape_forward_rf.py talks to the BCB API and compute nodes have no internet), so
    # it blocks. The fitted policy does NOT: bbl_run.sh runs it as a pre-step by
    # default, into data/output/bbl.
    cl_need_rf_curve || note_fail
fi
# The RC results, in the ONE place the RC ladder writes them and the CF/BBL stack
# reads them: data/output/blp. Reported, never a blocker — on a fresh cluster the RC
# phase has not run yet, and bbl_run.sh / cf_run.sh preflight the same file at their
# own submit time, when its absence actually means something.
echo "  RC results dir: ${CL_STEP_BLP}"
for k in ${ROUTINES}; do
    [[ -f "${CL_STEP_BLP}/blp_results_E${k}_spec_12_${CF_STAGE:-extended}.jls" ]] \
        && echo "    E${k} ok" \
        || echo "    E${k} --  (produced by the BLP phase:  bash blp_run.sh --routines ${k})"
done

echo ""
if [[ ${FAIL} -eq 0 ]]; then
    cl_banner "PREFLIGHT OK — nothing blocking." \
              "next:  bash pipeline_all.sh --dry-run   then   bash pipeline_all.sh" \
              "       (pipeline_all.sh submits this preflight itself, as gate G0:" \
              "        sbatch --export=ALL,ENV_STEP=preflight env_job.sh)"
    exit 0
else
    cl_banner "PREFLIGHT FOUND BLOCKERS — act on the REMEDIATION blocks above." \
              "Re-run it until it prints PREFLIGHT OK:" \
              "  sbatch --partition=day --time=00:30:00 --cpus-per-task=2 --mem=8G \\" \
              "         --export=ALL,ENV_STEP=preflight env_job.sh"
    exit 1
fi
