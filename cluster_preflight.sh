#!/bin/bash
# cluster_preflight.sh — the FIRST thing you run on Bouchet. It SUBMITS NOTHING.
# ==============================================================================
# WHAT IT DOES
#   Prints a numbered verdict block: what Julia the cluster actually offers,
#   whether Manifest.toml matches it (i.e. whether a Pkg.resolve is required),
#   whether each sysimage is usable, whether the conda env exists and imports the
#   stack the BBL solve needs, and which staged inputs are present.
#
# WHAT MUST EXIST FIRST
#   Only the uploaded scripts/ bundle. cluster_lib.sh must sit next to this file.
#
# WHAT TO RUN NEXT
#   Whatever the verdict block tells you: a resolve (env_job.sh ENV_STEP=resolve),
#   a sysimage build, an `export`, or straight on to blp_run.sh / bbl_run.sh /
#   cf_run.sh / pipeline_run.sh.
#
# THESE SCRIPTS ARE NEW. The submit_*.sh / zip_*.sh scripts they replace are still
# present and still work; they are the fallback and are retired only after one
# successful cluster cycle. Nothing in them has been modified.
#
# Usage:  bash cluster_preflight.sh [--routines "3 4"] [--dry-run] [-h]
#         JULIA_MODULE=Julia/1.10.4-foss-2022b bash cluster_preflight.sh
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
        --dry-run)  ;;                       # no-op: this script submits nothing
        -h|--help)  sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see -h)" >&2; exit 2 ;;
    esac
    shift
done

FAIL=0
note_fail () { FAIL=1; }

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
         bbl_job.sh bbl_run.sh pipeline_run.sh cf_job.sh cf_run.sh cf_eq_run.sh \
         cluster_archive.sh sleep_job.sh sleep_run.sh logit_job.sh pipeline_all.sh; do
    if [[ -f "${CL_ROOT}/${f}" ]]; then echo "  ok      ${f}"
    else echo "  MISSING ${f}  <- re-upload the code bundle"; note_fail; fi
done

# ── (1) what Julia does the cluster actually offer? ──────────────────────────
echo ""
echo "(1) JULIA MODULES AVAILABLE  (authoritative — nothing local can answer this)"
cl_julia_candidates

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
    echo "  Every consolidated script reads it from cluster_lib.sh; nothing else"
    echo "  needs editing. (The OLD submit_*.sh scripts hardcode the module name at"
    echo "  7 sites and would each need a hand edit — see cluster/RUNBOOK.md.)"
    echo "  -----------------------------------------------------------------------"
    exit 2
fi

# ── (3) manifest vs loaded julia — does a resolve have to happen? ────────────
echo ""
echo "(3) MANIFEST vs LOADED JULIA"
cl_check_julia_version
RESOLVE_RC=$?
if [[ ${RESOLVE_RC} -eq 0 ]]; then
    echo "RESOLVE REQUIRED: no"
else
    echo "RESOLVE REQUIRED: yes"
    echo ""
    echo "  REMEDIATION -----------------------------------------------------------"
    echo "  Manifest.toml was resolved under a different Julia. Re-resolve it ONCE,"
    echo "  ALONE, through env_job.sh — never concurrently, never inside an array"
    echo "  job (concurrent Pkg.resolve() on NFS corrupts Manifest.toml):"
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

# ── (4) sysimages ────────────────────────────────────────────────────────────
echo ""
echo "(4) SYSIMAGE PROVENANCE  (a hard gate at run time, not a warning)"
for t in gpu cpu; do cl_sysimage_verdict "${t}" || note_fail; done
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
            # Files land FLAT at their destination — stage_cluster_upload.py stores
            # each member as '<dest>/<basename>'. Patterns are bare filenames today,
            # so this is a no-op; it keeps the check right if one ever is not.
            p="${p##*/}"
            case "${bundle}" in
                produced)
                    PROD_SEEN=$((PROD_SEEN+1))
                    if ls "${CL_DATA_OUT}/${p}" >/dev/null 2>&1; then
                        PROD_HAVE=$((PROD_HAVE+1))
                    else
                        echo "  pending ${name//\{k\}/${k}}: data/output/${p}  (produced on cluster — checked, not uploaded)"
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
    cl_need_draws "${R:-2000}" "${SEED:-42}" || note_fail
    cl_need_rf_curve || note_fail
    cl_need_file "${CL_DATA_IN}/polfunc_fitted.csv" "fitted policy" \
        "build locally then upload: python estimation_bbl_1_polfunc.py" || note_fail
fi
CPD="$(cl_cp_dir ${ROUTINES})"
echo "  RC results dir (Julia's first-existing candidate): ${CPD}"
for k in ${ROUTINES}; do
    [[ -f "${CPD}/blp_E${k}_spec_12.jls" ]] && echo "    E${k} ok" \
        || echo "    E${k} --  (bbl_run.sh/cf_run.sh auto-build it from the newest blp_outputs_*.zip)"
done

# ── (8) sigma-directory layout note for cf_eq_run.sh ─────────────────────────
FLAT="$(ls -d "${CL_DATA_OUT}"/cf/cf5_base_E* "${CL_DATA_OUT}"/cf/cf5_shock_E* 2>/dev/null || true)"
if [[ -n "${FLAT}" ]]; then
    echo ""
    echo "(8) NOTE: flat cf5_base_/cf5_shock_ sigma directories exist:"
    printf '    %s\n' ${FLAT}
    echo "  cf_eq_run.sh writes the NESTED layout (cf5_E{k}_{stage}/{base,shock}), which is"
    echo "  what the cf5 archive pattern matches. Archive or move these with the old"
    echo "  zip_cf_outputs.sh before the new driver runs. This preflight does not move them."
fi

echo ""
if [[ ${FAIL} -eq 0 ]]; then
    cl_banner "PREFLIGHT OK — nothing blocking." \
              "next:  bash blp_run.sh --dry-run   then   bash blp_run.sh" \
              "       bash pipeline_run.sh --dry-run   then   bash pipeline_run.sh"
    exit 0
else
    cl_banner "PREFLIGHT FOUND BLOCKERS — act on the REMEDIATION blocks above." \
              "Re-run this script until it prints PREFLIGHT OK."
    exit 1
fi
