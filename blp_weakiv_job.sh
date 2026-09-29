#!/bin/bash
#SBATCH --job-name=blp_weakiv
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# ==============================================================================
# blp_weakiv_job.sh — the alpha weak-instruments battery, on the cluster.
#
# WHY IT IS A JOB. The battery inverts AR/LM over an alpha grid, runs the
# OLS/2SLS/LIML/Fuller ladder per subsample and bootstraps eff-F. That is
# computation, and computation belongs here: locally we render tables and
# figures, nothing else. Every number the alpha tables print comes from the
# JSONs this job writes.
#
# WHAT IT RUNS, in order (each step's output is the next one's input):
#   1. blp_delta_export.jl --stage ext1
#        cluster_processed/rc_delta_E{k}_spec_12_ext1.bin — the STRUCTURAL delta of
#        the RC ext1 stage. Without it step 3 has nothing to invert against, which
#        is why weak_iv_ext1.json has been stale since 2026-08-17: this export has
#        never run on the cluster.
#   2. blp_weak_iv.py
#        cluster_processed/weak_iv.json — the battery against the LOGIT delta
#        ln s − ln s0, read from the logit step's logit/logit_delta_E{k}_spec_12.bin.
#   3. blp_weak_iv.py --delta-stage ext1
#        cluster_processed/weak_iv_ext1.json — the same battery against the
#        structural delta from step 1.
#   4. blp_moment_reduction.py
#        cluster_processed/diag_moment_reduction.json — the moment-reduction fixes
#        the alpha tables read for their weight-matrix columns.
#
# WHAT THE BATTERY IS TESTING, and why it reports by SUBSAMPLE. The demand model is
# not a standard 2SLS: only the k=4,5 spread is projected, on H = [X, Z], with x_mat
# its own instrument (blp_engine_cpu.jl:1333-1337, 834-853). Rows for k=1,2 enter
# with the spread EXOGENOUS — no instrument at all. blp_weak_iv.py mirrors that
# (ENDOG_TYPES = [4, 5]), so the instrument question exists only for types 4/5 and
# the cells are reported per subsample: type4, type5, type45, type12, all. Pooling
# the instrumented types with the uninstrumented ones is what makes alpha's sign
# flip between subsamples, so the split is the finding, not a robustness afterthought.
#
# ROUTINES. config/routines.toml `link_ests` (E3/E4), matching every other reported
# table. Override with ROUTINES="1 2 3 4" to widen it.
#
# DEPENDENCIES. Reads data/output/blp (the RC results and their .jls) and the demand
# parquets. Inside pipeline_all.sh it chains afterok the RC terminals; standalone it
# can run any time the RC results are on disk.
# ==============================================================================
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

SPEC="${SPEC:-12}"
STAGE="${WIV_STAGE:-ext1}"
ROUTINES="${ROUTINES:-${CL_ROUTINES_RC}}"
# Comma form for the Python/Julia --routines flags; the shell carries it space-separated.
ROUTINES_CSV="$(echo "${ROUTINES}" | tr -s ' ' ',' | sed 's/^,//; s/,$//')"

mkdir -p "${CL_ROOT}/logs"
cl_bootstrap_tree
cl_export_step_dirs

cl_banner "Alpha weak-IV battery | spec ${SPEC} | stage ${STAGE}" \
          "routines '${ROUTINES}' (csv '${ROUTINES_CSV}')" \
          "writes ESTIMATION_OUTPUT/BLP_RESULTS/cluster_processed (resolved below)" \
          "node=$(hostname) | $(date)"

# Python first: step 0 below resolves BLP_RESULTS by ASKING utils.paths rather than hardcoding
# a second spelling of it, and the interpreter has to exist before that question can be asked.
# cl_setup_python is idempotent, so the later call in step 2-4 is left where it documents itself.
cl_setup_python "${CL_PY_REQ_POLFUNC}"
# Every python call below is "${PYBIN}", the interpreter cl_setup_python just VALIDATED --
# never a bare `python`. With the documented opt-out (CF_PYTHON=/path/to/python CONDA_ENV='')
# there may be no `python` on PATH at all, and a bare call would exit 127 under set -e.

# ---- 0. put the RC artifacts where the readers look --------------------------------------
# blp_delta_export.jl:get_paths() and blp_moment_reduction.py (via blp_cue_linear._dirs) both
# read the RC results out of ESTIMATION_OUTPUT/BLP_RESULTS/cluster_raw. On the cluster that
# folder is populated by the LOCAL ingest (cluster_ingest_blp.py), never by a cluster run: the
# RC ladder writes to ${CL_STEP_BLP} (data/output/blp) through of_root.jl's blp_dir(). So on
# Bouchet cluster_raw is empty and the export would report "MISSING ... skipped" for every
# routine, write no rc_delta, and leave weak_iv_ext1.json unwritten.
#
# Symlinks, not copies: the .jls are large and both live under data/output, so nothing leaves
# the archiver's domain root. Linking rather than teaching two analysis scripts a second input
# convention keeps this the only place that knows about the split.
RESROOT="$("${PYBIN}" -c 'import sys, os; sys.path.insert(0, os.environ["CL_ROOT"]); import utils.paths as p; print(p.blp_results_dir())')"
RAW="${RESROOT}/cluster_raw"
mkdir -p "${RAW}"
n_link=0
for k in ${ROUTINES}; do
    for ext in jls json; do
        src="${CL_STEP_BLP}/blp_results_E${k}_spec_12_${STAGE}.${ext}"
        if [[ -f "${src}" ]]; then
            ln -sfn "${src}" "${RAW}/$(basename "${src}")"; n_link=$((n_link + 1))
        else
            echo "[!] E${k}: ${src} absent — the ${STAGE} rung of the RC ladder has not run." >&2
        fi
    done
done
echo "linked ${n_link} RC artifact(s) into ${RAW}"
if [[ ${n_link} -eq 0 ]]; then
    echo "ERROR: no RC results for stage '${STAGE}' in ${CL_STEP_BLP}." >&2
    echo "  The alpha battery inverts against the STRUCTURAL delta of that rung, so there is" >&2
    echo "  nothing to compute. Run the RC phase first (bash blp_run.sh), or set WIV_STAGE to" >&2
    echo "  a rung that exists:  ls ${CL_STEP_BLP}/blp_results_E*_spec_12_*.jls" >&2
    exit 1
fi

# ---- 1. structural delta -----------------------------------------------------------------
# Julia only for this step. No sysimage: cl_require_sysimage would refuse the job over an
# image this export has no use for, and the cost is one cold JIT of the serialisation stack.
cl_load_julia
julia --project="${CL_ROOT}" --threads="${SLURM_CPUS_PER_TASK:-8}" \
    "${CL_ROOT}/blp_delta_export.jl" --stage "${STAGE}" --routines "${ROUTINES_CSV}"

# ---- 2-4. the battery and the moment-reduction diagnostic --------------------------------
# statsmodels and scipy are both needed: the ladder uses linearmodels' IV estimators where
# they converge and falls back to the numpy path where they do not.
cl_setup_python "${CL_PY_REQ_POLFUNC}"

echo "-- weak-IV battery: logit delta --"
"${PYBIN}" "${CL_ROOT}/blp_weak_iv.py" --routines "${ROUTINES_CSV}"

echo "-- weak-IV battery: structural delta (--delta-stage ${STAGE}) --"
"${PYBIN}" "${CL_ROOT}/blp_weak_iv.py" --routines "${ROUTINES_CSV}" --delta-stage "${STAGE}"

echo "-- moment-reduction diagnostic --"
"${PYBIN}" "${CL_ROOT}/blp_moment_reduction.py" --routines "${ROUTINES_CSV}" --stage "${STAGE}"

# ---- what landed -------------------------------------------------------------------------
# The destination is ASKED FOR, not assumed. All four writers resolve it through
# utils.paths.blp_results_dir(), which is ESTIMATION_OUTPUT/BLP_RESULTS -- NOT ${CL_STEP_BLP}
# (data/output/blp), where the RC results themselves live. Hardcoding either spelling here
# would mean a verification block that passes while the files land somewhere else, so the
# same accessor the writers use is queried for the answer.
CP="$("${PYBIN}" -c 'import sys, os; sys.path.insert(0, os.environ["CL_ROOT"]); \
import utils.paths as p; print(p.blp_results_dir() / "cluster_processed")')"
echo "resolved output dir: ${CP}"

# Named individually rather than globbed: a missing file has to be visible in the log, not
# absorbed by a wildcard that still matches the other three.
echo
echo "Outputs in ${CP}:"
for f in weak_iv.json "weak_iv_${STAGE}.json" diag_moment_reduction.json; do
    if [[ -s "${CP}/${f}" ]]; then
        echo "  ok      ${f}  ($(stat -c%s "${CP}/${f}" 2>/dev/null || echo '?') bytes)"
    else
        echo "  MISSING ${f}" >&2
        exit 1
    fi
done
echo "Alpha weak-IV battery COMPLETE."
