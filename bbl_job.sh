#!/bin/bash
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# (job-name + .out/.err are set per submission by bbl_run.sh via sbatch -J/-o/-e)
# ==============================================================================
# bbl_job.sh — THE generic BBL worker. This is the marginal-cost ESTIMATION
# stage: every artifact of it — polfunc_fitted.csv, the psi_eq/psi_dev shards and
# cost_params_*.json, which the counterfactuals CONSUME — lands in ONE folder,
# data/output/bbl. of_root.jl maps both COST_FWD and COST_POLFUNC there and
# cl_export_step_dirs points the Python halves (COST_POLFUNC_DIR, CF_COST_FWD) at
# the same place, so the fit, the shards and the solve share one directory.
#
# The #SBATCH block is a FLOOR (the panel iterates ~700k demand rows, so it is
# sized above the CF driver). bbl_run.sh overrides --partition/--gpus/--mem/-t
# per step on the sbatch command line.
#
# Driven by env vars:
#   BBL_STEP     warmup | polfunc | fwd_sim | solve | tables | report
#   BBL_ROUTINE  3 (headline) | 4 | ...        -> --estim
#   BBL_STAGE    extended (headline) | full | ...   -> --stage
#   R, SEED      draws (default 2000 / 42) — must match the staged BLP_DRAWS
#   BBL_EXTRA    extra CLI flags passed through
#   PSI_TAG      multi-start artifact tag, e.g. _ms8 (empty = single-start)
#   N_SHARDS     fwd_sim only: total deviation shards (default 1)
#   SHARD_ID     fwd_sim only: this shard (default = SLURM_ARRAY_TASK_ID, else 0)
#   CF_GPU       1 -> GPU path (gpu sysimage); 0 -> CPU path (cpu sysimage)
#
# NOTHING IS DOWNLOADED FROM THIS CLUSTER, which is why the last two steps exist and why
# `solve` ends by cat'ing its own JSON. A result that lives only as a file in data/output is a
# result nobody has: `tables` builds the .tex/.md and echoes them, and `report` re-echoes what
# is already on disk when a job log is lost. Both are CPU-only and cost minutes.
#
# WHAT MUST EXIST FIRST: cluster_preflight.sh green; both sysimages built; the RC
#   results in data/output/blp.
# WHAT TO RUN NEXT: nothing directly — bbl_run.sh owns the chain.
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

: "${BBL_STEP:?set BBL_STEP (warmup|polfunc|fwd_sim|solve|tables|report)}"
case "${BBL_STEP}" in warmup|polfunc|fwd_sim|solve|tables|report) ;;
    *) echo "Unknown BBL_STEP='${BBL_STEP}' (warmup|polfunc|fwd_sim|solve|tables|report)" >&2; exit 2 ;;
esac
BBL_ROUTINE="${BBL_ROUTINE:-3}"
BBL_STAGE="${BBL_STAGE:-extended}"
R="${R:-2000}"; SEED="${SEED:-42}"
BBL_EXTRA="${BBL_EXTRA:-}"
# The multi-start artifact tag, exported by bbl_run.sh. Empty for a single-start run, which is
# also what every filename built from it reduces to, so the solve/tables branches need no
# second code path for the two designs.
PSI_TAG="${PSI_TAG:-}"

mkdir -p "${CL_ROOT}/logs"

# The skeleton, then the step dirs. The polfunc branch below is PYTHON and resolves
# its output through utils/paths.polfunc_dir(), which reads COST_POLFUNC_DIR — set
# here and nowhere else, so a job that skipped this call would write the policy CSV
# into the local-layout default, exit 0, and strand every fwd_sim task behind it.
cl_bootstrap_tree
cl_export_step_dirs

# tables and report never start Julia — they read JSON and format text — so they skip the
# module load and the sysimage gate below. Not an economy: the gate REFUSES on a stale image,
# and these two are the steps that carry the results out of a cluster nothing is downloaded
# from. Losing the collection point to a Julia-toolchain verdict it does not depend on would
# leave a finished run with no readable output at all.
BBL_NEEDS_JULIA=1
case "${BBL_STEP}" in tables|report) BBL_NEEDS_JULIA=0 ;; esac

if [[ "${BBL_NEEDS_JULIA}" == "1" ]]; then
cl_load_julia
fi

# CF_GPU is set by bbl_run.sh (FWD_GPU=1 -> warmup + fwd_sim on the H200;
# solve/polfunc stay CPU). Default OFF here so a bare `BBL_STEP=... sbatch
# bbl_job.sh` stays CPU and concurrent array tasks on the shared NFS depot do not
# stampede the precompile/load lock loading CUDA.
: "${CF_GPU:=0}"; export CF_GPU

# ── Sysimage: the GPU/CPU pairing, as a REFUSAL. ─────────────────────────────
# A sysimage bakes its BUILD node's CPU target, so the GPU (gpu_h200 =
# sapphirerapids) and CPU (`day`, cpugen:turin) runs need DIFFERENT images.
# Loading the wrong one fails with "Unable to find compatible target in cached
# code image" and drops the task back to precompiling against the shared NFS
# depot; a task that loses that race gets CUDA.functional()==false, which
# cf_demand_eval.jl catches and silently downgrades to the CPU share path
# — the job then holds its H200 at 0% util. (Diagnosed 2026-07-31: 5/11 fwd_sim
# tasks fell back to CPU on FillArrays/StaticArrays cache races.)
if [[ "${BBL_NEEDS_JULIA}" == "1" ]]; then
    if [[ "${CF_GPU}" != "0" ]]; then cl_require_sysimage gpu; else cl_require_sysimage cpu; fi

    # Fatal CUDA gate, before the real work, only when the job actually holds a GPU.
    if [[ "${CF_GPU}" != "0" && -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]]; then
        cl_gpu_gate
    fi
fi

cl_banner "BBL cost stage | step=${BBL_STEP} | E${BBL_ROUTINE} | stage=${BBL_STAGE}" \
          "R=${R} seed=${SEED} threads=${SLURM_CPUS_PER_TASK:-8} CF_GPU=${CF_GPU}" \
          "node=$(hostname) | $(date)"

run_julia () {
    local script="$1"; shift
    julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} \
        --threads="${SLURM_CPUS_PER_TASK:-8}" \
        "${CL_ROOT}/${script}" \
        --estim "${BBL_ROUTINE}" --spec 12 --stage "${BBL_STAGE}" \
        --R "${R}" --seed "${SEED}" --hpc ${BBL_EXTRA} "$@"
}

case "${BBL_STEP}" in

    warmup)
        # JIT-load the BBL stack ONCE. With a sysimage the packages are already
        # baked, so the array never precompiles and there is no shared-NFS
        # stampede. Without one (ALLOW_SYSIMAGE_FALLBACK=1), precompile the depot
        # here so the parallel fwd_sim array at least does not all precompile at
        # the same moment.
        if [[ ${#CL_JULIA_SYS[@]} -gt 0 ]]; then
            echo "Sysimage present — skipping depot precompile."
        else
            echo "Precompiling depot..."
            julia --project="${CL_ROOT}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
        fi
        echo "Loading the BBL stack (CF_GPU=${CF_GPU})..."
        julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} \
            --threads="${SLURM_CPUS_PER_TASK:-8}" \
            -e "include(joinpath(\"${CL_ROOT}\", \"bbl_fwd_sim.jl\"))" || true
        echo "warmup complete: BBL stack loaded — the array can launch warm" ;;

    polfunc)
        # BBL Step 1, and the DEFAULT pre-step of every bbl_run.sh chain: fit the
        # parametric policy function. It needs only the market panel (uploaded as
        # market_panel.parquet), which is already there for the sleepiness stage,
        # so running it here costs one
        # short CPU job and removes a 59 MB upload that could go stale against the
        # panel. Writes ${COST_POLFUNC_DIR}/polfunc_fitted.csv — data/output/bbl —
        # which the fwd_sim tasks take through --policy-csv. The policy function is
        # SPEC-INVARIANT, so it takes no --spec (unlike fwd_sim/solve, whose
        # --spec 12 selects the BLP demand specification).
        cl_setup_python "${CL_PY_REQ_POLFUNC}"
        "${PYBIN}" "${CL_ROOT}/bbl_polfunc.py" ${BBL_EXTRA} ;;

    fwd_sim)
        # BBL Step 2 part 1: psi under sigma-hat and sigma-tilde deviations
        # (shardable). The forward r^f curve is a REQUIRED input — the sim errors
        # under --hpc if it is absent, and a flat r^f makes psi4 proportional to
        # psi2 and leaves zeta unidentified. It is fetched from the BCB APIs
        # locally (compute nodes have no internet) and uploaded to data/input.
        RF_CURVE="${RF_CURVE:-${CL_DATA_IN}/forward_rf_qoq.csv}"
        if [[ ! -f "${RF_CURVE}" ]]; then
            echo "ERROR: forward r^f curve missing: ${RF_CURVE}" >&2
            echo "  Generate locally (needs internet) and upload to data/input/:" >&2
            echo "    python scrape_forward_rf.py --horizon 50 --start 2026Q1" >&2
            echo "  Refusing to run: a flat r^f leaves zeta unidentified." >&2
            exit 1
        fi
        echo "forward r^f curve OK: ${RF_CURVE}"
        # The multi-start inputs, refused in the same style and for the same reason. The two are
        # gated on the FLAGS the driver actually passed rather than on a variable of this
        # script's own: bbl_run.sh puts --multi-start into BBL_EXTRA, so reading it back is what
        # makes the refusal describe the run being submitted instead of a second copy of the
        # decision that can disagree with it.
        #
        # Silence is the failure mode being prevented. Without the vintages there is ONE forward
        # curve for every launch quarter, which is precisely the design whose ridge leaves
        # omega and zeta unidentified — the sim would run to completion, write a full set of psi
        # shards, and hand the solve a design that cannot answer the question the multi-start run
        # was submitted to answer. Without the transitions there are no rate-path shocks, so the
        # S paths collapse onto the Focus mean and --n-paths becomes an S-fold repeat of one path.
        if [[ "${BBL_EXTRA}" == *"--multi-start"* ]]; then
            RF_VINT="${RF_VINTAGES:-${CL_DATA_IN}/forward_rf_vintages.csv}"
            ms_miss=0
            if [[ ! -f "${RF_VINT}" ]]; then
                echo "ERROR: forward r^f curve VINTAGES missing: ${RF_VINT}" >&2
                echo "  Generate locally (needs internet) and upload to data/input/:" >&2
                echo "    python scrape_forward_rf.py --vintage-from 2016Q1 --vintage-to 2024Q4 --horizon 50" >&2
                echo "  Refusing to run: with one shared curve every launch quarter gets the same" >&2
                echo "  r^f, which is the flat-ridge design multi-start exists to break." >&2
                ms_miss=1
            fi
            [[ "${ms_miss}" == "0" ]] || exit 1
            echo "multi-start inputs OK: ${RF_VINT}"
        fi
        # bbl_transitions.json is checked OUTSIDE the multi-start branch, because psi_under
        # calls assert_state_evolution on EVERY path and evolving states default ON
        # (cf_demand_eval.jl:133). Gated on --multi-start, a plain single-start array passed
        # this check and then died task by task inside Julia instead. CF_EVOLVING_STATES=0 is
        # the deliberate frozen-state opt-out; honour it here so the shell and Julia agree.
        # Multi-start OR evolving states -- see bbl_run.sh. Gating on the state switch alone
        # would let a multi-start run past this check with no rate.process to build paths from.
        if [[ "${BBL_EXTRA}" == *"--multi-start"* || "${CF_EVOLVING_STATES:-1}" != "0" ]]; then
            TRANS="${BBL_TRANSITIONS:-${CL_DATA_IN}/bbl_transitions.json}"
            if [[ ! -f "${TRANS}" ]]; then
                echo "ERROR: BBL transition parameters missing: ${TRANS}" >&2
                echo "  Generate locally and upload to data/input/:" >&2
                echo "    python bbl_transitions.py" >&2
                echo "  Refusing to run: psi_under asserts evolving states on every path, so" >&2
                echo "  without this file the sim errors per task rather than freezing states." >&2
                echo "  Deliberate frozen-state run: export CF_EVOLVING_STATES=0" >&2
                exit 1
            fi
            echo "transitions OK: ${TRANS}"
        else
            echo "CF_EVOLVING_STATES=0 -- frozen-state run, bbl_transitions.json not required"
        fi
        SHARD_ID="${SLURM_ARRAY_TASK_ID:-${SHARD_ID:-0}}"
        N_SHARDS="${N_SHARDS:-1}"
        run_julia bbl_fwd_sim.jl --n-shards "${N_SHARDS}" --shard-id "${SHARD_ID}" ;;

    solve)
        # BBL Step 2 part 2: the eq:17 minimisation -> data/output/bbl/cost_params_*.json.
        cl_setup_python "${CL_PY_REQ_SOLVE}"
        "${PYBIN}" "${CL_ROOT}/bbl_solve.py" \
            --estim "${BBL_ROUTINE}" --spec 12 --stage "${BBL_STAGE}" ${BBL_EXTRA}
        # The JSON, verbatim, into this job's log. Nothing is downloaded from this cluster, so a
        # file in data/output is not a result anyone can read: the log is where the numbers are
        # collected, and the solve's own stdout prints only the summary lines. Small (a few kB),
        # and echoing it here means a later question about ci_omega or by_start is answered from
        # the log instead of from a transfer that will not happen.
        cp_out="${CF_COST_FWD}/cost_params_E${BBL_ROUTINE}_spec_12_${BBL_STAGE}${PSI_TAG:-}.json"
        if [[ -f "${cp_out}" ]]; then
            echo "── cost_params (verbatim): $(basename "${cp_out}") ──"
            cat "${cp_out}"
            echo ""
            echo "── end cost_params ──"
        else
            echo "[!] solve finished but ${cp_out} is not on disk — check the traceback above." >&2
        fi ;;

    tables)
        # BBL Step 3: turn the solves into the paper tables ON THE CLUSTER.
        #
        # This step exists because nothing comes back from Bouchet. Locally this generator is a
        # convenience over downloaded artifacts; here it is the only thing that converts
        # cost_params and the psi shards into a readable result, and it echoes every table to
        # stdout so the SLURM log IS the deliverable.
        #
        # --psi-dir, not the psi_cost.zip default: the shards are read where the array wrote
        # them (CF_COST_FWD == the bbl step folder), which is before cluster_archive.sh has
        # packaged anything, so the tables do not have to wait on the archive job.
        # --cost-dir the same folder, since the solve writes cost_params there too.
        # Table files land in the step folder (and in Drafts/ when that exists, which on a
        # compute node it does not — the generator's _dests() decides, not this script).
        cl_setup_python "${CL_PY_REQ_SOLVE}"
        # A vintage mismatch is not a reason to produce nothing here: the from-psi pass is the
        # ridge half, and the cost half comes from cost_params either way, so the two passes are
        # run separately and the second is allowed to fail without taking the step down.
        # --psi-tag is passed even when EMPTY. make_bbl_cost_tables.py reads "" as "the untagged
        # single-start vintage only" but an ABSENT flag as "either vintage, preferring the largest
        # multi-start run" -- so dropping it for a single-start run reported a leftover
        # multi-start run's numbers under a single-start job.
        "${PYBIN}" "${CL_ROOT}/make_bbl_cost_tables.py" \
            --cost-dir "${CF_COST_FWD}" --psi-tag "${PSI_TAG}"
        "${PYBIN}" "${CL_ROOT}/make_bbl_cost_tables.py" --from-psi \
            --psi-dir "${CF_COST_FWD}" --cost-dir "${CF_COST_FWD}" \
            --psi-tag "${PSI_TAG}" \
            || echo "[!] the --from-psi (ridge) pass failed; the cost tables above still stand." ;;

    report)
        # BBL Step 4, and a recovery tool rather than part of the chain: echo whatever the step
        # folder currently holds into a FRESH log. With no downloads, a lost or truncated job log
        # is a lost result, and re-running the solve to read a number again costs an array.
        # Reads nothing but the step folder and computes nothing.
        echo "── step folder: ${CF_COST_FWD} ──"
        ls -la "${CF_COST_FWD}" || true
        for f in "${CF_COST_FWD}"/cost_params_*.json; do
            [[ -f "${f}" ]] || continue
            echo ""
            echo "── $(basename "${f}") ──"
            cat "${f}"
        done
        for f in "${CF_COST_FWD}"/tab_bbl_*.md; do
            [[ -f "${f}" ]] || continue
            echo ""
            echo "── $(basename "${f}") ──"
            cat "${f}"
        done
        echo ""
        echo "── end report ──" ;;

esac

echo "BBL step ${BBL_STEP} (E${BBL_ROUTINE} ${BBL_STAGE}) complete: $(date)"
