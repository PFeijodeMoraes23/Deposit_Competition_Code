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
#   BBL_STEP     warmup | polfunc | fwd_sim | sweep | solve | tables | report
#   BBL_ROUTINE  3 (headline) | 4 | ...        -> --estim
#   BBL_STAGE    extended (headline) | full | ...   -> --stage
#   R, SEED      draws (default 2000 / 42) — must match the staged BLP_DRAWS
#   BBL_EXTRA    extra CLI flags passed through
#   PSI_TAG      multi-start artifact tag, e.g. _ms8 (empty = single-start)
#   N_SHARDS     fwd_sim only: total deviation shards (default 1)
#   CF_GPU       1 -> GPU path (gpu sysimage); 0 -> CPU path (cpu sysimage)
#   fwd_sim — WHICH shard(s) this job runs, first match wins:
#     BBL_SHARD_IDS  an explicit list/spec ("0-3", "3,50") — the manual override
#     BBL_SHARD_MAP  a map file, one task per line; this task runs line SLURM_ARRAY_TASK_ID+1
#                    (a PACKED job: bbl_run.sh / the sweep write the map, one line = k shards)
#     SLURM_ARRAY_TASK_ID, else SHARD_ID, else 0 — one shard per job, the task id IS the shard
#   One shard runs exactly as it always has: one Julia process, output in this job's log. k>1
#   shards run as k Julia processes side by side, each on ONE device of the job's
#   CUDA_VISIBLE_DEVICES and with its own log (logs/bbl_fwd_<rtag>_shard<i>_<jobid>.out).
#   bbl_fwd_sim.jl itself reads the shard only from its --shard-id argument.
#   sweep — BBL_KEY names the run (data/output/bbl/.dispatch/<key>), BBL_RETRY the attempt;
#     see the sweep branch below.
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

: "${BBL_STEP:?set BBL_STEP (warmup|polfunc|fwd_sim|sweep|solve|tables|report)}"
case "${BBL_STEP}" in warmup|polfunc|fwd_sim|sweep|solve|tables|report) ;;
    *) echo "Unknown BBL_STEP='${BBL_STEP}' (warmup|polfunc|fwd_sim|sweep|solve|tables|report)" >&2; exit 2 ;;
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
# The sweep is the same kind of step: it reads file names and parquet footers and calls sbatch,
# on a small `day` allocation that holds no GPU.
BBL_NEEDS_JULIA=1
case "${BBL_STEP}" in tables|report|sweep) BBL_NEEDS_JULIA=0 ;; esac
LOGD="$(cl_log_dir)"

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

    # Fatal CUDA gate, before the real work, only when the job actually holds a GPU. It checks
    # EVERY device in CUDA_VISIBLE_DEVICES, so a packed job proves all k of its GPUs here.
    if [[ "${CF_GPU}" != "0" && -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]]; then
        cl_gpu_gate
    fi
fi

cl_banner "BBL cost stage | step=${BBL_STEP} | E${BBL_ROUTINE} | stage=${BBL_STAGE}" \
          "R=${R} seed=${SEED} threads=${SLURM_CPUS_PER_TASK:-8} CF_GPU=${CF_GPU}" \
          "node=$(hostname) | $(date)"

# BBL_JULIA_THREADS is set only for the k processes of a packed job, which share the job's CPUs;
# everything else keeps the whole allocation, as before.
run_julia () {
    local script="$1"; shift
    julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} \
        --threads="${BBL_JULIA_THREADS:-${SLURM_CPUS_PER_TASK:-8}}" \
        "${CL_ROOT}/${script}" \
        --estim "${BBL_ROUTINE}" --spec 12 --stage "${BBL_STAGE}" \
        --R "${R}" --seed "${SEED}" --hpc ${BBL_EXTRA} "$@"
}

# GPU-memory sampler for fwd_sim: one nvidia-smi line per device every BBL_GPUMEM_SECS (60 s),
# reduced at the end to one "[gpumem] device=<i> peak_mib=<m> total_mib=<t>" line per device.
# Measurement only — it never touches the simulation — and it is what tells whether a shard that
# fits an H200 (141 GB) also fits an H100 (80 GB) at the new horizon.
_gpumem_start () {
    GPUMEM_PID=""
    [[ "${CF_GPU}" != "0" ]] || return 0
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    GPUMEM_FILE="${LOGD}/.gpumem_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.csv"
    ( while :; do
          nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
              >> "${GPUMEM_FILE}" 2>/dev/null || true
          sleep "${BBL_GPUMEM_SECS:-60}"
      done ) &
    GPUMEM_PID=$!
}
_gpumem_stop () {
    [[ -n "${GPUMEM_PID:-}" ]] || return 0
    kill "${GPUMEM_PID}" 2>/dev/null || true
    wait "${GPUMEM_PID}" 2>/dev/null || true
    awk -F', *' 'NF >= 3 { if ($2 + 0 > m[$1]) m[$1] = $2 + 0; t[$1] = $3 + 0 }
        END { for (i in m) printf "[gpumem] device=%s peak_mib=%d total_mib=%d\n", i, m[i], t[i] }' \
        "${GPUMEM_FILE}" 2>/dev/null || true
}

# The one summary line per shard, in the job log, that bbl_status.sh --probe reads.
_shard_line () {   # _shard_line <shard> <rc> <elapsed_s> <gpu> <peak_rss_gib> <device> <log>
    printf '[pack] shard=%s rc=%s elapsed_s=%s gpu=%s peak_rss_gib=%s device=%s log=%s\n' "$@"
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
        # T is the horizon THIS sim will run, read from the flags bbl_run.sh passed (it always
        # passes --horizon); bbl_discount.env only when a hand-built BBL_EXTRA omits it.
        T_SIM="$(sed -n 's/.*--horizon[ =]\([0-9][0-9]*\).*/\1/p' <<< "${BBL_EXTRA}")"
        [[ -n "${T_SIM}" ]] || T_SIM="$(cl_bbl_discount_get BBL_HORIZON)" \
            || { echo "ERROR: no --horizon in BBL_EXTRA and no BBL_HORIZON in ${CL_BBL_DISCOUNT_FILE}" >&2; exit 2; }
        RF_CURVE="${RF_CURVE:-${CL_DATA_IN}/forward_rf_qoq.csv}"
        if [[ ! -f "${RF_CURVE}" ]]; then
            echo "ERROR: forward r^f curve missing: ${RF_CURVE}" >&2
            echo "  Generate locally (needs internet) and upload to data/input/:" >&2
            echo "    python scrape_forward_rf.py --horizon ${T_SIM} --start 2026Q1" >&2
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
                echo "    python scrape_forward_rf.py --vintage-from 2016Q1 --vintage-to 2024Q4 --horizon ${T_SIM}" >&2
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
        # The evolving sleepy share evaluates the routine's exported link; `--phi-path frozen` is
        # the only way this run does not need it (bbl_run.sh always passes the flag).
        if [[ "${BBL_EXTRA}" != *"--phi-path frozen"* ]]; then
            SLINK="$(sed -n 's/.*--sleep-link[ =]\([^ ]*\).*/\1/p' <<< "${BBL_EXTRA}")"
            SLINK="${SLINK:-${CL_DATA_IN}/sleep_link_E${BBL_ROUTINE}_spec_12.json}"
            if [[ ! -f "${SLINK}" ]]; then
                echo "ERROR: sleepiness link missing: ${SLINK}" >&2
                echo "  Generate locally and upload to data/input/:  python bbl_sleep_link.py --est ${BBL_ROUTINE}" >&2
                echo "  Refusing to run: --phi-path evolving re-evaluates phi from it every period." >&2
                exit 1
            fi
            echo "sleep link OK: ${SLINK}"
        fi
        if [[ "${BBL_EXTRA}" == *"--multi-start"* || "${CF_EVOLVING_STATES:-1}" != "0" \
              || "${BBL_EXTRA}" != *"--phi-path frozen"* || "${BBL_EXTRA}" != *"--z-path frozen"* ]]; then
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
        # The Focus curve must reach the horizon (cl_bbl_check_curve): checked on the file the sim
        # will actually read, at the T it will actually use, both taken from BBL_EXTRA.
        if [[ "${BBL_EXTRA}" == *"--multi-start"* ]]; then
            CURVE_F="$(sed -n 's/.*--rf-vintages[ =]\([^ ]*\).*/\1/p' <<< "${BBL_EXTRA}")"
            cl_bbl_check_curve "${T_SIM}" 1 "${CURVE_F:-${RF_VINT}}" || exit 1
        else
            cl_bbl_check_curve "${T_SIM}" 0 "${RF_CURVE}" || exit 1
        fi
        N_SHARDS="${N_SHARDS:-1}"
        # WHICH shards — see the header. The map is read by LINE, so a packed task can hold any
        # list (3 50 51 52), which is what lets the sweep re-run an arbitrary gap packed.
        if [[ -n "${BBL_SHARD_IDS:-}" ]]; then
            SHARD_IDS="$(cl_expand_spec "${BBL_SHARD_IDS}")" \
                || { echo "ERROR: BBL_SHARD_IDS='${BBL_SHARD_IDS}' is not an index list" >&2; exit 2; }
            SHARD_SRC="BBL_SHARD_IDS=${BBL_SHARD_IDS}"
        elif [[ -n "${BBL_SHARD_MAP:-}" ]]; then
            [[ -f "${BBL_SHARD_MAP}" ]] || { echo "ERROR: shard map not found: ${BBL_SHARD_MAP}" >&2; exit 1; }
            MAP_LINE=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))
            SHARD_IDS="$(sed -n "${MAP_LINE}p" "${BBL_SHARD_MAP}")"
            [[ -n "${SHARD_IDS// /}" ]] \
                || { echo "ERROR: ${BBL_SHARD_MAP} has no line ${MAP_LINE} (array task ${SLURM_ARRAY_TASK_ID:-0})" >&2; exit 1; }
            SHARD_SRC="line ${MAP_LINE} of $(basename "${BBL_SHARD_MAP}")"
        else
            SHARD_IDS="${SLURM_ARRAY_TASK_ID:-${SHARD_ID:-0}}"
            SHARD_SRC="the array task id"
        fi
        read -r -a SIDS <<< "${SHARD_IDS}"
        for s in "${SIDS[@]}"; do
            [[ "${s}" =~ ^[0-9]+$ ]] && (( s < N_SHARDS )) \
                || { echo "ERROR: shard '${s}' is not in 0..$(( N_SHARDS - 1 ))" >&2; exit 2; }
        done
        echo "fwd_sim: ${#SIDS[@]} shard(s) [${SIDS[*]}] of ${N_SHARDS} (${SHARD_SRC})"
        RTAG="${BBL_RTAG:-E${BBL_ROUTINE}${PSI_TAG}}"

        if (( ${#SIDS[@]} == 1 )); then
            # ONE shard: the invocation it has always been — one Julia process, its output in
            # this job's own log. Only the timing line after it is new.
            SHARD_ID="${SIDS[0]}"
            _gpumem_start
            t0=$(date +%s)
            if run_julia bbl_fwd_sim.jl --n-shards "${N_SHARDS}" --shard-id "${SHARD_ID}"; then rc=0; else rc=$?; fi
            _gpumem_stop
            _shard_line "${SHARD_ID}" "${rc}" "$(( $(date +%s) - t0 ))" "see-this-log" "see-this-log" \
                        "${CUDA_VISIBLE_DEVICES:-cpu}" "this-job"
            (( rc == 0 )) || exit "${rc}"
        else
            # k shards packed into one job: k Julia processes side by side, each pinned to ONE
            # device of this job's CUDA_VISIBLE_DEVICES (parsed, never assumed to be 0..k-1),
            # each given its share of the job's CPUs as Julia threads (BBL_THREADS_PER_SHARD) and
            # its own explicit --shard-id, each writing its own log. A packed job's wall is its
            # slowest child's. The sysimage and the CUDA gate above ran ONCE for the whole job,
            # and the gate exercised every device in the list.
            #
            # REFUSED without the sysimage, whatever ALLOW_SYSIMAGE_FALLBACK says: k processes
            # precompiling against the shared NFS depot at once is exactly the race that ends in
            # CUDA.functional()==false and a silent CPU share path. With the image nothing
            # compiles and k concurrent loads are safe.
            K=${#SIDS[@]}
            if (( ${#CL_JULIA_SYS[@]} == 0 )); then
                echo "REFUSING: ${K} shards in one job need the sysimage; without it ${K} concurrent Julia loads" >&2
                echo "  race on the shared-NFS precompile. ALLOW_SYSIMAGE_FALLBACK does not apply to a packed" >&2
                echo "  job: rebuild the image, or run PACK=1." >&2
                exit 1
            fi
            DEVS=()
            if [[ "${CF_GPU}" != "0" ]]; then
                IFS=, read -r -a DEVS <<< "${CUDA_VISIBLE_DEVICES:-}"
                if (( ${#DEVS[@]} < K )); then
                    echo "REFUSING: ${K} shards but CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' lists ${#DEVS[@]} device(s)." >&2
                    exit 1
                fi
            fi
            TPS="${BBL_THREADS_PER_SHARD:-$(( ${SLURM_CPUS_PER_TASK:-8} / K ))}"
            (( TPS >= 1 )) || TPS=1
            _gpumem_start
            declare -A CPID CLOG CDEV
            for i in "${!SIDS[@]}"; do
                s="${SIDS[$i]}"
                CLOG[$s]="${LOGD}/bbl_fwd_${RTAG}_shard${s}_${SLURM_JOB_ID:-local}.out"
                CDEV[$s]="${DEVS[$i]:-cpu}"
                (
                    if [[ "${CDEV[$s]}" != "cpu" ]]; then export CUDA_VISIBLE_DEVICES="${CDEV[$s]}"; fi
                    export BBL_JULIA_THREADS="${TPS}"
                    echo "[child] shard=${s} device=${CDEV[$s]} threads=${TPS} start_epoch=$(date +%s) host=$(hostname)"
                    if run_julia bbl_fwd_sim.jl --n-shards "${N_SHARDS}" --shard-id "${s}"; then crc=0; else crc=$?; fi
                    echo "[child] shard=${s} end_epoch=$(date +%s) rc=${crc}"
                    exit "${crc}"
                ) > "${CLOG[$s]}" 2>&1 &
                CPID[$s]=$!
                echo "[pack] launched shard=${s} device=${CDEV[$s]} threads=${TPS} pid=${CPID[$s]} log=${CLOG[$s]}"
            done
            NFAIL=0; NCPU=0
            for s in "${SIDS[@]}"; do
                if wait "${CPID[$s]}"; then crc=0; else crc=$?; fi
                st="$(sed -n 's/^\[child\] shard=[0-9]* device=.* start_epoch=\([0-9]*\).*/\1/p' "${CLOG[$s]}" | head -1)"
                en="$(sed -n 's/^\[child\] shard=[0-9]* end_epoch=\([0-9]*\).*/\1/p' "${CLOG[$s]}" | tail -1)"
                el="?"; [[ -n "${st}" && -n "${en}" ]] && el=$(( en - st ))
                gpu=no; grep -q "GPU share kernel enabled" "${CLOG[$s]}" && gpu=yes
                rss="$(sed -n 's/.*\[BBL\] shard [0-9]* peak RSS \([0-9.]*\) GiB.*/\1/p' "${CLOG[$s]}" | tail -1)"
                _shard_line "${s}" "${crc}" "${el}" "${gpu}" "${rss:-?}" "${CDEV[$s]}" "${CLOG[$s]}"
                if (( crc != 0 )); then NFAIL=$(( NFAIL + 1 )); fi
                if (( crc == 0 )) && [[ "${CF_GPU}" != "0" && "${gpu}" != "yes" ]]; then NCPU=$(( NCPU + 1 )); fi
            done
            _gpumem_stop
            if (( NCPU > 0 )); then
                echo "[!] ${NCPU} shard(s) finished WITHOUT 'GPU share kernel enabled' in their log: they took the CPU" >&2
                echo "    share path on a GPU node. The psi is still right (GPU == CPU to 3e-14) but the GPU sat idle;" >&2
                echo "    read the sysimage / CUDA lines at the top of those shard logs." >&2
            fi
            if (( NFAIL > 0 )); then
                echo "[pack] ${NFAIL} of ${K} shard(s) FAILED (rc above). The others wrote complete files (atomic" >&2
                echo "       rename); the sweep re-runs exactly the missing ones." >&2
                exit 1
            fi
        fi ;;

    sweep)
        # THE SELF-HEALING STEP. Submitted afterany a routine's fwd jobs (bbl_run.sh), it asks
        # which of the N shards really exist — a readable parquet under the final name, with the
        # psi columns, newer than the launch on a fresh run — and then does exactly one of:
        #   complete             exit 0; the solve (afterok this job) runs.
        #   gaps, retries left   re-submit EXACTLY the missing indices (same N_SHARDS, same tag,
        #                        the run's own settings from context.env, a longer wall), chain
        #                        the next sweep afterany them, move the solve's afterok onto that
        #                        sweep, exit 0. The solve keeps its job id, so the tables, the
        #                        archive and the CF jobs chained on it never need re-chaining.
        #   gaps, none left      exit 1 naming the indices: the afterok solve is CANCELLED rather
        #                        than run on a partial set, and everything after it with it.
        # A PROBLEM no re-run can fix (a second shard-count family beside this one, psi written
        # under another beta or T) exits 2 at once.
        : "${BBL_KEY:?the sweep needs BBL_KEY (bbl_run.sh exports it)}"
        DD="$(cl_bbl_dispatch_dir "${BBL_KEY}")"
        if ! cl_bbl_ctx_load "${DD}/context.env"; then
            echo "ERROR: no dispatch context ${DD}/context.env. The sweep re-submits with the run's OWN" >&2
            echo "  settings and has none to use. Re-launch through bbl_run.sh (--repair for an existing run)." >&2
            exit 2
        fi
        RETRY="${BBL_RETRY:-0}"; MAXR="${BBL_MAX_RETRIES:-3}"
        echo "sweep ${BBL_KEY}: attempt ${RETRY}/${MAXR} | N_SHARDS=${N_SHARDS} T=${HORIZON} beta=${BETA} | fresh=${BBL_FRESH:-0}"
        if [[ -f "${DD}/STOP" ]]; then
            echo "STOP marker present (${DD}/STOP, written by bbl_cancel.sh): re-submitting nothing."
            cl_bbl_event "${BBL_KEY}" "SWEEP retry=${RETRY} result=stopped"
            exit 1
        fi
        # A run this code did not launch (another design version, or none recorded) is never
        # completed by it: its missing shards would be simulated under a different design.
        if ! cl_bbl_version_guard "${BBL_KEY}" "the sweep" 2>&1; then
            cl_bbl_event "${BBL_KEY}" "SWEEP retry=${RETRY} result=problem (sim_version)"
            echo "SWEEP REFUSES ${BBL_KEY}: design version (see above). The solve (afterok this job) is cancelled."
            echo "SWEEP REFUSES ${BBL_KEY} (sim_version)" >&2
            exit 2
        fi
        cl_setup_python "${CL_PY_REQ_SWEEP}"
        COV=(coverage --dir "${CF_COST_FWD}" --key "${BBL_KEY}" --n-shards "${N_SHARDS}" --validate)
        if [[ "${MULTI_START}" == "1" ]]; then
            COV+=(--expect-starts --expect-beta "${BETA}" --expect-horizon "${HORIZON}")
            # The run's own switches and version (its context passed cl_bbl_version_guard above),
            # and the policy CSV the re-run shards would read: it must be the one the psi on disk
            # were simulated around (psi_starts records its sha256).
            COV+=(--expect-phi-path "${PHI_PATH}" --expect-z-path "${Z_PATH}" --expect-rdep-timing "${RDEP_TIMING}")
            COV+=(--expect-sim-version "${BBL_SIM_VERSION}")
            if [[ -n "${POLICY_CSV:-}" ]]; then COV+=(--policy-csv "${POLICY_CSV}"); fi
        fi
        if [[ "${BBL_FRESH:-0}" == "1" ]]; then COV+=(--newer-than "${RUN_EPOCH}"); fi
        # tr keeps the parse immune to a CRLF interpreter; pipefail keeps Python's exit code.
        if COUT="$("${PYBIN}" "${CL_ROOT}/bbl_shards.py" "${COV[@]}" | tr -d '\r')"; then CRC=0; else CRC=$?; fi
        printf '%s\n' "${COUT}"
        case "${CRC}" in
            0)  cl_bbl_event "${BBL_KEY}" "SWEEP retry=${RETRY} result=complete n=${N_SHARDS}"
                echo "SWEEP COMPLETE: ${N_SHARDS}/${N_SHARDS} shards valid; the solve runs next."
                exit 0 ;;
            1)  ;;
            2)  cl_bbl_event "${BBL_KEY}" "SWEEP retry=${RETRY} result=problem"
                echo "SWEEP REFUSES: the PROBLEM line(s) above cannot be fixed by re-running shards."
                echo "  The solve (afterok this job) is cancelled. Fix the cause, then:"
                echo "    bash bbl_run.sh --routines ${BBL_ROUTINE} --shards ${N_SHARDS} --psi-tag '${PSI_TAG}' --repair"
                echo "SWEEP REFUSES ${BBL_KEY} (see .out)" >&2
                exit 2 ;;
            *)  echo "SWEEP: the coverage check itself failed (rc=${CRC})." >&2; exit 3 ;;
        esac
        read -r -a MISS <<< "$(sed -n 's/^MISSING_LIST=//p' <<< "${COUT}")"
        MSPEC="$(cl_compress_ranges "${MISS[@]}")"
        # Shards another LIVE job is already computing (a probe, a hand-submitted re-run) are
        # waited for, never duplicated: two writers on one shard is how a truncated file used to
        # appear. A live job with no claim file could be computing anything, so the sweep waits
        # for it and re-submits nothing this round.
        INFL_J=""; INFL_I=""; UNKNOWN=0
        while read -r j rest; do
            [[ -n "${j}" ]] || continue
            if [[ "${rest}" == "UNKNOWN" ]]; then UNKNOWN=1; INFL_J="${INFL_J:+${INFL_J}:}${j}"; continue; fi
            hit="$(comm -12 <(printf '%s\n' ${rest} | sort -u) <(printf '%s\n' "${MISS[@]}" | sort -u) | paste -sd' ' -)"
            if [[ -n "${hit}" ]]; then INFL_J="${INFL_J:+${INFL_J}:}${j}"; INFL_I="${INFL_I} ${hit}"; fi
        done < <(cl_bbl_live_claims "${BBL_KEY}" "${BBL_RTAG}")
        if (( UNKNOWN )); then
            TODO=""
        else
            TODO="$(comm -23 <(printf '%s\n' "${MISS[@]}" | sort -u) <(printf '%s\n' ${INFL_I} | awk 'NF' | sort -u) \
                    | sort -n | paste -sd' ' -)"
        fi
        if (( RETRY >= MAXR )); then
            cl_bbl_event "${BBL_KEY}" "SWEEP retry=${RETRY} result=exhausted missing=${MSPEC}"
            echo "SWEEP EXHAUSTED: after ${RETRY} re-run(s), ${#MISS[@]} of ${N_SHARDS} shards of ${BBL_KEY} are still missing:"
            echo "    ${MSPEC}"
            echo "  The solve (afterok this job) is CANCELLED, so nothing is estimated from a partial set."
            echo "  Why those shards died:  sacct --name=bbl_fwd_${BBL_RTAG} -X -o JobID,State,Elapsed,MaxRSS"
            echo "                          ls ${LOGD}/bbl_fwd_${BBL_RTAG}_*"
            echo "  After fixing the cause (a longer --shard-time, more memory, ...):"
            echo "    bash bbl_run.sh --routines ${BBL_ROUTINE} --shards ${N_SHARDS} --psi-tag '${PSI_TAG}' --repair"
            echo "SWEEP EXHAUSTED ${BBL_KEY}: missing ${MSPEC}" >&2
            exit 1
        fi
        NEXT=$(( RETRY + 1 ))
        MULT="$(awk -v m="${BBL_RETRY_WALL_MULT:-1.5}" -v n="${NEXT}" 'BEGIN{ printf "%g", m ^ n }')"
        FWD_PARTITIONS="${SWEEP_PARTITIONS:-${FWD_PARTITIONS}}"
        echo "re-run ${NEXT}/${MAXR}: missing ${MSPEC}; already in flight: $(cl_compress_ranges ${INFL_I})$( (( UNKNOWN )) && printf ' + unclaimed live job(s)' || true ); re-submitting: $(cl_compress_ranges ${TODO}) at wall x ${MULT}"
        CL_BBL_JIDS=""
        if [[ -n "${TODO}" ]]; then
            cl_bbl_dispatch_fwd "${TODO}" "${MULT}" "" || { echo "SWEEP: the re-submission was refused (above)." >&2; exit 4; }
        fi
        NEW_FWD="${CL_BBL_JIDS}"
        DEPS="${NEW_FWD}${INFL_J:+${NEW_FWD:+:}${INFL_J}}"
        [[ -n "${DEPS}" ]] || { echo "SWEEP: shards are missing but nothing was submitted or is in flight; refusing." >&2; exit 4; }
        # Held until the solve points at it, so it can never run (and exit 0) while the solve still
        # waits on THIS job.
        NXT="$(cl_bbl_submit_sweep "${NEXT}" --hold "--dependency=afterany:${DEPS}")" || NXT=""
        if [[ -z "${NXT}" ]]; then
            echo "SWEEP: could not submit the next sweep; withdrawing this re-run." >&2
            if [[ -n "${NEW_FWD}" && "${CL_DRYRUN}" != "1" ]]; then scancel ${NEW_FWD//:/ } 2>/dev/null || true; fi
            exit 4
        fi
        echo "  bbl_sweep_${BBL_RTAG} (attempt ${NEXT}) afterany ${DEPS} -> ${NXT} [held]"
        if ! cl_bbl_retarget_solve "${BBL_KEY}" "${NXT}"; then
            echo "SWEEP: the solve could not be moved onto sweep ${NXT}; withdrawing this re-run so no orphan" >&2
            echo "  chain is left. The solve is cancelled with this job's failure." >&2
            if [[ "${CL_DRYRUN}" != "1" ]]; then scancel "${NXT}" ${NEW_FWD//:/ } 2>/dev/null || true; fi
            exit 4
        fi
        cl_bbl_event "${BBL_KEY}" "SWEEP retry=${RETRY} result=resubmit missing=${MSPEC} fwd=${NEW_FWD:-none} inflight=${INFL_J:-none} next_sweep=${NXT}"
        if [[ "${CL_DRYRUN}" == "1" ]]; then
            echo "  [dry-run] scontrol release ${NXT}"
        elif ! scontrol release "${NXT}"; then
            echo "SWEEP: sweep ${NXT} is chained and the solve points at it, but it is still HELD." >&2
            echo "  Release it by hand:  scontrol release ${NXT}" >&2
            exit 5
        fi
        echo "SWEEP RE-RUN ${NEXT}/${MAXR}: fwd ${NEW_FWD:-none} + in flight ${INFL_J:-none} -> sweep ${NXT} -> solve"
        exit 0 ;;

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
