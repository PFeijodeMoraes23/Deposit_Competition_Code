#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# (job-name + .out/.err are set per job by blp_run.sh via sbatch -J/-o/-e)
# ==============================================================================
# blp_stage_job.sh — THE generic RC-BLP worker: one (routine, engine, stage-set)
# per invocation. blp_run.sh submits these as --dependency=afterok chains so each
# stage warm-starts theta2 from the previous stage's on-disk checkpoint.
#
# The #SBATCH block is a FLOOR. --time and --mem always arrive as sbatch CLI
# overrides from blp_run.sh (cl_mem_for gives E1/E2 600G; 200G OOM-killed them at
# ext1 in job 21525240 — the n_pi*N*R hot buffer at blp_1_estimation.jl:357 grows
# ~38 GB -> ~51 GB from rc4 to ext1).
#
# Driven by env vars exported through sbatch --export=ALL:
#   RC_ROUTINE  1 | 2 | 3 | 4
#   RC_ENGINE   ift | numerical | cue     -> BLP_ENGINE (cue = continuously-updated GMM, suffix _cue)
#   RC_STAGE    sigma|rc2|rc3|rc4|full|ext1|ext2|extended, or several joined with '+'
#   R SEED TOL_INNER MAX_INNER TOL_OUTER   engine knobs
#   BLP_SE_ONLY=1                          recompute SEs from existing checkpoints
#
# WHAT MUST EXIST FIRST: cluster_preflight.sh green, the GPU sysimage built, the
#   R=2000 draws, and data/input/logit_delta_E{k}_spec_12.bin for the sigma stage.
# WHAT TO RUN NEXT: nothing directly — blp_run.sh owns the chain.
#
# THIS SCRIPT IS NEW. submit_blp_rc_stage.sh is still present and still works; it
# is the fallback and is retired only after one successful cluster cycle. Nothing
# in it has been modified.
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

: "${RC_ROUTINE:?set RC_ROUTINE (1..4)}"
: "${RC_ENGINE:?set RC_ENGINE (ift|numerical|cue)}"
: "${RC_STAGE:?set RC_STAGE (sigma|rc2|rc3|rc4|full|ext1|ext2|extended)}"

# ${VAR:?} only catches unset/empty, so validate the VALUE too. An unrecognised
# engine falls through to IFT with an EMPTY output suffix and OVERWRITES the
# production artifacts. This allow-list is deliberately duplicated in
# blp_2_rc.jl:123-127 and blp_gpu_engine.jl:2817-2824 — defence in depth against
# data loss, not waste. Failing here costs seconds instead of a module load plus
# Julia/CUDA startup.
case "${RC_ENGINE}" in
    ift|numerical|cue) ;;
    *) echo "RC_ENGINE='${RC_ENGINE}' is not a recognised engine (ift|numerical|cue)" >&2; exit 2 ;;
esac

mkdir -p "${CL_ROOT}/logs"
cl_load_julia
export BLP_ENGINE="${RC_ENGINE}"                 # read by blp_2_rc.jl

echo "GPU node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Sysimage: a REFUSAL, not a fall-back ─────────────────────────────────────
# Without a matching image every concurrent stage job loads packages fresh from
# the ONE shared-NFS depot and stampedes the precompile lock; the loser of that
# race gets CUDA.functional()==false and runs on CPU while holding an H200.
# cl_require_sysimage checks existence, mtime, the provenance sidecar
# (julia_version + Manifest sha256 + cpu_target) and then LOADS the image on this
# node. ALLOW_SYSIMAGE_FALLBACK=1 restores the old degrade, loudly.
cl_require_sysimage gpu

# ── Fatal CUDA gate, before the real work ────────────────────────────────────
# Only fires for a job actually holding a GPU, so a CPU-only re-run is untouched.
if [[ -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]]; then
    cl_gpu_gate
else
    echo "[note] no GPU allocated to this job — CUDA gate skipped."
fi

# ── Engine knobs ─────────────────────────────────────────────────────────────
#   TOL_INNER : inner delta-contraction tolerance. 1e-10 is tight; if a stage
#               stalls hitting MAX_INNER every outer step ("did not converge in
#               5000"), loosen to 1e-8 — the outer GMM does not need 1e-10.
#   MAX_INNER : a pure SAFETY CEILING, not a tuning knob. SQUAREM self-terminates
#               at tol-inner=1e-10 in ~120-185 iters, so 5000 never binds.
#               Lowering it would force an early exit at a NON-converged delta,
#               biasing Q and grad-Q (Dube-Fox-Su) and breaking outer convergence.
#               The deep stages get more wall-time instead.
#   R         : simulation draws. Lowering (e.g. 1000) cuts per-iter cost ~linearly
#               but REQUIRES regenerated draws: halton_nu_R<R>_seed<SEED>.jls.
R="${R:-2000}"
SEED="${SEED:-42}"
TOL_INNER="${TOL_INNER:-1e-10}"
MAX_INNER="${MAX_INNER:-5000}"
TOL_OUTER="${TOL_OUTER:-1e-6}"

# Grouped jobs pass several stages joined with '+', because sbatch --export uses
# commas to separate variables and so cannot carry a comma list. A single stage
# has no '+', so this is a no-op there.
STAGE_ARG="${RC_STAGE//+/,}"

cl_banner "RC-BLP E${RC_ROUTINE} | engine=${RC_ENGINE} | stage=${STAGE_ARG}" \
          "spec=12 | R=${R} | tol_inner=${TOL_INNER} | max_inner=${MAX_INNER} | threads=${SLURM_CPUS_PER_TASK:-6}" \
          "mem=${SLURM_MEM_PER_NODE:-?} | $(date)"

# BLP_SE_ONLY=1 -> recompute SEs from each stage's existing checkpoint instead of
# re-optimising. Valid only when the change is confined to the SE routine (it runs
# AFTER optimisation, so the point estimates provably cannot move). Stages are
# INDEPENDENT in this mode — no warm-start chain is needed.
SE_ONLY_ARG=()
[[ "${BLP_SE_ONLY:-0}" == "1" ]] && SE_ONLY_ARG=(--se-only)

julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} \
    --threads="${SLURM_CPUS_PER_TASK:-6}" \
    "${CL_ROOT}/blp_2_rc.jl" \
    --estim "${RC_ROUTINE}" --stage "${STAGE_ARG}" \
    --hpc --R "${R}" --seed "${SEED}" \
    --tol-inner "${TOL_INNER}" --max-inner "${MAX_INNER}" --tol-outer "${TOL_OUTER}" \
    ${SE_ONLY_ARG[@]+"${SE_ONLY_ARG[@]}"}

echo "RC-BLP E${RC_ROUTINE} ${RC_ENGINE} ${RC_STAGE} complete: $(date)"
