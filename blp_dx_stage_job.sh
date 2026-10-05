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
# (job-name + .out/.err are set per job by dx_suite_20261001.sh via sbatch -J/-o/-e)
# ==============================================================================
# blp_dx_stage_job.sh — the RC-BLP worker of the `dx` demand variant (the D-type
# dummy in the linear part; blp_dx.jl): one (routine, stage-set) per invocation,
# on the GPU IFT engine. The counterpart of blp_stage_job.sh, which it follows
# step for step; the one difference is the Julia entry point, blp_dx_rc.jl.
# dx_suite_20261001.sh submits these as --dependency=afterok chains. It is a cold
# ladder: nothing is seeded from the main specification. rc2-rc4, ext2 and extended
# start theta2 from the variant's own previous checkpoint; ext1 starts from the seed
# (the engine looks for a `full` checkpoint, which the grouped ladder never writes),
# as it did in the main run.
#
# The #SBATCH block is a FLOOR. --time and --mem arrive as sbatch CLI overrides.
#
# Driven by env vars exported through sbatch --export=ALL:
#   RC_ROUTINE  3 | 4
#   RC_STAGE    sigma|rc2|rc3|rc4|ext1|ext2|extended, or several joined with '+'
#   R SEED TOL_INNER MAX_INNER TOL_OUTER   engine knobs
#   BLP_SE_ONLY=1                          recompute SEs from the variant's checkpoints
#   DX_ALLOW_NO_SE=1                       accept ext1 without joint standard errors (flagged);
#                                          without it such a rung ends nonzero (blp_dx_rc.jl)
#   DX_SMOKE=1                             the smoke: the engine's --dry-run (self-test, nesting
#                                          check, design, GPU share check, 10 contraction
#                                          iterations); writes no result (one empty
#                                          blp_summary_*_dx.json). It ends nonzero and prints
#                                          DX SMOKE FAIL unless every step left evidence, and
#                                          prints DX SMOKE PASS otherwise. A smoke is its own job
#                                          (-J dx_smoke): a ladder rung (rc_ift_dx_*) that sees
#                                          DX_SMOKE=1 is refused, and the suite submits its rungs
#                                          with DX_SMOKE=0.
#
# WHERE ITS ARTIFACTS GO: data/output/blp, next to the main specification's and
#   never under its names: every file carries `_dx` before the extension
#   (blp_results_E{k}_spec_12_{stage}_dx.jls/.json, blp_checkpoint_..._dx.jls/.bin,
#   blp_summary_E{k}_{stage}_gpu_ift_dx.json, blp_meta_..._dx.json,
#   blp_linear_E{k}_spec_12_dx.json).
#
# WHAT MUST EXIST FIRST: the GPU sysimage, the R=2000 draws under
#   data/output/blp/draws, the demand parquet and the logit step's
#   data/output/logit/logit_delta_E{k}_spec_12.bin. All are the main
#   specification's; the variant writes none of them.
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

: "${RC_ROUTINE:?set RC_ROUTINE (3|4)}"
: "${RC_STAGE:?set RC_STAGE (sigma|rc2|rc3|rc4|ext1|ext2|extended)}"
case "+${RC_STAGE}+" in
    *+logit+*) echo "RC_STAGE='${RC_STAGE}': stage logit is not part of the variant (it rewrites the shared logit delta)" >&2; exit 2 ;;
esac
case "${DX_SMOKE:-0}" in
    0) SMOKE=0 ;;
    1) SMOKE=1 ;;
    *) echo "DX_SMOKE='${DX_SMOKE}': 1 makes this job the smoke (a dry run); leave it unset otherwise" >&2; exit 2 ;;
esac
if [[ "${SMOKE}" == "1" && "${SLURM_JOB_NAME:-}" == rc_ift_dx_* ]]; then
    echo "DX_SMOKE=1 reached the ladder rung ${SLURM_JOB_NAME}: refused. As a dry run the rung would end with status 0 and no result. Unset DX_SMOKE where the suite is launched." >&2
    exit 2
fi

mkdir -p "${CL_ROOT}/logs"

cl_bootstrap_tree
cl_export_step_dirs

cl_load_julia
export BLP_ENGINE="ift"                          # read by blp_rc.jl; blp_dx_rc.jl accepts no other
# A cold ladder: the cross-engine theta2 seed of the main pipeline has no place here, and an
# inherited value (--export=ALL) would start every stage of this job from another fit.
unset BLP_THETA2_INIT_FILE

echo "GPU node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Sysimage: a REFUSAL, not a fall-back (see blp_stage_job.sh) ──────────────
cl_require_sysimage gpu

# ── Fatal CUDA gate, before the real work ────────────────────────────────────
if [[ -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]]; then
    cl_gpu_gate
else
    echo "[note] no GPU allocated to this job — CUDA gate skipped."
fi

# ── Engine knobs: the main ladder's values (blp_stage_job.sh) ────────────────
R="${R:-2000}"
SEED="${SEED:-42}"
TOL_INNER="${TOL_INNER:-1e-10}"
MAX_INNER="${MAX_INNER:-5000}"
TOL_OUTER="${TOL_OUTER:-1e-6}"

STAGE_ARG="${RC_STAGE//+/,}"

cl_banner "RC-BLP E${RC_ROUTINE} | variant dx (D-type dummy in X1) | engine=ift | stage=${STAGE_ARG}" \
          "spec=12 | R=${R} | tol_inner=${TOL_INNER} | max_inner=${MAX_INNER} | threads=${SLURM_CPUS_PER_TASK:-6}" \
          "mem=${SLURM_MEM_PER_NODE:-?} | $(date)"

SE_ONLY_ARG=()
[[ "${BLP_SE_ONLY:-0}" == "1" ]] && SE_ONLY_ARG=(--se-only)
[[ "${SMOKE}" == "1" ]] && SE_ONLY_ARG+=(--dry-run)

# The smoke's verdict is julia's exit status: blp_dx_rc.jl ends a dry run nonzero unless every
# step left positive evidence (the engine itself catches a failed stage and exits 0).
rc=0
julia --project="${CL_ROOT}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} \
    --threads="${SLURM_CPUS_PER_TASK:-6}" \
    "${CL_ROOT}/blp_dx_rc.jl" \
    --estim "${RC_ROUTINE}" --stage "${STAGE_ARG}" \
    --hpc --R "${R}" --seed "${SEED}" \
    --tol-inner "${TOL_INNER}" --max-inner "${MAX_INNER}" --tol-outer "${TOL_OUTER}" \
    ${SE_ONLY_ARG[@]+"${SE_ONLY_ARG[@]}"} || rc=$?

if [[ "${rc}" -ne 0 ]]; then
    [[ "${SMOKE}" == "1" ]] && echo "DX SMOKE FAIL: E${RC_ROUTINE} ${RC_STAGE} (julia exit ${rc}); the reason is above. Do not launch the suite."
    exit "${rc}"
fi
if [[ "${SMOKE}" == "1" ]]; then echo "DX SMOKE PASS: E${RC_ROUTINE} ${RC_STAGE} (dry run, no result written): $(date)"
else echo "RC-BLP E${RC_ROUTINE} dx ${RC_STAGE} complete: $(date)"; fi
