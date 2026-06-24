#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --gpus=h200:1
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# (job-name + .out/.err are set per job by the orchestrator via sbatch -J/-o/-e)
#
# ONE coherence RC-BLP stage for one (routine, engine). The orchestrator
# (submit_blp_2_rc_coherence_all.sh) submits these as a --dependency=afterok chain
# so each stage warm-starts θ₂ from the previous stage's checkpoint. Driven by env
# vars exported by the orchestrator (sbatch --export):
#   COH_ROUTINE  1 | 2 | 3 | 4 | 5 | 6
#   COH_ENGINE   ift | numerical          → BLP_COHERENCE_ENGINE
#   COH_STAGE    sigma|rc2|rc3|rc4|full|ext1|ext2|extended
# The first stage (sigma) warm-starts δ from logit_delta_E{k}_spec_12_coherence.bin.

set -euo pipefail
: "${COH_ROUTINE:?set COH_ROUTINE (1..6)}"
: "${COH_ENGINE:?set COH_ENGINE (ift|numerical)}"
: "${COH_STAGE:?set COH_STAGE (sigma|rc2|rc3|rc4|full|ext1|ext2|extended)}"

# max-inner is a pure SAFETY CEILING, not a tuning knob: SQUAREM self-terminates
# at tol-inner=1e-10 in ~120-185 iters (see logs), so 5000 never actually binds.
# Lowering it would only matter at a pathological trial θ₂ — and there it would
# force an early exit at a NON-converged δ, biasing Q and ∇Q (Dubé–Fox–Su) and
# breaking outer convergence. Keep it high for every stage. The deep stages get
# more wall-time instead (sbatch --time in submit_blp_rc_coherence_all.sh); their
# cost is the NUMBER of contractions, which the IFT engine cuts ~8× — not the
# length of any single contraction.

# ── Environment ───────────────────────────────────────────────────────────────
module reset
module load Julia/1.11.4-linux-x86_64           # LLVM 16 → PTX for sm_90 (H200)
export JULIA_MKL_THREADING=tbb
export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/.julia_depot:${JULIA_DEPOT_PATH:-}"
export BLP_COHERENCE_ENGINE="${COH_ENGINE}"      # read by blp_2_rc_coherence.jl

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

echo "GPU node: $(hostname)"
echo "CUDA devices: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Sysimage-aware Julia setup (sets the JULIA_SYS array; skips precompile when
# blp_sysimage.so is present). Shared with the standalone submit scripts.
source "${PROJECT_DIR}/julia_sysimage_env.sh"

# Grouped (option-6) jobs pass several stages joined with '+', because sbatch
# --export uses commas to separate variables and so cannot carry a comma list.
# Translate '+' → ',' for the engine; a single stage has no '+', so this is a
# no-op and the per-stage orchestrator is unaffected.
STAGE_ARG="${COH_STAGE//+/,}"

# Engine run parameters — env-overridable (exported through sbatch --export=ALL).
#   TOL_INNER : inner δ-contraction tolerance. 1e-10 is tight; if a stage stalls
#               hitting MAX_INNER every outer step ("did not converge in 5000"),
#               loosen to 1e-8 (or 1e-7) — the outer GMM does NOT need δ to 1e-10.
#   R         : simulation draws. Lowering (e.g. 1000) cuts per-iter cost ~linearly
#               but REQUIRES regenerated draws: halton_nu_R<R>_seed<SEED>.jls etc.
R="${R:-2000}"
SEED="${SEED:-42}"
TOL_INNER="${TOL_INNER:-1e-10}"
MAX_INNER="${MAX_INNER:-5000}"
TOL_OUTER="${TOL_OUTER:-1e-6}"

echo "======================================"
echo " Coherence RC-BLP E${COH_ROUTINE} | engine=${COH_ENGINE} | stage=${STAGE_ARG} — $(date)"
echo " spec=12 | R=${R} | tol_inner=${TOL_INNER} | max_inner=${MAX_INNER} | threads=${SLURM_CPUS_PER_TASK}"
echo "======================================"

julia --project="${PROJECT_DIR}" ${JULIA_SYS[@]+"${JULIA_SYS[@]}"} --threads=${SLURM_CPUS_PER_TASK} \
    "${PROJECT_DIR}/blp_2_rc_coherence.jl" \
    --estim "${COH_ROUTINE}" --stage "${STAGE_ARG}" \
    --hpc --R "${R}" --seed "${SEED}" \
    --tol-inner "${TOL_INNER}" --max-inner "${MAX_INNER}" --tol-outer "${TOL_OUTER}"

echo "Coherence RC-BLP E${COH_ROUTINE} ${COH_ENGINE} ${COH_STAGE} complete: $(date)"
