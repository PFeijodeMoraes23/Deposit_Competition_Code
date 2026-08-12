#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
# ^ DEFAULT only. submit_blp_build_and_run_spec12.sh passes an sbatch command-line --mem per routine
#   (MEM_DEFAULT for E5-E8, MEM_BIG for E1/E2), which OVERRIDES this directive. E1/E2 are the largest
#   panels (E1 = 796,154 obs) and 200G OOM-killed them at ext1 in job 21525240 — the n_pi*N*R hot
#   buffer (blp_1_estimation.jl:357) grows ~38 GB -> ~51 GB from rc4 to ext1.
#SBATCH --gpus=h200:1
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# (job-name + .out/.err are set per job by the orchestrator via sbatch -J/-o/-e)
#
# ONE RC-BLP stage for one (routine, engine). The orchestrator (submit_blp_rc_all.sh)
# submits these as a --dependency=afterok chain so each stage warm-starts θ₂ from the
# previous stage's checkpoint. Driven by env vars exported by the orchestrator
# (sbatch --export):
#   RC_ROUTINE  1 | 2 | 3 | 4 | 5 | 6 | 7 | 8
#   RC_ENGINE   ift | numerical | cue    → BLP_ENGINE  (cue = continuously-updated GMM, suffix _cue)
#   RC_STAGE    sigma|rc2|rc3|rc4|full|ext1|ext2|extended
# The first stage (sigma) warm-starts δ from data/input/logit_delta_E{k}_spec_12.bin
# on the cluster (data/output kept as a fallback).

set -euo pipefail
: "${RC_ROUTINE:?set RC_ROUTINE (1..8)}"
: "${RC_ENGINE:?set RC_ENGINE (ift|numerical|cue)}"
: "${RC_STAGE:?set RC_STAGE (sigma|rc2|rc3|rc4|full|ext1|ext2|extended)}"
# ${VAR:?} only catches unset/empty, so validate the VALUE too: an unrecognised engine used to fall
# through to IFT with an empty output suffix and overwrite the production artifacts. Fails here in
# seconds rather than after the module load + Julia/CUDA startup (blp_2_rc.jl re-checks as a backstop).
case "${RC_ENGINE}" in
    ift|numerical|cue) ;;
    *) echo "RC_ENGINE='${RC_ENGINE}' is not a recognised engine (ift|numerical|cue)" >&2; exit 2 ;;
esac

# max-inner is a pure SAFETY CEILING, not a tuning knob: SQUAREM self-terminates
# at tol-inner=1e-10 in ~120-185 iters (see logs), so 5000 never actually binds.
# Lowering it would only matter at a pathological trial θ₂ — and there it would
# force an early exit at a NON-converged δ, biasing Q and ∇Q (Dubé–Fox–Su) and
# breaking outer convergence. Keep it high for every stage. The deep stages get
# more wall-time instead (sbatch --time in submit_blp_rc_all.sh); their cost is the
# NUMBER of contractions, which the IFT engine cuts ~8× — not the length of any
# single contraction.

# ── Environment ───────────────────────────────────────────────────────────────
module reset
module load Julia/1.11.4-linux-x86_64           # LLVM 16 → PTX for sm_90 (H200)
export JULIA_MKL_THREADING=tbb
export JULIA_DEPOT_PATH="${SLURM_SUBMIT_DIR}/.julia_depot:${JULIA_DEPOT_PATH:-}"
export BLP_ENGINE="${RC_ENGINE}"                 # read by blp_2_rc.jl

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

echo "GPU node: $(hostname)"
echo "CUDA devices: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Sysimage-aware Julia setup: use blp_sysimage.so if present (and SKIP the expensive
# Pkg.precompile), else resolve/instantiate/precompile. Sets the JULIA_SYS array spliced
# into the julia call below. (Inlined 2026-06-24 — was julia_sysimage_env.sh, now the
# only consumer after the standalone submit scripts were removed.)
SYSIMAGE="${PROJECT_DIR}/blp_sysimage.so"
if [ -f "${SYSIMAGE}" ]; then
    JULIA_SYS=(--sysimage "${SYSIMAGE}")
    echo "Using sysimage ${SYSIMAGE} (skipping precompile): $(date)"
    if ! julia --project="${PROJECT_DIR}" "${JULIA_SYS[@]}" -e 'using CUDA'; then
        echo "[!] Sysimage failed to load CUDA — falling back to no sysimage."
        JULIA_SYS=()
        julia --project="${PROJECT_DIR}" -e 'using Pkg; Pkg.instantiate(); using CUDA'
    fi
else
    JULIA_SYS=()
    # No Pkg.resolve() here: this branch can run in up to 36 concurrent RC-stage jobs
    # (submit_blp_rc_all.sh / submit_blp_rc_grouped.sh) sharing one PROJECT_DIR on NFS.
    # Concurrent resolve() calls race-corrupt Manifest.toml. Build Manifest.toml ONCE via
    # `sbatch setup_julia_env.sh` before submitting any RC chains; only instantiate here.
    echo "No sysimage — instantiating/precompiling Julia packages: $(date)"
    julia --project="${PROJECT_DIR}" -e '
        using Pkg
        Pkg.instantiate()
        Pkg.precompile()
        using CUDA
    '
fi

# Grouped (option-6) jobs pass several stages joined with '+', because sbatch
# --export uses commas to separate variables and so cannot carry a comma list.
# Translate '+' → ',' for the engine; a single stage has no '+', so this is a
# no-op and the per-stage orchestrator is unaffected.
STAGE_ARG="${RC_STAGE//+/,}"

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
echo " RC-BLP E${RC_ROUTINE} | engine=${RC_ENGINE} | stage=${STAGE_ARG} — $(date)"
echo " spec=12 | R=${R} | tol_inner=${TOL_INNER} | max_inner=${MAX_INNER} | threads=${SLURM_CPUS_PER_TASK}"
echo "======================================"

# BLP_SE_ONLY=1 → recompute SEs from each stage's existing checkpoint instead of re-optimising.
# Only valid when the change is confined to the SE routine (it runs after optimisation, so the point
# estimates cannot move). Stages are INDEPENDENT in this mode — no warm-start chain is needed.
SE_ONLY_ARG=()
[ "${BLP_SE_ONLY:-0}" = "1" ] && SE_ONLY_ARG=(--se-only)

julia --project="${PROJECT_DIR}" ${JULIA_SYS[@]+"${JULIA_SYS[@]}"} --threads=${SLURM_CPUS_PER_TASK} \
    "${PROJECT_DIR}/blp_2_rc.jl" \
    --estim "${RC_ROUTINE}" --stage "${STAGE_ARG}" \
    --hpc --R "${R}" --seed "${SEED}" \
    --tol-inner "${TOL_INNER}" --max-inner "${MAX_INNER}" --tol-outer "${TOL_OUTER}" \
    ${SE_ONLY_ARG[@]+"${SE_ONLY_ARG[@]}"}

echo "RC-BLP E${RC_ROUTINE} ${RC_ENGINE} ${RC_STAGE} complete: $(date)"
