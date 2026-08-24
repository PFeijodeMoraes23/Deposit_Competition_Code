#!/bin/bash
# ONE command: run the FULL RC-BLP sweep (E5-8, spec 12, IFT), optionally preceded by a sysimage
# rebuild and/or a draws regeneration. BOTH build steps are OPT-IN (--sysimage / --draws) because
# the common case — editing our own Julia source — needs NEITHER:
#   * the sysimage bakes only third-party PACKAGES, and our .jl files are include()d at runtime
#     (blp_1_estimation.jl:37), so a source edit takes effect on the next run with no rebuild;
#   * the draws (ν + demo) depend on the demographics/panel and R, not on the instruments or source.
# Rebuild the sysimage when Manifest.toml / package versions change; regenerate the draws when the
# demographics/panel or R change. Whatever IS requested runs first, with the estimation jobs waiting
# via SLURM --dependency=afterok so nothing runs against a stale input.
#
#   [sysimage build] ┐
#                    ├─afterok→ per-routine grouped chain:  HEAD(sigma..ext1) → ext2 → extended
#   [draws build]    ┘   (requested builds run in parallel; with neither, the chains start at once)
#
# Each routine is an INDEPENDENT afterok chain (grouped like submit_blp_rc_grouped.sh: the cheap head
# stages share one process; the deep tail stages are their own resilient jobs). A final afterany CPU
# job zips the outputs + logs into blp_outputs_<jobid>.zip / blp_logs_<jobid>.zip.
#
# PREREQUISITES on the cluster: the edited Julia source uploaded; the demand parquets
# demand_{5..8}_*spec_12.parquet (WITH estban_rival_branches_lag) in data/input + logit_delta_E{5..8}_spec_12.bin
# in data/input (data/output kept as a fallback), plus the new demographics/panel inputs blp_1_draws.jl reads. The engine hard-errors
# if the estban instrument column is missing.
#
# Usage:  bash submit_blp_build_and_run_spec12.sh                  # just run the BLP (the usual case)
#         bash submit_blp_build_and_run_spec12.sh --se-only        # SE-only: recompute SEs from the
#                 existing checkpoints, no re-optimisation (minutes, not hours). Use when ONLY the SE
#                 routine changed — it runs after optimisation so point estimates cannot move.
#         bash submit_blp_build_and_run_spec12.sh --sysimage       # + rebuild the sysimage first
#         bash submit_blp_build_and_run_spec12.sh --draws          # + regenerate the draws first
#         bash submit_blp_build_and_run_spec12.sh --sysimage --draws
#         bash submit_blp_build_and_run_spec12.sh --routines "5 6"
#         bash submit_blp_build_and_run_spec12.sh --engines "ift numerical"   # + numerical xcheck
#         bash submit_blp_build_and_run_spec12.sh --engines "ift cue"         # + CUE variant (_cue
#                 files; the LIML analogue). CUE_STAGES="ext1" by default; CUE_SSET=1 adds the
#                 Stock-Wright S-set grid (SSET_GRID="-1.5:0.125:1.5"). Nothing existing is touched.
#         (env vars REBUILD_SYSIMAGE=1 / BUILD_DRAWS=1 still work; flags win)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/logs"

ROUTINES="${ROUTINES:-3 4}"
ENGINES="${ENGINES:-ift}"                       # IFT only by default (no numerical cross-check, no CUE)

# ── CUE (continuously-updated GMM) — opt in with ENGINES="ift cue" ────────────────────────────
# A side-by-side variant, never a replacement: W(θ)=pinv(Ω̂(θ)) recomputed at every objective
# evaluation (the LIML analogue) instead of the fixed W=(Z'Z/N)⁻¹, with θ₁ by W-weighted GMM iterated
# to a fixed point. Every artifact carries the _cue suffix, so the IFT results — and every downstream
# consumer, which reads exact ift filenames — are untouched. Requires the IFT chain in the same
# invocation: the cue jobs are seeded from its checkpoints.
CUE_STAGES="${CUE_STAGES:-ext1}"                # '+'-separated; ext1 is the reported headline rung
CUE_SSET="${CUE_SSET:-0}"                       # 1 → also submit the Stock-Wright S-set grid job
SSET_STAGE="${SSET_STAGE:-ext1}"
SSET_GRID="${SSET_GRID:--1.5:0.125:1.5}"        # start:step:stop → 25 constrained solves per routine

# ── Per-routine memory ────────────────────────────────────────────────────────────────────────
# E1/E2 are the LARGEST panels (E1 = 796,154 obs vs 607k-714k for E5-E8) and were OOM-killed at the
# ext1 stage in job 21525240 ("Detected 1 oom_kill event"), even though submit_blp_rc_stage.sh asks
# for 200G. Cause: the hot buffer in blp_1_estimation.jl:357 scales as n_pi*N*R, and rc4 -> ext1 takes
# n_pi 3 -> 4, i.e. ~38 GB -> ~51 GB for E1 (plus N*R ~ 12.7 GB and the GPU staging copies). E5-E8
# (~45 GB at ext1) fit; E1/E2 do not. An sbatch COMMAND-LINE --mem overrides the #SBATCH --mem
# directive in the stage script, so the split lives here and the stage script stays single-purpose.
MEM_DEFAULT="${MEM_DEFAULT:-200G}"              # E5-E8 (proven sufficient through `extended`)
# 600G for E1/E2: gpu_h200 nodes carry 2,043,833 MB (~1.95 TiB, measured 2026-08-10 via
# `sinfo -p gpu_h200 -o "%n %m"`), so this is ~30% of one node and schedules freely — the QOS caps
# GPUs, not RAM. 200G demonstrably OOM'd at ext1 while the n_pi*N*R buffer alone is only ~51 GB, so
# the true footprint (WCB B=999 + GPU staging copies) is well above that arithmetic; the generous
# margin is close to free, whereas another OOM costs a multi-hour chain.
MEM_BIG="${MEM_BIG:-600G}"                      # E1/E2
mem_for () { case "$1" in 1|2) printf '%s' "${MEM_BIG}" ;; *) printf '%s' "${MEM_DEFAULT}" ;; esac; }

# Both build steps are OPT-IN, because the COMMON case — editing our own .jl source — needs NEITHER:
#   --sysimage  rebuild blp_sysimage.so. Needed ONLY when Manifest.toml / package versions change.
#               The sysimage bakes ONLY third-party packages (CUDA, Parquet2, DataFrames, Optim,
#               LBFGSB, QuasiMonteCarlo, Distributions, JSON3, ArgParse, SparseArrays) — NOT our
#               source. se_common.jl / blp_1_estimation.jl / blp_gpu_engine.jl are include()d at
#               RUNTIME (blp_1_estimation.jl:37), so source edits take effect with no rebuild.
#   --draws     regenerate the BLP draws (ν + demo). Needed ONLY when the demographics/panel or R
#               change — the draws are independent of the instruments and of our source.
REBUILD_SYSIMAGE="${REBUILD_SYSIMAGE:-0}"
BUILD_DRAWS="${BUILD_DRAWS:-0}"
# --se-only: recompute SEs from the existing per-stage checkpoints instead of re-optimising. Valid
# ONLY when the change is confined to the SE routine (se_common.jl / BLP_SE_METHOD / WCB knobs) —
# it runs after optimisation, so the point estimates provably cannot move. Minutes, not hours.
SE_ONLY="${SE_ONLY:-0}"
while [ $# -gt 0 ]; do
    case "$1" in
        --sysimage|--rebuild-sysimage) REBUILD_SYSIMAGE=1 ;;
        --draws|--redraw)              BUILD_DRAWS=1 ;;
        --no-sysimage)                 REBUILD_SYSIMAGE=0 ;;
        --no-draws)                    BUILD_DRAWS=0 ;;
        --se-only)                     SE_ONLY=1 ;;
        --routines) ROUTINES="$2"; shift ;;
        --engines)  ENGINES="$2";  shift ;;
        -h|--help)  sed -n '2,36p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done
export BLP_SE_ONLY="${SE_ONLY}"                 # read by submit_blp_rc_stage.sh (→ --se-only)
# SE knobs (read by the engine as BLP_SE_METHOD) — same defaults as the other submitters.
SE_METHOD="${SE_METHOD:-wcb}"; WCB_REPS="${WCB_REPS:-999}"; WCB_SCHEME="${WCB_SCHEME:-webb}"
export BLP_SE_METHOD="${SE_METHOD}" BLP_WCB_REPS="${WCB_REPS}" BLP_WCB_SCHEME="${WCB_SCHEME}"
echo "SE method: ${SE_METHOD} (WCB reps=${WCB_REPS}, scheme=${WCB_SCHEME}) — propagated via --export=ALL"

DATA_OUT="${HERE}/../data/output"
GENERIC="${HERE}/submit_blp_rc_stage.sh"
BUILD="${HERE}/submit_build_sysimage.sh"
DRAWS="${HERE}/submit_blp_1_draws.sh"          # blp_1_draws.jl --R 2000 --spec 12 (ν + demo draws)
# gpu_h200 QOS caps wall-per-job at 2 days; each grouped block fits.
WALL_HEAD="${WALL_HEAD:-2-00:00:00}"
WALL_DEEP="${WALL_DEEP:-2-00:00:00}"
# '+'-joined; submit_blp_rc_stage.sh turns it into a comma list. `full` is NOT in the ladder: with
# σ(ln assets) dropped from θ₂ it has the same θ₂ structure as rc4 and reproduces it exactly, so it
# was a redundant rung (ext1 now warm-starts from rc4 — see prev_stages in blp_1_estimation.jl).
HEAD_STAGES="sigma+rc2+rc3+rc4+ext1"
# SE-only runs need no warm-start chain (each stage reloads its own checkpoint), so all stages go
# in ONE short job per routine instead of the HEAD→ext2→extended chain.
ALL_STAGES="sigma+rc2+rc3+rc4+ext1+ext2+extended"
WALL_SE="${WALL_SE:-04:00:00}"

# Non-fatal staleness guard: since the sysimage is now opt-in, catch the one case where skipping it
# is wrong (the package set actually changed) instead of silently running against a stale image.
SYSIMG="${HERE}/blp_sysimage.so"
if [ "${REBUILD_SYSIMAGE}" = "0" ]; then
    if [ ! -f "${SYSIMG}" ]; then
        echo "[!] no blp_sysimage.so — jobs still run, but every job re-precompiles CUDA (slow start). Use --sysimage."
    elif [ -f "${HERE}/Manifest.toml" ] && [ "${HERE}/Manifest.toml" -nt "${SYSIMG}" ]; then
        echo "[!] Manifest.toml is NEWER than blp_sysimage.so — package set may have changed; consider --sysimage."
    else
        echo "── sysimage: reusing $(basename "${SYSIMG}") (source edits do NOT need a rebuild)"
    fi
fi

# ── 1. Build prerequisites (run in PARALLEL; every HEAD stage waits on afterok BOTH) ──
#   (a) rebuild the sysimage so the ESTBAN/SE/σ source edits are compiled into blp_sysimage.so;
#   (b) regenerate the BLP draws (ν + demo) from the new cluster demographics/panel files.
# The draws build (submit_blp_1_draws.sh, plain `julia --project`) is INDEPENDENT of the sysimage,
# so both start immediately; sys_dep accumulates a colon-list consumed as --dependency=afterok:<list>.
sys_dep=""
if [ "${REBUILD_SYSIMAGE}" = "1" ]; then
    sys_jid=$(sbatch --parsable \
        -o "${HERE}/logs/blp_build_sysimg_%j.out" -e "${HERE}/logs/blp_build_sysimg_%j.err" \
        "${BUILD}")
    sys_dep="${sys_jid}"
    echo "── sysimage build: ${sys_jid}  (all HEAD stages afterok this)"
fi
if [ "${BUILD_DRAWS}" = "1" ]; then
    draws_jid=$(sbatch --parsable \
        -o "${HERE}/logs/blp_1_draws_%j.out" -e "${HERE}/logs/blp_1_draws_%j.err" \
        "${DRAWS}")
    sys_dep="${sys_dep:+${sys_dep}:}${draws_jid}"
    echo "── draws build:    ${draws_jid}  (all HEAD stages afterok this; runs parallel to the sysimage)"
fi

# ── 2. Per-(routine,engine) grouped afterok chain: HEAD → ext2 → extended ──
submit_one () {   # $1=routine $2=engine $3=tag $4=stage $5=wall $6=jobtag [$7=dep_jobid]
    local dep=""
    [ -n "${7:-}" ] && dep="--dependency=afterok:$7"
    sbatch --parsable --time="$5" --mem="$(mem_for "$1")" ${dep} \
        --export=ALL,RC_ROUTINE=$1,RC_ENGINE=$2,RC_STAGE=$4 \
        -J "rcg_$3_E$1_$6" \
        -o "${HERE}/logs/rcg_$3_E$1_$6_%j.out" -e "${HERE}/logs/rcg_$3_E$1_$6_%j.err" \
        "${GENERIC}"
}
submit_grouped_chain () {   # $1=routine $2=engine $3=tag  → echoes the terminal (extended) job id
    local k="$1" eng="$2" tag="$3"
    local jh je2 je
    jh=$(submit_one "${k}" "${eng}" "${tag}" "${HEAD_STAGES}" "${WALL_HEAD}" "head" "${sys_dep}")
    je2=$(submit_one "${k}" "${eng}" "${tag}" "ext2"     "${WALL_DEEP}" "ext2" "${jh}")
    je=$(submit_one  "${k}" "${eng}" "${tag}" "extended" "${WALL_DEEP}" "ext"  "${je2}")
    echo "    E${k}/${tag}: head ${jh}${sys_dep:+ (afterok ${sys_dep})} → ext2 ${je2} → extended ${je}" >&2
    echo "${je}"
}

do_ift=0; do_num=0; do_cue=0
for e in ${ENGINES}; do
    case "$e" in
        ift)       do_ift=1 ;;
        numerical) do_num=1 ;;
        cue)       do_cue=1 ;;
        *) echo "[!] unknown engine token '$e' in ENGINES — ignored (valid: ift numerical cue)" >&2 ;;
    esac
done

njobs=0
term_jids=""
RUN_MARKER="${DATA_OUT}/.blp_run_marker.$$"
mkdir -p "${DATA_OUT}"; touch "${RUN_MARKER}"
for k in ${ROUTINES}; do
    ift_ext_jid=""
    if [ "${SE_ONLY}" = "1" ]; then
        # No homotopy chain: every stage reloads its own checkpoint, so one job does them all.
        jid=$(submit_one "${k}" "ift" "se" "${ALL_STAGES}" "${WALL_SE}" "seonly" "${sys_dep}")
        echo "    E${k}: SE-only — all stages in ONE job: ${jid}"
        njobs=$((njobs + 1)); term_jids="${term_jids} ${jid}"
        continue
    fi
    if [ "${do_ift}" = "1" ]; then
        echo "── grouped chain: E${k} / ift ──"
        ift_ext_jid=$(submit_grouped_chain "${k}" "ift" "ift")
        njobs=$((njobs + 3)); term_jids="${term_jids} ${ift_ext_jid}"
    fi
    if [ "${do_num}" = "1" ] && [ "${do_ift}" = "1" ]; then
        # Numerical cross-check at `extended` only, seeded from the IFT extended θ₂ (afterok IFT extended).
        ckpt="${DATA_OUT}/blp_checkpoint_E${k}_spec_12_extended.jls"
        jid=$(sbatch --parsable --time="${WALL_DEEP}" --mem="$(mem_for "${k}")" --dependency=afterok:${ift_ext_jid} \
            --export=ALL,RC_ROUTINE=${k},RC_ENGINE=numerical,RC_STAGE=extended,BLP_THETA2_INIT_FILE=${ckpt} \
            -J "rcg_num_E${k}_xcheck" \
            -o "${HERE}/logs/rcg_num_E${k}_xcheck_%j.out" -e "${HERE}/logs/rcg_num_E${k}_xcheck_%j.err" \
            "${GENERIC}")
        echo "    E${k} numerical xcheck: ${jid}  (afterok ${ift_ext_jid})"
        njobs=$((njobs + 1)); term_jids="${term_jids} ${jid}"
    fi
    if [ "${do_cue}" = "1" ] && [ "${do_ift}" = "1" ]; then
        # ── CUE (continuously-updated GMM): the LIML analogue, a SIDE-BY-SIDE variant ──────────
        # Every artifact carries the _cue suffix (blp_2_rc.jl ENGINE_SUFFIX), so nothing the IFT
        # chain wrote is touched and every downstream consumer — which reads EXACT ift filenames —
        # ignores it. One job PER STAGE (not a grouped '+' list): a grouped job would seed every
        # stage from the same BLP_THETA2_INIT_FILE, and the _cue prev-stage checkpoint chain does
        # not exist. Each is seeded from the IFT checkpoint of the SAME stage.
        for st in $(echo "${CUE_STAGES}" | tr '+' ' '); do
            ckpt="${DATA_OUT}/blp_checkpoint_E${k}_spec_12_${st}.jls"
            jid=$(sbatch --parsable --time="${WALL_DEEP}" --mem="$(mem_for "${k}")" --dependency=afterok:${ift_ext_jid} \
                --export=ALL,RC_ROUTINE=${k},RC_ENGINE=cue,RC_STAGE=${st},BLP_THETA2_INIT_FILE=${ckpt} \
                -J "rcg_cue_E${k}_${st}" \
                -o "${HERE}/logs/rcg_cue_E${k}_${st}_%j.out" -e "${HERE}/logs/rcg_cue_E${k}_${st}_%j.err" \
                "${GENERIC}")
            echo "    E${k} CUE ${st}: ${jid}  (afterok ${ift_ext_jid}; seed $(basename "${ckpt}"))"
            njobs=$((njobs + 1)); term_jids="${term_jids} ${jid}"
        done
        if [ "${CUE_SSET}" = "1" ]; then
            # Stock-Wright S-set: S(α₀)=min_{θ₂,β} N·ḡ'W(θ)ḡ on a grid of pinned α₀, inverted into
            # an identification-robust set that ACCOUNTS for θ₂ being estimated (unlike the
            # conditional AR/LM sets). One constrained solve per grid point, warm-started along the
            # grid; writes ONE blp_results_E{k}_spec_12_{stage}_sset_cue.json.
            ckpt="${DATA_OUT}/blp_checkpoint_E${k}_spec_12_${SSET_STAGE}.jls"
            jid=$(sbatch --parsable --time="${WALL_DEEP}" --mem="$(mem_for "${k}")" --dependency=afterok:${ift_ext_jid} \
                --export=ALL,RC_ROUTINE=${k},RC_ENGINE=cue,RC_STAGE=${SSET_STAGE},BLP_THETA2_INIT_FILE=${ckpt},BLP_ALPHA_GRID=${SSET_GRID} \
                -J "rcg_sset_E${k}" \
                -o "${HERE}/logs/rcg_sset_E${k}_%j.out" -e "${HERE}/logs/rcg_sset_E${k}_%j.err" \
                "${GENERIC}")
            echo "    E${k} S-set ${SSET_STAGE} [${SSET_GRID}]: ${jid}  (afterok ${ift_ext_jid})"
            njobs=$((njobs + 1)); term_jids="${term_jids} ${jid}"
        fi
    fi
done
echo "Submitted ${njobs} RC-BLP jobs (routines: ${ROUTINES}; engines: ${ENGINES})${sys_dep:+ + build prereqs [afterok ${sys_dep}]}."

# ── 3. Auto-zip: one short CPU job, afterany ALL chains + the build log. Files are MOVED into THREE
#      zips (no outside copies), restricted to THIS run (find -newer ${RUN_MARKER}):
#        blp_outputs_<jid>.zip      summaries + per-stage result .jls + logit (process_blp_outputs.py finds it)
#        blp_logs_<jid>.zip         the rcg/sysimg/draws .out + .err
#        blp_checkpoints_<jid>.zip  the warm-start blp_checkpoint_*.jls (~12 MB each) — MOVED off data/output
#                                   so they stop accumulating; only needed for --se-only / resuming a stage.
ZIP_PARTITION="${ZIP_PARTITION:-day}"
dep_csv="$(echo ${term_jids} | tr ' ' ':' | sed 's/^://; s/:$//')"
if [ -n "${dep_csv}" ]; then
    zip_jid=$(sbatch --parsable --dependency=afterany:${dep_csv} \
        --job-name=blp_zip --partition="${ZIP_PARTITION}" --time=00:20:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        -o "${HERE}/logs/blp_zip_%j.out" -e "${HERE}/logs/blp_zip_%j.err" \
        --wrap "cd '${DATA_OUT}' && { \
                  find . -maxdepth 1 -newer '${RUN_MARKER}' \\( -name 'blp_results_E*_spec_12_*.json' -o -name 'blp_results_E*_spec_12_*.jls' -o -name 'blp_summary_E*_gpu_*.json' -o -name 'logit_*' \\) -print0 | xargs -0 -r zip -jm \"blp_outputs_\${SLURM_JOB_ID}.zip\" ; \
                  find '${HERE}'/logs -maxdepth 1 -newer '${RUN_MARKER}' \\( -name 'rcg_*.out' -o -name 'rcg_*.err' -o -name 'blp_build_sysimg_*.out' -o -name 'blp_build_sysimg_*.err' -o -name 'blp_1_draws_*.out' -o -name 'blp_1_draws_*.err' \\) -print0 | xargs -0 -r zip -jm \"blp_logs_\${SLURM_JOB_ID}.zip\" ; \
                  find . -maxdepth 1 -newer '${RUN_MARKER}' -name 'blp_checkpoint_E*_spec_*_*.jls' -print0 | xargs -0 -r zip -jm \"blp_checkpoints_\${SLURM_JOB_ID}.zip\" ; \
                  rm -f '${RUN_MARKER}' ; \
                  echo \"wrote \${PWD}/blp_outputs + blp_logs + blp_checkpoints, job \${SLURM_JOB_ID}\"; }")
    echo "── auto-zip: ${zip_jid}  (afterany ${term_jids# })"
    echo "   → ${DATA_OUT}/blp_outputs_<${zip_jid}>.zip (process_blp_outputs.py auto-discovers blp_outputs_*.zip)"
    echo "   → ${DATA_OUT}/blp_logs_<${zip_jid}>.zip"
    echo "   → ${DATA_OUT}/blp_checkpoints_<${zip_jid}>.zip  (warm-start .jls moved off data/output; needed only for --se-only / resuming a stage — unzip first if so)"
fi
