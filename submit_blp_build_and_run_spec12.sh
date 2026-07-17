#!/bin/bash
# ONE command: rebuild the sysimage AND regenerate the BLP draws, then run the FULL RC-BLP sweep
# (E5-8, spec 12, IFT) once BOTH are ready. The estimation jobs wait on both builds via SLURM
# --dependency=afterok, so nothing runs against a stale sysimage or stale draws. Mandatory after the
# ESTBAN-instrument / SE-fix / σ-drop source edits (blp_1_estimation.jl, blp_gpu_engine.jl,
# se_common.jl are baked into blp_sysimage.so) and after the demographics/panel changed on the cluster
# (the draws — ν + demo — are rebuilt from the new files by blp_1_draws.jl).
#
#   sysimage build ┐
#                  ├─afterok BOTH→ per-routine grouped chain:  HEAD(sigma..ext1) → ext2 → extended
#   draws build    ┘   (the two builds run in parallel)
#
# Each routine is an INDEPENDENT afterok chain (grouped like submit_blp_rc_grouped.sh: the cheap head
# stages share one process; the deep tail stages are their own resilient jobs). A final afterany CPU
# job zips the outputs + logs into blp_outputs_<jobid>.zip / blp_logs_<jobid>.zip.
#
# PREREQUISITES on the cluster: the edited Julia source uploaded; the demand parquets
# demand_{5..8}_*spec_12.parquet (WITH estban_rival_branches_lag) + logit_delta_E{5..8}_spec_12.bin
# in data/output, plus the new demographics/panel inputs blp_1_draws.jl reads. The engine hard-errors
# if the estban instrument column is missing.
#
# Usage:  bash submit_blp_build_and_run_spec12.sh
#         REBUILD_SYSIMAGE=0 bash submit_blp_build_and_run_spec12.sh   # skip build (already current)
#         BUILD_DRAWS=0 bash submit_blp_build_and_run_spec12.sh        # skip draws (already regenerated)
#         ROUTINES="5 6" bash submit_blp_build_and_run_spec12.sh
#         ENGINES="ift numerical" bash submit_blp_build_and_run_spec12.sh   # also the numerical xcheck
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/logs"

ROUTINES="${ROUTINES:-5 6 7 8}"
ENGINES="${ENGINES:-ift}"                       # IFT only by default (no numerical cross-check)
REBUILD_SYSIMAGE="${REBUILD_SYSIMAGE:-1}"
BUILD_DRAWS="${BUILD_DRAWS:-1}"                  # regenerate the BLP draws (ν + demo) first — needed
                                                # when the demographics/panel changed on the cluster
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
HEAD_STAGES="sigma+rc2+rc3+rc4+full+ext1"       # '+'-joined; submit_blp_rc_stage.sh → comma list

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
    sbatch --parsable --time="$5" ${dep} \
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

do_ift=0; do_num=0
for e in ${ENGINES}; do
    [ "$e" = "ift" ] && do_ift=1
    [ "$e" = "numerical" ] && do_num=1
done

njobs=0
term_jids=""
RUN_MARKER="${DATA_OUT}/.blp_run_marker.$$"
mkdir -p "${DATA_OUT}"; touch "${RUN_MARKER}"
for k in ${ROUTINES}; do
    ift_ext_jid=""
    if [ "${do_ift}" = "1" ]; then
        echo "── grouped chain: E${k} / ift ──"
        ift_ext_jid=$(submit_grouped_chain "${k}" "ift" "ift")
        njobs=$((njobs + 3)); term_jids="${term_jids} ${ift_ext_jid}"
    fi
    if [ "${do_num}" = "1" ] && [ "${do_ift}" = "1" ]; then
        # Numerical cross-check at `extended` only, seeded from the IFT extended θ₂ (afterok IFT extended).
        ckpt="${DATA_OUT}/blp_checkpoint_E${k}_spec_12_extended.jls"
        jid=$(sbatch --parsable --time="${WALL_DEEP}" --dependency=afterok:${ift_ext_jid} \
            --export=ALL,RC_ROUTINE=${k},RC_ENGINE=numerical,RC_STAGE=extended,BLP_THETA2_INIT_FILE=${ckpt} \
            -J "rcg_num_E${k}_xcheck" \
            -o "${HERE}/logs/rcg_num_E${k}_xcheck_%j.out" -e "${HERE}/logs/rcg_num_E${k}_xcheck_%j.err" \
            "${GENERIC}")
        echo "    E${k} numerical xcheck: ${jid}  (afterok ${ift_ext_jid})"
        njobs=$((njobs + 1)); term_jids="${term_jids} ${jid}"
    fi
done
echo "Submitted ${njobs} RC-BLP jobs (routines: ${ROUTINES}; engines: ${ENGINES})${sys_dep:+ + build prereqs [afterok ${sys_dep}]}."

# ── 3. Auto-zip: one short CPU job, afterany ALL chains + the build log. Files are MOVED into the
#      zips (no outside copies) and restricted to THIS run (find -newer ${RUN_MARKER}), so
#      process_blp_outputs.py auto-discovers the newest blp_outputs_*.zip after download.
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
                  rm -f '${RUN_MARKER}' ; \
                  echo \"wrote \${PWD}/blp_outputs_\${SLURM_JOB_ID}.zip + blp_logs_\${SLURM_JOB_ID}.zip\"; }")
    echo "── auto-zip: ${zip_jid}  (afterany ${term_jids# })"
    echo "   → ${DATA_OUT}/blp_outputs_<${zip_jid}>.zip (process_blp_outputs.py auto-discovers blp_outputs_*.zip)"
    echo "   → ${DATA_OUT}/blp_logs_<${zip_jid}>.zip"
fi
