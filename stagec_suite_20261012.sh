#!/bin/bash
# ==============================================================================
# stagec_suite_20261012.sh -- stage C on a FRESH Bouchet tree: the launch (guards, install, the
# shipped Julia package set), ONE CPU chain job (prep, logit, draws and their gates), ONE GPU job
# (the GPU sysimage, the one-process smoke, then the BLP hand-off), ONE CPU-image job beside them
# (the CPU sysimage the BBL's CPU steps load), the RC ladders and the BBL on that fit. The fixed demand side (demand_chars) with the D-type dummy in the MAIN logit and RC-BLP.
# No counterfactuals. Every SLURM request lives in stagec_requests.env (REQ_*).
#
# BEFORE IT (login node, the documented steps; each one line):
#   1. upload stagec_code_<stamp>.zip, stagec_data_<stamp>.zip and stagec_upload_<stamp>.md5 into
#      ~/project_pi_mf2263/pf382/dep_comp (the project root)
#   2. cd ~/project_pi_mf2263/pf382/dep_comp && md5sum -c stagec_upload_<stamp>.md5 && unzip -o stagec_code_<stamp>.zip && unzip -o stagec_data_<stamp>.zip && mkdir -p scripts/logs && chmod +x scripts/*.sh
#   3. ONLY IF the compute nodes have no internet (variant ii; the probes decide): fill the depot
#      from the login node, downloads only (packages, artifacts, the lazy CUDA runtime/compiler
#      artifacts of CUDA 12.x and 13.x), in the background (it does not hold the terminal):
#      cd ~/project_pi_mf2263/pf382/dep_comp/scripts && mkdir -p logs && nohup bash stagec_suite_20261012.sh --julia-fetch > logs/stagec_julia_fetch.log 2>&1 &
#      check (repeat until it prints STAGE C JULIA FETCH DONE):
#      tail -n 40 ~/project_pi_mf2263/pf382/dep_comp/scripts/logs/stagec_julia_fetch.log
#
# THE LAUNCH (one line; it prints the job id and returns; the request is REQ_LAUNCH_*, no cpugen pin):
#   cd ~/project_pi_mf2263/pf382/dep_comp/scripts && mkdir -p logs && sbatch --parsable -p day -t 01:00:00 -c 4 --mem=16G -J stagec_launch -o logs/stagec_launch_%j.out --wrap "bash stagec_suite_20261012.sh"
# Variant (ii): the same line with --export=ALL,STAGEC_JULIA_OFFLINE=1 after --parsable. A dry run:
# --export=ALL,CL_DRYRUN=1 (nothing is copied, written or submitted). Then, each one line:
#   squeue -u $USER -h -o "%.10i %.28j %.9T %.22r %.10M"
#   tail -n 40 ~/project_pi_mf2263/pf382/dep_comp/scripts/logs/stagec_launch_<job id>.out
#
# THE LAUNCH JOB (T1)
#   L0 THE ENVIRONMENT (D12): every job inherits the submitting shell, so a variable that changes a
#      result is refused (SLEEP_STATE_BLOCK, any BLP_*, SE_*/WCB_*/TOL_*, the CF_*/BBL design
#      variables, R != 2000, SEED != 42, a DATA_ROOT/CL_ROOT other than this tree, the test knobs,
#      and at the launch whatever the chain sets itself); the placement knobs are printed.
#   L1 THE ARRIVAL: sha256sum -c of every uploaded file, code and data, Manifest-v1.11.toml included
#      (stagec_arrival_*.sha256). Later jobs re-check the code with both manifests exempt; every job
#      that runs Julia refuses if Manifest-v1.11.toml no longer hashes to the shipped sha256.
#   L2 the queue, fail-closed: refuses while any job of this project is live.
#   L3 THE OUTPUT GUARD (M3): refuses if data/output already holds any result of this chain (demand
#      parquets, logit outputs, ANY file in data/output/blp, draws, BBL files of _ms983 / _sc983, a
#      stage C gate, the ladder marker); and if data/input holds a logit_delta_* (m5).
#   L4 THE PYTHON ENV (MAJOR-1): CONDA_ENV (dep_comp_blp unless exported) imports the stack the
#      sleepiness estimators and the prep need; else it prints the pinned create line.
#   L5 a dry run of the submissions.
#   L6 THE INSTALL (P8): the policy (sha256 fd208b5f...) copied from data/input, re-hashed, never
#      overwritten; the frozen-set record (policy, panel, chars, sigma).
#   L7 THE JULIA ENVIRONMENT (P8): the shipped Manifest-v1.11.toml instantiated, NEVER resolved, no
#      precompile (pkgimages are CPU-specific: each node precompiles for itself); refuses if it changed.
#   L8 submits T4 (the CPU image, no dependency), then T3 and T2 (afterok T3), which inherit the
#      prep exports (DEMAND_CHARS_FILE, SLEEP_PREP_EXTRA, SLEEP_ACTIVE_ESTS) as every chain job did on
#      10-13 (T4 does not: its 10-13 job had none). A failure in L6-L7 submits nothing: launch again,
#      same line.
# T3 stagec_chain (REQ_CHAIN_*: day, 16c/128G/6h, cpugen:turin; the same CPU as T4, so the two never
#   precompile the shared depot for different CPUs), ONE job, steps logged apart:
#   step 0  precompile for this node, every package loads; gate G0 (env_job.sh preflight; both
#           images are excused: T2 builds the GPU image, T4 the CPU image)
#   sleep   THE SLEEPINESS ESTIMATES E1-E4 on the cluster (Pedro, 10-15): sleep_job.sh SLEEP_STEP=est
#           per routine, one after the other, spec 12 (--spec12 for E2-E4; E1 has no spec selector)
#   gate-sleep  automatic: every routine's pickle loads and holds the IV_RivalSpread x Tech estimate
#           with its second stage; phi_t and phi_mt finite and in [0,1], rows > 0. Then the
#           BOOTSTRAPS are submitted as separate jobs, off the critical path (REQ_BOOT_*,
#           REQ_WCB/AMEG/AME_*): E1/E2 percentile bands, the E3/E4 AME guard and AME bootstraps,
#           and their own archive (set sleep). Nothing waits on them.
#   prep    sleep_job.sh SLEEP_STEP=prep (spec 12, step 6 only, DEMAND_CHARS_FILE), reading the
#           estimates just made -> gate G2
#   gate-prep  automatic: one parquet per routine, rows > 0, the instrument columns, .chars_audit.json
#   logit   logit_job.sh || blp_draws_job.sh (8 Julia threads each, as their own jobs had) -> gate G4
#   gate-logit automatic: every sub-model's alpha finite, D in the full design; then THE SUMMARY
#           BLOCK per routine (first-stage effective F, second-stage rows, mean phi_t, logit alpha
#           with SE and rows), which Pedro reads while the BLP runs. No local reference values.
#   archive demand_prep + logit: ALWAYS (m1), then the job exits with the first failing step's code,
#           so T2 never starts after a stop.
# T2 stagec_gpu (REQ_GPU_*: gpu_h200, 1 GPU, 8c/360G/2h), afterok T3:
#   the GPU sysimage (built here when missing or invalid; env_job.sh build + CUDA.functional() load
#   proof), then ONE Julia launch that loads the image and proves CUDA on the device(s) and reads the
#   CPU target (cl_require_sysimage_gated: the sysimage probe merged into the CUDA gate), then THE
#   SMOKE in ONE Julia process over E1-E4 (blp_smoke.jl; memory released between routines): FAILS
#   unless every stage ran, the new delta loaded, and 100% of the observations mapped to the rebuilt
#   draws (THE DRAWS GATE); then, in the same job, THE BLP HAND-OFF: M3 again, the ladder marker,
#   blp_run.sh "3 4", stagec_next_bbl (afterok the E3/E4 terminals), blp_run.sh "1 2" (a refusal of
#   E1/E2 does not strand the BBL, m3), stagec_blp_zip afterany every ladder + the hand-off: the blp
#   set when E1-E4 are whole, else the E3/E4 archive _E34 (D8/m16).
#   ext1 and extended run two starts in the engine (v2 records: Q_opt, Q_reeval, reeval_converged,
#   choice_basis). A head job that fails (a TwoStartError) cancels its later stages and the BBL
#   hand-off with no log; stagec_blp_zip's log then names the missing results and prints
#   --resume blp, which resubmits the unfinished ladders only (only RC files newer than the ladder
#   marker .stagec_ladders_20261012 are accepted).
# T4 stagec_cpu_image (REQ_CPUIMG_*: day, 8c/32G/1h, cpugen:turin), beside T3 (no dependency):
#   blp_sysimage_cpu.so built when missing or invalid (env_job.sh ENV_STEP=sysimage_cpu: build + load
#   proof), then loaded once more here (cl_require_sysimage cpu: provenance + CPU target), then the
#   gate .gate_SC_cpu_image.json. Only the BBL needs it (bbl_job.sh loads it in every CPU solve job).
#   A failed sanity check STOPS the chain before any GPU job. Every stop prints its
#   one-line resume (m10): --resume chain [--from sleep|gate-sleep|prep|gate-prep|logit|gate-logit] | gpu | next-blp |
#   blp | next-bbl [tag] | cpu-image; --export=ALL,STAGEC_ALLOW_MISMATCH_STEP=gate-prep (or gate-logit) with
#   --resume chain --from that step accepts a mismatch at that step only.
# THE BBL HAND-OFF (stagec_next_bbl)
#   the CPU image must have passed (.gate_SC_cpu_image.json ok, and the image passes its static
#   provenance check here: the solve jobs refuse without it), E3/E4 ext1 must say se_status "ok" (joint SEs REQUIRED), the policy sha256, fresh launches only,
#   fit records first, dry runs of both launches; then _ms983 (multi-start, --promote: bbl_solve.py
#   promotes only past its identification gate; a failed gate exits 3) and AT ONCE its archive
#   (afterany its solves, --check-bbl first; m2/D10); then _sc983 (single curve, no promotion) and
#   its archive; then stagec_final. Both tags --no-tables (D3): the cost tables are rendered locally.
#   A sweep bbl_run.sh could not release is printed with its scontrol line (m8). A launch bbl_run.sh
#   refused leaves its dispatch folders; they are moved aside so the printed relaunch works (D3f).
#
# NEVER: blp_dx.jl / dx_suite_20261001.sh (superseded); a hand-run `bbl_run.sh ... --repair` for
#   _ms983 / _sc983 (it skips the fit binding); Pkg.resolve on the cluster (env_job.sh ENV_STEP=resolve).
# Modes: (none) launch | --chain [--from <step>] | --gpu | --cpu-image | --install | --julia-env | --julia-fetch
#        (login node) | --next-blp | --check-blp [routines] | --next-bbl [tag] | --check-bbl <tag> |
#        --repair-bbl <tag> | --final | --collect-logs | --resume <point> [--from <step>]
#        (--smoke-verdict <k> <log>: the smoke's per-routine verdict on a log, for the local test)
# Environment: CL_DRYRUN=1; STAGEC_ALLOW_MISMATCH=1 (every gate of a launch);
#   STAGEC_ALLOW_MISMATCH_STEP=gate-prep|gate-logit (one step of a resumed chain);
#   STAGEC_JULIA_OFFLINE=1 (variant ii); STAGEC_BBL_EXTRA / STAGEC_BLP_EXTRA (extra flags; the local
#   test passes --skip-preflight); STAGEC_LOG_DIR; STAGEC_CUDA_MAJORS (--julia-fetch, default "12 13");
#   STAGEC_LOCAL_TEST=1 with STAGEC_ASSUME_SYSIMAGE=1 / STAGEC_SKIP_PYPROBE=1 / STAGEC_PAYLOAD_RUNNER /
#   STAGEC_PAYLOAD_REAL
#   (the local test only).
# ==============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}" || exit 2
# What the submitting shell set, BEFORE cluster_lib.sh and this script set anything (env_check,
# D12): the names of the exported variables this job started with, and the two roots.
SC_ENV0=" $(compgen -e | tr '\n' ' ') "
SC_USER_DATA_ROOT="${DATA_ROOT-}"; SC_USER_CL_ROOT="${CL_ROOT-}"
CL_VERBOSE=1
. ./cluster_lib.sh

SELF="stagec_suite_20261012.sh"
PINS_FILE="${HERE}/stagec_pins_20261012.env"
MODE=launch; MODE_ARG=""; CHAIN_FROM=""; RESUME_FROM=""
case "${1:-}" in
    "") ;;
    --install|--gpu|--cpu-image|--next-blp|--final|--julia-env|--julia-fetch|--collect-logs) MODE="${1#--}" ;;
    --chain) MODE=chain; [[ "${2:-}" == "--from" ]] && CHAIN_FROM="${3:-}" ;;
    --next-bbl) MODE=next-bbl; MODE_ARG="${2:-}" ;;   # optional: one tag alone (a launch that failed)
    --check-blp) MODE=check-blp; MODE_ARG="${2:-}" ;;   # optional routines, e.g. "3 4" (default: all four)
    --check-bbl) MODE=check-bbl; MODE_ARG="${2:-}" ;;
    --resume) MODE=resume; MODE_ARG="${2:-}"; [[ "${3:-}" == "--from" ]] && RESUME_FROM="${4:-}" ;;
    --smoke-verdict) MODE=smoke-verdict; MODE_ARG="${2:-} ${3:-}" ;;   # the local test only
    --repair-bbl) MODE=repair-bbl; MODE_ARG="${2:-}" ;;
    *) echo "unknown option: $1 (see the header)" >&2; exit 2 ;;
esac

# Every SLURM request of this run, one place (REQ_*; a resize is a one-line change there).
REQ_FILE="${HERE}/stagec_requests.env"
[[ -f "${REQ_FILE}" ]] || { echo "STAGE C REFUSED: ${REQ_FILE} is missing (it ships in the code zip with this script)." >&2; exit 1; }
# shellcheck source=/dev/null
. "${REQ_FILE}"

ROUTINES_ALL="1 2 3 4"; BBL_ROUTINES="3 4"
MS_TAG="_ms983"; SC_TAG="_sc983"; BBL_STAGE="extended"; SE_STAGE="ext1"
POLICY_SHA256="fd208b5fc57b9697eefc010ed4a1076af776d9510b89f3bbc1edca009dc29728"
POLICY_CSV="${CL_STEP_BBL}/polfunc_fitted.csv"
FROZEN_REC="${CL_DATA_OUT}/.stagec_frozen_20261012.sha256"
SMOKE_STAGES="sigma,ext1,extended"
DESIGN=(--fwd-gpu --partitions gpu_h100 --shards 300 --beta 0.979 --horizon 250
        --phi-path evolving --z-path mean_reverting --rdep-timing lagged --no-warmup --no-polfunc
        --pack-h100 "${REQ_BBL_PACK_H100}" --mem-h100 "${REQ_BBL_MEM_H100}" --shard-time "${REQ_BBL_SHARD_TIME}" --no-zip)
read -r -a BBL_EXTRA <<< "${STAGEC_BBL_EXTRA:-}"
read -r -a BLP_EXTRA <<< "${STAGEC_BLP_EXTRA:-}"   # the local test passes --skip-preflight
# Both tags --no-tables (user, 10-13): the cost tables are rendered locally from the archived
# cost_params; no tables job can race the archive. --promote keeps bbl_solve's identification gate.
MS_FLAGS=(--routines "${BBL_ROUTINES}" "${DESIGN[@]}" --multi-start --n-paths 1 --psi-tag "${MS_TAG}" --promote --no-tables ${BBL_EXTRA[@]+"${BBL_EXTRA[@]}"})
SC_FLAGS=(--routines "${BBL_ROUTINES}" "${DESIGN[@]}" --psi-tag "${SC_TAG}" --no-tables ${BBL_EXTRA[@]+"${BBL_EXTRA[@]}"})
JL_MANIFEST="Manifest-v1.11.toml"   # resolved locally under Julia 1.11.4 and shipped; never re-resolved here
DRY="${CL_DRYRUN:-0}"
LOGD="${STAGEC_LOG_DIR:-$(cl_log_dir)}"
STAMP="$(date +%Y%m%d_%H%M%S)"
PROJ="$(cd "${CL_ROOT}/.." && pwd)"
R="${R:-2000}"; SEED="${SEED:-42}"; export R SEED

# Variant (ii) (no internet on compute nodes): every job of the chain runs Pkg offline.
if [[ "${STAGEC_JULIA_OFFLINE:-0}" == "1" ]]; then export JULIA_PKG_OFFLINE=true; fi
say () { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
# STOP_RESUME: the one line that continues this job after a refusal (set by the launch and by the
# chain, GPU and CPU-image jobs; a message that already carries a line of this script keeps its own).
# STOP_GATE: the gate a refusal writes false (T4's, so the BBL hand-off reads why it did not pass).
STOP_RESUME=""; STOP_GATE=""
die () {
    printf '%s  STAGE C REFUSED (%s): %s\n' "$(date '+%F %T')" "${MODE}" "$*"
    [[ -n "${STOP_GATE}" ]] && gate_json "${STOP_GATE}" false "$*"
    [[ -n "${STOP_RESUME}" && "$*" != *"--wrap \"bash ${SELF:-stagec_suite_20261012.sh}"* ]] && printf '   THE ONE LINE TO CONTINUE, once the cause is fixed: %s\n' "${STOP_RESUME}"
    exit 1
}
sha () { sha256sum "$1" | cut -d' ' -f1; }
is_sha256 () { [[ "$1" =~ ^[0-9a-f]{64}$ ]]; }
json_str () {   # json_str <file> <field>: the first string value of <field>
    grep -o "\"$2\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$1" 2>/dev/null | head -1 | sed 's/^"[^"]*"[[:space:]]*:[[:space:]]*"//; s/"$//'
}
gate_json () {   # gate_json <name> <true|false> <detail>: data/output/.gate_SC_<name>.json (the gates set)
    [[ "${DRY}" == "1" ]] && return 0
    mkdir -p "${CL_DATA_OUT}"
    printf '{\n "gate": "SC_%s",\n "ok": %s,\n "detail": "%s",\n "slurm_job_id": "%s",\n "written": "%s"\n}\n' \
        "$1" "$2" "$(printf '%s' "$3" | tr '"\n' "' ")" "${SLURM_JOB_ID:-}" "$(date '+%F %T')" > "${CL_DATA_OUT}/.gate_SC_$1.json"
}
# resume_line "<point> [--from <step>]" [<gate step>]: the ONE line that continues from <point>
# (m10). With a gate step (gate-prep | gate-logit), the line accepts a row/alpha mismatch at that
# step of the resumed chain only (STAGEC_ALLOW_MISMATCH_STEP, logged loudly). The line returns at
# once (sbatch --parsable); the request is REQ_HANDOFF_*.
launch_line () {   # THE launch line (the header documents the same one; the tests compare them)
    printf 'cd %s && mkdir -p logs && sbatch --parsable%s -p %s -t %s -c %s --mem=%s%s -J stagec_launch -o logs/stagec_launch_%%j.out --wrap "bash %s"' \
        "${HERE}" "$([[ "${STAGEC_JULIA_OFFLINE:-0}" == "1" ]] && echo " --export=ALL,STAGEC_JULIA_OFFLINE=1")" \
        "${REQ_LAUNCH_PART}" "${REQ_LAUNCH_TIME}" "${REQ_LAUNCH_CPUS}" "${REQ_LAUNCH_MEM}" "${REQ_LAUNCH_PIN:+ --constraint=${REQ_LAUNCH_PIN}}" "${SELF}"
}
resume_line () {
    printf 'cd %s && sbatch --parsable -p %s -t %s -c %s --mem=%s -J stagec_resume -o logs/stagec_resume_%%j.out%s --wrap "bash %s --resume %s"' \
        "${HERE}" "${REQ_HANDOFF_PART}" "${REQ_HANDOFF_TIME}" "${REQ_HANDOFF_CPUS}" "${REQ_HANDOFF_MEM}" \
        "$([[ -n "${2:-}" ]] && echo " --export=ALL,STAGEC_ALLOW_MISMATCH_STEP=$2")" "${SELF}" "$1"
}

# ── D12: the environment. Every job inherits the submitting shell (sbatch --export=ALL), so a
# variable that changes a result is REFUSED (unset it and submit again); the placement knobs
# are printed. env_check launch also refuses what the chain itself sets (it must start clean).
SC_REFUSE="SLEEP_STATE_BLOCK SLEEP_CENTER_STATES SLEEP_BOOT_B SLEEP_BOOT_SCHEME USE_FEE_PANEL
  BLP_ZERO_FILL_MISSING BLP_ALLOW_FILL BLP_TWO_START_ALLOW_SINGLE SE_METHOD SE_ONLY WCB_REPS WCB_SCHEME TOL_INNER TOL_OUTER MAX_INNER
  CF_STAGE CF_EVOLVING_STATES CF_PHI_PATH CF_Z_PATH CF_RDEP_TIMING CF_STATE_RHO_FIELD ASSET_MARGIN ASSET_RETURN_COL
  SHOCKS PERTURB_SCALE DEV_SCHEME N_SHARDS BETA HORIZON PSI_TAG MULTI_START N_PATHS PROMOTE BBL_WINSOR_PCT
  BBL_WINSOR_Z BBL_DISCOUNT_ENV ALLOW_SYSIMAGE_FALLBACK ALLOW_UNSTAMPED_SYSIMAGE ALLOW_RF_PAD"
SC_REFUSE_AT_LAUNCH="DEMAND_CHARS_FILE SLEEP_PREP_EXTRA SLEEP_ACTIVE_ESTS DEMAND_PREP_DIR SLEEP_OUT_ROOT CF_COST_FWD
  COST_POLFUNC_DIR CF_FOUNDATION_DIR STATE_CENTERING_JSON OPEN_FINANCE_ROOT POLICY_CSV"
SC_PRINT="PARTITION MEM CL_MEM_DEFAULT CL_MEM_BIG CL_WALL_DEEP ZIP_PARTITION SLEEP_PARTITION CONDA_ENV PY_MODULE
  JULIA_MODULE STAGEC_SLEEP_EXTRA STAGEC_BLP_EXTRA STAGEC_BBL_EXTRA STAGEC_LOG_DIR STAGEC_ALLOW_MISMATCH
  STAGEC_ALLOW_MISMATCH_STEP STAGEC_JULIA_OFFLINE"
same_dir () { [[ "$(cd "$1" 2>/dev/null && pwd -P)" == "$(cd "$2" 2>/dev/null && pwd -P)" ]]; }
in_env0 () { [[ "${SC_ENV0}" == *" $1 "* ]]; }   # exported when this job started (not set by this script)
env_check () {   # env_check launch|chain
    local v bad="" shown=""
    for v in ${SC_REFUSE} $([[ "$1" == "launch" ]] && echo "${SC_REFUSE_AT_LAUNCH}"); do
        in_env0 "${v}" && bad="${bad} ${v}='${!v-}'"
    done
    # every BLP_* steers the engine (BLP_TWO_START_ALLOW_SINGLE would silently drop a start of ext1/extended)
    for v in ${SC_ENV0}; do [[ "${v}" == BLP_* ]] && [[ " ${SC_REFUSE} " != *" ${v} "* ]] && bad="${bad} ${v}='${!v-}'"; done
    ! in_env0 R || [[ "${R}" == "2000" ]] || bad="${bad} R='${R}' (the run is R=2000)"
    ! in_env0 SEED || [[ "${SEED}" == "42" ]] || bad="${bad} SEED='${SEED}' (the run is SEED=42)"
    [[ -z "${SC_USER_DATA_ROOT}" ]] || same_dir "${SC_USER_DATA_ROOT}" "${PROJ}/data" || bad="${bad} DATA_ROOT='${SC_USER_DATA_ROOT}' (not ${PROJ}/data)"
    [[ -z "${SC_USER_CL_ROOT}" ]] || same_dir "${SC_USER_CL_ROOT}" "${HERE}" || bad="${bad} CL_ROOT='${SC_USER_CL_ROOT}' (not ${HERE})"
    if [[ "${STAGEC_LOCAL_TEST:-0}" != "1" ]]; then
        for v in STAGEC_ASSUME_SYSIMAGE STAGEC_SKIP_PYPROBE STAGEC_PAYLOAD_RUNNER STAGEC_PAYLOAD_REAL; do in_env0 "${v}" && bad="${bad} ${v}='${!v-}' (local tests only)"; done
    fi
    [[ -z "${bad}" ]] || die "the environment would change results:${bad}. Every job inherits it. Unset them (unset NAME ...) and submit again; nothing was submitted."
    for v in ${SC_PRINT}; do in_env0 "${v}" && shown="${shown} ${v}='${!v-}'"; done
    say "   environment: no result-changing variable set; placement/knobs set:${shown:- none}"
}

# ── the pin file (written by build_stagec_bundle.py at build time) ────────────────────────
[[ -f "${PINS_FILE}" ]] || { echo "STAGE C REFUSED: ${PINS_FILE} is missing (it ships in the code zip with this script)." >&2; exit 1; }
# shellcheck source=/dev/null
. "${PINS_FILE}"   # ARRIVAL_LIST CHARS_NAME CHARS_SHA256 SIGMA_SHA256 POLICY_UPLOAD JL_MANIFEST_SHA256 PY_REQ CONDA_CREATE
CHARS_FILE="${CL_DATA_IN}/${CHARS_NAME:?pin file names no CHARS_NAME}"
ARRIVAL="${HERE}/${ARRIVAL_LIST:?pin file names no ARRIVAL_LIST}"
is_sha256 "${JL_MANIFEST_SHA256:-}" || { echo "STAGE C REFUSED: the pin file names no JL_MANIFEST_SHA256." >&2; exit 1; }

# ── the arrival list: every uploaded file, sha256, paths relative to the project root ─────
check_arrival () {   # check_arrival all|code
    local lst n bad
    [[ -f "${ARRIVAL}" ]] || die "${ARRIVAL} is missing (it ships in the code zip)"
    lst="$(mktemp)"
    # scripts/Manifest.toml (the local 1.12.6 set) is never read by the cluster's Julia 1.11 while
    # Manifest-v1.11.toml is present: the unzip checks it, this does not. Manifest-v1.11.toml is
    # checked at the launch (L1, before any Julia runs) and exempt afterwards (user, 10-13): the
    # env_julia job refuses if its instantiate/precompile changed it (sha256 before == after).
    # sha256sum writes "<hash>  <path>" (text mode) or "<hash> *<path>" (binary mode): accept both.
    if [[ "$1" == "code" ]]; then grep -E '^[0-9a-f]{64} [ *]scripts/' "${ARRIVAL}" | grep -Ev ' [ *]scripts/(Manifest\.toml|Manifest-v1\.11\.toml)$' > "${lst}"
    else grep -Ev ' [ *]scripts/Manifest\.toml$' "${ARRIVAL}" > "${lst}"; fi
    [[ -s "${lst}" ]] || die "${ARRIVAL##*/} lists no file for the '$1' check"
    n="$(grep -c . "${lst}")"
    bad="$(cd "${PROJ}" && sha256sum -c --quiet "${lst}" 2>&1 | grep -v '^sha256sum: WARNING' || true)"
    rm -f "${lst}"
    [[ -z "${bad}" ]] || die "uploaded files are not the ones built locally: $(tr '\n' ';' <<< "${bad}"). Nothing was copied or submitted. Re-upload them."
    say "   arrival: ${n} file(s) ($1) match the local build"
}

# ── the frozen set: installed by stagec_install, re-hashed by every later job ─────────────
# The sleepiness estimates are produced by the chain itself (Pedro, 10-15: everything on the
# cluster), in data/output/sleep/est{k}; the frozen set is the uploads the chain reads and nothing
# in stage C writes.
frozen_files () {
    printf '%s\n' "${POLICY_CSV}" "${CL_DATA_IN}/market_panel.parquet" "${CHARS_FILE}" "${CL_DATA_IN}/demographics_sigma.parquet"
}
frozen_check () {
    if [[ ! -f "${FROZEN_REC}" ]]; then
        [[ "${DRY}" == "1" ]] && { say "   [dry-run] no frozen-set record yet"; return 0; }
        die "the frozen-set record ${FROZEN_REC} is missing: stagec_install did not finish."
    fi
    sha256sum -c --quiet "${FROZEN_REC}" || die "a frozen file changed since stagec_install (sha256sum -c ${FROZEN_REC}, above): the policy, the panel, the demand-characteristics file or the sigma file. Nothing in stage C writes them; find what did."
    say "   frozen set unchanged ($(grep -c . "${FROZEN_REC}") files)"
}

# ── the queue, fail-closed ────────────────────────────────────────────────────────────────
PROJECT_JOBS='^(sleep_|blp_|rc_|pipe_|pipeline_all|bbl_|cf_|dx_|stagec_|env_|logit)'
queue_check () {   # queue_check [regex] [exclude regex]: no live job whose name matches the first and not the second (default: any job of this project)
    local q own re="${1:-${PROJECT_JOBS}}" ex="${2:-}"
    q="$(squeue -h -u "${USER:-$(id -un)}" -t PENDING,RUNNING,CONFIGURING,SUSPENDED,REQUEUED -o '%i %j')" \
        || die "squeue failed, so the queue cannot be read. Nothing was submitted; run this again when squeue answers."
    own="$(awk -v me="${SLURM_JOB_ID:-none}" -v re="${re}" -v ex="${ex}" '$1 != me && $2 ~ re && !(ex != "" && $2 ~ ex)' <<< "${q}")"
    [[ -z "${own}" ]] || die "jobs of this project are live: $(tr '\n' ';' <<< "${own}"). Wait for them or cancel them (BBL chains with bbl_cancel.sh), then run this again."
    say "   no job of this project is live${1:+ (names matching ${1})}"
}

# ── THE OUTPUT GUARD (M3): nothing of this chain may already be in data/output ────────────
output_guard () {   # output_guard <where> [after]: dies naming every file of an earlier run
    # after = what may already exist because it passed: "" (nothing: launch, --resume chain),
    # prep (--resume chain --from gate-prep|logit), logit (--from gate-logit: + the logit and the
    # draws), smoke (the GPU job and --resume gpu). The stage C gates of the resumed step and later must be absent,
    # except the failed gate being resumed, which its job rewrites.
    local found="" a="${2:-}" gates='G2|G4|SC_(sleep|prep|logit|smoke|blp|blp_E34|blp_e12|bbl|final)'
    case "${a}" in sleep) gates='G2|G4|SC_(prep|logit|smoke|blp|blp_E34|blp_e12|bbl|final)' ;; prep) gates='G4|SC_(logit|smoke|blp|blp_E34|blp_e12|bbl|final)' ;; logit) gates='SC_(smoke|blp|blp_E34|blp_e12|bbl|final)' ;; smoke) gates='SC_(blp|blp_E34|blp_e12|bbl|final)' ;; esac
    found="$( { [[ -z "${a}" ]] && ls -A "${CL_STEP_SLEEP}" 2>/dev/null | sed "s|^|output/sleep/|"
               [[ -z "${a}" || "${a}" == "sleep" ]] && ls -A "${CL_STEP_DEMAND}" 2>/dev/null | sed "s|^|output/demand_prep/|"
               [[ -z "${a}" || "${a}" == "sleep" || "${a}" == "prep" ]] && ls -A "${CL_STEP_LOGIT}" 2>/dev/null | sed "s|^|output/logit/|"
               ls -A "${CL_STEP_BLP}" 2>/dev/null | grep -v '^draws$' | sed "s|^|output/blp/|"
               [[ -z "${a}" || "${a}" == "sleep" || "${a}" == "prep" ]] && ls -A "${CL_STEP_DRAWS}" 2>/dev/null | sed "s|^|output/blp/draws/|"
               ls -A "${CL_STEP_BBL}" 2>/dev/null | grep -E "(${MS_TAG}|${SC_TAG})(\.|_|$)" | sed "s|^|output/bbl/|"
               ls -A "${CL_STEP_BBL}/.dispatch" 2>/dev/null | grep -E "(${MS_TAG}|${SC_TAG})$" | sed "s|^|output/bbl/.dispatch/|"
               ls -A "${CL_DATA_OUT}" 2>/dev/null | grep -E "^\.gate_(${gates})\.json$" | sed "s|^|output/|"
               ls -A "${CL_DATA_OUT}" 2>/dev/null | grep -E '^\.stagec_ladders_' | sed "s|^|output/|"
             } | head -20 )"
    [[ -z "${found}" ]] || die "data/output already holds results of this chain ($1): $(tr '\n' ' ' <<< "${found}")... A launch never overwrites them, and the RC engine SKIPS a stage whose result exists. Move them aside as one attempt (mkdir -p ${CL_DATA_OUT}/_stagec_attempt_${STAMP} and mv the files there), then run this again."
    say "   output guard ($1): data/output holds no result of this chain${a:+ after the ${a} step}"
}
# m5: the engine looks for the logit delta in data/input BEFORE data/output/logit; an uploaded
# one would be loaded instead of the fresh logit's, and the smoke matches only the basename.
input_delta_guard () {
    local up; up="$(ls "${CL_DATA_IN}" 2>/dev/null | grep -E '^logit_delta_' || true)"
    [[ -z "${up}" ]] || die "data/input holds $(tr '\n' ' ' <<< "${up}"): the RC engine would warm-start from it instead of the fresh logit's delta. Move it out of data/input (it is never part of this upload), then run this again."
    say "   data/input holds no logit delta"
}

# ── where each routine's results are ───────────────────────────────────────────────────────
res_jls ()  { printf '%s/blp_results_E%s_spec_12_%s.jls' "${CL_STEP_BLP}" "$1" "$2"; }
res_json () { printf '%s/blp_results_E%s_spec_12_%s.json' "${CL_STEP_BLP}" "$1" "$2"; }
bbl_key ()  { printf 'E%s_spec_12_%s%s' "$1" "${BBL_STAGE}" "$2"; }
fit_rec ()  { printf '%s/stagec_bblfit_%s.json' "${CL_STEP_BLP}" "$(bbl_key "$1" "$2")"; }

sub () {   # sub <name> <dep|""> <sbatch args...> -> job id
    local name="$1" dep="$2" jid; shift 2
    # CL_NO_CPUGEN=1: the pin of every job this script submits is its REQ_*_PIN, never cl_sbatch's
    # default. A plain assignment inside the substitution: cl_sbatch sees it, the job does not
    # inherit it (a prefix assignment before a function call is exported to sbatch, m1).
    jid=$(CL_NO_CPUGEN=1; cl_sbatch ${dep:+--dependency=${dep}} -J "${name}" -o "${LOGD}/${name}_%j.out" -e "${LOGD}/${name}_%j.err" "$@") || true
    cl_require_jid "${jid}" "${name}" || return 1
    printf '%s\n' "${jid}"
}
self_job () {   # self_job <name> <dep|""> <mode> [--time=<wall>] [extra sbatch args]: a CPU job running this script
    local name="$1" dep="$2" m="$3" wall="${REQ_HANDOFF_TIME}"; shift 3
    # one --time per submission (REQ_HANDOFF_TIME unless given first among the extras)
    if [[ "${1:-}" == --time=* ]]; then wall="${1#--time=}"; shift; fi
    sub "${name}" "${dep}" --partition="${ZIP_PARTITION:-${REQ_HANDOFF_PART}}" --time="${wall}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${REQ_HANDOFF_CPUS}" --mem="${REQ_HANDOFF_MEM}" ${REQ_HANDOFF_PIN:+--constraint="${REQ_HANDOFF_PIN}"} "$@" --wrap "cd '${HERE}' && bash ${SELF} ${m}"
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# INSTALL (in the launch): the policy into the folder it is read from; the frozen-set record
# ══════════════════════════════════════════════════════════════════════════════════════════
install_one () {   # install_one <src> <dst> <sha256>: copy + verify; an existing dst must be identical
    local src="$1" dst="$2" want="$3"
    [[ "$(sha "${src}")" == "${want}" ]] || die "${src##*/} does not hash to the pinned ${want:0:12}..."
    if [[ -f "${dst}" ]]; then
        [[ "$(sha "${dst}")" == "${want}" ]] || die "${dst} exists with another sha256 ($(sha "${dst}" | cut -c1-12)...); it is never overwritten. Move it aside and run this again."
        say "   ${dst#${CL_DATA_ROOT}/}: already installed (identical)"; return 0
    fi
    mkdir -p "$(dirname "${dst}")"
    cp "${src}" "${dst}.tmp.$$" && mv -f "${dst}.tmp.$$" "${dst}" || { rm -f "${dst}.tmp.$$"; die "cannot install ${dst}"; }
    [[ "$(sha "${dst}")" == "${want}" ]] || die "the installed ${dst} does not hash to ${want:0:12}..."
    say "   ${dst#${CL_DATA_ROOT}/} <- data/input/${src##*/} (sha256 ${want:0:12}...)"
}
install_mode () {
    say "stage C install: ${CL_DATA_ROOT}"
    cl_export_step_dirs
    install_one "${CL_DATA_IN}/${POLICY_UPLOAD}" "${POLICY_CSV}" "${POLICY_SHA256}"
    cl_check_policy_units "${POLICY_CSV}" || die "the installed policy fails the lhs_unit check (above)"
    [[ "$(sha "${CHARS_FILE}")" == "${CHARS_SHA256}" ]] || die "${CHARS_FILE} does not hash to the pinned ${CHARS_SHA256:0:12}..."
    # the demographic sigma file the draws read (pinned by the build since the 10-14 rebuild)
    if [[ -n "${SIGMA_SHA256:-}" ]]; then
        [[ "$(sha "${CL_DATA_IN}/demographics_sigma.parquet")" == "${SIGMA_SHA256}" ]] || die "${CL_DATA_IN}/demographics_sigma.parquet does not hash to the pinned ${SIGMA_SHA256:0:12}... (a stale sigma file changes the draws)"
    fi
    if [[ -f "${FROZEN_REC}" ]]; then frozen_check
    else frozen_files | xargs sha256sum > "${FROZEN_REC}" || die "cannot write ${FROZEN_REC}"; say "   frozen set recorded: $(grep -c . "${FROZEN_REC}") files -> ${FROZEN_REC##*/}"; fi
    gate_json install true "policy installed; frozen set recorded (policy, panel, chars, sigma)"
    say "STAGE C INSTALL OK"
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# GATES (in the chain): AUTOMATIC SANITY CHECKS ONLY (Pedro, 10-15). Nothing is compared with a
# local run; every check fails closed. The logit gate prints, per routine, the block Pedro reads
# while the BLP runs: the sleepiness first-stage effective F (spec 12, Tech, rival_mean_spread),
# the second-stage rows, the mean phi_t, and the logit alpha with its SE and rows.
# ══════════════════════════════════════════════════════════════════════════════════════════
SLEEP_KEY="IV_RivalSpread x Tech"
SLEEP_SUMMARY="${CL_DATA_OUT}/.stagec_sleep_summary.json"
py_check () {   # py_check <sleep|prep|logit>: 0 ok / 1 broken
    cl_bootstrap_tree >/dev/null; cl_export_step_dirs
    export OPEN_FINANCE_VENV_ENFORCED=1 PYTHONUTF8=1
    cl_setup_python "${CL_PY_REQ_SLEEP}" >/dev/null || return 1
    export SC_KIND="$1" SC_ROUTINES="${ROUTINES_ALL}" SC_KEY="${SLEEP_KEY}" SC_SUMMARY="${SLEEP_SUMMARY}"
    "${PYBIN}" - <<'PY'
import json, math, os, pickle, re, sys
from pathlib import Path
kind = os.environ["SC_KIND"]; out = Path(os.environ["CL_DATA_OUT"])
ks = [int(k) for k in os.environ["SC_ROUTINES"].split()]
key = os.environ["SC_KEY"]; summ = Path(os.environ["SC_SUMMARY"])
broken = []

def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def find_eff_f(fs):
    """the first stage's effective F, wherever the estimator stored it (printed, not gated)"""
    cands = []
    if isinstance(fs, dict):
        cands = [(k, v) for k, v in fs.items() if re.search(r"eff", str(k), re.I) and re.search(r"f", str(k), re.I)]
        for k, v in fs.items():
            if isinstance(v, dict):
                cands += [(f"{k}.{kk}", vv) for kk, vv in v.items() if re.search(r"eff", str(kk), re.I)]
    else:
        for a in ("effective_F", "eff_F", "F_eff", "effective_f", "fvalue"):
            if hasattr(fs, a):
                cands.append((a, getattr(fs, a)))
    for name, v in cands:
        if num(v) is not None:
            return f"{num(v):.3f} ({name})"
    return "n/a (no effective-F field found in the stored first stage)"

if kind == "sleep":
    import pandas as pd
    summary = {}
    for k in ks:
        d = out / "sleep" / f"est{k}"
        pk = d / "estimation_results.pkl"
        if not pk.is_file():
            broken.append(f"E{k}: no {pk}"); continue
        try:
            res = pickle.load(open(pk, "rb"))
        except Exception as e:
            broken.append(f"E{k}: {pk.name} does not load ({e})"); continue
        keys = [m for m in res if key in str(m)]
        if not keys:
            broken.append(f"E{k}: no '{key}' estimate in {pk.name} (keys: {list(res)[:6]})"); continue
        mk = sorted(keys, key=len)[0]; ent = res[mk]
        ss = ent.get("second_stage") if isinstance(ent, dict) else None
        if ss is None:
            broken.append(f"E{k}: '{mk}' has no second stage"); continue
        safe = str(mk).replace(" ", "_").replace(".", "")
        nat = d / "national_phi_t.csv"
        phi_mean, nq = None, 0
        if not nat.is_file():
            broken.append(f"E{k}: no {nat.name}")
        else:
            df = pd.read_csv(nat)
            col = f"phi_t_{safe}"
            if col not in df.columns:
                broken.append(f"E{k}: {nat.name} has no {col}")
            else:
                v = pd.to_numeric(df[col], errors="coerce")
                nq = int(v.notna().sum())
                if nq == 0: broken.append(f"E{k}: {col} has no rows")
                elif not v.dropna().map(math.isfinite).all(): broken.append(f"E{k}: {col} not finite")
                elif v.min() < 0 or v.max() > 1: broken.append(f"E{k}: {col} outside [0,1] (min {v.min():.4f}, max {v.max():.4f})")
                else: phi_mean = float(v.mean())
        mp = d / "market_panel_phis.csv"
        if not mp.is_file():
            broken.append(f"E{k}: no {mp.name}")
        else:
            col = f"phi_mt_{safe}"
            try:
                v = pd.to_numeric(pd.read_csv(mp, usecols=[col])[col], errors="coerce").dropna()
                if len(v) == 0: broken.append(f"E{k}: {col} has no rows")
                elif v.min() < 0 or v.max() > 1 or not v.map(math.isfinite).all():
                    broken.append(f"E{k}: {col} outside [0,1] or not finite (min {v.min():.4f}, max {v.max():.4f})")
            except ValueError:
                broken.append(f"E{k}: {mp.name} has no {col}")
        ss_txt = ss.to_string(max_rows=20) if hasattr(ss, "to_string") else str(ss)[:1500]
        summary[str(k)] = {"key": str(mk), "eff_F": find_eff_f(ent.get("first_stage")), "second_stage": ss_txt,
                           "mean_phi_t": phi_mean, "n_quarters": nq}
        print(f"  E{k}: '{mk}': second stage present; phi_t over {nq} quarters, mean {phi_mean}; first-stage effective F {summary[str(k)]['eff_F']}")
    summ.write_text(json.dumps(summary, indent=1))
elif kind == "prep":
    import pyarrow.parquet as pq
    d = out / "demand_prep"
    for k in ks:
        pat = re.compile(rf"^demand_{k}(?:_.*)?_spec_12\.parquet$")
        hits = sorted(p for p in d.iterdir() if pat.match(p.name) and "_final_" not in p.name) if d.is_dir() else []
        if len(hits) != 1:
            broken.append(f"E{k}: {len(hits)} demand parquets in {d}"); continue
        p = hits[0]; f = pq.ParquetFile(str(p)); names = set(f.schema_arrow.names); n = f.metadata.num_rows
        if n <= 0: broken.append(f"E{k}: {p.name} has no rows")
        for c in ("loo_borrow_assets", "mean_loo_borrow_assets", "chars_src", "segment_src"):
            if c not in names: broken.append(f"E{k}: {p.name} lacks {c}")
        a = Path(str(p) + ".chars_audit.json")
        if not a.is_file():
            broken.append(f"E{k}: no {a.name} (the prep did not apply the demand fixes)"); continue
        au = json.loads(a.read_text())
        if au.get("x_null_rows") != 0 or au.get("iv_bad_rows") != 0:
            broken.append(f"E{k}: x_null_rows={au.get('x_null_rows')} iv_bad_rows={au.get('iv_bad_rows')}")
        print(f"  E{k}: {p.name} rows {n:,}; dropped (no characteristics) {au.get('dropped_no_characteristics')}; "
              f"dropped (no ESTBAN rival branches) {au.get('dropped_no_estban')}; chars file {au.get('chars_file')}")
else:
    j = out / "logit" / "logit_summary_spec_12.json"
    s = json.loads(j.read_text()) if j.is_file() else None
    if s is None: broken.append(f"no {j}")
    sl = json.loads(summ.read_text()) if summ.is_file() else {}
    for k in ks:
        subs = sorted(m for m in (s or {}) if m.startswith(f"E{k}_") and isinstance(s[m], dict))
        if s is not None and not subs: broken.append(f"E{k}: no logit sub-model in {j.name}")
        lines = []
        for m in subs:
            v = s[m]; names = list(v.get("param_names", []))
            if "alpha" not in names: broken.append(f"{m}: no alpha"); continue
            i = names.index("alpha"); a = num(v["theta1"][i])
            se = None
            for sk in ("se", "std_err", "std_errors", "se_theta1", "bse"):
                if isinstance(v.get(sk), list) and len(v[sk]) > i: se = num(v[sk][i]); break
            nobs = next((v[r] for r in ("n_obs", "nobs", "N", "rows", "n") if r in v), None)
            if a is None: broken.append(f"{m}: alpha not finite ({v['theta1'][i]})")
            if m.endswith("_full_dtype") and "dummy_D_type" not in names: broken.append(f"{m}: no dummy_D_type in param_names")
            lines.append(f"    logit {m}: alpha {a if a is None else f'{a:.6f}'} (SE {se if se is None else f'{se:.6f}'}), rows {nobs if nobs is not None else 'n/a'}")
        e = sl.get(str(k), {})
        print(f"\n=== STAGE C SUMMARY E{k} (spec 12; read while the BLP runs) ===")
        print(f"    sleepiness first stage ({e.get('key', key)}, rival_mean_spread): effective F {e.get('eff_F', 'n/a')}")
        print(f"    mean phi_t {e.get('mean_phi_t')} over {e.get('n_quarters')} quarters")
        print("    sleepiness second stage:")
        for ln in str(e.get("second_stage", "n/a")).splitlines()[:20]:
            print("      " + ln)
        for ln in lines: print(ln)
        print(f"=== END SUMMARY E{k} ===")
for b in broken: print("  [BROKEN] " + b)
sys.exit(1 if broken else 0)
PY
}
gate_mode () {   # gate_mode sleep|prep|logit
    say "stage C gate '$1' (automatic sanity checks): data root ${CL_DATA_ROOT}"
    frozen_check
    if py_check "$1"; then gate_json "$1" true "every check passed"; say "STAGE C GATE $1: PASSED"
    else gate_json "$1" false "broken (see the log)"; die "the $1 check failed (the [BROKEN] lines above). The chain stops here, before any GPU job. Once the cause is fixed: $(resume_line "chain --from gate-$1")"; fi
}

# ── the bootstraps (Pedro 10-15): separate jobs off the critical path. Submitted by the chain once
# the point estimates passed their gate; they read only those estimates and the panel; nothing waits
# on them, and a refused submission is reported, not a stop. Their own archive (set sleep, tag
# <chain id>_boot) runs afterany all of them. Sizes: sleep_run.sh's production requests (no sacct
# record of a cluster sleepiness run exists locally); cpugen:turin like the chain. The phi weak-IV
# battery (sleep_weak_iv.py) is not among them: it is a deferred task and the bundle excludes it.
OFFPATH_RE='^(stagec_cpu_image|stagec_boot_zip|sleep_(wcb_band|ame_gate|ame_E[0-9]+))$'
submit_sleep_boot () {
    local j ids="" ameg="" k bsub
    bsub () { local n="$1" dep="$2" c="$3" m="$4" w="$5"; shift 5
        sub "${n}" "${dep}" --partition="${REQ_BOOT_PART}" --time="${w}" --nodes=1 --ntasks=1 --cpus-per-task="${c}" --mem="${m}" \
            ${REQ_BOOT_PIN:+--constraint="${REQ_BOOT_PIN}"} "$@"; }
    say "   bootstraps, off the critical path (nothing waits on them):"
    if j=$(bsub sleep_wcb_band "" "${REQ_WCB_CPUS}" "${REQ_WCB_MEM}" "${REQ_WCB_TIME}" \
            --export=ALL,SLEEP_STEP=wcb_band,SPEC=12,SLEEP_WCB_ROUTINES="1 2" "${HERE}/sleep_job.sh"); then ids="${j}"; say "      sleep_wcb_band (E1/E2 percentile bands) -> ${j}"
    else say "   [!] sleep_wcb_band was refused (the chain goes on)"; fi
    if ameg=$(bsub sleep_ame_gate "" "${REQ_AMEG_CPUS}" "${REQ_AMEG_MEM}" "${REQ_AMEG_TIME}" \
            --export=ALL,SLEEP_STEP=ame_gate,SPEC=12,SLEEP_AME_ROUTINES="3 4" "${HERE}/sleep_job.sh"); then
        ids="${ids:+${ids}:}${ameg}"; say "      sleep_ame_gate (E3/E4 off-path guard) -> ${ameg}"
        for k in 3 4; do
            if j=$(bsub "sleep_ame_E${k}" "afterok:${ameg}" "${REQ_AME_CPUS}" "${REQ_AME_MEM}" "${REQ_AME_TIME}" \
                    --export=ALL,SLEEP_STEP=ame,SLEEP_EST=${k},SPEC=12,SLEEP_AME_BOOT_JOBS=${REQ_AME_CPUS} "${HERE}/sleep_job.sh"); then
                ids="${ids}:${j}"; say "      sleep_ame_E${k} (two-stage AME bootstrap) -> ${j}  (afterok ${ameg})"
            else say "   [!] sleep_ame_E${k} was refused (the chain goes on)"; fi
        done
    else say "   [!] sleep_ame_gate was refused (the chain goes on; no AME)"; fi
    if [[ -n "${ids}" ]]; then
        if j=$(sub stagec_boot_zip "afterany:${ids}" --partition="${REQ_ZIP_PART}" --time="${REQ_ZIP_TIME}" --nodes=1 --ntasks=1 \
                --cpus-per-task="${REQ_ZIP_CPUS}" --mem="${REQ_ZIP_MEM}" ${REQ_ZIP_PIN:+--constraint="${REQ_ZIP_PIN}"} \
                --wrap "cd '${HERE}' && bash cluster_archive.sh --set sleep --copy --tag \"\${SLURM_JOB_ID}_boot\""); then
            say "      stagec_boot_zip (archive set sleep, afterany ${ids//:/, }) -> ${j}"
        else say "   [!] stagec_boot_zip was refused (archive the sleep set by hand when the bootstraps end)"; fi
    fi
    return 0
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# PAYLOADS: every call of an estimation or toolchain script goes through run_payload
# ══════════════════════════════════════════════════════════════════════════════════════════
# run_payload <tag> <command...>: runs the command (a script line or a function of this file).
# The local test sets STAGEC_PAYLOAD_RUNNER (refused outside STAGEC_LOCAL_TEST=1) to a script that
# records the command line and its environment instead of running it: that record is what the
# payload-equivalence test compares with the deployed design's submissions.
run_payload () {
    local tag="$1"; shift
    if [[ -n "${STAGEC_PAYLOAD_RUNNER:-}" ]]; then
        [[ "${STAGEC_LOCAL_TEST:-0}" == "1" ]] || die "STAGEC_PAYLOAD_RUNNER is a local-test knob"
        # STAGEC_PAYLOAD_REAL="<tag> ...": these run for real in a local test (the CPU-image verdict
        # with the real manifest choice, B1 of the 10-15 review)
        if [[ " ${STAGEC_PAYLOAD_REAL:-} " != *" ${tag} "* ]]; then
            STAGEC_PAYLOAD_TAG="${tag}" "${STAGEC_PAYLOAD_RUNNER}" "$@"; return $?
        fi
    fi
    "$@"
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# THE SMOKE VERDICT (per routine), unchanged
# ══════════════════════════════════════════════════════════════════════════════════════════
# smoke_verdict <routine> <log>: what one routine's dry run must show; appends to SMOKE_BAD.
#   3 of 3 stages reached the engine's timing block; the NEW logit delta loaded on each; and
#   THE DRAWS GATE: every observation found its (market, quarter) in the REBUILT key index on
#   every stage ("Obs mapped to draws: N / N"). The engine maps a missing key to a padding row
#   without failing; the old index lacked ~11,270 rows of 2016Q1. 100%, or no ladder.
smoke_verdict () {
    local k="$1" lg="$2" nst nwarm nmap nmapall
    SMOKE_BAD=""
    nst="$(grep -c '\[DRY RUN GPU\] 10 iters' "${lg}" || true)"
    nwarm="$(grep -c 'WARM-START\] Loaded logit_delta_E'"${k}" "${lg}" || true)"
    nmap="$(grep -o 'Obs mapped to draws: [0-9]* / [0-9]*' "${lg}" | awk '$5 == $7' | wc -l | tr -d ' ')"
    nmapall="$(grep -c 'Obs mapped to draws:' "${lg}" || true)"
    say "   E${k}: ${nst} of 3 stages reached the timing block; new logit delta loaded in ${nwarm}; obs mapped to draws 100% in ${nmap} of ${nmapall} ($(grep -o 'Obs mapped to draws: [0-9]* / [0-9]*' "${lg}" | head -1 | cut -d: -f2))"
    (( nst == 3 )) || SMOKE_BAD="${SMOKE_BAD} E${k}(${nst}/3 stages ran)"
    (( nwarm == 3 )) || SMOKE_BAD="${SMOKE_BAD} E${k}(delta warm start ${nwarm}/3)"
    (( nmapall == 3 && nmap == 3 )) || SMOKE_BAD="${SMOKE_BAD} E${k}(obs mapped to draws below 100% or not reported: ${nmap} of ${nmapall} stages complete)"
    if grep -q 'DemandInputError' "${lg}"; then SMOKE_BAD="${SMOKE_BAD} E${k}(DemandInputError in ${lg##*/})"; fi
    return 0
}
# smoke_split <one-process log> <routine>: the routine's segment, between its blp_smoke.jl banners.
smoke_split () {
    awk -v b="=== SMOKE E$2 BEGIN ===" -v e="=== SMOKE E$2 END" 'index($0, b) == 1 {f = 1} f {print} index($0, e) == 1 {f = 0}' "$1"
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# T3 — THE CPU CHAIN (one job): precompile + G0, the sleepiness estimates E1-E4 (spec 12) and their
# sanity gate (then the bootstrap jobs, off the critical path), prep, G2, the prep gate, logit ||
# draws, G4, the logit gate (with the per-routine summary), then the archive (ALWAYS); exits with
# the first failing step's code.
# ══════════════════════════════════════════════════════════════════════════════════════════
read -r -a SLEEP_EXTRA <<< "${STAGEC_SLEEP_EXTRA:-}"   # accepted for the record; the chain calls sleep_job.sh directly
prep_env () { export DEMAND_CHARS_FILE="${CHARS_FILE}" SLEEP_PREP_EXTRA="--skip-steps 5 7 8 --only-spec-12" SLEEP_ACTIVE_ESTS="${ROUTINES_ALL}"; }
CHAIN_STEPS="sleep gate-sleep prep gate-prep logit gate-logit"
step_index () { local i=0 s; for s in ${CHAIN_STEPS}; do [[ "${s}" == "$1" ]] && { echo "${i}"; return 0; }; i=$((i + 1)); done; return 1; }
chain_log () { printf '%s/stagec_chain_%s_%s.log' "${LOGD}" "${SLURM_JOB_ID:-local}" "$1"; }
# step 0 (always): this node's precompile caches (pkgimages are CPU-specific, so T1's node may not
# serve them), the shipped manifest unchanged, every project package loads.
chain_precompile () {
    cl_load_julia || die "module load ${JULIA_MODULE} failed (above)"
    jl_manifest_check
    julia --startup-file=no --project="${HERE}" -e 'using Pkg; Pkg.precompile(); println("  precompiled")' \
        || die "Pkg.precompile() of ${JL_MANIFEST} failed (above)"
    [[ "$(sha "$(cl_manifest_path)")" == "${JL_MANIFEST_SHA256}" ]] || die "${JL_MANIFEST} CHANGED during precompile: the package set is no longer the validated one"
    julia --startup-file=no --project="${HERE}" -e '
        using MKL, LoopVectorization, LinearAlgebra, DataFrames, Optim, ArgParse, Parquet2, JSON3, Distributions, QuasiMonteCarlo, LBFGSB, CUDA
        println("  every project package loads; BLAS: ", BLAS.get_config(), "; threads ", Threads.nthreads())
    ' || die "the project packages do not load (above)"
}
# chain_step <step>: one step of the chain, its own log; returns the step's code.
chain_step () {
    local s="$1" rl rd pl pd
    case "${s}" in
        sleep)
            # Pedro 10-15: the point estimates of the four routines at spec 12, on this node, one
            # after the other (each holds a full market panel). E2-E4 take --spec12; E1 has no spec
            # selector and runs its grid. The bootstraps follow as separate jobs (gate-sleep).
            local k extra
            for k in ${ROUTINES_ALL}; do
                extra="--spec12"; [[ "${k}" == "1" ]] && extra=""
                say "   sleepiness E${k}: sleep_job.sh SLEEP_STEP=est SLEEP_EST=${k}${extra:+ (${extra})} -> $(basename "$(chain_log "sleep_e${k}")")"
                run_payload "sleep_e${k}" env SLEEP_STEP=est SLEEP_EST="${k}" SLEEP_EST_EXTRA="${extra}" SPEC=12 \
                    bash "${HERE}/sleep_job.sh" > "$(chain_log "sleep_e${k}")" 2>&1 || { tail -30 "$(chain_log "sleep_e${k}")"; return 1; }
            done ;;
        gate-sleep)
            ( gate_mode sleep ) || return $?
            submit_sleep_boot ;;
        prep)
            say "   prep: sleep_job.sh SLEEP_STEP=prep (spec 12, step 6 only; DEMAND_CHARS_FILE=${CHARS_NAME}) -> $(basename "$(chain_log prep)")"
            run_payload prep env SLEEP_STEP=prep SPEC=12 DEMAND_PREP_JOBS="${REQ_CHAIN_PREP_JOBS}" SLEEP_ACTIVE_ESTS="${ROUTINES_ALL}" \
                bash "${HERE}/sleep_job.sh" > "$(chain_log prep)" 2>&1 || { tail -30 "$(chain_log prep)"; return 1; }
            say "   gate G2: sleep_job.sh SLEEP_STEP=gate -> $(basename "$(chain_log g2)")"
            run_payload g2 env SLEEP_STEP=gate SLEEP_GATE=G2 SLEEP_GATE_ROUTINES="${ROUTINES_ALL}" SPEC=12 \
                bash "${HERE}/sleep_job.sh" > "$(chain_log g2)" 2>&1 || { tail -30 "$(chain_log g2)"; return 1; } ;;
        gate-prep)
            ( gate_mode prep ) ;;
        logit)
            say "   logit || draws, ${REQ_CHAIN_JULIA_THREADS} Julia threads each (as the two separate jobs had) -> $(basename "$(chain_log logit)"), $(basename "$(chain_log draws)")"
            ( run_payload logit env SLEEP_ACTIVE_ESTS="${ROUTINES_ALL}" SPEC=12 SLURM_CPUS_PER_TASK="${REQ_CHAIN_JULIA_THREADS}" \
                bash "${HERE}/logit_job.sh" > "$(chain_log logit)" 2>&1 ) & pl=$!
            ( run_payload draws env SLURM_CPUS_PER_TASK="${REQ_CHAIN_JULIA_THREADS}" \
                bash "${HERE}/blp_draws_job.sh" > "$(chain_log draws)" 2>&1 ) & pd=$!
            wait "${pl}"; rl=$?; wait "${pd}"; rd=$?
            say "   logit rc ${rl}, draws rc ${rd}"
            [[ ${rl} -eq 0 ]] || tail -30 "$(chain_log logit)"
            [[ ${rd} -eq 0 ]] || tail -30 "$(chain_log draws)"
            [[ ${rl} -eq 0 && ${rd} -eq 0 ]] || return 1
            say "   gate G4: sleep_job.sh SLEEP_STEP=gate -> $(basename "$(chain_log g4)")"
            run_payload g4 env SLEEP_STEP=gate SLEEP_GATE=G4 SLEEP_GATE_ROUTINES="${ROUTINES_ALL}" SPEC=12 \
                bash "${HERE}/sleep_job.sh" > "$(chain_log g4)" 2>&1 || { tail -30 "$(chain_log g4)"; return 1; } ;;
        gate-logit)
            ( gate_mode logit ) ;;
    esac
}
chain_advice () {   # chain_advice <failed step> <code>: the one-line resume after a stop
    case "$1" in
        precompile|g0) echo "Once the cause is fixed: $(resume_line "chain$([[ "${CHAIN_FROM}" != "sleep" ]] && echo " --from ${CHAIN_FROM}")")" ;;
        sleep) echo "Once the cause is fixed, move data/output/sleep aside (mkdir -p ${CL_DATA_OUT}/_stagec_attempt_${STAMP} and mv it there), then: $(resume_line chain)" ;;
        gate-sleep) echo "Once the cause is fixed: $(resume_line "chain --from gate-sleep")" ;;
        prep) echo "Once the cause is fixed, move data/output/demand_prep and .gate_G2.json aside (mkdir -p ${CL_DATA_OUT}/_stagec_attempt_${STAMP} and mv them there), then: $(resume_line chain)" ;;
        gate-prep) echo "Once the cause is fixed: $(resume_line "chain --from gate-prep")" ;;
        logit) echo "Once the cause is fixed, move data/output/logit, data/output/blp/draws and .gate_G4.json aside, then: $(resume_line "chain --from logit")" ;;
        gate-logit) echo "Once the cause is fixed: $(resume_line "chain --from gate-logit")" ;;
    esac
}
chain_mode () {
    local from="${CHAIN_FROM:-sleep}" i0 i s rc=0 failed=""
    CHAIN_FROM="${from}"
    i0="$(step_index "${from}")" || die "--chain --from takes ${CHAIN_STEPS// /, } (got '${from}')"
    say "stage C CPU chain (T3) from '${from}': data root ${CL_DATA_ROOT}"
    STOP_RESUME="$(resume_line "chain$([[ "${from}" != "sleep" ]] && echo " --from ${from}")")"
    env_check chain; check_arrival code; frozen_check; input_delta_guard
    prep_env
    STOP_RESUME=""   # from here every stop prints its own advice (chain_advice)
    say "   step 0: precompile for this node + every package loads -> $(basename "$(chain_log precompile)")"
    ( run_payload precompile chain_precompile ) > "$(chain_log precompile)" 2>&1 || { tail -30 "$(chain_log precompile)"; failed=precompile; rc=1; }
    if [[ -z "${failed}" ]]; then
        say "   step 0: gate G0 (env_job.sh preflight; the GPU image is built by the GPU job) -> $(basename "$(chain_log g0)")"
        run_payload g0 env ENV_STEP=preflight PF_ROUTINES="${ROUTINES_ALL}" PF_WILL_BUILD="draws sysimage" \
            bash "${HERE}/env_job.sh" > "$(chain_log g0)" 2>&1 || { tail -30 "$(chain_log g0)"; failed=g0; rc=1; }
    fi
    if [[ -n "${failed}" ]]; then
        die "the chain's step 0 (${failed}) failed (above); nothing was produced and the GPU job will not start. $(chain_advice "${failed}" "${rc}")"
    fi
    i=0
    for s in ${CHAIN_STEPS}; do
        if (( i >= i0 )); then
            say "step ${s}"
            chain_step "${s}"; rc=$?
            if [[ ${rc} -ne 0 ]]; then failed="${s}"; break; fi
        fi
        i=$((i + 1))
    done
    # The comparison archive ALWAYS runs (m1): a stop must still send what to compare with.
    say "archive: demand_prep + logit (whatever the gates said)"
    if [[ -n "$(ls -A "${CL_STEP_DEMAND}" 2>/dev/null)" ]]; then
        run_payload archive bash "${HERE}/cluster_archive.sh" --set demand_prep --copy --tag "${SLURM_JOB_ID:-local}" \
            > "$(chain_log archive)" 2>&1 || say "   [!] the demand_prep archive failed ($(basename "$(chain_log archive)"))"
    fi
    if [[ -n "$(ls -A "${CL_STEP_LOGIT}" 2>/dev/null)" ]]; then
        run_payload archive bash "${HERE}/cluster_archive.sh" --set logit --copy --tag "${SLURM_JOB_ID:-local}" \
            >> "$(chain_log archive)" 2>&1 || say "   [!] the logit archive failed ($(basename "$(chain_log archive)"))"
    fi
    if [[ -n "${failed}" ]]; then
        printf '%s  STAGE C REFUSED (chain): the step %s stopped the chain (code %s); the GPU job will not start. %s\n' \
            "$(date '+%F %T')" "${failed}" "${rc}" "$(chain_advice "${failed}" "${rc}")"
        exit "${rc}"
    fi
    echo "STAGE C CHAIN: PASSED (prep, G2, the prep gate, logit, draws, G4, the alpha gate)"
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# T2 — THE GPU JOB: the GPU sysimage (built if missing) with its load proof, ONE Julia launch that
# loads the image and proves CUDA (the sysimage probe merged into the CUDA gate), the ONE-PROCESS
# smoke over E1-E4 (blp_smoke.jl) with the draws gate, then the BLP hand-off in its tail.
# ══════════════════════════════════════════════════════════════════════════════════════════
# what cl_export_step_dirs exports (cluster_lib.sh)
STEP_DIR_VARS="OPEN_FINANCE_ROOT SLEEP_OUT_ROOT DEMAND_PREP_DIR CF_FOUNDATION_DIR CF_COST_FWD COST_POLFUNC_DIR STATE_CENTERING_JSON"
gpu_mode () {
    local k lg seg marker bad="" moved rc
    say "stage C GPU job (T2): sysimage, smoke (one process, E${ROUTINES_ALL// /, E}, stages ${SMOKE_STAGES}), then the BLP hand-off"
    STOP_RESUME="$(resume_line gpu)"
    env_check chain; check_arrival code; frozen_check; input_delta_guard
    need_gate SC_logit; need_draws
    output_guard "the GPU job" smoke
    run_payload load_julia cl_load_julia || die "module load ${JULIA_MODULE} failed (above). $(resume_line gpu)"
    if [[ "${STAGEC_ASSUME_SYSIMAGE:-0}" == "1" ]]; then say "   [test] sysimage assumed valid"
    elif cl_sysimage_missing gpu; then
        say "   the GPU sysimage is missing or invalid: building it here (env_job.sh ENV_STEP=sysimage_gpu: build + CUDA.functional() load proof)"
        run_payload sysimage_gpu env ENV_STEP=sysimage_gpu bash "${HERE}/env_job.sh" > "${LOGD}/stagec_gpu_${SLURM_JOB_ID:-local}_sysimage.log" 2>&1 \
            || { tail -40 "${LOGD}/stagec_gpu_${SLURM_JOB_ID:-local}_sysimage.log"; gate_json smoke false "GPU sysimage build failed"; die "the GPU sysimage build failed (above). Once fixed: $(resume_line gpu)"; }
    else say "   the GPU sysimage on disk is valid (kept)"; fi
    say "   sysimage load + CUDA gate in one Julia launch (a refusal ends this job; after the fix: $(resume_line gpu))"
    run_payload gpu_gate cl_require_sysimage_gated gpu
    # The step-dir exports serve the smoke only (env_job.sh sets its own, so the image build above
    # ran in this job's starting environment, as the 10-13 build job did); the hand-off below gets
    # the starting values back, so the ladder jobs inherit what a separate hand-off job passed them.
    local v; local -a pre=()
    for v in ${STEP_DIR_VARS}; do [[ -n "${!v+x}" ]] && pre+=("${v}=${!v}"); done
    cl_bootstrap_tree >/dev/null; cl_export_step_dirs
    marker="$(mktemp "${CL_DATA_OUT}/.stagec_smoke_marker.XXXXXX")"
    lg="${LOGD}/stagec_smoke_${SLURM_JOB_ID:-local}.log"
    say "   smoke: blp_smoke.jl, one process -> ${lg##*/}"
    run_payload smoke env BLP_ENGINE=ift julia --project="${HERE}" ${CL_JULIA_SYS[@]+"${CL_JULIA_SYS[@]}"} --threads="${REQ_SMOKE_THREADS}" \
        "${HERE}/blp_smoke.jl" --routines "${ROUTINES_ALL}" --stage "${SMOKE_STAGES}" --hpc --R "${R}" --seed "${SEED}" --dry-run > "${lg}" 2>&1; rc=$?
    for k in ${ROUTINES_ALL}; do
        seg="${LOGD}/stagec_smoke_E${k}_${SLURM_JOB_ID:-local}.log"
        smoke_split "${lg}" "${k}" > "${seg}"
        if [[ ! -s "${seg}" ]]; then bad="${bad} E${k}(no output: the process stopped before it)"; continue; fi
        if grep -q "=== SMOKE E${k} ERROR" "${seg}"; then bad="${bad} E${k}(threw: $(grep -m1 "=== SMOKE E${k} ERROR" "${seg}" | cut -c1-160))"; tail -30 "${seg}"; continue; fi
        smoke_verdict "${k}" "${seg}"; bad="${bad}${SMOKE_BAD}"
    done
    [[ ${rc} -eq 0 || -n "${bad}" ]] || bad=" (blp_smoke.jl exit ${rc})"
    # The dry run writes an empty stage summary per stage; it is not a result. Moved, not deleted.
    moved="${LOGD}/stagec_smoke_${SLURM_JOB_ID:-local}_strays"; mkdir -p "${moved}"
    find "${CL_STEP_BLP}" -maxdepth 1 -type f -newer "${marker}" -exec mv -t "${moved}" {} + 2>/dev/null || true
    rm -f "${marker}"
    say "   files the dry run left in ${CL_STEP_BLP}: moved to ${moved} ($(ls "${moved}" | wc -l))"
    if [[ -n "${bad}" ]]; then gate_json smoke false "${bad}"; die "SMOKE: FAIL:${bad}. No BLP ladder will start. Once the cause is fixed: $(resume_line gpu)"; fi
    gate_json smoke true "all routines, stages ${SMOKE_STAGES}, one process"
    say "SMOKE: PASS"
    say "the BLP hand-off, in this job's tail"
    STOP_RESUME=""   # the hand-off's stops print their own lines (next-blp, blp, next-bbl)
    unset ${STEP_DIR_VARS}; for v in ${pre[@]+"${pre[@]}"}; do export "${v}"; done
    next_blp
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# T4 — THE CPU-IMAGE JOB (user, 10-14 23:1x): the CPU sysimage the BBL's CPU steps load (bbl_job.sh
# refuses a solve job without it), built on the CPU it will run on (cpugen:turin), beside T3.
# ══════════════════════════════════════════════════════════════════════════════════════════
cpu_load_proof () {   # the image loads HERE and names this CPU (a refusal must not end the job before its gate)
    ( cl_require_sysimage cpu && [[ ${#CL_JULIA_SYS[@]} -gt 0 ]] )
}
# B1 (10-15 review): the verdict must see the manifest JULIA reads. cl_manifest_path picks
# Manifest-v1.11.toml only when julia 1.11 is on PATH; without it, it falls back to Manifest.toml
# (the 1.12.6 file the code zip also ships), whose sha256 is not the one the sidecar recorded, and
# every hand-off started from a login shell refused. So: Julia loaded first, in a subshell.
cpu_verdict_loaded () {
    ( cl_load_julia >/dev/null 2>&1 || { echo "  module load ${JULIA_MODULE} failed: the CPU sysimage cannot be checked"; exit 1; }
      cl_sysimage_verdict cpu )
}
cpu_image_mode () {
    local lg="${LOGD}/stagec_cpu_image_${SLURM_JOB_ID:-local}.log"
    say "stage C CPU-image job (T4): blp_sysimage_cpu.so for the BBL's CPU steps$([[ -n "${REQ_CPUIMG_PIN}" ]] && echo " (${REQ_CPUIMG_PIN})")"
    STOP_RESUME="$(resume_line cpu-image)"; STOP_GATE=cpu_image
    env_check chain; check_arrival code
    run_payload load_julia cl_load_julia || { gate_json cpu_image false "module load ${JULIA_MODULE} failed"; die "module load ${JULIA_MODULE} failed (above). $(resume_line cpu-image)"; }
    if [[ "${STAGEC_ASSUME_SYSIMAGE:-0}" == "1" ]]; then say "   [test] sysimage assumed valid"
    elif cl_sysimage_missing cpu; then
        say "   the CPU sysimage is missing or invalid: building it (env_job.sh ENV_STEP=sysimage_cpu: build + load proof) -> ${lg##*/}"
        run_payload sysimage_cpu env ENV_STEP=sysimage_cpu bash "${HERE}/env_job.sh" > "${lg}" 2>&1 \
            || { tail -40 "${lg}"; gate_json cpu_image false "CPU sysimage build failed (${lg##*/})"; die "the CPU sysimage build failed (above). Once fixed: $(resume_line cpu-image)"; }
    else say "   the CPU sysimage on disk is valid (kept)"; fi
    run_payload cpu_gate cpu_load_proof \
        || { gate_json cpu_image false "the CPU sysimage does not load on $(hostname) or fails its provenance check"; die "the CPU sysimage does not pass its load check (above). Once fixed: $(resume_line cpu-image)"; }
    gate_json cpu_image true "blp_sysimage_cpu.so loads on $(hostname) (provenance + CPU target)"
    say "CPU IMAGE: PASS"
}
cpu_image_gate_or_die () {   # cpu_image_gate_or_die <what refuses> <its rerun line>: T4 passed, and the image still checks
    local f="${CL_DATA_OUT}/.gate_SC_cpu_image.json"
    if [[ ! -f "${f}" ]] || grep -Eq '"ok"[[:space:]]*:[[:space:]]*false' "${f}" || ! grep -Eq '"ok"[[:space:]]*:[[:space:]]*true' "${f}"; then
        die "$1: the CPU sysimage job (stagec_cpu_image) has not passed (${f##*/} $([[ -f "${f}" ]] && echo "says: $(json_str "${f}" detail)" || echo "is missing")); every BBL solve job loads blp_sysimage_cpu.so. Nothing was submitted. If it is still queued or running, wait for it; else rebuild it (one line): $(resume_line cpu-image)  then: $2"
    fi
    run_payload cpu_verdict cpu_verdict_loaded \
        || die "$1: blp_sysimage_cpu.so no longer passes its provenance check (above). Nothing was submitted. Rebuild it (one line): $(resume_line cpu-image)  then: $2"
    say "   .gate_SC_cpu_image.json: ok; blp_sysimage_cpu.so passes its provenance check"
}
submit_cpu_image () {   # -> T4, no dependency
    T4=$(sub stagec_cpu_image "" --partition="${REQ_CPUIMG_PART}" --time="${REQ_CPUIMG_TIME}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${REQ_CPUIMG_CPUS}" --mem="${REQ_CPUIMG_MEM}" ${REQ_CPUIMG_PIN:+--constraint="${REQ_CPUIMG_PIN}"} \
        --wrap "cd '${HERE}' && bash ${SELF} --cpu-image") || return 1
    MAIN="${MAIN:+${MAIN}:}${T4}"
    say "   stagec_cpu_image -> ${T4}  (${REQ_CPUIMG_PART} ${REQ_CPUIMG_CPUS}c/${REQ_CPUIMG_MEM}/${REQ_CPUIMG_TIME}${REQ_CPUIMG_PIN:+, ${REQ_CPUIMG_PIN}}: the CPU sysimage for the BBL, beside the chain)"
}
submit_launch () {   # the launch's submissions: T4, then (with the prep exports) T3 and T2 afterok T3
    submit_cpu_image || return 1
    prep_env
    submit_main sleep
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# THE MAIN SUBMISSIONS (T3, then T2 afterok T3): from the launch and from --resume chain|gpu
# ══════════════════════════════════════════════════════════════════════════════════════════
submit_gpu () {   # submit_gpu <dep|""> -> T2
    T2=$(sub stagec_gpu "$1" --partition="${REQ_GPU_PART}" --gpus="${REQ_GPU_GRES}" --time="${REQ_GPU_TIME}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${REQ_GPU_CPUS}" --mem="${REQ_GPU_MEM}" --wrap "cd '${HERE}' && bash ${SELF} --gpu") || return 1
    MAIN="${MAIN:+${MAIN}:}${T2}"
    say "   stagec_gpu${1:+ ${1}} -> ${T2}  (${REQ_GPU_PART} ${REQ_GPU_GRES}, ${REQ_GPU_CPUS}c/${REQ_GPU_MEM}/${REQ_GPU_TIME}: sysimage, one-process smoke, BLP hand-off)"
}
submit_main () {   # submit_main <from>: T3 (--chain [--from]) then T2 afterok T3; sets MAIN T3 T2
    local from="${1:-sleep}" fromarg=""
    [[ "${from}" == "sleep" ]] || fromarg=" --from ${from}"
    T3=$(sub stagec_chain "" --partition="${REQ_CHAIN_PART}" --time="${REQ_CHAIN_TIME}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${REQ_CHAIN_CPUS}" --mem="${REQ_CHAIN_MEM}" ${REQ_CHAIN_PIN:+--constraint="${REQ_CHAIN_PIN}"} \
        --wrap "cd '${HERE}' && bash ${SELF} --chain${fromarg}") || return 1
    MAIN="${MAIN:+${MAIN}:}${T3}"
    say "   stagec_chain -> ${T3}  (${REQ_CHAIN_PART} ${REQ_CHAIN_CPUS}c/${REQ_CHAIN_MEM}/${REQ_CHAIN_TIME}${REQ_CHAIN_PIN:+, ${REQ_CHAIN_PIN}}: the CPU chain from ${from})"
    submit_gpu "afterok:${T3}" || return 1
}
dry_then_submit () {   # dry_then_submit <main|gpu|cpu-image> [from]
    ( export CL_DRYRUN=1; MAIN=""; case "$1" in gpu) submit_gpu "" ;; cpu-image) submit_cpu_image ;; *) submit_main "${2:-sleep}" ;; esac ) \
        || die "the dry run of the submissions failed (above). Nothing was submitted."
    [[ "${DRY}" == "1" ]] && return 0
    MAIN=""
    if [[ "$1" == "gpu" ]]; then submit_gpu "" || die "the submission was refused; nothing is queued."
    elif [[ "$1" == "cpu-image" ]]; then submit_cpu_image || die "the submission was refused; nothing is queued."
    else submit_main "${2:-sleep}" || { [[ -n "${MAIN}" ]] && scancel ${MAIN//:/ } 2>/dev/null; die "a submission was refused; every job already submitted was cancelled (${MAIN:-none})."; }; fi
}
print_watch_lines () {
    echo "  job ids: ${MAIN//:/ }"
    echo "STATUS (one line):  squeue -u \$USER -h -o \"%.10i %.28j %.9T %.22r %.10M\""
    [[ -n "${T4:-}" ]] && echo "THE CPU IMAGE (one line, when it has run):  tail -n 40 ${HERE}/logs/stagec_cpu_image_${T4}.out"
    [[ -n "${T3:-}" ]] && echo "THE CPU CHAIN (one line, when it has run):  tail -n 40 ${HERE}/logs/stagec_chain_${T3}.out"
    [[ -n "${T2:-}" ]] && echo "THE GPU JOB: SMOKE + HAND-OFF (one line, when it has run):  tail -n 40 ${HERE}/logs/stagec_gpu_${T2}.out"
    [[ -n "${NB:-}" ]] && echo "THE BLP HAND-OFF (one line, when it has run):  tail -n 40 ${HERE}/logs/stagec_next_blp_${NB}.out"
    echo "A STOP prints its own one-line resume (bash ${SELF} --resume <point>) in the log of the job that stopped."
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# RESUME (job): continue from a stop without re-running what passed (m10)
# ══════════════════════════════════════════════════════════════════════════════════════════
need_gate () {   # need_gate <name>: data/output/.gate_<name>.json says ok (every "ok" true)
    local f="${CL_DATA_OUT}/.gate_$1.json"
    [[ -f "${f}" ]] || die "--resume ${MODE_ARG:-${MODE}}: ${f##*/} is missing: the step before it did not pass. Resume from an earlier point."
    if grep -Eq '"ok"[[:space:]]*:[[:space:]]*false' "${f}" || ! grep -Eq '"ok"[[:space:]]*:[[:space:]]*true' "${f}"; then
        die "--resume ${MODE_ARG:-${MODE}}: ${f##*/} does not say ok: the step before it did not pass. Resume from an earlier point."
    fi
    say "   ${f##*/}: ok"
}
need_draws () {   # the draws the smoke and the ladders read
    local f miss=""
    for f in "demo_draws_R${R}_seed${SEED}.jls" "demo_key_index_R${R}_seed${SEED}.jls" "halton_nu_R${R}_seed${SEED}.jls"; do
        [[ -s "${CL_STEP_DRAWS}/${f}" ]] || miss="${miss} ${f}"
    done
    [[ -z "${miss}" ]] || die "the draws are not on disk (missing:${miss} in ${CL_STEP_DRAWS}); the chain's draws step did not finish. Nothing was submitted."
    say "   draws on disk (R=${R}, seed ${SEED})"
}
resume_mode () {
    local p="${MODE_ARG}" from="${RESUME_FROM:-sleep}"
    say "stage C resume from '${p}'$([[ "${p}" == "chain" ]] && echo " --from ${from}"): data root ${CL_DATA_ROOT}$([[ "${DRY}" == "1" ]] && echo '  [DRY RUN]')"
    case "${p}" in chain|gpu|next-blp|blp|next-bbl|cpu-image) ;; *) die "--resume takes chain [--from sleep|gate-sleep|prep|gate-prep|logit|gate-logit], gpu, next-blp, blp, next-bbl or cpu-image (got '${p}')" ;; esac
    if [[ "${STAGEC_ALLOW_MISMATCH:-0}" == "1" ]]; then
        die "STAGEC_ALLOW_MISMATCH=1 opens every gate of a job; a resume opens ONE: use --export=ALL,STAGEC_ALLOW_MISMATCH_STEP=gate-prep (or gate-logit) with --resume chain --from that step"
    fi
    if [[ -n "${STAGEC_ALLOW_MISMATCH_STEP:-}" ]]; then
        [[ "${p}" == "chain" && "${STAGEC_ALLOW_MISMATCH_STEP}" == "${from}" && "${from}" == gate-* ]] \
            || die "STAGEC_ALLOW_MISMATCH_STEP=${STAGEC_ALLOW_MISMATCH_STEP} applies to --resume chain --from that gate step only (got ${p}${from:+ --from ${from}})"
        say "   !!!!! the ${from} step of the resumed chain accepts a mismatch (STAGEC_ALLOW_MISMATCH_STEP, that step only) !!!!!"
    fi
    env_check chain; check_arrival code; frozen_check; input_delta_guard
    case "${p}" in
        blp)
            if [[ "${DRY}" == "1" ]]; then say "   [dry-run] queue not checked"; else queue_check "" "${OFFPATH_RE}"; fi
            need_gate SC_smoke; next_blp resume; return 0 ;;
        next-bbl)
            # The E1/E2 ladders may still run; only the BBL side must be quiet. next_bbl's B0-B3 do the rest.
            if [[ "${DRY}" == "1" ]]; then say "   [dry-run] queue not checked"; else queue_check '^(bbl_|stagec_(next_bbl|bbl_zip|final|repair))'; fi
            next_bbl; return 0 ;;
        cpu-image)
            # T4 alone (it failed, or the image went stale): never while it or the BBL runs.
            if [[ "${DRY}" == "1" ]]; then say "   [dry-run] queue not checked"; else queue_check '^(stagec_cpu_image$|stagec_(next_bbl|bbl_zip|final|repair)|bbl_)'; fi
            dry_then_submit cpu-image
            if [[ "${DRY}" == "1" ]]; then echo "STAGE C RESUME DRY RUN COMPLETE (${p})"; return 0; fi
            echo; echo "STAGE C RESUMED: the CPU-image job"; print_watch_lines; return 0 ;;
    esac
    # T4 and the bootstraps run beside the chain and write none of its files: they do not block these.
    if [[ "${DRY}" == "1" ]]; then say "   [dry-run] queue not checked"; else queue_check "" "${OFFPATH_RE}"; fi
    # the prep exports every chain job inherited on 10-13 (env_check allows them in a job)
    prep_env
    case "${p}" in
        chain)
            step_index "${from}" >/dev/null || die "--resume chain --from takes ${CHAIN_STEPS// /, } (got '${from}')"
            case "${from}" in
                sleep)      output_guard "resume chain" ;;
                gate-sleep) output_guard "resume chain --from ${from}" sleep ;;
                prep)       need_gate SC_sleep; output_guard "resume chain --from ${from}" sleep ;;
                gate-prep)  need_gate SC_sleep; need_gate G2; output_guard "resume chain --from ${from}" prep ;;
                logit)      need_gate SC_prep; output_guard "resume chain --from ${from}" prep ;;
                gate-logit) need_gate SC_prep; need_gate G4; need_draws; output_guard "resume chain --from ${from}" logit ;;
            esac
            dry_then_submit main "${from}" ;;
        gpu)
            need_gate SC_logit; need_draws; output_guard "resume gpu" smoke
            dry_then_submit gpu ;;
        next-blp)
            need_gate SC_smoke; blp_clean_or_die
            ( export CL_DRYRUN=1; self_job stagec_next_blp "" --next-blp >/dev/null ) || die "the dry run of the submission failed (above)"
            if [[ "${DRY}" != "1" ]]; then MAIN=$(self_job stagec_next_blp "" --next-blp) || die "the submission was refused"; NB="${MAIN}"; fi ;;
    esac
    if [[ "${DRY}" == "1" ]]; then echo "STAGE C RESUME DRY RUN COMPLETE (${p})"; return 0; fi
    echo
    echo "STAGE C RESUMED FROM ${p}$([[ "${p}" == "chain" ]] && echo " --from ${from}")$([[ -n "${STAGEC_ALLOW_MISMATCH_STEP:-}" ]] && echo " (the ${STAGEC_ALLOW_MISMATCH_STEP} step accepts the mismatch)")"
    print_watch_lines
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# THE BLP HAND-OFF (job)
# ══════════════════════════════════════════════════════════════════════════════════════════
blp_clean_or_die () {   # M3: no RC result or checkpoint may sit in data/output/blp before a ladder
    local left
    left="$(ls "${CL_STEP_BLP}" 2>/dev/null | grep -E '^blp_(results|checkpoint)_' || true)"
    [[ -z "${left}" ]] || die "data/output/blp holds $(grep -c . <<< "${left}") RC result/checkpoint file(s) ($(head -3 <<< "${left}" | tr '\n' ' ')...). The engine SKIPS a stage whose result exists, so the ladder would not run. Move them aside and run this again."
}
# The ladder marker: written by stagec_next_blp right before the first ladder is submitted (the blp
# folder was empty then, M3). A ladder resume (--resume blp) accepts only RC results and checkpoints
# written after it, i.e. by this chain's ladders, and lets the engine skip the stages on disk.
LADDER_MARK="${CL_DATA_OUT}/.stagec_ladders_20261012"
ladder_resume_check () {
    local old
    [[ -f "${LADDER_MARK}" ]] || die "--resume blp: no ladder marker (${LADDER_MARK##*/}): the ladders of this chain never started; use --resume next-blp"
    old="$(find "${CL_STEP_BLP}" -maxdepth 1 -type f \( -name 'blp_results_*' -o -name 'blp_checkpoint_*' \) ! -newer "${LADDER_MARK}" 2>/dev/null | sed 's|.*/||' | head -3 | tr '\n' ' ')"
    [[ -z "${old}" ]] || die "--resume blp: data/output/blp holds RC files older than this chain's ladders (${old}...): not this chain's; move them aside"
    say "   every RC file in data/output/blp was written by this chain's ladders (after ${LADDER_MARK##*/})"
}
routines_unfinished () {   # routines_unfinished <routines>: those without both ext1 and extended on disk
    local k out=""
    for k in $1; do [[ -f "$(res_jls "${k}" "${SE_STAGE}")" && -f "$(res_jls "${k}" "${BBL_STAGE}")" ]] || out="${out:+${out} }${k}"; done
    printf '%s' "${out}"
}
run_ladders () {   # run_ladders <routines> <log tag>: blp_run.sh; prints the terminal ids (empty on a refusal)
    local out="${LOGD}/stagec_blp_run_$2_${STAMP}.log"
    # the ladder memory (cl_mem_for) from REQ_RC_MEM / REQ_RC_MEM_BIG (values = cluster_lib.sh's defaults)
    if CL_MEM_DEFAULT="${REQ_RC_MEM}" CL_MEM_BIG="${REQ_RC_MEM_BIG}" bash "${HERE}/blp_run.sh" --routines "$1" --no-draws --no-sysimage --no-zip ${BLP_EXTRA[@]+"${BLP_EXTRA[@]}"} > "${out}" 2>&1; then
        sed -n 's/^BLP_TERM_JOBIDS=//p' "${out}" | tail -1
    else
        tail -30 "${out}" >&2
    fi
}
next_blp () {   # next_blp [resume]: the four ladders (fresh), or the unfinished ones (--resume blp)
    local mode="${1:-fresh}" g34="3 4" g12="1 2" term34="" term12="" hand="" zipj alljobs
    say "stage C BLP hand-off$([[ "${mode}" == "resume" ]] && echo ' (RESUME: the unfinished ladders only)'): data root ${CL_DATA_ROOT}"
    env_check chain; check_arrival code; frozen_check; input_delta_guard
    if [[ "${mode}" == "resume" ]]; then
        ladder_resume_check
        g34="$(routines_unfinished "3 4")"; g12="$(routines_unfinished "1 2")"
        [[ -n "${g34}${g12}" ]] || { echo "STAGE C: every ladder has ext1 and extended on disk; nothing to resume. If the BBL hand-off did not run: $(resume_line next-bbl)"; return 0; }
        say "   unfinished: ${g34:+E${g34// /, E}}${g34:+${g12:+, }}${g12:+E${g12// /, E}} (the engine skips every stage already on disk)"
    else
        blp_clean_or_die
        [[ "${DRY}" == "1" ]] || { printf 'ladders submitted by %s --next-blp, job %s, %s\n' "${SELF}" "${SLURM_JOB_ID:-}" "$(date '+%F %T')" > "${LADDER_MARK}" || die "cannot write ${LADDER_MARK}"; }
    fi
    # E3/E4 first, then the BBL hand-off at once, then E1/E2 (m3): a refusal of the E1/E2 ladders
    # can no longer strand the E3/E4 ladders without their BBL hand-off.
    if [[ -n "${g34}" ]]; then
        term34="$(run_ladders "${g34}" "${g34// /}")"
        [[ -n "${term34}" ]] || die "blp_run.sh --routines '${g34}' was refused or gave no terminal id (logs/stagec_blp_run_${g34// /}_${STAMP}.log); check squeue for rc_ift_E${g34// /, rc_ift_E} jobs and cancel any before resuming. Once fixed: $(resume_line "$([[ "${mode}" == "resume" ]] && echo blp || echo next-blp)")"
        say "   ladders E${g34// /, E}: terminals ${term34//:/, }"
        # A failed head job (a TwoStartError, say) never completes its terminal: this hand-off is then
        # cancelled with no log, and stagec_blp_zip's log names the missing results and the resume line.
        hand=$(self_job stagec_next_bbl "afterok:${term34}" --next-bbl) || die "stagec_next_bbl was refused; the E${g34// /, E} ladders stay queued (${term34}). When they have finished: $(resume_line next-bbl)"
        say "   stagec_next_bbl afterok ${term34//:/, } -> ${hand}  (E3/E4 only: the BBL does not wait for E1/E2)"
    fi
    if [[ -n "${g12}" ]]; then
        term12="$(run_ladders "${g12}" "${g12// /}")"
        if [[ -n "${term12}" ]]; then say "   ladders E${g12// /, E}: terminals ${term12//:/, }"
        else say "   [!] the E${g12// /, E} ladders were NOT submitted (logs/stagec_blp_run_${g12// /}_${STAMP}.log); ${g34:+E3/E4, the BBL hand-off and }the archive go on."; fi
    fi
    alljobs="${term34}${term12:+${term34:+:}${term12}}${hand:+:${hand}}"
    [[ -n "${alljobs}" ]] || die "no ladder was submitted (above). Once fixed: $(resume_line "$([[ "${mode}" == "resume" ]] && echo blp || echo next-blp)")"
    # D8/m16 (confirmed): afterANY every ladder (the engine writes its results in place, so nothing
    # is archived while a ladder runs): the four-routine blp set when E1-E4 all have ext1 and
    # extended, otherwise the SEPARATE E3/E4 archive (blp_outputs_<id>_E34.zip; ingest it with
    # --routines 3,4); a failing check prints the missing results and the --resume blp line.
    zipj=$(sub stagec_blp_zip "afterany:${alljobs}" --partition="${ZIP_PARTITION:-${REQ_ZIP_PART}}" --time="${REQ_ZIP_TIME}" --nodes=1 --ntasks=1 --cpus-per-task="${REQ_ZIP_CPUS}" --mem="${REQ_ZIP_MEM}" ${REQ_ZIP_PIN:+--constraint="${REQ_ZIP_PIN}"} \
        --wrap "cd '${HERE}' && { if bash ${SELF} --check-blp; then bash cluster_archive.sh --set blp --copy --tag \"\${SLURM_JOB_ID}\"; elif bash ${SELF} --check-blp '3 4'; then bash cluster_archive.sh --set blp --copy --tag \"\${SLURM_JOB_ID}_E34\"; fi; bash ${SELF} --collect-logs; bash cluster_archive.sh --set logs --copy --tag \"\${SLURM_JOB_ID}\"; }") \
        || die "stagec_blp_zip was refused; the ladders${hand:+ and the hand-off} stay queued. When the ladders have finished: cd ${HERE} && bash ${SELF} --check-blp && bash cluster_archive.sh --set blp --copy --tag <jobid>"
    say "   stagec_blp_zip afterany ${alljobs//:/, } -> ${zipj}  (blp set when E1-E4 are whole, else the E3/E4 archive _E34; logs always)"
    echo "STAGEC_BLP_TERM=${term34}${term12:+${term34:+:}${term12}}"
    echo "IF A LADDER FAILS (a TwoStartError in a head job, say): its later stages and stagec_next_bbl are cancelled with no log; stagec_blp_zip's log names the missing results and prints the resume line:  cd ${HERE} && cat logs/stagec_blp_zip_${zipj}.out"
    if [[ -n "${g12}" && -z "${term12}" ]]; then
        gate_json blp_e12 false "E${g12// /, E} ladders not submitted"
        die "the E${g12// /, E} ladders were not submitted (above); ${hand:+E3/E4 (${term34}), the BBL hand-off (${hand}) and }the archive (${zipj}) are. Once the cause is fixed and every ladder has ended (one line): $(resume_line blp)"
    fi
}
check_blp () {   # check_blp [routines]: ext1 (the joint SEs) and extended on disk for every routine named
    local k st miss="" ks="${1:-${ROUTINES_ALL}}" g
    [[ "${ks}" =~ ^[1-4]( [1-4])*$ ]] || die "--check-blp takes routines like '3 4' (got '${ks}')"
    g="blp"; [[ "${ks}" != "${ROUTINES_ALL}" ]] && g="blp_E${ks// /}"
    for k in ${ks}; do for st in ${SE_STAGE} ${BBL_STAGE}; do
        [[ -f "$(res_jls "${k}" "${st}")" ]] || miss="${miss} $(basename "$(res_jls "${k}" "${st}")")"
    done; done
    if [[ -n "${miss}" ]]; then
        gate_json "${g}" false "missing:${miss}"
        die "the RC results of E${ks// /, E} are incomplete (missing:${miss}); no blp archive of them is written, so no partial blp_outputs zip can be ingested. The logs are archived. The ladder(s) did not finish: see logs/rc_ift_E*_head_*.out (a TwoStartError stops a head job) and logs/rc_ift_E*_ext*_*.out. Once the cause is fixed and every ladder has ended, the unfinished ladders resume (the stages on disk are skipped; the BBL hand-off follows E3/E4): $(resume_line blp)"
    fi
    gate_json "${g}" true "ext1 + extended on disk for E${ks// /, E}"
    say "STAGE C CHECK-BLP OK (E${ks// /, E})"
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# THE BBL HAND-OFF (job)
# ══════════════════════════════════════════════════════════════════════════════════════════
fit_record_write () {   # fit_record_write <routine> <tag>
    local k="$1" t="$2" f js rec h hj
    f="$(res_jls "${k}" "${BBL_STAGE}")"; js="$(res_json "${k}" "${BBL_STAGE}")"; rec="$(fit_rec "${k}" "${t}")"
    h="$(sha "${f}")"; is_sha256 "${h}" || return 1
    hj="$(sha "${js}")"; is_sha256 "${hj}" || return 1
    printf '{\n "bbl_key": "%s",\n "fit": "%s",\n "fit_sha256": "%s",\n "fit_json": "%s",\n "fit_json_sha256": "%s",\n "recorded": "%s",\n "recorded_by": "%s --next-bbl, job %s",\n "bbl_archive_jobs": ""\n}\n' \
        "$(bbl_key "${k}" "${t}")" "${f##*/}" "${h}" "${js##*/}" "${hj}" "$(date '+%F %T')" "${SELF}" "${SLURM_JOB_ID:-}" > "${rec}.tmp" || return 1
    mv -f "${rec}.tmp" "${rec}" || return 1
    [[ "$(json_str "${rec}" fit_sha256)" == "${h}" && "$(json_str "${rec}" fit_json_sha256)" == "${hj}" ]]
}
fit_record_add_archive () {   # <routine> <tag> <jobid>: appended to the record's list of archive jobs
    local rec jobs; rec="$(fit_rec "$1" "$2")"
    [[ -f "${rec}" && "$3" =~ ^[0-9]+$ ]] || return 1
    jobs="$(json_str "${rec}" bbl_archive_jobs)"
    [[ " ${jobs} " == *" $3 "* ]] && return 0
    jobs="${jobs:+${jobs} }$3"
    sed 's/^ "bbl_archive_jobs": "[0-9 ]*"$/ "bbl_archive_jobs": "'"${jobs}"'"/' "${rec}" > "${rec}.tmp" && mv -f "${rec}.tmp" "${rec}"
    [[ "$(json_str "${rec}" bbl_archive_jobs)" == "${jobs}" ]]
}
bbl_files_of () {   # every file of a BBL run on disk: cost params, psi, the launch context
    local key="$1"
    ls "${CL_STEP_BBL}" 2>/dev/null | grep -E "^(cost_params_${key}\.json|psi_(eq|dev|starts)_${key}(_shard[0-9]+of[0-9]+)?\.(parquet|json))$" || true
    if [[ -d "$(cl_bbl_dispatch_dir "${key}")" ]]; then echo ".dispatch/${key}/"; fi
    return 0
}
held_sweep_check () {   # held_sweep_check <launch log> <tag> (m8): a sweep bbl_run.sh could not release stays HELD
    local lg="$1" t="$2" ids
    ids="$(grep -o 'could not release sweep [0-9]*' "${lg}" 2>/dev/null | grep -o '[0-9]*$' | tr '\n' ' ')"
    [[ -z "${ids// /}" ]] && return 0
    say "   !!!!! ${t}: sweep job(s) ${ids}stay HELD (bbl_run.sh could not release them): the solve, the archive and stagec_final pend until released. Release (one line): scontrol release ${ids% }"
    HELD="${HELD} ${t}:${ids% }"
}
bbl_launch_tag () {   # bbl_launch_tag <tag> -> sets SOLVES; the launch, its archive job, the held-sweep check
    local t="$1" lg z k
    if [[ "${t}" == "${MS_TAG}" ]]; then set -- "${MS_FLAGS[@]}"; lg="${LOGD}/stagec_bbl_ms_${STAMP}.log"; else set -- "${SC_FLAGS[@]}"; lg="${LOGD}/stagec_bbl_sc_${STAMP}.log"; fi
    SOLVES=""
    env POLICY_CSV="${POLICY_CSV}" CL_DRYRUN="${DRY}" CL_VERBOSE=1 bash "${HERE}/bbl_run.sh" "$@" > "${lg}" 2>&1 || { tail -40 "${lg}"; return 1; }
    SOLVES="$(sed -n 's/^BBL_SOLVE_JOBIDS=//p' "${lg}" | tail -1)"
    say "   ${t}: $(sed -n 's/^BBL_JOB_NAMES=//p' "${lg}" | tail -1); ids $(sed -n 's/^BBL_ALL_JOBIDS=//p' "${lg}" | tail -1)"
    [[ -n "${SOLVES}" ]] || { say "   ${t}: the launch returned no solve job id (${lg})"; return 2; }
    held_sweep_check "${lg}" "${t}"
    # B5: the archive of this tag, submitted right after its launch (m2/D10), afterANY its solves
    # (a refused promotion exits 3 and must not cancel the download), --check-bbl first.
    z=$(sub "stagec_bbl_zip${t}" "afterany:${SOLVES}" --partition="${ZIP_PARTITION:-${REQ_ZIP_PART}}" --time="${REQ_ZIP_TIME}" --nodes=1 --ntasks=1 --cpus-per-task="${REQ_ZIP_CPUS}" --mem="${REQ_ZIP_MEM}" ${REQ_ZIP_PIN:+--constraint="${REQ_ZIP_PIN}"} \
        --wrap "cd '${HERE}' && bash ${SELF} --check-bbl ${t} && bash cluster_archive.sh --set bbl --copy --slim --psi-tag '${t}' --tag \"\${SLURM_JOB_ID}\"") \
        || { say "   [!] the archive job of ${t} was refused; ${t} stays submitted. By hand later: cd ${HERE} && bash ${SELF} --check-bbl ${t} && bash cluster_archive.sh --set bbl --copy --slim --psi-tag ${t} --tag <jobid>"; return 3; }
    say "   stagec_bbl_zip${t} afterany ${SOLVES//:/, } -> ${z}  (-> data/output/download/bbl_outputs_${z}.zip)"
    if [[ "${DRY}" != "1" ]]; then
        for k in ${BBL_ROUTINES}; do fit_record_add_archive "${k}" "${t}" "${z}" || say "   [!] could not add ${z} to $(basename "$(fit_rec "${k}" "${t}")")"; done
    fi
    ZIPS="${ZIPS:+${ZIPS}:}${z}"; LAUNCHED="${LAUNCHED} ${t}"
}
submit_final () {   # submit_final <archive job ids, colon-separated>
    local fin
    fin=$(sub stagec_final "afterany:$1" --partition="${ZIP_PARTITION:-${REQ_ZIP_PART}}" --time="${REQ_ZIP_TIME}" --nodes=1 --ntasks=1 --cpus-per-task="${REQ_ZIP_CPUS}" --mem="${REQ_ZIP_MEM}" ${REQ_ZIP_PIN:+--constraint="${REQ_ZIP_PIN}"} \
        --wrap "cd '${HERE}' && bash ${SELF} --final; bash cluster_archive.sh --set gates --copy --tag \"\${SLURM_JOB_ID}\"; bash ${SELF} --collect-logs; bash cluster_archive.sh --set logs --copy --tag \"\${SLURM_JOB_ID}\"") \
        || { say "   [!] stagec_final was refused; everything else stays submitted"; return 1; }
    say "   stagec_final afterany ${1//:/, } -> ${fin}  (frozen set, .gate_SC_final.json, gates + logs)"
}
# clear_failed_launch <tag>: after a launch bbl_run.sh refused and rolled back (rc 1), the tag's
# dispatch folders (context.env, the shard map, the event log, a STOP marker) are all that is left,
# and the fresh-only check (B3) would refuse the relaunch on them. They are MOVED aside, never deleted,
# to bbl/_stagec_failed_launch<tag>_<stamp>/, and only for a run without a single shard or cost file
# on disk. Returns 0 when nothing of the tag is left in the way.
clear_failed_launch () {
    local t="$1" k key d dest="${CL_STEP_BBL}/_stagec_failed_launch${1}_${STAMP}" left=""
    for k in ${BBL_ROUTINES}; do
        key="$(bbl_key "${k}" "${t}")"
        if [[ -n "$(bbl_files_of "${key}" | grep -v '^\.dispatch/' || true)" ]]; then left="${left} ${key}"; continue; fi
        d="$(cl_bbl_dispatch_dir "${key}")"
        [[ -d "${d}" ]] || continue
        mkdir -p "${dest}" && mv "${d}" "${dest}/" || { left="${left} ${key}"; continue; }
        say "   ${t}: the refused launch left only .dispatch/${key}/ (no shard, no cost file): moved aside to bbl/${dest##*/}/"
    done
    [[ -z "${left}" ]] || say "   [!] ${t}: shard or cost files are on disk for${left}: nothing moved; inspect them (finish with --repair-bbl ${t}, or archive and move them aside)"
    [[ -z "${left}" ]]
}
next_bbl () {   # next_bbl [tag]: both tags (the hand-off), or one of them (a launch that failed)
    local k t st status lg rerun relaunch there tags rc cleared
    tags="${MS_TAG} ${SC_TAG}"
    if [[ -n "${MODE_ARG}" && "${MODE}" == "next-bbl" ]]; then
        [[ "${MODE_ARG}" == "${MS_TAG}" || "${MODE_ARG}" == "${SC_TAG}" ]] || die "--next-bbl takes no argument, ${MS_TAG} or ${SC_TAG} (got '${MODE_ARG}')"
        tags="${MODE_ARG}"
    fi
    rerun="$(resume_line next-bbl)"; relaunch="${rerun}"
    [[ "${tags}" == *" "* ]] || relaunch="cd ${HERE} && sbatch --parsable -p ${REQ_HANDOFF_PART} -t ${REQ_HANDOFF_TIME} -c ${REQ_HANDOFF_CPUS} --mem=${REQ_HANDOFF_MEM} -J stagec_next_bbl -o logs/stagec_next_bbl_%j.out --wrap \"bash ${SELF} --next-bbl ${tags}\""
    say "stage C BBL hand-off (${tags}): data root ${CL_DATA_ROOT}$([[ "${DRY}" == "1" ]] && echo '  [DRY RUN]')"
    say "B0. the environment, the code, the frozen set and the CPU sysimage"; env_check chain; check_arrival code; frozen_check
    cpu_image_gate_or_die "the BBL hand-off" "${rerun}"
    say "B1. the fits the BBL reads (E${BBL_ROUTINES// /, E}): ${SE_STAGE} with JOINT standard errors, and ${BBL_STAGE}"
    for k in ${BBL_ROUTINES}; do
        for st in ${SE_STAGE} ${BBL_STAGE}; do
            [[ -f "$(res_jls "${k}" "${st}")" && -f "$(res_json "${k}" "${st}")" ]] || die "E${k} ${st}: $(basename "$(res_jls "${k}" "${st}")") or its .json is missing; the ladder did not finish (logs/rc_ift_E${k}_*.out; a TwoStartError stops a head job). No BBL job was submitted. Once the cause is fixed and every ladder has ended, resume the unfinished ladders (the BBL hand-off follows them): $(resume_line blp)"
        done
        status="$(json_str "$(res_json "${k}" "${SE_STAGE}")" se_status)"
        [[ "${status}" == "ok" ]] || die "E${k} ${SE_STAGE} has no joint standard errors: se_status '${status:-<absent>}' in $(basename "$(res_json "${k}" "${SE_STAGE}")"). Joint SEs are REQUIRED before the BBL (the SE values themselves are never read as evidence). No BBL job was submitted. After the fit is fixed: ${rerun}"
        say "   E${k}: ${SE_STAGE} se_status ok; ${BBL_STAGE} on disk"
    done
    say "B2. the policy (never refit)"
    [[ -f "${POLICY_CSV}" ]] || die "${POLICY_CSV} is missing"
    [[ "$(sha "${POLICY_CSV}")" == "${POLICY_SHA256}" ]] || die "${POLICY_CSV} has sha256 $(sha "${POLICY_CSV}"), not the pinned ${POLICY_SHA256:0:12}..."
    say "   ${POLICY_CSV##*/} sha256 ${POLICY_SHA256:0:12}..."
    say "B3. fresh launches only, and the fit records"
    for t in ${tags}; do for k in ${BBL_ROUTINES}; do
        there="$(bbl_files_of "$(bbl_key "${k}" "${t}")")"
        [[ -z "${there}" ]] || die "files of the BBL run $(bbl_key "${k}" "${t}") are already on disk ($(tr '\n' ' ' <<< "${there}")). The hand-off launches fresh runs only; nothing was submitted. To finish that run on the same fit: bash ${SELF} --repair-bbl ${t} (in a small job). To start over: archive and move its files aside, then: ${relaunch}. Never bbl_run.sh --repair by hand."
    done; done
    for t in ${tags}; do for k in ${BBL_ROUTINES}; do
        if [[ "${DRY}" == "1" ]]; then say "   [dry-run] would record the sha256 of $(basename "$(res_jls "${k}" "${BBL_STAGE}")") in $(basename "$(fit_rec "${k}" "${t}")")"
        else fit_record_write "${k}" "${t}" || die "the fit record $(fit_rec "${k}" "${t}") could not be written whole. No BBL job was submitted."
             say "   $(basename "$(fit_rec "${k}" "${t}")"): fit sha256 $(json_str "$(fit_rec "${k}" "${t}")" fit_sha256 | cut -c1-12)..."; fi
    done; done
    say "B4. dry runs of every launch first"
    for t in ${tags}; do
        lg="${LOGD}/stagec_bbl_dry${t}_${STAMP}.log"
        if [[ "${t}" == "${MS_TAG}" ]]; then set -- "${MS_FLAGS[@]}"; else set -- "${SC_FLAGS[@]}"; fi
        env POLICY_CSV="${POLICY_CSV}" CL_DRYRUN=1 CL_VERBOSE=1 bash "${HERE}/bbl_run.sh" "$@" > "${lg}" 2>&1 \
            || { tail -25 "${lg}"; die "the dry run of the ${t} launch was refused (${lg}). No BBL job was submitted."; }
        say "   ${t}: dry run OK ($(grep -c '\[dry-run\] sbatch' "${lg}") jobs: $(sed -n 's/^BBL_JOB_NAMES=//p' "${lg}" | tail -1))"
    done
    say "B5. each launch, then at once its archive (afterANY its solves; --check-bbl first); then stagec_final"
    ZIPS=""; LAUNCHED=""; HELD=""
    for t in ${tags}; do
        bbl_launch_tag "${t}"; rc=$?
        if [[ ${rc} -eq 0 ]]; then continue; fi
        # rc 1: bbl_run.sh refused and rolled back; its dispatch folders are moved aside so that the
        # relaunch printed below passes the fresh-only check (D3f). rc 2/3: jobs are queued; nothing moved.
        cleared=0
        if [[ ${rc} -eq 1 && "${DRY}" != "1" ]] && clear_failed_launch "${t}"; then cleared=1; fi
        if [[ -z "${LAUNCHED}" && ${rc} -eq 1 ]]; then die "the ${t} launch failed (above); bbl_run.sh withdrew what it had submitted. No BBL job is queued. When fixed: ${relaunch}"; fi
        # A later tag failed, or an archive was refused: what is submitted keeps its archive and gets
        # the final job; the failed tag is relaunched alone.
        if [[ -n "${ZIPS}" ]]; then submit_final "${ZIPS}"; fi
        gate_json bbl false "submitted:${LAUNCHED:- none}; ${t} failed (rc ${rc})"
        die "the ${t} launch did not complete (rc ${rc}, above). Submitted and archived:${LAUNCHED:- none}. $( ((cleared)) && echo "When fixed, ${t} alone (one line): cd ${HERE} && sbatch --parsable -p ${REQ_HANDOFF_PART} -t ${REQ_HANDOFF_TIME} -c ${REQ_HANDOFF_CPUS} --mem=${REQ_HANDOFF_MEM} -J stagec_next_bbl -o logs/stagec_next_bbl_%j.out --wrap \"bash ${SELF} --next-bbl ${t}\"" || echo "Files of ${t} are in the way (above): inspect them before any relaunch.")"
    done
    submit_final "${ZIPS}" || true
    gate_json bbl true "submitted:${LAUNCHED}${HELD:+; HELD sweeps:${HELD}}"
    echo
    echo "STAGE C BBL SUBMITTED$([[ "${DRY}" == "1" ]] && echo ' [DRY RUN]'):${LAUNCHED} (${MS_TAG}: multi-start, --promote; ${SC_TAG}: single curve; both --no-tables: render the cost tables locally)"
    if [[ -n "${HELD}" ]]; then echo "!!!!! HELD SWEEPS (release them, or nothing of those tags runs):${HELD}  -> scontrol release <id>"; fi
    echo "STATUS (one line):"
    echo "  cd ${HERE} && for t in ${tags}; do bash bbl_status.sh --routines \"${BBL_ROUTINES}\" --psi-tag \$t; done"
    echo "NEVER run the 'bbl_run.sh ... --repair' line the BBL tools print for these tags: it skips the fit binding. To repair:"
    echo "  cd ${HERE} && sbatch --parsable -p ${REQ_HANDOFF_PART} -t ${REQ_HANDOFF_TIME} -c ${REQ_HANDOFF_CPUS} --mem=${REQ_HANDOFF_MEM} -J stagec_repair -o logs/stagec_repair_%j.out --wrap \"bash ${SELF} --repair-bbl <tag>\""
}
check_bbl () {   # check_bbl <tag>: the fit is the recorded one, and the tagged cost params are whole
    local t="$1" k rec f cp ne nf promoted
    [[ "${t}" == "${MS_TAG}" || "${t}" == "${SC_TAG}" ]] || die "--check-bbl takes ${MS_TAG} or ${SC_TAG} (got '${t}')"
    for k in ${BBL_ROUTINES}; do
        rec="$(fit_rec "${k}" "${t}")"; f="$(res_jls "${k}" "${BBL_STAGE}")"; cp="${CL_STEP_BBL}/cost_params_$(bbl_key "${k}" "${t}").json"
        [[ -f "${rec}" ]] || die "no fit record ${rec##*/}: these BBL files cannot be tied to a fit"
        [[ -f "${f}" && "$(sha "${f}")" == "$(json_str "${rec}" fit_sha256)" ]] || die "${f##*/} is not the fit ${rec##*/} records (replaced or missing since the BBL launch): the BBL results are NOT archived as its results"
        [[ -f "${cp}" ]] || die "${cp##*/} is missing: the solve did not finish (logs/bbl_solve_*)"
        ne="$(grep -o '"n_shards_expected"[[:space:]]*:[[:space:]]*[0-9]*' "${cp}" | grep -o '[0-9]*$' | head -1)"
        nf="$(grep -o '"n_shards_found"[[:space:]]*:[[:space:]]*[0-9]*' "${cp}" | grep -o '[0-9]*$' | head -1)"
        [[ -n "${ne}" && "${ne}" == "${nf}" ]] || die "${cp##*/}: n_shards_found ${nf:-?} != n_shards_expected ${ne:-?}"
        promoted="$(grep -o '"promoted"[[:space:]]*:[[:space:]]*[a-z]*' "${cp}" | grep -o '[a-z]*$' | head -1)"
        say "   E${k} ${t}: fit bound (sha256 $(json_str "${rec}" fit_sha256 | cut -c1-12)...), ${nf}/${ne} shards, promoted=${promoted:-n/a}"
    done
    say "STAGE C CHECK-BBL ${t} OK"
}
# --repair-bbl <tag>: the BOUND repair of a BBL run that left shards or a solve unfinished (the dx
# suite's repair, carried over). The fit on disk must be the one the run's records name; routines
# whose tagged cost params exist are left alone; bbl_run.sh --repair re-runs only what is missing,
# with --no-tables --no-zip, and this suite archives afterany the new solves after --check-bbl.
repair_bbl () {
    local t="$1" k key rec f todo="" lg solves z flags
    [[ "${t}" == "${MS_TAG}" || "${t}" == "${SC_TAG}" ]] || die "--repair-bbl takes ${MS_TAG} or ${SC_TAG} (got '${t}')"
    say "stage C BBL repair of ${t}: data root ${CL_DATA_ROOT}$([[ "${DRY}" == "1" ]] && echo '  [DRY RUN]')"
    env_check chain; check_arrival code; frozen_check
    cpu_image_gate_or_die "the BBL repair" "cd ${HERE} && sbatch --parsable -p ${REQ_HANDOFF_PART} -t ${REQ_HANDOFF_TIME} -c ${REQ_HANDOFF_CPUS} --mem=${REQ_HANDOFF_MEM} -J stagec_repair -o logs/stagec_repair_%j.out --wrap \"bash ${SELF} --repair-bbl ${t}\""
    for k in ${BBL_ROUTINES}; do
        key="$(bbl_key "${k}" "${t}")"; rec="$(fit_rec "${k}" "${t}")"; f="$(res_jls "${k}" "${BBL_STAGE}")"
        [[ -f "${rec}" ]] || die "no fit record ${rec##*/}: the files of ${key} cannot be tied to a fit, so they are not repaired"
        [[ -f "${f}" && "$(sha "${f}")" == "$(json_str "${rec}" fit_sha256)" ]] || die "${f##*/} is not the fit ${rec##*/} records: the shards of ${key} were simulated on another fit. Archive and move them aside; the BBL then needs a fresh launch (--next-bbl)."
        if [[ -f "${CL_STEP_BBL}/cost_params_${key}.json" ]]; then say "   E${k}: cost params on disk; nothing to repair"; continue; fi
        [[ -f "$(cl_bbl_dispatch_dir "${key}")/context.env" ]] || die "E${k}: no launch context for ${key}: nothing to repair (launch it with --next-bbl)"
        todo="${todo:+${todo} }${k}"
    done
    [[ -n "${todo}" ]] || { echo "STAGE C BBL ${t}: NOTHING TO REPAIR (cost params on disk for E${BBL_ROUTINES// /, E})"; return 0; }
    flags=(--routines "${todo}" --shards 300 --psi-tag "${t}" --repair --partitions gpu_h100 --pack-h100 "${REQ_BBL_PACK_H100}" --mem-h100 "${REQ_BBL_MEM_H100}" --shard-time "${REQ_BBL_SHARD_TIME}" --no-tables --no-zip)
    [[ "${t}" == "${MS_TAG}" ]] && flags+=(--promote)
    flags+=(${BBL_EXTRA[@]+"${BBL_EXTRA[@]}"})
    lg="${LOGD}/stagec_bbl_repair${t}_${STAMP}.log"
    env POLICY_CSV="${POLICY_CSV}" CL_DRYRUN="${DRY}" CL_VERBOSE=1 bash "${HERE}/bbl_run.sh" "${flags[@]}" > "${lg}" 2>&1 \
        || { tail -30 "${lg}"; die "the repair of ${t} was refused (${lg}); bbl_run.sh withdrew what it had submitted."; }
    solves="$(sed -n 's/^BBL_SOLVE_JOBIDS=//p' "${lg}" | tail -1)"
    say "   repair of E${todo// /, E}: $(sed -n 's/^BBL_JOB_NAMES=//p' "${lg}" | tail -1)"
    [[ -n "${solves}" ]] || die "the repair returned no solve job id (${lg})"
    HELD=""; held_sweep_check "${lg}" "${t}"
    z=$(sub "stagec_bbl_zip${t}" "afterany:${solves}" --partition="${ZIP_PARTITION:-${REQ_ZIP_PART}}" --time="${REQ_ZIP_TIME}" --nodes=1 --ntasks=1 --cpus-per-task="${REQ_ZIP_CPUS}" --mem="${REQ_ZIP_MEM}" ${REQ_ZIP_PIN:+--constraint="${REQ_ZIP_PIN}"} \
        --wrap "cd '${HERE}' && bash ${SELF} --check-bbl ${t} && bash cluster_archive.sh --set bbl --copy --slim --psi-tag '${t}' --tag \"\${SLURM_JOB_ID}\"") \
        || die "the archive job was refused; the repair stays submitted. By hand later: cd ${HERE} && bash ${SELF} --check-bbl ${t} && bash cluster_archive.sh --set bbl --copy --slim --psi-tag ${t} --tag <jobid>"
    if [[ "${DRY}" != "1" ]]; then for k in ${BBL_ROUTINES}; do fit_record_add_archive "${k}" "${t}" "${z}" || say "   [!] could not add ${z} to $(basename "$(fit_rec "${k}" "${t}")")"; done; fi
    echo "STAGE C BBL REPAIR SUBMITTED$([[ "${DRY}" == "1" ]] && echo ' [DRY RUN]'): ${t}, E${todo// /, E}; solves ${solves}; archive ${z} (afterany, --check-bbl first)"
}
final () {
    say "stage C final: data root ${CL_DATA_ROOT}"
    if [[ -f "${FROZEN_REC}" ]] && sha256sum -c --quiet "${FROZEN_REC}"; then gate_json final true "frozen set unchanged: the sleepiness estimates, the policy, the panel and the demand-characteristics file are the installed ones"
        say "STAGE C FINAL: the frozen set is unchanged"
    else gate_json final false "a frozen file changed or the record is missing (${FROZEN_REC})"; die "a frozen file changed since stagec_install"; fi
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# THE JULIA PACKAGE SET (D7): Manifest-v1.11.toml, resolved locally under Julia 1.11.4 from the
# local 1.12.6 set (172 of 173 packages at the same version; PrecompileTools 1.3.3 -> 1.2.1, the
# newest Julia 1.11 accepts) and validated there (stage C check c identical). Julia 1.11 reads it
# before Manifest.toml. Nothing here ever calls Pkg.resolve: env_julia replaces env_job.sh's
# resolve step (which always resolves) with instantiate + precompile of the shipped file.
# ══════════════════════════════════════════════════════════════════════════════════════════
jl_manifest_check () {   # the manifest Julia reads is the shipped one, for the loaded Julia
    local mf v mv
    v="$(julia --startup-file=no -e 'print(VERSION.major, ".", VERSION.minor)')" || die "julia does not run (module ${JULIA_MODULE})"
    mf="$(cl_manifest_path || true)"
    [[ "${mf##*/}" == "${JL_MANIFEST}" ]] || die "the loaded Julia ${v} reads ${mf##*/}, not the shipped ${JL_MANIFEST}, which is for Julia 1.11 (JULIA_MODULE=${JULIA_MODULE}). Load a Julia 1.11 module (export JULIA_MODULE=<a 1.11 module>) and submit again."
    mv="$(cl_manifest_julia_version "${mf}" || true)"
    [[ "${mv}" == "${v}".* ]] || die "${JL_MANIFEST} records julia_version '${mv}'; the loaded Julia is ${v}"
    [[ "$(sha "${mf}")" == "${JL_MANIFEST_SHA256}" ]] || die "${JL_MANIFEST} does not hash to the shipped ${JL_MANIFEST_SHA256:0:12}...: something rewrote it. Re-unzip the code zip (it never changes on the cluster)."
    say "   ${JL_MANIFEST}: julia_version ${mv}, loaded Julia ${v}, sha256 ${JL_MANIFEST_SHA256:0:12}... (as shipped)"
}
julia_env () {   # job (the warm-up's first step): instantiate + precompile the shipped set, never resolve
    local off=0 hint=""
    say "stage C Julia environment: ${HERE} (no Pkg.resolve: the shipped ${JL_MANIFEST} is instantiated as it is)"
    env_check chain
    cl_bootstrap_tree >/dev/null
    cl_load_julia || die "module load ${JULIA_MODULE} failed (above)"
    if [[ "${JULIA_PKG_OFFLINE:-}" == "true" ]]; then
        off=1; hint=". OFFLINE: a package or artifact is missing from the depot; run the login-node fetch again: cd ${HERE} && bash ${SELF} --julia-fetch"
        say "   depot ${JULIA_DEPOT_PATH}; OFFLINE (JULIA_PKG_OFFLINE=true): every package and artifact must already be in it (the login-node --julia-fetch)"
    else say "   depot ${JULIA_DEPOT_PATH}; online"; fi
    jl_manifest_check
    JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no --project="${HERE}" -e '
        using Pkg
        offline = get(ENV, "JULIA_PKG_OFFLINE", "") == "true"
        regs = try Pkg.Registry.reachable_registries() catch; [] end
        if isempty(regs) && !offline
            println("  no registry in this depot: adding General (to locate the package sources)")
            Pkg.Registry.add("General")
        end
        Pkg.instantiate(); println("  instantiated (no resolve)")
        Pkg.precompile();  println("  precompiled")
    ' || die "instantiate/precompile of ${JL_MANIFEST} failed (above)${hint}"
    [[ "$(sha "$(cl_manifest_path)")" == "${JL_MANIFEST_SHA256}" ]] || die "${JL_MANIFEST} CHANGED during instantiate/precompile (sha256 now $(sha "$(cl_manifest_path)" | cut -c1-12)...): the package set is no longer the validated one. Nothing after this job runs."
    say "   ${JL_MANIFEST} unchanged after instantiate + precompile"
    julia --startup-file=no --project="${HERE}" -e '
        using MKL, LoopVectorization, LinearAlgebra, DataFrames, Optim, ArgParse, Parquet2, JSON3, Distributions, QuasiMonteCarlo, LBFGSB, CUDA
        println("  every project package loads; BLAS: ", BLAS.get_config(), "; threads ", Threads.nthreads())
        import Pkg; Pkg.status()
    ' || die "the project packages do not load (above)"
    gate_json julia_env true "${JL_MANIFEST} (sha256 ${JL_MANIFEST_SHA256:0:12}) instantiated + precompiled, unchanged; every project package loads${hint:+; offline}"
    say "STAGE C JULIA ENV OK"
}
julia_env_instantiate () {   # the launch (P8): instantiate the shipped set; never resolve, no precompile
    local hint=""
    say "   ${JL_MANIFEST} instantiated as it is (no Pkg.resolve; no precompile: pkgimages are CPU-specific, each node builds its own)"
    cl_bootstrap_tree >/dev/null
    cl_load_julia || die "module load ${JULIA_MODULE} failed (above)"
    if [[ "${JULIA_PKG_OFFLINE:-}" == "true" ]]; then
        hint=". OFFLINE: a package or artifact is missing from the depot; run the login-node fetch again: cd ${HERE} && bash ${SELF} --julia-fetch"
        say "   depot ${JULIA_DEPOT_PATH}; OFFLINE (JULIA_PKG_OFFLINE=true): every package and artifact must already be in it"
    else say "   depot ${JULIA_DEPOT_PATH}; online"; fi
    jl_manifest_check
    JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no --project="${HERE}" -e '
        using Pkg
        offline = get(ENV, "JULIA_PKG_OFFLINE", "") == "true"
        regs = try Pkg.Registry.reachable_registries() catch; [] end
        if isempty(regs) && !offline
            println("  no registry in this depot: adding General (to locate the package sources)")
            Pkg.Registry.add("General")
        end
        Pkg.instantiate(); println("  instantiated (no resolve, no precompile)")
    ' || die "instantiate of ${JL_MANIFEST} failed (above)${hint}"
    [[ "$(sha "$(cl_manifest_path)")" == "${JL_MANIFEST_SHA256}" ]] || die "${JL_MANIFEST} CHANGED during instantiate (sha256 now $(sha "$(cl_manifest_path)" | cut -c1-12)...): the package set is no longer the validated one. Nothing was submitted."
    gate_json julia_env true "${JL_MANIFEST} (sha256 ${JL_MANIFEST_SHA256:0:12}) instantiated, unchanged; each node precompiles for itself${hint:+; offline}"
    say "   the Julia environment is ready"
}
# --julia-fetch (LOGIN NODE, variant (ii): the compute nodes have no internet). Downloads only
# (JULIA_PKG_PRECOMPILE_AUTO=0): the registry if the depot has none, every package and artifact of
# the shipped set, and the LAZY CUDA artifacts the GPU nodes select at load time from their driver
# (CUDA_Runtime_jll / CUDA_Compiler_jll; the newest toolkit of each CUDA major in STAGEC_CUDA_MAJORS,
# default "12 13": the runtime picks the newest toolkit whose major the driver supports). Then the
# launch runs with STAGEC_JULIA_OFFLINE=1, so no job touches the network.
julia_fetch () {
    say "stage C Julia fetch (login node): downloads into ${HERE}/.julia_depot, no precompile"
    [[ -z "${SLURM_JOB_ID:-}" ]] || die "--julia-fetch runs on the LOGIN node (it needs the internet the compute nodes lack); this is job ${SLURM_JOB_ID}"
    env_check chain; check_arrival code
    unset JULIA_PKG_OFFLINE
    cl_load_julia || die "module load ${JULIA_MODULE} failed (above)"
    jl_manifest_check
    STAGEC_CUDA_MAJORS="${STAGEC_CUDA_MAJORS:-12 13}" JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no --project="${HERE}" -e '
        using Pkg, TOML
        if isempty(try Pkg.Registry.reachable_registries() catch; [] end)
            Pkg.Registry.add("General")
        end
        Pkg.instantiate(); println("  instantiated: every package and eager artifact is in the depot")
        majors = split(ENV["STAGEC_CUDA_MAJORS"])
        for (name, uuid) in (("CUDA_Runtime_jll", "76a88914-d11a-5bdc-97e0-2f5a05c973a2"), ("CUDA_Compiler_jll", "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"))
            src = Base.locate_package(Base.PkgId(Base.UUID(uuid), name))
            src === nothing && error("$(name) is not in the instantiated set")
            toml = joinpath(dirname(dirname(src)), "Artifacts.toml")
            for (aname, entries) in TOML.parsefile(toml)
                linux = [e for e in entries if get(e, "os", "") == "linux" && get(e, "arch", "") == "x86_64" &&
                         get(e, "libc", "") == "glibc" && haskey(e, "cuda") && !haskey(e, "cuda_platform")]
                for mj in majors
                    es = [e for e in linux if first(split(e["cuda"], ".")) == mj]
                    isempty(es) && (println("  $(name)/$(aname): no CUDA $(mj).x artifact"); continue)
                    e = es[argmax([VersionNumber(x["cuda"]) for x in es])]
                    println("  $(name)/$(aname) cuda=$(e["cuda"]) (lazy) ", get(ENV, "STAGEC_FETCH_LIST_ONLY", "0") == "1" ? "[list only]" : "")
                    get(ENV, "STAGEC_FETCH_LIST_ONLY", "0") == "1" && continue
                    Pkg.Artifacts.ensure_artifact_installed(aname, e, toml)
                end
            end
        end
    ' || die "the fetch failed (above); nothing was submitted. Run it again (it resumes)."
    [[ "$(sha "$(cl_manifest_path)")" == "${JL_MANIFEST_SHA256}" ]] || die "${JL_MANIFEST} CHANGED during the fetch: re-unzip the code zip"
    echo
    echo "STAGE C JULIA FETCH DONE: the depot holds the shipped package set and the CUDA ${STAGEC_CUDA_MAJORS:-12 13}.x runtime/compiler artifacts."
    echo "THE LAUNCH, OFFLINE (one line; it prints the job id and returns):"
    echo "  $(STAGEC_JULIA_OFFLINE=1 launch_line)"
    echo "THEN (one line each):  squeue -u \$USER -h -o \"%.10i %.28j %.9T %.22r %.10M\"   and   tail -n 40 ${HERE}/logs/stagec_launch_<job id>.out"
}

# ══════════════════════════════════════════════════════════════════════════════════════════
# m6: every log into the logs archive. The logs set takes logs/*.out, *.err and orchestrator_*.log;
# the wrapper's sub-logs (stagec_*.log), the smoke's engine logs (the draws-gate evidence), the
# preflight logs and the smoke's stray summaries are copied beside them as <name>.out first.
# ══════════════════════════════════════════════════════════════════════════════════════════
collect_logs () {
    local f d n=0
    for f in "${LOGD}"/*.log; do
        [[ -f "${f}" ]] || continue
        case "${f##*/}" in orchestrator_*) continue ;; esac
        cp -f "${f}" "${f}.out" && n=$((n + 1))
    done
    for d in "${LOGD}"/stagec_smoke_*_strays; do
        [[ -d "${d}" ]] || continue
        for f in "${d}"/*; do [[ -f "${f}" ]] && cp -f "${f}" "${LOGD}/${d##*/}__${f##*/}.out" && n=$((n + 1)); done
    done
    say "   collected ${n} file(s) into ${LOGD} as *.out for the logs archive"
}

case "${MODE}" in
    install)    install_mode; exit 0 ;;
    chain)      chain_mode; exit 0 ;;
    gpu)        gpu_mode; exit 0 ;;
    cpu-image)  cpu_image_mode; exit 0 ;;
    next-blp)   next_blp; exit 0 ;;
    check-blp)  check_blp "${MODE_ARG}"; exit 0 ;;
    next-bbl)   next_bbl; exit 0 ;;
    check-bbl)  check_bbl "${MODE_ARG}"; exit 0 ;;
    repair-bbl) repair_bbl "${MODE_ARG}"; exit 0 ;;
    final)      final; exit 0 ;;
    julia-env)  julia_env; exit 0 ;;
    julia-fetch) julia_fetch; exit 0 ;;
    collect-logs) collect_logs; exit 0 ;;
    resume)     resume_mode; exit 0 ;;
    smoke-verdict) read -r _k _lg <<< "${MODE_ARG}"; smoke_verdict "${_k}" "${_lg}"; [[ -z "${SMOKE_BAD}" ]] && { echo "VERDICT PASS"; exit 0; }; echo "VERDICT FAIL:${SMOKE_BAD}"; exit 1 ;;
esac

# ══════════════════════════════════════════════════════════════════════════════════════════
# THE LAUNCH: checks, then the warm-up, then the prep hand-off
# ══════════════════════════════════════════════════════════════════════════════════════════
py_env_probe () {   # L4 (MAJOR-1/D6): the env imports the stack the estimators and the prep need
    local out rc
    # A subshell, like cl_python_probe: module/conda cannot leak into this shell.
    out="$( (
        if [[ -n "${PY_MODULE:-}" ]]; then module load ${PY_MODULE} || true; fi
        if [[ -n "${CONDA_ENV:-}" ]]; then source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}" || true; fi
        cd "${HERE}" && OPEN_FINANCE_VENV_ENFORCED=1 "${CF_PYTHON}" -c '
import importlib, sys
sys.path.insert(0, ".")
for m in sys.argv[1].split(","):
    importlib.import_module(m.strip())
import utils.sleep_links
import numpy, pandas, statsmodels
print("  imports ok: pandas", pandas.__version__, "| numpy", numpy.__version__, "| statsmodels", statsmodels.__version__)
' "${PY_REQ}"
    ) 2>&1 )"; rc=$?
    printf '%s\n' "${out}" | tail -8 | sed 's/^/   /'
    return ${rc}
}

STOP_RESUME="$(launch_line)"   # a refusal of the launch prints the launch line again
say "stage C launch: project ${PROJ}$([[ "${DRY}" == "1" ]] && echo '  [DRY RUN: nothing is copied, written or submitted]')"
say "L0. the environment"; env_check launch
say "L1. the arrival"; check_arrival all
say "L2. the queue"
if [[ "${DRY}" == "1" ]]; then say "   [dry-run] not checked"; else queue_check; fi
say "L3. the output guard"; output_guard "launch"; input_delta_guard
say "L4. the Python env (CONDA_ENV='${CONDA_ENV:-}'): imports ${PY_REQ}"
if [[ "${STAGEC_SKIP_PYPROBE:-0}" == "1" ]]; then say "   [test] probe skipped"
elif py_env_probe; then say "   '${CONDA_ENV:-<no conda>}' imports the stack"
else die "the Python env '${CONDA_ENV:-<no conda>}' cannot run the sleepiness estimators and the prep (the import failing above). Create the pinned env once (it needs the internet; one line, then launch again): cd ${HERE} && sbatch --parsable -p day -t 02:00:00 -c 4 --mem=16G -J env_conda -o logs/env_conda_%j.out --wrap \"${CONDA_CREATE}\"  -- if the day nodes cannot reach conda-forge, run the part in quotes on the login node instead. An existing env with older pandas: create the pinned one under another name and launch with --export=ALL,CONDA_ENV=<that name>."; fi
say "L5. the submissions, dry run first (T4 the CPU image; T3 the CPU chain; T2 the GPU job afterok T3)$([[ "${STAGEC_JULIA_OFFLINE:-0}" == "1" ]] && echo '  OFFLINE: JULIA_PKG_OFFLINE=true in every job; the depot was filled by --julia-fetch')"
( export CL_DRYRUN=1; MAIN=""; submit_launch ) || die "the dry run of the submissions failed (above). Nothing was submitted."
if [[ "${DRY}" == "1" ]]; then
    echo; echo "STAGE C DRY RUN COMPLETE: nothing was copied, written or submitted."; exit 0
fi
# P8: the install and the Julia environment run HERE, before anything is submitted; a failure stops
# the launch with nothing queued (relaunch with the same line: the install never overwrites an
# identical file, and the output guard ignores what the install wrote). Subshells: their exports
# (step folders, the Julia depot) must not reach the submitted jobs' environment.
say "L6. the install (policy, frozen set)"
( install_mode ) || die "the install failed (above); nothing was submitted. Once fixed, launch again with the same line."
say "L7. the Julia environment (instantiate only; each node precompiles for itself)"
( run_payload julia_env julia_env_instantiate ) || die "the Julia environment failed (above); nothing was submitted. Once fixed, launch again with the same line."
say "L8. submitting"
MAIN=""
submit_launch || { [[ -n "${MAIN}" ]] && scancel ${MAIN//:/ } 2>/dev/null; die "a submission was refused; every job already submitted was cancelled (${MAIN:-none})."; }
echo
echo "STAGE C SUBMITTED: demand fixes (demand_chars) + D in the main logit/BLP; prep+logit+BLP E1-E4; BBL ${MS_TAG} (promote) + ${SC_TAG} on E3/E4, both --no-tables; no counterfactuals"
print_watch_lines
echo "GATES (one line):  cd ${CL_DATA_OUT} && for g in .gate_SC_install .gate_SC_julia_env .gate_SC_cpu_image .gate_G0 .gate_G2 .gate_SC_prep .gate_G4 .gate_SC_logit .gate_SC_smoke .gate_SC_blp .gate_SC_blp_E34 .gate_SC_bbl .gate_SC_final; do [ -f \$g.json ] && { echo \$g; grep -E '\"ok\"|detail' \$g.json; }; done"
