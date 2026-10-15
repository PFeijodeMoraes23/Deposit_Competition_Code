#!/bin/bash
# test_stagec.sh — the stage C wrapper (stagec_suite_20261012.sh) against stub SLURM, in a sandbox
# OUTSIDE OneDrive (default C:/sc_tests; CLUSTER_TEST_ROOT overrides). Real cluster scripts copied
# from the repo (CR stripped), fake inputs, the real policy file; every payload (sleep_job.sh,
# logit_job.sh, blp_draws_job.sh, env_job.sh, cluster_archive.sh, the smoke's Julia) goes through
# stubs/payload_runner.sh, which records it and fakes its outputs.
#   A  the launch (T1): dry run; live: install + Julia env + T3 (chain) + T2 (GPU) afterok T3, the
#      REQ_* requests, no cpugen pin; refusals
#   B  --install, --final
#   C  --chain (T3): every step, the archive always, exit codes, step restarts (--from), the scoped
#      mismatch override
#   R  --resume chain [--from] | gpu | next-blp
#   S  the smoke verdict; --gpu (T2): build, merged gate, one-process smoke, split verdict, the BLP
#      hand-off in its tail; --collect-logs
#   P  PAYLOAD EQUIVALENCE: every payload command and its relevant environment against the
#      DEPLOYED design (its scripts: DEPLOYED_SCRIPTS, the bundle that ran on 10-13/14)
#   D  --next-blp, --check-blp, --next-bbl, --check-bbl, --repair-bbl, --resume blp (slow: bbl_run.sh)
# Usage: bash tests/cluster/test_stagec.sh [section letters, default "A B C R S P D"]
#        T2_D_ONLY=3ef|456 narrows D.
set -uo pipefail
HERE_T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE_T}/../.." && pwd)"
STUBS="${HERE_T}/stubs"
SBX="${CLUSTER_TEST_ROOT:-/c/sc_tests}"
T="${SBX}/logs"; mkdir -p "${T}"                       # per-check logs (the D block's ${T}/<x>.log)
STATE="${SBX}/state"
EO="/c/Users/pedro/OneDrive/Documentos/Yale/Year 3 (2024 - 2025)/Open Finance/Open-Finance/BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT"
PYW="C:/venvs/egan/Scripts/python.exe"
DEPLOYED_SCRIPTS="${DEPLOYED_SCRIPTS:-/c/Users/pedro/.claude/projects/c--Users-pedro-OneDrive-Documentos-Yale-Year-3--2024---2025--Open-Finance-Open-Finance-Code-Egan-et-al-2025-Rep/checkpoints_20260928/stageC_launch_20261012/bundle_final_20261013/unpacked/scripts}"
SB="${SBX}/sbx"; S="${SB}/scripts"; D="${SB}/data"
SECTIONS="${1:-A B C R S P D}"
PASS=0; FAIL=0
ok ()  { echo "  ok   $*"; PASS=$((PASS+1)); }
bad () { echo "  FAIL $*"; FAIL=$((FAIL+1)); return 1; }
check () { local what="$1"; shift; if "$@"; then ok "${what}"; else bad "${what}"; fi; }
checke () { local what="$1"; shift; if eval "$1"; then ok "${what}"; else bad "${what}"; fi; }   # one condition string (&&, ||)
has () { grep -q -- "$1" "$2"; }
hasn () { ! grep -q -- "$1" "$2"; }

POLICY_SRC="${EO}/COST_POLFUNC/polfunc_fitted.csv"
mkpickles () {
    "${PYW}" -c 'import pickle, sys, pandas as pd
for k, p in enumerate(sys.argv[1:], 1):
    pickle.dump({"IV_HausmanFull x Tech": {"second_stage": pd.DataFrame({"a": [float(k)], "s": ["x"]})}}, open(p, "wb"))' "$@"
}
# build_sandbox [src scripts dir] [wrapper]: the repo's scripts (default) or the deployed ones (P)
build_sandbox () {
    local SRC="${1:-${REPO}}" WR="${2:-${REPO}/stagec_suite_20261012.sh}" f
    rm -rf "${SB}"; mkdir -p "${S}/logs" "${S}/config" "${S}/utils" "${D}/input"
    for f in cluster_lib.sh sleep_run.sh sleep_job.sh logit_job.sh blp_draws_job.sh blp_run.sh blp_stage_job.sh \
             bbl_run.sh bbl_job.sh cluster_archive.sh env_job.sh bbl_discount.env; do
        tr -d '\r' < "${SRC}/${f}" > "${S}/${f}"
    done
    for f in stagec_requests.env blp_smoke.jl; do [[ -f "${SRC}/${f}" ]] && tr -d '\r' < "${SRC}/${f}" > "${S}/${f}"; done
    tr -d '\r' < "${SRC}/config/routines.toml" > "${S}/config/routines.toml"
    tr -d '\r' < "${WR}" > "${S}/stagec_suite_20261012.sh"
    : > "${S}/utils/__init__.py"; echo "# stub of utils/sleep_links.py (the probe imports it)" > "${S}/utils/sleep_links.py"
    echo "manifest-as-shipped (1.12.6)" > "${S}/Manifest.toml"
    printf '# test\njulia_version = "1.11.4"\nmanifest_format = "2.0"\n' > "${S}/Manifest-v1.11.toml"
    echo "panel" > "${D}/input/market_panel.parquet"
    echo "chars" > "${D}/input/demand_chars_T.parquet"
    echo "sigma 10-14" > "${D}/input/demographics_sigma.parquet"
    cp "${POLICY_SRC}" "${D}/input/polfunc_fitted_liverows_20260930.csv"
    for f in forward_rf_qoq.csv forward_rf_vintages.csv bbl_transitions.json sleep_link_E3_spec_12.json sleep_link_E4_spec_12.json; do
        cp "${EO}/COST_FWD/${f}" "${D}/input/${f}"
    done
    mkpickles "${D}"/input/sleep_est_E{1,2,3,4}_estimation_results.pkl
    for k in 1 2 3 4; do printf '{"key_kept": "IV_HausmanFull x Tech", "subkeys_dropped": ["first_stage"], "routine": %s}\n' "${k}" > "${D}/input/sleep_est_E${k}_spec12_slim.json"; done
    ( cd "${SB}" && sha256sum scripts/cluster_lib.sh scripts/sleep_run.sh scripts/stagec_suite_20261012.sh scripts/Manifest.toml scripts/Manifest-v1.11.toml data/input/* ) > "${S}/stagec_arrival_T.sha256"
    local est="" mk=""
    for k in 1 2 3 4; do
        est="${est} ${k}:sleep_est_E${k}_estimation_results.pkl:$(sha256sum "${D}/input/sleep_est_E${k}_estimation_results.pkl" | cut -d' ' -f1)"
        mk="${mk} ${k}:sleep_est_E${k}_spec12_slim.json:$(sha256sum "${D}/input/sleep_est_E${k}_spec12_slim.json" | cut -d' ' -f1)"
    done
    cat > "${S}/stagec_pins_20261012.env" <<EOF
ARRIVAL_LIST=stagec_arrival_T.sha256
CHARS_NAME=demand_chars_T.parquet
CHARS_SHA256=$(sha256sum "${D}/input/demand_chars_T.parquet" | cut -d' ' -f1)
SIGMA_SHA256=$(sha256sum "${D}/input/demographics_sigma.parquet" | cut -d' ' -f1)
POLICY_UPLOAD=polfunc_fitted_liverows_20260930.csv
EST_UPLOADS=(${est# })
EST_MARKERS=(${mk# })
JL_MANIFEST_SHA256=$(sha256sum "${S}/Manifest-v1.11.toml" | cut -d' ' -f1)
PY_REQ="numpy, pandas"
CONDA_CREATE="module load miniconda && conda create -y -n dep_comp_blp --override-channels --strict-channel-priority -c conda-forge python=3.11 numpy pandas"
EXPECT_ROWS=(1:10 2:10 3:10 4:10)
EXPECT_ALPHA=(E3_full_dtype:-0.0628 E4_full_dtype:-0.0567)
EOF
}
reset_stubs () { rm -rf "${STATE}"; mkdir -p "${STATE}"; : > "${STATE}/squeue.txt"; }
run () {   # run <log> <mode-args...> [-- env...]: the wrapper in the sandbox, stubs first on PATH
    local lg="$1"; shift; local args=() envs=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do args+=("$1"); shift; done
    [[ "${1:-}" == "--" ]] && { shift; envs=("$@"); }
    ( cd "${S}" && env PATH="${STUBS}:${PATH}" STUB_DIR="${STATE}" CL_ROOT="${S}" DATA_ROOT="${D}" \
          PY_MODULE="" CONDA_ENV="" CF_PYTHON="${PYW}" RUNNER_PY="${PYW}" STAGEC_PAYLOAD_RUNNER="${STUBS}/payload_runner.sh" \
          STAGEC_BLP_EXTRA="--skip-preflight" STAGEC_BBL_EXTRA="--skip-preflight" PY_PREFLIGHT=0 STAGEC_LOCAL_TEST=1 \
          ${envs[@]+"${envs[@]}"} bash stagec_suite_20261012.sh ${args[@]+"${args[@]}"} ) > "${lg}" 2>&1
}
SBL="${STATE}/sbatch.log"; PLOG="${STATE}/payload.log"
dep_of () { grep -- "-J $1 " "${SBL}" | grep -o -- '--dependency=[a-z]*:[0-9:]*' | head -1; }
id_of ()  { awk -v n="$1" '{ for (i = 1; i <= NF; i++) if ($i == "-J" && $(i+1) == n) print NR + 7000000 }' "${SBL}" | head -1; }
ln_of ()  { grep -n -- "$1" "${SBL}" | head -1 | cut -d: -f1; }
okgate () { printf '{\n "gate": "%s",\n "ok": true,\n "checks": [{"check": "x", "ok": true}]\n}\n' "$1" > "${D}/output/.gate_$1.json"; }
tags () { cut -f1 "${PLOG}" 2>/dev/null | tr '\n' ' '; }
mkparquets () {   # the prep's outputs, without the runner (for gate-only cases)
    ( cd "${S}" && env STUB_DIR="${STATE}" STAGEC_PAYLOAD_TAG=prep CL_DATA_ROOT="${D}" RUNNER_PY="${PYW}" RUNNER_ROWS="$*" \
        bash "${STUBS}/payload_runner.sh" true >/dev/null )
}
mkdraws () { mkdir -p "${D}/output/blp/draws"; for f in demo_draws demo_key_index halton_nu; do echo d > "${D}/output/blp/draws/${f}_R2000_seed42.jls"; done; }
REQ () { ( . "${REPO}/stagec_requests.env"; eval "printf '%s' \"\${$1}\"" ); }

# ─────────────────────────────────────────────────────────────────────────────────────────
if [[ " ${SECTIONS} " == *" A "* ]]; then
echo "== A1 dry run on an EMPTY tree"
build_sandbox; reset_stubs
run "${T}/a1.log" -- CL_DRYRUN=1 STAGEC_SKIP_PYPROBE=1; rc=$?
check "A1 exit 0" test ${rc} -eq 0 || tail -20 "${T}/a1.log"
check "A1 completes" has "STAGE C DRY RUN COMPLETE" "${T}/a1.log"
check "A1 no live sbatch, no install, no payload" test ! -s "${SBL}" -a ! -e "${D}/output" -a ! -s "${PLOG}"
order="$(grep -o -- '-J [a-zA-Z_0-9]*' "${T}/a1.log" | awk '{print $2}' | tr '\n' ' ')"
check "A1 the graph: stagec_chain, stagec_gpu (no env_*, no G0 job, no CPU sysimage)" test "${order}" == "stagec_chain stagec_gpu "

echo "== A2 the launch, live (stub SLURM)"
build_sandbox; reset_stubs
run "${T}/a2.log" -- STAGEC_SKIP_PYPROBE=1; rc=$?
check "A2 exit 0" test ${rc} -eq 0 || tail -25 "${T}/a2.log"
check "A2 installed in the launch (P8): 4 estimates + 4 markers, frozen set 11" test "$(ls "${D}"/output/sleep/est*/estimation_results.pkl* 2>/dev/null | wc -l)" -eq 8 -a "$(grep -c . "${D}/output/.stagec_frozen_20261012.sha256")" -eq 11
check "A2 the Julia env ran in the launch (P8), before any submission" test "$(tags)" == "julia_env "
check "A2 exactly 2 submissions" test "$(grep -c . "${SBL}")" -eq 2
C3="$(id_of stagec_chain)"; G2="$(id_of stagec_gpu)"
check "A2 T3 first, no dependency" test "$(head -1 "${SBL}" | grep -c -- '-J stagec_chain ')" -eq 1 -a -z "$(dep_of stagec_chain)"
check "A2 T2 afterok T3" test "$(dep_of stagec_gpu)" == "--dependency=afterok:${C3}"
grep -- "-J stagec_chain " "${SBL}" | grep -q -- "--partition=$(REQ REQ_CHAIN_PART) --time=$(REQ REQ_CHAIN_TIME) --nodes=1 --ntasks=1 --cpus-per-task=$(REQ REQ_CHAIN_CPUS) --mem=$(REQ REQ_CHAIN_MEM) --wrap cd '.*' && bash stagec_suite_20261012.sh --chain$" && ok "A2 T3 request = REQ_CHAIN_* (day 8c/96G/2h), --chain" || bad "A2 T3 request"
grep -- "-J stagec_gpu " "${SBL}" | grep -q -- "--partition=$(REQ REQ_GPU_PART) --gpus=$(REQ REQ_GPU_GRES) --time=$(REQ REQ_GPU_TIME) --nodes=1 --ntasks=1 --cpus-per-task=$(REQ REQ_GPU_CPUS) --mem=$(REQ REQ_GPU_MEM) --wrap cd '.*' && bash stagec_suite_20261012.sh --gpu$" && ok "A2 T2 request = REQ_GPU_* (gpu_h200 h200:1 8c/360G/2h), --gpu" || bad "A2 T2 request"
check "A2 no cpugen pin on any submission" bash -c "! grep -q -- 'cpugen\|--constraint\|-C ' '${SBL}'"
check "A2 the launch's Julia payload is julia_env_instantiate (the CPU-sysimage-free env step)" has "^julia_env	julia_env_instantiate$" "${PLOG}"

echo "== A3 refusals"
build_sandbox; reset_stubs
run "${T}/a3a.log" -- STAGEC_SKIP_PYPROBE=1 RUNNER_FAIL=julia_env; rc=$?
checke "A3a a failing Julia env stops the launch with nothing submitted" 'test ${rc} -ne 0 -a ! -s "${SBL}" && has "the Julia environment failed" "${T}/a3a.log"'
check "A3a ... and says to launch again" has "launch again with the same line" "${T}/a3a.log"
build_sandbox; reset_stubs
run "${T}/a3b.log" -- CF_PYTHON=false
check "A3b a Python env that cannot run the prep is refused (create line printed)" has "cannot run the prep" "${T}/a3b.log"
run "${T}/a3c.log" -- STAGEC_SKIP_PYPROBE=1 STAGEC_LOCAL_TEST=0
check "A3c test knobs outside the local test are refused" has "would change results:.*(local tests only)" "${T}/a3c.log"
run "${T}/a3d.log" -- STAGEC_SKIP_PYPROBE=1 BLP_TWO_START_ALLOW_SINGLE=1 R=500
check "A3d result-changing variables refused" bash -c "grep -q \"BLP_TWO_START_ALLOW_SINGLE='1'\" '${T}/a3d.log' && grep -q \"R='500'\" '${T}/a3d.log'"
echo tampered >> "${S}/Manifest-v1.11.toml"
run "${T}/a3e.log" -- STAGEC_SKIP_PYPROBE=1
check "A3e L1 checks Manifest-v1.11.toml" has "scripts/Manifest-v1.11.toml: FAILED" "${T}/a3e.log"
build_sandbox; reset_stubs; mkdir -p "${D}/output/blp"; echo x > "${D}/output/blp/blp_results_E3_spec_12_ext1.jls"
run "${T}/a3f.log" -- STAGEC_SKIP_PYPROBE=1
check "A3f output guard (M3)" has "data/output already holds results of this chain (launch): output/blp/blp_results_E3_spec_12_ext1.jls" "${T}/a3f.log"
rm -rf "${D}/output"; echo x > "${D}/input/logit_delta_E3_spec_12.bin"
run "${T}/a3g.log" -- STAGEC_SKIP_PYPROBE=1
check "A3g a logit delta in data/input refuses (m5)" has "data/input holds logit_delta_E3_spec_12.bin" "${T}/a3g.log"
rm -f "${D}/input/logit_delta_E3_spec_12.bin"; echo "4242 rc_ift_E3_head" > "${STATE}/squeue.txt"
run "${T}/a3h.log" -- STAGEC_SKIP_PYPROBE=1
check "A3h a live project job refuses" has "jobs of this project are live: 4242 rc_ift_E3_head" "${T}/a3h.log"
check "A3 no submission and no install in any refusal" test ! -s "${SBL}" -a ! -e "${D}/output/.stagec_frozen_20261012.sha256"
fi

# ─────────────────────────────────────────────────────────────────────────────────────────
if [[ " ${SECTIONS} " == *" B "* ]]; then
echo "== B --install, --final"
build_sandbox; reset_stubs
run "${T}/b1.log" --install; rc=$?
check "B1 exit 0, 4 estimates + 4 markers, frozen set 11" test ${rc} -eq 0 -a "$(ls "${D}"/output/sleep/est*/estimation_results.pkl* | wc -l)" -eq 8 -a "$(grep -c . "${D}/output/.stagec_frozen_20261012.sha256")" -eq 11
run "${T}/b2.log" --install
check "B2 a second install keeps identical files (9)" test "$(grep -c 'already installed (identical)' "${T}/b2.log")" -eq 9
run "${T}/b3.log" --final
check "B3 final: frozen set unchanged" has "STAGE C FINAL: the frozen set is unchanged" "${T}/b3.log"
echo changed >> "${D}/output/sleep/est3/estimation_results.pkl"
run "${T}/b4.log" --final; rc=$?
checke "B4 final refuses a changed frozen file" 'test ${rc} -ne 0 && has "\"ok\": false" "${D}/output/.gate_SC_final.json"'
build_sandbox; reset_stubs; echo "sigma 08-26" > "${D}/input/demographics_sigma.parquet"
run "${T}/b5.log" --install; rc=$?
checke "B5 a sigma file that is not the pinned one refuses the install" 'test ${rc} -ne 0 && has "demographics_sigma.parquet does not hash to the pinned" "${T}/b5.log"'
run "${T}/b6.log" -- STAGEC_SKIP_PYPROBE=1
checke "B6 ... and the launch (arrival list), before anything is submitted" 'has "data/input/demographics_sigma.parquet: FAILED" "${T}/b6.log" && test ! -s "${SBL}"'
fi

# ─────────────────────────────────────────────────────────────────────────────────────────
if [[ " ${SECTIONS} " == *" C "* ]]; then
echo "== C1 --chain, every step passes"
build_sandbox; reset_stubs; run "${T}/c0.log" --install
run "${T}/c1.log" --chain; rc=$?
check "C1 exit 0" test ${rc} -eq 0 || tail -25 "${T}/c1.log"
check "C1 payload order: precompile g0 prep g2 (gate) logit/draws g4 (gate) archive x2" bash -c "[[ '$(tags | sed 's/draws logit/logit draws/')' == 'precompile g0 prep g2 logit draws g4 archive archive ' ]]"
check "C1 G0 excuses the images and the draws (T2 builds the GPU image)" has 'PF_WILL_BUILD=draws sysimage' "${PLOG}"
check "C1 the gates written ok (SC_prep, SC_logit)" bash -c "grep -q '\"ok\": true' '${D}/output/.gate_SC_prep.json' && grep -q '\"ok\": true' '${D}/output/.gate_SC_logit.json'"
check "C1 archives: demand_prep and logit, tag = the job id" bash -c "grep -q 'cluster_archive.sh --set demand_prep --copy --tag local' '${PLOG}' && grep -q 'cluster_archive.sh --set logit --copy --tag local' '${PLOG}'"
check "C1 PASSED line" has "STAGE C CHAIN: PASSED" "${T}/c1.log"
check "C1 the chain submits nothing (T2 was submitted by the launch)" test ! -s "${SBL}"

echo "== C2 a row mismatch: exit 3, the archive still runs, the scoped resume line"
build_sandbox; reset_stubs; run "${T}/c0.log" --install
run "${T}/c2.log" --chain -- RUNNER_ROWS="10 11 10 10"; rc=$?
check "C2 exit 3 (a mismatch)" test ${rc} -eq 3 || tail -20 "${T}/c2.log"
check "C2 no logit/draws after the stop" bash -c "! grep -qE '^(logit|draws)	' '${PLOG}'"
check "C2 the archive ran anyway (m1)" has "^archive	" "${PLOG}"
check "C2 the resume line accepts the mismatch at that step only" has '--export=ALL,STAGEC_ALLOW_MISMATCH_STEP=gate-prep --wrap "bash stagec_suite_20261012.sh --resume chain --from gate-prep"' "${T}/c2.log"
check "C2 the prep gate json says false" has '"ok": false' "${D}/output/.gate_SC_prep.json"

echo "== C3 an alpha mismatch: exit 3, after the archive"
build_sandbox; reset_stubs; run "${T}/c0.log" --install
run "${T}/c3.log" --chain -- RUNNER_ALPHA=-0.062802; rc=$?
check "C3 exit 3" test ${rc} -eq 3
check "C3 the resume line for the alpha gate" has "STAGEC_ALLOW_MISMATCH_STEP=gate-logit --wrap \"bash stagec_suite_20261012.sh --resume chain --from gate-logit\"" "${T}/c3.log"
check "C3 both archives ran" test "$(grep -c '^archive	' "${PLOG}")" -eq 2

echo "== C4 the logit fails: the draws still finish, no G4, exit 1, advice --from logit"
build_sandbox; reset_stubs; run "${T}/c0.log" --install
run "${T}/c4.log" --chain -- RUNNER_FAIL=logit; rc=$?
check "C4 exit 1" test ${rc} -eq 1
check "C4 the draws ran beside it and were waited for" has "^draws	" "${PLOG}"
check "C4 no G4, no alpha gate" bash -c "! grep -q '^g4	' '${PLOG}' && [[ ! -e '${D}/output/.gate_SC_logit.json' ]]"
check "C4 the advice: move logit + draws aside, --resume chain --from logit" has "move data/output/logit, data/output/blp/draws and .gate_G4.json aside.*--resume chain --from logit" "${T}/c4.log"

echo "== C5 step 0 fails: nothing produced, no archive"
build_sandbox; reset_stubs; run "${T}/c0.log" --install
run "${T}/c5.log" --chain -- RUNNER_FAIL=precompile; rc=$?
check "C5 exit 1, only the precompile ran" test ${rc} -ne 0 -a "$(tags)" == "precompile "

echo "== C6 --chain --from gate-logit: the alpha gate and the archive only"
build_sandbox; reset_stubs; run "${T}/c0.log" --install; mkparquets 10 10 10 10; mkdraws
mkdir -p "${D}/output/logit"; printf '{"E3_full_dtype": {"param_names": ["alpha", "dummy_D_type"], "theta1": [-0.0628, -5.4]}, "E4_full_dtype": {"param_names": ["alpha", "dummy_D_type"], "theta1": [-0.0567, -5.5]}}\n' > "${D}/output/logit/logit_summary_spec_12.json"
: > "${PLOG}"
run "${T}/c6.log" --chain --from gate-logit; rc=$?
check "C6 exit 0; payloads: precompile g0 archive archive" test ${rc} -eq 0 -a "$(tags)" == "precompile g0 archive archive "

echo "== C7 the scoped override: gate-prep accepted, a later alpha mismatch still STOPS"
build_sandbox; reset_stubs; run "${T}/c0.log" --install
run "${T}/c7.log" --chain -- RUNNER_ROWS="10 11 10 10" RUNNER_ALPHA=-0.062802 STAGEC_ALLOW_MISMATCH_STEP=gate-prep; rc=$?
check "C7 the prep mismatch accepted at that step" has "MISMATCH ACCEPTED AT THIS STEP ONLY (STAGEC_ALLOW_MISMATCH_STEP=gate-prep)" "${T}/c7.log"
check "C7 ... the alpha mismatch still stops (exit 3)" test ${rc} -eq 3
run "${T}/c8.log" --chain -- RUNNER_ROWS="10 11 10 10" STAGEC_ALLOW_MISMATCH=1; rc=$?
checke "C8 STAGEC_ALLOW_MISMATCH=1 on a full chain accepts the prep mismatch and runs to the end" '[[ ${rc} -eq 0 ]] && has "MISMATCH ACCEPTED" "${T}/c8.log" && has "STAGE C CHAIN: PASSED" "${T}/c8.log"'
fi

# ─────────────────────────────────────────────────────────────────────────────────────────
if [[ " ${SECTIONS} " == *" R "* ]]; then
echo "== R --resume"
build_sandbox; reset_stubs; run "${T}/r0.log" --install; mkparquets 10 11 10 10
run "${T}/r1.log" --resume chain
check "R1 resume chain (from prep) refuses a demand_prep on disk" has "data/output already holds results of this chain (resume chain): output/demand_prep" "${T}/r1.log"
run "${T}/r2.log" --resume chain --from gate-prep
check "R2 --from gate-prep needs .gate_G2.json" has ".gate_G2.json is missing" "${T}/r2.log"
okgate G2; reset_stubs
run "${T}/r3.log" --resume chain --from gate-prep -- STAGEC_ALLOW_MISMATCH_STEP=gate-prep; rc=$?
check "R3 exit 0" test ${rc} -eq 0 || tail -20 "${T}/r3.log"
check "R3 T3 with --from gate-prep, T2 afterok it" bash -c "grep -q -- '-J stagec_chain .*--chain --from gate-prep$' '${SBL}' && [[ '$(dep_of stagec_gpu)' == '--dependency=afterok:$(id_of stagec_chain)' ]]"
check "R3 the override travels in the environment of T3 only for that step" has "the gate-prep step accepts the mismatch" "${T}/r3.log"
reset_stubs; run "${T}/r4.log" --resume chain --from logit -- STAGEC_ALLOW_MISMATCH_STEP=gate-prep
check "R4 the step override is refused on another --from" has "applies to --resume chain --from that gate step only" "${T}/r4.log"
reset_stubs; run "${T}/r5.log" --resume chain --from gate-prep -- STAGEC_ALLOW_MISMATCH=1
check "R5 STAGEC_ALLOW_MISMATCH=1 is refused at a resume (it opens every gate)" has "a resume opens ONE" "${T}/r5.log"
okgate SC_prep; okgate G4; reset_stubs
run "${T}/r6.log" --resume chain --from gate-logit
check "R6 --from gate-logit needs the draws" has "the draws are not on disk" "${T}/r6.log"
mkdraws; mkdir -p "${D}/output/logit"; echo '{}' > "${D}/output/logit/logit_summary_spec_12.json"; okgate SC_logit; reset_stubs
run "${T}/r7.log" --resume gpu; rc=$?
check "R7 resume gpu: one T2, no dependency" test ${rc} -eq 0 -a "$(grep -c . "${SBL}")" -eq 1 -a -z "$(dep_of stagec_gpu)"
okgate SC_smoke; reset_stubs
run "${T}/r8.log" --resume next-blp; rc=$?
check "R8 resume next-blp: one job" test ${rc} -eq 0 -a "$(grep -c . "${SBL}")" -eq 1 -a -n "$(id_of stagec_next_blp)"
reset_stubs; run "${T}/r9.log" --resume smoke
check "R9 a retired resume point is refused with the list" has "--resume takes chain \[--from prep|gate-prep|logit|gate-logit\], gpu, next-blp, blp or next-bbl" "${T}/r9.log"
fi

# ─────────────────────────────────────────────────────────────────────────────────────────
if [[ " ${SECTIONS} " == *" S "* ]]; then
echo "== S the smoke verdict (per routine), unchanged"
build_sandbox
mklog () {   # mklog <file> <mapped stage1> <mapped stage2> <mapped stage3> [extra line]
    : > "$1"
    for m in "$2" "$3" "$4"; do
        printf '  [δ WARM-START] Loaded logit_delta_E3_spec_12.bin [binary]
  Obs mapped to draws: %s / 644005 (100.0%%%%)
  [DRY RUN GPU] 10 iters: 1.2s (0.12 s/iter)
' "${m}" >> "$1"
    done
    [[ -n "${5:-}" ]] && echo "$5" >> "$1"; return 0
}
mklog "${T}/s_pass.log" 644005 644005 644005; run "${T}/s1.log" --smoke-verdict 3 "${T}/s_pass.log"; rc=$?
checke "S1 PASS" 'test ${rc} -eq 0 && has "VERDICT PASS" "${T}/s1.log"'
mklog "${T}/s_gap.log" 644005 632735 644005; run "${T}/s2.log" --smoke-verdict 3 "${T}/s_gap.log"; rc=$?
checke "S2 below 100% mapped: FAIL" 'test ${rc} -ne 0 && has "obs mapped to draws below 100%" "${T}/s2.log"'

echo "== S --gpu (T2): build, merged gate, one-process smoke, verdict, hand-off"
gpu_setup () { build_sandbox; reset_stubs; run "${T}/sg0.log" --install; mkparquets 10 10 10 10; mkdraws; okgate SC_logit; : > "${PLOG}"; }
gpu_setup
run "${T}/sg1.log" --gpu; rc=$?
check "SG1 exit 0" test ${rc} -eq 0 || tail -30 "${T}/sg1.log"
check "SG1 payloads: load_julia, sysimage_gpu (missing image), gpu_gate (merged), smoke" test "$(tags)" == "load_julia sysimage_gpu gpu_gate smoke "
check "SG1 the image is built by env_job.sh ENV_STEP=sysimage_gpu" has "^sysimage_gpu	env ENV_STEP=sysimage_gpu bash .*env_job.sh" "${PLOG}"
check "SG1 the merged gate is cl_require_sysimage_gated" has "^gpu_gate	cl_require_sysimage_gated gpu" "${PLOG}"
check "SG1 ONE smoke process: blp_smoke.jl over E1-E4, BLP_ENGINE=ift" has "^smoke	env BLP_ENGINE=ift julia .*blp_smoke.jl --routines 1 2 3 4 --stage sigma,ext1,extended --hpc --R 2000 --seed 42 --dry-run" "${PLOG}"
checke "SG1 four per-routine verdicts and SMOKE: PASS" 'test "$(grep -c "of 3 stages reached the timing block" "${T}/sg1.log")" -eq 4 && has "SMOKE: PASS" "${T}/sg1.log"'
check "SG1 the smoke gate ok" has '"ok": true' "${D}/output/.gate_SC_smoke.json"
for k in 1 2 3 4; do check "SG1 BLP hand-off in the tail: ladder E${k} submitted" test "$(grep -c -- "-J rc_ift_E${k}_" "${SBL}")" -eq 3; done
check "SG1 ... stagec_next_bbl afterok E3/E4, stagec_blp_zip afterany" test -n "$(id_of stagec_next_bbl)" -a -n "$(id_of stagec_blp_zip)"
check "SG1 the ladder marker written" test -f "${D}/output/.stagec_ladders_20261012"
grep -- "-J stagec_next_bbl " "${SBL}" | grep -q -- "--partition=$(REQ REQ_HANDOFF_PART) --time=[0-9:]* --nodes=1 --ntasks=1 --cpus-per-task=$(REQ REQ_HANDOFF_CPUS) --mem=$(REQ REQ_HANDOFF_MEM) " && ok "SG1 stagec_next_bbl on REQ_HANDOFF_* (day 2c/16G)" || bad "SG1 stagec_next_bbl request"
grep -- "-J stagec_blp_zip " "${SBL}" | grep -q -- "--partition=$(REQ REQ_ZIP_PART) --time=$(REQ REQ_ZIP_TIME) --nodes=1 --ntasks=1 --cpus-per-task=$(REQ REQ_ZIP_CPUS) --mem=$(REQ REQ_ZIP_MEM) " && ok "SG1 stagec_blp_zip on REQ_ZIP_* (day 2c/8G/30m)" || bad "SG1 stagec_blp_zip request"
check "SG1 no cpugen pin on the wrapper's own jobs" bash -c "! grep -E -- '-J stagec_' '${SBL}' | grep -q -- 'cpugen\|--constraint'"
for sc in "gap:3" "missing:2" "error:4" "cold:1"; do
    gpu_setup
    run "${T}/sg_${sc%%:*}.log" --gpu -- RUNNER_SMOKE="${sc}"; rc=$?
    check "SG ${sc}: SMOKE: FAIL, exit 1, no ladder, resume gpu printed" bash -c "[[ ${rc} -ne 0 ]] && grep -q 'SMOKE: FAIL' '${T}/sg_${sc%%:*}.log' && ! grep -q -- '-J rc_ift_' '${SBL}' 2>/dev/null && grep -q -- '--resume gpu' '${T}/sg_${sc%%:*}.log'"
done
gpu_setup; run "${T}/sg_build.log" --gpu -- RUNNER_FAIL=sysimage_gpu; rc=$?
checke "SG a failed image build: exit 1, no smoke, resume gpu" 'test ${rc} -ne 0 -a "$(tags)" == "load_julia sysimage_gpu " && has "--resume gpu" "${T}/sg_build.log"'
gpu_setup; rm -f "${D}/output/.gate_SC_logit.json"; run "${T}/sg_nogate.log" --gpu
check "SG the GPU job refuses without a passed alpha gate" has ".gate_SC_logit.json is missing" "${T}/sg_nogate.log"

echo "== S6 --collect-logs (m6)"
mkdir -p "${S}/logs/stagec_smoke_42_strays"; echo e > "${S}/logs/stagec_smoke_E3_42.log"; echo s > "${S}/logs/stagec_smoke_42_strays/x.json"
run "${T}/s6.log" --collect-logs
check "S6 engine logs and strays copied as .out" test -f "${S}/logs/stagec_smoke_E3_42.log.out" -a -f "${S}/logs/stagec_smoke_42_strays__x.json.out"
fi

# ─────────────────────────────────────────────────────────────────────────────────────────
# P — PAYLOAD EQUIVALENCE. The deployed design submitted each payload as its own job: its
# environment = the submitting process's environment (the stub dumps it) + the --export list +
# SLURM_CPUS_PER_TASK of that job. The new design runs them inside T3/T2: the runner dumps the
# environment each payload sees. Both runs use the SAME sandbox path, so values compare as they are.
if [[ " ${SECTIONS} " == *" P "* ]]; then
echo "== P payload equivalence against the deployed design (${DEPLOYED_SCRIPTS})"
GOLD="${SBX}/gold"; NEWD="${SBX}/new"; rm -rf "${GOLD}" "${NEWD}"; mkdir -p "${GOLD}" "${NEWD}"
REL='^(SLEEP_|DEMAND_|SPEC=|R=|SEED=|OPEN_FINANCE|STATE_CENTERING|CONDA_ENV=|PY_MODULE=|JULIA_MODULE=|JULIA_PKG_OFFLINE=|OMP_|MKL_|OPENBLAS_|BLP_|CF_STAGE|CF_EVOL|COST_|PYTHON|SLURM_CPUS_PER_TASK=|CL_DATA_ROOT=|DATA_ROOT=|CL_ROOT=)'
# effective env of a deployed job: dump + --export assignments + its cpus
gold_env () {   # gold_env <job name> <cpus default>
    local line exp cpus
    line="$(grep -- "-J $1 " "${GOLD}/sbatch.log" | head -1)"; [[ "${line}" == *.sh ]] && line="${line% *}"   # the job script path
    exp="$(grep -o -- '--export=ALL[^ ]*\( [^-][^ ]*\)*' <<< "${line}" | head -1 | sed 's/^--export=ALL,\{0,1\}//')"
    cpus="$(grep -o -- '--cpus-per-task=[0-9]*' <<< "${line}" | head -1 | cut -d= -f2)"
    { grep -v '^SLURM_CPUS_PER_TASK=' "${GOLD}/sbatch_env_$1.txt"
      "${PYW}" -c 'import sys
s = sys.argv[1]
out, cur = [], ""
for part in s.split(","):
    if "=" in part.split(" ")[0] and cur: out.append(cur); cur = part
    elif "=" in part and not cur: cur = part
    else: cur = cur + "," + part if cur else part
if cur: out.append(cur)
print("\n".join(o.replace("\"", "") for o in out))' "${exp}"
      echo "SLURM_CPUS_PER_TASK=${cpus:-$2}"; } | awk -F= '!seen[$1]++ || 1' | tac | awk -F= '!seen[$1]++' | grep -E "${REL}" | sort
}
new_env () {   # new_env <payload tag> <cpus default>
    { cat "${NEWD}/payload_env_$1.txt"; grep -q '^SLURM_CPUS_PER_TASK=' "${NEWD}/payload_env_$1.txt" || echo "SLURM_CPUS_PER_TASK=$2"; } | grep -E "${REL}" | sort
}
# the deployed run: install, then the prep hand-off (it submits every payload of the chain)
build_sandbox "${DEPLOYED_SCRIPTS}" "${DEPLOYED_SCRIPTS}/stagec_suite_20261012.sh"; reset_stubs
( cd "${S}" && env PATH="${STUBS}:${PATH}" STUB_DIR="${STATE}" CL_ROOT="${S}" DATA_ROOT="${D}" PY_MODULE="" CONDA_ENV="" CF_PYTHON="${PYW}" \
    STAGEC_SLEEP_EXTRA="--skip-preflight" STAGEC_BLP_EXTRA="--skip-preflight" PY_PREFLIGHT=0 STAGEC_LOCAL_TEST=1 bash stagec_suite_20261012.sh --install ) > "${T}/p_gold_install.log" 2>&1
( cd "${S}" && env PATH="${STUBS}:${PATH}" STUB_DIR="${STATE}" CL_ROOT="${S}" DATA_ROOT="${D}" PY_MODULE="" CONDA_ENV="" CF_PYTHON="${PYW}" \
    STAGEC_SLEEP_EXTRA="--skip-preflight" STAGEC_BLP_EXTRA="--skip-preflight" PY_PREFLIGHT=0 STAGEC_LOCAL_TEST=1 bash stagec_suite_20261012.sh --next-prep ) > "${T}/p_gold_prep.log" 2>&1
cp "${STATE}"/sbatch_env_*.txt "${STATE}/sbatch.log" "${GOLD}/" 2>/dev/null
# the deployed ladders: --next-blp after a passed smoke
mkdraws; ( cd "${S}" && env PATH="${STUBS}:${PATH}" STUB_DIR="${STATE}" CL_ROOT="${S}" DATA_ROOT="${D}" PY_MODULE="" CONDA_ENV="" CF_PYTHON="${PYW}" \
    STAGEC_BLP_EXTRA="--skip-preflight" PY_PREFLIGHT=0 STAGEC_LOCAL_TEST=1 bash stagec_suite_20261012.sh --next-blp ) > "${T}/p_gold_blp.log" 2>&1
grep -- '-J rc_ift_' "${STATE}/sbatch.log" | sed 's/--dependency=[a-z]*:[0-9:]*//' > "${GOLD}/ladders.txt"
for n in rc_ift_E3_head rc_ift_E1_head; do cp "${STATE}/sbatch_env_${n}.txt" "${GOLD}/ladder_env_${n}.txt" 2>/dev/null; done
# the new run, same sandbox path: install, the chain (payloads recorded), then the GPU job (smoke + ladders)
build_sandbox; reset_stubs; run "${T}/p_new_install.log" --install
run "${T}/p_new_chain.log" --chain
cp "${STATE}"/payload_env_*.txt "${STATE}/payload.log" "${NEWD}/" 2>/dev/null
mkdraws; okgate SC_logit; reset_stubs; run "${T}/p_new_gpu.log" --gpu
cp "${STATE}"/payload_env_smoke.txt "${NEWD}/" 2>/dev/null; cp "${STATE}/payload.log" "${NEWD}/payload_gpu.log"
grep -- '-J rc_ift_' "${STATE}/sbatch.log" | sed 's/--dependency=[a-z]*:[0-9:]*//' > "${NEWD}/ladders.txt"
for n in rc_ift_E3_head rc_ift_E1_head; do cp "${STATE}/sbatch_env_${n}.txt" "${NEWD}/ladder_env_${n}.txt" 2>/dev/null; done
check "P0 the deployed run submitted its chain (golden captured)" test -s "${GOLD}/sbatch_env_sleep_prep.txt" -a -s "${GOLD}/sbatch_env_blp_logit.txt" -a -s "${GOLD}/sbatch_env_blp_draws.txt"
# payload -> (deployed job name, deployed cpus default, new tag); prep/gates: their cpus never reach the payload's
# arithmetic (the prep's parallelism is DEMAND_PREP_JOBS; sleep_job.sh prints the count only), so they are
# compared WITHOUT SLURM_CPUS_PER_TASK; logit and draws read it for --threads, so it is compared.
for spec in "sleep_prep:16:prep:nocpu" "sleep_gate_G2:2:g2:nocpu" "blp_logit:8:logit:cpu" "sleep_gate_G4:2:g4:nocpu" "blp_draws:8:draws:cpu"; do
    IFS=: read -r job cpu tag mode <<< "${spec}"
    gold_env "${job}" "${cpu}" > "${T}/p_gold_${tag}.env"; new_env "${tag}" "$(REQ REQ_CHAIN_CPUS)" > "${T}/p_new_${tag}.env"
    if [[ "${mode}" == "nocpu" ]]; then sed -i '/^SLURM_CPUS_PER_TASK=/d' "${T}/p_gold_${tag}.env" "${T}/p_new_${tag}.env"; fi
    if diff "${T}/p_gold_${tag}.env" "${T}/p_new_${tag}.env" > "${T}/p_diff_${tag}.txt"; then ok "P ${tag}: the payload's relevant environment equals the deployed ${job} job's ($(grep -c . "${T}/p_new_${tag}.env") vars)"
    else bad "P ${tag}: environment differs from ${job} (${T}/p_diff_${tag}.txt)"; cat "${T}/p_diff_${tag}.txt" | head -10; fi
done
check "P cmd: prep = sleep_job.sh, SLEEP_STEP=prep SPEC=12 DEMAND_PREP_JOBS=4 SLEEP_ACTIVE_ESTS=1 2 3 4 (sleep_run.sh's --export)" grep -qx "prep	env SLEEP_STEP=prep SPEC=12 DEMAND_PREP_JOBS=4 SLEEP_ACTIVE_ESTS=1 2 3 4 bash ${S}/sleep_job.sh" "${NEWD}/payload.log"
check "P cmd: G2/G4 = sleep_job.sh SLEEP_STEP=gate (the gates sleep_run.sh / the 10-13 graph submitted)" bash -c "grep -qx 'g2	env SLEEP_STEP=gate SLEEP_GATE=G2 SLEEP_GATE_ROUTINES=1 2 3 4 SPEC=12 bash ${S}/sleep_job.sh' '${NEWD}/payload.log' && grep -qx 'g4	env SLEEP_STEP=gate SLEEP_GATE=G4 SLEEP_GATE_ROUTINES=1 2 3 4 SPEC=12 bash ${S}/sleep_job.sh' '${NEWD}/payload.log'"
check "P cmd: logit = logit_job.sh, draws = blp_draws_job.sh" bash -c "grep -q '^logit	.* bash .*/logit_job.sh$' '${NEWD}/payload.log' && grep -q '^draws	.* bash .*/blp_draws_job.sh$' '${NEWD}/payload.log'"
# the smoke: the deployed per-routine call vs blp_smoke.jl's plan
dep_smoke="$(grep -o '"${HERE}/blp_rc.jl" --estim "${k}" --stage "${SMOKE_STAGES}" --hpc --R "${R}" --seed "${SEED}" --dry-run' "${DEPLOYED_SCRIPTS}/stagec_suite_20261012.sh" | head -1)"
check "P smoke: the deployed call is blp_rc.jl --estim k --stage S --hpc --R R --seed SEED --dry-run" test -n "${dep_smoke}"
plan="$(cd /c/sc_tests && timeout 600 julia --startup-file=no "${REPO}/blp_smoke.jl" --routines "1 2 3 4" --print-plan --stage sigma,ext1,extended --hpc --R 2000 --seed 42 --dry-run 2>&1)"
want=""; for k in 1 2 3 4; do want="${want}E${k}: run_routine(${k}; passthrough=[\"--stage\", \"sigma,ext1,extended\", \"--hpc\", \"--R\", \"2000\", \"--seed\", \"42\", \"--dry-run\"])"$'\n'; done
check "P smoke: blp_smoke.jl passes each routine exactly what blp_rc.jl's CLI passed (run_routine(k; passthrough) after stripping --estim k)" test "${plan}"$'\n' == "${want}"
check "P smoke: the GPU job's smoke command carries the same flags and BLP_ENGINE=ift" grep -q '^smoke	env BLP_ENGINE=ift julia .*blp_smoke.jl --routines 1 2 3 4 --stage sigma,ext1,extended --hpc --R 2000 --seed 42 --dry-run$' "${NEWD}/payload_gpu.log"
# the smoke's environment: the deployed stagec_smoke job's environment + what its smoke() set before julia
# (cl_export_step_dirs, cl_load_julia, BLP_ENGINE=ift) vs what blp_smoke.jl sees in T2. The step dirs come
# from the same cl_export_step_dirs call in both designs and are compared as they are (same sandbox path);
# SLURM_CPUS_PER_TASK is not (the threads are explicit: --threads 6 in both).
{ grep -v '^SLURM_CPUS_PER_TASK=' "${GOLD}/sbatch_env_stagec_smoke.txt"; echo "BLP_ENGINE=ift"
  ( cd "${S}" && env -i PATH="${PATH}" CL_ROOT="${S}" DATA_ROOT="${D}" bash -c '. ./cluster_lib.sh >/dev/null 2>&1; cl_export_step_dirs; env' ) | grep -E '^(OPEN_FINANCE_ROOT|SLEEP_OUT_ROOT|DEMAND_PREP_DIR|COST_POLFUNC_DIR|STATE_CENTERING_JSON)='
} | tac | awk -F= '!seen[$1]++' | grep -E "${REL}" | grep -v '^SLURM_CPUS_PER_TASK=' | sort > "${T}/p_gold_smoke.env"
grep -E "${REL}" "${NEWD}/payload_env_smoke.txt" | grep -v '^SLURM_CPUS_PER_TASK=' | sort > "${T}/p_new_smoke.env"
if diff "${T}/p_gold_smoke.env" "${T}/p_new_smoke.env" > "${T}/p_diff_smoke.txt"; then ok "P smoke: blp_smoke.jl's relevant environment equals the deployed smoke's ($(grep -c . "${T}/p_new_smoke.env") vars)"
else bad "P smoke: environment differs (${T}/p_diff_smoke.txt)"; head -8 "${T}/p_diff_smoke.txt"; fi
# the ladders: the same submissions (job ids and dependencies removed)
if diff "${GOLD}/ladders.txt" "${NEWD}/ladders.txt" > "${T}/p_diff_ladders.txt"; then ok "P ladders: the 12 rc_ift submissions are identical ($(grep -c . "${NEWD}/ladders.txt"))"
else bad "P ladders differ (${T}/p_diff_ladders.txt)"; head -6 "${T}/p_diff_ladders.txt"; fi
for n in rc_ift_E3_head rc_ift_E1_head; do
    grep -E "${REL}" "${GOLD}/ladder_env_${n}.txt" | sort > "${T}/p_gold_${n}.env"; grep -E "${REL}" "${NEWD}/ladder_env_${n}.txt" | sort > "${T}/p_new_${n}.env"
    if diff "${T}/p_gold_${n}.env" "${T}/p_new_${n}.env" > "${T}/p_diff_${n}.txt"; then ok "P ${n}: the ladder job's relevant environment is identical"
    else bad "P ${n}: environment differs (${T}/p_diff_${n}.txt)"; head -6 "${T}/p_diff_${n}.txt"; fi
done
# the BBL flags: the deployed literal DESIGN array vs the new one expanded from stagec_requests.env
dep_design="$(sed -n '/^DESIGN=(/,/)$/p' "${DEPLOYED_SCRIPTS}/stagec_suite_20261012.sh" | tr -s ' \n' ' ')"
new_design="$( . "${REPO}/stagec_requests.env"; eval "$(sed -n '/^DESIGN=(/,/)$/p' "${REPO}/stagec_suite_20261012.sh")"; printf 'DESIGN=(%s)' "${DESIGN[*]}" | tr -s ' ' ' ')"
check "P BBL: the flag set (DESIGN) is identical after expanding REQ_*" test "$(echo "${dep_design}" | tr -d '()' | xargs)" == "$(echo "${new_design}" | tr -d '()' | xargs)"
fi

# ─────────────────────────────────────────────────────────────────────────────────────────
# D — unchanged hand-off code (from the 10-13/14 suite)
mkfits () {   # mkfits <se_status for E3 ext1>
    mkdir -p "${D}/output/blp/draws"; echo h > "${D}/output/blp/draws/halton_nu_R2000_seed42.jls"
    local k st
    for k in 1 2 3 4; do for st in ext1 extended; do
        echo "fit E${k} ${st}" > "${D}/output/blp/blp_results_E${k}_spec_12_${st}.jls"
        # the two-start record (ext1_start / extended_start) FIRST, with nested "status" keys: the
        # se_status read must take the top-level key only
        printf '{"%s_start": {"cold": {"status": "ran", "Q_opt": 0.15, "Q_reeval": 0.15, "Q": 0.15, "reeval_converged": true, "converged": true}, "warm_rc4": {"status": "ran", "Q_opt": 0.2, "Q_reeval": null, "reeval_converged": false}, "choice_basis": "converged re-evaluations", "chosen": "cold", "n_starts": 2}, "theta1": [-0.06], "Q_value": 0.15, "se_status": "%s"}\n' "${st}" "$([[ ${k} == 3 && ${st} == ext1 ]] && echo "$1" || echo ok)" > "${D}/output/blp/blp_results_E${k}_spec_12_${st}.json"
    done; done
}
if [[ " ${SECTIONS} " == *" D "* ]]; then
# T2_D_ONLY=456 runs D4-D6 alone (each builds its own sandbox); D1-D3 are skipped.
if [[ -z "${T2_D_ONLY:-}" ]]; then
echo "== D1 --next-blp"
build_sandbox; reset_stubs; run "${T}/d0.log" --install
mkdir -p "${D}/output/blp"; echo x > "${D}/output/blp/blp_checkpoint_E1_spec_12_sigma.jls"
run "${T}/d1a.log" --next-blp
check "D1a M3: a checkpoint in data/output/blp refuses the ladders" has "RC result/checkpoint file(s) (blp_checkpoint_E1_spec_12_sigma.jls" "${T}/d1a.log"
check "D1a nothing submitted" test ! -s "${SBL}"
rm -f "${D}/output/blp/blp_checkpoint_E1_spec_12_sigma.jls"; echo x > "${D}/output/blp/blp_results_E3_spec_12_extended_dx.jls"
run "${T}/d1c.log" --next-blp
check "D1c M3: a result of ANOTHER tag (_dx) refuses too" has "RC result/checkpoint file(s) (blp_results_E3_spec_12_extended_dx.jls" "${T}/d1c.log"
rm -f "${D}/output/blp/blp_results_E3_spec_12_extended_dx.jls"; mkdir -p "${D}/output/blp/draws"; echo h > "${D}/output/blp/draws/halton_nu_R2000_seed42.jls"
run "${T}/d1b.log" --next-blp; rc=$?
check "D1b exit 0" test ${rc} -eq 0 || tail -20 "${T}/d1b.log"
for k in 1 2 3 4; do check "D1b ladder E${k} (head, ext2, ext)" test "$(grep -c -- "-J rc_ift_E${k}_" "${SBL}")" -eq 3; done
check "D1b no draws or sysimage build" bash -c "! grep -Eq -- '-J (blp_draws|blp_build_sysimg)' '${SBL}'"
e3="$(id_of rc_ift_E3_ext)"; e4="$(id_of rc_ift_E4_ext)"; e1="$(id_of rc_ift_E1_ext)"; e2="$(id_of rc_ift_E2_ext)"
check "D1b next_bbl afterok the E3/E4 terminals only" test "$(dep_of stagec_next_bbl)" == "--dependency=afterok:${e3}:${e4}"
check "D1b order: E3/E4 ladders, the hand-off, then E1/E2 (m3)" test "$(ln_of '-J rc_ift_E4_ext ')" -lt "$(ln_of '-J stagec_next_bbl ')" -a "$(ln_of '-J stagec_next_bbl ')" -lt "$(ln_of '-J rc_ift_E1_head ')"
check "D1b blp_zip afterany every terminal and the hand-off" test "$(dep_of stagec_blp_zip)" == "--dependency=afterany:${e3}:${e4}:${e1}:${e2}:$(id_of stagec_next_bbl)"
grep -- "-J stagec_blp_zip " "${SBL}" | grep -q -- "if bash stagec_suite_20261012.sh --check-blp; then bash cluster_archive.sh --set blp --copy" && ok "D1b the blp set only after --check-blp" || bad "D1b blp zip wrap"
grep -- "-J stagec_blp_zip " "${SBL}" | grep -q -- "elif bash stagec_suite_20261012.sh --check-blp '3 4'; then bash cluster_archive.sh --set blp --copy --tag \"\${SLURM_JOB_ID}_E34\"" && ok "D1b else the separate E3/E4 archive _E34 (D8)" || bad "D1b E34 fallback"
grep -- "-J stagec_blp_zip " "${SBL}" | grep -q -- "--collect-logs; bash cluster_archive.sh --set logs" && ok "D1b logs collected before the logs archive (m6)" || bad "D1b collect logs"
check "D1b the ladder marker is written" test -f "${D}/output/.stagec_ladders_20261012"
check "D1b the failed-ladder line is printed" has "IF A LADDER FAILS" "${T}/d1b.log"
echo "== D1d an E1/E2 refusal does not strand the BBL (stub blp_run.sh)"
reset_stubs; cp "${S}/blp_run.sh" "${SB}/blp_run.sh.real"
printf '#!/bin/bash\ncase "$*" in *"--routines 1 2"*) echo "stub: E1/E2 refused"; exit 2 ;; esac\necho "BLP_TERM_JOBIDS=8000003:8000004"\n' > "${S}/blp_run.sh"
run "${T}/d1d.log" --next-blp; rc=$?
check "D1d exit nonzero (visible) ..." test ${rc} -ne 0
check "D1d ... the hand-off is submitted afterok E3/E4" test "$(dep_of stagec_next_bbl)" == "--dependency=afterok:8000003:8000004"
check "D1d ... the archive afterany E3/E4 + hand-off" test "$(dep_of stagec_blp_zip)" == "--dependency=afterany:8000003:8000004:$(id_of stagec_next_bbl)"
check "D1d ... and the resume line of the unfinished ladders is printed" has "--wrap \"bash stagec_suite_20261012.sh --resume blp\"" "${T}/d1d.log"
cp "${SB}/blp_run.sh.real" "${S}/blp_run.sh"

echo "== D2 --check-blp"
run "${T}/d2a.log" --check-blp; rc=$?
checke "D2a incomplete results refused (no partial blp archive)" 'test ${rc} -ne 0 && has "the RC results of E1, E2, E3, E4 are incomplete" "${T}/d2a.log"'
check "D2a ... and the --resume blp line is printed (a failed head job)" has "--wrap \"bash stagec_suite_20261012.sh --resume blp\"" "${T}/d2a.log"
mkfits ok; run "${T}/d2b.log" --check-blp
check "D2b complete results pass" has "STAGE C CHECK-BLP OK (E1, E2, E3, E4)" "${T}/d2b.log"
rm -f "${D}/output/blp/blp_results_E1_spec_12_extended.jls"
run "${T}/d2c.log" --check-blp; rc1=$?; run "${T}/d2d.log" --check-blp "3 4"; rc2=$?
check "D2c E1 missing: all four fails, E3/E4 passes (the E34 archive)" test ${rc1} -ne 0 -a ${rc2} -eq 0
check "D2c ... with its own gate" has '"ok": true' "${D}/output/.gate_SC_blp_E34.json"
run "${T}/d2e.log" --check-blp "3 x"
check "D2e bad routines refused" has "takes routines like '3 4'" "${T}/d2e.log"

echo "== D3 --next-bbl"
reset_stubs; mkfits "refused: K > L (17 > 16)"
run "${T}/d3a.log" --next-bbl; rc=$?
checke "D3a E3 ext1 without joint SEs: the BBL is refused" 'test ${rc} -ne 0 && has "E3 ext1 has no joint standard errors: se_status .refused: K > L (17 > 16)." "${T}/d3a.log"'
check "D3a ... with the resume line" has "bash stagec_suite_20261012.sh --resume next-bbl" "${T}/d3a.log"
check "D3a nothing submitted, no fit record" test ! -s "${SBL}" -a ! -e "${D}/output/blp/stagec_bblfit_E3_spec_12_extended_ms983.json"
mkfits ok; sed -i 's/, "se_status": "ok"//' "${D}/output/blp/blp_results_E4_spec_12_ext1.json"
run "${T}/d3b.log" --next-bbl
check "D3b an absent se_status is refused too" has "E4 ext1 has no joint standard errors: se_status '<absent>'" "${T}/d3b.log"
mkfits ok; reset_stubs
run "${T}/d3c.log" --next-bbl -- STUB_SCONTROL_FAIL=1; rc=$?
check "D3c exit 0" test ${rc} -eq 0 || tail -30 "${T}/d3c.log"
check "D3c four fit records" test "$(ls "${D}"/output/blp/stagec_bblfit_E*_spec_12_extended_*983.json | wc -l)" -eq 4
check "D3c fit record = sha256 of the extended fit" test "$(grep -o '"fit_sha256": "[0-9a-f]*"' "${D}/output/blp/stagec_bblfit_E3_spec_12_extended_ms983.json" | cut -d'"' -f4)" == "$(sha256sum "${D}/output/blp/blp_results_E3_spec_12_extended.jls" | cut -d' ' -f1)"
grep -- "-J bbl_solve_E3_ms983 " "${SBL}" | grep -q -- "--promote" && ok "D3c _ms983 solve carries --promote" || bad "D3c promote on _ms983"
grep -- "-J bbl_solve_E3_sc983 " "${SBL}" | grep -q -- "--promote" && bad "D3c _sc983 must not promote" || ok "D3c _sc983 not promoted"
check "D3c no tables job for either tag (D3: --no-tables on both)" test "$(grep -c -- '-J bbl_tables_' "${SBL}")" -eq 0
check "D3c no bbl_run.sh archive (the suite archives)" bash -c "! grep -q -- '-J bbl_zip' '${SBL}'"
grep -- "-J bbl_fwd_E3_ms983" "${SBL}" | head -1 | grep -q -- "--partition=gpu_h100" && ok "D3c fwd on gpu_h100" || bad "D3c fwd partition"
check "D3c no polfunc job (never refit)" bash -c "! grep -q -- '-J bbl_polfunc' '${SBL}'"
ms="$(id_of bbl_solve_E3_ms983):$(id_of bbl_solve_E4_ms983)"; sc="$(id_of bbl_solve_E3_sc983):$(id_of bbl_solve_E4_sc983)"
check "D3c _ms983 archive afterANY its solves" test "$(dep_of stagec_bbl_zip_ms983)" == "--dependency=afterany:${ms}"
check "D3c _sc983 archive afterANY its solves" test "$(dep_of stagec_bbl_zip_sc983)" == "--dependency=afterany:${sc}"
check "D3c the _ms983 archive is submitted before any _sc983 job (m2/D10)" test "$(ln_of '-J stagec_bbl_zip_ms983 ')" -lt "$(ln_of '_sc983')"
grep -- "-J stagec_bbl_zip_ms983 " "${SBL}" | grep -q -- "--check-bbl _ms983 && bash cluster_archive.sh --set bbl --copy --slim --psi-tag '_ms983'" && ok "D3c archive checks the binding first" || bad "D3c archive wrap"
check "D3c final afterany both archives" test "$(dep_of stagec_final)" == "--dependency=afterany:$(id_of stagec_bbl_zip_ms983):$(id_of stagec_bbl_zip_sc983)"
grep -- "-J stagec_final " "${SBL}" | grep -q -- "--collect-logs; bash cluster_archive.sh --set logs" && ok "D3c final collects the logs first (m6)" || bad "D3c final collect"
check "D3c archive job id recorded in the fit record" has "\"bbl_archive_jobs\": \"$(id_of stagec_bbl_zip_ms983)\"" "${D}/output/blp/stagec_bblfit_E3_spec_12_extended_ms983.json"
check "D3c a sweep that stays HELD is surfaced with its release line (m8)" has "stay HELD (bbl_run.sh could not release them).*scontrol release" "${T}/d3c.log"
check "D3c ... and in the summary" has "HELD SWEEPS (release them" "${T}/d3c.log"
reset_stubs; run "${T}/d3d.log" --next-bbl
check "D3d a second hand-off refuses (files of the run on disk)" has "are already on disk" "${T}/d3d.log"
check "D3d nothing submitted" test ! -s "${SBL}"
fi
if [[ -z "${T2_D_ONLY:-}" || "${T2_D_ONLY}" == "3ef" ]]; then
echo "== D3e an _sc983 launch failure: _ms983 keeps its archive and gets the final job"
build_sandbox; reset_stubs; run "${T}/d0.log" --install; mkfits ok
run "${T}/d3e.log" --next-bbl -- STUB_FAIL_RE=sc983; rc=$?
check "D3e exit nonzero" test ${rc} -ne 0
check "D3e _ms983 submitted with its archive" test -n "$(id_of stagec_bbl_zip_ms983)" -a -n "$(id_of bbl_solve_E3_ms983)"
check "D3e stagec_final afterany the _ms983 archive" test "$(dep_of stagec_final)" == "--dependency=afterany:$(id_of stagec_bbl_zip_ms983)"
check "D3e no _sc983 job left" test "$(grep -c -- '_sc983' "${SBL}")" -eq 0
check "D3e the one line that launches _sc983 alone" has "--wrap \"bash stagec_suite_20261012.sh --next-bbl _sc983\"" "${T}/d3e.log"
check "D3e the bbl gate says false" has '"ok": false' "${D}/output/.gate_SC_bbl.json"
check "D3e the refused launch's dispatch folders are moved aside (D3f fix)" has "moved aside to bbl/_stagec_failed_launch_sc983_" "${T}/d3e.log"
check "D3e ... none is left under .dispatch" test ! -d "${D}/output/bbl/.dispatch/E3_spec_12_extended_sc983" -a ! -d "${D}/output/bbl/.dispatch/E4_spec_12_extended_sc983"
reset_stubs; run "${T}/d3f.log" --next-bbl _sc983; rc=$?
check "D3f the recovery launches _sc983 alone" test ${rc} -eq 0 -a -n "$(id_of bbl_solve_E3_sc983)" -a "$(grep -c -- '_ms983' "${SBL}")" -eq 0 || tail -25 "${T}/d3f.log"
check "D3f ... with its archive and a final job" test "$(dep_of stagec_final)" == "--dependency=afterany:$(id_of stagec_bbl_zip_sc983)"

fi
if [[ -z "${T2_D_ONLY:-}" || "${T2_D_ONLY}" == "456" ]]; then
echo "== D4 --check-bbl"
build_sandbox; reset_stubs; run "${T}/d0.log" --install; mkfits ok; run "${T}/d4z.log" --next-bbl
for k in 3 4; do printf '{"run": {"n_shards_expected": 300, "n_shards_found": 300}, "identification": {"promoted": false}}\n' > "${D}/output/bbl/cost_params_E${k}_spec_12_extended_ms983.json"; done
run "${T}/d4a.log" --check-bbl _ms983
check "D4a bound fit + whole cost params: OK" has "STAGE C CHECK-BBL _ms983 OK" "${T}/d4a.log"
sed -i 's/"n_shards_found": 300/"n_shards_found": 299/' "${D}/output/bbl/cost_params_E4_spec_12_extended_ms983.json"
run "${T}/d4b.log" --check-bbl _ms983
check "D4b a missing shard refuses the archive" has "n_shards_found 299 != n_shards_expected 300" "${T}/d4b.log"
echo replaced > "${D}/output/blp/blp_results_E3_spec_12_extended.jls"
run "${T}/d4c.log" --check-bbl _ms983
check "D4c a replaced fit refuses the archive" has "is not the fit stagec_bblfit_E3_spec_12_extended_ms983.json records" "${T}/d4c.log"
run "${T}/d4d.log" --check-bbl _ms982
check "D4d another tag refused" has "takes _ms983 or _sc983" "${T}/d4d.log"

echo "== D5 --repair-bbl (the bound repair)"
build_sandbox; reset_stubs; run "${T}/d0.log" --install; mkfits ok
run "${T}/d5a.log" --next-bbl; check "D5a launch for the repair test" has "STAGE C BBL SUBMITTED" "${T}/d5a.log"
z0="$(id_of stagec_bbl_zip_ms983)"
printf '{"run": {"n_shards_expected": 300, "n_shards_found": 300}}\n' > "${D}/output/bbl/cost_params_E3_spec_12_extended_ms983.json"
reset_stubs
run "${T}/d5b.log" --repair-bbl _ms983; rc=$?
check "D5b exit 0" test ${rc} -eq 0 || tail -25 "${T}/d5b.log"
check "D5b E3 (cost params on disk) left alone" has "E3: cost params on disk; nothing to repair" "${T}/d5b.log"
check "D5b only E4 repaired: a sweep and a solve, no fwd launch" test "$(grep -c -- '-J bbl_sweep_E4_ms983 ' "${SBL}")" -eq 1 -a "$(grep -c -- '-J bbl_solve_E4_ms983 ' "${SBL}")" -eq 1 -a "$(grep -c -- '-J bbl_fwd_' "${SBL}")" -eq 0 -a "$(grep -c -- '_E3_ms983 ' "${SBL}")" -eq 0
grep -- "-J bbl_solve_E4_ms983 " "${SBL}" | grep -q -- "--promote" && ok "D5b the repaired _ms983 solve still promotes" || bad "D5b promote lost"
check "D5b no tables, no bare archive" bash -c "! grep -Eq -- '-J (bbl_tables|bbl_zip)' '${SBL}'"
check "D5b suite archive afterany the repair solve" test "$(dep_of stagec_bbl_zip_ms983)" == "--dependency=afterany:$(id_of bbl_solve_E4_ms983)"
check "D5b both archive ids in the record" has "\"bbl_archive_jobs\": \"${z0} $(id_of stagec_bbl_zip_ms983)\"" "${D}/output/blp/stagec_bblfit_E4_spec_12_extended_ms983.json"
reset_stubs; echo replaced > "${D}/output/blp/blp_results_E4_spec_12_extended.jls"
run "${T}/d5c.log" --repair-bbl _ms983
check "D5c a replaced fit is never repaired onto" has "were simulated on another fit" "${T}/d5c.log"
check "D5c nothing submitted" test ! -s "${SBL}"

echo "== D6 --resume blp (a failed head job: the unfinished ladders only)"
build_sandbox; reset_stubs; run "${T}/d0.log" --install; okgate SC_smoke
run "${T}/d6a.log" --resume blp
check "D6a no ladder marker: refused (the ladders never started)" has "no ladder marker" "${T}/d6a.log"
mkdir -p "${D}/output/blp"; echo old > "${D}/output/blp/blp_results_E1_spec_12_sigma.jls"; touch -d '2 hours ago' "${D}/output/blp/blp_results_E1_spec_12_sigma.jls"
echo m > "${D}/output/.stagec_ladders_20261012"; touch -d '1 hour ago' "${D}/output/.stagec_ladders_20261012"
run "${T}/d6b.log" --resume blp
check "D6b an RC file older than the ladder marker refuses (not this chain's)" has "older than this chain's ladders (blp_results_E1_spec_12_sigma.jls" "${T}/d6b.log"
rm -f "${D}/output/blp/blp_results_E1_spec_12_sigma.jls"; mkfits ok; rm -f "${D}/output/blp/blp_results_E3_spec_12_ext1.jls" "${D}/output/blp/blp_results_E3_spec_12_extended.jls"
reset_stubs; run "${T}/d6c.log" --resume blp; rc=$?
check "D6c exit 0" test ${rc} -eq 0 || tail -25 "${T}/d6c.log"
check "D6c only E3's ladder is resubmitted (head, ext2, ext)" test "$(grep -c -- '-J rc_ift_E3_' "${SBL}")" -eq 3 -a "$(grep -Ec -- '-J rc_ift_E[124]_' "${SBL}")" -eq 0
check "D6c the BBL hand-off afterok E3's new terminal" test "$(dep_of stagec_next_bbl)" == "--dependency=afterok:$(id_of rc_ift_E3_ext)"
check "D6c the archive afterany the new terminal + the hand-off" test "$(dep_of stagec_blp_zip)" == "--dependency=afterany:$(id_of rc_ift_E3_ext):$(id_of stagec_next_bbl)"
check "D6c the ladder marker is not rewritten" test "$(cat "${D}/output/.stagec_ladders_20261012")" == "m"
mkfits ok; reset_stubs; run "${T}/d6d.log" --resume blp
checke "D6d every ladder whole: nothing to resume, nothing submitted" 'test ! -s "${SBL}" && has "nothing to resume" "${T}/d6d.log"'
fi
fi


echo; echo "test_stagec: ${PASS} ok, ${FAIL} FAIL"
[[ ${FAIL} -eq 0 ]]
