#!/bin/bash
# test_stagec.sh — the stage C wrapper (stagec_suite_20261012.sh) against stub SLURM, in a sandbox
# OUTSIDE OneDrive (default C:/sc_tests; CLUSTER_TEST_ROOT overrides). Real cluster scripts copied
# from the repo (CR stripped), fake inputs, the real policy file; every payload (sleep_job.sh,
# logit_job.sh, blp_draws_job.sh, env_job.sh, cluster_archive.sh, the smoke's Julia) goes through
# stubs/payload_runner.sh, which records it and fakes its outputs.
#   A  the launch (T1): the cluster's starting shape (empty data/ + scripts/, the documented unzip); dry run; live: install + Julia env + T4 (CPU image, turin) + T3 (chain) + T2
#      (GPU) afterok T3, the REQ_* requests, the pin on T4 only, the prep exports; refusals
#   B  --install, --final
#   C  --chain (T3): every step, the archive always, exit codes, step restarts (--from), the scoped
#      mismatch override
#   R  --resume chain [--from] | gpu | next-blp
#   S  the smoke verdict; --gpu (T2): build, merged gate, one-process smoke, split verdict, the BLP
#      hand-off in its tail; --collect-logs
#   P  PAYLOAD EQUIVALENCE: every payload command and its relevant environment against the
#      DEPLOYED design (its scripts: DEPLOYED_SCRIPTS, the bundle that ran on 10-13/14)
#   D  --next-blp, --check-blp, --next-bbl, --check-bbl, --repair-bbl, --resume blp (slow: bbl_run.sh)
#   I  --cpu-image (T4): build, load proof, gate; the BBL hand-off and the repair refuse without it;
#      --resume cpu-image; a live T4 does not block the chain's resumes
#   L  the sleepiness stage on the cluster (Pedro 10-15): the E1-E4 estimates in the chain, the
#      automatic gate, the summary block, the bootstraps off the critical path
#   J  B1 (10-15 review): the CPU-image verdict with NO julia on PATH, on every login-shell path
#      (--next-bbl [tag], --repair-bbl, --resume next-bbl | blp | next-blp), the verdict un-stubbed
# Usage: bash tests/cluster/test_stagec.sh [section letters, default "A B C L R S P D I J"]
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
SECTIONS="${1:-A B C L R S P D I J}"
PASS=0; FAIL=0
ok ()  { echo "  ok   $*"; PASS=$((PASS+1)); }
bad () { echo "  FAIL $*"; FAIL=$((FAIL+1)); return 1; }
check () { local what="$1"; shift; if "$@"; then ok "${what}"; else bad "${what}"; fi; }
checke () { local what="$1"; shift; if eval "$1"; then ok "${what}"; else bad "${what}"; fi; }   # one condition string (&&, ||)
has () { grep -q -- "$1" "$2"; }
hasn () { ! grep -q -- "$1" "$2"; }

# the policy: the 10-16 refit under its upload name (the wrapper pins its sha256); the deployed
# 10-13 wrapper of section P pins the 09-30 file, taken from the 10-13 data zip (old_policy)
POLICY_SRC="${EO}/COST_POLFUNC/polfunc_fitted.csv"; POLICY_NAME="polfunc_fitted_20261016.csv"
OLD_DATA_ZIP="/c/Users/pedro/.claude/projects/c--Users-pedro-OneDrive-Documentos-Yale-Year-3--2024---2025--Open-Finance-Open-Finance-Code-Egan-et-al-2025-Rep/checkpoints_20260928/stageC_launch_20261012/bundle_final_20261013/stagec_data_20261013.zip"
old_policy () {   # old_policy -> the path of the 09-30 policy (sha256 fd208b5f...), extracted once
    local f="${SBX}/fixtures/polfunc_fitted_liverows_20260930.csv"
    [[ -s "${f}" ]] || { mkdir -p "${SBX}/fixtures"; "${PYW}" -c 'import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); open(sys.argv[2], "wb").write(z.read("data/input/polfunc_fitted_liverows_20260930.csv"))' "${OLD_DATA_ZIP}" "${f}"; }
    printf '%s' "${f}"
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
    cp "${POLICY_SRC}" "${D}/input/${POLICY_NAME}"
    for f in forward_rf_qoq.csv forward_rf_vintages.csv bbl_transitions.json; do
        cp "${EO}/COST_FWD/${f}" "${D}/input/${f}"
    done
    if [[ "${SRC}" != "${REPO}" ]]; then   # the deployed 10-13 design (section P) shipped slim estimates + markers
        for k in 1 2 3 4; do
            echo "slim E${k}" > "${D}/input/sleep_est_E${k}_estimation_results.pkl"
            printf '{"routine": %s}\n' "${k}" > "${D}/input/sleep_est_E${k}_spec12_slim.json"
        done
    fi
    ( cd "${SB}" && sha256sum scripts/cluster_lib.sh scripts/sleep_run.sh scripts/stagec_suite_20261012.sh scripts/Manifest.toml scripts/Manifest-v1.11.toml data/input/* ) > "${S}/stagec_arrival_T.sha256"
    cat > "${S}/stagec_pins_20261012.env" <<EOF
ARRIVAL_LIST=stagec_arrival_T.sha256
CHARS_NAME=demand_chars_T.parquet
CHARS_SHA256=$(sha256sum "${D}/input/demand_chars_T.parquet" | cut -d' ' -f1)
SIGMA_SHA256=$(sha256sum "${D}/input/demographics_sigma.parquet" | cut -d' ' -f1)
POLICY_UPLOAD=${POLICY_NAME}
JL_MANIFEST_SHA256=$(sha256sum "${S}/Manifest-v1.11.toml" | cut -d' ' -f1)
PY_REQ="numpy, pandas"
CONDA_CREATE="module load miniconda && conda create -y -n dep_comp_blp --override-channels --strict-channel-priority -c conda-forge python=3.11 numpy pandas"
EOF
    if [[ "${SRC}" != "${REPO}" ]]; then   # ... and its pin file named them, with the local expectations
        local est="" mk="" k
        for k in 1 2 3 4; do
            est="${est} ${k}:sleep_est_E${k}_estimation_results.pkl:$(sha256sum "${D}/input/sleep_est_E${k}_estimation_results.pkl" | cut -d' ' -f1)"
            mk="${mk} ${k}:sleep_est_E${k}_spec12_slim.json:$(sha256sum "${D}/input/sleep_est_E${k}_spec12_slim.json" | cut -d' ' -f1)"
        done
        printf 'EST_UPLOADS=(%s)\nEST_MARKERS=(%s)\nEXPECT_ROWS=(1:10 2:10 3:10 4:10)\nEXPECT_ALPHA=(E3_full_dtype:-0.0628 E4_full_dtype:-0.0567)\n' "${est# }" "${mk# }" >> "${S}/stagec_pins_20261012.env"
    fi
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
echo "== A0 the cluster's starting shape (user, 10-15): dep_comp holds only EMPTY data/ and scripts/; the documented unzip; then the launch"
build_sandbox; reset_stubs
# the two zips as the build writes them: members scripts/... and data/input/..., relative to the project root, no logs/
ZD="${SBX}/zips"; rm -rf "${ZD}"; mkdir -p "${ZD}"
"${PYW}" -c 'import os, sys, zipfile
root, zd = sys.argv[1], sys.argv[2]
for name, top in (("stagec_code_T.zip", "scripts"), ("stagec_data_T.zip", "data/input")):
    with zipfile.ZipFile(os.path.join(zd, name), "w", zipfile.ZIP_DEFLATED) as zf:
        for d, _sub, files in os.walk(os.path.join(root, top)):
            rel = os.path.relpath(d, root).replace(os.sep, "/")
            if rel == "scripts/logs" or rel.startswith("scripts/logs/"):
                continue
            for f in files:
                zi = zipfile.ZipInfo(rel + "/" + f, date_time=(2026, 10, 13, 0, 0, 0))
                zf.writestr(zi, open(os.path.join(d, f), "rb").read())' "${SB}" "${ZD}"
rm -rf "${SB}"; mkdir -p "${SB}/data" "${SB}/scripts"
check "A0 the starting shape: dep_comp/{data,scripts}, both empty" test -z "$(ls -A "${SB}/data")" -a -z "$(ls -A "${SB}/scripts")" -a "$(ls -A "${SB}" | tr '\n' ' ')" == "data scripts "
# the documented step 2 (README / header), minus the md5 check of the upload: unzip both, mkdir logs, chmod
( cd "${SB}" && unzip -o -q "${ZD}/stagec_code_T.zip" && unzip -o -q "${ZD}/stagec_data_T.zip" && mkdir -p scripts/logs && chmod +x scripts/*.sh ) > "${T}/a0_unzip.log" 2>&1; rc=$?
check "A0 the unzip into the existing empty folders succeeds" test ${rc} -eq 0
check "A0 after it: data/ holds input/ only (no output/, no logs/), scripts/ holds the code and an empty logs/" \
    test "$(ls -A "${D}" | tr '\n' ' ')" == "input " -a -z "$(ls -A "${S}/logs")" -a -f "${S}/stagec_suite_20261012.sh"
run "${T}/a0_dry.log" -- CL_DRYRUN=1 STAGEC_SKIP_PYPROBE=1; rc=$?
checke "A0 the dry run on that tree: arrival (L1), queue, output guard and data/input checks (L3) pass" 'test ${rc} -eq 0 && has "STAGE C DRY RUN COMPLETE" "${T}/a0_dry.log" && hasn "FAILED" "${T}/a0_dry.log" && test ! -e "${D}/output"'
run "${T}/a0.log" -- STAGEC_SKIP_PYPROBE=1; rc=$?
checke "A0 the live launch on that tree: exit 0, install + Julia env, T4 T3 T2 submitted" 'test ${rc} -eq 0 -a "$(grep -c . "${SBL}")" -eq 3 -a "$(tags)" == "julia_env " && has "L1. the arrival" "${T}/a0.log" && hasn "FAILED" "${T}/a0.log"'
check "A0 ... the install created data/output (the step folders, the frozen set)" test -f "${D}/output/.stagec_frozen_20261012.sha256" -a -d "${D}/output/sleep"

echo "== A1 dry run on an EMPTY tree"
build_sandbox; reset_stubs
run "${T}/a1.log" -- CL_DRYRUN=1 STAGEC_SKIP_PYPROBE=1; rc=$?
check "A1 exit 0" test ${rc} -eq 0 || tail -20 "${T}/a1.log"
check "A1 completes" has "STAGE C DRY RUN COMPLETE" "${T}/a1.log"
check "A1 no live sbatch, no install, no payload" test ! -s "${SBL}" -a ! -e "${D}/output" -a ! -s "${PLOG}"
order="$(grep -o -- '-J [a-zA-Z_0-9]*' "${T}/a1.log" | awk '{print $2}' | tr '\n' ' ')"
check "A1 the graph: stagec_cpu_image, stagec_chain, stagec_gpu (no env_*, no G0 job)" test "${order}" == "stagec_cpu_image stagec_chain stagec_gpu "

echo "== A2 the launch, live (stub SLURM)"
build_sandbox; reset_stubs
run "${T}/a2.log" -- STAGEC_SKIP_PYPROBE=1; rc=$?
check "A2 exit 0" test ${rc} -eq 0 || tail -25 "${T}/a2.log"
check "A2 installed in the launch (P8): the policy, no sleepiness estimate (made on the cluster), frozen set 4 (sigma included)" test -f "${D}/output/bbl/polfunc_fitted.csv" -a "$(ls "${D}"/output/sleep/est*/estimation_results.pkl* 2>/dev/null | wc -l)" -eq 0 -a "$(grep -c . "${D}/output/.stagec_frozen_20261012.sha256")" -eq 4 -a "$(grep -c 'data/input/demographics_sigma.parquet$' "${D}/output/.stagec_frozen_20261012.sha256")" -eq 1
check "A2 the Julia env ran in the launch (P8), before any submission" test "$(tags)" == "julia_env "
check "A2 exactly 3 submissions" test "$(grep -c . "${SBL}")" -eq 3
C3="$(id_of stagec_chain)"; G2="$(id_of stagec_gpu)"
check "A2 T4 first, then T3; neither has a dependency" test "$(head -1 "${SBL}" | grep -c -- '-J stagec_cpu_image ')" -eq 1 -a "$(sed -n 2p "${SBL}" | grep -c -- '-J stagec_chain ')" -eq 1 -a -z "$(dep_of stagec_chain)" -a -z "$(dep_of stagec_cpu_image)"
check "A2 T2 afterok T3" test "$(dep_of stagec_gpu)" == "--dependency=afterok:${C3}"
grep -- "-J stagec_chain " "${SBL}" | grep -q -- "--partition=$(REQ REQ_CHAIN_PART) --time=$(REQ REQ_CHAIN_TIME) --nodes=1 --ntasks=1 --cpus-per-task=$(REQ REQ_CHAIN_CPUS) --mem=$(REQ REQ_CHAIN_MEM) --constraint=$(REQ REQ_CHAIN_PIN) --wrap cd '.*' && bash stagec_suite_20261012.sh --chain$" && ok "A2 T3 request = REQ_CHAIN_* (day 8c/96G/2h, cpugen:turin as on 10-13), --chain" || bad "A2 T3 request"
grep -- "-J stagec_gpu " "${SBL}" | grep -q -- "--partition=$(REQ REQ_GPU_PART) --gpus=$(REQ REQ_GPU_GRES) --time=$(REQ REQ_GPU_TIME) --nodes=1 --ntasks=1 --cpus-per-task=$(REQ REQ_GPU_CPUS) --mem=$(REQ REQ_GPU_MEM) --wrap cd '.*' && bash stagec_suite_20261012.sh --gpu$" && ok "A2 T2 request = REQ_GPU_* (gpu_h200 h200:1 8c/360G/2h), --gpu" || bad "A2 T2 request"
grep -- "-J stagec_cpu_image " "${SBL}" | grep -q -- "--partition=$(REQ REQ_CPUIMG_PART) --time=$(REQ REQ_CPUIMG_TIME) --nodes=1 --ntasks=1 --cpus-per-task=$(REQ REQ_CPUIMG_CPUS) --mem=$(REQ REQ_CPUIMG_MEM) --constraint=$(REQ REQ_CPUIMG_PIN) --wrap cd '.*' && bash stagec_suite_20261012.sh --cpu-image$" && ok "A2 T4 request = REQ_CPUIMG_* (day 8c/32G/1h, cpugen:turin), --cpu-image" || bad "A2 T4 request"
check "A2 the cpugen pin on T3 and T4 (the CPU chain and the CPU image, as on 10-13), not on T2" bash -c "[[ \$(grep -c -- 'cpugen\|--constraint\|-C ' '${SBL}') -eq 2 ]] && grep -- '-J stagec_cpu_image ' '${SBL}' | grep -q -- '--constraint=cpugen:turin' && grep -- '-J stagec_chain ' '${SBL}' | grep -q -- '--constraint=cpugen:turin'"
check "A2 no submitted job inherits CL_NO_CPUGEN (m1)" bash -c "! grep -l '^CL_NO_CPUGEN=' '${STATE}'/sbatch_env_*.txt"
check "A2 T3 and T2 start with the prep exports (DEMAND_CHARS_FILE, SLEEP_PREP_EXTRA, SLEEP_ACTIVE_ESTS), T4 without" bash -c "for j in stagec_chain stagec_gpu; do grep -qx 'SLEEP_ACTIVE_ESTS=1 2 3 4' '${STATE}'/sbatch_env_\${j}.txt && grep -qx 'SLEEP_PREP_EXTRA=--skip-steps 5 7 8 --only-spec-12' '${STATE}'/sbatch_env_\${j}.txt && grep -qx 'DEMAND_CHARS_FILE=${D}/input/demand_chars_T.parquet' '${STATE}'/sbatch_env_\${j}.txt || exit 1; done; ! grep -qE '^(DEMAND_CHARS_FILE|SLEEP_PREP_EXTRA|SLEEP_ACTIVE_ESTS)=' '${STATE}/sbatch_env_stagec_cpu_image.txt'"
check "A2 the launch's Julia payload is julia_env_instantiate (the CPU-sysimage-free env step)" has "^julia_env	julia_env_instantiate$" "${PLOG}"

echo "== A3 refusals"
build_sandbox; reset_stubs
run "${T}/a3a.log" -- STAGEC_SKIP_PYPROBE=1 RUNNER_FAIL=julia_env; rc=$?
checke "A3a a failing Julia env stops the launch with nothing submitted" 'test ${rc} -ne 0 -a ! -s "${SBL}" && has "the Julia environment failed" "${T}/a3a.log"'
check "A3a ... and says to launch again" has "launch again with the same line" "${T}/a3a.log"
build_sandbox; reset_stubs
run "${T}/a3b.log" -- CF_PYTHON=false
check "A3b a Python env that cannot run the estimators and the prep is refused (create line printed)" has "cannot run the sleepiness estimators and the prep" "${T}/a3b.log"
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
check "A3h ... and prints the launch line to run again (m6)" grep -qF "THE ONE LINE TO CONTINUE, once the cause is fixed: cd ${S} && mkdir -p logs && sbatch --parsable -p $(REQ REQ_LAUNCH_PART) -t $(REQ REQ_LAUNCH_TIME) -c $(REQ REQ_LAUNCH_CPUS) --mem=$(REQ REQ_LAUNCH_MEM) -J stagec_launch -o logs/stagec_launch_%j.out --wrap \"bash stagec_suite_20261012.sh\"" "${T}/a3h.log"
hdr="$(grep -m1 '^#   cd ~/project_pi_mf2263/pf382/dep_comp/scripts && mkdir -p logs && sbatch' "${REPO}/stagec_suite_20261012.sh" | sed 's|^#   cd [^ ]* ||')"
prt="$(grep -m1 'THE ONE LINE TO CONTINUE' "${T}/a3h.log" | sed 's|.*once the cause is fixed: cd [^ ]* ||')"
check "A3h the header's documented launch line = the line the launch prints = REQ_LAUNCH_* (m8)" test -n "${hdr}" -a "${hdr}" == "${prt}" -a "${hdr}" == "&& mkdir -p logs && sbatch --parsable -p $(REQ REQ_LAUNCH_PART) -t $(REQ REQ_LAUNCH_TIME) -c $(REQ REQ_LAUNCH_CPUS) --mem=$(REQ REQ_LAUNCH_MEM) -J stagec_launch -o logs/stagec_launch_%j.out --wrap \"bash stagec_suite_20261012.sh\""
check "A3 no submission and no install in any refusal" test ! -s "${SBL}" -a ! -e "${D}/output/.stagec_frozen_20261012.sha256"
fi

# ─────────────────────────────────────────────────────────────────────────────────────────
if [[ " ${SECTIONS} " == *" B "* ]]; then
echo "== B --install, --final"
build_sandbox; reset_stubs
run "${T}/b1.log" --install; rc=$?
check "B1 exit 0, the policy installed, frozen set 4" test ${rc} -eq 0 -a -f "${D}/output/bbl/polfunc_fitted.csv" -a "$(grep -c . "${D}/output/.stagec_frozen_20261012.sha256")" -eq 4
run "${T}/b2.log" --install
check "B2 a second install keeps the identical policy" test "$(grep -c 'already installed (identical)' "${T}/b2.log")" -eq 1
run "${T}/b3.log" --final
check "B3 final: frozen set unchanged" has "STAGE C FINAL: the frozen set is unchanged" "${T}/b3.log"
echo changed >> "${D}/output/bbl/polfunc_fitted.csv"
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
echo "== C1 --chain, every step passes (the sleepiness estimates first, on the cluster)"
build_sandbox; reset_stubs; run "${T}/c0.log" --install
run "${T}/c1.log" --chain; rc=$?
check "C1 exit 0" test ${rc} -eq 0 || tail -25 "${T}/c1.log"
check "C1 payload order: precompile g0, sleep E1-E4, prep g2, link, logit/draws g4, archive x2" bash -c "[[ '$(tags | sed 's/draws logit/logit draws/')' == 'precompile g0 sleep_e1 sleep_e2 sleep_e3 sleep_e4 prep g2 link logit draws g4 archive archive ' ]]"
check "C1 G0 excuses the images and the draws (T2 builds the GPU image)" has 'PF_WILL_BUILD=draws sysimage' "${PLOG}"
check "C1 the gates written ok (SC_sleep, SC_prep, SC_link, SC_logit)" bash -c "grep -q '\"ok\": true' '${D}/output/.gate_SC_sleep.json' && grep -q '\"ok\": true' '${D}/output/.gate_SC_prep.json' && grep -q '\"ok\": true' '${D}/output/.gate_SC_link.json' && grep -q '\"ok\": true' '${D}/output/.gate_SC_logit.json'"
check "C1 archives: demand_prep and logit, tag = the job id" bash -c "grep -q 'cluster_archive.sh --set demand_prep --copy --tag local' '${PLOG}' && grep -q 'cluster_archive.sh --set logit --copy --tag local' '${PLOG}'"
check "C1 PASSED line" has "STAGE C CHAIN: PASSED" "${T}/c1.log"
check "C1 the chain submits only the off-path jobs (4 bootstraps + zip, 3 grids + zip), nothing else" test "$(grep -c . "${SBL}")" -eq 9 -a "$(grep -c -- '-J stagec_gpu \|-J stagec_chain ' "${SBL}")" -eq 0
check "C1 no local reference values anywhere (no EXPECT, no 'local' comparison)" bash -c "! grep -qi 'local [0-9]\|vs local\|MISMATCH' '${T}/c1.log'"
check "C1 the prep gate prints both new drop counts (ESTBAN rival branches, cost shifter)" bash -c "grep -q 'dropped (no ESTBAN rival branches) {.rows.: 193' '${T}/c1.log' && grep -q 'dropped (no cost shifter) {.rows.: 36' '${T}/c1.log'"

echo "== C2 a demand parquet with no rows: the prep gate refuses (exit 1), the archive still runs"
build_sandbox; reset_stubs; run "${T}/c0.log" --install
run "${T}/c2.log" --chain -- RUNNER_ROWS="10 0 10 10"; rc=$?
check "C2 exit 1" test ${rc} -eq 1 || tail -20 "${T}/c2.log"
check "C2 no logit/draws after the stop" bash -c "! grep -qE '^(logit|draws)	' '${PLOG}'"
check "C2 the archive ran anyway (m1)" has "^archive	" "${PLOG}"
check "C2 the [BROKEN] line names the empty parquet" has "\[BROKEN\] E2: demand_2_spec_12.parquet has no rows" "${T}/c2.log"
check "C2 the resume line is --from gate-prep (no mismatch override exists)" grep -qF -- '--wrap "bash stagec_suite_20261012.sh --resume chain --from gate-prep"' "${T}/c2.log"
check "C2 the prep gate json says false" has '"ok": false' "${D}/output/.gate_SC_prep.json"

echo "== C3 a non-finite logit alpha: the logit gate refuses (exit 1), after the archive"
build_sandbox; reset_stubs; run "${T}/c0.log" --install
run "${T}/c3.log" --chain -- RUNNER_ALPHA=NaN; rc=$?
check "C3 exit 1" test ${rc} -eq 1
check "C3 [BROKEN] alpha not finite" has "\[BROKEN\] E3_full_dtype: alpha not finite" "${T}/c3.log"
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

echo "== C6 --chain --from gate-logit: the logit gate and the archive only"
build_sandbox; reset_stubs; run "${T}/c0.log" --install; mkparquets 10 10 10 10; mkdraws
mkdir -p "${D}/output/logit"; printf '{"E1_full_dtype": {"param_names": ["alpha", "dummy_D_type"], "theta1": [-0.06, -5.4], "se": [0.01, 0.1], "n_obs": 10}, "E2_full_dtype": {"param_names": ["alpha", "dummy_D_type"], "theta1": [-0.06, -5.4], "se": [0.01, 0.1], "n_obs": 10}, "E3_full_dtype": {"param_names": ["alpha", "dummy_D_type"], "theta1": [-0.0628, -5.4], "se": [0.01, 0.1], "n_obs": 10}, "E4_full_dtype": {"param_names": ["alpha", "dummy_D_type"], "theta1": [-0.0567, -5.5], "se": [0.01, 0.1], "n_obs": 10}}\n' > "${D}/output/logit/logit_summary_spec_12.json"
: > "${PLOG}"
mkdir -p "${D}/output/bbl"; for k in 3 4; do printf '{"t0_check": {"n_rows": 10, "max_abs_diff": 0.0, "tol": 1e-10, "pass": true}}\n' > "${D}/output/bbl/sleep_link_E${k}_spec_12.json"; done
run "${T}/c6.log" --chain --from gate-logit; rc=$?
check "C6 exit 0; payloads: precompile g0 archive archive" test ${rc} -eq 0 -a "$(tags)" == "precompile g0 archive archive "

echo "== C9 a refusal before step 0 prints the one line to continue (m6)"
build_sandbox; reset_stubs
run "${T}/c9.log" --chain; rc=$?
checke "C9 no install: refused before step 0, no payload ran, --resume chain printed" 'test ${rc} -ne 0 -a ! -s "${PLOG}" && grep -qF "THE ONE LINE TO CONTINUE, once the cause is fixed: cd ${S} && sbatch --parsable" "${T}/c9.log" && grep -qF -- "--wrap \"bash stagec_suite_20261012.sh --resume chain\"" "${T}/c9.log"'
run "${T}/c9b.log" --chain --from logit
check "C9b ... with its --from" grep -qF -- "--wrap \"bash stagec_suite_20261012.sh --resume chain --from logit\"" "${T}/c9b.log"
fi

# ─────────────────────────────────────────────────────────────────────────────────────────
# L — the sleepiness stage ON THE CLUSTER (Pedro, 10-15): the point estimates in the chain, the
# automatic gate, the summary block, the bootstraps off the critical path
if [[ " ${SECTIONS} " == *" L "* ]]; then
echo "== L1 the sleep step: E1-E4 through sleep_job.sh SLEEP_STEP=est, spec 12"
build_sandbox; reset_stubs; run "${T}/l0.log" --install
run "${T}/l1.log" --chain; rc=$?
check "L1 exit 0" test ${rc} -eq 0 || tail -25 "${T}/l1.log"
check "L1 E1: SLEEP_STEP=est SLEEP_EST=1, no spec selector (sleep_est_e1.py has none)" grep -qx "sleep_e1	env SLEEP_STEP=est SLEEP_EST=1 SLEEP_EST_EXTRA= SPEC=12 bash ${S}/sleep_job.sh" "${PLOG}"
for k in 2 3 4; do check "L1 E${k}: SLEEP_STEP=est SLEEP_EST=${k} SLEEP_EST_EXTRA=--spec12" grep -qx "sleep_e${k}	env SLEEP_STEP=est SLEEP_EST=${k} SLEEP_EST_EXTRA=--spec12 SPEC=12 bash ${S}/sleep_job.sh" "${PLOG}"; done
check "L1 the estimates come before the prep, and the prep reads them (no upload, no install)" bash -c "[[ \$(grep -n '^sleep_e4	' '${PLOG}' | cut -d: -f1) -lt \$(grep -n '^prep	' '${PLOG}' | cut -d: -f1) ]] && ! ls '${D}'/input/ | grep -q 'sleep_est_'"
echo "== L2 the summary block, one per routine, in the chain log"
for k in 1 2 3 4; do checke "L2 E${k}: effective F, second stage, mean phi_t, logit alpha (SE, rows)" "has '=== STAGE C SUMMARY E${k} (spec 12' '${T}/l1.log' && grep -A12 '=== STAGE C SUMMARY E${k} ' '${T}/l1.log' | grep -q 'effective F 13.400' && grep -A12 '=== STAGE C SUMMARY E${k} ' '${T}/l1.log' | grep -q 'mean phi_t 0.9' && grep -A30 '=== STAGE C SUMMARY E${k} ' '${T}/l1.log' | grep -q 'logit E${k}_full_dtype: alpha .* (SE 0.0100.*), rows 10'"; done
echo "== L3 the bootstraps: separate jobs, off the critical path"
check "L3 four bootstrap jobs + their archive (no weak-IV battery: deferred, not shipped)" test "$(grep -Ec -- '-J (sleep_wcb_band|sleep_ame_gate|sleep_ame_E3|sleep_ame_E4|stagec_boot_zip) ' "${SBL}")" -eq 5 -a "$(grep -c -- '-J sleep_weakiv ' "${SBL}")" -eq 0
grep -- "-J sleep_ame_E3 " "${SBL}" | grep -q -- "--dependency=afterok:$(id_of sleep_ame_gate) .*--partition=day --time=$(REQ REQ_AME_TIME) --nodes=1 --ntasks=1 --cpus-per-task=$(REQ REQ_AME_CPUS) --mem=$(REQ REQ_AME_MEM) --constraint=cpugen:turin" && ok "L3 AME E3: afterok its guard only, REQ_AME_* (64c/256G/6h), turin" || bad "L3 AME E3 request"
check "L3 wcb and the AME guard depend on nothing (only the point estimates on disk)" test -z "$(dep_of sleep_wcb_band)" -a -z "$(dep_of sleep_ame_gate)"
check "L3 their archive: set sleep, afterany all four" bash -c "grep -- '-J stagec_boot_zip ' '${SBL}' | grep -q -- '--dependency=afterany:$(id_of sleep_wcb_band):$(id_of sleep_ame_gate):$(id_of sleep_ame_E3):$(id_of sleep_ame_E4) .*cluster_archive.sh --set sleep --copy'"
check "L3 nothing of the chain or the GPU job depends on them" bash -c "! grep -E -- '-J (stagec_gpu|stagec_chain|rc_ift_)' '${SBL}' | grep -q -- 'dependency'"
check "L3 three full grids E2-E4 (none for E1: its grid runs in the chain), REQ_GRID_*, turin, no dependency" bash -c "for k in 2 3 4; do grep -- \"-J sleep_grid_E\${k} \" '${SBL}' | grep -q -- '--partition=day --time=$(REQ REQ_GRID_TIME) --nodes=1 --ntasks=1 --cpus-per-task=$(REQ REQ_GRID_CPUS) --mem=$(REQ REQ_GRID_MEM) --constraint=cpugen:turin' || exit 1; done; [[ \$(grep -c -- '-J sleep_grid_E1 ' '${SBL}') -eq 0 ]] && ! grep -- '-J sleep_grid_E' '${SBL}' | grep -q -- '--dependency'"
check "L3 each grid writes under its own root (SLEEP_GRID_ROOT=data/output/sleep_grid), the full grid (no --spec12)" bash -c "for k in 2 3 4; do grep -- \"-J sleep_grid_E\${k} \" '${SBL}' | grep -q -- \"SLEEP_STEP=est,SLEEP_EST=\${k},SLEEP_EST_EXTRA=,SPEC=12,SLEEP_GRID_ROOT=${D}/output/sleep_grid \" || exit 1; done"
check "L3 the grid archive: --grid-check (report only), then set sleep_grid, afterany the three grids" bash -c "grep -- '-J stagec_grid_zip ' '${SBL}' | grep -q -- '--dependency=afterany:$(id_of sleep_grid_E2):$(id_of sleep_grid_E3):$(id_of sleep_grid_E4) .*--grid-check; bash cluster_archive.sh --set sleep_grid --copy'"
echo "== L4 a refused bootstrap does not stop the chain"
build_sandbox; reset_stubs; run "${T}/l0.log" --install
run "${T}/l4.log" --chain -- STUB_FAIL_RE=sleep_ame_gate; rc=$?
checke "L4 the guard refused: no AME jobs, a [!] line, the chain still PASSED (exit 0)" 'test ${rc} -eq 0 && has "sleep_ame_gate was refused (the chain goes on" "${T}/l4.log" && has "STAGE C CHAIN: PASSED" "${T}/l4.log" && test "$(grep -c -- "-J sleep_ame_E" "${SBL}")" -eq 0'
echo "== L5 the automatic sleep gate fails closed"
build_sandbox; reset_stubs; run "${T}/l0.log" --install
run "${T}/l5.log" --chain -- RUNNER_PHI=1.2; rc=$?
checke "L5 phi outside [0,1]: the shape-constrained E3/E4 refused at gate-sleep (exit 1, no prep, no bootstrap); the unconstrained E1 reported, not refused" 'test ${rc} -eq 1 && has "\[BROKEN\] E3: phi_t_IV_RivalSpread_x_Tech outside \[0,1\]" "${T}/l5.log" && hasn "\[BROKEN\] E1: phi" "${T}/l5.log" && has "E1: phi_t_IV_RivalSpread_x_Tech range .*unconstrained link" "${T}/l5.log" && hasn "^prep	" "${PLOG}" && test ! -s "${SBL}" && has "\"ok\": false" "${D}/output/.gate_SC_sleep.json"'
build_sandbox; reset_stubs; run "${T}/l0.log" --install
run "${T}/l6.log" --chain -- RUNNER_SLEEP_SKIP=3; rc=$?
checke "L6 a routine without its estimate: refused, the [BROKEN] line names it" 'test ${rc} -eq 1 && has "\[BROKEN\] E3: no .*est3.estimation_results.pkl" "${T}/l6.log"'
build_sandbox; reset_stubs; run "${T}/l0.log" --install
run "${T}/l7.log" --chain -- RUNNER_FAIL=sleep_e2; rc=$?
checke "L7 an estimator fails: the chain stops at the sleep step, advice moves data/output/sleep aside" 'test ${rc} -eq 1 && hasn "^sleep_e3	" "${PLOG}" && has "move data/output/sleep aside" "${T}/l7.log"'
echo "== L8 the sleep links on the cluster (Pedro 10-15)"
build_sandbox; reset_stubs; run "${T}/l0.log" --install
run "${T}/l8.log" --chain; rc=$?
checke "L8 link payload, SC_link true, the link JSONs in data/output/bbl (none in data/input)" 'test ${rc} -eq 0 && grep -qx "link	chain_link" "${PLOG}" && has "\"ok\": true" "${D}/output/.gate_SC_link.json" && test -f "${D}/output/bbl/sleep_link_E3_spec_12.json" -a -f "${D}/output/bbl/sleep_link_E4_spec_12.json" && ! ls "${D}"/input/ | grep -q sleep_link'
for k in 3 4; do checke "L8 E${k}: the summary block carries the link's t=0 check" "grep -A40 '=== STAGE C SUMMARY E${k} ' '${T}/l8.log' | grep -q 'sleep link (BBL): t=0 check max|phi0 - phi_mt| 0 over 10 rows (tol 1e-10): PASS'"; done
build_sandbox; reset_stubs; run "${T}/l0.log" --install
run "${T}/l9.log" --chain -- RUNNER_LINK_DIFF=1e-6; rc=$?
checke "L9 a link that does not reproduce phi_mt: the chain stops at link (exit 1), no logit, the resume --from link" 'test ${rc} -eq 1 && hasn "^logit	" "${PLOG}" && has "move data/output/bbl/sleep_link_E\*_spec_12.json aside" "${T}/l9.log" && grep -qF -- "--wrap \"bash stagec_suite_20261012.sh --resume chain --from link\"" "${T}/l9.log"'
echo "== L10 --grid-check: report only"
build_sandbox; reset_stubs; mkdir -p "${D}/output"
"${PYW}" -c 'import os, sys, pandas as pd
o = sys.argv[1]; c = "phi_t_IV_RivalSpread_x_Tech"
for root, shift in (("sleep", 0.0), ("sleep_grid", 1e-7)):
    for k in (2, 3, 4):
        d = os.path.join(o, root, f"est{k}"); os.makedirs(d, exist_ok=True)
        pd.DataFrame({"year_quarter": [1, 2, 3], c: [0.9 + (shift if k == 3 else 0.0)] * 3}).to_csv(os.path.join(d, "national_phi_t.csv"), index=False)' "${D}/output"
run "${T}/l10.log" --grid-check; rc=$?
checke "L10 prints max |dphi_t| per routine (E3 differs by 1e-7, E2/E4 identical) and exits 0" 'test ${rc} -eq 0 && has "E2: spec 12 full grid vs chain: max |dphi_t| = 0.000e+00 over 3 quarters (identical)" "${T}/l10.log" && has "E3: spec 12 full grid vs chain: max |dphi_t| = 1.000e-07 over 3 quarters (DIFFERENT -- report only)" "${T}/l10.log" && has "GRID CHECK DONE (report only)" "${T}/l10.log"'
fi

# ─────────────────────────────────────────────────────────────────────────────────────────
if [[ " ${SECTIONS} " == *" R "* ]]; then
echo "== R --resume"
build_sandbox; reset_stubs; run "${T}/r0.log" --install; mkparquets 10 11 10 10
run "${T}/r1.log" --resume chain
check "R1 resume chain (from sleep) refuses a demand_prep on disk" has "data/output already holds results of this chain (resume chain): output/demand_prep" "${T}/r1.log"
run "${T}/r2.log" --resume chain --from gate-prep
check "R2 --from gate-prep needs the passed sleep gate first" has ".gate_SC_sleep.json is missing" "${T}/r2.log"
okgate SC_sleep; run "${T}/r2b.log" --resume chain --from gate-prep
check "R2b ... then .gate_G2.json" has ".gate_G2.json is missing" "${T}/r2b.log"
okgate G2; reset_stubs
run "${T}/r3.log" --resume chain --from gate-prep; rc=$?
check "R3 exit 0" test ${rc} -eq 0 || tail -20 "${T}/r3.log"
check "R3 T3 with --from gate-prep, T2 afterok it" bash -c "grep -q -- '-J stagec_chain .*--chain --from gate-prep$' '${SBL}' && [[ '$(dep_of stagec_gpu)' == '--dependency=afterok:$(id_of stagec_chain)' ]]"
check "R3 no mismatch override exists any more (no --export of one in any printed line)" bash -c "! grep -q 'ALLOW_MISMATCH' '${T}/r3.log'"
reset_stubs; run "${T}/r4.log" --resume chain --from link
check "R4 --from link needs the passed prep gate" has ".gate_SC_prep.json is missing" "${T}/r4.log"
okgate SC_prep; reset_stubs; run "${T}/r5.log" --resume chain --from logit
check "R5 --from logit needs the passed link step" has ".gate_SC_link.json is missing" "${T}/r5.log"
okgate SC_link
okgate SC_prep; okgate G4; reset_stubs
run "${T}/r6.log" --resume chain --from gate-logit
check "R6 --from gate-logit needs the draws" has "the draws are not on disk" "${T}/r6.log"
mkdraws; mkdir -p "${D}/output/logit"; echo '{}' > "${D}/output/logit/logit_summary_spec_12.json"; okgate SC_logit; reset_stubs
run "${T}/r7.log" --resume gpu; rc=$?
check "R7 resume gpu: one T2, no dependency" test ${rc} -eq 0 -a "$(grep -c . "${SBL}")" -eq 1 -a -z "$(dep_of stagec_gpu)"
okgate SC_smoke; reset_stubs
run "${T}/r8.log" --resume next-blp; rc=$?
check "R8 resume next-blp: one job" test ${rc} -eq 0 -a "$(grep -c . "${SBL}")" -eq 1 -a -n "$(id_of stagec_next_blp)"
checke "R8 ... its watch line names logs/stagec_next_blp_<id>.out, not a GPU log (m3)" 'grep -qF "logs/stagec_next_blp_$(id_of stagec_next_blp).out" "${T}/r8.log" && hasn "stagec_gpu_" "${T}/r8.log"'
reset_stubs; run "${T}/r9.log" --resume smoke
check "R9 a retired resume point is refused with the list" has "--resume takes chain \[--from sleep|gate-sleep|prep|gate-prep|link|logit|gate-logit\], gpu, next-blp, blp, next-bbl or cpu-image" "${T}/r9.log"
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
check "SG1 no job submitted from T2's tail inherits CL_NO_CPUGEN (m1)" bash -c "! grep -l '^CL_NO_CPUGEN=' '${STATE}'/sbatch_env_*.txt"
for sc in "gap:3" "missing:2" "error:4" "cold:1"; do
    gpu_setup
    run "${T}/sg_${sc%%:*}.log" --gpu -- RUNNER_SMOKE="${sc}"; rc=$?
    check "SG ${sc}: SMOKE: FAIL, exit 1, no ladder, resume gpu printed" bash -c "[[ ${rc} -ne 0 ]] && grep -q 'SMOKE: FAIL' '${T}/sg_${sc%%:*}.log' && ! grep -q -- '-J rc_ift_' '${SBL}' 2>/dev/null && grep -q -- '--resume gpu' '${T}/sg_${sc%%:*}.log'"
done
gpu_setup; run "${T}/sg_build.log" --gpu -- RUNNER_FAIL=sysimage_gpu; rc=$?
checke "SG a failed image build: exit 1, no smoke, resume gpu" 'test ${rc} -ne 0 -a "$(tags)" == "load_julia sysimage_gpu " && has "--resume gpu" "${T}/sg_build.log"'
gpu_setup; rm -f "${D}/output/.gate_SC_logit.json"; run "${T}/sg_nogate.log" --gpu
check "SG the GPU job refuses without a passed alpha gate" has ".gate_SC_logit.json is missing" "${T}/sg_nogate.log"
check "SG ... and prints --resume gpu (m6)" grep -qF "THE ONE LINE TO CONTINUE, once the cause is fixed: cd ${S} && sbatch --parsable -p $(REQ REQ_HANDOFF_PART) -t $(REQ REQ_HANDOFF_TIME) -c $(REQ REQ_HANDOFF_CPUS) --mem=$(REQ REQ_HANDOFF_MEM) -J stagec_resume -o logs/stagec_resume_%j.out --wrap \"bash stagec_suite_20261012.sh --resume gpu\"" "${T}/sg_nogate.log"
gpu_setup; rm -rf "${D}/output/blp/draws"; run "${T}/sg_nodraws.log" --gpu
checke "SG no draws: refused, --resume gpu printed (m6)" 'has "the draws are not on disk" "${T}/sg_nodraws.log" && grep -qF -- "--wrap \"bash stagec_suite_20261012.sh --resume gpu\"" "${T}/sg_nodraws.log"'
gpu_setup; echo x > "${D}/output/blp/blp_results_E3_spec_12_ext1.jls"; run "${T}/sg_guard.log" --gpu
checke "SG the output guard: refused, --resume gpu printed (m6)" 'has "data/output already holds" "${T}/sg_guard.log" && grep -qF -- "--wrap \"bash stagec_suite_20261012.sh --resume gpu\"" "${T}/sg_guard.log"'

echo "== S6 --collect-logs (m6)"
mkdir -p "${S}/logs/stagec_smoke_42_strays"; echo e > "${S}/logs/stagec_smoke_E3_42.log"; echo s > "${S}/logs/stagec_smoke_42_strays/x.json"
run "${T}/s6.log" --collect-logs
check "S6 engine logs and strays copied as .out" test -f "${S}/logs/stagec_smoke_E3_42.log.out" -a -f "${S}/logs/stagec_smoke_42_strays__x.json.out"
fi

# ─────────────────────────────────────────────────────────────────────────────────────────
# P — PAYLOAD EQUIVALENCE. The deployed design submitted each payload as its own job: its
# environment = the submitting process's environment (the stub dumps it) + the --export list +
# SLURM_CPUS_PER_TASK of that job. The new design runs them inside T3/T2/T4: the runner dumps the
# environment each payload sees. Every job of both designs is replayed with the starting environment
# its submitter gave it (inherit: the stub's dump of that submission), so what a launch or a hand-off
# exported reaches its jobs as on the cluster. Both runs use the SAME sandbox path.
if [[ " ${SECTIONS} " == *" P "* ]]; then
echo "== P payload equivalence against the deployed design (${DEPLOYED_SCRIPTS})"
GOLD="${SBX}/gold"; NEWD="${SBX}/new"; rm -rf "${GOLD}" "${NEWD}"; mkdir -p "${GOLD}" "${NEWD}"
REL='^(SLEEP_|DEMAND_|SPEC=|R=|SEED=|OPEN_FINANCE|STATE_CENTERING|CONDA_ENV=|PY_MODULE=|JULIA_MODULE=|JULIA_PKG_OFFLINE=|OMP_|MKL_|OPENBLAS_|BLP_|CF_STAGE|CF_EVOL|COST_|PYTHON|SLURM_CPUS_PER_TASK=|CL_DATA_ROOT=|DATA_ROOT=|CL_ROOT=|ENV_STEP=|PF_)'
PREPV='^(DEMAND_CHARS_FILE|SLEEP_PREP_EXTRA|SLEEP_ACTIVE_ESTS)='
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
      echo "SLURM_CPUS_PER_TASK=${cpus:-$2}"; } | tac | awk -F= '!seen[$1]++' | grep -E "${REL}" | sort
}
new_env () {   # new_env <payload tag> <cpus of the job it runs in>
    { cat "${NEWD}/payload_env_$1.txt"; grep -q '^SLURM_CPUS_PER_TASK=' "${NEWD}/payload_env_$1.txt" || echo "SLURM_CPUS_PER_TASK=$2"; } | grep -E "${REL}" | sort
}
inherit () { grep -E "${REL}" "$1" | grep -v '^SLURM_'; }   # a job's starting environment (what its submitter exported)
cmp_env () {   # cmp_env <tag> <label> [extra grep -Ev filter]: diff the two .env files of <tag>
    local f="${3:-^$}"
    grep -Ev "${f}" "${T}/p_gold_$1.env" > "${T}/p_gold_$1.cmp"; grep -Ev "${f}" "${T}/p_new_$1.env" > "${T}/p_new_$1.cmp"
    if diff "${T}/p_gold_$1.cmp" "${T}/p_new_$1.cmp" > "${T}/p_diff_$1.txt"; then ok "P $1: $2 ($(grep -c . "${T}/p_new_$1.cmp") vars)"
    else bad "P $1: $2 -- DIFFERS (${T}/p_diff_$1.txt)"; head -10 "${T}/p_diff_$1.txt"; fi
}
gold_run () {   # gold_run <log> <mode args...> [-- env...]: the deployed wrapper in the gold sandbox
    local lg="$1"; shift; local args=() envs=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do args+=("$1"); shift; done
    [[ "${1:-}" == "--" ]] && { shift; envs=("$@"); }
    ( cd "${S}" && env PATH="${STUBS}:${PATH}" STUB_DIR="${STATE}" CL_ROOT="${S}" DATA_ROOT="${D}" PY_MODULE="" CONDA_ENV="" CF_PYTHON="${PYW}" \
        STAGEC_SLEEP_EXTRA="--skip-preflight" STAGEC_BLP_EXTRA="--skip-preflight" PY_PREFLIGHT=0 STAGEC_LOCAL_TEST=1 \
        ${envs[@]+"${envs[@]}"} bash stagec_suite_20261012.sh ${args[@]+"${args[@]}"} ) > "${lg}" 2>&1
}
# ── the deployed run: the launch (env_julia, both images, install, G0, the prep hand-off), the install
#    job, the prep hand-off job (every chain payload), then the BLP hand-off job (the ladders) ──
POLICY_SRC="$(old_policy)" POLICY_NAME="polfunc_fitted_liverows_20260930.csv" build_sandbox "${DEPLOYED_SCRIPTS}" "${DEPLOYED_SCRIPTS}/stagec_suite_20261012.sh"; reset_stubs
gold_run "${T}/p_gold_launch.log" -- STAGEC_SKIP_PYPROBE=1
mapfile -t IH < <(inherit "${STATE}/sbatch_env_stagec_install.txt"); gold_run "${T}/p_gold_install.log" --install -- "${IH[@]}"
mapfile -t IH < <(inherit "${STATE}/sbatch_env_stagec_next_prep.txt"); gold_run "${T}/p_gold_prep.log" --next-prep -- "${IH[@]}"
cp "${STATE}"/sbatch_env_*.txt "${STATE}/sbatch.log" "${GOLD}/" 2>/dev/null
mkdraws; mapfile -t IH < <(inherit "${GOLD}/sbatch_env_stagec_next_blp.txt"); gold_run "${T}/p_gold_blp.log" --next-blp -- "${IH[@]}"
grep -- '-J rc_ift_' "${STATE}/sbatch.log" | sed 's/--dependency=[a-z]*:[0-9:]*//' > "${GOLD}/ladders.txt"
for n in rc_ift_E3_head rc_ift_E1_head; do cp "${STATE}/sbatch_env_${n}.txt" "${GOLD}/ladder_env_${n}.txt" 2>/dev/null; done
check "P0 the deployed launch submitted its warm-up (env_julia, both images, G0) and the prep hand-off its chain" \
    test -s "${GOLD}/sbatch_env_env_julia.txt" -a -s "${GOLD}/sbatch_env_env_sysimage_gpu.txt" -a -s "${GOLD}/sbatch_env_env_sysimage_cpu.txt" -a -s "${GOLD}/sbatch_env_pipe_G0.txt" \
         -a -s "${GOLD}/sbatch_env_sleep_prep.txt" -a -s "${GOLD}/sbatch_env_blp_logit.txt" -a -s "${GOLD}/sbatch_env_blp_draws.txt" -a -s "${GOLD}/sbatch_env_stagec_smoke.txt" -a -s "${GOLD}/ladders.txt"
# ── the new run: the launch (install + Julia env in it; T4, T3, T2 submitted), then T3, T4 and T2,
#    each with the environment the launch gave it ──
build_sandbox; reset_stubs; run "${T}/p_new_launch.log" -- STAGEC_SKIP_PYPROBE=1
cp "${STATE}"/sbatch_env_*.txt "${STATE}/payload_env_julia_env.txt" "${NEWD}/" 2>/dev/null; cp "${STATE}/sbatch.log" "${NEWD}/sbatch_launch.log"
reset_stubs; mapfile -t IH < <(inherit "${NEWD}/sbatch_env_stagec_chain.txt"); run "${T}/p_new_chain.log" --chain -- "${IH[@]}"
cp "${STATE}"/payload_env_*.txt "${NEWD}/" 2>/dev/null; cp "${STATE}/payload.log" "${NEWD}/payload.log"
reset_stubs; mapfile -t IH < <(inherit "${NEWD}/sbatch_env_stagec_cpu_image.txt"); run "${T}/p_new_cpuimg.log" --cpu-image -- "${IH[@]}"
cp "${STATE}/payload_env_sysimage_cpu.txt" "${NEWD}/" 2>/dev/null; cp "${STATE}/payload.log" "${NEWD}/payload_cpuimg.log"
mkdraws; okgate SC_logit; reset_stubs; mapfile -t IH < <(inherit "${NEWD}/sbatch_env_stagec_gpu.txt"); run "${T}/p_new_gpu.log" --gpu -- "${IH[@]}"
cp "${STATE}/payload_env_smoke.txt" "${STATE}/payload_env_sysimage_gpu.txt" "${NEWD}/" 2>/dev/null; cp "${STATE}/payload.log" "${NEWD}/payload_gpu.log"
grep -- '-J rc_ift_' "${STATE}/sbatch.log" | sed 's/--dependency=[a-z]*:[0-9:]*//' > "${NEWD}/ladders.txt"
for n in rc_ift_E3_head rc_ift_E1_head; do cp "${STATE}/sbatch_env_${n}.txt" "${NEWD}/ladder_env_${n}.txt" 2>/dev/null; done
check "P0 the new launch, chain, CPU-image and GPU jobs ran (payload environments captured)" \
    test -s "${NEWD}/payload_env_julia_env.txt" -a -s "${NEWD}/payload_env_prep.txt" -a -s "${NEWD}/payload_env_g0.txt" -a -s "${NEWD}/payload_env_sysimage_cpu.txt" -a -s "${NEWD}/payload_env_sysimage_gpu.txt" -a -s "${NEWD}/payload_env_smoke.txt"
# what the launches handed their jobs: the prep exports reach the chain and GPU jobs (as they reached
# every 10-13 job the prep hand-off submitted: the smoke, the BLP hand-off, the ladders), not T4
grep -E "${PREPV}" "${GOLD}/sbatch_env_stagec_smoke.txt" | sort > "${T}/p_gold_prepv.txt"
check "P launch: T3 and T2 start with the prep exports the 10-13 smoke job had (3 variables)" \
    bash -c "[[ \$(grep -c . '${T}/p_gold_prepv.txt') -eq 3 ]] && diff -q '${T}/p_gold_prepv.txt' <(grep -E '${PREPV}' '${NEWD}/sbatch_env_stagec_gpu.txt' | sort) >/dev/null && diff -q '${T}/p_gold_prepv.txt' <(grep -E '${PREPV}' '${NEWD}/sbatch_env_stagec_chain.txt' | sort) >/dev/null"
check "P launch: T4 starts without them, as the 10-13 env_sysimage_cpu job did" \
    bash -c "! grep -qE '${PREPV}' '${NEWD}/sbatch_env_stagec_cpu_image.txt' && ! grep -qE '${PREPV}' '${GOLD}/sbatch_env_env_sysimage_cpu.txt'"
# ── where they run: the 10-13 chain jobs were all pinned to turin by cl_sbatch's default; T3 now
#    carries the same pin, T4 the CPU-image job's, T2 (the GPU build + smoke) none, as env_sysimage_gpu
#    and stagec_smoke had none ──
pin_of () { grep -- "-J $2 " "$1" | head -1 | grep -o -- '--constraint=[^ ]*' | head -1; }
gp=""; for j in sleep_prep sleep_gate_G2 blp_logit sleep_gate_G4 blp_draws pipe_G0; do gp="${gp}${j}=$(pin_of "${GOLD}/sbatch.log" "${j}") "; done
check "P pins: every 10-13 chain payload ran on cpugen:turin, and T3 carries the same pin" \
    test "${gp}" == "sleep_prep=--constraint=cpugen:turin sleep_gate_G2=--constraint=cpugen:turin blp_logit=--constraint=cpugen:turin sleep_gate_G4=--constraint=cpugen:turin blp_draws=--constraint=cpugen:turin pipe_G0=--constraint=cpugen:turin " -a "$(pin_of "${NEWD}/sbatch_launch.log" stagec_chain)" == "--constraint=cpugen:turin"
check "P pins: T4 = env_sysimage_cpu (cpugen:turin); T2 = env_sysimage_gpu = stagec_smoke (none)" \
    test "$(pin_of "${NEWD}/sbatch_launch.log" stagec_cpu_image)" == "$(pin_of "${GOLD}/sbatch.log" env_sysimage_cpu)" -a -n "$(pin_of "${GOLD}/sbatch.log" env_sysimage_cpu)" -a -z "$(pin_of "${NEWD}/sbatch_launch.log" stagec_gpu)" -a -z "$(pin_of "${GOLD}/sbatch.log" env_sysimage_gpu)" -a -z "$(pin_of "${GOLD}/sbatch.log" stagec_smoke)"
# ── the chain's payloads. prep/gates: their cpus never reach the payload's arithmetic (the prep's
#    parallelism is DEMAND_PREP_JOBS; sleep_job.sh prints the count only), so SLURM_CPUS_PER_TASK is
#    left out; logit and draws read it for --threads, so it is compared ──
for spec in "sleep_prep:16:prep:nocpu" "sleep_gate_G2:2:g2:nocpu" "blp_logit:8:logit:cpu" "sleep_gate_G4:2:g4:nocpu" "blp_draws:8:draws:cpu"; do
    IFS=: read -r job cpu tag mode <<< "${spec}"
    gold_env "${job}" "${cpu}" > "${T}/p_gold_${tag}.env"; new_env "${tag}" "$(REQ REQ_CHAIN_CPUS)" > "${T}/p_new_${tag}.env"
    if [[ "${mode}" == "nocpu" ]]; then cmp_env "${tag}" "the payload's environment equals the deployed ${job} job's" '^SLURM_CPUS_PER_TASK='
    else cmp_env "${tag}" "the payload's environment equals the deployed ${job} job's"; fi
done
check "P cmd: prep = sleep_job.sh, SLEEP_STEP=prep SPEC=12 DEMAND_PREP_JOBS=4 SLEEP_ACTIVE_ESTS=1 2 3 4 (sleep_run.sh's --export)" grep -qx "prep	env SLEEP_STEP=prep SPEC=12 DEMAND_PREP_JOBS=4 SLEEP_ACTIVE_ESTS=1 2 3 4 bash ${S}/sleep_job.sh" "${NEWD}/payload.log"
check "P cmd: G2/G4 = sleep_job.sh SLEEP_STEP=gate (the gates sleep_run.sh / the 10-13 graph submitted)" bash -c "grep -qx 'g2	env SLEEP_STEP=gate SLEEP_GATE=G2 SLEEP_GATE_ROUTINES=1 2 3 4 SPEC=12 bash ${S}/sleep_job.sh' '${NEWD}/payload.log' && grep -qx 'g4	env SLEEP_STEP=gate SLEEP_GATE=G4 SLEEP_GATE_ROUTINES=1 2 3 4 SPEC=12 bash ${S}/sleep_job.sh' '${NEWD}/payload.log'"
check "P cmd: logit = logit_job.sh, draws = blp_draws_job.sh" bash -c "grep -q '^logit	.* bash .*/logit_job.sh$' '${NEWD}/payload.log' && grep -q '^draws	.* bash .*/blp_draws_job.sh$' '${NEWD}/payload.log'"
# ── G0: env_job.sh ENV_STEP=preflight, now step 0 of T3. Intended differences: PF_WILL_BUILD adds
#    "sysimage" (both images are built later, by T2 and T4); the job's cpus (2 -> 8: the preflight
#    runs no threaded work); the prep exports (no reader in env_job.sh or cluster_preflight.sh, and
#    SLEEP_ACTIVE_ESTS equals the lineup default "1 2 3 4" any Python reader falls back to) ──
gold_env pipe_G0 2 > "${T}/p_gold_g0.env"; new_env g0 "$(REQ REQ_CHAIN_CPUS)" > "${T}/p_new_g0.env"
check "P G0: PF_WILL_BUILD = 'draws' (10-13) -> 'draws sysimage' (T3 runs it before T2/T4 build the images)" \
    bash -c "grep -qx 'PF_WILL_BUILD=draws' '${T}/p_gold_g0.env' && grep -qx 'PF_WILL_BUILD=draws sysimage' '${T}/p_new_g0.env'"
check "P G0: no reader of the prep exports in env_job.sh / cluster_preflight.sh; the lineup default is 1 2 3 4" \
    bash -c "! grep -qE 'DEMAND_CHARS_FILE|SLEEP_PREP_EXTRA|SLEEP_ACTIVE_ESTS' '${REPO}/env_job.sh' '${REPO}/cluster_preflight.sh' && grep -qx 'active = \[1, 2, 3, 4\]' '${REPO}/config/routines.toml'"
cmp_env g0 "the preflight's environment equals the deployed pipe_G0 job's, but for those" "^(SLURM_CPUS_PER_TASK|PF_WILL_BUILD|DEMAND_CHARS_FILE|SLEEP_PREP_EXTRA|SLEEP_ACTIVE_ESTS)="
# ── the Julia environment: the launch's instantiate vs the deployed env_julia job (whose precompile
#    moved to T3 step 0 and to the image builds). Instantiate runs no threaded work: cpus left out ──
gold_env env_julia 8 > "${T}/p_gold_julia_env.env"; new_env julia_env "$(REQ REQ_LAUNCH_CPUS)" > "${T}/p_new_julia_env.env"
cmp_env julia_env "the launch's instantiate sees the deployed env_julia job's environment" '^SLURM_CPUS_PER_TASK='
# ── the images: T4's build vs env_sysimage_cpu (everything, cpus included: env_job.sh builds with
#    NPROC=SLURM_CPUS_PER_TASK threads); T2's build vs env_sysimage_gpu (the prep exports T2 carries
#    for its smoke are left out: no reader in env_job.sh or the build scripts) ──
gold_env env_sysimage_cpu 8 > "${T}/p_gold_sysimage_cpu.env"; new_env sysimage_cpu "$(REQ REQ_CPUIMG_CPUS)" > "${T}/p_new_sysimage_cpu.env"
cmp_env sysimage_cpu "T4's build sees the deployed env_sysimage_cpu job's environment, cpus included"
check "P cmd: T4 builds with env_job.sh ENV_STEP=sysimage_cpu (as env_sysimage_cpu did)" grep -qx "sysimage_cpu	env ENV_STEP=sysimage_cpu bash ${S}/env_job.sh" "${NEWD}/payload_cpuimg.log"
gold_env env_sysimage_gpu 8 > "${T}/p_gold_sysimage_gpu.env"; new_env sysimage_gpu "$(REQ REQ_GPU_CPUS)" > "${T}/p_new_sysimage_gpu.env"
check "P GPU image: no reader of the prep exports in env_job.sh, blp_build_sysimage.jl, blp_build_sysimage_workload.jl" \
    bash -c "! grep -qE 'DEMAND_CHARS_FILE|SLEEP_PREP_EXTRA|SLEEP_ACTIVE_ESTS' '${REPO}/env_job.sh' '${REPO}/blp_build_sysimage.jl' '${REPO}/blp_build_sysimage_workload.jl'"
cmp_env sysimage_gpu "T2's build sees the deployed env_sysimage_gpu job's environment, cpus included, but for the prep exports" "${PREPV}"
check "P cmd: T2 builds with env_job.sh ENV_STEP=sysimage_gpu (as env_sysimage_gpu did)" grep -qx "sysimage_gpu	env ENV_STEP=sysimage_gpu bash ${S}/env_job.sh" "${NEWD}/payload_gpu.log"
# ── the smoke: the deployed per-routine call vs blp_smoke.jl's plan ──
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
} | tac | awk -F= '!seen[$1]++' | grep -E "${REL}" | sort > "${T}/p_gold_smoke.env"
grep -E "${REL}" "${NEWD}/payload_env_smoke.txt" | sort > "${T}/p_new_smoke.env"
cmp_env smoke "blp_smoke.jl sees the deployed smoke job's environment" '^SLURM_CPUS_PER_TASK='
# ── the ladders: the same submissions (job ids and dependencies removed), the same environments ──
if diff "${GOLD}/ladders.txt" "${NEWD}/ladders.txt" > "${T}/p_diff_ladders.txt"; then ok "P ladders: the 12 rc_ift submissions are identical ($(grep -c . "${NEWD}/ladders.txt"))"
else bad "P ladders differ (${T}/p_diff_ladders.txt)"; head -6 "${T}/p_diff_ladders.txt"; fi
for n in rc_ift_E3_head rc_ift_E1_head; do
    grep -E "${REL}" "${GOLD}/ladder_env_${n}.txt" | sort > "${T}/p_gold_${n}.env"; grep -E "${REL}" "${NEWD}/ladder_env_${n}.txt" | sort > "${T}/p_new_${n}.env"
    cmp_env "${n}" "the ladder job starts with the deployed one's environment"
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
    mkdir -p "${D}/output/bbl"; for k in 3 4; do cp "${EO}/COST_FWD/sleep_link_E${k}_spec_12.json" "${D}/output/bbl/sleep_link_E${k}_spec_12.json"; done   # the chain's link step ran
    okgate SC_cpu_image   # T4 passed (the BBL hand-off and the repair check it at B0; section I tests the refusals)
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
check "D3c B0: T4's gate and the image's provenance check passed (cpu_verdict_loaded: Julia loaded, then cl_sysimage_verdict cpu)" bash -c "grep -q 'gate_SC_cpu_image.json: ok; blp_sysimage_cpu.so passes its provenance check' '${T}/d3c.log' && grep -qx 'cpu_verdict	cpu_verdict_loaded' '${PLOG}'"
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


# ─────────────────────────────────────────────────────────────────────────────────────────
# I — T4, the CPU-image job (user, 10-14 23:1x), and the BBL's check of it
if [[ " ${SECTIONS} " == *" I "* ]]; then
echo "== I1 --cpu-image (T4): build when missing, load proof, gate"
build_sandbox; reset_stubs
run "${T}/i1.log" --cpu-image; rc=$?
check "I1 exit 0" test ${rc} -eq 0 || tail -20 "${T}/i1.log"
check "I1 payloads: load_julia, sysimage_cpu (missing image), cpu_gate" test "$(tags)" == "load_julia sysimage_cpu cpu_gate "
check "I1 the image is built by env_job.sh ENV_STEP=sysimage_cpu (the 10-13 env_sysimage_cpu payload)" grep -qx "sysimage_cpu	env ENV_STEP=sysimage_cpu bash ${S}/env_job.sh" "${PLOG}"
check "I1 the load proof runs on this node (cpu_load_proof: cl_require_sysimage cpu, an image required)" grep -qx "cpu_gate	cpu_load_proof" "${PLOG}"
checke "I1 the gate says ok; CPU IMAGE: PASS" 'has "\"ok\": true" "${D}/output/.gate_SC_cpu_image.json" && has "CPU IMAGE: PASS" "${T}/i1.log"'
check "I1 T4 submits nothing" test ! -s "${SBL}"

echo "== I2 a failed build or load proof: gate false, exit 1, the resume line"
build_sandbox; reset_stubs
run "${T}/i2.log" --cpu-image -- RUNNER_FAIL=sysimage_cpu; rc=$?
checke "I2 the build fails: exit 1, no load proof, gate false" 'test ${rc} -ne 0 -a "$(tags)" == "load_julia sysimage_cpu " && has "\"ok\": false" "${D}/output/.gate_SC_cpu_image.json"'
check "I2 ... and prints --resume cpu-image" has '--wrap "bash stagec_suite_20261012.sh --resume cpu-image"' "${T}/i2.log"
reset_stubs; run "${T}/i3.log" --cpu-image -- RUNNER_FAIL=cpu_gate; rc=$?
checke "I3 the load proof fails: exit 1, gate false" 'test ${rc} -ne 0 && has "\"ok\": false" "${D}/output/.gate_SC_cpu_image.json" && has "does not pass its load check" "${T}/i3.log"'

echo "== I4 the BBL hand-off and the repair refuse without a passed CPU image"
build_sandbox; reset_stubs; run "${T}/i0.log" --install; mkfits ok; rm -f "${D}/output/.gate_SC_cpu_image.json"
run "${T}/i4a.log" --next-bbl; rc=$?
checke "I4a no gate: refused at B0, before the fits, nothing submitted" 'test ${rc} -ne 0 -a ! -s "${SBL}" && has "the CPU sysimage job (stagec_cpu_image) has not passed (.gate_SC_cpu_image.json is missing)" "${T}/i4a.log" && hasn "B1. the fits" "${T}/i4a.log"'
check "I4a ... with both lines: rebuild it, then resume the hand-off" grep -q -- '--resume cpu-image"  then: .*--resume next-bbl"' "${T}/i4a.log"
printf '{\n "gate": "SC_cpu_image",\n "ok": false,\n "detail": "CPU sysimage build failed (x.log)"\n}\n' > "${D}/output/.gate_SC_cpu_image.json"
run "${T}/i4b.log" --next-bbl; rc=$?
checke "I4b a failed gate: refused, its detail quoted" 'test ${rc} -ne 0 -a ! -s "${SBL}" && has "says: CPU sysimage build failed" "${T}/i4b.log"'
okgate SC_cpu_image
run "${T}/i4c.log" --next-bbl -- RUNNER_FAIL=cpu_verdict; rc=$?
checke "I4c a passed gate but an image that now fails its provenance check: refused" 'test ${rc} -ne 0 -a ! -s "${SBL}" && has "no longer passes its provenance check" "${T}/i4c.log"'
rm -f "${D}/output/.gate_SC_cpu_image.json"
run "${T}/i4d.log" --repair-bbl _ms983; rc=$?
checke "I4d the repair refuses too" 'test ${rc} -ne 0 -a ! -s "${SBL}" && has "the BBL repair: the CPU sysimage job" "${T}/i4d.log"'

echo "== I5 --resume cpu-image; a live T4 does not block the chain's resumes"
build_sandbox; reset_stubs; run "${T}/i0.log" --install
run "${T}/i5a.log" --resume cpu-image; rc=$?
check "I5a exit 0, one submission: stagec_cpu_image, no dependency" test ${rc} -eq 0 -a "$(grep -c . "${SBL}")" -eq 1 -a -n "$(id_of stagec_cpu_image)" -a -z "$(dep_of stagec_cpu_image)"
grep -- "-J stagec_cpu_image " "${SBL}" | grep -q -- "--partition=$(REQ REQ_CPUIMG_PART) --time=$(REQ REQ_CPUIMG_TIME) --nodes=1 --ntasks=1 --cpus-per-task=$(REQ REQ_CPUIMG_CPUS) --mem=$(REQ REQ_CPUIMG_MEM) --constraint=$(REQ REQ_CPUIMG_PIN) --wrap cd '.*' && bash stagec_suite_20261012.sh --cpu-image$" && ok "I5a the request = REQ_CPUIMG_* with the turin pin" || bad "I5a the request"
check "I5a ... T4 starts without the prep exports (as the 10-13 env_sysimage_cpu job)" bash -c "! grep -qE '^(DEMAND_CHARS_FILE|SLEEP_PREP_EXTRA|SLEEP_ACTIVE_ESTS)=' '${STATE}/sbatch_env_stagec_cpu_image.txt'"
reset_stubs; echo "4243 bbl_solve_E3_ms983" > "${STATE}/squeue.txt"
run "${T}/i5b.log" --resume cpu-image
checke "I5b refused while a BBL job is live (a rebuild under a running solve)" 'has "jobs of this project are live: 4243 bbl_solve_E3_ms983" "${T}/i5b.log" && test ! -s "${SBL}"'
reset_stubs; echo "4244 stagec_cpu_image" > "${STATE}/squeue.txt"; mkdraws; okgate SC_logit
run "${T}/i5c.log" --resume gpu; rc=$?
check "I5c a live stagec_cpu_image does not block --resume gpu" test ${rc} -eq 0 -a -n "$(id_of stagec_gpu)"
check "I5c ... and the resumed T2 starts with the prep exports (as the 10-13 smoke job)" grep -qx 'SLEEP_ACTIVE_ESTS=1 2 3 4' "${STATE}/sbatch_env_stagec_gpu.txt"
reset_stubs; echo "4244 stagec_cpu_image" > "${STATE}/squeue.txt"
run "${T}/i5d.log" --resume cpu-image
checke "I5d --resume cpu-image refuses while a T4 is live" 'has "jobs of this project are live: 4244 stagec_cpu_image" "${T}/i5d.log" && test ! -s "${SBL}"'
echo "== I6 a T4 refused at its arrival check: gate false, --resume cpu-image printed (m6)"
build_sandbox; reset_stubs; echo "# tampered" >> "${S}/sleep_run.sh"
run "${T}/i6.log" --cpu-image; rc=$?
checke "I6 refused before any payload; the gate says false with the reason" 'test ${rc} -ne 0 -a ! -s "${PLOG}" && has "\"ok\": false" "${D}/output/.gate_SC_cpu_image.json" && has "uploaded files are not the ones built locally" "${D}/output/.gate_SC_cpu_image.json"'
check "I6 ... and prints --resume cpu-image" grep -qF -- "--wrap \"bash stagec_suite_20261012.sh --resume cpu-image\"" "${T}/i6.log"
fi

# ─────────────────────────────────────────────────────────────────────────────────────────
# J — B1 (10-15 review): the CPU-image verdict where NO julia is on PATH (every hand-off started
# from a login shell). The verdict runs for real (STAGEC_PAYLOAD_REAL=cpu_verdict) on a tree that
# ships BOTH manifests, with an image whose sidecar records Manifest-v1.11.toml's sha256 (what T4
# writes). `module load` is a stub that puts a stub julia 1.11.4 on PATH, as Lmod does.
if [[ " ${SECTIONS} " == *" J "* ]]; then
echo "== J B1: the CPU-image verdict from a login shell (no julia on PATH)"
NOJL="$(tr ':' '\n' <<< "${PATH}" | while IFS= read -r d; do [[ -n "${d}" && ( -e "${d}/julia" || -e "${d}/julia.exe" ) ]] || printf '%s:' "${d}"; done)"; NOJL="${NOJL%:}"
STUBJL="${SBX}/stub_julia"; rm -rf "${STUBJL}"; mkdir -p "${STUBJL}"
printf '#!/bin/bash\ncase "${1:-}" in --version) echo "julia version 1.11.4"; exit 0 ;; esac\nprintf znver5\n' > "${STUBJL}/julia"; chmod +x "${STUBJL}/julia"
module () { case "${1:-}" in load) [[ "${STUB_MODULE_FAIL:-0}" == "1" ]] && return 1; export PATH="${STUBJL}:${PATH}" ;; esac; return 0; }
export -f module; export STUBJL
check "J0 the test PATH has no julia (as a login shell before module load)" bash -c "! env PATH='${NOJL}' bash -c 'command -v julia' >/dev/null"
jsetup () {   # an installed tree, T4's gate, an image + sidecar as T4 writes them (sidecar sha = <manifest>)
    build_sandbox; reset_stubs; run "${T}/j0.log" --install; okgate SC_cpu_image
    sleep 1; echo "image" > "${S}/blp_sysimage_cpu.so"
    printf '{\n  "julia_version": "1.11.4",\n  "cpu_target": "znver5",\n  "manifest_sha256": "%s"\n}\n' "$(sha256sum "${S}/$1" | cut -d' ' -f1)" > "${S}/blp_sysimage_cpu.so.json"
}
JENV=(PATH="${STUBS}:${NOJL}" STUBJL="${STUBJL}" STAGEC_PAYLOAD_REAL=cpu_verdict)
jsetup Manifest-v1.11.toml
# the cause, as the pre-fix wrapper called it: cl_sysimage_verdict cpu with no julia loaded
out="$(cd "${S}" && env -i PATH="${NOJL}" HOME="${HOME}" CL_ROOT="${S}" DATA_ROOT="${D}" bash -c '. ./cluster_lib.sh >/dev/null 2>&1; cl_sysimage_verdict cpu' 2>&1)"; rc=$?
checke "J1 BEFORE (no julia loaded): the verdict reads Manifest.toml and refuses a valid image" 'test ${rc} -ne 0 && grep -q "Manifest.toml CONTENT changed" <<< "${out}"'
out="$(cd "${S}" && env -i PATH="${NOJL}" HOME="${HOME}" CL_ROOT="${S}" DATA_ROOT="${D}" STUBJL="${STUBJL}" bash -c 'module () { [[ "${1:-}" == load ]] && export PATH="${STUBJL}:${PATH}"; return 0; }; . ./cluster_lib.sh >/dev/null 2>&1; cl_load_julia >/dev/null 2>&1; cl_sysimage_verdict cpu' 2>&1)"; rc=$?
checke "J1 AFTER (julia loaded first, as cpu_verdict_loaded does): Manifest-v1.11.toml, the image passes" 'test ${rc} -eq 0 && grep -q "blp_sysimage_cpu.so OK" <<< "${out}"'
B0OK="gate_SC_cpu_image.json: ok; blp_sysimage_cpu.so passes its provenance check"
run "${T}/j2a.log" --next-bbl -- "${JENV[@]}"
checke "J2a --next-bbl (both tags): B0 passes with the real verdict (then B1 stops: no fits here)" 'has "${B0OK}" "${T}/j2a.log" && has "B1. the fits" "${T}/j2a.log" && test ! -s "${SBL}"'
run "${T}/j2b.log" --next-bbl _sc983 -- "${JENV[@]}"
checke "J2b --next-bbl _sc983 (the one-tag relaunch): B0 passes" 'has "${B0OK}" "${T}/j2b.log" && has "B1. the fits" "${T}/j2b.log"'
run "${T}/j2c.log" --repair-bbl _ms983 -- "${JENV[@]}"
checke "J2c --repair-bbl _ms983: the CPU-image check passes (then: no fit record here)" 'has "${B0OK}" "${T}/j2c.log" && has "no fit record" "${T}/j2c.log"'
run "${T}/j2d.log" --resume next-bbl -- "${JENV[@]}"
checke "J2d --resume next-bbl: B0 passes" 'has "${B0OK}" "${T}/j2d.log" && has "B1. the fits" "${T}/j2d.log"'
# --resume blp: the resume job submits stagec_next_bbl with ITS environment; that job runs --next-bbl
okgate SC_smoke; mkfits ok; rm -f "${D}/output/blp/blp_results_E3_spec_12_ext1.jls" "${D}/output/blp/blp_results_E3_spec_12_extended.jls"
echo m > "${D}/output/.stagec_ladders_20261012"; touch -d '1 hour ago' "${D}/output/.stagec_ladders_20261012"
reset_stubs; run "${T}/j2e.log" --resume blp -- "${JENV[@]}"; rc=$?
JP="$(sed -n 's/^PATH=//p' "${STATE}/sbatch_env_stagec_next_bbl.txt" 2>/dev/null)"
checke "J2e --resume blp: stagec_next_bbl submitted, and its environment carries no julia" 'test ${rc} -eq 0 -a -n "${JP}" && ! env PATH="${JP}" bash -c "command -v julia" >/dev/null'
run "${T}/j2e2.log" --next-bbl -- PATH="${JP}" STUBJL="${STUBJL}" STAGEC_PAYLOAD_REAL=cpu_verdict; rc=$?
checke "J2e ... that hand-off passes B0 and goes on (fits on disk but E3: B1 stops on E3)" 'has "${B0OK}" "${T}/j2e2.log" && has "B1. the fits" "${T}/j2e2.log"'
# --resume next-blp: the resume job submits stagec_next_blp; that job submits stagec_next_bbl
jsetup Manifest-v1.11.toml; okgate SC_smoke; mkdraws
run "${T}/j2f.log" --resume next-blp -- "${JENV[@]}"; rc=$?
JP="$(sed -n 's/^PATH=//p' "${STATE}/sbatch_env_stagec_next_blp.txt" 2>/dev/null)"
checke "J2f --resume next-blp: stagec_next_blp submitted, its environment carries no julia" 'test ${rc} -eq 0 -a -n "${JP}" && ! env PATH="${JP}" bash -c "command -v julia" >/dev/null'
reset_stubs; run "${T}/j2f2.log" --next-blp -- PATH="${JP}" STUBJL="${STUBJL}" STAGEC_PAYLOAD_REAL=cpu_verdict; rc=$?
JP="$(sed -n 's/^PATH=//p' "${STATE}/sbatch_env_stagec_next_bbl.txt" 2>/dev/null)"
checke "J2f ... that job submits stagec_next_bbl, again with no julia" 'test ${rc} -eq 0 -a -n "${JP}" && ! env PATH="${JP}" bash -c "command -v julia" >/dev/null'
run "${T}/j2f3.log" --next-bbl -- PATH="${JP}" STUBJL="${STUBJL}" STAGEC_PAYLOAD_REAL=cpu_verdict
checke "J2f ... and that hand-off passes B0" 'has "${B0OK}" "${T}/j2f3.log"'
# the verdict is real: an image built on the other manifest is refused, and a failed module load is too
jsetup Manifest.toml
run "${T}/j3.log" --next-bbl -- "${JENV[@]}"
checke "J3 an image whose sidecar records the OTHER manifest: refused at B0 by the real verdict" 'has "no longer passes its provenance check" "${T}/j3.log" && has "Manifest.toml CONTENT changed" "${T}/j3.log" && test ! -s "${SBL}"'
jsetup Manifest-v1.11.toml
run "${T}/j4.log" --next-bbl -- "${JENV[@]}" STUB_MODULE_FAIL=1
checke "J4 module load fails: refused at B0 (the image cannot be checked), the cpu-image + rerun lines printed" 'has "the CPU sysimage cannot be checked" "${T}/j4.log" && has "--resume cpu-image" "${T}/j4.log" && test ! -s "${SBL}"'
unset -f module
fi

echo; echo "test_stagec: ${PASS} ok, ${FAIL} FAIL"
[[ ${FAIL} -eq 0 ]]
