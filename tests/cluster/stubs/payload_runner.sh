#!/bin/bash
# payload_runner.sh — the local test's stand-in for every payload the wrapper runs (run_payload).
# It records the call ("<tag>\t<command>" in $STUB_DIR/payload.log) and the environment the payload
# would see (leading `env A=B ...` applied) in $STUB_DIR/payload_env_<tag>.txt, then fakes the
# payload's outputs so the next step has something to read. Scenario knobs:
#   RUNNER_FAIL="<tag> ..."   these payloads exit 1
#   RUNNER_ROWS="10 10 10 10" rows of the four demand parquets the prep writes
#   RUNNER_ALPHA=-0.0628      E3_full_dtype alpha of the logit summary
#   RUNNER_SMOKE=pass | gap:<k> | missing:<k> | error:<k> | cold:<k>   the one-process smoke log
#   RUNNER_PY                 python with pandas/pyarrow (the venv)
tag="${STAGEC_PAYLOAD_TAG:?}"; d="${STUB_DIR:?}"
printf '%s\t%s\n' "${tag}" "$*" >> "${d}/payload.log"
args=("$@")
if [[ "${args[0]:-}" == "env" ]]; then
    args=("${args[@]:1}")
    while [[ "${args[0]:-}" == *=* ]]; do export "${args[0]}"; args=("${args[@]:1}"); done
fi
env | sort > "${d}/payload_env_${tag}.txt"
printf '%s\n' "${args[*]}" > "${d}/payload_cmd_${tag}.txt"
if [[ " ${RUNNER_FAIL:-} " == *" ${tag} "* ]]; then echo "payload_runner: ${tag} FAILS (RUNNER_FAIL)"; exit 1; fi
OUT="${CL_DATA_ROOT:-${DATA_ROOT:?}}/output"
case "${tag}" in
    prep)
        SC_DIR="${OUT}/demand_prep" "${RUNNER_PY:?}" - ${RUNNER_ROWS:-10 10 10 10} <<'PY'
import json, os, sys
import pandas as pd
d = os.environ["SC_DIR"]; os.makedirs(d, exist_ok=True)
names = {1: "demand_1_spec_12", 2: "demand_2_spec_12", 3: "demand_3_index_spec_12", 4: "demand_4_index_time_spec_12"}
for k in (1, 2, 3, 4):
    n = int(sys.argv[k])
    df = pd.DataFrame({"loo_borrow_assets": [0.1] * n, "mean_loo_borrow_assets": [0.1] * n,
                       "chars_src": [0] * n, "segment_src": [0] * n})
    p = os.path.join(d, names[k] + ".parquet"); df.to_parquet(p)
    json.dump({"rows": n, "x_null_rows": 0, "iv_bad_rows": 0, "chars_file": "x",
               "dropped_no_characteristics": {"rows": 96, "B": 61, "D": 35}}, open(p + ".chars_audit.json", "w"))
PY
        ;;
    g2|g4)
        g="G${tag#g}"
        printf '{\n  "gate": "%s",\n  "checks": [{"check": "x", "ok": true}],\n  "ok": true\n}\n' "${g}" > "${OUT}/.gate_${g}.json" ;;
    logit)
        mkdir -p "${OUT}/logit"
        printf '{"E3_full_dtype": {"param_names": ["alpha", "fgc_covered", "dummy_D_type"], "theta1": [%s, 1.0, -5.4]}, "E4_full_dtype": {"param_names": ["alpha", "dummy_D_type"], "theta1": [-0.0567, -5.5]}}\n' "${RUNNER_ALPHA:--0.0628}" > "${OUT}/logit/logit_summary_spec_12.json" ;;
    draws)
        mkdir -p "${OUT}/blp/draws"
        for f in demo_draws demo_key_index halton_nu; do echo d > "${OUT}/blp/draws/${f}_R2000_seed42.jls"; done ;;
    smoke)
        mode="${RUNNER_SMOKE:-pass}"; bad="${mode#*:}"
        for k in 1 2 3 4; do
            if [[ "${mode}" == missing:* && "${k}" == "${bad}" ]]; then continue; fi
            printf '\n=== SMOKE E%s BEGIN ===\n' "${k}"
            if [[ "${mode}" == error:* && "${k}" == "${bad}" ]]; then
                printf '=== SMOKE E%s ERROR: DemandInputError: stub ===\n=== SMOKE E%s END (ERROR) ===\n' "${k}" "${k}"; continue
            fi
            for s in 1 2 3; do
                n=644005; [[ "${mode}" == gap:* && "${k}" == "${bad}" && "${s}" == 2 ]] && n=632735
                if [[ "${mode}" == cold:* && "${k}" == "${bad}" ]]; then echo "  [δ WARM-START] No logit checkpoint — log-share init"
                else echo "  [δ WARM-START] Loaded logit_delta_E${k}_spec_12.bin [binary]"; fi
                echo "  Obs mapped to draws: ${n} / 644005 (100.0%)"
                echo "  [DRY RUN GPU] 10 iters: 1.2s (0.12 s/iter)"
            done
            printf '=== SMOKE E%s END (ok) ===\n' "${k}"
        done ;;
esac
exit 0
