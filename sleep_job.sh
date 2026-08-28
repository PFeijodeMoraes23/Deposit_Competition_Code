#!/bin/bash
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# (job-name + .out/.err + the real --time/--mem/--cpus-per-task are set per
#  submission by sleep_run.sh / pipeline_all.sh; the block above is only a floor)
# ==============================================================================
# sleep_job.sh — THE generic sleepiness worker. Dispatches on SLEEP_STEP:
#
#   est        one estimator. The routine comes from SLURM_ARRAY_TASK_ID (or
#              SLEEP_EST), so ONE array covers the lineup:
#                1 -> sleep_est_e1.py            (E1: local B-type)
#                2 -> sleep_est_e2.py            (E2: pooled B+D linear)
#                3 -> sleep_est_single.py --est 3   (E3: single index)
#                4 -> sleep_est_single.py --est 4   (E4: single index + time)
#   merge      SPEC MODE only: sleep_est_single.py --merge-specs, the one
#              writer of est{K}/estimation_results.pkl once the spec array has drained
#   prep       sleep_pipeline.py --skip-sleep — steps 5-8 (exports, the
#              universal demand prep, the spec-12 analysis, desc_3)
#   ame_gate   sleep_ame_twostage.py --theta-off for BOTH routines. The
#              regression guard: it reproduces the stored conditional numbers
#              bit-for-bit or it fails, and the full AME jobs chain afterok it.
#   ame        one routine's full two-stage AME bootstrap (--loss robust, full B)
#   upsilon    sleep_upsilon_export.py --spec 12 for each SLEEP_UPSILON_ROUTINES id —
#              the CF4 inputs, one pair per routine the CF phase will run
#   gate       one content gate (SLEEP_GATE=G1|G2|G3|G4|G7); see THE GATES below
#
# Env vars: SLEEP_STEP SLEEP_EST SLEEP_GATE SLEEP_GATE_ROUTINES SLEEP_AME_ROUTINES
#           SLEEP_UPSILON_ROUTINES SLEEP_EST_EXTRA SLEEP_PREP_EXTRA SLEEP_AME_EXTRA SPEC
#
# WHY THE ENV BLOCK IS IDENTICAL ON EVERY BRANCH
#   The Python stack has to resolve the Open-Finance data root, write each output
#   into its own step folder under data/output, and find the state-centering
#   transform. cl_bootstrap_tree + cl_export_step_dirs do all of that, unconditionally,
#   before the dispatch — one decision for every branch. Per-branch exports are how
#   one branch writes a vintage into a directory the next branch does not read: the
#   estimators write it, prep and the AME read it back, and cf_4_upsilon_export reads
#   it and the centering json to build the CF4 pair.
#
# THE GATES
#   A gate is a content check, not a file-presence check, and it writes
#   data/output/.gate_<name>.json BEFORE exiting nonzero — so a failure leaves a
#   machine-readable record of WHICH check failed, which a cancelled-by-dependency
#   successor (no log at all) cannot. They live here, in the payload, rather than
#   as --wrap one-liners: the checks embed Python with quotes and heredocs, and
#   quoting those through sbatch --wrap is where they break silently.
#   Gates NEVER take a GPU.
#
# WHAT MUST EXIST FIRST: the uploaded panels (cluster/upload_manifest.txt) and a
#   Python env carrying CL_PY_REQ_SLEEP. cluster_preflight.sh checks both.
# WHAT TO RUN NEXT: nothing directly — sleep_run.sh owns the chain.
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/cluster_lib.sh"
set -e

: "${SLEEP_STEP:?set SLEEP_STEP (est|merge|prep|ame_gate|ame|upsilon|gate)}"
case "${SLEEP_STEP}" in
    est|merge|prep|ame_gate|ame|upsilon|gate) ;;
    *) echo "Unknown SLEEP_STEP='${SLEEP_STEP}' — see the header for all seven." >&2; exit 2 ;;
esac

SPEC="${SPEC:-12}"
SLEEP_AME_ROUTINES="${SLEEP_AME_ROUTINES:-3 4}"
# The AME two-stage driver serves the single-index routines only, by construction; the CF4
# export does not. cf4_pix.jl needs an upsilon_pix/phi_nopix pair for EVERY routine that
# goes through the CF phase, and E1/E2 get theirs from sleep_upsilon_export.py's identity
# branch, which writes phi_nopix and sets exact_nopix just as the single-index branch does.
# So upsilon carries its own list — sleep_run.sh sets it to the full routine set — and
# falls back to the AME pair only when nothing supplies one.
SLEEP_UPSILON_ROUTINES="${SLEEP_UPSILON_ROUTINES:-${SLEEP_AME_ROUTINES}}"
SLEEP_EST_EXTRA="${SLEEP_EST_EXTRA:-}"
SLEEP_PREP_EXTRA="${SLEEP_PREP_EXTRA:-}"
SLEEP_AME_EXTRA="${SLEEP_AME_EXTRA:-}"

mkdir -p "${CL_ROOT}/logs"

# ── The Open-Finance skeleton, then the step dirs. Both on EVERY branch. ─────
# cl_export_step_dirs is the single site that decides where each step writes: it exports
# OPEN_FINANCE_ROOT, SLEEP_OUT_ROOT, DEMAND_PREP_DIR, CF_FOUNDATION_DIR and
# STATE_CENTERING_JSON and creates every step folder, so nothing below has to.
cl_bootstrap_tree
cl_export_step_dirs
# The venv guard re-execs into a Windows .venv it cannot find here; ENFORCED=1 is
# the documented way to tell it the interpreter is already the right one (the
# conda env cl_setup_python activates below).
export OPEN_FINANCE_VENV_ENFORCED=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
# Two BLAS threads, not a full pool: every estimator spawns its own
# ProcessPoolExecutor, and workers x a full pool oversubscribes the node and runs
# SLOWER than serial.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2

cl_setup_python "${CL_PY_REQ_SLEEP}"
cd "${CL_ROOT}"

cl_banner "Sleepiness stage | step=${SLEEP_STEP}${SLEEP_GATE:+ (${SLEEP_GATE})}" \
          "OPEN_FINANCE_ROOT = ${OPEN_FINANCE_ROOT}" \
          "SLEEP_OUT_ROOT    = ${SLEEP_OUT_ROOT}" \
          "DEMAND_PREP_DIR   = ${DEMAND_PREP_DIR}" \
          "cpus=${SLURM_CPUS_PER_TASK:-8} node=$(hostname) | $(date)"

run_py () { local s="$1"; shift; echo "+ ${PYBIN} ${s} $*"; "${PYBIN}" "${CL_ROOT}/${s}" "$@"; }

case "${SLEEP_STEP}" in

    est)
        # Two shapes, selected by SLEEP_SPEC_MODE.
        #
        # DEFAULT (routine array): SLURM_ARRAY_TASK_ID IS the routine id — the array spec is
        # built from the routine list, never from a 0-based index that then needs +1 somewhere.
        # One task per routine runs that routine's whole spec grid serially.
        #
        # SPEC MODE (SLEEP_SPEC_MODE=1, opt in with sleep_run.sh --spec-array): the routine is
        # pinned by SLEEP_EST and SLURM_ARRAY_TASK_ID is the SPEC id, so one routine's grid fans
        # out across tasks. Each task's entire write set is _specs/spec_{S}.pkl — it never opens
        # estimation_results.pkl, the band or the phi CSVs, for reading or writing. The shared
        # pickle is written once, afterwards, by SLEEP_STEP=merge. That is what makes concurrent
        # spec tasks safe: there is no shared write to race on, rather than a guarded one.
        if [[ "${SLEEP_SPEC_MODE:-0}" == "1" ]]; then
            K="${SLEEP_EST:-}"
            S="${SLURM_ARRAY_TASK_ID:-}"
            [[ -n "${K}" && -n "${S}" ]] || {
                echo "est(spec): SPEC MODE needs SLEEP_EST (routine) and SLURM_ARRAY_TASK_ID (spec id)" >&2
                exit 2; }
            case "${K}" in
                3|4) ;;
                *) echo "est(spec): --spec-id exists only on the single-index routines (3|4); got '${K}'." \
                        "E1/E2 are separate scripts with no spec selector — run them in the routine array." >&2
                   exit 2 ;;
            esac
            echo "-- E${K} spec ${S} --"
            run_py sleep_est_single.py --est "${K}" --spec-id "${S}" ${SLEEP_EST_EXTRA}
        else
            K="${SLEEP_EST:-${SLURM_ARRAY_TASK_ID:-}}"
            [[ -n "${K}" ]] || { echo "est: neither SLEEP_EST nor SLURM_ARRAY_TASK_ID is set" >&2; exit 2; }
            echo "-- E${K} --"
            case "${K}" in
                1) run_py sleep_est_e1.py       ${SLEEP_EST_EXTRA} ;;
                2) run_py sleep_est_e2.py       ${SLEEP_EST_EXTRA} ;;
                3|4) run_py sleep_est_single.py --est "${K}" ${SLEEP_EST_EXTRA} ;;
                *) echo "est: routine '${K}' is not in the lineup (1|2|3|4)" >&2; exit 2 ;;
            esac
        fi ;;

    merge)
        # SPEC MODE only: the ONLY writer of est{K}/estimation_results.pkl, the band pickle and
        # the phi CSVs. One job per routine, afterok that routine's spec array, so the
        # load-modify-write happens once in one process. Refuses with exit 1 — before writing
        # anything — if any per-spec artifact is missing, naming the specs; --allow-partial
        # overrides that deliberately.
        K="${SLEEP_EST:-}"
        [[ -n "${K}" ]] || { echo "merge: SLEEP_EST is not set" >&2; exit 2; }
        echo "-- E${K} merge-specs --"
        run_py sleep_est_single.py --est "${K}" --merge-specs ${SLEEP_MERGE_EXTRA:-} ;;

    prep)
        # Steps 5-8. --skip-sleep is what makes it steps 5-8 and not 1-8: the
        # estimators already ran as the array above, and re-running them here would
        # overwrite the array's fits with a serial re-fit that need not land in the
        # same basin.
        run_py sleep_pipeline.py --skip-sleep ${SLEEP_PREP_EXTRA} ;;

    ame_gate)
        # BLOCKING and BOTH routines in ONE job: the off path is ~5 min per routine
        # and its whole value is being a prerequisite. off_path_gate() inside the
        # driver raises on a mismatch, so `set -e` is the gate.
        for K in ${SLEEP_AME_ROUTINES}; do
            echo "-- AME off-path regression guard: E${K} --"
            run_py sleep_ame_twostage.py --est "${K}" --loss robust --theta-off
        done
        echo "AME off-path guard PASSED for routines: ${SLEEP_AME_ROUTINES}" ;;

    ame)
        K="${SLEEP_EST:-${SLURM_ARRAY_TASK_ID:-}}"
        [[ -n "${K}" ]] || { echo "ame: set SLEEP_EST (3 or 4)" >&2; exit 2; }
        # SLEEP_AME_BLAS_THREADS is deliberately NOT set here. The driver defaults it
        # to 1 before numpy is imported, and that pinned reduction order is what lets
        # the parallel run reproduce the serial one bit-for-bit. Report a value
        # arriving from the environment rather than silently honouring it.
        if [[ -n "${SLEEP_AME_BLAS_THREADS:-}" ]]; then
            echo "[!] SLEEP_AME_BLAS_THREADS='${SLEEP_AME_BLAS_THREADS}' came in from the environment."
            echo "    The driver's default is 1, and anything else changes v_hat in the last ulp."
        fi
        echo "-- AME two-stage bootstrap: E${K} (workers <- SLEEP_AME_BOOT_JOBS=${SLEEP_AME_BOOT_JOBS:-<unset>}) --"
        run_py sleep_ame_twostage.py --est "${K}" --loss robust ${SLEEP_AME_EXTRA} ;;

    upsilon)
        # READ-ONLY w.r.t. the estimation: it reads est{k}/estimation_results.pkl +
        # market_panel_phis.csv and writes the CF4 pair. Every routine in one job — it is
        # minutes per routine, and CF4 opens the pair for each routine it runs.
        for K in ${SLEEP_UPSILON_ROUTINES}; do
            echo "-- Upsilon_pix / phi^noPix export: E${K} spec ${SPEC} --"
            run_py sleep_upsilon_export.py --estim "${K}" --spec "${SPEC}"
        done ;;

    gate)
        : "${SLEEP_GATE:?set SLEEP_GATE (G1|G2|G3|G4|G7)}"
        # CL_ROOT / CL_DATA_IN / CL_DATA_OUT and the step dirs the gate reads reach the
        # embedded Python through cl_export_step_dirs; SPEC and the routine list are this
        # branch's own inputs.
        export SPEC
        export SLEEP_GATE_ROUTINES="${SLEEP_GATE_ROUTINES:-${CL_ROUTINES_ALL}}"
        echo "-- gate ${SLEEP_GATE} | routines '${SLEEP_GATE_ROUTINES}' | spec ${SPEC} --"
        "${PYBIN}" - "${SLEEP_GATE}" <<'PYGATE'
"""One content gate for the sleepiness chain.

Every gate writes data/output/.gate_<NAME>.json and only THEN exits nonzero, so a
failure leaves a machine-readable record naming the check that failed. A successor
cancelled as DependencyNeverSatisfied writes no log at all, and the .gate json is
what stands in for it.
"""
import json
import os
import re
import struct
import sys
import traceback
from pathlib import Path

GATE = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SLEEP_GATE", "")).upper()
ROOT = Path(os.environ["CL_ROOT"])
OUT = Path(os.environ["CL_DATA_OUT"])
# PREP is the sleepiness step folder: est{k}/ and Rout/ live under it. DEMAND is the
# demand-prep step folder, the ONE place the parquets are written and read. They are
# separate steps with separate producers, and the gates below follow that split — G1/G3
# under PREP, G2/G4/G7 against DEMAND — so a gate cannot certify a file the consumer
# will not open.
PREP = Path(os.environ.get("SLEEP_OUT_ROOT") or (OUT / "sleep"))
DEMAND = Path(os.environ.get("DEMAND_PREP_DIR") or (OUT / "demand_prep"))
SPEC = os.environ.get("SPEC", "12")
KS = [int(x) for x in os.environ.get("SLEEP_GATE_ROUTINES", "1 2 3 4").replace(",", " ").split()]
AME_KS = [k for k in KS if k in (3, 4)] or [3, 4]
SPEC_KEY = "IV_HausmanFull x Tech"

# The promoted local vintage, in percentage points; each is exactly
# mean(phi_t_IV_HausmanFull_x_Tech) * 100 over that routine's national_phi_t.csv.
#
# A MATCH IS THE EXPECTED OUTCOME, so a mismatch is worth investigating rather than
# shrugging at. The cluster does NOT widen the search: sleep_est_single.py
# hardcodes NLLS_N_STARTS=4, so a 128-core `day` node runs the same multistart from
# the same starts as the local machine and should land in the same basin.
#
# It stays REPORT-ONLY all the same. If the cluster fit really does differ, the
# cluster vintage is the one that supersedes -- every downstream object in this
# chain is rebuilt from it -- and a gate that stopped the run would be stopping it
# for succeeding.
PROMOTED_PHI_PP = {1: 99.9945, 2: 99.5400, 3: 96.5627, 4: 96.6814}
PHI_TOL_PP = 0.05
PHI_COL = "phi_t_IV_HausmanFull_x_Tech"

# Columns the RC-BLP engine and every CF read out of the demand parquet. A parquet
# that is missing one of them fails ~14 h downstream, inside Julia, as a KeyError.
DEMAND_COLS = ["phi_mt", "pix_exists", "entity_id", "time_id", "spread_ann",
               "share_D", "share_B_cond", "is_B", "deposit_type",
               "CodConglomeradoPrudencial", "mca_code"]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

checks = []
notes = []


def chk(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}{(' — ' + str(detail)) if detail else ''}")
    return bool(ok)


def note(name, detail):
    notes.append({"note": name, "detail": str(detail)})
    print(f"  [note] {name} — {detail}")


def demand_parquets(k):
    """Non-final demand parquets for routine k under DEMAND, by the SAME regex
    blp_logit.jl:discover_estim_strategies uses. A glob of demand_{k}_*_spec_N
    would miss E1/E2, whose prefixes carry no middle segment at all."""
    if not DEMAND.is_dir():
        return []
    pat = re.compile(rf"^demand_{k}(?:_.*)?_spec_{SPEC}\.parquet$")
    return sorted(DEMAND / f for f in os.listdir(DEMAND)
                  if pat.match(f) and "_final_" not in f)


def parquet_rows(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(str(path)).metadata.num_rows


# ── G1: the estimators landed ────────────────────────────────────────────────
def g1():
    for k in KS:
        d = PREP / f"est{k}"
        # market_panel_phis.csv is the LAST artefact each estimator writes. Gate on
        # it, never on estimation_results.pkl: the pickle is checkpointed per block,
        # so a grid that died halfway still leaves one on disk looking complete.
        csv = d / "market_panel_phis.csv"
        sz = csv.stat().st_size if csv.is_file() else 0
        chk(f"E{k}: market_panel_phis.csv > 1 MB", sz > (1 << 20),
            f"{sz} bytes at {csv}")
        pkl = d / "estimation_results.pkl"
        if not chk(f"E{k}: estimation_results.pkl present", pkl.is_file(), str(pkl)):
            continue
        try:
            import pickle
            with open(pkl, "rb") as fh:
                res = pickle.load(fh)
        except Exception as exc:
            chk(f"E{k}: estimation_results.pkl loads", False, f"{type(exc).__name__}: {exc}")
            continue
        chk(f"E{k}: estimation_results.pkl loads", True, f"{len(res)} spec cell(s)")
        cell = res.get(SPEC_KEY) if hasattr(res, "get") else None
        if not chk(f"E{k}: has '{SPEC_KEY}'", cell is not None):
            continue
        ss = cell.get("second_stage") if hasattr(cell, "get") else getattr(cell, "second_stage", None)
        chk(f"E{k}: '{SPEC_KEY}'.second_stage is not None", ss is not None)

        # REPORT-ONLY basin comparison.
        natl = d / "national_phi_t.csv"
        if not natl.is_file():
            note(f"E{k}: phi-hat", f"no national_phi_t.csv at {natl} — no comparison made")
            continue
        try:
            import pandas as pd
            nd = pd.read_csv(natl)
            if PHI_COL not in nd.columns:
                note(f"E{k}: phi-hat", f"{natl.name} has no column {PHI_COL}")
                continue
            got = float(nd[PHI_COL].mean()) * 100.0
            want = PROMOTED_PHI_PP.get(k)
            verdict = "no promoted value on file"
            if want is not None:
                verdict = ("same basin (the expected outcome)"
                           if abs(got - want) <= PHI_TOL_PP else
                           "NEW BASIN — cluster vintage supersedes, but the multistart is "
                           "IDENTICAL here (8 threads, 4 starts), so a move is a red flag "
                           "worth investigating, not routine")
            note(f"E{k}: phi-hat",
                 f"mean({PHI_COL})*100 = {got:.4f} pp vs promoted {want} pp "
                 f"(tol {PHI_TOL_PP} pp): {verdict}")
            report.setdefault("phi_hat_pp", {})[str(k)] = {
                "measured_pp": got, "promoted_pp": want,
                "tol_pp": PHI_TOL_PP, "verdict": verdict,
                "statistic": f"unweighted mean of {PHI_COL} over national_phi_t.csv, x100",
            }
        except Exception as exc:
            note(f"E{k}: phi-hat", f"could not compute: {type(exc).__name__}: {exc}")


# ── G2: exactly one demand parquet per routine, carrying the columns ─────────
def g2():
    for k in KS:
        hits = demand_parquets(k)
        # count == 1, not >= 1. Two non-final parquets for one routine is the
        # failure mode with no symptom: blp_logit.jl silently takes the NEWEST,
        # and a relabelled leftover then rides through the whole stack.
        if not chk(f"E{k}: exactly ONE non-final demand parquet", len(hits) == 1,
                   f"{len(hits)} match(es) under {DEMAND}: {[p.name for p in hits]}"):
            continue
        p = hits[0]
        try:
            import pyarrow.parquet as pq
            names = set(pq.ParquetFile(str(p)).schema_arrow.names)
        except Exception as exc:
            chk(f"E{k}: {p.name} schema readable", False, f"{type(exc).__name__}: {exc}")
            continue
        missing = [c for c in DEMAND_COLS if c not in names]
        chk(f"E{k}: {p.name} carries every required column", not missing,
            f"missing {missing}" if missing else f"{len(names)} columns, "
            f"{parquet_rows(p)} rows")
        report.setdefault("demand_parquet", {})[str(k)] = {
            "path": str(p), "rows": parquet_rows(p)}


# ── G3: the AME bootstrap's own counters ────────────────────────────────────
def g3():
    import pickle
    rout = PREP / "Rout"
    for k in AME_KS:
        cands = [rout / f"ame_twostage_est{k}_robust_if.pkl",
                 rout / f"ame_twostage_est{k}_robust.pkl"]
        hit = next((c for c in cands if c.is_file()), None)
        if not chk(f"E{k}: AME result pkl present", hit is not None,
                   f"looked for {[c.name for c in cands]} under {rout}"):
            continue
        try:
            with open(hit, "rb") as fh:
                out = pickle.load(fh)
        except Exception as exc:
            chk(f"E{k}: {hit.name} loads", False, f"{type(exc).__name__}: {exc}")
            continue
        meta = out.get("meta", {})
        B = meta.get("B")
        if not chk(f"E{k}: {hit.name} records B", isinstance(B, (int, float)) and B > 0, f"B={B}"):
            continue
        bad = []
        for s in ("congl", "quarter"):
            r = out.get(s)
            if r is None:
                bad.append(f"{s}: missing")
                continue
            for key in ("n_newton_fail", "n_fail"):
                if float(r.get(key, 0)) > 0.01 * float(B):
                    bad.append(f"{s}: {key}={r.get(key)} > 1% of B={B}")
            for key in ("n_cos_neg", "n_vsd_fail", "n_drop"):
                if float(r.get(key, 0)):
                    bad.append(f"{s}: {key}={r.get(key)}")
        chk(f"E{k}: draw-cloud counters clean", not bad, "; ".join(bad) if bad else f"B={B}")
        stored = meta.get("counters_ok")
        if stored is not None:
            note(f"E{k}: stored counters_ok", stored)
        report.setdefault("ame", {})[str(k)] = {"path": str(hit), "B": B}


# ── G4: the logit deltas match the parquets they warm-start ─────────────────
def g4():
    # One location, because there is one producer: blp_logit.jl writes the deltas and
    # the summary through logit_dir(), which is the logit step folder for both trees. The
    # RC engine warm-starts from that same directory, so what this gate reads is what the
    # engine will read — a second candidate here would let the gate certify one delta
    # while the engine silently starts cold from another.
    def delta_candidates(k):
        return [OUT / "logit" / f"logit_delta_E{k}_spec_{SPEC}.bin"]

    def summary_candidates():
        return [OUT / "logit" / f"logit_summary_spec_{SPEC}.json"]

    for k in KS:
        cands = delta_candidates(k)
        b = next((c for c in cands if c.is_file()), None)
        if not chk(f"E{k}: logit_delta_E{k}_spec_{SPEC}.bin present", b is not None,
                   "looked in " + ", ".join(str(c.parent) for c in cands)):
            continue
        try:
            with open(b, "rb") as fh:
                (n_delta,) = struct.unpack("<q", fh.read(8))
        except Exception as exc:
            chk(f"E{k}: {b.name} readable", False, f"{type(exc).__name__}: {exc}")
            continue
        hits = demand_parquets(k)
        if not chk(f"E{k}: one demand parquet to compare against", len(hits) == 1,
                   f"{len(hits)} match(es)"):
            continue
        n_rows = parquet_rows(hits[0])
        # The engine SKIPS a delta whose length does not match and starts cold,
        # silently: the run still finishes, just from the wrong warm start.
        chk(f"E{k}: delta length == parquet rows", n_delta == n_rows,
            f"{n_delta} vs {n_rows} ({hits[0].name})")
    j = next((c for c in summary_candidates() if c.is_file()), None)
    if not chk(f"logit_summary_spec_{SPEC}.json present", j is not None,
               "looked in " + ", ".join(str(c.parent) for c in summary_candidates())):
        return
    try:
        data = json.loads(j.read_text())
    except Exception as exc:
        chk("logit summary parses", False, f"{type(exc).__name__}: {exc}")
        return
    chk("logit summary parses", True, f"{len(data)} sub-model entr(ies)")
    alphas, bad = {}, []
    for key, v in data.items():
        if not isinstance(v, dict) or "param_names" not in v or "theta1" not in v:
            continue
        try:
            a = float(v["theta1"][list(v["param_names"]).index("alpha")])
        except Exception as exc:
            bad.append(f"{key}: {type(exc).__name__}")
            continue
        alphas[key] = a
        if not (a == a and abs(a) != float("inf")):
            bad.append(f"{key}: alpha={a}")
    chk("every sub-model reports a finite alpha", alphas and not bad,
        "; ".join(bad) if bad else f"{len(alphas)} alpha(s), "
        f"range [{min(alphas.values()):.5f}, {max(alphas.values()):.5f}]" if alphas else "none found")
    report["logit_alpha"] = alphas


# ── G7: the CF4 pair is exact and aligned ───────────────────────────────────
def g7():
    for k in KS:
        # THE SAME single candidate as cf4_search_dirs() in cf4_pix.jl and cl_cf4_dirs()
        # in cluster_lib.sh: the counterfactuals step folder, where sleep_upsilon_export.py
        # writes the pair and where CF4 opens it. The gate has to check the file CF4 will
        # actually read, and with one producer writing to one location that is the same
        # file by construction — no ordering to keep in step across three places.
        jc = [OUT / "counterfactuals" / f"upsilon_pix_E{k}_spec_{SPEC}.json"]
        hit = next((c for c in jc if c.is_file()), None)
        if not chk(f"E{k}: upsilon_pix json present", hit is not None,
                   "looked in " + ", ".join(str(c.parent) for c in jc)):
            continue
        try:
            u = json.loads(hit.read_text())
        except Exception as exc:
            chk(f"E{k}: upsilon_pix json parses", False, f"{type(exc).__name__}: {exc}")
            continue
        chk(f"E{k}: upsilon_pix json parses", True, str(hit))
        # exact_nopix true == a per-row phi_nopix parquet was written, and BOTH branches
        # of sleep_upsilon_export.py write one: phi_from_native for the single-index
        # routines, the closed-form identity link for E1/E2. So the check is the same for
        # all four. False means CF4 gets no per-row φ^noPix and falls back to its scalar
        # approximation — the wrong object, silently.
        chk(f"E{k}: exact_nopix is true", bool(u.get("exact_nopix")),
            f"exact_nopix={u.get('exact_nopix')!r}, link={u.get('link')!r}")
        pq_path = u.get("phi_nopix_parquet")
        cands = ([Path(pq_path)] if pq_path else []) + \
                [c.with_name(f"phi_nopix_E{k}_spec_{SPEC}.parquet") for c in jc]
        phit = next((c for c in cands if c.is_file()), None)
        if not chk(f"E{k}: phi_nopix parquet present", phit is not None,
                   f"phi_nopix_parquet={pq_path!r}"):
            continue
        dh = demand_parquets(k)
        if not chk(f"E{k}: one demand parquet to compare against", len(dh) == 1,
                   f"{len(dh)} match(es)"):
            continue
        n_phi, n_dem = parquet_rows(phit), parquet_rows(dh[0])
        chk(f"E{k}: phi_nopix rows == demand rows", n_phi == n_dem,
            f"{n_phi} vs {n_dem} ({dh[0].name})")
        # Equal ROW COUNTS are not the property CF4 needs. cf4_pix.jl builds a lookup
        # keyed by (entity_id, time_id) and errors on ANY uncovered ctx row, so two
        # files of identical length built from different vintages sail through a count
        # check and then abort inside cf4 three phases later -- after the RC ladders,
        # the BBL solves and a 121 GB buffer allocation. Check the coverage cf4_pix.jl
        # actually requires, here, at the boundary that owns it.
        try:
            import pyarrow.parquet as _pq
            _kp = _pq.read_table(phit, columns=["entity_id", "time_id"]).to_pydict()
            _kd = _pq.read_table(dh[0], columns=["entity_id", "time_id"]).to_pydict()
            _have = {(str(a), str(b)) for a, b in zip(_kp["entity_id"], _kp["time_id"])}
            _miss = sum(1 for a, b in zip(_kd["entity_id"], _kd["time_id"])
                        if (str(a), str(b)) not in _have)
            chk(f"E{k}: every demand (entity_id,time_id) present in phi_nopix",
                _miss == 0,
                f"{_miss} uncovered of {n_dem}" + ("" if _miss == 0 else
                f" -- regenerate: python sleep_upsilon_export.py --estim {k} --spec {SPEC}"))
        except Exception as exc:
            chk(f"E{k}: phi_nopix key coverage readable", False,
                f"{type(exc).__name__}: {exc}")


report = {"gate": GATE, "routines": KS, "spec": SPEC,
          "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
          "sleep_out_root": str(PREP), "demand_prep_dir": str(DEMAND),
          "data_output": str(OUT)}

try:
    fn = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G7": g7}.get(GATE)
    if fn is None:
        raise SystemExit(f"unknown gate '{GATE}' (G1|G2|G3|G4|G7)")
    print(f"== gate {GATE} ==")
    fn()
except SystemExit:
    raise
except Exception:
    checks.append({"check": f"{GATE} ran to completion", "ok": False,
                   "detail": traceback.format_exc(limit=6)})
    print(traceback.format_exc(limit=6))

report["checks"] = checks
report["notes"] = notes
report["ok"] = bool(checks) and all(c["ok"] for c in checks)

OUT.mkdir(parents=True, exist_ok=True)
dest = OUT / f".gate_{GATE}.json"
dest.write_text(json.dumps(report, indent=2, default=str))
print(f"-> {dest}")
failed = [c["check"] for c in checks if not c["ok"]]
if not checks:
    print(f"gate {GATE}: NO CHECKS RAN — treating as a failure.")
    sys.exit(1)
if failed:
    print(f"gate {GATE}: FAILED {len(failed)} of {len(checks)} check(s): {failed}")
    sys.exit(1)
print(f"gate {GATE}: PASSED all {len(checks)} check(s).")
PYGATE
        ;;

esac

echo "Sleep step ${SLEEP_STEP}${SLEEP_GATE:+ (${SLEEP_GATE})} complete: $(date)"
