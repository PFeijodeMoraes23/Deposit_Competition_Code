#!/bin/bash
# Switch-cost investigation copy of sleep_job.sh (copied 2026-10-13 17:47, source md5 823267f9962a010a8df54dbb131b959e).
# It writes only to the switch-cost tree; the original is untouched.
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=pedro.feijodemoraes@yale.edu
# (job-name + .out/.err + the real --time/--mem/--cpus-per-task are set per
#  submission by switch_cost_sleep_run.sh; the block above is only a floor)
# ==============================================================================
# switch_cost_sleep_job.sh — the switch-cost sleepiness worker. Dispatches on SLEEP_STEP:
#
#   est    one spec-12 fit of one routine and one fit variant:
#              switch_cost_sleep_est_single.py --est K --spec12
#          K = SLEEP_EST (3 | 4), variant = SC_VARIANT (pooled | typeshift; the library
#          exports SLEEP_OUT_ROOT and SWITCH_COST_TYPE_SHIFTS from it). With
#          SLEEP_AME_TWOSTAGE=1, which switch_cost_sleep_run.sh always submits, the two-stage
#          (direction + link) AME bootstrap runs inside the estimation pass and is stored on
#          the fit: B = SLEEP_BOOT_B, workers = SLEEP_AME_BOOT_JOBS.
#          Writes est{K}/ (estimation_results.pkl, market_panel_phis.csv, national_phi_t.csv)
#          and Rout/ under data/output/switch_cost_sleep_<variant>.
#   prep   one demand-prep arm for one routine:
#              switch_cost_sleep_demand_prep_link.py --est K --spec 12 --arm SC_ARM
#                  --fits-dir <the arm's variant folder> --pooled-fits-dir <pooled folder>
#                  [--locked-csv SC_DATA_IN/r4_locked_bank_year.csv]   (F3hi, F3lo)
#                  --out-dir data/output/switch_cost_demand_<arm>
#          The pooled arm reads the pooled fit; F1/F3 read the typeshift fit plus the pooled
#          fit (phi_0, market sizes). The demand-prep orchestrator has no switch-cost copy.
#          The job fails when no parquet was written; the prep itself exits nonzero then too.
#   gate   one content gate, written to data/output/switch_cost_gates/.gate_switch_cost_<G1_variant|G2_arm>.json:
#            G1  SC_VARIANT's fits landed (and carry the two-stage AME when it was requested)
#            G2  exactly one parquet per routine in SC_ARM's folder, with the columns
#
# DISABLED in this copy, refused with exit 2 and a message naming what the step writes:
#   merge, ame_gate, ame, wcb_band, weakiv, entry, upsilon.
#
# Env vars: SLEEP_STEP SLEEP_EST SC_VARIANT SC_ARM SLEEP_GATE SLEEP_GATE_ROUTINES SPEC
#           SLEEP_AME_TWOSTAGE SLEEP_BOOT_B SLEEP_AME_BOOT_JOBS SC_BLAS_THREADS
#
# WHY THE ENV BLOCK IS IDENTICAL ON EVERY BRANCH
#   cl_bootstrap_tree + cl_export_step_dirs build the switch-cost skeleton, export
#   OPEN_FINANCE_ROOT, SLEEP_OUT_ROOT, SWITCH_COST_TYPE_SHIFTS, DEMAND_PREP_DIR (prep jobs) and
#   the input names, and create the job's own folders, unconditionally, before the dispatch —
#   one decision for every branch, none inherited from the submitting shell.
#
# THE GATES
#   A gate is a content check, not a file-presence check, and it writes its .gate json BEFORE
#   exiting nonzero — so a failure leaves a machine-readable record of WHICH check failed,
#   which a cancelled-by-dependency successor (no log at all) cannot. Gates NEVER take a GPU.
#
# WHAT MUST EXIST FIRST: the inputs in SC_DATA_IN and a Python env carrying CL_PY_REQ_SLEEP.
#   switch_cost_sleep_run.sh checks both (sc_preflight, read-only).
# WHAT TO RUN NEXT: nothing directly — switch_cost_sleep_run.sh owns the chain.
# ==============================================================================
set -uo pipefail
CL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
. "${CL_DIR}/switch_cost_cluster_lib.sh"
set -e

: "${SLEEP_STEP:?set SLEEP_STEP (est|prep|gate)}"
# What each original step writes, for the refusal message. None of them has a switch_cost_
# Python copy, and the two-stage AME that ame_gate/ame computed post hoc runs inside `est` here.
case "${SLEEP_STEP}" in
    est|prep|gate) ;;
    merge)    DISABLED_WHAT="sleep_est_single.py --merge-specs: est{K}/estimation_results.pkl, the spec-12 band and the phi CSVs from the 5-12 grid. This copy runs spec 12 only (--spec12 writes them directly), and the merge refuses a partial grid" ;;
    ame_gate) DISABLED_WHAT="sleep_ame_twostage.py --theta-off: the off-path AME regression guard against the stored production numbers (no switch-cost copy)" ;;
    ame)      DISABLED_WHAT="sleep_ame_twostage.py: Rout/ame_twostage_est{K}_robust_if.pkl (no switch-cost copy). Here the two-stage AME runs inside est (SLEEP_AME_TWOSTAGE=1)" ;;
    wcb_band) DISABLED_WHAT="sleep_wcb_band.py: Rout/wcb_band_est{1,2}.pkl, the E1/E2 linear bands (E1/E2 are outside this copy's scope)" ;;
    weakiv)   DISABLED_WHAT="sleep_weak_iv.py: BLP_RESULTS/cluster_processed/weak_iv_sleep.json and DIAG_WEAK_IV_SLEEP/ (the BLP family)" ;;
    entry)    DISABLED_WHAT="sleep_ident_entry_dynamics.py --compute-only: DIAG_PHI_SEPARATION/ and figure fragments; reads CF_FOUNDATION/phi_nopix_E{k}" ;;
    upsilon)  DISABLED_WHAT="sleep_upsilon_export.py: upsilon_pix_E{k}_spec_12.json and phi_nopix_E{k}_spec_12.parquet in CF_FOUNDATION_DIR (the counterfactual family)" ;;
    *) echo "Unknown SLEEP_STEP='${SLEEP_STEP}' (est|prep|gate)." >&2; exit 2 ;;
esac
if [[ -n "${DISABLED_WHAT:-}" ]]; then
    echo "REFUSED: SLEEP_STEP=${SLEEP_STEP} is disabled in switch_cost_sleep_job.sh." >&2
    echo "  In the main project it runs ${DISABLED_WHAT}." >&2
    echo "  See claude/switch_cost/CLUSTER_SCRIPTS.md (disabled steps)." >&2
    exit 2
fi

SPEC="${SPEC:-12}"
if [[ "${SPEC}" != "12" ]]; then
    echo "REFUSED: SPEC=${SPEC}. This copy runs spec 12 only." >&2; exit 2
fi
# Gates are checked here too, before anything is created: G3/G4/G7 check steps this copy
# does not run (the post-hoc AME files, the logit deltas, the CF4 pair).
if [[ "${SLEEP_STEP}" == "gate" ]]; then
    case "${SLEEP_GATE:-}" in
        G1) ;;
        G2) [[ -n "${SC_ARM:-}" ]] || { echo "gate G2 needs SC_ARM." >&2; exit 2; } ;;
        G3|G4|G7) echo "REFUSED: gate ${SLEEP_GATE} checks a step this copy does not run (G3 the post-hoc AME files, G4 the logit deltas, G7 the CF4 pair)." >&2; exit 2 ;;
        *) echo "set SLEEP_GATE (G1|G2); got '${SLEEP_GATE:-}'" >&2; exit 2 ;;
    esac
fi

mkdir -p "${SC_LOG_DIR}"

# ── The switch-cost skeleton, then the step dirs. Both on EVERY branch. ──────
# cl_export_step_dirs is the single site that decides where each step writes: it exports
# OPEN_FINANCE_ROOT, SLEEP_OUT_ROOT + SWITCH_COST_TYPE_SHIFTS (from SC_VARIANT),
# DEMAND_PREP_DIR (from SC_ARM, prep jobs), CF_FOUNDATION_DIR, STATE_CENTERING_JSON and
# DEMAND_CHARS_FILE, and creates the job's folders, so nothing below has to.
cl_bootstrap_tree
cl_export_step_dirs
# The venv guard re-execs into a Windows .venv it cannot find here; ENFORCED=1 is
# the documented way to tell it the interpreter is already the right one (the
# conda env cl_setup_python activates below). PYTHONDONTWRITEBYTECODE keeps Python from
# writing __pycache__ into the shared scripts/ checkout.
export OPEN_FINANCE_VENV_ENFORCED=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg \
       PYTHONDONTWRITEBYTECODE=1
# BLAS threads: SC_BLAS_THREADS, default 1, as the local switch-cost fits ran (the estimator's own
# default) and as the production AME driver pins it. Every estimator spawns its own process pool,
# and the two-stage AME runs SLEEP_AME_BOOT_JOBS workers, so more threads per worker oversubscribe
# the node.
SC_BLAS_THREADS="${SC_BLAS_THREADS:-1}"
export OMP_NUM_THREADS="${SC_BLAS_THREADS}" MKL_NUM_THREADS="${SC_BLAS_THREADS}" \
       OPENBLAS_NUM_THREADS="${SC_BLAS_THREADS}" VECLIB_MAXIMUM_THREADS="${SC_BLAS_THREADS}" \
       NUMEXPR_NUM_THREADS="${SC_BLAS_THREADS}"

cl_setup_python "${CL_PY_REQ_SLEEP}"
cd "${CL_ROOT}"

cl_banner "Switch-cost sleepiness | step=${SLEEP_STEP}${SLEEP_GATE:+ (${SLEEP_GATE})} | variant=${SC_VARIANT}${SC_ARM:+ | arm=${SC_ARM}}" \
          "OPEN_FINANCE_ROOT       = ${OPEN_FINANCE_ROOT}" \
          "SLEEP_OUT_ROOT          = ${SLEEP_OUT_ROOT}" \
          "SWITCH_COST_TYPE_SHIFTS = ${SWITCH_COST_TYPE_SHIFTS}" \
          "DEMAND_PREP_DIR         = ${DEMAND_PREP_DIR:-<unset: not a prep job>}" \
          "SLEEP_AME_TWOSTAGE=${SLEEP_AME_TWOSTAGE:-0}  SLEEP_BOOT_B=${SLEEP_BOOT_B:-<unset: 999>}  SLEEP_AME_BOOT_JOBS=${SLEEP_AME_BOOT_JOBS:-<unset: 1>}  SLEEP_RESUME=${SLEEP_RESUME:-0}" \
          "cpus=${SLURM_CPUS_PER_TASK:-8} blas=${SC_BLAS_THREADS} node=$(hostname) | $(date)"

run_py () { local s="$1"; shift; echo "+ ${PYBIN} ${s} $*"; "${PYBIN}" "${CL_ROOT}/${s}" "$@"; }

case "${SLEEP_STEP}" in

    est)
        # One routine, one variant, spec 12. --spec12 runs the single cell and writes the
        # pickle (merge-on-save), the band and the phi CSVs itself, so there is no merge step.
        K="${SLEEP_EST:-}"
        case "${K}" in
            3|4) ;;
            *) echo "est: SLEEP_EST must be 3 or 4 (got '${K}'); E1/E2 are outside this copy's scope." >&2; exit 2 ;;
        esac
        if [[ "${SLEEP_AME_TWOSTAGE:-0}" != "1" ]]; then
            echo "[!] SLEEP_AME_TWOSTAGE is not 1: the fit is stored without the two-stage AME."
        fi
        echo "-- E${K} spec 12, variant ${SC_VARIANT} --"
        run_py switch_cost_sleep_est_single.py --est "${K}" --spec12 ;;

    prep)
        K="${SLEEP_EST:-}"
        case "${K}" in
            3|4) ;;
            *) echo "prep: SLEEP_EST must be 3 or 4 (got '${K}')." >&2; exit 2 ;;
        esac
        ARM="${SC_ARM:-}"
        FV="$(sc_arm_variant "${ARM}")" || { echo "prep: SC_ARM must be one of ${SC_ARMS} (got '${ARM}')." >&2; exit 2; }
        if [[ "${SC_VARIANT}" != "${FV}" ]]; then
            echo "prep: arm ${ARM} reads the ${FV} fit, but SC_VARIANT=${SC_VARIANT}." >&2; exit 2
        fi
        LOCKED=()
        case "${ARM}" in F3hi|F3lo) LOCKED=(--locked-csv "${SC_DATA_IN}/r4_locked_bank_year.csv") ;; esac
        t0="$(date +%s)"
        echo "-- demand prep E${K} spec 12, arm ${ARM} --"
        run_py switch_cost_sleep_demand_prep_link.py --est "${K}" --spec 12 --arm "${ARM}" \
            --fits-dir "$(sc_sleep_dir "${FV}")" --pooled-fits-dir "$(sc_sleep_dir pooled)" \
            ${LOCKED[@]+"${LOCKED[@]}"} --out-dir "${DEMAND_PREP_DIR}"
        # The prep returns 0 having written nothing when a pickle or a spec is missing, so the
        # job counts the parquet itself, by mtime, in the folder it was told to write.
        n_new="$(find "${DEMAND_PREP_DIR}" -maxdepth 1 -name "demand_${K}_*spec_${SPEC}.parquet" -newermt "@${t0}" | wc -l)"
        if [[ "${n_new}" -lt 1 ]]; then
            echo "prep: E${K} arm ${ARM} exited 0 but wrote no demand_${K}_*spec_${SPEC}.parquet in ${DEMAND_PREP_DIR}." >&2
            exit 1
        fi
        echo "prep: E${K} arm ${ARM}: ${n_new} parquet(s) written to ${DEMAND_PREP_DIR}" ;;

    gate)
        # SLEEP_GATE was validated (G1|G2) before the skeleton was built.
        # CL_ROOT / CL_DATA_OUT / SC_GATE_DIR and the step dirs the gate reads reach the
        # embedded Python through cl_export_step_dirs; SPEC and the routine list are this
        # branch's own inputs.
        export SPEC
        export SLEEP_GATE_ROUTINES="${SLEEP_GATE_ROUTINES:-3 4}"
        echo "-- gate ${SLEEP_GATE} | variant ${SC_VARIANT}${SC_ARM:+ | arm ${SC_ARM}} | routines '${SLEEP_GATE_ROUTINES}' | spec ${SPEC} --"
        "${PYBIN}" - "${SLEEP_GATE}" <<'PYGATE'
"""One content gate for the switch-cost sleepiness chain.

Every gate writes <SC_GATE_DIR>/.gate_switch_cost_<NAME>.json and only THEN exits nonzero,
so a failure leaves a machine-readable record naming the check that failed. A successor
cancelled as DependencyNeverSatisfied writes no log at all, and the .gate json is what
stands in for it. NAME is G1_<variant> or G2_<arm>.
"""
import json
import os
import re
import sys
import traceback
from pathlib import Path

GATE = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SLEEP_GATE", "")).upper()
ROOT = Path(os.environ["CL_ROOT"])
OUT = Path(os.environ["CL_DATA_OUT"])
GATE_DIR = Path(os.environ.get("SC_GATE_DIR") or OUT)
VARIANT = os.environ.get("SC_VARIANT", "")
ARM = os.environ.get("SC_ARM", "")
# PREP is the variant's sleep step folder: est{k}/ and Rout/ live under it. DEMAND is the
# arm's parquet folder. Both come from cl_export_step_dirs and must be switch-cost folders.
PREP = Path(os.environ["SLEEP_OUT_ROOT"])
DEMAND = Path(os.environ["DEMAND_PREP_DIR"]) if os.environ.get("DEMAND_PREP_DIR") else None
SPEC = os.environ.get("SPEC", "12")
KS = [int(x) for x in os.environ.get("SLEEP_GATE_ROUTINES", "3 4").replace(",", " ").split()]
SPEC_KEY = "IV_RivalSpread x Tech"
EXPECT_2S = os.environ.get("SLEEP_AME_TWOSTAGE", "0") == "1"
NAME = f"{GATE}_{VARIANT}" if GATE == "G1" else f"{GATE}_{ARM}"

for _p in [PREP] + ([DEMAND] if DEMAND else []):
    if "switch_cost" not in _p.name.lower():
        raise SystemExit(f"gate: {_p} is not a switch-cost folder; refusing to read it.")

# The LOCAL switch-cost fits of 2026-10-13, in percentage points: exactly
# mean(phi_t_IV_RivalSpread_x_Tech) * 100 over each variant's national_phi_t.csv (same code, same
# data). The local pooled values equal the local production files to four decimals. For the
# typeshift variant this national path averages the four types' phi with equal weight per market
# row, so it is a check value, not an estimand. REPORT-ONLY either way.
PROMOTED_PHI_PP = {"pooled": {3: 98.2638, 4: 98.1471}, "typeshift": {3: 92.0371, 4: 91.1434}}
PHI_TOL_PP = 0.05
PHI_COL = "phi_t_IV_RivalSpread_x_Tech"

# Columns the RC-BLP engine and every CF read out of the demand parquet.
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
    blp_logit.jl:discover_estim_strategies uses."""
    if DEMAND is None or not DEMAND.is_dir():
        return []
    pat = re.compile(rf"^demand_{k}(?:_.*)?_spec_{SPEC}\.parquet$")
    return sorted(DEMAND / f for f in os.listdir(DEMAND)
                  if pat.match(f) and "_final_" not in f)


def parquet_rows(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(str(path)).metadata.num_rows


def _ame_counters(k, meta, schemes):
    """The two-stage AME stored on a fit, through the counters the main project's gate G3
    applies to sleep_ame_twostage.py's files (sleep_ame_twostage.counters_ok)."""
    B = (meta or {}).get("B")
    if not chk(f"E{k}: two-stage AME records B", isinstance(B, (int, float)) and B > 0, f"B={B}"):
        return
    bad = []
    for s in ("congl", "quarter"):
        r = (schemes or {}).get(s)
        if r is None:
            bad.append(f"{s}: missing")
            continue
        for key in ("n_newton_fail", "n_fail"):
            if float(r.get(key, 0)) > 0.01 * float(B):
                bad.append(f"{s}: {key}={r.get(key)} > 1% of B={B}")
        if "n_cos_idx_neg" not in r:
            bad.append(f"{s}: n_cos_idx_neg missing")
        elif float(r.get("n_cos_idx_neg", 0)) > 0.01 * float(B):
            bad.append(f"{s}: n_cos_idx_neg={r.get('n_cos_idx_neg')} > 1% of B={B}")
        elif float(r.get("n_cos_idx_neg", 0)):
            note(f"E{k}/{s}: reversed-index draws retained",
                 f"{r.get('n_cos_idx_neg')} of B={B} (within the 1% tolerance)")
        for key in ("n_vsd_fail", "n_drop"):
            if float(r.get(key, 0)):
                bad.append(f"{s}: {key}={r.get(key)}")
        flat = [n for n, se in (r.get("bse") or {}).items()
                if not float(se) > 1e-10 * max(1e-300, abs(float(r["ame"][n])))]
        if flat:
            bad.append(f"{s}: degenerate draws on {flat}")
        if float(r.get("n_cos_neg", 0)):
            note(f"E{k}/{s}: raw-coefficient cos_neg",
                 f"{r.get('n_cos_neg')} of B={B} (diagnostic, not gating)")
    chk(f"E{k}: two-stage AME draw-cloud counters clean", not bad, "; ".join(bad) if bad else f"B={B}")
    report.setdefault("ame_2s", {})[str(k)] = {"B": B}


# ── G1: the variant's fits landed ───────────────────────────────────────────
def g1():
    for k in KS:
        d = PREP / f"est{k}"
        # market_panel_phis.csv is the LAST artefact each estimator writes. Gate on it, never
        # on estimation_results.pkl alone.
        csv = d / "market_panel_phis.csv"
        sz = csv.stat().st_size if csv.is_file() else 0
        chk(f"E{k}: market_panel_phis.csv > 1 MB", sz > (1 << 20), f"{sz} bytes at {csv}")
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
        if not chk(f"E{k}: '{SPEC_KEY}'.second_stage is not None", ss is not None):
            continue
        # The reason this run is on the cluster: the two-stage AME. The estimator SKIPS it with
        # only a printed line when it raises, so its absence must fail here.
        if EXPECT_2S:
            boot = getattr(ss, "ame_boot", None)
            if chk(f"E{k}: two-stage AME stored on the fit (ame_boot)", boot is not None,
                   "absent: look for '[2s-AME] SKIPPED' in the est log" if boot is None else ""):
                _ame_counters(k, getattr(ss, "ame_2s_meta", None), boot)

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
            want = PROMOTED_PHI_PP.get(VARIANT, {}).get(k)
            if want is None:
                verdict = "no local value on file"
            else:
                verdict = ("same basin as the local fit (the expected outcome)"
                           if abs(got - want) <= PHI_TOL_PP else
                           "NEW BASIN against the local fit -- worth investigating")
            note(f"E{k}: phi-hat",
                 f"mean({PHI_COL})*100 = {got:.4f} pp vs local {want} pp "
                 f"(tol {PHI_TOL_PP} pp): {verdict}")
            report.setdefault("phi_hat_pp", {})[str(k)] = {
                "measured_pp": got, "promoted_pp": want, "tol_pp": PHI_TOL_PP,
                "verdict": verdict}
        except Exception as exc:
            note(f"E{k}: phi-hat", f"could not compute: {type(exc).__name__}: {exc}")


# ── G2: exactly one demand parquet per routine in the arm's folder ───────────
def g2():
    for k in KS:
        hits = demand_parquets(k)
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
            f"missing {missing}" if missing else f"{len(names)} columns, {parquet_rows(p)} rows")
        report.setdefault("demand_parquet", {})[str(k)] = {"path": str(p), "rows": parquet_rows(p)}


report = {"gate": NAME, "routines": KS, "spec": SPEC, "variant": VARIANT, "arm": ARM,
          "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
          "sleep_out_root": str(PREP), "demand_prep_dir": str(DEMAND) if DEMAND else "",
          "data_output": str(OUT)}

try:
    fn = {"G1": g1, "G2": g2}.get(GATE)
    if fn is None:
        raise SystemExit(f"unknown gate '{GATE}' (G1|G2)")
    if GATE == "G2" and DEMAND is None:
        raise SystemExit("gate G2: DEMAND_PREP_DIR is unset")
    print(f"== gate {NAME} ==")
    fn()
except SystemExit:
    raise
except Exception:
    checks.append({"check": f"{NAME} ran to completion", "ok": False,
                   "detail": traceback.format_exc(limit=6)})
    print(traceback.format_exc(limit=6))

report["checks"] = checks
report["notes"] = notes
report["ok"] = bool(checks) and all(c["ok"] for c in checks)

GATE_DIR.mkdir(parents=True, exist_ok=True)
dest = GATE_DIR / f".gate_switch_cost_{NAME}.json"
dest.write_text(json.dumps(report, indent=2, default=str))
print(f"-> {dest}")
failed = [c["check"] for c in checks if not c["ok"]]
if not checks:
    print(f"gate {NAME}: NO CHECKS RAN — treating as a failure.")
    sys.exit(1)
if failed:
    print(f"gate {NAME}: FAILED {len(failed)} of {len(checks)} check(s): {failed}")
    sys.exit(1)
print(f"gate {NAME}: PASSED all {len(checks)} check(s).")
PYGATE
        ;;

esac

echo "Switch-cost sleep step ${SLEEP_STEP}${SLEEP_GATE:+ (${SLEEP_GATE})} complete: $(date)"
