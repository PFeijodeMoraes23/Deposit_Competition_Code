"""run_phi_diagnostics.py -- one-command orchestrator for the phi-separation battery.

Author: Pedro Feijo de Moraes

Runs the full D0-D8 diagnostic battery (see identification_notes.md in
Drafts/Deposit Competition) in the cheapest-most-informative-first order:

    D0  identity/tautology + censoring     diag_phi_augmented_tests.py  --arm identity
    D5  deposit-type contrast              diag_phi_interaction_tests.py --arm types
    D2b lagged-Dep_Act augmentation        diag_phi_augmented_tests.py  --arm lagdepact
    D3  attractiveness placebo             diag_phi_interaction_tests.py --arm attractiveness
    D1s MC recovery smoke (harness check)  diag_phi_mc_recovery.py      --mode grid --smoke
    D1  MC recovery full + null stats      diag_phi_mc_recovery.py      --mode grid --stats-under-null
    D8  ACF overidentification             diag_phi_mc_recovery.py      --mode acf
    D6  Pix event study                    diag_phi_augmented_tests.py  --arm pix
    D2a fitted-share augmentation          diag_phi_augmented_tests.py  --arm fittedshare
    D4  spread level (qualitative)         diag_phi_augmented_tests.py  --arm spreadlevel

Steps run in isolated SUBPROCESSES (the known 0xC0000005 segfaults are threads inside
one process; separate processes are safe). Default is sequential; `--jobs N` runs up to
N steps concurrently with longest-first scheduling. Each step needs ~3-4 GB (its own
panel copy) and D1 spawns 8 workers of its own, so on a 32 GB / 12-core box:
  --jobs 3 is the sweet spot (~35-50 min wall vs ~2-2.5 h sequential);
  do NOT run alongside the Julia sieve (E7/E8) or other heavy jobs.
Each step's full output goes to
<PROCESSED>/ESTIMATION_OUTPUT/DIAG_PHI_SEPARATION/logs/<step>.log; the console gets a
timestamped line per step plus its VERDICT block. A failing step does not stop the
battery (status reported at the end); exit code is 0 only if all ran.

PREREQUISITE when attached after a re-estimation: D0/D2b/D2a and the D1/D8 calibration
read demand_2_spec_12.parquet, so rebuild the E2 demand prep (estimation_2_demand_1_prep,
spec 12) after re-running the sleep step and BEFORE this battery.

Usage:
  python run_phi_diagnostics.py              # full battery, sequential (~2-2.5h)
  python run_phi_diagnostics.py --jobs 3     # concurrent lanes (~35-50 min)
  python run_phi_diagnostics.py --quick      # skip the full MC grid (keeps the smoke)
  python run_phi_diagnostics.py --only D5 D2b
  python run_phi_diagnostics.py --skip D6 D4
  python run_phi_diagnostics.py --list       # print the plan and exit
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from utils import paths as _paths

HERE = Path(__file__).resolve().parent
OUT_DIR = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_PHI_SEPARATION"
LOG_DIR = OUT_DIR / "logs"

# (key, description, script, args, in_quick, est_minutes)
# est_minutes drive the longest-first schedule under --jobs; from the 2026-07-29 runs.
STEPS = [
    ("D0",  "identity/tautology + censoring",  "diag_phi_augmented_tests.py",   ["--arm", "identity"],       True,   4),
    ("D5",  "deposit-type contrast",           "diag_phi_interaction_tests.py", ["--arm", "types"],          True,  12),
    ("D2b", "lagged-Dep_Act augmentation",     "diag_phi_augmented_tests.py",   ["--arm", "lagdepact"],      True,  15),
    ("D3",  "attractiveness placebo",          "diag_phi_interaction_tests.py", ["--arm", "attractiveness"], True,  15),
    ("D1s", "MC recovery smoke (harness)",     "diag_phi_mc_recovery.py",       ["--mode", "grid", "--smoke"], True, 5),
    ("D1",  "MC recovery full + null stats",   "diag_phi_mc_recovery.py",       ["--mode", "grid", "--stats-under-null"], False, 20),
    ("D8",  "ACF overidentification",          "diag_phi_mc_recovery.py",       ["--mode", "acf"],           True,  10),
    ("D6",  "Pix event study",                 "diag_phi_augmented_tests.py",   ["--arm", "pix"],            True,  22),
    ("D2a", "fitted-share augmentation",       "diag_phi_augmented_tests.py",   ["--arm", "fittedshare"],    True,  28),
    ("D4",  "spread level (qualitative)",      "diag_phi_augmented_tests.py",   ["--arm", "spreadlevel"],    True,  10),
    ("D9",  "BLP elasticity consistency",      "diag_phi_augmented_tests.py",   ["--arm", "blpelast"],       True,  10),
]

_print_lock = threading.Lock()


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def run_step(step):
    key, desc, script, args, _, _ = step
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{key}.log"
    cmd = [sys.executable, "-u", str(HERE / script)] + args
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONFAULTHANDLER="1",
               PYTHONIOENCODING="utf-8")
    with _print_lock:
        print(f"[{_ts()}] {key:<4s} START  {desc}  ({script} {' '.join(args)})")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                           env=env, cwd=str(HERE))
    dt = time.time() - t0
    status = "OK" if p.returncode == 0 else f"FAIL(rc={p.returncode})"
    # surface the step's VERDICT block on the console
    verdict = []
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        v0 = next((i for i, ln in enumerate(lines) if "VERDICT" in ln), None)
        if v0 is not None:
            verdict = lines[v0:v0 + 6]
    except OSError:
        pass
    with _print_lock:
        print(f"[{_ts()}] {key:<4s} {status}  {dt/60:.1f} min  -> logs/{log.name}")
        for ln in verdict:
            print(f"        {ln.strip()}")
    return key, desc, p.returncode, dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="skip the full MC grid (D1); keep the smoke harness check")
    ap.add_argument("--jobs", type=int, default=1, metavar="N",
                    help="run up to N steps concurrently (isolated subprocesses, "
                         "longest-first schedule). 3 is the 32GB sweet spot; keep 1 if "
                         "anything heavy (Julia sieve, BLP) is running.")
    ap.add_argument("--only", nargs="+", metavar="STEP",
                    help="run only these step keys (e.g. --only D5 D2b)")
    ap.add_argument("--skip", nargs="+", metavar="STEP", default=[],
                    help="skip these step keys")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    a = ap.parse_args()

    keys = {k.upper() for k in (a.only or [])}
    skip = {k.upper() for k in a.skip}
    plan = [s for s in STEPS
            if (not a.quick or s[4])
            and (not keys or s[0].upper() in keys)
            and s[0].upper() not in skip]

    if a.list or not plan:
        print("battery plan:")
        for k, d, sc, ar, _, est in plan or STEPS:
            print(f"  {k:<4s} {d:<34s} ~{est:>2d} min  {sc} {' '.join(ar)}")
        return 0

    jobs = max(1, a.jobs)
    print(f"=== phi-separation battery: {len(plan)} step(s), "
          f"{'sequential' if jobs == 1 else f'{jobs} concurrent lanes'} ===")
    print(f"    outputs: {OUT_DIR}")
    t0 = time.time()
    if jobs == 1:
        results = [run_step(s) for s in plan]
    else:
        # longest-first keeps the tail short; each step is its own subprocess, so
        # concurrency here is process-level (safe on this stack) not thread-level.
        ordered = sorted(plan, key=lambda s: -s[5])
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(run_step, ordered))

    print(f"\n=== summary ({(time.time() - t0)/60:.1f} min total) ===")
    for k, d, rc, dt in results:
        print(f"  {k:<4s} {'OK  ' if rc == 0 else 'FAIL'}  {dt/60:6.1f} min  {d}")
    nfail = sum(1 for _, _, rc, _ in results if rc != 0)
    if nfail:
        print(f"\n{nfail} step(s) FAILED -- see logs in {LOG_DIR}")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
