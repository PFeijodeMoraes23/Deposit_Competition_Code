#!/usr/bin/env python
"""
run_diag_matrix.py
==================
Run a (routine x diagnostic) matrix with a HARD WALL-CLOCK DEADLINE, one unit at a time.

Why this exists. The diagnostic arms take minutes to hours and the machine is shared with cluster
prep and long estimations. A plain `for` loop either finishes late or gets killed mid-write,
leaving a half-formed CSV that looks complete. This runner:

  * checks the deadline BEFORE starting each unit and refuses to start one it cannot finish
    (using that unit's own measured runtime, not a guess) — so it never leaves a truncated output;
  * writes a manifest after every unit, so an interrupted run is resumable with --resume;
  * treats SIGINT as "stop cleanly after the current unit", not "die now";
  * records, for every unit, whether it RAN, was SKIPPED (deadline), or FAILED.

Deadline forms:
    --until 16:00            today at 16:00 local (tomorrow if already past)
    --until 2026-08-07T09:30 explicit
    --for 90m                relative: 90 minutes from now
Nothing is started after the deadline; a unit already running is allowed to finish (killing it
mid-write is what this exists to prevent). --grace lets you cap that too.

Usage:
    python run_diag_matrix.py --units d0:1,2,3,4 --until 16:00
    python run_diag_matrix.py --units d0:3,4 d2:3,4 --for 2h --dry-run
    python run_diag_matrix.py --resume            # continue the last run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
STATE = HERE / ".diag_matrix_state.json"

# unit key -> (script, args template, measured minutes per routine).
# The minutes are MEASURED, not guessed (2026-08-06 battery log); they drive the "can this finish
# before the deadline?" decision, so a wrong number here causes either a needless skip or a
# late finish. Update them when a run reports a materially different duration.
UNITS = {
    "d0":   ("step_phi_augmented_tests.py",   ["--arm", "identity"],     1.0),
    "d2":  ("step_phi_augmented_tests.py",   ["--arm", "lagdepact"],    2.5),
    "d5":   ("step_phi_augmented_tests.py",   ["--arm", "blpelast"],     2.0),
    "d4a":   ("step_phi_interaction_tests.py", ["--arm", "attractiveness"], 6.0),
    "d4b":   ("step_phi_interaction_tests.py", ["--arm", "types"],        1.5),
    "d1s":  ("step_phi_mc_recovery.py",       ["--mode", "grid", "--smoke"], 0.5),
    "d1":   ("step_phi_mc_recovery.py",       ["--mode", "grid", "--stats-under-null"], 3.0),
    "d4c":   ("step_phi_mc_recovery.py",       ["--mode", "acf"],         1.0),
}

# Units whose script does NOT accept --estim. step_phi_interaction_tests.py takes only --arm,
# and step_phi_mc_recovery.py exposes --mode/--phi-prod but no --estim (its calibration parquet
# and kernel are E2's by construction) -- appending --estim aborts either one in argparse before
# a single row is computed. Listed here so the runner omits the flag and reports the E2-only
# scope, instead of the whole unit dying.
NO_ESTIM = {"d4a", "d4b", "d1s", "d1", "d4c"}

# Arms whose ESTIMATOR is E2's regardless of --estim. Running them for another routine gives a
# HYBRID (that routine's phi inside E2's kernel), which is a legitimate reduced-form check but is
# not "the test for that routine". The runner labels these loudly rather than pretending.
E2_KERNEL = {"d2", "d4a", "d1", "d1s", "d4c", "d5"}


def parse_deadline(until: str | None, for_: str | None) -> dt.datetime | None:
    if for_:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([mh])", for_.strip().lower())
        if not m:
            raise SystemExit(f"--for wants e.g. 90m or 2h, got {for_!r}")
        mins = float(m.group(1)) * (60 if m.group(2) == "h" else 1)
        return dt.datetime.now() + dt.timedelta(minutes=mins)
    if not until:
        return None
    s = until.strip()
    if re.fullmatch(r"\d{1,2}:\d{2}", s):                     # HH:MM today (or tomorrow)
        h, mi = map(int, s.split(":"))
        now = dt.datetime.now()
        d = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        return d + dt.timedelta(days=1) if d <= now else d
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        raise SystemExit(f"--until wants HH:MM or an ISO timestamp, got {until!r}")


def parse_units(specs: list[str]) -> list[tuple[str, int]]:
    out = []
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"--units wants unit:routines, e.g. d0:3,4 — got {spec!r}")
        key, routines = spec.split(":", 1)
        key = key.strip().lower()
        if key not in UNITS:
            raise SystemExit(f"unknown unit {key!r}; known: {', '.join(UNITS)}")
        for r in routines.split(","):
            r = r.strip()
            if r:
                out.append((key, int(r)))
    return out


class Stopper:
    """SIGINT means 'finish the current unit, then stop' — not 'die mid-write'."""
    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, *_):
        if self.stop:
            print("\n[matrix] second interrupt — exiting now")
            sys.exit(130)
        self.stop = True
        print("\n[matrix] interrupt received — will stop after the current unit "
              "(press again to abort immediately)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", nargs="+", metavar="UNIT:ROUTINES",
                    help="e.g. d0:1,2,3,4 d2:3,4")
    ap.add_argument("--until", help="stop starting new units after this local time (HH:MM or ISO)")
    ap.add_argument("--for", dest="for_", help="relative deadline, e.g. 90m or 2h")
    ap.add_argument("--grace", type=float, default=0,
                    help="minutes past the deadline a RUNNING unit may use (0 = unlimited; "
                         "it is never killed mid-write, this only warns)")
    ap.add_argument("--resume", action="store_true", help="continue the previous run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="show known units and exit")
    a = ap.parse_args()

    if a.list:
        print(f"{'unit':6s} {'script':34s} {'args':44s} {'min/routine':>11s}  kernel")
        for k, (s, ar, m) in UNITS.items():
            print(f"{k:6s} {s:34s} {' '.join(ar):44s} {m:11.1f}  "
                  f"{'E2 (hybrid if --estim != 2)' if k in E2_KERNEL else 'routine-specific'}")
        return

    if a.resume:
        if not STATE.exists():
            raise SystemExit("no previous run to resume (.diag_matrix_state.json missing)")
        st = json.loads(STATE.read_text(encoding="utf-8"))
        units = [(u["unit"], u["estim"]) for u in st["units"] if u["status"] == "pending"]
        deadline = parse_deadline(a.until, a.for_)
        print(f"[matrix] resuming: {len(units)} unit(s) still pending")
    else:
        if not a.units:
            raise SystemExit("--units is required (or --resume / --list)")
        units = parse_units(a.units)
        deadline = parse_deadline(a.until, a.for_)
        st = {"started": dt.datetime.now().isoformat(timespec="seconds"),
              "deadline": deadline.isoformat(timespec="seconds") if deadline else None,
              "units": [{"unit": u, "estim": e, "status": "pending"} for u, e in units]}
        STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")

    est_total = sum(UNITS[u][2] for u, _ in units)
    print(f"[matrix] {len(units)} unit(s), ~{est_total:.0f} min estimated")
    if deadline:
        left = (deadline - dt.datetime.now()).total_seconds() / 60
        print(f"[matrix] deadline {deadline:%Y-%m-%d %H:%M} ({left:.0f} min from now)")
        if est_total > left:
            print(f"[matrix] NOTE: estimate exceeds the window by {est_total-left:.0f} min — "
                  f"later units will be skipped and left pending for --resume")
    else:
        print("[matrix] no deadline set")

    hybrid = sorted({u for u, e in units if u in E2_KERNEL and e != 2})
    if hybrid:
        print(f"[matrix] !! {', '.join(hybrid)} use E2's estimator: for --estim != 2 the result is "
              f"a HYBRID (that routine's phi in E2's kernel), not that routine's own test.")

    stopper = Stopper()
    ran = skipped = failed = 0

    for idx, (unit, estim) in enumerate(units, 1):
        script, base_args, mins = UNITS[unit]
        label = f"{unit}/E{estim}"

        if stopper.stop:
            print(f"[matrix] {label}: SKIP (stop requested)"); skipped += 1; continue

        if deadline:
            left = (deadline - dt.datetime.now()).total_seconds() / 60
            if left <= 0:
                print(f"[matrix] {label}: SKIP (past deadline)"); skipped += 1; continue
            if mins > left:
                print(f"[matrix] {label}: SKIP (needs ~{mins:.1f} min, {left:.1f} min left) "
                      f"— left pending for --resume")
                skipped += 1; continue

        cmd = [sys.executable, "-u", script] + base_args
        if unit in NO_ESTIM:
            if estim != 2:
                print(f"[matrix] {label}: SKIP ({script} has no --estim; it is E2-only)")
                skipped += 1
                continue
        else:
            cmd += ["--estim", str(estim)]
        print(f"\n[matrix] ({idx}/{len(units)}) {label}  {dt.datetime.now():%H:%M:%S}  "
              f"~{mins:.1f} min\n         {' '.join(cmd[2:])}")
        if a.dry_run:
            ran += 1; continue

        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(HERE)).returncode
        el = (time.time() - t0) / 60
        status = "ran" if rc == 0 else "failed"
        if rc == 0:
            ran += 1
            print(f"[matrix] {label}: OK in {el:.1f} min "
                  f"(estimate was {mins:.1f})")
        else:
            failed += 1
            print(f"[matrix] {label}: FAILED rc={rc} after {el:.1f} min — continuing")

        for u in st["units"]:
            if u["unit"] == unit and u["estim"] == estim and u["status"] == "pending":
                u["status"] = status; u["minutes"] = round(el, 2); break
        STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")

        if deadline and a.grace and dt.datetime.now() > deadline + dt.timedelta(minutes=a.grace):
            print(f"[matrix] past deadline + {a.grace:.0f} min grace — stopping")
            break

    print(f"\n[matrix] done {dt.datetime.now():%H:%M:%S}: {ran} ran, {skipped} skipped, "
          f"{failed} failed")
    if skipped:
        print(f"[matrix] resume the skipped units with:  python {Path(__file__).name} --resume")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
