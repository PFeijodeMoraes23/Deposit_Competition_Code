"""exhibits_local.py -- runs the exhibit generators that need only the market panel and local
side data, in order, one process at a time.

None of these steps reads a cluster output, so they can run as soon as market_panel.parquet is
current. The ordered steps (each one a call of this interpreter on a repo script):

  1  make_dep_anova.py
  2  make_pix_exposure.py
  3  make_state_vector_table.py
  4  make_dshare_decomp.py
  5  make_fee_ratio.py
  6  make_slide_figures.py --only panels,timedep,savings,table --refresh-shares
  7  make_slide_figures.py --only band,pricevol,reprice,dpremium,entrygap,relprice,anova,motivtables --png
  8  make_slide_figures.py --only depanova,pixexposure,paper,slidesize,slidesize2,slidesize3,slidesize6,ref --png
     the size and reference-style variants the papers and the deck input (the _ref motivation
     figures, the market-structure halves, the HHI chart) and the slides of steps 1-2's sidecars;
     none reads an estimation output. make_slide_figures draws these last, each with its own size.

make_entry_carry.py is not here: its events file (DIAG_PHI_SEPARATION/d6_entry_events.csv) is
written by the cluster's entry-dynamics run, so exhibits_post_cluster.py runs it. Run this runner
before exhibits_post_cluster.py: the latter's followups slides read step 5's sidecar.

A step whose script is absent is reported as MISSING. --dry-run and --status list it; --run
refuses to start while a selected step is MISSING, unless --skip names it.

Staleness. The panel is PROCESSED/market_panel.parquet (utils.paths.PROCESSED). After a step
runs, its return code, time and the md5 of the parquet it ran on are written to
Rout/exhibits_local_runs.json (utils.paths.rout_dir()). --status compares each recorded md5
with the parquet's current one: CURRENT (same), STALE (different), NEVER (no record).

Usage:
  python exhibits_local.py                 # --dry-run: the ordered commands
  python exhibits_local.py --status
  python exhibits_local.py --run [--from N] [--only N[,M]] [--skip N[,M]]

--run sets MPLBACKEND=Agg, streams each step's output to the console and to
Rout/exhibits_local_<timestamp>.log, and stops at the first nonzero return code.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys

from utils import paths

HERE = pathlib.Path(__file__).resolve().parent
PANEL = paths.PROCESSED / "market_panel.parquet"
SIDECAR_NAME = "exhibits_local_runs.json"
LOG_PREFIX = "exhibits_local"

# (script, arguments)
STEPS = [
    ("make_dep_anova.py", []),
    ("make_pix_exposure.py", []),
    ("make_state_vector_table.py", []),
    ("make_dshare_decomp.py", []),
    ("make_fee_ratio.py", []),
    ("make_slide_figures.py", ["--only", "panels,timedep,savings,table", "--refresh-shares"]),
    ("make_slide_figures.py", ["--only", "band,pricevol,reprice,dpremium,entrygap,relprice,anova,motivtables",
                              "--png"]),
    ("make_slide_figures.py", ["--only", "depanova,pixexposure,paper,slidesize,slidesize2,slidesize3,"
                                         "slidesize6,ref", "--png"]),
]


def md5_of(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def command(i: int) -> list:
    script, args = STEPS[i - 1]
    return [sys.executable, str(HERE / script), *args]


def label(i: int) -> str:
    script, args = STEPS[i - 1]
    return " ".join([script, *args])


def is_missing(i: int) -> bool:
    return not (HERE / STEPS[i - 1][0]).is_file()


def parse_list(s: str | None) -> set:
    return {int(x) for x in s.split(",") if x.strip()} if s else set()


def select(args) -> list:
    n = len(STEPS)
    chosen = list(range(1, n + 1))
    if args.only:
        chosen = [i for i in chosen if i in parse_list(args.only)]
    if args.from_step:
        chosen = [i for i in chosen if i >= args.from_step]
    skip = parse_list(args.skip)
    return [i for i in chosen if i not in skip]


def load_sidecar() -> dict:
    p = paths.rout_dir() / SIDECAR_NAME
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_sidecar(d: dict) -> None:
    (paths.rout_dir() / SIDECAR_NAME).write_text(json.dumps(d, indent=2), encoding="utf-8")


def dry_run(chosen: list) -> None:
    print(f"panel: {PANEL}  ({'present' if PANEL.is_file() else 'ABSENT'})")
    for i in range(1, len(STEPS) + 1):
        tag = "MISSING" if is_missing(i) else ("skipped" if i not in chosen else "ok")
        print(f"  [{i}] {tag:<8} {label(i)}")


def status() -> None:
    cur = md5_of(PANEL)
    rec = load_sidecar()
    print(f"panel: {PANEL}\n  current md5: {cur}\n  sidecar: {paths.rout_dir() / SIDECAR_NAME}")
    for i in range(1, len(STEPS) + 1):
        r = rec.get(label(i))
        if is_missing(i):
            print(f"  [{i}] MISSING script   {label(i)}")
        elif not r:
            print(f"  [{i}] NEVER            {label(i)}")
        else:
            state = "CURRENT" if r.get("panel_md5") == cur and cur else "STALE"
            print(f"  [{i}] {state:<7} last run {r['time']}  rc={r['rc']}  {label(i)}")


def run(chosen: list) -> int:
    missing = [i for i in chosen if is_missing(i)]
    if missing:
        print("refusing to run: MISSING step script(s): "
              + ", ".join(f"[{i}] {STEPS[i - 1][0]}" for i in missing)
              + "\n  create them or pass --skip " + ",".join(str(i) for i in missing))
        return 2
    if not PANEL.is_file():
        print(f"refusing to run: {PANEL} is absent")
        return 2
    md5 = md5_of(PANEL)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = paths.rout_dir() / f"{LOG_PREFIX}_{stamp}.log"
    rec = load_sidecar()
    print(f"log: {log_path}\npanel md5: {md5}")
    with open(log_path, "w", encoding="utf-8") as log:
        for i in chosen:
            head = f"[{i}] {label(i)}  ({datetime.datetime.now():%H:%M:%S})"
            print(head, flush=True)
            log.write(head + "\n")
            proc = subprocess.Popen(command(i), cwd=HERE, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace")
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
            rc = proc.wait()
            log.write(f"-- rc={rc}\n")
            rec[label(i)] = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
                             "rc": rc, "panel_md5": md5}
            save_sidecar(rec)
            if rc != 0:
                print(f"STOPPED at step [{i}] {label(i)}: return code {rc}")
                return rc
    print("all selected steps finished with return code 0")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the ordered commands (default)")
    mode.add_argument("--run", action="store_true", help="execute the selected steps in order")
    mode.add_argument("--status", action="store_true",
                      help="last run, return code and CURRENT/STALE/NEVER per step")
    ap.add_argument("--from", dest="from_step", type=int, default=None, help="start at step N")
    ap.add_argument("--only", default=None, help="comma-separated step numbers to run")
    ap.add_argument("--skip", default=None, help="comma-separated step numbers to skip")
    args = ap.parse_args()
    chosen = select(args)
    if args.status:
        status()
        return 0
    if args.run:
        return run(chosen)
    dry_run(chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
