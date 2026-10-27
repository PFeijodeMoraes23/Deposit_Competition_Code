"""exhibits_post_cluster.py -- runs the exhibit generators that read the cluster's estimation
outputs, in order, one process at a time. No step touches the BBL solve, the policy function or
the counterfactuals (no bbl_*, make_bbl_*, make_cf*, bbl_polfunc.py).

Run it after the cluster download has been ingested (cluster_ingest.py). Ordered steps:

   1  sleep_ame_twostage.py --est 3 --loss robust --attach-from <DEMAND_PREP>/Rout/ame_twostage_est3_robust_if.pkl
      sleep_ame_twostage.py --est 4 --loss robust --attach-from <DEMAND_PREP>/Rout/ame_twostage_est4_robust_if.pkl
      (DEMAND_PREP = utils.paths.demand_prep_root(); the pickles must exist, else the step is
      reported MISSING)
   2  sleep_export_all.py --estimation all
   3  sleep_desc_clusters.py
   4  make_blp_demand_comparison_table.py
   5  make_blp_rc_table.py
   6  make_xi_diagnostics.py
   7  sleep_ident_entry_dynamics.py --figures-only
      (draws fig_entry_dynamics_a/b and the other D6 exhibits from the CSVs that the cluster's
      --compute-only run left in DIAG_PHI_SEPARATION; computes nothing)
   8  make_entry_carry.py
      (reads DIAG_PHI_SEPARATION/d6_entry_events.csv, written by the cluster's entry-dynamics run)
   9  make_phi_variance.py
  10  make_active_truncation.py
  11  make_paper_numbers.py
  12  make_slide_figures.py --only ame,coef,wedge,flow,xi,slidesize4,slidesize5,refame,followups

Step 11 keys (every make_slide_figures.py visual outside the panel-only list of
exhibits_local.py, kept when its function reads estimation outputs and nothing from BBL):
  ame         reads the routine's AME table sidecar (T4_SIDECAR, from sleep_export_all.py)
  coef        reads the BLP results of the routine (make_blp_rc_table.load_stage) and paper_numbers.json
  wedge       reads paper_numbers.json demand.elast.blp and demand.impact.blp (BLP elasticities)
  flow        TikZ of equation (9-B); its text quotes the sleep.meanphi paper number
  xi          reads Rout/xi_diagnostics.json written by step 6
  slidesize4  deck-size ame and wedge charts (both above)
  slidesize5  deck-size ame chart (above) plus the D-premium panel
  refame      reference-style ame chart (above)
  followups   the phi-hat variance, entry-carry and fee-ratio charts: it reads the sidecars of
              steps 8-9 and of exhibits_local.py's step 5 (fee ratio), so exhibits_local.py runs
              first
Left out: violations (reads the bbl.ud.* paper numbers: BBL) and roadmap (static TikZ text).
exhibits_local.py step 9 draws the panel-only keys depanova, pixexposure, paper, slidesize,
slidesize2, slidesize3, slidesize6 and ref.

--after-compile runs only make_paper_equations.py. It reads the paper after the user's LaTeX
compile, so it is never part of the main sequence; --only, --from and --skip do not apply to it.

A step whose script (or required input pickle) is absent is reported as MISSING. --dry-run and
--status list it; --run refuses to start while a selected step is MISSING, unless --skip names it.

Staleness marker. The cluster's results are ingested into BLP_RESULTS/cluster_raw
(cluster_ingest.py); the marker is the md5 over the blp_results_E{k}_spec_12_ext1.json files of
the link routines (utils.routines.LINK_ESTS), the ext1 fits that the demand tables, the xi-hat
diagnostics and the sleep exhibits all start from. After a step runs, its return code, time and
the marker md5 are written to Rout/exhibits_post_cluster_runs.json (utils.paths.rout_dir()).
--status compares each record with the current marker: CURRENT, STALE or NEVER.

Usage:
  python exhibits_post_cluster.py                 # --dry-run: the ordered commands
  python exhibits_post_cluster.py --status
  python exhibits_post_cluster.py --run [--from N] [--only N[,M]] [--skip N[,M]]
  python exhibits_post_cluster.py --after-compile [--run | --dry-run]

--run sets MPLBACKEND=Agg, streams each step's output to the console and to
Rout/exhibits_post_cluster_<timestamp>.log, and stops at the first nonzero return code.
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
from utils import routines as _routines

HERE = pathlib.Path(__file__).resolve().parent
SIDECAR_NAME = "exhibits_post_cluster_runs.json"
LOG_PREFIX = "exhibits_post_cluster"
CLUSTER_RAW = paths.blp_results_dir() / "cluster_raw"
MARKER_FILES = [CLUSTER_RAW / f"blp_results_E{k}_spec_12_ext1.json" for k in _routines.LINK_ESTS]
DP_ROUT = paths.demand_prep_root() / "Rout"

SLIDE_KEYS = "ame,coef,wedge,flow,xi,slidesize4,slidesize5,refame,followups"


def _attach(est: int) -> tuple:
    pkl = DP_ROUT / f"ame_twostage_est{est}_robust_if.pkl"
    return ("sleep_ame_twostage.py",
            ["--est", str(est), "--loss", "robust", "--attach-from", str(pkl)], [pkl])


# Each step: a list of (script, arguments, required input files).
STEPS = [
    [_attach(3), _attach(4)],
    [("sleep_export_all.py", ["--estimation", "all"], [])],
    [("sleep_desc_clusters.py", [], [])],
    [("make_blp_demand_comparison_table.py", [], [])],
    [("make_blp_rc_table.py", [], [])],
    [("make_xi_diagnostics.py", [], [])],
    [("sleep_ident_entry_dynamics.py", ["--figures-only"], [])],
    [("make_entry_carry.py", [], [])],
    [("make_phi_variance.py", [], [])],
    [("make_active_truncation.py", [], [])],
    [("make_paper_numbers.py", [], [])],
    [("make_slide_figures.py", ["--only", SLIDE_KEYS], [])],
]
AFTER_COMPILE = [("make_paper_equations.py", [], [])]
AFTER_KEY = "after-compile"


def marker_md5() -> str | None:
    """md5 over the marker files' names and contents; None when any is absent."""
    h = hashlib.md5()
    for p in MARKER_FILES:
        if not p.is_file():
            return None
        h.update(p.name.encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
    return h.hexdigest()


def cmds(step: list) -> list:
    return [[sys.executable, str(HERE / s), *a] for s, a, _ in step]


def label(step: list) -> str:
    return "  &&  ".join(" ".join([s, *a]) for s, a, _ in step)


def problems(step: list) -> list:
    out = [f"script {s}" for s, _, _ in step if not (HERE / s).is_file()]
    out += [f"input {r}" for _, _, req in step for r in req if not r.is_file()]
    return out


def parse_list(s: str | None) -> set:
    return {int(x) for x in s.split(",") if x.strip()} if s else set()


def select(args) -> list:
    chosen = list(range(1, len(STEPS) + 1))
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


def dry_run(chosen: list, after: bool) -> None:
    if after:
        tag = "MISSING" if problems(AFTER_COMPILE) else "ok"
        print(f"  [after-compile] {tag:<8} {label(AFTER_COMPILE)}")
        return
    print("marker files: " + ", ".join(str(p) for p in MARKER_FILES))
    for i, step in enumerate(STEPS, 1):
        bad = problems(step)
        tag = "MISSING" if bad else ("skipped" if i not in chosen else "ok")
        print(f"  [{i}] {tag:<8} {label(step)}")
        for b in bad:
            print(f"       missing {b}")


def status() -> None:
    cur = marker_md5()
    rec = load_sidecar()
    print("marker files: " + ", ".join(str(p) for p in MARKER_FILES)
          + f"\n  current md5: {cur}\n  sidecar: {paths.rout_dir() / SIDECAR_NAME}")
    rows = [(label(s), s, str(i)) for i, s in enumerate(STEPS, 1)]
    rows.append((AFTER_KEY, AFTER_COMPILE, "after-compile"))
    for key, step, name in rows:
        r = rec.get(key if key == AFTER_KEY else label(step))
        if problems(step):
            print(f"  [{name}] MISSING          {label(step)}")
        elif not r:
            print(f"  [{name}] NEVER            {label(step)}")
        else:
            if key == AFTER_KEY:
                state = "RAN"
            else:
                state = "CURRENT" if r.get("marker_md5") == cur and cur else "STALE"
            print(f"  [{name}] {state:<7} last run {r['time']}  rc={r['rc']}  {label(step)}")


def execute(todo: list, md5: str | None) -> int:
    """todo: [(sidecar key, step name, step)]. Runs sequentially, stops at the first nonzero
    return code."""
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = paths.rout_dir() / f"{LOG_PREFIX}_{stamp}.log"
    rec = load_sidecar()
    print(f"log: {log_path}\nmarker md5: {md5}")
    with open(log_path, "w", encoding="utf-8") as log:
        for key, name, step in todo:
            rc = 0
            for cmd in cmds(step):
                head = f"[{name}] {' '.join(cmd[1:])}  ({datetime.datetime.now():%H:%M:%S})"
                print(head, flush=True)
                log.write(head + "\n")
                proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                        errors="replace")
                for line in proc.stdout:
                    print(line, end="")
                    log.write(line)
                rc = proc.wait()
                log.write(f"-- rc={rc}\n")
                if rc != 0:
                    break
            rec[key] = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
                        "rc": rc, "marker_md5": md5}
            save_sidecar(rec)
            if rc != 0:
                print(f"STOPPED at step [{name}] {label(step)}: return code {rc}")
                return rc
    print("all selected steps finished with return code 0")
    return 0


def run(chosen: list, after: bool) -> int:
    if after:
        bad = problems(AFTER_COMPILE)
        if bad:
            print("refusing to run: " + "; ".join(bad))
            return 2
        return execute([(AFTER_KEY, "after-compile", AFTER_COMPILE)], marker_md5())
    blocked = [(i, problems(STEPS[i - 1])) for i in chosen if problems(STEPS[i - 1])]
    if blocked:
        print("refusing to run: MISSING " + "; ".join(f"[{i}] " + ", ".join(b) for i, b in blocked)
              + "\n  create them or pass --skip " + ",".join(str(i) for i, _ in blocked))
        return 2
    md5 = marker_md5()
    if md5 is None:
        print("refusing to run: marker files absent: "
              + ", ".join(str(p) for p in MARKER_FILES if not p.is_file()))
        return 2
    return execute([(label(STEPS[i - 1]), str(i), STEPS[i - 1]) for i in chosen], md5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the ordered commands (default)")
    mode.add_argument("--run", action="store_true", help="execute the selected steps in order")
    mode.add_argument("--status", action="store_true",
                      help="last run, return code and CURRENT/STALE/NEVER per step")
    ap.add_argument("--after-compile", action="store_true",
                    help="only make_paper_equations.py, to run after the LaTeX compile")
    ap.add_argument("--from", dest="from_step", type=int, default=None, help="start at step N")
    ap.add_argument("--only", default=None, help="comma-separated step numbers to run")
    ap.add_argument("--skip", default=None, help="comma-separated step numbers to skip")
    args = ap.parse_args()
    chosen = select(args)
    if args.status:
        status()
        return 0
    if args.run:
        return run(chosen, args.after_compile)
    dry_run(chosen, args.after_compile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
