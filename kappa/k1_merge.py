"""
kappa/k1_merge.py
=================
Merge the Bouchet K1 array outputs (kappa/k1_job.sh: data/output/k1/task_<id>/k1_estimates.csv and
k1_numbers.json) into the folders the T9 reader reads:
  Rout/k1            every task except 5 (the headline invocation and its robustness rows)
  Rout/k1_lambda_one task 5 (--lambda-one)
Each output row keeps its columns and gains `task`; k1_numbers.json keeps the headline task's
(task 0, or task 5 for k1_lambda_one) top-level record and adds "tasks": {id: that task's record}.
Checks before writing: every task folder has both files, every record's guard says final_inputs and
names the same panel, and every row's spec and options match the task table of k1_job.sh.
Usage: python -I kappa/k1_merge.py --in <downloaded .../k1> [--rout <Rout>] [--dry-run]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import pandas as pd

from utils import paths

TASKS = {0: ("before_after", "p99", True, False), 1: ("dose", "p99", True, False),
         2: ("before_after+carried_kept", "p99", True, False), 3: ("before_after+zero_premium", "p99", True, False),
         4: ("before_after+no_negative_adv", "p99", True, False), 5: ("before_after", "p99", True, True),
         6: ("before_after", "none", True, False), 7: ("before_after", "p99", False, False)}
EXIT_BY_CELL = "theta_w free by market-quarter (whole movable stock)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="the downloaded k1 folder holding task_<id>/")
    ap.add_argument("--rout", default=str(paths.rout_dir()))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    src, rout = Path(args.src), Path(args.rout)
    found = {}
    for d in sorted(src.glob("task_*")):
        tid = int(d.name.split("_")[1])
        est, num = d / "k1_estimates.csv", d / "k1_numbers.json"
        if not (est.exists() and num.exists()):
            raise SystemExit(f"{d}: k1_estimates.csv or k1_numbers.json missing (task not finished?)")
        found[tid] = (pd.read_csv(est), json.loads(num.read_text(encoding="utf-8")))
    if not found:
        raise SystemExit(f"no task_<id> folders under {src}")
    panels = set()
    for tid, (E, N) in found.items():
        if tid not in TASKS:
            raise SystemExit(f"task {tid} is not in the k1_job.sh table")
        spec, trim, by_cell, lam1 = TASKS[tid]
        g = N.get("guard", {})
        if not g.get("final_inputs"):
            raise SystemExit(f"task {tid}: guard {g} is not final")
        panels.add(str(N.get("inputs", {}).get("panel_md5")))
        bad = E[(E["spec"] != spec) | (E["trim"] != trim) | ((E["exit"] == EXIT_BY_CELL) != by_cell)]
        if len(bad) or bool(N.get("args", {}).get("lambda_one")) != lam1:
            raise SystemExit(f"task {tid}: rows or options do not match the task table ({spec}, {trim}, {by_cell}, {lam1})")
        print(f"task {tid}: {spec} trim {trim} exit_by_cell {by_cell} lambda_one {lam1}: {len(E)} rows "
              f"({', '.join(E['solution'].astype(str))})")
    if len(panels) != 1:
        raise SystemExit(f"the tasks name different panels: {panels}")
    groups = {"k1": [t for t in sorted(found) if t != 5], "k1_lambda_one": [t for t in sorted(found) if t == 5]}
    for folder, ids in groups.items():
        if not ids:
            continue
        rows = pd.concat([found[t][0].assign(task=t) for t in ids], ignore_index=True)
        head = found[0 if folder == "k1" and 0 in ids else ids[0]][1]
        numbers = {**head, "tasks": {str(t): found[t][1] for t in ids}}
        out = rout / folder
        print(f"{out}: {len(rows)} rows from tasks {ids}")
        if args.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out / "k1_estimates.csv", index=False)
        (out / "k1_numbers.json").write_text(json.dumps(numbers, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
