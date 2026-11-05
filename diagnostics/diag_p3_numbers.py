"""
diag_p3_numbers.py
==================
Task P7 (pix_handoff_2026-11-04): the P3 numbers the paper's keys quote, by panel, in a file that
make_paper_numbers.py (collect_pix_stakes) can read instead of parsing p3_note.md.

Reads the p3 block of Rout/p_shifters.json (written by diag_p_shifters.py --only p3) and files its
numbers under the panel's md5 (first 8 characters, as p3_stakes_mca_quarter.csv's panel_md5) in
Rout/p3_numbers.json, keeping the entries of earlier panels. --add <p_shifters.json> files the p3
block of an earlier copy too (how the d01b8bae and 45dbfc58 entries were seeded).

  current       the panel of the p3 files now in Rout (checked against p3_stakes_mca_quarter.csv)
  panels.<md5>  median_S: median of S_mt over market-quarters, R$ a year (the CSV's median)
                f, f_placebo, f_net: F of log sd on log sd~ (MCA and quarter effects, MCA
                  clusters), with rho~ shuffled within the quarter, and net of lagged HHI and
                  number of banks; f_level, f_log: S on S~ in levels and logs; n_first_stage
                n_mq, n_mq_S, sd_spread_pp_median, cost_fit_joint_f
                median_S_measured, f_measured, f_net_measured, carried_weight_share_mean: the
                  measured-rate variant, where the panel has rate_a4_carried
                panel_md5 (full), written (the P3 run)

Usage:  python -I diagnostics/diag_p3_numbers.py [--add <earlier p_shifters.json> ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import pandas as pd

from utils import paths

OUT = paths.rout_dir()
SRC = OUT / "p_shifters.json"
STAKES = OUT / "p3_stakes_mca_quarter.csv"
DST = OUT / "p3_numbers.json"
DOC = ("P3 first-stage numbers by panel (8-character md5, as p3_stakes_mca_quarter.csv's panel_md5); "
       "written by diagnostics/diag_p3_numbers.py from the p3 block of p_shifters.json. median_S in R$ "
       "a year; f = F of log sd on log sd~, f_placebo = rho~ shuffled, f_net = net of lagged HHI and "
       "number of banks (MCA and quarter effects, MCA clusters).")


def entry(p3: dict) -> dict:
    d = p3["S_on_S_tilde"]
    e = dict(panel_md5=p3["_meta"]["panel_md5"], written=p3["_meta"]["written"],
             median_S=p3["S_quantiles"]["0.5"], f=d["sd_only_log"]["f"],
             f_placebo=d["sd_only_log_placebo"]["f"], f_net=d["sd_only_log_ctrl_structure"]["f"],
             f_level=d["level"]["f"], f_log=d["log"]["f"], n_first_stage=d["sd_only_log"]["n"],
             n_mq=p3["n_mq"], n_mq_S=p3["n_mq_S"],
             sd_spread_pp_median=p3["sd_spread_pp_quantiles"]["0.5"],
             cost_fit_joint_f=p3["cost_fit"]["joint_f"])
    m = p3.get("measured")
    if isinstance(m, dict):
        e.update(median_S_measured=m["S_measured_quantiles"]["0.5"],
                 f_measured=d["measured_sd_only_log"]["f"],
                 f_net_measured=d["measured_sd_only_log_ctrl_structure"]["f"],
                 carried_weight_share_mean=m["carried_weight_share_mean"])
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="append", default=[], type=Path,
                    help="an earlier p_shifters.json whose p3 block is filed too")
    args = ap.parse_args()
    panels = json.loads(DST.read_text(encoding="utf-8"))["panels"] if DST.exists() else {}
    for src in [*args.add, SRC]:
        e = entry(json.loads(src.read_text(encoding="utf-8"))["p3"])
        panels[e["panel_md5"][:8]] = e
        print(f"{e['panel_md5'][:8]}: median S {e['median_S']:.1f}, F {e['f']:.2f}, placebo "
              f"{e['f_placebo']:.2f}, net {e['f_net']:.2f}  ({src})")
    cur = entry(json.loads(SRC.read_text(encoding="utf-8"))["p3"])
    s = pd.read_csv(STAKES, usecols=["S", "panel_md5"])
    if str(s["panel_md5"].iloc[0]) != cur["panel_md5"][:8] or abs(s["S"].median() - cur["median_S"]) > 1e-9:
        raise SystemExit(f"{STAKES.name} (panel {s['panel_md5'].iloc[0]}, median {s['S'].median()}) does "
                         f"not match the p3 block of {SRC.name} ({cur['panel_md5'][:8]}, {cur['median_S']})")
    ordered = dict(sorted(panels.items(), key=lambda kv: kv[1]["written"]))
    DST.write_text(json.dumps({"_doc": DOC, "current": cur["panel_md5"][:8], "panels": ordered},
                              indent=1), encoding="utf-8")
    print(f"wrote {DST.name}: current {cur['panel_md5'][:8]}; panels {', '.join(ordered)}")


if __name__ == "__main__":
    main()
