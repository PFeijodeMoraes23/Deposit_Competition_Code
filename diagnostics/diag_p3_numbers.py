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
                f, f_net: F of log sd on log sd~ (MCA and quarter effects, MCA clusters), and the
                  same net of lagged HHI and number of banks
                placebo (rho~ shuffled across conglomerates within the quarter), placebo_shuffles
                  draws: f_placebo, f_net_placebo the median F of the draws, placebo_share_ge,
                  placebo_share_ge_net the share of draws with F at or above f, f_net;
                  placebo_seed. Entries with placebo_shuffles 1 (the p3 blocks written before the
                  99 draws) hold one draw in f_placebo, seed 20261103, and no net placebo
                f_state, f_net_state: the same F clustered by state (n_state_clusters);
                  f_placebo_state, f_net_placebo_state: the median state-cluster F of the same
                  draws; placebo_share_ge_state, placebo_share_ge_net_state: the share of draws
                  with state-cluster F at or above f_state, f_net_state. Only in entries whose p3
                  block has them
                f_level, f_log: S on S~ in levels and logs; n_first_stage
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
       "a year; f = F of log sd on log sd~, f_net = the same net of lagged HHI and number of banks "
       "(MCA and quarter effects, MCA clusters). Placebo, rho~ shuffled across conglomerates within "
       "the quarter, placebo_shuffles draws: f_placebo and f_net_placebo = median F of the draws, "
       "placebo_share_ge and placebo_share_ge_net = share of draws with F at or above f and f_net. "
       "Where placebo_shuffles is 1, f_placebo is a single draw (seed 20261103) and there is no net "
       "placebo. State clusters (the MCA's state, IBGE crosswalk), where present: f_state, "
       "f_net_state; f_placebo_state and f_net_placebo_state = median state-cluster F of the same "
       "draws; placebo_share_ge_state and placebo_share_ge_net_state = share of draws with "
       "state-cluster F at or above f_state and f_net_state.")


def entry(p3: dict) -> dict:
    d, pl = p3["S_on_S_tilde"], p3.get("placebo")
    e = dict(panel_md5=p3["_meta"]["panel_md5"], written=p3["_meta"]["written"],
             median_S=p3["S_quantiles"]["0.5"], f=d["sd_only_log"]["f"],
             f_placebo=pl["f_median"] if pl else d["sd_only_log_placebo"]["f"],
             f_net=d["sd_only_log_ctrl_structure"]["f"])
    if pl:
        e.update(f_net_placebo=pl["f_net_median"], placebo_share_ge=pl["share_ge"],
                 placebo_share_ge_net=pl["share_ge_net"], placebo_shuffles=pl["n_shuffles"],
                 placebo_seed=pl["seed"])
    else:
        e.update(placebo_shuffles=1)
    if "f_state" in d["sd_only_log"]:   # the state-cluster counterparts, where the p3 block has them
        e.update(f_state=d["sd_only_log"]["f_state"], f_net_state=d["sd_only_log_ctrl_structure"]["f_state"],
                 n_state_clusters=d["sd_only_log"]["n_state_clusters"])
        if pl and "f_state_median" in pl:
            e.update(f_placebo_state=pl["f_state_median"], f_net_placebo_state=pl["f_net_state_median"],
                     placebo_share_ge_state=pl["share_ge_state"],
                     placebo_share_ge_net_state=pl["share_ge_net_state"])
    e.update(f_level=d["level"]["f"], f_log=d["log"]["f"], n_first_stage=d["sd_only_log"]["n"],
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
    for v in panels.values():   # entries filed before the draws hold a single shuffle
        v.setdefault("placebo_shuffles", 1)
    for src in [*args.add, SRC]:
        e = entry(json.loads(src.read_text(encoding="utf-8"))["p3"])
        panels[e["panel_md5"][:8]] = e
        pl = (f"placebo median {e['f_placebo']:.2f} (share at or above {e['placebo_share_ge']:.3f}), "
              f"net {e['f_net']:.2f}, placebo median {e['f_net_placebo']:.2f} (share at or above "
              f"{e['placebo_share_ge_net']:.3f}), {e['placebo_shuffles']} draws"
              if e["placebo_shuffles"] > 1 else
              f"placebo {e['f_placebo']:.2f} (one draw), net {e['f_net']:.2f}")
        st = (f"; state clusters: F {e['f_state']:.2f}, placebo median {e['f_placebo_state']:.2f} "
              f"(share at or above {e['placebo_share_ge_state']:.3f}), net {e['f_net_state']:.2f}"
              if "f_placebo_state" in e else "")
        print(f"{e['panel_md5'][:8]}: median S {e['median_S']:.1f}, F {e['f']:.2f}, {pl}{st}  ({src})")
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
