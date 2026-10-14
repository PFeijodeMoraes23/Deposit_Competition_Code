"""switch_cost_prep_checks.py -- identity checks of the switch-cost copies (read only).

  fit       the pooled re-estimate (switch_cost_pooled/est{N}) against the production fit
            (ESTIMATION_OUTPUT/DEMAND_PREP/est{N}), spec 12: index coefficients, the link grid,
            the first stage, and the national phi_t path.
  parquet   two demand parquets row by row on (entity_id, time_id): row sets, column sets and
            the largest absolute and relative gap of every common numeric column.

The parquet pairs this is run on (claude/switch_cost/run_local_prep.sh):
  C2  pooled arm on the PRODUCTION fits  vs  ESTIMATION_OUTPUT/DEMAND_PREP_20261013 (code identity)
  C3  F1 fed the pooled fit as its type-shift fit  vs  the pooled arm (F1 machinery)
  C4  F3hi with r_4 = 0  vs  the pooled arm (F3 machinery)
Writes the report as JSON to ESTIMATION_OUTPUT/SWITCH_COST/switch_cost_checks/.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import switch_cost_sandbox  # noqa: F401  (output tree: import before any path is resolved)
from utils import paths
from utils.switch_cost_sleep_links import NonLinearResults  # noqa: F401 (unpickling)

SPEC12 = "IV_HausmanFull x Tech"
OUT_DIR = switch_cost_sandbox.check_dir(switch_cost_sandbox.LOCAL_BASE / "switch_cost_checks", "OUT_DIR")
KEYS = ["entity_id", "time_id"]


def _spec12(pkl: Path):
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    key = next(k for k in d if SPEC12 in k)
    return d[key]


def _gap(a, b) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    d = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(scale > 0, d / np.where(scale > 0, scale, 1.0), 0.0)
    return {"max_abs": float(np.nanmax(d)) if d.size else 0.0,
            "max_rel": float(np.nanmax(rel)) if d.size else 0.0}


def compare_fits(est: int, new_root: Path, prod_root: Path) -> dict:
    new = _spec12(new_root / f"est{est}" / "estimation_results.pkl")
    prod = _spec12(prod_root / f"est{est}" / "estimation_results.pkl")
    out = {}
    for stage in ("second_stage", "first_stage"):
        rn, rp = new.get(stage), prod.get(stage)
        if rn is None or rp is None:
            out[stage] = {"missing": [rn is None, rp is None]}
            continue
        pn = getattr(rn, "params_native", None)
        pn = rn.params if pn is None else pn
        pp = getattr(rp, "params_native", None)
        pp = rp.params if pp is None else pp
        common = [c for c in pp.index if c in pn.index]
        out[stage] = {"params": _gap(pn[common].values, pp[common].values),
                      "only_new": [c for c in pn.index if c not in pp.index],
                      "only_prod": [c for c in pp.index if c not in pn.index]}
        if stage == "second_stage":
            out[stage]["link"] = [getattr(rn, "link", None), getattr(rp, "link", None)]
            for g in ("si_vgrid", "si_ggrid"):
                if getattr(rn, g, None) is not None and getattr(rp, g, None) is not None:
                    out[stage][g] = _gap(getattr(rn, g), getattr(rp, g))
            out[stage]["nobs"] = [int(getattr(rn, "nobs", -1)), int(getattr(rp, "nobs", -1))]
    tn = pd.read_csv(new_root / f"est{est}" / "national_phi_t.csv")
    tp = pd.read_csv(prod_root / f"est{est}" / "national_phi_t.csv")
    col = next((c for c in tp.columns if SPEC12.replace(" ", "_") in c and c.startswith("phi_t_")), None)
    if col and col in tn.columns:
        m = tp[["year_quarter", col]].merge(tn[["year_quarter", col]], on="year_quarter", suffixes=("_prod", "_new"))
        out["national_phi_t"] = {"column": col, "quarters": len(m), **_gap(m[f"{col}_new"], m[f"{col}_prod"])}
    return out


def compare_parquets(a: Path, b: Path) -> dict:
    A, B = pd.read_parquet(a), pd.read_parquet(b)
    out = {"a": str(a), "b": str(b), "rows": [len(A), len(B)],
           "cols_only_a": sorted(set(A.columns) - set(B.columns)),
           "cols_only_b": sorted(set(B.columns) - set(A.columns))}
    m = A.merge(B, on=KEYS, how="outer", suffixes=("_a", "_b"), indicator=True)
    out["rows_only_a"] = int((m["_merge"] == "left_only").sum())
    out["rows_only_b"] = int((m["_merge"] == "right_only").sum())
    m = m[m["_merge"] == "both"]
    gaps = {}
    for c in sorted(set(A.columns) & set(B.columns) - set(KEYS)):
        ca, cb = m[f"{c}_a"], m[f"{c}_b"]
        if pd.api.types.is_numeric_dtype(ca) and pd.api.types.is_numeric_dtype(cb):
            na_a, na_b = ca.isna(), cb.isna()
            g = _gap(ca.where(~(na_a | na_b), 0.0), cb.where(~(na_a | na_b), 0.0))
            g["nan_mismatch"] = int((na_a != na_b).sum())
            gaps[c] = g
        else:
            gaps[c] = {"unequal": int((ca.astype(str) != cb.astype(str)).sum())}
    out["columns"] = gaps
    worst = max((v.get("max_rel", 0.0) for v in gaps.values() if isinstance(v, dict)), default=0.0)
    out["worst_rel"] = worst
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--est", type=int, required=True)
    f.add_argument("--new-root", default=str(switch_cost_sandbox.local_fit_root("pooled")))
    f.add_argument("--prod-root", default=str(paths.estimation_output() / "DEMAND_PREP"))
    q = sub.add_parser("parquet")
    q.add_argument("--name", required=True, help="label of the check, e.g. C2_est3")
    q.add_argument("a")
    q.add_argument("b")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "fit":
        rep = compare_fits(args.est, Path(args.new_root), Path(args.prod_root))
        name = f"fit_est{args.est}"
    else:
        rep = compare_parquets(Path(args.a), Path(args.b))
        name = args.name
    (OUT_DIR / f"{name}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    if args.cmd == "parquet":
        brief = {k: v for k, v in rep.items() if k != "columns"}
        bad = {c: g for c, g in rep["columns"].items()
               if g.get("max_rel", 0) > 1e-10 or g.get("nan_mismatch", 0) or g.get("unequal", 0)}
        print(json.dumps(brief, indent=1))
        print("columns with gaps above 1e-10 (relative):", json.dumps(bad, indent=1) if bad else "none")
    else:
        print(json.dumps(rep, indent=1))
    print(f"-> {OUT_DIR / (name + '.json')}")


if __name__ == "__main__":
    main()
