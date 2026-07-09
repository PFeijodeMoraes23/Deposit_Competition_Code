"""
cf_4_upsilon_export.py
======================
READ-ONLY recovery of the Pix sleepiness coefficient Υ_pix for CF4 (Pix-as-switching).

The sleep step (`estimation_1_sleep.py`) does not export Υ in a Julia-readable form — it
pickles the whole statsmodels object (`est{e}/estimation_results.pkl`) and writes only the
already-summed φ_mt. CF4 (`cf_4_pix.jl`) needs the scalar coefficient on `pix_exists` so it can
form the no-Pix counterfactual φ_cf = φ̂ − Υ_pix·pix_exists. This script recovers it WITHOUT
touching the sleep estimation: it reads the routine's pickle, figures out WHICH φ-spec actually
feeds the demand parquet's `phi_mt` (by matching against `est{e}/market_panel_phis.csv`), pulls
that spec's `interaction_pix_exists` coefficient, and writes a small JSON.

Because φ is (near-)linear in the sleep state (the demand `phi_mt` correlates ~0.999 with the
linear index), φ_cf = φ̂ − Υ_pix·pix is an accurate first pass; the script reports the match
fidelity (corr, median |Δ|) so the approximation is auditable. This is a DESCRIPTIVE CF.

Output:
  CF_FOUNDATION/upsilon_pix_E{e}_spec_{s}.json
    {upsilon_pix, source_key, phi_col, match_corr, match_med_abs_diff, estim, spec}

Usage:
  python cf_4_upsilon_export.py --estim 6 --spec 12
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
DEMAND_PREP = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "ESTIMATION_OUTPUT" / "DEMAND_PREP"
CF_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "ESTIMATION_OUTPUT" / "CF_FOUNDATION"

MERGE_KEYS = ["CodConglomeradoPrudencial", "mca_code", "deposit_type", "time_id"]


def _discover_demand_parquet(estim: int, spec: int) -> Path:
    cands = [p for p in DEMAND_PREP.glob(f"demand_{estim}_*spec_{spec}.parquet")
             if "final" not in p.name.lower()]
    if not cands:
        raise FileNotFoundError(f"No demand parquet for estim={estim} spec={spec} in {DEMAND_PREP}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def _phicol_to_pklkey(phi_col: str) -> str:
    """'phi_mt_IV_HausmanFull_x_Tech' -> 'IV_HausmanFull x Tech' (the results_dict key)."""
    body = phi_col[len("phi_mt_"):]
    iv, block = body.rsplit("_x_", 1)
    return f"{iv} x {block}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estim", type=int, default=6)
    ap.add_argument("--spec", type=int, default=12)
    args = ap.parse_args()
    e, s = args.estim, args.spec

    est_dir = DEMAND_PREP / f"est{e}"
    pkl_path = est_dir / "estimation_results.pkl"
    mp_path = est_dir / "market_panel_phis.csv"
    for p in (pkl_path, mp_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing sleep output for E{e}: {p}")

    # ── demand parquet φ_mt + market keys ────────────────────────────────────
    dm = pd.read_parquet(_discover_demand_parquet(e, s),
                         columns=MERGE_KEYS + ["phi_mt"])
    # ── est{e} market-level φ per (IV × block) ───────────────────────────────
    mp = pd.read_csv(mp_path, low_memory=False)
    phi_cols = [c for c in mp.columns if c.startswith("phi_mt_")]
    if not phi_cols:
        raise ValueError(f"No phi_mt_* columns in {mp_path.name}")
    mp["time_id"] = mp["year"].astype("Int64").astype(str) + "Q" + mp["quarter"].astype("Int64").astype(str)
    for k in MERGE_KEYS:
        dm[k] = dm[k].astype(str); mp[k] = mp[k].astype(str)
    m = dm.merge(mp[MERGE_KEYS + phi_cols].drop_duplicates(MERGE_KEYS), on=MERGE_KEYS, how="left")

    # ── which φ-spec feeds demand phi_mt? lowest median |Δ| wins ──────────────
    a = m["phi_mt"].to_numpy(float)
    best = None
    for c in phi_cols:
        b = m[c].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 100:
            continue
        med = float(np.median(np.abs(a[ok] - b[ok])))
        corr = float(np.corrcoef(a[ok], b[ok])[0, 1])
        # prefer smaller median |Δ|; tie-break toward the sleep headline IV (HausmanFull)
        rank = (med, 0 if "IV_HausmanFull" in c else 1)
        if best is None or rank < best[0]:
            best = (rank, c, corr, med)
    if best is None:
        raise RuntimeError("Could not match demand phi_mt to any est phi column.")
    _, phi_col, corr, med = best
    src_key = _phicol_to_pklkey(phi_col)

    # ── pull Υ_pix from the pickle for that spec ─────────────────────────────
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    if src_key not in res:
        raise KeyError(f"Matched φ-spec {src_key!r} not in {pkl_path.name} keys: {list(res)}")
    ss = res[src_key]["second_stage"]
    if ss is None or "interaction_pix_exists" not in ss.params:
        raise KeyError(f"No interaction_pix_exists in second_stage of {src_key!r}")
    upsilon = float(ss.params["interaction_pix_exists"])

    out = dict(upsilon_pix=upsilon, source_key=src_key, phi_col=phi_col,
               match_corr=round(corr, 5), match_med_abs_diff=round(med, 6),
               estim=e, spec=s, note="descriptive first pass; phi ~linear in state (see match_corr)")
    CF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CF_DIR / f"upsilon_pix_E{e}_spec_{s}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  Υ_pix = {upsilon:.6g}  (from {src_key}; match corr={corr:.4f}, med|Δ|={med:.2e})")
    print(f"  Wrote → {out_path.name}")


if __name__ == "__main__":
    main()
