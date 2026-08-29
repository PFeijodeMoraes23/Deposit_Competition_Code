"""
diag_cf1_franchise_dataonly.py
==========================
LOCAL-NOW descriptive version of CF1 (franchise value — sleepiness decomposition),
computable RIGHT NOW from the local demand-prep parquet, WITHOUT the BLP results
(δ̂, θ̂) and therefore without waiting for the cluster estimation to finish.

It uses the already-estimated sleeper share φ̂ and the data-implied active deposits
Dep^Act (both carried in the demand-prep parquet) instead of re-solving the demand
model. This gives a first, honest magnitude for the headline object; the
model-based cf1_franchise.jl reproduces it (and enables spread counterfactuals)
once results land.

Decomposition (mirrors foundation_deposit_sim so the two are directly comparable):
  accrual         g   = 1 + r^dep_q = 1 + (r^f_q − ρ_q)
  φ̂ path          Dep_t = Dep^Act + φ̂·g·Dep_{t−1},     Dep_0 = observed stock
  φ=0 path        Dep_t = Dep^Act/(1−φ̂)  (t≥1),         Dep_0 = observed stock
                  (the active term is (1−φ)·M·s, so φ=0 scales the influx up by 1/(1−φ̂))
  value           V = Σ_{t=0}^{T} β^t · Dep_t · ρ_q          (quarterly markdown)
  ΔV^sleep        = V(φ̂) − V(φ=0)

The t=0 term (observed stock × markdown) is identical across scenarios and cancels
in ΔV, exactly as in the model-based simulator.

⚠ Descriptive only: holds active deposits at their observed data-implied level (no
spread response, no re-solve). Not a substitute for the model-based version.

Usage (runs on LOCAL data; needs phî & Dep^Act columns in the parquet):
  python diag_cf1_franchise_dataonly.py --estim 6 --spec 12 --beta 0.9 --horizon 50
  python diag_cf1_franchise_dataonly.py --parquet <path> --beta 0.9
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import sys
from pathlib import Path

# Status lines print arrows ("phi <- phi_mt"); Windows consoles default to cp1252 and raise
# UnicodeEncodeError on them, which kills the script AFTER the data is loaded -- i.e. it
# reports failure for work that succeeded. Force UTF-8 (no-op where already UTF-8). Same
# guard as sleep_upsilon_export.py and scrape_forward_rf.py.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
DATA = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
DEMAND_PREP = DATA / "ESTIMATION_OUTPUT" / "DEMAND_PREP"
CF_DIR = DATA / "ESTIMATION_OUTPUT" / "CF_FOUNDATION"

def _discover_demand_parquet(estim: int, spec: int) -> Path:
    """Auto-discover the routine's demand parquet (new 8-routine scheme; newest mtime
    wins). The old static prefix dict is stale after the 2026-06-24 relabel
    (E3 Single-Index, E4 Single-Index + Time; cluster default {3,4})."""
    cands = [p for p in DEMAND_PREP.glob(f"demand_{estim}_*spec_{spec}.parquet")
             if "final" not in p.name.lower()]
    if not cands:
        raise FileNotFoundError(
            f"No demand parquet for estim={estim} spec={spec} in {DEMAND_PREP} "
            f"(demand_{estim}_*spec_{spec}.parquet, excl _final).")
    return max(cands, key=lambda p: p.stat().st_mtime)


def _pick(df: pd.DataFrame, candidates, what: str) -> np.ndarray:
    for c in candidates:
        if c in df.columns:
            print(f"    {what:<14s} ← {c}")
            return pd.to_numeric(df[c], errors="coerce").to_numpy()
    raise KeyError(f"{what}: none of {candidates} in parquet columns "
                   f"({sorted(df.columns)[:40]}…). Point --parquet at the demand-prep "
                   f"file that carries φ̂ and Dep^Act.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estim", type=int, default=6)
    ap.add_argument("--spec", type=int, default=12)
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--parquet", type=str, default=None)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--horizon", type=int, default=50)
    args = ap.parse_args()

    if args.parquet:
        path = Path(args.parquet)
    else:
        path = _discover_demand_parquet(args.estim, args.spec)
    if not path.exists():
        raise FileNotFoundError(f"Demand parquet not found: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows from {path.name}")

    phi = np.clip(_pick(df, ["phi_mt", "phi_local_mt", "phi_local", "phi"], "phi"), 0.0, 0.999)
    dep0 = np.maximum(_pick(df, ["deposit_balance", "Dep", "Dep_total"], "Dep stock"), 0.0)
    depact = np.maximum(_pick(df, ["Dep_Act", "active_deposits", "deposit_active"], "Dep^Act"), 0.0)
    # Quarterly accrual & markdown. CONVENTION: panel spreads are in BASIS POINTS
    # (spread_qoq median ~81 bps); risk_free_qoq is a FRACTION (~0.025). Convert
    # spreads to per-quarter fractions (÷1e4) and winsorize extreme outliers
    # (raw range −4939…4125 bps are data errors) before forming rates.
    BPS = 1e-4
    sp_raw = _pick(df, ["spread_qoq", "spread_q"], "spread_q(bps)")
    lo, hi = np.nanpercentile(sp_raw, [1, 99])
    sp_q = np.clip(sp_raw, lo, hi) * BPS
    markdown_q = sp_q
    try:
        rdep_q = _pick(df, ["deposit_rate_qoq", "rdep_qoq"], "r^dep_q")  # already a fraction
    except KeyError:
        rf_q = _pick(df, ["risk_free_qoq", "risk_free_qoq_lag", "selic_qoq"], "r^f_q")
        rdep_q = rf_q - sp_q

    is_B = df.get("is_B")
    is_B = (is_B.astype(bool).to_numpy() if is_B is not None
            else np.zeros(len(df), dtype=bool))
    dtype = pd.to_numeric(df.get("deposit_type", 0), errors="coerce").fillna(0).astype(int).to_numpy()
    firm = df.get("CodConglomeradoPrudencial", pd.Series(range(len(df)))).astype(str).to_numpy()

    beta, T = args.beta, args.horizon
    accr = 1.0 + rdep_q
    active_full = depact / np.clip(1.0 - phi, 1e-6, None)   # = M·s (φ=0 influx)

    # Roll both scenarios T quarters (vectorized over rows).
    dep_phi = dep0.copy()
    V_phi = dep0 * markdown_q          # t=0 term (β^0)
    V_0 = dep0 * markdown_q            # identical t=0 term (cancels in ΔV)
    for t in range(1, T + 1):
        bt = beta ** t
        dep_phi = depact + phi * accr * dep_phi
        V_phi += bt * dep_phi * markdown_q
        V_0 += bt * active_full * markdown_q

    dV = V_phi - V_0
    out = pd.DataFrame(dict(CodConglomeradoPrudencial=firm, is_B=is_B,
                            deposit_type=dtype, V_phi=V_phi, V_0=V_0, dV_sleep=dV))
    CF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CF_DIR / f"cf1_franchise_dataonly_E{args.estim}_spec_{args.spec}{args.suffix}.parquet"
    out.to_parquet(out_path, index=False)

    tot_phi, tot_dv = V_phi.sum(), dV.sum()
    print(f"\n  === CF1 (DATA-ONLY, descriptive) β={beta} T={T} ===")
    print(f"  Total V(φ̂)={tot_phi/1e9:,.3f} bn  ΔV_sleep={tot_dv/1e9:,.3f} bn  "
          f"({100*tot_dv/max(abs(tot_phi),1e-12):.1f}% from inertia)")
    for lbl, m in (("B", is_B), ("D", ~is_B)):
        if m.any():
            vp, dv = V_phi[m].sum(), dV[m].sum()
            print(f"    {lbl}: V(φ̂)={vp/1e9:,.3f}  ΔV={dv/1e9:,.3f}  ({100*dv/max(abs(vp),1e-12):.1f}%)")
    print(f"  Wrote → {out_path.name}  (feed to make_cf1_franchise_table.py)")


if __name__ == "__main__":
    main()
