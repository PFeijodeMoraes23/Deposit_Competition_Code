"""
cf_diag_spreads.py
==================
Diagnostic for the deposit-spread variable feeding CF1/CF2 and demand. Negative
spreads are economically valid (a bank paying above the risk-free rate, e.g. CDBs
at >100% CDI), so this does NOT clean anything — it characterizes the distribution,
its deposit-weighted center, its time profile, and how concentrated the extreme
tail is, so we can judge which (if any) values are implicit-rate artifacts.

Outputs:
  CF_FOUNDATION/diag_spreads_E{e}_spec_{s}.png   (2-panel figure)
  prints per-type percentile table + extreme-tail deposit-weight shares.

Usage:
  python cf_diag_spreads.py --estim 6 --spec 12
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # headless / OneDrive-safe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Windows consoles default to cp1252 and raise UnicodeEncodeError on any non-ASCII
# character in a print (phi, arrows, Upsilon, x). That usually fires on a STATUS line after
# the real work is done, so the script exits non-zero and reports failure for a computation
# that succeeded -- three such false failures on 2026-07-29. Force UTF-8 (no-op if already).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = Path(__file__).resolve().parents[2]
DATA = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
DEMAND_PREP = DATA / "ESTIMATION_OUTPUT" / "DEMAND_PREP"
CF_DIR = DATA / "ESTIMATION_OUTPUT" / "CF_FOUNDATION"
def _discover_demand_parquet(estim: int, spec: int) -> Path:
    """Auto-discover the routine's demand parquet (newest mtime wins) instead of a stale
    static prefix dict (E3 Single-Index, E4 Single-Index + Time; cluster default {3,4})."""
    cands = [p for p in DEMAND_PREP.glob(f"demand_{estim}_*spec_{spec}.parquet")
             if "final" not in p.name.lower()]
    if not cands:
        raise FileNotFoundError(f"No demand parquet for estim={estim} spec={spec} in {DEMAND_PREP}")
    return max(cands, key=lambda p: p.stat().st_mtime)
K_LABEL = {1: "demand", 2: "savings", 4: "CDB(k4)", 5: "prepaid(k5)"}


def wmean(v, w):
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return np.average(v[m], weights=w[m]) if m.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estim", type=int, default=6)
    ap.add_argument("--spec", type=int, default=12)
    ap.add_argument("--tail", type=float, default=1500.0, help="extreme-tail threshold |bps|")
    args = ap.parse_args()

    path = _discover_demand_parquet(args.estim, args.spec)
    df = pd.read_parquet(path)
    df["sp"] = pd.to_numeric(df["spread_qoq"], errors="coerce")
    df["dep"] = pd.to_numeric(df["deposit_balance"], errors="coerce").clip(lower=0)
    df["k"] = pd.to_numeric(df["deposit_type"], errors="coerce").astype("Int64")
    df["t"] = df["time_id"].astype(str)
    print(f"Loaded {len(df):,} rows from {path.name}\n")

    # ── Per-type percentile + tail table ───────────────────────────────────
    pcts = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    print(f"{'type':10s} {'n':>7s} {'%neg':>6s} " +
          " ".join(f"p{p:02d}" .rjust(8) for p in pcts) + f" {'depWmean':>9s} "
          f"{'tailDepW%':>9s}")
    for k in sorted(x for x in df["k"].dropna().unique()):
        g = df[df["k"] == k]
        s = g["sp"].to_numpy(); w = g["dep"].to_numpy()
        qs = np.nanpercentile(s, pcts)
        tail = np.abs(s) > args.tail
        tail_depw = 100 * np.nansum(w[tail & np.isfinite(s)]) / max(np.nansum(w), 1e-9)
        print(f"{K_LABEL.get(int(k), k):10s} {len(g):7d} {100*np.mean(s<0):6.1f} " +
              " ".join(f"{q:8.0f}" for q in qs) +
              f" {wmean(s, w):9.1f} {tail_depw:9.2f}")

    # ── Figure: (a) clipped distributions, (b) dep-wtd mean over time ──────
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    clip = 1500
    for k in sorted(x for x in df["k"].dropna().unique()):
        s = df.loc[df["k"] == k, "sp"].to_numpy()
        s = s[np.isfinite(s)]
        ax[0].hist(np.clip(s, -clip, clip), bins=80, histtype="step",
                   density=True, label=K_LABEL.get(int(k), str(k)))
    ax[0].axvline(0, color="k", lw=0.6)
    ax[0].set(title=f"spread_qoq distribution (clipped ±{clip} bps)",
              xlabel="spread (bps/qtr)", ylabel="density")
    ax[0].legend(fontsize=8)

    order = sorted(df["t"].unique())
    for k in sorted(x for x in df["k"].dropna().unique()):
        g = df[df["k"] == k]
        ser = [wmean(g.loc[g["t"] == t, "sp"].to_numpy(),
                     g.loc[g["t"] == t, "dep"].to_numpy()) for t in order]
        ax[1].plot(range(len(order)), ser, label=K_LABEL.get(int(k), str(k)), lw=1.2)
    ax[1].axhline(0, color="k", lw=0.6)
    step = max(1, len(order) // 12)
    ax[1].set_xticks(range(0, len(order), step))
    ax[1].set_xticklabels([order[i] for i in range(0, len(order), step)], rotation=45, fontsize=7)
    ax[1].set(title="deposit-weighted mean spread over time",
              ylabel="spread (bps/qtr)")
    ax[1].legend(fontsize=8)

    CF_DIR.mkdir(parents=True, exist_ok=True)
    out = CF_DIR / f"diag_spreads_E{args.estim}_spec_{args.spec}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nWrote figure -> {out}")


if __name__ == "__main__":
    main()
