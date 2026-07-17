"""
cf_forward_rf.py
================
Build the forward risk-free (Selic) path r^f_t for the counterfactual horizon and
cache it as a CSV that the Julia CF engines read (estimation_bbl_2_fwd_sim.jl ψ4 funding base;
later CF3/CF5 equilibrium re-solves).

WHY A REAL FORWARD CURVE (not a flat placeholder). In the BBL value basis (eq:16),
ψ4 = Σ_t β^t r^f_t · Σ_k Dep_t enters with coefficient −(1+ζ). If r^f_t is held FLAT,
ψ4 = r^f · Σ_t β^t Σ_k Dep_t is just a rescaling of ψ2 = Σ_t β^t Σ_k Dep_t, so ω and
ζ are COLLINEAR and ζ (funding-cost pass-through) is unidentified. A time-varying
forward path breaks that collinearity, which is the point of sourcing the market
curve here rather than holding the last Selic.

SOURCE (BCB open APIs, no key):
  * SGS series 4189 — Selic accumulated in the month, annualized (% p.a.): the
    current level anchor.
  * Expectations (Focus) — ExpectativasMercadoAnuais, Indicador='Selic': the market
    median annual Selic for each future calendar year (DataReferencia). We take the
    latest survey date (Data) and read one median per reference year.
The forward path declines from the current Selic toward the Focus long-run median
(the last available reference year, ≈ the neutral rate) and is held at that neutral
level beyond the Focus horizon. Annual → quarterly: r^f_q = (1+Selic/100)^{1/4} − 1.

OFFLINE FALLBACK (documented, deterministic): if the APIs are unreachable, mean-revert
the last observed panel Selic to a neutral rate with a quarterly AR coefficient. The
CSV records `source` so a fallback path is never mistaken for the live curve.

Output:
  processed/ESTIMATION_OUTPUT/COST_FWD/forward_rf_qoq.csv
    columns: h (1..T), cal_q, selic_ann_pct, rf_qoq, source

Usage:
  python cf_forward_rf.py --horizon 50 --start 2026Q1
  python cf_forward_rf.py --horizon 50 --start 2026Q1 --offline   # force fallback
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
COST_FWD = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "ESTIMATION_OUTPUT" / "COST_FWD"
DEMAND_PREP = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "ESTIMATION_OUTPUT" / "DEMAND_PREP"

SGS_SELIC = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados/ultimos/1?formato=json"
FOCUS_URL = ("https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
             "ExpectativasMercadoAnuais")
NEUTRAL_FALLBACK = 10.0   # % p.a. long-run Selic if Focus is unreachable
AR_FALLBACK = 0.85        # quarterly mean-reversion speed for the offline path


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _get_json(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _fetch_current_selic():
    """Latest annualized Selic (% p.a.) from SGS 4189."""
    js = _get_json(SGS_SELIC)
    return float(js[-1]["valor"])


def _fetch_focus_annual():
    """{reference_year:int -> median Selic % p.a.} from the latest Focus survey date."""
    # Build the odata query manually so $-params stay literal and spaces become %20
    # (olinda rejects the urlencode default of %24 for $ and + for space).
    params = ("$top=400&$orderby=Data desc&$format=json"
              "&$select=Data,DataReferencia,Mediana&$filter=Indicador eq 'Selic'")
    url = FOCUS_URL + "?" + urllib.parse.quote(params, safe="$=&,'")
    js = _get_json(url)
    rows = js.get("value", [])
    if not rows:
        raise RuntimeError("Focus returned no rows")
    latest = max(r["Data"] for r in rows)
    out = {}
    for r in rows:
        if r["Data"] != latest:
            continue
        try:
            yr = int(str(r["DataReferencia"])[:4])
        except (ValueError, TypeError):
            continue
        out[yr] = float(r["Mediana"])
    if not out:
        raise RuntimeError("Focus had no annual Selic reference years")
    return latest, out


def _cal_quarters(start: str, T: int):
    """['2026Q1', '2026Q2', …] of length T from a 'YYYYQn' start."""
    y, qn = int(start[:4]), int(start[5])
    out = []
    for _ in range(T):
        out.append(f"{y}Q{qn}")
        qn += 1
        if qn > 4:
            qn = 1
            y += 1
    return out


def _annual_to_quarterly(selic_ann_pct: np.ndarray) -> np.ndarray:
    return (1.0 + selic_ann_pct / 100.0) ** 0.25 - 1.0


def _last_panel_selic_ann():
    """Fallback anchor: last observed risk_free_qoq_lag → annualized %."""
    cands = sorted(DEMAND_PREP.glob("demand_6_*spec_12.parquet"))
    if not cands:
        cands = sorted(DEMAND_PREP.glob("demand_*spec_12.parquet"))
    df = pd.read_parquet(cands[-1], columns=["time_id", "risk_free_qoq_lag"])
    df["t"] = df["time_id"].astype(str)
    last = df[df["t"] == df["t"].max()]
    rf_q = pd.to_numeric(last["risk_free_qoq_lag"], errors="coerce").median()
    return 100.0 * ((1.0 + rf_q) ** 4 - 1.0)


def build_path(T: int, start: str, offline: bool):
    cal = _cal_quarters(start, T)
    years = np.array([int(c[:4]) for c in cal])
    qnum = np.array([int(c[5]) for c in cal])
    # fractional calendar position (year + (q-1)/4) for smooth interpolation
    fpos = years + (qnum - 1) / 4.0

    source = None
    try:
        if offline:
            raise RuntimeError("offline requested")
        cur = _fetch_current_selic()
        survey, focus = _fetch_focus_annual()
        # Anchor the current calendar year to max(current level, Focus this-year median):
        # the current spot Selic is the best near-term anchor.
        cur_year = years[0]
        focus.setdefault(cur_year, cur)
        yrs_sorted = sorted(focus)
        neutral = focus[yrs_sorted[-1]]
        # Build a per-year annual Selic covering all horizon years (hold neutral beyond Focus).
        node_years = np.array([yrs_sorted[0] - 0.0] + [y for y in yrs_sorted], dtype=float)
        node_vals = np.array([focus[yrs_sorted[0]]] + [focus[y] for y in yrs_sorted], dtype=float)
        # place each year's median at mid-year (y+0.5) for interpolation; extend flat to neutral
        node_pos = np.array([y + 0.5 for y in yrs_sorted], dtype=float)
        node_med = np.array([focus[y] for y in yrs_sorted], dtype=float)
        selic_ann = np.interp(fpos, node_pos, node_med, left=node_med[0], right=neutral)
        source = f"focus:{survey}|sgs4189:{cur:.2f}|neutral:{neutral:.2f}"
    except Exception as e:  # offline / API change → deterministic mean-reversion
        anchor = _last_panel_selic_ann()
        neutral = NEUTRAL_FALLBACK
        selic_ann = np.empty(T)
        prev = anchor
        for i in range(T):
            prev = neutral + AR_FALLBACK * (prev - neutral)
            selic_ann[i] = prev
        source = f"FALLBACK:meanrev(anchor={anchor:.2f}->neutral={neutral:.2f},ar={AR_FALLBACK}) [{type(e).__name__}]"

    rf_qoq = _annual_to_quarterly(selic_ann)
    return pd.DataFrame(dict(h=np.arange(1, T + 1), cal_q=cal,
                             selic_ann_pct=np.round(selic_ann, 4),
                             rf_qoq=rf_qoq, source=source))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--start", type=str, default="2026Q1", help="first forward quarter YYYYQn")
    ap.add_argument("--offline", action="store_true", help="force the mean-reversion fallback")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    df = build_path(args.horizon, args.start, args.offline)
    COST_FWD.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else COST_FWD / "forward_rf_qoq.csv"
    df.to_csv(out, index=False)
    src = df["source"].iloc[0]
    print(f"  Forward r^f path ({len(df)} quarters) → {out.name}")
    print(f"  source: {src}")
    show = df.iloc[[0, 3, 7, 11, 19, 39, len(df) - 1]] if len(df) >= 40 else df
    for _, r in show.iterrows():
        print(f"    h={int(r['h']):>2} {r['cal_q']}  Selic={r['selic_ann_pct']:>6.2f}%  "
              f"r^f_q={r['rf_qoq']:.5f}")


if __name__ == "__main__":
    main()
