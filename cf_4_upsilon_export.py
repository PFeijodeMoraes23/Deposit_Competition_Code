"""
cf_4_upsilon_export.py
======================
READ-ONLY recovery of the Pix sleepiness effect for CF4 (Pix-as-switching).

The sleep step does not export the sleepiness model in a Julia-readable form — it pickles the
whole results object (`est{e}/estimation_results.pkl`) and writes only the already-built φ_mt.
CF4 (`cf_4_pix.jl`) needs to form the no-Pix counterfactual φ^noPix. This script recovers what it
needs WITHOUT touching the sleep estimation: it reads the routine's pickle, figures out WHICH
φ-spec feeds the demand parquet's `phi_mt` (matching against `est{e}/market_panel_phis.csv`), and
produces two things for that spec:

  1. Υ_pix — the `interaction_pix_exists` coefficient (the Pix AME under a nonlinear link), for
     reporting and the identity-link fallback; written to a small JSON.
  2. The EXACT per-row no-Pix φ (nonlinear links only). Sleepiness is φ = G(S′θ) with G the link
     (identity for E1/E2; logit E3/E4; cubic single-index 'index' E5/E6; joint 'sieve' E7/E8), so
     the level subtraction φ̂ − Υ_pix·pix is exact ONLY for the identity link. For a nonlinear G
     the correct no-Pix φ re-applies the link to the Pix-removed index,
     φ^noPix = G(index − θ_pix·pix) = phi_from_native(df, ss, link) with pix_exists=0. We compute
     that per demand-parquet row and export it, keyed by (entity_id, time_id), which CF4 joins 1:1
     to its rows. An audit (corr, max|Δ| vs the stored φ_mt) confirms phi_from_native reproduces
     the estimation's own φ before it is re-evaluated at pix=0.

This is a DESCRIPTIVE CF and the script is READ-ONLY w.r.t. the estimation.

Output:
  CF_FOUNDATION/upsilon_pix_E{e}_spec_{s}.json
    {upsilon_pix, source_key, phi_col, match_corr, match_med_abs_diff, estim, spec, link,
     exact_nopix, phi_nopix_parquet, nopix_audit_corr, nopix_audit_max_abs}
  CF_FOUNDATION/phi_nopix_E{e}_spec_{s}.parquet   (nonlinear links only)
    [entity_id, time_id, CodConglomeradoPrudencial, deposit_type, mca_code, phi_mt, phi_mt_nopix]

Usage:
  python cf_4_upsilon_export.py --estim 6 --spec 12
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Our status lines print Greek (Υ, φ, Δ); Windows consoles default to cp1252 and would
# raise UnicodeEncodeError on them, so force UTF-8 output (no-op where already UTF-8).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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
    dpath = _discover_demand_parquet(e, s)
    dm = pd.read_parquet(dpath, columns=MERGE_KEYS + ["phi_mt"])
    # ── est{e} market-level φ per (IV × block) ───────────────────────────────
    # market_panel_phis.csv is ~150 columns and multi-GB; a full read can OOM the C parser.
    # We only need the phi_mt_* columns + the merge-key sources, so read just those (usecols) —
    # ~10× less memory and faster.
    phi_cols = [c for c in pd.read_csv(mp_path, nrows=0).columns if c.startswith("phi_mt_")]
    if not phi_cols:
        raise ValueError(f"No phi_mt_* columns in {mp_path.name}")
    mp = pd.read_csv(mp_path, low_memory=False,
                     usecols=["CodConglomeradoPrudencial", "mca_code", "deposit_type",
                              "year", "quarter"] + phi_cols)
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

    # ── exact link-aware no-Pix φ (nonlinear links only) ─────────────────────
    # For a nonlinear link G, φ = G(S′θ), so the level subtraction φ̂ − Υ_pix·pix
    # is WRONG: Υ_pix here is the Pix AME (`params`, reporting-only), not ∂φ/∂pix, and
    # G ≠ identity. The exact no-Pix counterfactual re-applies the link to the
    # Pix-removed index, φ^noPix = G(index − θ_pix·pix) = phi_from_native(df, ss, link)
    # with pix_exists = 0. We compute it per demand-parquet row (the exact CF grain)
    # and export it; cf_4_pix.jl consumes this instead of the scalar subtraction.
    # Identity links (E1/E2: a statsmodels result with no .link/.params_native) skip
    # this branch — there the scalar subtraction is exact and CF4 falls back to it.
    link = getattr(ss, "link", None)
    native = getattr(ss, "params_native", None)
    phi_nopix_name = None
    audit = {}
    if link is not None and native is not None and link != "identity":
        import pyarrow.parquet as pq
        from utils.sleep_links import phi_from_native
        state_svs = [p[len("interaction_"):] for p in native.index
                     if str(p).startswith("interaction_")]
        avail = set(pq.read_schema(dpath).names)
        gap = [c for c in (["entity_id"] + state_svs) if c not in avail]
        if gap:
            raise KeyError(f"demand parquet {dpath.name} lacks columns needed for the exact "
                           f"no-Pix φ (phi_from_native): {gap}")
        cols = [c for c in dict.fromkeys(["entity_id", "time_id", "phi_mt"] + MERGE_KEYS + state_svs)
                if c in avail]
        dd = pd.read_parquet(dpath, columns=cols)
        pm = dd["phi_mt"].to_numpy(float)
        # audit: phi_from_native (Pix as observed) must reproduce the stored phi_mt
        phi_hat = np.asarray(phi_from_native(dd, ss, link), float)
        audit = dict(nopix_link=link,
                     nopix_audit_corr=round(float(np.corrcoef(phi_hat, pm)[0, 1]), 8),
                     nopix_audit_max_abs=float(np.max(np.abs(phi_hat - pm))))
        # counterfactual: zero the Pix indicator, re-apply the link G
        dd0 = dd.copy()
        if "pix_exists" in dd0.columns:
            dd0["pix_exists"] = 0.0
        phi_nopix = np.asarray(phi_from_native(dd0, ss, link), float)
        outp = pd.DataFrame(dict(
            entity_id=dd["entity_id"].astype(str).to_numpy(),
            time_id=dd["time_id"].astype(str).to_numpy(),
            CodConglomeradoPrudencial=dd["CodConglomeradoPrudencial"].astype(str).to_numpy(),
            deposit_type=dd["deposit_type"].to_numpy(),
            mca_code=dd["mca_code"].astype(str).to_numpy(),
            phi_mt=pm, phi_mt_nopix=phi_nopix))
        phi_nopix_name = f"phi_nopix_E{e}_spec_{s}.parquet"
        CF_DIR.mkdir(parents=True, exist_ok=True)
        outp.to_parquet(CF_DIR / phi_nopix_name, index=False)

    out = dict(upsilon_pix=upsilon, source_key=src_key, phi_col=phi_col,
               match_corr=round(corr, 5), match_med_abs_diff=round(med, 6),
               estim=e, spec=s, link=link, exact_nopix=phi_nopix_name is not None,
               phi_nopix_parquet=phi_nopix_name,
               note=("exact link-aware no-Pix φ exported (phi_from_native, Pix zeroed); "
                     "upsilon_pix is the Pix AME, reporting-only" if phi_nopix_name
                     else "identity/linear link: φ_cf = φ̂ − Υ_pix·pix is exact"),
               **audit)
    CF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CF_DIR / f"upsilon_pix_E{e}_spec_{s}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  Υ_pix = {upsilon:.6g}  (AME; from {src_key}; match corr={corr:.4f}, med|Δ|={med:.2e})")
    if phi_nopix_name:
        print(f"  exact no-Pix φ ({link} link) → {phi_nopix_name}  "
              f"[audit corr={audit['nopix_audit_corr']}, max|Δ|={audit['nopix_audit_max_abs']:.1e}]")
    print(f"  Wrote → {out_path.name}")


if __name__ == "__main__":
    main()
