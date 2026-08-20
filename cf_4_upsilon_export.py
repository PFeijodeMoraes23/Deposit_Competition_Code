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
     (identity for E1/E2; single-index 'index' E3/E4), so
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

from utils import paths as _paths
from utils import state_transform as _st

# Our status lines print Greek (Υ, φ, Δ); Windows consoles default to cp1252 and would
# raise UnicodeEncodeError on them, so force UTF-8 output (no-op where already UTF-8).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Anchors come from utils.paths, the same accessors the rest of the sleepiness/CF chain uses.
# A `Path(__file__).parents[N]` walk hard-codes the repo's position inside the data tree, which
# holds on this machine and nowhere else: on the cluster the scripts live at HEAD/scripts and the
# walk lands on the parent of HEAD, where the reads below find nothing and the export is silently
# empty. demand_prep_root() also honours SLEEP_OUT_ROOT, so est{e}/estimation_results.pkl,
# est{e}/market_panel_phis.csv and the demand parquets all follow the vintage being exported.
# CF_FOUNDATION stays under estimation_output(): it is production CF input, not sandbox output.
DEMAND_PREP = _paths.demand_prep_root()
CF_DIR = _paths.cf_foundation_dir()

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
    # The estimation-units value of "no Pix". Exported for BOTH branches: the identity
    # branch never builds a phi_nopix parquet, so cf_4_pix.jl needs it to form the
    # scalar subtraction as Υ_pix·(pix − pix_level_zero) rather than Υ_pix·pix.
    _tf = _st.load_transform()
    pix_level_zero = _tf.level_of("pix_exists", raw=0.0)
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
        # counterfactual: set the Pix indicator to its RAW-ZERO level, re-apply the link G.
        # NOT the literal 0.0: the state block is grand-mean centred, so "no Pix" is the
        # value -mean(pix_exists), and writing 0.0 would quietly evaluate the
        # counterfactual at 53% of markets having Pix. A wrong constant here produces a
        # perfectly plausible-looking φ^noPix, so the level is read from the persisted
        # transform and the observed levels are asserted against it.
        pix_lo = pix_level_zero
        dd0 = dd.copy()
        if "pix_exists" in dd0.columns:
            _lv = _st.dummy_levels("pix_exists")
            _u = np.unique(dd0["pix_exists"].to_numpy(float))
            if _lv is not None and len(_u) == 2 and \
                    max(abs(_u[0] - _lv[0]), abs(_u[1] - _lv[1])) > 1e-8:
                raise ValueError(
                    f"{dpath.name}: pix_exists carries levels {tuple(np.round(_u, 8))} but the "
                    f"transform says {tuple(np.round(_lv, 8))}. This parquet was built under a "
                    "different centering than the current state_centering_means.json -- rebuild "
                    "the demand prep before exporting the no-Pix counterfactual."
                )
            dd0["pix_exists"] = pix_lo
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

    else:
        # IDENTITY / LINEAR LINK (E1, E2) — added 2026-08-06 at the user's request.
        #
        # These estimators are plain statsmodels results: no .link, no .params_native, so the
        # phi_from_native path above cannot run. For an identity link φ = S′θ, so the no-Pix
        # counterfactual is exact in closed form and PER ROW:
        #     φ^noPix = φ̂ − θ_pix · (pix − pix_level_zero)
        # This is numerically identical to the scalar fallback cf_4_pix.jl already performs, so
        # it buys no accuracy — what it buys is UNIFORMITY: every active routine now ships the
        # same artifact and CF4 takes one code path instead of branching on link type. The
        # fallback stays in place, so an older export without this file still works.
        #
        # pix_level_zero, NOT 0.0: the state block is grand-mean centred, so "no Pix" is
        # −mean(pix_exists). Writing 0.0 would evaluate the counterfactual at ~53% of markets
        # still having Pix, and would look entirely plausible. Same assertion as the nonlinear
        # branch, for the same reason.
        import pyarrow.parquet as pq
        avail = set(pq.read_schema(dpath).names)
        cols = [c for c in dict.fromkeys(["entity_id", "time_id", "phi_mt", "pix_exists"] + MERGE_KEYS)
                if c in avail]
        gap = [c for c in ("entity_id", "time_id", "phi_mt", "pix_exists") if c not in avail]
        if gap:
            raise KeyError(f"demand parquet {dpath.name} lacks columns needed for the identity-link "
                           f"no-Pix φ: {gap}")
        dd = pd.read_parquet(dpath, columns=cols)
        pm = dd["phi_mt"].to_numpy(float)
        pix = dd["pix_exists"].to_numpy(float)

        _lv = _st.dummy_levels("pix_exists")
        _u = np.unique(pix)
        if _lv is not None and len(_u) == 2 and \
                max(abs(_u[0] - _lv[0]), abs(_u[1] - _lv[1])) > 1e-8:
            raise ValueError(
                f"{dpath.name}: pix_exists carries levels {tuple(np.round(_u, 8))} but the "
                f"transform says {tuple(np.round(_lv, 8))}. This parquet was built under a "
                "different centering than the current state_centering_means.json -- rebuild "
                "the demand prep before exporting the no-Pix counterfactual.")

        phi_nopix = pm - upsilon * (pix - pix_level_zero)
        # Audit: rows already AT the no-Pix level must be unchanged. Exactness is by
        # construction here, so this checks the level constant rather than the arithmetic.
        at_zero = np.abs(pix - pix_level_zero) < 1e-12
        audit = dict(nopix_link="identity",
                     nopix_audit_corr=round(float(np.corrcoef(phi_nopix, pm)[0, 1]), 8)
                     if np.std(phi_nopix) > 0 and np.std(pm) > 0 else None,
                     nopix_audit_max_abs=float(np.max(np.abs(phi_nopix[at_zero] - pm[at_zero])))
                     if at_zero.any() else 0.0)
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
               estim=e, spec=s, link=(link or "identity"), exact_nopix=phi_nopix_name is not None,
               phi_nopix_parquet=phi_nopix_name,
               state_transform_version=_st.STATE_TRANSFORM_VERSION,
               pix_level_zero=pix_level_zero,
               note=("exact link-aware no-Pix φ exported (phi_from_native, Pix set to "
                     "pix_level_zero); upsilon_pix is the Pix AME, reporting-only"
                     if phi_nopix_name
                     else "identity/linear link: φ_cf = φ̂ − Υ_pix·(pix − pix_level_zero) is exact"),
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
