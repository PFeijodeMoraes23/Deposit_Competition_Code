"""check_state_centering.py -- verification for the grand-mean centering of the
sleepiness state block.  READ-ONLY: it never writes an estimation artefact.

Author: Pedro Feijó de Moraes

Centering S~ = S - S_bar is a pure reparametrisation.  Slopes, AMEs and every
fitted phi must be numerically unchanged; only the constant moves, by exactly
    theta_0_new = theta_0_old + sum_k theta_k * S_bar_k.
This script is what turns "must be" into "is".

    --tier0   ORACLE. No re-run, no re-estimation, seconds. Takes the CURRENT
              pickles, applies the translation formula by hand, and checks that
              the translated coefficients evaluated on the CENTRED frame give
              back the very same phi as the originals on the UNCENTRED frame.
              This proves the JSON and the formula BEFORE anything is touched.

    --gate    DIFF GATE. Run after re-estimation. Compares the new pickles and
              phi files against a pre-change backup directory: phi identity,
              theta reconciliation, and AME invariance (continuous AMEs
              unchanged; the pix AME unchanged EXACTLY -- the sharpest test of
              the centered-dummy AME fix).

Exit code is non-zero on any FAIL.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths as _paths_mod
from utils import state_transform as _st
from utils.sleep_links import NonLinearResults  # noqa: F401 (needed for unpickling)

DEMAND_PREP = _paths_mod.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP"

# max|d phi| per estimator family. E1/E2 are exactly equivariant (an invertible
# triangular reparametrisation of the design; two-way demeaning is linear and
# commutes). E7/E8 standardise S internally, so Sn is bit-identical under any
# affine pre-transform. E3-E6 run least_squares(trf, cauchy) from zeros with no
# x_scale, so the trust-region PATH is not invariant even though the optimum is.
TOL_PHI = {1: 1e-9, 2: 1e-9, 3: 2e-3, 4: 2e-3, 5: 5e-4, 6: 5e-4, 7: 1e-8, 8: 1e-8}
TOL_ORACLE = 1e-10          # the oracle is pure arithmetic; it should be ~1e-15


# ==============================================================================
# helpers
# ==============================================================================
def _native(res):
    """Index coefficients. phi is ALWAYS built from these, never from the AMEs."""
    p = getattr(res, "params_native", None)
    return res.params if p is None else p


def _link(res):
    """E1/E2 are statsmodels results: a bare linear index, no link."""
    return getattr(res, "link", None) or "identity"


def _phi(df, native, link, res):
    """phi at the given coefficients -- mirrors utils.sleep_links.phi_from_native,
    plus the 'identity' case used by the linear E1/E2 (which, matching
    calculate_pooled_phis, is NOT clipped)."""
    from utils.sleep_links import _build_phi_X, link_cdf

    phi_params = [p for p in native.index if not str(p).startswith("v_hat")]
    X = _build_phi_X(df, phi_params)
    index = X @ native[phi_params].values.astype(float)

    if link == "identity":
        return index
    if link == "index":
        b = np.asarray(res.si_b)
        vs = (index - res.si_vmu) / (res.si_vsd if res.si_vsd else 1.0)
        g = np.zeros(len(vs))
        for d in range(len(b)):
            g += b[d] * vs ** d
        return np.clip(g, 0.0, 1.0)
    if link in ("sieve", "kernel"):
        idx_params = [p for p in phi_params if p != "nr_lagged_dep"]
        Xi = _build_phi_X(df, idx_params)
        vv = Xi @ native[idx_params].values.astype(float)
        return np.clip(np.interp(vv, res.si_vgrid, res.si_ggrid), 0.0, 1.0)
    return np.clip(link_cdf(index, link), 0.0, 1.0)


def _shift(native, means):
    """sum_k theta_k * S_bar_k -- the amount the index would drop if the states
    were centred and the coefficients left alone."""
    tot = 0.0
    for p, c in native.items():
        p = str(p)
        if not p.startswith("interaction_"):
            continue
        sv = p[len("interaction_"):]
        if sv in means:
            tot += float(c) * float(means[sv])
    return tot


def _translate(res, means):
    """The reparametrisation, applied by hand to a fitted result.

    Two shapes exist.  Where the index carries a constant (E1-E6, param
    'nr_lagged_dep'), the shift goes into that constant.  Where it does not
    (E7/E8: the level is absorbed into the monotone link G), the shift goes into
    the link's own grid, which is stored in the native-index frame."""
    native = _native(res).copy()
    s = _shift(native, means)
    grid = None
    if "nr_lagged_dep" in native.index:
        native["nr_lagged_dep"] = float(native["nr_lagged_dep"]) + s
    else:
        grid = np.asarray(res.si_vgrid, float) - s
    return native, grid, s


class _Shim:
    """A results object carrying a shifted link grid, for the oracle only."""
    def __init__(self, res, grid):
        self._r, self.si_vgrid = res, grid

    def __getattr__(self, k):
        return getattr(self._r, k)


def _load(est):
    p = DEMAND_PREP / f"est{est}" / "estimation_results.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


# ==============================================================================
# TIER 0 -- the oracle
# ==============================================================================
def tier0(estimators, spec_filter):
    tf = _st.load_transform()
    means = tf.mean_scaled
    print(f"transform  : {_st.json_path()}")
    print(f"version    : {tf.version}   n={tf.n_rows:,}   built {tf.built_at}")
    print("means (estimation units): " +
          ", ".join(f"{k}={v:.6g}" for k, v in means.items()))

    print("\nbuilding the pooled frame once (uncentred), then a centred copy ...")
    from estimation_2_sleep import build_pooled_data
    df_unc = build_pooled_data(time_block=True, center=False)
    df_cen = df_unc.copy()
    tf.center(df_cen)

    # the centring itself must be exactly the advertised subtraction
    for c, m in means.items():
        if c in df_unc.columns:
            d = np.nanmax(np.abs((df_unc[c].values - m) - df_cen[c].values))
            assert d < 1e-12, f"{c}: centring is not a clean subtraction (max dev {d:.3g})"
    print(f"  frame ok: {len(df_unc):,} rows, subtraction exact to 1e-12")

    rows, worst, fails = [], 0.0, 0
    for est in estimators:
        rd = _load(est)
        if rd is None:
            print(f"\n[E{est}] no estimation_results.pkl -- skipped")
            continue
        print(f"\n[E{est}]")
        for spec, entry in sorted(rd.items()):
            if spec_filter and spec_filter not in spec:
                continue
            res = entry.get("second_stage") if isinstance(entry, dict) else None
            if res is None:
                continue
            native, grid, s = _translate(res, means)
            link = _link(res)
            phi_old = _phi(df_unc, _native(res), link, res)
            phi_new = _phi(df_cen, native, link,
                           _Shim(res, grid) if grid is not None else res)
            d = float(np.nanmax(np.abs(phi_old - phi_new)))
            ok = d <= TOL_ORACLE
            worst = max(worst, d)
            fails += (not ok)

            nat = _native(res)
            t0_old = float(nat["nr_lagged_dep"]) if "nr_lagged_dep" in nat.index else np.nan
            t0_new = float(native["nr_lagged_dep"]) if "nr_lagged_dep" in native.index else np.nan
            rows.append(dict(est=est, spec=spec, link=link, shift=s,
                             theta0_old=t0_old, theta0_new=t0_new,
                             mean_phi=float(np.nanmean(phi_old)), max_dphi=d,
                             ok=ok))
            tag = "OK " if ok else "FAIL"
            c_txt = (f"theta0 {t0_old:+.4f} -> {t0_new:+.4f}"
                     if np.isfinite(t0_old) else f"grid shift {-s:+.4f} (no constant)")
            print(f"  {tag} {spec:<34s} {link:<8s} {c_txt}   "
                  f"mean phi {np.nanmean(phi_old):.4f}   max|dphi| {d:.3e}")

    out = DEMAND_PREP / "state_centering_tier0.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(f"\nTIER 0 {'PASS' if fails == 0 else 'FAIL'}: "
          f"{len(rows) - fails}/{len(rows)} specs invariant, worst max|dphi| = {worst:.3e} "
          f"(tol {TOL_ORACLE:.0e})")
    return fails


# ==============================================================================
# TIER 1/2 -- the diff gate against a pre-change backup
# ==============================================================================
def _phi_frame(path):
    """phi_mt_* columns keyed on the panel identifiers."""
    keys = ["CodConglomeradoPrudencial", "mca_code", "deposit_type", "year", "quarter"]
    head = pd.read_csv(path, nrows=0)
    cols = [c for c in head.columns if c.startswith("phi_mt_")]
    use = [k for k in keys if k in head.columns] + cols
    df = pd.read_csv(path, usecols=use, dtype={"mca_code": str}, low_memory=False)
    return df.set_index([k for k in keys if k in df.columns]), cols


def gate(estimators, backup):
    backup = Path(backup)
    if not backup.exists():
        sys.exit(f"backup dir not found: {backup}")
    tf = _st.load_transform()
    means = tf.mean_scaled
    fails = 0

    for est in estimators:
        new_p = DEMAND_PREP / f"est{est}"
        old_p = backup / f"est{est}"
        if not (old_p / "estimation_results.pkl").exists():
            print(f"\n[E{est}] not in backup -- skipped")
            continue
        print(f"\n[E{est}]  tol max|dphi| = {TOL_PHI[est]:.0e}")

        # --- theta reconciliation + AME invariance -----------------------------
        with open(old_p / "estimation_results.pkl", "rb") as fh:
            rd_old = pickle.load(fh)
        rd_new = _load(est)
        for spec in sorted(set(rd_old) & set(rd_new or {})):
            r_o = rd_old[spec].get("second_stage")
            r_n = rd_new[spec].get("second_stage")
            if r_o is None or r_n is None:
                continue
            pred, _, s = _translate(r_o, means)
            nat_n = _native(r_n)
            shared = [p for p in pred.index if p in nat_n.index]
            slope = [p for p in shared if p != "nr_lagged_dep"]
            d_slope = max((abs(float(pred[p]) - float(nat_n[p])) for p in slope), default=0.0)
            d_const = (abs(float(pred["nr_lagged_dep"]) - float(nat_n["nr_lagged_dep"]))
                       if "nr_lagged_dep" in shared else np.nan)
            # AMEs: continuous scale 1:1, and the pix dummy must be EXACT
            a_o, a_n = getattr(r_o, "params", None), getattr(r_n, "params", None)
            d_pix = np.nan
            if a_o is not None and a_n is not None and "interaction_pix_exists" in a_o.index \
                    and "interaction_pix_exists" in a_n.index:
                d_pix = abs(float(a_o["interaction_pix_exists"]) - float(a_n["interaction_pix_exists"]))
            print(f"  {spec:<34s} shift {s:+.4f} | max|d slope| {d_slope:.3e} | "
                  f"|d const - shift| {d_const:.3e} | |d AME pix| {d_pix:.3e}")

        # --- phi identity ------------------------------------------------------
        for fn in ("market_panel_phis.csv", "national_phi_t.csv"):
            fo, fn_ = old_p / fn, new_p / fn
            if not (fo.exists() and fn_.exists()):
                continue
            if fn == "national_phi_t.csv":
                a = pd.read_csv(fo).set_index("year_quarter")
                b = pd.read_csv(fn_).set_index("year_quarter")
                cols = [c for c in a.columns if c.startswith("phi_t_") and c in b.columns]
            else:
                a, cols_a = _phi_frame(fo)
                b, cols_b = _phi_frame(fn_)
                cols = [c for c in cols_a if c in cols_b]
            if len(a) != len(b):
                print(f"  FAIL {fn}: row count {len(a):,} -> {len(b):,} "
                      "(the transform MOVED THE SAMPLE -- no tolerance applies)")
                fails += 1
                continue
            j = a[cols].join(b[cols], how="inner", lsuffix="_o", rsuffix="_n")
            for c in cols:
                d = float(np.nanmax(np.abs(j[f"{c}_o"].values - j[f"{c}_n"].values)))
                ok = d <= TOL_PHI[est]
                fails += (not ok)
                print(f"  {'OK ' if ok else 'FAIL'} {fn}:{c}  max|dphi| {d:.3e}")

    print(f"\nGATE {'PASS' if fails == 0 else 'FAIL'} ({fails} failing comparisons)")
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier0", action="store_true", help="oracle: no re-run needed")
    ap.add_argument("--gate", metavar="BACKUP_DIR", help="diff gate vs a pre-change backup")
    ap.add_argument("--est", default="1-8", help="estimators, e.g. '2' or '1-8' (default all)")
    ap.add_argument("--spec", default="", help="substring filter, e.g. 'IV_HausmanFull x Tech'")
    a = ap.parse_args()

    if "-" in a.est:
        lo, hi = a.est.split("-")
        ests = list(range(int(lo), int(hi) + 1))
    else:
        ests = [int(x) for x in a.est.split(",")]

    rc = 0
    if a.tier0:
        rc |= tier0(ests, a.spec)
    if a.gate:
        rc |= gate(ests, a.gate)
    if not a.tier0 and not a.gate:
        ap.error("choose --tier0 and/or --gate BACKUP_DIR")
    raise SystemExit(1 if rc else 0)
