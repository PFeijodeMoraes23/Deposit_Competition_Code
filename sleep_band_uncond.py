"""
sleep_band_uncond.py
================================================================================
UNCONDITIONAL (direction + link) national phi_t bands for the single-index
sleepiness estimators E3/E4 at spec 12, from STORED fits -- no re-estimation.

Why: the bands stored on the estimator pickles condition on the fitted index
direction theta-hat (they perturb only the sieve-link coefficients), and the
2026-08 multiplicity diagnostics show theta-hat is weakly identified. This
driver produces the joint band by score-perturbing theta with its reconstructed
cluster influence functions and RE-PROFILING the link exactly at each draw --
see utils/sleep_links.unconditional_phi_t_band for the construction and the
no-double-counting argument.

Reads the estimator pickle from utils.paths.demand_prep_root() -- set
SLEEP_OUT_ROOT to run against the sandbox (the intended mode while promotion
is pending). Writes standalone result pickles to <root>/Rout/
    ts_link_band_est{N}_uncond_{robust,ls}.pkl
and never touches the estimator pickle unless --attach is passed (off by
default; makes a timestamped .bak first).

Usage:
    python sleep_band_uncond.py --est 3 --loss both [--B 400] [--attach]
"""
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import pickle
import shutil
import time
from datetime import datetime

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd

from utils import paths as _paths_mod
from utils import routines as _routines
from utils.sleep_links import unconditional_phi_t_band

SPEC = _routines.SPEC12                 # spec 12; the only cell this driver serves
FE_TIME_COL = _routines.FE_TIME_COL
LOSS_OF = {"robust": "cauchy", "ls": "linear"}
KEY_OF = {"robust": "second_stage", "ls": "second_stage_ls"}


def _prep_frame(time_block):
    """Rebuild the estimation frame for spec 12 exactly as estimation_sleep_common._exec_spec:
    pooled panel -> first stage with the spec-12 instruments and active exogenous controls."""
    from sleep_est_e2 import (build_pooled_data, define_specifications,
                                    run_pooled_first_stage)
    df = build_pooled_data(time_block=time_block)
    _, iv_specs, state_blocks = define_specifications(time_block=time_block)
    s_cols = state_blocks["Tech"]
    iv_act = [c for c in iv_specs["IV_HausmanFull"]
              if c in df.columns and df[c].notnull().sum() > 0]
    exog_act = [c for c in s_cols if c in df.columns and df[c].notnull().sum() > 0]
    df_t, _ = run_pooled_first_stage(df, iv_act, exog_act)
    return df_t, s_cols


def link_only_gate(stored_band, out):
    """BLOCKING GATE: the link-only rebuild must reproduce the stored conditional band.

    Two tiers, because the two sides do not always hold the same objects:

    (a) DRAW-MATRIX -- when the stored band carries its draw matrix (`attrs["draws"]`,
        attached by `_phi_t_band` since 2026-08-11), compare draws directly: max|delta| over
        the B x nt values against 1e-8. Draws are Lipschitz in the inputs (the projection is
        non-expansive, the aggregation linear), so this is a true reproduction criterion, and
        a genuine seed/scheme/frame drift moves EVERY draw by orders of magnitude more. Key
        ambiguity does not arise: any pickle new enough to carry draws is mca-keyed.

    (b) EDGE + LOCAL-SPAN TOLERANCE -- bands stored without draws compare on their lo/hi edges.
        An interpolated empirical quantile moves by up to one local order-statistic gap under
        sub-tolerance draw noise (measured 2026-08-11: the six 2020Q1-2021Q3 knife edges on
        the robust cells, delta/gap 0.33-0.99, all other quarters ~1e-13), so the per-quarter
        tolerance is max(1e-8, the local order-statistic span around the corresponding
        quantile of the REBUILT draws). Verdicts are three-way: REPRODUCED (inside the 1e-8 floor everywhere),
        WITHIN-SPAN (knife edges only -- ok, reported distinctly), MISMATCH.
        WHICH market key the stored band was aggregated on depends on WHEN it was estimated
        (pre-2026-08-05 bands are CODMUN_IBGE-keyed), so both variants are compared and the
        gate takes the best; the gap tolerances come from the mca draw matrix in either case
        (the aggregation changes levels, not the scale of inter-draw spacing).
    """
    gate = dict(checked=False, tier=None, max_lo=np.nan, max_hi=np.nan, ok=None, key=None,
                n_knife=0)
    if stored_band is None:
        return gate
    s = stored_band.sort_values("time_id").reset_index(drop=True)

    stored_draws = stored_band.attrs.get("draws") if hasattr(stored_band, "attrs") else None
    rebuilt_draws = (out.get("draws") or {}).get("link_only")
    if stored_draws is not None and rebuilt_draws is not None \
            and np.shape(stored_draws) == np.shape(rebuilt_draws):
        d_max = float(np.max(np.abs(np.asarray(stored_draws) - rebuilt_draws)))
        gate.update(checked=True, tier="draws", key="mca_code", max_lo=d_max, max_hi=d_max,
                    ok=d_max < 1e-8)
        msg = "REPRODUCED" if gate["ok"] else "MISMATCH -- do not trust this band"
        print(f"  [gate] link-only draws vs stored draws: max|d|={d_max:.2e}  {msg}")
        if not gate["ok"]:
            print("  [gate] NOTE: a genuine seed/B/scheme/frame drift moves every draw "
                  "(need B=400, seed 20240624, webb).")
        return gate

    B_n = rebuilt_draws.shape[0] if rebuilt_draws is not None else 0
    tol_lo = tol_hi = None
    if B_n:
        # Tolerance = the LOCAL ORDER-STATISTIC SPAN, one reordering either side of the
        # interpolation interval: the edge at fractional index q(B-1) lives in
        # [srt[i], srt[i+1]], and sub-tolerance draw perturbations can carry it across the
        # adjacent intervals, so the reachable set is bounded by srt[i+2]-srt[i-1]. Measured
        # E3/robust needs 0.33-0.99 of one gap, E4/robust needs up to 2.4 gaps --
        # every quarter inside the span on both.
        srt = np.sort(rebuilt_draws, axis=0)
        i_lo = int(np.floor(0.025 * (B_n - 1)))
        i_hi = int(np.floor(0.975 * (B_n - 1)))
        tol_lo = np.maximum(1e-8, srt[min(i_lo + 2, B_n - 1)] - srt[max(i_lo - 1, 0)])
        tol_hi = np.maximum(1e-8, srt[min(i_hi + 2, B_n - 1)] - srt[max(i_hi - 1, 0)])
    # Key selection is VERDICT-aware, not magnitude-aware: the wrong key's deltas (the
    # aggregation gap, spread over every quarter) can have a smaller max than the right key's
    # single knife edge, so ranking by raw max picks the key that fails. Rank by (reproduced,
    # within-gap, max delta) instead.
    best = None
    for key_name, var in (("mca_code", "band_link_only"),
                          ("CODMUN_IBGE", "band_link_only_legacy")):
        cand = out.get(var)
        if cand is None or len(cand) != len(s):
            continue
        a = cand.sort_values("time_id").reset_index(drop=True)
        d_lo = np.abs(a["lo"].values - s["lo"].values)
        d_hi = np.abs(a["hi"].values - s["hi"].values)
        floor_ok = max(d_lo.max(), d_hi.max()) < 1e-8
        if tol_lo is not None:
            ok = bool(np.all(d_lo <= tol_lo) and np.all(d_hi <= tol_hi))
            n_knife = int((((d_lo > 1e-8) & (d_lo <= tol_lo))
                           | ((d_hi > 1e-8) & (d_hi <= tol_hi))).sum())
        else:
            ok, n_knife = floor_ok, 0
        rank = (not ok, not floor_ok, float(max(d_lo.max(), d_hi.max())))
        if best is None or rank < best[0]:
            best = (rank, key_name, d_lo, d_hi, floor_ok, ok, n_knife)
    if best is None:
        return gate
    _, key_name, d_lo, d_hi, floor_ok, ok, n_knife = best
    gate.update(checked=True, tier="edges", key=key_name,
                max_lo=float(d_lo.max()), max_hi=float(d_hi.max()), ok=ok, n_knife=n_knife)
    msg = ("REPRODUCED" if floor_ok else
           f"WITHIN LOCAL ORDER-STAT SPAN ({n_knife} knife-edge quarter(s))" if ok else
           "MISMATCH -- do not trust this band")
    print(f"  [gate] link-only[{key_name}] vs stored band: max|dlo|={d_lo.max():.2e} "
          f"max|dhi|={d_hi.max():.2e}  {msg}")
    if not ok:
        print("  [gate] NOTE: BOTH key variants mismatch beyond the local order-stat span -> "
              "seed/B/scheme differ from the stored band (need B=400, seed 20240624, webb) "
              "or the frame drifted.")
    return gate


def run_cell(est, loss_lbl, B, attach):
    t0 = time.time()
    root = _paths_mod.demand_prep_root()
    pkl_path = root / f"est{est}" / "estimation_results.pkl"
    print(f"\n=== est{est} / {loss_lbl} | root={root} ===")
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    si_res = d[SPEC].get(KEY_OF[loss_lbl])
    if si_res is None:
        print(f"  no {KEY_OF[loss_lbl]} on {SPEC}; skipped")
        return None
    stored_band = getattr(si_res, "phi_t_boot", None)

    df_t, s_cols = _prep_frame(time_block=(est == 4))
    out = unconditional_phi_t_band(df_t, s_cols, True, si_res, LOSS_OF[loss_lbl],
                                   degree=3, fe_time_col=FE_TIME_COL, B=B)

    gate = link_only_gate(stored_band, out)
    out["meta"]["link_only_gate"] = gate
    out["meta"].update(est=est, loss_label=loss_lbl, spec=SPEC,
                       runtime_s=time.time() - t0,
                       created=datetime.now().isoformat(timespec="seconds"))

    # ---- summary ----------------------------------------------------------------------------
    rng_pp = 100 * (out["band_total"]["phi_t"].max() - out["band_total"]["phi_t"].min())
    for k, lbl in (("band_total", "TOTAL"), ("band_theta_only", "theta"),
                   ("band_link_only", "link")):
        b = out[k]
        w = 100 * float((b["hi"] - b["lo"]).mean())
        wbc = 100 * float((b["hi_bc"] - b["lo_bc"]).mean())
        outn = int(((b["phi_t"] < b["lo"]) | (b["phi_t"] > b["hi"])).sum())
        print(f"  {lbl:6s} raw {w:6.2f}pp  BC {wbc:6.2f}pp  ratio(raw/range) "
              f"{w/max(rng_pp,1e-9):5.2f}  outside {outn}/{len(b)}")
    q = out["per_draw_diag"]["cos"].quantile([0.05, 0.5, 0.95])
    print(f"  theta-draw cosine 5/50/95%: {q.iloc[0]:.4f} / {q.iloc[1]:.4f} / {q.iloc[2]:.4f}"
          f"   (observed refit drift was 0.918-0.970)")
    print(f"  foc_norm={out['meta']['foc_norm']:.3e}  refit failures={out['meta']['n_fail']}"
          f"  runtime {out['meta']['runtime_s']/60:.1f} min")

    rout = _paths_mod.rout_dir()
    fp = rout / f"ts_link_band_est{est}_uncond_{loss_lbl}.pkl"
    with open(fp, "wb") as fh:
        pickle.dump(out, fh)
    print(f"  saved -> {fp}")

    if attach:
        bak = pkl_path.with_suffix(f".pkl.bak_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(pkl_path, bak)
        si_res.phi_t_boot_uncond = out["band_total"]
        d[SPEC][KEY_OF[loss_lbl]] = si_res
        with open(pkl_path, "wb") as fh:
            pickle.dump(d, fh)
        print(f"  attached phi_t_boot_uncond (backup: {bak.name})")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    p.add_argument("--est", type=int, required=True, choices=tuple(_routines.LINK_ESTS))
    p.add_argument("--loss", choices=("robust", "ls", "both"), default="both")
    p.add_argument("--B", type=int, default=400)
    p.add_argument("--attach", action="store_true")
    a = p.parse_args()
    losses = ("robust", "ls") if a.loss == "both" else (a.loss,)
    for ll in losses:
        run_cell(a.est, ll, a.B, a.attach)


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
