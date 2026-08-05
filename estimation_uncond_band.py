"""
estimation_uncond_band.py
================================================================================
UNCONDITIONAL (direction + link) national phi_t bands for the single-index
sleepiness estimators E5/E6 at spec 12, from STORED fits -- no re-estimation.

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
    python estimation_uncond_band.py --est 5 --loss both [--B 400] [--attach]
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
from utils.sleep_links import unconditional_phi_t_band

SPEC = "IV_HausmanFull x Tech"          # spec 12; the only cell this driver serves
FE_TIME_COL = "time_id"
LOSS_OF = {"robust": "cauchy", "ls": "linear"}
KEY_OF = {"robust": "second_stage", "ls": "second_stage_ls"}


def _prep_frame(time_block):
    """Rebuild the estimation frame for spec 12 exactly as estimation_sleep_common._exec_spec:
    pooled panel -> first stage with the spec-12 instruments and active exogenous controls."""
    from estimation_2_sleep import (build_pooled_data, define_specifications,
                                    run_pooled_first_stage)
    df = build_pooled_data(time_block=time_block)
    _, iv_specs, state_blocks = define_specifications(time_block=time_block)
    s_cols = state_blocks["Tech"]
    iv_act = [c for c in iv_specs["IV_HausmanFull"]
              if c in df.columns and df[c].notnull().sum() > 0]
    exog_act = [c for c in s_cols if c in df.columns and df[c].notnull().sum() > 0]
    df_t, _ = run_pooled_first_stage(df, iv_act, exog_act)
    return df_t, s_cols


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

    df_t, s_cols = _prep_frame(time_block=(est == 6))
    out = unconditional_phi_t_band(df_t, s_cols, True, si_res, LOSS_OF[loss_lbl],
                                   degree=3, fe_time_col=FE_TIME_COL, B=B)

    # ---- BLOCKING GATE: link-only variant must reproduce the stored conditional band -------
    # Compared on the LEGACY market key (CODMUN_IBGE), because that is what every stored band
    # was built on. The reported bands use MCA (V_Main.tex:182); checking the gate on the
    # legacy key keeps "is the rebuild faithful?" separate from "which market is correct?".
    gate = dict(checked=False, max_lo=np.nan, max_hi=np.nan, ok=None, key="CODMUN_IBGE")
    _gate_band = out.get("band_link_only_legacy", out["band_link_only"])
    if stored_band is not None and len(stored_band) == len(_gate_band):
        a = _gate_band.sort_values("time_id").reset_index(drop=True)
        s = stored_band.sort_values("time_id").reset_index(drop=True)
        gate.update(checked=True,
                    max_lo=float(np.max(np.abs(a["lo"].values - s["lo"].values))),
                    max_hi=float(np.max(np.abs(a["hi"].values - s["hi"].values))))
        gate["ok"] = max(gate["max_lo"], gate["max_hi"]) < 1e-8
        msg = "REPRODUCED" if gate["ok"] else "MISMATCH -- do not trust this band"
        print(f"  [gate] link-only vs stored band: max|dlo|={gate['max_lo']:.2e} "
              f"max|dhi|={gate['max_hi']:.2e}  {msg}")
        if not gate["ok"]:
            print("  [gate] NOTE: a mismatch here usually means seed/B/scheme differ from the "
                  "stored band (B must be 400, seed 20240624, webb) or the frame drifted.")
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

    rout = root / "Rout"
    rout.mkdir(parents=True, exist_ok=True)
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
    p.add_argument("--est", type=int, required=True, choices=(5, 6))
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
