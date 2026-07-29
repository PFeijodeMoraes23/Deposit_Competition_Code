"""diag_index_identification.py -- is the sleepiness index DIRECTION identified?

Author: Pedro Feijó de Moraes

Motivated by 2026-07-29: E7 spec 12 was found sitting at a corner solution (0.988 of a
unit-norm theta on `risk_free_qoq_lag`) whose monotone link saturated into a 1.5pp band of
phi, collapsing every AME by ~700x -- at an R2 of 0.95230 against 0.95250 for a completely
different, balanced direction. Two directions that disagree economically but agree to 4
decimal places in fit are not "one right and one wrong optimum": they are evidence that the
DATA does not pin down the direction, and that the optimizer is choosing it.

This script quantifies that directly. It refits each estimator from several starting points
and reports the SPREAD in attained fit. Reading:

    spread ~ 0        -> theta is weakly/set-identified. Report it as such; a better
                         optimizer buys stability, not identification.
    spread meaningful -> the objective does discriminate, and the best start wins.

Scope by routine (checked against the code, 2026-07-29):
  E1/E2  closed-form OLS/2SLS -- no starting values, nothing to diagnose.
  E3/E4  fit_nlls_link: ONE start from zeros. Probed here via the `init` hook.
  E5/E6  fit_single_index does NOT re-optimise theta -- it inherits E3/E4's direction and
         only fits the link. So they are not independent checks of theta; whatever E3/E4
         lands on propagates. Diagnosing E3/E4 diagnoses these.
  E7/E8  fit_joint_single_index now scores candidate directions on the full-sample
         objective and prints the spread itself; this script adds random starts on top.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import numpy as np

from estimation_2_sleep import (build_pooled_data, define_specifications,
                                run_pooled_first_stage)
from utils.sleep_links import fit_nlls_link

SPEC = "IV_HausmanFull x Tech"   # spec 12


def main(n_random: int, seed: int, time_block: bool) -> int:
    rng = np.random.default_rng(seed)
    print(f"=== index-direction identification, spec 12 "
          f"({'with' if time_block else 'without'} time block) ===")
    df = build_pooled_data(time_block=time_block)
    _, ivs, blocks = define_specifications(time_block=time_block)
    s_cols = blocks["Tech"]
    df, _ = run_pooled_first_stage(df, ivs["IV_HausmanFull"],
                                   [c for c in s_cols if c != "constant"])
    K = len(s_cols)
    G = 1                                    # v_hat_x_lagged_dep (has_cf=True)
    print(f"  n={len(df):,}  K={K} state cols + {G} CF term\n")

    # ---- E3/E4 family: NLLS logit, normally ONE start from zeros -------------------
    starts = [("zeros (production)", np.zeros(K + G)),
              ("ones", np.ones(K + G) / np.sqrt(K + G))]
    for i in range(n_random):
        r = rng.standard_normal(K + G)
        starts.append((f"random{i+1}", r / np.linalg.norm(r)))

    rows = []
    for nm, s0 in starts:
        res = fit_nlls_link(df, s_cols, has_cf=True, link="logit", loss="cauchy",
                            fe_time_col="time_id", bootstrap=False, init=s0)
        if res is None:
            print(f"  {nm:<20s}  FAILED")
            continue
        nat = res.params_native
        th = np.array([float(nat[p]) for p in nat.index if not str(p).startswith("v_hat")])
        thn = th / (np.linalg.norm(th) + 1e-12)
        rows.append((nm, float(res.rsquared), thn, nat))
        print(f"  {nm:<20s}  R2={res.rsquared:.8f}")

    if len(rows) < 2:
        print("\n  too few successful fits to judge identification")
        return 1

    r2 = np.array([r[1] for r in rows])
    spread = float(r2.max() - r2.min())
    print(f"\n  R2 spread across {len(rows)} starts: {spread:.3e}")

    # how different are the DIRECTIONS these starts converge to?
    base = rows[0][2]
    print(f"\n  cosine similarity of each direction to the production (zeros) start:")
    for nm, _, thn, _ in rows:
        cos = float(np.dot(base, thn))
        print(f"    {nm:<20s} cos={cos:+.4f}"
              + ("   <-- materially different direction" if abs(cos) < 0.95 else ""))

    print()
    if spread < 1e-6:
        print("  VERDICT: WEAKLY IDENTIFIED. Starts that converge to different directions")
        print("  attain the same fit to <1e-6. The reported theta is an optimizer artifact;")
        print("  disclose it as set-identified rather than presenting a point estimate.")
    elif spread < 1e-4:
        print("  VERDICT: FRAGILE. The objective discriminates only weakly between")
        print("  directions; report the multistart spread alongside the point estimate.")
    else:
        print("  VERDICT: IDENTIFIED. The objective clearly prefers one direction;")
        print("  ensure production uses enough starts to find it.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-random", type=int, default=3, help="random unit-vector starts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-block", action="store_true", help="E4/E6/E8 variant")
    a = ap.parse_args()
    raise SystemExit(main(a.n_random, a.seed, a.time_block))
