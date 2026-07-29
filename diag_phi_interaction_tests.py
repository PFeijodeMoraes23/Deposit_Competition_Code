"""diag_phi_interaction_tests.py -- restrictions the sleepiness model imposes on the carry.

Author: Pedro Feijo de Moraes

phi(S_mt) has no bank and no deposit-type index: within a market-quarter every product of
every bank must share one carry coefficient. Both arms test that restriction; a rejection
means the "retention" parameter tracks product/bank persistence, not consumer attention
(or that phi is heterogeneous -- either way the common-phi object CF1/CF4 simulate with
is misspecified). See identification_notes.md in Drafts/Deposit Competition (next to
V_Main.tex).

Arms (--arm):
  types           D5: type-specific carries phi_k, k in {1,2,4,5}. Savings (1,2) are
                  payroll-linked, CDB (4) rolls over contractually, prepaid (5) is
                  transactional -- mechanical persistence differs sharply by instrument
                  while the model says phi cannot. Difference coding gives WCB inference
                  on phi_k - phi_base directly.
  attractiveness  D3: predetermined (2016) within-market deposit-share rank x carry.
                  Under the model the carry cannot vary with bank attractiveness; a
                  loading means persistently attractive banks retain MORE than phi --
                  the woke-and-stayed channel by name.

Sample note: spec 12's CF term restricts it to k=4,5 (v_hat NaN elsewhere), so the types
arm contrasts 4 vs 5 there; the full k set runs under OLS x Tech.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse

import numpy as np
import pandas as pd

from diag_phi_augmented_tests import (load_sleep_frame, run_augmented, OUT_DIR)


def arm_types():
    """D5: type-specific carry coefficients."""
    print("\n=== D5: deposit-type contrast of the carry coefficient ===")
    df, s_cols = load_sleep_frame()
    rows = []
    for tag, has_cf, ks, base_k in (("OLSxTech(all k)", False, [1, 2, 4, 5], 1),
                                    ("spec12(k=4,5)", True, [4, 5], 4)):
        d0 = df.copy()
        extra = []
        for k in ks:
            if k == base_k:
                continue
            nm = f"Zdiff_k{k}"
            d0[nm] = (d0["deposit_type"] == k).astype(float) * d0["nr_lagged_dep"]
            extra.append(nm)
        # difference coding: nr_lagged_dep = phi(base type, avg market); Zdiff_k = phi_k - phi_base
        res_b, res_a, d = run_augmented(d0, s_cols, extra, has_cf)
        phi_base = float(res_a.params["nr_lagged_dep"])
        print(f"\n  [{tag}] n={len(d):,}   phi-hat_k at the average market "
              f"(base k={base_k}):")
        print(f"    k={base_k}: {phi_base:.4f}")
        rows.append({"spec": tag, "param": f"phi_k{base_k}", "coef": phi_base})
        for k in ks:
            if k == base_k:
                continue
            nm = f"Zdiff_k{k}"
            diff = float(res_a.params[nm])
            p = float(res_a.pvalues[nm])
            print(f"    k={k}: {phi_base + diff:.4f}   (phi_k - phi_{base_k} = {diff:+.4f}, "
                  f"WCB p={p:.4f})")
            rows.append({"spec": tag, "param": nm, "coef": diff, "p_wcb": p,
                         "phi_level": phi_base + diff})
    print("\n  VERDICT: the model requires equal carries across k within a market-quarter.")
    print("  Spread in phi-hat_k comparable to the awake margin (1-phi) rejects the common-")
    print("  phi restriction; a CDB (k=4) excess specifically indicates contractual-rollover")
    print("  persistence being read as sleepiness. (Composition heterogeneity is the")
    print("  benign alternative -- but it equally breaks CF1/CF4's common-phi simulation.)")
    return rows


def arm_attractiveness():
    """D3: predetermined attractiveness x carry placebo."""
    print("\n=== D3: attractiveness placebo (2016 within-market share rank x carry) ===")
    df, s_cols = load_sleep_frame()

    base = df[df["year"] == 2016].copy()
    base["_rank"] = base.groupby(["mca_code", "deposit_type", "time_id"],
                                 observed=True)["deposit_balance"].rank(pct=True)
    rank = base.groupby("entity_id")["_rank"].mean().rename("attr_rank")
    df = df.merge(rank, on="entity_id", how="left")
    df["attr_rank_c"] = df["attr_rank"] - df["attr_rank"].mean()
    df["rank_x_Z"] = df["attr_rank_c"] * df["nr_lagged_dep"]
    print(f"  entities with a 2016 rank: {rank.notna().sum():,} "
          f"(rows covered: {df['attr_rank'].notna().mean():.1%})")

    rows = []
    for tag, has_cf in (("spec12(k=4,5)", True), ("OLSxTech(all k)", False)):
        res_b, res_a, d = run_augmented(df, s_cols, ["rank_x_Z"], has_cf)
        co = float(res_a.params["rank_x_Z"])
        p = float(res_a.pvalues["rank_x_Z"])
        phi0 = float(res_a.params["nr_lagged_dep"])
        iqr = float(d["attr_rank"].quantile(.75) - d["attr_rank"].quantile(.25))
        print(f"\n  [{tag}] n={len(d):,}  phi-hat(avg market, mean-rank bank) = {phi0:.4f}")
        print(f"    rank x carry = {co:+.4f}  (WCB p={p:.4f})")
        print(f"    implied phi gap, p75-vs-p25 attractiveness: {co * iqr:+.4f} "
              f"(awake margin 1-phi = {1 - phi0:.4f})")
        rows.append({"spec": tag, "param": "rank_x_Z", "coef": co, "p_wcb": p,
                     "phi_gap_iqr": co * iqr, "awake_margin": 1 - phi0})
    print("\n  VERDICT: under the model the carry is bank-invariant. A positive loading")
    print("  with a phi gap comparable to (1-phi) says retention tracks persistent bank")
    print("  attractiveness -- consumers who woke and stayed being booked as asleep.")
    return rows


ARMS = {"types": arm_types, "attractiveness": arm_attractiveness}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=list(ARMS) + ["all"], required=True)
    a = ap.parse_args()
    all_rows = []
    for arm in (list(ARMS) if a.arm == "all" else [a.arm]):
        all_rows += ARMS[arm]()
    out = OUT_DIR / f"d_interactions_{a.arm}.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nresults -> {out}")
