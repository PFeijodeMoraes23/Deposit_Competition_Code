"""diag_phi_interaction_tests.py -- restrictions the sleepiness model imposes on the carry.

Author: Pedro Feijo de Moraes

phi(S_mt) has no bank and no deposit-type index: within a market-quarter every product of
every bank must share one carry coefficient. Both arms test that restriction; a rejection
means the "retention" parameter tracks product/bank persistence, not consumer attention
(or that phi is heterogeneous -- either way the common-phi object CF1/CF4 simulate with
is misspecified). See identification_notes.md in Drafts/Deposit Competition (next to
V_Main.tex).

Arms (--arm):
  types           D4b: type-specific carries phi_k, k in {1,2,4,5}. Savings (1,2) are
                  payroll-linked, CDB (4) rolls over contractually, prepaid (5) is
                  transactional -- mechanical persistence differs sharply by instrument
                  while the model says phi cannot. Difference coding gives WCB inference
                  on phi_k - phi_base directly.
  attractiveness  D4a: predetermined (2016) within-market deposit-share rank x carry.
                  Under the model the carry cannot vary with bank attractiveness; a
                  loading means persistently attractive banks retain MORE than phi --
                  the woke-and-stayed channel by name.
                  Reported TWICE: the production wild cluster bootstrap, and a Fisher
                  RANDOMIZATION p-value that needs no variance estimate (the rank is a
                  predetermined label, so the sharp null licenses reassigning it across
                  conglomerates). At G* ~ 6 the WCB reference distribution is the weak
                  link, and D1 puts this arm's power at 0.06-0.11, so a "pass" read off
                  that reference distribution alone is not worth much. See
                  diag_phi_augmented_tests.cluster_permutation_test for the scheme and,
                  in particular, for what the randomization test does NOT deliver.

Sample note: spec 12's CF term restricts it to k=4,5 (v_hat NaN elsewhere), so the types
arm contrasts 4 vs 5 there; the full k set runs under OLS x Tech.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse

import numpy as np
import pandas as pd

from diag_phi_augmented_tests import (load_sleep_frame, run_augmented, OUT_DIR,
                                      cluster_permutation_test, report_permutation,
                                      design_cols)


def arm_types():
    """D4b: type-specific carry coefficients."""
    print("\n=== D4b: deposit-type contrast of the carry coefficient ===")
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
    """D4a: predetermined attractiveness x carry placebo."""
    print("\n=== D4a: attractiveness placebo (2016 within-market share rank x carry) ===")
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

        # ---- Fisher randomization (ADDITIVE -- the WCB lines above are unchanged).
        # attr_rank is a PREDETERMINED 2016 label, so under the sharp null "attractiveness
        # shifts the carry for no bank" the outcome is invariant to reassigning it across
        # conglomerates. That gives an exact p-value with NO variance estimate, which
        # matters here because the Monte Carlo (D1) puts this arm's power at 0.06-0.11 --
        # its WCB "pass" is a statement about the reference distribution as much as about
        # the data. See the header of cluster_permutation_test for what this does and,
        # more importantly, what it does not deliver.
        for _mode in ("collapse", "between"):
            pr = cluster_permutation_test(
                d, "rank_x_Z",
                lambda a, C: {"rank_x_Z": a * C["nr_lagged_dep"]},
                "attr_rank_c", design_cols(s_cols, has_cf), mode=_mode)
            rows.append({"spec": tag, "awake_margin": 1 - phi0, **report_permutation(
                pr, "rank_x_Z", co, p, wcb_se=float(res_a.bse["rank_x_Z"]), carry=phi0)})
    print("\n  VERDICT: under the model the carry is bank-invariant. A positive loading")
    print("  with a phi gap comparable to (1-phi) says retention tracks persistent bank")
    print("  attractiveness -- consumers who woke and stayed being booked as asleep.")
    _pm = [r for r in rows if "p_perm_coef" in r]
    if _pm:
        _agree = all((r["p_perm_headline"] <= 0.05) == (r["p_wcb"] <= 0.05) for r in _pm)
        _div = [r for r in _pm if r["perm_stats_diverge"]]
        print("  RANDOMIZATION: the variance-free Fisher test " +
              ("AGREES with the WCB at 5% on every spec and scheme" if _agree
               else "DISAGREES with the WCB at 5% somewhere") +
              " (studentised perm p = " +
              ", ".join(f"{r['spec']}/{r['perm_mode']}:{r['p_perm_headline']:.3f}"
                        for r in _pm) + ")." +
              (f" NOTE: {len(_div)} of {len(_pm)} configurations flip on the UNSTUDENTISED "
               "statistic -- see report_permutation." if _div else
               " The unstudentised statistic agrees everywhere too."))
        print("  It removes the few-cluster VARIANCE ESTIMATE from the verdict; it does NOT")
        print("  remove the few-cluster problem -- top-5 R2 = " +
              ", ".join(f"{r['spec']}/{r['perm_mode']}:{r['top_r2']:.2f}" for r in _pm))
        print("  of the permutation variance. And it tests the SHARP null, so a pass is not a")
        print("  bound on an average effect. Read it with the D1 power numbers, not instead of them.")
        # POWER, on this arm's OWN yardstick. attr_rank is a percentile in [0,1], so the
        # coefficient is "carry units per unit of rank" and the natural comparison is the
        # awake margin 1-phi: an attractiveness channel big enough to matter for the
        # sleepiness reading would have to move the carry by an appreciable fraction of the
        # mass the model says is awake. If the randomization MDE exceeds 1-phi outright,
        # the design cannot see even a violation the size of the ENTIRE awake margin --
        # which is a stronger negative statement than "power 0.06-0.11 against the MC
        # alternative", because it needs no alternative to be specified.
        for r in [x for x in _pm if x["perm_mode"] == "between"]:
            am, m, c = r["awake_margin"], r["perm_mde80"], abs(r["coef_entity_level"])
            print(f"  POWER [{r['spec']}]: randomization MDE(80%) = {m:.4f} per unit of rank"
                  f" = {m/am:.1f}x the awake margin (1-phi = {am:.4f}) and {m/max(c,1e-12):.1f}x"
                  f" the estimated |coef| = {c:.4f}.")
            print("    -> " + ("the design cannot detect a rank effect even as large as the "
                               "whole awake margin; this arm's 'pass' carries no information "
                               "about attractiveness-driven retention."
                               if m > am else
                               "the design can resolve effects smaller than the awake margin, "
                               "so the null is an informative bound."))
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
