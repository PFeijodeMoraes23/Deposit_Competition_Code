"""diag_phi_augmented_tests.py -- is phi separately identified from awake-flow persistence?

Author: Pedro Feijo de Moraes

The sleep estimators regress Dep on phi(S)*nr_lagged_dep + CF + two-way FE, dropping the
awake-flow term (1-phi)*M*s^Act of eq. (9-B) into the error. phi is therefore identified
only under the assumption that awake-inflow INNOVATIONS are serially uncorrelated within
entity after quarter FE and the CF term; "woke up and chose the same bank" is excluded by
assumption, not by any moment. This script measures that assumption directly with
augmented regressions on the production E2 kernel (see identification_notes.md in
Drafts/Deposit Competition, next to V_Main.tex).

Arms (--arm):
  identity    D0: verify Dep = phi*g*L + Dep_Act is an accounting IDENTITY on the demand
              parquet (=> the naive contemporaneous augmentation is tautological), measure
              the selection the max{0,.} & >1e-6 filters introduce, and confirm the sleep
              step and demand prep share one accrual rate.
  lagdepact   D2b: augment the second stage with Dep_Act_{t-1}. Under the maintained
              assumption its coefficient is 0 (its effect on Dep_t flows only through
              Dep_{t-1}, which nr_lagged_dep already carries). A nonzero coefficient and
              the induced Delta-phi measure the awake-persistence confound.
  fittedshare D2a: augment with the characteristics-fitted awake inflow
              A_hat = (1-phi)*M*s_hat(X), with xi-hat EXCLUDED (xi-hat mechanically
              absorbs the accounting residual, so including it would be circular).
              A LOWER BOUND on the confound: only its observable component is used.
  spreadlevel D4: augment with the contemporaneous spread level. QUALITATIVE ONLY --
              collinear with the CF residual by construction and equally consistent
              with simultaneity; reported for completeness, carries no weight.
  pix         D6: event-study of the carry coefficient in 2-quarter bins around Pix
              launch (2020Q4), with a high/low pre-period connectivity split. A discrete
              break concentrated in connected markets supports the attention reading of
              the Pix component; this is an asymmetric test (passing does not validate
              the phi LEVEL).

Sample note: the production spec 12 (IV_HausmanFull x Tech) second stage is restricted to
deposit types 4,5 because v_hat is NaN elsewhere (estimation_2_sleep.py:299-301). Every
arm therefore reports BOTH spec 12 and OLS x Tech (full k in {1,2,4,5} sample).
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import statsmodels.api as sm

from estimation_2_sleep import (build_pooled_data, define_specifications,
                                run_pooled_first_stage, demean_variables_2way,
                                apply_imbalanced_cluster_correction)
from utils import paths as _paths

PARQUET = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "demand_2_spec_12.parquet"
BLP_RAW = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "BLP_RESULTS" / "cluster_raw"
OUT_DIR = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_PHI_SEPARATION"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEP_SCALE = 1e9   # sleep frame stores deposits in R$ bn; the demand parquet in raw R$

# Few-cluster constants. The panel has ~456 nominal conglomerates but the size distribution
# is extreme (CV ~ 8.5), so the Carter-Schnepel-Steigerwald effective count is G* ~ 6, i.e.
# nu = G*-1 ~ 5 degrees of freedom for the reference distribution.
#   MDE_MULT       t_.975,nu + t_.80,nu at nu=5.2  (the large-sample value is 2.80)
#   MDE_LO/HI      90% band for SE_hat/SE_true from chi2_nu -- the SE is uncertain by ~3x
#                  end to end, so a point MDE alone overstates what the design pins down.
MDE_MULT, MDE_LO, MDE_HI = 3.457, 0.676, 2.04
BLP_X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
              "log_total_assets_lag", "is_state_owned"]


def load_blp_theta1(est="E8", stage="extended", spec=12):
    """theta1 from the downloaded cluster results as {param_name: coef}, or None.
    NB: the BLP spread regressor is spread_ann/100 (annual pp) -- blp_1_logit.jl:219."""
    import json
    fp = BLP_RAW / f"blp_results_{est}_spec_{spec}_{stage}.json"
    if not fp.exists():
        return None
    with open(fp, encoding="utf-8") as fh:
        r = json.load(fh)
    if not r.get("converged", False):
        return None
    return dict(zip(r["param_names_theta1"], r["theta1"]))


# ==============================================================================
# SHARED MACHINERY
# ==============================================================================
def load_sleep_frame(time_block=False):
    df = build_pooled_data(time_block=time_block)
    _, ivs, blocks = define_specifications(time_block=time_block)
    s_cols = blocks["Tech"]
    df, _ = run_pooled_first_stage(df, ivs["IV_HausmanFull"],
                                   [c for c in s_cols if c != "constant"])
    df["qidx"] = df["year"].astype(int) * 4 + (df["quarter"].astype(int) - 1)
    return df, s_cols


def merge_parquet_cols(df, cols):
    """Bring parquet columns onto the sleep frame in SLEEP-FRAME units."""
    pq = pd.read_parquet(PARQUET, columns=["entity_id", "time_id"] + cols)
    for c in ("Dep_Act", "M_mt", "M_nat"):
        if c in cols:
            pq[c] = pq[c] / DEP_SCALE
    return df.merge(pq, on=["entity_id", "time_id"], how="left", validate="1:1")


def lag_within_entity(df, col):
    """t-1 value of `col` by exact quarter alignment (robust to gaps, unlike shift)."""
    prev = df[["entity_id", "qidx", col]].copy()
    prev["qidx"] = prev["qidx"] + 1
    prev = prev.rename(columns={col: f"{col}_lag"})
    return df.merge(prev, on=["entity_id", "qidx"], how="left", validate="1:1")


def run_augmented(df, s_cols, extra_cols, has_cf):
    """Baseline and augmented second stages on the IDENTICAL sample (all-columns dropna),
    so Delta-phi is not a sample-composition artifact. Returns (res_base, res_aug, d)."""
    df = df.copy()
    X_cols = []
    for sv in s_cols:
        col = "nr_lagged_dep" if sv == "constant" else f"interaction_{sv}"
        df[col] = df["nr_lagged_dep"] if sv == "constant" else df[sv] * df["nr_lagged_dep"]
        X_cols.append(col)
    if has_cf:
        X_cols.append("v_hat_x_lagged_dep")

    d = df.dropna(subset=X_cols + extra_cols + ["deposit_balance"]).copy()
    if len(d) == 0:
        raise RuntimeError("empty sample after dropna -- check merges")
    cl = d["CodConglomeradoPrudencial"].astype(str)

    def _fit(cols):
        y_dm = demean_variables_2way(d, ["deposit_balance"], "entity_id", "time_id")["deposit_balance"]
        X_dm = demean_variables_2way(d, cols, "entity_id", "time_id")
        res = sm.OLS(y_dm, X_dm).fit(cov_type="cluster", cov_kwds={"groups": cl}, use_t=True)
        return apply_imbalanced_cluster_correction(res, cl)

    res_base = _fit(X_cols)
    res_aug = _fit(X_cols + extra_cols) if extra_cols else None
    return res_base, res_aug, d


def report_delta_phi(tag, res_base, res_aug, extra_cols, n, n_full):
    """phi-hat at the average market = coefficient on nr_lagged_dep (states are centred)."""
    phi0 = float(res_base.params["nr_lagged_dep"])
    phi1 = float(res_aug.params["nr_lagged_dep"])
    print(f"\n  [{tag}] n={n:,} (full-sample n={n_full:,}; augmentation loses "
          f"{1 - n / max(n_full, 1):.1%})")
    print(f"    phi-hat(avg market)  baseline  = {phi0:.4f}")
    print(f"    phi-hat(avg market)  augmented = {phi1:.4f}   Delta = {phi1 - phi0:+.4f}")
    rows = [{"spec": tag, "param": "phi_avg_market_base", "coef": phi0},
            {"spec": tag, "param": "phi_avg_market_aug", "coef": phi1}]
    for c in extra_cols:
        co, t, p = (float(res_aug.params[c]), float(res_aug.tvalues[c]),
                    float(res_aug.pvalues[c]))
        print(f"    {c:<26s} coef={co:+.5g}  t={t:+.2f}  WCB p={p:.4f}")
        rows.append({"spec": tag, "param": c, "coef": co, "t": t, "p_wcb": p})
    return rows


# ==============================================================================
# ARMS
# ==============================================================================
def arm_identity():
    """D0: the accounting identity, its selection, and accrual-rate consistency."""
    print("\n=== D0: identity / tautology check ===")
    pq = pd.read_parquet(PARQUET)
    b = pq[pq["is_B"].astype(bool)]
    d_ = pq[~pq["is_B"].astype(bool)]
    res_b = (b["deposit_balance"] - b["phi_mt"] * b["gross_return_lag"] * b["lagged_deposits"]
             - b["Dep_Act"]).abs()
    res_d = (d_["deposit_balance"] - d_["phi_t"] * d_["gross_return_lag"] * d_["lagged_deposits"]
             - d_["Dep_Act"]).abs()
    print(f"  parquet identity |Dep - phi*g*L - Dep_Act|: B max={res_b.max():.3e} "
          f"(n={len(b):,}), D max={res_d.max():.3e} (n={len(d_):,})")

    df, _ = load_sleep_frame()
    df = merge_parquet_cols(df, ["Dep_Act", "phi_mt", "phi_t", "gross_return_lag"])
    matched = df["Dep_Act"].notna()
    print(f"  sleep-frame rows: {len(df):,}; matched in parquet: {matched.sum():,} "
          f"({matched.mean():.1%}) -- unmatched rows were dropped by the demand prep")

    # accrual-rate consistency: sleep nr/L vs demand gross_return_lag
    ok = matched & (df["lagged_deposits"] > 0)
    g_sleep = df.loc[ok, "nr_lagged_dep"] / df.loc[ok, "lagged_deposits"]
    g_diff = (g_sleep - df.loc[ok, "gross_return_lag"]).abs()
    print(f"  accrual consistency |g_sleep - g_demand|: max={g_diff.max():.3e}")

    # censoring measured on the FULL sleep frame (phi mapped from market level)
    phi_map = (pd.read_parquet(PARQUET, columns=["mca_code", "time_id", "phi_mt"])
               .drop_duplicates(["mca_code", "time_id"]))
    df = df.drop(columns=["phi_mt"]).merge(phi_map, on=["mca_code", "time_id"], how="left")
    df["phi_use"] = np.where(df["CODMUN_IBGE"].astype(str) != "0", df["phi_mt"], df["phi_t"])
    has_phi = df["phi_use"].notna() & (df["lagged_deposits"] > 0)
    g = df.loc[has_phi, "nr_lagged_dep"] / df.loc[has_phi, "lagged_deposits"]
    val = df.loc[has_phi, "deposit_balance"] - df.loc[has_phi, "phi_use"] * g * df.loc[has_phi, "lagged_deposits"]
    neg = (val < 0).mean()
    tiny = ((val >= 0) & (val <= 1e-6 / DEP_SCALE)).mean()  # parquet drops Dep_Act<=1e-6 RAW R$
    awake_share = (val.clip(lower=0) / df.loc[has_phi, "deposit_balance"]).median()
    print(f"  censoring on the full sleep frame (n={has_phi.sum():,}): "
          f"val<0 in {neg:.2%} of rows; val in (0,1e-6 raw] in {tiny:.2%}")
    print(f"  median awake-flow share of the stock, max(0,val)/Dep: {awake_share:.2%}")

    taut = (res_b.max() < 1e-4) and (res_d.max() < 1e-4)
    print("\n  VERDICT:", "IDENTITY CONFIRMED -- Dep = phi*g*L + Dep_Act holds row-by-row;" if taut
          else "identity FAILS -- accrual sources diverge, investigate before the other arms;")
    if taut:
        print("  a contemporaneous Dep_Act augmentation is TAUTOLOGICAL (returns coef 1, R2=1).")
        print("  Only the max{0,.} censoring plus the >1e-6 drop separate the two objects;")
        print("  D2 must use the LAGGED (D2b) or fitted (D2a) variants.")
    return [{"spec": "identity", "param": "max_abs_resid_B", "coef": float(res_b.max())},
            {"spec": "identity", "param": "share_val_neg", "coef": float(neg)},
            {"spec": "identity", "param": "matched_share", "coef": float(matched.mean())}]


def build_uncensored_inflow(df):
    """A_t = Dep_t - phi*g*Dep_{t-1} on EVERY row, signed, in R$bn.

    This is the regression's own error term (arm 'identity' verifies the accounting holds
    exactly). The demand step stores instead Dep_Act = max{0, A} and drops the zeros, because
    BLP shares must be non-negative -- a requirement of THAT step, not a property of this
    error. Using the censored version as D2b's regressor conditions the test on the sign of
    the lagged error it is testing, and costs 33-40% of rows; on the surviving subsample the
    two-way demeaning returns phi-hat ~ 0.80 against ~0.91 on the full frame, so the
    augmentation was being asked about a different carry coefficient from the headline.

    phi is a MARKET-level object (estimation_demand_link_common.py:433-463), so it can be
    mapped to censored rows too: dedupe phi_mt by (mca_code, time_id) and phi_t by time_id.
    """
    phi_mt = (pd.read_parquet(PARQUET, columns=["mca_code", "time_id", "phi_mt"])
              .dropna().drop_duplicates(["mca_code", "time_id"]))
    phi_t = (pd.read_parquet(PARQUET, columns=["time_id", "phi_t"])
             .dropna().drop_duplicates(["time_id"]))
    out = df.drop(columns=[c for c in ("phi_mt", "phi_t") if c in df.columns])
    out = out.merge(phi_mt, on=["mca_code", "time_id"], how="left")
    out = out.merge(phi_t, on="time_id", how="left")
    out["phi_use"] = np.where(out["CODMUN_IBGE"].astype(str) != "0",
                              out["phi_mt"], out["phi_t"])
    cov = out["phi_use"].notna().mean()
    out["A_full"] = out["deposit_balance"] - out["phi_use"] * out["nr_lagged_dep"]
    # censoring indicator, for the D2c margin test below
    out["C_lagpos"] = (out["A_full"] > 1e-6 / DEP_SCALE).astype(float)
    print(f"  uncensored inflow built on {cov:.2%} of rows (phi mapped from market level)")
    return out


def arm_lagdepact():
    """D2b: lagged awake inflow. Uses the UNCENSORED residual (see build_uncensored_inflow);
    the censored `Dep_Act` variant is reported alongside so the two are comparable."""
    print("\n=== D2b: lagged awake-inflow augmentation ===")
    df, s_cols = load_sleep_frame()
    df = merge_parquet_cols(df, ["Dep_Act"])
    df = build_uncensored_inflow(df)
    df = lag_within_entity(df, "Dep_Act")
    df = lag_within_entity(df, "A_full")
    df = lag_within_entity(df, "C_lagpos")
    df["C_lag_x_Z"] = df["C_lagpos_lag"] * df["nr_lagged_dep"]

    rows = []
    for tag, has_cf in (("spec12(k=4,5)", True), ("OLSxTech(all k)", False)):
        # headline: uncensored regressor on the full frame
        res_b, res_a, d = run_augmented(df, s_cols, ["A_full_lag"], has_cf)
        n_full = len(df.dropna(subset=["nr_lagged_dep", "deposit_balance"]
                               + (["v_hat_x_lagged_dep"] if has_cf else [])))
        rows += report_delta_phi(f"{tag}|uncensored", res_b, res_a, ["A_full_lag"],
                                 len(d), n_full)

        # D2c: does the carry itself differ across the censoring margin? Under H0 the
        # censoring indicator is a function of past errors only, so interacted with the
        # carry it must be zero -- this is what answers the "the 0.80 vs 0.91 gap is
        # evidence of contamination" reading directly, on the FULL sample.
        _, res_c, dc = run_augmented(df, s_cols, ["C_lag_x_Z"], has_cf)
        print(f"    [D2c margin test] C(A_lag>0) x carry: coef="
              f"{float(res_c.params['C_lag_x_Z']):+.5f}  "
              f"WCB p={float(res_c.pvalues['C_lag_x_Z']):.4f}  (n={len(dc):,})")
        rows.append({"spec": tag, "param": "C_lag_x_Z",
                     "coef": float(res_c.params["C_lag_x_Z"]),
                     "p_wcb": float(res_c.pvalues["C_lag_x_Z"]), "n": len(dc)})

        # legacy censored variant, for comparability with the archived numbers
        res_b2, res_a2, d2 = run_augmented(df, s_cols, ["Dep_Act_lag"], has_cf)
        rows += report_delta_phi(f"{tag}|censored(legacy)", res_b2, res_a2,
                                 ["Dep_Act_lag"], len(d2), n_full)

        # residual autocorrelation of the BASELINE regression (AB m1 analog)
        d = d.copy()
        d["_e"] = (demean_variables_2way(d, ["deposit_balance"], "entity_id", "time_id")["deposit_balance"]
                   - res_b.fittedvalues)
        d = lag_within_entity(d, "_e")
        de = d.dropna(subset=["_e", "_e_lag"])
        ar = sm.OLS(de["_e"], sm.add_constant(de["_e_lag"])).fit(
            cov_type="cluster", cov_kwds={"groups": de["CodConglomeradoPrudencial"].astype(str)})
        print(f"    baseline-residual AR(1) rho = {ar.params['_e_lag']:+.4f} "
              f"(t={ar.tvalues['_e_lag']:+.2f}) -- nonzero = the exact violation phi rides on")
        rows.append({"spec": tag, "param": "resid_ar1", "coef": float(ar.params["_e_lag"]),
                     "t": float(ar.tvalues["_e_lag"])})

    print("\n  VERDICT: a significant lagged-inflow coefficient (or any material Delta-phi)")
    print("  rejects 'awake-inflow innovations are serially uncorrelated within entity',")
    print("  the exact assumption separating sleepiness from woke-and-stayed persistence.")
    print("  Read the UNCENSORED rows: they run on ~95% of the frame at the production")
    print("  carry level. The censored rows are kept only to match the archived numbers --")
    print("  they condition on the sign of the lagged error and lose a third of the panel.")
    print("  D2c (censoring margin) is the direct answer to 'but phi-hat is lower on the")
    print("  augmentable subsample': if that interaction is null, the gap is a thinned-panel")
    print("  artifact of the within transform, not evidence against the assumption.")
    return rows


def _shares_from_dhat(b, dhat_col, out_col):
    """Logit shares from a fitted index within (mca x time), rescaled so the fitted
    inside shares sum to the observed inside-share total per market."""
    e = np.exp(b[dhat_col] - b.groupby("_grp")[dhat_col].transform("max"))
    denom = e.groupby(b["_grp"]).transform("sum")
    inside_tot = b.groupby("_grp")["share_B_cond"].transform("sum").clip(upper=0.95)
    b[out_col] = (1.0 - b["phi_mt"]) * b["M_mt"] * (e / denom * inside_tot) / DEP_SCALE
    return b


def arm_fittedshare():
    """D2a: characteristics-fitted awake inflow, xi-hat excluded. Two coefficient
    sources: the local within-market OLS (self-contained), and -- when the downloaded
    cluster results exist -- the BLP theta1 of E7/E8 spec 12 (stage `extended`)."""
    print("\n=== D2a: fitted-share augmentation (xi-hat excluded; lower bound) ===")
    x_blp = ["spread_ann"] + BLP_X_COLS
    pq = pd.read_parquet(PARQUET, columns=["entity_id", "time_id", "mca_code", "is_B",
                                           "share_B_cond", "phi_mt", "M_mt"] + x_blp)
    b = pq[pq["is_B"].astype(bool)].dropna(subset=["share_B_cond"] + x_blp).copy()
    b = b[b["share_B_cond"] > 0]
    b["_delta"] = np.log(b["share_B_cond"])
    b["_grp"] = b["mca_code"].astype(str) + "|" + b["time_id"].astype(str)

    # (a) local within-market OLS; fitted value EXCLUDES the residual (xi-hat), which
    # mechanically absorbs the accounting Dep_Act and would be circular.
    Xc = b[x_blp] - b.groupby("_grp")[x_blp].transform("mean")
    yc = b["_delta"] - b.groupby("_grp")["_delta"].transform("mean")
    beta = np.linalg.lstsq(Xc.to_numpy(float), yc.to_numpy(float), rcond=None)[0]
    b["_dhat"] = b[x_blp].to_numpy(float) @ beta
    b = _shares_from_dhat(b, "_dhat", "A_hat_c")
    variants = [("A_hat_c", "local within-market OLS")]

    # (b) cluster BLP theta1 (proper IV estimates; spread enters as spread_ann/100 = pp)
    for est in ("E7", "E8"):
        th = load_blp_theta1(est)
        if th is None:
            print(f"  [{est}] no converged extended results in {BLP_RAW.name}; skipped")
            continue
        col = f"A_hat_{est}"
        b["_dhat_blp"] = (th["alpha"] * b["spread_ann"] / 100.0
                          + sum(th[c] * b[c] for c in BLP_X_COLS))
        b = _shares_from_dhat(b, "_dhat_blp", col)
        variants.append((col, f"BLP theta1 {est} spec12 extended (alpha={th['alpha']:+.3f})"))

    df, s_cols = load_sleep_frame()
    keep = ["entity_id", "time_id"] + [v for v, _ in variants]
    df = df.merge(b[keep], on=["entity_id", "time_id"], how="left", validate="1:1")
    rows = []
    for col, desc in variants:
        print(f"\n  --- awake-inflow proxy: {desc} ---")
        for tag, has_cf in (("spec12(k=4,5)", True), ("OLSxTech(all k)", False)):
            res_b, res_a, d = run_augmented(df, s_cols, [col], has_cf)
            n_full = len(df.dropna(subset=["nr_lagged_dep", "deposit_balance"]
                                   + (["v_hat_x_lagged_dep"] if has_cf else [])))
            rows += report_delta_phi(f"{tag}|{col}", res_b, res_a, [col], len(d), n_full)
    print("\n  VERDICT: lambda > 0 with phi-hat falling = contamination through the")
    print("  OBSERVABLE component of awake-inflow persistence. Because xi-hat is excluded")
    print("  by construction, this is a LOWER BOUND on the confound. The BLP variants")
    print("  use the cluster-estimated demand coefficients (instrumented) in place of")
    print("  the local projection; agreement across sources is the robustness check.")
    return rows


def arm_blpelast():
    """D9: BLP-elasticity consistency. The model allows ONLY the awake margin to react
    to the contemporaneous spread: dDep/drho = (1-phi)*M*alpha*s(1-s). Compare that
    implied response (cluster alpha-hat, parquet phi/M/s) with the panel's instrumented
    spread response. Observed >> implied means the awake mass (1-phi-hat) is understated,
    i.e. phi-hat overstates sleepiness."""
    print("\n=== D9: BLP elasticity consistency (implied vs observed spread response) ===")
    pq = pd.read_parquet(PARQUET, columns=["entity_id", "time_id", "is_B", "deposit_type",
                                           "share_B_cond", "phi_mt", "M_mt"])
    b = pq[pq["is_B"].astype(bool) & pq["deposit_type"].isin([4, 5])].dropna(
        subset=["share_B_cond", "phi_mt", "M_mt"]).copy()
    s = b["share_B_cond"].clip(0.0, 0.95)
    # implied dDep/drho in SLEEP-FRAME units: R$bn per unit of QUARTERLY DECIMAL spread.
    # alpha-hat is per ANNUAL PP (spread_ann/100); 1 unit qoq-decimal = 400 annual pp.
    base = (1.0 - b["phi_mt"]) * b["M_mt"] * s * (1.0 - s) * 400.0 / DEP_SCALE

    # observed: 2SLS-style spread response. Fitted first-stage spread (rho_hat = spread
    # - v_hat on the instrumented k=4,5 rows) replaces the CF term; the carry block stays.
    df, s_cols = load_sleep_frame()
    df["rho_hat"] = df["spread_qoq"] - df["v_hat"]
    res_b, res_a, d = run_augmented(df, s_cols, ["rho_hat"], has_cf=False)
    coef = float(res_a.params["rho_hat"])
    pval = float(res_a.pvalues["rho_hat"])
    print(f"  observed dDep/drho (2SLS, within FE + carry block): {coef:+.4f} R$bn per "
          f"qoq-decimal (WCB p={pval:.4f}, n={len(d):,})")

    rows = [{"spec": "observed", "param": "dDep_drho", "coef": coef, "p_wcb": pval}]
    phi_bar = float(b["phi_mt"].mean())
    for est in ("E7", "E8"):
        th = load_blp_theta1(est)
        if th is None:
            print(f"  [{est}] no converged extended results; skipped")
            continue
        implied = float((base * th["alpha"]).mean())
        print(f"  [{est}] alpha={th['alpha']:+.4f}/annual-pp -> implied mean dDep/drho = "
              f"{implied:+.4f} R$bn per qoq-decimal")
        if implied != 0 and np.sign(coef) == np.sign(implied):
            k = coef / implied
            phi_implied = 1.0 - k * (1.0 - phi_bar)
            print(f"        observed/implied k = {k:.2f}  ->  phi consistent with the "
                  f"demand step = {phi_implied:.4f} (parquet mean phi-hat = {phi_bar:.4f})")
            rows.append({"spec": est, "param": "k_ratio", "coef": k,
                         "phi_implied": phi_implied, "phi_hat": phi_bar})
        else:
            print("        sign mismatch with the observed response -- ratio not "
                  "interpretable (simultaneity or weak response); recorded only")
            rows.append({"spec": est, "param": "implied", "coef": implied})
    print("\n  VERDICT: k >> 1 (observed response many times the phi-implied one) says")
    print("  deposits react to spreads far more than the sleepy carry allows -- the awake")
    print("  mass is understated and phi-hat overstates sleepiness. k ~ 1 = the sleep and")
    print("  demand steps are mutually consistent. Wrong-signed observed response =")
    print("  simultaneity dominates; treat as uninformative.")
    return rows


def arm_spreadlevel():
    """D4: contemporaneous spread level. Qualitative only."""
    print("\n=== D4: contemporaneous spread level (QUALITATIVE ONLY) ===")
    df, s_cols = load_sleep_frame()
    rows = []
    for tag, has_cf in (("spec12(k=4,5)", True), ("OLSxTech(all k)", False)):
        res_b, res_a, d = run_augmented(df, s_cols, ["spread_qoq"], has_cf)
        n_full = len(d)
        rows += report_delta_phi(tag, res_b, res_a, ["spread_qoq"], len(d), n_full)
    print("\n  VERDICT: not interpretable as a clean timing test -- the CF residual is the")
    print("  contemporaneous spread net of instruments (collinear by construction), and a")
    print("  loading is equally consistent with banks pricing expected inflows. Recorded only.")
    return rows


def arm_pix():
    """D6: event study of the carry coefficient around Pix launch (2020Q4)."""
    print("\n=== D6: Pix event study of the carry coefficient ===")
    df, s_cols = load_sleep_frame()
    s_cols = [c for c in s_cols if c != "pix_exists"]   # bins replace the pix step
    LAUNCH = 2020 * 4 + 3                               # qidx of 2020Q4
    df["_ev"] = df["qidx"] - LAUNCH

    # 2-quarter bins on [-8,+8); reference bin [-2,-1]; outside window uncontrolled
    bins = [(-8, -7), (-6, -5), (-4, -3), (0, 1), (2, 3), (4, 5), (6, 7)]
    # predetermined connectivity split: entity's pre-2020 mean (centred units keep order)
    pre = df[df["year"] < 2020].groupby("entity_id")["connections_per100"].mean()
    df["_hi_conn"] = (df["entity_id"].map(pre) > pre.median()).astype(float)

    extra, labels = [], []
    for lo, hi in bins:
        nm = f"evZ_{lo}_{hi}"
        m = ((df["_ev"] >= lo) & (df["_ev"] <= hi)).astype(float)
        df[nm] = m * df["nr_lagged_dep"]
        df[nm + "_hi"] = df[nm] * df["_hi_conn"]
        extra += [nm, nm + "_hi"]
        labels.append((lo, hi))

    rows = []
    for tag, has_cf in (("spec12(k=4,5)", True), ("OLSxTech(all k)", False)):
        res_b, res_a, d = run_augmented(df, s_cols, extra, has_cf)
        print(f"\n  [{tag}] n={len(d):,}  event-time carry deviations vs [-2,-1] "
              f"(base + high-connectivity extra):")
        for lo, hi in labels:
            nm = f"evZ_{lo}_{hi}"
            print(f"    [{lo:+d},{hi:+d}]  base={res_a.params[nm]:+.4f} "
                  f"(p={res_a.pvalues[nm]:.3f})   x hi-conn={res_a.params[nm + '_hi']:+.4f} "
                  f"(p={res_a.pvalues[nm + '_hi']:.3f})")
            rows.append({"spec": tag, "param": nm, "coef": float(res_a.params[nm]),
                         "p_wcb": float(res_a.pvalues[nm]),
                         "coef_hi": float(res_a.params[nm + "_hi"]),
                         "p_hi": float(res_a.pvalues[nm + "_hi"])})

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        mid = [(lo + hi) / 2 for lo, hi in labels]
        ax.axhline(0, lw=0.8, color="0.5")
        ax.axvline(-0.5, lw=0.8, color="0.5", ls="--")
        ax.plot(mid, [res_a.params[f"evZ_{lo}_{hi}"] for lo, hi in labels], "o-", label="base")
        ax.plot(mid, [res_a.params[f"evZ_{lo}_{hi}"] + res_a.params[f"evZ_{lo}_{hi}_hi"]
                      for lo, hi in labels], "s--", label="high connectivity")
        ax.set_xlabel("quarters since Pix launch (2020Q4)")
        ax.set_ylabel("carry-coefficient deviation vs [-2,-1]")
        ax.set_title(f"D6 Pix event study -- {tag}")
        ax.legend()
        fig.tight_layout()
        fp = OUT_DIR / f"d6_pix_event_{'cf' if has_cf else 'ols'}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f"    figure -> {fp.name}")

    print("\n  VERDICT: a post-launch drop in the carry coefficient concentrated in")
    print("  high-connectivity markets supports the ATTENTION reading of the Pix component")
    print("  (CF4's channel). Passing does NOT validate the phi level (asymmetric test).")
    return rows


# ==============================================================================
def arm_pixpooled():
    """D6b: the Pix quasi-experiment with ONE break parameter instead of D6's fourteen.

    D6 spends its degrees of freedom on 7 event-time bins x {base, high-connectivity} and
    finds nothing at ~9 effective clusters -- which is as consistent with low power as with
    no effect. This arm pools the post-launch window into a single interaction and uses
    connectivity CONTINUOUSLY (not a median split), then reports the minimum detectable
    effect so "no break" can be read as an actual bound rather than a silence.

    The triple post x carry x exposure is the test: it has cross-sectional variation, so
    conglomerate WCB is the right inference. The pooled post x carry term is identified off
    the time dimension alone (a national step), so it also gets quarter-clustered/DK
    treatment via utils.se_national and is labelled descriptive.
    """
    print("\n=== D6b: Pix pooled post x exposure (power-upgraded D6) ===")
    df, s_cols = load_sleep_frame()
    s_cols = [c for c in s_cols if c != "pix_exists"]     # the post dummy replaces the step
    LAUNCH = 2020 * 4 + 3
    df["_post"] = (df["qidx"] >= LAUNCH).astype(float)

    # predetermined exposure: entity's pre-2020 mean connectivity, standardised
    pre = df[df["year"] < 2020].groupby("entity_id")["connections_per100"].mean()
    e = df["entity_id"].map(pre)
    df["_expo"] = ((e - e.mean()) / e.std(ddof=0)).fillna(0.0)

    df["postZ"] = df["_post"] * df["nr_lagged_dep"]
    df["postZ_x"] = df["postZ"] * df["_expo"]
    df["Z_x"] = df["nr_lagged_dep"] * df["_expo"]          # level control
    df["_qc"] = df["qidx"] - df["qidx"].mean()
    df["trendZ"] = df["_qc"] * df["nr_lagged_dep"]         # secular-drift control

    rows = []
    for tag, has_cf in (("spec12(k=4,5)", True), ("OLSxTech(all k)", False)):
        for vname, extra in (("pooled", ["postZ", "postZ_x", "Z_x"]),
                             ("pooled+trend", ["postZ", "postZ_x", "Z_x", "trendZ"])):
            res_b, res_a, d = run_augmented(df, s_cols, extra, has_cf)
            print(f"\n  [{tag} | {vname}] n={len(d):,}")
            for p in extra:
                se = float(res_a.bse[p])
                # MDE at 5% size / 80% power. The multiplier is z_.975+z_.80 = 2.80 ONLY in
                # large samples; the reference distribution here has G*-1 ~ 5 degrees of
                # freedom (a handful of conglomerates carry the score mass), where
                # t_.975 + t_.80 = 3.46. Using 2.80 understates the detectable effect by
                # ~23%. The SE is itself a chi2_nu object at this nu, so the point MDE is
                # reported with the 90% band implied by that sampling uncertainty.
                mde = MDE_MULT * se
                print(f"    {p:<10s} coef={float(res_a.params[p]):+.5f}  se={se:.5f}  "
                      f"WCB p={float(res_a.pvalues[p]):.4f}   "
                      f"MDE(80%)={mde:.5f} [{MDE_LO*mde:.5f}, {MDE_HI*mde:.5f}]")
                rows.append({"spec": tag, "variant": vname, "param": p,
                             "coef": float(res_a.params[p]), "se": se,
                             "p_wcb": float(res_a.pvalues[p]), "mde80": mde,
                             "mde_lo": MDE_LO * mde, "mde_hi": MDE_HI * mde,
                             "n": len(d)})
            base_phi = float(res_a.params.get("nr_lagged_dep", np.nan))
            if np.isfinite(base_phi) and base_phi:
                m = 2.8 * float(res_a.bse["postZ_x"])
                print(f"    -> a Pix break in the carry larger than {m:.5f} per SD of "
                      f"connectivity ({100*m/abs(base_phi):.2f}% of the carry) would have "
                      "been detected 80% of the time.")
    pd.DataFrame(rows).to_csv(OUT_DIR / "d6b_pix_pooled.csv", index=False)
    print("\n  VERDICT: the TRIPLE (post x carry x connectivity) is the test -- it has")
    print("  cross-sectional variation, so the conglomerate WCB above applies. A null with")
    print("  a SMALL MDE bounds the Pix attention channel; a null with a LARGE MDE means the")
    print("  design cannot see it. The pooled post x carry term is time-identified only")
    print("  (a national step at ~35 quarters) and is descriptive, not a test.")
    print(f"\nresults -> {OUT_DIR / 'd6b_pix_pooled.csv'}")
    return rows


# ==============================================================================
ARMS = {"identity": arm_identity, "lagdepact": arm_lagdepact,
        "fittedshare": arm_fittedshare, "spreadlevel": arm_spreadlevel, "pix": arm_pix,
        "blpelast": arm_blpelast, "pixpooled": arm_pixpooled}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=list(ARMS) + ["all"], required=True)
    a = ap.parse_args()
    to_run = list(ARMS) if a.arm == "all" else [a.arm]
    all_rows = []
    for arm in to_run:
        all_rows += ARMS[arm]()
    out = OUT_DIR / f"d_augmented_{a.arm}.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nresults -> {out}")
