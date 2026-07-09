#!/usr/bin/env python3
"""weak_iv_analysis.py — weak-instruments battery for the BLP/logit demand first stage.

The demand model's ONLY endogenous regressor is the deposit spread (rf − dep_rate, the
markdown). The engine (`blp_1_logit.jl::project_spreads`) instruments it for deposit types 4
and 5 with 15 excluded instruments: leave-one-out rival characteristics (`loo_*`/`mean_loo_*`),
the rival count (`n_rivals`), cost ratios, and the capital ratio. The structural (logit) equation
is δ = α·spread + Xβ + ξ, with δ = ln(s_data) (share_D for D-type, share_B_cond for B-type) and
controls X = the product characteristics. Clustering is by `CodConglomeradoPrudencial`.

For each routine and each endogenous subsample (type 4, type 5, types 4+5 pooled) this reports:

  • 2SLS α̂ + cluster-robust SE                                       [linearmodels; == manual]
  • cluster-robust first-stage F = Kleibergen-Paap rk Wald F         [numpy; == statsmodels 5 dp]
  • Cragg–Donald F (homoskedastic first-stage F)                     [numpy; == statsmodels]
  • Montiel-Olea–Pflueger effective F (cluster-robust)               [numpy]
        F_eff = (x'P_Z x) / tr( Ŵ₂ (Z'Z/n)⁻¹ ),  Ŵ₂ = cluster cov of the first-stage moments
        (Olea & Pflueger 2013; collapses to the homoskedastic first-stage F under iid errors).
  • partial R² of the excluded instruments                           [linearmodels]
  • Kleibergen LM/K 95% CI for α (cluster-robust; isolates α from overid)         [numpy]
        — the weak-IV-robust CI to report; stays informative when AR is empty.
  • Anderson–Rubin 95% CI (joint with overid) + Hansen J overid test              [numpy]
        — min_α AR = the Hansen J, so AR is empty whenever overid is rejected.
  Both AR/LM CIs are grid-inverted against WILD-CLUSTER-BOOTSTRAP criticals (Webb, seed 0; score/
  no-refit, few-cluster valid at G*≈7) instead of asymptotic χ² — the asymptotic sets are kept in
  the JSON under *_chi2 keys for reference.

NOTE: linearmodels' `first_stage` F-stat is NOT cluster-robust in the installed version
(≈ the homoskedastic F), so the cluster-robust first-stage F is computed here in numpy and
validated against statsmodels' clustered F. Empirically the homoskedastic F is in the hundreds
but the cluster-robust effective/KP F is an order of magnitude smaller — the leave-one-out
instruments are highly correlated within conglomerate.

All first-stage statistics are computed AFTER partialling the included exogenous controls
(const + X) out of the spread and the instruments (Frisch–Waugh–Lovell). Writes
cluster_processed/weak_iv.json (consumed by process_blp_outputs.py) and prints a table.

Run:  python weak_iv_analysis.py [BLP_RESULTS_dir] [--routines 5,6,7,8]
"""
import os, sys, glob, json, argparse
import numpy as np

X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5", "log_total_assets_lag"]
IV_COLS = ["loo_log_assets", "mean_loo_log_assets", "loo_equity_ratio", "mean_loo_equity_ratio",
           "loo_basileia", "mean_loo_basileia", "loo_credit_assets", "mean_loo_credit_assets",
           "loo_npl_provision", "mean_loo_npl_provision", "n_rivals",
           "personnel_cost_ratio_lag", "admin_cost_ratio_lag", "tax_cost_ratio_lag",
           "indice_basileia_lag"]
ENDOG_TYPES = [4, 5]          # the engine instruments spread only for these deposit types
# Parsimonious instrument set = the leave-one-out rival CHARACTERISTICS only (drop the mean_loo_
# aggregates, n_rivals, the 3 cost ratios, and the capital ratio). Tests whether the low F is a
# many-weak-instruments dilution artifact vs genuine weakness, and cuts overid over-rejection.
PARSIMONIOUS_IV = ["loo_log_assets", "loo_equity_ratio", "loo_basileia",
                   "loo_credit_assets", "loo_npl_provision"]
AR_GRID = np.linspace(-2.0, 2.0, 4001)   # α grid (spread coef, percentage-point units)


def default_results_dir():
    repo = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(repo))   # .../Open-Finance
    return os.path.join(root, "BCB", "Egan_et_al_2025_Rep", "processed",
                        "ESTIMATION_OUTPUT", "BLP_RESULTS")


def _resid(M, W):
    """Residualize columns of M (N×k) on W (N×p) via least squares (FWL partialling)."""
    M = np.atleast_2d(M)
    if M.shape[0] != W.shape[0]:
        M = M.T
    beta, *_ = np.linalg.lstsq(W, M, rcond=None)
    return M - W @ beta


def _group_sum(A, codes, G):
    """Sum rows of A (N×K) within integer cluster codes (0..G-1) → (G×K)."""
    out = np.zeros((G, A.shape[1]), float)
    np.add.at(out, codes, A)
    return out


def _wild_weights(B, G, scheme, rng):
    """(B×G) wild bootstrap weights: 6-point Webb (default; robust to few/imbalanced clusters) or
    2-point Rademacher. Mean 0, variance 1 — mirrors se_common.jl `wild_weights`."""
    if scheme == "webb":
        vals = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
        return vals[rng.integers(0, 6, size=(B, G))]
    return rng.choice([-1.0, 1.0], size=(B, G))


def _mop_cv(M):
    """Simplified Montiel-Olea–Pflueger (2013) effective-F critical values (2SLS Nagar bias, 5% sig).
    M = Ŵ₂Q⁻¹ (its real eigenvalues are the generalized eigenvalues of Ŵ₂ w.r.t. Q). For each
    Nagar-bias tolerance τ, set x=1/τ; then (MOP eq. 4) the effective df is
        K_eff = tr(M)²(1+2x) / (tr(M²) + 2x·tr(M)·λ_max),
    and (MOP eq. 32) the conservative critical value is
        cv = χ²_{K_eff}(x·K_eff)_{0.95} / K_eff.
    This is the SIMPLIFIED test (bounds B_TSLS≤1 ⇒ valid under clustering); the famous "effective
    F > 23.1 for ≤10% bias" is τ=10%, K_eff=1. Homoskedastic (M∝I ⇒ K_eff=K) reproduces MOP Table 1
    (τ=10%: 23.11@K=1, 16.08@K=5, 13.86@K=15). Returns {bias05, bias10, bias20, bias30}."""
    from scipy.stats import ncx2
    lam = np.linalg.eigvals(M).real
    tr, tr2, mx = float(lam.sum()), float((lam ** 2).sum()), float(lam.max())
    out = {}
    for tag, x in (("bias05", 20.0), ("bias10", 10.0), ("bias20", 5.0), ("bias30", 10.0 / 3.0)):
        den = tr2 + 2.0 * x * tr * mx
        if den <= 0 or tr <= 0:
            out[tag] = None; continue
        keff = tr * tr * (1.0 + 2.0 * x) / den
        out[tag] = float(ncx2.ppf(0.95, keff, x * keff) / keff) if keff > 0 else None
    return out


def _fs_F(spread, X, Z, codes, G):
    """First-stage strength on an instrument (sub)set, FWL-partialled: cluster-robust effective-F
    (Montiel-Olea–Pflueger) and KP rk Wald F, the Patnaik effective number of instruments K_eff =
    tr(M)²/tr(M²) with M = Ŵ₂Q⁻¹, and the partial R². Same estimator as `_battery`'s first stage —
    used to re-run on the parsimonious instrument subset (many-weak-instruments / dilution check)."""
    n, K = len(spread), Z.shape[1]
    p = X.shape[1]
    if K == 0:
        return dict(eff_F=None, kp_F=None, n_iv=0, k_eff=None, partial_R2=None)
    s  = _resid(spread.reshape(-1, 1), X).ravel()
    Zt = _resid(Z, X)
    ZtZ = Zt.T @ Zt
    ZtZ_inv = np.linalg.pinv(ZtZ)
    pi = ZtZ_inv @ (Zt.T @ s)
    vhat = s - Zt @ pi
    ess = float(pi @ (ZtZ @ pi)); ss = float(s @ s)
    Mv = _group_sum(Zt * vhat[:, None], codes, G)
    W2 = (Mv.T @ Mv) / n
    M  = W2 @ np.linalg.pinv(ZtZ / n)
    trM, trM2 = float(np.trace(M)), float(np.trace(M @ M))
    eff_F = ess / trM if trM > 0 else None
    k_eff = (trM * trM / trM2) if trM2 > 0 else None
    c = (G / (G - 1.0)) * ((n - 1.0) / (n - p - K)) if G > 1 else 1.0
    Vpi = c * (ZtZ_inv @ (Mv.T @ Mv) @ ZtZ_inv)
    kp_F = float(pi @ np.linalg.pinv(Vpi) @ pi) / K
    return dict(eff_F=eff_F, kp_F=kp_F, n_iv=int(K), k_eff=k_eff, mop_cv=_mop_cv(M),
                partial_R2=(ess / ss if ss > 0 else None))


def _battery(delta, spread, X, Z, codes, G):
    """Core weak-IV battery on a (sub)sample. Returns a dict of statistics (numpy only).
    delta (N,), spread (N,), X (N×p controls incl. const), Z (N×K excluded instruments),
    codes (N,) integer cluster ids in 0..G-1."""
    from scipy.stats import chi2
    n, K = len(spread), Z.shape[1]
    p = X.shape[1]
    # FWL: partial controls X out of spread and instruments.
    s = _resid(spread.reshape(-1, 1), X).ravel()
    Zt = _resid(Z, X)
    ZtZ = Zt.T @ Zt
    ZtZ_inv = np.linalg.pinv(ZtZ)
    pi = ZtZ_inv @ (Zt.T @ s)                      # first-stage coefficients
    vhat = s - Zt @ pi                             # first-stage residuals
    ess = float(pi @ (ZtZ @ pi))                   # x'P_Z x  (numerator)
    # Cluster covariance of the first-stage moments Ŵ₂ = (1/n) Σ_g (Z̃_g'v̂_g)(Z̃_g'v̂_g)'
    Mv = _group_sum(Zt * vhat[:, None], codes, G)  # (G×K)
    W2 = (Mv.T @ Mv) / n
    Q = ZtZ / n
    Q_inv = np.linalg.pinv(Q)
    # Montiel-Olea–Pflueger effective F (cluster-robust)
    eff_F = ess / float(np.trace(W2 @ Q_inv))
    # Cragg–Donald F (homoskedastic first-stage F): ESS/K over σ̂²_v with first-stage residual df
    df_v = max(n - p - K, 1)
    s2v = float(vhat @ vhat) / df_v
    cragg_donald = (ess / K) / s2v if s2v > 0 else None
    # Kleibergen-Paap rk Wald F (cluster-robust first-stage F), small-sample corrected
    c = (G / (G - 1.0)) * ((n - 1.0) / (n - p - K)) if G > 1 else 1.0
    Vpi = c * (ZtZ_inv @ (Mv.T @ Mv) @ ZtZ_inv)    # cluster-robust VCV of π̂
    kp_F = float(pi @ np.linalg.pinv(Vpi) @ pi) / K
    # Patnaik effective # of instruments K_eff = tr(M)²/tr(M²), M = Ŵ₂Q⁻¹ (=K under homoskedastic
    # Ŵ₂∝Q; collapses toward 1 when the instruments are collinear within cluster). And the
    # subsample effective-cluster count G/(1+CV²) (the desc_3 G* metric, per subsample).
    M = W2 @ Q_inv
    trM, trM2 = float(np.trace(M)), float(np.trace(M @ M))
    k_eff = (trM * trM / trM2) if trM2 > 0 else None
    mop_cv = _mop_cv(M)          # simplified MOP effective-F critical values (compare eff_F against these)
    sizes = np.bincount(codes, minlength=G).astype(float)
    eff_clusters = float(G / (1.0 + sizes.var() / sizes.mean() ** 2)) if sizes.mean() > 0 else float(G)
    # Weak-IV-robust CIs via cluster-robust grid inversion of the moment ḡ(α₀)=(1/n)Σ_g(A_g−α₀B_g),
    # Ω̂(α₀)=(1/n)Σ_g(A_g−α₀B_g)(…)' (cluster cov of the moments). Two statistics per α₀:
    #   • Anderson-Rubin  S(α₀)=n·ḡ'Ω̂⁻¹ḡ ~ χ²_K  — joint with overid; min_α S = the Hansen J,
    #     so S is EMPTY whenever overid is rejected (common at large n).
    #   • Kleibergen LM/K  =n·(D̂'Ω̂⁻¹ḡ)²/(D̂'Ω̂⁻¹D̂) ~ χ²_1, where D̂ = d̄ − Σ̂_dg Ω̂⁻¹ḡ is the
    #     Jacobian orthogonalized from the moment (d̄=E[∂g/∂α]=−Σ_g B_g/n constant; Σ̂_dg cluster
    #     cov of the Jacobian moment with g). The K-test isolates the α direction from overid, so
    #     it does NOT go empty from overid rejection — the robust CI to report.
    dt = _resid(delta.reshape(-1, 1), X).ravel()
    A = _group_sum(Zt * dt[:, None], codes, G)     # Σ_g Z̃ δ̃   (G×K)
    B = _group_sum(Zt * s[:, None], codes, G)       # Σ_g Z̃ s̃    (G×K)
    crit_ar_chi2, crit_lm_chi2 = chi2.ppf(0.95, K), chi2.ppf(0.95, 1)
    # Wild cluster bootstrap criticals (score/no-refit): at each α₀ hold Ω̂ and D̂ fixed and resample
    # the cluster moment contributions with mean-0 wild weights → few-cluster-valid null distribution
    # (G*≈7). Same Webb scheme + seed 0 as the SE machinery. Asymptotic χ² sets kept for reference.
    Bwcb   = int(os.environ.get("BLP_WCB_REPS", os.environ.get("SLEEP_BOOT_B", "999")))
    scheme = os.environ.get("BLP_WCB_SCHEME", os.environ.get("SLEEP_BOOT_SCHEME", "webb")).lower()
    Wb     = _wild_weights(Bwcb, G, scheme, np.random.default_rng(0))    # (B×G)
    bA, bB = Wb @ A, Wb @ B                          # (B×K) each — bootstrap moment building blocks
    dbar = -B.sum(0) / n                            # E[∂g/∂α], constant in α
    acc_ar   = np.zeros(len(AR_GRID), bool); acc_ar_a = np.zeros(len(AR_GRID), bool)
    acc_lm   = np.zeros(len(AR_GRID), bool); acc_lm_a = np.zeros(len(AR_GRID), bool)
    J = np.inf
    for j, a0 in enumerate(AR_GRID):
        Mg = A - a0 * B
        g = Mg.sum(0) / n
        Oi = np.linalg.pinv((Mg.T @ Mg) / n)
        gstar = (bA - a0 * bB) / n                   # (B×K) bootstrap scores under H0: α=a0
        ar = n * float(g @ Oi @ g)
        ar_star = n * np.einsum("bk,kl,bl->b", gstar, Oi, gstar)
        acc_ar[j]   = ar <= np.quantile(ar_star, 0.95)      # WCB critical
        acc_ar_a[j] = ar <= crit_ar_chi2                    # asymptotic χ²_K
        if ar < J:
            J = ar                                 # min AR = Hansen J (efficient/CUE overid)
        D = dbar - (-(B.T @ Mg) / n) @ Oi @ g       # robust orthogonalized Jacobian
        den = float(D @ Oi @ D)
        if den > 1e-12:
            num = float(D @ Oi @ g); lm = n * num * num / den
            ns  = gstar @ (Oi @ D); lm_star = n * ns * ns / den     # (B,) bootstrap LM stats
            acc_lm[j]   = lm <= np.quantile(lm_star, 0.95)  # WCB critical
            acc_lm_a[j] = lm <= crit_lm_chi2                # asymptotic χ²_1
        else:
            acc_lm[j] = acc_lm_a[j] = True          # statistic ≡ 0 ⇒ accept
    def _ci(acc):
        if not acc.any():
            return dict(low=None, high=None, bounded=True, disconnected=False)
        idx = np.where(acc)[0]
        return dict(low=float(AR_GRID[idx[0]]), high=float(AR_GRID[idx[-1]]),
                    bounded=not (acc[0] or acc[-1]),
                    disconnected=bool((idx[-1] - idx[0] + 1) != len(idx)))
    ar_ci, lm_ci = _ci(acc_ar), _ci(acc_lm)             # WCB (primary)
    ar_ci_a, lm_ci_a = _ci(acc_ar_a), _ci(acc_lm_a)     # asymptotic χ² (reference)
    # Just-identified robust CI: collapse to the single optimal instrument ŝ = Z̃π̂ (fitted first
    # stage). With one instrument there is NO overidentification, so the AR set cannot be driven
    # empty by Hansen-J overid rejection (AR ≡ LM here) — it isolates weak-ID from that confound.
    # ŝ is a generated instrument, so treat this as an approximate robustness CI. WCB criticals.
    shat = Zt @ pi
    A1 = _group_sum((shat * dt).reshape(-1, 1), codes, G).ravel()   # Σ_g ŝ·δ̃
    B1 = _group_sum((shat * s).reshape(-1, 1), codes, G).ravel()    # Σ_g ŝ·s̃
    bA1, bB1 = Wb @ A1, Wb @ B1
    acc_ji = np.zeros(len(AR_GRID), bool)
    for j, a0 in enumerate(AR_GRID):
        m  = A1 - a0 * B1
        O1 = float(m @ m) / n
        if O1 <= 0:
            continue
        ar1 = n * (m.sum() / n) ** 2 / O1
        gs1 = (bA1 - a0 * bB1) / n
        acc_ji[j] = ar1 <= np.quantile(n * gs1 * gs1 / O1, 0.95)
    ji_ci = _ci(acc_ji)
    Jdf = max(K - 1, 1)
    return {
        "n_obs": int(n), "n_iv": int(K), "n_clusters": int(G),
        "kp_first_stage_F": kp_F, "cragg_donald_F": cragg_donald, "effective_F": eff_F,
        "k_eff": k_eff, "eff_clusters": eff_clusters,      # Patnaik eff. # instruments; per-subsample G*
        "mop_cv": mop_cv,                                  # simplified MOP effective-F critical values
        "ji_ci_low": ji_ci["low"], "ji_ci_high": ji_ci["high"], "ji_ci_bounded": ji_ci["bounded"],
        "ji_ci_disconnected": ji_ci["disconnected"],       # just-identified (single optimal IV) robust CI
        # AR/LM confidence sets: wild-cluster-bootstrap criticals (few-cluster valid at G*≈7).
        "ci_method": "wcb", "wcb_reps": int(Bwcb), "wcb_scheme": scheme,
        "lm_ci_low": lm_ci["low"], "lm_ci_high": lm_ci["high"],
        "lm_ci_bounded": lm_ci["bounded"], "lm_ci_disconnected": lm_ci["disconnected"],
        "ar_ci_low": ar_ci["low"], "ar_ci_high": ar_ci["high"], "ar_ci_bounded": ar_ci["bounded"],
        # asymptotic-χ² sets kept for reference (not few-cluster valid; the collapse to these shows
        # how much the WCB correction moves the CI).
        "lm_ci_chi2_low": lm_ci_a["low"], "lm_ci_chi2_high": lm_ci_a["high"],
        "ar_ci_chi2_low": ar_ci_a["low"], "ar_ci_chi2_high": ar_ci_a["high"],
        "hansen_J": float(J), "hansen_J_df": int(Jdf),
        "hansen_J_p": float(1.0 - chi2.cdf(J, Jdf)),
    }


def _add_linearmodels(rec, delta, spread, X, Z, clusters):
    """Augment a battery rec with the citable 2SLS α̂/SE + first-stage F/partial R² from
    linearmodels (graceful no-op if the package is unavailable)."""
    try:
        import pandas as pd
        from linearmodels.iv import IV2SLS
    except Exception:
        return rec
    try:
        ex = pd.DataFrame(X, columns=["const"] + [f"x{i}" for i in range(X.shape[1] - 1)])
        en = pd.DataFrame({"spread": spread})
        iv = pd.DataFrame(Z, columns=[f"z{i}" for i in range(Z.shape[1])])
        res = IV2SLS(pd.Series(delta, name="delta"), ex, en, iv).fit(
            cov_type="clustered", clusters=pd.Series(clusters))
        rec["alpha_2sls"] = float(res.params["spread"])     # verified == manual 2SLS to 4 dp
        rec["alpha_se"] = float(res.std_errors["spread"])   # cluster-robust, verified == manual
        # partial R² of the excluded instruments (cov-independent → reliable).
        # NB: do NOT use res.first_stage 'f.stat' as the first-stage F — in this linearmodels
        # version it is NOT cluster-robust (verified ≈ the homoskedastic F). The validated
        # cluster-robust first-stage F is the numpy `kp_first_stage_F` (matches the statsmodels
        # cluster-robust F to 5 dp); the homoskedastic one is `cragg_donald_F`.
        rec["partial_R2"] = float(res.first_stage.diagnostics.loc["spread", "partial.rsquared"])
    except Exception as e:
        print(f"    [linearmodels] 2SLS failed ({type(e).__name__}: {e})")
    return rec


def analyze_routine(k, dp):
    import pandas as pd
    fs = glob.glob(os.path.join(dp, f"demand_{k}_*spec_12.parquet"))
    if not fs:
        print(f"[weak-IV] E{k}: no demand parquet — skipped"); return None
    fs.sort(key=os.path.getmtime)
    need = list(dict.fromkeys(
        ["spread_ann", "share_D", "share_B_cond", "is_B", "deposit_type",
         "CodConglomeradoPrudencial"] + X_COLS + IV_COLS))
    df = pd.read_parquet(fs[-1], columns=[c for c in need if c is not None])
    is_B = df["is_B"].fillna(False).astype(bool).to_numpy()
    sD = df["share_D"].fillna(0.0).to_numpy(float)
    sB = df["share_B_cond"].fillna(0.0).to_numpy(float)
    delta = np.where(is_B, np.log(np.clip(sB, 1e-15, None)), np.log(np.clip(sD, 1e-15, None)))
    spread = df["spread_ann"].fillna(0.0).to_numpy(float) / 100.0
    Xmat = np.column_stack([np.ones(len(df))] +
                           [df[c].fillna(0.0).to_numpy(float) if c in df else np.zeros(len(df))
                            for c in X_COLS])
    iv_avail = [c for c in IV_COLS if c in df.columns]
    Zall = np.column_stack([np.nan_to_num(df[c].to_numpy(float), posinf=0.0, neginf=0.0)
                            for c in iv_avail])
    dtype = df["deposit_type"].fillna(0).astype(int).to_numpy()
    clus = df["CodConglomeradoPrudencial"].astype(str).to_numpy()

    out = {}
    subsamples = [("type4", dtype == 4), ("type5", dtype == 5),
                  ("type45", np.isin(dtype, ENDOG_TYPES))]
    for name, mask in subsamples:
        d, s, X, Z, cl = delta[mask], spread[mask], Xmat[mask], Zall[mask], clus[mask]
        ok = np.isfinite(d) & np.isfinite(s) & np.all(np.isfinite(X), 1) & np.all(np.isfinite(Z), 1)
        d, s, X, Z, cl = d[ok], s[ok], X[ok], Z[ok], cl[ok]
        # drop zero-variance instruments and collinear controls in this subsample
        Z = Z[:, Z.std(0) > 1e-10]
        X = X[:, np.r_[True, X[:, 1:].std(0) > 1e-10]] if X.shape[1] > 1 else X
        if len(s) <= X.shape[1] + Z.shape[1] + 1 or Z.shape[1] == 0:
            print(f"[weak-IV] E{k}/{name}: too few obs / no IVs — skipped"); continue
        uniq, codes = np.unique(cl, return_inverse=True)
        G = len(uniq)
        rec = _battery(d, s, X, Z, codes, G)
        rec = _add_linearmodels(rec, d, s, X, Z, cl)
        # Parsimonious subset (rival characteristics only): dilution / many-weak-instruments check.
        pcols = [i for i, c in enumerate(iv_avail) if c in PARSIMONIOUS_IV]
        if 0 < len(pcols) < Z.shape[1]:
            rec["parsimonious"] = _fs_F(s, X, Z[:, pcols], codes, G)
        out[name] = rec
        a = rec.get("alpha_2sls"); se = rec.get("alpha_se")
        astr = f"a_hat={a:+.3f}({se:.3f})" if a is not None else "a_hat=NA"
        lm = (f"[{rec['lm_ci_low']:+.3f},{rec['lm_ci_high']:+.3f}]"
              + ("*disc" if rec.get("lm_ci_disconnected") else "")
              if rec["lm_ci_low"] is not None else "(none)")
        ji = (f"[{rec['ji_ci_low']:+.3f},{rec['ji_ci_high']:+.3f}]"
              + ("*disc" if rec.get("ji_ci_disconnected") else "")
              if rec.get("ji_ci_low") is not None else "(none)")
        pf = (rec.get("parsimonious") or {}).get("eff_F")
        print(f"[weak-IV] E{k}/{name}: N={rec['n_obs']:,} K={rec['n_iv']}(Keff={rec['k_eff']:.1f}) "
              f"G={rec['n_clusters']}(G*={rec['eff_clusters']:.1f}) | {astr} | "
              f"eff-F={rec['effective_F']:,.0f} KP-F={rec['kp_first_stage_F']:,.0f} "
              f"parsF={(pf or 0):,.0f} | LM95 {lm} | JI95 {ji} | J p={rec['hansen_J_p']:.2f}")
    return out


def main():
    try:                                  # Windows consoles default to cp1252; keep prints safe
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=default_results_dir())
    ap.add_argument("--routines", default="5,6,7,8")
    args = ap.parse_args()
    RES = os.path.abspath(args.results_dir)
    dp = os.path.join(os.path.dirname(RES), "DEMAND_PREP")
    routines = [int(x) for x in args.routines.split(",") if x.strip()]
    try:
        import linearmodels  # noqa: F401
    except Exception:
        print("[weak-IV] NOTE: linearmodels not installed — 2SLS α̂/SE + partial R² will be "
              "omitted (effective-F, Cragg-Donald, KP-F, AR-CI still computed). "
              "Install with: pip install linearmodels")
    out = {}
    for k in routines:
        r = analyze_routine(k, dp)
        if r:
            out[str(k)] = r
    if out:
        os.makedirs(os.path.join(RES, "cluster_processed"), exist_ok=True)
        path = os.path.join(RES, "cluster_processed", "weak_iv.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[weak-IV] wrote {path}  ({len(out)} routines)")
    else:
        print("[weak-IV] nothing computed (no parquets / linearmodels?).")


if __name__ == "__main__":
    main()
