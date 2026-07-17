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
           "indice_basileia_lag",
           "estban_rival_branches_lag"]   # ESTBAN branch-competition IV (within-conglomerate variation)
ENDOG_TYPES = [4, 5]          # the engine instruments spread only for these deposit types
# Parsimonious instrument set = the leave-one-out rival CHARACTERISTICS only (drop the mean_loo_
# aggregates, n_rivals, the 3 cost ratios, and the capital ratio). Tests whether the low F is a
# many-weak-instruments dilution artifact vs genuine weakness, and cuts overid over-rejection.
PARSIMONIOUS_IV = ["loo_log_assets", "loo_equity_ratio", "loo_basileia",
                   "loo_credit_assets", "loo_npl_provision"]
AR_GRID = np.linspace(-2.0, 2.0, 4001)   # α grid (spread coef, percentage-point units)

# Named instrument groups (for the leave-one-group-out sensitivity). mean_loo is the collinear block.
IV_GROUPS = {
    "loo":      ["loo_log_assets", "loo_equity_ratio", "loo_basileia", "loo_credit_assets",
                 "loo_npl_provision"],
    "mean_loo": ["mean_loo_log_assets", "mean_loo_equity_ratio", "mean_loo_basileia",
                 "mean_loo_credit_assets", "mean_loo_npl_provision"],
    "n_rivals": ["n_rivals"],
    "cost":     ["personnel_cost_ratio_lag", "admin_cost_ratio_lag", "tax_cost_ratio_lag"],
    "capital":  ["indice_basileia_lag"],
    "estban":   ["estban_rival_branches_lag"],
}

# tF critical values — Lee, McCrary, Moreira & Porter (2022) "Valid t-ratio Inference for IV", AER
# 112(10):3260-3290, Table 3 Panel A (0.05 level). _TF_F = first-stage (effective) F; _TF_CV =
# √c_{0.05}(F), the F-adjusted t-ratio critical value. TRANSCRIBED VERBATIM from the published table
# (100 selected points; the paper prescribes LINEAR interpolation between them). Anchors verified:
# F=104.67→1.96; F=10.253→3.385 (the paper's worked example: SE adj 1.751 at F=10 ⇒ cv=1.751·1.96).
# Below F=q_{0.95}=1.96²≈3.8416 the tF CI is the whole real line (cv=∞); for F≥104.67, cv=1.96.
_TF_FMIN = 1.96 ** 2   # 3.8416 — vertical asymptote of the 0.05 tF function
_TF_F = [
    4.000, 4.008, 4.015, 4.023, 4.031, 4.040, 4.049, 4.059, 4.068, 4.079,
    4.090, 4.101, 4.113, 4.125, 4.138, 4.151, 4.166, 4.180, 4.196, 4.212,
    4.229, 4.247, 4.265, 4.285, 4.305, 4.326, 4.349, 4.372, 4.396, 4.422,
    4.449, 4.477, 4.507, 4.538, 4.570, 4.604, 4.640, 4.678, 4.717, 4.759,
    4.803, 4.849, 4.897, 4.948, 5.002, 5.059, 5.119, 5.182, 5.248, 5.319,
    5.393, 5.472, 5.556, 5.644, 5.738, 5.838, 5.944, 6.056, 6.176, 6.304,
    6.440, 6.585, 6.741, 6.907, 7.085, 7.276, 7.482, 7.702, 7.940, 8.196,
    8.473, 8.773, 9.098, 9.451, 9.835, 10.253, 10.711, 11.214, 11.766, 12.374,
    13.048, 13.796, 14.631, 15.566, 16.618, 17.810, 19.167, 20.721, 22.516, 24.605,
    27.058, 29.967, 33.457, 37.699, 42.930, 49.495, 57.902, 68.930, 83.823, 104.67,
]
_TF_CV = [
    18.656, 18.236, 17.826, 17.425, 17.033, 16.649, 16.275, 15.909, 15.551, 15.201,
    14.859, 14.524, 14.197, 13.878, 13.566, 13.260, 12.962, 12.670, 12.385, 12.107,
    11.834, 11.568, 11.308, 11.053, 10.804, 10.561, 10.324, 10.091, 9.864, 9.642,
    9.425, 9.213, 9.006, 8.803, 8.605, 8.412, 8.222, 8.037, 7.856, 7.680,
    7.507, 7.338, 7.173, 7.011, 6.854, 6.699, 6.549, 6.401, 6.257, 6.117,
    5.979, 5.844, 5.713, 5.584, 5.459, 5.336, 5.216, 5.098, 4.984, 4.872,
    4.762, 4.655, 4.550, 4.448, 4.348, 4.250, 4.154, 4.061, 3.969, 3.880,
    3.793, 3.707, 3.624, 3.542, 3.463, 3.385, 3.309, 3.234, 3.161, 3.090,
    3.021, 2.953, 2.886, 2.821, 2.758, 2.696, 2.635, 2.576, 2.518, 2.461,
    2.406, 2.352, 2.299, 2.247, 2.197, 2.147, 2.099, 2.052, 2.006, 1.96,
]


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


# ── Additional instrument-quality battery (Core + robustness) ─────────────────────────────────
# All FWL-partial the included controls X out first (via _resid), matching _battery / _fs_F, so the
# single-endogenous-regressor identification is on the same footing everywhere.

def _iv2sls_alpha(delta, spread, X, Z, codes, G):
    """FWL 2SLS α on instrument (sub)set Z, cluster-robust SE. α̂=(ŝ'δ̃)/(ŝ's̃), ŝ=Z̃π̂ (fitted
    endogenous). Cheap numpy path used in the leave-one-out loops; the point estimate equals the full
    2SLS α (matches _add_linearmodels' alpha_2sls). Returns (alpha, se) or (nan, nan)."""
    s  = _resid(spread.reshape(-1, 1), X).ravel()
    Zt = _resid(Z, X)
    dl = _resid(delta.reshape(-1, 1), X).ravel()
    pi = np.linalg.pinv(Zt.T @ Zt) @ (Zt.T @ s)
    shat = Zt @ pi
    denom = float(shat @ s)
    if abs(denom) < 1e-14:
        return (float("nan"), float("nan"))
    alpha = float(shat @ dl) / denom
    u = dl - alpha * s
    m = _group_sum((shat * u).reshape(-1, 1), codes, G).ravel()     # Σ_g ŝ_g u_g
    n = len(s)
    c = (G / (G - 1.0)) * ((n - 1.0) / max(n - X.shape[1] - 1, 1)) if G > 1 else 1.0
    var = c * float(m @ m) / denom ** 2
    return (alpha, float(np.sqrt(max(var, 0.0))))


def _tf_cv(F):
    """tF (LMMP 2022) 5% adjusted critical value at first-stage F, by linear interpolation of
    _TF_F/_TF_CV. cv=∞ for F≤3.8416 (asymptote → CI is the whole line); cv=1.96 for F≥104.67;
    table edge value for 3.8416<F<4.000 (flagged 'below_table' upstream)."""
    if F is None or not np.isfinite(F) or F <= _TF_FMIN:
        return float("inf")
    if F >= _TF_F[-1]:
        return 1.96
    if F <= _TF_F[0]:
        return _TF_CV[0]
    return float(np.interp(F, _TF_F, _TF_CV))


def _tf_adjust(eff_F, alpha, se):
    """Honest F-adjusted Wald CI for α: α̂ ± cv·se with cv=√c_{0.05}(F). At the observed eff-F≈4 the
    cv is huge (~18) so the CI is near-uninformative — that is the correct message. Returns tf_cv, the
    SE-inflation factor cv/1.96, the CI, and flags (tf_defined False when F below the 3.84 asymptote)."""
    cv = _tf_cv(eff_F)
    below = bool(eff_F is not None and np.isfinite(eff_F) and eff_F < _TF_F[0])
    if not np.isfinite(cv) or alpha is None or se is None or not np.isfinite(se):
        return {"tf_cv": (None if not np.isfinite(cv) else cv), "tf_defined": bool(np.isfinite(cv)),
                "tf_infl": (None if not np.isfinite(cv) else cv / 1.96),
                "tf_ci_low": None, "tf_ci_high": None, "below_table": below}
    return {"tf_cv": cv, "tf_defined": True, "tf_infl": cv / 1.96,
            "tf_ci_low": alpha - cv * se, "tf_ci_high": alpha + cv * se, "below_table": below}


def _liml_fuller(rec, delta, spread, X, Z, clusters):
    """Add LIML and Fuller(1) α̂ (cluster-robust) via linearmodels.IVLIML. The LIML−2SLS gap is the
    key weak-ID diagnostic (LIML is ~median-unbiased under weak ID; 2SLS is biased toward OLS).
    Fuller(1) restores finite moments. Graceful no-op if linearmodels is unavailable."""
    try:
        import pandas as pd
        from linearmodels.iv import IVLIML
    except Exception:
        return rec
    try:
        ex = pd.DataFrame(X, columns=["const"] + [f"x{i}" for i in range(X.shape[1] - 1)])
        en = pd.DataFrame({"spread": spread})
        iv = pd.DataFrame(Z, columns=[f"z{i}" for i in range(Z.shape[1])])
        dser, cl = pd.Series(delta, name="delta"), pd.Series(clusters)
        rl = IVLIML(dser, ex, en, iv).fit(cov_type="clustered", clusters=cl)
        rec["alpha_liml"] = float(rl.params["spread"]); rec["alpha_liml_se"] = float(rl.std_errors["spread"])
        rec["liml_kappa"] = float(getattr(rl, "kappa", np.nan))
        rf = IVLIML(dser, ex, en, iv, fuller=1).fit(cov_type="clustered", clusters=cl)
        rec["alpha_fuller"] = float(rf.params["spread"]); rec["alpha_fuller_se"] = float(rf.std_errors["spread"])
    except Exception as e:
        print(f"    [liml] failed ({type(e).__name__}: {e})")
    return rec


def _ols_dwh(rec, delta, spread, X, Z, codes, G):
    """Cluster-robust OLS α̂ and a control-function Durbin–Wu–Hausman endogeneity test (numpy, FWL).
    dwh_gap = OLS−2SLS shows the endogeneity/weak-IV drift. The CF t on the first-stage residual v̂
    IS the DWH statistic; p uses a t(G*) reference (few clusters). Flagged low-power under weak ID."""
    s  = _resid(spread.reshape(-1, 1), X).ravel()
    dl = _resid(delta.reshape(-1, 1), X).ravel()
    Zt = _resid(Z, X)
    n = len(s)
    pi = np.linalg.pinv(Zt.T @ Zt) @ (Zt.T @ s)
    vhat = s - Zt @ pi                                             # first-stage residual
    dOLS = float(s @ s)
    a_ols = float(s @ dl) / dOLS if dOLS > 0 else float("nan")
    u = dl - a_ols * s
    m = _group_sum((s * u).reshape(-1, 1), codes, G).ravel()
    c = (G / (G - 1.0)) * ((n - 1.0) / max(n - X.shape[1] - 1, 1)) if G > 1 else 1.0
    rec["alpha_ols"] = a_ols
    rec["alpha_ols_se"] = float(np.sqrt(max(c * float(m @ m) / dOLS ** 2, 0.0)))
    # control function: δ̃ = α·s̃ + γ·v̂ + e ; cluster-t on γ is the DWH test (b[0] ≈ alpha_2sls)
    W = np.column_stack([s, vhat])
    WtW_inv = np.linalg.pinv(W.T @ W)
    b = WtW_inv @ (W.T @ dl)
    e = dl - W @ b
    Mg = _group_sum(W * e[:, None], codes, G)
    Vb = c * (WtW_inv @ (Mg.T @ Mg) @ WtW_inv)
    gamma = float(b[1]); se_g = float(np.sqrt(max(Vb[1, 1], 0.0)))
    gstar = rec.get("eff_clusters") or G
    if se_g > 0:
        from scipy.stats import t as _t
        rec["dwh_cf_t"] = gamma / se_g
        rec["dwh_cf_p"] = float(2 * _t.sf(abs(gamma / se_g), df=max(gstar - 1, 1)))
    else:
        rec["dwh_cf_t"] = rec["dwh_cf_p"] = None
    a2 = rec.get("alpha_2sls")
    rec["dwh_gap"] = (a_ols - a2) if a2 is not None else None
    rec["dwh_lowpower"] = True
    return rec


def _condnum_vif(Zt, iv_names=None):
    """Collinearity of the (FWL-partialled) excluded instruments: correlation-matrix condition number,
    eigenvalue spectrum, and per-instrument VIF = diag(corr⁻¹). Turns the known mean_loo VIF≈707 into
    engine output. Reliable (depends only on Z̃, not on the rank-deficient cluster cov)."""
    sd = Zt.std(0)
    keep = sd > 1e-12
    Zk = Zt[:, keep]
    names = list(iv_names) if iv_names is not None else [f"z{i}" for i in range(Zt.shape[1])]
    kept = [names[i] for i in range(len(names)) if i < len(keep) and keep[i]]
    if Zk.shape[1] < 2:
        return {"cond_number": 1.0, "min_eig": 1.0, "max_vif": 1.0, "vif": {}, "eigenvalues": []}
    C = np.corrcoef(Zk, rowvar=False)
    eig = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    pos = eig[eig > 1e-12]
    cond = float(np.sqrt(eig.max() / pos.min())) if pos.size else float("inf")
    vif = np.clip(np.diag(np.linalg.pinv(C)), 0.0, None)
    return {"cond_number": cond, "min_eig": float(pos.min()) if pos.size else 0.0,
            "max_vif": float(np.max(vif)), "vif": {kept[i]: float(vif[i]) for i in range(len(kept))},
            "eigenvalues": [float(x) for x in np.sort(eig)[::-1]]}


def _per_iv_diag(delta, spread, X, Z, codes, G, iv_names):
    """Per-instrument first-stage diagnostics. Homoskedastic π̂ t-stats + signs are well defined and
    reported. Cluster-robust per-IV t's are RANK-DEFICIENT when K>G (K=15 > G*≈5-7 here) — they are
    computed but flagged 'cluster_rank_deficient' and should not be trusted. delta is unused (kept for
    a uniform signature)."""
    s  = _resid(spread.reshape(-1, 1), X).ravel()
    Zt = _resid(Z, X)
    n, K = Zt.shape
    ZtZ_inv = np.linalg.pinv(Zt.T @ Zt)
    pi = ZtZ_inv @ (Zt.T @ s)
    vhat = s - Zt @ pi
    dfv = max(n - X.shape[1] - K, 1)
    s2 = float(vhat @ vhat) / dfv
    se_h = np.sqrt(np.clip(s2 * np.diag(ZtZ_inv), 0.0, None))
    t_h = np.where(se_h > 0, pi / np.where(se_h > 0, se_h, 1.0), np.nan)
    Mv = _group_sum(Zt * vhat[:, None], codes, G)
    c = (G / (G - 1.0)) * ((n - 1.0) / dfv) if G > 1 else 1.0
    se_c = np.sqrt(np.clip(np.diag(c * (ZtZ_inv @ (Mv.T @ Mv) @ ZtZ_inv)), 0.0, None))
    t_c = np.where(se_c > 0, pi / np.where(se_c > 0, se_c, 1.0), np.nan)
    return {"names": list(iv_names), "pi": [float(x) for x in pi],
            "t_homosk": [float(x) for x in t_h], "signs": [int(np.sign(x)) for x in pi],
            "t_cluster": [float(x) for x in t_c], "cluster_rank_deficient": bool(K >= G),
            "n_pos": int(np.sum(pi > 0)), "n_neg": int(np.sum(pi < 0))}


def _loo_sensitivity(delta, spread, X, Zall, iv_names, codes, G):
    """Leave-one-instrument-out and leave-one-GROUP-out sensitivity of α̂ and the first stage. The
    drop-mean_loo-block row is the headline (that block is the collinear one). Reuses _fs_F + the numpy
    2SLS. Δ's are vs the full-set baseline."""
    base = _fs_F(spread, X, Zall, codes, G)
    a0, _ = _iv2sls_alpha(delta, spread, X, Zall, codes, G)

    def _one(cols):
        Zc = Zall[:, cols]
        if Zc.shape[1] == 0:
            return None
        f = _fs_F(spread, X, Zc, codes, G)
        a, se = _iv2sls_alpha(delta, spread, X, Zc, codes, G)
        return {"alpha": a, "alpha_se": se, "eff_F": f["eff_F"], "kp_F": f["kp_F"],
                "partial_R2": f["partial_R2"], "n_iv": f["n_iv"],
                "dAlpha": (a - a0) if (a is not None and a0 is not None
                                       and np.isfinite(a) and np.isfinite(a0)) else None,
                "dEffF": (f["eff_F"] - base["eff_F"]) if (f["eff_F"] and base["eff_F"]) else None}

    drop_one = {name: _one([i for i in range(len(iv_names)) if i != j])
                for j, name in enumerate(iv_names)}
    drop_group = {}
    for gname, members in IV_GROUPS.items():
        cols = [i for i, name in enumerate(iv_names) if name not in members]
        if 0 < len(cols) < len(iv_names):
            drop_group[gname] = _one(cols)
    return {"base": {"alpha": a0, "eff_F": base["eff_F"], "kp_F": base["kp_F"],
                     "partial_R2": base["partial_R2"]},
            "drop_one": drop_one, "drop_group": drop_group}


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
        # drop zero-variance instruments and collinear controls in this subsample; keep the surviving
        # instrument NAMES aligned to the surviving Z columns (needed by the per-IV / LOO diagnostics).
        zmask = Z.std(0) > 1e-10
        Z = Z[:, zmask]
        iv_kept = [iv_avail[i] for i in range(len(iv_avail)) if zmask[i]]
        X = X[:, np.r_[True, X[:, 1:].std(0) > 1e-10]] if X.shape[1] > 1 else X
        if len(s) <= X.shape[1] + Z.shape[1] + 1 or Z.shape[1] == 0:
            print(f"[weak-IV] E{k}/{name}: too few obs / no IVs — skipped"); continue
        uniq, codes = np.unique(cl, return_inverse=True)
        G = len(uniq)
        rec = _battery(d, s, X, Z, codes, G)
        rec = _add_linearmodels(rec, d, s, X, Z, cl)
        # Parsimonious subset (rival characteristics only): dilution / many-weak-instruments check.
        pcols = [i for i, c in enumerate(iv_kept) if c in PARSIMONIOUS_IV]
        if 0 < len(pcols) < Z.shape[1]:
            rec["parsimonious"] = _fs_F(s, X, Z[:, pcols], codes, G)
        # ── Core + robustness battery: estimator ladder, tF honest CI, LOO sensitivity, collinearity ──
        rec = _liml_fuller(rec, d, s, X, Z, cl)
        rec = _ols_dwh(rec, d, s, X, Z, codes, G)
        rec["tf"] = _tf_adjust(rec.get("effective_F"), rec.get("alpha_2sls"), rec.get("alpha_se"))
        rec["z_collinearity"] = _condnum_vif(_resid(Z, X), iv_kept)
        rec["per_iv"] = _per_iv_diag(d, s, X, Z, codes, G, iv_kept)
        rec["loo"] = _loo_sensitivity(d, s, X, Z, iv_kept, codes, G)
        rec["sw_note"] = "SW conditional F == first-stage F (single endogenous regressor)"
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
        # Second line: the estimator ladder + tF honest CI + instrument-set collinearity.
        tf = rec.get("tf") or {}; zc = rec.get("z_collinearity") or {}
        def _f(x, d=3): return f"{x:+.{d}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "NA"
        if tf.get("tf_defined") and tf.get("tf_ci_low") is not None:
            tfci = f"[{_f(tf['tf_ci_low'], 2)},{_f(tf['tf_ci_high'], 2)}]"
        elif not tf.get("tf_defined"):
            tfci = "(undef, F<3.84)"
        else:
            tfci = "NA"
        cvv = tf.get("tf_cv")
        cvs = f"{cvv:.1f}" if isinstance(cvv, (int, float)) and np.isfinite(cvv) else "inf"
        mv, cn = zc.get("max_vif"), zc.get("cond_number")
        coll = (f"maxVIF={mv:,.0f} cond#={cn:,.0f}"
                if isinstance(mv, (int, float)) and isinstance(cn, (int, float)) else "collinearity NA")
        print(f"           ladder OLS={_f(rec.get('alpha_ols'))} 2SLS={_f(a)} "
              f"LIML={_f(rec.get('alpha_liml'))} Fuller={_f(rec.get('alpha_fuller'))} | "
              f"tF95 {tfci} (cv={cvs}) | {coll}")
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
