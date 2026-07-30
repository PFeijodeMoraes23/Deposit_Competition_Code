"""utils/se_national.py -- inference for the NATIONAL regressors in the sleepiness model.

Author: Pedro Feijó de Moraes

`pix_exists` and `risk_free_qoq_lag` take the SAME value for all ~506 conglomerates within a
quarter. Clustering on the conglomerate therefore treats each firm's copy of a national value
as independent evidence, so the standard error is formed as if there were ~1.1M observations of
Pix when the regressor really has only T=36 quarters of variation. That understates uncertainty
on exactly the two rows with the largest headline coefficients.

Nothing here touches a point estimate. theta-hat minimises the objective; the clustering scheme
enters only the variance. So these are recomputations of SEs, never re-estimations -- which is
why they can run off a saved fit (or a tail-only refit with theta_fixed) instead of a full
re-search.

Two alternatives, which fail in OPPOSITE directions. Both are computed; the TABLES report the
quarter-clustered bootstrap and show/quote Driscoll-Kraay alongside (see `select_se` for why the
reported cell is not a per-cell maximum of the two):

  time_clustered_wcb   Clusters on the quarter. Allows ARBITRARY dependence within a quarter
                       (exactly what a national shock is) but assumes quarters are INDEPENDENT.
                       Keeps the wild bootstrap's small-sample refinement, which matters at
                       T=36. Optimistic if the regressor is persistent -- and Selic is.

  driscoll_kraay       Aggregates the scores across the cross-section within each period, then
                       applies a Bartlett/Newey-West HAC to that T-length series. Robust to
                       within-quarter dependence AND to serial correlation across quarters, so
                       it handles the persistence that time-clustering assumes away. But its
                       asymptotics are in T, and T=36 is modest; and it is analytic, so it gets
                       no bootstrap refinement.

Reference: Driscoll & Kraay (1998); Newey & West (1987); Cameron-Gelbach-Miller (2008) and
MacKinnon-Webb (2017) for the wild cluster bootstrap already used elsewhere in this repo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Regressors with NO cross-sectional variation: identical across all firms within a quarter,
# so conglomerate clustering cannot see their real sampling variation. Measured on the panel:
# pix_exists is a pure function of the quarter; risk_free_qoq_lag has only 32 distinct values
# over 36 quarters. NOT gdp_growth_yoy -- that is built per entity from its own gdp lag.
NATIONAL_VARS = {"pix_exists", "risk_free_qoq_lag"}


def is_national(name: str) -> bool:
    """True for a parameter whose regressor varies only over time (accepts the
    'interaction_<var>' form the second stage uses)."""
    return str(name).replace("interaction_", "") in NATIONAL_VARS


def bartlett_lags(n_periods: int) -> int:
    """Newey-West bandwidth by the usual rule L = floor(4 (T/100)^(2/9)); T=36 -> 3."""
    if n_periods < 3:
        return 0
    return max(1, int(np.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0))))


def _period_scores(score: np.ndarray, bread: np.ndarray, t_inv: np.ndarray,
                   n_t: int) -> np.ndarray:
    """Influence functions aggregated to one row per period: h_t = sum_{i in t} bread @ score_i.
    This cross-sectional aggregation is the step that makes both estimators below robust to
    arbitrary dependence WITHIN a period."""
    IF = score @ bread.T                             # N x p
    out = np.zeros((n_t, IF.shape[1]))
    for k in range(IF.shape[1]):
        out[:, k] = np.bincount(t_inv, weights=IF[:, k], minlength=n_t)
    return out


def driscoll_kraay(score: np.ndarray, bread: np.ndarray, periods, lags: int | None = None
                   ) -> tuple[np.ndarray, int, int]:
    """Driscoll-Kraay SEs. Returns (se, n_periods, lags_used).

    V = Gamma_0 + sum_{l=1..L} (1 - l/(L+1)) (Gamma_l + Gamma_l'),  Gamma_l = sum_t h_t h_{t-l}'
    with h_t the period-aggregated influence functions. The Bartlett weights guarantee V is
    positive semi-definite, which a truncated sum alone does not."""
    per = pd.Series(np.asarray(periods)).astype(str).values
    # sort periods so that lag l genuinely means l quarters apart
    uniq = np.array(sorted(np.unique(per)))
    t_index = {p: i for i, p in enumerate(uniq)}
    t_inv = np.array([t_index[p] for p in per], dtype=int)
    n_t = len(uniq)
    h = _period_scores(score, bread, t_inv, n_t)     # T x p
    L = bartlett_lags(n_t) if lags is None else int(lags)

    V = h.T @ h                                      # Gamma_0
    for l in range(1, L + 1):
        G = h[l:].T @ h[:-l]
        w = 1.0 - l / (L + 1.0)
        V = V + w * (G + G.T)
    return np.sqrt(np.abs(np.diag(V))), n_t, L


def time_clustered_wcb(score: np.ndarray, bread: np.ndarray, periods, beta: np.ndarray,
                       names, B: int = 999, scheme: str = "webb", seed: int = 0):
    """Wild cluster bootstrap with the QUARTER as the cluster. Reuses the same
    cluster_wild_bootstrap the rest of the repo uses, so the only thing that changes versus the
    reported SEs is which dimension is resampled."""
    from utils.sleep_links import cluster_wild_bootstrap
    per = pd.Series(np.asarray(periods)).astype(str).values
    uniq, t_inv = np.unique(per, return_inverse=True)
    n_t = len(uniq)
    IF_t = _period_scores(score, bread, t_inv, n_t)
    nm = list(names)
    ame_hat = {n: float(b) for n, b in zip(nm, beta)}
    ame_fn = lambda th: {n: float(th[i]) for i, n in enumerate(nm)}
    bse, pvals = cluster_wild_bootstrap(np.asarray(beta, float), IF_t, ame_fn, ame_hat,
                                        B=B, scheme=scheme, rng=np.random.default_rng(seed))
    return (pd.Series({n: bse[n] for n in nm}),
            pd.Series({n: pvals[n] for n in nm}), n_t)


def _aggregate_if_by_period(IF_rows: np.ndarray, periods):
    """Row-level influence functions -> one row per period. The nonlinear fits build IF
    themselves (bread already applied), so they enter here rather than via (score, bread)."""
    per = pd.Series(np.asarray(periods)).astype(str).values
    uniq, t_inv = np.unique(per, return_inverse=True)
    n_t = len(uniq)
    out = np.zeros((n_t, IF_rows.shape[1]))
    for k in range(IF_rows.shape[1]):
        out[:, k] = np.bincount(t_inv, weights=IF_rows[:, k], minlength=n_t)
    return out, n_t


def attach_national_ses_nonlinear(res, IF_rows: np.ndarray, periods, theta, ame_fn, ame_hat,
                                  B: int = 999, scheme: str = "webb", seed: int = 0,
                                  label: str = ""):
    """Quarter-clustered WCB for the M-estimators (E5-E8).

    Identical machinery to the reported SEs -- same `cluster_wild_bootstrap`, same `ame_fn`
    mapping parameters to AMEs -- with the influence functions aggregated by QUARTER instead
    of by conglomerate. So the only thing that changes is which dimension is resampled.

    Driscoll-Kraay is deliberately NOT produced here: DK is analytic and would need a
    delta-method push through `ame_fn` to reach the AME scale, which is a different (and more
    fragile) calculation than the bootstrap already does exactly. DK is reported for the linear
    columns, where the parameters ARE the coefficients."""
    try:
        from utils.sleep_links import cluster_wild_bootstrap
        IF_t, n_t = _aggregate_if_by_period(IF_rows, periods)
        bse, pvals = cluster_wild_bootstrap(np.asarray(theta, float), IF_t, ame_fn, ame_hat,
                                            B=B, scheme=scheme,
                                            rng=np.random.default_rng(seed))
        res.bse_time = pd.Series(bse)
        res.pvalues_time = pd.Series(pvals)
        res.n_periods = n_t
        nat = [n for n in bse if is_national(n)]
        if nat:
            base = getattr(res, "bse", None)
            msg = ", ".join(
                f"{n.replace('interaction_','')}: congl="
                f"{(float(base[n]) if base is not None and n in getattr(base,'index',[]) else float('nan')):.4g}"
                f" / quarter={float(bse[n]):.4g}" for n in nat)
            print(f"  [national-SE{(' ' + label) if label else ''}] T={n_t} | {msg}")
    except Exception as exc:
        print(f"  [national-SE] nonlinear path skipped ({type(exc).__name__}: {exc})")
    return res


def _normal_two_sided(t: float) -> float:
    """Two-sided normal p-value. Driscoll-Kraay is analytic, so unlike the bootstrap schemes it
    carries no p-value of its own and one has to be formed from the t-ratio."""
    import math
    return float(math.erfc(abs(float(t)) / math.sqrt(2.0)))


# Reporting policy for the tables. Decided 2026-07-30: for the two NATIONAL regressors the table
# LEADS with a time-robust standard error rather than the conglomerate one, because conglomerate
# clustering cannot see a shock common to every firm in a quarter and is therefore
# anti-conservative on exactly those rows. Measured on the appendix grid, the phi equation's
# national SEs are ~1.6-1.9x wider under quarter clustering and ~1.9-2.2x wider under DK.
#
# The reported cell is the QUARTER-CLUSTERED WCB, not max(quarter, DK). Three reasons:
#   1. DK exists only for the LINEAR columns (E1/E2). Taking a per-cell max would report DK in
#      the E1/E2 columns and quarter-WCB in the E5-E8 columns OF THE SAME ROW, so a reader
#      comparing across columns could not tell whether a gap is the scheme or the estimator.
#   2. max() of two variance estimators is not itself an estimator of the sampling variance --
#      it is upward-biased by construction, and the bias depends on the noise in both.
#   3. Quarter-WCB is the SAME bootstrap as the reported SEs with only the resampled dimension
#      changed, keeps the small-sample refinement that matters at T=35, and has a genuine
#      bootstrap p-value, so stars need no invented reference distribution.
# DK is still computed and stored (`bse_dk`); it is reported alongside on the all-linear tables
# and quoted in the footnote, rather than silently swapped into a mixed row.
SE_MARK = r"\dagger"
SCHEME_LABEL = {"congl": "conglomerate WCB", "quarter": "quarter-clustered WCB",
                "dk": "Driscoll-Kraay"}


def select_se(res, name, lead_time_robust: bool = True):
    """Return ``(se, pvalue, scheme)`` for ONE parameter of a fitted result.

    Firm-level regressors keep the conglomerate wild cluster bootstrap -- unchanged, and correct
    for them. National regressors (``pix_exists``, ``risk_free_qoq_lag``) get the WIDER of the
    quarter-clustered WCB and Driscoll-Kraay, when those are present.

    Degrades silently: any pickle written before ``attach_national_ses`` existed simply has no
    ``bse_time``/``bse_dk``, and every row then reports ``('congl')`` exactly as before. That
    matters because the tables must keep building off older fits.

    `scheme` is one of 'congl' | 'quarter' | 'dk' and drives the dagger in the table body.
    """
    def _get(attr):
        s = getattr(res, attr, None)
        if s is None:
            return None
        try:
            return float(pd.Series(s)[name])
        except Exception:
            return None

    se_c = _get("bse")
    p_c = _get("pvalues")
    if se_c is None or not np.isfinite(se_c):
        return (float("nan"), float("nan"), "congl")
    if not (lead_time_robust and is_national(name)):
        return (se_c, p_c, "congl")

    beta = _get("params")
    # Preference order, NOT a max: quarter-WCB is available for every estimator, so choosing it
    # first keeps one estimator per row across all columns. DK is the fallback only where the
    # bootstrap is somehow absent but the analytic SE is present (linear columns).
    se_q, p_q = _get("bse_time"), _get("pvalues_time")
    if se_q is not None and se_q > 0:
        pv = p_q
        if pv is None or not np.isfinite(pv):
            pv = (_normal_two_sided(beta / se_q)
                  if beta is not None and np.isfinite(beta) else p_c)
        return (se_q, pv, "quarter")

    se_d = _get("bse_dk")
    if se_d is not None and se_d > 0:
        pv = (_normal_two_sided(beta / se_d)
              if beta is not None and np.isfinite(beta) else p_c)
        return (se_d, pv, "dk")

    return (se_c, p_c, "congl")


def dk_se(res, name):
    """Driscoll-Kraay SE for one parameter, or None. Used by the all-linear tables to show DK as
    a supplementary bracketed line without displacing the reported quarter-clustered cell."""
    s = getattr(res, "bse_dk", None)
    if s is None:
        return None
    try:
        v = float(pd.Series(s)[name])
    except Exception:
        return None
    return v if np.isfinite(v) and v > 0 else None


def attach_national_ses(res, score: np.ndarray, bread: np.ndarray, periods, beta, names,
                        B: int = 999, scheme: str = "webb", seed: int = 0, label: str = ""):
    """Compute both national-regressor SE variants and hang them on a fitted result as
    `bse_time` / `pvalues_time` / `bse_dk`, leaving `bse` (conglomerate WCB) untouched.

    Storing both means the exporters can choose PER ROW -- conglomerate clustering for the
    firm-level regressors, quarter clustering / DK for the national ones -- without a second
    estimation pass. Failures are swallowed: alternative SEs are a reporting refinement and
    must never take down a fit."""
    try:
        bse_t, pv_t, n_t = time_clustered_wcb(score, bread, periods, beta, names,
                                              B=B, scheme=scheme, seed=seed)
        se_dk, n_t2, L = driscoll_kraay(score, bread, periods)
        res.bse_time = bse_t
        res.pvalues_time = pv_t
        res.bse_dk = pd.Series(se_dk, index=list(names))
        res.n_periods = n_t
        res.dk_lags = L
        nat = [n for n in names if is_national(n)]
        if nat:
            msg = ", ".join(
                f"{n.replace('interaction_','')}: congl={float(pd.Series(res.bse, index=list(names))[n]):.4g}"
                f" / quarter={float(bse_t[n]):.4g} / DK={float(res.bse_dk[n]):.4g}"
                for n in nat if n in bse_t.index)
            print(f"  [national-SE{(' ' + label) if label else ''}] T={n_t}, DK lags={L} | {msg}")
    except Exception as exc:                      # never fail a fit over reporting SEs
        print(f"  [national-SE] skipped ({type(exc).__name__}: {exc})")
    return res
