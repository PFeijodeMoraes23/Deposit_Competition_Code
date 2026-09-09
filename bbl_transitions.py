"""
bbl_transitions.py
==================
BBL Step 1, second half: the STATE TRANSITION PROCESSES.

Bajari, Benkard & Levin (2007) need two first-stage objects before the forward simulation:
a policy function AND a law of motion for the state. `bbl_polfunc.py` estimates the policy
(the pricing rule for k=4,5). This file estimates the transitions, which until 2026-09 did
not exist anywhere in the pipeline: the forward simulation held every state except the
deposit stock frozen at its starting value, and moved the risk-free rate along a single
deterministic Focus curve that never even reached the deposit path.

Three blocks, matching the state partition of V_Main sec:empirical:cost:

  1. EXOGENOUS MACRO — the short rate r^f_t (national, quarterly).

     THE PROCESS THAT IS USED: **the Focus curve is the conditional MEAN and a mean-zero
     AR(1) supplies the dispersion around it**,

         r_{t+h} = focus_v(h) + d_h,     d_h = phi_d * d_{h-1} + sigma_d * eps_h,   d_0 = 0,

     truncated at zero. Starting the deviation at d_0 = 0 is the right boundary condition:
     the rate at the launch quarter is OBSERVED, so uncertainty must start at nothing and
     widen with the horizon. Var(d_h) = sigma_d^2 (1-phi_d^(2h))/(1-phi_d^2) does exactly
     that, and (phi_d, sigma_d) are estimated from the FOCUS FORECAST ERRORS -- realised
     Selic minus what the survey published h quarters earlier -- which is the object that
     actually measures how wrong the market's expected path turns out to be.

     WHY NOT A PURE ESTIMATED CIR (measured 2026-09-09; kept below as a diagnostic).
     Fitting CIR to the 52 quarters of Brazilian Selic in this panel gives kappa = 0.0201
     with a standard error of 0.0370 -- insignificant, so theta is unidentified -- and a
     long-run mean of 18.2% annualised, ABOVE the sample maximum of 15.92%. Simulating it
     produces a mean path that RISES from 15.7% to 17.7% over 50 quarters and a 95th
     percentile of 33%, contradicting the Focus curve, which glides down to about 10%. Fifty
     two quarters cannot identify mean reversion in a series with AR(1) persistence of 0.97.
     This is why Egan et al. (2025) App. B.1 CALIBRATE (US fed funds: kappa=.11, theta=.03,
     sigma=.08 annual) rather than estimate. Anchoring the mean to Focus keeps the paper's
     market-expectations design and confines estimation to the dispersion, which the data
     can actually identify. The CIR and AR(1) fits are still reported in the JSON, flagged
     `usable_as_mean: false`, so the diagnosis is on the record rather than in a comment.

  2. EXOGENOUS FIRM — the cost shifters Z_jt that carry gamma in the BBL cost equation
     (personnel / admin / tax cost ratios and the Basel index). Firm-level accounting
     ratios, so they are collapsed to firm x quarter before anything is estimated. Modelled
     as mean-reverting to a FIRM-SPECIFIC level, which is what gamma'Z needs:

         z_{i,t+1} = (1-rho)*zbar_i + rho*z_{i,t} + u

     rho is the pooled within-firm OLS slope, estimated separately for B and D firms.

  3. ENDOGENOUS-ISH MARKET — the demographic/technology states S_mt that drive BOTH the
     sleepiness index (and hence phi) and the BLP demographic interactions (and hence the
     active share). Same mean-reverting form, within MCA. `pix_exists` is absorbing rather
     than stochastic and is handled as a deterministic step.

WHY BLOCK 3 IS EXPENSIVE, AND WHY IT IS STILL DONE. The S_mt variables are the same columns
the demand system uses as demographics d_imt. Letting them evolve makes mu_ijkmt -- and so
the active share -- time-varying, which removes the constant-spread fast path in
`simulate_deposits` (one share evaluation per simulation becomes T of them). That is a ~50x
increase in the dominant cost of the forward simulation. It is done anyway because the
alternative is a forward simulation whose demand block contradicts its own state block.

WHAT IS DELIBERATELY NOT MODELLED
  * The private cost shock chi: dropped from the BBL moments (V_Main), so it has no path.
  * Rival policies: frozen at sigma-hat BY CONSTRUCTION -- that is the unilateral-deviation
    content of the BBL inequality, not an omission.
  * Entry and exit: the firm set is fixed over the horizon.

OUTPUT (small, by design -- see MINIMAL UPLOAD below)
  COST_FWD/bbl_transitions.json    all estimated parameters + sample metadata

MINIMAL UPLOAD. The JSON is a few kB and is the ONLY new artifact the cluster needs: the
rate paths themselves are regenerated on the compute node from these parameters and a fixed
seed, so nothing large is ever staged. `--simulate` writes a paths CSV and `--plot` an
overlay, both for LOCAL inspection only; neither is an input to anything.

USAGE
  python bbl_transitions.py                          # estimate, write the JSON
  python bbl_transitions.py --simulate 50 --plot     # + 50 local paths and an overlay png
  python bbl_transitions.py --rate-only              # skip the panel blocks (fast)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils import paths  # noqa: E402

try:
    from utils.winsorize import winsorize_within_type as _winsorize_within_type
except Exception:  # pragma: no cover - the fallback keeps the script runnable standalone
    _winsorize_within_type = None

# The cost shifters that carry gamma in eq:8. Kept in ONE place here; the loader below
# fails loudly rather than silently dropping a column, because a missing shifter would
# quietly change what gamma means.
Z_COLS = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag",
          "tax_cost_ratio_lag", "indice_basileia_lag"]

# Market states. These drive phi through the sleepiness index AND enter the BLP
# demographics, which is exactly why block 3 is expensive (see the module docstring).
S_COLS = ["gdp_per_capita", "cadunico_families_per1000", "fraction_65plus",
          "fraction_young", "connections_per100"]

RATE_COL = "risk_free_qoq"          # quarterly decimal, national
FIRM_KEY = "CodConglomeradoPrudencial"
MKT_KEY = "mca_code"


# ---------------------------------------------------------------------------
# 1. Short rate: CIR (+ AR(1) alternative)
# ---------------------------------------------------------------------------
def estimate_cir(r: np.ndarray) -> dict:
    """CIR by the CKLS transformation. `r` is the quarterly rate in levels, ordered in time
    and gap-free. Returns quarterly parameters plus their annualised counterparts."""
    r = np.asarray(r, float)
    r0, r1 = r[:-1], r[1:]
    ok = np.isfinite(r0) & np.isfinite(r1) & (r0 > 0)
    r0, r1 = r0[ok], r1[ok]
    n = len(r0)
    if n < 8:
        raise ValueError(f"CIR needs >=8 usable transitions, got {n}")

    s = np.sqrt(r0)
    y = (r1 - r0) / s                    # homoskedastic innovation
    Xd = np.column_stack([1.0 / s, -s])  # coefficients are (kappa*theta, kappa)
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    kt, kappa = float(b[0]), float(b[1])
    theta = kt / kappa if kappa != 0 else float("nan")
    resid = y - Xd @ b
    sigma = float(np.std(resid, ddof=2))

    # OLS SEs on the transformed (homoskedastic) regression
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.diag(XtX_inv) * sigma ** 2)

    half_life = float(np.log(2.0) / kappa) if kappa > 0 else float("inf")
    return {
        "kappa_q": kappa, "theta_q": theta, "sigma_q": sigma,
        "kappa_se_q": float(se[1]), "kappa_theta_se_q": float(se[0]),
        # annualised for comparison with Egan et al.'s US calibration (.11/.03/.08)
        "kappa_ann": kappa * 4.0,
        "theta_ann": (1.0 + theta) ** 4 - 1.0 if np.isfinite(theta) else float("nan"),
        "sigma_ann": sigma * 2.0,          # sigma scales with sqrt(dt)
        "half_life_quarters": half_life,
        "feller_2kt_ge_s2": bool(2 * kappa * theta >= sigma ** 2),
        "feller_margin": float(2 * kappa * theta - sigma ** 2),
        "n_transitions": int(n),
    }


def estimate_ar1_level(r: np.ndarray) -> dict:
    """AR(1) on the level: r_{t+1} = a + rho*r_t + u. Reported alongside CIR."""
    r = np.asarray(r, float)
    r0, r1 = r[:-1], r[1:]
    ok = np.isfinite(r0) & np.isfinite(r1)
    r0, r1 = r0[ok], r1[ok]
    Xd = np.column_stack([np.ones_like(r0), r0])
    b, *_ = np.linalg.lstsq(Xd, r1, rcond=None)
    resid = r1 - Xd @ b
    rho = float(b[1])
    return {
        "a": float(b[0]), "rho": rho,
        "sigma_u": float(np.std(resid, ddof=2)),
        "long_run_mean": float(b[0] / (1 - rho)) if rho < 1 else float("nan"),
        "half_life_quarters": float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else float("inf"),
        "n_transitions": int(len(r0)),
    }


def estimate_focus_error_process(vintages_csv: str, rate_by_t: pd.Series) -> dict:
    """The deviation process, estimated from FOCUS FORECAST ERRORS.

    `vintages_csv` is the long file `scrape_forward_rf.py --vintage-from/--to` writes:
    columns `start_q, vintage_date, h, cal_q, rf_qoq, ...`, one row per (vintage, horizon).
    For every vintage v and horizon h whose target quarter is inside the sample we form

        e_{v,h} = r_realised(target quarter)  -  rf_qoq predicted by vintage v at horizon h

    and estimate an AR(1) on e WITHIN vintage (consecutive horizons only):

        e_{v,h} = phi_d * e_{v,h-1} + u

    phi_d is the pooled within-vintage slope and sigma_d the residual sd. The empirical sd of
    e by horizon is returned too, so the simulated Var(d_h) can be checked against the
    forecast errors the survey actually made instead of being asserted.
    """
    v = pd.read_csv(vintages_csv)
    need = {"start_q", "h", "rf_qoq"}
    if not need.issubset(v.columns):
        raise ValueError(f"{vintages_csv} is missing {sorted(need - set(v.columns))}")
    v = v[v["h"] >= 1].copy()

    def _q_to_t(q: str) -> float:
        try:
            y, qq = str(q).upper().split("Q")
            return int(y) * 4 + int(qq)
        except Exception:
            return np.nan

    v["t0"] = v["start_q"].map(_q_to_t)
    v["t_target"] = v["t0"] + v["h"]
    v = v.dropna(subset=["t0"])
    v["r_real"] = v["t_target"].map(rate_by_t)
    v = v.dropna(subset=["r_real", "rf_qoq"])
    if len(v) < 40:
        raise ValueError(f"only {len(v)} in-sample (vintage, horizon) pairs -- too few")
    v["e"] = v["r_real"] - v["rf_qoq"]

    v = v.sort_values(["start_q", "h"])
    v["e_lag"] = v.groupby("start_q")["e"].shift(1)
    v["dh"] = v.groupby("start_q")["h"].diff()
    m = v["e_lag"].notna() & (v["dh"] == 1)
    x = v.loc[m, "e_lag"].to_numpy(float)
    y = v.loc[m, "e"].to_numpy(float)
    phi_d = float((x @ y) / (x @ x)) if (x @ x) > 0 else float("nan")
    resid = y - phi_d * x
    sigma_d = float(np.std(resid, ddof=1))

    by_h = (v.groupby("h")["e"]
             .agg(["count", "mean", "std"])
             .rename(columns={"count": "n", "mean": "bias", "std": "sd"}))
    sd_raw = {int(h): float(r["sd"]) for h, r in by_h.iterrows() if np.isfinite(r["sd"])}
    n_by_h = {int(h): int(r["n"]) for h, r in by_h.iterrows()}
    bias_by_h = {int(h): float(r["bias"]) for h, r in by_h.iterrows() if np.isfinite(r["bias"])}

    # THE RAW TERM STRUCTURE IS NOT USABLE PAST THE POINT THE SAMPLE THINS, and the failure is
    # a composition artifact rather than economics. A horizon-h error needs a target quarter
    # inside the panel, so only the EARLIEST vintages reach long h: measured here, n falls
    # 36, 32, 28, 24, 20, 16, 12, 8, 5 at h = 1, 8, 12, 16, 20, 24, 28, 32, 35. The sd rises to
    # 0.01363 at h=16 and then COLLAPSES to 0.0043-0.0069 for h >= 24 -- at that point a handful
    # of overlapping windows are all forecasting the same 2022-24 realisation, so the statistic
    # describes one episode, not forecast uncertainty. Taking it at face value would tell the
    # simulation that a six-year-ahead Selic forecast is THREE TIMES more accurate than a
    # four-year-ahead one.
    #
    # Two corrections, both stated rather than hidden:
    #   (a) monotone non-decreasing in h -- a longer forecast cannot be more accurate;
    #   (b) beyond the last horizon with at least MIN_OBS observations, hold the sd flat.
    # beta = 0.9 puts about 90% of the discount weight inside the first ~22 quarters, so this
    # governs the tail rather than the headline, but the tail is exactly what rate risk is for.
    MIN_OBS = 20
    ks = sorted(sd_raw)
    well_sampled = [h for h in ks if n_by_h.get(h, 0) >= MIN_OBS]
    h_cut = max(well_sampled) if well_sampled else (max(ks) if ks else 0)
    sd_by_h, run = {}, 0.0
    for h in ks:
        run = max(run, sd_raw[h]) if h <= h_cut else run   # (a) rising, then (b) held flat
        sd_by_h[h] = run

    # WHY THE DEFAULT IS NOT THE AR(1). Fitting e_h on e_{h-1} pooled over vintages returns
    # phi_d = 1.031 on this data -- EXPLOSIVE, so simulating it diverges (measured: a 95th
    # percentile of 46% annualised at h=50 and a mean path drifting AWAY from Focus). The
    # reason is visible in the term structure: the error sd runs 0.0011, 0.0063, 0.0121 at
    # h = 1, 4, 8 and then FLATTENS at about 0.0118. Growing 5.8x from h=1 to h=4 is far
    # faster than the 2x a random walk allows, so no AR(1) -- stationary or not -- reproduces
    # it; the pooled regression just returns a near-unit root and overshoots. That shape is a
    # persistent regime surprise that takes a few quarters to show up and then saturates, not
    # accumulating iid noise.
    #
    # So the default is a HORIZON-SCALED LEVEL SHOCK: d_h = sd(h) * z, one standard normal z
    # per simulated path. It matches the empirical Var(d_h) at every horizon BY CONSTRUCTION
    # and gives a path-level surprise that persists, which is what the near-unit pooled AR(1)
    # was trying (and failing) to express. Its one strong assumption -- deviations perfectly
    # correlated across horizons within a path -- is the honest upper bound on persistence and
    # is stated here rather than buried.
    explosive = not np.isfinite(phi_d) or phi_d >= 0.995
    return {
        "mode": "focus_mean_plus_horizon_shock" if explosive else "focus_mean_plus_ar1_deviation",
        "sd_by_h": sd_by_h,
        "phi_d": phi_d,
        "sigma_d": sigma_d,
        "ar1_rejected": bool(explosive),
        "ar1_rejection_reason": (
            f"pooled AR(1) on forecast errors gives phi_d={phi_d:.4f} >= 0.995 (explosive or "
            "unit-root); simulating it diverges and the mean path leaves the Focus curve. The "
            "empirical sd(h) term structure saturates, which no AR(1) can match given its "
            "super-random-walk growth at short horizons."
        ) if explosive else None,
        "n_pairs": int(m.sum()),
        "n_vintages": int(v["start_q"].nunique()),
        "provisional": False,
        "sd_raw_by_h": sd_raw,          # before the monotone/flat correction, for audit
        "n_by_h": n_by_h,               # why the correction is needed: the sample thins with h
        "min_obs_for_sd": MIN_OBS,
        "h_flat_beyond": h_cut,
        "empirical_bias_by_h": bias_by_h,
        "source": os.path.basename(vintages_csv),
    }


def provisional_error_process(ar1: dict) -> dict:
    """Deviation process to use UNTIL the Focus vintages exist.

    Deviations from the expected path inherit the level's shock persistence and its
    one-step innovation sd. This is a placeholder with the right shape (uncertainty zero at
    h=0, widening with the horizon), NOT a measurement of survey forecast error, and it is
    flagged `provisional: true` so nothing downstream can mistake it for one.
    """
    return {
        "mode": "focus_mean_plus_ar1_deviation",
        "phi_d": float(ar1["rho"]),
        "sigma_d": float(ar1["sigma_u"]),
        "provisional": True,
        "note": ("PROVISIONAL: no Focus vintages file found. Re-run bbl_transitions.py after "
                 "scrape_forward_rf.py --vintage-from/--to writes forward_rf_vintages.csv, "
                 "which identifies the deviation process from actual survey forecast errors."),
    }


def _sd_at(sd_by_h: dict, h: int) -> float:
    """Forecast-error sd at horizon h, reusing the largest estimated horizon beyond the end.
    JSON turns the integer keys into strings, so accept both."""
    if not sd_by_h:
        return 0.0
    ks = sorted(int(k) for k in sd_by_h)
    kk = h if h in ks else max(k for k in ks if k <= h) if any(k <= h for k in ks) else ks[0]
    return float(sd_by_h.get(kk, sd_by_h.get(str(kk), 0.0)))


def simulate_around_mean(mean_path: np.ndarray, params: dict, n_paths: int,
                         seed: int = 42, antithetic: bool = True,
                         floor: float = 0.0) -> np.ndarray:
    """Rate paths around the Focus curve: r_h = max(floor, mean_path[h] + d_h).

    `mean_path` is the Focus curve for the launch quarter (length T, quarterly decimal). Two
    deviation processes, selected by `params["mode"]`:

    * `focus_mean_plus_horizon_shock` (the default, and what the data support) --
      d_h = sd(h)*z with ONE standard normal z per path. Var(d_h) then equals the empirical
      Focus forecast-error variance at every horizon by construction, and the surprise
      persists along a path, which is how a policy-regime surprise actually behaves.
    * `focus_mean_plus_ar1_deviation` -- d_h = phi*d_{h-1} + sigma*eps_h, kept for the case
      where a future sample identifies a genuinely stationary phi. It is NOT used when the
      estimator flags the fit explosive (see `estimate_focus_error_process`).

    Antithetic pairs make the simulated mean match `mean_path` far more closely at a given
    n_paths, which matters because psi is an expectation over paths: drift in the simulated
    mean would be indistinguishable from a change in what banks are assumed to believe.
    """
    T = len(mean_path)
    rng = np.random.default_rng(seed)
    out = np.empty((n_paths, T))

    if params.get("mode") == "focus_mean_plus_horizon_shock":
        half = (n_paths + 1) // 2 if antithetic else n_paths
        z = rng.standard_normal(half)
        if antithetic:
            z = np.concatenate([z, -z])[:n_paths]
        sd = np.array([_sd_at(params.get("sd_by_h", {}), h) for h in range(1, T + 1)])
        for t in range(T):
            out[:, t] = np.maximum(mean_path[t] + sd[t] * z, floor)
        return out

    phi, sig = float(params["phi_d"]), float(params["sigma_d"])
    half = (n_paths + 1) // 2 if antithetic else n_paths
    eps = rng.standard_normal((half, T))
    if antithetic:
        eps = np.vstack([eps, -eps])[:n_paths]
    d = np.zeros(n_paths)
    for t in range(T):
        d = phi * d + sig * eps[:, t]
        out[:, t] = np.maximum(mean_path[t] + d, floor)
    return out


def simulate_cir(params: dict, n_paths: int, T: int, r0: float,
                 seed: int = 42, antithetic: bool = True) -> np.ndarray:
    """Full-truncation Euler for CIR, which keeps the rate non-negative even when the Feller
    condition fails. Returns (n_paths, T). Antithetic pairs halve the simulation variance of
    the path mean at no cost, which matters because psi is an expectation over paths."""
    kappa, theta, sigma = params["kappa_q"], params["theta_q"], params["sigma_q"]
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2 if antithetic else n_paths
    eps = rng.standard_normal((half, T))
    if antithetic:
        eps = np.vstack([eps, -eps])[:n_paths]
    out = np.empty((n_paths, T))
    r = np.full(n_paths, float(r0))
    for t in range(T):
        rp = np.maximum(r, 0.0)
        r = r + kappa * (theta - rp) + sigma * np.sqrt(rp) * eps[:, t]
        r = np.maximum(r, 0.0)
        out[:, t] = r
    return out


# ---------------------------------------------------------------------------
# 2/3. Panel blocks: mean reversion to a unit-specific level
# ---------------------------------------------------------------------------
def _panel_ar1(df: pd.DataFrame, key: str, col: str) -> dict:
    """Pooled within-unit AR(1) on CONSECUTIVE quarters only.

    The panel is firm x market x type x quarter, so a firm-level or market-level series is
    repeated across many rows; it is collapsed to one value per (unit, quarter) FIRST. A
    naive shift over the raw rows compares a unit against itself across markets and returns
    a meaningless number.

    Reports the raw within-unit slope and the Nickell (1981) order-1/T correction, since T
    here is ~36-52 quarters and the within estimator is biased toward zero.
    """
    w = (df[[key, "t", col]].dropna()
         .groupby([key, "t"], as_index=False)[col].first()
         .sort_values([key, "t"]))
    w["lag"] = w.groupby(key)[col].shift(1)
    w["dt"] = w.groupby(key)["t"].diff()
    m = w["lag"].notna() & (w["dt"] == 1)
    w = w[m]
    if len(w) < 50:
        return {"rho": float("nan"), "n_pairs": int(len(w)), "note": "too few pairs"}
    # within transform: demean both sides by unit so rho is the within-unit slope
    gm_y = w.groupby(key)[col].transform("mean")
    gm_x = w.groupby(key)["lag"].transform("mean")
    y = (w[col] - gm_y).to_numpy(float)
    x = (w["lag"] - gm_x).to_numpy(float)
    denom = float(x @ x)
    if denom <= 0:
        return {"rho": float("nan"), "n_pairs": int(len(w)), "note": "degenerate"}
    rho = float((x @ y) / denom)
    resid = y - rho * x
    n_units = int(w[key].nunique())
    T_bar = float(len(w) / max(n_units, 1))
    nickell = -(1.0 + rho) / T_bar if T_bar > 1 else 0.0
    return {
        "rho": rho,
        "rho_nickell_corrected": float(rho - nickell),   # subtract the (negative) bias
        "sigma_u": float(np.std(resid, ddof=1)),
        "unit_mean_overall": float(w[col].mean()),
        "n_pairs": int(len(w)),
        "n_units": n_units,
        "T_bar": T_bar,
        "half_life_quarters": float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else float("inf"),
    }


def estimate_panel_block(df: pd.DataFrame, key: str, cols: list[str],
                         by_type: bool = False) -> dict:
    out = {}
    for c in cols:
        if c not in df.columns:
            out[c] = {"rho": float("nan"), "note": "column absent from the panel"}
            continue
        if by_type and "is_B" in df.columns:
            out[c] = {
                "B": _panel_ar1(df[df["is_B"].astype(bool)], key, c),
                "D": _panel_ar1(df[~df["is_B"].astype(bool)], key, c),
                "pooled": _panel_ar1(df, key, c),
            }
        else:
            out[c] = _panel_ar1(df, key, c)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_panel(cols: list[str]) -> pd.DataFrame:
    pq = str(paths.market_panel_csv()).replace(".csv", ".parquet")
    src = pq if os.path.exists(pq) else str(paths.market_panel_csv())
    have = None
    if src.endswith(".parquet"):
        import pyarrow.parquet as pqm
        have = set(pqm.ParquetFile(src).schema.names)
        df = pd.read_parquet(src, columns=[c for c in cols if c in have])
    else:
        df = pd.read_csv(src, usecols=lambda c: c in set(cols), low_memory=False)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            "market panel is missing column(s): " + ", ".join(missing) +
            "\nRebuild the panel (panel_market -> panel_loo_instruments -> panel_fee_merge)."
        )
    df["t"] = df["year"].astype(int) * 4 + df["quarter"].astype(int)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output JSON (default COST_FWD/bbl_transitions.json)")
    ap.add_argument("--vintages", default=None,
                    help="Focus vintages CSV from scrape_forward_rf.py (default "
                         "COST_FWD/forward_rf_vintages.csv). When present the deviation "
                         "process is identified from real survey forecast errors.")
    ap.add_argument("--rate-only", action="store_true", help="skip the two panel blocks")
    ap.add_argument("--simulate", type=int, default=0, metavar="N",
                    help="also write N simulated rate paths locally (inspection only)")
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true", help="overlay png of the simulated paths")
    a = ap.parse_args()

    need = ["year", "quarter", RATE_COL]
    if not a.rate_only:
        need += Z_COLS + S_COLS + [FIRM_KEY, MKT_KEY, "is_B"]
    df = load_panel(need)

    # --- block 1: the short rate ------------------------------------------------------
    rq = df.groupby("t")[RATE_COL].first().sort_index()
    gaps = np.diff(rq.index.to_numpy())
    if not np.all(gaps == 1):
        print(f"  [rate] WARNING: {int((gaps != 1).sum())} non-consecutive quarter(s); "
              "CIR/AR(1) use consecutive transitions only.")
    r = rq.to_numpy(float)
    cir = estimate_cir(r)
    ar1 = estimate_ar1_level(r)

    print("\n=== 1. SHORT RATE (quarterly decimal, national) ===")
    print(f"  sample: {len(r)} quarters, {rq.index.min()}..{rq.index.max()}  "
          f"level [{r.min():.5f}, {r.max():.5f}]  "
          f"annualised [{((1+r.min())**4-1)*100:.2f}%, {((1+r.max())**4-1)*100:.2f}%]")
    print(f"  CIR   kappa={cir['kappa_q']:.4f} (se {cir['kappa_se_q']:.4f})  "
          f"theta={cir['theta_q']:.5f}  sigma={cir['sigma_q']:.5f}   [quarterly]")
    print(f"        annualised: kappa={cir['kappa_ann']:.4f}  theta={cir['theta_ann']:.4f}  "
          f"sigma={cir['sigma_ann']:.4f}   (Egan et al. US: .11 / .03 / .08)")
    print(f"        half-life {cir['half_life_quarters']:.1f} quarters | "
          f"Feller 2*k*th>=s^2: {cir['feller_2kt_ge_s2']} (margin {cir['feller_margin']:+.2e})")
    print(f"  AR(1) rho={ar1['rho']:.4f}  long-run mean={ar1['long_run_mean']:.5f}  "
          f"half-life {ar1['half_life_quarters']:.1f} q")

    # --- the process actually used: Focus mean + estimated AR(1) dispersion --------------
    # CIR/AR(1) above are DIAGNOSTICS ONLY. See the module docstring for the measurement that
    # rules them out as a mean path (kappa insignificant, implied long-run mean above the
    # sample maximum, simulated mean drifting away from the Focus curve).
    vint = a.vintages or os.path.join(str(paths.cost_fwd_dir()), "forward_rf_vintages.csv")
    if os.path.exists(vint):
        try:
            proc = estimate_focus_error_process(vint, rq)
            print("\n=== RATE PROCESS: Focus mean + AR(1) deviation (from survey forecast errors) ===")
            print(f"  source {proc['source']}  |  {proc['n_vintages']} vintages, "
                  f"{proc['n_pairs']} consecutive-horizon pairs")
            print(f"  phi_d = {proc['phi_d']:.4f}   sigma_d = {proc['sigma_d']:.6f}  (quarterly decimal)")
            sd = proc.get("empirical_sd_by_h", {})
            for h in (1, 4, 8, 20):
                if h in sd:
                    print(f"    empirical forecast-error sd at h={h:<2d}: {sd[h]:.5f} "
                          f"({((1+sd[h])**4-1)*100:.2f} pp annualised-equivalent)")
        except Exception as exc:
            print(f"\n  [rate process] vintages present but unusable ({exc}); falling back.")
            proc = provisional_error_process(ar1)
    else:
        proc = provisional_error_process(ar1)
        print("\n=== RATE PROCESS: Focus mean + AR(1) deviation (PROVISIONAL) ===")
        print(f"  no {os.path.basename(vint)} yet -> phi_d={proc['phi_d']:.4f} "
              f"sigma_d={proc['sigma_d']:.6f} from the level AR(1).")
        print("  Re-run after scrape_forward_rf.py writes the vintages to identify this from")
        print("  actual Focus forecast errors.")

    res = {
        "_meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "bbl_transitions.py",
            "rate_col": RATE_COL,
            "units": "quarterly decimal rates; panel states in their panel units",
            "n_quarters": int(len(r)),
            "t_min": int(rq.index.min()), "t_max": int(rq.index.max()),
            "r_last": float(r[-1]),
        },
        "rate": {
            # The object the forward simulation consumes.
            "process": proc,
            # Diagnostics, retained so the rejection of a pure estimated CIR is on the record.
            "cir": {**cir, "usable_as_mean": False,
                    "why_not": ("kappa insignificant (se > estimate) so theta is unidentified; "
                                "implied long-run mean exceeds the sample maximum and the "
                                "simulated mean path contradicts the Focus curve")},
            "ar1_level": {**ar1, "usable_as_mean": False,
                          "why_not": "long-run mean at/above the sample maximum; same short-sample problem"},
        },
    }

    if not a.rate_only:
        # --- block 2: cost shifters (firm x quarter) ----------------------------------
        zdf = df[[FIRM_KEY, "t", "is_B"] + Z_COLS].copy()
        if _winsorize_within_type is not None:
            try:
                zdf = _winsorize_within_type(zdf, Z_COLS, pct=0.01, type_key="is_B")
            except Exception as exc:  # keep going; report rather than silently skip
                print(f"  [Z] winsorize_within_type unavailable ({exc}); using raw values.")
        res["cost_shifters"] = estimate_panel_block(zdf, FIRM_KEY, Z_COLS, by_type=True)
        print("\n=== 2. COST SHIFTERS Z (firm x quarter, winsorised 1/99 within type) ===")
        print("  %-26s %8s %8s %10s %9s" % ("var", "rho(B)", "rho(D)", "half-life B", "n pairs"))
        for c in Z_COLS:
            b, d_ = res["cost_shifters"][c]["B"], res["cost_shifters"][c]["D"]
            print("  %-26s %8.4f %8.4f %10.1f %9d"
                  % (c, b["rho"], d_["rho"], b.get("half_life_quarters", float("nan")),
                     b["n_pairs"]))

        # --- block 3: market states (MCA x quarter) -----------------------------------
        res["market_states"] = estimate_panel_block(df, MKT_KEY, S_COLS, by_type=False)
        res["market_states"]["pix_exists"] = {
            "type": "deterministic_absorbing",
            "note": "0 before 2020Q4, 1 from 2020Q4 onward; not stochastic.",
        }
        print("\n=== 3. MARKET STATES S (MCA x quarter) ===")
        print("  %-28s %8s %10s %12s %9s" % ("var", "rho", "half-life", "sigma_u", "n pairs"))
        for c in S_COLS:
            v = res["market_states"][c]
            print("  %-28s %8.4f %10.1f %12.5f %9d"
                  % (c, v["rho"], v.get("half_life_quarters", float("nan")),
                     v.get("sigma_u", float("nan")), v["n_pairs"]))
        print("  NOTE: these columns are ALSO the BLP demographics, so letting them evolve "
              "makes the\n        active share time-varying (see the module docstring).")

    out = a.out or os.path.join(str(paths.cost_fwd_dir()), "bbl_transitions.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, sort_keys=False)
    print(f"\n  [write] {out}  ({os.path.getsize(out)} bytes)")

    if a.simulate:
        # Prefer the process actually used: the latest vintage's Focus curve as the mean,
        # with the estimated AR(1) dispersion around it. Fall back to the CIR simulator only
        # to illustrate the rejected diagnostic.
        mean_path = None
        if os.path.exists(vint):
            try:
                vdf = pd.read_csv(vint)
                last = sorted(vdf["start_q"].astype(str).unique())[-1]
                mp = (vdf[(vdf["start_q"].astype(str) == last) & (vdf["h"] >= 1)]
                      .sort_values("h")["rf_qoq"].to_numpy(float))
                if len(mp) >= a.horizon:
                    mean_path = mp[:a.horizon]
                    print(f"  [simulate] mean path = Focus vintage {last}")
            except Exception as exc:
                print(f"  [simulate] could not read a mean path from {vint}: {exc}")
        if mean_path is None:
            fp = os.path.join(str(paths.cost_fwd_dir()), "forward_rf_qoq.csv")
            if os.path.exists(fp):
                fdf = pd.read_csv(fp).sort_values("h")
                mp = fdf.loc[fdf["h"] >= 1, "rf_qoq"].to_numpy(float)
                if len(mp) >= a.horizon:
                    mean_path = mp[:a.horizon]
                    print("  [simulate] mean path = the single Focus curve (forward_rf_qoq.csv)")
        if mean_path is not None:
            pth = simulate_around_mean(mean_path, proc, a.simulate, seed=a.seed)
        else:
            print("  [simulate] no Focus curve available; illustrating with the REJECTED CIR fit.")
            pth = simulate_cir(cir, a.simulate, a.horizon, r0=float(r[-1]), seed=a.seed)
        csv = os.path.join(os.path.dirname(out), "bbl_rate_paths_local.csv")
        pd.DataFrame({
            "path_id": np.repeat(np.arange(a.simulate), a.horizon),
            "h": np.tile(np.arange(1, a.horizon + 1), a.simulate),
            "rf_qoq": pth.ravel(),
        }).to_csv(csv, index=False)
        ann = (1 + pth) ** 4 - 1
        print(f"  [simulate] {a.simulate} paths x {a.horizon}q -> {csv}")
        print(f"      terminal annualised: p5 {np.percentile(ann[:, -1], 5)*100:.2f}%  "
              f"median {np.median(ann[:, -1])*100:.2f}%  p95 {np.percentile(ann[:, -1], 95)*100:.2f}%")
        print(f"      mean path annualised: h1 {ann[:, 0].mean()*100:.2f}%  "
              f"h{a.horizon} {ann[:, -1].mean()*100:.2f}%")
        if a.plot:
            os.environ.setdefault("MPLBACKEND", "Agg")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for i in range(min(a.simulate, 60)):
                ax.plot(np.arange(1, a.horizon + 1), ann[i] * 100, lw=0.5, alpha=0.35,
                        color="#4477aa")
            ax.plot(np.arange(1, a.horizon + 1), ann.mean(0) * 100, lw=2.0, color="#cc3311",
                    label="simulated mean")
            if mean_path is not None:
                ax.plot(np.arange(1, a.horizon + 1), ((1 + mean_path) ** 4 - 1) * 100,
                        ls="--", lw=1.6, color="k", label="Focus curve (target mean)")
            ax.set_xlabel("quarters ahead"); ax.set_ylabel("Selic, annualised (%)")
            ax.set_title("Rate paths: Focus mean with estimated dispersion")
            ax.legend(frameon=False)
            png = os.path.join(os.path.dirname(out), "bbl_rate_paths_local.png")
            fig.tight_layout(); fig.savefig(png, dpi=140)
            print(f"  [plot] {png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
