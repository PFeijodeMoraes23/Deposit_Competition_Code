"""
diag_p4_openings_advertising.py
===============================
Task P4 (pix_handoff_2026-11-03c, Update 11-04 12:20): the openings moment of the switching-cost
design (Drafts/Deposit Competition/kappa_design_2026-11-02.md, eq. ccs),
    Openings_jt / (Adults_t - N_j,t-1) = (1 - phi_t) A_jt P(j | aware),
against the awareness shifter. A side analysis: nothing here enters market_panel or an estimator.

Panel, conglomerate x quarter, 2016Q1-2024Q4, from the CCS (make_ccs_openings_pix's flows, stock and
adult population):
  o_jt      the quarter's mean monthly opening rate over the months AW1 keeps (not flagged, positive
            denominator): sum_m Openings_jm / sum_m (Adults_m - N_j,m-1); quarters with no kept month
            or no opening drop out
  adv_jt    P2's adv_brl_real_2024q4 (Cosif LAI advertising, R$ of 2024Q4) where P2's status is
            observed; negative entries (reversals) set to zero and counted
  rho_j,t-1 the conglomerate's national type-4 spread (spread_ann_a4_national of market_panel:
            annual, risk-free less rate), lagged one quarter, in pp; the RT1 fallback flag (national
            rate equal to the Selic, or zero) rides along
  big five  the five largest B conglomerates by 2019 deposits (P2's rank_2019)
Regressions, conglomerate and quarter effects (the quarter effects absorb ln(1 - phi_t) and the
adult population), unweighted, B conglomerates unless said:
  main        ln o_jt = a_j + t_q + g ln(1 + adv_jt) + p rho_j,t-1 + e; and without the price term
  groups      g for the big five and for the rest in one regression (common effects and price term),
              their difference; and each group alone
  lags        ln(1 + adv) at t, t-1 and t-2 by group, and their sum
  D           the main regression on D conglomerates, for scale
  checks      the ratio ln((1 + adv_jt) / max(Dep_jt, R$10m)) in place of ln(1 + adv) (P2's floor on
              tiny deposits; Dep from P2, types 1, 2, 4 and 5 over the panel's cells); the intensive
              margin, ln adv on quarters with adv > 0; the price term without RT1 fallbacks at t-1
              and inside the spread's 5th-95th percentiles
Inference: CR1 by conglomerate; the wild cluster restricted bootstrap by conglomerate (Webb weights,
999 draws, the same draws for every null), its p-value at zero and its 95% interval by test
inversion. Every tested quantity is one coefficient of its own parametrization (difference: adv
and adv x big five; lag sum: x_t, x_t-1 - x_t, x_t-2 - x_t). The big five alone have 5 clusters: the
bootstrap is reported but rests on 6^5 weight vectors.

P5 (pix_handoff_2026-11-03c, Update 11-04 13:55), on the same panel:
  IV          ln(1 + adv_jt) instrumented by ln(1 + adv_j,t-4), the budget of a year earlier (2015's
              advertising, by P2's rule from the advertising panel, for 2016): first stage (CR1 F),
              2SLS with CR1, the Anderson-Rubin test by the wild cluster restricted bootstrap and
              its 95% set by inversion, OLS on the same sample; on B, the rest, the big five
  placebo     ln o_jt on ln(1 + sum of the other B banks' advertising) beside own advertising and
              the spread (literal), and on the other banks of j's group, big five or rest (group);
              the quarter effects absorb all but a sliver of the literal sum, which is reported

Inputs (read only): the CCS inputs of make_ccs_openings_pix.py; Rout/p2_advertising_conglomerate_
quarter.csv and p2_advertising_coverage_B.csv; processed/market_panel.parquet, six columns read
once with pyarrow for the national spread and its fallback flag (its md5 is written in every row);
Awareness/processed/advertising_panel_quarterly.parquet for P5's advertising from 2015.
The spread changes with the rebuild (I6, I8, I9, D10): rerun with P3 at the new md5.
Outputs (Rout): p4_openings_advertising.csv (the panel, with P5's instrument and rivals),
p4_openings_advertising.json (P4's blocks and a p5 block).

Usage:  python -I diagnostics/diag_p4_openings_advertising.py             (P4 and P5; reads the panel)
        python -I diagnostics/diag_p4_openings_advertising.py --only p5   (P5 on the panel file P4 wrote)
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/ and pipeline modules
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import paths
from make_ccs_openings_pix import adults_monthly, flows_monthly, stock

PANEL = paths.PROCESSED / "market_panel.parquet"
OUT = paths.rout_dir()
ADV = OUT / "p2_advertising_conglomerate_quarter.csv"
ADV_PANEL = paths.AWARENESS_PROC / "advertising_panel_quarterly.parquet"
COVERAGE = OUT / "p2_advertising_coverage_B.csv"
OUT_CSV = OUT / "p4_openings_advertising.csv"
OUT_JSON = OUT / "p4_openings_advertising.json"
T0, T1 = (2016, 1), (2024, 4)
# As diag_p_shifters.OBSERVED: an imputed cell counts as observed.
OBSERVED = ("observed_positive", "observed_zero", "observed_negative", "imputed_from_row_total")
DEP_FLOOR = 1e7
N_BOOT = 999
SEED = 20261104
SEED_P5 = SEED + 5          # P5 draws its own, so P4's estimates do not move when P5 runs alone
# Internal names -> the panel file's columns.
RENAME = {"j": "CodConglomeradoPrudencial", "x": "ln1p_adv", "x_l1": "ln1p_adv_l1",
          "x_l2": "ln1p_adv_l2", "x_ratio": "ln_adv_ratio_floor", "x_l4": "ln1p_adv_l4",
          "riv_all": "ln1p_rivals_adv", "riv_group": "ln1p_group_rivals_adv"}
P5_COLS = ["x_l4", "riv_all", "riv_group"]
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def stamp(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def qi(year, quarter):
    return year * 4 + quarter - 1


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ inputs
def openings_quarterly() -> pd.DataFrame:
    adults = adults_monthly()
    d = stock(flows_monthly()).sort_values(["conglomerate", "month"])
    d["N_lag"] = d.groupby("conglomerate")["N"].shift(1)
    d["den"] = d["month"].map(adults) - d["N_lag"]
    d["year"], d["quarter"] = d["month"].dt.year, (d["month"].dt.month - 1) // 3 + 1
    d = d[(qi(d["year"], d["quarter"]) >= qi(*T0)) & (qi(d["year"], d["quarter"]) <= qi(*T1))]
    d["kept"] = ~d["flagged"] & (d["den"] > 0)
    g = d.groupby(["conglomerate", "year", "quarter"])
    q = g.agg(is_B=("is_B", "first"), N_source=("N_source", "first"),
              n_months_flagged=("flagged", "sum")).join(
        d[d["kept"]].groupby(["conglomerate", "year", "quarter"]).agg(
            openings=("openings", "sum"), den=("den", "sum"), n_months_kept=("month", "size")))
    q = q.reset_index()
    q["n_months_kept"] = q["n_months_kept"].fillna(0).astype(int)
    q["o"] = q["openings"] / q["den"]
    q["ln_o"] = np.log(q["o"].where(q["openings"] > 0))
    return q.rename(columns={"conglomerate": "j"})


def advertising() -> tuple[pd.DataFrame, list[str]]:
    a = pd.read_csv(ADV, dtype={"CodConglomeradoPrudencial": str}).rename(
        columns={"CodConglomeradoPrudencial": "j"})
    a["adv_observed"] = a["adv_status"].isin(OBSERVED)
    a["adv"] = a["adv_brl_real_2024q4"].where(a["adv_observed"])
    a["adv_negative"] = (a["adv"] < 0).astype(int)
    a["adv"] = a["adv"].clip(lower=0.0)
    cov = pd.read_csv(COVERAGE, dtype={"CodConglomeradoPrudencial": str})
    big5 = cov.sort_values("rank_2019")["CodConglomeradoPrudencial"].head(5).tolist()
    return a[["j", "year", "quarter", "name", "dep_brl", "adv", "adv_status", "adv_negative"]], big5


def national_spread() -> tuple[pd.DataFrame, str]:
    got = md5(PANEL)
    p = pq.read_table(PANEL, columns=["CodConglomeradoPrudencial", "year", "quarter",
                                      "spread_ann_a4_national", "rate_a4_national", "risk_free_qoq"],
                      filters=[("year", ">=", 2015)]).to_pandas()
    p = p.rename(columns={"CodConglomeradoPrudencial": "j"})
    p["j"] = p["j"].astype(str)
    if (p.groupby(["j", "year", "quarter"])["spread_ann_a4_national"].nunique() > 1).any():
        raise SystemExit("spread_ann_a4_national varies within a conglomerate-quarter")
    p = p.drop_duplicates(["j", "year", "quarter"])
    p["rate_fallback"] = (((p["rate_a4_national"] - p["risk_free_qoq"]).abs() < 1e-6)
                          | (p["rate_a4_national"] == 0)).astype(int)
    p["qi"] = qi(p["year"], p["quarter"]) + 1          # the lag: the spread of t-1 placed at t
    lag = p[["j", "qi", "spread_ann_a4_national", "rate_fallback"]].rename(
        columns={"spread_ann_a4_national": "spread_lag", "rate_fallback": "rate_fallback_lag"})
    stamp(f"panel {got[:8]}: national spread for {lag['j'].nunique():,} conglomerates")
    return lag, got


# -------------------------------------------------------------------------------------- estimation
def design(d: pd.DataFrame, regs: list[str]) -> np.ndarray:
    """The regressors first, in order, then the constant and the two sets of effects."""
    if not (d[regs] != 0).any(axis=0).all():
        raise SystemExit(f"a regressor is zero throughout: {regs}")
    fe_j = pd.get_dummies(d["j"], prefix="j", drop_first=True, dtype=float)
    fe_q = pd.get_dummies(d["qi"], prefix="q", drop_first=True, dtype=float)
    X = pd.concat([d[regs].astype(float), fe_j, fe_q], axis=1)
    X.insert(len(regs), "const", 1.0)
    return X.loc[:, (X != 0).any(axis=0)].to_numpy()


class Fit:
    """OLS with CR1 by cluster and the wild cluster restricted bootstrap (Webb weights)."""

    def __init__(self, X, y, g, rng):
        self.X, self.y = X, y
        self.n, k = X.shape
        self.k = int(np.linalg.matrix_rank(X))
        self.XtXi = np.linalg.pinv(X.T @ X)
        self.A = self.XtXi @ X.T
        self.b = self.A @ y
        self.e = y - X @ self.b
        _, self.inv = np.unique(g, return_inverse=True)
        self.G = int(self.inv.max()) + 1
        self.M = np.zeros((self.G, self.n))
        self.M[self.inv, np.arange(self.n)] = 1.0
        self.adj = self.G / (self.G - 1) * (self.n - 1) / (self.n - self.k)
        self.W = rng.choice(WEBB, size=(self.G, N_BOOT))[self.inv]      # n x B, the same for every null

    def refit(self, y):
        """The same design and bootstrap draws on another outcome."""
        f = copy.copy(self)
        f.y, f.b = y, self.A @ y
        f.e = y - self.X @ f.b
        return f

    def se(self, j, E):
        u = self.X @ self.XtXi[:, j]
        return np.sqrt(self.adj * ((self.M @ (u[:, None] * E)) ** 2).sum(axis=0))

    def cr1(self, j):
        return float(self.se(j, self.e[:, None])[0])

    def wcr_p(self, j, r=0.0):
        """p-value of H0: b_j = r, the bootstrap imposing the null."""
        Xr = np.delete(self.X, j, axis=1)
        yr = self.y - r * self.X[:, j]
        br = np.linalg.lstsq(Xr, yr, rcond=None)[0]
        er = yr - Xr @ br
        Ys = (Xr @ br + r * self.X[:, j])[:, None] + er[:, None] * self.W
        Bs = self.A @ Ys
        ts = (Bs[j] - r) / self.se(j, Ys - self.X @ Bs)
        t0 = (self.b[j] - r) / self.cr1(j)
        return float(np.mean(np.abs(ts) >= abs(t0)))

    def wcr_ci(self, j, level=0.95, tol=1e-4):
        """The interval of nulls the bootstrap does not reject at 1 - level, by bisection from the
        point estimate outwards on each side; an end is None where no null within 2^13 CR1
        standard errors is rejected (the interval is unbounded on that side)."""
        b, s = float(self.b[j]), self.cr1(j)
        alpha = 1 - level
        ends = []
        for sign in (-1, 1):
            inner, outer = b, b + sign * 2 * s
            k = 0
            while self.wcr_p(j, outer) > alpha and k < 12:
                inner, outer, k = outer, b + sign * 2 * s * 2 ** (k + 1), k + 1
            if self.wcr_p(j, outer) > alpha:
                ends.append(None)
                continue
            while abs(outer - inner) > tol * max(1.0, s):
                mid = 0.5 * (inner + outer)
                inner, outer = (mid, outer) if self.wcr_p(j, mid) > alpha else (inner, mid)
            ends.append(round(0.5 * (inner + outer), 4))
        return ends


def run(d, regs, test, lhs="ln_o", rng=None, ci=True):
    """Fit ln o on regs plus conglomerate and quarter effects; for each coefficient in test the
    CR1 interval, the bootstrap p-value at zero and (ci) the bootstrap interval."""
    d = d.dropna(subset=[lhs, *regs]).reset_index(drop=True)
    f = Fit(design(d, regs), d[lhs].to_numpy(float), d["j"].to_numpy(), rng)
    out = {"n_obs": int(f.n), "n_conglomerates": f.G, "coef": {}}
    for i, r in enumerate(regs):
        b, s = float(f.b[i]), f.cr1(i)
        c = {"b": round(b, 4), "se": round(s, 4), "ci95_cr1": [round(b - 1.96 * s, 4), round(b + 1.96 * s, 4)]}
        if r in test:
            c["wcr_p0"] = round(f.wcr_p(i), 3)
            if ci:
                c["ci95_wcr"] = f.wcr_ci(i)
        out["coef"][r] = c
    return out


# ---------------------------------------------------------------------------------------------- P5
EXCLUSION = ("The advertising budget of a year earlier reaches this quarter's openings only through "
             "this quarter's advertising.")
WEAKNESS = ("It fails if an acquisition push lasts a year or more (a campaign, a digital-account "
            "launch) beyond what the conglomerate effects absorb, or if advertising builds a stock of "
            "awareness that still raises openings four quarters on; either biases the IV elasticity "
            "up. The IV removes the same-quarter push, not a persistent one.")


def adv_series() -> pd.DataFrame:
    """Conglomerate-quarter advertising from 2015, by P2's rule (cosif_institution, measure adv, an
    observed status and no incomplete quarter), R$ of 2024Q4, negative entries set to zero."""
    a = pd.read_parquet(ADV_PANEL, columns=["panel_code", "year", "quarter", "source", "measure",
                                            "value_status", "flag_notes", "amount_brl_real"])
    s = a[(a["source"] == "cosif_institution") & (a["measure"] == "adv")
          & a["year"].between(T0[0] - 1, T1[0])].copy()
    if s.duplicated(["panel_code", "year", "quarter"]).any():
        raise SystemExit("advertising panel: cosif_institution repeats a conglomerate-quarter")
    ok = s["value_status"].isin(OBSERVED) & ~s["flag_notes"].fillna("").str.contains("incomplete_quarter")
    s["adv_s"] = s["amount_brl_real"].where(ok).clip(lower=0.0)
    s["qi"] = qi(s["year"], s["quarter"])
    s["j"] = s["panel_code"].astype(str)
    return s[["j", "qi", "adv_s"]]


def partial(W, v):
    return v - W @ np.linalg.lstsq(W, v, rcond=None)[0]


def ar_set(ar, b, se, alpha=0.05, n=121):
    """The Anderson-Rubin confidence set {b0 : p(b0) > alpha} on a grid of b +- max(8 se, 0.5),
    edges refined by bisection; None marks a piece that reaches the end of the grid."""
    h = max(8 * se, 0.5)
    grid = np.linspace(b - h, b + h, n)
    acc = np.array([ar(v) > alpha for v in grid])

    def edge(rej, acp):
        for _ in range(10):
            m = 0.5 * (rej + acp)
            rej, acp = (rej, m) if ar(m) > alpha else (m, acp)
        return round(0.5 * (rej + acp), 4)

    pieces, i = [], 0
    while i < n:
        if not acc[i]:
            i += 1
            continue
        k = i
        while k + 1 < n and acc[k + 1]:
            k += 1
        pieces.append([None if i == 0 else edge(grid[i - 1], grid[i]),
                       None if k == n - 1 else edge(grid[k + 1], grid[k])])
        i = k + 1
    return {"pieces": pieces, "grid": [round(grid[0], 4), round(grid[-1], 4)]}


def iv(d, rng):
    """ln o on ln(1 + adv) instrumented by ln(1 + adv) four quarters earlier, with the lagged spread
    and conglomerate and quarter effects: the first stage, 2SLS with CR1, the Anderson-Rubin test by
    the wild cluster restricted bootstrap (the reduced form of ln o - b0 x on the instrument) and its
    set, and OLS on the same sample."""
    d = d.dropna(subset=["ln_o", "x", "x_l4", "rho_lag_pp"]).reset_index(drop=True)
    y, x = d["ln_o"].to_numpy(float), d["x"].to_numpy(float)
    X = design(d, ["x_l4", "rho_lag_pp"])
    fs = Fit(X, x, d["j"].to_numpy(), rng)
    zt, xt, yt = (partial(X[:, 1:], v) for v in (X[:, 0], x, y))
    b = float(zt @ yt / (zt @ xt))
    sc = np.bincount(fs.inv, weights=zt * (yt - b * xt))
    se = float(np.sqrt(fs.adj * (sc ** 2).sum()) / abs(zt @ xt))
    rf = fs.refit(y)

    def ar(b0):
        return rf.refit(y - b0 * x).wcr_p(0)

    f_se = fs.cr1(0)
    ols = run(d, ["x", "rho_lag_pp"], ["x"], rng=rng)["coef"]["x"]
    return {"n_obs": int(fs.n), "n_conglomerates": fs.G,
            "first_stage": {"b": round(float(fs.b[0]), 4), "se": round(f_se, 4),
                            "F_cr1": round(float((fs.b[0] / f_se) ** 2), 2), "wcr_p0": round(fs.wcr_p(0), 3)},
            "reduced_form": {"b": round(float(rf.b[0]), 4), "se": round(rf.cr1(0), 4)},
            "iv": {"b": round(b, 4), "se_cr1": round(se, 4),
                   "ci95_cr1": [round(b - 1.96 * se, 4), round(b + 1.96 * se, 4)],
                   "ar_wcr_p0": round(ar(0.0), 3), "ar_wcr_set95": ar_set(ar, b, se)},
            "ols_same_sample": ols}


def absorbed(d, v):
    """The share of v's variance left after conglomerate and quarter effects, and the correlation of
    what is left with own advertising's remainder."""
    E = design(d, [])
    r, o = partial(E, d[v].to_numpy(float)), partial(E, d["x"].to_numpy(float))
    return {"sd": round(float(d[v].std()), 4), "sd_after_effects": round(float(r.std()), 4),
            "variance_share_left": round(float(r.var() / d[v].var()), 4),
            "corr_with_own_adv_left": round(float(np.corrcoef(r, o)[0, 1]), 3)}


def p5(d: pd.DataFrame) -> dict:
    """Adds x_l4, riv_all and riv_group to d (in place) and returns the P5 block."""
    rng = np.random.default_rng(SEED_P5)
    s = adv_series()
    l4 = s.assign(qi=s["qi"] + 4)
    d["x_l4"] = np.log1p(d[["j", "qi"]].merge(l4, on=["j", "qi"], how="left")["adv_s"].to_numpy())
    chk = d[["j", "qi", "adv"]].merge(s, on=["j", "qi"], how="left")
    both = chk["adv"].notna() & chk["adv_s"].notna()
    # Rivals: the other B banks (P2's B set), all of them or those of j's own group.
    p2 = pd.read_csv(ADV, dtype={"CodConglomeradoPrudencial": str}, usecols=["CodConglomeradoPrudencial", "is_B"])
    bset = set(p2.loc[p2["is_B"] == 1, "CodConglomeradoPrudencial"])
    big5 = set(d.loc[d["big5"] == 1, "j"])
    sb = s[s["j"].isin(bset)].assign(b5=lambda t: t["j"].isin(big5))
    tot, totg = sb.groupby("qi")["adv_s"].sum(), sb.groupby(["qi", "b5"])["adv_s"].sum()
    own = d[["j", "qi"]].merge(sb[["j", "qi", "adv_s"]], on=["j", "qi"], how="left")["adv_s"].fillna(0.0).to_numpy()
    grp = np.array([totg.get((q, bool(b)), np.nan) for q, b in zip(d["qi"], d["big5"])])
    d["riv_all"] = np.log1p(np.clip(d["qi"].map(tot).to_numpy() - own, 0.0, None))
    d["riv_group"] = np.log1p(np.clip(grp - own, 0.0, None))
    d.loc[d["is_B"] != 1, ["riv_all", "riv_group"]] = np.nan

    B = d[d["is_B"] == 1]
    rep = {"exclusion": EXCLUSION, "weakness": WEAKNESS,
           "adv_check_2016_on": {"cells_compared": int(both.sum()),
                                 "max_abs_diff_brl": float((chk.loc[both, "adv"] - chk.loc[both, "adv_s"]).abs().max())},
           "instrument_from_2015": int(B.loc[B["qi"] < qi(T0[0] + 1, 1), "x_l4"].notna().sum())}
    for lab, sub in (("B", B), ("rest", B[B["big5"] == 0]), ("big5_5_clusters", B[B["big5"] == 1])):
        rep[f"iv_{lab}"] = iv(sub, rng)
        r = rep[f"iv_{lab}"]
        stamp(f"P5 IV {lab}: first-stage F {r['first_stage']['F_cr1']}, IV {r['iv']['b']} "
              f"(AR p0 {r['iv']['ar_wcr_p0']}, set {r['iv']['ar_wcr_set95']['pieces']}), "
              f"OLS same sample {r['ols_same_sample']['b']}; n={r['n_obs']} G={r['n_conglomerates']}")
    ps = B.dropna(subset=["ln_o", "x", "rho_lag_pp", "riv_all", "riv_group"])
    rep["placebo_variation"] = {v: absorbed(ps, v) for v in ("riv_all", "riv_group")}
    rep["placebo_literal"] = run(B, ["x", "riv_all", "rho_lag_pp"], ["x", "riv_all"], rng=rng)
    rep["placebo_group"] = run(B, ["x", "riv_group", "rho_lag_pp"], ["x", "riv_group"], rng=rng)
    for k in ("placebo_literal", "placebo_group"):
        stamp(f"P5 {k}: {rep[k]['coef']} n={rep[k]['n_obs']}")
    rep["_meta"] = {"written": datetime.datetime.now().isoformat(timespec="seconds"), "seed": SEED_P5,
                    "n_boot": N_BOOT}
    return rep


def write_panel(d: pd.DataFrame):
    cols = ["j", "name", "is_B", "big5", "year", "quarter", "openings", "den", "n_months_kept",
            "n_months_flagged", "o", "ln_o", "N_source", "adv", "adv_status", "adv_negative", "x",
            "x_l1", "x_l2", "dep_brl", "x_ratio", "rho_lag_pp", "rate_fallback_lag", "panel_md5", *P5_COLS]
    d.sort_values(["is_B", "j", "year", "quarter"], ascending=[False, True, True, True])[cols].rename(
        columns=RENAME).to_csv(OUT_CSV, index=False)


# -------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["p5"], help="P5 alone, on the panel file P4 wrote (no panel read)")
    if ap.parse_args().only == "p5":
        d = pd.read_csv(OUT_CSV, dtype={"CodConglomeradoPrudencial": str}).rename(
            columns={v: k for k, v in RENAME.items()})
        d = d.drop(columns=[c for c in P5_COLS if c in d])
        d["qi"] = qi(d["year"], d["quarter"])
        rep = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        rep["p5"] = p5(d)
        write_panel(d)
        OUT_JSON.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        stamp(f"wrote the p5 block of {OUT_JSON.name}; {OUT_CSV.name} gains "
              f"{', '.join(RENAME[c] for c in P5_COLS)}")
        return

    rng = np.random.default_rng(SEED)
    q = openings_quarterly()
    a, big5 = advertising()
    lag, got = national_spread()
    d = q.merge(a, on=["j", "year", "quarter"], how="left")
    d["qi"] = qi(d["year"], d["quarter"])
    d = d.merge(lag, on=["j", "qi"], how="left")
    d["big5"] = d["j"].isin(big5).astype(int)
    d["x"] = np.log1p(d["adv"])
    for k in (1, 2):
        prev = d[["j", "qi", "x"]].assign(qi=d["qi"] + k).rename(columns={"x": f"x_l{k}"})
        d = d.merge(prev, on=["j", "qi"], how="left")
    d["x_ratio"] = d["x"] - np.log(d["dep_brl"].clip(lower=DEP_FLOOR))
    d["x_pos"] = np.log(d["adv"].where(d["adv"] > 0))
    d["rho_lag_pp"] = 100 * d["spread_lag"]
    d["panel_md5"] = got[:8]
    for v in ("x", "x_l1", "x_l2", "x_ratio", "x_pos"):
        d[f"{v}_b5"], d[f"{v}_rest"] = d[v] * d["big5"], d[v] * (1 - d["big5"])
    d["dl1"], d["dl2"] = d["x_l1"] - d["x"], d["x_l2"] - d["x"]
    for v in ("dl1", "dl2"):
        d[f"{v}_b5"], d[f"{v}_rest"] = d[v] * d["big5"], d[v] * (1 - d["big5"])

    B, D = d[d["is_B"] == 1], d[d["is_B"] == 0]
    sB = B.dropna(subset=["ln_o", "x", "rho_lag_pp"])
    rep = {"sample": {
        "B_conglomerates_in_ccs": int(B["j"].nunique()),
        "B_quarters": int(len(B)),
        "B_no_kept_month": int((B["n_months_kept"] == 0).sum()),
        "B_zero_openings": int(((B["n_months_kept"] > 0) & ~(B["openings"] > 0)).sum()),
        "B_adv_missing": int(B["x"].isna().sum()),
        "B_adv_negative_set_zero": int(B["adv_negative"].fillna(0).sum()),
        "B_adv_zero": int((B["adv"] == 0).sum()),
        "B_spread_missing": int(B["rho_lag_pp"].isna().sum()),
        "B_main_obs": int(len(sB)), "B_main_conglomerates": int(sB["j"].nunique()),
        "B_main_rate_fallback_lag": int(sB["rate_fallback_lag"].sum()),
        "big5": {c: str(B.loc[B["j"] == c, "name"].dropna().iloc[0]) if (B["j"] == c).any() else None
                 for c in big5},
        "big5_adv_zero_quarters": int(((B["big5"] == 1) & (B["adv"] == 0)).sum()),
        "D_conglomerates_in_ccs": int(D["j"].nunique()),
        "D_main_obs": int(len(D.dropna(subset=["ln_o", "x", "rho_lag_pp"]))),
        "N_source_B": B.drop_duplicates("j")["N_source"].value_counts().to_dict(),
        "dep_floor_binds_B": int((B["dep_brl"] < DEP_FLOOR).sum()),
    }}
    stamp(f"B: {rep['sample']['B_main_obs']:,} conglomerate-quarters, "
          f"{rep['sample']['B_main_conglomerates']} conglomerates in the main sample")

    rep["main_B"] = run(B, ["x", "rho_lag_pp"], ["x", "rho_lag_pp"], rng=rng)
    rep["main_B_no_price"] = run(B.dropna(subset=["rho_lag_pp"]), ["x"], ["x"], rng=rng)
    # Groups: one regression, two parametrizations, so each quantity is one coefficient.
    g1 = run(B, ["x", "x_b5", "rho_lag_pp"], ["x", "x_b5", "rho_lag_pp"], rng=rng)
    g2 = run(B, ["x", "x_rest", "rho_lag_pp"], ["x"], rng=rng)
    rep["groups_B"] = {"n_obs": g1["n_obs"], "n_conglomerates": g1["n_conglomerates"],
                       "adv_big5": g2["coef"]["x"], "adv_rest": g1["coef"]["x"],
                       "adv_big5_minus_rest": g1["coef"]["x_b5"], "price": g1["coef"]["rho_lag_pp"]}
    rep["big5_alone"] = run(B[B["big5"] == 1], ["x", "rho_lag_pp"], ["x", "rho_lag_pp"], rng=rng)
    rep["rest_alone"] = run(B[B["big5"] == 0], ["x", "rho_lag_pp"], ["x", "rho_lag_pp"], rng=rng)
    # Lags by group: the plain lags (CR1), and the sum as the coefficient on x_t of the
    # reparametrized design.
    lags = B.dropna(subset=["x_l1", "x_l2"])
    plain = run(lags, ["x_b5", "x_l1_b5", "x_l2_b5", "x_rest", "x_l1_rest", "x_l2_rest", "rho_lag_pp"],
                [], rng=rng)
    summ = run(lags, ["x_b5", "dl1_b5", "dl2_b5", "x_rest", "dl1_rest", "dl2_rest", "rho_lag_pp"],
               ["x_b5", "x_rest"], rng=rng)
    rep["lags_B"] = {"n_obs": plain["n_obs"], "n_conglomerates": plain["n_conglomerates"],
                     "big5": {"t": plain["coef"]["x_b5"], "t-1": plain["coef"]["x_l1_b5"],
                              "t-2": plain["coef"]["x_l2_b5"], "sum": summ["coef"]["x_b5"]},
                     "rest": {"t": plain["coef"]["x_rest"], "t-1": plain["coef"]["x_l1_rest"],
                              "t-2": plain["coef"]["x_l2_rest"], "sum": summ["coef"]["x_rest"]},
                     "price": plain["coef"]["rho_lag_pp"]}
    rep["D_main"] = run(D, ["x", "rho_lag_pp"], ["x", "rho_lag_pp"], rng=rng)
    rep["D_no_price"] = run(D, ["x"], ["x"], rng=rng)
    r1 = run(B, ["x_ratio", "x_ratio_b5", "rho_lag_pp"], ["x_ratio", "x_ratio_b5"], rng=rng, ci=False)
    r2 = run(B, ["x_ratio", "x_ratio_rest", "rho_lag_pp"], ["x_ratio"], rng=rng, ci=False)
    rep["ratio_B"] = {"n_obs": r1["n_obs"], "adv_big5": r2["coef"]["x_ratio"],
                      "adv_rest": r1["coef"]["x_ratio"], "adv_big5_minus_rest": r1["coef"]["x_ratio_b5"],
                      "price": r1["coef"]["rho_lag_pp"]}
    i1 = run(B, ["x_pos", "x_pos_b5", "rho_lag_pp"], ["x_pos", "x_pos_b5"], rng=rng, ci=False)
    i2 = run(B, ["x_pos", "x_pos_rest", "rho_lag_pp"], ["x_pos"], rng=rng, ci=False)
    rep["intensive_B"] = {"n_obs": i1["n_obs"], "adv_big5": i2["coef"]["x_pos"],
                          "adv_rest": i1["coef"]["x_pos"], "adv_big5_minus_rest": i1["coef"]["x_pos_b5"],
                          "price": i1["coef"]["rho_lag_pp"]}
    # The price term where the spread is least likely to be mismeasured: no RT1 fallback at t-1 and
    # the spread inside its 5th-95th percentiles on the main sample. Last, so the draws above stay.
    lo, hi = (float(v) for v in sB["rho_lag_pp"].quantile([0.05, 0.95]))
    trim = B[(B["rate_fallback_lag"] == 0) & B["rho_lag_pp"].between(lo, hi)]
    rep["main_B_price_trimmed"] = run(trim, ["x", "rho_lag_pp"], ["x", "rho_lag_pp"], rng=rng)
    rep["main_B_price_trimmed"]["trim_pp"] = [round(lo, 2), round(hi, 2)]
    rep["_meta"] = {"written": datetime.datetime.now().isoformat(timespec="seconds"), "seed": SEED,
                    "n_boot": N_BOOT, "window": [f"{T0[0]}Q{T0[1]}", f"{T1[0]}Q{T1[1]}"],
                    "panel_md5": got, "dep_floor_brl": DEP_FLOOR,
                    "units": "adv elasticities in logs of R$ of 2024Q4; price term per pp of the "
                             "annual national type-4 spread at t-1"}
    rep["p5"] = p5(d)
    write_panel(d)
    OUT_JSON.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    for k in ("main_B", "main_B_no_price", "big5_alone", "rest_alone", "D_main", "D_no_price",
              "main_B_price_trimmed"):
        stamp(f"{k}: {rep[k]['coef']} n={rep[k]['n_obs']} G={rep[k]['n_conglomerates']}")
    for k in ("groups_B", "ratio_B", "intensive_B"):
        stamp(f"{k}: " + json.dumps({x: rep[k][x] for x in rep[k] if x.startswith("adv") or x == "price"}))
    stamp(f"lags_B sums: big5 {rep['lags_B']['big5']['sum']}, rest {rep['lags_B']['rest']['sum']}")
    stamp(f"wrote {OUT_CSV.name}, {OUT_JSON.name}")


if __name__ == "__main__":
    main()
