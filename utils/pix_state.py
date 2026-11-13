"""utils/pix_state.py -- the Pix state of the sleepiness index, decided in one place.

Author: Pedro Feijó de Moraes

SLEEP_PIX_STATE selects it (unset means national):

  national   `pix_exists`, the national launch indicator (1 from 2020Q4), grand-mean centred through
             utils/state_transform and treated as a dummy. The specification every stored fit was
             estimated under.
  use_cf     Actual Pix use with a control function (Pedro, 11-13, A1.49). Two columns of the index:
               pix_use     Pix payers per inhabitant of the market (payers_pc, the variable of
                           make_pix_exposure.py), in levels;
               pix_use_cf  the residual of pix_use in its first stage on the three 2019 exposures.
             Both are 0 before 2020Q4: Pix did not exist, so zero use is not chosen and there is
             nothing to control for. Neither is centred or scaled, so "no Pix" is pix_use = 0 in
             estimation units too. The counterfactual of eq. (18) sets pix_use to 0 and holds
             pix_use_cf at its value (each market's attention shock fixed).

THE FIRST STAGE is the one of the paper's Section 7.1.2 (make_pix_exposure.first_stage), shared
with the availability index of Section 7: payers_pc on the three standardized 2019 exposures
(mandated institutions' deposit share, mobile lines per 100, 4G/5G share; population-weighted
moments over the MCAs with a 2019 exposure), quarter effects, 2019 Cadunico x quarter effects and
log population, population weights, 2020Q4 to 2024Q4. Its inputs are two files written by
make_pix_state_inputs.py into PANEL_INTERMED (on the cluster, data/input files linked there).

ROWS OUTSIDE THE FIRST-STAGE SAMPLE. A B market-quarter after the launch with no use data (0.003% of
the post-launch B deposits on panel fa38794d) takes its predicted use and a zero residual. A D row
(CODMUN_IBGE == '0', a national firm) takes the population-weighted national mean of use over the
first-stage markets of its quarter, and a zero residual: the population-weighted mean of the
residual is zero by construction, and a national aggregate has no market residual.

THE BOOTSTRAP of the generated regressor (sleep_pix_genreg.py) refits the first stage on wild draws
by state and rebuilds pix_use_cf with each draw's coefficients; use itself is data and does not move.
`wild_y(b)` gives draw b's outcome and `add_columns(df, beta=...)` the columns at any coefficients.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths as _paths

MODES = ("national", "use_cf")
USE_COL = "pix_use"
CF_COL = "pix_use_cf"
LAUNCH_T = 2020 * 4 + 3          # 2020Q4, as year * 4 + quarter - 1
EXPOSURES = ("mandated", "mobile", "g45")
SEED = 20261113                  # the wild draws of the first stage
N_DRAWS = 99
EXPOSURES_CSV = "pix_exposures_2019_mca.csv"
USE_CSV = "pix_use_mca_quarter.csv"


def mode() -> str:
    """The active Pix state; refuses anything but the known values."""
    m = os.environ.get("SLEEP_PIX_STATE", "").strip().lower() or "national"
    if m not in MODES:
        raise ValueError(f"SLEEP_PIX_STATE={m!r} is not one of {MODES}")
    return m


def state_cols() -> list:
    """The Pix columns of the Macro and Tech blocks, in block order."""
    return ["pix_exists"] if mode() == "national" else [USE_COL, CF_COL]


def main_col() -> str:
    """The column the no-Pix counterfactual moves."""
    return "pix_exists" if mode() == "national" else USE_COL


def continuous_main_cols() -> list:
    """Continuous Pix columns whose no-Pix level the link's knots must cover (none under national,
    whose dummy is handled by the dummy branch of utils.sleep_links._pix_shifted_range)."""
    return [] if mode() == "national" else [USE_COL]


def no_pix_level(col: str) -> float:
    """Value of a continuous Pix column without Pix, in estimation units (uncentred, unscaled)."""
    if col != USE_COL:
        raise KeyError(f"{col!r} has no no-Pix level here")
    return 0.0


def inputs_dir() -> Path:
    env = os.environ.get("PIX_STATE_DIR", "").strip()
    return Path(env) if env else _paths.PROCESSED / "PANEL_INTERMED"


@lru_cache(maxsize=1)
def _inputs():
    d0 = inputs_dir()
    x = pd.read_csv(d0 / EXPOSURES_CSV, dtype={"mca_code": str}).set_index("mca_code")
    u = pd.read_csv(d0 / USE_CSV, dtype={"mca_code": str, "state": str})
    w = x["pop2019"].to_numpy(float)
    mom = {}
    for e in EXPOSURES:
        v = x[e].to_numpy(float)
        mu = float(np.average(v, weights=w))
        mom[e] = (mu, float(np.sqrt(np.average((v - mu) ** 2, weights=w))))
    d = u.merge(x, left_on="mca_code", right_index=True, how="inner").dropna(
        subset=["pop_total", "cad2019", *EXPOSURES, "payers_pc"])
    d = d[d["pop_total"] > 0].reset_index(drop=True)
    qs = np.array(sorted(d["t"].unique()), dtype=int)
    cad_mean = float(np.average(x["cad2019"].to_numpy(float), weights=w))
    return x, d, mom, qs, cad_mean


def _z(x_rows: pd.DataFrame, mom) -> np.ndarray:
    return np.column_stack([(x_rows[e].to_numpy(float) - mom[e][0]) / mom[e][1] for e in EXPOSURES])


def _design(t, cad, lpop, Z, qs) -> np.ndarray:
    """Quarter effects, 2019 Cadunico x quarter effects (the first quarter dropped), log population
    and the standardized exposures, as make_pix_exposure.first_stage. Rows of a quarter outside the
    first-stage sample have no quarter effect, so they are not predicted here."""
    Q = (np.asarray(t, dtype=int)[:, None] == qs[None, :]).astype(float)
    return np.column_stack([Q, Q[:, 1:] * np.asarray(cad, float)[:, None],
                            np.asarray(lpop, float)[:, None], Z])


@lru_cache(maxsize=1)
def _fs_matrices():
    x, d, mom, qs, _ = _inputs()
    X = _design(d["t"].to_numpy(), d["cad2019"].to_numpy(), np.log(d["pop_total"].to_numpy(float)),
                _z(d, mom), qs)
    w = d["pop_total"].to_numpy(float)
    return X, w, d["payers_pc"].to_numpy(float), pd.factorize(d["state"].astype(str))[0]


def first_stage(y=None) -> dict:
    """The population-weighted first stage; `y` replaces the outcome (a bootstrap draw)."""
    X, w, y0, st = _fs_matrices()
    yy = y0 if y is None else np.asarray(y, float)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], yy * sw, rcond=None)
    fit = X @ beta
    return dict(beta=beta, fitted=fit, resid=yy - fit, n=int(len(yy)))


@lru_cache(maxsize=1)
def first_stage_hat() -> dict:
    return first_stage()


@lru_cache(maxsize=1)
def _wild_weights():
    """The 99 state-level Webb weight vectors, drawn in sequence from one generator."""
    from utils.sleep_links import _wild_weights as _ww
    _, _, _, st = _fs_matrices()
    rng = np.random.default_rng(SEED)
    G = int(st.max() + 1)
    return np.stack([_ww(G, "webb", rng) for _ in range(N_DRAWS)])


def wild_y(b: int) -> np.ndarray:
    """Draw b (1-based) of the first stage's outcome: fitted + w_state x residual."""
    if not 1 <= int(b) <= N_DRAWS:
        raise ValueError(f"draw {b} outside 1..{N_DRAWS}")
    fs = first_stage_hat()
    _, _, _, st = _fs_matrices()
    return fs["fitted"] + _wild_weights()[int(b) - 1][st] * fs["resid"]


def market_table(beta=None) -> pd.DataFrame:
    """(mca_code, t) -> pix_use, pix_use_cf over the first-stage sample, and the national use mean
    by quarter, at coefficients `beta` (default: the estimate)."""
    x, d, mom, qs, _ = _inputs()
    X, w, y0, _ = _fs_matrices()
    b = first_stage_hat()["beta"] if beta is None else np.asarray(beta, float)
    tab = pd.DataFrame({"mca_code": d["mca_code"].to_numpy(), "t": d["t"].to_numpy(int),
                        USE_COL: y0, CF_COL: y0 - X @ b})
    nat = (pd.Series(y0 * w).groupby(d["t"].to_numpy(int)).sum()
           / pd.Series(w).groupby(d["t"].to_numpy(int)).sum())
    return tab, nat


def add_columns(df: pd.DataFrame, beta=None, where: str = "") -> pd.DataFrame:
    """Write pix_use and pix_use_cf on a frame with mca_code, year, quarter, pop_total and
    CODMUN_IBGE. A no-op under SLEEP_PIX_STATE=national. Returns the frame (modified in place)."""
    if mode() == "national":
        return df
    x, d, mom, qs, cad_mean = _inputs()
    tab, nat = market_table(beta)
    b = first_stage_hat()["beta"] if beta is None else np.asarray(beta, float)
    t = (df["year"].astype(int) * 4 + df["quarter"].astype(int) - 1).to_numpy()
    post = t >= LAUNCH_T
    is_d = (df["CODMUN_IBGE"].astype(str) == "0").to_numpy()
    mca = df["mca_code"].astype(str).to_numpy()
    key = pd.MultiIndex.from_arrays([mca, t])
    tk = pd.MultiIndex.from_arrays([tab["mca_code"].to_numpy(), tab["t"].to_numpy()])
    use = pd.Series(tab[USE_COL].to_numpy(), index=tk).reindex(key).to_numpy(dtype=float, copy=True)
    cf = pd.Series(tab[CF_COL].to_numpy(), index=tk).reindex(key).to_numpy(dtype=float, copy=True)
    # Post-launch B rows without use data: predicted use from the exposures, zero residual.
    miss = post & ~is_d & np.isnan(use)
    if miss.any():
        if not np.isin(t[miss], qs).all():
            raise ValueError(f"{where}: a post-launch quarter outside the first stage's 2020Q4-2024Q4")
        xm = x.reindex(mca[miss])
        for e in EXPOSURES:
            xm[e] = xm[e].fillna(mom[e][0])
        cad = xm["cad2019"].fillna(cad_mean).to_numpy(float)
        # Population for the log-population term: the row's, else the market's 2019 population, else
        # the median of the first-stage sample.
        pop = df["pop_total"].to_numpy(float)[miss]
        bad = ~(np.isfinite(pop) & (pop > 0))
        if bad.any():
            p19 = xm["pop2019"].to_numpy(float)
            pop = np.where(bad & np.isfinite(p19) & (p19 > 0), p19, pop)
            bad = ~(np.isfinite(pop) & (pop > 0))
            pop = np.where(bad, float(np.median(d["pop_total"])), pop)
        use[miss] = _design(t[miss], cad, np.log(pop), _z(xm, mom), qs) @ b
        cf[miss] = 0.0
    # D rows: the national population-weighted mean of use; zero residual.
    if is_d.any():
        use[is_d] = np.where(post[is_d], pd.Series(t[is_d]).map(nat).to_numpy(), 0.0)
        cf[is_d] = 0.0
    use = np.where(post, use, 0.0)
    cf = np.where(post, cf, 0.0)
    if np.isnan(use).any() or np.isnan(cf).any():
        raise ValueError(f"{where}: {int(np.isnan(use).sum())} rows without a Pix use value")
    df[USE_COL] = use
    df[CF_COL] = cf
    n_miss = int(miss.sum())
    print(f"  [pix_state {mode()}] {where}: pix_use and pix_use_cf on {len(df):,} rows; "
          f"{n_miss:,} post-launch B rows take predicted use", flush=True)
    return df
