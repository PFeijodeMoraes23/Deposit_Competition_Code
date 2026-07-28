"""utils/state_transform.py — ONE source of truth for how sleepiness state
variables are transformed for estimation, and for the units they are REPORTED in.

Author: Pedro Feijó de Moraes

Two independent jobs live here on purpose, because they used to live in seven
places and drift:

1. THE ESTIMATION TRANSFORM  (changes theta; must be identical everywhere)
   ------------------------------------------------------------------------
   S_est = (S_raw / SCALE[k]) - mean_scaled[k]

   * SCALE was previously copy-pasted into four Python build sites plus a Julia
     mirror (`D_SCALE` in blp_1_draws.jl).  Every copy carried the comment "same
     scaling as estimation_*_sleep.build_pooled_data so the native index
     matches" -- which is exactly the invariant a copy cannot enforce.  phi is
     rebuilt DOWNSTREAM as params_native x parquet columns, so a site that
     scales differently from the site that estimated theta produces a silently
     wrong phi that still lies in [0,1]: no NaN, no exception, no assertion.
   * mean_scaled is the GRAND MEAN over the pooled estimation sample (NOT
     entity- or time-demeaning).  Subtracting it is a pure reparametrisation:
     slopes, AMEs and fitted phi are unchanged and only the constant moves,
         theta_0_new = theta_0_old + sum_k theta_k * mean_scaled[k],
     which turns theta_0 from "phi at S = 0" (a market with no people, no GDP
     and no broadband; it estimates to 1.72, outside [0,1]) into "phi-hat at the
     average market".

   WRITE ONCE / LOAD ALWAYS.  The means are computed by exactly one deliberate
   action (`estimation_2_sleep.py --write-centers`) and every other caller only
   ever LOADS them.  That is what makes E1 -- which drops digital banks after
   the shared prep and therefore has a different sample mean -- use the SAME
   S_bar as everyone else, which is the whole point of a constant that can be
   compared across the columns of one table.

2. THE DISPLAY UNITS  (never touch theta; tables only)
   ---------------------------------------------------
   Reported effects are in PERCENTAGE POINTS OF THE LHS, per the unit named in
   the row label, with shares and rates in percentage points.  A coefficient is
   multiplied (together with its standard error) by

       mult = LHS_DISPLAY * DISPLAY[k].u / SCALE[k]

   where DISPLAY[k].u is the size of ONE display unit measured in RAW panel
   units.  t-statistics, p-values and stars are unit-invariant and are never
   touched.

   Note the arithmetic: for a regressor that is already a decimal share or rate
   (fraction_65plus, risk_free_qoq_lag, ...), the /100 on the regressor and the
   x100 on phi CANCEL.  `Fraction 65+ = -2.4845` already IS "-2.48 pp of
   sleepiness per +1 pp elderly share" -- the number was never wrong, only its
   stated unit was.  Only genuine levels (gdp_per_capita, cadunico) move.

   The same registry serves the BBL policy functions
   (estimation_bbl_1_polfunc.py), whose units were previously pinned to the
   sleepiness tables BY COMMENT, and the descriptive tables, which report
   LEVELS and therefore divide by `u` instead of multiplying.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths as _paths_mod

# Bump when SCALE / CENTER / BINARY change in a way that invalidates a persisted
# JSON or an already-built demand parquet.  load_transform() refuses to load a
# JSON written under a different version.
STATE_TRANSFORM_VERSION = "2026-07-28.1"

DEFAULT_JSON = (_paths_mod.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP"
                / "state_centering_means.json")


def json_path() -> Path:
    """Persisted-transform location (STATE_CENTERING_JSON overrides, for tests)."""
    env = os.environ.get("STATE_CENTERING_JSON", "").strip()
    return Path(env) if env else DEFAULT_JSON


def centering_enabled() -> bool:
    """Master switch.  SLEEP_CENTER_STATES=0 reproduces the pre-centering
    parametrisation exactly (used to gate the AME refactor as a strict no-op
    before the substantive change, and as the rollback lever)."""
    return os.environ.get("SLEEP_CENTER_STATES", "1").strip() not in ("0", "false", "False")


# ==============================================================================
# 1. THE ESTIMATION TRANSFORM
# ==============================================================================
# raw panel units -> estimation units, as a DIVISOR.  These are the numbers that
# were duplicated at estimation_2_sleep.py:214-217, estimation_demand_link_common.py:193-196,
# estimation_1_demand_1_prep.py:225-228 and estimation_2_demand_1_prep.py:223-226.
# `indice_basileia_lag` is a MULTIPLY-by-100 at those sites, i.e. a divisor of 0.01.
SCALE = {
    'gdp_per_capita':            1e4,    # -> 10k R$
    'cadunico_families_per1000': 100.0,  # -> 100s of families per 1k inhabitants
    'connections_per100':        100.0,  # -> connections per inhabitant
    'indice_basileia_lag':       0.01,   # -> percentage points
    'pix_users_pf_per1000':      100.0,  # demand-side only (not a sleepiness state)
}

# Columns carrying a GRAND-MEAN subtraction.  Deliberately NOT 'constant' (it is
# the intercept), NOT 'nr_lagged_dep' (the regressor phi multiplies), NOT the
# instruments and NOT the control-function term -- centering those would change
# the model rather than reparametrise it.
CENTER = [
    'pix_exists',
    'gdp_per_capita',
    'cadunico_families_per1000',
    'fraction_65plus',
    'fraction_young',
    'risk_free_qoq_lag',
    'connections_per100',
    'gdp_growth_yoy',          # TIME_VARS; present only on time_block=True frames
]

# EXPLICIT dummy metadata, in RAW units.  This replaces the `len(unique)==2 and
# 0.0 in u and 1.0 in u` value sniff in utils/sleep_links.py, which silently
# misclassifies a CENTERED dummy (values {-p_bar, 1-p_bar}) as continuous and
# reverts its AME to a continuous average derivative -- the exact bug fixed on
# 2026-06-26.  Metadata cannot drift with the data; a value rule can.
BINARY = {
    'pix_exists': (0.0, 1.0),
}

# Never treated as a dummy even if a subsample happens to make it two-valued.
NEVER_BINARY = {'constant', 'nr_lagged_dep'}


def apply_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Raw panel units -> estimation units, in place.  Replaces the four
    duplicated scaling blocks.  Idempotence is NOT checked -- call exactly once
    per frame, immediately after the deposit /1e9 rescalings."""
    for col, f in SCALE.items():
        if col in df.columns:
            df[col] = df[col] / f
    return df


def compute_means(df: pd.DataFrame) -> dict:
    """Grand means of the CENTER columns over the frame as given.  The frame
    must already be scaled, dropna'd to the estimation sample, imputed, and
    carry the time block -- see the ordering contract in
    estimation_2_sleep.build_pooled_data."""
    return {c: float(np.nanmean(df[c].values.astype(float)))
            for c in CENTER if c in df.columns}


class StateTransform:
    """A loaded (scale, mean) pair plus the dummy levels implied by it."""

    def __init__(self, payload: dict):
        self.version = payload.get("version")
        self.scale = dict(payload.get("scale", {}))
        self.mean_scaled = dict(payload.get("mean_scaled", {}))
        self.mean_raw = dict(payload.get("mean_raw", {}))
        self.binary_levels = dict(payload.get("binary_levels", {}))
        self.n_rows = payload.get("n_rows")
        self.built_at = payload.get("built_at")

    def level_of(self, col: str, raw: float) -> float:
        """The ESTIMATION-units value of a raw level.  This is what a
        counterfactual that flips a state to a literal value must use:
        `dd0['pix_exists'] = tf.level_of('pix_exists', raw=0.0)`, never 0.0.

        Tracks `centering_enabled()`, so the flip stays correct under the
        SLEEP_CENTER_STATES=0 rollback path instead of silently subtracting a
        mean the data does not carry."""
        v = float(raw) / float(self.scale.get(col, 1.0))
        if centering_enabled():
            v -= float(self.mean_scaled.get(col, 0.0))
        return v

    def levels(self, col: str):
        """(lo, hi) in estimation units for a registered dummy, else None."""
        d = self.binary_levels.get(col)
        if not d:
            return None
        if centering_enabled():
            return (float(d["lo"]), float(d["hi"]))
        f = float(self.scale.get(col, 1.0))
        return (float(d["lo_raw"]) / f, float(d["hi_raw"]) / f)

    def center(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subtract the persisted grand means, in place.  No-op under
        SLEEP_CENTER_STATES=0."""
        if not centering_enabled():
            return df
        for col, m in self.mean_scaled.items():
            if col in df.columns:
                df[col] = df[col] - m
        return df


def write_transform(df: pd.DataFrame, path: Path | None = None) -> dict:
    """Compute and persist the grand means from `df`.  THE ONLY WRITER.

    `df` must be the pooled, scaled, imputed, time-block-carrying estimation
    frame, so that the persisted vector always has every CENTER entry -- a
    time_block=False frame would silently omit gdp_growth_yoy and leave two
    JSONs racing for one path."""
    path = Path(path) if path is not None else json_path()
    means = compute_means(df)
    missing = [c for c in CENTER if c not in means]
    if missing:
        raise ValueError(
            f"state_transform: cannot persist a partial mean vector; missing {missing}. "
            "write_transform() must be given the time_block=True pooled frame."
        )
    payload = {
        "version": STATE_TRANSFORM_VERSION,
        "built_by": "estimation_2_sleep.py --write-centers",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "sample": "pooled B+D estimation frame (build_pooled_data, time_block=True)",
        "scale": dict(SCALE),
        "mean_scaled": means,
        "mean_raw": {c: means[c] * float(SCALE.get(c, 1.0)) for c in means},
        "binary_levels": {
            c: {"lo": lo / float(SCALE.get(c, 1.0)) - means.get(c, 0.0),
                "hi": hi / float(SCALE.get(c, 1.0)) - means.get(c, 0.0),
                "gap": (hi - lo) / float(SCALE.get(c, 1.0)),
                "lo_raw": lo, "hi_raw": hi}
            for c, (lo, hi) in BINARY.items() if c in means
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    print(f"  [state_transform] wrote {path} (n={payload['n_rows']:,}, "
          f"{len(means)} means, version {STATE_TRANSFORM_VERSION})")
    return payload


_CACHE: dict = {}


def load_transform(path: Path | None = None, required: bool = True) -> StateTransform | None:
    """Load the persisted transform.  NEVER recomputes means on the fly: a
    silently-recomputed mean on a different sample is precisely the failure this
    module exists to prevent."""
    path = Path(path) if path is not None else json_path()
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    if not path.exists():
        if not required:
            return None
        raise FileNotFoundError(
            f"state_transform: {path} not found. Build it once with\n"
            f"    python estimation_2_sleep.py --write-centers\n"
            "and re-run. It is never regenerated implicitly."
        )
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    got = payload.get("version")
    if got != STATE_TRANSFORM_VERSION:
        raise ValueError(
            f"state_transform: {path} was written under version {got!r} but this code is "
            f"{STATE_TRANSFORM_VERSION!r}. Rebuild it (--write-centers) AND rebuild every demand "
            "parquet, or phi will be rebuilt from coefficients estimated on a different scale."
        )
    tf = StateTransform(payload)
    _CACHE[key] = tf
    return tf


def apply_transform(df: pd.DataFrame, tf: StateTransform | None = None) -> pd.DataFrame:
    """scale, then center.  The canonical call for every build site."""
    apply_scale(df)
    (tf if tf is not None else load_transform()).center(df)
    return df


def dummy_levels(col: str, tf: StateTransform | None = None):
    """(lo, hi) in estimation units for `col`, or None if it is not a dummy.

    Resolution order: the persisted JSON, then the BINARY registry under the
    current SCALE/centering state, then None.  Callers keep a value-based
    fallback for columns this registry does not know about."""
    if col in NEVER_BINARY:
        return None
    if tf is None:
        tf = load_transform(required=False)
    if tf is not None:
        lv = tf.levels(col)
        if lv is not None:
            return lv
    if col in BINARY:                       # no JSON yet (or centering off)
        lo, hi = BINARY[col]
        f = float(SCALE.get(col, 1.0))
        return (lo / f, hi / f)
    return None


def sniff_dummy_levels(col_values: np.ndarray):
    """Level-agnostic fallback for columns absent from BINARY: exactly two
    distinct finite values.  Deliberately NOT keyed on {0,1} -- that test is
    what breaks under centering -- and deliberately not used for registered
    columns, where metadata is authoritative."""
    u = np.unique(col_values[~np.isnan(col_values)])
    if len(u) == 2:
        return (float(u[0]), float(u[1]))
    return None


# ==============================================================================
# 2. THE DISPLAY UNITS
# ==============================================================================
# var -> (u, label).  u = the size of ONE display unit, in RAW panel units.
#   a decimal share/rate shown "per pp"      -> u = 0.01
#   a level shown per 10k R$                 -> u = 1e4
#   a 0/1 indicator (discrete 0 -> 1 effect) -> u = 1.0
_PP    = (0.01, 'pp')
_PP2   = (1e-4, 'pp$^{2}$')

DISPLAY = {
    # ---- sleepiness state block -------------------------------------------------
    'pix_exists':                (1.0,   r'$0\to1$'),   # discrete difference, not a derivative
    'gdp_per_capita':            (1e4,   r'10k R\$'),
    'cadunico_families_per1000': (100.0, '100s per 1k'),
    'fraction_65plus':           _PP,
    'fraction_young':            _PP,
    'risk_free_qoq_lag':         (0.01,  'pp per quarter'),
    'connections_per100':        (1.0,   'per 100 inhab.'),
    'gdp_growth_yoy':            (0.01,  'pp YoY'),
    'time_trend':                (1.0,   'years'),
    # ---- instruments / first-stage controls -------------------------------------
    'personnel_cost_ratio_lag':  _PP,
    'admin_cost_ratio_lag':      _PP,
    'tax_cost_ratio_lag':        _PP,
    'wholesale_ratio_lag':       _PP,
    'lci_lca_ratio_lag':         _PP,
    'indice_basileia_lag':       _PP,
    'leave_one_out_mean_spread': _PP,
    'estban_rival_branches_lag': (1.0,   'log pt'),
    # ---- BBL policy-function regressors (migrated from estimation_bbl_1_polfunc) --
    'branches_per1000':          (1.0,   'per 1k'),
    'pix_users_pf_per1000':      (100.0, 'per 100 per 1k'),
    'equity_ratio_lag':          _PP,
    'equity_ratio_lag_sq':       _PP2,
    'asset_return_qoq_lag':      _PP,
    'npl_provision_ratio_lag':   _PP,
    'credit_assets_lag':         _PP,
    'risk_free_qoq':             _PP,
    'risk_free_qoq_sq':          _PP2,
    'log_total_assets_lag':      (1.0,   'log pt'),
    'log_total_assets_lag_sq':   (1.0,   '(log pt)$^{2}$'),
    'has_ip':                    (1.0,   'indicator'),
}

# LHS display factors.  phi and the deposit spread are both shares/rates in
# [0,1]-ish decimals, so both are reported in percentage points.
PHI_DISPLAY    = 100.0   # sleepiness second stage: pp of the sleepy share
SPREAD_DISPLAY = 100.0   # sleepiness first stage: pp of the quarterly spread

# Rows that are NOT an effect on the dependent variable's scale and so take no LHS
# factor.  The control-function coefficient multiplies v_hat x lagged deposits: it is
# deposits per deposit, already unit-free, and scaling it by 100 would be meaningless.
NON_LHS_PARAMS = {'v_hat_x_lagged_dep'}


def display_unit(var: str):
    """(u, label) for a regressor; `_natl` variants inherit the base unit."""
    v = str(var).replace('interaction_', '')
    base = v[:-5] if v.endswith('_natl') else v
    return DISPLAY.get(base, (1.0, ''))


def display_mult(var: str, lhs: float = PHI_DISPLAY, scaled: bool = True) -> float:
    """Multiplier for a COEFFICIENT and its SE (never for t, p or stars).

    `lhs`    : display factor of the dependent variable (PHI_DISPLAY / SPREAD_DISPLAY / 1.0).
    `scaled` : True when the regressor was divided by SCALE before estimation
               (the sleepiness/demand chain); False when the caller built the
               column raw (the BBL policy functions).
    """
    v = str(var).replace('interaction_', '')
    base = v[:-5] if v.endswith('_natl') else v
    if base in NON_LHS_PARAMS:
        return 1.0
    u, _ = display_unit(base)
    f = float(SCALE.get(base, 1.0)) if scaled else 1.0
    return float(lhs) * float(u) / f


def display_level_div(var: str) -> float:
    """Divisor turning a RAW level into display units, for descriptive tables:
    a mean elderly share of 0.085 prints as 8.5 when its coefficient is per pp."""
    return float(display_unit(var)[0])


def label_with_unit(base_label: str, var: str) -> str:
    """Attach the display unit to a row label.

        'Fraction 65+'          -> 'Fraction 65+ (pp)'
        'Basel Index ($t-1$)'   -> 'Basel Index (pp, $t-1$)'

    A label that already ends in a parenthetical (the house style for lag markers)
    gets the unit merged INTO it rather than a second pair of brackets appended --
    otherwise the lag-suffixed instruments would silently lose their unit while the
    table notes promise one on every row."""
    base = base_label.rstrip()
    _, unit = display_unit(var)
    if not unit:
        return base_label
    if base.endswith(')'):
        i = base.rfind('(')
        if i > 0:
            return f"{base[:i]}({unit}, {base[i + 1:]}"
        return base_label
    return f"{base} ({unit})"


__all__ = [
    "STATE_TRANSFORM_VERSION", "SCALE", "CENTER", "BINARY", "NEVER_BINARY",
    "DISPLAY", "PHI_DISPLAY", "SPREAD_DISPLAY",
    "json_path", "centering_enabled", "apply_scale", "compute_means",
    "StateTransform", "write_transform", "load_transform", "apply_transform",
    "dummy_levels", "sniff_dummy_levels",
    "display_unit", "display_mult", "display_level_div", "label_with_unit",
]
