"""
scrape_forward_rf.py
================
Build forward risk-free (Selic) paths r^f_t and cache them as CSVs that the Julia CF
engines and the BBL forward simulation read (bbl_fwd_sim.jl psi4 funding base; later
CF3/CF5 equilibrium re-solves).

WHY A REAL FORWARD CURVE (not a flat placeholder). In the BBL value basis (eq:16),
psi4 = SUM_t beta^t r^f_t . SUM_k Dep_t enters with coefficient -(1+zeta). If r^f_t is
held FLAT, psi4 = r^f . SUM_t beta^t SUM_k Dep_t is just a rescaling of
psi2 = SUM_t beta^t SUM_k Dep_t, so omega and zeta are COLLINEAR and zeta
(funding-cost pass-through) is unidentified.

WHY ONE CURVE IS NOT ENOUGH (the vintage design). A single forward curve gives every
simulated firm-episode the SAME rate path, so psi4 and psi2 still move together across
episodes: only the deposit base varies, and the two basis columns stay ~collinear
(measured corr(dpsi2, dpsi4) = 0.99997 with the single 2026 curve). Launching the
forward simulation from MANY historical start quarters, each discounting the rate path
that agents actually expected AT THAT DATE, is what breaks the ridge: Selic ran from
14.25% (2016) to 2.00% (2020-21) and back to 14.75%+ (2024-25), so different vintages
carry genuinely different discounted funding costs for the same deposit base.

These vintage curves are also the conditional MEAN of a stochastic rate process --
dispersion around them is added by bbl_transitions.py -- so the accuracy of the mean
path matters, not just its cross-vintage spread.

SOURCE (BCB open APIs, no key):
  * SGS 4189 - Selic accumulated in the month, ANNUALIZED (% p.a.): the level anchor at
    the vintage month.
  * SGS 4390 - Selic accumulated in the month (% in the month): compounded over a
    quarter's three months this is the REALISED quarterly r^f. It reproduces the panel's
    own `risk_free_qoq` (which panel_deposit_rates.py builds by compounding daily SGS 11
    over the quarter) to within 0.0016 in quarterly decimal over 2013Q1-2025Q4 -- the
    residual is 4390's two-decimal rounding of the monthly percent.
  * Expectations (Focus) - ExpectativasMercadoAnuais, Indicador='Selic': the market
    median annual Selic per reference year (DataReferencia), as of a survey date (Data).

WHY ExpectativasMercadoAnuais AND NOT ExpectativasMercadoSelic. The meeting-level
resource (ExpectativasMercadoSelic) carries a median per COPOM meeting and would give a
finer near-term shape, which matters under beta=0.9 discounting. It is NOT used because:
  (a) HORIZON. A 2016 or 2020 vintage covers only 12 meetings there -- roughly 1.5 years
      -- against 5 reference years in the annual resource. The BBL horizon is 50
      quarters, so the annual resource is the one that actually spans the simulation.
  (b) NO DATE ON A MEETING. Its key is a label ("R3/2017"), not a date. Mapping labels to
      calendar quarters needs the COPOM meeting calendar, an external dependency that
      changes (the meetings-per-year count has not always been 8) and that would fail
      SILENTLY as a mis-dated node rather than loudly as a missing one.
  (c) The near-term gain is small once the path is anchored on the REALISED SGS 4189
      level at the vintage month, which already pins h=0..1 exactly.

baseCalculo. Focus publishes each (Data, DataReferencia) twice from mid-2021 on:
baseCalculo=0 (respondents from the last 30 days, ~130 of them) and baseCalculo=1 (last
5 business days, ~30). Only base 0 exists over the whole 2016-2024 vintage span, and it
is the deeper sample, so every query here filters `baseCalculo eq 0` server-side. This
also removes a latent nondeterminism: without the filter the two bases collide on the
same reference year and the median kept is whichever row the service happened to return
last.

NODE PLACEMENT (a correction to the earlier single-curve build). The Focus annual Selic
expectation is an END-of-period rate ("Selic - fim de periodo"): the median for reference
year Y is the rate expected to prevail in DECEMBER of Y, not its yearly average. Year Y's
node therefore sits at calendar position Y+1.0, and each quarter is placed at its own
MIDPOINT (y + (q-1)/4 + 1/8). The earlier build placed annual medians at mid-year
(Y+0.5) and quarters at their start, which pulled every rate change forward by roughly
1.5 quarters.

SPLICE FALLBACK (perfect foresight). If a vintage's survey returns fewer than two
reference years, or the Focus query fails, the path is spliced: REALISED quarterly Selic
(SGS 4390) from the start quarter up to the last realised quarter, then the terminal
(latest) Focus curve beyond it. That is a perfect-foresight path, not an expectation, so
it is tagged `splice:sgs4390(<q0>..<q1>)|focus:<date>` and never mistaken for the live
vintage curve. `--force-splice` produces it deliberately as a robustness variant.

OFFLINE FALLBACK for the single terminal curve (documented, deterministic): if the APIs
are unreachable, mean-revert the last observed panel Selic to a neutral rate with a
quarterly AR coefficient.

Annual -> quarterly throughout: r^f_q = (1 + Selic/100)^(1/4) - 1.

Outputs (processed/ESTIMATION_OUTPUT/COST_FWD/):
  forward_rf_qoq.csv       h (1..T), cal_q, selic_ann_pct, rf_qoq, source
                           the single terminal curve; unchanged interface.
  forward_rf_vintages.csv  start_q, vintage_date, h, cal_q, selic_ann_pct, rf_qoq, source
                           one block per start quarter. h=0 is the REALISED rate at the
                           start quarter (a consistency check against the panel's own
                           risk_free_qoq); h=1..T are the forward quarters AFTER it.
                           `source` is constant within a start_q block and describes the
                           forward leg -- h=0 is realised regardless of what it says.
  forward_rf_vintages.png  overlay of the vintage curves (--plot).

Usage:
  python scrape_forward_rf.py --horizon 50 --start 2026Q1          # terminal curve
  python scrape_forward_rf.py --horizon 50 --start 2026Q1 --offline
  python scrape_forward_rf.py --vintage-from 2016Q1 --vintage-to 2024Q4 --plot
  python scrape_forward_rf.py --vintages "2016Q1,2018Q3,2020Q4,2023Q1"
  python scrape_forward_rf.py --vintage-from 2016Q1 --vintage-to 2024Q4 --force-splice
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import datetime as _dt
import json
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path

# The status lines print Greek/arrows (r^f path "->"); Windows consoles default to cp1252 and
# raise UnicodeEncodeError on them, which crashes the script AFTER the CSV is already written
# -- i.e. it reports failure for work that succeeded. Force UTF-8 (no-op where already UTF-8).
# Same guard as sleep_upsilon_export.py.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[2]
COST_FWD = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "ESTIMATION_OUTPUT" / "COST_FWD"
DEMAND_PREP = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "ESTIMATION_OUTPUT" / "DEMAND_PREP"

SGS_SELIC = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados/ultimos/1?formato=json"
SGS_RANGE = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.{sid}/dados"
             "?formato=json&dataInicial={d0}&dataFinal={d1}")
SGS_ANN = 4189   # Selic accumulated in the month, ANNUALIZED (% p.a.)  -> level anchor
SGS_ACC = 4390   # Selic accumulated in the month (% in the month)      -> realised quarters
FOCUS_URL = ("https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
             "ExpectativasMercadoAnuais")
NEUTRAL_FALLBACK = 10.0   # % p.a. long-run Selic if Focus is unreachable
AR_FALLBACK = 0.85        # quarterly mean-reversion speed for the offline path
BETA_REPORT = 0.9         # discount factor used for the reported beta-weighted mean rate
MAX_PLAUSIBLE_ANN = 50.0  # % p.a.; any node above this is a parse error, not a rate


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _get_json(url: str, timeout: float = 30.0, tries: int = 4):
    """GET + parse JSON, retrying transient failures with linear backoff.

    A 36-vintage build fires ~40 requests at two BCB gateways, and both return sporadic
    502/503 under load. Without a retry one bad gateway kills the whole run after minutes
    of successful downloads, so transient errors are retried and only the last one raises.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code < 500 and e.code != 429:   # a real 4xx is not going to fix itself
                raise
        except Exception as e:
            last = e
        if i < tries - 1:
            time.sleep(2.0 * (i + 1))
    raise last


def _fetch_current_selic():
    """Latest annualized Selic (% p.a.) from SGS 4189."""
    js = _get_json(SGS_SELIC)
    return float(js[-1]["valor"])


# ---------------------------------------------------------------------------
# Calendar helpers.  A quarter is placed at its MIDPOINT on the fractional-year
# axis, so 2016Q1 -> 2016.125; a month at its midpoint, so 2016-03 -> 2016.2083.
# Both must live on the same axis as the Focus year nodes (Y+1.0 = end of Y).
# ---------------------------------------------------------------------------
def _q_to_tq(q: str) -> int:
    """'2016Q1' -> 8065 (year*4 + quarter), the panel's own time key."""
    return int(q[:4]) * 4 + int(q[5])


def _tq_to_q(tq: int) -> str:
    y = (tq - 1) // 4
    return f"{y}Q{tq - 4 * y}"


def _q_fpos(q: str) -> float:
    return int(q[:4]) + (int(q[5]) - 1) / 4.0 + 0.125


def _month_fpos(year: int, month: int) -> float:
    return year + (month - 0.5) / 12.0


def _q_last_day(q: str) -> _dt.date:
    """Last calendar day of 'YYYYQn' -- the survey date a vintage is read at."""
    y, qn = int(q[:4]), int(q[5])
    m = 3 * qn
    if m == 12:
        return _dt.date(y, 12, 31)
    return _dt.date(y, m + 1, 1) - _dt.timedelta(days=1)


def _cal_quarters(start: str, T: int):
    """['2026Q1', '2026Q2', ...] of length T from a 'YYYYQn' start."""
    y, qn = int(start[:4]), int(start[5])
    out = []
    for _ in range(T):
        out.append(f"{y}Q{qn}")
        qn += 1
        if qn > 4:
            qn = 1
            y += 1
    return out


def _annual_to_quarterly(selic_ann_pct):
    return (1.0 + np.asarray(selic_ann_pct, dtype=float) / 100.0) ** 0.25 - 1.0


def _quarterly_to_annual(rf_qoq):
    return 100.0 * ((1.0 + np.asarray(rf_qoq, dtype=float)) ** 4 - 1.0)


# ---------------------------------------------------------------------------
# BCB fetchers
# ---------------------------------------------------------------------------
_SGS_CACHE: dict = {}


def _fetch_sgs_range(series: int, d0, d1) -> pd.DataFrame:
    """SGS series `series` over [d0, d1] as DataFrame(dt: Timestamp, valor: float).

    d0/d1 accept 'YYYY-MM-DD' or datetime.date. SGS rejects very long windows on some
    series, so the range is requested in <=9-year chunks and concatenated. Results are
    memoised: a 36-vintage run must not re-download the same monthly series 36 times.
    """
    if isinstance(d0, str):
        d0 = _dt.date.fromisoformat(d0)
    if isinstance(d1, str):
        d1 = _dt.date.fromisoformat(d1)
    key = (int(series), d0, d1)
    if key in _SGS_CACHE:
        return _SGS_CACHE[key]

    frames = []
    lo = d0
    while lo <= d1:
        hi = min(d1, _dt.date(lo.year + 9, lo.month, 1) - _dt.timedelta(days=1))
        url = SGS_RANGE.format(sid=int(series),
                               d0=lo.strftime("%d/%m/%Y"), d1=hi.strftime("%d/%m/%Y"))
        js = _get_json(url)
        if js:
            frames.append(pd.DataFrame(js))
        lo = hi + _dt.timedelta(days=1)

    if not frames:
        raise RuntimeError(f"SGS {series} returned no rows for {d0}..{d1}")
    df = pd.concat(frames, ignore_index=True)
    df["dt"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["valor"]).drop_duplicates(subset=["dt"]).sort_values("dt")
    df = df[["dt", "valor"]].reset_index(drop=True)
    _SGS_CACHE[key] = df
    return df


def _sgs_monthly(series: int, d0, d1) -> pd.Series:
    """SGS monthly series keyed by (year, month)."""
    df = _fetch_sgs_range(series, d0, d1)
    return pd.Series(df["valor"].to_numpy(),
                     index=pd.MultiIndex.from_arrays([df["dt"].dt.year, df["dt"].dt.month]))


def _realised_quarterly_selic(d0, d1) -> pd.Series:
    """Realised quarterly r^f keyed by tq = year*4 + quarter, from SGS 4390.

    4390 is the Selic accumulated WITHIN each month (% in the month), so the quarter's
    realised return is the product over its three months. Quarters with fewer than three
    observed months are dropped rather than annualised from a partial window.
    """
    df = _fetch_sgs_range(SGS_ACC, d0, d1)
    tq = df["dt"].dt.year * 4 + df["dt"].dt.quarter
    g = (1.0 + df["valor"] / 100.0).groupby(tq)
    out = g.prod() - 1.0
    return out[g.size() == 3]


def _fetch_focus_annual(vintage_date=None):
    """Focus annual Selic medians as of a survey date.

    Returns (survey_date_used, {reference_year:int -> median Selic % p.a.}).

    vintage_date=None  -> the LATEST survey (the historical no-argument behaviour).
    vintage_date='YYYY-MM-DD' or a date -> the latest survey with Data <= that date, so
    the curve uses only information the market actually had at the vintage.

    Only baseCalculo=0 (30-day respondent window) is requested: it is the sole base that
    exists across the whole 2016-2024 span, it is the deeper sample, and pinning it
    removes the reference-year collision between the two bases.
    """
    # Build the odata query manually so $-params stay literal and spaces become %20
    # (olinda rejects the urlencode default of %24 for $ and + for space).
    flt = "Indicador eq 'Selic' and baseCalculo eq 0"
    if vintage_date is not None:
        if isinstance(vintage_date, (_dt.date, _dt.datetime)):
            vintage_date = vintage_date.strftime("%Y-%m-%d")
        flt += f" and Data le '{vintage_date}'"
    # $orderby=Data desc puts the newest admissible survey first; 200 rows is ~40 business
    # days at 5 reference years each, so the top survey and ALL its years are always inside.
    params = ("$top=200&$orderby=Data desc&$format=json"
              f"&$select=Data,DataReferencia,Mediana&$filter={flt}")
    url = FOCUS_URL + "?" + urllib.parse.quote(params, safe="$=&,'")
    js = _get_json(url)
    rows = js.get("value", [])
    if not rows:
        raise RuntimeError(f"Focus returned no rows (vintage={vintage_date})")
    latest = max(r["Data"] for r in rows)
    out = {}
    for r in rows:
        if r["Data"] != latest:
            continue
        try:
            yr = int(str(r["DataReferencia"])[:4])
        except (ValueError, TypeError):
            continue
        out[yr] = float(r["Mediana"])
    if not out:
        raise RuntimeError("Focus had no annual Selic reference years")
    return latest, out


def _last_panel_selic_ann():
    """Fallback anchor: last observed LEVEL r^f -> annualized %.

    Never `risk_free_qoq_lag` - utils/state_transform.CENTER grand-mean centres it, so the
    parquet ships a deviation (mean ~0). Reading it here anchored the curve at ~2%/yr instead
    of ~11%/yr for 2024Q4. A centred column is rejected rather than silently used.
    """
    cands = sorted(DEMAND_PREP.glob("demand_3_index*spec_12.parquet"))
    if not cands:
        cands = sorted(DEMAND_PREP.glob("demand_*spec_12.parquet"))
    schema = pq.read_schema(cands[-1]).names
    col = next((c for c in ("risk_free_qoq", "risk_free_qoq_lag_level") if c in schema), None)
    if col is None:
        raise SystemExit(
            f"{cands[-1].name} carries no LEVEL r^f column (looked for risk_free_qoq, "
            "risk_free_qoq_lag_level). Rebuild the demand parquets — see "
            "sleep_demand_prep_e1.py CF_COST_COLS.")
    df = pd.read_parquet(cands[-1], columns=["time_id", col])
    df["t"] = df["time_id"].astype(str)
    last = df[df["t"] == df["t"].max()]
    rf_q = pd.to_numeric(last[col], errors="coerce").median()
    if not (rf_q > 0.002):
        raise SystemExit(
            f"'{col}' median is {rf_q:.6f} — a quarterly Selic level is strictly positive, so "
            "this column looks grand-mean centred. Refusing to anchor the forward curve on it.")
    return 100.0 * ((1.0 + rf_q) ** 4 - 1.0)


# ---------------------------------------------------------------------------
# Curve construction
# ---------------------------------------------------------------------------
def _focus_curve_at(fpos, anchor_ann: float, anchor_fpos: float, focus: dict) -> np.ndarray:
    """Annualized Selic on the fractional-year grid `fpos`.

    Nodes: the realised level anchor at `anchor_fpos`, then one Focus median per reference
    year Y placed at Y+1.0 -- the END of Y, because the Focus annual Selic expectation is an
    end-of-period rate, not a yearly average. Linear in calendar time between nodes, flat at
    the anchor before the first node and flat at the last median beyond the Focus horizon
    (that last median is the market's long-run/neutral rate).
    """
    pos = [float(anchor_fpos)]
    val = [float(anchor_ann)]
    for y in sorted(focus):
        p = float(y) + 1.0
        if p > pos[-1] + 1e-9:      # keep xp strictly increasing for np.interp
            pos.append(p)
            val.append(float(focus[y]))
    pos = np.asarray(pos, dtype=float)
    val = np.asarray(val, dtype=float)
    return np.interp(np.asarray(fpos, dtype=float), pos, val, left=val[0], right=val[-1])


def build_path(T: int, start: str, offline: bool):
    """The single TERMINAL curve written to forward_rf_qoq.csv (h = 1..T, cal[0] = start).

    Interface unchanged. The curve itself now uses end-of-year Focus nodes and quarter
    midpoints (see the module docstring), and the spot SGS 4189 level is a real anchor node
    rather than a decoration in the `source` string.
    """
    cal = _cal_quarters(start, T)
    fpos = np.array([_q_fpos(c) for c in cal], dtype=float)

    try:
        if offline:
            raise RuntimeError("offline requested")
        cur = _fetch_current_selic()
        survey, focus = _fetch_focus_annual()
        # Anchor at the spot Selic, dated at the survey month so it sits on the same
        # calendar axis as the end-of-year Focus nodes.
        sd = _dt.date.fromisoformat(str(survey)[:10])
        anchor_fpos = _month_fpos(sd.year, sd.month)
        neutral = focus[sorted(focus)[-1]]
        selic_ann = _focus_curve_at(fpos, cur, anchor_fpos, focus)
        source = f"focus:{survey}|sgs4189:{cur:.2f}|neutral:{neutral:.2f}"
    except Exception as e:  # offline / API change -> deterministic mean-reversion
        anchor = _last_panel_selic_ann()
        neutral = NEUTRAL_FALLBACK
        selic_ann = np.empty(T)
        prev = anchor
        for i in range(T):
            prev = neutral + AR_FALLBACK * (prev - neutral)
            selic_ann[i] = prev
        source = (f"FALLBACK:meanrev(anchor={anchor:.2f}->neutral={neutral:.2f},"
                  f"ar={AR_FALLBACK}) [{type(e).__name__}]")

    rf_qoq = _annual_to_quarterly(selic_ann)
    return pd.DataFrame(dict(h=np.arange(1, T + 1), cal_q=cal,
                             selic_ann_pct=np.round(selic_ann, 4),
                             rf_qoq=rf_qoq, source=source))


def build_vintage_path(start_q: str, T: int, realised: pd.Series, ann_level: pd.Series,
                       terminal: tuple, force_splice: bool = False) -> pd.DataFrame:
    """The forward r^f path an agent standing at the END of `start_q` would discount with.

    Rows: h=0 carries the REALISED quarterly rate of `start_q` itself (the consistency
    check against the panel's risk_free_qoq); h=1..T are the T quarters AFTER it.

    Vintage date = the last calendar day of `start_q`, i.e. the newest Focus survey whose
    information set the agent actually had. Level anchor = SGS 4189 for the last month of
    `start_q`. Nodes = that survey's medians at end-of-year positions.

    Falls back to the realised/terminal splice when the survey carries fewer than two
    reference years, when the Focus query fails, or when `force_splice` is set.

    realised   : tq -> realised quarterly r^f (SGS 4390 compounded)
    ann_level  : (year, month) -> SGS 4189 annualized level
    terminal   : (survey_date, focus_dict, anchor_ann, anchor_fpos) for the latest survey
    """
    tq0 = _q_to_tq(start_q)
    if tq0 not in realised.index:
        raise ValueError(f"no realised quarterly Selic for start quarter {start_q}")
    cal = _cal_quarters(_tq_to_q(tq0 + 1), T)
    fpos = np.array([_q_fpos(c) for c in cal], dtype=float)
    vint = _q_last_day(start_q)

    reason = "forced" if force_splice else None
    survey, focus = None, {}
    if not force_splice:
        try:
            survey, focus = _fetch_focus_annual(vint)
            if len(focus) < 2:
                reason = f"only {len(focus)} reference year(s) in survey {survey}"
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"

    if reason is None:
        anchor = float(ann_level.loc[(vint.year, vint.month)])
        selic_ann = _focus_curve_at(fpos, anchor, _month_fpos(vint.year, vint.month), focus)
        neutral = focus[sorted(focus)[-1]]
        source = f"focus:{survey}|sgs4189:{anchor:.2f}|neutral:{neutral:.2f}"
        rf = _annual_to_quarterly(selic_ann)
    else:
        # Perfect-foresight splice: realised quarters first, terminal Focus curve beyond.
        t_survey, t_focus, t_anchor, t_apos = terminal
        last_real = int(realised.index.max())
        tqs = np.arange(tq0 + 1, tq0 + 1 + T)
        is_real = tqs <= last_real
        rf = np.where(is_real,
                      realised.reindex(tqs).to_numpy(),
                      _annual_to_quarterly(_focus_curve_at(fpos, t_anchor, t_apos, t_focus)))
        selic_ann = _quarterly_to_annual(rf)
        if is_real.any():
            span = f"{_tq_to_q(int(tqs[is_real][0]))}..{_tq_to_q(int(tqs[is_real][-1]))}"
            source = f"splice:sgs4390({span})|focus:{t_survey}"
        else:
            source = f"splice:sgs4390(none)|focus:{t_survey}"
        source += f" [{reason}]"

    rf0 = float(realised.loc[tq0])
    return pd.DataFrame(dict(
        start_q=start_q,
        vintage_date=vint.isoformat(),
        h=np.arange(0, T + 1),
        cal_q=[start_q] + cal,
        selic_ann_pct=np.round(np.concatenate([[_quarterly_to_annual(rf0)], selic_ann]), 4),
        rf_qoq=np.round(np.concatenate([[rf0], np.asarray(rf, dtype=float)]), 10),
        source=source))


def _vintage_starts(args) -> list:
    if args.vintages:
        return [s.strip().upper() for s in args.vintages.split(",") if s.strip()]
    lo, hi = _q_to_tq(args.vintage_from.upper()), _q_to_tq(args.vintage_to.upper())
    if hi < lo:
        raise SystemExit(f"--vintage-to {args.vintage_to} precedes --vintage-from {args.vintage_from}")
    return [_tq_to_q(t) for t in range(lo, hi + 1, max(1, int(args.vintage_every)))]


def _beta_weighted_mean(rf: np.ndarray, beta: float = BETA_REPORT) -> float:
    """SUM_h beta^h rf_h / SUM_h beta^h over h = 1..len(rf) (the h=0 row is excluded)."""
    w = beta ** np.arange(1, len(rf) + 1)
    return float(np.dot(w, rf) / w.sum())


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def _panel_rf() -> pd.Series:
    """Panel risk_free_qoq keyed by tq = year*4 + quarter (one national value per quarter)."""
    from utils import paths
    pq_path = str(paths.market_panel_csv()).replace(".csv", ".parquet")
    df = pd.read_parquet(pq_path, columns=["year", "quarter", "risk_free_qoq"])
    df["tq"] = df["year"].astype(int) * 4 + df["quarter"].astype(int)
    return df.groupby("tq")["risk_free_qoq"].median()


def _check_vintages(df: pd.DataFrame, tol: float = 0.002):
    """Anchor check + sanity bounds. Returns (anchor_failures, bound_violations)."""
    anchor_fail, bad = [], []
    try:
        panel = _panel_rf()
    except Exception as e:
        print(f"  [warn] panel anchor check skipped ({type(e).__name__}: {e})")
        panel = None

    if panel is not None:
        for sq, g in df[df["h"] == 0].groupby("start_q"):
            tq = _q_to_tq(sq)
            if tq not in panel.index:
                anchor_fail.append((sq, float(g["rf_qoq"].iloc[0]), np.nan,
                                    "quarter absent from panel"))
                continue
            d = float(g["rf_qoq"].iloc[0]) - float(panel.loc[tq])
            if abs(d) > tol:
                anchor_fail.append((sq, float(g["rf_qoq"].iloc[0]), float(panel.loc[tq]), f"|d|={abs(d):.5f}"))

    r, a = df["rf_qoq"].to_numpy(float), df["selic_ann_pct"].to_numpy(float)
    m = ~np.isfinite(r) | ~np.isfinite(a) | (r <= 0) | (a <= 0) | (a >= MAX_PLAUSIBLE_ANN)
    if m.any():
        bad = df.loc[m, ["start_q", "h", "cal_q", "selic_ann_pct", "rf_qoq"]].to_dict("records")
    return anchor_fail, bad


def _plot_vintages(df: pd.DataFrame, png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    starts = sorted(df["start_q"].unique(), key=_q_to_tq)
    cm = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for i, sq in enumerate(starts):
        g = df[df["start_q"] == sq].sort_values("h")
        spl = str(g["source"].iloc[0]).startswith("splice")
        ax.plot(g["h"], g["selic_ann_pct"], lw=1.2,
                ls="--" if spl else "-",
                color=cm(i / max(1, len(starts) - 1)), label=sq)
    ax.set_xlabel("h (quarters after the start quarter; h=0 is realised)")
    ax.set_ylabel("expected Selic, % p.a.")
    ax.set_title(f"Forward Selic curves by vintage ({starts[0]}-{starts[-1]}); "
                 "dashed = realised/terminal splice")
    ax.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, len(starts) - 1))
    cb = fig.colorbar(sm, ax=ax, ticks=range(0, len(starts), max(1, len(starts) // 8)))
    cb.ax.set_yticklabels([starts[i] for i in range(0, len(starts), max(1, len(starts) // 8))])
    cb.set_label("start quarter")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _run_vintages(args):
    starts = _vintage_starts(args)
    tq_lo, tq_hi = _q_to_tq(starts[0]), _q_to_tq(starts[-1])
    d0 = _dt.date((tq_lo - 1) // 4 - 1, 1, 1)
    d1 = _dt.date.today()
    how = "explicit list" if args.vintages else f"every {args.vintage_every}q"
    print(f"  vintages: {len(starts)} start quarters {starts[0]}..{starts[-1]} "
          f"({how}), horizon {args.horizon}")

    realised = _realised_quarterly_selic(d0, d1)
    ann_level = _sgs_monthly(SGS_ANN, d0, d1)
    t_survey, t_focus = _fetch_focus_annual()
    t_cur = _fetch_current_selic()
    t_last = ann_level.index[-1]
    terminal = (t_survey, t_focus, t_cur, _month_fpos(int(t_last[0]), int(t_last[1])))
    print(f"  SGS 4390 realised quarters: {_tq_to_q(int(realised.index.min()))}.."
          f"{_tq_to_q(int(realised.index.max()))}   terminal Focus survey: {t_survey}")

    blocks, skipped = [], []
    for sq in starts:
        try:
            blocks.append(build_vintage_path(sq, args.horizon, realised, ann_level,
                                             terminal, force_splice=args.force_splice))
        except Exception as e:
            skipped.append((sq, f"{type(e).__name__}: {e}"))
    if not blocks:
        raise SystemExit("no vintage produced a path")
    df = pd.concat(blocks, ignore_index=True)

    COST_FWD.mkdir(parents=True, exist_ok=True)
    out = Path(args.out_vintages) if args.out_vintages else COST_FWD / "forward_rf_vintages.csv"
    df.to_csv(out, index=False)
    print(f"  Vintage r^f paths ({df['start_q'].nunique()} starts x {args.horizon}+1 rows) "
          f"-> {out.name}")

    # ---- per-start summary + the go/no-go ratio -------------------------------
    rows = []
    for sq in sorted(df["start_q"].unique(), key=_q_to_tq):
        g = df[df["start_q"] == sq].sort_values("h")
        fwd = g[g["h"] >= 1]["rf_qoq"].to_numpy(float)
        rows.append(dict(start_q=sq, vintage=g["vintage_date"].iloc[0],
                         rf0=float(g["rf_qoq"].iloc[0]),
                         ann0=float(g["selic_ann_pct"].iloc[0]),
                         bwm=_beta_weighted_mean(fwd),
                         splice=str(g["source"].iloc[0]).startswith("splice"),
                         source=str(g["source"].iloc[0])))
    sm = pd.DataFrame(rows)
    sm["bwm_ann"] = _quarterly_to_annual(sm["bwm"].to_numpy())

    print(f"\n  beta-weighted mean forward rate, beta={BETA_REPORT}, h=1..{args.horizon}")
    print("    start_q  vintage      realised h=0   bwm rf_qoq   bwm ann %  splice")
    for _, r in sm.iterrows():
        print(f"    {r['start_q']:<8} {r['vintage']}   {r['rf0']:.5f}      "
              f"{r['bwm']:.5f}     {r['bwm_ann']:>6.2f}    {'Y' if r['splice'] else '.'}")

    lo, hi = sm["bwm"].min(), sm["bwm"].max()
    ratio = hi / lo if lo > 0 else np.inf
    print(f"\n  === GO/NO-GO ===")
    print(f"  beta-weighted mean rf_qoq  min={lo:.5f} ({sm.loc[sm['bwm'].idxmin(),'start_q']})"
          f"  median={sm['bwm'].median():.5f}"
          f"  max={hi:.5f} ({sm.loc[sm['bwm'].idxmax(),'start_q']})")
    print(f"  MAX/MIN RATIO = {ratio:.2f}   (design threshold 3.0)  "
          f"-> {'GO' if ratio >= 3.0 else 'NO-GO'}")

    # ---- checks --------------------------------------------------------------
    anchor_fail, bad = _check_vintages(df)
    n_spl = int(sm["splice"].sum())
    print(f"\n  splices: {n_spl}/{len(sm)}")
    for _, r in sm[sm["splice"]].iterrows():
        print(f"    {r['start_q']}  {r['source']}")
    if skipped:
        print(f"  skipped starts: {len(skipped)}")
        for sq, why in skipped:
            print(f"    {sq}  {why}")
    if anchor_fail:
        print(f"  ANCHOR CHECK FAILURES (|h=0 - panel risk_free_qoq| > 0.002): {len(anchor_fail)}")
        for sq, mine, theirs, why in anchor_fail:
            print(f"    {sq}  ours={mine:.5f} panel={theirs:.5f}  {why}")
    else:
        print("  anchor check: all h=0 rows within 0.002 of the panel's risk_free_qoq")
    if bad:
        print(f"  RATE BOUND VIOLATIONS (non-finite / <=0 / >={MAX_PLAUSIBLE_ANN}% p.a.): {len(bad)}")
        for b in bad[:20]:
            print(f"    {b}")
    else:
        print(f"  bounds: all rates finite, positive and under {MAX_PLAUSIBLE_ANN}% p.a.")

    if args.plot:
        png = out.with_suffix(".png")
        _plot_vintages(df, png)
        print(f"  plot -> {png.name}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--start", type=str, default="2026Q1", help="first forward quarter YYYYQn")
    ap.add_argument("--offline", action="store_true", help="force the mean-reversion fallback")
    ap.add_argument("--out", type=str, default=None)
    # vintage mode
    ap.add_argument("--vintage-from", type=str, default=None, help="first start quarter, e.g. 2016Q1")
    ap.add_argument("--vintage-to", type=str, default="2024Q4", help="last start quarter")
    ap.add_argument("--vintage-every", type=int, default=1, help="stride in quarters")
    ap.add_argument("--vintages", type=str, default=None,
                    help="explicit comma list, e.g. '2016Q1,2018Q3,2020Q4' (overrides the range)")
    ap.add_argument("--out-vintages", type=str, default=None)
    ap.add_argument("--force-splice", action="store_true",
                    help="build every vintage as the realised/terminal perfect-foresight splice")
    ap.add_argument("--plot", action="store_true", help="overlay PNG of the vintage curves")
    args = ap.parse_args()

    if args.vintage_from or args.vintages:
        # Vintage mode is exclusive: it never rewrites forward_rf_qoq.csv, so a vintage run
        # can never disturb the single curve the existing CF engines already read.
        _run_vintages(args)
        return

    df = build_path(args.horizon, args.start, args.offline)
    COST_FWD.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else COST_FWD / "forward_rf_qoq.csv"
    df.to_csv(out, index=False)
    src = df["source"].iloc[0]
    print(f"  Forward r^f path ({len(df)} quarters) → {out.name}")
    print(f"  source: {src}")
    show = df.iloc[[0, 3, 7, 11, 19, 39, len(df) - 1]] if len(df) >= 40 else df
    for _, r in show.iterrows():
        print(f"    h={int(r['h']):>2} {r['cal_q']}  Selic={r['selic_ann_pct']:>6.2f}%  "
              f"r^f_q={r['rf_qoq']:.5f}")


if __name__ == "__main__":
    main()
