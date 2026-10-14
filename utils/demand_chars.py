"""utils/demand_chars.py -- the bank characteristics, segments and balances the demand preps use.

What it is
----------
The three demand preps (sleep_demand_prep_e1.py, sleep_demand_prep_e2.py,
sleep_demand_prep_link.py) read market_panel.csv and pass the frame through `apply` before they
build anything, and through `finalize` before they write a parquet. `panel_demand_chars.py` builds
the values `apply` writes, in ONE side file, `processed/demand_chars.parquet`; the panel itself is
never written, so every other consumer of it -- the sleepiness estimators, the policy function, the
descriptive tables -- reads exactly what it reads without this module.

The four corrections, each a function below:

  restore_2016q1_lags          the lags of 2016Q1: balances of 2013-2015 on the scale of the raw
                               ESTBAN files; the 2015Q4 balance of the national firms that only
                               IF.data reports (`append_2015q4_rows`); the lagged characteristics
                               of 2016Q1 from the 2015Q4 filings of member institutions; the
                               rival-branch count of 2015Q4; the leave-one-out pairs rebuilt from
                               those values
  fill_lagged_characteristics  the other firm-quarters with no lagged characteristics: the true
                               lag where the firm filed at t-1, else the nearest filing within four
                               quarters, else the rows are dropped (in `finalize`)
  backfill_segments            a quarter with no recorded prudential segment takes the firm's first
                               recorded one
  retime_borrowings_ratio      the borrowings ratio (IF.data Passivo line 78295 over total assets)
                               built at one date and lagged, under its own name, and its
                               leave-one-out pair

    DEMAND_CHARS_FILE     the side file (default processed/demand_chars.parquet; on the cluster it
                          lives in data/input). A missing file is an error: the preps never run
                          without these corrections. So is a side file built from another market
                          panel: its schema metadata records the md5 of the panel it was built
                          from, and `apply` compares it with the panel the prep reads.

Reads   processed/demand_chars.parquet (written by panel_demand_chars.py)
Writes  nothing, except `<parquet>.chars_audit.json` beside each demand parquet, from `finalize`,
        when the caller passes the parquet's path.

Column names
------------
The estimators read the borrowings instruments as `loo_borrow_assets` and
`mean_loo_borrow_assets` (blp_logit.jl, blp_engine_cpu.jl): the re-timed ratio `borrow_assets_lag`
and its leave-one-out pair. The panel's own pair of the earlier ratio (`loo_credit_assets`,
`mean_loo_credit_assets`) is dropped from the demand parquets by `finalize`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths

FILE_ENV = "DEMAND_CHARS_FILE"

KEYS = ["CodConglomeradoPrudencial", "mca_code", "year", "quarter"]

# The lagged characteristics a corrected firm-quarter receives (all from ONE source filing).
CHAR_COLS = ["log_total_assets_lag", "equity_ratio_lag", "personnel_cost_ratio_lag",
             "admin_cost_ratio_lag", "tax_cost_ratio_lag", "indice_basileia_lag",
             "npl_provision_ratio_lag", "total_assets_lag"]
SEG_COLS = ["seg_S2", "seg_S3", "seg_S4", "seg_S5"]
# The leave-one-out pairs the panel carries (panel_loo_instruments.compute_loo_instruments).
LOO_COLS = ["loo_log_assets", "mean_loo_log_assets", "loo_equity_ratio", "mean_loo_equity_ratio",
            "loo_basileia", "mean_loo_basileia", "loo_credit_assets", "mean_loo_credit_assets",
            "loo_npl_provision", "mean_loo_npl_provision"]
BORROW_COLS = ["borrow_assets_lag", "loo_borrow_assets", "mean_loo_borrow_assets"]
# The panel's leave-one-out pair of the earlier borrowings ratio, dropped by `finalize`.
PANEL_BORROW_PAIR = ["loo_credit_assets", "mean_loo_credit_assets"]
DEP_COLS = ["dep_a1", "dep_a2", "dep_a4"]
ADDED_COLS = DEP_COLS + ["spread_a1", "spread_a2", "spread_a4", "spread_ann_a1", "spread_ann_a2",
                         "spread_ann_a4", "risk_free_qoq"]

# Flags that travel into the demand parquets. Values are documented in panel_demand_chars.py.
FLAG_COLS = ["chars_src", "chars_dist", "basel_imputed", "segment_src", "lag_from_2015q4",
             "lag_dep_src", "estban_lag_src", "borrow_changed", "borrow_den_guarded"]
# Columns a prep must carry into its parquets on top of its own list.
KEEP_COLS = FLAG_COLS + BORROW_COLS

# What the demand estimators read (blp_logit.jl and blp_engine_cpu.jl: X_COLS and the IV_* lists;
# the D-type dummy they add is built from is_B, not read).
X_ENGINE = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
            "log_total_assets_lag", "is_state_owned"]
LOO_ENGINE = [c for c in LOO_COLS if c not in PANEL_BORROW_PAIR] + ["loo_borrow_assets",
                                                                    "mean_loo_borrow_assets"]
IV_ENGINE = LOO_ENGINE + ["n_rivals", "estban_rival_branches_lag", "personnel_cost_ratio_lag",
                          "admin_cost_ratio_lag", "tax_cost_ratio_lag", "indice_basileia_lag"]

CHARS_DROPPED = 9      # chars_src of a firm-quarter with no filing within the maximum distance


def chars_path() -> Path:
    """The side file: DEMAND_CHARS_FILE, else processed/demand_chars.parquet."""
    override = os.environ.get(FILE_ENV, "").strip()
    return Path(override).expanduser() if override else paths.PROCESSED / "demand_chars.parquet"


# Schema-metadata key under which panel_demand_chars.py stores the md5 of the market panel the
# side file was built from.
PANEL_MD5_KEY = b"demand_chars.market_panel_md5"


def panel_path() -> Path:
    """The market-panel file the demand preps read: the Parquet twin of `paths.market_panel_csv()`,
    which `load_panel_cached` serves (refreshing it from the CSV first when the CSV is newer). On the
    cluster it is the only copy, processed/market_panel.parquet linked to data/input."""
    from utils import sidecar_path
    return Path(sidecar_path(paths.market_panel_csv()))


_MD5_CACHE = {}


def _md5(p: Path) -> str:
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _MD5_CACHE:
        import hashlib
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        _MD5_CACHE[key] = h.hexdigest()
    return _MD5_CACHE[key]


def check_panel_md5() -> str:
    """The side file must have been built from the market panel the prep reads. Compares the md5
    recorded in its schema metadata (panel_demand_chars.py) with the md5 of `panel_path()`; a
    missing record or a different md5 is an error. Returns the md5."""
    import pyarrow.parquet as pq
    meta = pq.read_schema(chars_path()).metadata or {}
    recorded = meta.get(PANEL_MD5_KEY, b"").decode()
    pp = panel_path()
    if not pp.exists():
        raise FileNotFoundError(f"demand chars: the market panel {pp} does not exist.")
    actual = _md5(pp)
    if recorded != actual:
        raise RuntimeError(
            f"demand chars: {chars_path()} was built from a market panel with md5 "
            f"{recorded or '(none recorded)'}, but the prep reads {pp} with md5 {actual}. The "
            f"corrections key on that panel's rows and values: rebuild the side file from the current "
            f"panel (python panel_demand_chars.py, panel_pipeline.py step 39), or point "
            f"{FILE_ENV} at the side file built from this panel.")
    return actual


_CACHE = {}


def _read() -> pd.DataFrame:
    p = chars_path()
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. The demand preps need it: build it with "
            f"`python panel_demand_chars.py`, or point {FILE_ENV} at it.")
    key = (str(p), p.stat().st_mtime_ns)
    if key not in _CACHE:
        _CACHE.clear()
        r = pd.read_parquet(p)
        r["CodConglomeradoPrudencial"] = r["CodConglomeradoPrudencial"].astype(str)
        r["mca_code"] = r["mca_code"].astype(str)
        _CACHE[key] = r
    return _CACHE[key]


def _h(c: str) -> str:
    """Name of side-file column `c` while it sits merged on the panel frame."""
    return f"__dc__{c}"


def _overwrite(df: pd.DataFrame, mask, cols, prefix: str = "") -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = np.where(mask, df[_h(prefix + c)].to_numpy(), df[c].to_numpy())


def restore_2016q1_lags(df: pd.DataFrame, src: np.ndarray, in_window: np.ndarray) -> None:
    """Balances of 2013-2015 on the scale of the raw ESTBAN files; the lagged characteristics of
    2016Q1 from the 2015Q4 filings of member institutions (chars_src 2); the rival-branch count of
    2015Q4; the leave-one-out pairs rebuilt from those values. The 2015Q4 rows of the national
    firms only IF.data reports are added by `append_2015q4_rows`."""
    has_dep = df[_h("r__dep_a1")].notna().to_numpy() | df[_h("r__dep_a2")].notna().to_numpy() \
        | df[_h("r__dep_a4")].notna().to_numpy()
    pre = (df["year"].to_numpy() < 2016) & has_dep
    for c in DEP_COLS:
        df[c] = np.where(pre, df[_h("r__" + c)].to_numpy(), df[c].to_numpy())
    _overwrite(df, src == 2, CHAR_COLS, "r__")
    m = df[_h("estban_lag_src")].fillna(0).to_numpy() == 1
    df["estban_rival_branches_lag"] = np.where(m, df[_h("r__estban_rival_branches_lag")].to_numpy(),
                                               df["estban_rival_branches_lag"].to_numpy())
    _overwrite(df, in_window, LOO_COLS, "s1__")


def fill_lagged_characteristics(df: pd.DataFrame, src: np.ndarray, in_window: np.ndarray) -> None:
    """The other firm-quarters with no lagged characteristics (chars_src 1, 3) take the file's
    values; a firm-quarter flagged for dropping (chars_src 9) takes its missing values too, so a
    stored zero lagged total never stands in for a characteristic. Leave-one-out pairs rebuilt."""
    _overwrite(df, (src == 1) | (src == 3) | (src == CHARS_DROPPED), CHAR_COLS, "r__")
    _overwrite(df, in_window, LOO_COLS, "s2__")


def backfill_segments(df: pd.DataFrame, in_window: np.ndarray) -> None:
    """A quarter with no recorded prudential segment takes the firm's first recorded one."""
    _overwrite(df, in_window, SEG_COLS, "r__")
    df["segment"] = np.where(in_window, df[_h("r__segment")].to_numpy(), df["segment"].to_numpy())


def retime_borrowings_ratio(df: pd.DataFrame) -> None:
    """The borrowings ratio built at one date and lagged (`borrow_assets_lag`) and its
    leave-one-out pair, under the names the estimators read."""
    for c in BORROW_COLS:
        df[c] = df[_h(c)].to_numpy()


def append_2015q4_rows(df: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """One 2015Q4 row per national firm whose 2015Q4 balance the file supplies. The preps form lags
    before they cut the window, so that row gives the 2016Q1 row its lagged balance and is then
    dropped by the window like every other 2015 row. Every other column is copied from the firm's
    own 2016Q1 row, so no column changes type."""
    add = R[R["row_kind"] == "added_2015q4"]
    if not len(add):
        return df
    tmpl = df.merge(add[["CodConglomeradoPrudencial", "mca_code"]].assign(year=2016, quarter=1),
                    on=KEYS, how="inner")
    if len(tmpl) != len(add):
        raise AssertionError("demand chars: an appended 2015Q4 row has no 2016Q1 row of its firm")
    vals = add.set_index(["CodConglomeradoPrudencial", "mca_code"])
    idx = pd.MultiIndex.from_frame(tmpl[["CodConglomeradoPrudencial", "mca_code"]])
    tmpl["year"], tmpl["quarter"] = 2015, 4
    for c in ADDED_COLS:
        tmpl[c] = vals["r__" + c].reindex(idx).to_numpy()
    for c in ("dep_a5", "spread_a5", "spread_ann_a5"):
        if c in tmpl.columns:
            tmpl[c] = np.nan
    df = pd.concat([df, tmpl[df.columns]], ignore_index=True)
    if df.duplicated(KEYS).any():
        raise AssertionError("demand chars: an appended 2015Q4 row duplicates a panel row")
    return df


def apply(df_raw: pd.DataFrame) -> pd.DataFrame:
    """The panel frame with the four corrections written into it (a copy).

    `df_raw` is the wide market panel as a prep loads it (one row per conglomerate x market x
    quarter, all years). Corrected columns are overwritten on the rows the side file names, the
    flag columns attached, and one 2015Q4 row appended per national firm whose 2015Q4 balance the
    file supplies. The side file must have been built from the panel the prep read
    (`check_panel_md5`).
    """
    R = _read()
    md5 = check_panel_md5()
    print(f"  [demand chars] side file built from this market panel (md5 {md5})")
    n0 = len(df_raw)
    df = df_raw.copy()
    df["CodConglomeradoPrudencial"] = df["CodConglomeradoPrudencial"].astype(str)
    df["mca_code"] = df["mca_code"].astype(str)
    if df.duplicated(KEYS).any():
        raise ValueError("demand chars: the panel frame is not unique on " + ", ".join(KEYS))

    on_panel = R[R["row_kind"] != "added_2015q4"]
    helper = [c for c in on_panel.columns if c not in KEYS and c != "row_kind"]
    df = df.merge(on_panel[KEYS + helper].rename(columns={c: _h(c) for c in helper}), on=KEYS,
                  how="left", validate="one_to_one")
    if len(df) != n0:
        raise AssertionError("demand chars: the merge of the side file changed the row count")
    in_window = df[_h("chars_src")].notna().to_numpy()
    src = df[_h("chars_src")].fillna(-1).to_numpy()

    restore_2016q1_lags(df, src, in_window)
    fill_lagged_characteristics(df, src, in_window)
    backfill_segments(df, in_window)
    retime_borrowings_ratio(df)
    for c in FLAG_COLS:
        df[c] = df[_h(c)].fillna(-1).astype("int8").to_numpy()
    df = df.drop(columns=[_h(c) for c in helper])
    return append_2015q4_rows(df, R)


def audit(df: pd.DataFrame) -> dict:
    """Counts of what the estimators would find missing in `df` (a demand frame, long format):
    null X, null or infinite instruments, and the flags."""
    out = {"rows": int(len(df))}
    xs = {}
    for c in X_ENGINE:
        xs[c] = int(pd.to_numeric(df[c], errors="coerce").isna().sum()) if c in df.columns else -1
    ivs = {}
    for c in IV_ENGINE:
        if c not in df.columns:
            ivs[c] = {"null": -1, "inf": -1}
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        ivs[c] = {"null": int(np.isnan(v).sum()), "inf": int(np.isinf(v).sum())}
    out["x_null"] = xs
    out["iv_null_or_inf"] = ivs
    out["x_null_rows"] = int(df[[c for c in X_ENGINE if c in df.columns]].isna().any(axis=1).sum())
    ivc = [c for c in IV_ENGINE if c in df.columns]
    arr = df[ivc].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    out["iv_bad_rows"] = int((~np.isfinite(arr)).any(axis=1).sum())
    for c in FLAG_COLS:
        if c in df.columns:
            out[c] = {str(k): int(v) for k, v in df[c].value_counts().sort_index().items()}
    return out


def finalize(df_spec: pd.DataFrame, out_path=None, summary=None) -> pd.DataFrame:
    """Last step of a prep, on the frame it is about to write.

    The rows of a firm-quarter with no filing within the maximum distance (chars_src == 9) leave
    here, after the shares were formed, so no other row's share moves. The panel's pair of the
    earlier borrowings ratio is dropped. A null X or a null or infinite instrument left after that
    is an error. `summary` (the prep's own per-spec dict) gets the row counts after the drop and
    the counts; `out_path` gets a `.chars_audit.json` twin.
    """
    n_before = len(df_spec)
    dropped = {"rows": 0, "B": 0, "D": 0}
    if "chars_src" in df_spec.columns:
        gone = df_spec["chars_src"] == CHARS_DROPPED
        is_b = df_spec["is_B"].astype(bool)
        dropped = {"rows": int(gone.sum()), "B": int((gone & is_b).sum()), "D": int((gone & ~is_b).sum())}
        df_spec = df_spec[~gone].copy()
    df_spec = df_spec.drop(columns=[c for c in PANEL_BORROW_PAIR if c in df_spec.columns])
    rep = audit(df_spec)
    rep["rows_before_drop"] = int(n_before)
    rep["dropped_no_characteristics"] = dropped
    rep["chars_file"] = str(chars_path())
    print(f"  [demand chars] {n_before:,} rows, {dropped['rows']} dropped for want of characteristics "
          f"(B {dropped['B']}, D {dropped['D']}); rows with a null X {rep['x_null_rows']:,}; rows with a "
          f"null or infinite instrument {rep['iv_bad_rows']:,}")
    if rep["x_null_rows"] or rep["iv_bad_rows"]:
        bad_x = {k: v for k, v in rep["x_null"].items() if v}
        bad_iv = {k: v for k, v in rep["iv_null_or_inf"].items() if v["null"] or v["inf"]}
        raise AssertionError(
            "demand chars: the frame still holds values the estimators would find missing. "
            f"X: {bad_x}. Instruments: {bad_iv}.")
    if summary is not None:
        is_b = df_spec["is_B"].astype(bool)
        summary["Rows"] = int(len(df_spec))
        summary["B_firms"] = int(is_b.sum())
        summary["D_firms"] = int((~is_b).sum())
        summary["demand_chars"] = {"dropped_no_characteristics": dropped,
                                   "x_null_rows": rep["x_null_rows"], "iv_bad_rows": rep["iv_bad_rows"]}
    if out_path is not None:
        p = Path(str(out_path) + ".chars_audit.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2)
    return df_spec
