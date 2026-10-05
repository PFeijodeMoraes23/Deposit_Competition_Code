"""utils/demand_repair.py -- the demand-side repair layer: one switch, one file, one function.

What it is
----------
The three demand preps (sleep_demand_prep_e1.py, sleep_demand_prep_e2.py,
sleep_demand_prep_link.py) read market_panel.csv. Four defects of that panel reach the demand
estimators as zeros (missing lagged bank characteristics, unrecorded prudential segments, an
unguarded ratio) or empty the first quarter of the window (balances of 2013-2015 on twice their
scale). `panel_demand_repair.py` writes the repaired values to ONE side file,
`processed/demand_repair.parquet`; this module applies that file to the frame a prep has just
loaded. The panel itself is never written, so every other consumer of it -- the sleepiness
estimators, the policy function, the descriptive tables -- reads exactly what it read before.

The switch
----------
`DEMAND_REPAIR=1` turns the layer on. Unset, or any other value, every function here returns
its argument untouched and the preps produce the frames they produce without this module.

    DEMAND_REPAIR         "1" = on
    DEMAND_REPAIR_STAGES  which repairs to apply, a subset of "1234" (default "1234"):
                            1  the lags of 2016Q1: balances of 2013-2015 on the scale of the raw
                               ESTBAN files; the 2015Q4 balance of the national firms that only
                               IF.data reports; the lagged characteristics of 2016Q1 from the
                               2015Q4 filings of member institutions; the rival-branch count of
                               2015Q4; the leave-one-out pairs rebuilt from those values
                            2  the other firm-quarters with no lagged characteristics: the true
                               lag where the firm filed at t-1, else the nearest filing within
                               four quarters, else the rows are dropped (in `finalize`)
                            3  prudential segments: a quarter with no recorded segment takes the
                               firm's first recorded one
                            4  the borrowings ratio (IF.data Passivo line 78295 over total
                               assets) built at one date and lagged, under its own name, and its
                               leave-one-out pair
    DEMAND_REPAIR_FILE    the repair file (default processed/demand_repair.parquet)
    DEMAND_REPAIR_OUT     where a repaired prep writes its parquets (default
                          ESTIMATION_OUTPUT/DEMAND_PREP_REPAIRED). It can never be the folder
                          the unrepaired parquets live in: `output_dir` refuses.

Reads   processed/demand_repair.parquet (written by panel_demand_repair.py)
Writes  nothing, except `<parquet>.repair_audit.json` beside each repaired parquet, from
        `finalize`, when the caller passes the parquet's path.

Column names
------------
The repaired ratio and its instruments are `borrow_assets_lag`, `loo_borrow_assets` and
`mean_loo_borrow_assets`. The estimators' instrument lists name `loo_credit_assets` and
`mean_loo_credit_assets`, so under stage 4 those two columns carry the same values as the two
`*_borrow_*` columns: one variable under two names in the repaired parquets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths

SWITCH_ENV = "DEMAND_REPAIR"
STAGES_ENV = "DEMAND_REPAIR_STAGES"
FILE_ENV = "DEMAND_REPAIR_FILE"
OUT_ENV = "DEMAND_REPAIR_OUT"
OUT_DIRNAME = "DEMAND_PREP_REPAIRED"
ALL_STAGES = (1, 2, 3, 4)

KEYS = ["CodConglomeradoPrudencial", "mca_code", "year", "quarter"]

# The lagged characteristics a repaired firm-quarter receives (all from ONE source filing).
CHAR_COLS = ["log_total_assets_lag", "equity_ratio_lag", "personnel_cost_ratio_lag",
             "admin_cost_ratio_lag", "tax_cost_ratio_lag", "indice_basileia_lag",
             "npl_provision_ratio_lag", "total_assets_lag"]
SEG_COLS = ["seg_S2", "seg_S3", "seg_S4", "seg_S5"]
# The leave-one-out pairs the panel carries (panel_loo_instruments.compute_loo_instruments).
LOO_COLS = ["loo_log_assets", "mean_loo_log_assets", "loo_equity_ratio", "mean_loo_equity_ratio",
            "loo_basileia", "mean_loo_basileia", "loo_credit_assets", "mean_loo_credit_assets",
            "loo_npl_provision", "mean_loo_npl_provision"]
BORROW_COLS = ["borrow_assets_lag", "loo_borrow_assets", "mean_loo_borrow_assets"]
BORROW_ALIAS = {"loo_credit_assets": "loo_borrow_assets",
                "mean_loo_credit_assets": "mean_loo_borrow_assets"}
DEP_COLS = ["dep_a1", "dep_a2", "dep_a4"]
ADDED_COLS = DEP_COLS + ["spread_a1", "spread_a2", "spread_a4", "spread_ann_a1", "spread_ann_a2",
                         "spread_ann_a4", "risk_free_qoq"]

# Flags that travel into the repaired parquets. Values are documented in panel_demand_repair.py.
FLAG_COLS = ["chars_src", "chars_dist", "basel_imputed", "segment_src", "lag_from_2015q4",
             "lag_dep_src", "estban_lag_src", "borrow_changed", "borrow_den_guarded"]

# What the demand estimators read (blp_logit.jl L.91-112; blp_engine_cpu.jl L.123-150).
X_ENGINE = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
            "log_total_assets_lag", "is_state_owned"]
IV_ENGINE = LOO_COLS + ["n_rivals", "estban_rival_branches_lag", "personnel_cost_ratio_lag",
                        "admin_cost_ratio_lag", "tax_cost_ratio_lag", "indice_basileia_lag"]

CHARS_DROPPED = 9      # chars_src of a firm-quarter with no filing within the maximum distance


def enabled() -> bool:
    """True when DEMAND_REPAIR=1."""
    return os.environ.get(SWITCH_ENV, "").strip() == "1"


def stages(explicit=None) -> tuple:
    """The stages to apply: `explicit` if given, else DEMAND_REPAIR_STAGES when the layer is on,
    else none."""
    if explicit is not None:
        st = tuple(sorted({int(s) for s in explicit}))
    elif enabled():
        raw = os.environ.get(STAGES_ENV, "").strip() or "1234"
        st = tuple(sorted({int(ch) for ch in raw if ch.isdigit()}))
    else:
        return ()
    bad = [s for s in st if s not in ALL_STAGES]
    if bad:
        raise ValueError(f"demand repair: unknown stage(s) {bad}; the stages are {ALL_STAGES}")
    return st


def repair_path() -> Path:
    """The repair file: DEMAND_REPAIR_FILE, else processed/demand_repair.parquet."""
    override = os.environ.get(FILE_ENV, "").strip()
    return Path(override).expanduser() if override else paths.PROCESSED / "demand_repair.parquet"


def output_dir(default) -> Path:
    """Where a prep writes its parquets: `default` when the layer is off; when it is on,
    DEMAND_REPAIR_OUT or ESTIMATION_OUTPUT/DEMAND_PREP_REPAIRED, and never `default` itself or
    the folder of the unrepaired parquets."""
    default = Path(default)
    if not enabled():
        return default
    override = os.environ.get(OUT_ENV, "").strip()
    out = Path(override).expanduser() if override else paths.estimation_output() / OUT_DIRNAME
    stored = {default.resolve(), paths.demand_prep_root().resolve()}
    if out.resolve() in stored:
        raise RuntimeError(
            f"demand repair: the repaired parquets would be written to {out}, the folder of the "
            f"unrepaired ones. Set {OUT_ENV} to a different folder.")
    return out


def keep_columns() -> list:
    """Columns a prep must carry into its parquets when the layer is on (none when it is off)."""
    return (FLAG_COLS + BORROW_COLS) if enabled() else []


_CACHE = {}


def _read() -> pd.DataFrame:
    p = repair_path()
    if not p.exists():
        raise FileNotFoundError(
            f"demand repair is on ({SWITCH_ENV}=1) but {p} does not exist. Build it with "
            f"`python panel_demand_repair.py`, or unset {SWITCH_ENV}.")
    key = (str(p), p.stat().st_mtime_ns)
    if key not in _CACHE:
        _CACHE.clear()
        r = pd.read_parquet(p)
        r["CodConglomeradoPrudencial"] = r["CodConglomeradoPrudencial"].astype(str)
        r["mca_code"] = r["mca_code"].astype(str)
        _CACHE[key] = r
    return _CACHE[key]


def apply(df_raw: pd.DataFrame, explicit_stages=None) -> pd.DataFrame:
    """Return the panel frame with the repairs of the active stages written into it.

    `df_raw` is the wide market panel as a prep loads it (one row per conglomerate x market x
    quarter, all years). With the layer off and no explicit stages the SAME object comes back.
    Otherwise a copy: repaired columns overwritten on the rows the file names, the flag columns
    attached, and, under stage 1, one 2015Q4 row appended per national firm whose 2015Q4 balance
    the file supplies (the preps form lags before they cut the window, so that row gives the
    2016Q1 row its lagged balance and is then dropped by the window like every other 2015 row).
    """
    st = stages(explicit_stages)
    if not st:
        return df_raw
    R = _read()
    n0 = len(df_raw)
    df = df_raw.copy()
    df["CodConglomeradoPrudencial"] = df["CodConglomeradoPrudencial"].astype(str)
    df["mca_code"] = df["mca_code"].astype(str)
    if df.duplicated(KEYS).any():
        raise ValueError("demand repair: the panel frame is not unique on " + ", ".join(KEYS))

    on_panel = R[R["row_kind"] != "added_2015q4"]
    helper = [c for c in on_panel.columns if c not in KEYS and c != "row_kind"]
    helper_names = {c: f"__rp__{c}" for c in helper}
    df = df.merge(on_panel[KEYS + helper].rename(columns=helper_names), on=KEYS, how="left",
                  validate="one_to_one")
    if len(df) != n0:
        raise AssertionError("demand repair: the merge of the repair file changed the row count")
    H = lambda c: f"__rp__{c}"                                   # noqa: E731
    in_window = df[H("chars_src")].notna().to_numpy()
    src = df[H("chars_src")].fillna(-1).to_numpy()

    def _overwrite(mask, cols, prefix=""):
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = np.where(mask, df[H(prefix + c)].to_numpy(), df[c].to_numpy())

    if 1 in st:
        # Balances of 2013-2015 on the scale of the raw files.
        has_dep = df[H("r__dep_a1")].notna().to_numpy() | df[H("r__dep_a2")].notna().to_numpy() \
            | df[H("r__dep_a4")].notna().to_numpy()
        pre = (df["year"].to_numpy() < 2016) & has_dep
        for c in DEP_COLS:
            df[c] = np.where(pre, df[H("r__" + c)].to_numpy(), df[c].to_numpy())
        # Lagged characteristics of 2016Q1 from the 2015Q4 filings of member institutions.
        _overwrite(src == 2, CHAR_COLS, "r__")
        # Rival branches of 2015Q4.
        m = df[H("estban_lag_src")].fillna(0).to_numpy() == 1
        df["estban_rival_branches_lag"] = np.where(m, df[H("r__estban_rival_branches_lag")].to_numpy(),
                                                   df["estban_rival_branches_lag"].to_numpy())
        _overwrite(in_window, LOO_COLS, "s1__")
    if 2 in st:
        # A firm-quarter flagged for dropping takes the file's (missing) values too: where its
        # lagged total is a stored zero, the zero must not stand in for a characteristic.
        _overwrite((src == 1) | (src == 3) | (src == CHARS_DROPPED), CHAR_COLS, "r__")
        _overwrite(in_window, LOO_COLS, "s2__")
    if 3 in st:
        _overwrite(in_window, SEG_COLS, "r__")
        df["segment"] = np.where(in_window, df[H("r__segment")].to_numpy(), df["segment"].to_numpy())
    for c in BORROW_COLS:
        df[c] = df[H(c)].to_numpy()
    if 4 in st:
        for old, new in BORROW_ALIAS.items():
            df[old] = np.where(in_window, df[new].to_numpy(), df[old].to_numpy())
    for c in FLAG_COLS:
        df[c] = df[H(c)].fillna(-1).astype("int8").to_numpy()
    df = df.drop(columns=list(helper_names.values()))

    if 1 in st:
        add = R[R["row_kind"] == "added_2015q4"]
        if len(add):
            # Every other column of an appended row is copied from the firm's own 2016Q1 row, so no
            # column changes type; the row serves the lags only and leaves with the window.
            tmpl = df.merge(add[["CodConglomeradoPrudencial", "mca_code"]].assign(year=2016, quarter=1),
                            on=KEYS, how="inner")
            if len(tmpl) != len(add):
                raise AssertionError("demand repair: an appended 2015Q4 row has no 2016Q1 row of its firm")
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
                raise AssertionError("demand repair: an appended 2015Q4 row duplicates a panel row")
    return df


def audit(df: pd.DataFrame) -> dict:
    """Counts of what the estimators would have to fill in `df` (a demand frame, long format):
    null X, null or infinite instruments, and the flags."""
    out = {"rows": int(len(df))}
    xs = {}
    for c in X_ENGINE:
        if c in df.columns:
            xs[c] = int(pd.to_numeric(df[c], errors="coerce").isna().sum())
        else:
            xs[c] = -1
    ivs = {}
    for c in IV_ENGINE + [b for b in BORROW_COLS if b != "borrow_assets_lag"]:
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


def finalize(df_spec: pd.DataFrame, out_path=None, summary=None, explicit_stages=None) -> pd.DataFrame:
    """Last step of a repaired prep, on the frame it is about to write.

    Under stage 2 the rows of a firm-quarter with no filing within the maximum distance
    (chars_src == 9) leave here, after the shares were formed, so no other row's share moves.
    With all four stages on, a null X or a null or infinite instrument is an error: the
    estimators would turn it into a zero. `summary` (the prep's own per-spec dict) gets the row
    counts after the drop and the repair counts; `out_path` gets a `.repair_audit.json` twin.
    With the layer off the frame comes back untouched.
    """
    st = stages(explicit_stages)
    if not st:
        return df_spec
    n_before = len(df_spec)
    dropped = {"rows": 0, "B": 0, "D": 0}
    if 2 in st and "chars_src" in df_spec.columns:
        gone = df_spec["chars_src"] == CHARS_DROPPED
        is_b = df_spec["is_B"].astype(bool)
        dropped = {"rows": int(gone.sum()), "B": int((gone & is_b).sum()), "D": int((gone & ~is_b).sum())}
        df_spec = df_spec[~gone].copy()
    rep = audit(df_spec)
    rep["stages"] = list(st)
    rep["rows_before_drop"] = int(n_before)
    rep["dropped_no_characteristics"] = dropped
    rep["repair_file"] = str(repair_path())
    print(f"  [demand repair] stages {''.join(map(str, st))}: {n_before:,} rows, "
          f"{dropped['rows']} dropped for want of characteristics (B {dropped['B']}, D {dropped['D']}); "
          f"rows with a null X {rep['x_null_rows']:,}; rows with a null or infinite instrument "
          f"{rep['iv_bad_rows']:,}")
    if tuple(st) == ALL_STAGES and (rep["x_null_rows"] or rep["iv_bad_rows"]):
        bad_x = {k: v for k, v in rep["x_null"].items() if v}
        bad_iv = {k: v for k, v in rep["iv_null_or_inf"].items() if v["null"] or v["inf"]}
        raise AssertionError(
            "demand repair: after all four stages the frame still holds values the estimators "
            f"would fill with zero. X: {bad_x}. Instruments: {bad_iv}.")
    if summary is not None:
        is_b = df_spec["is_B"].astype(bool)
        summary["Rows"] = int(len(df_spec))
        summary["B_firms"] = int(is_b.sum())
        summary["D_firms"] = int((~is_b).sum())
        summary["demand_repair"] = {"stages": list(st), "dropped_no_characteristics": dropped,
                                    "x_null_rows": rep["x_null_rows"], "iv_bad_rows": rep["iv_bad_rows"]}
    if out_path is not None:
        p = Path(str(out_path) + ".repair_audit.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2)
    return df_spec
