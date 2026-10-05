"""
make_desc_compressed_tables.py
=========
Compressed, literature-style descriptive tables built from
``market_panel.csv``.  Complements the exhaustive panel-style tables produced
by ``make_desc_panel_tables.py``.

Generates four compact tables, each saved to ``OUTPUT_DIR`` (CSV + TeX) and
written to the V_Main draft directory, following the same naming convention
as ``make_desc_panel_tables.py`` (``_weighted_by_<col>`` or ``_unweighted`` suffix):

  1.  ``Compressed_BankType_CrossSection``
        Firm-quarter cross-section, Panel A = type B, Panel B = type D.
        Columns: Mean, SD, p10, p50, p90, N.  Following the bank-level
        cross-sections of Egan, Hortaçsu & Matvos (2017 AER) and Drechsler et
        al. (2021 JF).

  2.  ``Compressed_MarketStructure_by_Year``
        MCA-quarter market-structure moments averaged within each year:
        number of B firms per MCA, B-deposit HHI, deposit-weighted spreads,
        national count of D firms.  Following Dick (2008 JBF), Ho & Ishii
        (2011 IJIO), Drechsler et al. (2017 QJE).

  3.  ``Compressed_LocalEnvironment_by_Region``
        Population-weighted MCA averages by the five Brazilian macro-regions:
        population, GDP/cap, age structure, social-register coverage,
        branch density, mobile lines, PIX adoption.  Motivated by Joaquim & van
        Doornik (2019), Fonseca & Matray (2024 JFE), Van Doornik et al.
        (2024 AER), Koont (2025).

  4.  ``Compressed_DFirm_PrePost_Pix``
        Pre- vs post-PIX comparison at the D-firm × quarter level, using
        variables defined on both sides of the November-2020 break.
        Difference-in-means with CRV1 standard errors clustered at the
        conglomerate level and Imbens & Kolesár (2016) / Carter, Schnepel &
        Steigerwald (2017) effective-cluster Satterthwaite dof.  Motivated by
        Sarkisyan (2024) and Koont (2025).

Usage
-----
    python make_desc_compressed_tables.py                          # unweighted variant
    python make_desc_compressed_tables.py --weight-col pop_total   # population-weighted variant
"""
from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path

try:
    from utils.venv_guard import ensure_project_venv
except ImportError:  # pragma: no cover - utility absent in some checkouts
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from utils.window import apply_window, MIN_YEAR, MAX_YEAR
from utils import paths as _paths

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (mirror make_desc_panel_tables.py)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR    = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
PANEL_CSV        = DATA_DIR / "market_panel.csv"
MACRO_RATES_CSV  = DATA_DIR / "PANEL_INTERMED" / "quarterly_macro_rates.csv"
OUTPUT_DIR  = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"
DRAFTS_DIR  = _paths.drafts_dir()

REGION_MAPPING = {
    "1": "North",
    "2": "Northeast",
    "3": "Southeast",
    "4": "South",
    "5": "Center-West",
}

# Pix introduction: Nov 16, 2020 -> post starts at 2020Q4
PIX_POST_START = (2020, 4)

# ---------------------------------------------------------------------------
# Spread units
# ---------------------------------------------------------------------------
# market_panel.csv stores spread_a{k} and risk_free_qoq as QUARTERLY FRACTIONS
# (spread_a4 median ~0.004, risk_free_qoq median ~0.025). Every spread these tables and
# the figure report is ANNUALIZED, COMPOUNDED and in PERCENTAGE POINTS:
#     spread_ann = 100 * [(1 + r^f_q)^4 - (1 + r^dep_q)^4],   r^dep_q = r^f_q - spread_q.
# This is 100 x the panel's own spread_ann_a{k} column, and the price the demand stage
# estimates on (rho = spread_ann[bp] / 100 in blp_engine_*.jl), so a spread printed here
# reads in the same unit as the price coefficient. The Selic rate is reported on the same
# compounding: 100 * [(1 + r^f_q)^4 - 1], in percent per year.
SPREAD_UNIT = "pp p.a."

# ---------------------------------------------------------------------------
# Table-note building blocks. Every descriptive table (Tables 1-2, A.3, A.4 and the
# Summary_* family in make_desc_panel_tables.py) states a shared convention with the
# same words, so the notes read identically across the paper. Notes are set in
# \footnotesize and kept short: sample, units not in the labels, inference, caveats.
# ---------------------------------------------------------------------------
NOTE_FONT   = r"\footnotesize"
NOTE_SPREAD = r"Spreads: Selic minus deposit rate, both compounded to annual rates, in pp."
NOTE_MONEY  = r"Values in nominal R\$ millions."
NOTE_STARS  = r"$^{*}\,p<0.10$, $^{**}\,p<0.05$, $^{***}\,p<0.01$."


def table_note(*sentences: str) -> str:
    """'Notes:' paragraph body (no font switch) from sentences; empty items are skipped."""
    return r"\textit{Notes:} " + " ".join(s.strip() for s in sentences if s and s.strip())


def annualized_spread_pp(spread_qoq, rf_qoq):
    """Annualized, compounded spread in percentage points from quarterly fractions."""
    return 100.0 * ((1.0 + rf_qoq) ** 4 - (1.0 + rf_qoq - spread_qoq) ** 4)


def annualized_rate_pct(r_qoq):
    """Quarterly rate (fraction) compounded to an annual rate in percent."""
    return 100.0 * ((1.0 + r_qoq) ** 4 - 1.0)


# ---------------------------------------------------------------------------
# Display formatting (consistent with make_desc_panel_tables.py)
# ---------------------------------------------------------------------------
# (label, unit string, scale divisor or None)
# Monetary values are all nominal R$ millions, so deposits and assets read on one scale.
LABEL_MAP = {
    "total_deposits":            ("Total Deposits", r"R\$M",        1e6),
    "total_assets":              ("Total Assets",            r"R\$M",        1e6),
    "log_total_assets":          ("Log Total Assets",        "",             None),
    "log_dep":                   ("Log Deposits",  "",             None),
    "equity_ratio":              ("Equity Ratio",            "",             None),
    "dep_a1":                    ("Deposits (1)",            r"R\$M",        1e6),
    "dep_a2":                    ("Deposits (2)",            r"R\$M",        1e6),
    "dep_a4":                    ("Deposits (4)",            r"R\$M",        1e6),
    "dep_a5":                    ("Deposits (5)",            r"R\$M",        1e6),
    "log_dep_a4":                ("Log Deposits (4)",        "",             None),
    "n_mcas_served":             ("MCAs Served",             "count",        None),
    # Table 2 specific
    "n_b_firms":                 ("Number of B Firms",       "count",        None),
    "hhi_b":                     ("HHI (B firms)",           "",              None),
    "hhi_d_natl":                ("HHI (D firms, national)", "",              None),
    "hhi_combined_natl":         ("HHI (B+D, national)",     "",              None),
    "risk_free_qoq":              ("SELIC (risk-free)",         r"\% qoq",      0.01),
    # Annualized spreads (percentage points) and the annualized Selic rate (percent),
    # both already on the display scale when computed (see annualized_spread_pp).
    "spread_ann_a4":             ("Spread (4)",              SPREAD_UNIT,    None),
    "spread_ann_a5":             ("Spread (5)",              SPREAD_UNIT,    None),
    "spread_ann_a4_w":           ("Spread (4), B firms",     SPREAD_UNIT,    None),
    "spread_ann_a5_w":           ("Spread (5), B firms",     SPREAD_UNIT,    None),
    "spread_ann_a4_d_w":         ("Spread (4), D firms",     SPREAD_UNIT,    None),
    "spread_ann_a5_d_w":         ("Spread (5), D firms",     SPREAD_UNIT,    None),
    "spread_ann_a4_natl_w":      ("Spread (4), National (B+D)", SPREAD_UNIT, None),
    "spread_ann_a5_natl_w":      ("Spread (5), National (B+D)", SPREAD_UNIT, None),
    "risk_free_ann":             ("SELIC (risk-free)",        r"\% p.a.",     None),
    "n_d_firms_natl":            ("Number of D Firms (nat.)", "count",       None),
    # Table 3 specific
    "pop_total":                 ("Population",              "Thousands",    1e3),
    "gdp_per_capita":            (r"GDP \textit{per capita}", r"R\$",         None),
    # Shares are reported in PERCENTAGE POINTS, matching the unit their coefficients carry
    # in the sleepiness and BBL policy-function tables (utils/state_transform.DISPLAY, where
    # fraction_65plus is (0.01, 'pp')). Printing a bare 0.109 here while the coefficient
    # reads "per pp" is what makes SD x coefficient impossible to do in one's head.
    "fraction_65plus":           ("Share Aged 65+",          "pp",           0.01),
    "cadunico_families_per1000": (r"Cad\'Unico Families",    "per 1{,}000",  None),
    "branches_per1000":          ("Bank Branches",           "per 1{,}000",  None),
    # ANATEL active mobile-telephony accesses, all technologies (scrape_anatel_mobile.py).
    "connections_per100":        ("Mobile Lines",            "per 100 inhabitants", None),
    "pix_users_pf_per1000":      ("PIX Users (PF)",          "per 1{,}000",  None),
    "pix_txns_pf":               ("PIX Transactions (PF)",   "Millions",     1e6),
}

# LEVELS (a monetary amount, an HHI on the 0-10,000 concentration-index scale, or a
# population/transaction count expressed in Thousands/Millions) print as integers with
# thousands separators, per the author's decision. Every other variable -- rates, shares,
# spreads, ratios, densities (per 100 / per 1,000 inhabitants) and log/coefficient-like
# values -- prints to 3 decimals. n_d_firms_natl is handled next to LEVEL_VARS in _fmt_val
# (same integer formatting) but is kept out of this set: it is an exact firm headcount, not
# a rounded aggregate.
LEVEL_VARS = frozenset({
    "total_deposits", "total_assets", "dep_a1", "dep_a2", "dep_a4", "dep_a5",
    "hhi_b", "hhi_d_natl", "hhi_combined_natl",
    "pop_total", "gdp_per_capita", "pix_txns_pf",
})


def _esc(s) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _fmt_val(val, var_base):
    if pd.isna(val):
        return "--"
    info = LABEL_MAP.get(var_base)
    if info is None:
        return f"{val:,.3f}"
    _lbl, _unit, scale = info
    v = val / scale if scale is not None else val
    # Levels (LEVEL_VARS) and n_d_firms_natl (an exact firm headcount) print as integers;
    # every rate, share, spread, ratio, density and log/coefficient prints to 3 decimals.
    dec = 0 if (var_base in LEVEL_VARS or var_base == "n_d_firms_natl") else 3
    s = f"{v:,.{dec}f}"
    # A value that rounds to zero prints unsigned ("-0.00" carries no sign information).
    if s.startswith("-") and float(s[1:].replace(",", "")) == 0.0:
        s = s[1:]
    return s


def _row_label(var_base) -> str:
    info = LABEL_MAP.get(var_base)
    if info is None:
        return _esc(var_base)
    lbl, unit, _ = info
    return lbl + (f" ({unit})" if unit else "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _weighted_mean(x: np.ndarray, w: np.ndarray | None) -> float:
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if w is None:
        return float(np.mean(x[mask])) if mask.any() else np.nan
    w = np.asarray(w, dtype=float)
    mask &= np.isfinite(w)
    if not mask.any() or w[mask].sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))


def _weighted_std(x: np.ndarray, w: np.ndarray | None) -> float:
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if w is None:
        return float(np.std(x[mask], ddof=1)) if mask.sum() > 1 else np.nan
    w = np.asarray(w, dtype=float)
    mask &= np.isfinite(w)
    if mask.sum() < 2 or w[mask].sum() == 0:
        return np.nan
    m = np.average(x[mask], weights=w[mask])
    return float(np.sqrt(np.average((x[mask] - m) ** 2, weights=w[mask])))


def _weighted_quantile(x: np.ndarray, q: float, w: np.ndarray | None) -> float:
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if not mask.any():
        return np.nan
    if w is None:
        return float(np.quantile(x[mask], q))
    w = np.asarray(w, dtype=float)
    mask &= np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    xs = x[mask]
    ws = w[mask]
    order = np.argsort(xs)
    xs, ws = xs[order], ws[order]
    cum = np.cumsum(ws) / ws.sum()
    return float(xs[np.searchsorted(cum, q)])


def _apply_g_star(t_stat: float, cluster_sizes: pd.Series) -> tuple[int, float, float]:
    """Carter-Schnepel-Steigerwald (2017) / Imbens-Kolesár (2016) correction.

    Returns (G_nominal, G_star, p_value) under t(G_star).
    """
    sizes = np.asarray(cluster_sizes.values, dtype=float)
    G = int(len(sizes))
    if G == 0:
        return 0, 0.0, np.nan
    mean_s = sizes.mean()
    cv2 = (sizes.std(ddof=0) / mean_s) ** 2 if mean_s > 0 else 0.0
    G_star = max(1.0, G / (1.0 + cv2))
    p = 2.0 * stats.t.sf(abs(t_stat), df=G_star)
    return G, float(G_star), float(p)


def _cluster_diff_test(
    df: pd.DataFrame,
    y_col: str,
    cluster_col: str,
    post_col: str = "post",
) -> dict:
    """Regress y on a constant + post dummy with CRV1 SE clustered by ``cluster_col``;
    apply IK/CSS effective-cluster dof for the p-value.
    """
    sub = df[[y_col, post_col, cluster_col]].dropna()
    if sub.empty or sub[post_col].nunique() < 2:
        return {"pre_mean": np.nan, "post_mean": np.nan, "diff": np.nan,
                "se": np.nan, "t": np.nan, "p": np.nan,
                "G": 0, "G_star": 0.0, "n_pre": 0, "n_post": 0}
    X = sm.add_constant(sub[[post_col]].astype(float))
    y = sub[y_col].astype(float)
    res = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub[cluster_col].values},
        use_t=True,
    )
    beta = float(res.params[post_col])
    se   = float(res.bse[post_col])
    t    = beta / se if se > 0 else np.nan
    cluster_sizes = sub.groupby(cluster_col).size()
    G, G_star, p = _apply_g_star(t, cluster_sizes)
    pre_mask  = sub[post_col] == 0
    post_mask = sub[post_col] == 1
    return {
        "pre_mean":  float(sub.loc[pre_mask,  y_col].mean()),
        "post_mean": float(sub.loc[post_mask, y_col].mean()),
        "diff":      beta,
        "se":        se,
        "t":         t,
        "p":         p,
        "G":         G,
        "G_star":    G_star,
        "n_pre":     int(pre_mask.sum()),
        "n_post":    int(post_mask.sum()),
    }


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:  return r"$^{***}$"
    if p < 0.05:  return r"$^{**}$"
    if p < 0.10:  return r"$^{*}$"
    return ""


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def _resolve_is_B(df):
    """Firm type as a boolean Series: True = B (brick-and-mortar), False = D (digital).

    `is_B` is decided once, in panel_6_market.attach_mca_code, from the physical-network
    verdict in digital_banks_diagnostic.csv, and stored in market_panel.csv as 0/1. It is
    used verbatim whenever present -- never recomputed or "corrected" here, and in
    particular not reconciled with mca_code: firm type and market tier are separate facts.
    A panel written before the column existed falls back to the CODMUN_IBGE sentinel,
    which approximates the verdict by where a bank books its deposits.
    """
    if 'is_B' in df.columns:
        s = df['is_B']
        if pd.api.types.is_bool_dtype(s):
            return s.astype(bool)
        if pd.api.types.is_numeric_dtype(s):
            # 0/1 as stored; a missing value reads as D, matching the Julia side's
            # Bool.(coalesce.(df.is_B, false)).
            return pd.to_numeric(s, errors='coerce').fillna(0) != 0
        # Text, from a CSV writer that spelled the column out ("true"/"True"/"1").
        return s.astype(str).str.strip().str.lower().isin(('1', 'true', 't', 'yes'))
    print("[WARNING] Column 'is_B' not found in the market panel: this panel predates the "
          "stored firm-type column. Falling back to the CODMUN_IBGE sentinel; re-run "
          "panel_market.py to store the authoritative column.")
    return df['CODMUN_IBGE'].astype(str).str.split('.').str[0] != '0'


def load_panel() -> pd.DataFrame:
    print(f"Loading {PANEL_CSV.name} ...")
    return prepare_panel(pd.read_csv(PANEL_CSV, low_memory=False))


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    """The sample and derived columns every table here is built on, from raw market-panel rows
    (all columns, or the subset a caller needs)."""
    # The descriptives must describe the same sample the model is fit on. See utils/window.py.
    df = apply_window(df, label="desc_2 panel")

    df["CODMUN_IBGE_str"] = df["CODMUN_IBGE"].astype(str).str.split(".").str[0]
    df["bank_type"] = np.where(_resolve_is_B(df), "B", "D")
    df["region_code"] = df["CODMUN_IBGE_str"].str[0]
    df["region"]      = df["region_code"].map(REGION_MAPPING)
    df.loc[df["bank_type"] == "D", "region"] = "National"
    df["post"] = (
        (df["year"] > PIX_POST_START[0])
        | ((df["year"] == PIX_POST_START[0]) & (df["quarter"] >= PIX_POST_START[1]))
    ).astype(int)

    # Type-5 deposits did not exist pre-Pix: force dep_a5 = 0 and spread_a5 = NaN
    # for pre-period rows so they do not pollute pooled means or weighted spreads.
    pre_mask = df["post"] == 0
    if "dep_a5" in df.columns:
        df.loc[pre_mask, "dep_a5"] = 0.0
    if "spread_a5" in df.columns:
        df.loc[pre_mask, "spread_a5"] = np.nan

    # Forward-fill COSIF balance-sheet variables within each conglomerate to cover periods
    # where COSIF reporting has not yet been ingested (e.g. most-recent year after panel
    # extension). This carries the last available quarterly value forward in time.
    _cosif_cols = [c for c in ['total_assets', 'equity_ratio', 'dep_a5'] if c in df.columns]
    if _cosif_cols:
        df = df.sort_values(['CodConglomeradoPrudencial', 'year', 'quarter'])
        df[_cosif_cols] = (
            df.groupby('CodConglomeradoPrudencial')[_cosif_cols]
              .transform(lambda x: x.ffill())
        )
    return df


# ---------------------------------------------------------------------------
# Table 1: Bank-type cross-section
# ---------------------------------------------------------------------------
T1_VARS_B = ["total_assets", "equity_ratio",
             "dep_a1", "dep_a2", "dep_a4", "dep_a5", "total_deposits",
             "spread_ann_a4", "spread_ann_a5", "n_mcas_served"]
T1_VARS_D = ["total_assets", "equity_ratio",
             "dep_a1", "dep_a2", "dep_a4", "dep_a5", "total_deposits",
             "spread_ann_a4", "spread_ann_a5"]


def _build_firm_quarter_B(df_b: pd.DataFrame) -> pd.DataFrame:
    """Aggregate B-firm panel (MCA-level) to firm-quarter:
       deposits summed across MCAs; spreads averaged weighted by deposits within
       that asset class; firm-level covariates taken as first non-null."""
    grp_keys = ["CodConglomeradoPrudencial", "year", "quarter"]

    out = (
        df_b.groupby(grp_keys, sort=False)
            .agg(
                dep_a1=("dep_a1", "sum"),
                dep_a2=("dep_a2", "sum"),
                dep_a4=("dep_a4", "sum"),
                dep_a5=("dep_a5", "sum"),
                total_assets=("total_assets", "first"),
                equity_ratio=("equity_ratio", "first"),
                n_mcas_served=("mca_code", "nunique"),
                risk_free_qoq=("risk_free_qoq", "first"),
            )
            .reset_index()
    )
    # Zero deposits mean no active product in that firm-quarter — treat as missing so N matches spread
    for _a in ["dep_a1", "dep_a2", "dep_a4", "dep_a5"]:
        out[_a] = out[_a].where(out[_a] > 0)
    out["total_deposits"] = out[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].sum(axis=1, min_count=1)

    # Deposit-weighted spreads (vectorized via numerator/denominator)
    work = df_b[grp_keys + ["spread_a4", "spread_a5", "dep_a4", "dep_a5"]].copy()
    for v, w in (("spread_a4", "dep_a4"), ("spread_a5", "dep_a5")):
        ww = work[w].where(work[w] > 0, 0.0).where(work[v].notna(), 0.0)
        work[f"_num_{v}"] = work[v] * ww
        work[f"_den_{v}"] = ww
    sp = (work.groupby(grp_keys, sort=False)
              .agg({"_num_spread_a4": "sum", "_den_spread_a4": "sum",
                    "_num_spread_a5": "sum", "_den_spread_a5": "sum"})
              .reset_index())
    sp["spread_a4"] = np.where(sp["_den_spread_a4"] > 0,
                               sp["_num_spread_a4"] / sp["_den_spread_a4"], np.nan)
    sp["spread_a5"] = np.where(sp["_den_spread_a5"] > 0,
                               sp["_num_spread_a5"] / sp["_den_spread_a5"], np.nan)
    out = out.merge(sp[grp_keys + ["spread_a4", "spread_a5"]], on=grp_keys, how="left")
    # Annualize the deposit-weighted quarterly spreads (percentage points).
    for _a in ("a4", "a5"):
        out[f"spread_ann_{_a}"] = annualized_spread_pp(out[f"spread_{_a}"], out["risk_free_qoq"])
    return out


def _build_firm_quarter_D(df_d: pd.DataFrame) -> pd.DataFrame:
    """Aggregate D-firm panel (MCA-level) to firm-quarter:
       deposits summed across MCAs; spreads averaged weighted by deposits within
       that asset class; firm-level covariates taken as first non-null."""
    grp_keys = ["CodConglomeradoPrudencial", "year", "quarter"]

    out = (
        df_d.groupby(grp_keys, sort=False)
            .agg(
                dep_a1=("dep_a1", "sum"),
                dep_a2=("dep_a2", "sum"),
                dep_a4=("dep_a4", "sum"),
                dep_a5=("dep_a5", "sum"),
                total_assets=("total_assets", "first"),
                equity_ratio=("equity_ratio", "first"),
                risk_free_qoq=("risk_free_qoq", "first"),
            )
            .reset_index()
    )
    # Zero deposits mean no active product in that firm-quarter — treat as missing so N matches spread
    for _a in ["dep_a1", "dep_a2", "dep_a4", "dep_a5"]:
        out[_a] = out[_a].where(out[_a] > 0)
    out["total_deposits"] = out[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].sum(axis=1, min_count=1)

    # Deposit-weighted spreads
    work = df_d[grp_keys + ["spread_a4", "spread_a5", "dep_a4", "dep_a5"]].copy()
    for v, w in (("spread_a4", "dep_a4"), ("spread_a5", "dep_a5")):
        ww = work[w].where(work[w] > 0, 0.0).where(work[v].notna(), 0.0)
        work[f"_num_{v}"] = work[v] * ww
        work[f"_den_{v}"] = ww
    sp = (work.groupby(grp_keys, sort=False)
              .agg({"_num_spread_a4": "sum", "_den_spread_a4": "sum",
                    "_num_spread_a5": "sum", "_den_spread_a5": "sum"})
              .reset_index())
    sp["spread_a4"] = np.where(sp["_den_spread_a4"] > 0,
                               sp["_num_spread_a4"] / sp["_den_spread_a4"], np.nan)
    sp["spread_a5"] = np.where(sp["_den_spread_a5"] > 0,
                               sp["_num_spread_a5"] / sp["_den_spread_a5"], np.nan)
    out = out.merge(sp[grp_keys + ["spread_a4", "spread_a5"]], on=grp_keys, how="left")
    for _a in ("a4", "a5"):
        out[f"spread_ann_{_a}"] = annualized_spread_pp(out[f"spread_{_a}"], out["risk_free_qoq"])
    return out


def build_table1(df: pd.DataFrame, weight_col: str | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df_b = df[df["bank_type"] == "B"].copy()
    df_d = df[df["bank_type"] == "D"].copy()

    fq_b = _build_firm_quarter_B(df_b)
    fq_d = _build_firm_quarter_D(df_d)

    # Population weights at firm-quarter level: sum pop_total across MCAs.
    if weight_col is not None:
        pop_b = (df_b.groupby(["CodConglomeradoPrudencial", "year", "quarter"],
                              sort=False)[weight_col].sum().reset_index())
        fq_b  = fq_b.merge(pop_b, on=["CodConglomeradoPrudencial", "year", "quarter"], how="left")
        if weight_col in df_d.columns:
            pop_d = (df_d.groupby(["CodConglomeradoPrudencial", "year", "quarter"],
                                  sort=False)[weight_col].sum().reset_index())
            fq_d = fq_d.merge(pop_d, on=["CodConglomeradoPrudencial", "year", "quarter"], how="left")

    def _moments(panel: pd.DataFrame, var_list: list[str]) -> pd.DataFrame:
        rows = []
        w = panel[weight_col].values if (weight_col and weight_col in panel.columns) else None
        for v in var_list:
            if v not in panel.columns:
                continue
            x = panel[v].values
            rows.append({
                "var":  v,
                "Mean": _weighted_mean(x, w),
                "SD":   _weighted_std(x, w),
                "p10":  _weighted_quantile(x, 0.10, w),
                "p25":  _weighted_quantile(x, 0.25, w),
                "p50":  _weighted_quantile(x, 0.50, w),
                "p75":  _weighted_quantile(x, 0.75, w),
                "p90":  _weighted_quantile(x, 0.90, w),
                "N":    int(np.isfinite(x).sum()),
            })
        return pd.DataFrame(rows)

    moments_b = _moments(fq_b, T1_VARS_B)
    moments_d = _moments(fq_d, T1_VARS_D)

    # Pooled B+D moments — variables common to both types. The rendered table shows
    # the B and D panels only; these feed the *_All companion CSV.
    T1_VARS_COMMON = T1_VARS_D   # n_mcas_served is B-type only; all others shared
    cols_keep = (["CodConglomeradoPrudencial", "year", "quarter"]
                 + T1_VARS_COMMON
                 + ([weight_col] if weight_col and weight_col in fq_b.columns else []))
    fq_all = pd.concat([
        fq_b[[c for c in cols_keep if c in fq_b.columns]],
        fq_d[[c for c in cols_keep if c in fq_d.columns]],
    ], ignore_index=True)
    moments_all = _moments(fq_all, T1_VARS_COMMON)

    # Coverage facts quoted in the notes. A zero type-k balance is a reported zero (ESTBAN /
    # IF-Data), i.e. the product is not offered, not a missing record.
    _firm = "CodConglomeradoPrudencial"
    b_ever_savings = fq_b.groupby(_firm)["dep_a2"].apply(lambda s: bool((s > 0).any()))
    d_ever_deposit = fq_d.groupby(_firm)["total_deposits"].apply(lambda s: bool((s > 0).any()))
    meta = {
        "n_firm_quarters_B":   int(len(fq_b)),
        "n_firm_quarters_D":   int(len(fq_d)),
        "n_firms_B":           int(fq_b[_firm].nunique()),
        "n_firms_D":           int(fq_d[_firm].nunique()),
        "n_firm_quarters_all": int(len(fq_all)),
        "n_firms_all":         int(fq_all[_firm].nunique()),
        "n_firms_B_no_savings": int((~b_ever_savings).sum()),
        "n_firms_D_no_deposits": int((~d_ever_deposit).sum()),
        "n_fq_D_no_deposits":  int((~(fq_d["total_deposits"] > 0)).sum()),
    }
    return moments_b, moments_d, moments_all, meta


def render_table1(moments_b, moments_d, moments_all, meta, weight_col, suffix) -> str:
    name      = "Compressed_BankType_CrossSection"
    tab_label = f"tab:{name}"
    weight_lbl = "(Population Weighted)" if weight_col else ""
    caption   = (f"Bank-Conglomerate Cross-Section by Type{(' ' + weight_lbl) if weight_lbl else ''}")

    # Fill \textwidth (extracolsep distributes slack evenly) so the table matches
    # the full-width \par note below; longtable keeps the table page-breakable.
    col_spec  = "@{\\extracolsep{\\fill}}l rrrrrrrr"
    n_cols    = 9
    header_row = r"Variable & Mean & SD & p10 & p25 & p50 & p75 & p90 & $N$ \\"

    def _panel(title: str, moments: pd.DataFrame, n_firms: int, n_fq: int) -> list[str]:
        lines = [
            r"\midrule",
            f"\\multicolumn{{9}}{{l}}{{\\textit{{{title}}}}} \\\\",
            r"\midrule",
        ]
        for _, r in moments.iterrows():
            vb  = r["var"]
            lbl = _row_label(vb)
            lines.append(
                f"{lbl} & "
                + " & ".join([_fmt_val(r["Mean"], vb),
                              _fmt_val(r["SD"],   vb),
                              _fmt_val(r["p10"],  vb),
                              _fmt_val(r["p25"],  vb),
                              _fmt_val(r["p50"],  vb),
                              _fmt_val(r["p75"],  vb),
                              _fmt_val(r["p90"],  vb)])
                + f" & {int(r['N']):,} \\\\"
            )
        lines.append(r"\addlinespace[0.3em]")
        lines.append(
            f"\\multicolumn{{9}}{{l}}{{\\footnotesize Unique conglomerates: {n_firms:,}; "
            f"firm-quarter observations: {n_fq:,}.}} \\\\"
        )
        return lines

    body = []
    body += _panel("Panel A: Brick-and-Mortar (B) Firms",
                   moments_b, meta["n_firms_B"], meta["n_firm_quarters_B"])
    body += _panel("Panel B: Digital (D) Firms",
                   moments_d, meta["n_firms_D"], meta["n_firm_quarters_D"])

    # The closing \par sits inside the group so the note is set single-spaced in
    # \footnotesize before the \doublespacing that follows the table takes effect.
    notes = (
        r"\par\noindent{" + NOTE_FONT + " " + table_note(
            f"Conglomerate-quarters, {MIN_YEAR}Q1--{MAX_YEAR}Q4; deposits summed across MCAs.",
            NOTE_MONEY,
            r"Zero balances (product not offered) are excluded, so $N$ varies by row; "
            f"{meta['n_firms_B_no_savings']} of {meta['n_firms_B']} B conglomerates never offer "
            r"savings. Type 5 from 2020Q4.",
            r"Spreads are deposit-weighted within firm-quarter and type.",
            NOTE_SPREAD,
            f"Panel B includes {meta['n_firms_D_no_deposits']} D conglomerates that never take "
            r"deposits (credit, finance and leasing firms); they enter only the assets and "
            r"equity rows.",
        ) + r"\par}"
    )

    tex = "\n".join([
        r"\setstretch{1.0}",
        r"\setlength{\LTleft}{\fill}",
        r"\setlength{\LTright}{\fill}",
        # Shrink longtable's ending glue (default \bigskipamount) so the Notes
        # paragraph sits close under the bottom rule.
        r"\setlength{\LTpost}{0.3em}",
        f"\\begin{{longtable}}[c]{{{col_spec}}}",
        f"    \\caption{{{caption}}}\\label{{{tab_label}}} \\\\",
        r"    \toprule",
        f"    {header_row}",
        r"    \endfirsthead",
        "",
        f"    \\multicolumn{{{n_cols}}}{{c}}{{{{\\bfseries Table \\thetable\\ continued from previous page}}}} \\\\",
        r"    \toprule",
        f"    {header_row}",
        r"    \midrule",
        r"    \endhead",
        "",
        r"    \midrule",
        f"    \\multicolumn{{{n_cols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\",
        r"    \endfoot",
        "",
        r"    \bottomrule",
        r"    \endlastfoot",
        "",
        *body,
        "",
        r"\end{longtable}",
        r"\setlength{\LTpost}{\bigskipamount}",
        notes,
        r"\doublespacing",
    ])
    return tex


# ---------------------------------------------------------------------------
# Table 2: MCA-level market structure by year
# ---------------------------------------------------------------------------
def build_table2(df: pd.DataFrame, weight_col: str | None) -> pd.DataFrame:
    df_b = df[df["bank_type"] == "B"].copy()
    df_d = df[df["bank_type"] == "D"].copy()

    df_b["dep_total_jkmt"] = df_b[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].sum(axis=1, min_count=1)

    # ---- Vectorized B-side MCA-quarter aggregates ----
    grp_keys = ["mca_code", "year", "quarter"]

    # Step 1: collapse to firm-level within MCA-quarter
    firm_mq = (
        df_b.groupby(grp_keys + ["CodConglomeradoPrudencial"], sort=False)
            .agg(dep_total=("dep_total_jkmt", "sum"))
            .reset_index()
    )
    firm_mq["dep_total"] = firm_mq["dep_total"].fillna(0.0)

    # Step 2: MCA-quarter totals + n_b_firms
    mq_tot = (
        firm_mq.groupby(grp_keys, sort=False)
               .agg(dep_total_mq=("dep_total", "sum"),
                    n_b_firms=("dep_total", lambda s: int((s > 0).sum())))
               .reset_index()
    )

    # Step 3: HHI — squared share sum within MCA-quarter
    firm_mq = firm_mq.merge(mq_tot[grp_keys + ["dep_total_mq"]], on=grp_keys, how="left")
    firm_mq["share_sq"] = np.where(
        firm_mq["dep_total_mq"] > 0,
        (firm_mq["dep_total"] / firm_mq["dep_total_mq"]) ** 2,
        np.nan,
    )
    hhi = (firm_mq.groupby(grp_keys, sort=False)["share_sq"]
                  .sum(min_count=1)
                  .mul(1e4)
                  .rename("hhi_b")
                  .reset_index())

    # Step 4: annualize spread at row level (percentage points), then deposit-weight
    for _a in ("a4", "a5"):
        df_b[f"spread_ann_{_a}"] = annualized_spread_pp(df_b[f"spread_{_a}"], df_b["risk_free_qoq"])

    for v, w in (("spread_ann_a4", "dep_a4"), ("spread_ann_a5", "dep_a5")):
        df_b[f"_num_{v}"] = df_b[v] * df_b[w].where(df_b[w] > 0, 0.0)
        df_b[f"_den_{v}"] = df_b[w].where(df_b[w] > 0, 0.0).where(df_b[v].notna(), 0.0)

    spread_agg = (
        df_b.groupby(grp_keys, sort=False)
            .agg({"_num_spread_ann_a4": "sum", "_den_spread_ann_a4": "sum",
                  "_num_spread_ann_a5": "sum", "_den_spread_ann_a5": "sum"})
            .reset_index()
    )
    spread_agg["spread_ann_a4_w"] = np.where(spread_agg["_den_spread_ann_a4"] > 0,
                                             spread_agg["_num_spread_ann_a4"] / spread_agg["_den_spread_ann_a4"],
                                             np.nan)
    spread_agg["spread_ann_a5_w"] = np.where(spread_agg["_den_spread_ann_a5"] > 0,
                                             spread_agg["_num_spread_ann_a5"] / spread_agg["_den_spread_ann_a5"],
                                             np.nan)

    # Step 5: MCA-quarter population (first non-null)
    pop_mq = (df_b.groupby(grp_keys, sort=False)["pop_total"]
                   .first()
                   .reset_index())

    mca_q = (mq_tot[grp_keys + ["n_b_firms", "dep_total_mq"]]
             .merge(hhi, on=grp_keys, how="left")
             .merge(spread_agg[grp_keys + ["spread_ann_a4_w", "spread_ann_a5_w"]], on=grp_keys, how="left")
             .merge(pop_mq, on=grp_keys, how="left"))

    # National D-firm count by year (active = positive total deposits)
    df_d["dep_total_jkmt"] = df_d[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].sum(axis=1, min_count=1)
    d_active = df_d[df_d["dep_total_jkmt"] > 0]
    d_count_yr = (d_active.groupby("year")["CodConglomeradoPrudencial"]
                          .nunique()
                          .rename("n_d_firms_natl")
                          .reset_index())

    # National D-firm HHI: squared sum of national deposit shares x 1e4
    d_firm_yr = (d_active.groupby(["year", "CodConglomeradoPrudencial"], sort=False)["dep_total_jkmt"]
                         .sum()
                         .reset_index())
    d_yr_total = (d_firm_yr.groupby("year")["dep_total_jkmt"]
                           .sum()
                           .rename("dep_total_yr")
                           .reset_index())
    d_firm_yr = d_firm_yr.merge(d_yr_total, on="year")
    d_firm_yr["share_sq"] = np.where(
        d_firm_yr["dep_total_yr"] > 0,
        (d_firm_yr["dep_total_jkmt"] / d_firm_yr["dep_total_yr"]) ** 2,
        np.nan,
    )
    d_hhi_yr = (d_firm_yr.groupby("year")["share_sq"]
                         .sum(min_count=1)
                         .mul(1e4)
                         .rename("hhi_d_natl")
                         .reset_index())

    # Combined national HHI: B + D firms, each firm's share = total national deposits / national total
    b_firm_yr = (df_b.groupby(["year", "CodConglomeradoPrudencial"], sort=False)["dep_total_jkmt"]
                     .sum()
                     .reset_index())
    all_firm_yr = pd.concat(
        [b_firm_yr,
         d_firm_yr[["year", "CodConglomeradoPrudencial", "dep_total_jkmt"]]],
        ignore_index=True,
    )
    all_yr_tot = (all_firm_yr.groupby("year")["dep_total_jkmt"]
                             .sum()
                             .rename("dep_all_yr")
                             .reset_index())
    all_firm_yr = all_firm_yr.merge(all_yr_tot, on="year")
    all_firm_yr["share_sq"] = np.where(
        all_firm_yr["dep_all_yr"] > 0,
        (all_firm_yr["dep_total_jkmt"] / all_firm_yr["dep_all_yr"]) ** 2,
        np.nan,
    )
    combined_hhi_yr = (all_firm_yr.groupby("year")["share_sq"]
                                  .sum(min_count=1)
                                  .mul(1e4)
                                  .rename("hhi_combined_natl")
                                  .reset_index())

    # D-firm annualized deposit-weighted spreads by year
    work_d = df_d[["year", "spread_a4", "spread_a5", "dep_a4", "dep_a5", "risk_free_qoq"]].copy()
    for _a in ("a4", "a5"):
        work_d[f"spread_ann_{_a}"] = annualized_spread_pp(work_d[f"spread_{_a}"], work_d["risk_free_qoq"])
    for _v, _w in (("spread_ann_a4", "dep_a4"), ("spread_ann_a5", "dep_a5")):
        _ww = work_d[_w].where(work_d[_w] > 0, 0.0).where(work_d[_v].notna(), 0.0)
        work_d[f"_num_{_v}"] = work_d[_v] * _ww
        work_d[f"_den_{_v}"] = _ww
    d_spread_yr = (work_d.groupby("year")
                         .agg({"_num_spread_ann_a4": "sum", "_den_spread_ann_a4": "sum",
                               "_num_spread_ann_a5": "sum", "_den_spread_ann_a5": "sum"})
                         .reset_index())
    d_spread_yr["spread_ann_a4_d_w"] = np.where(d_spread_yr["_den_spread_ann_a4"] > 0,
                                                 d_spread_yr["_num_spread_ann_a4"] / d_spread_yr["_den_spread_ann_a4"], np.nan)
    d_spread_yr["spread_ann_a5_d_w"] = np.where(d_spread_yr["_den_spread_ann_a5"] > 0,
                                                 d_spread_yr["_num_spread_ann_a5"] / d_spread_yr["_den_spread_ann_a5"], np.nan)

    # National (B+D) annualized deposit-weighted spreads by year
    work_all = pd.concat([
        df_b[["year", "spread_a4", "spread_a5", "dep_a4", "dep_a5", "risk_free_qoq"]],
        work_d[["year", "spread_a4", "spread_a5", "dep_a4", "dep_a5", "risk_free_qoq"]],
    ], ignore_index=True)
    for _a in ("a4", "a5"):
        work_all[f"spread_ann_{_a}"] = annualized_spread_pp(work_all[f"spread_{_a}"], work_all["risk_free_qoq"])
    for _v, _w in (("spread_ann_a4", "dep_a4"), ("spread_ann_a5", "dep_a5")):
        _ww = work_all[_w].where(work_all[_w] > 0, 0.0).where(work_all[_v].notna(), 0.0)
        work_all[f"_num_{_v}"] = work_all[_v] * _ww
        work_all[f"_den_{_v}"] = _ww
    natl_spread_yr = (work_all.groupby("year")
                              .agg({"_num_spread_ann_a4": "sum", "_den_spread_ann_a4": "sum",
                                    "_num_spread_ann_a5": "sum", "_den_spread_ann_a5": "sum"})
                              .reset_index())
    natl_spread_yr["spread_ann_a4_natl_w"] = np.where(natl_spread_yr["_den_spread_ann_a4"] > 0,
                                                       natl_spread_yr["_num_spread_ann_a4"] / natl_spread_yr["_den_spread_ann_a4"], np.nan)
    natl_spread_yr["spread_ann_a5_natl_w"] = np.where(natl_spread_yr["_den_spread_ann_a5"] > 0,
                                                       natl_spread_yr["_num_spread_ann_a5"] / natl_spread_yr["_den_spread_ann_a5"], np.nan)

    years = sorted(df["year"].unique().tolist())
    w_col = weight_col if (weight_col and weight_col in mca_q.columns) else None

    # Panel A: B firms (MCA-level)
    rows = []
    for v in ["n_b_firms", "hhi_b"]:
        row = {"var": v}
        for yr in years:
            sub = mca_q[mca_q["year"] == yr]
            x = sub[v].values
            w = sub[w_col].values if w_col else None
            row[yr] = _weighted_mean(x, w)
        rows.append(row)

    # Panel B: D firms (national)
    drow_count = {"var": "n_d_firms_natl"}
    for yr in years:
        v = d_count_yr.loc[d_count_yr["year"] == yr, "n_d_firms_natl"]
        drow_count[yr] = float(v.iloc[0]) if len(v) else 0.0
    rows.append(drow_count)

    drow_hhi_combined = {"var": "hhi_combined_natl"}
    for yr in years:
        v = combined_hhi_yr.loc[combined_hhi_yr["year"] == yr, "hhi_combined_natl"]
        drow_hhi_combined[yr] = float(v.iloc[0]) if len(v) else 0.0
    rows.append(drow_hhi_combined)

    # Panel C: Risk-free rate (annualized) then annualized deposit-weighted spreads
    # The year's mean quarterly rate, compounded to an annual percent.
    rf_yr = (df.groupby(["year", "quarter"])["risk_free_qoq"].first()
               .groupby("year").mean())
    rf_ann_yr = annualized_rate_pct(rf_yr)
    rf_row = {"var": "risk_free_ann"}
    for yr in years:
        rf_row[yr] = float(rf_ann_yr[yr]) if yr in rf_ann_yr.index else np.nan
    rows.append(rf_row)

    for v in ["spread_ann_a4_w", "spread_ann_a5_w"]:
        row = {"var": v}
        for yr in years:
            sub = mca_q[mca_q["year"] == yr]
            x = sub[v].values
            w = sub[w_col].values if w_col else None
            row[yr] = _weighted_mean(x, w)
        rows.append(row)
    for v, yr_df in [("spread_ann_a4_d_w", d_spread_yr), ("spread_ann_a5_d_w", d_spread_yr),
                     ("spread_ann_a4_natl_w", natl_spread_yr), ("spread_ann_a5_natl_w", natl_spread_yr)]:
        row = {"var": v}
        for yr in years:
            vv = yr_df.loc[yr_df["year"] == yr, v]
            row[yr] = float(vv.iloc[0]) if len(vv) else np.nan
        rows.append(row)

    # N row: number of MCA-quarters per year
    nrow = {"var": "N_obs"}
    for yr in years:
        nrow[yr] = int((mca_q["year"] == yr).sum())
    rows.append(nrow)

    return pd.DataFrame(rows)


def render_table2(t2_df, weight_col, suffix) -> str:
    name      = "Compressed_MarketStructure_by_Year"
    tab_label = f"tab:{name}"
    weight_lbl = ("(Population Weighted)" if weight_col == "pop_total"
                  else "(Deposit Weighted)" if weight_col == "dep_total_mq"
                  else "(Unweighted)")
    caption   = f"MCA-Level Market Structure, Annual Averages {weight_lbl}"

    year_cols = [c for c in t2_df.columns if c != "var"]
    n_groups  = len(year_cols)

    col_spec = "l@{\\hspace{0.2em}}" + ">{\\centering\\arraybackslash}X" * n_groups
    header   = " & " + " & ".join(str(y) for y in year_cols) + r" \\"

    PANELS = [
        ("Panel A: B Firms (MCA-Level)",                ["n_b_firms", "hhi_b"]),
        ("Panel B: D Firms (National)",                 ["n_d_firms_natl", "hhi_combined_natl"]),
        ("Panel C: Market Spreads (Deposit-Weighted)",  ["risk_free_ann",
                                                          "spread_ann_a4_w", "spread_ann_a5_w",
                                                          "spread_ann_a4_d_w", "spread_ann_a5_d_w",
                                                          "spread_ann_a4_natl_w", "spread_ann_a5_natl_w"]),
    ]
    PRE2020_BLANK = {"spread_ann_a5_w", "spread_ann_a5_d_w", "spread_ann_a5_natl_w"}
    t2_idx = t2_df.set_index("var")

    body = []
    for i, (panel_title, vars_in_panel) in enumerate(PANELS):
        if i > 0:
            body.append(r"\midrule")
        body.append(f"\\multicolumn{{{n_groups+1}}}{{l}}{{\\textit{{{panel_title}}}}} \\\\")
        body.append(r"\midrule")
        for vb in vars_in_panel:
            if vb not in t2_idx.index:
                continue
            row = t2_idx.loc[vb]
            lbl = _row_label(vb)
            cells = []
            for y in year_cols:
                val = row[y]
                if vb in PRE2020_BLANK and int(y) < 2020:
                    cells.append("")
                    continue
                cells.append(_fmt_val(val, vb))
            body.append(f"{lbl} & " + " & ".join(cells) + r" \\")
    # N row
    n_row = t2_df[t2_df["var"] == "N_obs"]
    if len(n_row):
        nvals = [f"{int(n_row.iloc[0][y]):,}" for y in year_cols]
        body.append(r"\midrule")
        body.append("MCA-Quarters ($N$) & " + " & ".join(nvals) + r" \\")

    tex = "\n".join([
        r"\begin{landscape}",
        r"\setstretch{1.0}",
        r"\setlength{\LTleft}{\fill}",
        r"\setlength{\LTright}{\fill}",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        f"\\begin{{xltabular}}{{\\linewidth}}{{{col_spec}}}",
        f"    \\caption{{{caption}}}\\label{{{tab_label}}} \\\\",
        r"    \toprule",
        f"    {header}",
        r"    \midrule",
        r"    \endfirsthead",
        r"    \toprule",
        f"    {header}",
        r"    \midrule",
        r"    \endhead",
        r"    \bottomrule",
        r"    \endlastfoot",
        *body,
        r"\end{xltabular}",
        r"\endgroup",
        r"\end{landscape}",
        r"\doublespacing",
    ])
    return tex


def render_table2a(t2_df, weight_col, suffix) -> str:
    """Market structure, policy rate and deposit spreads by year -- years as rows.

    Panel A: B-firm counts and local HHI (MCA level), the D-firm count and the pooled
    national HHI. Panel B: the Selic rate and the type-4 / type-5 spreads by firm type.
    Together the two panels hold the values behind every panel of
    fig_market_structure_by_year.png. `weight_col` is the cross-MCA weight used by
    build_table2 (MCA deposits); the weighting of each series is stated in the notes.
    """
    name      = "Compressed_MarketStructure_by_Year_AB"
    tab_label = f"tab:{name}"
    caption   = "Market Structure, Policy Rate and Deposit Spreads by Year"

    year_cols  = [c for c in t2_df.columns if c != "var"]
    t2_idx     = t2_df.set_index("var")
    n_row_data = t2_df[t2_df["var"] == "N_obs"]

    VARS_A = ["n_b_firms", "hhi_b", "n_d_firms_natl", "hhi_combined_natl"]
    VARS_B = ["risk_free_ann", "spread_ann_a4_w", "spread_ann_a4_d_w",
              "spread_ann_a5_w", "spread_ann_a5_d_w"]
    PRE2020_BLANK = {"spread_ann_a5_w", "spread_ann_a5_d_w"}

    def _val(v, yr):
        return t2_idx.loc[v, yr] if v in t2_idx.index else np.nan

    rows_a = []
    for yr in year_cols:
        cells = [str(yr)] + [_fmt_val(_val(v, yr), v) for v in VARS_A]
        cells.append(f"{int(n_row_data.iloc[0][yr]):,}" if len(n_row_data) else "--")
        rows_a.append(" & ".join(cells) + r" \\")

    rows_b = []
    for yr in year_cols:
        cells = [str(yr)]
        for v in VARS_B:
            cells.append("" if (v in PRE2020_BLANK and int(yr) < 2020)
                         else _fmt_val(_val(v, yr), v))
        rows_b.append(" & ".join(cells) + r" \\")

    unit_b = f"({SPREAD_UNIT})"
    notes = NOTE_FONT + " " + table_note(
        f"Annual means of quarterly values, {MIN_YEAR}--{MAX_YEAR}.",
        r"Panel A: No.\ B Firms counts conglomerates with positive deposits in the MCA-quarter "
        r"and HHI (B) is the within-MCA HHI of B deposits, both averaged across MCA-quarters "
        r"weighted by MCA deposits; No.\ D Firms counts D conglomerates with positive deposits; "
        r"HHI (B+D) is national; $N$ counts MCA-quarters with B presence.",
        r"Panel B: SELIC is the mean quarterly rate compounded to an annual rate.",
        NOTE_SPREAD,
        r"B spreads are deposit-weighted within and across MCA-quarters, D spreads across "
        r"firm-quarters. Type 5 from 2020Q4 (2020 covers Q4 only).",
    )

    tex = "\n".join([
        r"\begin{table}[htbp]",
        r"\setstretch{1.0}",
        r"\centering",
        r"\begin{threeparttable}",
        f"\\caption{{{caption}}}\\label{{{tab_label}}}",
        r"\footnotesize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l rrrrr@{}}",
        r"\toprule",
        r"\multicolumn{6}{@{}l}{\textit{Panel A: Market structure}} \\",
        r"\addlinespace[0.2em]",
        r" & \multicolumn{2}{c}{\textit{B Firms (MCA-Level)}}"
        r" & \multicolumn{2}{c}{\textit{D Firms (National)}} & \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Year & No.\ B Firms & HHI (B) & No.\ D Firms & HHI (B+D) & MCA-Qtrs.\ ($N$) \\",
        r" & (count) & ($0$--$10{,}000$) & (count) & ($0$--$10{,}000$) & \\",
        r"\midrule",
        *rows_a,
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\textit{Panel B: Policy rate and deposit spreads}} \\",
        r"\addlinespace[0.2em]",
        r" & & \multicolumn{2}{c}{\textit{Spread (4)}} & \multicolumn{2}{c}{\textit{Spread (5)}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"Year & SELIC & B Firms & D Firms & B Firms & D Firms \\",
        r" & (\% p.a.) & " + " & ".join([unit_b] * 4) + r" \\",
        r"\midrule",
        *rows_b,
        r"\bottomrule",
        r"\end{tabular*}",
        r"\begin{tablenotes}[flushleft]",
        r"\item " + notes,
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        r"\doublespacing",
    ])
    return tex


def render_table2b(t2_df, weight_col, suffix) -> str:
    """Deposit spreads by firm type and year — years as rows."""
    name      = "Compressed_MarketStructure_by_Year_C"
    tab_label = f"tab:{name}"
    weight_lbl = ("(Pop.\\ Weighted)" if weight_col == "pop_total"
                  else "(Dep.\\ Weighted)" if weight_col == "dep_total_mq"
                  else "")
    caption = ("Deposit Spreads by Year"
               + (f" {weight_lbl}" if weight_lbl else ""))

    year_cols = [c for c in t2_df.columns if c != "var"]
    t2_idx    = t2_df.set_index("var")

    VARS_C        = ["spread_ann_a4_w",     "spread_ann_a5_w",
                     "spread_ann_a4_d_w",   "spread_ann_a5_d_w",
                     "spread_ann_a4_natl_w","spread_ann_a5_natl_w"]
    PRE2020_BLANK = {"spread_ann_a5_w", "spread_ann_a5_d_w", "spread_ann_a5_natl_w"}

    col_spec   = ("l@{\\hspace{1.2em}}"
                  + "r@{\\hspace{1.2em}}"
                  + "rr@{\\hspace{1.2em}}"
                  + "rr@{\\hspace{1.2em}}"
                  + "rr")
    header_top = (
        r" & & \multicolumn{2}{c@{\hspace{1.2em}}}{\textit{B Firms}}"
        r" & \multicolumn{2}{c@{\hspace{1.2em}}}{\textit{D Firms}}"
        r" & \multicolumn{2}{c}{\textit{National (B+D)}} \\"
    )
    cmidrules  = r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}"
    col_labels = (r"Year & SELIC & Spread (4) & Spread (5)"
                  r" & Spread (4) & Spread (5) & Spread (4) & Spread (5) \\")
    col_units  = (r" & (\% p.a.) & " + " & ".join([f"({SPREAD_UNIT})"] * 6) + r" \\")

    body = []
    for yr in year_cols:
        rf_val = t2_idx.loc["risk_free_ann", yr] if "risk_free_ann" in t2_idx.index else np.nan
        cells  = [str(yr), _fmt_val(rf_val, "risk_free_ann")]
        for v in VARS_C:
            if v in PRE2020_BLANK and int(yr) < 2020:
                cells.append("")
            else:
                val = t2_idx.loc[v, yr] if v in t2_idx.index else np.nan
                cells.append(_fmt_val(val, v))
        body.append(" & ".join(cells) + r" \\")

    tex = "\n".join([
        r"\begin{table}[htbp]",
        r"\setstretch{1.0}",
        r"\centering",
        r"\begin{threeparttable}",
        f"\\caption{{{caption}}}\\label{{{tab_label}}}",
        r"\footnotesize",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_top,
        cmidrules,
        col_labels,
        col_units,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\item " + NOTE_FONT + " " + table_note(
            r"SELIC is the mean quarterly rate compounded to an annual rate.",
            NOTE_SPREAD,
            r"Spread\,(5) from 2020Q4; blank before 2020."),
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        r"\doublespacing",
    ])
    return tex


# ---------------------------------------------------------------------------
# Figure: market structure & deposit pricing by year (visual twin of Tables 2/3)
# ---------------------------------------------------------------------------
# Small-multiple grid rather than twin axes: every panel carries ONE y-scale, so
# no pair of series is forced onto an arbitrary shared scaling. Co-movement is
# read off the shared year axis down each column (entry vs concentration on the
# left; spreads vs the policy rate on the right).
#
# Colours reuse the paper's existing bank-type convention from
# make_margin_figures.py (incumbent/B blue, digital/D orange-red).
B_COLOR    = "#1565C0"   # brick-and-mortar (B) / incumbent
D_COLOR    = "#E64A19"   # digital (D)
NATL_COLOR = "#4F4F4F"   # pooled B+D / policy series (not a bank type)
GRID_COLOR = "#D5D5D0"
MUTED_INK  = "#5A5A57"
# Axis frame, ticks and axis labels. Kept distinct from MUTED_INK so annotation ink
# (gridlines, the Pix rule, threshold labels) stays light while the axes read solid.
AXIS_INK   = "#000000"
PIX_YEAR   = 2020.5      # Pix launched Nov 2020


def _t2_series(t2_df: pd.DataFrame, var_name: str, year_cols: list) -> "np.ndarray | None":
    """Numeric series for one `var` row of build_table2's frame, aligned to year_cols."""
    row = t2_df.loc[t2_df["var"] == var_name]
    if row.empty:
        return None
    r = row.iloc[0]
    return np.array([pd.to_numeric(r.get(c, np.nan), errors="coerce") for c in year_cols],
                    dtype=float)


def ms_series(t2_df: pd.DataFrame) -> dict:
    """The series the market-structure figure plots, from build_table2's frame (or the CSV
    main() writes from it): `years` plus one array per plotted line, None when absent."""
    year_cols = [c for c in t2_df.columns if c != "var"]
    S = {"years": np.array([int(c) for c in year_cols], dtype=float)}
    for key, var in (("hhi_b", "hhi_b"), ("hhi_bd", "hhi_combined_natl"),
                     ("n_d", "n_d_firms_natl"), ("n_b", "n_b_firms"),
                     ("selic", "risk_free_ann"),
                     ("sp4_b", "spread_ann_a4_w"), ("sp4_d", "spread_ann_a4_d_w"),
                     ("sp5_b", "spread_ann_a5_w"), ("sp5_d", "spread_ann_a5_d_w")):
        S[key] = _t2_series(t2_df, var, year_cols)
    return S


# build_table2 already returns the display units: spreads in annualized percentage
# points (annualized_spread_pp) and the Selic rate in percent per year.
MS_SPREAD_YLABEL = "percentage points (annualized)"
MS_LINE_KW = dict(linewidth=2, solid_capstyle="round", marker="o", markersize=4.5,
                  markeredgecolor="white", markeredgewidth=1.0, zorder=3)


def ms_style(ax, title, ylabel, pix_label=False, fs=1.0, pix_x=PIX_YEAR):
    """Axis styling shared by every panel. `fs` scales the fonts (1.0 = the paper's grid);
    `pix_x` places the Pix rule, for an x-axis that is not in whole years."""
    ax.set_title(title, fontsize=10.5 * fs, loc="left", pad=6, color="#111111")
    ax.set_ylabel(ylabel, fontsize=9 * fs, color=AXIS_INK)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS_INK)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(labelsize=9 * fs, colors=AXIS_INK, length=3)
    # Pix is a genuine event threshold, so a dashed rule is correct here
    # (gridlines stay solid); matches make_margin_figures.py.
    ax.axvline(pix_x, color="grey", lw=0.8, ls="--", alpha=0.6, zorder=1)
    if pix_label:
        # True labels the rule at the top of the panel; a number sets the height instead.
        pix_y = 0.97 if pix_label is True else float(pix_label)
        ax.text(pix_x + 0.08, pix_y, "Pix", transform=ax.get_xaxis_transform(),
                fontsize=8 * fs, color="grey", va="top", ha="left")


def ms_endlabel(ax, x, y, text, fs=1.0, dy=0):
    if y is None or not np.isfinite(y[-1]):
        return
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(5, dy), textcoords="offset points",
                fontsize=8.5 * fs, color=MUTED_INK, va="center", ha="left")


def ms_year_axis(ax, years, fs=1.0):
    ax.set_xlabel("Year", fontsize=9 * fs, color=MUTED_INK)
    ax.set_xticks(years)
    ax.set_xticklabels([f"{int(y)}" for y in years], rotation=0)


def _ms_title(letter, text, lettered):
    return f"({letter}) {text}" if lettered else text


# One function per panel, shared by the paper's grid and by make_slide_figures.py, which
# draws each panel alone. `lettered` keeps the "(a)" prefix of the grid; `pix_label` None
# keeps the grid's choice of which panels name the Pix rule.
def _ms_panel_a(ax, S, fs=1.0, lettered=True, pix_label=None):
    years, hhi_b, hhi_bd = S["years"], S["hhi_b"], S["hhi_bd"]
    # Reference thresholds, right-aligned: above each line the right-hand side is
    # empty, so the labels clear both series. Dotted keeps them distinct from the
    # dashed Pix rule.
    for thr, lab in ((2500, "2,500  highly concentrated"), (1500, "1,500  moderately conc.")):
        ax.axhline(thr, color="#C9C9C4", lw=0.7, ls=(0, (1, 3)), zorder=1)
        ax.text(years[-1] - 0.1, thr + 25, lab, fontsize=7 * fs, color="#9A9A95",
                va="bottom", ha="right")
    if hhi_b is not None:
        ax.plot(years, hhi_b, color=B_COLOR, label="B, local (within MCA)", **MS_LINE_KW)
        ms_endlabel(ax, years, hhi_b, f"{hhi_b[-1]:,.0f}", fs)
    if hhi_bd is not None:
        ax.plot(years, hhi_bd, color=NATL_COLOR, label="B+D, national", **MS_LINE_KW)
        ms_endlabel(ax, years, hhi_bd, f"{hhi_bd[-1]:,.0f}", fs)
    ms_style(ax, _ms_title("a", "Deposit concentration", lettered), "HHI (0–10,000)",
             pix_label=True if pix_label is None else pix_label, fs=fs)
    ax.legend(fontsize=8.5 * fs, loc="lower left", frameon=False)


def _ms_panel_b(ax, S, fs=1.0, lettered=True, pix_label=None):
    years, selic = S["years"], S["selic"]
    if selic is not None:
        ax.plot(years, selic, color=NATL_COLOR, **MS_LINE_KW)
        ms_endlabel(ax, years, selic, f"{selic[-1]:.1f}%", fs)
    ms_style(ax, _ms_title("b", "SELIC policy rate", lettered), "% p.a.",
             pix_label=True if pix_label is None else pix_label, fs=fs)


def _ms_panel_c(ax, S, fs=1.0, lettered=True, pix_label=None):
    years, n_d = S["years"], S["n_d"]
    if n_d is not None:
        ax.plot(years, n_d, color=D_COLOR, **MS_LINE_KW)
        ms_endlabel(ax, years, n_d, f"{n_d[-1]:,.0f}", fs)
    ms_style(ax, _ms_title("c", "Digital (D) firms, national", lettered), "count",
             pix_label=pix_label or False, fs=fs)


def _ms_panel_d(ax, S, fs=1.0, lettered=True, pix_label=None):
    years, n_b = S["years"], S["n_b"]
    if n_b is not None:
        ax.plot(years, n_b, color=B_COLOR, **MS_LINE_KW)
        ms_endlabel(ax, years, n_b, f"{n_b[-1]:.1f}", fs)
    ms_style(ax, _ms_title("d", "B firms per MCA (mean)", lettered), "count",
             pix_label=pix_label or False, fs=fs)


def _ms_panel_e(ax, S, fs=1.0, lettered=True, pix_label=None):
    years, sp4_b, sp4_d = S["years"], S["sp4_b"], S["sp4_d"]
    ax.axhline(0, color="#BFBFBA", lw=0.8, zorder=1)
    if sp4_b is not None:
        ax.plot(years, sp4_b, color=B_COLOR, label="B (brick-and-mortar)", **MS_LINE_KW)
    if sp4_d is not None:
        ax.plot(years, sp4_d, color=D_COLOR, label="D (digital)", **MS_LINE_KW)
    ms_style(ax, _ms_title("e", "Time-deposit spread (type 4)", lettered), MS_SPREAD_YLABEL,
             pix_label=pix_label or False, fs=fs)
    ax.legend(fontsize=8.5 * fs, loc="upper left", frameon=False)


def _ms_panel_f(ax, S, fs=1.0, lettered=True, pix_label=None):
    years = S["years"]
    ax.axhline(0, color="#BFBFBA", lw=0.8, zorder=1)
    # Both types sit indistinguishably at zero, so a solid D would hide B entirely.
    # Dashing D keeps both readable without displacing either value.
    for arr, col, lab, ls in ((S["sp5_b"], B_COLOR, "B (brick-and-mortar)", "-"),
                              (S["sp5_d"], D_COLOR, "D (digital)", (0, (5, 2)))):
        if arr is None:
            continue
        m = np.isfinite(arr)
        if m.any():
            kw = {**MS_LINE_KW, "linestyle": ls}
            ax.plot(years[m], arr[m], color=col, label=lab, **kw)
    ms_style(ax, _ms_title("f", "Prepaid spread (type 5), 2020–", lettered), MS_SPREAD_YLABEL,
             pix_label=pix_label or False, fs=fs)
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=8.5 * fs, loc="upper left", frameon=False)


#: Panel letter -> drawing function, in the grid's reading order (row by row).
MS_PANELS = {"a": _ms_panel_a, "b": _ms_panel_b, "c": _ms_panel_c,
             "d": _ms_panel_d, "e": _ms_panel_e, "f": _ms_panel_f}


def render_market_structure_figure(t2_df: pd.DataFrame, suffix: str = "") -> None:
    """2x3 small-multiple figure summarising Tables 2 and 3.

    Left column  - market structure: concentration, digital entry, local entry.
    Right column - deposit pricing:  type-4 spread, the SELIC policy rate, type-5 spread.
    """
    import matplotlib
    matplotlib.use("Agg")          # venv lives in OneDrive; Agg keeps batch runs safe
    import matplotlib.pyplot as plt

    S = ms_series(t2_df)
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharex="col")
    for ax, key in zip(axes.ravel(), MS_PANELS):
        MS_PANELS[key](ax, S)
    for ax in axes[2, :]:
        ms_year_axis(ax, S["years"])

    fig.tight_layout(h_pad=1.6, w_pad=2.4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"fig_market_structure_by_year{suffix}.png"
    p = DRAFTS_DIR / name
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Wrote {p}")


# ---------------------------------------------------------------------------
# Table 3: Local environment by macro-region
# ---------------------------------------------------------------------------
T3_VARS = ["pop_total", "gdp_per_capita", "fraction_65plus",
           "cadunico_families_per1000", "branches_per1000",
           "connections_per100", "pix_users_pf_per1000", "pix_txns_pf"]


def build_table3(df: pd.DataFrame, weight_col: str | None) -> pd.DataFrame:
    df_b = df[df["bank_type"] == "B"].copy()

    # Take the latest available year per MCA-variable to capture levels at end
    # of sample.  Use the year-mean within latest year to smooth quarterly noise.
    latest_year = int(df_b["year"].max())
    snap = df_b[df_b["year"] == latest_year]

    # One row per MCA (collapse over firms and quarters within the year)
    mca = snap.groupby("mca_code", sort=False).agg(
        **{v: (v, "mean") for v in T3_VARS if v in snap.columns},
        region=("region", "first"),
        weight=(weight_col, "mean") if (weight_col and weight_col in snap.columns) else ("year", "size"),
    ).reset_index()

    regions = ["North", "Northeast", "Southeast", "South", "Center-West"]
    rows = []
    for v in T3_VARS:
        if v not in mca.columns:
            continue
        row = {"var": v}
        for r in regions:
            sub = mca[mca["region"] == r]
            row[r] = _weighted_mean(sub[v].values,
                                    sub["weight"].values if weight_col else None)
        row["National"] = _weighted_mean(mca[v].values,
                                         mca["weight"].values if weight_col else None)
        rows.append(row)

    # N: # of MCAs per region
    nrow = {"var": "N_obs"}
    for r in regions:
        nrow[r] = int((mca["region"] == r).sum())
    nrow["National"] = int(len(mca))
    rows.append(nrow)

    return pd.DataFrame(rows)


def render_table3(t3_df, weight_col, suffix, latest_year) -> str:
    name      = "Compressed_LocalEnvironment_by_Region"
    tab_label = f"tab:{name}"
    weight_lbl = "(Population Weighted)" if weight_col else "(Unweighted)"
    caption = (f"Local Environment by Macro-Region, "
               f"{latest_year} Cross-Section {weight_lbl}")

    region_cols = ["North", "Northeast", "Southeast", "South", "Center-West", "National"]
    col_spec  = "l@{\\hspace{0.4em}}" + "r" * len(region_cols)
    header    = "Variable & " + " & ".join(region_cols) + r" \\"

    body = []
    for _, row in t3_df.iterrows():
        vb = row["var"]
        if vb == "N_obs":
            continue
        cells = [_fmt_val(row[c], vb) for c in region_cols]
        body.append(f"{_row_label(vb)} & " + " & ".join(cells) + r" \\")
    n_row = t3_df[t3_df["var"] == "N_obs"]
    if len(n_row):
        body.append(r"\midrule")
        body.append("MCAs ($N$) & " + " & ".join(
            f"{int(n_row.iloc[0][c]):,}" for c in region_cols) + r" \\")

    notes = (
        NOTE_FONT + r" \textit{Notes:} Each cell is the cross-MCA mean within "
        r"the macro-region, using only B-firm coverage to identify local "
        r"markets. MCA-level values are obtained by averaging across firms "
        r"and quarters within the latest sample year "
        f"({latest_year}). "
        + ("Cross-MCA averages are population-weighted using "
           r"\texttt{pop\_total}."
           if weight_col else "Cross-MCA averages are unweighted.")
    )

    tex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{threeparttable}",
        f"\\caption{{{caption}}}\\label{{{tab_label}}}",
        r"\footnotesize",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\item " + notes,
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        r"\doublespacing",
    ])
    return tex


# ---------------------------------------------------------------------------
# Table 4: D-firm pre/post Pix (IK/CSS cluster-robust)
# ---------------------------------------------------------------------------
T4_VARS = ["log_total_assets", "log_dep", "log_dep_a4", "spread_ann_a4", "equity_ratio_pct"]
# Unit of the post-minus-pre difference in each row: log points for the log rows,
# percentage points for the spread and for the equity ratio (shown in percent of assets).
T4_DIFF_UNIT = {
    "log_total_assets": "log points",
    "log_dep":          "log points",
    "log_dep_a4":       "log points",
    "spread_ann_a4":    "pp",
    "equity_ratio_pct": "pp",
}
LABEL_MAP["equity_ratio_pct"] = ("Equity Ratio", r"\% of assets", None)


def build_table4(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_d = df[df["bank_type"] == "D"].copy()
    df_d = df_d[["CodConglomeradoPrudencial", "year", "quarter", "post",
                 "total_assets", "equity_ratio", "dep_a1", "dep_a2", "dep_a4", "dep_a5",
                 "spread_a4", "risk_free_qoq"]].copy()
    df_d["log_total_assets"] = np.log(df_d["total_assets"].where(df_d["total_assets"] > 0))
    # Total deposits as in Table 1: the sum of the four retail types (1, 2, 4, 5), not the
    # panel's IF-Data total-deposits account, which also carries other deposit categories.
    dep_retail = df_d[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].sum(axis=1, min_count=1)
    df_d["log_dep"]       = np.log(dep_retail.where(dep_retail > 0))
    df_d["log_dep_a4"]       = np.log(df_d["dep_a4"].where(df_d["dep_a4"] > 0))
    # Spread only where the firm-quarter holds time deposits (the Table 1 D spread sample):
    # without a type-4 balance the panel carries a fallback spread, not an observed rate.
    holds_a4 = df_d["dep_a4"] > 0
    df_d["spread_ann_a4"] = annualized_spread_pp(df_d["spread_a4"], df_d["risk_free_qoq"]).where(holds_a4)
    df_d["equity_ratio_pct"] = 100.0 * df_d["equity_ratio"]

    results = []
    for v in T4_VARS:
        r = _cluster_diff_test(df_d, y_col=v, cluster_col="CodConglomeradoPrudencial",
                               post_col="post")
        r["var"] = v
        results.append(r)
    res_df = pd.DataFrame(results)

    # Active D-firm count, by period: context row, no clustering test
    actives = (df_d.assign(active=df_d["total_assets"].fillna(0) > 0)
                    .groupby(["CodConglomeradoPrudencial", "post"])["active"]
                    .max()
                    .reset_index())
    n_d_pre  = int(actives.loc[(actives["post"] == 0) & actives["active"],
                                "CodConglomeradoPrudencial"].nunique())
    n_d_post = int(actives.loc[(actives["post"] == 1) & actives["active"],
                                "CodConglomeradoPrudencial"].nunique())

    meta = {
        "n_firm_quarters_pre":  int((df_d["post"] == 0).sum()),
        "n_firm_quarters_post": int((df_d["post"] == 1).sum()),
        "n_d_firms_pre":        n_d_pre,
        "n_d_firms_post":       n_d_post,
        "n_clusters":           int(df_d["CodConglomeradoPrudencial"].nunique()),
        "n_fq":                 int(len(df_d)),
        "n_fq_dep4":            int(holds_a4.sum()),
    }
    return res_df, meta


def render_table4(t4_df, meta, suffix) -> str:
    name      = "Compressed_DFirm_PrePost_Pix"
    tab_label = f"tab:{name}"
    caption   = ("Digital (D) Firms Before and After Pix: Difference-in-Means "
                 "with Conglomerate-Clustered Inference")

    # tabular* at \textwidth with \extracolsep: the fill spreads the slack
    # across columns so the table (and its threeparttable notes) span the page.
    col_spec = "@{\\extracolsep{\\fill}}l cccccc@{}"
    header   = (r"Variable & \multicolumn{1}{c}{Pre Mean} & \multicolumn{1}{c}{Post Mean}"
                r" & \multicolumn{1}{c}{$\Delta$} & \multicolumn{1}{c}{Unit of $\Delta$}"
                r" & \multicolumn{1}{c}{$G$} & \multicolumn{1}{c}{$G^{\star}$} \\")

    body = []
    for _, r in t4_df.iterrows():
        vb = r["var"]
        diff_cell = f"{_fmt_val(r['diff'], vb)}{_stars(r['p'])}"
        se_cell   = f"({_fmt_val(r['se'], vb)})"
        body.append(
            f"{_row_label(vb)} & "
            f"{_fmt_val(r['pre_mean'],  vb)} & "
            f"{_fmt_val(r['post_mean'], vb)} & "
            f"{diff_cell} & "
            f"{T4_DIFF_UNIT.get(vb, '')} & "
            f"{int(r['G']):,} & "
            f"{r['G_star']:.3f} \\\\"
        )
        body.append(
            f" & & & {se_cell} & & & \\\\[0.3em]"
        )

    # Context row: active D-firm counts (no test).
    body.append(r"\addlinespace[0.4em]")
    body.append(r"\midrule")
    body.append(
        f"Active D Conglomerates "
        f"& {meta['n_d_firms_pre']:,} "
        f"& {meta['n_d_firms_post']:,} "
        f"& {meta['n_d_firms_post'] - meta['n_d_firms_pre']:+,} "
        r"& count & \multicolumn{2}{c}{\textit{(no test)}} \\"
    )

    notes = NOTE_FONT + " " + table_note(
        f"D conglomerate-quarters, {MIN_YEAR}--{MAX_YEAR}; post: from 2020Q4 (Pix).",
        r"$\Delta$: post minus pre mean, conglomerate-clustered SEs in parentheses, "
        r"$p$-values from $t(G^{\star})$; $G$: clusters; $G^{\star}$: "
        r"\textcite{carter2017asymptotic} effective clusters.",
        f"Composition changes: {meta['n_d_firms_pre']:,} active D conglomerates pre, "
        f"{meta['n_d_firms_post']:,} post.",
        r"Log Deposits sums types 1, 2, 4 and 5, as in "
        r"Table~\ref{tab:Compressed_BankType_CrossSection}.",
        f"Spread row: firm-quarters with time deposits only ({meta['n_fq_dep4']:,} of "
        f"{meta['n_fq']:,}).",
        NOTE_SPREAD,
        NOTE_STARS,
    )

    tex = "\n".join([
        # [H] (exact placement, `float` package) rather than [htbp]: the table is discussed inline
        # and must not drift to the top of another page.
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{threeparttable}",
        f"\\caption{{{caption}}}\\label{{{tab_label}}}",
        r"\footnotesize",
        f"\\begin{{tabular*}}{{\\textwidth}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular*}",
        r"\begin{tablenotes}[flushleft]",
        r"\item " + notes,
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        r"\doublespacing",
    ])
    return tex


# ---------------------------------------------------------------------------
# Appendix: Macro interest rates by year
# ---------------------------------------------------------------------------
def build_appendix_rates() -> pd.DataFrame:
    """Annual averages of macro interest rates from quarterly_macro_rates.csv.
    Quarterly compound rates are annualized as (1 + r_qoq/100)^4 - 1."""
    df = pd.read_csv(MACRO_RATES_CSV)
    df["year"]    = df["AnoMes"] // 100
    df["quarter"] = df["AnoMes"] % 100

    # This table reports rates BY YEAR, so it must cover the same years as the estimation sample.
    df = apply_window(df, label="desc_2 macro rates")

    for col in ["selic_qoq", "cdi_qoq", "savings_rate_qoq"]:
        df[f"{col}_ann"] = ((1 + df[col] / 100) ** 4 - 1) * 100

    annual = (
        df.groupby("year")
          .agg(
              selic_pa   = ("selic_qoq_ann",        "mean"),
              cdi_pa     = ("cdi_qoq_ann",           "mean"),
              savings_pa = ("savings_rate_qoq_ann",  "mean"),
              meta_selic = ("meta_selic",            "last"),
          )
          .reset_index()
    )
    return annual


def render_appendix_rates(df: pd.DataFrame) -> str:
    """Portrait table of annualized macro rates by year."""
    name      = "Appendix_Macro_Rates_by_Year"
    tab_label = f"tab:{name}"
    caption   = "Brazilian Macro Interest Rates by Year"

    # cols: Year | SELIC | CDI | Savings | COPOM target
    # col positions: 1=Year, 2-4=market rates (cmidrule), 5=COPOM target
    col_spec   = "l@{\\hspace{1.2em}}rrr@{\\hspace{1.2em}}r"
    header_top = (
        r" & \multicolumn{3}{c@{\hspace{1.2em}}}{\textit{Market Rates (annualized)}} & \\"
    )
    cmidrules  = r"\cmidrule(lr){2-4}"
    col_labels = r"Year & SELIC & CDI & Savings Rate & COPOM Target \\"
    col_units  = r" & (\% p.a.) & (\% p.a.) & (\% p.a.) & (\% p.a.) \\"

    body = []
    for _, row in df.iterrows():
        body.append(
            f"{int(row['year'])} & "
            f"{row['selic_pa']:.2f} & "
            f"{row['cdi_pa']:.2f} & "
            f"{row['savings_pa']:.2f} & "
            f"{row['meta_selic']:.2f} \\\\"
        )

    notes = (
        NOTE_FONT + r" \textit{Notes:} "
        r"SELIC overnight, CDI, and savings rates are quarterly rates compounded to annual: "
        r"$(1 + r_{\text{qoq}})^{4} - 1$, then averaged over the four quarters of each year. "
        r"COPOM Target is the last policy rate set by the Monetary Policy Committee within the year. "
        r"The SELIC overnight rate is used as the risk-free benchmark in the deposit spread construction "
        r"(see Table~\ref{tab:Compressed_MarketStructure_by_Year_C})."
    )

    # Non-floating: use \captionof so the table sits exactly in the text flow
    # alongside the surrounding longtables (which are also non-floating).
    tex = "\n".join([
        r"\setstretch{1.0}",
        r"\begin{center}",
        r"\begin{threeparttable}",
        f"\\captionof{{table}}{{{caption}}}\\label{{{tab_label}}}",
        r"\footnotesize",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_top,
        cmidrules,
        col_labels,
        col_units,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\item " + notes,
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{center}",
        r"\doublespacing",
    ])
    return tex


# ---------------------------------------------------------------------------
# Appendix: consolidated variable dictionaries (master tables A and B).
# These replace the former per-role glossary longtables (identifiers, deposits,
# rates, state variables, institution chars, cost shifters, wholesale funding,
# capital adequacy, BLP LOO, prod/demographic chars, BLP instruments) with two
# master tables: (A) estimation variables grouped by role with source and
# used-in columns; (B) instruments with the estimation stage they enter. Only
# variables that appear in an estimated specification are printed; collected-
# but-unused columns are documented in the replication package data dictionary.
# Each master table carries the labels of every table it absorbed, so all
# existing \ref{}s in V_Main.tex keep resolving after the \input swap.
# ---------------------------------------------------------------------------
def _master_longtable(caption: str, labels: list[str], col_spec: str,
                      header_row: str, n_cols: int, rows: list[str],
                      notes: str) -> str:
    """xltabular skeleton for the consolidated dictionaries: full \\textwidth,
    notes width matched to the table, continuation headers on page breaks."""
    label_str = "".join(f"\\label{{{l}}}" for l in labels)
    cont = f"{caption} (Continued)"
    return "\n".join([
        r"\setstretch{1.0}",
        r"\begingroup",
        r"\small",
        f"\\begin{{xltabular}}{{\\textwidth}}{{{col_spec}}}",
        f"    \\caption{{{caption}}}{label_str} \\\\",
        r"    \toprule",
        f"    {header_row} \\\\",
        r"    \midrule",
        r"    \endfirsthead",
        f"    \\caption[]{{{cont}}} \\\\",
        r"    \toprule",
        f"    {header_row} \\\\",
        r"    \midrule",
        r"    \endhead",
        r"    \midrule",
        f"    \\multicolumn{{{n_cols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\",
        r"    \endfoot",
        r"    \bottomrule",
        f"    \\multicolumn{{{n_cols}}}{{p{{\\dimexpr\\textwidth-2\\tabcolsep\\relax}}}}{{{notes}}} \\\\",
        r"    \endlastfoot",
        *rows,
        r"\end{xltabular}",
        r"\endgroup",
        r"\doublespacing",
    ])


def _grouped_rows(groups, n_cols: int) -> list[str]:
    """Emit group subheaders + data rows for a master table."""
    out = []
    for gi, (gname, grows) in enumerate(groups):
        if gi > 0:
            out.append(r"    \addlinespace[0.7em]")
        # No rule or glue under group headers: booktabs' below-rule/-space glue
        # is a legal longtable breakpoint (it orphans the header). The header
        # row ends \\*[0.15em], which both adds the small gap and forbids the
        # break, gluing the header to its first data row.
        out.append(f"    \\multicolumn{{{n_cols}}}{{l}}{{\\itshape {gname}}} \\\\*[0.15em]")
        for cells in grows:
            out.append("    " + " & ".join(cells) + r" \\")
    return out


# Master table A: variables used in estimation, grouped by role.
# 'Sleep' = sleepiness estimation E1-E4 at specification 12: the state block of
# sleep_est_e2.define_specifications (pix_exists, cadunico_families_per1000,
# fraction_65plus, risk_free_qoq_lag, connections_per100), entering phi interacted
# with lagged deposits. 'Demand' = the logit/BLP system as reported in the BLP table:
# X = blp_logit.X_COLS (fgc_covered, has_ip, seg_S2-S5, log_total_assets_lag,
# is_state_owned); pi = the "ext1" interactions of blp_engine_*.build_theta2_structure
# (spread x gdp_per_capita, spread x fraction_65plus, spread x connections_per100,
# log assets x gdp_per_capita). gdp_growth_yoy is the time block of E4, the only
# +Time routine in the lineup.
_VARS_MASTER_GROUPS = [
    ("Identifiers and panel structure", [
        (r"CodConglomeradoPrudencial", r"Conglomerate $j$ (prudential C-code)", "BCB", "All"),
        (r"CNPJ\_Lider, CNPJ", r"Lead-institution and root-level CNPJ of the conglomerate", "BCB", "All"),
        (r"CODMUN\_IBGE", r"7-digit IBGE municipality code; \texttt{0} = nationally active D institution", "IBGE", "All"),
        # The count is filled in by render_variables_master from the panel (count_mcas).
        (r"mca\_code", r"Market $m$ --- one of __N_MCAS__ MCAs, or \texttt{NATIONAL} for D institutions", "Constructed", "All"),
        (r"year, quarter, AnoMes", r"Period $t$; \texttt{AnoMes} is the BCB IF-Data code YYYYMM", "BCB", "All"),
        (r"deposit\_type", r"Category $k$: 1 = demand, 2 = savings, 3 = interbank, 4 = time/CDB, 5 = prepaid", "BCB", "All"),
    ]),
    ("Outcomes: deposit stocks", [
        (r"dep\_a1--dep\_a5, deposit\_balance", r"Type-$k$ deposit stock $\mathrm{Dep}_{jkmt}$; MCA level for B firms, national for D firms", "ESTBAN, IF-Data", "Sleep; Demand"),
        (r"lagged\_deposits", r"\texttt{deposit\_balance} shifted one quarter within institution $\times$ type", "Constructed", "Sleep"),
        (r"total\_deposits, lagged\_total\_deposits", r"Sum across the five deposit types, and its one-quarter lag", "Constructed", "Sleep"),
        (r"pop\_total", r"Total MCA population $\mathrm{Pop}_{mt}$; scales the market size $M_{mt} = \hat{\bar{d}}_{mt}\cdot\mathrm{Pop}_{mt}$", "IBGE", "Demand (shares)"),
    ]),
    ("Rates and spreads", [
        (r"selic\_qoq, risk\_free\_qoq", r"SELIC overnight compounded QoQ, $(1+r_{\text{daily}})^{63}-1$; the risk-free benchmark", "BCB SGS", "Sleep; Demand"),
        (r"deposit\_rate\_qoq", r"Type-specific deposit rate, QoQ decimal", "BCB, COSIF", "Sleep; Demand"),
        (r"spread\_qoq", r"$r^{\mathrm{f}}_t - r^{\mathrm{dep}}_{jkmt}$, QoQ decimal; the endogenous price in the sleepiness estimation", "Constructed", "Sleep"),
        (r"risk\_free\_ann, deposit\_rate\_ann", r"Annualized counterparts, $(1+x)^4-1$", "Constructed", "Demand"),
        (r"spread\_ann", r"$(1+r^{\mathrm{f}}_t)^4-(1+r^{\mathrm{dep}}_{jkmt})^4$; in pp it is the demand price $\rho_{jkmt}$", "Constructed", "Demand"),
        (r"risk\_free\_qoq\_lag", r"One-quarter lag of the Selic rate (state variable)", "BCB SGS", "Sleep"),
        (r"gdp\_growth\_yoy", r"Year-over-year growth of MCA GDP \textit{per capita}; the time block of the +Time estimator", "IBGE", "Sleep (+Time)"),
    ]),
    ("Market-level state variables and demographics", [
        (r"pix\_exists", r"Pix availability indicator (from 2020Q4)", "BCB", "Sleep"),
        (r"gdp\_per\_capita", r"Municipal GDP aggregated to MCA, divided by population", "IBGE", r"Demand ($\pi$)"),
        (r"fraction\_65plus", r"Share of population aged 65+", "IBGE Census", r"Sleep; Demand ($\pi$)"),
        (r"connections\_per100", r"Mobile lines: active mobile-telephone accesses (ANATEL), per 100 inhabitants; municipal from 2019, earlier years apportioned from area-code totals by population", "ANATEL", r"Sleep; Demand ($\pi$)"),
        (r"cadunico\_families\_per1000", r"Low-income families registered in Cad\'Unico per 1,000 inhabitants", "SAGI/MDS", "Sleep"),
    ]),
    ("Product characteristics", [
        (r"fgc\_covered", r"FGC deposit-insurance indicator (types 1, 2, and 4)", "Constructed", r"Demand ($X$)"),
        (r"has\_ip", r"Payment-institution subsidiary flag", "BCB", r"Demand ($X$)"),
        (r"is\_state\_owned", r"State-controlled conglomerate (BCB control type: public)", "IF-Data", r"Demand ($X$)"),
        (r"segment, seg\_S2--seg\_S5", r"BCB prudential segment S1--S5 and its dummies", "BCB", r"Demand ($X$)"),
        (r"total\_assets, log\_total\_assets\_lag", r"Total balance-sheet assets and $\ln(\cdot)$, lagged one quarter", "IF-Data", r"Demand ($X$, $\pi$)"),
        (r"equity\_ratio", r"\texttt{equity}/\texttt{total\_assets}; basis for the rival leave-one-out instruments (Table \ref{tab:instruments_master})", "IF-Data", "Instruments"),
    ]),
]


def count_mcas(df: pd.DataFrame) -> int:
    """Local markets in the sample: distinct MCAs with a B-firm row in the prepared panel."""
    return int(df.loc[df["bank_type"] == "B", "mca_code"].nunique())


def render_variables_master(n_mcas: int) -> str:
    notes = NOTE_FONT + " " + table_note(
        r"Only variables entering an estimated specification are listed.",
        r"``Sleep'': sleepiness estimation, specification (12), approaches "
        r"\ref{estimation:local}--\ref{estimation:single_idx_time} of "
        r"Section~\ref{sec:empirical:sleep}; ``+Time'': the \ref{estimation:single_idx_time} "
        r"time block; ``Demand'': logit/BLP ($X$: product characteristics; $\pi$: estimated "
        r"demographic interactions; shares: market-share construction).",
        r"Collected variables that enter no specification (population share aged 15--20, other "
        r"Pix measures, 4G/5G share, branch and access-point densities, Cad\'Unico poverty "
        r"shares, imputation and interpolation flags) are documented in the replication "
        r"package's data dictionary.",
    )
    rows = _grouped_rows(
        [(g, [(rf"{n}", d.replace("__N_MCAS__", f"{n_mcas:,}"), s, u) for n, d, s, u in rws])
         for g, rws in _VARS_MASTER_GROUPS],
        n_cols=4,
    )
    return _master_longtable(
        caption    = "Variables Used in Estimation, by Role",
        labels     = ["tab:variables_master", "tab:identifiers", "tab:deposits",
                      "tab:rates_spreads", "tab:state_variables_app",
                      "tab:institution_characteristics", "tab:prod_chars",
                      "tab:demographic_chars"],
        col_spec   = (r">{\ttfamily\footnotesize}p{4.6cm} "
                      r">{\raggedright\arraybackslash}X "
                      r">{\raggedright\arraybackslash}p{2.2cm} "
                      r">{\raggedright\arraybackslash}p{2.6cm}"),
        header_row = (r"\normalfont\small\textbf{Variable} & \textbf{Definition} "
                      r"& \textbf{Source} & \textbf{Used in}"),
        n_cols     = 4,
        rows       = rows,
        notes      = notes,
    )


# Master table B: instruments with the stage they enter. Stage strings follow
# the estimated sets exactly: the sleepiness first stage (nested IV sets in
# estimation_2_sleep.define_specifications) and the K=16 demand vector
# vcat(IV_BLP_LOO, IV_ESTBAN, IV_COST, IV_CAPITAL) in blp_engine_cpu.jl /
# blp_logit.jl. leave_one_out_mean_spread (the Hausman IV) was previously
# undocumented; the LOO lci/wholesale pairs listed in the old BLP-LOO table are
# NOT in the estimated demand vector and are therefore not printed.
_STAGE_BOTH   = "Sleep 1st stage; Demand"
_STAGE_SLEEP  = "Sleep 1st stage"
_STAGE_DEMAND = "Demand (logit/BLP)"

_INSTR_MASTER_GROUPS = [
    ("Operating cost shifters (IF-Data DRE)", [
        (r"personnel\_cost\_ratio\_lag", r"$|\text{personnel expense}|/\text{assets}$ (COSIF 78215), lagged 1Q", _STAGE_BOTH),
        (r"admin\_cost\_ratio\_lag", r"$|\text{administrative expense}|/\text{assets}$ (COSIF 78214), lagged 1Q", _STAGE_BOTH),
        (r"tax\_cost\_ratio\_lag", r"$|\text{tax expense}|/\text{assets}$ (COSIF 78220), lagged 1Q", _STAGE_BOTH),
    ]),
    ("Funding mix (IF-Data Passivo)", [
        (r"lci\_lca\_ratio\_lag", r"(LCI 78289 + LCA 78290)$/\text{assets}$, lagged 1Q; alternative funding-source pressure", _STAGE_SLEEP),
        (r"wholesale\_ratio\_lag", r"Wholesale funding accounts$/\text{assets}$, lagged 1Q; aggregate wholesale dependence", _STAGE_SLEEP),
    ]),
    ("Capital adequacy (IF-Data Capital)", [
        (r"indice\_basileia\_lag", r"Basel index --- total regulatory capital ratio (COSIF 79664), lagged 1Q", _STAGE_BOTH),
    ]),
    ("Hausman-style", [
        (r"leave\_one\_out\_mean\_spread", r"Mean spread of the other conglomerate $\times$ market rows (MCA; national for D firms) in the same deposit type $\times$ quarter. Only the own row is left out, so the conglomerate's spreads in its other markets stay in", _STAGE_SLEEP),
    ]),
    (r"BLP leave-one-out rival characteristics, group $g=(\text{MCA},\text{quarter})$", [
        (r"loo\_log\_assets, mean\_loo\_log\_assets", r"LOO sum and mean of rival $\ln(\text{Total Assets}_{t-1})$", _STAGE_DEMAND),
        (r"loo\_equity\_ratio, mean\_loo\_equity\_ratio", r"LOO sum and mean of rival equity ratio", _STAGE_DEMAND),
        (r"loo\_basileia, mean\_loo\_basileia", r"LOO sum and mean of rival Basel index", _STAGE_DEMAND),
        (r"loo\_credit\_assets, mean\_loo\_credit\_assets", r"LOO sum and mean of rival credit-to-assets ratio", _STAGE_DEMAND),
        (r"loo\_npl\_provision, mean\_loo\_npl\_provision", r"LOO sum and mean of rival non-performing-loan provisions", _STAGE_DEMAND),
        (r"n\_rivals", r"Number of non-null rivals in the group", _STAGE_DEMAND),
    ]),
    ("Branch competition (ESTBAN)", [
        (r"estban\_rival\_branches\_lag", r"Log of one plus the count of rival bank branches in the MCA, lagged 1Q", _STAGE_DEMAND),
    ]),
]


def render_instruments_master() -> str:
    notes = NOTE_FONT + " " + table_note(
        r"``Sleep 1st stage'' instruments enter the deposit-spread equation under the nested "
        r"IV sets (cost shifters $\subset$ wholesale $\subset$ Hausman); ``Demand'' instruments "
        r"form the $K=16$ vector $Z_{jt}$ of the logit/BLP moment conditions.",
        r"The demand leave-one-out (LOO) instruments are built within group "
        r"$g=(\text{MCA},\text{quarter})$: $\texttt{loo\_x}_j=\sum_{l\in g,\,l\neq j}x_l$ and "
        r"$\texttt{mean\_loo\_x}_j=\texttt{loo\_x}_j/n_{\mathrm{rivals},j}$.",
        r"The Hausman-style mean spread is built within deposit type and quarter over "
        r"conglomerate $\times$ market rows (MCA; national for D firms), leaving out only the "
        r"observation's own row.",
    )
    rows = _grouped_rows(_INSTR_MASTER_GROUPS, n_cols=3)
    return _master_longtable(
        caption    = "Instruments, by Estimation Stage",
        labels     = ["tab:instruments_master", "tab:operating_cost_shifters",
                      "tab:wholesale_funding", "tab:capital_adequacy",
                      "tab:blp_loo", "tab:blp_instruments"],
        col_spec   = (r">{\ttfamily\footnotesize}p{5.3cm} "
                      r">{\raggedright\arraybackslash}X "
                      r">{\raggedright\arraybackslash}p{3.4cm}"),
        header_row = (r"\normalfont\small\textbf{Instrument} & "
                      r"\textbf{Definition} & \textbf{Stage}"),
        n_cols     = 3,
        rows       = rows,
        notes      = notes,
    )


def _num_word(n: int) -> str:
    words = {11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
             16: "sixteen", 17: "seventeen", 18: "eighteen"}
    return words.get(n, str(n))


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def _write_tex(tex: str, basename: str, suffix: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (OUTPUT_DIR / f"{basename}{suffix}.tex",
                 DRAFTS_DIR / f"{basename}{suffix}.tex"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"  Wrote {path}")


def _write_csv(df: pd.DataFrame, basename: str, suffix: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_DIR / f"{basename}{suffix}.csv"
    df.to_csv(p, index=False)
    print(f"  Wrote {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compressed literature-style descriptive tables.")
    ap.add_argument("--weight-col", type=str, default=None,
                    help="Column to use for cross-sectional weighting (e.g., pop_total).")
    args = ap.parse_args()

    suffix = f"_weighted_by_{args.weight_col}" if args.weight_col else "_unweighted"
    df = load_panel()

    # --- Table 1 (always unweighted)
    print("\n[Table 1] Bank-conglomerate cross-section ...")
    mom_b, mom_d, mom_all, meta1 = build_table1(df, None)
    _write_csv(mom_b.assign(panel="B"),     "Compressed_BankType_CrossSection_B",   "_unweighted")
    _write_csv(mom_d.assign(panel="D"),     "Compressed_BankType_CrossSection_D",   "_unweighted")
    _write_csv(mom_all.assign(panel="All"), "Compressed_BankType_CrossSection_All", "_unweighted")
    tex1 = render_table1(mom_b, mom_d, mom_all, meta1, None, "_unweighted")
    _write_tex(tex1, "Compressed_BankType_CrossSection", "")

    # --- Table 2 (always deposit-weighted)
    print("\n[Table 2] MCA market structure by year ...")
    t2 = build_table2(df, "dep_total_mq")
    _write_csv(t2, "Compressed_MarketStructure_by_Year", "_dep_weighted")
    # Landscape version (kept for reference):
    # tex2 = render_table2(t2, "dep_total_mq", "_dep_weighted")
    # _write_tex(tex2, "Compressed_MarketStructure_by_Year", "")
    tex2a = render_table2a(t2, "dep_total_mq", "_dep_weighted")
    tex2b = render_table2b(t2, "dep_total_mq", "_dep_weighted")
    _write_tex(tex2a, "Compressed_MarketStructure_by_Year_AB", "")
    _write_tex(tex2b, "Compressed_MarketStructure_by_Year_C", "")
    render_market_structure_figure(t2, "")

    # --- Table 3
    print("\n[Table 3] Local environment by macro-region ...")
    latest_year = int(df["year"].max())
    t3 = build_table3(df, args.weight_col)
    _write_csv(t3, "Compressed_LocalEnvironment_by_Region", suffix)
    tex3 = render_table3(t3, args.weight_col, suffix, latest_year)
    _write_tex(tex3, "Compressed_LocalEnvironment_by_Region", "")

    # --- Table 4
    print("\n[Table 4] D-firm Pre/Post Pix ...")
    t4, meta4 = build_table4(df)
    _write_csv(t4, "Compressed_DFirm_PrePost_Pix", suffix)
    tex4 = render_table4(t4, meta4, suffix)
    _write_tex(tex4, "Compressed_DFirm_PrePost_Pix", "")

    # --- Appendix: Macro rates (B.3)
    print("\n[Appendix] Macro interest rates by year ...")
    app_rates = build_appendix_rates()
    tex_app   = render_appendix_rates(app_rates)
    _write_tex(tex_app, "Appendix_Macro_Rates_by_Year", "")

    # --- Appendix: consolidated variable dictionaries (master tables A & B)
    print("\n[Appendix] Consolidated variable dictionaries ...")
    _write_tex(render_variables_master(count_mcas(df)), "Appendix_Variables_Master", "")
    _write_tex(render_instruments_master(), "Appendix_Instruments_Master", "")

    print("\nDone.")


if __name__ == "__main__":
    main()
