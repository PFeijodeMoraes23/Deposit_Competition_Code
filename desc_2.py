"""
desc_2.py
=========
Compressed, literature-style descriptive tables built from
``market_panel.csv``.  Complements the exhaustive panel-style tables produced
by ``desc_1.py``.

Generates four compact tables, each saved to ``OUTPUT_DIR`` (CSV + TeX) and
mirrored to the V_Main draft directory, following the same naming convention
as ``desc_1.py`` (``_weighted_by_<col>`` or ``_unweighted`` suffix):

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
        branch density, broadband, PIX adoption.  Motivated by Joaquim & van
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
    python desc_2.py                          # unweighted variant
    python desc_2.py --weight-col pop_total   # population-weighted variant
"""
from __future__ import annotations

import argparse
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

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (mirror desc_1.py)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR    = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
PANEL_CSV        = DATA_DIR / "market_panel.csv"
MACRO_RATES_CSV  = DATA_DIR / "PANEL_INTERMED" / "quarterly_macro_rates.csv"
OUTPUT_DIR  = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"
DRAFTS_DIR  = Path(r"C:\Users\pedro\OneDrive\Documentos\Yale"
                   r"\Year 3 (2024 - 2025)\Open Finance\Open-Finance"
                   r"\Drafts\Deposit Competition")

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
# Display formatting (consistent with desc_1.py)
# ---------------------------------------------------------------------------
# (label, unit string, scale divisor or None, decimals)
LABEL_MAP = {
    "total_deposits":            ("Total Deposits", r"R\$M",        1e6,  2),
    "total_assets":              ("Total Assets",            r"R\$B",        1e9,  2),
    "log_total_assets":          ("Log Total Assets",        "",             None, 3),
    "log_dep":                   ("Log Deposits",  "",             None, 3),
    "equity_ratio":              ("Equity Ratio",            "",             None, 3),
    "dep_a1":                    ("Deposits (1)",            r"R\$M",        1e6,  2),
    "dep_a2":                    ("Deposits (2)",            r"R\$M",        1e6,  2),
    "dep_a4":                    ("Deposits (4)",            r"R\$M",        1e6,  2),
    "dep_a5":                    ("Deposits (5)",            r"R\$M",        1e6,  2),
    "log_dep_a4":                ("Log Deposits (4)",        "",             None, 3),
    "spread_a4":                 ("Spread (4)",              "bp",           0.01, 2),
    "spread_a5":                 ("Spread (5)",              "bp",           0.01, 2),
    "n_mcas_served":             ("MCAs Served",             "count",        None, 1),
    # Table 2 specific
    "n_b_firms":                 ("Number of B Firms",       "count",        None, 2),
    "hhi_b":                     ("HHI (B firms)",           "",              None, 0),
    "hhi_d_natl":                ("HHI (D firms, national)", "",              None, 0),
    "hhi_combined_natl":         ("HHI (B+D, national)",     "",              None, 0),
    "spread_a4_d_w":              ("Spread (4), D firms",      "bp",           0.01, 2),
    "spread_a5_d_w":              ("Spread (5), D firms",      "bp",           0.01, 2),
    "spread_a4_natl_w":           ("Spread (4), National (B+D)","bp",          0.01, 2),
    "spread_a5_natl_w":           ("Spread (5), National (B+D)","bp",          0.01, 2),
    "risk_free_qoq":              ("SELIC (risk-free)",         r"\% qoq",      0.01, 2),
    "spread_a4_w":               ("Spread (4), B firms",     "bp",           0.01, 2),
    "spread_a5_w":               ("Spread (5), B firms",     "bp",           0.01, 2),
    # Annualised spread variants (for Tables 1, 2b, 4)
    "spread_ann_a4":             ("Spread (4)",              r"\% p.a.",     0.01, 2),
    "spread_ann_a5":             ("Spread (5)",              r"\% p.a.",     0.01, 2),
    "spread_ann_a4_w":           ("Spread (4), B firms",     r"\% p.a.",     0.01, 2),
    "spread_ann_a5_w":           ("Spread (5), B firms",     r"\% p.a.",     0.01, 2),
    "spread_ann_a4_d_w":         ("Spread (4), D firms",     r"\% p.a.",     0.01, 2),
    "spread_ann_a5_d_w":         ("Spread (5), D firms",     r"\% p.a.",     0.01, 2),
    "spread_ann_a4_natl_w":      ("Spread (4), National (B+D)", r"\% p.a.", 0.01, 2),
    "spread_ann_a5_natl_w":      ("Spread (5), National (B+D)", r"\% p.a.", 0.01, 2),
    "risk_free_ann":             ("SELIC (risk-free)",        r"\% p.a.",     0.01, 2),
    "n_d_firms_natl":            ("Number of D Firms (nat.)", "count",       None, 0),
    # Table 3 specific
    "pop_total":                 ("Population",              "Thousands",    1e3,  1),
    "gdp_per_capita":            ("GDP per Capita",          r"R\$",         None, 0),
    "fraction_65plus":           ("Share Aged 65+",          "",             None, 3),
    "cadunico_families_per1000": (r"Cad\'Unico Families",    "per 1{,}000",  None, 2),
    "branches_per1000":          ("Bank Branches",           "per 1{,}000",  None, 3),
    "connections_per100":        ("Internet Connections",    "per 100",      None, 2),
    "pix_users_pf_per1000":      ("PIX Users (PF)",          "per 1{,}000",  None, 2),
    "pix_txns_pf":               ("PIX Transactions (PF)",   "Millions",     1e6,  2),
}


def _esc(s) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _fmt_val(val, var_base):
    if pd.isna(val):
        return "--"
    info = LABEL_MAP.get(var_base)
    if info is None:
        return f"{val:.3f}"
    _lbl, _unit, scale, dec = info
    v = val / scale if scale is not None else val
    if dec == 0:
        return f"{v:,.0f}"
    fmt = f"{{:,.{dec}f}}"
    return fmt.format(v)


def _row_label(var_base) -> str:
    info = LABEL_MAP.get(var_base)
    if info is None:
        return _esc(var_base)
    lbl, unit, _, _ = info
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
def load_panel() -> pd.DataFrame:
    print(f"Loading {PANEL_CSV.name} ...")
    df = pd.read_csv(PANEL_CSV, low_memory=False)
    df["CODMUN_IBGE_str"] = df["CODMUN_IBGE"].astype(str).str.split(".").str[0]
    df["bank_type"] = np.where(df["CODMUN_IBGE_str"] == "0", "D", "B")
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
                n_mcas_served=("CODMUN_IBGE", "nunique"),
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
    # Annualise deposit-weighted spreads: spread_ann = rf_ann - dep_rate_ann
    _rf_ann = (1 + out["risk_free_qoq"]) ** 4 - 1
    for _a in ("a4", "a5"):
        _dep_qoq = out["risk_free_qoq"] - out[f"spread_{_a}"] * 0.01
        out[f"spread_ann_{_a}"] = _rf_ann - ((1 + _dep_qoq) ** 4 - 1)
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
    _rf_ann = (1 + out["risk_free_qoq"]) ** 4 - 1
    for _a in ("a4", "a5"):
        _dep_qoq = out["risk_free_qoq"] - out[f"spread_{_a}"] * 0.01
        out[f"spread_ann_{_a}"] = _rf_ann - ((1 + _dep_qoq) ** 4 - 1)
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

    # Combined (pooled) panel for Panel C — variables common to both types
    T1_VARS_COMMON = T1_VARS_D   # n_mcas_served is B-type only; all others shared
    cols_keep = (["CodConglomeradoPrudencial", "year", "quarter"]
                 + T1_VARS_COMMON
                 + ([weight_col] if weight_col and weight_col in fq_b.columns else []))
    fq_all = pd.concat([
        fq_b[[c for c in cols_keep if c in fq_b.columns]],
        fq_d[[c for c in cols_keep if c in fq_d.columns]],
    ], ignore_index=True)
    moments_all = _moments(fq_all, T1_VARS_COMMON)

    meta = {
        "n_firm_quarters_B":   int(len(fq_b)),
        "n_firm_quarters_D":   int(len(fq_d)),
        "n_firms_B":           int(fq_b["CodConglomeradoPrudencial"].nunique()),
        "n_firms_D":           int(fq_d["CodConglomeradoPrudencial"].nunique()),
        "n_firm_quarters_all": int(len(fq_all)),
        "n_firms_all":         int(fq_all["CodConglomeradoPrudencial"].nunique()),
    }
    return moments_b, moments_d, moments_all, meta


def render_table1(moments_b, moments_d, moments_all, meta, weight_col, suffix) -> str:
    name      = "Compressed_BankType_CrossSection"
    tab_label = f"tab:{name}"
    weight_lbl = "(Population Weighted)" if weight_col else ""
    caption   = (f"Bank-Conglomerate Cross-Section by Type{(' ' + weight_lbl) if weight_lbl else ''}")

    col_spec  = "l@{\\hspace{0.5em}}rrrrrrrr"
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
    body += _panel("Panel C: All Firms (Pooled)",
                   moments_all, meta["n_firms_all"], meta["n_firm_quarters_all"])

    tex = "\n".join([
        r"\setstretch{1.0}",
        r"\setlength{\LTleft}{\fill}",
        r"\setlength{\LTright}{\fill}",
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
        r"\par\noindent{\footnotesize \textit{Notes:} Spreads are annualised: $(1 + r_{\text{qoq}})^4 - 1$. "
        r"Deposit spreads are defined as the risk-free rate minus the offered deposit rate.}",
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
    grp_keys = ["CODMUN_IBGE", "year", "quarter"]

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

    # Step 4: annualise spread at row level, then deposit-weight
    df_b["_rf_ann"] = (1 + df_b["risk_free_qoq"]) ** 4 - 1
    for _a in ("a4", "a5"):
        _dep = df_b["risk_free_qoq"] - df_b[f"spread_{_a}"] * 0.01
        df_b[f"spread_ann_{_a}"] = df_b["_rf_ann"] - ((1 + _dep) ** 4 - 1)

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

    # D-firm annualised deposit-weighted spreads by year
    work_d = df_d[["year", "spread_a4", "spread_a5", "dep_a4", "dep_a5", "risk_free_qoq"]].copy()
    work_d["_rf_ann"] = (1 + work_d["risk_free_qoq"]) ** 4 - 1
    for _a in ("a4", "a5"):
        _dep = work_d["risk_free_qoq"] - work_d[f"spread_{_a}"] * 0.01
        work_d[f"spread_ann_{_a}"] = work_d["_rf_ann"] - ((1 + _dep) ** 4 - 1)
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

    # National (B+D) annualised deposit-weighted spreads by year
    work_all = pd.concat([
        df_b[["year", "spread_a4", "spread_a5", "dep_a4", "dep_a5", "risk_free_qoq", "_rf_ann"]],
        work_d[["year", "spread_a4", "spread_a5", "dep_a4", "dep_a5", "risk_free_qoq", "_rf_ann"]],
    ], ignore_index=True)
    for _a in ("a4", "a5"):
        _dep = work_all["risk_free_qoq"] - work_all[f"spread_{_a}"] * 0.01
        work_all[f"spread_ann_{_a}"] = work_all["_rf_ann"] - ((1 + _dep) ** 4 - 1)
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

    # Panel C: Risk-free rate (annualised) then annualised deposit-weighted spreads
    rf_yr = (df.groupby(["year", "quarter"])["risk_free_qoq"].first()
               .groupby("year").mean())
    rf_ann_yr = (1 + rf_yr) ** 4 - 1
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
        ("Panel C: Market Spreads (Deposit-Weighted)",  ["spread_a4_w", "spread_a5_w",
                                                          "spread_a4_d_w", "spread_a5_d_w",
                                                          "spread_a4_natl_w", "spread_a5_natl_w"]),
    ]
    PRE2020_BLANK = {"spread_a5_w", "spread_a5_d_w", "spread_a5_natl_w"}
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
    """Market structure (B-firm MCA counts + D-firm national counts) — years as rows."""
    name      = "Compressed_MarketStructure_by_Year_AB"
    tab_label = f"tab:{name}"
    weight_lbl = ("(Pop.\\ Weighted)" if weight_col == "pop_total"
                  else "(Dep.\\ Weighted)" if weight_col == "dep_total_mq"
                  else "")
    caption = ("MCA-Level Market Structure by Year"
               + (f" {weight_lbl}" if weight_lbl else ""))

    year_cols  = [c for c in t2_df.columns if c != "var"]
    t2_idx     = t2_df.set_index("var")
    n_row_data = t2_df[t2_df["var"] == "N_obs"]

    VARS_A  = ["n_b_firms", "hhi_b"]
    VARS_B  = ["n_d_firms_natl", "hhi_combined_natl"]
    VARS_AB = VARS_A + VARS_B

    short_labels = {
        "n_b_firms":         r"No.\ B Firms",
        "hhi_b":             r"HHI (B)",
        "n_d_firms_natl":    r"No.\ D Firms",
        "hhi_combined_natl": r"HHI (B+D)",
    }
    sub_units = {
        "n_b_firms":         r"(count)",
        "hhi_b":             r"($0$--$10{,}000$)",
        "n_d_firms_natl":    r"(count)",
        "hhi_combined_natl": r"($0$--$10{,}000$)",
    }

    col_spec   = "l@{\\hspace{1.2em}}" + "rr@{\\hspace{1.2em}}" + "rr@{\\hspace{1.2em}}" + "r"
    header_top = (
        r" & \multicolumn{2}{c@{\hspace{1.2em}}}{\textit{B Firms (MCA-Level)}}"
        r" & \multicolumn{2}{c@{\hspace{1.2em}}}{\textit{D Firms (National)}} & \\"
    )
    cmidrules  = r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}"
    col_labels = ("Year & "
                  + " & ".join(short_labels[v] for v in VARS_AB)
                  + r" & MCA-Qtrs.\ ($N$) \\")
    col_units  = (" & "
                  + " & ".join(sub_units[v] for v in VARS_AB)
                  + r" & \\")

    body = []
    for yr in year_cols:
        cells = [str(yr)]
        for v in VARS_AB:
            val = t2_idx.loc[v, yr] if v in t2_idx.index else np.nan
            cells.append(_fmt_val(val, v))
        cells.append(f"{int(n_row_data.iloc[0][yr]):,}" if len(n_row_data) else "--")
        body.append(" & ".join(cells) + r" \\")

    tex = "\n".join([
        r"\begin{table}[htbp]",
        r"\setstretch{1.0}",
        r"\centering",
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
    col_units  = r" & (\% p.a.) & (\% p.a.) & (\% p.a.) & (\% p.a.) & (\% p.a.) & (\% p.a.) & (\% p.a.) \\"

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
        r"\item \scriptsize \textit{Notes:} All spreads and the SELIC rate are annualised: "
        r"$(1 + r_{\text{qoq}})^4 - 1$. "
        r"Spread\,(5) is undefined before 2020 and shown as blank. "
        r"Deposit spreads are defined as the risk-free rate minus the offered deposit rate.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        r"\doublespacing",
    ])
    return tex


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
    mca = snap.groupby("CODMUN_IBGE", sort=False).agg(
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
        r"\scriptsize \textit{Notes:} Each cell is the cross-MCA mean within "
        r"the macro-region, using only B-firm coverage to identify local "
        r"markets. MCA-level values are obtained by averaging across firms "
        r"and quarters within the latest sample year "
        f"({latest_year}). "
        + ("Cross-MCA averages are population-weighted using "
           r"\texttt{pop\_total}."
           if weight_col else "Cross-MCA averages are unweighted.")
    )

    tex = "\n".join([
        r"\begin{table}[H]",
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
T4_VARS = ["log_total_assets", "log_dep", "log_dep_a4", "spread_ann_a4", "equity_ratio"]


def build_table4(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_d = df[df["bank_type"] == "D"].copy()
    df_d = df_d[["CodConglomeradoPrudencial", "year", "quarter", "post",
                 "total_assets", "equity_ratio", "dep_a4", "spread_a4",
                 "risk_free_qoq", "total_deposits"]].copy()
    df_d["log_total_assets"] = np.log(df_d["total_assets"].where(df_d["total_assets"] > 0))
    df_d["log_dep"]       = np.log(df_d["total_deposits"].where(df_d["total_deposits"] > 0))
    df_d["log_dep_a4"]       = np.log(df_d["dep_a4"].where(df_d["dep_a4"] > 0))
    # Annualise spread: rf_ann - dep_rate_ann
    _rf_ann = (1 + df_d["risk_free_qoq"]) ** 4 - 1
    _dep_qoq = df_d["risk_free_qoq"] - df_d["spread_a4"] * 0.01
    df_d["spread_ann_a4"] = _rf_ann - ((1 + _dep_qoq) ** 4 - 1)

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
    }
    return res_df, meta


def render_table4(t4_df, meta, suffix) -> str:
    name      = "Compressed_DFirm_PrePost_Pix"
    tab_label = f"tab:{name}"
    caption   = ("Digital (D) Firms Before and After Pix: Difference-in-Means "
                 "with Conglomerate-Clustered Inference")

    col_spec = "l@{\\hspace{0.4em}}ccccc"
    header   = (r"Variable & \multicolumn{1}{c}{Pre Mean} & \multicolumn{1}{c}{Post Mean}"
                r" & \multicolumn{1}{c}{$\Delta$} & \multicolumn{1}{c}{$G$}"
                r" & \multicolumn{1}{c}{$G^{\star}$} \\")

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
            f"{int(r['G'])} & "
            f"{r['G_star']:.1f} \\\\"
        )
        body.append(
            f" & & & {se_cell} & & \\\\[0.3em]"
        )

    # Context row: active D-firm counts (no test).
    body.append(r"\addlinespace[0.4em]")
    body.append(r"\midrule")
    body.append(
        f"Active D Conglomerates "
        f"& {meta['n_d_firms_pre']:,} "
        f"& {meta['n_d_firms_post']:,} "
        f"& {meta['n_d_firms_post'] - meta['n_d_firms_pre']:+,} "
        r"& \multicolumn{2}{c}{\textit{(count, no test)}} \\"
    )

    notes = (
        r"\scriptsize \textit{Notes:} The PIX threshold is "
        r"$t \geq 2020\mathrm{Q}4$, the activation of the Brazilian instant "
        r"payment system on Nov.\ 16, 2020. Type-5 (prepaid) outcomes are not "
        r"reported because they are zero by construction prior to 2020Q4. "
        r"Deposit spreads are annualised: $(1 + r_{\text{qoq}})^4 - 1$. "
        r"Deposit spreads are defined as the risk-free rate minus the offered deposit rate. "
        r"Stars: $^{*}\,p<0.10$, $^{**}\,p<0.05$, $^{***}\,p<0.01$."
    )

    tex = "\n".join([
        r"\begin{table}[H]",
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
# Appendix: Macro interest rates by year
# ---------------------------------------------------------------------------
def build_appendix_rates() -> pd.DataFrame:
    """Annual averages of macro interest rates from quarterly_macro_rates.csv.
    Quarterly compound rates are annualised as (1 + r_qoq/100)^4 - 1."""
    df = pd.read_csv(MACRO_RATES_CSV)
    df["year"]    = df["AnoMes"] // 100
    df["quarter"] = df["AnoMes"] % 100

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
    """Portrait table of annualised macro rates by year."""
    name      = "Appendix_Macro_Rates_by_Year"
    tab_label = f"tab:{name}"
    caption   = "Brazilian Macro Interest Rates by Year"

    # cols: Year | SELIC | CDI | Savings | COPOM target
    # col positions: 1=Year, 2-4=market rates (cmidrule), 5=COPOM target
    col_spec   = "l@{\\hspace{1.2em}}rrr@{\\hspace{1.2em}}r"
    header_top = (
        r" & \multicolumn{3}{c@{\hspace{1.2em}}}{\textit{Market Rates (annualised)}} & \\"
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
        r"\scriptsize \textit{Notes:} "
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
# Appendix B: variable-description longtables (content extracted from V_Main.tex)
# These are not data-driven; they are hardcoded LaTeX matching the inline tables.
# ---------------------------------------------------------------------------
def _longtable(caption: str, label: str, col_spec: str,
               header_row: str, rows: list[str],
               cont_caption: str = "") -> str:
    """Standard variable-description longtable skeleton."""
    n_cols = col_spec.count("p{") + col_spec.count(">") + col_spec.count("l") \
             + col_spec.count("r") + col_spec.count("c")
    cont = cont_caption or f"{caption} (Continued)"
    body = "\n".join(rows)
    return "\n".join([
        r"\setstretch{1.0}",
        r"\setlength{\LTleft}{\fill}",
        r"\setlength{\LTright}{\fill}",
        f"\\begin{{longtable}}{{{col_spec}}}",
        f"    \\caption{{{caption}}} \\label{{{label}}} \\\\",
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
        f"    \\multicolumn{{2}}{{r}}{{\\textit{{Continued on next page}}}} \\\\",
        r"    \endfoot",
        r"    \bottomrule",
        r"    \endlastfoot",
        body,
        r"\end{longtable}",
        r"\doublespacing",
    ])


def render_appendix_b1() -> str:
    rows = [
        r"    CodConglomeradoPrudencial & Conglomerate $j$ (prudential C-code) \\",
        r"    CNPJ\_Lider               & Lead institution CNPJ of the conglomerate \\",
        r"    CNPJ                     & Institution $j$ at root-CNPJ level \\",
        r"    CODMUN\_IBGE             & 7-digit IBGE municipality code; \texttt{0} = nationally active D institution \\",
        r"    mca\_code                & Market $m$ --- one of 468 MCAs, or \texttt{NATIONAL} for D institutions \\",
        r"    year, quarter            & Period $t$ label \\",
        r"    AnoMes                   & BCB IF-Data period code YYYYMM \\",
        r"    deposit\_type            & Deposit category $k$: 1=demand, 2=savings, 3=interbank, 4=time/CDB, 5=prepaid \\",
    ]
    return _longtable(
        caption  = "Identifiers / indices",
        label    = "tab:identifiers",
        col_spec = r">{\ttfamily\small}p{6.0cm} p{9.5cm}",
        header_row = r"\normalfont\textbf{Variable} & \textbf{Role}",
        rows     = rows,
    )


def render_appendix_b2() -> str:
    rows = [
        r"    dep\_a1--dep\_a5          & Deposit balances by type from ESTBAN and IF-Data, aggregated to MCA level if $\mathrm{B}$ firm and national level if $\mathrm{D}$ firm \\",
        r"    deposit\_balance         & Type-specific deposit stock from IF-Data Passivo report \\",
        r"    lagged\_deposits         & \texttt{deposit\_balance} shifted 1 quarter within institution and type \\",
        r"    total\_deposits          & Sum across all 5 types of deposits for institution $j$ at period $t$ \\",
        r"    lagged\_total\_deposits   & \texttt{total\_deposits} shifted 1 quarter \\",
        r"    pop\_total               & Proxy for market size $M_{mt}$ from IBGE population \\",
    ]
    return _longtable(
        caption  = r"Outcome Variables --- Deposit Stocks $\mathrm{Dep}_{jkmt}$",
        label    = "tab:deposits",
        col_spec = r">{\ttfamily}p{6.0cm} p{10.0cm}",
        header_row = r"\normalfont\textbf{Variable} & \textbf{Role}",
        rows     = rows,
    )


def render_appendix_b3_vars() -> str:
    """New B.3: rate and spread variable descriptions."""
    rows = [
        r"    \multicolumn{2}{l}{\textit{Benchmark macro rates (from BCB SGS)}} \\",
        r"    \midrule",
        r"    selic\_qoq           & SELIC overnight rate compounded QoQ: $(1+r_{\text{daily}})^{63}-1$. Risk-free benchmark. \\",
        r"    cdi\_qoq             & CDI (interbank) rate compounded QoQ. \\",
        r"    savings\_rate\_qoq   & Regulated savings deposit rate, compounded QoQ. \\",
        r"    meta\_selic          & COPOM Selic target rate (\% p.a.) at end of quarter. \\",
        r"    \addlinespace[0.4em]",
        r"    \multicolumn{2}{l}{\textit{Institution-level rates and spreads}} \\",
        r"    \midrule",
        r"    risk\_free\_qoq      & Risk-free rate: equals \texttt{selic\_qoq}. \\",
        r"    deposit\_rate\_qoq   & Type-specific deposit rate, QoQ decimal. Varies by type $k$ (see text). \\",
        r"    spread\_qoq         & $r^{\mathrm{rf}}_t - r^{\mathrm{dep}}_{jkmt}$, QoQ decimal. Used in sleepiness estimation and deposit dynamics. \\",
        r"    \addlinespace[0.4em]",
        r"    \multicolumn{2}{l}{\textit{Annualised counterparts (for demand estimation)}} \\",
        r"    \midrule",
        r"    risk\_free\_ann      & $(1+\texttt{risk\_free\_qoq})^4 - 1$. \\",
        r"    deposit\_rate\_ann   & $(1+\texttt{deposit\_rate\_qoq})^4 - 1$. \\",
        r"    spread\_ann         & $r^{\mathrm{rf,ann}}_t - r^{\mathrm{dep,ann}}_{jkmt}$. Price variable in demand estimation. \\",
    ]
    return _longtable(
        caption    = r"Rates and Spreads",
        label      = "tab:rates_spreads",
        col_spec   = r">{\ttfamily\small}p{5.5cm} p{10.0cm}",
        header_row = r"\normalfont\textbf{Variable} & \textbf{Description}",
        rows       = rows,
    )


def render_appendix_b4() -> str:
    rows = [
        r"    gdp\_per\_capita           & Municipal GDP from IBGE, interpolated and imputed where needed \\",
        r"    gdp\_imputed              & Flag for when GDP was imputed rather than directly observed (2024 only) \\",
        r"    fraction\_65plus          & Share of population aged 65+ \\",
        r"    fraction\_young           & Share of population aged 15--20 \\",
        r"    age\_interpolated         & Flag for when age structure was interpolated \\",
        r"    pix\_exists               & Binary indicator for PIX available in the quarter \\",
        r"    pix\_users\_pf\_per1000   & Number of PIX users per 1,000 inhabitants \\",
        r"    pix\_txns\_pf             & Number of PIX transactions by individuals \\",
        r"    pix\_value\_pf\_r1000     & PIX transaction value by individuals \\",
        r"    pix\_users\_pj            & PIX business users \\",
        r"    connections\_per100       & Broadband subscriptions per 100 inhabitants \\",
        r"    frac\_4g5g                & Share of mobile connections that are 4G or 5G \\",
        r"    branches\_per1000         & Bank branches per 1,000 inhabitants \\",
        r"    access\_points\_per1000    & Total access points per 1,000 inhabitants \\",
        r"    cadunico\_families\_per1000 & Registered low-income families per 1,000 inhabitants \\",
        r"    cadunico\_extreme\_poverty & Share in extreme poverty \\",
        r"    cadunico\_poverty         & Share in poverty \\",
    ]
    return _longtable(
        caption  = r"$S_{mt}$ --- Market-level state variables entering the sleepiness function $\phi(S_{mt})$ and the auxiliary demand model",
        label    = "tab:state_variables_app",
        col_spec = r">{\ttfamily\small}p{6.0cm} p{9.5cm}",
        header_row = r"\normalfont\textbf{Variable} & \textbf{Role}",
        rows     = rows,
    )


def render_appendix_b5() -> str:
    rows = [
        r"    equity\_ratio             & Capital structure; basis for rival IVs --- \texttt{equity}/\texttt{total\_assets} \\",
        r"    log\_total\_assets         & Size shifter; basis for rival IVs --- $\ln(\text{total\_assets})$ \\",
        r"    fgc\_covered              & Insurance --- indicator for FGC-insured deposit types (1, 2, and 4) \\",
        r"    has\_ip                   & IP subsidiary flag (see Table \ref{tab:identifiers}) \\",
    ]
    return _longtable(
        caption  = r"$X_{jt}$ --- Institution-level product characteristics",
        label    = "tab:institution_characteristics",
        col_spec = r">{\ttfamily\small}p{6.0cm} p{9.5cm}",
        header_row = r"\normalfont\textbf{Variable} & \textbf{Role}",
        rows     = rows,
    )


def render_appendix_b6() -> str:
    rows = [
        r"    personnel\_cost\_ratio\_lag    & From COSIF 78215 --- $|\text{personnel}|/\text{assets}$, lag 1Q. IV Logic: wage costs shift marginal cost \\",
        r"    admin\_cost\_ratio\_lag        & From COSIF 78214 --- $|\text{admin}|/\text{assets}$, lag 1Q. IV Logic: general overhead cost shifter \\",
        r"    tax\_cost\_ratio\_lag          & From COSIF 78220 --- $|\text{tax}|/\text{assets}$, lag 1Q. IV Logic: tax burden cost shifter \\",
    ]
    return _longtable(
        caption  = r"$Z_{jt}$ --- Operating Cost Shifters from IF-Data DRE Report",
        label    = "tab:operating_cost_shifters",
        col_spec = r">{\ttfamily\small}p{6.0cm} p{9.5cm}",
        header_row = r"\normalfont\textbf{Variable} & \textbf{Role}",
        rows     = rows,
    )


def render_appendix_b7() -> str:
    rows = [
        r"    lci\_lca\_ratio\_lag      & COSIF LCI (78289) + LCA (78290) / \text{assets}. IV: alternative funding source pressure \\",
        r"    wholesale\_ratio\_lag     & Sum of wholesale accounts / \text{assets}. IV: aggregate wholesale dependence \\",
    ]
    return _longtable(
        caption  = r"$Z_{jt}$ --- Wholesale Funding Mix Cost Instruments from IF-Data Passivo report",
        label    = "tab:wholesale_funding",
        col_spec = r">{\ttfamily\small}p{6.0cm} p{9.5cm}",
        header_row = r"\normalfont\textbf{Variable} & \textbf{Role}",
        rows     = rows,
    )


def render_appendix_b8() -> str:
    rows = [
        r"    indice\_basileia\_lag          & From COSIF 79664 --- Constructed as total capital ratio (Basel Index), lagged 1Q \\",
    ]
    return _longtable(
        caption  = r"$Z_{jt}$ --- Capital adequacy cost instruments from IF-Data Capital report",
        label    = "tab:capital_adequacy",
        col_spec = r">{\ttfamily\small}p{6.0cm} p{9.5cm}",
        header_row = r"\normalfont\textbf{Variable} & \textbf{Role}",
        rows     = rows,
    )


def render_appendix_b9() -> str:
    rows = [
        r"    loo\_log\_assets / mean\_loo\_log\_assets       & log\_total\_assets         & Rival scale instrument \\",
        r"    loo\_equity\_assets / mean\_loo\_equity\_ratio  & equity\_ratio              & Rival capital structure instrument \\",
        r"    loo\_indice\_basileia / mean\_loo\_basileia     & indice\_basileia\_lag      & Rival capital slack \\",
        r"    loo\_credit\_assets / mean\_loo\_credit\_assets  & credit\_assets\_ratio\_lag  & Rival ALM pressure \\",
        r"    loo\_lci\_lca\_ratio / mean\_loo\_lci\_lca\_ratio & lci\_lca\_ratio\_lag       & Rival wholesale substitutability \\",
        r"    loo\_wholesale\_ratio / mean\_loo\_wholesale    & wholesale\_ratio\_lag      & Rival aggregate wholesale mix \\",
        r"    loo\_npl\_provision / mean\_loo\_npl\_provision & npl\_provision\_ratio\_lag & Rival credit quality \\",
        r"    n\_rivals                                     & --                          & Count of non-null rivals in the same group \\",
    ]
    # 3-column table — override the 2-col multicolumn in footer
    n_cols = 3
    cont = r"$Z_{jt}$ --- BLP leave-one-out rival instruments (Continued)"
    body  = "\n".join(rows)
    return "\n".join([
        r"\setstretch{1.0}",
        r"\setlength{\LTleft}{\fill}",
        r"\setlength{\LTright}{\fill}",
        r"\begin{longtable}{>{\ttfamily\small}p{5.5cm} >{\ttfamily\small}p{4.5cm} p{5.5cm}}",
        r"    \caption{$Z_{jt}$ --- BLP leave-one-out rival instruments} \label{tab:blp_loo} \\",
        r"    \toprule",
        r"    \normalfont\textbf{Variable Pair} & \normalfont\textbf{Source Characteristic} & \textbf{Logic} \\",
        r"    \midrule",
        r"    \endfirsthead",
        f"    \\caption[]{{{cont}}} \\\\",
        r"    \toprule",
        r"    \normalfont\textbf{Variable Pair} & \normalfont\textbf{Source Characteristic} & \textbf{Logic} \\",
        r"    \midrule",
        r"    \endhead",
        r"    \midrule",
        r"    \multicolumn{3}{r}{\textit{Continued on next page}} \\",
        r"    \endfoot",
        r"    \bottomrule",
        r"    \endlastfoot",
        body,
        r"\end{longtable}",
        r"\doublespacing",
    ])


def render_appendix_b10() -> str:
    """Rates and spreads variable descriptions — new table."""
    rows = [
        r"    selic\_qoq           & SELIC overnight rate, compounded QoQ; $\approx (1+r_{\text{daily}})^{63}-1$. Source: BCB SGS. \\",
        r"    cdi\_qoq             & CDI (interbank) rate, compounded QoQ. Source: BCB SGS. \\",
        r"    savings\_rate\_qoq   & Regulated savings rate, compounded QoQ. Source: BCB SGS. \\",
        r"    meta\_selic          & COPOM Selic target rate (\% p.a.) at end of quarter. Source: BCB SGS. \\",
        r"    risk\_free\_qoq      & Risk-free rate used in spread construction; equals \texttt{selic\_qoq}. \\",
        r"    deposit\_rate\_qoq   & Type-specific deposit rate, QoQ decimal. Varies by type (see text). \\",
        r"    spread\_qoq         & $r^{\text{rf}}_t - r^{\text{dep}}_{jkmt}$, QoQ decimal. Used in sleepiness estimation. \\",
        r"    risk\_free\_ann      & $(1+\texttt{risk\_free\_qoq})^4 - 1$; annualised SELIC. \\",
        r"    deposit\_rate\_ann   & $(1+\texttt{deposit\_rate\_qoq})^4 - 1$; annualised deposit rate. \\",
        r"    spread\_ann         & $r^{\text{rf,ann}}_t - r^{\text{dep,ann}}_{jkmt}$, annualised. Used in demand estimation. \\",
    ]
    return _longtable(
        caption  = r"Rates and Spreads",
        label    = "tab:rates_spreads",
        col_spec = r">{\ttfamily\small}p{5.5cm} p{10.0cm}",
        header_row = r"\normalfont\textbf{Variable} & \textbf{Description}",
        rows     = rows,
    )


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

    # --- Appendix B: variable description tables (B.1, B.2, B.4–B.10)
    print("\n[Appendix B] Variable description tables ...")
    for fname, render_fn in [
        ("Appendix_B_Identifiers",       render_appendix_b1),
        ("Appendix_B_Deposits",          render_appendix_b2),
        ("Appendix_B_RatesVars",         render_appendix_b3_vars),   # new B.3
        ("Appendix_B_StateVariables",    render_appendix_b4),
        ("Appendix_B_InstitutionChars",  render_appendix_b5),
        ("Appendix_B_CostShifters",      render_appendix_b6),
        ("Appendix_B_WholesaleFunding",  render_appendix_b7),
        ("Appendix_B_CapitalAdequacy",   render_appendix_b8),
        ("Appendix_B_BLPloo",            render_appendix_b9),
    ]:
        _write_tex(render_fn(), fname, "")

    print("\nDone.")


if __name__ == "__main__":
    main()
