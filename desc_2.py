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
PANEL_CSV   = DATA_DIR / "market_panel.csv"
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
    "total_assets":              ("Total Assets",            r"R\$B",        1e9,  2),
    "log_total_assets":          ("Log Total Assets",        "",             None, 3),
    "equity_ratio":              ("Equity Ratio",            "",             None, 3),
    "dep_a1":                    ("Deposits (1)",            r"R\$M",        1e6,  2),
    "dep_a2":                    ("Deposits (2)",            r"R\$M",        1e6,  2),
    "dep_a4":                    ("Deposits (4)",            r"R\$M",        1e6,  2),
    "dep_a5":                    ("Deposits (5)",            r"R\$M",        1e6,  2),
    "log_dep_a4":                ("Log Deposits (4)",        "",             None, 3),
    "spread_a4":                 ("Spread (4)",              "pp",           None, 4),
    "spread_a5":                 ("Spread (5)",              "pp",           None, 4),
    "n_mcas_served":             ("MCAs Served",             "count",        None, 1),
    # Table 2 specific
    "n_b_firms":                 ("Number of B Firms",       "count",        None, 2),
    "hhi_b":                     ("HHI (B firms)",           "x10{,}000",    None, 0),
    "spread_a4_w":               ("Spread (4), dep-weighted","pp",           None, 4),
    "spread_a5_w":               ("Spread (5), dep-weighted","pp",           None, 4),
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
             "dep_a1", "dep_a2", "dep_a4", "dep_a5",
             "spread_a4", "spread_a5", "n_mcas_served"]
T1_VARS_D = ["total_assets", "equity_ratio",
             "dep_a1", "dep_a2", "dep_a4", "dep_a5",
             "spread_a4", "spread_a5"]


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
            )
            .reset_index()
    )

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
    return out


def _build_firm_quarter_D(df_d: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["CodConglomeradoPrudencial", "year", "quarter",
                 "total_assets", "equity_ratio",
                 "dep_a1", "dep_a2", "dep_a4", "dep_a5",
                 "spread_a4", "spread_a5"]
    return df_d[keep_cols].copy()


def build_table1(df: pd.DataFrame, weight_col: str | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df_b = df[df["bank_type"] == "B"].copy()
    df_d = df[df["bank_type"] == "D"].copy()

    fq_b = _build_firm_quarter_B(df_b)
    fq_d = _build_firm_quarter_D(df_d)

    # Population weights at firm-quarter level: B uses pop sum across MCAs;
    # D doesn't aggregate over MCAs (national row already weighted upstream).
    if weight_col is not None:
        pop_b = (df_b.groupby(["CodConglomeradoPrudencial", "year", "quarter"],
                              sort=False)[weight_col].sum().reset_index())
        fq_b  = fq_b.merge(pop_b, on=["CodConglomeradoPrudencial", "year", "quarter"], how="left")
        if weight_col in df_d.columns:
            fq_d = fq_d.merge(df_d[["CodConglomeradoPrudencial", "year", "quarter", weight_col]],
                              on=["CodConglomeradoPrudencial", "year", "quarter"], how="left")

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
                "p50":  _weighted_quantile(x, 0.50, w),
                "p90":  _weighted_quantile(x, 0.90, w),
                "N":    int(np.isfinite(x).sum()),
            })
        return pd.DataFrame(rows)

    moments_b = _moments(fq_b, T1_VARS_B)
    moments_d = _moments(fq_d, T1_VARS_D)

    meta = {
        "n_firm_quarters_B": int(len(fq_b)),
        "n_firm_quarters_D": int(len(fq_d)),
        "n_firms_B":         int(fq_b["CodConglomeradoPrudencial"].nunique()),
        "n_firms_D":         int(fq_d["CodConglomeradoPrudencial"].nunique()),
    }
    return moments_b, moments_d, meta


def render_table1(moments_b, moments_d, meta, weight_col, suffix) -> str:
    name      = "Compressed_BankType_CrossSection"
    tab_label = f"tab:{name}{suffix}"
    weight_lbl = "(Population Weighted)" if weight_col else "(Unweighted)"
    caption   = (f"Bank-Conglomerate Cross-Section by Type {weight_lbl}")

    col_spec = "l@{\\hspace{0.5em}}rrrrrr"
    header = r"Variable & Mean & SD & p10 & p50 & p90 & $N$ \\"

    def _panel(title: str, moments: pd.DataFrame, n_firms: int, n_fq: int) -> list[str]:
        lines = [
            r"\midrule",
            f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\",
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
                              _fmt_val(r["p50"],  vb),
                              _fmt_val(r["p90"],  vb)])
                + f" & {int(r['N']):,} \\\\"
            )
        lines.append(r"\addlinespace[0.3em]")
        lines.append(
            f"\\multicolumn{{7}}{{l}}{{\\footnotesize Unique conglomerates: {n_firms:,}; "
            f"firm-quarter observations: {n_fq:,}.}} \\\\"
        )
        return lines

    notes = (
        r"\scriptsize \textit{Notes:} Each observation is a "
        r"prudential-conglomerate $\times$ calendar-quarter pair. For type-B "
        r"(brick-and-mortar) firms, MCA-level deposits are summed within "
        r"firm-quarter, and spreads are deposit-weighted across MCAs within "
        r"the corresponding asset class. Quantiles are computed over the pooled "
        r"firm-quarter sample. PIX-related products (type 5) are zero by "
        r"construction prior to 2020Q4. "
        + ("Statistics are population-weighted using \\texttt{pop\\_total}."
           if weight_col else "Statistics are unweighted.")
    )

    body = []
    body += _panel("Panel A: Brick-and-Mortar (B) Firms",
                   moments_b, meta["n_firms_B"], meta["n_firm_quarters_B"])
    body += _panel("Panel B: Digital (D) Firms",
                   moments_d, meta["n_firms_D"], meta["n_firm_quarters_D"])

    tex = "\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{threeparttable}",
        f"\\caption{{{caption}}}\\label{{{tab_label}}}",
        r"\footnotesize",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\item " + notes,
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
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

    # Step 4: deposit-weighted spreads (numerator/denominator vectorized)
    for v, w in (("spread_a4", "dep_a4"), ("spread_a5", "dep_a5")):
        df_b[f"_num_{v}"] = df_b[v] * df_b[w].where(df_b[w] > 0, 0.0)
        df_b[f"_den_{v}"] = df_b[w].where(df_b[w] > 0, 0.0).where(df_b[v].notna(), 0.0)

    spread_agg = (
        df_b.groupby(grp_keys, sort=False)
            .agg({"_num_spread_a4": "sum", "_den_spread_a4": "sum",
                  "_num_spread_a5": "sum", "_den_spread_a5": "sum"})
            .reset_index()
    )
    spread_agg["spread_a4_w"] = np.where(spread_agg["_den_spread_a4"] > 0,
                                         spread_agg["_num_spread_a4"] / spread_agg["_den_spread_a4"],
                                         np.nan)
    spread_agg["spread_a5_w"] = np.where(spread_agg["_den_spread_a5"] > 0,
                                         spread_agg["_num_spread_a5"] / spread_agg["_den_spread_a5"],
                                         np.nan)

    # Step 5: MCA-quarter population (first non-null)
    pop_mq = (df_b.groupby(grp_keys, sort=False)["pop_total"]
                   .first()
                   .reset_index())

    mca_q = (mq_tot[grp_keys + ["n_b_firms"]]
             .merge(hhi, on=grp_keys, how="left")
             .merge(spread_agg[grp_keys + ["spread_a4_w", "spread_a5_w"]], on=grp_keys, how="left")
             .merge(pop_mq, on=grp_keys, how="left"))

    # National D-firm count by year (active = positive total deposits)
    df_d["dep_total_jkmt"] = df_d[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].sum(axis=1, min_count=1)
    d_active = df_d[df_d["dep_total_jkmt"] > 0]
    d_count_yr = (d_active.groupby("year")["CodConglomeradoPrudencial"]
                          .nunique()
                          .rename("n_d_firms_natl")
                          .reset_index())

    years = sorted(df["year"].unique().tolist())
    w_col = weight_col if (weight_col and weight_col in mca_q.columns) else None

    rows = []
    for v in ["n_b_firms", "hhi_b", "spread_a4_w", "spread_a5_w"]:
        row = {"var": v}
        for yr in years:
            sub = mca_q[mca_q["year"] == yr]
            x = sub[v].values
            w = sub[w_col].values if w_col else None
            row[yr] = _weighted_mean(x, w)
        rows.append(row)

    # D-firm count: same value across the year
    drow = {"var": "n_d_firms_natl"}
    for yr in years:
        v = d_count_yr.loc[d_count_yr["year"] == yr, "n_d_firms_natl"]
        drow[yr] = float(v.iloc[0]) if len(v) else 0.0
    rows.append(drow)

    # N row: number of MCA-quarters per year
    nrow = {"var": "N_obs"}
    for yr in years:
        nrow[yr] = int((mca_q["year"] == yr).sum())
    rows.append(nrow)

    return pd.DataFrame(rows)


def render_table2(t2_df, weight_col, suffix) -> str:
    name      = "Compressed_MarketStructure_by_Year"
    tab_label = f"tab:{name}{suffix}"
    weight_lbl = "(Population Weighted)" if weight_col else "(Unweighted)"
    caption   = f"MCA-Level Market Structure, Annual Averages {weight_lbl}"

    year_cols = [c for c in t2_df.columns if c != "var"]
    n_groups  = len(year_cols)

    col_spec = "l@{\\hspace{0.2em}}" + ">{\\centering\\arraybackslash}X" * n_groups
    header   = " & " + " & ".join(str(y) for y in year_cols) + r" \\"

    PRE2020_BLANK = {"spread_a5_w"}

    body = []
    for _, row in t2_df.iterrows():
        vb  = row["var"]
        if vb == "N_obs":
            continue
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

    notes = (
        r"\scriptsize \textit{Notes:} Each year column is the mean across "
        r"MCA $\times$ quarter cells within that calendar year. The number of "
        r"B firms is the count of distinct prudential conglomerates with "
        r"strictly positive deposits in the MCA-quarter; HHI is scaled to "
        r"$0$--$10{,}000$ from B-firm deposit shares within MCA-quarter; "
        r"deposit-weighted spreads use the corresponding asset-class deposits "
        r"as weights. The D-firm count is national (constant across MCAs in a "
        r"given year). Type-5 (prepaid) spreads are undefined before 2020. "
        + ("Cross-MCA means are population-weighted using \\texttt{pop\\_total}."
           if weight_col else "Cross-MCA means are unweighted.")
    )

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
        f"    \\multicolumn{{{n_groups+1}}}{{p{{0.95\\linewidth}}}}{{{notes}}} \\\\",
        r"    \endlastfoot",
        *body,
        r"\end{xltabular}",
        r"\endgroup",
        r"\end{landscape}",
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
    tab_label = f"tab:{name}{suffix}"
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
        r"\begin{table}[ht]",
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
    ])
    return tex


# ---------------------------------------------------------------------------
# Table 4: D-firm pre/post Pix (IK/CSS cluster-robust)
# ---------------------------------------------------------------------------
T4_VARS = ["log_total_assets", "log_dep_a4", "spread_a4", "equity_ratio"]


def build_table4(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_d = df[df["bank_type"] == "D"].copy()
    # D rows are already firm-quarter
    df_d = df_d[["CodConglomeradoPrudencial", "year", "quarter", "post",
                 "total_assets", "equity_ratio", "dep_a4", "spread_a4"]].copy()
    df_d["log_total_assets"] = np.log(df_d["total_assets"].where(df_d["total_assets"] > 0))
    df_d["log_dep_a4"]       = np.log(df_d["dep_a4"].where(df_d["dep_a4"] > 0))

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
    tab_label = f"tab:{name}{suffix}"
    caption   = ("Digital (D) Firms Before and After Pix: Difference-in-Means "
                 "with Conglomerate-Clustered Inference")

    col_spec = "l@{\\hspace{0.4em}}rrrrrrrr"
    header   = (r"Variable & Pre Mean & Post Mean & $\Delta$ & "
                r"CR SE & $t$ & $G$ & $G^{\star}$ & $p$ \\")

    body = []
    for _, r in t4_df.iterrows():
        vb = r["var"]
        diff_cell = f"{_fmt_val(r['diff'], vb)}{_stars(r['p'])}"
        body.append(
            f"{_row_label(vb)} & "
            f"{_fmt_val(r['pre_mean'],  vb)} & "
            f"{_fmt_val(r['post_mean'], vb)} & "
            f"{diff_cell} & "
            f"{_fmt_val(r['se'], vb)} & "
            f"{r['t']:.2f} & "
            f"{int(r['G'])} & "
            f"{r['G_star']:.1f} & "
            f"{r['p']:.3f} \\\\"
        )

    # Context row: active D-firm counts (no test).
    body.append(r"\addlinespace[0.4em]")
    body.append(r"\midrule")
    body.append(
        f"Active D Conglomerates "
        f"& {meta['n_d_firms_pre']:,} "
        f"& {meta['n_d_firms_post']:,} "
        f"& {meta['n_d_firms_post'] - meta['n_d_firms_pre']:+,} "
        r"& \multicolumn{5}{c}{\textit{(count, no test)}} \\"
    )
    body.append(
        f"Firm-Quarters ($N$) "
        f"& {meta['n_firm_quarters_pre']:,} "
        f"& {meta['n_firm_quarters_post']:,} "
        r"& \multicolumn{6}{c}{} \\"
    )

    notes = (
        r"\scriptsize \textit{Notes:} Each row reports a univariate "
        r"regression $y_{jt} = \alpha + \beta\,\mathbf{1}\{t \geq 2020\mathrm{Q}4\} + "
        r"\varepsilon_{jt}$ over the D-firm $\times$ quarter panel, where "
        r"$j$ indexes prudential conglomerates and $t$ indexes calendar quarters. "
        r"$\Delta = \hat{\beta}$ is the post-minus-pre mean difference; "
        r"\textit{CR SE} is the cluster-robust (CRV1) standard error with "
        r"clustering at the conglomerate level. Inference uses the effective "
        r"number of clusters $G^{\star} = G / (1 + \mathrm{cv}^{2})$ as "
        r"Satterthwaite degrees of freedom, following \textcite{imbens2016robust} "
        r"and \textcite{carter2017asymptotic}. The PIX threshold is "
        r"$t \geq 2020\mathrm{Q}4$, the activation of the Brazilian instant "
        r"payment system on Nov.\ 16, 2020. Type-5 (prepaid) outcomes are not "
        r"reported because they are zero by construction prior to 2020Q4. "
        r"Stars: $^{*}\,p<0.10$, $^{**}\,p<0.05$, $^{***}\,p<0.01$."
    )

    tex = "\n".join([
        r"\begin{table}[ht]",
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
    ])
    return tex


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

    # --- Table 1
    print("\n[Table 1] Bank-conglomerate cross-section ...")
    mom_b, mom_d, meta1 = build_table1(df, args.weight_col)
    _write_csv(mom_b.assign(panel="B"), "Compressed_BankType_CrossSection_B", suffix)
    _write_csv(mom_d.assign(panel="D"), "Compressed_BankType_CrossSection_D", suffix)
    tex1 = render_table1(mom_b, mom_d, meta1, args.weight_col, suffix)
    _write_tex(tex1, "Compressed_BankType_CrossSection", suffix)

    # --- Table 2
    print("\n[Table 2] MCA market structure by year ...")
    t2 = build_table2(df, args.weight_col)
    _write_csv(t2, "Compressed_MarketStructure_by_Year", suffix)
    tex2 = render_table2(t2, args.weight_col, suffix)
    _write_tex(tex2, "Compressed_MarketStructure_by_Year", suffix)

    # --- Table 3
    print("\n[Table 3] Local environment by macro-region ...")
    latest_year = int(df["year"].max())
    t3 = build_table3(df, args.weight_col)
    _write_csv(t3, "Compressed_LocalEnvironment_by_Region", suffix)
    tex3 = render_table3(t3, args.weight_col, suffix, latest_year)
    _write_tex(tex3, "Compressed_LocalEnvironment_by_Region", suffix)

    # --- Table 4
    print("\n[Table 4] D-firm Pre/Post Pix ...")
    t4, meta4 = build_table4(df)
    _write_csv(t4, "Compressed_DFirm_PrePost_Pix", suffix)
    tex4 = render_table4(t4, meta4, suffix)
    _write_tex(tex4, "Compressed_DFirm_PrePost_Pix", suffix)

    print("\nDone.")


if __name__ == "__main__":
    main()
