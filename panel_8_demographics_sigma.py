"""
panel_8_demographics_sigma.py
=============================
Pipeline Step 4c — Compute within-MCA demographic dispersion for BLP draws.

For each of the 8 BLP demographic interaction columns (D_COLS), this script
computes the **population-weighted within-MCA standard deviation** of the
municipality-level values.  The output is a lightweight Parquet file:

    BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/DEMAND_PREP/demographics_sigma.parquet

Columns:  mca_code, year, quarter,
          gdp_per_capita_sigma, fraction_65plus_sigma, fraction_young_sigma,
          pix_users_pf_per1000_sigma, connections_per100_sigma, frac_4g5g_sigma,
          branches_per1000_sigma, cadunico_families_per1000_sigma

This file is consumed by blp_loop.jl to replace the ad-hoc 0.1 × σ_national
scaling with market-specific parametric draws:  d_im ~ N(μ_m, σ_m).

Inputs (municipality-level intermediates saved by the scrapers):
  1. IBGE/muni_demographics_panel.csv           (scrape_2)
  2. BCB/PIX/pix_muni_panel.csv                 (scrape_3)
  3. ANATEL/anatel_muni_panel.csv                (scrape_4)
  4. BCB/Inclusion/bcb_inclusion_muni_panel.csv  (scrape_5)
  5. CadUnico/cadunico_muni_panel.csv            (scrape_6)
  6. IBGE/mca_demographics_panel.csv             (for pop_total denominators)

Strategy for single-municipality MCAs (53 of 466, ~11%):
  Within-MCA σ is undefined (n=1).  We impute using the median σ from MCAs
  in the **same state** (via state_code from the crosswalk).  If the state
  has no multi-municipality MCAs, we fall back to the national median.
"""

import os
import sys
import logging
from pathlib import Path

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="latin-1", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="latin-1", errors="replace", line_buffering=True)

# ==============================================================================
# 0. Paths
# ==============================================================================
BASE = Path(__file__).resolve().parents[2]

IBGE_DIR   = BASE / "IBGE"
PIX_DIR    = BASE / "BCB" / "PIX"
ANATEL_DIR = BASE / "ANATEL"
INCL_DIR   = BASE / "BCB" / "Inclusion"
CAD_DIR    = BASE / "CadUnico"

# Municipality-level intermediates produced by modified scrapers
IBGE_MUNI   = IBGE_DIR / "muni_demographics_panel.csv"
PIX_MUNI    = PIX_DIR  / "pix_muni_panel.csv"
ANATEL_MUNI = ANATEL_DIR / "anatel_muni_panel.csv"
INCL_MUNI   = INCL_DIR / "bcb_inclusion_muni_panel.csv"
CAD_MUNI    = CAD_DIR  / "cadunico_muni_panel.csv"

# MCA-level panels (for population denominators)
DEMO_MCA    = IBGE_DIR / "mca_demographics_panel.csv"
MCA_XWALK   = IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"

# Output
OUTPUT_DIR = BASE / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "ESTIMATION_OUTPUT" / "DEMAND_PREP"
OUTPUT_PQ  = OUTPUT_DIR / "demographics_sigma.parquet"

D_COLS = ['gdp_per_capita', 'fraction_65plus', 'fraction_young',
          'pix_users_pf_per1000', 'connections_per100', 'frac_4g5g',
          'branches_per1000', 'cadunico_families_per1000']


# ==============================================================================
# 1. Helpers
# ==============================================================================

def _weighted_std(values: pd.Series, weights: pd.Series) -> float:
    """Population-weighted standard deviation (ddof=0, i.e. MLE)."""
    mask = values.notna() & weights.notna() & (weights > 0)
    v = values[mask].values.astype(float)
    w = weights[mask].values.astype(float)
    if len(v) < 2:
        return np.nan
    w_sum = w.sum()
    if w_sum == 0:
        return np.nan
    mu = np.average(v, weights=w)
    var = np.average((v - mu) ** 2, weights=w)
    return np.sqrt(var)


def _compute_sigma_by_mca(df: pd.DataFrame, val_col: str, pop_col: str,
                            group_cols: list[str]) -> pd.DataFrame:
    """Compute population-weighted within-MCA σ for a single variable."""
    result = (
        df.groupby(group_cols)
          .apply(lambda g: _weighted_std(g[val_col], g[pop_col]),
                 include_groups=False)
          .rename(f"{val_col}_sigma")
          .reset_index()
    )
    return result


# ==============================================================================
# 2. Load municipality-level data and compute per-variable σ
# ==============================================================================

def load_population_by_muni() -> pd.DataFrame:
    """Load municipality-level population from the IBGE intermediate."""
    if not IBGE_MUNI.exists():
        logging.warning(f"IBGE municipality intermediate not found: {IBGE_MUNI}")
        return pd.DataFrame()
    df = pd.read_csv(IBGE_MUNI, dtype={"mca_code": str})
    return df


def compute_ibge_sigmas(muni_pop: pd.DataFrame) -> pd.DataFrame:
    """Compute within-MCA σ for gdp_per_capita, fraction_65plus, fraction_young."""
    if muni_pop.empty:
        logging.warning("No IBGE municipality data — IBGE sigmas will be NaN.")
        return pd.DataFrame()

    frames = []
    for col in ["gdp_per_capita", "fraction_65plus", "fraction_young"]:
        sigma = _compute_sigma_by_mca(muni_pop, col, "population",
                                       ["mca_code", "year"])
        frames.append(sigma)

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["mca_code", "year"], how="outer")
    return result


def compute_pix_sigma() -> pd.DataFrame:
    """Compute within-MCA σ for pix_users_pf_per1000 (monthly unique users)."""
    if not PIX_MUNI.exists():
        logging.warning(f"PIX municipality intermediate not found: {PIX_MUNI}")
        return pd.DataFrame()

    pix = pd.read_csv(PIX_MUNI, dtype={"mca_code": str})

    # Need population to compute per-1000 at municipality level
    muni_pop = load_population_by_muni()
    if muni_pop.empty:
        return pd.DataFrame()

    # Aggregate QT_PES_PagadorPF to municipality × year × quarter
    pix_q = (pix.groupby(["municipality_code", "mca_code", "year", "quarter"],
                          as_index=False)["QT_PES_PagadorPF"].sum())

    # Merge population (annual, same year)
    pix_q = pix_q.merge(
        muni_pop[["municipality_code", "year", "population"]].drop_duplicates(),
        on=["municipality_code", "year"], how="left")

    pix_q["pix_users_pf_per1000"] = np.where(
        pix_q["population"] > 0,
        pix_q["QT_PES_PagadorPF"] / (pix_q["population"] / 1000),
        np.nan)

    return _compute_sigma_by_mca(pix_q, "pix_users_pf_per1000", "population",
                                  ["mca_code", "year", "quarter"])


def compute_anatel_sigma() -> pd.DataFrame:
    """Compute within-MCA σ for connections_per100 and frac_4g5g."""
    if not ANATEL_MUNI.exists():
        logging.warning(f"ANATEL municipality intermediate not found: {ANATEL_MUNI}")
        return pd.DataFrame()

    anatel = pd.read_csv(ANATEL_MUNI, dtype={"mca_code": str})

    # Need population to compute per-100 at municipality level
    muni_pop = load_population_by_muni()
    if muni_pop.empty:
        return pd.DataFrame()

    anatel = anatel.merge(
        muni_pop[["municipality_code", "year", "population"]].drop_duplicates(),
        left_on=["mun_code", "year"], right_on=["municipality_code", "year"],
        how="left")

    anatel["connections_per100"] = np.where(
        anatel["population"] > 0,
        anatel["acessos"] / anatel["population"] * 100,
        np.nan)
    anatel["frac_4g5g"] = np.where(
        anatel["acessos"] > 0,
        anatel["acessos_fast"] / anatel["acessos"],
        np.nan)

    sigma_conn = _compute_sigma_by_mca(anatel, "connections_per100", "population",
                                        ["mca_code", "year", "quarter"])
    sigma_4g   = _compute_sigma_by_mca(anatel, "frac_4g5g", "population",
                                        ["mca_code", "year", "quarter"])
    return sigma_conn.merge(sigma_4g, on=["mca_code", "year", "quarter"], how="outer")


def compute_branches_sigma() -> pd.DataFrame:
    """Compute within-MCA σ for branches_per1000."""
    if not INCL_MUNI.exists():
        logging.warning(f"BCB inclusion municipality intermediate not found: {INCL_MUNI}")
        return pd.DataFrame()

    incl = pd.read_csv(INCL_MUNI, dtype={"mca_code": str})

    # Need population to compute per-1000 at municipality level
    muni_pop = load_population_by_muni()
    if muni_pop.empty:
        return pd.DataFrame()

    incl = incl.merge(
        muni_pop[["municipality_code", "year", "population"]].drop_duplicates(),
        left_on=["mun_code", "year"], right_on=["municipality_code", "year"],
        how="left")

    incl["branches_per1000"] = np.where(
        incl["population"] > 0,
        incl["branches"] / (incl["population"] / 1000),
        np.nan)

    return _compute_sigma_by_mca(incl, "branches_per1000", "population",
                                  ["mca_code", "year"])


def compute_cadunico_sigma() -> pd.DataFrame:
    """Compute within-MCA σ for cadunico_families_per1000."""
    if not CAD_MUNI.exists():
        logging.warning(f"CadUnico municipality intermediate not found: {CAD_MUNI}")
        return pd.DataFrame()

    cad = pd.read_csv(CAD_MUNI, dtype={"mca_code": str})

    # Need population to compute per-1000 at municipality level (6-digit codes)
    muni_pop = load_population_by_muni()
    if muni_pop.empty:
        return pd.DataFrame()

    # CadUnico uses 6-digit IBGE codes; municipality_code in muni_pop is 7-digit
    muni_pop_6 = muni_pop[["municipality_code", "year", "population"]].drop_duplicates().copy()
    muni_pop_6["cod_ibge6"] = muni_pop_6["municipality_code"] // 10

    cad = cad.merge(muni_pop_6[["cod_ibge6", "year", "population"]].drop_duplicates(),
                     on=["cod_ibge6", "year"], how="left")

    cad["cadunico_families_per1000"] = np.where(
        cad["population"] > 0,
        cad["families_total"] / (cad["population"] / 1000),
        np.nan)

    return _compute_sigma_by_mca(cad, "cadunico_families_per1000", "population",
                                  ["mca_code", "year", "quarter"])


# ==============================================================================
# 3. Impute single-municipality MCAs
# ==============================================================================

def impute_single_muni_mcas(sigma_df: pd.DataFrame) -> pd.DataFrame:
    """
    For MCAs with n_municipalities == 1, within-MCA σ is NaN.
    Impute using the median σ from multi-municipality MCAs in the same state.
    """
    # Load state mapping
    xwalk = pd.read_csv(
        MCA_XWALK,
        usecols=["mca_code", "state_code", "year"],
        dtype={"mca_code": str, "state_code": int, "year": int}
    ).drop_duplicates(subset=["mca_code", "year"])

    sigma_df = sigma_df.merge(xwalk, on=["mca_code", "year"], how="left")

    sigma_cols = [c for c in sigma_df.columns if c.endswith("_sigma")]

    for col in sigma_cols:
        # Compute state-level median σ from non-NaN rows (multi-muni MCAs)
        state_median = (
            sigma_df[sigma_df[col].notna()]
            .groupby("state_code")[col]
            .median()
            .rename(f"{col}_state_med")
        )
        national_median = sigma_df[col].median()

        sigma_df = sigma_df.merge(state_median, on="state_code", how="left")
        fill_col = f"{col}_state_med"

        sigma_df[col] = sigma_df[col].fillna(sigma_df[fill_col])
        sigma_df[col] = sigma_df[col].fillna(national_median)
        sigma_df.drop(columns=[fill_col], inplace=True)

    sigma_df.drop(columns=["state_code"], inplace=True)
    return sigma_df


# ==============================================================================
# 4. Main
# ==============================================================================

def main():
    logging.info("Computing within-MCA demographic σ …")

    # ── IBGE: gdp_per_capita, fraction_65plus, fraction_young (annual) ────
    muni_pop = load_population_by_muni()
    ibge_sigma = compute_ibge_sigmas(muni_pop)
    if not ibge_sigma.empty:
        logging.info(f"  IBGE sigmas: {len(ibge_sigma):,} rows")

    # ── PIX: pix_users_pf_per1000 (quarterly) ────────────────────────────
    pix_sigma = compute_pix_sigma()
    if not pix_sigma.empty:
        logging.info(f"  PIX sigma: {len(pix_sigma):,} rows")

    # ── ANATEL: connections_per100, frac_4g5g (quarterly) ─────────────────
    anatel_sigma = compute_anatel_sigma()
    if not anatel_sigma.empty:
        logging.info(f"  ANATEL sigmas: {len(anatel_sigma):,} rows")

    # ── BCB Inclusion: branches_per1000 (annual) ─────────────────────────
    branches_sigma = compute_branches_sigma()
    if not branches_sigma.empty:
        logging.info(f"  Branches sigma: {len(branches_sigma):,} rows")

    # ── CadUnico: cadunico_families_per1000 (quarterly) ──────────────────
    cadunico_sigma = compute_cadunico_sigma()
    if not cadunico_sigma.empty:
        logging.info(f"  CadUnico sigma: {len(cadunico_sigma):,} rows")

    # ── Build skeleton: all MCA × year × quarter from demographics ───────
    demo = pd.read_csv(DEMO_MCA, usecols=["mca_code", "year"],
                        dtype={"mca_code": str, "year": int})
    quarters = pd.DataFrame({"quarter": [1, 2, 3, 4]})
    skeleton = demo.merge(quarters, how="cross")
    skeleton.drop_duplicates(inplace=True)

    # ── Merge annual sigmas (broadcast to all quarters) ──────────────────
    if not ibge_sigma.empty:
        skeleton = skeleton.merge(ibge_sigma, on=["mca_code", "year"], how="left")
    else:
        for c in ["gdp_per_capita", "fraction_65plus", "fraction_young"]:
            skeleton[f"{c}_sigma"] = np.nan

    if not branches_sigma.empty:
        skeleton = skeleton.merge(branches_sigma, on=["mca_code", "year"], how="left")
    else:
        skeleton["branches_per1000_sigma"] = np.nan

    # ── Merge quarterly sigmas ───────────────────────────────────────────
    if not pix_sigma.empty:
        skeleton = skeleton.merge(pix_sigma, on=["mca_code", "year", "quarter"], how="left")
    else:
        skeleton["pix_users_pf_per1000_sigma"] = np.nan

    if not anatel_sigma.empty:
        skeleton = skeleton.merge(anatel_sigma, on=["mca_code", "year", "quarter"], how="left")
    else:
        for c in ["connections_per100", "frac_4g5g"]:
            skeleton[f"{c}_sigma"] = np.nan

    if not cadunico_sigma.empty:
        skeleton = skeleton.merge(cadunico_sigma, on=["mca_code", "year", "quarter"], how="left")
    else:
        skeleton["cadunico_families_per1000_sigma"] = np.nan

    # ── Impute single-municipality MCAs ──────────────────────────────────
    skeleton = impute_single_muni_mcas(skeleton)

    # ── Final column ordering & save ─────────────────────────────────────
    # Build time_id to match BLP prep format: "{year}Q{quarter}"
    skeleton["time_id"] = skeleton["year"].astype(str) + "Q" + skeleton["quarter"].astype(str)

    sigma_cols = [f"{c}_sigma" for c in D_COLS]
    col_order = ["mca_code", "year", "quarter", "time_id"] + sigma_cols
    col_order = [c for c in col_order if c in skeleton.columns]
    skeleton = skeleton[col_order]
    skeleton.sort_values(["mca_code", "year", "quarter"], inplace=True)
    skeleton.reset_index(drop=True, inplace=True)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    skeleton.to_parquet(OUTPUT_PQ, engine="pyarrow", index=False)
    logging.info(f"Saved demographics sigma to {OUTPUT_PQ}")

    # Summary
    print(f"\nDemographics sigma summary")
    print(f"  Rows:    {len(skeleton):,}")
    print(f"  MCAs:    {skeleton['mca_code'].nunique()}")
    print(f"  Years:   {skeleton['year'].min()} – {skeleton['year'].max()}")
    for col in sigma_cols:
        if col in skeleton.columns:
            nna = skeleton[col].notna().sum()
            med = skeleton[col].median()
            print(f"  {col:<40s} non-null: {nna:,}  median: {med:.4f}")
    print(f"  Output:  {OUTPUT_PQ}")


if __name__ == "__main__":
    main()
