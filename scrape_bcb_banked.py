## scrape_bcb_banked.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-04-16
#
# Purpose: Build an MCA × year panel of total household deposit volume per capita
#          and a locally-adjusted banked-fraction proxy, for use as a data-driven
#          replacement of the ad hoc 1.1 multiplier in eq:16 of the BLP estimation.
#
#   Sources:
#     (A)  BCB ESTBAN individual monthly CSVs (Documento 4500 por município),
#          already downloaded to:
#            BCB/ESTBAN/Relatório por município/{YYYYMM}_ESTBAN.CSV
#          Only December snapshots are used (annual stock).
#          Relevant deposit columns (identified by keyword match):
#            VERBETE_401* — demand deposits à vista aggregate
#                          (all subtypes: government, individuals, legal entities,
#                           judicial, mandatory, investment-linked, earmarked, etc.)
#            VERBETE_420  — poupança (savings deposits)
#            VERBETE_432  — depósitos a prazo (time deposits / CDB)
#          Note: COSIF verbetes change slightly across vintages; column identification
#          uses partial-name matching, not positional indexing.
#          Note: Conta de Pagamento Pré-Paga (type 5, IF-Data account 110560) is
#          absent from ESTBAN — BCB tracks it under a separate COSIF group (2.3.7.xx)
#          not published in the ESTBAN verbete structure. No ESTBAN proxy available.
#
#     (B)  World Bank API — Global Findex indicator FX.OWN.TOTL.ZS:
#            "Account at a financial institution (% age 15+)", Brazil.
#          Available for survey years only: 2011, 2014, 2017, 2021, 2024.
#          Linearly interpolated to all years between survey points.
#          Fetched live; if API unavailable, falls back to hard-coded values.
#
#   Methodology:
#     dep_percapita[m,t]    = (dep_vista + dep_poupanca + dep_prazo)[m,t] / pop[m,t]
#     nat_p95[t]            = 95th percentile of dep_percapita across MCAs in year t
#     dep_intensity[m,t]    = clip(dep_percapita[m,t] / nat_p95[t], 0, 1)
#     banked_frac_proxy[m,t] = findex_interp[t] * dep_intensity[m,t]
#     banked_correction[m,t] = 1.0 / banked_frac_proxy[m,t]
#
#   The 95th-percentile denominator avoids large-city financial centres (São Paulo,
#   Brasília) pulling all other MCAs toward zero intensity.
#   banked_frac_proxy is bounded above by findex_interp by construction.
#   MCAs with zero population or zero deposits get banked_frac_proxy = NaN.
#
#   Output: BCB/Inclusion/bcb_banked_mca_panel.csv
#     mca_code              — MCA geographic unit code (string)
#     year                  — Calendar year (December snapshot)
#     dep_vista_total       — Sum of VERBETE_401 balances in MCA (R$ thousands)
#     dep_poupanca_total    — Sum of VERBETE_420 balances in MCA (R$ thousands)
#     dep_prazo_total       — Sum of VERBETE_432 balances in MCA (R$ thousands)
#     dep_total             — dep_vista_total + dep_poupanca_total + dep_prazo_total (R$ thousands)
#     pop_total             — MCA population
#     dep_percapita         — dep_total / pop_total (R$ thousands per person)
#     findex_banked_frac    — National FX.OWN.TOTL.ZS / 100, linearly interpolated
#     nat_p95_dep_percapita — 95th-percentile dep_percapita nationally for that year
#     dep_intensity         — dep_percapita / nat_p95_dep_percapita, clipped to [0,1]
#     banked_frac_proxy     — findex_banked_frac * dep_intensity
#     banked_correction     — 1 / banked_frac_proxy  (replacement for 1.1 in eq:16)
###────────────────────────────────────────────────────────────────────────────

import os
import sys
import logging

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import json
import urllib.request
import numpy as np
import pandas as pd

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

###────────────────────────────────────────────────────────────────────────────
## 1) PATHS & CONSTANTS
###────────────────────────────────────────────────────────────────────────────

from utils import paths
BASE      = str(paths.OPEN_FINANCE)
INCL_DIR  = str(paths.INCLUSION_DIR)
IBGE_DIR  = str(paths.IBGE_DIR)
ESTBAN_DIR = str(paths.ESTBAN_RAW_MUN)

MCA_CSV   = os.path.join(IBGE_DIR, "muni_mca_regions_2010_2024_panel.csv")
DEMO_CSV  = os.path.join(IBGE_DIR, "mca_demographics_panel.csv")
OUTPUT_CSV = os.path.join(INCL_DIR, "bcb_banked_mca_panel.csv")

os.makedirs(INCL_DIR, exist_ok=True)

PANEL_START_YEAR = 2013
PANEL_END_YEAR   = 2025

# World Bank API — Findex indicator for Brazil
WB_URL = (
    "https://api.worldbank.org/v2/country/BR/indicator/FX.OWN.TOTL.ZS"
    "?format=json&per_page=100"
)
# Hard-coded fallback values (% age 15+) in case API is unreachable
FINDEX_FALLBACK = {
    2011: 55.860,
    2014: 68.123,
    2017: 70.044,
    2021: 84.036,
    2024: 86.381,
}

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "scrape_8_bcb_banked",
        {
            "incl_dir":  INCL_DIR,
            "ibge_dir":  IBGE_DIR,
            "estban_dir": ESTBAN_DIR,
            "mca_csv":   MCA_CSV,
            "demo_csv":  DEMO_CSV,
            "output_csv": OUTPUT_CSV,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    INCL_DIR   = _paths["incl_dir"]
    IBGE_DIR   = _paths["ibge_dir"]
    ESTBAN_DIR = _paths["estban_dir"]
    MCA_CSV    = _paths["mca_csv"]
    DEMO_CSV   = _paths["demo_csv"]
    OUTPUT_CSV = _paths["output_csv"]


###────────────────────────────────────────────────────────────────────────────
## 2) FINDEX — FETCH AND INTERPOLATE
###────────────────────────────────────────────────────────────────────────────

def fetch_findex() -> dict:
    """
    Fetch FX.OWN.TOTL.ZS for Brazil from the World Bank API.
    Returns {year (int): value (float, percent scale 0-100)} for non-null years.
    Falls back to FINDEX_FALLBACK on any network or parse error.
    """
    try:
        with urllib.request.urlopen(WB_URL, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        records = payload[1]  # payload = [metadata_dict, [records...]]
        out = {}
        for r in records:
            if r.get("value") is not None:
                try:
                    out[int(r["date"])] = float(r["value"])
                except (ValueError, KeyError):
                    pass
        if not out:
            raise ValueError("Empty response from World Bank API")
        log.info(f"Findex: fetched {len(out)} survey-year observations from WB API.")
        return out
    except Exception as exc:
        log.warning(f"World Bank API unavailable ({exc}); using hard-coded Findex fallback.")
        return dict(FINDEX_FALLBACK)


def interpolate_findex(survey_dict: dict, years: range) -> dict:
    """
    Linearly interpolate Findex survey observations to every integer year in `years`.
    Extrapolates linearly beyond the first/last survey point.
    Returns {year: fraction (0-1)}.
    """
    if not survey_dict:
        return {}

    survey_years = sorted(survey_dict.keys())
    survey_vals  = [survey_dict[y] for y in survey_years]

    result = {}
    for y in years:
        # numpy.interp handles interior interpolation and clamps at endpoints for extrapolation;
        # we want linear extrapolation beyond range, so handle manually for boundary years.
        if y <= survey_years[0]:
            # Extrapolate backward from first two survey points
            if len(survey_years) >= 2:
                slope = (survey_vals[1] - survey_vals[0]) / (survey_years[1] - survey_years[0])
                val = survey_vals[0] + slope * (y - survey_years[0])
            else:
                val = survey_vals[0]
        elif y >= survey_years[-1]:
            # Extrapolate forward (conservative: hold last value)
            val = survey_vals[-1]
        else:
            val = float(np.interp(y, survey_years, survey_vals))
        # Convert from percent to fraction, clip to [0, 1]
        result[y] = float(np.clip(val / 100.0, 0.0, 1.0))

    log.info(
        f"Findex interpolated for {min(years)}–{max(years)}: "
        f"{min(result.values()):.3f}–{max(result.values()):.3f}"
    )
    return result


###────────────────────────────────────────────────────────────────────────────
## 3) ESTBAN — LOAD DECEMBER SNAPSHOTS
###────────────────────────────────────────────────────────────────────────────

def _find_col(columns: list, keyword: str) -> str | None:
    """Return first column name that contains `keyword` (case-insensitive)."""
    kw = keyword.upper()
    for c in columns:
        if kw in c.upper():
            return c
    return None


def load_estban_december(year: int) -> pd.DataFrame | None:
    """
    Load one December ESTBAN file for `year`.
    Returns a DataFrame with columns [mun_code (int), dep_vista, dep_poupanca]
    aggregated at municipality level, or None if the file is unavailable.

    Negative balances (netting artefacts) are clipped to 0 before aggregation.
    Deposits are in R$ thousands (as published by BCB).
    """
    # Try both upper- and lower-case .CSV extension
    for ext in ("CSV", "csv"):
        fname = f"{year}12_ESTBAN.{ext}"
        fpath = os.path.join(ESTBAN_DIR, fname)
        if os.path.exists(fpath):
            break
    else:
        log.warning(f"ESTBAN December file not found for {year} in {ESTBAN_DIR}")
        return None

    try:
        # The file has 2 preamble rows before the header:
        #   Row 1: "ESTBAN (Documento 4500) por municipio"
        #   Row 2: "Data de geracao dos dados: YYYY-MM-DD"
        #   Row 3: actual header (starts with "#DATA_BASE")
        df = pd.read_csv(
            fpath,
            sep=";",
            skiprows=2,          # skip preamble, use row 3 as header
            encoding="latin-1",
            dtype=str,
            on_bad_lines="skip",
            low_memory=False,
        )
        # Strip BOM / leading '#' from header that BCB adds
        df.columns = [c.strip().lstrip("#") for c in df.columns]

        # Identify required columns by keyword
        col_vista    = _find_col(df.columns, "VERBETE_401")
        col_poupanca = _find_col(df.columns, "VERBETE_420")
        col_prazo    = _find_col(df.columns, "VERBETE_432")
        col_mun      = _find_col(df.columns, "CODMUN_IBGE")

        if col_mun is None:
            log.warning(f"{year}: CODMUN_IBGE column not found — skipping file.")
            return None
        if col_vista is None:
            log.warning(f"{year}: VERBETE_401 column not found — dep_vista will be 0.")
        if col_poupanca is None:
            log.warning(f"{year}: VERBETE_420 column not found — dep_poupanca will be 0.")
        if col_prazo is None:
            log.warning(f"{year}: VERBETE_432 column not found — dep_prazo will be 0.")

        keep = {col_mun: "mun_code"}
        if col_vista:
            keep[col_vista] = "dep_vista"
        if col_poupanca:
            keep[col_poupanca] = "dep_poupanca"
        if col_prazo:
            keep[col_prazo] = "dep_prazo"

        df = df[list(keep.keys())].rename(columns=keep).copy()

        # Parse and clean
        df["mun_code"] = pd.to_numeric(df["mun_code"], errors="coerce")
        df = df.dropna(subset=["mun_code"])
        df["mun_code"] = df["mun_code"].astype(int)

        for dep_col in ("dep_vista", "dep_poupanca", "dep_prazo"):
            if dep_col in df.columns:
                df[dep_col] = pd.to_numeric(df[dep_col], errors="coerce").fillna(0.0)
                df[dep_col] = df[dep_col].clip(lower=0.0)
            else:
                df[dep_col] = 0.0

        # Aggregate across institutions within each municipality
        agg = (
            df.groupby("mun_code")[["dep_vista", "dep_poupanca", "dep_prazo"]]
              .sum()
              .reset_index()
        )
        agg["year"] = year

        log.info(
            f"{year}: {len(agg):,} municipalities | "
            f"dep_vista={agg['dep_vista'].sum()/1e6:.1f}B R$k | "
            f"dep_poupanca={agg['dep_poupanca'].sum()/1e6:.1f}B R$k | "
            f"dep_prazo={agg['dep_prazo'].sum()/1e6:.1f}B R$k"
        )
        return agg

    except Exception as exc:
        log.error(f"{year}: failed to load ESTBAN — {exc}")
        return None


def build_estban_panel() -> pd.DataFrame:
    """
    Load December ESTBAN for all years PANEL_START_YEAR..PANEL_END_YEAR.
    Returns concatenated DataFrame [mun_code, year, dep_vista, dep_poupanca, dep_prazo].
    """
    frames = []
    for yr in range(PANEL_START_YEAR, PANEL_END_YEAR + 1):
        df = load_estban_december(yr)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("No ESTBAN December files could be loaded.")

    panel = pd.concat(frames, ignore_index=True)
    log.info(f"ESTBAN panel total: {len(panel):,} municipality-year rows.")
    return panel


###────────────────────────────────────────────────────────────────────────────
## 4) CROSSWALK & POPULATION
###────────────────────────────────────────────────────────────────────────────

def load_mca_crosswalk() -> pd.DataFrame:
    """Load municipality → MCA crosswalk. Keeps only relevant columns."""
    df = pd.read_csv(
        MCA_CSV,
        usecols=["municipality_code", "mca_code", "year"],
        dtype={"municipality_code": int, "mca_code": str, "year": int},
    )
    log.info(f"Crosswalk: {len(df):,} municipality-year rows.")
    return df


def load_population() -> pd.DataFrame:
    """Load MCA × year population panel."""
    df = pd.read_csv(
        DEMO_CSV,
        usecols=["mca_code", "year", "pop_total"],
        dtype={"mca_code": str, "year": int},
    )
    df["pop_total"] = pd.to_numeric(df["pop_total"], errors="coerce")
    log.info(f"Demographics: {len(df):,} MCA-year rows.")
    return df


###────────────────────────────────────────────────────────────────────────────
## 5) AGGREGATE TO MCA × YEAR
###────────────────────────────────────────────────────────────────────────────

def aggregate_to_mca(estban: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ESTBAN municipality panel with MCA crosswalk, aggregate deposits to MCA × year.
    """
    merged = estban.merge(
        crosswalk,
        left_on=["mun_code", "year"],
        right_on=["municipality_code", "year"],
        how="left",
    )
    n_miss = merged["mca_code"].isna().sum()
    if n_miss > 0:
        pct = 100 * n_miss / len(merged)
        log.warning(f"{n_miss:,} ({pct:.1f}%) municipality-year rows unmatched to MCA — dropped.")
    merged = merged.dropna(subset=["mca_code"])

    agg = (
        merged.groupby(["mca_code", "year"])[["dep_vista", "dep_poupanca", "dep_prazo"]]
              .sum()
              .reset_index()
    )
    agg.rename(
        columns={
            "dep_vista":    "dep_vista_total",
            "dep_poupanca": "dep_poupanca_total",
            "dep_prazo":    "dep_prazo_total",
        },
        inplace=True,
    )
    agg["dep_total"] = agg["dep_vista_total"] + agg["dep_poupanca_total"] + agg["dep_prazo_total"]
    log.info(f"MCA panel after aggregation: {len(agg):,} rows.")
    return agg


###────────────────────────────────────────────────────────────────────────────
## 6) COMPUTE PROXY
###────────────────────────────────────────────────────────────────────────────

def compute_proxy(
    panel: pd.DataFrame,
    pop: pd.DataFrame,
    findex_map: dict,
) -> pd.DataFrame:
    """
    Merge population, compute dep_percapita, dep_intensity, and banked_frac_proxy.
    """
    # --- merge population ---
    panel = panel.merge(pop, on=["mca_code", "year"], how="left")
    n_pop_miss = panel["pop_total"].isna().sum()
    if n_pop_miss > 0:
        log.warning(f"{n_pop_miss:,} MCA-year rows have no population data — dep_percapita will be NaN.")

    panel["dep_percapita"] = np.where(
        panel["pop_total"] > 0,
        panel["dep_total"] / panel["pop_total"],
        np.nan,
    )

    # --- national 95th percentile per year ---
    p95 = (
        panel.groupby("year")["dep_percapita"]
             .quantile(0.95)
             .rename("nat_p95_dep_percapita")
             .reset_index()
    )
    panel = panel.merge(p95, on="year", how="left")

    panel["dep_intensity"] = np.where(
        panel["nat_p95_dep_percapita"] > 0,
        (panel["dep_percapita"] / panel["nat_p95_dep_percapita"]).clip(upper=1.0),
        np.nan,
    )

    # --- Findex ---
    panel["findex_banked_frac"] = panel["year"].map(findex_map)

    # --- proxy and correction ---
    panel["banked_frac_proxy"] = panel["findex_banked_frac"] * panel["dep_intensity"]
    panel["banked_correction"] = np.where(
        panel["banked_frac_proxy"] > 0,
        1.0 / panel["banked_frac_proxy"],
        np.nan,
    )

    return panel


###────────────────────────────────────────────────────────────────────────────
## 7) MAIN
###────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=== scrape_8_bcb_banked: START ===")

    # --- Findex (can be fetched before ESTBAN loads) ---
    survey = fetch_findex()
    findex_map = interpolate_findex(survey, range(PANEL_START_YEAR, PANEL_END_YEAR + 1))

    # --- ESTBAN ---
    estban = build_estban_panel()

    # --- Crosswalk & population ---
    crosswalk = load_mca_crosswalk()
    pop       = load_population()

    # --- Aggregate to MCA ---
    mca_panel = aggregate_to_mca(estban, crosswalk)

    # --- Compute proxy ---
    out = compute_proxy(mca_panel, pop, findex_map)

    # --- Column order ---
    col_order = [
        "mca_code", "year",
        "dep_vista_total", "dep_poupanca_total", "dep_prazo_total", "dep_total",
        "pop_total", "dep_percapita",
        "findex_banked_frac",
        "nat_p95_dep_percapita", "dep_intensity",
        "banked_frac_proxy", "banked_correction",
    ]
    out = out[[c for c in col_order if c in out.columns]]
    out = out.sort_values(["mca_code", "year"]).reset_index(drop=True)

    # --- Write ---
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    pa_csv.write_csv(pa.Table.from_pandas(out, preserve_index=False), OUTPUT_CSV)
    log.info(f"Written {len(out):,} rows -> {OUTPUT_CSV}")

    # --- Summary statistics ---
    log.info("\n--- Summary by year ---")
    summary = (
        out.groupby("year")
           .agg(
               n_mca              = ("mca_code",          "nunique"),
               dep_percapita_mean = ("dep_percapita",      "mean"),
               findex             = ("findex_banked_frac", "first"),
               proxy_mean         = ("banked_frac_proxy",  "mean"),
               proxy_min          = ("banked_frac_proxy",  "min"),
               proxy_max          = ("banked_frac_proxy",  "max"),
               correction_mean    = ("banked_correction",  "mean"),
           )
    )
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary.to_string())

    log.info("=== scrape_8_bcb_banked: DONE ===")


if __name__ == "__main__":
    main()
