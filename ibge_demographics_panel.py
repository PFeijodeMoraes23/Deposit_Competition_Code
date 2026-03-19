## ibge_demographics_panel.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-03-06
#
# Purpose: Build a panel of MCA (Área Mínima Comparável) level demographics
#          for use as market controls in the deposit competition model.
#
#   Inputs:
#     1. IBGE SIDRA REST API
#          - Table 6579, Variable 9324  : municipal population estimates (2013-2022)
#          - Table 4714, Variable 93    : population projections (2023-2024 extension)
#          - Table 5938, Variable 37    : municipal GDP at current prices, R$ thousands
#                                        (2002-2021); carry-forward for 2022+
#          - Table 9514, Variable 93    : population by age group, 2022 Census
#          - Table 1552, Variable 93    : population by age group, 2010 Census
#            (Tables 9514/1552 used for fraction_65plus and fraction_young; linearly
#             interpolated for years 2013-2024 between the two census years)
#     2. IBGE/muni_mca_regions_2010_2024_panel.csv
#          - Municipality -> MCA crosswalk with year dimension (already in repo)
#
#   Output:
#     IBGE/mca_demographics_panel.csv
#       Columns: mca_code, year,
#                pop_total,            (sum of municipality populations)
#                gdp_total_r1000,      (sum of municipal GDPs, R$ thousands)
#                gdp_per_capita,       (gdp_total_r1000*1000 / pop_total, R$)
#                fraction_65plus,      (population-weighted share aged 65+)
#                fraction_young,       (population-weighted share aged 0-14)
#                n_municipalities,     (distinct municipalities in MCA)
#                gdp_imputed,          (True if GDP year was carried forward)
#                age_interpolated      (True if age shares were linearly interpolated)
#
# SIDRA API: https://api.ibge.gov.br/
#   Endpoint: /api/v3/agregados/{tabela}/periodos/{periodos}/variaveis/{variavel}
#             ?localidades=N6[all][&classificacao={classif}[{cats}]]
#
# Notes:
#   * Census age data (Tables 9514/1552) are fetched for the census years 2022 and 2010
#     using age-group classification 287 with specific 5-year category IDs (municipalities
#     are batched in groups of 100 to avoid HTTP 500 from SIDRA). All inter-census years
#     are linearly interpolated.
#   * GDP municipal data (table 5938) is published with a ~2 year lag. Later years
#     are carried forward from the last available year (flagged gdp_imputed=True).
###-----------------------------------------------------------------------------

import concurrent.futures
import os
import time
import logging
import requests
import pandas as pd
import numpy as np

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

## -----------------------------------------------------------------------------
## 1) PATHS & CONSTANTS
## -----------------------------------------------------------------------------
BASE      = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
IBGE_DIR  = os.path.join(BASE, "IBGE")
MCA_CSV   = os.path.join(IBGE_DIR, "muni_mca_regions_2010_2024_panel.csv")
OUTPUT    = os.path.join(IBGE_DIR, "mca_demographics_panel.csv")

SIDRA_BASE = "https://servicodados.ibge.gov.br/api/v3/agregados"
HEADERS    = {"User-Agent": "research/pedro.feijodemoraes@yale.edu"}

START_YEAR = 2013
END_YEAR   = 2024

# Cache directory for large census age-group downloads.
# If the IBGE census API returns 500 (common for large table requests),
# the code falls back to any previously cached file at this path.
# You can also drop a manually-downloaded CSV here to skip the API entirely:
#   IBGE/census_age_9514_2022_raw.csv  (2022 Census, Table 9514)
#   IBGE/census_age_1552_2010_raw.csv  (2010 Census, Table 1552)
# Required columns: municipio_code (int), age_label (str), value (float)
AGE_CACHE_DIR = IBGE_DIR

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "ibge_demographics_panel",
        {
            "mca_csv": MCA_CSV,
            "output_csv": OUTPUT,
            "age_cache_dir": AGE_CACHE_DIR,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    MCA_CSV = _paths["mca_csv"]
    OUTPUT = _paths["output_csv"]
    AGE_CACHE_DIR = _paths["age_cache_dir"]

# Specific SIDRA category IDs for each census age table.
# Requesting only needed categories avoids the HTTP 500 that SIDRA returns
# when N6[all] is combined with a full [all] classification (too large).
# Categories: Total + 5-year groups for 0-4, 5-9, 10-14, 65-69, ..., 100+
_AGE_CATS_2022 = "100362,93070,93084,93085,93096,93097,93098,49108,49109,60040,60041,6653"
_AGE_CATS_2010 = "0,93070,93084,93085,93096,93097,93098,93099,93100,6653"


def _age_cache_path(tabela: int, year: int) -> str:
    return os.path.join(AGE_CACHE_DIR, f"census_age_{tabela}_{year}_raw.csv")


def _load_age_cache(tabela: int, year: int) -> pd.DataFrame | None:
    path = _age_cache_path(tabela, year)
    if os.path.exists(path):
        logging.info(f"Loading census age cache: {path}")
        return pd.read_csv(
            path,
            dtype={"municipio_code": int, "age_label": str},
            encoding="latin-1",
        )
    return None


def _save_age_cache(df: pd.DataFrame, tabela: int, year: int) -> None:
    path = _age_cache_path(tabela, year)
    df[["municipio_code", "age_label", "value"]].to_csv(
        path, index=False, encoding="latin-1"
    )
    logging.info(f"Saved census age cache: {path}")


def _fetch_valid_munis(tabela: int) -> set[int]:
    """
    Return the set of municipality codes that are present in the given SIDRA table.
    Called once per census table to pre-filter our crosswalk so that municipalities
    created after the census (and thus absent from SIDRA) are excluded -- those codes
    cause HTTP 500 for any batch that includes them.
    """
    url = f"{SIDRA_BASE}/{tabela}/localidades/N6"
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return {int(item["id"]) for item in resp.json()}


def _fetch_age_batch(mun_batch: list[int], tabela: int, year: int, variavel: int,
                     class_id: int, categories: str,
                     retries: int = 4, pause: float = 3.0) -> pd.DataFrame:
    """
    Fetch census age data for a batch of municipalities with specific category IDs.
    Batching is required because N6[all] with any classification causes HTTP 500.
    Returns DataFrame with columns: municipio_code (int), age_label (str), value (float).
    """
    mun_str = ",".join(str(m) for m in mun_batch)
    url = (
        f"{SIDRA_BASE}/{tabela}/periodos/{year}/variaveis/{variavel}"
        f"?localidades=N6[{mun_str}]&classificacao={class_id}[{categories}]"
    )
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=120)
            if resp.status_code in (429, 500, 503):
                wait = (2 ** attempt) * pause
                logging.warning(
                    f"HTTP {resp.status_code} table {tabela} year {year} "
                    f"(attempt {attempt+1}/{retries}); waiting {wait:.0f}s ..."
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep((2 ** attempt) * pause)

    # After the retry loop, resp holds the last response. If it was never
    # successful (i.e. we never hit `break`), raise so the caller skips the batch.
    if resp.status_code != 200:
        raise requests.HTTPError(
            f"HTTP {resp.status_code} for table {tabela} year {year} "
            f"after {retries} attempts",
            response=resp,
        )

    rows = []
    for obj in resp.json():
        for resultado in obj.get("resultados", []):
            age_label = ""
            for cl in resultado.get("classificacoes", []):
                if str(cl.get("id", "")) == str(class_id):
                    cats_dict = cl.get("categoria", {})
                    age_label = next(iter(cats_dict.values()), "")
                    break
            for series_item in resultado.get("series", []):
                mun_code_str = series_item["localidade"]["id"]
                try:
                    mun_code = int(mun_code_str)
                except ValueError:
                    continue
                val_str = series_item["serie"].get(str(year), "")
                try:
                    val = float(val_str)
                except (ValueError, TypeError):
                    val = np.nan   # "-" or blank -> treat as NaN (zero / suppressed)
                rows.append({
                    "municipio_code": mun_code,
                    "age_label":      age_label,
                    "value":          val,
                })
    return (pd.DataFrame(rows) if rows
            else pd.DataFrame(columns=["municipio_code", "age_label", "value"]))

## -----------------------------------------------------------------------------
## 2) SIDRA API HELPERS
## -----------------------------------------------------------------------------

def _sidra_url(tabela: int, periodos: str, variavel: int) -> str:
    return (
        f"{SIDRA_BASE}/{tabela}/periodos/{periodos}"
        f"/variaveis/{variavel}?localidades=N6[all]"
    )


def fetch_sidra(tabela: int, years: list[int], variavel: int,
                retries: int = 3, pause: float = 2.0) -> pd.DataFrame:
    """
    Fetch all municipalities for the given table / variable / years from SIDRA.
    Returns a tidy DataFrame with columns: municipio_code (int), year (int), value (float).
    Handles HTTP 429 / 503 with exponential back-off.
    Batches years in groups of 8 to avoid URL length limits.
    """
    frames = []
    batch_size = 8

    for i in range(0, len(years), batch_size):
        batch = years[i: i + batch_size]
        periodos = "|".join(str(y) for y in batch)
        url = _sidra_url(tabela, periodos, variavel)

        for attempt in range(retries):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=120)
                if resp.status_code == 429 or resp.status_code == 503:
                    wait = (2 ** attempt) * pause
                    logging.warning(f"HTTP {resp.status_code}; waiting {wait:.0f}s ...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    logging.error(f"Failed fetching table {tabela} years {batch}: {exc}")
                    raise
                time.sleep((2 ** attempt) * pause)

        data = resp.json()
        # SIDRA v3 returns a list of result objects; iterate each variable result
        for obj in data:
            for result in obj.get("resultados", []):
                period_series = result.get("series", [])
                for series_item in period_series:
                    mun_code_str = series_item["localidade"]["id"]
                    try:
                        mun_code = int(mun_code_str)
                    except ValueError:
                        continue

                    for periodo, val_str in series_item["serie"].items():
                        try:
                            val = float(val_str)
                        except (ValueError, TypeError):
                            val = np.nan
                        frames.append({"municipio_code": mun_code,
                                       "year": int(periodo),
                                       "value": val})

        logging.info(f"Table {tabela} var {variavel} years {batch}: {len(frames)} cumulative rows")
        time.sleep(pause)   # polite pacing between batches

    if not frames:
        return pd.DataFrame(columns=["municipio_code", "year", "value"])

    df = pd.DataFrame(frames)
    df = df.drop_duplicates(subset=["municipio_code", "year"])
    return df


## -----------------------------------------------------------------------------
## 3) POPULATION DATA
##    Primary source : Table 6579 (Estimativas de populacao), Variable 9324
##                     Coverage: 2013-2022 in this series
##    Extended to 2023+ via Table 4714 (Projeções da Populacao), Variable 93
## -----------------------------------------------------------------------------

def fetch_population(years: list[int]) -> pd.DataFrame:
    """
    Return municipal population panel for the requested years.
    Stitches together Table 6579 (main estimates) and Table 4714 (projections).
    Falls back to Table 6579 last available year for any year still missing.
    """
    # Table 6579 covers annual estimates; newer estimates are in table 4714.
    # We try 6579 first, then supplement missing years with 4714.
    logging.info("Fetching population -- Table 6579 ...")
    pop_6579 = fetch_sidra(tabela=6579, years=years, variavel=9324)
    pop_6579["source"] = "6579"

    found_years = set(pop_6579["year"].unique())
    missing     = [y for y in years if y not in found_years]

    if missing:
        logging.info(f"Population missing for {missing}; trying Table 9514/4714 ...")
        try:
            pop_ext = fetch_sidra(tabela=4714, years=missing, variavel=93)
            pop_ext["source"] = "4714"
            pop = pd.concat([pop_6579, pop_ext], ignore_index=True)
        except Exception as e:
            logging.warning(f"Extension table failed ({e}); using carry-forward.")
            pop = pop_6579.copy()
    else:
        pop = pop_6579.copy()

    pop = pop.drop_duplicates(subset=["municipio_code", "year"])
    pop.rename(columns={"value": "population"}, inplace=True)
    pop = pop[["municipio_code", "year", "population"]]

    # Carry forward last known population for any years still missing
    # (IBGE publishes estimates with a lag; e.g. 2023 may not yet be in any table)
    still_missing = [y for y in years if y not in set(pop["year"].unique())]
    if still_missing:
        last_known_year = int(pop.dropna(subset=["population"])["year"].max())
        logging.info(
            f"Population still missing for {still_missing}; "
            f"carrying forward from {last_known_year}."
        )
        base = pop[pop["year"] == last_known_year][["municipio_code", "population"]].copy()
        extra_frames = []
        for yr in still_missing:
            tmp = base.copy()
            tmp["year"] = yr
            extra_frames.append(tmp)
        pop = pd.concat([pop] + extra_frames, ignore_index=True)
        pop = pop.drop_duplicates(subset=["municipio_code", "year"])

    return pop


## -----------------------------------------------------------------------------
## 4) GDP DATA
##    Source: Table 5938 (PIB dos Municípios), Variable 37
##            = "Produto interno bruto a preços correntes (Mil Reais)"
##    Coverage: 2002-2021 (published with ~2 year lag)
##    Years beyond last available: carry forward last observed value.
## -----------------------------------------------------------------------------

def fetch_gdp(years: list[int]) -> pd.DataFrame:
    """
    Return municipal GDP panel (total, R$ thousands, current prices).
    Automatically marks imputed (carry-forward) rows.
    """
    logging.info("Fetching GDP -- Table 5938 ...")
    gdp_raw = fetch_sidra(tabela=5938, years=years, variavel=37)
    gdp_raw.rename(columns={"value": "gdp_total_r1000"}, inplace=True)
    gdp_raw["gdp_imputed"] = False

    # Identify the highest published year with real data
    max_real_year = gdp_raw.dropna(subset=["gdp_total_r1000"])["year"].max()
    missing_gdp_years = [y for y in years if y > max_real_year]

    if missing_gdp_years:
        logging.info(
            f"GDP data not available for {missing_gdp_years}; "
            f"carrying forward from {max_real_year}."
        )
        base = gdp_raw[gdp_raw["year"] == max_real_year][["municipio_code", "gdp_total_r1000"]].copy()
        extra_frames = []
        for yr in missing_gdp_years:
            tmp = base.copy()
            tmp["year"] = yr
            tmp["gdp_imputed"] = True
            extra_frames.append(tmp)
        gdp_raw = pd.concat([gdp_raw] + extra_frames, ignore_index=True)

    gdp_raw = gdp_raw.drop_duplicates(subset=["municipio_code", "year"])
    return gdp_raw[["municipio_code", "year", "gdp_total_r1000", "gdp_imputed"]]


## -----------------------------------------------------------------------------
## 5) AGE STRUCTURE FROM CENSUS TABLES
##    Source:
##      Table 9514 (2022 Census): "Pessoas residentes em domicílios particulares
##        ocupados, por grupos de idade" -- Classification 287, Variable 93
##      Table 136  (2010 Census): "Pessoas residentes, por grupos de idade"
##        -- Classification 2, Variable 93
##    Strategy:
##      * Fetch all age-group categories for 2022 and 2010.
##      * Identify 65+ and 0-14 buckets by matching category labels.
##      * Compute municipality-level fraction_65plus and fraction_young.
##      * Linearly interpolate for 2013-2024.
## -----------------------------------------------------------------------------

def fetch_sidra_classified(tabela: int, year: int, variavel: int, class_id: int,
                            retries: int = 5, pause: float = 5.0) -> pd.DataFrame:
    """
    Fetch municipality x age-group data from a SIDRA census table.
    Returns DataFrame with columns: municipio_code (int), age_label (str), value (float).
    The 'classificacao=<class_id>[all]' parameter requests every category; each
    resultados entry corresponds to one category.
    """
    url = (
        f"{SIDRA_BASE}/{tabela}/periodos/{year}/variaveis/{variavel}"
        f"?localidades=N6[all]&classificacao={class_id}[all]"
    )
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=300)
            # Retry on rate-limit, service unavailable, AND server error (500)
            if resp.status_code in (429, 500, 503):
                wait = (2 ** attempt) * pause
                logging.warning(
                    f"HTTP {resp.status_code} on table {tabela} (attempt {attempt+1}/{retries}); "
                    f"waiting {wait:.0f}s ..."
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == retries - 1:
                logging.error(f"Failed fetching table {tabela} age classification: {exc}")
                raise
            time.sleep((2 ** attempt) * pause)

    data = resp.json()
    rows = []
    for obj in data:
        for resultado in obj.get("resultados", []):
            # Extract the category label for this resultado
            classifs = resultado.get("classificacoes", [])
            age_label = ""
            for cl in classifs:
                if str(cl.get("id", "")) == str(class_id):
                    cats = cl.get("categoria", {})
                    # categoria is {cat_id: cat_name}
                    age_label = next(iter(cats.values()), "")
                    break

            for series_item in resultado.get("series", []):
                mun_str = series_item["localidade"]["id"]
                try:
                    mun_code = int(mun_str)
                except ValueError:
                    continue
                val_str = series_item["serie"].get(str(year), "")
                try:
                    val = float(val_str)
                except (ValueError, TypeError):
                    val = np.nan
                rows.append({"municipio_code": mun_code,
                             "age_label": age_label,
                             "value": val})

    logging.info(f"Table {tabela} year {year}: {len(rows):,} age x municipality rows")
    return pd.DataFrame(rows)


def _classify_age_bucket(label: str) -> str | None:
    """
    Map a SIDRA age-group label to one of 'young', 'senior', or None (other).
    Handles both Portuguese bracket labels and the catch-all "65 anos ou mais".
    """
    lbl = label.lower().strip()
    # Exact senior catch-all label
    if "65 anos ou mais" in lbl or "65 ou mais" in lbl:
        return "senior"
    # Bracket-form labels -- any bracket starting at or above 65
    for age_start in (65, 70, 75, 80, 85, 90, 95, 100):
        if f"{age_start} a " in lbl or f"{age_start} anos" in lbl:
            return "senior"
    # Young: 0-14
    for bracket in ("0 a 4", "5 a 9", "10 a 14"):
        if bracket in lbl:
            return "young"
    return None


def fetch_age_structure() -> pd.DataFrame:
    """
    Fetch age-group population by municipality for 2010 and 2022 census years.
    Returns DataFrame: municipio_code (int), year (int), fraction_65plus (float),
                       fraction_young (float)
    Missing municipalities (no data) are dropped; fractions are bounded [0, 1].

    Uses targeted SIDRA category IDs (5-year groups only) and batches municipalities
    in groups of 100 to avoid the HTTP 500 that N6[all] triggers on large tables.
    """
    census_specs = [
        # (tabela, year, variavel, class_id, categories)
        (9514, 2022, 93, 287, _AGE_CATS_2022),   # 2022 Census age groups
        (1552, 2010, 93, 287, _AGE_CATS_2010),   # 2010 Census age groups
    ]

    # Load all municipality codes from the crosswalk once
    all_munis = (
        pd.read_csv(MCA_CSV, usecols=["municipality_code"],
                    dtype={"municipality_code": int})["municipality_code"]
        .unique()
        .tolist()
    )
    batch_size = 100

    all_frames = []
    for spec in census_specs:
        tabela, year, variavel, class_id, categories = spec

        # 1. Check local cache first
        cached = _load_age_cache(tabela, year)
        if cached is not None:
            cached["census_year"] = year
            all_frames.append(cached)
            continue

        # Load municipality list and filter to those present in this census table
        # (municipalities created after the census cause HTTP 500 for any batch
        # that includes them -- pre-filtering eliminates all such errors).
        try:
            valid_munis_in_table = _fetch_valid_munis(tabela)
            excluded = [m for m in all_munis if m not in valid_munis_in_table]
            if excluded:
                logging.info(
                    f"Table {tabela}: excluding {len(excluded)} municipalities "
                    f"not in this census (e.g. post-census creations): {excluded[:5]}"
                )
            fetch_munis = [m for m in all_munis if m in valid_munis_in_table]
        except Exception as exc:
            logging.warning(f"Could not fetch valid muni list for table {tabela}: {exc}. Using all.")
            fetch_munis = all_munis

        # 3. Fetch via SIDRA in municipality batches
        logging.info(f"Fetching census age structure -- Table {tabela} ({year}) ...")
        batch_frames = []
        n_batches = (len(fetch_munis) + batch_size - 1) // batch_size
        for i in range(0, len(fetch_munis), batch_size):
            batch = fetch_munis[i: i + batch_size]
            batch_num = i // batch_size + 1
            try:
                df_batch = _fetch_age_batch(batch, tabela, year, variavel,
                                             class_id, categories)
                batch_frames.append(df_batch)
            except Exception as exc:
                logging.warning(
                    f"Table {tabela} ({year}) batch {batch_num}/{n_batches} failed: {exc}"
                )
            if i + batch_size < len(all_munis):
                time.sleep(0.5)   # polite pacing between batches

        if not batch_frames:
            logging.error(
                f"Skipping Table {tabela} ({year}): all batches failed.\n"
                f"  Tip: place a manually-downloaded CSV at "
                f"{_age_cache_path(tabela, year)} to bypass the API."
            )
            continue

        df = pd.concat(batch_frames, ignore_index=True)
        df["census_year"] = year
        _save_age_cache(df, tabela, year)
        all_frames.append(df)
        logging.info(f"Table {tabela} ({year}): {len(df):,} rows fetched.")
        time.sleep(3)   # pause between the two large census table requests

    if not all_frames:
        logging.warning("No census age data retrieved; age structure will be NaN.")
        return pd.DataFrame(columns=["municipio_code", "year",
                                     "fraction_65plus", "fraction_young"])

    census_df = pd.concat(all_frames, ignore_index=True)

    # Label each row as "senior", "young", "total", or "other"
    census_df["bucket"] = census_df["age_label"].apply(_classify_age_bucket)

    # Identify the "Total" row (label is blank or exactly "Total")
    def is_total(lbl: str) -> bool:
        l = lbl.lower().strip()
        return l in ("", "total", "total da população")

    census_df["is_total"] = census_df["age_label"].apply(is_total)

    results = []
    for (mun_code, census_year), g in census_df.groupby(["municipio_code", "census_year"]):
        total_rows = g[g["is_total"]]["value"]
        senior_rows = g[g["bucket"] == "senior"]["value"]
        young_rows  = g[g["bucket"] == "young"]["value"]

        total  = total_rows.sum() if len(total_rows) > 0 else g["value"].sum()
        senior = senior_rows.sum()
        young  = young_rows.sum()

        if total <= 0 or np.isnan(total):
            continue

        results.append({
            "municipio_code": mun_code,
            "year":           census_year,
            "fraction_65plus": min(1.0, senior / total),
            "fraction_young":  min(1.0, young  / total),
        })

    age_df = pd.DataFrame(results)
    logging.info(
        f"Age structure: {len(age_df):,} municipality x census-year observations."
    )
    return age_df


def interpolate_age_structure(age_census: pd.DataFrame,
                               years: list[int]) -> pd.DataFrame:
    """
    Linearly interpolate fraction_65plus and fraction_young for all non-census years
    in `years`, using 2010 and 2022 as anchor points.

    Returns DataFrame: municipio_code, year, fraction_65plus, fraction_young,
                       age_interpolated (True for non-census years).
    """
    if age_census.empty:
        # Return NaN-filled stub so the rest of the pipeline still runs
        mca = pd.read_csv(MCA_CSV, usecols=["municipality_code"],
                          dtype={"municipality_code": int}).drop_duplicates()
        stub_rows = []
        for yr in years:
            tmp = mca.copy()
            tmp["year"] = yr
            tmp["fraction_65plus"] = np.nan
            tmp["fraction_young"]  = np.nan
            tmp["age_interpolated"] = True
            stub_rows.append(tmp)
        return pd.concat(stub_rows, ignore_index=True).rename(
            columns={"municipality_code": "municipio_code"})

    census_years = sorted(age_census["year"].unique())  # typically [2010, 2022]
    y_lo = min(census_years)
    y_hi = max(census_years)

    munis = age_census["municipio_code"].unique()
    rows  = []

    # Pivot to wide so we have one row per municipality
    wide = age_census.pivot(index="municipio_code", columns="year",
                            values=["fraction_65plus", "fraction_young"])
    wide.columns = [f"{v}_{y}" for v, y in wide.columns]
    wide = wide.reset_index()

    for yr in years:
        is_census = yr in census_years
        tmp = wide.copy()
        tmp["year"] = yr

        if is_census:
            tmp["fraction_65plus"]  = tmp.get(f"fraction_65plus_{yr}", np.nan)
            tmp["fraction_young"]   = tmp.get(f"fraction_young_{yr}", np.nan)
            tmp["age_interpolated"] = False
        else:
            # Linear interpolation weight
            y_lo_c = max([cy for cy in census_years if cy <= yr], default=y_lo)
            y_hi_c = min([cy for cy in census_years if cy >= yr], default=y_hi)

            if y_lo_c == y_hi_c:
                w = 1.0
            else:
                w = (yr - y_lo_c) / (y_hi_c - y_lo_c)   # weight on y_hi_c

            col_lo_65 = f"fraction_65plus_{y_lo_c}"
            col_hi_65 = f"fraction_65plus_{y_hi_c}"
            col_lo_yg = f"fraction_young_{y_lo_c}"
            col_hi_yg = f"fraction_young_{y_hi_c}"

            def safe_interp(lo, hi, w):
                if lo in tmp.columns and hi in tmp.columns:
                    return tmp[lo] * (1 - w) + tmp[hi] * w
                elif lo in tmp.columns:
                    return tmp[lo]
                elif hi in tmp.columns:
                    return tmp[hi]
                else:
                    return np.nan

            tmp["fraction_65plus"]  = safe_interp(col_lo_65, col_hi_65, w)
            tmp["fraction_young"]   = safe_interp(col_lo_yg, col_hi_yg, w)
            tmp["age_interpolated"] = True

        rows.append(tmp[["municipio_code", "year",
                          "fraction_65plus", "fraction_young", "age_interpolated"]])

    result = pd.concat(rows, ignore_index=True)
    return result


## -----------------------------------------------------------------------------
## 6) AGGREGATE MUNICIPALITY -> MCA
## -----------------------------------------------------------------------------

def aggregate_to_mca(pop: pd.DataFrame, gdp: pd.DataFrame,
                     age: pd.DataFrame) -> pd.DataFrame:
    """
    1. Loads the MCA crosswalk (municipality_code x year -> mca_code).
    2. Merges population, GDP, and age structure onto it.
    3. Aggregates to MCA x year:
         pop_total        = sum of constituent municipality populations
         gdp_total_r1000  = sum of constituent municipal GDPs in R$ thousands
         gdp_per_capita   = (gdp_total_r1000 * 1000) / pop_total   [in R$]
         fraction_65plus  = population-weighted mean of municipality fraction_65plus
         fraction_young   = population-weighted mean of municipality fraction_young
         n_municipalities = count of distinct municipalities
         gdp_imputed      = True if any constituent municipality used carry-forward GDP
         age_interpolated = True if any constituent municipality used interpolated age
    """
    logging.info("Loading MCA crosswalk ...")
    mca = pd.read_csv(MCA_CSV, usecols=["municipality_code", "mca_code", "year"],
                      dtype={"municipality_code": int, "mca_code": str, "year": int})
    mca = mca[mca["year"].between(START_YEAR, END_YEAR)].copy()

    # Merge population
    merged = mca.merge(pop, left_on=["municipality_code", "year"],
                       right_on=["municipio_code", "year"], how="left")
    merged.drop(columns=["municipio_code"], errors="ignore", inplace=True)

    # Merge GDP
    merged = merged.merge(gdp, left_on=["municipality_code", "year"],
                          right_on=["municipio_code", "year"], how="left")
    merged.drop(columns=["municipio_code"], errors="ignore", inplace=True)

    # Merge age structure
    if not age.empty:
        merged = merged.merge(age, left_on=["municipality_code", "year"],
                              right_on=["municipio_code", "year"], how="left")
        merged.drop(columns=["municipio_code"], errors="ignore", inplace=True)
    else:
        merged["fraction_65plus"]  = np.nan
        merged["fraction_young"]   = np.nan
        merged["age_interpolated"] = True

    # Population-weighted helpers for age shares.
    # Multiply population only where the fraction is non-NaN, so that MCAs with
    # completely missing age data produce NaN (not 0.0) in the output.
    merged["pop_x_65plus"]    = merged["population"] * merged["fraction_65plus"]
    merged["pop_x_young"]     = merged["population"] * merged["fraction_young"]
    # Track how much population actually had age data (denominator for weighted mean)
    merged["pop_with_age"]    = np.where(merged["fraction_65plus"].notna(),
                                          merged["population"], np.nan)

    # Aggregate
    agg = (
        merged.groupby(["mca_code", "year"])
              .agg(
                  pop_total        = ("population",      "sum"),
                  gdp_total_r1000  = ("gdp_total_r1000", "sum"),
                  n_municipalities = ("municipality_code", "nunique"),
                  gdp_imputed      = ("gdp_imputed",     lambda x: x.any()),
                  age_interpolated = ("age_interpolated", lambda x: x.any()),
                  pop_x_65plus_sum = ("pop_x_65plus",    "sum"),
                  pop_x_young_sum  = ("pop_x_young",     "sum"),
                  pop_with_age_sum = ("pop_with_age",    "sum"),
              )
              .reset_index()
    )

    # Population-weighted age fractions; NaN when no municipality had age data
    agg["fraction_65plus"] = np.where(
        agg["pop_with_age_sum"] > 0,
        agg["pop_x_65plus_sum"] / agg["pop_with_age_sum"],
        np.nan,
    )
    agg["fraction_young"] = np.where(
        agg["pop_with_age_sum"] > 0,
        agg["pop_x_young_sum"] / agg["pop_with_age_sum"],
        np.nan,
    )
    agg.drop(columns=["pop_x_65plus_sum", "pop_x_young_sum", "pop_with_age_sum"],
             inplace=True)

    # GDP per capita in R$
    agg["gdp_per_capita"] = np.where(
        agg["pop_total"] > 0,
        (agg["gdp_total_r1000"] * 1000) / agg["pop_total"],
        np.nan,
    )

    # Replace 0 aggregates with NaN (no data)
    for col in ["pop_total", "gdp_total_r1000"]:
        agg.loc[agg[col] == 0, col] = np.nan

    # Reorder columns
    col_order = ["mca_code", "year", "pop_total", "gdp_total_r1000", "gdp_per_capita",
                 "fraction_65plus", "fraction_young",
                 "n_municipalities", "gdp_imputed", "age_interpolated"]
    agg = agg[col_order]

    agg.sort_values(["mca_code", "year"], inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg


## -----------------------------------------------------------------------------
## 7) MAIN
## -----------------------------------------------------------------------------

def main():
    # Early exit: if output already exists, skip the full rebuild
    if os.path.exists(OUTPUT) and os.path.getsize(OUTPUT) > 0:
        print(f"Output already exists, skipping: {OUTPUT}")
        logging.info(f"Output already exists -- skipping rebuild: {OUTPUT}")
        return

    years = list(range(START_YEAR, END_YEAR + 1))

    # Fetch population and GDP concurrently (different SIDRA tables -> safe to
    # parallelise).  Age structure is fetched AFTER they finish: the batched
    # age requests generate heavy API load and interleaving them with pop/GDP
    # requests causes intermittent HTTP 500 errors from SIDRA.
    logging.info("Fetching population and GDP in parallel ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future_pop = pool.submit(fetch_population, years)
        future_gdp = pool.submit(fetch_gdp,        years)
        pop        = future_pop.result()
        gdp        = future_gdp.result()

    logging.info("Fetching census age structure (sequential, after pop/GDP) ...")
    age_census = fetch_age_structure()

    logging.info(f"Population rows: {len(pop):,}  years: {sorted(pop['year'].unique())}")
    logging.info(f"GDP rows: {len(gdp):,}  years: {sorted(gdp['year'].unique())}")

    age = interpolate_age_structure(age_census, years)
    logging.info(f"Age structure rows: {len(age):,}  interpolated: {age['age_interpolated'].sum():,}")

    panel = aggregate_to_mca(pop, gdp, age)
    logging.info(f"MCA panel: {len(panel):,} rows  ({panel['mca_code'].nunique()} MCAs)")

    panel.to_csv(OUTPUT, index=False, encoding="latin-1")
    logging.info(f"Saved to {OUTPUT}")

    print(
        f"\nDemographics panel summary"
        f"\n  Rows:              {len(panel):,}"
        f"\n  MCAs:              {panel['mca_code'].nunique()}"
        f"\n  Years:             {panel['year'].min()} - {panel['year'].max()}"
        f"\n  GDP imputed:       {panel['gdp_imputed'].sum():,} rows"
        f"\n  Age interpolated:  {panel['age_interpolated'].sum():,} rows"
        f"\n  fraction_65plus:   {panel['fraction_65plus'].notna().sum():,} non-null"
        f"\n  Output:            {OUTPUT}"
    )
    print(panel.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
