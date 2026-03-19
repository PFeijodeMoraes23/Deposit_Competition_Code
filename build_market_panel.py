## build_market_panel.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-03-06
#
# Purpose: Merge all MCA-level characteristic panels with the conglomerate
#          deposit panel to produce a single analysis-ready dataset for
#          the deposit competition / BLP demand model.
#
#   Inputs  (all built by earlier pipeline scripts):
#     A. BCB/Panel/deposits_panel.csv
#          Conglomerate x CODMUN_IBGE (7-digit) x Year x Quarter
#          Core left-hand-side data (deposit balances by type, source)
#
#     B. IBGE/muni_mca_regions_2010_2024_panel.csv
#          Municipality -> MCA crosswalk used to map CODMUN_IBGE -> mca_code
#
#     C. IBGE/mca_demographics_panel.csv
#          MCA x Year : pop_total, gdp_per_capita, fraction_65plus, fraction_young
#
#     D. BCB/PIX/pix_mca_panel.csv
#          MCA x Year x Quarter : pix_active, pix_users_pf_per1000, pix_txns_pf, ...
#
#     E. ANATEL/anatel_mca_panel.csv
#          MCA x Year x Quarter : connections_per100, frac_4g5g
#
#     F. INSS/inss_mca_panel.csv
#          MCA x Year x Quarter : retirees_per1000
#
#     G. BCB/Inclusion/bcb_inclusion_mca_panel.csv
#          MCA x Year : branches_per1000, correspondents_per1000, access_points_per1000
#
#     H. CadUnico/cadunico_mca_panel.csv
#          MCA x Year x Quarter : cadunico_families_per1000, cadunico_extreme_poverty
#
#   Output:
#     BCB/Panel/market_panel.csv
#       Unit of observation: CodConglomeradoPrudencial x mca_code x Year x Quarter
#
#       Deposit columns (from A):
#         dep_a1, dep_a2, dep_a3, dep_a4, dep_outros, dep_a5
#
#       Market-level columns (from C-H):
#         pop_total, gdp_per_capita
#         fraction_65plus, fraction_young, age_interpolated
#         pix_active, pix_users_pf_per1000, pix_txns_pf, pix_value_pf_r1000
#         connections_per100, frac_4g5g
#         retirees_per1000
#         branches_per1000, correspondents_per1000, access_points_per1000
#         cadunico_families_per1000, cadunico_extreme_poverty
#
#       Identifier / flag columns:
#         CodConglomeradoPrudencial, CNPJ_Lider, NomeInstituicao
#         CODMUN_IBGE (original 7-digit code)
#         mca_code, Year, Quarter, Source (ESTBAN | IF_DATA)
#
# Notes:
#   * Tier-1 rows (ESTBAN, CODMUN_IBGE ≠ 0) are merged to market characteristics
#     via the MCA crosswalk.  Tier-2 rows (IF Data only, CODMUN_IBGE = 0) are
#     kept as-is; their market-characteristic columns are left NaN (they appear
#     only in the aggregation-moment equation, not the local-market equations).
#   * Annual-frequency panels (C = demographics, G = BCB inclusion) are broadcast
#     to all four quarters of each year before merging.
#   * Missing panel files are handled gracefully; their columns are filled with NaN
#     and logged as warnings.  This lets the panel be built incrementally as
#     downloads complete.
###-----------------------------------------------------------------------------

import concurrent.futures
import os
import sys
import logging
import numpy as np
import pandas as pd

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# Keep the default console encoding (CP1252 / Latin-1 on Windows).
# All print strings below use plain ASCII to avoid garbled output.

## -----------------------------------------------------------------------------
## 1) PATHS
## -----------------------------------------------------------------------------
BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

BCB_DIR    = os.path.join(BASE, "BCB")
IBGE_DIR   = os.path.join(BASE, "IBGE")
ANATEL_DIR = os.path.join(BASE, "ANATEL")
INSS_DIR   = os.path.join(BASE, "INSS")
CAD_DIR    = os.path.join(BASE, "CadUnico")

PANEL_DIR  = os.path.join(BCB_DIR, "Egan_et_al_2025_Rep", "processed")
os.makedirs(PANEL_DIR, exist_ok=True)

DEPOSITS_CSV   = os.path.join(PANEL_DIR,                  "deposits_panel.csv")
MCA_XWALK_CSV  = os.path.join(IBGE_DIR,                   "muni_mca_regions_2010_2024_panel.csv")
DEMO_CSV       = os.path.join(IBGE_DIR,                   "mca_demographics_panel.csv")
PIX_CSV        = os.path.join(BCB_DIR, "PIX",             "pix_mca_panel.csv")
ANATEL_CSV     = os.path.join(ANATEL_DIR,                 "anatel_mca_panel.csv")
INSS_CSV       = os.path.join(INSS_DIR,                   "inss_mca_panel.csv")
INCLUSION_CSV  = os.path.join(BCB_DIR, "Inclusion",       "bcb_inclusion_mca_panel.csv")
CADUNICO_CSV   = os.path.join(CAD_DIR,                    "cadunico_mca_panel.csv")
TARIFAS_FEE_CSV = os.path.join(BCB_DIR, "Tarifas", "processed", "tarifas_fee_summary.csv")
BANK_CHARS_CSV = os.path.join(PANEL_DIR,                  "bank_chars_panel.csv")

OUTPUT_CSV     = os.path.join(PANEL_DIR,                  "market_panel.csv")

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "build_market_panel",
        {
            "deposits_csv": DEPOSITS_CSV,
            "mca_xwalk_csv": MCA_XWALK_CSV,
            "demo_csv": DEMO_CSV,
            "pix_csv": PIX_CSV,
            "anatel_csv": ANATEL_CSV,
            "inss_csv": INSS_CSV,
            "inclusion_csv": INCLUSION_CSV,
            "cadunico_csv": CADUNICO_CSV,
            "tarifas_fee_csv": TARIFAS_FEE_CSV,
            "bank_chars_csv": BANK_CHARS_CSV,
            "output_csv": OUTPUT_CSV,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    DEPOSITS_CSV = _paths["deposits_csv"]
    MCA_XWALK_CSV = _paths["mca_xwalk_csv"]
    DEMO_CSV = _paths["demo_csv"]
    PIX_CSV = _paths["pix_csv"]
    ANATEL_CSV = _paths["anatel_csv"]
    INSS_CSV = _paths["inss_csv"]
    INCLUSION_CSV = _paths["inclusion_csv"]
    CADUNICO_CSV = _paths["cadunico_csv"]
    TARIFAS_FEE_CSV = _paths["tarifas_fee_csv"]
    BANK_CHARS_CSV = _paths["bank_chars_csv"]
    OUTPUT_CSV = _paths["output_csv"]


## -----------------------------------------------------------------------------
## 2) HELPERS
## -----------------------------------------------------------------------------

def _try_load(path: str, desc: str, **kwargs) -> pd.DataFrame | None:
    """Load a CSV if it exists; log a warning and return None otherwise."""
    if not os.path.exists(path):
        logging.warning(f"File not found (will be NaN in output): {path}  [{desc}]")
        return None
    kwargs.setdefault("encoding", "latin-1")
    df = pd.read_csv(path, **kwargs)
    logging.info(f"Loaded {desc}: {len(df):,} rows")
    return df


def _broadcast_annual_to_quarters(df: pd.DataFrame,
                                   year_col: str = "year") -> pd.DataFrame:
    """
    Given an annual panel, create four copies -- one per quarter -- and return
    a long DataFrame with an added 'quarter' column (1-4).
    """
    frames = []
    for q in (1, 2, 3, 4):
        tmp = df.copy()
        tmp["quarter"] = q
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


## -----------------------------------------------------------------------------
## 3) LOAD & PREPARE DEPOSITS PANEL
## -----------------------------------------------------------------------------

def load_deposits() -> pd.DataFrame:
    dep = pd.read_csv(DEPOSITS_CSV, low_memory=False,
                      dtype={"CODMUN_IBGE": "Int64",
                             "CodConglomeradoPrudencial": str,
                             "CNPJ_Lider": str})
    # Normalise column names to lower-case year/quarter for consistent merging
    dep.rename(columns={"Year": "year", "Quarter": "quarter"}, inplace=True)
    dep["year"]    = dep["year"].astype(int)
    dep["quarter"] = dep["quarter"].astype(int)
    logging.info(f"Loaded deposits panel: {len(dep):,} rows  "
                 f"({dep['CodConglomeradoPrudencial'].nunique()} conglomerates  "
                 f"years {dep['year'].min()}-{dep['year'].max()})")
    return dep


## -----------------------------------------------------------------------------
## 4) ADD mca_code TO DEPOSIT PANEL (Tier-1 rows only)
## -----------------------------------------------------------------------------

def attach_mca_code(dep: pd.DataFrame) -> pd.DataFrame:
    """
    Map CODMUN_IBGE (7-digit) -> mca_code using the MCA crosswalk.
    Tier-1 (ESTBAN) rows have a real municipality code.
    Tier-2 (IF Data national, CODMUN_IBGE = 0) get mca_code = 'NATIONAL'.
    """
    xwalk = pd.read_csv(MCA_XWALK_CSV,
                        usecols=["municipality_code", "mca_code", "year"],
                        dtype={"municipality_code": int, "mca_code": str, "year": int})

    dep = dep.copy()

    # Tier-2 sentinel
    dep["mca_code"] = "NATIONAL"

    # Merge mca_code for Tier-1 rows (CODMUN_IBGE is a real 7-digit code)
    tier1_mask = dep["CODMUN_IBGE"].notna() & (dep["CODMUN_IBGE"] != 0)

    tier1 = dep[tier1_mask].copy()
    tier1["CODMUN_IBGE_int"] = tier1["CODMUN_IBGE"].astype(int)

    tier1_merged = tier1.merge(
        xwalk.rename(columns={"municipality_code": "CODMUN_IBGE_int"}),
        on=["CODMUN_IBGE_int", "year"],
        how="left",
        suffixes=("", "_xwalk"),
    )

    n_missing = tier1_merged["mca_code_xwalk"].isna().sum()
    if n_missing > 0:
        logging.warning(
            f"{n_missing:,} Tier-1 deposit rows could not be matched to an MCA "
            f"(municipality not in crosswalk for that year)."
        )

    # Write resolved mca_code back to the main df
    tier1_merged["mca_code"] = tier1_merged["mca_code_xwalk"].fillna("UNKNOWN")
    tier1_merged.drop(columns=["mca_code_xwalk", "CODMUN_IBGE_int"], inplace=True)

    # Recombine
    tier2 = dep[~tier1_mask].copy()
    result = pd.concat([tier1_merged, tier2], ignore_index=True)

    logging.info(
        f"MCA mapping: {result['mca_code'].nunique()} unique mca_code values  "
        f"(including NATIONAL={( result['mca_code'] == 'NATIONAL').sum():,} rows)"
    )
    return result


## -----------------------------------------------------------------------------
## 5) LOAD MARKET CHARACTERISTIC PANELS
## -----------------------------------------------------------------------------

def load_demographics() -> pd.DataFrame | None:
    """Annual MCA panel: pop_total, gdp_per_capita, fraction_65plus, fraction_young."""
    df = _try_load(DEMO_CSV, "IBGE demographics",
                  dtype={"mca_code": str, "year": int})
    if df is None:
        return None
    # Broadcast to quarters (annual stock variable)
    df = _broadcast_annual_to_quarters(df)
    keep = ["mca_code", "year", "quarter",
            "pop_total", "gdp_per_capita",
            "fraction_65plus", "fraction_young", "age_interpolated",
            "gdp_imputed"]
    return df[[c for c in keep if c in df.columns]]


def load_pix() -> pd.DataFrame | None:
    """Quarterly MCA panel: pix adoption."""
    df = _try_load(PIX_CSV, "PIX MCA panel",
                  dtype={"mca_code": str, "year": int, "quarter": int})
    if df is None:
        return None
    keep = ["mca_code", "year", "quarter",
            "pix_active", "pix_users_pf_per1000",
            "pix_txns_pf", "pix_value_pf_r1000", "pix_users_pj"]
    return df[[c for c in keep if c in df.columns]]


def load_anatel() -> pd.DataFrame | None:
    """Quarterly MCA panel: mobile connectivity."""
    df = _try_load(ANATEL_CSV, "ANATEL mobile panel",
                  dtype={"mca_code": str, "year": int, "quarter": int})
    if df is None:
        return None
    keep = ["mca_code", "year", "quarter",
            "connections_per100", "frac_4g5g"]
    return df[[c for c in keep if c in df.columns]]


def load_inss() -> pd.DataFrame | None:
    """Quarterly MCA panel: retirement beneficiaries."""
    df = _try_load(INSS_CSV, "INSS retirees panel",
                  dtype={"mca_code": str, "year": int, "quarter": int})
    if df is None:
        return None
    keep = ["mca_code", "year", "quarter",
            "retirees_total", "retirees_per1000"]
    return df[[c for c in keep if c in df.columns]]


def load_bcb_inclusion() -> pd.DataFrame | None:
    """Annual MCA panel: banking access-points (broadcast to quarters)."""
    df = _try_load(INCLUSION_CSV, "BCB inclusion panel",
                  dtype={"mca_code": str, "year": int})
    if df is None:
        return None
    df = _broadcast_annual_to_quarters(df)
    keep = ["mca_code", "year", "quarter",
            "branches_per1000", "correspondents_per1000", "access_points_per1000"]
    return df[[c for c in keep if c in df.columns]]


def load_cadunico() -> pd.DataFrame | None:
    """Quarterly MCA panel: CadUnico low-income families."""
    df = _try_load(CADUNICO_CSV, "CadUnico panel",
                  dtype={"mca_code": str, "year": int, "quarter": int})
    if df is None:
        return None
    keep = ["mca_code", "year", "quarter",
            "cadunico_families_per1000", "cadunico_extreme_poverty", "cadunico_poverty"]
    return df[[c for c in keep if c in df.columns]]


## -----------------------------------------------------------------------------
## 6) MERGE EVERYTHING
## -----------------------------------------------------------------------------

def merge_characteristics(dep: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join all market-characteristic panels onto the deposit panel.
    Tier-2 rows (mca_code = 'NATIONAL') receive NaN for all characteristic cols.
    """
    loaders = [
        ("demographics",  load_demographics),
        ("PIX",           load_pix),
        ("ANATEL",        load_anatel),
        ("INSS",          load_inss),
        ("BCB inclusion", load_bcb_inclusion),
        ("CadUnico",      load_cadunico),
    ]

    # Load all six characteristic panels concurrently (all are local CSV reads)
    loaded: dict[str, pd.DataFrame | None] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        future_map = {pool.submit(fn): name for name, fn in loaders}
        for future in concurrent.futures.as_completed(future_map):
            name = future_map[future]
            try:
                loaded[name] = future.result()
            except Exception as exc:
                logging.warning(f"  {name} loading error: {exc}")
                loaded[name] = None

    # Maintain original order for deterministic merge sequence
    panels = [(name, loaded.get(name)) for name, _ in loaders]

    result = dep.copy()

    for name, char_df in panels:
        if char_df is None:
            logging.warning(f"  {name} panel missing -- all its columns will be NaN.")
            continue

        char_df = char_df.copy()
        char_df["mca_code"] = char_df["mca_code"].astype(str)
        char_df["year"]     = char_df["year"].astype(int)
        char_df["quarter"]  = char_df["quarter"].astype(int)

        # Drop any columns already in result (except merge keys) to avoid _x/_y
        existing = set(result.columns) - {"mca_code", "year", "quarter"}
        char_df  = char_df.drop(columns=[c for c in char_df.columns
                                          if c in existing], errors="ignore")

        result = result.merge(char_df, on=["mca_code", "year", "quarter"],
                              how="left")
        logging.info(f"  Merged {name}: result now {len(result):,} rows "
                     f"x {len(result.columns)} cols")

    return result


## ---------------------------------------------------------------------------------
## 6.5) LOAD AND MERGE TARIFF FEE SUMMARY
## ---------------------------------------------------------------------------------
#
# Tariff fees are conglomerate-level product characteristics (not MCA-level).
# They are joined on CodConglomeradoPrudencial x year x quarter using the
# most recent tariff snapshot available on or before the quarter’s end date.
# With only a single snapshot the join degenerates to a simple left-merge.
# ---------------------------------------------------------------------------------

def load_tarifas() -> pd.DataFrame | None:
    """Load the fee summary panel produced by tarifas_scrape_1.py."""
    if not os.path.exists(TARIFAS_FEE_CSV):
        logging.warning(
            f"Tarifas fee summary not found -- fee columns will be NaN: {TARIFAS_FEE_CSV}"
        )
        return None
    df = pd.read_csv(TARIFAS_FEE_CSV, dtype={"cod_cong_prudencial": str})
    df["data_coleta"] = pd.to_datetime(df["data_coleta"], errors="coerce")
    logging.info(
        f"Loaded tarifas fee summary: {len(df):,} rows  "
        f"({df['cod_cong_prudencial'].nunique()} conglomerates  "
        f"{len(df.columns) - 2} fee columns)"
    )
    return df


def merge_tarifas(panel: pd.DataFrame, tarifs: pd.DataFrame | None) -> pd.DataFrame:
    """
    Left-join tariff fee columns onto the market panel on CodConglomeradoPrudencial.

    Time matching
    -------------
    For each panel (year, quarter), the most recent scrape whose data_coleta falls
    on or before the quarter’s end date is used.  With a single snapshot this is
    trivially the one available snapshot.  When multiple scrapes exist the join
    becomes a time-varying last-observation-carried-forward.
    """
    if tarifs is None or tarifs.empty:
        return panel

    # Build a quarter-end date for each (year, quarter) in the panel
    qends = (
        panel[["year", "quarter"]]
        .drop_duplicates()
        .copy()
    )
    # Quarter Q ends on month Q*3, last day: use period arithmetic
    qends["qend"] = pd.PeriodIndex(
        qends["year"].astype(str) + "Q" + qends["quarter"].astype(str), freq="Q"
    ).to_timestamp(how="end").normalize()

    # For each (CodConglPrud, year, quarter), pick the most recent scrape date
    # that is <= qend.  Build a merge table: (CodConglPrud, year, quarter) -> fees.
    snapshots = tarifs["data_coleta"].sort_values().unique()  # sorted asc
    as_of_rows: list[dict] = []
    fee_cols = [c for c in tarifs.columns if c.startswith("fee_")]

    for _, qrow in qends.iterrows():
        valid = [s for s in snapshots if s <= qrow["qend"]]
        if not valid:
            continue
        snap = max(valid)  # most recent snapshot <= quarter end
        snap_fees = tarifs[tarifs["data_coleta"] == snap].copy()
        snap_fees = snap_fees.drop(columns=["data_coleta"])
        snap_fees["year"]    = qrow["year"]
        snap_fees["quarter"] = qrow["quarter"]
        as_of_rows.append(snap_fees)

    if not as_of_rows:
        logging.warning("merge_tarifas: no valid snapshot found for any panel quarter.")
        return panel

    as_of_df = pd.concat(as_of_rows, ignore_index=True)
    as_of_df = as_of_df.rename(columns={"cod_cong_prudencial": "CodConglomeradoPrudencial"})
    as_of_df["CodConglomeradoPrudencial"] = as_of_df["CodConglomeradoPrudencial"].astype(str)

    before = len(panel)
    panel = panel.merge(
        as_of_df,
        on=["CodConglomeradoPrudencial", "year", "quarter"],
        how="left",
    )
    assert len(panel) == before, "merge_tarifas unexpectedly duplicated rows"

    matched = panel[fee_cols[0]].notna().sum() if fee_cols else 0
    logging.info(
        f"  Tarifas merged: {len(fee_cols)} fee columns  "
        f"{matched:,}/{len(panel):,} rows matched"
    )
    return panel


## -----------------------------------------------------------------------------
## 7) FILL NATIONAL ROWS WITH POPULATION-WEIGHTED NATIONAL AVERAGES
## -----------------------------------------------------------------------------
#
# Fintechs / digital banks (Tier-2 rows, mca_code = 'NATIONAL') operate
# nationally and have no single local market.  For the BLP demand estimation
# they need well-defined market-characteristic controls and instruments.
# We assign each variable a population-weighted national average computed from
# Tier-1 (ESTBAN) rows for the same (year, quarter), using MCA pop_total as
# weights.  Two exceptions use sums instead of weighted means:
#   * pop_total      -> national population (sum of all MCAs)
#   * retirees_total -> national retiree count (sum of all MCAs)
# -----------------------------------------------------------------------------

# Columns aggregated as national total (sum)
_NATL_SUM_COLS: list[str] = ["pop_total", "retirees_total"]

# Columns aggregated as population-weighted mean
_NATL_RATE_COLS: list[str] = [
    "gdp_per_capita",
    "fraction_65plus", "fraction_young",
    "pix_users_pf_per1000", "pix_txns_pf", "pix_value_pf_r1000", "pix_users_pj",
    "connections_per100", "frac_4g5g",
    "retirees_per1000",
    "branches_per1000", "correspondents_per1000", "access_points_per1000",
    "cadunico_families_per1000", "cadunico_extreme_poverty", "cadunico_poverty",
]


def fill_national_averages(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For Tier-2 rows (mca_code = 'NATIONAL'), replace NaN market-characteristic
    columns with population-weighted national averages derived from Tier-1 rows.

    Rationale
    ---------
    A fintech that operates nationally faces, in expectation, the average
    Brazilian market conditions weighted by where the population lives.  Giving
    the NATIONAL rows meaningful covariate values (a) allows the same IV/control
    strategy as for local banks and (b) prevents the BLP outside-good share from
    soaking up all macro variation.

    Aggregation rules
    -----------------
    * pop_total, retirees_total  -> sum across all MCAs  (true national total)
    * all rate / per-X columns   -> population-weighted mean across MCAs
    """
    nat_mask = panel["mca_code"] == "NATIONAL"
    if nat_mask.sum() == 0:
        return panel

    tier1 = panel[~nat_mask].copy()
    if tier1.empty or "pop_total" not in tier1.columns:
        logging.warning(
            "fill_national_averages: no Tier-1 rows or pop_total missing -- "
            "NATIONAL rows remain NaN."
        )
        return panel

    # -- deduplicate to one row per MCA x year x quarter for aggregation ----
    # (the deposit panel has many bank rows per MCA; we only need one copy of
    # the market characteristics, so drop-duplication on the key + char columns)
    char_cols = (
        [c for c in _NATL_SUM_COLS + _NATL_RATE_COLS if c in tier1.columns]
    )
    t1 = (
        tier1[["mca_code", "year", "quarter"] + char_cols]
        .drop_duplicates(subset=["mca_code", "year", "quarter"])
        .copy()
    )
    t1["_w"] = t1["pop_total"].fillna(0.0)

    # -- sum columns --------------------------------------------------------
    sum_cols_present = [c for c in _NATL_SUM_COLS if c in t1.columns]
    natl_sums = (
        t1.groupby(["year", "quarter"])[sum_cols_present]
          .sum(min_count=1)
          .reset_index()
    )

    # -- population-weighted means ------------------------------------------
    rate_cols_present = [c for c in _NATL_RATE_COLS if c in t1.columns]
    wavg_rows: list[dict] = []
    for (yr, qt), grp in t1.groupby(["year", "quarter"]):
        row: dict = {"year": yr, "quarter": qt}
        w = grp["_w"].values
        for col in rate_cols_present:
            v = grp[col].values.astype(float)
            mask = ~np.isnan(v) & (w > 0)
            row[col] = float(np.average(v[mask], weights=w[mask])) if mask.any() else np.nan
        wavg_rows.append(row)
    natl_rates = pd.DataFrame(wavg_rows) if wavg_rows else pd.DataFrame(
        columns=["year", "quarter"] + rate_cols_present
    )

    # -- combine sum + rate aggregates ------------------------------------
    natl = natl_sums.merge(natl_rates, on=["year", "quarter"], how="outer")

    # -- fill NaN cells in NATIONAL rows ----------------------------------
    panel = panel.copy()
    fill_cols = [c for c in sum_cols_present + rate_cols_present if c in panel.columns]
    panel_nat = panel[nat_mask].copy().merge(
        natl, on=["year", "quarter"], how="left", suffixes=("", "_natl")
    )
    for col in fill_cols:
        src = col + "_natl"
        if src in panel_nat.columns:
            panel_nat[col] = panel_nat[col].fillna(panel_nat[src])
            panel_nat.drop(columns=[src], inplace=True)

    panel = pd.concat([panel[~nat_mask], panel_nat], ignore_index=True)

    nat_mask2 = panel["mca_code"] == "NATIONAL"
    n_filled  = panel.loc[nat_mask2, fill_cols].notna().sum().sum()
    logging.info(
        f"fill_national_averages: filled {nat_mask.sum():,} NATIONAL rows "
        f"with pop-weighted national aggregates "
        f"across {tier1['mca_code'].nunique()} MCAs "
        f"({n_filled:,} cell values written)"
    )
    return panel


## -----------------------------------------------------------------------------
## 7b) BANK CHARS
## -----------------------------------------------------------------------------

def load_bank_chars() -> pd.DataFrame | None:
    return _try_load(BANK_CHARS_CSV, "Bank chars panel")

def merge_bank_chars(panel: pd.DataFrame, bank_chars: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join bank sizes and characteristics onto the panel on CodConglomeradoPrudencial x year x quarter."""
    if bank_chars is None or bank_chars.empty:
        return panel
    return panel.merge(
        bank_chars,
        on=["CodConglomeradoPrudencial", "year", "quarter"],
        how="left"
    )

## -----------------------------------------------------------------------------
## 8) FINAL COLUMN ORDERING & SAVE
## -----------------------------------------------------------------------------

def calculate_hausman_iv_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate the leave-one-out mean spread for each deposit type (a1 through a5)."""
    for k in [1, 2, 3, 4, 5]:
        spread_col = f'spread_a{k}'
        iv_col = f'leave_one_out_mean_spread_a{k}'
        if spread_col in panel.columns:
            group_cols = ['year', 'quarter']
            g_count = panel.groupby(group_cols)[spread_col].transform(lambda x: x.notnull().sum())
            g_sum = panel.groupby(group_cols)[spread_col].transform(lambda x: x.sum())
            panel[iv_col] = (g_sum - panel[spread_col].fillna(0)) / (g_count - panel[spread_col].notnull().astype(int)).clip(lower=1)
            panel.loc[g_count <= 1, iv_col] = float('nan')
    return panel

_COL_ORDER = [
    # Identifiers
    "CodConglomeradoPrudencial", "CNPJ_Lider", "NomeInstituicao",
    "CODMUN_IBGE", "mca_code",
    "year", "quarter",
    "Source",
    # Deposit balances
    "dep_a1", "dep_a2", "dep_a3", "dep_a4", "dep_outros", "dep_a5",
    "spread_a1", "spread_a2", "spread_a3", "spread_a4", "spread_a5",
    "leave_one_out_mean_spread_a1", "leave_one_out_mean_spread_a2",
    "leave_one_out_mean_spread_a3", "leave_one_out_mean_spread_a4",
    "leave_one_out_mean_spread_a5",
    # Market size
    "pop_total", "gdp_per_capita", "gdp_imputed",
    # Demographics
    "fraction_65plus", "fraction_young", "age_interpolated",
    # Digital adoption
    "pix_active", "pix_users_pf_per1000", "pix_txns_pf", "pix_value_pf_r1000",
    "pix_users_pj",
    # Mobile connectivity
    "connections_per100", "frac_4g5g",
    # Retirees (elderly income dependency)
    "retirees_total", "retirees_per1000",
    # Banking access
    "branches_per1000", "correspondents_per1000", "access_points_per1000",
    # Poverty / unbanked proxy
    "cadunico_families_per1000", "cadunico_extreme_poverty", "cadunico_poverty",
    # Bank characteristics
    "total_assets", "log_total_assets", "equity", "equity_ratio",
    "has_ip", "segment", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
]



def save(panel: pd.DataFrame) -> None:
    # Put columns in desired order, then any remaining columns alphabetically
    ordered   = [c for c in _COL_ORDER if c in panel.columns]
    remaining = sorted([c for c in panel.columns if c not in ordered])
    panel     = panel[ordered + remaining]

    panel.sort_values(
        ["CodConglomeradoPrudencial", "mca_code", "year", "quarter"],
        inplace=True,
    )
    panel.reset_index(drop=True, inplace=True)

    panel.to_csv(OUTPUT_CSV, index=False, encoding="latin-1")
    logging.info(f"Saved market panel to {OUTPUT_CSV}")


## -----------------------------------------------------------------------------
## 9) MAIN
## -----------------------------------------------------------------------------

def main() -> None:
    # A. Load deposit panel
    dep = load_deposits()

    # B. Map CODMUN_IBGE -> mca_code
    dep = attach_mca_code(dep)

    # C. Merge all market characteristic panels
    panel = merge_characteristics(dep)

    # D. Fill NATIONAL (fintech) rows with pop-weighted national averages
    panel = fill_national_averages(panel)

    # D2. Merge tariff fee columns (conglomerate-level, time-matched)
    tarifs = load_tarifas()
    panel = merge_tarifas(panel, tarifs)

    # D3. Merge bank characteristics (total assets, etc.)
    bank_chars = load_bank_chars()
    panel = merge_bank_chars(panel, bank_chars)

    # D4. Calculate the Leave-One-Out Mean Spread (Hausman IV) natively over wide format deposit columns
    panel = calculate_hausman_iv_wide(panel)

    # E. Save
    save(panel)

    # -- Summary ----------------------------------------------------------------
    tier1 = panel[panel["mca_code"] != "NATIONAL"]
    tier2 = panel[panel["mca_code"] == "NATIONAL"]

    print(
        f"\nMarket panel summary"
        f"\n  Total rows:                  {len(panel):,}"
        f"\n  -- Tier-1 (ESTBAN, local):   {len(tier1):,}  rows  "
         f"({tier1['mca_code'].nunique()} MCAs)"
        f"\n  -- Tier-2 (IF Data, national): {len(tier2):,}  rows"
        f"\n  Conglomerates:               {panel['CodConglomeradoPrudencial'].nunique()}"
        f"\n  Years:                       {panel['year'].min()} - {panel['year'].max()}"
        f"\n  Columns:                     {len(panel.columns)}"
        f"\n  pop_total non-null:          {panel['pop_total'].notna().sum():,}"
        f"\n  pix_users_pf_per1000 non-null: {panel['pix_users_pf_per1000'].notna().sum():,}"
        f"\n  Output:                      {OUTPUT_CSV}"
    )
    print("\nFirst 5 rows (key columns):")
    preview_cols = ["CodConglomeradoPrudencial", "mca_code", "year", "quarter",
                    "dep_a1", "dep_a2", "pop_total", "pix_users_pf_per1000"]
    preview_cols = [c for c in preview_cols if c in panel.columns]
    print(panel[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
