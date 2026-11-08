## panel_deposit_rates.py
# Author: Pedro Feijo de Moraes
#
# Pipeline step 16. Adds quarterly deposit rates and spreads by deposit type to
# processed/deposits_panel.csv (rewritten in place) and writes the quarterly macro rates, in
# percent, to processed/PANEL_INTERMED/quarterly_macro_rates.csv.
#
# Inputs:
#   processed/deposits_panel.csv                    panel_deposits.py (step 14)
#   SGS_RAW/macro_series_full.csv                   SGS 11 Selic over, 432 Selic target, 4389 CDI,
#                                                   226 TR; staged out of band (utils/macro_rates.py)
#   SGS_RAW/savings_yields/                         SGS 195 and 25 savings yields
#                                                   (scrape_bcb_savings_yields.py)
#   IF-Data List files                              CNPJ -> prudential conglomerate
#   COSIF_PROCESSED/custos_implicitos_*.csv         panel_cosif_extract.py (step 12)
#   COSIF_PROCESSED/cosif_cdb_rate_corrected.csv    panel_cosif_calibrate.py (step 13)
#   COSIF_PROCESSED/ip_rates_quarterly.csv          panel_ip_rates.py (step 15)
#   config/type5_balance_rates.csv                  posted type-5 balance rules (utils/type5_rates.py)
#
# Quarterly rate by deposit type (fractions), assembled in append_rates_to_panel:
#   1 demand     0
#   2 savings    savings_rate_qoq
#   3 interbank  cdi_qoq
#   4 time/CDB   corrected per-bank CDB rate where panel_cosif_calibrate attempted one, else the
#                COSIF residual type-4 rate, else the blended COSIF implicit rate; a rate above
#                RATE_A4_MAX_MULT x the highest Selic of t-7..t is dropped (rate_a4_guarded); a
#                quarter without a measured rate takes the bank's carried rate (median rate/CDI of
#                its measured quarters within +-4, x CDI; rate_a4_carried) or stays missing; an
#                uncorrected rate of exactly 0, or equal to the Selic, on a positive type-4 stock
#                is a fill, not a measurement, and goes to the carry too, as does a benchmarked
#                rate outside 0.5-1.5 x CDI (calibrate's rate_flag 4) and an imputed rate more
#                than 4 pp a year above the CDI or below 0.5 x CDI (calibrate's rate_flag 5, or
#                the uncorrected rate here on a positive type-4 stock);
#                rate_a4_fallback gives the reason, rate_a4_vintage_src the savings-cost source,
#                rate_a4_src where the rate comes from (statement, benchmarked, imputed, leaf,
#                carried)
#   5 prepaid    the issuer's posted rule (config/type5_balance_rates.csv, a percent of the CDI,
#                compounded over the days it covers) where one covers the quarter, flagged in
#                rate_a5_posted; otherwise the conglomerate's payment-institution rate if it has
#                one, else the quarterly median of those rates, else cdi_qoq
# Spreads are Selic minus the rate, quarterly and annualised by compounding.
# PANEL_INTERMED/type5_rate_rules_applied.csv lists every conglomerate-quarter a posted rule changed,
# with the chain's rate and the posted one.
## ---------------------------------------------------------------------------


## 1) Load necessary packages and set folders up:

# Packages
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import numpy as np
import logging
import os
import glob
import time
import pyarrow as pa
import pyarrow.csv as pa_csv

# Folders
from utils import paths
from utils.macro_rates import load_macro_series, make_quarterly_macro, savings_cost_qoq, OLD_RULE_SHARE
from utils.conglomerates import deposit_panel_codes
from utils import type5_rates
from utils import firm_remaps

# IF-Data List files: CNPJ -> Conglomerate mapping (monthly snapshots)
IF_DATA_LIST_DIR = str(paths.IF_DATA_LIST)

# Egan replication I/O paths
PROCESSED_PATH = str(paths.PROCESSED)
OUTPUT_PATH    = os.path.join(PROCESSED_PATH, "PANEL_INTERMED")
DEPOSITS_CSV   = os.path.join(PROCESSED_PATH, "deposits_panel.csv")

# COSIF processed output (from panel_cosif_extract.py)
COSIF_PROCESSED_PATH = os.path.join(PROCESSED_PATH, "COSIF_PROCESSED")

# rate_a4 (type-4 / CDB) plausibility band, as a multiple of the highest quarterly Selic of the last
# eight quarters.  A CDB pays ~90-115% of CDI; past a few multiples of Selic the implicit
# expense/stock ratio is broken, not a deposit rate.  See the rate_a4 assembly.
RATE_A4_MAX_MULT = float(os.environ.get("RATE_A4_MAX_MULT", 3.0))
# The rate/CDI range a measured quarter must lie in to lend its ratio to a neighbour's carry (I12,
# Pedro 2026-11-04): a CDB pays about 0.9-1.15 x CDI, and a neighbour far outside that is an
# imputation artefact that the carry would spread.
CARRY_DONOR_RANGE = (0.5, 1.5)
# The band an imputed rate must lie in (I17; Pedro, D1 2026-11-05): at most IMPUTED_GAP_PP points a
# year above the CDI, compounded annual, and at least IMPUTED_FLOOR x CDI.  panel_cosif_calibrate
# applies the same band to its strip x k rate (rate_flag 5); here it applies to the uncorrected
# residual or blended rate on a positive type-4 stock (code 9).  Keep the two in step.
IMPUTED_GAP_PP = 4.0
IMPUTED_FLOOR = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

## 2) User-defined functions:

# 2.1) Data loading functions:

def load_conglomerate_mapping(list_dir: str = IF_DATA_LIST_DIR) -> pd.DataFrame:
    """
    Load all IF-Data List snapshots (monthly) to build a time-varying
    CNPJ -> CodConglomeradoPrudencial mapping.

    Each row in a List file represents an institution.  We keep ALL
    active individual institutions (Td == 'I', Situacao == 'A'):
      - For those WITH a prudential conglomerate code, the mapping is
        CodInst -> CodConglomeradoPrudencial.
      - For standalone institutions (no conglomerate code -- mostly
        cooperativas), we assign their own CodInst as their
        CodConglomeradoPrudencial so they are not dropped from merges.

    Returns
    -------
    df_map : DataFrame with columns:
        CNPJ, AnoMes_list, CodConglomeradoPrudencial, CnpjInstituicaoLider
    """
    pattern = os.path.join(list_dir, "IF_DATA_List_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No IF-Data List files found in {list_dir}"
        )

    logging.info(
        f"Loading conglomerate mapping from {len(files)} List files ..."
    )
    frames = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath, dtype=str, encoding="utf-8")
            df.columns = [c.strip() for c in df.columns]
            # Keep ALL active individual institutions (including standalones)
            mask = (df["Td"] == "I") & (df["Situacao"] == "A")
            # Keep Sr (segment) and SegmentoTb (to identify Payment
            # Institutions) during mapping load so we don't need to
            # re-read List files later for make_segment_panel.
            keep_cols = [
                "CodInst", "Data",
                "CodConglomeradoPrudencial", "CnpjInstituicaoLider",
            ]
            if "Sr" in df.columns:
                keep_cols.append("Sr")
            if "SegmentoTb" in df.columns:
                keep_cols.append("SegmentoTb")
            sub = df.loc[mask, keep_cols].copy()
            # Standalones: assign own CodInst as conglomerate code
            standalone_mask = (
                sub["CodConglomeradoPrudencial"].isna()
                | (sub["CodConglomeradoPrudencial"] == "null")
            )
            sub.loc[standalone_mask, "CodConglomeradoPrudencial"] = (
                sub.loc[standalone_mask, "CodInst"]
            )
            sub.rename(columns={"CodInst": "CNPJ", "Data": "AnoMes_list"}, inplace=True)
            frames.append(sub)
        except Exception as e:
            logging.warning(f"  Error loading {fpath}: {e}")

    if not frames:
        raise ValueError("No valid conglomerate mapping loaded")

    df_map = pd.concat(frames, ignore_index=True)

    # The List file 'Data' is YYYYMM (monthly) but we want to align to
    # the nearest quarter-end for merging.  We keep the raw monthly value
    # so that we can do a time-aware merge (as-of join).
    df_map["AnoMes_list"] = pd.to_numeric(
        df_map["AnoMes_list"], errors="coerce"
    ).astype("Int64")

    # Pad CNPJ to 8 digits (COSIF uses 8-digit roots)
    df_map["CNPJ"] = df_map["CNPJ"].str.strip().str.zfill(8)

    # Deduplicate: if a CNPJ appears multiple times in the same month
    # (shouldn't happen, but be safe), keep first
    df_map = df_map.drop_duplicates(subset=["CNPJ", "AnoMes_list"])

    # -- Identify Payment Institutions (IPs) --
    # IPs can ONLY offer prepaid payment accounts (Lei 12.865/2013),
    # so their entire COSIF "Despesas de Captação" = prepaid cost.
    if "SegmentoTb" in df_map.columns:
        df_map["is_ip"] = (
            df_map["SegmentoTb"]
            .str.contains("Instituição de Pagamento", case=False, na=False)
            .astype(int)
        )
        # Latest IP flag per CNPJ
        _latest_ip = (
            df_map.sort_values("AnoMes_list")
            .drop_duplicates(subset=["CNPJ"], keep="last")
            [["CNPJ", "is_ip"]]
        )
        n_ips = _latest_ip["is_ip"].sum()
        logging.info(
            f"Payment Institutions identified: {n_ips} IPs out of "
            f"{len(_latest_ip)} unique CNPJs"
        )
    else:
        df_map["is_ip"] = 0
        logging.warning(
            "SegmentoTb column not found -- cannot identify "
            "Payment Institutions"
        )

    # Key each CNPJ by the code deposits_panel.csv files its deposits under
    # (utils.conglomerates.deposit_panel_codes): an ESTBAN bank's static code in every quarter,
    # otherwise the reported code, canonicalised. Any other key leaves quarters of a firm's
    # deposits without their COSIF rate.
    df_map["CodConglomeradoPrudencial"] = deposit_panel_codes(df_map["CNPJ"], df_map["CodConglomeradoPrudencial"])

    # Log coverage breakdown
    is_ccode = df_map["CodConglomeradoPrudencial"].str.startswith("C", na=False)
    n_congl = df_map.loc[is_ccode, "CodConglomeradoPrudencial"].nunique()
    n_standalone = df_map.loc[~is_ccode, "CNPJ"].nunique()
    logging.info(
        f"Conglomerate mapping loaded: "
        f"{len(df_map):,} CNPJ-month obs | "
        f"{df_map['CNPJ'].nunique()} unique CNPJs | "
        f"{n_congl} conglomerates + {n_standalone} standalones"
    )
    return df_map

def load_cosif_processed(cosif_dir: str = COSIF_PROCESSED_PATH) -> pd.DataFrame:
    """
    Load all COSIF processed files (output of panel_cosif_extract.py).

    Each file is a CNPJ x DATA_BASE panel for one taxonomy type, with
    deposit stocks, disaccumulated funding expense, and a blended
    implicit funding rate.

    Returns
    -------
    df_cosif : DataFrame with CNPJ, DATA_BASE, deposit stocks per type, funding expense, and blended implicit rate.
    """
    pattern = os.path.join(cosif_dir, "custos_implicitos_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        logging.warning(
            f"No COSIF processed files found in {cosif_dir}. "
            "COSIF implicit rates will not be available."
        )
        return pd.DataFrame()

    logging.info(f"Loading {len(files)} COSIF processed files ...")
    frames = []
    for fpath in files:
        try:
            # CNPJ must stay string; all other columns are numeric or dates.
            # CSVs are written by pandas to_csv() so decimals use '.'.
            df = pd.read_csv(fpath, dtype={"CNPJ": str})
            # Tag with taxonomy type from filename (used to separate
            # pre-aggregated conglomerate data from institution-level)
            taxonomy = (
                os.path.basename(fpath)
                .replace("custos_implicitos_", "")
                .replace(".csv", "")
            )
            df["cosif_taxonomy"] = taxonomy
            frames.append(df)
        except Exception as e:
            logging.warning(f"  Error loading {fpath}: {e}")

    if not frames:
        return pd.DataFrame()

    df_cosif = pd.concat(frames, ignore_index=True)
    df_cosif["DATA_BASE"] = pd.to_datetime(
        df_cosif["DATA_BASE"], errors="coerce"
    )
    # Pad CNPJ to 8 digits
    df_cosif["CNPJ"] = df_cosif["CNPJ"].astype(str).str.strip().str.zfill(8)

    logging.info(
        f"COSIF loaded: {len(df_cosif):,} rows | "
        f"{df_cosif['CNPJ'].nunique()} institutions"
    )
    return df_cosif

def load_ip_rates_processed(cosif_dir: str = COSIF_PROCESSED_PATH) -> pd.DataFrame:
    """Load the payment-institution rates written by panel_ip_rates.py (step 15)."""
    fpath = os.path.join(cosif_dir, "ip_rates_quarterly.csv")
    if not os.path.exists(fpath):
        logging.warning("ip_rates_quarterly.csv not found. Re-run panel_ip_rates.py.")
        return pd.DataFrame()
    df = pd.read_csv(fpath, dtype={"CNPJ": str})
    df["CNPJ"] = df["CNPJ"].str.strip().str.zfill(8)
    return df

def _collapse_monthly_to_quarterly(
    df_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse conglomerate x month COSIF data to conglomerate x quarter.

    - Estoque_Total (stock): quarter-end value (month in {3, 6, 9, 12})
    - Despesa_Captacao_Marginal (flow): sum of all 3 months in the quarter
    - cosif_implicit_rate: |quarterly expense| / lagged total deposits
    """
    df = df_monthly.copy()

    # Assign fiscal quarter
    df["q_month"] = ((df["month"] - 1) // 3 + 1) * 3
    df["AnoMes"] = (df["year"] * 100 + df["q_month"]).astype("Int64")

    # ---- Quarter-end STOCK: keep only months 3, 6, 9, 12 ----
    df_stocks_q = (
        df.loc[
            df["month"].isin([3, 6, 9, 12]),
            ["CodConglomeradoPrudencial", "AnoMes", "Estoque_Total"],
        ].copy()
    )

    # ---- Quarterly FLOW: sum all 3 months in the quarter ----
    df_flows_q = (
        df
        .groupby(["CodConglomeradoPrudencial", "AnoMes"])
        ["Despesa_Captacao_Marginal"]
        .sum()
        .reset_index()
    )

    # Merge quarter-end stock with summed quarterly flow
    df_agg = df_stocks_q.merge(
        df_flows_q,
        on=["CodConglomeradoPrudencial", "AnoMes"],
        how="left",
    )

    # Lagged total deposits (one quarter earlier)
    df_agg = df_agg.sort_values(["CodConglomeradoPrudencial", "AnoMes"])
    df_agg["Estoque_Total_Lag_Congl"] = (
        df_agg
        .groupby("CodConglomeradoPrudencial")["Estoque_Total"]
        .shift(1)
    )

    # Blended implicit rate = |quarterly expense| / lagged total deposits
    df_agg["cosif_implicit_rate"] = (
        df_agg["Despesa_Captacao_Marginal"].astype(float).abs()
        / df_agg["Estoque_Total_Lag_Congl"].astype(float).replace(0, np.nan)
    ).astype(float)

    # Clean non-finite and cap extreme outliers
    df_agg.loc[
        ~np.isfinite(df_agg["cosif_implicit_rate"]),
        "cosif_implicit_rate",
    ] = np.nan
    # A quarterly deposit rate above 50% is economically implausible
    df_agg.loc[
        df_agg["cosif_implicit_rate"] > 0.5,
        "cosif_implicit_rate",
    ] = np.nan

    return df_agg


def _map_cosif_to_congl(df_indiv: pd.DataFrame, df_map: pd.DataFrame) -> pd.DataFrame:
    map_cols = ["CNPJ", "AnoMes_list", "CodConglomeradoPrudencial"]
    if "is_ip" in df_map.columns:
        map_cols.append("is_ip")

    merged = df_indiv.merge(
        df_map[map_cols],
        left_on=["CNPJ", "AnoMes_cosif"],
        right_on=["CNPJ", "AnoMes_list"],
        how="inner",
    )

    if unmerged_cnpjs := set(df_indiv["CNPJ"].unique()) - set(merged["CNPJ"].unique()):
        latest_map = (
            df_map
            .sort_values("AnoMes_list")
            .drop_duplicates(subset=["CNPJ"], keep="last")
            [["CNPJ", "CodConglomeradoPrudencial"]
             + (["is_ip"] if "is_ip" in df_map.columns else [])]
        )
        df_unmerged = df_indiv[df_indiv["CNPJ"].isin(unmerged_cnpjs)]
        fallback = df_unmerged.merge(latest_map, on="CNPJ", how="inner")
        if not fallback.empty:
            merged = pd.concat([merged, fallback], ignore_index=True)
            logging.info(
                f"Fallback mapping added {len(fallback):,} rows for "
                f"{fallback['CNPJ'].nunique()} CNPJs outside List date range"
            )

    is_ccode = merged["CodConglomeradoPrudencial"].str.startswith("C", na=False)
    n_congl = merged.loc[is_ccode, "CodConglomeradoPrudencial"].nunique()
    n_standalone = merged.loc[~is_ccode, "CodConglomeradoPrudencial"].nunique()
    logging.info(
        f"COSIF-conglomerate merge: {len(merged):,} rows | "
        f"{n_congl} conglomerates + {n_standalone} standalones"
    )

    return merged

def _aggregate_ip_prepaid_data(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    has_ip_flag = "is_ip" in merged.columns and (merged["is_ip"] == 1).any()
    df_ip_monthly = pd.DataFrame()
    congl_has_ip = pd.DataFrame()
    
    if has_ip_flag:
        ip_rows = merged[merged["is_ip"] == 1]
        if not ip_rows.empty:
            if prepaid_agg_cols := [
                c for c in ["Estoque_Prepago", "Desp_Prepago_Marginal"]
                if c in ip_rows.columns
            ]:
                df_ip_monthly = (
                    ip_rows
                    .groupby(["CodConglomeradoPrudencial", "year", "month"])
                    [prepaid_agg_cols]
                    .sum(min_count=1)
                    .reset_index()
                )
                df_ip_monthly.rename(columns={
                    "Estoque_Prepago":      "Estoque_Prepago_IP",
                    "Desp_Prepago_Marginal": "Desp_Prepago_IP",
                }, inplace=True)
                logging.info(
                    f"IP prepaid aggregation: {len(df_ip_monthly):,} congl×month obs "
                    f"from {ip_rows['CNPJ'].nunique()} IP CNPJs in "
                    f"{df_ip_monthly['CodConglomeradoPrudencial'].nunique()} conglomerates"
                )

        congl_has_ip = (
            merged[merged["is_ip"] == 1]
            .drop_duplicates(subset=["CodConglomeradoPrudencial"])
            [["CodConglomeradoPrudencial"]]
            .assign(has_ip=1)
        )
        
    return df_ip_monthly, congl_has_ip, has_ip_flag

def _correct_cosif_time_aggregation(df_indiv: pd.DataFrame) -> pd.DataFrame:
    """
    Repair two COSIF time-aggregation artifacts at the institution (CNPJ) x
    month level, BEFORE any aggregation across institutions or quarters.
    Both inject a quarterly see-saw into the implicit funding rate (and thus
    into inferred spreads and the sleepiness estimation) if left uncorrected.

    (1) DISACCUMULATE THE FUNDING EXPENSE.
        COSIF group-8 result accounts accumulate within the SEMESTER and reset
        in January and July (confirmed semiannual for 40/40 sampled banks).
        Despite its name, the pipeline column `Despesa_Captacao_Marginal`
        (and `Desp_Prepago_Marginal`) holds this semester-cumulative balance,
        not a monthly flow -- so summing three of them within a quarter
        (see `_collapse_monthly_to_quarterly`) produces a ramp-and-crash
        sawtooth.  We difference within each (CNPJ, year, semester) run to
        recover the true monthly flow; the first month of a run keeps its
        level (the previous semester already reset to zero).  Differencing
        across an internal gap still yields the correct cumulative increment,
        so quarterly sums stay right even with missing months.

    (2) UN-DOUBLE THE SEMESTER-CLOSE STOCK.
        At June and December BCB ships a second document (4016 balanco
        patrimonial) that duplicates every group-4 balance.  The (missing)
        producer summed 4010+4016, doubling `Estoque_Total` / `Estoque_Prepago`
        at those two months (factor exactly 2.00; 4016 carries no group-8 rows,
        so the expense is unaffected).  We halve those two months.  4010's CNPJ
        set is a subset of 4016's at semester close, so every institution is
        doubled and the blanket halving is safe.
    """
    if df_indiv.empty:
        return df_indiv

    df = df_indiv.sort_values(["CNPJ", "year", "month"]).copy()
    df["_sem"] = np.where(df["month"] <= 6, 1, 2)

    # (1) Disaccumulate expense flows within (CNPJ, year, semester)
    for col in ["Despesa_Captacao_Marginal", "Desp_Prepago_Marginal"]:
        if col in df.columns:
            df[col] = (
                df.groupby(["CNPJ", "year", "_sem"])[col]
                .transform(lambda s: s.diff().where(s.shift(1).notna(), s))
            )

    # (2) Halve the doubled June/December stocks
    close_mask = df["month"].isin([6, 12])
    for col in ["Estoque_Total", "Estoque_Prepago"]:
        if col in df.columns:
            df.loc[close_mask, col] = df.loc[close_mask, col] / 2.0

    df.drop(columns=["_sem"], inplace=True)
    logging.info(
        "COSIF time-agg correction: expense disaccumulated within "
        "(CNPJ, semester); Jun/Dec stocks halved (4016 double-count)."
    )
    return df


def aggregate_cosif_to_conglomerates(
    df_cosif: pd.DataFrame,
    df_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate institution-level COSIF data up to prudential conglomerates.

    Strategy:
        1. Separate pre-aggregated COSIF data (CONGLOMERADO_PRUDENCIAL /
           CONGLOMERADO_FINANCEIRO) from institution-level data to avoid
           double-counting.
        2. Institution-level path: assign each CNPJ x month to its
           prudential conglomerate via the mapping (including standalones,
           where CNPJ = CodConglPrud). Sum stocks and expenses across
           member institutions.
        3. Collapse monthly -> quarterly and recompute blended rates.

    Returns a conglomerate x quarter panel with cosif_implicit_rate.
    """
    if df_cosif.empty or df_map.empty:
        logging.warning("Cannot aggregate COSIF -- missing data.")
        return pd.DataFrame()

    # ---- 0. Prep: add time keys ----
    df = df_cosif.copy()
    df["year"]  = df["DATA_BASE"].dt.year
    df["month"] = df["DATA_BASE"].dt.month
    df["AnoMes_cosif"] = df["year"] * 100 + df["month"]

    # ---- 1. Separate pre-aggregated from institution-level ----
    AGGREGATE_TYPES = {"CONGLOMERADO_PRUDENCIAL", "CONGLOMERADO_FINANCEIRO"}
    if "cosif_taxonomy" in df.columns:
        is_agg = df["cosif_taxonomy"].isin(AGGREGATE_TYPES)
        df_indiv = df[~is_agg].copy()
        n_dropped = is_agg.sum()
        logging.info(
            f"COSIF separation: {len(df_indiv):,} institution-level rows kept, "
            f"{n_dropped:,} pre-aggregated rows excluded (avoid double-counting)"
        )
    else:
        df_indiv = df.copy()
        logging.info("No cosif_taxonomy column -- using all rows as institution-level")

    # ---- 1b. Repair COSIF time-aggregation artifacts (expense accumulation
    #          + Jun/Dec stock doubling) at the institution x month level ----
    df_indiv = _correct_cosif_time_aggregation(df_indiv)

    # ---- 2. Map institution-level COSIF CNPJs to conglomerates ----
    merged = _map_cosif_to_congl(df_indiv, df_map)

    if merged.empty:
        logging.warning("COSIF-conglomerate merge produced 0 rows.")
        return pd.DataFrame()

    # ---- 3. Aggregate to conglomerate x month ----
    #   Core value columns: Estoque_Total (total deposit stock) and
    #   Despesa_Captacao_Marginal (blended expense).  From the 2025 COSIF
    #   recode, Estoque_Prepago and Desp_Prepago_Marginal are also available
    #   for Payment Institutions.
    agg_cols = [
        "Estoque_Total", "Despesa_Captacao_Marginal",
        "Estoque_Prepago", "Desp_Prepago_Marginal",
    ]
    agg_cols = [c for c in agg_cols if c in merged.columns]
    df_monthly = (
        merged
        .groupby(["CodConglomeradoPrudencial", "year", "month"])
        [agg_cols]
        .sum(min_count=0)
        .reset_index()
    )

    df_ip_monthly, congl_has_ip, has_ip_flag = _aggregate_ip_prepaid_data(merged)

    # ---- 4. Collapse to quarterly and compute rates ----
    df_agg = _collapse_monthly_to_quarterly(df_monthly)

    # ---- 4b. Collapse IP prepaid data to quarterly and compute ip_prepaid_rate ----
    if not df_ip_monthly.empty:
        # Reuse _collapse_monthly_to_quarterly by temporarily renaming columns
        df_ip_for_collapse = df_ip_monthly.rename(columns={
            "Estoque_Prepago_IP": "Estoque_Total",
            "Desp_Prepago_IP":    "Despesa_Captacao_Marginal",
        })
        df_ip_agg = _collapse_monthly_to_quarterly(df_ip_for_collapse)
        df_ip_agg.rename(
            columns={"cosif_implicit_rate": "ip_prepaid_rate"},
            inplace=True,
        )
        df_agg = df_agg.merge(
            df_ip_agg[["CodConglomeradoPrudencial", "AnoMes", "ip_prepaid_rate"]],
            on=["CodConglomeradoPrudencial", "AnoMes"],
            how="left",
        )
        logging.info(
            f"IP prepaid rate coverage: "
            f"{df_agg['ip_prepaid_rate'].notna().sum():,} / "
            f"{len(df_agg):,} congl×quarter obs "
            f"(zero for pre-2025 months, ~CDI for 2025+ reporters)"
        )
    else:
        df_agg["ip_prepaid_rate"] = np.nan

    # ---- 4c. Merge conglomerate has_ip flag ----
    if not congl_has_ip.empty:
        df_agg = df_agg.merge(congl_has_ip, on="CodConglomeradoPrudencial", how="left")
        df_agg["has_ip"] = df_agg["has_ip"].fillna(0).astype(int)
    else:
        df_agg["has_ip"] = 0

    # Expose raw expense so append_rates_to_panel can apply the residual method for Type 4
    df_agg["cosif_desp_captacao"] = df_agg["Despesa_Captacao_Marginal"].abs()

    keep = [
        "CodConglomeradoPrudencial", "AnoMes", "cosif_implicit_rate",
        "cosif_desp_captacao",   # raw absolute quarterly funding expense (R$)
        "ip_prepaid_rate", "has_ip",
    ]
    result = df_agg[keep].copy()
    logging.info(
        f"COSIF conglomerate panel: {len(result):,} obs | "
        f"{result['CodConglomeradoPrudencial'].nunique()} conglomerates | "
        f"{result['AnoMes'].nunique()} quarters"
    )
    return result

## 3) Append rates and spreads to the deposit panel:

def append_rates_to_panel():
    logging.info(f"Adding rates and spreads to {DEPOSITS_CSV}...")

    df = pd.read_csv(DEPOSITS_CSV, low_memory=False)
    
    # 1. Add AnoMes
    df["AnoMes"] = df["Year"].astype(int) * 100 + df["Quarter"].astype(int) * 3
    
    # 2. National lags
    groupby_cols = ["CodConglomeradoPrudencial", "AnoMes"]
    nat_stocks = df.groupby(groupby_cols)[["dep_a1", "dep_a2", "dep_a3", "dep_a4", "dep_a5"]].sum().reset_index()
    nat_stocks = nat_stocks.sort_values(groupby_cols)
    nat_stocks["total_deposits"] = nat_stocks[["dep_a1", "dep_a2", "dep_a4", "dep_a5"]].sum(axis=1, min_count=1)

    for col in ["dep_a2", "dep_a3", "dep_a4", "dep_a5"]:
        nat_stocks[f"lag_{col}"] = nat_stocks.groupby("CodConglomeradoPrudencial")[col].shift(1).fillna(0.0)
    
    # 3. Macro 
    df_macro = load_macro_series()
    df_macro_q = make_quarterly_macro(df_macro)

    # PANEL_INTERMED/quarterly_macro_rates.csv: the same quarterly rates in PERCENT, read by
    # make_desc_compressed_tables.py, make_slide_figures.py and the diag_cosif_k4_validate*
    # diagnostics (panel_cosif_calibrate.py computes its own copy with make_quarterly_macro).
    _macro_pct = df_macro_q.copy()
    for col in ["selic_qoq", "cdi_qoq", "savings_rate_qoq", "savings_old_qoq"]:
        if col in _macro_pct.columns:
            _macro_pct[col] = _macro_pct[col] * 100
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    _macro_pct.to_csv(os.path.join(OUTPUT_PATH, "quarterly_macro_rates.csv"), index=False)
    
    # 4. COSIF
    df_congl_map = load_conglomerate_mapping()
    df_cosif = load_cosif_processed()
    df_cosif_q = aggregate_cosif_to_conglomerates(df_cosif, df_congl_map)

    # df_cosif_q carries AnoMes, cosif_implicit_rate, cosif_desp_captacao, ip_prepaid_rate and
    # has_ip (the keep list of aggregate_cosif_to_conglomerates).
    aux =nat_stocks.merge(df_macro_q[["AnoMes", "selic_qoq", "cdi_qoq", "savings_rate_qoq", "savings_old_qoq", "meta_selic"]], on="AnoMes", how="left")
    aux = aux.merge(df_cosif_q, on=["CodConglomeradoPrudencial", "AnoMes"], how="left")
    
    # 4b. Inject NEW IP rates
    df_ip_rates = load_ip_rates_processed()
    if not df_ip_rates.empty:
        # Map CNPJ -> CodConglomeradoPrudencial
        ip_map = df_congl_map[['CNPJ', 'CodConglomeradoPrudencial']].drop_duplicates()
        df_ip_rates = df_ip_rates.merge(ip_map, on='CNPJ', how='left')
        
        # Aggregate to conglomerate (median rate if multiple IPs per congl)
        congl_ip = df_ip_rates.groupby(['CodConglomeradoPrudencial', 'AnoMes'])['ip_implicit_rate'].median().reset_index()
        congl_ip.rename(columns={'ip_implicit_rate': 'new_ip_rate'}, inplace=True)
        
        aux = aux.merge(congl_ip, on=['CodConglomeradoPrudencial', 'AnoMes'], how='left')
        # The panel_ip_rates rate takes precedence over the COSIF prepaid rate
        aux['ip_prepaid_rate'] = aux['new_ip_rate'].fillna(aux['ip_prepaid_rate'])
        aux.drop(columns=['new_ip_rate'], inplace=True)

    # 4c. PicPay is one firm, C0088022 (utils/firm_remaps.py): the prepaid rate of Banco Original's
    # conglomerate in 2019Q4-2024Q3 is PicPay's wallet rate, and the wallet balance now sits on C0088022.
    if 'ip_prepaid_rate' in aux.columns:
        _src = aux[(aux['CodConglomeradoPrudencial'] == firm_remaps.PICPAY_FROM)
                   & firm_remaps.picpay_window_anomes(aux['AnoMes'])].set_index('AnoMes')['ip_prepaid_rate'].dropna()
        _dst = (aux['CodConglomeradoPrudencial'] == firm_remaps.PICPAY_TO) & aux['AnoMes'].isin(_src.index)
        aux.loc[_dst, 'ip_prepaid_rate'] = aux.loc[_dst, 'AnoMes'].map(_src)
        logging.info(f"PicPay re-map: prepaid rate of {firm_remaps.PICPAY_FROM} -> {firm_remaps.PICPAY_TO} "
                     f"on {int(_dst.sum())} quarters ({len(_src)} available)")

    # T2 & T3 imputed expenses
    _imputed_t2 = savings_cost_qoq(aux, OLD_RULE_SHARE).fillna(0.0) * aux["lag_dep_a2"]
    _imputed_t3 = aux["cdi_qoq"].fillna(0.0)          * aux["lag_dep_a3"]
    _residual_raw = aux["cosif_desp_captacao"] - _imputed_t2 - _imputed_t3
    _residual = _residual_raw.where(_residual_raw > 0)      # a residual <= 0 measures nothing

    _denom_t45 = aux["lag_dep_a4"] + aux["lag_dep_a5"]
    _denom_t45 = _denom_t45.replace(0.0, np.nan)

    _type4_raw = _residual.astype(float) / _denom_t45.astype(float)
    _type4_raw.loc[~np.isfinite(_type4_raw)] = np.nan
    _type4_raw.loc[_type4_raw > 0.5] = np.nan
    aux["cosif_type4_rate"] = _type4_raw

    # Median IP rate per quarter, over quarters with at least 2 positive IP rates
    _ip_mask = aux["ip_prepaid_rate"].notna() & (aux["ip_prepaid_rate"] > 0)
    valid_ip = aux[_ip_mask].copy()
    sizes = valid_ip.groupby("AnoMes").size()
    valid_anomes = sizes[sizes >= 2].index
    medians = valid_ip[valid_ip["AnoMes"].isin(valid_anomes)].groupby("AnoMes")["ip_prepaid_rate"].median()
    aux = aux.merge(medians.rename("median_ip_rate"), on="AnoMes", how="left")

    # The CORRECTED per-bank CDB rate from panel_cosif_calibrate.py (cosif_cdb_rate_corrected.csv):
    # one row per conglomerate-quarter calibrate attempted, with the rate (missing when the attempt
    # failed) and its rate_flag.
    _corr_path = os.path.join(COSIF_PROCESSED_PATH, "cosif_cdb_rate_corrected.csv")
    if os.path.exists(_corr_path):
        _corr = pd.read_csv(_corr_path, dtype={"CodConglomeradoPrudencial": str})
        _corr = _corr.rename(columns={"rate_a4_corrected_qoq": "cosif_cdb_rate_corrected",
                                      "rate_flag": "cosif_cdb_flag"})[
            ["CodConglomeradoPrudencial", "AnoMes", "cosif_cdb_rate_corrected", "cosif_cdb_flag",
             "rate_a4_vintage_src", "rate_a4_src"]
        ]
        aux["CodConglomeradoPrudencial"] = aux["CodConglomeradoPrudencial"].astype(str)
        aux = aux.merge(_corr, on=["CodConglomeradoPrudencial", "AnoMes"], how="left")
        logging.info(
            f"Corrected CDB rate merged: {aux['cosif_cdb_flag'].notna().sum():,} attempted cells, "
            f"{aux['cosif_cdb_rate_corrected'].notna().sum():,} measured"
        )
    else:
        aux["cosif_cdb_rate_corrected"] = np.nan
        aux["cosif_cdb_flag"] = np.nan
        aux["rate_a4_vintage_src"] = np.nan
        aux["rate_a4_src"] = np.nan
        logging.info(f"Corrected CDB rate not found ({_corr_path}); using residual.")
    aux["rate_a4_vintage_src"] = aux["rate_a4_vintage_src"].fillna("none")

    aux["rate_a1"] = 0.0
    aux["rate_a2"] = aux["savings_rate_qoq"]
    aux["rate_a3"] = aux["cdi_qoq"]

    # ---- rate_a4 --------------------------------------------------------------------------------
    # (1) The corrected rate where calibrate attempted one; a failed attempt stays missing (strip or
    #     leaf <= 0, or above its cap).  Where calibrate has no row, the residual type-4 rate, then
    #     the blended COSIF implicit rate.
    # (2) The plausibility band: a CDB pays ~90-115% of CDI, so a rate above RATE_A4_MAX_MULT x the
    #     highest quarterly Selic of the last eight quarters (t-7..t) is a broken expense/stock ratio,
    #     not a deposit rate.  The band looks back because a quarter's funding expense prices a stock
    #     raised over the previous two years, at the rates of the day: against the current Selic alone
    #     the band binds on legacy funding whenever the Selic falls (2019-2021).
    # (3) A quarter left without a measured rate takes the bank's carried rate: its median rate/CDI
    #     over its measured quarters within four quarters either side, times this quarter's CDI
    #     (rate_a4_carried = 1).  A neighbour counts only if its rate/CDI lies in CARRY_DONOR_RANGE,
    #     which also keeps out an uncorrected rate of exactly 0 where the quarter holds no type-4
    #     deposits.  With no such neighbour it stays missing, and the row leaves every sample that
    #     needs the type-4 spread.
    # rate_a4_fallback records why a quarter has no measured rate: 0 measured, 1 above the band,
    # 2 strip or leaf <= 0, 3 above calibrate's cap or in its RATE_FAIL, 4 residual <= 0 or an
    # uncorrected rate of exactly 0, 5 nothing to measure (no lagged stock or no expense), 6 an
    # uncorrected rate equal to the Selic to 1e-6 (4 and 6 on a positive type-4 stock, where the
    # blended or residual rate returns a fill, not a price), 7 a benchmarked rate outside
    # calibrate's BENCH_RANGE x CDI (its rate_flag 4), 8 an imputed strip x k rate outside
    # calibrate's IMPUTED_* band (its rate_flag 5), 9 an uncorrected rate outside the same band on a
    # positive type-4 stock.  rate_a4_src records where a rate
    # comes from: calibrate's source (statement, benchmarked, imputed, leaf) for a corrected rate,
    # imputed for the residual or blended rate, carried for a carried one; missing where the
    # quarter has no rate.
    _has_corr = aux["cosif_cdb_flag"].notna()
    _fallback = aux["cosif_type4_rate"].fillna(aux["cosif_implicit_rate"])
    aux["rate_a4"] = aux["cosif_cdb_rate_corrected"].where(_has_corr, _fallback)
    _code = pd.Series(0, index=aux.index, dtype="int8")
    _code[_has_corr & (aux["cosif_cdb_flag"] == 2)] = 2
    _code[_has_corr & (aux["cosif_cdb_flag"] == 3)] = 3
    _code[_has_corr & (aux["cosif_cdb_flag"] == 4)] = 7
    _code[_has_corr & (aux["cosif_cdb_flag"] == 5)] = 8
    _code[~_has_corr & aux["rate_a4"].isna() & (_residual_raw <= 0)] = 4
    _code[aux["rate_a4"].isna() & (_code == 0)] = 5
    _pos4 = aux["dep_a4"].fillna(0) > 0
    _zero = ~_has_corr & _pos4 & (_code == 0) & (aux["rate_a4"] == 0)
    _at_selic = ~_has_corr & _pos4 & (_code == 0) & ((aux["rate_a4"] - aux["selic_qoq"]).abs() < 1e-6)
    _code[_zero] = 4
    _code[_at_selic] = 6
    aux.loc[_zero | _at_selic, "rate_a4"] = np.nan
    _gap_pp = 100 * ((1 + aux["rate_a4"]) ** 4 - (1 + aux["cdi_qoq"]) ** 4)
    _unc_out = (~_has_corr & _pos4 & (_code == 0) & aux["rate_a4"].notna()
                & ((_gap_pp > IMPUTED_GAP_PP) | (aux["rate_a4"] < IMPUTED_FLOOR * aux["cdi_qoq"])))
    _code[_unc_out] = 9
    aux.loc[_unc_out, "rate_a4"] = np.nan

    _selic_max = (df_macro_q.sort_values("AnoMes").set_index("AnoMes")["selic_qoq"]
                  .rolling(8, min_periods=1).max())
    _band = RATE_A4_MAX_MULT * aux["AnoMes"].map(_selic_max)
    _bad = aux["rate_a4"].notna() & _band.notna() & (aux["rate_a4"] > _band)
    aux["rate_a4_guarded"] = _bad.astype(int)
    aux.loc[_bad, "rate_a4"] = np.nan
    _code[_bad] = 1

    _qi = (aux["AnoMes"] // 100) * 4 + (aux["AnoMes"] % 100) // 3
    _r_cdi = aux["rate_a4"] / aux["cdi_qoq"]
    _ratio = _r_cdi.where((_code == 0) & _r_cdi.between(*CARRY_DONOR_RANGE))
    _carried = pd.Series(np.nan, index=aux.index)
    for _, g in aux[["CodConglomeradoPrudencial"]].groupby(aux["CodConglomeradoPrudencial"]):
        need = g.index[_code[g.index] != 0]
        if not len(need):
            continue
        r, q = _ratio[g.index], _qi[g.index]
        for i in need:
            near = r[(q - q[i]).abs().between(1, 4) & r.notna()]
            if len(near):
                _carried[i] = near.median() * aux.at[i, "cdi_qoq"]
    _fill = (_code != 0) & _carried.notna()
    aux.loc[_fill, "rate_a4"] = _carried[_fill]
    aux["rate_a4_carried"] = _fill.astype("int8")
    aux["rate_a4_fallback"] = _code
    _src = aux["rate_a4_src"].where(_has_corr, "imputed").where(_code == 0)
    _src[_fill] = "carried"
    aux["rate_a4_src"] = _src
    _why = {int(k): int(v) for k, v in _code[_code != 0].value_counts().sort_index().items()}
    logging.info(
        f"rate_a4: {int((_code == 0).sum()):,} measured of {len(aux):,} bank-quarters; no measured rate "
        f"{_why} (1 band > {RATE_A4_MAX_MULT:g}x max Selic t-7..t, 2 strip/leaf <= 0, 3 cap, "
        f"4 residual <= 0 or exactly 0, 5 nothing to measure, 6 equal to the Selic, "
        f"7 benchmarked outside 0.5-1.5 x CDI, 8 imputed strip x k outside the band, "
        f"9 uncorrected outside the band); "
        f"carried {int(_fill.sum()):,}, "
        f"left missing {int(((_code != 0) & ~_fill).sum()):,}"
    )

    aux["rate_a5"] = np.where(
        aux["has_ip"] == 1,
        aux["ip_prepaid_rate"].fillna(aux["median_ip_rate"]).fillna(aux["cdi_qoq"]),
        aux["median_ip_rate"].fillna(aux["cdi_qoq"])
    )

    # Posted balance rules replace the chain above for the days an issuer's own documents cover.
    aux, _t5 = type5_rates.apply_rules(aux)
    _t5 = _t5.merge(aux[["CodConglomeradoPrudencial", "AnoMes", "selic_qoq"]].drop_duplicates(),
                    on=["CodConglomeradoPrudencial", "AnoMes"], how="left")
    _t5["spread_chain"] = _t5["selic_qoq"] - _t5["rate_chain"]
    _t5["spread_posted"] = _t5["selic_qoq"] - _t5["rate_posted"]
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    _t5.to_csv(os.path.join(OUTPUT_PATH, "type5_rate_rules_applied.csv"), index=False)
    logging.info(
        f"type-5 posted rules: {len(_t5):,} conglomerate-quarters changed across "
        f"{_t5['CodConglomeradoPrudencial'].nunique() if len(_t5) else 0} conglomerates "
        f"(PANEL_INTERMED/type5_rate_rules_applied.csv)"
    )

    aux["risk_free_qoq"] = aux["selic_qoq"]
    aux["risk_free_ann"] = (1 + aux["risk_free_qoq"]) ** 4 - 1

    for t in [1, 2, 3, 4, 5]:
        aux[f"spread_a{t}"]     = aux["risk_free_qoq"] - aux[f"rate_a{t}"]
        # Annualised: named spread_ann_a{t} so wide_to_long(stub='spread_ann_a') works
        aux[f"rate_a{t}_ann"]      = (1 + aux[f"rate_a{t}"]) ** 4 - 1
        aux[f"spread_ann_a{t}"]    = aux["risk_free_ann"] - aux[f"rate_a{t}_ann"]

    drop_cols = ["AnoMes", "risk_free_qoq", "risk_free_ann",
                 "rate_a1", "rate_a2", "rate_a3", "rate_a4", "rate_a5",
                 "rate_a1_ann", "rate_a2_ann", "rate_a3_ann", "rate_a4_ann", "rate_a5_ann",
                 "spread_a1", "spread_a2", "spread_a3", "spread_a4", "spread_a5",
                 "spread_ann_a1", "spread_ann_a2", "spread_ann_a3", "spread_ann_a4", "spread_ann_a5"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # The rate_a4 flags ride along so downstream code (and the paper) can see which bank-quarters
    # have no measured type-4 rate, why, and which carry an imputed one.
    cols_to_merge = ["CodConglomeradoPrudencial", "AnoMes", "risk_free_qoq", "risk_free_ann",
                     "total_deposits", "rate_a4_guarded", "rate_a4_fallback", "rate_a4_carried",
                     "rate_a4_vintage_src", "rate_a4_src", "rate_a5_posted"] + \
                    [f"rate_a{t}" for t in range(1, 6)] + \
                    [f"spread_a{t}" for t in range(1, 6)] + \
                    [f"spread_ann_a{t}" for t in range(1, 6)]
                    
    df["AnoMes"] = df["Year"].astype(int) * 100 + df["Quarter"].astype(int) * 3

    # Every column this step merges is dropped from df first, with any _x / _y copy of it, so the step
    # can run again on a deposits_panel.csv that already carries its output (a re-run of step 16
    # without step 14) and the merge still yields one clean column of each name.
    merged = [c for c in cols_to_merge if c not in ("CodConglomeradoPrudencial", "AnoMes")]
    stale = [c for c in df.columns if c in merged or (c[-2:] in ("_x", "_y") and c[:-2] in merged)]
    df = df.drop(columns=stale)
    df = df.merge(aux[cols_to_merge], on=["CodConglomeradoPrudencial", "AnoMes"], how="left")
    df = df.drop(columns=["AnoMes"])

    pa_csv.write_csv(pa.Table.from_pandas(df, preserve_index=False), DEPOSITS_CSV)
    logging.info(f"Updated {DEPOSITS_CSV} with rates and spreads.")

if __name__ == "__main__":
    t0 = time.perf_counter()
    append_rates_to_panel()
    print(f"Done in {time.perf_counter() - t0:.2f}s")

