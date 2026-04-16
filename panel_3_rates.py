## egan_panel_build.py
# Author: Pedro Feijo de Moraes
# Last edited: 2026-02-23
#
# Objective: Build an Egan et al. (2025)-style panel dataset for Brazilian
#            prudential conglomerates, using BCB IF-Data, COSIF, and SGS.
#
# Output panel structure: CodConglomeradoPrudencial x Quarter x Deposit_Type
#
#   Variables:
#     CodConglPrud        - Prudential conglomerate code (e.g. "C0080099")
#     quarter             - Period label (e.g. "2020Q1")
#     AnoMes              - IF-Data period code (e.g. 202003)
#     deposit_type        - 1 = demand (à vista), 2 = savings (poupança),
#                           3 = interbank, 4 = time (a prazo/CDB),
#                           5 = prepaid payment accounts
#     deposit_balance     - End-of-quarter stock (R$ thousands, IF-Data units)
#     lagged_deposits     - Previous quarter's deposit_balance
#     risk_free_qoq       - Selic overnight, compounded quarter-over-quarter (%)
#     deposit_rate_qoq    - Deposit rate for the type (% qoq)
#     spread_qoq          - risk_free_rate - deposit_rate (% qoq, typically >= 0, opportunity cost)
#     cosif_implicit_rate - COSIF-derived blended implicit funding rate (%)
#                           (institution-specific, aggregated to conglomerate)
#
#   Deposit-rate logic:
#     Type 1 (demand)     : 0  (checking accounts pay no interest in Brazil)
#     Type 2 (savings)    : TR + regulated component (old/new poupança rule)
#     Type 3 (interbank)  : CDI (100 % of CDI is market convention)
#     Type 4 (time / CDB) : COSIF implicit rate for the conglomerate when
#                           available; falls back to CDI proxy otherwise
#     Type 5 (prepaid)    : ENDOGENOUS -- conglomerate-specific:
#                           - Conglomerates with IP subsidiary: IP member's
#                             COSIF implicit rate (= prepaid cost, since IPs
#                             can only offer prepaid per Lei 12.865/2013)
#                           - Others: quarterly median IP rate (cross-market proxy)
#                           - Fallback: CDI
#
#   Risk-free rate: Selic overnight (SGS 11) compounded over business days.
#
# Data sources:
#   - IF-Data Prudential Conglomerates (BCB/IF Data/Prudential Conglomerates/)
#     - Passivo report  (report 3): deposit stocks by type
#   - IF-Data List (BCB/IF Data/List/): CNPJ -> conglomerate mapping
#   - COSIF processed (BCB/Egan_et_al_2025_Rep/processed/COSIF_PROCESSED/):
#     institution-level implicit funding costs (monthly, by CNPJ)
#   - SGS macro series (pre-downloaded by data_collection_1.py)
#
# Prerequisites:
#   pip install pandas numpy
#   Run data_collection_1.py first to download SGS macro series
#   Run cosif_process_1.py first to process COSIF data (COSIF data is downloaded manually)
## ---------------------------------------------------------------------------


## 1) Load necesary packages and set folders up:

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
from logging.handlers import RotatingFileHandler

# Folders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BCB_PATH   = os.path.join(PARENT_DIR, "BCB")

# Primary data source: complete IF-Data Prudential Conglomerate quarterly files
IF_DATA_DIR = os.path.join(BCB_PATH, "IF Data", "Prudential Conglomerates")

# IF-Data List files: CNPJ -> Conglomerate mapping (monthly snapshots)
IF_DATA_LIST_DIR = os.path.join(BCB_PATH, "IF Data", "List")

# Egan replication I/O paths
EGAN_PATH      = os.path.join(BCB_PATH, "Egan_et_al_2025_Rep")
RAW_PATH       = os.path.join(EGAN_PATH, "raw")
SGS_PATH       = os.path.join(RAW_PATH, "SGS_RAW")
PROCESSED_PATH = os.path.join(EGAN_PATH, "processed")
OUTPUT_PATH    = os.path.join(PROCESSED_PATH, "PANEL_INTERMED")

# COSIF processed output (from cosif_process_1.py)
COSIF_PROCESSED_PATH = os.path.join(PROCESSED_PATH, "COSIF_PROCESSED")

for _p in [RAW_PATH, SGS_PATH, PROCESSED_PATH, OUTPUT_PATH, COSIF_PROCESSED_PATH]:
    os.makedirs(_p, exist_ok=True)

# Logging
_log_file = os.path.join(SCRIPT_DIR, "egan_panel_build.log")
_handler  = RotatingFileHandler(
    _log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
logging.basicConfig(
    handlers=[_handler],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

## 2) Reference constants:

# -- IF-Data Passivo account codes (Conta) for each deposit type --
#    BCB renumbered all Passivo/Resumo accounts starting in 2025.
#    We keep the pre-2025 codes as canonical and normalise 2025+ codes
#    to them via CONTA_REMAP (see below).
DEPOSIT_ACCOUNTS = {
    78282:  {"type": 1, "name": "demand",    "label": "Depósitos à Vista (a1)"},
    78283:  {"type": 2, "name": "savings",   "label": "Depósitos de Poupança (a2)"},
    78284:  {"type": 3, "name": "interbank", "label": "Depósitos Interfinanceiros (a3)"},
    78286:  {"type": 4, "name": "time",      "label": "Depósitos a Prazo (a4)"},
    110560: {"type": 5, "name": "prepaid",   "label": "Conta de Pagamento Pré-Paga (a5)"},
}
TOTAL_DEPOSIT_CONTA = 78287   # Depósito Total (a)

# -- IF-Data Resumo account codes for bank characteristics --
RESUMO_ACCOUNTS = {
    78182: "total_assets",      # Ativo Total (Total Assets)
    78186: "equity",            # Patrimônio Líquido (Equity)
}

# -- IF-Data DRE (Demonstração de Resultado) account codes --
DRE_ACCOUNTS = {
    78218: "personnel_expenses",   # Despesas de Pessoal
    78219: "admin_expenses",       # Despesas Administrativas
}

# -- 2025 COSIF accounting remap (new code -> canonical old code) --
#    Starting Jan 2025 BCB changed all Passivo & Resumo Conta numbers.
#    This dict maps each new code to its pre-2025 equivalent so the
#    rest of the pipeline can use a single set of canonical codes.
CONTA_REMAP = {
    # Passivo - deposits
    140222: 78282,   # Depósitos à Vista (a1)
    140223: 78283,   # Depósitos de Poupança (a2)
    140224: 78284,   # Depósitos Interfinanceiros (a3)
    140225: 78286,   # Depósitos a Prazo (a4)
    140226: 110560,  # Conta de Pagamento Pré-Paga (a5)
    140228: 78287,   # Depósito Total (a)
    # Passivo - other (not used by panel but harmless to remap)
    140227: 78285,   # Depósitos Outros (a6)
    140230: 78288,   # Obrigações por Operações Compromissadas (b)
    140239: 78185,   # Captações (e)
    140246: 78186,   # Patrimônio Líquido (in Passivo report)
    # Resumo
    140220: 78182,   # Ativo Total
}

# -- BCB SGS series codes --
SGS_CODES = {
    "Selic_Over":     11,     # Overnight Selic (% per business day)
    "Meta_Selic":     432,    # Selic target    (% per year)
    "CDI_Anualizado": 4389,   # CDI annualized  (% per year)
    "TR":             226,    # Taxa Referencial (% per month)
}

## 3) User-defined functions:

# 3.1) Helper functions: 

def parse_saldo(series: pd.Series) -> pd.Series:
    """
    Parse IF-Data 'Saldo' column from Brazilian CSV number format to float.
    Handles '1969895,46' (comma decimal) and '1.969.895,46' (dot thousands +
    comma decimal).  Falls back to standard parsing for values without commas.
    """
    s = series.astype(str).str.strip()
    has_comma = s.str.contains(",", na=False)

    result = pd.Series(np.nan, index=s.index, dtype=float)

    # Brazilian format
    if has_comma.any():
        br = (
            s[has_comma]
            .str.replace(".", "", regex=False)   # remove thousands sep
            .str.replace(",", ".", regex=False)   # comma -> dot decimal
        )
        result[has_comma] = pd.to_numeric(br, errors="coerce")

    # Standard / integer format
    if (~has_comma).any():
        result[~has_comma] = pd.to_numeric(s[~has_comma], errors="coerce")

    return result

# 3.2) Data loading functions:

def load_if_data(source_dir: str = IF_DATA_DIR) -> pd.DataFrame:
    """
    Load and concatenate quarterly IF-Data Prudential Conglomerates CSVs.

    Keeps only the Passivo (report 3) and Demonstração de Resultado (report 4)
    rows, which contain deposit stocks and DRE cost items respectively.
    """
    pattern = os.path.join(source_dir, "IF_DATA_Values_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No IF-Data files found in {source_dir}")

    logging.info(f"Loading {len(files)} IF-Data files from {source_dir}")
    frames = []
    for fpath in files:
        try:
            df = pd.read_csv(
                fpath,
                encoding="latin-1",
                sep=",",
                dtype=str,
                skip_blank_lines=True,
                on_bad_lines="skip",
            )
            df.columns = [c.strip() for c in df.columns]
            df = df.dropna(how="all")

            # Keep only the three reports we need
            df = df[
                df["NomeRelatorio"].isin(
                    ["Passivo", "Demonstração de Resultado", "Resumo"]
                )
            ]
            if not df.empty:
                frames.append(df)
                logging.info(f"  {os.path.basename(fpath)}: {len(df)} rows")
        except Exception as e:
            logging.error(f"  Error loading {fpath}: {e}")

    if not frames:
        raise ValueError("No valid IF-Data frames loaded")

    df_all = pd.concat(frames, ignore_index=True)

    # Parse numeric columns
    df_all["Saldo"]  = parse_saldo(df_all["Saldo"])
    df_all["Conta"]  = pd.to_numeric(df_all["Conta"], errors="coerce").astype("Int64")
    df_all["AnoMes"] = pd.to_numeric(df_all["AnoMes"], errors="coerce").astype("Int64")

    # --- Normalise 2025+ account codes to pre-2025 canonical codes ---
    remap_mask = df_all["Conta"].isin(CONTA_REMAP)
    if remap_mask.any():
        n_remapped = remap_mask.sum()
        df_all.loc[remap_mask, "Conta"] = (
            df_all.loc[remap_mask, "Conta"].map(CONTA_REMAP).astype("Int64")
        )
        logging.info(
            f"Remapped {n_remapped:,} rows from 2025+ account codes "
            f"to canonical pre-2025 codes"
        )

    # Clean embedded newlines in labels
    if "NomeColuna" in df_all.columns:
        df_all["NomeColuna"] = (
            df_all["NomeColuna"].astype(str)
            .str.replace("\n", " ", regex=False)
            .str.strip()
        )

    logging.info(
        f"IF-Data loaded: {len(df_all):,} rows | "
        f"{df_all['CodInst'].nunique()} institutions | "
        f"{df_all['AnoMes'].nunique()} quarters "
        f"({df_all['AnoMes'].min()} - {df_all['AnoMes'].max()})"
    )
    return df_all

def load_macro_series() -> pd.DataFrame:
    """
    Load daily macro series from the pre-downloaded CSV produced by
    data_collection_1.py.

    Returns a DataFrame indexed by date with columns:
        Selic_Over, Meta_Selic, CDI_Anualizado, TR
    """
    macro_file = os.path.join(SGS_PATH, "macro_series_full.csv")

    if not os.path.exists(macro_file):
        raise FileNotFoundError(
            f"Macro series file not found: {macro_file}\n"
            "Please run data_collection_1.py first to download SGS data."
        )

    df = pd.read_csv(macro_file, index_col=0, parse_dates=True)

    if df.empty:
        raise ValueError(
            f"Macro series file is empty: {macro_file}\n"
            "Please re-run data_collection_1.py to download SGS data."
        )

    # Ensure the index is a proper DatetimeIndex (handles CSV round-trip issues)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    logging.info(f"Macro series loaded from {macro_file} ({len(df)} days)")
    return df

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
    Load all COSIF processed files (output of cosif_process_1.py).

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
    """Load IP rates generated by cosif_process_2_ip_rates.py"""
    fpath = os.path.join(cosif_dir, "ip_rates_quarterly.csv")
    if not os.path.exists(fpath):
        logging.warning("ip_rates_quarterly.csv not found. Re-run cosif_process_2_ip_rates.py.")
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

    # Expose raw expense so build_panel can apply the residual method for Type 4
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

# 3.3) Bank characteristics and segment functions:

def make_bank_chars_panel(df_ifdata: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Total Assets (78182) and Equity (78186) from Resumo report,
    compute equity_ratio and log_total_assets at the conglomerate x quarter level.

    Returns
    -------
    df_chars : DataFrame  (CodConglPrud x AnoMes) with
               total_assets, equity, equity_ratio, log_total_assets
    """
    resumo_contas = list(RESUMO_ACCOUNTS.keys())
    df_res = (
        df_ifdata.loc[
            df_ifdata["Conta"].isin(resumo_contas),
            ["CodConglPrud", "AnoMes", "Conta", "Saldo"],
        ].copy()
    )
    if df_res.empty:
        logging.warning("No Resumo accounts (78182/78186) found in IF-Data")
        return pd.DataFrame()

    # Pivot: one row per (CodConglPrud, AnoMes)
    piv = df_res.pivot_table(
        index=["CodConglPrud", "AnoMes"],
        columns="Conta", values="Saldo", aggfunc="sum",
    ).reset_index()
    piv.columns.name = None
    piv.rename(
        columns=RESUMO_ACCOUNTS,
        inplace=True,
    )

    # Equity ratio
    piv["equity_ratio"] = piv["equity"] / piv["total_assets"]
    piv.loc[~np.isfinite(piv["equity_ratio"]), "equity_ratio"] = np.nan

    # Log assets
    piv["log_total_assets"] = np.log(piv["total_assets"].clip(lower=1))
    piv.loc[~np.isfinite(piv["log_total_assets"]), "log_total_assets"] = np.nan

    logging.info(
        f"Bank chars panel: {len(piv):,} conglomeratexquarter obs from "
        f"{piv['CodConglPrud'].nunique()} conglomerates"
    )
    return piv


def make_dre_panel(
    df_dre: pd.DataFrame, df_bank_chars: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute conglomerate-level labour-cost ratios from DRE data.

    Accounts used:
        78218  Despesas de Pessoal      -> personnel_expenses
        78219  Despesas Administrativas -> admin_expenses

    DRE accounts record *cumulative-flow* values that are negative
    (expenses).  We take the absolute value and divide by contemporaneous
    total assets:

        personnel_cost_ratio = |personnel_expenses| / total_assets
        admin_cost_ratio     = |admin_expenses|      / total_assets

    Both ratios are then *lagged one quarter* per conglomerate so they are
    predetermined relative to the deposit pricing decision at time t.

    Returns DataFrame: CodConglPrud x AnoMes with columns
        personnel_cost_ratio_lag, admin_cost_ratio_lag
    """
    if df_dre is None or df_dre.empty:
        logging.warning("DRE data is empty -- skipping labour-cost ratios")
        return pd.DataFrame()
    if df_bank_chars is None or df_bank_chars.empty:
        logging.warning("Bank chars data is empty -- skipping labour-cost ratios")
        return pd.DataFrame()

    dre_contas = list(DRE_ACCOUNTS.keys())
    df = (
        df_dre
        .loc[df_dre["Conta"].isin(dre_contas),
             ["CodConglPrud", "AnoMes", "Conta", "Saldo"]]
        .copy()
    )
    if df.empty:
        logging.warning("No DRE expense rows found")
        return pd.DataFrame()

    # Pivot: one column per expense type
    piv = df.pivot_table(
        index=["CodConglPrud", "AnoMes"],
        columns="Conta", values="Saldo", aggfunc="sum",
    ).reset_index()
    piv.columns.name = None
    piv.rename(columns=dict(DRE_ACCOUNTS), inplace=True)

    # DRE cumulative flows reset each calendar year (Jan = quarterly value,
    # Dec = full-year cumulative).  Convert to pure quarterly flows by
    # differencing within each calendar year.
    piv = piv.sort_values(["CodConglPrud", "AnoMes"])

    for col in ["personnel_expenses", "admin_expenses"]:
        if col not in piv.columns:
            piv[col] = np.nan
            continue
        piv["_year"] = piv["AnoMes"] // 100
        piv[col] = (
            piv.groupby(["CodConglPrud", "_year"])[col]
            .transform(lambda s: s.diff().where(s.shift(1).notna(), s))
        )
    piv.drop(columns=["_year"], errors="ignore", inplace=True)

    # Merge total_assets from bank chars panel
    assets = df_bank_chars[["CodConglPrud", "AnoMes", "total_assets"]].copy()
    piv = piv.merge(assets, on=["CodConglPrud", "AnoMes"], how="left")

    # Compute ratios (absolute value of expenses / total assets)
    for exp_col, ratio_col in [
        ("personnel_expenses", "personnel_cost_ratio"),
        ("admin_expenses",     "admin_cost_ratio"),
    ]:
        if exp_col in piv.columns:
            piv[ratio_col] = piv[exp_col].abs() / piv["total_assets"]
            piv.loc[~np.isfinite(piv[ratio_col]), ratio_col] = np.nan
        else:
            piv[ratio_col] = np.nan

    # Lag one quarter per conglomerate (so ratios are predetermined)
    piv = piv.sort_values(["CodConglPrud", "AnoMes"])
    for ratio_col, lag_col in [
        ("personnel_cost_ratio", "personnel_cost_ratio_lag"),
        ("admin_cost_ratio",     "admin_cost_ratio_lag"),
    ]:
        piv[lag_col] = (
            piv.groupby("CodConglPrud")[ratio_col].shift(1)
        )

    result = piv[["CodConglPrud", "AnoMes", "personnel_cost_ratio_lag",
                  "admin_cost_ratio_lag"]].copy()

    n_valid = result["personnel_cost_ratio_lag"].notna().sum()
    logging.info(
        f"DRE ratios panel: {len(result):,} obs | "
        f"{result['CodConglPrud'].nunique()} conglomerates | "
        f"personnel_cost_ratio_lag non-null: {n_valid:,}"
    )
    return result


def make_segment_panel(df_congl_map: pd.DataFrame) -> pd.DataFrame:
    """
    Extract prudential segment (S1-S5) from the conglomerate mapping
    (already loaded from IF-Data List files by load_conglomerate_mapping).

    For conglomerated institutions, the segment is taken at the conglomerate
    level (CodConglomeradoPrudencial).  For standalone institutions (those
    whose CodConglPrud == CNPJ), the segment comes from their own record.

    Parameters
    ----------
    df_congl_map : DataFrame  as returned by load_conglomerate_mapping()
                   Must include Sr column.

    Returns
    -------
    df_seg : DataFrame  (CodConglPrud) with segment, seg_S2 ... seg_S5 dummies.
    """
    if df_congl_map is None or df_congl_map.empty:
        logging.warning("No conglomerate mapping -- skipping segment")
        return pd.DataFrame()

    if "Sr" not in df_congl_map.columns:
        logging.warning("Mapping lacks 'Sr' column -- skipping segment")
        return pd.DataFrame()

    # Build CodConglPrud -> Sr.  Use CodConglomeradoPrudencial as entity id
    # (for conglomerates this is the C-code; for standalones it's the CNPJ).
    df = df_congl_map[[
        "CodConglomeradoPrudencial", "AnoMes_list", "Sr"
    ]].copy()
    df.rename(columns={"CodConglomeradoPrudencial": "CodConglPrud"}, inplace=True)
    df["CodConglPrud"] = df["CodConglPrud"].astype(str).str.strip()

    # Take latest observation per entity
    df = df.sort_values("AnoMes_list").drop_duplicates(
        subset=["CodConglPrud"], keep="last"
    )

    # Clean segment
    df["segment"] = df["Sr"].str.strip().str.upper()
    valid_segments = {"S1", "S2", "S3", "S4", "S5"}
    df.loc[~df["segment"].isin(valid_segments), "segment"] = np.nan

    # Dummies (S1 omitted)
    for s in ["S2", "S3", "S4", "S5"]:
        df[f"seg_{s}"] = (df["segment"] == s).astype(int)

    result = df[
        ["CodConglPrud", "segment", "seg_S2", "seg_S3", "seg_S4", "seg_S5"]
    ].copy()

    n_ccodes = result["CodConglPrud"].str.startswith("C").sum()
    n_standalone = len(result) - n_ccodes
    logging.info(
        f"Segment panel: {len(result):,} entities "
        f"({n_ccodes} conglomerate C-codes, {n_standalone} standalone)"
    )
    logging.info(
        f"Segment distribution: {result['segment'].value_counts().to_dict()}"
    )
    return result


# 3.4) Panel-building functions:

def make_deposit_panel(df_ifdata: pd.DataFrame):
    """
    Extract deposit balances from Passivo report, aggregated to
    prudential conglomerate level.

    Returns
    -------
    df_deposits : DataFrame  (CodConglPrud x AnoMes x deposit_type)
    df_total    : DataFrame  (CodConglPrud x AnoMes  with total_deposits column)
    """
    deposit_contas = list(DEPOSIT_ACCOUNTS.keys())

    df_dep = (
        df_ifdata.loc[df_ifdata["Conta"].isin(deposit_contas),["CodConglPrud", "AnoMes", "Conta", "Saldo"]]
        .copy()
    )
    df_dep["deposit_type"] = df_dep["Conta"].map(
        {k: v["type"] for k, v in DEPOSIT_ACCOUNTS.items()}
    )
    df_dep["deposit_type_name"] = df_dep["Conta"].map(
        {k: v["name"] for k, v in DEPOSIT_ACCOUNTS.items()}
    )

    # Aggregate to conglomerate level (sum balances of member institutions)
    df_dep = (
        df_dep
        .groupby(["CodConglPrud", "AnoMes", "deposit_type", "deposit_type_name"], as_index=False)["Saldo"]
        .sum()
    )
    df_dep.rename(columns={"Saldo": "deposit_balance"}, inplace=True)

    # Total deposits (for implicit-rate denominator)
    df_total = (
        df_ifdata.loc[df_ifdata["Conta"] == TOTAL_DEPOSIT_CONTA, ["CodConglPrud", "AnoMes", "Saldo"]]
        .groupby(["CodConglPrud", "AnoMes"], as_index=False)["Saldo"]
        .sum()
        .rename(columns={"Saldo": "total_deposits"})
    )

    logging.info(
        f"Deposit panel: {len(df_dep):,} obs | "
        f"{df_dep['CodConglPrud'].nunique()} conglomerates | "
        f"{df_dep['deposit_type'].nunique()} types"
    )
    return df_dep, df_total


def make_quarterly_macro(df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Compound daily macro rates into quarter-over-quarter aggregates aligned
    with IF-Data quarter-end months (3, 6, 9, 12).

    Returns one row per quarter with:
        AnoMes, selic_qoq, cdi_qoq, savings_rate_qoq, meta_selic
    """
    df = df_macro.copy()
    df["year"]    = df.index.year
    df["month"]   = df.index.month
    df["q_month"] = ((df["month"] - 1) // 3 + 1) * 3   # 3, 6, 9, 12

    rows = []
    for (yr, qm), grp in df.groupby(["year", "q_month"]):
        ano_mes = yr * 100 + qm

        # ---- 1) Selic QoQ: compound daily overnight rates ----
        selic = grp["Selic_Over"].dropna()
        selic_qoq = (1 + selic / 100).prod() - 1 if len(selic) else np.nan

        # ---- 2) CDI QoQ ----
        # CDI_Anualizado is % p.a.  ->  daily rate = (1 + r_aa)^(1/252) - 1
        cdi = grp["CDI_Anualizado"].dropna()
        if len(cdi):
            cdi_daily = (1 + cdi / 100) ** (1 / 252) - 1
            cdi_qoq = (1 + cdi_daily).prod() - 1
        else:
            cdi_qoq = np.nan

        # ---- 3) Meta Selic (end-of-quarter policy rate) ----
        meta = grp["Meta_Selic"].dropna()
        meta_selic = meta.iloc[-1] if len(meta) else np.nan

        # ---- 4) TR (Taxa Referencial, % per month) ----
        tr_col = "TR" if "TR" in grp.columns else "TR_Diaria"
        tr = grp[tr_col].dropna() if tr_col in grp.columns else pd.Series(dtype=float)
        avg_tr_monthly = tr.mean() if len(tr) else 0.0

        # ---- 5) Savings-deposit rate (poupança) ----
        #   Old rule (Selic > 8.5 %): TR + 0.5 % per month
        #   New rule (Selic <= 8.5 %): TR + 70 % x Selic_target / 12
        if pd.notna(meta_selic):
            if meta_selic > 8.5:
                monthly_component = 0.005               # fixed 0.5 % p.m.
            else:
                monthly_component = 0.70 * (meta_selic / 100) / 12
            monthly_savings = avg_tr_monthly / 100 + monthly_component
            savings_qoq = (1 + monthly_savings) ** 3 - 1
        else:
            savings_qoq = np.nan

        rows.append({
            "AnoMes":           ano_mes,
            "selic_qoq":        selic_qoq,
            "cdi_qoq":          cdi_qoq,
            "savings_rate_qoq": savings_qoq,
            "meta_selic":       meta_selic,
        })

    df_rates = pd.DataFrame(rows)
    logging.info(f"Quarterly macro rates computed: {len(df_rates)} quarters")
    return df_rates

## 3.4) Main panel-building function:

def _map_ifdata_to_congl(df_ifdata: pd.DataFrame, df_congl_map: pd.DataFrame) -> pd.DataFrame:
    """
    Map IF-Data CodInst to prudential conglomerates using List crosswalk.
    """
    if df_congl_map is not None and not df_congl_map.empty:
        logging.info("Mapping IF-Data CodInst to prudential conglomerates ...")

        unique_codinst = df_ifdata["CodInst"].unique()
        is_ccode = pd.Series(unique_codinst).str.startswith("C")
        ccode_map = pd.DataFrame({
            "CodInst": pd.Series(unique_codinst)[is_ccode],
            "CodConglPrud": pd.Series(unique_codinst)[is_ccode],
        })

        numeric_codinst = pd.Series(unique_codinst)[~is_ccode]
        if len(numeric_codinst) > 0:
            latest = (
                df_congl_map
                .sort_values("AnoMes_list")
                .drop_duplicates(subset=["CNPJ"], keep="last")
            )
            num_map = (
                numeric_codinst
                .to_frame("CodInst")
                .assign(CNPJ=lambda x: x["CodInst"].str.strip().str.zfill(8))
                .merge(
                    latest[["CNPJ", "CodConglomeradoPrudencial"]],
                    on="CNPJ", how="left",
                )
                .rename(columns={"CodConglomeradoPrudencial": "CodConglPrud"})
                [["CodInst", "CodConglPrud"]]
            )
            num_map["CodConglPrud"] = num_map["CodConglPrud"].fillna(
                num_map["CodInst"]
            )
            codinst_to_congl = pd.concat(
                [ccode_map, num_map], ignore_index=True
            )
        else:
            codinst_to_congl = ccode_map

        df_ifdata = df_ifdata.merge(codinst_to_congl, on="CodInst", how="left")
        df_ifdata["CodConglPrud"] = df_ifdata["CodConglPrud"].fillna(
            df_ifdata["CodInst"]
        )
    else:
        df_ifdata["CodConglPrud"] = df_ifdata["CodInst"]

    return df_ifdata

def _map_ifdata_to_congl(df_ifdata: pd.DataFrame, df_congl_map: pd.DataFrame) -> pd.DataFrame:
    """
    Map IF-Data CodInst to prudential conglomerates using List crosswalk.
    """
    if df_congl_map is not None and not df_congl_map.empty:
        logging.info("Mapping IF-Data CodInst to prudential conglomerates ...")

        unique_codinst = df_ifdata["CodInst"].unique()
        is_ccode = pd.Series(unique_codinst).str.startswith("C")
        ccode_map = pd.DataFrame({
            "CodInst": pd.Series(unique_codinst)[is_ccode],
            "CodConglPrud": pd.Series(unique_codinst)[is_ccode],
        })

        numeric_codinst = pd.Series(unique_codinst)[~is_ccode]
        if len(numeric_codinst) > 0:
            latest = (
                df_congl_map
                .sort_values("AnoMes_list")
                .drop_duplicates(subset=["CNPJ"], keep="last")
            )
            num_map = (
                numeric_codinst
                .to_frame("CodInst")
                .assign(CNPJ=lambda x: x["CodInst"].str.strip().str.zfill(8))
                .merge(
                    latest[["CNPJ", "CodConglomeradoPrudencial"]],
                    on="CNPJ", how="left",
                )
                .rename(columns={"CodConglomeradoPrudencial": "CodConglPrud"})
                [["CodInst", "CodConglPrud"]]
            )
            num_map["CodConglPrud"] = num_map["CodConglPrud"].fillna(
                num_map["CodInst"]
            )
            codinst_to_congl = pd.concat(
                [ccode_map, num_map], ignore_index=True
            )
        else:
            codinst_to_congl = ccode_map

        df_ifdata = df_ifdata.merge(codinst_to_congl, on="CodInst", how="left")
        df_ifdata["CodConglPrud"] = df_ifdata["CodConglPrud"].fillna(
            df_ifdata["CodInst"]
        )
    else:
        df_ifdata["CodConglPrud"] = df_ifdata["CodInst"]

    return df_ifdata

def _merge_bank_features(panel: pd.DataFrame, df_bank_chars: pd.DataFrame, df_dre_ratios: pd.DataFrame, df_segments: pd.DataFrame) -> pd.DataFrame:
    if not df_bank_chars.empty:
        panel = panel.merge(
            df_bank_chars[["CodConglPrud", "AnoMes",
                          "total_assets", "equity", "equity_ratio",
                          "log_total_assets"]],
            on=["CodConglPrud", "AnoMes"], how="left",
        )
        logging.info(
            f"Bank chars coverage: "
            f"{panel['total_assets'].notna().sum():,} / "
            f"{len(panel):,} obs have total_assets"
        )
    else:
        for col in ["total_assets", "equity", "equity_ratio", "log_total_assets"]:
            panel[col] = np.nan

    if df_dre_ratios is not None and not df_dre_ratios.empty:
        panel = panel.merge(
            df_dre_ratios[["CodConglPrud", "AnoMes",
                           "personnel_cost_ratio_lag", "admin_cost_ratio_lag"]],
            on=["CodConglPrud", "AnoMes"], how="left",
        )
        logging.info(
            f"DRE cost ratio coverage: "
            f"personnel={panel['personnel_cost_ratio_lag'].notna().sum():,} "
            f"/ {len(panel):,}"
        )
    else:
        panel["personnel_cost_ratio_lag"] = np.nan
        panel["admin_cost_ratio_lag"]     = np.nan
        logging.info("No DRE ratios available -- labour-cost columns will be NaN.")

    if not df_segments.empty:
        panel = panel.merge(df_segments, on="CodConglPrud", how="left")
        logging.info(
            f"Segment coverage: "
            f"{panel['segment'].notna().sum():,} / "
            f"{len(panel):,} obs have segment"
        )
    else:
        panel["segment"] = np.nan
        for s in ["S2", "S3", "S4", "S5"]:
            panel[f"seg_{s}"] = np.nan
        
    return panel

def _calculate_residual_type4_rate(panel: pd.DataFrame) -> pd.DataFrame:
    """
    F4b. Residual COSIF rate for Type 4 (time / CDB)
    --------------------------------------------------
    The public COSIF 8.1.1 blends expenses across ALL deposit types.
    """
    _lag_wide = (
        panel[["CodConglPrud", "AnoMes", "deposit_type", "lagged_deposits"]]
        .pivot_table(
            index=["CodConglPrud", "AnoMes"],
            columns="deposit_type",
            values="lagged_deposits",
            aggfunc="first",
        )
        .reset_index()
    )
    _lag_wide.columns.name = None
    _lag_wide = _lag_wide.rename(
        columns={t: f"_lag_t{t}" for t in [1, 2, 3, 4, 5] if t in _lag_wide.columns}
    )
    panel = panel.merge(_lag_wide, on=["CodConglPrud", "AnoMes"], how="left")

    for _t in [2, 3, 4, 5]:
        col = f"_lag_t{_t}"
        panel[col] = 0.0 if col not in panel.columns else panel[col].fillna(0.0)

    _imputed_t2 = panel["savings_rate_qoq"].fillna(0.0) * panel["_lag_t2"]
    _imputed_t3 = panel["cdi_qoq"].fillna(0.0)          * panel["_lag_t3"]
    
    _residual   = (panel["cosif_desp_captacao"] - _imputed_t2 - _imputed_t3).clip(lower=0.0)
    _denom_t45  = panel["_lag_t4"] + panel["_lag_t5"]

    _type4_raw = _residual / _denom_t45
    _type4_raw[~np.isfinite(_type4_raw)] = np.nan
    _type4_raw[_type4_raw > 0.5] = np.nan   
    panel["cosif_type4_rate"] = _type4_raw

    panel.drop(columns=[c for c in panel.columns if c.startswith("_lag_t")], inplace=True)
    logging.info(
        f"Residual Type-4 rate coverage: "
        f"{panel['cosif_type4_rate'].notna().sum():,} / {len(panel):,} obs"
    )
    return panel

def _compute_median_ip_rate(panel: pd.DataFrame) -> pd.DataFrame:
    _ip_mask = (
        panel["ip_prepaid_rate"].notna()
        & (panel["ip_prepaid_rate"] > 0)
    )
    if _ip_mask.any():
        _uniq_ip = (
            panel.loc[_ip_mask]
            .drop_duplicates(subset=["CodConglPrud", "AnoMes"])
        )
        _ip_count = _uniq_ip.groupby("AnoMes")["ip_prepaid_rate"].count()
        _valid_q  = _ip_count[_ip_count >= 2].index
        _uniq_ip_valid = _uniq_ip.loc[_uniq_ip["AnoMes"].isin(_valid_q)]
        if not _uniq_ip_valid.empty:
            median_ip_rate_q = (
                _uniq_ip_valid
                .groupby("AnoMes")["ip_prepaid_rate"]
                .median()
                .rename("median_ip_rate")
            )
            panel = panel.merge(median_ip_rate_q, on="AnoMes", how="left")
            logging.info(
                f"Median IP rate computed for {len(median_ip_rate_q)} quarters "
                f"(>=2 reporters) | overall median = {median_ip_rate_q.median():.6f}"
            )
        else:
            panel["median_ip_rate"] = np.nan
            logging.warning("No quarters with >=2 IP reporters -- median_ip_rate will be NaN")
    else:
        panel["median_ip_rate"] = np.nan
        logging.warning("No IP prepaid rates -- median_ip_rate will be NaN")
    return panel

def build_panel(df_ifdata: pd.DataFrame,
                df_macro: pd.DataFrame,
                df_cosif_congl: pd.DataFrame = None,
                df_congl_map: pd.DataFrame = None) -> pd.DataFrame:
    """
    Assemble the full Egan-style panel from IF-Data, macro, and COSIF.

    Parameters
    ----------
    df_ifdata      : IF-Data Prudential Conglomerates (Passivo + DRE)
    df_macro       : Daily macro series (Selic, CDI, TR)
    df_cosif_congl : COSIF implicit rates aggregated to conglomerate level
    df_congl_map   : CNPJ -> CodConglomeradoPrudencial mapping (from List)
    """
    logging.info("=" * 50)
    logging.info("Building Egan et al. (2025) panel ...")
    logging.info("=" * 50)

    # ------------------------------------------------------------------
    # 0. Map IF-Data CodInst -> CodConglomeradoPrudencial
    # ------------------------------------------------------------------
    df_ifdata = _map_ifdata_to_congl(df_ifdata, df_congl_map)

    # ------------------------------------------------------------------
    # A. Split IF-Data once (instead of scanning 2.8M rows 3 times)
    # ------------------------------------------------------------------
    all_deposit_contas = set(DEPOSIT_ACCOUNTS.keys()) | {TOTAL_DEPOSIT_CONTA}
    all_resumo_contas  = set(RESUMO_ACCOUNTS.keys())
    all_dre_contas     = set(DRE_ACCOUNTS.keys())

    df_passivo = df_ifdata[df_ifdata["Conta"].isin(all_deposit_contas)].copy()
    df_resumo  = df_ifdata[df_ifdata["Conta"].isin(all_resumo_contas)].copy()
    df_dre     = df_ifdata[df_ifdata["Conta"].isin(all_dre_contas)].copy()
    del df_ifdata  # free ~2.8M rows

    # A1. Deposit balances (CodConglPrud x AnoMes x deposit_type)
    df_deposits, df_total = make_deposit_panel(df_passivo)

    # A2. Bank characteristics from Resumo (total assets, equity)
    df_bank_chars = make_bank_chars_panel(df_resumo)

    # A3. Labour-cost ratios from DRE (personnel & admin expenses / assets)
    df_dre_ratios = make_dre_panel(df_dre, df_bank_chars)

    # A4. Prudential segments (S1-S5) -- uses mapping already in memory
    df_segments = make_segment_panel(df_congl_map)

    # C. Quarterly macro rates (AnoMes level)
    df_macro_q = make_quarterly_macro(df_macro)

    # D. Lagged deposits per type
    df_deposits = df_deposits.sort_values(
        ["CodConglPrud", "deposit_type", "AnoMes"]
    )
    df_deposits["lagged_deposits"] = (
        df_deposits
        .groupby(["CodConglPrud", "deposit_type"])["deposit_balance"]
        .shift(1)
    )

    # E. Lagged total deposits (for implicit-rate denominator)
    df_total = df_total.sort_values(["CodConglPrud", "AnoMes"])
    df_total["lagged_total_deposits"] = (
        df_total.groupby("CodConglPrud")["total_deposits"].shift(1)
    )

    # F. Merge everything onto the deposit panel
    panel = df_deposits.copy()
    panel = panel.merge(df_macro_q,  on="AnoMes",                     how="left")
    panel = panel.merge(df_total,    on=["CodConglPrud", "AnoMes"], how="left")

    panel = _merge_bank_features(panel, df_bank_chars, df_dre_ratios, df_segments)

    # F4. Merge COSIF implicit rates (conglomerate level)
    if df_cosif_congl is not None and not df_cosif_congl.empty:
        panel = panel.merge(
            df_cosif_congl,
            left_on=["CodConglPrud", "AnoMes"],
            right_on=["CodConglomeradoPrudencial", "AnoMes"],
            how="left",
        )
        if "CodConglomeradoPrudencial" in panel.columns:
            panel.drop(columns=["CodConglomeradoPrudencial"], inplace=True)
        logging.info(
            f"COSIF rate coverage: "
            f"{panel['cosif_implicit_rate'].notna().sum():,} / "
            f"{len(panel):,} obs have COSIF implicit rates"
        )
        # Ensure IP-related columns exist
        for _col in ["ip_prepaid_rate", "has_ip", "cosif_desp_captacao"]:
            if _col not in panel.columns:
                panel[_col] = np.nan if _col != "has_ip" else 0
    else:
        panel["cosif_implicit_rate"] = np.nan
        panel["cosif_desp_captacao"] = np.nan
        panel["ip_prepaid_rate"] = np.nan
        panel["has_ip"] = 0
        logging.info("No COSIF rates available -- using market proxies only.")

    # F4b. Residual COSIF rate for Type 4 (time / CDB)
    panel = _calculate_residual_type4_rate(panel)

    # G0. Compute median IP prepaid rate per quarter
    panel = _compute_median_ip_rate(panel)

    # G. Assign deposit rates by type
    #    Type 4 (time/CDB): residual COSIF rate (strips T1-T3 contamination)
    #    -> fallback to blended cosif_implicit_rate -> fallback to CDI
    type4_rate = (
        panel["cosif_type4_rate"]
        .fillna(panel["cosif_implicit_rate"])
        .fillna(panel["cdi_qoq"])
    )

    #    Type 5 (prepaid): ENDOGENOUS, conglomerate-specific
    #      - Conglomerates WITH an IP subsidiary: use ip_prepaid_rate, which
    #        is derived from the dedicated COSIF accounts introduced in the
    #        2025 BCB recode (8.1.1.9.8 expense / 4.1.9.3 stock).  NaN for
    #        pre-2025 periods (accounts did not exist) → falls back to CDI.
    #      - Conglomerates WITHOUT an IP: cross-sectional median of IPs
    #        that do report → CDI fallback.
    type5_rate = np.where(
        panel["has_ip"] == 1,
        # Conglomerate has IP: explicit prepaid rate -> median -> CDI
        panel["ip_prepaid_rate"]
            .fillna(panel["median_ip_rate"])
            .fillna(panel["cdi_qoq"]),
        # No IP in conglomerate: median IP rate -> CDI
        panel["median_ip_rate"].fillna(panel["cdi_qoq"]),
    )

    panel["deposit_rate_qoq"] = np.select(
        [
            panel["deposit_type"] == 1,                 # demand  -> 0
            panel["deposit_type"] == 2,                 # savings -> formula
            panel["deposit_type"] == 3,                 # interbank -> CDI
            panel["deposit_type"] == 4,                 # time    -> COSIF / CDI
            panel["deposit_type"] == 5,                 # prepaid -> IP-specific
        ],
        [
            0.0,
            panel["savings_rate_qoq"],
            panel["cdi_qoq"],
            type4_rate,
            type5_rate,
        ],
        default=np.nan,
    )

    # H. Risk-free rate = Selic QoQ
    panel["risk_free_qoq"] = panel["selic_qoq"]

    # I. Spread = risk-free minus deposit rate (opportunity cost for bank,
    #    i.e. how much less than the risk-free rate the bank is paying depositors)
    panel["spread_qoq"] = panel["risk_free_qoq"] - panel["deposit_rate_qoq"]

    # K. FGC deposit-insurance coverage dummy
    #    Types 1 (demand), 2 (savings), 4 (time/CDB): covered by FGC
    #      (Resolução CMN 4.222/2013, updated by CMN 4.860/2020; R$250k cap
    #       per CPF/CNPJ per institution; R$1M rolling 4-year aggregate
    #       per Resolução CMN 4.678/2018)
    #    Type 3 (interbank): explicitly excluded from FGC (wholesale B2B only)
    #    Type 5 (prepaid): NOT FGC-insured; client funds ring-fenced in
    #      government bonds / BCB reserves under BCB Res. 80/2021, Art. 22 §8
    panel["fgc_covered"] = panel["deposit_type"].isin([1, 2, 4]).astype(int)

    # J. Quarter label  (e.g. "2020Q1")
    panel["quarter"] = panel["AnoMes"].apply(
        lambda x: f"{x // 100}Q{(x % 100) // 3}" if pd.notna(x) else None
    )

    # L. Final column selection & ordering
    col_order = [
        # identifiers
        "CodConglPrud", "quarter", "AnoMes",
        "deposit_type", "deposit_type_name",
        # conglomerate classification
        "has_ip", "fgc_covered",
        # deposit quantities
        "deposit_balance", "lagged_deposits",
        # rates & spread
        "risk_free_qoq", "deposit_rate_qoq", "spread_qoq",
        # implicit rates
        "cosif_type4_rate",        # residual COSIF rate (T4+T5 expense / T4+T5 stock)
        "cosif_implicit_rate",     # blended COSIF rate (all types)
        "cosif_desp_captacao",     # raw absolute quarterly funding expense (R$)
        # ip_prepaid_rate: from dedicated COSIF accounts 8.1.1.9.8/4.1.9.3 (2025+);
        #   NaN for pre-2025 → T5 falls back to median_ip_rate → CDI
        "ip_prepaid_rate", "median_ip_rate",
        "total_deposits", "lagged_total_deposits",
        # bank characteristics
        "total_assets", "equity", "equity_ratio", "log_total_assets",
        # labour-cost ratios (lagged)
        "personnel_cost_ratio_lag", "admin_cost_ratio_lag",
        # prudential segment
        "segment", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
        # macro detail
        "meta_selic", "selic_qoq", "cdi_qoq", "savings_rate_qoq",
    ]
    col_order = [c for c in col_order if c in panel.columns]
    panel = (
        panel[col_order]
        .sort_values(["CodConglPrud", "deposit_type", "AnoMes"])
        .reset_index(drop=True)
    )

    logging.info(
        f"Panel complete: {len(panel):,} obs | "
        f"{panel['CodConglPrud'].nunique()} conglomerates | "
        f"{panel['AnoMes'].nunique()} quarters | "
        f"{panel['deposit_type'].nunique()} deposit types"
    )
    return panel

## 4) Main execution:

def __egan_aux_main__():
    logging.info("=" * 60)
    logging.info("Starting Egan et al. (2025) Panel Construction")
    logging.info("=" * 60)

    # ---- 4.1: Load IF-Data Prudential Conglomerates ----
    print("Step 1/6: Loading IF-Data Prudential Conglomerates ...")
    try:
        df_ifdata = load_if_data()
    except (FileNotFoundError, ValueError) as e:
        logging.error(str(e))
        print(f"ERROR: {e}")
        print(f"Please ensure IF-Data files exist in:\n  {IF_DATA_DIR}")
        exit(1)

    # ---- 4.2: Load macro series (pre-downloaded by data_collection_1.py) ----
    print("Step 2/6: Loading macro series (Selic, CDI, TR) ...")
    try:
        df_macro = load_macro_series()
    except FileNotFoundError as e:
        logging.error(str(e))
        print(f"ERROR: {e}")
        exit(1)

    # ---- 4.3: Load conglomerate mapping ----
    print("Step 3/6: Loading conglomerate mapping (IF-Data List) ...")
    try:
        df_congl_map = load_conglomerate_mapping()
    except (FileNotFoundError, ValueError) as e:
        logging.warning(f"Conglomerate mapping not available: {e}")
        print(f"WARNING: {e}")
        print("Proceeding without conglomerate mapping.")
        df_congl_map = None

    # ---- 4.4: Load COSIF processed data ----
    print("Step 4/6: Loading COSIF processed data ...")
    df_cosif = load_cosif_processed()

    df_cosif_congl = pd.DataFrame()
    if not df_cosif.empty and df_congl_map is not None:
        print("         Aggregating COSIF to conglomerate level ...")
        df_cosif_congl = aggregate_cosif_to_conglomerates(
            df_cosif, df_congl_map
        )
    else:
        logging.info(
            "COSIF or mapping unavailable -- skipping COSIF aggregation."
        )

    # ---- 4.5: Build panel ----
    print("Step 5/6: Building panel ...")
    panel = build_panel(
        df_ifdata, df_macro,
        df_cosif_congl=None if df_cosif_congl.empty else df_cosif_congl,
        df_congl_map=df_congl_map,
    )

    # ---- 4.6: Save outputs ----
    print("Step 6/6: Saving ...")

    # Convert all rate columns from decimal to percentage for precision
    RATE_COLS = [
        "risk_free_qoq", "deposit_rate_qoq", "spread_qoq",
        "cosif_implicit_rate", "ip_prepaid_rate",
        "median_ip_rate",
        "selic_qoq", "cdi_qoq", "savings_rate_qoq",
    ]
    for col in RATE_COLS:
        if col in panel.columns:
            panel[col] = panel[col] * 100

    panel_path = os.path.join(OUTPUT_PATH, "egan_panel_deposits.csv")
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    pa_csv.write_csv(pa.Table.from_pandas(panel, preserve_index=False), panel_path)
    logging.info(f"Panel saved to {panel_path}")

    # Also save the quarterly macro rates for reference (convert to %)
    # Reuse the df_macro_q already computed inside build_panel via panel merge
    df_macro_q = make_quarterly_macro(df_macro)  # cheap (<1s), keep for clarity
    MACRO_RATE_COLS = ["selic_qoq", "cdi_qoq", "savings_rate_qoq"]
    for col in MACRO_RATE_COLS:
        if col in df_macro_q.columns:
            df_macro_q[col] = df_macro_q[col] * 100
    macro_path = os.path.join(OUTPUT_PATH, "quarterly_macro_rates.csv")
    pa_csv.write_csv(pa.Table.from_pandas(df_macro_q, preserve_index=False), macro_path)
    logging.info(f"Quarterly macro rates saved to {macro_path}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"Panel saved: {panel_path}")
    print(f"  Shape         : {panel.shape[0]:,} rows x {panel.shape[1]} cols")
    print(f"  Conglomerates : {panel['CodConglPrud'].nunique()}")
    print(f"  Quarters      : {panel['quarter'].nunique()}"
          f"  ({panel['quarter'].min()} - {panel['quarter'].max()})")
    print(f"  Deposit types : {panel['deposit_type'].nunique()}")

    # COSIF coverage stats
    cosif_count = panel["cosif_implicit_rate"].notna().sum()
    cosif_pct   = 100 * cosif_count / len(panel) if len(panel) else 0
    print(f"  COSIF coverage: {cosif_count:,} / {len(panel):,}"
          f" ({cosif_pct:.1f}%)")
    print(f"{'=' * 60}")

    print("\n--- Deposit Balance Summary (R$ thousands) ---")
    print(
        panel.groupby("deposit_type_name")["deposit_balance"]
        .describe()
        .round(1)
        .to_string()
    )

    print("\n--- Rate Summary (% quarter-over-quarter) ---")
    rate_cols = [
        "risk_free_qoq", "deposit_rate_qoq",
        "spread_qoq", "cosif_implicit_rate",
    ]
    rate_cols = [c for c in rate_cols if c in panel.columns]
    print(panel[rate_cols].describe().round(6).to_string())

    logging.info("Script completed successfully.")
    print("\nDone.")


# ==== END OF EGAN_PANEL_BUILD ==== 

import os
import sys
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import numpy as np
import logging
import time

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

sys.path.append(os.path.dirname(__file__))


# Paths (anchored to project root, two levels above this script)
BASE         = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEPOSITS_CSV = os.path.join(BASE, "BCB", "Egan_et_al_2025_Rep", "processed", "deposits_panel.csv")

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "append_rates",
        {"deposits_csv": DEPOSITS_CSV},
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    DEPOSITS_CSV = _paths["deposits_csv"]

def append_rates_to_panel():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(f"Adding rates and spreads to {DEPOSITS_CSV}...")

    df = pd.read_csv(DEPOSITS_CSV, low_memory=False)
    
    # 1. Add AnoMes
    df["AnoMes"] = df["Year"].astype(int) * 100 + df["Quarter"].astype(int) * 3
    
    # 2. National lags
    groupby_cols = ["CodConglomeradoPrudencial", "AnoMes"]
    nat_stocks = df.groupby(groupby_cols)[["dep_a2", "dep_a3", "dep_a4", "dep_a5"]].sum().reset_index()
    nat_stocks = nat_stocks.sort_values(groupby_cols)
    
    for col in ["dep_a2", "dep_a3", "dep_a4", "dep_a5"]:
        nat_stocks[f"lag_{col}"] = nat_stocks.groupby("CodConglomeradoPrudencial")[col].shift(1).fillna(0.0)
    
    # 3. Macro 
    df_macro = load_macro_series()
    df_macro_q = make_quarterly_macro(df_macro)
    
    # 4. COSIF
    df_congl_map = load_conglomerate_mapping()
    df_cosif = load_cosif_processed()
    df_cosif_q = aggregate_cosif_to_conglomerates(df_cosif, df_congl_map)
    
    # The output of aggregate_cosif_to_conglomerates has 'year' and 'quarter', we need AnoMes inside it
    if "AnoMes" not in df_cosif_q.columns and "year" in df_cosif_q.columns and "quarter" in df_cosif_q.columns:
        df_cosif_q["AnoMes"] = df_cosif_q["year"] * 100 + df_cosif_q["quarter"] * 3
        
    aux = nat_stocks.merge(df_macro_q[["AnoMes", "selic_qoq", "cdi_qoq", "savings_rate_qoq", "meta_selic"]], on="AnoMes", how="left")
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
        # Overwrite legacy ip_prepaid_rate
        if 'ip_prepaid_rate' in aux.columns:
            aux['ip_prepaid_rate'] = aux['new_ip_rate'].fillna(aux['ip_prepaid_rate'])
        else:
            aux['ip_prepaid_rate'] = aux['new_ip_rate']
            
        aux.drop(columns=['new_ip_rate'], inplace=True)
    
    for col in ["cosif_desp_captacao", "cosif_implicit_rate", "ip_prepaid_rate"]:
        if col not in aux.columns:
            aux[col] = np.nan
            
    # T2 & T3 imputed expenses
    _imputed_t2 = aux["savings_rate_qoq"].fillna(0.0) * aux["lag_dep_a2"]
    _imputed_t3 = aux["cdi_qoq"].fillna(0.0)          * aux["lag_dep_a3"]
    _residual = (aux["cosif_desp_captacao"] - _imputed_t2 - _imputed_t3).clip(lower=0.0)
    
    _denom_t45 = aux["lag_dep_a4"] + aux["lag_dep_a5"]
    _denom_t45 = _denom_t45.replace(0.0, np.nan)
    
    _type4_raw = _residual.astype(float) / _denom_t45.astype(float)
    _type4_raw.loc[~np.isfinite(_type4_raw)] = np.nan
    _type4_raw.loc[_type4_raw > 0.5] = np.nan
    aux["cosif_type4_rate"] = _type4_raw
    
    # Median IP
    if "has_ip" not in aux.columns:
        if "is_ip" in df_congl_map.columns:
            ip_check = df_congl_map[df_congl_map["is_ip"] == 1]["CodConglomeradoPrudencial"].unique()
            aux["has_ip"] = aux["CodConglomeradoPrudencial"].isin(ip_check).astype(int)
        else:
            aux["has_ip"] = 0

    if "ip_prepaid_rate" in aux.columns:
        _ip_mask = aux["ip_prepaid_rate"].notna() & (aux["ip_prepaid_rate"] > 0)
        # Apply median directly over AnoMes
        valid_ip = aux[_ip_mask].copy()
        
        # We need groups that have at least 2 entries.
        sizes = valid_ip.groupby("AnoMes").size()
        valid_anomes = sizes[sizes >= 2].index
        
        medians = valid_ip[valid_ip["AnoMes"].isin(valid_anomes)].groupby("AnoMes")["ip_prepaid_rate"].median()
        aux = aux.merge(medians.rename("median_ip_rate"), on="AnoMes", how="left")
    else:
        aux["median_ip_rate"] = np.nan
        
    aux["rate_a1"] = 0.0
    aux["rate_a2"] = aux["savings_rate_qoq"]
    aux["rate_a3"] = aux["cdi_qoq"]
    aux["rate_a4"] = aux["cosif_type4_rate"].fillna(aux["cosif_implicit_rate"]).fillna(aux["cdi_qoq"])
    
    if "ip_prepaid_rate" in aux.columns:
        aux["rate_a5"] = np.where(
            aux["has_ip"] == 1,
            aux["ip_prepaid_rate"].fillna(aux["median_ip_rate"]).fillna(aux["cdi_qoq"]),
            aux["median_ip_rate"].fillna(aux["cdi_qoq"])
        )
    else:
        aux["rate_a5"] = aux["median_ip_rate"].fillna(aux["cdi_qoq"])
        
    aux["risk_free_qoq"] = aux["selic_qoq"]
    
    for t in [1, 2, 3, 4, 5]:
        aux[f"spread_a{t}"] = aux["risk_free_qoq"] - aux[f"rate_a{t}"]
    
    drop_cols = ["AnoMes", "risk_free_qoq", "rate_a1", "rate_a2", "rate_a3", "rate_a4", "rate_a5",
                 "spread_a1", "spread_a2", "spread_a3", "spread_a4", "spread_a5"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
                 
    cols_to_merge = ["CodConglomeradoPrudencial", "AnoMes", "risk_free_qoq"] + \
                    [f"rate_a{t}" for t in range(1, 6)] + \
                    [f"spread_a{t}" for t in range(1, 6)]
                    
    df["AnoMes"] = df["Year"].astype(int) * 100 + df["Quarter"].astype(int) * 3
    df = df.merge(aux[cols_to_merge], on=["CodConglomeradoPrudencial", "AnoMes"], how="left")
    df = df.drop(columns=["AnoMes"])
    
    pa_csv.write_csv(pa.Table.from_pandas(df, preserve_index=False), DEPOSITS_CSV)
    logging.info(f"Updated {DEPOSITS_CSV} with rates and spreads.")

if __name__ == "__main__":
    t0 = time.perf_counter()
    append_rates_to_panel()
    print(f"Done in {time.perf_counter() - t0:.2f}s")

