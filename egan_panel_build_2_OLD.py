## egan_panel_build_2.py
# Author: Pedro Feijo de Moraes
# Last edited: 2026-02-24
#
# Objective: Build an Egan et al. (2025)-style panel dataset at the
#            INDIVIDUAL INSTITUTION (CNPJ) level, using BCB IF-Data
#            Individual Institutions, COSIF, and SGS.
#
# Contrast with egan_panel_build.py:
#   - egan_panel_build.py uses IF-Data Prudential Conglomerates (Type 1),
#     which reports pre-aggregated data for conglomerates (C-codes).
#   - This script uses IF-Data Individual Institutions (Type 3), which
#     reports data at the individual CNPJ level.
#   - No conglomerate aggregation is needed; COSIF institution-level rates
#     are matched directly by CNPJ.
#
# Output panel structure: CNPJ x Quarter x Deposit_Type
#
#   Variables (same as conglomerate panel, replacing CodConglPrud -> CNPJ):
#     CNPJ                - Institution root CNPJ (8 digits)
#     quarter             - Period label (e.g. "2020Q1")
#     AnoMes              - IF-Data period code (e.g. 202003)
#     deposit_type        - 1-5 (same deposit types as conglomerate panel)
#     deposit_balance     - End-of-quarter stock (R$ thousands, IF-Data units)
#     lagged_deposits     - Previous quarter's deposit_balance
#     risk_free_qoq       - Selic overnight, compounded QoQ (%)
#     deposit_rate_qoq    - Deposit rate for the type (% qoq)
#     spread_qoq          - risk_free_rate - deposit_rate (% qoq, typically >= 0, opportunity cost)
#     cosif_implicit_rate - COSIF-derived implicit funding rate (% qoq)
#                           (directly at CNPJ level -- no aggregation needed)
#     total_deposits, lagged_total_deposits
#     total_assets, equity, equity_ratio, log_total_assets
#     segment, seg_S2 ... seg_S5
#     meta_selic
#
# Data sources:
#   - IF-Data Individual Institutions (BCB/IF Data/Individual Institutions/)
#     - Passivo report  (report 3): deposit stocks by type
#     - Resumo report   (report 1): total assets, equity
#   - COSIF quarterly institution panel (from cosif_process_2.py)
#   - IF-Data List (BCB/IF Data/List/): CNPJ -> segment mapping
#   - SGS macro series (pre-downloaded by data_collection_1.py)
#
# Prerequisites:
#   pip install pandas numpy
#   Run data_collection_1.py   -> SGS macro series
#   Run cosif_process_1.py     -> monthly COSIF processing
#   Run cosif_process_2.py     -> quarterly institution-level COSIF rates
## ---------------------------------------------------------------------------

## 1) Load necessary packages and set folders up:

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

# Primary data: IF-Data Individual Institutions (consolidated quarterly files)
IF_DATA_INDIV_DIR = os.path.join(BCB_PATH, "IF Data", "Individual Institutions")
# Fallback: Olinda API downloads (per-report files, covers 2014+)
IF_DATA_API_DIR = os.path.join(
    BCB_PATH, "Egan_et_al_2025_Rep", "raw", "IF_DATA_RAW", "Individual"
)
# DRE (Demonstração de Resultado) per-quarter API downloads
IF_DATA_FIN_DIR = os.path.join(
    BCB_PATH, "Egan_et_al_2025_Rep", "raw", "IF_DATA_RAW", "Financial"
)

# IF-Data List: CNPJ metadata (segment, status)
IF_DATA_LIST_DIR = os.path.join(BCB_PATH, "IF Data", "List")

# Egan replication I/O paths
EGAN_PATH      = os.path.join(BCB_PATH, "Egan_et_al_2025_Rep")
RAW_PATH       = os.path.join(EGAN_PATH, "raw")
SGS_PATH       = os.path.join(RAW_PATH, "SGS_RAW")
PROCESSED_PATH = os.path.join(EGAN_PATH, "processed")
OUTPUT_PATH    = os.path.join(PROCESSED_PATH, "PANEL_INTERMED")

# COSIF quarterly institution panel (from cosif_process_2.py)
COSIF_PROCESSED_PATH = os.path.join(PROCESSED_PATH, "COSIF_PROCESSED")
COSIF_QUARTERLY_FILE = os.path.join(
    COSIF_PROCESSED_PATH, "cosif_quarterly_institution.csv"
)

for _p in [RAW_PATH, SGS_PATH, PROCESSED_PATH, OUTPUT_PATH, COSIF_PROCESSED_PATH]:
    os.makedirs(_p, exist_ok=True)

# Logging
_log_file = os.path.join(SCRIPT_DIR, "egan_panel_build_2.log")
_handler  = RotatingFileHandler(
    _log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
logging.basicConfig(
    handlers=[_handler],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

## 2) Reference constants (same as conglomerate panel):

# -- IF-Data Passivo account codes for each deposit type --
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

# -- IF-Data Resumo account codes --
RESUMO_ACCOUNTS = {
    78182: "total_assets",      # Ativo Total
    78186: "equity",            # Patrimônio Líquido
}

# -- IF-Data DRE (Demonstração de Resultado) account codes --
DRE_ACCOUNTS = {
    78218: "personnel_expenses",   # Despesas de Pessoal
    78219: "admin_expenses",       # Despesas Administrativas
}

# -- IF-Data DRE: fee income and other P&L accounts (Report 4) --
# These are loaded alongside DRE_ACCOUNTS (same file / filter).
# Like DRE_ACCOUNTS they record cumulative flows and receive the same
# within-year differencing treatment in make_dre_panel().
FEE_INCOME_ACCOUNTS = {
    78216: "rendas_servicos",         # Rendas de Prestação de Serviços (d1)
    78217: "rendas_tarifas",          # Rendas de Tarifas Bancárias (d2)
    78213: "resultado_provisao_cdl",  # Resultado de Provisão para CDL (b5)  [expense]
    78220: "despesas_tributarias",    # Despesas Tributárias (d5)             [expense]
}

# -- IF-Data Passivo: wholesale / non-deposit funding accounts (Report 3) --
# Loaded alongside deposit stock accounts from the same IF Data files.
WHOLESALE_FUNDING_ACCOUNTS = {
    78288: "repos",                # Obrigações por Operações Compromissadas (b)
    78289: "lci",                  # Letras de Crédito Imobiliário (c1)
    78290: "lca",                  # Letras de Crédito do Agronegócio (c2)
    78291: "letras_financeiras",   # Letras Financeiras (c3)
    78295: "emprestimos_repasses", # Obrigações por Empréstimos e Repasses (d)
}
# NOTE: 140230->78288 (repos) remap already in CONTA_REMAP.
# 2025 codes for 78289-78295 are not yet mapped; these will be NaN for
# quarters from 2025 onward until the remap is extended.

# -- IF-Data Capital: Basel / regulatory capital accounts (Report 5) --
# The BCB Olinda API reports these only from 2014Q4 onward.
CAPITAL_ACCOUNTS = {
    79664: "indice_basileia",          # Índice de Basileia (%)
    79659: "indice_capital_principal", # Índice de Capital Principal / CET1 (%)
    79661: "razao_alavancagem",        # Razão de Alavancagem (%)
    79650: "rwa_credito",              # RWA para Risco de Crédito (R$)
    79665: "rwa_total",                # Ativos Ponderados pelo Risco – total (R$)
}

# -- IF-Data Ativo: credit portfolio accounts (Report 2) --
CREDIT_PORTFOLIO_ACCOUNTS = {
    78191: "credito_bruto",       # Operações de Crédito – gross (d1)
    78192: "provisao_credito",    # Provisão sobre Operações de Crédito (d2)
    78190: "tvm_derivativos",     # TVM e Instrumentos Financeiros Derivativos (c)
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

# Pre-compute all target account codes for early filtering at read time
# Include both pre-2025 canonical codes AND 2025+ new codes
ALL_TARGET_CONTAS = (
    set(DEPOSIT_ACCOUNTS.keys())
    | {TOTAL_DEPOSIT_CONTA}
    | set(RESUMO_ACCOUNTS.keys())
    | set(CONTA_REMAP.keys())                    # 2025+ codes (will be remapped later)
    | set(WHOLESALE_FUNDING_ACCOUNTS.keys())     # LCI, LCA, Letras, Repos, Empréstimos
    | set(CAPITAL_ACCOUNTS.keys())               # Basel, CET1, leverage, RWA
    | set(CREDIT_PORTFOLIO_ACCOUNTS.keys())      # Credit portfolio, NPL, TVM
)
# String versions for filtering before numeric conversion
_TARGET_CONTAS_STR = {str(c) for c in ALL_TARGET_CONTAS}

# DRE account codes (loaded separately from Financial directory)
# Includes both cost accounts (DRE_ACCOUNTS) and fee income accounts (FEE_INCOME_ACCOUNTS)
_DRE_CONTAS_STR = (
    {str(c) for c in DRE_ACCOUNTS.keys()}
    | {str(c) for c in FEE_INCOME_ACCOUNTS.keys()}
)

# Columns we actually need from the IF-Data CSVs
_USECOLS = ["CodInst", "AnoMes", "Conta", "Saldo"]

# -- BCB SGS series codes --
SGS_CODES = {
    "Selic_Over":     11,
    "Meta_Selic":     432,
    "CDI_Anualizado": 4389,
    "TR":             226,
}

## 3) User-defined functions:

# -- 3.1) Helpers ------------------------------------------------------------

def parse_saldo(series: pd.Series) -> pd.Series:
    """Parse IF-Data 'Saldo' from Brazilian CSV number format to float."""
    s = series.astype(str).str.strip()
    has_comma = s.str.contains(",", na=False)
    result = pd.Series(np.nan, index=s.index, dtype=float)

    if has_comma.any():
        br = (
            s[has_comma]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        result[has_comma] = pd.to_numeric(br, errors="coerce")

    if (~has_comma).any():
        result[~has_comma] = pd.to_numeric(s[~has_comma], errors="coerce")

    return result


# -- 3.2) Data-loading functions ---------------------------------------------

def load_if_data_individual() -> pd.DataFrame:
    """
    Load IF-Data Individual Institutions CSVs.

    Strategy:
    1. Try consolidated files in BCB/IF Data/Individual Institutions/
       (IF_DATA_Values_*.csv -- each has all reports for a quarter,
       typically starting from 2016).
    2. Supplement with per-report API downloads in
       BCB/Egan_et_al_2025_Rep/raw/IF_DATA_RAW/Individual/
       (IF_DATA_Passivo_*.csv, IF_DATA_*Resultado*.csv, IF_DATA_Resumo_*.csv,
       from 2014+).
    3. Deduplicate on (CodInst, AnoMes, Conta) to avoid double-counting
       quarters covered by both sources.

    Only Passivo and Resumo reports are kept.
    """
    frames = []

    # --- Source 1: Consolidated files ---
    pattern_consolidated = os.path.join(IF_DATA_INDIV_DIR, "IF_DATA_Values_*.csv")
    consolidated_files = sorted(glob.glob(pattern_consolidated))
    if consolidated_files:
        logging.info(
            f"Loading {len(consolidated_files)} consolidated Individual "
            f"Institution files from {IF_DATA_INDIV_DIR}"
        )
        for fpath in consolidated_files:
            try:
                df = pd.read_csv(
                    fpath, encoding="latin-1", sep=",",
                    usecols=_USECOLS,
                    dtype=str, skip_blank_lines=True, on_bad_lines="skip",
                )
                df.columns = [c.strip() for c in df.columns]
                df.dropna(how="all", inplace=True)
                # Early filter: keep only rows for target Conta codes
                df = df[df["Conta"].str.strip().isin(_TARGET_CONTAS_STR)]
                if not df.empty:
                    frames.append(df)
                    logging.info(f"  {os.path.basename(fpath)}: {len(df)} rows")
            except Exception as e:
                logging.error(f"  Error loading {fpath}: {e}")

    # --- Source 2: Per-report API downloads ---
    if os.path.isdir(IF_DATA_API_DIR):
        # Match Passivo, Resumo, and DRE (with possible encoding mangling)
        api_patterns = [
            os.path.join(IF_DATA_API_DIR, "IF_DATA_Passivo_*.csv"),
            os.path.join(IF_DATA_API_DIR, "IF_DATA_Resumo_*.csv"),
            os.path.join(IF_DATA_API_DIR, "IF_DATA_Ativo_*.csv"),
            os.path.join(IF_DATA_API_DIR, "IF_DATA_Capital_*.csv"),
        ]
        api_files = []
        for p in api_patterns:
            api_files.extend(sorted(glob.glob(p)))

        if api_files:
            logging.info(
                f"Loading {len(api_files)} per-report API files from "
                f"{IF_DATA_API_DIR}"
            )
            for fpath in api_files:
                try:
                    df = pd.read_csv(
                        fpath, encoding="utf-8", sep=",",
                        usecols=_USECOLS,
                        dtype=str, skip_blank_lines=True, on_bad_lines="skip",
                    )
                    df.columns = [c.strip() for c in df.columns]
                    df.dropna(how="all", inplace=True)
                    # Early filter: keep only rows for target Conta codes
                    df = df[df["Conta"].str.strip().isin(_TARGET_CONTAS_STR)]
                    if not df.empty:
                        frames.append(df)
                except Exception as e:
                    logging.error(f"  Error loading {fpath}: {e}")

    if not frames:
        raise FileNotFoundError(
            "No Individual Institution IF-Data files found.\n"
            f"  Looked in: {IF_DATA_INDIV_DIR}\n"
            f"        and: {IF_DATA_API_DIR}"
        )

    df_all = pd.concat(frames, ignore_index=True)
    del frames  # free memory immediately

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

    # Standardise CNPJ: CodInst in Individual = institution's root CNPJ
    df_all["CNPJ"] = df_all["CodInst"].astype(str).str.strip().str.zfill(8)
    df_all.drop(columns=["CodInst"], inplace=True)  # free redundant column
    df_all["CNPJ"] = df_all["CNPJ"].astype("category")  # memory optimisation

    # Remove system-level aggregate rows (CNPJ "00000000" = SFN total)
    n_before = len(df_all)
    df_all = df_all[df_all["CNPJ"] != "00000000"].copy()
    if n_before - len(df_all) > 0:
        logging.info(
            f"Removed {n_before - len(df_all):,} SFN aggregate rows "
            f"(CodInst = 00000000)"
        )

    # Deduplicate: same (CNPJ, AnoMes, Conta) may appear from both sources
    df_all = df_all.drop_duplicates(subset=["CNPJ", "AnoMes", "Conta"], keep="first")

    logging.info(
        f"IF-Data Individual loaded: {len(df_all):,} rows | "
        f"{df_all['CNPJ'].nunique()} institutions | "
        f"{df_all['AnoMes'].nunique()} quarters "
        f"({df_all['AnoMes'].min()} - {df_all['AnoMes'].max()})"
    )
    return df_all


def load_if_data_financial() -> pd.DataFrame:
    """
    Load IF-Data Financial (Demonstração de Resultado) CSVs.

    Reads per-quarter DRE files from:
        BCB/Egan_et_al_2025_Rep/raw/IF_DATA_RAW/Financial/
        (IF_DATA_Demonstra*_de_Resultado_*.csv -- one file per quarter,
        covers 2014+, latin-1 encoding).

    Returns a tidy DataFrame with columns:
        CNPJ, AnoMes, Conta, Saldo
    filtered to DRE account codes (Despesas de Pessoal=78218,
    Despesas Administrativas=78219).
    """
    if not os.path.isdir(IF_DATA_FIN_DIR):
        logging.warning(
            f"Financial DRE directory not found: {IF_DATA_FIN_DIR}\n"
            "  labour-cost ratios will be missing from panel."
        )
        return pd.DataFrame()

    pattern = os.path.join(IF_DATA_FIN_DIR, "IF_DATA_Demonstra*.csv")
    fin_files = sorted(glob.glob(pattern))
    if not fin_files:
        logging.warning(
            f"No DRE files matching 'IF_DATA_Demonstra*.csv' in {IF_DATA_FIN_DIR}"
        )
        return pd.DataFrame()

    logging.info(
        f"Loading {len(fin_files)} DRE (Demonstração de Resultado) files "
        f"from {IF_DATA_FIN_DIR}"
    )
    frames = []
    for fpath in fin_files:
        try:
            df = pd.read_csv(
                fpath, encoding="latin-1", sep=",",
                usecols=_USECOLS,
                dtype=str, skip_blank_lines=True, on_bad_lines="skip",
            )
            df.columns = [c.strip() for c in df.columns]
            df.dropna(how="all", inplace=True)
            # Early filter: keep only DRE expense accounts
            df = df[df["Conta"].str.strip().isin(_DRE_CONTAS_STR)]
            if not df.empty:
                frames.append(df)
        except Exception as e:
            logging.error(f"  Error loading {fpath}: {e}")

    if not frames:
        logging.warning("No DRE rows found after filtering -- returning empty")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    del frames

    # Parse numeric columns
    df_all["Saldo"]  = parse_saldo(df_all["Saldo"])
    df_all["Conta"]  = pd.to_numeric(df_all["Conta"], errors="coerce").astype("Int64")
    df_all["AnoMes"] = pd.to_numeric(df_all["AnoMes"], errors="coerce").astype("Int64")

    # Standardise CNPJ
    df_all["CNPJ"] = df_all["CodInst"].astype(str).str.strip().str.zfill(8)
    df_all.drop(columns=["CodInst"], inplace=True)
    df_all["CNPJ"] = df_all["CNPJ"].astype("category")

    # Remove SFN aggregate rows
    df_all = df_all[df_all["CNPJ"] != "00000000"].copy()

    # Deduplicate
    df_all = df_all.drop_duplicates(subset=["CNPJ", "AnoMes", "Conta"], keep="first")

    logging.info(
        f"DRE data loaded: {len(df_all):,} rows | "
        f"{df_all['CNPJ'].nunique()} institutions | "
        f"{df_all['AnoMes'].nunique()} quarters "
        f"({df_all['AnoMes'].min()} - {df_all['AnoMes'].max()})"
    )
    return df_all


def load_macro_series() -> pd.DataFrame:
    """Load daily macro series from SGS CSV (same as conglomerate version)."""
    macro_file = os.path.join(SGS_PATH, "macro_series_full.csv")
    if not os.path.exists(macro_file):
        raise FileNotFoundError(
            f"Macro series file not found: {macro_file}\n"
            "Please run data_collection_1.py first."
        )
    df = pd.read_csv(macro_file, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError(f"Macro series file is empty: {macro_file}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    logging.info(f"Macro series loaded: {len(df)} daily obs")
    return df


def load_institution_info(list_dir: str = IF_DATA_LIST_DIR) -> pd.DataFrame:
    """
    Load CNPJ-level metadata from IF-Data List files: segment (Sr),
    institution name, status.

    Returns a DataFrame with one row per CNPJ (latest snapshot).
    """
    pattern = os.path.join(list_dir, "IF_DATA_List_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        logging.warning(f"No List files found in {list_dir}")
        return pd.DataFrame()

    logging.info(f"Loading institution info from {len(files)} List files")
    frames = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath, dtype=str, encoding="utf-8")
            df.columns = [c.strip() for c in df.columns]
            # Keep individual active institutions
            mask = (df["Td"] == "I") & (df["Situacao"] == "A")
            keep_cols = ["CodInst", "Data"]
            if "Sr" in df.columns:
                keep_cols.append("Sr")
            if "SegmentoTb" in df.columns:
                keep_cols.append("SegmentoTb")
            sub = df.loc[mask, keep_cols].copy()
            sub.rename(columns={"CodInst": "CNPJ", "Data": "AnoMes_list"}, inplace=True)
            frames.append(sub)
        except Exception as e:
            logging.warning(f"  Error loading {fpath}: {e}")

    if not frames:
        return pd.DataFrame()

    df_info = pd.concat(frames, ignore_index=True)
    df_info["CNPJ"] = df_info["CNPJ"].str.strip().str.zfill(8)
    df_info["AnoMes_list"] = pd.to_numeric(
        df_info["AnoMes_list"], errors="coerce"
    ).astype("Int64")

    # Take latest snapshot per CNPJ
    df_info = (
        df_info
        .sort_values("AnoMes_list")
        .drop_duplicates(subset=["CNPJ"], keep="last")
    )

    # -- Identify Payment Institutions (IPs) --
    # IPs can ONLY offer prepaid payment accounts (Lei 12.865/2013),
    # so their entire "Despesas de Captação" = prepaid remuneration cost.
    # This means their COSIF blended implicit rate IS the prepaid rate.
    if "SegmentoTb" in df_info.columns:
        df_info["is_ip"] = (
            df_info["SegmentoTb"]
            .str.contains("Instituição de Pagamento", case=False, na=False)
            .astype(int)
        )
        n_ip = df_info["is_ip"].sum()
        logging.info(
            f"Payment Institutions identified: {n_ip} IPs out of "
            f"{len(df_info)} total institutions"
        )
    else:
        df_info["is_ip"] = 0
        logging.warning(
            "SegmentoTb column not found in List files -- "
            "cannot identify Payment Institutions"
        )

    logging.info(
        f"Institution info loaded: {len(df_info)} unique CNPJs"
    )
    return df_info


def load_cosif_quarterly(
    cosif_file: str = COSIF_QUARTERLY_FILE,
) -> pd.DataFrame:
    """
    Load the quarterly institution-level COSIF panel produced by
    cosif_process_2.py.

    Returns a DataFrame with CNPJ, AnoMes, cosif_implicit_rate.
    """
    # Try Parquet first (faster), fall back to CSV
    parquet_file = cosif_file.replace(".csv", ".parquet")
    if os.path.exists(parquet_file):
        logging.info(f"Loading COSIF quarterly from Parquet: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        df["CNPJ"] = df["CNPJ"].astype(str).str.strip().str.zfill(8)
        df["AnoMes"] = pd.to_numeric(df["AnoMes"], errors="coerce").astype("Int64")
        # Rename raw expense column to canonical name used by build_panel
        if "Desp_Captacao_Q" in df.columns:
            df = df.rename(columns={"Desp_Captacao_Q": "cosif_desp_captacao"})
        logging.info(
            f"COSIF quarterly loaded: {len(df):,} obs | "
            f"{df['CNPJ'].nunique()} institutions | "
            f"rate coverage: {df['cosif_implicit_rate'].notna().sum():,}"
        )
        keep_cols = [c for c in ["CNPJ", "AnoMes", "cosif_implicit_rate",
                                   "cosif_prepaid_rate", "cosif_desp_captacao"]
                    if c in df.columns]
        return df[keep_cols]

    if not os.path.exists(cosif_file):
        logging.warning(
            f"COSIF quarterly file not found: {cosif_file}\n"
            "Run cosif_process_2.py first. Proceeding without COSIF rates."
        )
        return pd.DataFrame()

    df = pd.read_csv(cosif_file, dtype={"CNPJ": str})
    df["CNPJ"] = df["CNPJ"].astype(str).str.strip().str.zfill(8)
    df["AnoMes"] = pd.to_numeric(df["AnoMes"], errors="coerce").astype("Int64")
    if "Desp_Captacao_Q" in df.columns:
        df = df.rename(columns={"Desp_Captacao_Q": "cosif_desp_captacao"})

    logging.info(
        f"COSIF quarterly loaded: {len(df):,} obs | "
        f"{df['CNPJ'].nunique()} institutions | "
        f"rate coverage: {df['cosif_implicit_rate'].notna().sum():,}"
    )
    keep_cols = [c for c in ["CNPJ", "AnoMes", "cosif_implicit_rate",
                              "cosif_prepaid_rate", "cosif_desp_captacao"]
                if c in df.columns]
    return df[keep_cols]


# -- 3.3) Panel sub-component functions --------------------------------------

def make_deposit_panel(df_ifdata: pd.DataFrame):
    """
    Extract deposit balances from Passivo report at institution level.

    Returns
    -------
    df_deposits : DataFrame  (CNPJ x AnoMes x deposit_type)
    df_total    : DataFrame  (CNPJ x AnoMes with total_deposits column)
    """
    deposit_contas = list(DEPOSIT_ACCOUNTS.keys())

    df_dep = (
        df_ifdata
        .loc[df_ifdata["Conta"].isin(deposit_contas),
             ["CNPJ", "AnoMes", "Conta", "Saldo"]]
        .copy()
    )
    df_dep["deposit_type"] = df_dep["Conta"].map(
        {k: v["type"] for k, v in DEPOSIT_ACCOUNTS.items()}
    )
    df_dep["deposit_type_name"] = df_dep["Conta"].map(
        {k: v["name"] for k, v in DEPOSIT_ACCOUNTS.items()}
    )

    # At institution level, each (CNPJ, AnoMes, deposit_type) should be unique.
    # Sum just in case of duplicates from the two data sources.
    # observed=True prevents Cartesian-product explosion from categorical CNPJ
    df_dep = (
        df_dep
        .groupby(["CNPJ", "AnoMes", "deposit_type", "deposit_type_name"],
                 as_index=False, observed=True)["Saldo"]
        .sum()
    )
    df_dep.rename(columns={"Saldo": "deposit_balance"}, inplace=True)

    # Total deposits
    df_total = (
        df_ifdata
        .loc[df_ifdata["Conta"] == TOTAL_DEPOSIT_CONTA,
             ["CNPJ", "AnoMes", "Saldo"]]
        .groupby(["CNPJ", "AnoMes"], as_index=False, observed=True)["Saldo"]
        .sum()
        .rename(columns={"Saldo": "total_deposits"})
    )

    logging.info(
        f"Deposit panel: {len(df_dep):,} obs | "
        f"{df_dep['CNPJ'].nunique()} institutions | "
        f"{df_dep['deposit_type'].nunique()} types"
    )
    return df_dep, df_total


def make_bank_chars_panel(df_ifdata: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Total Assets and Equity from Resumo report at institution level.
    Compute equity_ratio and log_total_assets.
    """
    resumo_contas = list(RESUMO_ACCOUNTS.keys())
    df_res = (
        df_ifdata
        .loc[df_ifdata["Conta"].isin(resumo_contas),
             ["CNPJ", "AnoMes", "Conta", "Saldo"]]
        .copy()
    )
    if df_res.empty:
        logging.warning("No Resumo accounts found in Individual IF-Data")
        return pd.DataFrame()

    piv = df_res.pivot_table(
        index=["CNPJ", "AnoMes"],
        columns="Conta", values="Saldo", aggfunc="sum",
        observed=True,
    ).reset_index()
    piv.columns.name = None
    piv.rename(
        columns={k: v for k, v in RESUMO_ACCOUNTS.items()},
        inplace=True,
    )

    piv["equity_ratio"] = piv["equity"] / piv["total_assets"]
    piv.loc[~np.isfinite(piv["equity_ratio"]), "equity_ratio"] = np.nan

    piv["log_total_assets"] = np.log(piv["total_assets"].clip(lower=1))
    piv.loc[~np.isfinite(piv["log_total_assets"]), "log_total_assets"] = np.nan

    logging.info(
        f"Bank chars panel: {len(piv):,} institution-quarter obs from "
        f"{piv['CNPJ'].nunique()} institutions"
    )
    return piv


def make_dre_panel(
    df_dre: pd.DataFrame, df_bank_chars: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute institution-level labour-cost ratios from DRE data.

    Accounts used:
        78218  Despesas de Pessoal   -> personnel_expenses
        78219  Despesas Administrativas -> admin_expenses

    DRE accounts record *cumulative-flow* values that are negative
    (expenses).  We take the absolute value and divide by contemporaneous
    total assets to obtain quarter-level ratios:

        personnel_cost_ratio = |personnel_expenses| / total_assets
        admin_cost_ratio     = |admin_expenses|      / total_assets

    Both ratios are then *lagged one quarter* per institution so they are
    predetermined relative to the deposit pricing decision at time t.

    Returns DataFrame: CNPJ x AnoMes with columns
        personnel_cost_ratio_lag, admin_cost_ratio_lag
    """
    if df_dre is None or df_dre.empty:
        logging.warning("DRE data is empty -- skipping labour-cost ratios")
        return pd.DataFrame()
    if df_bank_chars is None or df_bank_chars.empty:
        logging.warning("Bank chars data is empty -- skipping labour-cost ratios")
        return pd.DataFrame()

    # All DRE flow accounts: cost + fee-income
    all_dre_contas = list(DRE_ACCOUNTS.keys()) + list(FEE_INCOME_ACCOUNTS.keys())
    all_dre_names  = {**DRE_ACCOUNTS, **FEE_INCOME_ACCOUNTS}

    df = (
        df_dre
        .loc[df_dre["Conta"].isin(all_dre_contas),
             ["CNPJ", "AnoMes", "Conta", "Saldo"]]
        .copy()
    )
    if df.empty:
        logging.warning("No DRE expense rows found")
        return pd.DataFrame()

    # Pivot: one column per account
    piv = df.pivot_table(
        index=["CNPJ", "AnoMes"],
        columns="Conta", values="Saldo", aggfunc="sum",
        observed=True,
    ).reset_index()
    piv.columns.name = None
    piv.rename(columns={k: v for k, v in all_dre_names.items()}, inplace=True)

    # DRE cumulative flows reset each calendar year (Jan = quarterly value,
    # Dec = full-year cumulative).  Convert to pure quarterly flows by
    # differencing within each calendar year.
    # AnoMes format: YYYYMM  (e.g. 202003 = Q1 2020, 202012 = Q4 2020)
    piv = piv.sort_values(["CNPJ", "AnoMes"])
    piv["_year"] = piv["AnoMes"] // 100

    for col in list(all_dre_names.values()):
        if col not in piv.columns:
            piv[col] = np.nan
            continue
        # Within each (CNPJ, year) group, diff to get quarterly flow;
        # first quarter of each year keeps its cumulative value as the flow.
        piv[col] = (
            piv.groupby(["CNPJ", "_year"], observed=True)[col]
            .transform(lambda s: s.diff().where(s.shift(1).notna(), s))
        )
    piv.drop(columns=["_year"], errors="ignore", inplace=True)

    # Merge total_assets from bank chars panel
    assets = df_bank_chars[["CNPJ", "AnoMes", "total_assets"]].copy()
    piv = piv.merge(assets, on=["CNPJ", "AnoMes"], how="left")

    # Compute ratios (all scaled by total_assets; expenses use abs value)
    ratio_specs = [
        # (source_col,              ratio_col,                   use_abs)
        ("personnel_expenses",      "personnel_cost_ratio",       True),
        ("admin_expenses",          "admin_cost_ratio",           True),
        ("rendas_servicos",         "service_fee_income_ratio",   False),
        ("rendas_tarifas",          "tariff_fee_income_ratio",    False),
        ("resultado_provisao_cdl",  "npl_provision_ratio",        True),
        ("despesas_tributarias",    "tax_cost_ratio",             True),
    ]
    for src_col, ratio_col, use_abs in ratio_specs:
        if src_col in piv.columns:
            vals = piv[src_col].abs() if use_abs else piv[src_col]
            piv[ratio_col] = vals / piv["total_assets"]
            piv.loc[~np.isfinite(piv[ratio_col]), ratio_col] = np.nan
        else:
            piv[ratio_col] = np.nan

    # Lag one quarter per institution (so ratios are predetermined)
    piv = piv.sort_values(["CNPJ", "AnoMes"])
    lag_pairs = [
        ("personnel_cost_ratio",     "personnel_cost_ratio_lag"),
        ("admin_cost_ratio",         "admin_cost_ratio_lag"),
        ("service_fee_income_ratio", "service_fee_income_ratio_lag"),
        ("tariff_fee_income_ratio",  "tariff_fee_income_ratio_lag"),
        ("npl_provision_ratio",      "npl_provision_ratio_lag"),
        ("tax_cost_ratio",           "tax_cost_ratio_lag"),
    ]
    for ratio_col, lag_col in lag_pairs:
        piv[lag_col] = piv.groupby("CNPJ", observed=True)[ratio_col].shift(1)

    result_cols = (
        ["CNPJ", "AnoMes",
         "personnel_cost_ratio_lag", "admin_cost_ratio_lag",
         "service_fee_income_ratio_lag", "tariff_fee_income_ratio_lag",
         "npl_provision_ratio_lag", "tax_cost_ratio_lag"]
    )
    result = piv[result_cols].copy()

    n_valid = result["personnel_cost_ratio_lag"].notna().sum()
    logging.info(
        f"DRE ratios panel: {len(result):,} obs | "
        f"{result['CNPJ'].nunique()} institutions | "
        f"personnel_cost_ratio_lag non-null: {n_valid:,} | "
        f"service_fee_income_ratio_lag non-null: "
        f"{result['service_fee_income_ratio_lag'].notna().sum():,}"
    )
    return result


def make_funding_mix_panel(
    df_wholesale: pd.DataFrame, df_bank_chars: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute wholesale / non-deposit funding composition ratios.

    Accounts (Report 3 / Passivo):
        78288  Repos (Obrigações por Operações Compromissadas)
        78289  LCI (Letras de Crédito Imobiliário)
        78290  LCA (Letras de Crédito do Agronegócio)
        78291  Letras Financeiras
        78295  Empréstimos e Repasses

    Derived ratios (all as fraction of total_assets; lagged one quarter):
        lci_lca_ratio_lag         = (LCI + LCA) / total_assets
        ltf_ratio_lag             = Letras Financeiras / total_assets
        repos_ratio_lag           = Repos / total_assets
        emprestimos_ratio_lag     = Empréstimos e Repasses / total_assets
        wholesale_ratio_lag       = (LCI + LCA + LTF + Empréstimos) / total_assets

    Instrument validity for types 4-5: all of these variables are on the
    bank's LIABILITY side (how it funds itself wholesale).  They directly
    shift the bank's marginal funding cost but are excluded from depositor
    utility by construction.  LCI+LCA / assets is particularly strong
    because LCI/LCA are tax-exempt and substitute for type-4 time deposits
    from the bank's perspective.
    """
    if df_wholesale is None or df_wholesale.empty:
        logging.warning("No wholesale funding data -- skipping funding mix panel")
        return pd.DataFrame()

    contas = list(WHOLESALE_FUNDING_ACCOUNTS.keys())
    df = df_wholesale[df_wholesale["Conta"].isin(contas)].copy()
    if df.empty:
        logging.warning("Wholesale funding accounts not found in IF-Data -- skipping")
        return pd.DataFrame()

    piv = df.pivot_table(
        index=["CNPJ", "AnoMes"],
        columns="Conta", values="Saldo", aggfunc="sum",
        observed=True,
    ).reset_index()
    piv.columns.name = None
    piv.rename(columns=WHOLESALE_FUNDING_ACCOUNTS, inplace=True)

    for col in WHOLESALE_FUNDING_ACCOUNTS.values():
        if col not in piv.columns:
            piv[col] = np.nan

    piv = piv.merge(
        df_bank_chars[["CNPJ", "AnoMes", "total_assets"]],
        on=["CNPJ", "AnoMes"], how="left",
    )

    piv["lci_lca_ratio"]     = (piv["lci"].fillna(0) + piv["lca"].fillna(0)) / piv["total_assets"]
    piv["ltf_ratio"]         = piv["letras_financeiras"] / piv["total_assets"]
    piv["repos_ratio"]       = piv["repos"] / piv["total_assets"]
    piv["emprestimos_ratio"] = piv["emprestimos_repasses"] / piv["total_assets"]
    piv["wholesale_ratio"]   = (
        piv["lci"].fillna(0) + piv["lca"].fillna(0)
        + piv["letras_financeiras"].fillna(0)
        + piv["emprestimos_repasses"].fillna(0)
    ) / piv["total_assets"]

    ratio_cols = [
        "lci_lca_ratio", "ltf_ratio",
        "repos_ratio", "emprestimos_ratio", "wholesale_ratio",
    ]
    for col in ratio_cols:
        piv.loc[~np.isfinite(piv[col]), col] = np.nan

    piv = piv.sort_values(["CNPJ", "AnoMes"])
    result_cols = ["CNPJ", "AnoMes"]
    for col in ratio_cols:
        lag_col = f"{col}_lag"
        piv[lag_col] = piv.groupby("CNPJ", observed=True)[col].shift(1)
        result_cols.append(lag_col)

    logging.info(
        f"Funding mix panel: {len(piv):,} obs | "
        f"lci_lca_ratio_lag non-null: {piv['lci_lca_ratio_lag'].notna().sum():,}"
    )
    return piv[result_cols].copy()


def make_capital_panel(df_capital: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Basel / regulatory capital ratios from IF-Data Report 5.

    Accounts:
        79664  Índice de Basileia (%)
        79659  Índice de Capital Principal / CET1 (%)
        79661  Razão de Alavancagem (%)
        79650  RWA para Risco de Crédito (R$)
        79665  Ativos Ponderados pelo Risco – total (R$)

    All ratio columns are lagged one quarter.

    Instrument validity: the Basel ratio and CET1 are regulatory capital
    adequacy ratios set by BACEN rules.  A better-capitalised bank faces
    lower regulatory pressure to attract deposits and can therefore offer
    lower type-4/5 rates.  Excluded from depositor utility by construction.
    Coverage starts 2014Q4 (BCB Olinda Capital report availability).
    """
    if df_capital is None or df_capital.empty:
        logging.warning("No capital adequacy data -- skipping capital panel")
        return pd.DataFrame()

    contas = list(CAPITAL_ACCOUNTS.keys())
    df = df_capital[df_capital["Conta"].isin(contas)].copy()
    if df.empty:
        logging.warning("Capital accounts not found in IF-Data -- skipping")
        return pd.DataFrame()

    # Capital ratios are reported as levels (not flows): take mean when duplicated
    piv = df.pivot_table(
        index=["CNPJ", "AnoMes"],
        columns="Conta", values="Saldo", aggfunc="mean",
        observed=True,
    ).reset_index()
    piv.columns.name = None
    piv.rename(columns=CAPITAL_ACCOUNTS, inplace=True)

    ratio_cols = [c for c in CAPITAL_ACCOUNTS.values() if c in piv.columns]

    piv = piv.sort_values(["CNPJ", "AnoMes"])
    result_cols = ["CNPJ", "AnoMes"]
    for col in ratio_cols:
        lag_col = f"{col}_lag"
        piv[lag_col] = piv.groupby("CNPJ", observed=True)[col].shift(1)
        result_cols.append(lag_col)

    n_basileia = piv["indice_basileia_lag"].notna().sum() if "indice_basileia_lag" in piv.columns else 0
    logging.info(
        f"Capital panel: {len(piv):,} obs | "
        f"indice_basileia_lag non-null: {n_basileia:,}"
    )
    return piv[result_cols].copy()


def make_credit_panel(
    df_credit: pd.DataFrame,
    df_bank_chars: pd.DataFrame,
    df_total_deposits: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute credit portfolio ratios from IF-Data Report 2 (Ativo).

    Accounts:
        78191  Operações de Crédito – gross (d1)
        78192  Provisão sobre Operações de Crédito (d2)
        78190  TVM e Instrumentos Financeiros Derivativos (c)

    Derived ratios (lagged one quarter):
        credit_assets_ratio_lag   = credito_bruto / total_assets
        npl_ratio_lag             = |provisao_credito| / credito_bruto  (provisioning rate)
        tvm_assets_ratio_lag      = tvm_derivativos / total_assets
        credit_deposit_ratio_lag  = credito_bruto / total_deposits

    Instrument validity for types 4-5:
        credit_assets_ratio  -- banks with heavier loan books need more deposit
                                funding -> upward pressure on deposit rates
        npl_ratio            -- higher write-offs signal a riskier asset mix;
                                banks may need to pay a risk premium on deposits
        tvm_assets_ratio     -- securities-heavy banks have short-duration assets;
                                lower need for long-term deposit funding -> lower type-4 rates
        credit_deposit_ratio -- ALM pressure: high loans/deposits forces higher type-4 rates
    """
    if df_credit is None or df_credit.empty:
        logging.warning("No credit portfolio data -- skipping credit panel")
        return pd.DataFrame()

    contas = list(CREDIT_PORTFOLIO_ACCOUNTS.keys())
    df = df_credit[df_credit["Conta"].isin(contas)].copy()
    if df.empty:
        logging.warning("Credit portfolio accounts not found in IF-Data -- skipping")
        return pd.DataFrame()

    piv = df.pivot_table(
        index=["CNPJ", "AnoMes"],
        columns="Conta", values="Saldo", aggfunc="sum",
        observed=True,
    ).reset_index()
    piv.columns.name = None
    piv.rename(columns=CREDIT_PORTFOLIO_ACCOUNTS, inplace=True)

    for col in CREDIT_PORTFOLIO_ACCOUNTS.values():
        if col not in piv.columns:
            piv[col] = np.nan

    piv = piv.merge(
        df_bank_chars[["CNPJ", "AnoMes", "total_assets"]],
        on=["CNPJ", "AnoMes"], how="left",
    )
    piv = piv.merge(
        df_total_deposits[["CNPJ", "AnoMes", "total_deposits"]],
        on=["CNPJ", "AnoMes"], how="left",
    )

    piv["credit_assets_ratio"]  = piv["credito_bruto"] / piv["total_assets"]
    piv["npl_ratio"]            = piv["provisao_credito"].abs() / piv["credito_bruto"].clip(lower=1)
    piv["tvm_assets_ratio"]     = piv["tvm_derivativos"] / piv["total_assets"]
    piv["credit_deposit_ratio"] = piv["credito_bruto"] / piv["total_deposits"].clip(lower=1)

    ratio_cols = [
        "credit_assets_ratio", "npl_ratio",
        "tvm_assets_ratio", "credit_deposit_ratio",
    ]
    for col in ratio_cols:
        piv.loc[~np.isfinite(piv[col]), col] = np.nan

    piv = piv.sort_values(["CNPJ", "AnoMes"])
    result_cols = ["CNPJ", "AnoMes"]
    for col in ratio_cols:
        lag_col = f"{col}_lag"
        piv[lag_col] = piv.groupby("CNPJ", observed=True)[col].shift(1)
        result_cols.append(lag_col)

    logging.info(
        f"Credit portfolio panel: {len(piv):,} obs | "
        f"credit_assets_ratio_lag non-null: {piv['credit_assets_ratio_lag'].notna().sum():,}"
    )
    return piv[result_cols].copy()


def make_segment_panel(df_info: pd.DataFrame) -> pd.DataFrame:
    """
    Extract prudential segment (S1-S5) per CNPJ from the institution
    info loaded from IF-Data List files.

    Returns DataFrame with CNPJ, segment, seg_S2...seg_S5 dummies.
    """
    if df_info is None or df_info.empty:
        logging.warning("No institution info -- skipping segment")
        return pd.DataFrame()
    if "Sr" not in df_info.columns:
        logging.warning("Institution info lacks 'Sr' column -- skipping segment")
        return pd.DataFrame()

    df = df_info[["CNPJ", "Sr"]].copy()
    df["segment"] = df["Sr"].str.strip().str.upper()
    valid_segments = {"S1", "S2", "S3", "S4", "S5"}
    df.loc[~df["segment"].isin(valid_segments), "segment"] = np.nan

    for s in ["S2", "S3", "S4", "S5"]:
        df[f"seg_{s}"] = (df["segment"] == s).astype(int)

    result = df[
        ["CNPJ", "segment", "seg_S2", "seg_S3", "seg_S4", "seg_S5"]
    ].copy()

    logging.info(
        f"Segment panel: {len(result)} institutions | "
        f"distribution: {result['segment'].value_counts().to_dict()}"
    )
    return result


def make_quarterly_macro(df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Compound daily macro rates into QoQ aggregates.
    Identical to the conglomerate version.
    """
    df = df_macro.copy()
    df["year"]    = df.index.year
    df["month"]   = df.index.month
    df["q_month"] = ((df["month"] - 1) // 3 + 1) * 3

    rows = []
    for (yr, qm), grp in df.groupby(["year", "q_month"]):
        ano_mes = yr * 100 + qm

        # Selic QoQ
        selic = grp["Selic_Over"].dropna()
        selic_qoq = (1 + selic / 100).prod() - 1 if len(selic) else np.nan

        # CDI QoQ
        cdi = grp["CDI_Anualizado"].dropna()
        if len(cdi):
            cdi_daily = (1 + cdi / 100) ** (1 / 252) - 1
            cdi_qoq = (1 + cdi_daily).prod() - 1
        else:
            cdi_qoq = np.nan

        # Meta Selic (end-of-quarter)
        meta = grp["Meta_Selic"].dropna()
        meta_selic = meta.iloc[-1] if len(meta) else np.nan

        # TR
        tr_col = "TR" if "TR" in grp.columns else "TR_Diaria"
        tr = grp[tr_col].dropna() if tr_col in grp.columns else pd.Series(dtype=float)
        avg_tr_monthly = tr.mean() if len(tr) else 0.0

        # Savings rate (poupança)
        if pd.notna(meta_selic):
            if meta_selic > 8.5:
                monthly_component = 0.005
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
    logging.info(f"Quarterly macro rates: {len(df_rates)} quarters")
    return df_rates


# -- 3.4) Main panel-building function --------------------------------------

def build_panel(
    df_ifdata: pd.DataFrame,
    df_macro: pd.DataFrame,
    df_cosif_q: pd.DataFrame = None,
    df_info: pd.DataFrame = None,
    df_dre: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Assemble the full institution-level Egan-style panel.

    Parameters
    ----------
    df_ifdata  : IF-Data Individual Institutions (Passivo + DRE + Resumo)
    df_macro   : Daily macro series (Selic, CDI, TR)
    df_cosif_q : COSIF quarterly institution rates (from cosif_process_2.py)
    df_info    : Institution info from List files (segment)
    df_dre     : IF-Data Demonstração de Resultado (from load_if_data_financial)
    """
    logging.info("=" * 60)
    logging.info("Building INSTITUTION-LEVEL Egan et al. (2025) panel ...")
    logging.info("=" * 60)

    # ------------------------------------------------------------------
    # A. Split IF-Data once (instead of scanning all rows multiple times)
    # ------------------------------------------------------------------
    all_deposit_contas   = set(DEPOSIT_ACCOUNTS.keys()) | {TOTAL_DEPOSIT_CONTA}
    all_resumo_contas    = set(RESUMO_ACCOUNTS.keys())
    all_wholesale_contas = set(WHOLESALE_FUNDING_ACCOUNTS.keys())
    all_capital_contas   = set(CAPITAL_ACCOUNTS.keys())
    all_credit_contas    = set(CREDIT_PORTFOLIO_ACCOUNTS.keys())

    df_passivo   = df_ifdata[df_ifdata["Conta"].isin(all_deposit_contas)].copy()
    df_resumo    = df_ifdata[df_ifdata["Conta"].isin(all_resumo_contas)].copy()
    df_wholesale = df_ifdata[df_ifdata["Conta"].isin(all_wholesale_contas)].copy()
    df_capital_d = df_ifdata[df_ifdata["Conta"].isin(all_capital_contas)].copy()
    df_credit_d  = df_ifdata[df_ifdata["Conta"].isin(all_credit_contas)].copy()
    del df_ifdata  # free large input DataFrame

    # A1. Deposit balances (CNPJ x AnoMes x deposit_type)
    df_deposits, df_total = make_deposit_panel(df_passivo)

    # A2. Bank characteristics from Resumo (total assets, equity)
    df_bank_chars = make_bank_chars_panel(df_resumo)

    # A3. Labour-cost and fee-income ratios from DRE
    df_dre_ratios = make_dre_panel(df_dre, df_bank_chars)

    # A4. Prudential segments (S1-S5)
    df_segments   = make_segment_panel(df_info)

    # A5. Wholesale funding mix (LCI/LCA ratios, Letras Financeiras, etc.)
    df_funding_mix = make_funding_mix_panel(df_wholesale, df_bank_chars)

    # A6. Capital adequacy (Basel ratio, CET1, leverage)
    df_capital_p   = make_capital_panel(df_capital_d)

    # A7. Credit portfolio (credit/assets, NPL, TVM/assets)
    df_credit_p    = make_credit_panel(df_credit_d, df_bank_chars, df_total)

    # A8. Quarterly macro rates (AnoMes level)
    df_macro_q    = make_quarterly_macro(df_macro)

    # -- B. Lagged deposits per type --
    df_deposits = df_deposits.sort_values(["CNPJ", "deposit_type", "AnoMes"])
    df_deposits["lagged_deposits"] = (
        df_deposits
        .groupby(["CNPJ", "deposit_type"], observed=True)["deposit_balance"]
        .shift(1)
    )

    # -- C. Lagged total deposits --
    df_total = df_total.sort_values(["CNPJ", "AnoMes"])
    df_total["lagged_total_deposits"] = (
        df_total.groupby("CNPJ", observed=True)["total_deposits"].shift(1)
    )

    # -- D. Merge everything onto the deposit panel --
    panel = df_deposits.copy()
    panel = panel.merge(df_macro_q,  on="AnoMes",               how="left")
    panel = panel.merge(df_total,    on=["CNPJ", "AnoMes"],     how="left")

    # Bank characteristics
    if not df_bank_chars.empty:
        panel = panel.merge(
            df_bank_chars[["CNPJ", "AnoMes",
                          "total_assets", "equity", "equity_ratio",
                          "log_total_assets"]],
            on=["CNPJ", "AnoMes"], how="left",
        )
        logging.info(
            f"Bank chars coverage: "
            f"{panel['total_assets'].notna().sum():,} / {len(panel):,}"
        )
    else:
        for col in ["total_assets", "equity", "equity_ratio", "log_total_assets"]:
            panel[col] = np.nan

    # DRE labour-cost, fee-income, provisioning, and tax ratios (all lagged)
    _dre_lag_cols = [
        "personnel_cost_ratio_lag", "admin_cost_ratio_lag",
        "service_fee_income_ratio_lag", "tariff_fee_income_ratio_lag",
        "npl_provision_ratio_lag", "tax_cost_ratio_lag",
    ]
    if df_dre_ratios is not None and not df_dre_ratios.empty:
        _dre_merge_cols = ["CNPJ", "AnoMes"] + [
            c for c in _dre_lag_cols if c in df_dre_ratios.columns
        ]
        panel = panel.merge(
            df_dre_ratios[_dre_merge_cols],
            on=["CNPJ", "AnoMes"], how="left",
        )
        logging.info(
            f"DRE cost ratio coverage: "
            f"personnel={panel['personnel_cost_ratio_lag'].notna().sum():,} "
            f"service_fee={panel.get('service_fee_income_ratio_lag', pd.Series(dtype=float)).notna().sum():,} "
            f"/ {len(panel):,}"
        )
    else:
        for _col in _dre_lag_cols:
            panel[_col] = np.nan
        logging.info("No DRE ratios available -- cost/fee-income columns will be NaN.")

    # Wholesale funding mix (institution-level; LCI/LCA, LTF, repos, empréstimos)
    _funding_lag_cols = [
        "lci_lca_ratio_lag", "ltf_ratio_lag",
        "repos_ratio_lag", "emprestimos_ratio_lag", "wholesale_ratio_lag",
    ]
    if df_funding_mix is not None and not df_funding_mix.empty:
        _fund_merge_cols = ["CNPJ", "AnoMes"] + [
            c for c in _funding_lag_cols if c in df_funding_mix.columns
        ]
        panel = panel.merge(
            df_funding_mix[_fund_merge_cols],
            on=["CNPJ", "AnoMes"], how="left",
        )
        logging.info(
            f"Funding mix coverage: "
            f"lci_lca_ratio_lag={panel['lci_lca_ratio_lag'].notna().sum():,} "
            f"/ {len(panel):,}"
        )
    else:
        for _col in _funding_lag_cols:
            panel[_col] = np.nan
        logging.info("No wholesale funding data -- funding mix columns will be NaN.")

    # Capital adequacy (Basel, CET1, leverage, RWA; institution-level)
    _capital_lag_cols = [
        "indice_basileia_lag", "indice_capital_principal_lag",
        "razao_alavancagem_lag", "rwa_credito_lag", "rwa_total_lag",
    ]
    if df_capital_p is not None and not df_capital_p.empty:
        _cap_merge_cols = ["CNPJ", "AnoMes"] + [
            c for c in _capital_lag_cols if c in df_capital_p.columns
        ]
        panel = panel.merge(
            df_capital_p[_cap_merge_cols],
            on=["CNPJ", "AnoMes"], how="left",
        )
        logging.info(
            f"Capital adequacy coverage: "
            f"indice_basileia_lag={panel.get('indice_basileia_lag', pd.Series(dtype=float)).notna().sum():,} "
            f"/ {len(panel):,}"
        )
    else:
        for _col in _capital_lag_cols:
            panel[_col] = np.nan
        logging.info("No capital adequacy data -- Basel columns will be NaN.")

    # Credit portfolio (credit/assets, NPL rate, TVM/assets; institution-level)
    _credit_lag_cols = [
        "credit_assets_ratio_lag", "npl_ratio_lag",
        "tvm_assets_ratio_lag", "credit_deposit_ratio_lag",
    ]
    if df_credit_p is not None and not df_credit_p.empty:
        _cred_merge_cols = ["CNPJ", "AnoMes"] + [
            c for c in _credit_lag_cols if c in df_credit_p.columns
        ]
        panel = panel.merge(
            df_credit_p[_cred_merge_cols],
            on=["CNPJ", "AnoMes"], how="left",
        )
        logging.info(
            f"Credit portfolio coverage: "
            f"credit_assets_ratio_lag={panel['credit_assets_ratio_lag'].notna().sum():,} "
            f"/ {len(panel):,}"
        )
    else:
        for _col in _credit_lag_cols:
            panel[_col] = np.nan
        logging.info("No credit portfolio data -- credit ratio columns will be NaN.")

    # Segments
    if not df_segments.empty:
        panel = panel.merge(df_segments, on="CNPJ", how="left")
        logging.info(
            f"Segment coverage: "
            f"{panel['segment'].notna().sum():,} / {len(panel):,}"
        )
    else:
        panel["segment"] = np.nan
        for s in ["S2", "S3", "S4", "S5"]:
            panel[f"seg_{s}"] = np.nan

    # COSIF implicit rates (direct CNPJ match -- no aggregation needed!)
    if df_cosif_q is not None and not df_cosif_q.empty:
        cosif_merge_cols = [
            c for c in [
                "CNPJ", "AnoMes",
                "cosif_implicit_rate",
                "cosif_prepaid_rate",   # explicit prepaid rate (2025+ reporters only)
                "cosif_desp_captacao",
            ] if c in df_cosif_q.columns
        ]
        panel = panel.merge(
            df_cosif_q[cosif_merge_cols],
            on=["CNPJ", "AnoMes"],
            how="left",
        )
        logging.info(
            f"COSIF rate coverage: "
            f"{panel['cosif_implicit_rate'].notna().sum():,} / {len(panel):,} | "
            f"prepaid rate: {panel['cosif_prepaid_rate'].notna().sum():,}"
            if "cosif_prepaid_rate" in panel.columns else
            f"COSIF rate coverage: "
            f"{panel['cosif_implicit_rate'].notna().sum():,} / {len(panel):,}"
        )
        if "cosif_desp_captacao" not in panel.columns:
            panel["cosif_desp_captacao"] = np.nan
        if "cosif_prepaid_rate" not in panel.columns:
            panel["cosif_prepaid_rate"] = np.nan
    else:
        panel["cosif_implicit_rate"] = np.nan
        panel["cosif_prepaid_rate"]  = np.nan
        panel["cosif_desp_captacao"] = np.nan
        logging.info("No COSIF rates available -- using market proxies only.")

    # -- E0. Merge Payment Institution (IP) flag --
    # IPs report zero in COSIF 8.1.1 (Despesas de Captação) because prepaid
    # accounts are not classified as deposits under Lei 12.865/2013.  From
    # the 2025 BCB recode, the dedicated account 8119800007 captures their
    # prepaid remuneration expense, available through cosif_prepaid_rate.
    # For pre-2025 data, T5 falls back to the cross-sectional median → CDI.
    if df_info is not None and "is_ip" in df_info.columns:
        ip_flag = df_info[["CNPJ", "is_ip"]].copy()
        panel = panel.merge(ip_flag, on="CNPJ", how="left")
        panel["is_ip"] = panel["is_ip"].fillna(0).astype(int)
    else:
        panel["is_ip"] = 0

    n_ip_obs = (panel["is_ip"] == 1).sum()
    logging.info(
        f"IP flag merged: {n_ip_obs:,} obs from Payment Institutions "
        f"({panel.loc[panel['is_ip']==1, 'CNPJ'].nunique()} unique IPs)"
    )

    # -- E1. Compute quarterly median prepaid rate --
    # Source: any institution with a non-null, positive cosif_prepaid_rate
    # (the dedicated 2025+ COSIF account 8.1.1.9.8).  We do NOT gate on is_ip
    # because SegmentoTb text-matching can miss reporters. Pre-2025 all values
    # are NaN so median will also be NaN -> T5 falls back to CDI.
    # Fall back to is_ip-filtered cosif_implicit_rate only if the prepaid
    # rate column is absent entirely (older parquet without the column).
    if "cosif_prepaid_rate" in panel.columns:
        _pr_mask = panel["cosif_prepaid_rate"].notna() & (panel["cosif_prepaid_rate"] > 0)
        _pr_src  = panel.loc[_pr_mask, ["AnoMes", "cosif_prepaid_rate"]]
    else:
        _pr_mask = (
            (panel["is_ip"] == 1)
            & panel["cosif_implicit_rate"].notna()
            & (panel["cosif_implicit_rate"] > 0)
        )
        _pr_src = panel.loc[_pr_mask, ["AnoMes", "cosif_implicit_rate"]].rename(
            columns={"cosif_implicit_rate": "cosif_prepaid_rate"}
        )
    if not _pr_src.empty:
        # Use unique CNPJ x AnoMes pairs to avoid counting multi-type repeats
        _pr_src_uniq = (
            _pr_src
            .rename(columns={_pr_src.columns[-1]: "cosif_prepaid_rate"})
            .drop_duplicates(subset=["AnoMes", "cosif_prepaid_rate"])
        )
        _pr_count = _pr_src_uniq.groupby("AnoMes")["cosif_prepaid_rate"].count()
        # Require at least 2 reporters per quarter so one outlier doesn't
        # contaminate the median used by all other institutions in that quarter
        _valid_quarters = _pr_count[_pr_count >= 2].index
        median_ip_rate_q = (
            _pr_src_uniq.loc[_pr_src_uniq["AnoMes"].isin(_valid_quarters)]
            .groupby("AnoMes")["cosif_prepaid_rate"]
            .median()
            .rename("median_ip_rate")
        )
        if not median_ip_rate_q.empty:
            panel = panel.merge(
                median_ip_rate_q, on="AnoMes", how="left"
            )
            logging.info(
                f"Median prepaid rate computed for {len(median_ip_rate_q)} quarters "
                f"(>=2 reporters) | overall median = {median_ip_rate_q.median():.6f}"
            )
        else:
            panel["median_ip_rate"] = np.nan
            logging.warning(
                "No quarters with >=2 prepaid reporters -- median_ip_rate will be NaN"
            )
    else:
        panel["median_ip_rate"] = np.nan
        logging.warning(
            "No prepaid COSIF rates available -- median_ip_rate will be NaN (CDI fallback)"
        )

    # -- E2. Residual COSIF rate for Type 4 (time / CDB) --
    # Pivot lagged deposits to wide so we can compute known-type imputed expenses
    _lag_wide = (
        panel[["CNPJ", "AnoMes", "deposit_type", "lagged_deposits"]]
        .pivot_table(
            index=["CNPJ", "AnoMes"],
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
    panel = panel.merge(_lag_wide, on=["CNPJ", "AnoMes"], how="left")
    for _t in [2, 3, 4, 5]:
        col = f"_lag_t{_t}"
        if col not in panel.columns:
            panel[col] = 0.0
        else:
            panel[col] = panel[col].fillna(0.0)

    _imputed_t2 = panel["savings_rate_qoq"].fillna(0.0) * panel["_lag_t2"]
    _imputed_t3 = panel["cdi_qoq"].fillna(0.0)          * panel["_lag_t3"]
    _residual   = (panel["cosif_desp_captacao"].abs() - _imputed_t2 - _imputed_t3).clip(lower=0.0)
    _denom_t45  = panel["_lag_t4"] + panel["_lag_t5"]
    _type4_raw  = _residual / _denom_t45
    _type4_raw[~np.isfinite(_type4_raw)] = np.nan
    _type4_raw[_type4_raw > 0.5] = np.nan
    panel["cosif_type4_rate"] = _type4_raw
    panel.drop(columns=[c for c in panel.columns if c.startswith("_lag_t")], inplace=True)

    logging.info(
        f"Residual Type-4 rate coverage: "
        f"{panel['cosif_type4_rate'].notna().sum():,} / {len(panel):,} obs"
    )

    # -- E3. Assign deposit rates by type --
    #   Type 4 (CDB/time): residual COSIF rate -> blended COSIF -> CDI fallback
    type4_rate = (
        panel["cosif_type4_rate"]
        .fillna(panel["cosif_implicit_rate"])
        .fillna(panel["cdi_qoq"])
    )

    #   Type 5 (prepaid): ENDOGENOUS, institution-specific
    #     Use the explicit prepaid COSIF rate (2025+ reporters via account
    #     8119800007 / 4193000009) for any institution that has it, regardless
    #     of the is_ip flag (which relies on SegmentoTb text matching and may
    #     miss some reporters).  Cascade:
    #       cosif_prepaid_rate -> median prepaid rate (cross-section) -> CDI
    #     NOTE: cosif_implicit_rate is intentionally EXCLUDED from the T5 chain.
    #     For banks, cosif_implicit_rate = blended total-deposit funding cost
    #     (all types combined), which is NOT an appropriate measure of prepaid
    #     deposit costs.  For true IPs, cosif_implicit_rate is NaN anyway.
    if "cosif_prepaid_rate" in panel.columns:
        type5_rate = (
            panel["cosif_prepaid_rate"]
            .fillna(panel["median_ip_rate"])
            .fillna(panel["cdi_qoq"])
        )
    else:
        type5_rate = panel["median_ip_rate"].fillna(panel["cdi_qoq"])

    panel["deposit_rate_qoq"] = np.select(
        [
            panel["deposit_type"] == 1,   # demand  -> 0
            panel["deposit_type"] == 2,   # savings -> formula
            panel["deposit_type"] == 3,   # interbank -> CDI
            panel["deposit_type"] == 4,   # time -> COSIF / CDI
            panel["deposit_type"] == 5,   # prepaid -> IP-specific
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

    # -- F. Risk-free rate & spread --
    panel["risk_free_qoq"] = panel["selic_qoq"]
    # Spread = risk-free minus deposit rate (opportunity cost: how much less
    # than the risk-free rate the bank is paying depositors)
    panel["spread_qoq"]    = panel["risk_free_qoq"] - panel["deposit_rate_qoq"]

    # -- G. Quarter label --
    panel["quarter"] = panel["AnoMes"].apply(
        lambda x: f"{x // 100}Q{(x % 100) // 3}" if pd.notna(x) else None
    )

    # -- H. Column ordering & sort --
    col_order = [
        # identifiers
        "CNPJ", "quarter", "AnoMes",
        "deposit_type", "deposit_type_name",
        # institution classification
        "is_ip",
        # deposit quantities
        "deposit_balance", "lagged_deposits",
        # rates & spread
        "risk_free_qoq", "deposit_rate_qoq", "spread_qoq",
        # implicit rates (used to BUILD deposit rates; not valid as instruments for T4/T5)
        "cosif_type4_rate",
        "cosif_implicit_rate", "cosif_prepaid_rate",
        "cosif_desp_captacao", "median_ip_rate",
        "total_deposits", "lagged_total_deposits",
        # bank characteristics (valid BLP/cost instruments)
        "total_assets", "equity", "equity_ratio", "log_total_assets",
        # DRE cost ratios (lagged) -- valid cost instruments
        "personnel_cost_ratio_lag", "admin_cost_ratio_lag",
        "tax_cost_ratio_lag",
        # DRE fee-income ratios (lagged) -- cross-subsidy instruments for T4/T5
        "service_fee_income_ratio_lag", "tariff_fee_income_ratio_lag",
        # DRE credit risk (lagged)
        "npl_provision_ratio_lag",
        # wholesale funding mix (lagged) -- substitutability instruments for T4/T5
        "lci_lca_ratio_lag", "ltf_ratio_lag",
        "repos_ratio_lag", "emprestimos_ratio_lag", "wholesale_ratio_lag",
        # capital adequacy (lagged) -- regulatory cost instruments for T4/T5
        "indice_basileia_lag", "indice_capital_principal_lag",
        "razao_alavancagem_lag", "rwa_credito_lag", "rwa_total_lag",
        # credit portfolio (lagged) -- ALM-pressure instruments for T4
        "credit_assets_ratio_lag", "npl_ratio_lag",
        "tvm_assets_ratio_lag", "credit_deposit_ratio_lag",
        # prudential segment
        "segment", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
        # macro detail
        "meta_selic", "selic_qoq", "cdi_qoq", "savings_rate_qoq",
    ]
    col_order = [c for c in col_order if c in panel.columns]
    panel = (
        panel[col_order]
        .sort_values(["CNPJ", "deposit_type", "AnoMes"])
        .reset_index(drop=True)
    )

    logging.info(
        f"Panel complete: {len(panel):,} obs | "
        f"{panel['CNPJ'].nunique()} institutions | "
        f"{panel['AnoMes'].nunique()} quarters | "
        f"{panel['deposit_type'].nunique()} deposit types"
    )
    return panel


## 4) Main execution:

if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Starting INSTITUTION-LEVEL Panel Construction")
    logging.info("=" * 60)

    # -- 4.1: Load IF-Data Individual Institutions --
    print("Step 1/6: Loading IF-Data Individual Institutions ...")
    try:
        df_ifdata = load_if_data_individual()
    except (FileNotFoundError, ValueError) as e:
        logging.error(str(e))
        print(f"ERROR: {e}")
        exit(1)

    # -- 4.2: Load macro series --
    print("Step 2/6: Loading macro series (Selic, CDI, TR) ...")
    try:
        df_macro = load_macro_series()
    except FileNotFoundError as e:
        logging.error(str(e))
        print(f"ERROR: {e}")
        exit(1)

    # -- 4.3: Load institution info (segment) from List files --
    print("Step 3/6: Loading institution info (segments) ...")
    df_info = load_institution_info()

    # -- 4.4: Load COSIF quarterly institution-level rates --
    print("Step 4/6: Loading COSIF quarterly institution rates ...")
    df_cosif_q = load_cosif_quarterly()

    # -- 4.5: Load DRE (Demonstração de Resultado) financial data --
    print("Step 5/6: Loading DRE labour-cost data ...")
    df_dre = load_if_data_financial()

    # -- 4.6: Build panel --
    print("Step 6/6: Building panel ...")
    panel = build_panel(
        df_ifdata, df_macro,
        df_cosif_q=df_cosif_q if not df_cosif_q.empty else None,
        df_info=df_info if not df_info.empty else None,
        df_dre=df_dre if not df_dre.empty else None,
    )

    # -- 4.6: Save outputs --
    print("Saving ...")

    # Convert rate columns from decimal to percentage
    RATE_COLS = [
        "risk_free_qoq", "deposit_rate_qoq", "spread_qoq",
        "cosif_implicit_rate", "cosif_prepaid_rate",
        "median_ip_rate",
        "selic_qoq", "cdi_qoq", "savings_rate_qoq",
    ]
    for col in RATE_COLS:
        if col in panel.columns:
            panel[col] = panel[col] * 100

    panel_path = os.path.join(OUTPUT_PATH, "egan_panel_institution.csv")
    panel.to_csv(panel_path, index=False)
    logging.info(f"Panel saved to {panel_path}")

    # Parquet for fast downstream loading (estimation_2.py)
    parquet_path = os.path.join(OUTPUT_PATH, "egan_panel_institution.parquet")
    panel.to_parquet(parquet_path, index=False, engine="pyarrow")
    logging.info(f"Panel saved to {parquet_path}")

    # Quarterly macro rates for reference
    df_macro_q = make_quarterly_macro(df_macro)
    MACRO_RATE_COLS = ["selic_qoq", "cdi_qoq", "savings_rate_qoq"]
    for col in MACRO_RATE_COLS:
        if col in df_macro_q.columns:
            df_macro_q[col] = df_macro_q[col] * 100
    macro_path = os.path.join(OUTPUT_PATH, "quarterly_macro_rates.csv")
    if not os.path.exists(macro_path):
        df_macro_q.to_csv(macro_path, index=False)
        logging.info(f"Quarterly macro rates saved to {macro_path}")

    # -- Summary --
    print(f"\n{'=' * 60}")
    print(f"Panel saved: {panel_path}")
    print(f"  Shape         : {panel.shape[0]:,} rows x {panel.shape[1]} cols")
    print(f"  Institutions  : {panel['CNPJ'].nunique()}")
    print(f"  Quarters      : {panel['quarter'].nunique()}"
          f"  ({panel['quarter'].min()} - {panel['quarter'].max()})")
    print(f"  Deposit types : {panel['deposit_type'].nunique()}")

    cosif_count = panel["cosif_implicit_rate"].notna().sum()
    cosif_pct   = 100 * cosif_count / len(panel) if len(panel) else 0
    print(f"  COSIF coverage: {cosif_count:,} / {len(panel):,}"
          f" ({cosif_pct:.1f}%)")
    print(f"{'=' * 60}")

    print("\n--- Deposit Balance Summary (R$ thousands) ---")
    print(
        panel.groupby("deposit_type_name", observed=True)["deposit_balance"]
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
