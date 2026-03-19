## deposits_panel_build.py
# Authors: Pedro Feijó de Moraes
#
# Last edited on 2026_03_06
#
# Purpose: Build a balanced panel at the conglomerate × municipality × quarter level
#          combining two data sources:
#
#   TIER 1 – ESTBAN (BCB): banking institutions with branch networks.
#            Source: BCB/ESTBAN/ESTBAN.csv (2016–2024, processed by ESTBAN_Process_1.R)
#                    + raw monthly CSVs in BCB/ESTBAN/Relatório por município/ (for 2013–2015).
#            Granularity: institution × municipality × month  →  conglomerate × municipality × quarter.
#            Deposit columns: dep_a1 (V400_401), dep_a2 (V420), dep_a3 (V431), dep_a4 (V432).
#            a5 (prepaid) is NOT in ESTBAN; set to NaN.
#
#   TIER 2 – IF Data (BCB prudential conglomerate reports):
#            Source: BCB/IF Data/Aggregated Data/IF_DATA_type_1_report_3.csv
#            Granularity: conglomerate × quarter (national total only).
#            Only institutions NOT represented in ESTBAN are included here; for them
#            CODMUN_IBGE is set to 0 (a code that matches no real municipality).
#            Deposit columns: dep_a1 (NumeroConta 78282), dep_a2 (78283), dep_a3 (78284),
#                             dep_a4 (78286), dep_outros (78285), dep_a5 (110560).
#
#   OUTPUT: BCB/Panel/deposits_panel.csv
#           Columns: CodConglomeradoPrudencial, CNPJ_Lider, NomeInstituicao,
#                    CODMUN_IBGE, Year, Quarter,
#                    dep_a1, dep_a2, dep_a3, dep_a4, dep_outros, dep_a5, Source
#
# Notes:
#   • CNPJ → conglomerate mapping is built from IF Data List files (quarterly snapshots).
#     Primary key: CnpjInstituicaoLider (CNPJ of the conglomerate's lead institution).
#     Secondary key: numeric CodInst values (where CodInst == institution's own CNPJ).
#   • Quarter = last month of the quarter as reported in ESTBAN (March=Q1, June=Q2,
#     September=Q3, December=Q4).  These are stock (balance-sheet) quantities.
###-------------------------------------------------------------------------------------------


## 1) Load necessary packages and set paths and constants:

# Load necessary packages:
import os
import glob
import re
import unicodedata
import string
import logging
import pandas as pd
import numpy as np

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Set paths:
BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

ESTBAN_PROC_CSV  = os.path.join(BASE, "BCB", "ESTBAN", "ESTBAN.csv")           # processed 2016–2024
ESTBAN_RAW_MUN   = os.path.join(BASE, "BCB", "ESTBAN", "Relatório por município")  # raw monthly CSVs
IF_AGG_DIR       = os.path.join(BASE, "BCB", "IF Data", "Aggregated Data")
IF_LIST_DIR      = os.path.join(BASE, "BCB", "IF Data", "List")
OUTPUT_DIR       = os.path.join(BASE, "BCB", "Egan_et_al_2025_Rep", "processed")

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "deposits_panel_build",
        {
            "estban_proc_csv": ESTBAN_PROC_CSV,
            "estban_raw_mun_dir": ESTBAN_RAW_MUN,
            "if_agg_dir": IF_AGG_DIR,
            "if_list_dir": IF_LIST_DIR,
            "output_dir": OUTPUT_DIR,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    ESTBAN_PROC_CSV = _paths["estban_proc_csv"]
    ESTBAN_RAW_MUN = _paths["estban_raw_mun_dir"]
    IF_AGG_DIR = _paths["if_agg_dir"]
    IF_LIST_DIR = _paths["if_list_dir"]
    OUTPUT_DIR = _paths["output_dir"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global constants and variables:
# IF Data report 3 (Passivo – Captações) NumeroConta values confirmed from data:
ACCT_A1     = 78282   # Depositos a Vista           (demand deposits)
ACCT_A2     = 78283   # Depositos de Poupanca       (savings)
ACCT_A3     = 78284   # Depositos Interfinanceiros  (interbank)
ACCT_A4     = 78286   # Depositos a Prazo           (time deposits)
ACCT_OUTROS = 78285   # Outros Depositos            (other conventional deposits)
ACCT_A5     = 110560  # Conta de Pagamento PrePaga  (prepaid payment accounts, true a5)

# Quarter-end months: use these ESTBAN months to represent each quarter
QUARTER_END_MONTHS = {3: 1, 6: 2, 9: 3, 12: 4}   # month → quarter number

# Sentinel municipality code for institutions with no geographic breakdown
NO_MUN_CODE = 0

## 2) User-defined functions:

# 2.1) String normalization:
def normalize_str(s):
    """Remove accents, punctuation, and non-ASCII characters from a string."""
    if pd.isna(s):
        return s
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = "".join(c for c in s if ord(c) < 128)
    return s.strip().upper()


def coerce_cnpj(series):
    """Convert a series to nullable Int64 CNPJ root (8-digit integer)."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")

# 2.2) CNPJ to conglomerate map
def build_cnpj_conglomerate_map() -> dict:
    """
    Return {cnpj_int: (CodConglomeradoPrudencial, CNPJ_Lider_int, NomeInstituicao)}
    by scanning IF Data List files (all quarters).

    Two matching strategies:
      1. CnpjInstituicaoLider  →  conglomerate  (always available; maps the leader's CNPJ)
      2. numeric CodInst       →  conglomerate  (when an individual institution's CodInst
                                                 equals its own CNPJ, i.e. it is the leader)
    Strategy 1 is applied after strategy 2 so leaders take priority.
    """
    list_files = sorted(glob.glob(os.path.join(IF_LIST_DIR, "IF_DATA_List_*.csv")))
    # Skip headers-only stubs (< 500 bytes = no data rows)
    list_files = [f for f in list_files if os.path.getsize(f) > 500]
    if not list_files:
        raise FileNotFoundError(f"No IF Data List files found in {IF_LIST_DIR}")

    frames = []
    for path in list_files:
        try:
            df = pd.read_csv(path, encoding="latin1", low_memory=False,
                             usecols=["CodInst", "NomeInstituicao",
                                      "CodConglomeradoPrudencial", "CnpjInstituicaoLider"])
            frames.append(df)
        except Exception as e:
            logging.warning(f"Could not read List file {os.path.basename(path)}: {e}")

    lf = pd.concat(frames, ignore_index=True).drop_duplicates()

    # Keep only rows that have a prudential conglomerate code
    lf = lf.dropna(subset=["CodConglomeradoPrudencial"])
    lf["lider_int"] = coerce_cnpj(lf["CnpjInstituicaoLider"])

    mapping: dict = {}

    # Strategy 2: numeric CodInst entries (individual institution CNPJs)
    lf["codinst_int"] = coerce_cnpj(lf["CodInst"])
    strat2 = lf.dropna(subset=["codinst_int"])
    for _, row in strat2.iterrows():
        cnpj = int(row["codinst_int"])
        mapping[cnpj] = (
            row["CodConglomeradoPrudencial"],
            int(row["lider_int"]) if pd.notna(row["lider_int"]) else None,
            normalize_str(row["NomeInstituicao"]),
        )

    # Strategy 1: CnpjInstituicaoLider (overrides – leaders take precedence)
    strat1 = lf.dropna(subset=["lider_int"])
    # Use the most recent name for each leader
    name_map = (strat1.sort_values("NomeInstituicao")
                      .drop_duplicates(subset=["lider_int"], keep="last"))
    for _, row in name_map.iterrows():
        cnpj = int(row["lider_int"])
        mapping[cnpj] = (
            row["CodConglomeradoPrudencial"],
            cnpj,
            normalize_str(row["NomeInstituicao"]),
        )

    logging.info(f"CNPJ→conglomerate map: {len(mapping)} entries")
    return mapping


# 2.3) ESTBAN raw file processing:

# Column rename map for the pre-July-2022 ESTBAN format.
# Keys are substrings that uniquely identify each raw verbete column.
# Values are the standardised V-code names used in the processed ESTBAN.csv.
_PRE2022_RENAME = {
    "VERBETE_110_ENCAIXE":                       "V110",
    "VERBETE_114_APLIC_TEMPORARIAS":              "V114",
    "VERBETE_111_CAIXA":                          "V111",
    "VERBETE_112_DEPOSITOS_BANCARIOS":            "V112",
    "VERBETE_113_BACEN":                          "V113",
    "VERBETE_120_APLIC_INTERFINANC":              "V120",
    "VERBETE_130_TIT_E_VAL_MOB":                  "V130",
    "VERBETE_140_REL_INTERFINANC":                "V140",
    # V141_142 and V144_... are compound columns – handled separately below
    "VERBETE_158_OUTR_REL":                       "V158",
    "VERBETE_160_OPERACOES_DE_CREDITO":           "V160",
    "VERBETE_161_EMPRES":                         "V161",
    "VERBETE_162_FINANCIAMENTOS":                 "V162",
    "VERBETE_163_FIN_RURAIS_AGRICUL_CUST":        "V163",
    "VERBETE_169_FINANCIAMENTOS_IMOBILIARIOS":    "V169",
    "VERBETE_171_OUTRAS_OPERACOES":               "V171",
    "VERBETE_172_OUTROS_CREDITOS":                "V172",
    "VERBETE_174_PROV":                           "V174",
    "VERBETE_176_OPERACOES_ESPECIAIS":            "V176",
    "VERBETE_180_ARRENDAMENTO":                   "V180",
    "VERBETE_184_PROV_P":                         "V184",
    "VERBETE_190_OUTROS_VALORES":                 "V190",
    "VERBETE_200_PERMANENTE":                     "V200",
    "VERBETE_399_TOTAL_DO_ATIVO":                 "V399",
    # Compound demand-deposit column (many verbetes 401–419)
    "VERBETE_401_SERVICOS_PUBLICOS":              "V400_401",
    "VERBETE_420_DEPOSITOS_DE_POUPANCA":          "V420",
    "VERBETE_430_DEPOSITOS_INTERIFNANCEIROS":     "V430",   # old typo in BCB data
    "VERBETE_431_DEPOSITOS_INTERFINANCEIROS":     "V431",
    "VERBETE_432_DEPOSITOS_A_PRAZO":              "V432",
    "VERBETE_433_CAPTACOES":                      "V433",
    "VERBETE_440_REL_INTERFINANC_E_INTERDEPEND":  "V440",
    "VERBETE_460_OBRIG_POR_EMP":                  "V460",
    "VERBETE_470_INST_FINANCEIROS_DERIV":         "V470",
    "VERBETE_480_OBRIGACOES_POR_RECEBIMENTO":     "V480",
    "VERBETE_490_CHEQUES":                        "V490_500",
    "VERBETE_610_PATRIMONIO_LIQUIDO":             "V610",
    "VERBETE_710_CONTAS_DE_RESULTADO":            "V710",
    "VERBETE_711_CONTAS_CREDORAS":                "V711",
    "VERBETE_712_CONTAS_DEVEDORAS":               "V712",
    "VERBETE_899_TOTAL_DO_PASSIVO":               "V899",
}

_POST2022_RENAME = {
    "VERBETE_110_DISPONIBILIDADES":               "V110",
    "VERBETE_111_CAIXA":                          "V111",
    "VERBETE_112_DEPOSITOS_BANCARIOS":            "V112",
    "VERBETE_113_BACEN":                          "V113",
    # V114 must be computed: V110 - V111 - V112 - V113
    "VERBETE_120_APLIC_INTERFINANC":              "V120",
    "VERBETE_130_TIT_E_VAL_MOB":                  "V130",
    "VERBETE_140_REL_INTERFINANC_E_INTERDEPEND":  "V140",
    "VERBETE_158_OUTR_REL":                       "V158",
    "VERBETE_160_OPERACOES_DE_CREDITO":           "V160",
    "VERBETE_161_EMPRES":                         "V161",
    "VERBETE_162_FINANCIAMENTOS":                 "V162",
    "VERBETE_163_FIN_RURAIS_AGRICUL_CUST":        "V163",
    "VERBETE_169_FINANCIAMENTOS_IMOBILIARIOS":    "V169",
    "VERBETE_171_OUTRAS_OPERACOES":               "V171",
    "VERBETE_172_OUTROS_CREDITOS":                "V172",
    "VERBETE_174_PROV":                           "V174",
    "VERBETE_176_OPERACOES_ESPECIAIS":            "V176",
    "VERBETE_180_ARRENDAMENTO":                   "V180",
    "VERBETE_184_PROV_P":                         "V184",
    "VERBETE_190_OUTROS_VALORES":                 "V190",
    "VERBETE_200_PERMANENTE":                     "V200",
    "VERBETE_399_TOTAL_DO_ATIVO":                 "V399",
    "VERBETE_401_SERVICOS_PUBLICOS":              "V400_401",
    "VERBETE_420_DEPOSITOS_DE_POUPANCA":          "V420",
    "VERBETE_430_DEPOSITOS_INTERIFNANCEIROS":     "V430",
    "VERBETE_431_DEPOSITOS_INTERFINANCEIROS":     "V431",
    "VERBETE_432_DEPOSITOS_A_PRAZO":              "V432",
    "VERBETE_433_CAPTACOES":                      "V433",
    "VERBETE_440_REL_INTERFINANC_E_INTERDEPEND":  "V440",
    "VERBETE_460_OBRIG_POR_EMP":                  "V460",
    "VERBETE_470_INST_FINANCEIROS_DERIV":         "V470",
    "VERBETE_480_OBRIGACOES_POR_RECEBIMENTO":     "V480",
    "VERBETE_490_CHEQUES":                        "V490_500",
    "VERBETE_610_PATRIMONIO_LIQUIDO":             "V610",
    "VERBETE_710_CONTAS_DE_RESULTADO":            "V710",
    "VERBETE_711_CONTAS_CREDORAS":                "V711",
    "VERBETE_712_CONTAS_DEVEDORAS":               "V712",
    "VERBETE_899_TOTAL_DO_PASSIVO":               "V899",
}

# Columns to drop (present in some ESTBAN raw files but not needed)
_COLS_TO_DROP_SUBSTRINGS = [
    "VERBETE_143_", "VERBETE_153_", "VERBETE_164_", "VERBETE_165_",
    "VERBETE_166_", "VERBETE_167_", "VERBETE_168_",
    "VERBETE_173_", "VERBETE_443_", "VERBETE_461_",
    "VERBETE_481_", "VERBETE_486_", "VERBETE_300_",
    "VERBETE_457_", "VERBETE_800_",
]

# Threshold date for format change (July 2022)
_FORMAT_CHANGE_DATE = pd.Timestamp("2022-07-01")

def _rename_verbete_cols(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Match raw verbete column names against _PRE2022_RENAME/_POST2022_RENAME
    substrings and rename.  Compound columns (multiple verbetes in one) are
    handled by matching the first verbete's substring that appears in the name.
    """
    col_map = {}
    for raw_col in df.columns:
        raw_upper = raw_col.upper().replace(" ", "_")
        for substr, vcode in rename_map.items():
            if substr.upper() in raw_upper and vcode not in col_map.values():
                col_map[raw_col] = vcode
                break
    return df.rename(columns=col_map)


def process_raw_estban_csv(filepath: str) -> pd.DataFrame | None:
    """
    Read one raw ESTBAN municipality-level CSV file and return a DataFrame
    with standardised V-code columns, CNPJ (int), CODMUN_IBGE (int), YEAR, MONTH.
    Returns None if the file cannot be read or is empty.
    """
    try:
        # BCB ESTBAN files have 2 header rows; sep=';'; decimal=','
        df = pd.read_csv(filepath, sep=";", decimal=",", encoding="latin1",
                         skiprows=2, low_memory=False, on_bad_lines="warn")
    except Exception as e:
        logging.warning(f"Cannot read {os.path.basename(filepath)}: {e}")
        return None

    if df.empty:
        return None

    # Normalise column names: strip spaces, uppercase
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

    # Extract date from #DATA_BASE (YYYYMM format)
    date_col = next((c for c in df.columns if "DATA_BASE" in c), None)
    if date_col is None:
        logging.warning(f"No DATA_BASE column in {os.path.basename(filepath)}")
        return None

    df["DATA_BASE_STR"] = df[date_col].astype(str).str.strip()
    sample_date = df["DATA_BASE_STR"].dropna().iloc[0] if len(df) > 0 else ""
    try:
        file_date = pd.to_datetime(sample_date, format="%Y%m")
    except Exception:
        try:
            file_date = pd.to_datetime(sample_date[:7].replace("-", ""), format="%Y%m")
        except Exception:
            logging.warning(f"Cannot parse date '{sample_date}' in {os.path.basename(filepath)}")
            return None

    df["YEAR"]  = file_date.year
    df["MONTH"] = file_date.month
    df.drop(columns=[date_col, "DATA_BASE_STR"], inplace=True, errors="ignore")

    # Drop columns that should not be in the standardised output
    drop_cols = [c for c in df.columns
                 if any(s.upper() in c.upper() for s in _COLS_TO_DROP_SUBSTRINGS)]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Apply format-appropriate renaming
    rename_map = _PRE2022_RENAME if file_date < _FORMAT_CHANGE_DATE else _POST2022_RENAME
    df = _rename_verbete_cols(df, rename_map)

    # Compute V114 for post-2022 files (derived, not directly reported)
    if file_date >= _FORMAT_CHANGE_DATE and "V110" in df.columns:
        for col in ["V111", "V112", "V113"]:
            if col not in df.columns:
                df[col] = 0
        df["V114"] = df["V110"] - df["V111"] - df["V112"] - df["V113"]

    # Standardise CNPJ (int) and CODMUN_IBGE (int)
    cnpj_col = next((c for c in df.columns if c == "CNPJ"), None)
    if cnpj_col is None:
        logging.warning(f"No CNPJ column in {os.path.basename(filepath)}")
        return None

    df["CNPJ"] = coerce_cnpj(df["CNPJ"])
    if "CODMUN_IBGE" in df.columns:
        df["CODMUN_IBGE"] = pd.to_numeric(df["CODMUN_IBGE"], errors="coerce").astype("Int64")

    # Convert all V-code columns to numeric
    v_cols = [c for c in df.columns if re.match(r"^V\d", c)]
    for col in v_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logging.info(f"Processed raw ESTBAN: {os.path.basename(filepath)} ({len(df)} rows)")
    return df

# 2.4) ESTBAN panel building: 

ESTBAN_DEPOSIT_COLS = ["V400_401", "V420", "V431", "V432"]   # a1, a2, a3, a4
ESTBAN_KEEP_COLS    = ["CNPJ", "NOME_INSTITUICAO", "CODMUN_IBGE", "YEAR", "MONTH"] + ESTBAN_DEPOSIT_COLS

def load_estban_processed() -> pd.DataFrame:
    """Load the pre-processed ESTBAN.csv (2016–2024)."""
    logging.info("Loading ESTBAN.csv …")
    df = pd.read_csv(ESTBAN_PROC_CSV, encoding="latin1", low_memory=False)

    # Keep only quarter-end months
    df = df[df["MONTH"].isin(QUARTER_END_MONTHS)].copy()

    # Ensure deposit columns exist
    for col in ESTBAN_DEPOSIT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Standardise types
    df["CNPJ"] = coerce_cnpj(df["CNPJ"])
    df["CODMUN_IBGE"] = pd.to_numeric(df.get("CODMUN_IBGE"), errors="coerce").astype("Int64")
    for col in ESTBAN_DEPOSIT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    available = [c for c in ESTBAN_KEEP_COLS if c in df.columns]
    return df[available].copy()

def load_estban_raw_pre2016() -> pd.DataFrame:
    """
    Process raw ESTBAN CSVs for any year < 2016 found in the raw folder.
    Returns a DataFrame with the same columns as load_estban_processed().
    """
    pattern_upper = os.path.join(ESTBAN_RAW_MUN, "*.CSV")
    pattern_lower = os.path.join(ESTBAN_RAW_MUN, "*.csv")
    all_files = sorted(glob.glob(pattern_upper) + glob.glob(pattern_lower))

    pre2016_files = []
    for fp in all_files:
        fname = os.path.basename(fp)
        # Expect format: YYYYMM_ESTBAN.CSV
        match = re.match(r"^(\d{4})\d{2}_ESTBAN", fname, re.IGNORECASE)
        if match and int(match.group(1)) < 2016:
            pre2016_files.append(fp)

    if not pre2016_files:
        logging.info("No pre-2016 raw ESTBAN files found. Skipping raw processing.")
        return pd.DataFrame(columns=ESTBAN_KEEP_COLS)

    logging.info(f"Processing {len(pre2016_files)} raw pre-2016 ESTBAN files …")
    frames = []
    for fp in pre2016_files:
        df = process_raw_estban_csv(fp)
        if df is None or df.empty:
            continue
        # Keep only quarter-end months
        df = df[df["MONTH"].isin(QUARTER_END_MONTHS)].copy()
        if df.empty:
            continue
        # Ensure required columns
        for col in ESTBAN_DEPOSIT_COLS:
            if col not in df.columns:
                df[col] = np.nan
        if "CODMUN_IBGE" not in df.columns:
            df["CODMUN_IBGE"] = pd.NA
        if "NOME_INSTITUICAO" not in df.columns:
            df["NOME_INSTITUICAO"] = pd.NA
        available = [c for c in ESTBAN_KEEP_COLS if c in df.columns]
        frames.append(df[available].copy())

    if not frames:
        return pd.DataFrame(columns=ESTBAN_KEEP_COLS)

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df["CNPJ"] = coerce_cnpj(raw_df["CNPJ"])
    raw_df["CODMUN_IBGE"] = pd.to_numeric(raw_df["CODMUN_IBGE"], errors="coerce").astype("Int64")
    logging.info(f"Raw pre-2016 ESTBAN: {len(raw_df)} total rows after quarter filter")
    return raw_df

def build_estban_panel(cnpj_map: dict) -> pd.DataFrame:
    """
    Combine processed (2016–2024) and raw pre-2016 ESTBAN data.
    Maps each institution CNPJ to its prudential conglomerate.
    Aggregates to conglomerate × CODMUN_IBGE × quarter.

    Returns DataFrame with columns:
        CodConglomeradoPrudencial, CNPJ_Lider, NomeInstituicao,
        CODMUN_IBGE, Year, Quarter, dep_a1, dep_a2, dep_a3, dep_a4
    """
    # --- Load data ---
    proc = load_estban_processed()
    raw  = load_estban_raw_pre2016()
    estban = pd.concat([proc, raw], ignore_index=True)

    if estban.empty:
        logging.error("No ESTBAN data loaded. Check paths.")
        return pd.DataFrame()

    logging.info(f"ESTBAN combined: {len(estban)} rows ({estban['YEAR'].min()}–{estban['YEAR'].max()})")

    # --- Map CNPJ → conglomerate ---
    def _lookup(cnpj):
        if pd.isna(cnpj):
            return (None, None, None)
        return cnpj_map.get(int(cnpj), (None, None, None))

    lookup_results = estban["CNPJ"].apply(_lookup)
    estban["CodConglomeradoPrudencial"] = lookup_results.apply(lambda x: x[0])
    estban["CNPJ_Lider"]               = lookup_results.apply(lambda x: x[1])
    estban["NomeInstituicao"]           = lookup_results.apply(lambda x: x[2])

    # For institutions not found in the map, fall back to their own CNPJ and name
    mask_no_cong = estban["CodConglomeradoPrudencial"].isna()
    if mask_no_cong.sum() > 0:
        unmatched_cnpjs = estban.loc[mask_no_cong, "CNPJ"].dropna().unique()
        logging.warning(
            f"{mask_no_cong.sum()} ESTBAN rows could not be mapped to a conglomerate "
            f"({len(unmatched_cnpjs)} unique CNPJs). "
            f"They will be kept with CodConglomeradoPrudencial = 'CNPJ_' + CNPJ."
        )
        estban.loc[mask_no_cong, "CodConglomeradoPrudencial"] = \
            "CNPJ_" + estban.loc[mask_no_cong, "CNPJ"].astype(str)
        estban.loc[mask_no_cong, "CNPJ_Lider"] = estban.loc[mask_no_cong, "CNPJ"]
        if "NOME_INSTITUICAO" in estban.columns:
            fallback_names = estban.loc[mask_no_cong, "NOME_INSTITUICAO"]
            estban.loc[mask_no_cong, "NomeInstituicao"] = fallback_names.where(
                fallback_names.notna(), other="UNKNOWN"
            )

    # --- Assign quarter ---
    estban["Quarter"] = estban["MONTH"].map(QUARTER_END_MONTHS)
    estban = estban.dropna(subset=["Quarter"])
    estban["Quarter"] = estban["Quarter"].astype(int)

    # --- Aggregate: conglomerate × municipality × quarter ---
    # Fill NaN CODMUN_IBGE with 0
    estban["CODMUN_IBGE"] = estban["CODMUN_IBGE"].fillna(0).astype(int)

    agg_cols = {col: "sum" for col in ESTBAN_DEPOSIT_COLS if col in estban.columns}
    group_cols = ["CodConglomeradoPrudencial", "CNPJ_Lider", "NomeInstituicao","CODMUN_IBGE", "YEAR", "Quarter"]

    panel = (
        estban.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()
    )

    # Rename deposit columns to standardised names
    panel.rename(columns={
        "V400_401": "dep_a1",
        "V420":     "dep_a2",
        "V431":     "dep_a3",
        "V432":     "dep_a4",
        "YEAR":     "Year",
    }, inplace=True)

    # a5 / outros not available from ESTBAN
    panel["dep_outros"] = np.nan
    panel["dep_a5"]     = np.nan
    panel["Source"]     = "ESTBAN"

    logging.info(f"ESTBAN panel: {len(panel)} conglomerate × municipality × quarter rows")
    return panel


# 2.5) IF DATA panel building:

def build_ifdata_panel() -> pd.DataFrame:
    """
    Load IF Data type-1 report-3 (Passivo – Captações).
    Extract deposit accounts (a1–a5 + outros) for all prudential conglomerates.
    """
    report_path = os.path.join(IF_AGG_DIR, "IF_DATA_type_1_report_3.csv")
    if not os.path.exists(report_path):
        logging.error(f"IF Data report 3 not found: {report_path}")
        return pd.DataFrame()

    logging.info("Loading IF Data type-1 report-3 …")
    df = pd.read_csv(report_path, encoding="latin1", low_memory=False)
    df["NumeroConta"] = pd.to_numeric(df["NumeroConta"], errors="coerce").astype("Int64")
    df["Value"]       = pd.to_numeric(df["Value"],       errors="coerce")

    # Keep only the deposit accounts we care about
    deposit_accounts = [ACCT_A1, ACCT_A2, ACCT_A3, ACCT_A4, ACCT_OUTROS, ACCT_A5]
    df = df[df["NumeroConta"].isin(deposit_accounts)].copy()

    # Map NumeroConta → deposit column name
    acct_col_map = {
        ACCT_A1:     "dep_a1",
        ACCT_A2:     "dep_a2",
        ACCT_A3:     "dep_a3",
        ACCT_A4:     "dep_a4",
        ACCT_OUTROS: "dep_outros",
        ACCT_A5:     "dep_a5",
    }
    df["dep_col"] = df["NumeroConta"].map(acct_col_map)

    # Assign quarter (IF Data is already quarterly: months 3, 6, 9, 12)
    df["Quarter"] = df["Month"].map(QUARTER_END_MONTHS)
    df = df.dropna(subset=["Quarter", "CodConglomeradoPrudencial"])
    df["Quarter"] = df["Quarter"].astype(int)

    # Pivot to wide format
    group_cols = ["CodConglomeradoPrudencial", "CNPJ_Lider", "NomeInstituicao","Year", "Quarter"]
    for col in group_cols:
        if col not in df.columns:
            df[col] = pd.NA

    pivot = (
        df.groupby(group_cols + ["dep_col"], dropna=False)["Value"].sum().unstack("dep_col").reset_index()
    )

    # Ensure all deposit columns exist
    for dep_col in ["dep_a1", "dep_a2", "dep_a3", "dep_a4", "dep_outros", "dep_a5"]:
        if dep_col not in pivot.columns:
            pivot[dep_col] = np.nan

    # Standardise CNPJ_Lider
    pivot["CNPJ_Lider"] = coerce_cnpj(pivot["CNPJ_Lider"])

    # Normalise NomeInstituicao
    pivot["NomeInstituicao"] = pivot["NomeInstituicao"].apply(normalize_str)

    pivot["CODMUN_IBGE"] = NO_MUN_CODE
    pivot["Source"]      = "IFDATA"

    logging.info(
        f"IF Data-only panel: {len(pivot)} rows "
        f"({pivot['CodConglomeradoPrudencial'].nunique()} conglomerates)"
    )
    return pivot

## 3) Main execution:

def main():
    # Step 1: Build CNPJ → conglomerate mapping
    cnpj_map = build_cnpj_conglomerate_map()

    # Step 2: Build ESTBAN panel
    estban_panel = build_estban_panel(cnpj_map)

    # Step 3: Identify conglomerates in the ESTBAN panel
    estban_conglomerates = set(estban_panel["CodConglomeradoPrudencial"].dropna().unique())
    logging.info(f"ESTBAN panel covers {len(estban_conglomerates)} distinct conglomerates")

    # Step 4: Build IF Data full panel
    ifdata_full = build_ifdata_panel()

    # Step 4b: Allocate IF-Data dep_a5 to ESTBAN banks
    # We distribute the national dep_a5 (prepaid) total for ESTBAN banks
    # proportionally across municipalities based on their dep_a1 footprint.
    logging.info("Allocating national dep_a5 to ESTBAN banks using spatial dep_a1 weights...")
    
    # 1. Get national a5
    if_a5 = ifdata_full[["CodConglomeradoPrudencial", "Year", "Quarter", "dep_a5"]].copy()
    if_a5.rename(columns={"dep_a5": "national_a5"}, inplace=True)
    if_a5 = if_a5.dropna(subset=["national_a5"])
    
    # 2. Merge into ESTBAN
    estban_panel = estban_panel.merge(
        if_a5, on=["CodConglomeradoPrudencial", "Year", "Quarter"], how="left"
    )
    
    # 3. Get total ESTBAN a1 per conglomerate-quarter
    estban_a1_totals = estban_panel.groupby(["CodConglomeradoPrudencial", "Year", "Quarter"], as_index=False)["dep_a1"].sum()
    estban_a1_totals.rename(columns={"dep_a1": "national_estban_a1"}, inplace=True)
    
    estban_panel = estban_panel.merge(
        estban_a1_totals, on=["CodConglomeradoPrudencial", "Year", "Quarter"], how="left"
    )
    
    # 4. Calculate local weights and distribute a5
    # If a1 total > 0, weight = a1 / total_a1
    # If a1 total == 0 but we have a5, distribute uniformly (though unlikely for banks)
    estban_panel["a1_weight"] = np.where(
        estban_panel["national_estban_a1"] > 0,
        estban_panel["dep_a1"] / estban_panel["national_estban_a1"],
        0  # Or equal weight: 1.0 / num_municipalities_for_that_bank... keep 0 for simplicity/safety
    )
    
    # Apply weight. Only override dep_a5 if we actually have national_a5 to allocate.
    estban_panel["dep_a5"] = np.where(
        estban_panel["national_a5"].notna(),
        estban_panel["national_a5"] * estban_panel["a1_weight"],
        np.nan
    )
    
    # Clean up working columns
    estban_panel.drop(columns=["national_a5", "national_estban_a1", "a1_weight"], inplace=True)

    # Step 4c: Filter IF Data panel to strictly Non-ESTBAN conglomerates (Tier-2)
    ifdata_panel = ifdata_full[~ifdata_full["CodConglomeradoPrudencial"].isin(estban_conglomerates)].copy()
    
    # Step 5: Stack both tiers
    final_col_order = [
        "CodConglomeradoPrudencial", "CNPJ_Lider", "NomeInstituicao",
        "CODMUN_IBGE", "Year", "Quarter",
        "dep_a1", "dep_a2", "dep_a3", "dep_a4", "dep_outros", "dep_a5",
        "Source",
    ]

    frames = []
    for frame in [estban_panel, ifdata_panel]:
        if frame is not None and not frame.empty:
            for col in final_col_order:
                if col not in frame.columns:
                    frame[col] = np.nan
            frames.append(frame[final_col_order])

    if not frames:
        logging.error("No data to save.")
        return

    panel = pd.concat(frames, ignore_index=True)

    # Sort
    panel.sort_values(
        ["CodConglomeradoPrudencial", "CODMUN_IBGE", "Year", "Quarter"],
        inplace=True, na_position="last"
    )
    panel.reset_index(drop=True, inplace=True)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "deposits_panel.csv")
    panel.to_csv(out_path, index=False, encoding="utf-8")
    logging.info(f"Saved panel to {out_path}  ({len(panel):,} rows)")

    # Summary
    n_cong  = panel["CodConglomeradoPrudencial"].nunique()
    n_mun   = panel[panel["Source"] == "ESTBAN"]["CODMUN_IBGE"].nunique()
    n_est   = (panel["Source"] == "ESTBAN").sum()
    n_ifd   = (panel["Source"] == "IFDATA").sum()
    yr_min  = panel["Year"].min()
    yr_max  = panel["Year"].max()
    print(
        f"\nPanel summary\n"
        f"  Rows total:          {len(panel):,}\n"
        f"    ESTBAN rows:       {n_est:,}  ({n_mun} distinct municipalities)\n"
        f"    IF Data-only rows: {n_ifd:,}  (CODMUN_IBGE = {NO_MUN_CODE})\n"
        f"  Conglomerates:       {n_cong}\n"
        f"  Year range:          {yr_min} – {yr_max}\n"
        f"  Output:              {out_path}"
    )


if __name__ == "__main__":
    main()
