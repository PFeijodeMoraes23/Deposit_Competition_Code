"""utils/estban.py — reader for one raw ESTBAN municipality-level CSV (monthly BCB file).

process_raw_estban_csv(path) returns standardised V-code columns (pre- and post-July-2022
layouts), CNPJ (Int64), CODMUN_IBGE (Int64), YEAR, MONTH. Moved verbatim from panel_deposits.py
(2026-10-15). Readers: panel_deposits (pre-2016 balances), scrape_estban_concat (ESTBAN.csv).
"""
from __future__ import annotations

import logging
import os
import re

import pandas as pd

from utils.conglomerates import coerce_cnpj


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
