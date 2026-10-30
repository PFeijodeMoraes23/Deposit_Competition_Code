"""utils/estban.py — reader for one raw ESTBAN municipality-level CSV (monthly BCB file), and the
treatment of cells that did not file.

process_raw_estban_csv(path) returns standardised V-code columns (pre- and post-July-2022
layouts), CNPJ (Int64), CODMUN_IBGE (Int64), YEAR, MONTH. Moved verbatim from panel_deposits.py
(2026-10-15). Readers: panel_deposits (pre-2016 balances), scrape_estban_concat (ESTBAN.csv).

fill_unfiled_quarter_ends(monthly, ...) classifies and fills the quarter-end cells whose branch did
not file (AGEN_ESPERADAS > 0, AGEN_PROCESSADAS = 0, every balance blank); UNFILED_CELLS_CSV is the
record panel_deposits writes and panel_demand_chars reads back to compare the panel with the raw
files.
"""
from __future__ import annotations

import logging
import os
import re

import numpy as np
import pandas as pd

from utils import paths
from utils.conglomerates import coerce_cnpj

# One row per unfiled quarter-end cell, written by panel_deposits.build_estban_panel.
UNFILED_CELLS_CSV = paths.PROCESSED / "PANEL_INTERMED" / "estban_unfiled_cells.csv"


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


# ── Cells that did not file ─────────────────────────────────────────────────────────────────────
# ESTBAN lists every branch cell (institution x municipality) the BCB expects each month. A cell
# whose branches were expected but not processed (AGEN_ESPERADAS > 0, AGEN_PROCESSADAS = 0) comes
# with every balance blank: the bank did not file, it did not hold zero. Summed as zero, the cell
# vanished for the quarter -- C6 in 2024-03 (R$24.9 bn), BTG Pactual in 2019-12 (R$8.5 bn) -- and the
# panel read an exit and a re-entry. Each unfiled quarter-end cell is now one of:
#   gap      it filed in an earlier month of the same quarter and files again later: it takes the
#            balances of the nearest filed month of the quarter (one month back, then two). ESTBAN is
#            monthly, so these are the bank's own reported balances, not an interpolation.
#   closure  it never files again in any month the files hold: the branch closed. An exit; no row.
#   nodonor  it files again later, but no month of the quarter was filed: the quarter stays missing
#            (no row) and is logged.
# Later months, 2025-2026 included, only decide whether a cell files again; no balance ever comes
# from a later month.
UNFILED_KEY = ["CNPJ", "CODMUN"]


def unfiled_mask(df: pd.DataFrame, dep_cols: list) -> pd.Series:
    """Cells whose expected branches were not processed and whose balances are all blank."""
    exp = pd.to_numeric(df["AGEN_ESPERADAS"], errors="coerce").fillna(0)
    pro = pd.to_numeric(df["AGEN_PROCESSADAS"], errors="coerce").fillna(0)
    return (exp > 0) & (pro == 0) & df[dep_cols].isna().all(axis=1)


def fill_unfiled_quarter_ends(monthly: pd.DataFrame, dep_cols: list,
                              quarter_end_months) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify every unfiled quarter-end cell of `monthly` (all months, the columns of UNFILED_KEY,
    CODMUN_IBGE, YEAR, MONTH, the two AGEN columns and `dep_cols`) and fill the gaps.

    Returns (frame, cells). `frame` is `monthly` with each gap cell's balances (and, where its own is
    blank, the donor's CODMUN_IBGE) filled, and the closure and nodonor quarter-end cells removed.
    `cells` has one row per unfiled quarter-end cell: CNPJ, CODMUN, CODMUN_IBGE, NOME_INSTITUICAO,
    YEAR, MONTH, kind, donor (YYYYMM, 0 if none), next_filed (YYYYMM, 0 if never), whole_bank (every
    cell of the CNPJ that month unfiled) and the balances the cell received."""
    df = monthly.copy()
    df["_t"] = df["YEAR"].astype(int) * 12 + df["MONTH"].astype(int) - 1
    unf = unfiled_mask(df, dep_cols)
    qe = df["MONTH"].isin(list(quarter_end_months))

    filed = df.loc[~unf, UNFILED_KEY + ["_t", "CODMUN_IBGE"] + dep_cols]
    donors = filed.groupby(UNFILED_KEY + ["_t"]).agg(
        {**{c: "sum" for c in dep_cols}, "CODMUN_IBGE": "first"})

    tg = df.loc[unf & qe, UNFILED_KEY + ["_t"]].copy()
    tg["donor"] = 0
    for lag in (1, 2):
        open_ = tg["donor"] == 0
        idx = pd.MultiIndex.from_arrays([tg.loc[open_, c] for c in UNFILED_KEY] + [tg.loc[open_, "_t"] - lag])
        hit = donors.reindex(idx)
        ok = hit["CODMUN_IBGE"].notna().to_numpy() | hit[dep_cols].notna().any(axis=1).to_numpy()
        rows = tg.index[open_][ok]
        for c in dep_cols:
            tg.loc[rows, c] = hit[c].to_numpy()[ok]
        tg.loc[rows, "CODMUN_IBGE_donor"] = hit["CODMUN_IBGE"].to_numpy()[ok]
        t_d = tg.loc[rows, "_t"] - lag
        tg.loc[rows, "donor"] = (t_d // 12) * 100 + t_d % 12 + 1

    later = filed[UNFILED_KEY + ["_t"]].drop_duplicates()
    nxt = (tg[UNFILED_KEY + ["_t"]].reset_index()
           .merge(later, on=UNFILED_KEY, how="left", suffixes=("", "_f")))
    nxt = nxt[nxt["_t_f"] > nxt["_t"]].groupby("index")["_t_f"].min()
    tg["next_filed"] = 0
    tg.loc[nxt.index, "next_filed"] = ((nxt // 12) * 100 + nxt % 12 + 1).astype(int)
    tg["kind"] = np.where(tg["next_filed"] == 0, "closure", np.where(tg["donor"] > 0, "gap", "nodonor"))

    gap = tg.index[tg["kind"] == "gap"]
    df.loc[gap, dep_cols] = tg.loc[gap, dep_cols].to_numpy(dtype=float)
    own = df.loc[gap, "CODMUN_IBGE"]
    df.loc[gap, "CODMUN_IBGE"] = own.where(own.notna(), tg.loc[gap, "CODMUN_IBGE_donor"]).astype("Int64")

    n_cells = df.loc[qe].groupby(["CNPJ", "_t"]).size()
    n_unf = tg.groupby(["CNPJ", "_t"]).size()
    whole = (n_unf == n_cells.reindex(n_unf.index)).rename("whole_bank")
    cells = df.loc[tg.index, UNFILED_KEY + ["CODMUN_IBGE", "NOME_INSTITUICAO", "YEAR", "MONTH", "_t"]].join(
        tg[["kind", "donor", "next_filed"] + dep_cols])
    cells = cells.join(whole, on=["CNPJ", "_t"]).drop(columns="_t")
    cells.loc[cells["kind"] != "gap", dep_cols] = np.nan

    drop = tg.index[tg["kind"] != "gap"]
    out = df.drop(index=drop).drop(columns="_t")
    return out, cells.sort_values(["YEAR", "MONTH", "CNPJ", "CODMUN"]).reset_index(drop=True)


def load_unfiled_gaps(year: int, month: int) -> pd.DataFrame:
    """The gap cells panel_deposits filled at one month (UNFILED_CELLS_CSV), for comparisons of the
    panel with a raw file: the raw file has those balances blank, the panel holds the donor's."""
    if not UNFILED_CELLS_CSV.exists():
        raise FileNotFoundError(f"{UNFILED_CELLS_CSV} does not exist: run panel_deposits.py first")
    c = pd.read_csv(UNFILED_CELLS_CSV)
    return c[(c["kind"] == "gap") & (c["YEAR"] == year) & (c["MONTH"] == month)].copy()
