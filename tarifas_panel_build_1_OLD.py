## tarifas_panel_build_1.py
# Author: Pedro Feijo de Moraes
# Last edited: 2026-02-26
#
# Objective: Enrich the existing Egan-style panels with consumer tariff data
#            from the BCB Tarifas Bancárias API (scraped by tarifas_scrape_1.py).
#
# Outputs two new panels:
#
#   egan_panel_institution_tariffs.csv  -- CNPJ x quarter x deposit_type
#       All columns from egan_panel_institution.csv PLUS, for each of the 50
#       regulated service codes, one column tarif_{code} with the maximum fee
#       (R$) the institution was allowed to charge, as of the most recent
#       scrape where the tariff had already taken effect.
#
#   egan_panel_deposits_tariffs.csv     -- CodConglPrud x quarter x deposit_type
#       All columns from egan_panel_deposits.csv PLUS tariff columns aggregated
#       to the conglomerate level (median across member institutions).
#
# -- Deposit-type annotation of service codes --------------------------------
# BCB Resolution 3.919/2010 defines tariff components by service category.
# The 50 service codes (all Pessoa Física, i.e. retail/consumer) fall into:
#
#  Group            Codes       Deposit types
#  -------------------------------------------------------------------------
#  Cadastro         1101        All (registration fee to open any account)
#  Cartão débito    1201        Type 1 - demand (checking)
#  Cartão poupança  1202        Type 2 - savings
#  Cheque           1203-1208   Type 1 - demand (cheque is a demand-dep. svc)
#  Saque            1209-1211   Types 1 & 2 (both checking and savings)
#  Depósito         1212        Type 1 - demand
#  Extrato          1213-1219   Types 1 & 2
#  Transferências   1304-1315   Type 1 - demand (DOC/TED / between-account)
#  Overdraft        1401        Type 1 - demand (adiantamento a depositante)
#  Pacotes          1501-1504   Type 1 - demand (standardised service bundles)
#  Cartão crédito   1601-1607   No direct deposit-type mapping (credit product)
#  Câmbio           1701-1707   No direct deposit-type mapping (FX service)
#
# -- Time dimension of tariffs -----------------------------------------------
# The BCB API returns the CURRENT tariff plus DataVigencia (effective date).
# We only populate tarif_{code} for panel quarters >= data_vigencia_quarter,
# leaving earlier quarters as NaN (historical tariff levels are unknown until
# additional scrape runs accumulate; see tarifas_scrape_1.py).
#
# If multiple scrape runs are available, we pick the most recent scrape whose
# data_vigencia <= panel quarter end-date, i.e. the tariff in force that quarter.
#
# -- Conglomerate aggregation -------------------------------------------------
# For each (CodConglPrud, service_code, quarter): we take the MEDIAN valor_maximo
# across all member institutions that reported a tariff.  The conglomerate code
# used in COL CodConglPrud is derived by stripping the 'C' prefix from
# CodConglomeradoPrudencial in the IF-Data List and converting to integer.
# Standalones (no C prefix) are matched directly by numeric CNPJ.
#
# Prerequisites:
#   python tarifas_scrape_1.py          -> BCB/Tarifas/processed/tarifa...long.csv
#   egan_panel_build_2.py               -> egan_panel_institution.csv
#   egan_panel_build.py                 -> egan_panel_deposits.csv
## ---------------------------------------------------------------------------

import logging
import os
import glob
from pathlib import Path

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler

# -- Paths --------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
PARENT_DIR  = SCRIPT_DIR.parents[1]
BCB_PATH    = PARENT_DIR / "BCB"

EGAN_PATH       = BCB_PATH / "Egan_et_al_2025_Rep"
PANEL_INTERMED  = EGAN_PATH / "processed" / "PANEL_INTERMED"

IF_DATA_LIST_DIR = BCB_PATH / "IF Data" / "List"
TARIF_PROC       = BCB_PATH / "Tarifas" / "processed"

# Input panels
INST_PANEL_IN   = PANEL_INTERMED / "egan_panel_institution.csv"
CONG_PANEL_IN   = PANEL_INTERMED / "egan_panel_deposits.csv"

# Tariff source
TARIF_LONG_IN   = TARIF_PROC / "tarifas_panel_institution_long.csv"

# Output panels
INST_PANEL_OUT  = PANEL_INTERMED / "egan_panel_institution_tariffs.csv"
CONG_PANEL_OUT  = PANEL_INTERMED / "egan_panel_deposits_tariffs.csv"

# SCR segment credit-structure instruments (built by scr_panel_build.py)
SCR_SEGMENT_IN  = PANEL_INTERMED / "scr_segment_panel.parquet"

PANEL_INTERMED.mkdir(parents=True, exist_ok=True)

# -- Logging ------------------------------------------------------------------
_log_file = SCRIPT_DIR / "tarifas_panel_build_1.log"
_handler  = RotatingFileHandler(
    str(_log_file), maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
)
logging.basicConfig(
    handlers=[_handler],
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# -- Service-code metadata -----------------------------------------------------
# Maps each BCB service code to (economic category, deposit types it relates to,
# short English description).
# dep_types: 1=demand, 2=savings, 3=interbank, 4=time/CDB, 5=prepaid
# Empty list means "institution-level, not tied to a particular deposit type".
SERVICE_CATALOG: dict[int, dict] = {
    1101: {"cat": "cadastro",        "dep_types": [1, 2, 3, 4, 5], "en": "Account opening / registration fee"},
    1201: {"cat": "card_debit",      "dep_types": [1],             "en": "Replacement debit card (checking)"},
    1202: {"cat": "card_savings",    "dep_types": [2],             "en": "Replacement debit card (savings)"},
    1203: {"cat": "cheque",          "dep_types": [1],             "en": "Cheque - CCF exclusion registry"},
    1204: {"cat": "cheque",          "dep_types": [1],             "en": "Cheque - stop payment"},
    1205: {"cat": "cheque",          "dep_types": [1],             "en": "Cheque - chequebook delivery"},
    1206: {"cat": "cheque",          "dep_types": [1],             "en": "Cheque - bank draft (cheque administrativo)"},
    1208: {"cat": "cheque",          "dep_types": [1],             "en": "Cheque - certified cheque (cheque visado)"},
    1209: {"cat": "withdrawal",      "dep_types": [1, 2],          "en": "Cash withdrawal - teller"},
    1210: {"cat": "withdrawal",      "dep_types": [1, 2],          "en": "Cash withdrawal - ATM"},
    1211: {"cat": "withdrawal",      "dep_types": [1, 2],          "en": "Cash withdrawal - bank correspondent"},
    1212: {"cat": "deposit",         "dep_types": [1],             "en": "Identified deposit"},
    1213: {"cat": "statement",       "dep_types": [1, 2],          "en": "Account statement (period) - teller"},
    1214: {"cat": "statement",       "dep_types": [1, 2],          "en": "Account statement (period) - electronic"},
    1215: {"cat": "statement",       "dep_types": [1, 2],          "en": "Account statement (period) - digital"},
    1216: {"cat": "statement",       "dep_types": [1, 2],          "en": "Monthly statement - teller"},
    1217: {"cat": "statement",       "dep_types": [1, 2],          "en": "Monthly statement - electronic"},
    1218: {"cat": "statement",       "dep_types": [1, 2],          "en": "Monthly statement - digital"},
    1219: {"cat": "statement",       "dep_types": [1, 2],          "en": "Microfilm / microfiche copy"},
    1304: {"cat": "transfer",        "dep_types": [1],             "en": "Scheduled DOC/TED - teller"},
    1305: {"cat": "transfer",        "dep_types": [1],             "en": "Scheduled DOC/TED - electronic"},
    1306: {"cat": "transfer",        "dep_types": [1],             "en": "Scheduled DOC/TED - internet"},
    1307: {"cat": "transfer",        "dep_types": [1],             "en": "Own-bank transfer - teller"},
    1308: {"cat": "transfer",        "dep_types": [1],             "en": "Own-bank transfer - electronic / internet"},
    1309: {"cat": "transfer",        "dep_types": [1],             "en": "Payment order (ordem de pagamento)"},
    1310: {"cat": "transfer",        "dep_types": [1],             "en": "DOC - teller"},
    1311: {"cat": "transfer",        "dep_types": [1],             "en": "DOC - electronic"},
    1312: {"cat": "transfer",        "dep_types": [1],             "en": "DOC - internet"},
    1313: {"cat": "transfer",        "dep_types": [1],             "en": "TED - teller"},
    1314: {"cat": "transfer",        "dep_types": [1],             "en": "TED - electronic"},
    1315: {"cat": "transfer",        "dep_types": [1],             "en": "TED - internet"},
    1401: {"cat": "overdraft",       "dep_types": [1],             "en": "Overdraft (adiantamento a depositante)"},
    1501: {"cat": "package",         "dep_types": [1],             "en": "Standardised service bundle I (basic)"},
    1502: {"cat": "package",         "dep_types": [1],             "en": "Standardised service bundle II"},
    1503: {"cat": "package",         "dep_types": [1],             "en": "Standardised service bundle III"},
    1504: {"cat": "package",         "dep_types": [1],             "en": "Standardised service bundle IV (full)"},
    1601: {"cat": "credit_card",     "dep_types": [],              "en": "Credit card annual fee - domestic basic"},
    1602: {"cat": "credit_card",     "dep_types": [],              "en": "Replacement credit card"},
    1603: {"cat": "credit_card",     "dep_types": [],              "en": "Cash advance - domestic (credit card)"},
    1604: {"cat": "credit_card",     "dep_types": [],              "en": "Bill payment via credit card (cash)"},
    1605: {"cat": "credit_card",     "dep_types": [],              "en": "Emergency credit evaluation"},
    1606: {"cat": "credit_card",     "dep_types": [],              "en": "Credit card annual fee - international basic"},
    1607: {"cat": "credit_card",     "dep_types": [],              "en": "Cash advance - abroad (credit card)"},
    1701: {"cat": "fx",              "dep_types": [],              "en": "FX sale - cash (foreign currency)"},
    1702: {"cat": "fx",              "dep_types": [],              "en": "FX sale - traveller's cheque"},
    1703: {"cat": "fx",              "dep_types": [],              "en": "FX sale - prepaid card issuance / load"},
    1704: {"cat": "fx",              "dep_types": [],              "en": "FX sale - prepaid card reload"},
    1705: {"cat": "fx",              "dep_types": [],              "en": "FX purchase - cash"},
    1706: {"cat": "fx",              "dep_types": [],              "en": "FX purchase - traveller's cheque"},
    1707: {"cat": "fx",              "dep_types": [],              "en": "FX purchase - prepaid card"},
}

# ================================================================================
# STEP 1 -- Load and prepare the tariff long panel
# ================================================================================

def load_tariff_long() -> pd.DataFrame:
    """
    Load the institution-level tariff long panel from tarifas_scrape_1.py.
    Normalise CNPJ to 8-digit zero-padded string.
    Only keeps Pessoa Física (customer_type == 'F') rows.
    """
    if not TARIF_LONG_IN.exists():
        raise FileNotFoundError(
            f"Tariff panel not found: {TARIF_LONG_IN}\n"
            "Run Code/tarifas_scrape_1.py first."
        )
    df = pd.read_csv(str(TARIF_LONG_IN), low_memory=False, dtype={"cnpj": str})
    df["cnpj"] = df["cnpj"].str.strip().str.zfill(8)
    df = df[df["customer_type"] == "F"].copy()
    df["data_vigencia"] = pd.to_datetime(df["data_vigencia"], errors="coerce")
    df["data_coleta"]   = pd.to_datetime(df["data_coleta"],   errors="coerce")
    logging.info(
        "Tariff long panel loaded: %d rows | %d institutions | %d services | "
        "%d scrape dates",
        len(df),
        df["cnpj"].nunique(),
        df["codigo_servico"].nunique(),
        df["data_coleta"].nunique(),
    )
    return df


def build_tariff_snapshot(df_tarif: pd.DataFrame) -> pd.DataFrame:
    """
    Build a wide (cnpj x service) snapshot table with the most recently
    effective tariff for each institution-service pair.

    Columns:
        cnpj             - 8-digit zero-padded CNPJ
        tarif_{code}     - valor_maximo (R$) for service {code}
        vigencia_{code}  - data_vigencia of the current tariff for {code}

    For the time-conditional join, we also store data_vigencia so that
    callers can set tarif_{code} = NaN for panel quarters before it was effective.
    """
    # For each (cnpj, service_code): keep the row with the latest data_vigencia
    # (= the tariff currently in force, regardless of when we scraped it)
    snap = (
        df_tarif
        .sort_values("data_vigencia")
        .groupby(["cnpj", "codigo_servico"])
        .last()          # most recent vigência
        .reset_index()
        [["cnpj", "codigo_servico", "valor_maximo", "data_vigencia"]]
    )

    # Compute vigência quarter  (e.g. 2025-06-01 -> "2025Q2")
    snap["vigencia_quarter"] = snap["data_vigencia"].apply(
        lambda d: f"{d.year}Q{(d.month - 1) // 3 + 1}" if pd.notna(d) else None
    )

    # Pivot valores
    val_wide = snap.pivot(
        index="cnpj",
        columns="codigo_servico",
        values="valor_maximo",
    )
    val_wide.columns = [f"tarif_{c}" for c in val_wide.columns]
    val_wide = val_wide.reset_index()

    # Pivot vigências
    vig_wide = snap.pivot(
        index="cnpj",
        columns="codigo_servico",
        values="vigencia_quarter",
    )
    vig_wide.columns = [f"vigencia_{c}" for c in vig_wide.columns]
    vig_wide = vig_wide.reset_index()

    wide = val_wide.merge(vig_wide, on="cnpj", how="outer")
    logging.info(
        "Tariff snapshot: %d institutions x (%d value + %d vigência) columns",
        len(wide),
        len(val_wide.columns) - 1,
        len(vig_wide.columns) - 1,
    )
    return wide


def apply_vigencia_mask(
    panel: pd.DataFrame,
    wide: pd.DataFrame,
    entity_col: str,
) -> pd.DataFrame:
    """
    Merge wide (entity x tarif_* columns) onto panel and zero-out
    tarif_{code} values for panel quarters BEFORE the tariff took effect.

    Parameters
    ----------
    panel      : panel DataFrame (must have `entity_col` and `quarter` columns)
    wide       : output of build_tariff_snapshot(), keyed by `cnpj`
    entity_col : column in `panel` that corresponds to wide['cnpj']
                 (either 'CNPJ' for institution panel or a derived CNPJ column)

    Returns
    -------
    panel with tarif_* columns added and masked.
    """
    tarif_cols   = [c for c in wide.columns if c.startswith("tarif_")]
    vigencia_cols = [c for c in wide.columns if c.startswith("vigencia_")]

    # Merge tariff snapshot onto panel
    panel = panel.merge(
        wide,
        left_on=entity_col,
        right_on="cnpj",
        how="left",
    )
    # Drop the extra 'cnpj' key col from wide if it differs from entity_col
    if entity_col != "cnpj" and "cnpj" in panel.columns:
        panel = panel.drop(columns=["cnpj"])

    # For each service code: mask tarif_{code} = NaN where panel quarter < vigencia
    # Quarter comparison: convert both to YYYYQQ int for sorting (e.g. 2025Q2 -> 20252)
    def qtr_to_int(q: str) -> int:
        if not isinstance(q, str) or "Q" not in q:
            return 0
        y, n = q.split("Q")
        return int(y) * 10 + int(n)

    panel_qtr_int = panel["quarter"].apply(qtr_to_int)

    for tarif_col in tarif_cols:
        code         = tarif_col.split("_", 1)[1]  # e.g. "1101"
        vigencia_col = f"vigencia_{code}"
        if vigencia_col in panel.columns:
            vig_int  = panel[vigencia_col].apply(qtr_to_int)
            too_early = panel_qtr_int < vig_int          # panel quarter precedes vigência
            panel.loc[too_early, tarif_col] = np.nan

    # Drop vigência columns (metadata, keep separate if needed)
    panel = panel.drop(columns=vigencia_cols, errors="ignore")

    n_filled = panel[tarif_cols].notna().any(axis=1).sum()
    logging.info(
        "Tariff mask applied on %-15s: %d / %d rows have >=1 tariff value",
        entity_col, n_filled, len(panel),
    )
    return panel


# ================================================================================
# STEP 1b -- Merge SCR segment credit-structure instruments
# ================================================================================
# Produced by scr_panel_build.py (pipeline stage 2b).  Joins on segment × AnoMes.
# Columns added (all lagged 1 quarter by scr_panel_build.py):
#   scr_short_term_share_lag   – share of credit < 1 year
#   scr_medium_term_share_lag  – share of credit 1–5 years
#   scr_long_term_share_lag    – share of credit > 5 years
#   scr_npl_rate_lag           – overdue > 15 days / active portfolio
#   scr_fixed_rate_share_lag   – prefixed-rate credit / active portfolio
#   scr_problematic_share_lag  – problematic assets / active portfolio

def merge_scr_instruments(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Merge SCR segment-level credit structure instruments into the institution
    panel.  Silently skips if the SCR segment panel does not exist yet.

    Join key: segment × AnoMes
    Source:   PANEL_INTERMED/scr_segment_panel.parquet (or .csv fallback)
    """
    scr_csv = PANEL_INTERMED / "scr_segment_panel.csv"
    if SCR_SEGMENT_IN.exists():
        df_scr = pd.read_parquet(str(SCR_SEGMENT_IN))
    elif scr_csv.exists():
        df_scr = pd.read_csv(str(scr_csv), low_memory=False)
    else:
        logging.warning(
            "SCR segment panel not found (%s) — skipping SCR instrument merge. "
            "Run scr_panel_build.py first.",
            SCR_SEGMENT_IN.name,
        )
        return panel

    # Lagged SCR columns to merge (produced by scr_panel_build.add_lagged_columns)
    scr_lag_cols = [
        c for c in df_scr.columns
        if c.endswith("_lag") and c.startswith("scr_")
    ]
    if not scr_lag_cols:
        logging.warning("SCR segment panel has no lagged columns — skipping.")
        return panel

    # Ensure AnoMes types match
    df_scr["AnoMes"] = df_scr["AnoMes"].astype(int)
    if "AnoMes" in panel.columns:
        panel = panel.copy()
        panel["AnoMes"] = panel["AnoMes"].astype(int)

    merge_cols = ["segment", "AnoMes"] + scr_lag_cols
    df_scr_m   = df_scr[merge_cols].drop_duplicates(subset=["segment", "AnoMes"])

    if "segment" not in panel.columns:
        logging.warning("Institution panel has no 'segment' column — SCR merge skipped.")
        return panel

    n_before = len(panel)
    panel = panel.merge(df_scr_m, on=["segment", "AnoMes"], how="left")
    assert len(panel) == n_before, "SCR merge changed row count — check for duplicates."

    n_matched = panel[scr_lag_cols[0]].notna().sum()
    logging.info(
        "SCR segment instruments merged: %d / %d rows matched (%d lag columns)",
        n_matched, len(panel), len(scr_lag_cols),
    )
    return panel


# ================================================================================
# STEP 2 -- Build institution-level tariff panel
# ================================================================================

def build_institution_tariff_panel(
    df_tarif: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load egan_panel_institution.csv and merge tariff snapshot.
    Join key: CNPJ (int in existing panel) <-> cnpj (8-char str in tariff).
    """
    if not INST_PANEL_IN.exists():
        raise FileNotFoundError(
            f"Institution panel not found: {INST_PANEL_IN}\n"
            "Run egan_panel_build_2.py first."
        )

    panel = pd.read_csv(str(INST_PANEL_IN), dtype={"CNPJ": str})
    panel["CNPJ"] = panel["CNPJ"].astype(str).str.strip().str.zfill(8)
    logging.info(
        "Institution panel loaded: %d rows | %d institutions | %d quarters",
        len(panel),
        panel["CNPJ"].nunique(),
        panel["quarter"].nunique(),
    )

    wide = build_tariff_snapshot(df_tarif)

    # Apply time-conditional merge
    panel = apply_vigencia_mask(panel, wide, entity_col="CNPJ")

    # -- Add deposit-type relevance flag --------------------------------------
    # For each row, marks whether its deposit_type is in the primary target
    # group for the tariff category loaded.  This is a convenience column;
    # all tarif_* values are still present regardless.
    dep_type_codes = panel["deposit_type"].astype(float)
    relevant_by_type = {
        1: [c for c, m in SERVICE_CATALOG.items() if 1 in m["dep_types"]],
        2: [c for c, m in SERVICE_CATALOG.items() if 2 in m["dep_types"]],
        3: [c for c, m in SERVICE_CATALOG.items() if 3 in m["dep_types"]],
        4: [c for c, m in SERVICE_CATALOG.items() if 4 in m["dep_types"]],
        5: [c for c, m in SERVICE_CATALOG.items() if 5 in m["dep_types"]],
    }
    primary_tarifs = []
    for dt, codes in relevant_by_type.items():
        mask = dep_type_codes == dt
        panel.loc[mask, "tarif_primary_codes"] = ",".join(str(c) for c in codes)
        primary_tarifs.extend(codes)
    panel["tarif_primary_codes"] = panel["tarif_primary_codes"].astype(str)

    tarif_cols = sorted([c for c in panel.columns if c.startswith("tarif_") and c != "tarif_primary_codes"])
    logging.info(
        "Institution tariff panel complete: %d rows | %d tariff columns",
        len(panel), len(tarif_cols),
    )
    return panel


# ================================================================================
# STEP 3 -- Build conglomerate-level tariff panel
# ================================================================================

def norm_cong_code(s: str) -> int | None:
    """
    Normalise a CodConglomeradoPrudencial value to the integer used in
    egan_panel_deposits.csv as CodConglPrud.

    BCB uses two formats:
      'C0080075'  -> strip 'C', cast int -> 80075   (grouped prudential entity)
      '60746948'  -> cast int directly -> 60746948  (standalone, CNPJ = entity code)
      '00068389'  -> cast int -> 68389              (zero-padded CNPJ / entity code)
    """
    if pd.isna(s):
        return None
    s = str(s).strip()
    # Remove trailing ".0" from float-converted strings
    if s.endswith(".0"):
        s = s[:-2]
    if not s or s.lower() in ("nan", "null", "none"):
        return None
    if s.startswith("C"):
        try:
            return int(s[1:])  # 'C0080075' -> int('0080075') = 80075
        except ValueError:
            return None
    try:
        return int(s)          # '60746948' -> 60746948
    except ValueError:
        return None


def load_cnpj_to_cong_map() -> pd.DataFrame:
    """
    Build a comprehensive CNPJ (8-digit str) -> CodConglPrud (int) mapping
    by scanning ALL IF-Data List files.

    The mapping uses two join paths:
      A) Direct: rows where CodInst is a numeric CNPJ -> map to their
         CodConglomeradoPrudencial
      B) Leader: map CnpjInstituicaoLider -> normalized CodConglomeradoPrudencial
         so that a bank filing tariffs under its own CNPJ can be assigned to
         its conglomerate in the Prudential IF-Data panel

    Using ALL List files (not just the latest) covers historical conglomerates
    that may have been dissolved or restructured before the most recent snapshot.

    Returns DataFrame with columns:
        cnpj           - 8-digit zero-padded CNPJ string
        cod_cong_prud  - integer conglomerate code matching CodConglPrud in panel
    """
    list_files = sorted(glob.glob(str(IF_DATA_LIST_DIR / "IF_DATA_List_*.csv")))
    if not list_files:
        raise FileNotFoundError(
            f"No IF-Data List files found under {IF_DATA_LIST_DIR}"
        )
    logging.info(
        "Building CNPJ->conglomerate map from %d List files (%s ... %s)",
        len(list_files),
        Path(list_files[0]).name,
        Path(list_files[-1]).name,
    )

    frames: list[pd.DataFrame] = []
    for fpath in list_files:
        try:
            df = pd.read_csv(fpath, sep=",", encoding="latin-1", low_memory=False)
            df.columns = [c.strip() for c in df.columns]

            df["CodInst_str"] = df["CodInst"].astype(str).str.strip()
            numeric_mask = df["CodInst_str"].str.match(r"^\d+$")

            # Path A: numeric CodInst -> their own conglomerate code
            part_a = (
                df.loc[numeric_mask, ["CodInst_str", "CodConglomeradoPrudencial"]]
                .rename(columns={"CodInst_str": "cnpj", "CodConglomeradoPrudencial": "cod_cong_raw"})
            )
            part_a["cnpj"] = part_a["cnpj"].str.zfill(8)

            # Path B: CnpjInstituicaoLider -> conglomerate
            # This covers C-code entities (whose own CodInst is not a CNPJ)
            leader_valid = df["CnpjInstituicaoLider"].notna()
            part_b = (
                df.loc[leader_valid, ["CnpjInstituicaoLider", "CodConglomeradoPrudencial"]]
                .rename(columns={"CnpjInstituicaoLider": "cnpj", "CodConglomeradoPrudencial": "cod_cong_raw"})
            )
            part_b["cnpj"] = (
                part_b["cnpj"].astype(str)
                .str.replace(r"\.0$", "", regex=True)
                .str.strip().str.zfill(8)
            )

            frames.append(pd.concat([part_a, part_b], ignore_index=True))
        except Exception as exc:
            logging.warning("Skipping %s: %s", Path(fpath).name, exc)

    if not frames:
        raise ValueError("Could not load any IF-Data List files.")

    all_rows = pd.concat(frames, ignore_index=True)

    # Normalise raw conglomerate code -> int
    all_rows = all_rows.dropna(subset=["cod_cong_raw"])
    all_rows["cod_cong_prud"] = all_rows["cod_cong_raw"].apply(norm_cong_code)
    all_rows = all_rows.dropna(subset=["cod_cong_prud"])
    all_rows["cod_cong_prud"] = all_rows["cod_cong_prud"].astype(int)

    # For each institution CNPJ, keep the most recent conglomerate assignment
    # (dedup: same CNPJ in multiple files -> keep last occurrence = most recent)
    cmap = (
        all_rows
        .drop_duplicates(subset="cnpj", keep="last")
        [["cnpj", "cod_cong_prud"]]
        .reset_index(drop=True)
    )

    logging.info(
        "CNPJ->conglomerate map: %d institutions -> %d conglomerates",
        cmap["cnpj"].nunique(),
        cmap["cod_cong_prud"].nunique(),
    )
    return cmap


def build_cong_tariff_snapshot(
    df_tarif:  pd.DataFrame,
    cmap:      pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate institution-level tariffs to the conglomerate level.

    For each (cod_cong_prud, service_code, vigencia_quarter):
      - n_institutions: count of members reporting this tariff
      - tarif_median:   median valor_maximo across members
      - tarif_max:      max valor_maximo across members

    Returns a wide DataFrame keyed by cod_cong_prud.
    We use the MEDIAN as the primary column (tarif_{code}) to mitigate the
    influence of extreme values; the max is stored as tarif_{code}_max.
    """
    # Merge CNPJ -> conglomerate
    df = df_tarif.merge(cmap, on="cnpj", how="left")
    unmapped = df["cod_cong_prud"].isna().sum()
    if unmapped:
        logging.info(
            "  %d tariff rows (%d CNPJs) not matched to a conglomerate -- dropped",
            unmapped, df.loc[df["cod_cong_prud"].isna(), "cnpj"].nunique(),
        )
    df = df.dropna(subset=["cod_cong_prud"])
    df["cod_cong_prud"] = df["cod_cong_prud"].astype(int)

    # Best vigencia per (conglomerate, service_code): latest vigencia_quarter
    # across all member institutions (most informative for masking)
    snap = (
        df.sort_values("data_vigencia")
        .groupby(["cod_cong_prud", "codigo_servico"])
        .agg(
            valor_median=("valor_maximo", "median"),
            valor_max=("valor_maximo", "max"),
            vigencia_quarter=("data_vigencia", lambda s: (
                lambda d: f"{d.year}Q{(d.month - 1) // 3 + 1}" if pd.notna(d) else None
            )(s.dropna().max() if not s.dropna().empty else pd.NaT)),
        )
        .reset_index()
    )

    # Pivot median -> tarif_{code}
    val_wide = snap.pivot(
        index="cod_cong_prud",
        columns="codigo_servico",
        values="valor_median",
    )
    val_wide.columns = [f"tarif_{c}" for c in val_wide.columns]
    val_wide = val_wide.reset_index()

    # Pivot max -> tarif_{code}_max
    max_wide = snap.pivot(
        index="cod_cong_prud",
        columns="codigo_servico",
        values="valor_max",
    )
    max_wide.columns = [f"tarif_{c}_max" for c in max_wide.columns]
    max_wide = max_wide.reset_index()

    # Pivot vigência
    vig_wide = snap.pivot(
        index="cod_cong_prud",
        columns="codigo_servico",
        values="vigencia_quarter",
    )
    vig_wide.columns = [f"vigencia_{c}" for c in vig_wide.columns]
    vig_wide = vig_wide.reset_index()

    wide = val_wide.merge(max_wide, on="cod_cong_prud", how="outer")
    wide = wide.merge(vig_wide,  on="cod_cong_prud", how="outer")

    logging.info(
        "Conglomerate tariff snapshot: %d conglomerates | %d service codes",
        len(wide),
        snap["codigo_servico"].nunique(),
    )
    return wide


def build_cong_tariff_panel(
    df_tarif: pd.DataFrame,
    cmap:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Load egan_panel_deposits.csv and merge conglomerate-level tariff snapshot.
    """
    if not CONG_PANEL_IN.exists():
        raise FileNotFoundError(
            f"Conglomerate panel not found: {CONG_PANEL_IN}\n"
            "Run egan_panel_build.py first."
        )

    panel = pd.read_csv(str(CONG_PANEL_IN))
    panel["CodConglPrud"] = pd.to_numeric(panel["CodConglPrud"], errors="coerce").astype("Int64")
    logging.info(
        "Conglomerate panel loaded: %d rows | %d conglomerates | %d quarters",
        len(panel),
        panel["CodConglPrud"].nunique(),
        panel["quarter"].nunique(),
    )

    wide = build_cong_tariff_snapshot(df_tarif, cmap)

    # Merge onto panel
    panel = panel.merge(
        wide,
        left_on="CodConglPrud",
        right_on="cod_cong_prud",
        how="left",
    )
    panel = panel.drop(columns=["cod_cong_prud"], errors="ignore")

    # Apply time-conditional vigencia mask
    def qtr_to_int(q: str) -> int:
        if not isinstance(q, str) or "Q" not in q:
            return 0
        y, n = q.split("Q")
        return int(y) * 10 + int(n)

    panel_qtr_int = panel["quarter"].apply(qtr_to_int)
    tarif_median_cols = [c for c in panel.columns if c.startswith("tarif_") and not c.endswith("_max")]
    tarif_max_cols    = [c for c in panel.columns if c.startswith("tarif_") and c.endswith("_max")]
    vigencia_cols     = [c for c in panel.columns if c.startswith("vigencia_")]

    for tarif_col in tarif_median_cols:
        code         = tarif_col.split("_", 1)[1]
        vigencia_col = f"vigencia_{code}"
        if vigencia_col in panel.columns:
            vig_int  = panel[vigencia_col].apply(qtr_to_int)
            too_early = panel_qtr_int < vig_int
            panel.loc[too_early, tarif_col] = np.nan
            max_col = f"tarif_{code}_max"
            if max_col in panel.columns:
                panel.loc[too_early, max_col] = np.nan

    panel = panel.drop(columns=vigencia_cols, errors="ignore")

    n_filled = panel[tarif_median_cols].notna().any(axis=1).sum()
    logging.info(
        "Conglomerate tariff panel complete: %d rows | %d / %d rows with >=1 tariff",
        len(panel), n_filled, len(panel),
    )
    return panel


# ================================================================================
# MAIN
# ================================================================================

def _print_coverage(df: pd.DataFrame, entity_col: str, label: str) -> None:
    tarif_cols = [c for c in df.columns if c.startswith("tarif_") and not c.endswith("_max") and c != "tarif_primary_codes"]
    n_total    = len(df)
    n_any      = df[tarif_cols].notna().any(axis=1).sum()
    n_full     = df[tarif_cols].notna().all(axis=1).sum()
    print(f"\n{'-' * 60}")
    print(f"{label}")
    print(f"{'-' * 60}")
    print(f"  Shape                  : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Unique {entity_col:<14}: {df[entity_col].nunique():,}")
    print(f"  Unique quarters        : {df['quarter'].nunique()}"
          f"  ({df['quarter'].min()} - {df['quarter'].max()})")
    print(f"  Rows with >=1 tariff    : {n_any:,} / {n_total:,}"
          f"  ({100 * n_any / n_total:.1f}%)")
    print(f"  Rows with all tariffs  : {n_full:,} / {n_total:,}"
          f"  ({100 * n_full / n_total:.1f}%)")
    print(f"  Tariff columns         : {len(tarif_cols)}")

    # Per deposit type coverage
    if "deposit_type_name" in df.columns:
        print()
        cov_dt = (
            df.groupby("deposit_type_name", observed=True)
            .apply(lambda g: g[tarif_cols].notna().any(axis=1).mean())
            .rename("tariff_coverage")
            .mul(100)
            .round(1)
        )
        print("  Tariff coverage (>=1 tariff) by deposit type:")
        for nm, pct in cov_dt.items():
            print(f"    {nm:<25} {pct:.1f}%")


if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Starting Tariff Panel Build")
    logging.info("=" * 60)

    # -- Step 1: Load tariff data ---------------------------------------------
    print("Step 1/4: Loading tariff long panel ...")
    df_tarif = load_tariff_long()
    n_svc    = df_tarif["codigo_servico"].nunique()
    n_inst   = df_tarif["cnpj"].nunique()
    print(f"         {n_inst:,} institutions | {n_svc} service codes | "
          f"{df_tarif['data_coleta'].nunique()} scrape date(s)")

    # -- Step 2: CNPJ -> conglomerate mapping ----------------------------------
    print("Step 2/4: Loading CNPJ -> conglomerate mapping ...")
    cmap = load_cnpj_to_cong_map()

    # -- Step 3: Build institution-level tariff panel --------------------------
    print("Step 3/4: Building institution tariff panel ...")
    inst_panel = build_institution_tariff_panel(df_tarif)

    # -- Step 3b: Merge SCR segment credit-structure instruments ---------------
    print("Step 3b/4: Merging SCR segment instruments (if available) ...")
    inst_panel = merge_scr_instruments(inst_panel)

    inst_panel.to_csv(str(INST_PANEL_OUT), index=False)
    logging.info("Institution tariff panel saved: %s", INST_PANEL_OUT.name)

    # -- Step 4: Build conglomerate-level tariff panel ------------------------
    print("Step 4/4: Building conglomerate tariff panel ...")
    cong_panel = build_cong_tariff_panel(df_tarif, cmap)
    cong_panel.to_csv(str(CONG_PANEL_OUT), index=False)
    logging.info("Conglomerate tariff panel saved: %s", CONG_PANEL_OUT.name)

    # -- Coverage summary -----------------------------------------------------
    _print_coverage(inst_panel, "CNPJ", f"Institution panel  ->  {INST_PANEL_OUT.name}")
    _print_coverage(cong_panel, "CodConglPrud", f"Conglomerate panel ->  {CONG_PANEL_OUT.name}")

    print(f"\n{'-' * 60}")
    print("Tariff column dictionary (50 service codes)")
    print("-" * 60)
    print(f"  {'Code':<6} {'Category':<15} {'Dep.types':<15} Description")
    print(f"  {'----':<6} {'--------':<15} {'---------':<15} {'-------------------------------------------------'}")
    for code, meta in sorted(SERVICE_CATALOG.items()):
        dt = ",".join(str(x) for x in meta["dep_types"]) if meta["dep_types"] else "--"
        print(f"  {code:<6} {meta['cat']:<15} {dt:<15} {meta['en']}")

    print("\nDone.")
    logging.info("Script completed successfully.")
