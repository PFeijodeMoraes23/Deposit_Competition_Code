## deposits_panel_build.py
# Authors: Pedro Feijó de Moraes
#
# Last edited on 2026_03_06
#
# Purpose: Build a balanced panel at the conglomerate × municipality × quarter level
#          combining two data sources:
#
#   TIER 1 – ESTBAN (BCB): banking institutions with branch networks.
#            Source: BCB/ESTBAN/ESTBAN.csv 
#                    + raw monthly CSVs in BCB/ESTBAN/Relatório por município/ (for 2013–2015).
#            Granularity: institution × municipality × month  →  conglomerate × municipality × quarter.
#            Deposit columns: dep_a1 (V400_401), dep_a2 (V420), dep_a3 (V431), dep_a4 (V432).
#            a5 (prepaid) is NOT in ESTBAN; set to NaN.
#
#   TIER 2 – IF Data (BCB prudential conglomerate reports):
#            Source: BCB/IF Data/Aggregated Data/IF_DATA_type_1_report_3.csv
#            Granularity: conglomerate × quarter (national total only).
#            Only institutions with no ESTBAN row in 2016-2024 are included here, and only in
#            quarters where they have no ESTBAN row; for them CODMUN_IBGE is set to 0 (a code
#            that matches no real municipality).
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
#   • A quarter-end cell whose branch did not file (AGEN_PROCESSADAS = 0, balances blank) is not
#     zero: a gap takes the same cell's nearest filed month of the quarter, a closure (the cell
#     never files again) is an exit, and a cell with no filed month in the quarter stays missing.
#     Every such cell is listed in PANEL_INTERMED/estban_unfiled_cells.csv (utils/estban.py).
###-------------------------------------------------------------------------------------------


## 1) Load necessary packages and set paths and constants:

# Load necessary packages:
import os
import glob
import re
import logging
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import numpy as np


from utils import paths
from utils.conglomerates import (normalize_str, coerce_cnpj, build_conglomerate_canonical_map,
                                 build_cnpj_conglomerate_map)
from utils.estban import UNFILED_CELLS_CSV, fill_unfiled_quarter_ends, process_raw_estban_csv
from utils import firm_remaps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Set paths (canonical locations in utils/paths.py):
ESTBAN_PROC_CSV  = str(paths.ESTBAN_CSV)        # processed 2016–2024
ESTBAN_RAW_MUN   = str(paths.ESTBAN_RAW_MUN)    # raw monthly CSVs
IF_AGG_DIR       = str(paths.IF_DATA_AGG)
IF_LIST_DIR      = str(paths.IF_DATA_LIST)
OUTPUT_DIR       = str(paths.PROCESSED)


os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global constants and variables:
# IF Data report 3 (Passivo – Captações) NumeroConta values confirmed from data:
# Pre-2025 format:
ACCT_A1     = 78282   # Depositos a Vista           (demand deposits)
ACCT_A2     = 78283   # Depositos de Poupanca       (savings)
ACCT_A3     = 78284   # Depositos Interfinanceiros  (interbank)
ACCT_A4     = 78286   # Depositos a Prazo           (time deposits)
ACCT_OUTROS = 78285   # Outros Depositos            (other conventional deposits)
ACCT_A5     = 110560  # Conta de Pagamento PrePaga  (prepaid payment accounts, true a5)

# 2025+ format (BCB renumbered all accounts in the new IF Data plan):
ACCT_A1_NEW     = 140222  # Depositos a Vista
ACCT_A2_NEW     = 140223  # Depositos de Poupanca
ACCT_A3_NEW     = 140224  # Depositos Interfinanceiros
ACCT_A4_NEW     = 140225  # Depositos a Prazo
ACCT_OUTROS_NEW = 140227  # Depositos Outros
ACCT_A5_NEW     = 140226  # Conta de Pagamento PrePaga

# Quarter-end months: use these ESTBAN months to represent each quarter
QUARTER_END_MONTHS = {3: 1, 6: 2, 9: 3, 12: 4}   # month → quarter number

# The analysis window on which a firm's tier (ESTBAN or IF Data national) is decided (main, step 3).
TIER_WINDOW = (2016, 2024)

# Sentinel municipality code for institutions with no geographic breakdown
NO_MUN_CODE = 0

# Hard cap: the analysis panel ends at 2025-Q4.  Raw data may extend further
# (e.g. the 2026 ESTBAN months BCB has already published), but everything after
# 2025-Q4 is dropped so the panel has a fixed, reproducible end point.
PANEL_END_YEAR    = 2025
PANEL_END_QUARTER = 4

## 2) User-defined functions:




# 2.4) ESTBAN panel building: 

ESTBAN_DEPOSIT_COLS = ["V400_401", "V420", "V431", "V432"]   # a1, a2, a3, a4
ESTBAN_KEEP_COLS    = ["CNPJ", "NOME_INSTITUICAO", "CODMUN_IBGE", "YEAR", "MONTH"] + ESTBAN_DEPOSIT_COLS
# Every month is read, not only the quarter-ends: an unfiled quarter-end cell takes the balances of
# the same cell's nearest filed month in the quarter (utils.estban.fill_unfiled_quarter_ends), and the
# months after it decide whether the branch closed. CODMUN (the BCB municipality code) is the cell key,
# because ESTBAN.csv leaves CODMUN_IBGE blank on an unfiled cell.
ESTBAN_MONTHLY_COLS = ESTBAN_KEEP_COLS + ["CODMUN", "AGEN_ESPERADAS", "AGEN_PROCESSADAS"]


def _standardise_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """ESTBAN_MONTHLY_COLS with their types: CNPJ, CODMUN and CODMUN_IBGE Int64, balances numeric."""
    for col in ESTBAN_MONTHLY_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[ESTBAN_MONTHLY_COLS].copy()
    df["CNPJ"] = coerce_cnpj(df["CNPJ"])
    for col in ("CODMUN", "CODMUN_IBGE"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ESTBAN_DEPOSIT_COLS + ["AGEN_ESPERADAS", "AGEN_PROCESSADAS"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_estban_processed() -> pd.DataFrame:
    """Every month of the pre-processed ESTBAN.csv (2016–present; built by scrape_estban_concat.py),
    in ESTBAN_MONTHLY_COLS."""
    logging.info("Loading ESTBAN.csv …")
    head = pd.read_csv(ESTBAN_PROC_CSV, encoding="latin1", nrows=0).columns
    df = pd.read_csv(ESTBAN_PROC_CSV, encoding="latin1", low_memory=False,
                     usecols=[c for c in ESTBAN_MONTHLY_COLS if c in head])
    return _standardise_monthly(df)

def load_estban_raw_pre2016() -> pd.DataFrame:
    """
    Process raw ESTBAN CSVs for any year < 2016 found in the raw folder.
    Returns every month, with the same columns as load_estban_processed().
    """
    pattern_upper = os.path.join(ESTBAN_RAW_MUN, "*.CSV")
    pattern_lower = os.path.join(ESTBAN_RAW_MUN, "*.csv")
    # On Windows the match is case-insensitive, so both patterns list every file: keep one path
    # per file, or each month's balances are read, and summed, twice.
    found = glob.glob(pattern_upper) + glob.glob(pattern_lower)
    all_files = sorted({os.path.normcase(os.path.abspath(p)): p for p in found}.values())

    pre2016_files = []
    for fp in all_files:
        fname = os.path.basename(fp)
        # Expect format: YYYYMM_ESTBAN.CSV
        match = re.match(r"^(\d{4})\d{2}_ESTBAN", fname, re.IGNORECASE)
        if match and int(match.group(1)) < 2016:
            pre2016_files.append(fp)

    if not pre2016_files:
        logging.info("No pre-2016 raw ESTBAN files found. Skipping raw processing.")
        return pd.DataFrame(columns=ESTBAN_MONTHLY_COLS)

    logging.info(f"Processing {len(pre2016_files)} raw pre-2016 ESTBAN files …")
    frames = []
    for fp in pre2016_files:
        df = process_raw_estban_csv(fp)
        if df is None or df.empty:
            continue
        frames.append(_standardise_monthly(df))

    if not frames:
        return pd.DataFrame(columns=ESTBAN_MONTHLY_COLS)

    raw_df = pd.concat(frames, ignore_index=True)
    logging.info(f"Raw pre-2016 ESTBAN: {len(raw_df)} rows, every month")
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
    monthly = pd.concat([proc, raw], ignore_index=True)

    if monthly.empty:
        logging.error("No ESTBAN data loaded. Check paths.")
        return pd.DataFrame()

    # --- Cells that did not file: fill the gaps, drop the closures (utils.estban) ---
    monthly, cells = fill_unfiled_quarter_ends(monthly, ESTBAN_DEPOSIT_COLS, QUARTER_END_MONTHS)
    os.makedirs(os.path.dirname(UNFILED_CELLS_CSV), exist_ok=True)
    cells.to_csv(UNFILED_CELLS_CSV, index=False)
    kinds = cells["kind"].value_counts().to_dict()
    logging.info(f"Unfiled quarter-end cells: {len(cells)} {kinds} -> {UNFILED_CELLS_CSV}")
    gap = cells[cells["kind"] == "gap"]
    if not gap.empty:
        by_bank = (gap.assign(bn=gap[ESTBAN_DEPOSIT_COLS].fillna(0).sum(axis=1) / 1e9)
                      .groupby(["YEAR", "MONTH", "CNPJ", "NOME_INSTITUICAO"], dropna=False)
                      .agg(cells=("CODMUN", "size"), whole_bank=("whole_bank", "all"), bn=("bn", "sum")))
        for (y, m, cnpj, name), r in by_bank[by_bank["bn"] >= 0.05].iterrows():
            logging.info(f"  filled {y}-{m:02d} {cnpj} {str(name).strip()}: {r['cells']} cell(s), "
                         f"R$ {r['bn']:.2f} bn{' (whole bank)' if r['whole_bank'] else ''}")

    # Keep only quarter-end months
    estban = monthly[monthly["MONTH"].isin(QUARTER_END_MONTHS)][ESTBAN_KEEP_COLS].copy()

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
    # Drop rows where CODMUN_IBGE is missing or 0 to prevent glitchy D-type firm classification
    estban = estban.dropna(subset=["CODMUN_IBGE"])
    estban["CODMUN_IBGE"] = estban["CODMUN_IBGE"].astype(int)
    estban = estban[estban["CODMUN_IBGE"] != 0]

    agg_cols = {col: "sum" for col in ESTBAN_DEPOSIT_COLS if col in estban.columns}
    # NomeInstituicao is intentionally excluded from group_cols: multiple institutions
    # within the same prudential conglomerate may carry different names (e.g. Bradesco
    # + Agora Corretora both map to C0080075). Including NomeInstituicao would split
    # what should be a single conglomerate×market×quarter observation into multiple rows,
    # inflating G and violating the panel structure.
    # We instead aggregate purely on the organisational keys and re-attach a
    # representative name afterwards.
    # CNPJ_Lider is excluded for the same reason and re-attached the same way: it is a
    # property of the conglomerate, not of the market-quarter, so two member banks
    # whose List rows name different leaders would split one observation into two rows
    # carrying the same market and quarter.
    group_cols = ["CodConglomeradoPrudencial", "CODMUN_IBGE", "YEAR", "Quarter"]

    panel = (
        estban.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()
    )

    # Attach representative CNPJ_Lider: the modal leader across the conglomerate's
    # member banks.  This also carries the 'CNPJ_<cnpj>' fallback rows, where the code
    # is unique to one bank and the modal leader is that bank's own CNPJ.
    cong_lider = (
        estban.dropna(subset=["CNPJ_Lider"])
              .groupby("CodConglomeradoPrudencial")["CNPJ_Lider"]
              .agg(lambda s: s.value_counts().idxmax())
    )
    panel["CNPJ_Lider"] = panel["CodConglomeradoPrudencial"].map(cong_lider)

    # Attach representative NomeInstituicao: prefer the lead institution's own name
    # (i.e. the entry where institution CNPJ == CNPJ_Lider).
    cong_nome = (
        estban[["CodConglomeradoPrudencial", "CNPJ", "CNPJ_Lider", "NomeInstituicao"]]
        .copy()
        .assign(is_leader=(estban["CNPJ"].astype("Int64") == estban["CNPJ_Lider"].astype("Int64")))
        .sort_values("is_leader", ascending=False)          # leaders first
        .drop_duplicates(subset=["CodConglomeradoPrudencial"])
        .set_index("CodConglomeradoPrudencial")["NomeInstituicao"]
    )
    panel["NomeInstituicao"] = panel["CodConglomeradoPrudencial"].map(cong_nome)

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

    # Keep only the deposit accounts we care about (both pre-2025 and 2025+ codes)
    deposit_accounts = [
        ACCT_A1, ACCT_A2, ACCT_A3, ACCT_A4, ACCT_OUTROS, ACCT_A5,
        ACCT_A1_NEW, ACCT_A2_NEW, ACCT_A3_NEW, ACCT_A4_NEW, ACCT_OUTROS_NEW, ACCT_A5_NEW,
    ]
    df = df[df["NumeroConta"].isin(deposit_accounts)].copy()

    # Map NumeroConta → deposit column name (old and new codes map to same columns)
    acct_col_map = {
        ACCT_A1:         "dep_a1",
        ACCT_A2:         "dep_a2",
        ACCT_A3:         "dep_a3",
        ACCT_A4:         "dep_a4",
        ACCT_OUTROS:     "dep_outros",
        ACCT_A5:         "dep_a5",
        ACCT_A1_NEW:     "dep_a1",
        ACCT_A2_NEW:     "dep_a2",
        ACCT_A3_NEW:     "dep_a3",
        ACCT_A4_NEW:     "dep_a4",
        ACCT_OUTROS_NEW: "dep_outros",
        ACCT_A5_NEW:     "dep_a5",
    }
    df["dep_col"] = df["NumeroConta"].map(acct_col_map)

    # Assign quarter (IF Data is already quarterly: months 3, 6, 9, 12)
    df["Quarter"] = df["Month"].map(QUARTER_END_MONTHS)
    df = df.dropna(subset=["Quarter", "CodConglomeradoPrudencial"])
    df["Quarter"] = df["Quarter"].astype(int)

    # One prudential code can be carried by more than one reporting entity in the same
    # quarter: the conglomerate's own aggregate row (CNPJ == the code), rows of member
    # institutions, and the row of a FINANCIAL conglomerate sitting inside it.  Their
    # values repeat the same balance, so summing them counts it twice -- Nu Pagamentos
    # C0084693 reports dep_a4 of R$75.3bn for 2022-Q4 under two entities, and the panel
    # carried both.  Keep the conglomerate's own row wherever it exists; every one of
    # the 663 repeated account-quarters has exactly one, and the handful of
    # account-quarters with no such row have only a single row anyway.
    if "CNPJ" in df.columns:
        _acct_key = ["CodConglomeradoPrudencial", "Year", "Quarter", "NumeroConta"]
        df["_own_row"] = (df["CNPJ"].astype(str)
                          == df["CodConglomeradoPrudencial"].astype(str))
        _n_before = len(df)
        df = (df.sort_values("_own_row", ascending=False, kind="stable")
                .drop_duplicates(subset=_acct_key, keep="first")
                .drop(columns="_own_row"))
        if (_repeats := _n_before - len(df)):
            logging.info(
                f"IF Data: dropped {_repeats:,} rows repeating a conglomerate's account "
                f"under a member or financial-conglomerate entity."
            )

    # Names are keyed on the code AS REPORTED, before canonicalisation, so a code that
    # survives keeps exactly the name it had.
    nome_src = (df[["CodConglomeradoPrudencial", "NomeInstituicao"]].copy()
                if "NomeInstituicao" in df.columns else None)

    # Collapse each firm's successive codes onto its canonical one (see
    # build_conglomerate_canonical_map).  Done BEFORE the pivot so that a quarter
    # reported under both an old and a new code -- Agibank's 2016, where Banco Gerador
    # C0081809 and Agiplan C0083694 overlap until the Q3 acquisition -- becomes one
    # firm rather than two, and so that the ESTBAN/IF Data de-duplication in main()
    # compares like with like.
    canonical_code, canonical_lider = build_conglomerate_canonical_map()
    _reported_code = df["CodConglomeradoPrudencial"]
    df["CodConglomeradoPrudencial"] = _reported_code.map(canonical_code).fillna(_reported_code)

    # Pivot to wide format.
    # NomeInstituicao is excluded from group_cols: a prudential conglomerate can
    # have multiple member institutions in the report (e.g. XP Investimentos +
    # Rico Corretora both under C0082475). Including NomeInstituicao would split
    # them into separate rows, fragmenting the deposit totals. Instead we
    # aggregate purely on the organisational keys and re-attach a representative
    # name afterwards (preferring any entry whose NomeInstituicao contains
    # "PRUDENCIAL", falling back to the first available name).
    # CNPJ_Lider is excluded for the same reason: the report's leader field changes
    # when the lead institution changes, and Nu Pagamentos C0084693 carries two leaders
    # at once in 2022-Q4 and 2023-Q1/Q2, which split one conglomerate-quarter into two
    # rows holding the same deposits.  It is re-attached per conglomerate below.
    group_cols = ["CodConglomeradoPrudencial", "Year", "Quarter"]
    for col in group_cols:
        if col not in df.columns:
            df[col] = pd.NA
    if "CNPJ_Lider" not in df.columns:
        df["CNPJ_Lider"] = pd.NA

    pivot = (
        df.groupby(group_cols + ["dep_col"], dropna=False)["Value"].sum().unstack("dep_col").reset_index()
    )

    # Attach CNPJ_Lider: the canonical code's lead institution, falling back to the
    # modal leader the report itself gives for codes the List files do not cover.
    reported_lider = (df.dropna(subset=["CNPJ_Lider"])
                        .groupby("CodConglomeradoPrudencial")["CNPJ_Lider"]
                        .agg(lambda s: s.value_counts().idxmax()))
    pivot["CNPJ_Lider"] = pivot["CodConglomeradoPrudencial"].map(canonical_lider)
    pivot["CNPJ_Lider"] = pivot["CNPJ_Lider"].fillna(
        pivot["CodConglomeradoPrudencial"].map(reported_lider)
    )

    # Derive representative NomeInstituicao: prefer the "PRUDENCIAL" entry.
    if nome_src is not None:
        nome_df = nome_src.drop_duplicates()
        nome_df = nome_df.dropna(subset=["NomeInstituicao"])
        # sort so "PRUDENCIAL" names appear first, then take first per conglomerate
        nome_df = nome_df.assign(
            _prud=nome_df["NomeInstituicao"].astype(str).str.contains("PRUDENCIAL", case=False)
        ).sort_values("_prud", ascending=False).drop(columns="_prud")
        nome_map = nome_df.drop_duplicates(subset=["CodConglomeradoPrudencial"]).set_index(
            "CodConglomeradoPrudencial"
        )["NomeInstituicao"]
        pivot["NomeInstituicao"] = pivot["CodConglomeradoPrudencial"].map(nome_map)
    else:
        pivot["NomeInstituicao"] = pd.NA

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
    # Step 0: Resolve one conglomerate code per firm.  Both tiers are built against
    # this, so ESTBAN and IF Data name the same firm the same way and the Tier-1/Tier-2
    # de-duplication in step 4c can see when they describe the same deposits.
    canonical_code, _canonical_lider = build_conglomerate_canonical_map()

    # Step 1: Build CNPJ → conglomerate mapping
    cnpj_map = build_cnpj_conglomerate_map()

    # Step 2: Build ESTBAN panel
    estban_panel = build_estban_panel(cnpj_map)

    # Step 3: Identify conglomerates in the ESTBAN panel. The tier is decided per firm on the analysis
    # window: a firm with ESTBAN rows in 2016-2024 is ESTBAN tier; one whose ESTBAN rows all fall
    # outside the window (Rodobens, Randon, Andbank before 2016; Travelex from 2025) keeps its IF Data
    # national rows in every quarter where it has no ESTBAN row, instead of vanishing from the window.
    estban_conglomerates = set(estban_panel["CodConglomeradoPrudencial"].dropna().unique())
    estban_window = set(estban_panel.loc[estban_panel["Year"].between(*TIER_WINDOW),
                                         "CodConglomeradoPrudencial"].dropna().unique())
    estban_cq = set(zip(estban_panel["CodConglomeradoPrudencial"], estban_panel["Year"], estban_panel["Quarter"]))
    logging.info(f"ESTBAN panel covers {len(estban_conglomerates)} distinct conglomerates, "
                 f"{len(estban_window)} with ESTBAN rows in {TIER_WINDOW[0]}-{TIER_WINDOW[1]}")

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
    # PicPay is one firm, C0088022 (utils/firm_remaps.py): Banco Original's conglomerate prepaid
    # balance of 2019Q4-2024Q3 is PicPay's wallet and is spread over PicPay's own cells.
    _pp = (if_a5["CodConglomeradoPrudencial"] == firm_remaps.PICPAY_FROM) & \
        firm_remaps.picpay_window(if_a5["Year"], if_a5["Quarter"])
    logging.info(f"PicPay re-map: {int(_pp.sum())} quarters of {firm_remaps.PICPAY_FROM} prepaid "
                 f"(R$ {if_a5.loc[_pp, 'national_a5'].sum() / 1e9:.2f} bn-quarters) -> {firm_remaps.PICPAY_TO}")
    if_a5.loc[_pp, "CodConglomeradoPrudencial"] = firm_remaps.PICPAY_TO
    if_a5 = if_a5.groupby(["CodConglomeradoPrudencial", "Year", "Quarter"], as_index=False)["national_a5"].sum()
    
    # 2. Merge into ESTBAN.  ifdata_full holds one row per conglomerate-quarter, so
    #    this left join cannot multiply ESTBAN rows; assert it rather than assume it,
    #    because a duplicated national row silently doubles a whole municipal footprint.
    _dup_a5 = if_a5.duplicated(subset=["CodConglomeradoPrudencial", "Year", "Quarter"]).sum()
    if _dup_a5:
        raise ValueError(
            f"IF Data national a5 has {_dup_a5} duplicate conglomerate-quarters; the "
            f"merge below would multiply ESTBAN rows."
        )
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
    # Weight = the municipality's share of the conglomerate-quarter's demand deposits (a1). A
    # conglomerate with no ESTBAN demand deposits that quarter is weighted by its total ESTBAN
    # deposits (a1 + a2 + a4) instead, and one with no ESTBAN deposits at all equally over its ESTBAN
    # rows: weighting by a1 alone gave every municipality zero, and the whole prepaid balance dropped
    # out (PagBank, R$1-2.3 bn a quarter in 2020-2024; PicPay's wallet in 2020Q3-2022Q2, when PicPay
    # Bank's one ESTBAN row held no deposits).
    _tot = estban_panel[["dep_a1", "dep_a2", "dep_a4"]].fillna(0).sum(axis=1)
    estban_panel["_tot"] = _tot
    _grp = estban_panel.groupby(["CodConglomeradoPrudencial", "Year", "Quarter"])
    national_tot = _grp["_tot"].transform("sum")
    n_rows = _grp["_tot"].transform("size")
    estban_panel["a1_weight"] = np.where(
        estban_panel["national_estban_a1"] > 0,
        estban_panel["dep_a1"] / estban_panel["national_estban_a1"],
        np.where(national_tot > 0, estban_panel["_tot"] / national_tot, 1.0 / n_rows)
    )
    for _lab, _m in (("total ESTBAN deposits (no demand deposits)",
                      (estban_panel["national_estban_a1"] <= 0) & (national_tot > 0)),
                     ("equal weights over the ESTBAN rows (no ESTBAN deposits)", national_tot <= 0)):
        _m = _m & (estban_panel["national_a5"] > 0)
        if _m.any():
            _cq = estban_panel.loc[_m].drop_duplicates(["CodConglomeradoPrudencial", "Year", "Quarter"])
            logging.info(f"a5 spread by {_lab}: {len(_cq)} conglomerate-quarters, "
                         f"R$ {_cq['national_a5'].sum() / 1e9:.2f} bn; conglomerates "
                         f"{sorted(_cq['CodConglomeradoPrudencial'].astype(str).unique())}")

    # Apply weight. Only override dep_a5 if we actually have national_a5 to allocate.
    estban_panel["dep_a5"] = np.where(
        estban_panel["national_a5"].notna(),
        estban_panel["national_a5"] * estban_panel["a1_weight"],
        np.nan
    )

    # Clean up working columns
    estban_panel.drop(columns=["national_a5", "national_estban_a1", "a1_weight", "_tot"], inplace=True)

    # Step 4c: Filter IF Data panel to strictly Non-ESTBAN conglomerates (Tier-2)
    #
    # TIER RULE, stated explicitly because it now decides cases it used to miss: when a
    # firm is present in BOTH sources, its ESTBAN municipality rows are what survive and
    # its IF Data national row is dropped.  That has always been the panel's design --
    # Tier 2 exists to cover the institutions ESTBAN cannot see -- and canonicalising
    # the codes only makes the test fire on firms whose two tiers used to carry
    # different codes and so both survived.  The rule does not depend on which of a
    # firm's codes is canonical: after canonicalisation both tiers carry that one code
    # either way, so the same tier wins.
    #
    # Consequence to keep in view for Agibank, Industval and BBM.  They stop being
    # counted twice and stay LOCAL, which for Agibank means its national row goes and
    # its ESTBAN rows are all that is left.  ESTBAN reports the municipality where a
    # bank BOOKS a balance, not where it serves, so Agibank's whole balance sits in one
    # municipality at a time and migrates Recife (2013-16) -> Porto Alegre (2016-21) ->
    # Campinas (2021-25) as the head office moves.  For a bank with a branch network
    # that is a fair summary; for one whose network is 991 postos across 679
    # municipalities it is not, and the moves read downstream as multi-billion entry and
    # exit events in the two markets involved.
    _collapsed_codes = {c for c, keep in canonical_code.items() if c != keep}
    for _keep in sorted({canonical_code[c] for c in _collapsed_codes}):
        if _keep not in estban_window:
            continue
        _n_est = int((estban_panel["CodConglomeradoPrudencial"] == _keep).sum())
        _n_ifd = int((ifdata_full["CodConglomeradoPrudencial"] == _keep).sum())
        _muns = estban_panel.loc[estban_panel["CodConglomeradoPrudencial"] == _keep,
                                 "CODMUN_IBGE"].nunique()
        logging.warning(
            f"Tier rule: {_keep} is reported in both sources once its codes are "
            f"collapsed. Keeping {_n_est} ESTBAN rows across {_muns} municipalities and "
            f"dropping {_n_ifd} IF Data national rows; the firm's surviving geography is "
            f"its ESTBAN booking municipality, not its service network."
        )

    _in_estban_q = pd.Series([cq in estban_cq for cq in zip(ifdata_full["CodConglomeradoPrudencial"],
                                                             ifdata_full["Year"], ifdata_full["Quarter"])],
                             index=ifdata_full.index)
    ifdata_panel = ifdata_full[~(ifdata_full["CodConglomeradoPrudencial"].isin(estban_window) | _in_estban_q)].copy()
    _kept = ifdata_panel[ifdata_panel["CodConglomeradoPrudencial"].isin(estban_conglomerates)]
    if not _kept.empty:
        logging.info(f"Tier per firm on {TIER_WINDOW[0]}-{TIER_WINDOW[1]}: {len(_kept)} IF Data national rows kept "
                     f"for {_kept['CodConglomeradoPrudencial'].nunique()} firms whose ESTBAN rows lie outside "
                     f"the window: {sorted(_kept['CodConglomeradoPrudencial'].astype(str).unique())}")

    # Every firm-quarter must now sit in exactly one tier, and every firm with ESTBAN rows in the
    # window in the ESTBAN tier only.
    _both_q = set(zip(ifdata_panel["CodConglomeradoPrudencial"], ifdata_panel["Year"], ifdata_panel["Quarter"])) & estban_cq
    _both = set(ifdata_panel["CodConglomeradoPrudencial"].dropna()) & estban_window
    if _both_q or _both:
        raise ValueError(f"{len(_both_q)} conglomerate-quarters survive in both tiers "
                         f"({sorted(_both_q)[:5]}); {len(_both)} window ESTBAN firms keep IF rows ({sorted(_both)[:10]})")
    _codes_per_lider = (pd.concat([estban_panel, ifdata_panel])
                          .dropna(subset=["CNPJ_Lider"])
                          .groupby("CNPJ_Lider")["CodConglomeradoPrudencial"].nunique())
    if (_codes_per_lider > 1).any():
        raise ValueError(
            f"{int((_codes_per_lider > 1).sum())} lead institutions still carry more than "
            f"one conglomerate code: {_codes_per_lider[_codes_per_lider > 1].to_dict()}"
        )


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
    
    # Filter out BNDES (Development Bank, should not be in the sample)
    panel = panel[~panel["NomeInstituicao"].astype(str).str.contains("BNDES", case=False, na=False)]
    panel.reset_index(drop=True, inplace=True)

    # Hard cap at 2025-Q4: drop any later quarters present in the raw data.
    _cap = PANEL_END_YEAR * 4 + PANEL_END_QUARTER
    _n_before = len(panel)
    panel = panel[~((panel["Year"] * 4 + panel["Quarter"]) > _cap)].copy()
    panel.reset_index(drop=True, inplace=True)
    if (_dropped := _n_before - len(panel)):
        logging.info(f"Capped panel at {PANEL_END_YEAR}-Q{PANEL_END_QUARTER}: dropped {_dropped:,} later-quarter rows.")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "deposits_panel.csv")
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    pa_csv.write_csv(pa.Table.from_pandas(panel, preserve_index=False), out_path)
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
