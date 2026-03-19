## cosif_process_1.py
# Author: Pedro Feijo de Moraes
# Last edited: 2026-02-26
# Objective: Process locally-downloaded COSIF data on deposits and deposit rates for Brazil
## ------------------------------------------------------------------------------------------

## 1) Load packages and set up paths
import pandas as pd
import numpy as np
import zipfile
import logging
import os
import sys
import re

from logging.handlers import RotatingFileHandler

# Directory Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BCB_PATH = os.path.join(PARENT_DIR, "BCB")
main_path = os.path.join(BCB_PATH, "Egan_et_al_2025_Rep")
RAW_DATA_PATH = os.path.join(main_path, "raw")
PROCESSED_DATA_PATH = os.path.join(main_path, "processed")
INPUT_COSIF_DIR = os.path.join(RAW_DATA_PATH, "COSIF_RAW")
output_path = os.path.join(PROCESSED_DATA_PATH, "COSIF_PROCESSED")

for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, output_path]:
    os.makedirs(path, exist_ok=True)

# Logging
log_file = os.path.join(SCRIPT_DIR, 'cosif_process_1.log')
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='latin1')
logging.basicConfig(handlers=[handler], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## 2) User-defined functions:

# Filename pattern: YYYYMMSUFFIX.ZIP | YYYYMMSUFFIX.csv.zip | YYYYMMSUFFIX.csv | YYYYMMSUFFIX.CSV
FILE_PATTERN = re.compile(r'^(\d{6})([A-Za-z]+)\.(zip|ZIP|csv|CSV|csv\.zip|CSV\.ZIP)$')

# File types to SKIP (tipo_base, lowercase):
#   consorcios  - Consortium administrators; not deposit-taking institutions.
#                 Their COSIF files have a different schema and parse to 0 rows.
#   combinados  - Combined cooperative system balance sheets (docs 4413-4433).
#                 These aggregate cooperativas + central cooperative banks into
#                 a single combined view.  The SAME institutions already appear
#                 individually in COOPERATIVAS and BANCOS files, so including
#                 COMBINADOS would double-count deposits and expenses.
SKIP_TIPOS = {"consorcios", "combinados"}

def scan_local_files(input_dir):
    """
    Scans `input_dir` for COSIF files and returns a dict: {ano_mes: [(filepath, tipo_base), ...]}.
    Handles .ZIP, .csv.zip, .csv, and .CSV extensions (case-insensitive).
    """
    files_by_month = {}
    n_skipped = 0
    
    for fname in os.listdir(input_dir):
        match = FILE_PATTERN.match(fname)
        if not match:
            logging.warning(f"Skipping unrecognized file: {fname}")
            continue
            
        ano_mes = match.group(1)        # e.g. "201301"
        tipo_base = match.group(2).lower()  # e.g. "bancos"

        if tipo_base in SKIP_TIPOS:
            n_skipped += 1
            continue

        filepath = os.path.join(input_dir, fname)
        files_by_month.setdefault(ano_mes, []).append((filepath, tipo_base))
    
    n_files = sum(len(v) for v in files_by_month.values())
    n_months = len(files_by_month)
    logging.info(
        f"Found {n_files} files across {n_months} months in {input_dir} "
        f"(skipped {n_skipped} files with tipo_base in {SKIP_TIPOS})"
    )
    
    return files_by_month

def read_cosif_file(filepath):
    """
    Reads a single COSIF file (ZIP, csv.zip, or plain CSV) into a DataFrame.
    """
    fname = os.path.basename(filepath).upper()
    
    try: 
        # if .csv.zip or .zip, treat as ZIP archive
        if fname.endswith('.ZIP'):
            with zipfile.ZipFile(filepath, 'r') as zf:
                csv_names = [n for n in zf.namelist() if n.upper().endswith('.CSV')]
                if not csv_names:
                    logging.warning(f"No CSV found inside {filepath}")
                    return pd.DataFrame()
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f, sep = ';', encoding = 'latin-1', skiprows = 3, dtype = str, on_bad_lines='skip')
        else:
            df = pd.read_csv(filepath, sep = ';', encoding = 'latin-1', skiprows = 3, dtype = str, on_bad_lines='skip')
            
        return df
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

def processar_cosif_local(ano_mes, file_list):
    """
    Processes all locally-downloaded COSIF files for a given month (ano_mes = 'YYYYMM').
    `file_list` is a list of (filepath, tipo_base) tuples from scan_local_files.
    """
    df_consolidado = pd.DataFrame()
    tipos_processados = set()
    
    for filepath, tipo_base in file_list:
        
        if tipo_base in tipos_processados:
            logging.info(f"Already processed {tipo_base} for {ano_mes}. Skipping: {filepath}")
            continue
        
        tipos_processados.add(tipo_base)
        logging.info(f"Reading {tipo_base} for {ano_mes}: {os.path.basename(filepath)}")
        
        try:
            df_temp = read_cosif_file(filepath)
            df_temp.columns = [col.upper() for col in df_temp.columns]
            
            if 'SALDO' in df_temp.columns:
                df_temp['SALDO'] = pd.to_numeric(
                    df_temp['SALDO'].str.replace(',', '.'), errors='coerce'
                )
            
            df_temp['DATA_BASE'] = pd.to_datetime(ano_mes, format='%Y%m')
            
            if 'TAXONOMIA' in df_temp.columns:
                df_temp['TIPO_INSTITUICAO'] = df_temp['TAXONOMIA']
            else:
                df_temp['TIPO_INSTITUICAO'] = tipo_base
            
            df_consolidado = pd.concat([df_consolidado, df_temp], ignore_index=True)
            
        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")
    
    return df_consolidado

def disaccumulate_flows(df_panel, col_value='SALDO', col_date='DATA_BASE', col_id='CNPJ'):
    """
    Disaccumulates the semi-annual DRE flow from COSIF.

    COSIF DRE accounts are accumulated within each semester (Jan-Jun and
    Jul-Dec).  The marginal (monthly) flow is obtained by differencing
    within each (CNPJ, Semester) group.

    For the **first observation** in each group -- which is month 1 or 7
    for monthly reporters, but may be month 3 or 9 for quarterly
    reporters -- the raw SALDO already equals the accumulated flow from
    the semester start, so it IS the correct marginal flow.
    """
    if df_panel.empty:
        return df_panel
    df_panel = df_panel.sort_values(by=[col_id, col_date]).copy()
    df_panel['Semester'] = (
        df_panel[col_date].dt.year.astype(str)
        + "_"
        + (df_panel[col_date].dt.month > 6).astype(int).astype(str)
    )
    df_panel['FLUXO_MARGINAL'] = (
        df_panel.groupby([col_id, 'Semester'])[col_value].diff()
    )
    # For the first observation in each (CNPJ, Semester) group, diff()
    # gives NaN.  The raw SALDO is the correct marginal flow no matter
    # which calendar month it falls in (handles monthly, quarterly, and
    # irregular reporters alike).
    is_first_in_group = (
        df_panel.groupby([col_id, 'Semester']).cumcount() == 0
    )
    df_panel.loc[is_first_in_group, 'FLUXO_MARGINAL'] = (
        df_panel.loc[is_first_in_group, col_value]
    )
    return df_panel

# ---------- COSIF rubric code mappings (public data) ----------
#
# Balance-sheet deposit stocks: the public COSIF has type-level sub-accounts
# (4.1.1 demand, 4.1.2 savings, 4.1.3 interbank, 4.1.5 time, 4.1.9 other)
# but the expense side (8.1.1 "Despesas de Captação") is published ONLY as
# a single aggregate -- NO sub-accounts by deposit type.  Therefore we can
# only compute a blended implicit rate.
#
# Exception: from the 2025 COSIF recode (BCB 10-digit account codes), two
# new accounts encode prepaid account activity for Payment Institutions:
#   8119800007 = DESPESAS DE REMUNERAÇÃO DE CONTA DE PAGAMENTO PRÉ-PAGA
#   4193000009 = CONTA DE PAGAMENTO PRÉ-PAGA (liability stock)
# These are zero for all institutions in the pre-2025 8-digit format.
#
COSIF_TOTAL_DEPOSIT_PREFIX   = '4100000'  # 4.1.0 DEPÓSITOS (total stock)
COSIF_FUNDING_EXPENSE_PREFIX = '8110000'  # 8.1.1 Despesas de Captação (aggregate)
COSIF_PREPAID_EXPENSE_PREFIX = '8119800'  # 8.1.1.9.8 Remuneração Conta Pag. Pré-Paga (2025+)
COSIF_PREPAID_STOCK_PREFIX   = '41930'    # 4.1.9.3   Conta de Pagamento Pré-Paga (2025+)

def calcular_custo_implicito_painel(df_panel_cosif):
    """
    Calculates effective funding costs using a disaccumulated panel of COSIF data.

    Uses correct public-COSIF rubric codes:
      Stock:   4.1.0  total deposits (the only stock we need)
      Expense: 8.1.1  aggregate "Despesas de Captação" (sub-codes by deposit
               type do NOT exist in the publicly available COSIF files for
               pre-2025 data).

    Additionally, from the 2025 BCB recode onward, extracts:
      Stock:   4.1.9.3  (41930xxx) Conta de Pagamento Pré-Paga
      Expense: 8.1.1.9.8 (8119800x) Remuneração Conta Pagamento Pré-Paga

    Returns a CNPJ x DATA_BASE panel with:
      - Estoque_Total            (total deposits, end-of-month stock)
      - Estoque_Total_Lag        (lagged one month)
      - Despesa_Captacao_Marginal (disaccumulated monthly blended funding expense)
      - Custo_Efetivo_Blended    (|expense| / lagged total deposits, in %)
      - Estoque_Prepago          (prepaid account balance; NaN pre-2025)
      - Estoque_Prepago_Lag      (lagged one month)
      - Desp_Prepago_Marginal    (disaccumulated monthly prepaid expense; 0 pre-2025)
      - Custo_Efetivo_Prepago    (|prepaid expense| / lagged prepaid stock, in %)
    """
    if df_panel_cosif.empty:
        return pd.DataFrame()

    df = df_panel_cosif.copy()
    df['CONTA_LIMPA'] = (
        df['CONTA'].astype(str).str.replace(r'[\.\-]', '', regex=True)
    )

    # --- A) Extract total deposit STOCK (4.1.0) ---
    df_stocks = (
        df[df['CONTA_LIMPA'].str.startswith(COSIF_TOTAL_DEPOSIT_PREFIX)]
        .groupby(['CNPJ', 'DATA_BASE'])['SALDO']
        .sum()
        .reset_index()
        .rename(columns={'SALDO': 'Estoque_Total'})
        .sort_values(['CNPJ', 'DATA_BASE'])
    )
    df_stocks['Estoque_Total_Lag'] = (
        df_stocks.groupby('CNPJ')['Estoque_Total'].shift(1)
    )

    # --- A') Extract prepaid account STOCK (4.1.9.3, 2025+ only) ---
    df_prepaid_stock = (
        df[df['CONTA_LIMPA'].str.startswith(COSIF_PREPAID_STOCK_PREFIX)]
        .groupby(['CNPJ', 'DATA_BASE'])['SALDO']
        .sum()
        .reset_index()
        .rename(columns={'SALDO': 'Estoque_Prepago'})
        .sort_values(['CNPJ', 'DATA_BASE'])
    )
    if not df_prepaid_stock.empty:
        df_prepaid_stock['Estoque_Prepago_Lag'] = (
            df_prepaid_stock.groupby('CNPJ')['Estoque_Prepago'].shift(1)
        )

    # --- B) Extract and disaccumulate funding EXPENSE (8.1.x) ---
    df_fluxo = df[df['CONTA_LIMPA'].str.startswith('81')].copy()
    df_fluxo = disaccumulate_flows(df_fluxo)

    if not df_fluxo.empty:
        desp_captacao = (
            df_fluxo[df_fluxo['CONTA_LIMPA'].str.startswith(COSIF_FUNDING_EXPENSE_PREFIX)]
            .groupby(['CNPJ', 'DATA_BASE'])['FLUXO_MARGINAL']
            .sum()
            .reset_index()
            .rename(columns={'FLUXO_MARGINAL': 'Despesa_Captacao_Marginal'})
        )
        # --- B') Extract prepaid remuneration EXPENSE (8.1.1.9.8, 2025+ only) ---
        desp_prepago = (
            df_fluxo[df_fluxo['CONTA_LIMPA'].str.startswith(COSIF_PREPAID_EXPENSE_PREFIX)]
            .groupby(['CNPJ', 'DATA_BASE'])['FLUXO_MARGINAL']
            .sum()
            .reset_index()
            .rename(columns={'FLUXO_MARGINAL': 'Desp_Prepago_Marginal'})
        )
    else:
        desp_captacao = pd.DataFrame(
            columns=['CNPJ', 'DATA_BASE', 'Despesa_Captacao_Marginal']
        )
        desp_prepago = pd.DataFrame(
            columns=['CNPJ', 'DATA_BASE', 'Desp_Prepago_Marginal']
        )

    # --- C) Combine total stock + blended expense ---
    df_custos = df_stocks.merge(
        desp_captacao, on=['CNPJ', 'DATA_BASE'], how='outer'
    ).fillna({'Despesa_Captacao_Marginal': 0, 'Estoque_Total': 0})

    # --- C') Left-join prepaid stock and expense (zero / NaN for pre-2025) ---
    if not df_prepaid_stock.empty:
        df_custos = df_custos.merge(
            df_prepaid_stock, on=['CNPJ', 'DATA_BASE'], how='left'
        )
    else:
        df_custos['Estoque_Prepago'] = np.nan
        df_custos['Estoque_Prepago_Lag'] = np.nan

    if not desp_prepago.empty:
        df_custos = df_custos.merge(
            desp_prepago, on=['CNPJ', 'DATA_BASE'], how='left'
        )
        df_custos['Desp_Prepago_Marginal'] = (
            df_custos['Desp_Prepago_Marginal'].fillna(0.0)
        )
    else:
        df_custos['Desp_Prepago_Marginal'] = 0.0

    # --- D) Blended implicit funding rate ---
    #   = |marginal expense| / lagged total deposits
    df_custos['Custo_Efetivo_Blended'] = (
        df_custos['Despesa_Captacao_Marginal'].abs()
        / df_custos['Estoque_Total_Lag']
    )
    df_custos.loc[
        ~np.isfinite(df_custos['Custo_Efetivo_Blended']),
        'Custo_Efetivo_Blended',
    ] = np.nan
    # Save as percentage for precision (e.g. 1.5 means 1.5% per month)
    df_custos['Custo_Efetivo_Blended'] *= 100

    # --- D') Prepaid implicit rate ---
    #   = |prepaid expense| / lagged prepaid account stock (percentage, monthly)
    #   NaN for all observations where prepaid stock was absent (pre-2025).
    df_custos['Custo_Efetivo_Prepago'] = (
        df_custos['Desp_Prepago_Marginal'].abs()
        / df_custos['Estoque_Prepago_Lag']
    )
    df_custos.loc[
        ~np.isfinite(df_custos['Custo_Efetivo_Prepago']),
        'Custo_Efetivo_Prepago',
    ] = np.nan
    df_custos['Custo_Efetivo_Prepago'] *= 100

    return df_custos

def _read_one_file(args):
    """Read a single COSIF file and return a small DataFrame with only
    the columns we need (CNPJ, CONTA, SALDO, DATA_BASE, TIPO_INSTITUICAO).
    Runs inside a worker process, so must be a module-level function."""
    filepath, tipo_base, ano_mes = args
    try:
        df = read_cosif_file(filepath)
        if df.empty:
            return df
        df.columns = [c.upper() for c in df.columns]
        if 'SALDO' in df.columns:
            df['SALDO'] = pd.to_numeric(
                df['SALDO'].str.replace(',', '.'), errors='coerce'
            )
        df['DATA_BASE'] = pd.to_datetime(ano_mes, format='%Y%m')
        if 'TAXONOMIA' in df.columns:
            df['TIPO_INSTITUICAO'] = df['TAXONOMIA']
        else:
            df['TIPO_INSTITUICAO'] = tipo_base
        # Keep only the columns we need -- drop the rest early to save memory
        keep = {'CNPJ', 'CONTA', 'SALDO', 'DATA_BASE', 'TIPO_INSTITUICAO'}
        df = df[[c for c in df.columns if c in keep]]
        return df
    except Exception as e:
        return pd.DataFrame()

def _process_taxonomy(tipo_inst, df_tipo, out_dir):
    """Process one taxonomy group end-to-end and save CSV.
    Returns (tipo_inst, n_rows, output_path | None)."""
    if df_tipo.empty:
        return (tipo_inst, 0, None)
    safe = re.sub(r'[\\/*?:"<>|]', "", str(tipo_inst)).replace(" ", "_")
    df_custos = calcular_custo_implicito_painel(df_tipo)
    if df_custos.empty:
        return (tipo_inst, 0, None)
    out = os.path.join(out_dir, f"custos_implicitos_{safe}.csv")
    df_custos.to_csv(out, index=False)
    return (tipo_inst, len(df_custos), out)

## 3) Execution Routine:
if __name__ == "__main__":
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t0 = time.perf_counter()

    # 3.1) Scan local COSIF files:
    logging.info("Starting COSIF Processing (local files)")
    print("Starting COSIF Processing ...")

    if not os.path.isdir(INPUT_COSIF_DIR):
        logging.error(f"Input directory does not exist: {INPUT_COSIF_DIR}")
        sys.exit(1)

    files_by_month = scan_local_files(INPUT_COSIF_DIR)
    if not files_by_month:
        logging.error("No valid COSIF files found. Exiting.")
        sys.exit(1)

    meses_alvo = sorted(files_by_month.keys())
    logging.info(f"Months to process: {meses_alvo[0]} to {meses_alvo[-1]} ({len(meses_alvo)} months)")

    # 3.2) Read all files in parallel (I/O-bound -> ThreadPool) -----------
    # Build a flat list of (filepath, tipo_base, ano_mes) tasks,
    # deduplicating by (ano_mes, tipo_base).
    seen = set()
    read_tasks = []
    for mes in meses_alvo:
        for filepath, tipo_base in files_by_month[mes]:
            key = (mes, tipo_base)
            if key in seen:
                continue
            seen.add(key)
            read_tasks.append((filepath, tipo_base, mes))

    n_workers = min(8, len(read_tasks))
    print(f"Reading {len(read_tasks)} files with {n_workers} threads ...")
    logging.info(f"Reading {len(read_tasks)} files with {n_workers} threads")

    frames = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_read_one_file, t): t for t in read_tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)
            if done % 100 == 0:
                print(f"  ... {done}/{len(read_tasks)} files read")

    if not frames:
        logging.error("All files returned empty DataFrames.")
        sys.exit(1)

    painel_cosif = pd.concat(frames, ignore_index=True)
    del frames  # free memory
    t1 = time.perf_counter()
    print(f"Read complete: {len(painel_cosif):,} rows in {t1-t0:.1f}s")
    logging.info(f"Read complete: {len(painel_cosif):,} rows in {t1-t0:.1f}s")

    # 3.3) Process each taxonomy sequentially --------------------------------
    # (Avoids pickling large DataFrames across process boundaries.)
    tipos_encontrados = sorted(painel_cosif['TIPO_INSTITUICAO'].dropna().unique())
    print(f"Processing {len(tipos_encontrados)} taxonomy groups ...")
    logging.info(f"Processing {len(tipos_encontrados)} taxonomy groups")

    for i, tipo in enumerate(tipos_encontrados, 1):
        df_tipo = painel_cosif[painel_cosif['TIPO_INSTITUICAO'] == tipo].copy()
        tipo_label, n_rows, fpath = _process_taxonomy(tipo, df_tipo, output_path)
        if fpath:
            logging.info(f"Saved {tipo_label}: {n_rows:,} rows -> {fpath}")
            print(f"  [{i}/{len(tipos_encontrados)}] {tipo_label}: {n_rows:,} rows")
        else:
            logging.warning(f"Empty result for {tipo_label}")
    del painel_cosif

    t2 = time.perf_counter()
    print(f"Done. Total time: {t2-t0:.1f}s (read {t1-t0:.1f}s + process {t2-t1:.1f}s)")
    logging.info(f"Script completed in {t2-t0:.1f}s")