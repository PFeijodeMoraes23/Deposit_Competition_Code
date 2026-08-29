import sys
import os
import glob
import zipfile
import logging
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Paths ---
from utils import paths
_ROOT = Path(__file__).resolve().parents[2]
# Raw COSIF zips resolve via the shared dataset path (utils.paths), matching
# scrape_15 and panel_3; the old BCB/.../raw/COSIF_RAW location is empty.
RAW_DIR = Path(paths.COSIF_RAW)
OUT_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "COSIF_PROCESSED"
OUT_CSV = OUT_DIR / "ip_rates_quarterly.csv"

def process_file(fpath):
    """ Extract IP rows and summarize accounts. """
    records = []
    
    with zipfile.ZipFile(fpath) as zf:
        # Some zips might have multiple files, but usually one. Take the first.
        fname = zf.namelist()[0]
        with zf.open(fname) as f:
            # We skip 3 rows (standard BCB header format)
            try:
                df = pd.read_csv(f, sep=';', encoding='latin1', dtype=str, skiprows=3)
            except Exception as e:
                # Some files might not have 3 header rows, try without skiprows
                f.seek(0)
                try:
                    df = pd.read_csv(f, sep=';', encoding='latin1', dtype=str)
                except Exception as e2:
                    logging.warning(f"Failed to read {fpath}: {e2}")
                    return pd.DataFrame()
            
    # Normalize columns
    df.columns = [c.replace('#', '').strip() for c in df.columns]

    # Keep only the monthly balancete (DOCUMENTO 4010).  At June/December BCB
    # also ships the semester balanco patrimonial (4016), which duplicates
    # every group-4 balance and would double the deposit stock at those two
    # months.  4016 carries no group-8 rows, so the expense is unaffected.
    if 'DOCUMENTO' in df.columns:
        df = df[df['DOCUMENTO'].astype(str).str.strip() == '4010']

    if 'TAXONOMIA' not in df.columns:
        return pd.DataFrame()
        
    df_ip = df[df['TAXONOMIA'].str.contains('PAGAMENTO', case=False, na=False)].copy()
    if df_ip.empty:
        return pd.DataFrame()
        
    # Clean data
    df_ip['CNPJ'] = df_ip['CNPJ'].str.strip().str.zfill(8)
    df_ip['DATA_BASE'] = pd.to_numeric(df_ip['DATA_BASE'], errors='coerce')
    
    # Fix Brazilian formatting for SALDO (280436787,46 -> 280436787.46)
    df_ip['SALDO'] = df_ip['SALDO'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df_ip['SALDO'] = pd.to_numeric(df_ip['SALDO'], errors='coerce').fillna(0.0)
    
    # Extract Stock vs Expense
    # Stocks: 419 (Outros Depósitos), 499 (Diversas), 411 (Depósitos à Vista)
    # Expenses: 811 (Despesas de Captação)
    
    df_ip['is_stock'] = df_ip['CONTA'].str.startswith('4', na=False)
    df_ip['is_expense'] = df_ip['CONTA'].str.startswith('811', na=False)
    
    # Group by CNPJ, DATA_BASE
    # To reduce noise, we only sum the specific sub-accounts that matter
    stock_mask = df_ip['CONTA'].str.startswith(('419', '499', '411', '41900004'), na=False)
    expense_mask = df_ip['CONTA'].str.startswith('811', na=False)
    
    # Filter to only relevant accounts before grouping to avoid excessive sums of irrelevant liabilities
    df_relevant = df_ip[stock_mask | expense_mask].copy()
    if df_relevant.empty:
        return pd.DataFrame()
        
    df_relevant['Account_Type'] = np.where(df_relevant['CONTA'].str.startswith('811', na=False), 'Expense', 'Stock')
    
    agg = df_relevant.groupby(['CNPJ', 'DATA_BASE', 'Account_Type'])['SALDO'].sum().reset_index()
    
    # Pivot
    piv = agg.pivot(index=['CNPJ', 'DATA_BASE'], columns='Account_Type', values='SALDO').reset_index()
    
    # Ensure columns exist
    if 'Stock' not in piv.columns: piv['Stock'] = 0.0
    if 'Expense' not in piv.columns: piv['Expense'] = 0.0
    
    # Fill NAs
    piv['Stock'] = piv['Stock'].fillna(0.0)
    piv['Expense'] = piv['Expense'].fillna(0.0)

    # In COSIF, Expenses are often reported as negative numbers since they are "Despesas" (-)
    piv['Expense'] = piv['Expense'].abs()

    return piv

def main():
    logging.info("Starting IP Rate extraction from COSIF...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # We mainly expect IP data in SOCIEDADES and BLOPRUDENCIAL
    zip_files = []
    for pattern in ["*SOCIEDADES*.zip", "*SOCIEDADES*.ZIP", "*BLOPRUDENCIAL*.zip", "*BLOPRUDENCIAL*.ZIP"]:
        zip_files.extend(glob.glob(str(RAW_DIR / pattern)))
        
    zip_files = sorted(list(set(zip_files)))
    logging.info(f"Found {len(zip_files)} target ZIP files.")
    
    all_ip_data = []
    
    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor, as_completed

    max_workers = max(1, multiprocessing.cpu_count() - 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in zip_files}
        for i, future in enumerate(as_completed(futures), 1):
            if i % 100 == 0:
                logging.info(f"Processed {i}/{len(zip_files)} files...")
            try:
                res = future.result()
                if not res.empty:
                    all_ip_data.append(res)
            except Exception as e:
                logging.error(f"Error processing {futures[future]}: {e}")
                
    if not all_ip_data:
        logging.warning("No IP data found.")
        sys.exit(0)
        
    df = pd.concat(all_ip_data, ignore_index=True)
    
    # Deduplicate in case overlapping reports exist
    df = df.sort_values(by=['CNPJ', 'DATA_BASE', 'Stock'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=['CNPJ', 'DATA_BASE'], keep='first')
    
    logging.info("Building quarterly aggregates...")
    df['year'] = df['DATA_BASE'] // 100
    df['month'] = df['DATA_BASE'] % 100
    df['q_month'] = ((df['month'] - 1) // 3 + 1) * 3
    df['AnoMes'] = df['year'] * 100 + df['q_month']

    # COSIF group-8 expense accounts (811) accumulate within the SEMESTER and
    # reset in January and July.  The raw monthly SALDO is therefore a
    # semester-cumulative balance, not a monthly flow; summing it within a
    # quarter manufactures a see-saw.  Disaccumulate to a true monthly flow
    # within each (CNPJ, year, semester) run before the quarterly sum.
    df = df.sort_values(['CNPJ', 'year', 'month'])
    df['_sem'] = np.where(df['month'] <= 6, 1, 2)
    df['Expense'] = (
        df.groupby(['CNPJ', 'year', '_sem'])['Expense']
        .transform(lambda s: s.diff().where(s.shift(1).notna(), s))
    )
    df = df.drop(columns=['_sem'])

    # Stock is End of Quarter (4016 double-count already removed in process_file)
    df_q_stock = df[df['month'].isin([3, 6, 9, 12])][['CNPJ', 'AnoMes', 'Stock']].rename(columns={'Stock': 'Estoque_Total'})

    # Expense is Sum of Quarter (now a sum of true monthly flows)
    df_q_exp = df.groupby(['CNPJ', 'AnoMes'])['Expense'].sum().reset_index().rename(columns={'Expense': 'Despesa_Captacao'})
    
    ip_q = pd.merge(df_q_stock, df_q_exp, on=['CNPJ', 'AnoMes'], how='outer').fillna(0.0)
    
    # Sort and Lag
    ip_q = ip_q.sort_values(['CNPJ', 'AnoMes'])
    ip_q['Estoque_Total_Lag'] = ip_q.groupby('CNPJ')['Estoque_Total'].shift(1)
    
    # Compute Implicit Rate
    ip_q['ip_implicit_rate'] = ip_q['Despesa_Captacao'] / ip_q['Estoque_Total_Lag'].replace(0.0, np.nan)
    ip_q.loc[~np.isfinite(ip_q['ip_implicit_rate']), 'ip_implicit_rate'] = np.nan
    ip_q.loc[ip_q['ip_implicit_rate'] == 0.0, 'ip_implicit_rate'] = np.nan # Missing expense = NaN (not 0% cost)
    ip_q.loc[ip_q['ip_implicit_rate'] > 0.5, 'ip_implicit_rate'] = np.nan # Hard cap 50% quarterly
    
    logging.info(f"Final Quarterly Panel: {len(ip_q)} rows, {ip_q['ip_implicit_rate'].notna().sum()} explicit rate obs.")
    
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    pa_csv.write_csv(pa.Table.from_pandas(ip_q, preserve_index=False), OUT_CSV)
    logging.info(f"Saved -> {OUT_CSV}")

if __name__ == "__main__":
    main()
