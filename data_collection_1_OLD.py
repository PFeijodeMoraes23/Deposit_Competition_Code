## data_collection_1.py
# Author: Pedro Feijo de Moraes
# Last edited: 2026-02-26
# Objective: Collect data on deposits and deposit rates for Brazil
## ------------------------------------------------------------------------------------------

## 1) Load necessary packages and define paths:

# Packages
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logging
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
import re

# Directory Mapping
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BCB_PATH = os.path.join(PARENT_DIR, "BCB")
main_path = os.path.join(BCB_PATH,"Egan_et_al_2025_Rep")
RAW_DATA_PATH = os.path.join(main_path,"raw")
PROCESSED_DATA_PATH = os.path.join(main_path,"processed")

output_if_prudential = os.path.join(RAW_DATA_PATH, "IF_DATA_RAW", "Prudential")
output_if_financial = os.path.join(RAW_DATA_PATH, "IF_DATA_RAW", "Financial")
output_if_individual = os.path.join(RAW_DATA_PATH, "IF_DATA_RAW", "Individual")
output_macro_raw = os.path.join(RAW_DATA_PATH, "SGS_RAW")

directories = [RAW_DATA_PATH, PROCESSED_DATA_PATH, output_if_financial, output_if_prudential, output_if_individual, output_macro_raw]
for path in directories:
    os.makedirs(path, exist_ok=True)
    
# Logging -- write to file AND console so errors are visible
log_file = os.path.join(SCRIPT_DIR,'data_collection_1.log')
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)   # only warnings/errors to console
log_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_fmt)
console_handler.setFormatter(log_fmt)
logging.basicConfig(handlers=[file_handler, console_handler], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
# Event flag for thread control
stop_event = threading.Event()

# Persistent HTTP session with connection pooling & automatic retries
session = requests.Session()
_retry_strategy = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[502, 503, 504],
    allowed_methods=["GET"],
)
_adapter = HTTPAdapter(
    max_retries=_retry_strategy,
    pool_connections=15,
    pool_maxsize=15,
)
session.mount("https://", _adapter)
session.mount("http://", _adapter)

# Concurrency settings
MAX_WORKERS_REPORTS = 4       # parallel report downloads per quarter
MAX_WORKERS_TYPES   = 3       # parallel institution types per report
INTER_REQUEST_SLEEP = 2       # seconds between individual API calls

## 2) User-defined functions:

# 2.1) Functions for General Series Extraction (SGS - BCB):
def extrair_sgs(codigo_sgs, data_inicio, data_fim, retries = 5):
    """
    Extracts time series from the BCB's Time Series Management System (SGS).
    Uses the persistent session for connection reuse.
    """
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_sgs}/dados"
    params = {
        "formato": "json",
        "dataInicial": data_inicio,
        "dataFinal": data_fim
    }
    
    for i in range(retries):
        try:
            response = session.get(url, params=params, timeout=120)
            response.raise_for_status()
    
            df = pd.DataFrame(response.json())
            df['data'] = pd.to_datetime(df['data'], format = '%d/%m/%Y')
            df['valor'] = df['valor'].astype(float)
            df.set_index('data', inplace = True)
            
            return df
        
        except requests.exceptions.HTTPError as err:
            logging.error(f"HTTP error extracting SGS {codigo_sgs}: {err}")
            if response.status_code in [500,502,503,504]:
                if i < retries - 1:
                    wait_time = 5*(2**i)
                    logging.info(f"Server error. Retrying SGS {codigo_sgs} in {wait_time} seconds.")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to extract SGS {codigo_sgs} after {retries} attempts.")
                    return pd.DataFrame()
            else:
                logging.error(f"Client error for SGS {codigo_sgs}. Aborting.")
                return pd.DataFrame()
        
        except requests.exceptions.RequestException as err:
            logging.error(f"Connection error extracting SGS {codigo_sgs}: {err}")
            if i < retries - 1:
                wait_time = 5*(2**i)
                logging.info(f"Connection error. Retrying SGS {codigo_sgs} in {wait_time} seconds.")
                time.sleep(wait_time)
            else:
                return pd.DataFrame()

def obter_dados_macro(start_year: int, end_year: int):
    """
    Download SGS macro series in yearly chunks to avoid HTTP 406
    (BCB API rejects date ranges that are too large).
    """
    series = {
        'Selic_Over': 11, # % a.d.
        'Meta_Selic': 432, # % a.a.
        'CDI_Anualizado': 4389, # % a.a.
        'TR_Diaria': 226 # % a.m.
    }
    
    all_frames = []
    for yr in range(start_year, end_year + 1):
        data_inicio = f"01/01/{yr}"
        data_fim    = f"31/12/{yr}"
        df_year = pd.DataFrame()
        for nome, codigo in series.items():
            df_temp = extrair_sgs(codigo, data_inicio, data_fim)
            if df_temp.empty:
                continue
            df_temp.rename(columns={'valor': nome}, inplace=True)
            if df_year.empty:
                df_year = df_temp
            else:
                df_year = df_year.join(df_temp, how='outer')
        if not df_year.empty:
            all_frames.append(df_year)
            print(f"  SGS {yr}: {len(df_year)} daily obs")
        else:
            logging.warning(f"SGS: no data for {yr}")

    if all_frames:
        return pd.concat(all_frames).sort_index()
    return pd.DataFrame()

# 2.2) IF-Data extraction via Olinda API (IF-DATA - BCB):
base_url_values = "https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata/IfDataValores(AnoMes=@AnoMes,TipoInstituicao=@TipoInstituicao,Relatorio=@Relatorio)?@AnoMes={year_quarter}&@TipoInstituicao={tipo}&@Relatorio='{relatorio}'&$top=100000000&$format=text/csv&$select=CodInst,AnoMes,NomeRelatorio,NumeroRelatorio,Grupo,Conta,NomeColuna,Saldo"

def download_values(year, quarter, tipo, relatorio_num, relatorio_nome, retries_number):
    """
    Download institutional values data for dynamic Relatorio.
    Skips download if the output file already exists and is non-empty.
    Uses the persistent session for connection pooling.

    Report availability by institution type (from BCB IF.data):
      - Reports 1-4 (Resumo,Ativo,Passivo,DRE): All types, from 2014+
      - Report 5  (Informacoes de Capital):      Type 1 from 2015; Types 2,3 vary
      - Report 6  (Segmentacao):                 Type 1 only, from 2017
      - Reports 7-14 (Credit portfolio):         Mostly Type 2; limited Type 1/3
      - Report 15 (Cambio):                      Type 4 only (not downloaded here)
    """
    global stop_event
    if stop_event.is_set(): return
    
    year_quarter = f"{year}{str(quarter).zfill(2)}"
    
    if tipo == '1': output_dir = output_if_prudential
    elif tipo == '2': output_dir = output_if_financial
    else: output_dir = output_if_individual
    
    safe_relatorio_nome = re.sub(r'[\\/*?:"<>|]', "", relatorio_nome).replace(" ", "_")
    
    file_path = os.path.join(output_dir, f"IF_DATA_{safe_relatorio_nome}_{year}_{quarter}.csv")
    
    # Skip if file already exists and is non-empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        logging.info(f"Local file found. Skipping download for {relatorio_nome} ({year}-{quarter}, Type {tipo}).")
        return
        
    url = base_url_values.format(year_quarter=year_quarter, tipo=tipo, relatorio=relatorio_num)
    logging.info(f"Requesting URL: {url}")
    
    for i in range(retries_number):
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            
            if not response.content:
                logging.warning(f"No data received for {year}-{quarter} (Type {tipo}, Relatorio {relatorio_num}). Retrying.")
                time.sleep(5*(2**i))
                continue
            
            with open(file_path, "wb") as file:
                file.write(response.content)
                
            df = pd.read_csv(file_path, sep=',', encoding='utf-8', dtype=str)
            if df.empty or len(df) == 0:
                # API returned 200 with header-only CSV -- this report/type/quarter
                # combo has no data. This is NOT a transient error; don't retry.
                logging.info(
                    f"No data available for {relatorio_nome} "
                    f"({year}-{quarter}, Type {tipo}). Skipping."
                )
                try: os.remove(file_path)
                except OSError: pass
                break   # <-- break, not continue
            
            logging.info(f"Downloaded {relatorio_nome} data for {year}-{quarter} (Type {tipo}).")
            time.sleep(INTER_REQUEST_SLEEP)  # polite throttle after success
            break
        
        except requests.exceptions.HTTPError as err:
            if response.status_code == 500:
                logging.error(f"Server error 500 for {relatorio_nome} ({year}-{quarter}, Type {tipo}). Backing off 60s.")
                time.sleep(60)
                # Don't freeze the entire pipeline -- just retry this item
                continue
            logging.error(f"HTTP {response.status_code} for {relatorio_nome} ({year}-{quarter}, Type {tipo}): {err}")
            break
        except requests.exceptions.RequestException as err:
            logging.error(f"Request error: {err}")
            if i < retries_number - 1:
                time.sleep(5*(2**i))
            else:
                break
            
def download_all_values_for_report(year, quarter, relatorio_num, relatorio_nome, retries_number):
    """
    Download values for a specific report across all institution types in parallel.
    Uses MAX_WORKERS_TYPES threads (default 3 -- one per institution type).
    """
    global stop_event
    if stop_event.is_set():
        return
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_TYPES) as executor:
        futures = {
            executor.submit(download_values, year, quarter, tipo, relatorio_num, relatorio_nome, retries_number): tipo
            for tipo in ['1','2','3']
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error parallel-downloading values (type {futures[future]}): {e}")

def download_quarter(year, quarter, relatorios_alvo, retries_number):
    """
    Download all reports for a given quarter in parallel.
    Parallelizes across reports (MAX_WORKERS_REPORTS), and within each report
    parallelizes across institution types (MAX_WORKERS_TYPES).
    """
    global stop_event
    if stop_event.is_set():
        return
    
    logging.info(f"--- Starting quarter {year}-{quarter} ({len(relatorios_alvo)} reports) ---")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_REPORTS) as executor:
        futures = {}
        for rel_dict in relatorios_alvo:
            num_rel = str(rel_dict['NumeroRelatorio'])
            nome_rel = rel_dict['NomeRelatorio']
            f = executor.submit(
                download_all_values_for_report,
                year, quarter, num_rel, nome_rel, retries_number
            )
            futures[f] = f"{num_rel} ({nome_rel})"
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error downloading report {futures[future]} for {year}-{quarter}: {e}")
    
    logging.info(f"--- Finished quarter {year}-{quarter} ---")
    
## 3) Main Execution:
if __name__ == "__main__":
    # Settings:
    start_year = 2013       # SGS macro series start
    if_start_year = 2014    # IF-Data Prudential available from 2014Q1 (BCB IF.data)
    end_year = 2025
    quarters = [3,6,9,12]
    retries_number = 3
    
    # 3.1) Extract Macro Series:
    logging.info("Starting Macro Series Extraction.")
    print("Downloading SGS macro series (yearly chunks)...")
    df_macro = obter_dados_macro(start_year, 2025)
    macro_path = os.path.join(output_macro_raw, "macro_series_full.csv")
    if df_macro.empty:
        logging.error("All SGS downloads failed -- macro_series_full.csv NOT saved.")
        print("ERROR: SGS macro download failed. Check log for details.")
    else:
        df_macro.to_csv(macro_path)
        logging.info(f"Macro series saved to {macro_path} ({len(df_macro)} rows)")
        print(f"Macro series: {len(df_macro)} daily observations saved.")
    
    # 3.2) Extract IF-Data Via Olinda
    reports_file_path = os.path.join(BCB_PATH, "IF Data", "Reports_List.csv")
    try:
        # Added Latin1 encoding to handle Portuguese characters
        df_reports = pd.read_csv(reports_file_path, encoding='latin1')
        df_reports = df_reports.iloc[:-1]
        relatorios_alvo = df_reports.to_dict('records')
        logging.info(f"Loaded {len(relatorios_alvo)} reports to process from {reports_file_path}.")
    except Exception as e:
        logging.error(f"Failed to read Reports_List.csv from {reports_file_path}: {e}")
        relatorios_alvo = []

    logging.info("Starting IF-Data Extraction")
    if relatorios_alvo:
        total_quarters = (end_year - if_start_year + 1) * len(quarters)
        done = 0
        for year in range(if_start_year, end_year + 1):
            for quarter in quarters:
                done += 1
                print(f"  Quarter {done}/{total_quarters}: {year}-Q{quarter // 3}")
                download_quarter(year, quarter, relatorios_alvo, retries_number)
    else:
        logging.warning("No reports to download in IF-DATA section. Skipping.")
        
    logging.info("Script execution completed.")