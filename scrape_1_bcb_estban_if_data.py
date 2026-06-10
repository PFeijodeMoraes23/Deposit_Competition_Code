## if_data_scrape_1.py
# Authors: Pedro Feijó de Moraes
#
# Last edited on 2026_03_06
#
# Purpose: this script fetches data from BCB IF Data API (Olinda endpoint)
#          and from BCB's ESTBAN static file server (COSIF).
#
# ESTBAN files are hosted by BCB at (primary URL pattern):
#   https://www4.bcb.gov.br/fis/cosif/estban/{YYYYMM}_ESTBAN.csv      (municipality level)
#   https://www4.bcb.gov.br/fis/cosif/estban/{YYYYMM}_ESTBAN_AG.csv   (municipality + agency level)
# If the automated download fails, files can be manually downloaded from:
#   https://www.bcb.gov.br/estatisticas/estabilidadefinanceira/estban
#
# Attention: before running this script, make sure to install the required packages.
###--------------------------------------------------------------------------------------------------------------------

## 1) Load packages
import os
import io
import zipfile
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import requests
import time
import logging
import random
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

from utils import paths
from utils import refresh

## 2) Set up logging
log_file = 'if_data_scrape.log'
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3) # 5MB max size, 3 backups
logging.basicConfig(handlers=[handler], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## 3) Set API URLs and output directories

# --- IF Data (Olinda API) ---
base_url_list = "https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata/IfDataCadastro(AnoMes=@AnoMes)?@AnoMes={year_month}&$top=10000&$format=text/csv&$select=CodInst,Data,NomeInstituicao,DataInicioAtividade,Tcb,Td,Tc,SegmentoTb,Atividade,Uf,Municipio,Sr,CodConglomeradoFinanceiro,CodConglomeradoPrudencial,CnpjInstituicaoLider,Situacao"
base_url_values = "https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata/IfDataValores(AnoMes=@AnoMes,TipoInstituicao=@TipoInstituicao,Relatorio=@Relatorio)?@AnoMes={year_quarter}&@TipoInstituicao={tipo}&@Relatorio='T'&$top=100000000&$format=text/csv&$select=CodInst,AnoMes,NomeRelatorio,NumeroRelatorio,Grupo,Conta,NomeColuna,Saldo"

# IF Data output directories (canonical locations in utils/paths.py)
output_dir = str(paths.IF_DATA_LIST)
os.makedirs(output_dir, exist_ok=True)

output_type_1 = str(paths.IF_DATA_PRUDENTIAL)
os.makedirs(output_type_1, exist_ok=True)
output_type_3 = str(paths.IF_DATA_INDIVIDUAL)
os.makedirs(output_type_3, exist_ok=True)

# --- ESTBAN (BCB new content server) ---
# BCB migrated ESTBAN files to a new URL structure in late 2024.
# New page:  https://www.bcb.gov.br/estatisticas/estatisticabancariamunicipios
# New files: https://www.bcb.gov.br/content/estatisticas/estatistica_bancaria_estban/{pasta}/{Nome}
# Files are delivered as ZIP archives.  Naming convention:
#   2023-02+:  {YYYYMM}_ESTBAN.csv.zip  /  {YYYYMM}_ESTBAN_AG.csv.zip
#   pre-2023:  {YYYYMM}_ESTBAN.ZIP      /  {YYYYMM}_ESTBAN_AG.ZIP
# The listing API exposes all available files (back to 1988), keyed by a stable guidLista.
ESTBAN_API_BASE   = "https://www.bcb.gov.br"
ESTBAN_GUID_MUN   = "f6391806-fd85-43af-acf1-c86d5b8dd6df"   # guidLista for municipio + agencia files
ESTBAN_CONTENT_ROOT = ESTBAN_API_BASE + "/content/estatisticas/estatistica_bancaria_estban"

ESTBAN_DATA_ROOT = str(paths.ESTBAN_DIR)
output_estban_mun = str(paths.ESTBAN_RAW_MUN)
output_estban_ag  = str(paths.ESTBAN_RAW_AG)

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "if_data_scrape_1",
        {
            "if_list_output_dir": output_dir,
            "if_type1_output_dir": output_type_1,
            "if_type3_output_dir": output_type_3,
            "estban_mun_output_dir": output_estban_mun,
            "estban_ag_output_dir": output_estban_ag,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    output_dir = _paths["if_list_output_dir"]
    output_type_1 = _paths["if_type1_output_dir"]
    output_type_3 = _paths["if_type3_output_dir"]
    output_estban_mun = _paths["estban_mun_output_dir"]
    output_estban_ag = _paths["estban_ag_output_dir"]

os.makedirs(output_estban_mun, exist_ok=True)
os.makedirs(output_estban_ag, exist_ok=True)

# flag to stop events:
stop_event = threading.Event()

## 4) User-defined functions:
def download_list(year,month, retries_number):
    """Downloand institutional list data for a specific year and month"""
    # Gets data from IF DATA on list of institutions
    year_month = f"{year}{str(month).zfill(2)}" # year_month string

    # Skip if file already exists and has real data (> 500 bytes avoids caching headers-only stubs)
    file_path = os.path.join(output_dir, f"IF_DATA_List_{year}_{month}.csv")
    # Skip only if cached AND outside the rolling refresh window (quarterly series).
    if (os.path.exists(file_path) and os.path.getsize(file_path) > 500
            and not refresh.is_recent_quarter(year, (month - 1) // 3 + 1)):
        logging.info(f"File already exists, skipping: {file_path}")
        print(f"Skipping (already exists): IF_DATA_List_{year}_{month}.csv")
        return False  # no network request made

    url = base_url_list.format(year_month=year_month) # format URL
    print(f"Requesting URL: {url}")
    
    retries = retries_number
    for i in range(retries):
        try:
            response = requests.get(url, timeout = 180) # send GET request
            response.raise_for_status() # check for errors
            
            if not response.content or len(response.text) < 500:
                logging.warning(f"No data (or only stub) received for {year}--{month}. Skipping.")
                print(f"No data received for {year}--{month}. Skipping.")
                return False

            file_path = os.path.join(output_dir, f"IF_DATA_List_{year}_{month}.csv") # file path
            with open(file_path, "wb") as file:
                file.write(response.content)
            logging.info(f"Downloaded data for {year}-{month}")
            print(f"Downloaded data for {year}-{month}")
            return True  # network request made
            break # to stop the retries if successful
        except requests.exceptions.HTTPError as err:
            with open(file_path, "w") as f:
                f.write("CodInst,Data,NomeInstituicao\n" + "SKIP_DUMMY\n" * 50)
            if response.status_code == 500:
                logging.error(f"Server error 500 for {year}--{month}. Treating as unavailable and skipping.")
                print(f"Server error 500 for {year}--{month}. Treating as unavailable and skipping.")
                return False
            else:
                logging.error(f"HTTP error for {year}--{month}: {err}")
                print(f"HTTP error for {year}--{month}: {err}")
                break
        except requests.exceptions.RequestException as err:
            logging.error(f"Failed to download data for {year_month}: {err}")
            print(f"Failed to download data for {year_month}: {err}")
            if i < retries - 1:
                wait_time = 2**i
                logging.info(f"Waiting {wait_time} seconds before retrying...")
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to download data for {year}-{month} after {retries} attempts. Skipping...")
                print(f"Failed to download data for {year}-{month} after {retries} attempts. Skipping...")
                with open(file_path, "w") as f:
                    f.write("CodInst,Data,NomeInstituicao\n" + "SKIP_DUMMY\n" * 50)
                break
        
def download_values(year, quarter, tipo, retries_number):
    """Downloand institutional values data for a specific year, quarter and type"""
    global stop_event
    if stop_event.is_set():
        return # stop the function if the stop event is set
    
    # Skip if file already exists and is non-empty
    if tipo == '1':
        output_dir = output_type_1
    else:
        output_dir = output_type_3
    file_path = os.path.join(output_dir, f"IF_DATA_Values_{year}_{quarter}.csv")
    # Skip only if cached AND outside the rolling refresh window (quarterly series).
    if (os.path.exists(file_path) and os.path.getsize(file_path) > 0
            and not refresh.is_recent_quarter(year, (quarter - 1) // 3 + 1)):
        logging.info(f"File already exists, skipping: {file_path}")
        print(f"Skipping (already exists): {year}-{quarter} (Type {tipo})")
        return False  # no network request made
    
    # Gets data from IF Data on valuse and institutions
    year_quarter = f"{year}{str(quarter).zfill(2)}" # year_quarter string
    url = base_url_values.format(year_quarter=year_quarter, tipo=tipo) # format URL
    print(f"Requesting URL: {url}")
    
    retries = retries_number
    for i in range(retries):
        try:
            response = requests.get(url, timeout = 180) # send GET request
            response.raise_for_status() # check for errors
            
            if not response.content or len(response.text) < 50:
                logging.warning(f"No data received for {year}-{quarter} (Type {tipo}). Skipping.")
                print(f"No data received for {year}-{quarter} (Type {tipo}). Skipping.")
                return False
        
            file_path = os.path.join(output_dir, f"IF_DATA_Values_{year}_{quarter}.csv") # file path
            with open(file_path, "w") as file:
                file.write(response.text)
                
            # check if file is empty (contains only headers)
            df = pd.read_csv(file_path, sep = ",", encoding = "latin1")
            if df.empty or len(df) == 0:
                logging.warning(f"Empty data for {year}-{quarter} (Type {tipo}) – treating as unavailable. Skipping further retries.")
                print(f"Empty data for {year}-{quarter} (Type {tipo}). Skipping.")
                return False
                
            print(f"Downloaded data for {year}-{quarter} (Type {tipo})")
            logging.info(f"Downloaded data for {year}-{quarter} (Type {tipo})")
            return True  # network request made
            break # to stop the retries if successful
        except requests.exceptions.HTTPError as err:
            with open(file_path, "w") as f:
                f.write("CodInst,AnoMes,NomeRelatorio,NumeroRelatorio,Grupo,Conta,NomeColuna,Saldo\nSKIP_DUMMY\n")
            if response.status_code == 500:
                logging.error(f"Server error 500 for {year}-{quarter} (Type {tipo}). Treating as unavailable and skipping.")
                print(f"Server error 500 for {year}-{quarter} (Type {tipo}). Treating as unavailable and skipping.")
                return False
            else:
                logging.error(f"HTTP error for {year}-{quarter} (Type {tipo}): {err}")
                print(f"HTTP error for {year}-{quarter} (Type {tipo}): {err}")
                break
        except requests.exceptions.RequestException as err:
            logging.error(f"Failed to download data for {year_quarter} (Type {tipo}): {err}")
            print(f"Failed to download data for {year_quarter} (Type {tipo}): {err}")
            if i  < retries - 1:
                wait_time = 2**i
                print(f"Waiting {wait_time} seconds before retrying...")
                logging.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to download data for {year}-{quarter} (Type {tipo}) after {retries} attempts. Skipping...")
                print(f"Failed to download data for {year}-{quarter} (Type {tipo}) after {retries} attempts. Skipping...")
                with open(file_path, "w") as f:
                    f.write("CodInst,AnoMes,NomeRelatorio,NumeroRelatorio,Grupo,Conta,NomeColuna,Saldo\nSKIP_DUMMY\n")
                break
   
def download_all_values(year,quarter, retries_number):
    """Download all values for all institutions types in a given year and quarter in parallel.
    Returns True if any type was actually downloaded (network request made)."""
    global stop_event
    if stop_event.is_set():
        return False
    
    any_fetched = False
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_values, year, quarter, tipo, retries_number) for tipo in ['1', '3']]
        # 1 = Prudential Conglomerates, 3 = Individual Institutions
        for future in futures:
            try:
                result = future.result()  # Wait for each future to complete
                if result:
                    any_fetched = True
            except Exception as e:
                logging.error(f"Error parallel-downloading values: {e}")
                print(f"Error parallel-downloading values: {e}")
    return any_fetched
                
def open_log_file():
    """Open log file in the default text editor."""
    log_path = os.path.abspath(log_file)
    print(f"Opening log file.")
    os.system(f"notepad.exe {log_path}")
    
def fetch_estban_listing(pasta, retries_number=3):
    """Fetch the full ESTBAN file listing from BCB's API for the given pasta ('municipio' or 'agencia').

    Returns a dict mapping YYYYMM -> {Nome, Url} for all available files, or {} on failure.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "pt-BR,pt;q=0.9",
        "Referer": ESTBAN_API_BASE + "/estatisticas/estatisticabancariamunicipios",
    }
    url = (
        ESTBAN_API_BASE
        + "/api/servico/sitebcb/Documentos/byListGuid"
        + "?tronco=estatisticas"
        + "&guidLista=" + ESTBAN_GUID_MUN
        + "&ordem=DataDocumento%20desc"
        + "&pasta=" + pasta
    )
    for attempt in range(retries_number):
        try:
            r = requests.get(url, headers=headers, timeout=180)
            r.raise_for_status()
            items = r.json().get("conteudo", [])
            result = {}
            for item in items:
                nome = item["Nome"]
                # Extract YYYYMM from filename like 202412_ESTBAN.csv.zip
                import re
                m = re.match(r"(\d{6})_ESTBAN", nome, re.IGNORECASE)
                if m:
                    result[m.group(1)] = {"Nome": nome, "Url": item["Url"]}
            logging.info(f"ESTBAN listing ({pasta}): {len(result)} files found")
            print(f"ESTBAN listing ({pasta}): {len(result)} files available on BCB")
            return result
        except Exception as err:
            wait = 2 ** attempt
            logging.warning(f"ESTBAN listing fetch failed (attempt {attempt+1}): {err}")
            print(f"  Listing fetch failed: {err}. Retry in {wait}s")
            time.sleep(wait)
    logging.error(f"Could not fetch ESTBAN listing for pasta={pasta}")
    return {}


def download_estban(yyyymm, url_path, out_dir, csv_fname, retries_number):
    """Download a single ESTBAN file (municipality or agency level) for a given YYYYMM.

    url_path  -- the Url field from the BCB listing (e.g. /content/.../202412_ESTBAN.csv.zip)
    csv_fname -- the target CSV filename on disk (e.g. 202412_ESTBAN.CSV)

    BCB now delivers all files as ZIP archives.  This function downloads the ZIP,
    extracts the inner CSV, and saves it with csv_fname in out_dir.
    Skips if csv_fname already exists and is non-empty.
    """
    file_path = os.path.join(out_dir, csv_fname)
    # Skip only if cached AND outside the rolling refresh window (monthly series).
    _y, _m = int(str(yyyymm)[:4]), int(str(yyyymm)[4:6])
    if (os.path.exists(file_path) and os.path.getsize(file_path) > 0
            and not refresh.is_recent_month(_y, _m)):
        logging.info(f"Skipping existing ESTBAN file: {csv_fname}")
        return False  # no network request made

    full_url = ESTBAN_API_BASE + url_path
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": ESTBAN_API_BASE + "/estatisticas/estatisticabancariamunicipios",
    }

    print(f"Downloading ESTBAN: {full_url}")
    for attempt in range(retries_number):
        try:
            response = requests.get(full_url, timeout=120, headers=headers)
            if response.status_code == 404:
                logging.warning(f"ESTBAN 404: {full_url}")
                print(f"  404 – file not available")
                return False
            response.raise_for_status()

            if not response.content or len(response.content) < 200:
                wait = 30 * (2 ** attempt)
                logging.warning(f"ESTBAN empty/tiny response for {yyyymm}. Retry in {wait}s")
                print(f"  Empty response. Retry in {wait}s")
                time.sleep(wait)
                continue

            # Decompress ZIP and write CSV
            try:
                zf = zipfile.ZipFile(io.BytesIO(response.content))
                csv_members = [n for n in zf.namelist() if n.upper().endswith(".CSV")]
                if not csv_members:
                    raise ValueError(f"No CSV inside ZIP for {yyyymm}")
                csv_data = zf.read(csv_members[0])
                with open(file_path, "wb") as f:
                    f.write(csv_data)
            except (zipfile.BadZipFile, ValueError) as ze:
                logging.warning(f"ZIP handling failed for {yyyymm}: {ze}. Saving raw.")
                with open(file_path, "wb") as f:
                    f.write(response.content)

            logging.info(f"Downloaded ESTBAN: {csv_fname}")
            print(f"Downloaded ESTBAN: {csv_fname}")
            return True  # success

        except requests.exceptions.HTTPError as err:
            if response.status_code == 500:
                logging.error(f"Server error 500 for ESTBAN {yyyymm}. Treating as unavailable and skipping.")
                print(f"  Server 500. Treating as unavailable and skipping.")
                return False
            logging.error(f"HTTP error ESTBAN {yyyymm}: {err}")
            print(f"  HTTP error: {err}")
            break
        except requests.exceptions.RequestException as err:
            wait = 2 ** attempt
            logging.error(f"Request error ESTBAN {yyyymm}: {err}")
            print(f"  Request error: {err}. Retry in {wait}s")
            time.sleep(wait)

    logging.error(
        f"Failed to download ESTBAN {yyyymm}. "
        f"Please download manually from https://www.bcb.gov.br/estatisticas/estatisticabancariamunicipios"
    )
    print(
        f"  FAILED: {yyyymm}. Download manually from: "
        f"https://www.bcb.gov.br/estatisticas/estatisticabancariamunicipios"
    )
    with open(file_path, "w") as f:
        f.write("CO_CNPJ,CO_MUNICIPIO\nSKIP_DUMMY\n")
    return False


## 5) Run fetch loops:
start_year = 2016
end_year = 2025
months = list(range(1, 13))
quarters = [3,6,9,12]

retries_number = 3

# Download IF Data
for year in range(start_year, end_year+1):
    # Download list -- IF Data only publishes institution registry for quarter-end months (3,6,9,12).
    # All other months return a headers-only stub, so we only iterate over quarters.
    any_list_downloaded = False
    for quarter in quarters:
        fetched = download_list(year, quarter, retries_number)
        if fetched:
            any_list_downloaded = True
            random_wait = random.uniform(0,2)
            time.sleep(15 + random_wait)
    
    if any_list_downloaded:
        random_wait = random.uniform(0,15)
        time.sleep(60 + random_wait)
    
    # Download values
    for quarter in quarters:
        fetched = download_all_values(year, quarter, retries_number)
        if fetched:
            random_wait = random.uniform(0,2)
            time.sleep(15 + random_wait)

## 6) ESTBAN download loop
# Fetches the full file listing from BCB's API and downloads any missing files.
# Files are delivered as ZIPs and decompressed to CSV on disk.
# The BCB listing goes back to 1988 but we only need from estban_start_year.
estban_start_year = 2013

print("\n--- Starting ESTBAN downloads ---")

# Fetch the available file listings from BCB
listing_mun = fetch_estban_listing("municipio", retries_number)
listing_ag  = fetch_estban_listing("agencia",   retries_number)

import datetime
current_ym = int(datetime.date.today().strftime("%Y%m"))

for yyyymm_key, info in listing_mun.items():
    ym_int = int(yyyymm_key)
    if ym_int < estban_start_year * 100 + 1:  # e.g. 201301
        continue
    if ym_int > current_ym:
        continue
    fname_mun = f"{yyyymm_key}_ESTBAN.CSV"
    fetched_mun = download_estban(
        yyyymm_key, info["Url"],
        output_estban_mun, fname_mun,
        retries_number
    )
    if fetched_mun:
        random_wait = random.uniform(0, 2)
        time.sleep(5 + random_wait)

for yyyymm_key, info in listing_ag.items():
    ym_int = int(yyyymm_key)
    if ym_int < estban_start_year * 100 + 1:
        continue
    if ym_int > current_ym:
        continue
    fname_ag = f"{yyyymm_key}_ESTBAN_AG.CSV"
    fetched_ag = download_estban(
        yyyymm_key, info["Url"],
        output_estban_ag, fname_ag,
        retries_number
    )
    if fetched_ag:
        random_wait = random.uniform(0, 2)
        time.sleep(5 + random_wait)

