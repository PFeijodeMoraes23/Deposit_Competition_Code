## scrape_bcb_estban_ifdata.py
# Authors: Pedro Feijó de Moraes
#
# Pipeline step 1. Downloads:
#   * IF-Data List (institution registry) and Values from the BCB Olinda API, for institution
#     types 1 (prudential conglomerates) and 3 (individual institutions) by default. Type 2
#     (financial conglomerates) has no reader and is fetched only with --if-types 1,2,3.
#   * ESTBAN municipality-level monthly files from the BCB content server (listing API below).
#     If a download fails, the files can be fetched by hand from
#     https://www.bcb.gov.br/estatisticas/estatisticabancariamunicipios
#
# A failed download is never cached: nothing is written for it, any earlier good copy stays,
# the failure is listed at the end and the script exits 1, so the pipeline stops here.
###--------------------------------------------------------------------------------------------------------------------

## 1) Load packages
import os
import io
import sys
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
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler


from utils import paths
from utils import refresh
from utils import bcb_http

## 2) Set up logging
log_file = 'if_data_scrape.log'
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3) # 5MB max size, 3 backups
logging.basicConfig(handlers=[handler], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## 3) Set API URLs and output directories

# --- IF Data (Olinda API) ---
base_url_list = "https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata/IfDataCadastro(AnoMes=@AnoMes)?@AnoMes={year_month}&$top=10000&$format=text/csv&$select=CodInst,Data,NomeInstituicao,DataInicioAtividade,Tcb,Td,Tc,SegmentoTb,Atividade,Uf,Municipio,Sr,CodConglomeradoFinanceiro,CodConglomeradoPrudencial,CnpjInstituicaoLider,Situacao"
base_url_values = "https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata/IfDataValores(AnoMes=@AnoMes,TipoInstituicao=@TipoInstituicao,Relatorio=@Relatorio)?@AnoMes={year_quarter}&@TipoInstituicao={tipo}&@Relatorio='T'&$top=100000000&$format=text/csv&$select=CodInst,AnoMes,NomeRelatorio,NumeroRelatorio,Grupo,Conta,NomeColuna,Saldo"

# IF Data output directories (canonical locations in utils/paths.py); created on first write
output_dir = str(paths.IF_DATA_LIST)
output_type_1 = str(paths.IF_DATA_PRUDENTIAL)
output_type_2 = str(paths.IF_DATA_FINANCIAL)
output_type_3 = str(paths.IF_DATA_INDIVIDUAL)

# --- ESTBAN (BCB new content server) ---
# BCB migrated ESTBAN files to a new URL structure in late 2024.
# New page:  https://www.bcb.gov.br/estatisticas/estatisticabancariamunicipios
# New files: https://www.bcb.gov.br/content/estatisticas/estatistica_bancaria_estban/{pasta}/{Nome}
# Files are delivered as ZIP archives.  Naming convention:
#   2023-02+:  {YYYYMM}_ESTBAN.csv.zip
#   pre-2023:  {YYYYMM}_ESTBAN.ZIP
# The listing API exposes all available files (back to 1988), keyed by a stable guidLista.
# The same listing also serves the per-branch files (pasta "agencia", {YYYYMM}_ESTBAN_AG); no
# script reads them, so they are not downloaded.
ESTBAN_API_BASE   = "https://www.bcb.gov.br"
ESTBAN_GUID_MUN   = "f6391806-fd85-43af-acf1-c86d5b8dd6df"   # guidLista for municipio + agencia files

output_estban_mun = str(paths.ESTBAN_RAW_MUN)
os.makedirs(output_estban_mun, exist_ok=True)

# Every download that failed in this run; main() lists them and exits 1 when any exist.
FAILURES: list[str] = []

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
                logging.error(f"No data (or only a stub) received for List {year}-{month}.")
                print(f"No data received for List {year}-{month}.")
                FAILURES.append(f"IF-Data List {year}-{month}: empty response")
                return False

            os.makedirs(output_dir, exist_ok=True)
            with open(file_path, "wb") as file:
                file.write(response.content)
            logging.info(f"Downloaded data for {year}-{month}")
            print(f"Downloaded data for {year}-{month}")
            return True  # network request made
        except requests.exceptions.HTTPError as err:
            sc = err.response.status_code if err.response is not None else None
            logging.error(f"HTTP {sc} for List {year}-{month}: {err}")
            print(f"HTTP {sc} for List {year}-{month}: {err}")
            FAILURES.append(f"IF-Data List {year}-{month}: HTTP {sc}")
            return False
        except requests.exceptions.RequestException as err:
            logging.error(f"Failed to download data for {year_month}: {err}")
            print(f"Failed to download data for {year_month}: {err}")
            if i < retries - 1:
                wait_time = 2**i
                logging.info(f"Waiting {wait_time} seconds before retrying...")
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to download data for {year}-{month} after {retries} attempts.")
                print(f"Failed to download data for {year}-{month} after {retries} attempts.")
                FAILURES.append(f"IF-Data List {year}-{month}: {err}")
                return False

def download_values(year, quarter, tipo, retries_number):
    """Downloand institutional values data for a specific year, quarter and type"""
    # Skip if file already exists and is non-empty
    if tipo == '1':
        output_dir = output_type_1
    elif tipo == '2':
        output_dir = output_type_2
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

            os.makedirs(output_dir, exist_ok=True)
            with open(file_path, "w") as file:
                file.write(response.text)

            # A header-only body is a genuine empty (e.g. type 1 before the prudential
            # conglomerate existed, 2013): kept on disk, reported, not a failure.
            _head = response.text.split("\n", 2)
            if len(_head) < 2 or not _head[1].strip():
                logging.warning(f"Empty data for {year}-{quarter} (Type {tipo}) – treating as unavailable.")
                print(f"Empty data for {year}-{quarter} (Type {tipo}). Skipping.")
                return False

            print(f"Downloaded data for {year}-{quarter} (Type {tipo})")
            logging.info(f"Downloaded data for {year}-{quarter} (Type {tipo})")
            return True  # network request made
        except requests.exceptions.HTTPError as err:
            sc = err.response.status_code if err.response is not None else None
            # 502/503/504 are transient gateway/timeout errors (large @Relatorio='T'
            # queries); retry with backoff instead of giving up immediately.
            if sc in (502, 503, 504) and i < retries - 1:
                wait_time = 30 * (2 ** i)
                logging.warning(f"Transient HTTP {sc} for {year}-{quarter} (Type {tipo}); retry in {wait_time}s.")
                print(f"Transient HTTP {sc} for {year}-{quarter} (Type {tipo}). Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
                continue
            logging.error(f"HTTP {sc} for {year}-{quarter} (Type {tipo}): {err}")
            print(f"HTTP {sc} for {year}-{quarter} (Type {tipo}): {err}")
            FAILURES.append(f"IF-Data Values {year}-{quarter} type {tipo}: HTTP {sc}")
            return False
        except requests.exceptions.RequestException as err:
            logging.error(f"Failed to download data for {year_quarter} (Type {tipo}): {err}")
            print(f"Failed to download data for {year_quarter} (Type {tipo}): {err}")
            if i  < retries - 1:
                wait_time = 2**i
                print(f"Waiting {wait_time} seconds before retrying...")
                logging.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to download data for {year}-{quarter} (Type {tipo}) after {retries} attempts.")
                print(f"Failed to download data for {year}-{quarter} (Type {tipo}) after {retries} attempts.")
                FAILURES.append(f"IF-Data Values {year}-{quarter} type {tipo}: {err}")
                return False

def download_all_values(year, quarter, retries_number, if_types=('1', '3')):
    """Download values for the requested institution types in a given year and quarter in parallel.
    Returns True if any type was actually downloaded (network request made).
    Types: 1 = Prudential Conglomerates, 2 = Financial Conglomerates, 3 = Individual Institutions."""
    any_fetched = False
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_values, year, quarter, tipo, retries_number) for tipo in if_types]
        for future in futures:
            try:
                result = future.result()  # Wait for each future to complete
                if result:
                    any_fetched = True
            except Exception as e:
                logging.error(f"Error parallel-downloading values: {e}")
                print(f"Error parallel-downloading values: {e}")
    return any_fetched


def fetch_estban_listing(pasta, retries_number=3):
    """Fetch the full ESTBAN file listing from BCB's API for the given pasta ('municipio' or 'agencia').

    Returns a dict mapping YYYYMM -> {Nome, Url} for all available files, or {} on failure
    (recorded in FAILURES).
    """
    import re
    try:
        items = bcb_http.cms_documents(
            bcb_http.retry_session(total=retries_number), "estatisticas", ESTBAN_GUID_MUN, pasta,
            timeout=180, referer=ESTBAN_API_BASE + "/estatisticas/estatisticabancariamunicipios")
    except Exception as err:
        logging.error(f"Could not fetch ESTBAN listing for pasta={pasta}: {err}")
        print(f"  Listing fetch failed: {err}")
        FAILURES.append(f"ESTBAN listing ({pasta}): {err}")
        return {}
    result = {}
    for item in items:
        nome = item["Nome"]
        # Extract YYYYMM from filename like 202412_ESTBAN.csv.zip
        m = re.match(r"(\d{6})_ESTBAN", nome, re.IGNORECASE)
        if m:
            result[m.group(1)] = {"Nome": nome, "Url": item["Url"]}
    logging.info(f"ESTBAN listing ({pasta}): {len(result)} files found")
    print(f"ESTBAN listing ({pasta}): {len(result)} files available on BCB")
    return result


def download_estban(yyyymm, url_path, out_dir, csv_fname, retries_number):
    """Download a single municipality-level ESTBAN file for a given YYYYMM.

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
                # The listing advertised this file, so a 404 is a failure, not an absence.
                logging.error(f"ESTBAN 404: {full_url}")
                print(f"  404 – file not available")
                FAILURES.append(f"ESTBAN {yyyymm}: HTTP 404")
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
    FAILURES.append(f"ESTBAN {yyyymm}")
    return False


## 5) Run fetch loops:
quarters = [3, 6, 9, 12]


def run_if_data(start_year, end_year, retries_number, if_types=('1', '3'), skip_list=False):
    """Download IF-Data List + Values for the given years and institution types."""
    for year in range(start_year, end_year + 1):
        # IF Data only publishes the institution registry (List) for quarter-end
        # months (3,6,9,12); other months return a headers-only stub.
        if not skip_list:
            any_list_downloaded = False
            for quarter in quarters:
                fetched = download_list(year, quarter, retries_number)
                if fetched:
                    any_list_downloaded = True
                    time.sleep(15 + random.uniform(0, 2))
            if any_list_downloaded:
                time.sleep(60 + random.uniform(0, 15))

        for quarter in quarters:
            fetched = download_all_values(year, quarter, retries_number, if_types)
            if fetched:
                time.sleep(15 + random.uniform(0, 2))


def run_estban(retries_number, estban_start_year=2013):
    """Fetch the BCB ESTBAN listing and download any missing municipality-level files."""
    import datetime
    print("\n--- Starting ESTBAN downloads ---")
    listing_mun = fetch_estban_listing("municipio", retries_number)
    current_ym = int(datetime.date.today().strftime("%Y%m"))

    for yyyymm_key, info in listing_mun.items():
        ym_int = int(yyyymm_key)
        if ym_int < estban_start_year * 100 + 1 or ym_int > current_ym:
            continue
        if download_estban(yyyymm_key, info["Url"], output_estban_mun,
                            f"{yyyymm_key}_ESTBAN.CSV", retries_number):
            time.sleep(5 + random.uniform(0, 2))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Download BCB IF-Data + ESTBAN.")
    ap.add_argument("--if-types", default="1,3",
                    help="Comma list of IF-Data institution types to fetch "
                         "(1=Prudential, 2=Financial, 3=Individual). Default: 1,3; "
                         "type 2 has no reader in the pipeline.")
    # 2013 matches the ESTBAN/deposits start (run_estban(..., estban_start_year=2013)),
    # so bank characteristics cover the same years as the deposit panel. Existing files
    # are skipped, so widening the window is idempotent.
    # NOTE: the Conglomerado Prudencial (type 1) did not exist before 2014 — the API
    # returns empty for 2013; types 2/3 do have 2013 data.
    ap.add_argument("--start-year", type=int, default=2013)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--skip-list", action="store_true", help="Skip the IF-Data List download.")
    ap.add_argument("--skip-if", action="store_true", help="Skip IF-Data entirely.")
    ap.add_argument("--skip-estban", action="store_true", help="Skip the ESTBAN download.")
    args = ap.parse_args()

    if_types = tuple(t.strip() for t in args.if_types.split(",") if t.strip())
    retries_number = 3

    if not args.skip_if:
        run_if_data(args.start_year, args.end_year, retries_number, if_types, args.skip_list)
    if not args.skip_estban:
        run_estban(retries_number)

    if FAILURES:
        print(f"\n{len(FAILURES)} download(s) FAILED (nothing was written for them):")
        for f in FAILURES:
            print(f"  - {f}")
        logging.error(f"{len(FAILURES)} download(s) failed: {FAILURES}")
        sys.exit(1)


if __name__ == "__main__":
    main()

