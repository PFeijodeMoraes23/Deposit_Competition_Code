## data_collection_2.py
# Author: Pedro Feijo de Moraes
# Last edited: 2026-02-23
#
# Objective: Collect institution-level CDB/RDB/LCI/LCA deposit rates from the
#            Open Finance Brasil open-data APIs (Phase 1 -- no auth required).
#
# How it works:
#   1. Fetch the Open Finance participant directory to discover all institutions
#      and their "bank-fixed-incomes" API endpoints.
#   2. For each endpoint, paginate through the standardised JSON to collect
#      product-level rate data (indexer, rate quartile distributions, conditions).
#   3. Save raw JSON per institution + a consolidated CSV snapshot.
#
# The consolidated CSV is consumed by egan_panel_build.py (read-only).
#
# Data source:
#   Directory: https://data.directory.openbankingbrasil.org.br/participants
#   Endpoint:  {base}/open-banking/opendata-investments/v1/bank-fixed-incomes
#
# Prerequisites:
#   pip install pandas requests
## ---------------------------------------------------------------------------

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
import json
import os
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler

# =============================================================================
# 1) SETUP
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BCB_PATH   = os.path.join(PARENT_DIR, "BCB")

EGAN_PATH      = os.path.join(BCB_PATH, "Egan_et_al_2025_Rep")
RAW_PATH       = os.path.join(EGAN_PATH, "raw")
OF_RAW_PATH    = os.path.join(RAW_PATH, "OF_RATES_RAW")       # per-institution JSON
PROCESSED_PATH = os.path.join(EGAN_PATH, "processed")
OF_PROC_PATH   = os.path.join(PROCESSED_PATH, "OF_RATES")     # consolidated CSV

for _p in [OF_RAW_PATH, OF_PROC_PATH]:
    os.makedirs(_p, exist_ok=True)

# Logging
_log_file = os.path.join(SCRIPT_DIR, "data_collection_2.log")
_handler  = RotatingFileHandler(
    _log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
logging.basicConfig(
    handlers=[_handler],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Persistent HTTP session
session = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[502, 503, 504],
    allowed_methods=["GET"],
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=10, pool_maxsize=10)
session.mount("https://", _adapter)
session.mount("http://", _adapter)

# Concurrency
MAX_WORKERS = 6
PAGE_SIZE   = 25           # max page-size allowed by the spec
REQUEST_TIMEOUT = 30       # seconds
INTER_REQUEST_SLEEP = 0.5  # polite throttle

# Directory URL
DIRECTORY_URL = (
    "https://data.directory.openbankingbrasil.org.br/participants"
)

# We only want bank fixed-income (CDB, RDB, LCI, LCA) endpoints
TARGET_API_FAMILY = "opendata-investments_bank-fixed-incomes"


# =============================================================================
# 2) DISCOVER ENDPOINTS
# =============================================================================

def fetch_participant_directory() -> list[dict]:
    """
    Download the Open Finance participant directory and extract all
    bank-fixed-incomes endpoints.

    Returns a list of dicts:
        org_name, org_cnpj, brand, endpoint_url
    """
    logging.info(f"Fetching participant directory from {DIRECTORY_URL} ...")
    resp = session.get(DIRECTORY_URL, timeout=60)
    resp.raise_for_status()
    participants = resp.json()
    logging.info(f"Directory returned {len(participants)} organisations.")

    endpoints = []
    for org in participants:
        org_name = org.get("OrganisationName", "")
        org_cnpj = org.get("RegistrationNumber", "")

        for auth_server in org.get("AuthorisationServers", []):
            brand = auth_server.get("CustomerFriendlyName", org_name)

            for api_res in auth_server.get("ApiResources", []):
                if api_res.get("ApiFamilyType") != TARGET_API_FAMILY:
                    continue
                if api_res.get("Status", "") != "Active":
                    continue

                for disc in api_res.get("ApiDiscoveryEndpoints", []):
                    url = disc.get("ApiEndpoint", "")
                    if url:
                        endpoints.append({
                            "org_name":  org_name,
                            "org_cnpj":  org_cnpj,
                            "brand":     brand,
                            "endpoint":  url,
                        })

    logging.info(
        f"Found {len(endpoints)} bank-fixed-incomes endpoints "
        f"across {len(set(e['org_cnpj'] for e in endpoints))} organisations."
    )
    return endpoints


# =============================================================================
# 3) SCRAPE INDIVIDUAL ENDPOINTS
# =============================================================================

def fetch_bank_fixed_incomes(endpoint_info: dict) -> list[dict]:
    """
    Paginate through a single institution's /bank-fixed-incomes endpoint
    and return all product records.

    Each returned dict has the endpoint_info fields merged in so we know
    which institution the record belongs to.
    """
    url       = endpoint_info["endpoint"]
    org_name  = endpoint_info["org_name"]
    org_cnpj  = endpoint_info["org_cnpj"]
    brand     = endpoint_info["brand"]

    all_records: list[dict] = []
    page = 1

    while True:
        try:
            resp = session.get(
                url,
                params={"page": page, "page-size": PAGE_SIZE},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.exceptions.RequestException as e:
            logging.warning(f"Error fetching {org_name} (page {page}): {e}")
            break
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Bad JSON from {org_name} (page {page}): {e}")
            break

        data = body.get("data", [])
        if not data:
            break

        for record in data:
            record["_org_name"]  = org_name
            record["_org_cnpj"]  = org_cnpj
            record["_brand"]     = brand
            record["_endpoint"]  = url
            all_records.append(record)

        meta = body.get("meta", {})
        total_pages = meta.get("totalPages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(INTER_REQUEST_SLEEP)

    if all_records:
        logging.info(
            f"  {org_name}: {len(all_records)} products collected."
        )
    else:
        logging.info(f"  {org_name}: no data or endpoint error.")

    return all_records


def scrape_all_endpoints(endpoints: list[dict]) -> list[dict]:
    """
    Scrape all endpoints in parallel and return the merged record list.
    """
    logging.info(f"Scraping {len(endpoints)} endpoints (max {MAX_WORKERS} workers) ...")
    all_records: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(fetch_bank_fixed_incomes, ep): ep
            for ep in endpoints
        }
        for future in as_completed(futures):
            ep = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as e:
                logging.error(
                    f"Unhandled error for {ep['org_name']}: {e}"
                )

    logging.info(f"Total records collected: {len(all_records)}")
    return all_records


# =============================================================================
# 4) FLATTEN TO TABULAR
# =============================================================================

def flatten_records(records: list[dict]) -> pd.DataFrame:
    """
    Convert the nested JSON product records into a flat DataFrame.

    Columns produced:
        org_name, org_cnpj, brand,
        participant_cnpj, participant_name,
        issuer_cnpj, issuer_name,
        investment_type,  (CDB, RDB, LCI, LCA)
        indexer,          (CDI, PRE_FIXADO, IPCA, ...)
        indexer_info,
        rate_min, rate_max,
        rate_1q_value, rate_1q_pct,
        rate_2q_value, rate_2q_pct,
        rate_3q_value, rate_3q_pct,
        rate_4q_value, rate_4q_pct,
        min_amount,
        redemption_term,
        expiration_period,
        grace_period,
        target_audience,   (PESSOA_NATURAL, PESSOA_JURIDICA)
        snapshot_date
    """
    rows = []
    snap_date = datetime.utcnow().strftime("%Y-%m-%d")

    for rec in records:
        participant = rec.get("participant", {})
        index       = rec.get("index", {})
        conditions  = rec.get("investmentConditions", {})
        remun       = index.get("issueRemunerationRate", {})

        # Build quartile columns
        prices = {
            p["interval"]: p
            for p in remun.get("prices", [])
            if "interval" in p
        }

        row = {
            "org_name":          rec.get("_org_name", ""),
            "org_cnpj":          rec.get("_org_cnpj", ""),
            "brand":             rec.get("_brand", ""),
            "participant_cnpj":  participant.get("cnpjNumber", ""),
            "participant_name":  participant.get("name", ""),
            "issuer_cnpj":       rec.get("issuerInstitutionCnpjNumber", ""),
            "issuer_name":       rec.get("issuerInstitutionName", ""),
            "investment_type":   rec.get("investmentType", ""),
            "indexer":           index.get("indexer", ""),
            "indexer_info":      index.get("indexerAdditionalInfo", ""),
            "rate_min":          remun.get("minimum", ""),
            "rate_max":          remun.get("maximum", ""),
            "rate_1q_value":     prices.get("1_FAIXA", {}).get("value", ""),
            "rate_1q_pct":       prices.get("1_FAIXA", {}).get("operationRate", ""),
            "rate_2q_value":     prices.get("2_FAIXA", {}).get("value", ""),
            "rate_2q_pct":       prices.get("2_FAIXA", {}).get("operationRate", ""),
            "rate_3q_value":     prices.get("3_FAIXA", {}).get("value", ""),
            "rate_3q_pct":       prices.get("3_FAIXA", {}).get("operationRate", ""),
            "rate_4q_value":     prices.get("4_FAIXA", {}).get("value", ""),
            "rate_4q_pct":       prices.get("4_FAIXA", {}).get("operationRate", ""),
            "min_amount":        conditions.get("minimumAmount", ""),
            "redemption_term":   conditions.get("redemptionTerm", ""),
            "expiration_period": conditions.get("expirationPeriod", ""),
            "grace_period":      conditions.get("gracePeriod", ""),
            "target_audience":   rec.get("targetAudience", ""),
            "snapshot_date":     snap_date,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Parse numeric rate columns
    rate_cols = [
        "rate_min", "rate_max",
        "rate_1q_value", "rate_1q_pct",
        "rate_2q_value", "rate_2q_pct",
        "rate_3q_value", "rate_3q_pct",
        "rate_4q_value", "rate_4q_pct",
        "min_amount",
    ]
    for col in rate_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logging.info(
        f"Flattened {len(df)} rows | "
        f"{df['org_cnpj'].nunique()} organisations | "
        f"{df['investment_type'].nunique()} product types"
    )
    return df


# =============================================================================
# 5) MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Starting Open Finance bank-fixed-incomes collection")
    logging.info("=" * 60)

    # ---- Step 1: Discover endpoints ----
    print("Step 1/4: Fetching participant directory ...")
    endpoints = fetch_participant_directory()
    if not endpoints:
        print("ERROR: No bank-fixed-incomes endpoints found.")
        exit(1)
    print(f"  Found {len(endpoints)} endpoints.")

    # Save directory snapshot
    dir_path = os.path.join(OF_RAW_PATH, "participant_directory.json")
    with open(dir_path, "w", encoding="utf-8") as f:
        json.dump(endpoints, f, ensure_ascii=False, indent=2)
    logging.info(f"Directory snapshot saved to {dir_path}")

    # ---- Step 2: Scrape all endpoints ----
    print(f"Step 2/4: Scraping {len(endpoints)} endpoints ...")
    records = scrape_all_endpoints(endpoints)
    if not records:
        print("WARNING: No product records collected.")
        logging.warning("No product records collected from any endpoint.")

    # Save raw JSON
    snap_date = datetime.utcnow().strftime("%Y%m%d")
    raw_json_path = os.path.join(
        OF_RAW_PATH, f"bank_fixed_incomes_raw_{snap_date}.json"
    )
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Raw JSON saved: {raw_json_path}")

    # ---- Step 3: Flatten to CSV ----
    print("Step 3/4: Flattening to tabular format ...")
    df = flatten_records(records)

    # ---- Step 4: Save outputs ----
    print("Step 4/4: Saving ...")

    # Timestamped snapshot
    snap_csv = os.path.join(
        OF_PROC_PATH, f"of_bank_fixed_incomes_{snap_date}.csv"
    )
    df.to_csv(snap_csv, index=False)

    # Latest file (overwritten each run, used by egan_panel_build.py)
    latest_csv = os.path.join(OF_PROC_PATH, "of_bank_fixed_incomes_latest.csv")
    df.to_csv(latest_csv, index=False)

    logging.info(f"Snapshot CSV: {snap_csv}")
    logging.info(f"Latest CSV:   {latest_csv}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"Collection complete.")
    print(f"  Records       : {len(df):,}")
    print(f"  Institutions  : {df['org_cnpj'].nunique()}")
    print(f"  Product types : {sorted(df['investment_type'].unique().tolist())}")
    print(f"  Indexers      : {sorted(df['indexer'].unique().tolist())}")
    print(f"  Snapshot CSV  : {snap_csv}")
    print(f"  Latest CSV    : {latest_csv}")
    print(f"{'=' * 60}")

    # CDB-specific summary
    cdb = df[df["investment_type"] == "CDB"]
    if not cdb.empty:
        print(f"\n--- CDB Rate Summary (indexer=CDI) ---")
        cdb_cdi = cdb[cdb["indexer"] == "CDI"]
        if not cdb_cdi.empty:
            print(f"  Institutions offering CDB+CDI : {cdb_cdi['org_cnpj'].nunique()}")
            print(f"  Rate range (% of CDI):")
            print(f"    min across all: {cdb_cdi['rate_min'].min():.4f}")
            print(f"    max across all: {cdb_cdi['rate_max'].max():.4f}")
            print(f"    median of 1st quartile medians: {cdb_cdi['rate_1q_value'].median():.4f}")

    logging.info("Script completed successfully.")
    print("\nDone.")
