# COMBINED SCRIPT: generate_mca_json.py -> rectangularize_mca_json.py -> ibge_demographics_panel.py

# --- FROM generate_mca_json.py ---
# generate_mca_json.py
# This script defines the microregions and immediate regions of Brazil as per IBGE classification.
# Author: Pedro Feijo de Moraes
# Last Edited: 2025-11-12
# -----------------------------------------------------------------------------------------------

# 1) Import necessary libraries and set directories:

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

from utils import paths

import pandas as pd
import geopandas as gpd
import geobr
from pathlib import Path
import sys
import json 
import warnings

# Suppress warnings from geopandas/shapely during buffer operation and sjoin
warnings.filterwarnings('ignore', 'The CRS of the two GeoSeries', UserWarning)
warnings.filterwarnings('ignore', '.*Initializing tools.', UserWarning)

try:
    root_dir = Path(__file__).parent.parent
except NameError:
    root_dir = Path.cwd()
output_dir = paths.IBGE_DIR

output_dir.mkdir(parents = True, exist_ok=True)

# 2) User-defined functions:

def get_closest_predecessor(target_year, available_years, classification_name):
    """
    Returns most recent available year less than or equal target_year from the list of available years.
    """
    if valid_years := [y for y in available_years if y <= target_year]:
        closest_year = max(valid_years)
        if closest_year < target_year:
            # Provide a warning only if we are actually using a predecessor year
            print(f"  > Warning: Data for {classification_name} in {target_year} is not available. Using {closest_year}.")
        return closest_year
    
    else:
        # Fallback only executed if target is less than all available years 
        # # Use oldest available year as a last resort
        oldest_available = min(available_years)
        print(f"  > Warning: Target year {target_year} is older than oldest data for {classification_name}. Using {oldest_available}.")
        return oldest_available

def safe_mode(series):
    """ Safely calculates mode, handles NaN, convert to integer string, or returns None if no mode exists. """
    valid_series = series.dropna() # Drop NaNs before calculating mode
    
    if valid_series.empty: # sjon failed for the whole MCA area
        return 'N/A'
    
    mode_value = valid_series.mode()
    if mode_value.empty:
        return 'N/A'
    
    mode_val = mode_value[0]
    
    if pd.isna(mode_val):
        return 'N/A'
    
    try:
        return str(int(float(mode_val)))
    except ValueError:
        return mode_val

def copy_predecessor_data(target_year, predecessor_year, final_data):
    """
    Copies regional data from predecessor_year to target_year in final_data dictionary for all MCAs as needed.
    This is for the case of 2011 and 2012 where no new data is available.
    """
    print(f" > Imputing data for {target_year} from predecessor year {predecessor_year}.")
    
    pred_year_str = str(predecessor_year)
    target_year_str = str(target_year)
    
    for mca_code, data in final_data.items():
        if pred_year_str in data['regions'] and target_year_str not in data['regions']:
            data['regions'][target_year_str] = data['regions'][pred_year_str].copy()
         

def generate_mca_main():
    
    START_YEAR = 2010
    END_YEAR = 2025
    MCA_REFERENCE = 2010
    YEARS = list(range(START_YEAR, END_YEAR + 1))
    # NOTE: the filename is pinned to the legacy "2010_2024" label so the 9
    # downstream consumers (panel_6, panel_9, scrape_5/4/5/6/8, scrape_inss)
    # keep finding the crosswalk; the CONTENT now spans 2010..END_YEAR. The
    # hardcoded reader in rectangularize_mca_main() reads this same name.
    final_json_file = output_dir / "muni_mca_regions_2010_2024.json"
    
    REGION_YEARS = [2017, 2019, 2020]
    MICRO_MESO_YEARS = [2010, 2013, 2014, 2015, 2016]
    ALL_REGION_YEARS = set(MICRO_MESO_YEARS) | set(REGION_YEARS)
    
    # Caching for efficiency
    REGION_CACHE = {}
    MUNI_CACHE = {}
    
    print(f"MCA-based Mapping from {START_YEAR} to {END_YEAR} with Closest Year Logic.")
    print(f"Using {MCA_REFERENCE} boundaries as the reference for Minimum Comparable Areas (MCA).")
    
    # Step 1: Load MCA Crosswalk Table (2010 boundaries):
    try:
        # load the polygons defined by the old/newest year combination available and rename for clarity
        mca_geo_df = geobr.read_comparable_areas(start_year = 1872, end_year = MCA_REFERENCE)
        
        mca_geo_df.rename(columns = {'code_amc': 'mca_code', 'list_name_muni_2010': 'mca_name'}, inplace = True)
        
        mca_geo_df = mca_geo_df[['mca_code', 'mca_name','geometry', 'list_code_muni_2010']].drop_duplicates(subset='mca_code')
        mca_geo_df.geometry = mca_geo_df.geometry.buffer(0) # Resolves potential issues
        mca_crs = mca_geo_df.crs
        print(f"  > Successfully loaded MCA data for {MCA_REFERENCE} with {len(mca_geo_df)} areas.")
    except Exception as e:
        if 'mca_geo_df' in locals():
            print(f"Available columns in MCA DataFrame: {mca_geo_df.columns.tolist()}")
        print(f"!!! FATAL ERROR: Unable to load Comparable Areas (MCA). Error: {e}.")
        sys.exit(1)
                   
    # Step 2: Cache all necessary regional boundaries:
    print("Caching Regional Boundaries for all relevant years.")
    for year in ALL_REGION_YEARS:
        try:
            if year < 2017:
                # Micro and Meso Regions
                REGION_CACHE[('micro', year)] = geobr.read_micro_region(year = year)
                REGION_CACHE[('meso', year)] = geobr.read_meso_region(year = year)
            else:
                # Immediate and Intermediate Regions
                REGION_CACHE[('immgr', year)] = geobr.read_immediate_region(year = year)
                REGION_CACHE[('intgr', year)] = geobr.read_intermediate_region(year = year)
        except Exception as e:
            print(f"!!! ERROR caching regions for year {year}: {e}. Continuing.")
      
    # 4) Main Processing Loop:
    
    # Masters and targets:
    final_data = {}
    TARGET_CRS = None
    
    for year in YEARS:
        print(f"\nProcessing year: {year}.")
        
        # A. Check 2011, 2012 issue:
        if year in [2011, 2012]: 
            if 2010 in YEARS and next(iter(final_data.values()), {}).get('regions', {}).get(str(2010)):
                copy_predecessor_data(year, 2010, final_data)
                print(f"  > Data for {year} imputed from 2010.")
            else: 
                print(f"  > No data available to impute for {year}. Predecessor (2010) not yet available or failed to process. Skipping.")
        
            continue
    
        try:
            # B. Download Dynamic Muni Boundaries:
            if year not in MUNI_CACHE:
                muni_df = geobr.read_municipality(year = year)
                MUNI_CACHE[year] = muni_df
            else:
                muni_df = MUNI_CACHE[year]
            
            target_crs = muni_df.crs
            
            # C. Optimize CRS transformation (only transforms MCA once):
            if TARGET_CRS is None:
                TARGET_CRS = target_crs
                if mca_crs != TARGET_CRS:
                    mca_geo_df = mca_geo_df.to_crs(TARGET_CRS)
                print(f"  > Set TARGET_CRS to {TARGET_CRS.to_string()}.")
            
            # D. Assign MCA code to Current Municipality Set:
            muni_with_mca = gpd.sjoin(
                    muni_df[['code_muni', 'name_muni', 'geometry']],
                    mca_geo_df[['mca_code', 'mca_name', 'geometry']],
                    how = 'inner',
                    predicate = 'intersects'
                ).drop_duplicates(subset =['code_muni']).reset_index(drop = True)
            print(f"  > Mapped {len(muni_with_mca)} municipalities to MCA codes for {year}.")
            
            # E. Closest year and state data
            region_years_to_use = MICRO_MESO_YEARS if year < 2017 else REGION_YEARS
            closest_regional_year = get_closest_predecessor(year, region_years_to_use, "Regional")
                         
            # F. Conditional Region Selection and Join w Closest Year:     
            low_prefix, high_prefix = ('micro', 'meso') if year < 2017 else ('immgr', 'intgr')
            
            low_level_df = REGION_CACHE.get((low_prefix, closest_regional_year)).to_crs(target_crs)
            high_level_df = REGION_CACHE.get((high_prefix, closest_regional_year)).to_crs(target_crs)
            
            low_col_root, high_col_root = ('micro', 'meso') if year < 2017 else ('immediate', 'intermediate')
            
            low_cols = [f'code_{low_col_root}', f'name_{low_col_root}', 'geometry']
            high_cols = [f'code_{high_col_root}', f'name_{high_col_root}', 'geometry']
            
            if any(col not in low_level_df.columns for col in low_cols):
                 print(f"  > Warning: Missing columns in low_level_df for {year}. Expected: {low_cols}")
                 continue
            if any(col not in high_level_df.columns for col in high_cols):
                 print(f"  > Warning: Missing columns in high_level_df for {year}. Expected: {high_cols}")
                 continue
    
            low_level_df = low_level_df[low_cols].rename(
                columns={low_cols[0]: f'{low_prefix}_code', low_cols[1]: f'{low_prefix}_name'}
            )
            high_level_df = high_level_df[high_cols].rename(
                columns={high_cols[0]: f'{high_prefix}_code', high_cols[1]: f'{high_prefix}_name'}
            )
            
            # G. Spatial Join and Aggregation:
            low_join = gpd.sjoin(muni_with_mca[['code_muni', 'mca_code', 'geometry']], low_level_df, how='left', predicate='intersects').drop(columns=['geometry', 'index_right'], errors='ignore')
            low_join = low_join[['code_muni', f'{low_prefix}_code', f'{low_prefix}_name']].drop_duplicates(subset=['code_muni'])
            
            high_join = gpd.sjoin(muni_with_mca[['code_muni', 'mca_code', 'geometry']], high_level_df, how='left', predicate='intersects').drop(columns=['geometry', 'index_right'], errors='ignore')
            high_join = high_join[['code_muni', f'{high_prefix}_code', f'{high_prefix}_name']].drop_duplicates(subset=['code_muni'])
            
            # Merge all regional data with MCA code
            mapping_table = muni_with_mca[['code_muni', 'mca_code']].merge(
                low_join, on='code_muni', how='left'
            ).merge(
                high_join, on='code_muni', how='left'
            )
            
            # H. Aggregate Regional Codes by MCA (Predecessor Logic):
            agg_map = {
                'mca_name': ('mca_code', 'size'), 
                f'{low_prefix}_code': (f'{low_prefix}_code', safe_mode),
                f'{low_prefix}_name': (f'{low_prefix}_name', safe_mode),
                f'{high_prefix}_code': (f'{high_prefix}_code', safe_mode),
                f'{high_prefix}_name': (f'{high_prefix}_name', safe_mode)
            }
    
            mca_aggregated = mapping_table.groupby('mca_code').agg(
                **{k: v for k, v in agg_map.items() if k != 'mca_name'}
            ).reset_index()
                    
            mca_name_lookup = mca_geo_df.set_index('mca_code')['mca_name'].to_dict()
            mca_aggregated['mca_name'] = mca_aggregated['mca_code'].apply(lambda x: mca_name_lookup.get(x, 'N/A'))
            
            # I. Populate final df
            for index, row in mca_aggregated.iterrows():
                mca_code = str(int(row['mca_code']))
                
                if mca_code not in final_data:
                    final_data[mca_code] = {
                        'name': row['mca_name'],
                        'geocode': mca_code,
                        'regions': {}
                    }
                    muni_list_str = mca_geo_df.loc[mca_geo_df['mca_code'] == row['mca_code'], 'list_code_muni_2010'].iloc[0]
                    final_data[mca_code]['muni_codes_list'] = muni_list_str
                
                final_data[mca_code]['regions'][str(year)] = {
                    f'{low_prefix}_code': row[f'{low_prefix}_code'],
                    f'{low_prefix}_name': row[f'{low_prefix}_name'],
                    f'{high_prefix}_code': row[f'{high_prefix}_code'],
                    f'{high_prefix}_name': row[f'{high_prefix}_name']
                }
            print(f"  > Aggregated regional data for {len(mca_aggregated)} MCAs for {year}.") 
            
        except Exception as e:
            print(f"!!! ERROR processing year {year}: {e}. Continuing to next year.")
    
    print("\nProcessing Complete.")
                
    # 5) Save final JSON output:
    if final_data:
        for data in final_data.values():
            data['name'] = data.pop('muni_codes_list') # Rename for clarity in output
        
        json_list = list(final_data.values())
        
        with open(final_json_file, 'w', encoding='utf-8') as f:
            
            json.dump(json_list, f, indent=4, ensure_ascii=False)
        
        print(f'Data for {len(final_data)} MCAs saved to {final_json_file}.')
        print("JSON encoding used: UTF-8 to correctly preserve Portuguese characters.")
    else:
        print("No data to save. Final data dictionary is empty.") 
    

# --- FROM rectangularize_mca_json.py ---
# rectangularize_mca_json.py
# This script rectangularizes the output of generate_mca_json.py.
# Author: Pedro Feijo de Moraes
# Last Edited: 2025-11-12
# -----------------------------------------------------------------------------------------------

# 1) Import necessary libraries and set directories:
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import warnings
try:
    root_dir = Path(__file__).parent.parent
except NameError:
    root_dir = Path.cwd()
output_dir = paths.IBGE_DIR
output_dir.mkdir(parents = True, exist_ok=True)

# 2) User-defined functions:
def load_uf_lookup():
    """
    Loads a lookup table for Brazilian states (UFs) with their codes, abbreviations, and names.
    """
    print("Loading state metadata from geobr.")
    
    try: 
        state_df_meta = geobr.read_state(year = 2020) # Most recent state data
        state_df_meta = state_df_meta[['code_state', 'abbrev_state', 'name_state']].drop_duplicates()
        state_df_meta['code_state'] = state_df_meta['code_state'].astype(float).astype(int).astype(str) # ensure key is a string
        lookup = state_df_meta.set_index('code_state').to_dict(orient = 'index')
        print(" > State metadata lookup created successfully.")
        return lookup
    except Exception as e:
        print(f"Error loading state metadata: {e}. State information will be incomplete.")
        return {}
    
def load_municipality_lookup():
    """
    Loads a lookup dictionary for municipality codes to names.
    We use year = 2010 as it matches the MCA reference year.
    """
    print("Loading municipality metadata from geobr.")
    try:
        muni_df_meta = geobr.read_municipality(year = 2010)
        muni_df_meta = muni_df_meta[['code_muni', 'name_muni']].drop_duplicates()
        muni_df_meta['code_muni'] = muni_df_meta['code_muni'].astype(float).astype(int).astype(str) # ensure key is a string
        lookup = muni_df_meta.set_index('code_muni')['name_muni'].to_dict()
        print(" > Municipality metadata lookup created successfully.")
        return lookup
    except Exception as e:
        print(f"Error loading municipality metadata: {e}. Municipality names will be incomplete.")
        return {}

def rectangularize_mca_json(json_file, state_lookup, muni_lookup):
    """
    Converts the nested MCA-year JSON into a flat muncipiality-year panel DataFrame.
    
    Args:
        json_file (Path): The path to the input JSON file.
        state_lookup (dict): The pre-loaded dictionary of state metadata.
        muni_lookup (dict): The pre-loaded dictionary of municipality metadata.
    """
    
    print(f"Loading data from {json_file}.")
    
    with open(json_file, 'r', encoding = 'utf-8') as f:
        data_json = json.load(f)
    
    all_rows = []
    
    # Iterate through MCA (each item in top-level list)
    for mca_entry in data_json:
        
        # A. Extract static MCA-lvl data:
        mca_static_data = {'mca_code': mca_entry.get('geocode')}
        muni_codes_str = mca_entry.get('name', '') 
        muni_codes_list = [code.strip() for code in muni_codes_str.split(',') if code.strip()]
                
        # B. Iterate over each year's data for this MCA
        for year, region_data in mca_entry.get('regions', {}).items():
            year_region_data = {
                'year': int(year),
                'micro_code': region_data.get('micro_code'),
                'micro_name': region_data.get('micro_name'),
                'meso_code': region_data.get('meso_code'),
                'meso_name': region_data.get('meso_name'),
                'immgr_code': region_data.get('immgr_code'),
                'immgr_name': region_data.get('immgr_name'),
                'intgr_code': region_data.get('intgr_code'),
                'intgr_name': region_data.get('intgr_name')
            }
            
            for muni_code in muni_codes_list:
                row = {
                    'municipality_code': muni_code,
                    **mca_static_data,
                    **year_region_data
                }
                all_rows.append(row)
    
    print(f"Total panel rows created: {len(all_rows)}")
    
    # D. Convert list to a DataFrame:
    df_panel = pd.DataFrame(all_rows)
    
    df_panel['municipality_code'] = df_panel['municipality_code'].astype(str).str.strip()
    df_panel['year'] = df_panel['year'].astype(int)
    
    # E. Add State information:
    print("Deriving state information from municipality codes...")
    df_panel['municipality_code'] = df_panel['municipality_code'].astype(str).str.strip()
    df_panel['state_code'] = df_panel['municipality_code'].str[:2]
    df_panel['state_uf'] = df_panel['state_code'].map(lambda x: state_lookup.get(x, {}).get('abbrev_state', 'N/A'))
    df_panel['state_name'] = df_panel['state_code'].map(lambda x: state_lookup.get(x, {}).get('name_state', 'N/A'))
    df_panel['municipality_name'] = df_panel['municipality_code'].map(muni_lookup).fillna('N/A')
    
    # F. Sort and reset index:
    cols_order = [
        'municipality_code', 'municipality_name', 
        'state_code', 'state_uf', 'state_name', 
        'mca_code', 'year', 
        'micro_code', 'micro_name', 'meso_code', 'meso_name',
        'immgr_code', 'immgr_name', 'intgr_code', 'intgr_name'
    ]
    final_cols = [col for col in cols_order if col in df_panel.columns]
    df_panel = df_panel[final_cols].sort_values(by=['municipality_code', 'year']).reset_index(drop=True)

    return df_panel

# 3) Main execution:


# --- FROM ibge_demographics_panel.py ---
## ibge_demographics_panel.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-03-06
#
# Purpose: Build a panel of MCA (Área Mínima Comparável) level demographics
#          for use as market controls in the deposit competition model.
#
#   Inputs:
#     1. IBGE SIDRA REST API
#          - Table 6579, Variable 9324  : municipal population estimates (2013-2022)
#          - Table 4714, Variable 93    : population projections (2023-2024 extension)
#          - Table 5938, Variable 37    : municipal GDP at current prices, R$ thousands
#                                        (2002-2021); carry-forward for 2022+
#          - Table 9514, Variable 93    : population by age group, 2022 Census
#          - Table 1552, Variable 93    : population by age group, 2010 Census
#            (Tables 9514/1552 used for fraction_65plus and fraction_young; linearly
#             interpolated for years 2013-2024 between the two census years)
#     2. IBGE/muni_mca_regions_2010_2024_panel.csv
#          - Municipality -> MCA crosswalk with year dimension (already in repo)
#
#   Output:
#     IBGE/mca_demographics_panel.csv
#       Columns: mca_code, year,
#                pop_total,            (sum of municipality populations)
#                gdp_total_r1000,      (sum of municipal GDPs, R$ thousands)
#                gdp_per_capita,       (gdp_total_r1000*1000 / pop_total, R$)
#                fraction_65plus,      (population-weighted share aged 65+)
#                fraction_young,       (population-weighted share aged 0-14)
#                n_municipalities,     (distinct municipalities in MCA)
#                gdp_imputed,          (True if GDP year was carried forward)
#                age_interpolated      (True if age shares were linearly interpolated)
#
# SIDRA API: https://api.ibge.gov.br/
#   Endpoint: /api/v3/agregados/{tabela}/periodos/{periodos}/variaveis/{variavel}
#             ?localidades=N6[all][&classificacao={classif}[{cats}]]
#
# Notes:
#   * Census age data (Tables 9514/1552) are fetched for the census years 2022 and 2010
#     using age-group classification 287 with specific 5-year category IDs (municipalities
#     are batched in groups of 100 to avoid HTTP 500 from SIDRA). All inter-census years
#     are linearly interpolated.
#   * GDP municipal data (table 5938) is published with a ~2 year lag. Later years
#     are carried forward from the last available year (flagged gdp_imputed=True).
###-----------------------------------------------------------------------------

import concurrent.futures
import os
import time
import logging
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import requests
import pandas as pd
import numpy as np

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

## -----------------------------------------------------------------------------
## 1) PATHS & CONSTANTS
## -----------------------------------------------------------------------------
BASE      = str(paths.OPEN_FINANCE)
IBGE_DIR  = str(paths.IBGE_DIR)
MCA_CSV   = os.path.join(IBGE_DIR, "muni_mca_regions_2010_2024_panel.csv")
OUTPUT    = os.path.join(IBGE_DIR, "mca_demographics_panel.csv")

SIDRA_BASE = "https://servicodados.ibge.gov.br/api/v3/agregados"
HEADERS    = {"User-Agent": "research/pedro.feijodemoraes@yale.edu"}

START_YEAR = 2013
END_YEAR   = 2025

# Cache directory for large census age-group downloads.
# If the IBGE census API returns 500 (common for large table requests),
# the code falls back to any previously cached file at this path.
# You can also drop a manually-downloaded CSV here to skip the API entirely:
#   IBGE/census_age_9514_2022_raw.csv  (2022 Census, Table 9514)
#   IBGE/census_age_1552_2010_raw.csv  (2010 Census, Table 1552)
# Required columns: municipio_code (int), age_label (str), value (float)
AGE_CACHE_DIR = str(paths.IBGE_RAW)

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "ibge_demographics_panel",
        {
            "mca_csv": MCA_CSV,
            "output_csv": OUTPUT,
            "age_cache_dir": AGE_CACHE_DIR,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    MCA_CSV = _paths["mca_csv"]
    OUTPUT = _paths["output_csv"]
    AGE_CACHE_DIR = _paths["age_cache_dir"]

# Specific SIDRA category IDs for each census age table.
# Requesting only needed categories avoids the HTTP 500 that SIDRA returns
# when N6[all] is combined with a full [all] classification (too large).
# Categories: Total + 5-year groups for 0-4, 5-9, 10-14, 65-69, ..., 100+
_AGE_CATS_2022 = "100362,93070,93084,93085,93096,93097,93098,49108,49109,60040,60041,6653"
_AGE_CATS_2010 = "0,93070,93084,93085,93096,93097,93098,93099,93100,6653"


def _age_cache_path(tabela: int, year: int) -> str:
    return os.path.join(AGE_CACHE_DIR, f"census_age_{tabela}_{year}_raw.csv")


def _load_age_cache(tabela: int, year: int) -> pd.DataFrame | None:
    path = _age_cache_path(tabela, year)
    if os.path.exists(path):
        logging.info(f"Loading census age cache: {path}")
        return pd.read_csv(
            path,
            dtype={"municipio_code": int, "age_label": str},
            encoding="latin-1",
        )
    return None


def _save_age_cache(df: pd.DataFrame, tabela: int, year: int) -> None:
    path = _age_cache_path(tabela, year)
    df[["municipio_code", "age_label", "value"]].to_csv(
        path, index=False, encoding="latin-1"
    )
    logging.info(f"Saved census age cache: {path}")


def _parse_sidra_classified_response(data: list[dict], year: int, class_id: int) -> list[dict]:
    """Parse JSON response from SIDRA API for age-group data."""
    rows = []
    for obj in data:
        for resultado in obj.get("resultados", []):
            age_label = ""
            for cl in resultado.get("classificacoes", []):
                if str(cl.get("id", "")) == str(class_id):
                    cats_dict = cl.get("categoria", {})
                    age_label = next(iter(cats_dict.values()), "")
                    break
            
            for series_item in resultado.get("series", []):
                try:
                    mun_code = int(series_item["localidade"]["id"])
                except ValueError:
                    continue
                
                val_str = series_item["serie"].get(str(year), "")
                try:
                    val = float(val_str)
                except (ValueError, TypeError):
                    val = np.nan
                
                rows.append({
                    "municipio_code": mun_code,
                    "age_label": age_label,
                    "value": val,
                })
    return rows


def _make_sidra_request(url: str, retries: int, pause: float, context: str) -> requests.Response:
    """Helper to perform requests with exponential backoff for common IBGE errors."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=120)
            if resp.status_code in {429, 500, 503}:
                wait = (2 ** attempt) * pause
                logging.warning(
                    f"HTTP {resp.status_code} {context} "
                    f"(attempt {attempt+1}/{retries}); waiting {wait:.0f}s ..."
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries - 1:
                logging.error(f"Failed {context}: {exc}")
                raise
            time.sleep((2 ** attempt) * pause)
    
    raise requests.HTTPError(f"Failed {context} after {retries} attempts")


def _fetch_valid_munis(tabela: int) -> set[int]:
    """
    Return the set of municipality codes that are present in the given SIDRA table.
    Called once per census table to pre-filter our crosswalk so that municipalities
    created after the census (and thus absent from SIDRA) are excluded -- those codes
    cause HTTP 500 for any batch that includes them.
    """
    url = f"{SIDRA_BASE}/{tabela}/localidades/N6"
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return {int(item["id"]) for item in resp.json()}


def _fetch_age_batch(mun_batch: list[int], tabela: int, year: int, variavel: int,
                     class_id: int, categories: str,
                     retries: int = 4, pause: float = 3.0) -> pd.DataFrame:
    """
    Fetch census age data for a batch of municipalities with specific category IDs.
    Batching is required because N6[all] with any classification causes HTTP 500.
    Returns DataFrame with columns: municipio_code (int), age_label (str), value (float).
    """
    mun_str = ",".join(str(m) for m in mun_batch)
    url = (
        f"{SIDRA_BASE}/{tabela}/periodos/{year}/variaveis/{variavel}"
        f"?localidades=N6[{mun_str}]&classificacao={class_id}[{categories}]"
    )
    context = f"table {tabela} year {year}"
    resp = _make_sidra_request(url, retries, pause, context)

    rows = _parse_sidra_classified_response(resp.json(), year, class_id)
    return (pd.DataFrame(rows) if rows
            else pd.DataFrame(columns=["municipio_code", "age_label", "value"]))

## -----------------------------------------------------------------------------
## 2) SIDRA API HELPERS
## -----------------------------------------------------------------------------

def _sidra_url(tabela: int, periodos: str, variavel: int) -> str:
    return (
        f"{SIDRA_BASE}/{tabela}/periodos/{periodos}"
        f"/variaveis/{variavel}?localidades=N6[all]"
    )


def _parse_sidra_response(data: list[dict], frames: list[dict]) -> None:
    """Parse JSON response from SIDRA API for simple variables."""
    for obj in data:
        for result in obj.get("resultados", []):
            for series_item in result.get("series", []):
                try:
                    mun_code = int(series_item["localidade"]["id"])
                except ValueError:
                    continue

                for periodo, val_str in series_item["serie"].items():
                    try:
                        val = float(val_str)
                    except (ValueError, TypeError):
                        val = np.nan
                    frames.append({
                        "municipio_code": mun_code,
                        "year": int(periodo),
                        "value": val,
                    })


def fetch_sidra(tabela: int, years: list[int], variavel: int,
                retries: int = 3, pause: float = 2.0) -> pd.DataFrame:
    """
    Fetch all municipalities for the given table / variable / years from SIDRA.
    Returns a tidy DataFrame with columns: municipio_code (int), year (int), value (float).
    Handles HTTP 429 / 503 with exponential back-off.
    Batches years in groups of 8 to avoid URL length limits.
    """
    frames = []
    batch_size = 8

    for i in range(0, len(years), batch_size):
        batch = years[i: i + batch_size]
        periodos = "|".join(str(y) for y in batch)
        url = _sidra_url(tabela, periodos, variavel)

        context = f"fetching table {tabela} years {batch}"
        resp = _make_sidra_request(url, retries, pause, context)

        data = resp.json()
        _parse_sidra_response(data, frames)

        logging.info(f"Table {tabela} var {variavel} years {batch}: {len(frames)} cumulative rows")
        time.sleep(pause)   # polite pacing between batches

    if not frames:
        return pd.DataFrame(columns=["municipio_code", "year", "value"])

    df = pd.DataFrame(frames)
    df = df.drop_duplicates(subset=["municipio_code", "year"])
    return df


## -----------------------------------------------------------------------------
## 3) POPULATION DATA
##    Primary source : Table 6579 (Estimativas de populacao), Variable 9324
##                     Coverage: 2013-2022 in this series
##    Extended to 2023+ via Table 4714 (Projeções da Populacao), Variable 93
## -----------------------------------------------------------------------------

def fetch_population(years: list[int]) -> pd.DataFrame:
    """
    Return municipal population panel for the requested years.
    Stitches together Table 6579 (main estimates) and Table 4714 (projections).
    Falls back to Table 6579 last available year for any year still missing.
    """
    # Table 6579 covers annual estimates; newer estimates are in table 4714.
    # We try 6579 first, then supplement missing years with 4714.
    logging.info("Fetching population -- Table 6579 ...")
    pop_6579 = fetch_sidra(tabela=6579, years=years, variavel=9324)
    pop_6579["source"] = "6579"

    found_years = set(pop_6579["year"].unique())

    if missing := [y for y in years if y not in found_years]:
        logging.info(f"Population missing for {missing}; trying Table 9514/4714 ...")
        try:
            pop_ext = fetch_sidra(tabela=4714, years=missing, variavel=93)
            pop_ext["source"] = "4714"
            pop = pd.concat([pop_6579, pop_ext], ignore_index=True)
        except Exception as e:
            logging.warning(f"Extension table failed ({e}); using carry-forward.")
            pop = pop_6579.copy()
    else:
        pop = pop_6579.copy()

    pop = pop.drop_duplicates(subset=["municipio_code", "year"])
    pop.rename(columns={"value": "population"}, inplace=True)
    pop = pop[["municipio_code", "year", "population"]]

    # Carry forward last known population for any years still missing
    # (IBGE publishes estimates with a lag; e.g. 2023 may not yet be in any table)
    if still_missing := [
        y for y in years if y not in set(pop["year"].unique())
    ]:
        last_known_year = int(pop.dropna(subset=["population"])["year"].max())
        logging.info(
            f"Population still missing for {still_missing}; "
            f"carrying forward from {last_known_year}."
        )
        base = pop[pop["year"] == last_known_year][["municipio_code", "population"]].copy()
        extra_frames = []
        for yr in still_missing:
            tmp = base.copy()
            tmp["year"] = yr
            extra_frames.append(tmp)
        pop = pd.concat([pop] + extra_frames, ignore_index=True)
        pop = pop.drop_duplicates(subset=["municipio_code", "year"])

    return pop


## -----------------------------------------------------------------------------
## 4) GDP DATA
##    Source: Table 5938 (PIB dos Municípios), Variable 37
##            = "Produto interno bruto a preços correntes (Mil Reais)"
##    Coverage: 2002-2021 (published with ~2 year lag)
##    Years beyond last available: carry forward last observed value.
## -----------------------------------------------------------------------------

def fetch_gdp(years: list[int]) -> pd.DataFrame:
    """
    Return municipal GDP panel (total, R$ thousands, current prices).
    Automatically marks imputed (carry-forward) rows.
    """
    logging.info("Fetching GDP -- Table 5938 ...")
    gdp_raw = fetch_sidra(tabela=5938, years=years, variavel=37)
    gdp_raw.rename(columns={"value": "gdp_total_r1000"}, inplace=True)
    gdp_raw["gdp_imputed"] = False

    # Identify the highest published year with real data
    max_real_year = gdp_raw.dropna(subset=["gdp_total_r1000"])["year"].max()

    if missing_gdp_years := [y for y in years if y > max_real_year]:
        logging.info(
            f"GDP data not available for {missing_gdp_years}; "
            f"carrying forward from {max_real_year}."
        )
        base = gdp_raw[gdp_raw["year"] == max_real_year][["municipio_code", "gdp_total_r1000"]].copy()
        extra_frames = []
        for yr in missing_gdp_years:
            tmp = base.copy()
            tmp["year"] = yr
            tmp["gdp_imputed"] = True
            extra_frames.append(tmp)
        gdp_raw = pd.concat([gdp_raw] + extra_frames, ignore_index=True)

    gdp_raw = gdp_raw.drop_duplicates(subset=["municipio_code", "year"])
    return gdp_raw[["municipio_code", "year", "gdp_total_r1000", "gdp_imputed"]]


## -----------------------------------------------------------------------------
## 5) AGE STRUCTURE FROM CENSUS TABLES
##    Source:
##      Table 9514 (2022 Census): "Pessoas residentes em domicílios particulares
##        ocupados, por grupos de idade" -- Classification 287, Variable 93
##      Table 136  (2010 Census): "Pessoas residentes, por grupos de idade"
##        -- Classification 2, Variable 93
##    Strategy:
##      * Fetch all age-group categories for 2022 and 2010.
##      * Identify 65+ and 0-14 buckets by matching category labels.
##      * Compute municipality-level fraction_65plus and fraction_young.
##      * Linearly interpolate for 2013-2024.
## -----------------------------------------------------------------------------

def fetch_sidra_classified(tabela: int, year: int, variavel: int, class_id: int,
                            retries: int = 5, pause: float = 5.0) -> pd.DataFrame:
    """
    Fetch municipality x age-group data from a SIDRA census table.
    Returns DataFrame with columns: municipio_code (int), age_label (str), value (float).
    The 'classificacao=<class_id>[all]' parameter requests every category; each
    resultados entry corresponds to one category.
    """
    url = (
        f"{SIDRA_BASE}/{tabela}/periodos/{year}/variaveis/{variavel}"
        f"?localidades=N6[all]&classificacao={class_id}[all]"
    )
    context = f"table {tabela} age classification"
    resp = _make_sidra_request(url, retries, pause, context)

    rows = _parse_sidra_classified_response(resp.json(), year, class_id)

    logging.info(f"Table {tabela} year {year}: {len(rows):,} age x municipality rows")
    return pd.DataFrame(rows)


def _classify_age_bucket(label: str) -> str | None:
    """
    Map a SIDRA age-group label to one of 'young', 'senior', or None (other).
    Handles both Portuguese bracket labels and the catch-all "65 anos ou mais".
    """
    lbl = label.lower().strip()
    # Exact senior catch-all label
    if "65 anos ou mais" in lbl or "65 ou mais" in lbl:
        return "senior"
    # Bracket-form labels -- any bracket starting at or above 65
    for age_start in (65, 70, 75, 80, 85, 90, 95, 100):
        if f"{age_start} a " in lbl or f"{age_start} anos" in lbl:
            return "senior"
    # Young: 0-14
    if any(bracket in lbl for bracket in ("0 a 4", "5 a 9", "10 a 14")):
        return "young"
    return None


def fetch_age_structure() -> pd.DataFrame:
    """
    Fetch age-group population by municipality for 2010 and 2022 census years.
    Returns DataFrame: municipio_code (int), year (int), fraction_65plus (float),
                       fraction_young (float)
    Missing municipalities (no data) are dropped; fractions are bounded [0, 1].

    Uses targeted SIDRA category IDs (5-year groups only) and batches municipalities
    in groups of 100 to avoid the HTTP 500 that N6[all] triggers on large tables.
    """
    census_specs = [
        # (tabela, year, variavel, class_id, categories)
        (9514, 2022, 93, 287, _AGE_CATS_2022),   # 2022 Census age groups
        (1552, 2010, 93, 287, _AGE_CATS_2010),   # 2010 Census age groups
    ]

    # Load all municipality codes from the crosswalk once
    all_munis = (
        pd.read_csv(MCA_CSV, usecols=["municipality_code"],
                    dtype={"municipality_code": int})["municipality_code"]
        .unique()
        .tolist()
    )
    batch_size = 100

    all_frames = []
    for spec in census_specs:
        tabela, year, variavel, class_id, categories = spec

        # 1. Check local cache first
        cached = _load_age_cache(tabela, year)
        if cached is not None:
            cached["census_year"] = year
            all_frames.append(cached)
            continue

        # Load municipality list and filter to those present in this census table
        # (municipalities created after the census cause HTTP 500 for any batch
        # that includes them -- pre-filtering eliminates all such errors).
        try:
            valid_munis_in_table = _fetch_valid_munis(tabela)
            excluded = [m for m in all_munis if m not in valid_munis_in_table]
            if excluded:
                logging.info(
                    f"Table {tabela}: excluding {len(excluded)} municipalities "
                    f"not in this census (e.g. post-census creations): {excluded[:5]}"
                )
            fetch_munis = [m for m in all_munis if m in valid_munis_in_table]
        except Exception as exc:
            logging.warning(f"Could not fetch valid muni list for table {tabela}: {exc}. Using all.")
            fetch_munis = all_munis

        # 3. Fetch via SIDRA in municipality batches
        logging.info(f"Fetching census age structure -- Table {tabela} ({year}) ...")
        batch_frames = []
        n_batches = (len(fetch_munis) + batch_size - 1) // batch_size
        for i in range(0, len(fetch_munis), batch_size):
            batch = fetch_munis[i: i + batch_size]
            batch_num = i // batch_size + 1
            try:
                df_batch = _fetch_age_batch(batch, tabela, year, variavel,
                                             class_id, categories)
                batch_frames.append(df_batch)
            except Exception as exc:
                logging.warning(
                    f"Table {tabela} ({year}) batch {batch_num}/{n_batches} failed: {exc}"
                )
            if i + batch_size < len(all_munis):
                time.sleep(0.5)   # polite pacing between batches

        if not batch_frames:
            logging.error(
                f"Skipping Table {tabela} ({year}): all batches failed.\n"
                f"  Tip: place a manually-downloaded CSV at "
                f"{_age_cache_path(tabela, year)} to bypass the API."
            )
            continue

        df = pd.concat(batch_frames, ignore_index=True)
        df["census_year"] = year
        _save_age_cache(df, tabela, year)
        all_frames.append(df)
        logging.info(f"Table {tabela} ({year}): {len(df):,} rows fetched.")
        time.sleep(3)   # pause between the two large census table requests

    if not all_frames:
        logging.warning("No census age data retrieved; age structure will be NaN.")
        return pd.DataFrame(columns=["municipio_code", "year",
                                     "fraction_65plus", "fraction_young"])

    census_df = pd.concat(all_frames, ignore_index=True)

    # Label each row as "senior", "young", "total", or "other"
    census_df["bucket"] = census_df["age_label"].apply(_classify_age_bucket)

    # Identify the "Total" row (label is blank or exactly "Total")
    def is_total(lbl: str) -> bool:
        l = lbl.lower().strip()
        return l in {"", "total", "total da população"}

    census_df["is_total"] = census_df["age_label"].apply(is_total)

    results = []
    for (mun_code, census_year), g in census_df.groupby(["municipio_code", "census_year"]):
        total_rows = g[g["is_total"]]["value"]
        senior_rows = g[g["bucket"] == "senior"]["value"]
        young_rows  = g[g["bucket"] == "young"]["value"]

        total  = total_rows.sum() if len(total_rows) > 0 else g["value"].sum()
        senior = senior_rows.sum()
        young  = young_rows.sum()

        if total <= 0 or np.isnan(total):
            continue

        results.append({
            "municipio_code": mun_code,
            "year":           census_year,
            "fraction_65plus": min(1.0, senior / total),
            "fraction_young":  min(1.0, young  / total),
        })

    age_df = pd.DataFrame(results)
    logging.info(
        f"Age structure: {len(age_df):,} municipality x census-year observations."
    )
    return age_df


def interpolate_age_structure(age_census: pd.DataFrame,
                               years: list[int]) -> pd.DataFrame:
    """
    Linearly interpolate fraction_65plus and fraction_young for all non-census years
    in `years`, using 2010 and 2022 as anchor points.

    Returns DataFrame: municipio_code, year, fraction_65plus, fraction_young,
                       age_interpolated (True for non-census years).
    """
    if age_census.empty:
        # Return NaN-filled stub so the rest of the pipeline still runs
        mca = pd.read_csv(MCA_CSV, usecols=["municipality_code"],
                          dtype={"municipality_code": int}).drop_duplicates()
        stub_rows = []
        for yr in years:
            tmp = mca.copy()
            tmp["year"] = yr
            tmp["fraction_65plus"] = np.nan
            tmp["fraction_young"]  = np.nan
            tmp["age_interpolated"] = True
            stub_rows.append(tmp)
        return pd.concat(stub_rows, ignore_index=True).rename(
            columns={"municipality_code": "municipio_code"})

    census_years = sorted(age_census["year"].unique())  # typically [2010, 2022]
    y_lo = min(census_years)
    y_hi = max(census_years)

    munis = age_census["municipio_code"].unique()
    rows  = []

    # Pivot to wide so we have one row per municipality
    wide = age_census.pivot(index="municipio_code", columns="year",
                            values=["fraction_65plus", "fraction_young"])
    wide.columns = [f"{v}_{y}" for v, y in wide.columns]
    wide = wide.reset_index()

    for yr in years:
        is_census = yr in census_years
        tmp = wide.copy()
        tmp["year"] = yr

        if is_census:
            tmp["fraction_65plus"]  = tmp.get(f"fraction_65plus_{yr}", np.nan)
            tmp["fraction_young"]   = tmp.get(f"fraction_young_{yr}", np.nan)
            tmp["age_interpolated"] = False
        else:
            # Linear interpolation weight
            y_lo_c = max((cy for cy in census_years if cy <= yr), default=y_lo)
            y_hi_c = min((cy for cy in census_years if cy >= yr), default=y_hi)

            w = 1.0 if y_lo_c == y_hi_c else (yr - y_lo_c) / (y_hi_c - y_lo_c)  # weight on y_hi_c

            col_lo_65 = f"fraction_65plus_{y_lo_c}"
            col_hi_65 = f"fraction_65plus_{y_hi_c}"
            col_lo_yg = f"fraction_young_{y_lo_c}"
            col_hi_yg = f"fraction_young_{y_hi_c}"

            def safe_interp(lo, hi, w):
                if lo in tmp.columns and hi in tmp.columns:
                    return tmp[lo] * (1 - w) + tmp[hi] * w
                elif lo in tmp.columns:
                    return tmp[lo]
                elif hi in tmp.columns:
                    return tmp[hi]
                else:
                    return np.nan

            tmp["fraction_65plus"]  = safe_interp(col_lo_65, col_hi_65, w)
            tmp["fraction_young"]   = safe_interp(col_lo_yg, col_hi_yg, w)
            tmp["age_interpolated"] = True

        rows.append(tmp[["municipio_code", "year",
                          "fraction_65plus", "fraction_young", "age_interpolated"]])

    result = pd.concat(rows, ignore_index=True)
    return result


## -----------------------------------------------------------------------------
## 6) AGGREGATE MUNICIPALITY -> MCA
## -----------------------------------------------------------------------------

def aggregate_to_mca(pop: pd.DataFrame, gdp: pd.DataFrame,
                     age: pd.DataFrame) -> pd.DataFrame:
    """
    1. Loads the MCA crosswalk (municipality_code x year -> mca_code).
    2. Merges population, GDP, and age structure onto it.
    3. Aggregates to MCA x year:
         pop_total        = sum of constituent municipality populations
         gdp_total_r1000  = sum of constituent municipal GDPs in R$ thousands
         gdp_per_capita   = (gdp_total_r1000 * 1000) / pop_total   [in R$]
         fraction_65plus  = population-weighted mean of municipality fraction_65plus
         fraction_young   = population-weighted mean of municipality fraction_young
         n_municipalities = count of distinct municipalities
         gdp_imputed      = True if any constituent municipality used carry-forward GDP
         age_interpolated = True if any constituent municipality used interpolated age
    """
    logging.info("Loading MCA crosswalk ...")
    mca = pd.read_csv(MCA_CSV, usecols=["municipality_code", "mca_code", "year"],
                      dtype={"municipality_code": int, "mca_code": str, "year": int})
    mca = mca[mca["year"].between(START_YEAR, END_YEAR)].copy()

    # Merge population
    merged = mca.merge(pop, left_on=["municipality_code", "year"],
                       right_on=["municipio_code", "year"], how="left")
    merged.drop(columns=["municipio_code"], errors="ignore", inplace=True)

    # Merge GDP
    merged = merged.merge(gdp, left_on=["municipality_code", "year"],
                          right_on=["municipio_code", "year"], how="left")
    merged.drop(columns=["municipio_code"], errors="ignore", inplace=True)

    # Merge age structure
    if not age.empty:
        merged = merged.merge(age, left_on=["municipality_code", "year"],
                              right_on=["municipio_code", "year"], how="left")
        merged.drop(columns=["municipio_code"], errors="ignore", inplace=True)
    else:
        merged["fraction_65plus"]  = np.nan
        merged["fraction_young"]   = np.nan
        merged["age_interpolated"] = True

    # ── Save municipality-level intermediate for within-MCA σ computation ──
    muni_gdp_pc = np.where(
        merged["population"] > 0,
        (merged["gdp_total_r1000"] * 1000) / merged["population"],
        np.nan,
    )
    muni_save = merged[["municipality_code", "mca_code", "year",
                         "population", "gdp_total_r1000",
                         "fraction_65plus", "fraction_young"]].copy()
    muni_save["gdp_per_capita"] = muni_gdp_pc
    muni_out = os.path.join(IBGE_DIR, "muni_demographics_panel.csv")
    muni_save.to_csv(muni_out, index=False)
    logging.info(f"Saved municipality-level demographics to {muni_out}")

    # Population-weighted helpers for age shares.
    # Multiply population only where the fraction is non-NaN, so that MCAs with
    # completely missing age data produce NaN (not 0.0) in the output.
    merged["pop_x_65plus"]    = merged["population"] * merged["fraction_65plus"]
    merged["pop_x_young"]     = merged["population"] * merged["fraction_young"]
    # Track how much population actually had age data (denominator for weighted mean)
    merged["pop_with_age"]    = np.where(merged["fraction_65plus"].notna(),
                                          merged["population"], np.nan)

    # Aggregate
    agg = (
        merged.groupby(["mca_code", "year"])
              .agg(
                  pop_total        = ("population",      "sum"),
                  gdp_total_r1000  = ("gdp_total_r1000", "sum"),
                  n_municipalities = ("municipality_code", "nunique"),
                  gdp_imputed      = ("gdp_imputed",     lambda x: x.any()),
                  age_interpolated = ("age_interpolated", lambda x: x.any()),
                  pop_x_65plus_sum = ("pop_x_65plus",    "sum"),
                  pop_x_young_sum  = ("pop_x_young",     "sum"),
                  pop_with_age_sum = ("pop_with_age",    "sum"),
              )
              .reset_index()
    )

    # Population-weighted age fractions; NaN when no municipality had age data
    agg["fraction_65plus"] = np.where(
        agg["pop_with_age_sum"] > 0,
        agg["pop_x_65plus_sum"] / agg["pop_with_age_sum"],
        np.nan,
    )
    agg["fraction_young"] = np.where(
        agg["pop_with_age_sum"] > 0,
        agg["pop_x_young_sum"] / agg["pop_with_age_sum"],
        np.nan,
    )
    agg.drop(columns=["pop_x_65plus_sum", "pop_x_young_sum", "pop_with_age_sum"],
             inplace=True)

    # GDP per capita in R$
    agg["gdp_per_capita"] = np.where(
        agg["pop_total"] > 0,
        (agg["gdp_total_r1000"] * 1000) / agg["pop_total"],
        np.nan,
    )

    # Replace 0 aggregates with NaN (no data)
    for col in ["pop_total", "gdp_total_r1000"]:
        agg.loc[agg[col] == 0, col] = np.nan

    # Reorder columns
    col_order = ["mca_code", "year", "pop_total", "gdp_total_r1000", "gdp_per_capita",
                 "fraction_65plus", "fraction_young",
                 "n_municipalities", "gdp_imputed", "age_interpolated"]
    agg = agg[col_order]

    agg.sort_values(["mca_code", "year"], inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg


## -----------------------------------------------------------------------------
## 7) MAIN
## -----------------------------------------------------------------------------

def main():
    # Early exit: if output already exists AND muni intermediate exists, skip
    muni_csv = os.path.join(IBGE_DIR, "muni_demographics_panel.csv")
    if (os.path.exists(OUTPUT) and os.path.getsize(OUTPUT) > 0
            and os.path.exists(muni_csv) and os.path.getsize(muni_csv) > 0):
        print(f"Output already exists, skipping: {OUTPUT}")
        logging.info(f"Output already exists -- skipping rebuild: {OUTPUT}")
        return

    years = list(range(START_YEAR, END_YEAR + 1))

    # Fetch population and GDP concurrently (different SIDRA tables -> safe to
    # parallelise).  Age structure is fetched AFTER they finish: the batched
    # age requests generate heavy API load and interleaving them with pop/GDP
    # requests causes intermittent HTTP 500 errors from SIDRA.
    logging.info("Fetching population and GDP in parallel ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future_pop = pool.submit(fetch_population, years)
        future_gdp = pool.submit(fetch_gdp,        years)
        pop        = future_pop.result()
        gdp        = future_gdp.result()

    logging.info("Fetching census age structure (sequential, after pop/GDP) ...")
    age_census = fetch_age_structure()

    logging.info(f"Population rows: {len(pop):,}  years: {sorted(pop['year'].unique())}")
    logging.info(f"GDP rows: {len(gdp):,}  years: {sorted(gdp['year'].unique())}")

    age = interpolate_age_structure(age_census, years)
    logging.info(f"Age structure rows: {len(age):,}  interpolated: {age['age_interpolated'].sum():,}")

    panel = aggregate_to_mca(pop, gdp, age)
    logging.info(f"MCA panel: {len(panel):,} rows  ({panel['mca_code'].nunique()} MCAs)")

    import pyarrow as pa
    import pyarrow.csv as pa_csv
    pa_csv.write_csv(pa.Table.from_pandas(panel, preserve_index=False), OUTPUT)
    logging.info(f"Saved to {OUTPUT}")

    print(
        f"\nDemographics panel summary"
        f"\n  Rows:              {len(panel):,}"
        f"\n  MCAs:              {panel['mca_code'].nunique()}"
        f"\n  Years:             {panel['year'].min()} - {panel['year'].max()}"
        f"\n  GDP imputed:       {panel['gdp_imputed'].sum():,} rows"
        f"\n  Age interpolated:  {panel['age_interpolated'].sum():,} rows"
        f"\n  fraction_65plus:   {panel['fraction_65plus'].notna().sum():,} non-null"
        f"\n  Output:            {OUTPUT}"
    )
    print(panel.head(10).to_string(index=False))



# --- MAIN FUNCTIONS ---


def rectangularize_mca_main():
    input_json_file = output_dir / "muni_mca_regions_2010_2024.json"
    output_csv_file = output_dir / "muni_mca_regions_2010_2024_panel.csv"
    
    state_look_up_dict = load_uf_lookup()
    municipality_lookup_dict = load_municipality_lookup()
    
    if state_look_up_dict and municipality_lookup_dict: # Only proceed if lookup was created.
        panel_df = rectangularize_mca_json(input_json_file, state_look_up_dict, municipality_lookup_dict)

        print("\nSuccessfully created rectangular panel DataFrame.")
        print("DataFrame Head:")
        print(panel_df.head())
        
        print("\nDataFrame Info:")
        panel_df.info()
        
        # Save to CSV
        panel_df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        print(f"\nPanel data successfully saved to {output_csv_file}")
    else:
        print("State lookup failed. Panel DataFrame not created.")

def ibge_demographics_main():
    main()


if __name__ == "__main__":
    from pathlib import Path
    try:
        root_dir = Path(__file__).parent.parent
    except NameError:
        root_dir = Path.cwd()
    
    output_dir = paths.IBGE_DIR
    output_csv_file = output_dir / "muni_mca_regions_2010_2024_panel.csv"
    
    if output_csv_file.exists():
        print(f"\n--- SKIPPING MCA JSON PREP: {output_csv_file.name} already exists ---")
    else:
        print("\n--- GENERATING MCA JSON ---")
        generate_mca_main()
        print("\n--- RECTANGULARIZING JSON ---")
        rectangularize_mca_main()

    print("\n--- BUILDING DEMOGRAPHICS PANEL ---")
    ibge_demographics_main()
