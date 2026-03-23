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
output_dir = root_dir / "IBGE"

output_dir.mkdir(parents = True, exist_ok=True)

# 2) User-defined functions:

def get_closest_predecessor(target_year, available_years, classification_name):
    """
    Returns most recent available year less than or equal target_year from the list of available years.
    """
    valid_years = [y for y in available_years if y <= target_year]
    
    if valid_years:
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
         
# 3) Data Preparation:

START_YEAR = 2010
END_YEAR = 2024
MCA_REFERENCE = 2010
YEARS = list(range(START_YEAR, END_YEAR + 1))
final_json_file = output_dir / f"muni_mca_regions_{START_YEAR}_{END_YEAR}.json"

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
        
        if not all(col in low_level_df.columns for col in low_cols):
             print(f"  > Warning: Missing columns in low_level_df for {year}. Expected: {low_cols}")
             continue
        if not all(col in high_level_df.columns for col in high_cols):
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