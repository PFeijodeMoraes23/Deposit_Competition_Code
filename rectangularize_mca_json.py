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

import pandas as pd
import json
from pathlib import Path
import geobr

try:
    root_dir = Path(__file__).parent.parent
except NameError:
    root_dir = Path.cwd()
output_dir = root_dir / "IBGE"
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

if __name__ == "__main__":
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