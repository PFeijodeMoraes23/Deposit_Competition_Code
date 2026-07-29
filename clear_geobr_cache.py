# clear_geobr_cache.py
# Clear geobr cache
# Author: Pedro Feijo de Moraes
# Last edited: 2025-11-11
# Adopted 2026-07-29 from Code/_utils/ (the geobr consumer, scrape_4_ibge_demographics.py,
# lives in this repo).
#------------------------------------------------------------------------------------

import os
import appdirs

# Setting up
cache_dir = appdirs.user_cache_dir('geobr', 'ipeagit')
cache_file_path = os.path.join(cache_dir, 'geobr_cache.sqlite')

if os.path.exists(cache_file_path):
    os.remove(cache_file_path)
    print(f"Successfully deleted the geobr cache file: {cache_file_path}")
else:
    print("geobr cache file was not found or has already been cleared.")

try:
    os.rmdir(cache_dir)
    print(f"Removed empty cache directory: {cache_dir}")
except OSError:
    print(f"Cache directory {cache_dir} is not empty and was not removed.")
