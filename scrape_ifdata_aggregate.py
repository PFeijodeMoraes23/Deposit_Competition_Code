# scrape_ifdata_aggregate.py
# Last edited: 2026-06-02
# -----------------------------------------------------------------------------
# Build the IF-Data "Aggregated Data" reports (IF_DATA_type_<t>_report_<r>.csv)
# from the per-period IF_DATA_Values_<YYYY>_<Q>.csv files that scrape_1 downloads
# into the Prudential/Financial/Individual subfolders.
#
# This is a faithful port of the now-DEPRECATED Code/if_data_process_1.py (which
# was a standalone script hardcoded to 2016-2024 and not part of the pipeline).
# The transform functions are copied verbatim; only the driver changed:
#   * paths derived from the repo BASE (../../BCB/IF Data) instead of absolute;
#   * the year range auto-detects the latest available period (2016..max), so
#     the reports extend automatically as scrape_1 pulls new quarters;
#   * the existing "Aggregated Data" folder is backed up once before overwrite;
#   * --verify compares the regenerated type-1 reports against the backup on the
#     overlapping (Year,Month) cells and reports ADDED periods vs CHANGED values.
#
# Consumers: panel_deposits.py and panel_bank_chars.py read
#   Aggregated Data/IF_DATA_type_1_report_<r>.csv  (type 1 = Prudential Cong.).
# (panel_deposit_rates.py reads the per-period Prudential files directly, not these.)
#
# Usage:
#     python scrape_ifdata_aggregate.py            # rebuild Aggregated Data
#     python scrape_ifdata_aggregate.py --verify   # rebuild + overlap report
# -----------------------------------------------------------------------------
import argparse
import glob
import os
import re
import shutil
import string
import unicodedata

import numpy as np
import pandas as pd

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

from utils import paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_FOLDER = str(paths.IF_DATA_ROOT)

# Lower bound of the aggregation window. Set to 2013 to match the ESTBAN/deposits
# start (run_estban(..., estban_start_year=2013) in scrape_1), so bank characteristics
# span the same years as the deposit panel. The legacy builder used 2016, which left
# 2013-2015 deposits with no matching bank characteristics.
# NOTE: the Conglomerado Prudencial (type 1) did not exist before 2014 — the API returns
# empty for 2013 — so type-1 reports legitimately start in 2014 while types 2/3 start 2013.
START_YEAR = 2013


# ===== Transform functions (verbatim from if_data_process_1.py) ==============

def normalize_string(s):
    """Normalize strings by removing accents, special characters, and spaces."""
    if pd.isna(s):
        return s
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ''.join(c for c in s if ord(c) < 128)
    return s.strip()


def normalize_string_parentheses(s):
    """Normalize strings, but keep parentheses."""
    if pd.isna(s):
        return s
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = s.translate(str.maketrans('', '', string.punctuation.replace('()', '')))
    s = ''.join(c for c in s if ord(c) < 128)
    return s.strip()


def setup_directories(main_folder, folder_names):
    folders = {key: os.path.join(main_folder, value) for key, value in folder_names.items()}
    for folder_path in folders.values():
        os.makedirs(folder_path, exist_ok=True)
    return folders


def create_merge_key(df):
    return df['CNPJ'].astype(str) + '_' + df['Year'].astype(str) + '_' + df['Month'].astype(str)


def process_list_dataframe(df):
    df = df.replace(['null', np.nan], pd.NA)
    df.rename(columns={'CodInst': 'CNPJ', 'CnpjInstituicaoLider': 'CNPJ_Lider'}, inplace=True)
    df['CNPJ'] = df['CNPJ'].astype(str).str.zfill(8)
    df['CNPJ_Lider'] = df['CNPJ_Lider'].apply(
        lambda x: str(int(x)).zfill(8) if pd.notna(x) and str(x).strip() != '' else pd.NA
    )
    df['Data'] = pd.to_datetime(df['Data'], format='%Y%m')
    df['DataInicioAtividade'] = pd.to_datetime(df['DataInicioAtividade'], format='%Y%m', errors='coerce')
    df['Year'] = df['Data'].dt.year
    df['Month'] = df['Data'].dt.month
    df['Year_Inicio'] = df['DataInicioAtividade'].dt.year
    df['Month_Inicio'] = df['DataInicioAtividade'].dt.month
    df['merge_key'] = create_merge_key(df)
    text_columns = ['NomeInstituicao', 'SegmentoTb', 'Atividade', 'Municipio']
    df[text_columns] = df[text_columns].apply(lambda col: col.map(normalize_string))
    return df


def process_list_files(list_folder, start_year, end_year):
    list_frames = []
    for year in range(start_year, end_year + 1):
        for quarter in [3, 6, 9, 12]:
            file_name = f"IF_DATA_List_{year}_{quarter}.csv"
            file_path = os.path.join(list_folder, file_name)
            if os.path.exists(file_path):
                print(f"Processing file: {file_name}")
                try:
                    df = pd.read_csv(file_path, sep=",", encoding='latin1')
                    df = process_list_dataframe(df)
                    list_frames.append(df)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
            else:
                print(f"File not found: {file_name}.")
    if list_frames:
        return pd.concat(list_frames, ignore_index=True)
    print("No valid list files were found.")
    return pd.DataFrame()


def process_values_dataframe(df, list_quarterly):
    df = df.replace(['null', np.nan], pd.NA)
    df.rename(columns={'CodInst': 'CNPJ', 'AnoMes': 'Data', 'Conta': 'NumeroConta'}, inplace=True)
    df['CNPJ'] = df['CNPJ'].astype(str).str.zfill(8)
    df['Data'] = pd.to_datetime(df['Data'], format='%Y%m', errors='coerce')
    if df['Data'].isna().any():
        print("Warning, Some rows in 'Data' could not be converted to datetime and will be dropped.")
        df = df.dropna(subset=['Data'])
    df['Year'] = df['Data'].dt.year.astype('int16')
    df['Month'] = df['Data'].dt.month.astype('int8')
    df.drop(columns=['Data'], inplace=True)
    df['merge_key'] = create_merge_key(df)
    df['NumeroConta'] = df['NumeroConta'].astype(str).str.zfill(5)
    if 'Saldo' in df.columns:
        df.rename(columns={'Saldo': 'Value'}, inplace=True)
        df['Value'] = df['Value'].str.replace(',', '.', regex=False)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce', downcast='float')
    text_columns = ['NomeRelatorio', 'NomeColuna']
    df[text_columns] = df[text_columns].apply(lambda col: col.map(normalize_string_parentheses))
    # Defensive: BCB's Olinda API occasionally returns the same row multiple times
    # (e.g. Financial 2025_9 came back ~3x). Drop exact-duplicate value rows so the
    # aggregates are never inflated, regardless of which type/period is affected.
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        print(f"  dropped {before - len(df):,} exact-duplicate input rows")
    df = df.merge(list_quarterly, on='merge_key', how='left', suffixes=('', '_List'))
    redundant_columns = [col for col in df.columns if col.endswith('_List')]
    df.drop(columns=redundant_columns, inplace=True)
    if 'merge_key' in df.columns:
        df.drop(columns=['merge_key'], inplace=True)
    if 'Data' in df.columns:
        df.drop(columns=['Data'], inplace=True)
    return df


def get_file_path(inst_type, file_name, folders):
    if inst_type == 1:
        return os.path.join(folders['prud_cong'], file_name), "Prudential Conglomerates"
    elif inst_type == 2:
        return os.path.join(folders['fin_cong'], file_name), "Financial Conglomerates"
    else:
        return os.path.join(folders['ind_inst'], file_name), "Individual Institutions"


def save_type_report_frames(type_report_frames, output_folder, chunk_size=100000):
    for type_report, frames in type_report_frames.items():
        if frames:
            try:
                inst_type, report_id = type_report.split('_')[1], type_report.split('_')[3]
                output_file_name = f"IF_DATA_type_{inst_type}_report_{report_id}.csv"
                output_file_path = os.path.join(output_folder, output_file_name)
                with open(output_file_path, 'w', encoding='utf-8', newline='') as f:
                    for i, frame in enumerate(frames):
                        for j in range(0, len(frame), chunk_size):
                            chunk = frame.iloc[j:j + chunk_size]
                            chunk.to_csv(f, index=False, header=(i == 0 and j == 0), sep=',', lineterminator='\n')
                print(f"Saved IF_DATA_type_{inst_type}_report_{report_id}.csv")
            except Exception as e:
                print(f"Error saving {type_report}: {e}")


def process_main_loop(inst_types, reports_list, list_quarterly, folders, start_year, end_year):
    type_report_frames = {f"type_{t}_report_{r}": [] for t in inst_types for r in reports_list}
    for year in range(start_year, end_year + 1):
        for quarter in [3, 6, 9, 12]:
            file_name = f"IF_DATA_Values_{year}_{quarter}.csv"
            for inst_type in inst_types:
                file_path, folder_name = get_file_path(inst_type, file_name, folders)
                if os.path.exists(file_path):
                    print(f"Processing file: {file_name} for type {inst_type}.")
                    try:
                        df = pd.read_csv(
                            file_path, sep=",", encoding='latin1',
                            dtype={'NomeRelatorio': str, 'NumeroRelatorio': 'Int64',
                                   'AnoMes': str, 'Conta': str, 'CodInst': str, 'Grupo': str},
                            low_memory=True)
                        df = process_values_dataframe(df, list_quarterly)
                        missing_reports = []
                        for report_id in reports_list:
                            df_report = df[df['NumeroRelatorio'] == report_id]
                            if df_report.empty:
                                missing_reports.append(report_id)
                                continue
                            type_report_frames[f"type_{inst_type}_report_{report_id}"].append(df_report)
                        if missing_reports:
                            print(f"Missing reports for type {inst_type} in {file_name}: "
                                  f"{', '.join(map(str, missing_reports))}.")
                    except Exception as e:
                        print(f"Error processing {file_name} for type {inst_type} ({folder_name}): {e}")
                else:
                    print(f"File not found: {file_name} in ({folder_name}).")
    return type_report_frames


# ===== Driver ================================================================

def _detect_end_year(folders) -> int:
    """Latest year with any IF_DATA_Values_* or IF_DATA_List_* file present."""
    years = set()
    pats = [os.path.join(folders[k], "IF_DATA_Values_*.csv")
            for k in ("prud_cong", "fin_cong", "ind_inst")]
    pats.append(os.path.join(folders["list"], "IF_DATA_List_*.csv"))
    for pat in pats:
        for fp in glob.glob(pat):
            m = re.search(r"_(\d{4})_\d{1,2}\.csv$", os.path.basename(fp))
            if m:
                years.add(int(m.group(1)))
    return max(years) if years else START_YEAR


def main():
    ap = argparse.ArgumentParser(description="Rebuild IF-Data Aggregated Data reports.")
    ap.add_argument("--verify", action="store_true",
                    help="After writing, compare type-1 reports vs the backup on the overlap.")
    args = ap.parse_args()

    folder_names = {'list': 'List', 'fin_cong': 'Financial Conglomerates',
                    'prud_cong': 'Prudential Conglomerates',
                    'ind_inst': 'Individual Institutions', 'output': 'Aggregated Data'}
    folders = setup_directories(MAIN_FOLDER, folder_names)

    end_year = _detect_end_year(folders)
    print(f"IF-Data aggregation: years {START_YEAR}..{end_year}")

    out_dir = folders['output']

    # Process List crosswalk.
    list_quarterly = process_list_files(folders['list'], START_YEAR, end_year)

    # Reports list (exclude report 15 — not in IF Data for types 1/2/3).
    reports_list = pd.read_csv(os.path.join(MAIN_FOLDER, "Reports_List.csv"),
                               sep=",", encoding='utf-8')['NumeroRelatorio'].tolist()
    reports_list = [r for r in reports_list if r != 15]

    type_report_frames = process_main_loop(
        [1, 2, 3], reports_list, list_quarterly, folders, START_YEAR, end_year)

    # Hygiene: warn about stale report files with no current-input source, and
    # back up only the reports we are about to overwrite (timestamped).
    bak_dir = _prepare_output(out_dir, type_report_frames)
    save_type_report_frames(type_report_frames, out_dir)
    print("IF-Data aggregation complete.")

    if args.verify and bak_dir and os.path.isdir(bak_dir):
        _verify(out_dir, bak_dir)


def _prepare_output(out_dir, type_report_frames):
    """Pre-write hygiene before overwriting the Aggregated Data folder.

    (1) Warn about existing report files that the current inputs will NOT
        regenerate (left STALE) — this is exactly the trap that froze old
        Financial history when its raw inputs went missing.
    (2) Back up ONLY the report files we are about to overwrite, into a
        timestamped ``Aggregated Data_bak_<ts>`` dir (cheap; never relies on a
        single stale one-time backup). Returns that dir, or None.
    """
    import datetime
    regenerated = {
        f"IF_DATA_type_{k.split('_')[1]}_report_{k.split('_')[3]}.csv"
        for k, frames in type_report_frames.items() if frames
    }
    existing = {os.path.basename(p)
                for p in glob.glob(os.path.join(out_dir, "IF_DATA_type_*_report_*.csv"))}
    stale = sorted(existing - regenerated)
    if stale:
        print("WARNING: existing report files have NO current-input source and will be "
              "left STALE (not refreshed):")
        for s in stale:
            print(f"   STALE: {s}")
        print("   -> not reproducible from the current inputs; verify/preserve before "
              "trusting or deleting them.")
    overwrite = sorted(existing & regenerated)
    if not overwrite:
        _prune_old_backups(out_dir)
        return None
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak_dir = f"{out_dir}_bak_{ts}"
    os.makedirs(bak_dir, exist_ok=True)
    print(f"Backing up {len(overwrite)} report(s) about to be overwritten -> "
          f"{os.path.basename(bak_dir)}")
    for name in overwrite:
        shutil.copy2(os.path.join(out_dir, name), os.path.join(bak_dir, name))
    _prune_old_backups(out_dir)
    return bak_dir


def _prune_old_backups(out_dir, keep=1):
    """Retention: keep only the newest ``keep`` timestamped _bak dirs.

    Without this every refresh leaves another full-size snapshot behind (three
    had accumulated to 23 GB by 2026-07).  The ``_bak_%Y%m%d_%H%M%S`` suffix is
    fixed-width, so lexical order is chronological and the bak the caller just
    wrote is always the survivor.
    """
    baks = sorted(p for p in glob.glob(f"{out_dir}_bak_*") if os.path.isdir(p))
    for old in (baks[:-keep] if keep > 0 else baks):
        print(f"Pruning old backup: {os.path.basename(old)}")
        shutil.rmtree(old, ignore_errors=True)


def _verify(out_dir: str, bak_dir: str):
    """For each type-1 report present in both, compare per-(Year,Month) Value sums."""
    print("\nVERIFY: comparing regenerated type-1 reports vs backup ...")
    rep_files = sorted(glob.glob(os.path.join(bak_dir, "IF_DATA_type_1_report_*.csv")))
    any_changed = False
    for bak_fp in rep_files:
        name = os.path.basename(bak_fp)
        new_fp = os.path.join(out_dir, name)
        if not os.path.exists(new_fp):
            print(f"  {name}: MISSING in regenerated output"); any_changed = True; continue
        try:
            uc = lambda c: c in ('Year', 'Month', 'Value')
            o = pd.read_csv(bak_fp, encoding='utf-8', usecols=uc, low_memory=False)
            n = pd.read_csv(new_fp, encoding='utf-8', usecols=uc, low_memory=False)
        except Exception as e:
            print(f"  {name}: read error {e}"); continue
        go = o.groupby(['Year', 'Month'])['Value'].sum()
        gn = n.groupby(['Year', 'Month'])['Value'].sum()
        common = go.index.intersection(gn.index)
        added = sorted(set(map(tuple, gn.index.difference(go.index))))
        # changed existing cells (tolerance for float)
        diff = (gn.loc[common] - go.loc[common]).abs()
        maxrel = 0.0
        for idx in common:
            base = abs(go.loc[idx])
            if base > 0:
                maxrel = max(maxrel, diff.loc[idx] / base)
        flag = "" if maxrel < 1e-6 else "  <-- EXISTING VALUES CHANGED"
        if flag:
            any_changed = True
        print(f"  {name}: overlap_max_rel_diff={maxrel:.2e}  added_periods={added}{flag}")
    print("VERIFY: " + ("WARNING - some historical values changed; inspect before trusting."
                        if any_changed else
                        "PASS - historical cells unchanged; only new periods added."))


if __name__ == "__main__":
    main()
