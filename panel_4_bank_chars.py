import os
import logging
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
IF_AGG_DIR  = os.path.join(BASE, 'BCB', 'IF Data', 'Aggregated Data')
OUTPUT_DIR  = os.path.join(BASE, 'BCB', 'Egan_et_al_2025_Rep', 'processed')
OUT_CSV     = os.path.join(OUTPUT_DIR, 'bank_chars_panel.csv')

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        'bank_chars_panel_build',
        {
            'if_agg_dir': IF_AGG_DIR,
            'output_dir': OUTPUT_DIR,
            'out_csv': OUT_CSV,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    IF_AGG_DIR = _paths['if_agg_dir']
    OUTPUT_DIR = _paths['output_dir']
    OUT_CSV = _paths['out_csv']

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_pivot(report_num, accounts_dict):
    csv_path = os.path.join(IF_AGG_DIR, f'IF_DATA_type_1_report_{report_num}.csv')
    if not os.path.exists(csv_path):
        logging.warning(f'IF Data report {report_num} not found: {csv_path}')
        return pd.DataFrame()

    logging.info(f'Loading IF Data Report {report_num} ...')
    df = pd.read_csv(csv_path, encoding='latin1', low_memory=False)

    if 'CodConglomeradoPrudencial' not in df.columns:
        return pd.DataFrame()

    # Fallback to CNPJ if missing
    mask_no_cong = df['CodConglomeradoPrudencial'].isna() | (df['CodConglomeradoPrudencial'].astype(str).str.strip() == '')
    if mask_no_cong.any():
        df.loc[mask_no_cong, 'CodConglomeradoPrudencial'] = 'CNPJ_' + df.loc[mask_no_cong, 'CNPJ'].astype(str)
    
    # Strip .0 if parsed as float string
    df['CodConglomeradoPrudencial'] = df['CodConglomeradoPrudencial'].astype(str).str.replace(r'\.0$', '', regex=True)

    # Time variables
    df['Month'] = pd.to_numeric(df.get('Month'), errors='coerce')
    df['Year']  = pd.to_numeric(df.get('Year'),  errors='coerce')
    df = df.dropna(subset=['Month', 'Year']).copy()
    df['Quarter'] = df['Month'].map({3: 1, 6: 2, 9: 3, 12: 4})
    df = df.dropna(subset=['Quarter']).copy()
    df['Quarter'] = df['Quarter'].astype(int)
    df['Year']    = df['Year'].astype(int)

    # Filter needed accounts
    df['Conta'] = pd.to_numeric(df.get('NumeroConta'), errors='coerce').astype('Int64')
    df['Value'] = pd.to_numeric(df.get('Value'), errors='coerce')
    df_sub = df[df['Conta'].isin(accounts_dict.keys())].copy()

    if df_sub.empty:
        return pd.DataFrame()

    df_sub['prop_name'] = df_sub['Conta'].map(accounts_dict)

    # Pivot accounts
    pivot = (
        df_sub.groupby(['CodConglomeradoPrudencial', 'Year', 'Quarter', 'prop_name'])['Value']
        .sum().unstack('prop_name').reset_index()
    )
    pivot.columns.name = None

    # For Report 1, also keep segment details
    if report_num == 1:
        if 'SegmentoTb' in df.columns:
            df['is_ip'] = (
                df['SegmentoTb'].astype(str).str.contains('Institu', case=False, na=False) &
                df['SegmentoTb'].astype(str).str.contains('Pagamento', case=False, na=False)
            )
            df['is_coop'] = df['SegmentoTb'].astype(str).str.contains('Cooperativa', case=False, na=False)
        else:
            df['is_ip'] = False
            df['is_coop'] = False

        if 'NomeInstituicao' in df.columns:
            df['is_state_owned'] = (
                df['SegmentoTb'].astype(str).str.contains('Caixa Econômica', case=False, na=False) |
                df['NomeInstituicao'].astype(str).str.contains('Banco do Brasil|BNDES|BANRISUL|NORDESTE|AMAZONIA|BANZ|BANPARA|BANESE|BRB|BANDES', case=False, na=False)
            )
        else:
            df['is_state_owned'] = False

        if 'Atividade' in df.columns:
            df['is_captive'] = df['Atividade'].astype(str).str.contains('Mútuo PF sem conta corrente', case=False, na=False)
        else:
            df['is_captive'] = False

        if 'Sr' in df.columns:
            df['segment_raw'] = df['Sr'].astype(str).str.strip().str.upper()
            df['segment_raw'] = df['segment_raw'].where(df['segment_raw'].isin(['S1', 'S2', 'S3', 'S4', 'S5']), np.nan)
        else:
            df['segment_raw'] = np.nan

        cat_agg = (
            df.groupby(['CodConglomeradoPrudencial', 'Year', 'Quarter'])
            .agg(has_ip=('is_ip', 'max'), 
                 is_coop=('is_coop', 'max'),
                 is_state_owned=('is_state_owned', 'max'),
                 is_captive=('is_captive', 'max'),
                 segment=('segment_raw', 'first'))
            .reset_index()
        )
        pivot = pivot.merge(cat_agg, on=['CodConglomeradoPrudencial', 'Year', 'Quarter'], how='right')

    return pivot

def build_panel() -> pd.DataFrame:
    # 1. Base (Resumo) size attributes
    p1 = load_and_pivot(1, {78182: 'total_assets', 78186: 'equity'})
    if p1.empty: return pd.DataFrame()
    for col in ['total_assets', 'equity']:
        if col not in p1.columns: p1[col] = np.nan

    # 2. Asset quality (Ativo) — NPL provision
    p2 = load_and_pivot(2, {78192: 'npl_provision'})

    # 3. Wholesale (Passivo)
    p3 = load_and_pivot(3, {
        78288: 'repos', 78289: 'lci', 78290: 'lca', 
        78291: 'letras_financeiras', 78295: 'emprestimos_repasses'
    })

    # 4. Costs (DRE)
    p4 = load_and_pivot(4, {
        78218: 'personnel_expenses',
        78219: 'admin_expenses', 
        78220: 'tax_expenses'
    })

    # 5. Capital (Informacoes de Capital)
    p5 = load_and_pivot(5, {79664: 'indice_basileia_raw'})

    # Merge everything
    panel = p1
    if not p2.empty: panel = panel.merge(p2, on=['CodConglomeradoPrudencial', 'Year', 'Quarter'], how='left')
    if not p3.empty: panel = panel.merge(p3, on=['CodConglomeradoPrudencial', 'Year', 'Quarter'], how='left')
    if not p4.empty: panel = panel.merge(p4, on=['CodConglomeradoPrudencial', 'Year', 'Quarter'], how='left')
    if not p5.empty: panel = panel.merge(p5, on=['CodConglomeradoPrudencial', 'Year', 'Quarter'], how='left')

    panel.sort_values(['CodConglomeradoPrudencial', 'Year', 'Quarter'], inplace=True)

    # Note: IF Data DRE is reported cumulatively by year. Standardize to quarter flows.
    if not p4.empty:
        for cost_col in ['personnel_expenses', 'admin_expenses', 'tax_expenses']:
            if cost_col in panel.columns:
                panel[cost_col] = panel[cost_col].fillna(0)
                # Group differencing by Conglomerate-Year
                val_diff = panel.groupby(['CodConglomeradoPrudencial', 'Year'])[cost_col].diff()
                panel[cost_col] = val_diff.fillna(panel[cost_col])

    # Core characteristics
    total_assets_no0 = panel['total_assets'].replace(0, np.nan)
    panel['equity_ratio'] = panel['equity'] / total_assets_no0
    panel['log_total_assets'] = np.log(panel['total_assets'].clip(lower=1))

    # NPL provision ratio (provision is reported as negative; take abs)
    if 'npl_provision' in panel.columns:
        panel['npl_provision_ratio'] = panel['npl_provision'].abs() / total_assets_no0
    else:
        panel['npl_provision_ratio'] = np.nan

    # Segment defaults
    for s in ['S2', 'S3', 'S4', 'S5']:
        panel[f'seg_{s}'] = (panel.get('segment') == s).astype(int)

    # Wholesale ratios
    if 'lci' in panel.columns and 'lca' in panel.columns:
        panel['lci_lca_ratio'] = panel[['lci', 'lca']].sum(axis=1) / total_assets_no0
    else:
        panel['lci_lca_ratio'] = np.nan

    wholesale_cols = [c for c in ['repos', 'lci', 'lca', 'letras_financeiras', 'emprestimos_repasses'] if c in panel.columns]
    panel['wholesale_ratio'] = panel[wholesale_cols].sum(axis=1) / total_assets_no0 if wholesale_cols else np.nan

    # Cost ratios (take absolute value since DRE sums are negative expenses)
    if not p4.empty:
        for c, out_n in [('personnel_expenses', 'personnel_cost_ratio'), 
                         ('admin_expenses', 'admin_cost_ratio'), 
                         ('tax_expenses', 'tax_cost_ratio')]:
            if c in panel.columns:
                panel[out_n] = panel[c].abs() / total_assets_no0
            else:
                panel[out_n] = np.nan
    else:
        for c in ['personnel_cost_ratio', 'admin_cost_ratio', 'tax_cost_ratio']:
            panel[c] = np.nan

    if 'indice_basileia_raw' in panel.columns:
        panel['indice_basileia'] = panel['indice_basileia_raw']
    else:
        panel['indice_basileia'] = np.nan

    # FINALLY, Lag all variables to be used cleanly in regressions
    lag_cols = ['total_assets', 'equity', 'equity_ratio', 'log_total_assets',
                'lci_lca_ratio', 'wholesale_ratio', 'indice_basileia',
                'personnel_cost_ratio', 'admin_cost_ratio', 'tax_cost_ratio',
                'npl_provision_ratio']
    
    present_lag_cols = [c for c in lag_cols if c in panel.columns]
    panel[[c + '_lag' for c in present_lag_cols]] = panel.groupby('CodConglomeradoPrudencial')[present_lag_cols].shift(1)

    panel.rename(columns={'Year': 'year', 'Quarter': 'quarter'}, inplace=True)
    return panel

def main() -> None:
    panel = build_panel()
    if not panel.empty:
        import pyarrow as pa
        import pyarrow.csv as pa_csv
        pa_csv.write_csv(pa.Table.from_pandas(panel, preserve_index=False), OUT_CSV)
        logging.info(f'Saved bank characteristics panel to {OUT_CSV}')
    else:
        logging.warning('Build returned an empty panel.')

if __name__ == '__main__':
    main()
