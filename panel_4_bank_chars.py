import os
import unicodedata
import logging
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd
from pathlib import Path

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

from utils import paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

BASE = str(paths.OPEN_FINANCE)
IF_AGG_DIR  = str(paths.IF_DATA_AGG)
IF_LIST_DIR = paths.IF_DATA_LIST   # Path object; List files have IPs, Prudential report does not
OUTPUT_DIR  = str(paths.PROCESSED)
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


def _has_ip_from_list_files(list_dir: Path) -> dict[str, int]:
    """
    Build a {CodConglomeradoPrudencial -> has_ip} lookup from IF-Data List files.

    The Prudential Conglomerate report (Report 1) contains NO Payment Institutions
    (they are not reported under the prudential consolidation). Only the List files
    include them, identified by SegmentoTb containing 'Instituição de Pagamento'.
    has_ip=1 if ANY member institution across ALL available quarterly List files is a PI.

    Returns an empty dict on failure (fallback: has_ip stays as derived from Report 1).
    """
    def _norm(s: str) -> str:
        return (unicodedata.normalize("NFKD", str(s))
                .encode("ascii", "ignore").decode("ascii").lower())

    try:
        files = sorted(list_dir.glob("IF_DATA_List_*.csv"))
        if not files:
            logging.warning(f"[has_ip] No List files found in {list_dir}; has_ip may be all-zero.")
            return {}
        frames = []
        for f in files:
            try:
                d = pd.read_csv(f, dtype=str, encoding="utf-8")
            except UnicodeDecodeError:
                d = pd.read_csv(f, dtype=str, encoding="latin-1")
            d.columns = [c.strip() for c in d.columns]
            need = {"CodInst", "CodConglomeradoPrudencial", "SegmentoTb"}
            if not need.issubset(d.columns):
                continue
            if {"Td", "Situacao"}.issubset(d.columns):
                d = d[(d["Td"] == "I") & (d["Situacao"] == "A")]
            cong = d["CodConglomeradoPrudencial"].astype(str)
            congl = np.where(cong.isna() | (cong == "null") | (cong == "nan"),
                             d["CodInst"].astype(str), cong)
            seg = d["SegmentoTb"].map(_norm)
            is_ip = (seg.str.contains("institui", na=False)
                     & seg.str.contains("pagamento", na=False)).astype(int)
            frames.append(pd.DataFrame({"congl": congl.astype(str), "is_ip": is_ip.values}))
        if not frames:
            return {}
        congl_ip = pd.concat(frames, ignore_index=True).groupby("congl")["is_ip"].max()
        lookup = {str(k).replace(".0", ""): int(v) for k, v in congl_ip.items()}
        n_ip = sum(v for v in lookup.values() if v)
        logging.info(f"[has_ip] List files: {len(files)} files, {n_ip} IP conglomerates identified.")
        return lookup
    except Exception as exc:
        logging.warning(f"[has_ip] List-file derivation failed ({exc}); falling back to Prudential report.")
        return {}


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

    # 2. Asset quality (Ativo) — NPL provision + the EARNING-ASSET stocks.
    #    The earning-asset stocks feed the bank's asset return r^j (V_Main eq 16, ψ1 row): the
    #    return on what deposits actually fund. Disponibilidades (78188) is deliberately EXCLUDED
    #    — it is non-earning cash. Compulsórios are not broken out separately in IF-Data's condensed
    #    Ativo; they sit inside the asset base earning little, so the realized yield below is already
    #    reserve-drag-adjusted (see counterfactuals_plan.md §9.4).
    p2 = load_and_pivot(2, {
        78192: 'npl_provision',
        78189: 'aplic_interfin',   # Aplicacoes Interfinanceiras de Liquidez
        78190: 'tvm',              # TVM e Instrumentos Financeiros Derivativos
        78193: 'credit_net',       # Operacoes de Credito Liquidas de Provisao
        78198: 'leasing_net',      # Arrendamento Mercantil Liquido de Provisao
    })

    # 3. Wholesale (Passivo)
    p3 = load_and_pivot(3, {
        78288: 'repos', 78289: 'lci', 78290: 'lca',
        78291: 'letras_financeiras', 78295: 'emprestimos_repasses'
    })

    # 4. Costs + financial income (DRE)
    p4 = load_and_pivot(4, {
        78218: 'personnel_expenses',
        78219: 'admin_expenses',
        78220: 'tax_expenses',
        78208: 'fin_income',       # Receitas de Intermediacao Financeira (= credit + TVM + deriv + ...)
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
    # fin_income is a DRE flow too — it accumulates within the calendar year exactly like the
    # expense lines, so it MUST go through the same differencing or the asset yield is nonsense
    # (Q4 would carry four quarters of income against one quarter of assets).
    if not p4.empty:
        for cost_col in ['personnel_expenses', 'admin_expenses', 'tax_expenses', 'fin_income']:
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

    # ── ASSET RETURN r^j (V_Main eq 16, ψ1 row) ─────────────────────────────────────────────
    # The return the bank earns on the assets its deposits fund. CF2 needs (r^j − r^f); with r^j
    # left at 0 the deposit franchise is worth only (ρ − c) and eq-18 can only rationalise the
    # observed spreads with a NEGATIVE marginal cost (see counterfactuals_plan.md §9.4).
    #
    # Realized portfolio yield = quarterly financial income ÷ LAGGED earning assets. Dividing by
    # the lagged stock mirrors the deposit implicit-rate convention (expense ÷ lagged stock) and
    # keeps the flow/stock timing honest. Because compulsórios sit inside the asset base earning
    # little, this realized yield is already reserve-drag ("compulsório") adjusted.
    ea_cols = [c for c in ['aplic_interfin', 'tvm', 'credit_net', 'leasing_net'] if c in panel.columns]
    if ea_cols and 'fin_income' in panel.columns:
        panel['earning_assets'] = panel[ea_cols].sum(axis=1, min_count=1)
        ea_lag = panel.groupby('CodConglomeradoPrudencial')['earning_assets'].shift(1)
        ea_lag = ea_lag.where(ea_lag > 0)                      # guard: no yield off a zero/neg base
        # The DRE differencing above fillna(0)s missing flows. A bank with NO reported income would
        # then get r^j = 0 and hence an asset margin of −r^f (≈ −10.5pp/yr) — reintroducing exactly
        # the pathology this column exists to remove. So only trust a strictly positive income, and
        # impute the rest from the cross-sectional median for that quarter (r^j is a bank
        # characteristic; the quarter median is the natural fallback and keeps the sign right).
        raw = panel['fin_income'].where(panel['fin_income'] > 0) / ea_lag
        raw = raw.replace([np.inf, -np.inf], np.nan)
        raw = raw.where((raw > 0) & (raw < 0.5))               # drop absurd yields (>50%/quarter)
        med_by_q = raw.groupby([panel['Year'], panel['Quarter']]).transform('median')
        panel['asset_return_qoq'] = raw.fillna(med_by_q).fillna(raw.median())
        panel['asset_return_imputed'] = raw.isna().astype(int)
    else:
        panel['earning_assets'] = np.nan
        panel['asset_return_qoq'] = np.nan
        panel['asset_return_imputed'] = 1

    # Override has_ip using IF-Data List files, which include Payment Institutions.
    # The Prudential Conglomerate report (Report 1) never lists IPs, so has_ip from
    # the pivot above is always 0. List files are the authoritative source.
    ip_lookup = _has_ip_from_list_files(IF_LIST_DIR)
    if ip_lookup:
        panel['has_ip'] = (
            panel['CodConglomeradoPrudencial'].astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .map(ip_lookup)
            .fillna(panel.get('has_ip', 0))
            .astype(int)
        )

    # FINALLY, Lag all variables to be used cleanly in regressions
    lag_cols = ['total_assets', 'equity', 'equity_ratio', 'log_total_assets',
                'lci_lca_ratio', 'wholesale_ratio', 'indice_basileia',
                'personnel_cost_ratio', 'admin_cost_ratio', 'tax_cost_ratio',
                'npl_provision_ratio', 'asset_return_qoq']

    present_lag_cols = [c for c in lag_cols if c in panel.columns]
    panel[[c + '_lag' for c in present_lag_cols]] = panel.groupby('CodConglomeradoPrudencial')[present_lag_cols].shift(1)

    # Gross factor for the Julia consumer (cost_2_fwd_sim.jl --asset-return-col), which does
    #     gr = col − 1 ;  asset_ret = gr − risk_free_qoq_lag
    # so this must be 1 + a LAGGED quarterly DECIMAL rate — the same vintage and units as
    # gross_return_lag = 1 + deposit_rate_lag (≈1.0126) and risk_free_qoq_lag (≈0.0253).
    if 'asset_return_qoq_lag' in panel.columns:
        panel['asset_gross_return_lag'] = 1.0 + panel['asset_return_qoq_lag']

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
