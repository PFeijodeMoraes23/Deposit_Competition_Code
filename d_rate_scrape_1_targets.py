"""
d_rate_scrape_1_targets.py
==============================
Initializes the target list for historical Internet Archive deposit rate scraping.

Self-sufficient: if `d_type_firms_summary_final.csv` is missing or empty
(header-only), this script regenerates it directly from `market_panel.csv`
+ `digital_banks_diagnostic.csv` + `ip_rates_quarterly.csv` using the same
logic as the D-Type block in `desc_1.py` (with the `df_b`/`df` filter bug
fixed). It then proceeds to build the scraper target list.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import pandas as pd
import json
from pathlib import Path
import re

# 1. Locate inputs
root = Path(__file__).resolve().parents[2]
processed = root / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed'
summary_path = processed / 'ESTIMATION_OUTPUT' / 'DESCRIPTIVES' / 'd_type_firms_summary_final.csv'
panel_path = processed / 'market_panel.csv'


def _build_d_type_summary(panel_csv: Path, processed_dir: Path, out_path: Path) -> pd.DataFrame:
    """Rebuild d_type_firms_summary_final.csv from market_panel + diagnostics.

    Mirrors the D-Type block in desc_1.py but uses the full panel (filtered to
    CODMUN_IBGE == "0", i.e. national/D-type rows) instead of the buggy
    df_b-then-filter chain.
    """
    print(f"Building D-Type summary from {panel_csv.name} ...")
    usecols = ['CODMUN_IBGE', 'CodConglomeradoPrudencial', 'NomeInstituicao',
               'total_assets', 'year', 'quarter', 'Source', 'CNPJ_Lider']
    df = pd.read_csv(panel_csv, usecols=usecols, low_memory=False)

    # D-type rows: national-aggregate (CODMUN_IBGE == "0")
    df['CODMUN_IBGE_str'] = df['CODMUN_IBGE'].astype(str).str.split('.').str[0]
    d_firms = df[df['CODMUN_IBGE_str'] == '0'].copy()
    d_firms['cnpj_8'] = (pd.to_numeric(d_firms['CNPJ_Lider'], errors='coerce')
                          .astype('Int64').astype(str).str.zfill(8))

    # Filter to IFDATA rows OR digital-bank candidates from diagnostic
    diag_candidates = list(processed_dir.glob('**/digital_banks_diagnostic.csv'))
    if not diag_candidates:
        raise FileNotFoundError(
            f"digital_banks_diagnostic.csv not found under {processed_dir}")
    df_flags = pd.read_csv(diag_candidates[0])
    dig_cands = set(df_flags.loc[df_flags['is_digital_candidate'] == True,
                                 'CNPJ_root'].astype(str).str.zfill(8))
    d_firms = d_firms[(d_firms['Source'] == 'IFDATA')
                      | (d_firms['cnpj_8'].isin(dig_cands))]

    # IP-rate availability check
    ip_candidates = list(processed_dir.glob('**/ip_rates_quarterly.csv'))
    if not ip_candidates:
        raise FileNotFoundError(
            f"ip_rates_quarterly.csv not found under {processed_dir}")
    df_ip = pd.read_csv(ip_candidates[0], low_memory=False)
    cnpjs_with_ip = set(df_ip['CNPJ'].astype(str).str.zfill(8).unique())

    d_summ = (d_firms.groupby(['CodConglomeradoPrudencial', 'NomeInstituicao'])
                     .agg(total_assets=('total_assets', 'max'),
                          first_year=('year', 'min'),
                          first_quarter=('quarter', 'min'),
                          obs=('year', 'count'))
                     .reset_index())
    d_summ = d_summ.sort_values(['total_assets', 'first_year', 'first_quarter'],
                                 ascending=[False, True, True])

    # Mark conglomerates that have native (IP) k5 rate data
    cp_to_native = {}
    for cp, grp in d_firms.groupby('CodConglomeradoPrudencial'):
        cp_to_native[cp] = ('Yes (IP Rates)'
                            if grp['cnpj_8'].isin(cnpjs_with_ip).any()
                            else 'No (COSIF Fallback)')
    d_summ['native_k5_rate'] = d_summ['CodConglomeradoPrudencial'].map(cp_to_native)

    d_summ['Asset_Size'] = d_summ['total_assets'].apply(
        lambda x: f"{x/1e9:.2f} B" if pd.notna(x) else 'Unknown')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    d_summ.to_csv(out_path, index=False)
    print(f"  wrote {len(d_summ):,} D-type conglomerates -> {out_path}")
    return d_summ


def _needs_rebuild(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        head = pd.read_csv(path, nrows=1)
        return head.empty  # header-only => 0 data rows
    except Exception:
        return True


if _needs_rebuild(summary_path):
    if not panel_path.exists():
        print(f"File not found: {panel_path}")
        print("Cannot regenerate D-Type summary; run the panel pipeline first.")
        exit(1)
    df = _build_d_type_summary(panel_path, processed, summary_path)
else:
    df = pd.read_csv(summary_path)
    print(f"Using existing summary: {summary_path} ({len(df):,} rows)")

# 2. Filter the top targets (Top 40 by default), excluding development banks like BNDES
df = df[~df['NomeInstituicao'].str.contains('BNDES', case=False, na=False)]
top_targets = df.head(40).copy()

# Known domains dictionary for quick start - manually map the most obvious ones
KNOWN_DOMAINS = {
    'NUBANK': 'nubank.com.br',
    'NU PAGAMENTOS  PRUDENCIAL': 'nubank.com.br',
    'NU FINANCEIRA SA  SOCIEDADE DE CRADITO FINANCIAMENTO E INVESTIMENTO': 'nubank.com.br',
    'MERCADO PAGO IP  PRUDENCIAL': 'mercadopago.com.br',
    'MERCADOPAGO PRUDENCIAL': 'mercadopago.com.br',
    'MERCADO CRADITO SOCIEDADE DE CRADITO FINANCIAMENTO E INVESTIMENTO SA': 'mercadopago.com.br',
    'STONE IP  PRUDENCIAL': 'stone.com.br',
    'STONE SOCIEDADE DE CRADITO DIRETO SA': 'stone.com.br',
    'STONE PAGAMENTOS  PRUDENCIAL': 'stone.com.br',
    'CIELO IP  PRUDENCIAL': 'cielo.com.br',
    'PICPAY SERVICOS SA': 'picpay.com',
    'C6 CORRETORA DE TATULOS E VALORES MOBILIARIOS LTDA': 'c6bank.com.br',
    'BANCO INTER': 'bancointer.com.br',
    'BANCO ORIGINAL': 'original.com.br',
    'AGIBANK  PRUDENCIAL': 'agibank.com.br',
    'GRUPO BONSUCESSO  BS2': 'bs2.com',
    'WILL FINANCEIRA SA CRADITO FINANCIAMENTO E INVESTIMENTO': 'willbank.com.br',
    'WILL IP  PRUDENCIAL': 'willbank.com.br',
    'CLOUDWALK IP  PRUDENCIAL': 'cloudwalk.io'
}

targets = []
for _, row in top_targets.iterrows():
    name = str(row['NomeInstituicao']).strip()
    domain = KNOWN_DOMAINS.get(name, "")
    
    # We heuristically guess domains if missing (e.g. remove spaces, add .com.br)
    if not domain:
        clean_name = re.sub(r'[^a-zA-Z]', '', name.lower().replace('banco', '').replace('prudencial', '').replace('sa', '').replace('ltda', ''))
        if clean_name:
            domain = f"{clean_name}.com.br"

    targets.append({
        'cod_conglomerado': row['CodConglomeradoPrudencial'],
        'name': name,
        'asset_size': row['Asset_Size'],
        'native_k5_rate': row.get('native_k5_rate', 'Unknown'),
        'domain': domain,
        'first_year': str(row['first_year'])
    })

# 3. Save to a JSON for the next pipeline step
out_dir = root / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed' / 'IP_SCRAPE'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'scraper_targets.json'

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(targets, f, indent=4, ensure_ascii=False)

print(f"Target list generated with {len(targets)} firms.")
print(f"Saved to {out_path.resolve()}")
print("Please review and manually correct any auto-generated domains in scraper_targets.json before running step 2.")
