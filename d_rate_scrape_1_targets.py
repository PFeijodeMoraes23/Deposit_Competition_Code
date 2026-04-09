"""
d_rate_scrape_1_targets.py
==============================
Initializes the target list for historical Internet Archive deposit rate scraping.
Reads the D-Type summary file and constructs domains to be queued up for scraping.
"""
import pandas as pd
import json
from pathlib import Path
import re

# 1. Locate the summary file
root = Path('.').resolve().parents[2]
summary_path = root / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed' / 'ESTIMATION_OUTPUT' / 'DESCRIPTIVES' / 'd_type_firms_summary_final.csv'

if not summary_path.exists():
    print(f"File not found: {summary_path}")
    exit(1)

df = pd.read_csv(summary_path)

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
