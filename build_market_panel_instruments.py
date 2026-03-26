"""
build_market_panel_instruments.py
=================================
Pipeline Step 4b

Reads the merged market_panel.csv (from Step 4a) and appends BLP-style 
Leave-One-Out (LOO) instruments and the FGC insurance dummy.

Specifically, it computes:
  - fgc_covered: 1 for ALL institutions (in Brazil, all commercial/multiple banks,
    credit unions, and savings banks are FGC covered for retail deposits up to 250k BRL.
    Payment institutions are covered by BCB direct routing, which is equivalent).
  - n_rivals: Number of other active depository institutions in the MCA-quarter.
  - loo_log_assets: Sum of log_total_assets_lag of all *other* rivals in the MCA.
  - mean_loo_log_assets: loo_log_assets / n_rivals.
  - loo_equity_assets: Sum of equity_ratio_lag of all *other* rivals.
  - mean_loo_equity_ratio: loo_equity_assets / n_rivals.
  ...and similarly for basileia, credit_assets (emprestimos_repasses/total_assets),
  and npl_provision (if available, else we skip).

The augmented panel is saved back to market_panel.csv.
"""

import os
import sys
import logging
from pathlib import Path

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_ROOT = Path(__file__).resolve().parents[2]
PANEL_CSV = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "market_panel.csv"

def compute_loo_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Leave-One-Out (LOO) BLP instruments per MCA-quarter."""
    df = df.copy()

    # Define the characteristics we want to build LOO instruments for
    chars = [
        ('log_assets', 'log_total_assets_lag'),
        ('equity_ratio', 'equity_ratio_lag'),
        ('basileia', 'indice_basileia_lag')
    ]

    # Create credit_assets ratio if emprestimos_repasses and total_assets exist
    if 'emprestimos_repasses' in df.columns and 'total_assets_lag' in df.columns:
        df['credit_assets_lag'] = df['emprestimos_repasses'] / df['total_assets_lag']
        chars.append(('credit_assets', 'credit_assets_lag'))

    # FGC covered is 1 for all institutions in our sample (universal coverage up to 250k)
    df['fgc_covered'] = 1

    logging.info("Computing BLP LOO instruments for each mca_code-quarter...")

    # We only compute LOO for B-type (local) markets. National (NB) is a single market
    # per quarter. We'll compute it universally grouping by (mca_code, year, quarter).
    
    # We only want to count active rivals in the market. Since an institution's presence 
    # in the panel implies it's active, we group by mca, year, quarter.
    
    # Fill NAs for characteristics with 0 solely for the sum (keeps it robust)
    char_cols = [c[1] for c in chars if c[1] in df.columns]
    df_chars = df[['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter'] + char_cols].copy()
    
    # Ensure one row per conglomerate-market-time
    df_chars = df_chars.drop_duplicates(subset=['CodConglomeradoPrudencial', 'mca_code', 'year', 'quarter'])
    
    for _, col in chars:
        if col in df_chars.columns:
            df_chars[col] = df_chars[col].fillna(0)
    
    # Compute market totals
    market_totals = df_chars.groupby(['mca_code', 'year', 'quarter'])[char_cols].sum().reset_index()
    market_counts = df_chars.groupby(['mca_code', 'year', 'quarter']).size().reset_index(name='n_firms')
    
    market_agg = pd.merge(market_totals, market_counts, on=['mca_code', 'year', 'quarter'])
    
    # Merge back to full panel
    df = pd.merge(df, market_agg, on=['mca_code', 'year', 'quarter'], suffixes=('', '_mkt_sum'))
    
    # Rival count is n_firms - 1
    df['n_rivals'] = np.maximum(0, df['n_firms'] - 1)
    
    # Compute LOO and Mean LOO sums
    for name, col in chars:
        if col in df.columns:
            # sum of others = total sum - own value
            loo_col = f'loo_{name}'
            mean_loo_col = f'mean_loo_{name}'
            
            own_val = df[col].fillna(0)
            mkt_sum = df[f'{col}_mkt_sum']
            
            df[loo_col] = mkt_sum - own_val
            
            # mean = sum / n_rivals (0 if no rivals)
            df[mean_loo_col] = np.where(df['n_rivals'] > 0, 
                                        df[loo_col] / df['n_rivals'], 
                                        0)
            
            # Clean up intermediate sum
            df.drop(columns=[f'{col}_mkt_sum'], inplace=True)

    df.drop(columns=['n_firms'], inplace=True, errors='ignore')
    
    logging.info(f"Generated {len(chars) * 2 + 2} IV columns: fgc_covered, n_rivals, loo_*, mean_loo_*")
    return df

def main():
    if not PANEL_CSV.exists():
        logging.error(f"Cannot find panel CSV at {PANEL_CSV}")
        sys.exit(1)

    logging.info(f"Loading {PANEL_CSV}")
    df = pd.read_csv(PANEL_CSV, low_memory=False, dtype={'mca_code': str, 'CODMUN_IBGE': str})
    
    # Compute instruments
    df_aug = compute_loo_instruments(df)
    
    # Overwrite the panel
    logging.info(f"Saving augmented panel back to {PANEL_CSV}")
    df_aug.to_csv(PANEL_CSV, index=False)
    logging.info("Done.")

if __name__ == '__main__':
    main()
