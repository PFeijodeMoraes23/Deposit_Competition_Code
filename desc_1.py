"""
descriptive_1.py
==============================
Generates summary statistics for the Egan et al. market panel setup.
Outputs are saved as CSV and LaTeX tables partitioned by overall, bank type, and region,
as well as broken down by year.

CLI Options:
------------
usage: descriptive_1.py [-h] [--weight-col WEIGHT_COL]

Generate descriptive statistics for the market panel.

options:
  -h, --help            show this help message and exit
  --weight-col WEIGHT_COL
                        Column to use for market-weighted statistics (e.g.,
                        pop_total)
"""
import argparse
from pathlib import Path
import sys
import warnings

try:
    from utils.venv_guard import ensure_project_venv
except ImportError:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed"
PANEL_CSV = DATA_DIR / "market_panel.csv"
OUTPUT_DIR = DATA_DIR / "ESTIMATION_OUTPUT" / "Rout"

# Map exactly to the 5 macro-regions of Brazil using the 1st digit of CODMUN_IBGE
REGION_MAPPING = {
    '1': 'North',
    '2': 'Northeast',
    '3': 'Southeast',
    '4': 'South',
    '5': 'Center-West'
}

def weighted_mean_std(df, col, weight_col):
    sub = df[[col, weight_col]].dropna()
    if len(sub) == 0:
        return np.nan, np.nan
    w = np.array(sub[weight_col], dtype=float)
    x = np.array(sub[col], dtype=float)
    w_sum = np.sum(w)
    if w_sum == 0.0:
        return np.nan, np.nan
    mean = np.average(x, weights=w)
    var = np.average((x - mean)**2, weights=w)
    return mean, np.sqrt(var)

def generate_summary(df, group_cols, vars_to_summarize, weight_col=None):
    results = []
    
    if group_cols:
        groups = df.groupby(group_cols)
    else:
        groups = [('Overall', df)]

    for name, group in groups:
        row = {'Group': name if not isinstance(name, tuple) else ' - '.join(map(str, name))}
        
        for v in vars_to_summarize:
            if v not in group.columns:
                continue
            
            if weight_col and weight_col in group.columns:
                mean, std = weighted_mean_std(group, v, weight_col)
            else:
                mean = group[v].mean()
                std = group[v].std()
                
            row[f'{v}_Mean'] = mean
            row[f'{v}_SD'] = std
        
        row['N_obs'] = len(group)
        results.append(row)
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Generate descriptive statistics for the market panel.")
    parser.add_argument('--weight-col', type=str, default=None, help='Column to use for market-weighted statistics (e.g., pop_total)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(PANEL_CSV, low_memory=False)
    
    # 1. Identify Bank Type (D vs B)
    # The convention dictates B-type are municipal (CODMUN_IBGE != 0), D-type are national (= 0).
    df['CODMUN_IBGE_str'] = df['CODMUN_IBGE'].astype(str).str.split('.').str[0]
    df['bank_type'] = np.where(df['CODMUN_IBGE_str'] == '0', 'D', 'B')
    
    # 2. Extract Region for B-type
    df['region_code'] = df['CODMUN_IBGE_str'].str[0]
    df['region'] = df['region_code'].map(REGION_MAPPING)
    df.loc[df['bank_type'] == 'D', 'region'] = 'National'

    # Defining target analytical variables
    vars_to_summarize = [
        'dep_a1', 'dep_a2', 'dep_a3', 'dep_a4', 'dep_a5',
        'spread_a1', 'spread_a2', 'spread_a3', 'spread_a4', 'spread_a5',
        'gdp_per_capita', 'pop_total', 'fraction_65plus', 'cadunico_families_per1000',
        'pix_users_pf_per1000', 'pix_txns_pf', 'has_ip', 'connections_per100', 'branches_per1000',
        'total_assets', 'equity_ratio'
    ]

    UNIT_MAP = {
        'dep_a1': 'R$', 'dep_a2': 'R$', 'dep_a3': 'R$', 'dep_a4': 'R$', 'dep_a5': 'R$',
        'spread_a1': '%', 'spread_a2': '%', 'spread_a3': '%', 'spread_a4': '%', 'spread_a5': '%',
        'gdp_per_capita': 'R$', 'pop_total': 'Count', 'fraction_65plus': 'Fraction',
        'cadunico_families_per1000': 'Per 1000', 'pix_users_pf_per1000': 'Per 1000',
        'pix_txns_pf': 'Count', 'has_ip': 'Binary', 'connections_per100': 'Per 100',
        'branches_per1000': 'Per 1000', 'total_assets': 'R$', 'equity_ratio': 'Fraction'
    }

    # Purge any missing columns from the target list safely
    vars_to_summarize = [v for v in vars_to_summarize if v in df.columns]
    
    weight_str = f"_weighted_by_{args.weight_col}" if args.weight_col else "_unweighted"

    print(f"Generating summaries ({'weighted by ' + args.weight_col if args.weight_col else 'unweighted'})...")

    # Overall Summary
    print("  -> National Overall")
    sum_nat = generate_summary(df, None, vars_to_summarize, args.weight_col)
    sum_nat_yr = generate_summary(df, ['year'], vars_to_summarize, args.weight_col)

    # By Bank Type
    print("  -> By Bank Type (D vs B)")
    sum_bt = generate_summary(df, ['bank_type'], vars_to_summarize, args.weight_col)
    sum_bt_yr = generate_summary(df, ['bank_type', 'year'], vars_to_summarize, args.weight_col)

    # By Region (B-Type only)
    print("  -> By Region (B-type)")
    df_b = df[df['bank_type'] == 'B']
    sum_reg = generate_summary(df_b, ['region'], vars_to_summarize, args.weight_col)
    sum_reg_yr = generate_summary(df_b, ['region', 'year'], vars_to_summarize, args.weight_col)

    DRAFTS_DIR = Path(r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Drafts\Deposit Competition")
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

    # Batch save output forms
    def save_output(res_df, name):
        out_path = OUTPUT_DIR / f"{name}{weight_str}.csv"
        res_df.to_csv(out_path, index=False)
        # Also save transposed version as LaTeX for easy table copy/pasting
        tex_path = OUTPUT_DIR / f"{name}{weight_str}.tex"
        drafts_tex_path = DRAFTS_DIR / f"{name}{weight_str}.tex"
        try:
            transposed = res_df.set_index('Group').T
            new_idx = []
            for idx in transposed.index:
                if idx == 'N_obs':
                    new_idx.append(idx)
                    continue
                
                base_var = idx.replace('_Mean', '').replace('_SD', '')
                unit = UNIT_MAP.get(base_var, '')
                if unit:
                    new_idx.append(f"{idx} ({unit})".replace('_', '\\_'))
                else:
                    new_idx.append(idx.replace('_', '\\_'))
            
            transposed.index = new_idx
            
            # Format as longtable with setstretch 1.0 to match other fragments
            latex_str = transposed.to_latex(longtable=True, float_format="%.3f")
            latex_str = "\\setstretch{1.0}\n" + latex_str
            
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(latex_str)
            with open(drafts_tex_path, 'w', encoding='utf-8') as f:
                f.write(latex_str)
        except Exception as e:
            print(f"Failed to generate latex: {e}")

    save_output(sum_nat, "Summary_National")
    save_output(sum_nat_yr, "Summary_National_by_Year")
    save_output(sum_bt, "Summary_BankType")
    save_output(sum_bt_yr, "Summary_BankType_by_Year")
    save_output(sum_reg, "Summary_Region")
    save_output(sum_reg_yr, "Summary_Region_by_Year")

    print("Generating D-Type firm summary...")
    d_firms = df_b[df_b["CODMUN_IBGE"].astype(str) == "0"].copy()
    d_firms["cnpj_8"] = pd.to_numeric(d_firms["CNPJ_Lider"], errors="coerce").astype("Int64").astype(str).str.zfill(8)

    path_flags = list(_ROOT.glob("**/digital_banks_diagnostic.csv"))[0]
    df_flags = pd.read_csv(path_flags)
    dig_cands = set(df_flags[df_flags["is_digital_candidate"] == True]["CNPJ_root"].astype(str).str.zfill(8))
    d_firms = d_firms[(d_firms["Source"] == "IFDATA") | (d_firms["cnpj_8"].isin(dig_cands))]

    path_ip = list(_ROOT.glob("**/ip_rates_quarterly.csv"))[0]
    df_ip = pd.read_csv(path_ip, low_memory=False)
    cnpjs_with_ip_rates = set(df_ip["CNPJ"].astype(str).str.zfill(8).unique())

    d_summ = d_firms.groupby(["CodConglomeradoPrudencial", "NomeInstituicao"]).agg(
        total_assets=("total_assets", "max"),
        first_year=("year", "min"),
        first_quarter=("quarter", "min"),
        obs=("year", "count")
    ).reset_index()

    d_summ = d_summ.sort_values(["total_assets", "first_year", "first_quarter"], ascending=[False, True, True])

    for idx, row in d_summ.iterrows():
        cp = row["CodConglomeradoPrudencial"]
        f_rows = d_firms[d_firms["CodConglomeradoPrudencial"] == cp]
        matches = f_rows["cnpj_8"].isin(cnpjs_with_ip_rates).any()
        d_summ.at[idx, "native_k5_rate"] = "Yes (IP Rates)" if matches else "No (COSIF Fallback)"

    d_summ["Asset_Size"] = d_summ["total_assets"].apply(lambda x: f"{x/1e9:.2f} B" if pd.notna(x) else "Unknown")
    
    out_csv = OUTPUT_DIR / "d_type_firms_summary_final.csv"
    d_summ.to_csv(out_csv, index=False)
    print(f"Saved D-Type summary to: {out_csv}")

    print(f"Descriptive statistics successfully saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()