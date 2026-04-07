import os
import pandas as pd
import numpy as np
import logging

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Read directly from BCB/ESTBAN/ESTBAN.csv to avoid any repo logic contamination
ESTBAN_CSV = os.path.join(BASE, "BCB", "ESTBAN", "ESTBAN.csv")
INTERMED_DIR = os.path.join(BASE, "BCB", "Egan_et_al_2025_Rep", "processed", "PANEL_INTERMED")
os.makedirs(INTERMED_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(INTERMED_DIR, "digital_banks_diagnostic.csv")

def main():
    if not os.path.exists(ESTBAN_CSV):
        logging.error(f"Raw ESTBAN data not found at: {ESTBAN_CSV}")
        return

    logging.info("Loading RAW ESTBAN file...")
    # ESTBAN has columns CNPJ, NOME_INSTITUICAO, CODMUN_IBGE, YEAR, MONTH, and many V_cols
    # Depositos a vista (V400_401), poupanca (V420), interfinanceiros (V431), a prazo (V432)
    # We will use pandas, only keeping needed cols to save memory
    cols_to_use = ["CNPJ", "NOME_INSTITUICAO", "CODMUN_IBGE", "YEAR", "MONTH"]
    v_cols = ["V400_401", "V420", "V431", "V432"]
    
    # Let's peek at the first row to see exactly what V cols exist just in case
    sample = pd.read_csv(ESTBAN_CSV, encoding="latin1", nrows=1)
    
    valid_cols = []
    for c in cols_to_use + v_cols:
        if c in sample.columns:
            valid_cols.append(c)
        else:
            logging.warning(f"Column '{c}' not found in ESTBAN.csv.")
            
    # Load just one recent cross-section to avoid memory issues (e.g. 2024 December)
    # If we read the whole file, it's huge. We'll read it in chunks unless we use usecols
    logging.info(f"Scanning for the latest available YEAR and MONTH in ESTBAN...")
    max_year = sample['YEAR'].iloc[0] if 'YEAR' in sample.columns else 2024
    
    # We will just load the full CSV specifically selecting the valid cols
    df = pd.read_csv(ESTBAN_CSV, usecols=valid_cols, encoding="latin1", low_memory=False)
    
    if df.empty:
        logging.error("ESTBAN was empty.")
        return
        
    for col in v_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0
            
    df["Total_Dep"] = df["V400_401"] + df["V420"] + df["V431"] + df["V432"]
    
    # Find latest period
    latest_year = df["YEAR"].max()
    latest_month = df[df["YEAR"] == latest_year]["MONTH"].max()
    logging.info(f"Analyzing cross-section from {latest_year}-{latest_month}")
    
    current_period = df[(df["YEAR"] == latest_year) & (df["MONTH"] == latest_month)].copy()
    
    if current_period.empty:
        logging.error("No data for the latest period.")
        return

    # Clean Name
    current_period["NOME_INSTITUICAO"] = current_period["NOME_INSTITUICAO"].astype(str).str.strip().str.upper()

    # Aggregate by CNPJ root (8 digits) and Municipality
    # (Just in case some report under different branches of the same CNPJ root, though ESTBAN usually groups by 8-digit root in the CNPJ column if it's standardized, let's cast it)
    current_period["CNPJ_root"] = pd.to_numeric(current_period["CNPJ"], errors='coerce').astype("Int64")
    
    # We group by CNPJ_root to find how many municipalities they report to
    mun_dep = current_period.groupby(["CNPJ_root", "NOME_INSTITUICAO", "CODMUN_IBGE"], as_index=False)[["Total_Dep", "V400_401", "V420", "V432"]].sum()
    
    # 1. Total deposits per CNPJ
    cong_totals = mun_dep.groupby(["CNPJ_root", "NOME_INSTITUICAO"], as_index=False)[["Total_Dep", "V400_401", "V420", "V432"]].sum()
    cong_totals.rename(columns={"Total_Dep": "Inst_Total_Dep", "V400_401": "Demand_Dep", "V420": "Savings_Dep", "V432": "Time_Dep"}, inplace=True)
    cong_totals["Retail_Dep"] = cong_totals["Demand_Dep"] + cong_totals["Savings_Dep"]
    
    # 2. Max deposits in a single municipality
    max_mun_dep = mun_dep.groupby("CNPJ_root", as_index=False)["Total_Dep"].max()
    max_mun_dep.rename(columns={"Total_Dep": "Max_Mun_Dep"}, inplace=True)
    
    # 3. Number of active municipalities
    mun_dep_positive = mun_dep[mun_dep["Total_Dep"] > 0]
    n_muns = mun_dep_positive.groupby("CNPJ_root", as_index=False)["CODMUN_IBGE"].nunique()
    n_muns.rename(columns={"CODMUN_IBGE": "N_mun"}, inplace=True)
    
    metrics = cong_totals.merge(max_mun_dep, on="CNPJ_root", how="left")
    metrics = metrics.merge(n_muns, on="CNPJ_root", how="left")
    
    metrics["N_mun"] = metrics["N_mun"].fillna(0).astype(int)
    metrics["Max_Share"] = np.where(
        metrics["Inst_Total_Dep"] > 0,
        metrics["Max_Mun_Dep"] / metrics["Inst_Total_Dep"],
        0.0
    )
    
    # Scale filter (percentiles of institutions with positive deposits)
    positive_deps = metrics[metrics["Inst_Total_Dep"] > 0]["Inst_Total_Dep"]
    p25 = positive_deps.quantile(0.25) if not positive_deps.empty else 0
    
    # Define flags
    def flag_digital(row):
        large_enough = row["Inst_Total_Dep"] > p25
        geo_flag = (row["N_mun"] > 0) and (row["N_mun"] <= 3) and (row["Max_Share"] > 0.95) and large_enough
        has_retail = row["Retail_Dep"] > 0
        
        if geo_flag and has_retail:
            return "Retail Digital Bank (Geo-Heuristic)"
        elif geo_flag and not has_retail:
            return "Pure Investment/Wholesale Bank (Geo-Heuristic)"
        else:
            return "Traditional / Other"
            
    metrics["Flag"] = metrics.apply(flag_digital, axis=1)
    
    # Enhance heuristic with BCB Regulatory Categorization
    import glob
    if_files = sorted(glob.glob(os.path.normpath(os.path.join(BASE, "BCB", "IF Data", "List", "IF_DATA_List*.csv"))))
    
    cnpj_map = {}
    for f in if_files:
        try:
            temp = pd.read_csv(f, sep=None, engine='python')
            if 'CnpjInstituicaoLider' not in temp.columns: continue
            temp['Cnpj_8'] = temp['CnpjInstituicaoLider'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(8)
            subset = temp[['Cnpj_8']].copy()
            if 'Atividade' in temp.columns: subset['Atividade'] = temp['Atividade']
            else: subset['Atividade'] = np.nan
            if 'SegmentoTb' in temp.columns: subset['SegmentoTb'] = temp['SegmentoTb']
            else: subset['SegmentoTb'] = np.nan
            
            subset = subset.dropna(how='all', subset=['Atividade', 'SegmentoTb'])
            subset = subset.drop_duplicates('Cnpj_8', keep='last')
            for _, r in subset.iterrows():
                if r['Cnpj_8'] not in cnpj_map: cnpj_map[r['Cnpj_8']] = {}
                if pd.notna(r['Atividade']): cnpj_map[r['Cnpj_8']]['Atividade'] = r['Atividade']
                if pd.notna(r['SegmentoTb']): cnpj_map[r['Cnpj_8']]['SegmentoTb'] = r['SegmentoTb']
        except Exception as e:
            logging.warning(f"Failed to read {f}: {e}")

    metrics['Cnpj_str_8'] = metrics['CNPJ_root'].astype(str).str.zfill(8)
    def apply_regulatory_rules(row):
        flag = row["Flag"]
        if flag != "Retail Digital Bank (Geo-Heuristic)": return flag
        
        info = cnpj_map.get(row['Cnpj_str_8'], {})
        ativ = str(info.get('Atividade', '')).strip().lower()
        seg = str(info.get('SegmentoTb', '')).strip().lower()
        
        whitelist_names = ['nupagamentos', 'inter', 'c6', 'nubank', 'picpay', 'pagseguro', 'mercadopago']
        blacklist_ativ = ['tesouraria e negócios', 'crédito atacado', 'filial estrangeiro', 'câmbio', 'tesouraria e negocios', 'credito atacado', 'cambio']
        blacklist_seg = ['banco comercial estrangeiro', 'sociedade distribuidora de tvm', 'banco de investimento', 'sociedade corretora de tvm', 'sociedade corretora de câmbio', 'agência de fomento']
        
        for w in whitelist_names:
            if w in str(row['NOME_INSTITUICAO']).lower():
                return "True Digital Retail Bank"
                
        for bd in blacklist_ativ:
            if bd in ativ:
                return "Wholesale/Investment Bank (Regulatory Filtered)"
        for bs in blacklist_seg:
            if bs in seg:
                return "Wholesale/Investment Bank (Regulatory Filtered)"
                
        # Additionally, if they report NO Savings AND they are not a clearly massive retail operation, we could flag them but let's rely strictly on regulatory tags.
        return "True Digital Retail Bank"
        
    metrics["Flag"] = metrics.apply(apply_regulatory_rules, axis=1)
    metrics["is_digital_candidate"] = metrics["Flag"] == "True Digital Retail Bank"

    
    # Reorder and Sort
    out_cols = ["CNPJ_root", "NOME_INSTITUICAO", "Inst_Total_Dep", "Demand_Dep", "Savings_Dep", "Retail_Dep", "Time_Dep", "Max_Mun_Dep", "N_mun", "Max_Share", "Flag", "is_digital_candidate"]
    out_df = metrics[out_cols].sort_values(by=["is_digital_candidate", "Inst_Total_Dep"], ascending=[False, False])
    
    # Drop zero-deposit ones visually to clear noise
    out_df = out_df[out_df["Inst_Total_Dep"] > 0]

    out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logging.info(f"Report exported to: {OUTPUT_PATH}")

    digital_count = sum(out_df["is_digital_candidate"])
    print(f"\n--- Raw ESTBAN Diagnostic Complete ---")
    print(f"Total Institutions analyzing: {len(out_df)}")
    print(f"Digital candidates (1-5 Municipalities & >95% share): {digital_count}")
    print(f"Sample Digital candidates:")
    top_cands = out_df[out_df["is_digital_candidate"]][["NOME_INSTITUICAO", "Inst_Total_Dep", "N_mun"]].head(10)
    for _, r in top_cands.iterrows():
        print(f"  - {r['NOME_INSTITUICAO']} (Mun_Count: {r['N_mun']}, Dep: {r['Inst_Total_Dep']/1e9:.2f}B)")

if __name__ == "__main__":
    main()
