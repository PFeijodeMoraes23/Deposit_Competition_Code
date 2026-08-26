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

from utils import paths

BASE = str(paths.OPEN_FINANCE)

# Read directly from the consolidated ESTBAN.csv to avoid any repo logic contamination
ESTBAN_CSV = str(paths.ESTBAN_CSV)
INTERMED_DIR = os.path.join(str(paths.PROCESSED), "PANEL_INTERMED")
os.makedirs(INTERMED_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(INTERMED_DIR, "digital_banks_diagnostic.csv")

# Per-institution physical access points, built by scrape_25_bcb_access_points.py from the
# BCB Informes services (agencias / postos de atendimento / correspondentes).
ACCESS_POINTS_CSV = os.path.join(str(paths.INCLUSION_DIR), "bcb_access_points_by_institution.csv")

# Physical-network counts are RECORDED for every institution (see load_access_points) but do
# NOT decide the B/D split. The B category in this pipeline means a bank whose deposits sit in
# a local market, and that is a statement about deposit BOOKING, which is what the geo test
# below measures. An institution booking its whole balance at one head office has no local
# market to belong to however many storefronts it runs, so a wide service network is not on its
# own grounds to call it brick-and-mortar.
#
# Setting DIGITAL_MAX_OWN_MUNS to a positive integer turns the network evidence into a demotion
# rule: a candidate operating its own points (agencias + postos de atendimento, NOT
# correspondentes) in more than that many municipalities is reclassified brick-and-mortar. The
# candidate distribution is strongly bimodal -- Agibank at 679 municipalities, then a gap down
# to 18 (Banco PAN), 14 (C6, BTG) and 8 or fewer for the rest -- so any value between about 20
# and 600 isolates Agibank alone. Correspondentes are excluded from the count because a purely
# digital bank can contract thousands for cash-in/cash-out without holding any premises
# (Banco Votorantim has 14,844 against a single own-network municipality).
#
# Unset (the default) means the network evidence is reported and never acts.
_max_own = os.environ.get("DIGITAL_MAX_OWN_MUNS", "").strip()
DIGITAL_MAX_OWN_MUNS = int(_max_own) if _max_own else None


def load_access_points() -> pd.DataFrame:
    """Per-institution own-network reach, keyed on the 8-digit CNPJ root.

    Returns an empty frame with the expected columns when the file is absent, so the
    deposit-side heuristic still runs (it just cannot demote anyone).
    """
    cols = ["cnpj8", "n_agencias", "n_postos", "n_own_points", "n_mun_own", "n_correspondentes"]
    if not os.path.exists(ACCESS_POINTS_CSV):
        logging.warning(
            f"Access-point file not found at {ACCESS_POINTS_CSV} -- the digital verdict will "
            f"rest on deposit booking alone. Run scrape_25_bcb_access_points.py.")
        return pd.DataFrame(columns=cols)
    ap = pd.read_csv(ACCESS_POINTS_CSV, dtype={"cnpj8": str})
    keep = [c for c in cols if c in ap.columns]
    ap = ap[keep].copy()
    ap["cnpj8"] = ap["cnpj8"].astype(str).str.zfill(8)
    logging.info(f"Access points: {len(ap):,} institutions from {os.path.basename(ACCESS_POINTS_CSV)}")
    return ap


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
    # Vectorised flag_digital (replaces row-wise apply)
    large_enough = metrics["Inst_Total_Dep"] > p25
    geo_flag = (metrics["N_mun"] > 0) & (metrics["N_mun"] <= 3) & (metrics["Max_Share"] > 0.95) & large_enough
    has_retail = metrics["Retail_Dep"] > 0
    metrics["Flag"] = np.select(
        [geo_flag & has_retail, geo_flag & ~has_retail],
        ["Retail Digital Bank (Geo-Heuristic)", "Pure Investment/Wholesale Bank (Geo-Heuristic)"],
        default="Traditional / Other"
    )
    
    # Enhance heuristic with BCB Regulatory Categorization
    import glob
    if_files = sorted(glob.glob(os.path.join(str(paths.IF_DATA_LIST), "IF_DATA_List*.csv")))
    
    cnpj_map = {}
    for f in if_files:
        try:
            temp = pd.read_csv(f, sep=None, engine='python')
            if 'CnpjInstituicaoLider' not in temp.columns: continue
            temp['Cnpj_8'] = temp['CnpjInstituicaoLider'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(8)
            col_ativ = 'Atividade' if 'Atividade' in temp.columns else None
            col_seg  = 'SegmentoTb' if 'SegmentoTb' in temp.columns else None
            keep = ['Cnpj_8'] + [c for c in [col_ativ, col_seg] if c]
            subset = temp[keep].drop_duplicates('Cnpj_8', keep='last')
            for rec in subset.to_dict('records'):
                k = rec['Cnpj_8']
                if k not in cnpj_map: cnpj_map[k] = {}
                if col_ativ and pd.notna(rec.get(col_ativ)): cnpj_map[k]['Atividade'] = rec[col_ativ]
                if col_seg  and pd.notna(rec.get(col_seg)):  cnpj_map[k]['SegmentoTb'] = rec[col_seg]
        except Exception as e:
            logging.warning(f"Failed to read {f}: {e}")

    metrics['Cnpj_str_8'] = metrics['CNPJ_root'].astype(str).str.zfill(8)

    whitelist_names = ['nupagamentos', 'inter', 'c6', 'nubank', 'picpay', 'pagseguro', 'mercadopago', 'btg', 'bs2', 'genial', 'sofisa', 'rendimento']
    blacklist_ativ = ['tesouraria e negócios', 'crédito atacado', 'filial estrangeiro', 'câmbio', 'tesouraria e negocios', 'credito atacado', 'cambio']
    blacklist_seg = ['banco comercial estrangeiro', 'sociedade distribuidora de tvm', 'banco de investimento', 'sociedade corretora de tvm', 'sociedade corretora de câmbio', 'agência de fomento']

    # Attach regulatory info from cnpj_map
    metrics['_ativ'] = metrics['Cnpj_str_8'].map(lambda k: str(cnpj_map.get(k, {}).get('Atividade', '')).strip().lower())
    metrics['_seg']  = metrics['Cnpj_str_8'].map(lambda k: str(cnpj_map.get(k, {}).get('SegmentoTb',  '')).strip().lower())
    metrics['_nome'] = metrics['NOME_INSTITUICAO'].str.lower()

    # Only modify the geo-flagged retail digital candidates
    is_candidate = metrics['Flag'] == 'Retail Digital Bank (Geo-Heuristic)'
    in_whitelist = is_candidate & metrics['_nome'].apply(lambda n: any(w in n for w in whitelist_names))
    in_bl_ativ   = is_candidate & metrics['_ativ'].apply(lambda a: any(b in a for b in blacklist_ativ))
    in_bl_seg    = is_candidate & metrics['_seg'].apply(lambda s: any(b in s for b in blacklist_seg))

    wholesale_mask = is_candidate & (in_bl_ativ | in_bl_seg) & ~in_whitelist
    true_digital_mask = is_candidate & ~wholesale_mask

    metrics.loc[true_digital_mask, 'Flag'] = 'True Digital Retail Bank'
    metrics.loc[wholesale_mask,    'Flag'] = 'Wholesale/Investment Bank (Regulatory Filtered)'
    metrics.drop(columns=['_ativ', '_seg', '_nome'], inplace=True)

    # -- Physical-network evidence -------------------------------------------------------
    # Everything above reads deposit BOOKING. This step reads the actual retail footprint and
    # demotes any candidate that turns out to operate premises across many municipalities.
    ap = load_access_points()
    metrics["cnpj8"] = metrics["CNPJ_root"].astype(str).str.zfill(8)
    metrics = metrics.merge(ap, on="cnpj8", how="left")
    for c in ["n_agencias", "n_postos", "n_own_points", "n_mun_own", "n_correspondentes"]:
        if c not in metrics.columns:
            metrics[c] = 0
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce").fillna(0).astype(int)
    metrics["has_access_point_data"] = (metrics["cnpj8"].isin(set(ap["cnpj8"]))
                                        if len(ap) else False)

    geo_digital = metrics["Flag"] == "True Digital Retail Bank"
    wide_network = geo_digital & (metrics["n_mun_own"] > (DIGITAL_MAX_OWN_MUNS
                                                         if DIGITAL_MAX_OWN_MUNS is not None
                                                         else float("inf")))
    if DIGITAL_MAX_OWN_MUNS is not None:
        metrics.loc[wide_network, "Flag"] = "Brick-and-Mortar (Physical Network)"

    # Report the wide-network candidates either way: whether or not the rule is armed, an
    # institution booking centrally while running premises in hundreds of municipalities is
    # worth seeing in the log.
    _wide = geo_digital & (metrics["n_mun_own"] > 25)
    for _, r in metrics[_wide].sort_values("n_mun_own", ascending=False).iterrows():
        verb = "demoted to brick-and-mortar" if DIGITAL_MAX_OWN_MUNS is not None else                "kept digital (DIGITAL_MAX_OWN_MUNS unset), wide physical network"
        logging.info(
            f"  {verb}: {r['NOME_INSTITUICAO']} "
            f"({r['n_agencias']} agencias + {r['n_postos']} postos across {r['n_mun_own']} "
            f"municipalities; books deposits in {r['N_mun']})")

    # THE VERDICT. panel_6_market.load_digital_conglomerates reads this column and nothing else.
    metrics["is_digital_candidate"] = metrics["Flag"] == "True Digital Retail Bank"

    
    # Reorder and Sort
    out_cols = ["CNPJ_root", "NOME_INSTITUICAO", "Inst_Total_Dep", "Demand_Dep", "Savings_Dep", "Retail_Dep", "Time_Dep", "Max_Mun_Dep", "N_mun", "Max_Share",
                "n_agencias", "n_postos", "n_own_points", "n_mun_own", "n_correspondentes", "has_access_point_data",
                "Flag", "is_digital_candidate"]
    out_df = metrics[out_cols].sort_values(by=["is_digital_candidate", "Inst_Total_Dep"], ascending=[False, False])
    
    # Drop zero-deposit ones visually to clear noise
    out_df = out_df[out_df["Inst_Total_Dep"] > 0]

    import pyarrow as pa
    import pyarrow.csv as pa_csv
    pa_csv.write_csv(pa.Table.from_pandas(out_df, preserve_index=False), OUTPUT_PATH)
    logging.info(f"Report exported to: {OUTPUT_PATH}")

    digital_count = sum(out_df["is_digital_candidate"])
    print(f"\n--- Raw ESTBAN Diagnostic Complete ---")
    print(f"Total Institutions analyzing: {len(out_df)}")
    _rule = (f"AND own network <= {DIGITAL_MAX_OWN_MUNS} municipalities"
             if DIGITAL_MAX_OWN_MUNS is not None else "(network evidence recorded, not applied)")
    print(f"Digital candidates (booking geo-test {_rule}): {digital_count}")
    print(f"Demoted by physical network: {int((out_df['Flag'] == 'Brick-and-Mortar (Physical Network)').sum())}")
    print(f"Sample Digital candidates:")
    top_cands = out_df[out_df["is_digital_candidate"]][["NOME_INSTITUICAO", "Inst_Total_Dep", "N_mun"]].head(10)
    for _, r in top_cands.iterrows():
        print(f"  - {r['NOME_INSTITUICAO']} (Mun_Count: {r['N_mun']}, Dep: {r['Inst_Total_Dep']/1e9:.2f}B)")

if __name__ == "__main__":
    main()
