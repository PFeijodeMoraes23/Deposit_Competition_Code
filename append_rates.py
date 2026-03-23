import os
import sys
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import numpy as np
import logging
import time

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

sys.path.append(os.path.dirname(__file__))
import egan_panel_build as egb

# Paths (anchored to project root, two levels above this script)
BASE         = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEPOSITS_CSV = os.path.join(BASE, "BCB", "Egan_et_al_2025_Rep", "processed", "deposits_panel.csv")

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "append_rates",
        {"deposits_csv": DEPOSITS_CSV},
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    DEPOSITS_CSV = _paths["deposits_csv"]

def append_rates_to_panel():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(f"Adding rates and spreads to {DEPOSITS_CSV}...")

    df = pd.read_csv(DEPOSITS_CSV, low_memory=False)
    
    # 1. Add AnoMes
    df["AnoMes"] = df["Year"].astype(int) * 100 + df["Quarter"].astype(int) * 3
    
    # 2. National lags
    groupby_cols = ["CodConglomeradoPrudencial", "AnoMes"]
    nat_stocks = df.groupby(groupby_cols)[["dep_a2", "dep_a3", "dep_a4", "dep_a5"]].sum().reset_index()
    nat_stocks = nat_stocks.sort_values(groupby_cols)
    
    for col in ["dep_a2", "dep_a3", "dep_a4", "dep_a5"]:
        nat_stocks[f"lag_{col}"] = nat_stocks.groupby("CodConglomeradoPrudencial")[col].shift(1).fillna(0.0)
    
    # 3. Macro 
    df_macro = egb.load_macro_series()
    df_macro_q = egb.make_quarterly_macro(df_macro)
    
    # 4. COSIF
    df_congl_map = egb.load_conglomerate_mapping()
    df_cosif = egb.load_cosif_processed()
    df_cosif_q = egb.aggregate_cosif_to_conglomerates(df_cosif, df_congl_map)
    # The output of aggregate_cosif_to_conglomerates has 'year' and 'quarter', we need AnoMes inside it
    if "AnoMes" not in df_cosif_q.columns and "year" in df_cosif_q.columns and "quarter" in df_cosif_q.columns:
        df_cosif_q["AnoMes"] = df_cosif_q["year"] * 100 + df_cosif_q["quarter"] * 3
        
    aux = nat_stocks.merge(df_macro_q[["AnoMes", "selic_qoq", "cdi_qoq", "savings_rate_qoq", "meta_selic"]], on="AnoMes", how="left")
    aux = aux.merge(df_cosif_q, on=["CodConglomeradoPrudencial", "AnoMes"], how="left")
    
    for col in ["cosif_desp_captacao", "cosif_implicit_rate", "ip_prepaid_rate"]:
        if col not in aux.columns:
            aux[col] = np.nan
            
    # T2 & T3 imputed expenses
    _imputed_t2 = aux["savings_rate_qoq"].fillna(0.0) * aux["lag_dep_a2"]
    _imputed_t3 = aux["cdi_qoq"].fillna(0.0)          * aux["lag_dep_a3"]
    _residual = (aux["cosif_desp_captacao"] - _imputed_t2 - _imputed_t3).clip(lower=0.0)
    
    _denom_t45 = aux["lag_dep_a4"] + aux["lag_dep_a5"]
    _denom_t45 = _denom_t45.replace(0.0, np.nan)
    
    _type4_raw = _residual.astype(float) / _denom_t45.astype(float)
    _type4_raw.loc[~np.isfinite(_type4_raw)] = np.nan
    _type4_raw.loc[_type4_raw > 0.5] = np.nan
    aux["cosif_type4_rate"] = _type4_raw
    
    # Median IP
    if "has_ip" not in aux.columns:
        if "is_ip" in df_congl_map.columns:
            ip_check = df_congl_map[df_congl_map["is_ip"] == 1]["CodConglomeradoPrudencial"].unique()
            aux["has_ip"] = aux["CodConglomeradoPrudencial"].isin(ip_check).astype(int)
        else:
            aux["has_ip"] = 0

    if "ip_prepaid_rate" in aux.columns:
        _ip_mask = aux["ip_prepaid_rate"].notna() & (aux["ip_prepaid_rate"] > 0)
        # Apply median directly over AnoMes
        valid_ip = aux[_ip_mask].copy()
        
        # We need groups that have at least 2 entries.
        sizes = valid_ip.groupby("AnoMes").size()
        valid_anomes = sizes[sizes >= 2].index
        
        medians = valid_ip[valid_ip["AnoMes"].isin(valid_anomes)].groupby("AnoMes")["ip_prepaid_rate"].median()
        aux = aux.merge(medians.rename("median_ip_rate"), on="AnoMes", how="left")
    else:
        aux["median_ip_rate"] = np.nan
        
    aux["rate_a1"] = 0.0
    aux["rate_a2"] = aux["savings_rate_qoq"]
    aux["rate_a3"] = aux["cdi_qoq"]
    aux["rate_a4"] = aux["cosif_type4_rate"].fillna(aux["cosif_implicit_rate"]).fillna(aux["cdi_qoq"])
    
    if "ip_prepaid_rate" in aux.columns:
        aux["rate_a5"] = np.where(
            aux["has_ip"] == 1,
            aux["ip_prepaid_rate"].fillna(aux["median_ip_rate"]).fillna(aux["cdi_qoq"]),
            aux["median_ip_rate"].fillna(aux["cdi_qoq"])
        )
    else:
        aux["rate_a5"] = aux["median_ip_rate"].fillna(aux["cdi_qoq"])
        
    aux["risk_free_qoq"] = aux["selic_qoq"]
    
    for t in [1, 2, 3, 4, 5]:
        aux[f"spread_a{t}"] = aux["risk_free_qoq"] - aux[f"rate_a{t}"]
    
    drop_cols = ["AnoMes", "risk_free_qoq", "rate_a1", "rate_a2", "rate_a3", "rate_a4", "rate_a5",
                 "spread_a1", "spread_a2", "spread_a3", "spread_a4", "spread_a5"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
                 
    cols_to_merge = ["CodConglomeradoPrudencial", "AnoMes", "risk_free_qoq"] + \
                    [f"rate_a{t}" for t in range(1, 6)] + \
                    [f"spread_a{t}" for t in range(1, 6)]
                    
    df["AnoMes"] = df["Year"].astype(int) * 100 + df["Quarter"].astype(int) * 3
    df = df.merge(aux[cols_to_merge], on=["CodConglomeradoPrudencial", "AnoMes"], how="left")
    df = df.drop(columns=["AnoMes"])
    
    df.to_csv(DEPOSITS_CSV, index=False)
    logging.info(f"Updated {DEPOSITS_CSV} with rates and spreads.")

if __name__ == "__main__":
    t0 = time.perf_counter()
    append_rates_to_panel()
    print(f"Done in {time.perf_counter() - t0:.2f}s")

