import pandas as pd
import numpy as np
import os

PANEL_CSV = r"c:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\BCB\Egan_et_al_2025_Rep\processed\market_panel.csv"

def calculate_effective_clusters():
    df = pd.read_csv(PANEL_CSV)
    
    # Filter for B-Type Institutions (same as the estimation script)
    df = df[df['CODMUN_IBGE'] > 0].copy()
    
    # Check the unique clusters
    cluster_col = 'CodConglomeradoPrudencial'
    
    # Calculate observations per cluster
    Ns = df.groupby(cluster_col).size()
    G = len(Ns)
    
    # Carter, Schnepel, Steigerwald (2017) Effective Number of Clusters
    # Approximation based on unbalanced cluster sizes
    mean_Ng = np.mean(Ns)
    std_Ng = np.std(Ns, ddof=0)
    
    cv_Ng = std_Ng / mean_Ng
    G_star = G / (1 + (cv_Ng ** 2))
    
    print(f"Total Nominal Clusters (G) = {G}")
    print(f"Mean Obs per Cluster = {mean_Ng:.2f}")
    print(f"Std Dev of Obs per Cluster = {std_Ng:.2f}")
    print(f"Coefficient of Variation (cv) = {cv_Ng:.2f}")
    print(f"Effective Number of Clusters (G*) = {G_star:.2f}")
    
    # Heavy tails check
    top_5 = Ns.sort_values(ascending=False).head(5)
    total_obs = len(df)
    print("\nTop 5 Clusters Observation Shares:")
    for c, n in top_5.items():
        print(f"Cluster {c}: {n} obs ({(n / total_obs) * 100:.2f}%)")
        
calculate_effective_clusters()
