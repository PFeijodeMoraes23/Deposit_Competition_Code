
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import pickle
import json
from pathlib import Path

code_dir = Path(r'C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Code\Egan_et_al_2025_Rep')
DATA_DIR = code_dir.parents[1] / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed'
OUTPUT_DIR = DATA_DIR / 'ESTIMATION_OUTPUT' / 'SLEEPINESS' / 'SLEEPINESS_NEW'

with open(OUTPUT_DIR / 'estimation_results.pkl', 'rb') as f:
    results_dict = pickle.load(f)
with open(OUTPUT_DIR / 'cluster_diagnostics.json', 'r') as f:
    cluster_data = json.load(f)

G = cluster_data['G_nominal']
G_star = cluster_data['G_star']

def stars(p):
    if p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    return ''

def clean_name(v):
    return v.replace('interaction_', '').replace('_', '\\_')

columns = [
    ('Spec1-OLS', 'OLS'), 
    ('Spec2-IV_CostShifters', 'IV Cost'), 
    ('Spec3-IV_Wholesale', 'IV Wholesale'), 
    ('Spec4-IV_HausmanFull', 'Hausman')
]
panels = [
    ('Base', ['nr_lagged_dep']), 
    ('Macro', ['nr_lagged_dep', 'interaction_gdp_per_capita', 'interaction_cadunico_families_per1000', 'interaction_fraction_65plus', 'interaction_fraction_young']), 
    ('Tech', ['nr_lagged_dep', 'interaction_gdp_per_capita', 'interaction_cadunico_families_per1000', 'interaction_fraction_65plus', 'interaction_fraction_young', 'interaction_pix_users_pf_per1000', 'interaction_connections_per100', 'interaction_branches_per1000'])
]

print("\\begin{table}[htbp]\\centering")
print("\\caption{Second Stage Estimation}")
print("\\begin{tabular}{l" + "c"*4 + "}\\toprule")
print(" & " + " & ".join([n for _, n in columns]) + " \\\\ \\midrule")

for p_name, p_vars in panels:
    print(f"\\multicolumn{{5}}{{c}}{{\\textbf{{Panel: {p_name}}}}} \\\\ \\midrule")
    for var in p_vars:
        coef_strs = []
        se_strs = []
        for col_key, _ in columns:
            spec_key = f"{col_key} x {p_name}"
            res = results_dict[spec_key]['second_stage']
            if var in res.params:
                c = res.params[var]
                se = res.bse[var]
                p = res.pvalues[var]
                coef_strs.append(f"${c:.4f}^{{{stars(p)}}}$")
                se_strs.append(f"$({se:.4f})$")
            else:
                coef_strs.append("")
                se_strs.append("")
        print(f"{clean_name(var)} & " + " & ".join(coef_strs) + " \\\\")
        print(f" & " + " & ".join(se_strs) + " \\\\")
    
    # stats
    obs_strs = []
    rsq_strs = []
    for col_key, _ in columns:
        res = results_dict[f"{col_key} x {p_name}"]['second_stage']
        obs_strs.append(f"{int(res.nobs):,}")
        rsq_strs.append(f"{res.rsquared:.4f}")
    
    print("\\midrule")
    print("Obs & " + " & ".join(obs_strs) + " \\\\")
    print("$R^2$ & " + " & ".join(rsq_strs) + " \\\\")
    print("Fixed Effects & Yes & Yes & Yes & Yes \\\\")
    print(f"Clusters (G) & {G} & {G} & {G} & {G} \\\\")
    print(f"Effective Clusters ($G^*$) & {G_star:.2f} & {G_star:.2f} & {G_star:.2f} & {G_star:.2f} \\\\")
    if p_name != 'Tech':
        print("\\midrule")

print("\\bottomrule")
print("\\end{tabular}\\end{table}")
