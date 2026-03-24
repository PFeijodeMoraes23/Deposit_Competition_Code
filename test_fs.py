import pickle
from pathlib import Path
code_dir = Path(r'C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Code\Egan_et_al_2025_Rep')
DATA_DIR = code_dir.parents[1] / 'BCB' / 'Egan_et_al_2025_Rep' / 'processed'
OUTPUT_DIR = DATA_DIR / 'ESTIMATION_OUTPUT' / 'SLEEPINESS_NEW'
with open(OUTPUT_DIR / 'estimation_results.pkl', 'rb') as f:
    results_dict = pickle.load(f)
for spec in ['Spec2-IV_CostShifters x Base', 'Spec3-IV_Wholesale x Base', 'Spec4-IV_HausmanFull x Base']:
    res = results_dict[spec]['first_stage']
    print(list(res.params.index))
