# BLP Estimation Loop Setup & Execution (HPC)

The BLP estimation pipeline is now partitioned to protect against RAM exhaustion. 

1. **[estimation_1_demand_2_secondprep.py](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20%282024%20-%202025%29/Open%20Finance/Open-Finance/Code/Egan_et_al_2025_Rep/estimation_1_demand_2_secondprep.py)** 
   *Run locally.* This consumes the large 1.4 GB `market_panel.csv`, loops over specifications, merges the data with the prep files, and outputs memory-efficient binary pickles inside `processed/ESTIMATION_OUTPUT/DEMAND_PREP/`.
2. **[estimation_1_demand_3_loop.py](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20%282024%20-%202025%29/Open%20Finance/Open-Finance/Code/Egan_et_al_2025_Rep/estimation_1_demand_3_loop.py)** 
   *Run on HPC.* This grabs the pre-aggregated [.pkl](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20%282024%20-%202025%29/Open%20Finance/Open-Finance/BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/BLP_RESULTS/blp_results_spec_1_logit.pkl) chunks directly. It never loads the master panel CSV, saving significant overhead during multiprocessing and inner loop optimization.

---

### Instructions for HPC

1. Make sure you have transferred the generated `demand_final_spec_{X}.pkl` files (located in the `DEMAND_PREP` folder) to the HPC along with your scripts.
2. Ensure `statsmodels`, `scipy`, `pandas`, and `numpy` are installed in the HPC Python environment.
3. The script will automatically detect the number of available CPU cores (via `multiprocessing.cpu_count()`) and spawn workers up to that amount (cap of 6 by default) to solve the specifications in parallel.

### Bash Command

To run the loop across all specifications (`--spec all`) and automatically flow through all parameter build-up stages sequentially (`--stage sequence`), run the following command. Note that `--R` defaults to `100` for testing, but since you are on the HPC, you can override this flag to `500` or `1000` for final rigorous results.

```bash
# Basic test (R=100)
python estimation_1_demand_3_loop.py --spec all --stage sequence

# Final publication-grade estimates (R=1000, as recommended by Conlon & Gortmaker)
# Add nohup to keep it running in the background if your ssh session drops.
nohup python estimation_1_demand_3_loop.py --spec all --stage sequence --R 1000 > blp_hpc_output.log 2>&1 &
```

### Outputs to monitor
While the job runs, it will spawn [.pkl](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20%282024%20-%202025%29/Open%20Finance/Open-Finance/BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/BLP_RESULTS/blp_results_spec_1_logit.pkl) and [.json](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20%282024%20-%202025%29/Open%20Finance/Open-Finance/BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/BLP_RESULTS/blp_summary_logit.json) objects in:  
`processed/ESTIMATION_OUTPUT/BLP_RESULTS/`

The [.json](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20%282024%20-%202025%29/Open%20Finance/Open-Finance/BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/BLP_RESULTS/blp_summary_logit.json) outputs ([blp_summary_logit.json](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20%282024%20-%202025%29/Open%20Finance/Open-Finance/BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/BLP_RESULTS/blp_summary_logit.json), `..._sigma.json`, etc.) are plain-text and highly readable, allowing you to quickly tail or [cat](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20%282024%20-%202025%29/Open%20Finance/Open-Finance/Code/Egan_et_al_2025_Rep/estimation_1_demand_1_prep.py#186-300) them on the HPC node to check convergence progress across all your specifications.
