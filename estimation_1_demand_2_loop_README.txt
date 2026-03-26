================================================================================
estimation_1_demand_2_loop.py : EXCRUCIATING DETAIL README
================================================================================

This document breaks down the mathematical and programmatic pipeline implemented in 
the `estimation_1_demand_2_loop.py` script. The script performs the Berry, Levinsohn, 
and Pakes (1995) Random Coefficients Logit Demand Estimation, heavily customized 
to account for the unique spatial topology of Branchless (NB) vs. Branch (B) banks 
in the Brazilian banking system (per Egan et al.).

The script uses a Generalized Method of Moments (GMM) framework where:
  - Outer Loop: Searches for non-linear parameters $\theta_2$ (random coefficients $\Sigma$ 
    and demographic interactions $\Pi$).
  - Inner Loop: Solves the contraction mapping to find the mean utilities $\delta$ 
    that match observed conditional market shares.

--------------------------------------------------------------------------------
1. CLI Arguments and Sequential Build-Up (Conlon & Gortmaker, 2020)
--------------------------------------------------------------------------------
The estimation employs sequential build-up to aid the non-linear optimizer:
  - `--stage logit` : Sets $\theta_2 = 0$. Solves standard multinomial logit via 2SLS.
  - `--stage sigma` : Only allows price/spread variance ($\sigma_{spread}$). $\Pi=0$.
  - `--stage full`  : Turns on $\Sigma$ for size and $\Pi$ for core demographics. 
  - `--stage extended`: Allows custom product characteristic by demographic interactions.

Other CLI flags:
  - `--spec`: Chooses the sleepiness specification (1-12) to estimate.
  - `--R`: Number of Monte Carlo draws for the integrals (default: 200). 
  - `--tol-inner`: Strict tolerance for contraction (default: 1e-14).
  - `--method`: scipy minimize routine (L-BFGS-B or Nelder-Mead).

--------------------------------------------------------------------------------
2. Data Pre-Processing (`load_panel_selective`, `merge_panel_with_prep`)
--------------------------------------------------------------------------------
The script merges `market_panel.csv` (which contains product characteristics, BLP 
Leave-One-Out (LOO) instruments, and MCA-level demographic indices) with 
`demand_prep_spec_{X}.csv` (which holds the conditional shares processed via the 
first-stage sleepiness equations).

NOTE ON DEPOSIT TYPES (k): The script loops over `K_LIST = [1, 2, 4, 5]`. Interbank 
deposits (`k=3`) are deliberately excluded from estimation as they follow a different 
competitive mechanism than retail deposits.

--------------------------------------------------------------------------------
3. Simulation Draws (`generate_halton_draws`, `generate_demographic_draws`)
--------------------------------------------------------------------------------
To integrate out unobserved consumer heterogeneity:
  - a) Halton Draws ($\nu$): $R$ draws of uniform quasi-random sequences transformed 
       by the inverse normal CDF. These $R$ draws are fixed across all markets to 
       ensure smooth objective functions globally. Dimension: `L_PROD + K_TYPES` (12).
  - b) Demographic Draws ($d$): Normal distributions $N(\mu_{mt}, \sigma.10)$ sampled 
       using local MCA characteristics. 
       
--------------------------------------------------------------------------------
4. Structural Setup (`build_theta2_structure`, `unpack_theta2`)
--------------------------------------------------------------------------------
Based on `--stage`, the script maps a flat vector $\theta_2$ into:
  - `sigma_indices`: Which variables (e.g., spread, assets) have an unobserved variance.
  - `pi_interactions`: Explicit tuples matching a product characteristic to a demographic 
    (e.g., Spread x GDP_per_capita).

--------------------------------------------------------------------------------
5. Computing Individual Utilities (`compute_mu`)
--------------------------------------------------------------------------------
Calculates $\mu_{rjkmt} = [x_{jt}, \rho_k]' (\Sigma \nu_r + \Pi d_{rmt})$
  - A $N \times R$ array representing the deviation from mean utility $\delta$ for 
    each simulated consumer $r$ assessing product $(j,k)$ in market $m$ at time $t$.
  - The script uses the selector matrix $T_k$ by dynamically turning on the column 
    corresponding to deposit type $k$ and zeroing the others.

--------------------------------------------------------------------------------
6. Inner Contraction Mapping (`compute_model_shares`, `blp_contraction`)
--------------------------------------------------------------------------------
This is the workhorse of the algorithm and features the model's most critical 
departure from standard BLP topology:
  
  a) `compute_model_shares`:
      - Applies the softmax function: $ \exp(\delta + \mu) / \sum \exp(\delta+\mu) $
      - Prevents numerical overflow via the log-sum-exp trick (subtracting max val).
      - Computes local shares across $R$ individuals.
      - FOR NB-TYPE: It does not sum purely locally; it calculates $s^{NB} = \sum_{m} (M_{mt}/M_t) s^{NB}_{jkmt}$, aggregating demand nationally to match the national observed share.
      - Computes $\Omega_{mt} = 1 - \sum s^{NB}$: The residual share allocated to local B-type banks.

  b) `blp_contraction`:
      - Standard BLP requires subtraction of an outside good. Because active depositors 
        sum to 1 (the outside option is effectively non-participation/sleepiness), 
        mean utility levels are anchored directly to the conditional market shares.
      - Updates NB-banks nationally: $\delta_{h+1} = \delta_h + \ln(s^{Data}) - \ln(s^{Model})$
      - Updates B-banks locally with the $\Omega$ restraint: 
        $\delta_{h+1} = \delta_h + \ln(s^{B|B, Data}) + \ln(\Omega) - \ln(s^{Model})$
      - Iterates until $\max|\delta_{h+1} - \delta_h| < 1e-14$.

--------------------------------------------------------------------------------
7. Linear IV & First Stage Projection (`estimate_theta1`)
--------------------------------------------------------------------------------
Given converged $\delta(\theta_2)$, we isolate the linear parameters $\theta_1$:
  - $\delta_{jkmt} = x_{jt}\bar{\beta} + \bar{\alpha}_k \rho_{jkmt} + \xi_{jkmt}$
  - Endogeneity: Spreads for buckets k=4, 5 are likely correlated with unobserved 
    quality $\xi$. 
  - 1st Stage: The script projects the endogenous spread arrays onto the matrix 
    of excluded instruments $H$ (BLP LOO IVs + Cost Shifter IVs + Capital IVs), generating $\hat{spread}$.
  - 2nd Stage: Ordinarly Least Squares regress $\delta$ on exogenous $X$ and $\hat{spread}$ 
    to obtain unbiased $\theta_1$ and structural errors $\xi$.
  - Also calculates IK2016 finite-cluster-robust standard errors for $\theta_1$ here.

--------------------------------------------------------------------------------
8. GMM Outer Loop (`compute_gmm_moments`, `gmm_objective`, `run_blp_for_spec`)
--------------------------------------------------------------------------------
  - Moments: $G(\theta_2) = \frac{1}{N} \sum \xi_{jkmt}(\theta_2) \cdot Z_{jkmt}$
    (The structural errors must be orthogonal to exogenous instruments).
  - Weight Matrix W: Initialized as $(Z'Z)^{-1}$ (2SLS).
  - Minimizer (`scipy.optimize.minimize`): Searches the $\theta_2$ space to minimize 
    $Q = G' W G$, repeatedly calling the Inner Loop and Linear IV steps at each guess.
  - Final Variance/Standard Errors: Computes the Sandwich Variance matrix for $\theta_2$ 
    using a finite difference numerical Jacobian $D = \partial G / \partial \theta_2$.

--------------------------------------------------------------------------------
9. Save and Exit
--------------------------------------------------------------------------------
Stores a pickle file (`blp_results_spec_{id}.pkl`) capturing:
  - Point estimates ($\theta_1, \theta_2$) and Cluster Robust Standard Errors.
  - Final converged vectors ($\delta, \xi$).
  - GMM diagnostic values ($Q$). 
  - Convergence metadata for robustness checking.
