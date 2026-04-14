# Sleepiness Function Estimation Strategies

## Structural Context

The sleepiness function $\phi_{mt}(\mathbf{s}_{mt})$ governs the fraction of depositors in market $m$ at time $t$ who do **not** re-optimise their deposit allocation, i.e., who remain "asleep" and mechanically roll over their balance. The structural equation (Eq-11/12 of Egan et al., 2025) posits:

$$
D_{jmkt} = \phi_{mt}(\mathbf{s}_{mt}) \cdot \widetilde{D}_{jmkt-1} + (1 - \phi_{mt}(\mathbf{s}_{mt})) \cdot \hat{D}_{jmkt}
$$

where $\widetilde{D}_{jmkt-1} = (1 + r^f_t - p_{jkt}) \cdot D_{jmkt-1}$ is the **no-rebalancing counterfactual** deposit (lagged deposits grown at the net return), and $\hat{D}_{jmkt}$ is the BLP-implied demand.

Re-arranging and substituting $\hat{D}_{jmkt}$ out via fixed effects:

$$
D_{jmkt} = \phi(\mathbf{s}_{mt}) \cdot \widetilde{D}_{jmkt-1} + \alpha_{jmk} + \varepsilon_{jmkt}
$$

All estimation scripts estimate this equation via **within-group (entity-demeaned) OLS** on:

| Variable | Role |
|---|---|
| `deposit_balance` | Dependent variable $D_{jmkt}$ |
| `nr_lagged_dep` $\equiv \widetilde{D}_{jmkt-1}$ | Core regressor (no-rebalancing counterfactual) |
| State-space interactions $s_v \cdot \widetilde{D}_{jmkt-1}$ | Heterogeneity in $\phi$ |
| `v_hat`, `v_hat_2`, `v_hat_3` | Control Function (CF) residuals (when IVs used) |

The coefficient on `nr_lagged_dep` (the "constant" state variable) is the intercept of $\phi$; the coefficients on the interaction terms trace how $\phi$ varies with observables.

---

## Common Econometric Infrastructure

All five scripts share the following building blocks.

### 1. Data Construction

1. Load `market_panel.csv`; reshape from wide to long if deposit types are stored as columns (`dep_a1`, `dep_a2`, ...).
2. **Drop deposit type 3** (demand deposits / current accounts — not subject to sleepiness friction).
3. Construct entity ID: `CodConglomeradoPrudencial × deposit_type × mca_code`.
4. Generate lagged variables: `spread_qoq_lag`, `risk_free_qoq_lag`, `lagged_deposits`.
5. Build $\widetilde{D}_{jmkt-1}$: `nr_lagged_dep` $= (1 + r^f_{t-1} - p_{jk,t-1}) \times D_{jmk,t-1}$.
6. Scale numerical magnitudes (deposits ÷ $10^9$; GDP per capita ÷ $10^4$; etc.) for numerical stability.

### 2. First Stage — Control Function Approach (CFA)

Endogeneity of the spread $p_{jkt}$ is addressed via a **Petrin–Train (2010)** control function:

1. **Restrict** to endogenous deposit types ($k \in \{4, 5\}$ — CDBs and Letters of Credit).
2. **Regress** $p_{jkt}$ on instruments $\mathbf{z}$ and exogenous state variables:
   $$
   p_{jkt} = \mathbf{z}_{jt}' \boldsymbol{\pi} + \mathbf{x}_{mt}' \boldsymbol{\gamma} + v_{jkt}
   $$
3. **Extract** residuals $\hat{v}_{jkt}$ and their powers ($\hat{v}^2, \hat{v}^3$).
4. These enter the second stage as **flexible CF corrections** — a polynomial in the first-stage residual.

### 3. Instrumental Variable Sets

Four IV strategies are defined across all scripts (identical IV blocks):

| IV Label | Instruments |
|---|---|
| **OLS** | None — no CF correction |
| **IV_CostShifters** | `personnel_cost_ratio_lag`, `admin_cost_ratio_lag`, `tax_cost_ratio_lag` |
| **IV_Wholesale** | CostShifters + `lci_lca_ratio_lag`, `wholesale_ratio_lag`, `indice_basileia_lag` |
| **IV_HausmanFull** | Wholesale + `leave_one_out_mean_spread` (Hausman-type instrument) |

### 4. Imbens & Kolesar (2016) Cluster Correction

All scripts apply a **Satterthwaite-type effective cluster count**, denoted $G^*$:

$$
G^{*} = \frac{G}{1 + \text{CV}^2(N_g)}
$$

where $G$ is the nominal cluster count and $\text{CV}(N_g)$ is the coefficient of variation of cluster sizes. Highly imbalanced clusters (e.g., dominated by large banking conglomerates) inflate nominal $G$; the correction ensures conservative inference by using $t(G^*)$ critical values.

### 5. Phi Aggregation

After estimation, $\hat{\phi}_{mt}$ is constructed for every observation by evaluating the estimated linear combination $\hat{\boldsymbol{\beta}}' \mathbf{s}_{mt}$. Two levels of aggregation follow:

- **Local** $\hat{\phi}_{mt}$: Mean across entities within market $m$ and quarter $t$.
- **National** $\hat{\phi}_t$: Deposit-weighted average across all markets:
  $$\hat{\phi}_t = \frac{\sum_m \hat{\phi}_{mt} \cdot M_{mt}}{\sum_m M_{mt}}$$
  where $M_{mt}$ is the market deposit stock.

---

## Specification 1 — Baseline (B-Type Only, Linear, Post-2020 Break)

**File**: [estimation_1_sleep.py](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20(2024%20-%202025)/Open%20Finance/Open-Finance/Code/Egan_et_al_2025_Rep/estimation_1_sleep.py)

### Purpose

Baseline sleepiness estimation using **B-type (branch-based) institutions only**. This is the primary specification for the Egan et al. (2025) replication.

### Sample Restriction

- **B-type only**: Filtered by `CODMUN_IBGE != '0'` (entities with a geographic market code).
- Deposit types: $k \in \{1, 2, 4, 5\}$ (type 3 excluded).

### State Variable Blocks

Three nested blocks, all including `post_2020` as a structural break dummy:

| Block | States $\mathbf{s}_{mt}$ |
|---|---|
| **Base** | `constant`, `post_2020` |
| **Macro** | Base + `gdp_per_capita`, `cadunico_families_per1000`, `fraction_65plus`, `fraction_young`, `risk_free_qoq_lag` |
| **Tech** | Macro + `pix_users_pf_per1000`, `connections_per100`, `branches_per1000` |

### Second Stage

Standard **within-group OLS** (entity fixed effects via demeaning). The regression is:

$$
\tilde{D}_{jmkt} = \sum_{v \in \mathbf{s}} \beta_v \cdot (s_v \times \widetilde{D}_{jmkt-1})^{\sim} + \text{CF terms}^{\sim} + \tilde{\varepsilon}_{jmkt}
$$

where $\tilde{\cdot}$ denotes entity-demeaned values.

### Specification Grid

The script runs $4 \text{ IV sets} \times 3 \text{ state blocks} = 12$ specifications in parallel via `ProcessPoolExecutor`.

### Outputs

- TeX tables for each specification (second stage + first stage).
- `cluster_diagnostics.json`: $G$, $G^*$, CV, top-5 clusters.
- `estimation_results.pkl`: Serialised `statsmodels` results.
- `market_panel_phis.csv` and `national_phi_t.csv`: Phi estimates.

### Key Design Choices

- `post_2020` enters the state space, allowing the **level** of $\phi$ to shift after 2020 (PIX launch, COVID effects).
- Missing state variables imputed with column medians.
- Parallel execution across specifications for throughput.

---

## Specification 3 — No Structural Break (B-Type Local + D-Type National)

**File**: [estimation_3_sleep.py](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20(2024%20-%202025)/Open%20Finance/Open-Finance/Code/Egan_et_al_2025_Rep/estimation_3_sleep.py)

### Purpose

Robustness check that **omits `post_2020`** from the state space. Tests whether the post-2020 structural break is driving identified sleepiness patterns or is absorbed by other time-varying covariates.

### Key Differences from Spec 1

1. **No `post_2020`**: The base block is just `['constant']`; a secondary base block `Base_Selic` adds `risk_free_qoq_lag`.
2. **Dual estimation**: Runs **local** (B-type) and **national** (D-type) separately.
3. **Three national model options** for D-type firms:

| National Option | Approach |
|---|---|
| **Option 1** | Standard within-group OLS (identical to local, but using D-type sample) |
| **Option 2** | Adds `log_total_assets_lag` cross-interactions: each state variable $s_v$ gets interacted with both $\widetilde{D}$ and $s_v \times \text{log\_assets} \times \widetilde{D}$ |
| **Option 3** | Reduces multi-dimensional state vector to a **single PCA index** via `sklearn.decomposition.PCA(n_components=1)`. The regression then uses only `nr_lagged_dep` and `interaction_pca_index` |

### State Variable Blocks

| Block | States $\mathbf{s}_{mt}$ |
|---|---|
| **Base** | `constant` |
| **Base_Selic** | `constant`, `risk_free_qoq_lag` |
| **Macro** | Base_Selic + `gdp_per_capita`, `cadunico_families_per1000`, `fraction_65plus`, `fraction_young` |
| **Tech** | Macro + `pix_users_pf_per1000`, `connections_per100`, `branches_per1000` |

### National Data Construction

- D-type firms identified by `Source == 'IFDATA'` (national-level supervisory data).
- Deduplicated at `CodConglomeradoPrudencial × year × quarter`.
- Entity ID: `CodConglomeradoPrudencial × deposit_type` (no market code since D-type is national).

### Plotting

Generates comparison plots overlaying local $\hat{\phi}^{\text{Loc}}$ and three national variants ($\hat{\phi}^{\text{Nat},1}$, $\hat{\phi}^{\text{Nat},2}$, $\hat{\phi}^{\text{Nat},3}$) with 95% CI bands.

### Why This Matters

- Removing the break dummy tests identification **purely through time-varying macro/tech covariates**.
- PCA index (Option 3) tests whether the identified variation is dominated by a single latent factor — if so, the multi-covariate specification may be over-parameterised.
- Asset-size interactions (Option 2) test whether sleepiness differs systematically between large and small D-type institutions.

---

## Specification 4 — Pooled B + D (Linear, PIX Dummy)

**File**: [estimation_4_sleep.py](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20(2024%20-%202025)/Open%20Finance/Open-Finance/Code/Egan_et_al_2025_Rep/estimation_4_sleep.py)

### Purpose

Robustness check that **pools B-type and D-type firms** in a single regression. Uses `dummy_D_type` and `pix_exists` as structural controls instead of the `post_2020` break.

### Key Differences from Spec 1

1. **Pooled sample**: Both B-type (`CODMUN_IBGE != '0'`) and D-type (`CODMUN_IBGE == '0'`) firms.
2. **D-type dummy** (`dummy_D_type`): Allows a level shift in $\phi$ for digital-only/national institutions.
3. **PIX existence dummy** (`pix_exists`): Equals 1 for $t \geq$ 2020Q4, capturing the PIX instant payment system launch. More precisely timed than `post_2020`.
4. **D-type interaction terms**: `dummy_D_type × risk_free_qoq_lag`, `dummy_D_type × fraction_65plus`, `dummy_D_type × fraction_young` — allows differential sensitivity of B vs. D firms to monetary policy and demographics.
5. **Population-weighted state imputation for D-firms**: D-type firms lack municipality-level covariates. For each quarter, their state variables are set to the **population-weighted national average** of B-type firms' covariates.

### State Variable Blocks

| Block | States $\mathbf{s}_{mt}$ |
|---|---|
| **Base** | `constant`, `dummy_D_type`, `pix_exists` |
| **Base_Selic** | Base + `risk_free_qoq_lag`, `dummy_D_type × risk_free_qoq_lag` |
| **Macro** | Base_Selic + `gdp_per_capita`, `cadunico_families_per1000`, `fraction_65plus`, `fraction_young`, `dummy_D_type × fraction_65plus`, `dummy_D_type × fraction_young` |
| **Tech** | Macro + `pix_users_pf_per1000`, `connections_per100`, `branches_per1000` |

### Second Stage

Identical within-group OLS as Spec 1, but on the pooled sample.

### Plotting

Produces time-series plots of $\hat{\phi}_t$ separately for B-type and D-type sub-populations. Weighted-variance confidence bands are computed using effective sample sizes ($n_{\text{eff}} = V_1^2 / V_2$ where $V_k = \sum w_i^k$).

### Why This Matters

- Tests whether the B-type and D-type sleepiness functions are **structurally similar** once controlled for size and demographic differences.
- The PIX dummy is a more precise event-study indicator than `post_2020`.
- Pooling increases statistical power, especially for identifying differential effects through interaction terms.

---

## Specification 5 — Pooled B + D (Logistic Link, NLLS)

**File**: [estimation_5_sleep.py](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20(2024%20-%202025)/Open%20Finance/Open-Finance/Code/Egan_et_al_2025_Rep/estimation_5_sleep.py)

### Purpose

Identical sample and state variables to Spec 4, but replaces the **linear second stage** with a **logistic link function** to enforce $\phi \in [0, 1]$.

### Structural Modification

Instead of:
$$
D_{jmkt} = \left(\sum_v \beta_v \cdot s_v \right) \widetilde{D}_{jmkt-1} + \alpha_{jmk} + \varepsilon_{jmkt}
$$

this specification uses:
$$
D_{jmkt} = \Lambda\!\left(\sum_v \beta_v \cdot s_v \right) \widetilde{D}_{jmkt-1} + \alpha_{jmk} + \varepsilon_{jmkt}
$$

where $\Lambda(x) = \frac{e^x}{1+e^x}$ is the **logistic sigmoid**. This guarantees $0 \leq \hat{\phi}_{mt} \leq 1$ regardless of the estimated parameters.

### Estimation Method

**Non-Linear Least Squares (NLLS)** via `scipy.optimize.least_squares`:

1. The objective function `nlls_objective` computes entity-demeaned residuals:
   - For each observation, compute $\hat{Y} = \Lambda(\mathbf{x}_i' \boldsymbol{\theta}) \cdot Z_i + \mathbf{CF}_i' \boldsymbol{\gamma}$.
   - Entity-demean $\hat{Y}$ via `np.bincount(entity_idx, weights=Y_hat)`.
   - Return $\tilde{y} - \widetilde{\hat{Y}}$.
2. **Solver**: Trust Region Reflective (`method='trf'`) with **Cauchy loss** (`loss='cauchy'`) for outlier robustness.
3. **Max function evaluations**: 150 (deliberately capped to avoid overfitting in high-dimensional specifications).

### Inference: Average Marginal Effects (AME)

Since the logistic link is nonlinear, raw coefficients $\hat{\boldsymbol{\theta}}$ have no direct marginal interpretation. The script computes **Average Marginal Effects** via the Delta Method:

- For **continuous** variables: $\text{AME}_k = \frac{1}{N}\sum_{i=1}^N \Lambda'(\mathbf{x}_i' \hat{\boldsymbol{\theta}}) \cdot \hat{\theta}_k$
- For **dummy** variables: $\text{AME}_k = \frac{1}{N}\sum_{i=1}^N \left[\Lambda(\mathbf{x}_i^{(k=1)} \hat{\boldsymbol{\theta}}) - \Lambda(\mathbf{x}_i^{(k=0)} \hat{\boldsymbol{\theta}})\right]$

Standard errors of AMEs are obtained via **numerical Jacobian** of the AME function w.r.t. $\hat{\boldsymbol{\theta}}$, then $\text{Var}(\text{AME}) = J \cdot \hat{\Sigma} \cdot J'$.

### Phi Construction

Uses the **native (un-transformed) parameters** `params_native` to evaluate the sigmoid:
$$
\hat{\phi}_{mt} = \frac{1}{1 + \exp\left(-\hat{\boldsymbol{\theta}}' \mathbf{s}_{mt}\right)}
$$

This ensures $\hat{\phi} \in [0,1]$ by construction.

### `NonLinearResults` Container

A custom results class stores:
- `params`: AME values (for comparability with linear specs).
- `params_native`: Raw logistic coefficients $\hat{\boldsymbol{\theta}}$.
- `bse`, `tvalues`, `pvalues`: Delta-method SEs and IK2016-corrected inference.
- `G_star`: Effective cluster count.

### Why This Matters

- The linear specification can produce $\hat{\phi} < 0$ or $\hat{\phi} > 1$ — economically nonsensical. The logistic link eliminates this.
- Cauchy loss provides **robustness to outlier markets** (e.g., very small or very large banks dominating a municipality).
- AME reporting allows direct comparison of effect magnitudes with the linear specifications.

---

## Specification 6 — Pooled B + D + Institutional Heterogeneity (Linear + Logistic)

**File**: [estimation_6_sleep.py](file:///c:/Users/pedro/OneDrive/Documentos/Yale/Year%203%20(2024%20-%202025)/Open%20Finance/Open-Finance/Code/Egan_et_al_2025_Rep/estimation_6_sleep.py)

### Purpose

Extends the pooled estimation by introducing **institutional type heterogeneity**: cooperative banks (`is_coop`) and state-owned banks (`is_state_owned`). Runs **both** linear (from Spec 4) and logistic (from Spec 5) second stages for each specification.

### Key Additions

1. **Ownership dummies**: `is_coop`, `is_state_owned` enter the state space directly.
2. **Two alternative interaction structures** (CLI-selectable via `--alt`):

#### Alternative 1: Full Interactions
Every state variable $s_v$ (except `constant`) gets interacted with both ownership dummies:
$$
\phi_{mt} = \beta_0 + \beta_{\text{coop}} \cdot \mathbb{1}_{\text{coop}} + \beta_{\text{state}} \cdot \mathbb{1}_{\text{state}} + \sum_{v} \left[\beta_v + \beta_{v,\text{coop}} \cdot \mathbb{1}_{\text{coop}} + \beta_{v,\text{state}} \cdot \mathbb{1}_{\text{state}}\right] s_v
$$

This is a **fully saturated** model w.r.t. institutional type × state variable interactions. For the Tech block, this can produce $\sim30+$ parameters.

#### Alternative 2: Targeted Interactions (Default)
Only demographics and monetary policy interact with ownership:
- `is_coop × risk_free_qoq_lag`, `is_state_owned × risk_free_qoq_lag`
- `is_coop × fraction_65plus`, `is_state_owned × fraction_65plus`
- `is_coop × fraction_young`, `is_state_owned × fraction_young`

This is **more parsimonious** and economically motivated: cooperative banks serve older/rural demographics; state-owned banks may have different interest-rate pass-through.

### Dual Second Stage

Each specification is run with **both** the linear within-group OLS (identical to Spec 4) and the logistic NLLS (identical to Spec 5). This produces paired estimates for comparison:

| Model Type | Second Stage | $\phi$ Bounds | Inference |
|---|---|---|---|
| `linear` | Within-group OLS | Unbounded | Cluster-robust SEs + IK2016 |
| `logistic` | NLLS with sigmoid | $[0, 1]$ | AME + Delta-method SEs + IK2016 |

### Parallelisation

Uses `joblib.Parallel(n_jobs=4)` (rather than `ProcessPoolExecutor` used elsewhere) to manage memory under the expanded parameter space.

### Phi Construction

Identical to Specs 4 and 5 respectively, depending on `model_type`:
- **Linear**: $\hat{\phi}_{mt} = \hat{\boldsymbol{\beta}}' \mathbf{s}_{mt}$
- **Logistic**: $\hat{\phi}_{mt} = \Lambda(\hat{\boldsymbol{\theta}}' \mathbf{s}_{mt})$

### Why This Matters

- Brazil's banking system has substantial heterogeneity: cooperatives (e.g., Sicoob, Sicredi) serve primarily rural markets with older depositors; state-owned banks (BB, CEF) have implicit government guarantees.
- If cooperatives exhibit **higher** sleepiness (less withdrawal risk), this has direct implications for deposit pricing and market power.
- Running both linear and logistic provides a **specification sensitivity check** on the functional form assumption.

---

## Summary Comparison

| Feature | Spec 1 | Spec 3 | Spec 4 | Spec 5 | Spec 6 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Sample** | B-type only | B (local) + D (national) | Pooled B+D | Pooled B+D | Pooled B+D |
| **Post-2020 break** | ✓ (`post_2020`) | ✗ | ✗ (`pix_exists` instead) | ✗ (`pix_exists` instead) | ✗ (`pix_exists` instead) |
| **D-type dummy** | ✗ | N/A (separate) | ✓ | ✓ | ✓ |
| **Institutional heterogeneity** | ✗ | ✗ | ✗ | ✗ | ✓ (`is_coop`, `is_state_owned`) |
| **Second stage link** | Linear | Linear | Linear | **Logistic** | Both |
| **National model options** | — | 3 variants | — | — | — |
| **PCA dimension reduction** | ✗ | ✓ (Option 3) | ✗ | ✗ | ✗ |
| **Asset-size cross-terms** | ✗ | ✓ (Option 2) | ✗ | ✗ | ✗ |
| **AME reporting** | ✗ | ✗ | ✗ | ✓ | ✓ (logistic) |
| **Parallelisation** | `ProcessPoolExecutor` | `ProcessPoolExecutor` | `ProcessPoolExecutor` | `joblib` | `joblib` |
| **Output directory** | `rout_1/` | `rout_3/` | `rout_4/` | `rout_5/` | `rout_6/` |
| **State blocks** | Base/Macro/Tech | Base/Base_Selic/Macro/Tech | Base/Base_Selic/Macro/Tech | Base/Base_Selic/Macro/Tech | Base/Base_Selic/Macro/Tech |

> [!NOTE]
> There is no `estimation_2_sleep.py` in this repository. Specification numbering skips from 1 to 3.

---

## References

- Egan, M., Hortaçsu, A., & Matvos, G. (2025). *Deposit Competition and Financial Fragility*. Working Paper.
- Petrin, A. & Train, K. (2010). A Control Function Approach to Endogeneity in Consumer Choice Models. *Journal of Marketing Research*, 47(1), 3–13.
- Imbens, G. W. & Kolesar, M. (2016). Robust Standard Errors in Small Samples: Some Practical Advice. *Review of Economics and Statistics*, 98(4), 701–712.
