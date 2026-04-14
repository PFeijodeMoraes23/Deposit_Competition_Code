# Market Size Weights and Outside Option: Technical Notes

## Overview

This document addresses three questions about the weighting scheme used across the sleepiness and BLP demand estimation pipelines in the Egan et al. (2025) replication:

1. What are the appropriate weights for aggregating the sleepiness function $\hat{\phi}_{mt}$ to the national level?
2. How does the demand estimation handle market size $M_{mt}$ for D-firms and the outside option, given that "active deposit market size" is only well-defined for B-firms?
3. Why is the $\gamma$-ceiling used to compute the outside option share innocuous?

---

## 1. Sleepiness Aggregation Weights

### The Aggregation Problem

The sleepiness estimation produces a local estimate $\hat{\phi}_{mt}$ for each market $m$ at time $t$. To compute the national aggregate $\hat{\phi}_t$ (used for D-firms in the demand prep), we need a weighted average:

$$\hat{\phi}_t = \frac{\sum_m M_{mt} \cdot \hat{\phi}_{mt}}{\sum_m M_{mt}}$$

The question is: what is $M_{mt}$?

### Why Population Is Correct

Under the model assumption that local deposit market potential is a constant fraction of population:

$$M_{mt} = c \cdot \text{pop}_{mt}$$

the constant $c$ cancels from the numerator and denominator:

$$\hat{\phi}_t = \frac{\sum_m c \cdot \text{pop}_{mt} \cdot \hat{\phi}_{mt}}{\sum_m c \cdot \text{pop}_{mt}} = \frac{\sum_m \text{pop}_{mt} \cdot \hat{\phi}_{mt}}{\sum_m \text{pop}_{mt}}$$

This makes `pop_total` the correct weight. Using lagged deposits instead would introduce an endogeneity concern: markets with high deposits (partly *because* of low sleepiness) receive disproportionate weight, biasing the national aggregate downward.

### Implementation

The `calculate_phis()` and `calculate_pooled_phis()` functions in all five sleepiness scripts (`estimation_{1,3,4,5,6}_sleep.py`) now set:

```python
if 'pop_total' in df.columns:
    df['market_size'] = df['pop_total'].fillna(0)
else:
    df['market_size'] = 1.0
```

This weight enters the aggregation:

```python
market_agg = df.groupby(['year_quarter', 'CODMUN_IBGE']).agg(
    phi_mt=('phi_mt_Spec', 'mean'),
    M_mt=('market_size', 'sum')
)
national_phi_t = Σ(phi_mt * M_mt) / Σ(M_mt)
```

The demand prep script (`estimation_1_demand_1_prep.py`) already uses `pop_total` for the same aggregation:

```python
def weighted_mean(g):
    v, w = g['phi_mt'], g['pop_total']
    return v.mean() if w.sum() == 0 else np.average(v, weights=w)
```

So the two stages are now consistent.

### Scripts Modified

| Script | Function | Previous Weight | New Weight |
|--------|----------|-----------------|------------|
| `estimation_1_sleep.py` | `calculate_phis()` | `lagged_deposits` | `pop_total` |
| `estimation_3_sleep.py` | `calculate_local_phis()`, plotting | `lagged_deposits` | `pop_total` |
| `estimation_4_sleep.py` | `calculate_pooled_phis()` | `lagged_deposits` | `pop_total` |
| `estimation_5_sleep.py` | `calculate_pooled_phis()` | `lagged_deposits` | `pop_total` |
| `estimation_6_sleep.py` | `calculate_pooled_phis()` | `lagged_deposits` | `pop_total` |

---

## 2. Market Size in the Demand Estimation

### The Identification Challenge

In the BLP demand estimation, the "market size" $M_{mt}$ defines the potential number of consumers. Shares are then:

$$s_{jkmt} = \frac{q_{jkmt}}{M_{mt}}$$

where $q_{jkmt}$ is a quantity (active deposits). The market size must satisfy $\sum_j q_j < M_{mt}$ so that the outside option share $s_0 > 0$.

For **B-firms**, market size is straightforward: $M_{mt}^B = \gamma \cdot \text{pop}_{mt}$, where $\gamma$ is calibrated from the data.

For **D-firms** and the **outside option**, there is no natural market-level quantity — D-firms are national, and the outside option has no observed deposits at all.

### How We Handle It

The demand prep script (`estimation_1_demand_1_prep.py`) calibrates a single global per-capita ceiling $\gamma$:

$$\gamma = 1.1 \times \max_{m,t}\left(\frac{\sum_{j} \text{Dep}^{\text{Act}}_{jmt}}{\text{pop}_{mt}}\right)$$

This defines:

- **B-firm shares**: $s^B_{jkmt} = \text{Dep}^{\text{Act}}_{jkmt} / (\gamma \cdot \text{pop}_{mt})$
- **D-firm shares**: $s^D_{jkt} = \text{Dep}^{\text{Act}}_{jkt} / (\gamma \cdot \text{Pop}^{\text{National}}_t)$

Both use the same $\gamma$ multiplied by population. The outside option share in market $m$ is implicitly:

$$s_{0mt} = 1 - \sum_{j \in J^B_{mt}} s^B_{jmt} - \sum_{j \in J^D_t} s^{D,\text{local}}_{jmt}$$

Inside the BLP loop (`compute_model_shares`), the D-firm national share is aggregated using population weights:

$$s^{D,\text{National}}_{jkt} = \sum_m \frac{\text{pop}_{mt}}{\text{Pop}_t} \cdot \frac{\exp(V_{jkt})}{\text{Denom}_{mt}}$$

This is model-consistent: since $M_{mt} = \gamma \cdot \text{pop}_{mt}$, the weight ratio is:

$$\frac{M_{mt}}{M_t} = \frac{\gamma \cdot \text{pop}_{mt}}{\gamma \cdot \text{Pop}_t} = \frac{\text{pop}_{mt}}{\text{Pop}_t}$$

The $\gamma$ cancels. So population weights are the *only* internally consistent choice. Using active deposit totals as weights would be circular (deposits depend on shares, which depend on the weights).

### Summary of Weight Flow

```
┌──────────────────────────────┐
│   estimation_X_sleep.py      │
│   ─────────────────────────  │
│   phi_mt → phi_t via         │
│   pop_total weights          │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ estimation_X_demand_1_prep.py│
│ ──────────────────────────── │
│ phi_t → Dep_Act → shares     │
│ M_mt = γ × pop_total         │
│ share = Dep_Act / M_mt       │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ estimation_X_demand_2_loop.py│
│ ──────────────────────────── │
│ D-firm national share via    │
│ pop_total / Pop_National     │
│ Outside option: OUTSIDE_EPS=1│
└──────────────────────────────┘
```

All three stages are internally consistent under the assumption $M_{mt} = c \cdot \text{pop}_{mt}$.

---

## 3. Why the $\gamma$-Ceiling Is Innocuous

The ceiling $\gamma$ is computed as:

```python
gamma = dep_per_capita.max() * 1.1
```

where `dep_per_capita` = total active deposits in market $(m,t)$ divided by `pop_total`. Three reasons this is harmless:

### 3.1 $\gamma$ Drops Out of the Estimation

Inside the BLP loop, the contraction mapping operates on:

$$T(\delta)_j = \delta_j + \ln s^{\text{data}}_j - \ln s^{\text{model}}_j$$

Since $s^{\text{data}}_j = \text{Dep}^{\text{Act}}_j / (\gamma \cdot \text{pop}_{mt})$, changing $\gamma$ shifts *all* data shares by $1/\gamma$. In log-space, this is $\ln s^{\text{data}} = \ln \text{Dep}^{\text{Act}} - \ln \gamma - \ln \text{pop}$. The $-\ln \gamma$ term shifts all $\delta_j$ by a common constant, which is absorbed into the intercept of $\theta_1$ (the FGC dummy or the overall constant in the linear IV step). The structural parameters $\theta_2$ (random coefficients), the relative $\alpha_k$ (deposit-type spread coefficients), and all $\beta$ (product characteristic coefficients) are **invariant to $\gamma$**.

More precisely: if we replace $\gamma$ with $\gamma' = a \cdot \gamma$ for any $a > 0$, then $s^{\text{data}'}_j = s^{\text{data}}_j / a$ and $\delta'_j = \delta_j - \ln a$ for all $j$. Since $\theta_1$ is recovered from $\delta = X\theta_1 + \xi$, the shift $-\ln a$ is absorbed into whichever column of $X$ acts as an intercept. The residuals $\xi$ are unchanged, so the GMM moments and the entire $\theta_2$ estimation are unaffected.

### 3.2 The 10% Padding Guarantees $s_0 > 0$

By construction, the densest market has $\sum_j s_j \leq \gamma^{-1} \cdot \max_{m,t}(\sum_j \text{Dep}^{\text{Act}}_j / \text{pop}_{mt}) = 1/1.1 \approx 0.91$. This guarantees at least a 9% outside option share in every market. Without this padding, the densest market could have $\sum_j s_j \approx 1$, causing:
- $\ln s_0 \to -\infty$, destabilizing the contraction.
- The outside option failing to pin the level of $\delta$, reintroducing shift-invariance.

### 3.3 The Outside Option Level Is Not Separately Identified

BLP identifies utility *differences* relative to the outside good. Changing $\gamma$ changes the implied $s_0$ and shifts all $\delta_j$ uniformly, which is isomorphic to changing the intercept. Since no utility parameter is estimated for the outside option (its utility is normalized to $V_0 = 0$), the absolute level of $\gamma$ is irrelevant for every identified parameter in the model.

This is a standard result: see Berry (1994, Section 2), where the normalization $u_{i0t} = 0$ makes the outside option share a function of the data transformation ($\ln s_j - \ln s_0$), not a structural parameter. The $\gamma$-ceiling simply determines what $s_0$ *is*, which affects the level of $\delta$ but not the parameters that the econometrician cares about.

---

## References

- Berry, S. (1994). "Estimating Discrete-Choice Models of Product Differentiation." *RAND Journal of Economics*, 25(2), 242–262.
- Berry, S., Levinsohn, J. & Pakes, A. (1995). "Automobile Prices in Market Equilibrium." *Econometrica*, 63(4), 841–890.
- Conlon, C. & Gortmaker, J. (2020). "Best Practices for Differentiated Products Demand Estimation with PyBLP." *RAND Journal of Economics*, 51(4), 1108–1161.
- Egan, M., Hortaçsu, A. & Matvos, G. (2017). "Deposit Competition and Financial Fragility: Evidence from the US Banking Sector." *American Economic Review*, 107(1), 169–216.
- Nevo, A. (2001). "A Practitioner's Guide to Estimation of Random-Coefficients Logit Models of Demand." *Journal of Economics & Management Strategy*, 9(4), 513–548.
