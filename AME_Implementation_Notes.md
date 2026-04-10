# Average Marginal Effects (AME) Implementation in Sleepiness Pipeline

The AME implementation in the sleepiness pipeline (specifically within `estimation_5_sleep.py`'s `get_nlls_ame_and_se` function) uses a **hybrid approach: Analytical for the effects, and Numerical for the Standard Errors (via Delta Method).**

Here is exactly what it is computing under the hood:

## 1. Computing the Marginal Effects (Analytical)
The script iterates through every covariate in your non-linear least squares estimation and checks if it's a binary dummy (only 0s and 1s) or a continuous variable. It then applies the appropriate analytical formula based on the Logit CDF:

*   **For Dummy Variables:**
    It computationally identical twins your entire dataset—in one copy, it forces the dummy variable to `1`, and in the other, it forces it to `0`. It passes both matrices through the predicted probability function and takes the mean of the exact differences. 
    *Analytical formula:* `AME = mean( P_1 - P_0 )`
*   **For Continuous Variables:**
    It computes the exact analytical derivative of the logistic function for each observation, multiplied by the estimated coefficient $\theta_k$, and takes the average.
    *Analytical formula:* `AME = mean( P * (1 - P) * theta_k )`

## 2. Computing the Adjusted Standard Errors (Numerical)
Because marginal effects are non-linear transformations of your estimated parameters $\hat{\theta}$, the Standard Errors must be appropriately scaled using the Delta Method: $Cov_{AME} \approx J \cdot Cov(\hat{\theta}) \cdot J^T$.

To get the Jacobian matrix ($J$) required for the Delta Method, the script uses a **Numerical Derivative**:
*   It loops through every estimated parameter and perturbs it by a tiny finite-difference step size (`h = 1e-5`).
*   It recalculates the analytical AMEs under the perturbed parameter (`ame_step`).
*   It populates the Jacobian simply as `(ame_step - AME) / h`. 

So, in short: the **effects themselves are exact analytical derivatives** (or discrete differences for dummies), but the **Jacobian used to adjust the Standard Errors is taken numerically**.
