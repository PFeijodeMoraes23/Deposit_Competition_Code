# Handoff: variance decomposition of B-firm deposits (2026-10-19)

**To:** coding agent. **From:** main thread, for Pedro. No paper text.

## Task

Order-free (Shapley) decomposition of the variance of log Dep^B_{jkmt} (B firms, conglomerate j × deposit type k × MCA m × quarter t) into:
- bank (conglomerate) effects,
- market (MCA) effects,
- product (deposit type) effects,
- time (quarter) effects,
- rest.

Same method and weights as `tab_motiv_anova_levels.tex` (balance weights). State the weights. Also report a second version with bank × product and market × product terms added, so we see whether bank effects differ by type.

Note for Pedro's request: Oaxaca–Blinder splits a mean gap between two groups into characteristics and coefficients. A split by bank, market, product and time is a variance decomposition, so this uses the Shapley split already in the motivation table.

## Outputs

- Keys `desc.dep_anova.B.{bank,market,product,time,rest}` and `desc.dep_anova.B.int.{bank,market,product,time,bank_product,market_product,rest}`, via `make_paper_numbers.py`.
- `fig_slide_dep_anova_slide.pdf`: stacked bars styled like `fig_slide_xi_anova_slide.pdf`, 8.0 × 4.6 cm, `_slide` fonts.
- `tab_dep_anova.tex`, label `tab:dep_anova`, three decimals.

## Acceptance

Shares sum to one. Keys resolve with no red `??`. Report sample size, file times, and anything not feasible.
