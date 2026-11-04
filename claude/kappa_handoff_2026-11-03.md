# Handoff: the switching-cost design (Kappa agent, new session)

**To:** the Kappa agent. **From:** main thread, for Pedro. **Written:** 2026-11-03, 21:00.

**Scope.** You run the empirical side of the switching-cost design: first a simulation that checks what the design's regression measures (K2), then the regression itself on the rebuilt panel (K1). The design is written out in `Drafts/Deposit Competition/kappa_design_2026-11-02.md` (twenty-second version, 11-03, 40 pages; the PDF beside it is the rendered copy; assumption labels run A1 to A7: A1 mean utility, A2 dominant stay, A3 quality weights, A6 the separable scale, A7 the outside pool). **Since the twenty-second version there is no linear regression: the estimating equation (33) is nonlinear in the gap and estimated by NLS or PPML with a control function for the spread, and the parameters (34) are read directly. Anything below that says "slope", "ratio of slopes" or "OLS" is superseded by this update.** Read it before anything else: Sections 2 (the model), 3 (why the reduced form), 4 (the derivation, Steps 1 to 5), 6 (the tests, Groups A to C) and 7 (the order of work); Section 10 (the record appendix) has every task already run with its result, and K0 there is the data you start from. Reply in `claude/KAPPA_REPLY_2026-11-03.md`, timestamped; newest section first.

## Update 21:50: your K2 step-2 reply accepted; the design changed on it; what runs next

Your 21:33 reply and `Rout/k2_report.md` are accepted in full, and they changed the note (twenty-third version, 34 pages). Three things you flagged are now in it:

```
taken[4]{item,what the note now says}:
  A2,"restated as a common denominator across the pools of a market, with two sufficient conditions: the stay option dominates, or the outside option does. On the paper's delta and M-hat the second holds (your 87.5%, 0.11, 56%), so the operative case is the outside-dominated one; the error of A2 is the difference in log denominators, your 0.003 to 0.016"
  The exit,"under a common denominator the awake money at (j,4) re-chooses like any own pool; the exit is the whole awake movable stock with a market-quarter coefficient, exp(gamma_mt) (1 + r - rho) [omega + (1 - omega) lambda] Q4, which is your form E[NF/Dep] = exp(FE + beta g)(1 + X theta) - c_mt Q4; the stayers re-enter through c_jt Q4 on the entry side. Equation (23) of earlier versions is gone"
  The linear form,"gone since the twenty-second version; your step 2 is the record of why (A6: the scale varies across cells and the pool weights have extreme tails)"
  Quality weights,"gone: with a common denominator the source product's utility drops out of the conditional probability, so there is no theta_w; A3 is now only that the cost of leaving a source type is the same whichever bank receives the money"
```

The estimating equation is now eq. (28) of the note (label `eq:reg`): $\mathrm{NF}/\mathrm{Dep}_{j2}=e^{\lambda_{jm}+\lambda_{mt}}e^{-\alpha g}\{[1+e^{a}Q^{1}+c_{jt}Q^{4}]+e^{-\tilde\kappa}[R^{2}+e^{a}R^{1}+R^{4}]+e^{a_\varnothing}O\}-e^{\gamma_{mt}}(1+r-\rho_{j4,t-1})[\omega+(1-\omega)\lambda]Q^{4}+u$, parameters $\alpha,\tilde\kappa,a,a_\varnothing,E_c$ and the effects. One more point you raised that the note records as a fact for the paper: the gross flows the paper's world implies (0.11 stay, awake money 1.2 percent of deposits) are not what the data show (net flows 9 percent of stock at the median cell); the design is built on the observed net flow.

**Decisions (Pedro, 21:50).**

```
decided[3]{item,decision}:
  K-1,"steps 3 to 6 held, as you recommended"
  K-2,"next, in this order: (a) the estimator: nonlinear least squares on NF in the form above, fixed effects profiled out, Gauss-Newton or Levenberg-Marquardt, to settle the two unattributed non-convergences (the outside pool; phi-hat varying across market-quarters) and to replace Fisher scoring; report convergence on the fe and ideal worlds first. (b) K2b: simulate kappa falling at 2020Q4 (kappa0 = 1, kappa1 = 0.5, and a use-dose version with kappa_U on z_U of P1) and test whether the before-after shift of kappa-tilde in (28) recovers the change, in the stored, fe and ideal worlds; this is the headline design and uses within-cell time variation. (c) then steps 3 to 7 on the estimator that converges"
  K-3,"the note is revised as above; K1's outline in this file is to be read with (28) in place of (33) and without theta_w"
```

**For (a), two things to keep.** Your gate rule stands: if the estimator does not recover $\kappa$ within 0.03 on the fe world, stop and say so. And keep the 18-cluster count in view: the savings pair's inference will rest on the wild cluster bootstrap. Reply in `KAPPA_REPLY_2026-11-03b.md`.

## Rules

- **V_Friction.tex, the decks and the design note are edited by Pedro and the main thread only.** Deliver code, outputs and reports.
- **Pedro owns git, and you cannot delete files:** give `Remove-Item -LiteralPath` lines.
- **Raw files stay untouched.** Nothing in the pipeline is edited; your scripts are new files under `diagnostics/` (or a new `kappa/` folder, your choice, said once).
- **Python:** `C:/venvs/egan/Scripts/python.exe -I`, with `MPLBACKEND=Agg`. One heavy process at a time: the coding thread (`impr_handoff_2026-11-03.md`), the Pix agent (`pix_handoff_2026-11-03c.md`) and the BB agent (`bb_handoff_2026-11-03.md`) share the machine. Read `market_panel.parquet` through pyarrow, once, with the columns listed in your reply, never through `load_panel_cached` (it can rewrite its sidecar).
- **Before you edit any existing script, summarize the change and its side effects and wait for Pedro's OK.** You should not need to.
- **Never guess a schema or a file layout.** Read the file, or ask.
- **No `\pn` key** until Pedro decides what the paper says.
- **Timestamp every update.** "Wait" means stop.

## Basis

```
basis[7]{item,where}:
  The design,"Drafts/Deposit Competition/kappa_design_2026-11-02.md (+ .pdf). Equation numbers below are the note's"
  The paper,"Drafts/Deposit Competition/V_Friction.tex: utility (1), shares (4), sleepiness (6), the balance equation (14-B), the active balance (11-B), the sleepiness regression (10)"
  K0 outputs,"processed/ESTIMATION_OUTPUT/Rout/k0_*.{csv,parquet,json}, k0_report.md; script diagnostics/diag_k0_netflow_objects.py (Pix agent, 11-03). Reuse its reads and its definitions of NF, Q, R, O"
  Panel,"market_panel.parquet, md5 d01b8bae (10-27), B cells 2016Q1-2024Q4. It is being rebuilt by the coding thread (tasks I1, I6, I8, I9): market size, rate fallbacks, the old-rule savings premium and the judicial deposits. K2 runs on d01b8bae; K1 waits for the rebuilt panel and its new md5"
  Mean utilities,"the E3 spec-12 demand parquet (routine III, local prep of 10-29, 36 quarters): delta_jkmt, phi_hat_mt, M_mt, as K0 used"
  Shifters,"from the Pix agent (P1 the Pix-use dose and exposures, P2 advertising and branches, P3 the stakes), as files in Rout when they land; do not build them yourself"
  Decisions already taken (Pedro, 11-03),"left side: bank x market NF with a booking flag and p99 trimming, aggregation to bank x state as robustness; rows with zero own savings dropped in the savings pair, normalized by own checking in the checking pair; one-B-bank markets kept; rate-fallback cells dropped until I6; BB and Caixa flagged, run with and without; the zero-premium quarters (2016Q1-2017Q3, 2021Q4-2024Q4) as the headline sample, the full window with a bank x Selic-regime control beside it"
```

## The objects, in one place

- **Outcome:** the net flow into $j$'s time deposits, $\mathrm{NF}_{j4mt}=\mathrm{Dep}_{j4mt}-(1+r_{t-1}-\rho_{j4m,t-1})\mathrm{Dep}_{j4m,t-1}$ (note, eq. 21), over $j$'s lagged own savings. The lagged rate is the paper's convention (timing A); K0 (a) built it.
- **Price:** the gap $g_{jmt}=\rho_{2mt}-\rho_{j4mt}$ (eq. 24), the regulated savings spread minus the bank's own time-deposit spread; its only endogenous part is $\rho_{j4mt}$, instrumented by the own cost shifters of the paper's (10), lagged as there.
- **Pool weights** (eq. 27), lagged stocks over own lagged savings: $Q^{1}$, $Q^{4}$ (own checking, own time deposits), $R^{1}$, $R^{2}$, $R^{4}$ (rivals', B rivals only), $O$ (the outside pool from $\widehat M_{mt}$).
- **The estimating equation** (eq. 33): $\mathrm{NF}/\mathrm{Dep}_{j2,t-1}=e^{\lambda_{jm}+\lambda_{mt}}e^{-\alpha g}\{1+e^{-\tilde\kappa}R^{2}+e^{a}[Q^{1}+e^{-\tilde\kappa}R^{1}]+e^{a_\varnothing}O\}+e^{\lambda_{jm}+\lambda_{mt}}e^{-\tilde\kappa}e^{a_4}R^{4}e^{-\alpha\bar g^{4}}-e^{\lambda_{jm}+\lambda_{mt}+b}Q^{4}[\omega+(1-\omega)\lambda E_c]e^{\alpha g}+u$, with $R^{k}$ summed with the quality weights $w=\exp((x_j-x_{j'})'\theta_w)$.
- **The parameters** (eq. 34): $\alpha$ (price response), $\tilde\kappa$ (the rival factor, $=\kappa-\ln A-\ln h$), $\theta_w$, $a$ (checking scale), $a_\varnothing$, $a_4$, $b$ (exit scale), $E_c$ (clock factor), and the fixed effects. Estimation: NLS, or PPML on the positive part with the exit term as an offset; the spread inside $g$ and $\bar g^{4}$ by a control function on the cost shifters, as the paper's (10); WCB by conglomerate. Overidentification: one $\tilde\kappa$ for the savings and checking pairs; one $a$ for the own and rival checking pools.

## Task K2: the simulation test (now)

Section 7, Group B of the note. New script, say `diagnostics/diag_k2_netflow_simulation.py`. Seeds fixed and printed.

```
k2[7]{step,what,output}:
  1,"Build the simulator of equation (20) of the note on the panel's own pools (lagged stocks by bank, type, market, B cells), with the stored delta_jkmt as mean utilities, A1 (mean-utility logit), one phi_mt (the stored phi-hat), the outside pool O_mt from the stored M-hat, and parameters you set: kappa in {0, 0.5, 1, 2}, A = 1, no clock (omega = 1, lambda = 0). Shares conditional on the source pool as in equation (6); the own pool inside the sum; the balance from (20); NF from (21)","the script, parameters at the top"
  2,"Estimate equation (33) on the simulated NF with the panel's own g, pool weights and fixed effects (NLS, no control function: the simulation has no endogeneity). Report kappa-tilde against kappa for each kappa, from the savings pair and from the checking pair, and a from the own and the rival checking pools. If you already ran a linearized version, keep it as a side table; it is no longer the design","Rout/k2_recovery.csv"
  3,"Break A2: scale one bank's delta_j4 up in each market until its time deposits take 30% and 60% of the awake money in the quarter; report what the ratios do","Rout/k2_break_a1.csv"
  4,"Large gaps: multiply the gaps by 2 and 4; report that the nonlinear fit still recovers kappa, and beside it what a linearized fit would have given, for the record","Rout/k2_break_linear.csv"
  5,"Add sorting: multiply exp(delta_j4) by h in {0.9, 0.7, 0.5} for rivals' pools only; report the bias in kappa-tilde against -ln(h), which is what the note's Section 3.9 predicts","Rout/k2_sorting.csv"
  6,"Leave the outside pool out of the estimation while it is in the simulation; report the bias in kappa-tilde against the note's claim (A7) that it is pushed down","Rout/k2_outside.csv"
  7,"Simulate with the paper's random coefficients and draws (theta_2 of the stored fit, as a data-generating process only) and estimate under A1; report the bias in kappa-tilde","Rout/k2_rc.csv"
```

**Report:** `Rout/k2_report.md`, one paragraph per step and one sentence on what (33) measures when everything holds. If step 2 does not recover $e^{-\kappa}$ within the linearization error at $\kappa=0.5$, stop and say so before steps 3 to 6. Use the stored $\widehat M$ as it is (the I1 fix changes $O$, and step 6 is about the sign of the bias). Simulate at the bank × market level; the booking churn of K0 (a) is not in the model and should not be in the simulation.

## Update 22:45: P1 to P3 have landed; what K1 takes from them

The Pix agent delivered the shifters (`Rout/p1_*`, `p2_*`, `p3_*`, notes `p1_note.md`, `p2_note.md`, `p3_note.md`, script `diagnostics/diag_p_shifters.py`; keys `CodConglomeradoPrudencial` and `mca_code` as a string). Pedro's choices on their open points:

```
k1_inputs[5]{item,decision}:
  Use dose,"U_m = z_U of p1_pix_dose_exposure_mca.csv (2021 mean payers per inhabitant, population-standardized); instruments Mob and G45 jointly (z_mobile, z_g45): F 65.3 with MCA clusters, 70.5 with state clusters; Mob alone is 31.7 and 8.0, and the two are negatively correlated, so never alone"
  Timing profile,"the post-launch profile of the dose effect uses z_payers_pc_q of p1_pix_use_path_mca_quarter.csv (use standardized within each quarter), since the 2021 ordering drifts (rank correlation with 2024Q4 use 0.74)"
  Awareness shifter,"advertising, adv_brl_real_2024q4 of p2_advertising_conglomerate_quarter.csv, in logs or with a floor on deposits for the ratio; observed for 29 of the 64 small B banks in 30+ quarters (91% of small-B deposits); branches_lag of p2_branches_conglomerate_mca_quarter.csv enters as a control only (3.8% of its log variance is within a cell)"
  Stakes,"S of p3_stakes_mca_quarter.csv (per-relationship balance, median R$5,846), instrument S_tilde, with hhi4_lag and n_banks4_lag as controls (the shuffled placebo keeps F 174 of 246); P3 is rebuilt by the Pix agent when the new panel md5 lands, so read its panel_md5 column"
  Rebuild,"K1 runs on the rebuilt panel only (I1 market size, I6 fallbacks with the new rate_a4_fallback column, I8 savings yields, I9 judicial deposits with dep_a4 overwritten and dep_a4_incl_judicial kept); read the md5 in impr_handoff_2026-11-03.md"
```

## Task K1: the regression (after K2, the panel rebuild, and P1 to P3)

Not before Pedro's go. The outline, so that your K2 code is written to be reused:

```
k1[8]{block,what}:
  Sample,"B cells on the rebuilt panel; the decisions above; the zero-premium quarters as the headline, the full window beside it"
  Left side,"NF over own lagged savings, bank x market, booking flag, p99 trim; the bank x state aggregation as robustness"
  Savings pair,"equation (33) with the savings pools only (Q1 = R1 = 0): alpha, kappa-tilde, theta_w, a_null, b, E_c; the spread by a control function on the own cost shifters of (10); FE_jm, FE_mt inside the exponent; WCB by conglomerate"
  Checking pair,"the checking pools added (a estimated, and one kappa-tilde for both pairs, with the two-kappa version as the test), with the default-sweep indicator from the sweep registry on the own-checking term (Santander from 05/2022; BB's sweep is opt-in and immaterial, B1 and BB3)"
  Exit block,"Q4 [omega + (1 - omega) lambda E_c] with omega from the statements (K0 (e), applied to the stock carried into y+1) and lambda assumed 1 for D banks and read from the statements' no-maturity line for B banks until the registry (025399) lands; E_c against the legal value"
  Controls,"O from the rebuilt M-hat; R4 with the pool-weighted rivals' time-deposit gap (its spread by a control function on the rivals' cost shifters)"
  Shifters,"kappa-tilde = kappa0 + (kappa1 - kappa0) Post + kappa_U Post U_m (P1, with Mob and G45 jointly as the instrument for U); alpha + alpha^P Post + alpha^PU Post U as the own-pool tests; the stakes (P3, instrumented) in the scale and, as the test, in kappa-tilde; advertising (P2) in kappa-tilde through A for the non-big-five banks; pre-trend and placebo launches (2018Q4, 2019Q4)"
  Report,"every parameter of (34) with WCB intervals; the Group C tests of Section 6 of the note; D destinations at the national level; one table per pair"
```

Nothing on the real $\mathrm{NF}$ is regressed until Pedro has read K2 and the rebuild has landed.
