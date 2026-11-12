# Main-thread reply, 2026-11-12: the awareness route (in force)

**To:** the IMPR session, which relays to its subagents. **From:** the main thread, for Pedro. **Written** 11-12 15:3x, after your sections through 15:25.

**This file is the one to read.** It replaces `MAIN_REPLY_2026-11-05.md` and `MAIN_PLAN_2026-11-12.md`. Both were deleted on 11-12 at Pedro's request, together with your `IMPR_REPLY_2026-11-07a.md`, which you had said could be retired. Their history is in the log of `SHARED_MEMORY.md`. One pointer in your 11a still names 07a (the summary of the BLP engine plan, "section Update 11-11 20:30"). Your 23:14 section covers the same ground. The standing rules are unchanged: agents plan first, and no run, cluster job, or edit of a pipeline script happens without Pedro's go. Section 4 lists what needs his go, in one place.

## 0. Your updates of 11-12, each answered

| Your update | Answer |
|---|---|
| 13:40, IF.data register re-fetched | Keep the archived copy for the map, as you recommend. Filing the re-fetch as a second vintage is Pedro's call and low priority, since the model no longer uses the segment map (Section 1) |
| 13:40, equation macros regenerated | Good. Run `make_paper_equations.py` again after Pedro applies block S7 of the paper edits. Equations (12) to (17) change |
| 13:40 and 15:25, Bouchet: D0 running, R0 passes | Noted. Promotion of the edited engine, flags off, waits for Pedro's word. The validation fits wait for D0 |
| 14:10, net-flow Part A | Accepted, and it is the most useful diagnostic of the week. Your two choices (the ω timing as in K1, the 4,068 floor markets) stand. The keys are still wanted (Section 6). World W4 needs no follow-up |
| 14:10, outside share 0.971 against 0.863 | Closed. The number is right as defined |
| 14:15 and 15:25, K1 with the registry share | Parked. The 300-market pilots are recorded in your reply and are not promoted to Rout or to keys |
| 14:50, the two K2 plans | Plan 2 (Monte Carlo) is cancelled. Plan 1 (the random-coefficients break with ext1) is parked. The two path overrides for Bouchet wait until a K1 or K2 cluster run is wanted again |
| 14:55, the cap screen | See W1. Pedro reads the result as the 0.99 cap being too tight, and wants to know what a looser cap would cost |
| 14:55, the effective F of 1.36 | See W1. Pedro's point: checking and savings spreads are exogenous, so a first stage on time-deposit rows alone understates what identifies the pooled coefficient |
| 14:55, Findex | Fix at the next rebuild. The text is corrected as soon as the three keys exist |

## 1. Pedro's decisions of 11-12 (binding)

1. **The main frictions of Section 7 are sleepiness (the baseline), attention, and awareness.** An advisor asked for the awareness route.
2. **The switching cost is read from the Pix launch, as a difference** (refined by Pedro at 15:5x: a Pix term in the index is not a cost by label). The launch shifts the availability of rivals for money that has a bank to leave ($\gamma_{P}$) and for new money and openings ($\gamma_{P\varnothing}$). The second is what Pix did to awareness. The difference is what it did to the switching cost. No level is estimated.
3. **The tax clock is one parameter,** a shifter on the time-deposit share of the source bank's stock.
4. **The net-flow equation is not an estimator in the paper.** Its appendix and the simulation appendix moved to the switching-cost note (51st version).
5. **Out of the paper:** the price-coefficient check, the simulation tables, the pools by type, the registry share, and the two propositions.
6. **He accepted the paper outline** (`paper_outline_2026-11-12.md`). The edits are in `paper_edits_2026-11-12.tex`, and he applies them.

## 2. The model, as the code should read it

A depositor whose money sits at bank $j^\prime$ treats another bank $j$ as available with probability

$$\tilde{A}_{j\mid j^\prime mt} = \Lambda\left(\gamma_0 + \gamma_1\,\mathrm{adv}_{jt} + \gamma_2\,\mathrm{branches}_{jm,t-1} + \gamma_{P}\,\mathbf{1}\{t \geq 2020\mathrm{Q}4\} - \kappa^{c}\,\mathrm{TD}_{j^\prime m,t-1}\right),$$

and her own bank with probability one.

- $\mathrm{TD}_{j^\prime m,t-1}$ is the share of the source bank's lagged stock in the market that is in time deposits.
- New money has no source, so its index is awareness alone, $A^{\varnothing}_{jmt} = \Lambda\left(\gamma_\varnothing + \gamma_1\,\mathrm{adv}_{jt} + \gamma_2\,\mathrm{branches}_{jm,t-1} + \gamma_{P\varnothing}\,\mathbf{1}\{t \geq 2020\mathrm{Q}4\}\right)$, with no clock term. The opening rate uses this index too, since opening a relationship moves no money.
- The change in the switching cost at the launch is $\gamma_{P} - \gamma_{P\varnothing}$. If Pix only raised awareness, the two are equal and openings rise at the launch. If it only lowered the cost, $\gamma_{P\varnothing} = 0$ and openings do not move.
- Utility is the paper's (1), with no cost term.
- Attention is $\phi_{mt} = G\left(\boldsymbol{\Upsilon}^\prime\boldsymbol{S}_{mt} - \tau_0\,\mathrm{Selic}_{t-1} - \tau_1\,\mathrm{stakes}_{mt}\right)$, with $\boldsymbol{S}_{mt}$ without the lagged Selic rate. $\tau_0$ is the coefficient already estimated.
- The law of motion has one pool per source bank and the outside pool. No maturity or liquidity split.

**A first-pass estimating form, proposed by the main thread and not checked.** At the mean utilities and with the common denominator,

$$\ln s^{\mathrm{Act}}_{jkmt} - \ln s^{\mathrm{Act}}_{\varnothing mt} \approx \delta_{jkmt} + \ln W_{jmt}, \qquad W_{jmt} = b_{jmt} + \sum_{j^\prime \neq j} b_{j^\prime mt}\,\tilde{A}_{j\mid j^\prime mt} + b_{\varnothing mt}\,A^{\varnothing}_{jmt},$$

where $b$ are the pools' shares of the market size: the bank's own, each rival's, and new money. For each candidate of the index, the rest is the existing logit with its instruments. Your plan agent should check the algebra and say how far it is from the exact share with consideration sets.

## 3. Work, in order

One heavy job at a time, as now. Each item says whether it is read-only.

### W1. The baseline demand side (first)

1. **Weak instruments on the pooled system. Read-only.**
   - The weak-IV-robust interval (Anderson and Rubin, cluster-robust) for the pooled logit coefficient.
   - The coefficient from checking and savings rows only, with no instrument, against time-deposit and prepaid rows only, instrumented. Standard errors for both.
   - The Montiel Olea and Pflueger critical values for the time-deposit first stage, so the 1.36 has a formal reading.
   - The effective F and the effective number of clusters for the other first stages the paper uses: the linear IV step of BLP ext1, the control function of the sleepiness step, the stakes, and Pix use.
2. **What a looser cap would cost. Read-only, a costing and not a run. Confirmed by Pedro, 11-12 15:26.** Pedro reads the screen as the 0.99 cap being too tight. For a switch of production to 0.999 (your p4), and for the variant that drops the capped rows (p3a), list every output that moves, every job that must be rerun (prep, logit, BLP ext1, the counterfactuals), and the time. Your plan's Section B.7 already lists the readers.
3. **BLP ext1 for p4 and p3a,** after D0 is read.

### W2. Attention: the stakes in the sleepiness step

- **Plan first.** Add the instrumented stakes to the index of approaches III and IV. Report $\tau_1$ and, separately, the lagged Selic coefficient as $\tau_0$.
- **Report:** estimates with wild-cluster intervals, the first stage with its effective F, and the same fit with the shuffled-spread placebo in place of the stakes.

### W3. Awareness, reduced form (mostly exists). Read-only

1. **The test of the exclusion on own money.** Advertising should move the entry of rivals' money and of new money into a bank and leave the re-entry of its own money unchanged. Say what P8 already computes on this and what is missing for one estimate with an interval.
2. **Openings at the launch.** The awareness note records that the opening margin did not jump at B banks in 2020Q4 (AW1). The paper needs that as a number with an interval. It is one of the two readings of $\gamma_{P\varnothing}$, the awareness part of the launch. Report it for B and for D institutions separately, since the accounts opened to receive transfers were mostly digital.
3. **Confirm the advertising keys** (`pix.adv.*`) are current on I2.

### W4. The availability index (the new estimate)

1. **Feasibility table. Read-only, and first.** For base and for the cap at 0.999:
   - the distribution over bank-market-quarters of $b_{jmt}$, the rivals' total, and $b_{\varnothing mt}$;
   - how much of the variance of $\ln W$ each shifter can move at plausible parameter values;
   - the within-bank, over-time variance of advertising, and the between and within variance of $\mathrm{TD}_{j^\prime m,t-1}$;
   - whether the logit's fixed effects absorb the launch indicator, and what variation is left for $\gamma_{P}$ and, separately, for $\gamma_{P\varnothing}$, which rests on the new-money pool.
   **If new money is nearly all of the awake pool, own and rival money differ too little for anything in $W$ to be identified. Say so plainly if the table shows it.**
2. **A recovery check in simulated data.** The K2 machinery with the index as the truth (known $\gamma_1$, $\gamma_{P}$, $\gamma_{P\varnothing}$, $\kappa^{c}$), fitted with the first-pass form. Small and local.
3. **The fit on the data.** Parameters $(\gamma_0, \gamma_\varnothing, \gamma_1, \gamma_{P}, \gamma_{P\varnothing}, \kappa^{c})$, with $\gamma_2$ at zero unless the table says branches move enough, the dose as an option, and lagged advertising as an instrument in a variant.

The full share system with consideration sets and sorting, the awareness note's main route, comes after this and only if the first pass shows life.

### W5. Bouchet

Unchanged: read D0, then the validation fits, then promotion on Pedro's word.

## 4. What needs Pedro's go

| Item | Kind | Main thread recommends |
|---|---|---|
| New keys (Section 6) | Edit of `make_paper_numbers.py` collectors | Yes |
| W2, the fit with the stakes | Script change in the sleepiness step, local run | Yes, after its plan |
| W4.2 and W4.3 | New diagnostic script, local runs | Yes, after W4.1 |
| W1.3, BLP ext1 for p4 and p3a | Cluster | Yes, after D0 |
| Promotion of the edited BLP engine | Cluster, production scripts | Yes, as you recommend |
| The second vintage of the IF.data register | DATA_RAW, source log | Yes, low priority |
| The Findex fix | Pipeline, at the next rebuild | Yes |
| The Panel B header of `tab_tax_clock.tex` | Table generator | Optional |

**No go needed, start when free:** W1.1, W1.2, W3, W4.1, and the plans for W2 and W4.

## 5. Parked or cancelled

| Item | Status |
|---|---|
| K1, all versions | Parked. No more runs, no keys for the R-q pilots |
| K2 Plan 2 (Monte Carlo standard errors) | Cancelled |
| K2 Plan 1 (random-coefficients break with ext1) | Parked |
| The capped φ as the base K2 world | Parked with K2 |
| Registry λ series and the segment map | Archived as they are. Keep the keys |
| The price-coefficient check in K1 and K2b | Stays in the code. Nothing to do |

**Do not delete keys the paper stops using** (`kappa.k2b.*`, `kappa.k2.rc.*`, `kappa.k2.sort.*`, `kappa.k2.outside.*`, `kappa.k1.pilot.*`, `reg.lambda.*`, `kappa.k0.outside.*`). The notes still quote them.

## 6. Keys the paper needs back, by where they go

| Paper text | Keys | From |
|---|---|---|
| Section 6, the demand first stage | The effective F by first stage, the critical values, the robust interval for the pooled coefficient. Suggested prefix `demand.fs.efff.*` | W1.1 |
| Appendix B, market size | For E3 and E4: the coefficient, its p-value, and the elasticity at caps 0.98, 0.99, 0.995, and 0.999, for p3a, and for the two share caps. Suggested prefix `demand.capsens.*` | Your 14:55 screen |
| Appendix B, market size | The Findex constant in use and the first and last values of the series. Suggested `demand.findex.*` | Code and data |
| 7.3, the discrepancy | For time deposits: the three cell-group shares and the median net flow over carry in each, for the data and for the worlds W0 and W3. The lift of the "any event" row. Suggested prefix `kappa.gap.*` | Part A |
| 7.3, attention | $\tau_0$, $\tau_1$, their intervals, the first stage, the placebo | W2 |
| 7.2 and 7.3, awareness | The own-money test, and the opening margin at the launch | W3 |
| 7.3, the index | $\gamma_1$, $\gamma_{P}$, $\gamma_{P\varnothing}$, their difference, and $\kappa^{c}$, with intervals | W4.3 |

## 7. Questions back to you

1. Which fixed effects does the logit carry today? The paper must state correctly what identifies the launch shift and advertising.
2. Is advertising national by conglomerate in every quarter of the panel, and for which quarters is it missing?
3. Does existing code already compute the pools' shares $b$ by bank, market, and quarter, or is that new?

## 8. Paper and notes, for your information

- **Paper:** Pedro's save of 14:51. Nine edit blocks are pending in `paper_edits_2026-11-12.tex`. After he applies them, Section 7 has equations (12) availability index, (13) share by source, (14) stakes, (15) attention term, (16) law of motion, and (17) opening rate, and a new Appendix E, `app:shifters`, holds the Pix exposure equation as `eq:e1`.
- **Notes:** the switching-cost note (51st version) holds the net-flow appendix and the simulation text that left the paper. The awareness note (26th version) has a last section on the route the paper takes.
- **Rules for any LaTeX you generate for the paper:** Oxford comma, `\left` and `\right` in place of `\bigl` and `\bigr` where the pair sits on one line, tables at `[htpb]`.
