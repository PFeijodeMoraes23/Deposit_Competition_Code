# Paper outline under the awareness route, 2026-11-12

**For:** Pedro, who edits the paper. **From:** the main thread. **Basis:** `V_Friction.tex` of 11-12 14:51 and your decisions of 11-12 (about 15:00).

This is the plan for Section 7 and its appendices. The earlier Section 7 plan and review were deleted on 11-12. It is an outline, not text. I write the edit blocks after you accept it, and the evidence paragraphs after IMPR reports.

## 1. The route you chose

1. **Main frictions:** sleepiness (the baseline), attention (the lagged Selic rate and the stakes), and awareness (advertising).
2. **Switching costs:** one parameter, the shift at the Pix launch. No level.
3. **Tax clock:** one parameter, a shifter on the time-deposit share of the source bank's stock. No pools, no maturity split, no registry share.
4. **The net-flow equation is no longer an estimator.** IMPR's 300-market runs with the registry share (15:25) confirm it: no fit converges, and the interval for the change at Pix is [−39.9, 34.5] in the headline form and [−93.1, 83.4] in the savings-only form.

## 2. The model change, in one page

A depositor whose money sits at bank $j^\prime$ treats another bank $j$ as available with probability

$$\tilde{A}_{j\mid j^\prime mt} = \Lambda\left(\gamma_0 + \gamma_1\,\mathrm{adv}_{jt} + \gamma_2\,\mathrm{branches}_{jm,t-1} + \gamma_{P}\,\mathbf{1}\{t \geq 2020\mathrm{Q}4\} - \kappa^{c}\,\mathrm{TD}_{j^\prime m,t-1}\right)$$

and always treats her own bank as available. $\mathrm{TD}_{j^\prime m,t-1}$ is the share of the source bank's lagged stock held in time deposits.

- **One object, two readings.** A bank is unavailable because the depositor does not know it or because moving to it costs too much. Balances cannot tell these apart, and the model no longer tries. It separates them by what moves them.
- **Awareness** moves with advertising ($\gamma_1$) and branches ($\gamma_2$).
- **The switching cost** moves with Pix ($\gamma_{P}$, and optionally a dose term in instrumented Pix use). Its level is inside $\gamma_0$.
- **The tax clock** makes money at banks heavy in time deposits harder to move ($\kappa^{c}$).
- **New money** has no source, so it pays no cost. Its index has its own constant and no Pix or clock term. In principle the gap between the two constants is the level of the switching cost. In practice the outside pool is too poorly measured to read it, so that constant is a nuisance parameter.
- **Utility is equation (1), unchanged.** The utility with a cost $\kappa$ and a clock premium (the current equation (12)) is dropped.
- **Attention** is the term you accepted last night: $\phi_{mt} = G\left(\boldsymbol{\Upsilon}^\prime\boldsymbol{S}_{mt} - \tau_0\,\mathrm{Selic}_{t-1} - \tau_1\,\mathrm{stakes}_{mt}\right)$.
- **The law of motion** loses the maturity and liquidity split ($\omega$, $\lambda$, $\ell$).

**What is estimated:** $\tau_0$, $\tau_1$, $\gamma_1$, $\gamma_{P}$, $\kappa^{c}$, and possibly $\gamma_2$ and the dose. **What is not:** the level of the switching cost, the level of awareness, and sorting.

## 3. Section 7, new outline

| Part | Content | Paragraphs (target) |
|---|---|---|
| Opening | What measured sleepiness may contain. The two questions and Table 5. Where each friction loads today | 3 |
| 7.1 Model with Frictions | The source. Availability at the bank boundary (the index above). Shares by pool. Attention. The law of motion. The paper's model as the special case | 11 |
| 7.2 What Identifies Each Friction | Attention: the Selic rate in the time series, the stakes across markets, instrumented. Awareness: advertising, its exclusion from utility, the test on own money, and the opening margin. Switching cost: the launch, read as a cost because openings at B banks did not jump, and the dose. Tax clock: the time-deposit share of the source. What stays unidentified | 10 |
| 7.3 Evidence so Far | The advertising elasticity of openings and the test of the difference. The Selic sign. The first stage of the stakes. The Pix result of Section 6. One sentence on the net-flow equation, which does not converge. Then, as IMPR reports: $\tau_1$, the own-money test, and the first fit of the index | 8 |
| **Total** | | **about 32, against 81 today** |

**Equations.** Seven in place of eleven. Labels follow the printed numbers, so they change.

| New | Content | Old |
|---|---|---|
| (12) | Availability index | (14), with Pix and the clock added. The old (12), utility with costs, is dropped |
| (13) | Share by pool | (13) |
| (14) | Stakes | (15) |
| (15) | Attention term | (16) |
| (16) | Law of motion, without the maturity split | (17) |
| (17) | Opening rate | (22) |
| (18) | Pix exposure instrument, if the dose stays | (21) |
| none | Net-flow identity, pool weights, estimating equation | (18), (19), (20) |

**Tables.** Table 5 stays, with the switching-cost cell pointing to 7.2. Table 6 leaves, as you decided.

## 4. The appendices

| Appendix today | Under the new route | Why |
|---|---|---|
| D, Tax Clock, AC1 to AC8 (the pricing and Table D.1) | Keep | It is what the law says the clock costs |
| D, AC9 to AC13 (in the model, in the net-flow equation) | Replace with one paragraph: the clock enters as one shifter | The pools and the pair gap are gone |
| E.1 Derivation, E.2 Scales and Exit, E.3 Identification with the two propositions, E.4 Estimation and Inference | Move to the switching-cost note | They serve an estimator the paper no longer uses. The note already has this material |
| E.5 Pix Exposures (not typed yet) | Keep, as an appendix on the Pix dose | Still the instrument for the dose |
| E.6 Within-Bank Margin (not typed yet) | Keep, shortened | Fees are the argument that the cost is not a fee |
| F, Simulation Evidence | Drop from the paper | It tests the net-flow estimator. It is in the note |
| New, short | The awareness index: the exclusion, the test on own money, and how the index is fitted | Needed once there is a fit |

**The cost to you.** You typed Appendix E through E.4 yesterday and today. Under this route most of it leaves the paper. The alternative is to keep it as an appendix titled as a check from the model, as your awareness note does. I recommend moving it, because a twelve-page appendix for an equation that produces no estimate is what made Section 7 heavy in the first place. It is your call.

## 5. The pending edits file

`paper_edits_2026-11-12.tex` is on hold. Do not apply these blocks:

| Block | Status | Reason |
|---|---|---|
| B1 (Section 7) | Superseded | Written for the old model |
| B5 (short Appendix F) | Superseded | Appendix F leaves the paper |
| A4 (rest of Appendix E) | Hold | Only E.5 and E.6 survive, in a new place |
| R1, R2, C1 | Hold | They edit Appendix E text that moves |
| F8 (slips in Appendix E) | Apply only if you keep Appendix E | Two are sign errors in the math |

Already applied and still right: the Oxford commas, the shorter table notes, B2, B3, B4.

## 6. Sequence

1. **You:** accept or change this outline, and decide Appendix E (move or keep as a check).
2. **Main thread:** write the blocks for 7.1 and 7.2 and for Appendix D. They need no new number.
3. **IMPR:** the work in `MAIN_REPLY_2026-11-12.md`. The first results that feed the text are the stakes coefficient and the test on own money.
4. **Main thread:** write 7.3 and the short awareness appendix as results arrive. Align both notes.

## 7. Cautions that the text will have to carry

1. **No shifter, no separation.** Every claim is about a shifter. The Pix shift could be awareness. Two things answer that: advertising is in the same index, and openings at B banks did not jump at the launch.
2. **One launch date.** The pandemic, the emergency aid, and the tightening cycle share it.
3. **The time-deposit share is not random.** Banks heavy in time deposits have different clients, so $\kappa^{c}$ may pick up sorting. It is an attempt, as you said.
4. **New money dominates the awake pool by the current measure** (the outside pool is 97 percent of the market size). If that holds, own and rival money differ too little for the test on own money to have power. IMPR checks this first, and under more than one cap.
5. **The baseline price coefficient is not settled.** It moves with the cap, and its first stage is under review, with your point that checking and savings spreads are exogenous now part of the test.
6. **None of this has been estimated or simulated in this form.**

## 8. Refinement, 11-12 15:52: the Pix term is not a switching cost by label

Your point: a Pix term inside the availability index reads as the effect of Pix on awareness as much as on a cost, and Section 2 of the paper itself says Pix may have raised awareness. The edits now treat it as a test.

- **Two coefficients.** $\gamma_{P}$ shifts the availability of rivals for money that has a bank to leave. $\gamma_{P\varnothing}$ shifts the index of new money and of account openings, which is awareness alone, since neither involves leaving a bank.
- **The switching cost is the difference,** $\gamma_{P} - \gamma_{P\varnothing}$. If Pix only raised awareness, the two are equal and openings rise at the launch. If it only lowered the cost, $\gamma_{P\varnothing} = 0$ and openings do not move.
- **Cautions in the text.** Accounts opened to receive transfers reflect convenience as well as awareness, so openings may overstate the awareness part. New money is the worst-measured pool. The launch shares its date with the pandemic, the emergency aid, and the tightening cycle.
- **Where this changes the outline above:** every statement that the Pix shift is the switching cost (Sections 1, 2, and 7 of this file) should be read with this refinement. The symbol $\kappa^{P}$ is gone. The tax-clock term $\kappa^{c}$ is unchanged.
