# Handoff: follow-ups from the IO Tea and the advisor meeting (2026-10-20)

**To:** coding agent. **From:** main thread, for Pedro.

**Rule:** `V_Main.tex` is edited by the author only, and so is the deck. This brief asks for numbers, tables, and figures, with no paper text.

## Basis

```
files[7]{file,role,saved}:
  market_panel.parquet,final panel,10-15 22:35
  DEMAND_PREP/est3|est4/national_phi_t.csv,phi_t per routine (instrument RivalSpread_x_Tech),10-18 16:30/16:35
  ESTIMATION_OUTPUT/DIAG_STATE_SELECTION/level_by_block.csv,state-vector diagnostic (diag_state_vector_selection.py),10-12
  DIAG_PHI_SEPARATION/d6_*,entry paths on the final panel,10-19 18:06
  Rout/xi_diagnostics.json,demand residual diagnostics,10-19 12:53
  tab_dep_anova.tex,deposit variance split (make_dep_anova.py),10-19 14:37
  paper_numbers_macros.tex,macros,10-19 18:07
```

I have not seen where the fitted φ̂_mt (by MCA and quarter, routines E3 and E4) is saved. Find it first. If it is not saved, say so and stop. Do not re-estimate.

## Tasks

```
tasks[6]{id,agent,output,priority}:
  C1,code,"keys sleep.phi_anova.* + fig_slide_phi_anova_ref.pdf",high
  C2,code,"keys entry.carry.* + fig_slide_entry_carry_ref.pdf",high
  C3,code,"keys demand.flows.neg.* + tab_negative_active.tex",high
  C4,code,"keys pix.exposure.* (first stage only)",medium
  C5,code,"keys desc.dshare_decomp.*",low
  C6,code,"keys desc.fees.* + fig_slide_fee_ratio_ref.pdf",medium
```

### C1. Variance of the sleepy share across markets and quarters

**Data:** φ̂_mt for E3 and E4, one value per MCA and quarter.

- **(a) Order-free split.** Split the variance of φ̂_mt into MCA effects, quarter effects, and the rest, with the order-free (Shapley) method of `tab_motiv_anova_levels.tex`. Do it twice: with population weights, and with lagged-deposit weights.
- **(b) Split of the index.** Split the variance of the index Υ̂′S_mt by state: Cadúnico, share aged 65 and over, lagged Selic, mobile lines, Pix. Use the covariance split, Υ̂_k cov(S_k, index) / var(index), since the index is linear.
- **(c) Ceiling.** Report the share of cells at φ̂ = 1, and the split in (a) without those cells.

**Keys:**

- `sleep.phi_anova.{E3,E4}.{pop,dep}.{market,quarter,rest}`
- `sleep.phi_index.{E3,E4}.{cadunico,age65,selic,mobile,pix}`
- `sleep.phi_ceiling.{E3,E4}.share`

**Figure:** bars in the style of `fig_slide_xi_anova_slide.pdf`, drawn in the reference style of request 7, 8.0 × 4.6 cm.

**Flag to keep visible:** φ̂_mt is a deterministic function of S_mt. So (a) measures how much the estimated function spreads the states across markets and quarters. It is not separate evidence on where sleepiness varies.

### C2. Does entry wake incumbents' depositors?

**Why a new design.** Pedro proposed using the φ decomposition around entry events. That cannot work as it stands. φ̂_mt = G(Υ̂′S_mt) moves only when the states move, and entry is not a state. An event study of φ̂_mt around entry would only show whether entrants pick markets whose states are changing.

**Design.** Test entry on the incumbents' carry instead:

- Event-time interactions of the incumbents' lagged balance in the B-firm sleepiness regression.
- `Dep_jkmt = Σ_h β_h 1{t − t_entry,m = h} · g_jkm,t−1 · Dep_jkm,t−1 + controls + bank × type × market FE + quarter FE`, for incumbents j only.
- h from −4 to +8, with the omitted category h ≤ −5.
- Use the 198 B entries of the window rule (DIAG_PHI_SEPARATION/d6).

**Report:**

- β_h relative to the pre-period
- the number of markets with entry
- the wild cluster bootstrap by conglomerate, as elsewhere.

**Keys:** `entry.carry.{h}.{estimate,ci.lo,ci.hi}` and `entry.carry.n_markets`.

**Figure:** an event-study plot in the reference style, 10.8 × 4.6 cm.

**Flags:**

- Entry is chosen. Entrants may enter where incumbents' depositors were about to move.
- Santander makes most of the entries. Report the result without Santander as well.

### C3. Negative active demand and the truncation in (14-B)

**Count.** Count the cells where Dep_jkmt < φ̂_mt · g_jkm,t−1 · Dep_jkm,t−1, the cells that (14-B) truncates at zero. Report:

- the share of cells and the share of balances
- the split by deposit type, by firm type (B, D), by market size quintile, and by year
- the share of truncated cells among rows with φ̂·g ≥ 1, the stationarity-bound rows of the state-vector diagnostic (32 to 43 percent of rows under E3).

**Market bound.** If the model holds, φ_mt ≤ min_j Dep_jkmt / (g_jkm,t−1 · Dep_jkm,t−1) within a market and quarter. Report:

- the share of market-quarters where φ̂_mt exceeds this bound
- the same share with the 10th percentile across banks in place of the minimum.

**Keys:** `demand.flows.neg.{E3,E4}.{share_cells,share_bal}`, `demand.flows.neg.{E3,E4}.bound_violation.{min,p10}`, plus the splits.

**Table:** `tab_negative_active.tex`, label `tab:negative_active`.

**Flag:** this is a diagnostic only. Do not change (14-B).

### C4. Pix exposure through mobile access (first stage only)

**Exposure.** Exposure_m = the 2019 mobile measure × 1{t ≥ 2020Q4}:

- once with mobile lines per 100 inhabitants, the current state
- once with the 4G/5G share, if the state-vector diagnostic's "+4G/5G" variable is in the panel.

**Correlation.** Report its correlation with the existing exposure, the 2019 local share of institutions mandated into Pix (`frame:diagnosis`).

**First stage.** If S3 of the scraping handoff lands, run it: Pix transactions per capita by MCA on each exposure, with quarter effects and Cadúnico × post as a control.

**Keys:** `pix.exposure.{mobile,g45,mandated}.{sd,corr_mandated,fs.slope,fs.f}`.

**Flag:** mobile lines are already in S_mt. Exposure × post is then a heterogeneous Pix effect, and it is identified only if 2019 mobile access is unrelated to other post-2020 shocks. The emergency aid paid through Caixa Tem went to poorer markets, which is why Cadúnico × post is a control.

### C5. Oaxaca–Blinder-style split of the D-share gain

Split the change in the D firms' national share of deposits, 2016 to 2024, into two parts:

- a part from the deposit-type mix, a between-type term at fixed within-type shares
- a part from within-type gains.

Use the standard two-fold split, with the 2016 type weights and with the 2024 type weights.

**Keys:** `desc.dshare_decomp.{between,within}.{w2016,w2024}`.

### C6. Cost of holding an account

**Fee ratio.** The measure is in `service_fees_notes.md`, Sections 2 and 5: fee revenue over total deposits, from COSIF (the accounting plan), corrected to quarterly increments, through 2024. Build keys for:

- the incumbent median and the D-firm median by year, 2016 to 2024
- the change from 2019 to 2024.

Drop 2025, which is the COSIF reclassification artifact.

**Keys:** `desc.fees.{incumbent,digital}.{year}` and `desc.fees.incumbent.change_2019_2024`.

**Figure:** reference style, 10.8 × 4.6 cm.

**Flag:** the effective price of a deposit is the spread plus fees. If incumbents' fees fell after Pix, the demand model's price omits that change. State this in the figure's note.

## Acceptance

- Every key resolves with no red `??`.
- Shares sum to one where they should.
- Values match the source files. Report each file and its time.
- Notes pass the style rules: American English, no semicolons, no dashes separating ideas, acronyms defined.
- Report anything not feasible and why.

## Open questions for Pedro

1. C2: are the incumbents in an entered market the right comparison, or the same banks in markets without entry?
2. C3: should φ̂ be capped at the market bound in a robustness run, or only reported?
3. C4: mobile lines or 4G/5G as the preferred exposure?
